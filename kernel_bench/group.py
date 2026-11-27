"""One measurement group: n_comm_ranks processes, one die each.

The harness ran a single process, so every collective was the identity of
a local copy. This module is the only place that turns `n_comm_ranks`
into a Die list, starts that many children, watches them, and kills the
whole group if any one fails or stalls — HCCL has no timeout of its own.

Die mapping stays here (`die_id = origin + rank * stride`) and never
enters the 12 identity columns. Rank 0 of the parent is the unique CSV
writer: children return their rows on a queue; this process takes
`max_r(dt_r)` and emits once.
"""
from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .gpulock import GpuLock


class GroupFailed(RuntimeError):
    """The group did not finish a measurement. Siblings were killed."""


def dies_for(n_ranks: int, stride: int = 1, origin: int = 0) -> List[int]:
    """Physical die indices for a group of `n_ranks`.

    `die_id = origin + rank * stride`. The stride is the group's spacing
    on the mesh (one die per package is 2 on a 910C), not an identity
    column — two groups of the same size at different strides are
    different measurements, and that fact already lives on the cell as
    `n_comm_stride`.
    """
    if int(n_ranks) < 1:
        raise ValueError(f"n_ranks must be >= 1, got {n_ranks!r}")
    if int(stride) < 1:
        raise ValueError(f"stride must be >= 1, got {stride!r}")
    return [int(origin) + r * int(stride) for r in range(int(n_ranks))]


def dies_for_jobs(jobs: Iterable[Dict[str, Any]], *,
                  stride: int = 1, origin: int = 0) -> Optional[List[int]]:
    """The union of dies every collective job in `jobs` will touch.

    None when nothing in the list needs a group: the caller then takes
    the default single-device lock. A mixed list also includes `origin`,
    which is the die local kernels run on.
    """
    needed = set()
    for job in jobs:
        dims = job.get("dims") or {}
        if isinstance(dims, str):
            dims = json.loads(dims)
        n = int(dims.get("n_comm_ranks") or 0)
        if n >= 2:
            needed.update(dies_for(n, stride, origin))
    if not needed:
        return None
    needed.add(int(origin))
    return sorted(needed)


def visible_devices_env(device: Optional[str], dies: Sequence[int]) -> Dict[str, str]:
    """Remap the group's dies to 0..N-1 in the child.

    After this, `set_device(local_rank)` lands on `dies[local_rank]`.
    """
    joined = ",".join(str(d) for d in dies)
    if device == "npu":
        return {"ASCEND_RT_VISIBLE_DEVICES": joined}
    if device == "cuda":
        return {"CUDA_VISIBLE_DEVICES": joined}
    return {}


def dist_backend(device: Optional[str]) -> str:
    if device == "npu":
        return "hccl"
    if device == "cuda":
        return "nccl"
    return "gloo"


def spread_note(values: Sequence[float]) -> str:
    """Provenance for the cross-rank clock, no new CSV column.

    `max/min` is 1 in the steady state the barrier is for; a slow rank
    makes it large, which is how a later reader tells 'the fabric was
    slow' from 'one die stalled'.
    """
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    ratio = (hi / lo) if lo > 0 else float("inf")
    return f"rank_spread max/min={ratio:.4g} min_us={lo:.4f} max_us={hi:.4f}"


def aggregate_latency(rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    """Rank 0's row, with every latency column the max across ranks."""
    if not rows:
        raise GroupFailed("no rank returned a row")
    by_rank = sorted(rows, key=lambda r: int(r.get("_rank", 0) or 0))
    out = dict(by_rank[0])
    p50s, p95s, means = [], [], []
    for row in by_rank:
        if row.get("p50_us"):
            p50s.append(float(row["p50_us"]))
        if row.get("p95_us"):
            p95s.append(float(row["p95_us"]))
        if row.get("mean_us"):
            means.append(float(row["mean_us"]))
        outcome = row.get("outcome") or "measured"
        if outcome != (out.get("outcome") or "measured"):
            raise GroupFailed(
                f"ranks disagreed on outcome: {out.get('outcome')} vs {outcome}")
    if p50s:
        out["p50_us"] = out["latency_us"] = f"{max(p50s):.4f}"
    if p95s:
        out["p95_us"] = f"{max(p95s):.4f}"
    if means:
        out["mean_us"] = f"{max(means):.4f}"
    note = spread_note(p50s)
    if note:
        out["_rank_spread"] = note
        launch = out.get("launch_signature") or ""
        if launch:
            out["launch_signature"] = f"{launch}; {note}"
        else:
            out["launch_signature"] = note
    out.pop("_rank", None)
    return out


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _kill_all(procs) -> None:
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
    deadline = time.monotonic() + 2.0
    for proc in procs:
        remaining = deadline - time.monotonic()
        proc.join(timeout=max(0.05, remaining))
    for proc in procs:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)


def _reap(procs, queue, world: int, timeout_s: float) -> List[dict]:
    """Collect one report per rank, or kill the group."""
    deadline = time.monotonic() + timeout_s
    from queue import Empty
    reports: List[dict] = []
    pending = {id(p): p for p in procs}
    while len(reports) < world and time.monotonic() < deadline:
        try:
            reports.append(queue.get(timeout=0.2))
        except Empty:
            pass
        for key, proc in list(pending.items()):
            if proc.is_alive():
                continue
            pending.pop(key, None)
            if proc.exitcode not in (0, None) and len(reports) < world:
                _kill_all(procs)
                raise GroupFailed(
                    f"a rank exited {proc.exitcode} before the group finished")
    if len(reports) < world:
        _kill_all(procs)
        raise GroupFailed(
            f"group timed out after {timeout_s:.0f}s "
            f"({len(reports)}/{world} ranks reported)")
    errors = [r.get("error") for r in reports if r.get("error")]
    if errors:
        _kill_all(procs)
        raise GroupFailed("; ".join(str(e) for e in errors))
    return reports


def _hccl_net_env(master_addr: str, *, nodes: int) -> Dict[str, str]:
    """`HCCL_IF_IP` / `HCCL_SOCKET_IFNAME` for a group that spans nodes.

    Empty on one node: HCCL picks a loopback-reachable interface on its
    own and naming one here would override a working default with a guess.
    Across nodes the local address of the interface that routes to the
    rendezvous is what the peers can reach, so it is read off the routing
    table rather than assumed to be the first interface up.

    An environment that already states either keeps it: an operator who
    pinned the control plane knows something about the fabric this does
    not.
    """
    if nodes <= 1:
        return {}
    out: Dict[str, str] = {}
    if not os.environ.get("HCCL_IF_IP"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                # Connectionless: this selects a route and a source address
                # without sending anything.
                probe.connect((master_addr, 1))
                out["HCCL_IF_IP"] = probe.getsockname()[0]
        except OSError:
            pass
    return out


def measure_job(
    job: dict,
    *,
    env,
    stride: int = 1,
    origin: int = 0,
    timeout_s: float = 120.0,
    take_lock: bool = True,
    worker: Optional[Callable] = None,
    extra_child_env: Optional[Dict[str, str]] = None,
    date: str = "",
    contributor: str = "",
    run_id: str = "",
    box: str = "",
    master_addr: str = "127.0.0.1",
    master_port: Optional[int] = None,
    nnodes: int = 1,
    node_rank: int = 0,
) -> List[Dict[str, str]]:
    """Run a group over `n_comm_ranks` processes and return aggregated rows.

    `worker` is the child entry (tests inject a stub). The live entry is
    `rankworker.entry`.

    One node by default, and then every rank of the group is a child of
    this process — the arrangement every row measured so far was taken in.
    At `nnodes > 1` this process owns only ITS SHARE: ranks
    `node_rank * per_node` through the next `per_node`, on its own dies,
    talking to the others over `master_addr:master_port`. Each node runs
    this same call and writes its own rows; nothing here reaches across a
    machine, because the two halves of a measurement meet in the corpus
    and not in a socket this harness owns.
    """
    dims = job.get("dims") or {}
    if isinstance(dims, str):
        dims = json.loads(dims)
    n_ranks = int(dims.get("n_comm_ranks") or 0)
    if n_ranks < 2:
        raise GroupFailed(f"a group needs n_comm_ranks>=2, got {n_ranks}")
    nodes = max(int(nnodes or 1), 1)
    if nodes > 1:
        if n_ranks % nodes:
            raise GroupFailed(
                f"a group of {n_ranks} does not divide over {nodes} nodes; "
                f"this harness places an equal share on each")
        if not 0 <= int(node_rank) < nodes:
            raise GroupFailed(
                f"node_rank {node_rank} is not a node of {nodes}")
        if master_addr in ("127.0.0.1", "localhost", ""):
            raise GroupFailed(
                "a group spanning nodes needs a --master-addr the other "
                "nodes can reach; loopback is this machine only")
        if not master_port:
            raise GroupFailed(
                "a group spanning nodes needs a fixed --master-port: a "
                "port picked here is not one another node can guess")
    local_ranks = n_ranks // nodes
    dies = dies_for(local_ranks, stride, origin)
    lock = GpuLock(devices=dies) if take_lock else None
    procs: List = []
    if lock is not None:
        lock.acquire()
    try:
        return _spawn(job, env=env, n_ranks=n_ranks, dies=dies,
                      timeout_s=timeout_s, worker=worker,
                      extra_child_env=extra_child_env, procs=procs,
                      date=date, contributor=contributor,
                      run_id=run_id, box=box,
                      local_ranks=local_ranks,
                      rank_offset=int(node_rank) * local_ranks,
                      master_addr=master_addr, master_port=master_port)
    finally:
        _kill_all(procs)
        if lock is not None:
            lock.release()


def _spawn(job: dict, *, env, n_ranks: int, dies: List[int],
           timeout_s: float, worker: Optional[Callable],
           extra_child_env: Optional[Dict[str, str]],
           procs: List, date: str = "", contributor: str = "",
           run_id: str = "", box: str = "",
           local_ranks: Optional[int] = None, rank_offset: int = 0,
           master_addr: str = "127.0.0.1",
           master_port: Optional[int] = None) -> List[Dict[str, str]]:
    import multiprocessing as mp

    from .rankworker import entry as live_entry

    target = worker or live_entry
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    # This node's share. One node: every rank, and `_free_port()` on
    # loopback is right because the only peers are siblings of this
    # process. Several: a port the other nodes were told, on an address
    # they can route to.
    here = int(local_ranks or n_ranks)
    port = int(master_port) if master_port else _free_port()
    child_env = {
        "KERNEL_BENCH_IN_GROUP": "1",
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(port),
        "WORLD_SIZE": str(n_ranks),
        **visible_devices_env(getattr(env, "device", None), dies),
        # Which interface HCCL's own control plane talks over. On one node
        # it finds a usable one by itself; across nodes it has to be told,
        # because the address the group rendezvous on is this host's and
        # the default pick may be a link that does not reach the peer.
        # The deployed stack states the same two (vLLM-Ascend's run
        # templates set HCCL_IF_IP and HCCL_SOCKET_IFNAME together).
        **_hccl_net_env(master_addr, nodes=n_ranks // max(here, 1)),
        **(extra_child_env or {}),
    }
    backend = dist_backend(getattr(env, "device", None))
    for local in range(here):
        # The two numbers a rank needs and which are the same only on one
        # node: RANK places it in the GROUP (and so decides what it holds,
        # since `Invocation.seed_for_rank` draws the payload from it), and
        # LOCAL_RANK places it on THIS BOX's visible dies. Setting the
        # device from the global rank puts node 1 of a 32-rank group at
        # device 16 of 16.
        rank = rank_offset + local
        cfg = {
            "rank": rank,
            "world": n_ranks,
            "job": job,
            "backend": backend,
            "child_env": {**child_env, "LOCAL_RANK": str(local),
                          "RANK": str(rank)},
            "env_device": getattr(env, "device", None),
            "env_library": getattr(env, "library", None),
            "gpu_sku": getattr(env, "gpu_sku", None),
            # The card's bandwidth travels with its name. The parent
            # resolved both off the BOX when the driver's device name does
            # not name a sheet SKU (`cli._cmd_run`: `Ascend910_9362` has no
            # digitless family token, so the matcher cannot resolve it), and
            # a child that re-detects gets neither. Every row is validated
            # against `peak_bw_at_test`, so without this the group measures
            # correctly and then drops every row at schema validation.
            "peak_bw_gbps": getattr(env, "peak_bw_gbps", None),
            # The child's rendezvous deadline, so HCCL's own watchdog and
            # the reaper below are the same number rather than two.
            "timeout_s": timeout_s,
            "date": date,
            "contributor": contributor,
            "run_id": run_id,
            "box": box,
        }
        proc = ctx.Process(target=target, args=(cfg, queue),
                           name=f"kb-rank-{rank}")
        procs.append(proc)
    for proc in procs:
        proc.start()
    # This node reaps ITS OWN children. Waiting for the whole group's
    # worth of reports would hang on a node that holds half of it, and
    # the other half's rows are not ours to write: each node emits its
    # share and the corpus is where they meet.
    reports = _reap(procs, queue, here, timeout_s)
    # One aggregated row per kernel_impl the children measured.
    by_impl: Dict[str, List[dict]] = {}
    for report in reports:
        for row in report.get("rows") or []:
            row = dict(row)
            row["_rank"] = report["rank"]
            by_impl.setdefault(row.get("kernel_impl") or "", []).append(row)
        if not report.get("rows") and report.get("skipped"):
            raise GroupFailed(
                f"rank {report['rank']} skipped: {report['skipped']}")
    if not by_impl:
        raise GroupFailed("every rank returned no rows")
    return [aggregate_latency(group) for group in by_impl.values()]
