"""Child process of a measurement group.

Sets the visible die, initialises the process group, runs the cell, and
puts the rows on the parent's queue. Never opens the results CSV — the
parent is the unique writer (`emit.IncrementalCsvWriter`).
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Dict


def entry(cfg: Dict[str, Any], queue) -> None:
    """`multiprocessing` spawn target. `cfg` is pickle-friendly."""
    rank = int(cfg["rank"])
    world = int(cfg["world"])
    try:
        os.environ.update(cfg["child_env"])
        report = _measure(cfg, rank, world)
        queue.put(report)
    except Exception as exc:
        queue.put({
            "rank": rank,
            "rows": [],
            "error": f"rank {rank}: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


def _align(dist, backend_dev):
    def align() -> None:
        if dist is not None:
            dist.barrier()
        if backend_dev is not None:
            backend_dev.synchronize()
    return align


def _agree(dist, backend_dev):
    """Rank 0's sample counts, for every rank of the group.

    `time_callable` sizes warmup and samples against a budget using the
    `per_call` IT measured, and that is a threshold test: ranks either
    side of it choose different counts. Every sample is a barrier
    (`_align`), so different counts deadlock the group until the reaper
    kills it.

    Broadcasting rank 0's keeps the sizing adaptive -- a slow cell still
    has its budget cut -- while making it one decision instead of N.
    """
    if dist is None:
        return None

    def agree(warmup: int, iters: int):
        import torch
        # The collective runs on the group's own backend, so the tensor
        # has to live where that backend sends from: HCCL and NCCL move
        # device memory, and a CPU tensor is not something they can read.
        where = (backend_dev.tensor_device
                 if backend_dev is not None else "cpu")
        pair = torch.tensor([int(warmup), int(iters)], dtype=torch.int64,
                            device=where)
        dist.broadcast(pair, src=0)
        if backend_dev is not None:
            backend_dev.synchronize()
        return int(pair[0].item()), int(pair[1].item())

    return agree


def _measure(cfg: Dict[str, Any], rank: int, world: int) -> Dict[str, Any]:
    from dataclasses import replace

    from .device import detect_backend
    from .driver import run_jobs
    from .env import detect_env

    backend_dev = detect_backend()
    if backend_dev is not None:
        # The die this process runs on is a LOCAL index into the dies the
        # parent made visible, never the global rank: on the second node
        # of a 32-rank group the global ranks are 16..31 and the visible
        # dies are still 0..15. One node leaves the two equal, which is
        # why this read as the rank for as long as there was one node.
        backend_dev.set_device(int(os.environ.get("LOCAL_RANK", rank)))

    import torch.distributed as dist_mod
    # A rendezvous that crosses machines can take much longer than one
    # between siblings, and the default would give up under the group
    # watchdog rather than beside it. One timeout, stated once.
    timeout_s = float(cfg.get("timeout_s") or 0.0)
    kwargs = {}
    if timeout_s > 0:
        from datetime import timedelta
        kwargs["timeout"] = timedelta(seconds=timeout_s)
    dist_mod.init_process_group(
        backend=cfg["backend"], rank=rank, world_size=world, **kwargs)
    dist = dist_mod

    env = detect_env()
    if cfg.get("env_library") is not None:
        env = replace(env, library=cfg["env_library"])
    if cfg.get("gpu_sku") and not env.gpu_sku:
        env = replace(env, gpu_sku=cfg["gpu_sku"])
    # The card the PARENT resolved, not what this child can detect. On a
    # box whose driver device name names no sheet SKU the parent reads both
    # the SKU and its bandwidth off the box profile; a child that only
    # inherits the name writes an empty `peak_bw_at_test` and every row it
    # measured is dropped at schema validation.
    if cfg.get("peak_bw_gbps") and not env.peak_bw_gbps:
        env = replace(env, peak_bw_gbps=cfg["peak_bw_gbps"])

    job = dict(cfg["job"])
    job["rank"] = rank
    rows: list = []
    result = run_jobs(
        [job], env=env,
        date=cfg.get("date") or "1970-01-01",
        contributor=cfg.get("contributor") or "@kernel-bench",
        run_id=cfg.get("run_id") or "",
        box=cfg.get("box") or "",
        in_group=True, align=_align(dist, backend_dev),
        agree=_agree(dist, backend_dev),
        on_row=rows.append)
    if dist is not None:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    return {
        "rank": rank,
        "rows": rows,
        "skipped": list(result.skipped),
        "failures": list(result.failures),
        "error": (result.failures[0] if result.failures else None),
    }
