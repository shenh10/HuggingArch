"""A collective's schedule coefficients: the bytes a rank moves and the
link traversals it waits on.

Layout materializes `all_reduce` / `reduce_scatter` / `all_gather` /
`all_to_all` and needs the bytes a rank moves: the flat-ring volume,
`2(N-1)/N · T` for all-reduce, `(N-1)/N · T` for the others
(`ring_factor`). The other coefficient is the number of link traversals
one collective waits on in sequence (`latency_hops`): the latency-optimal
small-message schedule's log-depth count. Both are structural and
seconds-free; N=1 is zero for both. Seconds live in inference: the bytes
over the link's bandwidth plus the hops times one link latency, the same
on every fabric (`inference.comm_cost`).

Those two are what NCCL's schedule costs, and what an all-reduce costs
wherever no faster algorithm runs. Inside one NVLink domain a serving
framework all-reduces a small or medium payload with its own kernels
instead (`ALGORITHMS`: one-shot, two-shot, NVLS); which of them a payload
takes is the framework's dispatch (`framework_bindings`, `collectives:`),
and what each costs is stated here, beside the ring's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from backend.arch.ops import OP_REGISTRY, collective_schedule

#: The collectives, read off the registry's `is_comm_op` flag rather than
#: spelled out again here: the registration is where an op says what it is.
COMM_OPS = frozenset(
    name for name, spec in OP_REGISTRY.items() if spec.is_comm_op)


def collective_per_rank_bytes(op: str, world: int, tensor_bytes: float) -> float:
    """Per-rank interconnect traffic for one collective, flat ring.

    `world` is the number of ranks in the collective group (N);
    `tensor_bytes` is the full communicated tensor's size in bytes.
    Returns 0 for N <= 1 (nothing leaves the rank) or a non-comm op.
    Hierarchical time accounting lives in `inference.comm_cost.collective_phases`.
    """
    n = int(world or 1)
    if n <= 1 or op not in COMM_OPS:
        return 0.0
    # An empty payload moves nothing: the ring's share of zero bytes.
    return ring_factor(op, n) * tensor_bytes


def ring_factor(op: str, world: int) -> float:
    """Per-rank bytes as a fraction of the tensor for a flat ring of
    `world` ranks: `(N-1)/N`, doubled for `all_reduce` (reduce-scatter then
    all-gather). The one place the coefficient is written; the formula
    renderers read it through `ring_factor_symbolic`."""
    n = int(world or 1)
    if collective_schedule(op) == "point_to_point":
        # One hop: the sender's whole tensor, whatever the group's size.
        return 1.0
    factor = (n - 1) / n
    if op == "all_reduce":
        factor *= 2.0
    return factor


def latency_hops(op: str, world: int) -> int:
    """How many link traversals one collective of `world` ranks waits on in
    sequence, each a link latency apart, by the latency-optimal small-message
    schedule (Thakur, Rabenseifner & Gropp, "Optimization of Collective
    Communication Operations in MPICH", 2005): a point-to-point send is one
    hop; reduce-scatter (recursive halving), all-gather (recursive doubling)
    and all-to-all (Bruck) take `ceil(log2 N)` rounds; all-reduce is a
    reduce-scatter then an all-gather (Rabenseifner), `2 ceil(log2 N)`. The
    one place the count is written, beside the bandwidth coefficient
    (`ring_factor`); a hierarchical collective adds its phases' counts
    (`inference.comm_cost.collective_latency`)."""
    n = int(world or 1)
    if n <= 1:
        return 0
    if collective_schedule(op) == "point_to_point":
        return 1
    rounds = math.ceil(math.log2(n))
    return 2 * rounds if op == "all_reduce" else rounds


@dataclass(frozen=True)
class CollectiveAlgorithm:
    """One way to run an all-reduce of N ranks: the link traversals one
    instance waits on in sequence (`hops`), the bytes a rank moves per
    payload byte (`factor`), and the topology it needs: `"nvlink"`, every
    rank reaching every other over NVLink (a peer-to-peer kernel);
    `"nvls"`, a switch that reduces in the network; `""`, any fabric."""
    name: str
    needs: str
    hops: Callable[[str, int], int]
    factor: Callable[[str, int], float]

    def factor_symbolic(self, op: str, world: Optional[int] = None, *,
                        symbol: str = "N") -> str:
        """`factor` as text: over `symbol` when `world` is None, substituted
        otherwise (`7` for a one-shot of 8, `2*7/8`, `7/8`)."""
        if self.name in (NCCL, "direct_exchange"):
            return ring_factor_symbolic(op, world, symbol=symbol)
        less = f"({symbol}-1)" if world is None else f"{int(world) - 1}"
        over = symbol if world is None else f"{int(world)}"
        return {"one_shot": less, "two_shot": f"2*{less}/{over}",
                "nvls": f"{less}/{over}"}[self.name]


#: The algorithm every collective takes where no dispatch says otherwise.
NCCL = "nccl"

#: The all-reduce algorithms, by the names frameworks and NCCL give them;
#: the vocabulary a binding's `collectives:` is checked against at load.
#:
#: * `one_shot` — every rank writes its whole payload to every peer (or
#:   reads every peer's), then reduces locally: one traversal, and a rank
#:   receives `(N-1) T` (FlashInfer's MNNVL Lamport one-shot, vLLM's
#:   `cross_device_reduce_1stage`, sglang's `1shot_push` / `1shot_pull`).
#: * `two_shot` — a reduce-scatter then an all-gather, each one direct
#:   exchange: two traversals, `2(N-1)/N T` like the ring.
#: * `nvls` — NVLink SHARP: each rank reduces its `T/N` slice in the switch
#:   (`multimem.ld_reduce`) and multicasts it back (`multimem.st`): two
#:   traversals, and a rank moves `(N-1)/N T` each way, half the ring's.
#: * `nccl` — NCCL's own schedule, priced as the ring's bytes
#:   (`ring_factor`) and the latency-optimal log-depth hops (`latency_hops`):
#:   what runs wherever none of the above does, and on every other
#:   collective.
ALGORITHMS: Dict[str, CollectiveAlgorithm] = {
    a.name: a for a in (
        # Sparse DCP's peer multicast/push kernels exchange directly within
        # one NVLink domain. Recipient-volume accounting, one transfer stage;
        # synchronization polling and local merge work are separate costs.
        CollectiveAlgorithm("direct_exchange", "nvlink",
                            lambda op, n: 1 if n > 1 else 0,
                            lambda op, n: ring_factor(op, n) if n > 1 else 0.0),
        CollectiveAlgorithm("one_shot", "nvlink",
                            lambda op, n: 1 if n > 1 else 0,
                            lambda op, n: float(n - 1) if n > 1 else 0.0),
        CollectiveAlgorithm("two_shot", "nvlink",
                            lambda op, n: 2 if n > 1 else 0,
                            lambda op, n: 2.0 * (n - 1) / n if n > 1 else 0.0),
        CollectiveAlgorithm("nvls", "nvls",
                            lambda op, n: 2 if n > 1 else 0,
                            lambda op, n: (n - 1) / n if n > 1 else 0.0),
        CollectiveAlgorithm(NCCL, "",
                            latency_hops,
                            lambda op, n: ring_factor(op, n) if n > 1 else 0.0),
    )}


def ring_factor_symbolic(op: str, world: Optional[int] = None, *,
                         symbol: str = "N") -> str:
    """The same coefficient as text: symbolic (`2*(N-1)/N`) when `world`
    is None, substituted (`2*7/8`) otherwise. `symbol` names the world in
    the symbolic form (`N_inter` for the inter-node phase of a hierarchical
    all-reduce)."""
    if collective_schedule(op) == "point_to_point":
        return "1"
    doubled = op == "all_reduce"
    if world is None:
        return f"2*({symbol}-1)/{symbol}" if doubled else f"({symbol}-1)/{symbol}"
    n = int(world)
    return f"2*{n - 1}/{n}" if doubled else f"{n - 1}/{n}"
