"""A collective's schedule coefficients: the bytes a rank moves and the
link traversals it waits on.

Layout materializes `all_reduce` / `reduce_scatter` / `all_gather` /
`all_to_all` and needs the bytes a rank moves: the flat-ring volume,
`2(N-1)/N · T` for all-reduce, `(N-1)/N · T` for the others
(`ring_factor`). A routed MoE exchange's rows are one token's top-k picks,
and picks that land on one destination travel as ONE copy — so the
exchange's coefficient is the ring's generalisation over the DISTINCT
destinations a token reaches (`RouteFanout`), which falls back to the
ring's exactly where every pick has its own destination. The other
coefficient is the number of link traversals
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
from fractions import Fraction
from typing import Any, Callable, Dict, Mapping, Optional

from backend.arch.ops import OP_REGISTRY, collective_schedule

#: The collectives, read off the registry's `is_comm_op` flag rather than
#: spelled out again here: the registration is where an op says what it is.
COMM_OPS = frozenset(
    name for name, spec in OP_REGISTRY.items() if spec.is_comm_op)


@dataclass(frozen=True)
class RouteFanout:
    """A routed MoE exchange's top-k fan-out: how many DISTINCT destinations
    one token reaches, which is how many copies of it the exchange sends.

    This is `ring_factor` generalised, and it lives beside it because it is
    the same coefficient. A flat exchange hands every payload row to its own
    destination, so a rank sends the `(D-1)/D` of the payload whose
    destination is not itself. A routed exchange's rows are one token's `k`
    picks, and two picks that land on ONE destination travel as one copy —
    DeepEP's dispatch sends a token once per distinct destination and the
    receiver fans it out to the experts it holds. So a rank sends `(D-1)/D`
    of the DISTINCT destinations per token rather than of the copies per
    token, and where every pick has its own destination the two are equal
    and the coefficient is the ring's, unchanged.

    The three counts are the dispatch's own (`layout.moe.permute` reads them
    off the node binding it already reads):

      * `picks` — the top-k width, `n_experts_active` (`k`);
      * `targets` — the destinations the top-k draws from,
        `n_route_targets` (`T`); and
      * `placed` — those of them that hold weights, `n_expert_copies` (`E`).

    `T > E` is a zero-compute reserve (LongCat-Flash): a pick that lands
    there sends nothing, so a token's copies are `k·E/T` and not `k`.

    Expected distinct destinations is hypergeometric and exact: a
    destination holding `E_d` of the `T` targets is missed by all `k` picks
    with probability `C(T-E_d, k)/C(T, k)`, so a token reaches
    `D·(1 - C(T-E_d, k)/C(T, k))` of the `D` destinations on average
    (`distinct`). Group-limited routing (`n_group` / `topk_group`) is NOT
    modelled — the picks are drawn from all `T` targets here, which
    over-counts the destinations a group-limited router reaches and so
    over-counts the bytes; see `docs/inference_cost_model.md`.
    """
    #: `k` — the top-k width (`n_experts_active`).
    picks: int
    #: `T` — the destinations the top-k draws from (`n_route_targets`).
    targets: int
    #: `E` — those of them that hold weights (`n_expert_copies`).
    placed: int

    def __post_init__(self) -> None:
        for name in ("picks", "targets", "placed"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"RouteFanout.{name} must be a positive integer, got "
                    f"{value!r}")
        if self.picks > self.targets:
            raise ValueError(
                f"a top-k of {self.picks} cannot draw from {self.targets} "
                "destinations")
        if self.placed > self.targets:
            raise ValueError(
                f"{self.placed} weighted experts among {self.targets} route "
                "targets: the weighted ones are a subset of the destinations")

    @property
    def copies_per_token(self) -> Fraction:
        """`k_eff = k·E/T` — the picks of one token that land on a weighted
        expert, which is how many copies of it the payload holds."""
        return Fraction(self.picks * self.placed, self.targets)

    def experts_per_destination(self, world: int, dest_members: int) -> int:
        """`E_d` — the weighted experts one destination of a placement level
        holds: the bank's `placed` experts spread over the route group's
        `world` ranks, `dest_members` ranks to a destination."""
        world, dest_members = int(world), int(dest_members)
        if world <= 0 or dest_members <= 0:
            raise ValueError("a placement level has a positive world and "
                             "positive members per destination")
        if (self.placed * dest_members) % world:
            raise ValueError(
                f"{self.placed} experts over {world} ranks do not divide onto "
                f"destinations of {dest_members} ranks; the dispatch requires "
                "an even expert placement")
        return self.placed * dest_members // world

    def distinct(self, destinations: int, experts_each: int) -> Fraction:
        """`A(D, E_d)` — the distinct destinations of `destinations` that one
        token's picks reach on average, each destination holding
        `experts_each` of the `targets`. Exact."""
        d, e_d = int(destinations), int(experts_each)
        if d <= 0 or e_d < 0:
            raise ValueError("a level has positive destinations and "
                             "non-negative experts each")
        # A destination with k or more targets left outside it may be
        # missed; one with fewer is always reached (`comb` is 0 there).
        miss = Fraction(math.comb(self.targets - e_d, self.picks),
                        math.comb(self.targets, self.picks))
        return d * (1 - miss)

    def remote_factor(self, world: int, *, scope_members: Optional[int] = None,
                      dest_members: int = 1) -> Fraction:
        """The payload a rank sends per payload byte at one placement level:
        `(D-1)/D · A(D, E_d) / k_eff`, over `D = scope_members/dest_members`
        destinations.

        `scope_members` is how many of the route group's `world` ranks the
        level reaches over (the world itself for the whole group, one node's
        members for the exchange inside a node) and `dest_members` how many
        ranks one destination is (one rank, or a node's members). The
        The default — the RANK level of the whole group — is the exchange's
        TOTAL: every distinct destination rank a token reaches has to
        receive it once, whatever the topology, and that needs no card
        fact, so the layer publishes it (`ring_factor`). A card's levels
        (`inference.comm_cost.placement_levels`) do not add to that total;
        they PARTITION it across fabrics.

        Zero at one destination, so a level that exchanges nothing prices
        nothing, and `(D-1)/D` exactly wherever every pick has its own
        destination."""
        world = int(world)
        scope = world if scope_members is None else int(scope_members)
        if scope % dest_members:
            raise ValueError(
                f"{scope} ranks do not divide into destinations of "
                f"{dest_members}")
        d = scope // int(dest_members)
        if d <= 1:
            return Fraction(0)
        reached = self.distinct(d, self.experts_per_destination(
            world, dest_members))
        return Fraction(d - 1, d) * reached / self.copies_per_token

    @staticmethod
    def symbolic_text(symbol: str = "D") -> str:
        """The coefficient's shape over `symbol` destinations, as a popover
        writes it: the three terms `remote_factor` multiplies — the share
        whose destination is not this rank, the distinct destinations a
        token reaches, and the copies the payload holds of it."""
        return f"({symbol}-1)/{symbol} * distinct({symbol},k)/k_eff"

    def factor_text(self, destinations: int, dest_members: int,
                    world: int) -> str:
        """`symbolic_text` with this level's numbers in it."""
        d = int(destinations)
        reached = self.distinct(d, self.experts_per_destination(
            world, dest_members))
        return (f"{d - 1}/{d} * {float(reached):.4g}/"
                f"{float(self.copies_per_token):.4g}")

    @property
    def symbol_suffix(self) -> str:
        """This fan-out as part of a price symbol's name
        (`inference.comm_cost.CollectiveSignature.symbol_suffix`): two MoE
        pools of different top-k are priced apart."""
        return f"_k{self.picks}t{self.targets}e{self.placed}"

    def as_dict(self) -> Dict[str, int]:
        return {"picks": self.picks, "targets": self.targets,
                "placed": self.placed}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RouteFanout":
        return cls(picks=int(d["picks"]), targets=int(d["targets"]),
                   placed=int(d["placed"]))


def collective_per_rank_bytes(op: str, world: int, tensor_bytes: float,
                              fanout: Optional[RouteFanout] = None) -> float:
    """Per-rank interconnect traffic for one collective, flat ring.

    `world` is the number of ranks in the collective group (N);
    `tensor_bytes` is the full communicated tensor's size in bytes.
    `fanout` is the routed exchange's top-k fan-out where the collective is
    one (`RouteFanout`), which dedups the copies a token sends.
    Returns 0 for N <= 1 (nothing leaves the rank) or a non-comm op.
    Hierarchical time accounting lives in `inference.comm_cost.collective_phases`.
    """
    n = int(world or 1)
    if n <= 1 or op not in COMM_OPS:
        return 0.0
    # An empty payload moves nothing: the ring's share of zero bytes.
    return ring_factor(op, n, fanout) * tensor_bytes


def ring_factor(op: str, world: int,
                fanout: Optional[RouteFanout] = None) -> float:
    """Per-rank bytes as a fraction of the tensor for a flat ring of
    `world` ranks: `(N-1)/N`, doubled for `all_reduce` (reduce-scatter then
    all-gather). The one place the coefficient is written; the formula
    renderers read it through `ring_factor_symbolic`.

    A routed MoE exchange states its top-k fan-out (`fanout`) and takes that
    generalisation of the same coefficient instead — the RANK level of the
    whole group, the level that needs no card fact. Only a route exchange
    ever carries one (`materialize_comm.PhysicalCollective.__post_init__` is
    the gate); the levels a CARD's nodes make are priced where the card is
    known (`inference.comm_cost.collective_phases`)."""
    n = int(world or 1)
    if collective_schedule(op) == "point_to_point":
        # One hop: the sender's whole tensor, whatever the group's size.
        return 1.0
    if fanout is not None:
        return float(fanout.remote_factor(n))
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
        if self.name == NCCL:
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
