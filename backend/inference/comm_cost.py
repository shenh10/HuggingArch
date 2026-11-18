"""Per-rank traffic of the parallelism comm primitives.

Layout materializes the collectives and owns the flat-ring volume
(`layout.comm_volume.collective_per_rank_bytes`). This module turns
that volume into time: hierarchical phases, then bytes over the
per-direction fabric bandwidth (not HBM), plus each phase's fixed latency
(the α of the α-β model): the link latency (the request's, the same on
every link at this granularity) times the link traversals the phase's
schedule waits on in sequence (`comm_volume.latency_hops`: 2 log2 N for an
all-reduce, log2 N for a reduce-scatter, all-gather or all-to-all, one for
a point-to-point send). The latency is what a
one-token decode pays per collective whatever its bytes; a cost linear in
the payload alone priced a 64-way tensor-parallel decode at a rate no
deployment reaches.

`collective_phases` is the hierarchical schedule. When a collective's
world spans multiple nodes, NCCL does not run one flat global ring —
intra-node reduce-scatter, then a smaller inter-node all-reduce, then
intra-node all-gather. A flat `2(N-1)/N · T / rdma_bw` under-counts
because the RDMA hop only moves `T/N_intra` per rank.

Inside one NVLink domain a framework does not run NCCL's schedule on every
all-reduce: it dispatches by payload to its own one-shot and two-shot
kernels and, above them, to NVLS where the switch reduces
(`comm_volume.ALGORITHMS`). Which payload takes which is the framework's
binding (`framework_bindings`, `collectives:`, read here as
`CollectiveRung`s); whether the topology admits an algorithm is the card's
(`gpu_specs`: an NVLink domain, NVLS). `collective_ladder` is the one
place both are put together: the payload ranges and, for each, the bytes
and hops it is priced at — the walk's row, the row's law and every re-read
of it on another card or at another payload read that one answer.

`collective_per_rank_bytes` is re-exported so older callers keep the
same import. At N=1 every formula collapses to zero.
"""
from __future__ import annotations

import json
import math
import re
from typing import Tuple, Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

from backend.arch.ops import collective_schedule, sp_axis_effect
from backend.arch.layout.comm_volume import (ALGORITHMS, COMM_OPS, NCCL,
                                             RouteFanout, ring_factor_symbolic)
from backend.arch.layout.comm_volume import (
    collective_per_rank_bytes as collective_per_rank_bytes,
)
from . import gpu_specs
from .phase_formulas import _fmt_si

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def sp_axis_index(axes: Optional[List[str]]) -> Optional[int]:
    """Position whose symbolic axis string carries a free `s_ctx`."""
    if not axes:
        return None
    for i, axis_str in enumerate(axes):
        if "s_ctx" in _IDENT_RE.findall(axis_str or ""):
            return i
    return None


def comm_display_shape(op: str, base_shape: Optional[List[int]],
                       axes: Optional[List[str]], world: int
                       ) -> Optional[List[int]]:
    """The tensor shape a collective presents on each rank after it runs.

    Shape-preserving except `reduce_scatter` (SP axis /world) and
    `all_gather` (×world). `base_shape` is the producer's per-rank
    output. Display only — interconnect bytes use the unadjusted
    producer shape.
    """
    if not base_shape:
        return None
    out = [int(d) for d in base_shape]
    effect = sp_axis_effect(op)
    if world > 1 and effect:
        i = sp_axis_index(axes)
        if i is not None and 0 <= i < len(out):
            v = int(out[i])
            if effect == "divide" and v % world == 0:
                out[i] = v // world
            elif effect == "multiply":
                out[i] = v * world
    return out


class Fabric(NamedTuple):
    """One card's interconnect, as `price_collective` reads it: its node
    size, each fabric's per-direction bandwidth (bytes/s) and label, one
    link traversal's latency (s, the request's `comm_latency_us`), and
    what its topology admits of a framework's own all-reduces
    (`gpu_specs.has_nvlink_domain`, `has_nvls`) beside the compute
    capability a dispatch tests. A fabric built from bandwidths alone
    admits none of them: NCCL's schedule prices every collective."""
    gpus_per_node: Optional[int]
    intra_bw: Optional[float]
    inter_bw: Optional[float]
    intra_label: str = "NVLink"
    inter_label: str = "RDMA"
    step_latency: float = 0.0
    nvlink: bool = False
    nvls: bool = False
    compute_capability: Optional[str] = None

    @classmethod
    def of(cls, gpu: "gpu_specs.GPUSpec", step_latency: float = 0.0
           ) -> "Fabric":
        """A card's fabric off its sheet row, at `step_latency`."""
        return cls(gpu_specs.gpus_per_node(gpu),
                   gpu_specs.peak_intra_bw_bytes_per_sec(gpu),
                   gpu_specs.peak_inter_bw_bytes_per_sec(gpu),
                   gpu_specs.intra_bw_label(gpu), gpu_specs.inter_bw_label(gpu),
                   step_latency)._replace(**topology(gpu))

    def without_latency(self) -> "Fabric":
        """This fabric with free steps: what its bandwidth alone charges."""
        return self._replace(step_latency=0.0)


def topology(gpu: Optional["gpu_specs.GPUSpec"]) -> Dict[str, Any]:
    """What a card's sheet row says its NVLink domain admits, as `Fabric`'s
    fields: none of it without a card."""
    if gpu is None:
        return {}
    return {"nvlink": gpu_specs.has_nvlink_domain(gpu),
            "nvls": gpu_specs.has_nvls(gpu),
            "compute_capability": gpu.compute_capability or None}


def _capability(text: str) -> Tuple[int, int]:
    major, _dot, minor = str(text).partition(".")
    return int(major), int(minor)


def members_per_node(world: Any, gpus_per_node: Any, base_stride: Any) -> int:
    """How many of a collective group's `world` members sit inside ONE node
    of `gpus_per_node` GPUs, the group's consecutive members `base_stride`
    ranks apart on the mesh: `min(world, gpus_per_node // base_stride)`, at
    least one.

    `base_stride` is the size of the parallelism nested INSIDE this group
    (`layout.mesh.base_stride`), so a TP collective (members adjacent) has
    1 and an EP collective over `tp_moe_expert` has that degree. The one
    statement of it: which fabric a group's traffic crosses, which rungs a
    framework's dispatch admits and how a routed exchange's destinations
    are placed are three readings of this one number, and they were three
    spellings of it."""
    n = max(int(world or 1), 1)
    g = max(int(gpus_per_node or 1), 1)
    s = max(int(base_stride or 1), 1)
    return min(n, max(1, g // s))


class PlacementLevel(NamedTuple):
    """One level of a collective group's placement on a card: the `fabric`
    the level's traffic crosses, how many of the group's ranks the level
    reaches over (`scope_members`) and how many ranks one destination at
    that level is (`dest_members`).

    A card places a group at two levels, and a routed exchange is priced at
    each (`collective_phases`): the RANK level, every rank inside one node
    its own destination, on the intra fabric; and the NODE level, every node
    in the world one destination, on the inter fabric. A level with one
    destination is no exchange and is left out (`placement_levels`), so a
    single node has only the rank level and a group with one member per node
    only the node level.

    A level does not ADD traffic; it says which fabric the exchange's
    traffic crosses. The total is the exchange's own — every distinct
    destination RANK a token reaches has to receive it once, whatever the
    topology (`RouteFanout.remote_factor` of the whole group, which is what
    the layer publishes as the row's bytes). `routed_shares` partitions that
    total: a level whose destination is a NODE counts its own destinations
    (the RDMA sends, one per distinct remote node), and the level whose
    destination is a RANK takes every copy left over — the destinations in
    the sender's own node, plus the forwards inside each remote node past
    the one the RDMA copy landed on. DeepEP's internode dispatch is that
    shape, and so is the hierarchical all-reduce beside it, whose intra
    all-gather leg distributes a cross-node result inside the receiving
    node and is counted.

    Conserving the total is the point rather than a nicety: the bytes a
    row publishes and the bytes its popover adds up are then one number,
    and moving a group onto a worse topology can only move copies to the
    slower fabric, never make the exchange cheaper."""
    fabric: str
    scope_members: int
    dest_members: int

    @property
    def destinations(self) -> int:
        """`D` — the distinct destinations this level sends to."""
        return self.scope_members // self.dest_members

    @property
    def per_rank_destination(self) -> bool:
        """Whether this level's destination is one RANK — the level that
        takes the remainder of the exchange's total (`routed_shares`)."""
        return self.dest_members == 1

    def own_share(self, fanout: RouteFanout, world: int) -> float:
        """The copies this level's own destination count accounts for."""
        return float(fanout.remote_factor(
            world, scope_members=self.scope_members,
            dest_members=self.dest_members))

    def factor_text_of(self, fanout: RouteFanout, world: int) -> str:
        """`own_share` as the popover writes it."""
        return fanout.factor_text(self.destinations, self.dest_members, world)


def routed_shares(fanout: RouteFanout, world: int,
                  levels: Sequence[PlacementLevel]
                  ) -> Tuple[Tuple[PlacementLevel, float], ...]:
    """Each level's share of a routed exchange's bytes, per payload byte.

    The TOTAL is the exchange's, not a level's: `fanout.remote_factor` of
    the whole group — the distinct destination ranks a token reaches, which
    is also what `comm_volume.ring_factor` publishes, so the row's bytes
    column and the sum of its phases are one number. The levels split it:
    each level whose destination is a node accounts for its own
    destinations, and the rank level takes the remainder.
    """
    total = float(fanout.remote_factor(int(world)))
    own = {index: level.own_share(fanout, world)
           for index, level in enumerate(levels)
           if not level.per_rank_destination}
    if len(levels) - len(own) > 1:
        raise ValueError(
            f"{levels!r}: two levels send to one rank, so the remainder of "
            "the exchange's total would be charged twice")
    remainder = total - sum(own.values())
    if remainder < -1e-12:
        raise ValueError(
            f"a routed exchange of {world} ranks accounts "
            f"{sum(own.values())} of a total {total}: the levels' "
            "destinations are not a partition of the exchange's")
    return tuple((level, own.get(index, max(remainder, 0.0)))
                 for index, level in enumerate(levels))


def placement_levels(world: Any, gpus_per_node: Any, base_stride: Any
                     ) -> Tuple[PlacementLevel, ...]:
    """The levels a group of `world` ranks exchanges at on this card, in
    path order (inside a node, then between nodes), with the levels that
    reach one destination left out.

    A world that is not a whole number of nodes has no node level: its nodes
    hold different numbers of the group's members, so the hypergeometric
    count of destinations per node does not apply to any of them. It gets
    ONE level instead — every distinct destination rank, charged on the
    inter fabric, which is pessimistic in the fabric and is the one
    placement this does not level. One level or two, routed pricing is the
    same loop over what this returns.

    Empty only for a group that exchanges nothing (one rank).

    Computed from the CARD (`Fabric.gpus_per_node`) and the group's spacing,
    so it is computed where the card is known and not carried from a walk —
    the same group is placed differently on a 4-GPU node and a 72-GPU one,
    and a row re-read on another card must be re-levelled there."""
    n = max(int(world or 1), 1)
    m = members_per_node(n, gpus_per_node, base_stride)
    if n % m:
        return (PlacementLevel("inter", n, 1),)
    levels = (PlacementLevel("intra", m, 1), PlacementLevel("inter", n, m))
    return tuple(level for level in levels if level.destinations > 1)


class CollectiveRung(NamedTuple):
    """One entry of a framework's all-reduce dispatch (`framework_bindings`,
    `collectives:`): the algorithm it runs (`comm_volume.ALGORITHMS`) on a
    payload of at most `max_bytes` (inclusive) and at most
    `max_bytes_times_world` summed over the group, in a group whose size
    is one of `worlds` (the sizes the framework's kernel accepts; a group of
    any other size runs NCCL), on a card whose compute capability is at
    least `min_compute_capability` and below `below_compute_capability`. A
    row carries its collective's rungs (`as_dict`), so a re-read of it on
    another card needs no binding."""
    algorithm: str
    max_bytes: Optional[int] = None
    max_bytes_times_world: Optional[int] = None
    min_compute_capability: Optional[str] = None
    below_compute_capability: Optional[str] = None
    worlds: Tuple[int, ...] = ()

    def cap(self, world: int) -> float:
        """The largest payload (bytes, inclusive) this rung takes in a
        group of `world`."""
        caps = [math.inf]
        if self.max_bytes is not None:
            caps.append(float(self.max_bytes))
        if self.max_bytes_times_world is not None:
            caps.append(float(self.max_bytes_times_world) / int(world))
        return min(caps)

    def runs_on(self, fabric: Fabric, *, world: int, base_stride: int) -> bool:
        """Whether the card's topology and capability admit this rung for a
        group of `world` ranks `base_stride` apart: every member in one
        NVLink domain, the topology the algorithm needs, and the capability
        the framework tests."""
        n = int(world)
        if n <= 1 or members_per_node(
                n, fabric.gpus_per_node, base_stride) < n:
            return False
        if n not in self.worlds:
            return False
        needs = ALGORITHMS[self.algorithm].needs
        if (needs == "nvlink" and not fabric.nvlink) or (
                needs == "nvls" and not fabric.nvls):
            return False
        if self.min_compute_capability or self.below_compute_capability:
            if not fabric.compute_capability:
                return False
            cc = _capability(fabric.compute_capability)
            if (self.min_compute_capability
                    and cc < _capability(self.min_compute_capability)):
                return False
            if (self.below_compute_capability
                    and cc >= _capability(self.below_compute_capability)):
                return False
        return True

    def as_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if k == "worlds" else v)
                for k, v in self._asdict().items() if v is not None}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CollectiveRung":
        d = dict(d)
        return cls(**{**d, "worlds": tuple(int(w) for w in d.get("worlds", ()))})


class Segment(NamedTuple):
    """One payload range of a collective on a card: payloads up to
    `max_bytes` (inclusive, past the ranges before it) run `algorithm`, at
    `s_per_byte` seconds per payload byte and `latency_s` fixed over the
    step's instances."""
    algorithm: str
    max_bytes: float
    s_per_byte: float
    latency_s: float


class CollectiveSignature(NamedTuple):
    """One collective as everything but the card: what `collective_ladder`
    reads besides the fabric.

    A row's collective is priced in more than one place — the walk's own
    row, a re-read of that row on another card or at another batch
    (`row_roofline.CardForm`), the step's law in the browser
    (`sol_law`) — and all of them have to name the SAME collective and
    price it the same way. The facts were spelled out at each: five places
    wrote `(op, world, count, stride, rungs)` in three notations (a kwargs
    call, a JSON object, an underscore-joined symbol name) and nothing held
    them to each other. A sixth fact would have had to be added five times.

    So the facts are a type, and the three notations are its three
    projections: `key` (the canonical JSON a cached form and a law record
    carry), `symbol_suffix` (the name a step's law prices this collective
    under — two collectives that differ in any fact here must differ in
    that name, or one's price is read for the other) and `ladder` (the
    pricing itself).

    `route_fanout` is the sixth fact: a routed MoE exchange's top-k
    fan-out, which prices its dedup (`comm_volume.RouteFanout`). It is part
    of the signature and not of the card because two expert pools of
    different top-k in ONE step — a speculative draft's 128-choose-3 pool
    beside the main 384-choose-6 — are two prices, and a symbol naming only
    `(op, world, count, stride)` would hand one of them the other's.
    """
    op: str
    world: int
    count: int
    stride: int
    rungs: Tuple[CollectiveRung, ...] = ()
    route_fanout: Optional[RouteFanout] = None

    @classmethod
    def of(cls, row: Mapping[str, Any]) -> Optional["CollectiveSignature"]:
        """The collective a per-op row prices, or None for a row that prices
        none: not a collective, or a group of one (nothing leaves the rank).
        The row's own facts (`phases.compute_per_op_timing` writes them, and
        `row_roofline.row_law` is what a re-read carries) — read hard, so a
        row that is missing them raises here instead of being taken for a
        row that is not a collective."""
        if not row["is_comm"] or int(row["comm_world"] or 1) <= 1:
            return None
        fanout = row.get("comm_route_fanout")
        return cls(op=str(row["comm_op"]), world=int(row["comm_world"]),
                   count=int(row["count_per_step"]),
                   stride=int(row["comm_base_stride"]),
                   rungs=tuple(CollectiveRung.from_dict(r)
                               for r in row.get("comm_rungs") or ()),
                   route_fanout=(RouteFanout.from_dict(fanout)
                                 if fanout else None))

    @property
    def key(self) -> str:
        """These facts as canonical JSON: what a cached card-independent
        form carries instead of the row (`CardForm.sigs`), and what a law
        record names its collectives by (`sol_law`). A collective with no
        fan-out states none, so a record written before routed exchanges
        were deduped reads back as the collective it was."""
        facts: Dict[str, Any] = {
            "op": self.op, "world": self.world, "count": self.count,
            "stride": self.stride,
            "rungs": [r.as_dict() for r in self.rungs]}
        if self.route_fanout is not None:
            facts["route_fanout"] = self.route_fanout.as_dict()
        return json.dumps(facts, sort_keys=True)

    @classmethod
    def from_key(cls, key: str) -> "CollectiveSignature":
        d = json.loads(key)
        fanout = d.get("route_fanout")
        return cls(op=str(d["op"]), world=int(d["world"]),
                   count=int(d["count"]), stride=int(d["stride"]),
                   rungs=tuple(CollectiveRung.from_dict(r)
                               for r in d.get("rungs") or ()),
                   route_fanout=(RouteFanout.from_dict(fanout)
                                 if fanout else None))

    @property
    def symbol_suffix(self) -> str:
        """What a step's law appends to a collective card symbol's name
        (`row_roofline.collective_symbol`, `sol_law.cardBinding`) so that
        rows of different collectives add: one name per distinct
        signature.

        The rungs are the one fact of the signature this name leaves out,
        and may be: they are the FRAMEWORK's dispatch of this collective,
        which one walk's rows all share (`ServingLayers.collectives` is per
        op), so two rows of one step cannot differ in them while agreeing
        on everything else. `key` carries them, because a cached form and a
        law record are re-read on another card and have to re-price the
        ladder."""
        out = f"_{self.op}_{self.world}_{self.count}_{self.stride}"
        if self.route_fanout is not None:
            out += self.route_fanout.symbol_suffix
        return out

    def ladder(self, fabric: "Fabric") -> Tuple[Segment, ...]:
        """This collective's payload ranges on `fabric`
        (`collective_ladder`)."""
        return collective_ladder(
            op=self.op, world=self.world, count_per_step=self.count,
            base_stride=self.stride, fabric=fabric, rungs=self.rungs,
            fanout=self.route_fanout)


def collective_ladder(*, op: str, world: int, count_per_step: int,
                      base_stride: int, fabric: Fabric,
                      rungs: Sequence[CollectiveRung] = (),
                      fanout: Optional[RouteFanout] = None
                      ) -> Tuple[Segment, ...]:
    """A collective's payload ranges on `fabric`: one per rung of the
    framework's dispatch, in its order, then NCCL's schedule for every
    payload past them. A rung the card does not admit is an empty range
    (`max_bytes` 0) priced as NCCL's, so the ladder has one shape on every
    card and a zero payload costs what it does without one. `fanout` is the
    routed exchange's top-k fan-out where the collective is one. The one
    answer to which algorithm a payload takes (`Segment.of`) and what it
    costs."""
    def priced(algorithm: str) -> Tuple[float, float]:
        per_byte, _rows = price_collective(
            op=op, world=world, tensor_bytes=1.0, count_per_step=count_per_step,
            fabric=fabric.without_latency(), base_stride=base_stride,
            algorithm=algorithm, fanout=fanout)
        return per_byte, collective_latency(
            op=op, world=world, count_per_step=count_per_step,
            gpus_per_node=fabric.gpus_per_node, base_stride=base_stride,
            step_latency=fabric.step_latency, algorithm=algorithm,
            fanout=fanout)

    nccl = priced(NCCL)
    out = []
    for rung in rungs:
        if rung.runs_on(fabric, world=world, base_stride=base_stride):
            out.append(Segment(rung.algorithm, rung.cap(world),
                               *priced(rung.algorithm)))
        else:
            out.append(Segment(NCCL, 0.0, *nccl))
    out.append(Segment(NCCL, math.inf, *nccl))
    return tuple(out)


def ladder_segment(ladder: Sequence[Segment], tensor_bytes: float) -> Segment:
    """The range a payload of `tensor_bytes` falls in: the first whose
    bound holds it, as the row's law chooses (`row_roofline`)."""
    for seg in ladder:
        if tensor_bytes <= seg.max_bytes:
            return seg
    return ladder[-1]


def collective_latency(*, op: str, world: int, count_per_step: int,
                       gpus_per_node: Optional[int], base_stride: int,
                       step_latency: float, algorithm: str = NCCL,
                       fanout: Optional[RouteFanout] = None) -> float:
    """The fixed part of a collective's time over a step: each phase of its
    schedule (`collective_phases`, run by `algorithm`) pays `step_latency`
    per link traversal it waits on, for every instance in the step.
    Independent of the payload once the algorithm is chosen."""
    phases = collective_phases(
        op=op, world=int(world), tensor_bytes=1.0,
        gpus_per_node=int(gpus_per_node or 1),
        base_stride=int(base_stride or 1), algorithm=algorithm, fanout=fanout)
    return sum(step_latency * ph["hops"] for ph in phases) * int(count_per_step)


def price_collective(
    *,
    op: str,
    world: int,
    tensor_bytes: float,
    count_per_step: int,
    fabric: Fabric,
    base_stride: int,
    algorithm: str = NCCL,
    fanout: Optional[RouteFanout] = None,
    t_sym: Optional[str] = None,
    t_sub: Optional[str] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """One collective's time and per-phase rows on `fabric`, run by
    `algorithm` (`collective_ladder` chooses it for a payload).

    The anchor row in `phases` and the Gallery's re-stamp of the same
    layout on another card both call this, so a second card is priced by
    the same loop as the first rather than by a scalar ratio of link
    bandwidths applied to the first card's total.
    """
    phases_list = collective_phases(
        op=op, world=int(world), tensor_bytes=float(tensor_bytes),
        gpus_per_node=int(fabric.gpus_per_node or 1),
        base_stride=int(base_stride or 1), algorithm=algorithm,
        fanout=fanout, t_sym=t_sym, t_sub=t_sub,
    )
    t_comm = 0.0
    rows: List[Dict[str, Any]] = []
    for ph in phases_list:
        where = ph["fabric"]
        fabric_bw = fabric.intra_bw if where == "intra" else fabric.inter_bw
        fabric_label = (fabric.intra_label if where == "intra"
                        else fabric.inter_label)
        phase_bytes = float(ph["bytes_per_rank"]) * int(count_per_step)
        phase_latency = (fabric.step_latency * ph["hops"]
                         * int(count_per_step))
        phase_t = (phase_bytes / fabric_bw if fabric_bw and fabric_bw > 0
                   else 0.0) + phase_latency
        t_comm += phase_t
        rows.append({
            "name": ph["name"],
            "algorithm": ph["algorithm"],
            "fabric": where,
            "fabric_label": fabric_label,
            "world": ph["world"],
            "bytes_per_rank": phase_bytes,
            "bw_bytes_per_sec": float(fabric_bw or 0.0),
            # The link traversals one instance of the phase waits on, and
            # their time over the row's instances.
            "hops": ph["hops"],
            "latency_s": phase_latency,
            "t_phase_s": phase_t,
            "symbolic": ph["symbolic"],
            "substituted": ph["substituted"],
        })
    return t_comm, rows


def collective_phases(
    op: str,
    world: int,
    tensor_bytes: float,
    *,
    gpus_per_node: int,
    base_stride: int = 1,
    algorithm: str = NCCL,
    fanout: Optional[RouteFanout] = None,
    t_sym: Optional[str] = None,
    t_sub: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Per-phase wire bytes for one collective, hierarchy-aware.

    `algorithm` (`comm_volume.ALGORITHMS`) runs a group that fits one
    node as one phase at its bytes and hops; anything but NCCL's schedule is
    refused for a group that does not (`collective_ladder` never asks).
    Every phase states its `algorithm` and `hops`.

    `fanout` is the top-k fan-out of a ROUTED MoE exchange
    (`comm_volume.RouteFanout`, carried by the exchange the dispatch
    grafted). Such an exchange is priced one phase per PLACEMENT LEVEL
    (`placement_levels`) — the ranks inside a node over NVLink and the nodes
    inside the world over RDMA — because a token is sent once per distinct
    destination at each level, not once per expert it picked. A collective
    with no fan-out, a Ulysses CP exchange included, is priced exactly as
    below: that is a declared fact about the collective, not a gap.

    Returns a list of `{name, fabric, world, tensor_bytes, bytes_per_rank,
    symbolic, substituted}` dicts. `fabric` is `"intra"` (fits inside one
    NVLink/PCIe domain) or `"inter"` (RDMA / cross-node). Sum of
    `bytes_per_rank` across the returned phases is the total wire bytes
    a rank actually transmits when the cost is summed `bytes / fabric_bw`.

    `base_stride` is the number of ranks BETWEEN consecutive members of
    this collective's group — i.e. the size of the parallelism inner_group
    nested INSIDE it on the device mesh. The standard layout nests TP
    innermost, so a TP collective has `base_stride = 1` (members are
    adjacent ranks), but an OUTER axis composed over TP is spaced out:
    a CP collective over `tp_attention` has `base_stride = tp_attention`,
    an EP collective over `tp_moe_expert` has `base_stride =
    tp_moe_expert`. Members co-located in one node are
    `members_per_node(world, gpus_per_node, base_stride)`.

    With `base_stride = 1` that is `min(world, gpus_per_node)` and the
    whole function reduces EXACTLY to the previous size-only behavior —
    so single-TP costs are byte-identical. It only changes the composed
    case: a small outer-axis collective (e.g. cp=4) whose stride pushes
    every member onto a separate node (`members_per_node == 1`) correctly
    takes the inter fabric instead of being mis-priced as intra.

    Fabrics:
      * Routed exchange (`fanout`): one phase per placement level with more
        than one destination (`placement_levels`) — the ranks inside a node
        on the intra fabric, the nodes inside the world on the inter one.
        The exchange's TOTAL bytes are its own (every distinct destination
        rank receives the token once) and the levels partition it
        (`routed_shares`), so the phases always sum to the bytes the row
        publishes. A single node has only the first level (and so keeps
        today's `ceil(log2 world)` hops: deduping inside a node forwards
        nothing further); a group with one member per node, and one whose
        world is not a whole number of nodes, only the second — the whole
        total on the slower fabric, pessimistic in the fabric and the one
        placement this does not level.
      * Point-to-point (`send_recv`): one hop of the whole tensor, on the
        intra fabric when the group fits in one node and on the inter fabric
        otherwise — the hop's place in the group is not known here, so a
        group that straddles nodes prices every hop at its slowest.
      * Single node (`world <= members_per_node`): one intra ring.
      * Fully cross-node (`members_per_node == 1`): one inter ring.
      * Straddle, AR, divisible: 3-phase Megatron/NCCL hierarchical AR —
          1. intra reduce-scatter on T     → (N_intra-1)/N_intra · T
          2. inter all-reduce on the shard → 2(N_inter-1)/N_inter · T/N_intra
          3. intra all-gather on T         → (N_intra-1)/N_intra · T
        where `N_intra = members_per_node`, `N_inter = world / N_intra`.
      * Straddle, non-AR or non-divisible: single inter ring (RS/AG/A2A
        have no widely-deployed hierarchical decomposition that wins, and
        the hierarchical AR schedule isn't well-defined when world isn't a
        multiple of the per-node group).
    """
    n = int(world or 1)
    if n <= 1 or op not in COMM_OPS or tensor_bytes <= 0:
        return []
    g = max(int(gpus_per_node or 1), 1)
    s = max(int(base_stride or 1), 1)
    # Group members that land inside one node, given the inner-inner_group
    # stride. base_stride=1 → min(n, g) (the legacy size-only rule).
    m = members_per_node(n, g, s)
    T = float(tensor_bytes)
    # `T` token in the formula. When the caller passes the tensor's
    # axis·dtype expansion (`t_sym` = "(B·s_ctx·d_model·dt_a)", `t_sub` =
    # "(237·1·7168·2)"), it replaces the opaque `T` / SI-byte literal so the
    # popover shows where the bytes come from. Defaults keep the prior
    # behavior byte-for-byte ("T" symbolic, SI-byte substituted).
    Tsym = t_sym or "T"
    Tsub_full = t_sub or _fmt_si(int(T), " B")

    def _phase(name: str, fabric: str, w: int, tb: float, sym: str, sub: str,
               alg: str = NCCL, *,
               factor: Optional[float] = None) -> Dict[str, Any]:
        if w <= 1 or tb <= 0:
            return {}
        # The coefficients are declared once, in `comm_volume.ALGORITHMS`;
        # a routed level's is the fan-out's own (`RouteFanout`), which is
        # that same coefficient generalised and is passed in as `factor`.
        run = ALGORITHMS[alg]
        return {
            "name": name,
            "algorithm": alg,
            "fabric": fabric,
            "world": int(w),
            "tensor_bytes": float(tb),
            "bytes_per_rank": (run.factor(name, w) if factor is None
                               else float(factor)) * tb,
            "hops": run.hops(name, w),
            "symbolic": sym,
            "substituted": sub,
        }

    # A routed exchange, level by level: the exchange's own total split
    # across the fabrics the card's levels name (`routed_shares`).
    if fanout is not None:
        levels = placement_levels(n, g, s)
        if algorithm != NCCL:
            raise ValueError(
                f"a routed {op} of {n} ranks runs NCCL's schedule, not "
                f"{algorithm}: the framework's own kernels dispatch "
                "all-reduces")
        shares = routed_shares(fanout, n, levels)
        total_sym = RouteFanout.symbolic_text("N_ranks")
        total_sub = fanout.factor_text(n, 1, n)
        out = []
        for level, share in shares:
            if level.per_rank_destination and len(shares) > 1:
                # The remainder: the exchange's total less the copies the
                # coarser levels carry, so the popover shows the
                # subtraction rather than a number from nowhere.
                coarse_sym = " - ".join(
                    RouteFanout.symbolic_text("N_nodes")
                    for other, _s in shares if other is not level)
                coarse_sub = " - ".join(
                    other.factor_text_of(fanout, n)
                    for other, _s in shares if other is not level)
                sym = f"[{total_sym} - {coarse_sym}] * {Tsym}"
                sub = f"[{total_sub} - ({coarse_sub})] * {Tsub_full}"
            elif level.per_rank_destination:
                sym = f"{total_sym} * {Tsym}"
                sub = f"{total_sub} * {Tsub_full}"
            else:
                sym = f"{RouteFanout.symbolic_text('N_nodes')} * {Tsym}"
                sub = f"{level.factor_text_of(fanout, n)} * {Tsub_full}"
            ph = _phase(op, level.fabric, level.destinations, T, sym, sub,
                        factor=share)
            if ph:
                out.append(ph)
        # A routed exchange is priced HERE and nowhere else: the flat paths
        # below are for a collective that carries no fan-out, and falling
        # through to them would price this one twice over as a ring. An
        # empty list is a group of one, which `collective_phases` returned
        # for at the top.
        return out

    # Intra-domain — the whole group fits inside one NVLink/PCIe inner_group
    # (every member co-located). One phase, by `algorithm`.
    if n <= m:
        run = ALGORITHMS[algorithm]
        ph = _phase(
            op, "intra", n, T,
            f"{run.factor_symbolic(op)} * {Tsym}",
            f"{run.factor_symbolic(op, n)} * {Tsub_full}",
            algorithm,
        )
        return [ph] if ph else []
    if algorithm != NCCL:
        raise ValueError(
            f"{op} of {n} ranks spans more than one node of {g}: only NCCL's "
            f"schedule runs it, not {algorithm}")

    # Cross-domain. AR straddling node boundaries (more than one member
    # per node) uses a 3-phase hierarchy; everything else — fully
    # cross-node (one member per node), non-AR, or non-divisible — falls
    # back to a single inter-fabric ring.
    if collective_schedule(op) != "hierarchical" or m <= 1 or (n % m) != 0:
        coeff_sym = ring_factor_symbolic(op)
        coeff_sub = ring_factor_symbolic(op, n)
        ph = _phase(
            op, "inter", n, T,
            f"{coeff_sym} * {Tsym}",
            f"{coeff_sub} * {Tsub_full}",
        )
        return [ph] if ph else []

    n_intra = m
    n_inter = n // m
    shard_T = T / n_intra
    # Inter phase runs on the post-RS shard T/N_intra. When T is expanded
    # the shard reads `(B·s_ctx·d_model·dt_a)/N_intra` (division explicit); the
    # default path keeps the pre-divided SI-byte literal byte-for-byte.
    shard_sub = f"{Tsub_full}/{n_intra}" if t_sub else _fmt_si(int(shard_T), " B")
    out: List[Dict[str, Any]] = []
    p1 = _phase(
        "reduce_scatter", "intra", n_intra, T,
        f"{ring_factor_symbolic('reduce_scatter', symbol='N_intra')} * {Tsym}",
        f"{ring_factor_symbolic('reduce_scatter', n_intra)} * {Tsub_full}",
    )
    if p1:
        out.append(p1)
    p2 = _phase(
        "all_reduce", "inter", n_inter, shard_T,
        f"{ring_factor_symbolic('all_reduce', symbol='N_inter')} * {Tsym}/N_intra",
        f"{ring_factor_symbolic('all_reduce', n_inter)} * {shard_sub}",
    )
    if p2:
        out.append(p2)
    p3 = _phase(
        "all_gather", "intra", n_intra, T,
        f"{ring_factor_symbolic('all_gather', symbol='N_intra')} * {Tsym}",
        f"{ring_factor_symbolic('all_gather', n_intra)} * {Tsub_full}",
    )
    if p3:
        out.append(p3)
    return out

