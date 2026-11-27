"""Per-rank traffic of the parallelism comm primitives.

Layout materializes the collectives and owns the flat-ring volume
(`layout.comm_volume.collective_per_rank_bytes`). This module turns
that volume into time: hierarchical phases, then bytes over the
per-direction fabric bandwidth (not HBM), plus each phase's fixed latency
(the α of the α-β model): one link traversal's latency (the request's
link latency, and on a phase that crosses nodes the card's RDMA hop on top
of it, `Fabric.hop_latency`) times the link traversals the phase's
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

A collective's time is its ENVELOPE: the fastest, at its payload, of the
algorithms that run it (`comm_volume.ALGORITHMS`: one-shot, two-shot and
NVLS all-reduces inside one NVLink domain, NCCL's schedule everywhere) that
the card's fabric can run (`Envelope`). Which ones it can run is a hardware
fact and nothing else — an NVLink domain, a switch that reduces, the group
inside one node (`Fabric`, off `gpu_specs`) — so the price is the same under
every framework: a framework that falls back to a slower path is slower
than this bound, not a different bound. Each algorithm's time is the α-β
law (`price_collective`, `collective_latency`), so a small payload takes a
one-traversal algorithm and a large one the algorithm that moves the fewest
bytes, by the same law at every payload.

`collective_per_rank_bytes` is re-exported so older callers keep the
same import. At N=1 every formula collapses to zero.
"""
from __future__ import annotations

import json
import math
import re
from typing import Tuple, Any, Dict, List, Mapping, NamedTuple, Optional, Sequence

from backend.arch.ops import collective_schedule, sp_axis_effect
from backend.arch.layout.comm_volume import (ALGORITHMS, COMM_OPS,
                                             FABRIC_CAPABILITIES, NCCL,
                                             STAGE_KINDS, RouteFanout,
                                             algorithms_for,
                                             ring_factor_symbolic)
from backend.arch.layout.comm_volume import (
    collective_per_rank_bytes as collective_per_rank_bytes,
)
from backend.code_trace.memo import memoize
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


#: The physical reaches a collective group can have, outermost last. The
#: names are the domain's, not a vendor's: one chassis, the scale-up domain
#: that chassis belongs to, and everything past it. On an Atlas A3 that
#: reads 16 dies / a supernode over Lingqu (UB) / RoCE; on an NVL72 rack it
#: reads one tray / the NVLink rack / InfiniBand. `docs/parallelism.md`
#: registers the words.
#: A TIER, not a size. The obvious encoding — how many ranks the domain
#: holds — cannot be written down for `scaleout`, which has no fixed
#: extent at all, and would split one fabric's measurements across
#: several keys the day a 384-die and a 768-die supernode both exist.
#: What a lookup needs to know is which fabric a group's traffic crossed,
#: and that is an ordered three, so it is numbered: a corpus cell's
#: values are positive integers (`benchmarks` refuses anything else), and
#: a name would have to be spelled identically by every writer forever.
COMM_REACH_HOST = 1
COMM_REACH_SCALEUP = 2
COMM_REACH_SCALEOUT = 3
COMM_REACHES = (COMM_REACH_HOST, COMM_REACH_SCALEUP, COMM_REACH_SCALEOUT)

#: What a person writes (`--comm-domain`) against what a cell carries.
#: One mapping, read by the CLI that parses the flag and by anything
#: rendering a row back: two spellings of this would drift the day a
#: fourth tier appears.
COMM_REACH_NAMES = {"host": COMM_REACH_HOST,
                    "scaleup": COMM_REACH_SCALEUP,
                    "scaleout": COMM_REACH_SCALEOUT}
COMM_REACH_LABELS = {code: name for name, code in COMM_REACH_NAMES.items()}


def comm_reach(world: int, *, base_stride: int = 1,
               ranks_per_host: Optional[int],
               ranks_per_scaleup_domain: Optional[int] = None) -> str:
    """How far a collective group of `world` ranks physically stretches.

    `host` when every member sits in one chassis, `scaleup` when it leaves
    the chassis but stays inside the scale-up domain that chassis belongs
    to, `scaleout` when it leaves that domain too. The three differ by an
    order of magnitude of bandwidth each, which is why a measurement of one
    must never answer for another.

    `members_per_node` is the division this shares with the pricing
    side, so a lookup and a roofline cannot disagree about where a
    group sits.

    `ranks_per_scaleup_domain` absent means the card declares no domain
    wider than its chassis, and the answer is the two-valued one this
    module has always given: a group either fits the node or does not.
    That is every row on the sheet today, so nothing changes until a card
    states a wider domain.
    """
    n = max(int(world or 1), 1)
    if n <= 1:
        return COMM_REACH_HOST
    host = max(int(ranks_per_host or 1), 1)
    if n <= members_per_node(n, host, base_stride):
        return COMM_REACH_HOST
    domain = max(int(ranks_per_scaleup_domain or host), host)
    if domain > host and n <= members_per_node(n, domain, base_stride):
        return COMM_REACH_SCALEUP
    return COMM_REACH_SCALEOUT


class Fabric(NamedTuple):
    """One card's interconnect, as `price_collective` reads it: its node
    size, each fabric's per-direction bandwidth (bytes/s) and label, one
    link traversal's latency (s, the request's `comm_latency_us`) and what
    a traversal across nodes waits on beyond it (`inter_latency`, the
    card's `rdma_hop_latency_us`; `hop_latency`), how far the scale-up
    domain reaches when it is wider than the chassis (`scaleup_domain`),
    and what its topology admits of the all-reduces that need more than
    a link (`gpu_specs.has_scaleup_domain`, `has_nvls`). A fabric built
    from bandwidths alone admits none of them: NCCL's schedule prices
    every collective, every hop at the one link latency."""
    gpus_per_node: Optional[int]
    intra_bw: Optional[float]
    inter_bw: Optional[float]
    intra_label: str = "NVLink"
    inter_label: str = "RDMA"
    step_latency: float = 0.0
    nvlink: bool = False
    nvls: bool = False
    inter_latency: float = 0.0
    #: Ranks the SCALE-UP domain holds, when it is wider than the
    #: chassis (`gpu_specs.ranks_per_scaleup_domain`). None reads as the
    #: chassis, which is every card that declares no supernode. It is
    #: this, not `gpus_per_node`, that splits a collective into intra
    #: and inter phases: the question a split asks is whether the
    #: traffic leaves the FAST FABRIC, and on a supernode 768 dies
    #: answer that together across machines.
    scaleup_domain: Optional[int] = None
    #: The card this is the fabric of (`GPUSpec.sku`), stated with the rest
    #: of its topology; empty for a fabric built with no card. It names the
    #: card a refusal is about (`bandwidth`).
    sku: str = ""

    @classmethod
    def of(cls, gpu: "gpu_specs.GPUSpec", step_latency: float = 0.0
           ) -> "Fabric":
        """A card's fabric off its sheet row, at `step_latency`.

        The topology fields are `topology()`'s to state -- including the
        scale-up width, which this used to read a second time of its own.
        Two readings of one row agreed only by inspection, and the field
        was lost on the other construction site precisely because no single
        place owned it."""
        return cls(gpu_specs.gpus_per_node(gpu),
                   gpu_specs.peak_intra_bw_bytes_per_sec(gpu),
                   gpu_specs.peak_inter_bw_bytes_per_sec(gpu),
                   gpu_specs.intra_bw_label(gpu), gpu_specs.inter_bw_label(gpu),
                   step_latency)._replace(**topology(gpu))

    def scaleup_ranks(self) -> int:
        """Ranks one scale-up domain holds: the width that splits a
        collective into intra and inter phases. The question a split asks
        is whether the group's traffic stays on the fast fabric, and on a
        supernode 768 dies do that across machines; a card declaring no
        supernode has this equal to its chassis. A phase's bytes and the
        hops it waits on are both read off that one split, so the two
        cannot disagree about which fabric a phase crossed."""
        return int(self.scaleup_domain or self.gpus_per_node or 1)

    def bandwidth(self, where: str) -> Optional[float]:
        """The bandwidth (bytes/s) a phase on fabric `where` (`"intra"` /
        `"inter"`) moves its bytes at, or None where no times were asked
        for. A CARD that states none for a fabric a phase crosses is
        refused here (`gpu_specs.NoFabricBandwidth`): this is where every
        phase's bytes meet a bandwidth, so there is no path on which a
        blank column divides into zero seconds.

        "No times were asked for" is a fabric handed neither bandwidth: a
        caller walking a step for its bytes and launches (a kernel plan, a
        fusion walk, a single module's bench) passes the card for its
        dispatch and no interconnect at all. That cannot be a card with
        both cells blank -- every sheet row states the fabric inside its
        node (`gpu_specs.check_node_fabric`) -- so one bandwidth present
        and the other absent is the sheet not saying."""
        bw = self.intra_bw if where == "intra" else self.inter_bw
        if bw and bw > 0:
            return bw
        if not self.sku or not (self.intra_bw or self.inter_bw):
            return None
        domain = self.scaleup_ranks()
        if where == "intra":
            raise gpu_specs.NoFabricBandwidth(
                self.sku, "scaleup_gbps",
                f"inside its {domain}-rank scale-up domain")
        raise gpu_specs.NoFabricBandwidth(
            self.sku, "rdma_gbps",
            f"between its {domain}-rank scale-up domains")

    def without_latency(self) -> "Fabric":
        """This fabric with free steps: what its bandwidth alone charges."""
        return self._replace(step_latency=0.0, inter_latency=0.0)

    def hop_latency(self, where: str) -> float:
        """What one link traversal on fabric `where` (`"intra"` /
        `"inter"`, a phase's `fabric`) waits on: the link latency, and
        across nodes the card's RDMA hop on top of it."""
        return self.step_latency + (self.inter_latency if where == "inter"
                                    else 0.0)

    def offers(self) -> Dict[str, bool]:
        """Each fabric capability (`comm_volume.FABRIC_CAPABILITIES`) and
        whether this fabric has it, with why not where it does not."""
        return {"nvlink": self.nvlink, "nvls": self.nvls}

    def lacks(self, capability: str) -> str:
        """What a refusal says of a capability this fabric does not have."""
        return {"nvlink": f"card has no NVLink domain ({self.intra_label})",
                "nvls": "card has no NVLink multicast (NVLS)"}[capability]

    def refusal(self, algorithm: str, *, op: str, world: int,
                base_stride: int) -> Optional[str]:
        """Why this fabric cannot run `algorithm` for a collective `op` of
        `world` ranks `base_stride` apart, or None when it can.

        Hardware facts only: the capabilities the algorithm needs
        (`CollectiveAlgorithm.needs`), each the NVLink domain's, so an
        algorithm that needs any runs only on a group inside one node
        (`members_per_node`). One that needs none — NCCL's schedule — runs
        everywhere. Asking for an algorithm that does not run `op` at all
        is the caller's error (`comm_volume.algorithms_for` lists those
        that do)."""
        run = ALGORITHMS[algorithm]
        if not run.runs(op):
            raise ValueError(f"{algorithm} runs only {sorted(run.ops)}, "
                             f"not {op}")
        if not run.needs:
            return None
        n = int(world)
        inside = members_per_node(n, self.gpus_per_node, base_stride)
        if inside < n:
            return (f"the group spans {-(-n // inside)} nodes; "
                    f"{algorithm.replace('_', '-')} runs inside one NVLink "
                    "domain")
        offers = self.offers()
        for capability in sorted(run.needs):
            if not offers[capability]:
                return self.lacks(capability)
        return None


if set(Fabric(None, None, None).offers()) != FABRIC_CAPABILITIES:
    raise ValueError("Fabric.offers answers other capabilities than "
                     "comm_volume.FABRIC_CAPABILITIES declares")


def topology(gpu: Optional["gpu_specs.GPUSpec"]) -> Dict[str, Any]:
    """What a card's sheet row says of its fabrics beyond their bandwidths,
    as `Fabric`'s fields: what its NVLink domain admits, how WIDE that
    domain is, and what a cross-node hop waits on. None of it without a
    card.

    The width belongs here rather than at each construction site because
    there are two of them (`Fabric.of` and `phases`' own), and a field only
    one of them passes is a field the other silently defaults. That is what
    happened to `scaleup_domain`: the sheet knew the Ascend supernode holds
    768 ranks, `Fabric.of` passed it, and the phase walk — the path an
    estimate actually takes — built its fabric by hand without it. Every
    group wider than one chassis was then split at `gpus_per_node` and its
    middle phase priced as RDMA, which is the error the column was added to
    remove."""
    if gpu is None:
        return {}
    return {"sku": gpu.sku,
            "nvlink": gpu_specs.has_scaleup_domain(gpu),
            "nvls": gpu_specs.has_nvls(gpu),
            "scaleup_domain": gpu_specs.ranks_per_scaleup_domain(gpu),
            "inter_latency": gpu_specs.inter_hop_latency_s(gpu)}


def members_per_node(world: Any, gpus_per_node: Any, base_stride: Any) -> int:
    """How many of a collective group's `world` members sit inside ONE node
    of `gpus_per_node` GPUs, the group's consecutive members `base_stride`
    ranks apart on the mesh: `min(world, gpus_per_node // base_stride)`, at
    least one.

    `base_stride` is the size of the parallelism nested INSIDE this group
    (`layout.mesh.base_stride`), so a TP collective (members adjacent) has
    1 and an EP collective over `tp_moe_expert` has that degree. The one
    statement of it: which fabric a group's traffic crosses, which
    algorithms its fabric admits and how a routed exchange's destinations
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


class Candidate(NamedTuple):
    """One algorithm of a collective's envelope on one fabric: the seconds
    `price_collective` charges per payload byte and the fixed time over the
    step's instances (`collective_latency`), the per-rank wire bytes per
    payload byte over those instances (the sum of its phases' bytes), the
    link traversals ONE instance waits on, its stages in order
    (`(kind, label)`, `comm_volume.STAGE_KINDS`), and — where the fabric
    cannot run it — why not (`Fabric.refusal`).

    A refused candidate has no price: it never finishes (an infinite fixed
    time, `REFUSED`), so it is never the minimum of anything it sits in —
    the row law's `Min` included — and nothing borrows another algorithm's
    price for it."""
    algorithm: str
    s_per_byte: float
    latency_s: float
    wire_per_byte: float
    hops: int
    stages: Tuple[Tuple[str, str], ...] = ()
    refusal: Optional[str] = None

    @property
    def feasible(self) -> bool:
        return self.refusal is None

    def t(self, payload: float) -> float:
        """This algorithm's time over the step at a payload of `payload`
        bytes per instance; infinite where the fabric cannot run it."""
        if not self.feasible:
            return math.inf
        return float(payload) * self.s_per_byte + self.latency_s

    @classmethod
    def refused(cls, algorithm: str, reason: str) -> "Candidate":
        """`algorithm`, which the fabric cannot run, for `reason`."""
        return cls(algorithm, 0.0, math.inf, 0.0, 0,
                   ALGORITHMS[algorithm].stages, reason)


#: The payloads a candidate's curve is sampled at (`Envelope.choice`): 25
#: points evenly spaced in log from 1 KiB to 1 GiB.
CURVE_PAYLOADS: Tuple[int, ...] = tuple(
    round(1024 * 2 ** (20 * i / 24)) for i in range(25))


class Envelope(NamedTuple):
    """A collective's ENVELOPE on one fabric: every algorithm that runs its
    op (`comm_volume.algorithms_for`, in that order), each priced on the
    fabric (`Candidate`). Its time at a payload is the minimum over them;
    ties go to the earlier. A candidate the fabric cannot run is listed,
    refused, and never finishes, so the minimum is over the feasible ones —
    NCCL's schedule always among them.

    One shape on every card (the row law's `Min` names every candidate,
    `law_values`), and `choose` is that minimum's argument: everything a
    row publishes about its collective — its time, its wire bytes, its
    per-phase rows, the `comm_choice` column — reads that one choice. The
    envelope carries the collective and the fabric it was priced for, so a
    reader cannot hand it a card it was not built on."""
    sig: "CollectiveSignature"
    fabric: "Fabric"
    candidates: Tuple[Candidate, ...]

    def choose(self, payload: float) -> Candidate:
        """The fastest feasible candidate at `payload` bytes per instance."""
        return min((c for c in self.candidates if c.feasible),
                   key=lambda c: c.t(payload))

    def law_values(self) -> Tuple[Tuple[float, float], ...]:
        """Each candidate's (seconds per byte, fixed time), in order: what
        the row law's `Min` reads. A refused one's fixed time is infinite."""
        return tuple((c.s_per_byte, c.latency_s) for c in self.candidates)

    def choice(self, payload: float) -> Dict[str, Any]:
        """The `comm_choice` a row publishes at `payload` (bytes per
        instance): the collective, the fabric facts that decided which
        algorithms were feasible, the one chosen, the envelope itself on the
        curve grid, and every candidate with its time and wire bytes over
        the row's instances (null where refused, with the reason), its
        curve — the same law over `CURVE_PAYLOADS` — and what it is
        (`description`, `stages`, `note`, declared beside the algorithm)."""
        sig, fabric = self.sig, self.fabric
        n = int(sig.world)
        stride = max(int(sig.stride or 1), 1)
        inside = members_per_node(n, fabric.gpus_per_node, stride)
        spans = -(-n // inside)
        chosen = self.choose(payload)
        return {
            "op": sig.op, "world": n, "count": int(sig.count),
            # The group that holds rank 0: its members `stride` apart on the
            # mesh (`members_per_node`).
            "ranks": [k * stride for k in range(n)],
            "payload_bytes": float(payload),
            "fabric": {
                "link": fabric.inter_label if spans > 1 else fabric.intra_label,
                "nvls": bool(fabric.nvls),
                "gpus_per_node": int(fabric.gpus_per_node or 1),
                "nodes_spanned": int(spans),
            },
            "algorithm": chosen.algorithm,
            "envelope_curve": [[p, c.t(p), c.algorithm]
                               for p in CURVE_PAYLOADS
                               for c in (self.choose(p),)],
            "candidates": [{
                "algorithm": c.algorithm,
                "description": ALGORITHMS[c.algorithm].description,
                "note": ALGORITHMS[c.algorithm].note,
                "stages": [{"kind": kind, "label": label}
                           for kind, label in c.stages],
                "feasible": c.feasible,
                "reason": c.refusal,
                "t_s": c.t(payload) if c.feasible else None,
                "wire_bytes_per_rank": (c.wire_per_byte * float(payload)
                                        if c.feasible else None),
                "hops": c.hops if c.feasible else None,
                "curve": ([[p, c.t(p)] for p in CURVE_PAYLOADS]
                          if c.feasible else []),
            } for c in self.candidates],
        }


class CollectiveSignature(NamedTuple):
    """One collective as everything but the card: what its envelope
    (`Envelope`) reads besides the fabric.

    A row's collective is priced in more than one place — the walk's own
    row, a re-read of that row on another card or at another batch
    (`row_roofline.CardForm`), the step's law in the browser
    (`sol_law`) — and all of them have to name the SAME collective and
    price it the same way. So the facts are a type, and its notations are
    its projections: `key` (the canonical JSON a cached form and a law
    record carry), `symbol_suffix` (the name a step's law prices this
    collective under — two collectives that differ in any fact here must
    differ in that name, or one's price is read for the other) and
    `envelope` (the pricing itself).

    `route_fanout` is a routed MoE exchange's top-k fan-out, which prices
    its dedup (`comm_volume.RouteFanout`). It is part of the signature and
    not of the card because two expert pools of different top-k in ONE step
    — a speculative draft's 128-choose-3 pool beside the main 384-choose-6
    — are two prices, and a symbol naming only `(op, world, count, stride)`
    would hand one of them the other's.
    """
    op: str
    world: int
    count: int
    stride: int
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
                   route_fanout=(RouteFanout.from_dict(fanout)
                                 if fanout else None))

    @property
    def key(self) -> str:
        """These facts as canonical JSON: what a cached card-independent
        form carries instead of the row (`CardForm.sigs`), and what a law
        record names its collectives by (`sol_law`). A collective with no
        fan-out states none."""
        facts: Dict[str, Any] = {
            "op": self.op, "world": self.world, "count": self.count,
            "stride": self.stride}
        if self.route_fanout is not None:
            facts["route_fanout"] = self.route_fanout.as_dict()
        return json.dumps(facts, sort_keys=True)

    @classmethod
    def from_key(cls, key: str) -> "CollectiveSignature":
        d = json.loads(key)
        fanout = d.get("route_fanout")
        return cls(op=str(d["op"]), world=int(d["world"]),
                   count=int(d["count"]), stride=int(d["stride"]),
                   route_fanout=(RouteFanout.from_dict(fanout)
                                 if fanout else None))

    @property
    def symbol_suffix(self) -> str:
        """What a step's law appends to a collective card symbol's name
        (`row_roofline.collective_symbol`, `sol_law.cardBinding`) so that
        rows of different collectives add: one name per distinct
        signature."""
        out = f"_{self.op}_{self.world}_{self.count}_{self.stride}"
        if self.route_fanout is not None:
            out += self.route_fanout.symbol_suffix
        return out

    def envelope(self, fabric: "Fabric") -> Envelope:
        """This collective's envelope on `fabric` (`collective_envelope`)."""
        return collective_envelope(self, fabric)


@memoize(maxsize=8192)
def collective_envelope(sig: CollectiveSignature, fabric: Fabric) -> Envelope:
    """The envelope of the collective `sig` on `fabric`: each algorithm that
    runs its op priced there — per payload byte on the fabric with free
    steps, its fixed time by its own phases' hops, each at its fabric's hop
    latency (`collective_latency`, `Fabric.hop_latency`) —
    and refused where the fabric cannot run it (`Fabric.refusal`). One per
    (collective, card): every point of a walk shares its collectives."""
    out = []
    for algorithm in algorithms_for(sig.op):
        refusal = fabric.refusal(algorithm, op=sig.op, world=sig.world,
                                 base_stride=sig.stride)
        if refusal is not None:
            out.append(Candidate.refused(algorithm, refusal))
            continue
        per_byte, rows = price_collective(
            op=sig.op, world=sig.world, tensor_bytes=1.0,
            count_per_step=sig.count, fabric=fabric.without_latency(),
            base_stride=sig.stride, algorithm=algorithm,
            fanout=sig.route_fanout)
        out.append(Candidate(
            algorithm, per_byte,
            collective_latency(
                op=sig.op, world=sig.world, count_per_step=sig.count,
                fabric=fabric, base_stride=sig.stride, algorithm=algorithm,
                fanout=sig.route_fanout),
            sum(float(r["bytes_per_rank"]) for r in rows),
            sum(int(r["hops"]) for r in rows),
            ALGORITHMS[algorithm].stages or tuple(map(_phase_stage, rows))))
    return Envelope(sig, fabric, tuple(out))


def _phase_stage(phase: Mapping[str, Any]) -> Tuple[str, str]:
    """One phase of NCCL's schedule (`collective_phases`) as a stage: a hop
    between nodes, or inside one node a reduce-scatter, an all-gather, a
    direct exchange (an all-to-all, a point-to-point send) or a ring."""
    name = str(phase["name"])
    label = (f"{name.replace('_', ' ')} over {phase['fabric_label']}, "
             f"{phase['world']} ranks")
    if phase["fabric"] == "inter":
        return "inter_node", label
    if name in STAGE_KINDS:
        return name, label
    if collective_schedule(name) == "point_to_point":
        return "all_to_all", label
    return "ring", label


def collective_latency(*, op: str, world: int, count_per_step: int,
                       fabric: Fabric, base_stride: int,
                       algorithm: str = NCCL,
                       fanout: Optional[RouteFanout] = None) -> float:
    """The fixed part of a collective's time over a step: each phase of its
    schedule (`collective_phases`, run by `algorithm`) pays its fabric's
    `hop_latency` per link traversal it waits on, for every instance in the
    step. Independent of the payload once the algorithm is chosen."""
    phases = collective_phases(
        op=op, world=int(world), tensor_bytes=1.0,
        gpus_per_node=fabric.scaleup_ranks(),
        base_stride=int(base_stride or 1), algorithm=algorithm, fanout=fanout)
    return sum(fabric.hop_latency(ph["fabric"]) * ph["hops"]
               for ph in phases) * int(count_per_step)


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
    `algorithm` (`Envelope.choose` chooses it for a payload).

    The anchor row in `phases` and the Gallery's re-stamp of the same
    layout on another card both call this, so a second card is priced by
    the same loop as the first rather than by a scalar ratio of link
    bandwidths applied to the first card's total.
    """
    # Split at the scale-up domain, not the chassis (`Fabric.scaleup_ranks`).
    # Cards declaring no supernode have the two equal, so their phases --
    # and every number on the GPU baseline -- are unchanged.
    phases_list = collective_phases(
        op=op, world=int(world), tensor_bytes=float(tensor_bytes),
        gpus_per_node=fabric.scaleup_ranks(),
        base_stride=int(base_stride or 1), algorithm=algorithm,
        fanout=fanout, t_sym=t_sym, t_sub=t_sub,
    )
    t_comm = 0.0
    rows: List[Dict[str, Any]] = []
    for ph in phases_list:
        where = ph["fabric"]
        fabric_bw = fabric.bandwidth(where)
        fabric_label = (fabric.intra_label if where == "intra"
                        else fabric.inter_label)
        phase_bytes = float(ph["bytes_per_rank"]) * int(count_per_step)
        hop_latency = fabric.hop_latency(where)
        phase_latency = hop_latency * ph["hops"] * int(count_per_step)
        phase_t = (phase_bytes / fabric_bw if fabric_bw else 0.0
                   ) + phase_latency
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
            "hop_latency_s": hop_latency,
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
    refused for a group that does not (`Fabric.refusal` never offers it).
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
    # The one owner of this division; `comm_reach` reads it too, so
    # the price and the lookup cannot disagree about where a group sits.
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

