"""Region-aware device mesh — the rank LAYOUT of a deployment.

Single source of truth for how parallel axes nest on the physical device
grid. Two consumers read it instead of each re-encoding the layout:

  * `ParallelConfig.validate` / `per_stage_size` — the per-region group
    size (∏ axis sizes) that must fill a pipeline stage.
  * collective materialization — each collective's `base_stride`
    (∏ sizes of axes nested INSIDE it on the mesh), which the comm-cost
    layer uses to tell whether the group is node-local (intra fabric) or
    spread one-member-per-node (inter fabric).

Keeping the layout in ONE place is the point: previously the nesting was
encoded twice (validate's group products + the graft sites' literal
`tp_attention` / `tp_moe_expert` strides) and nothing forced them to
agree — change the assumed nesting in one and the other silently
disagrees (the classic two-consumers-of-one-fact bug).

This is the cost-model analogue of a DTensor / GSPMD DeviceMesh. One
difference this tool needs that a single uniform mesh can't express: TP
may differ per region (`tp_attention` ≠ `tp_dense_ffn` ≠
`tp_moe_expert`), so the layout is a set of per-region sub-meshes rather
than one global grid.

Layout convention (INNER → OUTER), matching Megatron / vLLM defaults —
TP innermost (per-layer collectives, highest bandwidth), then the
region's token/outer sharding (CP for attention AND dense — see below —
EP for MoE), then DP within the stage; PP (splits layers across stages)
and full-model DP replicas are global axes outer to every region:

  attention : tp_attention → cp → dp_attention
  dense     : tp_dense_ffn → cp → dp_attention
  moe       : tp_moe_expert → ep_moe
  (global)  : … → pp → dp_replicas

Why `cp` is a DENSE-region axis too: cp shards the SEQUENCE, not any
weight. The FFN is token-local, so under cp each rank processes its
s_ctx/cp tokens with a REPLICATED weight — cp spans the dense ranks exactly
like attention but never enters the FFN weight divisor (that stays
tp_dense_ffn). Omitting cp here forced `tp_dense_ffn` to absorb the cp
factor to fill the stage, which over-divided the FFN weight by cp.

The MoE region deliberately has NO cp axis: there, the expert
all-to-all routing redistributes the cp-sharded tokens across the `ep`
group, so `ep` already spans the stage's ranks. On the same physical
ranks cp (attention/dense) and ep (MoE) ALIAS — e.g. an 8-GPU Kimi
deployment is cp=8 in attention and ep=8 in MoE over the same 8 ranks —
so adding cp to the MoE product would double-count it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional

if TYPE_CHECKING:
    from .parallel_config import ParallelConfig

# Per-region parallel axes, INNER → OUTER. Names are ParallelConfig
# fields. This table is the single source for both the group-size product
# (validate / per_stage_size) and the per-collective base_stride.
REGION_AXES = {
    "attention": ("tp_attention", "cp", "dp_attention"),
    # `cp` is a TOKEN axis here (shards the sequence, replicates the FFN
    # weight) — it spans the dense ranks but never divides the weight.
    "dense":     ("tp_dense_ffn", "cp", "dp_attention"),
    # No `cp`: the expert all-to-all already redistributes cp-sharded
    # tokens across `ep`, which aliases cp on the same physical ranks.
    "moe":       ("tp_moe_expert", "ep_moe"),
}

# Global axes outer to every region, INNER → OUTER. `pp` splits layers
# across pipeline stages; `dp_replicas` replicates the whole model. A
# pipeline cut's point-to-point send runs on the `pp` world, whose members
# sit `per_stage_size` ranks apart (`resolver.pipeline_family_for`), and is
# priced on that spacing like any region collective. DP replicas are
# independent and exchange nothing.
GLOBAL_OUTER_AXES = ("pp", "dp_replicas")

# Subgroup worlds: a world that PARTITIONS ranks a region has already
# counted, so it appears in no region product above. Value is the region it
# partitions and the order in which it FILLS that region's axes, outer-first.
#
# `dcp` fills `cp` before `tp_attention` because vLLM builds the groups that
# way: ranks are laid out `... x PCP x TP` with TP innermost, and
# `parallel_state.py` forms DCP groups as
# `dcp_ranks.transpose(-1, -2).reshape(-1, dcp_size)` — the transpose is what
# makes consecutive members of a DCP group differ in their PCP index, not
# their TP index. Which is also why the legal set is `dcp in (1, pcp,
# tp * pcp)`: a group either fits inside the PCP column or swallows it whole
# and spills into TP.
SUBGROUP_AXES = {"dcp": ("attention", ("cp", "tp_attention"))}

# For an axis that spans MORE than one region, the region whose nesting spaces
# its collectives. Membership is not written here — `REGION_AXES` already says
# which regions an axis is in, and a second table saying it again is a second
# thing to keep true. Only the tie-break is, and both TOKEN axes need one: each
# spans the attention and the dense region alike, and the collectives a state
# transition derives on them — the Ulysses all_to_all pair, the dispatch and
# combine of the MoE seam — are spaced by the attention nesting, because that
# is where the frameworks form the group (vLLM builds both with TP innermost of
# the attention sub-mesh).
_MULTI_REGION_AXIS = {"cp": "attention", "dp_attention": "attention"}


def _sz(pc: "ParallelConfig", axis: str) -> int:
    return int(getattr(pc, axis))


def region_of(axis: str) -> Optional[str]:
    """The region whose nesting spaces a collective on `axis`, or None for an
    axis no region contains (a subgroup world — see `subgroup_stride`)."""
    regions = sorted(r for r, axes in REGION_AXES.items() if axis in axes)
    if not regions:
        return None
    if len(regions) == 1:
        return regions[0]
    region = _MULTI_REGION_AXIS.get(axis)
    if region is None:
        raise ValueError(
            f"axis {axis!r} spans regions {regions} and no spacing is "
            f"declared for it in _MULTI_REGION_AXIS")
    return region


def axis_degrees(pc: "ParallelConfig") -> dict:
    """`{axis: degree}` for every axis the mesh lays out — the argument
    `spans` wants, in `ParallelConfig`'s vocabulary."""
    return {axis: _sz(pc, axis)
            for axes in REGION_AXES.values() for axis in axes}


def spans(degrees: Mapping[str, int]) -> dict:
    """`{axis: (stride, size)}` — where each axis's group starts on
    `REGION_AXES` inner→outer, and how far it reaches. Only axes above 1 are
    listed; an axis of 1 spans one rank and answers nothing.

    This is the only thing the mesh says about a world's ranks: the resolver
    (`resolver.families_for`) turns each span into an explicit
    `ProcessGroupFamily`, and every question of how two worlds relate — the
    same ranks, one group inside another, interleaved — is asked of those
    members (`groups.same` / `refines` / `orthogonal` / `overlaps`, and the
    route seam's `LayoutPropagation._axis_group_invariant`), never of the
    stride arithmetic here. Membership is what prices a token cut made of
    TWO worlds at two strides (`cp` and `dp_attention` both above 1 under
    `ep_moe`): the dispatch joins the token owner cells with the route
    groups (`LayoutPropagation.routed_rows_layout`) — what varies inside a
    route group is absorbed, which is the all_to_all exchange, and what
    varies across route groups survives as a row cut. A
    stride/size equivalence over spans could only compare one span with one,
    and read the two-world cut as replicated."""
    out: dict = {}
    for region in sorted(REGION_AXES):
        stride = 1
        for axis in REGION_AXES[region]:
            size = int(degrees.get(axis, 1) or 1)
            if size > 1 and region_of(axis) == region:
                out[axis] = (stride, size)
            stride *= size
    return out


def group_size(pc: "ParallelConfig", region: str) -> int:
    """Ranks spanned by one pipeline stage's `region` sub-mesh — the
    product of its axis sizes. All regions' group sizes must agree
    (== per_stage_size); `ParallelConfig.validate` enforces it."""
    n = 1
    for axis in REGION_AXES[region]:
        n *= _sz(pc, axis)
    return n


def base_stride(pc: "ParallelConfig", region: str, axis: str) -> int:
    """Ranks between consecutive members of a collective on `axis` within
    `region` — the product of the sizes of all axes nested INSIDE `axis`.

    The innermost axis (TP) has stride 1; an outer axis is spaced by the
    islands beneath it, so a CP collective is spaced by `tp_attention`
    and an EP collective by `tp_moe_expert`. The comm-cost layer turns
    this into members-per-node to pick the intra vs inter fabric.
    """
    axes = REGION_AXES[region]
    if axis not in axes:
        raise ValueError(f"axis {axis!r} not in region {region!r} ({axes})")
    n = 1
    for inner in axes[:axes.index(axis)]:
        n *= _sz(pc, inner)
    return n


def subgroup_stride(pc: "ParallelConfig", world: str) -> int:
    """Ranks between consecutive members of a collective on a SUBGROUP world.

    A subgroup has no place of its own in the region's nesting — it borrows
    the ranks of the axes it fills, in `SUBGROUP_AXES` order — so its spacing
    is the spacing of the LAST axis it reaches into. A `dcp` that fits inside
    the `cp` column is spaced like `cp` (one member per `tp_attention`
    island); one that swallows the column and spills into TP is spaced like
    TP, which is to say contiguous. That is the whole difference between a
    DCP combine that stays on NVLink and one that crosses the fabric, and it
    is not a function of the group's SIZE alone — hence a walk, not a
    constant.
    """
    region, order = SUBGROUP_AXES[world]
    degree = _sz(pc, world)
    stride, span = 1, 1
    for axis in order:
        stride = base_stride(pc, region, axis)
        span *= _sz(pc, axis)
        if degree <= span:
            break
    return stride


def subgroup_degrees(pc: "ParallelConfig", world: str) -> tuple:
    """Every degree `world` may run at on this mesh, ascending.

    A subgroup is a mesh axis or it is nothing, so its members have to be
    evenly spaced, and the `SUBGROUP_AXES` fill order is what makes them so:
    a group spans whole columns, outermost-first. The one place a PARTIAL
    span is also legal is the innermost axis, whose ranks are adjacent — any
    divisor of it names a contiguous group. A partial outer column would be
    evenly spaced too, but vLLM declines to build one (`config/parallel.py`:
    `dcp in (1, pcp, tp * pcp)`, "DCP must be disabled, span the PCP axis, or
    span the full TP x PCP axis"), and a set this tool accepts that vLLM
    refuses is a deployment nobody can run — so the innermost axis is where
    the divisors are, not a general rule about columns.
    """
    _region, order = SUBGROUP_AXES[world]
    legal, span = {1}, 1
    for i, axis in enumerate(order):
        n = _sz(pc, axis)
        if span == 1 and i == len(order) - 1:
            legal.update(d for d in range(1, n + 1) if n % d == 0)
        span *= n
        legal.add(span)
    return tuple(sorted(legal))
