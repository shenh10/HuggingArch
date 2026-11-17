"""Price arrived vs required EdgeState as named collectives."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from backend.arch import dims
from backend.arch.eval.dim import floored

from .attention_plan import added_cut, cut_swap
from .edge import EdgeState
from .groups import cross_replica_family, same
from .layout import (
    AxisLayout, Block, Product, Replicated, Segmented, UniformFiberError,
    axis_fully_replicated, physical_local_extent, sharded,
)
from .value import ONLINE_ATTENTION, SUM


def lower_diff(rule: Any, arrived: EdgeState, required: EdgeState, *,
               node: Any = None, input_index: Optional[int] = None,
               ) -> Tuple[Tuple[str, str], ...]:
    """The collectives one edge's state change needs.

    A resolved pending sum is an `all_reduce` (or `reduce_scatter` when
    the same step also shards that group). A shard gathered whole is an
    `all_gather`. Replicated → sharded is a local slice priced at
    nothing. Anything else is a hard error, not a silent zero.
    """
    if not isinstance(arrived, EdgeState) or not isinstance(required,
                                                            EdgeState):
        raise ValueError("lower_diff speaks EdgeState only")
    if len(arrived.layout) != len(required.layout):
        raise ValueError(
            f"lower_diff across ranks {len(arrived.layout)} != "
            f"{len(required.layout)}")

    exclusive = reduce_scatter(arrived, required)
    if exclusive is not None:
        return exclusive

    out: List[Tuple[str, str]] = []
    out.extend(finalize_value(arrived, required))
    if arrived.layout == required.layout:
        return tuple(out)

    swap = cut_swap(arrived.layout, required.layout)
    if swap is not None:
        gained_axis, gained, lost_axis, lost = swap
        if rule.attention.is_ulysses_entry(
                gained, lost, gained_axis, lost_axis,
                node, input_index):
            out.append((gained.group.id, "all_to_all"))
            return tuple(out)
        raise ValueError(
            f"{getattr(node, 'display_path', '<edge>')} input "
            f"{input_index}: axis cut migration {lost!r} -> "
            f"{gained!r} is not a lowering row: one all_to_all exists "
            "only for the ulysses plan seam moving an identical "
            "resident-sequence cut between axes")

    for got, need in zip(arrived.layout, required.layout, strict=True):
        if got == need:
            continue
        collective = axis_collective(rule, got, need, arrived)
        if collective is not None:
            out.append(collective)
    return tuple(out)

def reduce_scatter(
        arrived: EdgeState, required: EdgeState
        ) -> Optional[Tuple[Tuple[str, str], ...]]:
    """SUM finalize that is also the unique Replicated→Block of that group."""
    if arrived.value is None or required.value is not None:
        return None
    if arrived.value.reducer != SUM:
        return None
    changed = [
        (got, need)
        for got, need in zip(
            arrived.layout, required.layout, strict=True
        )
        if got != need
    ]
    if len(changed) != 1:
        return None
    got, need = changed[0]
    if not (isinstance(got, Replicated) and isinstance(need, Block)
            and need.is_sharded_on(arrived.value.group)):
        return None
    return ((arrived.value.group.id, "reduce_scatter"),)

def finalize_value(
        arrived: EdgeState, required: EdgeState
        ) -> List[Tuple[str, str]]:
    """Collectives that resolve a pending reduction on the arriving layout.

    Layout repair, if any, is a second ordered step after this.
    """
    if arrived.value is not None and required.value is None:
        if arrived.value.reducer == SUM:
            return [(arrived.value.group.id, "all_reduce")]
        if arrived.value.reducer == ONLINE_ATTENTION:
            group = arrived.value.group.id
            return [(group, "all_gather_lse"), (group, "all_reduce")]
        raise ValueError(
            f"no lowering for reducer {arrived.value.reducer!r}")
    if arrived.value != required.value:
        raise ValueError(
            f"no lowering from value {arrived.value!r} to "
            f"{required.value!r} — communication cannot create a "
            "pending reduction")
    return []

def axis_collective(rule: Any, got: AxisLayout, need: AxisLayout, arrived: EdgeState
        ) -> Optional[Tuple[str, str]]:
    """One axis: a gather, a free local slice, or refuse."""
    if isinstance(need, Replicated) and isinstance(got, Block) \
            and got.replication_degree == 1:
        world = got.group.id
        expert_tp = rule.families.get("tp_moe_expert")
        if (arrived.workload is not None
                and expert_tp is not None
                and same(got.group, expert_tp)):
            world = expert_tp.id
        return (world, "all_gather")
    if isinstance(need, Replicated) and isinstance(got, Block) \
            and got.replication_degree > 1:
        # Each shard is already held by one replica cell. Gathering the
        # matching members of every cell hands every rank the whole axis;
        # that derived cross-replica group is the collective's world.
        return (cross_replica_family(
            got.group, got.replica_cells).id, "all_gather")
    if isinstance(need, Replicated) and isinstance(got, Product):
        expert_tp = rule.families.get("tp_moe_expert")
        if (arrived.workload is not None and expert_tp is not None
                and all(cut.group.ranks() <= expert_tp.ranks()
                        for cut in got.cuts)
                and frozenset().union(*(cut.group.ranks()
                                        for cut in got.cuts))
                    == expert_tp.ranks()):
            return (expert_tp.id, "all_gather")
    if isinstance(need, Replicated) and isinstance(got, Segmented):
        return segmented_gather(got)
    # Every rank already holds every position — a Replicated axis, or a
    # concat's Segmented record whose every segment is replicated — so a cut
    # is a free local slice.
    if axis_fully_replicated(got) and isinstance(need, Block):
        return None
    if added_cut(got, need) is not None:
        return None
    raise ValueError(
        f"no lowering for axis {got!r} -> {need!r}")


def free_layout_repair(rule: Any, arrived: EdgeState,
                       required: EdgeState) -> bool:
    """Whether every state change after a collective is free local work.

    This is the same rule `lower_diff` uses when it omits a second
    collective. The materializer asks it before attaching a local-slice post
    transform, so a shape mismatch can never be guessed to be a slice: the
    layouts must say that every rank already holds the required positions.
    """
    if arrived.layout == required.layout:
        return False
    for got, need in zip(arrived.layout, required.layout, strict=True):
        if got == need:
            continue
        try:
            if axis_collective(rule, got, need, arrived) is not None:
                return False
        except ValueError:
            return False
    return True

def segmented_gather(got: Segmented) -> Optional[Tuple[str, str]]:
    """One axis, one gather: every cut segment must share a rank group.

    Downstream prices a collective from the whole producer tensor, so a
    row per segment would bill that tensor once per segment. Replicated
    segments are already home. Replica-cell gathers are unpriced and
    refuse rather than mislabel.
    """
    cut_groups = []
    for _offset, _extent, child in got.segments:
        if isinstance(child, Replicated):
            continue
        if isinstance(child, Block) and child.replication_degree == 1:
            cut_groups.append(child.group)
        else:
            raise ValueError(
                f"no lowering for segment {child!r} -> replicated "
                "(cross-replica gathers are unpriced)")
    if not cut_groups:
        return None
    first = cut_groups[0]
    if any(not same(group, first) for group in cut_groups[1:]):
        raise ValueError(
            f"segments of {got!r} gather over different rank groups "
            "— no single whole-axis collective exists")
    return (first.id, "all_gather")

def cache_write_endpoint(rule: Any, node: Any, arrived: EdgeState, shape: Any
                         ) -> Tuple[EdgeState,
                                    Tuple[Tuple[str, str], ...]]:
    """The prefill producer checked against `s_kv -> Block(dcp)`.

    The producer names this axis by the query length (`dims.query_axes`);
    the persistent store names the same positions `s_kv`. Every other axis
    retains the producer's owner
    map. Matching ownership under differently named families is already a
    closed endpoint and keeps the arriving constructor as the canonical
    state.
    """
    if arrived.workload is not None:
        raise ValueError(
            f"{node.display_path}: a routed tensor cannot be a persistent "
            "cache endpoint")
    position = shape.index_of_any(dims.query_axes())
    if position is None or position >= len(arrived.layout):
        raise ValueError(
            f"{node.display_path}: cache endpoint shape {shape.out!r} "
            "has no sequence axis")

    extent = floored(shape.out[position])
    # Striped cache owners hold ceil(tokens/world) physical slots. SGLang
    # pads each owner's prefill gather then trims it back to logical rows;
    # this storage boundary must not require an exactly divisible prompt.
    cache_pad = 1 if rule.sparse_dcp.owns_cache_node(node) else None
    target_axis: AxisLayout = (
        sharded(rule.cache_family, pad_to=cache_pad)
        if rule.cache_family is not None else rule.replicated())
    physical_local_extent(target_axis, extent)
    got = arrived.layout[position]
    if same_block_ownership(got, target_axis):
        target_axis = got
    layout = list(arrived.layout)
    layout[position] = target_axis
    try:
        required = EdgeState(tuple(layout), None)
    except UniformFiberError as exc:
        raise ValueError(
            f"{node.display_path}: dcp crosses KV head shards; "
            "decode context parallelism may only split replica "
            "cells left by KV-head placement") from exc
    return required, lower_cache_write(rule, 
        arrived, required, position)

def same_block_ownership(left: AxisLayout,
                          right: AxisLayout) -> bool:
    if left == right:
        return True
    return (isinstance(left, Block) and isinstance(right, Block)
            and same(left.group, right.group)
            and left.replica_cells == right.replica_cells
            and left.pad_to == right.pad_to)

def lower_cache_write(rule: Any, arrived: EdgeState, required: EdgeState, position: int
        ) -> Tuple[Tuple[str, str], ...]:
    """Lower a persistent write without fabricating a flow consumer."""
    out: List[Tuple[str, str]] = []
    settled = arrived
    if arrived.value is not None:
        settled = arrived.finalized()
        out.extend(lower_diff(rule, arrived, settled))
    if settled.layout == required.layout:
        return tuple(out)
    changed = [i for i, (got, need) in enumerate(
        zip(settled.layout, required.layout, strict=True)) if got != need]
    if changed != [position]:
        raise ValueError(
            f"{required!r}: a cache write may change only its position "
            f"axis, got axes {changed}")
    got = settled.layout[position]
    need = required.layout[position]
    if isinstance(got, Replicated) and isinstance(need, Block):
        return tuple(out)       # local owner slice
    if isinstance(got, Block) and isinstance(need, Replicated):
        if got.replication_degree != 1:
            raise ValueError(
                f"cache ownership gather from {got!r} replica cells "
                "needs the cross-replica lowering row")
        out.append((got.group.id, "all_gather"))
        return tuple(out)
    if isinstance(got, Block) and isinstance(need, Block):
        if got.replication_degree != 1:
            raise ValueError(
                "cache re-shard from replica cells needs the "
                "cross-replica lowering row")
        # Gather the producer partition, then take the destination's
        # local slice. The second step has no wire traffic.
        out.append((got.group.id, "all_gather"))
        return tuple(out)
    raise ValueError(
        f"no persistent cache lowering for {got!r} -> {need!r}")
