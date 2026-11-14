"""Collective rows: all_reduce, reduce_scatter, all_to_all, all_gather, endpoints, DCP."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from typing import Any, List, Optional, Tuple

from backend.arch.eval.dim import value_of
from backend.arch.eval.traffic import payload_width
from backend.arch.eval.transition import RedistributeSite, Redistribute

from .groups import cross_replica_family, same
from .layout import Block, Replicated
from .lower import free_layout_repair
from .materialize_flow import (
    _output_axes,
    axis_of_kind,
    local_shape,
    model_output_flow_shape,
    rewrite_flow_shape,
    routed_flow_shape,
)
from .materialize_comm import (
    FlowRef,
    LogicalBindingGap,
    PayloadRole,
    PhysicalCollective,
    Placement,
    PostTransform,
    RewriteTarget,
    flow_port,
)
from .value import ONLINE_ATTENTION, SUM

_LSE_BYTES = 4.0


def reduction_row(mat: Any, record: Redistribute, world: str) -> PhysicalCollective:
    """A finalized Partial(sum): all_reduce, anchored at the
    contraction/join producer declared by propagation, moving the
    full-shaped partial."""
    value = record.arrived.value
    if value is None or value.reducer != SUM:
        raise ValueError(
            f"{mat._node(record.node).display_path}: all_reduce "
            f"record without a pending sum ({value!r})"
        )
    producer = record.producer if record.producer >= 0 else record.node
    port = flow_port(record.producer_port)
    routed = _routed_flow(mat, record, world, producer, port)
    if routed is not None:
        flow_in = flow_out = routed
    else:
        flow_in = local_shape(mat, producer, record.arrived, port)
        flow_out = local_shape(mat, producer, record.required, port)
    provenance = mat._partial_provenance(record, value)
    dedup_key = None
    if record.site in (
        RedistributeSite.EDGE,
        RedistributeSite.INTERNAL_IN,
    ):
        dedup_key = (
            record.producer,
            port,
            record.arrived,
            record.required,
            tuple(record.collectives),
            world,
            "all_reduce",
        )
    # A free repair after the collective is a local slice only when the
    # layouts say every rank already holds the required positions. Shape
    # equality/mismatch alone cannot classify the transition.
    post_transform = (
        PostTransform.LOCAL_SLICE
        if free_layout_repair(mat.analysis.rule, record.arrived, record.required)
        else PostTransform.NONE)
    if (post_transform is PostTransform.NONE
            and record.arrived.layout != record.required.layout):
        raise ValueError(
            f"{mat._node(record.node).display_path}: all_reduce followed by "
            f"an unsupported layout repair {record.arrived.layout!r} -> "
            f"{record.required.layout!r}")
    return mat._event(
        record=record,
        anchor_uid=provenance.producer,
        op="all_reduce",
        group_size=value.group.group_size,
        count_uid=record.node,
        world=world,
        flow_in_shape=flow_in,
        collective_in_shape=flow_in,
        collective_out_shape=flow_in,
        flow_out_shape=flow_out,
        payload_shape=flow_in,
        post_transform=post_transform,
        attribution_uid=producer,
        dedup_key=dedup_key,
        count=mat._value_count(record),
        family=value.group,
    )

def scatter_row(mat: Any, record: Redistribute, world: str) -> PhysicalCollective:
    """Finalize a sum while restoring a sequence-parallel row cut."""
    value = record.arrived.value
    if value is None or value.reducer != SUM:
        raise ValueError(
            f"{mat._node(record.node).display_path}: reduce_scatter "
            f"record without a pending sum ({value!r})"
        )
    producer = record.producer if record.producer >= 0 else record.node
    port = flow_port(record.producer_port)
    flow_in = local_shape(mat, producer, record.arrived, port)
    flow_out = local_shape(mat, producer, record.required, port)
    provenance = mat._partial_provenance(record, value)
    dedup_key = None
    if record.site in (
        RedistributeSite.EDGE,
        RedistributeSite.INTERNAL_IN,
    ):
        dedup_key = (
            record.producer,
            port,
            record.arrived,
            record.required,
            tuple(record.collectives),
            world,
            "reduce_scatter",
        )
    return mat._event(
        record=record,
        anchor_uid=provenance.producer,
        op="reduce_scatter",
        group_size=value.group.group_size,
        count_uid=record.node,
        world=world,
        flow_in_shape=flow_in,
        collective_in_shape=flow_in,
        collective_out_shape=flow_out,
        flow_out_shape=flow_out,
        payload_shape=flow_in,
        attribution_uid=producer,
        dedup_key=dedup_key,
        count=mat._value_count(record),
        family=value.group,
    )

def exchange_row(
        mat: Any, record: Redistribute, world: str, *,
        source_event: str | None = None,
) -> PhysicalCollective:
    """An exchange leg, anchored at the consumer, moving the per-rank tensor.

    A route exchange may consume an expert-tensor-parallel reduction of the
    same value. ``source_event`` names that ordered dependency; its expert
    buffer is locally packed into the exchange's row-major rows.
    """
    size = mat._family_size(world)
    port = flow_port(record.producer_port)
    route = mat.analysis.rule.route_family
    family = mat.analysis.rule.families.get(world)
    route_evidence = record.arrived.workload or record.required.workload
    is_route = (
        route is not None
        and family is not None
        and world == route.id
    )
    if record.site == RedistributeSite.INTERNAL_OUT:
        # The exit leg happens inside the node; its owning region
        # anchors it and the RESTORED (required) layout is the moved tensor.
        anchor = mat._consumer_of(record.node, port)
        producer = record.node
    else:
        anchor = record.node
        producer = record.producer
    if is_route:
        if route_evidence is None or not same(
            route_evidence.group, family
        ):
            raise ValueError(
                f"{mat._node(record.node).display_path}: route exchange "
                f"on {world!r} has no matching routed-load evidence"
            )
        evidence_state = (
            record.arrived
            if record.arrived.workload is not None
            else record.required
        )
        local_experts = (
            route_evidence.num_experts // family.group_size
        )
        route_flow = routed_flow_shape(mat, 
            producer, evidence_state, local_experts, port
        )
        flow_in = route_flow
        flow_out = route_flow
    else:
        flow_in = local_shape(mat, producer, record.arrived, port)
        flow_out = local_shape(mat, producer, record.required, port)
    meta = ()
    return mat._event(
        record=record,
        anchor_uid=anchor,
        op="all_to_all",
        group_size=size,
        count_uid=record.node,
        world=world,
        flow_in_shape=flow_in,
        collective_in_shape=flow_in,
        collective_out_shape=flow_out,
        flow_out_shape=flow_out,
        payload_shape=flow_in,
        meta=meta,
        attribution_uid=anchor,
        **({} if source_event is None else {
            "source": None,
            "source_event": source_event,
            "tree_anchor_event": source_event,
        }),
    )


def _routed_flow(mat: Any, record: Redistribute, world: str,
                 producer: int, port: str) -> Optional[List[Any]]:
    """The one packed routed-row buffer every expert-zone collective moves.

    Inside an expert-parallel routed zone, the EP exchange and an expert
    tensor-parallel reduction read the same `ExpectedRoutedLoad` record
    through `RoutedLaunch`; neither re-projects the routed tensor's
    expert-major shape. Returning None means this event is not on that routed
    zone and keeps the ordinary tensor projection.
    """
    family = mat.analysis.rule.families.get(world)
    workload = record.arrived.workload or record.required.workload
    if family is None or workload is None or workload.group is None:
        return None
    state = (record.arrived if record.arrived.workload is not None
             else record.required)
    node = mat.analysis.program.by_uid(producer)
    dispatch = mat.analysis.program.dispatch_of(node)
    local_experts = mat.analysis.routed[dispatch].local_experts
    return routed_flow_shape(mat, producer, state, local_experts, port)


def _cross_replica_family(record: Redistribute, world: str):
    """The derived family named by a replicated-cell all-gather."""
    axes = (*record.arrived.layout, *record.required.layout)
    for axis in axes:
        if (isinstance(axis, Block)
                and cross_replica_family(
                    axis.group, axis.replica_cells).id == world):
            return cross_replica_family(axis.group, axis.replica_cells)
    return None


def gather_row(mat: Any, record: Redistribute, world: str) -> PhysicalCollective:
    """A whole-axis materialization: anchored at the consumer."""
    producer = record.producer if record.producer >= 0 else record.node
    port = flow_port(record.producer_port)
    routed = _routed_flow(mat, record, world, producer, port)
    if routed is not None:
        flow_in = flow_out = routed
    else:
        flow_in = local_shape(mat, producer, record.arrived, port)
        flow_out = local_shape(mat, producer, record.required, port)
    dedup_key = None
    if record.site in (
        RedistributeSite.EDGE,
        RedistributeSite.INTERNAL_IN,
    ):
        dedup_key = (
            record.producer,
            port,
            record.arrived,
            record.required,
            tuple(record.collectives),
            world,
            "all_gather",
        )
    derived_family = _cross_replica_family(record, world)
    return mat._event(
        record=record,
        anchor_uid=record.node,
        op="all_gather",
        group_size=(derived_family.group_size if derived_family is not None
                    else mat._family_size(world)),
        count_uid=record.node,
        world=world,
        flow_in_shape=flow_in,
        collective_in_shape=flow_in,
        collective_out_shape=flow_out,
        flow_out_shape=flow_out,
        payload_shape=flow_out,
        attribution_uid=record.node,
        family=derived_family,
        dedup_key=dedup_key,
        count=mat._value_count(record),
    )

def output_width(mat: Any, uid: int, payload: Any) -> Optional[float]:
    """The element width a collective moving a node's output carries over
    its payload buffer `payload`, where it is declared rather than the
    deployment's activation width (DSA's int32 top-k, the fp32 indexer
    logits, a quantised record with its scale share), else None — the
    activation width. `traffic.payload_width`, the one rule for it."""
    program = mat.analysis.program
    return payload_width(program.by_uid(uid), program, mat.analysis.shapes,
                         payload)

def crossing_meta(mat: Any, record: Redistribute
                  ) -> Tuple[Tuple[str, Any], ...]:
    """A hop's row metadata: the cut it crosses (sent by stage `cut`), the
    occurrence that produced the value and the ones that read it past the
    cut, each as `(display path, iteration)`."""
    crossing = record.crossing

    def named(occurrence: Any) -> Tuple[str, Any]:
        return (mat._node(occurrence.uid).display_path, occurrence.iteration)

    uid, _port = crossing.producer
    return (("cut", crossing.cut),
            ("source_occurrence",
             (mat._node(uid).display_path, crossing.source_iteration)),
            ("sink_occurrences",
             tuple(named(sink) for sink, _index in crossing.sinks)))

def send_row(mat: Any, record: Redistribute, world: str) -> PhysicalCollective:
    """A value sent across a pipeline cut: one hop, in its own layout.

    Each rank of the sending stage hands its local tensor to its
    counterpart on the next stage (`pipeline_family_for`). Once per step,
    whatever the producer's layer count: a cut is crossed once. A stage
    whose rule distributes nothing holds the whole tensor. A cache-resident
    value sends what its readers read of the cache this step
    (`StageCrossing.cache_read`)."""
    family = mat.analysis.rule.pipeline_family
    port = flow_port(record.producer_port)
    cache_read = record.crossing.cache_read
    if cache_read is not None:
        flow, width = cache_read.payload()
    else:
        if record.arrived is None:
            axes = _output_axes(
                mat.analysis.shapes.values.get(record.producer), port)
            flow = list(axes)
        else:
            flow = local_shape(mat, record.producer, record.arrived, port)
        width = output_width(mat, record.producer, flow)
    return mat._event(
        record=record,
        anchor_uid=record.producer,
        op="send_recv",
        group_size=family.group_size,
        count_uid=record.producer,
        count=1,
        world=world,
        flow_in_shape=flow,
        collective_in_shape=flow,
        collective_out_shape=flow,
        flow_out_shape=flow,
        payload_shape=flow,
        payload_dtype_bytes=width,
        family=family,
        meta=crossing_meta(mat, record),
    )

def endpoint_rows(mat: Any, record: Redistribute, *, anchor_uid: int, meta: Tuple[Tuple[str, Any], ...] = ()
) -> List[PhysicalCollective]:
    out: List[PhysicalCollective] = []
    port = flow_port(record.producer_port)
    current = record.arrived
    current_flow, current_owners = endpoint_flow_shape(mat, 
        record, current, port
    )
    previous_event: Optional[str] = None
    for world, name in record.collectives:
        if name == "all_reduce":
            # A pending sum reduced where the walk hands the value on (an
            # encoder's output, which the language model reads whole).
            reduced = replace(current, value=None)
            event = reduction_row(
                mat, replace(record, arrived=current, required=reduced),
                world)
            out.append(event)
            previous_event = event.event_id
            current = reduced
            current_flow, current_owners = endpoint_flow_shape(
                mat, record, current, port)
            continue
        if name != "all_gather":
            raise ValueError(
                f"{mat._node(record.node).display_path}: endpoint "
                f"collective {name!r} needs an explicit state step"
            )
        flow_in = current_flow
        next_state, axis = endpoint_gather_state(mat, 
            record, current, world
        )
        projected_out, next_owners = endpoint_flow_shape(mat, 
            record, next_state, port
        )
        group_size = mat._family_size(world)
        collective_out = list(flow_in)
        collective_out[axis] *= group_size
        owner = dict(current_owners).get(axis)
        binding_gap = None
        post_transform = PostTransform.NONE
        if owner == "request_B":
            raise ValueError(
                f"{mat._node(record.node).display_path}: request-owned "
                f"axis {axis} must remain local at the model output"
            )
        if owner == "sampled_position":
            flow_out = list(collective_out)
            flow_out[axis] = projected_out[axis]
            post_transform = PostTransform.SAMPLED_POSITION_SELECT
        elif isinstance(next_state.layout[axis], Block):
            flow_out = projected_out
            post_transform = PostTransform.LOCAL_SLICE
        elif (isinstance(current.layout[axis], Block)
              and current.layout[axis].gathers_padding(
                  collective_out[axis], projected_out[axis])):
            # A padded cut (`fit: pad`, e.g. a vocab-parallel head whose
            # vocabulary does not divide by the world) gathers the padded
            # extent; the padding rows are then dropped locally, as vLLM's
            # LogitsProcessor._get_logits does after _gather_logits
            # (`logits = logits[..., : self.org_vocab_size]`).
            flow_out = list(collective_out)
            flow_out[axis] = projected_out[axis]
            post_transform = PostTransform.PADDING_TRIM
        else:
            flow_out = collective_out
            if flow_out[axis] != projected_out[axis]:
                raise ValueError(
                    f"{mat._node(record.node).display_path}: endpoint "
                    f"gather on {world!r} produces extent "
                    f"{flow_out[axis]}, not projected extent "
                    f"{projected_out[axis]} on axis {axis}"
                )
        for other_axis, projected in enumerate(projected_out):
            if other_axis == axis:
                continue
            if projected != flow_in[other_axis] and not any(
                gap_axis == other_axis and gap_owner == "request_B"
                for gap_axis, gap_owner in current_owners
            ):
                raise ValueError(
                    f"{mat._node(record.node).display_path}: endpoint "
                    f"gather on axis {axis} also changes axis {other_axis} "
                    f"from {flow_in[other_axis]} to {projected}"
                )
        event = mat._event(
            record=record,
            anchor_uid=anchor_uid,
            op=name,
            group_size=group_size,
            count_uid=record.node,
            world=world,
            flow_in_shape=flow_in,
            collective_in_shape=flow_in,
            collective_out_shape=collective_out,
            flow_out_shape=flow_out,
            payload_shape=collective_out,
            post_transform=post_transform,
            binding_gap=binding_gap,
            meta=(*meta, *((f"logical_owner:{logical_owner}", True)
                           for _, logical_owner in current_owners)),
            source=(FlowRef(record.producer, port)
                    if previous_event is None else None),
            source_event=previous_event,
            placement_uid=record.node,
            placement=Placement.AFTER,
            attribution_uid=anchor_uid,
            tree_anchor_event=previous_event,
        )
        out.append(event)
        previous_event = event.event_id
        current = next_state
        current_flow = flow_out
        current_owners = tuple(dict.fromkeys((
            *(
                (owner_axis, logical_owner)
                for owner_axis, logical_owner in current_owners
                if logical_owner == "request_B" and owner_axis != axis
            ),
            *(
                (owner_axis, logical_owner)
                for owner_axis, logical_owner in next_owners
                if owner_axis != axis
            ),
        )))
        if binding_gap is not None:
            current_owners = (*current_owners, (axis, owner))
    if current != record.required:
        raise ValueError(
            f"{mat._node(record.node).display_path}: endpoint chain "
            f"stopped at {current!r}, not required state {record.required!r}"
        )
    return out

def endpoint_flow_shape(mat: Any, record: Any, state: Any, port: str
) -> Tuple[List[int], Tuple[Tuple[int, str], ...]]:
    """Project one endpoint-chain state, including sampled-position B."""
    if record.site == RedistributeSite.MODEL_OUTPUT:
        # Sampled logits retain the sequence owner's routing evidence
        # after the output shape has rebound to one position. That cut is
        # not a uniform allocation of extent one; every other axis still
        # uses its ordinary physical projection.
        return model_output_flow_shape(mat, record.node, state, port)
    return local_shape(mat, record.node, state, port), ()

def endpoint_gather_state(mat: Any, record: Any, current: Any, world: str
) -> Tuple[Any, int]:
    """Apply one declared endpoint gather to exactly one physical cut."""
    family = mat.analysis.rule.families.get(world)
    if family is None:
        raise ValueError(f"endpoint world {world!r} resolved no family")
    candidates = [
        axis
        for axis, (got, need) in enumerate(
            zip(current.layout, record.required.layout, strict=True)
        )
        if got != need
        and isinstance(got, Block)
        and isinstance(need, (Replicated, Block))
        and same(got.group, family)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{mat._node(record.node).display_path}: endpoint gather "
            f"on {world!r} must settle exactly one matching cut, got "
            f"axes {candidates!r} from {current!r} to {record.required!r}"
        )
    layout = list(current.layout)
    axis = candidates[0]
    layout[axis] = record.required.layout[axis]
    return replace(current, layout=tuple(layout)), axis

def dcp_trio(mat: Any, record: Any) -> List[PhysicalCollective]:
    """Online-attention finalize: query all_gather, LSE all_gather, output RS.

    One semantic record; three physical rows. Head expansion lives in
    kernel geometry, recorded as a binding gap rather than guessed here.
    """
    from .sparse_dcp_comm import sparse_dcp_finalize

    sparse = sparse_dcp_finalize(mat, record)
    if sparse is not None:
        return sparse
    f = dcp_finalize_inputs(mat, record)
    query = mat._event(
        record=record,
        anchor_uid=f.score_uid,
        op="all_gather",
        group_size=f.dcp,
        count_uid=f.score_uid,
        world=f.world,
        flow_in_shape=f.query_local,
        collective_in_shape=f.query_local,
        collective_out_shape=f.query_collective_out,
        flow_out_shape=f.query_gathered,
        payload_shape=f.query_collective_out,
        post_transform=f.query_post,
        binding_gap=LogicalBindingGap.DCP_HEAD_FLOW,
        source=FlowRef(f.query_uid),
        rewrites=(f.query_target,),
        placement_uid=f.score_uid,
        placement=Placement.BEFORE,
        attribution_uid=f.score_uid,
        site=RedistributeSite.EDGE,
        family=f.value.group,
        dedup_key=(*f.finalize_key, "query"),
    )
    lse = mat._event(
        record=record,
        anchor_uid=f.exit_uid,
        op="all_gather",
        group_size=f.dcp,
        count_uid=f.apply_uid,
        world=f.world,
        flow_in_shape=f.combined_apply,
        collective_in_shape=f.lse_in,
        collective_out_shape=f.lse_out,
        flow_out_shape=f.combined_apply,
        payload_shape=f.lse_out,
        payload_role=PayloadRole.AUXILIARY_STATISTIC,
        payload_dtype_bytes=_LSE_BYTES,
        meta=(("lse_combine", f.dcp),),
        binding_gap=LogicalBindingGap.DCP_HEAD_FLOW,
        source=FlowRef(f.apply_uid),
        rewrites=(),
        placement_uid=f.exit_uid,
        placement=Placement.BEFORE,
        attribution_uid=f.exit_uid,
        site=RedistributeSite.INTERNAL_OUT,
        family=f.value.group,
        dedup_key=(*f.finalize_key, "lse"),
    )
    standard_scatter = (
        len(f.combined_apply) == len(f.exit_local)
        and sum(
            before != after
            for before, after in zip(
                f.combined_apply, f.exit_local, strict=True
            )
        ) == 1
        and prod(f.combined_apply) == prod(f.exit_local) * f.dcp
    )
    output_op = "reduce_scatter" if standard_scatter else "all_reduce"
    output_collective = f.exit_local if standard_scatter else f.combined_apply
    output_post = (
        PostTransform.NONE
        if standard_scatter else PostTransform.LOCAL_SLICE
    )
    scattered = mat._event(
        record=record,
        anchor_uid=f.exit_uid,
        op=output_op,
        group_size=f.dcp,
        count_uid=f.apply_uid,
        world=f.world,
        flow_in_shape=f.combined_apply,
        collective_in_shape=f.combined_apply,
        collective_out_shape=output_collective,
        flow_out_shape=f.exit_local,
        payload_shape=f.combined_apply,
        post_transform=output_post,
        binding_gap=LogicalBindingGap.DCP_HEAD_FLOW,
        source=None,
        source_event=lse.event_id,
        rewrites=f.exit_targets,
        placement_uid=f.exit_uid,
        placement=Placement.BEFORE,
        attribution_uid=f.exit_uid,
        tree_anchor_event=lse.event_id,
        site=RedistributeSite.INTERNAL_OUT,
        family=f.value.group,
        dedup_key=(*f.finalize_key, "output"),
    )
    return [query, lse, scattered]

@dataclass(frozen=True)
class DcpFinalizeInputs:
    """Everything `dcp_trio` needs to emit its three physical collectives —
    located and validated by `dcp_finalize_inputs`. Named fields, not an
    18-slot positional tuple: the emitters below read what they name."""

    value: Any
    world: Any
    dcp: Any
    apply_uid: Any
    score_uid: Any
    query_uid: Any
    query_target: Any
    query_local: Any
    query_gathered: Any
    query_post: Any
    query_collective_out: Any
    exit_uid: Any
    combined_apply: Any
    exit_local: Any
    exit_targets: Any
    lse_in: Any
    lse_out: Any
    finalize_key: Any


def dcp_finalize_inputs(mat: Any, record: Any) -> DcpFinalizeInputs:
    """Locate query, score, apply, and the physical buffers they move."""
    value = record.arrived.value
    if value is None or value.reducer != ONLINE_ATTENTION:
        raise ValueError(
            f"{mat._node(record.node).display_path}: expected an "
            f"online-attention finalize, got {value!r}"
        )
    if record.site != RedistributeSite.EDGE:
        raise ValueError(
            f"DCP finalize must occur on an edge, got {record.site!r}"
        )
    world = value.group.id
    signature = (
        (world, "all_gather_lse"),
        (world, "all_reduce"),
    )
    if tuple(record.collectives) != signature:
        raise ValueError(
            f"DCP finalize requires exact physical signature "
            f"{signature!r}, got {record.collectives!r}"
        )
    resolved_family = mat.analysis.rule.families.get(world)
    if resolved_family is None or not same(resolved_family, value.group):
        raise ValueError(
            f"DCP value group {value.group!r} does not match resolver "
            f"family {resolved_family!r} for world {world!r}"
        )
    dcp = value.group.group_size
    provenance = mat._partial_provenance(record, value)
    apply_uid = provenance.producer
    score_uid = provenance.operand("score")
    query_uid = provenance.operand("query")
    query_targets = mat._targets_for_source(score_uid, query_uid, "out")
    if len(query_targets) != 1:
        raise ValueError(
            f"{mat._node(score_uid).display_path}: DCP query must feed "
            f"exactly one score input, got {query_targets!r}"
        )
    query_target = query_targets[0]
    query_required = mat.analysis.input_states.get(
        (score_uid, query_target.input_index)
    )
    if query_required is None:
        raise ValueError(
            f"{mat._node(score_uid).display_path}: DCP query input has "
            "no consumer-required state"
        )
    query_state = mat.analysis.states.values.get(query_uid)
    if query_state is None:
        raise ValueError(
            f"{mat._node(query_uid).display_path}: DCP query has no "
            "producer state"
        )
    if (
        query_state.layout != query_required.layout
        or query_state.value != query_required.value
        or query_state.workload
        != query_required.workload
    ):
        raise ValueError(
            f"{mat._node(score_uid).display_path}: DCP query head "
            "expansion requires identical producer and consumer "
            f"states, got {query_state!r} -> {query_required!r}"
        )
    query_local = local_shape(mat, query_uid, query_state)
    query_required_local = local_shape(mat, query_uid, query_required)
    if query_local != query_required_local:
        raise ValueError(
            f"{mat._node(score_uid).display_path}: DCP query states "
            f"project differently, {query_local!r} != "
            f"{query_required_local!r}"
        )
    score_kernel = mat.analysis.kernel_shapes.values.get(score_uid)
    if score_kernel is None:
        raise ValueError(
            f"{mat._node(score_uid).display_path}: DCP score has no "
            "physical kernel geometry"
        )
    if query_target.input_index >= len(score_kernel.ins):
        raise ValueError(
            f"{mat._node(score_uid).display_path}: DCP query input "
            f"{query_target.input_index} is absent from kernel geometry"
        )
    query_gathered = list(score_kernel.ins[query_target.input_index])
    semantic_exit_uid = (
        record.producer if record.producer >= 0 else record.node
    )
    # A deferred Partial may leave apply through a merge wrapper, or
    # reach the record's edge consumer directly. The physical RS sits
    # immediately before whichever node first consumes the apply value.
    exit_uid = (
        record.node
        if semantic_exit_uid == apply_uid
        else semantic_exit_uid
    )
    apply_kernel = mat.analysis.kernel_shapes.values.get(apply_uid)
    if apply_kernel is None or apply_kernel.out is None:
        raise ValueError(
            f"{mat._node(apply_uid).display_path}: DCP apply has no "
            "physical kernel output geometry"
        )
    if score_kernel.out is None or not apply_kernel.ins:
        raise ValueError(
            f"{mat._node(apply_uid).display_path}: DCP score/apply "
            "kernel chain is incomplete"
        )
    score_output = tuple(value_of(axis) for axis in score_kernel.out)
    apply_scores = tuple(value_of(axis) for axis in apply_kernel.ins[0])
    if score_output != apply_scores:
        raise ValueError(
            f"{mat._node(apply_uid).display_path}: DCP score output "
            f"{score_output!r} does not feed apply input {apply_scores!r}"
        )
    combined_apply = list(apply_kernel.out)
    apply_head_axis = axis_of_kind(mat, apply_uid, "head_count")
    exit_targets = mat._targets_for_source(exit_uid, apply_uid, "out")
    if semantic_exit_uid == apply_uid:
        direct_target = RewriteTarget(record.node, record.input_index)
        if direct_target not in exit_targets:
            raise ValueError(
                f"{mat._node(record.node).display_path}: direct DCP "
                f"consumer target {direct_target!r} is absent from "
                f"physical apply fanout {exit_targets!r}"
            )
    exit_local = rewrite_flow_shape(mat, exit_targets)
    if apply_head_axis == len(combined_apply) - 1:
        raise ValueError(
            f"{mat._node(apply_uid).display_path}: DCP apply has no "
            "trailing feature axis to remove from the LSE statistic"
        )
    lse_in = [*combined_apply[:-1], 1]
    lse_out = [*combined_apply[:-1], dcp]
    finalize_key = (
        "online_attention",
        value.group,
        apply_uid,
        score_uid,
        query_uid,
    )
    changed_query_axes = [
        axis for axis, (local, gathered) in enumerate(
            zip(query_local, query_gathered, strict=True)
        ) if local != gathered
    ]
    if len(changed_query_axes) != 1:
        raise ValueError(
            f"{mat._node(score_uid).display_path}: DCP query gather "
            f"must expand exactly one axis, got {query_local!r} -> "
            f"{query_gathered!r}"
        )
    query_axis = changed_query_axes[0]
    query_collective_out = list(query_local)
    query_collective_out[query_axis] *= dcp
    query_post = (
        PostTransform.NONE
        if query_collective_out == query_gathered
        else PostTransform.REPLICA_SELECT
    )
    return DcpFinalizeInputs(
        value=value, world=world, dcp=dcp, apply_uid=apply_uid,
        score_uid=score_uid, query_uid=query_uid, query_target=query_target,
        query_local=query_local, query_gathered=query_gathered,
        query_post=query_post, query_collective_out=query_collective_out,
        exit_uid=exit_uid, combined_apply=combined_apply,
        exit_local=exit_local, exit_targets=exit_targets,
        lse_in=lse_in, lse_out=lse_out, finalize_key=finalize_key,
    )
