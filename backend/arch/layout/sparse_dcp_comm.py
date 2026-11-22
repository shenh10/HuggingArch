"""Sparse DCP exchange rows: multicast candidate selection and packed combine.

All-gather/all-to-all name bandwidth classes, not NCCL implementation claims.
Payloads count logical recipients, not measured NVLink transactions. No query
all-gather is emitted for the explicitly replicated query runtime plan.
"""
from typing import Any

import sympy as sp

from backend.arch.eval.dim import exact_div, minimum, require_equal, value_of, where
from backend.arch.eval.transition import RedistributeSite
from .materialize_comm import (
    FlowRef, LogicalBindingGap, PayloadRole, Placement, PostTransform,
)
from .materialize_flow import rewrite_flow_shape
from .runtime_stage import RuntimeStageWork
from .sparse_dcp_contract import CandidateExchange, LocalAttentionCombine, OutputExchange, SparseDcpLocalStages, SparseDcpWorkload, SparseMlaLocalStages, causal_query_lengths
from .value import ONLINE_ATTENTION


def sparse_dcp_finalize(mat: Any, record: Any):
    """Return sparse events or None; zero fallback is a labelled baseline."""
    plans = mat.analysis.rule.sparse_dcp
    value = record.arrived.value
    if value is None or value.reducer != ONLINE_ATTENTION or value.provenance is None:
        return None
    matches = [plan for plan in plans.plans if plan.decode and (
        value.provenance.producer == plan.nodes['apply'].uid
        or record.producer == plan.nodes['value_projection'].uid)]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError('sparse DCP finalize must have exactly one bound plan')
    plan = matches[0]
    if not plan.replicate_q or not plan.preproject_v:
        raise NotImplementedError('sparse exchange requires replicated Q and preprojected V')
    contract = mat.analysis.rule.sparse_dcp.contract
    sources = contract.sources
    topk = plan.nodes['indexer_topk']
    projected = plan.nodes['value_projection']
    exit_uid = record.node if record.producer == projected.uid else record.producer
    if record.site != RedistributeSite.EDGE:
        raise ValueError('sparse DCP finalize must occur at a consumer edge')
    physical = tuple(mat.analysis.kernel_shapes.values[projected.uid].out)
    candidate_shape = tuple(mat.analysis.kernel_shapes.values[topk.uid].out)
    if len(physical) != 4 or len(candidate_shape) != 3:
        raise ValueError('sparse DCP expects [B,H,Q,D] and [B,Q,K] kernel geometry')
    batch, heads, queries, width = physical
    rows = batch * queries
    candidates = CandidateExchange(rows, candidate_shape[-1], plan.degree,
                               failed_owners=mat.analysis.rule.dcp_failed_owners,
                               candidate_count=contract.candidate_count,
                               pair_bytes=contract.pair_bytes)
    local_indexer = mat.analysis.kernel_shapes.values[plan.nodes['indexer_logits'].uid]
    local_stages = SparseDcpLocalStages(rows, candidate_shape[-1], plan.degree,
                                      minimum(local_indexer.out[-1], candidate_shape[-1]),
                                      candidates.failed_owners,
                                      candidates=contract.candidate_count,
                                      sources=sources)
    output = OutputExchange.from_contract(rows, heads, plan.degree, width, contract)
    lse_bytes = output.lse_slot_bytes
    targets = mat._targets_for_source(exit_uid, projected.uid, 'out')
    semantic = rewrite_flow_shape(mat, targets)
    expected = (batch, exact_div(heads, plan.degree), queries, width)
    for got, wanted in zip(semantic, expected, strict=True):
        require_equal(got, wanted, lambda at: ValueError('sparse DCP local output geometry mismatch'))
    key = ('sparse_dcp', value.group, projected.uid)
    # Two sequential rounds, each its own collective row — the envelope
    # prices two all_gather latencies, so the dependent fallback is a second
    # round, not bytes folded into one multicast. Round one multicasts the
    # candidate certificate; round two — only when some query's certificate
    # failed an owner — is the busiest failed owner serving its full K to
    # every peer. `minimum(failed_owners, 1)` is the workload fact that says
    # whether round two happens: 0 when every owner held its candidates (the
    # row prices nothing), 1 when any did not.
    candidate_in = (*candidate_shape[:-1], candidates.candidate_count)
    candidate_out = (*candidate_shape[:-1], candidates.candidate_count * plan.degree)
    index_event = mat._event(
        record=record, anchor_uid=topk.uid, op='all_gather',
        group_size=plan.degree, count_uid=topk.uid, world=value.group.id,
        flow_in_shape=candidate_shape, flow_out_shape=candidate_shape,
        collective_in_shape=candidate_in, collective_out_shape=candidate_out,
        payload_shape=candidate_out, payload_role=PayloadRole.AUXILIARY_STATISTIC,
        payload_dtype_bytes=candidates.pair_bytes,
        source=FlowRef(topk.uid), rewrites=(), placement_uid=topk.uid,
        placement=Placement.AFTER, attribution_uid=topk.uid,
        site=RedistributeSite.INTERNAL_OUT, family=value.group,
        dedup_key=(*key, 'candidate_exchange'),
        runtime_work=local_stages.candidate_merge(),
        meta=(('sparse_dcp_protocol', 'candidate_multicast'),
              ('candidate_count', candidates.candidate_count),
              ('traffic_basis', 'logical_remote_recipient_payload_not_hardware_counter'),
              ('source', sources['candidate_merge'])),
    )
    failed_serve = candidate_shape[-1] * minimum(candidates.failed_owners, 1)
    fallback_in = (*candidate_shape[:-1], failed_serve)
    fallback_out = (*candidate_shape[:-1], failed_serve * plan.degree)
    fallback_event = mat._event(
        record=record, anchor_uid=topk.uid, op='all_gather',
        group_size=plan.degree, count_uid=topk.uid, world=value.group.id,
        flow_in_shape=candidate_shape, flow_out_shape=candidate_shape,
        collective_in_shape=fallback_in, collective_out_shape=fallback_out,
        payload_shape=fallback_out, payload_role=PayloadRole.AUXILIARY_STATISTIC,
        payload_dtype_bytes=candidates.pair_bytes,
        source=None, source_event=index_event.event_id, rewrites=(),
        placement_uid=topk.uid, placement=Placement.AFTER,
        attribution_uid=topk.uid, site=RedistributeSite.INTERNAL_OUT,
        family=value.group, dedup_key=(*key, 'candidate_fallback'),
        tree_anchor_event=index_event.event_id,
        meta=(('sparse_dcp_protocol', 'candidate_fallback'),
              ('collective_active', minimum(candidates.failed_owners, 1)),
              ('sparse_dcp_fallback_assumption', 'explicit_failed_owner_hotspot'),
              ('sparse_dcp_traffic_scope', 'maximum_sender_recipient_payload'),
              ('sparse_dcp_mean_recipient_bytes', candidates.fallback_mean_receive_bytes),
              ('sparse_dcp_max_receive_bytes', candidates.fallback_max_receive_bytes),
              ('traffic_basis', 'logical_remote_recipient_payload_not_hardware_counter'),
              ('source', sources['candidate_merge'])),
    )
    packed = (rows, heads, width + exact_div(lse_bytes, output.element_bytes))
    local_work = LocalAttentionCombine(rows, heads, plan.degree, width,
                                       lse_bytes, where(lambda r: sp.Le(r, contract.push_max_rows), (rows,), 1, 0))
    combine_work = RuntimeStageWork(
        'dcp_lse_weighted_combine', local_work.read_bytes, local_work.write_bytes,
        sources['combine'],
        flops=local_work.flops, compute_dtype='fp32',
        work_basis='local_lse_merge_math_floor; polling_excluded')
    output_event = mat._event(
        record=record, anchor_uid=exit_uid, op='all_to_all',
        group_size=plan.degree, count_uid=projected.uid, world=value.group.id,
        flow_in_shape=physical, flow_out_shape=semantic,
        collective_in_shape=packed, collective_out_shape=packed,
        payload_shape=packed, payload_role=PayloadRole.PACKED_ATTENTION_PARTIAL,
        payload_dtype_bytes=output.element_bytes,
        post_transform=PostTransform.ONLINE_ATTENTION_COMBINE,
        binding_gap=LogicalBindingGap.DCP_HEAD_FLOW,
        source=FlowRef(projected.uid), rewrites=targets,
        placement_uid=exit_uid, placement=Placement.BEFORE,
        attribution_uid=exit_uid, site=RedistributeSite.EDGE,
        family=value.group, dedup_key=(*key, 'projected_combine'),
        runtime_work=combine_work,
        meta=(('sparse_dcp_protocol', 'projected_push_or_fi_a2a'),
              ('sparse_dcp_reference_profile', f'registered_push_max{contract.push_max_rows}_bf16_prewarmed_no_lora'),
              ('sparse_dcp_lse_slot_bytes', lse_bytes),
              ('replicated_query', True), ('query_all_gather', False),
              ('traffic_basis', 'logical_remote_recipient_payload_not_hardware_counter'),
              ('source', sources['combine'])),
    )
    localized = mat._event(
        record=record, anchor_uid=topk.uid, op='runtime_stage',
        group_size=plan.degree, count_uid=topk.uid, world=value.group.id,
        flow_in_shape=candidate_shape, flow_out_shape=candidate_shape,
        collective_in_shape=candidate_shape, collective_out_shape=candidate_shape,
        payload_shape=(0,), payload_role=PayloadRole.RUNTIME_STAGE,
        payload_dtype_bytes=4, runtime_work=local_stages.localize(),
        source=None, source_event=fallback_event.event_id, rewrites=(), placement_uid=topk.uid,
        placement=Placement.AFTER, attribution_uid=topk.uid,
        tree_anchor_event=fallback_event.event_id,
        site=RedistributeSite.INTERNAL_OUT, family=value.group,
        dedup_key=(*key, 'localize'),
    )
    context = value_of(mat.analysis.rule.env[plan.nodes['indexer_logits'].uid]['s_kv'])
    verify_width = int(value_of(mat.analysis.rule.env[topk.uid]['n_verify']))
    local_lengths = tuple(value_of(SparseDcpWorkload(
        1, length, value_of(candidate_shape[-1]),
        plan.degree, 'busiest_rank', value_of(mat.analysis.rule.dcp_selected_load_factor)
    ).valid_per_query) for length in causal_query_lengths(context, verify_width))
    mla = SparseMlaLocalStages(value_of(batch), local_lengths, int(value_of(heads)),
                              int(value_of(mat.analysis.kernel_shapes.values[plan.nodes['apply'].uid].out[-1])),
                              mat.sm_count,
                              block_size=contract.flashmla_block_size,
                              overhead=contract.flashmla_overhead,
                              sources=sources)
    scheduler = mat._event(
        record=record, anchor_uid=topk.uid, op='runtime_stage',
        group_size=plan.degree, count_uid=topk.uid, world=value.group.id,
        flow_in_shape=candidate_shape, flow_out_shape=candidate_shape,
        collective_in_shape=candidate_shape, collective_out_shape=candidate_shape,
        payload_shape=(0,), payload_role=PayloadRole.RUNTIME_STAGE,
        payload_dtype_bytes=4, runtime_work=mla.scheduler(),
        source=None, source_event=localized.event_id, rewrites=(), placement_uid=topk.uid,
        placement=Placement.AFTER, attribution_uid=topk.uid,
        tree_anchor_event=localized.event_id,
        site=RedistributeSite.INTERNAL_OUT, family=value.group,
        dedup_key=(*key, 'mla_scheduler'))
    apply = plan.nodes['apply']
    apply_shape = tuple(mat.analysis.shapes.values[apply.uid].out)
    split_combine = mat._event(
        record=record, anchor_uid=apply.uid, op='runtime_stage',
        group_size=plan.degree, count_uid=apply.uid, world=value.group.id,
        flow_in_shape=apply_shape, flow_out_shape=apply_shape,
        collective_in_shape=apply_shape, collective_out_shape=apply_shape,
        payload_shape=(0,), payload_role=PayloadRole.RUNTIME_STAGE,
        payload_dtype_bytes=4, runtime_work=mla.combine(apply.uid),
        source=FlowRef(apply.uid), rewrites=(), placement_uid=apply.uid,
        placement=Placement.AFTER, attribution_uid=apply.uid,
        site=RedistributeSite.INTERNAL_OUT, family=value.group,
        dedup_key=(*key, 'mla_split_combine'))
    return [index_event, fallback_event, localized, scheduler, split_combine,
            output_event]
