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
from .sparse_dcp_contract import CandidateExchange, LocalAttentionCombine, OutputExchange, SparseDcpLocalStages, SparseDcpWorkload, SparseMlaLocalStages
from .value import ONLINE_ATTENTION

_SOURCE = ('https://github.com/sgl-project/sglang/blob/'
           '1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/')


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
    candidates = CandidateExchange(rows, candidate_shape[-1], plan.degree, failed_owners=mat.analysis.rule.dcp_failed_owners)
    local_indexer = mat.analysis.kernel_shapes.values[plan.nodes['indexer_logits'].uid]
    local_stages = SparseDcpLocalStages(rows, candidate_shape[-1], plan.degree,
                                      minimum(local_indexer.out[-1], candidate_shape[-1]),
                                      candidates.failed_owners)
    output = OutputExchange(rows, heads, plan.degree, width, 'push')
    # Reference profile: registered push plane, MAX_ROWS=1536, BF16 and
    # prewarmed V weights. Above capacity, R>=512 still enables projection;
    # the direct A2A range ends at1024, so FlashInfer supplies 8-byte stats.
    lse_bytes = where(lambda r: sp.Le(r, 1536), (rows,), 16, 8)
    targets = mat._targets_for_source(exit_uid, projected.uid, 'out')
    semantic = rewrite_flow_shape(mat, targets)
    expected = (batch, exact_div(heads, plan.degree), queries, width)
    for got, wanted in zip(semantic, expected, strict=True):
        require_equal(got, wanted, lambda at: ValueError('sparse DCP local output geometry mismatch'))
    key = ('sparse_dcp', value.group, projected.uid)
    # One kernel and one timing event: C multicast candidates plus the
    # busiest failed owner serving its full K to all peers. This is
    # hotspot recipient accounting, not a claim
    # of one allocated packed buffer or a separate fallback launch.
    effective_candidates = (candidates.candidate_count
                            + candidate_shape[-1] * minimum(candidates.failed_owners, 1))
    candidate_in = (*candidate_shape[:-1], effective_candidates)
    candidate_out = (*candidate_shape[:-1], effective_candidates * plan.degree)
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
              ('sparse_dcp_traffic_scope', 'maximum_sender_recipient_payload'),
              ('sparse_dcp_fallback_assumption', 'explicit_failed_owner_hotspot'),
              ('sparse_dcp_dependency_rounds', where(lambda f: sp.Gt(f, 0), (candidates.failed_owners,), 2, 1)),
              ('sparse_dcp_mean_recipient_bytes', candidates.multicast_recipient_bytes + candidates.fallback_mean_receive_bytes),
              ('sparse_dcp_max_receive_bytes', candidates.multicast_recipient_bytes + candidates.fallback_max_receive_bytes),
              ('traffic_basis', 'logical_remote_recipient_payload_not_hardware_counter'),
              ('source', _SOURCE + 'kernels/jit/csrc/dsa/dcp_topk_merge.cuh#L301')),
    )
    packed = (rows, heads, width + exact_div(lse_bytes, output.element_bytes))
    local_work = LocalAttentionCombine(rows, heads, plan.degree, width,
                                       lse_bytes, where(lambda r: sp.Le(r, 1536), (rows,), 1, 0))
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
        meta=(('sparse_dcp_protocol', 'projected_push_or_fi_a2a'),
              ('sparse_dcp_local_flops', local_work.flops),
              ('sparse_dcp_local_read_bytes', local_work.read_bytes),
              ('sparse_dcp_local_write_bytes', local_work.write_bytes),
              ('sparse_dcp_local_compute_dtype', 'fp32'),
              ('sparse_dcp_reference_profile', 'registered_push_max1536_bf16_prewarmed_no_lora'),
              ('sparse_dcp_lse_slot_bytes', '16_if_rows_le1536_else8'),
              ('replicated_query', True), ('query_all_gather', False),
              ('traffic_basis', 'logical_remote_recipient_payload_not_hardware_counter'),
              ('source', _SOURCE + 'srt/layers/dcp/comm.py#L552')),
    )
    localized = mat._event(
        record=record, anchor_uid=topk.uid, op='runtime_stage',
        group_size=plan.degree, count_uid=topk.uid, world=value.group.id,
        flow_in_shape=candidate_shape, flow_out_shape=candidate_shape,
        collective_in_shape=candidate_shape, collective_out_shape=candidate_shape,
        payload_shape=(0,), payload_role=PayloadRole.RUNTIME_STAGE,
        payload_dtype_bytes=4, runtime_work=local_stages.localize(),
        source=None, source_event=index_event.event_id, rewrites=(), placement_uid=topk.uid,
        placement=Placement.AFTER, attribution_uid=topk.uid,
        tree_anchor_event=index_event.event_id,
        site=RedistributeSite.INTERNAL_OUT, family=value.group,
        dedup_key=(*key, 'localize'),
    )
    context = value_of(mat.analysis.rule.env[plan.nodes['indexer_logits'].uid]['s_kv'])
    verify_width = int(value_of(mat.analysis.rule.env[topk.uid]['n_verify']))
    local_lengths = tuple(value_of(SparseDcpWorkload(
        1, context - verify_width + 1 + offset, value_of(candidate_shape[-1]),
        plan.degree, 'busiest_rank', value_of(mat.analysis.rule.dcp_selected_load_factor)
    ).valid_per_query) for offset in range(verify_width))
    mla = SparseMlaLocalStages(value_of(batch), local_lengths, int(value_of(heads)),
                              int(value_of(mat.analysis.kernel_shapes.values[plan.nodes['apply'].uid].out[-1])),
                              mat.sm_count)
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
    return [index_event, localized, scheduler, split_combine, output_event]
