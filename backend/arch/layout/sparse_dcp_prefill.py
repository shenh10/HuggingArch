"""SGLang sparse prefill gathers its striped cache into two packed buffers.

The communication pads each request to equally sized rank slices, gathers,
then trims to its logical context. MLA's later 64-row page padding is local
allocation, not network traffic. Kernel computation uses the logical context.
"""
from backend.arch.eval import cache
from backend.arch.eval.dim import ceil_div
from backend.arch.eval.transition import Redistribute, RedistributeSite
from .materialize_comm import FlowRef, PayloadRole, Placement
from .materialize_flow import local_shape

_SOURCE = ('https://github.com/sgl-project/sglang/blob/'
           '1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/')


def sparse_dcp_prefill_events(mat):
    """Additional cache-buffer collectives; the graph's logical flow is unchanged."""
    plans = tuple(plan for plan in mat.analysis.rule.sparse_dcp.plans if not plan.decode)
    if not plans:
        return []
    caches = cache.derive(mat.analysis)
    events = []
    for plan in plans:
        entry = caches.values[plan.component.uid]
        for role, label, kinds, evidence in (
            ('indexer_logits', 'index', ('index_key',),
             'srt/layers/attention/dsa/dsa_indexer.py#L1986'),
            ('score', 'mla', ('latent', 'rope_key'),
             'srt/layers/attention/dsa_backend.py#L1387'),
        ):
            anchor = plan.nodes[role]
            tensors = tuple(tensor for tensor in entry.tensors if tensor.kind in kinds)
            if {tensor.kind for tensor in tensors} != set(kinds):
                raise ValueError(f'sparse prefill {label} gather has incomplete cache slots')
            # Cache formats own both payload data and their per-row scales.
            row_bytes = sum(tensor.position_bytes for tensor in tensors)
            env = mat.analysis.node_dims.values[anchor.uid]
            batch, context = env['B'], env['s_kv']
            local_rows = ceil_div(context, plan.degree)
            collective_in = (batch, local_rows, row_bytes)
            collective_out = (batch, local_rows * plan.degree, row_bytes)
            # Both primitives declare the cached key as their second input.
            # This is a buffer gather, not a rewrite of that logical view.
            source = anchor.inputs_obj[1].src
            state = mat.analysis.states.values[source.uid]
            flow = tuple(local_shape(mat, source.uid, state))
            record = Redistribute(
                node=anchor.uid, input_index=1, producer=source.uid,
                arrived=state, required=state,
                collectives=((mat.analysis.rule.cache_family.id, 'all_gather'),),
                site=RedistributeSite.INTERNAL_IN)
            events.append(mat._event(
                record=record, anchor_uid=anchor.uid, op='all_gather',
                group_size=plan.degree, count_uid=anchor.uid,
                world=mat.analysis.rule.cache_family.id,
                flow_in_shape=flow, flow_out_shape=flow,
                collective_in_shape=collective_in, collective_out_shape=collective_out,
                payload_shape=collective_out, payload_dtype_bytes=1,
                payload_role=PayloadRole.AUXILIARY_STATISTIC,
                source=FlowRef(source.uid), rewrites=(),
                placement_uid=anchor.uid, placement=Placement.BEFORE,
                attribution_uid=anchor.uid, site=RedistributeSite.INTERNAL_IN,
                family=mat.analysis.rule.cache_family,
                dedup_key=('sparse_dcp_prefill', plan.component.uid, label),
                meta=(('sparse_dcp_protocol', 'prefill_cache_gather'),
                      ('cache_buffer', label), ('packed_row_bytes', row_bytes),
                      ('logical_context', context),
                      ('gathered_rows_per_request', local_rows * plan.degree),
                      ('trim_padding_after_gather', True),
                      ('traffic_basis', 'packed_cache_bytes_with_equal_rank_padding'),
                      ('source', _SOURCE + evidence)),
            ))
    return events
