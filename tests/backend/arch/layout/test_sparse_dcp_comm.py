"""Sparse DCP must emit its actual exchanges, not the dense DCP trio."""
import pytest

from backend.arch.eval.transforms import Deployment
from backend.arch.layout.materializer import materialize_comm_tree
from backend.arch.layout.materialize_comm import PayloadRole, PostTransform
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import shared_program


def test_sparse_dcp_24_layer_communication_payloads_and_semantic_rewrite():
    program = shared_program('moonshotai__Kimi-K3.1-Pro')
    analysis = Deployment(program, config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).during('decode').at(batch=1, input_tokens=32768).analyze()
    result = materialize_comm_tree(analysis, sm_count=148)
    events = [event for event in result.events if event.world == 'dcp']
    candidate = [e for e in events if dict(e.meta).get('sparse_dcp_protocol') == 'candidate_multicast']
    combine = [e for e in events if dict(e.meta).get('sparse_dcp_protocol') == 'projected_push_or_fi_a2a']
    localize = [e for e in events if e.runtime_work is not None
                and e.runtime_work.name == 'dcp_selected_owner_localize']
    scheduler = [e for e in events if e.runtime_work is not None
                 and e.runtime_work.name == 'flashmla_windowed_scheduler']
    split = [e for e in events if e.runtime_work is not None
             and e.runtime_work.name == 'flashmla_split_combine']
    assert len(events) == len(candidate) + len(combine) + len(localize) + len(scheduler) + len(split)
    assert sum(e.count for e in scheduler) == sum(e.count for e in split) == 24
    assert all(int(e.runtime_work.write_bytes) == 74 * 32 + 8 for e in scheduler)
    # Balanced K2048/8 produces 256 valid keys, four windowed splits.
    assert all(int(e.runtime_work.read_bytes) == 4 * 96 * 513 * 4 + 8 for e in split)
    assert all(int(e.runtime_work.producer_write_bytes) == 4 * 96 * 513 * 4 for e in split)
    assert all(int(result.comm_meta.values[e.runtime_work.producer_uid]['runtime_write_bytes'])
               == 4 * 96 * 513 * 4 for e in split)
    assert sum(e.count for e in localize) == 24
    assert all(e.op == 'runtime_stage' and e.bytes_per_rank == 0 for e in localize)
    assert sum(e.count for e in candidate) == sum(e.count for e in combine) == 24
    assert sum(e.bytes_per_rank for e in candidate) == 24 * 21504
    assert sum(e.bytes_per_rank for e in combine) == 24 * 22848
    assert all(e.op == 'all_gather' and e.payload_dtype_bytes == 8 for e in candidate)
    assert all(not e.rewrites for e in candidate)
    assert all(e.op == 'all_to_all' and e.payload_role == PayloadRole.PACKED_ATTENTION_PARTIAL
               and e.post_transform == PostTransform.ONLINE_ATTENTION_COMBINE for e in combine)
    assert all(tuple(map(int, e.flow_in_shape)) == (1, 96, 1, 128) for e in combine)
    assert all(tuple(map(int, e.flow_out_shape)) == (1, 12, 1, 128) for e in combine)
    assert all(tuple(map(int, e.payload_shape)) == (1, 96, 136) for e in combine)
    assert all(len(e.rewrites) >= 1 for e in combine)
    assert all(int(result.cost.values[result.node_uid_by_event[e.event_id]].flops) == 26484
               for e in combine)
    assert all(int(dict(e.meta)['sparse_dcp_local_read_bytes']) == 47808 for e in combine)
    assert all(int(dict(e.meta)['sparse_dcp_local_write_bytes']) == 48768 for e in combine)


def test_sparse_dcp_fallback_scenario_changes_same_event_hotspot_bytes():
    program = shared_program('moonshotai__Kimi-K3.1-Pro')
    analysis = Deployment(program, config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).during('decode').at(batch=1, input_tokens=32768, dcp_failed_owners=2).analyze()
    result = materialize_comm_tree(analysis)
    candidate = [e for e in result.events
                 if dict(e.meta).get('sparse_dcp_protocol') == 'candidate_multicast']
    assert sum(e.count for e in candidate) == 24
    assert sum(e.bytes_per_rank for e in candidate) == 24 * (21504 + 114688)
    assert all(dict(e.meta)['sparse_dcp_traffic_scope'] == 'maximum_sender_recipient_payload'
               for e in candidate)
    assert not any(dict(e.meta).get('sparse_dcp_protocol') == 'candidate_fallback_mean'
                   for e in result.events)


@pytest.mark.parametrize('rows,width,bytes_per_query', [(1537, 132, 22176), (1536, 136, 22848)])
def test_sparse_dcp_push_capacity_selects_statistics_width(rows, width, bytes_per_query):
    program = shared_program('moonshotai__Kimi-K3.1-Pro')
    analysis = Deployment(program, config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).during('decode').at(batch=rows, input_tokens=32768).analyze()
    result = materialize_comm_tree(analysis)
    combine = [e for e in result.events
               if dict(e.meta).get('sparse_dcp_protocol') == 'projected_push_or_fi_a2a']
    assert sum(e.bytes_per_rank for e in combine) == 24 * rows * bytes_per_query
    assert all(tuple(map(int, e.payload_shape)) == (rows, 96, width) for e in combine)
