"""Packed cache gathers precede sparse prefill without padding its FLOPs."""
import pytest

from backend.arch.eval.transforms import Deployment
from backend.arch.eval.dim import value_of
from backend.arch.layout.materializer import materialize_comm_tree
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import shared_program


@pytest.mark.parametrize('context', [17, 8193, 2048])
def test_prefill_gathers_fp8_index_and_mla_cache_with_equal_rank_padding(context):
    analysis = Deployment(shared_program('moonshotai__Kimi-K3.1-Pro'), config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).at(batch=2, input_tokens=context).during('prefill').analyze()
    result = materialize_comm_tree(analysis)
    events = [event for event in result.events
              if dict(event.meta).get('sparse_dcp_protocol') == 'prefill_cache_gather']
    for kind, row_bytes in (('index', 132), ('mla', 656)):
        gathered = [event for event in events if dict(event.meta)['cache_buffer'] == kind]
        assert sum(event.count for event in gathered) == 24
        for event in gathered:
            assert event.payload_dtype_bytes == 1
            assert tuple(map(value_of, event.collective_in_shape)) == (2, (context + 7) // 8, row_bytes)
            assert tuple(map(value_of, event.collective_out_shape)) == (2, ((context + 7) // 8) * 8, row_bytes)
            assert value_of(event.bytes_per_rank) == event.count * 2 * ((context + 7) // 8) * row_bytes * 7
            assert not event.rewrites
    # No decode candidate or partial-output exchanges during prefill.
    assert not any(dict(event.meta).get('sparse_dcp_protocol') in
                   ('candidate_multicast', 'projected_push_or_fi_a2a') for event in result.events)
