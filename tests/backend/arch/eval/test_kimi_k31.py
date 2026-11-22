"""K3.1 SGLang contracts, anchored to 1b2870d3 and a B300 cache-writer probe.

The constants are independent of the spec: checkpoint header shapes, SGLang's
physical FP8 writer, and KimiLinearStateShape.create define the expected values.
"""
from collections import Counter

import pytest
import sympy as sp

from backend.arch.eval import cache
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.eval.values import CACHE_LEN
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import shared_loader, snapshot_config

MODEL = 'moonshotai/Kimi-K3.1-Pro'


@pytest.fixture(scope='module')
def program():
    return Program.of(shared_loader().match(snapshot_config(MODEL), model_id=MODEL))


@pytest.fixture(scope='module')
def analysis(program):
    return Deployment(program).at(batch=1, input_tokens=4096).analyze()


def test_layer_schedule_and_nope_graph(program):
    counts = Counter()
    sparse = []
    for block in program.blocks:
        if block.name == 'kimi_k3_vision_tower':
            continue
        attn = block.attention_call()
        counts[attn.op] += len(block.layer_indices)
        if attn.op == 'kimi_k31_dsa_attention':
            sparse.extend(block.layer_indices)
            ops = {n.op for n in block.nodes()}
            assert not ops.intersection({'rope', 'rope_emb', 'hadamard'})
            assert {'dsa_attention_score', 'dsa_attention_apply', 'sigmoid', 'mul'} <= ops
    assert counts == {'kimi_k3_kda_attention': 69, 'kimi_k31_dsa_attention': 24}
    assert sorted(sparse) == [*range(3, 93, 4), 92]


def test_segment_start_blocks_output_attention_plus_ffn(program):
    # A segment-start layer restarts its prefix sum at the attention output
    # and ffn_update=PLAIN_ADD then joins the FFN onto it, so the block
    # graphs one top-level residual add (attention + FFN) and declares that
    # add its output — never the bare FFN.
    ffn = {'block_kda_dense_first': 'mlp',
           'block_kda_moe_start': 'block_sparse_moe'}
    covered = []
    for block in program.blocks:
        if block.name not in ffn:
            continue
        covered.extend(block.layer_indices)
        top_adds = [n for n in block.roots if n.op == 'add']
        assert [n.name for n in top_adds] == ['mlp_residual']
        assert list(top_adds[0].inputs) == ['self_attn', ffn[block.name]]
        (out,) = block.output_roots()
        assert out.name == 'mlp_residual'
    assert sorted(covered) == list(range(0, 93, 12))


def test_physical_cache_and_independent_indexer(analysis):
    growth = 0
    fixed = 0
    layer_count = 0
    for uid, entry in cache.derive(analysis).values.items():
        if not entry.persists:
            continue
        count = int(value_of(analysis.instances.get(uid, 1)))
        layer_count += count
        op = analysis.program.by_uid(uid).op
        if op == 'kimi_k31_dsa_attention':
            assert int(value_of(entry.bytes_at(1))) == 656 + 132
            latent, tail, key = entry.tensors
            for t in (latent, tail):
                assert t.stored == CACHE_LEN
                assert t.read == sp.Min(2048, CACHE_LEN)
            assert latent.format_name == 'fp8_pow2_group128'
            assert int(value_of(latent.position_bytes + tail.position_bytes)) == 656
            assert int(value_of(key.position_bytes)) == 132
            assert int(value_of(key.elements)) == 128  # MQA, not 32*128
            assert key.stored == key.read == CACHE_LEN
            assert key.format_name == 'fp8_group128'
        elif op == 'kimi_k3_kda_attention':
            # A four-tap convolution retains three prior positions.
            assert int(value_of(entry.bytes_at(1))) == 96 * 128 * 128 * 4 + 3 * 96 * 128 * 3 * 2
            fixed += int(value_of(entry.bytes_at(1))) * count
            continue
        growth += int(value_of(entry.bytes_at(1))) * count
    assert growth == 24 * (656 + 132) == 18912
    assert fixed == 69 * 6512640
    # Indexer storage belongs to its DSA layer, not 24 extra layers.
    assert layer_count == 93


def test_dcp8_physical_cache(program):
    from backend.arch.layout.physical_cache import cache_geometry
    analysis = Deployment(program, config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).at(batch=1, input_tokens=4096).analyze()
    entries = cache.derive(analysis, geometry=cache_geometry(analysis))
    growth = sum(
        int(value_of(e.bytes_at(8192) - e.bytes_at(4096))) * int(value_of(analysis.instances.get(uid, 1)))
        for uid, e in entries.values.items() if e.persists
    )
    assert growth == 4096 * 18912 // 8


def test_k3_serving_conv_cache_retains_three_prior_tokens():
    model = 'moonshotai/Kimi-K3'
    program = Program.of(shared_loader().match(snapshot_config(model), model_id=model))
    a = Deployment(program).at(batch=1, input_tokens=128).analyze()
    entry = next(e for uid, e in cache.derive(a).values.items()
                 if a.program.by_uid(uid).op == 'kimi_k3_kda_attention')
    assert int(value_of(entry.bytes_at(1))) == 96 * 128 * 128 * 4 + 3 * 96 * 128 * 3 * 2


@pytest.mark.regression  # a snapshot of the whole B300 estimate, not an oracle
def test_b300_fitting_deployment_publishes_nonempty_rows_and_dtypes():
    from backend.inference.capacity import estimate_capacity
    from backend.inference.framework_bindings import THEORY

    cfg = snapshot_config(MODEL)
    resolved = shared_loader().match(cfg, model_id=MODEL)
    result = estimate_capacity(
        config=cfg,
        resolved=resolved,
        gpu_sku='B300',
        num_gpus=8,
        parallel=ParallelConfig(
            tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2,
        ),
        framework=THEORY,
        avg_input_tokens=4096,
        avg_output_tokens=1024,
    )
    wb = result['weight_breakdown']
    text_params = (
        sum(g['params'] for g in wb['groups'])
        + sum(m['params'] for m in wb['model_level'])
    )
    assert text_params == 2_779_663_004_416
    assert result['max_batch_per_gpu'] == 354
    assert len(result['per_op_timing']) == 1890
    assert len(result['per_op_timing_decode']) == 1890
    decode_pairs = {(r['op'], r['act_dtype']) for r in result['per_op_timing_decode']}
    assert {
        ('dsa_attention_score', 'BF16'),
        ('dsa_attention_apply', 'BF16'),
        ('dsa_topk_mask', 'F32'),
        ('indexer_logits', 'F8_E4M3'),
        ('gated_delta_rule', 'BF16'),
        ('attn_res_combine', 'BF16'),
    } <= decode_pairs


def test_b300_sglang_dcp8_sparse_attention():
    from backend.inference.capacity import estimate_capacity

    cfg = snapshot_config(MODEL)
    resolved = shared_loader().match(cfg, model_id=MODEL)
    result = estimate_capacity(
        config=cfg,
        resolved=resolved,
        gpu_sku='B300',
        num_gpus=8,
        parallel=ParallelConfig(
            tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
        ),
        framework='sglang',
        avg_input_tokens=4096,
        avg_output_tokens=1024,
    )
    assert result['max_batch_per_gpu'] > 0
    assert result['per_op_timing']
    decode = result['per_op_timing_decode']
    indexers = [row for row in decode if row['op'] == 'indexer_logits']
    assert indexers
    # Decode uses prompt + half the output: 4608 global keys, 576 per rank.
    assert all(row['shape_in'][1][-2] == 576 for row in indexers)
    assert all(row['flops_per_rank'] == row['count_per_step'] * row['shape_in'][0][0]
               * 32 * (2 * 128 + 3) * 576
               for row in indexers)
    stages = {row.get('runtime_stage_name') for row in decode}
    assert {'dcp_topk_merge_exchange_certificate', 'dcp_selected_owner_localize',
            'flashmla_split_combine'} <= stages
    assumptions = {item['id'] for item in result['assumptions']}
    assert {'sparse_dcp_selected_ownership', 'sparse_dcp_candidate_fallback'} <= assumptions
