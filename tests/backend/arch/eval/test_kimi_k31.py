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
            # SGLang stores 3 historical positions for a four-tap conv,
            # plus a float32 [96,128,128] recurrent state per request.
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


def test_k3_reference_conv_cache_is_unchanged():
    model = 'moonshotai/Kimi-K3'
    program = Program.of(shared_loader().match(snapshot_config(model), model_id=model))
    a = Deployment(program).at(batch=1, input_tokens=128).analyze()
    entry = next(e for uid, e in cache.derive(a).values.items()
                 if a.program.by_uid(uid).op == 'kimi_k3_kda_attention')
    assert int(value_of(entry.bytes_at(1))) == 96 * 128 * 128 * 4 + 3 * 96 * 128 * 4 * 2
