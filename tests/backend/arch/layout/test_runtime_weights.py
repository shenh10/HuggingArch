"""SGLang allocation ownership, including views into gathered Q weights."""
from dataclasses import replace
from math import prod

import pytest

from backend.arch.eval.transforms import Deployment
from backend.arch.layout.persistent import physical_artifacts, resident_weights, RUNTIME_ALLOCATION
from backend.arch.layout.runtime_weights import join_runtime_allocations
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import shared_program


@pytest.fixture(scope='module')
def artifacts():
    program = shared_program('moonshotai__Kimi-K3.1-Pro')
    analysis = Deployment(program, config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).at(batch=1, input_tokens=128).analyze()
    return physical_artifacts(analysis)


def test_real_k31_separate_allocations_and_q_alias(artifacts):
    extra = [row for row in artifacts if row.surface == RUNTIME_ALLOCATION]
    base = tuple(row for row in artifacts if row.surface != RUNTIME_ALLOCATION)
    assert sum(row.resident_bytes_per_rank for row in extra) == 2_170_552_320
    assert {row.artifact_name for row in extra} == {
        'w_kc', 'w_vc', 'q_b_proj_qrep_weight', 'w_kc_qrep', 'w_vc_qrep', '_dcp_v_weights'}
    assert all(row.saved_bytes_per_rank == 0 and not row.is_parameter for row in extra)
    assert resident_weights(artifacts).parameters == resident_weights(base).parameters
    q = next(row for row in extra if row.artifact_name == 'q_b_proj_qrep_weight')
    assert q.runtime_shape == (18432, 1536)
    assert q.resident_bytes_per_instance == 56_623_104 - 7_077_888
    assert q.runtime_allocation.aliases_source
    v = next(row for row in extra if row.artifact_name == '_dcp_v_weights')
    assert v.runtime_shape == (96, 512, 128)
    assert v.resident_bytes_per_instance == 12_582_912


def test_full_execution_q_is_not_counted_twice(artifacts):
    base = tuple(row for row in artifacts if row.surface != RUNTIME_ALLOCATION)
    allocations = [row.runtime_allocation for row in artifacts if row.surface == RUNTIME_ALLOCATION]
    aliases = {item.source: item for item in allocations if item.aliases_source}
    # Decode executes the gathered Q projection while prefill executes its view.
    decoded = tuple(replace(row, runtime_shape=aliases[row.key].runtime_shape,
                            resident_bytes_per_instance=2 * prod(aliases[row.key].runtime_shape),
                            resident_bytes_per_rank=2 * prod(aliases[row.key].runtime_shape) * row.runtime_instances)
                    if row.key in aliases else row for row in base)
    joined = join_runtime_allocations(decoded, allocations)
    assert sum(row.resident_bytes_per_rank for row in joined) == sum(row.resident_bytes_per_rank for row in artifacts)
    assert all(row.resident_bytes_per_instance == 0 for row in joined
               if row.surface == RUNTIME_ALLOCATION and row.runtime_allocation.aliases_source)


def test_aliases_require_unique_unpacked_source(artifacts):
    row = next(row for row in artifacts if row.surface == RUNTIME_ALLOCATION and row.runtime_allocation.aliases_source)
    source = next(source for source in artifacts if source.key == row.runtime_allocation.source)
    allocation = row.runtime_allocation
    with pytest.raises(ValueError, match='duplicate runtime allocation'):
        join_runtime_allocations((source,), (allocation, allocation))
    with pytest.raises(ValueError, match='multiple runtime allocations alias'):
        join_runtime_allocations((source,), (allocation, replace(allocation, name='other')))
    with pytest.raises(NotImplementedError, match='unpacked'):
        join_runtime_allocations((replace(source, resident_storage_dtype='U8', resident_packing_factor=2),), (allocation,))


@pytest.mark.parametrize('phase,weight_bytes', [('prefill', 1_572_864), ('decode', 12_582_912)])
def test_real_projection_reads_runtime_absorbed_weights(phase, weight_bytes):
    from backend.arch.eval.dim import value_of
    from backend.arch.layout.physical_cost import PhysicalProjection, physical_access
    program = shared_program('moonshotai__Kimi-K3.1-Pro')
    analysis = Deployment(program, config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).at(batch=1, input_tokens=128).during(phase).analyze()
    projection = PhysicalProjection(analysis)
    access = physical_access(analysis, scale_dtype_bytes=4, projection=projection)
    for plan in analysis.rule.sparse_dcp.plans:
        for role in ('absorbed_query', 'value_projection'):
            row = projection.traffic.values[plan.nodes[role].uid]
            assert value_of(row.weight_reads + row.borrowed_weight_reads) == weight_bytes
            assert access.values[plan.nodes[role].uid].weight_bytes_per_rank == weight_bytes
    # Both phases share the same physical allocations despite different Q execution.
    assert sum(row.resident_bytes_per_rank for row in projection.artifacts) == sum(
        row.resident_bytes_per_rank for row in physical_artifacts(
            Deployment(program, config=analysis.rule.parallel).at(batch=1, input_tokens=128).analyze()))


@pytest.mark.parametrize('context,factor,selected', [(8193, 1, 256), (8193, 2, 512), (1025, 2, 129)])
def test_sparse_reads_keep_full_context_capacity(context, factor, selected):
    from backend.arch.eval import cache
    from backend.arch.eval.dim import value_of
    from backend.arch.layout.physical_cost import PhysicalProjection
    analysis = Deployment(shared_program('moonshotai__Kimi-K3.1-Pro'), config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).at(batch=1, input_tokens=context, dcp_selected_load_factor=factor).during('decode').analyze()
    projection = PhysicalProjection(analysis)
    capacity = cache.derive(analysis, geometry=projection.cache_geometry)
    for plan in analysis.rule.sparse_dcp.plans:
        assert value_of(projection.traffic.values[plan.nodes['score'].uid].kv_reads) == 656 * selected
        assert value_of(projection.traffic.values[plan.nodes['indexer_logits'].uid].kv_reads) == 132 * ((context + 7) // 8)
        for tensor in capacity.values[plan.component.uid].tensors:
            assert value_of(tensor.positions(context)) == ((context + 7) // 8) * 8
            assert value_of(tensor.bytes_at(context)) == value_of(tensor.position_bytes) * ((context + 7) // 8)


@pytest.mark.parametrize('framework', [None, 'reference', 'vllm'])
@pytest.mark.parametrize('dcp', [1, 8])
def test_sparse_runtime_plan_refuses_non_sglang_serving(framework, dcp):
    from types import SimpleNamespace
    from backend.inference.serving import build_serving_layers
    plan = object()
    plans = SimpleNamespace(plans=(plan,) if dcp > 1 else (), storage_plans=(plan,))
    analysis = SimpleNamespace(rule=SimpleNamespace(sparse_dcp=plans),
                               program=SimpleNamespace(required_runtime_framework="sglang"))
    cell = SimpleNamespace(built_walk=lambda: analysis)
    binding = SimpleNamespace(framework=framework) if framework else None
    with pytest.raises(NotImplementedError, match='require the sglang framework profile'):
        build_serving_layers(cell, framework_binding=binding, stage=None, timed=frozenset())


@pytest.mark.parametrize('context,factor', [(8192, 1), (6, 1), (2046, 2)])
def test_verification_cache_reads_sum_each_causal_query(context, factor):
    from backend.arch.eval.dim import value_of
    from backend.arch.layout.physical_cost import PhysicalProjection
    analysis = Deployment(shared_program('moonshotai__Kimi-K3.1-Pro'), config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).at(batch=1, input_tokens=context, s_verify=5, dcp_selected_load_factor=factor).during('decode').analyze()
    projection = PhysicalProjection(analysis)
    selected_pairs = sum(min((min(2048, length) * factor + 7) // 8,
                             (length + 7) // 8)
                         for length in range(context, context + 5))
    plan = analysis.rule.sparse_dcp.plans[0]
    assert value_of(projection.traffic.values[plan.nodes['score'].uid].kv_reads) == 656 * selected_pairs
    assert value_of(projection.traffic.values[plan.nodes['apply'].uid].kv_reads) == 528 * selected_pairs


def test_reference_preproject_rejects_f16_dense_weights():
    from backend.arch.precision_profile import PrecisionProfile
    precision = PrecisionProfile.canonicalize({'weights': [
        {'select': {'roles': ['attention']}, 'format': {'dtype': 'F16'}}]})
    analysis = Deployment(shared_program('moonshotai__Kimi-K3.1-Pro'), config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    ), precision=precision).at(batch=1, input_tokens=128).during('decode').analyze()
    with pytest.raises(NotImplementedError, match='requires BF16 dense weights'):
        physical_artifacts(analysis)


def test_dcp_verification_widths_do_not_share_compiled_cells():
    from backend.inference.step_eval import StepCells
    cells = StepCells()
    spec = object()
    parallel = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8, dcp=8)
    first = cells.cell(spec, parallel, phase='decode', speculative=True, verify_width=5)
    assert first is cells.cell(spec, parallel, phase='decode', speculative=True, verify_width=5)
    assert first is not cells.cell(spec, parallel, phase='decode', speculative=True, verify_width=3)


def test_public_step_cell_keeps_fixed_verification_width():
    from backend.arch.eval.dim import value_of
    from backend.inference.step_eval import evaluate_step, StepCells
    from tests.snapshots import resolve_snapshot
    resolved = resolve_snapshot('moonshotai__Kimi-K3.1-Pro')
    parallel = ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8)
    cells = StepCells()
    steps = [evaluate_step(resolved, parallel, phase='decode', batch=1,
                           input_tokens=8192, output_tokens=0, s_verify=width, cells=cells)
             for width in (3, 5)]
    assert steps[0].cell is not steps[1].cell
    for step in steps:
        assert value_of(step.walk.binding['n_verify']) == step.s_verify
        # Builds invocation laws against symbolic context without int(symbol).
        with step.walk.assumed():
            assert step.cell.physical().traffic.values


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_dcp1_keeps_local_absorbed_copies_without_execution_plan(phase):
    from backend.arch.layout.physical_cost import PhysicalProjection
    analysis = Deployment(shared_program('moonshotai__Kimi-K3.1-Pro'), config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2,
    )).at(batch=1, input_tokens=128).during(phase).analyze()
    assert not analysis.rule.sparse_dcp.plans
    assert analysis.rule.sparse_dcp.storage_plans
    projection = PhysicalProjection(analysis)
    allocations = [row for row in projection.artifacts if row.surface == RUNTIME_ALLOCATION]
    assert {row.artifact_name for row in allocations} == {'w_kc', 'w_vc'}
    assert sum(row.resident_bytes_per_rank for row in allocations) == 75_497_472
    for plan in analysis.rule.sparse_dcp.storage_plans:
        assert not plan.decode and not plan.replicate_q and not plan.preproject_v
        for role in ('absorbed_query', 'value_projection'):
            from backend.arch.eval.dim import value_of
            assert value_of(projection.traffic.values[plan.nodes[role].uid].weight_reads) == 1_572_864


@pytest.mark.parametrize('context', [17, 8193])
def test_sparse_prefill_nondivisible_cache_is_gathered_for_computation(context):
    from backend.arch.eval import cache
    from backend.arch.eval.dim import value_of
    from backend.arch.layout.physical_cost import PhysicalProjection
    program = shared_program('moonshotai__Kimi-K3.1-Pro')
    projections = []
    for dcp in (1, 8):
        analysis = Deployment(program, config=ParallelConfig(
            tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=dcp,
        )).at(batch=1, input_tokens=context).during('prefill').analyze()
        projections.append(PhysicalProjection(analysis))
    plain, striped = projections
    for plan in striped.analysis.rule.sparse_dcp.plans:
        for role in ('score', 'apply', 'indexer_logits'):
            uid = plan.nodes[role].uid
            assert value_of(striped.cost_plan.cost.values[uid].flops) == value_of(plain.cost_plan.cost.values[uid].flops)
            assert value_of(striped.traffic.values[uid].kv_reads) == value_of(plain.traffic.values[uid].kv_reads)
        persistent = cache.derive(striped.analysis, geometry=striped.cache_geometry)
        assert all(tensor.positions_over == 8 for tensor in persistent.values[plan.component.uid].tensors)
        assert all(tensor.positions_over == 1 for tensor in striped.traffic_geometry.cache.values[plan.component.uid].tensors)


def test_striped_capacity_reader_and_formulas_use_whole_owner_slots():
    from backend.arch.layout.physical_cache import physical_cache_view
    analysis = Deployment(shared_program('moonshotai__Kimi-K3.1-Pro'), config=ParallelConfig(
        tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8,
    )).at(batch=1, input_tokens=17).during('decode').analyze()
    view = physical_cache_view(analysis)
    # KDA's fixed state does not change; all 24 DSA layers add one rank-local
    # row across 16 -> 17, then no further storage through length 24.
    assert view.storage_bytes_at(17) - view.storage_bytes_at(16) == 24 * (656 + 132)
    assert view.storage_bytes_at(17) == view.storage_bytes_at(24)
    assert sum(row['request_bytes'] for row in view.storage_rows(17)) == view.storage_bytes_at(17)
    assert view.request_formula(17)['result_bytes'] == view.storage_bytes_at(17)
    assert view.growth_and_floor()[0] == 24 * (656 + 132) / 8
    assert view.growth_formula['result_bytes'] == view.growth_and_floor()[0]


def test_encoder_only_program_does_not_bind_decoder_runtime_copies():
    program = shared_program('moonshotai__Kimi-K3.1-Pro')
    assert program.sparse_dcp_components
    assert not program.encoding('vision').sparse_dcp_components
