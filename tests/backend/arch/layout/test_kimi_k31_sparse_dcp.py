"""Supported sparse DCP paths keep ownership and physical cache scans separate."""
import pytest

from backend.arch.eval.analysis import Point
from backend.arch.eval.dim import value_of
from backend.arch.eval.transforms import Deployment
from backend.inference.framework_bindings import get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step
from tests.snapshots import resolve_snapshot, shared_program


def parallel():
    return ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2, dcp=8)


@pytest.mark.parametrize('context,rho,selected,scan', [
    (32769, 8, 2048, 4097), (32769, 1, 256, 4097), (129, 1, 17, 17),
])
def test_sparse_dcp_selected_ownership_and_rounded_resident_scan(context, rho, selected, scan):
    a = Deployment(shared_program('moonshotai__Kimi-K3.1-Pro'), config=parallel()).during(
        'decode').at(batch=1, input_tokens=context, dcp_selected_load_factor=rho).analyze()
    plan = a.rule.sparse_dcp.plans[0]
    assert value_of(a.kernel_shapes.values[plan.nodes['apply'].uid].ins[0][-1]) == selected
    assert value_of(a.kernel_shapes.values[plan.nodes['indexer_logits'].uid].out[-1]) == scan


def test_public_sglang_serving_rereads_sparse_communication_laws():
    step = evaluate_step(resolve_snapshot('moonshotai__Kimi-K3.1-Pro'), parallel(),
                         phase='decode', batch=1, input_tokens=32769, output_tokens=0,
                         dcp_selected_load_factor=8, dcp_failed_owners=2)
    _ = step.walk
    layers = step.cell.serving(get_binding('sglang'), None, frozenset({'lm'}))
    point = layers.at(step.reading)
    metas = [m for m in point.comm_meta.values.values()
             if m.get('sparse_dcp_protocol') == 'candidate_multicast']
    assert metas
    assert all(int(m['tensor_bytes']) == 8 * (384 + 2048) * 8 for m in metas)
    assert all(int(m['sparse_dcp_dependency_rounds']) == 2 for m in metas)
    # Re-read the same symbolic serving graph at short context and both
    # capacity-boundary batches; no second walk or materialization.
    short = layers.at(Point({**step.point, 's_input': 129, 'rho_dcp': 1,
                             'dcp_failed_owners': 0}))
    short_candidates = [m for m in short.comm_meta.values.values()
                        if m.get('sparse_dcp_protocol') == 'candidate_multicast']
    assert all(int(m['tensor_bytes']) == 8 * 384 * 8 for m in short_candidates)
    assert all(int(m['sparse_dcp_dependency_rounds']) == 1 for m in short_candidates)
    boundary = layers.at(Point({**step.point, 'B': 1536}))
    above = layers.at(Point({**step.point, 'B': 1537}))
    boundary_outputs = [m for m in boundary.comm_meta.values.values()
                        if m.get('sparse_dcp_protocol') == 'projected_push_or_fi_a2a']
    above_outputs = [m for m in above.comm_meta.values.values()
                     if m.get('sparse_dcp_protocol') == 'projected_push_or_fi_a2a']
    assert all(int(m['tensor_bytes']) == 1536 * 96 * 136 * 2 for m in boundary_outputs)
    assert all(int(m['tensor_bytes']) == 1537 * 96 * 132 * 2 for m in above_outputs)


@pytest.mark.parametrize('dcp,cp', [(4, 1), (16, 2)])
def test_sparse_dcp_refuses_unmodelled_reference_profile(dcp, cp):
    with pytest.raises(NotImplementedError, match='reference profile requires'):
        Deployment(shared_program('moonshotai__Kimi-K3.1-Pro'), config=ParallelConfig(
            tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4 * cp, ep_moe=2, dcp=dcp, cp=cp,
        )).during('decode').at(batch=1, input_tokens=32768).analyze()


def test_verify_pairs_follow_each_causal_query_not_last_query_rectangle():
    from backend.arch.layout.physical_cost import physical_cost_plan
    from backend.arch.layout.sparse_dcp_contract import sparse_dcp_query_pairs
    # Input cache6 plus5 query rows exposes lengths6,7,8,9,10. The busiest
    # striped owner holds1,1,1,2,2 positions, hence7 useful pairs, not5*2.
    assert sparse_dcp_query_pairs(10, 5, 2048, 8) == (7, 7)
    deployment = Deployment(shared_program('moonshotai__Kimi-K3.1-Pro'), config=parallel()).during(
        'decode').at(batch=1, input_tokens=6, s_verify=5)
    a = deployment.analyze()
    plan = a.rule.sparse_dcp.plans[0]
    physical = physical_cost_plan(a).cost
    assert int(physical.values[plan.nodes['score'].uid].flops) == 110592 * 7
    assert int(physical.values[plan.nodes['apply'].uid].flops) == 98304 * 7
    assert int(physical.values[plan.nodes['indexer_logits'].uid].flops) == 8288 * 7



def test_sparse_verify_symbolic_walk_rejects_another_verification_width():
    step = evaluate_step(resolve_snapshot('moonshotai__Kimi-K3.1-Pro'), parallel(),
                         phase='decode', batch=1, input_tokens=6, output_tokens=0, s_verify=5)
    walk = step.walk
    with pytest.raises(ValueError, match='verification width requires a new walk'):
        walk.admit({**step.point, 's_verify': 3})



def test_public_runtime_scheduler_reprices_context_and_owner_scenario():
    from backend.inference.framework_bindings import ServingDeployment
    step = evaluate_step(resolve_snapshot('moonshotai__Kimi-K3.1-Pro'), parallel(),
                         phase='decode', batch=1, input_tokens=32769, output_tokens=0)
    _ = step.walk
    binding = get_binding('sglang').deployed(
        ServingDeployment.at(gpu_sku='B300', activation_dtype='BF16'))
    layers = step.cell.serving(binding, None, frozenset({'lm'}))
    for context, rho, splits in ((32769, 1, 4), (32769, 8, 32), (129, 1, 0)):
        point = layers.at(Point({**step.point, 's_input': context, 'rho_dcp': rho}))
        metas = [m for m in point.comm_meta.values.values()
                 if m.get('runtime_stage_name') == 'flashmla_split_combine']
        assert metas
        assert all(int(m['runtime_read_bytes']) == splits * 96 * 513 * 4 + 8 for m in metas)
        assert all('unresolved' not in m['runtime_work_basis'] for m in metas)
    with pytest.raises(ValueError, match='48KiB admission'):
        rejected = layers.at(Point({**step.point, 'B': 2221}))
        list(rejected.comm_meta.values.values())
    unknown = step.cell.serving(get_binding('sglang'), None, frozenset({'lm'})).at(step.reading)
    assert any(m.get('runtime_work_basis', '').startswith('unresolved_sm_count')
               for m in unknown.comm_meta.values.values())
