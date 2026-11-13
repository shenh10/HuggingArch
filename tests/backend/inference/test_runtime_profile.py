"""Architectural walks must not reuse serving-runtime allocation contracts."""
from dataclasses import replace

from backend.arch.eval.transforms import Deployment
from backend.arch.layout.persistent import physical_artifacts, RUNTIME_ALLOCATION
from backend.arch.runtime_profile import RuntimeProfile
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import StepCells
from backend.inference.walk_registry import WalkCellKey
from tests.snapshots import shared_program


def test_architecture_profile_has_no_sglang_allocations():
    deployment = Deployment(
        shared_program('moonshotai__Kimi-K3.1-Pro'),
        config=ParallelConfig(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=4, ep_moe=2),
        runtime_profile=RuntimeProfile.ARCHITECTURE,
    ).at(batch=1, input_tokens=128)
    analysis = deployment.analyze()
    assert not analysis.rule.sparse_dcp.storage_plans
    assert not any(a.surface == RUNTIME_ALLOCATION for a in physical_artifacts(analysis))
    assert not deployment.specializes_verification


def test_runtime_profile_is_part_of_walk_identity_and_cell_input():
    key = WalkCellKey('fixture', 'version', '{}', 'parallel', 'precision')
    assert key.key() != replace(key, runtime_profile=RuntimeProfile.ARCHITECTURE).key()
    cells = StepCells(RuntimeProfile.ARCHITECTURE)
    cell = cells.cell(object(), ParallelConfig(), phase='prefill')
    assert cell.runtime_profile == RuntimeProfile.ARCHITECTURE


def test_architecture_dcp_does_not_specialize_sparse_verification():
    deployment = Deployment(
        shared_program('moonshotai__Kimi-K3.1-Pro'),
        config=ParallelConfig(tp_attention=8, dcp=8),
        runtime_profile=RuntimeProfile.ARCHITECTURE,
    ).during('decode')
    assert not deployment.specializes_verification
