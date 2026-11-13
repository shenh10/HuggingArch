"""Sparse DCP scenario controls must remain visible in public results."""
from backend.inference.assumptions import deployment_assumptions
from backend.inference.parallel_config import ParallelConfig


def test_sparse_dcp_assumptions_are_explicit_for_default_and_adverse_scenarios():
    parallel = ParallelConfig(tp_attention=8, tp_dense_ffn=8,
                              tp_moe_expert=4, ep_moe=2, dcp=8)
    for rho, failed in ((1, 0), (8, 8)):
        rows = {row['id']: row for row in deployment_assumptions(
            parallel, sparse_dcp=True, dcp_selected_load_factor=rho,
            dcp_failed_owners=failed)}
        assert rows['sparse_dcp_selected_ownership']['value'] == rho
        assert rows['sparse_dcp_candidate_fallback']['value'] == failed
        assert rows['sparse_dcp_runtime_profile']['value'] == 'sglang_fi_a2a_push1536'
    assert deployment_assumptions(parallel) == []
