"""Corpus measurement follows declared runtime ownership, not a model allowlist."""
from scripts.dump_sol_surface import runtime_framework_inputs
from tests.snapshots import shared_program


def test_runtime_specific_corpus_measurement_selects_its_framework():
    program = shared_program('moonshotai__Kimi-K3.1-Pro')
    assert program.required_runtime_framework == 'sglang'
    assert runtime_framework_inputs(program.resolved) == {'framework': 'sglang'}


def test_ordinary_corpus_measurement_preserves_existing_framework_default():
    program = shared_program('moonshotai__Kimi-K3')
    assert program.required_runtime_framework is None
    assert runtime_framework_inputs(program.resolved) == {}


def test_encoder_only_walk_does_not_claim_decoder_runtime_requirement():
    program = shared_program('moonshotai__Kimi-K3.1-Pro').encoding('vision')
    assert program.required_runtime_framework is None
