"""SGLang's packed-MXFP4 launch is chosen from runtime, not checkpoint, facts.

The two scheme-matched declarations (`mxfp4_pack_quantized_trtllm_mxfp8` and
`_weight_only`) answer the runner and precision the deployment states, not a
model name: the auto runner resolves to `flashinfer_mxfp4` per model override
on SM100, an explicit `flashinfer_mxfp4` reaches the same launch on any
model, and every other runner/precision is the BF16 weight-only path.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from backend.arch.eval.precision import AS_SHIPPED, ResolvedPrecision
from backend.inference.framework_bindings import (
    DispatchCondition,
    ServingDeployment,
    _conditions_can_overlap,
    _precision_launches,
    get_binding,
)
from tests.backend.inference.test_precision_launch_selection import _coordinate
from tests.snapshots import shared_program

MOONSHOT = "moonshotai__Kimi-K3"
MIMO = "XiaomiMiMo__MiMo-V2.6-Pro-RL"


def _selection(*, stem=MOONSHOT, sku="B300", flags=None):
    program = shared_program(stem)
    deployment = ServingDeployment.at(
        gpu_sku=sku, activation_dtype="BF16", server_flags=flags)
    binding = get_binding("sglang").deployed(deployment)
    precision = ResolvedPrecision(AS_SHIPPED, program).with_launches(
        binding.precision_launches_for(program),
        peak_families=deployment.peak_families,
        unavailable_launches=binding.unavailable_precision_launches_for(
            program))
    return binding, precision, _coordinate(stem).formats


@pytest.mark.parametrize("stem", [MIMO, MOONSHOT],
                         ids=["mimo-explicit", "moonshot-explicit"])
def test_an_explicit_flashinfer_mxfp4_backend_selects_mxfp8(stem):
    """The explicit backend reaches the SM100 trtllm-gen runner on any model;
    only `auto` reads the per-model override."""
    for sku in ("B200", "B300"):
        _, precision, formats = _selection(
            stem=stem, sku=sku,
            flags={"moe_runner_backend": "flashinfer_mxfp4"})
        launch = precision._launch_for(None, "", formats)
        assert launch is not None
        assert launch.name == "mxfp4_pack_quantized_trtllm_mxfp8"
        assert precision.compute_dtype(None, "", formats) == "F8_E4M3"
        assert precision.mma_dtypes(None, "", formats) == ("F8_E4M3", "MXFP4")
        assert precision.required_format(
            None, "", formats).name == "fp8_token_group32_ue8m0"


@pytest.mark.parametrize("stem", [MOONSHOT, "moonshotai__Kimi-K3.1-Pro", MIMO])
def test_the_auto_runner_resolves_to_mxfp8_on_sm100(stem):
    """The Kimi-K3 and MiMo-V2 SM100 overrides both resolve auto to
    flashinfer_mxfp4, so the scheme-matched default is MXFP8 — not a
    MiMo-excluding Kimi-only entry."""
    for sku in ("B200", "B300"):
        _, precision, formats = _selection(stem=stem, sku=sku)
        launch = precision._launch_for(None, "", formats)
        assert launch is not None
        assert launch.name == "mxfp4_pack_quantized_trtllm_mxfp8"
        assert precision.compute_dtype(None, "", formats) == "F8_E4M3"


def test_bf16_precision_keeps_the_operand_bf16():
    _, precision, formats = _selection(
        flags={"flashinfer_mxfp4_moe_precision": "bf16"})
    launch = precision._launch_for(None, "", formats)
    assert launch.name == "mxfp4_pack_quantized_weight_only"
    assert precision.required_format(None, "", formats) is None
    assert precision.compute_dtype(None, "", formats) == "BF16"
    assert precision.mma_dtypes(None, "", formats) == ("BF16",)


def test_below_sm100_is_the_weight_only_path():
    _, precision, formats = _selection(sku="H100-SXM")
    launch = precision._launch_for(None, "", formats)
    assert launch.name == "mxfp4_pack_quantized_weight_only"
    assert precision.compute_dtype(None, "", formats) == "BF16"


@pytest.mark.parametrize("flags,sku", [
    ({"flashinfer_mxfp4_moe_precision": "fp8"}, "B300"),
])
def test_a_non_flashinfer_backend_or_fp8_precision_has_no_mxfp8_launch(
        flags, sku):
    _, precision, formats = _selection(sku=sku, flags=flags)
    assert precision._launch_for(None, "", formats) is None


def test_conjunctive_server_flags_round_trip_and_disjointness():
    raw = {"server_flag": [
        {"name": "moe_runner_backend", "default": "auto",
         "value": ["auto", "flashinfer_mxfp4"]},
        {"name": "flashinfer_mxfp4_moe_precision", "default": "default",
         "value": "default"},
    ]}
    when = DispatchCondition.parse(raw, "test")
    assert DispatchCondition.parse(when.as_dict(), "roundtrip") == when
    assert when.holds_for(ServingDeployment.of())
    assert not when.holds_for(
        ServingDeployment.of(server_flags={"flashinfer_mxfp4_moe_precision": "bf16"}))
    assert not when.holds_for(
        ServingDeployment.of(server_flags={"moe_runner_backend": "marlin"}))
    other = DispatchCondition.parse({"server_flag": {
        "name": "flashinfer_mxfp4_moe_precision", "default": "default",
        "value": "bf16"}}, "other")
    assert not _conditions_can_overlap(when, other)
    # A subset backend any-of still overlaps the wider one.
    subset = DispatchCondition.parse({"server_flag": {
        "name": "moe_runner_backend", "default": "auto", "value": "auto"}},
        "subset")
    assert _conditions_can_overlap(when, subset)


def test_precision_launches_refuse_unanswerable_match_conditions():
    with pytest.raises(ValueError, match="whole-program"):
        _precision_launches({"bad": {
            "schemes": ["mxfp4-pack-quantized"],
            "when": {"head_dim": [128]},
            "launch": {"activation": "none", "compute_dtype": "BF16",
                       "mma_dtypes": ["BF16"]},
            "source_anchor": "src/quant.py:1",
        }}, "test")


def test_precision_phase_gate_requires_matching_walk():
    binding, _, _ = _selection()
    launch = binding.precision_launches[0]
    launch = replace(launch, whens=tuple(
        replace(when, graph=replace(when.graph, phase=("decode",)))
        for when in launch.whens))
    binding = replace(binding, precision_launches=(launch,))
    program = shared_program(MOONSHOT)
    assert binding.precision_launches_for(program) == ()
    prefill = SimpleNamespace(rule=SimpleNamespace(phase="prefill"))
    decode = SimpleNamespace(rule=SimpleNamespace(phase="decode"))
    assert binding.precision_launches_for(program, prefill) == ()
    assert binding.unavailable_precision_launches_for(program, prefill) == ()
    assert binding.precision_launches_for(program, decode) == (launch,)
@pytest.mark.parametrize("stem,sku,flags", [
    (MOONSHOT, "A100-80G-SXM", {"moe_runner_backend": "marlin"}),
    (MIMO, "B200", {"moe_a2a_backend": "deepep"}),
    (MIMO, "B200", {"moe_runner_backend": "flashinfer_mxfp4", "moe_a2a_backend": "deepep"}),
    (MIMO, "H100-SXM", {"moe_a2a_backend": "megamoe"}),
    (MOONSHOT, "B300", {"moe_a2a_backend": "megamoe"}),
    (MIMO, "B200", {"moe_runner_backend": "flashinfer_mxfp4", "moe_a2a_backend": "megamoe"}),
    (MOONSHOT, "H100-SXM", {"moe_runner_backend": "flashinfer_mxfp4", "flashinfer_mxfp4_moe_precision": "fp8"}),
    (MOONSHOT, "A100-80G-SXM", {"moe_runner_backend": "flashinfer_mxfp4"}),
])
def test_other_dispatch_paths_do_not_claim_a_flashinfer_precision(stem, sku, flags):
    _, precision, formats = _selection(stem=stem, sku=sku, flags=flags)
    assert precision._launch_for(None, "", formats) is None

@pytest.mark.parametrize("sku", ["B300", "H100-SXM"])
def test_supported_marlin_runs_weight_only(sku):
    _, precision, formats = _selection(sku=sku, flags={"moe_runner_backend": "marlin"})
    assert precision._launch_for(None, "", formats).name == "mxfp4_pack_quantized_weight_only"
