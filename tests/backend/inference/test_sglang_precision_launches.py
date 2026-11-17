"""SGLang's Kimi MXFP4 launch is chosen from runtime, not checkpoint, facts."""
import pytest

from backend.arch.eval.precision import AS_SHIPPED, ResolvedPrecision
from backend.inference.framework_bindings import (
    DispatchCondition, SourceAnchor, ServingDeployment, get_binding,
    _conditions_can_overlap, _precision_launches,
)
from tests.backend.inference.test_precision_launch_selection import _coordinate
from tests.snapshots import shared_program

KIMI = "moonshotai__Kimi-K3"


def selection(*, stem=KIMI, sku="B300", flags=None):
    program = shared_program(stem)
    deployment = ServingDeployment.at(
        gpu_sku=sku, activation_dtype="BF16", server_flags=flags)
    binding = get_binding("sglang").deployed(deployment)
    precision = ResolvedPrecision(AS_SHIPPED, program).with_launches(
        binding.precision_launches_for(program),
        peak_families=deployment.peak_families,
        unavailable_launches=binding.unavailable_precision_launches_for(program))
    return binding, precision, _coordinate(stem).formats


@pytest.mark.parametrize("backend", ["auto", "flashinfer_mxfp4"])
@pytest.mark.parametrize("sku", ["B200", "B300"])
def test_kimi_default_uses_mxfp8_activations_with_mxfp4_weights(backend, sku):
    _, precision, formats = selection(sku=sku, flags={"moe_runner_backend": backend})
    assert _coordinate(KIMI).request is None  # checkpoint remains weight-only
    launch = precision._launch_for(None, "", formats)
    assert launch.name == "kimi_mxfp4_default"
    assert precision.required_format(None, "", formats).name == "fp8_token_group32_ue8m0"
    assert precision.compute_dtype(None, "", formats) == "F8_E4M3"
    assert precision.mma_dtypes(None, "", formats) == ("F8_E4M3", "MXFP4")


def test_bf16_override_keeps_the_operand_bf16():
    _, precision, formats = selection(flags={"flashinfer_mxfp4_moe_precision": "bf16"})
    assert precision._launch_for(None, "", formats).name == "kimi_mxfp4_bf16"
    assert precision.required_format(None, "", formats) is None
    assert precision.compute_dtype(None, "", formats) == "BF16"


@pytest.mark.parametrize("kwargs", [
    {"sku": "H100-SXM"},
    {"flags": {"moe_runner_backend": "marlin"}},
    {"flags": {"flashinfer_mxfp4_moe_precision": "fp8"}},
])
def test_other_deployments_do_not_inherit_the_blackwell_default(kwargs):
    _, precision, formats = selection(**kwargs)
    assert precision._launch_for(None, "", formats) is None


def test_other_pack_models_do_not_inherit_kimi_autoconfiguration():
    _, precision, formats = selection(stem="XiaomiMiMo__MiMo-V2.6-Pro-RL")
    assert precision._launch_for(None, "", formats) is None
    assert precision.launch_gaps(None, "", formats) == ()


def test_missing_program_does_not_satisfy_architecture_condition():
    binding = get_binding("sglang").deployed(ServingDeployment.at(gpu_sku="B300", activation_dtype="BF16"))
    assert binding.precision_launches_for() == ()


def test_conjunctive_server_flags_round_trip_and_disjointness():
    raw = {"server_flag": [
        {"name": "backend", "default": "auto", "value": "auto"},
        {"name": "precision", "default": "default", "value": "default"},
    ]}
    when = DispatchCondition.parse(raw, "test")
    assert DispatchCondition.parse(when.as_dict(), "roundtrip") == when
    assert when.holds_for(ServingDeployment.of())
    assert not when.holds_for(ServingDeployment.of(server_flags={"precision": "bf16"}))
    other = DispatchCondition.parse({"server_flag": {
        "name": "precision", "default": "default", "value": "bf16"}}, "other")
    assert not _conditions_can_overlap(when, other)


def test_precision_launches_refuse_unanswerable_match_conditions():
    with pytest.raises(ValueError, match="whole-program"):
        _precision_launches({"bad": {
            "schemes": ["mxfp4-pack-quantized"],
            "when": {"head_dim": [128]},
            "launch": {"activation": "none", "compute_dtype": "BF16", "mma_dtypes": ["BF16"]},
            "source_anchor": "src/quant.py:1",
        }}, "test")


def test_per_citation_revision_preserves_the_audited_source():
    binding = get_binding("sglang")
    assert binding.audited_at.tag == "v0.5.22"
    anchor = binding.precision_launches[0].source_anchor
    cited = binding.audited_at.cite(anchor)
    assert "".join(segment["text"] for segment in cited["segments"]) == anchor.text
    links = [segment["url"] for segment in cited["segments"] if "url" in segment]
    assert links
    assert all("/1b2870d33de7b5f31844fd74e97eea6d7896c53a/" in url for url in links)
    legacy = SourceAnchor.parse("python/sglang/srt/layers/attention/dsa_backend.py:2741", "test")
    url = binding.audited_at.cite(legacy)["segments"][0]["url"]
    assert url.endswith("/baa9befa381511cd4d56f83980951e5a45fffc6d/"
                        "python/sglang/srt/layers/attention/dsa_backend.py#L2741")


def test_precision_phase_gate_requires_matching_walk():
    from dataclasses import replace
    from types import SimpleNamespace

    binding, _, _ = selection()
    launch = binding.precision_launches[0]
    launch = replace(launch, whens=tuple(
        replace(when, graph=replace(when.graph, phase=("decode",)))
        for when in launch.whens))
    binding = replace(binding, precision_launches=(launch,))
    program = shared_program(KIMI)
    assert binding.precision_launches_for(program) == ()
    prefill = SimpleNamespace(rule=SimpleNamespace(phase="prefill"))
    decode = SimpleNamespace(rule=SimpleNamespace(phase="decode"))
    assert binding.precision_launches_for(program, prefill) == ()
    assert binding.unavailable_precision_launches_for(program, prefill) == ()
    assert binding.precision_launches_for(program, decode) == (launch,)
