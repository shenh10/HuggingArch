"""Ordered precision-launch selection against the card capability table."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import pytest

from backend.arch.eval.precision import AS_SHIPPED, ResolvedPrecision
from backend.arch.eval.layer import Layer
from backend.arch.eval.transforms import Deployment
from backend.inference.gpu_specs import get_gpu, peak_flops_for_dtype
from backend.inference.framework_bindings import (
    ServingDeployment,
    get_binding,
)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from backend.inference.step_eval import StepCells, evaluate_step
from tests.snapshots import SNAPSHOT_DIR, shared_program, shared_walk
from tests.snapshots import resolve_snapshot

# The corpus classes are named by snapshot. Their scheme and activation
# request are *not* spelled here: `_coordinate` reads both from the resident
# format that the snapshot's own walk materializes.
_DOT_STEMS = (
    "deepseek-ai__DeepSeek-V4-Flash",
    "deepseek-ai__DeepSeek-V4-Pro",
    "deepseek-ai__DeepSeek-V4.1-Flash",
)
_PACK_STEMS = (
    "XiaomiMiMo__MiMo-V2.6-Pro-RL",
    "moonshotai__Kimi-K3",
)
_GPT_OSS_STEM = "openai__gpt-oss-20b"
_ALL_STEMS = (*_DOT_STEMS, *_PACK_STEMS, _GPT_OSS_STEM)
# SGLang-only model coordinates are covered in test_sglang_precision_launches.
_SGLANG_STEMS = ("moonshotai__Kimi-K3.1-Pro",)


@dataclass(frozen=True)
class _WalkCoordinate:
    """One real resident MXFP4 coordinate and the leaf formats that carry it."""

    scheme: Any
    request: str | None
    formats: tuple[Any, ...]


def _is_mxfp4(rep: Any) -> bool:
    # MX identity comes from the scheme declaration, not from a hand-written
    # scheme string or an operand-width inference.
    return "mx_e2m1" in tuple(rep.resident.scheme.numeric_format)


@lru_cache(maxsize=None)
def _coordinate(stem: str) -> _WalkCoordinate:
    program = shared_program(stem)
    analysis = shared_walk(
        Deployment(program).at(batch=1, input_tokens=16))
    # `QuantScheme` carries config mappings, so its walked `name` is the dict
    # key; the real scheme object itself remains on `_WalkCoordinate`.
    coordinates: dict[tuple[str, str | None], tuple[Any, ...]] = {}
    for reps in analysis.formats.values.values():
        mxfp4 = tuple(rep for rep in reps
                      if rep.resident.scheme is not None
                      and _is_mxfp4(rep))
        if not mxfp4:
            continue
        leaf_coordinates = {
            (rep.resident.scheme.name, rep.resident.activation_format)
            for rep in mxfp4
        }
        assert len(leaf_coordinates) == 1, (stem, leaf_coordinates)
        coordinate = next(iter(leaf_coordinates))
        coordinates.setdefault(coordinate, mxfp4)
    if len(coordinates) != 1:
        raise AssertionError(
            f"{stem}: expected one MXFP4 scheme/request coordinate, got "
            f"{sorted(coordinates, key=repr)}")
    (_scheme_name, request), formats = next(iter(coordinates.items()))
    scheme = formats[0].resident.scheme
    return _WalkCoordinate(scheme, request, formats)


def _precision(stem: str, sku: str | None, *, peaks=(), flags=None,
               compute_capability: str | None = None):
    if sku is not None:
        deployment = ServingDeployment.at(
            gpu_sku=sku, activation_dtype="BF16", server_flags=flags)
        peaks = deployment.peak_families
    else:
        deployment = ServingDeployment.of(
            device="cuda" if compute_capability is not None else None,
            compute_capability=compute_capability,
            server_flags=flags,
            peak_families=tuple(peaks))
    binding = get_binding("vllm").deployed(deployment)
    precision = ResolvedPrecision(AS_SHIPPED, shared_program(stem))
    return binding, precision.with_launches(
        binding.precision_launches_for(),
        peak_families=tuple(peaks),
        unavailable_launches=binding.unavailable_precision_launches_for())


def _selected(stem: str, **kwargs):
    binding, precision = _precision(stem, **kwargs)
    launch = precision._launch_for(
        None, "", _coordinate(stem).formats)
    return binding, precision, launch


def _mxfp4_snapshots():
    return {
        path.stem for path in SNAPSHOT_DIR.glob("*.json")
        if any(marker in path.read_text(encoding="utf-8").lower()
               for marker in ("mxfp4", "float4", "e2m1"))
    }


def _with_fp8_group128_request(formats):
    """gpt-oss's real `mxfp4` coordinate, with an FP8 request materialized.

    The resident weight/scheme stays exactly what the snapshot walked.  The
    synthetic intent only states the activation treatment a checkpoint may
    declare, which is the coordinate the Humming ladder owns.
    """
    return tuple(
        replace(group, segments=tuple(
            replace(segment, intent=replace(
                segment.resident,
                activation_format="fp8_token_group128_ue8m0"))
            for segment in group))
        for group in formats)


def _with_fp8_group128_segment(segment):
    return replace(
        segment,
        intent=replace(segment.resident,
                       activation_format="fp8_token_group128_ue8m0"))


def _gpt_oss_step_with_fp8_request():
    """A real serving cell whose gpt-oss MXFP4 weights request block FP8.

    The graph, scheme and checkpoint format resolution all stay real.  Only
    the resident format's activation intent is synthetic, exactly the input
    coordinate the new audited launch declares.  The cell's cached symbolic
    walk is replaced before any serving layer is built, so selector, access
    and per-op pricing all read this one coordinate.
    """
    resolved = resolve_snapshot(_GPT_OSS_STEM)
    step = evaluate_step(
        resolved, ParallelConfig(), phase="prefill", batch=1,
        input_tokens=16, output_tokens=0, cells=StepCells())
    walk = step.walk
    values = {}
    for uid, groups in walk.formats.values.items():
        values[uid] = tuple(
            replace(group, segments=tuple(
                _with_fp8_group128_segment(segment)
                if (segment.resident.scheme is not None
                    and segment.resident.scheme.name == "mxfp4")
                else segment
                for segment in group))
            for group in groups)
    step.cell._walk = replace(
        walk,
        formats=Layer(walk.formats.program, values))
    return resolved, step


@pytest.mark.parametrize("sku,expected", [
    ("H100-SXM", "mxfp4_dot_suffix_weight_only"),
    ("B200", "mxfp4_dot_suffix_trtllm_mxfp8"),
])
def test_dot_suffix_matrix_is_bit_for_bit_with_the_sm_selector(
        sku, expected):
    """All three dot-suffix snapshots answer from their own walked request.

    The new selector reaches the old Hopper W4A16 and Blackwell TRT-LLM
    answers from capability columns and ordered fallback, not from an upper
    SM bound on the TRT-LLM entry.
    """
    for stem in _DOT_STEMS:
        _binding, _precision, launch = _selected(stem, sku=sku)
        assert launch is not None
        assert launch.name == expected


@pytest.mark.parametrize("stem", _PACK_STEMS)
@pytest.mark.parametrize("sku,expected", [
    ("H100-SXM", "mxfp4_pack_quantized_weight_only"),
    ("B200", None),
])
def test_pack_quantized_matrix_is_bit_for_bit_with_the_sm_selector(
        stem, sku, expected):
    """The two real pack-quantized snapshots stay W4A16/gap respectively."""
    coordinate = _coordinate(stem)
    binding, precision, launch = _selected(stem, sku=sku)
    if expected is None:
        assert launch is None
        assert precision.launch_gap(
            None, "", coordinate.formats).startswith(
                "missing_deployment_fact:mxfp4_pack_quantized_weight_only")
        assert binding.unavailable_precision_launches_for()
    else:
        assert launch is not None
        assert launch.name == expected


def test_gpt_oss_is_a_separate_weight_only_coordinate():
    """gpt-oss is not relabeled into the compressed-tensors pack family.

    vLLM recognizes its checkpoint `mxfp4` as `gpt_oss_mxfp4` and invokes the
    GPT-OSS selector without an activation key. That selector filters to BF16
    activation variants: TRT-LLM BF16 on CUDA family 100, or the OAI Triton
    W4A16 path on Hopper. The walk therefore has no activation request, and
    neither card turns it into a selector gap.
    """
    coordinate = _coordinate(_GPT_OSS_STEM)
    assert coordinate.request is None
    assert coordinate.scheme.name not in {
        _coordinate(stem).scheme.name for stem in _PACK_STEMS
    }
    for sku in ("H100-SXM", "B200"):
        _binding, precision, launch = _selected(_GPT_OSS_STEM, sku=sku)
        assert launch is None
        assert precision.launch_gaps(None, "", coordinate.formats) == ()
        assert precision.required_format(None, "", coordinate.formats) is None
        assert precision.compute_dtype(None, "", coordinate.formats) == "BF16"


def test_an_fp8_gpt_oss_coordinate_selects_the_audited_humming_ladder():
    """The explicit Humming path is a subset-capable launch, not a name guess.

    H100 has FP8 and BF16 but no FP4 column.  The default TRT/DeepGEMM mixed
    launches do not match this distinct `mxfp4` scheme; the FP8-requested
    Humming declaration does, and its audited ladder keeps the launch while
    pricing later promotes MXFP4 to F8.
    """
    coordinate = _coordinate(_GPT_OSS_STEM)
    formats = _with_fp8_group128_request(coordinate.formats)
    _binding, precision, default_launch = _selected(
        _GPT_OSS_STEM, sku="H100-SXM",
        flags={"moe_backend": "humming"})
    assert default_launch is None
    selected_launch = precision._launch_for(None, "", formats)
    assert selected_launch is not None
    assert selected_launch.name == "mxfp4_humming_hopper_w4a8_ladder"
    assert selected_launch.promotion_ladder is True
    assert selected_launch.mma_dtypes == ("F8_E4M3", "MXFP4")
    assert precision._launch_for(None, "", formats) is selected_launch
    assert precision.launch_gaps(None, "", formats) == ()
    assert precision.required_format(
        None, "", formats).name == "fp8_token_group128_ue8m0"
    assert precision.mma_dtypes(None, "", formats) == ("F8_E4M3", "MXFP4")
    assert precision.launch_promotion_ladder(None, "", formats) is True


def test_a_ladder_with_no_declared_family_is_skipped():
    """A promotion ladder still needs one card-supported rung."""
    coordinate = _coordinate(_GPT_OSS_STEM)
    formats = _with_fp8_group128_request(coordinate.formats)
    _binding, precision, launch = _selected(
        _GPT_OSS_STEM, sku=None, peaks=("bf16",),
        compute_capability="9.0", flags={"moe_backend": "humming"})
    assert launch is None
    assert precision.launch_gap(None, "", formats) == (
        "missing_card_mma:mxfp4_humming_hopper_w4a8_ladder:"
        "F8_E4M3+MXFP4")
    # No launch was selected, so the old no-declaration fallback owns the
    # pricing answer and will drop the conversion to the BF16 edge.
    assert precision.mma_dtypes(None, "", formats) == ("F8_E4M3",)
    assert precision.launch_promotion_ladder(None, "", formats) is False


def test_humming_ladder_prices_end_to_end_at_f8_on_h100():
    """Selector, access and the real per-op peak closure read one law."""
    resolved, step = _gpt_oss_step_with_fp8_request()
    gpu = get_gpu("H100-SXM")
    binding = get_binding("vllm").deployed(ServingDeployment.at(
        gpu_sku="H100-SXM", activation_dtype="BF16",
        server_flags={"moe_backend": "humming"}))
    result = analyze(
        resolved, parallel=step.parallel,
        framework_binding=binding, step_eval=step,
        peak_flops_per_sec=peak_flops_for_dtype(gpu, "BF16"),
        peak_hbm_bw_bytes_per_sec=3.35e12,
        gpu=gpu, breakdown=True)

    expert_rows = []
    expert_uids = set()
    for uid, row in result.access.values.items():
        if row.weight_dtype == "MXFP4":
            expert_rows.append(row)
            expert_uids.add(uid)
    assert expert_rows
    assert all(row.promotion_ladder for row in expert_rows)
    assert all(row.mma_dtypes == ("F8_E4M3", "MXFP4")
               for row in expert_rows)
    assert all(row.launch_gap is None for row in expert_rows)

    timing = [row for row in result.per_op_timing
              if row["uid"] in expert_uids]
    assert timing
    fp8_peak = peak_flops_for_dtype(gpu, "F8_E4M3")
    bf16_peak = peak_flops_for_dtype(gpu, "BF16")
    assert fp8_peak != bf16_peak
    assert all(row["act_dtype"] == "F8_E4M3" for row in timing)
    assert all(row["peak_flops_per_sec_for_op"] == fp8_peak
               for row in timing)


def test_humming_ladder_falls_back_when_no_rung_is_priced():
    """All ladder families absent skips it; pricing takes the BF16 edge."""
    resolved, step = _gpt_oss_step_with_fp8_request()
    gpu = replace(get_gpu("H100-SXM"),
                  fp8_tflops=None, fp4_tflops=None)
    deployment = ServingDeployment.of(
        server_flags={"moe_backend": "humming"}, activation_dtype="BF16",
        device="cuda", compute_capability="9.0",
        peak_families=("bf16",))
    binding = get_binding("vllm").deployed(deployment)
    result = analyze(
        resolved, parallel=step.parallel,
        framework_binding=binding, step_eval=step,
        peak_flops_per_sec=peak_flops_for_dtype(gpu, "BF16"),
        peak_hbm_bw_bytes_per_sec=3.35e12,
        gpu=gpu, breakdown=True)

    expert_rows = []
    expert_uids = set()
    for uid, row in result.access.values.items():
        if row.weight_dtype == "MXFP4":
            expert_rows.append(row)
            expert_uids.add(uid)
    assert expert_rows
    assert all(not row.promotion_ladder for row in expert_rows)
    assert all(row.launch_gap == (
        "missing_card_mma:mxfp4_humming_hopper_w4a8_ladder:"
        "F8_E4M3+MXFP4") for row in expert_rows)

    timing = [row for row in result.per_op_timing
              if row["uid"] in expert_uids]
    assert timing
    assert all(row["act_dtype"] == "BF16" for row in timing)
    assert all(row["peak_flops_per_sec_for_op"]
               == peak_flops_for_dtype(gpu, "BF16") for row in timing)
    assert all(row["convert_bytes_per_rank"] == 0 for row in timing)


def test_the_matrix_covers_every_mxfp4_snapshot():
    """No corpus MXFP4 model escapes the snapshot-derived matrix."""
    assert _mxfp4_snapshots() == set((*_ALL_STEMS, *_SGLANG_STEMS))


def test_deepgemm_needs_its_explicit_backend_fact_even_on_b200():
    stem = _DOT_STEMS[0]
    _binding, precision, launch = _selected(
        stem, sku=None, peaks=("fp8", "fp4", "bf16"),
        compute_capability="10.0")
    assert launch is not None
    assert launch.name == "mxfp4_dot_suffix_trtllm_mxfp8"
    assert "missing_deployment_fact:mxfp4_dot_suffix_deepgemm_group128" in (
        precision.launch_gaps(None, "", _coordinate(stem).formats))

    _binding, _precision, launch = _selected(
        stem, sku=None, peaks=("fp8", "fp4", "bf16"),
        compute_capability="10.0",
        flags={"moe_backend": "deep_gemm"})
    assert launch is not None
    assert launch.name == "mxfp4_dot_suffix_deepgemm_group128"


def test_a_new_card_family_needs_no_new_sm_window_in_yaml():
    """Capability columns, not an audited SM range, answer a future card."""
    stem = _DOT_STEMS[0]
    _binding, _precision, launch = _selected(
        stem, sku=None, peaks=("fp8", "fp4"),
        compute_capability="11.0")
    assert launch is not None
    assert launch.name == "mxfp4_dot_suffix_trtllm_mxfp8"


def test_a_card_with_a_missing_middle_column_falls_to_the_next_candidate():
    """A card with FP4 but no FP8 column skips the mixed launch, not its FP4
    operand width, and takes the declared BF16 fallback."""
    stem = _DOT_STEMS[0]
    _binding, precision, launch = _selected(
        stem, sku=None, peaks=("fp4", "bf16"),
        compute_capability="9.0")
    assert launch is not None
    assert launch.name == "mxfp4_dot_suffix_weight_only"
    assert precision.launch_gaps(
        None, "", _coordinate(stem).formats) == (
        "missing_card_mma:mxfp4_dot_suffix_trtllm_mxfp8:F8_E4M3",
        "missing_deployment_fact:mxfp4_dot_suffix_deepgemm_group128")


def test_no_viable_candidate_is_an_explicit_gap():
    stem = _DOT_STEMS[0]
    _binding, precision, launch = _selected(
        stem, sku=None, peaks=("fp8",), compute_capability="10.0")
    assert launch is None
    assert precision.launch_gap(
        None, "", _coordinate(stem).formats
    ).startswith("missing_card_mma:mxfp4_dot_suffix_trtllm_mxfp8:")
