"""Cross-layer checks for serving dispatch declarations.

A dispatch is not just a string in YAML. It must name the provider library
and the COMPLETE identity that provider resolves for a real invocation; then
latest must use that exact pair and fall no further.
"""
from __future__ import annotations

import functools
import json
from typing import Optional, Tuple

import pytest

from backend.inference import pipeline
from backend.inference.calibration_overlay import (
    WorkloadShape, build_deployment_context, point_keys)
from backend.inference.framework_bindings import (
    DispatchDecision, ServingDeployment, get_binding)
from backend.inference.op_calibration import (
    CalibrationRegistry, merge_validated_rows, select_time)
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import phase_binding
from kernel_bench.env import Env
from kernel_bench.invocation import Invocation
from kernel_bench.providers import (
    all_providers, applicable, catalog_kernel, kernel_library)
from tests.corpus_row import row as corpus_row
from tests.snapshots import resolve_snapshot


GPU = "H100-SXM"
MODEL = "meta-llama/Llama-3.2-1B"


@functools.cache
def _points(model: str, phase: str, batch: int,
            kv_block_size: Optional[int] = None,
            parallel: Optional[Tuple[int, int]] = None,
            precision=None, server_flags=None, gpu_sku: str = GPU):
    activation = "BF16"
    if precision is not None:
        from backend.arch.eval.precision import ResolvedPrecision
        from backend.arch.eval.program import Program

        activation = ResolvedPrecision(
            precision, Program.of(resolve_snapshot(model))).model_dtype
    binding = get_binding("vllm").deployed(ServingDeployment.at(
        gpu_sku=gpu_sku, activation_dtype=activation,
        server_flags=dict(server_flags or ())))
    resolved = resolve_snapshot(model)
    phase_info = phase_binding(
        phase=phase, avg_input_tokens=2048, avg_output_tokens=0)
    parallel_config = (
        ParallelConfig() if parallel is None
        else ParallelConfig(tp_attention=parallel[0],
                            tp_dense_ffn=parallel[1]))
    walk = pipeline.analyze(
        resolved, parallel=parallel_config, framework_binding=binding,
        batch_size=batch, avg_input_tokens=2048, avg_output_tokens=0,
        phase=phase, include_aux_timing=True, breakdown=False,
        precision=precision)
    deployment = build_deployment_context(
        framework="vllm", framework_binding=binding,
        parallel=parallel_config, gpu_sku=gpu_sku, phase=phase,
        kv_block_size=kv_block_size)
    points = point_keys(
        walk, walk.per_op_timing,
        WorkloadShape.of_walk(walk, seq_len=int(phase_info["s_query"])),
        deployment, model_id=model, spec_hash="dispatch-test")
    return walk, points


def _invocation(point, phase: str, extra_dims=None) -> Invocation:
    dims = json.loads(point.key.shape_json)
    dims.update(extra_dims or {})
    return Invocation.from_job({
        "signature": point.signature,
        "op": point.op,
        "dims": dims,
        "dtype_env": json.loads(point.dtype_env_json),
        "phase": phase,
        "physical_key": [point.signature, point.key.shape_json,
                         point.key.dtype_env_json, phase],
    })


def _provider_env() -> Env:
    return Env("2.11", None, "sm_90", None, None,
               libs={"torch": "2.11", "vllm": "0.32.0"})


@pytest.mark.parametrize("model", [
    MODEL,
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-8B",
])
@pytest.mark.regression
def test_every_non_parameterised_dispatch_is_the_provider_for_that_cell(model):
    """The audited Llama/Qwen rows are real walks; provider.serves and Provider.impl_id
    answer the same cell the YAML dispatch names."""
    env = _provider_env()
    checked = 0
    for phase in ("prefill", "decode"):
        _walk, points = _points(model, phase, 16)
        dispatches = {(d.op, d.impl_id): d
                      for d in get_binding("vllm").kernel_dispatches
                      if "flash_attention" not in d.impl_id}
        for point in points:
            if point.key is None or point.dispatch.state != DispatchDecision.DECLARED:
                continue
            if point.op == "flash_attention":
                continue
            dispatch = dispatches[(point.op, point.dispatch.impl_id)]
            inv = _invocation(point, phase)
            providers = [p for p in all_providers()
                         if p.library == dispatch.library
                         and p.kernel_impl == catalog_kernel(dispatch.impl_id)]
            assert len(providers) == 1, (dispatch, providers)
            provider = providers[0]
            assert provider.signature == point.signature
            assert kernel_library(catalog_kernel(dispatch.impl_id)) == dispatch.library
            assert provider.impl_id(inv) == dispatch.impl_id
            assert provider.accepts(inv, env), list(provider.refusals(inv, env))
            checked += 1
    assert checked


@pytest.mark.regression
@pytest.mark.parametrize("model", [MODEL, "Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B"])
def test_vllm_fa_dispatches_resolve_their_complete_parameter_suffixes(model):
    """The two vLLM FA declarations use the provider-owned page suffix, not a
    bare kvcache base name. Their provider scope is a real sink-bearing GQA
    invocation at the audited Hopper/page-16 cell."""
    env = _provider_env()
    binding = get_binding("vllm")
    for phase, impl in (
            ("prefill", "vllm_flash_attention_v3"),
            ("decode", "vllm_flash_attention_v3_kvcache@_b16")):
        _walk, points = _points(model, phase, 16)
        point = next(p for p in points if p.op == "flash_attention")
        # These providers are the sink-bearing vLLM FA3 cells. Presence of
        # n_attn_sinks is the signature's own variant fact, not a dispatch
        # condition; Llama's no-sink attention correctly has no provider row.
        inv = _invocation(point, phase, {"n_attn_sinks": 1})
        provider = next(p for p in all_providers()
                        if p.library == "vllm"
                        and p.impl_id(inv) == impl)
        assert provider.signature == point.signature
        assert provider.accepts(inv, env), list(provider.refusals(inv, env))
        dispatch = next(d for d in binding.kernel_dispatches
                        if d.op == "flash_attention" and d.impl_id == impl)
        assert catalog_kernel(dispatch.impl_id) == provider.kernel_impl


def test_a_page_32_cell_is_not_declared_by_the_page_16_identity():
    """The audited dispatch is the provider's `_b16` cell only. A page-32
    deployment has a different complete identity, so it is explicitly
    unselected rather than borrowing the page-16 declaration."""
    _walk, points = _points(MODEL, "decode", 16, kv_block_size=32)
    point = next(p for p in points if p.op == "flash_attention")
    assert point.key is not None
    assert json.loads(point.key.shape_json)["kv_block_size"] == 32
    assert (point.dispatch.state, point.dispatch.reason) == (
        "no_active", "no_active_kernel_dispatch")


def test_tensor_parallel_embedding_is_outside_the_audited_tp1_dispatch():
    """The anchor is VocabParallelEmbedding's unsharded path. At TP2 it must
    not claim the torch embedding identity."""
    _walk, points = _points(MODEL, "prefill", 16, parallel=(2, 2))
    point = next(p for p in points if p.op == "embedding")
    assert (point.dispatch.state, point.dispatch.reason) == (
        "no_active", "no_active_kernel_dispatch")


def test_no_dispatch_condition_names_a_checkpoint():
    """A dispatch states the facts the framework's own dispatch reads (the
    model class, card, launch, TP); a checkpoint name is not one of them
    (CLAUDE.md §1.3). The vocabulary refuses it at load."""
    from dataclasses import fields

    from backend.arch.eval.fusion import GraphCondition
    from backend.inference.framework_bindings import DispatchCondition

    assert "models" not in {f.name for f in fields(GraphCondition)}
    with pytest.raises(ValueError, match="not in the vocabulary"):
        DispatchCondition.parse(
            {"architectures": ["LlamaForCausalLM"],
             "models": ["meta-llama__Llama-3.2-1B"]}, "<test>")


def test_a_program_field_that_is_not_a_graph_condition_field_is_refused(
        monkeypatch):
    """`PROGRAM_FIELDS` may stay a literal list only because the loader
    rejects it when it drifts (CLAUDE.md §1.4: a name list is allowed when
    something hard-rejects on it at load). Without this the list could name
    a field nobody has, and `_KERNEL_DISPATCH_WHEN` would admit a `when:`
    key that answers nothing."""
    from backend.arch.eval.fusion import GraphCondition
    from backend.inference.framework_bindings import (
        _check_dispatch_vocabulary)

    monkeypatch.setattr(GraphCondition, "PROGRAM_FIELDS",
                        (*GraphCondition.PROGRAM_FIELDS, "checkpoint_name"))
    with pytest.raises(ValueError, match="not fields of GraphCondition"):
        _check_dispatch_vocabulary()


@pytest.mark.parametrize("model", [
    # The class whose scope this branch widened furthest, first.
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen3-32B",
])
def test_a_dispatch_holds_wherever_its_facts_hold(model):
    """The anchors (`UnquantizedLinearMethod` → `F.linear`, `RMSNorm`,
    `SiluAndMul`) are model-independent, so a declaration applies to every
    checkpoint of a listed class whose deployment and launch facts satisfy
    it — not only to the one the audit was first read for."""
    _walk, points = _points(model, "prefill", 16)
    declared = {p.op: p.dispatch for p in points
                if p.dispatch.state == DispatchDecision.DECLARED}
    assert declared.keys() >= {
        "linear", "qkv_parallel_linear", "merged_column_parallel_linear",
        "embedding", "rmsnorm", "silu_and_mul", "fused_qk_rope",
        "flash_attention"}, sorted(declared)
    assert declared["linear"].impl_id == "torch_cublas_linear"
    assert declared["rmsnorm"].impl_id == "vllm_rms_norm"


@pytest.mark.parametrize("model,op,server_flags", [
    # A server selector that swaps the provider implementation. Llama first:
    # dropping the checkpoint pin widened these entries from one checkpoint
    # to every `LlamaForCausalLM`, so this is the half most likely to
    # regress — and a declaration here attaches a torch measurement row to a
    # cell vLLM runs FlashInfer's `mm_bf16` on (a wrong number, not a gap).
    ("meta-llama/Llama-3.2-3B", "linear",
     (("linear_backend", "flashinfer_cutedsl"),)),
    ("meta-llama/Llama-3.2-3B", "qkv_parallel_linear",
     (("linear_backend", "flashinfer_cutedsl"),)),
    ("meta-llama/Llama-3.2-3B", "merged_column_parallel_linear",
     (("linear_backend", "flashinfer_cutedsl"),)),
    ("meta-llama/Llama-3.2-3B", "flash_attention",
     (("attention_backend", "FLASHINFER"),)),
    ("Qwen/Qwen3-8B", "linear",
     (("linear_backend", "flashinfer_cutedsl"),)),
    ("Qwen/Qwen3-8B", "flash_attention",
     (("attention_backend", "FLASHINFER"),)),
])
def test_a_dispatch_does_not_hold_where_a_stated_fact_fails(
        model, op, server_flags):
    """The audited cells are `linear_backend=auto` and automatic attention
    selection. `dispatch_unquantized_gemm` reads the platform and that flag
    and nothing else (`vllm/model_executor/layers/utils.py:620-632`), so a
    request naming another backend selects a cell nobody audited — for
    every class the entry lists, not just the one with a test."""
    assert {"linear_backend", "attention_backend"} <= (
        get_binding("vllm").server_flag_names())
    _walk, default = _points(model, "prefill", 16)
    assert next(p for p in default if p.op == op).dispatch.state == (
        DispatchDecision.DECLARED)

    _walk, chosen = _points(model, "prefill", 16, server_flags=server_flags)
    point = next(p for p in chosen if p.op == op)
    assert (point.dispatch.state, point.dispatch.reason) == (
        "no_active", "no_active_kernel_dispatch")


@pytest.mark.parametrize("gpu_sku,flash_attention", [
    # Outside the audited window first: FA's version resolution is by
    # capability, so it is the one op a card generation decides.
    ("B200", DispatchDecision.NO_ACTIVE),
    ("H100-SXM", DispatchDecision.DECLARED),
])
def test_only_the_capability_windowed_dispatch_leaves_the_audited_card(
        gpu_sku, flash_attention):
    """The audit covered Hopper — but only where the card decides anything.

    vLLM resolves the FlashAttention version from the device capability
    (`fa_utils.py:94-104`), so that entry states a window and a Blackwell
    request falls outside it. `ir.ops.rms_norm` and
    `torch.ops._C.silu_and_mul` read no capability at all
    (`layernorm.py:74-96`, `activation.py:128-150`), so those entries state
    none and a Blackwell request keeps them. Asked through the serving path
    so the row's own resolved launch answers the launch key: asking with no
    launch would make every entry that states one false whatever the card,
    and the check would pass on Hopper too.
    """
    _walk, points = _points("Qwen/Qwen3-8B", "prefill", 16, gpu_sku=gpu_sku)
    state = {p.op: p.dispatch.state for p in points}
    assert state["flash_attention"] == flash_attention
    # The serving row need not keep the primitive `add`: vLLM's rewrite can
    # hand its operands to `fused_add_rmsnorm`, whose launch is the row that
    # asks the declaration.
    add_launch = state.get("add", state.get("fused_add_rmsnorm"))
    assert add_launch == DispatchDecision.DECLARED, add_launch
    for op in ("rmsnorm", "silu_and_mul", "fused_qk_rope"):
        assert state[op] == DispatchDecision.DECLARED, (op, state[op])


@pytest.mark.regression
@pytest.mark.parametrize("model", ["Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B"])
def test_a_precision_resolved_fp8_launch_is_not_the_declared_bf16_qwen_gemm(model):
    """A custom profile owns the launch identity; latest must see a
    declaration gap, not reclassify an FP8 PhysicalKey as a missing BF16
    corpus row."""
    from backend.arch.precision_profile import PrecisionProfile

    profile = PrecisionProfile.canonicalize({"weights": [
        {"select": {"roles": "all"}, "format": {"scheme": "fp8"}},
    ]})
    _walk, points = _points(model, "prefill", 16, precision=profile)
    point = next(p for p in points if p.op == "linear")
    assert point.key is not None
    env = json.loads(point.dtype_env_json)
    assert (env["activation"], env["weight"]) == ("fp8_e4m3", "fp8_e4m3")
    assert (point.dispatch.state, point.dispatch.reason) == (
        "no_active", "no_active_kernel_dispatch")

    empty = CalibrationRegistry()
    _got, state = select_time(
        empty, point.key, strategy="latest", framework="vllm",
        kernel_dispatch=point.dispatch.as_dict(),
        current_peak_tflops_at=lambda _d: 1.0, current_peak_bw_gbps=1.0)
    assert state == "ambiguous"

    _walk, default_points = _points(model, "prefill", 16)
    default = next(p for p in default_points if p.op == "linear")
    assert default.dispatch.state == DispatchDecision.DECLARED
    assert default.dispatch.impl_id == "torch_cublas_linear"
    _got, default_state = select_time(
        empty, default.key, strategy="latest", framework="vllm",
        kernel_dispatch=default.dispatch.as_dict(),
        current_peak_tflops_at=lambda _d: 1.0, current_peak_bw_gbps=1.0)
    assert default_state == "missing"


@pytest.mark.regression
def test_fp32_promotion_is_selected_by_its_full_provider_identity():
    """A promoted torch GEMM is a different identity; base-name matching or a
    library-less filter must not turn it into the BF16 row."""
    from tests.backend.inference.test_weight_promotion import _walk
    from backend.inference.benchmark_identity import RowIdentity

    walk = _walk("MiniMaxAI/MiniMax-M2.7")
    row = next(r for r in walk.per_op_timing
               if str(r.get("aggregation_id")).endswith(
                   "block_sparse_moe.gate"))
    op, shape = RowIdentity(walk).of(row)
    identity = RowIdentity(walk)
    dtype_json = identity.dtype_env(row)
    inv = Invocation.from_job({
        "signature": "gemm", "op": op, "dims": json.loads(shape),
        "dtype_env": json.loads(dtype_json), "phase": "prefill",
        "physical_key": ["gemm", shape, dtype_json, "prefill"]})
    provider = next(p for p in applicable(inv, _provider_env())
                    if p.library == "torch")
    full_id = provider.impl_id(inv)
    assert full_id == "torch_cublas_linear@compute=fp32"

    registry = merge_validated_rows(
        CalibrationRegistry(),
        [corpus_row(model="MiniMaxAI/MiniMax-M2.7", spec_hash="promotion",
                    phase="prefill", parallel="tp=1", framework="torch",
                    gpu=GPU, op=op, node_spec_id="gate",
                    shape_json=shape, dtype_env_json=dtype_json,
                    kernel_impl=full_id, latency_us="17.0",
                    achieved_tflops="0.0", achieved_bw_gbps="3000.0",
                    peak_tflops_at_test="989.0", peak_bw_at_test="3350.0")],
        source="<test>", origin="shared")
    key = next(iter(registry.rows_by_key)).to_physical()
    got, state = select_time(
        registry, key, strategy="latest", framework="vllm",
        kernel_dispatch={"state": "declared", "library": "torch",
                         "impl_id": full_id},
        current_peak_tflops_at=lambda _d: 1.0, current_peak_bw_gbps=1.0)
    assert state == "measured" and got is not None
    _, base_state = select_time(
        registry, key, strategy="latest", framework="vllm",
        kernel_dispatch={"state": "declared", "library": "torch",
                         "impl_id": "torch_cublas_linear"},
        current_peak_tflops_at=lambda _d: 1.0, current_peak_bw_gbps=1.0)
    assert base_state == "missing"


def _row_for(point, library: str, impl: str, latency_us: str):
    return corpus_row(
        model=MODEL, spec_hash="dispatch-test", phase=point.key.phase,
        parallel="tp=1", framework=library, framework_version="2.11.0",
        gpu=GPU, op=point.op, node_spec_id=point.aggregation_id,
        shape_json=point.key.shape_json,
        dtype_env_json=point.key.dtype_env_json, kernel_impl=impl,
        latency_us=latency_us, achieved_tflops="0.0",
        achieved_bw_gbps="3000.0", peak_tflops_at_test="989.0",
        peak_bw_at_test="3350.0")


@pytest.mark.regression
def test_llama_latest_coverage_reads_declared_torch_rows_and_keeps_real_gaps():
    """The decode B=16 stop-gap: the old 13/22 set becomes 20/22 once the
    audited torch rows exist; rope and one attention shape remain explicit
    missing rather than borrowing another corpus row."""
    _walk, points = _points(MODEL, "decode", 16)
    keyed = [p for p in points if p.key is not None]
    by_op = {p.op: p for p in keyed}

    # The old selector's corpus: vLLM-provider rows only. Rope has no data and
    # this Llama no-sink attention shape has no provider row, leaving 13/22.
    rows = [_row_for(p, "vllm", p.dispatch.impl_id, "20.0")
            for p in keyed
            if p.dispatch.state == DispatchDecision.DECLARED
            and p.dispatch.library == "vllm"
            and p.op not in ("fused_qk_rope", "flash_attention")]
    old = merge_validated_rows(CalibrationRegistry(), rows,
                               source="<test>", origin="shared")

    # Add exactly the declared torch identities. No row is retagged as vLLM.
    rows += [_row_for(p, "torch", p.dispatch.impl_id, "10.0")
             for p in keyed
             if p.dispatch.state == DispatchDecision.DECLARED
             and p.dispatch.library == "torch"]
    new = merge_validated_rows(CalibrationRegistry(), rows,
                               source="<test>", origin="shared")

    def measured(registry) -> int:
        count = sum(p.zero_cost for p in points)
        for point in keyed:
            _got, state = select_time(
                registry, point.key, strategy="latest", framework="vllm",
                kernel_dispatch=point.dispatch.as_dict(),
                current_peak_tflops_at=lambda _d: 1.0,
                current_peak_bw_gbps=1.0)
            count += state == "measured"
        return count

    assert measured(old) == 13
    assert measured(new) == 20
    got, state = select_time(
        new, by_op["linear"].key, strategy="latest", framework="vllm",
        kernel_dispatch=by_op["linear"].dispatch.as_dict(),
        current_peak_tflops_at=lambda _d: 1.0, current_peak_bw_gbps=1.0)
    assert state == "measured"
    assert got.source_badge["framework"] == "torch"
    assert got.source_badge["serving_framework"] == "vllm"
    assert got.source_badge["declared_kernel_impl"] == "torch_cublas_linear"


@pytest.mark.regression
def test_undeclared_op_does_not_fall_back_to_a_matching_corpus_row():
    row = corpus_row(framework="vllm", kernel_impl="vllm_rms_norm")
    registry = merge_validated_rows(
        CalibrationRegistry(), [row], source="<test>", origin="shared")
    key = next(iter(registry.rows_by_key)).to_physical()
    got, state = select_time(
        registry, key, strategy="latest", framework="vllm",
        kernel_dispatch={"state": "undeclared",
                         "reason": "undeclared_kernel_dispatch"},
        current_peak_tflops_at=lambda _d: 1.0, current_peak_bw_gbps=1.0)
    assert got is None and state == "ambiguous"


# ---------------------------------------------------------------------------
# A condition's dtype and a key's dtype are the same fact, so they have to be
# the same string.
# ---------------------------------------------------------------------------

def _conditions_naming_an_activation_dtype():
    """`{stored value: [where it is declared]}` over every binding.

    Both tables that carry conditions, because a value is as unreachable in
    one as in the other.
    """
    import collections
    from backend.inference.framework_bindings import FRAMEWORK_BINDINGS
    seen = collections.defaultdict(list)
    for name, binding in sorted(FRAMEWORK_BINDINGS.items()):
        for key, impl in binding.op_implementations:
            for when in (impl.whens or ()):
                for value in (when.activation_dtype or ()):
                    seen[value].append(f"{name}: fusion {key}")
        for entry in (binding.kernel_dispatches or ()):
            for when in (entry.whens or ()):
                for value in (when.activation_dtype or ()):
                    seen[value].append(f"{name}: {entry.op} -> {entry.impl_id}")
    return seen


def _activation_dtypes_a_key_can_present():
    """Every spelling a real measurement key puts in front of a condition.

    Through the one constructor, not a list written here: a key states its
    dtypes in the corpus vocabulary and `ResolvedLaunch` is what a condition
    is asked about, so whatever that pair produces IS the reachable set.
    """
    from backend.arch.dtypes import CORPUS_TOKENS
    from backend.inference.framework_bindings import ResolvedLaunch
    return {ResolvedLaunch.from_dtype_env_json(
        json.dumps({"activation": token})).activation_dtype
        for token in CORPUS_TOKENS}


def test_every_activation_dtype_a_condition_names_is_one_a_key_can_present():
    """The oracle for a bug that is silent by construction.

    `DispatchCondition.of` canonicalises a condition's own values, so
    `activation_dtype: [int8]` is stored as `i8`. A key spelled the dtype
    the corpus spells it, and `_among` compares the two strings — so every
    condition naming a dtype whose canonical form differs from its corpus
    token could never match, and nothing failed: the dispatch merely
    resolved `no_active` and the estimate priced on the roofline. Three
    bindings had such conditions (`f16` x11, `f32` x2, `i8` x3) and the
    only reason anything worked is that `bf16` is one of the five tokens
    canonicalisation leaves alone.

    Asserted as a round trip rather than as a mapping: this compares what
    the loader stores against what the key constructor produces, so the two
    sides cannot drift without this going red — which is the thing a
    counter of known-bad spellings would not do.
    """
    declared = _conditions_naming_an_activation_dtype()
    reachable = _activation_dtypes_a_key_can_present()
    unreachable = {value: where for value, where in declared.items()
                   if value not in reachable}
    assert not unreachable, (
        "these dispatch conditions can never hold — no measurement key "
        "spells its activation dtype the way the condition stores it:\n"
        + "\n".join(f"  {value!r}: {where[:3]}"
                    for value, where in sorted(unreachable.items()))
        + f"\n  reachable spellings: {sorted(x for x in reachable if x)}")


def test_an_int8_launch_reaches_the_ascend_w8a8_dispatch():
    """The case the oracle above was written for, end to end.

    vLLM-Ascend's W8A8 deployment runs `npu_quant_matmul_w8a8` for a
    quantised linear (`vllm_ascend.yaml`), and the corpus holds rows
    measured on a 910C die under exactly this key. Until the launch side
    was canonicalised the condition read `i8` against a key saying `int8`,
    so the op dispatched `no_active` and every one of those rows was
    invisible to the estimate that had asked for it.
    """
    from backend.inference.framework_bindings import ResolvedLaunch
    binding = get_binding("vllm_ascend")
    deployment = ServingDeployment.at(gpu_sku="Ascend-910C-A3",
                                      activation_dtype="bf16")
    launch = ResolvedLaunch.from_dtype_env_json(json.dumps({
        "activation": "int8", "activation_granularity": "per-token",
        "metadata": "bf16", "output": "bf16",
        "weight": "int8", "weight_granularity": "per-channel"}))
    entries = [e for e in binding.kernel_dispatches if e.op == "linear"]
    active = [e for e in entries
              if any(w.holds_for(deployment, launch) for w in (e.whens or ()))]
    assert [e.impl_id for e in active] == ["npu_quant_matmul_w8a8"], (
        f"an INT8 linear launch should reach exactly the W8A8 kernel; "
        f"active: {[e.impl_id for e in active]}")


def test_a_condition_naming_the_output_slot_reads_the_output_slot():
    """A quantise kernel is decided by what it WRITES, not what it reads.

    Every `activation_dtype` condition asks about the operand a kernel
    consumes. A quantiser's whole job is to consume a wide tensor and
    produce a narrow one, so its int8 lives in the OUTPUT slot — and a
    condition saying `activation_dtype: [int8]` asks "is the thing this
    kernel reads already int8", which is never true of a quantiser. The
    dispatch then resolved `no_active` and the measured rows went unread.

    `output_dtype` is the launch's own fact, supplied per row after
    `precision_launches` has materialized the walk — the same provenance
    `unquantized_bf16_launch` has, which is why it belongs in LAUNCH_KEYS.
    """
    from backend.inference.framework_bindings import ResolvedLaunch
    binding = get_binding("vllm_ascend")
    deployment = ServingDeployment.at(gpu_sku="Ascend-910C-A3",
                                      activation_dtype="bf16")
    launch = ResolvedLaunch.from_dtype_env_json(json.dumps({
        "activation": "bf16", "activation_granularity": "per-token",
        "metadata": "fp32", "output": "int8"}))
    entries = [e for e in binding.kernel_dispatches
               if e.op == "quantise_activation"]
    active = [e for e in entries
              if any(w.holds_for(deployment, launch) for w in (e.whens or ()))]
    assert [e.impl_id for e in active] == ["npu_dynamic_quant_int8"], (
        f"a cast whose OUTPUT is int8 should reach the quantise kernel; "
        f"active: {[e.impl_id for e in active]}")


def test_an_output_dtype_condition_cannot_hold_without_a_launch():
    """No launch, no output slot — and no guess.

    A deployment has no output dtype; it is a property of one launch. The
    condition must be unsatisfiable there rather than falling back to the
    request's model dtype, which would be the second derivation this
    module exists to remove.
    """
    binding = get_binding("vllm_ascend")
    deployment = ServingDeployment.at(gpu_sku="Ascend-910C-A3",
                                      activation_dtype="bf16")
    entries = [e for e in binding.kernel_dispatches
               if e.op == "quantise_activation"]
    active = [e for e in entries
              if any(w.holds_for(deployment, None) for w in (e.whens or ()))]
    assert active == []


def test_the_output_slot_is_compared_in_the_condition_vocabulary():
    """`int8` in a cell and `i8` in a condition are the same dtype.

    The launch side canonicalises (0016); a slot read straight from a
    `dtype_env` does not. Comparing them raw is the same defect one slot
    over, so the output slot takes the same route.
    """
    from backend.inference.framework_bindings import (
        DispatchCondition, ResolvedLaunch)
    deployment = ServingDeployment.at(gpu_sku="Ascend-910C-A3",
                                      activation_dtype="bf16")
    cond = DispatchCondition.parse({"output_dtype": ["int8"]}, "test")
    for spelling in ("int8", "I8", " i8 "):
        launch = ResolvedLaunch.from_dtype_env_json(json.dumps({
            "activation": "bf16", "output": spelling}))
        assert cond.holds_for(deployment, launch), spelling
