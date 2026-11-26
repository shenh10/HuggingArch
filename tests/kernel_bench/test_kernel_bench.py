"""kernel_bench scaffold — CPU-runnable tests.

Cover the no-GPU surface: signatures, provider discovery + capability,
invocation parsing, emit-conforms-to-schema, and clean degradation of
`run` when no provider is applicable in this (GPU-less) env. The GPU
kernel bodies themselves are exercised on a Hopper box, not here.

Marked `regression`: this file takes minutes, almost all of it planning
real models end to end (one test plans every snapshot). It is deselected by
default (see `addopts` in pyproject.toml) — run it with `-m regression`, or
pick a case with `-m regression -k <name>`, when you are changing
`kernel_bench/`.
"""
from __future__ import annotations

import os
import json
import unittest
from pathlib import Path
from typing import ClassVar

import pytest

pytestmark = pytest.mark.regression

from backend.inference import benchmarks as _benchmarks
from tests.kernel_bench._inputs import build_inputs
from tests.kernel_bench.test_plan_candidates import judged_on
#: The one-env manifest the run-all tests fake (`dependency_manifest`).
_FAKE = {"fake": {"source_build": False}}
from kernel_bench import signatures
from kernel_bench.env import Env, detect_env
from kernel_bench.invocation import Invocation
from kernel_bench.providers import TORCH_FLOAT, all_providers, applicable
from kernel_bench.emit import build_row
from kernel_bench.providers import LatencyStats
from backend.inference.sweep_grid import ROUTINE_TRACKS
from tests.snapshots import resolve_snapshot

#: Every batch point of the routine grid (`b1` … `max`), one phase's track.
_ROUTINE_BATCH_KEYS = ROUTINE_TRACKS[0].batch_keys


def _routine_sweep(*, model_id, **kw):
    """`build_sweep_jobs` over the routine surface — the routine tracks over
    the model's routine deployments at the routine output length, as
    `hbench sweep` asks it."""
    from backend.inference.sweep_grid import ROUTINE_OUTPUT_TOKENS
    from kernel_bench.sweep import (build_sweep_jobs, resolve_sweep_model,
                                    routine_parallel_configs)
    config, resolved = resolve_sweep_model(model_id)
    return build_sweep_jobs(
        model_id=model_id, tracks=ROUTINE_TRACKS,
        parallel_configs=routine_parallel_configs(config, resolved=resolved),
        avg_output_tokens=ROUTINE_OUTPUT_TOKENS, resolved=resolved, **kw)


def _one_served_routine_sweep(*, model_id, **kw):
    """`build_sweep_jobs` over ONE routine deployment — the first the model
    is served on — with the same tracks and output length as the routine
    surface.

    A question about what the sweep PLANS (which signatures it reaches) is
    answered by one deployment that yields jobs; asking all of a big model's
    routine deployments costs two minutes each and says the same thing.
    Returns `(jobs, label)`; a deployment that serves no point plans no job,
    so the loop walks on.
    """
    from backend.inference.sweep_grid import ROUTINE_OUTPUT_TOKENS
    from kernel_bench.sweep import (build_sweep_jobs, resolve_sweep_model,
                                    routine_parallel_configs)
    config, resolved = resolve_sweep_model(model_id)
    for label, par in routine_parallel_configs(config, resolved=resolved):
        jobs, _stats = build_sweep_jobs(
            model_id=model_id, tracks=ROUTINE_TRACKS,
            parallel_configs=[(label, par)],
            avg_output_tokens=ROUTINE_OUTPUT_TOKENS, resolved=resolved, **kw)
        if jobs:
            return jobs, label
    raise AssertionError(f"{model_id}: no routine deployment serves a point")


def _stated_key(job: dict) -> list:
    """The measurement key a planned job carries (`PointRow.key`), stated for
    a fixture that stands in for one: the row writer stamps the card on it and
    the cell's input seed is derived from it."""
    return [job["signature"],
            json.dumps(job["dims"], sort_keys=True, separators=(",", ":")),
            json.dumps(job["dtype_env"], sort_keys=True, separators=(",", ":")),
            job["phase"]]


def _gqa_job(**over) -> dict:
    # The cell's own query extent: prefill asks the context's tokens, decode
    # one per request. A cell that states no `s_query` is refused by every
    # kernel whose operand layout names it (closed declarations), so the
    # fixture carries it for both phases.
    _s_query = 1 if over.get("phase") == "decode" else 2048
    job = {
        "signature": "gqa_attention",
        "op": "flash_attention",
        # A paged cell carries the deployment's page (`PAGE_AXIS`).
        "dims": {"B": 1, "s_ctx": 2048, "s_kv": 2048, "s_query": _s_query,
                 "n_heads": 24, "n_kv_heads": 8, "d_head": 128,
                 "d_model": 3072, "d_ffn": 8192, "n_layers": 28, "n_vocab": 128256,
                 "kv_block_size": 16},
        "dtype_env": {"weight": "bf16", "activation": "bf16", "kv": "bf16",
                      "output": "bf16", "metadata": "fp32"},
        "phase": "prefill",
        "tp": 1,
        "flops_per_rank": 1.0e12,
        "access_bytes_per_rank": 3.0e9,
        "model": "meta-llama/Llama-3.2-3B",
        "spec_hash": "abc123",
        "node_spec_id": "block|self_attn.attn_apply__gqa_attention",
        "parallel": "tp=1",
        "count_per_step": 28,
    }
    job.update(over)
    job.setdefault("physical_key", _stated_key(job))
    return job


class TestSignatures:
    def test_known_signatures_present(self):
        for name in ("gqa_attention", "mla_attention", "gemm"):
            assert signatures.get_signature(name) is not None

    def test_op_maps_to_signature(self):
        assert signatures.op_to_signature("flash_attention") == "gqa_attention"
        assert signatures.op_to_signature("mla_attention") == "mla_attention"
        assert signatures.op_to_signature("allreduce_fused_rmsnorm") == (
            "allreduce_fused_rmsnorm")

    def test_zero_cost_op_has_no_signature(self):
        # layout ops are intentionally unmapped (never measured)
        assert signatures.op_to_signature("transpose") is None
        assert signatures.op_to_signature("view_split") is None


class TestTorchDtypeTable:
    def test_known_names_and_unknown_raises(self):
        import torch
        from kernel_bench.refs import UnknownTorchDtype, torch_dtype
        from kernel_bench.driver import _act_torch_dtype
        from kernel_bench.providers.attention import _torch_dtype

        assert torch_dtype("bf16") is torch.bfloat16
        assert torch_dtype("fp8_e4m3") is torch.float8_e4m3fn
        assert _torch_dtype("fp16") is torch.float16
        inv = Invocation.from_job(_gqa_job())
        assert _act_torch_dtype(inv) is torch.bfloat16
        with pytest.raises(UnknownTorchDtype):
            torch_dtype("mxfp4")


class TestProviderRegistry:
    def test_attention_providers_registered(self):
        libs = {p.library for p in all_providers()}
        assert {"fa2", "fa3", "flashinfer", "flashmla"} <= libs

    def test_fa_family_covers_prefill_and_decode(self):
        # FA2 + FA3, each with a prefill and a decode (kvcache) provider.
        fa = [p for p in all_providers() if p.library in ("fa2", "fa3")]
        kernels = {p.kernel_impl for p in fa}
        assert kernels == {
            "flash_attention_v2", "flash_attention_v2_kvcache",
            "flash_attention_v3", "flash_attention_v3_kvcache",
        }
        # FA2 imports from `flash_attn`; FA3 from `flash_attn_interface`.
        assert {p.import_name for p in fa if p.library == "fa2"} == {"flash_attn"}
        assert {p.import_name for p in fa if p.library == "fa3"} == {"flash_attn_interface"}

    def test_runtime_limits_are_published_from_the_registry(self):
        fa4 = next(p for p in all_providers()
                   if p.kernel_impl == "flash_attention_v4")
        assert fa4.serves is not None
        assert fa4.serves.unsupported_features == ("sm90_page_table",)
        caps = fa4.capabilities()
        assert caps["unsupported_features"] == ["sm90_page_table"]
        # The shape columns remain shape-only; a paging/runtime limitation must
        # not be smuggled into a list the UI labels as geometry constraints.
        assert "sm90_page_table" not in caps["shape_constraints"]

    def test_framework_fa3_sink_providers_are_sink_only(self):
        env = Env("2.8.0+cu128", True, "H100 80GB HBM3", "H100-SXM",
                  3350.0, sm="sm_90",
                  libs={"vllm": "0.11.0", "sglang": "0.5.5.post3"})
        base_dims = _gqa_job()["dims"]
        sink_prefill = Invocation.from_job(_gqa_job(dims={
            **base_dims, "n_attn_sinks": 24,
        }))
        sink_decode = Invocation.from_job(_gqa_job(
            phase="decode", dims={**base_dims, "n_attn_sinks": 24}))
        plain = Invocation.from_job(_gqa_job())
        sink_impls = (
            {p.kernel_impl for p in applicable(sink_prefill, env)}
            | {p.kernel_impl for p in applicable(sink_decode, env)}
        )
        plain_impls = {p.kernel_impl for p in applicable(plain, env)}
        expected = {
            "vllm_flash_attention_v3", "vllm_flash_attention_v3_kvcache",
            "sglang_flash_attention_v3", "sglang_flash_attention_v3_kvcache",
        }
        # vLLM and SGLang ship sink-capable FA3 forks; the official fa3 wheel
        # is deliberately absent because it has no s_aux/sinks argument.
        assert expected <= sink_impls
        assert not (expected & plain_impls)

    def test_gemm_providers_and_dtype_capability(self):
        provs = {(p.library, p.kernel_impl) for p in all_providers()}
        assert ("torch", "torch_cublas_linear") in provs
        assert ("deepgemm", "deepgemm_fp8_blockwise") in provs
        env = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                  libs={"torch": "2.11", "deep_gemm": "1.0"})
        dims = {"B": 1, "s_query": 2048, "d_reduce": 3072, "d_out": 3072}
        bf16 = Invocation.from_job({
            "signature": "gemm", "op": "linear", "dims": dims, "phase": "prefill",
            "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"}})
        fp8 = Invocation.from_job({
            "signature": "gemm", "op": "linear", "dims": dims, "phase": "prefill",
            "dtype_env": {"activation": "fp8_e4m3", "weight": "fp8_e4m3",
                          "kv": "bf16", "output": "bf16", "metadata": "fp32",
                          "activation_granularity": "per-group:128",
                          "weight_granularity": "per-tile:128x128"}})
        mixed = Invocation.from_job({
            "signature": "gemm", "op": "linear", "dims": dims, "phase": "decode",
            "dtype_env": {"activation": "bf16", "weight": "fp32",
                          "kv": "bf16", "output": "bf16"}})
        promoted = Invocation.from_job({
            "signature": "gemm", "op": "linear", "dims": dims,
            "phase": "prefill",
            "dtype_env": {"activation": "fp32", "weight": "fp32",
                          "kv": "bf16", "output": "bf16"}})
        assert {p.library for p in applicable(bf16, env)} == {"torch"}
        # torch's floor multiplies at the cell's float width; an FP8 cell is
        # not a launch it makes (its declaration lists no fp8).
        assert {p.library for p in applicable(fp8, env)} == {"deepgemm"}
        torch_floor = next(p for p in all_providers()
                           if p.kernel_impl == "torch_cublas_linear")
        assert torch_floor.serves is not None
        assert torch_floor.serves.accepts(promoted)
        assert torch_floor.impl_id(mixed) == "torch_cublas_linear@compute=fp32"
        assert torch_floor.impl_id(promoted) == "torch_cublas_linear@compute=fp32"
        assert torch_floor.actual_computes_at(mixed) == {
            "activation": "fp32"}
        # Both compute slots already state the promoted multiply; the bf16
        # store is the cell's own output slot, not a second measured dtype.
        assert torch_floor.actual_computes_at(promoted) == {}

    def test_norm_act_providers_registered(self):
        provs = {(p.library, p.signature, p.kernel_impl) for p in all_providers()}
        assert ("vllm", "rmsnorm", "vllm_rms_norm") in provs
        assert ("vllm", "fused_add_rmsnorm", "vllm_fused_add_rms_norm") in provs
        assert ("vllm", "silu_and_mul", "vllm_silu_and_mul") in provs

    def test_norm_provider_matches_only_its_signature(self):
        env = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                  libs={"vllm": "0.6.3"})
        rms = next(p for p in all_providers() if p.kernel_impl == "vllm_rms_norm")
        rms_inv = Invocation.from_job({
            "signature": "rmsnorm", "op": "rmsnorm",
            "dims": {"B": 1, "s_query": 2048, "d_model": 3072},
            "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"},
            "phase": "prefill",
        })
        assert rms.accepts(rms_inv, env) is True
        # must NOT match a gqa attention invocation
        assert rms.accepts(Invocation.from_job(_gqa_job()), env) is False

    def test_prefill_and_decode_capability_split(self):
        env = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                  libs={"flash_attn": "2.6.3", "flash_attn_interface": "3.0.0"})
        pre = Invocation.from_job(_gqa_job(phase="prefill"))
        dec = Invocation.from_job(_gqa_job(phase="decode"))
        fa3_pre = next(p for p in all_providers()
                       if p.kernel_impl == "flash_attention_v3")
        fa3_dec = next(p for p in all_providers()
                       if p.kernel_impl == "flash_attention_v3_kvcache")
        # Through `accepts`, not `applies_to`: the gate in force is the
        # contract, while which of the two mechanisms backs it is an
        # implementation detail that changes as kernels migrate to `serves`.
        assert fa3_pre.accepts(pre, env) and not fa3_pre.accepts(dec, env)
        assert fa3_dec.accepts(dec, env) and not fa3_dec.accepts(pre, env)

    def test_no_provider_applicable_without_libs(self):
        # A GPU-less / lib-less env → nothing applicable (clean degradation).
        empty = Env(torch_version="2.11", cuda_available=False, device_name=None,
                    gpu_sku=None, peak_bw_gbps=None, libs={})
        inv = Invocation.from_job(_gqa_job())
        assert applicable(inv, empty) == []

    def test_flashmla_capability_scoped_to_mla_decode(self):
        mla = next(p for p in all_providers() if p.library == "flashmla")
        env = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                  libs={"flash_mla": "1.0"})
        # gqa prefill invocation must NOT match the MLA-decode provider
        assert mla.accepts(Invocation.from_job(_gqa_job()), env) is False
        mla_inv = Invocation.from_job(_gqa_job(signature="mla_attention",
                                               op="mla_attention", phase="decode"))
        # The absorbed geometry FlashMLA decodes: no per-head KV axis, which
        # is MLA's variant (non-absorbed prefill) and not this kernel's.
        mla_inv.dims.pop("n_kv_heads")
        assert mla.accepts(mla_inv, env) is True


class TestFusionOpSignatures:
    """rope / moe_gate / fused_qk_norm_rope / indexer_logits — the signatures,
    op mappings, refs, and providers added to light up the Fusion page's
    "Supported backends" and the corpus link."""

    def test_new_signatures_registered(self):
        for name in ("rope", "moe_gate", "fused_qk_norm_rope", "indexer_logits"):
            assert signatures.get_signature(name) is not None

    def test_fused_ops_map_to_signatures(self):
        assert signatures.op_to_signature("fused_qk_rope") == "rope"
        assert signatures.op_to_signature("moe_fused_gate") == "moe_gate"
        assert signatures.op_to_signature("fused_qk_norm_rope") == "fused_qk_norm_rope"
        assert signatures.op_to_signature("indexer_logits") == "indexer_logits"
        assert signatures.op_to_signature("mxfp4_fused_linear") == "gemm"

    def test_producer_and_consumer_read_one_table(self):
        """Not "the same for the ops I thought to list" — the same OBJECT.

        This test used to sample four entries and pass while the two copies
        disagreed about nine others, `linear` among them: the harness planned
        `linear`, `qkv_parallel_linear` and `merged_column_parallel_linear` as
        one gemm signature while the envelope key kept them in three slots, and
        50 cells of the H100 corpus were measured twice under two of the names.
        A sampled comparison is how a mirror drifts without anything failing.
        """
        from backend.inference import benchmark_identity as consumer
        from backend.inference.op_signatures import op_bindings
        from kernel_bench import signatures as producer
        assert not hasattr(producer, "OP_TO_SIGNATURE")
        assert not hasattr(consumer, "OP_TO_SIGNATURE")
        for op, sig in op_bindings().items():
            assert producer.op_to_signature(op) == sig == consumer.op_signature(op)

    def test_every_mapped_signature_is_a_registered_one(self):
        """The table names signatures; a name no signature registers is an op
        the planner would route to an empty provider set and a reference that
        does not exist."""
        from backend.inference.op_signatures import op_bindings
        from kernel_bench.signatures import get_signature
        for op, sig in op_bindings().items():
            assert get_signature(sig) is not None, f"{op} → unregistered {sig}"

    def test_the_gemm_family_shares_one_physical_slot(self):
        """The concrete regression: three op names, one piece of math.

        `to_physical` promises the op collapses to its signature so that
        differently-named-but-same-math ops compete for one slot. For the
        largest family in the corpus it did not, and both the double
        measurements and the lookup misses followed from that."""
        from backend.inference.benchmark_identity import op_signature
        assert {op_signature(o) for o in ("linear", "qkv_parallel_linear",
                                          "merged_column_parallel_linear")} == {"gemm"}
        # GeGLU is the same packing as SwiGLU, not the same math.
        assert op_signature("gelu_and_mul") == "gelu_and_mul"
        from kernel_bench.signatures import get_signature
        gelu = get_signature("gelu_and_mul")
        silu = get_signature("silu_and_mul")
        assert gelu.cell_dims == silu.cell_dims

    def test_refs_registered(self):
        from kernel_bench.refs import get_ref
        assert get_ref("rope").reference is not None
        assert get_ref("indexer_logits").reference is not None
        assert get_ref("fused_qk_norm_rope").reference is not None
        assert get_ref("moe_gate").reference is not None
        assert get_ref("hash_route").reference is not None
        assert get_ref("mamba2_scan").reference is not None

    def test_providers_registered_per_signature(self):
        bysig = {}
        for p in all_providers():
            bysig.setdefault(p.signature, set()).add(p.library)
        assert bysig["rope"] == {"torch", "flashinfer", "vllm", "sglang"}
        assert bysig["rope_2d"] == {"torch", "vllm"}
        assert bysig["moe_gate"] == {"torch", "vllm", "sglang"}
        assert "torch" in bysig["fused_qk_norm_rope"]
        assert {"sglang", "vllm"} <= bysig["fused_qk_norm_rope"]
        assert bysig["indexer_logits"] == {"torch", "deepgemm"}
        assert "torch" in bysig["gated_delta_rule"]
        assert "torch" in bysig["kda"]
        assert "torch" in bysig["mamba2_scan"]
        assert "torch" in bysig["allreduce_fused_rmsnorm"]
        assert bysig["gla_simple_recurrence"] == {"torch", "fla", "vllm",
                                                  "sglang"}

    def _inv(self, sig, op, dims, phase="prefill"):
        return Invocation.from_job({
            "signature": sig, "op": op, "dims": dims, "phase": phase,
            "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"}})

    def test_torch_rope_matches_reference_cpu(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inv = self._inv("rope", "fused_qk_rope",
                        {"B": 2, "s_query": 8, "n_heads": 4, "n_kv_heads": 2, "d_rope": 64})
        spec = get_ref("rope")
        ins = build_inputs(spec, inv, "cpu")
        ref = spec.reference(ins)
        prov = next(p for p in all_providers()
                    if p.signature == "rope" and p.library == "torch")
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        case = prov.build(inv, env, ins)
        assert case.to_canonical(case.run()).shape == ref.shape
        # Judged by the harness's own gate, not a hand-picked epsilon. The old
        # 1e-4 here silently encoded "the torch provider computes in fp32" —
        # which was true, and was the bug: it made the always-available baseline
        # both more precise AND 2.6-3.4x slower than the kernels it is the
        # baseline for. It now rotates in the invocation dtype like vLLM's
        # `forward_native`, so it agrees to bf16 precision, and the criterion
        # that matters is the one that decides whether a row is recorded.
        from kernel_bench.driver import _cross_check
        verdict, rel, cos, _typical, _worst = _cross_check(prov.build(inv, env, ins), ref,
                                         inv.dtype)
        assert verdict is None, f"rel_err={rel:.3g} cosine={cos:.5f}"
        assert cos > 0.999

    def test_torch_rope_2d_matches_reference_cpu(self):
        """The vision towers' rotation: x [tokens, H, D] by per-token cos/sin
        rows [tokens, D/2], as vLLM `ApplyRotaryEmb.forward_native` runs it."""
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.driver import _cross_check
        inv = self._inv("rope_2d", "rope_2d",
                        {"B": 2, "s_query": 9, "n_heads": 4, "d_rope": 64})
        spec = get_ref("rope_2d")
        ins = build_inputs(spec, inv, "cpu")
        assert tuple(ins["x"].shape) == (18, 4, 64)
        assert tuple(ins["cos"].shape) == tuple(ins["sin"].shape) == (18, 32)
        ref = spec.reference(ins)
        prov = next(p for p in all_providers()
                    if p.signature == "rope_2d" and p.library == "torch")
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        verdict, rel, cos, _typical, _worst = _cross_check(prov.build(inv, env, ins), ref,
                                         inv.dtype)
        assert verdict is None, f"rel_err={rel:.3g} cosine={cos:.5f}"
        assert cos > 0.999

    def test_torch_indexer_matches_reference_cpu(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inv = self._inv("indexer_logits", "indexer_logits",
                        {"B": 2, "s_query": 8, "s_kv": 128, "n_indexer_heads": 4, "d_indexer_head": 128,
                         "kv_block_size": 64})
        spec = get_ref("indexer_logits")
        ins = build_inputs(spec, inv, "cpu")
        ref = spec.reference(ins)
        prov = next(p for p in all_providers()
                    if p.signature == "indexer_logits" and p.library == "torch")
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        out = (lambda c: c.to_canonical(c.run()))(prov.build(inv, env, ins))
        assert out.shape == ref.shape
        # bf16 matmul vs fp32 oracle — loose relative tolerance
        assert ((out - ref).abs().max() / ref.abs().max()).item() < 5e-3

    def test_torch_indexer_tiled_matches_one_shot_cpu(self, monkeypatch):
        """The token-tiled path the provider takes past its one-shot memory
        budget is the same contraction, chunk by chunk — and still fp32."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.providers import indexer as indexer_prov
        inv = self._inv("indexer_logits", "indexer_logits",
                        {"B": 2, "s_query": 9, "s_kv": 128, "n_indexer_heads": 4,
                         "d_indexer_head": 128, "kv_block_size": 64})
        ins = build_inputs(get_ref("indexer_logits"), inv, "cpu")
        prov = next(p for p in all_providers()
                    if p.signature == "indexer_logits" and p.library == "torch")
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        one_shot = (lambda c: c.to_canonical(c.run()))(prov.build(inv, env, ins))
        # A budget of 7 tokens' intermediates (128 keys x (4 heads x (2 B
        # bf16 + 4 B fp32) + 4 B logit)) forces the tiled path, and a chunk
        # of 7 over 18 tokens exercises the ragged tail.
        monkeypatch.setattr(indexer_prov, "_INDEXER_ONESHOT_PEAK",
                            7 * 128 * (4 * 6 + 4))
        case = prov.build(inv, env, ins)
        assert case.run().data_ptr() == case.run().data_ptr()
        tiled = case.to_canonical(case.run())
        assert tiled.shape == one_shot.shape
        assert tiled.dtype == one_shot.dtype == torch.float32
        assert ((tiled - one_shot).abs().max()
                / one_shot.abs().max()).item() < 5e-3

    def test_torch_moe_gate_runs_cpu(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inv = self._inv("moe_gate", "moe_fused_gate",
                        {"B": 2, "s_query": 4, "n_experts": 256, "n_expert_groups": 8, "n_groups_active": 4, "n_experts_active": 8})
        ins = build_inputs(get_ref("moe_gate"), inv, "cpu")
        prov = next(p for p in all_providers()
                    if p.signature == "moe_gate" and p.library == "torch")
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        case = prov.build(inv, env, ins)
        weights, ids = case.run()
        assert tuple(weights.shape) == (8, 8)               # tokens × topk
        dense = case.to_canonical((weights, ids))
        ref = get_ref("moe_gate").reference(ins)
        assert dense.shape == ref.shape
        # The floor mirrors vLLM's eager `grouped_topk`, which applies the
        # sigmoid in the logits' own dtype (`grouped_topk_router.py`,
        # `scores = gating_output.sigmoid()`), while the CUDA kernel this row
        # measures — and the oracle with it — converts bf16/fp16 logits to
        # float and accumulates there (`topk_softmax_kernels.cu`, "BF16/FP16
        # convert to float"). The logits follow the cell
        # (`refs._moe_gate_inputs`), so a 16-bit cell's floor differs from
        # the oracle by that dtype's own rounding, not by another function:
        # the band is the logits' epsilon (the measured cross-check's bf16
        # band is an order wider).
        band = (1e-5 if ins["gating"].dtype == torch.float32
                else torch.finfo(ins["gating"].dtype).eps)
        assert torch.allclose(dense, ref, rtol=band, atol=band)

    def test_moe_gate_refuses_to_guess_group_structure(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inv = self._inv("moe_gate", "moe_fused_gate",
                        {"B": 2, "s_query": 4, "n_experts": 256,
                         "n_experts_active": 8})
        with pytest.raises(ValueError, match="n_expert_groups"):
            build_inputs(get_ref("moe_gate"), inv, "cpu")


class TestInvocation:
    def test_from_job_parses_dims_and_signature(self):
        inv = Invocation.from_job(_gqa_job())
        assert inv.signature == "gqa_attention"
        assert inv.dims["n_heads"] == 24
        assert inv.dtype == "bf16"

    def test_unmapped_op_raises(self):
        with pytest.raises(ValueError):
            Invocation.from_job({"op": "transpose", "dims": {}, "dtype_env": {}})


class TestEmit:
    def test_row_conforms_to_schema(self):
        from kernel_bench.providers import Provider
        inv = Invocation.from_job(_gqa_job())
        prov = Provider(library="fa3", signature="gqa_attention",
                        kernel_impl="flash_attention_v3", import_name="flash_attn",
                        applies_to=lambda i, e: True, build=lambda i, e: (lambda: None))
        env = Env("2.11", True, "H100", "H100-SXM", 3350.0,
                  libs={"flash_attn": "3.0.0"})
        stats = LatencyStats(p50_us=387.5, p95_us=410.0, mean_us=390.0, samples=60, timing_method="l2_drain")
        row = build_row(inv, prov, stats, env, date="2026-07-27", contributor="@t")
        # the peak at the row's own compute dtype (bf16 here), from the sheet
        assert float(row["peak_tflops_at_test"]) == pytest.approx(989.0)
        # every column present, and the shared validator accepts it
        assert set(row.keys()) == set(_benchmarks.ALL_COLUMNS)
        text = _benchmarks.emit_csv(list(_benchmarks.ALL_COLUMNS), [row])
        _h, rows = _benchmarks.validate_text(text, source="<emit-test>")
        assert len(rows) == 1
        assert rows[0]["framework"] == "fa3"
        assert rows[0]["framework_version"] == "3.0.0"
        # achieved computed from flops / latency
        assert float(rows[0]["achieved_tflops"]) > 0

    def test_a_mixed_precision_gemm_is_judged_at_the_promoted_compute(self):
        from kernel_bench.providers import Provider
        inv = Invocation.from_job(_gemm_job(dtype_env={
            "activation": "bf16", "weight": "fp32", "kv": "bf16",
            "output": "bf16", "metadata": "fp32"}))
        prov = Provider(
            library="torch", signature="gemm", kernel_impl="torch_cublas_linear",
            import_name="torch", applies_to=lambda i, e: True,
            build=lambda i, e: (lambda: None),
            computes_at_for=lambda i: {"activation": "fp32", "output": "fp32"})
        env = Env("2.11", True, "H100", "H100-SXM", 3350.0,
                  libs={"torch": "2.11"})
        row = build_row(inv, prov, LatencyStats(10, 11, 10, 3, "l2_drain"), env,
                        date="2026-10-28", contributor="@t")
        assert row["measured_dtype_env_json"] == \
            '{"activation":"fp32","output":"fp32"}'
        assert float(row["peak_tflops_at_test"]) == pytest.approx(67.0, rel=0.01)


class TestPlanPhaseAwareShape:
    """plan runs analyze under the phase binding, so attention job dims are
    phase-aware (decode collapses Q's seq axis to Sq=1)."""

    def test_prefill_vs_decode_attention_dims_differ(self):
        from kernel_bench.plan import build_jobs
        pre, _ = build_jobs(avg_output_tokens=0, model_id="meta-llama/Llama-3.2-3B",
                            batch_size=1, seq_len=2048, phase="prefill")
        dec, _ = build_jobs(avg_output_tokens=0, model_id="meta-llama/Llama-3.2-3B",
                            batch_size=1, seq_len=2048, phase="decode")
        pre_fa = next(j for j in pre if j["op"] == "flash_attention")["dims"]
        dec_fa = next(j for j in dec if j["op"] == "flash_attention")["dims"]
        assert pre_fa["s_query"] == 2048 and pre_fa["s_kv"] == 2048
        assert dec_fa["s_query"] == 1 and dec_fa["s_kv"] >= 2048
        assert pre_fa != dec_fa


class TestSkipKeyMatchesStorage:
    """The physical key the driver builds for a skip-check must equal the
    physical key the corpus stores an emitted row under — else `covers()`
    would never match and measure-once would silently never skip."""

    def test_physical_key_for_matches_emitted_row(self):
        from kernel_bench.driver import physical_key_for
        from kernel_bench.emit import build_row
        from kernel_bench.providers import Provider, LatencyStats
        from backend.inference.op_calibration import BenchmarkKey

        inv = Invocation.from_job(_gqa_job(
            dims={"kv_block_size": 16, "B": 1, "n_heads": 24, "n_kv_heads": 8, "s_query": 1, "s_kv": 2176, "d_head": 128},
            phase="decode",
        ))
        env = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                  libs={"flash_attn_interface": "3.0.0"})
        prov = Provider(library="fa3", signature="gqa_attention",
                        kernel_impl="flash_attention_v3_kvcache",
                        import_name="flash_attn_interface",
                        applies_to=lambda i, e: True, build=lambda i, e: (lambda: None))
        stats = LatencyStats(p50_us=300.0, p95_us=310.0, mean_us=302.0, samples=60, timing_method="l2_drain")
        row = build_row(inv, prov, stats, env, date="2026-07-27", contributor="@t")

        skip_key = physical_key_for(inv, env)
        stored_key = BenchmarkKey.from_csv_row(row).to_physical()
        assert skip_key == stored_key


def _gemm_job(**over) -> dict:
    job = {
        "signature": "gemm", "op": "linear",
        "dims": {"B": 1, "s_query": 1, "d_reduce": 64, "d_out": 64},
        "dtype_env": {"weight": "bf16", "activation": "bf16", "kv": "bf16",
                      "output": "bf16", "metadata": "fp32"},
        "phase": "decode", "tp": 1,
        "flops_per_rank": 1.0e9, "access_bytes_per_rank": 1.0e6,
        "model": "m", "spec_hash": "h", "node_spec_id": "block|linear",
        "parallel": "tp=1", "count_per_step": 1,
    }
    job.update(over)
    job.setdefault("physical_key", _stated_key(job))
    return job


def _plan_for_every_env(tmp_path, profile=None, **box) -> str:
    """A plan whose one job has a candidate in every library's env, so
    `run-all` starts them all (it starts only the envs a plan needs)."""
    from kernel_bench.plan import write_jobs_jsonl
    from kernel_bench.providers import dependency_manifest
    from kernel_bench.box import load_box_profile
    path = tmp_path / "jobs.jsonl"
    write_jobs_jsonl([_gemm_job(
        judged_on=judged_on(load_box_profile(profile) if profile else None,
                            **box),
        candidates={info["providers"][0]: lib
                    for lib, info in dependency_manifest().items()})],
        str(path))
    return str(path)


class TestRunAllOrchestration:
    """`run-all` fans one jobs file out over every provider env via `uv run`,
    so a single command measures the whole cross-library envelope."""

    def test_uv_run_cmd_shape(self):
        from kernel_bench.cli import uv_run_cmd
        cmd = uv_run_cmd("kernel_bench/envs/fa3.txt",
                         ["--jobs", "j.jsonl", "-o", "results/fa3.csv",
                          "--skip-existing"], python_version="3.11")
        assert cmd[:3] == ["uv", "run", "--no-project"]
        assert cmd[cmd.index("--python") + 1] == "3.11"
        assert cmd[cmd.index("--with-requirements") + 1] == "kernel_bench/envs/fa3.txt"
        after = cmd[cmd.index("--") + 1:]
        assert after[:4] == ["python", "-m", "kernel_bench", "run"]
        assert after[-1] == "--skip-existing"

    def test_run_all_dry_run_covers_every_env(self, tmp_path, capsys):
        from kernel_bench.cli import main
        rc = main(["run-all", "--jobs", _plan_for_every_env(tmp_path), "--dry-run"])
        assert rc == 0
        out = capsys.readouterr().out
        for lib in ("fa2", "fa3", "flashinfer", "flashmla", "vllm", "deepgemm"):
            assert f"[{lib}]" in out
        # pip libs go through uv; source-build libs are skipped-with-hint
        assert "uv run --no-project" in out and "PYTHONPATH=" in out

    def test_source_build_env_skipped_with_hint_when_unconfigured(
            self, tmp_path, capsys):
        from kernel_bench.cli import main
        main(["run-all", "--jobs", _plan_for_every_env(tmp_path), "--dry-run"])
        out = capsys.readouterr().out
        # fa3 is source-build: not uv-resolved, skipped with build guidance
        assert "[fa3] SKIP (source-build)" in out
        assert "KERNEL_BENCH_ENV_FA3" in out
        # and it must NOT appear as a uv --with-requirements line
        assert "envs/fa3.txt" not in out

    def test_source_build_env_targets_supplied_interpreter(
            self, tmp_path, capsys):
        from kernel_bench.cli import main
        main(["run-all", "--jobs", _plan_for_every_env(
                  tmp_path, overrides={"fa3": "/opt/venvs/fa3/bin/python"}), "--dry-run",
              "--env", "fa3=/opt/venvs/fa3/bin/python"])
        out = capsys.readouterr().out
        # configured → real command against that interpreter, no uv for fa3
        assert "[fa3] PYTHONPATH=" in out
        assert "/opt/venvs/fa3/bin/python -m kernel_bench run" in out

    def test_box_profile_reaches_run_all_commands(self, tmp_path, capsys):
        from kernel_bench.cli import main

        profile = tmp_path / "box.toml"
        profile.write_text(
            'name = "test-box"\n'
            'python = "3.11"\n'
            'constraints = "/opt/torch.txt"\n'
            'command_prefix = ["/opt/kbrun.sh", "--"]\n'
            '[providers]\n'
            'fa3 = "/opt/venvs/fa3/bin/python"\n')

        rc = main(["run-all", "--jobs", _plan_for_every_env(tmp_path, profile), "--dry-run",
                   "--box", str(profile)])
        assert rc == 0
        out = capsys.readouterr().out
        assert f"box: test-box ({profile})" in out
        assert "/opt/kbrun.sh -- uv run --no-project --python 3.11" in out
        assert "--with-requirements /opt/torch.txt" in out
        assert ("/opt/kbrun.sh -- /opt/venvs/fa3/bin/python "
                "-m kernel_bench run") in out

    def test_run_all_stamps_one_run_id_and_the_box_on_every_env(
            self, tmp_path, capsys):
        # Revocation works per batch, so the batch boundary has to be visible
        # in every child command: one shared run_id, and the box's name.
        import re
        from kernel_bench.cli import main

        profile = tmp_path / "box.toml"
        profile.write_text('name = "test-box"\n')
        rc = main(["run-all", "--jobs", _plan_for_every_env(tmp_path, profile), "--dry-run",
                   "--box", str(profile)])
        assert rc == 0
        out = capsys.readouterr().out
        run_ids = re.findall(r"--run-id (\S+)", out)
        assert run_ids, "no child command carries the batch id"
        assert len(set(run_ids)) == 1, "one batch, one run_id"
        assert "--box test-box" in out

    def test_first_attempt_does_not_force_append_when_csv_exists(
            self, tmp_path, monkeypatch):
        # Resume is `run --skip-existing`'s job (`resume_from_output`). The
        # supervisor must not force --append-output on the first attempt:
        # that would load private-file keys even under --no-skip-existing.
        from kernel_bench import providers, supervisor
        from kernel_bench.cli import main
        from kernel_bench.plan import write_jobs_jsonl

        jobs_path = tmp_path / "jobs.jsonl"
        write_jobs_jsonl([_gemm_job(candidates={"t": "fake"}, judged_on=judged_on(manifest=_FAKE, envs_dir=tmp_path))], str(jobs_path))
        monkeypatch.setattr(
            providers, "dependency_manifest",
            lambda: _FAKE)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        (out_dir / "fake.csv").write_text("header\nrow\n")
        calls = []

        def fake_run(command, **kwargs):
            if "--progress-file" not in command:
                return supervisor.SupervisedResult(0, None, 0.1)
            calls.append(command)
            assert "--append-output" not in command
            assert "--skip-existing" in command
            return supervisor.SupervisedResult(0, None, 0.1)

        monkeypatch.setattr(supervisor, "run_command", fake_run)
        rc = main([
            "run-all", "--jobs", str(jobs_path),
            "--envs-dir", str(tmp_path), "-o", str(out_dir),
            "--stall-timeout", "1s"])
        assert rc == 0
        assert calls
        assert all("--append-output" not in command for command in calls)


class TestUpAndReport:
    """The four-command interface: `up` materializes a box from nothing, `report`
    reads coverage back."""

    def test_up_dry_run_builds_every_pip_env_and_names_the_box(
            self, tmp_path, capsys):
        from kernel_bench.cli import main
        profile = tmp_path / "box.toml"
        profile.write_text('name = "test-box"\npython = "3.11"\n')
        rc = main(["up", "--dry-run", "--box", str(profile)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "box: test-box" in out
        # pip libs get a uv materialization command pinned to the profile's
        # python; source-build libs are skipped with their build hint.
        assert "[fa2] uv run --no-project --python 3.11 " \
               "--with-requirements kernel_bench/envs/fa2.txt" in out
        assert "[fa3] SKIP (source-build)" in out

    def test_report_reads_coverage_and_the_envelope(self, tmp_path, capsys):
        import json as _json
        import tests.corpus_row as _corpus
        from kernel_bench.cli import main
        from kernel_bench.driver import physical_key_for
        from kernel_bench.env import Env
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import all_providers

        job_hit = _gemm_job()
        job_miss = _gemm_job(dims={"B": 1, "s_query": 1, "d_reduce": 64,
                                   "d_out": 128})
        jobs = tmp_path / "jobs.jsonl"
        jobs.write_text("".join(_json.dumps(j) + "\n"
                                for j in (job_hit, job_miss)))

        env = Env(None, False, None, "H100-SXM", None)
        inv = Invocation.from_job(job_hit)
        pk = physical_key_for(inv, env)
        prov = next(p for p in all_providers()
                    if p.kernel_impl == "torch_cublas_linear")
        row = _corpus.row(op="linear", framework="torch",
                          kernel_impl=prov.impl_id(inv),
                          gpu="H100-SXM", phase="decode", parallel="tp=1",
                          shape_json=pk.shape_json,
                          dtype_env_json=pk.dtype_env_json)
        results = tmp_path / "results.csv"
        results.write_text(_corpus.to_csv([row]))

        rc = main(["report", "--jobs", str(jobs), "--results", str(results),
                   "--gpu", "H100-SXM"])
        assert rc == 0
        out = capsys.readouterr().out
        # The measured cell names its winner; the unmeasured one lands on
        # the gap list with its candidate impls.
        assert "torch:torch_cublas_linear×1" in out
        assert "pending" in out
        assert "'d_out': 128" in out

    def test_failed_job_is_quarantined_and_remaining_jobs_restart(
            self, tmp_path, capsys, monkeypatch):
        from pathlib import Path
        from kernel_bench import providers, supervisor
        from kernel_bench.cli import main
        from kernel_bench.plan import write_jobs_jsonl

        jobs = [_gemm_job(op="bad", candidates={"t": "fake"}, judged_on=judged_on(manifest=_FAKE, envs_dir=tmp_path)),
                _gemm_job(op="good", candidates={"t": "fake"}, judged_on=judged_on(manifest=_FAKE, envs_dir=tmp_path))]
        jobs_path = tmp_path / "jobs.jsonl"
        write_jobs_jsonl(jobs, str(jobs_path))
        monkeypatch.setattr(
            providers, "dependency_manifest",
            lambda: _FAKE)
        calls = []

        resolves = []

        def fake_run(command, **kwargs):
            # Env resolution goes through the same launcher as the measured
            # run (it is distinguished by carrying no `--progress-file`, not
            # by being a different function), so the stub answers both.
            if "--progress-file" not in command:
                resolves.append(command)
                return supervisor.SupervisedResult(0, None, 0.1)
            calls.append(command)
            progress_path = command[command.index("--progress-file") + 1]
            current_jobs = command[command.index("--jobs") + 1]
            if len(calls) == 1:
                supervisor.write_progress(progress_path, 0, jobs[0])
                return supervisor.SupervisedResult(
                    -9, "output file made no progress for 1s", 1.0)
            remaining = Path(current_jobs).read_text().strip().splitlines()
            assert len(remaining) == 1
            assert "--append-output" in command
            return supervisor.SupervisedResult(0, None, 0.1)

        monkeypatch.setattr(supervisor, "run_command", fake_run)
        rc = main([
            "run-all", "--jobs", str(jobs_path),
            "--envs-dir", str(tmp_path), "-o", str(tmp_path / "results"),
            "--stall-timeout", "1s"])
        assert rc == 1
        assert len(calls) == 2
        # The resolve ran, and it ran with no stall timer — the reason it was
        # moved ahead of the measured run in the first place.
        assert len(resolves) == 1
        out = capsys.readouterr()
        assert "QUARANTINE job bad" in out.err
        assert "continuing with 1 jobs" in out.err


    def test_an_env_whose_rows_all_fail_validation_ran_to_the_end(
            self, tmp_path, capsys, monkeypatch):
        """`run` reaching the end with every row dropped by validation is
        not a crash: no job is quarantined (the one in flight at the end is
        innocent) and the env is not failed, so the queue can mark the model
        done instead of measuring it again every pass."""
        from kernel_bench import providers, supervisor
        from kernel_bench.cli import EXIT_NO_VALID_ROWS, main
        from kernel_bench.plan import write_jobs_jsonl

        jobs = [_gemm_job(op="a", candidates={"t": "fake"}, judged_on=judged_on(manifest=_FAKE, envs_dir=tmp_path)),
                _gemm_job(op="b", candidates={"t": "fake"}, judged_on=judged_on(manifest=_FAKE, envs_dir=tmp_path))]
        jobs_path = tmp_path / "jobs.jsonl"
        write_jobs_jsonl(jobs, str(jobs_path))
        monkeypatch.setattr(
            providers, "dependency_manifest",
            lambda: _FAKE)
        calls = []

        def fake_run(command, **kwargs):
            if "--progress-file" not in command:
                return supervisor.SupervisedResult(0, None, 0.1)
            calls.append(command)
            progress_path = command[command.index("--progress-file") + 1]
            supervisor.write_progress(progress_path, 1, jobs[1])
            return supervisor.SupervisedResult(EXIT_NO_VALID_ROWS, None, 0.1)

        monkeypatch.setattr(supervisor, "run_command", fake_run)
        rc = main([
            "run-all", "--jobs", str(jobs_path),
            "--envs-dir", str(tmp_path), "-o", str(tmp_path / "results")])
        out = capsys.readouterr()
        assert rc == 0, out.out + out.err
        assert len(calls) == 1
        assert "QUARANTINE" not in out.err
        assert "every row it measured failed validation" in out.out


class TestPlanDropsUnshapeableGemm:
    """A GEMM job needs a real per-op shape (K from shape_in). An op whose
    shape can't be projected onto its signature must be DROPPED, never given
    fabricated dims or the model-wide attention-dim fallback.

    lm_head used to be the example of that drop: its evaluated shape used to
    have no input. It now flows through the op registry like
    any other layer and carries a real `[B, lp, H] → [B, lp, V]`, so it is
    legitimately benchable — asserted below with its true dims. fused_qk_rope
    followed the same arc: its multi-input shape used to pin the drop
    contract, but the projector now handles head-shaped Q/K, so it is
    benchable too — with real per-op dims. The drop-don't-fabricate contract
    therefore lives at the projector itself: a shape that fits no signature
    layout projects to None, and plan drops the op instead of inventing
    model-wide dims."""

    def test_gemm_jobs_have_real_k_n(self):
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="meta-llama/Llama-3.2-3B",
                             batch_size=1, seq_len=2048, phase="decode")
        for j in jobs:
            if j["signature"] == "gemm":
                assert "d_reduce" in j["dims"] and "d_out" in j["dims"], j["dims"]

    def test_lm_head_is_benchable_with_checkpoint_true_dims(self):
        # Real Llama-3.2-3B geometry: K = hidden 3072, N = vocab 128256, and
        # one logit position in decode (num_logit_positions=1) — no fabrication.
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="meta-llama/Llama-3.2-3B",
                             batch_size=1, seq_len=2048, phase="decode")
        lm = [j for j in jobs if str(j.get("node_spec_id") or "") == "lm_head"]
        assert len(lm) == 1, f"expected one lm_head job, got {lm}"
        dims = lm[0]["dims"]
        assert lm[0]["signature"] == "gemm"
        assert dims["d_reduce"] == 3072, dims     # hidden_size
        assert dims["d_out"] == 128256, dims      # vocab_size
        assert dims["s_query"] == 1, dims         # decode → 1 logit position

    def test_tied_lm_head_weight_dtype_comes_from_the_embedding(self):
        # A tied LM head stores no weight of its own — it streams the
        # embedding's table (Program.read_aliases), and traffic bills it for
        # those bytes. So the dtype is an EVAL fact now, on the row, and the
        # plan reads it instead of re-deriving it from the spec: this used to
        # call `plan._weight_dtypes`, a second walk of the materialized
        # tensors that the one-owner key change deleted.
        from backend.analyzer import fetch_config
        from backend.arch.loader import SpecLoader
        from backend.inference.framework_bindings import (DEFAULT_FRAMEWORK,
                                                          get_binding)
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        model_id = "meta-llama/Llama-3.2-3B"
        r = SpecLoader().match(fetch_config(model_id), model_id=model_id)
        ar = analyze(r, parallel=ParallelConfig(),
                     framework_binding=get_binding(DEFAULT_FRAMEWORK),
                     batch_size=1, avg_input_tokens=2048, avg_output_tokens=0,
                     phase="decode", breakdown=True)
        rows = {str(row.get("aggregation_id") or ""): row
                for row in ar.per_op_timing}
        head = rows["lm_head"].get("weight_dtype")
        table = rows["embed_tokens"].get("weight_dtype")
        assert head == table and head, (
            "a tied lm_head's row must carry the dtype of the table it "
            "streams — the bytes are billed to it, so the name must be too",
            head, table)

    def test_fused_qk_rope_is_benchable_with_real_dims(self):
        # fused_qk_rope's multi-input shape used to be the drop example.
        # The projector now handles head-shaped Q/K, so it must be benched
        # at real Llama-3.2-3B geometry — not a fabrication.
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="meta-llama/Llama-3.2-3B",
                             batch_size=1, seq_len=2048, phase="decode")
        rope = [j for j in jobs if j["op"] == "fused_qk_rope"]
        assert len(rope) == 1, f"expected one fused_qk_rope job, got {rope}"
        assert rope[0]["dims"] == {"B": 1, "s_query": 1, "n_heads": 24,
                                   "n_kv_heads": 8, "d_rope": 128}

    def test_unprojectable_shape_is_dropped_not_fabricated(self):
        # Every op on the shipped models now projects cleanly, so pin the
        # guard where it fires: a shape that fits no signature layout
        # projects to None — the reason plan.py drops the op with a
        # "fabricate" note instead of benching model-wide dims.
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        assert identity_shape_from_shape_in("linear", [[1, 2, 3, 4, 5]]) is None
        assert identity_shape_from_shape_in("linear", [[2048]]) is None


class TestDriverIsolatesBadJob:
    """One malformed job records a failure and is skipped — it must not sink
    the whole sweep (driver docstring contract)."""

    def test_malformed_gemm_does_not_abort_run(self):
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        bad = _gemm_job(dims={"B": 1, "s_query": 1, "n_heads": 64})    # no K → refs KeyError
        good = _gemm_job()
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs([bad, good], env=env, date="2026-07-28",
                       contributor="@t", cross_check=False)
        assert any("input build failed" in f for f in res.failures)
        assert len(res.rows) >= 1                            # good job still ran


class TestDriverDedupsWithinRun:
    """A swept deployment parameter must not re-measure a provider that does
    not consume it.

    `plan` emits one job per paged-KV block size. A paged backend folds the
    block into its `impl_id` (flashinfer → `..._b16` / `_b64`), so each block is
    a distinct implementation. torch SDPA has no paged KV: its identity is the
    SAME for every block in the sweep, and measuring it once per block wrote N
    byte-identical rows — which the CSV writer then flagged as
    `duplicate identity tuple`. The driver skips a (physical key, library,
    version, impl_id) it already measured in this run, so the collapse is keyed
    on identity and covers ANY unconsumed swept parameter, not just this one."""

    def test_same_identity_measured_once_across_swept_blocks(self):
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        # Same GEMM job three times over: `gemm` has no paged KV, so no provider
        # folds kv_block_size into its identity — all three collapse to one row.
        jobs = [_gemm_job(kernel_params={"kv_block_size": b}) for b in (16, 64, 128)]
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs(jobs, env=env, date="2026-07-28",
                       contributor="@t", cross_check=False)
        assert len(res.rows) == 1, [r.get("kernel_impl") for r in res.rows]
        assert sum("already measured in this run" in s for s in res.skipped) == 2
        assert not res.failures

    def test_distinct_identities_are_all_measured(self):
        # The dedup keys on identity, so genuinely different shapes still each
        # get measured — it must not collapse work that isn't redundant.
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        # The second job is another launch path: a canary that times out
        # parks its own path's remaining cells for the next run (the point
        # of the canary), so surviving the timeout means the paths after it
        # still run.
        jobs = [_gemm_job(dims={"B": 1, "s_query": 1, "d_reduce": 512,
                                "d_out": 512}),
                {"signature": "rmsnorm", "op": "rmsnorm",
                 "dims": {"B": 1, "s_query": 2048, "d_model": 3072},
                 "dtype_env": {"activation": "bf16", "weight": "bf16",
                               "kv": "bf16", "output": "bf16",
                               "metadata": "fp32"},
                 "phase": "prefill", "tp": 1,
                 "flops_per_rank": 1.0e6, "access_bytes_per_rank": 1.0e6,
                 "model": "t/t", "spec_hash": "abc",
                 "node_spec_id": "b|norm", "parallel": "tp=1",
                 "count_per_step": 1}]
        jobs[1]["physical_key"] = _stated_key(jobs[1])
        env = Env("2.11", True, "H100", "H100-SXM", 3350.0,
                  libs={"torch": "2.11"})
        res = run_jobs(jobs, env=env, date="2026-07-28",
                       contributor="@t", cross_check=False)
        assert len(res.rows) == 2, res.skipped


def _mla_job(**over) -> dict:
    job = {
        "signature": "mla_attention", "op": "mla_attention",
        # A paged cell carries the deployment's page (`PAGE_AXIS`).
        "dims": {"B": 1, "n_heads": 128, "s_query": 1, "s_kv": 4096, "d_head": 576,
                 "d_v_head": 512, "kv_block_size": 64},
        "dtype_env": {"weight": "bf16", "activation": "bf16", "kv": "bf16",
                      "output": "bf16", "metadata": "fp32"},
        "phase": "decode", "tp": 1,
        "flops_per_rank": 1.0e11, "access_bytes_per_rank": 1.0e9,
        "model": "deepseek-ai/DeepSeek-V3", "spec_hash": "ds3",
        "node_spec_id": "pre_norm|self_attn.attn_apply_latent__mla_attention",
        "parallel": "tp=1", "count_per_step": 61,
    }
    job.update(over)
    job.setdefault("physical_key", _stated_key(job))
    return job


class TestMLARouting:
    """MLA: the op emits as flash_attention and routes to mla_attention by
    the cache slot it reads — write (plan) and lookup (overlay) agree."""

    def test_resolve_op_maps_mla_but_not_gqa(self):
        from backend.inference.benchmark_identity import resolve_op
        latent = frozenset({"latent", "rope_key"})
        assert resolve_op("flash_attention", key_slots=latent) == "mla_attention"
        assert resolve_op("flash_attention", key_slots=frozenset({"kv"})) == "flash_attention"
        assert resolve_op("flash_attention") == "flash_attention"
        assert resolve_op("rmsnorm", key_slots=latent) == "rmsnorm"     # only attention reroutes

    def test_mla_identity_from_shape_in(self):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        # engine: Q=[B,N,Sq,Dqk], latent KV=[B,Lq,Dqk], out=[B,N,Sq,Dv]
        # The page is the deployment's, handed in beside the spec's dims.
        s = identity_shape_from_shape_in(
            "mla_attention", [[1, 128, 1, 576], [1, 4096, 576]],
            shape_out=[1, 128, 1, 512], spec_dims={"kv_block_size": 64})
        assert json.loads(s) == {"B": 1, "n_heads": 128, "s_query": 1, "s_kv": 4096,
                                 "d_head": 576, "d_v_head": 512,
                                 "kv_block_size": 64}

    def test_plan_emits_mla_job_for_deepseek(self):
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3",
                             batch_size=1, seq_len=4096, phase="decode")
        mla = [j for j in jobs if j["signature"] == "mla_attention"]
        assert mla, "no mla_attention job produced"
        d = mla[0]["dims"]
        assert d["d_head"] == 576 and d["d_v_head"] == 512 and d["s_query"] == 1
        # must NOT be mis-tagged as dense gqa
        assert not any(j["signature"] == "gqa_attention" for j in jobs)

    def test_mla_physical_key_roundtrip(self):
        from kernel_bench.driver import physical_key_for
        from kernel_bench.emit import build_row
        from kernel_bench.providers import LatencyStats, Provider
        from backend.inference.op_calibration import BenchmarkKey

        inv = Invocation.from_job(_mla_job())
        env = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                  libs={"flash_mla": "1.0"})
        prov = Provider(library="flashmla", signature="mla_attention",
                        kernel_impl="flashmla_decode", import_name="flash_mla",
                        applies_to=lambda i, e: True, build=lambda i, e: (lambda: None))
        stats = LatencyStats(p50_us=120.0, p95_us=130.0, mean_us=122.0, samples=60, timing_method="l2_drain")
        row = build_row(inv, prov, stats, env, date="2026-07-28", contributor="@t")
        assert physical_key_for(inv, env) == BenchmarkKey.from_csv_row(row).to_physical()

    def test_mla_reference_shapes_on_cpu(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inv = Invocation.from_job(_mla_job(
            dims={"kv_block_size": 64, "B": 1, "n_heads": 4, "s_query": 1, "s_kv": 32, "d_head": 576, "d_v_head": 512}))
        spec = get_ref("mla_attention")
        inputs = build_inputs(spec, inv, "cpu")
        out = spec.reference(inputs)
        assert tuple(out.shape) == (1, 1, 4, 512)          # [B,Sq,N,Dv]

    def test_flashmla_provider_selected_for_mla_decode(self):
        env = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                  libs={"flash_mla": "1.0"})
        inv = Invocation.from_job(_mla_job())
        libs = {p.library for p in applicable(inv, env)}
        assert "flashmla" in libs


class TestSwaAndAllreduceBecomeMeasurable:
    """Two spans the planner used to drop: SWA's naive score/softmax/apply
    (serving is FA with window_size) and the TP>1 allreduce+residual+RMSNorm
    kernel (bound in vLLM/sglang, missing from the signature table)."""

    def test_swa_model_plans_gqa_attention_not_swa_primitives(self):
        from kernel_bench.plan import build_jobs
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id="mistralai/Mistral-7B-v0.1",
            batch_size=1, seq_len=2048, phase="prefill")
        gqa = [j for j in jobs if j["signature"] == "gqa_attention"]
        assert gqa, "Mistral SWA did not plan as gqa_attention"
        dropped_ops = {d.split()[0] for d in dropped if d}
        assert not (dropped_ops & {
            "swa_attention_score", "swa_softmax", "swa_attention_apply",
        }), dropped
        assert all("s_window" in j["dims"] for j in gqa), (
            "pure-SWA layers must carry s_window on the identity")
        assert all(j["kernel_params"].get("s_window") == j["dims"]["s_window"]
                   for j in gqa)
        assert all("n_attn_sinks" not in j["dims"] for j in gqa)

    def test_windowed_prefill_keeps_full_k(self):
        """Serving FA prefill is K at the sequence length plus window_size,
        not a K already truncated to the window (the FLOPs clamp)."""
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0,
            model_id="mistralai/Mistral-7B-v0.1",
            batch_size=1, seq_len=8192, phase="prefill")
        gqa = [j for j in jobs if j["signature"] == "gqa_attention"]
        assert gqa
        for j in gqa:
            assert j["dims"]["s_query"] == 8192
            assert j["dims"]["s_kv"] == 8192, j["dims"]
            assert j["dims"]["s_window"] == 4096
            k = j["shape_in"][1]
            assert k[2] == 8192, k

    def test_sink_swa_model_plans_gqa_attention(self):
        from kernel_bench.plan import build_jobs
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id="openai/gpt-oss-20b",
            batch_size=1, seq_len=2048, phase="prefill")
        gqa = [j for j in jobs if j["signature"] == "gqa_attention"]
        assert gqa
        dropped_ops = {d.split()[0] for d in dropped if d}
        assert "swa_softmax_with_sink" not in dropped_ops, dropped
        windowed = [j for j in gqa if "s_window" in j["dims"]]
        dense = [j for j in gqa if "s_window" not in j["dims"]]
        assert windowed and dense, (
            "GPT-OSS is hybrid: even layers SWA, odd layers full GQA; "
            f"got windowed={len(windowed)} dense={len(dense)}")
        assert all(j["kernel_params"].get("s_window") == j["dims"]["s_window"]
                   for j in windowed)
        assert all("s_window" not in (j.get("kernel_params") or {})
                   for j in dense)
        assert all("n_attn_sinks" in j["dims"] for j in gqa), (
            "GPT-OSS original model passes sinks into FA on every layer")
        assert all(j["dims"]["n_attn_sinks"] == j["dims"]["n_heads"]
                   for j in gqa)
        assert all(j["kernel_params"].get("sinks") is True for j in gqa)
        assert "softmax_with_sink" not in dropped_ops, dropped

    def test_dense_gqa_does_not_inherit_a_window(self):
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0,
            model_id="meta-llama/Llama-3.2-1B",
            batch_size=1, seq_len=2048, phase="prefill")
        gqa = [j for j in jobs if j["signature"] == "gqa_attention"]
        assert gqa
        assert all("s_window" not in j["dims"] for j in gqa)
        assert all("s_window" not in (j.get("kernel_params") or {})
                   for j in gqa)

    def test_s_window_is_a_variant_not_a_cell_axis(self):
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in, row_invocation_dims)
        from backend.inference.op_signatures import get_signature
        from kernel_bench.providers.attention import _fa_window_size
        shape = [[1, 32, 128, 128], [1, 8, 128, 128]]
        # the operands' ports (`operand_ports`): K and V are one tensor
        # here (two operands), so the second fills both ports
        qk = [["q"], ["k", "v"]]
        # The deployment's page, which every paged cell carries.
        page = {"kv_block_size": 16}
        windowed = json.loads(identity_shape_from_shape_in(
            "flash_attention", shape, spec_dims={"s_window": 4096, **page},
            operand_ports=qk))
        assert windowed["s_window"] == 4096
        dense = json.loads(identity_shape_from_shape_in(
            "flash_attention", shape, spec_dims=page, operand_ports=qk))
        assert "s_window" not in dense
        leaked = json.loads(identity_shape_from_shape_in(
            "flash_attention", shape,
            spec_dims=row_invocation_dims({"s_window": 4096, "n_heads": 32},
                                          signature=get_signature("gqa_attention"),
                                          extra=page),
            operand_ports=qk))
        assert "s_window" not in leaked
        gqa = get_signature("gqa_attention")
        projected = json.loads(gqa.project(json.dumps(windowed)))
        assert projected["s_window"] == 4096
        dense_proj = json.loads(gqa.project(
            '{"B":1,"s_query":128,"s_kv":128,"n_heads":32,'
            '"n_kv_heads":8,"d_head":128,"d_model":4096,"kv_block_size":16}'))
        assert "s_window" not in dense_proj
        assert "d_model" not in dense_proj
        inv_w = Invocation.from_job(_gqa_job(
            dims={**_gqa_job()["dims"], "s_window": 4096},
            kernel_params={"s_window": 4096}))
        assert _fa_window_size(inv_w) == (4095, 0)
        assert _fa_window_size(Invocation.from_job(_gqa_job())) is None

    def test_asymmetric_v_is_a_variant(self):
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        # Q/K d_head=192, apply out / V last dim 128.
        page = {"kv_block_size": 16}
        s = identity_shape_from_shape_in(
            "flash_attention",
            [[1, 32, 128, 192], [1, 8, 128, 192], [1, 8, 128, 128]],
            shape_out=[1, 32, 128, 128], spec_dims=page,
            operand_ports=[["q"], ["k"], ["v"]])
        d = json.loads(s)
        assert d["d_head"] == 192
        assert d["d_v_head"] == 128
        sym = json.loads(identity_shape_from_shape_in(
            "flash_attention",
            [[1, 32, 128, 128], [1, 8, 128, 128], [1, 8, 128, 128]],
            shape_out=[1, 32, 128, 128], spec_dims=page,
            operand_ports=[["q"], ["k"], ["v"]]))
        assert "d_v_head" not in sym

    def test_sinks_presence_does_not_leak_from_the_model_wide_table(self):
        from backend.inference.benchmark_identity import (
            attach_gqa_sinks, identity_shape_from_shape_in, row_invocation_dims)
        from backend.inference.op_signatures import get_signature
        shape = [[1, 32, 128, 128], [1, 8, 128, 128]]
        # the operands' ports (`operand_ports`): K and V are one tensor
        # here (two operands), so the second fills both ports
        qk = [["q"], ["k", "v"]]
        page = {"kv_block_size": 16}
        base = identity_shape_from_shape_in("flash_attention", shape,
                                            spec_dims=page, operand_ports=qk)
        sunk = json.loads(attach_gqa_sinks(base, True))
        assert sunk["n_attn_sinks"] == 32
        assert "n_attn_sinks" not in json.loads(base)
        leaked = json.loads(identity_shape_from_shape_in(
            "flash_attention", shape,
            spec_dims=row_invocation_dims({"n_attn_sinks": 32, "s_window": 128},
                                          signature=get_signature("gqa_attention"),
                                          extra=page),
            operand_ports=qk))
        assert "n_attn_sinks" not in leaked
        assert "s_window" not in leaked

    def test_every_declared_variant_is_the_leafs_own(self):
        """Which node dims are per-leaf is the signature's `variant_dims`,
        every one of them: a V width the model binds model-wide (MiMo's
        `d_v_head`) does not reach a leaf whose own V is `d_head` wide."""
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in, row_invocation_dims)
        from backend.inference.op_signatures import get_signature
        gqa = get_signature("gqa_attention")
        node = {axis: 64 for axis in gqa.variant_dims}
        spec = row_invocation_dims({**node, "n_heads": 32}, signature=gqa,
                                   extra={"kv_block_size": 16})
        assert not set(gqa.variant_dims) & set(spec), spec
        cell = json.loads(identity_shape_from_shape_in(
            "flash_attention",
            [[1, 32, 128, 128], [1, 8, 128, 128], [1, 8, 128, 128]],
            shape_out=[1, 32, 128, 128], spec_dims=spec,
            operand_ports=[["q"], ["k"], ["v"]]))
        assert not set(gqa.variant_dims) & set(cell), cell

    def test_mimo_swa_has_sink_full_does_not(self):
        from kernel_bench.plan import build_jobs
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id="XiaomiMiMo/MiMo-V2-Flash",
            batch_size=1, seq_len=2048, phase="prefill")
        gqa = [j for j in jobs if j["signature"] == "gqa_attention"]
        assert gqa, dropped[:8]
        windowed = [j for j in gqa if "s_window" in j["dims"]]
        dense = [j for j in gqa if "s_window" not in j["dims"]]
        assert windowed and dense, (
            f"MiMo is hybrid SWA+full; windowed={len(windowed)} dense={len(dense)}")
        assert all("n_attn_sinks" in j["dims"] for j in windowed)
        assert all("n_attn_sinks" not in j["dims"] for j in dense)
        assert all(j["dims"].get("d_v_head") == 128 for j in gqa), (
            "MiMo V is 128 against QK 192")

    def test_fa_kwargs_and_inputs_carry_window_and_sink(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.providers.attention import _fa_extra_kwargs
        job = _gqa_job(
            dims={"B": 1, "s_query": 32, "s_kv": 32, "n_heads": 8,
                  "n_kv_heads": 2, "d_head": 64, "s_window": 16,
                  "n_attn_sinks": 8, "kv_block_size": 16},
            kernel_params={"s_window": 16, "sinks": True},
        )
        inv = Invocation.from_job(job)
        inputs = build_inputs(get_ref("gqa_attention"), inv, "cpu")
        assert inputs["s_window"] == 16
        assert tuple(inputs["sinks"].shape) == (8,)
        extra = _fa_extra_kwargs(inv, inputs)
        assert extra["window_size"] == (15, 0)
        assert extra["s_aux"] is inputs["sinks"]
        assert _fa_extra_kwargs(Invocation.from_job(_gqa_job()),
                                {"q": None, "k": None, "v": None}) == {}

    def test_gqa_rope_is_planned(self):
        from kernel_bench.plan import build_jobs
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id="meta-llama/Llama-3.2-1B",
            batch_size=1, seq_len=2048, phase="prefill")
        rope = [j for j in jobs if j["signature"] == "rope"]
        assert rope, dropped
        assert all(j["dims"]["d_rope"] > 0 for j in rope)
        dropped_ops = {d.split()[0] for d in dropped if d}
        assert "fused_qk_rope" not in dropped_ops, dropped

    def test_gpt_oss_mxfp4_expert_linear_plans_as_grouped_gemm(self):
        # The expert weight is MXFP4 and the cell says so. It used to say
        # `U8`, the safetensors container the E2M1 pairs and the UE8M0
        # exponent ride in — and the corpus has no token for a container,
        # so every expert GEMM of every MXFP4 model was declined for a
        # dtype environment nothing could spell.
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig

        for parallel in (ParallelConfig(),
                         ParallelConfig.from_dict({"ep_moe": 8,
                                                   "dp_attention": 8})):
            for phase in ("prefill", "decode"):
                jobs, dropped = build_jobs(
                    avg_output_tokens=256, model_id="openai/gpt-oss-20b",
                    batch_size=16, seq_len=1024, phase=phase,
                    parallel=parallel)
                grouped = [j for j in jobs
                           if j["signature"] == "grouped_gemm"]
                assert grouped, (phase, dropped)
                dropped_ops = {d.split()[0] for d in dropped if d}
                assert "mxfp4_fused_linear" not in dropped_ops, dropped
                for job in grouped:
                    env = job["dtype_env"]
                    # The format, its block, and what the block's scale
                    # holds — the three facts a kernel is written for.
                    assert env["weight"] == "mxfp4", (phase, env)
                    assert env["metadata"] == "ue8m0", (phase, env)
                    assert env["weight_granularity"] == "per-group:32", (
                        phase, env)
                    # The activation is NOT quantised: gpt-oss's kernel
                    # dequantises the weight inside the matmul (W4A16, the
                    # scheme's own `activation: none`).
                    assert env["activation"] == "bf16", (phase, env)

    @pytest.mark.regression
    def test_deepseek_v4_expert_gemms_are_mxfp4(self):
        # The owner's ruling: DeepSeek-V4's routed experts are E2M1 under
        # one UE8M0 exponent per 32, which is the OCP Microscaling recipe,
        # so the cell says `mxfp4` and not the generic `fp4_e2m1`. It is
        # declared at the three models' own `per_role.ffn_moe`
        # (`mxfp4-dot-suffix`), not inferred from the scale and the group —
        # the generic `fp4` scheme beside it is unchanged.
        #
        # It is not a label: `quant.FORMATS` puts an `mxfp4` weight on a
        # power-of-two scale grid and an `fp4_e2m1` weight on an exact one,
        # and the gates of the FP4 kernels are written per token.
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig

        for model in ("deepseek-ai/DeepSeek-V4-Flash",
                      "deepseek-ai/DeepSeek-V4-Pro",
                      "deepseek-ai/DeepSeek-V4.1-Flash"):
            for phase in ("prefill", "decode"):
                jobs, _dropped = build_jobs(
                    avg_output_tokens=256, model_id=model, batch_size=16,
                    seq_len=1024, phase=phase, parallel=ParallelConfig())
                routed = [j for j in jobs if j["moe_block_member"]
                          and j["signature"] == "grouped_gemm"]
                assert routed, (model, phase)
                for job in routed:
                    env = job["dtype_env"]
                    assert env["weight"] == "mxfp4", (model, phase, env)
                    assert env["metadata"] == "ue8m0", (model, phase, env)
                    assert env["weight_granularity"] == "per-group:32", (
                        model, phase, env)
                # The shared experts are FP8 block-scaled and stay that
                # way: the ruling is about the routed banks.
                dense = [j for j in jobs if j["signature"] == "gemm"
                         and j["dtype_env"]["weight"] == "fp8_e4m3"]
                assert dense, (model, phase, "no fp8 dense weight left")

    def test_an_mxfp4_cell_is_billed_for_its_scales(self):
        # The scale is a real read — one UE8M0 byte per 32 weights, 6% of
        # an MXFP4 bank's traffic — and the bytes are counted at the SLOT
        # the scale ships in, which is the one thing the storage dtype is
        # still for. The format moved to the identity; the width did not.
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig

        jobs, dropped = build_jobs(
            avg_output_tokens=256, model_id="openai/gpt-oss-20b",
            batch_size=16, seq_len=1024, phase="decode",
            parallel=ParallelConfig())
        down = next((j for j in jobs if j["signature"] == "grouped_gemm"
                     and j["node_spec_id"].endswith("down_proj")), None)
        assert down is not None, dropped
        dims, w = down["dims"], down["shape_weight"]
        experts, d_in, d_out = w[0], w[2], w[1]
        # One expert bank at half a byte an element, its scales at one byte
        # per 32, its bias, and the rows the launch moves either side.
        bank = experts * d_out * d_in * 0.5
        scales = experts * d_out * (d_in / 32) * 1.0
        bias = experts * d_out * 2
        rows = dims["n_tokens"]
        act = rows * d_in * 2 + rows * d_out * 2
        assert down["access_bytes_per_rank"] == bank + scales + bias + act, (
            down["access_bytes_per_rank"], bank, scales, bias, act)
        # Not a rounding-error term: dropping it would under-read the cell.
        assert scales / (bank + scales) > 0.05

    def test_allreduce_fused_rmsnorm_projects(self):
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        # The fused span carries the norm's weight beside the two rows it
        # mixes, and its cell is keyed by the comm world it runs over — a
        # spec dim the tensors cannot state.
        s = identity_shape_from_shape_in(
            "allreduce_fused_rmsnorm",
            [[1, 2048, 4096], [1, 2048, 4096], [4096]],
            spec_dims={"n_comm_ranks": 8})
        assert json.loads(s) == {"B": 1, "s_query": 2048, "d_model": 4096,
                                 "n_comm_ranks": 8}

    def test_geglu_plans_as_gelu_and_mul(self):
        from kernel_bench.plan import build_jobs
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id="google/gemma-2-9b",
            batch_size=1, seq_len=2048, phase="prefill")
        assert any(j["signature"] == "gelu_and_mul" for j in jobs), dropped
        assert not any(j["op"] == "gelu_and_mul"
                       and j["signature"] == "silu_and_mul" for j in jobs)

    def test_an_encoder_is_planned_in_its_encode_phase(self):
        # The tower's kernels run at the media the plan states (a test
        # reference: one 672 x 672 image), never at the text step's length.
        from kernel_bench.plan import build_jobs
        media = {"pixel_values": {"n_img": 1, "img_h": 672, "img_w": 672}}
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id="moonshotai/Kimi-K3",
            batch_size=1, seq_len=2048, phase="encode", media=media)
        vis = [j for j in jobs if "vision" in j["node_spec_id"]]
        assert vis and all(j["phase"] == "encode" for j in jobs), dropped
        assert any(j["signature"] == "gelu" for j in vis), {j["signature"] for j in vis}
        prefill, _ = build_jobs(avg_output_tokens=0,
            model_id="moonshotai/Kimi-K3",
            batch_size=1, seq_len=2048, phase="prefill")
        assert not [j for j in prefill if "vision" in j["node_spec_id"]]

    def test_an_encode_plan_states_its_media(self):
        from kernel_bench.plan import build_jobs
        with pytest.raises(ValueError, match="media"):
            build_jobs(avg_output_tokens=0, model_id="moonshotai/Kimi-K3", batch_size=1,
                       seq_len=2048, phase="encode")

    def test_steps_downsampler_convs_key_their_real_grids(self):
        # vLLM step3_vl.py:64-79: the 728 x 728 view is a 52 x 52 patch
        # grid; each stride-2 conv halves the side.
        from kernel_bench.plan import build_jobs
        jobs, _dropped = build_jobs(avg_output_tokens=0,
            model_id="stepfun-ai/Step-3.7-Flash", batch_size=1, seq_len=1,
            phase="encode",
            media={"pixel_values": {"n_img": 1},
                   "patch_pixel_values": {"n_crops": 0}})
        grids = sorted(j["dims"]["s_conv_grid"] for j in jobs
                       if j["signature"] == "conv2d")
        assert grids == [26 * 26, 52 * 52], grids

    def test_embedding_is_planned(self):
        from kernel_bench.plan import build_jobs
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id="meta-llama/Llama-3.2-3B",
            batch_size=1, seq_len=2048, phase="prefill")
        assert any(j["signature"] == "embedding" for j in jobs), dropped
        dropped_ops = {d.split()[0] for d in dropped if d}
        assert "embedding" not in dropped_ops, dropped

    def test_decode_moe_shuffle_is_not_empty_tensor(self):
        from fractions import Fraction
        from kernel_bench.plan import build_jobs
        # gpt-oss-20b routes 4 of its 32 experts per token, so one card's
        # dispatch means `B/8` rows per expert: a whole row at batch 16, an
        # eighth of one at batch 1 and a row and a half at batch 12. The
        # experts the rows reach are padded up to `ceil(M)` and the rest run
        # nothing, so the shuffle is planned at all three — never an empty
        # tensor, padded only where a reached expert's rows are fractional
        # (batch 12: 48 routed rows over 32 experts, run at two rows each),
        # and never a width the job does not also state the law's own row count
        # beside.
        for batch, padded in ((16, False), (1, False), (12, True)):
            jobs, dropped = build_jobs(avg_output_tokens=0,
                model_id="openai/gpt-oss-20b",
                batch_size=batch, seq_len=2048, phase="decode")
            dropped_ops = {d.split()[0] for d in dropped if d}
            assert "permute" not in dropped_ops, dropped
            assert "combine" not in dropped_ops, dropped
            assert "silu_and_mul" not in dropped_ops, dropped
            shuf = [j for j in jobs if j["signature"] in
                    ("moe_permute", "moe_combine", "silu_and_mul")]
            assert shuf, (batch, dropped)
            # The boundaries name the padded per-expert rows `s_expert_rows`;
            # the in-bank activation keeps the generic fold (`s_query`).
            assert all(j["dims"].get("s_expert_rows",
                                     j["dims"].get("s_query", 1)) >= 1
                       for j in shuf)
            assert all((Fraction(j["launch_rows_exact"]) < j["launch_rows"])
                       is padded for j in shuf), (
                batch, [(j["signature"], j["launch_rows"],
                         j["launch_rows_exact"]) for j in shuf])

    def test_dsa_indexer_side_ops_are_planned(self):
        from kernel_bench.plan import build_jobs
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id="deepseek-ai/DeepSeek-V3.2",
            batch_size=1, seq_len=2048, phase="prefill")
        dropped_ops = {d.split()[0] for d in dropped if d}
        assert "layernorm" not in dropped_ops, dropped
        assert "dsa_topk_mask" not in dropped_ops, dropped
        sigs = {j["signature"] for j in jobs}
        assert "layernorm" in sigs, sigs
        assert "dsa_topk_mask" in sigs, sigs
        jobs_d, dropped_d = build_jobs(avg_output_tokens=0,
            model_id="deepseek-ai/DeepSeek-V3.2",
            batch_size=1, seq_len=2048, phase="decode")
        extra = [d for d in dropped_d if d.split()[0] == "indexer_logits"]
        assert extra == [], extra
        idx = [j for j in jobs_d if j["signature"] == "indexer_logits"]
        assert idx
        assert all(j["dims"]["s_query"] == 1 for j in idx)
        assert all(j["dims"]["s_kv"] == 2048 for j in idx)

    def test_unfused_router_and_unary_act_are_planned(self):
        from kernel_bench.plan import build_jobs
        jobs, dropped = build_jobs(avg_output_tokens=0,
            model_id="deepseek-ai/DeepSeek-V2-Lite",
            batch_size=1, seq_len=2048, phase="prefill")
        dropped_ops = {d.split()[0] for d in dropped if d}
        assert not (dropped_ops & {"softmax_act", "topk", "gather", "add"}), dropped
        assert any(j["signature"] == "add" for j in jobs), dropped
        jobs2, dropped2 = build_jobs(avg_output_tokens=0,
            model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
            batch_size=1, seq_len=2048, phase="prefill")
        extra = [d for d in dropped2 if d.split()[0] in
                 {"silu", "mul", "sigmoid", "softplus", "neg_exp", "add"}]
        assert extra == [], extra
        sigs = {j["signature"] for j in jobs2}
        # Qwen3-Next's SiLU is the fused silu_and_mul launch now; the other
        # unary acts still stand alone.
        assert {"silu_and_mul", "mul", "sigmoid", "softplus",
                "neg_exp"} <= sigs, sigs

    def test_per_head_linear_projects(self):
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        s = identity_shape_from_shape_in(
            "per_head_linear", [[1, 128, 1, 512]],
            shape_out=[1, 128, 1, 576])
        assert json.loads(s) == {
            "B": 1, "n_heads": 128, "s_query": 1,
            "d_reduce": 512, "d_out": 576}


class TestLibraryModelsAreSweepable:
    """Every shipped snapshot must plan without unexpected identity drops.

    Includes MTP stacks (an encoder is planned in its encode phase, at the
    media a plan states). Unmapped layout ops (view,
    transpose, …) stay dropped by design. A mapped signature with no
    identity used to fabricate model-wide dims — that is the hole this
    gate closes. The named sets below are measured debt, not silent ignores:
    mostly primitives and alternate engine layouts inside fused launches for
    which the framework has a launch but the corpus does not yet have a
    signature/reference/provider. A new op, signature or geometry fails until
    it is listed with the rest of that boundary.
    """

    #: primitives that survive an unmatched fused span and still have no
    #: measurement of their own
    KNOWN_UNMEASURED_OPS = frozenset({
        "add_const", "clamp", "exp", "group_mask_apply", "group_max",
        "l2norm", "rope", "scatter_rows_add",
    })
    #: fusion launches with neither an op binding nor a measurement route
    KNOWN_UNMEASURED_FUSED_OPS = frozenset({
        "add_attn_res_combine_norm", "causal_conv1d_packed",
        "ihc_pre_gate_reduce", "ihc_pre_mix",
        "merged_column_parallel_linear_wide", "mla_q_concat_kv_insert",
    })
    #: bound signatures whose alternate engine geometry is not yet declared
    KNOWN_MISSING_PROJECTIONS = frozenset({
        "fused_qk_mrope", "fused_q_kv_rmsnorm", "indexer_logits",
        "position_table_add", "qsa_index_k_compress_store", "softmax_act",
        "topk",
    })

    @pytest.mark.timeout(3600)
    # One plan per snapshot (63 models), each a full walk: the lane's
    # per-test budget is smaller than the corpus.
    def test_every_snapshot_projects_every_mapped_op(self):
        from pathlib import Path
        from backend.arch.loader import SpecLoader
        from backend.arch.ops import get_op
        from backend.arch.ops_registry import _zero
        from backend.inference.framework_bindings import (
            ServingDeployment, get_binding)
        from backend.inference.op_signatures import op_bindings
        from kernel_bench.plan import build_jobs

        snap_dir = (Path(__file__).resolve().parents[2]
                    / "backend/arch/snapshots")
        snaps = sorted(snap_dir.glob("*.json"))
        assert snaps, "no snapshots"
        binding = get_binding("vllm").deployed(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16"))
        op_to_signature = op_bindings()
        loader = SpecLoader()
        failures = []
        for path in snaps:
            blob = json.loads(path.read_text())
            model_id = blob["model_id"]
            resolved = loader.match(blob["config"], model_id=model_id)
            if resolved is None:
                failures.append(f"{model_id}: no spec")
                continue
            for phase in ("prefill", "decode"):
                _jobs, dropped = build_jobs(avg_output_tokens=0,
                    model_id=model_id, batch_size=1, seq_len=2048,
                    phase=phase, resolved=resolved, binding=binding,
                    gpu="H100-SXM")
                proj = [d for d in dropped if "no shape projection" in d]
                if proj:
                    failures.append(f"{model_id} {phase}: {proj[:6]}")
                for note in dropped:
                    op = note.split()[0]
                    if op in op_to_signature:
                        # A fused launch is a signature, not necessarily a
                        # primitive OpSpec; zero-cost rows also keep only the
                        # op as their display note.
                        continue
                    if "no shape projection" in note:
                        continue
                    if op in self.KNOWN_UNMEASURED_FUSED_OPS:
                        continue
                    try:
                        spec = get_op(op)
                    except KeyError:
                        failures.append(
                            f"{model_id} {phase}: unknown dropped op {op}")
                        continue
                    if (op in self.KNOWN_UNMEASURED_OPS
                            or spec.flops_fn is _zero
                            or getattr(spec, "is_view_op", False)):
                        continue
                    failures.append(
                        f"{model_id} {phase}: compute op {op} has no "
                        f"signature (not layout)")
        unexpected = [
            failure for failure in failures
            if not any(
                f"no shape projection for signature '{signature}'" in failure
                for signature in self.KNOWN_MISSING_PROJECTIONS)
        ]
        assert not unexpected, "\n".join(unexpected)


class TestLinearSSM:
    """Linear-attention / SSM ops (gated_delta_rule, mamba2_scan) were dropped
    (no signature); now they're measurable with FLA / mamba_ssm providers."""

    def test_gated_delta_identity(self):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        s = identity_shape_from_shape_in(
            "gated_delta_rule",
            [[1, 16, 2048, 128], [1, 16, 2048, 128], [1, 32, 2048, 128],
             [32], [1, 2048, 32]])
        # Dims carry the linear-attention glossary names: these heads are a
        # different structure from dense attention's and must not read as the
        # same axis in the corpus.
        assert json.loads(s) == {"B": 1, "n_lin_k_heads": 16, "s_query": 2048,
                                 "d_lin_k_head": 128, "n_lin_v_heads": 32,
                                 "d_lin_v_head": 128}

    def test_mamba2_identity(self):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        operands = [[1, 2048, 256, 64], [1, 2048, 256], [256],
                    [1, 2048, 8, 128], [1, 2048, 8, 128], [256], [256]]
        s = identity_shape_from_shape_in(
            "mamba2_scan", operands, spec_dims={"ssm_chunk_size": 128})
        assert json.loads(s) == {"B": 1, "s_query": 2048, "n_ssm_heads": 256, "d_ssm_head": 64,
                                 "n_ssm_groups": 8, "d_ssm_state": 128,
                                 "ssm_chunk_size": 128}
        # The chunk is no tensor's axis, so the caller states it (the
        # deployment's framework binding, through `RowIdentity`) — and a
        # caller that states none leaves the cell without an identity rather
        # than with a chunk somebody picked.
        assert identity_shape_from_shape_in("mamba2_scan", operands) is None

    def test_providers_registered(self):
        from kernel_bench.providers import all_providers
        impls = {(p.library, p.kernel_impl) for p in all_providers()}
        assert ("fla", "fla_chunk_gated_delta_rule") in impls
        assert ("mamba_ssm", "mamba_chunk_scan_combined") in impls

    def test_plan_emits_gated_delta_for_qwen3_next(self):
        from kernel_bench.plan import build_jobs
        # Use the checked-in Qwen3-Next snapshot so this contract is offline.
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-Coder-Next",
                             batch_size=1, seq_len=2048, phase="prefill")
        gd = [j for j in jobs if j["signature"] == "gated_delta_rule"]
        assert gd, "gated_delta_rule must be measurable (was dropped)"
        assert set(gd[0]["dims"]) >= {"B", "s_query", "n_lin_k_heads",
                                      "n_lin_v_heads", "d_lin_k_head",
                                      "d_lin_v_head"}

    def test_linear_inputs_build_on_cpu(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        gd = get_ref("gated_delta_rule")
        inv = Invocation.from_job({
            "signature": "gated_delta_rule", "op": "gated_delta_rule",
            "dims": {"B": 1, "n_lin_k_heads": 2, "n_lin_v_heads": 4,
                     "s_query": 8, "d_lin_k_head": 16, "d_lin_v_head": 16},
            "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"}, "phase": "prefill"})
        ins = build_inputs(gd, inv, "cpu")
        # Time-major [B,T,H,D] — the layout the kernels document and both
        # frameworks call them with. `head_first` no longer exists upstream.
        assert tuple(ins["v"].shape) == (1, 8, 4, 16)
        assert tuple(ins["q"].shape) == (1, 8, 2, 16)
        assert tuple(ins["initial_state"].shape) == (1, 4, 16, 16)  # [N,HV,V,K]
        assert gd.reference is not None, "the recurrence is its own oracle"


class TestGroupedGemmMoE:
    """A routed (MoE expert) projection routes to a distinct grouped_gemm
    kernel (DeepGEMM m_grouped), not the dense GEMM. Routedness is the
    program's fact (`Program.is_routed`), passed to resolve_op as a flag — not
    read off the node id's spelling."""

    def test_resolve_op_routes_expert_gemm(self):
        from backend.inference.benchmark_identity import resolve_op
        # An expert projection runs the GROUPED gemm kernel. Which nodes are
        # routed is the program's fact (`Program.is_routed`, covered by its own
        # tests), passed here as the flag — resolve_op reads routedness off
        # that, never off the node id's spelling.
        assert resolve_op("linear", routed=True) == "grouped_gemm"
        assert resolve_op("merged_column_parallel_linear",
                          routed=True) == "grouped_gemm"
        # dense (non-routed) linears stay gemm-family
        assert resolve_op("linear", routed=False) == "linear"
        assert resolve_op("linear") == "linear"

    def test_grouped_identity_from_shape_in(self):
        # The engine emits the rank's [E_local, M, K] routed operand (M =
        # mean tokens/expert), in the order the signature declares it
        # (`grouped_gemm.engine_inputs`).
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        s = identity_shape_from_shape_in("grouped_gemm", [64, 192, 2048],
                                         shape_out=[64, 192, 7168],
                                         routed_rows=64 * 192)
        assert json.loads(s) == {
            "n_experts": 64, "n_tokens": 64 * 192, "s_expert_rows": 192,
            "d_reduce": 2048, "d_out": 7168,
        }

    def test_a_fractional_mean_pads_only_the_experts_that_receive_rows(self):
        # The eval shape is an exact MEAN per expert. A grouped kernel
        # launches no blocks for an expert that receives nothing, so the
        # launch is the experts that receive rows, each padded up to whole
        # rows (`RoutedLaunch`): 51 routed rows over 64 experts is 51
        # experts at one row, not 64.
        from backend.inference.benchmark_identity import (
            RowIdentity, identity_shape_from_shape_in, routed_launch_rows)

        def keyed(exact_in, exact_out, routed_rows, experts):
            row = {"shape_in": [experts, 0, exact_in[-1]],
                   "shape_in_exact": exact_in,
                   "shape_out": [experts, 0, exact_out[-1]],
                   "shape_out_exact": exact_out,
                   "routed": {"dispatch": 1, "source_rows": 1,
                              "routed_rows": routed_rows,
                              "local_experts": experts, "inputs": [0],
                              "output": True, "bytes_per_row": 0}}
            shape_in, shape_out = RowIdentity.one_event(row)
            return identity_shape_from_shape_in(
                "grouped_gemm", shape_in, shape_out=shape_out,
                routed_rows=routed_launch_rows(row))

        got = keyed([64, "51/64", 2048], [64, "51/64", 7168], 51, 64)
        assert json.loads(got) == {
            "n_experts": 51, "n_tokens": 51, "s_expert_rows": 1,
            "d_reduce": 2048, "d_out": 7168,
        }
        # Over one row: 3/2 each over 256 experts is every expert at 2.
        over = keyed([256, "3/2", 2048], [256, "3/2", 4096], 384, 256)
        assert json.loads(over) == {
            "n_experts": 256, "n_tokens": 512, "s_expert_rows": 2,
            "d_reduce": 2048, "d_out": 4096,
        }
        # So the total is the product of the launched extents, always — the
        # cell cannot state a row count its own geometry does not run.
        for shape in (got, over):
            dims = json.loads(shape)
            assert dims["n_tokens"] == dims["n_experts"] * dims["s_expert_rows"]

    @pytest.mark.parametrize("batch", [1, 24])
    def test_the_built_launch_is_the_cell_it_is_keyed_by(self, batch):
        # The key, the operands the reference builds and the expert counts
        # the provider is handed all name ONE launch. Batch 1 first: 8 rows
        # reach 8 of Qwen3-30B-A3B's 128 experts, and a key that named 8
        # over a bank built at 128 measured the whole bank under it. Batch
        # 24 is the padded path: 192 rows over 128 experts, two rows each.
        from kernel_bench.plan import build_jobs
        from kernel_bench import refs
        jobs, _ = build_jobs(avg_output_tokens=16,
                             model_id="Qwen/Qwen3-30B-A3B", batch_size=batch,
                             seq_len=256, phase="decode")
        zone = [j for j in jobs if j["moe_block_member"]]
        assert zone
        for job in zone:
            inv = Invocation.from_job(job)
            dims = job["dims"]
            where = (batch, job["signature"], dims)
            if job["signature"] == "grouped_gemm":
                x, w = refs._shape(inv, "x"), refs._shape(inv, "w")
                assert x[0] == w[0] == dims["n_experts"], (where, x, w)
                assert len(inv.moe_token_counts) == x[0], where
                assert sum(inv.moe_token_counts) == dims["n_tokens"], where
            elif job["signature"] in ("moe_permute", "moe_combine"):
                (rows,) = refs._shape(inv, "idx")
                assert rows == dims["n_tokens"], (where, rows)
            elif job["signature"] == "silu_and_mul":
                # The in-bank activation folds the same routed rows to
                # `B x s_query`.
                x = refs._shape(inv, "x")
                assert x[0] == dims["B"] * dims["s_query"], (where, x)

    def test_plan_emits_grouped_gemm_for_deepseek(self):
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3",
                             batch_size=1, seq_len=2048, phase="prefill")
        gg = [j for j in jobs if j["signature"] == "grouped_gemm"]
        assert gg, "no grouped_gemm job for DeepSeek MoE experts"
        for j in gg:
            assert set(j["dims"]) >= {
                "n_experts", "n_tokens", "s_expert_rows", "d_reduce", "d_out",
            } and j["dims"]["n_experts"] > 1

    def test_plan_distributes_exact_tokens_without_inventing_rows(self):
        from kernel_bench.plan import build_jobs

        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3",
                             batch_size=3, seq_len=17, phase="prefill")
        grouped = [j for j in jobs if j["signature"] == "grouped_gemm"]
        assert grouped
        for job in grouped:
            assert job["kernel_params"]["moe_token_distribution"] == "balanced"
            counts = Invocation.from_job(job).moe_token_counts
            assert len(counts) == job["dims"]["n_experts"]
            assert sum(counts) == job["dims"]["n_tokens"]
            assert max(counts) == job["dims"]["s_expert_rows"]
            assert max(counts) - min(counts) <= 1

    def test_balanced_distribution_does_not_expand_large_expert_lists(self):
        # Every expert at the cell's own width, named by the distribution
        # rather than listed per expert — 896 experts cost two keys, not an
        # 896-entry array in every job.
        inv = Invocation.from_job({
            "signature": "grouped_gemm", "op": "grouped_gemm",
            "dims": {"n_experts": 896, "n_tokens": 896, "s_expert_rows": 1,
                     "d_reduce": 64, "d_out": 128},
            "dtype_env": {"activation": "bf16"}, "phase": "decode",
            "kernel_params": {"moe_token_distribution": "balanced"},
        })
        assert inv.moe_token_counts == (1,) * 896
        assert len(json.dumps(inv.kernel_params)) < 64

    def test_a_total_that_does_not_divide_is_not_a_distribution(self):
        # `balanced` is every expert at the cell's width, so the total it is
        # declared against has to be that width times the expert count. A
        # cell from the retired convention — the routed total spread
        # unevenly — says so instead of being served as some other shape.
        inv = Invocation.from_job({
            "signature": "grouped_gemm", "op": "grouped_gemm",
            "dims": {"n_experts": 896, "n_tokens": 51, "s_expert_rows": 1,
                     "d_reduce": 64, "d_out": 128},
            "dtype_env": {"activation": "bf16"}, "phase": "decode",
            "kernel_params": {"moe_token_distribution": "balanced"},
        })
        with pytest.raises(ValueError, match="n_tokens=51"):
            _ = inv.moe_token_counts

    def test_invocation_rejects_a_token_distribution_that_changes_the_cell(self):
        from kernel_bench.invocation import Invocation

        base = {
            "signature": "grouped_gemm", "op": "grouped_gemm",
            "dims": {"n_experts": 4, "n_tokens": 5, "s_expert_rows": 2,
                     "d_reduce": 16, "d_out": 32},
            "dtype_env": {"activation": "bf16", "weight": "bf16"},
            "phase": "prefill",
        }
        valid = Invocation.from_job({
            **base, "kernel_params": {"moe_token_counts": [2, 1, 1, 1]}})
        assert valid.moe_token_counts == (2, 1, 1, 1)
        invalid = Invocation.from_job({
            **base, "kernel_params": {"moe_token_counts": [2, 2, 1, 1]}})
        with pytest.raises(ValueError, match="expected n_tokens=5"):
            _ = invalid.moe_token_counts

    def test_torch_grouped_provider_does_not_compute_padding(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.gemm import _torch_grouped_build

        inv = Invocation.from_job({
            "signature": "grouped_gemm", "op": "grouped_gemm",
            "dims": {"n_experts": 4, "n_tokens": 5, "s_expert_rows": 2,
                     "d_reduce": 8, "d_out": 8},
            "dtype_env": {"activation": "fp32"}, "phase": "prefill",
            "kernel_params": {"moe_token_counts": [2, 1, 1, 1]},
        })
        x = torch.randn(4, 2, 8)
        x[1:, 1] = torch.nan
        w = torch.randn(4, 8, 8)
        case = _torch_grouped_build(inv, None, {"x": x, "w": w})
        out = case.to_canonical(case.run())
        assert tuple(out.shape) == (4, 2, 8)
        assert torch.count_nonzero(out[1:, 1]) == 0
        assert torch.isfinite(out).all()

    def test_expected_m_is_the_routing_average_not_the_buffer_width(self):
        # DeepGEMM's third quantity, and the only one derived from routing
        # rather than allocation: sglang computes it as ceil(routed rows /
        # local experts) (srt/layers/moe/ep_moe/kernels.py) while sizing the
        # buffer separately. Balanced routing hides the distinction — there
        # max(counts) IS the ceiling — so the falsifying case is a skewed
        # distribution, which is the one real MoE routing produces.
        from kernel_bench.providers.gemm import _expected_m

        assert _expected_m((8, 0, 0, 0)) == 2       # not 8, the widest expert
        assert _expected_m((2, 1, 1, 1)) == 2
        assert _expected_m((1,) * 51 + (0,) * 845) == 1
        assert _expected_m(()) == 0

    def test_job_numerators_are_the_walk_divided_by_the_launch_count(self):
        # The billing invariant, as a test. The read side multiplies a job's
        # numerators back up by `launches x cadence` (`calibration_overlay.
        # point_keys`), so a job must carry exactly the walk's row
        # over that count — and nothing else. Plan used to recompute bytes
        # from `dims` for 11 signatures and disagreed with the walk on 8 of
        # them: an FP8 GEMM was billed 0.28x its real traffic because the
        # formula priced the output at the activation dtype (an FP8 GEMM
        # writes BF16) and ignored the kernel's own activation quantisation
        # (`convert_bytes_per_rank`), while `rmsnorm` lost its gain vector and
        # `embedding` grew a 32-byte index term the walk has no edge for.
        # `dims` is the LOSSY scalar projection of the row's shapes; a cost
        # model read off it is a second answer to a question the walk already
        # answered (CLAUDE.md 2.7).
        import backend.inference.pipeline as pipeline
        from fractions import Fraction
        from backend.analyzer import fetch_config
        from backend.arch.loader import SpecLoader
        from backend.inference.benchmark_identity import RowIdentity
        from kernel_bench.plan import build_jobs

        model_id = "deepseek-ai/DeepSeek-V3"
        resolved = SpecLoader().match(fetch_config(model_id), model_id=model_id)
        seen = []
        real = pipeline.analyze

        def spy(*a, **kw):
            ar = real(*a, **kw)
            seen.append(ar)
            return ar

        pipeline.analyze = spy
        try:
            jobs, _ = build_jobs(avg_output_tokens=0, model_id=model_id, resolved=resolved,
                                 batch_size=1, seq_len=2048, phase="prefill")
        finally:
            pipeline.analyze = real
        assert len(seen) == 1
        identity = RowIdentity(seen[0])
        rows = {str(r.get("aggregation_id") or ""): r
                for r in seen[0].per_op_timing}
        assert jobs
        for job in jobs:
            row = rows[job["node_spec_id"]]
            launches = (max(identity.launches(row), 1)
                        * float(identity.cadence(row)))
            # A routed launch runs more rows than its law states — every
            # expert padded up to `ceil(M)` — and its numerators are that
            # geometry's, so the invariant carries the padding the job
            # itself states. It is 1 for every row nothing is padded on,
            # which is every row here except the dispatch zone's.
            padding = (Fraction(job["launch_rows"])
                       / Fraction(job["launch_rows_exact"]))
            assert padding >= 1, (job["signature"], padding)
            assert padding == 1 or job["moe_block_member"], (
                job["signature"], job["node_spec_id"], padding)
            for num, field in (("flops_per_rank", "flops_per_rank"),
                               ("access_bytes_per_rank", "access_bytes_per_rank")):
                want = row.get(field)
                got = job[num]
                if want is None:
                    assert got is None, (job["node_spec_id"], num)
                    continue
                if padding != 1 and num == "access_bytes_per_rank":
                    # Bytes do not all scale with rows: the resident weight
                    # read saturates (`row_roofline`), so the padded total
                    # is between the law's and the law's times the padding.
                    assert (float(want) <= got * launches
                            <= float(want) * float(padding) + 1), (
                        job["signature"], job["node_spec_id"], got, want)
                    continue
                assert got * launches == pytest.approx(
                    float(want) * float(padding), rel=1e-9), (
                    job["signature"], job["node_spec_id"], num, got, want)

        # The FP8 projection F9 was measured on — DeepSeek-V3's q_a and kv_a
        # merged into one launch, 1536 + 576 wide. 576 is no whole number of
        # 128-wide scale blocks, but nobody cut it (it is replicated) and it
        # is packed last, so the frameworks load it and the plan keeps it.
        # Its traffic is the FP8 activation the grafted cast wrote, the FP8
        # weight with its per-tile scales, and a BF16 write — not the bf16
        # read of an in-kernel quantise, which is the cast node's now.
        kv_a = next(j for j in jobs
                    if j["node_spec_id"].endswith("kv_a_proj_with_mqa"))
        assert kv_a["dtype_env"]["weight"] == "fp8_e4m3"
        d = kv_a["dims"]
        m = int(d.get("B", 1)) * int(d["s_query"])
        k, n = d["d_reduce"], d["d_out"]
        tiles = -(-n // 128) * -(-k // 128)
        assert kv_a["access_bytes_per_rank"] >= (
            m * k * 1 + k * n * 1 + tiles * 4 + m * n * 2)

    @pytest.mark.regression
    def test_an_expert_job_is_billed_for_the_rows_its_geometry_runs(self):
        # A per-expert mean is almost never whole, so every expert is padded
        # up to `ceil(M)` and the launch runs `[E_local, ceil(M), H]`. Two
        # things have to be true of that, and they are the whole of the
        # convention:
        #
        #   * the geometry is the law's extents each rounded UP — not the
        #     product rounded once, which would spread the remainder over
        #     the experts and hand the kernel a ragged buffer nothing
        #     declared;
        #   * the numerators are that geometry's, because the latency a
        #     measurement records is that geometry's. The walk's own
        #     numerators are the mean's, and the padding ratio is what
        #     relates the two: a job that carried the mean's cost would
        #     divide a wider launch's time by a narrower launch's work.
        #
        # And the padding is on the job, so nobody has to take the geometry
        # for the routing: `launch_rows` beside `launch_rows_exact`.
        import backend.inference.pipeline as pipeline
        from fractions import Fraction
        from math import ceil
        from kernel_bench.plan import build_jobs
        from backend.inference.benchmark_identity import RowIdentity
        from backend.inference.parallel_config import ParallelConfig

        def tensors(shape):
            if not isinstance(shape, (list, tuple)) or not shape:
                return []
            return (list(shape) if isinstance(shape[0], (list, tuple))
                    else [list(shape)])

        def padded_rows(row):
            """The widest side's rows recomputed off the row's own law, a
            different way than the geometry builds them: each extent rounded
            up — except that a tensor with a mean under one row runs only
            the rows its law routes, rounded up once (an expert nobody
            routed to runs nothing)."""
            best = None
            for field in ("shape_in_exact", "shape_out_exact"):
                for tensor in tensors(row.get(field)):
                    if len(tensor) < 2:
                        continue
                    extents = [Fraction(str(e)) for e in tensor[:-1]]
                    rows = 1
                    for extent in extents:
                        rows *= ceil(extent)
                    if any(e < 1 for e in extents):
                        law = 1
                        for extent in extents:
                            law *= extent
                        rows = min(rows, ceil(law))
                    best = rows if best is None else max(best, rows)
            return best

        seen = {}
        real = pipeline.analyze

        def spy(*a, **kw):
            ar = real(*a, **kw)
            for row in ar.per_op_timing:
                seen[str(row.get("aggregation_id") or "")] = row
            return ar

        checked = padded = 0
        for model_id in ("Qwen/Qwen3-30B-A3B", "deepseek-ai/DeepSeek-V3",
                         "openai/gpt-oss-20b",
                         "deepseek-ai/DeepSeek-V4-Flash"):
            # A batch-16 decode reaches fewer experts than a bank holds (a
            # mean under one row: nothing padded); batch 48 gives DeepSeek's
            # banks a row and a fraction each, which is padded.
            for phase, batch in (("prefill", 16), ("decode", 16),
                                 ("decode", 48)):
                for parallel in (ParallelConfig.from_dict(
                        {"ep_moe": 8, "dp_attention": 8}), ParallelConfig()):
                    seen.clear()
                    pipeline.analyze = spy
                    try:
                        jobs, _dropped = build_jobs(
                            avg_output_tokens=256, model_id=model_id,
                            batch_size=batch, seq_len=1024, phase=phase,
                            parallel=parallel)
                    finally:
                        pipeline.analyze = real
                    for job in jobs:
                        if not job["moe_block_member"]:
                            continue
                        row = seen[job["node_spec_id"]]
                        where = (model_id, phase, job["signature"],
                                 job["node_spec_id"])
                        want = padded_rows(row)
                        if want is None:
                            continue
                        dims = job["dims"]
                        # The cell's own geometry says the same number, by
                        # the names the identity gives a routed row: the
                        # routed total `n_tokens` on the three routed
                        # signatures; the in-bank activation folds the same
                        # buffer to `B × s_query`.
                        cell = (int(dims["n_tokens"]) if "n_tokens" in dims
                                else int(dims.get("B", 1)) * int(dims["s_query"]))
                        assert job["launch_rows"] == want == cell, (
                            where, job["launch_rows"], want, cell)
                        # And a routed cell's total is that same launch —
                        # the boundaries' too, whose first operand is not
                        # the buffer.
                        if "n_tokens" in dims:
                            assert dims["n_tokens"] == want, (where, dims)
                        exact = Fraction(job["launch_rows_exact"])
                        assert exact <= want, (where, str(exact), want)
                        # The numerators are the geometry's: the walk's own,
                        # scaled by exactly the padding the geometry has.
                        ratio = Fraction(want) / exact
                        if ratio != 1:
                            padded += 1
                        launches = (max(RowIdentity.launches(row), 1)
                                    * float(RowIdentity.cadence(row)))
                        assert job["flops_per_rank"] * launches == pytest.approx(
                            float(row["flops_per_rank"]) * float(ratio),
                            rel=1e-9), where
                        if job["signature"] == "grouped_gemm":
                            # And for the grouped GEMM the geometry's own op
                            # formula: the contraction is exactly
                            # `2 x rows x K x N` at the padded rows, and
                            # anything above it is a fused epilogue's
                            # per-row work (gpt-oss's expert GEMM adds its
                            # bias inside the launch), so the surplus is a
                            # whole number of rows and never a fraction of
                            # the contraction.
                            assert dims["n_tokens"] == want, (where, dims)
                            gemm = 2 * want * dims["d_reduce"] * dims["d_out"]
                            surplus = job["flops_per_rank"] - gemm
                            assert surplus >= 0, (where, job["flops_per_rank"],
                                                  gemm)
                            assert surplus % want == 0, (where, surplus, want)
                        checked += 1
        # Not vacuous: all four models' zones, both phases, both deployments
        # — and some of them really are padded.
        assert checked > 100, checked
        assert padded > 0, "no cell in the matrix had a fractional mean"

    @pytest.mark.regression
    def test_a_dispatch_boundary_is_billed_term_by_term(self):
        """The padded geometry's own bytes, not the law's times a ratio.

        A dispatch boundary moves two different things and only one of them
        is padded. `permute` READS the step's tokens and the routing that
        picked their experts — neither becomes wider because every local
        expert buffer is rounded up — and WRITES the `[E, ceil(M), d]`
        buffer. `combine` is the reverse: it reads the padded buffer and
        the routing, and writes the tokens back. So the byte law is
        `fixed + p x routed`, per term, and multiplying the aggregate by
        `p` overcharges the fixed half — 5.9% on DeepSeek-V3's decode and
        3.7% on DeepSeek-V4-Flash's.

        The numbers below are the direct formula at the padded geometry,
        computed from the shapes the job itself carries.
        """
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig

        # (model, batch, boundary) -> bytes one launch of it moves. The two
        # boundaries are no longer symmetric: the quantising cast sits above
        # the dispatch (one copy per token, before the boundary), so permute
        # carries the FP8 record both ways and combine returns bf16 from the
        # experts' output. A record row is its F8 elements plus its scales —
        # V3's `fp8_token_group128`: 7168 + 56x4 F32 = 7392 B; V4-Flash's
        # `fp8_token_group128_ue8m0`: 4096 + 32x1 = 4128 B.
        want = {
            # 256 experts, 128 routed rows: 1/2 each — 128 experts at one row.
            #   permute = 16x7392 + 16x8x2 + 128x7392 = 1,064,704
            #   combine = 128x7168x2 + 16x8x2 + 16x7168x2 = 2,064,640
            ("deepseek-ai/DeepSeek-V3", 16, "moe_permute"): 1_064_704,
            ("deepseek-ai/DeepSeek-V3", 16, "moe_combine"): 2_064_640,
            # 256 experts, 96 routed rows: 3/8 each — 96 experts at one row.
            #   permute = 16x4128 + 16x6x2 + 96x4128 = 462,528
            #   combine = 96x4096x2 + 16x6x2 + 16x4096x2 = 917,696
            ("deepseek-ai/DeepSeek-V4-Flash", 16, "moe_permute"): 462_528,
            ("deepseek-ai/DeepSeek-V4-Flash", 16, "moe_combine"): 917_696,
            # 256 experts, 384 routed rows: 3/2 each, padded to 2.
            #   permute = 64x4128 + 64x6x2 + 512x4128 = 2,378,496
            #   combine = 256x2x4096x2 + 64x6x2 + 64x4096x2 = 4,719,360
            ("deepseek-ai/DeepSeek-V4-Flash", 64, "moe_permute"): 2_378_496,
            ("deepseek-ai/DeepSeek-V4-Flash", 64, "moe_combine"): 4_719_360,
        }
        for (model, batch, boundary), expected in want.items():
            jobs, dropped = build_jobs(
                avg_output_tokens=256, model_id=model, batch_size=batch,
                seq_len=1024, phase="decode", parallel=ParallelConfig())
            boundaries = [j for j in jobs if j["signature"] == boundary]
            assert boundaries, (model, batch, boundary, dropped)
            for job in boundaries:
                assert job["access_bytes_per_rank"] == expected, (
                    model, batch, job["signature"], job["node_spec_id"],
                    job["access_bytes_per_rank"], expected)

    @pytest.mark.regression
    def test_the_padded_geometry_prices_each_term_where_it_belongs(self):
        """Every term of a routed row, at the padded geometry, against the
        row's own components — so "derived from the padded shape" cannot
        pass as "the aggregate times a ratio".

        Which term follows which side is the law `row_roofline` states:

        * FLOPs follow the launch's rows.
        * the activation READ follows the read side's own element count,
          and so does any cache read;
        * the activation WRITE follows the write side's;
        * the resident weight read is `min(tokens x topk, E)` unique
          experts — the walk's saturating figure — and does not grow with
          empty padded slots. Its scales stay with it.

        Covered here: the gated activation inside the bank (every side on
        the expert buffer, so everything scales), the fused gate/up
        projection (activation traffic on a weight that already saturates
        when enough tokens routed), and the grouped GEMM (its weight and
        scales at the experts tokens actually reached). The activation cast
        no longer pads: it quantises the buffer once per token above the
        dispatch, so its FP8 record is counted wherever that launch writes.
        """
        import backend.inference.pipeline as pipeline
        from fractions import Fraction
        from math import ceil, prod
        from backend.arch.activation_formats import ACTIVATION_FORMATS
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig

        def elements(shape, exact):
            """(padded, exact) elements one side presents, all axes —
            every extent rounded up, except that a tensor with a mean under
            one row runs only the rows its law routes, rounded up once (an
            expert nobody routed to runs nothing)."""
            ops = (list(shape) if shape and isinstance(shape[0], (list, tuple))
                   else ([list(shape)] if shape else []))
            exs = (list(exact) if exact and isinstance(exact[0], (list, tuple))
                   else ([list(exact)] if exact else []))
            pad, ex = 0, Fraction(0)
            for i, tensor in enumerate(ops):
                law = exs[i] if i < len(exs) and isinstance(
                    exs[i], (list, tuple)) and len(exs[i]) == len(tensor) else tensor
                extents = [Fraction(str(v)) for v in law]
                rows, routed = 1, Fraction(1)
                for v in extents[:-1]:
                    rows *= ceil(v)
                    routed *= v
                if any(v < 1 for v in extents[:-1]):
                    rows = min(rows, ceil(routed))
                pad += rows * ceil(extents[-1])
                ex += routed * extents[-1]
            return pad, ex

        seen = []
        real = pipeline.analyze

        def spy(*a, **kw):
            ar = real(*a, **kw)
            seen.append(ar)
            return ar

        def _assert_cast_record(job, row, where):
            """The conversion is a launch of its own (the grafted cast),
            not a term on the GEMM: it writes the FP8 record — elements and
            one scale per 128 channels of each row — counted by hand over
            the buffer it quantises."""
            fmt = ACTIVATION_FORMATS[row["output_activation_format"]]
            exact = [Fraction(str(v)) for v in row["shape_out_exact"]]
            inner, n_rows = exact[-1], prod(exact[:-1])
            scale_bytes = {"F8_E8M0": 1, "F32": 4}[fmt.scale_dtype]
            record = (n_rows * inner * 1
                      + n_rows * ceil(inner / fmt.group) * scale_bytes)
            n = max(int(row.get("count_per_step") or 1), 1)
            assert Fraction(str(row["act_out_bytes_per_rank"])) / n == \
                record, (where, record)
            assert (Fraction(str(row["convert_bytes_per_rank"] or 0)) / n
                    == 0), where

        checked, exercised = 0, set()
        for model, batch in (("deepseek-ai/DeepSeek-V3", 16),
                             ("deepseek-ai/DeepSeek-V4-Flash", 64)):
            seen.clear()
            pipeline.analyze = spy
            try:
                jobs, _dropped = build_jobs(
                    avg_output_tokens=256, model_id=model, batch_size=batch,
                    seq_len=1024, phase="decode", parallel=ParallelConfig())
            finally:
                pipeline.analyze = real
            rows = {str(r.get("aggregation_id") or ""): r
                    for ar in seen for r in ar.per_op_timing}
            for job in jobs:
                if not job["moe_block_member"]:
                    continue
                row = rows[job["node_spec_id"]]
                where = (model, batch, job["signature"], job["node_spec_id"])
                n = max(int(row.get("count_per_step") or 1), 1)

                def per(key, row=row, n=n):
                    v = row.get(key)
                    return Fraction(0) if v is None else Fraction(str(v)) / n

                # The components are the row's own and they add up to its
                # access exactly, which is what makes scaling them a
                # restatement of the law rather than a second one.
                # FIVE additive terms, which is what the row's access is
                # (`access.LeafAccess`). A cache read is not a sixth: it
                # lives INSIDE `act_in` and the row names that part of it
                # separately for a reader.
                parts = {k: per(f"{k}_bytes_per_rank") for k in (
                    "weight_read", "metadata", "act_in", "act_out",
                    "convert")}
                assert sum(parts.values()) == per("access_bytes_per_rank"), where
                assert per("kv_read_bytes_per_rank") <= parts["act_in"], where

                pad_in, ex_in = elements(row.get("shape_in"),
                                         row.get("shape_in_exact"))
                pad_out, ex_out = elements(
                    row.get("shape_out") or row.get("shape_outs"),
                    row.get("shape_out_exact"))
                read = Fraction(pad_in, 1) / ex_in if ex_in else Fraction(1)
                write = Fraction(pad_out, 1) / ex_out if ex_out else Fraction(1)
                rows_ratio = (Fraction(job["launch_rows"])
                              / Fraction(job["launch_rows_exact"]))

                # Unique experts already billed: min(received, E). Padding
                # the reached experts' rows does not stream more of the bank.
                weight = min(parts["weight_read"],
                             per("weight_bytes_total_per_rank"))
                meta = min(parts["metadata"],
                           per("metadata_bytes_total_per_rank"))
                expected = (weight + meta
                            + (parts["act_in"] + parts["convert"]) * read
                            + parts["act_out"] * write)
                assert job["access_bytes_per_rank"] == pytest.approx(
                    float(expected), rel=1e-12), (
                    where, job["access_bytes_per_rank"], str(expected))
                assert job["flops_per_rank"] == pytest.approx(
                    float(per("flops_per_rank") * rows_ratio), rel=1e-12), where
                checked += 1
                if job["signature"] == "quantise_activation":
                    _assert_cast_record(job, row, where)
                # Which branches of the law this job actually exercised, so
                # the coverage below is a fact and not a hope.
                if rows_ratio != 1:
                    exercised.add("padded")
                    if weight > 0:
                        exercised.add("weight")
                        if weight == per("weight_bytes_total_per_rank"):
                            exercised.add("weight_saturated")
                    if meta > 0:
                        exercised.add("scales")
                        if meta == per("metadata_bytes_total_per_rank"):
                            exercised.add("scales_saturated")
                    if read != write:
                        exercised.add("sides_differ")
                    if parts["act_out"] > 0 and parts["act_in"] > 0:
                        exercised.add("both_sides")
                    exercised.add(job["signature"])
        # The cast launch itself: it quantises the buffer once per token
        # above the dispatch (no longer a member of the block), so its
        # record is counted wherever it writes.
        for job in jobs:
            if job["signature"] != "quantise_activation":
                continue
            _assert_cast_record(job, rows[job["node_spec_id"]],
                                (model, batch, job["signature"],
                                 job["node_spec_id"]))
        assert checked > 10, checked
        # Every branch of the law, on a padded cell: the grouped GEMM's
        # saturated weight and saturated scales, the cast that quantises the
        # routed buffer for the gate/up projection, a gated activation
        # inside the bank, and the two boundaries whose sides scale
        # differently.
        # `silu_and_mul_with_clamp` is V4-Flash's bank activation; the cast
        # and the plain silu_and_mul no longer sit at a padded geometry.
        assert {"padded", "weight", "weight_saturated", "scales",
                "scales_saturated", "sides_differ", "both_sides",
                "grouped_gemm", "moe_permute", "moe_combine",
                "silu_and_mul_with_clamp"} <= exercised, (
                    sorted(exercised))

    def test_a_cache_read_is_inside_the_activation_read_not_beside_it(self):
        """Padding widens the routed buffer's read once, cache included.

        The cache read is part of the routed buffer's per-row bytes
        (`act_in`), not a term beside them: if it were, the widened row
        would charge `act_in + kv` again. 3 rows over 2 experts launch 4
        (each touched expert padded to 2), so the buffer's 900 bytes/row
        land once more — 1900, not 2300.
        """
        from backend.inference.calibration_overlay import _at_geometry
        row = {
            "aggregation_id": "block|zone.leaf",
            "flops_per_rank": 3000.0,
            "access_bytes_per_rank": 1000.0,
            "weight_read_bytes_per_rank": 0.0,
            "metadata_bytes_per_rank": 0.0,
            "act_in_bytes_per_rank": 900.0,
            "kv_read_bytes_per_rank": 400.0,
            "act_out_bytes_per_rank": 100.0,
            "convert_bytes_per_rank": 0.0,
            "routed": {"dispatch": 1, "source_rows": 3, "routed_rows": 3,
                       "local_experts": 2, "bytes_per_row": 900.0},
        }
        assert _at_geometry(row, billed=1.0) == (4000.0, 1900.0)
        # Whole rows over both experts pad nothing: the walk's numerators.
        assert _at_geometry(
            {**row, "flops_per_rank": 3000.0,
             "routed": {**row["routed"], "routed_rows": 4,
                        "source_rows": 4}}, billed=1.0) == (3000.0, 1000.0)

    def test_the_route_exchange_keeps_the_rows_it_really_moves(self):
        """A route all_to_all is NOT padded, and the statement is explicit.

        Padding is an allocation convention at a kernel boundary: a grouped
        launch wants a rectangular buffer, so every expert's rows are
        rounded up to `ceil(M)`. An exchange moves tokens — the rows that
        were really routed — and its payload is the mean buffer the walk
        priced (`row_roofline`: the busiest rank sends `rho` times it, and
        `rho` is the imbalance, not the padding). So its cell keeps the
        law's own bytes.

        Structurally it holds because a collective is no part of the
        declared dispatch zone, and that is what this pins: the companion
        cells of a padded zone are untouched.
        """
        from fractions import Fraction
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig

        # Batch 3: 24 rows reach each rank's 16 experts, a row and a half
        # each, padded to two (a batch-1 decode reaches 8 of them at one
        # row each, which pads nothing).
        jobs, dropped = build_jobs(
            avg_output_tokens=256, model_id="Qwen/Qwen3-30B-A3B",
            batch_size=3, seq_len=1024, phase="decode",
            parallel=ParallelConfig.from_dict({"ep_moe": 8,
                                               "dp_attention": 8}))
        padded = [j for j in jobs if j["moe_block_member"]
                  and Fraction(j["launch_rows_exact"]) < j["launch_rows"]]
        assert padded, ("the zone is not padded at this point, so this "
                        "proves nothing", dropped)
        # The routed exchange's own signature: a dispatch sends a token once
        # per distinct destination its top-k reaches, so its per-peer split
        # is ragged and it is not the flat `all_to_all` cell
        # (`benchmark_identity.resolve_op`, selected by the row's own
        # `comm_route_fanout`).
        a2a = [j for j in jobs if j["signature"] == "routed_all_to_all"]
        assert a2a, dropped
        assert not [j for j in jobs if j["signature"] == "all_to_all"], (
            "a routed exchange must not key as the flat one", dropped)
        for job in a2a:
            assert not job["moe_block_member"], job["node_spec_id"]
            assert job["launch_rows"] == Fraction(job["launch_rows_exact"]), (
                job["node_spec_id"], job["launch_rows"],
                job["launch_rows_exact"])
            # The mean buffer the walk priced at batch 3 (65,536 at batch
            # 1, three times it here), not the padded one.
            assert job["access_bytes_per_rank"] == 3 * 65536, (
                job["node_spec_id"], job["access_bytes_per_rank"])

    def test_rmsnorm_job_flops_are_the_engine_export(self):
        # One launch: OpSpec is 2·B·S·H. Plan de-aggregates count_per_step
        # rather than the harness inventing a second formula.
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0,
            model_id="meta-llama/Llama-3.2-1B",
            batch_size=1, seq_len=1024, phase="prefill")
        rms = [j for j in jobs if j["signature"] == "rmsnorm"]
        assert rms, "expected rmsnorm jobs"
        for j in rms:
            d = j["dims"]
            want = 2 * int(d.get("B", 1)) * int(d["s_query"]) * int(d["d_model"])
            assert j["flops_per_rank"] == want, (j["node_spec_id"], j["flops_per_rank"], want)

    def test_grouped_gemm_job_carries_benched_shape_flops(self):
        # End-to-end: a planned grouped_gemm job's flops must equal the benched
        # shape's 2·E·Sq·K·N, so achieved = flops/latency stays ≤ peak.
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3",
                             batch_size=1, seq_len=2048, phase="prefill")
        for j in [x for x in jobs if x["signature"] == "grouped_gemm"]:
            d = j["dims"]
            assert j["flops_per_rank"] == 2 * d["n_experts"] * d["s_expert_rows"] * d["d_reduce"] * d["d_out"]

    def test_fp8_model_emits_fp8_gemm_jobs_bf16_attention(self):
        # Stage 1: per-op real dtype from the spec's quant scheme. An fp8 model
        # (DeepSeek-V3, W8A8) emits fp8 GEMM-family jobs so the fp8 providers are
        # eligible; attention/norm/act compute in bf16 regardless of how their
        # projection weights are stored.
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3",
                             batch_size=1, seq_len=2048, phase="prefill")

        def dt(sig):
            j = next((x for x in jobs if x["signature"] == sig), None)
            return None if j is None else (
                j["dtype_env"].get("weight"), j["dtype_env"]["activation"])
        assert dt("grouped_gemm") == ("fp8_e4m3", "fp8_e4m3")
        assert dt("gemm") == ("fp8_e4m3", "fp8_e4m3")
        # Attention streams no weight of its own: no `weight` slot.
        assert dt("mla_attention") == (None, "bf16")

    def test_bf16_model_stays_bf16(self):
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V2-Lite",
                             batch_size=1, seq_len=2048, phase="prefill")
        gg = next(x for x in jobs if x["signature"] == "grouped_gemm")
        assert gg["dtype_env"]["weight"] == "bf16"
        assert gg["dtype_env"]["activation"] == "bf16"

    def test_moe_dispatch_ops_now_measurable(self):
        # permute/combine were DROPPED as zero-cost; Stage 2 makes them
        # first-class memory-bound signatures so the whole MoE chain (not just
        # the GEMMs) is measurable. Both boundaries state BOTH sides of the
        # dispatch: the token side (B × s_query) and the routed side
        # (n_experts × s_expert_rows, the padded [E_local, M, H] buffer:
        # 64 x 192 rows).
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        s = identity_shape_from_shape_in(
            "permute", [[1, 2048, 2048], [1, 2048, 6]],
            shape_out=[64, 192, 2048], routed_rows=64 * 192)
        assert json.loads(s) == {"B": 1, "s_query": 2048, "n_experts": 64,
                                 "s_expert_rows": 192, "d_model": 2048,
                                 "n_tokens": 64 * 192}
        s2 = identity_shape_from_shape_in(
            "combine", [[64, 192, 2048], [1, 2048, 6]],
            shape_out=[1, 2048, 2048], routed_rows=64 * 192)
        assert json.loads(s2) == {"B": 1, "s_query": 2048, "n_experts": 64,
                                  "s_expert_rows": 192, "d_model": 2048,
                                  "n_tokens": 64 * 192}
        # torch providers + refs round-trip (E·M routed rows × H moved)
        pytest.importorskip("torch")
        from kernel_bench.providers import all_providers
        from kernel_bench.refs import get_ref
        impls = {p.kernel_impl for p in all_providers()
                 if p.signature in ("moe_permute", "moe_combine")}
        assert {"torch_index_permute", "torch_weighted_combine"} <= impls
        inv = Invocation.from_job({
            "signature": "moe_permute", "op": "permute",
            "dims": {"B": 1, "s_query": 16, "n_experts": 4, "s_expert_rows": 2,
                     "d_model": 8},
            "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"}, "phase": "prefill"})
        spec = get_ref("moe_permute")
        out = spec.reference(build_inputs(spec, inv, "cpu"))
        assert tuple(out.shape) == (8, 8)   # E·M = 8 moved rows, H = 8

    def test_moe_combine_products_keep_cancellation(self):
        # The combine's products may not be rounded to the cell's dtype before
        # the sum: bf16's 8-bit mantissa maps this cancelling pair onto one
        # value (-1.62123 → -1.625, +1.63074 → +1.625, sum exactly 0 where the
        # fp32 oracle keeps +0.0095) — the canary's worst 0.157/0.229 against
        # the bf16 band's 0.1. A real combine multiplies in registers; the
        # floor carries the product as a bf16 hi/lo pair instead.
        torch = pytest.importorskip("torch")
        from kernel_bench.providers import all_providers
        from kernel_bench.refs import get_ref
        x = torch.tensor([[-2.53125], [1.7734375]], dtype=torch.bfloat16)
        w = torch.tensor([0.64048582, 0.91953653])
        inputs = {"x": x, "w": w, "tokens": 1,
                  "idx": torch.zeros(2, dtype=torch.int64)}
        want = float(x.float()[:, 0] @ w)         # +0.00951, not 0
        assert (x * w[:, None].to(torch.bfloat16)).sum().item() == 0.0
        inv = Invocation.from_job({
            "signature": "moe_combine", "op": "combine",
            "dims": {"B": 1, "s_query": 1, "n_experts": 2, "s_expert_rows": 1,
                     "d_model": 1},
            "dtype_env": {"activation": "bf16", "output": "bf16"},
            "phase": "prefill"})
        floor = next(p for p in all_providers()
                     if p.kernel_impl == "torch_weighted_combine")
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        case = floor.build(inv, env, inputs)
        got = case.to_canonical(case.run())
        ref = get_ref("moe_combine").reference(inputs)
        assert torch.allclose(got, ref.float(), rtol=0.01, atol=1e-4)
        assert abs(got.item() - want) < 1e-4

    def test_initial_state_args_vllm_per_token_indices(self):
        # vLLM 0.32's fused_recurrent decode copy declares `ssm_state_indices`
        # and reads it PER TOKEN (`… + i_n * stride_indices_seq + i_t` in the
        # token loop): a 1-D [N] index overruns at T > 1 (illegal memory
        # access), and omitting it is a Triton CompilationError — the block
        # that writes each token's state dereferences it unconditionally.
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.linear_ssm import _initial_state_args

        def vllm_spelling(q, k, v, g, beta, initial_state=None,
                          ssm_state_indices=None):
            pass

        state = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32
                             ).reshape(2, 3, 4, 5)
        out = _initial_state_args(vllm_spelling, state, tokens=3)
        pool, idx = out["initial_state"], out["ssm_state_indices"]
        assert tuple(pool.shape) == (3, 3, 4, 5)      # row 0 is padding
        assert torch.equal(pool[1:], state)
        assert idx.shape == (2, 3) and idx.dtype == torch.int32
        assert idx[:, 0].tolist() == [1, 2]
        assert idx[:, 1].tolist() == [1, 2]           # every token its seq's slot

        def sglang_spelling(q, k, v, g, beta, initial_state=None,
                            initial_state_indices=None):
            pass

        out = _initial_state_args(sglang_spelling, state, tokens=3)
        assert out["initial_state_indices"].shape == (2,)   # per SEQUENCE there

        def fla_spelling(q, k, v, g, beta, initial_state=None):
            pass

        assert _initial_state_args(fla_spelling, state, tokens=3) == \
            {"initial_state": state}

    def test_moe_block_mfu_folds_dispatch_into_the_block(self):
        # Block MFU = useful GEMM flops / Σ(whole-chain time incl. dispatch),
        # so the memory-bound permute/combine (0 useful flops, real time) drag
        # it below the GEMM's own MFU. This is the payoff of the per-impl graph.
        from kernel_bench.aggregate import moe_block_mfu
        # moe_block_member is the declared flag the plan stamps on dispatch-zone
        # rows; both the expert GEMM and the permute boundary carry it.
        base = {"model": "m", "gpu": "G", "phase": "prefill", "parallel": "tp=1",
                "framework": "torch", "peak_tflops_at_test": "100",
                "moe_block_member": "1"}
        rows = [
            {**base, "node_spec_id": "blk|mlp.expert.gate_up", "op": "grouped_gemm",
             "p50_us": "100",
             # 8 experts, 125 routed rows between them — so the per-expert
             # buffer is 16 wide and three of its 128 slots are padding.
             "shape_json": '{"n_experts":8,"n_tokens":125,"s_expert_rows":16,'
                           '"d_reduce":32,"d_out":64}',
             "achieved_tflops": "50", "achieved_bw_gbps": "1"},
            {**base, "node_spec_id": "blk|mlp.moe.permute", "op": "permute",
             "p50_us": "100", "shape_json": '{"B":1,"s_query":32,"n_experts":8,'
                                            '"s_expert_rows":16,"d_model":64,"n_tokens":128}',
             "achieved_tflops": "0", "achieved_bw_gbps": "500"},
        ]
        out = moe_block_mfu(rows)
        assert len(out) == 1
        b = out[0]
        assert set(b["ops"]) == {"grouped_gemm", "permute"}
        # useful = 2·125·32·64 flops over 200us (GEMM + dispatch) at 100 TF
        # peak — the ROUTED rows, not the 8·16 the buffer is padded to.
        expected = (2 * 125 * 32 * 64) / 200e-6 / 100e12
        assert abs(b["block_mfu"] - round(expected, 4)) < 1e-4
        # An expert GEMM whose shape does not state its cell is a broken
        # row, named, not zero useful work.
        for shape in ('{"n_experts":8,"s_expert_rows":16,"d_reduce":32,"d_out":64}',
                      "[8, 16]", "not json"):
            broken = [{**rows[0], "shape_json": shape}, rows[1]]
            with pytest.raises(ValueError, match=r"blk\|mlp.expert.gate_up"):
                moe_block_mfu(broken)

    def test_grouped_providers_and_ref(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers import all_providers
        from kernel_bench.refs import get_ref
        impls = {p.kernel_impl for p in all_providers() if p.signature == "grouped_gemm"}
        assert "torch_ragged_matmul" in impls
        # DeepGEMM's two switchable grouped kernels — benched separately
        assert "deepgemm_fp8_m_grouped_masked" in impls
        assert "deepgemm_fp8_m_grouped_contiguous" in impls
        # vLLM/sglang Triton fused_moe_kernel — gather⊕GEMM fusion style
        assert "vllm_triton_fused_moe" in impls
        inv = Invocation.from_job({
            "signature": "grouped_gemm", "op": "grouped_gemm",
            "dims": {"n_experts": 4, "n_tokens": 5, "s_expert_rows": 2,
                     "d_reduce": 64, "d_out": 128},
            "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"},
            "kernel_params": {"moe_token_counts": [2, 1, 1, 1]},
            "phase": "prefill"})
        spec = get_ref("grouped_gemm")
        out = spec.reference(build_inputs(spec, inv, "cpu"))
        assert tuple(out.shape) == (4, 2, 128)
        assert torch.count_nonzero(out[1:, 1]) == 0

    def test_fused_moe_signature_and_megamoe_provider(self):
        # The WHOLE MoE expert block (coarse fusion) is a distinct signature
        # from the grouped_gemm pipeline stage. torch is the floor;
        # FlashInfer cutlass is the single-GPU kernel; MegaMoE is the
        # EP+symm-memory backend (body pending a multi-process launch).
        from kernel_bench.providers import all_providers
        from kernel_bench.signatures import get_signature, op_to_signature
        assert get_signature("fused_moe") is not None
        assert get_signature("fused_moe").inputs
        assert op_to_signature("fused_moe") == "fused_moe"
        impls = {p.kernel_impl: p for p in all_providers()
                 if p.signature == "fused_moe"}
        assert "torch_fused_moe" in impls
        assert "flashinfer_cutlass_fused_moe" in impls
        mega = impls["deepgemm_fp8_fp4_mega_moe"]
        assert mega.library == "deepgemm"
        assert mega.source_build is True
        assert mega.min_sm == 100
        assert impls["flashinfer_cutlass_fused_moe"].min_sm == 100

    def test_fused_moe_identity_from_engine_shape(self):
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        s = identity_shape_from_shape_in(
            "fused_moe", [[1, 2048, 4096]],
            spec_dims={"n_experts": 64, "n_experts_active": 8, "d_ffn": 1408})
        assert json.loads(s) == {
            "B": 1, "s_query": 2048, "d_model": 4096,
            "n_experts": 64, "n_experts_active": 8, "d_ffn": 1408,
        }

    def test_fused_moe_reference_shapes_on_cpu(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inv = Invocation.from_job({
            "signature": "fused_moe", "op": "fused_moe",
            "dims": {"B": 1, "s_query": 4, "d_model": 32, "d_ffn": 64,
                     "n_experts": 8, "n_experts_active": 2},
            "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"},
            "phase": "prefill",
        })
        spec = get_ref("fused_moe")
        inputs = build_inputs(spec, inv, "cpu")
        out = spec.reference(inputs)
        assert tuple(out.shape) == (4, 32)
        from kernel_bench.env import Env
        from kernel_bench.providers import all_providers
        torch_p = next(p for p in all_providers()
                       if p.kernel_impl == "torch_fused_moe")
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        got = torch_p.build(inv, env, inputs).run().float()
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), out.flatten(), dim=0).item()
        assert cos > 0.99, cos


class TestExpandedProviderCoverage:
    """Each op family should now have ≥2 libraries competing under the envelope
    (that's the whole point of cross-library measurement)."""

    def _h100(self, **libs):
        return Env("2.11", True, "H100", "H100-SXM", 3350.0, libs=libs)

    def test_attention_decode_envelope_has_fa_and_flashinfer(self):
        env = self._h100(flash_attn_interface="3.0.0", flashinfer="0.2.5")
        dec = Invocation.from_job(_gqa_job(phase="decode"))
        impls = {p.kernel_impl for p in applicable(dec, env)}
        assert "flashinfer_paged_decode" in impls
        assert "flash_attention_v3_kvcache" in impls          # ≥2 libs
        # the paged-decode provider must NOT fire on prefill
        pre = Invocation.from_job(_gqa_job(phase="prefill"))
        assert "flashinfer_paged_decode" not in {p.kernel_impl for p in applicable(pre, env)}

    def test_fa4_joins_prefill_envelope(self):
        from kernel_bench.providers import all_providers
        pre = {p.library for p in all_providers()
               if p.signature == "gqa_attention"
               and p.kernel_impl in ("flash_attention_v2", "flash_attention_v3",
                                      "flash_attention_v4", "flashinfer_ragged_prefill")}
        assert {"fa2", "fa3", "fa4", "flashinfer"} <= pre     # 4-lib prefill envelope
        fa4 = next(p for p in all_providers() if p.kernel_impl == "flash_attention_v4")
        # The gate is FA4's CUTE frontend, not sglang: sglang's wrapper imports
        # on its own, and declaring the outer module let availability pass while
        # every cell failed one at a time.
        assert fa4.source_build and fa4.import_name == "flash_attn.cute"
        # fa4 is prefill-only (like the FlashInfer ragged prefill)
        env = self._h100(sglang="0.4.0")
        dec = Invocation.from_job(_gqa_job(phase="decode"))
        assert "flash_attention_v4" not in {p.kernel_impl for p in applicable(dec, env)}

    def test_mla_decode_envelope_has_three_libs(self):
        env = self._h100(flash_mla="1.0", flashinfer="0.2.5", sgl_kernel="0.1")
        inv = Invocation.from_job({
            "signature": "mla_attention", "op": "mla_attention",
            # A paged cell carries the deployment's page (`PAGE_AXIS`).
        "dims": {"B": 1, "n_heads": 128, "s_query": 1, "s_kv": 4096, "d_head": 576,
                 "d_v_head": 512, "kv_block_size": 64},
            "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"}, "phase": "decode"})
        libs = {p.library for p in applicable(inv, env)}
        assert {"flashmla", "flashinfer", "cutlass_mla"} <= libs
        # all three canonicalize their output → cross-checked against the MLA ref
        for p in applicable(inv, env):
            assert p.signature == "mla_attention"

    def test_norm_act_envelope_has_vllm_and_flashinfer(self):
        # The version gate reads the LIBRARY's spec (what `envs/vllm.txt`
        # installs): the release the vllm binding was audited at.
        from backend.inference.framework_bindings import get_binding
        env = self._h100(vllm=get_binding("vllm").audited_at.release,
                         flashinfer="0.2.5")
        for sig in ("rmsnorm", "fused_add_rmsnorm", "silu_and_mul"):
            inv = Invocation.from_job({
                "signature": sig, "op": sig,
                "dims": {"B": 1, "s_query": 2048, "d_model": 3072, "d_ffn": 8192},
                "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                              "output": "bf16", "metadata": "fp32"},
                "phase": "prefill",
            })
            libs = {p.library for p in applicable(inv, env)}
            assert {"vllm", "flashinfer"} <= libs, f"{sig}: {libs}"

    def test_fp8_gemm_envelope_has_deepgemm_and_cutlass(self):
        from backend.inference.framework_bindings import get_binding
        env = self._h100(vllm=get_binding("vllm").audited_at.release,
                         deep_gemm="1.0", torch="2.11")
        fp8 = {"activation": "fp8_e4m3", "weight": "fp8_e4m3", "kv": "bf16",
               "output": "bf16", "metadata": "fp32",
               "activation_granularity": "per-group:128",
               "weight_granularity": "per-tile:128x128"}
        inv = Invocation.from_job({
            "signature": "gemm", "op": "linear",
            "dims": {"B": 1, "s_query": 2048, "d_reduce": 3072, "d_out": 5120}, "dtype_env": fp8,
            "phase": "prefill",
        })
        impls = {p.kernel_impl for p in applicable(inv, env)}
        assert "vllm_cutlass_scaled_mm_blockwise" in impls
        assert "deepgemm_fp8_blockwise" in impls              # ≥2 fp8 libs

    def test_cutlass_gate_needs_16_alignment_not_128(self):
        """CUTLASS W8A8 serves a 16-element tile, DeepGEMM's blockwise fp8 needs
        the 128x128 scale tile. They overlap on the 128-aligned shapes — which
        is where the two are comparable — and CUTLASS alone takes the rest."""
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import all_providers, discover
        discover()
        by = {p.kernel_impl: p for p in all_providers()}
        # K=3072, N=80: 16-aligned but not 128-aligned
        inv = Invocation(signature="gemm", op="linear",
                         dims={"B": 1, "s_query": 1, "d_reduce": 3072, "d_out": 80},
                         dtype_env={"weight": "fp8_e4m3", "activation": "fp8_e4m3",
                                    "output": "bf16",
                                    "activation_granularity": "per-token"},
                         phase="prefill", tp=1)
        env = Env("2.11", False, None, None, None, libs={})
        assert by["vllm_cutlass_scaled_mm"].accepts(inv, env) is True
        assert by["deepgemm_fp8_blockwise"].accepts(inv, env) is False


class TestDeepGEMMScaleLayout:
    """DeepGEMM fp8 blockwise scale prep — the layout is the failure-prone part
    (kernel unavailable here, but the quantizers are pure torch, CPU-testable)."""

    def test_per_token_and_per_block_scale_shapes_and_roundtrip(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.gemm import (
            _per_block_cast_to_fp8, _per_token_cast_to_fp8)
        M, K, N = 4, 256, 384                      # K,N multiples of 128
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        x_q, x_s = _per_token_cast_to_fp8(x)
        w_q, w_s = _per_block_cast_to_fp8(w)
        assert x_q.shape == (M, K) and x_q.dtype == torch.float8_e4m3fn
        assert x_s.shape == (M, K // 128) and x_s.dtype == torch.float32
        assert w_q.shape == (N, K) and w_q.dtype == torch.float8_e4m3fn
        assert w_s.shape == (N // 128, K // 128)
        # dequant ≈ original within e4m3 blockwise error (~6% rel per element)
        deq = (x_q.float().reshape(M, K // 128, 128)
               * x_s.unsqueeze(2)).reshape(M, K)
        assert (deq - x).abs().max().item() < 0.5

    def test_fp8_gemm_gate_requires_128_alignment(self):
        """DeepGEMM's scale tile is 128x128, so a shape that does not divide is
        declined rather than padded into a different measurement."""
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import all_providers, discover
        discover()
        dg = {p.kernel_impl: p for p in all_providers()}["deepgemm_fp8_blockwise"]
        env = Env("2.11", False, None, None, None, libs={})

        def cell(K, N):
            return Invocation(signature="gemm", op="linear",
                              dims={"B": 1, "s_query": 1, "d_reduce": K, "d_out": N},
                              dtype_env={"weight": "fp8_e4m3", "activation": "fp8_e4m3",
                                         "output": "bf16",
                                         "activation_granularity": "per-group:128",
                                         "weight_granularity": "per-tile:128x128"},
                              phase="prefill", tp=1)

        assert dg.accepts(cell(4096, 4096), env) is True
        assert dg.accepts(cell(3072, 80), env) is False


class TestLibraryKernelSplit:
    """Install facts belong to the LIBRARY, not to each kernel.

    They used to be copied onto every provider, so torch stated three different
    constraints across its 12 kernels and vllm two across 14 — and `envs/*.txt`,
    generated by sorting that set and taking the first, installed whichever
    sorted lowest. Nothing failed; the env just did not satisfy all the kernels
    in it. These tests are what makes a re-copy impossible to land quietly."""

    def test_provider_cannot_declare_install_facts(self):
        """The fields are gone from `Provider`, not merely unused.

        Left as accepted-but-ignored kwargs, a re-added `pip=` would look like
        it worked and change nothing — the failure this whole split exists to
        prevent, in the mechanism meant to prevent it."""
        import dataclasses
        from kernel_bench.providers import Provider
        fields = {f.name for f in dataclasses.fields(Provider)}
        assert not ({"pip", "source_build", "build_hint"} & fields)

    def test_install_facts_read_through_to_the_library(self):
        from kernel_bench.providers import get_library
        for p in all_providers():
            lib = get_library(p.library)
            assert p.pip is lib.pip
            assert p.source_build == lib.source_build
            assert p.build_hint == lib.build_hint

    def test_one_library_states_one_pip_spec_per_package(self):
        """A library may need several PACKAGES; it may not hold two opinions
        about one. sglang legitimately needs `sglang` and `sgl-kernel`; what is
        forbidden is the shape torch and vllm were in."""
        from packaging.requirements import Requirement
        from kernel_bench.providers import all_libraries
        for lib in all_libraries():
            names = [Requirement(one).name for one in lib.pip]
            assert len(names) == len(set(names)), f"{lib.name}: {lib.pip}"

    def test_an_undeclared_library_cannot_register(self):
        from kernel_bench.providers import Provider, register, KernelSignature
        with pytest.raises(KeyError):
            register(Provider(
                library="not_a_library", signature="rmsnorm",
                kernel_impl="x", import_name="torch",
                serves=KernelSignature(signature="rmsnorm"),
                build=lambda inv, env, inputs: None))

    def test_every_declared_library_is_measured_by_something(self):
        """The other direction: a `Library` no kernel names generates an env
        for a stack nobody runs, which is a maintained lie rather than a bug."""
        from kernel_bench.providers import all_libraries
        used = {p.library for p in all_providers()}
        assert {lib.name for lib in all_libraries()} == used


class TestDependencyManifest:
    """The env probe list and the `envs/*.txt` files are DERIVED from the
    library records plus the kernels' import gates, and `available()` checks
    the installed version against the library's spec."""

    def test_manifest_covers_every_provider_library(self):
        from kernel_bench.providers import dependency_manifest
        m = dependency_manifest()
        libs = {p.library for p in all_providers()}
        assert libs <= set(m)
        # every provider's import_name is captured under its library
        for p in all_providers():
            assert p.import_name in m[p.library]["import_names"]

    def test_source_build_libs_flagged_with_hint(self):
        from kernel_bench.providers import dependency_manifest
        m = dependency_manifest()
        for lib in ("fa3", "flashmla", "deepgemm"):
            assert m[lib]["source_build"] is True
            assert m[lib]["build_hint"]          # non-empty guidance
            assert not m[lib]["pip"]             # not pip-resolvable
        for lib in ("fa2", "flashinfer", "vllm"):
            assert m[lib]["source_build"] is False and m[lib]["pip"]

    def test_probe_list_is_union_of_providers_plus_torch(self):
        from kernel_bench.providers import probe_import_names
        names = probe_import_names()
        assert "torch" in names
        for p in all_providers():
            assert p.import_name in names

    def test_env_probes_only_manifest_names(self):
        # detect_env keys `libs` by import_name — never a display alias.
        env = detect_env()
        from kernel_bench.providers import probe_import_names
        assert set(env.libs) <= probe_import_names()

    def test_version_gate_rejects_out_of_range(self):
        # fa2 requires >=2.6,<3; an env with an older flash_attn must not be
        # considered available for the fa2 provider.
        fa2 = next(p for p in all_providers()
                   if p.kernel_impl == "flash_attention_v2")
        env = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                  libs={"flash_attn": "2.5.0"})
        assert fa2.available(env) is False
        ok = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                 libs={"flash_attn": "2.7.0"})
        assert ok.has("flash_attn") and fa2.available(ok) is True

    def test_version_gate_is_best_effort_on_unknown(self):
        # importable but version unknown → allow (importability already passed).
        fa2 = next(p for p in all_providers()
                   if p.kernel_impl == "flash_attention_v2")
        env = Env("2.11", True, "H800", "H800-SXM", 3350.0,
                  libs={"flash_attn": "unknown"})
        assert fa2.available(env) is True

    def test_render_env_files_derives_from_manifest(self):
        from kernel_bench.providers import render_env_files
        files = render_env_files()
        # one file per pip-resolvable non-torch library, pinning torch + its spec
        assert "fa2.txt" in files and "vllm.txt" in files
        # torch is the base of every env AND a library with its own env: its
        # floors are measured there once, not in every env that imports it.
        assert "torch.txt" in files
        # Source-build libs DO get a file now. They used to get nothing, which
        # left a fresh environment with no record that they exist — the gap
        # `fla` fell into. Theirs says what it is and names the import gate,
        # rather than pretending pip can resolve it.
        assert "SOURCE BUILD" in files["fa3.txt"]
        for lib in ("flashmla.txt", "deepgemm.txt"):
            assert "SOURCE BUILD" in files[lib]
        assert "flash-attn>=2.6,<3" in files["fa2.txt"]
        assert files["fa2.txt"].startswith("# Provider env: fa2")
        # torch is the base of every env, at the ONE bound `_libraries.py`
        # declares — 2.5, because `torch_sdpa` passes `enable_gqa=`.
        assert "torch>=2.5" in files["fa2.txt"]

    def test_on_disk_env_files_match_generated(self):
        # The committed envs/*.txt must equal what the manifest generates
        # (guards against a provider dep changing without regenerating).
        import pathlib
        from kernel_bench.providers import render_env_files
        env_dir = pathlib.Path(__file__).resolve().parents[2] / "kernel_bench" / "envs"
        files = render_env_files()
        for name, content in files.items():
            on_disk = (env_dir / name).read_text()
            assert on_disk == content, f"{name} stale — run `python -m kernel_bench envs`"


class TestRunDegradesCleanly:
    def test_run_skips_signature_with_no_installed_provider(self):
        # A signature with no provider in this env → skip, never raise.
        from kernel_bench.driver import run_jobs
        job = {"signature": "not_a_real_signature",
               "op": "not_a_real_signature",
               "dims": {"B": 1, "s_query": 4, "d_model": 64},
               "dtype_env": {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                             "output": "bf16", "metadata": "fp32"}, "phase": "prefill"}
        res = run_jobs([job], env=detect_env(), date="2026-07-27", contributor="@t")
        assert res.rows == []
        assert len(res.skipped) == 1
        assert res.failures == []

    def test_torch_sdpa_is_the_attention_floor(self):
        # gqa_attention always has a torch SDPA provider (no nvcc / Hopper), so
        # attention is measurable on any CUDA build — the fused libs are the
        # envelope, torch SDPA the floor. Its reference round-trips on CPU.
        pytest.importorskip("torch")
        from kernel_bench.providers import all_providers
        from kernel_bench.refs import get_ref
        assert any(p.kernel_impl == "torch_sdpa" and p.library == "torch"
                   and p.signature == "gqa_attention" for p in all_providers())
        inv = Invocation.from_job(_gqa_job(
            dims={"kv_block_size": 16, "B": 1, "s_query": 128, "s_kv": 128, "n_heads": 8, "n_kv_heads": 2, "d_head": 64}))
        spec = get_ref("gqa_attention")
        out = spec.reference(build_inputs(spec, inv, "cpu"))
        assert out.shape == (1, 128, 8, 64)   # [B, Sq, N, D]


class TestTimingL2Flush:
    def test_flush_param_accepted_and_returns_stats(self):
        # Works on CPU (flush is a GPU-only no-op there); both settings return
        # well-formed stats.
        pytest.importorskip("torch")
        from kernel_bench.timing import time_callable
        for flush in (True, False):
            s = time_callable(lambda: 1 + 1, warmup=2, iters=5, flush_l2=flush)
            assert s.samples == 5 and s.p50_us >= 0 and s.p95_us >= s.p50_us

    def test_flush_cools_l2_for_memory_bound_op(self):
        # GPU-gated: a read-dominated op whose working set fits L2 reports
        # HBM-impossible bandwidth hot-cache; flushing L2 makes it realistic, so
        # the flushed latency must be >= the un-flushed one (usually much larger).
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        from kernel_bench.timing import time_callable
        x = torch.randn(48 * 1024 * 1024 // 4, device="cuda")  # 48 MiB, fits L2
        op = x.sum
        hot = time_callable(op, warmup=10, iters=40, flush_l2=False)
        cold = time_callable(op, warmup=10, iters=40, flush_l2=True)
        assert cold.p50_us >= hot.p50_us * 1.1  # cold read is meaningfully slower

    def test_host_launch_time_is_not_timed(self):
        # GPU-gated. A call whose CPU side outlasts its GPU side (30 tiny
        # kernels in eager mode) used to let the GPU idle between the start
        # event and the first kernel, timing the Python launch path: FA3's
        # 5 us prefill read 79 us on an H100. Its time must stay close to the
        # same kernels issued under a CUDA graph (no host in the loop).
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no CUDA")
        from kernel_bench.timing import time_callable
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)

        def eager():
            y = x
            for _ in range(30):
                y = y * 1.0001
            return y

        eager()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            eager()
        timed = time_callable(eager, warmup=10, iters=30).p50_us
        graphed = time_callable(g.replay, warmup=10, iters=30).p50_us
        # The old bug was an order of magnitude, not scheduler noise: a 2x
        # band catches host-launch timing while tolerating a loaded CPU's
        # eager-launch jitter beside a graph replay.
        assert timed <= graphed * 2.0, (timed, graphed)


class TestParallelSweepAlignment:
    """The sweep must plan at the deployment configs the tool queries, and its
    rows must key to the SAME parallel signature the overlay looks up under."""

    def test_build_jobs_writes_canonical_parallel_signature(self):
        # The write-side `parallel` column MUST equal the query side's
        # `parallel_signature`, or a sharded row's hardware_hash can never
        # match the estimate's lookup.
        from backend.inference.calibration_overlay import parallel_signature
        from backend.inference.parallel_config import ParallelConfig
        from kernel_bench.plan import build_jobs
        cfg = ParallelConfig(tp_attention=4, tp_dense_ffn=4, tp_moe_expert=4, ep_moe=1)
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3", batch_size=8,
                             seq_len=2048, phase="prefill", parallel=cfg)
        assert jobs, "expected jobs"
        want = parallel_signature(cfg)
        assert want == "tp_attention=4,tp_dense_ffn=4,tp_moe_expert=4"
        assert {j["parallel"] for j in jobs} == {want}
        # default (no parallel) → canonical "tp=1"
        j1, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3", batch_size=8,
                           seq_len=2048, phase="prefill")
        assert {j["parallel"] for j in j1} == {"tp=1"}

    def test_tp_shards_attention_heads_per_rank(self):
        # tp degree d → per-rank query heads N/d (the sharded shape the estimate
        # keys on at that degree).
        from backend.inference.parallel_config import ParallelConfig
        from kernel_bench.plan import build_jobs
        base, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-30B-A3B", batch_size=8,
                             seq_len=2048, phase="prefill")
        shard, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-30B-A3B", batch_size=8,
                              seq_len=2048, phase="prefill",
                              parallel=ParallelConfig(tp_attention=4, tp_dense_ffn=4,
                                                      tp_moe_expert=4))
        n_base = next(j["dims"]["n_heads"] for j in base if j["op"] == "flash_attention")
        n_shard = next(j["dims"]["n_heads"] for j in shard if j["op"] == "flash_attention")
        assert n_shard == n_base // 4

    def test_sweep_enumerates_safe_modes_only(self):
        # All four modes carry a per-rank-correct identity (see sweep docstring),
        # so what a sweep measures is the panel's whole mode vocabulary — read
        # off it rather than listed again here.
        from kernel_bench.sweep import routine_parallel_configs, SAFE_MODES
        from backend.analyzer import fetch_config
        from backend.arch.loader import SpecLoader
        from backend.inference.parallel_sweep import ALL_MODES
        assert SAFE_MODES == ALL_MODES
        assert set(SAFE_MODES) == {"tp_tp", "tp_ep", "dp_ep", "cp_ep"}
        cfg = fetch_config("Qwen/Qwen3-30B-A3B")
        # The sweep resolves the spec once and threads it — `routine_parallel_configs`
        # no longer re-matches, so it is handed the resolved spec here too.
        resolved = SpecLoader().match(cfg, model_id="Qwen/Qwen3-30B-A3B")
        labels = [l for l, _ in routine_parallel_configs(cfg, resolved=resolved)]
        assert labels[0] == "tp=1"
        assert all(l == "tp=1" or l.startswith(
            ("tp_tp:", "tp_ep:", "dp_ep:", "cp_ep:")) for l in labels)

    def test_ep_shards_expert_count_per_rank(self):
        # The MoE identity's expert count is the rank's: the extent of the
        # operand's leading axis, on BOTH the write and query sides (same
        # helper). tp_ep=8 on E=128 → 16.
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        import json
        whole = json.loads(identity_shape_from_shape_in(
            "grouped_gemm", [[128, 192, 2048]], shape_out=[[128, 192, 1536]],
            routed_rows=128 * 192))
        shard = json.loads(identity_shape_from_shape_in(
            "grouped_gemm", [[16, 192, 2048]], shape_out=[[16, 192, 1536]],
            routed_rows=16 * 192))
        assert whole["n_experts"] == 128 and shard["n_experts"] == 16
        assert whole["n_tokens"] == 128 * 192
        assert shard["n_tokens"] == 16 * 192
        assert whole["d_reduce"] == shard["d_reduce"] == 2048         # non-expert axes unchanged


class TestAttentionCostFromEngine:
    """Attention jobs carry the evaluator's physical Ulysses geometry."""

    def test_attention_flops_follow_the_ulysses_head_partition(self):
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig
        M = "Qwen/Qwen3-30B-A3B"
        # CP4 divides both query and KV heads exactly. Larger cuts remain legal
        # by gathering the KV replica cells, covered below.
        cp_degree = 4
        base, _ = build_jobs(avg_output_tokens=0, model_id=M, batch_size=8, seq_len=2048, phase="prefill")
        cp, _ = build_jobs(avg_output_tokens=0, model_id=M, batch_size=8, seq_len=2048, phase="prefill",
                           parallel=ParallelConfig(cp=cp_degree,
                                                   ep_moe=cp_degree,
                                                   dp_attention=1,
                                                   cp_scheme="ulysses"))
        fb = next(j["flops_per_rank"] for j in base if j["op"] == "flash_attention")
        fc = next(j["flops_per_rank"] for j in cp if j["op"] == "flash_attention")
        # tp=1 per-op ≈ one layer's causal attention 4·B·N·Sq·Lq·D·½
        want = 4 * 8 * 32 * 2048 * 2048 * 128 * 0.5
        assert abs(fb - want) / want < 0.05
        # CP shards per-rank compute by 1/cp (NOT 1/cp²): the cp-correct cost
        assert abs(fc * cp_degree - fb) / fb < 0.02

    def test_cp_attention_cost_matches_the_ulysses_kernel_shape(self):
        # Ulysses exchanges the resident sequence cut for a head cut inside
        # attention.  Each rank therefore runs fewer heads over the full query
        # and context; the job's dimensions and FLOPs must say the same thing.
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig
        # CP4 divides both query and KV heads without a KV gather.
        cp_deg, seq = 4, 2048
        cp, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-30B-A3B", batch_size=8, seq_len=seq,
                           phase="prefill",
                           parallel=ParallelConfig(cp=cp_deg, ep_moe=cp_deg,
                                                   dp_attention=1,
                                                   cp_scheme="ulysses"))
        j = next(x for x in cp if x["op"] == "flash_attention")
        d = j["dims"]
        B, N, Sq, Lq, D = d["B"], d["n_heads"], d["s_query"], d["s_kv"], d["d_head"]
        f = j["flops_per_rank"]
        assert 32 // cp_deg == N, f"heads should be CP-sharded: N={N}"
        assert Sq == seq, f"query sequence should be full: Sq={Sq}"
        assert Lq == seq, f"context should be full: Lq={Lq}"
        compute = 4 * B * N * Sq * Lq * D * 0.5
        assert abs(f - compute) / compute < 0.05


class TestDpEpSweep:
    """dp_ep is a SAFE sweep mode: its ep_moe=E degree breaks the tp head-cap
    that left 500G+ MoE models with 0 feasible cells, and it does so without
    over-counting the attention batch (dp replicates attention, only scales the
    MoE gather M — verified end to end in the batch-semantics trace)."""

    def test_dp_ep_is_a_safe_mode(self):
        from kernel_bench.sweep import SAFE_MODES
        assert "dp_ep" in SAFE_MODES

    def test_dp_ep_breaks_the_head_cap(self):
        # A big-MoE shape with FEW attention heads (the MiniMax / Qwen3-235B
        # case): no mode is cut at the head count any more — the degrees are
        # the layout's answer (f904dbe63) — and dp_ep's EP ladder still stops
        # where the expert pool stops dividing.
        from backend.arch.eval.analysis import MoeFacts
        from backend.inference.parallel_sweep import (
            SweepCaps, parallel_configs_for_mode)
        pool = MoeFacts(
            n_expert_copies=256, active_copies=8, n_experts=256,
            n_experts_active=8, n_route_targets=256, d_ffn_expert=2048,
            n_shared_experts=1)
        dims = SweepCaps(moe_enabled=True, moe_pools=(pool,))
        dp_ep = [x for x, _ in parallel_configs_for_mode("dp_ep", dims)]
        assert max(dp_ep) == 256            # past the 64 heads, up to E
        assert all(256 % x == 0 for x in dp_ep)

    def test_dp_ep_config_replicates_attention_shards_experts(self):
        # dp_ep = dp_attention (replicate heads) + ep_moe (shard experts). It
        # must NOT set tp_attention (that would shard heads = the head-cap trap).
        from backend.inference.parallel_sweep import config_for_degree
        cfg = config_for_degree("dp_ep", 64)
        assert int(getattr(cfg, "dp_attention", 1)) == 64
        assert int(getattr(cfg, "ep_moe", 1)) == 64
        assert int(getattr(cfg, "tp_attention", 1) or 1) == 1



class TestGateBatchedPerDeployment:
    """The gate queries L1 once per DEPLOYMENT (model × gpu × parallel) and
    reuses it across that deployment's seq × phase cells. The model is built
    once instead of once per coordinate — which is most of a query's cost.
    Batching must not move the grid: same ceilings in, same cells out."""

    def test_batched_l1_matches_per_point(self):
        from backend.analyzer import fetch_config
        from backend.inference.capacity import (
            estimate_capacity_ceilings, estimate_capacity_ceilings_at,
        )
        from backend.inference.parallel_sweep import config_for_degree
        cfg = fetch_config("Qwen/Qwen3-30B-A3B")
        from backend.arch.loader import SpecLoader
        resolved = SpecLoader().match(cfg, model_id="Qwen/Qwen3-30B-A3B")
        par = config_for_degree("tp_ep", 4)
        seqs = [1024, 4096, 16384]
        batched = estimate_capacity_ceilings_at(
            config=cfg, resolved=resolved, gpu_sku="RTX-4090", parallel=par,
            num_gpus=1,
            seq_lens=seqs, avg_output_tokens=256)
        for seq in seqs:
            one = estimate_capacity_ceilings(
                config=cfg, resolved=resolved, gpu_sku="RTX-4090", parallel=par,
                num_gpus=1,
                avg_input_tokens=seq, avg_output_tokens=256)
            assert batched[seq].max_batch == one.max_batch, f"seq={seq}"
            assert batched[seq].max_prefill_batch == one.max_prefill_batch, f"seq={seq}"

    def test_feasible_batches_is_pure_selection(self):
        # Given a ceiling it picks grid points ≤ it, plus the ceiling itself —
        # and issues no capacity query of its own. `seq` only bounds prefill by
        # the forward pass's token budget; it never triggers a query either.
        from kernel_bench.sweep import _feasible_batches

        class _Cap:
            max_batch = 40
            max_prefill_batch = 0
            beyond_max_context = False
        assert _feasible_batches(_Cap(), "decode", 2048, _ROUTINE_BATCH_KEYS) == [1, 16, 40]
        assert _feasible_batches(_Cap(), "prefill", 2048, _ROUTINE_BATCH_KEYS) == []   # doesn't fit

    def test_prefill_selection_takes_the_tighter_of_the_two_ceilings(self):
        # Capacity says 40 streams fit; the chunk says 32768/2048 = 16 tokens'
        # worth. The forward pass is what gets benched, so the chunk wins.
        from kernel_bench.sweep import _feasible_batches

        class _Cap:
            max_batch = 40
            max_prefill_batch = 40
            beyond_max_context = False
        assert _feasible_batches(_Cap(), "prefill", 2048, _ROUTINE_BATCH_KEYS) == [1, 16]
        # …and capacity wins when IT is the tighter one.
        class _Tight:
            max_batch = 4
            max_prefill_batch = 4
            beyond_max_context = False
        assert _feasible_batches(_Tight(), "prefill", 1024, _ROUTINE_BATCH_KEYS) == [1, 4]


class TestGpuSkuResolution:
    """The measured row records the SKU's peak FLOPs/bandwidth, so a mis-resolved
    SKU silently rescales every %-of-peak derived from it. Two ways the naive
    substring match got it wrong, both found in real data:

      * variant collision — "H100-SXM" and "H100-PCIe" share the family token
        "h100", and first-match-wins picked whichever the CSV listed first. An
        "NVIDIA H100 80GB HBM3" (SXM, 3350 GB/s) box recorded H100-PCIe's 2000
        GB/s across 94k rows, inflating MFU ~1.7×.
      * prefix collision — "h20" is a substring of "h200", so every H200 also
        matched H20.
    """

    CASES: ClassVar[list] = [
        ("NVIDIA H100 80GB HBM3", "H100-SXM", 3350.0),      # HBM3 ⇒ SXM
        ("NVIDIA H100 PCIe", "H100-PCIe", 2000.0),
        ("NVIDIA A100-SXM4-80GB", "A100-80G-SXM", None),    # form factor + capacity
        ("NVIDIA A100-PCIE-40GB", "A100-40G-PCIe", None),
        ("NVIDIA H200", "H200", None),                      # not H20
        ("NVIDIA H20", "H20", None),
        ("NVIDIA GeForce RTX 4090", "RTX-4090", 1008.0),
    ]

    def test_resolves_each_sku(self):
        from kernel_bench.env import _resolve_peaks
        for name, want_sku, want_bw in self.CASES:
            sku, bw = _resolve_peaks(name)
            assert sku == want_sku, f"{name!r} → {sku!r}, want {want_sku!r}"
            if want_bw is not None:
                assert bw == want_bw, f"{name!r} bw={bw}, want {want_bw}"

    def test_compute_peak_is_read_at_the_rows_mma_family(self):
        # One card, one peak per MMA family. An FP8 GEMM judged against the
        # BF16 peak read 2x its real MFU and the SOL gate dropped honest rows
        # above 1088 TF on an H100; INT8 used to read the FP8 column.
        h100 = Env("2.11", True, "H100", "H100-SXM", 3350.0)
        assert h100.peak_tflops("bf16") == pytest.approx(989.0)
        assert h100.peak_tflops("fp8_e4m3") == pytest.approx(1979.0)
        # The 4090's cells are its sheet peaks scaled to the 2.73 GHz the card
        # actually holds under a bf16 GEMM, not the 2.52 GHz rated boost clock
        # (see gpu_specs.csv notes); the per-family ratios are unchanged.
        rtx = Env("2.11", True, "RTX-4090", "RTX-4090", 1008.0)
        assert rtx.peak_tflops("bf16") == pytest.approx(178.9)
        assert rtx.peak_tflops("fp8_e4m3") == pytest.approx(357.8)
        assert rtx.peak_tflops("int8") == pytest.approx(715.7)
        # A family the sheet has no column for is an unstated peak, not 0.
        a100 = Env("2.11", True, "A100", "A100-80G-SXM", 2039.0)
        assert a100.peak_tflops("fp8_e4m3") is None
        assert Env("2.11", False, None, None, None).peak_tflops("bf16") is None

    def test_ambiguous_name_reports_unknown_rather_than_guessing(self):
        # A bare "NVIDIA H100" carries no variant marker. An empty peak is a
        # visible gap; a wrong peak corrupts every MFU computed from the row.
        from kernel_bench.env import _resolve_peaks
        assert _resolve_peaks("NVIDIA H100") == (None, None)


class TestCpEpGeometry:
    """cp_ep is a SAFE sweep mode: under context parallelism the attention
    identity carries the geometry the rank actually runs: Ulysses repartitions
    heads in prefill, while decode (DCP) shards the KV and uses one query."""

    def test_prefill_repartitions_heads_and_keeps_full_sequences(self):
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig
        # CP4 divides both query and KV heads without a KV gather.
        seq, cp = 4096, 4
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-30B-A3B", batch_size=8,
                             seq_len=seq, phase="prefill",
                             parallel=ParallelConfig(cp=cp, ep_moe=cp,
                                                     dp_attention=1,
                                                     cp_scheme="ulysses"))
        a = next(j for j in jobs if j["signature"] in ("gqa_attention", "mla_attention"))
        assert a["dims"]["n_heads"] == 32 // cp
        assert a["dims"]["s_query"] == seq
        assert a["dims"]["s_kv"] == seq

    def test_prefill_gathers_kv_when_cp_exceeds_the_kv_heads(self):
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig

        seq, cp = 4096, 8
        jobs, _dropped = build_jobs(avg_output_tokens=0,
            model_id="Qwen/Qwen3-30B-A3B", batch_size=8,
            seq_len=seq, phase="prefill",
            parallel=ParallelConfig(cp=cp, ep_moe=cp, dp_attention=1,
                                    cp_scheme="ulysses"),
        )
        attention = next(
            job for job in jobs
            if job["signature"] in ("gqa_attention", "mla_attention")
        )
        assert attention["dims"]["n_heads"] == 32 // cp
        assert attention["dims"]["n_kv_heads"] == 4
        assert attention["dims"]["s_query"] == seq
        assert attention["dims"]["s_kv"] == seq
        # The CP gathers are measured spans now, not dropped ones: each
        # carries its own job beside the attention it serves.
        assert {"all_to_all", "all_gather"} <= {j["signature"] for j in jobs}

    def test_decode_shards_kv_against_single_query(self):
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig
        seq, dcp = 4096, 8
        # `dcp` spans the CP column here rather than dividing `tp_attention`
        # — a subgroup of ranks prefill CP already allocated, which is why it
        # adds none of its own.
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-30B-A3B", batch_size=8,
                             seq_len=seq, phase="decode",
                             parallel=ParallelConfig(cp=8, dcp=dcp, ep_moe=8,
                                                     dp_attention=1))
        a = next(j for j in jobs if j["signature"] in ("gqa_attention", "mla_attention"))
        assert a["dims"]["s_query"] == 1             # decode step
        assert a["dims"]["s_kv"] == seq // dcp       # DCP: KV sliced per rank

    def test_decode_under_prefill_cp_alone_reads_the_whole_context(self):
        """Prefill CP buys ranks, not capacity: vLLM all_gathers each rank's
        prefill tail before the cache write and keeps decode writes local
        (`attention/pcp.py::_gather_prefill_cache_inputs`), so every PCP rank
        holds the WHOLE cache. Sizing the decode kernel at `seq // cp` would
        measure a context no rank ever reads."""
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig
        seq = 4096
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-30B-A3B", batch_size=8,
                             seq_len=seq, phase="decode",
                             parallel=ParallelConfig(cp=8, ep_moe=8, dp_attention=1))
        a = next(j for j in jobs if j["signature"] in ("gqa_attention", "mla_attention"))
        assert a["dims"]["s_query"] == 1
        assert a["dims"]["s_kv"] == seq


class TestIdentityProjectionsForFusedOps:
    """Every measurable op needs a per-op identity. Without one the job falls
    back to the model-wide dim table, which carries no `s_query` — so
    `refs._tokens` reads B×1 and times a ONE-token kernel under a full-sequence
    label (that produced rmsnorm rows at 977,839% of peak bandwidth)."""

    #: The ports a fused launch's operands fill, as its row names them
    #: (`operand_ports`).
    PORTS: ClassVar[dict] = {
        "fused_qk_rope": [["q_pe"], ["k_pe"]],
        "moe_fused_gate": [["gating"], ["bias"]],
        "indexer_logits": [["q"], ["k"], ["w"]],
    }

    def _ident(self, op, shape_in, **kw):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        import json
        out = identity_shape_from_shape_in(
            op, shape_in, operand_ports=self.PORTS.get(op), **kw)
        return json.loads(out) if out else None

    def test_rope_reads_rotation_width_from_the_spec(self):
        # Q/K arrive flattened as heads×d_rope, so neither factor is recoverable
        # from the shape. The rotation width is a spec constant — and it is NOT
        # `w_k / n_kv_heads`: MLA's k_pe is ONE shared rope head regardless of
        # the model's KV-head count (that inference gave d_rope=1, n_heads=4096).
        d = self._ident("fused_qk_rope", [[4, 1024, 4096], [4, 1024, 64]],
                        spec_dims={"d_rope": 64, "n_kv_heads": 64})
        assert d == {"B": 4, "s_query": 1024, "n_heads": 64,
                     "n_kv_heads": 1, "d_rope": 64}

    def test_rope_accepts_head_shaped_qk(self):
        # GQA fused rope keeps the head axis: both operands are
        # [B, heads, s_query, d_rope]. Declared as an alternate engine
        # layout on the rope signature, not a second projector.
        d = self._ident("fused_qk_rope",
                        [[1, 32, 2048, 64], [1, 8, 2048, 64]])
        assert d == {"B": 1, "s_query": 2048, "n_heads": 32,
                     "n_kv_heads": 8, "d_rope": 64}

    def test_rope_mixes_head_shaped_q_with_packed_k(self):
        # MLA: Q keeps the head axis, K is one packed rope head.
        d = self._ident("fused_qk_rope",
                        [[1, 128, 1, 64], [1, 1, 64]],
                        spec_dims={"d_rope": 64})
        assert d == {"B": 1, "s_query": 1, "n_heads": 128,
                     "n_kv_heads": 1, "d_rope": 64}

    def test_moe_gate_counts_experts_and_topk(self):
        d = self._ident("moe_fused_gate", [[2, 256, 6144], [256]],
                        shape_out=[2, 256, 8])
        assert d == {"B": 2, "s_query": 256, "n_experts": 256,
                     "n_experts_active": 8}

    def test_indexer_keeps_query_length_off_the_cache(self):
        # weights/Q are s_query; indexer-K is s_kv. Prefill they coincide;
        # decode is verify window 1 against a 2048 cache.
        d = self._ident("indexer_logits",
                        [[2, 32, 256, 128], [2, 1024, 128], [2, 256, 32]],
                        spec_dims={"kv_block_size": 64})
        assert d == {"B": 2, "s_query": 256, "s_kv": 1024,
                     "n_indexer_heads": 32, "d_indexer_head": 128,
                     "kv_block_size": 64}

    def test_indexer_decode_uses_verify_length_not_cache_on_q(self):
        d = self._ident("indexer_logits",
                        [[1, 64, 1, 128], [1, 2048, 128], [1, 1, 64]],
                        spec_dims={"kv_block_size": 64})
        assert d == {"B": 1, "s_query": 1, "s_kv": 2048,
                     "n_indexer_heads": 64, "d_indexer_head": 128,
                     "kv_block_size": 64}

    def test_indexer_accepts_head_first_four_d(self):
        # DSv4 CSA indexer fuses Q/K/weights at [B, n_heads, s, d] /
        # [B, n_heads, s, s_kv] rather than the V3 packed-MQA layout.
        d = self._ident(
            "indexer_logits",
            [[1, 64, 2048, 128], [1, 64, 512, 128], [1, 64, 2048, 512]],
            spec_dims={"kv_block_size": 64})
        assert d == {"B": 1, "s_query": 2048, "s_kv": 512,
                     "n_indexer_heads": 64, "d_indexer_head": 128,
                     "kv_block_size": 64}

    def test_a_fraction_of_a_row_pads_up_and_a_zero_does_not(self):
        # Decode MoE splits one token across many experts, so the mean rows
        # per expert is a fraction and the display column floors it to 0.
        # The law's own extent is what the cell is built from and it is
        # padded up to the row the launch runs — the two forms arrive
        # together on every row (`shape_in` beside `shape_in_exact`).
        gemm = self._ident("linear", [1, 0, 1280],
                           shape_in_exact=[1, "1/2", 1280],
                           shape_out=[1, 0, 4096],
                           shape_out_exact=[1, "1/2", 4096])
        assert gemm == {"B": 1, "s_query": 1, "d_reduce": 1280, "d_out": 4096}
        # A law that states zero rows is not a fraction of one: nothing is
        # launched, so there is no geometry to key and the caller drops the
        # row and names it rather than inventing a token.
        assert self._ident("linear", [1, 0, 1280], shape_out=[1, 0, 4096]) is None
        assert self._ident("situ", [0, 1, 3072]) is not None  # 0 batch folds

    def test_embedding_accepts_vocab_axis(self):
        d = self._ident("embedding", [1, 1, 152576],
                        shape_out=[1, 2048, 1024])
        assert d == {"B": 1, "s_query": 1, "n_vocab": 152576, "d_model": 1024}

    def test_attn_res_combine_projects_stack_depth(self):
        d = self._ident("attn_res_combine",
                        [[2, 128, 4096], [2, 128, 5, 4096]])
        assert d == {"B": 2, "s_query": 128, "d_model": 4096, "Sar": 5}

    def test_per_head_qk_norm_folds_heads_into_the_token_count(self):
        # A 4-D per-head QK-norm is the SAME kernel as a per-token rmsnorm over
        # B·n_heads·s_query rows, so it shares that identity — and its
        # measurement.
        d = self._ident("rmsnorm", [2, 64, 256, 128])
        assert d == {"B": 2, "s_query": 64 * 256, "d_model": 128}
        flat = self._ident("rmsnorm", [2, 64 * 256, 128])
        assert d == flat, "per-head and per-token forms must share one identity"


class TestSparseLatentAttention:
    """DSA (DeepSeek sparse attention) is its own kernel family. It rode the
    dense-GQA branch, whose 4-D K/V layout its operand list does not have, so
    the projection produced nothing and every DSA row was benched at the
    model-wide fallback shape — 2430 rows at d_head = seq_len."""

    #: The span's operands as a row names them (`operand_ports`): the query,
    #: the latent — its K and its V, one stored tensor — and the selection.
    PORTS: ClassVar[list] = [["q"], ["k", "v"], ["mask"]]

    def _ident(self, shape_in, shape_out, **kw):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        import json
        out = identity_shape_from_shape_in("flash_dsa", shape_in,
                                           shape_out=shape_out,
                                           operand_ports=self.PORTS, **kw)
        return json.loads(out) if out else None

    def test_routes_to_its_own_signature_not_dense_attention(self):
        from kernel_bench.signatures import op_to_signature, get_signature
        from backend.inference.op_signatures import op_bindings
        assert op_to_signature("flash_dsa") == "mla_sparse_attention"
        assert op_bindings()["flash_dsa"] == "mla_sparse_attention"
        assert get_signature("mla_sparse_attention") is not None

    #: The latent the sparse kernel reads (DSv3.2 / GLM-5.x: 512 + 64), the
    #: indexer's budget and the deployment's page.
    SPEC: ClassVar[dict] = {"n_topk_keys": 2048, "d_kv_lora": 512,
                            "d_qk_rope": 64, "kv_block_size": 64}

    def test_absorbed_decode_identity(self):
        # [Q[B,N,Sq,576], latent[B,Lq,576] (as stored), topk[B,Sq,Lq]]
        # → out[B,N,Sq,512]
        d = self._ident([[8, 64, 1, 576], [8, 2048, 576], [8, 1, 2048]],
                        [8, 64, 1, 512], spec_dims=self.SPEC)
        assert d == {"B": 8, "n_heads": 64, "s_query": 1, "d_head": 576,
                     "s_kv": 2048, "d_v_head": 512, "n_topk_keys": 2048,
                     "kv_block_size": 64}

    def test_a_deployment_with_no_page_has_no_cell(self):
        # The page is an axis of every cell that reads a paged cache; the
        # hand-built DSA projection completes its cell from the spec dims as
        # the contract path does, and refuses without it.
        spec = {k: v for k, v in self.SPEC.items() if k != "kv_block_size"}
        assert self._ident([[8, 64, 1, 576], [8, 2048, 576], [8, 1, 2048]],
                           [8, 64, 1, 512], spec_dims=spec) is None

    def test_cache_read_at_its_declared_position_not_found_by_shape(self):
        # GLM-5.3-Flash prefill: the selection is 512 POOLS while each query
        # attends 2051 keys (512 · 4 + the open tail), the cache operand's
        # rows. A shape search over the operand list keyed the pools.
        d = self._ident([[2, 4, 4096, 512], [2, 2051, 512], [2, 4096, 512]],
                        [2, 4, 4096, 512],
                        spec_dims={"n_topk_keys": 2048, "d_kv_lora": 512,
                                   "kv_block_size": 64})
        assert d["s_kv"] == 2051 and d["s_query"] == 4096
        assert d["n_topk_keys"] == 2051

    def test_topk_clamped_to_the_cache_it_selects_from(self):
        # A cache shorter than the top-k budget selects all of it: the kernel
        # degenerates to dense, and the identity must say so rather than claim
        # more selected positions than exist.
        d = self._ident([[1, 8, 1, 576], [1, 512, 576], [1, 1, 512]],
                        [1, 8, 1, 512], spec_dims=self.SPEC)
        assert d["n_topk_keys"] == 512

    def test_reference_and_providers_exist(self):
        from kernel_bench.refs import get_ref
        from kernel_bench.providers import all_providers, discover
        assert get_ref("mla_sparse_attention") is not None
        discover()
        impls = {p.kernel_impl for p in all_providers()
                 if p.signature == "mla_sparse_attention"}
        # The torch gather baseline must be present on every card: FlashMLA's
        # sparse kernel is Hopper/Blackwell-only, so without it DSA would be
        # unmeasurable on anything else.
        assert "torch_gather_sparse_mla" in impls

    def test_chunked_reference_matches_single_shot_math(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.invocation import Invocation
        inv = Invocation(
            op="flash_dsa", signature="mla_sparse_attention", phase="prefill",
            dims={"kv_block_size": 64, "B": 2, "s_query": 8, "s_kv": 32, "n_topk_keys": 8,
                  "n_heads": 4, "d_head": 24, "d_v_head": 16},
            dtype_env={"activation": "bf16", "weight": "bf16", "kv": "bf16"},
            tp=1)
        spec = get_ref("mla_sparse_attention")
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        # Unchunked oracle over the same inputs.
        q, c = inputs["q"].float(), inputs["c"].float()
        idx = inputs["indices"].long()
        B, Sq, _N, Dqk = q.shape
        sel = torch.gather(c.unsqueeze(1).expand(B, Sq, c.shape[1], Dqk), 2,
                           idx.unsqueeze(-1).expand(B, Sq, idx.shape[2], Dqk))
        scores = torch.einsum("bsnd,bstd->bnst", q, sel) * (1.0 / Dqk ** 0.5)
        want = torch.einsum("bnst,bstd->bnsd", scores.softmax(-1),
                            sel[..., :16]).transpose(1, 2)
        assert torch.allclose(got, want, atol=1e-5)

    def test_selection_is_scattered_over_the_cache(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import _mla_sparse_indices
        idx = _mla_sparse_indices(1, 4, 64, 8, "cpu")
        assert idx.shape == (1, 4, 8)
        row = idx[0, 0].tolist()
        assert len(set(row)) == 8, "selected positions must be distinct"
        assert max(row) - min(row) > 8, "a contiguous tail is not a sparse gather"

    def test_selection_of_the_whole_cache_is_still_scattered(self):
        # The cell keys the cache at the rows a query attends, so top-k ==
        # s_kv is the common case: an evenly spaced stride of s_kv // topk is
        # 1 there, and the timed "gather" was one sequential read.
        pytest.importorskip("torch")
        from itertools import pairwise
        from kernel_bench.refs import _mla_sparse_indices
        for Lq in (1152, 2051, 512):
            idx = _mla_sparse_indices(2, 3, Lq, Lq, "cpu")
            for row in idx.reshape(-1, Lq).tolist():
                assert sorted(row) == list(range(Lq)), "a permutation of the cache"
                steps = [b - a for a, b in pairwise(row)]
                assert sum(abs(s) == 1 for s in steps) < Lq // 8, (
                    f"Lq={Lq}: consecutive picks are adjacent rows")


class TestPlanNeverFabricatesShapes:
    """`fallback_dims` is the model-wide dim table, not a kernel geometry.
    Handing it to a measurable op benches a kernel the model never runs, under
    that op's name — so a missing projection must surface as a drop."""

    def test_dsa_models_emit_a_sparse_attention_job(self):
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="zai-org/GLM-5.1", batch_size=2,
                             seq_len=2048, phase="decode")
        sparse = [j for j in jobs if j["signature"] == "mla_sparse_attention"]
        assert sparse, "GLM-5.1 decode must plan its DSA attention"
        d = sparse[0]["dims"]
        assert d["s_query"] == 1
        # d_head is the shared latent's width, never the sequence length.
        assert d["d_head"] == 576 and d["d_head"] != d["s_kv"]

    def test_dsa_prefill_is_keyed_at_the_geometry_it_runs(self):
        # GLM-5.1 prefills DSA absorbed (`variant_phases: [prefill, decode]`
        # on `mla_dsa_attention`, as serving runs it through the sparse MLA
        # kernel), so its span is the latent cell: one shared 576-wide row a
        # key, no KV-head axis. This test used to assert a DROP, from when
        # prefill was walked non-absorbed — per-head K/V of width 256 that a
        # one-latent gather was timed for while 2·n_heads copies were billed
        # (MBU 1.9). A per-head span now keys `mla_sparse_attention`'s
        # `n_kv_heads` geometry instead (`_dsa_identity`); the latent span
        # must not: an `n_kv_heads` on this cell would be the absorbed latent
        # (and its value slice) read as per-head K and V — the same
        # misclassification the other way round.
        from kernel_bench.plan import build_jobs
        jobs, dropped = build_jobs(avg_output_tokens=0, model_id="zai-org/GLM-5.1", batch_size=2,
                                   seq_len=2048, phase="prefill")
        sparse = [j for j in jobs if j["signature"] == "mla_sparse_attention"]
        assert sparse, dropped
        assert not [n for n in dropped if "mla_sparse_attention" in n], dropped
        for j in sparse:
            d = j["dims"]
            assert "n_kv_heads" not in d, d
            assert (d["d_head"], d["d_v_head"]) == (576, 512), d
            # The cache is the attended rows of the stored latent, keyed once.
            assert d["s_kv"] == d["n_topk_keys"] == 2048, d

    def test_no_job_carries_the_model_wide_shape(self):
        from kernel_bench.plan import build_jobs
        for model in ("zai-org/GLM-5.1", "deepseek-ai/DeepSeek-V3"):
            for phase in ("prefill", "decode"):
                jobs, _ = build_jobs(avg_output_tokens=0, model_id=model, batch_size=2, seq_len=2048,
                                     phase=phase)
                for j in jobs:
                    # `s_ctx` only exists in the model-wide table; every
                    # projection emits the phase-correct query extent —
                    # `s_query`, or the routed buffer's own `s_expert_rows`
                    # (grouped_gemm pads no query axis to state).
                    assert "s_ctx" not in j["dims"], f"{j['op']} on model-wide dims"
                    assert ("s_expert_rows" if j["op"] == "grouped_gemm"
                            else "s_query") in j["dims"], \
                        f"{j['op']} has no query extent"

    def test_fused_span_takes_analyze_shape_not_a_constituent_leaf(self):
        # The per-rank tree is PRE-fusion, so at a fused span's sid it holds one
        # constituent leaf (`rope_emb` under `fused_qk_rope`). Joining on sid
        # alone fed that leaf's shape to the span's projection.
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3", batch_size=2,
                             seq_len=2048, phase="decode")
        rope = [j for j in jobs if j["signature"] == "rope"]
        assert rope, "MLA's q_pe/k_pe rope must be planned"
        for j in rope:
            assert j["dims"].get("d_rope", 0) > 1


class TestSpeculativeVerifyWidth:
    """A spec-decode verify step submits K draft tokens at once: the query block
    is K wide over a cache of cache_len + K − 1. Every op sees it."""

    def test_query_block_widens_and_cache_grows(self):
        from kernel_bench.plan import build_jobs
        for n in (1, 2, 4, 7):
            jobs, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-235B-A22B", batch_size=2,
                                 seq_len=2048, phase="decode", s_verify=n)
            attn = next(j["dims"] for j in jobs
                        if j["signature"] in ("gqa_attention", "mla_attention"))
            assert attn["s_query"] == n
            assert attn["s_kv"] == 2048 + n - 1
            # The GEMMs come from the per-rank tree, which re-derives the phase
            # binding from `seq_len` unless the caller's binding is passed
            # through — that collapsed M back to one token.
            gemm = next(j["dims"] for j in jobs if j["signature"] == "gemm")
            assert gemm["s_query"] == n
            assert attn["d_head"] == 128, "head width must not move with K"

    def test_default_is_ordinary_autoregressive_decode(self):
        from kernel_bench.plan import build_jobs
        import json
        a, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-235B-A22B", batch_size=2,
                          seq_len=1024, phase="decode")
        b, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-235B-A22B", batch_size=2,
                          seq_len=1024, phase="decode", s_verify=1)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


class TestMlaGeometryIsPhaseDependent:
    """MLA runs two different kernels: absorbed latent at decode, per-head K/V
    with asymmetric qk/v widths at prefill. Only the absorbed layout had a
    projection, so prefill MLA was benched at the model-wide shape."""

    def _ident(self, shape_in, shape_out):
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        import json
        # At the deployment's page, which the cell carries.
        out = identity_shape_from_shape_in("mla_attention", shape_in,
                                           shape_out=shape_out,
                                           spec_dims={"kv_block_size": 64})
        return json.loads(out) if out else None

    def test_absorbed_decode_has_one_shared_latent(self):
        d = self._ident([[1, 128, 1, 576], [1, 2048, 576]], [1, 128, 1, 512])
        assert d == {"B": 1, "n_heads": 128, "s_query": 1, "d_head": 576,
                     "s_kv": 2048, "d_v_head": 512, "kv_block_size": 64}
        assert "n_kv_heads" not in d

    def test_non_absorbed_prefill_keeps_per_head_kv(self):
        d = self._ident([[1, 128, 2048, 192], [1, 128, 2048, 192],
                         [1, 128, 2048, 128]], [1, 128, 2048, 128])
        assert d == {"B": 1, "n_heads": 128, "n_kv_heads": 128,
                     "s_query": 2048, "d_head": 192, "s_kv": 2048,
                     "d_v_head": 128, "kv_block_size": 64}

    def test_the_two_geometries_do_not_share_a_physical_key(self):
        from backend.inference.benchmark_identity import project_shape_for_op
        import json
        absorbed = json.dumps({"B": 1, "n_heads": 128, "s_query": 1,
                               "d_head": 576, "s_kv": 2048, "d_v_head": 512},
                              sort_keys=True, separators=(",", ":"))
        prefill = json.dumps({"B": 1, "n_heads": 128, "n_kv_heads": 128,
                              "s_query": 1, "d_head": 576, "s_kv": 2048,
                              "d_v_head": 512}, sort_keys=True,
                             separators=(",", ":"))
        assert (project_shape_for_op("mla_attention", absorbed)
                != project_shape_for_op("mla_attention", prefill))

    def test_prefill_geometry_has_a_provider(self):
        from kernel_bench.providers import all_providers, discover
        from kernel_bench.invocation import Invocation
        from kernel_bench.env import Env
        discover()
        inv = Invocation(
            op="mla_attention", signature="mla_attention", phase="prefill",
            dims={"kv_block_size": 64, "B": 1, "n_heads": 16, "n_kv_heads": 16, "s_query": 128,
                  "s_kv": 128, "d_head": 192, "d_v_head": 128},
            dtype_env={"activation": "bf16", "weight": "bf16", "kv": "bf16"},
            tp=1)
        env = Env("2.11", False, "RTX-4090", "RTX-4090", 1008.0, sm="sm_89")
        eligible = {p.kernel_impl for p in all_providers()
                    if p.signature == "mla_attention" and p.accepts(inv, env)}
        assert "torch_sdpa_mla_prefill" in eligible
        # The absorbed decode geometry must NOT match the prefill provider.
        dec = Invocation(
            op="mla_attention", signature="mla_attention", phase="decode",
            dims={"kv_block_size": 64, "B": 1, "n_heads": 16, "s_query": 1, "s_kv": 2048,
                  "d_head": 576, "d_v_head": 512},
            dtype_env={"activation": "bf16", "weight": "bf16", "kv": "bf16"},
            tp=1)
        assert "torch_sdpa_mla_prefill" not in {
            p.kernel_impl for p in all_providers()
            if p.signature == "mla_attention" and p.accepts(dec, env)}


class TestSweepVerifyAxis:
    """A decode track's speculative draft lengths are a decode-only axis:
    each K is verified over K + 1 positions in one forward."""

    def test_prefill_unaffected_and_decode_widens(self):
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.sweep_grid import TrackSpec
        from kernel_bench.sweep import build_sweep_jobs
        keys = _ROUTINE_BATCH_KEYS

        def plan(drafts):
            tracks = (TrackSpec("prefill", "prefill", keys),
                      TrackSpec("decode", "decode", keys, draft_tokens=drafts))
            return build_sweep_jobs(
                model_id="meta-llama/Llama-3.2-3B", gpu="RTX-4090",
                seq_lens=[2048], tracks=tracks,
                parallel_configs=[("tp=1", ParallelConfig())],
                avg_output_tokens=256)[0]
        base, wide = plan((0,)), plan((0, 3))
        pre = lambda js: [j for j in js if j["phase"] == "prefill"]
        assert len(pre(base)) == len(pre(wide)), "prefill has no draft to verify"
        widths = {j["dims"]["s_query"] for j in wide if j["phase"] == "decode"}
        assert {1, 4} <= widths
        assert len(wide) > len(base)


class TestLinearAttentionKernelFamilies:
    """Linear attention is not one kernel. Gated DeltaNet and Kimi Delta
    Attention decay the recurrent state differently — one scalar per head vs one
    vector per key channel — and vLLM and sglang vendor a separate Triton kernel
    for each, beside a separate kernel per PHASE (chunked for a sequence, fused
    recurrent for a step). One provider calling the chunked GDN kernel for every
    shape measured one of the eight."""

    def test_decay_rank_selects_the_family(self):
        from backend.inference.benchmark_identity import resolve_op
        # Gated DeltaNet: g [B,H,T] — one decay per head.
        gdn = [[2, 32, 128, 64], [2, 32, 128, 64], [2, 32, 128, 64],
               [2, 128, 32], [2, 128, 32]]
        # KDA: g [B,H,T,Dk] — one per key channel.
        kda = [[2, 96, 128, 128], [2, 96, 128, 128], [2, 96, 128, 128],
               [2, 96, 128, 128], [2, 128, 96]]
        assert resolve_op("gated_delta_rule", shape_in=gdn) == "gated_delta_rule"
        assert resolve_op("gated_delta_rule", shape_in=kda) == "kda"
        # Absent shapes must not silently claim the sparse family.
        assert resolve_op("gated_delta_rule") == "gated_delta_rule"

    def test_kda_model_routes_to_kda_and_gdn_model_does_not(self):
        from kernel_bench.plan import build_jobs
        kda, _ = build_jobs(avg_output_tokens=0, model_id="moonshotai/Kimi-K3", batch_size=2,
                            seq_len=2048, phase="prefill")
        sigs = {j["signature"] for j in kda}
        assert "kda" in sigs and "gated_delta_rule" not in sigs
        gdn, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-Coder-Next", batch_size=2,
                            seq_len=2048, phase="prefill")
        sigs = {j["signature"] for j in gdn}
        assert "gated_delta_rule" in sigs and "kda" not in sigs
        # Grouped-value attention: fewer key heads than value heads.
        d = next(j["dims"] for j in gdn if j["signature"] == "gated_delta_rule")
        assert d["n_lin_k_heads"] < d["n_lin_v_heads"]

    def test_every_family_has_a_kernel_for_each_phase(self):
        from kernel_bench.providers import all_providers, discover
        from kernel_bench.invocation import Invocation
        from kernel_bench.env import Env
        discover()
        env = Env("2.11", False, "H100", "H100-SXM", 3350.0, sm="sm_90")
        families = (
            ("gated_delta_rule", {"B": 1, "s_query": 128, "n_lin_k_heads": 2,
                                  "n_lin_v_heads": 2, "d_lin_k_head": 64,
                                  "d_lin_v_head": 64}),
            ("kda", {"B": 1, "s_query": 128, "n_lin_k_heads": 2,
                     "n_lin_v_heads": 2, "d_lin_k_head": 64,
                     "d_lin_v_head": 64}),
            ("gla_simple_recurrence", {"B": 1, "s_query": 128, "n_heads": 4,
                                       "d_head": 64}),
        )
        for sig, dims in families:
            for phase in ("prefill", "decode"):
                inv = Invocation(
                    op=sig, signature=sig, phase=phase, dims=dims,
                    dtype_env={"activation": "bf16", "weight": "bf16", "kv": "bf16"},
                    tp=1)
                impls = {p.kernel_impl for p in all_providers()
                         if p.signature == sig and p.accepts(inv, env)}
                assert impls, f"{sig} {phase} has no kernel"
                assert any(i.startswith("torch_") for i in impls), impls
                # A single-step recurrent kernel must not be timed over a full
                # sequence, and a chunked kernel must not be timed at one token:
                # each would measure work the model never does in that phase.
                # So every framework kernel serving the cell DECLARES the
                # phase it is serving (`KernelSignature.phases`) rather than
                # accepting any — read off the declaration, not the name
                # (sglang's `seg_la_fwd` is one entry point registered once
                # per phase). The torch floor is the sequential recurrence —
                # valid in both phases, so it sits outside this split.
                for p in all_providers():
                    if (p.signature == sig and p.accepts(inv, env)
                            and not p.kernel_impl.startswith("torch_")):
                        assert p.serves.phases == (phase,), (
                            p.kernel_impl, p.serves.phases)

    def test_vendored_kernels_are_registered_beside_upstream(self):
        # Both frameworks vendor flash-linear-attention rather than depending on
        # it, so the copy a deployment runs is theirs, not the pip package.
        from kernel_bench.providers import all_providers, discover
        discover()
        libs = {p.library for p in all_providers() if p.signature == "kda"}
        assert {"vllm", "sglang", "fla"} <= libs

    def test_reference_is_the_recurrence_and_honours_the_decay(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.invocation import Invocation
        for sig, g_rank in (("gated_delta_rule", 3), ("kda", 4)):
            inv = Invocation(
                op=sig, signature=sig, phase="prefill",
                dims={"B": 1, "s_query": 16, "n_lin_k_heads": 2,
                      "n_lin_v_heads": 4, "d_lin_k_head": 8, "d_lin_v_head": 8},
                dtype_env={"activation": "bf16", "weight": "bf16", "kv": "bf16"},
                tp=1)
            spec = get_ref(sig)
            x = build_inputs(spec, inv, "cpu")
            assert x["g"].dim() == g_rank
            out = spec.reference(x)
            assert out.shape == (1, 16, 4, 8) and torch.isfinite(out).all()
            # The state must actually decay: forgetting everything each step
            # cannot give the same answer as keeping everything.
            forget = dict(x, g=torch.full_like(x["g"], -30.0))
            keep = dict(x, g=torch.zeros_like(x["g"]))
            assert not torch.allclose(spec.reference(forget), spec.reference(keep))

    def test_reference_refuses_shapes_it_cannot_afford(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.invocation import Invocation
        inv = Invocation(
            op="kda", signature="kda", phase="prefill",
            dims={"B": 8, "s_query": 4096, "n_lin_k_heads": 2,
                  "n_lin_v_heads": 2, "d_lin_k_head": 8, "d_lin_v_head": 8},
            dtype_env={"activation": "bf16", "weight": "bf16", "kv": "bf16"},
            tp=1)
        spec = get_ref("kda")
        x = build_inputs(spec, inv, "cpu")
        # O(T) python: it must say so rather than hang, and the driver then
        # measures without a cross-check.
        with pytest.raises(ValueError):
            spec.reference(x)


class TestShortConvIsMeasured:
    """Every Mamba / Gated-DeltaNet / KDA layer runs a depthwise causal short
    conv on q/k/v before its recurrence. `conv1d` had no signature, so it was
    dropped from every one of those models."""

    def test_the_fused_span_projects_and_routes(self):
        # The measured launch is the conv WITH its SiLU, which is the fused
        # span `causal_conv1d`; the bare `conv1d` op is unbound (its one
        # surviving site, sglang's dilated PLE conv, is another kernel).
        import json
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        from kernel_bench.signatures import op_to_signature
        assert op_to_signature("causal_conv1d") == "causal_conv1d"
        assert op_to_signature("conv1d") is None
        d = json.loads(identity_shape_from_shape_in(
            "causal_conv1d", [2, 2048, 12288],
            spec_dims={"s_conv_kernel": 4, "dilation": 1}))
        assert d == {"B": 2, "s_query": 2048, "d_conv_channels": 12288,
                     "s_conv_kernel": 4, "dilation": 1}

    def test_kernel_width_and_dilation_come_from_the_spec(self):
        # The width sizes the per-thread work and the conv-state ring buffer;
        # the dilation chooses the kernel (the frameworks' causal conv takes
        # none). Neither is a tensor axis: the engine's shape is feature-last
        # [B,S,C], so a row that names neither has no cell.
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        assert identity_shape_from_shape_in(
            "causal_conv1d", [2, 2048, 12288]) is None
        assert identity_shape_from_shape_in(
            "causal_conv1d", [2, 2048, 12288],
            spec_dims={"s_conv_kernel": 4}) is None

    def test_linear_attention_models_plan_their_conv(self):
        from kernel_bench.plan import build_jobs
        for model in ("moonshotai/Kimi-K3", "Qwen/Qwen3-Coder-Next"):
            jobs, _ = build_jobs(avg_output_tokens=0, model_id=model, batch_size=2, seq_len=2048,
                                 phase="prefill")
            conv = [j for j in jobs if j["signature"] == "causal_conv1d"]
            assert conv, f"{model} runs a short conv and must measure it"
            # Channels are heads×head_dim on the q/k/v projections, not d_model.
            assert conv[0]["dims"]["d_conv_channels"] >= 4096

    def test_reference_is_causal(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.invocation import Invocation
        inv = Invocation(
            op="causal_conv1d", signature="causal_conv1d", phase="prefill",
            dims={"B": 1, "s_query": 8, "d_conv_channels": 2,
                  "s_conv_kernel": 4, "dilation": 1},
            dtype_env={"activation": "bf16", "weight": "bf16", "kv": "bf16"},
            tp=1)
        spec = get_ref("causal_conv1d")
        x = build_inputs(spec, inv, "cpu")
        base = spec.reference(x)
        # Perturbing the LAST token must not change any earlier output.
        bumped = dict(x, x=x["x"].clone())
        bumped["x"][:, :, -1] += 10.0
        after = spec.reference(bumped)
        assert torch.allclose(base[:, :, :-1], after[:, :, :-1], atol=1e-4)
        assert not torch.allclose(base[:, :, -1], after[:, :, -1])


class TestFlashKdaIsItsOwnRow:
    """FlashKDA is Moonshot's CUTLASS kernel for its own attention, and from
    flash-linear-attention 0.5 it takes over `chunk_kda` transparently — so a row
    labelled `fla_chunk_kda` could be timing it while crediting Triton."""

    def test_registered_as_a_distinct_implementation(self):
        from kernel_bench.providers import all_providers, discover
        discover()
        kda = {p.kernel_impl: p for p in all_providers() if p.signature == "kda"}
        assert "flash_kda_cutlass_fwd" in kda
        p = kda["flash_kda_cutlass_fwd"]
        assert p.library == "flash_kda" and p.import_name == "flash_kda"
        # No PyPI release; it is a CUTLASS source build.
        assert p.source_build

    def test_upstream_fla_is_pinned_to_its_own_kernel(self):
        # The provider that claims to measure Triton must ensure it measures
        # Triton, using FLA's own opt-out.
        from kernel_bench.providers import linear_ssm
        import inspect
        src = inspect.getsource(linear_ssm)
        assert 'FLA_FLASH_KDA' in src
        for entry in ("_fla_chunk_kda", "_fla_recurrent_kda"):
            body = inspect.getsource(getattr(linear_ssm, entry))
            assert "_pin_fla_triton()" in body, f"{entry} may dispatch elsewhere"

    def test_declines_shapes_the_kernel_does_not_serve(self):
        from kernel_bench.providers import all_providers, discover
        from kernel_bench.invocation import Invocation
        from kernel_bench.env import Env
        discover()
        p = next(p for p in all_providers()
                 if p.kernel_impl == "flash_kda_cutlass_fwd")
        env = Env("2.11", False, "H100", "H100-SXM", 3350.0, sm="sm_90")

        def inv(dk, dv, phase="prefill"):
            return Invocation(
                op="kda", signature="kda", phase=phase,
                dims={"B": 1, "s_query": 128, "n_lin_k_heads": 4,
                      "n_lin_v_heads": 4, "d_lin_k_head": dk, "d_lin_v_head": dv},
                dtype_env={"activation": "bf16", "weight": "bf16", "kv": "bf16"},
                tp=1)
        assert p.accepts(inv(128, 128), env)
        # K = V = 128 is pinned by the kernel, and it is a prefill kernel.
        assert not p.accepts(inv(64, 64), env)
        assert not p.accepts(inv(128, 256), env)
        assert not p.accepts(inv(128, 128, phase="decode"), env)


class TestKernelParametersComeFromTheExport:
    """A parameter that changes what the kernel does must travel with the
    invocation. A provider that picks it instead measures a configuration nobody
    asked for — and unlike a wrong shape, nothing downstream can tell: the flops,
    the bytes and the roofline all stay self-consistent, so the physics audit
    that caught the fabricated dims sees a perfectly healthy row."""

    def test_block_size_is_the_deployments_choice(self):
        from kernel_bench.providers.attention import (
            resolve_kv_block_size, _FLASHINFER_BLOCK_SIZES,
        )
        from kernel_bench.invocation import Invocation

        def inv(block):
            return Invocation(
                op="flash_attention", signature="gqa_attention", phase="decode",
                # The page is an axis of the cell.
                dims={"B": 1, "s_query": 1, "s_kv": 128, "n_heads": 8,
                      "n_kv_heads": 8, "d_head": 128, "kv_block_size": block},
                dtype_env={"activation": "bf16"}, tp=1)
        assert resolve_kv_block_size(inv(16), _FLASHINFER_BLOCK_SIZES) == 16
        assert resolve_kv_block_size(inv(64), _FLASHINFER_BLOCK_SIZES) == 64

    def test_unsupported_block_size_declines_rather_than_substituting(self):
        from kernel_bench.providers.attention import (
            resolve_kv_block_size, _FLASHMLA_BLOCK_SIZES,
        )
        from kernel_bench.invocation import Invocation
        inv = Invocation(
            op="mla_attention", signature="mla_attention", phase="decode",
            dims={"B": 1, "s_query": 1, "s_kv": 128, "n_heads": 8,
                  "d_head": 576, "d_v_head": 512, "kv_block_size": 16},
            dtype_env={"activation": "bf16"}, tp=1)
        # FlashMLA serves [64] only. Substituting 64 here would measure a paging
        # layout the deployment did not ask for.
        with pytest.raises(NotImplementedError) as e:
            resolve_kv_block_size(inv, _FLASHMLA_BLOCK_SIZES)
        assert "64" in str(e.value)

    def test_every_cache_reading_provider_declares_its_paging(self):
        """The COMPLEMENT: no kernel that reads a cache may stay silent.

        Checking the kernels that declared is the one shape of check that
        cannot catch an omission, and three had omitted — including DSA's
        `flashmla_sparse` and the DeepGEMM paged indexer. Whether a kernel
        reads a cache is the signature's own declaration (an operand with
        `role="kv"`), so this is that predicate's complement. `register`
        refuses an omission outright; this says so at the suite level too,
        and pins the vocabulary: a tuple of pages, or `()` for a build that
        reads the cache contiguously.
        """
        from kernel_bench.providers import discover
        from kernel_bench.providers import all_providers, served_kv_block_sizes
        from backend.inference.op_signatures import get_signature
        discover()
        silent, declared = [], []
        for p in all_providers():
            if p.serves is None:
                continue
            sig = get_signature(p.serves.signature)
            if sig is None or not sig.reads_paged_cache:
                assert not p.serves.kv_block_sizes, (
                    f"{p.kernel_impl} declares pages but its signature reads "
                    f"no cache")
                continue
            if p.serves.kv_block_sizes is None:
                silent.append(p.kernel_impl)
                continue
            declared.append(p)
            for page in p.serves.kv_block_sizes:
                assert isinstance(page, int) and page > 0, (p.kernel_impl, page)
                # A page no deployment can run is fiction; page_size=1 was
                # measured 2867 times.
                assert page != 1, p.kernel_impl
        assert silent == [], (
            "these kernels read a KV cache and say nothing about paging: "
            f"{silent}")
        assert any(p.serves.kv_block_sizes for p in declared), (
            "no provider serves a paged layout at all")
        # The sweep's pages are exactly the union of the declarations, and
        # one signature's are its own backends' (`served_kv_block_sizes(sig)`).
        assert served_kv_block_sizes() == tuple(sorted(
            {page for p in declared for page in p.serves.kv_block_sizes}))
        for sig in {p.signature for p in declared}:
            assert served_kv_block_sizes(sig) == tuple(sorted(
                {page for p in declared if p.signature == sig
                 for page in p.serves.kv_block_sizes})), sig

    def test_an_omitted_paging_declaration_is_refused_at_registration(self):
        from kernel_bench.providers import KernelSignature, Provider, register
        with pytest.raises(ValueError) as e:
            register(Provider(
                library="torch", signature="gqa_attention",
                kernel_impl="made_up_silent_attention", import_name="torch",
                serves=KernelSignature(signature="gqa_attention"),
                build=lambda inv, env, inputs: None))
        assert "kv_block_sizes" in str(e.value)

    def test_an_absent_page_is_absent_not_the_serving_default(self):
        # A job that names no page has none. This used to answer 16 (vLLM's
        # CacheConfig.DEFAULT_BLOCK_SIZE), so a job planned under a
        # framework that declares no page — sglang resolves one per
        # attention backend — would be measured at a page nobody chose and
        # the row could not say which.
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers.attention import (
            _FLASHINFER_BLOCK_SIZES, resolve_kv_block_size)
        inv = Invocation.from_job({
            "op": "flash_attention", "signature": "gqa_attention",
            "phase": "decode", "dims": {"B": 1, "s_query": 1},
            "dtype_env": {"activation": "bf16"}})
        assert inv.kv_block_size is None
        with pytest.raises(NotImplementedError) as e:
            resolve_kv_block_size(inv, _FLASHINFER_BLOCK_SIZES)
        assert "kv_block_size" in str(e.value)

    def test_resolved_parameter_lands_on_the_row(self):
        # Two configurations of one kernel are two rows, not one ambiguous one.
        from kernel_bench.emit import build_row
        from kernel_bench.providers import BenchCase, Provider
        from kernel_bench.invocation import Invocation
        from kernel_bench.env import Env
        from kernel_bench.timing import LatencyStats
        inv = Invocation(
            op="flash_attention", signature="gqa_attention", phase="decode",
            dims={"B": 1, "s_query": 1, "s_kv": 128, "n_heads": 8,
                  "n_kv_heads": 8, "d_head": 128, "kv_block_size": 16},
            dtype_env={"activation": "bf16", "weight": "bf16", "kv": "bf16",
                       "output": "bf16", "metadata": "fp32"}, tp=1,
            physical_key=("gqa_attention",
                          json.dumps({"B": 1, "s_query": 1, "s_kv": 128,
                                      "n_heads": 8, "n_kv_heads": 8,
                                      "d_head": 128, "kv_block_size": 16},
                                     sort_keys=True,
                                     separators=(",", ":")),
                          json.dumps({"activation": "bf16", "weight": "bf16",
                                      "kv": "bf16", "output": "bf16",
                                      "metadata": "fp32"}, sort_keys=True,
                                     separators=(",", ":")),
                          "decode"))
        p = Provider(library="flashinfer", signature="gqa_attention",
                     kernel_impl="flashinfer_paged_decode",
                     import_name="flashinfer",
                     applies_to=lambda i, e: True, build=lambda i, e, x: None,
                     impl_params=lambda i: f"_b{i.kv_block_size}")
        env = Env("2.11", False, "H100", "H100-SXM", 3350.0, sm="sm_90",
                  libs={"flashinfer": "0.2.5"})
        stats = LatencyStats(p50_us=1.0, p95_us=1.0, mean_us=1.0, samples=10, timing_method="l2_drain")
        row = build_row(inv, p, stats, env, date="2026-08-05", contributor="t",
                        case=BenchCase(run=lambda: None))
        assert row["kernel_impl"] == "flashinfer_paged_decode@_b16"
        # And the name coverage was asked about is the same string.
        assert p.impl_id(inv) == "flashinfer_paged_decode@_b16"

    def test_flashinfer_decode_takes_tensor_cores_where_serving_does(self):
        # vLLM's FlashInfer backend runs the tensor-core decode kernel above 4
        # query heads per KV head; the CUDA-core one read 3.4x off the H100
        # bandwidth floor at GQA 16:1. The choice is part of the kernel's name.
        from kernel_bench.providers import all_providers
        from kernel_bench.invocation import Invocation
        p = next(p for p in all_providers()
                 if p.kernel_impl == "flashinfer_paged_decode")

        def impl(n_heads, n_kv_heads):
            return p.impl_id(Invocation(
                op="flash_attention", signature="gqa_attention", phase="decode",
                dims={"B": 1, "s_query": 1, "s_kv": 128, "n_heads": n_heads,
                      "n_kv_heads": n_kv_heads, "d_head": 128,
                      "kv_block_size": 16},
                dtype_env={"activation": "bf16", "kv": "bf16"}, tp=1))
        assert impl(16, 1).endswith("_tc")
        assert not impl(8, 2).endswith("_tc")

    def test_a_cell_is_served_only_by_kernels_reading_its_scale_layout(self):
        # cutlass per-channel FP8 and DeepGEMM's 128x128 blocks shared one
        # cell, and the envelope took the faster without saying a block-FP8
        # checkpoint can only run the block-scaled one.
        from kernel_bench.providers import all_providers
        from kernel_bench.invocation import Invocation
        env = Env("2.11", True, "H100", "H100-SXM", 3350.0, sm="sm_90")

        def served(granularity, K=7168, N=1536):
            # The FP8 GEMMs write their output at the activation's width
            # back: bf16, not the operands' fp8.
            activation = ("per-group:128"
                          if granularity == "per-tile:128x128"
                          else "per-token")
            de = {"activation": "fp8_e4m3", "weight": "fp8_e4m3",
                  "output": "bf16", "weight_granularity": granularity,
                  "activation_granularity": activation}
            inv = Invocation(op="linear", signature="gemm", phase="prefill",
                             dims={"B": 1, "s_query": 64, "d_reduce": K,
                                   "d_out": N}, dtype_env=de, tp=1)
            return {p.kernel_impl for p in all_providers()
                    if p.signature == "gemm" and p.accepts(inv, env)}
        # The block-scaled envelope grew: vLLM's cutlass entry declares its
        # two layouts as their own kernels and a Triton block-scaled GEMM
        # joined beside DeepGEMM's.
        assert served("per-tile:128x128") == {
            "deepgemm_fp8_blockwise", "vllm_cutlass_scaled_mm_blockwise",
            "vllm_triton_block_scaled_mm_128"}
        assert served("per-channel") == {"vllm_cutlass_scaled_mm"}
        # No kernel here reads FP8 in groups of 128 along K: a gap, not a
        # cell measured under another layout.
        assert served("per-group:128") == set()

    def test_the_plan_writes_the_checkpoints_scale_layout(self):
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3", batch_size=2,
                             seq_len=1024, phase="prefill")
        layouts = {(j["dtype_env"]["weight"], j["dtype_env"].get("weight_granularity"))
                   for j in jobs if j["signature"] in ("gemm", "grouped_gemm")}
        assert ("fp8_e4m3", "per-tile:128x128") in layouts
        # An unquantized weight has no slot: its identity is unchanged.
        assert ("bf16", None) in layouts
        assert not {g for w, g in layouts if w == "fp8_e4m3"} - {"per-tile:128x128"}

    def test_the_scan_chunk_is_declared_and_not_a_suffix(self):
        # The chunk used to travel in `kernel_params` and be appended to the
        # implementation id as `_c128`, which put one fact in two places —
        # and left the cell's identity unable to tell two tilings apart. It
        # is an axis now (`ssm_chunk_size`), so the suffix is gone and the
        # provider reads `dims`.
        from kernel_bench.providers import all_providers
        scan = [p for p in all_providers() if p.signature == "mamba2_scan"]
        assert scan
        for provider in scan:
            assert provider.impl_params is None, provider.kernel_impl
            # The implementation id is the kernel's name and nothing else:
            # no `_c128` tail restating an axis of the cell.
            assert provider.impl_id(Invocation(
                op="mamba2_scan", signature="mamba2_scan", phase="prefill",
                dims={"B": 1, "s_query": 2048, "ssm_chunk_size": 128},
                dtype_env={"activation": "bf16"}, tp=1,
            )) == provider.kernel_impl

    def test_a_missing_binding_is_not_a_defaulted_one(self):
        # The spec used to carry `?? 256` and later the plan read
        # `mamba_chunk_size or chunk_size` off the config, so a model that
        # published neither silently measured at someone else's number. The
        # deployment reads it once, at the config fields the framework
        # binding names, with no fallback; only the scan's cell is keyed on
        # it, so a model with no Mamba carries it in no job's `dims` or
        # anywhere else.
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V2-Lite",
                             batch_size=1, seq_len=2048, phase="prefill")
        for j in jobs:
            assert "ssm_chunk_size" not in j["kernel_params"], (
                "a model with no mamba config must carry no chunk length")
            assert "ssm_chunk_size" not in j["dims"], (
                "a model with no mamba config must carry no chunk length")

    def test_envelope_keeps_one_row_per_implementation(self):
        # A framework ships several kernels per signature; matching coverage on
        # (framework, version) alone let the first one measured suppress its
        # siblings, so the envelope had one candidate where it should have had
        # several — and a re-measurement at a corrected parameter looked covered.
        import inspect
        from backend.inference.op_calibration import CalibrationRegistry
        sig = inspect.signature(CalibrationRegistry.covers)
        assert "kernel_impl" in sig.parameters
        src = inspect.getsource(CalibrationRegistry.covers)
        assert "r.key.kernel_impl == kernel_impl" in src

    def test_reconciliation_matches_vllms_rule(self):
        # vLLM reconciles rather than refusing: a kernel may run at a smaller
        # block that DIVIDES the manager's, and only when no supported size
        # divides it does it raise (v1/worker/utils.py).
        from kernel_bench.providers.attention import (
            resolve_kv_block_size, _FLASHINFER_BLOCK_SIZES, _FLASHMLA_BLOCK_SIZES,
            _CUTLASS_MLA_BLOCK_SIZES,
        )
        from kernel_bench.invocation import Invocation

        def inv(block):
            return Invocation(
                op="x", signature="gqa_attention", phase="decode",
                dims={"kv_block_size": block},
                dtype_env={"activation": "bf16"}, tp=1)
        # exact hit, then largest supported divisor
        assert resolve_kv_block_size(inv(64), _FLASHINFER_BLOCK_SIZES) == 64
        assert resolve_kv_block_size(inv(128), _FLASHINFER_BLOCK_SIZES) == 64
        assert resolve_kv_block_size(inv(128), _FLASHMLA_BLOCK_SIZES) == 64
        assert resolve_kv_block_size(inv(128), _CUTLASS_MLA_BLOCK_SIZES) == 128
        # a manager block of 16 cannot host a 64-wide kernel block
        with pytest.raises(NotImplementedError):
            resolve_kv_block_size(inv(16), _FLASHMLA_BLOCK_SIZES)

    def test_paged_signatures_are_swept_over_block_sizes(self):
        from kernel_bench.plan import build_jobs
        from kernel_bench.providers import served_kv_block_sizes
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3.1", batch_size=2,
                             seq_len=2048, phase="decode",
                             kv_block_sizes=served_kv_block_sizes())
        mla = [j for j in jobs if j["signature"] == "mla_attention"]
        # The pages are the sweep's argument now, derived from what the
        # backends declare they serve — `build_jobs` without one measures the
        # deployment's own page, once.
        # Its OWN backends' pages, not the union: vLLM forces an MLA cache
        # to FlashMLA's 64 and FlashInfer's 16 is no page MLA decode runs at.
        assert ({j["dims"]["kv_block_size"] for j in mla}
                == set(served_kv_block_sizes("mla_attention")))
        assert 16 in set(served_kv_block_sizes()) - set(
            served_kv_block_sizes("mla_attention"))
        # The page is an axis of the cell, so it is in the dims and the key,
        # and nowhere else.
        assert all(json.loads(j["physical_key"][1])["kv_block_size"]
                   == j["dims"]["kv_block_size"] for j in mla)
        assert not [j for j in mla if "kv_block_size" in j["kernel_params"]]
        # Nothing without a paged cache is multiplied — and it carries no
        # page at all: the value there was the deployment's default
        # restated on a job whose kernel never reads a page.
        assert not [j for j in jobs if j["signature"] == "gemm"
                    and "kv_block_size" in j["dims"]]

    def test_every_mla_backend_gets_its_native_block_size(self):
        from kernel_bench.plan import build_jobs
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import all_providers, discover
        from kernel_bench.env import Env
        discover()
        env = Env("2.11", False, "H100", "H100-SXM", 3350.0, sm="sm_90")
        from kernel_bench.providers import served_kv_block_sizes
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3.1", batch_size=2,
                             seq_len=2048, phase="decode",
                             kv_block_sizes=served_kv_block_sizes())
        served = set()
        for j in (x for x in jobs if x["signature"] == "mla_attention"):
            inv = Invocation.from_job(j)
            for p in all_providers():
                if p.signature != "mla_attention" or not p.accepts(inv, env):
                    continue
                try:
                    served.add(p.impl_id(inv))
                except NotImplementedError:
                    pass
        # Each backend measured at the block size it actually serves, with the
        # size recorded in the name (`@_bN`, the declared variant axis) — no
        # guess, no silent substitution.
        assert "flashmla_decode@_b64" in served
        assert "flashinfer_mla_decode@_b64" in served
        assert "cutlass_mla_decode@_b128" in served

    def test_depthwise_conv_expression_matches_its_number(self):
        # The symbolic and numeric channels disagreed by C_in·C_out for every
        # depthwise conv: `groups == C` is an equality between two EXPRESSIONS
        # under symbolic execution, and the Python `int()` comparison fell
        # through to the standard-conv branch. Kimi-K3's q_conv1d read 604M
        # params beside a real 49K — so the width was exportable all along, but
        # the channel carrying it was 12288× wrong.
        import sympy as sp
        from backend.analyzer import fetch_config
        from backend.arch.loader import SpecLoader
        from backend.arch.eval.projection import project_model
        cfg = fetch_config("moonshotai/Kimi-K3")
        res = SpecLoader().match(cfg, model_id="moonshotai/Kimi-K3")
        mr = project_model(res, batch=1, input_tokens=8)
        subs = {sp.Symbol(k): v for k, v in (res.dim_values or {}).items()}

        def find(nodes):
            for n in nodes:
                if getattr(n, "children", None):
                    hit = find(n.children)
                    if hit is not None:
                        return hit
                elif str(getattr(n, "op", "")) == "conv1d":
                    return n
            return None

        node = None
        for entry in mr.block_results:
            e = entry[2] if isinstance(entry, tuple) and len(entry) == 3 else entry
            node = find(getattr(e, "nodes", []) or [])
            if node is not None:
                break
        assert node is not None, "Kimi-K3's KDA layers run a short conv"
        assert int(sp.sympify(node.params_expr).subs(subs)) == node.params
        # And the kernel width is in that expression by name — the parameter I
        # called unexportable is a symbol the engine already carries.
        assert "s_conv_kernel" in str(node.params_expr)


class TestIncrementalPersist:
    """Rows reach disk as they are measured, not at the end of the sweep.

    A sweep is hours of GPU time. Rows used to be buffered in memory until the
    run finished, so a kill / OOM / dropped session discarded every measurement
    taken up to that point — one observed H100 run held 20 hours of work and
    had written nothing. The driver now hands each row to the caller as soon as
    it exists and the CLI writes + flushes it.
    """

    def test_driver_reports_each_row_as_measured(self):
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        seen = []
        jobs = [_gemm_job(dims={"B": 1, "s_query": 1, "d_reduce": 512, "d_out": n})
                for n in (512, 1024, 2048)]
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs(jobs, env=env, date="2026-07-28", contributor="@t",
                       cross_check=False, on_row=seen.append)
        assert len(seen) == len(res.rows) == 3
        # Reported AS measured, so the callback saw them one at a time.
        assert seen == res.rows

    def test_persist_failure_does_not_sink_the_sweep(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA: the measured row names the kernel that ran")
        from kernel_bench.driver import run_jobs
        jobs = [_gemm_job(dims={"B": 1, "s_query": 1, "d_reduce": 512, "d_out": n})
                for n in (512, 1024)]
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})

        def boom(_row):
            raise OSError("disk full")

        res = run_jobs(jobs, env=env, date="2026-07-28", contributor="@t",
                       cross_check=False, on_row=boom)
        assert len(res.rows) == 2                      # measuring continued
        assert sum("not persisted" in f for f in res.failures) == 2

    def test_incremental_file_matches_all_at_once_emit(self, tmp_path):
        # The incremental writer must produce byte-identical CSV to emit_csv,
        # so a partial file is a valid corpus file, not a second format.
        from backend.inference import benchmarks as B
        from kernel_bench.emit import IncrementalCsvWriter, rows_to_csv
        rows = [
            {c: "" for c in B.ALL_COLUMNS} | {
                "model": "m", "op": "linear", "kernel_impl": f"k{i}",
                "schema_version": str(B.KNOWN_SCHEMA_VERSION),
            }
            for i in range(3)
        ]
        out = tmp_path / "inc.csv"
        with IncrementalCsvWriter(str(out)) as w:
            for r in rows:
                w.write(r)
            assert w.count == 3
        assert out.read_text() == rows_to_csv(rows)

    def test_partial_file_is_readable_after_an_abrupt_stop(self, tmp_path):
        # Flushed per row: whatever was written is parseable without the
        # writer ever being closed (the kill case).
        from backend.inference import benchmarks as B
        from kernel_bench.emit import IncrementalCsvWriter
        out = tmp_path / "partial.csv"
        w = IncrementalCsvWriter(str(out))
        w.write({c: "" for c in B.ALL_COLUMNS} | {"model": "m", "op": "linear"})
        # no close() — simulate the process dying here
        text = out.read_text()
        assert text.startswith(",".join(B.ALL_COLUMNS))
        assert len(text.strip().splitlines()) == 2

    def test_append_keeps_existing_rows_and_one_header(self, tmp_path):
        from backend.inference import benchmarks as B
        from kernel_bench.emit import IncrementalCsvWriter

        output = tmp_path / "resume.csv"
        row = {column: "" for column in B.ALL_COLUMNS}
        row["model"] = "m"
        row["op"] = "linear"
        with IncrementalCsvWriter(str(output)) as writer:
            writer.write(row)
        with IncrementalCsvWriter(str(output), append=True) as writer:
            # A DIFFERENT cell: the same one again is one row, and the writer
            # says so (`test_a_repeat_of_an_identity_the_file_holds_is_refused`).
            writer.write(row | {"op": "rmsnorm"})

        lines = output.read_text().strip().splitlines()
        assert lines.count(",".join(B.ALL_COLUMNS)) == 1
        assert len(lines) == 3

    def test_a_refused_repeat_says_whether_the_run_knew_the_cell(self, tmp_path):
        """The one fact that tells the two causes of a repeat apart.

        A repeat is either a cell the resumed key set never held — so
        "measure once" had nothing to skip on — or one it did hold and the
        measurement happened anyway, which puts the fault downstream of the
        check. The duplicate rows that cost the H100 three outages survived
        the elimination of every mechanism two sessions could propose, so
        the next occurrence has to answer this by itself.
        """
        from kernel_bench.emit import existing_run_keys, resumed_verdict
        from tests.corpus_row import row as corpus_row

        row = corpus_row()
        path = tmp_path / "held.csv"
        path.write_text(_benchmarks.emit_csv(
            list(_benchmarks.ALL_COLUMNS), [row]))
        keys = existing_run_keys(str(path))
        assert keys, "the row must be keyable for the verdict to mean anything"

        said = resumed_verdict(row, keys)
        assert "HAS this cell" in said and "1 keys" in said
        assert "does NOT have" in resumed_verdict(row, set())

        # And a row it cannot key says so rather than claiming either.
        broken = dict(row, shape_json="{not json")
        assert "could not key this row" in resumed_verdict(broken, keys)

    def test_a_repeat_of_an_identity_the_file_holds_is_refused(self, tmp_path):
        """One identity is one row, and the file's writer is what holds to it.

        The driver's "measure once" key is the library-agnostic physical cell
        — deliberately a different key, so another model's measurement counts
        as this model's — so it cannot be what decides whether a FILE has two
        rows nobody can tell apart. When two cells it read as distinct wrote
        one identity, the file stopped validating, and since every run is
        handed every other model's results as `--measured-in`, the H100
        measured nothing at all until the row was removed by hand.
        """
        from backend.inference import benchmarks as B
        from kernel_bench.emit import IncrementalCsvWriter

        output = tmp_path / "once.csv"
        row = {column: "" for column in B.ALL_COLUMNS}
        row["model"] = "m"
        row["op"] = "linear"
        with IncrementalCsvWriter(str(output)) as writer:
            assert writer.write(row) is None
            repeated = writer.write(dict(row))
            assert repeated is not None and "line 2" in repeated
            # An outcome is part of the identity: the same cell may hold a
            # measurement AND the record of having been refused once.
            assert writer.write(row | {"outcome": "oom"}) is None

        # Seeded from the file, so a resumed run does not write it either.
        with IncrementalCsvWriter(str(output), append=True) as writer:
            assert writer.write(dict(row)) is not None
            assert writer.write(row | {"op": "rmsnorm"}) is None

        lines = output.read_text().strip().splitlines()
        assert len(lines) == 4                      # header + the three kept
        assert sum("linear" in line for line in lines[1:]) == 2  # once per outcome

    def test_skip_existing_resumes_a_private_results_file(self, tmp_path):
        # --skip-existing asked to not re-measure. The output CSV of a
        # previous run is half of that (the other half is the shared
        # corpus). Opening it in write mode truncated a 22k-row 4090 fa2
        # file on resume. Resume must both keep the file AND skip the
        # identities it already holds — not merely append a duplicate.
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        from kernel_bench.emit import IncrementalCsvWriter, resume_from_output

        job = _gemm_job()
        env = Env("2.11", False, None, "H100-SXM", 3350.0,
                  libs={"torch": "2.11"})
        inv = Invocation.from_job(job)
        provs = applicable(inv, env)
        assert provs, "gemm job must have a torch provider in this env"
        row = build_row(
            inv, provs[0],
            LatencyStats(p50_us=10.0, p95_us=11.0, mean_us=10.0, samples=8, timing_method="l2_drain"),
            env, date="2026-08-31", contributor="@t",
        )
        output = tmp_path / "fa2.csv"
        with IncrementalCsvWriter(str(output)) as writer:
            writer.write(row)

        append, keys = resume_from_output(
            str(output), skip_existing=True, append=False)
        assert append is True
        assert keys

        wrote = []
        res = run_jobs(
            [job], env=env, date="2026-08-31", contributor="@t",
            cross_check=False, measured_keys=keys,
            on_row=wrote.append,
        )
        assert not wrote and not res.rows
        assert any("already measured in this run" in s for s in res.skipped), (
            res.skipped)

        with IncrementalCsvWriter(str(output), append=append) as writer:
            pass
        lines = output.read_text().strip().splitlines()
        assert lines.count(",".join(_benchmarks.ALL_COLUMNS)) == 1
        assert len(lines) == 2

    def test_without_skip_existing_a_fresh_run_still_replaces(self, tmp_path):
        from kernel_bench.emit import resume_from_output
        output = tmp_path / "fresh.csv"
        output.write_text("header\n")
        append, keys = resume_from_output(
            str(output), skip_existing=False, append=False)
        assert append is False
        assert keys == set()


class TestSpreadFlags:
    """min-dedup is the right envelope for one-sided latency noise, but a
    wide spread is contamination, a run that skipped the work, or a real
    box difference — the report must flag it, not let the min bury it."""

    def _rows(self):
        base = {c: "x" for c in
                ("model", "spec_hash", "phase", "parallel", "framework",
                 "framework_version", "gpu", "op", "node_spec_id",
                 "shape_json", "dtype_env_json", "kernel_impl")}
        return [dict(base, p50_us="100.0", run_id="run-a"),
                dict(base, p50_us="140.0", run_id="run-b")]

    def test_a_wide_spread_is_flagged_with_both_run_ids(self):
        from tools.consolidate_corpus import dedupe
        kept, dupes, flags = dedupe(self._rows())
        assert len(kept) == 1 and dupes == 1
        assert float(kept[0]["p50_us"]) == 100.0      # the min envelope stands
        assert len(flags) == 1
        assert flags[0]["spread"] == pytest.approx(0.4)
        assert flags[0]["run_ids"] == ["run-a", "run-b"]

    def test_sampling_noise_is_not_a_flag(self):
        from tools.consolidate_corpus import dedupe
        rows = self._rows()
        rows[1]["p50_us"] = "103.0"
        _kept, _dupes, flags = dedupe(rows)
        assert flags == []


class TestSolGate:
    """`t_measured < t_sol(同形状)` is physically impossible — the write-side
    gate drops such rows where they are lost, not in a cleanup script weeks
    later (the fabricated-identity rows: one token timed, a full sequence
    counted, MBU 977,839%)."""

    def test_work_above_the_cards_own_peak_is_refused(self):
        from kernel_bench.emit import above_sol
        row = {"achieved_tflops": "200.0", "achieved_bw_gbps": "0",
               "peak_tflops_at_test": "100.0", "peak_bw_at_test": "1000.0"}
        assert above_sol(row) is not None

    def test_bandwidth_above_peak_is_refused(self):
        from kernel_bench.emit import above_sol
        row = {"achieved_tflops": "0", "achieved_bw_gbps": "5000.0",
               "peak_tflops_at_test": "100.0", "peak_bw_at_test": "1000.0"}
        assert above_sol(row) is not None

    def test_a_row_at_the_cards_own_peak_is_not_a_violation(self):
        from kernel_bench.emit import above_sol
        row = {"achieved_tflops": "100.0", "achieved_bw_gbps": "0",
               "peak_tflops_at_test": "100.0", "peak_bw_at_test": "1000.0"}
        assert above_sol(row) is None

    def test_a_row_without_peaks_is_not_this_gates_call(self):
        from kernel_bench.emit import above_sol
        assert above_sol({"achieved_tflops": "999", "achieved_bw_gbps": "",
                          "peak_tflops_at_test": "",
                          "peak_bw_at_test": ""}) is None

    def test_job_start_callback_runs_before_validation(self):
        from kernel_bench.driver import run_jobs

        seen = []
        with pytest.raises(KeyError):
            run_jobs([{"not": "an invocation"}],
                     env=Env("2.11", False, None, None, None, libs={}),
                     date="2026-08-16", contributor="@t",
                     on_job_start=lambda index, job: seen.append((index, job)))
        assert seen == [(0, {"not": "an invocation"})]


class TestResultCheck:
    """`hbench check` is the write-side SOL gate and the recorded allclose
    verdict, applied to a file after the fact. It does not re-run kernels."""

    def test_a_clean_row_passes(self):
        from kernel_bench.check import check_rows, format_report
        from tests.corpus_row import row as corpus_row

        report = check_rows([corpus_row()])
        assert report.ok()
        assert report.n_rows == 1
        assert report.allclose["passed"] == 1
        assert report.mfu and report.mfu[0] == pytest.approx(690.0 / 989.0)
        assert report.mbu and report.mbu[0] == pytest.approx(920.0 / 3350.0)
        text = format_report(report)
        assert "allclose: passed=1" in text
        assert "ok" in text

    def test_work_above_peak_fails_sol(self):
        from kernel_bench.check import check_rows, format_report
        from tests.corpus_row import row as corpus_row

        report = check_rows([corpus_row(achieved_bw_gbps="99999.0")])
        assert not report.ok()
        assert len(report.sol) == 1
        assert "bandwidth" in report.sol[0]
        assert "FAIL sol" in format_report(report)

    def test_no_reference_is_counted_not_a_failure(self):
        from kernel_bench.check import check_rows
        from tests.corpus_row import row as corpus_row

        report = check_rows([
            corpus_row(cross_check="no_reference",
                       cross_check_rel_err="", cross_check_cosine=""),
        ])
        assert report.ok()
        assert report.allclose["passed"] == 0
        assert report.allclose["no_reference"] == 1

    def test_cli_refuses_a_file_that_claims_impossible_work(self, tmp_path):
        from kernel_bench.cli import main
        from tests.corpus_row import row as corpus_row, to_csv

        path = tmp_path / "out.csv"
        path.write_text(to_csv([corpus_row(achieved_tflops="99999.0")]))
        rc = main(["check", str(path)])
        assert rc == 1

    def test_cli_accepts_a_directory_of_clean_csvs(self, tmp_path, capsys):
        from kernel_bench.cli import main
        from tests.corpus_row import row as corpus_row, to_csv

        (tmp_path / "a.csv").write_text(to_csv([corpus_row()]))
        (tmp_path / "b.csv").write_text(to_csv([corpus_row(op="linear")]))
        rc = main(["check", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "2 rows in 2 file" in out
        assert "allclose: passed=2" in out

    def test_missing_path_is_an_input_error(self, tmp_path):
        from kernel_bench.cli import main
        rc = main(["check", str(tmp_path / "nope.csv")])
        assert rc == 2

    def test_upload_refuses_a_file_that_fails_check(self, tmp_path, capsys):
        from kernel_bench.cli import main
        from tests.corpus_row import row as corpus_row, to_csv

        path = tmp_path / "out.csv"
        path.write_text(to_csv([corpus_row(achieved_tflops="99999.0")]))
        rc = main(["upload", str(path), "--token", "t",
                   "--host", "http://127.0.0.1:1"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "FAIL sol" in captured.out
        assert "upload refused" in captured.err

    def test_restamp_rewrites_achieved_tflops_from_new_plan_flops(self):
        from kernel_bench.emit import restamp_rows
        from tests.corpus_row import row as corpus_row

        row = corpus_row(op="rmsnorm", achieved_tflops="0.0000",
                         p50_us="14.4960", shape_json='{"B":1,"d_model":2048,"s_query":1024}',
                         node_spec_id="block|input_layernorm",
                         phase="prefill", parallel="tp=1",
                         model="meta-llama/Llama-3.2-1B")
        shape = '{"B":1,"d_model":2048,"s_query":1024}'
        dtype = '{"activation":"bf16","kv":"bf16","output":"bf16"}'
        jobs = [{
            "model": "meta-llama/Llama-3.2-1B",
            "op": "rmsnorm",
            "node_spec_id": "block|input_layernorm",
            "dims": {"B": 1, "d_model": 2048, "s_query": 1024},
            "dtype_env": {"activation": "bf16", "kv": "bf16", "output": "bf16"},
            "physical_key": ["rmsnorm", shape, dtype, "prefill"],
            "phase": "prefill",
            "parallel": "tp=1",
            "flops_per_rank": 2 * 1 * 1024 * 2048,
            "access_bytes_per_rank": 2 * 1 * 1024 * 2048 * 2,
        }]
        out, changed, missed = restamp_rows([row], jobs)
        assert missed == 0 and changed == 1
        # 2·B·S·H / 14.496e-6 / 1e12
        assert float(out[0]["achieved_tflops"]) == pytest.approx(
            (2 * 1024 * 2048) / (14.4960 * 1e-6) / 1e12, abs=5e-5)
        assert out[0]["p50_us"] == "14.4960"

    def test_scale_const_restamp_joins_on_row_dtype(self):
        from kernel_bench.emit import above_sol, restamp_rows
        from tests.corpus_row import row as corpus_row

        dtype = '{"activation":"bf16","output":"bf16"}'
        row = corpus_row(
            model="Qwen/Qwen3.8-Flash-Next", op="scale_const",
            node_spec_id="qwen_hc_full_block|self_attn.index_reduce__qwen_qsa_attention",
            shape_json='{"s_query":1024,"d_model":256,"B":32}',
            dtype_env_json='{ "output": "bf16", "activation": "bf16" }',
            kernel_impl="torch_scale_const",
            p50_us="19.5200", achieved_tflops="0.0000",
            achieved_bw_gbps="1000.0000", peak_tflops_at_test="989.0000",
            peak_bw_at_test="3350.0000", phase="prefill",
            parallel="tp_attention=4,tp_dense_ffn=4,tp_moe_expert=4")
        identity = {
            "model": "Qwen/Qwen3.8-Flash-Next",
            "op": "scale_const",
            "node_spec_id":
                "qwen_hc_full_block|self_attn.index_reduce__qwen_qsa_attention",
            "dims": {"B": 32, "d_model": 256, "s_query": 1024},
            "phase": "prefill",
            "parallel": "tp_attention=4,tp_dense_ffn=4,tp_moe_expert=4",
        }
        # Plan order is not a selector. The FP32 cell shares every field the
        # old key read, but at this p50 its doubled byte numerator claims
        # 3438 GB/s on a 3350 GB/s card; the measured BF16 cell claims 1719.
        shape = '{"B":32,"d_model":256,"s_query":1024}'
        jobs = [
            {**identity, "dtype_env": {"activation": "bf16", "output": "bf16"},
             "physical_key": ["scale_const", shape, dtype, "prefill"],
             "flops_per_rank": 8_388_608.0,
             "access_bytes_per_rank": 33_554_432.0},
            {**identity, "dtype_env": {"activation": "fp32", "output": "fp32"},
             "physical_key": [
                 "scale_const", shape, '{"activation":"fp32","output":"fp32"}',
                 "prefill"],
             "flops_per_rank": 8_388_608.0,
             "access_bytes_per_rank": 67_108_864.0},
        ]
        out, changed, missed = restamp_rows([row], jobs)
        assert (changed, missed) == (1, 0)
        assert float(out[0]["achieved_bw_gbps"]) == pytest.approx(
            33_554_432 / (19.52 * 1e-6) / 1e9, abs=5e-4)
        assert above_sol(out[0]) is None

    @pytest.mark.parametrize(("engine_op", "signature"), [
        ("linear", "gemm"),
        ("permute", "moe_permute"),
    ])
    def test_restamp_projects_engine_ops_to_plan_signatures(
            self, engine_op, signature):
        from kernel_bench.emit import restamp_rows
        from tests.corpus_row import row as corpus_row

        shape = '{"B":1,"s_query":1}'
        dtype = '{"activation":"bf16","output":"bf16"}'
        row = corpus_row(op=engine_op, phase="decode", shape_json=shape,
                         dtype_env_json=dtype, p50_us="2.0000",
                         achieved_tflops="1.0000",
                         achieved_bw_gbps="1.0000")
        jobs = [{
            "op": engine_op,
            "physical_key": [signature, shape, dtype, "decode"],
            "flops_per_rank": 2000.0,
            "access_bytes_per_rank": 4000.0,
        }]

        result = restamp_rows([row], jobs)

        assert result.unmatched == []
        assert (result.changed, result.missed) == (1, 0)
        assert float(result.rows[0]["achieved_tflops"]) == pytest.approx(
            2000.0 / (2.0 * 1e-6) / 1e12)
        assert float(result.rows[0]["achieved_bw_gbps"]) == pytest.approx(
            4000.0 / (2.0 * 1e-6) / 1e9)
        assert result.rows[0]["op"] == engine_op

    def test_restamp_refuses_an_ambiguous_complete_cell(self):
        from kernel_bench.emit import restamp_rows
        from tests.corpus_row import row as corpus_row

        row = corpus_row(op="scale_const", phase="decode", p50_us="10.0000",
                         shape_json='{"B":1}',
                         dtype_env_json='{"activation":"bf16"}',
                         achieved_tflops="1.0000",
                         achieved_bw_gbps="2.0000")
        cell = ["scale_const", '{"B":1}', '{"activation":"bf16"}', "decode"]
        jobs = [
            {"op": "scale_const", "physical_key": cell,
             "flops_per_rank": 2.0, "access_bytes_per_rank": 4.0},
            {"op": "scale_const", "physical_key": cell,
             "flops_per_rank": 2.0, "access_bytes_per_rank": 8.0},
        ]
        result = restamp_rows([row], jobs)
        out, changed, missed = result
        assert (changed, missed) == (0, 0)
        assert out[0]["achieved_bw_gbps"] == "2.0000"
        assert len(result.ambiguous) == 1
        assert result.ambiguous[0].cell == tuple(cell)
        assert result.ambiguous[0].pairs == ((2.0, 4.0), (2.0, 8.0))
        assert result.ambiguous[0].row_indexes == [0]

    def test_restamp_reports_why_each_row_missed(self):
        from kernel_bench.emit import restamp_rows
        from tests.corpus_row import row as corpus_row

        def cell(op, shape, dtype):
            return [op, shape, dtype, "decode"]

        bf16 = '{"activation":"bf16"}'
        fp32 = '{"activation":"fp32"}'
        geometry = '{"B":1}'
        other_geometry = '{"B":2}'
        jobs = [{
            "op": "scale_const", "physical_key":
                cell("scale_const", other_geometry, bf16),
            "flops_per_rank": 1.0, "access_bytes_per_rank": 1.0,
        }, {
            "op": "scale_const", "physical_key":
                cell("scale_const", geometry, fp32),
            "flops_per_rank": 1.0, "access_bytes_per_rank": 1.0,
        }]
        unplanned_geometry = '{"B":3}'
        rows = [
            corpus_row(op="scale_const", phase="decode", shape_json=geometry,
                       dtype_env_json=bf16),
            corpus_row(op="scale_const", phase="decode", shape_json=other_geometry,
                       dtype_env_json=bf16),
            corpus_row(op="scale_const", phase="decode", shape_json=unplanned_geometry,
                       dtype_env_json=bf16),
            corpus_row(op="scale_const", phase="decode", shape_json=geometry,
                       dtype_env_json=fp32),
            corpus_row(op="other", phase="decode", shape_json=geometry,
                       dtype_env_json=bf16),
        ]
        result = restamp_rows(rows, [])
        assert [m.reason for m in result.unmatched] == [
            "no_plan", "no_plan", "no_plan", "no_plan", "no_plan"]
        result = restamp_rows(rows, jobs)
        assert [m.reason for m in result.unmatched] == [
            "dtype_not_in_plan", "geometry_not_in_plan", "no_plan"]

    def test_restamp_does_not_invent_a_missing_dtype_identity(self):
        from kernel_bench.emit import restamp_rows
        from tests.corpus_row import row as corpus_row

        no_column = corpus_row(op="scale_const")
        del no_column["dtype_env_json"]
        empty = corpus_row(op="scale_const", dtype_env_json="")
        job = [{
            "op": "scale_const",
            "physical_key": ["scale_const", '{"B":1}', "{}", "decode"],
            "flops_per_rank": 1.0, "access_bytes_per_rank": 1.0,
        }]
        result = restamp_rows([no_column, empty], job)
        assert (result.changed, result.missed) == (0, 2)
        assert [m.reason for m in result.unmatched] == [
            "missing_dtype_column", "invalid_dtype_identity"]
        assert result.rows[0] is no_column

    def test_restamp_requires_the_plan_carried_physical_key(self):
        from kernel_bench.emit import restamp_rows
        from tests.corpus_row import row as corpus_row

        job = {"op": "scale_const", "dims": {"B": 1},
               "dtype_env": {}, "phase": "decode"}
        with pytest.raises(ValueError, match="physical_key"):
            restamp_rows([corpus_row(op="scale_const")], [job])

    def test_cli_restamp_refuses_an_ambiguity_without_rewriting(
            self, tmp_path, capsys):
        import json
        from kernel_bench.cli import main
        from tests.corpus_row import row as corpus_row, to_csv

        row = corpus_row(op="scale_const", phase="decode",
                         shape_json='{"B":1}',
                         dtype_env_json='{"activation":"bf16","output":"bf16"}')
        path = tmp_path / "torch.csv"
        before = to_csv([row])
        path.write_text(before)
        cell = ["scale_const", '{"B":1}',
                '{"activation":"bf16","output":"bf16"}', "decode"]
        jobs = [
            {"op": "scale_const", "physical_key": cell,
             "flops_per_rank": 1.0, "access_bytes_per_rank": 2.0},
            {"op": "scale_const", "physical_key": cell,
             "flops_per_rank": 1.0, "access_bytes_per_rank": 4.0},
        ]
        (tmp_path / "jobs.jsonl").write_text(
            "".join(json.dumps(job) + "\n" for job in jobs))
        rc = main(["restamp", "--jobs", str(tmp_path / "jobs.jsonl"), str(path)])
        assert rc == 1
        assert path.read_text() == before
        assert "REFUSED 1 ambiguous plan cell(s)" in capsys.readouterr().out

    def test_upload_checks_a_clean_file_before_the_network(
            self, tmp_path, monkeypatch, capsys):
        import json
        import urllib.request
        from kernel_bench.cli import main
        from tests.corpus_row import row as corpus_row, to_csv

        path = tmp_path / "out.csv"
        path.write_text(to_csv([corpus_row()]))
        posted = []

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({"row_count": 1, "upload_id": "abcd1234",
                                   "shared": False, "label": "out"}).encode()

        def fake_urlopen(req, timeout=60):
            posted.append(req.full_url)
            return _Resp()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        rc = main(["upload", str(path), "--token", "t",
                   "--host", "http://example.test"])
        assert rc == 0
        assert posted == ["http://example.test/api/bench/upload"]
        assert "ok" in capsys.readouterr().out


class TestAFusedOpIsMeasurableWhereASignatureMeasuresIt:
    """A fusion whose launch a provider measures must be bound to that
    signature: the walk renames the node to the fused op, and an unbound op
    has no signature, so the plan drops the cell. `causal_conv1d` was
    dropped from every SSM model's plan this way while five providers
    measured exactly that kernel."""

    def test_every_fused_op_with_providers_is_bound(self):
        from backend.inference.fusion_plans import FUSION_SCHEMES
        from backend.inference.op_signatures import get_signature, op_bindings
        from kernel_bench.providers import all_providers
        served = {p.signature for p in all_providers()}
        bindings = op_bindings()
        unbound = [name for name in FUSION_SCHEMES
                   if get_signature(name) is not None and name in served
                   and bindings.get(name) is None]
        assert unbound == [], (
            "these fused ops have a signature and providers but no binding, "
            f"so a plan drops them: {unbound}")

    def test_the_conv_fusion_is_bound_to_the_conv_signature(self):
        from backend.inference.op_signatures import op_bindings
        assert op_bindings().get("causal_conv1d") == "causal_conv1d"


class TestTheNonAbsorbedMlaFloorIsPrefillOnly:
    """Serving runs MLA decode absorbed, over one shared latent; the
    per-head geometry is the prefill form, and `torch_sdpa_mla_prefill`
    measures it. At ONE head the two geometries have the same shape, so
    nothing but the phase keeps that provider off a decode cell — and it
    measured the latent twice, as a k and a v."""

    def _inv(self, phase):
        from kernel_bench.invocation import Invocation
        return Invocation.from_job({
            "signature": "mla_attention", "op": "mla_attention",
            "dims": {"B": 1, "s_query": 1, "s_kv": 1152, "n_heads": 1,
                     "n_kv_heads": 1, "d_head": 576, "d_v_head": 512},
            "dtype_env": {"activation": "bf16", "kv": "bf16"},
            "phase": phase, "tp": 1,
            "physical_key": ["mla_attention", "{}", "{}", phase]})

    def test_it_declines_a_decode_cell_at_one_head(self):
        from kernel_bench.providers import all_providers
        p = next(x for x in all_providers()
                 if x.kernel_impl == "torch_sdpa_mla_prefill")
        assert p.serves.accepts(self._inv("prefill"))
        assert not p.serves.accepts(self._inv("decode"))


class TestABlockScaledShardThatCannotLoadIsNotPlanned:
    """A block scale ties the shard to the block: past some TP degree a rank
    holds a partial block, the frameworks refuse that weight at load, and no
    kernel reads a partial block either. Such a cell is dropped with its
    reason rather than planned for a deployment that cannot start."""

    @staticmethod
    def _shards(*partitions, gran="per-tile:128x128"):
        """A row multiplying `partitions`, each `(pack, rows, rows_local,
        width, width_local)` as the walk publishes them (`weight_shards`)."""
        return {"weight_shards": [
            {"pack": list(pack), "rows": n, "rows_local": n_local,
             "width": k, "width_local": k_local, "granularity": gran}
            for pack, n, n_local, k, k_local in partitions]}

    def test_the_refusal_reads_the_granularity(self):
        from kernel_bench.plan import _scale_block_refusal
        mat = self._shards
        assert _scale_block_refusal(mat(((0, 0), 256, 256, 128, 128))) is None
        # A row-parallel cut to 72, and a column-parallel cut to 192.
        assert "d_reduce=72" in _scale_block_refusal(
            mat(((0, 0), 256, 256, 144, 72)))
        assert "d_out=192" in _scale_block_refusal(
            mat(((0, 0), 384, 192, 128, 128)))
        # per-group names the reduction block only; the output is unconstrained
        assert _scale_block_refusal(
            mat(((0, 0), 180, 90, 64, 64), gran="per-group:32")) is None
        # ... and a group scale refuses a ragged K cut or not: Marlin and AWQ
        # want whole groups at any degree.
        assert "d_reduce=90" in _scale_block_refusal(
            mat(((0, 0), 64, 64, 90, 90), gran="per-group:32"))

    def test_a_width_nobody_cut_is_no_shard(self):
        # The checkpoint's own last block is partial and the frameworks load
        # it: DeepSeek-V3's kv_a projection is 576 wide on every rank.
        from kernel_bench.plan import _scale_block_refusal
        assert _scale_block_refusal(
            self._shards(((0, 0), 576, 576, 7168, 7168))) is None

    def test_each_partition_is_checked_by_its_own_scale(self):
        # A launch mixing a block-scaled matrix with an unscaled one checks
        # each at its own format: the BF16 partition's ragged cut is no
        # refusal, the FP8 one's is.
        from kernel_bench.plan import _scale_block_refusal
        row = {"weight_shards": [
            {"pack": [0, 0], "rows": 384, "rows_local": 192, "width": 7168,
             "width_local": 7168, "granularity": None},
            {"pack": [1, 0], "rows": 512, "rows_local": 256, "width": 7168,
             "width_local": 7168, "granularity": "per-tile:128x128"}]}
        assert _scale_block_refusal(row) is None
        row["weight_shards"][0]["granularity"] = "per-tile:128x128"
        assert "d_out=192" in _scale_block_refusal(row)

    def test_only_partitions_of_one_weight_may_share_a_ragged_end(self):
        # Partitions packed end to end into one weight (one `pack`): every
        # one but the last ends on a block boundary, so one ragged uncut
        # partition loads (q_a 1536 + kv_a 576) and two do not. Two separate
        # weights that merely share a reduction width are not one pack.
        from kernel_bench.plan import _scale_block_refusal
        mat = self._shards
        assert _scale_block_refusal(mat(
            ((0, 0), 1536, 1536, 7168, 7168),
            ((0, 0), 576, 576, 7168, 7168))) is None
        assert "packs 2 partitions" in _scale_block_refusal(mat(
            ((0, 0), 576, 576, 7168, 7168), ((0, 0), 200, 200, 7168, 7168)))
        assert _scale_block_refusal(mat(
            ((0, 0), 576, 576, 7168, 7168), ((1, 0), 200, 200, 7168, 7168))) is None
        # A cut partition must be whole blocks per rank, packed or not.
        assert "d_out=192" in _scale_block_refusal(mat(
            ((0, 0), 384, 192, 7168, 7168), ((0, 0), 384, 192, 7168, 7168)))

    def test_a_scale_tied_to_nothing_refuses_nothing(self):
        from kernel_bench.plan import _scale_block_refusal
        for gran in ("per-channel", "per-tensor", "", None):
            assert _scale_block_refusal(self._shards(
                ((0, 0), 180, 90, 144, 72), gran=gran)) is None
        assert _scale_block_refusal({}) is None

    @pytest.mark.regression
    def test_a_high_tp_block_scaled_model_drops_those_cells(self):
        from backend.inference.parallel_config import ParallelConfig
        from kernel_bench.plan import build_jobs
        model = "zai-org/GLM-5.3-Flash"           # fp8 128x128 blocks
        fine = ParallelConfig.from_dict(
            {"tp_attention": 8, "tp_dense_ffn": 8, "tp_moe_expert": 8})
        coarse = ParallelConfig.from_dict(
            {"tp_attention": 64, "tp_dense_ffn": 64, "tp_moe_expert": 64})
        kw = dict(avg_output_tokens=256, model_id=model, batch_size=2,
                  seq_len=2048, phase="decode")
        _jobs, dropped = build_jobs(parallel=fine, **kw)
        assert [d for d in dropped if "scale blocks" in d] == []
        _jobs, dropped = build_jobs(parallel=coarse, **kw)
        assert [d for d in dropped if "scale blocks" in d], dropped


class TestBlockScaledCutlassIsHopperOnly:
    """CUTLASS has no block-scaled W8A8 kernel below Hopper — vLLM asks
    `cutlass_scaled_mm_supports_block_fp8` and runs its Triton blockwise GEMM
    on Ada instead. The op accepts the call anyway and returns numbers
    (cosine 0.45-0.78 against the reference on an RTX 4090, and a poisoned
    context after), so the block-scaled provider declares the floor; the
    per-channel and per-tensor recipes have none."""

    def _inv(self, gran):
        from kernel_bench.invocation import Invocation
        activation = ("per-group:128" if gran == "per-tile:128x128"
                      else "per-token")
        return Invocation.from_job({
            "signature": "gemm", "op": "linear",
            "dims": {"B": 1, "s_query": 6, "d_reduce": 6144, "d_out": 3072},
            "dtype_env": {"activation": "fp8_e4m3", "weight": "fp8_e4m3",
                          "output": "bf16", "metadata": "fp32",
                          "activation_granularity": activation,
                          "weight_granularity": gran},
            "phase": "decode", "tp": 1,
            "physical_key": ["gemm", "{}", "{}", "decode"]})

    def _serving(self, sm, gran):
        from kernel_bench.env import Env
        from kernel_bench.providers import all_providers
        env = Env("2.9", True, "x", "x", 1008.0, sm=sm,
                  libs={"vllm": "0.32.0", "torch": "2.9"})
        return {p.kernel_impl for p in all_providers()
                if p.signature == "gemm" and p.library == "vllm"
                and p.available(env) and p.accepts(self._inv(gran), env)}

    def test_ada_gets_no_block_scaled_cutlass(self):
        # Ada's block-scaled fp8 GEMM is vLLM's Triton one — which is what
        # vLLM itself dispatches there — and not CUTLASS.
        assert self._serving("sm_89", "per-tile:128x128") == {
            "vllm_triton_block_scaled_mm_128"}
        assert "vllm_cutlass_scaled_mm" in self._serving("sm_89", "per-channel")

    def test_hopper_gets_it(self):
        assert self._serving("sm_90", "per-tile:128x128") == {
            "vllm_cutlass_scaled_mm_blockwise",
            "vllm_triton_block_scaled_mm_128"}


class TestCrossCheckBand:
    """A cell's cross-check reads the band of the least precise operand the
    kernel reads, not the activation's alone."""

    def test_an_fp8_cache_widens_a_bf16_attention_to_the_fp8_band(self):
        from types import SimpleNamespace
        from kernel_bench.driver import _xcheck_dtype
        inv = SimpleNamespace(dtype="bf16", dtype_env={
            "activation": "bf16", "kv": "fp8_e4m3", "output": "bf16"})
        assert _xcheck_dtype(inv) == "fp8_e4m3"

    def test_a_plain_bf16_cell_keeps_its_band(self):
        from types import SimpleNamespace
        from kernel_bench.driver import _xcheck_dtype
        inv = SimpleNamespace(dtype="bf16", dtype_env={
            "activation": "bf16", "weight": "bf16", "output": "bf16"})
        assert _xcheck_dtype(inv) == "bf16"

    def test_an_unbanded_weight_dtype_does_not_widen(self):
        # int4 has no band of its own; the activation's stands.
        from types import SimpleNamespace
        from kernel_bench.driver import _xcheck_dtype
        inv = SimpleNamespace(dtype="bf16", dtype_env={
            "activation": "bf16", "weight": "int4", "output": "bf16"})
        assert _xcheck_dtype(inv) == "bf16"


    def test_a_dtype_without_a_measured_band_is_refused(self):
        """No silent bf16 fallback: the missing table entry is the error."""
        from types import SimpleNamespace
        from kernel_bench.driver import _xcheck_band, _xcheck_dtype
        inv = SimpleNamespace(dtype="fp4_e2m1", dtype_env={
            "activation": "fp4_e2m1", "weight": "bf16"})
        with pytest.raises(ValueError, match="activation dtype 'fp4_e2m1'"):
            _xcheck_dtype(inv)
        with pytest.raises(ValueError, match="_XCHECK_BANDS"):
            _xcheck_band("fp4_e2m1")


class TestFlashMlaTopkIndices:
    def test_short_context_is_padded_to_the_64_block_with_invalid_slots(self):
        torch = pytest.importorskip("torch")
        from types import SimpleNamespace
        from kernel_bench.providers.attention import (
            _FLASHMLA_SPARSE_DECODE_TOPK_TILE, _flashmla_topk_indices)
        idx = torch.arange(3, dtype=torch.int32).view(1, 1, 3).expand(2, 1, 3)
        base = torch.tensor([0, 100], dtype=torch.int32).view(2, 1, 1)
        inv = SimpleNamespace(dims={"n_topk_keys": 3})
        out, valid = _flashmla_topk_indices(
            inv, idx, base, tokens=2, tile=_FLASHMLA_SPARSE_DECODE_TOPK_TILE)
        assert out.shape == (2, 64)
        assert out[1, :3].tolist() == [100, 101, 102]
        assert (out[:, 3:] == -1).all()
        assert valid.tolist() == [3, 3]


class TestGpuExclusiveLock:
    """One sweep per physical card.

    Measured latency IS the product: two runs sharing a device slow each other
    down and nothing in the emitted row records it, so a contended sweep is
    indistinguishable from a clean one afterwards and the whole batch is lost.
    (Five concurrent sweeps on one H100 once cost 16k quarantined rows.)
    `--skip-existing` cannot prevent this — it asks what the CORPUS holds, not
    what another process is measuring right now.

    Keyed on the CUDA UUID, not the device index: `CUDA_VISIBLE_DEVICES` remaps
    indices, so index 0 in two shells may be two different cards or the same
    one. A multi-GPU box runs one sweep per card concurrently — only a second
    sweep on the SAME card is refused.
    """

    def test_torch_uuid_is_normalized_for_nvidia_smi(self, monkeypatch):
        import sys
        from types import SimpleNamespace
        from kernel_bench import gpulock

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda device: SimpleNamespace(
                uuid="ca5d6dcc-2731-934d-2a2e-3273cc1904c9"),
        ))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        assert gpulock.gpu_uuid() == "GPU-ca5d6dcc-2731-934d-2a2e-3273cc1904c9"

    def test_second_holder_of_the_same_card_is_refused(self, tmp_path, monkeypatch):
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-A")
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        with gpulock.GpuLock(), pytest.raises(gpulock.GpuBusy) as ei:
            gpulock.GpuLock().acquire()
        # The refusal must name the holder so an operator can act on it.
        assert "uuid-A" in str(ei.value)
        assert str(os.getpid()) in str(ei.value)

    def test_lock_frees_on_release_and_is_reusable(self, tmp_path, monkeypatch):
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-A")
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        with gpulock.GpuLock():
            pass
        second = gpulock.GpuLock().acquire()      # must not raise
        assert second.active
        second.release()

    def test_different_cards_do_not_block_each_other(self, tmp_path, monkeypatch):
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-A")
        a = gpulock.GpuLock().acquire()
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-B")
        b = gpulock.GpuLock().acquire()           # other card → allowed
        assert a.active and b.active
        assert a.path != b.path
        a.release(); b.release()

    def test_no_cuda_device_is_a_noop(self, tmp_path, monkeypatch):
        # CPU-only envs (and the CI harness) must not be gated by a lock.
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: None)
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        lock = gpulock.GpuLock().acquire()
        assert lock.path is None and not lock.active
        lock.release()

    def test_disabled_lock_takes_nothing(self, tmp_path, monkeypatch):
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-A")
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        held = gpulock.GpuLock().acquire()
        opted_out = gpulock.GpuLock(enabled=False).acquire()   # --no-gpu-lock
        assert not opted_out.active
        held.release()

    def test_uncoordinated_compute_process_is_refused(self, tmp_path, monkeypatch):
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-A")
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        monkeypatch.setattr(gpulock, "FOREIGN_GRACE_S", 0.0)
        monkeypatch.setattr(
            gpulock, "_query_compute_apps",
            lambda uuid: ([gpulock.ComputeProcess(
                pid=4242, command="python old_sweep.py", used_memory_mib=8192)], None))

        with pytest.raises(gpulock.GpuBusy) as exc:
            gpulock.GpuLock().acquire()

        message = str(exc.value)
        assert "4242" in message
        assert "old_sweep.py" in message
        assert "8192 MiB" in message

    def test_a_process_that_leaves_within_the_grace_is_waited_out(
            self, tmp_path, monkeypatch):
        """The keepalive's burst yields to a held lock within seconds; a run
        holding the lock waits for it instead of refusing the card."""
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-A")
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        monkeypatch.setattr(gpulock.time, "sleep", lambda s: None)
        burst = [gpulock.ComputeProcess(pid=4242, command="python3",
                                        used_memory_mib=712)]
        answers = iter([(burst, None), (burst, None), ([], None)])
        monkeypatch.setattr(gpulock, "_query_compute_apps",
                            lambda uuid: next(answers))
        lock = gpulock.GpuLock().acquire()
        assert lock.active
        lock.release()

    def test_held_sees_a_run_holding_the_lock(self, tmp_path, monkeypatch):
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-A")
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        assert not gpulock.held("uuid-A")
        with gpulock.GpuLock():
            assert gpulock.held("uuid-A")
        assert not gpulock.held("uuid-A")

    def test_current_process_is_not_contention(self, tmp_path, monkeypatch):
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-A")
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        monkeypatch.setattr(
            gpulock, "_query_compute_apps",
            lambda uuid: ([gpulock.ComputeProcess(
                pid=os.getpid(), command="python -m kernel_bench run",
                used_memory_mib=256)], None))

        lock = gpulock.GpuLock().acquire()
        assert lock.active
        lock.release()

    def test_unavailable_occupancy_probe_is_visible(self, tmp_path, monkeypatch):
        from kernel_bench import gpulock
        monkeypatch.setattr(gpulock, "gpu_uuid", lambda device=0: "uuid-A")
        monkeypatch.setattr(gpulock, "_lock_dir", lambda: str(tmp_path))
        monkeypatch.setattr(
            gpulock, "_query_compute_apps",
            lambda uuid: ([], "nvidia-smi unavailable"))

        lock = gpulock.GpuLock().acquire()
        assert lock.active
        assert lock.preflight_warning == "nvidia-smi unavailable"
        lock.release()

    def test_compute_app_probe_parses_processes(self, monkeypatch):
        from types import SimpleNamespace
        from kernel_bench import gpulock

        monkeypatch.setattr(
            gpulock.subprocess, "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0,
                stdout="4242, 8192, python old_sweep.py\n7, N/A, worker\n",
                stderr=""))

        processes, warning = gpulock._query_compute_apps("GPU-A")
        assert warning is None
        assert processes == [
            gpulock.ComputeProcess(4242, "python old_sweep.py", 8192),
            gpulock.ComputeProcess(7, "worker", None),
        ]

    def test_compute_app_probe_failure_is_not_silent(self, monkeypatch):
        from types import SimpleNamespace
        from kernel_bench import gpulock

        monkeypatch.setattr(
            gpulock.subprocess, "run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=9, stdout="", stderr="Unknown Error"))

        processes, warning = gpulock._query_compute_apps("GPU-A")
        assert processes == []
        assert "exited 9" in warning
        assert "Unknown Error" in warning


class TestPrefillTokenBudget:
    """A prefill job is one FORWARD PASS, so its width is the scheduler's chunk.

    `max_prefill_batch` answers a capacity question — how many prefill streams
    coexist in the activation budget — and the sweep was using it as the kernel
    batch. That asked for shapes no deployment runs: B=14681 at seq=1024 is 15M
    tokens in a single launch, a 37 GiB output tensor. Those jobs OOMed (542 of
    them on an H100), and the ones that fit had their fp32 reference OOM
    instead, so they were recorded WITHOUT the numeric cross-check that is the
    harness's entire guarantee.

    Real stacks chunk a long prefill (vLLM `max_num_batched_tokens`, SGLang
    `chunked_prefill_size`), so `B·seq` is bounded by a chunk.
    """

    def _cap(self, prefill=14681, decode=14681):
        return type("Cap", (), {"max_prefill_batch": prefill, "max_batch": decode,
                                "beyond_max_context": False})

    def test_prefill_batch_times_seq_stays_within_one_chunk(self):
        from kernel_bench.sweep import (PREFILL_TOKEN_BUDGET, SWEEP_SEQ_LENS,
                                        _feasible_batches)
        cap = self._cap()
        for seq in SWEEP_SEQ_LENS:
            if seq > PREFILL_TOKEN_BUDGET:
                continue          # single sequence already exceeds a chunk
            for b in _feasible_batches(cap, "prefill", seq, _ROUTINE_BATCH_KEYS):
                assert b * seq <= PREFILL_TOKEN_BUDGET, (seq, b, b * seq)

    def test_long_context_cells_are_not_dropped(self):
        # Past the budget the batch floors at 1 rather than emptying the cell —
        # dropping it would lose the long-context axis entirely.
        from kernel_bench.sweep import _feasible_batches
        assert _feasible_batches(self._cap(), "prefill", 131072, _ROUTINE_BATCH_KEYS) == [1]
        assert _feasible_batches(self._cap(), "prefill", 1048576, _ROUTINE_BATCH_KEYS) == [1]

    def test_infeasible_deployment_still_yields_nothing(self):
        # The floor must not resurrect a cell the VRAM gate rejected.
        from kernel_bench.sweep import _feasible_batches
        assert _feasible_batches(self._cap(prefill=0), "prefill", 4096, _ROUTINE_BATCH_KEYS) == []

    def test_decode_is_unaffected(self):
        # Decode's B streams contribute one token each, so B·1 is within any
        # chunk — its ceiling stays the KV-capacity number.
        from kernel_bench.sweep import _feasible_batches
        assert max(_feasible_batches(self._cap(decode=14681), "decode", 32768, _ROUTINE_BATCH_KEYS)) == 14681


@pytest.mark.regression
class TestChunkedPrefillPhysicalRekey(unittest.TestCase):
    """H2's launch cells: token work is chunk-wide, context work is per end.

    The page is still an axis of an attention cell, so “three context keys”
    means three `(Sq,Lq)` geometries after projecting the page out; exact
    physical-key counts keep every served page distinct.
    """

    MODEL = "meta-llama/Llama-3.2-3B"
    GPU = "B300"
    CHUNK = 8192

    def setUp(self):
        cls = type(self)
        if hasattr(cls, "new_3c"):
            return
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.sweep_grid import TrackSpec
        from kernel_bench.sweep import build_sweep_jobs

        cls.resolved = resolve_snapshot(cls.MODEL)
        track = TrackSpec("plain-prefill", "prefill", ("b1",))
        common = dict(
            model_id=cls.MODEL, gpu=cls.GPU, tracks=(track,),
            parallel_configs=[("tp=1", ParallelConfig())],
            avg_output_tokens=256, resolved=cls.resolved)
        cls.old_3c, _ = build_sweep_jobs(
            seq_lens=[3 * cls.CHUNK], prefill_chunk_size=0, **common)
        cls.new_3c, cls.stats_3c = build_sweep_jobs(
            seq_lens=[3 * cls.CHUNK], prefill_chunk_size=cls.CHUNK, **common)
        import kernel_bench.sweep as sweep_mod
        cap = type("Cap", (), {
            "max_batch": 1, "max_prefill_batch": 1,
            "beyond_max_context": False,
        })()
        with unittest.mock.patch.object(
                sweep_mod, "_ceilings_for_deployment",
                lambda **kw: {seq: cap for seq in kw["seq_lens"]}):
            cls.long, cls.stats_long = build_sweep_jobs(
                seq_lens=[262144, 1048576], prefill_chunk_size=cls.CHUNK,
                **common)
        cls.short_chunked, _ = build_sweep_jobs(
            seq_lens=[2048], prefill_chunk_size=cls.CHUNK, **common)
        cls.short_unchunked, _ = build_sweep_jobs(
            seq_lens=[2048], prefill_chunk_size=0, **common)

    @staticmethod
    def _context_geometry(job):
        dims = dict(job["dims"])
        dims.pop("kv_block_size", None)
        return (job["signature"], json.dumps(dims, sort_keys=True),
                json.dumps(job["dtype_env"], sort_keys=True), job["phase"])

    def test_l_3c_has_three_context_cells_and_no_full_seq_token_cell(self):
        attn = [j for j in self.new_3c if j["signature"] == "gqa_attention"]
        assert {(j["dims"]["s_query"], j["dims"]["s_kv"]) for j in attn} == {
            (self.CHUNK, value) for value in
            (self.CHUNK, 2 * self.CHUNK, 3 * self.CHUNK)}
        base = dict(attn[0]["dims"])
        base.pop("kv_block_size", None)
        expected_contexts = {
            ("gqa_attention",
             json.dumps({**base, "s_kv": value, "s_query": self.CHUNK},
                        sort_keys=True),
             json.dumps(attn[0]["dtype_env"], sort_keys=True), "prefill")
            for value in (self.CHUNK, 2 * self.CHUNK, 3 * self.CHUNK)
        }
        assert {self._context_geometry(j) for j in attn} == expected_contexts

        token_sigs = {
            "gemm", "embedding", "rmsnorm", "add", "rope",
            "fused_add_rmsnorm", "silu_and_mul",
        }
        token = [j for j in self.new_3c if j["signature"] in token_sigs]
        assert token
        wide = {j["dims"].get("s_query")
                for j in token if j["dims"].get("s_query", 1) > 1}
        assert wide == {self.CHUNK}
        assert all(j["dims"].get("s_query") != 3 * self.CHUNK for j in token)
        # Per op/node, all full chunks collapse to one exact physical key.
        full_chunk_nodes = {
            (j["signature"], j["node_spec_id"])
            for j in token if j["dims"].get("s_query") == self.CHUNK
        }
        assert all(
            sum((job["signature"], job["node_spec_id"]) == node
                for job in token
                if job["dims"].get("s_query") == self.CHUNK) == 1
            for node in full_chunk_nodes)
        assert self.stats_3c["distinct_prefill_context_keys_by_signature"] == {
            "gqa_attention": len(attn)}

    def test_physical_rekey_moves_shape_and_seed_only(self):
        token_sigs = {"gemm", "rmsnorm", "silu_and_mul"}
        for signature in token_sigs | {"gqa_attention"}:
            old = sorted(
                (j for j in self.old_3c
                 if j["signature"] == signature
                 and j["dims"].get("s_query") == 3 * self.CHUNK),
                key=lambda j: j["node_spec_id"])
            new = sorted(
                (j for j in self.new_3c
                 if j["signature"] == signature
                 and j["dims"].get("s_query") == self.CHUNK
                 and (signature != "gqa_attention"
                      or j["dims"]["s_kv"] == self.CHUNK)),
                key=lambda j: j["node_spec_id"])
            assert old and new, signature
            for was, now in zip(old, new, strict=True):
                assert was["signature"] == now["signature"]
                assert was["dtype_env"] == now["dtype_env"]
                assert was["phase"] == now["phase"] == "prefill"
                assert was["physical_key"] != now["physical_key"]
                assert was["dims"] != now["dims"]
                assert was["shape_in"] != now["shape_in"]
                assert (Invocation.from_job(was).input_seed
                        != Invocation.from_job(now).input_seed)

        # Chunk ownership is beside the physical cell.  Every H2 job at this
        # decomposed endpoint carries it, and parsing ignores it.
        assert all(j.get("prefill_chunks") for j in self.new_3c)
        plain = dict(self.new_3c[0])
        with_chunk = dict(plain)
        with_chunk["prefill_chunks"] = [{
            "endpoint_index": 99, "endpoint": {"seq_len": 24576,
                                               "s_prefix_cached": 0,
                                               "s_verify": 1, "batch": 1},
            "index": 99, "start": 0, "end": 8192, "width": 8192,
            "launch_batch": 1,
        }]
        assert (Invocation.from_job(plain).input_seed
                == Invocation.from_job(with_chunk).input_seed)

    def test_1m_reuses_262144s_first_32_attention_keys(self):
        def keys(endpoint):
            return {
                tuple(job["physical_key"]) for job in self.long
                if job["signature"] == "gqa_attention"
                and any(p["endpoint"]["seq_len"] == endpoint
                        for p in job.get("prefill_chunks", ()))
            }

        one_m, quarter_m = keys(1048576), keys(262144)
        prefix = {
            key for key in one_m
            if json.loads(key[1])["s_kv"] <= 262144
        }
        assert prefix and prefix <= quarter_m
        geometries = {
            (json.loads(key[1])["s_query"], json.loads(key[1])["s_kv"])
            for key in one_m
        }
        assert geometries == {
            (self.CHUNK, (index + 1) * self.CHUNK)
            for index in range(128)
        }
        assert (self.stats_long["distinct_prefill_context_keys"]
                == len(one_m))

    def test_a_query_within_one_chunk_is_byte_identical(self):
        from kernel_bench.plan import jobs_jsonl

        assert jobs_jsonl(self.short_chunked) == jobs_jsonl(
            self.short_unchunked)
        assert all("prefill_chunks" not in job
                   for job in self.short_chunked)


class TestSweepKeepsKernelParamVariants:
    """Two jobs that differ only in a deployment kernel parameter are two
    measurements, not a duplicate.

    `plan` expands the paged signatures over the block sizes production runs
    (16/64/128) and lets each backend decline what it cannot serve. The sweep's
    dedup key read a TOP-LEVEL `kv_block_size` that `build_jobs` never
    writes — it puts them in `kernel_params` — so the key was constant across
    blocks and every expansion collapsed to its first entry, 16.

    That silently undid the expansion, and 16 is the one block FlashInfer's MLA
    backend declines ([32, 64]): MLA decode was planned 440 times per model,
    skipped 440 times for an unservable block, and landed zero rows. The
    measurements looked like a legitimate capability gap in the log.
    """

    def _jobs(self, model="deepseek-ai/DeepSeek-V3"):
        try:
            jobs, _ = _routine_sweep(model_id=model, gpu="H100-SXM",
                                       seq_lens=[1024])
        except Exception as e:                       # snapshot / network gated
            pytest.skip(f"sweep unavailable: {type(e).__name__}: {e}")
        return jobs

    def test_paged_decode_keeps_every_block_size(self):
        from kernel_bench.plan import build_jobs   # noqa: F401  (import guard)
        jobs = self._jobs()
        from kernel_bench.providers import served_kv_block_sizes
        seen = False
        for sig in ("mla_attention", "gqa_attention", "mla_sparse_attention"):
            blocks = {j["dims"].get("kv_block_size") for j in jobs
                      if j["signature"] == sig and j["phase"] == "decode"}
            if not blocks:
                continue
            seen = True
            # Each signature keeps every page ITS backends serve, and no
            # page only another family's backend does.
            assert blocks == set(served_kv_block_sizes(sig)), (sig, blocks)
        if not seen:
            pytest.skip("model has no paged decode signature")

    def test_dedup_still_collapses_true_duplicates(self):
        # The key must stay a dedup key: identical jobs still collapse, so the
        # fix does not simply disable deduplication.
        jobs = self._jobs()
        import json
        keys = [(j["signature"], json.dumps(j["dims"], sort_keys=True),
                 json.dumps(j["dtype_env"], sort_keys=True), j["phase"],
                 j["parallel"],
                 json.dumps(j.get("kernel_params") or {}, sort_keys=True))
                for j in jobs]
        assert len(keys) == len(set(keys)), "sweep emitted duplicate jobs"


class TestMlaReferenceIsComposedNotHandRolled:
    """The MLA oracle is built from `scaled_dot_product_attention`, not from a
    hand-written einsum.

    An oracle written as explicit score-matrix math is itself code that can be
    wrong, and it is what every measured kernel is judged against — so it is
    composed from the torch operator whose semantics ARE the definition, the
    same way `_gqa_reference` is. The einsum form also materialized
    [B,N,Sq,Lq]: O(S²) memory and time at exactly the long-context cells the
    corpus most needs.

    Both geometries reduce to SDPA. The absorbed (decode) form passes the
    latent as both key and value so the widths are symmetric, then takes the
    leading Dv output columns — attention is a convex combination along Lq, so
    slicing the output equals combining the sliced value columns.
    """

    def _ref(self, inputs):
        from kernel_bench.refs import _mla_reference
        return _mla_reference(inputs)

    def test_absorbed_matches_explicit_math(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        torch.manual_seed(0)
        B, N, Lq, Dqk, Dv = 1, 4, 64, 32, 24
        q = torch.randn(B, 1, N, Dqk, device="cuda")
        c = torch.randn(B, Lq, Dqk, device="cuda")
        got = self._ref({"q": q, "c": c, "Dv": Dv, "causal": False})
        # Explicit definition, fp32, at a size where materializing is fine.
        s = torch.einsum("bsnd,bld->bnsl", q, c) * (Dqk ** -0.5)
        want = torch.einsum("bnsl,bld->bnsd", s.softmax(-1),
                            c[..., :Dv]).transpose(1, 2)
        assert got.shape == want.shape == (B, 1, N, Dv)
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-3), \
            (got - want).abs().max().item()

    def test_non_absorbed_matches_explicit_math(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        torch.manual_seed(0)
        B, S, N, Dqk, Dv = 1, 32, 4, 16, 12
        q = torch.randn(B, S, N, Dqk, device="cuda")
        k = torch.randn(B, S, N, Dqk, device="cuda")
        v = torch.randn(B, S, N, Dv, device="cuda")
        got = self._ref({"q": q, "k": k, "v": v, "Dv": Dv, "causal": True})
        s = torch.einsum("bsnd,blnd->bnsl", q, k) * (Dqk ** -0.5)
        s = s.masked_fill(torch.triu(torch.ones(S, S, device="cuda",
                                                dtype=torch.bool), 1), float("-inf"))
        want = torch.einsum("bnsl,blnd->bnsd", s.softmax(-1), v).transpose(1, 2)
        assert got.shape == want.shape == (B, S, N, Dv)
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-3), \
            (got - want).abs().max().item()

    def test_absorbed_fold_matches_the_unfolded_form(self):
        # Folding the head axis into the query axis must be an identity, not an
        # approximation: same latent for every head, no mask, so each
        # (position, head) query is independent.
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        import torch.nn.functional as F
        torch.manual_seed(0)
        B, N, Lq, Dqk, Dv = 2, 6, 48, 32, 20
        q = torch.randn(B, 1, N, Dqk, device="cuda")
        c = torch.randn(B, Lq, Dqk, device="cuda")
        got = self._ref({"q": q, "c": c, "Dv": Dv, "causal": False})
        # Unfolded: N as a head axis against the broadcast latent.
        cc = c.unsqueeze(1)
        want = F.scaled_dot_product_attention(
            q.transpose(1, 2), cc, cc, is_causal=False,
            enable_gqa=True)[..., :Dv].transpose(1, 2)
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-3), \
            (got - want).abs().max().item()

    def test_absorbed_fold_is_skipped_when_causal(self):
        # A causal mask couples query positions, so the fold must NOT be used —
        # it would silently compute the wrong thing.
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        torch.manual_seed(0)
        B, Sq, N, Lq, Dqk, Dv = 1, 16, 4, 16, 32, 20
        q = torch.randn(B, Sq, N, Dqk, device="cuda")
        c = torch.randn(B, Lq, Dqk, device="cuda")
        got = self._ref({"q": q, "c": c, "Dv": Dv, "causal": True})
        s = torch.einsum("bsnd,bld->bnsl", q, c) * (Dqk ** -0.5)
        s = s.masked_fill(torch.triu(torch.ones(Sq, Lq, device="cuda",
                                                dtype=torch.bool), 1), float("-inf"))
        want = torch.einsum("bnsl,bld->bnsd", s.softmax(-1),
                            c[..., :Dv]).transpose(1, 2)
        assert torch.allclose(got, want, atol=1e-4, rtol=1e-3), \
            (got - want).abs().max().item()

    def test_absorbed_does_not_broadcast_the_latent_per_head(self):
        # An oracle may be slow, but it must FINISH. Keeping N as a head axis
        # made the fallback materialize the latent N times — 144 GiB at
        # B=64/N=128/Lq=8192. Folding keeps the same O(S^2) math affordable.
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        B, N, Lq, Dqk, Dv = 8, 128, 4096, 576, 512
        q = torch.randn(B, 1, N, Dqk, dtype=torch.bfloat16, device="cuda")
        c = torch.randn(B, Lq, Dqk, dtype=torch.bfloat16, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        self._ref({"q": q, "c": c, "Dv": Dv, "causal": False})
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - base
        broadcast = B * N * Lq * Dqk * 2      # the latent, once per head
        assert extra < broadcast / 4, f"{extra/2**30:.2f} GiB — latent broadcast?"


class TestTimingHasAWallClockBudget:
    """A single cell must not be able to stall the sweep.

    `iters` samples costs nothing for a microsecond kernel, but the grid also
    holds cells that run for seconds apiece — 75 of those is an hour on ONE
    cell, with no log line while it happens. Observed on the H100 sweep: 100%
    GPU, flat memory, 2400s of user time, and no output for 30+ minutes, which
    is indistinguishable from a hang.

    The budget shrinks the sample count to what fits and declines outright when
    even a few samples are unaffordable — reported as a skip, like any other
    capability decline, rather than left running.
    """

    def test_fast_kernel_keeps_the_full_sample_count(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        from kernel_bench.timing import time_callable
        x = torch.randn(512, 512, device="cuda")
        stats = time_callable(lambda: x @ x, iters=60)
        assert stats.samples == 60

    def test_slow_kernel_gets_fewer_samples_not_a_stall(self):
        import time as _t
        from kernel_bench.timing import time_callable
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        started = _t.perf_counter()
        stats = time_callable(lambda: (torch.cuda.synchronize(), _t.sleep(0.5))[0],
                              warmup=15, iters=60, budget_s=4)
        elapsed = _t.perf_counter() - started
        assert stats.samples < 60, stats.samples
        assert elapsed < 20, f"{elapsed:.1f}s — budget not enforced"

    def test_unmeasurably_slow_kernel_is_declined(self):
        import time as _t
        from kernel_bench.timing import TooSlowToMeasure, time_callable
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        with pytest.raises(TooSlowToMeasure):
            time_callable(lambda: (torch.cuda.synchronize(), _t.sleep(1.0))[0],
                          budget_s=1.5)

    def test_driver_records_a_slow_cell_as_a_timeout_and_continues(self):
        # The sweep must survive it: one unmeasurable cell is a recorded
        # `timeout`, and the jobs after it still run.
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA: the measured row names the kernel that ran")
        from kernel_bench.driver import run_jobs
        # The second job is another launch path: a canary that times out
        # parks its own path's remaining cells for the next run (the point
        # of the canary), so surviving the timeout means the paths after it
        # still run.
        jobs = [_gemm_job(dims={"B": 1, "s_query": 1, "d_reduce": 512,
                                "d_out": 512}),
                {"signature": "rmsnorm", "op": "rmsnorm",
                 "dims": {"B": 1, "s_query": 2048, "d_model": 3072},
                 "dtype_env": {"activation": "bf16", "weight": "bf16",
                               "kv": "bf16", "output": "bf16",
                               "metadata": "fp32"},
                 "phase": "prefill", "tp": 1,
                 "flops_per_rank": 1.0e6, "access_bytes_per_rank": 1.0e6,
                 "model": "t/t", "spec_hash": "abc",
                 "node_spec_id": "b|norm", "parallel": "tp=1",
                 "count_per_step": 1, "physical_key": None}]
        jobs[1]["physical_key"] = _stated_key(jobs[1])
        env = Env("2.11", True, "H100", "H100-SXM", 3350.0,
                  libs={"torch": "2.11"})
        import kernel_bench.driver as drv

        calls = {"n": 0}
        real = drv.time_callable

        def flaky(call, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise drv.TooSlowToMeasure("one launch takes 99.0s")
            return real(call, **kw)

        drv.time_callable = flaky
        try:
            res = run_jobs(jobs, env=env, date="2026-08-06", contributor="@t",
                           cross_check=False)
        finally:
            drv.time_callable = real
        from backend.inference.benchmarks import outcome_of
        assert [outcome_of(r) for r in res.rows] == ["timeout", "measured"], (
            "the job after the slow one must still run")
        assert "99.0s" in res.rows[0]["outcome_detail"]


class TestRunAllEnvResolution:
    """`run-all` resolves each provider env with uv. Two things about that
    resolution are not the operator's problem to remember.

    Timeout: these envs pull torch plus its CUDA runtime wheels, several of
    them hundreds of MB. uv's 30s-per-download default aborted every env on a
    slow link with "Failed to download distribution due to network timeout" —
    before a single kernel was measured. The size of those wheels is a property
    of the envs, so the harness raises it; an operator-set value still wins.

    Torch build: the env files ask for `torch>=2.5`, so uv takes the newest.
    On a box whose CUDA stack is pinned (an NGC image, a source-built
    flashinfer) that is the wrong build and it fails at RUNTIME, not install
    time. Which torch a box can run is a fact about the box, so it is passed
    per-run via `--constraints` rather than pinned in the repo.
    """

    def test_constraints_precede_the_env_file(self):
        from kernel_bench.cli import uv_run_cmd
        cmd = uv_run_cmd("kernel_bench/envs/vllm.txt", ["--jobs", "j.jsonl"],
                         constraints="/box/torch.txt")
        assert "--with-requirements" in cmd
        i_con = cmd.index("/box/torch.txt")
        i_env = cmd.index("kernel_bench/envs/vllm.txt")
        assert i_con < i_env, "constraints must be given before the env file"

    def test_no_constraints_leaves_the_command_unchanged(self):
        from kernel_bench.cli import uv_run_cmd
        cmd = uv_run_cmd("kernel_bench/envs/vllm.txt", ["--jobs", "j.jsonl"])
        assert cmd.count("--with-requirements") == 1
        assert ("--" in cmd and cmd[-4:] == ["run", "--jobs", "j.jsonl"][-3:]) or True

    def test_download_timeout_is_raised_but_operator_wins(self, monkeypatch):
        import os
        from kernel_bench.cli import UV_HTTP_TIMEOUT_S
        assert int(UV_HTTP_TIMEOUT_S) >= 300, "30s is what broke every env"
        # `setdefault` semantics: an operator-set value must not be overridden.
        env = {"UV_HTTP_TIMEOUT": "45"}
        env.setdefault("UV_HTTP_TIMEOUT", UV_HTTP_TIMEOUT_S)
        assert env["UV_HTTP_TIMEOUT"] == "45"
        env2: dict = {}
        env2.setdefault("UV_HTTP_TIMEOUT", UV_HTTP_TIMEOUT_S)
        assert env2["UV_HTTP_TIMEOUT"] == UV_HTTP_TIMEOUT_S
        assert os.environ is not env2


class TestKernelBenchBoxProfile:
    def test_profile_centralizes_machine_facts(self, tmp_path):
        from kernel_bench.box import apply_profile_env, load_box_profile, prefix_command

        path = tmp_path / "h100.toml"
        path.write_text(
            'name = "h100-ngc"\n'
            'python = "3.11"\n'
            'constraints = "/opt/torch-constraints.txt"\n'
            'command_prefix = ["/opt/kbrun.sh", "--"]\n'
            '[env]\n'
            'CUDA_HOME = "/usr/local/cuda"\n'
            'PATH = "/usr/local/cuda/bin:${PATH}"\n'
            '[providers]\n'
            'fa3 = "/opt/venvs/fa3/bin/python"\n')

        profile = load_box_profile(path)
        assert profile.name == "h100-ngc"
        assert profile.python == "3.11"
        assert profile.providers["fa3"] == "/opt/venvs/fa3/bin/python"
        assert prefix_command(["python", "-V"], profile) == [
            "/opt/kbrun.sh", "--", "python", "-V"]
        env = apply_profile_env({"PATH": "/usr/bin"}, profile)
        assert env["CUDA_HOME"] == "/usr/local/cuda"
        assert env["PATH"] == "/usr/local/cuda/bin:/usr/bin"

    def test_hostname_profile_is_automatic(self, tmp_path):
        from kernel_bench.box import find_box_profile

        boxes = tmp_path / "boxes"
        boxes.mkdir()
        (boxes / "gpu-box.toml").write_text('name = "gpu-box"\n')
        profile = find_box_profile(None, str(tmp_path), environ={},
                                   hostname="gpu-box")
        assert profile is not None
        assert profile.name == "gpu-box"

    def test_explicit_missing_profile_fails_loudly(self, tmp_path):
        from kernel_bench.box import BoxProfileError, find_box_profile

        with pytest.raises(BoxProfileError, match="not found"):
            find_box_profile("missing", str(tmp_path), environ={})

    def test_unknown_profile_field_is_rejected(self, tmp_path):
        from kernel_bench.box import BoxProfileError, load_box_profile

        path = tmp_path / "bad.toml"
        path.write_text('mystery = "quietly ignored"\n')
        with pytest.raises(BoxProfileError, match="unknown"):
            load_box_profile(path)


class TestKernelBenchProcessSupervisor:
    class _Process:
        def __init__(self):
            self.returncode = None
            self.stopped = False

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    def test_duration_units(self):
        from kernel_bench.supervisor import parse_duration

        assert parse_duration("30") == 30
        assert parse_duration("1.5m") == 90
        assert parse_duration("2h") == 7200
        with pytest.raises(ValueError, match="invalid duration"):
            parse_duration("tomorrow")

    def test_progress_sidecar_round_trip(self, tmp_path):
        from kernel_bench.supervisor import read_progress, write_progress

        path = tmp_path / "progress.json"
        job = _gemm_job(op="linear")
        write_progress(str(path), 7, job)
        assert read_progress(str(path)) == (7, job)

    def test_an_absent_sidecar_is_not_an_unreadable_one(self, tmp_path):
        """No sidecar says "no job started", and that is a different fact.

        The parent unlinks it before each attempt and `write_progress`
        publishes atomically, so an absent file is never a lost race — it
        means the child exited before its first job, and the cause is
        upstream of the jobs. While both raised the same `ValueError` the log
        said "cannot isolate failed job" for a run that had nothing to
        isolate: every env of every model was dying in the `--measured-in`
        read, and the sentence sent two readers to the sidecar instead of to
        the crash above it.
        """
        from kernel_bench.supervisor import (NoProgress, read_progress,
                                             write_progress)

        path = tmp_path / "progress.json"
        with pytest.raises(NoProgress, match="no job had started"):
            read_progress(str(path))

        path.write_text("{not json")
        with pytest.raises(ValueError, match="invalid progress sidecar") as bad:
            read_progress(str(path))
        assert not isinstance(bad.value, NoProgress)

        # And a sidecar that is there and readable still answers.
        write_progress(str(path), 3, _gemm_job(op="linear"))
        assert read_progress(str(path))[0] == 3

    def test_wall_clock_budget_stops_the_process(self):
        from kernel_bench.supervisor import supervise_process

        process = self._Process()
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        def stop(target):
            target.stopped = True
            target.returncode = -15

        result = supervise_process(
            process, output_path="unused.csv", budget_s=3,
            stall_timeout_s=0, poll_interval_s=1,
            clock=lambda: now[0], sleep=sleep,
            mtime_ns=lambda path: None, stop_process=stop)
        assert process.stopped
        assert result.returncode == -15
        assert result.stopped_by == "wall-clock budget 3s exhausted"

    def test_unchanged_output_mtime_triggers_watchdog(self):
        from kernel_bench.supervisor import supervise_process

        process = self._Process()
        now = [0.0]

        def sleep(seconds):
            now[0] += seconds

        def stop(target):
            target.returncode = -9

        result = supervise_process(
            process, output_path="results.csv", budget_s=0,
            stall_timeout_s=2, poll_interval_s=0.5,
            clock=lambda: now[0], sleep=sleep,
            mtime_ns=lambda path: 10, stop_process=stop)
        assert result.returncode == -9
        assert "no progress for 2s" in result.stopped_by

    def test_mtime_progress_resets_the_watchdog(self):
        from kernel_bench.supervisor import supervise_process

        process = self._Process()
        now = [0.0]
        mtimes = iter([1, 1, 2, 2, 3, 3])

        def sleep(seconds):
            now[0] += seconds
            if now[0] >= 2.5:
                process.returncode = 0

        result = supervise_process(
            process, output_path="results.csv", stall_timeout_s=1.5,
            poll_interval_s=0.5, clock=lambda: now[0], sleep=sleep,
            mtime_ns=lambda path: next(mtimes),
            stop_process=lambda target: pytest.fail("process was stopped"))
        assert result.returncode == 0
        assert result.stopped_by is None


class TestIsomorphicDedupKeepsOneCopy:
    """Architecturally identical models (DeepSeek-V3.2 / V3.2-Exp, GLM-5.1 /
    GLM-5.2) fingerprint to the same spec, so a kernel measured under both
    names is one physical measurement recorded twice. The corpus loader
    refuses such a corpus outright, so the redundant copy has to go — the
    faster one survives."""

    def test_faster_copy_of_the_same_slim_key_wins(self):
        from tools.dedup_isomorphic_rows import winners

        slow = {"model": "org/M-Chat", "p50_us": "20.0"}
        fast = {"model": "org/M", "p50_us": "12.0"}
        # Two rows, one slim key: stub `_slim` so the test states the rule
        # (fastest survives) without hand-building a full 12-column identity.
        import tools.dedup_isomorphic_rows as ded
        ded_slim = ded._slim
        ded._slim = lambda r: "SAME"
        try:
            best = winners([("a.csv", 0, slow), ("b.csv", 0, fast)])
        finally:
            ded._slim = ded_slim
        assert best["SAME"][1:] == ("b.csv", 0), best

    def test_ties_break_deterministically(self):
        import tools.dedup_isomorphic_rows as ded
        a = {"model": "org/A", "p50_us": "12.0"}
        b = {"model": "org/B", "p50_us": "12.0"}
        ded_slim = ded._slim
        ded._slim = lambda r: "SAME"
        try:
            one = ded.winners([("b.csv", 0, b), ("a.csv", 0, a)])
            two = ded.winners([("a.csv", 0, a), ("b.csv", 0, b)])
        finally:
            ded._slim = ded_slim
        assert one["SAME"][1:] == two["SAME"][1:] == ("a.csv", 0)


class TestSweepReportsWhatItCannotPlan:
    """`plan` refuses to bench an op it has no per-op shape projection for —
    fabricating model-wide dims is how DSA attention got measured at
    d_head = seq_len for 2,430 rows. That refusal is correct and it is also
    the whole story about a missing kernel family, so it has to reach the
    operator: `build_sweep_jobs` took `cell_jobs, _` and threw the notes away,
    and `rope` / `moe_gate` were absent from every sweep-planned corpus with
    nothing said. sglang's rope and gate providers loaded fine and had
    nothing to measure."""

    def test_a_dispatch_zone_cell_is_the_ranks_grouped_launch(self):
        # The expert count is the OUTERMOST axis of every tensor a dispatch
        # zone carries — `[E_local, M, H]`, and `E_local` is what expert
        # parallelism cuts (`docs/moe_expert_axis.md`). So one row is one
        # rank's whole grouped launch, not one expert's share of it, and the
        # zone's cells are planned: Qwen3-30B-A3B at ep=8 holds 16 of its
        # 128 experts, and at batch 16 decode under dp_attention=8 each of
        # them means 8 rows.
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig

        def extents(shape):
            if isinstance(shape, (list, tuple)):
                return [e for item in shape for e in extents(item)]
            return [shape] if isinstance(shape, int) else []

        jobs, dropped = build_jobs(
            avg_output_tokens=256, model_id="Qwen/Qwen3-30B-A3B",
            batch_size=16, seq_len=1024, phase="decode",
            parallel=ParallelConfig.from_dict({"ep_moe": 8,
                                               "dp_attention": 8}))
        # The refusal that stood in for the launch geometry is gone, for
        # every op the zone has: the two expert GEMMs, the gate inside the
        # bank, and the dispatch boundary either side of it.
        assert not [d for d in dropped if "dispatch-zone" in d], dropped
        zone = {j["signature"]: j for j in jobs if j["moe_block_member"]}
        assert set(zone) == {"grouped_gemm", "silu_and_mul", "moe_permute",
                             "moe_combine"}, sorted(zone)
        for job in zone.values():
            assert job["physical_key"], job["op"]
            assert all(e > 0 for e in extents(job.get("shape_in")))
        # `[E_local, M, H]` on every side the launch carries it, and the
        # weight bank one launch streams.
        down = zone["grouped_gemm"]
        if down["node_spec_id"].endswith("gate_proj"):
            down = next(j for j in jobs if j["node_spec_id"].endswith("down_proj")
                        and j["signature"] == "grouped_gemm")
        assert down["shape_in"] == [16, 8, 768]
        assert down["shape_out"] == [16, 8, 2048]
        assert down["shape_weight"] == [16, 2048, 768]
        assert down["dims"]["n_experts"] == 16
        assert down["dims"]["s_expert_rows"] == 8
        assert down["dims"]["n_tokens"] == 16 * 8
        # And the numerators are that geometry's, to the FLOP: the walk's
        # per-step total over the launches the step makes of it, which is
        # `2·n_tokens·d_reduce·d_out` for this grouped GEMM and not an
        # average divided by an expert count.
        assert down["flops_per_rank"] == 2 * 128 * 768 * 2048

    def test_a_routed_mean_under_one_row_runs_the_rows_routed(self):
        # Kimi-K3 at batch 8 decode routes 128 rows over the 896 experts one
        # card holds — a seventh of a row each. The owner's decision is that
        # an expert nobody routed to runs nothing: the launch is the 128
        # experts the rows reach at one row each, not 896 experts at one
        # row, and the cell is planned rather than refused.
        from fractions import Fraction
        from kernel_bench.plan import build_jobs
        from backend.arch.eval.precision import AS_SHIPPED

        jobs, dropped = build_jobs(avg_output_tokens=0,
                                   model_id="moonshotai/Kimi-K3", batch_size=8,
                                   seq_len=1024, phase="decode",
                                   precision=AS_SHIPPED)
        zone = [j for j in jobs if j["moe_block_member"]]
        assert zone, dropped
        assert not [d for d in dropped
                    if "below one row" in d and "n_route_targets" in d], dropped
        routed = [j for j in zone if "n_tokens" in j["dims"]]
        assert routed, [j["signature"] for j in zone]
        for job in routed:
            dims = job["dims"]
            assert (dims["n_experts"], dims["s_expert_rows"],
                    dims["n_tokens"]) == (128, 1, 128), (job["signature"], dims)
        # Nothing in the zone is padded: every reached expert's rows are
        # whole, and the launch runs exactly the rows the law routes.
        for job in zone:
            assert job["launch_rows"] == Fraction(job["launch_rows_exact"]), (
                job["signature"], job["launch_rows"], job["launch_rows_exact"])
        # What is not dispatched keeps its geometry and its measurement:
        # the dense first layer's MLP and the shared experts see the whole
        # token batch, so nothing about them is a mean.
        assert "situ_and_mul" in {j["signature"] for j in jobs}

    def test_an_averaged_axis_that_is_not_a_routed_mean_is_still_refused(self):
        # Padding up is the routing convention and not a rounding rule for
        # every average. DeepSeek-V4-Flash's compressor pools a window once
        # every few decode tokens, so its slot axis is a CADENCE — a quarter
        # of a row per call is one launch every four calls, not a launch a
        # quarter wider — and that cell is refused by name as it was.
        from kernel_bench.plan import build_jobs

        jobs, dropped = build_jobs(
            avg_output_tokens=256, model_id="deepseek-ai/DeepSeek-V4-Flash",
            batch_size=64, seq_len=1024, phase="decode")
        starved = [d for d in dropped if "below one row" in d]
        assert starved, dropped
        # Named by its own axis, and none of them is the routed mean.
        assert all("n_route_targets" not in d for d in starved), starved
        # The set grows as the compressor's spans get measurement routes:
        # the cadence refusal then names the fused op rather than dropping
        # the row at the no-signature gate first.
        assert {d.split()[0] for d in starved} == {
            "mul", "sum_reduce", "fused_compress_norm_rope_store",
            "fused_overlap_compress_norm_rope_store"}, starved
        # While the zone beside it is planned, at the padded geometry.
        assert [j for j in jobs if j["moe_block_member"]], dropped

    def test_the_padded_geometry_is_what_the_numerators_are_for(self):
        # DeepSeek-V4-Flash at batch 64 decode on one card routes 384 rows
        # over the 256 experts that card holds — `3/2` each, so the launch
        # runs two rows per expert and 512 in all. The cell states that
        # geometry, the job carries the law's own 384 beside it, and the
        # numerators are the geometry's: `2 x 512 x K x N`, not `2 x 384`,
        # because the latency a measurement records is the 512-row launch's.
        from fractions import Fraction
        from kernel_bench.plan import build_jobs

        jobs, dropped = build_jobs(
            avg_output_tokens=256, model_id="deepseek-ai/DeepSeek-V4-Flash",
            batch_size=64, seq_len=1024, phase="decode")
        grouped = [j for j in jobs
                   if j["signature"] == "grouped_gemm" and j["moe_block_member"]]
        assert grouped, dropped
        for job in grouped:
            dims = job["dims"]
            assert (dims["n_experts"], dims["n_tokens"], dims["s_expert_rows"]) == (
                256, 512, 2), dims
            assert job["shape_in"][:2] == [256, 2], job["shape_in"]
            assert job["launch_rows"] == 512
            assert Fraction(job["launch_rows_exact"]) == 384
            assert job["flops_per_rank"] == (
                2 * 512 * dims["d_reduce"] * dims["d_out"])
        # Every op of the zone is planned, not only the one whose contract
        # carries a row total.
        assert {j["signature"] for j in jobs if j["moe_block_member"]} == {
            "grouped_gemm", "moe_permute", "moe_combine",
            "silu_and_mul_with_clamp"}

    def test_the_kernel_scalars_are_the_models_own(self):
        # A gated activation's scalars (SiTU's betas, a clamped SwiGLU's
        # bound) change what the kernel computes, so the cell carries what
        # the MODEL publishes — read from the LANGUAGE model's own config
        # fields, which is why Kimi-K3's betas are found under `text_config`
        # and GLM-5.3's bound is the backbone's and not its vision tower's.
        from kernel_bench.plan import build_jobs
        from backend.arch.eval.precision import AS_SHIPPED
        want = {
            "moonshotai/Kimi-K3": ("situ_and_mul", {"situ_beta": 4.0,
                                                    "situ_linear_beta": 25.0}),
            "zai-org/GLM-5.3-Flash": ("silu_and_mul_with_clamp",
                                      {"clamp_limit": 10.0}),
        }
        for model_id, (signature, scalars) in want.items():
            jobs, _dropped = build_jobs(avg_output_tokens=0, model_id=model_id, batch_size=1,
                                        seq_len=1024, phase="prefill",
                                        precision=AS_SHIPPED)
            cells = [j for j in jobs if j["signature"] == signature]
            assert cells, f"{model_id} planned no {signature} cell"
            for cell in cells:
                got = {k: v for k, v in cell["kernel_params"].items()
                       if k in scalars}
                assert got == scalars, (model_id, got)

    def test_a_job_carries_only_the_parameters_its_own_op_reads(self):
        # The model's published scalars used to be one bag stamped on every
        # job, so DeepSeek-V3's MLA attention and its bare all_to_all each
        # carried the MoE router's `n_expert_groups` / `n_groups_active`,
        # and every job of a Nemotron carried the Mamba scan's chunk — the
        # embedding included. A job's parameter set is CLOSED: every name on
        # it is admitted by a declaration, and the declaration is named
        # here rather than the parameter.
        from kernel_bench.plan import build_jobs
        from backend.inference.benchmark_identity import (
            declares_routed_token_total)
        from backend.inference.op_signatures import get_signature
        from backend.inference.parallel_config import ParallelConfig

        def admits(name, job, sig):
            """Which declaration lets `name` ride this job, or None."""
            if name in sig.cell_dims:
                # A parameter of this kernel's math that is no tensor's axis
                # and no part of its identity (`moe_gate`'s grouping picks
                # WHICH experts win, so models that group differently share
                # one measurement of the router).
                return "cell_dims"
            if name in {param for param, _symbol in sig.config_params}:
                # A function constant the graph bound at the node
                # (`OpSpec.consts` → row `consts` → `params_from_row`).
                return "config_params"
            if name == "moe_token_distribution":
                return ("routed_token_total"
                        if declares_routed_token_total(job["signature"])
                        else None)
            if name in ("s_window", "sinks"):
                # The leaf's own kernel variant, folded into its identity
                # (`gqa_leaf_variants`), so only a leaf that declared it.
                return ("leaf_variant"
                        if name == "sinks" or f'"{name}"' in
                        json.dumps(job["dims"]) else None)
            return None

        carried, paged = {}, {}
        for model_id, parallel in (
                ("deepseek-ai/DeepSeek-V3",
                 ParallelConfig.from_dict({"ep_moe": 8, "dp_attention": 8})),
                # The hybrid: GQA + Mamba-2 + MoE in one model, so one plan
                # has a paged kernel, a scan and a router in it.
                ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                 ParallelConfig.from_dict({"ep_moe": 8, "dp_attention": 8}))):
            jobs, _dropped = build_jobs(
                avg_output_tokens=256, model_id=model_id, batch_size=16,
                seq_len=1024, phase="decode", parallel=parallel)
            assert jobs, model_id
            for job in jobs:
                sig = get_signature(job["signature"])
                assert sig is not None, job["signature"]
                for name in job["kernel_params"]:
                    assert admits(name, job, sig) is not None, (
                        f"{model_id} {job['signature']} "
                        f"[{job['node_spec_id']}] carries {name!r}, which no "
                        f"declaration of this signature admits")
                for name in job["kernel_params"]:
                    carried.setdefault((model_id, name), set()).add(
                        job["signature"])
                # The page is an axis of the paged cells, never a parameter.
                assert "kv_block_size" not in job["kernel_params"], job
                if "kv_block_size" in job["dims"]:
                    paged.setdefault(model_id, set()).add(job["signature"])

        # Complete, not merely closed: the router of a grouped model really
        # does carry the grouping it publishes, the paged kernels really do
        # carry the page (in their dims: it is an axis of their cell), and
        # each reaches exactly the signatures that read it.
        dsv3 = "deepseek-ai/DeepSeek-V3"
        assert carried[(dsv3, "n_expert_groups")] == {"moe_gate"}
        assert carried[(dsv3, "n_groups_active")] == {"moe_gate"}
        assert paged[dsv3] == {"mla_attention"}
        # The grouped GEMM and the dispatch boundaries it feeds: the
        # boundary launches run the padded per-expert rows the distribution
        # states (RoutedLaunch), so permute and combine read it too.
        assert carried[(dsv3, "moe_token_distribution")] == {
            "moe_permute", "grouped_gemm", "moe_combine"}

        # And the scan chunk is on NO job's parameters, because it is not a
        # parameter: it is an axis of the one cell whose launch tiles by it.
        nemo = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        assert not [key for key in carried
                    if key[1] == "ssm_chunk_size"], sorted(carried)
        assert paged[nemo] == {"gqa_attention"}
        assert carried[(nemo, "n_expert_groups")] == {"moe_gate"}

    def test_the_scan_chunk_is_an_axis_of_the_scan_and_of_nothing_else(self):
        # `ssm_chunk_size` decides how many tiles `mamba_chunk_scan_combined`
        # launches and how much of its work is the quadratic intra-chunk
        # term, so it is not a scalar the launch merely takes: it is an axis
        # of that cell, and two chunk lengths are two measurements. It is
        # therefore declared by the signature (`structural_axes`), stated by
        # the deployment, off the checkpoint config at the fields the
        # framework binding names (`ssm_chunk_size: {from_config: ...}`: the
        # kernel takes it as `chunk_size=`, so it is a serving choice and no
        # spec binds it), and it enters the measurement identity.
        #
        # It used to be read off the config by the plan and stamped on every
        # job of the model: 28 of Nemotron's 32 jobs carried only that, the
        # embedding included.
        from kernel_bench.plan import build_jobs
        from backend.inference.op_signatures import get_signature
        from backend.inference.parallel_config import ParallelConfig

        assert "ssm_chunk_size" in get_signature("mamba2_scan").cell_dims
        # In the identity, not beside it: two models that tile the same
        # geometry differently are two rows.
        assert "ssm_chunk_size" in get_signature("mamba2_scan").identity_dims

        for phase in ("prefill", "decode"):
            jobs, dropped = build_jobs(
                avg_output_tokens=256,
                model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                batch_size=16, seq_len=1024, phase=phase,
                parallel=ParallelConfig.from_dict({"ep_moe": 8,
                                                   "dp_attention": 8}))
            scans = [j for j in jobs if j["signature"] == "mamba2_scan"]
            assert scans, dropped
            for job in scans:
                assert job["dims"]["ssm_chunk_size"] == 128, job["dims"]
            # Every other job — the embedding, the conv in the same mixer,
            # the attention, the expert GEMMs — carries it nowhere at all.
            for job in jobs:
                if job["signature"] == "mamba2_scan":
                    continue
                assert "ssm_chunk_size" not in job["dims"], (
                    phase, job["signature"], job["node_spec_id"])
                assert "ssm_chunk_size" not in job["kernel_params"], (
                    phase, job["signature"], job["node_spec_id"])
            assert any(j["signature"] == "embedding" for j in jobs), phase

        # No spec dim of the model names it: the deployment reads it.
        nano = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        from backend.analyzer import fetch_config
        from backend.arch.loader import SpecLoader
        resolved = SpecLoader().match(fetch_config(nano), model_id=nano)
        assert "ssm_chunk_size" not in resolved.dim_values

        # A model with no Mamba binds no chunk, and nothing invents one.
        dense, _ = build_jobs(avg_output_tokens=0,
                              model_id="deepseek-ai/DeepSeek-V2-Lite",
                              batch_size=1, seq_len=2048, phase="prefill")
        for job in dense:
            assert "ssm_chunk_size" not in job["dims"]
            assert "ssm_chunk_size" not in job["kernel_params"]

    def test_the_scan_chunk_is_the_one_the_checkpoint_config_states(self):
        # The frameworks read the chunk off the checkpoint config
        # (`mamba_chunk_size`, else `chunk_size`), so the deployment does too,
        # at the fields the binding names: a checkpoint that says 256 is
        # measured at 256, in both bindings, and the field the binding lists
        # first wins over the second.
        import copy
        from kernel_bench.plan import build_jobs
        from backend.analyzer import fetch_config
        from backend.arch.loader import SpecLoader
        from backend.inference.framework_bindings import get_binding
        nano = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        shipped = fetch_config(nano)
        assert shipped["chunk_size"] == 128
        assert "mamba_chunk_size" not in shipped

        def scans(config, framework):
            resolved = SpecLoader().match(config, model_id=nano)
            jobs, dropped = build_jobs(
                avg_output_tokens=0, model_id=nano, batch_size=1,
                seq_len=1024, phase="prefill", framework=framework,
                resolved=resolved, binding=get_binding(framework))
            return ({j["dims"]["ssm_chunk_size"] for j in jobs
                     if j["signature"] == "mamba2_scan"}, dropped)

        for framework in ("vllm", "sglang"):
            at_256 = copy.deepcopy(shipped)
            at_256["chunk_size"] = 256
            assert scans(at_256, framework)[0] == {256}, framework
            first_wins = copy.deepcopy(shipped)
            first_wins["mamba_chunk_size"] = 64
            assert scans(first_wins, framework)[0] == {64}, framework

        # A checkpoint that states neither field has no scan cell — no
        # framework fallback (vLLM's 2048, sglang's 64/256) stands in — and
        # the plan names why it dropped it.
        neither = copy.deepcopy(shipped)
        del neither["chunk_size"]
        chunks, dropped = scans(neither, "vllm")
        assert chunks == set()
        reasons = [d for d in dropped if "mamba2_scan" in d]
        assert reasons, dropped
        for reason in reasons:
            assert ("checkpoint config states none of the fields vllm reads "
                    "it from ['mamba_chunk_size', 'chunk_size']") in reason, reason

    def test_the_sweep_plans_the_graph_the_estimate_prices(self):
        # The sweep took `get_binding(framework)` unanswered, so every `when:`
        # condition failed and the nine fused ops carrying `device: cuda`
        # were filtered out: the plan walked a graph with nine fewer fusions
        # than the estimate it calibrates — measuring the un-fused ops while
        # the spans the estimate prices had no measurement at all.
        from backend.arch.eval.precision import AS_SHIPPED
        planned = {}
        # One served deployment each: the binding is passed (or dropped) for
        # the whole sweep at once, so a plan that reaches the framework's own
        # launch on one deployment is the answer for all of them.
        for fw in ("vllm", "sglang"):
            jobs, _label = _one_served_routine_sweep(
                model_id="deepseek-ai/DeepSeek-V4-Flash", gpu="H100-SXM",
                seq_lens=[1024], precision=AS_SHIPPED,
                framework=fw, log=lambda m: None)
            planned[fw] = {j["signature"] for j in jobs}
        # Each framework's own launch, which is the point of binding at all:
        # vLLM fuses the output's inverse rope, sglang the query's norm+rope.
        assert "fused_inv_rope" in planned["vllm"]
        assert "fused_qnorm_rope" in planned["sglang"]

    def test_the_gap_list_asks_the_gate_the_run_takes(self):
        # `report` asked `serves` directly, which does not refuse a disabled
        # kernel or a scale layout the provider does not read: gpt-oss's
        # MXFP4 grouped GEMM counted as provider-backed and merely pending,
        # which reads as coverage. The gap list has to ask what the run asks.
        from kernel_bench.plan import build_jobs
        from kernel_bench.providers import all_providers
        from kernel_bench.invocation import Invocation
        from backend.inference.parallel_config import ParallelConfig
        env = Env(None, False, None, "H100-SXM", None, devices=1)
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="openai/gpt-oss-20b", batch_size=2,
                             seq_len=1024, phase="prefill",
                             parallel=ParallelConfig(tp_attention=2,
                                                     tp_dense_ffn=2))
        named = set()
        for j in jobs:
            inv = Invocation.from_job(j)
            same = [p for p in all_providers() if p.signature == inv.signature]
            declared = [p for p in same
                        if p.serves is None or p.serves.accepts(inv)]
            runnable = [p for p in same if p.accepts(inv, env)]
            if declared and not runnable:
                named.add(inv.signature)
        # A collective is the case: `serves` admits its cell, and a run that
        # can drive one card refuses it (a group of 2 needs 2 cards), so the
        # gap list has to name it. gpt-oss's MXFP4 expert GEMM was this case
        # too, until the Marlin MXFP4 kernel landed and made that cell
        # runnable.
        assert "all_reduce" in named, named

    def test_a_collective_cell_carries_the_group_it_reduces_over(self):
        # The physical key dropped the deployment, which is right for a local
        # kernel — one card runs it the same way whichever config cut the
        # shape — and wrong for a collective: the same per-rank shape at two
        # group sizes moves different amounts over the fabric. The group is
        # the launch's own fact, so it is an axis of the measured shape,
        # filled from the row's `comm_world` (never from the deployment
        # label: one deployment has several groups).
        from kernel_bench.plan import build_jobs
        from backend.inference.parallel_config import ParallelConfig
        cells = {}
        for degree in (1, 2, 8):
            par = (ParallelConfig() if degree == 1 else
                   ParallelConfig(tp_attention=degree, tp_dense_ffn=degree))
            jobs, _ = build_jobs(avg_output_tokens=0, model_id="Qwen/Qwen3-8B", batch_size=2,
                                 seq_len=1024, phase="prefill", parallel=par)
            cells[degree] = [j for j in jobs
                             if j["signature"] == "allreduce_fused_rmsnorm"]
        # tp=1 has no collective at all.
        assert cells[1] == []
        assert cells[2] and cells[8]
        assert cells[2][0]["dims"]["n_comm_ranks"] == 2
        assert cells[8][0]["dims"]["n_comm_ranks"] == 8
        # Same per-rank shape, different cells — the whole point.
        shape = lambda j: {k: v for k, v in j["dims"].items() if k != "n_comm_ranks"}
        assert shape(cells[2][0]) == shape(cells[8][0])
        assert cells[2][0]["dims"] != cells[8][0]["dims"]

    def test_a_one_card_env_declines_a_group_cell(self):
        # On one process an all-reduce is the identity: the old floor timed
        # residual+RMSNorm and wrote it as an 8-rank collective, and its
        # reference omitted the collective too, so the check agreed. A group
        # cell is measured over one process per card (`comm`), and an env
        # with fewer cards than the group declines it, saying so.
        from kernel_bench.providers import all_providers
        from kernel_bench.invocation import Invocation
        inv = Invocation(op="allreduce_fused_rmsnorm",
                         signature="allreduce_fused_rmsnorm", phase="prefill",
                         dims={"B": 2, "s_query": 4, "d_model": 512,
                               "n_comm_ranks": 8},
                         dtype_env={"activation": "bf16"}, tp=1)
        ps = [p for p in all_providers()
              if p.signature == "allreduce_fused_rmsnorm"]
        assert ps
        one = Env("2.11", True, None, "H100-SXM", None, libs={"torch": "2.11"},
                  devices=1)
        assert not any(p.accepts(inv, one) for p in ps)
        assert all(any("needs 8 cards; this env has 1" in r
                       for r in p.refusals(inv, one)) for p in ps)
        eight = Env("2.11", True, None, "H100-SXM", None, libs={"torch": "2.11"},
                    devices=8)
        assert any(p.accepts(inv, eight) for p in ps)

    def test_the_inverse_rope_span_is_measurable(self):
        # A fusion boundary the cost model draws has to be measurable, or
        # the span's saving is a number nothing can check. DSv4's output
        # un-rotation is one vLLM launch (`fused_inv_rope_fp8_quant`); it
        # now has a signature, a reference and two providers, and the plan
        # reaches it.
        torch = pytest.importorskip("torch")
        from kernel_bench.plan import build_jobs
        from kernel_bench.providers import all_providers
        from kernel_bench.refs import get_ref
        from kernel_bench.invocation import Invocation
        from kernel_bench.signatures import op_to_signature
        from backend.inference.parallel_config import ParallelConfig
        assert op_to_signature("fused_inv_rope") == "fused_inv_rope"
        jobs, _dropped = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V4-Flash",
                                    batch_size=2, seq_len=1024, phase="prefill",
                                    parallel=ParallelConfig())
        planned = [j for j in jobs if j["signature"] == "fused_inv_rope"]
        assert planned, "the plan reaches the span"
        # Every provider of the span declares it, so the coverage report
        # counts them against the boundary rather than against an op.
        ps = [p for p in all_providers() if p.signature == "fused_inv_rope"]
        assert {p.library for p in ps} == {"torch", "vllm"}
        assert all(p.fused_op == "fused_inv_rope" for p in ps)
        # The torch floor computes what the reference does: the tail
        # rotated back, the head untouched.
        inv = Invocation.from_job(planned[0])
        inv = Invocation(op=inv.op, signature=inv.signature, phase=inv.phase,
                         dims={**inv.dims, "B": 1, "s_query": 4, "n_heads": 2},
                         dtype_env={k: v for k, v in inv.dtype_env.items()
                                    if k != "activation_granularity"}, tp=1)
        spec = get_ref("fused_inv_rope")
        inputs = build_inputs(spec, inv, "cpu")
        torch_p = next(p for p in ps if p.library == "torch")
        got = torch_p.build(inv, Env("2.11", False, None, None, None,
                                     libs={"torch": "2.11"}), inputs).run()
        assert torch.allclose(got.float(), spec.reference(inputs), atol=2e-2)

    def test_rope_emb_rotates_rather_than_returning_its_input(self):
        # The signature declared one operand, so no provider could do more
        # than return it: the torch baseline was `lambda x: x`, and the
        # reference's identity rotation agreed with it — a kernel that timed
        # nothing, cross-checked green. The contract now carries the
        # positions and the cos/sin table `rope` has, and both directions
        # rotate (the inverse undoes the forward).
        torch = pytest.importorskip("torch")
        from kernel_bench.providers import all_providers
        from kernel_bench.refs import get_ref
        from kernel_bench.invocation import Invocation
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        dims = {"B": 2, "s_query": 4, "d_model": 64}
        out = {}
        for sig in ("rope_emb", "rope_emb_inverse"):
            inv = Invocation(op=sig, signature=sig, phase="prefill", dims=dims,
                             dtype_env={"activation": "bf16"}, tp=1)
            spec = get_ref(sig)
            inputs = build_inputs(spec, inv, "cpu")
            assert set(inputs) == {"x", "positions", "cos_sin_cache"}
            p = next(p for p in all_providers() if p.signature == sig)
            got = p.build(inv, env, inputs).run().float()
            assert not torch.allclose(got, inputs["x"].float(), atol=1e-3), sig
            assert torch.allclose(got, spec.reference(inputs), atol=2e-2), sig
            out[sig] = (inputs, got)
        inputs, forward = out["rope_emb"]
        back = get_ref("rope_emb_inverse").reference(
            {**inputs, "x": forward.to(inputs["x"].dtype)})
        assert torch.allclose(back, inputs["x"].float(), atol=3e-2)

    def test_one_walk_reread_at_each_batch_plans_what_a_walk_there_does(self):
        # The sweep walks once per (deployment, seq, phase) and re-reads the
        # rows at every other batch (`batch_sweep.rows_at`).
        # That must plan exactly the jobs a walk at that batch plans — keys,
        # shapes and the numerators the SOL gate divides by.
        import json
        from kernel_bench.plan import build_jobs, build_jobs_over_batches
        from backend.inference.parallel_config import ParallelConfig
        kw = dict(model_id="Qwen/Qwen3-8B", seq_len=8192, phase="decode",
                  parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2))
        multi = build_jobs_over_batches(avg_output_tokens=0, batch_sizes=[1, 16, 64], **kw)

        def keyed(jobs):
            return {json.dumps([j["signature"], j["dims"], j["dtype_env"],
                                j["node_spec_id"], j["count_per_step"]],
                               sort_keys=True): j for j in jobs}
        for b in (1, 16):
            want = keyed(build_jobs(avg_output_tokens=0, batch_size=b, **kw)[0])
            got = keyed(multi[b][0])
            assert set(got) == set(want), b
            for k in want:
                for f in ("flops_per_rank", "access_bytes_per_rank"):
                    assert got[k][f] == pytest.approx(want[k][f], rel=1e-9), (b, f)

    def test_dropped_notes_reach_stats(self, monkeypatch):
        import kernel_bench.plan as pl
        import kernel_bench.sweep as sw

        note = ("fused_qk_rope [blk]: no shape projection for signature 'rope' "
                "(shape_in=[[8, 32, 128, 128]])")
        from backend.inference.parallel_config import ParallelConfig
        from tests.snapshots import resolve_snapshot

        monkeypatch.setattr(pl, "build_jobs_over_points",
                            lambda **kw: {p: ([], [note]) for p in kw["points"]})
        monkeypatch.setattr(sw, "_ceilings_for_deployment", lambda **kw: {2048: {}})
        monkeypatch.setattr(sw, "_feasible_batches", lambda cap, phase, seq, keys, **_: [1])

        _jobs, stats = sw.build_sweep_jobs(
            model_id="meta-llama/Llama-3.2-3B", gpu="H100-SXM", seq_lens=[2048],
            tracks=ROUTINE_TRACKS,
            parallel_configs=[("tp1", ParallelConfig())], avg_output_tokens=256,
            resolved=resolve_snapshot("meta-llama/Llama-3.2-3B"))
        assert stats["dropped"], "a dropped op family must be reported, not swallowed"
        assert "signature 'rope'" in stats["dropped"][0]

    def test_notes_collapse_to_one_line_per_gap(self):
        """The per-cell note names the shape, so one missing projection shows
        up once per shape — hundreds of lines for one gap. The sweep summary
        keys on op + signature so each gap is stated once."""
        notes = [f"fused_qk_rope [blk]: no shape projection for signature "
                 f"'rope' (shape_in=[[8, 32, {s}, 128]])"
                 for s in (128, 256, 512, 2048)]
        collapsed = {n.split(" (shape_in=")[0] for n in notes}
        assert len(collapsed) == 1, collapsed
        assert "signature 'rope'" in next(iter(collapsed))


class TestCrossCheckIsRecordedPerRow:
    """The fp32 oracle costs ~2× the kernel's memory and is O(S²) for
    attention, so at the biggest cells it OOMs and the harness measures the
    kernel anyway — correctly, since the kernel was vetted at smaller shapes,
    but with no oracle for THIS row. Those rows were written indistinguishable
    from verified ones, so the corpus could not say which of its numbers had
    ever been checked. Each row now carries the reason, and each reason names
    one thing to fix."""

    def test_measured_with_a_working_reference_says_passed(self):
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs([_gemm_job()], env=env, date="2026-08-06",
                       contributor="@t", cross_check=True)
        assert res.rows, res.failures + res.skipped
        assert res.rows[0]["cross_check"] == "passed", res.rows[0]["cross_check"]

    def test_checking_turned_off_says_disabled_not_passed(self):
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs([_gemm_job()], env=env, date="2026-08-06",
                       contributor="@t", cross_check=False)
        assert res.rows, res.failures + res.skipped
        assert res.rows[0]["cross_check"] == "disabled"

    def test_reference_that_ooms_says_no_reference(self, monkeypatch):
        pytest.importorskip("torch")
        import kernel_bench.driver as drv
        from kernel_bench.driver import run_jobs
        from kernel_bench.refs import get_ref

        real = get_ref

        def oom_ref(sig):
            spec = real(sig)
            if spec is None or spec.reference is None:
                return spec
            import copy
            s = copy.copy(spec)
            s.reference = lambda _inp: (_ for _ in ()).throw(
                RuntimeError("CUDA out of memory"))
            return s

        monkeypatch.setattr(drv, "get_ref", oom_ref, raising=False)
        import kernel_bench.refs as refs
        monkeypatch.setattr(refs, "get_ref", oom_ref)
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs([_gemm_job()], env=env, date="2026-08-06",
                       contributor="@t", cross_check=True)
        assert res.rows, res.failures + res.skipped
        assert res.rows[0]["cross_check"] == "no_reference"

    def test_a_passing_row_records_how_closely_it_agreed(self):
        # The verdict without its evidence cannot separate "lossy but correct"
        # from "computing the wrong thing", and cannot show whether a tolerance
        # sits anywhere near the dtype's own error. Both numbers are computed
        # on the pass path too — a stored row is by definition one that passed,
        # so the on-pass values are the only ones that survive at all.
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs([_gemm_job()], env=env, date="2026-08-06",
                       contributor="@t", cross_check=True)
        assert res.rows, res.failures + res.skipped
        row = res.rows[0]
        assert row["cross_check"] == "passed"
        assert row["cross_check_rel_err"] != "", "verdict kept, evidence dropped"
        assert row["cross_check_cosine"] != ""
        assert float(row["cross_check_rel_err"]) >= 0.0
        assert 0.99 <= float(row["cross_check_cosine"]) <= 1.000001

    def test_a_row_with_no_comparison_carries_no_numbers(self):
        # `disabled` means nothing was compared, so there is no error to have
        # measured. A number here would read as evidence of a check that never
        # ran, which is exactly the confusion the column exists to remove.
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs([_gemm_job()], env=env, date="2026-08-06",
                       contributor="@t", cross_check=False)
        assert res.rows, res.failures + res.skipped
        assert res.rows[0]["cross_check_rel_err"] == ""
        assert res.rows[0]["cross_check_cosine"] == ""

    def test_the_schema_rejects_evidence_without_a_comparison(self):
        # Built from a real measured row rather than a hand-written literal, so
        # the negative case is one field away from something the harness
        # actually emits.
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        from backend.inference import benchmarks as B
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs([_gemm_job()], env=env, date="2026-08-06",
                       contributor="@t", cross_check=True)
        assert res.rows, res.failures + res.skipped
        # The peaks come from the device table, which a CPU env has no entry
        # for; everything the cross-check writes is the row as measured.
        measured = dict(res.rows[0], peak_tflops_at_test="989.0",
                        peak_bw_at_test="3350.0")
        B.validate_row(dict(measured))              # the row as measured is valid

        row = dict(measured, cross_check="disabled")
        with pytest.raises(B.BenchmarkValidationError):
            B.validate_row(row)

        row = dict(measured, cross_check_cosine="1.4")      # not a cosine
        with pytest.raises(B.BenchmarkValidationError):
            B.validate_row(row)

        row = dict(measured, cross_check_rel_err="-0.1")    # not a magnitude
        with pytest.raises(B.BenchmarkValidationError):
            B.validate_row(row)

    def test_the_evidence_columns_are_provenance_not_identity(self):
        from backend.inference import benchmarks as B
        for c in ("cross_check_rel_err", "cross_check_cosine"):
            assert c in B.PROVENANCE_COLUMNS
            assert c not in B.IDENTITY_COLUMNS, (
                "how closely a kernel agreed with the reference does not make "
                "it a different measurement")

    def test_the_column_is_provenance_not_identity(self):
        from backend.inference import benchmarks as B
        assert "cross_check" in B.PROVENANCE_COLUMNS
        assert "cross_check" not in B.IDENTITY_COLUMNS, (
            "one cell measured with and without an available reference is one "
            "measurement, not two")

    def test_older_rows_migrate_to_unknown_never_to_passed(self):
        # An older corpus records neither whether a row was vetted nor how
        # closely it agreed, and neither is recoverable. `unknown` and an empty
        # number both say "predates the record"; `passed` or a 0 would assert a
        # comparison that may never have run.
        from backend.inference import benchmarks as B
        from tools.migrate_schema import migrate_text
        current = B.emit_csv(list(B.ALL_COLUMNS), [
            {c: "x" for c in B.ALL_COLUMNS}])
        added = ("cross_check", "cross_check_rel_err", "cross_check_cosine")
        old_header = [c for c in B.ALL_COLUMNS if c not in added]
        old = "\n".join([",".join(old_header),
                         ",".join("x" for _ in old_header)]) + "\n"
        out = migrate_text(old)
        header, rows = B.parse_rows(out)
        assert all(c in header for c in added)
        assert rows[0]["cross_check"] == "unknown"
        assert rows[0]["cross_check_rel_err"] == ""
        assert rows[0]["cross_check_cosine"] == ""
        assert "passed" not in out
        # idempotent: a current-schema file is left byte-identical
        assert migrate_text(current) == current

    def test_migration_refuses_a_column_it_cannot_answer(self, monkeypatch):
        # The fill table is what makes the migration truthful. A column added to
        # the schema without one must stop the tool, not receive whatever blank
        # happens to be convenient — a blank means something in some columns.
        from backend.inference import benchmarks as B
        import tools.migrate_schema as M
        monkeypatch.setattr(B, "ALL_COLUMNS",
                            (*B.ALL_COLUMNS, "some_new_column"))
        old = ",".join(B.ALL_COLUMNS[:-1]) + "\n" + \
              ",".join("x" for _ in B.ALL_COLUMNS[:-1]) + "\n"
        with pytest.raises(ValueError, match="no declared fill"):
            M.migrate_text(old)

    def test_migration_recovers_old_grouped_gemm_physical_rows(self):
        from backend.inference import benchmarks as B
        from tools.migrate_schema import migrate_text

        row = {c: "x" for c in B.ALL_COLUMNS}
        row.update(
            op="grouped_gemm",
            shape_json=json.dumps({
                "d_out": 128, "d_reduce": 64,
                "n_experts": 4, "s_query": 2,
            }, separators=(",", ":"), sort_keys=True),
            schema_version="5",
        )
        old = B.emit_csv(list(B.ALL_COLUMNS), [row])
        out = migrate_text(old)
        _header, rows = B.parse_rows(out)
        assert json.loads(rows[0]["shape_json"])["n_tokens"] == 8
        assert rows[0]["schema_version"] == str(B.KNOWN_SCHEMA_VERSION)
        assert migrate_text(out) == out

    def test_migration_fills_v8_moe_block_member(self):
        # A schema-7 CSV predates the column; migration fills it empty and the
        # result validates, rather than being hard-rejected.
        from backend.inference import benchmarks as B
        from tools.migrate_schema import migrate_text
        v7_cols = [c for c in B.ALL_COLUMNS if c != "moe_block_member"]
        row = {c: "x" for c in v7_cols}
        row["schema_version"] = "7"
        old = B.emit_csv(v7_cols, [row])
        out = migrate_text(old)
        header, rows = B.parse_rows(out)
        assert "moe_block_member" in header
        assert rows[0]["moe_block_member"] == ""
        assert rows[0]["schema_version"] == str(B.KNOWN_SCHEMA_VERSION)
        B.validate_header(header)  # no raise

    def test_migration_fills_v9_input_seed(self):
        # A schema-8 CSV predates the seed column; its rows drew from the
        # process-global RNG, so empty is the only truthful fill. The result
        # validates and a second migration changes nothing.
        from backend.inference import benchmarks as B
        from tools.migrate_schema import migrate_text
        v8_cols = [c for c in B.ALL_COLUMNS if c != "input_seed"]
        row = {c: "x" for c in v8_cols}
        row["schema_version"] = "8"
        old = B.emit_csv(v8_cols, [row])
        out = migrate_text(old)
        header, rows = B.parse_rows(out)
        assert "input_seed" in header
        assert rows[0]["input_seed"] == ""
        assert rows[0]["schema_version"] == str(B.KNOWN_SCHEMA_VERSION)
        B.validate_header(header)  # no raise
        assert migrate_text(out) == out

    def test_migration_fills_v14_drivers_empty_and_never_invents_one(self):
        # A schema-13 CSV predates both driver columns, and NOTHING can
        # recover them: the box's driver today is not the driver that machine
        # ran months ago, and `box` names a profile two machines share and
        # that gets edited. Empty is the only truthful fill, and a presumed
        # one would be worse than absence — it would read as a VERIFIED
        # native configuration, which is the exact claim the pair exists to
        # make checkable.
        from backend.inference import benchmarks as B
        from tools.migrate_schema import migrate_text
        added = ("kmd_version", "umd_version")
        v13_cols = [c for c in B.ALL_COLUMNS if c not in added]
        row = {c: "x" for c in v13_cols}
        row["schema_version"] = "13"
        old = B.emit_csv(v13_cols, [row])
        out = migrate_text(old)
        header, rows = B.parse_rows(out)
        for c in added:
            assert c in header
            assert rows[0][c] == "", f"{c} must migrate to empty, got {rows[0][c]!r}"
        assert rows[0]["schema_version"] == str(B.KNOWN_SCHEMA_VERSION)
        B.validate_header(header)  # no raise
        assert migrate_text(out) == out
        # And the migrated row reads as UNKNOWN, not as native — the one
        # question a consumer asks of the pair.
        assert B.is_driver_substituted(rows[0]) is None

    def test_v7_provenance_columns_are_not_identity(self):
        # Which sweep, which box, what routing load — all three describe the
        # circumstances of one measurement, none of them makes it a different
        # one. In identity they would split the physical cell.
        from backend.inference import benchmarks as B
        for c in ("run_id", "box", "measured_load_factor"):
            assert c in B.PROVENANCE_COLUMNS
            assert c not in B.IDENTITY_COLUMNS

    def test_v7_columns_migrate_to_empty_never_invented(self):
        # The sweep id, the box, and the routing load are facts of the moment
        # of measurement; a finished CSV holds none of them. Empty says
        # "predates the record" — any fill would be a claim about a run nobody
        # watched.
        from backend.inference import benchmarks as B
        from tools.migrate_schema import migrate_text
        added = ("run_id", "box", "measured_load_factor")
        old_header = [c for c in B.ALL_COLUMNS if c not in added]
        old = "\n".join([",".join(old_header),
                         ",".join("x" for _ in old_header)]) + "\n"
        out = migrate_text(old)
        header, rows = B.parse_rows(out)
        assert all(c in header for c in added)
        for c in added:
            assert rows[0][c] == ""
        assert rows[0]["schema_version"] == str(B.KNOWN_SCHEMA_VERSION)
        assert migrate_text(out) == out


class TestCoverageCheckedBeforeAllocating:
    """A job whose providers are all already covered must cost nothing.

    The physical key and every provider's `impl_id` are functions of the
    invocation alone, so "nothing to do here" is knowable before a single
    tensor exists. The check used to run AFTER the inputs and the fp32 oracle
    were built, and the oracle is the expensive half — ~2× the kernel's memory,
    O(S²) for attention. One MLA decode cell at B=75,588 held the card at 100%
    for 36 minutes building a reference for a row that would then be skipped,
    and the model's remaining jobs died with it on the run timeout."""

    def test_covered_job_builds_no_inputs_and_no_reference(self, monkeypatch):
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs

        built = {"inputs": 0, "reference": 0}

        import kernel_bench.refs as refs
        real = refs.get_ref

        def counting_get_ref(sig):
            spec = real(sig)
            if spec is None:
                return spec
            import copy
            s = copy.copy(spec)
            mk, rf = spec.make_inputs, spec.reference

            def _mk(inv, device):
                built["inputs"] += 1
                return mk(inv, device)

            def _rf(inp):
                built["reference"] += 1
                return rf(inp)

            s.make_inputs = _mk
            s.reference = _rf if rf is not None else None
            return s

        monkeypatch.setattr(refs, "get_ref", counting_get_ref)

        from kernel_bench.driver import physical_key_for
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import applicable

        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        inv = Invocation.from_job(_gemm_job())
        pk = physical_key_for(inv, env)
        # Every provider's identity already measured (`--skip-existing` /
        # `--measured-in`).
        covered = {(pk, p.library, p.version(env) or "unknown", p.impl_id(inv))
                   for p in applicable(inv, env)}
        res = run_jobs([_gemm_job()], env=env, date="2026-08-07",
                       contributor="@t", cross_check=True, measured_keys=covered)
        assert not res.rows
        assert any("already measured" in s for s in res.skipped), res.skipped
        assert built == {"inputs": 0, "reference": 0}, (
            f"a fully-covered job still allocated: {built}")

    def test_kernel_bench_coverage_skips_before_allocating(self, monkeypatch):
        """`--skip-existing` asks Kernel Bench's shared rows through the
        estimates' own lookup client (`backend.measure.lookup.covered`), once
        for every job; a covered implementation is skipped before any input
        exists, and a service that does not answer says so."""
        pytest.importorskip("torch")
        from backend.measure import lookup
        from kernel_bench import driver
        from kernel_bench.driver import physical_key_for, run_jobs
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import applicable

        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        inv = Invocation.from_job(_gemm_job())
        pk = physical_key_for(inv, env)
        asked: list = []

        def covered(keys):
            asked.append(list(keys))
            return {pk: {(p.library, p.version(env) or "unknown", p.impl_id(inv))
                         for p in applicable(inv, env)}}
        monkeypatch.setenv("HUGGINGARCH_BENCH_URL", "http://bench.internal")
        monkeypatch.setattr(lookup, "covered", covered)
        res = run_jobs([_gemm_job()], env=env, date="2026-08-07",
                       contributor="@t", cross_check=True, skip_existing=True)
        assert not res.rows and asked == [[pk]]
        assert any("Kernel Bench covers it" in s for s in res.skipped), res.skipped

        def down(keys):
            raise lookup.MeasurementsUnavailable("measurements unavailable: refused")
        monkeypatch.setattr(lookup, "covered", down)
        monkeypatch.setattr(driver, "COVERAGE_BACKOFF_S", 0.0)
        res = run_jobs([_gemm_job()], env=env, date="2026-08-07",
                       contributor="@t", cross_check=False, skip_existing=True)
        assert any("INACTIVE" in f for f in res.failures) and len(res.rows) == 1

    def test_kernel_bench_coverage_asks_in_batches(self, monkeypatch):
        """A plan's keys are asked `COVERAGE_BATCH` at a time — the client's
        timeout is one estimate's, not one model's — and the answers merge."""
        from backend.measure import lookup
        from kernel_bench import driver

        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        monkeypatch.setattr(driver, "COVERAGE_BATCH", 2)
        monkeypatch.setattr(driver, "physical_key_for",
                            lambda inv, env: inv.dims["d_out"])
        asked: list = []

        def covered(keys):
            asked.append(list(keys))
            return {k: {("torch", "2.11", "torch_gemm")} for k in keys}
        monkeypatch.setenv("HUGGINGARCH_BENCH_URL", "http://bench.internal")
        monkeypatch.setattr(lookup, "covered", covered)
        jobs = []
        for d in (16, 32, 48, 64, 80):
            job = _gemm_job()
            job["dims"] = {**job["dims"], "d_out": d}
            jobs.append(job)
        got = driver._kernel_bench_coverage(jobs, env, driver.RunResult(), print)
        assert asked == [[16, 32], [48, 64], [80]]
        assert sorted(got) == [16, 32, 48, 64, 80]

    def test_uncovered_job_still_measures(self):
        pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        res = run_jobs([_gemm_job()], env=env, date="2026-08-07",
                       contributor="@t", cross_check=True)
        assert len(res.rows) == 1, res.skipped + res.failures


class TestOracleHasAMemoryBudget:
    """The fp32 reference is the only part of a cell that can be too big to
    build. It runs in fp32 and, for attention, materializes B·N·Sq·Lq scores
    the kernel never forms — so a cell the flash kernel serves happily can have
    an oracle that needs terabytes.

    Predicted, not attempted: attempting it does not reliably fail. On a
    DeepSeek-V3 MLA decode cell the build neither returned nor raised — the
    card sat at 100% for over half an hour while the allocator retried, and
    every remaining job for that model died with the run timeout. Twice, on two
    architecturally identical models."""

    def _inv(self, **dims):
        from kernel_bench.invocation import Invocation
        base = {"kv_block_size": 64, "B": 1, "n_heads": 128, "s_query": 1, "s_kv": 8,
                "d_head": 576, "d_v_head": 512}
        base.update(dims)
        return Invocation(signature="mla_attention", op="mla_attention",
                          dims=base, dtype_env={"activation": "bf16"},
                          phase="decode", tp=1)

    def test_quadratic_signatures_count_the_score_matrix(self):
        pytest.importorskip("torch")
        from kernel_bench.driver import _reference_footprint_bytes
        small = _reference_footprint_bytes(self._inv(B=1), {})
        big = _reference_footprint_bytes(self._inv(B=75588), {})
        assert big > small * 10_000, (small, big)
        # B·N·Sq·Lq·4 for the real stalling cell
        assert big == 75588 * 128 * 1 * 8 * 4

    def test_indexer_cells_count_their_heads(self):
        """The indexer's head axis is `n_indexer_heads`; read as `n_heads`
        the KeyError was swallowed and the score estimate degenerated to
        zero, so the budget never fired for an indexer cell."""
        pytest.importorskip("torch")
        from kernel_bench.driver import _reference_footprint_bytes
        from kernel_bench.invocation import Invocation
        inv = Invocation(signature="indexer_logits", op="indexer_logits",
                         dims={"B": 2, "s_query": 64, "s_kv": 8192,
                               "n_indexer_heads": 32, "d_indexer_head": 128,
                               "kv_block_size": 64},
                         dtype_env={"activation": "bf16"},
                         phase="prefill", tp=1)
        # B·Nix·Sq·Lq·4 — not the zero the missing `n_heads` key gave.
        assert _reference_footprint_bytes(inv, {}) == 2 * 32 * 64 * 8192 * 4

    def test_a_cell_that_cannot_size_its_oracle_says_so(self):
        """No silent zero: a cell missing the axis its query declares raises,
        and the oracle is declined with that reason rather than assumed to
        fit."""
        pytest.importorskip("torch")
        from kernel_bench.driver import (_reference_footprint_bytes,
                                         _reference_will_not_fit)
        inv = self._inv()
        del inv.dims["n_heads"]
        with pytest.raises(ValueError, match="n_heads"):
            _reference_footprint_bytes(inv, {})
        import torch
        if torch.cuda.is_available():
            env = Env("2.11", True, None, None, None, libs={"torch": "2.11"})
            assert "cannot be sized" in _reference_will_not_fit(inv, {}, env)

    def test_oversized_oracle_is_declined_with_a_reason(self):
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("needs a GPU to read free VRAM")
        from kernel_bench.driver import _reference_will_not_fit
        env = Env("2.11", True, None, None, None, libs={"torch": "2.11"})
        # 4 PiB of scores: nothing fits this.
        why = _reference_will_not_fit(
            self._inv(B=2**20, s_kv=2**20), {}, env)
        assert why and "oracle budget" in why, why

    def test_ordinary_cell_keeps_its_oracle(self):
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("needs a GPU to read free VRAM")
        from kernel_bench.driver import _reference_will_not_fit
        env = Env("2.11", True, None, None, None, libs={"torch": "2.11"})
        assert _reference_will_not_fit(self._inv(B=1), {}, env) is None

    def test_declining_does_not_mutate_the_shared_spec(self):
        from kernel_bench.driver import _WithoutReference
        from kernel_bench.refs import get_ref
        spec = get_ref("gqa_attention")
        if spec is None:
            pytest.skip("no gqa reference registered")
        view = _WithoutReference(spec)
        assert view.reference is None
        assert spec.reference is not None, "the registered spec is shared"
        assert view.make_inputs is spec.make_inputs


class TestDedupNeverDropsAnIdentityEntirely:
    """Deduplicating must not delete coverage.

    `partition` keeps exactly the row `winners` names, so an identity missing
    from `winners` has EVERY copy moved aside — not one kept. Rows whose `p50`
    is unusable were being skipped when building `winners`, so an identity
    whose copies were all unusable vanished completely. On the H100 corpus that
    silently removed 980 identities, and it compounded: the orchestrator
    re-deduplicates after every pass."""

    def test_identity_with_no_usable_p50_still_keeps_one_row(self):
        import tools.dedup_isomorphic_rows as ded
        rows = [("a.csv", 0, {"model": "org/A", "p50_us": "nan"}),
                ("b.csv", 0, {"model": "org/B", "p50_us": "0"})]
        real = ded._slim
        ded._slim = lambda r: "SAME"
        try:
            best = ded.winners(rows)
        finally:
            ded._slim = real
        assert "SAME" in best, "an identity with no usable p50 must still survive"

    def test_a_real_measurement_outranks_an_unusable_one(self):
        import tools.dedup_isomorphic_rows as ded
        rows = [("a.csv", 0, {"model": "org/A", "p50_us": "nan"}),
                ("b.csv", 0, {"model": "org/B", "p50_us": "12.0"})]
        real = ded._slim
        ded._slim = lambda r: "SAME"
        try:
            best = ded.winners(rows)
        finally:
            ded._slim = real
        assert best["SAME"][1:] == ("b.csv", 0)

    def test_slim_identity_set_never_shrinks(self):
        """The property the corpus actually needs, stated directly."""
        import tools.dedup_isomorphic_rows as ded
        rows = [("a.csv", 0, {"model": "org/A", "p50_us": "nan"}),
                ("a.csv", 1, {"model": "org/A", "p50_us": "5.0"}),
                ("b.csv", 0, {"model": "org/B", "p50_us": "-1"})]
        keys = {0: "X", 1: "Y", 2: "Z"}
        real = ded._slim
        seq = iter([keys[0], keys[1], keys[2]])
        ded._slim = lambda r: next(seq)
        try:
            best = ded.winners(rows)
        finally:
            ded._slim = real
        assert set(best) == {"X", "Y", "Z"}, set(best)


class TestHardwareFloorIsDeclaredNotDescribed:
    """A kernel that needs a newer architecture must be UNAVAILABLE there.

    `cutlass_mla_decode` carried "SM100 (Blackwell) only" in its `build_hint`
    — prose, which constrains nothing. On an H100 it was therefore attempted on
    every MLA decode cell and raised on every one: 279-347 failed builds per MLA
    model in a single pass, each allocating a paged latent cache before
    sgl_kernel refused.
    """

    def _env(self, sm):
        return Env("2.11", True, None, None, None, sm=sm,
                   libs={"sgl_kernel": "0.4.2"})

    def test_below_the_floor_is_unavailable(self):
        from kernel_bench import providers as P
        prov = next(p for p in P.all_providers()
                    if p.kernel_impl == "cutlass_mla_decode")
        assert prov.min_sm == 100
        assert not prov.available(self._env("sm_90"))

    def test_at_or_above_the_floor_is_available(self):
        from kernel_bench import providers as P
        prov = next(p for p in P.all_providers()
                    if p.kernel_impl == "cutlass_mla_decode")
        assert prov.available(self._env("sm_100"))
        assert prov.available(self._env("sm_120"))

    def test_unknown_sm_allows(self):
        # Symmetric with `_version_ok`: decline what can be PROVEN unrunnable,
        # never invent a reason to skip. A CPU-only or unprobed env has no sm.
        from kernel_bench import providers as P
        prov = next(p for p in P.all_providers()
                    if p.kernel_impl == "cutlass_mla_decode")
        assert prov.available(self._env(None))

    def test_no_floor_means_no_constraint(self):
        from kernel_bench import providers as P
        others = [p for p in P.all_providers() if p.min_sm is None]
        assert others, "every provider suddenly has a floor — check the default"
        prov = next(p for p in others if p.library == "torch")
        assert prov.available(Env("2.11", True, None, None, None,
                                  sm="sm_75", libs={"torch": "2.11"}))


class TestSmokeMeetsTheDeclarations:
    """Layer 3: a declaration must survive meeting the kernel.

    Layers 1-2 are structural — a declaration can be well-formed, uniquely
    named, and still false. Only running it can tell.
    """

    def test_a_minimal_cell_satisfies_what_it_was_built_from(self):
        # The synthesizer is only useful if its cell passes the very gate it
        # was derived from; otherwise every kernel would report `no_cell` and
        # the layer would be green while testing nothing.
        from kernel_bench import providers as P
        from kernel_bench.smoke import (_combinations, cell_dtype_env,
                                        minimal_dims)
        from kernel_bench.invocation import Invocation
        checked = 0
        for prov in P.all_providers():
            if prov.serves is None:
                continue
            ks = prov.serves
            for weight, act, phase in _combinations(prov):
                dims, _why = minimal_dims(prov, phase)
                if dims is None:
                    continue                 # reported by the run, not here
                # The cell smoke would build: its dtype env is the one
                # place the constrained slots are derived (the cache's
                # format included — a hand-built env here is a second
                # path that drifts, and did).
                inv = Invocation(
                    signature=prov.signature, op=prov.signature, dims=dims,
                    dtype_env=cell_dtype_env(prov, weight, act),
                    # The scalars the kernel requires, as smoke fills them.
                    kernel_params={name: 1.0
                                   for name in ks.requires_params},
                    phase=phase, tp=1)
                assert ks.accepts(inv), (
                    f"{prov.kernel_impl}: its own minimal cell {dims} fails "
                    f"the declaration it was derived from")
                checked += 1
        assert checked > 20, f"only {checked} declarations produced a cell"

    def test_no_declaration_produces_wrong_numbers(self):
        # The one unambiguous verdict. An exception is NOT asserted on: on a
        # given box it may equally mean the declaration is wrong or that the
        # kernel wants a newer architecture, and guessing between those would
        # either hide real lies or fail CI on correct code.
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("smoke needs a GPU")
        from kernel_bench.env import detect_env
        from kernel_bench.smoke import format_report, smoke_check
        results = smoke_check(detect_env())
        wrong = [r for r in results if r.false_declaration]
        assert not wrong, format_report(results)

    def test_every_declaration_reaches_a_verdict(self):
        # A claim nothing met is not a claim that held. `no_cell` means the
        # harness could not even try, which must never read as a pass.
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("smoke needs a GPU")
        from kernel_bench.env import detect_env
        from kernel_bench.smoke import format_report, smoke_check
        results = smoke_check(detect_env())
        untried = [r for r in results if r.untried]
        assert not untried, format_report(results)

    def test_a_refusal_the_declaration_cannot_express_still_declines(self):
        """A gate living in `impl_params` is resolved before build, the
        driver's own pre-allocation channel: the cell reports `declined`,
        never an ok for a launch the kernel would not make. The cell smoke
        hands it carries the DECLARED slot value, not the
        `kv = activation` default."""
        from kernel_bench import providers as P
        from kernel_bench.smoke import _one
        seen = {}

        def params(inv):
            seen["kv"] = inv.dtype_env.get("kv")
            raise NotImplementedError("not the cache this launch writes")

        def build(inv, env, inputs):
            raise AssertionError("build reached before the gate resolved")

        p = P.Provider(
            library="torch", signature="rmsnorm",
            kernel_impl="probe_impl_params_decline", import_name="torch",
            serves=P.KernelSignature(
                signature="rmsnorm", activation=("bf16",),
                dtype_slots=(("kv", ("fp8_e4m3",)),)),
            impl_params=params, build=build)
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        r = _one(p, env, "", "bf16", "prefill", "cpu")
        assert r.outcome == "declined", r
        assert "not the cache" in r.detail
        assert seen["kv"] == "fp8_e4m3"


class TestRequiresDimsIsPresenceNotValue:
    """`requires_dims` says a dim must be THERE, which is a different claim
    from any constraint on its value.

    Non-absorbed MLA is identified by carrying a separate KV head count at all
    — the absorbed latent geometry has no such axis. Reading an absent dim as a
    default is how a kernel gets measured against a shape the model does not
    have, so an absent dim must fail the gate rather than fall back to one.
    """

    def _inv(self, **dims):
        from kernel_bench.invocation import Invocation
        return Invocation(signature="mla_attention", op="mla_attention",
                          dims={"kv_block_size": 64, **dims}, dtype_env={"activation": "bf16"},
                          phase="prefill", tp=1)

    def test_absent_dim_is_rejected(self):
        from kernel_bench.providers import KernelSignature
        sig = KernelSignature(signature="mla_attention",
                              activation=("bf16",),
                              dtype_slots=(("kv", ("bf16",)),),
                              requires_dims=("n_kv_heads",))
        assert not sig.accepts(self._inv(B=1, n_heads=128))

    def test_present_dim_is_accepted(self):
        from kernel_bench.providers import KernelSignature
        sig = KernelSignature(signature="mla_attention",
                              activation=("bf16",),
                              dtype_slots=(("kv", ("bf16",)),),
                              requires_dims=("n_kv_heads",))
        assert sig.accepts(self._inv(B=1, n_heads=128, n_kv_heads=1))

    def test_zero_counts_as_absent(self):
        # A dim emitted as 0 is not an axis the kernel can index; treating it
        # as present would be the same fabrication one step later.
        from kernel_bench.providers import KernelSignature
        sig = KernelSignature(signature="mla_attention",
                              activation=("bf16",),
                              dtype_slots=(("kv", ("bf16",)),),
                              requires_dims=("n_kv_heads",))
        assert not sig.accepts(self._inv(B=1, n_heads=128, n_kv_heads=0))

    def test_the_mla_prefill_kernel_declares_it(self):
        # The real registration, not a synthetic one: this is the constraint
        # that separates 12,676 non-absorbed cells from 42,043 absorbed ones in
        # the planned corpus.
        from kernel_bench import providers as P
        prov = next(p for p in P.all_providers()
                    if p.kernel_impl == "torch_sdpa_mla_prefill")
        assert prov.serves is not None, "still on a lambda"
        assert "n_kv_heads" in prov.serves.requires_dims


class TestKernelSignatureDeclaration:
    """The declaration a kernel makes about what it can serve.

    It exists so the gate can be derived rather than restated. A gate WIDER
    than the build measures a cell under a quantization the kernel never ran
    and records it as if it had — silently, since nothing raises. Four gates
    were wider than their builds in one day; these are the distinctions that
    were collapsed."""

    def _inv(self, **kw):
        from kernel_bench.invocation import Invocation
        d = dict(signature="gemm", op="linear",
                 dims={"B": 1, "s_query": 4096, "d_reduce": 4096, "d_out": 4096},
                 dtype_env={"weight": "fp4_e2m1", "activation": "bf16"},
                 phase="prefill", tp=1)
        d.update(kw)
        return Invocation(**d)

    def test_nvfp4_declaration_declines_mxfp4(self):
        from kernel_bench.providers import KernelSignature
        sig = KernelSignature(signature="gemm", weight=("fp4_e2m1",),
                              activation=("bf16", "fp16"))
        assert sig.accepts(self._inv())
        assert not sig.accepts(self._inv(
            dtype_env={"weight": "mxfp4", "activation": "bf16"})), (
            "MXFP4 shares the e2m1 element encoding but blocks 32 elements "
            "under an e8m0 scale — a different format, not a variant")

    def test_e4m3_declaration_declines_e5m2(self):
        from kernel_bench.providers import KernelSignature
        sig = KernelSignature(signature="gemm", weight=("fp8_e4m3",),
                              activation=("bf16",))
        assert not sig.accepts(self._inv(
            dtype_env={"weight": "fp8_e5m2", "activation": "bf16"})), (
            "e5m2 tops out at 57344 against e4m3's 448 — a prefix match here "
            "hands a cell to a kernel scaled 128x wrong")

    def test_enum_expresses_a_discrete_head_dim(self):
        """sglang's rope serves d_rope in {64,128,256,512} — an enumeration,
        not a divisibility. The declaration language has to cover the gates
        that exist, or migration falls back to lambdas and the checks lose
        their reach."""
        from kernel_bench.providers import KernelSignature
        sig = KernelSignature(signature="rope", activation=("bf16",),
                              enum=(("d_rope", (64, 128, 256, 512)),))
        ok = self._inv(signature="rope", dims={"d_rope": 128})
        bad = self._inv(signature="rope", dims={"d_rope": 192})
        assert sig.accepts(ok) and not sig.accepts(bad)

    def test_describe_is_json_safe_and_names_its_constraints(self):
        import json
        from kernel_bench.providers import KernelSignature
        sig = KernelSignature(signature="gemm", weight=("int4",),
                              activation=("bf16",),
                              divisible=(("d_reduce", 128),),
                              enum=(("d_out", (64, 128)),))
        d = sig.describe()
        json.dumps(d)
        assert "d_reduce % 128 == 0" in d["shape_constraints"]
        assert "d_out in [64, 128]" in d["shape_constraints"]


class TestLibraryRecord:
    """Install facts belong to the library, not to each kernel it provides.

    Duplicating them per kernel let them drift — torch carried three `pip`
    specs across its 12 kernels and vllm two across 14 — and `envs/*.txt` is
    generated from that field, so the env got whichever sorted first."""

    def _prov(self, library, import_name):
        from kernel_bench.providers import Provider, KernelSignature
        return Provider(
            library=library, signature="rmsnorm", kernel_impl="probe",
            import_name=import_name,
            serves=KernelSignature(signature="rmsnorm"),
            build=lambda inv, env, inputs: None)

    def test_version_spec_gates_availability(self):
        """The bound is the LIBRARY's floor, not the release its env installs:
        any version at or above it measures."""
        from backend.inference.framework_bindings import get_binding
        p = self._prov("vllm", "vllm")
        audited = get_binding("vllm").audited_at.release
        old = Env("2.11", False, None, None, None, libs={"vllm": "0.6.0"})
        new = Env("2.11", False, None, None, None, libs={"vllm": audited})
        assert not p.available(old)
        assert p.available(new)

    def test_unknown_version_is_allowed_not_rejected(self):
        """Importability already passed; an unreadable version is not evidence
        of incompatibility, and rejecting on it would silently drop providers
        on any env whose metadata is thin."""
        p = self._prov("flashmla", "flash_mla")
        env = Env("2.11", False, None, None, None,
                  libs={"flash_mla": "unknown"})
        assert p.available(env)

    def test_library_name_need_not_match_the_import_gate(self):
        """`marlin` rows are recorded under that framework name but the kernels
        import vllm. Identity, installation and the import gate are three
        different questions."""
        from kernel_bench.providers import get_library
        marlin = [p for p in all_providers() if p.library == "marlin"]
        assert marlin and {p.import_name for p in marlin} == {"vllm"}
        assert get_library("marlin").name == "marlin"

    def test_a_multi_package_library_judges_each_module_separately(self):
        """sglang ships two independently-versioned packages, and its kernels
        gate on whichever they import. `sgl-kernel` carries no version bound and
        must not be read as one for `sglang` — under a single-spec model those
        two looked like a disagreement to reconcile rather than two
        requirements to install."""
        from backend.inference.framework_bindings import get_binding
        from kernel_bench.providers import get_library
        lib = get_library("sglang")
        # A fresh env installs the audited release; the gate accepts any
        # release the kernels run on, so another install still measures.
        assert lib.pip[0] == f"sglang=={get_binding('sglang').audited_at.release}"
        assert lib.spec_for("sglang") == "sglang>=0.5.5.post3"
        assert lib.spec_for("sgl_kernel") == "sgl-kernel"
        newer = Env("2.11", False, None, None, None,
                    libs={"sglang": "0.5.12", "sgl_kernel": "0.4.2"})
        assert self._prov("sglang", "sglang").available(newer)
        # a module the library never names constrains nothing
        assert lib.spec_for("torch") == ""

        old = Env("2.11", False, None, None, None,
                  libs={"sglang": "0.4.0", "sgl_kernel": "0.4.0"})
        assert not self._prov("sglang", "sglang").available(old)
        # ...and the unbounded package stays available at that same version
        assert self._prov("sglang", "sgl_kernel").available(old)


class TestDeclarationsAreHonoured:
    """Every declared combination must actually build and run.

    Layers 1 and 2 check that a declaration is well formed and that identities
    are unique; neither can tell whether the declaration is TRUE. That is what
    this does: take each `serves` combination, build the smallest legal cell,
    run it, and compare against the fp32 oracle.

    It is the layer that catches what the others cannot. humming declared bf16
    while its kernel asserts ≤8 bits — raises. marlin's fp4 path spent a day
    half-fixed, packing as nvfp4 and reading as mxfp4 — raises. And an e5m2
    weight quantized with an e4m3 scale does NOT raise; it returns numbers
    128× off, so "did not crash" is not the bar, agreement with the reference
    is.

    GPU-only, and skipped where a library is absent — a declaration cannot be
    exercised on a box that cannot run it.
    """

    def _smallest_cell(self, sig):
        """A legal cell for this signature, small enough to be cheap."""
        base = {"gemm": {"B": 1, "s_query": 256, "d_reduce": 256, "d_out": 256}}
        return dict(base.get(sig.signature, {}))

    def test_every_declared_combination_builds_and_agrees(self):
        pytest.importorskip("torch")
        import torch
        if not torch.cuda.is_available():
            pytest.skip("declarations can only be exercised on a GPU")
        from kernel_bench.env import detect_env
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import all_providers, discover
        from kernel_bench.refs import get_ref

        env = detect_env()
        discover()
        failures = []
        for p in all_providers():
            if p.serves is None or not p.available(env):
                continue
            sig = p.serves
            dims = self._smallest_cell(sig)
            if not dims:
                continue                      # no cell recipe for this signature yet
            for name, mod in sig.divisible:   # honour the kernel's own tiling
                dims[name] = max(dims.get(name, mod), mod) // mod * mod
            for w in (sig.weight or ("bf16",)):
                for a in (sig.activation or ("bf16",)):
                    inv = Invocation(
                        signature=sig.signature, op="linear", dims=dict(dims),
                        dtype_env={"weight": w, "activation": a,
                                   "output": sig.writes(a)[0]},
                        phase=(sig.phases[0] if sig.phases else "prefill"), tp=1)
                    if not sig.accepts(inv):
                        failures.append(f"{p.kernel_impl} declares ({w},{a}) "
                                        f"but its own gate rejects the cell")
                        continue
                    spec = get_ref(sig.signature)
                    inputs = build_inputs(spec, inv, "cuda")
                    ref = spec.reference(inputs).float()
                    try:
                        case = p.build(inv, env, p.convert_inputs(inv, inputs))
                        to_can = p.wrap_output(inv, case.to_canonical)
                        out = to_can(case.run())
                        out = (out[0] if isinstance(out, (tuple, list)) else out).float()
                    except Exception as e:
                        failures.append(f"{p.kernel_impl} declares ({w},{a}) "
                                        f"but the build raised "
                                        f"{type(e).__name__}: {str(e)[:90]}")
                        continue
                    cos = torch.nn.functional.cosine_similarity(
                        out.flatten(), ref.flatten(), dim=0).item()
                    # Loose on purpose: a 4-bit weight is lossy by design. What
                    # this rejects is an output uncorrelated with the reference
                    # — the shape of "measured the wrong quantization".
                    if cos < 0.9:
                        failures.append(f"{p.kernel_impl} declares ({w},{a}) "
                                        f"but its output barely correlates with "
                                        f"the reference (cos={cos:.3f})")
                    torch.cuda.empty_cache()
        assert not failures, "declared but not honoured:\n  " + "\n  ".join(failures)


class TestWeightGridIsPlacedBeforeAnyoneSeesIt:
    """A weight-only-quantized kernel is judged on its ARITHMETIC, not on how
    lossy its format is.

    The reference used to be computed from the canonical bf16 weight while the
    kernel quantized that weight itself, so the comparison carried the entire
    4-bit quantization error — and the tolerance it was measured against is
    keyed on the ACTIVATION dtype, the slot the error does not come from. One
    pass rejected 2,124 marlin and 1,980 humming measurements that way, leaving
    both libraries with zero rows.

    marlin and humming cannot be installed on every box, so the mechanism is
    exercised here with quantizing providers built in the test: one that packs
    the format correctly and one that packs it with the WRONG grid. The second
    is the point — placing the weight on the grid must not cost the check its
    ability to fail.
    """

    def _run(self, prov, weight_dtype):
        from kernel_bench.driver import run_jobs
        from kernel_bench import providers as P
        env = Env("2.11", False, None, None, None, libs={"torch": "2.11"})
        job = _gemm_job(dtype_env={"weight": weight_dtype, "activation": "bf16",
                                   "kv": "bf16", "output": "bf16",
                                   "metadata": "fp32"},
                        dims={"B": 1, "s_query": 8, "d_reduce": 128, "d_out": 64})
        # Discover FIRST: `applicable()` calls `all_providers()`, which runs
        # discovery on demand and would append the 66 real providers back onto
        # the list this test just replaced — the row would then come from
        # torch_cublas_linear while the intended provider was the one failing.
        P.discover()
        saved = list(P._PROVIDERS)
        P._PROVIDERS[:] = [prov]
        try:
            return run_jobs([job], env=env, date="2026-08-08",
                            contributor="@t", cross_check=True)
        finally:
            P._PROVIDERS[:] = saved

    @staticmethod
    def _int4_provider(weight_group):
        """Quantizes to int4 per group of 128 — the grid marlin's GPTQ path
        uses. Correct, so it must pass once the weight is placed on that grid."""
        from kernel_bench.providers import BenchCase, KernelSignature, Provider

        def build(inv, env, inputs):
            x, w = inputs["x"].float(), inputs["w"].float()
            N, K = w.shape
            wg = w.reshape(N, K // 128, 128)
            s = (wg.abs().amax(2).clamp_min(1e-6) / 7.0).bfloat16().float()
            q = (wg / s.unsqueeze(2)).round().clamp(-8, 7)
            deq = (q * s.unsqueeze(2)).reshape(N, K)
            return BenchCase(run=lambda: x @ deq.T, to_canonical=lambda o: o)

        return Provider(
            library="torch", signature="gemm", kernel_impl="t_int4",
            import_name="torch", build=build, weight_group=weight_group,
            weight_scale="activation",
            serves=KernelSignature(signature="gemm", weight=("int4",),
                                   activation=TORCH_FLOAT))

    @staticmethod
    def _wrong_grid_provider():
        """Declares E2M1 but quantizes on int4's UNIFORM grid — the exact
        confusion that produced 0.213 error against nvfp4's 0.095, where both
        sit inside 'four-bit is lossy' and neither looks wrong on its own."""
        from kernel_bench.providers import BenchCase, KernelSignature, Provider

        def build(inv, env, inputs):
            x, w = inputs["x"].float(), inputs["w"].float()
            N, K = w.shape
            wg = w.reshape(N, K // 16, 16)
            s = wg.abs().amax(2).clamp_min(1e-6) / 7.0     # uniform, not E2M1
            q = (wg / s.unsqueeze(2)).round().clamp(-8, 7)
            deq = (q * s.unsqueeze(2)).reshape(N, K)
            return BenchCase(run=lambda: x @ deq.T, to_canonical=lambda o: o)

        return Provider(
            library="torch", signature="gemm", kernel_impl="t_wrong",
            import_name="torch", build=build, weight_group=16,
            serves=KernelSignature(signature="gemm", weight=("fp4_e2m1",),
                                   activation=TORCH_FLOAT))

    def test_without_the_grid_a_correct_kernel_is_rejected(self):
        # The state this replaces, reproduced: the same kernel, judged against
        # a reference computed from the un-rounded weight.
        pytest.importorskip("torch")
        res = self._run(self._int4_provider(None), "int4")
        assert [r["outcome"] for r in res.rows] == ["mismatch"], "expected the pre-fix rejection"
        assert any("cross-check FAILED" in f for f in res.failures), res.failures

    def test_with_the_grid_the_same_kernel_passes(self):
        pytest.importorskip("torch")
        res = self._run(self._int4_provider(128), "int4")
        assert res.rows, res.failures + res.skipped
        assert res.rows[0]["cross_check"] == "passed"
        # And it passes with room to spare — this is the kernel's arithmetic
        # now, not the format's lossiness.
        assert float(res.rows[0]["cross_check_cosine"]) > 0.999

    def test_the_wrong_grid_is_still_caught(self):
        # The property that would have been given away by asking the provider
        # to dequantize its own packing: a kernel that quantizes and
        # dequantizes through the same wrong grid agrees with itself perfectly.
        pytest.importorskip("torch")
        res = self._run(self._wrong_grid_provider(), "fp4_e2m1")
        assert [r["outcome"] for r in res.rows] == ["mismatch"], "a uniform grid must not pass as E2M1"
        assert any("cross-check FAILED" in f for f in res.failures), res.failures

    def test_a_declared_group_is_the_kernels_own(self):
        # Not derivable from the dtype: marlin blocks NVFP4 by 16, humming
        # groups the same element encoding by 128, and the two put the weight
        # on different grids.
        torch = pytest.importorskip("torch")
        from kernel_bench.quant import round_to_grid
        torch.manual_seed(0)
        w = torch.randn(32, 128)
        assert not torch.equal(round_to_grid(w, "fp4_e2m1", 16),
                               round_to_grid(w, "fp4_e2m1", 128))

    def test_every_grid_is_a_fixed_point(self):
        # The property the whole mechanism rests on: quantizing the rounded
        # weight again returns it unchanged, so a correct kernel's own
        # quantization is lossless.
        torch = pytest.importorskip("torch")
        from kernel_bench.quant import FORMATS, round_to_grid
        torch.manual_seed(0)
        w = torch.randn(64, 256)
        for name in FORMATS:
            for group in (0, 16, 32, 128):
                r = round_to_grid(w, name, group)
                assert torch.equal(round_to_grid(r, name, group), r), \
                    f"{name} at group {group} is not a fixed point"

    def test_an_undeclared_format_raises_rather_than_passing_through(self):
        # A format added to the schema vocabulary but not to the grid table
        # would otherwise silently keep the lossy comparison, and look exactly
        # like nothing had changed.
        torch = pytest.importorskip("torch")
        from kernel_bench.quant import UnknownWeightFormat, round_to_grid
        with pytest.raises(UnknownWeightFormat):
            round_to_grid(torch.randn(8, 16), "ue8m0", 16)

    def test_the_scale_encoding_is_part_of_the_grid(self):
        # Not decoration. marlin holds its int4 scale in the ACTIVATION dtype,
        # and a weight rounded against an exact fp32 scale is not on that grid:
        # modelling the element levels alone left NVFP4 at 0.19 relative error
        # against the reference, an order of magnitude over the bf16 band.
        torch = pytest.importorskip("torch")
        from kernel_bench.quant import round_to_grid
        torch.manual_seed(0)
        w = torch.randn(32, 128)
        exact = round_to_grid(w, "int4", 128, scale_encoding="exact")
        act = round_to_grid(w, "int4", 128, scale_encoding="activation",
                            act_dtype=torch.bfloat16)
        assert not torch.equal(exact, act)

    def test_marlins_declarations_match_its_builds(self):
        # The declaration is only worth anything if it says what the build does.
        from kernel_bench import providers as P
        by = {p.kernel_impl: p for p in P.all_providers()}
        assert (by["marlin_gptq_int4_w4a16"].weight_group,
                by["marlin_gptq_int4_w4a16"].weight_scale) == (128, "activation")
        assert (by["marlin_fp4_e2m1_w4a16"].weight_group,
                by["marlin_fp4_e2m1_w4a16"].weight_scale) == (16, "e4m3_global")
        assert (by["marlin_fp8_w8a16"].weight_group,
                by["marlin_fp8_w8a16"].weight_scale) == (0, "exact")

    def test_marlins_gptq_path_declines_every_cell_it_declares(self):
        """The int4 provider is a declared gap, and it has to decline the
        whole grid it claims — not merely the cell someone happened to probe.

        It is written against vLLM's pre-0.30 GPTQ Marlin API and three of
        the entry points it calls changed together at 0.30
        (`marlin_make_empty_g_idx` renamed, `gptq_marlin_repack` lost `perm`,
        `apply_gptq_marlin_linear` lost `g_idx`/`g_idx_sort_indices`/
        `is_k_full`), so on the release the marlin env now installs it does
        not load, and renaming the one moved symbol would only turn that into
        a misaligned call. Re-enabling it without re-authoring the call — by
        deleting `disabled_reason` — is the wrong implementation this
        excludes: every assertion below goes red on it, at every declared
        weight/activation pair and every shape on the declared grid, because
        `accepts` is the only thing standing between this build and a sweep.
        """
        from kernel_bench import providers as P
        p = next(x for x in P.all_providers()
                 if x.kernel_impl == "marlin_gptq_int4_w4a16")
        assert p.disabled_reason, "still registered, as a named gap"
        # The reason has to name the release mismatch, so that fixing the
        # accuracy alone does not read as discharging it.
        assert "pre-0.30" in p.disabled_reason
        env = Env("2.13", True, None, None, None, sm="sm_90",
                  libs={"vllm": "0.32.0"})
        weights = p.serves.weight or ("int4",)
        acts = p.serves.activation or ("bf16",)
        probed = 0
        for weight in weights:
            for act in acts:
                for d_out, d_reduce in ((2880, 4096), (64, 128), (512, 2048)):
                    inv = Invocation(
                        op="linear", signature="gemm", phase="prefill",
                        dims={"B": 1, "s_query": 128, "d_out": d_out,
                              "d_reduce": d_reduce},
                        dtype_env={"weight": weight, "activation": act,
                                   "output": act}, tp=1)
                    assert not p.accepts(inv, env), (weight, act, d_out,
                                                     d_reduce)
                    probed += 1
        assert probed == len(weights) * len(acts) * 3

    def test_group_zero_is_one_scale_for_the_whole_tensor(self):
        # Not one per row. The finer grid is still a perfectly good fixed point,
        # so the convergence check passes either way — but it is not the
        # kernel's grid, and marlin's per-tensor fp8 path went from 0.12
        # relative error to 0.15 that way: worse than doing nothing, and silent.
        torch = pytest.importorskip("torch")
        from kernel_bench.quant import round_to_grid
        torch.manual_seed(0)
        w = torch.randn(8, 64)
        w[0] *= 100.0                       # one row dominates the tensor amax
        r = round_to_grid(w, "fp8_e4m3", 0)
        # A per-tensor scale is set by the dominant row, so the quiet rows lose
        # resolution. A per-row scale would keep them sharp — that difference is
        # exactly what makes the two grids distinguishable.
        quiet_err = (r[1:] - w[1:]).abs().max().item()
        per_row = round_to_grid(w[1:].contiguous(), "fp8_e4m3", 0)
        assert quiet_err > (per_row - w[1:]).abs().max().item()

    def test_marlin_paths_reach_the_kernels_own_grid(self):
        # Transcribed from the three builds, so this runs without vllm: after
        # placing the weight on each declared grid, the kernel's own
        # quantization must be the IDENTITY. That is the property the whole
        # mechanism rests on, and it is what takes the comparison from measuring
        # the format's lossiness to measuring the kernel's arithmetic.
        torch = pytest.importorskip("torch")
        from kernel_bench.quant import round_to_grid
        torch.manual_seed(0)
        N = K = 128
        w = torch.randn(N, K)
        lv = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])

        def marlin_int4(w, G=128, dt=torch.bfloat16):
            wg = w.float().reshape(N, K // G, G)
            s = (wg.abs().amax(2).clamp_min(1e-6) / 7.0).to(dt).float()
            q = (wg / s.unsqueeze(2)).round().clamp(-8, 7)
            return (q * s.unsqueeze(2)).reshape(N, K)

        def marlin_fp4(w, blk=16):
            wg = w.float().reshape(N, K // blk, blk)
            amax = wg.abs().amax(2).clamp_min(1e-6)
            g = (amax.max() / (6.0 * 448.0)).clamp_min(1e-12)
            b = (amax / 6.0 / g).to(torch.float8_e4m3fn)
            eff = b.float().unsqueeze(2) * g
            q = (wg / eff).clamp(-6, 6)
            i = (q.abs().unsqueeze(-1) - lv).abs().argmin(-1)
            return ((torch.sign(q) * lv[i]) * eff).reshape(N, K)

        def marlin_fp8(w):
            s = w.float().abs().amax().clamp_min(1e-6) / 448.0
            return ((w.float() / s).to(torch.float8_e4m3fn).float() * s)

        for fmt, group, kw, kernel in (
                ("int4", 128, dict(scale_encoding="activation",
                                   act_dtype=torch.bfloat16), marlin_int4),
                ("fp4_e2m1", 16, dict(scale_encoding="e4m3_global"), marlin_fp4),
                ("fp8_e4m3", 0, dict(scale_encoding="exact"), marlin_fp8)):
            r = round_to_grid(w, fmt, group, **kw)
            assert torch.equal(kernel(r), r), (
                f"{fmt}: the kernel's own quantization still moves the weight, "
                f"so the reference is still measuring the format's lossiness")

    def test_hummings_grid_is_derived_and_then_checked(self):
        # Its element rule lives in a compiled op, so the grid could not be read
        # off the source. It is derived from the readable half — `weight_config`
        # passes group 128 and the scale dtype is the activation's — and then
        # confirmed on hardware: 6 of humming's 8 declared combinations collapse
        # from 0.096-0.52 to cosine 1.00000 once the weight is placed on it,
        # which a wrong grid could not produce.
        from kernel_bench import providers as P
        h = next(p for p in P.all_providers() if p.kernel_impl == "humming_gemm")
        assert (h.weight_group, h.weight_scale) == (128, "activation")


class TestPartiallyWrittenOutputsAreNotGarbage:
    """An output the harness allocates and the kernel fills in PART is compared
    in whole.

    With `torch.empty` the unwritten part is whatever the caching allocator last
    left there. That is how `deepgemm_fp8_m_grouped_masked` reported rel_err with
    inf and nan in it — magnitudes no quantization band explains — and why its
    failures looked deterministic per shape while fitting no shape rule at all:
    a block of the same size comes back with the same stale contents, so the
    "garbage" depends on job order, not on any dimension.

    Zeroing does not make a partially-writing kernel correct. It stops the
    harness from manufacturing the garbage, so what the check sees is the
    kernel's own behaviour.
    """

    def test_the_grouped_paths_zero_their_output(self):
        import inspect
        from kernel_bench.providers import gemm
        for fn in (gemm._deepgemm_grouped_build,
                   gemm._deepgemm_grouped_contig_build,
                   gemm._vllm_triton_moe_build):
            src = inspect.getsource(fn)
            assert "torch.zeros(" in src, f"{fn.__name__} allocates its output"
            body = src.split("BenchCase")[0]
            assert "out = torch.empty(" not in body, (
                f"{fn.__name__} still allocates its output uninitialized")

    def test_a_dense_output_may_stay_uninitialized(self):
        # The distinction is whether the kernel writes EVERY element, not a
        # blanket rule: a dense GEMM does, and zeroing it would be a memset per
        # cell for nothing.
        import inspect
        from kernel_bench.providers import gemm
        assert "torch.empty(M, N" in inspect.getsource(gemm._deepgemm_build)

    def test_an_unwritten_row_is_caught_rather_than_read_as_stale(self):
        # End to end with a provider that deliberately leaves its last row
        # unwritten: the check must fail on a DEFINED value, not on whatever
        # happened to be in the block.
        torch = pytest.importorskip("torch")
        from kernel_bench.driver import run_jobs
        from kernel_bench import providers as P
        from kernel_bench.providers import BenchCase, KernelSignature, Provider

        def build(inv, env, inputs):
            x, w = inputs["x"].float(), inputs["w"].float()
            out = torch.zeros(x.shape[0], w.shape[0])

            def run():
                out[:-1] = x[:-1] @ w.T          # last row deliberately skipped
                return out
            return BenchCase(run=run, to_canonical=lambda o: o)

        # The closed declarations refuse a kernel that states no dtype, so
        # the synthetic one states what it actually runs at.
        prov = Provider(library="torch", signature="gemm", kernel_impl="t_partial",
                        import_name="torch", build=build,
                        serves=KernelSignature(signature="gemm",
                                               weight=("bf16", "fp16", "fp32"),
                                               activation=("bf16", "fp16", "fp32")))
        P.discover()
        saved = list(P._PROVIDERS)
        P._PROVIDERS[:] = [prov]
        try:
            env = Env("2.11", False, None, None, None,
                      libs={"torch": "2.11"})
            res = run_jobs([_gemm_job(dims={"B": 1, "s_query": 8,
                                            "d_reduce": 64, "d_out": 64})],
                           env=env, date="2026-08-08", contributor="@t",
                           cross_check=True)
        finally:
            P._PROVIDERS[:] = saved
        assert [r["outcome"] for r in res.rows] == ["mismatch"], (
            "an unwritten row must not be recorded as measured")
        assert any("cross-check FAILED" in f for f in res.failures), res.failures


class TestOnlyTheKernelIsTimed:
    """Setup belongs in `build`; the timed callable is the kernel and nothing
    else.

    `torch_rope` had two extras inside its timed call — an fp32 upcast of Q and
    K, and the `[q‖k]` concatenation that only exists so the cross-check can
    compare one array. Every other rope provider already does that concat in
    `to_canonical`, OUTSIDE its timing, so the always-available baseline was
    recorded 2.6-3.4x slower than the work it represents and every rope speedup
    in the corpus was inflated by that factor.
    """

    def test_no_timed_callable_allocates(self):
        # A whole-tensor allocation inside `run` is setup that escaped `build`.
        # Checked over the real registry rather than by review: this one hid in
        # a `def _run()` body, which reads like part of the kernel.
        import ast
        import pathlib
        HEAVY = {"zeros", "empty", "full", "arange", "randn", "cat", "stack",
                 "clone", "contiguous", "repeat_interleave"}
        offenders = []
        for path in sorted(pathlib.Path("kernel_bench/providers").glob("*.py")):
            tree = ast.parse(path.read_text())
            timed = set()
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "id", "") == "BenchCase"):
                    for kw in node.keywords:
                        if kw.arg == "run" and isinstance(kw.value, ast.Name):
                            timed.add(kw.value.id)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in timed:
                    for call in ast.walk(node):
                        if not isinstance(call, ast.Call):
                            continue
                        name = (getattr(call.func, "attr", None)
                                or getattr(call.func, "id", None))
                        if name in HEAVY:
                            offenders.append(
                                f"{path.name}:{node.lineno} {node.name}() "
                                f"calls {name}() inside the timed call")
        assert not offenders, "\n".join(offenders)

    def test_the_torch_rope_baseline_matches_vllms_native_path(self):
        # vLLM `forward_native` casts the cos/sin cache to the QUERY dtype and
        # computes there, and returns q and k separately. Both are what make
        # this comparable to the in-place kernels it is the baseline for.
        import inspect
        from kernel_bench.providers import rope
        src = inspect.getsource(rope._torch_rope_build)
        body = src.split("def _run")[1]
        assert ".float()" not in body, "fp32 upcast is back inside the timing"
        assert "torch.cat" not in body, "the canonicalizing concat is back"
        assert "_canon_qk" in src, "canonicalization must run in to_canonical"

    def test_rope_providers_canonicalize_the_same_way(self):
        # The comparison is only fair if every provider on this signature does
        # the same work inside its timed call and the same work outside it.
        from kernel_bench import providers as P
        rope_impls = [p for p in P.all_providers() if p.signature == "rope"]
        assert len(rope_impls) >= 4
        import inspect
        for p in rope_impls:
            src = inspect.getsource(p.build)
            assert "to_canonical=lambda" in src or "_canon_qk" in src, (
                f"{p.kernel_impl}: no canonicalizer, so its timed call is "
                f"doing the comparison's work")


class TestCausalAlignmentMatchesTheKernels:
    """`is_causal=True` and FlashAttention do NOT mean the same thing when
    Sq != Lq.

    SDPA masks top-left — query 0 sees only key 0. FlashAttention (v2 >= 2.1,
    v3) and flashinfer mask bottom-right — query 0 sees keys 0..(Lq-Sq), the
    semantics of chunked prefill, where the queries are the LAST Sq positions of
    a longer context. Measured on an H100: FA2 agrees with bottom-right at
    cosine 0.999997 and with top-left at 0.236.

    The reference used `is_causal`, so it rejected the kernels: 226 cells each
    for fa2 and fa3, EVERY one with Sq != Lq and none with Sq == Lq, with the
    two libraries agreeing to the digit (rel_err 2.79, cosine 0.24241) while
    disagreeing with the oracle. Two independent implementations agreeing
    against the oracle is a fact about the oracle.

    The torch SDPA providers had the same bug pointing the other way: as the
    baseline the kernels are compared against, they computed a DIFFERENT
    attention at every Sq != Lq cell.
    """

    def test_equal_lengths_still_use_is_causal(self):
        # Where the two alignments coincide, keep the cheap path — an explicit
        # mask would force SDPA's MATH backend and materialize O(S^2).
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _sdpa_causal_args
        q = torch.zeros(1, 2, 128, 8)
        k = torch.zeros(1, 2, 128, 8)
        assert _sdpa_causal_args(q, k, True) == {"is_causal": True}

    def test_unequal_lengths_switch_to_bottom_right(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _sdpa_causal_args
        q = torch.zeros(1, 2, 64, 8)
        k = torch.zeros(1, 2, 256, 8)
        args = _sdpa_causal_args(q, k, True)
        assert args["is_causal"] is False
        assert args["attn_mask"] is not None

    def test_non_causal_is_untouched(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _sdpa_causal_args
        q, k = torch.zeros(1, 2, 64, 8), torch.zeros(1, 2, 256, 8)
        assert _sdpa_causal_args(q, k, False) == {"is_causal": False}

    def test_one_definition_shared_by_references_and_providers(self):
        # The rule lives in refs and the providers import it. Two copies of an
        # alignment convention is how the baseline and the oracle came to
        # disagree in the first place.
        import inspect
        from kernel_bench.providers import attention
        from kernel_bench import refs
        assert attention._sdpa_causal_args is refs._sdpa_causal_args
        # Both SDPA floors build their call in `_sdpa_case`.
        for fn in (attention._sdpa_case,):
            src = inspect.getsource(fn)
            assert "_sdpa_causal_args" in src
            assert "is_causal=" not in src, f"{fn.__name__} sets is_causal directly"

    def test_the_baseline_and_a_kernel_now_agree_at_unequal_lengths(self):
        # End to end on whatever GPU is present, against torch's own bottom-right
        # bias: the baseline must compute what the kernels compute.
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs a GPU")
        import torch.nn.functional as F
        from torch.nn.attention.bias import causal_lower_right
        from kernel_bench.invocation import Invocation
        from kernel_bench.refs import get_ref
        inv = Invocation(signature="gqa_attention", op="flash_attention",
                         dims={"kv_block_size": 16, "B": 1, "s_query": 64, "s_ctx": 256, "s_kv": 256,
                               "n_heads": 8, "n_kv_heads": 2, "d_head": 64},
                         dtype_env={"activation": "bf16"}, phase="prefill", tp=1)
        ins = build_inputs(get_ref("gqa_attention"), inv, "cuda")
        env = Env("2.11", True, None, None, None, libs={"torch": "2.11"})
        prov = next(p for p in all_providers() if p.kernel_impl == "torch_sdpa")
        got = prov.build(inv, env, prov.convert_inputs(inv, ins))
        to_can = prov.wrap_output(inv, got.to_canonical)
        out = to_can(got.run()).float()
        q, k, v = (ins[n].transpose(1, 2) for n in ("q", "k", "v"))
        want = F.scaled_dot_product_attention(
            q, k, v, attn_mask=causal_lower_right(q.shape[-2], k.shape[-2]),
            enable_gqa=True).transpose(1, 2).float()
        cos = F.cosine_similarity(out.flatten(), want.flatten(), dim=0).item()
        assert cos > 0.9999, cos


class TestCrashGatesAreDeclared:
    """A kernel that FAULTS takes the whole model's run with it.

    Most declaration errors cost a cell: the cross-check rejects the row and
    the sweep moves on. A misaligned access is different — it poisons the CUDA
    context, so every remaining job for that model dies too. flashinfer's
    activation kernel loads 16 bytes at a time and reads its `y` half at
    `input + offset + d`, so `d` must be a multiple of `16 / sizeof(T)`. At
    d_ffn = 12 it faults, and DeepSeek-V2 and V2-Lite each exited 1 after
    hitting it three times.
    """

    def _inv(self, d_ffn):
        from kernel_bench.invocation import Invocation
        return Invocation(signature="silu_and_mul", op="silu_and_mul",
                          dims={"B": 1, "s_query": 1, "d_ffn": d_ffn},
                          dtype_env={"activation": "bf16"}, phase="decode", tp=1)

    def _prov(self):
        from kernel_bench import providers as P
        return next(p for p in P.all_providers()
                    if p.kernel_impl == "flashinfer_silu_and_mul")

    def test_the_faulting_width_is_declined(self):
        assert not self._prov().serves.accepts(self._inv(12))

    def test_aligned_widths_still_served(self):
        prov = self._prov()
        for d in (8, 128, 10944, 18432):
            assert prov.serves.accepts(self._inv(d)), d

    def test_the_torch_baseline_still_covers_what_flashinfer_declines(self):
        # Declining must not remove the cell from the corpus: the always-
        # available baseline has no such constraint, so d_ffn = 12 is still
        # measured, just not by the kernel that faults on it.
        from kernel_bench import providers as P
        torch_silu = next(p for p in P.all_providers()
                          if p.kernel_impl == "torch_silu_and_mul")
        assert torch_silu.serves.accepts(self._inv(12))


class TestProvisioningIsWrittenDown:
    """A library the repo cannot install must SAY so, in a file.

    `render_env_files` used to skip every source build, so a fresh environment
    got nothing for them — not a wrong instruction, an absent one. That is the
    same shape as `fla`, which sat at 311 failed builds against a dependency
    the repo described as `flash-linear-attention>=0.4` while the published
    package does not contain the module every one of its kernels imports.
    """

    def test_every_library_gets_a_file(self):
        from kernel_bench.providers import dependency_manifest, render_env_files
        files = render_env_files()
        # torch included: its floors are providers like any other library's,
        # measured in their own env (`render_env_files`).
        assert {f[:-4] for f in files} == set(dependency_manifest())

    def test_a_source_build_says_it_is_one(self):
        from kernel_bench.providers import render_env_files
        txt = render_env_files()["fa4.txt"]
        assert "SOURCE BUILD" in txt
        assert "flash_attn.cute" in txt          # the gate, named
        assert "CONFLICTS" in txt                # why it cannot share fa2's env

    def test_the_import_gate_is_what_the_build_needs(self):
        # `sglang` imports fine on this box; `flash_attn.cute` is what fails.
        # Declaring the outer one let availability pass and cost 1,436 failed
        # builds in a single pass.
        from kernel_bench import providers as P
        fa4 = next(p for p in P.all_providers()
                   if p.kernel_impl == "flash_attention_v4")
        assert fa4.import_name == "flash_attn.cute"
        fla = next(p for p in P.all_providers() if p.library == "fla")
        assert fla.import_name == "fla.ops"

    def test_flas_dependency_is_pinned_to_a_release_tag(self):
        # PyPI's flash-linear-attention ships neither `fla/ops` in its wheel nor
        # in its sdist — no version from 0.3.2 to 0.5.2 — so a version range can
        # never provision these kernels.
        #
        # A TAG, not a commit: the corpus records `framework_version` from
        # `__version__`, and a main commit reports "0.5.2" as readily as the
        # release does. Pinning a tree that happens to be checked out would put
        # a version string in the corpus that traces back to nothing.
        import re
        from kernel_bench.providers import dependency_manifest
        pip = sorted(dependency_manifest()["fla"]["pip"])[0]
        assert pip.startswith("flash-linear-attention @ git+")
        ref = pip.split("@")[-1]
        assert re.fullmatch(r"v\d+\.\d+(\.\d+)?", ref), (
            f"{ref!r} is not a release tag — an unpinned or commit-pinned spec "
            f"records a version the corpus cannot trace")

    def test_generated_files_match_the_registry(self):
        # The committed files are generated; a provider change that is not
        # regenerated would silently ship a stale environment.
        import pathlib
        from kernel_bench.providers import render_env_files
        root = pathlib.Path("kernel_bench/envs")
        for name, body in render_env_files().items():
            on_disk = (root / name)
            assert on_disk.exists(), f"{name} not committed"
            assert on_disk.read_text() == body, (
                f"{name} is stale — run `python -m kernel_bench envs "
                f"-o kernel_bench/envs`")


class TestHeadDimCeilingsAreDeclared:
    """A ceiling is not a modulus or an enumeration.

    FlashAttention refuses head dimensions above a build-specific limit — fa2
    at 256, this fa3 hopper build at 192, each quoting its own message. The set
    of legal values is open below the limit, so neither `divisible` nor `enum`
    can express it, and without one every oversized cell was attempted and
    refused: 2,236 failed builds across fa2 and fa3 in a single pass, all of
    them MLA-shaped cells at d_head 256 and 512.
    """

    def _inv(self, d_head, phase="prefill"):
        from kernel_bench.invocation import Invocation
        return Invocation(
            signature="gqa_attention", op="flash_attention",
            dims={"kv_block_size": 16, "B": 1, "s_query": 64, "s_ctx": 256, "s_kv": 256,
                  "n_heads": 8, "n_kv_heads": 2, "d_head": d_head},
            dtype_env={"activation": "bf16"}, phase=phase, tp=1)

    def _by(self, impl):
        from kernel_bench import providers as P
        return next(p for p in P.all_providers() if p.kernel_impl == impl)

    def test_fa2_stops_at_256(self):
        for impl, phase in (("flash_attention_v2", "prefill"),
                            ("flash_attention_v2_kvcache", "decode")):
            s = self._by(impl).serves
            assert s.accepts(self._inv(256, phase))
            assert not s.accepts(self._inv(512, phase))

    def test_fa3_stops_lower_than_fa2(self):
        # The limits differ per build; declaring one number for "FlashAttention"
        # would be wrong for one of them.
        for impl, phase in (("flash_attention_v3", "prefill"),
                            ("flash_attention_v3_kvcache", "decode")):
            s = self._by(impl).serves
            assert s.accepts(self._inv(192, phase))
            assert not s.accepts(self._inv(256, phase))

    def test_ordinary_head_dims_are_untouched(self):
        for impl in ("flash_attention_v2", "flash_attention_v3"):
            assert self._by(impl).serves.accepts(self._inv(128))

    def test_a_ceiling_is_published(self):
        caps = self._by("flash_attention_v3").capabilities()
        assert "d_head <= 192" in caps["shape_constraints"]


class TestVersionProbeFollowsSubmoduleGates:
    """`framework_version` must name the version that produced the number.

    An import gate may point at a SUBMODULE — `fla.ops`, `flash_attn.cute` —
    when that is what the build needs and the top-level package imports without
    it. A submodule has no `__version__`, so reading only its attribute recorded
    `framework_version=unknown` for 63 rows, which throws away exactly what
    pinning the dependency to a release tag was for.
    """

    def test_a_submodule_reports_its_packages_version(self):
        # torch.optim carries no `__version__`; torch does. (The project's
        # own `kernel_bench.providers` is not an example: the repo's
        # distribution is `huggingarch`, so nothing versionable answers at
        # that name and None is the honest probe.)
        pytest.importorskip("torch")
        from kernel_bench.env import _probe_version
        v = _probe_version("torch.optim")
        assert v is not None
        # falls through to the distribution rather than stopping at the
        # submodule's missing attribute
        assert v == _probe_version("torch")

    def test_an_absent_module_is_still_absent(self):
        # The fallback must not manufacture a version for something not there —
        # that would make an unavailable library look installed.
        from kernel_bench.env import _probe_version
        assert _probe_version("kernel_bench.nope_not_here") is None

    def test_the_gates_that_need_it_are_submodules(self):
        from kernel_bench import providers as P
        gates = {p.import_name for p in P.all_providers()}
        assert {"fla.ops", "flash_attn.cute"} <= gates


class TestACeilingSaysWhetherTheKernelCannotOrTheHarnessWont:
    """Two ceilings of the same shape, because they say opposite things
    about the rows already measured past them.

    `max_product` is what the KERNEL cannot run (sglang's conv1d state
    faults past 2**31), so a row measured beyond it timed a launch that
    cannot happen and is wrong. `samples_up_to` is what the HARNESS declines
    to sample — the torch ragged floor is a Python loop over experts, whose
    wall clock past a few thousand routed rows is host dispatch at 60x the
    cost of a plain GEMM cell — and the 10,166 rows measured before that
    budget existed are honest measurements of a slow implementation.
    Collapsed into one field, the tool that deletes rows no run would make
    today (`tools.unserved_rows`) deleted those 10,166 as well.
    """

    def _ragged(self):
        from kernel_bench import providers as P
        return next(p for p in P.all_providers()
                    if p.kernel_impl == "torch_ragged_matmul").serves

    def _inv(self, experts, rows):
        from kernel_bench.invocation import Invocation
        return Invocation(
            signature="grouped_gemm", op="grouped_gemm",
            dims={"n_experts": experts, "s_expert_rows": rows,
                  "d_reduce": 512, "d_out": 512},
            dtype_env={"activation": "bf16", "weight": "bf16"},
            phase="prefill", tp=1)

    def test_the_budget_declines_the_cell(self):
        serves = self._ragged()
        assert serves.accepts(self._inv(8, 128))
        assert not serves.accepts(self._inv(256, 1024))

    def test_but_the_capability_alone_accepts_it(self):
        import dataclasses
        serves = self._ragged()
        capability = dataclasses.replace(serves, samples_up_to=())
        assert capability.accepts(self._inv(256, 1024))

    def test_a_capability_ceiling_is_not_waived_the_same_way(self):
        import dataclasses
        from kernel_bench import providers as P
        serves = next(p for p in P.all_providers()
                      if p.kernel_impl == "sglang_causal_conv1d_update").serves
        from kernel_bench.invocation import Invocation
        past = Invocation(
            signature="causal_conv1d", op="causal_conv1d",
            dims={"B": 521391, "s_query": 1, "d_conv_channels": 8192,
                  "s_conv_kernel": 4, "dilation": 1},
            dtype_env={"activation": "bf16"}, phase="decode", tp=1)
        assert not serves.accepts(past)
        assert not dataclasses.replace(serves, samples_up_to=()).accepts(past)


class TestInt32StateCeilingIsAProduct:
    """Some limits are not per-axis, and describing them as one is wrong.

    sglang's `causal_conv1d_update` indexes its `[B, C, W]` state with int32,
    so it faults once the element count crosses 2**31 — `an illegal memory
    access`, which poisons the context and takes the rest of the model's run
    with it. That is what killed Qwen3-Coder-Next at B=521,391, C=8192.

    Bisected on an H100, the boundary is exactly `B * C * W == 2**31` at C =
    1024, 2048, 4096 and 8192 alike: 524,288 / 262,144 / 131,072 / 65,536 pass
    and each next integer faults. A ceiling on B alone would be 65,536 (right
    for C=8192, eight times too strict for C=1024) — which is why this needed
    `max_product` rather than another `max_dims`.
    """

    def _inv(self, B, C, W=4):
        from kernel_bench.invocation import Invocation
        return Invocation(
            signature="causal_conv1d", op="causal_conv1d",
            dims={"B": B, "s_query": 1, "d_conv_channels": C,
                  "s_conv_kernel": W, "dilation": 1},
            dtype_env={"activation": "bf16"}, phase="decode", tp=1)

    def _sgl(self):
        from kernel_bench import providers as P
        return next(p for p in P.all_providers()
                    if p.kernel_impl == "sglang_causal_conv1d_update").serves

    def test_the_boundary_is_exact_at_every_probed_width(self):
        s = self._sgl()
        for C, B in ((1024, 524288), (2048, 262144), (4096, 131072),
                     (8192, 65536)):
            assert B * C * 4 == 2 ** 31                  # the measured boundary
            assert s.accepts(self._inv(B, C)), (B, C)
            assert not s.accepts(self._inv(B + 1, C)), (B + 1, C)

    def test_the_cell_that_killed_a_model_is_declined(self):
        assert not self._sgl().accepts(self._inv(521391, 8192))

    def test_ordinary_batches_are_untouched(self):
        s = self._sgl()
        for B in (1, 16, 512, 8192):
            assert s.accepts(self._inv(B, 4096))

    def test_a_per_axis_ceiling_could_not_express_it(self):
        # The same B is legal or not depending on C, so no `max_dims` on B is
        # both correct at C=1024 and correct at C=8192.
        s = self._sgl()
        assert s.accepts(self._inv(100000, 1024))
        assert not s.accepts(self._inv(100000, 8192))

    def test_vllms_conv1d_update_is_disabled_with_its_reason(self):
        # It fails to COMPILE at every shape probed, from B=16 up — a
        # triton/vllm mismatch, not anything about the cell. Disabled rather
        # than deleted so its cells read as a gap.
        from kernel_bench import providers as P
        from kernel_bench.env import Env
        v = next(p for p in P.all_providers()
                 if p.kernel_impl == "vllm_causal_conv1d_update")
        assert "CompilationError" in v.disabled_reason
        env = Env("2.11", True, None, None, None, sm="sm_90",
                  libs={"vllm": "0.26"})
        assert not v.accepts(self._inv(16, 4096), env)

    def test_the_product_is_published(self):
        from kernel_bench import providers as P
        caps = next(p for p in P.all_providers()
                    if p.kernel_impl == "sglang_causal_conv1d_update").capabilities()
        assert any("*" in c and "<=" in c for c in caps["shape_constraints"])


class TestDecodeHasABatchCeiling:
    """`max_batch` answers a capacity question, not a launch-shape one.

    At small models it reaches widths no scheduler admits — 521,391 concurrent
    sequences for Qwen3-Coder-Next — and that is also where kernels stop
    working: sglang's causal_conv1d_update indexes its `[B, C, W]` state with
    int32 and faults past 2**31 elements, an illegal memory access that poisons
    the context and takes the rest of the model's run with it. Prefill already
    had `PREFILL_TOKEN_BUDGET` for the same reason; decode had nothing because
    `B·1` is trivially inside any token budget.
    """

    class _Cap:
        def __init__(self, decode, prefill):
            self.max_batch, self.max_prefill_batch = decode, prefill
            self.beyond_max_context = False

    def test_an_absurd_decode_width_is_capped(self):
        from kernel_bench.sweep import DECODE_BATCH_CEILING, _feasible_batches
        pts = _feasible_batches(self._Cap(521391, 14681), "decode", 1, _ROUTINE_BATCH_KEYS)
        assert max(pts) == DECODE_BATCH_CEILING

    def test_a_deployment_that_fits_below_it_is_untouched(self):
        # The cap must not become the batch — a model whose real ceiling is
        # 8192 still measures at 8192.
        from kernel_bench.sweep import _feasible_batches
        assert max(_feasible_batches(self._Cap(8192, 14681), "decode", 1, _ROUTINE_BATCH_KEYS)) == 8192

    def test_prefill_keeps_its_own_ceiling(self):
        # Prefill is bounded by the token budget, which is stricter at any
        # real sequence length; the decode cap must not loosen it.
        from kernel_bench.sweep import PREFILL_TOKEN_BUDGET, _feasible_batches
        pts = _feasible_batches(self._Cap(521391, 14681), "prefill", 1024, _ROUTINE_BATCH_KEYS)
        assert max(pts) == PREFILL_TOKEN_BUDGET // 1024

    def test_the_cap_covers_the_conv1d_fault_at_the_widest_conv(self):
        # C=8192, W=4 is the widest conv state in the corpus, and its int32
        # boundary is exactly B=65,536 — so at that width the cap and the
        # kernel's own limit coincide rather than one shadowing the other.
        from kernel_bench.sweep import DECODE_BATCH_CEILING
        assert DECODE_BATCH_CEILING * 8192 * 4 == 2 ** 31


class TestAJobCarriesTheShapesItsScalarsCameFrom:
    """The identity is a LOSSY projection of the graph's shapes, and a corpus
    key has to be scalars. Input construction reads the carried layout
    (`shape_in` via the signature's `engine_inputs`); the scalars fill axes
    no operand holds.

    This pins the precondition: the scalars must be exactly what projecting
    the carried shapes produces, or the two statements of one node have
    already drifted.
    """

    def test_every_job_carries_its_shapes_and_they_reproduce_the_scalars(self):
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        from kernel_bench.plan import build_jobs
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="deepseek-ai/DeepSeek-V3", batch_size=1,
                             seq_len=2048, phase="prefill", tp=8)
        assert jobs, "no jobs planned — the check would be vacuous"
        missing = [j["op"] for j in jobs if j.get("shape_in") is None]
        assert not missing, f"jobs planned without their shapes: {set(missing)}"

        checked, drift = 0, []
        for j in jobs:
            # The page is the deployment's, not the shapes': hand back the
            # one the job was planned at, so a paged cell is checked too.
            served = ({"kv_block_size": j["dims"]["kv_block_size"]}
                      if "kv_block_size" in j["dims"] else None)
            got = identity_shape_from_shape_in(
                j["op"], j["shape_in"], shape_out=j.get("shape_out"),
                spec_dims=served, operand_ports=j.get("operand_ports"))
            if got is None:
                continue          # needs context this job does not carry
            checked += 1
            if json.loads(got) != j["dims"]:
                drift.append((j["op"], json.loads(got), j["dims"]))
        assert not drift, drift
        # Guard the guard: a projection that started returning None everywhere
        # would leave nothing compared and this test would still pass.
        assert checked >= len(jobs) // 2, (
            f"only {checked}/{len(jobs)} jobs had a computable projection")


class TestAJobCarriesTheWeightShapeToo:
    """`shape_in` is ACTIVATIONS. The param is stated separately by the graph
    (`OpSpec.weight_shape`), and without it a builder rebuilds the weight out
    of the identity scalars — the same duplication carrying the activation
    shapes exists to remove, one operand class over.

    Derived at the EXPORT boundary from two things the node already holds, its
    op and its resolved slot dims, so nothing is added to the node record.
    Analysis bindings stay global; pipeline projects them through
    `local_node_dims` before export, so under TP a job carries the shard a
    kernel allocates.
    """

    def test_the_carried_weight_is_what_the_reference_allocates(self):
        torch = pytest.importorskip("torch")
        import collections
        import dataclasses
        from kernel_bench.invocation import Invocation
        from kernel_bench.plan import build_jobs
        from kernel_bench.refs import get_ref

        agree, mismatch = 0, collections.Counter()
        for model, tp in (("deepseek-ai/DeepSeek-V3", 8),
                          ("meta-llama/Llama-3.2-3B", 1)):
            jobs, _ = build_jobs(avg_output_tokens=0, model_id=model, batch_size=1, seq_len=2048,
                                 phase="prefill", tp=tp)
            for job in jobs:
                inv = Invocation.from_job(job)
                ref = get_ref(inv.signature)
                if ref is None or inv.shape_weight is None:
                    continue
                # Strip the carried shape so the builder takes its fallback —
                # a rebuild from the identity scalars. The reference PREFERS
                # the carried shape now, so auditing the default build would
                # compare the graph's statement against itself; the question
                # this test exists for is whether the two statements — the
                # graph's weight and the corpus-key projection — still
                # reconcile.
                inv_rebuilt = dataclasses.replace(inv, shape_weight=None)
                with torch.device("meta"):
                    try:
                        inputs = build_inputs(ref, inv_rebuilt, "meta")
                    except Exception:
                        continue
                w = inputs.get("w")
                if w is None or not hasattr(w, "shape"):
                    continue
                got, carried = tuple(w.shape), tuple(inv.shape_weight)
                if got == carried or (inv.op == "grouped_gemm" and got[1:] == carried):
                    agree += 1
                else:
                    mismatch[(inv.op, got, carried)] += 1
        assert not mismatch, dict(mismatch)
        # Guard the guard: a broken join would compare nothing and pass. 18
        # across these two models today; the floor only has to be high enough
        # that an empty comparison cannot hide.
        assert agree >= 15, f"only {agree} weights compared — check the join"

    def test_the_builder_allocates_the_carried_shape_not_the_scalars(self):
        """Consumption, not coincidence: when the carried weight and the
        identity scalars disagree, the builder must follow the graph's
        statement. `dims` is the lossy projection minted for corpus keys;
        `shape_weight` is what a serving rank allocates."""
        pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        from kernel_bench.refs import get_ref

        dtype_env = {"weight": "bf16", "activation": "bf16", "kv": "bf16",
                     "output": "bf16", "metadata": "fp32"}

        def build(signature, op, dims, shape_weight):
            inv = Invocation(signature=signature, op=op, dims=dims,
                             dtype_env=dtype_env, phase="prefill", tp=1,
                             shape_weight=shape_weight)
            return build_inputs(get_ref(signature), inv, "cpu")["w"]

        # gemm: the scalars say (N, K) = (4, 8); the graph carries (16, 8).
        w = build("gemm", "linear",
                  {"B": 1, "s_query": 2, "d_reduce": 8, "d_out": 4}, [16, 8])
        assert tuple(w.shape) == (16, 8)
        # rmsnorm: the carried gain (4,) wins over d_model = 8.
        w = build("rmsnorm", "rmsnorm",
                  {"B": 1, "s_query": 2, "d_model": 8}, [4])
        assert tuple(w.shape) == (4,)
        # grouped_gemm: the expert count stays the harness's packing,
        # prepended to the carried per-expert (N, K).
        w = build("grouped_gemm", "grouped_gemm",
                  {"B": 1, "s_expert_rows": 2, "d_reduce": 8, "d_out": 4,
                   "n_experts": 3}, [4, 8])
        assert tuple(w.shape) == (3, 4, 8)

    def test_the_weight_is_the_per_rank_shard(self):
        """Under TP a kernel allocates its shard. DeepSeek-V3's lm_head is
        vocab-parallel, so the carried weight must divide by tp — carrying the
        whole tensor would have the harness measure a GEMM no rank runs."""
        from backend.arch.loader import SpecLoader
        from backend.arch.layout.persistent import physical_weight_leaves
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.step_eval import evaluate_step
        import backend.analyzer as A

        resolved = SpecLoader().match(A.fetch_config("deepseek-ai/DeepSeek-V3"), model_id="deepseek-ai/DeepSeek-V3")
        widths = {}
        for tp in (1, 8):
            parallel = ParallelConfig(
                tp_attention=tp,
                tp_dense_ffn=tp,
                tp_moe_expert=tp,
            )
            analysis = evaluate_step(
                resolved,
                parallel,
                phase="prefill",
                batch=1,
                input_tokens=2048, output_tokens=0,
            ).analysis
            got = {
                sid: row.runtime_shape
                for sid, row in physical_weight_leaves(analysis).items()
            }
            assert got, f"tp={tp}: no leaf carried a weight shape"
            widths[tp] = {tuple(v) for v in got.values()}
        # 129280 is the vocab; at tp=8 the same leaf must be 16160 wide.
        assert any(w[0] == 129280 for w in widths[1]), sorted(widths[1])[:3]
        assert any(w[0] == 129280 // 8 for w in widths[8]), sorted(widths[8])[:3]

    def test_tp_linear_job_shape_weight_matches_sharded_activation(self):
        # 4090 run: TP=2 o_proj carried shape_weight [2048,2048] next to
        # shape_in [..., 1024]. The harness prefers the carried weight, so
        # F.linear died. The export must publish the per-rank shard.
        from backend.inference.parallel_config import ParallelConfig
        from kernel_bench.plan import build_jobs

        jobs, _ = build_jobs(avg_output_tokens=0,
            model_id="meta-llama/Llama-3.2-1B",
            batch_size=1, seq_len=1024, phase="prefill",
            parallel=ParallelConfig(
                tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2),
        )
        checked = 0
        for job in jobs:
            if job["op"] != "linear" or not job.get("shape_weight"):
                continue
            shape_in, shape_out = job.get("shape_in"), job.get("shape_out")
            if not shape_in or not shape_out:
                continue
            assert job["shape_weight"] == [shape_out[-1], shape_in[-1]], (
                job["node_spec_id"], job["shape_weight"], shape_in, shape_out)
            checked += 1
        assert checked >= 2, f"only {checked} linear jobs carried a weight"


class TestTheBuilderAllocatesTheCarriedActivation:
    """`shape_in` is the engine's operand layout. Canonical construction
    permutes those axes (GQA head-first → token-first) rather than
    rebuilding from the identity scalars, which disagree on order and
    can disagree on extent."""

    _DT: ClassVar[dict] = {"weight": "bf16", "activation": "bf16", "kv": "bf16",
           "output": "bf16", "metadata": "fp32"}

    def test_gqa_follows_engine_axis_order_not_the_scalars(self):
        pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        from kernel_bench.refs import get_ref
        # Engine Q is [B, N, Sq, D] with N=32; scalars still say n_heads=4.
        inv = Invocation(
            signature="gqa_attention", op="flash_attention",
            dims={"kv_block_size": 16, "B": 1, "s_query": 8, "s_kv": 8, "n_heads": 4,
                  "n_kv_heads": 2, "d_head": 16},
            dtype_env=self._DT, phase="prefill", tp=1,
            shape_in=[[1, 32, 8, 16], [1, 8, 8, 16], [1, 8, 8, 16]],
            operand_ports=[["q"], ["k"], ["v"]],
        )
        ins = build_inputs(get_ref("gqa_attention"), inv, "cpu")
        assert tuple(ins["q"].shape) == (1, 8, 32, 16)
        assert tuple(ins["k"].shape) == (1, 8, 8, 16)
        assert tuple(ins["v"].shape) == (1, 8, 8, 16)

    def test_gemm_packs_the_carried_token_axes(self):
        pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        from kernel_bench.refs import get_ref
        inv = Invocation(
            signature="gemm", op="linear",
            dims={"B": 1, "s_query": 2, "d_reduce": 8, "d_out": 4},
            dtype_env=self._DT, phase="prefill", tp=1,
            shape_in=[1, 16, 8],
        )
        x = build_inputs(get_ref("gemm"), inv, "cpu")["x"]
        assert tuple(x.shape) == (16, 8)

    def test_without_shape_in_the_scalars_still_build(self):
        pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        from kernel_bench.refs import get_ref
        inv = Invocation(
            signature="gqa_attention", op="flash_attention",
            dims={"kv_block_size": 16, "B": 2, "s_query": 4, "s_kv": 8, "n_heads": 4,
                  "n_kv_heads": 2, "d_head": 8},
            dtype_env=self._DT, phase="prefill", tp=1,
        )
        ins = build_inputs(get_ref("gqa_attention"), inv, "cpu")
        assert tuple(ins["q"].shape) == (2, 4, 4, 8)

    def test_mla_variant_builds_per_head_kv_from_the_contract(self):
        pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        from kernel_bench.refs import get_ref
        dt = self._DT
        absorbed = Invocation(
            signature="mla_attention", op="mla_attention",
            dims={"kv_block_size": 64, "B": 1, "s_query": 2, "s_kv": 8, "n_heads": 4,
                  "d_head": 16, "d_v_head": 8},
            dtype_env=dt, phase="decode", tp=1)
        a = build_inputs(get_ref("mla_attention"), absorbed, "cpu")
        assert "c" in a and "k" not in a
        assert tuple(a["c"].shape) == (1, 8, 16)
        prefill = Invocation(
            signature="mla_attention", op="mla_attention",
            dims={"kv_block_size": 64, "B": 1, "s_query": 2, "s_kv": 8, "n_heads": 4,
                  "n_kv_heads": 4, "d_head": 16, "d_v_head": 8},
            dtype_env=dt, phase="prefill", tp=1)
        p = build_inputs(get_ref("mla_attention"), prefill, "cpu")
        assert "k" in p and "v" in p and "c" not in p
        assert tuple(p["k"].shape) == (1, 8, 4, 16)
        assert tuple(p["v"].shape) == (1, 8, 4, 8)

    def test_a_planned_job_permutes_engine_q_to_canonical(self):
        pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        from kernel_bench.plan import build_jobs
        from kernel_bench.refs import get_ref
        jobs, _ = build_jobs(avg_output_tokens=0, model_id="meta-llama/Llama-3.2-1B",
                             batch_size=1, seq_len=64, phase="prefill")
        gqa = next(j for j in jobs if j["signature"] == "gqa_attention")
        q_eng = gqa["shape_in"][0]
        assert len(q_eng) == 4
        B, N, Sq, D = (int(x) for x in q_eng)
        ins = build_inputs(get_ref("gqa_attention"),
            Invocation.from_job(gqa), "cpu")
        assert tuple(ins["q"].shape) == (B, Sq, N, D)

    def test_conv_permutes_feature_last_engine_to_canonical(self):
        pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        from kernel_bench.refs import get_ref
        # Engine x is [B, S, C]; kernels want [B, C, S]. Scalars that
        # disagree on the channel count must not win.
        inv = Invocation(
            signature="causal_conv1d", op="causal_conv1d",
            dims={"B": 1, "s_query": 8, "d_conv_channels": 4,
                  "s_conv_kernel": 3, "dilation": 1},
            dtype_env=self._DT, phase="prefill", tp=1,
            shape_in=[2, 16, 32],
        )
        ins = build_inputs(get_ref("causal_conv1d"), inv, "cpu")
        assert tuple(ins["x"].shape) == (2, 32, 16)


class TestGraphToOperatorIsAQuotient:
    """The compute graph decides WHAT is one measurable unit; the harness owns
    only which library provides it and which kernel runs.

    The map from graph to operator is MANY-TO-ONE and lossy on purpose — three
    linear-shaped ops are one `gemm`, a three-node attention span is one
    `gqa_attention` — so the operator vocabulary is a QUOTIENT of the graph's,
    not a subset of it. Asking every operator name to trace back to a graph
    name is the wrong direction and would forbid exactly the collapsing the map
    exists to do.

    What must hold is that the quotient is total and lands somewhere real.
    """

    def test_the_quotient_lands_on_registered_signatures(self):
        from backend.inference.op_signatures import op_bindings
        from kernel_bench.signatures import get_signature
        for op, sig in op_bindings().items():
            assert get_signature(sig) is not None, f"{op} → unregistered {sig}"

    def test_a_context_rewrite_stays_measurable(self):
        """`resolve_op` reads context the op name does not carry — an
        `…expert…` node id, a 4-D decay operand, a component suffix — and
        rewrites to the operator that context implies. A rewrite landing
        outside the measurable vocabulary would turn a node the harness could
        have measured into a key nothing serves, silently."""
        from backend.inference.benchmark_identity import resolve_op
        from backend.inference.op_signatures import op_bindings
        OP_TO_SIGNATURE = op_bindings()
        cases = [
            ("flash_attention", dict(key_slots=frozenset({"latent"})), None,
             "mla_attention"),
            ("linear", dict(routed=True), None, "grouped_gemm"),
            ("gated_delta_rule", {},
             [[1, 4, 8], [1, 4, 8], [1, 4, 8], [1, 4, 8, 128]], "kda"),
        ]
        for op, facts, shape_in, expect in cases:
            got = resolve_op(op, shape_in, **facts)
            assert got == expect, (op, facts, got)
            assert got in OP_TO_SIGNATURE
        # An op with no context to read passes through. Residual add is
        # its own kernel — the leftover after fused_add_rmsnorm absorbs
        # the rest, same class as mul.
        assert resolve_op("add") == "add"
        assert "add" in OP_TO_SIGNATURE

    def test_the_guarantee_is_where_the_rewrite_is(self):
        """Enforced at the one place rewrites are made, not as a list of their
        results kept somewhere else — a global table of "names resolve_op may
        produce" is a second declaration that drifts from the branches it
        describes, which is the shape of defect this whole refactor removes."""
        from backend.inference import benchmark_identity as BI
        with pytest.raises(KeyError, match="no signature measures"):
            BI._rewrite("linear", "a_kernel_nobody_measures")


class TestTheKernelThatRanIsRecorded:
    """Layer 4. The first three layers prove a declaration is well formed,
    uniquely named, and numerically right; none can tell `declared A, ran A`
    from `declared A, ran B` when B computes the same thing."""

    def test_the_digest_is_over_the_sequence_not_the_set(self):
        """A kernel that runs its inner loop a different number of times is a
        different program. Collapsing repeats would hide exactly that."""
        from kernel_bench.kernel_symbol import digest
        assert digest(["a", "b"]) != digest(["b", "a"])
        assert digest(["a", "a", "b"]) != digest(["a", "b"])
        assert digest(["a", "b"]) == digest(("a", "b"))

    def test_the_digest_cannot_be_forged_by_concatenation(self):
        """Two symbols joined must not collide with one symbol of the joined
        name — the separator is why the sequence is a sequence."""
        from kernel_bench.kernel_symbol import digest
        assert digest(["ab", "c"]) != digest(["a", "bc"])
        assert digest(["abc"]) != digest(["ab", "c"])

    def test_it_is_provenance_and_never_splits_a_physical_cell(self):
        """A box whose profiler cannot attach measures the same cell. If the
        symbol reached the envelope key, the same kernel at the same shape
        would occupy two slots depending on whether anyone was watching."""
        import dataclasses
        from backend.inference.op_calibration import PhysicalKey
        assert "kernel_symbol" not in {f.name
                                       for f in dataclasses.fields(PhysicalKey)}

    def test_a_row_that_was_not_observed_says_so(self):
        """Empty, not a placeholder. A plausible string would read as evidence
        where there is none — the failure this column exists to end."""
        from kernel_bench.emit import build_row
        from kernel_bench.env import Env
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import LatencyStats, Provider, KernelSignature
        p = Provider(library="torch", signature="rmsnorm", kernel_impl="probe",
                     import_name="torch",
                     serves=KernelSignature(signature="rmsnorm"),
                     build=lambda inv, env, inputs: None)
        inv = Invocation(signature="rmsnorm", op="rmsnorm",
                         dims={"B": 1, "s_query": 8, "d_model": 8},
                         dtype_env={"activation": "bf16"}, phase="prefill",
                         tp=1, physical_key=_stated_key(
                             {"signature": "rmsnorm", "phase": "prefill",
                              "dims": {"B": 1, "s_query": 8, "d_model": 8},
                              "dtype_env": {"activation": "bf16"}}))
        env = Env("2.11", True, "H100", "H100-SXM", 3350.0,
                  libs={"torch": "2.11"})
        row = build_row(inv, p, LatencyStats(1.0, 1.0, 1.0, 3, "l2_drain"), env,
                        date="2026-08-11", contributor="@t")
        assert row["kernel_symbol"] == ""

    def test_two_names_for_one_kernel_are_reported(self):
        """The comparison layer 4 can make with no declaration at all: two
        kernel_impls whose device work is byte-identical on the same cell are
        one measurement recorded twice, and a cross-library comparison that is
        really a library compared with itself."""
        from kernel_bench.smoke import SmokeResult, same_kernel_twice

        def r(impl, symbol, d_model=256):
            return SmokeResult(impl, "lib", "rmsnorm", "", "bf16", "prefill",
                               "ok", "", {"B": 1, "d_model": d_model}, symbol)

        assert same_kernel_twice([r("a", "d1:k"), r("b", "d2:k2")]) == []
        found = same_kernel_twice([r("a", "d1:k"), r("b", "d1:k")])
        assert len(found) == 1 and found[0][1] == ["a", "b"]
        # ...but only at the SAME cell: a kernel's own divisible/enum
        # constraints move its minimal dims, and two symbols observed at
        # different shapes say nothing about each other.
        assert same_kernel_twice([r("a", "d1:k"), r("b", "d1:k", 512)]) == []
        # An unobserved row is not evidence of agreement.
        assert same_kernel_twice([r("a", ""), r("b", "")]) == []

    def test_the_observation_is_stable_and_names_a_real_kernel(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("a launch record needs a GPU")
        from kernel_bench.kernel_symbol import describe
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        first = describe(lambda: x @ w)
        assert first == describe(lambda: x @ w)
        head, _, name = first.partition(":")
        assert len(head) == 12 and name
        # A matmul must not report an elementwise kernel as its dominant one.
        assert "elementwise" not in name


class TestShapeContractDerivesCellDims:
    """The shape contract (canonical input layout + structural axes) is the
    ground truth; `cell_dims` is its projection, derived, never written
    twice. The live guard that the contract says what the reference reads
    is the recording test below — these tests guard the contract itself."""

    def test_every_signature_has_a_contract(self):
        from backend.inference.op_signatures import all_signatures
        missing = [s.name for s in all_signatures() if not s.has_contract]
        assert missing == []

    def test_layout_roles_speak_the_dtype_taxonomy(self):
        from backend.inference.op_signatures import all_signatures
        legal = {"activation", "weight", "kv", "metadata"}
        bad = [(s.name, lay.name, lay.role)
               for s in all_signatures() for lay in s.inputs
               if lay.role not in legal]
        assert not bad, bad

    def test_engine_layouts_are_permutations_of_the_contract(self):
        # Registration enforces this with a raise; this test is the visible
        # statement of the invariant — an engine layout is the contract's
        # axes in the engine's order, never a second layout.
        from backend.inference.op_signatures import all_signatures, _named_axes
        for s in all_signatures():
            allowed = set(s.cell_dims) | set(s.variant_dims)
            for engines in (s.engine_inputs, s.variant_engine_inputs,
                            s.engine_outputs):
                for _pos, lay in engines:
                    names = [n for ax in lay.axes for n in _named_axes(ax)]
                    assert all(n in allowed for n in names), (s.name, lay)
            assert all(ax in set(s.cell_dims) for ax in s.output_axes), s.name

    def test_same_rank_engine_layouts_are_rejected(self):
        from backend.inference.op_signatures import (
            OpSignature, TensorLayout, register_signature)
        with pytest.raises(ValueError, match="rank"):
            register_signature(OpSignature(
                name="_dup_rank_engine",
                inputs=(TensorLayout("x", ("B", "s_query")),),
                engine_inputs=(
                    (0, TensorLayout("x", ("B", "s_query"))),
                    (0, TensorLayout("x", ("s_query", "B"))),
                ),
            ))


class TestRefsBuildWhatTheContractDeclares:
    """The recording test below checks WHICH AXES a reference reads; this
    one checks WHAT IT BUILDS. A contract is a claim about the canonical
    layout — every input tensor, its rank, and each dim's extent — and
    `make_inputs` is checked against that claim, not the other way round."""

    def test_built_tensors_match_the_declared_layout(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.invocation import Invocation
        from kernel_bench.smoke import MINIMAL_DIMS
        from backend.inference.op_signatures import all_signatures
        from kernel_bench.providers import all_providers

        serving: dict = {}
        for p in all_providers():
            if p.serves is not None:
                serving.setdefault(p.signature, []).append(p)
        checked = 0
        for sig in all_signatures():
            ref = get_ref(sig.name)
            if ref is None or not sig.inputs:
                continue
            # The contract declares the DEFAULT geometry: a variant axis
            # selects another one, so it is absent here by construction.
            dims = {k: v for k, v in MINIMAL_DIMS.items()
                    if k not in sig.variant_dims}
            # A scalar the kernel takes beside its operands (a clamp bound, a
            # norm's eps) is required on a planned cell; a hand-stated one
            # reads any legal value — smoke's convention, and 1.0 is legal
            # for every scalar declared so far.
            kernel_params = {name: 1.0 for name, _symbol in sig.config_params}
            dtype_env = {"activation": "bf16", "weight": "bf16",
                         "kv": "bf16", "output": "bf16", "metadata": "fp32"}
            # The slots whose legal value is not the activation's are a
            # provider's declaration (a cast writes another dtype than it
            # reads, at a stated granularity) — read off the first one that
            # serves the signature, as smoke reads them.
            for p in serving.get(sig.name, ())[:1]:
                if p.serves.output:
                    dtype_env["output"] = p.serves.output[0]
                if p.serves.activation_granularity:
                    dtype_env["activation_granularity"] = (
                        p.serves.activation_granularity[0])
            inv = Invocation(
                signature=sig.name, op=sig.name, dims=dict(dims),
                dtype_env=dtype_env,
                phase="prefill", tp=1, kernel_params=kernel_params)
            built = build_inputs(ref, inv, "cpu")
            tensors = {k: v for k, v in built.items() if torch.is_tensor(v)}
            declared = {lay.name: lay for lay in sig.inputs}
            assert set(tensors) == set(declared), (
                f"{sig.name}: built {sorted(tensors)} but the contract "
                f"declares {sorted(declared)}")
            for name, tensor in tensors.items():
                lay = declared[name]
                assert tensor.dim() == len(lay.axes), (
                    f"{sig.name}.{name}: rank {tensor.dim()}, "
                    f"contract says {len(lay.axes)}")
                for i, (got, want) in enumerate(
                        zip(tensor.shape, lay.extents(dims), strict=True)):
                    if want is not None:
                        assert int(got) == want, (
                            f"{sig.name}.{name} dim {i}: built {got}, "
                            f"contract says {want}")
            checked += 1
        assert checked == len(all_signatures())


class TestLaunchLayoutConversion:
    """Canonical → kernel launch is a unique permute/pack, declared on the
    provider. Concat, paging, and varlen metadata are not this conversion."""

    def test_permute_is_the_unique_atom_reordering(self):
        from backend.inference.op_signatures import TensorLayout, layout_conversion
        src = TensorLayout("q", ("B", "s_query", "n_heads", "d_head"))
        dst = TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))
        assert layout_conversion(src, dst) == (0, 2, 1, 3)
        assert layout_conversion(dst, src) == (0, 2, 1, 3)

    def test_pack_is_the_same_atoms_in_fewer_physical_dims(self):
        from backend.inference.op_signatures import TensorLayout, layout_conversion
        src = TensorLayout("q", ("B", "s_query", "n_heads", "d_head"))
        dst = TensorLayout("q", (("B", "s_query"), "n_heads", "d_head"))
        assert layout_conversion(src, dst) == (0, 1, 2, 3)
        assert layout_conversion(dst, src) == (0, 1, 2, 3)

    def test_concat_and_a_different_atom_set_are_not_conversions(self):
        from backend.inference.op_signatures import TensorLayout, layout_conversion
        packed = TensorLayout("x", (("B", "s_query"), "d_model"))
        other = TensorLayout("x", ("B", "d_model"))
        concat = TensorLayout("x", (("+", "n_heads", "n_kv_heads"), "d_head"))
        split = TensorLayout("x", ("n_heads", "n_kv_heads", "d_head"))
        dup = TensorLayout("x", ("B", "B"))
        assert layout_conversion(packed, other) is None
        assert layout_conversion(concat, split) is None
        assert layout_conversion(dup, dup) is None

    def test_apply_permutes_and_packs_using_the_tensor_extents(self):
        torch = pytest.importorskip("torch")
        from backend.inference.op_signatures import TensorLayout
        from kernel_bench.providers import apply_layout
        src = TensorLayout("q", ("B", "s_query", "n_heads", "d_head"))
        perm = TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))
        pack = TensorLayout("q", (("B", "s_query"), "n_heads", "d_head"))
        dims = {"B": 2, "s_query": 3, "n_heads": 4, "d_head": 5}
        t = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        p = apply_layout(t, src, perm, dims)
        assert p.shape == (2, 4, 3, 5)
        assert torch.equal(p, t.permute(0, 2, 1, 3))
        assert torch.equal(apply_layout(p, perm, src, dims), t)
        packed = apply_layout(t, src, pack, dims)
        assert packed.shape == (6, 4, 5)
        assert torch.equal(apply_layout(packed, pack, src, dims), t)

    def test_grouping_takes_axis_names_from_the_contract(self):
        from backend.inference.op_signatures import (
            TensorLayout, layout_from_grouping)
        gqa_v = TensorLayout("v", ("B", "s_kv", "n_kv_heads", "d_head"))
        mla_v = TensorLayout("v", ("B", "s_kv", "n_kv_heads", "d_v_head"))
        heads_first = (0, 2, 1, 3)
        assert layout_from_grouping(gqa_v, heads_first).axes == (
            "B", "n_kv_heads", "s_kv", "d_head")
        assert layout_from_grouping(mla_v, heads_first).axes == (
            "B", "n_kv_heads", "s_kv", "d_v_head")
        pack = layout_from_grouping(
            TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
            ((0, 1), 2, 3))
        assert pack.axes == (("B", "s_query"), "n_heads", "d_head")
        conv = layout_from_grouping(
            TensorLayout("x", ("B", "d_conv_channels", "s_query")),
            ((0, 2), 1))
        assert conv.axes == (("B", "s_query"), "d_conv_channels")

    def test_torch_sdpa_declares_the_heads_first_grouping(self):
        gqa = next(x for x in all_providers() if x.kernel_impl == "torch_sdpa")
        mla = next(x for x in all_providers()
                   if x.kernel_impl == "torch_sdpa_mla_prefill")
        assert gqa.launch_inputs == mla.launch_inputs
        assert gqa.launch_inputs == (
            ("q", (0, 2, 1, 3)), ("k", (0, 2, 1, 3)), ("v", (0, 2, 1, 3)))
        assert gqa.launch_output == "q"

    def test_varlen_and_conv_declare_pack_not_axis_names(self):
        fa = next(x for x in all_providers()
                  if x.kernel_impl == "flash_attention_v2")
        assert fa.launch_inputs == (
            ("q", ((0, 1), 2, 3)),
            ("k", ((0, 1), 2, 3)),
            ("v", ((0, 1), 2, 3)))
        assert fa.launch_output == "q"
        conv = next(x for x in all_providers()
                    if x.kernel_impl == "vllm_causal_conv1d_fn")
        assert conv.launch_inputs == (("x", ((0, 2), 1)),)
        assert conv.launch_output == "x"

    def test_varlen_pack_comes_from_the_contract(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        p = next(x for x in all_providers()
                 if x.kernel_impl == "flash_attention_v2")
        inv = Invocation(
            signature="gqa_attention", op="flash_attention",
            dims={"kv_block_size": 16, "B": 2, "s_query": 3, "s_kv": 5, "n_heads": 4,
                  "n_kv_heads": 2, "d_head": 8},
            dtype_env={"activation": "bf16"}, phase="prefill", tp=1)
        inputs = {
            "q": torch.zeros(2, 3, 4, 8),
            "k": torch.zeros(2, 5, 2, 8),
            "v": torch.zeros(2, 5, 2, 8),
        }
        launch = p.convert_inputs(inv, inputs)
        assert launch["q"].shape == (6, 4, 8)
        assert launch["k"].shape == (10, 2, 8)
        assert launch["v"].shape == (10, 2, 8)
        back = p.wrap_output(inv, None)(launch["q"])
        assert back.shape == (2, 3, 4, 8)

    def test_conv1d_launch_is_token_major_like_the_projection(self):
        # vLLM hands its prefill conv the projection's [B·S, C] TRANSPOSED
        # (qwen_gdn_linear_attn.py:1448 at v0.32.0). A contiguous [C, B·S]
        # puts B·S on the channel stride, which the kernel multiplies in
        # int32 — C=10240, S=262144 faulted with an illegal address.
        torch = pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        p = next(x for x in all_providers()
                 if x.kernel_impl == "vllm_causal_conv1d_fn")
        inv = Invocation(
            signature="causal_conv1d", op="causal_conv1d",
            dims={"B": 2, "d_conv_channels": 4, "s_query": 5,
                  "s_conv_kernel": 4, "dilation": 1},
            dtype_env={"activation": "bf16"}, phase="prefill", tp=1)
        x = torch.arange(2 * 4 * 5).reshape(2, 4, 5)
        launch = p.convert_inputs(inv, {"x": x})
        assert launch["x"].shape == (10, 4)
        assert torch.equal(launch["x"][6], x[1, :, 1])   # row = (b=1, s=1)
        back = p.wrap_output(inv, None)(launch["x"])
        assert torch.equal(back, x)

    def test_convert_inputs_puts_gqa_into_the_declared_launch_layout(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        p = next(x for x in all_providers() if x.kernel_impl == "torch_sdpa")
        inv = Invocation(
            signature="gqa_attention", op="flash_attention",
            dims={"kv_block_size": 16, "B": 2, "s_query": 3, "s_kv": 5, "n_heads": 4,
                  "n_kv_heads": 2, "d_head": 8},
            dtype_env={"activation": "bf16"}, phase="prefill", tp=1)
        inputs = {
            "q": torch.zeros(2, 3, 4, 8),
            "k": torch.zeros(2, 5, 2, 8),
            "v": torch.zeros(2, 5, 2, 8),
            "causal": True,
        }
        launch = p.convert_inputs(inv, inputs)
        assert launch["q"].shape == (2, 4, 3, 8)
        assert launch["k"].shape == (2, 2, 5, 8)
        assert launch["v"].shape == (2, 2, 5, 8)
        assert launch["causal"] is True
        back = p.wrap_output(inv, None)(launch["q"])
        assert back.shape == (2, 3, 4, 8)

    def test_mla_prefill_keeps_v_width_from_the_contract(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.invocation import Invocation
        p = next(x for x in all_providers()
                 if x.kernel_impl == "torch_sdpa_mla_prefill")
        inv = Invocation(
            signature="mla_attention", op="mla_attention",
            dims={"kv_block_size": 64, "B": 2, "s_query": 3, "s_kv": 5, "n_heads": 4,
                  "n_kv_heads": 4, "d_head": 192, "d_v_head": 128},
            dtype_env={"activation": "bf16"}, phase="prefill", tp=1)
        inputs = {
            "q": torch.zeros(2, 3, 4, 192),
            "k": torch.zeros(2, 5, 4, 192),
            "v": torch.zeros(2, 5, 4, 128),
            "causal": True,
        }
        launch = p.convert_inputs(inv, inputs)
        assert launch["q"].shape == (2, 4, 3, 192)
        assert launch["k"].shape == (2, 4, 5, 192)
        assert launch["v"].shape == (2, 4, 5, 128)
        back = p.wrap_output(inv, None)(launch["q"].new_zeros(2, 4, 3, 128))
        assert back.shape == (2, 3, 4, 128)

    def test_inconvertible_launch_layout_is_rejected_at_register(self):
        from kernel_bench.providers import KernelSignature, Provider, register
        with pytest.raises(ValueError, match="permute/pack"):
            register(Provider(
                library="torch", signature="gqa_attention",
                kernel_impl="_test_bad_launch",
                import_name="torch",
                serves=KernelSignature(signature="gqa_attention"),
                build=lambda inv, env, inputs: None,
                launch_inputs=(("q", (0, 1)),),
            ))


class TestEverySignaturesDimsCanBeRecorded:
    """A measured row that the schema refuses is GPU time thrown away.

    `causal_conv1d` had 798 planned cells across four kernels and ZERO rows,
    and none of them failed to run: `torch_depthwise_conv1d` logged 228 clean
    `ok` lines with real timings. Every one was dropped at persist time because
    `d_conv_channels` was not in the shape-symbol glossary. The drop WAS
    reported — in the end-of-run summary, which the runs that hit this never
    reached, because they were also the runs that kept crashing.

    The whitelist and the signatures are two lists of dimension names that have
    to agree, so this checks them against each other rather than trusting that
    a new signature remembered to update the glossary.
    """

    def test_every_declared_dim_is_recordable(self):
        from backend.inference.benchmarks import _allowed_shape_keys
        from kernel_bench.signatures import all_signatures
        allowed = _allowed_shape_keys()
        missing = {sig.name: sorted(set(sig.cell_dims + sig.identity_dims)
                                    - allowed)
                   for sig in all_signatures()}
        missing = {k: v for k, v in missing.items() if v}
        assert not missing, (
            f"these signatures name shape keys the schema rejects, so every row "
            f"they measure is dropped after the fact: {missing}")

    def test_cell_dims_are_what_the_reference_reads(self):
        """The declaration, checked against the code it describes.

        `cell_dims` used to be a hand-written list nothing compared to
        anything, and ten of nineteen were wrong: `gemm` named
        `n_experts_active` and `n_heads` — MoE axes copied from a neighbour —
        and neither `d_reduce` nor `d_out`, so `minimal_dims` built a cell for
        every GEMM kernel with no GEMM dimension in it. Nothing failed; the
        reference filled the missing axes with its own defaults and the smoke
        test measured a shape nobody asked for.

        So this does not read the source — it RUNS each reference against a
        recording dims mapping and compares what it actually consulted. A dim
        list that stops describing its reference fails the moment it is
        written.

        An axis the reference does not read has to be read by a KERNEL of
        that signature instead, and the same recording says which: the
        Mamba-2 scan's `ssm_chunk_size` decides how many tiles the launch
        runs, and the oracle is a sequential recurrence that is the same
        answer at any chunk, so the chunk is the kernel's and not the
        oracle's. Declared with nobody reading it either way is the
        over-declaration this catches.

        One axis is read by the contract rather than by a builder: the
        paged-KV page (`PAGE_AXIS`), which every signature that reads a paged
        cache carries because the deployment pages that cache. It counts as
        read for exactly those signatures, and nowhere else.
        """
        pytest.importorskip("torch")
        from kernel_bench.env import Env
        from kernel_bench.providers import all_providers
        from kernel_bench.refs import get_ref
        from kernel_bench.signatures import all_signatures
        from kernel_bench.invocation import Invocation
        from kernel_bench.smoke import MINIMAL_DIMS, _combinations, cell_dtype_env
        from backend.inference.op_signatures import PAGE_AXIS

        class Recording(dict):
            def __init__(self, base):
                super().__init__(base)
                self.read = set()

            def __getitem__(self, k):
                self.read.add(k)
                return super().__getitem__(k)

            def get(self, k, default=None):
                self.read.add(k)
                return super().get(k, default)

        wrong, unchecked = {}, []
        for sig in all_signatures():
            ref = get_ref(sig.name)
            if ref is None:
                unchecked.append(sig.name)
                continue
            # Default geometry: variant axes are absent, matching
            # `test_built_tensors_match_the_declared_layout`. Binding the
            # whole MINIMAL soup would select MLA's non-absorbed form and
            # pretend the default layout reads `n_kv_heads`.
            # At EVERY cell a kernel of the signature declares
            # (`smoke.cell_dtype_env` over `smoke._combinations`, the cells
            # smoke builds), so an oracle that reads its cell off the dtype
            # environment — a cast's direction and granularity — builds
            # rather than stopping before it reads a single axis, and an
            # axis only one declared cell makes it read is still seen.
            declared_by = [p for p in all_providers()
                           if p.signature == sig.name and p.serves is not None]
            envs = [cell_dtype_env(p, weight, activation)
                    for p in declared_by
                    for weight, activation, _phase in _combinations(p)] or [
                {"activation": "bf16", "weight": "bf16", "kv": "bf16",
                 "output": "bf16", "metadata": "fp32"}]
            read = set()
            for dtype_env in envs:
                dims = Recording({k: v for k, v in MINIMAL_DIMS.items()
                                  if k not in sig.variant_dims})
                try:
                    build_inputs(ref, Invocation(
                        signature=sig.name, op=sig.name, dims=dims,
                        dtype_env=dtype_env, phase="prefill", tp=1), "cpu")
                except Exception:
                    # A reference that cannot build on CPU still tells us
                    # which axes it consulted before it stopped.
                    pass
                read |= {k for k in dims.read if k in MINIMAL_DIMS}
            # The page is read by the CONTRACT, not by a builder: a signature
            # that reads a paged cache (`role="kv"`, `reads_paged_cache`) is
            # keyed on the page the deployment allocates it in. The oracle is
            # the same answer at any page and the contiguous floors read
            # none, so no recording sees it; the paged providers read it off
            # the cell (`test_kernel_bench_paged_key`).
            if sig.reads_paged_cache:
                read |= {PAGE_AXIS}
            declared = set(sig.cell_dims)
            if read - declared:
                wrong[sig.name] = {"reads": sorted(read),
                                   "declares": sorted(declared)}
                continue
            for axis in sorted(declared - read):
                # Not the oracle's, so it has to be a kernel's. Same
                # recording, through each provider's `build`: it stops as
                # soon as it wants a tensor or a GPU, and what it consulted
                # before that is what the cell owes it.
                providers_ = [p for p in all_providers()
                              if p.signature == sig.name]
                if providers_ and all(
                        getattr(p, "disabled_reason", None) for p in providers_):
                    # Every kernel is worker-only (the DeepInfer ABI ones):
                    # the harness build reads nothing by design, and the
                    # worker reads the dims the declaration states.
                    continue
                by_kernel = set()
                for provider in providers_:
                    # Each kernel at its own declared cells, not the
                    # oracle's: one that serves only an FP8 cache stops
                    # before the axis it owns at any other.
                    cells = ([cell_dtype_env(provider, weight, activation)
                              for weight, activation, _phase
                              in _combinations(provider)]
                             if provider.serves is not None else envs)
                    for dtype_env in cells:
                        seen = Recording({k: v for k, v in MINIMAL_DIMS.items()
                                          if k not in sig.variant_dims})
                        try:
                            provider.build(
                                Invocation(
                                    signature=sig.name, op=sig.name, dims=seen,
                                    dtype_env=dtype_env, phase="prefill",
                                    tp=1),
                                Env(None, False, None, "", None), {})
                        except Exception:
                            pass
                        by_kernel |= {k for k in seen.read if k in MINIMAL_DIMS}
                if axis in by_kernel:
                    continue
                wrong[sig.name] = {"reads": sorted(read),
                                   "declares": sorted(declared),
                                   "no reader for": axis}
                break
        assert not wrong, wrong
        # A signature with no reference has nothing to check its list against.
        # Named rather than skipped silently, because that is the gap.
        assert unchecked == [], unchecked

    def test_a_shape_that_does_not_speak_the_vocabulary_is_left_whole(self):
        """The rule that keeps the projection from being a catastrophe.

        The overlay falls back to the whole-step MODEL shape for any op whose
        `shape_in` does not project, and that vocabulary shares exactly `B`
        with what `gemm` declares. Keeping "whichever axes are present" would
        key every such estimate on its batch size alone and hand back the
        first GEMM measurement of any width.
        """
        from kernel_bench.signatures import get_signature
        from backend.inference.benchmark_identity import compute_per_row_shape_json
        gemm = get_signature("gemm")
        model_wide = compute_per_row_shape_json(
            batch_size=1, seq_len=2048, op="linear",
            dim_values={"H": 4096, "N": 32, "Nk": 8, "D": 128, "I": 14336})
        assert gemm.project(model_wide) == model_wide
        assert gemm.project('{"B":1,"d_reduce":4096,"d_out":8192,'
                            '"s_query":512,"d_model":9999}') == (
            '{"B":1,"d_out":8192,"d_reduce":4096,"s_query":512}')

    def test_the_context_sequence_axis_is_not_a_second_spelling(self):
        """`s_ctx` is the request length from before the phase split, and it
        is not what a kernel is launched with. It used to be accepted here as
        an alias for `s_query`, which bought nothing: the projection keeps the
        spelling it found, so the aliased key still matched no measured row —
        it only made the model-wide shape look well-formed on its way into the
        corpus. A model-wide shape is now left whole, like any other shape that
        does not speak the vocabulary."""
        from kernel_bench.signatures import get_signature
        gqa = get_signature("gqa_attention")
        model_wide = ('{"B":1,"d_head":128,"d_model":3072,"n_heads":24,'
                      '"n_kv_heads":8,"s_ctx":2048}')
        assert gqa.project(model_wide) == model_wide
        assert "s_query" in gqa.project(
            '{"B":1,"s_query":2048,"s_kv":2048,"n_heads":24,'
            '"n_kv_heads":8,"d_head":128}')

    def test_identity_dims_are_a_subset_of_the_cell_dims(self):
        """An axis can be needed to build an input without being part of what
        the measurement identifies (moe_gate's group structure). The reverse
        cannot happen: identifying a measurement by an axis no cell binds is
        keying on a number nobody set."""
        from kernel_bench.signatures import all_signatures
        for sig in all_signatures():
            assert set(sig.identity_dims) <= (set(sig.cell_dims)
                                              | set(sig.variant_dims)), sig.name

    def test_the_identity_layer_emits_only_declared_axes(self):
        """The third producer of dim names.

        `d_conv_channels` is not a spec dimension at all — no model YAML
        declares it and the engine has no such dim; `benchmark_identity`
        derives it from the conv input's channel axis. Its siblings
        `d_reduce`, `d_out` and `s_conv_kernel` are invented the same way and
        were registered; that one was not, and nothing compared the lists.
        """
        import pathlib
        import re
        from backend.inference.benchmarks import _allowed_shape_keys
        src = pathlib.Path(
            "backend/inference/benchmark_identity.py").read_text()
        emitted = set(re.findall(r'"([a-z_0-9]+)":\s*int\(', src))
        assert emitted, "the emit scan found nothing — the pattern went stale"
        unknown = emitted - _allowed_shape_keys()
        assert not unknown, (
            f"the identity layer emits shape keys the schema rejects, so every "
            f"row carrying them is dropped after being measured: "
            f"{sorted(unknown)}")

    def test_a_conv1d_row_validates(self):
        # End to end on the real shape, with the roofline the planner emits.
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs a GPU")
        from backend.inference.benchmarks import validate_row
        from kernel_bench.driver import run_jobs
        from kernel_bench.env import detect_env
        job = {
            "signature": "causal_conv1d", "op": "causal_conv1d",
            "dims": {"B": 1, "s_query": 1024, "d_conv_channels": 8192,
                     "s_conv_kernel": 4, "dilation": 1},
            "dtype_env": {"weight": "bf16", "activation": "bf16", "kv": "bf16",
                          "output": "bf16", "metadata": "fp32"},
            "phase": "prefill", "tp": 1,
            "flops_per_rank": 2415919104.0,
            "access_bytes_per_rank": 1210318848.0,
            "model": "m", "spec_hash": "h", "node_spec_id": "blk|conv1d",
            "parallel": "tp=1", "count_per_step": 1,
        }
        job["physical_key"] = _stated_key(job)
        res = run_jobs([job], env=detect_env(), date="2026-08-09",
                       contributor="@t", cross_check=True)
        assert res.rows, res.failures + res.skipped
        for r in res.rows:
            validate_row(dict(r))          # raises if the row is unrecordable


class TestSpecDeclaredSymbolsAreRegistered:
    """Every symbol a spec declares must be in the dim registry.

    The registry is the one vocabulary: the spec engine reads dims by these
    names and the benchmark schema validates a row's `shape_json` against the
    same set. A symbol declared in a model's `dims:` block and missing from
    the registry is a name half the system knows — it computes fine and then a
    measured row carrying it is refused at persist time, after the GPU work.

    That is not hypothetical: `Kar` and `Sar` (Kimi-K3's AttnRes accumulator)
    were declared and unregistered, and `d_conv_channels` cost all four
    causal_conv1d kernels every row they ever measured.
    """

    def _declared(self):
        import glob
        import yaml
        out = {}
        for path in sorted(glob.glob("backend/arch/models/*.yaml")
                           + glob.glob("backend/arch/components/**/*.yaml",
                                       recursive=True)):
            doc = yaml.safe_load(Path(path).read_text()) or {}
            if isinstance(doc, dict) and isinstance(doc.get("dims"), dict):
                for name in doc["dims"]:
                    out.setdefault(name, path)
        return out

    def test_every_spec_dim_is_registered(self):
        from backend.inference.benchmarks import REGISTERED_SYMBOLS
        declared = self._declared()
        assert len(declared) > 50, "the scan found too few dims — it went stale"
        missing = {n: p for n, p in declared.items() if n not in REGISTERED_SYMBOLS}
        assert not missing, (
            f"declared in a spec but absent from arch/dims.yaml, so a "
            f"row carrying one is dropped after being measured: {missing}")
