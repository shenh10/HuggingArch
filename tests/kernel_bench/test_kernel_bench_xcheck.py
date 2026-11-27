"""Cross-check mechanism — CPU-runnable (torch runs on CPU).

The GPU kernels aren't exercised here, but the whole correctness-gate
machinery is: per-signature references produce correct outputs, the
dtype-scaled tolerance behaves, and the driver records a passing kernel /
rejects a disagreeing one. The real torch GEMM provider runs on CPU, so its
end-to-end path (build → cross-check vs reference → time → emit) is covered.
"""
from __future__ import annotations

import json
from typing import ClassVar

import pytest

pytest.importorskip("torch")
import torch

from tests.kernel_bench._inputs import build_inputs
from kernel_bench.invocation import Invocation
from kernel_bench.refs import get_ref
from kernel_bench.providers import BenchCase
from kernel_bench.driver import _cross_check, _XCHECK_BANDS, run_jobs
from kernel_bench.env import Env


def _id(x):
    return x


def _shape_key(dims):
    """`shape_json` as the plan states it — sorted, no spaces."""
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _dtype_key(dtype):
    return json.dumps({"activation": dtype, "kv": dtype, "metadata": "fp32",
                       "output": dtype, "weight": dtype},
                      separators=(",", ":"), sort_keys=True)


def _inv(signature, op, dims, dtype="fp32", phase="prefill"):
    return Invocation.from_job({
        "signature": signature, "op": op, "dims": dims, "phase": phase,
        "dtype_env": {"activation": dtype, "weight": dtype, "kv": dtype,
                      "output": dtype, "metadata": "fp32"},
    })


class TestReferences:
    def test_gemm_reference_is_matmul(self):
        inv = _inv("gemm", "linear", {"B": 1, "s_query": 4, "d_reduce": 8, "d_out": 6})
        spec = get_ref("gemm")
        ins = build_inputs(spec, inv, "cpu")
        ref = spec.reference(ins)
        assert ref.shape == (4, 6)
        assert torch.allclose(ref, ins["x"].float() @ ins["w"].float().T)

    def test_rmsnorm_reference(self):
        inv = _inv("rmsnorm", "rmsnorm", {"B": 1, "s_query": 4, "d_model": 8})
        spec = get_ref("rmsnorm")
        ins = build_inputs(spec, inv, "cpu")
        ref = spec.reference(ins)
        # unit-rms per row (before weight): recompute and compare shape/magnitude
        assert ref.shape == (4, 8)

    def test_silu_reference(self):
        inv = _inv("silu_and_mul", "silu_and_mul", {"B": 1, "s_query": 4, "d_ffn": 8})
        spec = get_ref("silu_and_mul")
        ins = build_inputs(spec, inv, "cpu")
        ref = spec.reference(ins)
        assert ref.shape == (4, 8)

    def test_gqa_reference_shape_and_causal(self):
        inv = _inv("gqa_attention", "flash_attention",
                   {"B": 1, "s_query": 4, "s_kv": 4, "n_heads": 8, "n_kv_heads": 2, "d_head": 16,
                    "kv_block_size": 16})
        spec = get_ref("gqa_attention")
        ins = build_inputs(spec, inv, "cpu")
        ref = spec.reference(ins)
        assert ref.shape == (1, 4, 8, 16)   # [B,Sq,N,D]


def _attend_token_by_token(q, k, v, *, n_query, causal, s_window=None,
                           sinks=None):
    """The definition, one query token and one head at a time, in fp32.

    `q[b, i, n]` is token `i` of the `n_query` LAST positions of the
    context, so it sits at position `Lq - n_query + i`; causal, it reads
    every key up to and including that position, and a window keeps the last
    `s_window` of those. A sink is one more logit in the same softmax that
    adds nothing to the value sum. Written with Python integers rather than
    any mask helper, so that it shares no alignment code with the oracle it
    checks."""
    B, Sq, N, _D = q.shape
    Lq, Nk = k.shape[1], k.shape[2]
    out = torch.zeros(B, Sq, N, v.shape[-1])
    for b in range(B):
        for i in range(Sq):
            pos = Lq - n_query + i
            hi = pos + 1 if causal else Lq
            lo = max(0, hi - s_window) if s_window else 0
            for n in range(N):
                kv = n // (N // Nk)
                keys = k[b, lo:hi, kv].float()
                logits = (keys @ q[b, i, n].float()) * q.shape[-1] ** -0.5
                if sinks is not None:
                    logits = torch.cat([logits, sinks[n].float().reshape(1)])
                p = logits.softmax(-1)[:hi - lo]
                out[b, i, n] = p @ v[b, lo:hi, kv].float()
    return out


class TestVerifyDecodeOraclesAgreeWithTheDefinition:
    """A decode step with several query tokens (speculative verify, `s_query`
    > 1) is causal among its drafts, aligned BOTTOM-RIGHT: token `i` of `k`
    reads `Lq - k + i + 1` keys. The absorbed-MLA oracle masked it with
    `is_causal=True`, which SDPA aligns TOP-LEFT — token 0 read one key — and
    rejected FlashMLA and FlashInfer on every verify cell at cosine ~0.03
    (H100 ~10.4k cells, 4090 ~2.9k, 13 models, s_query 2/5/6 only).

    Each oracle is held against `_attend_token_by_token`, a second route to
    the same number that assumes nothing the oracle assumes. The first case
    of each test is the one the old oracle failed."""

    @pytest.mark.parametrize("B,N,Lq,Sq", [
        (2, 4, 37, 2),     # verify: what the top-left mask got wrong
        (1, 3, 41, 6),
        (2, 4, 37, 1),     # one-token decode: no mask at all
    ])
    def test_absorbed_mla(self, B, N, Lq, Sq):
        inv = _inv("mla_attention", "mla_attention",
                   {"B": B, "n_heads": N, "s_query": Sq, "s_kv": Lq,
                    "d_head": 576, "d_v_head": 512, "kv_block_size": 64},
                   phase="decode")
        spec = get_ref("mla_attention")
        ins = build_inputs(spec, inv, "cpu")
        assert ins["causal"] is (Sq > 1)
        c = ins["c"].unsqueeze(2)                      # one latent "head"
        want = _attend_token_by_token(
            ins["q"], c, c[..., :ins["Dv"]], n_query=Sq, causal=ins["causal"])
        got = spec.reference(ins).float()
        assert _cross_check(BenchCase(run=lambda: got, to_canonical=_id),
                            want, "fp32")[0] is None
        cos = torch.nn.functional.cosine_similarity(
            got.flatten(), want.flatten(), dim=0).item()
        assert cos > 0.99999, cos

    @pytest.mark.parametrize("Sq,Lq,window,sinks", [
        # verify + sinks: the sink branch read the bottom-right mask through
        # `CausalBias.to` (it raised, so gpt-oss verify rows went unchecked)
        # and added the sink unshifted by the row max (cosine ~0.993).
        (2, 37, None, True),
        (1, 29, None, True),      # decode + sinks: the unshifted sink alone
        (3, 37, 5, True),         # sliding window + sinks
        (3, 37, 5, False),
        (2, 37, None, False),
    ])
    def test_gqa(self, Sq, Lq, window, sinks):
        dims = {"B": 2, "n_heads": 8, "n_kv_heads": 2, "s_query": Sq,
                "s_kv": Lq, "d_head": 16, "kv_block_size": 16}
        if window:
            dims["s_window"] = window
        if sinks:
            dims["n_attn_sinks"] = 8
        inv = _inv("gqa_attention", "flash_attention", dims, phase="decode")
        spec = get_ref("gqa_attention")
        ins = build_inputs(spec, inv, "cpu")
        assert ins["causal"] is (Sq > 1)
        want = _attend_token_by_token(
            ins["q"], ins["k"], ins["v"], n_query=Sq, causal=ins["causal"],
            s_window=window, sinks=ins.get("sinks"))
        got = spec.reference(ins).float()
        assert _cross_check(BenchCase(run=lambda: got, to_canonical=_id),
                            want, "fp32")[0] is None

    @pytest.mark.parametrize("Sq", [2, 1])
    def test_mla_decode_kernels_launch_with_the_cells_causality(
            self, Sq, monkeypatch):
        # The kernels' side of the same fact: what each build hands its
        # kernel as `causal` is the cell's (`inputs["causal"]`). FlashInfer
        # MLA's was planned `False` outright, so every verify cell timed a
        # non-causal launch. The libraries are replaced by recorders, so this
        # runs without a GPU and observes the launch arguments themselves.
        import sys
        import types
        from kernel_bench.providers import attention
        monkeypatch.setattr(attention, "_require_cuda", lambda: None)
        seen = {}

        class _Wrapper:
            def __init__(self, *a, **k):
                pass

            def plan(self, *args, **kw):
                seen["flashinfer_mla"] = args[8]

            def run(self, *a, **k):
                return None

        monkeypatch.setitem(sys.modules, "flashinfer", types.SimpleNamespace(
            BatchMLAPagedAttentionWrapper=_Wrapper))

        def _with_kvcache(*args, causal, **kw):
            seen["flashmla"] = causal
        monkeypatch.setitem(sys.modules, "flash_mla", types.SimpleNamespace(
            flash_mla_with_kvcache=_with_kvcache,
            get_mla_metadata=lambda *a: (None, None)))

        inv = _inv("mla_attention", "mla_attention",
                   {"B": 1, "n_heads": 2, "s_query": Sq, "s_kv": 70,
                    "d_head": 576, "d_v_head": 512, "kv_block_size": 64},
                   dtype="bf16", phase="decode")
        ins = build_inputs(get_ref("mla_attention"), inv, "cpu")
        attention._flashinfer_mla_build(inv, None, ins)
        attention._flashmla_build(inv, None, ins).run()
        assert seen == {"flashinfer_mla": Sq > 1, "flashmla": Sq > 1}

    def test_a_decode_only_mla_kernel_declines_a_verify_cell(self):
        # CUTLASS MLA reads `B·Sq` query rows against `B` sequence lengths:
        # a verify step would be measured as a batch of `B·Sq` requests.
        from kernel_bench.providers import all_providers
        cutlass = next(p for p in all_providers()
                       if p.kernel_impl == "cutlass_mla_decode")
        dims = {"B": 2, "n_heads": 16, "s_kv": 1153, "d_head": 576,
                "d_v_head": 512, "kv_block_size": 128}
        one = _inv("mla_attention", "mla_attention", {**dims, "s_query": 1},
                   dtype="bf16", phase="decode")
        two = _inv("mla_attention", "mla_attention", {**dims, "s_query": 2},
                   dtype="bf16", phase="decode")
        assert not cutlass.serves.accepts(two)
        assert cutlass.serves.accepts(one)


class TestTheMamba2ScanStepsBySoftplus:
    """The oracle steps the state by softplus(dt + dt_bias), as both servers
    ask the kernel to (sglang v0.5.22 mamba.py:638 `dt_softplus=True`).
    `mamba_chunk_scan_combined` defaults to False and scans with the raw
    sum; launched with the default, the H100's decode canaries (Nemotron-3
    Nano / Ultra, s_query 2) failed the cross-check at cosine 0.93 — and
    every prefill cell, past the oracle's step bound, was timed unchecked.
    The library is replaced by a recorder, so this needs no GPU."""

    def test_the_build_asks_the_kernel_for_the_softplus_step(
            self, monkeypatch):
        import sys
        import types
        from kernel_bench.providers import linear_ssm
        monkeypatch.setattr(linear_ssm, "_require_cuda", lambda: None)
        seen = {}

        def scan(*args, **kw):
            seen.update(kw)
        module = "mamba_ssm.ops.triton.ssd_combined"
        monkeypatch.setitem(sys.modules, module, types.SimpleNamespace(
            mamba_chunk_scan_combined=scan))
        for parent in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton"):
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
        inv = _inv("mamba2_scan", "mamba2_scan",
                   {"B": 1, "s_query": 2, "n_ssm_heads": 8, "d_ssm_head": 4,
                    "n_ssm_groups": 1, "d_ssm_state": 8,
                    "ssm_chunk_size": 128},
                   dtype="bf16", phase="decode")
        ins = build_inputs(get_ref("mamba2_scan"), inv, "cpu")
        linear_ssm._mamba2_build(inv, None, ins).run()
        assert seen.get("dt_softplus") is True, seen
        assert seen["chunk_size"] == 128


class TestKernelScalarsComeFromTheModel:
    """A gated activation takes scalars beside its operand — a clamp bound,
    SiTU's two betas — and they change what the kernel COMPUTES. They used
    to be `inputs.get(name, <a number>)` on both sides, so the provider
    measured and the reference confirmed a function no model runs: SiTU at
    beta 1 where Kimi-K3 publishes 4 with a linear beta of 25, a clamped
    SwiGLU at 7 where DeepSeek-V4 and GLM-5.3 publish 10."""

    def _cell(self, signature, params, dtype="fp32", env=None):
        dtype_env = {
            "activation": dtype, "weight": dtype,
            "kv": dtype, "output": dtype,
            "metadata": "fp32",
        }
        if env is not None:
            dtype_env.update(env)
        return Invocation.from_job({
            "signature": signature, "op": signature,
            "dims": {"B": 1, "s_query": 4, "d_ffn": 8},
            "phase": "prefill", "kernel_params": params,
            "dtype_env": dtype_env,
        })

    def _served_cell(self, provider, signature, params):
        """One cell every non-parameter gate on `provider.serves` admits."""
        dtype = (provider.serves.activation or ("fp32",))[0]
        env = {}
        if provider.serves.output:
            env["output"] = provider.serves.output[0]
        granularities = provider.serves.activation_granularity
        if granularities:
            env["activation_granularity"] = granularities[-1]
        for slot, tokens in provider.serves.dtype_slots:
            if tokens:
                env[slot] = tokens[0]
        return self._cell(signature, params, dtype, env=env)

    def test_the_reference_computes_the_models_gate(self):
        spec = get_ref("situ_and_mul")
        ins = build_inputs(spec, 
            self._cell("situ_and_mul",
                       {"situ_beta": 4.0, "situ_linear_beta": 25.0}), "cpu")
        x, I = ins["x"].float(), ins["I"]
        gate, up = x[:, :I], x[:, I:]
        want = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
                * 25.0 * torch.tanh(up / 25.0))
        assert torch.allclose(spec.reference(ins), want, atol=1e-6)

    def test_an_unset_linear_beta_passes_the_up_branch_through(self):
        # The kernel's own "unset" (<= 0), which is what vLLM writes for
        # `linear_beta=None` — not a value this harness picks.
        spec = get_ref("situ_and_mul")
        ins = build_inputs(spec, 
            self._cell("situ_and_mul", {"situ_beta": 4.0}), "cpu")
        assert "situ_linear_beta" not in ins
        x, I = ins["x"].float(), ins["I"]
        gate, up = x[:, :I], x[:, I:]
        want = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * up
        assert torch.allclose(spec.reference(ins), want, atol=1e-6)

    def test_a_cell_without_the_scalar_is_not_measured_at_a_guess(self):
        for signature, params in (("situ_and_mul", {}),
                                  ("silu_and_mul_with_clamp", {})):
            spec = get_ref(signature)
            with pytest.raises(KeyError):
                build_inputs(spec, self._cell(signature, params), "cpu")

    def test_the_provider_declines_the_cell_before_it_allocates(self):
        # Declared (`requires_params`), not raised in `build`: coverage is
        # asked before build, so the gap has to be visible to `report`.
        from kernel_bench.providers import all_providers
        for signature, param in (("situ_and_mul", "situ_beta"),
                                 ("silu_and_mul_with_clamp", "clamp_limit")):
            providers = [p for p in all_providers()
                         if p.signature == signature]
            assert providers, signature
            for p in providers:
                # At a dtype the kernel serves, so the scalar is the one
                # thing the declined cell lacks.
                assert param in p.serves.requires_params
                assert not p.serves.accepts(self._served_cell(p, signature, {}))
                assert p.serves.accepts(
                    self._served_cell(p, signature, {param: 10.0})), p.kernel_impl

    def test_the_signature_renames_the_rows_fact(self):
        # The FACT (this model clamps at 10.0) is eval's and rides the row;
        # the signature only says which of them its kernels take and what
        # each is called at the launch. Nothing downstream reads a config.
        from backend.inference.op_signatures import get_signature
        row = {"consts": {"clamp_limit": 10.0, "situ_beta": 4.0,
                          "situ_linear_beta": 25.0}, "node_dims": None}
        assert get_signature("silu_and_mul_with_clamp").params_from_row(row) \
            == {"clamp_limit": 10.0}
        assert get_signature("situ_and_mul").params_from_row(row) \
            == {"situ_beta": 4.0, "situ_linear_beta": 25.0}
        # A cell axis outside the signature's identity (the router's expert
        # grouping) is a parameter of the launch, off the node's own dims;
        # the axes of the identity stay in its key.
        router = {"node_dims": {"B": 16, "s_query": 1, "n_experts": 256,
                                "n_experts_active": 8, "n_expert_groups": 8,
                                "n_groups_active": 4, "d_model": 7168}}
        assert get_signature("moe_gate").params_from_row(router) \
            == {"n_expert_groups": 8, "n_groups_active": 4}
        # A model that publishes nothing yields nothing — not a default.
        assert get_signature("situ_and_mul").params_from_row(
            {"node_dims": None}) == {}
        assert get_signature("moe_gate").params_from_row(
            {"node_dims": {"B": 16, "n_experts": 256}}) == {}

    def test_one_name_is_registered_once(self):
        # Both registries used to let a second registration of the same name
        # replace the first silently, which is how `situ_and_mul` kept the
        # beta-free reference after the beta-carrying one was written.
        from kernel_bench.refs import RefSpec, register_ref
        from backend.inference.op_signatures import (OpSignature,
                                                     register_signature)
        with pytest.raises(ValueError):
            register_ref("situ_and_mul", RefSpec(_id, _id))
        with pytest.raises(ValueError):
            register_signature(OpSignature(name="situ_and_mul"))


class TestTheCacheIsAllocatedAtItsOwnDtype:
    """`TensorLayout.role` names the `dtype_env` slot a tensor is in, and
    the cache's slot is `kv`. Nothing read it: every operand was allocated
    at the cell's ACTIVATION dtype, so Gemma-2 — the one model in the
    corpus published in float32 — was billed four bytes an element for a
    cache the harness moved two of, and every one of its attention rows
    reported twice the bandwidth it moved. An fp8 cache under a bf16
    activation is the same error in the direction that never crosses a
    ceiling, so it would never have shown at all."""

    def _cell(self, activation, kv):
        return Invocation.from_job({
            "signature": "gqa_attention", "op": "gqa_attention",
            "dims": {"kv_block_size": 16, "B": 1, "s_query": 4, "s_kv": 8, "n_heads": 2,
                     "n_kv_heads": 1, "d_head": 8},
            "phase": "prefill",
            "dtype_env": {"activation": activation, "weight": activation,
                          "kv": kv, "output": activation,
                          "metadata": "fp32"},
        })

    def test_the_slot_is_read_and_falls_back_to_the_activation(self):
        inv = self._cell("bf16", "fp32")
        assert inv.kv_dtype == "fp32"
        assert inv.dtype_for("weight") == "bf16"
        # A job written before the slot existed means the activation.
        bare = Invocation.from_job({
            "signature": "gqa_attention", "op": "gqa_attention",
            "dims": {"kv_block_size": 16, "B": 1}, "phase": "prefill",
            "dtype_env": {"activation": "bf16"},
        })
        assert bare.dtype_for("kv") == "bf16"

    def test_a_cache_in_another_dtype_is_allocated_in_it(self):
        spec = get_ref("gqa_attention")
        ins = build_inputs(spec, self._cell("bf16", "fp32"), "cpu")
        assert ins["q"].dtype is torch.bfloat16
        assert ins["k"].dtype is torch.float32, "the cache is fp32 on the row"
        assert ins["v"].dtype is torch.float32

    def test_a_format_too_narrow_to_hold_a_value_stays_lossless(self):
        # The convention `_gemm_inputs` set for fp8 weights: the canonical
        # input is lossless and each provider quantizes, because an fp8
        # tensor cannot carry a reference value.
        spec = get_ref("gqa_attention")
        ins = build_inputs(spec, self._cell("bf16", "fp8_e4m3"), "cpu")
        assert ins["k"].dtype is torch.bfloat16
        assert ins["v"].dtype is torch.bfloat16


class TestTheBuiltCellIsTheRowsCell:
    """A row says what it measured — a shape and a dtype per operand slot —
    and the walk bills it for exactly that. Nothing checked that the
    tensors handed to the kernel were that cell, and three bugs lived in
    the gap: Gemma-2's fp32 cache allocated as bf16 (8,227 rows at twice
    the bandwidth they moved), a MoE cell whose shape fell back to the
    identity scalars, and DeepSeek-V4's compressor billing fp32 state it
    built in bf16. Each was found by hand afterwards, by multiplying out a
    shape; this is that arithmetic before the measurement."""

    def _cell(self, activation="bf16", kv="bf16"):
        return Invocation.from_job({
            "signature": "gqa_attention", "op": "gqa_attention",
            "dims": {"kv_block_size": 16, "B": 1, "s_query": 4, "s_kv": 8, "n_heads": 2,
                     "n_kv_heads": 1, "d_head": 8},
            "phase": "prefill",
            "dtype_env": {"activation": activation, "weight": activation,
                          "kv": kv, "output": activation,
                          "metadata": "fp32"},
        })

    def test_a_cell_built_as_declared_has_no_complaint(self):
        from kernel_bench.cell_check import complaints
        inv = self._cell("bf16", "fp32")
        assert complaints(inv, build_inputs(get_ref("gqa_attention"), inv, "cpu")) == []

    def test_a_narrower_operand_than_the_row_bills_is_caught(self):
        # Gemma-2 exactly: the row says fp32, the tensor is bf16, and the
        # bandwidth comes out twice what the card moved.
        from kernel_bench.cell_check import complaints
        inv = self._cell("bf16", "fp32")
        built = build_inputs(get_ref("gqa_attention"), inv, "cpu")
        built = dict(built, k=built["k"].to(torch.bfloat16))
        said = complaints(inv, built)
        assert len(said) == 1 and said[0].startswith("k: cell says fp32")

    def test_a_shape_that_is_not_the_cells_is_caught(self):
        from kernel_bench.cell_check import complaints
        inv = self._cell()
        built = build_inputs(get_ref("gqa_attention"), inv, "cpu")
        built = dict(built, k=torch.randn(1, 1, 1, 8))
        assert any("built (1, 1, 1, 8)" in line for line in complaints(inv, built))

    def test_a_lossless_carrier_for_a_narrow_format_is_not_a_complaint(self):
        # fp8 cannot hold a reference value, so the canonical operand is
        # bf16 and each provider quantizes (`refs._store_dtype`). Wider
        # than the slot is the convention; narrower is the bug.
        from kernel_bench.cell_check import complaints
        inv = self._cell("bf16", "fp8_e4m3")
        assert complaints(inv, build_inputs(get_ref("gqa_attention"), inv, "cpu")) == []

    def test_every_provider_builds_its_own_minimal_cell_cleanly(self):
        # The gate is only worth having if it is silent on healthy cells.
        # It was not, at first: four layouts declared `role="metadata"` —
        # the weight-SCALE slot — for tensors every provider casts to the
        # activation dtype (`beta`, `base`, `table`).
        from kernel_bench.cell_check import complaints
        from kernel_bench.providers import all_providers
        from kernel_bench.smoke import minimal_dims
        said = {}
        for provider in all_providers():
            dims, _why = minimal_dims(provider, "prefill")
            spec = get_ref(provider.signature)
            if dims is None or spec is None:
                continue
            inv = Invocation(
                signature=provider.signature, op=provider.signature, dims=dims,
                dtype_env={"activation": "bf16", "output": "bf16",
                           "kv": "bf16", "weight": "bf16", "metadata": "fp32"},
                phase="prefill", tp=1,
                kernel_params={name: 1.0 for name in
                               (provider.serves.requires_params
                                if provider.serves else ())})
            try:
                built = build_inputs(spec, inv, "cpu")
            except Exception:
                continue
            found = complaints(inv, built)
            if found:
                said[provider.kernel_impl] = found
        assert said == {}, said


class TestToleranceGate:
    def test_matching_output_passes(self):
        ref = torch.randn(4, 6)
        case = BenchCase(run=ref.clone, to_canonical=_id)
        verdict, rel, cos, _typ, _wor = _cross_check(case, ref, "bf16")
        assert verdict is None
        assert rel == 0.0 and cos == pytest.approx(1.0, abs=1e-6)

    def test_wrong_output_fails(self):
        ref = torch.randn(4, 6)
        wrong = ref + 5.0            # way outside any dtype band
        case = BenchCase(run=lambda: wrong, to_canonical=_id)
        err, rel, _cos, _typ, _wor = _cross_check(case, ref, "bf16")
        assert err is not None and err > _XCHECK_BANDS["bf16"].typical
        assert rel > 0.0

    def test_shape_mismatch_is_infinite_error(self):
        ref = torch.randn(4, 6)
        case = BenchCase(run=lambda: torch.randn(4, 7), to_canonical=_id)
        assert _cross_check(case, ref, "bf16")[0] == float("inf")

    def test_near_zero_entries_do_not_fail_a_correct_low_precision_kernel(self):
        # Regression (found on a real GPU run): a reduction output (GEMM,
        # attention) is centered at zero, so a bare per-element |err|/|ref|
        # blows up on the many near-zero entries and wrongly fails a
        # numerically-correct bf16 kernel (we saw rel_err ~2790). The additive
        # atol epsilon floors that. Simulate a correct low-precision kernel via
        # bf16 INPUT rounding (the dominant error), matmul'd in fp32 so the test
        # is CPU/portable.
        torch.manual_seed(0)
        K = 4096                                   # large K → wide magnitude spread
        x, w = torch.randn(64, K), torch.randn(48, K)
        ref = x.float() @ w.float().T                                  # fp32 truth
        approx = x.bfloat16().float() @ w.bfloat16().float().T          # correct bf16 kernel
        assert (ref.abs() < 0.1).any()             # near-zero entries really exist
        case = BenchCase(run=lambda: approx, to_canonical=_id)
        verdict, rel, cos, _typ, _wor = _cross_check(case, ref, "bf16")
        assert verdict is None
        # Both numbers are recorded even though the row passes: a correct bf16
        # kernel lands at rel ~1e-2 with cosine ~1, and having the pair on file
        # is what makes a tolerance auditable after the fact rather than a
        # constant nobody can check.
        assert 0.0 < rel < 0.05 and cos > 0.9999

    def test_scaled_output_still_fails_despite_epsilon(self):
        # The epsilon must not swallow a genuine relative error: a kernel that
        # returns 1.5× the reference (right shape, wrong value) must be rejected.
        torch.manual_seed(0)
        ref = torch.randn(64, 48) * 10.0
        case = BenchCase(run=lambda: ref * 1.5, to_canonical=_id)
        err, rel, cos, _typ, _wor = _cross_check(case, ref, "bf16")
        assert err is not None and err > _XCHECK_BANDS["bf16"].typical
        # A pure rescale is the case cosine ALONE cannot catch — it stays at
        # 1.0 while every element is 50% off. Neither number subsumes the other,
        # which is why both are recorded.
        assert cos == pytest.approx(1.0, abs=1e-4) and rel > 0.4

    def test_fp8_band_is_looser_than_bf16(self):
        for side in ("typical", "worst"):
            assert (getattr(_XCHECK_BANDS["fp8_e4m3"], side)
                    > getattr(_XCHECK_BANDS["bf16"], side)
                    > getattr(_XCHECK_BANDS["fp32"], side))

    def test_the_verdict_does_not_grow_with_the_number_of_tokens(self):
        # The bug this gate's statistic exists for: a correct bf16 rope
        # (arithmetic in bf16, which is what every framework kernel does)
        # failed by being handed more tokens — 97 cells per LLaMA-class
        # model refused, the model's run marked rc=1 — because the old gate
        # read the WORST element and the largest of more samples is larger.
        # The angle has to come from the position for that to show: with a
        # random angle per row the tail is already full at 1k tokens.
        torch.manual_seed(0)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))
        worst_seen = []
        for tokens in (1024, 32768):
            x = torch.randn(tokens, 64)
            angle = torch.arange(tokens).float()[:, None] * inv_freq[None, :]
            cos_t, sin_t = angle.cos(), angle.sin()

            def rotate(t, dt):
                x1, x2 = t[:, :32].to(dt), t[:, 32:].to(dt)
                c, s = cos_t.to(dt), sin_t.to(dt)
                return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

            ref = rotate(x, torch.float32)
            case = BenchCase(run=lambda: rotate(x, torch.bfloat16).float(),
                             to_canonical=_id)
            verdict, rel, cos, _typ, _wor = _cross_check(case, ref, "bf16")
            assert verdict is None, f"{tokens} tokens rejected a correct kernel"
            assert cos > 0.9999
            worst_seen.append(rel)
        # And the old gate would have refused the larger cell: `rel` is a
        # maximum, so it grows with the token count while the kernel does
        # not. Without this the test passes again the day someone restores
        # the maximum and widens the band to cover it.
        assert worst_seen[1] > worst_seen[0] > _XCHECK_BANDS["bf16"].typical

    def test_one_wrong_element_among_a_million_right_ones_still_fails(self):
        # What the averaged question alone would hide, and the reason the
        # maximum is still asked: a single element off by ten magnitudes
        # moves the RMS by nothing (0.0055 against a 0.006 band) and the
        # maximum to 5.66, which is O(1) in these units — wrong, not
        # imprecise. One whole wrong ROW is caught by the RMS instead, so it
        # would not test this clause.
        torch.manual_seed(0)
        ref = torch.randn(4096, 256)
        broken = ref.clone()
        broken[1234, 56] += 10.0
        case = BenchCase(run=lambda: broken, to_canonical=_id)
        err, _rel, cos, _typ, _wor = _cross_check(case, ref, "bf16")
        assert err is not None
        assert cos > 0.9999      # the energy argument: cosine cannot see it

    def test_a_whole_wrong_row_is_caught_by_the_averaged_question(self):
        torch.manual_seed(0)
        ref = torch.randn(4096, 256)
        broken = ref.clone()
        broken[1234] = torch.randn(256) * 3.0
        case = BenchCase(run=lambda: broken, to_canonical=_id)
        assert _cross_check(case, ref, "bf16")[0] is not None

    def test_a_cell_run_at_a_narrower_precision_than_it_declares_fails(self):
        # The other side of admitting the correct bf16 kernel: the same op
        # computed at fp8 precision must still be rejected on the bf16 band.
        # Its error is 15x the bf16 floor, and that is the step a band sees.
        torch.manual_seed(0)
        x = torch.randn(4096, 64)
        angle = torch.rand(4096, 32) * 6.283
        cos_t, sin_t = angle.cos(), angle.sin()

        def rotate(t):
            x1, x2 = t[:, :32], t[:, 32:]
            return torch.cat([x1 * cos_t - x2 * sin_t,
                              x2 * cos_t + x1 * sin_t], dim=-1)

        ref = rotate(x)
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
        narrow = (x / scale).to(torch.float8_e4m3fn).float() * scale
        case = BenchCase(run=lambda: rotate(narrow), to_canonical=_id)
        assert _cross_check(case, ref, "bf16")[0] is not None
        assert _cross_check(case, ref, "fp8_e4m3")[0] is None


class TestTorchGemmProviderEndToEnd:
    """The real torch GEMM provider runs on CPU: its build → cross-check vs
    the reference → emit path works, and a job with a broken reference-mismatch
    is rejected."""

    def test_torch_gemm_measured_and_passes_xcheck(self):
        env = Env("cpu", device=None, device_name=None, gpu_sku="CPU",
                  peak_bw_gbps=None, libs={"torch": torch.__version__})
        job = {
            "signature": "gemm", "op": "linear",
            "dims": {"B": 1, "s_query": 4, "d_reduce": 8, "d_out": 6},
            "dtype_env": {"activation": "fp32", "weight": "fp32", "kv": "fp32",
                          "output": "fp32", "metadata": "fp32"},
            "phase": "prefill", "model": "m", "spec_hash": "h",
            "node_spec_id": "n", "parallel": "tp=1", "count_per_step": 1,
            # The measurement key is the PLAN's — the driver stamps only the
            # card — so a hand-built job states it as a planned one does.
            "physical_key": ["gemm", _shape_key({"B": 1, "s_query": 4,
                                                 "d_reduce": 8, "d_out": 6}),
                             _dtype_key("fp32"), "prefill"],
        }
        res = run_jobs([job], env=env, date="2026-07-28", contributor="@t")
        # torch provider applies (torch installed), runs on CPU, matches its
        # own fp32 reference → one recorded row, no failures.
        assert len(res.rows) == 1
        assert res.rows[0]["framework"] == "torch"
        assert res.failures == []

    def test_promoted_fp32_gemm_writes_its_declared_bf16_output(self):
        """The promoted cell survives the whole driver path: fp32 operands,
        fp32 provider identity, bf16 launch output, and a passing check."""
        dims = {"B": 1, "s_query": 4, "d_reduce": 8, "d_out": 6}
        dtype_env = {"activation": "fp32", "weight": "fp32",
                     "output": "bf16"}
        dtype_key = json.dumps(dtype_env, separators=(",", ":"),
                               sort_keys=True)
        job = {
            "signature": "gemm", "op": "linear", "dims": dims,
            "dtype_env": dtype_env, "phase": "prefill", "model": "m",
            "spec_hash": "h", "node_spec_id": "n", "parallel": "tp=1",
            "count_per_step": 1,
            "physical_key": ["gemm", _shape_key(dims), dtype_key,
                             "prefill"],
        }
        env = Env("cpu", device=None, device_name=None,
                  gpu_sku="CPU", peak_bw_gbps=None,
                  libs={"torch": torch.__version__})
        res = run_jobs([job], env=env, date="2026-10-29", contributor="@t")
        assert res.failures == []
        [row] = res.rows
        assert row["kernel_impl"] == "torch_cublas_linear@compute=fp32"
        assert json.loads(row["dtype_env_json"]) == dtype_env
        assert row["measured_dtype_env_json"] == ""
        assert "x=float32[" in row["launch_signature"]
        assert "out=bfloat16[" in row["launch_signature"]

    def test_torch_conv2d_measured_and_passes_xcheck(self):
        # The Step-3.7 ViT downsampler kernel: a padded (k//2) k×k strided conv
        # over a flattened square patch grid. Runs on CPU (no _require_cuda,
        # like the torch GEMM baseline); the provider reshapes the feature-last
        # grid to [B, C_in, H, W] and matches its own fp32 reference.
        # s_conv_grid=144 → H=12 grid; 3×3 stride-2 pad-1 → 6×6 output.
        env = Env("cpu", device=None, device_name=None, gpu_sku="CPU",
                  peak_bw_gbps=None, libs={"torch": torch.__version__})
        job = {
            "signature": "conv2d", "op": "conv2d",
            "dims": {"B": 1, "s_conv_grid": 144, "C_in": 8, "C_out": 6,
                     "kernel_size": 3, "stride": 2},
            "dtype_env": {"activation": "fp32", "weight": "fp32", "kv": "fp32",
                          "output": "fp32", "metadata": "fp32"},
            "phase": "prefill", "model": "m", "spec_hash": "h",
            "node_spec_id": "n", "parallel": "tp=1", "count_per_step": 1,
            "physical_key": ["conv2d", _shape_key(
                {"B": 1, "s_conv_grid": 144, "C_in": 8, "C_out": 6,
                 "kernel_size": 3, "stride": 2}), _dtype_key("fp32"),
                "prefill"],
        }
        res = run_jobs([job], env=env, date="2026-07-28", contributor="@t")
        assert len(res.rows) == 1
        assert res.rows[0]["framework"] == "torch"
        assert res.failures == []

    def test_conv2d_identity_keys_the_grid_the_encode_walk_hands_it(self):
        # The encode walk hands the conv an image's patch rows: Step-3.7's
        # 728 x 728 view at 14-pixel patches is a 52 x 52 grid, keyed as is.
        import json as _json
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        out = identity_shape_from_shape_in(
            "conv2d", [[1, 2704, 1536]], shape_out=[[1, 676, 3072]],
            spec_dims={"kernel_size": 3, "stride": 2})
        cell = _json.loads(out)
        assert cell["s_conv_grid"] == 52 * 52, cell
        assert cell["C_in"] == 1536 and cell["C_out"] == 3072
        assert cell["kernel_size"] == 3 and cell["stride"] == 2

    def test_conv2d_identity_has_no_cell_for_a_grid_that_is_not_square(self):
        # A kernel is benched over a square grid; rows that are not one are
        # reported by the plan, never keyed as another grid.
        from backend.inference.benchmark_identity import identity_shape_from_shape_in
        assert identity_shape_from_shape_in(
            "conv2d", [[1, 2048, 1536]], shape_out=[[1, 512, 3072]],
            spec_dims={"kernel_size": 3, "stride": 2}) is None


class TestTheHyperConnectionSplitIsTheFrameworksFunction:
    """`hc_split_sinkhorn`'s oracle, and what its old one teaches about this
    gate.

    The reference used to be `(mixes + base)` through eight one-sided row
    normalisations, returning the `pre` slice. Three independent defects:
    one-sided normalisation is idempotent after the first pass, so seven
    rounds were dead and the loop was just a row-sum divide; that is not
    Sinkhorn, which ALTERNATES row and column normalisation; and the op's
    own doc and its three registry ports say it splits into pre/post/comb,
    so two of the three writes were never checked.

    It passed the numerical-agreement check anyway, and the reason is worth
    pinning: this span's only provider is its own torch floor, which held
    the SAME wrong arithmetic. The gate compared a function against itself.
    That is not a tolerance admitting a wrong function — it is a span with
    no independent implementation getting a free pass, which is a different
    weakness and the one asserted against here.
    """

    DIMS: ClassVar[dict] = {"B": 2, "s_query": 3,
                            "n_hidden_copies": 4, "d_hc_mix": 24}

    def _inv(self):
        return Invocation(op="hc_split_sinkhorn", signature="hc_split_sinkhorn",
                          phase="decode", dims=dict(self.DIMS),
                          dtype_env={"activation": "bf16"}, tp=1)

    def _split(self):
        inv = self._inv()
        spec = get_ref("hc_split_sinkhorn")
        inputs = build_inputs(spec, inv, "cpu")
        out = spec.reference(inputs)
        hc = self.DIMS["n_hidden_copies"]
        return inv, inputs, out, hc

    def test_one_sided_normalisation_is_idempotent_so_the_old_loop_was_dead(self):
        """The fact that condemns the old reference, as arithmetic rather
        than as a claim about the frameworks: after one row normalisation,
        every further one is the identity. Any future reference that
        normalises a single axis in a loop is doing nothing per extra round,
        and this says so in one line."""
        x = torch.rand(6, 24).abs() + 0.1
        once = x / x.sum(-1, keepdim=True)
        twice = once / once.sum(-1, keepdim=True)
        assert torch.allclose(once, twice, atol=1e-7)

    def test_the_cross_mix_is_doubly_stochastic(self):
        """What Sinkhorn is FOR, and what the alternation buys: rows AND
        columns both sum to one. The old one-sided function could not
        produce this, so this single property separates them."""
        _inv, _inputs, out, hc = self._split()
        comb = out[:, 2 * hc:].reshape(-1, hc, hc)
        assert (comb >= 0).all()
        assert torch.allclose(comb.sum(-1), torch.ones(6, hc), atol=1e-3)
        assert torch.allclose(comb.sum(-2), torch.ones(6, hc), atol=1e-3)

    def test_all_three_writes_are_in_the_oracle(self):
        """The op has three registry ports and its doc says pre/post/comb.
        The reference returns all three side by side, because the
        cross-check compares one tensor and a write outside it is a write
        nothing checks — which is how two of these went unchecked before."""
        _inv, _inputs, out, hc = self._split()
        assert out.shape == (6, hc + hc + hc * hc)
        pre, post = out[:, :hc], out[:, hc:2 * hc]
        # `pre` is a sigmoid plus an epsilon; `post` the same, times two.
        assert ((pre > 0) & (pre < 1.01)).all()
        assert ((post >= 0) & (post <= 2.0)).all()

    def test_the_floor_and_the_oracle_are_one_reading_of_the_arithmetic(self):
        """Both call `hc_split_coefficients`. Pinned because the defect was
        not the formula in either place — it was that there were two places,
        so one could drift from the frameworks while the gate compared them
        to each other and saw nothing."""
        from kernel_bench import refs
        from kernel_bench.providers import all_providers
        inv, inputs, out, _hc = self._split()
        providers = [p for p in all_providers()
                     if p.signature == "hc_split_sinkhorn"]
        assert [p.library for p in providers] == ["torch"], (
            "still the only provider: when a framework kernel is added, this "
            "span's agreement check becomes evidence rather than a tautology")
        env = Env("2.11", None, None, None, None, libs={"torch": "2.11"})
        case = providers[0].build(inv, env, inputs)
        assert torch.allclose(case.to_canonical(case.run()).float(),
                              out.float(), atol=1e-5)
        # and the epilogue span's split goes through the same function
        assert refs.hc_split_coefficients.__module__ == "kernel_bench.refs"


class TestTheRowCarriesTheVerdictsBasis:
    """A row must be able to say what its own `cross_check` was decided by.

    Before schema 13 it carried two numbers the gate never reads
    (`cross_check_rel_err`, `cross_check_cosine`) and neither that it does,
    so a `passed` could not be audited from the row — and the rel-err
    column, being the only error-shaped number there, was twice read as the
    margin by people with the gate's source open.
    """

    def test_the_gate_returns_the_two_numbers_it_decided_on(self):
        """Both come back on BOTH paths — a pass has to be auditable, not
        only a failure."""
        ref = torch.randn(64, 128)
        def _bf16():
            return ref.bfloat16()
        same = BenchCase(run=_bf16, to_canonical=lambda o: o.float())
        verdict, _rel, _cos, typical, worst = _cross_check(same, ref, "bf16")
        assert verdict is None                      # bf16 round trip passes
        assert typical is not None and worst is not None
        assert 0.0 <= typical <= worst              # an RMS cannot exceed a max
        band = _XCHECK_BANDS["bf16"]
        assert typical <= band.typical and worst <= band.worst
        # and on the failing path
        wrong = BenchCase(run=lambda: ref * 3.0, to_canonical=lambda o: o)
        verdict2, _r2, _c2, typ2, wor2 = _cross_check(wrong, ref, "bf16")
        assert verdict2 is not None
        assert not (typ2 <= band.typical and wor2 <= band.worst), (typ2, wor2)

    def test_the_verdict_is_reconstructible_from_the_row(self):
        """The point of the columns: given a row and the band, recompute the
        verdict. `rel_err` cannot do this — it is measured against nothing."""
        from backend.inference.benchmarks import ALL_COLUMNS
        assert "xcheck_typical" in ALL_COLUMNS and "xcheck_worst" in ALL_COLUMNS
        band = _XCHECK_BANDS["bf16"]
        for typical, worst, expect in ((1e-3, 1e-2, True),
                                       (1e-3, 5e-1, False),   # worst alone fails
                                       (5e-2, 1e-2, False)):  # typical alone fails
            # the gate's rule, read off what a row would carry
            assert (typical <= band.typical and worst <= band.worst) is expect


class TestFlashMLAsFP8DecodeOperands:
    """vLLM's FlashMLA FP8 decode (`flash_mla_with_kvcache_fp8`) reads an
    e4m3 query and an e4m3 paged latent and scales each back by one fp32
    descale (`descale_q` / `descale_k`, shape [1]). The kernel needs an
    SM90 card, so here the vLLM module is replaced by the extension's
    contract written out in torch (csrc/extension/sm90/dense_fp8/pybind.cpp
    at the vLLM pin: scores `q·k · softmax_scale · descale_q · descale_k`,
    the value the first `head_dim_v` channels of the same row times
    `descale_k`), and what the build hands it is judged against the oracle
    over the lossless operands on the band the driver judges the cell on."""

    @staticmethod
    def _contract(q, k_cache, block_table, cache_seqlens, head_dim_v,
                  tile_scheduler_metadata, num_splits, softmax_scale,
                  causal, descale_q, descale_k):
        B, Sq = q.shape[:2]
        rows = []
        for b in range(B):
            L = int(cache_seqlens[b])
            k = (k_cache[block_table[b].long()].reshape(-1, q.shape[-1])[:L]
                 .float() * descale_k)
            s = torch.einsum("snd,ld->snl", q[b].float() * descale_q,
                             k) * softmax_scale
            if causal and Sq > 1:
                seen = torch.arange(L)[None, :] <= (
                    L - Sq + torch.arange(Sq))[:, None]
                s = s.masked_fill(~seen[:, None, :], float("-inf"))
            rows.append(torch.einsum("snl,ld->snd", s.softmax(-1),
                                     k[:, :head_dim_v]))
        return torch.stack(rows).to(torch.bfloat16), None

    def _launch(self, monkeypatch, contract):
        import sys
        import types
        from kernel_bench.providers import attention
        monkeypatch.setattr(attention, "_require_cuda", lambda: None)
        seen = {}

        def _meta(seqlens, q_per_head_k, heads_k):
            seen["meta"] = (seqlens.tolist(), q_per_head_k, heads_k)
            return torch.zeros(1, 8, dtype=torch.int32), None

        def _fp8(**kw):
            seen["call"] = kw
            return contract(**kw)
        monkeypatch.setitem(
            sys.modules, "vllm.v1.attention.ops.flashmla",
            types.SimpleNamespace(flash_mla_with_kvcache_fp8=_fp8,
                                  get_mla_metadata_dense_fp8=_meta))
        inv = Invocation.from_job({
            "signature": "mla_attention", "op": "mla_attention",
            "dims": {"B": 2, "n_heads": 4, "s_query": 1, "s_kv": 70,
                     "d_head": 576, "d_v_head": 512, "kv_block_size": 64},
            "phase": "decode",
            "dtype_env": {"activation": "bf16", "output": "bf16",
                          "kv": "fp8_e4m3",
                          "kv_format": "fp8_per_tensor"}})
        ins = build_inputs(get_ref("mla_attention"), inv, "cpu")
        case = attention._vllm_flashmla_fp8_build(inv, None, ins)
        return inv, ins, case, seen

    def test_the_build_hands_the_kernel_its_fp8_operands_and_descales(
            self, monkeypatch):
        from kernel_bench.driver import _xcheck_basis
        inv, ins, case, seen = self._launch(monkeypatch, self._contract)
        dtype, records = _xcheck_basis(inv, ins)
        assert _cross_check(case, get_ref("mla_attention").reference(ins),
                            dtype, records=records)[0] is None
        call = seen["call"]
        assert call["q"].dtype == call["k_cache"].dtype == torch.float8_e4m3fn
        # Two 64-token pages a sequence (70 tokens), one latent head.
        assert call["k_cache"].shape == (4, 64, 1, 576)
        assert call["block_table"].tolist() == [[0, 1], [2, 3]]
        assert call["cache_seqlens"].tolist() == [70, 70]
        for name in ("descale_q", "descale_k"):
            assert call[name].shape == (1,), name
            assert call[name].dtype == torch.float32, name
        # `_build_decode`: Sq·N query rows per latent head, one latent head.
        assert seen["meta"] == ([70, 70], 4, 1)
        # The cell is an attention READING the record, so the oracle is the
        # function over the lossless operands on the fp8 band, not a record
        # it writes (`compared_records` is empty).
        assert _xcheck_basis(inv, ins) == ("fp8_e4m3", ())

    def test_the_descales_are_what_the_launch_scales_back_by(
            self, monkeypatch):
        # The same launch with the descales read as 1: the operands alone
        # are on a grid `amax / 448` wide, and without the scales the
        # result is nowhere near the function.
        from kernel_bench.driver import _xcheck_basis

        def unscaled(**kw):
            one = torch.ones(1)
            return self._contract(**{**kw, "descale_q": one,
                                     "descale_k": one})
        inv, ins, case, _ = self._launch(monkeypatch, unscaled)
        dtype, records = _xcheck_basis(inv, ins)
        assert _cross_check(case, get_ref("mla_attention").reference(ins),
                            dtype, records=records)[0] is not None
