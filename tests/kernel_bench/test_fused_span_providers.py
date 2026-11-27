"""Measurement routes for fused spans the cost model prices.

Each fusion plan (`backend/inference/fusion_plans/*.yaml`) names a span that
one framework launch replaces. A span is measurable when a signature binds
its fused op, a reference computes its function and providers call the
framework's own kernel. These pin that for the spans wired up here: the plan
reaches the span with the cell the signature declares, the providers claim
the span (`fused_op`), and the CPU-runnable pieces (reference, torch floor,
the providers' input shaping) compute what the plan's naive pattern does.
GPU kernels cannot run here; their builds stop at `_require_cuda`.
"""
from __future__ import annotations

import json
from typing import ClassVar

import pytest

from kernel_bench.env import Env
from kernel_bench.invocation import Invocation
from kernel_bench.providers import TORCH_FLOAT, all_providers
from tests.kernel_bench._inputs import build_inputs

TORCH_ENV = Env("2.11", None, None, None, None, libs={"torch": "2.11"})


def _providers(signature):
    return [p for p in all_providers() if p.signature == signature]


def _plan(model_id, phase, framework):
    """The plan a sweep builds on an H100: the binding answered for the card,
    so `device:`-conditioned fusions are the ones the estimate prices."""
    from backend.inference.framework_bindings import (ServingDeployment,
                                                      get_binding)
    from kernel_bench.plan import build_jobs
    binding = get_binding(framework).deployed(ServingDeployment.at(
        gpu_sku="H100-SXM", activation_dtype="BF16"))
    jobs, dropped = build_jobs(avg_output_tokens=128, model_id=model_id,
                               batch_size=2, seq_len=1024, phase=phase,
                               framework=framework, binding=binding,
                               gpu="H100-SXM")
    return jobs, dropped


def _ports(*names):
    """A fused launch's operand ports as a row names them (`operand_ports`):
    one port per operand, or a tuple — in plan order — where one tensor
    fills several."""
    return [list(name) if isinstance(name, tuple) else [name]
            for name in names]


def _inv(signature, dims, phase="prefill"):
    return Invocation(op=signature, signature=signature, phase=phase,
                      dims=dict(dims), dtype_env={"activation": "bf16"}, tp=1)


def _floor_meets_the_oracle(sig, dims, output="bf16", env=None,
                            output_format=None):
    """A span's torch floor at a bf16 cell: inside the bf16 band the driver
    checks it against the fp32 oracle with, and a launch the driver would record — its widest operand at the
    cell's precision, not the oracle's fp32 (`launch_trace.disagreement`)."""
    torch = pytest.importorskip("torch")
    from backend.inference.op_signatures import get_signature
    from kernel_bench import launch_trace
    from kernel_bench.refs import get_ref
    spec = get_ref(sig)
    env = env or {"activation": "bf16", "weight": "bf16", "output": output}
    # The record the cell's output is written in: a format name, not a
    # dtype the launch's tensors are allocated at.
    record = {} if output_format is None else {"output_format": output_format}
    inv = Invocation(op=sig, signature=sig, phase="prefill", tp=1,
                     dims={a: dims[a]
                           for a in get_signature(sig).cell_dims_for(dims)},
                     dtype_env={**env, **record})
    inputs = build_inputs(spec, inv, "cpu")
    want = spec.reference(inputs).float()
    floor = next(p for p in _providers(sig) if p.library == "torch")
    case = floor.build(inv, TORCH_ENV, inputs)
    # The call alone, as `driver._cross_check` records it: canonicalising
    # the result is the check's own arithmetic, not the launch.
    recording = launch_trace.record()
    got = case.to_canonical(recording.run(case.run)).float()
    # The cross-check's own reading (`driver._cross_check`) on the band the
    # driver judges the cell on (`_xcheck_basis`) — a record cell's worst
    # element at one step of its grid.
    from kernel_bench.driver import _xcheck_band, _xcheck_basis
    band = _xcheck_band(*_xcheck_basis(inv, inputs))
    error = (got - want).abs() / (want.abs() + want.abs().mean())
    assert error.pow(2).mean().sqrt() <= band.typical, sig
    assert error.max() <= band.worst, sig
    elements = [v.numel() for v in inputs.values() if torch.is_tensor(v)]
    assert launch_trace.disagreement(
        recording.trace, declared_dtypes=sorted(set(env.values())),
        declared_elements=elements,
        records=_xcheck_basis(inv, inputs)[1]) is None, sig
    return recording.trace


class TestEveryPlanHasARouteToAMeasurement:
    """The join between the two registries a fused launch is declared in:
    the plan (`fusion_plans/*.yaml`, what the framework fuses) and its
    measurement route (a signature's `binds_ops`, or a provider's
    `fused_op`). Static and second-scale, so it runs in the default set: it
    used to sit in `test_kernel_bench.py`, whose module-wide `regression`
    mark kept it out of every push and deploy, and two plans landed with no
    route while it was red and unseen."""

    def test_a_span_with_no_route_to_a_measurement_is_named(self):
        """A fusion span the cost model prices and nothing measures.

        The span's saving is then a number nothing can check. Most of these
        are not unmeasurable — the framework ships the kernel, so what is
        missing is ours: a signature, a reference and a provider. The list
        shrinks as each is wired up, and a span JOINING it is a boundary
        that quietly went unchecked.
        """
        import glob
        import os
        from pathlib import Path
        import yaml
        from backend.inference.op_signatures import op_bindings
        OP_TO_SIGNATURE = op_bindings()
        from backend.inference import fusion_plans as FP
        from kernel_bench.providers import all_providers
        spans = {(yaml.safe_load(Path(p).read_text()) or {})["fused_op"]
                 for p in glob.glob(os.path.join(os.path.dirname(FP.__file__),
                                                 "*.yaml"))}
        claimed = {p.fused_op for p in all_providers() if p.fused_op}
        unreachable = sorted(spans - set(OP_TO_SIGNATURE) - claimed)
        # `gated_act_ffn` is genuinely unmeasurable: neither framework ships
        # a mega-FFN kernel. The six that joined with the Kimi-K3, Hy4 and
        # DeepInfer plans are ours to wire — the frameworks ship each launch
        # (the plans' own anchors say where), so what is missing is a
        # signature, a reference and a provider. The list names what remains
        # so the gap stays countable rather than reading as coverage.
        assert unreachable == [
            # Kimi-K3's AttnRes readout + sublayer norm, one launch.
            "add_attn_res_combine_norm",
            # K3's packed q|k|v short convolutions, one causal-conv launch.
            "causal_conv1d_packed",
            # No kernel in either framework.
            "gated_act_ffn",
            # Hy4's pre-gate reduction and first exchange mix.
            "ihc_pre_gate_reduce", "ihc_pre_mix",
            # K3 KDA's six input projections, one merged GEMM.
            "merged_column_parallel_linear_wide",
            # Absorbed-MLA decode's two latent concats, one DeepInfer launch.
            "mla_q_concat_kv_insert",
        ], f"the set of spans with no route to a measurement moved: {unreachable}"


class TestCausalConv1d:
    """The short conv with its SiLU (`fusion_plans/causal_conv1d.yaml`)."""

    def test_the_span_binds_and_the_bare_conv_does_not(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert bound["causal_conv1d"] == "causal_conv1d"
        # A bare conv is another function (no activation); the reference
        # applies the SiLU the span folds in.
        assert "conv1d" not in bound
        ps = _providers("causal_conv1d")
        assert ps and all(p.fused_op == "causal_conv1d" for p in ps)

    @pytest.mark.regression          # walks Qwen3.8-Flash-Next four times
    def test_a_dilated_conv_is_its_own_cell_and_only_torch_serves_it(self):
        # vLLM fuses Qwen PLE's dilated conv (dilation = n-gram size, 3) with
        # its SiLU in `ple_conv`; the gated-delta-net conv of the same model
        # has the same width and channel count. Without `dilation` in the
        # cell the two would be one measurement.
        cells = {}
        for fw in ("vllm", "sglang"):
            for phase in ("prefill", "decode"):
                jobs, dropped = _plan("Qwen/Qwen3.8-Flash-Next", phase, fw)
                conv = [j for j in jobs if j["signature"] == "causal_conv1d"]
                assert conv, (fw, phase)
                assert all(j["op"] == "causal_conv1d" for j in conv)
                cells[fw, phase] = {j["dims"]["dilation"] for j in conv}
                if fw == "sglang":
                    # sglang runs the PLE conv as F.conv1d then F.silu: the
                    # bare conv is named, not measured as the fused kernel.
                    assert "conv1d" in dropped
        assert cells["vllm", "prefill"] == cells["vllm", "decode"] == {1, 3}
        assert cells["sglang", "prefill"] == cells["sglang", "decode"] == {1}

        dims = {"B": 2, "s_query": 64, "d_conv_channels": 256,
                "s_conv_kernel": 4, "dilation": 3}
        inv = _inv("causal_conv1d", dims)
        serving = {p.kernel_impl for p in _providers("causal_conv1d")
                   if p.accepts(inv, TORCH_ENV)}
        assert serving == {"torch_depthwise_conv1d"}
        undilated = _inv("causal_conv1d", {**dims, "dilation": 1})
        assert {p.library for p in _providers("causal_conv1d")
                if p.accepts(undilated, TORCH_ENV)} >= {"vllm", "sglang", "torch"}

    def test_the_torch_floor_computes_the_dilated_reference(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("causal_conv1d")
        for dilation in (1, 3):
            inv = _inv("causal_conv1d", {
                "B": 2, "s_query": 9, "d_conv_channels": 8,
                "s_conv_kernel": 4, "dilation": dilation})
            inputs = build_inputs(spec, inv, "cpu")
            assert inputs["dilation"] == dilation
            floor = next(p for p in _providers("causal_conv1d")
                         if p.library == "torch")
            got = floor.build(inv, TORCH_ENV, inputs).run().float()
            want = spec.reference(inputs)
            assert torch.allclose(got, want, atol=3e-2), dilation
        # Dilation changes the function: the same inputs at 1 and 3 differ.
        x = build_inputs(spec, _inv("causal_conv1d", {
            "B": 1, "s_query": 9, "d_conv_channels": 4, "s_conv_kernel": 4,
            "dilation": 1}), "cpu")
        assert not torch.allclose(spec.reference(x),
                                  spec.reference({**x, "dilation": 3}))


    def test_the_dilated_decode_step_convolves_every_token_of_the_cell(self):
        """sglang's dilated short-conv step runs ONE conv over
        `cat(state, tokens)` — one token on the plain decode path,
        `draft_token_num` of them on a verify step (qwen4_exp.py:1151-1165).
        Timing only the last token left every speculative cell's output a
        different shape from the cell's, and its cross-check failed on that
        (294 H100 cells, Qwen3.8-Flash-Next). The multi-token cell comes
        first: it is the one that was red."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.linear_ssm as linear_ssm
        from kernel_bench.refs import get_ref
        spec = get_ref("causal_conv1d")
        build = linear_ssm._sglang_dilated_conv1d_update_build
        for s_query in (2, 1):
            inv = _inv("causal_conv1d", {
                "B": 2, "s_query": s_query, "d_conv_channels": 8,
                "s_conv_kernel": 4, "dilation": 3}, phase="decode")
            inputs = build_inputs(spec, inv, "cpu")
            case = build(inv, TORCH_ENV, inputs)
            got = case.to_canonical(case.run())
            want = spec.reference(inputs)
            assert got.shape == want.shape, s_query
            assert torch.allclose(got, want, atol=3e-2), s_query
            if s_query > 1:
                # The step advanced the window by its own tokens, so a
                # second call's later positions read the first call's
                # tokens through the dilated taps — no longer the
                # zero-state conv. (At one token a 3-dilated window of 9
                # shifts the taps onto zeros either way, so the control
                # only discriminates on the multi-token step.)
                assert not torch.allclose(case.to_canonical(case.run()),
                                          want, atol=3e-2), s_query


class TestRmsnormGated:
    """The linear-attention output norm gated by act(z)
    (`fusion_plans/rmsnorm_gated.yaml`)."""

    def test_every_gated_norm_kernel_claims_the_span(self):
        from backend.inference.op_signatures import op_bindings
        assert op_bindings()["rmsnorm_gated"] == "rmsnorm_gated"
        ps = _providers("rmsnorm_gated")
        assert {p.kernel_impl for p in ps} == {
            "vllm_rmsnorm_gated", "vllm_fused_rmsnorm_gated",
            "sgl_rmsnorm_gated", "sgl_fused_rmsnorm_gated",
            "torch_rmsnorm_gated"}
        assert all(p.fused_op == "rmsnorm_gated" for p in ps)

    def test_the_floor_is_the_plans_three_nodes(self):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        from kernel_bench.refs import RMSNORM_GATED_ACTIVATION, get_ref
        spec = get_ref("rmsnorm_gated")
        inv = _inv("rmsnorm_gated", {"B": 2, "s_query": 6, "d_model": 64})
        inputs = build_inputs(spec, inv, "cpu")
        assert inputs["activation"] == RMSNORM_GATED_ACTIVATION
        x, z, w = (inputs[k].float() for k in ("x", "z", "w"))
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        assert torch.allclose(spec.reference(inputs), norm * F.silu(z),
                              atol=1e-5)
        floor = next(p for p in _providers("rmsnorm_gated")
                     if p.library == "torch")
        for act in ("swish", "sigmoid"):
            case = {**inputs, "activation": act}
            got = floor.build(inv, TORCH_ENV, case).run().float()
            assert torch.allclose(got, spec.reference(case), atol=5e-2), act

    @pytest.mark.regression          # walks two models in both phases
    def test_per_head_norms_fold_into_the_rows(self):
        # Gated-DeltaNet norms each value head ([B, n_heads, s_query, d])
        # and Ling's GLA ends in a group RMS (f883d987f: 8192 = 8 groups x
        # 1024): one RMS per (head-or-group, token) row — the per-token
        # cell, the fold into the rows the plain per-head rmsnorm is keyed
        # by.
        for model, width, rows in (("Qwen/Qwen3.5-35B-A3B", 128, 32),
                                   ("inclusionAI/Ling-2.6-1T", 1024, 8)):
            for fw in ("vllm", "sglang"):
                jobs, _ = _plan(model, "decode", fw)
                cells = {tuple(sorted(j["dims"].items())) for j in jobs
                         if j["signature"] == "rmsnorm_gated"}
                assert cells == {(("B", 2), ("d_model", width),
                                  ("s_query", rows))}, (model, fw, cells)


class TestQKvRmsnormQuant:
    """The DeepSeek-V4.1 a-norm pair writing q as MXFP8
    (`fusion_plans/fused_q_kv_rmsnorm_quant.yaml`)."""

    DIMS: ClassVar[dict] = {"B": 2, "s_query": 5, "d_q_lora": 64,
                            "d_kv_lora": 32}

    def _cell(self, **overrides):
        inv = _inv("fused_q_kv_rmsnorm_quant", dict(self.DIMS))
        inv.dtype_env.update({"weight": "bf16", "output": "mxfp8",
                              "activation_granularity": "per-group:32"})
        inv.dtype_env.update(overrides)
        return inv

    def test_the_span_binds_and_its_launches_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        assert op_bindings()["fused_q_kv_rmsnorm_quant"] == \
            "fused_q_kv_rmsnorm_quant"
        ps = _providers("fused_q_kv_rmsnorm_quant")
        assert {p.library for p in ps} == {"vllm", "torch"}
        assert all(p.fused_op == "fused_q_kv_rmsnorm_quant" for p in ps)
        vllm = next(p for p in ps if p.library == "vllm")
        assert vllm.serves.output == ("mxfp8",)
        assert vllm.serves.activation_granularity == ("per-group:32",)
        assert vllm.min_sm == 100
        # The launch's own width and output-format facts are declarations, not
        # wrapper-side refusals.
        assert not [p for p in ps
                    if p.accepts(self._cell(output="fp8_e4m3"), TORCH_ENV)]
        odd = self._cell()
        odd.dims.update(d_q_lora=48)
        assert not [p for p in ps if p.accepts(odd, TORCH_ENV)]

    def test_the_kv_write_is_a_published_secondary_output(self):
        """The launch returns the normalized KV rows BESIDE the MXFP8 q
        record (`kvo`, query_quant.py:105, :132), in the activation dtype —
        so the providers split by activation and each states the write as a
        secondary output. The pin goes through `capabilities()`, the record
        the API publishes: a field the publication path dropped would leave
        an equality-on-`serves` assertion green."""
        for p in _providers("fused_q_kv_rmsnorm_quant"):
            dt = p.serves.activation[0]
            assert p.serves.activation == (dt,)       # split by activation
            assert p.serves.secondary_outputs == (("kv", dt),), p.kernel_impl
            record = p.capabilities()
            assert record["secondary_outputs"] == [
                {"tensor": "kv", "dtype": dt}], p.kernel_impl

    def test_the_oracle_rounds_q_then_places_it_on_the_mxfp8_grid(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _mxfp8_rows_on_grid, get_ref
        spec = get_ref("fused_q_kv_rmsnorm_quant")
        inv = self._cell()
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)

        def norm(x, w):
            y = x.float() * torch.rsqrt(
                x.float().pow(2).mean(-1, keepdim=True) + inputs["eps"]
            ) * w.float()
            return y.to(x.dtype)

        q_want = _mxfp8_rows_on_grid(
            norm(inputs["q"], inputs["q_weight"]))
        kv_want = norm(inputs["kv"], inputs["kv_weight"]).float()
        assert torch.allclose(got, torch.cat([q_want, kv_want], dim=-1),
                              atol=2e-2)
        plain = norm(inputs["q"], inputs["q_weight"]).float()
        assert not torch.allclose(got[:, :plain.shape[-1]], plain, atol=1e-6)

        floor = next(p for p in _providers("fused_q_kv_rmsnorm_quant")
                     if p.library == "torch")
        built = floor.build(inv, TORCH_ENV, inputs)
        assert torch.allclose(built.run(), got, atol=1e-6)

    def test_the_scale_canonicalizer_undoes_flashinfers_swizzle(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.norm_act import _unswizzle_mxfp8_scales
        tokens, width = 130, 128
        groups = width // 32
        padded_tokens = -(-tokens // 128) * 128
        padded_groups = -(-groups // 4) * 4
        row_major = torch.arange(padded_tokens * padded_groups,
                                 dtype=torch.float32).reshape(
            padded_tokens, padded_groups) % 251
        # The same permutation vLLM's `swizzle_mxfp8_scale` performs.
        rows, tiles = padded_tokens // 128, padded_groups // 4
        packed = row_major.view(rows, 4, 32, tiles, 4).transpose(
            1, 3).contiguous().view(-1).to(torch.uint8)
        got = _unswizzle_mxfp8_scales(packed, tokens, width)
        assert torch.equal(got, row_major[:tokens, :groups])


class TestIndexerQRopeQuant:
    """The CSA2 indexer-Q tail rotation, quantisation and weight rescale
    (`fusion_plans/fused_indexer_q_rope_quant{,_split}.yaml`)."""

    DIMS: ClassVar[dict] = {"B": 2, "s_query": 5, "n_heads": 4,
                            "d_head": 128, "d_rope": 64}

    #: The record each output dtype is keyed with: the scale layout and
    #: the scales' dtype, as the engine keys a grafted cast's record.
    RECORDS: ClassVar[dict] = {
        "fp4_e2m1": {"activation_granularity": "per-group:32",
                     "metadata": "ue8m0"},
        "fp8_e4m3": {"activation_granularity": "per-token",
                     "metadata": "fp32"}}

    @staticmethod
    def _cell(signature, output="fp4_e2m1", phase="decode", **dims):
        inv = _inv(signature, {**TestIndexerQRopeQuant.DIMS, **dims},
                   phase=phase)
        inv.dtype_env["output"] = output
        inv.dtype_env.update(TestIndexerQRopeQuant.RECORDS.get(output, {}))
        return inv

    def test_the_engine_keyed_cells_reach_the_launches_that_write_them(self):
        """The engine keys the query launch at the record the index key's
        readers want (`index_key.paired_operand`): FP8 with one F32 scale a
        head, or MXFP4. Each real launch takes the cell of its own record and
        no other; a BF16 cell — what the walk keyed before the cast was
        grafted — is no launch's."""
        def claims(sig, output):
            return {p.kernel_impl for p in _providers(sig)
                    if p.library != "torch"
                    and p.serves.accepts(self._cell(sig, output=output))}
        sig = "fused_indexer_q_rope_quant"
        assert claims(sig, "fp8_e4m3") == {
            "sglang_fused_q_indexer_rope_hadamard_quant",
            "vllm_fused_indexer_q_rope_quant_fp8_e4m3"}
        assert claims(sig, "fp4_e2m1") == {
            "vllm_fused_indexer_q_rope_quant_mxfp4"}
        assert claims(sig, "bf16") == set()
        split = "fused_indexer_q_rope_quant_split"
        assert claims(split, "fp4_e2m1") == {
            "sglang_index_q_rope_pack_weights",
            "vllm_fused_indexer_q_rope_quant_split_mxfp4"}
        assert claims(split, "bf16") == set()
        # The record's scale layout is part of the claim: an FP4 cell with
        # NVFP4's per-16 scales is not the MXFP4 launch's.
        nvfp4 = self._cell(sig, output="fp4_e2m1")
        nvfp4.dtype_env["activation_granularity"] = "per-group:16"
        assert not [p for p in _providers(sig) if p.serves.accepts(nvfp4)]

    def test_both_spans_bind_and_the_real_launches_claim_them(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert bound["fused_indexer_q_rope_quant"] == \
            "fused_indexer_q_rope_quant"
        assert bound["fused_indexer_q_rope_quant_split"] == \
            "fused_indexer_q_rope_quant_split"

        unsplit = _providers("fused_indexer_q_rope_quant")
        assert {p.library for p in unsplit} == {"sglang", "vllm", "torch"}
        assert all(p.fused_op == "fused_indexer_q_rope_quant" for p in unsplit)
        sglang = next(p for p in unsplit if p.library == "sglang")
        assert sglang.serves.output == ("fp8_e4m3",)
        assert sglang.serves.secondary_outputs == (("weights", "fp32"),)
        assert sglang.import_name == (
            "sglang.kernels.ops.attention.dsv4.elementwise")
        # The plan, not a signature-name switch, says which span has V4's
        # Hadamard. Only that vLLM launch needs its no-H deviation oracle.
        assert all(p.reference is not None
                   for p in unsplit if p.library == "vllm")

        split = _providers("fused_indexer_q_rope_quant_split")
        assert {p.library for p in split} == {"vllm", "sglang", "torch"}
        assert all(p.fused_op == "fused_indexer_q_rope_quant_split"
                   for p in split)
        assert all(p.reference is None for p in split if p.library == "vllm")
        sglang = next(p for p in split if p.library == "sglang")
        assert sglang.serves.output == ("fp4_e2m1",)
        assert sglang.serves.activation_granularity == ("per-group:32",)
        assert sglang.serves.phases == ("decode",)
        assert sglang.min_sm == 100
        assert sglang.serves.secondary_outputs == (("weights", "fp32"),)

    def test_both_writings_key_the_head_split_query(self):
        """The walk hands q head-split, `[B, n_heads, s_query, d_head]`,
        beside token-major weights `[B, s_query, n_heads]` (DeepSeek-V4
        prefill, V4.1 decode as planned). Read as token-major q, every row
        dropped unkeyed."""
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        for sig, shape_in, want in (
                ("fused_indexer_q_rope_quant",
                 [[2, 64, 1024, 128], [2, 1024, 64]],
                 {"B": 2, "n_heads": 64, "s_query": 1024}),
                ("fused_indexer_q_rope_quant_split",
                 [[2, 32, 1, 128], [2, 1, 32]],
                 {"B": 2, "n_heads": 32, "s_query": 1})):
            got = identity_shape_from_shape_in(
                sig, shape_in, shape_out=shape_in[0],
                spec_dims={"d_rope": 64},
                operand_ports=_ports("q", "weights"))
            assert got is not None, sig
            assert json.loads(got) == {**want, "d_head": 128, "d_rope": 64}

    def test_the_launch_facts_are_declared_not_hidden_in_build(self):
        unsplit = _providers("fused_indexer_q_rope_quant")
        assert not [p for p in unsplit if p.library == "sglang" and
                    p.accepts(self._cell(
                        "fused_indexer_q_rope_quant", output="fp4_e2m1"),
                        TORCH_ENV)]
        narrow = self._cell("fused_indexer_q_rope_quant", d_head=64)
        assert not [p for p in unsplit if p.accepts(narrow, TORCH_ENV)]

        split = _providers("fused_indexer_q_rope_quant_split")
        assert not [p for p in split if p.library == "sglang" and
                    p.accepts(self._cell(
                        "fused_indexer_q_rope_quant_split",
                        output="fp8_e4m3"), TORCH_ENV)]
        assert not [p for p in split if p.library == "sglang" and
                    p.accepts(self._cell(
                        "fused_indexer_q_rope_quant_split", phase="prefill"),
                        TORCH_ENV)]
        narrow = self._cell("fused_indexer_q_rope_quant_split",
                            d_head=64)
        assert not [p for p in split if p.accepts(narrow, TORCH_ENV)]
        h100 = Env("2.11", None, None, None, None, sm="sm_90",
                   libs={"sglang.kernels.ops.attention.dsv4."
                         "fp4_indexer_rope": "0.5.22"})
        assert not next(p for p in split if p.library == "sglang").available(h100)

    def test_the_unsplit_oracle_hadamards_then_quantises_on_sglangs_grid(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.indexer import _hadamard_rotate
        from kernel_bench.refs import (
            _hadamard_matrix, _indexer_q_rope_quant_vllm_reference,
            _rope_rotate, get_ref)
        sig = "fused_indexer_q_rope_quant"
        spec = get_ref(sig)
        inv = self._cell(sig, output="fp8_e4m3")
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        tokens, heads, width = 10, 4, 128
        rd = int(inputs["d_rope"])
        fixed = (heads * width) ** -0.5
        scale = got[:, -heads:].reshape(tokens, heads) / (
            inputs["weights"].float() * fixed)
        q = got[:, :heads * width].reshape(tokens, heads, width)

        raw = inputs["q"].float()
        roped = torch.cat((
            raw[..., :-rd],
            _rope_rotate(raw[..., -rd:], inputs["positions"],
                         inputs["cos_sin_cache"], rd, is_neox=False)), dim=-1)
        want = torch.matmul(
            roped, _hadamard_matrix(width, roped.device)) * width ** -0.5
        assert torch.all((q - want).abs()
                         <= 0.07 * want.abs() + scale[..., None])
        assert not torch.allclose(q, roped, atol=1e-4)
        assert torch.any(torch.log2(scale) != torch.round(torch.log2(scale)))

        floor = next(p for p in _providers(sig) if p.library == "torch")
        assert torch.allclose(floor.build(inv, TORCH_ENV, inputs).run(), got,
                              atol=1e-6)

        vllm = next(p for p in _providers(sig) if p.library == "vllm")
        launched = _indexer_q_rope_quant_vllm_reference(inputs)
        deviation = vllm.reference(inputs)
        assert torch.equal(
            deviation,
            torch.cat((_hadamard_rotate(
                launched[:, :-heads].reshape(tokens, heads, width)
            ).reshape(tokens, -1), launched[:, -heads:]), dim=-1))
        assert not torch.allclose(deviation, got, atol=1e-6)

    def test_the_oracle_rotates_only_the_tail_and_pays_both_writes(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        sig = "fused_indexer_q_rope_quant_split"
        spec = get_ref(sig)
        inv = self._cell(sig)
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        tokens, heads, width = 10, 4, 128
        assert got.shape == (tokens, heads * width + heads)
        q = inputs["q"].float()
        assert torch.allclose(got[:, :heads * (width - inputs["d_rope"])],
                              q[..., :-inputs["d_rope"]].reshape(tokens, -1),
                              atol=8.0)
        tail = got[:, heads * (width - inputs["d_rope"]):heads * width]
        assert not torch.allclose(tail.reshape(tokens, heads, -1),
                                  q[..., -inputs["d_rope"]:], atol=1e-4)
        fixed = (heads * width) ** -0.5
        # sglang's weights write is `float(bf16(w * weight_scale))`
        # (fp4_indexer_rope.cuh:101-103); a plain fp32 multiply is a
        # different function.
        want_w = (inputs["weights"].float() * fixed).to(
            torch.bfloat16).float()
        assert torch.allclose(got[:, -heads:], want_w, atol=1e-7)
        assert not torch.allclose(got[:, -heads:],
                                  inputs["weights"].float() * fixed,
                                  atol=1e-7)
        floor = next(p for p in _providers(sig) if p.library == "torch")
        assert torch.allclose(floor.build(inv, TORCH_ENV, inputs).run(), got,
                              atol=1e-6)

    def test_the_split_oracle_keeps_the_packers_own_floor(self):
        """sglang's split kernel quantises TWICE (fp4_indexer_rope.cuh:
        89-144): the packer's amax floor is 1e-4 past the divide by 6, where
        the fake-quant's is 6·2^-126 before it. A block the first pass
        resolved below the packer's floor is zeroed by the second; the
        one-pass vLLM oracle keeps it, so the two are different functions
        there and the split signature gets its own reference."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        sig = "fused_indexer_q_rope_quant_split"
        spec = get_ref(sig)
        inputs = build_inputs(spec, self._cell(sig), "cpu")
        inputs["q"] = inputs["q"].clone()
        inputs["q"][0, 0] *= 1e-30        # one head row below both floors
        got = spec.reference(inputs)
        assert torch.all(got[0, :128] == 0)
        assert torch.any(got[1:, :128] != 0)
        one_pass = get_ref("fused_indexer_q_rope_quant").reference(inputs)
        assert torch.any(one_pass[0, :128] != 0)

    def test_sglang_is_handed_the_interleaved_float_cis_table(self):
        """Production passes `view_as_real(freqs_cis).flatten(-2)`
        (scoring.py:125-131): fp32, real/imag interleaved, trailing dim
        d_rope. The wrapper JIT-specializes on `freqs_cis.shape[-1]`
        (fp4_indexer_rope.py:119) and the matcher requires the interleaved
        float form (.cuh:41); the complex [max_pos, 32] table the provider
        used to build specializes the kernel the static_assert rejects."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.indexer import _sgl_indexer_freqs
        from kernel_bench.refs import _rope_cos_sin_cache
        table = _rope_cos_sin_cache(torch.arange(8), 64, 10000.0)
        got = _sgl_indexer_freqs(table, 64)
        assert got.dtype == torch.float32
        assert got.shape == (8, 64)
        want = torch.view_as_real(
            torch.complex(table[:, :32], table[:, 32:])).flatten(-2)
        assert torch.equal(got, want)

    def test_sglang_hadamard_is_handed_the_complex_cis_table(self):
        """C4's wrapper performs `view_as_real(...).flatten(-2)` itself, so
        the provider must hand it the model's complex table, not pre-flatten
        the V4.1 split kernel's float form."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.indexer import _sgl_indexer_cis
        from kernel_bench.refs import _rope_cos_sin_cache
        table = _rope_cos_sin_cache(torch.arange(8), 64, 10000.0)
        got = _sgl_indexer_cis(table, 64)
        assert got.dtype == torch.complex64
        assert got.shape == (8, 32)
        assert torch.equal(got, torch.complex(table[:, :32], table[:, 32:]))

    def test_the_vllm_canonicalizer_restores_hadamard_without_timing_it(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.indexer import _hadamard_rotate
        from kernel_bench.refs import _hadamard_matrix
        x = torch.randn(3, 4, 128)
        got = _hadamard_rotate(x)
        want = torch.matmul(x, _hadamard_matrix(128, x.device)) * 128 ** -0.5
        assert torch.allclose(got, want, atol=1e-6)

    def test_mxfp4_canonicalization_decodes_both_framework_records(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.indexer import _decode_mxfp4
        payload_bytes = torch.zeros(16, dtype=torch.uint8)
        payload_bytes[:2] = torch.tensor([0x10, 0xF2], dtype=torch.uint8)
        # Low nibble is even logical elements, high nibble odd. One packed
        # word per block holds all four UE8M0 exponents, little-endian.
        payload = payload_bytes.view(torch.int8)
        scales = torch.tensor([127], dtype=torch.int32)
        got = _decode_mxfp4(payload, scales, 1, 1, 32)
        want = torch.tensor([0.0, 0.5, 1.0, -6.0])
        assert torch.allclose(got.flatten()[:4], want)

    def test_mxfp4_scales_unpack_per_head_not_per_batch(self):
        """The (T, H) int32 scale word viewed as uint8 is (T, H*4): cropping
        the flat last axis keeps the FIRST head's bytes for every head. Two
        heads with different exponents catch it; the tokens=1, heads=1 cell
        above cannot."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.indexer import _decode_mxfp4
        tokens, heads, width = 2, 3, 32
        payload = torch.zeros(tokens, heads, width // 2, dtype=torch.int8)
        payload[..., 0] = 0x21        # logical elements 0/1 = (0.5, 1.0)
        # A distinct exponent per head: 2^-2, 2^-1, 2^0 (byte 0; the other
        # three bytes of each word are beyond this width's one group).
        words = torch.tensor(
            [[125, 126, 127], [125, 126, 127]], dtype=torch.int32)
        for scales in (words, words.reshape(-1)):   # vLLM (T,H); sglang (T*H,)
            got = _decode_mxfp4(payload, scales, tokens, heads, width)
            want = torch.tensor([0.5, 1.0]) * torch.exp2(
                torch.arange(-2.0, 1.0)).unsqueeze(-1)
            assert torch.allclose(got[..., :2].reshape(tokens, heads, 2),
                                  want.expand(tokens, heads, 2))
            assert torch.all(got[..., 2:] == 0)


class TestFusedQkMrope:
    """Multimodal RoPE on Q and K (`fusion_plans/fused_qk_mrope.yaml`)."""

    DIMS: ClassVar[dict] = {"B": 2, "s_query": 7, "n_heads": 4,
                            "n_kv_heads": 2, "d_head": 256, "d_rope": 64,
                            "mrope_section_t": 11,
                            "mrope_section_h": 11,
                            "mrope_section_w": 10,
                            "mrope_interleaved": 0}

    @staticmethod
    def _cell(**overrides):
        return _inv("fused_qk_mrope",
                    {**TestFusedQkMrope.DIMS, **overrides})

    def test_the_span_binds_and_all_three_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        assert op_bindings()["fused_qk_mrope"] == "fused_qk_mrope"
        ps = _providers("fused_qk_mrope")
        assert {p.library for p in ps} == {"vllm", "sglang", "torch"}
        assert all(p.fused_op == "fused_qk_mrope" for p in ps)
        # The sections are structural axes, carried by every cell; no
        # presence gate (one refused Qwen3.5's `mrope_interleaved=0`).
        assert all(p.serves.requires_dims == () for p in ps)
        assert all(p.serves.enum == (("d_head", (128, 256)),
                                     ("d_rope", (64, 128))) for p in ps)

    def test_sections_are_part_of_the_cell_not_a_build_default(self):
        ps = _providers("fused_qk_mrope")
        # A section the cell does not carry is no default the launch picks:
        # resolving the launch's parameters reads it, before any input.
        missing = self._cell()
        del missing.dims["mrope_section_h"]
        for p in ps:
            with pytest.raises(KeyError):
                p.impl_params(missing)
        # ...while Qwen3.5's packed sections (`mrope_interleaved=0`) are a
        # cell like any other: a presence gate on that axis refused them.
        assert all(p.accepts(self._cell(mrope_interleaved=0), TORCH_ENV)
                   for p in ps)
        wrong_sum = self._cell(mrope_section_w=11)
        for p in ps:
            with pytest.raises(ValueError):
                p.impl_params(wrong_sum)
        narrow = self._cell(d_head=64)
        assert not [p for p in ps if p.accepts(narrow, TORCH_ENV)]

    def test_the_oracle_reads_each_channel_sections_own_position(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _mrope_axis, get_ref
        spec = get_ref("fused_qk_mrope")
        inv = self._cell()
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        pos, cache = inputs["positions"], inputs["cos_sin_cache"]
        lane = _mrope_axis(inputs["sections"], False)
        q = inputs["q"].float().view(14, 4, 256)
        k = inputs["k"].float().view(14, 2, 256)

        def manual(x, heads):
            cs = torch.stack([cache.index_select(0, pos[a]) for a in range(3)])
            c = torch.where(lane[None, :] == 0, cs[0, :, :32],
                            torch.where(lane[None, :] == 1, cs[1, :, :32],
                                        cs[2, :, :32]))
            s = torch.where(lane[None, :] == 0, cs[0, :, 32:],
                            torch.where(lane[None, :] == 1, cs[1, :, 32:],
                                        cs[2, :, 32:]))
            y0, y1 = x[..., :32], x[..., 32:64]
            c, s = c[:, None, :], s[:, None, :]
            return torch.cat((y0 * c - y1 * s, y1 * c + y0 * s,
                              x[..., 64:]), -1).to(inputs["q"].dtype).float()

        want = torch.cat([manual(q, 4).reshape(14, -1),
                          manual(k, 2).reshape(14, -1)], dim=1)
        assert torch.allclose(got, want, atol=1e-5)
        floor = next(p for p in _providers("fused_qk_mrope")
                     if p.library == "torch")
        assert torch.allclose(floor.build(inv, TORCH_ENV, inputs).run(), got,
                              atol=1e-6)

    def test_the_oracle_interleaves_lanes_when_the_flag_says_so(self):
        """`mrope_interleaved=1` is Qwen3-VL's modulo-three lane ownership —
        the kernels' `<= 3*section` tests (vLLM mrope.py:68-74, sglang
        rotary_triton.py:56-59) — a different mask than the packed sections,
        and the variant smoke walks. The pinned-cell class above carries
        `mrope_interleaved=0`, so the interleaved path gets its own cell."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _mrope_axis, get_ref
        spec = get_ref("fused_qk_mrope")
        inv = self._cell(mrope_interleaved=1)
        inputs = build_inputs(spec, inv, "cpu")
        assert inputs["interleaved"] == 1
        got = spec.reference(inputs)
        pos, cache = inputs["positions"], inputs["cos_sin_cache"]
        lane = _mrope_axis(inputs["sections"], True).to(pos.device)
        # The interleaved mask is not the packed one...
        assert not torch.equal(lane.cpu(),
                               _mrope_axis(inputs["sections"], False))

        def manual(x):
            cs = torch.stack([cache.index_select(0, pos[a])
                              for a in range(3)])
            c = torch.where(lane[None, :] == 0, cs[0, :, :32],
                            torch.where(lane[None, :] == 1, cs[1, :, :32],
                                        cs[2, :, :32]))
            s = torch.where(lane[None, :] == 0, cs[0, :, 32:],
                            torch.where(lane[None, :] == 1, cs[1, :, 32:],
                                        cs[2, :, 32:]))
            y0, y1 = x[..., :32], x[..., 32:64]
            c, s = c[:, None, :], s[:, None, :]
            return torch.cat((y0 * c - y1 * s, y1 * c + y0 * s,
                              x[..., 64:]), -1).to(inputs["q"].dtype).float()

        want = torch.cat([manual(inputs["q"].float().view(14, 4, 256)
                                 ).reshape(14, -1),
                          manual(inputs["k"].float().view(14, 2, 256)
                                 ).reshape(14, -1)], dim=1)
        assert torch.allclose(got, want, atol=1e-5)
        # ...and the flag changes the function: the packed-mask reference on
        # the same inputs disagrees.
        packed = spec.reference({**inputs, "interleaved": 0})
        assert not torch.allclose(got, packed, atol=1e-6)

    def test_the_timed_views_stay_rows_of_the_fused_qkv(self):
        """The timed q/k rows keep the packed projection's pitch (the unread
        V tail widens the row stride). Production hands rotary the qk-norm's
        contiguous rows — see `_production_qkv_views`; what is pinned here
        is the form the provider times, and that the kernels admit it."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.rope import _production_qkv_views
        from kernel_bench.refs import get_ref
        inputs = build_inputs(get_ref("fused_qk_mrope"), self._cell(), "cpu")
        q, k, q3, k3 = _production_qkv_views(inputs, torch, torch.bfloat16)
        assert q.stride() == (2048, 1) and k.stride() == (2048, 1)
        assert q.shape == (14, 1024) and k.shape == (14, 512)
        assert torch.equal(q.view(torch.bfloat16).view(14, 4, 256), q3)
        assert torch.equal(k.view(torch.bfloat16).view(14, 2, 256), k3)


class TestFusedQkNormPartialRope:
    """MiniMax-M3's per-head Gemma norms plus partial rope
    (`fusion_plans/fused_qk_norm_partial_rope.yaml`)."""

    DIMS: ClassVar[dict] = {"B": 2, "s_query": 7, "n_heads": 4,
                            "n_kv_heads": 2, "d_head": 128, "d_rope": 64,
                            "n_index_heads": 4, "kv_block_size": 16}

    @classmethod
    def _cell(cls, activation="bf16", **dims):
        inv = _inv("fused_qk_norm_partial_rope",
                   {**cls.DIMS, **dims})
        inv.dtype_env.update({"activation": activation,
                              "weight": activation, "kv": activation})
        return inv

    def test_the_span_binds_and_each_real_launch_claims_it(self):
        from backend.inference.op_signatures import op_bindings
        assert op_bindings()["fused_qk_norm_partial_rope"] == \
            "fused_qk_norm_partial_rope"
        ps = _providers("fused_qk_norm_partial_rope")
        assert {p.library for p in ps} == {"vllm", "sglang", "torch"}
        assert all(p.fused_op == "fused_qk_norm_partial_rope" for p in ps)
        sglang = next(p for p in ps if p.library == "sglang")
        assert sglang.serves.activation == ("bf16",)
        assert sglang.serves.enum == (("d_head", (128,)),
                                      ("d_rope", (64,)))
        for dtype in ("bf16", "fp16"):
            vllm = next(p for p in ps if p.library == "vllm"
                        and p.serves.activation == (dtype,))
            assert vllm.serves.dtype_slots == (("kv", (dtype,)),)
            assert vllm.serves.secondary_outputs == (
                ("index_q", dtype), ("kv_cache", dtype),
                ("index_cache", dtype))
            assert vllm.serves.enum == (("d_head", (128,)),
                                        ("d_rope", (64,)),
                                        ("kv_block_size", (16,)))

    def test_the_launch_scopes_are_not_widened(self):
        ps = _providers("fused_qk_norm_partial_rope")
        assert not [p for p in ps if p.library == "sglang" and
                    p.accepts(self._cell("fp16"), TORCH_ENV)]
        assert not [p for p in ps if p.library == "vllm" and
                    p.accepts(self._cell(kv_block_size=32), TORCH_ENV)]
        assert not [p for p in ps if p.library == "sglang" and
                    p.accepts(self._cell(n_index_heads=8), TORCH_ENV)]
        # The dense geometry: neither variant axis (the page joins only
        # with the index branch whose cache it writes).
        missing_index = self._cell()
        del missing_index.dims["n_index_heads"]
        del missing_index.dims["kv_block_size"]
        assert not [p for p in ps if p.library == "vllm" and
                    p.accepts(missing_index, TORCH_ENV)]
        assert {p.library for p in ps
                if p.accepts(missing_index, TORCH_ENV)} == {"sglang", "torch"}

    def test_the_sparse_geometry_is_selected_not_mandatory(self):
        """`n_index_heads` / `kv_block_size` are variant axes: their PRESENCE
        selects vLLM's sparse cache-writing launch, their absence the dense
        [q|k|v] one. As unconditional structural axes they made the dense
        cell illegal — smoke's minimal cell carried `n_index_heads`, which
        the dense provider had to refuse, so sglang reported `no_cell` for
        its own declaration."""
        pytest.importorskip("torch")
        from backend.inference.op_signatures import get_signature
        from kernel_bench.refs import get_ref
        from kernel_bench.smoke import minimal_dims
        sig = get_signature("fused_qk_norm_partial_rope")
        assert "n_index_heads" not in sig.cell_dims
        assert "kv_block_size" not in sig.cell_dims
        assert set(sig.variant_dims) == {"n_index_heads", "kv_block_size"}
        assert "kv_block_size" not in sig.identity_dims
        dense = {axis: self.DIMS[axis] for axis in sig.cell_dims}
        projected = json.loads(sig.project(json.dumps(dense)))
        assert "kv_block_size" not in projected
        for page in (16, 64):
            sparse = sig.project(json.dumps(
                {**dense, "n_index_heads": 4, "kv_block_size": page}))
            assert json.loads(sparse)["kv_block_size"] == page
        assert sig.project(json.dumps(
            {**dense, "n_index_heads": 4, "kv_block_size": 16})) != \
            sig.project(json.dumps(
                {**dense, "n_index_heads": 4, "kv_block_size": 64}))
        # Every provider accepts its own minimal smoke cell.
        for p in _providers("fused_qk_norm_partial_rope"):
            dims, why = minimal_dims(p, "prefill")
            assert dims is not None, (p.kernel_impl, why)
            act = p.serves.activation[0]
            inv = _inv("fused_qk_norm_partial_rope", dims)
            inv.dtype_env.update(
                {"activation": act, "weight": act, "kv": act})
            assert p.accepts(inv, TORCH_ENV), p.kernel_impl
        # The sparse launch's minimal cell carries the selecting axes; the
        # dense launches' do not.
        vllm = next(p for p in _providers("fused_qk_norm_partial_rope")
                    if p.library == "vllm")
        dims, _ = minimal_dims(vllm, "prefill")
        assert dims["n_index_heads"] > 0 and dims["kv_block_size"] == 16
        dense = next(p for p in _providers("fused_qk_norm_partial_rope")
                     if p.library == "sglang")
        dims, _ = minimal_dims(dense, "prefill")
        assert "n_index_heads" not in dims
        # And the reference builds the dense cell without the sparse axes —
        # they are the sparse launch's build facts, not the oracle's.
        inv = _inv("fused_qk_norm_partial_rope", dims)
        assert "qkv" in build_inputs(get_ref("fused_qk_norm_partial_rope"),
                                     inv, "cpu")

    def test_the_cache_dtype_gate_is_behavioral_not_field_equality(self):
        """`dtype_slots` is a GATE: vLLM's bf16 registration declines a cell
        whose kv cache is fp16. An equality pin on the field would stay
        green if `refusals()` stopped reading `dtype_slots`."""
        vllm_bf16 = next(p for p in _providers("fused_qk_norm_partial_rope")
                         if p.library == "vllm"
                         and p.serves.activation == ("bf16",))
        assert vllm_bf16.accepts(self._cell(), TORCH_ENV)
        wrong = self._cell()
        wrong.dtype_env["kv"] = "fp16"
        assert not vllm_bf16.accepts(wrong, TORCH_ENV)

    def test_the_oracle_is_gemma_norm_then_partial_neox_rope(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _rope_rotate, get_ref
        spec = get_ref("fused_qk_norm_partial_rope")
        inv = self._cell()
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        qkv = inputs["qkv"].float()
        q, k = qkv[:, :512].view(14, 4, 128), qkv[:, 512:768].view(14, 2, 128)

        def want(x, w):
            y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)
                                + inputs["eps"]) * (1 + w.float())
            return _rope_rotate(y, inputs["positions"],
                                inputs["cos_sin_cache"], 64, True).to(
                                    inputs["qkv"].dtype).float()

        expected = torch.cat([
            want(q, inputs["q_norm_w"]).reshape(14, -1),
            want(k, inputs["k_norm_w"]).reshape(14, -1)], dim=1)
        assert torch.allclose(got, expected, atol=1e-5)
        floor = next(p for p in _providers("fused_qk_norm_partial_rope")
                     if p.library == "torch")
        assert torch.allclose(floor.build(inv, TORCH_ENV, inputs).run(), got,
                              atol=1e-6)

    def test_vllm_is_handed_a_sparse_row_real_slots_and_a_paged_cache(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.rope import _vllm_sparse_operands
        from kernel_bench.refs import get_ref
        inv = self._cell()
        inputs = build_inputs(get_ref("fused_qk_norm_partial_rope"), inv, "cpu")
        operands = _vllm_sparse_operands(inv, inputs, torch,
                                         torch.bfloat16)
        qkv, q_out, index_q, cache, index_cache, index_w, slots, index_slots = \
            operands
        assert qkv.shape == (14, 1664)
        assert qkv.stride() == (1664, 1)
        assert cache.shape == (2, 2, 16, 256)
        assert cache.dtype == torch.bfloat16
        assert q_out.shape == (14, 512) and index_q.shape == (14, 512)
        assert index_cache.shape == (2, 16, 128)
        assert index_w.shape == (2, 128)
        assert slots.shape == (14,) and slots.dtype == torch.int64
        assert index_slots.shape == (14,)
        assert len(torch.unique(slots)) == slots.numel()
        assert slots.equal(inputs["slots"])
        assert index_slots.equal(inputs["index_slots"])
        assert index_w.equal(inputs["index_w"])
        assert qkv[:, :1024].equal(inputs["qkv"])


class TestGatedDeltaRuleDecodeStep:
    """The Gated DeltaNet decode step with gates and q/k norms folded in
    (`fusion_plans/fused_recurrent_gated_delta_rule.yaml`)."""

    DIMS: ClassVar[dict] = {"B": 3, "s_query": 1, "n_lin_k_heads": 2, "d_lin_k_head": 16,
                            "n_lin_v_heads": 4, "d_lin_v_head": 16}

    def test_the_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        sig = "fused_recurrent_gated_delta_rule"
        assert op_bindings()[sig] == sig
        ps = _providers(sig)
        assert {p.library for p in ps} == {"vllm", "sglang", "torch"}
        assert all(p.fused_op == sig for p in ps)
        # One token per sequence is the packed kernel's contract.
        wide = _inv(sig, {**self.DIMS, "s_query": 4}, phase="decode")
        assert {p.library for p in ps if p.accepts(wide, TORCH_ENV)} == {
            "torch"}

    @pytest.mark.regression          # walks Qwen3.5-35B-A3B in both phases
    def test_the_plan_reaches_the_decode_step_only(self):
        sig = "fused_recurrent_gated_delta_rule"
        for fw in ("vllm", "sglang"):
            jobs, _ = _plan("Qwen/Qwen3.5-35B-A3B", "decode", fw)
            cells = {tuple(sorted(j["dims"].items())) for j in jobs
                     if j["signature"] == sig}
            assert cells == {tuple(sorted({
                "B": 2, "s_query": 1, "n_lin_k_heads": 16,
                "d_lin_k_head": 128, "n_lin_v_heads": 32,
                "d_lin_v_head": 128}.items()))}, (fw, cells)
            jobs, _ = _plan("Qwen/Qwen3.5-35B-A3B", "prefill", fw)
            assert not [j for j in jobs if j["signature"] == sig], fw


class TestKdaDecodeStep:
    """The KDA decode step with the bounded gate folded in
    (`fusion_plans/fused_recurrent_kda.yaml`, GLM-5.3-Flash, and
    `fused_recurrent_kda_per_channel.yaml`, Kimi-K3 — one cell, because both
    frameworks read Kimi's stored rate vector one scalar a head)."""

    DIMS: ClassVar[dict] = {"B": 3, "s_query": 1, "n_lin_k_heads": 2,
                            "d_lin_k_head": 16, "n_lin_v_heads": 2,
                            "d_lin_v_head": 16}

    def test_the_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        sig = "fused_recurrent_kda"
        assert op_bindings()[sig] == sig
        assert op_bindings()["fused_recurrent_kda_per_channel"] == sig
        ps = _providers(sig)
        assert {p.kernel_impl for p in ps} == {
            "vllm_fused_recurrent_kda_compute_gate",
            "vllm_fused_recurrent_kda_packed_decode",
            "sglang_fused_sigmoid_gating_delta_rule_update_kda",
            "torch_fused_recurrent_kda"}
        assert all(p.fused_op == sig for p in ps)
        # vLLM's wrapper allocates o like k: H == HV and K == V only.
        gva = _inv(sig, {**self.DIMS, "n_lin_v_heads": 4}, phase="decode")
        assert "vllm" not in {p.library for p in ps
                              if p.accepts(gva, TORCH_ENV)}

    def test_the_canonical_inputs_are_the_cells_own_dtype(self):
        """`make_inputs` builds the canonical tensors for the invocation:
        an fp32 cell (the torch floor's, now that no framework provider
        accepts one) gets fp32 q/k/v/f/b — not a bf16 grid the floor then
        widens back, which would time values the cell never had."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        sig = "fused_recurrent_kda"
        for act, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            inv = Invocation(op=sig, signature=sig, phase="decode",
                             dims=dict(self.DIMS), tp=1,
                             dtype_env={"activation": act})
            inputs = build_inputs(get_ref(sig), inv, "cpu")
            for name in ("q", "k", "v", "f", "b"):
                assert inputs[name].dtype is dt, (act, name)

    def test_the_frameworks_serve_only_the_models_own_precisions(self):
        """The step is launched at the model's activation dtype — bf16 in
        every checkpoint that binds the span, fp16 the other width the
        Triton kernels compile for. An fp32 cell is the torch floor's, not
        a launch either framework makes — the builds used to take one and
        run it at bf16 under an fp32 key."""
        sig = "fused_recurrent_kda"
        for act, want in (("bf16", {"vllm", "sglang"}),
                          ("fp16", {"vllm", "sglang"}), ("fp32", set())):
            inv = Invocation(op=sig, signature=sig, phase="decode",
                             dims=dict(self.DIMS), tp=1,
                             dtype_env={"activation": act})
            got = {p.library for p in _providers(sig)
                   if p.library != "torch" and p.serves.accepts(inv)}
            assert got == want, act

    @pytest.mark.regression          # walks GLM-5.3-Flash and Kimi-K3
    def test_the_plan_reaches_each_models_decode_step(self):
        sig = "fused_recurrent_kda"

        def cells(model, fw):
            jobs, _ = _plan(model, "decode", fw)
            return {tuple(sorted(j["dims"].items())) for j in jobs
                    if j["signature"] == sig}

        def cell(heads):
            return tuple(sorted({
                "B": 2, "s_query": 1, "n_lin_k_heads": heads,
                "d_lin_k_head": 128, "n_lin_v_heads": heads,
                "d_lin_v_head": 128}.items()))
        for fw in ("vllm", "sglang"):
            assert cells("zai-org/GLM-5.3-Flash", fw) == {cell(64)}, fw
        # Kimi-K3 at attention TP 1: sglang runs this step; vLLM runs the
        # whole mixer in one launch instead (`fused_kda_decode`).
        assert cells("moonshotai/Kimi-K3", "sglang") == {cell(96)}
        assert cells("moonshotai/Kimi-K3", "vllm") == set()


class TestGdnGating:
    """The Gated DeltaNet gate prep (`fusion_plans/fused_gdn_gating.yaml`),
    sglang's prefill launch."""

    def test_the_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        sig = "fused_gdn_gating"
        assert op_bindings()[sig] == sig
        ps = _providers(sig)
        assert {p.kernel_impl for p in ps} == {"sglang_fused_gdn_gating",
                                               "torch_fused_gdn_gating"}
        assert all(p.fused_op == sig for p in ps)

    @pytest.mark.regression          # walks Qwen3.5-35B-A3B four times
    def test_the_plan_reaches_sglangs_prefill_only(self):
        sig = "fused_gdn_gating"
        jobs, _ = _plan("Qwen/Qwen3.5-35B-A3B", "prefill", "sglang")
        cells = {tuple(sorted(j["dims"].items())) for j in jobs
                 if j["signature"] == sig}
        assert cells == {(("B", 2), ("n_lin_v_heads", 32),
                          ("s_query", 1024))}, cells
        for fw, phase in (("sglang", "decode"), ("vllm", "prefill"),
                          ("vllm", "decode")):
            jobs, _ = _plan("Qwen/Qwen3.5-35B-A3B", phase, fw)
            assert not [j for j in jobs if j["signature"] == sig], (fw, phase)


class TestKdaGate:
    """The KDA prefill gates (`fusion_plans/fused_kda_gate.yaml`) — sglang's
    `kda_gate_chunk_cumsum`, the bounded gate with the chunk-local cumsum and
    the log2 scale folded in, beta sigmoid beside it."""

    DIMS: ClassVar[dict] = {"B": 2, "s_query": 130, "n_lin_v_heads": 4,
                            "d_lin_k_head": 16}

    def test_the_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        sig = "fused_kda_gate"
        assert op_bindings()[sig] == sig
        ps = _providers(sig)
        assert {p.kernel_impl for p in ps} == {"sglang_kda_gate_chunk_cumsum",
                                               "torch_fused_kda_gate"}
        assert all(p.fused_op == sig for p in ps)

    def test_the_oracle_is_the_launchs_gate_cumsum_and_beta(self):
        """The launch writes the bounded gate CHUNK-CUMSUMMED and in log2 —
        the form the chunk kernels read — not the plain gate the naive
        pattern computes. The oracle must state that wider function, checked
        here against the naive gate recomputed by hand (a short last chunk
        included, so the padding path runs)."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import (KDA_GATE_CHUNK, RCP_LN2,
                                       kda_bounded_gates, get_ref)
        spec = get_ref("fused_kda_gate")
        inv = _inv("fused_kda_gate", dict(self.DIMS))
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        K = self.DIMS["d_lin_k_head"]
        assert got.shape == (2, 130, self.DIMS["n_lin_v_heads"], K + 1)
        g, beta = kda_bounded_gates(inputs["f"], inputs["b"], inputs["A_log"],
                                    inputs["dt_bias"], inputs["lower_bound"])
        T = g.shape[1]
        pad = -T % KDA_GATE_CHUNK
        padded = torch.nn.functional.pad(g, (0, 0, 0, 0, 0, pad))
        want_g = padded.unflatten(1, (-1, KDA_GATE_CHUNK)).cumsum(2).flatten(
            1, 2)[:, :T] * RCP_LN2
        assert not torch.allclose(want_g, g * RCP_LN2)   # the cumsum is real
        assert torch.allclose(got[..., :K], want_g, atol=1e-6)
        assert torch.allclose(got[..., K], beta, atol=1e-6)
        floor = next(p for p in _providers("fused_kda_gate")
                     if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        assert torch.allclose(case.to_canonical(case.run()), got, atol=1e-6)

    @pytest.mark.regression          # walks GLM-5.3-Flash and Kimi-K3
    def test_the_plan_reaches_each_models_prefill_on_sglang_only(self):
        sig = "fused_kda_gate"
        for model, heads in (("zai-org/GLM-5.3-Flash", 64),
                             ("moonshotai/Kimi-K3", 96)):
            jobs, _ = _plan(model, "prefill", "sglang")
            cells = {tuple(sorted(j["dims"].items())) for j in jobs
                     if j["signature"] == sig}
            assert cells == {tuple(sorted({
                "B": 2, "s_query": 1024, "n_lin_v_heads": heads,
                "d_lin_k_head": 128}.items()))}, (model, cells)
            # vLLM's KDA prefill is FlashKDA's wider launch; decode is the
            # recurrent step. Neither is this span.
            for fw, phase in (("sglang", "decode"), ("vllm", "prefill"),
                              ("vllm", "decode")):
                jobs, _ = _plan(model, phase, fw)
                assert not [j for j in jobs if j["signature"] == sig], \
                    (model, fw, phase)


class TestQkRmsnormRopeGate:
    """Qwen's gated attention prologue (`fusion_plans/fused_qk_rmsnorm_rope_
    gate{,_split_gain}.yaml`): per-head QK-RMSNorm over the whole head, NeoX
    RoPE on its first `d_rope` channels, the gate copied out."""

    SIG = "fused_qk_rmsnorm_rope_gate"
    DIMS: ClassVar[dict] = {"B": 2, "s_query": 5, "n_heads": 4,
                            "n_kv_heads": 2, "d_head": 256, "d_rope": 64}

    def _cell(self, kv="bf16", activation="bf16", **dims):
        return Invocation(op=self.SIG, signature=self.SIG, phase="prefill",
                          dims={**self.DIMS, **dims}, tp=1,
                          dtype_env={"activation": activation,
                                     "weight": "bf16", "kv": kv,
                                     "output": activation})

    def test_both_writings_bind_and_the_real_launches_claim_them(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert bound["fused_qk_rmsnorm_rope_gate"] == self.SIG
        assert bound["fused_qk_rmsnorm_rope_gate_split_gain"] == self.SIG
        ps = _providers(self.SIG)
        assert {p.library for p in ps} == {"torch", "vllm", "sglang"}
        # One provider serves both plans: the plan sibling is the same
        # launch over another spelling of the norm, so `fused_op` names the
        # signature's own span.
        assert all(p.fused_op == self.SIG for p in ps)
        sgl = next(p for p in ps if p.library == "sglang")
        assert sgl.kernel_impl == "sglang_fused_qk_gemma_rmsnorm_rope_gate"

    def test_both_writings_key_one_partial_rotation_cell(self):
        """The engine's operands as each plan hands them: q|gate head-split
        at twice the head, then k (one-node norm) or the q gain then k
        (split gain). Both key the cell at the head width, with the
        rotation width read off the leaf."""
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        want = {"B": 2, "s_query": 1024, "n_heads": 24, "n_kv_heads": 2,
                "d_head": 256, "d_rope": 64}
        for op, shape_in, ports in (
                ("fused_qk_rmsnorm_rope_gate",
                 [[2, 24, 1024, 512], [2, 2, 1024, 256]],
                 _ports("q_gate", "k")),
                ("fused_qk_rmsnorm_rope_gate_split_gain",
                 [[2, 24, 1024, 512], [256], [2, 2, 1024, 256], [256]],
                 _ports("q_gate", "q_weight", "k", "k_weight"))):
            got = identity_shape_from_shape_in(
                op, shape_in, shape_out=[2, 24, 1024, 256],
                spec_dims={"d_rope": 64}, operand_ports=ports)
            assert got is not None and json.loads(got) == want, op
            # No rotation width on the leaf: no cell, not a guessed one.
            assert identity_shape_from_shape_in(
                op, shape_in, shape_out=[2, 24, 1024, 256],
                spec_dims={}, operand_ports=ports) is None, op

    def test_the_launch_facts_are_declared_not_hidden_in_build(self):
        ps = [p for p in _providers(self.SIG) if p.library != "torch"]
        assert all(p.accepts(self._cell(), TORCH_ENV) for p in ps)
        # Both launches write k as fresh activation rows; a cell whose key
        # cache is FP8 is the insert's format, which neither writes.
        assert not [p for p in ps if p.accepts(self._cell(kv="fp8_e4m3"),
                                               TORCH_ENV)]
        assert not [p for p in ps if p.accepts(self._cell(d_rope=63),
                                               TORCH_ENV)]
        assert not [p for p in ps if p.accepts(
            self._cell(activation="fp32"), TORCH_ENV)]

    def test_the_oracle_rotates_only_the_rope_slice(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref(self.SIG)
        inputs = build_inputs(spec, self._cell(), "cpu")
        got = spec.reference(inputs)
        N, Nk, D, rd = 4, 2, 256, 64
        rows = got.shape[0]
        assert got.shape == (rows, (2 * N + Nk) * D)
        q = inputs["q_gate"].float().view(rows, N, 2, D)
        normed = q[:, :, 0] * torch.rsqrt(
            q[:, :, 0].pow(2).mean(-1, keepdim=True) + inputs["eps"]) \
            * inputs["q_norm_w"].float()
        q_out = got[:, :N * D].view(rows, N, D)
        assert torch.allclose(q_out[..., rd:], normed[..., rd:], atol=1e-5)
        assert not torch.allclose(q_out[1:, :, :rd], normed[1:, :, :rd],
                                  atol=1e-3)
        gate = got[:, (N + Nk) * D:].view(rows, N, D)
        assert torch.equal(gate, q[:, :, 1])

    def test_the_kernels_are_handed_the_checkpoint_gain_exactly(self):
        """Both kernels add one to the gain they read (sglang
        `WEIGHT_SHIFT`, fused_qk_rmsnorm_rope_gate.py:64, :92; vLLM
        `norm_beta=1.0`, qwen3_next.py:427). The provider hands `g - 1`;
        shifting it back in the kernel's fp32 recovers the oracle's gain to
        the bit, and the Gemma-form norm then is the oracle's norm."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.rope import _gemma_gain
        from kernel_bench.refs import get_ref
        inputs = build_inputs(get_ref(self.SIG), self._cell(), "cpu")
        for name in ("q_norm_w", "k_norm_w"):
            g = inputs[name]
            raw = _gemma_gain(g)
            assert raw.dtype == g.dtype == torch.bfloat16
            assert torch.equal(raw.float() + 1.0, g.float())

    def test_the_operands_are_the_fused_projections_strided_views(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.rope import _qkv_gate_views
        from kernel_bench.refs import get_ref
        inputs = build_inputs(get_ref(self.SIG), self._cell(), "cpu")
        qg, k = _qkv_gate_views(inputs, torch, torch.bfloat16)
        N, Nk, D = 4, 2, 256
        assert qg.stride(0) == k.stride(0) == 2 * N * D + 2 * Nk * D
        assert qg.data_ptr() + 2 * N * D * qg.element_size() == k.data_ptr()
        rows = qg.shape[0]
        assert torch.equal(qg.reshape(rows, N, 2, D),
                           inputs["q_gate"].reshape(rows, N, 2, D))
        assert torch.equal(k, inputs["k"].reshape(rows, Nk * D))

    def test_the_floor_meets_the_oracle(self):
        _floor_meets_the_oracle(self.SIG, self.DIMS)


class TestIndexerBlockScore:
    """MiniMax-M3's block-max indexer scores
    (`fusion_plans/indexer_block_score.yaml`)."""

    DIMS: ClassVar[dict] = {"B": 1, "s_query": 200, "s_kv": 200,
                            "n_indexer_heads": 2, "d_indexer_head": 16,
                            "s_sparse_block": 128}

    def test_the_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        sig = "indexer_block_score"
        assert op_bindings()[sig] == sig
        ps = _providers(sig)
        assert {p.kernel_impl for p in ps} == {"torch_dense_block_max_scores",
                                               "vllm_minimax_m3_index_score"}
        assert all(p.fused_op == sig for p in ps)
        # vLLM's score launch is the prefill kernel, compiled for M3's exact
        # geometry; its decode launch force-writes init/local sentinels the
        # span does not model, so it is not a provider.
        vllm = next(p for p in ps if p.library == "vllm")
        assert vllm.serves.phases == ("prefill",)
        assert vllm.serves.enum == (("s_sparse_block", (128,)),
                                    ("d_indexer_head", (128,)))
        odd = _inv(sig, {**self.DIMS, "s_sparse_block": 64, "kv_block_size": 64})
        odd.dtype_env["output"] = "fp32"     # the scores both launches write
        assert {p.library for p in ps if p.accepts(odd, TORCH_ENV)} == {"torch"}

    def test_the_oracle_is_causal_and_the_floor_matches(self):
        """Every implementation of the span (HF, vLLM, sglang) masks keys
        after the query before the block max; a block the query cannot see
        floors to the sentinel. A one-token decode sees the whole cache,
        which is the only geometry where the non-causal reading agreed."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import MASKED_BLOCK_SCORE, get_ref
        spec = get_ref("indexer_block_score")
        inv = _inv("indexer_block_score", dict(self.DIMS))
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        assert got.shape == (1, 2, 200, 2)
        # Query 0 sees key 0 alone: its second block floors. The last query
        # sees every block. A non-causal max over the same block differs.
        assert (got[0, :, 0, 1] == MASKED_BLOCK_SCORE).all()
        assert (got[0, :, -1, 1] != MASKED_BLOCK_SCORE).all()
        scores = torch.einsum("bshd,btd->bhst", inputs["q"].float(),
                              inputs["k"].float())
        scores = torch.nn.functional.pad(scores, (0, -200 % 128),
                                         value=float("-inf"))
        plain = scores.unflatten(-1, (-1, 128)).amax(-1)
        assert not torch.allclose(got[0, :, 1:, 0], plain[0, :, 1:, 0])
        floor = next(p for p in _providers("indexer_block_score")
                     if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        got_floor = case.to_canonical(case.run())
        assert torch.allclose(got_floor, got, atol=0.1)   # the floor is bf16
        decode = _inv("indexer_block_score",
                      {**self.DIMS, "s_query": 1}, phase="decode")
        seen = spec.reference(build_inputs(spec, decode, "cpu"))
        assert (seen != MASKED_BLOCK_SCORE).all()


class TestMamba2GroupedGatedNorm:
    """sglang's Mamba-2 output norm: silu gate first, RMS per group, then
    the per-channel gain (`mamba2_rmsnorm_gated.yaml`)."""

    @pytest.mark.parametrize("heads,groups", [(6, 3), (6, 1), (8, 2)])
    def test_the_oracle_normalizes_each_group_not_the_whole_row(
            self, heads, groups):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        dims = {"B": 2, "s_query": 5, "n_ssm_heads": heads,
                "d_ssm_head": 8, "n_ssm_groups": groups,
                "d_ssm_state": 20}
        inputs = build_inputs(get_ref("mamba2_rmsnorm_gated"),
                              _inv("mamba2_rmsnorm_gated", dims), "cpu")
        got = get_ref("mamba2_rmsnorm_gated").reference(inputs)
        x = inputs["x"].float() * torch.nn.functional.silu(
            inputs["gate"].float())
        width = x.shape[-1]
        grouped = x.view(2, 5, groups, width // groups)
        want = (grouped * torch.rsqrt(
            grouped.pow(2).mean(-1, keepdim=True) + 1e-6
        )).view_as(x) * inputs["weight"].float()
        assert torch.allclose(got, want, atol=1e-5)

    def test_the_span_binds_and_its_launch_claims_it(self):
        from backend.inference.op_signatures import op_bindings
        assert op_bindings()["mamba2_rmsnorm_gated"] == "mamba2_rmsnorm_gated"
        ps = _providers("mamba2_rmsnorm_gated")
        assert {p.library for p in ps} == {"sglang", "torch"}
        assert all(p.fused_op == "mamba2_rmsnorm_gated" for p in ps)
        inv = _inv("mamba2_rmsnorm_gated", {
            "B": 2, "s_query": 5, "n_ssm_heads": 6, "d_ssm_head": 8,
            "n_ssm_groups": 3, "d_ssm_state": 20})
        inv.dtype_env.update({"activation": "bf16", "weight": "fp32"})
        assert not [p for p in ps if p.serves.accepts(inv)]

    def test_the_floor_runs_the_chain_at_the_cells_width(self):
        _floor_meets_the_oracle(
            "mamba2_rmsnorm_gated",
            {"B": 2, "s_query": 5, "n_ssm_heads": 6, "d_ssm_head": 8,
             "n_ssm_groups": 3, "d_ssm_state": 20},
            env={"activation": "bf16", "weight": "bf16", "output": "bf16"})

    def test_serving_hands_the_gate_a_strided_projection_slice(
            self, monkeypatch):
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        seen = {}

        def rms_norm_gated(**kwargs):
            seen.update(kwargs)
            return torch.empty_like(kwargs["x"])

        TestProductionLaunchLayouts._fake_module(
            monkeypatch,
            "sglang.kernels.ops.attention.fla.layernorm_gated",
            rms_norm_gated=rms_norm_gated)
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        dims = {"B": 2, "s_query": 5, "n_ssm_heads": 6, "d_ssm_head": 8,
                "n_ssm_groups": 3, "d_ssm_state": 20}
        inv = _inv("mamba2_rmsnorm_gated", dims)
        inputs = build_inputs(get_ref("mamba2_rmsnorm_gated"), inv, "cpu")
        na._sgl_mamba2_norm_build(inv, TORCH_ENV, inputs).run()
        x, z = seen["x"], seen["z"]
        assert x.is_contiguous() and x.shape == (10, 48)
        assert z.shape == (10, 48) and z.stride() == (174, 1)
        assert not z.is_contiguous()
        assert torch.equal(z, inputs["gate"].flatten(0, 1))
        assert seen["group_size"] == 16
        assert seen["norm_before_gate"] is False

    @pytest.mark.regression          # walks Nemotron-H's Mamba-2 layers
    def test_the_plan_reaches_the_grouped_norm(self):
        jobs, dropped = _plan(
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "prefill", "sglang")
        assert "mamba2_rmsnorm_gated" not in dropped
        cells = [j["dims"] for j in jobs
                 if j["signature"] == "mamba2_rmsnorm_gated"]
        assert cells and cells[0]["n_ssm_groups"] == 8, cells


class TestGdnPostConvPrep:
    """vLLM's one-launch prefill preparation after the GDN short conv
    (`fused_gdn_post_conv_prep.yaml`)."""

    DIMS: ClassVar[dict] = {
        "B": 2, "s_query": 5, "n_lin_k_heads": 3, "d_lin_k_head": 16,
        "n_lin_v_heads": 4, "d_lin_v_head": 16}

    def test_the_span_binds_and_its_secondary_writes_are_published(self):
        from backend.inference.op_signatures import op_bindings
        assert op_bindings()["fused_gdn_post_conv_prep"] == (
            "fused_gdn_post_conv_prep")
        ps = _providers("fused_gdn_post_conv_prep")
        assert {p.library for p in ps} == {"vllm", "torch"}
        assert all(p.fused_op == "fused_gdn_post_conv_prep" for p in ps)
        vllm = next(p for p in ps if p.library == "vllm"
                    and p.serves.activation == ("bf16",))
        assert vllm.serves.secondary_outputs == (
            ("k", "bf16"), ("v", "bf16"), ("g", "fp32"), ("beta", "fp32"))
        assert vllm.serves.phases == ("prefill",)

    def test_the_oracle_writes_all_five_ports(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import gdn_post_conv_prep, get_ref
        inputs = build_inputs(
            get_ref("fused_gdn_post_conv_prep"),
            _inv("fused_gdn_post_conv_prep", self.DIMS), "cpu")
        q, k, v, g, beta = gdn_post_conv_prep(inputs)
        qkv = inputs["qkv"].float()
        raw_q, raw_k, raw_v = qkv.split([48, 48, 64], dim=-1)
        norm = lambda x: x / torch.sqrt(x.square().sum(-1, keepdim=True)
                                        + 1e-6)
        assert q.shape == (2, 5, 3, 16) and k.shape == q.shape
        assert v.shape == (2, 5, 4, 16)
        assert torch.allclose(q, norm(raw_q.view(2, 5, 3, 16)), atol=1e-5)
        assert torch.allclose(k, norm(raw_k.view(2, 5, 3, 16)), atol=1e-5)
        assert torch.equal(v, raw_v.view(2, 5, 4, 16))
        want_g = -inputs["A_log"].float().exp() * torch.nn.functional.softplus(
            inputs["a"].float() + inputs["dt_bias"].float())
        assert torch.allclose(g, want_g, atol=1e-5)
        assert torch.allclose(beta, torch.sigmoid(inputs["b"].float()),
                              atol=1e-5)
        assert g.dtype is torch.float32 and beta.dtype is torch.float32

    def test_the_floor_runs_the_chain_at_the_cells_width(self):
        _floor_meets_the_oracle(
            "fused_gdn_post_conv_prep", self.DIMS,
            env={"activation": "bf16", "weight": "fp32", "output": "bf16"})

    def test_the_kernels_addressing_reproduces_the_oracle(self, monkeypatch):
        # The kernel reads `mixed_qkv[t, c]` at `t * stride(0) + c` and never
        # reads stride(1) (vLLM v0.32.0 fused_gdn_prefill_post_conv.py:77-123),
        # so this stand-in addresses the conv output the same way. Handed a
        # transposed channel-first buffer, the vLLM build met the oracle at
        # cosine 0.008 on the H100 (2026-10-26).
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.linear_ssm as ssm
        from kernel_bench.refs import get_ref, gdn_gates

        def fused_post_conv_prep(conv, a, b, A_log, dt_bias, H, K, V,
                                 apply_l2norm, output_g_exp):
            L, width = conv.shape
            rows = torch.as_strided(conv, (L, width), (conv.stride(0), 1),
                                    conv.storage_offset()).float()
            q, k, v = rows.split([H * K, H * K, width - 2 * H * K], dim=-1)
            norm = lambda x: x / torch.sqrt(x.square().sum(-1, keepdim=True)
                                            + 1e-6)
            g, beta = gdn_gates(a, b, A_log, dt_bias)
            return (norm(q.reshape(L, H, K)).to(conv.dtype),
                    norm(k.reshape(L, H, K)).to(conv.dtype),
                    v.reshape(L, -1, V).to(conv.dtype), g, beta)

        TestProductionLaunchLayouts._fake_module(
            monkeypatch, "vllm.third_party.flash_linear_attention.ops",
            fused_post_conv_prep=fused_post_conv_prep)
        monkeypatch.setattr(ssm, "_require_cuda", lambda: None)
        inv = _inv("fused_gdn_post_conv_prep", self.DIMS, phase="prefill")
        spec = get_ref("fused_gdn_post_conv_prep")
        inputs = build_inputs(spec, inv, "cpu")
        case = ssm._vllm_gdn_post_conv_build(inv, TORCH_ENV, inputs)
        got = case.to_canonical(case.run()).float().flatten()
        want = spec.reference(inputs).float().flatten()
        assert torch.nn.functional.cosine_similarity(got, want, dim=0) > 0.999

    def test_serving_hands_the_kernel_token_major_conv_and_split_ba_views(
            self, monkeypatch):
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.linear_ssm as ssm
        from kernel_bench.refs import get_ref
        seen = {}

        def fused_post_conv_prep(*args, **kwargs):
            seen.update(args=args, kwargs=kwargs)
            return tuple(torch.empty(args[0].shape[0], 3, 16)
                         for _ in range(5))

        TestProductionLaunchLayouts._fake_module(
            monkeypatch, "vllm.third_party.flash_linear_attention.ops",
            fused_post_conv_prep=fused_post_conv_prep)
        monkeypatch.setattr(ssm, "_require_cuda", lambda: None)
        inv = _inv("fused_gdn_post_conv_prep", self.DIMS, phase="prefill")
        inputs = build_inputs(get_ref("fused_gdn_post_conv_prep"), inv, "cpu")
        ssm._vllm_gdn_post_conv_build(inv, TORCH_ENV, inputs).run()
        conv, a, b, A_log, dt_bias, H, K, V = seen["args"][:8]
        assert conv.shape == (10, 160) and conv.is_contiguous()
        assert torch.equal(conv, inputs["qkv"].flatten(0, 1))
        assert a.stride() == b.stride() == (8, 1)
        assert not a.is_contiguous() and not b.is_contiguous()
        assert a.storage_offset() == 4 and b.storage_offset() == 0
        assert A_log.dtype.is_floating_point and dt_bias.dtype == torch.bfloat16
        assert (H, K, A_log.numel(), V) == (3, 16, 4, 16)
        assert seen["kwargs"] == {"apply_l2norm": True,
                                  "output_g_exp": False}

    @pytest.mark.regression          # walks Qwen4Exp's linear-attention layers
    def test_the_plan_reaches_the_prefill_preparation(self):
        jobs, dropped = _plan("Qwen/Qwen3.8-Flash-Next", "prefill", "vllm")
        assert "fused_gdn_post_conv_prep" not in dropped
        cells = [j["dims"] for j in jobs
                 if j["signature"] == "fused_gdn_post_conv_prep"]
        assert cells and cells[0]["n_lin_k_heads"] == 16, cells


class TestUngroupedRouters:
    """The ungrouped top-k routers (`fusion_plans/moe_topk_*.yaml`)."""

    SPANS: ClassVar[tuple] = ("moe_topk_softmax", "moe_topk_sigmoid",
                              "moe_topk_softmax_bias", "moe_topk_sqrtsoftplus",
                              "moe_topk_hash_sqrtsoftplus")

    def test_each_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        for sig in self.SPANS:
            assert bound[sig] == sig
            ps = _providers(sig)
            libs = {p.library for p in ps}
            # sglang binds every router span but the softmax-with-bias one.
            assert libs == ({"vllm", "torch"} if sig == "moe_topk_softmax_bias"
                            else {"vllm", "sglang", "torch"}), sig
            assert all(p.fused_op == sig for p in ps), sig

    @pytest.mark.parametrize("sig", SPANS)
    def test_the_oracle_is_the_plans_chain(self, sig):
        torch = pytest.importorskip("torch")
        import torch.nn.functional as F
        from kernel_bench.refs import get_ref
        spec = get_ref(sig)
        dims = {"B": 2, "s_query": 3, "n_experts": 16, "n_experts_active": 4}
        if sig == "moe_topk_hash_sqrtsoftplus":
            dims["n_vocab"] = 50
        inputs = build_inputs(spec, _inv(sig, dims), "cpu")
        g = inputs["gating"].float()
        score = {"moe_topk_softmax": torch.softmax(g, -1),
                 "moe_topk_softmax_bias": torch.softmax(g, -1),
                 "moe_topk_sigmoid": torch.sigmoid(g)}.get(
                     sig, torch.sqrt(F.softplus(g)))
        if sig == "moe_topk_hash_sqrtsoftplus":
            ids = inputs["table"].long()[inputs["input_ids"]]
            assert all(len(set(r.tolist())) == 4 for r in ids)
        else:
            sel = score + inputs["bias"] if "bias" in inputs else score
            ids = sel.topk(4, dim=-1).indices
        w = score.gather(-1, ids)
        if inputs["renormalize"]:
            w = w / w.sum(-1, keepdim=True)
        want = torch.zeros_like(score).scatter(-1, ids, w)
        got = spec.reference(inputs)
        assert torch.allclose(got, want, atol=1e-6), sig
        assert got.shape == (6, 16)
        floor = next(p for p in _providers(sig) if p.library == "torch")
        # Through `to_canonical`, as the driver reads every provider: a
        # floor returns what its launch returns (the softmax floor, like the
        # kernels, its weights and ids).
        case = floor.build(_inv(sig, dims), TORCH_ENV, inputs)
        assert torch.allclose(case.to_canonical(case.run()), want, atol=1e-6)

    @pytest.mark.regression          # walks four models in decode
    def test_the_plan_reaches_each_router(self):
        expect = {
            ("Qwen/Qwen3.5-35B-A3B", "moe_topk_softmax"): (256, 8),
            ("tencent/Hy3", "moe_topk_sigmoid"): (192, 8),
            ("deepseek-ai/DeepSeek-V4-Flash", "moe_topk_sqrtsoftplus"): (256, 6),
        }
        for (model, sig), (E, k) in expect.items():
            for fw in ("vllm", "sglang"):
                jobs, _ = _plan(model, "decode", fw)
                cells = {(j["dims"]["n_experts"], j["dims"]["n_experts_active"])
                         for j in jobs if j["signature"] == sig}
                assert cells == {(E, k)}, (model, fw, cells)
        jobs, _ = _plan("meituan-longcat/LongCat-Flash-Thinking-2601",
                        "decode", "vllm")
        assert {(j["dims"]["n_experts"], j["dims"]["n_experts_active"])
                for j in jobs if j["signature"] == "moe_topk_softmax_bias"
                } == {(768, 12)}


class TestPartialRope:
    """Partial rotary embedding on whole heads
    (`fusion_plans/fused_qk_rope_partial{,_q}.yaml`)."""

    def test_the_spans_bind_and_their_kernels_claim_them(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert bound["fused_qk_rope_partial"] == "rope_partial"
        assert bound["fused_qk_rope_partial_q"] == "rope_partial_q"
        for sig, span in (("rope_partial", "fused_qk_rope_partial"),
                          ("rope_partial_q", "fused_qk_rope_partial_q")):
            ps = _providers(sig)
            assert {p.library for p in ps} == {"vllm", "sglang", "torch"}
            assert all(p.fused_op == span for p in ps)

    def test_gqa_rotates_the_head_and_leaves_the_tail(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _rope_rotate, get_ref
        spec = get_ref("rope_partial")
        dims = {"B": 1, "s_query": 5, "n_heads": 4, "n_kv_heads": 2,
                "d_head": 16, "d_rope": 8}
        inv = _inv("rope_partial", dims)
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        q = inputs["q"].float().view(5, 4, 16)
        pos, cache = inputs["positions"], inputs["cos_sin_cache"]
        # The plan: rope(slice(q)[:8]) re-joined with the untouched tail.
        want_q = torch.cat([_rope_rotate(q[..., :8], pos, cache, 8, True),
                            q[..., 8:]], dim=-1)
        assert torch.allclose(got[:, :64].view(5, 4, 16), want_q, atol=1e-5)
        assert torch.equal(got[:, :64].view(5, 4, 16)[..., 8:], q[..., 8:])
        floor = next(p for p in _providers("rope_partial")
                     if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        assert torch.allclose(case.to_canonical(case.run()), got, atol=2e-2)

    def test_mla_rotates_the_query_tail_and_the_whole_rope_key(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _rope_rotate, get_ref
        spec = get_ref("rope_partial_q")
        dims = {"B": 1, "s_query": 5, "n_heads": 3, "d_head": 24,
                "d_rope": 8}
        inv = _inv("rope_partial_q", dims)
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        assert got.shape == (5, 3 * 24 + 8)
        q = inputs["q"].float().view(5, 3, 24)
        pos, cache = inputs["positions"], inputs["cos_sin_cache"]
        assert torch.equal(got[:, :72].view(5, 3, 24)[..., :16], q[..., :16])
        assert torch.allclose(got[:, :72].view(5, 3, 24)[..., 16:],
                              _rope_rotate(q[..., 16:], pos, cache, 8, True),
                              atol=1e-5)
        k = _rope_rotate(inputs["k_pe"].float().view(5, 1, 8), pos, cache, 8,
                         True)
        assert torch.allclose(got[:, 72:], k.view(5, 8), atol=1e-5)
        floor = next(p for p in _providers("rope_partial_q")
                     if p.library == "torch")
        # The floor computes at the cell's dtype (bf16), compared as the
        # driver compares it: canonical, in fp32.
        case = floor.build(inv, TORCH_ENV, inputs)
        assert case.to_canonical(case.run()).dtype == torch.bfloat16
        assert torch.allclose(case.to_canonical(case.run()).float(), got,
                              atol=5e-2)

    @pytest.mark.regression          # walks three models in prefill
    def test_the_plan_keys_head_and_rotation_width(self):
        for model, sig, cell in (
                ("zai-org/GLM-4.5-Air", "rope_partial",
                 {"B": 2, "s_query": 1024, "n_heads": 96, "n_kv_heads": 8,
                  "d_head": 128, "d_rope": 64}),
                ("MiniMaxAI/MiniMax-M2.7", "rope_partial",
                 {"B": 2, "s_query": 1024, "n_heads": 48, "n_kv_heads": 8,
                  "d_head": 128, "d_rope": 64}),
                ("deepseek-ai/DeepSeek-V3", "rope_partial_q",
                 {"B": 2, "s_query": 1024, "n_heads": 128, "d_head": 192,
                  "d_rope": 64})):
            for fw in ("vllm", "sglang"):
                jobs, _ = _plan(model, "prefill", fw)
                cells = [j["dims"] for j in jobs if j["signature"] == sig]
                assert cells and all(c == cell for c in cells), (model, fw,
                                                                 cells)


#: (model, phase, framework) cells whose plans reach every span above.
_PLANNED = (
    ("Qwen/Qwen3.5-35B-A3B", "decode", "vllm"),
    ("Qwen/Qwen3.5-35B-A3B", "prefill", "sglang"),
    ("Qwen/Qwen3.8-Flash-Next", "prefill", "vllm"),
    ("zai-org/GLM-5.3-Flash", "decode", "vllm"),
    ("zai-org/GLM-5.3-Flash", "decode", "sglang"),
    ("zai-org/GLM-5.3-Flash", "prefill", "sglang"),
    ("inclusionAI/Ling-2.6-1T", "decode", "vllm"),
    ("tencent/Hy3", "decode", "sglang"),
    ("deepseek-ai/DeepSeek-V4-Flash", "decode", "vllm"),
    ("meituan-longcat/LongCat-Flash-Thinking-2601", "decode", "vllm"),
    ("zai-org/GLM-4.5-Air", "prefill", "vllm"),
    ("deepseek-ai/DeepSeek-V3", "prefill", "sglang"),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "prefill", "vllm"),
    ("deepseek-ai/DeepSeek-V4-Flash", "prefill", "vllm"),
    ("zai-org/GLM-5.3-Flash", "prefill", "vllm"),
)

_SPAN_SIGNATURES = frozenset({
    "causal_conv1d", "rmsnorm_gated", "fused_recurrent_gated_delta_rule",
    "fused_recurrent_kda", "fused_gdn_gating", "fused_kda_gate",
    "moe_topk_softmax",
    "moe_topk_sigmoid", "moe_topk_softmax_bias", "moe_topk_sqrtsoftplus",
    "rope_partial", "rope_partial_q",
    "fused_save_compress_norm", "fused_compress_rope_store",
    "fused_overlap_compress_norm_rope_store",
    "fused_compress_norm_rope_store", "fused_kpool_compress_store",
    "qsa_index_q_norm_rope", "qsa_index_k_compress_store"})


@pytest.mark.regression              # walks eleven model cells
@pytest.mark.parametrize("model,phase,fw", _PLANNED)
def test_a_planned_job_builds_the_operands_the_graph_emits(model, phase, fw):
    """The contract is the layout the kernel takes; the job carries the
    tensors the fused node emits. Each operand the signature names in the
    engine's order (`engine_inputs`) must be built from that very tensor —
    the same element count — or the measurement launches some other tensor
    under the cell's key."""
    import math
    pytest.importorskip("torch")
    from backend.inference.op_signatures import get_signature
    from kernel_bench.refs import get_ref
    jobs, _ = _plan(model, phase, fw)
    jobs = [j for j in jobs if j["signature"] in _SPAN_SIGNATURES]
    assert jobs, (model, phase, fw)
    for job in jobs:
        sig = get_signature(job["signature"])
        inv = Invocation.from_job(job)
        built = build_inputs(get_ref(sig.name), inv, "meta")
        operands = (job["shape_in"] if isinstance(job["shape_in"][0], list)
                    else [job["shape_in"]])
        # Each layout at the operand that fills its port (`at_ports`); a
        # port the launch has no operand for is no tensor to compare.
        for pos, layout in sig.at_ports(job["operand_ports"]).engine_inputs:
            if pos >= len(operands):
                continue
            emitted = math.prod(int(x) for x in operands[pos])
            got = built[layout.name].numel()
            assert got == emitted, (
                f"{model} {fw} {job['signature']}.{layout.name}: built "
                f"{tuple(built[layout.name].shape)} ({got}) from operand "
                f"{pos} {operands[pos]} ({emitted})")


#: The FlashMLA record DeepSeek-V4's latent and compressed KV are written
#: in, as the cells writing it state it: its element dtype and its name.
_V4_RECORD = {"output": "fp8_e4m3",
              "output_format": "fp8_ue8m0_group64_bf16_rope_tail"}

#: One realistic cell per span (the planned cells of the models above, at
#: decode batch 64 / prefill batch 2), run on a GPU box through each
#: framework provider the environment can import. A fourth element is the
#: cell's kernel parameters; a fifth the dtype slots it states beyond the
#: bf16 defaults — the record a store writes (`output`, `output_format`),
#: or a cache whose record the cell states only by its `kv` dtype.
_GPU_CELLS = (
    ("causal_conv1d", "prefill", {"B": 2, "s_query": 2048,
                                  "d_conv_channels": 8192,
                                  "s_conv_kernel": 4, "dilation": 1}),
    ("causal_conv1d", "decode", {"B": 64, "s_query": 1,
                                 "d_conv_channels": 8192,
                                 "s_conv_kernel": 4, "dilation": 1}),
    ("rmsnorm_gated", "decode", {"B": 64, "s_query": 32, "d_model": 128}),
    ("rmsnorm_gated", "prefill", {"B": 2, "s_query": 1024, "d_model": 8192}),
    ("fused_recurrent_gated_delta_rule", "decode", {
        "B": 64, "s_query": 1, "n_lin_k_heads": 16, "d_lin_k_head": 128,
        "n_lin_v_heads": 32, "d_lin_v_head": 128}),
    ("fused_recurrent_kda", "decode", {
        "B": 64, "s_query": 1, "n_lin_k_heads": 64, "d_lin_k_head": 128,
        "n_lin_v_heads": 64, "d_lin_v_head": 128}),
    ("fused_gdn_gating", "prefill", {"B": 2, "s_query": 2048,
                                     "n_lin_v_heads": 32}),
    ("fused_kda_gate", "prefill", {"B": 2, "s_query": 2048,
                                   "n_lin_v_heads": 64, "d_lin_k_head": 128}),
    ("moe_topk_softmax", "decode", {"B": 64, "s_query": 1, "n_experts": 256,
                                    "n_experts_active": 8}),
    ("moe_topk_sigmoid", "decode", {"B": 64, "s_query": 1, "n_experts": 192,
                                    "n_experts_active": 8}),
    ("moe_topk_softmax_bias", "decode", {"B": 64, "s_query": 1,
                                         "n_experts": 768,
                                         "n_experts_active": 12}),
    ("moe_topk_sqrtsoftplus", "decode", {"B": 64, "s_query": 1,
                                         "n_experts": 256,
                                         "n_experts_active": 6}),
    ("moe_topk_hash_sqrtsoftplus", "decode", {
        "B": 64, "s_query": 1, "n_experts": 256, "n_experts_active": 6,
        "n_vocab": 129280}),
    ("rope_partial", "prefill", {"B": 2, "s_query": 2048, "n_heads": 96,
                                 "n_kv_heads": 8, "d_head": 128,
                                 "d_rope": 64}),
    ("rope_partial_q", "prefill", {"B": 2, "s_query": 2048, "n_heads": 128,
                                   "d_head": 192, "d_rope": 64}),
    ("fused_save_compress_norm", "prefill", {"B": 2, "s_query": 2048,
                                             "d_head": 512,
                                             "compress_ratio": 2}),
    ("fused_compress_rope_store", "prefill", {"B": 2, "s_slots": 1024,
                                              "d_head": 512, "d_rope": 64,
                                              "compress_ratio": 2,
                                              "kv_block_size": 64},
     {}, _V4_RECORD),
    ("fused_overlap_compress_norm_rope_store", "prefill",
     {"B": 2, "s_slots": 256, "Cape": 4, "d_head": 512, "d_rope": 64,
      "kv_block_size": 64}, {}, _V4_RECORD),
    # V4's indexer compressor: the FP8 index-key record.
    ("fused_overlap_compress_norm_rope_store", "prefill",
     {"B": 2, "s_slots": 256, "Cape": 4, "d_head": 128, "d_rope": 64,
      "kv_block_size": 64}, {},
     {"output": "fp8_e4m3", "output_format": "fp8_pow2_group128"}),
    ("fused_compress_norm_rope_store", "prefill",
     {"B": 2, "s_slots": 8, "d_head": 512, "d_rope": 64,
      "compress_ratio": 128, "kv_block_size": 64}, {}, _V4_RECORD),
    # GLM's k-pool store: the walk bills its index-K cache bf16 and names
    # no record, so its FP8 record is stated by the `kv` slot (and banded
    # by it, `driver._xcheck_dtype`).
    ("fused_kpool_compress_store", "prefill",
     {"B": 2, "s_slots": 256, "compress_ratio": 4, "d_indexer_head": 128,
      "kv_block_size": 64}, {}, {"kv": "fp8_e4m3"}),
    ("qsa_index_q_norm_rope", "prefill",
     {"B": 2, "s_query": 1024, "n_indexer_heads": 4, "d_indexer_head": 128,
      "d_rope": 64}),
    ("qsa_index_k_compress_store", "prefill",
     {"B": 2, "s_query": 1024, "s_slots": 256, "compress_ratio": 4,
      "d_indexer_head": 128, "d_rope": 64, "kv_block_size": 64}),
    ("fused_qnorm_rope", "decode", {"B": 64, "s_query": 1, "n_heads": 64,
                                    "d_head": 512, "d_rope": 64}),
    # At `output: bf16` these run sglang's in-place launch; at the FlashMLA
    # record, its SWA store (`fused_k_norm_rope_flashmla`).
    ("fused_norm_rope", "prefill", {"B": 2, "s_query": 1024, "d_head": 512,
                                    "d_rope": 64}),
    ("fused_norm_rope", "decode", {"B": 64, "s_query": 1, "d_head": 128,
                                   "d_rope": 64}),
    ("fused_norm_rope", "prefill", {"B": 2, "s_query": 1024, "d_head": 512,
                                    "d_rope": 64}, {}, _V4_RECORD),
    ("fused_norm_rope", "decode", {"B": 64, "s_query": 1, "d_head": 512,
                                   "d_rope": 64}, {}, _V4_RECORD),
    # A fourth element is the cell's kernel parameters — what the model
    # publishes (`OpSignature.config_params`); DeepSeek-V4's bound is 10.0.
    ("silu_and_mul_with_clamp", "prefill",
     {"B": 2, "s_query": 2048, "d_ffn": 2048}, {"clamp_limit": 10.0}),
    ("silu_and_mul_with_clamp", "decode",
     {"B": 64, "s_query": 1, "d_ffn": 2048}, {"clamp_limit": 10.0}),
    # Kimi-K3's dense MLP at TP 1 and its betas (4.0, 25.0).
    ("situ_and_mul", "prefill", {"B": 2, "s_query": 2048, "d_ffn": 6144},
     {"situ_beta": 4.0, "situ_linear_beta": 25.0}),
    ("situ_and_mul", "decode", {"B": 64, "s_query": 1, "d_ffn": 6144},
     {"situ_beta": 4.0, "situ_linear_beta": 25.0}),


    # sglang's in-place per-head Q/K norm at the bound models' geometry:
    # Qwen3-8B (32 / 8 heads of 128) and GLM-4.5 (96 / 8 of 128).
    ("fused_q_kv_rmsnorm", "decode",
     {"B": 64, "s_query": 1, "n_heads": 32, "n_kv_heads": 8, "d_head": 128}),
    ("fused_q_kv_rmsnorm", "prefill",
     {"B": 2, "s_query": 2048, "n_heads": 96, "n_kv_heads": 8,
      "d_head": 128}),
    # Qwen3.5-35B-A3B (16 / 2 heads) and Qwen3.8-Flash-Next (24 / 2): a
    # 256-wide head, its first 64 channels rotated.
    ("fused_qk_rmsnorm_rope_gate", "decode",
     {"B": 64, "s_query": 1, "n_heads": 16, "n_kv_heads": 2, "d_head": 256,
      "d_rope": 64}),
    ("fused_qk_rmsnorm_rope_gate", "prefill",
     {"B": 2, "s_query": 1024, "n_heads": 24, "n_kv_heads": 2,
      "d_head": 256, "d_rope": 64}),
)

def _gpu_cell_id(cell):
    written = cell[4] if len(cell) > 4 else {}
    return "-".join([cell[0], cell[1], *([written.get("output_format")
                                          or written.get("kv")]
                                         if written else [])])


@pytest.mark.parametrize("sig,phase,dims,params,written",
                         [(*c[:3], c[3] if len(c) > 3 else {},
                           c[4] if len(c) > 4 else {})
                          for c in _GPU_CELLS],
                         ids=[_gpu_cell_id(c) for c in _GPU_CELLS])
def test_the_framework_kernels_agree_with_the_oracle_on_a_gpu(sig, phase,
                                                             dims, params,
                                                             written):
    """The proof a provider is a measurement: the framework's own kernel,
    launched as the provider launches it, computes what the oracle does.
    Runs where a GPU and the framework are installed (the bench box);
    elsewhere there is nothing to run, and it says so."""
    import importlib
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from kernel_bench.driver import _cross_check, _xcheck_basis
    from kernel_bench.env import detect_env
    from kernel_bench.refs import get_ref
    env = detect_env()
    inv = Invocation(op=sig, signature=sig, phase=phase, dims=dict(dims),
                     dtype_env={"activation": "bf16", "weight": "bf16",
                                "kv": "bf16", "output": "bf16",
                                "metadata": "fp32", **written}, tp=1,
                     kernel_params=dict(params))

    def importable(p):
        try:
            importlib.import_module(p.import_name)
        except Exception:
            return False
        return True
    runnable = [p for p in _providers(sig) if p.library != "torch"
                and p.accepts(inv, env) and importable(p)]
    if not runnable:
        pytest.skip(f"no framework kernel for {sig} importable here")
    spec = get_ref(sig)
    inputs = build_inputs(spec, inv, "cuda")
    ref = spec.reference(inputs)
    for p in runnable:
        case = p.build(inv, env, p.convert_inputs(inv, inputs))
        case.to_canonical = p.wrap_output(inv, case.to_canonical)
        x_dtype, x_records = _xcheck_basis(inv, inputs)
        err, rel, cos, _typ, _wor = _cross_check(case, ref, x_dtype,
                                                 records=x_records)
        assert err is None, (p.kernel_impl, rel, cos)


class TestARelocatedKernel:
    """A vendored kernel whose module path moved between the releases a box
    may have installed — the frameworks' FLA copies did (vLLM v0.30, sglang
    v0.5.21) — is taken from the first location that exists."""

    def test_the_first_location_that_exists_wins(self):
        import sys
        import types
        from kernel_bench.providers.linear_ssm import _relocated
        mod = types.ModuleType("fzg_fake_new")
        mod.kernel = "new"
        old = types.ModuleType("fzg_fake_old")
        old.kernel = "old"
        try:
            sys.modules["fzg_fake_old"] = old
            assert _relocated("kernel", "fzg_fake_new", "fzg_fake_old") == "old"
            sys.modules["fzg_fake_new"] = mod
            assert _relocated("kernel", "fzg_fake_new", "fzg_fake_old") == "new"
        finally:
            sys.modules.pop("fzg_fake_new", None)
            sys.modules.pop("fzg_fake_old", None)

    def test_neither_location_is_an_import_error(self):
        from kernel_bench.providers.linear_ssm import _relocated
        with pytest.raises(ImportError, match="is in none of"):
            _relocated("kernel", "fzg_absent_a", "fzg_absent_b")

    def test_a_missing_dependency_is_not_a_missing_location(self):
        """A target module that imports something absent must raise, not look
        like a location that is not there — otherwise a broken install reads
        as an older release and the next candidate is tried instead."""
        import sys
        import types
        from kernel_bench.providers.linear_ssm import _relocated

        class Broken(types.ModuleType):
            def __getattr__(self, name):
                raise ModuleNotFoundError("No module named 'triton'",
                                          name="triton")
        try:
            sys.modules["fzg_broken"] = Broken("fzg_broken")
            with pytest.raises(ModuleNotFoundError, match="triton"):
                _relocated("kernel", "fzg_broken", "fzg_absent_b")
        finally:
            sys.modules.pop("fzg_broken", None)


class TestTheChunkProvidersHandTheStateOverTheWayServingDoes:
    """The recurrent state reaches the kernel in the shape that kernel's own
    serving call site uses, and WHICH shape that is comes from the callee.

    The two vendored copies diverged. sglang v0.5.22 takes the state POOL
    plus one slot index per sequence (`kernels/ops/attention/fla/chunk.py`
    :136-144, `kda.py:1274-1282`), and the index is not optional however the
    `= None` default reads: both entries forward it to a `_fwd` declared
    without a default (`chunk.py:44`, `kda.py:1155`) and the Triton kernel
    dereferences it above its own `USE_INITIAL_STATE` guards
    (`chunk_delta_h.py:126`). sglang's GDN backend always supplies it
    (`TritonGDNKernel.extend`, that class at
    `srt/layers/attention/linear/kernels/gdn_triton.py:40`, the pair built at
    `:183-184` and the call at `:203-215`). vLLM
    v0.32.0's copy has no such parameter and is handed a dense `[N, H, V, K]`
    its mixer gathered itself (`third_party/flash_linear_attention/ops/
    chunk.py:138-152`, `mamba/gdn/qwen_gdn_linear_attn.py:1599`).

    So three wrong implementations have to stay excluded, and each assertion
    below names the one it excludes:
      1. always hand over the dense state (what shipped, and what fails to
         compile against sglang v0.5.22),
      2. always hand over a pool plus indices — which `_accepted` would make
         SILENT rather than loud on vLLM and upstream fla: it filters the
         index keyword they do not declare, leaving them a state tensor one
         sequence too long whose row 0 is zeros, and their kernels index it
         by sequence directly, so every sequence would read its neighbour's
         state and no row could show it,
      3. index the pool from row 0 — sglang hands sequences slots 1..size and
         keeps row 0 as the padded-batch row (`srt/mem_cache/memory_pool.py`
         :398-400, `:407`),
      4. build the pool inside the timed call, which bills every repeat for
         an allocation a server does once at startup.
    """

    DIMS: ClassVar[dict] = {"B": 2, "s_query": 8, "n_lin_k_heads": 1,
                            "n_lin_v_heads": 1, "d_lin_k_head": 4,
                            "d_lin_v_head": 4}

    @staticmethod
    def _sglang_shaped(q, k, v, g, beta, scale=None, initial_state=None,
                       initial_state_indices=None, cu_seqlens=None,
                       head_first=False, use_qk_l2norm_in_kernel=False,
                       inplace_update=True):
        """sglang v0.5.22 `chunk_gated_delta_rule`'s signature (chunk.py
        :136-149), returning its `(o, None, h)` (`:269`)."""
        return ("o", None, "h")

    @staticmethod
    def _vllm_shaped(q, k, v, g, beta, scale=None, initial_state=None,
                     output_final_state=False, cu_seqlens=None,
                     chunk_indices=None, chunk_offsets=None,
                     use_qk_l2norm_in_kernel=False, core_attn_out=None):
        """vLLM v0.32.0's copy (third_party/flash_linear_attention/ops/
        chunk.py:138-152), returning its `(o, final_state)`."""
        return ("o", "final_state")

    def _handed_over(self, monkeypatch, callee, builder="_chunk_build"):
        """The kwargs `builder`'s timed call hands `callee`, and the call
        itself so a second invocation can be checked for fresh allocations.
        `_require_cuda` is patched out because the question is what the
        adapter passes, not what card runs it."""
        import torch

        from kernel_bench.providers import linear_ssm

        monkeypatch.setattr(linear_ssm, "_require_cuda", lambda: None)
        seen = []

        def record(**kwargs):
            seen.append(kwargs)
            return callee(**kwargs)

        record.__signature__ = __import__("inspect").signature(callee)
        d = self.DIMS
        shape4 = (d["B"], d["s_query"], d["n_lin_v_heads"], d["d_lin_v_head"])
        inputs = {
            "q": torch.zeros(d["B"], d["s_query"], d["n_lin_k_heads"],
                             d["d_lin_k_head"]),
            "k": torch.zeros(d["B"], d["s_query"], d["n_lin_k_heads"],
                             d["d_lin_k_head"]),
            "v": torch.zeros(shape4),
            "g": torch.zeros(shape4[:3]),
            "beta": torch.zeros(shape4[:3]),
            # Distinct per sequence, so a pool built off the wrong row cannot
            # pass by looking like zeros.
            "initial_state": torch.arange(
                float(d["B"] * d["n_lin_v_heads"] * d["d_lin_v_head"]
                      * d["d_lin_k_head"])).reshape(
                d["B"], d["n_lin_v_heads"], d["d_lin_v_head"],
                d["d_lin_k_head"]) + 1.0,
        }
        inv = Invocation(op="gated_delta_rule", signature="gated_delta_rule",
                         phase="prefill", dims=dict(d),
                         dtype_env={"activation": "bf16", "weight": "bf16",
                                    "kv": "bf16", "output": "bf16",
                                    "metadata": "fp32"}, tp=1)
        case = getattr(linear_ssm, builder)(lambda: record)(inv, TORCH_ENV,
                                                            inputs)
        case.run()
        return seen, case, inputs["initial_state"]

    def test_a_callee_that_declares_slot_indices_gets_the_pool_and_the_slots(
            self, monkeypatch):
        """Excludes wrong implementation 1 (always dense) and 3 (row 0)."""
        import torch
        seen, _case, state = self._handed_over(monkeypatch,
                                               self._sglang_shaped)
        (kwargs,) = seen
        assert "initial_state_indices" in kwargs, \
            "sglang's kernel dereferences this index unconditionally"
        assert kwargs["initial_state_indices"].tolist() == [1, 2]
        pool = kwargs["initial_state"]
        assert pool.shape == (3, 1, 4, 4)
        # Row 0 is sglang's padded-batch row; the sequences are the rows the
        # indices name, and they carry the cell's state unchanged.
        assert float(pool[0].abs().sum()) == 0.0
        assert torch.equal(pool[1:], state.float())

    def test_a_callee_without_that_parameter_gets_the_dense_state(
            self, monkeypatch):
        """Excludes wrong implementation 2 (always pool + indices). vLLM and
        upstream fla declare no index parameter, so `_accepted` would drop it
        and leave them a pool where they expect the dense state — not an
        error, a wrong answer. The state must arrive as the cell declares it,
        `[B, HV, V, K]`, row per sequence."""
        import torch
        seen, _case, state = self._handed_over(monkeypatch, self._vllm_shaped)
        (kwargs,) = seen
        assert "initial_state_indices" not in kwargs
        assert torch.equal(kwargs["initial_state"], state.float())

    def test_the_pool_is_built_once_not_per_timed_call(self, monkeypatch):
        """Excludes wrong implementation 4: resolving the state inside the
        timed lambda would allocate `[B+1, HV, V, K]` and copy into it on
        every repeat, and the row would price that, not the kernel."""
        seen, case, _state = self._handed_over(monkeypatch,
                                               self._sglang_shaped)
        case.run()
        case.run()
        assert len(seen) == 3
        first = seen[0]
        for later in seen[1:]:
            assert later["initial_state"] is first["initial_state"]
            assert later["initial_state_indices"] is \
                first["initial_state_indices"]

    def test_the_output_is_taken_whole_when_the_callee_returns_one_tensor(
            self, monkeypatch):
        """sglang v0.5.22's `chunk_kda` returns a bare `o`, not a tuple
        (`kernels/ops/attention/fla/kda.py:1274` → `chunk_kda_fwd`'s
        `return o` at `kda.py:1271`; the `return o, h` at `:1269` is the
        `output_intermediate_states` branch, which this build does not ask
        for), where `chunk_gated_delta_rule` beside it returns `(o, None, h)`
        (`chunk.py:269`). Indexing `[0]` unconditionally takes a batch ROW off
        the bare one — a `[T, H, V]` where the cell declares `[B, T, H, V]`.
        """
        import torch

        from kernel_bench.providers import linear_ssm

        monkeypatch.setattr(linear_ssm, "_require_cuda", lambda: None)
        out = torch.arange(24.0).reshape(2, 3, 2, 2)

        def bare(q, k, v, g, beta, scale=None, initial_state=None,
                 initial_state_indices=None, use_qk_l2norm_in_kernel=False):
            return out

        _seen, case, _state = self._handed_over(monkeypatch,
                                                self._vllm_shaped)
        assert case.run() == "o"            # a tuple: the first member
        _seen2, case2, _ = self._handed_over(monkeypatch, bare)
        assert case2.run() is out           # one tensor: the tensor itself


class TestGlaRecurrenceReadsAndWritesItsState:
    """`gla_simple_recurrence` (Ling 2.x's linear-attention layers) is billed
    for the per-sequence fp32 state `[B, H, d, d]` — on Ling at tp=64 a
    decode cell's numerator is ~128 MB of state beside ~2 MB of q/k/v/o — and
    every launch serving runs reads that state and writes it back. The rows
    deleted from the lake timed launches that never touched it: FLA called
    with no `initial_state` and `output_final_state=False`, the torch floor
    scanning from zeros.

    So: the cell carries a state of the cell's shape, the oracle's answer
    depends on it (and on the per-head decay serving passes), and every
    provider's build hands the state to its kernel, as the slot pool the
    framework keeps where it keeps one, and gets the final state written
    back. The framework kernels are stood in for by fakes that run the
    recurrence from exactly what they are handed, so a build that drops,
    misplaces or zeroes the state gives the wrong answer here without a GPU.
    """

    SIG = "gla_simple_recurrence"
    DIMS: ClassVar[dict] = {"B": 3, "s_query": 2, "n_heads": 2, "d_head": 4}

    def _inv(self, phase="prefill", act="bf16", **dims):
        return Invocation(op=self.SIG, signature=self.SIG, phase=phase,
                          dims={**self.DIMS, **dims},
                          dtype_env={"activation": act, "kv": "fp32",
                                     "output": act}, tp=1)

    def _inputs(self, inv):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        return build_inputs(get_ref(self.SIG), inv)

    def test_the_cell_carries_a_state_of_its_own_shape(self):
        import torch
        from backend.inference.op_signatures import get_signature
        layouts = {t.name: t.axes for t in get_signature(self.SIG).inputs}
        assert layouts["initial_state"] == ("B", "n_heads", "d_head",
                                            "d_head")
        assert layouts["slope"] == ("n_heads",)
        x = self._inputs(self._inv())
        assert tuple(x["initial_state"].shape) == (3, 2, 4, 4)
        assert x["initial_state"].dtype == torch.float32
        # A zero state would let a kernel that never reads it pass.
        assert float(x["initial_state"].abs().sum()) > 0
        assert bool((x["slope"] > 0).all())

    def test_the_engine_tensor_is_read_heads_first(self):
        """The op lays q/k/v out `[B, H, s, d]` (`head_split`'s order). Read
        as `[B, s, H, d]`, Ling's tp=1 decode step — 64 heads, one token —
        was keyed and timed as one head over 64 tokens."""
        from backend.inference.op_signatures import (get_signature,
                                                     zip_engine_layouts)
        engine = [[4, 64, 1, 128]] * 3
        assert zip_engine_layouts(get_signature(self.SIG).engine_inputs,
                                  engine) == {"B": 4, "n_heads": 64,
                                              "s_query": 1, "d_head": 128}

    def test_the_oracle_starts_from_the_state_and_decays_it(self):
        import torch
        from kernel_bench.refs import get_ref
        ref = get_ref(self.SIG).reference
        x = self._inputs(self._inv())
        out = ref(x)
        assert not torch.allclose(
            out, ref(dict(x, initial_state=torch.zeros_like(
                x["initial_state"]))))
        assert not torch.allclose(
            out, ref(dict(x, slope=torch.zeros_like(x["slope"]))))
        # One step by hand: S1 = exp(-slope)·S0 + k⊗v, o = (q/√d)·S1.
        q, k, v = (x[n][:, 0].float() for n in ("q", "k", "v"))
        S1 = ((-x["slope"]).exp()[None, :, None, None] * x["initial_state"]
              + k[..., :, None] * v[..., None, :])
        o0 = torch.einsum("bhk,bhkv->bhv", q * 4 ** -0.5, S1)
        assert torch.allclose(out[:, 0], o0, rtol=1e-5, atol=1e-5)

    # -- the fakes: each runs the recurrence from what its call site hands it

    @staticmethod
    def _scan_slots(q, k, v, pool, slots, slope, scale):
        """[B, T, H, D] operands against a slot pool: read each sequence's
        state from its slot, write the final one back — what the serving
        kernels do with `kv_cache` / `s`."""
        import torch
        from kernel_bench.refs import _gla_scan
        idx = slots.long()
        final = torch.empty_like(pool[idx])
        D = q.shape[-1]
        out = _gla_scan(q.float() * scale * D ** 0.5, k, v, slope,
                        pool[idx], final_state=final)
        pool[idx] = final
        return out

    def _fake(self, monkeypatch, name, **attrs):
        """Install module `name` (and empty parents) carrying `attrs`."""
        import sys
        import types
        parts = name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                monkeypatch.setitem(sys.modules, parent,
                                    types.ModuleType(parent))
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)

    def _check(self, monkeypatch, build, inv, pool_of):
        """Run `build` on the cell, then require: the canonical output is
        the oracle's (so the state reached the kernel), the kernel's state
        holds the oracle's final state (so it was written back), and the
        cell's own inputs are untouched (other providers share them)."""
        import torch
        from kernel_bench.providers import linear_ssm
        from kernel_bench.refs import _gla_scan, get_ref
        monkeypatch.setattr(linear_ssm, "_require_cuda", lambda: None)
        x = self._inputs(inv)
        before = x["initial_state"].clone()
        case = build(inv, TORCH_ENV, x)
        got = case.to_canonical(case.run())
        want = get_ref(self.SIG).reference(x)
        assert torch.allclose(got, want, rtol=2e-2, atol=2e-2)
        final = torch.empty_like(before)
        _gla_scan(x["q"], x["k"], x["v"], x["slope"], before,
                  final_state=final)
        assert torch.allclose(pool_of(), final, rtol=2e-2, atol=2e-2)
        assert torch.equal(x["initial_state"], before)

    def test_fla_is_handed_the_state_and_asked_for_the_final_one(
            self, monkeypatch):
        import inspect
        from kernel_bench.providers import linear_ssm
        seen = {}

        def fla_shaped(q, k, v, g=None, g_gamma=None, scale=None,
                       initial_state=None, output_final_state=False,
                       cu_seqlens=None, **kwargs):
            """fla 0.5.2 `chunk_simple_gla` / `fused_recurrent_simple_gla`
            (simple_gla/chunk.py:238-251), state `[N, H, K, V]`."""
            import torch
            from kernel_bench.refs import _gla_scan
            assert g is None, "the decay is head-wise: g_gamma, not g"
            if initial_state is None:
                initial_state = torch.zeros(q.shape[0], q.shape[2],
                                            q.shape[3], v.shape[3])
            ht = torch.empty_like(initial_state)
            o = _gla_scan(q, k, v, -g_gamma, initial_state, final_state=ht)
            seen["ht"] = ht if output_final_state else initial_state * 0
            return o, (ht if output_final_state else None)

        for entry_phase in ("prefill", "decode"):
            inv = self._inv(entry_phase,
                            **({"s_query": 1} if entry_phase == "decode"
                               else {}))
            self._check(monkeypatch,
                        linear_ssm._fla_gla_build(lambda: fla_shaped), inv,
                        lambda: seen["ht"])
        assert "g_gamma" in inspect.signature(fla_shaped).parameters

    def test_vllm_decode_reads_and_rewrites_the_slot_pool(self, monkeypatch):
        from kernel_bench.providers import linear_ssm
        seen = {}

        def linear_attention_decode(q, k, v, kv_cache, slope_rate,
                                    state_indices_tensor, q_start=0,
                                    q_end=None, slot_start=0, slot_end=None,
                                    block_size=32):
            """vLLM v0.32.0 minimax_linear_attn.py:49-68: token-major
            `[B, H, D]` q already scaled by 1/√d, state by slot in place."""
            seen["pool"], seen["slots"] = kv_cache, state_indices_tensor
            B, H, D = q.shape
            o = self._scan_slots(q[:, None], k[:, None], v[:, None],
                                 kv_cache, state_indices_tensor, slope_rate,
                                 scale=1.0)
            return o.reshape(B, H * D)

        self._fake(monkeypatch,
                   "vllm.model_executor.layers.mamba.linear."
                   "minimax_linear_attn",
                   linear_attention_decode=linear_attention_decode)
        inv = self._inv("decode", s_query=1)
        self._check(monkeypatch, linear_ssm._vllm_gla_decode_build, inv,
                    lambda: seen["pool"][1:])
        assert seen["slots"].tolist() == [1, 2, 3]

    def test_vllm_prefill_reads_and_rewrites_the_slot_pool(self, monkeypatch):
        from kernel_bench.providers import linear_ssm
        seen = {}
        prefix = object()

        def linear_attention_prefill_and_mix(q, k, v, kv_cache,
                                             state_indices_tensor,
                                             attn_metadata, slope_rate,
                                             block_size, decode_fn, prefix_fn,
                                             layer_idx=None):
            """vLLM v0.32.0 bailing_linear_attn.py:85: flat `[N·T, H, D]`
            prefills, q already scaled, each from its slot's state."""
            assert prefix_fn is prefix
            assert attn_metadata.num_decode_tokens == 0
            seen["pool"] = kv_cache
            B = attn_metadata.num_prefills
            T = q.shape[0] // B
            o = self._scan_slots(
                *(x.reshape(B, T, *x.shape[1:]) for x in (q, k, v)),
                kv_cache, state_indices_tensor, slope_rate, scale=1.0)
            return o.reshape(B * T, -1)

        class MiniMaxText01LinearKernel:
            jit_linear_forward_prefix = prefix

        self._fake(monkeypatch,
                   "vllm.model_executor.layers.mamba.linear."
                   "bailing_linear_attn",
                   linear_attention_prefill_and_mix=(
                       linear_attention_prefill_and_mix))
        self._fake(monkeypatch,
                   "vllm.model_executor.layers.mamba.linear."
                   "minimax_linear_attn",
                   MiniMaxText01LinearKernel=MiniMaxText01LinearKernel)
        self._check(monkeypatch, linear_ssm._vllm_gla_prefill_build,
                    self._inv("prefill", act="fp32"),
                    lambda: seen["pool"][1:])

    def test_sglang_seg_la_reads_and_rewrites_the_slot_pool(self,
                                                             monkeypatch):
        import types

        from kernel_bench.providers import linear_ssm
        seen = {}

        def seg_la_fwd(q, k, v, s, decay_scales, meta, caches=None,
                       cache_indices=None, track_lens=None,
                       track_state_indices=None, softmax_scale=None,
                       decouple=False):
            """sglang v0.5.22 seg_la.py:682: flat `[N·T, H, D]`, the state
            by `s_offsets`, read where `s_scales` says, written back."""
            assert bool(meta.s_scales.all()), "every sequence has a state"
            seen["pool"] = s
            B = meta.batch_size
            T = q.shape[0] // B
            o = self._scan_slots(
                *(x.reshape(B, T, *x.shape[1:]) for x in (q, k, v)),
                s, meta.s_offsets, decay_scales,
                scale=softmax_scale)
            return o.reshape(B * T, *q.shape[1:])

        self._fake(monkeypatch, "sglang.kernels.ops.attention.linear.seg_la",
                   seg_la_fwd=seg_la_fwd,
                   SegLaMeta=types.SimpleNamespace)
        for phase, s in (("prefill", 2), ("decode", 1)):
            self._check(monkeypatch, linear_ssm._sglang_seg_la_build,
                        self._inv(phase, s_query=s),
                        lambda: seen["pool"][1:])

    def test_the_torch_floor_scans_from_the_state_and_writes_it_back(
            self, monkeypatch):
        """The floor runs on CPU: no fake needed. Its state is its own copy,
        written back in place; the copy is the BenchCase's closure, so the
        check reads it through the scan's write rather than a handle."""
        import torch

        from kernel_bench.providers import linear_ssm
        from kernel_bench.refs import get_ref
        x = self._inputs(self._inv())
        before = x["initial_state"].clone()
        case = linear_ssm._torch_gla_build(self._inv(), TORCH_ENV, x)
        got = case.to_canonical(case.run())
        assert torch.allclose(got, get_ref(self.SIG).reference(x),
                              rtol=1e-5, atol=1e-5)
        assert torch.equal(x["initial_state"], before)
        # A second run starts from the state the first wrote back.
        again = case.to_canonical(case.run())
        assert not torch.allclose(got, again)


def test_sglangs_fused_projection_conv_is_a_named_gap():
    """sglang v0.5.22 runs a Qwen3.5-class GDN decode conv as a WIDER launch
    than this span, and nothing measures that launch.

    `srt/models/qwen3_5.py:1008-1023` hands the backend `(projected_qkvz,
    projected_ba)` whenever `_gdn_decode_fused_proj_conv` holds — CUDA and
    `SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV`, `EnvBool(True)` at
    `srt/environ.py:1873` — and `gdn_backend.py:723-801` then runs
    `fused_qkvzba_causal_conv1d_update_contiguous`
    (`kernels/ops/attention/triton_gdn_fused_proj.py:541`), one launch that
    unpacks the projection, convolves, and writes `mixed_qkv`, `z`, `b`, `a`
    and the conv state. Different bytes and fewer launches than
    conv1d → silu, so it is a different cell, and no plan draws it.

    The gap is declared on the kernel that is NOT that route rather than
    narrowed away: the cell cannot tell a Qwen3.5 decode conv from a KDA one
    (same batch, channels and width), and the models that do run this kernel
    — KDA, Mamba-2, and the GDN mixers that hand an unpacked tensor — would
    otherwise lose a correct measurement. When a plan takes the wider
    boundary, the token comes off here and this test says so.
    """
    sgl = next(p for p in _providers("causal_conv1d")
               if p.kernel_impl == "sglang_causal_conv1d_update")
    assert sgl.serves.unsupported_features == (
        "sglang_gdn_fused_projection_conv",)
    assert sgl.capabilities()["unsupported_features"] == [
        "sglang_gdn_fused_projection_conv"]
    # A runtime route, not a shape: it must not read as a geometry limit.
    assert not any("fused_projection" in c
                   for c in sgl.capabilities()["shape_constraints"])
    # Nothing claims the wider launch yet — no provider, and no span of it.
    from backend.inference.op_signatures import op_bindings
    assert not [p for p in all_providers()
                if "fused_proj" in p.kernel_impl or "qkvzba" in p.kernel_impl]
    assert not [op for op in op_bindings() if "qkvzba" in op]


class TestHyperConnectionSeam:
    """The hyper-connection sublayer boundary, one plan per launch
    (`fusion_plans/mhc_{post,pre_gemm,pre_big_fuse,fused_post_pre}.yaml`).

    A manifold-hyper-connection model keeps `n_hidden_copies` parallel
    residual copies and mixes them at every sublayer boundary; both
    frameworks run that boundary as two or three TileLang launches, and each
    launch is one span. The properties asserted here are the seam's, not any
    model's: the cell is the residual's own geometry, the mix vector is
    `hc² + 2·hc` wide by construction, and each oracle is the launch's
    function recomputed from its definition.
    """

    SPANS: ClassVar[tuple] = ("mhc_post", "mhc_pre_gemm", "mhc_pre_big_fuse",
                              "mhc_fused_post_pre")
    #: A cell small enough to recompute by hand, stated as the seam's axes.
    DIMS: ClassVar[dict] = {"B": 2, "s_query": 3, "n_hidden_copies": 4,
                            "d_model": 128}

    def test_each_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        for sig in self.SPANS:
            assert bound[sig] == sig
            ps = _providers(sig)
            libs = {p.library for p in ps}
            # Both frameworks run every launch of the seam, and a torch floor
            # runs the naive chain the saving is measured against. A SUBSET,
            # so another hardware family may serve the seam too — vLLM-Ascend
            # does for `mhc_post` — while the three that must not quietly
            # drop out are still named.
            assert {"vllm", "sglang", "torch"}.issubset(libs), (sig, libs)
            assert all(p.fused_op == sig for p in ps), sig
            # The CUDA wrappers are audited at bf16 and assert it; the floor
            # takes every float width, because it runs the chain at whatever
            # width the cell states. A kernel of neither kind states its own
            # card's widths (the Ascend cube runs bf16 and fp16 at one rate,
            # `providers/ascend._ASCEND_FLOAT`), which is that card's fact
            # and not a width this seam gets to pick.
            for p in ps:
                if p.library == "torch":
                    assert p.serves.activation == TORCH_FLOAT, p.kernel_impl
                elif p.library in ("vllm", "sglang"):
                    assert p.serves.activation == ("bf16",), p.kernel_impl
                else:
                    assert "bf16" in p.serves.activation, p.kernel_impl

    def test_the_projection_names_which_of_its_two_kernels_ran(self):
        """vLLM's projection wrapper has TWO kernels behind it and picks by
        package availability (`use_deep_gemm = is_deep_gemm_supported() or
        not use_tilelang_fallback`). One provider through it would record one
        `kernel_impl` for either, so a row from a box with DeepGEMM and a row
        from a box without would share an identity and be silently
        incomparable. Two providers, each naming its kernel.

        And there is no bare `deep_gemm` arm: one existed, passing
        `num_split=1`, which is not a schedule any framework launches — so
        its rows timed an under-parallel reduction. DeepGEMM is reachable for
        this span only as vLLM's own choice inside that wrapper, which is
        also what keeps the split count the framework's rather than ours.
        """
        impls = {p.kernel_impl for p in _providers("mhc_pre_gemm")}
        assert impls == {"vllm_mhc_pre_gemm_deepgemm",
                         "vllm_mhc_pre_gemm_tilelang",
                         "sglang_mhc_pre_gemm", "torch_mhc_pre_gemm"}, impls
        assert not [p for p in _providers("mhc_pre_gemm")
                    if p.library == "deepgemm"], "no bare DeepGEMM arm"

    def test_deepinfer_bills_the_shifted_seam_as_one_mega_launch(self):
        """V4.1 in DeepInfer does not run the two-framework small/large path:
        `_mega_mhc` performs the previous post, the current projection, and
        the carried-pre reduce + norm in one DeepGEMM launch.

        The split stays outside the plan (a stated residual, as above), but
        its `mixes` operand is an exposed port; the residual survives for the
        sublayer's own closing post-mix. The floor is held to the same
        precision invariant as the other seam spans.
        """
        from backend.inference.fusion_plans import FUSION_SCHEMES
        from backend.inference.op_signatures import op_bindings

        sig = "mhc_mega_post_pre"
        assert op_bindings()[sig] == sig
        ps = _providers(sig)
        assert {p.library for p in ps} == {"deepinfer", "torch"}
        assert all(p.fused_op == sig for p in ps)
        pattern = FUSION_SCHEMES[sig].naive_pattern
        assert {"mixes_out", "residual_out"} <= set(pattern.io_mapping)

        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref(sig)
        inv = _inv(sig, self.DIMS)
        inputs = build_inputs(spec, inv, "cpu")
        assert inputs["pre"].shape == (6, 4)
        floor = next(p for p in ps if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        got = case.to_canonical(case.run())
        want = spec.reference(inputs)
        assert got.shape == want.shape
        assert got.dtype == torch.float32
        rel = float((got - want).abs().max() / want.abs().max())
        assert 1e-3 <= rel <= 2e-2, rel

    def test_the_mix_width_follows_the_copies_and_is_not_its_own_axis(self):
        """The pre-mix projection writes `hc² + 2·hc` numbers per token — the
        cross-mix matrix plus the `pre` and `post` gates. It is a function of
        the copy count, so the contract states it as one and the cell carries
        no separate mix axis; a span whose operands do not carry the mix
        width (the big fuse reads `pre` and the residual) could not bind one.
        """
        from kernel_bench.refs import get_ref
        for hc in (2, 4, 6):
            dims = dict(self.DIMS, n_hidden_copies=hc)
            inputs = build_inputs(get_ref("mhc_pre_gemm"),
                                  _inv("mhc_pre_gemm", dims), "cpu")
            assert inputs["fn"].shape == (hc * hc + 2 * hc,
                                          hc * dims["d_model"]), hc
        # and the cell is the residual's geometry — no mix axis of its own,
        # which the big fuse (whose operands are `pre` and the residual)
        # could not bind.
        from backend.inference.op_signatures import get_signature
        for sig in self.SPANS:
            assert set(get_signature(sig).cell_dims) == {
                "B", "s_query", "n_hidden_copies", "d_model"}, sig

    def test_the_post_mix_oracle_spreads_the_sublayer_and_cross_mixes(self):
        """`y[j,h] = Σᵢ comb[i,j]·residual[i,h] + post[j]·sublayer[h]`,
        recomputed with an explicit loop over the copies rather than the
        einsum the reference uses — so the test cannot pass by sharing the
        reference's contraction. Note which index `comb` sums over: the
        kernels contract its FIRST copy index.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("mhc_post")
        inv = _inv("mhc_post", self.DIMS)
        inputs = build_inputs(spec, inv, "cpu")
        hc, d = self.DIMS["n_hidden_copies"], self.DIMS["d_model"]
        comb, res = inputs["comb"].float(), inputs["residual"].float()
        post, sub = inputs["post"].float(), inputs["sublayer"].float()
        want = torch.zeros(comb.shape[0], hc, d)
        for j in range(hc):
            for i in range(hc):
                want[:, j, :] += comb[:, i, j, None] * res[:, i, :]
            want[:, j, :] += post[:, j, None] * sub
        got = spec.reference(inputs)
        assert got.shape == (6, hc, d)
        assert torch.allclose(got, want, atol=1e-5)
        # The floor runs the same chain at the CELL's precision (an fp32
        # floor would report another precision's latency), so it is checked
        # at that width: one bf16 ULP is 2^-8 relative.
        floor = next(p for p in _providers("mhc_post") if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        got_floor = case.to_canonical(case.run())
        assert got_floor.dtype == torch.float32          # canonical is fp32
        assert ((got_floor - want).abs().max()
                <= 2e-2 * want.abs().max()), "floor is the chain in bf16"

    def test_sglangs_post_runs_inside_a_published_serving_config(
            self, monkeypatch):
        """sglang's `mhc_post` wrapper reads the parallel layout
        (`is_dsa_prefill_cp_interleave` → `get_parallel().attn_cp_size`),
        which raises "config namespace 'parallel' not published" in a
        process that never served — every canary on the H100 died of it.
        The build must run the call inside a default serving context, and
        leave the process's own config untouched outside the call."""
        import contextlib
        import sys
        import types
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.mhc as mhc
        from kernel_bench.refs import get_ref
        active = []

        def mhc_post(x, residual, post, comb):
            if not active:
                raise ValueError("config namespace 'parallel' not published")
            return torch.empty_like(residual)

        @contextlib.contextmanager
        def use_context(context):
            active.append(context)
            try:
                yield context
            finally:
                active.pop()

        modules = {
            "sglang.kernels.ops.layernorm.mhc": dict(mhc_post=mhc_post),
            "sglang.srt.runtime_context": dict(
                use_context=use_context,
                create_context=lambda args, **kw: ("context", args, kw),
                SpawnRanks=lambda **kw: kw),
            "sglang.srt.server_args": dict(ServerArgs=lambda **kw: kw),
        }
        for name, funcs in modules.items():
            mod = types.ModuleType(name)
            for key, value in funcs.items():
                setattr(mod, key, value)
            monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(mhc, "_require_cuda", lambda: None)
        monkeypatch.setattr(mhc, "_SGL_CONTEXT", None, raising=False)
        inv = _inv("mhc_post", self.DIMS)
        inputs = build_inputs(get_ref("mhc_post"), inv, "cpu")
        case = mhc._sgl_post_build(inv, TORCH_ENV, inputs)
        assert case.run().shape == inputs["residual"].shape
        assert active == []                 # scoped to the call

    def test_the_projection_oracle_scales_by_the_residuals_own_rms(self):
        """`mixes = fn @ flatten(residual)`, scaled by the residual's RMS.

        The kernels write the unnormalised dots plus a per-token sum of
        squares and the following big fuse applies the rsqrt; scaling a row
        before or after the projection is the same function, and the span
        declares its output in the normalised form. Checked as that identity:
        normalising the input first must give the same answer.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("mhc_pre_gemm")
        inv = _inv("mhc_pre_gemm", self.DIMS)
        inputs = build_inputs(spec, inv, "cpu")
        flat = inputs["residual"].float().flatten(1)
        rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        want = (flat * rms) @ inputs["fn"].float().t()
        got = spec.reference(inputs)
        assert got.shape == (6, 4 * 4 + 2 * 4)
        assert torch.allclose(got, want, atol=1e-4)

    def test_the_epilogue_oracle_gates_the_mix_and_norms_the_input(self):
        """The big fuse's three writes: the `post` gate in [0, 2], the
        cross-mix doubly normalised by its Sinkhorn rounds, and the sublayer
        input, RMS-normalised so its rows have unit root-mean-square.

        Asserted as those properties rather than against a second copy of the
        arithmetic: a gate outside its range, a cross-mix whose rows do not
        sum to one, or an un-normalised input are what going wrong looks
        like here.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("mhc_pre_big_fuse")
        inv = _inv("mhc_pre_big_fuse", self.DIMS)
        inputs = build_inputs(spec, inv, "cpu")
        hc, d = self.DIMS["n_hidden_copies"], self.DIMS["d_model"]
        got = spec.reference(inputs)
        # `cat([layer_input, post, comb.flatten(1)], -1)`, the launch's three
        # writes side by side so the one-tensor cross-check covers all three.
        assert got.shape == (6, d + hc + hc * hc)
        layer_input, post, comb = (got[:, :d], got[:, d:d + hc],
                                   got[:, d + hc:].view(-1, hc, hc))
        # DOUBLY stochastic — rows AND columns. The column sum is the
        # discriminating half: one-sided row normalisation satisfies the row
        # sum too, so asserting only that would pass against the function
        # this repo held for `hc_split_sinkhorn` before it was corrected.
        # Alternating the two axes is what Sinkhorn means and the only
        # reason both sums are one.
        assert torch.allclose(comb.sum(-1), torch.ones(6, hc), atol=5e-3)
        assert torch.allclose(comb.sum(-2), torch.ones(6, hc), atol=5e-3)
        # The gates are non-degenerate. `0 <= post <= 2` is NOT asserted: it
        # holds for `2*sigmoid` of anything, correct or not, so it cannot
        # fail and would be decoration. That the gate VARIES across tokens
        # can fail — it is what a gate collapsed to a constant, or read off
        # the wrong slice of a zeroed mix vector, would show.
        assert post.std() > 1e-3, "post gate is constant across tokens"
        assert comb.std() > 1e-3, "cross-mix is constant"
        # The sublayer input went through an RMSNorm, which is invariant to
        # the scale of what it normalises: multiply the residual by a
        # constant and that write is unchanged. Asserted as the invariance
        # rather than by dividing the gain back out, which a gain near zero
        # turns into noise.
        # The projection's own writes scale with the residual, so they move
        # with it: the gates then see the same mix vector and only the
        # reduced row is four times larger, which the norm divides back out.
        scaled = dict(inputs, residual=inputs["residual"] * 4.0,
                      mixes=inputs["mixes"] * 4.0,
                      sqrsum=inputs["sqrsum"] * 16.0)
        again = spec.reference(scaled)
        assert torch.allclose(again[:, :d], layer_input, atol=2e-2, rtol=2e-2)

    def test_the_fused_span_is_the_post_mix_then_the_projection(self):
        """One launch for a boundary's post-mix and the next pre-mix
        projection. Its oracle must be exactly the two spans' oracles
        composed — that composition IS the span — so it is checked against
        them rather than against a third copy of the arithmetic.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref, hc_normalised_mixes, hc_post_mix
        spec = get_ref("mhc_fused_post_pre")
        inv = _inv("mhc_fused_post_pre", self.DIMS)
        inputs = build_inputs(spec, inv, "cpu")
        hc, d = self.DIMS["n_hidden_copies"], self.DIMS["d_model"]
        got = spec.reference(inputs)
        assert got.shape == (6, hc * hc + 2 * hc + hc * d)
        # the span's two ports: the mix vector and the new residual
        residual_out = hc_post_mix(inputs).to(inputs["residual"].dtype)
        assert torch.allclose(got[:, hc * hc + 2 * hc:],
                              residual_out.flatten(1).float(), atol=1e-5)
        mixes = hc_normalised_mixes(residual_out.reshape(6, -1),
                                    inputs["fn"], 1e-6)
        assert torch.allclose(got[:, :hc * hc + 2 * hc], mixes, atol=1e-4)

    @pytest.mark.parametrize("sig", SPANS)
    def test_every_launch_has_a_floor_running_its_chain_at_the_cells_width(
            self, sig):
        """Each of the four launches has a torch floor, and each one runs its
        span's naive chain AT THE CELL'S PRECISION.

        The floor is the denominator of the span's saving, so a span whose
        floor does not run has no measured saving even where its kernel does.
        The precision is the part that breaks quietly: the oracle computes in
        fp32 on purpose, to be a trusted reference, and a floor that simply
        calls the oracle reports fp32 latency for a bf16 cell — which the
        driver refuses ("the launch's largest operand is float32 where the
        cell declares bfloat16"), so the span loses its floor and keeps its
        kernel.

        Held two-sidedly for that reason: close enough to the oracle to be
        the same arithmetic, and far enough from it to prove the cast
        happened. `mhc_post` is checked in more detail above, by hand; this
        is the breadth pass that keeps the other three from going quiet.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        phase = "decode" if sig == "mhc_fused_post_pre" else "prefill"
        inv = _inv(sig, self.DIMS, phase=phase)
        spec = get_ref(sig)
        inputs = build_inputs(spec, inv, "cpu")
        floor = next(p for p in _providers(sig) if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        got = case.to_canonical(case.run())
        want = spec.reference(inputs)
        from kernel_bench.providers.attention import _torch_dtype
        assert got.shape == want.shape, sig
        assert got.dtype == torch.float32, sig       # canonical is fp32
        rel = float((got - want).abs().max() / want.abs().max())
        # Two-sided on purpose, and this is the whole point of the test. One
        # bf16 ULP is 2^-8 ≈ 3.9e-3 relative, so the floor running the chain
        # at the cell's width lands a few ULP from the fp32 oracle — near
        # enough to be the same arithmetic. The LOWER bound is what catches
        # the regression: a floor that just called the oracle would agree to
        # fp32 rounding (~1e-7) and sail through an upper bound alone, while
        # reporting fp32 latency for a bf16 cell.
        assert 1e-3 <= rel <= 2e-2, (sig, rel)
        # Same fact read off the output: the launch's activation-valued write
        # carries the cell's dtype. (The epilogue also writes its two gates,
        # which are fp32 in both frameworks' runtime — so this asks that one
        # of the writes is at the cell's width, not that all of them are.)
        raw = case.run()
        widths = {t.dtype for t in (raw if isinstance(raw, tuple) else (raw,))
                  if hasattr(t, "dtype")}
        assert _torch_dtype(inv.dtype) in widths, (sig, widths)

    @pytest.mark.regression          # walks three models in both phases
    def test_the_plan_reaches_every_launch_of_the_seam(self):
        """The cell is the residual's geometry and nothing else, so two
        models with the same copies and width share one measurement — which
        is correct, it is one physical launch.
        """
        expect = {"deepseek-ai/DeepSeek-V4-Flash": 4096,
                  "deepseek-ai/DeepSeek-V4-Pro": 7168,
                  "zai-org/GLM-5.3-Flash": 4096}
        for model, d_model in expect.items():
            for phase in ("prefill", "decode"):
                jobs, _ = _plan(model, phase, "vllm")
                for sig in ("mhc_post", "mhc_pre_gemm", "mhc_pre_big_fuse"):
                    cells = {(j["dims"]["n_hidden_copies"],
                              j["dims"]["d_model"])
                             for j in jobs if j["signature"] == sig}
                    assert cells == {(4, d_model)}, (model, phase, sig, cells)

    def test_v41s_shifted_schedule_is_the_same_spans_over_a_flat_stream(self):
        """DeepSeek-V4.1 runs the seam on the delayed schedule: the same
        launches, but its mix stats read the already-flattened stream, so
        the two-node `mhc_pre_gemm_stream` plan bills the projection under
        the `mhc_pre_gemm` signature — and the epilogue is V4.1's own kernel
        under the `mhc_pre_big_fuse` span, which the two V4.1 arms name.
        """
        from backend.inference.op_signatures import op_bindings
        # One signature for both spellings of the projection — the op names
        # the graph spelling, the span is the launch's.
        assert op_bindings()["mhc_pre_gemm_stream"] == "mhc_pre_gemm"
        impls = {p.kernel_impl for p in _providers("mhc_pre_big_fuse")}
        assert {"vllm_mhc_pre_norm", "sglang_hc_combine_norm"} <= impls

    def test_v41s_epilogue_arms_declare_their_geometry(self):
        """sglang's `hc_combine_norm` is compiled for V4.1's exact seam
        (4 copies of 5120); its arm serves that cell and declines the
        seam's other cells rather than asserting inside the kernel. The
        delayed vLLM epilogue takes the carried pre at any cell, as the
        kernel does."""
        vllm_arm = next(p for p in _providers("mhc_pre_big_fuse")
                        if p.kernel_impl == "vllm_mhc_pre_norm")
        sgl_arm = next(p for p in _providers("mhc_pre_big_fuse")
                       if p.kernel_impl == "sglang_hc_combine_norm")
        v41 = _inv("mhc_pre_big_fuse", dict(self.DIMS, d_model=5120))
        assert vllm_arm.accepts(v41, TORCH_ENV)
        assert sgl_arm.accepts(v41, TORCH_ENV)
        other = _inv("mhc_pre_big_fuse", self.DIMS)
        assert vllm_arm.accepts(other, TORCH_ENV)
        assert not sgl_arm.accepts(other, TORCH_ENV)

    @pytest.mark.regression          # walks one model in both phases, twice
    def test_the_plan_reaches_v41s_shifted_seam(self):
        """V4.1's cells are the seam's geometry too — (4, 5120) for every
        launch, the projection keyed under `mhc_pre_gemm` whichever plan
        spelled it."""
        for framework in ("vllm", "sglang"):
            for phase in ("prefill", "decode"):
                jobs, dropped = _plan("deepseek-ai/DeepSeek-V4.1-Flash",
                                      phase, framework)
                assert not [d for d in dropped if "mhc" in d]
                for sig in ("mhc_post", "mhc_pre_gemm", "mhc_pre_big_fuse"):
                    cells = {(j["dims"]["n_hidden_copies"],
                              j["dims"]["d_model"])
                             for j in jobs if j["signature"] == sig}
                    assert cells == {(4, 5120)}, (framework, phase, sig,
                                                  cells)
                stream = [j for j in jobs
                          if j["op"] == "mhc_pre_gemm_stream"]
                assert stream, (framework, phase)
                assert all(j["signature"] == "mhc_pre_gemm" for j in stream)


class TestDeepseekV4AttentionPrologue:
    """DeepSeek-V4's whole attention input preparation in one launch
    (`fusion_plans/fused_qnorm_rope_kv_insert{,_no_q_norm}.yaml`).

    The query heads normalised (where the span norms them) and rotated, the
    latent-KV row rotated, and that row quantised into the paged
    sliding-window cache at each token's slot. Two spans, one vLLM op with
    `apply_q_norm` the flag between them — as `gelu_and_mul` sits beside
    `silu_and_mul`: the same operands and bytes, a different function, and
    the oracle has to compute its own kernel's.
    """

    SPANS: ClassVar[tuple] = ("fused_qnorm_rope_kv_insert",
                              "fused_qnorm_rope_kv_insert_no_q_norm")
    #: A cell the launch actually runs: the kernel compiles in a 512-wide
    #: head with a 64-wide rotation, and 64 live heads is a count at which
    #: vLLM's own q-buffer padding is the identity.
    DIMS: ClassVar[dict] = {"B": 2, "s_query": 3, "n_heads": 64, "d_head": 512,
                            "d_rope": 64, "kv_block_size": 32}
    #: The record the launch writes the latent row in — the cell's
    #: `kv_format`, as the walk names DeepSeek-V4's sliding-window cache.
    RECORD: ClassVar[str] = "fp8_ue8m0_group64_bf16_rope_tail"

    def _inputs(self, sig, **over):
        from kernel_bench.refs import get_ref
        dims = dict(self.DIMS, **over)
        inv = Invocation(op=sig, signature=sig, phase="prefill", dims=dims,
                         dtype_env={"activation": "bf16", "kv": "fp8_e4m3",
                                    "kv_format": self.RECORD},
                         tp=1)
        return inv, build_inputs(get_ref(sig), inv, "cpu")

    def test_both_spans_bind_and_differ_only_in_the_query_norm(self):
        from backend.inference.op_signatures import get_signature, op_bindings
        bound = op_bindings()
        shapes = {}
        for sig in self.SPANS:
            assert bound[sig] == sig
            ps = _providers(sig)
            # vLLM alone ships this launch; the torch floor runs the chain.
            assert {p.library for p in ps} == {"vllm", "torch"}, sig
            assert all(p.fused_op == sig for p in ps), sig
            shapes[sig] = tuple(t.name for t in get_signature(sig).inputs)
            _, inputs = self._inputs(sig)
            assert inputs["q_norm"] is (sig == self.SPANS[0]), sig
        # Same operands, same bytes — only the function differs.
        assert shapes[self.SPANS[0]] == shapes[self.SPANS[1]]

    def test_the_oracle_rotates_the_query_and_the_latent_gptj_style(self):
        """GPT-J rope pairs channel `2i` with `2i+1`; the untouched `d_head -
        d_rope` head of every row passes through. Recomputed here from the
        pair definition, so the test does not share the reference's routine.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        for sig in self.SPANS:
            _inv_unused, inputs = self._inputs(sig)
            n_heads, d_head = self.DIMS["n_heads"], self.DIMS["d_head"]
            # The rotated width and the untouched head below it, both read
            # off the one dim: `nope` used to hard-code 64 beside `d_rope`,
            # which is the same number stated twice.
            rd = self.DIMS["d_rope"]
            nope = d_head - rd
            got = get_ref(sig).reference(inputs)
            assert got.shape == (6, n_heads * d_head + d_head)
            q = got[:, :n_heads * d_head].view(6, n_heads, d_head)
            # the NoPE head of the query is the input, normed where the span
            # norms it and otherwise untouched
            src = inputs["q"].float()
            if inputs["q_norm"]:
                src = src * torch.rsqrt(src.pow(2).mean(-1, keepdim=True)
                                        + 1e-6)
            assert torch.allclose(q[..., :nope], src[..., :nope], atol=1e-5)
            # and the rotated tail is the pair rotation at the token's angle
            cos, sin = inputs["cos_sin_cache"][inputs["positions"]].chunk(2, -1)
            x = src[..., nope:].view(6, n_heads, rd // 2, 2)
            want = torch.stack(
                [x[..., 0] * cos[:, None] - x[..., 1] * sin[:, None],
                 x[..., 1] * cos[:, None] + x[..., 0] * sin[:, None]], dim=-1)
            assert torch.allclose(q[..., nope:], want.flatten(-2), atol=1e-5)

    def test_the_two_spans_agree_wherever_the_query_norm_is_the_identity(self):
        """The only difference between the spans is the query norm, so on a
        query whose rows are already unit root-mean-square the two oracles
        must agree — the property that says the flag is the whole difference
        and nothing else drifted between them.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        _, a = self._inputs(self.SPANS[0])
        _, b = self._inputs(self.SPANS[1])
        unit = a["q"].float()
        unit = unit * torch.rsqrt(unit.pow(2).mean(-1, keepdim=True) + 1e-6)
        a, b = dict(a), dict(b)
        a["q"] = b["q"] = unit.to(a["q"].dtype)
        b["kv"] = a["kv"]
        assert torch.allclose(get_ref(self.SPANS[0]).reference(a),
                              get_ref(self.SPANS[1]).reference(b), atol=2e-2)

    def test_the_kernel_declines_a_page_its_cache_never_runs_at(self):
        """vLLM's sliding-window-MLA backend declares `MultipleOf(32)` and
        prefers 256 (`v1/attention/backends/mla/sparse_swa.py:154`, `:158`),
        so a 32-byte-aligned page is the only thing this cache is ever
        allocated in. The provider is gated on that property — a cell keyed
        at vLLM's global 16 is declined rather than measured at a paging
        layout that never happened.
        """
        for sig in self.SPANS:
            vllm = next(p for p in _providers(sig) if p.library == "vllm")
            pages = vllm.serves.kv_block_sizes
            assert pages and all(p % 32 == 0 for p in pages), (sig, pages)
            assert 16 not in pages and 256 in pages, (sig, pages)

    def test_the_geometry_the_kernel_compiles_in_is_what_it_accepts(self):
        """HEAD_DIM 512 / ROPE_DIM 64 are compiled into the kernel and
        checked at entry (`fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`
        :19-27, :1159-1161), so a cell of another width is a cell this launch
        cannot run — declined, not measured at the wrong geometry.
        """
        for sig in self.SPANS:
            vllm = next(p for p in _providers(sig) if p.library == "vllm")
            enum = dict(vllm.serves.enum or ())
            assert enum["d_head"] == (512,) and enum["d_rope"] == (64,), sig
            assert not vllm.accepts(self._inputs(sig, d_head=576)[0],
                                    TORCH_ENV), sig
            assert vllm.accepts(self._inputs(sig)[0], TORCH_ENV), sig

    def test_a_cache_this_overload_does_not_write_is_declined(self):
        """vLLM has three overloads of this op and dispatches on the cache
        tensor's dtype (attention.py:791 / :1056): the packed uint8 UE8M0
        record, a plain bf16 row, and a per-tensor-fp8 one. The build always
        allocates the packed layout, so a bf16-cache cell measured here
        would name one KV precision and time another — the row would be
        wrong in a way nothing downstream could see.

        The cache format is a permanent fact of the launch, so it is
        declined through the DECLARATION's `kv` slot — where smoke, the
        published capability record and the stale-row audit all read it —
        not through an `impl_id` side channel only the driver resolves.
        """
        for sig in self.SPANS:
            vllm = next(p for p in _providers(sig) if p.library == "vllm")
            inv_ok, _ = self._inputs(sig)
            assert vllm.serves.accepts(inv_ok), sig
            other = Invocation(
                op=sig, signature=sig, phase="prefill", dims=dict(self.DIMS),
                dtype_env={"activation": "bf16", "kv": "bf16"}, tp=1)
            assert not vllm.serves.accepts(other), sig
            assert any(token.startswith("kv:bf16")
                       for token in vllm.serves.refusals(other)), sig

    def test_a_per_tensor_fp8_cache_is_declined_by_its_record(self):
        """The per-tensor-FP8 overload's cell is `kv=fp8_e4m3` too: only the
        record's name (`kv_format`) tells it from the packed UE8M0 one this
        build allocates. Declined by name, and so is a cell that names no
        record (a plain FP8 cache)."""
        for sig in self.SPANS:
            vllm = next(p for p in _providers(sig) if p.library == "vllm")
            assert vllm.serves.kv_format == (self.RECORD,), sig
            for named in ("fp8_per_tensor", None):
                env = {"activation": "bf16", "kv": "fp8_e4m3"}
                if named:
                    env["kv_format"] = named
                other = Invocation(op=sig, signature=sig, phase="prefill",
                                   dims=dict(self.DIMS), dtype_env=env, tp=1)
                assert not vllm.serves.accepts(other), (sig, named)
                assert any(
                    token.startswith(f"kv_format:{named or '-'}")
                    for token in vllm.serves.refusals(other)), (sig, named)
            assert vllm.capabilities()["kv_formats"] == [self.RECORD]

    def test_the_oracle_writes_the_record_on_the_kv_port(self):
        """The latent half of the oracle is the row as the FlashMLA record
        holds it, re-derived here by hand (not `rows_on_record`): the 448
        unrotated channels on an e4m3 grid with one exponent per 64, rounded
        UP (`2^ceil(log2(amax/448))`), the 64 rotated channels rounded once
        to bf16. The query half is the fp32 function it always was."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        rd, d = self.DIMS["d_rope"], self.DIMS["d_head"]
        for sig in self.SPANS:
            _inv, inputs = self._inputs(sig)
            got = get_ref(sig).reference(inputs)
            plain = get_ref(sig).reference(dict(inputs, kv_format=None))
            nq = self.DIMS["n_heads"] * d
            assert torch.equal(got[:, :nq], plain[:, :nq]), sig
            row = plain[:, nq:]
            body = inputs["kv"][:, :d - rd].float().view(-1, (d - rd) // 64, 64)
            amax = body.abs().amax(-1, keepdim=True).clamp_min(1e-4)
            scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
            want = torch.cat([
                ((body / scale).to(torch.float8_e4m3fn).float() * scale
                 ).flatten(1),
                row[:, d - rd:].to(torch.bfloat16).float()], dim=-1)
            assert torch.equal(got[:, nq:], want), sig
            # and it is not the fp32 row: the record moved the latent
            assert not torch.equal(got[:, nq:], row), sig

    def test_the_record_round_trips_through_the_kernels_page_layout(self):
        """The oracle's record, packed into the paged uint8 layout vLLM's
        insert writes (448 e4m3 bytes and 64 bf16 values a token, then the
        page's exponent bytes, one pad byte a token: .cu:29-34, :99-101) and
        decoded by the provider's own reader (`_dsv4_swa_rows`), is the
        oracle's record again — so a kernel that writes that record is
        compared value for value."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.rope import (_dsv4_swa_record_bytes,
                                                 _dsv4_swa_rows)
        from kernel_bench.refs import get_ref
        sig = self.SPANS[0]
        rd, d = self.DIMS["d_rope"], self.DIMS["d_head"]
        page = self.DIMS["kv_block_size"]
        _inv, inputs = self._inputs(sig)
        nq = self.DIMS["n_heads"] * d
        record = get_ref(sig).reference(inputs)[:, nq:]
        tokens = record.shape[0]
        data_bytes, scale_bytes = _dsv4_swa_record_bytes(d, rd)
        groups = (d - rd) // 64
        body = record[:, :d - rd].view(tokens, groups, 64)
        exponent = torch.log2(body.abs().amax(-1).clamp_min(2.0 ** -120)
                              / 448.0).ceil()
        blocks = -(-tokens // page)
        cache = torch.zeros(blocks, page * (data_bytes + scale_bytes),
                            dtype=torch.uint8)
        slots = inputs["slot_mapping"]
        for t in range(tokens):
            b, w = int(slots[t]) // page, int(slots[t]) % page
            q = (body[t] / torch.exp2(exponent[t])[:, None]).to(
                torch.float8_e4m3fn).flatten().view(torch.uint8)
            tail = record[t, d - rd:].to(torch.bfloat16).view(torch.uint8)
            cache[b, w * data_bytes:(w + 1) * data_bytes] = torch.cat([q, tail])
            at = page * data_bytes + w * scale_bytes
            cache[b, at:at + groups] = (exponent[t] + 127).to(torch.uint8)
        decoded = _dsv4_swa_rows(cache, slots, page, d, rd)
        assert torch.equal(decoded, record)

    def test_the_floor_writes_the_record_the_oracle_writes(self):
        """The eager floor pays the launch's quantisation: at a record cell
        it lands on the oracle's record, judged on the band the driver reads
        for it — bf16's typical, a worst element of one e4m3 step — not the
        loose fp8 band an fp32 oracle needs. A cell naming no record is
        still the fp32 row on the `kv` slot's fp8 band."""
        pytest.importorskip("torch")
        from kernel_bench.driver import _xcheck_basis
        for sig in self.SPANS:
            env = {"activation": "bf16", "kv": "fp8_e4m3",
                   "kv_format": self.RECORD}
            _floor_meets_the_oracle(sig, self.DIMS, env=env)
            inv, inputs = self._inputs(sig)
            assert _xcheck_basis(inv, inputs) == ("bf16", (self.RECORD,))
            bare = Invocation(op=sig, signature=sig, phase="prefill",
                              dims=dict(self.DIMS), tp=1,
                              dtype_env={"activation": "bf16",
                                         "kv": "fp8_e4m3"})
            from kernel_bench.refs import get_ref
            assert _xcheck_basis(bare, build_inputs(
                get_ref(sig), bare, "cpu")) == ("fp8_e4m3", ())

    @pytest.mark.regression          # walks three models in both phases
    def test_the_plan_reaches_each_span_with_the_heads_it_bills(self):
        expect = {"deepseek-ai/DeepSeek-V4-Flash": (self.SPANS[0], 64),
                  "deepseek-ai/DeepSeek-V4-Pro": (self.SPANS[0], 128),
                  "deepseek-ai/DeepSeek-V4.1-Flash": (self.SPANS[1], 64)}
        for model, (sig, heads) in expect.items():
            for phase in ("prefill", "decode"):
                jobs, _ = _plan(model, phase, "vllm")
                cells = {(j["dims"]["n_heads"], j["dims"]["d_head"],
                          j["dims"]["d_rope"])
                         for j in jobs if j["signature"] == sig}
                assert cells == {(heads, 512, 64)}, (model, phase, cells)


class TestDeepseekV4QueryNormRope:
    """DeepSeek-V4's query track alone (`fusion_plans/fused_qnorm_rope.yaml`):
    sglang's `fused_q_norm_rope`, a gain-less per-head RMSNorm then the GPT-J
    rotation of each head's last `d_rope` channels, q in and q out."""

    SIG: ClassVar[str] = "fused_qnorm_rope"
    #: V4's geometry: a 512-wide head with a 64-wide rotation.
    DIMS: ClassVar[dict] = {"B": 2, "s_query": 3, "n_heads": 4, "d_head": 512,
                            "d_rope": 64}

    def test_sglang_and_the_floor_claim_the_span(self):
        from backend.inference.op_signatures import op_bindings
        assert op_bindings()[self.SIG] == self.SIG
        ps = _providers(self.SIG)
        assert {p.library for p in ps} == {"sglang", "torch"}
        assert all(p.fused_op == self.SIG for p in ps)

    def test_the_layers_freqs_cis_rotates_as_the_oracles_table(self):
        """The kernel reads `view_as_real(freqs_cis)` and rotates each
        ADJACENT pair of the rope tail as one complex number
        (main_norm_rope.cuh:171-197). Recomputed here that way — complex
        multiplication on the table the provider builds — so the check does
        not share the oracle's `[cos | sin]` routine: a NeoX pairing, or a
        table whose halves were swapped, would disagree."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.rope import _freqs_cis_from_cos_sin
        from kernel_bench.refs import get_ref
        spec = get_ref(self.SIG)
        inputs = build_inputs(spec, _inv(self.SIG, self.DIMS), "cpu")
        freqs_cis = _freqs_cis_from_cos_sin(inputs["cos_sin_cache"])
        assert freqs_cis.dtype == torch.complex64
        # unit modulus, as `polar(1, θ)` builds it
        assert torch.allclose(freqs_cis.abs(), torch.ones(()), atol=1e-6)
        x = inputs["x"].float()
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"])
        n = n.permute(0, 2, 1, 3).flatten(0, 1)            # token-major
        rd = self.DIMS["d_rope"]
        tail = torch.view_as_complex(
            n[..., -rd:].contiguous().view(*n.shape[:-1], rd // 2, 2))
        rotated = torch.view_as_real(
            tail * freqs_cis[inputs["positions"]][:, None]).flatten(-2)
        want = torch.cat([n[..., :-rd], rotated], dim=-1)
        assert torch.allclose(spec.reference(inputs), want, atol=1e-5)

    def test_the_geometry_the_template_compiles_is_what_it_accepts(self):
        """One (re, im) pair per lane fixes the rotation at 64, and the head
        must split into whole 16-byte vectors per lane (main_norm_rope.cuh
        :87-95) — any other cell is declined, not crashed into."""
        sgl = next(p for p in _providers(self.SIG) if p.library == "sglang")
        assert sgl.serves.accepts(_inv(self.SIG, self.DIMS))
        for bad in ({"d_rope": 32}, {"d_head": 576}, {"d_head": 192}):
            assert not sgl.serves.accepts(
                _inv(self.SIG, dict(self.DIMS, **bad))), bad


class TestDeepseekV4RowNormRope:
    """The KV-side norm+rope span (`fusion_plans/fused_norm_rope.yaml`): one
    row per token — V4's latent KV, the compressor's pooled entry, V4.1's
    indexer key — RMSNorm with a gain, then the GPT-J rotation of the row's
    last `d_rope` channels."""

    SIG: ClassVar[str] = "fused_norm_rope"
    DIMS: ClassVar[dict] = {"B": 2, "s_query": 3, "d_head": 512, "d_rope": 64}

    def test_the_contract_is_one_row_per_token(self):
        """The graph hands the span a `[B, s, d]` row. A per-head contract
        could not project it, and every match dropped before identity
        ("no shape projection") — the span was priced but never measured."""
        from backend.inference.op_signatures import get_signature
        sig = get_signature(self.SIG)
        x = next(t for t in sig.inputs if t.name == "x")
        assert x.axes == (("B", "s_query"), "d_head")

    def test_the_oracle_rotates_the_normed_row_as_one_head(self):
        """Checked against the rotation written out by hand (adjacent pairs
        as complex numbers), not the oracle's own `_rope_rotate`; and the
        floor meets it at the cell's precision."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref(self.SIG)
        inv = _inv(self.SIG, self.DIMS)
        inputs = build_inputs(spec, inv, "cpu")
        x = inputs["x"].float()
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"])
        n = n * inputs["w"].float()
        rd = self.DIMS["d_rope"]
        table = inputs["cos_sin_cache"][inputs["positions"]]
        rot = torch.complex(table[:, :rd // 2], table[:, rd // 2:])
        tail = torch.view_as_complex(n[:, -rd:].contiguous().view(-1, rd // 2, 2))
        want = torch.cat([n[:, :-rd],
                          torch.view_as_real(tail * rot).flatten(-2)], dim=-1)
        got = spec.reference(inputs)
        assert got.shape == (6, 512)
        assert torch.allclose(got, want, atol=1e-5)
        floor = next(p for p in _providers(self.SIG) if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        assert torch.allclose(case.to_canonical(case.run()), got,
                              rtol=2e-2, atol=5e-2)

    def _sgl(self, impl):
        return next(p for p in _providers(self.SIG) if p.kernel_impl == impl)

    def test_sglangs_two_launches_and_the_floor_claim_the_span(self):
        from backend.inference.op_signatures import op_bindings
        assert op_bindings()[self.SIG] == self.SIG
        ps = _providers(self.SIG)
        assert {(p.library, p.kernel_impl) for p in ps} == {
            ("torch", "torch_fused_norm_rope"),
            ("sglang", "sglang_compress_fused_norm_rope_inplace"),
            ("sglang", "sglang_fused_k_norm_rope_flashmla"),
            # V4.1's SWA row in DeepInfer, another record
            # (test_output_format_cells).
            ("deepinfer", "deepinfer_swa_norm_rope_store")}
        assert all(p.fused_op == self.SIG for p in ps)

    #: The FlashMLA V4 record's cell slots: its element dtype and the record.
    RECORD: ClassVar[dict] = {
        "output": "fp8_e4m3",
        "output_format": "fp8_ue8m0_group64_bf16_rope_tail"}

    def test_each_launch_is_keyed_by_the_format_it_writes(self):
        """The in-place launch writes the bf16 row back; the store writes the
        FP8 record and no row. A cell's `output` and `output_format` pick
        exactly one of them — and an FP8 cell keyed at another record (one
        F32 scale a row) neither."""
        inplace = self._sgl("sglang_compress_fused_norm_rope_inplace")
        store = self._sgl("sglang_fused_k_norm_rope_flashmla")
        for written, want in (({"output": "bf16"}, [inplace]),
                              (self.RECORD, [store]),
                              ({"output": "fp8_e4m3",
                                "output_format": "fp8_group128"}, [])):
            inv = Invocation(op=self.SIG, signature=self.SIG, phase="prefill",
                             dims=dict(self.DIMS), tp=1, dtype_env={
                                 "activation": "bf16", "weight": "bf16",
                                 **written})
            assert [p for p in (inplace, store)
                    if p.serves.accepts(inv)] == want, written

    @pytest.mark.parametrize("phase", ["prefill", "decode"])
    def test_the_compressor_plan_lands_every_row_on_its_position(self, phase):
        """The handle the provider builds, read back with the kernel's own
        arithmetic (fused_norm_rope.cuh:71-81) written out here: decode
        rotates row i at `seq_len - ratio` and skips it unless `seq_len %
        ratio == 0`; prefill reads `{ragged_id, batch_id, position,
        window_len}` and rotates row `ragged_id` at `position + 1 - ratio`."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.rope import (_SGL_COMPRESSOR_PLAN_RATIO,
                                                 _sgl_compressor_plan_handle)
        ratio = _SGL_COMPRESSOR_PLAN_RATIO
        positions = torch.tensor([0, 1, 2, 7, 4095, 5], dtype=torch.int64)
        handle = _sgl_compressor_plan_handle(phase, positions)
        if phase == "decode":
            assert handle.dtype == torch.int32
            assert bool((handle % ratio == 0).all())        # no row skipped
            rows = torch.arange(len(positions))
            at = handle.long() - ratio
        else:
            assert handle.dtype == torch.uint8
            assert handle.shape == (len(positions), 16)
            plan = handle.view(torch.int32)
            rows, at = plan[:, 0].long(), plan[:, 2].long() + 1 - ratio
        assert rows.tolist() == list(range(len(positions)))
        assert at.tolist() == positions.tolist()

    def test_the_record_decoder_reads_sglangs_padded_pages(self):
        """A V4 record written by hand where sglang's store puts it
        (kv_layout.cuh: data at `slot * 576`, scales from `page * 576`, the
        page stride padded up to a multiple of 576) decodes to the values
        written: e4m3 times `2^(byte - 127)` per 64 channels, then the bf16
        tail. Slot 5 at page 4 sits on the second page, so a decoder that
        assumed vLLM's unpadded stride would read the wrong bytes."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.rope import _dsv4_swa_rows
        page, slot = 4, 5
        stride = -(-page * 584 // 576) * 576
        assert stride != page * 584                  # the padding is real
        cache = torch.zeros(2, stride, dtype=torch.uint8)
        gen = torch.Generator().manual_seed(0)
        nope = (torch.rand(7, 64, generator=gen) * 400 - 200).to(
            torch.float8_e4m3fn)
        exps = torch.tensor([120, 127, 130, 125, 127, 140, 100])
        tail = torch.randn(64, generator=gen).to(torch.bfloat16)
        within = slot % page
        record = torch.cat([nope.flatten().view(torch.uint8),
                            tail.view(torch.uint8)])
        cache[1, within * 576:(within + 1) * 576] = record
        base = page * 576 + within * 8
        cache[1, base:base + 7] = exps.to(torch.uint8)
        cache[1, base + 7] = 0xAB                    # the pad byte, unread
        got = _dsv4_swa_rows(cache, torch.tensor([slot]), page, 512, 64)
        want = torch.cat([(nope.float() * torch.exp2(exps.float() - 127)
                           [:, None]).flatten(), tail.float()])
        assert torch.equal(got[0], want)

    def test_the_geometry_each_template_compiles_is_what_it_accepts(self):
        """The in-place template: rope exactly 64, the row a power of two
        from 64 to 2048 (fused_norm_rope.cuh:46-56; 96, 192 and 4096 fail to
        build). The store: (512, 64) only (main_norm_rope.cuh:450)."""
        inplace = self._sgl("sglang_compress_fused_norm_rope_inplace")
        store = self._sgl("sglang_fused_k_norm_rope_flashmla")
        def inv(written, **dims):
            return Invocation(op=self.SIG, signature=self.SIG,
                              phase="prefill", dims=dict(self.DIMS, **dims),
                              tp=1, dtype_env={"activation": "bf16",
                                               "weight": "bf16", **written})
        plain = {"output": "bf16"}
        for d in (64, 128, 512, 2048):
            assert inplace.serves.accepts(inv(plain, d_head=d)), d
        for bad in ({"d_rope": 32}, {"d_head": 192}, {"d_head": 4096}):
            assert not inplace.serves.accepts(inv(plain, **bad)), bad
        assert store.serves.accepts(inv(self.RECORD))
        for bad in ({"d_rope": 32}, {"d_head": 128}):
            assert not store.serves.accepts(inv(self.RECORD, **bad)), bad

    @pytest.mark.regression          # walks DeepSeek-V4-Flash and V4.1-Flash
    def test_the_plan_reaches_each_site_it_matches(self):
        # V4 on sglang: the latent KV and compressor entry (512) and the
        # indexer's own compressor entry (128). V4.1's indexer key went with
        # the DeepInfer binding (6eeceac9e): vLLM serves it as
        # fused_compress_norm_rope_store, its own tests' reach.
        for model, fw, widths in (("deepseek-ai/DeepSeek-V4-Flash", "sglang",
                                   {512, 128}),):
            jobs, dropped = _plan(model, "prefill", fw)
            cells = [j["dims"] for j in jobs if j["signature"] == self.SIG]
            assert cells, (model, fw)
            assert {c["d_head"] for c in cells} == widths, (model, cells)
            assert not [d for d in dropped if "no shape projection" in str(d)
                        and self.SIG in str(d)], (model, fw)


class TestHyperConnectionGatesAndExitReduce:
    """The hyper-connection boundaries that are not the cross-mixing seam
    (`fusion_plans/{ihc_post,fused_hc_head,hc_silu,hc_gate_mix,
    hc_combine_norm}.yaml`).

    A hyper-connection model carries `n_hidden_copies` parallel residual
    copies and has to read one hidden state out of them before each sublayer
    and write the sublayer's output back into all of them after. `mhc.py`
    covers the variant that cross-mixes the copies through a Sinkhorn
    matrix; these five are the rest — the spread with no cross-mix, the exit
    reduce, and the three launches of a flattened-stream model's gates.

    What is asserted here is each launch's own function and the declarations
    that keep a provider from measuring a different one. The recurring
    hazard in this family is that nearly every operand is an activation
    tensor of one of three shapes, so a transposed or mis-split call
    succeeds, returns the right shape, and is wrong.
    """

    #: THE ORDER IS LOAD-BEARING. `tests/conftest.py` marks every
    #: parametrised case but the first as `regression`, so the floor test
    #: below guards only `SPANS[0]` in the default run and in the deploy
    #: gate. `hc_combine_norm` is first because it is the floor most likely
    #: to regress and the only one where a wrong floor is hard to see: it is
    #: the only span with two outputs, the only one with an
    #: activation-dtype round trip in the middle of its chain, and the only
    #: one whose dtype assertion has to find the cell's width inside a
    #: tuple. It is also the floor that actually caught a bad assertion
    #: while this was being written. The other four are simple enough that
    #: `-m regression` is the right place for them.
    SPANS: ClassVar[tuple] = ("hc_combine_norm", "fused_hc_head",
                              "ihc_post", "hc_silu", "hc_gate_mix")
    #: A cell small enough to recompute by hand. `d_hc_mix` is the low-rank
    #: mix width, which only `hc_silu` reads.
    DIMS: ClassVar[dict] = {"B": 2, "s_query": 3, "n_hidden_copies": 4,
                            "d_model": 128, "d_hc_mix": 24}
    #: Which frameworks are BOUND for each span, which is what the provider
    #: set has to match. sglang binds only `hc_combine_norm`: the v0.5.22
    #: audit (`providers/hc.py`) found its read fused into the mix GEMMs —
    #: no `hc_silu` / `hc_gate_mix` launch exists to bind — and its write
    #: fused with the next norm on the one-token SM100 decode path
    #: (`hc_combine_apply_norm`). An arm beyond that would be a measurement
    #: of a kernel no binding says sglang launches.
    LIBS: ClassVar[dict] = {
        "ihc_post": {"vllm", "sglang", "torch"},
        "fused_hc_head": {"vllm", "sglang", "torch"},
        "hc_silu": {"vllm", "torch"},
        "hc_gate_mix": {"vllm", "torch"},
        "hc_combine_norm": {"vllm", "sglang", "torch"},
    }

    def _cell(self, sig, **over):
        from kernel_bench.refs import get_ref
        inv = _inv(sig, dict(self.DIMS, **over))
        return inv, build_inputs(get_ref(sig), inv, "cpu")

    def test_each_span_binds_and_only_bound_frameworks_claim_it(self):
        """A span with a signature but no provider is off the unreachable
        list while nothing can measure it, so the two are asserted together.

        The per-span library set is the discriminating half: registering an
        sglang arm for `hc_gate_mix` would be a measurement of a kernel no
        binding says sglang launches, and nothing else here would notice.
        What makes that easy to do by accident is that sglang HAS a
        `hc_combine_norm` — DeepSeek-V4's, a different function (see
        `providers/hc.py`).
        """
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        for sig in self.SPANS:
            assert bound[sig] == sig
            ps = _providers(sig)
            assert {p.library for p in ps} == self.LIBS[sig], sig
            assert all(p.fused_op == sig for p in ps), sig

    def test_the_declared_widths_are_each_kernels_own_not_the_familys(self):
        """vLLM's exit reduce allocates a BF16 output whatever it was handed
        (tilelang.py:1050) and its TileLang kernel types the residual and
        the output `T.bfloat16`, while sglang's Triton kernel writes
        `x.dtype`. So the two arms of ONE span declare different widths.

        Asserted because the tempting thing is one constant for the family:
        with fp16 added to the vLLM arm, an fp16 cell would be accepted,
        measured, and recorded as fp16 while the kernel wrote bf16.
        """
        widths = {(p.library, p.serves.activation)
                  for p in _providers("fused_hc_head")}
        assert ("vllm", ("bf16",)) in widths, widths
        assert ("sglang", ("bf16", "fp16")) in widths, widths
        assert ("torch", TORCH_FLOAT) in widths, widths
        # And a floor declares every float width, for every span: it runs
        # the chain at whatever width the cell states, which is what a
        # floor is for — said, not left open.
        for sig in self.SPANS:
            floor = next(p for p in _providers(sig) if p.library == "torch")
            assert floor.serves.activation == TORCH_FLOAT, sig

    def test_the_sglang_write_arm_declares_its_kernels_own_specialization(
            self):
        """sglang's `hc_combine_apply_norm` is compiled for exactly one
        geometry — bf16, 4 copies of 2560, at most 4 rows (the TensorMatcher
        and CHECK_HOST in hc_combine_decode.cuh) — and reached only on the
        decode step. The arm's gate states those facts and no more: the
        one-token dispatch that brings the model to the launch is the
        binding's `when:`, not this gate's, because the kernel itself runs
        2-4 rows as faithfully as one.

        Asserted clause by clause, through `refusals` rather than the
        dataclass fields, so a gate that quietly stops discriminating (an
        enum dropped, a ceiling widened) fails here with the clause named.
        """
        arm = next(p for p in _providers("hc_combine_norm")
                   if p.library == "sglang")
        good = dict(self.DIMS, B=1, s_query=1, d_model=2560)
        inv = _inv("hc_combine_norm", good, phase="decode")
        assert list(arm.serves.refusals(inv)) == []
        for why, dims, phase, dtype in (
                ("prefill", good, "prefill", "bf16"),
                ("fp16", good, "decode", "fp16"),
                ("copies", dict(good, n_hidden_copies=8), "decode", "bf16"),
                ("width", dict(good, d_model=2048), "decode", "bf16"),
                ("rows", dict(good, B=2, s_query=3), "decode", "bf16")):
            inv = _inv("hc_combine_norm", dims, phase=phase)
            inv.dtype_env["activation"] = dtype
            refused = list(arm.serves.refusals(inv))
            assert refused, (why, "accepted")
        # The vLLM arm of the SAME span keeps its own, wider claim — the
        # asymmetry is the kernels', not a family constant.
        vllm = next(p for p in _providers("hc_combine_norm")
                    if p.library == "vllm")
        assert vllm.serves.activation == ("bf16", "fp16")
        assert list(vllm.serves.refusals(
            _inv("hc_combine_norm", self.DIMS))) == []

    def test_sglangs_decode_write_is_not_offered_below_blackwell(self):
        """sglang reaches `hc_combine_apply_norm` only when the gate stream
        exists, and it creates that stream only on capability major 10
        (srt/layers/hyperconnection.py:186-196 at v0.5.22) — the binding's
        `min_compute_capability: "10.0"`. The kernel needs it: its
        `AlignedVector<bf16x2, 8>` is a 32-byte vector, which
        `sgl_kernel/vec.cuh:79` static-asserts away below Blackwell, so on
        an H100 the JIT build fails and its canary disabled the path on
        every pass. Declined there, not attempted."""
        arm = next(p for p in _providers("hc_combine_norm")
                   if p.library == "sglang")
        lib = {"sglang.kernels.ops.elementwise.hc_combine_decode": "0.5.22"}
        h100 = Env("2.11", "cuda", None, None, None, sm="sm_90", libs=lib)
        b200 = Env("2.11", "cuda", None, None, None, sm="sm_100", libs=lib)
        assert not arm.available(h100)
        assert arm.available(b200)

    @pytest.mark.regression          # walks the Qwen4Exp model four times
    def test_sglang_binds_the_write_only_where_the_walk_proves_one_token(
            self):
        """The binding's `B: [1]` clause, read through the plan.

        The plan walk leaves B SYMBOLIC — it is walked once and re-read at
        every batch — and `dims:` binds only a width the walk walked as a
        number, so sglang's fused write reaches no plan, batched or not.
        That is the conservative reading and the honest one: sglang fuses
        the write with the next norm only on a one-token decode step, and
        an estimate that cannot prove the token count must not claim it —
        the alternative (dropping the clause) would price a fusion on
        every batched decode step that sglang only runs at batch 1. Where
        the walk DOES run B as a number the fusion fires; the eval rewrite
        pins that direction (test_eval_fusion).
        """
        from backend.inference.framework_bindings import (
            ServingDeployment, get_binding)
        from kernel_bench.plan import build_jobs

        def cells(sku, phase, batch):
            binding = get_binding("sglang").deployed(ServingDeployment.at(
                gpu_sku=sku, activation_dtype="BF16"))
            jobs, _ = build_jobs(avg_output_tokens=128,
                                 model_id="Qwen/Qwen3.8-Flash-Next",
                                 batch_size=batch, seq_len=1024, phase=phase,
                                 framework="sglang", binding=binding, gpu=sku)
            return {j["signature"] for j in jobs} & set(self.SPANS)

        # No hc job of any span, in any of these — the read gates because
        # sglang has no launch at their boundary, the write because the
        # plan walk cannot prove the one-token step.
        for sku, phase, batch in (("B200", "decode", 1),
                                  ("H100-SXM", "decode", 1),
                                  ("B200", "prefill", 1),
                                  ("B200", "decode", 2)):
            assert cells(sku, phase, batch) == set(), (sku, phase, batch)

    def test_the_spread_gates_each_copy_by_its_own_scalar(self):
        """`out[c] = post[c]·sublayer + residual[c]` — the independent
        hyper-connection, where a copy sees only its own gate.

        Recomputed with an explicit loop over the copies, so the test does
        not share the reference's broadcast. The discriminating part is that
        the gate indexes the COPY and the sublayer output does not: the two
        broadcasts go on different axes, and swapping them (gate over the
        width, output over the copies) also produces a
        `[tokens, copies, width]` tensor.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        hc, d = self.DIMS["n_hidden_copies"], self.DIMS["d_model"]
        _i, inputs = self._cell("ihc_post")
        post, sub = inputs["post"].float(), inputs["sublayer"].float()
        res = inputs["residual"].float()
        want = torch.zeros(6, hc, d)
        for c in range(hc):
            want[:, c, :] = post[:, c, None] * sub + res[:, c, :]
        got = get_ref("ihc_post").reference(inputs)
        assert got.shape == (6, hc, d)
        assert torch.allclose(got, want, atol=1e-5)
        # There is NO cross-mix: copy c's output must not move when another
        # copy's residual changes. That is the whole difference from
        # `mhc_post`, and a reference that reached for the seam's arithmetic
        # would fail here rather than quietly computing a mixed answer.
        other = dict(inputs)
        res2 = inputs["residual"].clone()
        res2[:, 1, :] += 1.0
        other["residual"] = res2
        moved = get_ref("ihc_post").reference(other)
        assert torch.allclose(moved[:, 0, :], got[:, 0, :], atol=1e-5)
        assert not torch.allclose(moved[:, 1, :], got[:, 1, :], atol=1e-2)

    def test_the_exit_reduce_is_a_convex_like_gate_over_the_copies(self):
        """The exit reduce normalises the flattened stream, projects it onto
        one dot per copy, sigmoids that into a gate and sums the copies under
        it.

        Asserted as properties of the function rather than against a second
        copy of it:

        * it is LINEAR in the copies at a frozen gate — the gate is a
          function of the whole row, so this is checked by holding the row
          fixed and comparing against the gate the reference itself implies;
        * the RMS normalisation makes the GATE scale-invariant, so scaling
          the whole residual scales the output by exactly that factor. A
          reference that forgot the rsqrt would move the gate instead and
          fail this.
        * `+ hc_eps` after the sigmoid is in both kernels, which is only
          observable where the sigmoid itself vanishes — so it is driven
          there rather than asserted as a bound the sigmoid satisfies on its
          own. `gate > hc_eps` on ordinary input is decoration: the gate
          sits near 0.26-0.55 there and the claim holds whether or not the
          epsilon is added at all.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _HC_EPS, get_ref
        d = self.DIMS["d_model"]
        _i, inputs = self._cell("fused_hc_head")
        got = get_ref("fused_hc_head").reference(inputs)
        assert got.shape == (6, d)
        # Scale invariance of the gate ⇒ homogeneity of degree 1 overall.
        scaled = dict(inputs, x=(inputs["x"].float() * 4.0).to(
            inputs["x"].dtype))
        again = get_ref("fused_hc_head").reference(scaled)
        assert torch.allclose(again, 4.0 * got, atol=2e-2, rtol=2e-2)
        # The gate the output implies, recovered on a one-hot copy: zeroing
        # every copy but one leaves `pre_0 · x_0`, so the ratio IS the gate.
        one = inputs["x"].clone()
        one[:, 1:, :] = 0
        lone = get_ref("fused_hc_head").reference(dict(inputs, x=one))
        ref0 = one[:, 0, :].float()
        live = ref0.abs() > 1e-2
        gate = lone[live] / ref0[live]
        # It is not the degenerate constant a zeroed projection would give.
        assert float(gate.std()) > 1e-4, "gate is constant across tokens"
        # Now the epsilon, where it is the only thing left: a large negative
        # bias underflows sigmoid to exactly 0.0 in fp32, so the gate IS
        # `hc_eps` — and 0.0 without it.
        floored = get_ref("fused_hc_head").reference(
            dict(inputs, x=one,
                 base=torch.full_like(inputs["base"], -1000.0)))
        gate0 = floored[live] / ref0[live]
        assert gate0.abs().max() > 0, "the gate floor collapsed to zero"
        assert torch.allclose(gate0, torch.full_like(gate0, _HC_EPS),
                              rtol=1e-3), float(gate0.max())

    def test_the_mix_activation_divides_inside_the_silu(self):
        """`silu(x / hc)`, not `silu(x) / hc`. SiLU is not homogeneous, so
        the two are different functions — and this is the one launch here
        whose divisor the cell cannot supply (its operand is the low-rank
        mix vector, which has no copy axis), so it comes from the one stated
        constant that the oracle and the provider both read.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _HC_SILU_COPIES, get_ref
        _i, inputs = self._cell("hc_silu")
        got = get_ref("hc_silu").reference(inputs)
        x = inputs["x"].float()
        inside = (x / _HC_SILU_COPIES) * torch.sigmoid(x / _HC_SILU_COPIES)
        outside = (x * torch.sigmoid(x)) / _HC_SILU_COPIES
        assert torch.allclose(got, inside, atol=1e-6)
        # The two really do differ on this input, so the assertion above is
        # discriminating rather than true of both.
        assert not torch.allclose(inside, outside, atol=1e-3)

    def test_the_read_splits_the_leading_copy_index(self):
        """`y = (1/hc)·Σ_c sigmoid(gate_c)·xn_c` over a stream held
        `[copies, width]`-major (HC outer, HS inner —
        `hyperconnection.py:11-12`).

        The discriminating assertion is the SPLIT, not the arithmetic: the
        row is `copies·width` long, so viewing it as `[width, copies]` and
        reducing the other axis also yields a `[tokens, d_model]` answer, on
        every cell where the two happen to be equal in size. Checked by
        building a stream whose copies are constant along the width, where
        the two readings disagree by construction.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        hc, d = self.DIMS["n_hidden_copies"], self.DIMS["d_model"]
        _i, inputs = self._cell("hc_gate_mix")
        # copy c is the constant (c + 1); the gate is open everywhere.
        xn = torch.arange(1, hc + 1, dtype=torch.float32).repeat_interleave(d)
        xn = xn.unsqueeze(0).expand(6, hc * d).contiguous()
        big = torch.full((6, hc * d), 20.0)          # sigmoid(20) ≈ 1
        got = get_ref("hc_gate_mix").reference(
            dict(inputs, xn=xn.to(inputs["xn"].dtype),
                 gate=big.to(inputs["gate"].dtype)))
        # mean over copies of (1, 2, 3, 4) = 2.5, the same for every channel
        assert torch.allclose(got, torch.full((6, d), 2.5), atol=2e-2)
        # The transposed reading would average channels within a copy, which
        # on this stream gives the ramp 1,1,..,2,2,.. reduced the other way —
        # a different answer of the same shape.
        wrong = (torch.sigmoid(big).view(6, d, hc)
                 * xn.view(6, d, hc)).sum(-1) / hc
        assert wrong.shape == got.shape
        assert not torch.allclose(wrong, got, atol=1e-1)

    def test_the_write_rounds_the_combined_stream_before_norming_it(self):
        """The write's two outputs: the combined stream and its per-copy
        RMS norm scaled by `(1 + w)`.

        Three things the kernel fixes, each asserted because getting it
        wrong leaves a plausible tensor:

        * the injection gate divides BEFORE the sigmoid
          (`2·sigmoid(logit/hc)`, kernel :322) — dividing after is another
          function with the same range;
        * the norm reads the stream ROUNDED to the activation dtype
          (:325-327, "matches the unfused combine -> RMSNorm boundary"), so
          the second output is a function of the FIRST output and not of the
          fp32 intermediate;
        * the norm is per COPY over that copy's own `d_model`, not over the
          whole flattened row.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import _HC_RMS_EPS, get_ref
        hc, d = self.DIMS["n_hidden_copies"], self.DIMS["d_model"]
        _i, inputs = self._cell("hc_combine_norm")
        got = get_ref("hc_combine_norm").reference(inputs)
        assert got.shape == (6, 2 * hc * d)
        out, y = got[:, :hc * d], got[:, hc * d:]
        # The combined stream went to HBM at the cell's width and the norm
        # read back what was stored, so every element of the first write is
        # EXACTLY representable in bf16. This is the assertion that catches
        # a reference keeping the fp32 sum: recomputing `y` from the
        # published `out` below does not, because dropping the round trip
        # makes both writes the unrounded value and leaves them consistent
        # with each other.
        assert torch.equal(out, out.to(torch.bfloat16).float()), (
            "the combined stream was never rounded to the cell's width")
        # The second write is the first one normed, per copy — recomputed
        # from the published first output, so it also pins WHICH tensor the
        # norm read.
        rows = out.view(6, hc, d)
        want = rows * torch.rsqrt(rows.square().mean(-1, keepdim=True)
                                  + _HC_RMS_EPS)
        want = want.view(6, hc * d) * (1.0 + inputs["norm_weight"].float())
        assert torch.allclose(y, want, atol=1e-6), "norm is not of `out`"
        # Per COPY: normalising the whole flattened row instead would give a
        # different answer on this input, so the assertion above excludes it.
        whole = out * torch.rsqrt(out.square().mean(-1, keepdim=True)
                                  + _HC_RMS_EPS)
        whole = whole * (1.0 + inputs["norm_weight"].float())
        assert not torch.allclose(y, whole, atol=1e-2)
        # The gate divides before the sigmoid. Recovered from the output: at
        # a zero residual, `out = block · 2·sigmoid(inj/hc)`, so the ratio
        # per copy is that gate.
        zero = dict(inputs, residual=torch.zeros_like(inputs["residual"]))
        lone = get_ref("hc_combine_norm").reference(zero)[:, :hc * d]
        block = inputs["block"].float()
        live = block.abs() > 1e-2
        inj = inputs["injection"].float()
        for c in range(hc):
            ratio = (lone.view(6, hc, d)[:, c, :][live]
                     / block[live])
            before = (2.0 * torch.sigmoid(inj[:, c] / hc)).unsqueeze(-1)
            before = before.expand(6, d)[live]
            assert torch.allclose(ratio, before, atol=2e-2), c
        # and dividing AFTER the sigmoid is a different gate on this input
        after = (2.0 * torch.sigmoid(inj) / hc)
        assert not torch.allclose(after, 2.0 * torch.sigmoid(inj / hc),
                                  atol=1e-2)

    @pytest.mark.parametrize("sig", SPANS)
    def test_every_launch_has_a_floor_running_its_chain_at_the_cells_width(
            self, sig):
        """Each span has a torch floor, and each floor runs that span's naive
        chain AT THE CELL'S PRECISION.

        The floor is the denominator of the span's saving, so a span whose
        floor does not run has no measured saving even where its kernel does.
        The precision is the part that breaks quietly: the oracle computes in
        fp32 on purpose, to be a trusted reference, and a floor that simply
        calls the oracle reports fp32 latency for a bf16 cell — which the
        driver refuses ("the launch's largest operand is float32 where the
        cell declares bfloat16"), so the span loses its floor and keeps its
        kernel.

        Held TWO-SIDEDLY for that reason. One bf16 ULP is 2^-8 ≈ 3.9e-3
        relative, so a floor running the chain at the cell's width lands a
        few ULP from the fp32 oracle — near enough to be the same
        arithmetic. The LOWER bound is the one that catches the regression: a
        floor that just called the oracle would agree to fp32 rounding
        (~1e-7) and sail through an upper bound alone.
        """
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.attention import _torch_dtype
        from kernel_bench.refs import get_ref
        inv, inputs = self._cell(sig)
        floor = next(p for p in _providers(sig) if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        got = case.to_canonical(case.run())
        want = get_ref(sig).reference(inputs)
        assert got.shape == want.shape, sig
        assert got.dtype == torch.float32, sig       # canonical is fp32
        rel = float((got - want).abs().max() / want.abs().max())
        assert 1e-3 <= rel <= 2e-2, (sig, rel)
        # The same fact read off the raw output: the launch's
        # activation-valued write carries the cell's dtype.
        raw = case.run()
        widths = {t.dtype for t in (raw if isinstance(raw, tuple) else (raw,))
                  if hasattr(t, "dtype")}
        assert _torch_dtype(inv.dtype) in widths, (sig, widths)

    @pytest.mark.regression          # walks three models in both phases
    def test_the_declared_engine_order_is_the_order_the_walk_emits(self):
        """The declaration, checked against the walk it describes.

        `engine_inputs` states each span's operand layouts as the engine
        emits them, and three of these five permute against the frameworks'
        argument order — so a wrong declaration is a silently transposed
        measurement, not a crash. The check is this direction on purpose:
        the declaration is the source, and the walk either agrees with it or
        does not.

        Asserted through the plan rather than by re-reading the graph: a
        layout that does not match what the walk emits fails to BIND (zip
        looks up `(position, rank)` and does not search), so the span's cells
        drop with "no shape projection" and the span reaches no job. That is
        exactly how `hc_gate_mix` failed when its two operands were declared
        in the 4-D split form the write's residual arrives in.
        """
        expect = {"tencent/Hy4-preview": ("ihc_post", 6144),
                  "deepseek-ai/DeepSeek-V4-Pro": ("fused_hc_head", 7168),
                  "Qwen/Qwen3.8-Flash-Next": ("hc_gate_mix", 2560)}
        for model, (sig, d_model) in expect.items():
            for phase in ("prefill", "decode"):
                jobs, dropped = _plan(model, phase, "vllm")
                cells = {(j["dims"]["n_hidden_copies"], j["dims"]["d_model"])
                         for j in jobs if j["signature"] == sig}
                assert cells == {(4, d_model)}, (model, phase, sig, cells,
                                                 [d for d in dropped
                                                  if sig in d])

    @pytest.mark.regression          # walks the Qwen4Exp model in both phases
    def test_the_qwen_launches_all_reach_the_plan_at_one_cell_each(self):
        """All three gates of a flattened-stream model, at the geometry the
        estimate prices them at. `hc_silu` keys on the low-rank mix width and
        the other two on the copies and the model width, so they do not share
        a cell — which is correct, they are three different launches.
        """
        for phase in ("prefill", "decode"):
            jobs, _ = _plan("Qwen/Qwen3.8-Flash-Next", phase, "vllm")
            by_sig = {}
            for sig in ("hc_silu", "hc_gate_mix", "hc_combine_norm"):
                by_sig[sig] = {tuple(sorted(j["dims"].items()))
                               for j in jobs if j["signature"] == sig}
                assert len(by_sig[sig]) == 1, (phase, sig, by_sig[sig])
            silu = dict(next(iter(by_sig["hc_silu"])))
            assert silu["d_hc_mix"] == 320 and "d_model" not in silu, silu
            read = dict(next(iter(by_sig["hc_gate_mix"])))
            assert read["d_model"] == 2560 and read["n_hidden_copies"] == 4


class TestFusedNormSpans:
    """Norm spans one launch replaces: the MTP entry (`fused_eh_norm`), the
    attention a-norm pair (`fused_q_kv_rmsnorm`) and Gemma-4's three
    (`gemma_qkv_rmsnorm`, `gemma_rmsnorm_residual_scalar`,
    `gemma_dual_rmsnorm_residual_scalar`). Each oracle is checked against the
    span recomputed another way — by row, by head, or through a sibling span
    it reduces to — so a test cannot pass by sharing the oracle's code."""

    SPANS: ClassVar[dict] = {
        "fused_eh_norm": {"vllm", "sglang", "torch"},
        "fused_q_kv_rmsnorm": {"vllm", "sglang", "torch"},
        "gemma_qkv_rmsnorm": {"sglang", "torch"},
        "gemma_rmsnorm_residual_scalar": {"sglang", "torch"},
        "gemma_dual_rmsnorm_residual_scalar": {"sglang", "torch"},
    }

    @staticmethod
    def _rms(x, w=None):
        torch = pytest.importorskip("torch")
        rows = []
        for row in x.float():                 # one token at a time
            r = row / torch.sqrt(row.pow(2).sum() / row.numel() + 1e-6)
            rows.append(r if w is None else r * w.float())
        return torch.stack(rows)

    def test_each_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        for sig, libs in self.SPANS.items():
            assert bound[sig] == sig
            ps = _providers(sig)
            assert {p.library for p in ps} == libs, sig
            assert all(p.fused_op == sig for p in ps), sig

    def test_each_floor_runs_the_chain_at_the_cells_width(self):
        dims = {"B": 2, "s_query": 3, "d_model": 64, "d_q_lora": 96,
                "d_kv_lora": 32, "n_heads": 4, "n_kv_heads": 2, "d_head": 16}
        for sig in self.SPANS:
            _floor_meets_the_oracle(sig, dims)

    def test_sglangs_mtp_entry_serves_only_the_widths_its_jit_builds(self):
        """`is_supported_fused_eh_norm_hidden_size` (fused_eh_norm.py:18-19)
        takes a multiple of 256 in (256, 8192] — 256 itself refused, so the
        declaration enumerates the widths; a modulus plus a ceiling would
        claim a cell the kernel raises on."""
        sglang = next(p for p in _providers("fused_eh_norm")
                      if p.library == "sglang")
        for width, ok in ((256, False), (384, False), (512, True),
                          (7168, True), (8192, True), (8448, False)):
            inv = _inv("fused_eh_norm",
                       {"B": 2, "s_query": 1, "d_model": width})
            assert sglang.serves.accepts(inv) is ok, width

    def test_the_mtp_entry_is_the_two_norms_side_by_side(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("fused_eh_norm")
        inputs = build_inputs(spec, _inv("fused_eh_norm", {
            "B": 2, "s_query": 3, "d_model": 64}), "cpu")
        got = spec.reference(inputs)
        assert got.shape == (6, 128)
        want_e = self._rms(inputs["inputs_embeds"], inputs["enorm_w"])
        want_h = self._rms(inputs["hidden"], inputs["hnorm_w"])
        assert (got[:, :64] - want_e).abs().max() < 1e-5
        assert (got[:, 64:] - want_h).abs().max() < 1e-5

    def test_the_a_norm_pair_keeps_each_row_its_own_width_and_gain(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("fused_q_kv_rmsnorm")
        inputs = build_inputs(spec, _inv("fused_q_kv_rmsnorm", {
            "B": 2, "s_query": 3, "d_q_lora": 96, "d_kv_lora": 32}), "cpu")
        got = spec.reference(inputs)
        assert got.shape == (6, 128)
        assert (got[:, :96] - self._rms(inputs["q"], inputs["q_weight"])
                ).abs().max() < 1e-5
        assert (got[:, 96:] - self._rms(inputs["kv"], inputs["kv_weight"])
                ).abs().max() < 1e-5

    def test_the_q_k_norm_pair_has_a_row_form_and_a_per_head_form(self):
        """One signature, two geometries, selected by the per-head axes the
        cell carries (`variant_selectors`) and nothing else: the MLA a-norm
        cell (vLLM) has none of them, the per-head Q/K cell (sglang
        `fused_inplace_qknorm`) all three."""
        from kernel_bench.signatures import get_signature
        sig = get_signature("fused_q_kv_rmsnorm")
        assert set(sig.variant_selectors) == {"n_heads", "n_kv_heads",
                                              "d_head"}
        row = {"B": 2, "s_query": 1, "d_q_lora": 1536, "d_kv_lora": 512}
        head = {"B": 2, "s_query": 1, "n_heads": 32, "n_kv_heads": 8,
                "d_head": 128}
        assert sig.layouts_for(row) is sig.inputs
        assert sig.layouts_for(head) is sig.variant_inputs
        assert set(sig.cell_dims_for(row)) == set(row)
        assert set(sig.cell_dims_for(head)) == set(head)
        # A per-head key projects whole, a row key as before.
        import json
        assert json.loads(sig.project(json.dumps(head))) == head
        assert json.loads(sig.project(json.dumps(row))) == row

    def test_each_kernel_serves_only_its_own_geometry(self):
        """Declined through the declarations: vLLM's row kernel refuses a
        per-head cell, sglang's per-head JIT a row cell, a head width it
        builds no kernel for (norm.py:101-110) and an fp32 activation
        (qknorm.cuh:108)."""
        row = {"B": 2, "s_query": 1, "d_q_lora": 1536, "d_kv_lora": 512}
        head = {"B": 2, "s_query": 1, "n_heads": 32, "n_kv_heads": 8,
                "d_head": 128}
        by = {p.library: p for p in _providers("fused_q_kv_rmsnorm")}
        vllm, sgl, torch_ = by["vllm"], by["sglang"], by["torch"]
        assert vllm.serves.accepts(_inv("fused_q_kv_rmsnorm", row))
        assert not vllm.serves.accepts(_inv("fused_q_kv_rmsnorm", head))
        assert sgl.serves.accepts(_inv("fused_q_kv_rmsnorm", head))
        assert sgl.serves.accepts(_inv("fused_q_kv_rmsnorm", head, "decode"))
        assert not sgl.serves.accepts(_inv("fused_q_kv_rmsnorm", row))
        assert not sgl.serves.accepts(
            _inv("fused_q_kv_rmsnorm", {**head, "d_head": 96}))
        fp32 = Invocation(op="fused_q_kv_rmsnorm",
                          signature="fused_q_kv_rmsnorm", phase="decode",
                          dims=head, dtype_env={"activation": "fp32"}, tp=1)
        assert not sgl.serves.accepts(fp32)
        assert torch_.serves.accepts(_inv("fused_q_kv_rmsnorm", row))
        assert torch_.serves.accepts(_inv("fused_q_kv_rmsnorm", head))

    def test_the_per_head_pair_normalises_each_head_with_its_sides_gain(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("fused_q_kv_rmsnorm")
        inputs = build_inputs(spec, _inv("fused_q_kv_rmsnorm", {
            "B": 1, "s_query": 3, "n_heads": 4, "n_kv_heads": 2,
            "d_head": 16}), "cpu")
        assert tuple(inputs["q"].shape) == (3, 4, 16)
        assert tuple(inputs["kv"].shape) == (3, 2, 16)
        assert tuple(inputs["q_weight"].shape) == (16,)
        got = spec.reference(inputs)
        assert got.shape == (3, (4 + 2) * 16)
        want = [self._rms(inputs["q"][:, h], inputs["q_weight"])
                for h in range(4)]
        want += [self._rms(inputs["kv"][:, h], inputs["kv_weight"])
                 for h in range(2)]
        assert (got - torch.cat(want, dim=-1)).abs().max() < 1e-5
        # The row form's floor too: the class's shared dims select the
        # per-head one (`test_each_floor_runs_the_chain_at_the_cells_width`).
        _floor_meets_the_oracle("fused_q_kv_rmsnorm", {
            "B": 2, "s_query": 3, "d_q_lora": 96, "d_kv_lora": 32})

    def test_gemmas_norms_are_per_head_and_v_takes_no_gain(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("gemma_qkv_rmsnorm")
        inputs = build_inputs(spec, _inv("gemma_qkv_rmsnorm", {
            "B": 1, "s_query": 3, "n_heads": 4, "n_kv_heads": 2,
            "d_head": 16}), "cpu")
        got = spec.reference(inputs)
        assert got.shape == (3, (4 + 2 + 2) * 16)
        want = []
        for name, w in (("q", inputs["q_weight"]), ("k", inputs["k_weight"]),
                        ("v", None)):
            x = inputs[name]
            want.extend(self._rms(x[:, h * 16:(h + 1) * 16], w)
                        for h in range(x.shape[1] // 16))
        assert (got - torch.cat(want, dim=-1)).abs().max() < 1e-5

    def test_the_moe_tail_with_one_silent_branch_is_the_dense_tail(self):
        """rmsnorm(0) is 0, so with the expert branch silent the dual tail
        is the single one over the dense branch's normed output."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        dual = get_ref("gemma_dual_rmsnorm_residual_scalar")
        single = get_ref("gemma_rmsnorm_residual_scalar")
        dims = {"B": 1, "s_query": 4, "d_model": 64}
        inputs = build_inputs(dual, _inv(
            "gemma_dual_rmsnorm_residual_scalar", dims), "cpu")
        inputs["x2"] = torch.zeros_like(inputs["x2"])
        branch = self._rms(inputs["x1"], inputs["w1"])
        tail = single.reference({"x": branch, "w": inputs["w3"],
                                 "residual": inputs["residual"],
                                 "scalar": inputs["scalar"], "eps": 1e-6})
        assert (dual.reference(inputs) - tail).abs().max() < 1e-4

    def test_the_cells_are_read_off_the_engine_operands(self):
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        cases = {
            # DeepSeek-V3.2 MTP, decode
            "fused_eh_norm": ([[2, 1, 7168], [2, 1, 7168]],
                              {"B": 2, "s_query": 1, "d_model": 7168}),
            # Kimi-K3's MLA a-norms: q-LoRA 1536, latent 512
            "fused_q_kv_rmsnorm": ([[2, 1, 1536], [2, 1, 512]],
                                   {"B": 2, "s_query": 1, "d_q_lora": 1536,
                                    "d_kv_lora": 512}),
            # Qwen3-8B's per-head Q/K norms, decode: 32 / 8 heads of 128
            "fused_q_kv_rmsnorm#per-head": (
                [[2, 32, 1, 128], [2, 8, 1, 128]],
                {"B": 2, "s_query": 1, "n_heads": 32, "n_kv_heads": 8,
                 "d_head": 128}),
            # Gemma-4-12B sliding layer: 16 query heads, 8 KV heads of 256
            "gemma_qkv_rmsnorm": ([[2, 16, 1, 256], [2, 8, 1, 256],
                                   [2, 8, 1, 256]],
                                  {"B": 2, "s_query": 1, "n_heads": 16,
                                   "n_kv_heads": 8, "d_head": 256}),
            "gemma_rmsnorm_residual_scalar": (
                [[2, 1, 3840], [2, 1, 3840], [1]],
                {"B": 2, "s_query": 1, "d_model": 3840}),
        }
        ports = {"fused_eh_norm": _ports("inputs_embeds", "hidden"),
                 "fused_q_kv_rmsnorm": _ports("q", "kv"),
                 "gemma_qkv_rmsnorm": _ports("q", "k", "v"),
                 "gemma_rmsnorm_residual_scalar": _ports(
                     "x", "residual", "scalar")}
        for case, (shape_in, want) in cases.items():
            op = case.split("#")[0]
            got = identity_shape_from_shape_in(op, shape_in,
                                               operand_ports=ports[op])
            assert got is not None and json.loads(got) == want, (case, got)

    @pytest.mark.regression          # walks three models
    def test_the_plan_reaches_each_span(self):
        expect = (
            ("deepseek-ai/DeepSeek-V3.2", "sglang", "fused_eh_norm",
             {"d_model": 7168}),
            ("deepseek-ai/DeepSeek-V3.2", "vllm", "fused_eh_norm",
             {"d_model": 7168}),
            ("moonshotai/Kimi-K3", "vllm", "fused_q_kv_rmsnorm",
             {"d_q_lora": 1536, "d_kv_lora": 512}),
            # sglang's per-head form (`fused_inplace_qknorm`): Qwen3-8B's 32 /
            # 8 heads of 128, GLM-4.5's 96 / 8.
            ("Qwen/Qwen3-8B", "sglang", "fused_q_kv_rmsnorm",
             {"n_heads": 32, "n_kv_heads": 8, "d_head": 128}),
            ("zai-org/GLM-4.5", "sglang", "fused_q_kv_rmsnorm",
             {"n_heads": 96, "n_kv_heads": 8, "d_head": 128}),
            ("google/gemma-4-12B-it", "sglang", "gemma_qkv_rmsnorm",
             {"d_head": 256}),
            ("google/gemma-4-12B-it", "sglang",
             "gemma_rmsnorm_residual_scalar", {"d_model": 3840}),
        )
        for model, fw, sig, want in expect:
            for phase in ("prefill", "decode"):
                jobs, dropped = _plan(model, phase, fw)
                assert sig not in dropped, (model, fw, phase)
                cells = [j["dims"] for j in jobs if j["signature"] == sig]
                assert cells, (model, fw, phase, sig)
                assert any(all(c.get(k) == v for k, v in want.items())
                           for c in cells), (model, fw, phase, cells)


class TestElementwiseGates:
    """The attention output gate (`sigmoid_gate_mul`) and the final-logit
    soft-cap, each one elementwise launch in sglang."""

    SOFTCAP: ClassVar[str] = "softcap_" + "logits"

    def test_each_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        want = {"sigmoid_gate_mul": {"sglang_fused_sigmoid_mul",
                                     "sglang_sigmoid_gate_mul",
                                     "torch_sigmoid_gate_mul"},
                self.SOFTCAP: {"sglang_softcap_to_float32",
                               "sglang_softcap_inplace", "torch_softcap"}}
        for sig, impls in want.items():
            assert bound[sig] == sig
            ps = _providers(sig)
            assert {p.kernel_impl for p in ps} == impls, sig
            assert all(p.fused_op == sig for p in ps), sig

    def test_the_two_softcap_kernels_split_by_the_logits_dtype(self):
        """sglang upcasts bf16 / fp16 logits inside the cap and caps fp32
        logits in place (logits_processor.py:948-962): one kernel each. Both
        WRITE fp32, so a cell billed at another output dtype is declined —
        the walk states no output dtype today (every row defaults to bf16),
        which is the plan-side gap this refusal keeps visible."""
        dims = {"B": 1, "s_query": 2, "d_model": 64}

        def serving(act, out):
            inv = Invocation(op=self.SOFTCAP, signature=self.SOFTCAP,
                             phase="decode", dims=dims, tp=1,
                             dtype_env={"activation": act, "output": out})
            return {p.kernel_impl for p in _providers(self.SOFTCAP)
                    if p.library == "sglang" and p.serves.accepts(inv)}
        assert serving("bf16", "fp32") == {"sglang_softcap_to_float32"}
        assert serving("fp32", "fp32") == {"sglang_softcap_inplace"}
        # Billed at bf16 (the walk's default), neither kernel writes the
        # dtype the row would record.
        assert serving("bf16", "bf16") == set()
        assert serving("fp32", "bf16") == set()

    def test_the_oracles_are_the_plans_chains(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import SOFTCAP, get_ref
        dims = {"B": 1, "s_query": 3, "d_model": 64}
        gate = get_ref("sigmoid_gate_mul")
        g = build_inputs(gate, _inv("sigmoid_gate_mul", dims), "cpu")
        want = g["a"].float() / (1 + torch.exp(-g["b"].float()))
        assert (gate.reference(g) - want).abs().max() < 1e-5
        cap = get_ref(self.SOFTCAP)
        c = build_inputs(cap, _inv(self.SOFTCAP, dims), "cpu")
        got = cap.reference(c)
        # bounded by the cap, odd, and the identity near zero
        assert c["cap"] == SOFTCAP and got.abs().max() < SOFTCAP
        assert (got - c["x"].float()).abs()[c["x"].float().abs() < 0.5].max() < 1e-3
        assert torch.allclose(cap.reference({**c, "x": -c["x"]}), -got)

    def test_each_floor_runs_the_chain_at_the_cells_width(self):
        for sig in ("sigmoid_gate_mul", self.SOFTCAP):
            _floor_meets_the_oracle(sig, {"B": 2, "s_query": 3, "d_model": 64})

    @pytest.mark.regression          # walks three models
    def test_the_plan_reaches_each_span(self):
        expect = (("Qwen/Qwen3.5-35B-A3B", "sigmoid_gate_mul", 4096),
                  ("tencent/Hy4-preview", "sigmoid_gate_mul", 16384),
                  ("google/gemma-2-9b", self.SOFTCAP, 256000))
        for model, sig, width in expect:
            for phase in ("prefill", "decode"):
                jobs, dropped = _plan(model, phase, "sglang")
                assert sig not in dropped, (model, phase)
                cells = {j["dims"]["d_model"] for j in jobs
                         if j["signature"] == sig}
                assert width in cells, (model, phase, cells)


class TestRetrievalGates:
    """DeepSeek-V4.1 Engram and Qwen PLE: the normalized-dot retrieval gates
    that add a learned table value into a hyper-connection stream."""

    ENGRAM_DIMS: ClassVar[dict] = {
        "B": 2, "s_query": 5, "n_hidden_copies": 3, "d_model": 17}
    #: Unequal copy count and width: a gate that normalizes across copies or
    #: broadcasts the value on the wrong axis cannot pass this cell.
    PLE_DIMS: ClassVar[dict] = {
        "B": 2, "s_query": 5, "n_hidden_copies": 3, "d_model": 17}

    @staticmethod
    def _rms(x):
        torch = pytest.importorskip("torch")
        y = x.float()
        return y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)

    @staticmethod
    def _fake_module(monkeypatch, name, **funcs):
        import sys
        import types
        mod = types.ModuleType(name)
        for key, value in funcs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    def test_each_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert bound["fused_engram_gate"] == "fused_engram_gate"
        assert bound["ple_gate"] == "ple_gate"
        assert {p.library for p in _providers("fused_engram_gate")} == {
            "vllm", "sglang", "deepinfer", "torch"}
        assert {p.kernel_impl for p in _providers("ple_gate")} == {
            "vllm_ple_gate_bf16", "vllm_ple_gate_fp16",
            "torch_ple_gate_bf16", "torch_ple_gate_fp16"}
        assert all(p.fused_op == "fused_engram_gate"
                   for p in _providers("fused_engram_gate"))
        assert all(p.fused_op == "ple_gate" for p in _providers("ple_gate"))

    def test_the_engram_oracle_gates_each_copy_independently(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inputs = build_inputs(
            get_ref("fused_engram_gate"),
            _inv("fused_engram_gate", self.ENGRAM_DIMS), "cpu")
        got = get_ref("fused_engram_gate").reference(inputs)
        x = inputs["x"].float().view(2, 5, 3, 17)
        key, value = inputs["kv"].float().split([51, 17], dim=-1)
        key = key.view(2, 5, 3, 17)
        h, k = self._rms(x), self._rms(key)
        dot = ((h * inputs["q_weight"].float())
               * (k * inputs["k_weight"].float())).sum(-1) / 17 ** 0.5
        gate = torch.sigmoid(torch.copysign(
            dot.abs().clamp_min(1e-6).sqrt(), dot))
        want = x + gate.unsqueeze(-1) * value.unsqueeze(-2)
        assert torch.allclose(got, want.flatten(-2), atol=1e-5)

    def test_the_ple_oracle_writes_both_ports(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inputs = build_inputs(get_ref("ple_gate"),
                              _inv("ple_gate", self.PLE_DIMS), "cpu")
        gated, conv = get_ref("ple_gate").reference(inputs).unbind(dim=-2)
        d, hc = 17, 3

        # Recompute from the plan's nodes, not the shared helper.
        kq = (self._rms(inputs["key"].float().view(2, 5, hc, d))
              * (1 + inputs["w_key"].float()))
        qq = (self._rms(inputs["x"].float().view(2, 5, hc, d))
              * (1 + inputs["w_query"].float()))
        dot = (kq * qq).view(2, 5, hc, d).sum(-1) / d ** 0.5
        gate = torch.sigmoid(torch.copysign(
            dot.abs().clamp_min(1e-6).sqrt(), dot))
        want_gated = gate.unsqueeze(-1) * inputs["value"].float().unsqueeze(-2)
        want_conv = (self._rms(want_gated)
                     * (1 + inputs["w_conv"].float())).flatten(-2)
        assert torch.allclose(gated, want_gated.flatten(-2), atol=1e-5)
        assert torch.allclose(conv, want_conv, atol=1e-5)

    def test_each_floor_runs_its_chain_at_the_cells_width(self):
        _floor_meets_the_oracle(
            "fused_engram_gate", self.ENGRAM_DIMS,
            env={"activation": "bf16", "weight": "bf16", "output": "bf16"})
        _floor_meets_the_oracle(
            "ple_gate", self.PLE_DIMS,
            env={"activation": "bf16", "weight": "bf16", "output": "bf16"})

    def test_vllm_launches_the_engram_grid_one_token_copy_at_a_time(
            self, monkeypatch):
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        seen = {}

        class Kernel:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    seen.update(grid=grid, args=args, kwargs=kwargs)
                    return torch.empty_like(args[0])
                return run

        self._fake_module(
            monkeypatch, "vllm.models.deepseek_v41.common.engram",
            _fused_engram_post_wkv_kernel=Kernel())
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        inv = _inv("fused_engram_gate", self.ENGRAM_DIMS)
        inputs = build_inputs(get_ref("fused_engram_gate"), inv, "cpu")
        na._vllm_engram_build(inv, TORCH_ENV, inputs).run()
        assert seen["grid"] == (2 * 5 * 3,)
        x, kv, qw, _kw, mask, out, num = seen["args"][:7]
        assert x.shape == (10, 3, 17) and x.is_contiguous()
        assert kv.shape == (10, 68) and kv.is_contiguous()
        assert torch.equal(qw, inputs["q_weight"])
        assert mask is x and out.shape == x.shape
        assert num == 10 and seen["kwargs"]["HAS_MASK"] is False

    def test_each_engram_arm_hands_its_rows_back_at_the_oracles_shape(
            self, monkeypatch):
        """Every framework launch writes `[tokens, hc, d]`; the oracle is
        `[B, s_query, hc*d]`. A canonicalisation that flattened both ends
        returned a 1-D vector, and the vLLM and sglang canaries failed the
        cross-check on shape alone (DeepSeek-V4.1-Flash, H100)."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.deepinfer as di
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref

        class Kernel:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: None

        def written(x, *args, **kwargs):
            return torch.empty_like(x)

        self._fake_module(
            monkeypatch, "vllm.models.deepseek_v41.common.engram",
            _fused_engram_post_wkv_kernel=Kernel())
        self._fake_module(
            monkeypatch, "sglang.kernels.ops.embeddings.engram_gate",
            fused_engram_gate=written)
        self._fake_module(monkeypatch, "tile_kernels")
        self._fake_module(
            monkeypatch, "tile_kernels.engram",
            fused_weight=lambda q, k: torch.cat([q, k]),
            engram_gate_fwd=lambda x, *a, **k: (x,))
        import sys
        sys.modules["tile_kernels"].engram = sys.modules["tile_kernels.engram"]
        for name in ("deepinfer", "deepinfer.model_executor",
                     "deepinfer.model_executor.models",
                     "deepinfer.model_executor.models.v4_modules",
                     "deepinfer.model_executor.models.v4_modules.engram"):
            self._fake_module(monkeypatch, name)
        for module in (na, di):
            monkeypatch.setattr(module, "_require_cuda", lambda: None)
        inv = _inv("fused_engram_gate", self.ENGRAM_DIMS)
        inputs = build_inputs(get_ref("fused_engram_gate"), inv, "cpu")
        want = get_ref("fused_engram_gate").reference(inputs).shape
        for build in (na._vllm_engram_build, na._sgl_engram_build,
                      di._engram_build):
            case = build(inv, TORCH_ENV, inputs)
            assert case.to_canonical(case.run()).shape == want, build

    def test_vllm_hands_ple_the_split_projection_views(self, monkeypatch):
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        seen = {}

        def ple_gate(key, value, hidden, wk, wq, wc, eps):
            seen.update(key=key, value=value, hidden=hidden, weights=(wk, wq, wc))
            return torch.empty_like(hidden), torch.empty_like(hidden)

        self._fake_module(
            monkeypatch, "vllm.models.qwen4_exp.common.ops.ple",
            ple_gate=ple_gate)
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        inv = _inv("ple_gate", self.PLE_DIMS)
        inputs = build_inputs(get_ref("ple_gate"), inv, "cpu")
        na._vllm_ple_build(inv, TORCH_ENV, inputs).run()
        key, value, hidden = seen["key"], seen["value"], seen["hidden"]
        assert key.stride() == value.stride() == (68, 1)
        assert not key.is_contiguous() and not value.is_contiguous()
        assert value.storage_offset() - key.storage_offset() == 51
        assert torch.equal(key, inputs["key"].flatten(0, 1))
        assert torch.equal(value, inputs["value"].flatten(0, 1))
        assert hidden.is_contiguous()

    @pytest.mark.regression          # walks DeepSeek-V4.1 and Qwen4Exp
    def test_the_plans_reach_each_gate(self):
        from backend.inference.framework_bindings import (ServingDeployment,
                                                          get_binding)
        from kernel_bench.plan import build_jobs
        for fw in ("vllm", "sglang"):
            binding = get_binding(fw).deployed(ServingDeployment.at(
                gpu_sku="H100-SXM", activation_dtype="BF16"))
            jobs, _dropped = build_jobs(
                avg_output_tokens=128, model_id="deepseek-ai/DeepSeek-V4.1-Flash",
                batch_size=2, seq_len=1024, phase="prefill", framework=fw,
                binding=binding, gpu="H100-SXM")
            cells = [j["dims"] for j in jobs
                     if j["signature"] == "fused_engram_gate"]
            assert cells and cells[0]["n_hidden_copies"] == 4, (fw, cells)
        jobs, _dropped = _plan("Qwen/Qwen3.8-Flash-Next", "prefill", "vllm")
        cells = [j["dims"] for j in jobs if j["signature"] == "ple_gate"]
        assert cells and cells[0]["n_hidden_copies"] == 4, cells


class TestDsaPrologue:
    """DSA attention's input preparation: vLLM's `fused_norm_rope` (the two
    a-norms and the rope key, plus the indexer key on an indexer layer) and
    sglang's two rope-first indexer launches (the key into the paged FP8
    index cache, the query FP8 with its scale folded into the head
    weights)."""

    SPANS: ClassVar[dict] = {
        "fused_dsa_qkv_norm_rope": {"vllm", "torch"},
        "fused_dsa_qkv_norm_rope_indexer": {"vllm", "torch"},
        "fused_indexer_k_norm_rope_store": {"sglang", "torch"},
        "fused_indexer_q_rope_first_quant": {"sglang", "torch"},
        "fused_dsa_q_rope_quant": {"vllm", "torch"},
    }
    DIMS: ClassVar[dict] = {"B": 2, "s_query": 3, "d_q_lora": 96,
                            "d_kv_lora": 32, "d_rope": 16,
                            "d_indexer_head": 32, "n_indexer_heads": 4,
                            "n_topk_keys": 64, "kv_block_size": 4,
                            "n_heads": 4, "d_qk_nope": 24}

    def test_each_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        for sig, libs in self.SPANS.items():
            assert bound[sig] == sig
            ps = _providers(sig)
            assert {p.library for p in ps} == libs, sig
            assert all(p.fused_op == sig for p in ps), sig

    def test_the_indexer_layer_is_the_shared_layer_plus_the_indexer_key(self):
        """The two spans are one kernel with `has_indexer` the switch: on the
        same operands the indexer form's first ports are the shared form's."""
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        full = get_ref("fused_dsa_qkv_norm_rope_indexer")
        shared = get_ref("fused_dsa_qkv_norm_rope")
        inputs = build_inputs(full, _inv(
            "fused_dsa_qkv_norm_rope_indexer", self.DIMS), "cpu")
        a = full.reference(inputs)
        b = shared.reference({k: v for k, v in inputs.items()
                              if not k.startswith("ik")})
        assert a.shape[1] == b.shape[1] + 32
        assert (a[:, :b.shape[1]] - b).abs().max() == 0

    def test_the_rope_key_rotates_adjacent_pairs_by_its_position(self):
        """k_pe is rotated GPT-J style: channel pairs (2j, 2j+1) by angle
        pos·θ_j. Recomputed from the pair definition, position by position."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("fused_dsa_qkv_norm_rope")
        dims = {k: self.DIMS[k] for k in
                ("B", "s_query", "d_q_lora", "d_kv_lora", "d_rope")}
        inputs = build_inputs(spec, _inv("fused_dsa_qkv_norm_rope", dims), "cpu")
        got = spec.reference(inputs)[:, 96 + 32:]
        x = inputs["k_pe"].float()
        theta = 10000.0 ** (-torch.arange(0, 16, 2).float() / 16)
        for t in range(x.shape[0]):
            ang = inputs["positions"][t].float() * theta
            c, s = torch.cos(ang), torch.sin(ang)
            even, odd = x[t, 0::2], x[t, 1::2]
            want = torch.stack([even * c - odd * s, odd * c + even * s],
                               dim=-1).flatten()
            assert (got[t] - want).abs().max() < 1e-4, t

    def test_the_indexer_preps_are_the_fp8_grid_of_the_rotated_rows(self):
        """Each record is FP8 e4m3 with one exact scale: every value is a
        representable e4m3 level times the row's amax/448."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        for sig in ("fused_indexer_k_norm_rope_store",
                    "fused_indexer_q_rope_first_quant"):
            spec = get_ref(sig)
            inputs = build_inputs(spec, _inv(sig, self.DIMS), "cpu")
            rows = spec.reference(inputs).reshape(-1, 32)
            scale = rows.abs().amax(-1, keepdim=True) / 448.0
            levels = rows / scale
            on_grid = levels.to(torch.float8_e4m3fn).float()
            assert (levels - on_grid).abs().max() < 1e-4 * 448, sig

    def test_each_floor_runs_the_chain_at_the_cells_width(self):
        for sig in self.SPANS:
            _floor_meets_the_oracle(sig, self.DIMS)

    def test_the_index_k_store_writes_only_the_fp8_record(self):
        """The store kernel writes the 128-value FP8 e4m3 record with one
        fp32 scale a key (indexer_k.cuh:350) whatever the cell declares, so
        a cell whose cache is billed at another dtype is declined through
        the DECLARATION — where smoke, the published capability record and
        the stale-row audit all read it — not timed at one precision and
        recorded as another."""
        sig = "fused_indexer_k_norm_rope_store"
        sgl = next(p for p in _providers(sig) if p.library == "sglang")
        dims = {**self.DIMS, "d_indexer_head": 128, "d_rope": 64}

        def inv(output, record=None):
            env = {"activation": "bf16"}
            if output is not None:
                env["output"] = output
            if record is not None:
                env["output_format"] = record
            return Invocation(op=sig, signature=sig, phase="prefill",
                              dims=dims, tp=1, dtype_env=env)
        # The cached key IS the launch's output, so its record is the
        # cell's `output` dtype and `output_format` (`RowIdentity.
        # leaf_dtype_env`), not `kv`: one exact fp32 scale a 128-wide key.
        assert sgl.serves.accepts(inv("fp8_e4m3", "fp8_group128"))
        assert not sgl.serves.accepts(inv("bf16"))
        assert not sgl.serves.accepts(inv(None))
        assert list(sgl.serves.refusals(inv("bf16"))) == [
            "output:bf16 is not served (serves fp8_e4m3)",
            "output_format:- is not served (serves fp8_group128)"]
        # Another FP8 record over the same dtype is another launch's:
        # vLLM's power-of-two scale, DeepSeek-V4's FlashMLA record.
        for other in ("fp8_pow2_group128", "fp8_ue8m0_group64_bf16_rope_tail"):
            assert list(sgl.serves.refusals(inv("fp8_e4m3", other))) == [
                f"output_format:{other} is not served "
                "(serves fp8_group128)"]
        # Published, not buried in a build hint only the driver resolves.
        assert sgl.impl_params is None
        assert sgl.capabilities()["output_dtypes"] == ["fp8_e4m3"]
        assert sgl.capabilities()["output_formats"] == ["fp8_group128"]

    def test_the_cache_writing_overloads_declare_their_cache_format(self):
        """The same permanent launch fact, on the other kernels that carry
        it: vLLM's q-norm/rope/insert packed-UE8M0 overload and the two
        FlashMLA sparse paths each serve exactly the cache dtype their
        build allocates."""
        from kernel_bench.providers import all_providers
        want = {"vllm_c_fused_qnorm_rope_kv_insert": "fp8_e4m3",
                "vllm_c_fused_qnorm_rope_kv_insert_no_q_norm": "fp8_e4m3",
                "flashmla_sparse_fp8": "fp8_e4m3",
                "flashmla_sparse": "bf16"}
        for impl, kv in want.items():
            p = next(p for p in all_providers() if p.kernel_impl == impl)
            assert dict(p.serves.dtype_slots) == {"kv": (kv,)}, impl
            assert p.impl_params is None, impl
            assert p.capabilities()["kv_dtypes"] == [kv], impl

    def test_the_indexer_query_leaves_the_kernel_as_fp8(self):
        """`fused_q_indexer_rope_first_quant` returns the FP8 e4m3 query
        (its scale folded into the head weights), so a cell billed at
        another output dtype is not this launch's to measure. And the
        launch's second write — the rescaled head weights, fp32
        (elementwise.py:257-270) — is declared beside the primary one, so
        the published contract does not under-describe what the measured
        launch writes."""
        sig = "fused_indexer_q_rope_first_quant"
        sgl = next(p for p in _providers(sig) if p.library == "sglang")
        dims = {**self.DIMS, "d_indexer_head": 128, "d_rope": 64}

        def serving(out, granularity="per-token"):
            # The engine keys this cell with the scale layout the launch
            # writes: one fp32 scale per (token, head) row.
            inv = Invocation(op=sig, signature=sig, phase="prefill",
                             dims=dims, tp=1,
                             dtype_env={"activation": "bf16", "output": out,
                                        "metadata": "fp32",
                                        "activation_granularity": granularity})
            return sgl.serves.accepts(inv)
        assert serving("fp8_e4m3")
        assert not serving("bf16")
        # A cell that names no scale layout is not what this launch writes.
        assert not serving("fp8_e4m3", granularity="")
        assert sgl.serves.secondary_outputs == (("weights", "fp32"),)
        assert sgl.capabilities()["secondary_outputs"] == [
            {"tensor": "weights", "dtype": "fp32"}]

    QUANT_ENV: ClassVar[dict] = {"activation": "bf16", "output": "fp8_e4m3",
                                 "activation_granularity": "per-token",
                                 "metadata": "fp32"}

    def test_the_dsa_query_floor_writes_the_cells_record(self):
        """The query span's torch floor at its quantised cell — the indexer
        query on the cell's per-head FP8 grid — beside the plain cell the
        default floor test reads (an index key kept BF16 grafts no cast)."""
        _floor_meets_the_oracle("fused_dsa_q_rope_quant", self.DIMS,
                                output="fp8_e4m3", env=self.QUANT_ENV)

    def test_vllm_hands_fused_q_the_serving_operands(self, monkeypatch):
        """vLLM `fused_q` (deepseek_v32/common/kernels.py:953) as
        `DeepseekV32Attention.forward` calls it (attention.py:433-447): q_pe
        the rope tail of the per-head `q_b_proj` row, the indexer query
        contiguous, the head weights the trailing slice of the wk/weights
        row, the BF16-query path. A stand-in computing the kernel's own
        arithmetic (GPT-J q_pe, the stated indexer pairing, the power-of-two
        FP8 scale folded into the weights, :62-74, :930-950) must land on
        the provider's declared oracle through the provider's canonical
        reading — and off the cell's exact-scale record, which is why that
        oracle exists."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.rope as rope
        from kernel_bench.refs import _rope_rotate, get_ref
        seen = {}

        def fused_q(positions, q_pe, q_pe_table, index_q, index_table,
                    ql_nope, q_scale, weights, softmax_scale, head_scale, *,
                    has_indexer, index_rope_interleave, quantize_mqa):
            seen.update(q_pe=q_pe, ql_nope=ql_nope, index_q=index_q,
                        weights=weights, has_indexer=has_indexer,
                        quantize_mqa=quantize_mqa, positions=positions)
            rd = int(q_pe_table.shape[-1])
            q_out = _rope_rotate(q_pe.float(), positions, q_pe_table, rd,
                                 is_neox=False).to(q_pe.dtype)
            rows = _rope_rotate(index_q.float(), positions, index_table, rd,
                                is_neox=not index_rope_interleave)
            amax = rows.abs().amax(-1).clamp_min(1e-4)
            scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
            fp8 = (rows / scale[..., None]).to(torch.float8_e4m3fn)
            w_out = weights.float() * scale * softmax_scale * head_scale
            return fp8, w_out, q_out

        _fake_module(monkeypatch, "vllm.models.deepseek_v32.common.kernels",
                     fused_q=fused_q)
        monkeypatch.setattr(rope, "_require_cuda", lambda: None)
        sig = "fused_dsa_q_rope_quant"
        vllm = next(p for p in _providers(sig) if p.library == "vllm")
        inv = Invocation(op=sig, signature=sig, phase="prefill", tp=1,
                         dims={a: self.DIMS[a] for a in
                               ("B", "s_query", "n_heads", "d_rope",
                                "n_indexer_heads", "d_indexer_head",
                                "d_qk_nope")},
                         dtype_env=dict(self.QUANT_ENV))
        assert vllm.serves.accepts(inv)
        inputs = build_inputs(get_ref(sig), inv, "cpu")
        case = vllm.build(inv, TORCH_ENV, inputs)
        got = case.to_canonical(case.run()).float()
        heads, nope, rd = 4, 24, 16
        assert seen["q_pe"].stride()[1:] == (nope + rd, 1)
        assert not seen["q_pe"].is_contiguous()
        assert seen["ql_nope"].shape[-1] == nope
        assert seen["index_q"].is_contiguous()
        assert seen["weights"].stride() == (32 + heads, 1)
        assert seen["has_indexer"] and not seen["quantize_mqa"]
        assert seen["positions"].dtype == torch.int64
        own = vllm.reference(inputs).float()
        assert (got - own).abs().max() <= 1e-2 * own.abs().max()
        exact = get_ref(sig).reference(inputs).float()
        assert (own - exact).abs().max() > 0

    def test_a_bare_norm_with_its_cast_is_the_norms_quantising_launch(self):
        """`fused_rmsnorm_quant` (the residual-free norm taking its quantised
        GEMM's cast) is measured under `rmsnorm`, told apart from the plain
        norm by the cell's output record — the cell vLLM's
        `rms_norm_per_block_quant` (rms_quant_fusion.py:463) serves, the
        kernel `RMSNormGroupQuantPattern` rewrites the pair into."""
        from backend.inference.op_signatures import op_bindings
        assert op_bindings()["fused_rmsnorm_quant"] == "rmsnorm"
        dims = {"B": 1, "s_query": 3, "d_model": 7168}
        quant = Invocation(
            op="fused_rmsnorm_quant", signature="rmsnorm", phase="prefill",
            tp=1, dims=dims,
            dtype_env={"activation": "bf16", "weight": "bf16",
                       "output": "fp8_e4m3",
                       "activation_granularity": "per-group:128",
                       "metadata": "fp32"})
        served = {p.kernel_impl for p in _providers("rmsnorm")
                  if p.serves.accepts(quant)}
        assert served == {"vllm_rms_norm_per_block_quant"}, served

    def test_the_cells_are_read_off_the_engine_operands(self):
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        glm = [[2, 1, 2048], [2, 1, 512], [2, 1, 64]]
        got = identity_shape_from_shape_in("fused_dsa_qkv_norm_rope", glm,
                                           operand_ports=_ports("q", "kv",
                                                              "k_pe"))
        assert json.loads(got) == {"B": 2, "s_query": 1, "d_q_lora": 2048,
                                   "d_kv_lora": 512, "d_rope": 64}
        # The indexer layer's top-k reset is no operand: its width is the
        # spec's, and without it the cell is not stated.
        indexer = [*glm, [2, 1, 128]]
        ports = _ports("q", "kv", "k_pe", "index_key")
        assert identity_shape_from_shape_in(
            "fused_dsa_qkv_norm_rope_indexer", indexer,
            operand_ports=ports) is None
        got = identity_shape_from_shape_in(
            "fused_dsa_qkv_norm_rope_indexer", indexer,
            spec_dims={"n_topk_keys": 2048}, operand_ports=ports)
        assert json.loads(got)["n_topk_keys"] == 2048

    @pytest.mark.regression          # walks two models
    def test_the_plan_reaches_each_span(self):
        from backend.inference.framework_bindings import (ServingDeployment,
                                                          get_binding)
        from kernel_bench.plan import build_jobs
        expect = (("deepseek-ai/DeepSeek-V3.2", "vllm",
                   ("fused_dsa_qkv_norm_rope_indexer",
                    "fused_dsa_q_rope_quant")),
                  ("zai-org/GLM-5.2", "vllm",
                   ("fused_dsa_qkv_norm_rope",
                    "fused_dsa_qkv_norm_rope_indexer")),
                  ("zai-org/GLM-5.2", "sglang",
                   ("fused_indexer_k_norm_rope_store",
                    "fused_indexer_q_rope_first_quant")))
        for model, fw, sigs in expect:
            binding = get_binding(fw).deployed(ServingDeployment.at(
                gpu_sku="H100-SXM", activation_dtype="BF16"))
            for phase in ("prefill", "decode"):
                # sglang states no page of its own; the index-K store is a
                # paged cell, so the sweep names one.
                jobs, _dropped = build_jobs(
                    avg_output_tokens=128, model_id=model, batch_size=2,
                    seq_len=1024, phase=phase, framework=fw, binding=binding,
                    kv_block_sizes=[64], gpu="H100-SXM")
                got = {j["signature"] for j in jobs}
                for sig in sigs:
                    assert sig in got, (model, fw, phase, sig)
                if "fused_dsa_q_rope_quant" in sigs:
                    self._dsa_query_cell_is_the_checkpoints(model, jobs)

    @staticmethod
    def _dsa_query_cell_is_the_checkpoints(model, jobs):
        """The DSA query span's two rank-4 operands are zipped by position
        (the engine hands a span's inputs over by member uid, not by the
        plan's port names), and a swap would bind every axis without a
        conflict. So its cell is read against the checkpoint config — a
        route that shares nothing with the projection: q_pe carries the
        attention heads and the MLA rope width, the indexer query the
        indexer's heads and head width, and the rotation is narrower than
        the indexer head it rotates the front of."""
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        cfg = json.loads((root / "backend/arch/snapshots"
                          / f"{model.replace('/', '__')}.json").read_text()
                         )["config"]
        cells = {tuple(sorted(j["dims"].items())) for j in jobs
                 if j["signature"] == "fused_dsa_q_rope_quant"}
        assert cells, model
        for cell in map(dict, cells):
            assert cell["n_heads"] == cfg["num_attention_heads"], cell
            assert cell["d_rope"] == cfg["qk_rope_head_dim"], cell
            assert cell["n_indexer_heads"] == cfg["index_n_heads"], cell
            assert cell["d_indexer_head"] == cfg["index_head_dim"], cell
            assert cell["d_qk_nope"] == cfg["qk_nope_head_dim"], cell
            assert cell["d_rope"] < cell["d_indexer_head"], cell


class TestSparseAttentionRoutes:
    """The two remaining sparse-attention spans: Hy4's learnable sink and
    Qwen QSA's output-gated unsplit prefill kernel."""

    SINK_DIMS: ClassVar[dict] = {
        "B": 2, "s_query": 5, "s_kv": 37, "n_heads": 5, "d_head": 576,
        "d_v_head": 512, "n_topk_keys": 11, "kv_block_size": 64}
    #: GQA with an odd group size and more than one request, so the wrong
    #: head/token axis cannot cancel against a 1×1 cell.
    QSA_DIMS: ClassVar[dict] = {
        "B": 2, "s_query": 5, "s_kv": 37, "n_heads": 6, "n_kv_heads": 2,
        "d_head": 16, "n_topk_keys": 11, "kv_block_size": 16}

    @staticmethod
    def _fake_module(monkeypatch, name, **funcs):
        import sys
        import types
        mod = types.ModuleType(name)
        for key, value in funcs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)

    def test_each_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert bound["flash_dsa_sink"] == "mla_sparse_sink_attention"
        assert bound["flash_dsa_gated"] == "gqa_sparse_gated_attention"
        assert {p.kernel_impl for p in _providers(
            "mla_sparse_sink_attention")} == {
            "vllm_flashmla_sparse_sink_bf16",
            "vllm_flashmla_sparse_sink_fp8_e4m3",
            "sglang_flashmla_sparse_sink", "torch_sparse_mla_sink"}
        assert {p.kernel_impl for p in _providers(
            "gqa_sparse_gated_attention")} == {
            "vllm_qsa_sparse_gated_bf16",
            "vllm_qsa_sparse_gated_fp8_e4m3",
            "torch_qsa_sparse_gated"}
        assert all(p.fused_op == "flash_dsa_sink"
                   for p in _providers("mla_sparse_sink_attention"))
        assert all(p.fused_op == "flash_dsa_gated"
                   for p in _providers("gqa_sparse_gated_attention"))

    def test_the_launch_scopes_are_the_kernels_compile_time_facts(self):
        sink = {p.kernel_impl: p for p in _providers(
            "mla_sparse_sink_attention")}
        def sink_inv(dtype_env, dims=None):
            return Invocation(
                op="mla_sparse_sink_attention",
                signature="mla_sparse_sink_attention", phase="decode",
                dims=dict(dims or self.SINK_DIMS), tp=1, dtype_env=dtype_env)

        good = {"activation": "bf16", "kv": "bf16", "weight": "fp32",
                "output": "bf16"}
        inv = sink_inv(good)
        assert sink["sglang_flashmla_sparse_sink"].serves.accepts(inv)
        assert sink["vllm_flashmla_sparse_sink_bf16"].serves.accepts(inv)
        wrong_weight = sink_inv({**good, "weight": "bf16"})
        assert not sink["sglang_flashmla_sparse_sink"].serves.accepts(
            wrong_weight)
        wrong_kv = sink_inv({**good, "kv": "fp8_e4m3"})
        assert not sink["sglang_flashmla_sparse_sink"].serves.accepts(wrong_kv)
        wrong_geometry = dict(self.SINK_DIMS, d_head=512, d_v_head=256)
        wrong = sink_inv(good, wrong_geometry)
        assert not sink["vllm_flashmla_sparse_sink_bf16"].serves.accepts(wrong)

        qsa = next(p for p in _providers("gqa_sparse_gated_attention")
                   if p.library == "vllm")
        qinv = Invocation(
            op="gqa_sparse_gated_attention",
            signature="gqa_sparse_gated_attention", phase="prefill",
            dims=dict(self.QSA_DIMS), tp=1,
            dtype_env={"activation": "bf16", "kv": "bf16",
                       "output": "bf16"})
        assert qsa.serves.accepts(qinv)
        decode = Invocation(**{**qinv.__dict__, "phase": "decode"})
        fp16 = Invocation(**{**qinv.__dict__, "dtype_env": {
            **qinv.dtype_env, "activation": "fp16"}})
        assert not qsa.serves.accepts(decode)
        assert not qsa.serves.accepts(fp16)

    def test_the_sparse_cache_is_the_cells_and_the_walks(self):
        """The cache a sparse-MLA build allocates and the rows each query
        gathers are the ones the walk bills: the cache operand's rows (the
        keys a query attends), bound by the same engine layout the identity
        reads (`_dsa_identity`). GLM-5.3-Flash over a 1152-key cache selects
        288 pools of 4; the cell was keyed at the 288, and the launch built
        and gathered 288 rows of the 1152 the row bills."""
        pytest.importorskip("torch")
        from backend.inference.benchmark_identity import _dsa_identity
        from backend.inference.op_signatures import get_signature
        from kernel_bench.refs import get_ref
        import json
        # The row keys the latent as stored (`stored_operands` drops the
        # broadcast head axis the walk marks headless).
        # The latent is the span's K and its V, one operand of both ports.
        stored = [[2, 4, 1, 512], [2, 1152, 512], [2, 1, 288]]
        ports = _ports("q", ("k", "v"), "mask")
        for shape_in in (stored,):
            dims = json.loads(_dsa_identity(
                get_signature("mla_sparse_attention").at_ports(ports),
                shape_in, [2, 4, 1, 512],
                {"n_topk_keys": 2048, "d_kv_lora": 512}))
            assert (dims["s_kv"], dims["n_topk_keys"]) == (1152, 1152)
            inv = Invocation(op="flash_dsa", signature="mla_sparse_attention",
                             phase="decode", tp=1,
                             dims={**dims, "kv_block_size": 64},
                             dtype_env={"activation": "bf16", "kv": "bf16"},
                             shape_in=shape_in, operand_ports=ports)
            inputs = build_inputs(get_ref("mla_sparse_attention"), inv, "cpu")
            assert tuple(inputs["c"].shape) == (2, 1152, 512)
            assert tuple(inputs["indices"].shape) == (2, 1, 1152)
            # Every row once, in a scattered order: serving's top-k gathers
            # rows spread over the cache, not one sequential run.
            row = inputs["indices"][0, 0].tolist()
            from itertools import pairwise
            assert sorted(row) == list(range(1152))
            adjacent = sum(abs(b - a) == 1 for a, b in pairwise(row))
            assert adjacent < 1152 // 8, adjacent

    #: Top-k GQA: an odd group and two KV heads, so a wrong head/group axis
    #: cannot cancel against a 1x1 cell.
    GQA_SPARSE_DIMS: ClassVar[dict] = {
        "B": 2, "s_query": 5, "s_kv": 37, "n_heads": 6, "n_kv_heads": 2,
        "d_head": 16, "d_v_head": 16, "n_topk_keys": 37}

    def test_the_per_head_geometry_has_its_own_floor(self):
        """Qwen QSA's split decode and Step-5 carry K and V per KV head
        (`mla_sparse_attention`'s `n_kv_heads` geometry). The latent kernels
        decline that cell, the GQA floor declines the latent one, and the
        floor computes what the oracle does."""
        pytest.importorskip("torch")
        from kernel_bench.providers.attention import _torch_sparse_gqa_build
        from kernel_bench.refs import get_ref
        inv = Invocation(op="flash_dsa", signature="mla_sparse_attention",
                         phase="decode", tp=1, dims=dict(self.GQA_SPARSE_DIMS),
                         dtype_env={"activation": "bf16", "kv": "bf16"})
        latent = Invocation(**{**inv.__dict__, "dims": {
            "B": 2, "s_query": 5, "s_kv": 37, "n_heads": 6, "d_head": 576,
            "d_v_head": 512, "n_topk_keys": 37}})
        by_impl = {p.kernel_impl: p for p in _providers("mla_sparse_attention")}
        floor = by_impl.pop("torch_gather_sparse_gqa")
        assert floor.serves.accepts(inv)
        assert not floor.serves.accepts(latent)
        for impl, p in by_impl.items():
            assert any(token.startswith("variant_dims:n_kv_heads")
                       for token in p.serves.refusals(inv)), impl
        spec = get_ref("mla_sparse_attention")
        inputs = build_inputs(spec, inv, "cpu")
        assert tuple(inputs["k"].shape) == (2, 37, 2, 16)
        want = spec.reference(inputs)
        got = _torch_sparse_gqa_build(inv, TORCH_ENV, inputs).run()
        assert tuple(got.shape) == (2, 5, 6, 16)
        assert (got.float() - want).abs().max().item() < 2e-2

    def test_the_gqa_floor_reads_what_repeating_kv_read(self):
        """The floor broadcasts each KV head over its group instead of
        repeat_interleave-ing K/V to one copy per query head (48 GiB at
        QSA's prefill cell). Same values in the same order, so at bf16 the
        output is bit-identical to the old spelling, not only near the
        oracle."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.attention import _torch_sparse_gqa_build
        from kernel_bench.refs import get_ref
        inv = Invocation(op="flash_dsa", signature="mla_sparse_attention",
                         phase="decode", tp=1, dims=dict(self.GQA_SPARSE_DIMS),
                         dtype_env={"activation": "bf16", "kv": "bf16"})
        inputs = build_inputs(get_ref("mla_sparse_attention"), inv, "cpu")
        got = _torch_sparse_gqa_build(inv, TORCH_ENV, inputs).run()
        q, k, v = (inputs[n].to(torch.bfloat16) for n in ("q", "k", "v"))
        B, Sq, N, D = q.shape
        _B, Lq, Nkv, _D = k.shape
        idx = inputs["indices"].long()
        topk = idx.shape[2]
        sel = idx.unsqueeze(-1).unsqueeze(-1)
        ks = torch.gather(k.unsqueeze(1).expand(B, Sq, Lq, Nkv, D), 2,
                          sel.expand(B, Sq, topk, Nkv, D)).permute(
            0, 1, 3, 2, 4).repeat_interleave(N // Nkv, dim=2)
        vs = torch.gather(v.unsqueeze(1).expand(B, Sq, Lq, Nkv, D), 2,
                          sel.expand(B, Sq, topk, Nkv, D)).permute(
            0, 1, 3, 2, 4).repeat_interleave(
            N // Nkv, dim=2).permute(0, 2, 1, 3, 4)
        scores = torch.einsum("bsnd,bsntd->bnst", q, ks)
        attn = scores.float().mul_(D ** -0.5).softmax(dim=-1).to(v.dtype)
        old = torch.einsum("bnst,bnstd->bnsd", attn, vs).transpose(1, 2)
        assert torch.equal(got, old.to(got.dtype))

    def test_the_sink_joins_the_softmax_and_leaves_no_value_contribution(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inputs = build_inputs(get_ref("mla_sparse_sink_attention"),
                              _inv("mla_sparse_sink_attention", self.SINK_DIMS,
                                   phase="decode"), "cpu")
        got = get_ref("mla_sparse_sink_attention").reference(inputs)
        q, c = inputs["q"].float(), inputs["c"].float()
        idx = inputs["indices"].long()
        sel = torch.gather(c.unsqueeze(1).expand(2, 5, 37, 576), 2,
                           idx.unsqueeze(-1).expand(2, 5, 11, 576))
        scores = torch.einsum("bsnd,bstd->bnst", q, sel) / 576 ** 0.5
        scores = torch.cat([scores, inputs["sinks"].float().view(
            1, 5, 1, 1).expand(2, 5, 5, 1)], dim=-1)
        p = scores.softmax(-1)[..., :11]
        want = torch.einsum("bnst,bstd->bnsd", p, sel[..., :512])
        assert torch.allclose(got, want.transpose(1, 2).contiguous(),
                              atol=1e-5)

    def test_the_sink_floor_attends_at_the_cells_precision(self):
        """The torch sink floor used to time the fp32 oracle itself — a row
        of fp32 work billed as a bf16 cell, whose cross-check compared the
        oracle with itself. At a bf16 cell (bf16 sinks first: there an fp32
        launch is refused outright) it meets the oracle inside the bf16
        band and its launch is the cell's precision; with fp32 sinks the
        same. Remove the sink from its softmax and it is off the oracle."""
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        dims = dict(self.SINK_DIMS, B=1)
        for weight in ("bf16", "fp32"):
            # The launch gate inside refuses an fp32 launch at the bf16 cell.
            _floor_meets_the_oracle(
                "mla_sparse_sink_attention", dims,
                env={"activation": "bf16", "kv": "bf16", "weight": weight,
                     "output": "bf16"})
        spec = get_ref("mla_sparse_sink_attention")
        inv = _inv("mla_sparse_sink_attention", dims, phase="decode")
        inputs = build_inputs(spec, inv, "cpu")
        floor = next(p for p in _providers("mla_sparse_sink_attention")
                     if p.library == "torch")
        no_sink = {k: v for k, v in inputs.items() if k != "sinks"}
        plain = floor.build(inv, TORCH_ENV, no_sink).run().float()
        want = spec.reference(inputs)
        assert (plain - want).abs().max() > 1e-2

    def test_the_qsa_oracle_gates_after_the_sparse_gqa_apply(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        inputs = build_inputs(get_ref("gqa_sparse_gated_attention"),
                              _inv("gqa_sparse_gated_attention", self.QSA_DIMS),
                              "cpu")
        got = get_ref("gqa_sparse_gated_attention").reference(inputs)
        q, k, v, gate = (inputs[n].float() for n in ("q", "k", "v", "gate"))
        idx = inputs["indices"].long()
        exp = idx.unsqueeze(-1).unsqueeze(-1).expand(2, 5, 11, 2, 16)
        ks = torch.gather(k.unsqueeze(1).expand(2, 5, 37, 2, 16), 2, exp
                          ).permute(0, 1, 3, 2, 4).repeat_interleave(3, dim=2)
        vs = torch.gather(v.unsqueeze(1).expand(2, 5, 37, 2, 16), 2, exp
                          ).permute(0, 1, 3, 2, 4).repeat_interleave(
                              3, dim=2).permute(0, 2, 1, 3, 4)
        scores = torch.einsum("bsnd,bsntd->bnst", q, ks) / 16 ** 0.5
        attn = scores.softmax(-1)
        o = torch.einsum("bnst,bnstd->bnsd", attn, vs).transpose(1, 2)
        assert torch.allclose(got, o * torch.sigmoid(gate), atol=1e-5)

    #: The latent geometry the gather floor serves (`n_kv_heads` absent),
    #: small enough that the single-shot gather is the honest launch.
    GATHER_DIMS: ClassVar[dict] = {
        "B": 2, "s_query": 5, "s_kv": 37, "n_heads": 6, "d_head": 24,
        "d_v_head": 16, "n_topk_keys": 11}

    def test_the_gather_floor_blocks_the_query_axis_only_past_the_budget(
            self, monkeypatch):
        """Past `refs.GATHER_BLOCK_BYTES` the gather floor walks the query axis
        in blocks (`refs._mla_sparse_reference`'s budget, at the cell's own
        dtype) — and a block walk computes what the one-shot launch computes,
        which is what the oracle computes."""
        pytest.importorskip("torch")
        import kernel_bench.providers.attention as att
        from kernel_bench.refs import get_ref
        inv = _inv("mla_sparse_attention", self.GATHER_DIMS)
        spec = get_ref("mla_sparse_attention")
        inputs = build_inputs(spec, inv, "cpu")
        want = spec.reference(inputs)
        one_shot = att._torch_sparse_mla_build(inv, TORCH_ENV, inputs).run()
        assert (one_shot.float() - want).abs().max().item() < 2e-2
        # A budget under one row's gather forces a block a query.
        monkeypatch.setattr(att._refs, "GATHER_BLOCK_BYTES", 1)
        chunked = att._torch_sparse_mla_build(inv, TORCH_ENV, inputs).run()
        assert (chunked.float() - one_shot.float()).abs().max().item() < 2e-2

    def test_the_gather_floor_holds_one_block_and_serves_every_planned_size(
            self, monkeypatch):
        """No size bound: the floor is offered the largest sparse-MLA cells
        the plans carry (1048576 queries x 4104 keys, Step-5 prefill), and
        their cost is the timer's to judge (`timing.TIME_BUDGET_S` cuts the
        samples or records a `timeout`). What lets it run there: past the
        budget nothing query-sized is held but the inputs and the output —
        the int32 indices are widened to int64 a block at a time, never
        whole, and the timed call allocates no buffer of its own."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.attention as att
        from kernel_bench.refs import get_ref
        floor = next(p for p in _providers("mla_sparse_attention")
                     if p.kernel_impl == "torch_gather_sparse_mla")
        for s_query, topk, heads, d in ((1048576, 4104, 4, 192),
                                        (262144, 2048, 64, 576)):
            big = _inv("mla_sparse_attention", {
                "B": 1, "s_query": s_query, "s_kv": topk, "n_heads": heads,
                "d_head": d, "d_v_head": min(d, 512), "n_topk_keys": topk,
                "kv_block_size": 64})
            assert floor.serves.accepts(big), floor.serves.refusals(big)

        inv = _inv("mla_sparse_attention", self.GATHER_DIMS)
        inputs = build_inputs(get_ref("mla_sparse_attention"), inv, "cpu")
        whole = inputs["indices"].numel()
        from torch.utils._python_dispatch import TorchDispatchMode

        sizes: list = []

        class Int64Made(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                # Storage, not numel: an expand is a view.
                sizes.extend(
                    t.untyped_storage().nbytes() // 8
                    for t in (out if isinstance(out, (tuple, list)) else (out,))
                    if isinstance(t, torch.Tensor) and t.dtype == torch.int64)
                return out

        monkeypatch.setattr(att._refs, "GATHER_BLOCK_BYTES", 1)
        with Int64Made():
            case = att._torch_sparse_mla_build(inv, TORCH_ENV, inputs)
            first = case.run()
            second = case.run()
        assert second is first               # the output was built, not timed
        assert sizes                         # the blocks did widen indices
        assert max(sizes) < whole, sizes

    def test_both_floors_stay_at_their_cells_widths(self):
        _floor_meets_the_oracle(
            "mla_sparse_sink_attention", self.SINK_DIMS,
            env={"activation": "bf16", "kv": "bf16", "weight": "fp32",
                 "output": "bf16"})
        _floor_meets_the_oracle(
            "gqa_sparse_gated_attention", self.QSA_DIMS,
            env={"activation": "bf16", "kv": "bf16", "output": "bf16"})

    def test_the_sink_launches_pad_heads_and_carry_global_indices(
            self, monkeypatch):
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.attention as att
        from kernel_bench.refs import get_ref
        seen = {}

        def flash(q, kv, indices, scale, **kwargs):
            seen.update(q=q, kv=kv, indices=indices, scale=scale, **kwargs)
            return torch.empty(q.shape[0], q.shape[1], 512), None, None

        self._fake_module(
            monkeypatch, "vllm.v1.attention.ops.flashmla",
            flash_mla_sparse_fwd=flash, flash_mla_with_kvcache=flash,
            get_mla_metadata=lambda: (None, None))
        monkeypatch.setattr(att, "_require_cuda", lambda: None)
        inv = _inv("mla_sparse_sink_attention", self.SINK_DIMS, phase="decode")
        inputs = build_inputs(get_ref("mla_sparse_sink_attention"), inv, "cpu")
        att._vllm_sparse_sink_bf16_build(
            inv, Env("2.11", None, "90", None, None,
                     libs={"torch": "2.11"}), inputs).run()
        assert seen["q"].shape == (10, 64, 576)
        assert seen["kv"].shape == (74, 1, 576)
        # 37 keys on the sparse prefill kernel's 128-key tile
        # (`topk % (2*B_TOPK)`), the valid count handed beside them.
        assert seen["indices"].shape == (10, 1, 128)
        assert seen["attn_sink"].shape == (64,)
        assert torch.equal(seen["attn_sink"][:5], inputs["sinks"])
        assert torch.isinf(seen["attn_sink"][5:]).all()
        # The selection's own 11 keys, not the 37 cached: past them the
        # list is -1 padding (vLLM's valid counts; sglang's seqlens
        # clipped to index_topk).
        assert seen["topk_length"].tolist() == [11] * 10
        assert seen["indices"][0, 0, 0].item() == 0
        assert seen["indices"][5, 0, 0].item() == 37

        sgl_seen = {}

        def sgl_flash(q, kv, indices, sm_scale, **kwargs):
            sgl_seen.update(q=q, kv=kv, indices=indices, sm_scale=sm_scale,
                            **kwargs)
            return torch.empty(q.shape[0], q.shape[1], 512), None, None

        self._fake_module(
            monkeypatch, "sgl_kernel.flash_mla",
            flash_mla_sparse_fwd=sgl_flash)
        att._sgl_sparse_sink_build(
            inv, Env("2.11", None, "90", None, None,
                     libs={"torch": "2.11"}), inputs).run()
        assert sgl_seen["q"].shape == (10, 64, 576)
        assert torch.equal(sgl_seen["attn_sink"][:5], inputs["sinks"])
        assert torch.count_nonzero(sgl_seen["attn_sink"][5:]) == 0
        assert sgl_seen["topk_length"].tolist() == [11] * 10

    def test_the_qsa_launch_reads_the_split_paged_cache_and_packed_selection(
            self, monkeypatch):
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.attention as att
        from kernel_bench.refs import get_ref
        seen = {}

        def qsa(q, k_cache, v_cache, logical_indices, block_table,
                token_to_req, use_prefill_config, out, **kwargs):
            seen.update(q=q, k=k_cache, v=v_cache, idx=logical_indices,
                        table=block_table, req=token_to_req,
                        prefill=use_prefill_config, out=out, **kwargs)
            return out

        self._fake_module(
            monkeypatch, "vllm.models.qwen4_exp.nvidia.ops.qsa",
            qsa_sparse_paged_attention=qsa)
        monkeypatch.setattr(att, "_require_cuda", lambda: None)
        inv = _inv("gqa_sparse_gated_attention", self.QSA_DIMS)
        inputs = build_inputs(get_ref("gqa_sparse_gated_attention"), inv, "cpu")
        att._vllm_qsa_gated_build(inv, TORCH_ENV, inputs).run()
        assert seen["q"].shape == (10, 6, 16) and seen["out"].shape == (
            10, 6, 16)
        assert seen["k"].shape == seen["v"].shape == (6, 2, 16, 16)
        assert seen["k"].stride() == seen["v"].stride() == (1024, 32, 64, 1)
        assert seen["v"].storage_offset() - seen["k"].storage_offset() == 16
        assert not seen["k"].is_contiguous()
        assert seen["idx"].shape == (10, 12)
        assert seen["idx"][:, -1].tolist() == [11] * 10
        assert seen["table"].shape == (2, 3)
        assert seen["req"].tolist() == [0] * 5 + [1] * 5
        assert seen["prefill"] is True
        assert seen["output_gate"].is_contiguous()
        assert torch.equal(seen["output_gate"],
                           inputs["gate"].flatten(0, 1))

    @pytest.mark.regression          # walks Hy4 and Qwen4Exp
    def test_the_plans_reach_each_attention_span(self):
        from backend.inference.framework_bindings import (ServingDeployment,
                                                          get_binding)
        from kernel_bench.plan import build_jobs
        for fw in ("vllm", "sglang"):
            binding = get_binding(fw).deployed(ServingDeployment.at(
                gpu_sku="H100-SXM", activation_dtype="BF16"))
            for phase in ("prefill", "decode"):
                jobs, dropped = build_jobs(
                    avg_output_tokens=128, model_id="tencent/Hy4-preview",
                    batch_size=2, seq_len=1024, phase=phase, framework=fw,
                    binding=binding, kv_block_sizes=[64], gpu="H100-SXM")
                cells = [j["dims"] for j in jobs
                         if j["signature"] == "mla_sparse_sink_attention"]
                assert cells, (fw, phase, dropped)
                assert cells[0]["d_head"] == 576, cells
        binding = get_binding("vllm").deployed(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16"))
        jobs, dropped = build_jobs(
            avg_output_tokens=128, model_id="Qwen/Qwen3.8-Flash-Next",
            batch_size=2, seq_len=1024, phase="prefill", framework="vllm",
            binding=binding, kv_block_sizes=[16], gpu="H100-SXM")
        cells = [j["dims"] for j in jobs
                 if j["signature"] == "gqa_sparse_gated_attention"]
        assert cells and cells[0]["n_heads"] == 24, (cells, dropped)


class TestProductionLaunchLayouts:
    """The strided operands serving hands these launches are part of the
    call (`Launch` renders `:strided` because "for these kernels the layout
    is half the call", providers/__init__.py). The framework kernels cannot
    run here, so each build is driven against a fake module injected at its
    import path and the tensors it would launch with are asserted: stride,
    storage offset, values. A build that contiguousizes an operand again —
    `stride(0) == width`, offset 0, `is_contiguous()` — fails these."""

    @staticmethod
    def _fake_module(monkeypatch, name, **funcs):
        import sys
        import types
        mod = types.ModuleType(name)
        for k, v in funcs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    def test_the_a_norm_pair_is_handed_the_split_views(self, monkeypatch):
        """Serving splits q and latent KV off one fused projection and the
        wrapper passes their row strides to the kernel
        (fused_qk_rmsnorm.py:181-184, :207): the build must hand over views
        at the projection row's pitch — DeepSeek-V4's tail-free q|kv row,
        Kimi-K3's rope-tailed one (kimi_k3/nvidia/mla.py:518-521) — not
        contiguous rows."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        seen = {}

        def fused_q_kv_rmsnorm(q, kv, qw, kvw, eps):
            seen.update(q=q, kv=kv)
            return q, kv
        self._fake_module(monkeypatch,
                          "vllm.models.common.ops.fused_qk_rmsnorm",
                          fused_q_kv_rmsnorm=fused_q_kv_rmsnorm)
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        # DeepSeek-V4's geometry: the row is exactly q|kv.
        dims = {"B": 2, "s_query": 3, "d_q_lora": 96, "d_kv_lora": 32}
        inv = _inv("fused_q_kv_rmsnorm", dims)
        inputs = build_inputs(get_ref("fused_q_kv_rmsnorm"), inv, "cpu")
        case = na._vllm_fused_q_kv_rmsnorm_build(inv, TORCH_ENV, inputs)
        case.run()
        q, kv = seen["q"], seen["kv"]
        assert q.stride() == (128, 1) and kv.stride() == (128, 1)
        assert not q.is_contiguous() and not kv.is_contiguous()
        assert kv.storage_offset() - q.storage_offset() == 96
        assert torch.equal(q, inputs["q"]) and torch.equal(kv, inputs["kv"])
        # Kimi-K3's cell (1536, 512): the row carries the 64-wide rope
        # tail, so both views pitch at 2112, not 2048.
        dims = {"B": 1, "s_query": 2, "d_q_lora": 1536, "d_kv_lora": 512}
        inv = _inv("fused_q_kv_rmsnorm", dims)
        inputs = build_inputs(get_ref("fused_q_kv_rmsnorm"), inv, "cpu")
        na._vllm_fused_q_kv_rmsnorm_build(inv, TORCH_ENV, inputs).run()
        q, kv = seen["q"], seen["kv"]
        assert q.stride() == (2112, 1) and kv.stride() == (2112, 1)
        assert not q.is_contiguous() and not kv.is_contiguous()
        assert kv.storage_offset() - q.storage_offset() == 1536
        assert torch.equal(q, inputs["q"]) and torch.equal(kv, inputs["kv"])

    def test_the_per_head_q_k_norm_is_handed_the_qkv_rows_views(
            self, monkeypatch):
        """sglang splits q|k|v off one projection row (qwen3.py:174) and
        hands `fused_inplace_qknorm` `q.view(T, -1, d_head)` /
        `k.view(T, -1, d_head)` (utils.py:524-525), normalised in place. The
        build must hand over those views at the row's pitch, and read the
        result back off the row. Checked by a second route: the fake kernel
        normalises each head in a Python loop, and the build's readback must
        equal the oracle while v's columns stay as they were."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        seen = {}

        def fused_inplace_qknorm(q, k, q_weight, k_weight, eps=1e-6, *,
                                 head_dim=0):
            seen.update(q=q, k=k, qw=q_weight, head_dim=head_dim)
            for x, w in ((q, q_weight), (k, k_weight)):
                for t in range(x.shape[0]):
                    for h in range(x.shape[1]):
                        r = x[t, h].float()
                        x[t, h] = (r / torch.sqrt(r.pow(2).mean() + eps)
                                   * w.float()).to(x.dtype)
        self._fake_module(monkeypatch, "sglang.kernels.ops.layernorm.norm",
                          fused_inplace_qknorm=fused_inplace_qknorm)
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        dims = {"B": 1, "s_query": 3, "n_heads": 4, "n_kv_heads": 2,
                "d_head": 16}
        inv = _inv("fused_q_kv_rmsnorm", dims)
        spec = get_ref("fused_q_kv_rmsnorm")
        inputs = build_inputs(spec, inv, "cpu")
        case = na._sgl_fused_inplace_qknorm_build(inv, TORCH_ENV, inputs)
        row, q_cols, k_cols = (4 + 2 * 2) * 16, 4 * 16, 2 * 16
        out = case.run()
        q, k = seen["q"], seen["k"]
        assert q.shape == (3, 4, 16) and k.shape == (3, 2, 16)
        assert q.stride() == (row, 16, 1) and k.stride() == (row, 16, 1)
        assert k.storage_offset() - q.storage_offset() == q_cols
        # v's columns are the rest of the same row, and nothing wrote them.
        v = q.as_strided((3, k_cols), (row, 1),
                         q.storage_offset() + q_cols + k_cols)
        v_before = v.clone()
        case.run()
        assert torch.equal(v, v_before)
        # Back to the fresh row for the readback check.
        case = na._sgl_fused_inplace_qknorm_build(inv, TORCH_ENV, inputs)
        out = case.run()
        assert seen["head_dim"] == 16 and seen["qw"].dtype == torch.bfloat16
        got = case.to_canonical(out)
        want = spec.reference(inputs)
        assert got.shape == want.shape
        assert (got - want).abs().max() < 0.05

    def test_the_qwen_gate_is_handed_the_3d_strided_view(self, monkeypatch):
        """Serving leaves the gate as `q_gate.view(T, heads, 2·d_head).
        chunk(2, -1)[1]` (qwen3_5.py:1465-1473) and the wrapper takes its
        explicit-stride path on a 3-D gate (elementwise.py:430-445): the
        build's gate must be 3-D with `stride(-2) == 2·d_head` and the FULL
        qkv row's pitch — `gate_stride_row` is a runtime argument the
        kernel indexes by (`:394-406`), k|v columns included."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        seen = {}

        def fused_sigmoid_mul(attn_output, gate, inplace=False, quant=False):
            seen.update(attn=attn_output, gate=gate, inplace=inplace)
            return attn_output
        self._fake_module(monkeypatch,
                          "sglang.kernels.ops.elementwise.elementwise",
                          fused_sigmoid_mul=fused_sigmoid_mul)
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        # A width no bound model has: the synthetic stand-in, one KV
        # column per head — 2·512 + 2·256, still the strided path.
        dims = {"B": 1, "s_query": 2, "d_model": 512}
        inv = _inv("sigmoid_gate_mul", dims)
        inputs = build_inputs(get_ref("sigmoid_gate_mul"), inv, "cpu")
        case = na._sgl_fused_sigmoid_mul_build(inv, TORCH_ENV, inputs)
        case.run()
        gate = seen["gate"]
        assert seen["inplace"] and seen["attn"].is_contiguous()
        assert gate.ndim == 3 and gate.shape == (2, 2, 256)
        assert gate.stride() == (1536, 512, 1)
        assert gate.storage_offset() == 256          # the row's second half
        assert torch.equal(gate.reshape(2, 512), inputs["b"])
        # The bound cells, two tokens each so the single-row contiguous
        # normalization cannot drop the pitch (production's slice
        # normalizes the same way, and pid_row == 0 never indexes by it):
        #   Qwen3.5-35B-A3B   16 q / 2 KV  -> row 9216
        #   Qwen3.8-27B       24 q / 4 KV  -> row 14336
        #   Qwen3.8-2.4T-A95B 64 q / 4 KV  -> row 34816
        for d_model, heads, row in ((4096, 16, 9216), (6144, 24, 14336),
                                    (16384, 64, 34816)):
            dims = {"B": 1, "s_query": 2, "d_model": d_model}
            inv = _inv("sigmoid_gate_mul", dims)
            inputs = build_inputs(get_ref("sigmoid_gate_mul"), inv, "cpu")
            na._sgl_fused_sigmoid_mul_build(inv, TORCH_ENV, inputs).run()
            gate = seen["gate"]
            assert gate.shape == (2, heads, 256), d_model
            assert gate.stride() == (row, 512, 1), d_model
            assert torch.equal(gate.reshape(2, d_model), inputs["b"])
        # Qwen3.5-35B at TP2: the local gate is 2048 wide and one local KV
        # head remains (qwen3_5.py:1261-1265) — the local row 4608.
        dims = {"B": 1, "s_query": 2, "d_model": 2048}
        inv = Invocation(op="sigmoid_gate_mul", signature="sigmoid_gate_mul",
                         phase="decode", dims=dims, tp=2,
                         dtype_env={"activation": "bf16"})
        inputs = build_inputs(get_ref("sigmoid_gate_mul"), inv, "cpu")
        na._sgl_fused_sigmoid_mul_build(inv, TORCH_ENV, inputs).run()
        gate = seen["gate"]
        assert gate.shape == (2, 8, 256)
        assert gate.stride() == (4608, 512, 1)

    @pytest.mark.regression          # walks two models at TP2
    def test_the_planned_tp2_cells_carry_the_real_shard(self, monkeypatch):
        """The routine sweep plans sharded cells with `parallel=`, not the
        legacy `tp` scalar (sweep.py:392-396), so the job must carry the
        config's attention-TP degree: a 4-KV-head model at TP2 (27B: local
        q 3072, two local KV heads — `max(1, 4 // 2)`, qwen3_5.py:1261-1265)
        has production row 7168, which a stale `tp: 1` stamp cannot build
        (the full-width key misses, the fallback pitches 6656)."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from backend.inference.framework_bindings import (
            ServingDeployment, get_binding)
        from backend.inference.parallel_sweep import config_for_degree
        from kernel_bench.plan import build_jobs
        from kernel_bench.refs import get_ref
        seen = {}

        def fused_sigmoid_mul(attn_output, gate, inplace=False, quant=False):
            seen["gate"] = gate
            return attn_output
        self._fake_module(monkeypatch,
                          "sglang.kernels.ops.elementwise.elementwise",
                          fused_sigmoid_mul=fused_sigmoid_mul)
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        binding = get_binding("sglang").deployed(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16"))
        # local q width, local heads, production row — 27B and 2.4T at TP2.
        for model, d_model, heads, row in (
                ("Qwen/Qwen3.8-27B", 3072, 12, 7168),
                ("Qwen/Qwen3.8-2.4T-A95B", 8192, 32, 17408)):
            jobs, dropped = build_jobs(
                avg_output_tokens=128, model_id=model, batch_size=2,
                seq_len=1024, phase="decode", framework="sglang",
                binding=binding, parallel=config_for_degree("tp_tp", 2),
                gpu="H100-SXM")
            assert "sigmoid_gate_mul" not in dropped, model
            cell = next(j for j in jobs
                        if j["signature"] == "sigmoid_gate_mul")
            assert cell["dims"]["d_model"] == d_model, model
            assert cell["tp"] == 2, (model, cell["tp"])
            inv = Invocation.from_job(cell)
            assert inv.tp == 2, model
            inputs = build_inputs(get_ref("sigmoid_gate_mul"), inv, "cpu")
            na._sgl_fused_sigmoid_mul_build(inv, TORCH_ENV, inputs).run()
            gate = seen["gate"]
            assert gate.shape == (2, heads, 256), model
            assert gate.stride() == (row, 512, 1), model
            assert torch.equal(gate.reshape(2, d_model), inputs["b"])

    def test_softcap_is_handed_the_padded_vocab_slice(self, monkeypatch):
        """Serving caps the `[:, :vocab]` slice of the 64-padded LM-head row
        (logits_processor.py:1162-1163; `pad_vocab_size`,
        vocab_parallel_embedding.py:57-60), and the kernel takes the row
        stride as an argument (softcap.py:108-137): a vocab off the 64 grid
        must reach the launch strided; one on it, contiguous — the two
        layouts serving makes, and no `.contiguous()` either way."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        seen = {}

        def softcap_to_float32_logits(x, cap, out):
            seen["x"] = x
            return out

        def softcap_inplace_logits(x, cap):
            seen["x_inplace"] = x
            return x
        self._fake_module(monkeypatch,
                          "sglang.kernels.ops.activation.softcap",
                          softcap_to_float32_logits=softcap_to_float32_logits,
                          softcap_inplace_logits=softcap_inplace_logits)
        import sys
        import types
        monkeypatch.setitem(
            sys.modules, "sglang.kernels.ops.activation",
            types.SimpleNamespace(
                softcap=sys.modules[
                    "sglang.kernels.ops.activation.softcap"]))
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        sig = "softcap_" + "logits"
        # 250 pads to 256: the strided slice, row stride 256.
        inv = _inv(sig, {"B": 1, "s_query": 2, "d_model": 250})
        inputs = build_inputs(get_ref(sig), inv, "cpu")
        na._sgl_softcap_build(inv, TORCH_ENV, inputs).run()
        x = seen["x"]
        assert x.shape == (2, 250) and x.stride() == (256, 1)
        assert not x.is_contiguous()
        assert torch.equal(x, inputs["x"])
        # The fp32 in-place arm gets the same layout.
        na._sgl_softcap_inplace_build(inv, TORCH_ENV, inputs).run()
        assert seen["x_inplace"].stride() == (256, 1)
        # 256 is a 64-multiple: the slice is the whole row, contiguous, as
        # serving's own no-op slice leaves it.
        inv = _inv(sig, {"B": 1, "s_query": 2, "d_model": 256})
        inputs = build_inputs(get_ref(sig), inv, "cpu")
        na._sgl_softcap_build(inv, TORCH_ENV, inputs).run()
        assert seen["x"].is_contiguous()

    def test_the_index_key_is_handed_the_wk_slice(self, monkeypatch):
        """Serving passes the key as the leading slice of the fused
        wk/weights row (dsa_indexer.py:1070-1082), whose batch stride the
        kernel carries (indexer_k.cuh:416): the build's key must have
        `stride(0) == d_indexer_head + heads`, not 128."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.attention as att
        import kernel_bench.providers.indexer as idx
        from kernel_bench.refs import get_ref
        seen = {}

        def fused_k_indexer_norm_rope_store(k, cache, slots, w, b, eps,
                                            table, positions, page):
            seen.update(k=k, cache=cache, slots=slots)
        self._fake_module(monkeypatch,
                          "sglang.kernels.ops.attention.dsa.indexer_k",
                          fused_k_indexer_norm_rope_store=
                          fused_k_indexer_norm_rope_store)
        monkeypatch.setattr(att, "_require_cuda", lambda: None)
        sig = "fused_indexer_k_norm_rope_store"
        dims = {"B": 2, "s_query": 3, "d_indexer_head": 32, "d_rope": 16,
                "kv_block_size": 4}
        inv = _inv(sig, dims)
        inputs = build_inputs(get_ref(sig), inv, "cpu")
        idx._sgl_indexer_k_store_build(inv, TORCH_ENV, inputs).run()
        k = seen["k"]
        assert k.stride() == (32 + idx._INDEXER_WK_HEADS, 1)
        assert not k.is_contiguous() and k.storage_offset() == 0
        assert torch.equal(k, inputs["k"])
        # The packed record cache and the slots, as before.
        assert seen["cache"].dtype == torch.uint8
        assert seen["cache"].shape[1] == 4 * (128 + 4)
        assert seen["slots"].dtype == torch.int64

    def test_the_index_query_weights_are_handed_the_wk_slice(self,
                                                             monkeypatch):
        """Serving hands the head weights as the trailing slice of the same
        fused row (dsa_indexer.py:1070-1071), whose batch stride the kernel
        reads (main_norm_rope.cuh:751-755, :793), while the query it
        contiguousizes itself (:1084) and the kernel requires contiguous
        (:774-787): the build's weights must be the strided slice, its
        query contiguous."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.attention as att
        import kernel_bench.providers.indexer as idx
        from kernel_bench.refs import get_ref
        seen = {}

        def fused_q_indexer_rope_first_quant(q, weights, ws, table, pos):
            seen.update(q=q, weights=weights)
            return q, weights
        self._fake_module(monkeypatch,
                          "sglang.kernels.ops.attention.dsv4.elementwise",
                          fused_q_indexer_rope_first_quant=
                          fused_q_indexer_rope_first_quant)
        monkeypatch.setattr(att, "_require_cuda", lambda: None)
        sig = "fused_indexer_q_rope_first_quant"
        dims = {"B": 2, "s_query": 3, "d_indexer_head": 32, "d_rope": 16,
                "n_indexer_heads": 4}
        inv = _inv(sig, dims)
        inputs = build_inputs(get_ref(sig), inv, "cpu")
        idx._sgl_indexer_q_build(inv, TORCH_ENV, inputs).run()
        w = seen["weights"]
        assert w.stride() == (32 + 4, 1) and w.storage_offset() == 32
        assert not w.is_contiguous()
        assert torch.equal(w, inputs["weights"])
        assert seen["q"].is_contiguous()      # serving's own q.contiguous()

    def test_the_dsa_launch_writes_the_production_caches(self, monkeypatch):
        """Serving's launch runs with `slot_mapping` set and writes the
        paged MLA cache on every layer and the packed FP8 index-K record on
        an indexer layer (attention.py:378-405; kernels.py:357-405,
        :488-506): the build must pass both caches, real slot mappings, and
        the inputs as the strided splits of the projection rows."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.rope as rope
        from kernel_bench.refs import get_ref
        seen = {}

        def fused_norm_rope(positions, q_c, q_w, q_eps, kv_c, kv_w, kv_eps,
                            k_pe, table, index_k, ik_w, ik_b, ik_eps,
                            ik_table, topk, **kwargs):
            seen.update(q=q_c, kv=kv_c, k_pe=k_pe, ik=index_k, **kwargs)
            return q_c
        self._fake_module(monkeypatch,
                          "vllm.models.deepseek_v32.common.kernels",
                          fused_norm_rope=fused_norm_rope)
        monkeypatch.setattr(rope, "_require_cuda", lambda: None)
        dims = {"B": 2, "s_query": 3, "d_q_lora": 96, "d_kv_lora": 32,
                "d_rope": 16, "d_indexer_head": 32, "n_topk_keys": 64}

        inv = _inv("fused_dsa_qkv_norm_rope_indexer", dims)
        inputs = build_inputs(get_ref("fused_dsa_qkv_norm_rope_indexer"),
                              inv, "cpu")
        rope._vllm_dsa_norm_rope_build(inv, TORCH_ENV, inputs).run()
        # The cache-write branch: real slot mappings and both caches.
        slots = seen["slot_mapping"]
        assert slots is not None and slots.dtype == torch.int64
        assert torch.equal(slots, torch.arange(6))
        mla = seen["mla_kv_cache"]
        assert mla.shape == (1, 64, 32 + 16) and mla.dtype == torch.bfloat16
        assert seen["mla_kv_cache_dtype"] == "auto"
        cache = seen["indexer_k_cache"]
        assert cache.dtype == torch.uint8 and cache.shape == (1, 64, 32 + 4)
        assert seen["indexer_slot_mapping"] is not None
        # The inputs are the strided splits of the qkv_lora / wk rows.
        q, kv, k_pe, ik = seen["q"], seen["kv"], seen["k_pe"], seen["ik"]
        assert q.stride(0) == 96 + 32 + 16 and q.stride(1) == 1
        assert kv.storage_offset() - q.storage_offset() == 96
        assert k_pe.storage_offset() - kv.storage_offset() == 32
        assert not q.is_contiguous()
        assert ik.stride(0) == 32 + 64     # d_q_lora 96: DeepSeek-V3.2's 64
        assert not ik.is_contiguous()
        assert torch.equal(q, inputs["q"]) and torch.equal(kv, inputs["kv"])
        assert torch.equal(k_pe, inputs["k_pe"])
        assert torch.equal(ik, inputs["ik"])
        # The outputs serving passes, plus the stated `index_k_out`
        # boundary (serving's PCP form, attention.py:350).
        assert seen["kv_c_out"] is not None and seen["k_pe_out"] is not None
        assert seen["index_k_out"] is not None

        # GLM-5.x's cell (d_q_lora 2048): the wk row carries 32 heads
        # beside the key (snapshots/zai-org__GLM-5.2.json), not 64.
        seen.clear()
        glm_dims = {"B": 1, "s_query": 2, "d_q_lora": 2048, "d_kv_lora": 512,
                    "d_rope": 64, "d_indexer_head": 32, "n_topk_keys": 64}
        inv = _inv("fused_dsa_qkv_norm_rope_indexer", glm_dims)
        inputs = build_inputs(get_ref("fused_dsa_qkv_norm_rope_indexer"),
                              inv, "cpu")
        rope._vllm_dsa_norm_rope_build(inv, TORCH_ENV, inputs).run()
        assert seen["ik"].stride(0) == 32 + 32
        assert not seen["ik"].is_contiguous()

        # The shared-indexer layer: the MLA cache write stays, the index
        # cache leaves with the indexer.
        seen.clear()
        dims = {k: dims[k] for k in
                ("B", "s_query", "d_q_lora", "d_kv_lora", "d_rope")}
        inv = _inv("fused_dsa_qkv_norm_rope", dims)
        inputs = build_inputs(get_ref("fused_dsa_qkv_norm_rope"), inv, "cpu")
        rope._vllm_dsa_norm_rope_build(inv, TORCH_ENV, inputs).run()
        assert seen["slot_mapping"] is not None
        assert seen["mla_kv_cache"] is not None
        assert seen["indexer_k_cache"] is None
        assert seen["ik"] is None and seen["index_k_out"] is None
        assert seen["q"].stride(0) == 96 + 32 + 16


class TestKdaWholeMixer:
    """Kimi-K3's whole KDA mixer for one decode token
    (`fusion_plans/fused_kda_decode.yaml`): the three short convs, the KDA
    step and the gated output norm in one launch."""

    DIMS: ClassVar[dict] = {"B": 3, "s_query": 1, "n_lin_k_heads": 2,
                            "d_lin_k_head": 16, "n_lin_v_heads": 2,
                            "d_lin_v_head": 16, "s_conv_kernel": 4}

    def test_the_span_binds_and_its_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        sig = "fused_kda_decode"
        assert op_bindings()[sig] == sig
        ps = _providers(sig)
        assert {p.kernel_impl for p in ps} == {
            "vllm_c_fused_kda_decode", "sglang_kda_fused_decode",
            "torch_fused_kda_decode"}
        assert all(p.fused_op == sig for p in ps)

    def test_each_kernel_takes_the_head_counts_it_instantiates(self):
        """vLLM's kernel checks 12/24/48/96 local heads, sglang's JIT builds
        3/6/12 — the two bindings' `world_local` sets."""
        dims = {**self.DIMS, "d_lin_k_head": 128, "d_lin_v_head": 128}
        for heads, want in ((12, {"vllm", "sglang", "torch"}),
                            (96, {"vllm", "torch"}),
                            (6, {"sglang", "torch"}),
                            (64, {"torch"})):
            inv = _inv("fused_kda_decode", {**dims, "n_lin_k_heads": heads,
                                             "n_lin_v_heads": heads},
                       phase="decode")
            got = {p.library for p in _providers("fused_kda_decode")
                   if p.accepts(inv, TORCH_ENV)}
            assert got == want, heads

    def test_with_the_newest_tap_alone_the_mixer_is_the_step_then_the_norm(self):
        """Conv taps (0, 0, 0, 1) make each conv the identity on the new
        token, so the mixer is `fused_recurrent_kda` on silu(q|k|v), then
        rmsnorm·w·sigmoid(g) — recomputed through that sibling's oracle."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        mixer = get_ref("fused_kda_decode")
        step = get_ref("fused_recurrent_kda")
        inputs = build_inputs(mixer, _inv("fused_kda_decode", self.DIMS,
                                          phase="decode"), "cpu")
        for n in ("q_conv_w", "k_conv_w", "v_conv_w"):
            w = torch.zeros_like(inputs[n])
            w[:, -1] = 1
            inputs[n] = w
        B, H, D = 3, 2, 16

        def heads(x):
            return torch.nn.functional.silu(x.float()).reshape(B, 1, H, D)
        o = step.reference({**inputs, "q": heads(inputs["q"]),
                            "k": heads(inputs["k"]), "v": heads(inputs["v"])})
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5)
        want = (o * inputs["o_norm_w"].float()
                * torch.sigmoid(inputs["g"].float().reshape(B, 1, H, D)))
        assert (mixer.reference(inputs) - want).abs().max() < 1e-5

    def test_the_conv_reads_its_state_oldest_first(self):
        """A tap on the oldest state slot alone returns that slot, so the
        state is read oldest first and the new token last."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import kda_short_conv
        x = torch.randn(2, 1, 5)
        state = torch.randn(2, 5, 3)
        w = torch.zeros(5, 4)
        w[:, 0] = 1
        got = kda_short_conv(x, state, w)
        assert torch.allclose(got[:, 0], torch.nn.functional.silu(state[..., 0]))

    def test_the_floor_runs_the_chain_at_the_cells_width(self):
        """The real cell (Kimi-K3) bills bf16 activations, fp32 weights and
        the fp32 recurrent pool. The widest-operand check alone cannot pin
        the convs and the gated norm to the cell: the pool already explains
        every fp32 the env declares, so a floor that widened the conv state
        or the output gate to fp32 would pass it. The trace pins what the
        check cannot — the conv pools the floor rolls are stored bf16, as
        both frameworks require their conv state to be (sglang
        kda_fused_decode.py:87-92, fused_kda_decode_kernel.cu:996-1009),
        the output gate is applied bf16, and only the recurrent pool's
        store is fp32."""
        trace = _floor_meets_the_oracle(
            "fused_kda_decode", self.DIMS,
            env={"activation": "bf16", "weight": "fp32", "kv": "fp32",
                 "output": "bf16"})
        B, HV, DK, DV, W = 3, 2, 16, 16, 4
        stores = [(tuple(c.operands[0].shape), c.operands[0].dtype)
                  for c in trace.calls
                  if c.op == "aten.copy_" and c.operands]
        conv_segment = (B, 2 * DK, W - 1)
        assert (conv_segment, "bfloat16") in stores            # conv tap
        assert ((B, HV, DV, DK), "float32") in stores      # recurrent pool
        # No conv segment is ever stored at a wider precision.
        assert [dt for shape, dt in stores if shape == conv_segment
                ] == ["bfloat16"] * 3
        # The output gate is applied at the cell's dtype, not widened.
        assert any(o.dtype == "bfloat16"
                   and tuple(o.shape) == (B, 1, HV, DV)
                   for c in trace.calls if c.op == "aten.sigmoid"
                   for o in c.operands)

    def test_the_floor_writes_the_states_the_span_bills(self):
        """The launch updates both pools in place — the convs roll one tap,
        the step stores its new state — and that write traffic is the
        span's. A floor that only computes the output (the oracle's own
        routine never writes a state) is cheaper than the launch it floors
        by exactly those stores, so they must be in the timed call."""
        pytest.importorskip("torch")
        from kernel_bench import launch_trace
        from kernel_bench.refs import get_ref
        sig = "fused_kda_decode"
        inv = _inv(sig, self.DIMS, phase="decode")
        inputs = build_inputs(get_ref(sig), inv, "cpu")
        floor = next(p for p in _providers(sig) if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        recording = launch_trace.record()
        with recording:
            case.run()
        written = {tuple(c.operands[0].shape) for c in recording.trace.calls
                   if c.op == "aten.copy_" and c.operands}
        B, HV, DK, DV, W = 3, 2, 16, 16, 4
        assert (B, HV, DV, DK) in written     # the recurrent pool's new state
        assert (B, 2 * DK, W - 1) in written  # a conv segment, rolled one tap

    def test_the_cell_is_read_off_the_engine_operands(self):
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        kimi = [[2, 1, 12288]] * 4 + [[12288], [128], [2, 1, 96],
                                      [2, 1, 12288]]
        got = identity_shape_from_shape_in(
            "fused_kda_decode", kimi,
            spec_dims={"d_lin_k_head": 128, "d_lin_v_head": 128,
                       "s_conv_kernel": 4},
            operand_ports=_ports("q", "k", "v", "forget_raw", "dt_bias",
                               "a_log", "beta_raw", "out_gate_raw"))
        assert json.loads(got) == {
            "B": 2, "s_query": 1, "n_lin_k_heads": 96, "d_lin_k_head": 128,
            "n_lin_v_heads": 96, "d_lin_v_head": 128, "s_conv_kernel": 4}

    @pytest.mark.regression          # walks Kimi-K3
    def test_the_plan_reaches_vllms_tp1_mixer(self):
        jobs, dropped = _plan("moonshotai/Kimi-K3", "decode", "vllm")
        assert "fused_kda_decode" not in dropped
        cells = [j["dims"] for j in jobs if j["signature"] == "fused_kda_decode"]
        assert cells and all(c["n_lin_v_heads"] == 96 for c in cells), cells


class TestDsv41CompressStoreSpans:
    """DeepSeek-V4.1's CSA2 compressor chain: the pool + norm
    (`fused_save_compress_norm`, both spellings) and the compressed-cache
    store (`fused_compress_rope_store`), each one vLLM launch
    (`fused_compress_quant_cache.py:22` / `:224`). The oracles are checked
    against the spans recomputed another way — the pool per group in a
    python loop, the rotation by the pair definition — so a test cannot
    pass by sharing the oracle's code."""

    @staticmethod
    def _loop_pool(kv, score, w, cr, eps):
        torch = pytest.importorskip("torch")
        kv, score, w = kv.float(), score.float(), w.float()
        rows = []
        for g in range(kv.shape[0] // cr):
            k, s = kv[g * cr:(g + 1) * cr], score[g * cr:(g + 1) * cr]
            p = torch.exp(s - s.max(0, keepdim=True).values)
            x = (k * (p / p.sum(0, keepdim=True))).sum(0)
            rows.append(x * torch.rsqrt(x.pow(2).mean() + eps) * w)
        return torch.stack(rows)

    def test_the_span_binds_both_spellings_and_the_kernels_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert bound["fused_save_compress_norm"] == "fused_save_compress_norm"
        assert (bound["fused_save_compress_norm_decode"]
                == "fused_save_compress_norm")
        assert bound["fused_compress_rope_store"] == "fused_compress_rope_store"
        for sig in ("fused_save_compress_norm", "fused_compress_rope_store"):
            ps = _providers(sig)
            assert {p.library for p in ps} == {"vllm", "torch"}, sig
            assert all(p.fused_op == sig for p in ps), sig

    def test_the_floors_track_the_oracles_inside_the_bf16_band(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        for sig, dims in (
            ("fused_save_compress_norm",
             {"B": 2, "s_query": 8, "d_head": 512, "compress_ratio": 2}),
            ("fused_save_compress_norm",
             {"B": 2, "s_query": 6, "d_head": 512, "compress_ratio": 1}),
            ("fused_compress_rope_store",
             {"B": 2, "s_slots": 4, "d_head": 512, "d_rope": 64,
              "compress_ratio": 2, "kv_block_size": 64}),
            ("fused_compress_rope_store",
             {"B": 1, "s_slots": 5, "d_head": 512, "d_rope": 64,
              "compress_ratio": 1, "kv_block_size": 64}),
        ):
            spec = get_ref(sig)
            inv = _inv(sig, dims)
            inputs = build_inputs(spec, inv, "cpu")
            ref = spec.reference(inputs)
            floor = next(p for p in _providers(sig) if p.library == "torch")
            case = floor.build(inv, TORCH_ENV, floor.convert_inputs(inv, inputs))
            err = (case.to_canonical(case.run()) - ref).abs().max()
            assert err / ref.abs().max() < 8e-3, (sig, err)

    def test_the_pool_oracle_is_the_group_loop(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("fused_save_compress_norm")
        for cr, tokens in ((2, 8), (1, 6)):
            inv = _inv("fused_save_compress_norm", {
                "B": 1, "s_query": tokens, "d_head": 512,
                "compress_ratio": cr})
            inputs = build_inputs(spec, inv, "cpu")
            want = self._loop_pool(inputs["kv"], inputs["score"],
                                   inputs["norm_weight"], cr, inputs["eps"])
            got = spec.reference(inputs)
            assert got.shape == (tokens // cr, 512)
            assert (got - want).abs().max() < 1e-5

    def test_the_store_oracle_rotates_at_the_groups_first_position(self):
        """By the pair definition, adjacent channels paired: the closing
        token's row rotates at (position // ratio) · ratio, not at its own
        position — the bug this case is first to catch."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("fused_compress_rope_store")
        inv = _inv("fused_compress_rope_store", {
            "B": 1, "s_slots": 3, "d_head": 512, "d_rope": 64,
            "compress_ratio": 2, "kv_block_size": 64})
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        x, table = inputs["latent"].float(), inputs["cos_sin_cache"]
        for i in range(3):
            # Row i closes the group at position 2i+1 and rotates at 2i.
            pos = inputs["positions"][i].item()
            assert (pos + 1) % 2 == 0
            cos, sin = table[pos // 2 * 2][:32], table[pos // 2 * 2][32:]
            pe = x[i, -64:]
            even, odd = pe[0::2], pe[1::2]
            want = torch.empty(64)
            want[0::2], want[1::2] = even * cos - odd * sin, odd * cos + even * sin
            assert (got[i, :448] - x[i, :448]).abs().max() == 0
            assert (got[i, 448:] - want).abs().max() < 1e-5, i

    def test_the_cells_are_read_off_the_engine_operands(self):
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        cases = {
            # DeepSeek-V4.1 prefill, the gated compressor: kv and gate rows.
            "fused_save_compress_norm": (
                [[2, 1024, 512], [2, 1024, 512]],
                {"B": 2, "s_query": 1024, "d_head": 512,
                 "compress_ratio": 2}),
            # The pooled latent rows of the same step, 512 closed groups.
            "fused_compress_rope_store": (
                [2, 512, 512],
                {"B": 2, "s_slots": 512, "d_head": 512,
                 "compress_ratio": 2, "d_rope": 64, "kv_block_size": 64}),
        }
        for op, (shape_in, want) in cases.items():
            # The group size, the rotation width and the cache's page ride
            # no tensor of these spans: the plan's nodes and the deployment
            # state them, and the projection fills them from there.
            got = identity_shape_from_shape_in(
                op, shape_in,
                spec_dims={"compress_ratio": 2, "d_rope": 64,
                           "kv_block_size": 64},
                operand_ports=(_ports("kv", "score")
                             if op == "fused_save_compress_norm" else None))
            assert got is not None and json.loads(got) == want, (op, got)

    @pytest.mark.regression          # walks DeepSeek-V4.1 in two phases
    def test_the_plan_reaches_each_span(self):
        # Prefill: the pool and both store rows (the gated compressor's 512
        # groups, the ratio-1 candidate source's 1024 rows).
        jobs, dropped = _plan("deepseek-ai/DeepSeek-V4.1-Flash", "prefill",
                              "vllm")
        pool = {tuple(sorted(j["dims"].items()))
                for j in jobs if j["signature"] == "fused_save_compress_norm"}
        assert pool == {(("B", 2), ("compress_ratio", 2), ("d_head", 512),
                         ("s_query", 1024))}, (pool, [
                             d for d in dropped if "save_compress" in d])
        store = {tuple(sorted(j["dims"].items()))
                 for j in jobs if j["signature"] == "fused_compress_rope_store"}
        assert store == {
            (("B", 2), ("compress_ratio", 2), ("d_head", 512), ("d_rope", 64),
             ("kv_block_size", 16), ("s_slots", 512)),
            (("B", 2), ("compress_ratio", 1), ("d_head", 512), ("d_rope", 64),
             ("kv_block_size", 16), ("s_slots", 1024))}, store


class TestDsv4CompressorSpans:
    """DeepSeek-V4's compressor: the overlap pool+norm+rope+store
    (`fused_overlap_compress_norm_rope_store`, one span over two kernels —
    the 512-wide CuteDSL C4 and the indexer's 128-wide Triton one) and the
    no-overlap tail (`fused_compress_norm_rope_store`, the CuteDSL C128
    store). The overlap oracle is checked against the pool recomputed per
    window in a loop, with window 0 pooling its own Cb half alone — the
    gather mask the kernels state."""

    def test_the_span_binds_and_both_kernels_claim_their_own_width(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert (bound["fused_overlap_compress_norm_rope_store"]
                == "fused_overlap_compress_norm_rope_store")
        assert (bound["fused_compress_norm_rope_store"]
                == "fused_compress_norm_rope_store")
        ps = _providers("fused_overlap_compress_norm_rope_store")
        by_head = {p.serves.enum[0][1][0]: p.kernel_impl
                   for p in ps if p.library == "vllm"}
        assert by_head == {
            512: "vllm_cutedsl_compress_norm_rope_store_c4",
            128: "vllm_triton_kv_compress_norm_rope_insert_indexer"}
        assert all(p.fused_op == "fused_overlap_compress_norm_rope_store"
                   for p in ps)
        ps2 = _providers("fused_compress_norm_rope_store")
        # DeepInfer's V4.1 arms are the other record writers
        # (TestDsv41IndexKeyStore); V4's compressor tail is vLLM's.
        assert {p.library for p in ps2} == {"vllm", "deepinfer", "torch"}

    def test_the_overlap_oracle_is_the_window_loop(self):
        """Window w of a request pools the request's previous window's Ca
        half and its own Cb — each request's first window pools its own Cb
        alone (the kernel gathers through the request's own block table).
        Get the shift wrong (own Ca, no mask, or across the batch) and a
        boundary window is off O(1), the case this test exists to catch."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("fused_overlap_compress_norm_rope_store")
        inv = _inv("fused_overlap_compress_norm_rope_store", {
            "B": 2, "s_slots": 3, "Cape": 4, "d_head": 512, "d_rope": 64,
            "kv_block_size": 64})
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        kv, score = inputs["kv"].float(), inputs["score"].float()
        w, eps = inputs["norm_weight"].float(), inputs["eps"]
        table, cape, d, rd = inputs["cos_sin_cache"], 4, 512, 64
        # The kernel gathers through each request's own block table, so the
        # loop is per request: request r's window w pools request r's window
        # w-1's Ca half, and its window 0 has none.
        kvw = kv.view(2, 3, cape, 2 * d)
        scw = score.view(2, 3, cape, 2 * d)
        rows = []
        for req in range(2):
            for win in range(3):
                ks, ss = [kvw[req, win][:, d:]], [scw[req, win][:, d:]]
                if win > 0:
                    ks.insert(0, kvw[req, win - 1][:, :d])
                    ss.insert(0, scw[req, win - 1][:, :d])
                x = (torch.cat(ks) * torch.softmax(torch.cat(ss), 0)).sum(0)
                x = x * torch.rsqrt(x.pow(2).mean() + eps) * w
                pos = inputs["positions"][req * 3 + win].item() // cape * cape
                cos, sin = table[pos][:32], table[pos][32:]
                pe = x[-rd:]
                even, odd = pe[0::2], pe[1::2]
                rot = torch.empty(rd)
                rot[0::2] = even * cos - odd * sin
                rot[1::2] = odd * cos + even * sin
                rows.append(torch.cat([x[:-rd], rot]))
        assert (got - torch.stack(rows)).abs().max() < 1e-5
        # Revert-goes-red: the flat shift across the batch hands request 1's
        # first window request 0's last Ca half — a different function.
        flat = kv.view(6, cape, 2 * d)
        cross = torch.cat([flat[2][:, :d], flat[3][:, d:]], 0)
        own = torch.cat([torch.zeros_like(flat[3][:, :d]), flat[3][:, d:]], 0)
        assert not torch.allclose(cross, own)

    def test_the_floors_track_the_oracles_inside_the_bf16_band(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        for sig, dims in (
            ("fused_overlap_compress_norm_rope_store",
             {"B": 2, "s_slots": 3, "Cape": 4, "d_head": 512, "d_rope": 64,
              "kv_block_size": 64}),
            ("fused_overlap_compress_norm_rope_store",
             {"B": 1, "s_slots": 2, "Cape": 4, "d_head": 128, "d_rope": 64,
              "kv_block_size": 64}),
            ("fused_compress_norm_rope_store",
             {"B": 2, "s_slots": 4, "d_head": 512, "d_rope": 64,
              "compress_ratio": 128, "kv_block_size": 64}),
        ):
            spec = get_ref(sig)
            inv = _inv(sig, dims)
            inputs = build_inputs(spec, inv, "cpu")
            ref = spec.reference(inputs)
            floor = next(p for p in _providers(sig) if p.library == "torch")
            case = floor.build(inv, TORCH_ENV, floor.convert_inputs(inv, inputs))
            err = (case.to_canonical(case.run()) - ref).abs().max()
            assert err / ref.abs().max() < 8e-3, (sig, err)

    def test_the_overlap_floor_writes_a_record_cell_at_the_state_precision(
            self):
        """The H100 cells the overlap floor's canary failed on: the fp32
        compressor state (`kv=fp32`) written as a record. The record row
        buffer is one normed row a window (shaped like the kv operand it
        raised a size mismatch at every record cell), and the pool runs at
        the state's fp32, not the activation's bf16 — pooled in bf16 the
        mxfp4 cell lands a grid step off (typical 0.022 against a 0.008
        band on the box). The mxfp4 cell first: it is red under both."""
        for d_head, out, record in (
                (128, "fp4_e2m1", "mxfp4"),
                (512, "fp8_e4m3", "fp8_ue8m0_group64_bf16_rope_tail")):
            _floor_meets_the_oracle(
                "fused_overlap_compress_norm_rope_store",
                {"B": 1, "Cape": 4, "d_head": d_head, "d_rope": 64,
                 "kv_block_size": 64, "s_slots": 256},
                env={"activation": "bf16", "kv": "fp32", "output": out,
                     "weight": "bf16"},
                output_format=record)

    def test_the_cells_are_read_off_the_engine_operands(self):
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        cases = {
            # DeepSeek-V4 prefill, the overlap compressor's window rows:
            # 256 windows of 4 positions, two 512-wide halves.
            "fused_overlap_compress_norm_rope_store": (
                [[2, 256, 4, 1024], [2, 256, 4, 1024]],
                {"B": 2, "s_slots": 256, "Cape": 4, "d_head": 512,
                 "d_rope": 64, "kv_block_size": 64}),
            # The no-overlap compressor's pooled rows: 8 of 1024 tokens.
            "fused_compress_norm_rope_store": (
                [2, 8, 512],
                {"B": 2, "s_slots": 8, "d_head": 512, "compress_ratio": 128,
                 "d_rope": 64, "kv_block_size": 64}),
        }
        for op, (shape_in, want) in cases.items():
            # The rotation width and the cache's page ride no tensor of
            # these spans, the tail's group size neither: the plan's nodes
            # and the deployment state them.
            spec = {"d_rope": 64, "kv_block_size": 64,
                    "compress_ratio": 128}
            got = identity_shape_from_shape_in(
                op, shape_in, spec_dims=spec,
                operand_ports=(_ports("kv", "score") if op ==
                             "fused_overlap_compress_norm_rope_store"
                             else None))
            assert got is not None and json.loads(got) == want, (op, got)

    @pytest.mark.regression          # walks DeepSeek-V4-Flash
    def test_the_plan_reaches_each_span_at_both_widths(self):
        jobs, dropped = _plan("deepseek-ai/DeepSeek-V4-Flash", "prefill",
                              "vllm")
        overlap = {tuple(sorted(j["dims"].items())) for j in jobs
                   if j["signature"] == "fused_overlap_compress_norm_rope_store"}
        assert overlap == {
            (("B", 2), ("Cape", 4), ("d_head", 512), ("d_rope", 64),
             ("kv_block_size", 16), ("s_slots", 256)),
            (("B", 2), ("Cape", 4), ("d_head", 128), ("d_rope", 64),
             ("kv_block_size", 16), ("s_slots", 256))}, (overlap, [
                 d for d in dropped if "overlap_compress" in d])
        tail = {tuple(sorted(j["dims"].items())) for j in jobs
                if j["signature"] == "fused_compress_norm_rope_store"}
        assert tail == {
            (("B", 2), ("compress_ratio", 128), ("d_head", 512),
             ("d_rope", 64), ("kv_block_size", 16), ("s_slots", 8))}, tail


class TestDsv41IndexKeyStore:
    """DeepSeek-V4.1's index key under `fused_compress_norm_rope_store`:
    vLLM `indexer_k_norm_rope_store` normalises, rotates, quantises and
    stores each boundary row in one launch. The record it writes is the
    launch's OUTPUT (FP8 + one fp32 scale a key, or MXFP4 + UE8M0 per 32),
    one provider per record; the oracle is the row on that record's grid."""

    SIG: ClassVar[str] = "fused_compress_norm_rope_store"
    #: A gated source layer's cell (compress_ratio 2), the graph's rows.
    DIMS: ClassVar[dict] = {"B": 2, "s_slots": 6, "d_head": 128,
                            "d_rope": 64, "compress_ratio": 2,
                            "kv_block_size": 64}

    #: The records the launch writes, by cache-format name, and their
    #: element dtype (the cell's `output_format` and `output`).
    RECORDS: ClassVar[dict] = {"fp8_pow2_group128": "fp8_e4m3",
                               "mxfp4": "fp4_e2m1"}
    #: The kernel_impl suffix each record's provider was registered under.
    IMPLS: ClassVar[dict] = {"fp8_pow2_group128": "fp8_e4m3",
                             "mxfp4": "mxfp4"}

    def _cell(self, record, dims=None):
        env = {"activation": "bf16", "weight": "bf16",
               "output": self.RECORDS.get(record, "bf16")}
        if record in self.RECORDS:
            env["output_format"] = record
        return Invocation(op=self.SIG, signature=self.SIG, phase="prefill",
                          dims=dict(dims or self.DIMS), tp=1, dtype_env=env)

    def test_each_record_is_its_own_provider(self):
        ps = {p.kernel_impl: p for p in _providers(self.SIG)
              if p.library == "vllm"}
        assert set(ps) == {"vllm_cutedsl_norm_rope_store_c128",
                           "vllm_indexer_k_norm_rope_store_fp8_e4m3",
                           "vllm_indexer_k_norm_rope_store_mxfp4"}
        for record, output in self.RECORDS.items():
            p = ps[f"vllm_indexer_k_norm_rope_store_{self.IMPLS[record]}"]
            assert p.serves.output == (output,)
            assert p.serves.output_format == (record,)
            # A writer states its record as its output, not as a cache read.
            assert not p.serves.dtype_slots
            assert dict(p.serves.enum) == {"d_head": (128,), "d_rope": (64,),
                                           "compress_ratio": (1, 2)}
        # The E2M1 pack is an SM100 instruction; the FP8 record is not.
        assert ps["vllm_indexer_k_norm_rope_store_mxfp4"].min_sm == 100
        assert ps["vllm_indexer_k_norm_rope_store_fp8_e4m3"].min_sm is None

    def test_deepinfer_declares_its_own_index_record_and_page(self):
        """DeepInfer allocates V4.1's cache in 128-token blocks and writes
        the 128-wide index key as MXFP4: its arm names that record and that
        page, so it cannot hide behind the shared signature."""
        ps = [p for p in _providers(self.SIG) if p.library == "deepinfer"]
        assert [p.kernel_impl for p in ps] == [
            "deepinfer_v41_index_fp4_norm_rope_store"]
        index = ps[0]
        assert index.serves.output_format == ("mxfp4",)
        assert dict(index.serves.enum) == {"d_head": (128,)}
        assert index.serves.kv_block_sizes == (128,)
        assert index.min_sm == 100

    def test_a_cell_reaches_the_kernel_of_its_record_only(self):
        ps = [p for p in _providers(self.SIG) if p.library == "vllm"]

        def claimed(record, dims=None):
            inv = self._cell(record, dims)
            return {p.kernel_impl for p in ps if p.serves.accepts(inv)}
        assert claimed("fp8_pow2_group128") == {
            "vllm_indexer_k_norm_rope_store_fp8_e4m3"}
        assert claimed("mxfp4") == {"vllm_indexer_k_norm_rope_store_mxfp4"}
        # A cell that states no record is not this launch's to measure.
        assert claimed("bf16") == set()
        # The kernel asserts a group of 1 or 2 (indexer_k_store.py:61).
        assert claimed("fp8_pow2_group128",
                       {**self.DIMS, "compress_ratio": 4}) == set()

    def test_an_nvfp4_cell_is_not_the_mxfp4_kernels(self):
        """MXFP4 and NVFP4 are both `fp4_e2m1` elements; only the record
        tells the launch that writes one from a cell keyed at the other."""
        ps = [p for p in _providers(self.SIG) if p.library == "vllm"]
        inv = Invocation(op=self.SIG, signature=self.SIG, phase="prefill",
                         dims=dict(self.DIMS), tp=1,
                         dtype_env={"activation": "bf16", "weight": "bf16",
                                    "output": "fp4_e2m1",
                                    "output_format": "nvfp4"})
        assert {p.kernel_impl for p in ps if p.serves.accepts(inv)} == set()

    def test_the_oracle_is_the_row_on_the_records_grid(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref, rows_on_record
        spec = get_ref(self.SIG)
        plain = spec.reference(build_inputs(spec, self._cell("bf16"), "cpu"))
        for record, levels in (("fp8_pow2_group128", 256), ("mxfp4", 16)):
            inputs = build_inputs(spec, self._cell(record), "cpu")
            got = spec.reference(inputs)
            # Revert-goes-red: without the grid the oracle is the plain row.
            assert not torch.equal(got, plain), record
            assert (got - plain).abs().max() < 0.5 * plain.abs().max()
            # On the grid: placing it again moves nothing, and each 32-wide
            # group holds no more distinct magnitudes than the format has.
            again = rows_on_record(got, record, torch.bfloat16)
            assert torch.equal(again, got), record
            groups = got.view(-1, 32).abs()
            assert max(len(g.unique()) for g in groups) <= levels, record

    def test_every_record_a_launch_declares_has_an_oracle_grid(self):
        """Every cached record a provider says it writes is one the oracle
        can write too (`rows_on_record`, the grid read off the registered
        format) — a record declared without one would be compared to an
        off-grid row. `PLAIN` is the row itself, not a record."""
        torch = pytest.importorskip("torch")
        from backend.inference.benchmark_identity import PLAIN
        from kernel_bench.refs import rows_on_record
        declared = {f for p in all_providers() if p.serves is not None
                    for f in p.serves.output_format} - {PLAIN}
        assert declared
        row = torch.randn(4, 512)
        for record in declared:
            assert rows_on_record(row, record, torch.bfloat16,
                                  tail=64).shape == row.shape, record

    @pytest.mark.parametrize("record", ["fp8_pow2_group128", "mxfp4"])
    def test_the_floor_writes_the_record_the_oracle_reads(self, record):
        # Enough keys that the rows, not the 8192-position rotation table,
        # are the launch's widest operand (`launch_trace.disagreement`).
        for cr in (2, 1):
            _floor_meets_the_oracle(
                self.SIG, {**self.DIMS, "s_slots": 4096, "compress_ratio": cr},
                output=self.RECORDS[record], output_format=record)

    def test_the_record_decodes_back_to_its_rows(self):
        """A second route to the record: the oracle's grid packed by hand
        into the page layout the kernel writes (values for the page, then
        the scales; E2M1 low nibble first), decoded by the provider's
        reader."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.compress_store import _index_key_rows
        from kernel_bench.quant import FORMATS, quantise_groups
        gen = torch.Generator().manual_seed(0)
        rows, page = 70, 64
        x = torch.randn(rows, 128, generator=gen).to(torch.bfloat16)
        slots = torch.arange(rows)
        # MXFP4: 64 packed bytes + four UE8M0 exponents a key.
        q, scale = quantise_groups(x.float(), FORMATS["mxfp4"], 32, "pow2",
                                   x.dtype, eps=6.0 * 2.0 ** -126)
        vals = q.float().reshape(rows, 128)
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        code = ((vals.abs()[..., None] == grid).float().argmax(-1)
                + 8 * (vals < 0).long())
        packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
        exps = (torch.log2(scale).round() + 127).to(torch.uint8)
        cache = torch.zeros(2, page, 68, dtype=torch.uint8)
        flat = cache.view(2, -1)
        for i in range(rows):
            b, w = divmod(i, page)
            flat[b, w * 64:(w + 1) * 64] = packed[i]
            flat[b, page * 64 + w * 4:page * 64 + w * 4 + 4] = exps[i]
        want = (q.float() * scale.unsqueeze(-1)).reshape(rows, 128)
        assert torch.equal(_index_key_rows(cache, slots, page, "mxfp4",
                                           128, torch), want)
        # FP8: 128 e4m3 bytes + one fp32 scale a key.
        sc = torch.rand(rows, generator=gen) + 0.5
        e4 = (x.float() / sc[:, None]).to(torch.float8_e4m3fn)
        cache = torch.zeros(2, page, 132, dtype=torch.uint8)
        flat = cache.view(2, -1)
        for i in range(rows):
            b, w = divmod(i, page)
            flat[b, w * 128:(w + 1) * 128] = e4[i].view(torch.uint8)
            flat[b, page * 128 + w * 4:page * 128 + w * 4 + 4] = (
                sc[i:i + 1].view(torch.uint8))
        assert torch.equal(_index_key_rows(cache, slots, page,
                                           "fp8_pow2_group128", 128, torch),
                           e4.float() * sc[:, None])

    @pytest.mark.parametrize("record", ["fp8_pow2_group128", "mxfp4"])
    def test_vllms_kernel_agrees_with_the_oracle_on_a_gpu(self, record):
        """Where vLLM (or its source tree) and a GPU are there: the launch
        as `_produce_k` makes it, decoded from the cache, against the oracle
        on the bf16 band."""
        import importlib
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("needs a GPU")
        from kernel_bench.driver import _cross_check, _xcheck_basis
        from kernel_bench.env import detect_env
        from kernel_bench.refs import get_ref
        env = detect_env()
        p = next(p for p in _providers(self.SIG)
                 if p.kernel_impl
                 == f"vllm_indexer_k_norm_rope_store_{self.IMPLS[record]}")
        try:
            importlib.import_module(p.import_name)
        except Exception:
            pytest.skip("vLLM's V4.1 ops are not importable here")
        for cr in (2, 1):
            inv = self._cell(record, {**self.DIMS, "compress_ratio": cr,
                                      "s_slots": 512})
            if not p.accepts(inv, env):
                pytest.skip(f"{p.kernel_impl} does not run on this card")
            spec = get_ref(self.SIG)
            inputs = build_inputs(spec, inv, "cuda")
            case = p.build(inv, env, p.convert_inputs(inv, inputs))
            case.to_canonical = p.wrap_output(inv, case.to_canonical)
            x_dtype, x_records = _xcheck_basis(inv, inputs)
            err, rel, cos, _t, _w = _cross_check(
                case, spec.reference(inputs), x_dtype, records=x_records)
            assert err is None, (cr, rel, cos)

    @pytest.mark.regression          # walks DeepSeek-V4.1-Flash
    def test_the_plan_reaches_the_span_at_both_ratios(self):
        jobs, _ = _plan("deepseek-ai/DeepSeek-V4.1-Flash", "prefill", "vllm")
        cells = {(j["dims"]["compress_ratio"], j["dims"]["s_slots"],
                  j["dims"]["d_head"]) for j in jobs
                 if j["signature"] == self.SIG}
        assert cells == {(2, 512, 128), (1, 1024, 128)}, cells


class TestGlmKpoolCompressStore:
    """GLM-5.3-Flash's k-pool indexer compressor (`fused_kpool_compress_store`):
    softmax(gate+ape) pool, two bf16 round-trips, Hadamard-128, per-pool
    pow2-scale FP8 — one launch in each framework (vLLM
    `kpool_compress_and_write_cache`, sglang
    `kpool_softmax_rotate_write_cache`), writing the record the model ships
    (`fp8_pow2_hadamard128`). The rotation is the record format's
    `transform`, a member of the plan the graph grafts from that format, so
    the oracle rotates exactly when the cell's record carries it. Checked
    against the pool recomputed per group in a loop, the Hadamard as the
    explicit matrix and the FP8 grid spelled out, so a test cannot pass by
    sharing the oracle's code."""

    RECORD: ClassVar[str] = "fp8_pow2_hadamard128"
    DIMS: ClassVar[dict] = {"B": 2, "s_slots": 3, "compress_ratio": 4,
                            "d_indexer_head": 128, "kv_block_size": 64}

    def _inv(self, record):
        env = {"activation": "bf16"}
        if record is not None:
            env.update(output="fp8_e4m3", output_format=record)
        return Invocation(op="fused_kpool_compress_store",
                          signature="fused_kpool_compress_store",
                          phase="prefill", dims=dict(self.DIMS),
                          dtype_env=env, tp=1)

    @staticmethod
    def _pooled(inputs):
        torch = pytest.importorskip("torch")
        gate = inputs["gate"].float().view(6, 4, 128)
        keys = inputs["keys"].float().view(6, 4, 128)
        ape, dt = inputs["ape"].float(), inputs["keys"].dtype
        return torch.stack([(keys[g] * torch.softmax(gate[g] + ape, 0)).sum(0)
                            for g in range(6)]).to(dt).float()

    def test_the_span_binds_and_both_frameworks_claim_it(self):
        from backend.inference.op_signatures import op_bindings
        assert (op_bindings()["fused_kpool_compress_store"]
                == "fused_kpool_compress_store")
        ps = _providers("fused_kpool_compress_store")
        assert {p.library for p in ps} == {"vllm", "sglang", "torch"}
        assert all(p.fused_op == "fused_kpool_compress_store" for p in ps)
        for p in ps:
            if p.library != "torch":
                assert p.serves.activation == ("bf16",)
                assert p.serves.enum == (("d_indexer_head", (128,)),)
                # Both kernels write the one record, rotated.
                assert p.serves.output == ("fp8_e4m3",)
                assert p.serves.output_format == (self.RECORD,)

    def test_the_plan_names_the_rotation_as_a_required_member(self):
        from backend.inference.fusion_plans import FUSION_SCHEMES
        nodes = FUSION_SCHEMES["fused_kpool_compress_store"].naive_pattern.nodes
        rotate = [n for n in nodes.values() if n.op == "hadamard_transform"]
        assert len(rotate) == 1 and not rotate[0].optional

    def test_the_oracle_rotates_exactly_when_the_record_does(self):
        torch = pytest.importorskip("torch")
        import math
        from kernel_bench.refs import get_ref
        spec = get_ref("fused_kpool_compress_store")
        # No record: the pooled key, unrotated.
        inputs = build_inputs(spec, self._inv(None), "cpu")
        pooled = self._pooled(inputs)
        assert (spec.reference(inputs) - pooled).abs().max() < 1e-4
        # The shipped record: rotated by the explicit matrix, rounded to
        # bf16, then e4m3 with one pow2 scale a pool (amax floored 1e-4).
        inputs = build_inputs(spec, self._inv(self.RECORD), "cpu")
        pooled = self._pooled(inputs)
        H = torch.ones(1, 1)
        for _ in range(7):
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        rotated = (pooled @ H / math.sqrt(128)).to(torch.bfloat16).float()
        amax = rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        q = (rotated / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        want = q.float() * scale
        got = spec.reference(inputs)
        assert (got - want).abs().max() <= 1e-6 * want.abs().max()
        # The bias belongs INSIDE the softmax: adding it after is another
        # function, and off by O(1) on this data.
        keys = inputs["keys"].float().view(6, 4, 128)
        gate = inputs["gate"].float().view(6, 4, 128)
        shifted = torch.stack([(keys[g] * torch.softmax(gate[g], 0)).sum(0)
                               for g in range(6)])
        assert not torch.allclose(pooled, shifted, atol=1e-2)

    @pytest.mark.parametrize("record", [RECORD, None])
    def test_the_floor_tracks_the_oracle(self, record):
        """The floor pools in bf16 where the oracle pools in fp32: inside
        the bf16 band for a cell with no record, and within one step of the
        record's grid where a value sat at a rounding boundary."""
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref, record_step
        spec = get_ref("fused_kpool_compress_store")
        inv = self._inv(record)
        inputs = build_inputs(spec, inv, "cpu")
        ref = spec.reference(inputs)
        floor = next(p for p in _providers("fused_kpool_compress_store")
                     if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, floor.convert_inputs(inv, inputs))
        err = (case.to_canonical(case.run()) - ref).abs().max()
        band = 8e-3 if record is None else record_step(record)
        assert err / ref.abs().max() <= band, err

    def test_the_cells_are_read_off_the_engine_operands(self):
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        # GLM-5.3-Flash prefill: 256 pools of 4 keys, in the plan's port
        # order — the keys, the gate, the bias table.
        shape_in = [[2, 256, 4, 128], [2, 256, 4, 128], [4, 128]]
        got = identity_shape_from_shape_in(
            "fused_kpool_compress_store", shape_in,
            spec_dims={"kv_block_size": 64},
            operand_ports=_ports("keys", "gate", "pos_table"))
        want = {"B": 2, "s_slots": 256, "compress_ratio": 4,
                "d_indexer_head": 128, "kv_block_size": 64}
        assert got is not None and json.loads(got) == want, got

    @pytest.mark.regression          # walks GLM-5.3-Flash in both frameworks
    def test_the_plan_reaches_the_span(self):
        jobs, dropped = _plan("zai-org/GLM-5.3-Flash", "prefill", "vllm")
        cells = {tuple(sorted(j["dims"].items())) for j in jobs
                 if j["signature"] == "fused_kpool_compress_store"}
        assert cells == {
            (("B", 2), ("compress_ratio", 4), ("d_indexer_head", 128),
             ("kv_block_size", 16), ("s_slots", 256))}, (cells, [
                 d for d in dropped if "kpool" in d])
        # sglang's binding states no page, so its cells plan only where the
        # sweep names one — named, not guessed (the DSA indexer store gets
        # the same).
        jobs, dropped = _plan("zai-org/GLM-5.3-Flash", "prefill", "sglang")
        assert not [j for j in jobs
                    if j["signature"] == "fused_kpool_compress_store"]
        assert any("fused_kpool_compress_store" in d and "states no page"
                   in d for d in dropped)


class TestGlmIndexerQueryRotation:
    """The query side of GLM-5.3-Flash's Hadamard-rotated index-K cache:
    the grafted `hadamard_transform` (sglang `rotate_activation`, its own
    launch) and vLLM's `fwht128_quant_fp8`, the rotation and the FP8 cast in
    one (`fused_fwht128_quant`). Checked against the Sylvester matrix and
    the e4m3 grid spelled out here."""

    DIMS: ClassVar[dict] = {"B": 3, "s_query": 4, "d_model": 128}

    @staticmethod
    def _inv(sig, output):
        return Invocation(op=sig, signature=sig, phase="decode",
                          dims={"B": 3, "s_query": 4, "d_model": 128},
                          dtype_env={"activation": "bf16", "output": output},
                          tp=1)

    @staticmethod
    def _rotated(x):
        torch = pytest.importorskip("torch")
        import math
        H = torch.ones(1, 1)
        for _ in range(7):
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        return x.float() @ H / math.sqrt(128)

    def test_the_spans_bind_and_who_claims_them(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert bound["hadamard_transform"] == "hadamard_transform"
        assert bound["fused_fwht128_quant"] == "fused_fwht128_quant"
        assert {p.library for p in _providers("hadamard_transform")
                } == {"sglang", "torch"}
        assert {p.library for p in _providers("fused_fwht128_quant")
                } == {"vllm", "torch"}
        vllm = next(p for p in _providers("fused_fwht128_quant")
                    if p.library == "vllm")
        # The kernel writes FP8 only: a cell the plan keys at the edge's
        # bf16 (no quantise cast on the rotated query yet) is not its launch.
        assert vllm.serves.output == ("fp8_e4m3",)
        assert vllm.fused_op == "fused_fwht128_quant"

    def test_the_rotation_oracle_and_floor_are_the_matrix(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("hadamard_transform")
        inv = self._inv("hadamard_transform", "bf16")
        inputs = build_inputs(spec, inv, "cpu")
        want = self._rotated(inputs["x"])
        assert (spec.reference(inputs) - want).abs().max() < 1e-5
        floor = next(p for p in _providers("hadamard_transform")
                     if p.library == "torch")
        case = floor.build(inv, TORCH_ENV, inputs)
        got = case.to_canonical(case.run())
        assert (got - want).abs().max() / want.abs().max() < 8e-3

    def test_the_fused_oracle_quantises_only_a_cell_that_writes_fp8(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("fused_fwht128_quant")
        inputs = build_inputs(spec, self._inv("fused_fwht128_quant",
                                              "bf16"), "cpu")
        want = self._rotated(inputs["x"])
        assert (spec.reference(inputs) - want).abs().max() < 1e-5
        inputs = build_inputs(spec, self._inv("fused_fwht128_quant",
                                              "fp8_e4m3"), "cpu")
        rotated = self._rotated(inputs["x"]).to(torch.bfloat16).float()
        amax = rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
        q = (rotated / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        want = (q.float() * scale)
        got = spec.reference(inputs).reshape(want.shape)
        assert (got - want).abs().max() <= 1e-6 * want.abs().max()


class TestQsaIndexerSpans:
    """Qwen QSA's indexer preparation: the query norm+rope
    (`qsa_index_q_norm_rope`) and the key block pool+norm+rope+store
    (`qsa_index_k_compress_store`) — one JIT launch each in sglang, work
    items of vLLM's one `qsa_prepare`. The oracles are checked against
    the spans recomputed per head / per block with the NeoX pair
    definition, so a test cannot pass by sharing the oracle's code."""

    @staticmethod
    def _neox_row(x, w, eps, pos, table):
        torch = pytest.importorskip("torch")
        x = x * torch.rsqrt(x.pow(2).mean() + eps) * (1 + w)
        cos, sin = table[pos][:32], table[pos][32:]
        x1, x2 = x[:32], x[32:64]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[64:]])

    def test_the_spans_bind_and_all_three_arms_claim_them(self):
        from backend.inference.op_signatures import op_bindings
        bound = op_bindings()
        assert bound["qsa_index_q_norm_rope"] == "qsa_index_q_norm_rope"
        assert (bound["qsa_index_k_compress_store"]
                == "qsa_index_k_compress_store")
        assert {p.library for p in _providers("qsa_index_q_norm_rope")
                } == {"vllm", "sglang", "torch"}
        assert {p.library for p in _providers("qsa_index_k_compress_store")
                } == {"vllm", "sglang", "torch"}
        for sig in ("qsa_index_q_norm_rope", "qsa_index_k_compress_store"):
            assert all(p.fused_op == sig for p in _providers(sig))

    def test_the_query_oracle_is_the_per_head_loop(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("qsa_index_q_norm_rope")
        inv = _inv("qsa_index_q_norm_rope", {
            "B": 2, "s_query": 5, "n_indexer_heads": 4,
            "d_indexer_head": 128, "d_rope": 64})
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        q, w, eps, table = (inputs["q"].float(), inputs["weight"].float(),
                            inputs["eps"], inputs["cos_sin_cache"])
        want = torch.stack([torch.cat([
            self._neox_row(q[t, h], w, eps, inputs["positions"][t].item(),
                           table) for h in range(4)])
            for t in range(10)])
        assert (got - want).abs().max() < 1e-5
        # The gain is Gemma's (1 + w): plain rmsnorm·w is another function.
        plain = torch.stack([torch.cat([
            self._neox_row(q[t, h], w - 1, eps,
                           inputs["positions"][t].item(), table)
            for h in range(4)]) for t in range(10)])
        assert not torch.allclose(got, plain, atol=1e-2)

    def test_the_key_oracle_is_the_block_loop(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        spec = get_ref("qsa_index_k_compress_store")
        inv = _inv("qsa_index_k_compress_store", {
            "B": 2, "s_query": 8, "d_indexer_head": 128, "compress_ratio": 4,
            "d_rope": 64, "kv_block_size": 64, "s_slots": 2})
        inputs = build_inputs(spec, inv, "cpu")
        got = spec.reference(inputs)
        raw, w, eps, table = (inputs["raw_key"].float(),
                              inputs["weight"].float(), inputs["eps"],
                              inputs["cos_sin_cache"])
        dt = inputs["raw_key"].dtype
        want = torch.stack([
            self._neox_row(raw[g * 4:(g + 1) * 4].mean(0).to(dt).float(),
                           w, eps, inputs["positions"][g * 4].item(), table)
            for g in range(4)])
        assert (got - want).abs().max() < 1e-4

    def test_the_floors_track_the_oracles_inside_the_bf16_band(self):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        for sig, dims in (
            ("qsa_index_q_norm_rope",
             {"B": 2, "s_query": 5, "n_indexer_heads": 4,
              "d_indexer_head": 128, "d_rope": 64}),
            ("qsa_index_k_compress_store",
             {"B": 2, "s_query": 8, "d_indexer_head": 128,
              "compress_ratio": 4, "d_rope": 64, "kv_block_size": 64,
              "s_slots": 2}),
        ):
            spec = get_ref(sig)
            inv = _inv(sig, dims)
            inputs = build_inputs(spec, inv, "cpu")
            ref = spec.reference(inputs)
            floor = next(p for p in _providers(sig) if p.library == "torch")
            case = floor.build(inv, TORCH_ENV, floor.convert_inputs(inv, inputs))
            err = (case.to_canonical(case.run()) - ref).abs().max()
            assert err / ref.abs().max() < 8e-3, (sig, err)

    def test_the_cells_are_read_off_the_engine_operands(self):
        import json
        from backend.inference.benchmark_identity import (
            identity_shape_from_shape_in)
        cases = {
            # Qwen3.8-Flash-Next prefill: the head-split indexer query.
            "qsa_index_q_norm_rope": (
                [[2, 4, 1024, 128], [128]],
                {"B": 2, "n_indexer_heads": 4, "s_query": 1024,
                 "d_indexer_head": 128, "d_rope": 64}),
            # The raw keys of the same step, 256 completed blocks of 4.
            "qsa_index_k_compress_store": (
                [[2, 1024, 128], [128]],
                {"B": 2, "s_query": 1024, "d_indexer_head": 128,
                 "compress_ratio": 4, "s_slots": 256, "d_rope": 64,
                 "kv_block_size": 64}),
        }
        for op, (shape_in, want) in cases.items():
            # The rotation width, the block size and the cache's page ride
            # no tensor of these spans: the plan's nodes and the deployment
            # state them.
            got = identity_shape_from_shape_in(
                op, shape_in,
                spec_dims={"d_rope": 64, "compress_ratio": 4, "s_slots": 256,
                           "kv_block_size": 64},
                operand_ports=_ports("q" if op == "qsa_index_q_norm_rope"
                                   else "raw_key", "weight"))
            assert got is not None and json.loads(got) == want, (op, got)

    @pytest.mark.regression          # walks Qwen3.8-Flash-Next in both frameworks
    def test_the_plan_reaches_both_spans(self):
        for fw in ("vllm", "sglang"):
            jobs, dropped = _plan("Qwen/Qwen3.8-Flash-Next", "prefill", fw)
            q_cells = {tuple(sorted(j["dims"].items())) for j in jobs
                       if j["signature"] == "qsa_index_q_norm_rope"}
            assert q_cells == {
                (("B", 2), ("d_indexer_head", 128), ("d_rope", 64),
                 ("n_indexer_heads", 4), ("s_query", 1024))}, (fw, q_cells)
            k_jobs = [j for j in jobs
                      if j["signature"] == "qsa_index_k_compress_store"]
            if fw == "vllm":
                assert {tuple(sorted(j["dims"].items())) for j in k_jobs} == {
                    (("B", 2), ("compress_ratio", 4), ("d_indexer_head", 128),
                     ("d_rope", 64), ("kv_block_size", 16),
                     ("s_query", 1024), ("s_slots", 256))}
            else:
                # sglang's binding states no page: the paged cell is named,
                # not keyed under a guessed one.
                assert not k_jobs
                assert any("qsa_index_k_compress_store" in d
                           and "states no page" in d for d in dropped)


class TestSegregatedRecordReader:
    """`_segregated_record_rows` decodes the record layout the store
    kernels write: within a block, `page` records of `d` FP8 values first,
    then `page` fp32 scales (vLLM kpool_compress.py:237-248 and
    fused_compress_quant_cache.py:750-756; sglang kpool_fp8_index.py:
    1001-1020). An interleaved `[d+4]`-per-row reading — the failure this
    test exists to catch — decodes other bytes."""

    def test_the_reader_decodes_the_kernels_layout(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.compress_store import _segregated_record_rows
        page, d = 4, 128
        block = torch.zeros(1, page * (d + 4), dtype=torch.uint8)
        # Token 1: values 1.0 everywhere (0x38 is e4m3's 1.0), scale 2.0.
        block[0, 1 * d:2 * d] = 0x38
        block[0, page * d + 4:page * d + 8] = torch.tensor(
            [2.0]).view(torch.uint8)
        # Token 3: values 0.5 (0x30 in e4m3), scale 0.25.
        block[0, 3 * d:4 * d] = 0x30
        block[0, page * d + 12:page * d + 16] = torch.tensor(
            [0.25]).view(torch.uint8)
        slots = torch.tensor([1, 3])
        got = _segregated_record_rows(block, slots, page, d)
        assert torch.allclose(got[0], torch.full((d,), 2.0))
        assert torch.allclose(got[1], torch.full((d,), 0.125))
        # The interleaved reading is not a legal answer: token 1's "scale"
        # would come out of bytes 128:132 — the first record's values here.
        interleaved = block[0, 128:132].contiguous().view(torch.float32)
        assert not torch.allclose(interleaved, torch.tensor([2.0]))


class TestConstructionHelpers:
    """The operand constructions the framework builds run before
    `_require_cuda` — exercised on CPU so a regression there is red here
    rather than as a NameError / shape error on the bench box."""

    def test_the_csa2_fused_row_is_kv_only_at_ratio_1(self):
        """At ratio 1 the kernel reads kv alone (the row is 512 wide,
        fused_compress_quant_cache.py:51, :65) — writing the gate beside it
        is the shape error this test would have caught."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.compress_store import (
            _dsv41_fused_kv_score)
        kv = torch.randn(5, 512)
        score = torch.randn(5, 512)
        row1 = _dsv41_fused_kv_score(kv, score, 1, torch)
        assert row1.shape == (5, 512) and row1.dtype == torch.float32
        assert torch.equal(row1, kv)
        row2 = _dsv41_fused_kv_score(kv, score, 2, torch)
        assert row2.shape == (5, 1024)
        assert torch.equal(row2[:, :512], kv) and torch.equal(
            row2[:, 512:], score)

    def test_the_sglang_key_arms_operands(self):
        """Ring, group slots with the group-start slot in column 0, the
        (t, h, w) coordinates of a text forward, the flat buffer — built
        here on CPU, so the arm's construction is testable off the box."""
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.providers.compress_store import _sgl_qsa_k_operands
        spec = get_ref("qsa_index_k_compress_store")
        inv = _inv("qsa_index_k_compress_store", {
            "B": 1, "s_query": 8, "d_indexer_head": 128,
            "compress_ratio": 4, "d_rope": 64, "kv_block_size": 64,
            "s_slots": 2})
        inputs = build_inputs(spec, inv, "cpu")
        ring, group_locs, rope_pos, buffer, d, cr, groups = (
            _sgl_qsa_k_operands(inputs, torch.bfloat16, torch))
        assert ring.shape == (8, 128) and group_locs.shape == (2, 4)
        assert group_locs[:, 0].tolist() == [0, 4]
        assert rope_pos.shape == (8, 3)
        assert rope_pos[3].tolist() == [3, 3, 3]
        assert buffer.shape == (2, 128) and d == 128 and cr == 4
        assert groups == 2


class TestProductionLaunchShapes:
    """The launch-shape constructions the framework builds run before
    `_require_cuda`: the full token grids, the per-request state paging,
    and the per-token candidate windows. Exercised on CPU — a rows-only
    grid, a one-block state, or a compacted candidate set is the failure
    each assertion names."""

    def test_the_store_launch_is_one_program_per_token(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.compress_store import _pooled_store_launch
        latent = torch.randn(4, 8)          # 2 requests x 2 pools
        rows, pos, slots = _pooled_store_launch(latent, 2, 2, torch)
        # The grid is every token (8), not every pool (4); positions
        # restart per request; only the closing tokens carry a slot, in
        # canonical (request, group) order.
        assert rows.shape == (8, 8)
        assert pos.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
        assert slots.tolist() == [-1, 0, -1, 1, -1, 2, -1, 3]
        assert torch.equal(rows[slots >= 0], latent)
        assert rows[slots < 0].abs().sum() == 0

    def test_the_overlap_state_is_paged_at_productions_four_rows(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        from kernel_bench.providers.compress_store import (
            _overlap_state_and_meta)
        spec = get_ref("fused_overlap_compress_norm_rope_store")
        inv = _inv("fused_overlap_compress_norm_rope_store", {
            "B": 2, "s_slots": 3, "Cape": 4, "d_head": 512, "d_rope": 64,
            "kv_block_size": 64})
        inputs = build_inputs(spec, inv, "cpu")
        (state, positions, _state_slots, _requests, block_table,
         kv_slots) = _overlap_state_and_meta(inv, inputs, torch)
        # C4 state blocks are 4 rows (compressor.py:167-169), each
        # request's blocks its own — not one chunk-wide block.
        assert state.shape == (2 * 3, 4, 4 * 512)
        assert block_table.tolist() == [[0, 1, 2], [3, 4, 5]]
        assert positions.tolist()[:5] == [0, 1, 2, 3, 4]
        assert positions.tolist()[12:14] == [0, 1]
        assert kv_slots.tolist()[3] == 0 and kv_slots.tolist()[12 + 3] == 3
        # The state row of request 1's position 0 sits in block 3 — its
        # own table, so window 0 of request 1 gathers no request-0 rows.
        assert torch.equal(state[3, 0, :1024],
                           inputs["kv"].view(2, 12, 1024)[1, 0].float())

    def test_the_kpool_candidates_are_one_window_per_token(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.compress_store import (
            _kpool_candidate_windows)
        keys = torch.arange(8 * 4, dtype=torch.float32).view(8, 4)
        gate = torch.zeros(8, 4)
        slot_k, _slot_score, loc, mask = _kpool_candidate_windows(
            keys, gate, 2, torch)
        # One candidate per token, each the window ENDING at it (leading
        # rows clamped to the chunk start, as production's own gather does:
        # token 0's window is rows 0 and 1).
        assert slot_k.shape == (8, 2, 4)
        assert torch.equal(slot_k[1], keys[0:2])
        assert torch.equal(slot_k[0], keys[0:2])
        assert loc.tolist() == [-1, 0, -1, 1, -1, 2, -1, 3]
        assert mask.tolist() == [False, True, False, True,
                                 False, True, False, True]


class TestDeclinedCells:
    """A provider's page / record promises are enforced, not commented:
    `accepts()` does not read `kv_block_sizes`, so these arms refuse at
    `impl_id` — the resolution that runs before any input is allocated."""

    def test_a_cell_at_an_unserved_page_is_refused(self):
        # The planned V4.1 store cell is keyed at the binding's default
        # page 16; the vLLM arms serve the FlashMLA page. A provider that
        # answered there would measure a paging layout that never happened.
        from kernel_bench.providers.compress_store import _served_page_only
        check = _served_page_only((64,))
        inv16 = _inv("fused_compress_rope_store", {
            "B": 1, "s_slots": 4, "d_head": 512, "d_rope": 64,
            "compress_ratio": 2, "kv_block_size": 16})
        with pytest.raises(NotImplementedError):
            check(inv16)
        inv64 = _inv("fused_compress_rope_store", {
            "B": 1, "s_slots": 4, "d_head": 512, "d_rope": 64,
            "compress_ratio": 2, "kv_block_size": 64})
        assert check(inv64) == ""
        # Wired into the paged arms: impl_id resolves the refusal. Page 8
        # is outside every arm's declared set (the MLA record caches serve
        # 64, QSA's compressed cache 16/32/64).
        inv8 = _inv("fused_compress_rope_store", {
            "B": 1, "s_slots": 4, "d_head": 512, "d_rope": 64,
            "compress_ratio": 2, "kv_block_size": 8})
        for sig in ("fused_compress_rope_store",
                    "fused_compress_norm_rope_store",
                    "fused_overlap_compress_norm_rope_store",
                    "fused_kpool_compress_store",
                    "qsa_index_k_compress_store"):
            for p in _providers(sig):
                if p.library == "torch" or not p.serves.kv_block_sizes:
                    continue
                with pytest.raises(NotImplementedError):
                    p.impl_id(inv8)

    def test_the_v41_store_declines_sm100s_mxfp8_record(self):
        from kernel_bench.providers.compress_store import (
            _decline_v41_sm100_record)
        with pytest.raises(NotImplementedError):
            _decline_v41_sm100_record((10, 0))
        _decline_v41_sm100_record((9, 0))


def _fake_module(monkeypatch, name, **funcs):
    """A module injected at `name`'s import path, carrying `funcs` — a
    framework build driven on CPU, so the operands it would launch with can
    be read off the call."""
    import sys
    import types
    mod = types.ModuleType(name)
    for k, v in funcs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


class TestSglangClampedSwiglu:
    """sglang v0.5.22's clamped SwiGLU: `DeepseekV2MLP.forward` hands the
    contiguous gate_up row and a fresh half-width output to
    `silu_and_mul_clamp` (srt/models/deepseek_v2.py:481-484; JIT
    `SiluAndMulClampKernel`, silu_and_mul_masked_post_quant.cuh:334-375)."""

    SIG = "silu_and_mul_with_clamp"
    IMPL = "sglang_silu_and_mul_clamp"

    def _sgl(self):
        return next(p for p in _providers(self.SIG)
                    if p.kernel_impl == self.IMPL)

    @staticmethod
    def _cell(dims=None, env=None, params=None):
        return Invocation(
            op="silu_and_mul_with_clamp", signature="silu_and_mul_with_clamp",
            phase="decode", tp=1,
            dims=dict({"B": 4, "s_query": 1, "d_ffn": 64}, **(dims or {})),
            dtype_env=dict({"activation": "bf16", "output": "bf16"},
                           **(env or {})),
            kernel_params=({"clamp_limit": 10.0} if params is None
                           else params))

    def test_sglangs_launch_claims_the_span(self):
        p = self._sgl()
        assert p.library == "sglang" and p.fused_op == self.SIG
        assert {q.library for q in _providers(self.SIG)} == {
            "vllm", "sglang", "torch"}

    def test_the_template_geometry_is_what_it_accepts(self):
        """bf16 only (an fp16 row is cast to bf16 before the clamp,
        .cuh:239), written at bf16 (a cell keyed at FP8 is the fused down
        path's quantising launch, deepseek_v2.py:390-422), the output width
        a whole number of 8-element vectors (.cuh:360), and the bound from
        the cell."""
        p = self._sgl()
        assert p.serves.accepts(self._cell())
        assert p.serves.accepts(self._cell({"d_ffn": 2056}))
        for bad in (self._cell({"d_ffn": 36}),
                    self._cell(env={"activation": "fp16", "output": "fp16"}),
                    self._cell(env={"output": "fp8_e4m3"}),
                    self._cell(params={})):
            assert not p.serves.accepts(bad), bad

    def test_the_build_hands_productions_operands(self, monkeypatch):
        """`(gate_up, x, limit)`: the cell's packed row as is, a fresh
        `[M, I]` output at its dtype the canonical result, the model's bound
        as a float."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        seen = {}

        def silu_and_mul_clamp(inp, out, limit):
            seen.update(inp=inp, out=out, limit=limit)
            out.fill_(3.0)
        _fake_module(monkeypatch, "sglang.kernels.ops.moe.dsv4",
                     silu_and_mul_clamp=silu_and_mul_clamp)
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        inv = self._cell({"B": 3, "s_query": 2, "d_ffn": 16})
        inputs = build_inputs(get_ref(self.SIG), inv, "cpu")
        case = na._sgl_silu_and_mul_clamp_build(inv, TORCH_ENV, inputs)
        got = case.to_canonical(case.run())
        assert torch.equal(seen["inp"], inputs["x"])
        assert seen["inp"].is_contiguous() and seen["inp"].shape == (6, 32)
        assert seen["out"].shape == (6, 16)
        assert seen["out"].dtype == seen["inp"].dtype == torch.bfloat16
        assert seen["limit"] == 10.0 and isinstance(seen["limit"], float)
        assert got is seen["out"] and bool((got == 3.0).all())


class TestSglangSituGate:
    """sglang v0.5.22's SiTU gate: `SituAndMul.forward_cuda`
    (srt/layers/activation.py:209-212) is the JIT `SituAndMulKernel`
    (kimi_k3/situ_and_mul.cuh:113-170) through `situ_and_mul(x, None, beta,
    linear_beta)`, `None` the linear beta's "unset"."""

    SIG = "situ_and_mul"
    IMPL = "sglang_situ_and_mul"

    def _sgl(self):
        return next(p for p in _providers(self.SIG)
                    if p.kernel_impl == self.IMPL)

    @staticmethod
    def _cell(dims=None, env=None, params=None):
        return Invocation(
            op="situ_and_mul", signature="situ_and_mul", phase="prefill",
            tp=1, dims=dict({"B": 2, "s_query": 3, "d_ffn": 64},
                            **(dims or {})),
            dtype_env=dict({"activation": "bf16", "output": "bf16"},
                           **(env or {})),
            kernel_params=({"situ_beta": 4.0, "situ_linear_beta": 25.0}
                           if params is None else params))

    def test_sglangs_launch_claims_the_span(self):
        p = self._sgl()
        assert p.library == "sglang" and p.fused_op == self.SIG
        assert {q.library for q in _providers(self.SIG)} == {
            "vllm", "sglang", "torch"}

    def test_the_template_geometry_is_what_it_accepts(self):
        p = self._sgl()
        assert p.serves.accepts(self._cell())
        assert p.serves.accepts(self._cell({"d_ffn": 3072 + 8}))
        assert p.serves.accepts(self._cell(params={"situ_beta": 4.0}))
        for bad in (self._cell({"d_ffn": 36}),
                    self._cell(env={"activation": "fp16", "output": "fp16"}),
                    self._cell(env={"output": "fp8_e4m3"}),
                    self._cell(params={"situ_linear_beta": 25.0})):
            assert not p.serves.accepts(bad), bad

    @pytest.mark.parametrize("linear", [25.0, None])
    def test_the_build_hands_productions_operands(self, monkeypatch, linear):
        """The packed row as is, no output (the wrapper allocates it), the
        gate's beta, and the linear beta as the model publishes it — `None`
        when it publishes none, which the wrapper turns into
        `has_linear_beta=False`, not a value."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        seen = {}

        def situ_and_mul(inp, out, beta, linear_beta):
            seen.update(inp=inp, out=out, beta=beta, linear=linear_beta)
            return torch.zeros(inp.shape[0], inp.shape[1] // 2)
        _fake_module(monkeypatch, "sglang.kernels.ops.activation",
                     situ_and_mul=situ_and_mul)
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        monkeypatch.setattr(torch.cuda, "get_device_capability",
                            lambda *a: (9, 0))
        params = {"situ_beta": 4.0}
        if linear is not None:
            params["situ_linear_beta"] = linear
        inv = self._cell(params=params)
        inputs = build_inputs(get_ref(self.SIG), inv, "cpu")
        case = na._sgl_situ_and_mul_build(inv, TORCH_ENV, inputs)
        got = case.to_canonical(case.run())
        assert torch.equal(seen["inp"], inputs["x"])
        assert seen["inp"].is_contiguous() and seen["inp"].shape == (6, 128)
        assert seen["out"] is None and seen["beta"] == 4.0
        assert seen["linear"] == linear
        assert got.shape == (6, 64)

    def test_sm100_declines_a_width_its_wider_vector_does_not_divide(
            self, monkeypatch):
        """From SM100 the kernel's vector is 32 bytes (utils.cuh:149), so a
        bf16 output width of 8 mod 16 fails its check (situ_and_mul.cuh:149)
        there and runs below it."""
        torch = pytest.importorskip("torch")
        import kernel_bench.providers.norm_act as na
        from kernel_bench.refs import get_ref
        _fake_module(monkeypatch, "sglang.kernels.ops.activation",
                     situ_and_mul=lambda *a: None)
        monkeypatch.setattr(na, "_require_cuda", lambda: None)
        inv = self._cell({"d_ffn": 24})
        inputs = build_inputs(get_ref(self.SIG), inv, "cpu")
        for cap, ok in (((9, 0), True), ((10, 0), False)):
            monkeypatch.setattr(torch.cuda, "get_device_capability",
                                lambda *a, cap=cap: cap)
            if ok:
                na._sgl_situ_and_mul_build(inv, TORCH_ENV, inputs)
            else:
                with pytest.raises(NotImplementedError):
                    na._sgl_situ_and_mul_build(inv, TORCH_ENV, inputs)


class TestSglangRopeRunsTheKernelSglangRuns:
    """`sgl_apply_rope_cos_sin_cache` must time the kernel sglang v0.5.22
    runs for a 64/128/256/512 head on CUDA — its own
    `sglang.kernels.ops.attention.rope.apply_rope_with_cos_sin_cache_inplace`
    on `[tokens, heads, head_size]` views with the fp32 cache — not
    `sgl_kernel.rotary_embedding`, the fallback that wants the cache at the
    query's dtype and failed every cell. The fake stands in for the kernel
    and refuses anything but what sglang hands it, then rotates in place."""

    SIG = "rope"

    def test_the_build_hands_sglangs_own_kernel_its_fp32_cache(self, monkeypatch):
        import sys
        import types

        torch = pytest.importorskip("torch")
        from kernel_bench.providers import rope
        from kernel_bench.refs import _rope_rotate, get_ref
        monkeypatch.setattr(rope, "_require_cuda", lambda: None)
        seen = []

        def apply_rope_with_cos_sin_cache_inplace(q, k, cos_sin_cache, positions,
                                                  *, is_neox, rope_dim=0,
                                                  fused_args=None):
            assert cos_sin_cache.dtype == torch.float32
            assert q.dim() == 3 and k.dim() == 3
            seen.append((tuple(q.shape), tuple(k.shape)))
            rd = cos_sin_cache.size(-1)
            q.copy_(_rope_rotate(q.float(), positions, cos_sin_cache, rd, is_neox))
            k.copy_(_rope_rotate(k.float(), positions, cos_sin_cache, rd, is_neox))

        name = "sglang.kernels.ops.attention.rope"
        parts = name.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
        mod = types.ModuleType(name)
        mod.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace
        monkeypatch.setitem(sys.modules, name, mod)

        inv = Invocation(op="fused_qk_rope", signature=self.SIG, phase="prefill",
                         dims={"B": 2, "s_query": 3, "n_heads": 4,
                               "n_kv_heads": 2, "d_rope": 64},
                         dtype_env={"activation": "bf16", "output": "bf16"}, tp=1)
        x = build_inputs(get_ref(self.SIG), inv)
        [p] = [p for p in all_providers()
               if p.kernel_impl == "sgl_apply_rope_cos_sin_cache"]
        assert p.import_name == "sglang"
        case = p.build(inv, TORCH_ENV, x)
        got = case.to_canonical(case.run())
        assert seen == [((6, 4, 64), (6, 2, 64))]
        assert torch.allclose(got, get_ref(self.SIG).reference(x),
                              rtol=2e-2, atol=2e-2)


class TestFlashMLARecordOracle:
    """DeepSeek-V4's FlashMLA record (`cache_formats/
    fp8_ue8m0_group64_bf16_rope_tail.yaml`: FP8 e4m3 NoPE channels with one
    UE8M0 exponent per 64, the last 64 rotated channels BF16) written by the
    oracle the way its writers write it (`refs.rows_on_record`), so a cell
    whose launch writes it is compared record to record rather than an fp32
    row against a quantised one."""

    RECORD: ClassVar[str] = "fp8_ue8m0_group64_bf16_rope_tail"

    @staticmethod
    def _by_hand(row, group=64, tail=64):
        """The record by a second route: the exponent as an integer
        `ceil(log2(max(amax, 1e-4) / 448))` from frexp, each value the
        nearest finite e4m3 (ties to the even code) found by enumerating
        all 256 codes — no `torch.float8` cast — and the tail cast to bf16.
        Returns `(codes [rows, d - tail] uint8, exponents [rows, groups],
        tail bf16, decoded fp32)`."""
        import torch
        body = row[:, :-tail].to(torch.bfloat16).float()
        rows, width = body.shape
        g = body.view(rows, width // group, group)
        amax = g.abs().amax(-1).clamp_min(1e-4) / 448.0
        m, e = torch.frexp(amax)
        exp = torch.where(m == 0.5, e - 1, e)          # ceil(log2) exactly
        v = g / torch.exp2(exp.float())[..., None]
        codes = torch.arange(256, dtype=torch.int32).to(torch.uint8)
        values = codes.view(torch.float8_e4m3fn).float()
        finite = values.isfinite()
        codes, values = codes[finite], values[finite]
        gap = (v[..., None] - values).abs()
        best = gap.min(-1, keepdim=True).values
        tied = gap == best
        # ties to even: of the tied codes, the one with an even mantissa
        even = tied & (codes.to(torch.int32) % 2 == 0)
        pick = torch.where(even.any(-1, keepdim=True), even, tied)
        index = pick.int().argmax(-1)
        q = values[index]
        decoded = (q * torch.exp2(exp.float())[..., None]).view(rows, width)
        held = row[:, -tail:].to(torch.bfloat16)
        return (codes[index].view(rows, width), exp, held,
                torch.cat([decoded, held.float()], dim=-1))

    def _row(self, rows=96, seed=0):
        torch = pytest.importorskip("torch")
        gen = torch.Generator().manual_seed(seed)
        row = torch.randn(rows, 512, generator=gen)
        # Groups whose amax sits exactly on, and a hair over, a power-of-two
        # boundary of `amax / 448` — where rounding up and rounding to
        # nearest part — and one all-zero group (the 1e-4 floor).
        row[0, :64] *= 0.0
        row[1, 64:128] = row[1, 64:128] / row[1, 64:128].abs().max() * 448.0
        row[2, 128:192] = (row[2, 128:192] / row[2, 128:192].abs().max()
                           * 448.0 * 1.01)
        return row

    def test_the_grid_is_read_off_the_registered_format(self, monkeypatch):
        """The oracle's grid follows the format's declaration: the same row
        with the registry entry's group set to 32, or its tail dropped,
        comes out on that other grid."""
        torch = pytest.importorskip("torch")
        import dataclasses
        from backend.arch import cache_formats
        from kernel_bench.refs import rows_on_record
        row = self._row()
        got = rows_on_record(row, self.RECORD, torch.bfloat16, tail=64)
        assert torch.equal(got, self._by_hand(row)[3])
        fmt = cache_formats.CACHE_FORMATS[self.RECORD]
        monkeypatch.setitem(
            cache_formats.CACHE_FORMATS, self.RECORD,
            dataclasses.replace(fmt, scale=dataclasses.replace(fmt.scale,
                                                               group=32)))
        assert torch.equal(
            rows_on_record(row, self.RECORD, torch.bfloat16, tail=64),
            self._by_hand(row, group=32)[3])
        monkeypatch.setitem(cache_formats.CACHE_FORMATS, self.RECORD,
                            dataclasses.replace(fmt, tail=None))
        untailed = rows_on_record(row, self.RECORD, torch.bfloat16)
        # the rotated channels are now quantised too
        assert not torch.equal(untailed[:, -64:], got[:, -64:])
        assert torch.equal(untailed[:, :448], got[:, :448])

    def test_the_scale_rounds_up_and_floors_an_empty_group(self):
        torch = pytest.importorskip("torch")
        from kernel_bench.refs import rows_on_record
        row = self._row()
        _codes, exp, _tail, _ = self._by_hand(row)
        # amax / 448 exactly 2^0 keeps exponent 0; 1% over takes 2^1; the
        # empty group takes ceil(log2(1e-4 / 448)) = ceil(-22.09) = -22.
        assert exp[1, 1].item() == 0 and exp[2, 2].item() == 1
        assert exp[0, 0].item() == -22
        got = rows_on_record(row, self.RECORD, torch.bfloat16, tail=64)
        assert got[1, 64:128].abs().max().item() == 448.0
        assert torch.equal(got[0, :64], torch.zeros(64))

    def test_a_hand_built_record_decodes_to_the_oracle(self):
        """The record packed by hand into the page layout the stores write
        (data rows at `slot · 576`, exponent bytes biased by 127 from
        `page · 576`, a pad byte), read back by the providers' decoder,
        is the oracle's row bit for bit."""
        torch = pytest.importorskip("torch")
        from kernel_bench.providers.rope import _dsv4_swa_rows
        from kernel_bench.refs import rows_on_record
        row = self._row(rows=40)
        codes, exp, tail, _ = self._by_hand(row)
        page = 16
        cache = torch.zeros(-(-40 // page), page * 584, dtype=torch.uint8)
        for slot in range(40):
            b, w = divmod(slot, page)
            cache[b, w * 576:w * 576 + 448] = codes[slot]
            cache[b, w * 576 + 448:(w + 1) * 576] = tail[slot].view(
                torch.uint8)
            base = page * 576 + w * 8
            cache[b, base:base + 7] = (exp[slot] + 127).to(torch.uint8)
        got = _dsv4_swa_rows(cache, torch.arange(40), page, 512, 64)
        assert torch.equal(
            got, rows_on_record(row, self.RECORD, torch.bfloat16, tail=64))

    def test_a_record_cell_is_judged_record_to_record(self):
        """The negative controls the band must keep rejecting, at the
        record cell's band (`driver._xcheck_band`: bf16's typical, a worst
        of one e4m3 step): the fp32 row the oracle used to be, and the
        record with its exponents rounded to nearest instead of up. A
        record that differs only where a bf16 rounding before the write
        moved a value across a step (sglang's store quantises the fp32
        row) is inside it."""
        torch = pytest.importorskip("torch")
        from kernel_bench.driver import _cross_check, _xcheck_band
        from kernel_bench.providers import BenchCase
        from kernel_bench.refs import rows_on_record
        gen = torch.Generator().manual_seed(1)
        x = torch.randn(4096, 512, generator=gen).to(torch.bfloat16).float()
        w = torch.randn(512, generator=gen).to(torch.bfloat16).float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
        oracle = rows_on_record(y, self.RECORD, torch.bfloat16, tail=64)
        band = _xcheck_band("bf16", (self.RECORD,))
        assert band.worst == 0.125 and band.typical == _xcheck_band(
            "bf16").typical

        def judged(out):
            case = BenchCase(run=lambda: out, to_canonical=lambda o: o)
            return _cross_check(case, oracle, "bf16", records=(self.RECORD,))[0]

        assert judged(oracle.clone()) is None
        g = y[:, :448].to(torch.bfloat16).float().view(-1, 7, 64)
        sc = torch.exp2(torch.log2(g.abs().amax(-1) / 448.0).round())
        nearest = torch.cat([
            ((g / sc[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
             .float() * sc[..., None]).view(-1, 448), oracle[:, 448:]], -1)
        assert judged(nearest) is not None
        assert judged(y) is not None
        # fp32 quantised (no bf16 rounding first): one step on a few values
        fp32_written = torch.cat([
            rows_on_record(y, self.RECORD, torch.float32, tail=64)[:, :448],
            oracle[:, 448:]], -1)
        differs = (fp32_written != oracle).float().mean().item()
        assert 0.0 < differs < 0.05
        assert judged(fp32_written) is None
        # ... which the dtype's own worst bound alone would have refused
        assert _cross_check(BenchCase(run=lambda: fp32_written,
                                      to_canonical=lambda o: o),
                            oracle, "bf16")[0] is not None

    @pytest.mark.parametrize("sig,dims", [
        ("fused_norm_rope", {"B": 2, "s_query": 2048, "d_head": 512,
                             "d_rope": 64}),
        ("fused_compress_rope_store", {"B": 2, "s_slots": 2048,
                                       "d_head": 512, "d_rope": 64,
                                       "compress_ratio": 2,
                                       "kv_block_size": 64}),
        ("fused_compress_norm_rope_store", {"B": 2, "s_slots": 4096,
                                            "d_head": 512, "d_rope": 64,
                                            "compress_ratio": 2,
                                            "kv_block_size": 64}),
        ("fused_overlap_compress_norm_rope_store", {
            "B": 2, "s_slots": 256, "Cape": 4, "d_head": 512, "d_rope": 64,
            "kv_block_size": 64}),
    ])
    def test_each_floor_writes_the_record_the_oracle_writes(self, sig, dims):
        """The floor pays the launch's quantisation (its latency is the
        record's precision) and lands on the oracle's record."""
        trace = _floor_meets_the_oracle(sig, dims, output="fp8_e4m3",
                                        output_format=self.RECORD)
        # the e4m3 values themselves pass through the launch
        assert any("float8" in o.dtype for c in trace.calls
                   for o in c.operands), sig


def _h100_plan(model_id, phase, framework):
    """The plan the H100 queue builds: the framework's binding deployed on
    the card, so `device:`-conditioned fusions apply."""
    from backend.inference.framework_bindings import (ServingDeployment,
                                                      get_binding)
    from kernel_bench.plan import build_jobs
    binding = get_binding(framework).deployed(ServingDeployment.at(
        gpu_sku="H100-SXM", activation_dtype="BF16"))
    jobs, _dropped = build_jobs(avg_output_tokens=128, model_id=model_id,
                                batch_size=2, seq_len=1024, phase=phase,
                                framework=framework, binding=binding,
                                gpu="H100-SXM")
    return jobs


def _serving(job):
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers
    inv = Invocation.from_job(job)
    return sorted(p.kernel_impl for p in all_providers()
                  if p.signature == job["signature"] and p.serves.accepts(inv))


@pytest.mark.regression          # walks DeepSeek-V4.1-Flash in prefill
def test_vllm_runs_the_compressor_pool_inside_its_state_taking_launch():
    # vLLM's DeepSeek-V4.1 compressor is ONE launch, `fused_save_compress_norm`
    # (fused_compress_quant_cache.py:22), which writes the open group's fp32
    # ring: the plan absorbs the state-carrying `weighted` multiply into it,
    # so no plain multiply carries the state, and the fused cell is served by
    # the vLLM kernel that takes the ring.
    jobs = _h100_plan("deepseek-ai/DeepSeek-V4.1-Flash", "prefill", "vllm")
    assert not [j for j in jobs if j["signature"] == "mul"
                and j.get("carries_state")]
    fused = [j for j in jobs if j["signature"] == "fused_save_compress_norm"
             and j.get("carries_state")]
    assert fused
    assert all("vllm_fused_save_compress_norm" in _serving(j) for j in fused)


@pytest.mark.regression          # walks DeepSeek-V4.1-Flash in prefill
def test_sglangs_unfused_compressor_multiply_is_a_counted_gap():
    # On an H100 sglang pools DeepSeek-V4.1's low-ratio compressor in eager
    # torch: its fused low-ratio kernels need sm100 (`fused_low_ratio_
    # compress_supported`, srt/layers/attention/dsv4/dsv41_sparse.py:101-110),
    # so `_low_ratio_compress_torch` (deepseek_v4_backend.py:3273-3309)
    # gathers the carried rows and runs `pool_pairs`'s softmax, multiply and
    # sum (dsv41_sparse.py:169-171). The multiply is a real launch that
    # carries the state, and no `mul` kernel takes one: a counted gap.
    jobs = _h100_plan("deepseek-ai/DeepSeek-V4.1-Flash", "prefill", "sglang")
    carried = [j for j in jobs if j["signature"] == "mul"
               and j.get("carries_state")]
    assert carried and all("weighted" in j["node_spec_id"] for j in carried)
    assert all(_serving(j) == [] for j in carried)


class TestVllmPerTokenNormQuant:
    """vLLM's RMSNorm + dynamic per-token FP8 launch
    (`rms_norm_dynamic_per_token_quant`), the one its compile pass puts in
    place of the norm and the cast. Ling-2.6's per-token FP8 activation
    planned norm cells nothing declared. The build is driven on CPU against
    a stand-in kernel that computes what vLLM's does, so what is checked is
    the operands the build hands it (the residual when the cell has one) and
    that its pair reads back on the oracle's grid."""

    @staticmethod
    def _kernel(calls):
        def rms_norm_dynamic_per_token_quant(x, w, eps, qdtype, scale_ub=None,
                                             residual=None):
            import torch
            calls.append(residual is not None)
            r = x.float() + (residual.float() if residual is not None else 0)
            y = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps)
            y = y * w.float()
            scale = (y.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-10)
            return (y / scale).to(qdtype), scale
        return rms_norm_dynamic_per_token_quant

    def _cell(self, sig):
        env = {"activation": "bf16", "weight": "bf16", "output": "fp8_e4m3",
               "activation_granularity": "per-token", "metadata": "fp32"}
        return Invocation(op=sig, signature=sig, phase="decode", tp=1,
                          dims={"B": 3, "s_query": 2, "d_model": 256},
                          dtype_env=env)

    @pytest.mark.parametrize("sig", ["fused_add_rmsnorm", "rmsnorm"])
    def test_the_launch_writes_the_cells_per_token_record(self, monkeypatch,
                                                          sig):
        pytest.importorskip("torch")
        from kernel_bench.refs import get_ref
        calls = []
        _fake_module(monkeypatch, "vllm")
        _fake_module(monkeypatch, "vllm._custom_ops",
                     rms_norm_dynamic_per_token_quant=self._kernel(calls))
        import sys
        sys.modules["vllm"]._custom_ops = sys.modules["vllm._custom_ops"]
        monkeypatch.setattr(
            "kernel_bench.providers.norm_act._require_cuda", lambda: None)
        inv = self._cell(sig)
        provider = next(p for p in _providers(sig)
                        if p.kernel_impl.endswith("per_token_quant"))
        assert provider.serves.accepts(inv)
        spec = get_ref(sig)
        inputs = build_inputs(spec, inv, "cpu")
        case = provider.build(inv, TORCH_ENV, inputs)
        got = case.to_canonical(case.run())
        want = spec.reference(inputs)
        assert calls == [sig == "fused_add_rmsnorm"]
        assert got.shape == want.shape
        # On the grid: within one FP8 step of the oracle's record.
        assert ((got - want).abs() / (want.abs() + 1e-3)).max() < 0.13
