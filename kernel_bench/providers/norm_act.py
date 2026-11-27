"""Norm / activation providers — vLLM custom C++ kernels.

Memory-bound per-token ops (rmsnorm, fused add+rmsnorm, SwiGLU gate). vLLM
ships these as `torch.ops._C` extension kernels (no Triton); other stacks
(FlashInfer, sglang, Triton) implement the same math differently, so these
register against library-agnostic signatures and compete under the envelope.

Import-guarded on vLLM; GPU bodies run only in a vLLM provider env.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..env import Env
from ..invocation import Invocation
from . import TORCH_FLOAT, BenchCase, KernelSignature, Launch, Provider, register
from .attention import _require_cuda, _torch_dtype
from ..refs import situ_betas
from ..signatures import get_signature


# Norm/act run in both prefill and decode; the token count (B·Sq) is already
# in the shape, so these declare no phase — the signature alone is the gate.



def _rmsnorm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    import torch
    from vllm import _custom_ops as ops
    dt = _torch_dtype(inv.dtype)
    x, w = inputs["x"].to(dt), inputs["w"].to(dt)
    out = torch.empty_like(x)
    return BenchCase(run=lambda: ops.rms_norm(out, x, w, inputs["eps"]),
                     to_canonical=lambda _: out)


def _fused_add_rmsnorm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from vllm import _custom_ops as ops
    dt = _torch_dtype(inv.dtype)
    # in-place: residual += x; x = rmsnorm(residual). Clone so a re-run in the
    # timing loop sees fresh inputs and the canonical output is stable.
    x = inputs["x"].to(dt).clone()
    residual = inputs["residual"].to(dt).clone()
    w = inputs["w"].to(dt)
    return BenchCase(
        run=lambda: ops.fused_add_rms_norm(x, residual, w, inputs["eps"]),
        to_canonical=lambda _: x,
    )


def _silu_and_mul_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    # Measure vLLM's BEST/canonical silu kernel for this op-signature — the C
    # kernel `torch.ops._C.silu_and_mul` that SiluAndMul.forward_cuda dispatches
    # to when custom ops are enabled. Call it directly: it's vLLM's OWN kernel
    # (not a foreign substitute), and the corpus wants the framework's best impl
    # for a compute-graph node — NOT a replica of each model's config-dependent
    # dispatch branch (eager-native vs cuda vs compiled — combinatorial, not the
    # point). Matches the graph's silu(gate)·up. x=[M,2I] (gate|up) → out=[M,I].
    import torch
    import vllm._custom_ops  # noqa: F401  — registers torch.ops._C
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    # Elementwise over the whole output, so nothing is left uninitialized —
    # unlike the grouped/masked GEMM paths, which write a subset and are zeroed.
    out = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=dt, device=x.device)
    op = torch.ops._C.silu_and_mul

    def _run():
        op(out, x)
        return out
    return BenchCase(run=_run, to_canonical=lambda o: o)


def _vllm_rmsnorm_quant_build(inv: Invocation, env: Env,
                              inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's norm epilogue: RMSNorm and the activation's FP8 record in one
    launch (`rms_norm_per_block_quant`), canonicalized through the cast both
    sides share."""
    _require_cuda()
    from vllm import _custom_ops as ops
    from ..refs import torch_dtype
    cast = inputs["cast"]
    dt = _torch_dtype(cast.wide)
    x = inputs["x"].to(dt)
    w = inputs["w"].to(dt)
    return BenchCase(
        run=Launch(ops.rms_norm_per_block_quant, x, w, inputs["eps"],
                   torch_dtype(cast.narrow), [1, cast.group]),
        to_canonical=cast.dequantise,
    )


def _vllm_fused_add_rmsnorm_quant_build(
        inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """The residual form of the same norm+quant launch: the residual operand
    is the kernel's own (`residual=`), not a second arithmetic invented here."""
    _require_cuda()
    from vllm import _custom_ops as ops
    from ..refs import torch_dtype
    cast = inputs["cast"]
    dt = _torch_dtype(cast.wide)
    x = inputs["x"].to(dt)
    residual = inputs["residual"].to(dt)
    w = inputs["w"].to(dt)
    return BenchCase(
        run=Launch(ops.rms_norm_per_block_quant, x, w, inputs["eps"],
                   torch_dtype(cast.narrow), [1, cast.group],
                   residual=residual),
        to_canonical=cast.dequantise,
    )


def _vllm_rmsnorm_per_token_quant_build(
        inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's norm epilogue for a per-token FP8 activation:
    `rms_norm_dynamic_per_token_quant` (`_custom_ops.py:423`, v0.32.0) — the
    launch vLLM's compile pass puts in place of RMSNorm (with its residual,
    when there is one) followed by a dynamic per-token FP8 quant
    (`compilation/passes/fusion/rms_quant_fusion.py:195-198`). One scale a
    row, fp32."""
    _require_cuda()
    from vllm import _custom_ops as ops
    from ..refs import torch_dtype
    cast = inputs["cast"]
    dt = _torch_dtype(cast.wide)
    x = inputs["x"].to(dt).flatten(end_dim=-2)
    w = inputs["w"].to(dt)
    extra = {}
    if "residual" in inputs:
        extra["residual"] = inputs["residual"].to(dt).flatten(end_dim=-2)
    return BenchCase(
        run=Launch(ops.rms_norm_dynamic_per_token_quant, x, w, inputs["eps"],
                   torch_dtype(cast.narrow), **extra),
        to_canonical=cast.dequantise,
    )


def _vllm_silu_and_mul_quant_build(
        inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's SwiGLU epilogue for a block-quantised down GEMM. The kernel's
    ABI is two-dimensional, so the canonical token rows are flattened once;
    `cast.dequantise` is the one reader of the record it writes."""
    _require_cuda()
    from vllm import _custom_ops as ops
    from ..refs import torch_dtype
    cast = inputs["cast"]
    x = inputs["x"].to(_torch_dtype(cast.wide)).flatten(end_dim=-2)
    return BenchCase(
        run=Launch(ops.silu_and_mul_per_block_quant, x, cast.group,
                   torch_dtype(cast.narrow)),
        to_canonical=cast.dequantise,
    )


def _vllm_clamped_act_build(op_name: str, extra):
    """vLLM's two other gated activations, beside `silu_and_mul`: the clamped
    SwiGLU and the SiTU gate. Same operand and same boundary — one packed
    [M, 2I] in, [M, I] out — so only the kernel and its scalar parameters
    differ. Those parameters (the clamp bound, the SiTU betas) are the
    kernel's own rather than cell axes, and they come from the cell, which
    carries what the MODEL publishes: the signature declares them
    (`OpSignature.config_params`), the plan reads them, and the provider
    declines a cell without them (`requires_params`) instead of choosing
    the bound it measures at."""
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        _require_cuda()
        import torch
        import vllm._custom_ops  # noqa: F401  — registers torch.ops._C
        dt = _torch_dtype(inv.dtype)
        x = inputs["x"].to(dt)
        out = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=dt, device=x.device)
        op = getattr(torch.ops._C, op_name)
        args = extra(inputs)

        def _run():
            op(out, x, *args)
            return out
        return BenchCase(run=_run, to_canonical=lambda o: o)
    return build


register(Provider(
    library="vllm", signature="silu_and_mul_with_clamp",
    kernel_impl="vllm_silu_and_mul_with_clamp", import_name="vllm",
    # bf16 measured (8618 rows). Writes `out` at the input dtype: the
    # fp8-output rows the open gate let in were not this launch.
    serves=KernelSignature(signature="silu_and_mul_with_clamp",
                           activation=("bf16",),
                           requires_params=("clamp_limit",)),
    # `(output, input, clamp_limit, alpha, beta)` — the SwiGLU-OAI form
    # vLLM calls for a packed w13 (fused_moe/activation.py:387), at the
    # alpha 1 / beta 0 it passes there.
    build=_vllm_clamped_act_build(
        "silu_and_mul_with_clamp",
        lambda i: (float(i["clamp_limit"]), 1.0, 0.0)),
    fused_op="silu_and_mul_with_clamp",
))


# sglang v0.5.22's clamped SwiGLU: `DeepseekV2MLP.forward`, when the model
# sets `swiglu_limit` and the fused FP8 down path does not take the launch
# (srt/models/deepseek_v2.py:390-422 is that path, which writes FP8 and its
# scales — the quantising form, a cell keyed at another `output`), allocates
# `x = gate_up.new_empty((M, N // 2))` and calls
# `silu_and_mul_clamp(gate_up, x, float(self.swiglu_limit))` (:481-484) on
# the contiguous `[M, 2I]` gate_up projection. On CUDA that is the JIT
# `SiluAndMulClampKernel<dtype, pdl>` (kernels/ops/moe/dsv4.py:186-197,
# module :99-107; kernels/jit/csrc/deepseek_v4/
# silu_and_mul_masked_post_quant.cuh:334-375, kernel :214-245): one launch,
# `min(gate, limit)` and `clamp(up, ±limit)` then `silu(gate) * up`, the
# clamps done in bf16 (:45-50) — an fp16 row is cast to bf16 first (:239),
# so an fp16 cell is not this kernel's function and is declined. Geometry
# (:346-360): input `[M, 2·H]` contiguous at the template's dtype, output
# `[M, H]` at the same dtype, `H % 8 == 0` (16-byte vectors of a 2-byte
# dtype). Both phases run it alike.
def _sgl_silu_and_mul_clamp_build(inv: Invocation, env: Env,
                                  inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    import torch
    from sglang.kernels.ops.moe.dsv4 import silu_and_mul_clamp
    dt = _torch_dtype(inv.dtype)
    gate_up = inputs["x"].to(dt).contiguous()
    m, n = gate_up.shape
    x = torch.empty(m, n // 2, dtype=dt, device=gate_up.device)
    return BenchCase(
        run=Launch(silu_and_mul_clamp, gate_up, x,
                   float(inputs["clamp_limit"])),
        to_canonical=lambda _o: x)


register(Provider(
    library="sglang", signature="silu_and_mul_with_clamp",
    kernel_impl="sglang_silu_and_mul_clamp",
    import_name="sglang.kernels.ops.moe.dsv4",
    serves=KernelSignature(signature="silu_and_mul_with_clamp",
                           activation=("bf16",), output=("bf16",),
                           divisible=(("d_ffn", 8),),
                           requires_params=("clamp_limit",)),
    build=_sgl_silu_and_mul_clamp_build,
    fused_op="silu_and_mul_with_clamp",
))


register(Provider(
    library="vllm", signature="rmsnorm", kernel_impl="vllm_rms_norm",
    import_name="vllm",
    # All three widths measured (lake); writes `out` at the input dtype.
    serves=KernelSignature(signature="rmsnorm", activation=TORCH_FLOAT),
    build=_rmsnorm_build,
))
register(Provider(
    library="vllm", signature="fused_add_rmsnorm",
    kernel_impl="vllm_fused_add_rms_norm", import_name="vllm",
    serves=KernelSignature(signature="fused_add_rmsnorm",
                           activation=TORCH_FLOAT),
    build=_fused_add_rmsnorm_build,
))
register(Provider(
    library="vllm", signature="silu_and_mul", kernel_impl="vllm_silu_and_mul",
    import_name="vllm",
    serves=KernelSignature(signature="silu_and_mul",
                           activation=("bf16", "fp16")),
    build=_silu_and_mul_build,
))

register(Provider(
    library="vllm", signature="rmsnorm",
    kernel_impl="vllm_rms_norm_per_block_quant", import_name="vllm",
    serves=KernelSignature(signature="rmsnorm",
                           activation=("bf16",),
                           weight=("bf16",),
                           output=("fp8_e4m3",),
                           activation_granularity=("per-group:64",
                                                   "per-group:128"),
                           dtype_slots=(("metadata", ("fp32",)),)),
    build=_vllm_rmsnorm_quant_build,
))
register(Provider(
    library="vllm", signature="fused_add_rmsnorm",
    kernel_impl="vllm_fused_add_rms_norm_per_block_quant",
    import_name="vllm",
    serves=KernelSignature(signature="fused_add_rmsnorm",
                           activation=("bf16",),
                           weight=("bf16",),
                           output=("fp8_e4m3",),
                           activation_granularity=("per-group:64",
                                                   "per-group:128"),
                           dtype_slots=(("metadata", ("fp32",)),)),
    build=_vllm_fused_add_rmsnorm_quant_build,
))
for _sig, _impl in (("rmsnorm", "vllm_rms_norm_per_token_quant"),
                    ("fused_add_rmsnorm",
                     "vllm_fused_add_rms_norm_per_token_quant")):
    register(Provider(
        library="vllm", signature=_sig, kernel_impl=_impl,
        import_name="vllm",
        serves=KernelSignature(signature=_sig,
                               activation=("bf16", "fp16"),
                               weight=("bf16", "fp16"),
                               output=("fp8_e4m3",),
                               activation_granularity=("per-token",),
                               dtype_slots=(("metadata", ("fp32",)),)),
        build=_vllm_rmsnorm_per_token_quant_build,
    ))
register(Provider(
    library="vllm", signature="silu_and_mul",
    kernel_impl="vllm_silu_and_mul_per_block_quant", import_name="vllm",
    serves=KernelSignature(signature="silu_and_mul",
                           activation=("bf16", "fp16"),
                           output=("fp8_e4m3",),
                           activation_granularity=("per-group:128",),
                           dtype_slots=(("metadata", ("fp32",)),),
                           divisible=(("d_ffn", 128),)),
    build=_vllm_silu_and_mul_quant_build,
))


def _gelu_and_mul_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    import torch
    import vllm._custom_ops  # noqa: F401
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    out = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=dt, device=x.device)
    op = torch.ops._C.gelu_and_mul

    def _run():
        op(out, x)
        return out
    return BenchCase(run=_run, to_canonical=lambda o: o)


register(Provider(
    library="vllm", signature="gelu_and_mul", kernel_impl="vllm_gelu_and_mul",
    import_name="vllm",
    serves=KernelSignature(signature="gelu_and_mul",
                           activation=("bf16", "fp32")),
    build=_gelu_and_mul_build,
))


def _situ_and_mul_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    import torch
    import vllm._custom_ops  # noqa: F401
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    out = torch.empty(*x.shape[:-1], x.shape[-1] // 2, dtype=dt, device=x.device)
    op = torch.ops._C.situ_and_mul
    # `(output, input, situ_beta, linear_beta)`, both from the cell. The
    # gate's beta is mandatory — vLLM refuses the launch without it ("SITU
    # requires activation_situ_beta from FusedMoEConfig") and so does this
    # provider, via `requires_params`. The linear beta is genuinely
    # optional and the kernel's own "unset" is a non-positive value, which
    # passes the up branch through (`SituAndMul.forward_cuda` writes
    # `-1.0` for None); a model that publishes one (Kimi-K3: 25.0) gets it.
    beta, linear = situ_betas(inputs)

    def _run():
        op(out, x, beta, linear)
        return out
    return BenchCase(run=_run, to_canonical=lambda o: o)


register(Provider(
    library="vllm", signature="situ_and_mul", kernel_impl="vllm_situ_and_mul",
    import_name="vllm",
    serves=KernelSignature(signature="situ_and_mul", activation=("bf16",),
                           requires_params=("situ_beta",)),
    build=_situ_and_mul_build,
    fused_op="situ_and_mul",
))


# sglang v0.5.22's SiTU gate is a real kernel, not a torch chain:
# `KimiK3MLP` builds `SituAndMul(beta=activation_situ_beta or 1.0,
# linear_beta=activation_situ_linear_beta)` (srt/models/kimi_k3.py:330-334)
# and calls it on the contiguous `[M, 2I]` gate_up projection (:339-340);
# `SituAndMul.forward_cuda` (srt/layers/activation.py:209-212) is
# `situ_and_mul(x, None, self.beta, self.linear_beta)`, whose wrapper
# (kernels/ops/activation/_jit_situ_and_mul.py:36-80) allocates the output at
# the input's dtype (:57-59), takes a contiguous 2-D row as is (:64-66), and
# launches the JIT `SituAndMulKernel<in, out, pdl>` (kernels/jit/csrc/kimi_k3/
# situ_and_mul.cuh:113-170, kernel :67-109, the element :33-43) with
# `has_linear_beta = linear_beta is not None` (:71) — None, not a sentinel,
# is its "unset". Both phases alike. Geometry (:133-150): the output width a
# whole number of `kMaxVecBytes / 2` elements, and the input row stride too;
# `kMaxVecBytes` is 16 below SM100 and 32 from it
# (kernels/jit/include/sgl_kernel/utils.cuh:149), so 8 is declared and the
# build declines the rest on SM100. bf16 in and out is what Kimi-K3 serves
# (an fp32 input would write bf16, :58). The dense MLP and the shared
# expert only: the routed experts' SiTU is the MoE runner's, never this
# kernel (the binding's `routed: false`, framework_bindings/sglang.yaml).
_SGL_SITU_VEC_SM100 = 16


def _sgl_situ_and_mul_build(inv: Invocation, env: Env,
                            inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    import torch
    dt = _torch_dtype(inv.dtype)
    gate_up = inputs["x"].to(dt).contiguous()
    if (torch.cuda.get_device_capability() >= (10, 0)
            and (gate_up.shape[-1] // 2) % _SGL_SITU_VEC_SM100):
        raise NotImplementedError(
            "on SM100 SituAndMulKernel moves 32-byte vectors: the output "
            "width must be a multiple of 16 (situ_and_mul.cuh:149)")
    from sglang.kernels.ops.activation import situ_and_mul
    beta, linear = situ_betas(inputs)
    return BenchCase(
        run=Launch(situ_and_mul, gate_up, None, beta,
                   linear if linear > 0 else None),
        to_canonical=lambda o: o)


register(Provider(
    library="sglang", signature="situ_and_mul",
    kernel_impl="sglang_situ_and_mul",
    import_name="sglang.kernels.ops.activation",
    serves=KernelSignature(signature="situ_and_mul",
                           activation=("bf16",), output=("bf16",),
                           divisible=(("d_ffn", 8),),
                           requires_params=("situ_beta",)),
    build=_sgl_situ_and_mul_build,
    fused_op="situ_and_mul",
))


# ---------------------------------------------------------------------------
# FlashInfer norm/activation — same math, different kernels; competes with vLLM
# under the same signatures. API from flashinfer.norm / flashinfer.activation
# (sgl_kernel mirrors it): rmsnorm returns out; fused_add_rmsnorm is in-place
# (residual += x; x = rmsnorm(residual)); silu_and_mul returns out.
# ---------------------------------------------------------------------------

def _fi_rmsnorm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from flashinfer.norm import rmsnorm
    dt = _torch_dtype(inv.dtype)
    x, w = inputs["x"].to(dt), inputs["w"].to(dt)
    return BenchCase(run=lambda: rmsnorm(x, w, inputs["eps"]),
                     to_canonical=lambda out: out)


def _fi_fused_add_rmsnorm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from flashinfer.norm import fused_add_rmsnorm
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt).clone()
    residual = inputs["residual"].to(dt).clone()
    w = inputs["w"].to(dt)
    return BenchCase(
        run=lambda: fused_add_rmsnorm(x, residual, w, inputs["eps"]),
        to_canonical=lambda _: x,          # in-place: x holds the normalized output
    )


def _fi_silu_and_mul_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from flashinfer.activation import silu_and_mul
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)                 # [M, 2I] concatenated gate|up
    return BenchCase(run=lambda: silu_and_mul(x),
                     to_canonical=lambda out: out)


register(Provider(
    library="flashinfer", signature="rmsnorm", kernel_impl="flashinfer_rmsnorm",
    import_name="flashinfer",
    # fp32 measured too (2526 fp32-out rows): FlashInfer's own entry point,
    # not the fp16/bf16-only csrc copy sgl_kernel compiles in.
    serves=KernelSignature(signature="rmsnorm", activation=TORCH_FLOAT),
    build=_fi_rmsnorm_build,
))
register(Provider(
    library="flashinfer", signature="fused_add_rmsnorm",
    kernel_impl="flashinfer_fused_add_rmsnorm",
    import_name="flashinfer",
    serves=KernelSignature(signature="fused_add_rmsnorm",
                           activation=TORCH_FLOAT),
    build=_fi_fused_add_rmsnorm_build,
))


def _sgl_rmsnorm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sgl_kernel import rmsnorm
    dt = _torch_dtype(inv.dtype)
    x, w = inputs["x"].to(dt), inputs["w"].to(dt)
    return BenchCase(run=lambda: rmsnorm(x, w, inputs["eps"]),
                     to_canonical=lambda out: out)


def _sgl_fused_add_rmsnorm_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sgl_kernel import fused_add_rmsnorm
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt).clone()
    residual = inputs["residual"].to(dt).clone()
    w = inputs["w"].to(dt)
    return BenchCase(
        run=lambda: fused_add_rmsnorm(x, residual, w, inputs["eps"]),
        to_canonical=lambda _: x,
    )


def _sgl_silu_and_mul_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sgl_kernel import silu_and_mul
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    return BenchCase(run=lambda: silu_and_mul(x), to_canonical=lambda out: out)


def _sgl_gelu_and_mul_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sgl_kernel import gelu_and_mul
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    return BenchCase(run=lambda: gelu_and_mul(x), to_canonical=lambda out: out)


register(Provider(
    library="sglang", signature="rmsnorm", kernel_impl="sgl_rmsnorm",
    import_name="sgl_kernel",
    # sgl_kernel's `rmsnorm` is FlashInfer's csrc/norm.cu compiled in
    # (sglang v0.5.22 kernels/aot/CMakeLists.txt:78 pins flashinfer
    # bc29697, :300 compiles its norm.cu), whose `rmsnorm` dispatches the
    # input through the fp16/bf16-only `*_DTYPE_TO_CTYPE_FP16` macro (the
    # same in flashinfer 0.7.2 csrc/norm.cu) — "rmsnorm(at::Tensor&, ...)
    # failed to dispatch data type Float" on Gemma-2's fp32
    # cells, ~200 failed builds per H100 pass. Its `fused_add_rmsnorm` is
    # sglang's own kernel (csrc/elementwise/fused_add_rms_norm_kernel.cu:43,
    # DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16 — float included), so that
    # one keeps fp32.
    serves=KernelSignature(signature="rmsnorm", activation=("bf16", "fp16")),
    build=_sgl_rmsnorm_build,
))
register(Provider(
    library="sglang", signature="fused_add_rmsnorm",
    kernel_impl="sgl_fused_add_rmsnorm", import_name="sgl_kernel",
    serves=KernelSignature(signature="fused_add_rmsnorm",
                           activation=TORCH_FLOAT),
    build=_sgl_fused_add_rmsnorm_build,
))
register(Provider(
    library="sglang", signature="silu_and_mul", kernel_impl="sgl_silu_and_mul",
    import_name="sgl_kernel",
    # The kernel moves 16-byte vectors (8 bf16) per row, so a d_ffn that is
    # not a multiple of 8 leaves the next row misaligned: `CUDA error:
    # misaligned address`, which poisons the context. On an H100 DeepSeek-V3
    # run all 443 measured rows had d_ffn % 8 == 0 and all 20 faults had
    # d_ffn 4 or 36 (the expert FFN cut at high TP).
    serves=KernelSignature(signature="silu_and_mul",
                           activation=("bf16", "fp16"),
                           divisible=(("d_ffn", 8),)),
    build=_sgl_silu_and_mul_build,
))
register(Provider(
    library="sglang", signature="gelu_and_mul", kernel_impl="sgl_gelu_and_mul",
    import_name="sgl_kernel",
    # The same 16-byte vector kernel as `sgl_silu_and_mul` above, and the
    # same `d_ffn % 8`. Its wrapper refuses a ROW that is not whole 16-byte
    # vectors ("The pointers must be multiple of 16 bytes.",
    # sgl_kernel/elementwise.py:289-290 at v0.5.22) — gemma-4-12B's
    # 15,360-wide FFN cut to d_ffn = 30, every cell of it — but checks
    # `2 * d_ffn`, not the `up` half's offset `d_ffn`, so d_ffn = 36 passes
    # it and then faults misaligned as silu's did. 8 is the bf16 vector;
    # fp32 would need only 4, and the stricter constant is the one that
    # cannot cost the model's run (see flashinfer_silu_and_mul).
    serves=KernelSignature(signature="gelu_and_mul",
                           activation=("bf16", "fp32"),
                           divisible=(("d_ffn", 8),)),
    build=_sgl_gelu_and_mul_build,
))


register(Provider(
    library="flashinfer", signature="silu_and_mul",
    kernel_impl="flashinfer_silu_and_mul",
    import_name="flashinfer",
    # `d_ffn % 8`, and it is a CRASH gate, not a coverage one. flashinfer's
    # activation kernel loads 16 bytes at a time
    # (`constexpr vec_size = 16 / sizeof(T)`, so 8 for bf16) and reads the `y`
    # half at `input + offset + d` — an offset that is only 16-byte aligned when
    # `d` is a multiple of vec_size. At d_ffn = 12 that load faults with
    # `CUDA error: misaligned address`, which is not a failure the driver can
    # contain: it poisons the context, so every remaining job in the model dies
    # with it. DeepSeek-V2 and V2-Lite each hit it 3 times and then exited 1.
    #
    # 8 is the bf16/fp16 requirement; fp32 would only need 4. The stricter
    # constant is used because this signature is measured in 16-bit throughout,
    # and a gate that is slightly narrow costs coverage while one that is
    # slightly wide costs the whole model's run.
    serves=KernelSignature(signature="silu_and_mul",
                           activation=("bf16", "fp16"),
                           divisible=(("d_ffn", 8),)),
    build=_fi_silu_and_mul_build,
))


def _torch_layernorm_build(inv, env, inputs) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    w = inputs["w"].to(dt)
    b = inputs["b"].to(dt)
    return BenchCase(
        run=lambda: torch.nn.functional.layer_norm(
            x, (x.shape[-1],), w, b, inputs["eps"]),
        to_canonical=lambda o: o,
    )


register(Provider(
    library="torch", signature="layernorm", kernel_impl="torch_layer_norm",
    import_name="torch",
    serves=KernelSignature(signature="layernorm", activation=TORCH_FLOAT),
    build=_torch_layernorm_build,
))


def _torch_fn(fn_of):
    """The function `fn_of(torch, F)` names, resolved when a case is built:
    registering a provider imports nothing (the bench service annotates rows
    with the catalog's names and never runs a kernel)."""
    import torch
    import torch.nn.functional as F
    return fn_of(torch, F)


def _unary_torch(fn_of):
    def build(inv, env, inputs) -> BenchCase:
        fn = _torch_fn(fn_of)
        x = inputs["x"].to(_torch_dtype(inv.dtype))
        return BenchCase(run=lambda: fn(x), to_canonical=lambda o: o)
    return build


def _unary_torch_with_params(fn_of):
    """The same eager floor for a unary whose function takes a scalar the
    cell carries beside its operand — `situ`'s beta, which the model
    publishes and the signature declares."""
    def build(inv, env, inputs) -> BenchCase:
        fn = _torch_fn(fn_of)
        x = inputs["x"].to(_torch_dtype(inv.dtype))
        return BenchCase(run=lambda: fn(x, inputs), to_canonical=lambda o: o)
    return build


def _register_unary(name, impl, fn_of, requires_params=()):
    """A torch unary; `fn_of(torch, F)` names its function (`_torch_fn`)."""
    register(Provider(
        library="torch", signature=name, kernel_impl=impl,
        import_name="torch",
        serves=KernelSignature(signature=name, activation=TORCH_FLOAT,
                               requires_params=requires_params),
        build=(_unary_torch_with_params(fn_of) if requires_params
               else _unary_torch(fn_of)),
    ))


def _torch_rmsnorm_build(inv, env, inputs) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    x, w = inputs["x"].to(dt), inputs["w"].to(dt)
    eps = inputs["eps"]
    return BenchCase(
        run=lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w,
        to_canonical=lambda o: o,
    )


register(Provider(
    library="torch", signature="rmsnorm", kernel_impl="torch_rmsnorm",
    import_name="torch",
    serves=KernelSignature(signature="rmsnorm", activation=TORCH_FLOAT),
    build=_torch_rmsnorm_build,
))


def _torch_fused_add_rmsnorm_build(inv, env, inputs) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt)
    residual = inputs["residual"].to(dt)
    w = inputs["w"].to(dt)
    eps = inputs["eps"]

    def run():
        r = residual + x
        return r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * w
    return BenchCase(run=run, to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="fused_add_rmsnorm",
    kernel_impl="torch_fused_add_rmsnorm",
    import_name="torch",
    serves=KernelSignature(signature="fused_add_rmsnorm",
                           activation=TORCH_FLOAT),
    build=_torch_fused_add_rmsnorm_build,
))


def _unswizzle_mxfp8_scales(packed: Any, tokens: int, width: int) -> Any:
    """Invert FlashInfer's F8_128x4 permutation for the cross-check.

    The launch's packed record is deliberately left in production layout; only
    the canonicalizer reads it back. This is the inverse of
    `mxfp8_utils.swizzle_mxfp8_scale`, and of the offset formula the fused
    kernel itself uses (query_quant.py:25-33).
    """
    import torch
    groups = width // 32
    padded_tokens = -(-tokens // 128) * 128
    padded_groups = -(-groups // 4) * 4
    rows = padded_tokens // 128
    tiles = padded_groups // 4
    sf = packed.view(torch.uint8).view(rows, tiles, 32, 4, 4)
    sf = sf.transpose(1, 3).reshape(padded_tokens, padded_groups)
    return sf[:tokens, :groups]


def _vllm_q_kv_rmsnorm_quant_build(inv: Invocation, env: Env,
                                   inputs: Dict[str, Any]) -> BenchCase:
    """The production split-view launch: `_split_qkv_and_norm` makes q and kv
    as strided rows of one fused projection (deepseek_v41/attention.py:781),
    exactly the operands built here. The norm gains are contiguous module
    parameters in production, as the wrapper requires (:100)."""
    _require_cuda()
    import torch
    from vllm.models.deepseek_v41.common.ops.query_quant import (
        fused_q_kv_rmsnorm_quant)
    dt = _torch_dtype(inv.dtype)
    q, kv = inputs["q"].to(dt), inputs["kv"].to(dt)
    packed = torch.empty(q.shape[0], q.shape[-1] + kv.shape[-1],
                         dtype=dt, device=q.device)
    packed[:, :q.shape[-1]] = q
    packed[:, q.shape[-1]:] = kv
    qr, kv_view = packed.split([q.shape[-1], kv.shape[-1]], dim=-1)
    qw = inputs["q_weight"].to(dt).contiguous()
    kvw = inputs["kv_weight"].to(dt).contiguous()
    tokens, width = int(qr.shape[0]), int(qr.shape[-1])

    def canonical(out):
        quantized, kv_out = out
        sf = _unswizzle_mxfp8_scales(quantized.scale, tokens, width)
        scale = torch.exp2(sf.to(torch.float32) - 127.0)
        q_out = quantized.data.view(torch.float8_e4m3fn).float()
        q_out = q_out.view(tokens, width // 32, 32) * scale.unsqueeze(-1)
        return torch.cat([q_out.reshape(tokens, width), kv_out.float()], dim=1)

    return BenchCase(
        run=lambda: fused_q_kv_rmsnorm_quant(
            qr, kv_view, qw, kvw, inputs["eps"]),
        to_canonical=canonical,
    )


def _register_q_kv_rmsnorm_quant(library: str, impl: str, import_name: str,
                                 build: Any, min_sm: Optional[int],
                                 gains: Optional[tuple] = None) -> None:
    # Split by activation so each registration can state the KV write's
    # dtype: the launch returns the normalized KV rows BESIDE the MXFP8 q
    # record (`kvo`, query_quant.py:105, :132), and the capability record
    # must not under-describe that second write.
    for _dt in ("bf16", "fp16"):
        register(Provider(
            library=library, signature="fused_q_kv_rmsnorm_quant",
            kernel_impl=f"{impl}_{_dt}", import_name=import_name,
            serves=KernelSignature(
                signature="fused_q_kv_rmsnorm_quant",
                activation=(_dt,), output=("mxfp8",),
                weight=gains or (_dt,),
                activation_granularity=("per-group:32",),
                divisible=(("d_q_lora", 32),),
                secondary_outputs=(("kv", _dt),)),
            min_sm=min_sm, build=build,
            fused_op="fused_q_kv_rmsnorm_quant",
        ))


_register_q_kv_rmsnorm_quant("vllm", "vllm_fused_q_kv_rmsnorm_quant",
                             "vllm", _vllm_q_kv_rmsnorm_quant_build, 100)
# ^ gains at the row dtype: the build casts them to it, as the module
# parameters it reads are held.


def _torch_q_kv_rmsnorm_quant_build(inv: Invocation, env: Env,
                                    inputs: Dict[str, Any]) -> BenchCase:
    from ..refs import get_ref
    reference = get_ref(inv.signature).reference
    return BenchCase(run=lambda: reference(inputs), to_canonical=lambda out: out)


_register_q_kv_rmsnorm_quant("torch", "torch_fused_q_kv_rmsnorm_quant",
                             "torch", _torch_q_kv_rmsnorm_quant_build, None,
                             gains=TORCH_FLOAT)  # the reference widens any gain


register(Provider(
    library="torch", signature="allreduce_fused_rmsnorm",
    kernel_impl="torch_allreduce_fused_rmsnorm",
    import_name="torch",
    serves=KernelSignature(signature="allreduce_fused_rmsnorm",
                           activation=TORCH_FLOAT),
    build=_torch_fused_add_rmsnorm_build,
    # DISABLED, not deleted: its cells stay a visible gap rather than a
    # measurement nobody can check. On one process the all-reduce is the
    # identity, so this timed residual+RMSNorm and nothing else — and the
    # reference omitted the collective too, so the cross-check agreed. The
    # rows then said a `n_comm_ranks=8` all-reduce costs what a local norm
    # costs; with the group now in the shape they would still be the fastest
    # candidate of their own cell.
    #
    # Multi-process groups arrived and this stayed disabled, because a process
    # per rank is only half of what it needs. The other half is a framework
    # that fuses the two, and vLLM-Ascend does not: `npu_mm_all_reduce`,
    # `matmul_all_reduce` and `all_reduce_fusion` are absent from the
    # v0.30.0-a3 plugin, and GLM-5.3-Flash reaches the collective through
    # plain `tensor_model_parallel_all_gather`
    # (`vllm_ascend/models/glm5next/model.py:261`). The one kernel that
    # looks like this — `npu_add_rms_norm_quant` — adds, norms and
    # quantises without communicating at all.
    disabled_reason=(
        "vLLM-Ascend has no fused allreduce+rmsnorm launch (no "
        "npu_mm_all_reduce / matmul_all_reduce / all_reduce_fusion in "
        "v0.30.0-a3), so there is nothing on an NPU for this cell to "
        "measure; on one process the all-reduce is the identity and the "
        "row would say an 8-rank collective costs what a local norm does"),
))


_register_unary("silu", "torch_silu", lambda torch, F: F.silu)
_register_unary("gelu", "torch_gelu", lambda torch, F: F.gelu)
_register_unary("gelu_tanh", "torch_gelu_tanh",
                lambda torch, F: lambda x: F.gelu(x, approximate="tanh"))
_register_unary("sigmoid", "torch_sigmoid", lambda torch, F: torch.sigmoid)
_register_unary("softplus", "torch_softplus", lambda torch, F: F.softplus)
_register_unary("sqrt", "torch_sqrt", lambda torch, F: torch.sqrt)
_register_unary("neg_exp", "torch_neg_exp", lambda torch, F: lambda x: torch.exp(-x))
_register_unary("relu_squared", "torch_relu_squared",
                lambda torch, F: lambda x: torch.relu(x) ** 2)
# Two activations a plan used to drop for want of a kernel. Each is EAGER
# TORCH IN AN AUDITED SERVING PATH, so the floor is the launch itself and not
# a stand-in — which is the whole bar for declaring one here (the signatures'
# note in `op_signatures_catalog` cites both paths).
#
# vLLM 0.32.0's soft-cap, verbatim: `logits = logits / self.soft_cap`,
# `logits = torch.tanh(logits)`, `logits = logits * self.soft_cap`
# (model_executor/layers/logits_processor.py:114-116).
_register_unary("tanh", "torch_tanh", lambda torch, F: torch.tanh)
# sglang 0.5.22's PLE gate, verbatim: `gate.abs().clamp_min(1e-6).sqrt() *
# gate.sign()` (srt/models/qwen4_exp.py:1377). Written as the same five
# elementwise launches the op's cost coefficient prices — `copysign` would
# fold `sign` and the multiply into one, timing three launches and one fewer
# intermediate than the framework runs (and differing at signed zero).
_register_unary("signed_sqrt", "torch_signed_sqrt",
                lambda torch, F: lambda x: (
                    x.abs().clamp_min(1e-6).sqrt() * x.sign()))
# Kimi-K3's gate at the beta the model publishes. `tanh(x)·sigmoid(x)`
# is this function at beta 1 only, and no model publishes 1.
_register_unary("situ", "torch_situ",
                lambda torch, F: lambda x, i: (
                    lambda b: b * torch.tanh(x / b) * torch.sigmoid(x))(float(i["situ_beta"])),
                requires_params=("situ_beta",))
_register_unary("scale_const", "torch_scale_const", lambda torch, F: lambda x: x * 2)
_register_unary("rmsnorm_unscaled", "torch_rmsnorm_unscaled",
                lambda torch, F: lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True)
                                                           + 1e-6))
_register_unary("sum_reduce", "torch_sum_reduce", lambda torch, F: lambda x: x.sum(-2))
_register_unary("normalize_sum_one", "torch_normalize_sum_one",
                lambda torch, F: lambda x: x / (x.sum(-1, keepdim=True) + 1e-6))


def _torch_gated_act_build(fn):
    """The eager floor for a gated activation: the packed [M, 2I] split, the
    gate's own function, the product."""
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        dt = _torch_dtype(inv.dtype)
        x = inputs["x"].to(dt)
        half = x.shape[-1] // 2
        return BenchCase(
            run=lambda: fn(x[:, :half], x[:, half:], inputs),
            to_canonical=lambda o: o)
    return build


def _torch_clamped_swiglu(gate, up, inputs):
    import torch
    limit = float(inputs["clamp_limit"])
    g = torch.clamp(gate, max=limit)
    return g * torch.sigmoid(g) * torch.clamp(up, min=-limit, max=limit)



register(Provider(
    library="torch", signature="silu_and_mul_with_clamp",
    kernel_impl="torch_silu_and_mul_with_clamp", import_name="torch",
    serves=KernelSignature(signature="silu_and_mul_with_clamp",
                           activation=TORCH_FLOAT,
                           requires_params=("clamp_limit",)),
    build=_torch_gated_act_build(_torch_clamped_swiglu),
    fused_op="silu_and_mul_with_clamp",
))


# The same gated activations with the cast to the consuming GEMM's activation
# format folded in (the fusion plans take `quantise_activation` as an
# optional member, and the output's precision follows the GEMM that reads
# it): the launch writes the quantised record — elements and scales — at the
# cell's granularity. The floor is the two steps eager: the activation of
# the cell's rows evaluated in fp32, as a fused epilogue holds it in
# registers before it rounds once onto the grid (rounding it to the
# activation dtype first puts a second rounding in front of the grid's and
# flips grid steps the oracle does not), then the one grid implementation
# (`quantise.torch_quantise_activation`). A separate registration from the
# plain floor because the plain one writes the activation's own dtype.
#: Bytes of fp32 working rows (the two widened halves, the activation, the
#: grid's quotient) one block of a quantised gated floor may hold before the
#: rows are walked in blocks. A cell that fits keeps the one-shot launch.
_GATED_QUANT_BLOCK_BYTES = 2 << 30


def _torch_gated_act_quant_build(fn):
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        import torch
        from ..quant import ACTIVATION_EPS
        from .quantise import torch_quantise_activation
        cast = inputs["cast"]
        dt = _torch_dtype(inv.dtype)
        x = inputs["x"].to(dt)
        rows, half = x.shape[0], x.shape[-1] // 2
        wide = cast._torch_wide()

        def _quantised(block):
            y = fn(block[:, :half].float(), block[:, half:].float(), inputs)
            return torch_quantise_activation(y, cast.narrow, cast.group,
                                             cast.scale_encoding, wide,
                                             ACTIVATION_EPS)

        blk = max(1, _GATED_QUANT_BLOCK_BYTES // (4 * 4 * half))
        if blk >= rows:
            return BenchCase(run=lambda: _quantised(x),
                             to_canonical=cast.dequantise)
        # The record each block writes into, built here so the timed call
        # allocates nothing beyond one block's working rows.
        q0, s0 = _quantised(x[:1])
        q = torch.empty(rows, *q0.shape[1:], dtype=q0.dtype, device=x.device)
        sc = torch.empty(rows, *s0.shape[1:], dtype=s0.dtype, device=x.device)

        def _run():
            for lo in range(0, rows, blk):
                hi = min(rows, lo + blk)
                q[lo:hi], sc[lo:hi] = _quantised(x[lo:hi])
            return q, sc
        return BenchCase(run=_run, to_canonical=cast.dequantise)
    return build


def _torch_swiglu(gate, up, inputs):
    import torch.nn.functional as F
    return F.silu(gate) * up


def _torch_geglu(gate, up, inputs):
    import torch.nn.functional as F
    return F.gelu(gate) * up


def _register_gated_act_quant_floors():
    from .quantise import (_EVERY_GRANULARITY, _NARROW, _NARROW_SCALES,
                           declines_an_untileable_row)
    # `fused_op` as each plain floor beside it states it.
    for sig, fn, params, fused in (
            ("silu_and_mul", _torch_swiglu, (), None),
            ("gelu_and_mul", _torch_geglu, (), None),
            ("silu_and_mul_with_clamp", _torch_clamped_swiglu,
             ("clamp_limit",), "silu_and_mul_with_clamp")):
        register(Provider(
            library="torch", signature=sig, kernel_impl=f"torch_{sig}_quant",
            import_name="torch",
            serves=KernelSignature(signature=sig, activation=TORCH_FLOAT,
                                   output=_NARROW,
                                   activation_granularity=_EVERY_GRANULARITY,
                                   dtype_slots=(("metadata", _NARROW_SCALES),),
                                   requires_params=params),
            # The written row is the gate's `d_ffn` half.
            impl_params=declines_an_untileable_row("d_ffn"),
            build=_torch_gated_act_quant_build(fn),
            fused_op=fused,
        ))


_register_gated_act_quant_floors()


def _torch_rope_emb_build(conjugate: bool):
    """The eager rotation of ONE activation at each row's position — MLA's
    and DSA's rope slice, and the inverse rotation DeepSeek-V4 applies to the
    attention output.

    `rope` rotates q and k together and every library ships a kernel for it;
    the single-tensor form has no such entry point (vLLM's
    `rotary_embedding` and sglang's `apply_rope_with_cos_sin_cache_inplace`
    both require a key), so torch is the only provider — and it has to do the
    rotation. It used to be `lambda x: x`, which timed nothing: the
    reference was an identity rotation too, so the cross-check agreed.
    """
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        import torch
        from ..refs import _rope_rotate
        dt = _torch_dtype(inv.dtype)
        x = inputs["x"].to(dt)
        pos = inputs["positions"]
        cache = inputs["cos_sin_cache"]                   # fp32, as the kernels take it
        if conjugate:
            half = cache.shape[-1] // 2
            cache = torch.cat([cache[:, :half], -cache[:, half:]], dim=-1)
        rd = x.shape[-1]
        # GPT-J interleaving: DeepSeek's `apply_rotary_emb` pairs adjacent
        # channels of the rope slice.
        return BenchCase(
            run=lambda: _rope_rotate(x[:, None, :], pos, cache, rd,
                                     is_neox=False).squeeze(1),
            to_canonical=lambda o: o,
        )
    return build


for _name, _impl, _conj in (("rope_emb", "torch_rope_emb", False),
                            ("rope_emb_inverse", "torch_rope_emb_inverse", True)):
    register(Provider(
        library="torch", signature=_name, kernel_impl=_impl,
        import_name="torch",
        serves=KernelSignature(signature=_name, activation=TORCH_FLOAT),
        build=_torch_rope_emb_build(_conj),
    ))


def _torch_binary(fn):
    def build(inv, env, inputs) -> BenchCase:
        dt = _torch_dtype(inv.dtype)
        a, b = inputs["a"].to(dt), inputs["b"].to(dt)
        return BenchCase(run=lambda: fn(a, b), to_canonical=lambda o: o)
    return build


register(Provider(
    library="torch", signature="mul", kernel_impl="torch_mul",
    import_name="torch",
    serves=KernelSignature(signature="mul", activation=TORCH_FLOAT),
    build=_torch_binary(lambda a, b: a * b),
))


register(Provider(
    library="torch", signature="add", kernel_impl="torch_add",
    import_name="torch",
    serves=KernelSignature(signature="add", activation=TORCH_FLOAT),
    build=_torch_binary(lambda a, b: a + b),
))


def _torch_hc_split_build(inv, env, inputs) -> BenchCase:
    """The frameworks' split of the mix vector, eager.

    This floor used to run `(mixes + base)` through eight one-sided row
    normalisations and return the `pre` slice — the same wrong function the
    reference held, so the numerical-agreement check compared the floor
    against itself and passed without ever testing whether either was the
    kernel's. Both now call the one place that arithmetic lives
    (`refs.hc_split_coefficients`): sigmoid gates and a softmax plus the
    alternating Sinkhorn rounds.

    Computed in fp32 because that is the width the frameworks hold this
    vector at — `fn` is an fp32 parameter in both and both assert it — and
    the operand is one 24-wide row per token either way, so there is no
    large tensor whose precision this chooses.
    """
    import torch
    from ..refs import (_HC_EPS, _HC_POST_MULT, _HC_SINKHORN_ITERS,
                        hc_split_coefficients)
    n = int(inv.dims["n_hidden_copies"])
    mixes = inputs["mixes"].float()
    mixes = mixes.reshape(-1, mixes.shape[-1])
    base, scale = inputs["base"].float(), inputs["scale"]

    def run():
        return hc_split_coefficients(
            mixes, scale, base, n, pre_eps=_HC_EPS,
            post_mult=_HC_POST_MULT, sinkhorn_eps=_HC_EPS,
            sinkhorn_iters=_HC_SINKHORN_ITERS)

    return BenchCase(
        run=run,
        to_canonical=lambda o: torch.cat(
            [o[0], o[1], o[2].flatten(1)], dim=-1))


register(Provider(
    library="torch", signature="hc_split_sinkhorn",
    kernel_impl="torch_hc_split_sinkhorn",
    import_name="torch",
    # Reads every operand `.float()`, so the cell's width never changes
    # the launch.
    serves=KernelSignature(signature="hc_split_sinkhorn",
                           activation=TORCH_FLOAT),
    build=_torch_hc_split_build,
))


def _torch_hc_mix_build(inv, env, inputs) -> BenchCase:
    dt = _torch_dtype(inv.dtype)
    comb = inputs["comb"].to(dt)
    residual = inputs["residual"].to(dt)
    return BenchCase(run=lambda: comb @ residual, to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="hc_mix", kernel_impl="torch_hc_mix",
    import_name="torch",
    serves=KernelSignature(signature="hc_mix", activation=TORCH_FLOAT),
    build=_torch_hc_mix_build,
))


def _torch_hash_route_build(inv, env, inputs) -> BenchCase:
    table = inputs["tid2eid"]
    ids = inputs["ids"].long()
    return BenchCase(run=lambda: table[ids], to_canonical=lambda o: o.float())


register(Provider(
    library="torch", signature="hash_route", kernel_impl="torch_hash_route",
    import_name="torch",
    # The table is the cell's fp32 metadata (157 rows), indexed, not cast.
    serves=KernelSignature(signature="hash_route",
                           dtype_slots=(("metadata", ("fp32",)),)),
    build=_torch_hash_route_build,
))


def _torch_step3p5_build(inv, env, inputs) -> BenchCase:
    dt = _torch_dtype(inv.dtype)
    attn = inputs["attn"].to(dt)
    gate = inputs["gate"].to(dt)
    return BenchCase(
        run=lambda: attn * gate.transpose(1, 2).unsqueeze(-1),
        to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="step3p5_headwise_gate",
    kernel_impl="torch_step3p5_headwise_gate",
    import_name="torch",
    serves=KernelSignature(signature="step3p5_headwise_gate", activation=TORCH_FLOAT),
    build=_torch_step3p5_build,
))


def _torch_position_table_build(inv, env, inputs) -> BenchCase:
    dt = _torch_dtype(inv.dtype)
    x, table = inputs["x"].to(dt), inputs["table"].to(dt)
    return BenchCase(run=lambda: x + table, to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="position_table_add",
    kernel_impl="torch_position_table_add",
    import_name="torch",
    serves=KernelSignature(signature="position_table_add", activation=TORCH_FLOAT),
    build=_torch_position_table_build,
))


def _torch_attn_res_build(inv, env, inputs) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    prefix = inputs["prefix"].to(dt)
    stack = inputs["stack"].to(dt)
    # Concat is setup: stack and prefix are fixed inputs, so the [stack‖prefix]
    # tensor is built once here, outside the timed call.
    cat = torch.cat([stack, prefix.unsqueeze(2)], dim=2)

    def run():
        x = cat * torch.rsqrt(cat.pow(2).mean(-1, keepdim=True) + 1e-6)
        scores = (x[:, :, -1:, :] * x).sum(-1)
        w = torch.softmax(scores, dim=-1)
        return (w.unsqueeze(-1) * cat).sum(2)
    return BenchCase(run=run, to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="attn_res_combine",
    kernel_impl="torch_attn_res_combine",
    import_name="torch",
    serves=KernelSignature(signature="attn_res_combine", activation=TORCH_FLOAT),
    build=_torch_attn_res_build,
))


# ---------------------------------------------------------------------------
# rmsnorm_gated — the linear-attention output norm gated by act(z), one launch
# (the fused span `rmsnorm_gated`). Each framework ships two Triton kernels
# for it, on the same operands, and which one runs is the model's class:
#
#   FLA `layer_norm_fwd` with `z`, norm before gate
#     vLLM   RMSNormGated (model_executor/layers/layernorm.py:289) →
#            third_party/flash_linear_attention/ops/layernorm_guard.py
#            `rmsnorm_fn` (:562); built by qwen_gdn_linear_attn.py:572 with
#            group_size=None, norm_before_gate=True; Ling's
#            BailingGroupRMSNormGate (mamba/linear/bailing_linear_attn.py:343)
#
# Ling's norm is a GROUP norm in both frameworks (one RMS per group of the
# row: bailing_linear_attn.py:455-462, `group_norm_size` 8 for Ling-2.6, so
# 1024-wide groups), while the graph's `g_norm` spans the whole row; the cell
# follows the graph (one group per row): the same bytes in the same launch,
# a different reduction width.
#     sglang kernels/ops/attention/fla/layernorm_gated.py `RMSNorm.forward` →
#            `layernorm_fn(..., is_rms_norm=True)` (:529); qwen3_5.py:503,
#            bailing_moe_linear.py:504 (`BailingGroupRMSNormGate`, :370)
#   KDA-family `rms_norm_gated`
#     vLLM   FusedRMSNormGated → third_party/flash_linear_attention/ops/kda.py
#            `rms_norm_gated` (:407); kimi_k3/nvidia/kda.py:788,
#            glm5next/common/kda.py:313 (activation="sigmoid")
#     sglang kernels/ops/attention/fla/fused_norm_gate.py `rms_norm_gated`
#            (:383); kimi_k3.py:1752, glm5_next.py:568
#
# (vLLM v0.32.0, sglang v0.5.22.) Both run in prefill and decode on the
# per-token rows, so no phase is declared. The gate activation is the one
# stated cell value (`refs.RMSNORM_GATED_ACTIVATION`).
# ---------------------------------------------------------------------------

def _gated_norm_operands(inv: Invocation, inputs: Dict[str, Any]):
    """x and z at the activation dtype; the gain as the cell stores it (fp32
    for Qwen3.5's, bf16 for GLM's) — the kernels load it and widen."""
    dt = _torch_dtype(inv.dtype)
    return (inputs["x"].to(dt), inputs["z"].to(dt), inputs["w"],
            float(inputs["eps"]), inputs["activation"])


def _vllm_rmsnorm_gated_build(inv: Invocation, env: Env,
                              inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        rmsnorm_fn,
    )
    x, z, w, eps, act = _gated_norm_operands(inv, inputs)
    return BenchCase(
        run=lambda: rmsnorm_fn(x, w, None, z=z, eps=eps, group_size=None,
                               norm_before_gate=True, activation=act),
        to_canonical=lambda out: out.float())


def _vllm_fused_rmsnorm_gated_build(inv: Invocation, env: Env,
                                    inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated
    x, z, w, eps, act = _gated_norm_operands(inv, inputs)
    # In place: `layer_norm_gated_fwd` writes y into x (`y = x`, kda.py:342),
    # so the kernel gets its own copy and the shared inputs stay as built.
    x = x.clone()
    return BenchCase(
        run=lambda: rms_norm_gated(x, z, w, None, activation=act, eps=eps),
        to_canonical=lambda out: out.float())


def _sgl_rmsnorm_gated_build(inv: Invocation, env: Env,
                             inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn
    x, z, w, eps, act = _gated_norm_operands(inv, inputs)
    return BenchCase(
        run=lambda: layernorm_fn(x, w, None, z=z, eps=eps, group_size=None,
                                 norm_before_gate=True, is_rms_norm=True,
                                 activation=act),
        to_canonical=lambda out: out.float())


def _sgl_fused_rmsnorm_gated_build(inv: Invocation, env: Env,
                                   inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
    x, z, w, eps, act = _gated_norm_operands(inv, inputs)
    # In place, like vLLM's copy of this kernel (fused_norm_gate.py:223,
    # `y = x`): the kernel gets its own copy.
    x = x.clone()
    return BenchCase(
        run=lambda: rms_norm_gated(x, z, w, None, act, eps=eps),
        to_canonical=lambda out: out.float())


def _torch_rmsnorm_gated_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """The three ops the span replaces, eager — the floor that bounds the
    fused kernels."""
    import torch
    import torch.nn.functional as F
    x, z, w, eps, act = _gated_norm_operands(inv, inputs)
    gate = torch.sigmoid if act == "sigmoid" else F.silu

    def run():
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        return y * gate(z)
    return BenchCase(run=run, to_canonical=lambda o: o.float())


# The four Triton kernels are measured at bf16 only (~4k rows each).
for _lib, _impl, _imp, _build, _act in (
    ("vllm", "vllm_rmsnorm_gated", "vllm", _vllm_rmsnorm_gated_build,
     ("bf16",)),
    ("vllm", "vllm_fused_rmsnorm_gated", "vllm",
     _vllm_fused_rmsnorm_gated_build, ("bf16",)),
    ("sglang", "sgl_rmsnorm_gated", "sglang", _sgl_rmsnorm_gated_build,
     ("bf16",)),
    ("sglang", "sgl_fused_rmsnorm_gated", "sglang",
     _sgl_fused_rmsnorm_gated_build, ("bf16",)),
    ("torch", "torch_rmsnorm_gated", "torch", _torch_rmsnorm_gated_build,
     TORCH_FLOAT),
):
    register(Provider(
        library=_lib, signature="rmsnorm_gated", kernel_impl=_impl,
        import_name=_imp,
        serves=KernelSignature(signature="rmsnorm_gated", activation=_act),
        build=_build, fused_op="rmsnorm_gated",
    ))


# ---------------------------------------------------------------------------
# Two retrieval gates whose normalized dot selects how much of a shared
# value enters each hyper-connection copy. Both production launches compute
# in fp32 and cast only at their outputs.
# ---------------------------------------------------------------------------
def _dtype_floor(chain):
    """A local eager floor at the cell's activation dtype, for spans whose
    output is one tensor (the shared helper below is defined after older
    registrations and owns multi-port canonicalization)."""
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        import torch
        dt = _torch_dtype(inv.dtype)
        cast = {k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point()
                    else v) for k, v in inputs.items()}
        return BenchCase(run=lambda: chain(cast, torch),
                         to_canonical=lambda o: o.float())
    return build


def _engram_floor_chain(i, torch):
    """DeepSeek-V4.1's engram gate as its own source writes it.

    Its signed square root is `copysign` (`source/model.py`, quoted in
    `components/library/dsv41_engram.yaml:130-131`) and the PLE span's is
    `* sign()`. THESE ARE TWO SPANS' TWO LOWERINGS, not two spellings of one
    formula — each floor runs the chain its own model runs, which is the
    whole point of a floor. The timing difference between them (three
    elementwise launches against five) therefore belongs to the two spans
    and not to this floor: it would only be a defect if one span's floor
    borrowed the other's chain, which is what `copysign` in `_ple_floor_chain`
    used to do.
    """
    hc, d = i["q_weight"].shape
    x = i["x"].unflatten(-1, (hc, d))
    key, value = i["kv"].split([hc * d, d], dim=-1)
    key = key.unflatten(-1, (hc, d))
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + i["eps"])
    k = key * torch.rsqrt(key.pow(2).mean(-1, keepdim=True) + i["eps"])
    dot = ((h * i["q_weight"]) * (k * i["k_weight"])).sum(-1) * d ** -0.5
    gate = torch.sigmoid(torch.copysign(
        dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (x + gate.unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)


def _engram_canonical(out, inputs):
    """A launch's `[tokens, hc, d]` rows back at the cell's `x` shape
    (`[B, s_query, hc*d]`), which is the oracle's. Flattening both ends
    instead left one 1-D vector, and every cross-check failed on shape."""
    return out.reshape(inputs["x"].shape)


def _vllm_engram_build(inv: Invocation, env: Env,
                       inputs: Dict[str, Any]) -> BenchCase:
    """vLLM's own Triton launch (`_fused_engram_post_wkv_kernel`, grid one
    program per token/copy), called with the producer's contiguous rows and
    with token masking compiled out as `Engram.forward` does when no serving
    mask exists."""
    _require_cuda()
    import torch
    import triton
    from vllm.models.deepseek_v41.common.engram import (
        _fused_engram_post_wkv_kernel)

    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt).flatten(0, 1).unflatten(-1, (
        inputs["q_weight"].shape[0], inputs["q_weight"].shape[1]))
    kv = inputs["kv"].to(dt).flatten(0, 1).contiguous()
    qw = inputs["q_weight"].to(dt).contiguous()
    kw = inputs["k_weight"].to(dt).contiguous()
    out = torch.empty_like(x)
    tokens, hc, d = x.shape

    def run():
        _fused_engram_post_wkv_kernel[(tokens * hc,)](
            x, kv, qw, kw, x, out, tokens,
            *x.stride(), kv.stride(0), kv.stride(1),
            *qw.stride(), *kw.stride(), 0, *out.stride(),
            float(inputs["eps"]), 1e-6,
            DIM=d, HC_MULT=hc, BLOCK_SIZE=triton.next_power_of_2(d),
            HAS_MASK=False, num_warps=8 if triton.next_power_of_2(d) >= 2048
            else 4,
        )
        return out
    return BenchCase(run=run,
                     to_canonical=lambda o: _engram_canonical(o, inputs))


def _sgl_engram_build(inv: Invocation, env: Env,
                      inputs: Dict[str, Any]) -> BenchCase:
    """sglang `fused_engram_gate`, whose wrapper requires exactly the
    contiguous x/kv/weight layout production has after `wkv`."""
    _require_cuda()
    from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt).flatten(0, 1).unflatten(-1, inputs["q_weight"].shape)
    kv = inputs["kv"].to(dt).flatten(0, 1).contiguous()
    qw = inputs["q_weight"].to(dt).contiguous()
    kw = inputs["k_weight"].to(dt).contiguous()
    return BenchCase(
        run=lambda: fused_engram_gate(x, kv, qw, kw, float(inputs["eps"]),
                                      1e-6),
        to_canonical=lambda o: _engram_canonical(o, inputs))


register(Provider(
    library="vllm", signature="fused_engram_gate",
    kernel_impl="vllm_fused_engram_post_wkv",
    import_name="vllm.models.deepseek_v41.common.engram",
    serves=KernelSignature(signature="fused_engram_gate",
                           activation=("bf16",), weight=("bf16",)),
    build=_vllm_engram_build, fused_op="fused_engram_gate",
))
register(Provider(
    library="sglang", signature="fused_engram_gate",
    kernel_impl="sglang_fused_engram_gate",
    import_name="sglang.kernels.ops.embeddings.engram_gate",
    serves=KernelSignature(signature="fused_engram_gate",
                           activation=("bf16", "fp32"),
                           weight=("bf16", "fp32")),
    build=_sgl_engram_build, fused_op="fused_engram_gate",
))
register(Provider(
    library="torch", signature="fused_engram_gate",
    kernel_impl="torch_fused_engram_gate", import_name="torch",
    # The floor casts the gains to the cell's dtype too.
    serves=KernelSignature(signature="fused_engram_gate",
                           activation=TORCH_FLOAT, weight=TORCH_FLOAT),
    build=_dtype_floor(_engram_floor_chain),
    fused_op="fused_engram_gate",
))


def _ple_floor_chain(i, torch):
    d = i["value"].shape[-1]
    hc = i["x"].shape[-1] // d

    def norm_gain(x, w):
        y = x.unflatten(-1, (hc, d))
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + i["eps"])
        return (y * (1 + w)).flatten(-2)

    k = norm_gain(i["key"], i["w_key"])
    q = norm_gain(i["x"], i["w_query"])
    dot = (k * q).unflatten(-1, (hc, d)).sum(-1) * d ** -0.5
    # sglang runs this span as written, and its gate line is
    # `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`
    # (srt/models/qwen4_exp.py:1377) — five elementwise launches, which is
    # what the `signed_sqrt` op's cost coefficient prices. `copysign` folds
    # `sign` and the multiply into one: four steps where the framework runs
    # five, so the floor would be timing a shorter chain than the span.
    # `_engram_floor_chain` above keeps `copysign` because that is ITS
    # model's lowering; the two oracles are named in `refs.py`.
    gate = torch.sigmoid(dot.abs().clamp_min(1e-6).sqrt() * dot.sign())
    gated = gate.unsqueeze(-1) * i["value"].unsqueeze(-2)
    return gated.flatten(-2), norm_gain(gated.flatten(-2), i["w_conv"])


def _ple_floor_build(inv: Invocation, env: Env,
                     inputs: Dict[str, Any]) -> BenchCase:
    import torch
    dt = _torch_dtype(inv.dtype)
    cast = {k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point()
                else v) for k, v in inputs.items()}
    return BenchCase(
        run=lambda: torch.stack(_ple_floor_chain(cast, torch), dim=-2),
        to_canonical=lambda o: o.float())


def _vllm_ple_build(inv: Invocation, env: Env,
                    inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `ple_gate`, with key/value as the strided splits of one kv
    projection row (ple_layer.py:421-423); the Triton kernel takes both row
    strides, while hidden and all gains stay contiguous."""
    _require_cuda()
    import torch
    from vllm.models.qwen4_exp.common.ops.ple import ple_gate
    dt = _torch_dtype(inv.dtype)
    key = inputs["key"].to(dt).flatten(0, 1)
    value = inputs["value"].to(dt).flatten(0, 1)
    hidden = inputs["x"].to(dt).flatten(0, 1).contiguous()
    kv = torch.empty(key.shape[0], key.shape[1] + value.shape[1],
                     dtype=dt, device=key.device)
    kv[:, :key.shape[1]] = key
    kv[:, key.shape[1]:] = value
    key, value = kv[:, :key.shape[1]], kv[:, key.shape[1]:]
    weights = tuple(inputs[n].to(dt).contiguous()
                    for n in ("w_key", "w_query", "w_conv"))
    out = torch.empty(key.shape[0], 2, key.shape[1], dtype=dt,
                      device=key.device)

    def run():
        gated, normed = ple_gate(key, value, hidden, *weights,
                                 float(inputs["eps"]))
        out[:, 0] = gated
        out[:, 1] = normed
        return out
    return BenchCase(run=run, to_canonical=lambda o: o.float())


for _lib, _act in (("vllm", "bf16"), ("vllm", "fp16"),
                   ("torch", "bf16"), ("torch", "fp16")):
    # The PLE gate is Qwen3.8's, and the Qwen3.8 checkpoint that has one
    # (Qwen3.8-Flash-Next) ships bf16; the corpus's one fp16 model is
    # Llama-2-7b, which has no PLE. The fp16 twin is the same launch at a
    # width nothing here deploys.
    register(Provider(
        library=_lib, signature="ple_gate",
        kernel_impl=f"{_lib}_ple_gate_{_act}",
        import_name=("vllm.models.qwen4_exp.common.ops.ple"
                     if _lib == "vllm" else "torch"),
        serves=KernelSignature(signature="ple_gate", activation=(_act,),
                               weight=(_act,), output=(_act,),
                               secondary_outputs=(("gated", _act),)),
        unreached_reason=("" if _act == "bf16" else
                          "no corpus checkpoint with a PLE gate runs fp16; "
                          "Qwen3.8-Flash-Next ships bf16"),
        build=(_vllm_ple_build if _lib == "vllm"
               else _ple_floor_build),
        fused_op="ple_gate",
    ))


def _mamba2_norm_floor(i, torch):
    x = i["x"] * torch.nn.functional.silu(i["gate"])
    group = int(i["group_size"])
    grouped = x.unflatten(-1, (-1, group))
    normed = (grouped * torch.rsqrt(
        grouped.pow(2).mean(-1, keepdim=True) + i["eps"])).flatten(-2)
    return normed * i["weight"]


def _sgl_mamba2_norm_build(inv: Invocation, env: Env,
                           inputs: Dict[str, Any]) -> BenchCase:
    """sglang `Mixer2RMSNormGated.forward_cuda`: x is the scan's contiguous
    output, while gate is the leading strided slice of the projection row
    that also carries conv/B/C and dt columns (mamba.py:485-493)."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.fla.layernorm_gated import (
        rms_norm_gated)
    dt = _torch_dtype(inv.dtype)
    x = inputs["x"].to(dt).flatten(0, 1).contiguous()
    gate_values = inputs["gate"].to(dt).flatten(0, 1)
    width = gate_values.shape[-1]
    row_width = (width + 2 * int(inputs["d_ssm_state"])
                 * int(inv.dims["n_ssm_groups"])
                 + int(inv.dims["n_ssm_heads"]))
    row = torch.empty(gate_values.shape[0], row_width, dtype=dt,
                      device=x.device)
    row[:, :width] = gate_values
    gate = row[:, :width]
    weight = inputs["weight"].to(dt).flatten().contiguous()
    return BenchCase(
        run=lambda: rms_norm_gated(
            x=x, weight=weight, bias=None, z=gate,
            eps=float(inputs["eps"]), group_size=int(inputs["group_size"]),
            norm_before_gate=False, is_rms_norm=True),
        to_canonical=lambda o: o.reshape(inputs["x"].shape).float())


# The mixer creates its gain in the model dtype, and the Triton kernel takes
# x, gate and gain at one width; tied here rather than accepting a mixed cell
# the serving module never constructs.
for _dt_name in ("bf16", "fp16", "fp32"):
    # One kernel, three widths, and only bf16 is reachable here: the Mamba-2
    # checkpoints in this corpus (Nemotron-3 Nano and Ultra) ship bf16, and
    # the corpus's one fp16 model (Llama-2-7b) has no SSM mixer at all. The
    # twins stay registered — the Triton kernel really does run at those
    # widths — with the reason beside them, so neither reads as a gap.
    _unreached = "" if _dt_name == "bf16" else (
        f"no corpus checkpoint runs a Mamba-2 gated norm in {_dt_name}; "
        f"every SSM model here ships bf16")
    register(Provider(
        library="sglang", signature="mamba2_rmsnorm_gated",
        kernel_impl=f"sglang_mamba2_rmsnorm_gated_{_dt_name}",
        import_name="sglang",
        serves=KernelSignature(signature="mamba2_rmsnorm_gated",
                               activation=(_dt_name,), weight=(_dt_name,)),
        unreached_reason=_unreached,
        build=_sgl_mamba2_norm_build, fused_op="mamba2_rmsnorm_gated",
    ))
    register(Provider(
        library="torch", signature="mamba2_rmsnorm_gated",
        kernel_impl=f"torch_mamba2_rmsnorm_gated_{_dt_name}",
        import_name="torch",
        serves=KernelSignature(signature="mamba2_rmsnorm_gated",
                               activation=(_dt_name,), weight=(_dt_name,)),
        unreached_reason=_unreached,
        build=_dtype_floor(_mamba2_norm_floor),
        fused_op="mamba2_rmsnorm_gated",
    ))


# ---------------------------------------------------------------------------
# Fused norm spans that read two rows and write them in one launch — the MTP
# entry (`fused_eh_norm`) and the attention a-norm pair (`fused_q_kv_rmsnorm`).
# Each is called the way serving calls it at the audited releases (vLLM
# v0.32.0, sglang v0.5.22); a torch floor runs the span's naive chain.
# ---------------------------------------------------------------------------

def _rms_at(x: Any, w: Any, eps: float) -> Any:
    """RMSNorm at `x`'s own dtype, accumulated in fp32 without an fp32 copy
    of the row — what an unfused `rms_norm` does on the card, and what keeps
    the floor's widest operand at the cell's precision (`launch_trace`)."""
    import torch
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True, dtype=torch.float32)
                        + eps).to(x.dtype)
    return y if w is None else y * w.to(x.dtype)


def _cell_floor(chain):
    """A torch floor running the span's naive chain at the cell's dtype.

    `chain(inputs, torch)` gets the canonical inputs with every floating
    tensor cast to the cell's activation dtype (gains included: a bf16 row
    times an fp32 gain is an fp32 row, another precision's launch), and
    returns the span's output — or a tuple of its ports, which
    `to_canonical` lays side by side as the oracle does. Not the oracle's
    own routine: that one computes in fp32 on purpose, and timed it is an
    fp32 latency the driver refuses."""
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        import torch
        dt = _torch_dtype(inv.dtype)
        cast = {k: (v.to(dt) if torch.is_tensor(v) and v.is_floating_point()
                    else v) for k, v in inputs.items()}

        def canonical(out):
            parts = out if isinstance(out, tuple) else (out,)
            return torch.cat([p.float().flatten(1) for p in parts], dim=-1)
        return BenchCase(run=lambda: chain(cast, torch), to_canonical=canonical)
    return build


def _eh_operands(inv: Invocation, inputs: Dict[str, Any]):
    """The two `[tokens, d_model]` rows at the activation dtype and the two
    gains as the cell stores them."""
    dt = _torch_dtype(inv.dtype)
    return (inputs["inputs_embeds"].to(dt).contiguous(),
            inputs["hidden"].to(dt).contiguous(),
            inputs["enorm_w"].contiguous(), inputs["hnorm_w"].contiguous(),
            float(inputs["eps"]))


def _vllm_fused_eh_norm_build(inv: Invocation, env: Env,
                              inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `fused_eh_norm` (models/deepseek_v32/common/kernels.py:1150,
    kernel :1116), as the V3.2 / GLM-5 MTP calls it (nvidia/mtp.py:127).

    The kernel also zeroes the embedding of a token at position 0
    (`tl.where(pos == 0, …)`, :1135) — the MTP's shift of a sequence's first
    token, one row per sequence. The positions handed it are 1..tokens, so
    every row is normalised as the span states it; the masked row costs the
    same load and store either way. GLM-5.3-Flash's own copy
    (models/glm5next/nvidia/ops/fused_eh_norm.py:52) is the same kernel
    source line for line, so it is not registered a second time."""
    _require_cuda()
    import torch
    from vllm.models.deepseek_v32.common.kernels import fused_eh_norm
    embeds, hidden, ew, hw, eps = _eh_operands(inv, inputs)
    positions = torch.arange(1, embeds.shape[0] + 1, dtype=torch.int64,
                             device=embeds.device)
    return BenchCase(
        run=lambda: fused_eh_norm(positions, embeds, hidden, ew, hw, eps),
        to_canonical=lambda out: out.float())


def _sgl_fused_eh_norm_build(inv: Invocation, env: Env,
                             inputs: Dict[str, Any]) -> BenchCase:
    """sglang `fused_eh_norm` (kernels/ops/layernorm/fused_eh_norm.py:33), as
    `DeepseekModelNextN.forward` calls it (srt/models/deepseek_nextn.py:203).
    It takes no positions: the caller zeroes the first token's embedding
    itself. Its gains must be in the activation dtype (`TensorMatcher(...)
    .with_dtype<T>()`, kernels/jit/csrc/elementwise/fused_eh_norm.cuh:90-91),
    which is what sglang holds them in; a cell that declares fp32 gains
    (DeepSeek's MTP norms) gets them cast, two `d_model` vectors beside the
    rows."""
    _require_cuda()
    from sglang.kernels.ops.layernorm.fused_eh_norm import fused_eh_norm
    embeds, hidden, ew, hw, eps = _eh_operands(inv, inputs)
    ew, hw = ew.to(embeds.dtype), hw.to(embeds.dtype)
    return BenchCase(
        run=lambda: fused_eh_norm(embeds, hidden, ew, hw, eps),
        to_canonical=lambda out: out.float())


register(Provider(
    library="vllm", signature="fused_eh_norm",
    kernel_impl="vllm_fused_eh_norm",
    import_name="vllm.models.deepseek_v32.common.kernels",
    serves=KernelSignature(signature="fused_eh_norm", activation=("bf16",)),
    build=_vllm_fused_eh_norm_build, fused_op="fused_eh_norm",
))
#: The hidden sizes sglang's JIT builds `fused_eh_norm` for
#: (`is_supported_fused_eh_norm_hidden_size`, fused_eh_norm.py:18-19): a
#: multiple of 256 in (256, 8192] — 256 itself refused, which neither a
#: modulus nor a ceiling states.
_EH_NORM_WIDTHS = tuple(range(512, 8193, 256))

register(Provider(
    library="sglang", signature="fused_eh_norm",
    kernel_impl="sglang_fused_eh_norm",
    import_name="sglang.kernels.ops.layernorm.fused_eh_norm",
    serves=KernelSignature(
        signature="fused_eh_norm",
        # fp16 / bf16 only (:41-43).
        activation=("bf16", "fp16"), enum=(("d_model", _EH_NORM_WIDTHS),)),
    build=_sgl_fused_eh_norm_build, fused_op="fused_eh_norm",
))
register(Provider(
    library="torch", signature="fused_eh_norm",
    kernel_impl="torch_fused_eh_norm", import_name="torch",
    serves=KernelSignature(signature="fused_eh_norm", activation=TORCH_FLOAT),
    build=_cell_floor(lambda i, torch: torch.cat((
        _rms_at(i["inputs_embeds"], i["enorm_w"], i["eps"]),
        _rms_at(i["hidden"], i["hnorm_w"], i["eps"])), dim=-1)),
    fused_op="fused_eh_norm",
))


#: The fused projection row's tail past the q|kv columns, per bound model:
#: DeepSeek-V4 / V4.1 fuse exactly q|kv (`fused_wqa_wkv` sized
#: `[q_lora_rank, head_dim]`, deepseek_v4/attention.py:248-255), and
#: GLM-5.3-Flash's row is likewise tail-free (`qk_rope_head_dim: 0`,
#: snapshots/zai-org__GLM-5.3-Flash.json; `mla_use_nope`, its split at
#: model_executor/layers/mla.py:184-187). Kimi-K3's `qkv_lora` carries the
#: 64-wide rope tail beside the two (kimi_k3/nvidia/mla.py:518-521 — wider
#: still when the output-gate projection is fused in, :546-584, a runtime
#: branch, so 2112 is the floor). The (1536, 512) cell is shared by all
#: three — the invocation carries nothing that tells them apart — and is
#: built at Kimi-K3's geometry, stated here once, the treatment the DSA
#: index-rope pairing gets; the tail's columns are never written, only the
#: pitch the kernel is handed (`q_in_stride` / `kv_in_stride`,
#: fused_qk_rmsnorm.py:181-184) changes.
_QKV_PROJ_ROPE_TAILS = {(1536, 512): 64}


def _vllm_fused_q_kv_rmsnorm_build(inv: Invocation, env: Env,
                                   inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `fused_q_kv_rmsnorm` (models/common/ops/fused_qk_rmsnorm.py:195,
    `FusedQKVRMSNormKernel`), as `DeepseekV4Attention._split_qkv_and_norm`
    calls it (models/deepseek_v4/attention.py:518-535). Serving splits the
    two rows off ONE fused projection and hands the kernel the strided
    views: `qr_kv.split([q_lora_rank, kv_width], dim=-1)`
    (deepseek_v4/attention.py:527-528; Kimi-K3 the same off `qkv_lora`,
    kimi_k3/nvidia/mla.py:518-521, its row wider by the rope tail —
    `_QKV_PROJ_ROPE_TAILS`). The wrapper asserts only last-dim contiguity
    and passes each row's stride to the kernel (fused_qk_rmsnorm.py:207,
    `:181-184`), so the launch is given the same views of one projection
    row — not contiguous copies, whose `stride(0) == width` launch serving
    never makes. The gains are production-contiguous (asserted at
    :208-209); the outputs are the wrapper's own fresh buffers (:211-212)."""
    _require_cuda()
    import torch
    from vllm.models.common.ops.fused_qk_rmsnorm import fused_q_kv_rmsnorm
    dt = _torch_dtype(inv.dtype)
    q0, kv0 = inputs["q"].to(dt), inputs["kv"].to(dt)
    tokens, dq = q0.shape
    dkv = int(kv0.shape[1])
    tail = _QKV_PROJ_ROPE_TAILS.get((dq, dkv), 0)
    qr_kv = torch.empty(tokens, dq + dkv + tail, dtype=dt, device=q0.device)
    qr_kv[:, :dq] = q0
    qr_kv[:, dq:dq + dkv] = kv0
    q, kv = qr_kv[:, :dq], qr_kv[:, dq:dq + dkv]
    qw = inputs["q_weight"].contiguous()
    kvw = inputs["kv_weight"].contiguous()
    eps = float(inputs["eps"])
    return BenchCase(
        run=lambda: fused_q_kv_rmsnorm(q, kv, qw, kvw, eps),
        to_canonical=lambda out: torch.cat([out[0].float(), out[1].float()],
                                           dim=-1))


#: The axes whose presence makes a `fused_q_kv_rmsnorm` cell the per-head
#: geometry (`OpSignature.variant_selectors`), read off the signature so the
#: two kernels below split the cells along the declaration that selects them.
_PER_HEAD_AXES = get_signature("fused_q_kv_rmsnorm").variant_selectors

register(Provider(
    library="vllm", signature="fused_q_kv_rmsnorm",
    kernel_impl="vllm_fused_q_kv_rmsnorm",
    import_name="vllm.models.common.ops.fused_qk_rmsnorm",
    # The row form only: the call sites the binding names hand it one q-LoRA
    # row and one latent row per token (deepseek_v4/attention.py:527-529).
    serves=KernelSignature(signature="fused_q_kv_rmsnorm",
                           activation=("bf16",)),
    build=_vllm_fused_q_kv_rmsnorm_build, fused_op="fused_q_kv_rmsnorm",
))


def _sgl_fused_inplace_qknorm_build(inv: Invocation, env: Env,
                                    inputs: Dict[str, Any]) -> BenchCase:
    """sglang `fused_inplace_qknorm` (kernels/ops/layernorm/norm.py:114,
    JIT `QKNormKernel`, kernels/jit/csrc/elementwise/qknorm.cuh:252), as
    `apply_qk_norm` calls it (srt/models/utils.py:514-531) from the Qwen3 /
    Qwen3-MoE / GLM-4-MoE attentions (qwen3.py:173-182, glm4_moe.py:316-326).

    Serving splits q, k and v off ONE qkv projection row
    (`qkv.split([q_size, kv_size, kv_size], dim=-1)`, qwen3.py:174) and
    hands the kernel `q.view(T, -1, head_dim)` / `k.view(T, -1, head_dim)`
    (utils.py:524-525): strided views whose token stride is the whole
    `(n_heads + 2·n_kv_heads)·d_head` row, which the kernel takes as its
    `q_stride` / `k_stride` (qknorm.cuh:130-139, `with_strides({Sq, D, 1})`;
    :204-213 for the CTA form above 256).
    Both are normalised in place (utils.py:531 returns q and k themselves);
    v is never read. The gains are the `RMSNorm` weights, held at the model
    dtype — the kernel matches them to q's dtype (qknorm.cuh:140-144, :214-218) — so a
    cell that stores them wider gets them cast, two `[d_head]` vectors.

    The cross-check runs the launch once on a fresh buffer and reads the
    q|k columns back; the timed repeats renormalise the same columns, the
    same bytes."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
    dt = _torch_dtype(inv.dtype)
    q0, k0 = inputs["q"].to(dt), inputs["kv"].to(dt)
    tokens, n_heads, d_head = q0.shape
    n_kv = int(k0.shape[1])
    q_cols, k_cols = n_heads * d_head, n_kv * d_head
    qkv = torch.empty(tokens, q_cols + 2 * k_cols, dtype=dt, device=q0.device)
    qkv[:, :q_cols] = q0.flatten(1)
    qkv[:, q_cols:q_cols + k_cols] = k0.flatten(1)
    qkv[:, q_cols + k_cols:].normal_()
    q, k, _v = qkv.split([q_cols, k_cols, k_cols], dim=-1)
    q, k = q.view(tokens, -1, d_head), k.view(tokens, -1, d_head)
    qw = inputs["q_weight"].to(dt).contiguous()
    kw = inputs["kv_weight"].to(dt).contiguous()
    eps = float(inputs["eps"])
    return BenchCase(
        run=lambda: fused_inplace_qknorm(q=q, k=k, q_weight=qw, k_weight=kw,
                                         head_dim=d_head, eps=eps),
        to_canonical=lambda _o: qkv[:, :q_cols + k_cols].float())


register(Provider(
    library="sglang", signature="fused_q_kv_rmsnorm",
    kernel_impl="sglang_fused_inplace_qknorm",
    import_name="sglang.kernels.ops.layernorm.norm",
    serves=KernelSignature(
        signature="fused_q_kv_rmsnorm",
        # The per-head form only, both phases (apply_qk_norm runs on every
        # forward). The JIT is built for these head widths alone
        # (`can_use_fused_inplace_qknorm`, norm.py:101-110) and for fp16 /
        # bf16 (static_assert, qknorm.cuh:108, :181).
        requires_dims=_PER_HEAD_AXES,
        activation=("bf16", "fp16"),
        enum=(("d_head", (64, 128, 256, 512, 1024)),)),
    build=_sgl_fused_inplace_qknorm_build, fused_op="fused_q_kv_rmsnorm",
))
register(Provider(
    library="torch", signature="fused_q_kv_rmsnorm",
    kernel_impl="torch_fused_q_kv_rmsnorm", import_name="torch",
    # Both geometries: the row form and the per-head one.
    serves=KernelSignature(signature="fused_q_kv_rmsnorm",
                           activation=TORCH_FLOAT,
                           variant_dims=_PER_HEAD_AXES),
    build=_cell_floor(lambda i, torch: (
        _rms_at(i["q"], i["q_weight"], i["eps"]),
        _rms_at(i["kv"], i["kv_weight"], i["eps"]))),
    fused_op="fused_q_kv_rmsnorm",
))


# --- Gemma-4 (sglang v0.5.22, kernels/ops/layernorm/gemma4_fused_ops.py) ----

def _sgl_gemma_qkv_rmsnorm_build(inv: Invocation, env: Env,
                                 inputs: Dict[str, Any]) -> BenchCase:
    """`gemma_qkv_rmsnorm` (:246, kernel :161), as `Gemma4Attention.forward`
    calls it on the q/k/v slices of the qkv projection (srt/models/
    gemma4_causal.py:478-488): in place, with the head counts and width as
    arguments. The launch shape is the wrapper's own choice by token count
    (one program per token and head up to 256 tokens, :303)."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm
    dt = _torch_dtype(inv.dtype)
    # One buffer laid out as the projection writes it, sliced as :448 slices
    # it — the kernel reads each slice at the buffer's row stride.
    q0, k0, v0 = (inputs[n].to(dt) for n in ("q", "k", "v"))
    qkv = torch.cat([q0, k0, v0], dim=-1).contiguous()
    q, k, v = qkv.split([q0.shape[1], k0.shape[1], v0.shape[1]], dim=-1)
    d_head = int(inputs["q_weight"].shape[0])
    n_heads, n_kv = q0.shape[1] // d_head, k0.shape[1] // d_head
    qw, kw = inputs["q_weight"], inputs["k_weight"]
    eps = float(inputs["eps"])
    return BenchCase(
        run=lambda: gemma_qkv_rmsnorm(q, k, v, qw, kw, num_q_heads=n_heads,
                                      num_kv_heads=n_kv, head_dim=d_head,
                                      eps=eps),
        to_canonical=lambda _o: qkv.float())


def _sgl_gemma_rmsnorm_residual_scalar_build(inv: Invocation, env: Env,
                                             inputs: Dict[str, Any]
                                             ) -> BenchCase:
    """`gemma_rmsnorm_residual_scalar` (:56), as `Gemma4DecoderLayer.
    _next_stream` calls it (gemma4_causal.py:804-818)."""
    _require_cuda()
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import (
        gemma_rmsnorm_residual_scalar,
    )
    dt = _torch_dtype(inv.dtype)
    x, residual = inputs["x"].to(dt), inputs["residual"].to(dt)
    w, scalar, eps = inputs["w"], inputs["scalar"], float(inputs["eps"])
    return BenchCase(
        run=lambda: gemma_rmsnorm_residual_scalar(x, w, residual, scalar, eps),
        to_canonical=lambda out: out.float())


def _sgl_gemma_dual_rmsnorm_residual_scalar_build(inv: Invocation, env: Env,
                                                  inputs: Dict[str, Any]
                                                  ) -> BenchCase:
    """`gemma_dual_rmsnorm_residual_scalar` (:322), as the MoE layers'
    `_next_stream` calls it (gemma4_causal.py:771-791), one epsilon per norm
    — all three the same here, as in every Gemma-4 config."""
    _require_cuda()
    from sglang.kernels.ops.layernorm.gemma4_fused_ops import (
        gemma_dual_rmsnorm_residual_scalar,
    )
    dt = _torch_dtype(inv.dtype)
    x1, x2, residual = (inputs[n].to(dt) for n in ("x1", "x2", "residual"))
    w1, w2, w3, scalar = (inputs[n] for n in ("w1", "w2", "w3", "scalar"))
    eps = float(inputs["eps"])
    return BenchCase(
        run=lambda: gemma_dual_rmsnorm_residual_scalar(
            x1, w1, x2, w2, w3, residual, scalar, eps, eps, eps),
        to_canonical=lambda out: out.float())


def _gemma_qkv_chain(i, torch):
    """The three per-head norms, each a launch of its own."""
    d = int(i["q_weight"].shape[0])

    def per_head(x, w):
        return _rms_at(x.view(x.shape[0], -1, d), w, i["eps"])
    return (per_head(i["q"], i["q_weight"]), per_head(i["k"], i["k_weight"]),
            per_head(i["v"], None))


def _gemma_dual_chain(i, torch):
    eps = i["eps"]
    branches = _rms_at(i["x1"], i["w1"], eps) + _rms_at(i["x2"], i["w2"], eps)
    return (_rms_at(branches, i["w3"], eps) + i["residual"]) * i["scalar"]


for _sig, _build, _floor in (
    ("gemma_qkv_rmsnorm", _sgl_gemma_qkv_rmsnorm_build, _gemma_qkv_chain),
    ("gemma_rmsnorm_residual_scalar", _sgl_gemma_rmsnorm_residual_scalar_build,
     lambda i, torch: (_rms_at(i["x"], i["w"], i["eps"]) + i["residual"])
     * i["scalar"]),
    ("gemma_dual_rmsnorm_residual_scalar",
     _sgl_gemma_dual_rmsnorm_residual_scalar_build, _gemma_dual_chain),
):
    register(Provider(
        library="sglang", signature=_sig, kernel_impl=f"sglang_{_sig}",
        import_name="sglang.kernels.ops.layernorm.gemma4_fused_ops",
        # Gemma-4 serves bf16; no other width measured.
        serves=KernelSignature(signature=_sig, activation=("bf16",)),
        build=_build, fused_op=_sig,
    ))
    register(Provider(
        library="torch", signature=_sig, kernel_impl=f"torch_{_sig}",
        import_name="torch",
        serves=KernelSignature(signature=_sig, activation=TORCH_FLOAT),
        build=_cell_floor(_floor), fused_op=_sig,
    ))


# --- elementwise gates and caps --------------------------------------------

#: The head width serving's gate view is cut at: both bound models run
#: 256-wide heads (snapshots/Qwen__Qwen3.5-35B-A3B.json — 16 query heads =
#: the 4096-wide gate the walk bills; Qwen__Qwen3.8-Flash-Next.json,
#: Qwen4Exp — 24 query heads), and `forward_prepare_native` leaves the gate
#: as the 3-D strided view `q_gate.view(T, heads, 2·d_head).chunk(2, -1)[1]`
#: of the qkv row (srt/models/qwen3_5.py:1465-1473). `HEAD_DIM` is a kernel
#: constexpr and the wrapper takes its explicit-stride path on that view
#: (kernels/ops/elementwise/elementwise.py:430-445), so the cell's flat gate
#: is rebuilt as the same view rather than flattened — the flat path the old
#: build took (:446-454, `gate_stride_head = hidden_dim`) is a launch
#: serving never makes. A cell whose width is not a 256-multiple is no Qwen
#: geometry; it is cut at one head so the strided path is still the one
#: timed.
_QWEN_GATE_HEAD_DIM = 256

#: The KV head count beside the q|gate columns in the qkv row, keyed by
#: the FULL q width (the cell's local width × tp): serving splits
#: `[2·q_size, kv_size, kv_size]` (qwen3_5.py:1465-1467) with `q_size` /
#: `kv_size` the TP-LOCAL head counts × 256 (`:1257-1268`, `num_kv_heads =
#: max(1, total // kv_tp)`), so the gate view's row stride spans the full
#: row — `gate_stride_row` a runtime argument the kernel indexes every
#: gate address by (elementwise.py:394-406, passed at :441-442), not an
#: unread tail. The bound models (snapshots' `num_attention_heads` /
#: `num_key_value_heads`, head_dim 256): Qwen3.5-35B-A3B 16 q / 2 KV,
#: Qwen3.8-27B 24 / 4, Qwen3.8-2.4T-A95B 64 / 4.
#:
#: COLLISION, stated: Qwen3.8-Flash-Next (Qwen4Exp) is ALSO 24 heads of
#: 256 — the same `d_model` 6144 cell as Qwen3.8-27B — but runs TWO KV
#: heads (its snapshot), a 13312 row against the 14336 built here. The
#: invocation carries nothing that tells the two apart; 27B's geometry is
#: built, as the review asks, the treatment the (1536, 512) a-norm cell
#: gets.
_QWEN_GATE_KV_HEADS = {4096: 2, 6144: 4, 16384: 4}


def _qwen_gate_view(b: "Any", tp: int = 1) -> "Any":
    """The flat `[tokens, heads·d_head]` gate as serving's strided view of
    the whole qkv projection row: per head the query half then the gate
    half, then the k|v columns, so the gate is `[tokens, heads, d_head]` at
    `stride(-2) == 2·d_head` and `stride(0) == 2·hidden + 2·kv_size` — the
    pitch serving's slice of the qkv row has, with `kv_size` the local KV
    heads (`_QWEN_GATE_KV_HEADS` by the full width, sharded as
    `max(1, total // tp)`, qwen3_5.py:1261-1265). The query half and the
    k|v columns are never written: the kernel's gate addressing stays
    inside each head's gate half (`d < HEAD_DIM`, elementwise.py:405). A
    width no bound model has falls back to one KV column per head — a
    synthetic stand-in that keeps the 3-D strided path timed, not a
    production layout."""
    import torch
    tokens, hidden = b.shape
    d_head = (_QWEN_GATE_HEAD_DIM if hidden % _QWEN_GATE_HEAD_DIM == 0
              else hidden)
    heads = hidden // d_head
    total_kv = _QWEN_GATE_KV_HEADS.get(hidden * tp)
    kv = (max(1, total_kv // tp) * d_head if total_kv is not None
          else d_head)
    qkv = torch.empty(tokens, 2 * hidden + 2 * kv, dtype=b.dtype,
                      device=b.device)
    q_gate = qkv[:, :2 * hidden].view(tokens, heads, 2 * d_head)
    q_gate[..., d_head:] = b.view(tokens, heads, d_head)
    return q_gate.chunk(2, dim=-1)[1]


def _sgl_fused_sigmoid_mul_build(inv: Invocation, env: Env,
                                 inputs: Dict[str, Any]) -> BenchCase:
    """sglang `fused_sigmoid_mul` (kernels/ops/elementwise/elementwise.py:419,
    kernel :380), as Qwen3.5 / Qwen4Exp gated attention calls it
    (srt/models/qwen3_5.py:1582, qwen4_exp.py:1879): in place on the
    attention output, the gate the 3-D strided view serving leaves it as
    (`_qwen_gate_view`)."""
    _require_cuda()
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
    dt = _torch_dtype(inv.dtype)
    a = inputs["a"].to(dt).contiguous().clone()
    b = _qwen_gate_view(inputs["b"].to(dt), inv.tp)
    return BenchCase(run=lambda: fused_sigmoid_mul(a, b, inplace=True),
                     to_canonical=lambda out: out.float())


def _sgl_sigmoid_gate_mul_build(inv: Invocation, env: Env,
                                inputs: Dict[str, Any]) -> BenchCase:
    """sglang `sigmoid_gate_mul` (kernels/ops/moe/triton_sigmoid_gate_mul.py:38,
    kernel :22), as HY V4 calls it on its attention output
    (srt/models/hunyuan_v4.py:505): same-shape, contiguous, out of place."""
    _require_cuda()
    from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import sigmoid_gate_mul
    dt = _torch_dtype(inv.dtype)
    a = inputs["a"].to(dt).contiguous()
    b = inputs["b"].to(dt).contiguous()
    return BenchCase(run=lambda: sigmoid_gate_mul(a, b),
                     to_canonical=lambda out: out.float())


def _padded_vocab_row(x: "Any") -> "Any":
    """The logits row as the LM head writes it and the logits processor
    slices it: sglang pads the vocab to a multiple of 64
    (`pad_vocab_size(vocab, 64)`, layers/vocab_parallel_embedding.py:57-60),
    and `_copy_logits_to_buffer` caps the `[:, :vocab]` slice of that
    padded row (layers/logits_processor.py:1162-1163) — a strided view
    whenever the cell's vocab is no 64-multiple, the whole row (contiguous,
    the kernel's flat path, softcap.py:110-112) when it is. The kernel
    takes the row stride as an argument (:108-137), so the launch is handed
    the layout serving hands it, not a contiguous copy."""
    import torch
    tokens, vocab = x.shape
    padded = -(-vocab // 64) * 64
    row = torch.empty(tokens, padded, dtype=x.dtype, device=x.device)
    row[:, :vocab] = x
    return row[:, :vocab]


def _sgl_softcap_build(inv: Invocation, env: Env,
                       inputs: Dict[str, Any]) -> BenchCase:
    """sglang `softcap_to_float32_logits` (kernels/ops/activation/softcap.py:145,
    kernel :72, CAST_TO_FP32), as `LogitsProcessor._copy_logits_to_buffer`
    calls it (srt/layers/logits_processor.py:1171) on bf16/fp16 logits: the
    cap and the fp32 upcast in one launch, into an fp32 buffer the caller
    allocates (`torch.empty_like` of the strided slice — contiguous,
    :1164-1168, which the kernel's `output.stride(0) >= ncols` admits,
    softcap.py:119-122)."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.activation import softcap as S
    dt = _torch_dtype(inv.dtype)
    x = _padded_vocab_row(inputs["x"].to(dt))
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    cap = float(inputs["cap"])
    return BenchCase(run=lambda: S.softcap_to_float32_logits(x, cap, out),
                     to_canonical=lambda _o: out)


def _sgl_softcap_inplace_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """sglang `softcap_inplace_logits` (softcap.py:141 — the same kernel
    without the upcast), which `LogitsProcessor` takes when the logits are
    already fp32 (logits_processor.py:960-962): Gemma-2, which ships fp32.
    The logits are the same padded-row slice as the upcast path's — capped
    in place on it (:1162-1163, then `fused_softcap`)."""
    _require_cuda()
    from sglang.kernels.ops.activation import softcap as S
    x = _padded_vocab_row(inputs["x"].float())
    cap = float(inputs["cap"])
    return BenchCase(run=lambda: S.softcap_inplace_logits(x, cap),
                     to_canonical=lambda _o: x)


_SOFTCAP = "softcap_" + "logits"
for _lib, _sig, _impl, _imp, _build, _act, _out in (
    ("sglang", "sigmoid_gate_mul", "sglang_fused_sigmoid_mul",
     "sglang.kernels.ops.elementwise.elementwise",
     _sgl_fused_sigmoid_mul_build, ("bf16",), ()),
    ("sglang", "sigmoid_gate_mul", "sglang_sigmoid_gate_mul",
     "sglang.kernels.ops.moe.triton_sigmoid_gate_mul",
     _sgl_sigmoid_gate_mul_build, ("bf16",), ()),
    ("torch", "sigmoid_gate_mul", "torch_sigmoid_gate_mul", "torch",
     _cell_floor(lambda i, torch: i["a"] * torch.sigmoid(i["b"])),
     TORCH_FLOAT, ()),
    # The fused path is taken for fp16 / bf16 logits only
    # (logits_processor.py:948-952); fp32 logits take the in-place kernel.
    # Both WRITE fp32 — the first upcasts into the caller's fp32 buffer, the
    # second caps the fp32 row in place — so a cell billed at another output
    # dtype is not theirs to measure, whatever the walk states today.
    ("sglang", _SOFTCAP, "sglang_softcap_to_float32",
     "sglang.kernels.ops.activation.softcap", _sgl_softcap_build,
     ("bf16", "fp16"), ("fp32",)),
    ("sglang", _SOFTCAP, "sglang_softcap_inplace",
     "sglang.kernels.ops.activation.softcap", _sgl_softcap_inplace_build,
     ("fp32",), ("fp32",)),
    # The plan's three ops at the logits' dtype.
    ("torch", _SOFTCAP, "torch_softcap", "torch",
     _cell_floor(lambda i, torch: torch.tanh(i["x"] / i["cap"]) * i["cap"]),
     TORCH_FLOAT, ()),
):
    register(Provider(
        library=_lib, signature=_sig, kernel_impl=_impl, import_name=_imp,
        serves=KernelSignature(signature=_sig, activation=_act, output=_out),
        build=_build, fused_op=_sig,
    ))
