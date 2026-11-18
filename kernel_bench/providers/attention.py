"""Attention providers — FA3 / FlashInfer / FlashMLA.

Each targets an attention op-signature and, when its library is present,
measures the real kernel. Providers register UNCONDITIONALLY; `available()`
gates execution on the library being importable in the current env. So on
a box without the library (or without a GPU) the provider is known but
simply never runs — the driver reports "no applicable providers".

The library import + the actual kernel call live LAZILY inside `build`
(not at module import), so `discover()` never needs the library present.
`build` allocates inputs at the PER-RANK shape (heads sharded by TP, so
the measured kernel matches the shape the estimate assumes) and returns a
zero-arg callable the driver times.

GPU-touching bodies are written against the real library APIs so filling
them out on a GPU box is a matter of turning them on, not designing them.
They raise clearly if reached without CUDA.
"""
from __future__ import annotations

from ..env import Env
from ..invocation import Invocation
from .. import refs as _refs
from ..refs import _sdpa_causal_args, torch_dtype as _torch_dtype, visible_keys
from backend.inference.benchmark_identity import PLAIN
from . import TORCH_FLOAT, BenchCase, KernelSignature, Provider, register


# The cache a kernel reads, by cache-format name (`KernelSignature.
# kv_format`), for the kernels below that read the elements a plain or a
# per-tensor-scaled cache stores. FA2/3/4 and FlashInfer read an fp8 cache
# with one dequant scale a tensor (`k_descale`/`v_descale` expanded from the
# layer's scalar, vLLM v0.32.0 v1/attention/backends/flash_attn.py:1283-1304;
# sglang v0.5.22 flashattention_backend.py:1385; FlashInfer's `k_scale` /
# `v_scale` floats, flashinfer.py:401) — vLLM's `--kv-cache-dtype fp8`, the
# `fp8_per_tensor` record — and its bf16/fp16 cache plain. The builds pass
# no descale, so the launch runs at scale 1: the same kernel, the scalar a
# runtime operand. The torch floors read the contract's K/V at the cell's
# `kv` dtype and no scale; one scalar is no bytes and folds into the
# softmax scale, so those cells are theirs too, and a record with scales
# per group or a bf16 tail (FlashMLA's) is bytes they never read.
_PLAIN_OR_PER_TENSOR = (PLAIN, "fp8_per_tensor")


# torch SDPA takes heads-before-sequence: swap the contract's seq and head
# axes. Axis names stay on the signature — GQA's `d_head` and MLA's
# `d_v_head` come through `layouts_for`, not a second copy here.
_SDPA_HEADS_FIRST = (0, 2, 1, 3)
_SDPA_LAUNCH = (
    ("q", _SDPA_HEADS_FIRST),
    ("k", _SDPA_HEADS_FIRST),
    ("v", _SDPA_HEADS_FIRST),
)
# FA / FlashInfer varlen: fold batch and sequence into one token axis.
# cu_seqlens stay in `build`.
_PACK_BATCH_SEQ = ((0, 1), 2, 3)
_ATTN_VARLEN_LAUNCH = (
    ("q", _PACK_BATCH_SEQ),
    ("k", _PACK_BATCH_SEQ),
    ("v", _PACK_BATCH_SEQ),
)


def _require_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "attention provider reached without CUDA — this body only runs "
            "in a GPU provider env. (available() should have gated it.)"
        )


def _primary(out):
    """The attention output out of whatever the launch returned: the tensor
    itself, or the first of several. Several arrive as a tuple from a Python
    wrapper and as a LIST from a C++ entry point that returns
    `std::vector<Tensor>` — FlashMLA's `flash_mla_sparse_fwd` hands back
    `flash_mla_cuda.sparse_prefill_fwd`'s result as is
    (flash_mla/flash_mla_interface.py:217-220 at ba89a34), and a tuple-only
    test read that list as the output and sliced it: "list indices must be
    integers or slices, not tuple" on every GLM-5 cell (2026-10-28)."""
    return out[0] if isinstance(out, (tuple, list)) else out


# ---------------------------------------------------------------------------
# FlashAttention family (FA2 / FA3 / FA4) — dense GQA/MHA.
#
# All target the same op-signature (gqa_attention). They differ by version
# AND by import path: FA2 lives in `flash_attn`, FA3 (Hopper) ships as a
# separate `flash_attn_interface` build, FA4 (unreleased) will be another
# module. So each is its own version-keyed provider, gated by the module it
# imports from — run in its own isolated env (envs/fa2.txt, fa3.txt, …).
# The prefill/decode kernels take the SAME shapes across versions, so the
# input builders are shared; only the import + kernel_impl tag differ.
# ---------------------------------------------------------------------------

def _varlen_meta(inv, inputs):
    """cu_seqlens for already-packed varlen q/k/v.

    Packing is `launch_inputs`; this is the library convention those
    kernels still need: token-axis prefixes and the sequence lengths.
    """
    import torch
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    B = int(inv.dims["B"])
    Sq = int(inv.dims["s_query"])
    Lq = int(inv.dims["s_kv"])
    N, D = q.shape[1], q.shape[2]
    Dv = v.shape[-1]
    dev = q.device
    cu_q = torch.arange(0, (B + 1) * Sq, Sq, dtype=torch.int32, device=dev)
    cu_k = torch.arange(0, (B + 1) * Lq, Lq, dtype=torch.int32, device=dev)
    return q, k, v, cu_q, cu_k, Sq, Lq, B, N, D, Dv


def _sliding_window(inv) -> int | None:
    """The SWA window in tokens, or None for full causal.

    Reads `s_window` from kernel_params (the plan's per-node copy) then
    the identity dims. Presence is the variant: a hybrid's full-attention
    leaf does not carry the key.
    """
    raw = (getattr(inv, "kernel_params", None) or {}).get("s_window")
    if raw is None and "s_window" in getattr(inv, "dims", {}):
        raw = inv.dims["s_window"]
    try:
        w = int(raw)
    except (TypeError, ValueError):
        return None
    return w if w > 0 else None


def _fa_window_size(inv):
    """vLLM / FA convention: `window_size=(sliding_window-1, 0)`.

    None means omit the argument so the library keeps its full-attention
    default `(-1, -1)`.
    """
    w = _sliding_window(inv)
    return None if w is None else (w - 1, 0)


def _fa_extra_kwargs(inv, inputs, *, sink_arg="s_aux"):
    """Kernel kwargs serving actually passes: window and optional per-head sink."""
    kw = {}
    ws = _fa_window_size(inv)
    if ws is not None:
        kw["window_size"] = ws
    sinks = inputs.get("sinks")
    if sinks is not None:
        kw[sink_arg] = sinks
    return kw


def _fa_prefill_case(inv, inputs, varlen_func, causal, extra=None) -> BenchCase:
    qp, kp, vp, cu_q, cu_k, Sq, Lq, _B, _N, _D, _Dv = _varlen_meta(inv, inputs)
    extra = dict(extra or {})
    run = lambda: varlen_func(
        qp, kp, vp, cu_q, cu_k, Sq, Lq, causal=causal, **extra)
    return BenchCase(
        run=run,
        to_canonical=_primary,
    )


def _fa_decode_case(inputs, kvcache_func, extra=None) -> BenchCase:
    # canonical q[B,1,N,D], k/v[B,Lq,Nk,D|Dv]; FA with_kvcache is [B,seqlen,H,D].
    import torch
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    B = q.shape[0]
    cs = torch.full((B,), k.shape[1], dtype=torch.int32, device=q.device)
    extra = dict(extra or {})
    run = lambda: kvcache_func(
        q, k, v, cache_seqlens=cs, causal=True, **extra)
    return BenchCase(
        run=run,
        to_canonical=_primary,
    )


def _fa2_prefill_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    from flash_attn import flash_attn_varlen_func
    return _fa_prefill_case(
        inv, inputs, flash_attn_varlen_func, inputs["causal"],
        extra=_fa_extra_kwargs(inv, inputs))


def _fa2_decode_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    from flash_attn import flash_attn_with_kvcache
    return _fa_decode_case(
        inputs, flash_attn_with_kvcache,
        extra=_fa_extra_kwargs(inv, inputs))


def _fa3_prefill_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    from flash_attn_interface import flash_attn_varlen_func   # FA3 = separate module
    return _fa_prefill_case(
        inv, inputs, flash_attn_varlen_func, inputs["causal"],
        extra=_fa_extra_kwargs(inv, inputs))


def _fa3_decode_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    from flash_attn_interface import flash_attn_with_kvcache
    return _fa_decode_case(
        inputs, flash_attn_with_kvcache,
        extra=_fa_extra_kwargs(inv, inputs))


def _fa4_prefill_build(inv, env, inputs) -> BenchCase:
    # FlashAttention 4's published CuTeDSL package. Call with kwargs: its
    # varlen signature orders seqused_q/k before the max_seqlen arguments.
    _require_cuda()
    from flash_attn.cute import flash_attn_varlen_func
    qp, kp, vp, cu_q, cu_k, Sq, Lq, _B, _N, _D, _Dv = _varlen_meta(inv, inputs)
    causal = inputs["causal"]
    extra = _fa_extra_kwargs(inv, inputs, sink_arg="learnable_sink")
    run = lambda: flash_attn_varlen_func(
        qp, kp, vp, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=Sq, max_seqlen_k=Lq, causal=causal, **extra)
    return BenchCase(
        run=run,
        to_canonical=_primary,
    )


def _vllm_vendored_fa3_prefill_build(
    inv, env, inputs,
) -> BenchCase:
    """vLLM's sink-capable FA3 fork, distinct from the official FA3 wheel.

    The serving stack does not import Dao-FA3's `flash_attn_interface`; it
    calls `vllm.vllm_flash_attn`, whose extension accepts per-head sink logits
    as `s_aux`. Restrict this provider to sink cells: non-sink geometry is
    already measured under the official fa3 library identity.
    """
    _require_cuda()
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    qp, kp, vp, cu_q, cu_k, Sq, Lq, _B, _N, _D, _Dv = _varlen_meta(
        inv, inputs)
    extra = _fa_extra_kwargs(inv, inputs)
    if "s_aux" in extra:
        extra["s_aux"] = extra["s_aux"].to(qp.dtype)
    extra["fa_version"] = 3
    run = lambda: flash_attn_varlen_func(
        qp, kp, vp, Sq, cu_q, Lq, cu_k,
        causal=inputs["causal"], **extra)
    return BenchCase(
        run=run,
        to_canonical=_primary,
    )


#: The pages vLLM's FlashAttention backend runs: any multiple of 16
#: (`get_supported_kernel_block_sizes` -> `[MultipleOf(16)]`,
#: v1/attention/backends/flash_attn.py:355-365 at v0.32.0), of which these
#: are the ones the dense-attention sweep already measures (FlashInfer's
#: set), so declaring the backend adds no page to the sweep.
_VLLM_FA_BLOCK_SIZES = (16, 32, 64)


def _vllm_vendored_fa3_decode_build(
    inv, env, inputs,
) -> BenchCase:
    """vLLM's decode step through its vendored FA3, as serving makes it.

    At v0.32.0 decode and prefill are one entry: `flash_attn_varlen_func`
    over the PAGED cache, `seqused_k` and `block_table` instead of
    `cu_seqlens_k`, sinks as `s_aux` (v1/attention/backends/
    flash_attn.py:1399-1422). `vllm.vllm_flash_attn` exports no
    `flash_attn_with_kvcache` at that release (vllm_flash_attn/
    __init__.py:23-31), which this build imported: every gpt-oss decode
    cell raised ImportError and the canary disabled the path (2026-10-26).
    """
    _require_cuda()
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    q, k, v = inputs["q"], inputs["k"], inputs["v"]   # q[B,Sq,N,D], k/v[B,Lq,Nk,D]
    B, Sq, N, _D = q.shape
    Lq, Dv = k.shape[1], v.shape[-1]
    k_cache, v_cache, _indptr, indices, _last, _page = _paged_cache(
        inv, k, v, B, _VLLM_FA_BLOCK_SIZES)
    block_table = indices.reshape(B, -1)
    seqused_k = torch.full((B,), Lq, dtype=torch.int32, device=q.device)
    cu_seqlens_q = torch.arange(0, (B + 1) * Sq, Sq, dtype=torch.int32,
                                device=q.device)
    qp = q.reshape(B * Sq, N, -1)
    extra = _fa_extra_kwargs(inv, inputs)
    if "s_aux" in extra:
        extra["s_aux"] = extra["s_aux"].to(q.dtype)
    run = lambda: flash_attn_varlen_func(
        qp, k_cache, v_cache, Sq, cu_seqlens_q, Lq, seqused_k=seqused_k,
        block_table=block_table, causal=True, fa_version=3, **extra)
    return BenchCase(
        run=run,
        to_canonical=lambda out: _primary(out).reshape(B, Sq, N, Dv),
    )


def _sglang_vendored_fa3_prefill_build(
    inv, env, inputs,
) -> BenchCase:
    """sgl-kernel's sink-capable FA3 fork (`sinks`, not official FA3's API)."""
    _require_cuda()
    from sgl_kernel.flash_attn import flash_attn_varlen_func
    qp, kp, vp, cu_q, cu_k, Sq, Lq, _B, _N, _D, _Dv = _varlen_meta(
        inv, inputs)
    extra = _fa_extra_kwargs(inv, inputs, sink_arg="sinks")
    if "sinks" in extra:
        extra["sinks"] = extra["sinks"].to(qp.dtype)
    extra["ver"] = 3
    run = lambda: flash_attn_varlen_func(
        qp, kp, vp, cu_q, cu_k, Sq, Lq,
        causal=inputs["causal"], **extra)
    return BenchCase(
        run=run,
        to_canonical=_primary,
    )


def _sglang_vendored_fa3_decode_build(
    inv, env, inputs,
) -> BenchCase:
    _require_cuda()
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    extra = _fa_extra_kwargs(inv, inputs, sink_arg="sinks")
    if "sinks" in extra:
        extra["sinks"] = extra["sinks"].to(inputs["q"].dtype)
    extra["ver"] = 3
    return _fa_decode_case(inputs, flash_attn_with_kvcache, extra=extra)


# The element types each FlashAttention build runs, from its own entry
# checks. FA2 (flash-attn 2.8.3 csrc/flash_attn/flash_api.cpp:373-375, the
# same pair in every entry point through `mha_fwd_kvcache` :1233-1235):
# "FlashAttention only support fp16 and bf16 data type", then "query and key
# must have the same dtype" — so the cache is held to the same two. The FA3
# hopper build's check admits e4m3 as well ("FlashAttention only supports
# fp16, bf16, and fp8_e4m3 data type", flash_attn_3/_C.abi3.so on the H100).
# Undeclared, every Gemma-2 cell — its checkpoint and cache are fp32 — was
# attempted and refused: ~3,000 failed builds per H100 pass.
_FA2_DTYPES = dict(activation=("bf16", "fp16"),
                   dtype_slots=(("kv", ("bf16", "fp16")),))
_FA3_DTYPES = dict(activation=("bf16", "fp16", "fp8_e4m3"),
                   dtype_slots=(("kv", ("bf16", "fp16", "fp8_e4m3")),),
                   kv_format=_PLAIN_OR_PER_TENSOR)
# The vendored FA3 forks (vLLM's, sglang's): the same hopper dtype check for
# the query, and an FP8 cache read through per-tensor descales.
_VENDORED_FA3_DTYPES = dict(activation=("bf16", "fp16"),
                            dtype_slots=(("kv", ("bf16", "fp16", "fp8_e4m3")),))


register(Provider(
    library="fa2", signature="gqa_attention",
    kernel_impl="flash_attention_v2", import_name="flash_attn",
    build=_fa2_prefill_build,
    # `d_head <= 256`, from the library's own refusal ("FlashAttention forward
    # only supports head dimension at most 256"). Undeclared, every MLA-shaped
    # cell at d_head 512 was attempted and refused: 332 failed builds in one
    # pass.
    # ...and one width for K and V: FA2 checks V against `head_size`
    # ("v must have shape (total_k, num_heads_k, head_size)", "vcache must
    # have shape (batch_size_c, seqlen_k, num_heads_k, head_size_og)" —
    # flash_api.cpp `mha_varlen_fwd` / `mha_fwd_kvcache`), so a cell with a
    # narrower V (`d_v_head`) is refused by the library: 1,947 failed builds
    # on MiMo-V2-Flash in one pass.
    serves=KernelSignature(signature="gqa_attention", phases=("prefill",),
                           max_dims=(("d_head", 256),),
                           # The window it takes (`window_size`); no sink
                           # argument, and one head width for K and V.
                           variant_dims=("s_window",),
                           **_FA2_DTYPES,
                           # Contiguous KV: this build allocates no page
                           # table, so it serves no paged layout.
                           kv_block_sizes=()),
    launch_inputs=_ATTN_VARLEN_LAUNCH,
    launch_output="q",
))
register(Provider(
    library="fa2", signature="gqa_attention",
    kernel_impl="flash_attention_v2_kvcache", import_name="flash_attn",
    build=_fa2_decode_build,
    # `d_head <= 256`, from the library's own refusal ("FlashAttention forward
    # only supports head dimension at most 256"). Undeclared, every MLA-shaped
    # cell at d_head 512 was attempted and refused: 332 failed builds in one
    # pass.
    # ...and one width for K and V: FA2 checks V against `head_size`
    # ("v must have shape (total_k, num_heads_k, head_size)", "vcache must
    # have shape (batch_size_c, seqlen_k, num_heads_k, head_size_og)" —
    # flash_api.cpp `mha_varlen_fwd` / `mha_fwd_kvcache`), so a cell with a
    # narrower V (`d_v_head`) is refused by the library: 1,947 failed builds
    # on MiMo-V2-Flash in one pass.
    serves=KernelSignature(signature="gqa_attention", phases=("decode",),
                           max_dims=(("d_head", 256),),
                           # The window it takes (`window_size`); no sink
                           # argument, and one head width for K and V.
                           variant_dims=("s_window",),
                           **_FA2_DTYPES,
                           # Contiguous KV: this build allocates no page
                           # table, so it serves no paged layout.
                           kv_block_sizes=()),
))
register(Provider(
    library="fa3", signature="gqa_attention",
    kernel_impl="flash_attention_v3", import_name="flash_attn_interface",
    build=_fa3_prefill_build,
    # `d_head <= 192` — THIS hopper build's ceiling, quoted from its own
    # refusal (flash_api_stable.cpp: "at most 192"). Lower than fa2's, and
    # undeclared it cost 1,904 failed builds in one pass, at d_head 256 and 512.
    serves=KernelSignature(signature="gqa_attention", phases=("prefill",),
                           max_dims=(("d_head", 192),),
                           # `window_size` and a narrower V; the official
                           # export takes no sink.
                           variant_dims=("s_window", "d_v_head"),
                           **_FA3_DTYPES,
                           # Contiguous KV: this build allocates no page
                           # table, so it serves no paged layout.
                           kv_block_sizes=()),
    launch_inputs=_ATTN_VARLEN_LAUNCH,
    launch_output="q",
))
register(Provider(
    library="fa3", signature="gqa_attention",
    kernel_impl="flash_attention_v3_kvcache", import_name="flash_attn_interface",
    build=_fa3_decode_build,
    # `d_head <= 192` — THIS hopper build's ceiling, quoted from its own
    # refusal (flash_api_stable.cpp: "at most 192"). Lower than fa2's, and
    # undeclared it cost 1,904 failed builds in one pass, at d_head 256 and 512.
    serves=KernelSignature(signature="gqa_attention", phases=("decode",),
                           max_dims=(("d_head", 192),),
                           # `window_size` and a narrower V; the official
                           # export takes no sink.
                           variant_dims=("s_window", "d_v_head"),
                           **_FA3_DTYPES,
                           # Contiguous KV: this build allocates no page
                           # table, so it serves no paged layout.
                           kv_block_sizes=()),
))
register(Provider(
    library="fa4", signature="gqa_attention",
    # The gate is `flash_attn.cute`, not `sglang`. sglang's wrapper imports
    # fine on its own — what it needs is FlashAttention 4's CUTE frontend, and
    # THAT is what is missing here (`cutlass.cute.core has no attribute
    # ThrMma`, a cutlass-dsl version mismatch). Declared as `sglang`, the
    # availability check passed and every cell failed one at a time: 1,436 in a
    # single pass. The import to declare is the one the build actually needs.
    kernel_impl="flash_attention_v4", import_name="flash_attn.cute",
    build=_fa4_prefill_build,
    serves=KernelSignature(
        signature="gqa_attention",
        phases=("prefill",),
        unsupported_features=("sm90_page_table",),
        # Contiguous KV: this build allocates no page table.
        kv_block_sizes=(),
        # `window_size`, `learnable_sink`, and a V of its own width
        # (measured on the H100 with each; the 4090's sm89 path asserts
        # "Learnable sink is not supported").
        variant_dims=("s_window", "n_attn_sinks", "d_v_head"),
        max_dims=(("d_head", 256), ("d_v_head", 256)),
        requires_divisible_dims=(("n_heads", "n_kv_heads"),),
        # The library's own refusal (flash-attn-4 4.0.0b36,
        # flash_attn/cute/interface.py:801-802, the H100's build): "inputs
        # must be float16, bfloat16, fp8 e4m3fn, or fp8 e5m2", asserted on V
        # after Q/K are held to V's dtype — so the cache is held to the same
        # four. Undeclared, every Gemma-2 cell (fp32 checkpoint and cache)
        # reached the GPU and its canary disabled the path on every pass.
        # The same assertion holds q and k to v's dtype (:805-807), which a
        # per-slot list cannot say: a cell whose query and cache differ is
        # admitted here and refused by the library.
        activation=("bf16", "fp16", "fp8_e4m3", "fp8_e5m2"),
        dtype_slots=(("kv", ("bf16", "fp16", "fp8_e4m3", "fp8_e5m2")),),
        # FA4's FP8 path takes q/k/v_descale per (batch, kv head)
        # (flash_attn/cute/interface.py `_validate_tensor(..., (batch_size,
        # num_head_kv))`), which serving fills from one scalar.
        kv_format=_PLAIN_OR_PER_TENSOR,
    ),
    launch_inputs=_ATTN_VARLEN_LAUNCH,
    launch_output="q",
))

# The floor is per framework, because their dispatch differs. vLLM runs its
# vendored FA3 on Hopper only (`get_flash_attn_version`, v1/attention/
# backends/fa_utils.py:99 at v0.32.0: `major == 9`, else FA2) and refuses a
# sink below sm90 outright (flash_attn.py:469-470, "sink not supported on
# compute capability < 9.0") — and every cell these providers serve carries a
# sink — so on Ada they would time a launch vLLM never makes. sglang's
# registry allows FA3 on sm8x for non-MLA models (srt/layers/attention/
# attention_registry.py:248 at v0.5.22), so its pair has no floor.
# The decode pages: vLLM's build reads a paged cache, sglang's (sgl-kernel
# `flash_attn_with_kvcache`) the contiguous one.
#
# Both take `window_size` beside the sink. A V of its own width is sglang's
# only: vLLM's fork refuses it with a sink ("We don't support S aux with
# hdim != hdim_v", hopper/flash_api.cpp:1248 — 226 failed builds), while
# sglang's measured it on the H100 (its sm8x path says "Only Hopper supports
# different V headdim").
for _lib, _prefix, _prefill, _decode, _min_sm, _pages, _variants in (
    ("vllm", "vllm_flash_attention_v3",
     _vllm_vendored_fa3_prefill_build, _vllm_vendored_fa3_decode_build, 90,
     _VLLM_FA_BLOCK_SIZES, ("s_window",)),
    ("sglang", "sglang_flash_attention_v3",
     _sglang_vendored_fa3_prefill_build, _sglang_vendored_fa3_decode_build,
     None, (), ("s_window", "d_v_head")),
):
    register(Provider(
        library=_lib,
        signature="gqa_attention",
        kernel_impl=_prefix,
        import_name=_lib,
        min_sm=_min_sm,
        build=_prefill,
        serves=KernelSignature(
            signature="gqa_attention",
            phases=("prefill",),
            requires_dims=("n_attn_sinks",),
            variant_dims=_variants,
            **_VENDORED_FA3_DTYPES,
            # Contiguous KV: this build allocates no page table.
            kv_block_sizes=(),
            max_dims=(("d_head", 256),),
            kv_format=_PLAIN_OR_PER_TENSOR,
        ),
        launch_inputs=_ATTN_VARLEN_LAUNCH,
        launch_output="q",
    ))
    register(Provider(
        library=_lib,
        signature="gqa_attention",
        kernel_impl=_prefix + "_kvcache",
        import_name=_lib,
        min_sm=_min_sm,
        build=_decode,
        serves=KernelSignature(
            signature="gqa_attention",
            phases=("decode",),
            requires_dims=("n_attn_sinks",),
            variant_dims=_variants,
            **_VENDORED_FA3_DTYPES,
            kv_block_sizes=_pages,
            max_dims=(("d_head", 256),),
            kv_format=_PLAIN_OR_PER_TENSOR,
        ),
        # A paged build's page is part of what it ran.
        impl_params=((lambda inv, pages=_pages:
                      f"_b{resolve_kv_block_size(inv, pages)}")
                     if _pages else None),
    ))


# ---------------------------------------------------------------------------
# FlashInfer — dense attention (prefill + decode via paged wrapper).
# First slice: prefill ragged. Decode wrapper is a follow-on.
# ---------------------------------------------------------------------------

# What FlashInfer's dense GQA wrappers (ragged prefill, paged decode, paged
# verify) run, declared once for the three.
#
# Query dtype fp16/bf16. The JIT names each module by its dtypes through
# `filename_safe_dtype_map` (flashinfer 0.7.2 jit/utils.py:69-80), which has
# no float32 — an fp32 cell dies there with `KeyError: torch.float32` — and
# vLLM's FlashInfer backend says the same (`supported_dtypes = [float16,
# bfloat16]`, v1/attention/backends/flashinfer.py:447 at v0.32.0). Gemma-2's
# fp32 cells cost ~2,300 ragged + ~6,600 paged failed builds per H100 pass.
#
# V head width in {64, 128, 256, 512}: `DISPATCH_HEAD_DIM` (include/
# flashinfer/utils.cuh:200-226, "Unsupported head_dim") is what the split-KV
# state merge dispatches V's width through, and it is the list vLLM
# republishes as the backend's head sizes (flashinfer.py:566-568). A width
# outside it runs only while the scheduler happens not to split KV, and the
# prefill template has a second wall at 192: a ragged prefill longer than 64
# packed query rows takes CTA_TILE_Q=128 (utils.cuh `FA2DetermineCtaTileQ`),
# where 192 hits `KernelTraits::IsInvalid`'s register clause
# (attention/prefill.cuh:385-389, NUM_MMA_Q·(8·12 + 2·4·4) = 256) and
# raises "Invalid configuration". ~350 ragged and ~540 paged failed builds
# per 4090 pass, all at a 192-wide V. Read off V's layout, so MiMo's
# 192-QK/128-V cells — V 128 wide — stay served.
# None of the three serves an attention sink (`n_attn_sinks`), which is what
# serving says too: on SM90 vLLM v0.32.0 runs a sink model through another
# wrapper, `BatchAttentionWithAttentionSinkWrapper` over the paged cache
# (v1/attention/backends/flashinfer.py:1283-1295). In the H100's flashinfer
# 0.7.2.post1 the ragged wrapper's `run` takes no `sinks` (prefill.py:
# 5881-5900; every gpt-oss cell raised), and the paged decode and paged
# prefill wrappers take the argument and DROP it on SM90's backends:
# measured on the H100 (fa2 / CUDA-core decode, fa3 paged prefill), the
# output with `sinks=` equals the output without, to the bit. The cross-
# check caught it as a scale error — typical 0.012 / 0.024 against the bf16
# band's 0.008, cosine 0.9998 / 1.0000 — and the canaries disabled both
# decode paths (2026-10-26). A gap in the kernel, not in the tolerance.
# So none of the three lists the sink among the axes it serves.

_FLASHINFER_SERVES = dict(
    activation=("bf16", "fp16"),
    dtype_slots=(("kv", ("bf16", "fp16", "fp8_e4m3")),),
    operand_enum=(("v", -1, (64, 128, 256, 512)),),
    # `kv_data_type` is the cache tensor's dtype and the dequant scale the
    # `k_scale`/`v_scale` scalars; the NVFP4 cache vLLM also routes here is
    # a packed layout with block scales these builds never allocate.
    kv_format=_PLAIN_OR_PER_TENSOR,
)


def _flashinfer_build(inv, env, inputs) -> BenchCase:
    _require_cuda()
    import torch
    import flashinfer  # lazy

    qp, kp, vp, cu_q, cu_k, _Sq, _Lq, _B, N, D, Dv = _varlen_meta(inv, inputs)
    Nk = kp.shape[1]
    wbuf = torch.empty(_FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=qp.device)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(wbuf, "NHD")
    # plan() defaults q_data_type to fp16; our canonical inputs are the
    # invocation's dtype (bf16) — pin it so run() doesn't reject the qkv dtype.
    # `head_dim_vo` is V's own width. Left out, `plan` takes it to be D
    # (prefill.py:4272, `BatchPrefillWithRaggedKVCacheWrapper.plan`), and an
    # asymmetric-V cell (MiMo's 192/128) is planned at 192 for both — its
    # split-KV merge then refuses ("VariableLengthMergeStates ...
    # Unsupported head_dim: 192") on a V that was never 192 wide.
    plan_kw = dict(q_data_type=qp.dtype, kv_data_type=kp.dtype,
                   head_dim_vo=Dv)
    ws = _fa_window_size(inv)
    if ws is not None:
        plan_kw["window_left"] = ws[0]   # W-1; -1 is the library's full-attn default
    wrapper.plan(cu_q, cu_k, N, Nk, D, causal=inputs["causal"], **plan_kw)
    return BenchCase(run=lambda: wrapper.run(qp, kp, vp))


register(Provider(
    library="flashinfer",
    signature="gqa_attention",
    kernel_impl="flashinfer_ragged_prefill",
    import_name="flashinfer",
    serves=KernelSignature(signature="gqa_attention", phases=("prefill",),
                           # Ragged/varlen buffers, not a page table.
                           kv_block_sizes=(),
                           # `window_left` and `head_dim_vo`; no sink
                           # (see `_FLASHINFER_SERVES`).
                           variant_dims=("s_window", "d_v_head"),
                           **_FLASHINFER_SERVES),
    build=_flashinfer_build,
    launch_inputs=_ATTN_VARLEN_LAUNCH,
    launch_output="q",
))


# Paged-KV block sizes each backend serves, mirrored from the backends' own
# `get_supported_kernel_block_sizes()` in vLLM v1/attention/backends/. These are
# BACKEND properties; the deployment's choice among them is an axis of the
# cell (`Invocation.kv_block_size`, read off its `dims`), with no default.
_FLASHINFER_BLOCK_SIZES = (16, 32, 64)          # flashinfer.py

#: FlashInfer's workspace, at vLLM's size (`FLASHINFER_WORKSPACE_BUFFER_SIZE`,
#: v1/attention/backends/flashinfer.py). At 128 MiB the paged prefill
#: wrapper faulted on a 10k-request verify step under sglang's FlashInfer.
_FLASHINFER_WORKSPACE_BYTES = 256 * 1024 * 1024
_FLASHINFER_MLA_BLOCK_SIZES = (32, 64)          # mla/flashinfer_mla.py
_FLASHMLA_BLOCK_SIZES = (64,)                   # mla/flashmla.py
_FLASHMLA_SPARSE_BLOCK_SIZES = (64,)            # mla/flashmla_sparse.py
#: FlashMLA's sparse kernels exist for Hopper and Blackwell only: the
#: library's sparse entry points refuse any other card at the first call
#: ("Sparse Attention Forward Kernel is only supported on SM90a and SM100f
#: architectures", FlashMLA csrc/api/sparse_prefill.cpp — the error the
#: RTX 4090's vLLM and sglang sink rows recorded). One fact for every
#: launch that reaches them — FlashMLA's own and the copies vLLM and sglang
#: vendor.
_FLASHMLA_SPARSE_MIN_SM = 90
_CUTLASS_MLA_BLOCK_SIZES = (128,)               # mla/cutlass_mla.py


def resolve_kv_block_size(inv: Invocation, supported) -> int:
    """The paged-KV block size to run at: the DEPLOYMENT's choice, checked
    against what this backend supports.

    The two halves belong in different places. Which sizes a kernel can serve is
    the backend's own property — vLLM asks each one via
    `get_supported_kernel_block_sizes()`, and the answers differ (FlashInfer
    dense [16,32,64], FlashInfer MLA [32,64], FlashMLA [64], CUTLASS MLA [128]).
    WHICH of them runs is the serving configuration, and it has to arrive with
    the invocation; a provider that picks for itself measures a paging layout
    nobody asked for, and the row cannot say which one it was.

    The reconciliation mirrors vLLM's own (`v1/worker/utils.py`): the requested
    size is the KV-MANAGER's block, and a kernel may run at a smaller block that
    DIVIDES it — a manager block splits into whole kernel blocks. So the rule is
    "the requested size if the backend serves it, else the largest supported size
    that divides it", and vLLM raises when neither holds. Here that raise becomes
    a decline: the driver records a skip with the reason rather than quietly
    substituting a size nobody asked for — the same rule the shape path follows,
    one layer down.
    """
    want = inv.kv_block_size
    if want is None:
        raise NotImplementedError(
            "no kv_block_size on this job: the paged-KV page is the "
            "deployment's choice and it has to arrive with the invocation. "
            "A framework that declares none (sglang resolves its page per "
            "attention backend) needs the request to name one; measuring at "
            "a page nobody chose is a row that cannot say which page it is.")
    want = int(want)
    if want in supported:
        return want
    for size in sorted(supported, reverse=True):
        if want % size == 0:
            return size
    raise NotImplementedError(
        f"kv_block_size={want} has no common block with this backend, which "
        f"serves {sorted(supported)} (mirrors vLLM's "
        f"get_supported_kernel_block_sizes + block-size reconciliation)")


def _flashinfer_decode_build(inv, env, inputs) -> BenchCase:
    """FlashInfer paged decode (BatchDecodeWithPagedKVCacheWrapper). Adapt
    canonical q[B,1,N,D] + k/v[B,Lq,Nk,D] to FlashInfer's paged KV layout at the
    deployment's block size. API from vLLM v1/attention/backends/flashinfer.py +
    sglang flashinfer_backend. plan()/run() are the modern names
    (begin_forward/forward are aliases).

    The cache is laid out as [num_pages, page, Nk, D] with each sequence's pages
    contiguous, and the last page partially filled when the context is not a
    whole multiple — which is the ordinary state of a real paged cache, not an
    edge case."""
    _require_cuda()
    import torch
    import flashinfer

    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    # q is already packed [B·Sq, N, D]; k/v stay canonical for paging.
    B = int(inv.dims["B"])
    N, D = q.shape[1], q.shape[2]
    Nk = k.shape[2]
    dev, dt = q.device, k.dtype          # the cache's dtype, not the query's
    qp = q
    k_cache, v_cache, indptr, indices, last_page_len, page = (
        _paged_cache(inv, k, v, B, _FLASHINFER_BLOCK_SIZES))

    wbuf = torch.empty(_FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        wbuf, "NHD", use_tensor_cores=_fi_decode_tensor_cores(inv))
    plan_kw = dict(pos_encoding_mode="NONE", q_data_type=dt, kv_data_type=dt)
    ws = _fa_window_size(inv)
    if ws is not None:
        plan_kw["window_left"] = ws[0]
    wrapper.plan(indptr, indices, last_page_len, N, Nk, D, page, **plan_kw)
    return BenchCase(run=lambda: wrapper.run(qp, (k_cache, v_cache)))


def _paged_cache(inv, k, v, B: int, served):
    """k/v [B,Lq,Nk,D|Dv] laid out as a paged cache `[pages, page, Nk, D]`
    at the deployment's page, resolved against the `served` pages:
    (k_cache, v_cache, indptr, indices, last_page_len, page). Each
    sequence's pages contiguous, the last one partial; `indices` reshaped
    `[B, pages per sequence]` is the block table.

    A V narrower than K is stored as serving stores it: one pool whose
    token rows hold K then V, `[pages, page, Nk, D + Dv]`, split along the
    last axis into two views (vLLM's asymmetric-V cache, v1/attention/
    backends/flash_attn_diffkv.py:104-106, 138, 193 at v0.32.0 — the
    backend model_executor/models/mimo_v2.py:300-321 sends every V narrower
    than K to). The two views
    share one page stride and one token stride, which FlashInfer's SM90
    paged prefill requires of K and V ("K and V must have same page stride
    for sparse attention", data/csrc/batch_prefill_sm90.cu:233-237 at
    0.7.2.post1): two separately allocated caches of different widths were
    refused on every MiMo verify cell (2026-10-28)."""
    import torch
    Lq, Nk, D = k.shape[1], k.shape[2], k.shape[3]
    Dv = v.shape[-1]
    dev, dt = k.device, k.dtype
    page = resolve_kv_block_size(inv, served)
    pages_per_seq = (Lq + page - 1) // page
    num_pages = B * pages_per_seq
    padded = pages_per_seq * page
    indptr = torch.arange(0, (B + 1) * pages_per_seq, pages_per_seq,
                          dtype=torch.int32, device=dev)
    indices = torch.arange(num_pages, dtype=torch.int32, device=dev)
    tail = Lq - (pages_per_seq - 1) * page              # 1..page
    last_page_len = torch.full((B,), tail, dtype=torch.int32, device=dev)
    if Dv != D:
        pool = torch.zeros(B, padded, Nk, D + Dv, dtype=dt, device=dev)
        pool[:, :Lq, :, :D] = k
        pool[:, :Lq, :, D:] = v
        pool = pool.reshape(num_pages, page, Nk, D + Dv)
        return (pool[..., :D], pool[..., D:],
                indptr, indices, last_page_len, page)
    k_cache = torch.zeros(B, padded, Nk, D, dtype=dt, device=dev)
    v_cache = torch.zeros(B, padded, Nk, Dv, dtype=dt, device=dev)
    k_cache[:, :Lq] = k
    v_cache[:, :Lq] = v
    return (k_cache.reshape(num_pages, page, Nk, D).contiguous(),
            v_cache.reshape(num_pages, page, Nk, Dv).contiguous(),
            indptr, indices, last_page_len, page)


def _flashinfer_verify_build(inv, env, inputs) -> BenchCase:
    """A decode step with several query tokens per request (speculative
    verify, `s_query` > 1) over the paged cache. FlashInfer's decode wrapper
    takes one query token per request, so vLLM's FlashInfer backend sends any
    longer query to BatchPrefillWithPagedKVCacheWrapper: causal, the queries
    being the last `s_query` positions of each context."""
    _require_cuda()
    import torch
    import flashinfer

    q, k, v = inputs["q"], inputs["k"], inputs["v"]   # q packed [B·Sq,N,D]
    B, Sq = int(inv.dims["B"]), int(inv.dims["s_query"])
    N, D = q.shape[1], q.shape[2]
    Nk, Dv = k.shape[2], v.shape[-1]
    dev, dt = q.device, k.dtype
    k_cache, v_cache, indptr, indices, last_page_len, page = (
        _paged_cache(inv, k, v, B, _FLASHINFER_BLOCK_SIZES))
    qo_indptr = torch.arange(0, (B + 1) * Sq, Sq, dtype=torch.int32, device=dev)
    wbuf = torch.empty(_FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(wbuf, "NHD")
    plan_kw = dict(causal=True, pos_encoding_mode="NONE", q_data_type=q.dtype,
                   kv_data_type=dt, head_dim_vo=Dv)
    ws = _fa_window_size(inv)
    if ws is not None:
        plan_kw["window_left"] = ws[0]
    wrapper.plan(qo_indptr, indptr, indices, last_page_len, N, Nk, D, page,
                 **plan_kw)
    return BenchCase(run=lambda: wrapper.run(q, (k_cache, v_cache)))


def _one_query_token(inv: Invocation) -> None:
    if int(inv.dims.get("s_query", 1)) != 1:
        raise NotImplementedError(
            "FlashInfer's decode wrapper takes one query token per request; "
            "a verify step goes to the paged prefill wrapper "
            "(flashinfer_paged_verify)")


def _several_query_tokens(inv: Invocation) -> None:
    if int(inv.dims.get("s_query", 1)) == 1:
        raise NotImplementedError(
            "one query token per request is the decode wrapper's "
            "(flashinfer_paged_decode)")


def _fi_decode_tensor_cores(inv: Invocation) -> bool:
    """FlashInfer's decode wrapper runs a CUDA-core kernel unless told to use
    the tensor-core (prefill-style) one, and the CUDA-core kernel is ALU-bound
    once many query heads share a KV head. vLLM's FlashInfer backend switches
    above 4 query heads per KV head (`num_qo_heads // num_kv_heads > 4`), which
    is what a deployment runs: without it GQA 16:1 decode on an H100 read 3.4x
    off the bandwidth floor while FA3 sat on it."""
    return int(inv.dims["n_heads"]) // int(inv.dims["n_kv_heads"]) > 4


register(Provider(
    library="flashinfer",
    signature="gqa_attention",
    kernel_impl="flashinfer_paged_decode",
    import_name="flashinfer",
    # One head width for Q·K and for V. The decode wrapper's `plan` takes a
    # single `head_dim` and compiles both kernels with it as head_dim_qk AND
    # head_dim_vo (flashinfer 0.7.2 `decode.py:1841`, `:2387-2388`,
    # `:2403-2404`; the CUDA-core template says so itself: "only supports
    # equal head dim for QK and VO"). A narrower V (`d_v_head`, present only
    # when it differs) is therefore read 192 wide out of a 128-wide cache:
    # the tensor-core path faults with an illegal address that poisons the
    # context (MiMo-V2-Flash, 22 cells quarantined on a 4090), the CUDA-core
    # path refuses (2,004 failed builds). The paged PREFILL wrapper takes
    # `head_dim_vo` and serves the verify step below.
    serves=KernelSignature(signature="gqa_attention", phases=("decode",),
                           # `window_left`; one width for V (above), and no
                           # sink (see `_FLASHINFER_SERVES`).
                           variant_dims=("s_window",),
                           kv_block_sizes=_FLASHINFER_BLOCK_SIZES,
                           **_FLASHINFER_SERVES),
    build=_flashinfer_decode_build,
    impl_params=lambda inv: (
        _one_query_token(inv)
        or f"_b{resolve_kv_block_size(inv, _FLASHINFER_BLOCK_SIZES)}"
        + ("_tc" if _fi_decode_tensor_cores(inv) else "")),
    launch_inputs=(("q", _PACK_BATCH_SEQ),),
    launch_output="q",
))
register(Provider(
    library="flashinfer",
    signature="gqa_attention",
    kernel_impl="flashinfer_paged_verify",
    import_name="flashinfer",
    serves=KernelSignature(signature="gqa_attention", phases=("decode",),
                           kv_block_sizes=_FLASHINFER_BLOCK_SIZES,
                           # The paged prefill wrapper: `window_left` and
                           # `head_dim_vo`; no sink (above).
                           variant_dims=("s_window", "d_v_head"),
                           **_FLASHINFER_SERVES),
    build=_flashinfer_verify_build,
    impl_params=lambda inv: (
        _several_query_tokens(inv)
        or f"_b{resolve_kv_block_size(inv, _FLASHINFER_BLOCK_SIZES)}"),
    launch_inputs=(("q", _PACK_BATCH_SEQ),),
    launch_output="q",
))


# ---------------------------------------------------------------------------
# FlashMLA — MLA decode specialist (DeepSeek-style absorbed latent).
# ---------------------------------------------------------------------------

# FlashMLA serves exactly one block size, so `resolve_kv_block_size` either
# returns 64 or declines — the deployment's value is still what asks.


def _flashmla_build(inv, env, inputs) -> BenchCase:
    """FlashMLA absorbed-decode kernel. Adapts the canonical MLA inputs
    (q[B,Sq,N,Dqk], single latent c[B,Lq,Dqk]) into FlashMLA's paged latent-KV
    layout. API from vLLM v1/attention/backends/mla/flashmla.py + sglang
    flashmla_backend.py:
        get_mla_metadata(cache_seqlens, Sq*N // num_heads_k=1, num_heads_k=1)
        flash_mla_with_kvcache(q, k_cache[num_blocks,page,1,Dqk], block_table,
            cache_seqlens, head_dim_v=Dv, tile_scheduler_metadata, num_splits,
            softmax_scale, causal) -> (out[B,Sq,N,Dv], lse)
    """
    _require_cuda()
    import torch
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    q, c = inputs["q"], inputs["c"]              # [B,Sq,N,Dqk], [B,Lq,Dqk]
    B, Sq, N, Dqk = q.shape
    Lq = c.shape[1]
    Dv = int(inputs["Dv"])
    dev = q.device

    # Pack the per-batch latent [B,Lq,Dqk] into a paged cache
    # [num_blocks, page, 1, Dqk] with a contiguous block_table per sequence.
    page = resolve_kv_block_size(inv, _FLASHMLA_BLOCK_SIZES)
    blocks_per_seq = (Lq + page - 1) // page
    num_blocks = B * blocks_per_seq
    k_cache = torch.zeros(num_blocks, page, 1, Dqk, dtype=c.dtype, device=dev)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        B, blocks_per_seq)
    padded = blocks_per_seq * page
    c_pad = torch.zeros(B, padded, Dqk, dtype=c.dtype, device=dev)
    c_pad[:, :Lq] = c
    k_cache.copy_(c_pad.reshape(B, blocks_per_seq, page, 1, Dqk)
                  .reshape(num_blocks, page, 1, Dqk))
    cache_seqlens = torch.full((B,), Lq, dtype=torch.int32, device=dev)

    md, splits = get_mla_metadata(cache_seqlens, Sq * N // 1, 1)
    scale = 1.0 / (Dqk ** 0.5)
    return BenchCase(
        run=lambda: flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, Dv, md, splits,
            softmax_scale=scale, causal=bool(inputs["causal"])),
        to_canonical=_primary,
    )


register(Provider(
    library="flashmla",
    signature="mla_attention",
    kernel_impl="flashmla_decode",
    import_name="flash_mla",
    serves=KernelSignature(signature="mla_attention", phases=("decode",),
                           kv_block_sizes=_FLASHMLA_BLOCK_SIZES,
                           # `flash_mla_with_kvcache` over a cache at the
                           # latent's own dtype, no descale: the bf16 path.
                           # vLLM's FP8 cache is another entry point
                           # (`flash_mla_with_kvcache_fp8` with descale_q /
                           # descale_k, v1/attention/backends/mla/
                           # flashmla.py:311-323), which this build never
                           # calls (`vllm_flashmla_decode_fp8` does).
                           kv_format=(PLAIN,),
                           # The bf16/fp16 kernel, the cache at the query's
                           # dtype (4,354 bf16 rows).
                           activation=("bf16", "fp16"),
                           dtype_slots=(("kv", ("bf16", "fp16")),)),
    build=_flashmla_build,
    impl_params=lambda inv: f"_b{resolve_kv_block_size(inv, _FLASHMLA_BLOCK_SIZES)}",
))


# The dense MLA decode over an FP8 latent cache, as vLLM v0.32.0 runs it on
# Hopper. With `--kv-cache-dtype fp8` / `fp8_e4m3` the SM90 priority list
# (platforms/cuda.py:146-152) passes FLASH_ATTN_MLA (bf16/fp16 caches only,
# mla/flashattn_mla.py:43-47) and lands on FLASHMLA, whose decode takes
# `flash_mla_with_kvcache_fp8` (mla/flashmla.py:311-323): the vllm-project
# FlashMLA fork's SM90 dense FP8 extension (`fwd_kvcache_mla_fp8`,
# csrc/extension/sm90/dense_fp8/pybind.cpp at 0eee43b, the pin in
# cmake/external_projects/flashmla.cmake). What it reads:
#   * the cache: the latent row as `concat_and_cache_mla` wrote it, `x /
#     k_scale` cast to e4m3, 576 bytes a token — no per-group scale, no
#     bf16 tail — paged [num_blocks, 64, 1, 576] (the kernel refuses any
#     page but 64, and any d_head / d_v_head but 576 / 512);
#   * the query: e4m3 too ("query and key must have the same dtype"),
#     quantised in the launch before (`_decode_concat_quant_fp8_op` with the
#     layer's static `_q_scale`, mla_attention.py:1090-1095);
#   * `descale_q` / `descale_k`: each a [1] fp32, the layer's `_q_scale` /
#     `_k_scale` — one scale a tensor, the `fp8_per_tensor` record.
# The output is bf16 whatever the query was. sglang's FlashMLA backend
# (opt-in; its Hopper MLA default is FA3) calls the same extension from the
# sgl-project fork (flashmla_backend.py:454-488, q quantised at the layer's
# k_scale), so one provider measures the kernel both run.
def _fp8_per_tensor(x):
    """`x` on the `fp8_per_tensor` grid — e4m3 with one fp32 scale for the
    whole tensor (`quant.quantise_groups`, group 0) — and that scale as the
    [1] fp32 tensor the kernel's descale operands are."""
    from ..quant import FORMATS, quantise_groups
    q, scale = quantise_groups(x, FORMATS["fp8_e4m3"], 0, "exact", x.dtype)
    return q.reshape(x.shape), scale.reshape(-1)[:1].contiguous()


def _flashmla_fp8_operands(inv, inputs):
    """The operands `flash_mla_with_kvcache_fp8` is handed, built from the
    canonical lossless q[B,Sq,N,Dqk] and latent c[B,Lq,Dqk]: the query and
    the paged latent each on its own per-tensor e4m3 grid, the block table,
    the context lengths and the two descales. Every sequence's pages are
    contiguous and its last page partly filled; the zero padding is never
    read (`cache_seqlens`)."""
    import torch
    q, c = inputs["q"], inputs["c"]
    B, _Sq, _N, Dqk = q.shape
    Lq = c.shape[1]
    page = resolve_kv_block_size(inv, _FLASHMLA_BLOCK_SIZES)
    blocks_per_seq = -(-Lq // page)
    padded = torch.zeros(B, blocks_per_seq * page, Dqk, dtype=c.dtype,
                         device=c.device)
    padded[:, :Lq] = c
    latent, descale_k = _fp8_per_tensor(padded)
    k_cache = latent.reshape(B * blocks_per_seq, page, 1, Dqk)
    q8, descale_q = _fp8_per_tensor(q)
    block_table = torch.arange(B * blocks_per_seq, dtype=torch.int32,
                               device=c.device).reshape(B, blocks_per_seq)
    cache_seqlens = torch.full((B,), Lq, dtype=torch.int32, device=c.device)
    return dict(q=q8, k_cache=k_cache, block_table=block_table,
                cache_seqlens=cache_seqlens, head_dim_v=int(inputs["Dv"]),
                softmax_scale=1.0 / (Dqk ** 0.5),
                causal=bool(inputs["causal"]),
                descale_q=descale_q, descale_k=descale_k)


def _vllm_flashmla_fp8_build(inv, env, inputs) -> BenchCase:
    """`FlashMLAImpl.forward_mqa`'s FP8 branch: the scheduler metadata from
    `get_mla_metadata_dense_fp8(seq_lens, Sq·N, 1)` (`_build_decode`), then
    the one launch."""
    _require_cuda()
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache_fp8, get_mla_metadata_dense_fp8)
    _B, Sq, N, _ = inputs["q"].shape
    ops = _flashmla_fp8_operands(inv, inputs)
    meta, splits = get_mla_metadata_dense_fp8(ops["cache_seqlens"], Sq * N, 1)
    return BenchCase(
        run=lambda: flash_mla_with_kvcache_fp8(
            **ops, tile_scheduler_metadata=meta, num_splits=splits),
        to_canonical=_primary,
    )


register(Provider(
    library="vllm",
    signature="mla_attention",
    kernel_impl="vllm_flashmla_decode_fp8",
    import_name="vllm.v1.attention.ops.flashmla",
    # The extension checks `major == 9 && minor == 0` ("Dense FP8 MLA is only
    # supported on SM90"), and vLLM gates the dense backend on the SM90
    # family (`is_flashmla_dense_supported`).
    min_sm=90,
    serves=KernelSignature(signature="mla_attention", phases=("decode",),
                           # FlashMLA backend's model dtypes; the query
                           # arrives at the activation's and is quantised
                           # before this launch.
                           activation=("bf16", "fp16"),
                           output=("bf16",),
                           dtype_slots=(("kv", ("fp8_e4m3",)),),
                           kv_format=("fp8_per_tensor",),
                           enum=(("d_head", (576,)), ("d_v_head", (512,))),
                           kv_block_sizes=_FLASHMLA_BLOCK_SIZES),
    build=_vllm_flashmla_fp8_build,
    impl_params=lambda inv: f"_b{resolve_kv_block_size(inv, _FLASHMLA_BLOCK_SIZES)}",
))


# ---------------------------------------------------------------------------
# Sparse MLA (DeepSeek DSA / NSA) — the main attention behind the lightning
# indexer. Two realizations:
#   * FlashMLA `flash_mla_sparse_fwd` — the production kernel, Hopper/Blackwell
#     only (vLLM `is_flashmla_sparse_supported` gates on capability 90/100).
#   * a torch gather baseline — no dependency, so DSA is measurable on any card
#     (Ada included) and there is always a floor beneath the fused kernel.
# ---------------------------------------------------------------------------

def _torch_sdpa_mla_prefill_build(inv: Invocation, env: Env, inputs) -> BenchCase:
    """torch SDPA over launch-layout q[B,N,Sq,Dqk] / k[B,Nk,Lq,Dqk] / v[B,Nk,Lq,Dv].
    `enable_gqa` handles Nk<N; SDPA's value width is independent of the qk
    width, which is the whole point of this geometry."""
    dt, kv_dt = _torch_dtype(inv.dtype), _torch_dtype(inv.kv_dtype)
    # The cache is allocated at the dtype the cache is IN, which is the
    # cell's own `kv` slot and not its activation (`Invocation.dtype_for`).
    return _sdpa_case(inputs["q"].to(dt), inputs["k"].to(kv_dt),
                      inputs["v"].to(kv_dt), bool(inputs["causal"]))


register(Provider(
    library="torch",
    signature="mla_attention",
    kernel_impl="torch_sdpa_mla_prefill",
    import_name="torch",
    # The non-absorbed MLA geometry (K/V per head, V narrower than Q/K) — MLA
    # prefill. `n_kv_heads` is what separates it from the absorbed latent form:
    # the absorbed geometry has no separate KV head axis at all, so its presence
    # IS the discriminator, which is why this is a `requires_dims` and not a
    # value constraint. torch SDPA takes a different value width natively, so it
    # covers this shape on any card; the fused prefill kernels (FA3's
    # `head_dim_v`) can be layered on top as the envelope.
    serves=KernelSignature(signature="mla_attention",
                           # PREFILL, as the name says: serving runs MLA
                           # decode absorbed, over the one shared latent.
                           # Unstated, this took the decode cells of a
                           # deployment sharded down to ONE head, where the
                           # absorbed and per-head geometries have the same
                           # shape and nothing else tells them apart — and
                           # measured the latent twice, as a k and a v, at
                           # ~1.9x the bytes of the launch that runs (880
                           # Kimi-K2 rows, 2026-10-20).
                           phases=("prefill",),
                           requires_dims=("n_kv_heads",),
                           # Contiguous KV: this build allocates no page
                           # table, so it serves no paged layout.
                           kv_block_sizes=(),
                           activation=TORCH_FLOAT,
                           dtype_slots=(("kv", (*TORCH_FLOAT, "fp8_e4m3")),),
                           kv_format=_PLAIN_OR_PER_TENSOR),
    build=_torch_sdpa_mla_prefill_build,
    launch_inputs=_SDPA_LAUNCH,
    launch_output="q",
))


def _torch_sparse_mla_build(inv: Invocation, env: Env, inputs) -> BenchCase:
    """Gather-then-attend baseline at the kernel's own dtype: gather the
    selected latent rows per query, score against them, softmax, apply. Same
    math as the reference (which runs in fp32 and in query blocks); this one is
    the dtype-faithful version that gets timed.

    With `sinks` (`mla_sparse_sink_attention`) each head's learned sink
    joins its softmax denominator as one more score, at the compute dtype,
    and leaves no value."""
    import torch
    dt = _torch_dtype(inv.dtype)
    q = inputs["q"].to(dt)                      # [B,Sq,N,Dqk]
    sinks = inputs.get("sinks")
    if sinks is not None:
        sinks = sinks.to(dt).view(1, -1, 1, 1)  # [1,N,1,1]
    c = inputs["c"].to(_torch_dtype(inv.kv_dtype))   # [B,Lq,Dqk], the cache's dtype
    idx = inputs["indices"]                     # [B,Sq,topk], int32
    B, Sq, N, Dqk = q.shape
    topk = idx.shape[2]
    Dv = int(inputs["Dv"])
    scale = 1.0 / (Dqk ** 0.5)

    def _attend(qb, pool, ib):
        # The cache stays in its own dtype; the selected rows are what gets
        # widened to the compute dtype (an fp8 cache cannot meet bf16 q in
        # one matmul). `pool` and `ib` are the gather's operands as views
        # built with the case, so the timed call is the gather and the two
        # dots.
        sel = torch.gather(pool, 2, ib).to(dt)
        scores = torch.einsum("bsnd,bstd->bnst", qb, sel) * scale
        if sinks is not None:
            # One learned score a head, concatenated before the softmax —
            # the stable form of FlashMLA's `out *= exp(lse) / (exp(lse) +
            # exp(sink))` — and dropped from the value sum.
            scores = torch.cat([scores, sinks.expand(
                *scores.shape[:-1], 1)], dim=-1)
        attn = scores.softmax(dim=-1)[..., :topk]
        return torch.einsum("bnst,bstd->bnsd", attn, sel[..., :Dv])

    # The single gather is [B,Sq,topk,Dqk] — 192 GiB at the 131072-query DSA
    # prefill cell, the OOM the queue logged. Past the budget the query axis
    # is walked in blocks, the way the oracle walks it, at the cell's own
    # dtype; a cell whose block covers every query keeps the one-shot launch
    # its measured rows were timed with. A selected row of a block holds:
    # its gathered latent (twice at the widening, when the cache is narrower
    # than the compute dtype), its int64 index (torch.gather wants int64;
    # the int32 input is converted a block at a time, never whole — at the
    # 1048576x4104 cell the whole int64 copy alone is 32 GiB), and its
    # score and probability per head.
    dsz = torch.empty((), dtype=dt).element_size()
    widest = c.element_size() + (0 if c.dtype == dt else dsz)
    per_query = B * topk * (Dqk * widest + 8 + 2 * N * dsz)
    blk = max(1, min(Sq, _refs.GATHER_BLOCK_BYTES // max(1, per_query)))
    if blk >= Sq:
        pool = c.unsqueeze(1).expand(B, Sq, c.shape[1], Dqk)
        ib_all = idx.long().unsqueeze(-1).expand(B, Sq, topk, Dqk)

        def _run():
            return _attend(q, pool, ib_all).transpose(1, 2)
    else:
        # Built here, so the timed call allocates nothing of its own and
        # builds no view: the output every block writes into, the int64
        # index block each reuses, and per block its operands. Device memory
        # is then the inputs, the output and one block.
        out = torch.empty(B, Sq, N, Dv, dtype=dt, device=q.device)
        widened = torch.empty(B, blk, topk, dtype=torch.long, device=q.device)
        blocks = []
        for s0 in range(0, Sq, blk):
            s1 = min(Sq, s0 + blk)
            rows = widened[:, :s1 - s0]
            blocks.append((
                q[:, s0:s1],
                c.unsqueeze(1).expand(B, s1 - s0, c.shape[1], Dqk),
                rows, idx[:, s0:s1],
                rows.unsqueeze(-1).expand(B, s1 - s0, topk, Dqk),
                out[:, s0:s1]))

        def _run():
            for qb, pool, rows, source, ib, dest in blocks:
                rows.copy_(source)
                dest.copy_(_attend(qb, pool, ib).transpose(1, 2))
            return out

    return BenchCase(run=_run, to_canonical=lambda out: out.contiguous())


#: The top-k width each FlashMLA sparse entry point walks its selection in.
#: The sparse PREFILL kernel (`flash_mla_sparse_fwd`, which both frameworks
#: run for a bf16 cache) asserts `topk % (2*B_TOPK) == 0` with B_TOPK 64 on
#: SM90 (csrc/kernels/sm90/prefill/sparse/phase1.cuh:577, config.h:27) and
#: `topk % 128` on SM100 (sm100/prefill/sparse/fwd/head128/phase1.cuh:638);
#: the sparse DECODE kernel (`flash_mla_with_kvcache(..., indices=)`, the
#: fp8 cache's) asserts `topk % 64` (sm90/decode/sparse/splitkv_mla.cuh:689,
#: config.h:33). FlashMLA ba89a34.
_FLASHMLA_SPARSE_PREFILL_TOPK_TILE = 128
_FLASHMLA_SPARSE_DECODE_TOPK_TILE = 64


def _flashmla_topk_indices(inv: Invocation, idx, base, tokens: int,
                           tile: int):
    """Global cache-slot indices `[tokens, width]` and each row's count of
    valid ones `[tokens]`, as serving hands them to FlashMLA.

    Serving's index buffer is the indexer's whole `index_topk` wide, -1
    where a short context had nothing to select (vLLM `topk_indices_buffer`,
    model_executor/models/deepseek_v2.py:1440-1447; sglang the same), and
    both pass the valid count beside it — vLLM's `return_valid_counts=True`
    into `topk_length` (v1/attention/backends/mla/flashmla_sparse.py:
    896-920, 1384-1390), sglang `dsa_cache_seqlens_int32`
    (srt/layers/attention/dsa_backend.py:2787-2806) — so the kernel walks
    only the valid blocks. The cell carries the attended count, not the
    model's `index_topk`, so the width here is that count rounded up to the
    kernel's tile (`tile`, the entry point's own assertion above): a padded
    tail the kernel is told to skip costs what serving's wider one does, and
    a width off the tile is refused outright (GLM-5.3-Flash's 288 keys,
    Hy4's 576 and 1156, 2026-10-28)."""
    import torch
    count = idx.shape[2]
    width = int(inv.dims.get("n_topk_keys") or count)
    width = -(-max(width, count) // tile) * tile
    out = torch.full((tokens, width), -1, dtype=torch.int32, device=idx.device)
    out[:, :count] = (idx + base).reshape(tokens, count)
    valid = torch.full((tokens,), count, dtype=torch.int32, device=idx.device)
    return out, valid


def _flashmla_sparse_build(inv: Invocation, env: Env, inputs) -> BenchCase:
    """FlashMLA sparse kernel. Call shape from sglang's NSA backend
    (`_forward_flashmla_sparse`) and vLLM's FlashMLASparse `_forward_bf16_kv`:

        flash_mla_sparse_fwd(q=[tokens, heads, Dqk], kv=[slots, Dqk],
                             indices=[tokens, h_kv=1, topk], sm_scale, d_v)
            -> (out[tokens, heads, Dv], lse, …)

    Indices are GLOBAL cache slots, so the per-request selection is offset by
    the request's base slot — the same conversion
    `triton_convert_req_index_to_global_index` performs in production. Heads are
    padded to the kernel's required multiple (64 on Hopper, 128 on Blackwell)
    and the output trimmed back, as both frameworks do under TP."""
    _require_cuda()
    import torch
    from flash_mla import flash_mla_sparse_fwd

    q, c = inputs["q"], inputs["c"]              # packed [B·Sq,N,Dqk], [B·Lq,Dqk]
    idx = inputs["indices"]                      # [B,Sq,topk] int32, per-request
    B = int(inv.dims["B"])
    Sq = int(inv.dims["s_query"])
    Lq = int(inv.dims["s_kv"])
    N, Dqk = q.shape[1], q.shape[2]
    Dv = int(inputs["Dv"])
    dev = q.device
    tokens = B * Sq

    q_flat = q
    kv_flat = c.to(torch.bfloat16).unsqueeze(1)  # [B·Lq, h_kv=1, Dqk]
    base = (torch.arange(B, dtype=torch.int32, device=dev) * Lq).view(B, 1, 1)
    idx_global, valid = _flashmla_topk_indices(
        inv, idx, base, tokens, _FLASHMLA_SPARSE_PREFILL_TOPK_TILE)
    idx_global = idx_global.unsqueeze(1)

    pad_to = 128 if (env.sm or "").endswith(("100", "103", "120")) else 64
    if N % pad_to:
        q_in = q_flat.new_zeros((tokens, pad_to, Dqk))
        q_in[:, :N, :] = q_flat
    else:
        q_in = q_flat
    scale = 1.0 / (Dqk ** 0.5)

    def _canon(out):
        o = _primary(out)
        return o[:, :N, :]

    return BenchCase(
        run=lambda: flash_mla_sparse_fwd(q=q_in, kv=kv_flat, indices=idx_global,
                                         sm_scale=scale, d_v=Dv,
                                         topk_length=valid),
        to_canonical=_canon,
    )


def _quantize_fp8_ds_mla(c):
    """Latent rows [T, 576] → vLLM's `fp8_ds_mla` cache rows [T, 656] uint8:
    512 e4m3 NoPE values, 4 fp32 scales (one per 128, rounded up to a power of
    two), 64 bf16 RoPE values — FlashMLA tests/quant.py `V32_FP8Sparse`."""
    import torch
    nope, rope = c[:, :512].float(), c[:, 512:].to(torch.bfloat16)
    T = c.shape[0]
    out = torch.empty(T, 656, dtype=torch.uint8, device=c.device)
    tiles = nope.view(T, 4, 128)
    scale = (tiles.abs().amax(dim=-1) / 448.0).clamp_min(1e-4)
    scale = torch.pow(2, scale.log2().ceil())                  # ue8m0
    out[:, :512] = (tiles / scale.unsqueeze(-1)).to(
        torch.float8_e4m3fn).view(T, 512).view(torch.uint8)
    out[:, 512:528] = scale.contiguous().view(torch.uint8).view(T, 16)
    out[:, 528:] = rope.contiguous().view(torch.uint8).view(T, 128)
    return out


def _flashmla_sparse_fp8_build(inv: Invocation, env: Env, inputs) -> BenchCase:
    """FlashMLA sparse attention over an fp8 cache, as vLLM's
    `FlashMLASparseImpl._forward_fp8_kv` runs it: every query token one row of
    a batch of 1, the cache in `fp8_ds_mla` pages of 64, indices as global
    cache slots, `cache_seqlens` large enough that no index reads as invalid."""
    _require_cuda()
    import torch
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    q, c = inputs["q"], inputs["c"]              # packed [B·Sq,N,Dqk], [B·Lq,Dqk]
    idx = inputs["indices"]                      # [B,Sq,topk] int32, per-request
    B = int(inv.dims["B"])
    Sq = int(inv.dims["s_query"])
    Lq = int(inv.dims["s_kv"])
    N, Dqk = q.shape[1], q.shape[2]
    Dv = int(inputs["Dv"])
    dev = q.device
    tokens = B * Sq
    page = _FLASHMLA_SPARSE_BLOCK_SIZES[0]
    slots = B * Lq
    num_blocks = -(-slots // page)
    rows = torch.zeros(num_blocks * page, Dqk, dtype=torch.bfloat16, device=dev)
    rows[:slots] = c.to(torch.bfloat16)
    k_cache = _quantize_fp8_ds_mla(rows).view(num_blocks, page, 1, 656)
    base = (torch.arange(B, dtype=torch.int32, device=dev) * Lq).view(B, 1, 1)
    indices, _valid = _flashmla_topk_indices(
        inv, idx, base, tokens, _FLASHMLA_SPARSE_DECODE_TOPK_TILE)
    indices = indices.unsqueeze(0)
    pad_to = 128 if (env.sm or "").endswith(("100", "103", "120")) else 64
    q_in = q.to(torch.bfloat16)
    if N % pad_to:
        q_in = q_in.new_zeros((tokens, pad_to, Dqk))
        q_in[:, :N, :] = q
    q_in = q_in.unsqueeze(0)                     # batch of 1, as vLLM passes it
    cache_lens = torch.full((1,), num_blocks * page, dtype=torch.int32, device=dev)
    block_table = torch.zeros(1, 1, dtype=torch.int32, device=dev)
    meta, splits = get_mla_metadata()
    scale = 1.0 / (Dqk ** 0.5)
    return BenchCase(
        run=lambda: flash_mla_with_kvcache(
            q=q_in, k_cache=k_cache, block_table=block_table,
            cache_seqlens=cache_lens, head_dim_v=Dv,
            tile_scheduler_metadata=meta, num_splits=splits,
            softmax_scale=scale, is_fp8_kvcache=True, indices=indices),
        to_canonical=lambda out: _primary(out)[0][:, :N, :],
    )


# The sparse pair dispatch on the cache tensor's dtype, as vLLM's
# `FlashMLASparseImpl.forward` picks the path: a bf16 cache goes to
# `flash_mla_sparse_fwd`, the fp8 one (`fp8_ds_mla`) to
# `flash_mla_with_kvcache(..., is_fp8_kvcache=True, indices=…)`. Each
# build reads exactly its own cache format, so the split is declared as
# the launch's cache dtype — a cell keyed at the other format is the other
# kernel's, on every surface that reads the declaration. The fp8 record is
# FP8 e4m3 with one F32 scale per 128 latent channels rounded up to a power
# of two (`_quantize_fp8_ds_mla`): `fp8_pow2_group128` by name, which a
# per-tensor-scaled FP8 cache (`kv=fp8_e4m3` too) is not.
register(Provider(
    library="flashmla",
    signature="mla_sparse_attention",
    kernel_impl="flashmla_sparse_fp8",
    import_name="flash_mla",
    serves=KernelSignature(signature="mla_sparse_attention",
                           # A bf16 query (2,915 rows) over the FP8 record.
                           activation=("bf16",),
                           dtype_slots=(("kv", ("fp8_e4m3",)),),
                           kv_format=("fp8_pow2_group128",),
                           kv_block_sizes=_FLASHMLA_SPARSE_BLOCK_SIZES),
    build=_flashmla_sparse_fp8_build,
    min_sm=_FLASHMLA_SPARSE_MIN_SM,
    launch_inputs=(
        ("q", _PACK_BATCH_SEQ),
        ("c", ((0, 1), 2)),
    ),
    launch_output="q",
))
register(Provider(
    library="flashmla",
    signature="mla_sparse_attention",
    kernel_impl="flashmla_sparse",
    import_name="flash_mla",
    serves=KernelSignature(signature="mla_sparse_attention",
                           # `flash_mla_sparse_fwd` over a bf16 query and
                           # cache.
                           activation=("bf16",),
                           dtype_slots=(("kv", ("bf16",)),),
                           # The latent widths the kernel instantiates: Q·K
                           # over 576 (DSv3.2: 512 NoPE + 64 RoPE) or 512
                           # (GLM-5.3-Flash's NoPE latent), V 512 wide —
                           # `TORCH_CHECK(d_qk == 576 || d_qk == 512)` and
                           # `TORCH_CHECK(d_v == 512)` (FlashMLA ba89a34,
                           # csrc/api/sparse_prefill.cpp:125-126), the list
                           # vLLM republishes as the backend's head sizes
                           # (flashmla_sparse.py:190-192). Undeclared, a
                           # 192- or 256-wide cell reached the launch and
                           # raised "Invalid d_qk" (2026-10-28).
                           enum=(("d_head", (512, 576)),
                                 ("d_v_head", (512,))),
                           kv_block_sizes=_FLASHMLA_SPARSE_BLOCK_SIZES),
    build=_flashmla_sparse_build,
    min_sm=_FLASHMLA_SPARSE_MIN_SM,
    launch_inputs=(
        ("q", _PACK_BATCH_SEQ),
        ("c", ((0, 1), 2)),
    ),
    launch_output="q",
))
register(Provider(
    library="torch",
    signature="mla_sparse_attention",
    kernel_impl="torch_gather_sparse_mla",
    import_name="torch",
    serves=KernelSignature(signature="mla_sparse_attention",
                           # Gathers the selected rows directly; no pages.
                           kv_block_sizes=(),
                           # The latent at the `kv` dtype, no scales and no
                           # bf16 tail: not the 656-byte `fp8_ds_mla` row
                           # FlashMLA reads (`fp8_pow2_group128`).
                           kv_format=_PLAIN_OR_PER_TENSOR,
                           activation=TORCH_FLOAT,
                           dtype_slots=(("kv", (*TORCH_FLOAT, "fp8_e4m3")),)),
    build=_torch_sparse_mla_build,
))


# ---------------------------------------------------------------------------
# Hy4's sparse MLA with a learned per-head sink. vLLM threads the sink
# through both FlashMLA calls; sglang forces the same sparse backend when a
# checkpoint carries sinks. The sink is fp32 at the ABI, and the only cache
# sglang permits with one is bf16.
# ---------------------------------------------------------------------------
def _sparse_sink_common(inv: Invocation, env: Env, inputs, q, c, tile: int):
    """The packed q/cache/index/sink form both FlashMLA launchers take, the
    selection at the width the entry point walks (`tile`,
    `_flashmla_topk_indices`)."""
    import torch
    B, Sq = inputs["indices"].shape[:2]
    N = q.shape[1]
    Lq = int(c.shape[1])
    tokens = B * Sq
    base = (torch.arange(B, dtype=torch.int32, device=q.device)
            * Lq).view(B, 1, 1)
    logical, valid = _flashmla_topk_indices(
        inv, inputs["indices"], base, tokens, tile)
    pad_to = 128 if (env.sm or "").endswith(("100", "103", "120")) else 64
    padded = max(N, pad_to)
    if q.shape[1] < padded:
        q_in = q.new_zeros((tokens, padded, q.shape[2]))
        q_in[:, :N, :] = q
    else:
        q_in = q
    sink = torch.full((padded,), float("-inf"), dtype=torch.float32,
                      device=q.device)
    sink[:N] = inputs["sinks"].float()
    return q_in, logical, sink, valid, N, tokens


def _sink_output_shape(inputs):
    """The canonical output `[B, Sq, N, Dv]`: q's leading axes at V's width,
    which is not q's (576 → 512) — reshaped to q's own shape, every sink
    build refused its output ("shape '[1, 1, 1, 576]' is invalid for input
    of size 512", sglang, 2026-10-26)."""
    return (*inputs["q"].shape[:-1], int(inputs["Dv"]))


def _vllm_sparse_sink_bf16_build(inv: Invocation, env: Env,
                                 inputs) -> BenchCase:
    """vLLM `HYV4FlashMLASparseImpl._bf16_flash_mla_kernel`
    (models/hy_v4/nvidia/flashmla_sparse.py:217-279): global flat latent rows,
    `attn_sink` padded with -inf, and the output trimmed."""
    _require_cuda()
    import torch
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    q = inputs["q"].to(torch.bfloat16).reshape(-1, inputs["q"].shape[2],
                                               inputs["q"].shape[3])
    c = inputs["c"].to(torch.bfloat16)
    q_in, logical, sink, valid, N, _tokens = _sparse_sink_common(
        inv, env, inputs, q, c, _FLASHMLA_SPARSE_PREFILL_TOPK_TILE)
    flat_c = c.reshape(-1, 1, c.shape[-1])
    scale = 1.0 / (c.shape[-1] ** 0.5)

    def _canon(out):
        o = _primary(out)
        return o[:, :N, :].reshape(_sink_output_shape(inputs))

    return BenchCase(
        run=lambda: flash_mla_sparse_fwd(
            q_in, flat_c, logical.unsqueeze(1), scale,
            attn_sink=sink, topk_length=valid),
        to_canonical=_canon,
    )


def _vllm_sparse_sink_fp8_build(inv: Invocation, env: Env,
                                inputs) -> BenchCase:
    """vLLM's FP8 sparse branch (:175-215), with Hy4's sink threaded into
    `flash_mla_with_kvcache` and the same fp8_ds_mla pages the parent uses."""
    _require_cuda()
    import torch
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_with_kvcache, get_mla_metadata)

    q = inputs["q"].to(torch.bfloat16).reshape(-1, inputs["q"].shape[2],
                                               inputs["q"].shape[3])
    c = inputs["c"].to(torch.bfloat16)
    q_in, logical, sink, _valid, N, tokens = _sparse_sink_common(
        inv, env, inputs, q, c, _FLASHMLA_SPARSE_DECODE_TOPK_TILE)
    page = 64
    slots = c.shape[0] * c.shape[1]
    blocks = -(-slots // page)
    rows = torch.zeros(blocks * page, c.shape[-1], dtype=torch.bfloat16,
                       device=q.device)
    rows[:slots] = c.reshape(-1, c.shape[-1])
    cache = _quantize_fp8_ds_mla(rows).view(blocks, page, 1, 656)
    indices = logical.view(1, tokens, -1)
    lens = torch.full((1,), blocks * page, dtype=torch.int32, device=q.device)
    table = torch.zeros(1, 1, dtype=torch.int32, device=q.device)
    meta, splits = get_mla_metadata()
    scale = 1.0 / (c.shape[-1] ** 0.5)
    q_batch = q_in.unsqueeze(0)

    def _canon(out):
        o = _primary(out)
        return o[0, :, :N, :].reshape(_sink_output_shape(inputs))

    return BenchCase(
        run=lambda: flash_mla_with_kvcache(
            q=q_batch, k_cache=cache, block_table=table,
            cache_seqlens=lens, head_dim_v=int(inputs["Dv"]),
            tile_scheduler_metadata=meta, num_splits=splits,
            softmax_scale=scale, is_fp8_kvcache=True, indices=indices,
            attn_sink=sink),
        to_canonical=_canon,
    )


def _sgl_sparse_sink_build(inv: Invocation, env: Env, inputs) -> BenchCase:
    """sglang `_forward_flashmla_sparse` (dsa_backend.py:2741-2813): the
    same global rows and fp32 sink, with `topk_length` handed to the kernel."""
    _require_cuda()
    import torch
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    q = inputs["q"].to(torch.bfloat16).reshape(-1, inputs["q"].shape[2],
                                               inputs["q"].shape[3])
    c = inputs["c"].to(torch.bfloat16)
    q_in, logical, sink, valid, N, _tokens = _sparse_sink_common(
        inv, env, inputs, q, c, _FLASHMLA_SPARSE_PREFILL_TOPK_TILE)
    # sglang's padding cache is zero-filled, not -inf-filled
    # (dsa_backend.py:2779-2782); the padded output lanes are trimmed away.
    sink = torch.zeros_like(sink)
    sink[:N] = inputs["sinks"].float()
    flat_c = c.reshape(-1, 1, c.shape[-1])
    scale = 1.0 / (c.shape[-1] ** 0.5)
    return BenchCase(
        run=lambda: flash_mla_sparse_fwd(
            q=q_in, kv=flat_c, indices=logical.unsqueeze(1),
            sm_scale=scale, d_v=int(inputs["Dv"]), attn_sink=sink,
            topk_length=valid),
        to_canonical=lambda out: _primary(out)[:, :N, :].reshape(
            _sink_output_shape(inputs)),
    )


# The fp8 branch reads the same `fp8_ds_mla` pages as `flashmla_sparse_fp8`
# (`fp8_pow2_group128`); the bf16 one a plain cache, which names no record.
for _kv, _record, _vllm_builder in (
        ("bf16", (), _vllm_sparse_sink_bf16_build),
        ("fp8_e4m3", ("fp8_pow2_group128",), _vllm_sparse_sink_fp8_build)):
    register(Provider(
        library="vllm", signature="mla_sparse_sink_attention",
        kernel_impl=f"vllm_flashmla_sparse_sink_{_kv}",
        import_name="vllm.v1.attention.ops.flashmla",
        serves=KernelSignature(signature="mla_sparse_sink_attention",
                               activation=("bf16",), weight=("fp32",),
                               dtype_slots=(("kv", (_kv,)),),
                               kv_format=_record,
                               enum=(("d_head", (576,)),
                                     ("d_v_head", (512,))),
                               kv_block_sizes=(64,)),
        build=_vllm_builder, fused_op="flash_dsa_sink",
        min_sm=_FLASHMLA_SPARSE_MIN_SM,
    ))
register(Provider(
    library="sglang", signature="mla_sparse_sink_attention",
    kernel_impl="sglang_flashmla_sparse_sink",
    import_name="sgl_kernel.flash_mla",
    serves=KernelSignature(signature="mla_sparse_sink_attention",
                           activation=("bf16",), weight=("fp32",),
                           dtype_slots=(("kv", ("bf16",)),),
                           enum=(("d_head", (576,)), ("d_v_head", (512,))),
                           kv_block_sizes=(64,)),
    build=_sgl_sparse_sink_build, fused_op="flash_dsa_sink",
    min_sm=_FLASHMLA_SPARSE_MIN_SM,
))
register(Provider(
    library="torch", signature="mla_sparse_sink_attention",
    kernel_impl="torch_sparse_mla_sink", import_name="torch",
    serves=KernelSignature(signature="mla_sparse_sink_attention",
                           kv_block_sizes=(),
                           # The oracle over the latent at the `kv` dtype.
                           kv_format=_PLAIN_OR_PER_TENSOR,
                           activation=TORCH_FLOAT,
                           # The per-head sinks: fp32 as the kernels take
                           # them, or at the model's dtype.
                           weight=TORCH_FLOAT,
                           dtype_slots=(("kv", (*TORCH_FLOAT, "fp8_e4m3")),)),
    # The gather floor with the sink in its softmax, at the cell's dtype —
    # not the fp32 oracle, which this registration used to time verbatim.
    build=_torch_sparse_mla_build, fused_op="flash_dsa_sink",
))


# ---------------------------------------------------------------------------
# Qwen QSA's unsplit sparse prefill and its output-gate epilogue.
# ---------------------------------------------------------------------------
def _qsa_paged_caches(inv: Invocation, k, v, dtype):
    """vLLM's production cache split: one [pages,page,H,2D] pool, transposed
    to [pages,H,page,2D], then split into strided K/V views."""
    import torch
    B, Lq, H, D = k.shape
    page = int(inv.dims.get("kv_block_size") or 16)
    pages_per_request = -(-Lq // page)
    pages = B * pages_per_request
    pool = torch.zeros(pages, page, H, 2 * D, dtype=torch.bfloat16,
                       device=k.device)
    for b in range(B):
        flat = pool[b * pages_per_request:(b + 1) * pages_per_request
                    ].view(-1, H, 2 * D)
        flat[:Lq, :, :D] = k[b].to(torch.bfloat16)
        flat[:Lq, :, D:] = v[b].to(torch.bfloat16)
    block_table = torch.arange(pages, dtype=torch.int32,
                               device=k.device).view(B, -1)
    scales = (1.0, 1.0)
    if dtype == torch.float8_e4m3fn:
        # Production passes one host K and V dequant scale a layer
        # (qsa.py:298-307, :640-649). Quantize the canonical lossless cache
        # on the same per-tensor e4m3 grid rather than merely reinterpreting
        # its bf16 bytes.
        k_raw, v_raw = pool[..., :D], pool[..., D:]
        k_scale = max(float(k_raw.abs().max()) / 448.0, 1e-12)
        v_scale = max(float(v_raw.abs().max()) / 448.0, 1e-12)
        quant = torch.empty_like(pool, dtype=dtype)
        quant[..., :D] = (k_raw / k_scale).to(dtype)
        quant[..., D:] = (v_raw / v_scale).to(dtype)
        pool, scales = quant, (k_scale, v_scale)
    return pool.transpose(1, 2).split(D, dim=-1), block_table, scales


def _vllm_qsa_gated_build(inv: Invocation, env: Env, inputs) -> BenchCase:
    """`qsa_sparse_paged_attention` exactly as `forward_qsa` calls it
    (qsa.py:287-330): packed int32 selection with a trailing count column,
    the request map, a real block table, and the fresh contiguous gate."""
    _require_cuda()
    import torch
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        qsa_sparse_paged_attention)

    q = inputs["q"].to(torch.bfloat16).reshape(-1, *inputs["q"].shape[-2:])
    gate = inputs["gate"].to(torch.bfloat16).reshape_as(q).contiguous()
    idx = inputs["indices"]
    B, Sq = idx.shape[:2]
    tokens = B * Sq
    topk = idx.shape[2]
    packed = torch.zeros(tokens, topk + 1, dtype=torch.int32,
                         device=q.device)
    packed[:, :topk] = idx.reshape(tokens, topk).to(torch.int32)
    packed[:, topk] = min(topk, int(inputs["k"].shape[1]))
    (k_cache, v_cache), block_table, (k_scale, v_scale) = _qsa_paged_caches(
        inv, inputs["k"], inputs["v"], _torch_dtype(inv.kv_dtype))
    token_to_req = torch.arange(B, dtype=torch.int32, device=q.device
                                ).repeat_interleave(Sq)
    out = torch.empty_like(q)
    return BenchCase(
        run=lambda: qsa_sparse_paged_attention(
            q, k_cache, v_cache, packed, block_table, token_to_req,
            True, out, k_scale=k_scale, v_scale=v_scale,
            output_gate=gate),
        to_canonical=lambda _o: out.reshape(inputs["q"].shape),
    )


def _torch_sparse_gqa_run(inv: Invocation, inputs):
    """Top-k GQA at the kernel's own dtypes: each query's selected K/V rows
    of its KV head, every query head of the group over them. Returns the
    launch as a callable producing `[B, Sq, N, Dv]` at the activation dtype."""
    import torch
    dt = _torch_dtype(inv.dtype)
    kv_dt = _torch_dtype(inv.kv_dtype)
    q = inputs["q"].to(dt)
    k, v = inputs["k"].to(kv_dt), inputs["v"].to(kv_dt)
    idx = inputs["indices"].long()
    B, Sq, N, D = q.shape
    _B, Lq, Nkv, _D = k.shape
    Dv = v.shape[3]
    topk = idx.shape[2]
    group = N // Nkv
    scale = D ** -0.5
    sel = idx.unsqueeze(-1).unsqueeze(-1)
    work = q if q.dtype == k.dtype == v.dtype else torch.promote_types(
        q.dtype, torch.promote_types(k.dtype, v.dtype))
    # Selecting each query's K/V rows is input prep — the kernel's own gather
    # hands its math the same operands — so it happens here, not inside the
    # timed call. The rows stay per KV head: the math broadcasts each head of
    # the group over them (`_torch_eager_gqa`'s spelling). A repeat_interleave
    # to one copy per query head read the same values but materialised
    # [B,Sq,N,topk,D] twice — 48 GiB at QSA's 2048-token prefill cell, the
    # OOM the queue logged.
    ks = torch.gather(
        k.unsqueeze(1).expand(B, Sq, Lq, Nkv, D), 2,
        sel.expand(B, Sq, topk, Nkv, D)).permute(0, 1, 3, 2, 4)
    vs = torch.gather(
        v.unsqueeze(1).expand(B, Sq, Lq, Nkv, Dv), 2,
        sel.expand(B, Sq, topk, Nkv, Dv)).permute(0, 1, 3, 2, 4)

    def run():
        # The dot starts at the operands' width (so no widened K/V copy is
        # timed), then its fp32 accumulator and softmax match the Triton
        # kernel's online normalization. Head n of the group reads kv head
        # n // group — repeat_interleave's order — as a reshape.
        scores = torch.einsum(
            "bskgd,bsktd->bkgst",
            q.to(work).reshape(B, Sq, Nkv, group, D), ks.to(work))
        # FlashMLA casts P back to V's width for the PV dot (qsa.py:173-176).
        attn = scores.float().mul_(scale).softmax(dim=-1).to(v.dtype)
        o = torch.einsum("bkgst,bsktd->bskgd", attn, vs)
        return o.reshape(B, Sq, N, Dv).to(q.dtype)

    return run


def _torch_sparse_gqa_build(inv: Invocation, env: Env, inputs) -> BenchCase:
    return BenchCase(run=_torch_sparse_gqa_run(inv, inputs),
                     to_canonical=lambda out: out.contiguous())


def _torch_qsa_gated_build(inv: Invocation, env: Env, inputs) -> BenchCase:
    import torch
    attend = _torch_sparse_gqa_run(inv, inputs)
    gate = inputs["gate"].to(_torch_dtype(inv.dtype))
    return BenchCase(run=lambda: attend() * torch.sigmoid(gate),
                     to_canonical=lambda out: out)


# Top-k GQA — `mla_sparse_attention`'s per-head geometry (`n_kv_heads`
# bound): Qwen QSA's split decode and Step-5's region-sparse attention. The
# torch floor, so every card measures it. vLLM's own QSA kernel
# (`qsa_sparse_paged_attention`, models/qwen4_exp/nvidia/ops/qsa.py:594)
# always takes the output gate (`output_gate` is keyword-required, :606):
# at decode its split partial kernel (:694) writes fp32 partials and the
# merge (:741) gates them, two launches the plan prices as this span and
# `sigmoid_gate_mul`. Timing the entry point here would time both, and the
# partial kernel alone is not an entry point; Step-5 has no kernel in either
# audited framework (vLLM v0.32.0, sglang v0.5.22).
register(Provider(
    library="torch",
    signature="mla_sparse_attention",
    kernel_impl="torch_gather_sparse_gqa",
    import_name="torch",
    serves=KernelSignature(signature="mla_sparse_attention",
                           requires_dims=("n_kv_heads",),
                           requires_divisible_dims=(("n_heads", "n_kv_heads"),),
                           # Gathers the selected rows directly; no pages.
                           kv_block_sizes=(),
                           # The QSA floor's own chain (below): it dots at
                           # the operands' width and casts back to the
                           # query's, any float width; the plain cache only.
                           activation=TORCH_FLOAT,
                           dtype_slots=(("kv", TORCH_FLOAT),)),
    build=_torch_sparse_gqa_build,
))


# The fp8 kernel reads one host K and one V dequant scale a layer
# (`_qsa_paged_caches`): `fp8_per_tensor`. The bf16 one a plain cache.
for _kv, _record in (("bf16", ()), ("fp8_e4m3", ("fp8_per_tensor",))):
    register(Provider(
        library="vllm", signature="gqa_sparse_gated_attention",
        kernel_impl=f"vllm_qsa_sparse_gated_{_kv}",
        import_name="vllm.models.qwen4_exp.nvidia.ops.qsa",
        serves=KernelSignature(signature="gqa_sparse_gated_attention",
                               activation=("bf16",),
                               dtype_slots=((("kv", (_kv,))),),
                               kv_format=_record,
                               phases=("prefill",),
                               kv_block_sizes=(16, 32, 64, 128, 256)),
        build=_vllm_qsa_gated_build, fused_op="flash_dsa_gated",
    ))
register(Provider(
    library="torch", signature="gqa_sparse_gated_attention",
    kernel_impl="torch_qsa_sparse_gated", import_name="torch",
    serves=KernelSignature(signature="gqa_sparse_gated_attention",
                           kv_block_sizes=(), phases=("prefill",),
                           activation=TORCH_FLOAT,
                           dtype_slots=(("kv", (*TORCH_FLOAT, "fp8_e4m3")),),
                           kv_format=_PLAIN_OR_PER_TENSOR),
    build=_torch_qsa_gated_build, fused_op="flash_dsa_gated",
))


# ---------------------------------------------------------------------------
# More MLA decode backends (envelope vs FlashMLA). All use the ABSORBED latent
# geometry that matches the canonical MLA inputs: q_nope[.,H,512] + q_rope[.,H,
# 64], latent ckv[.,512] + kpe[.,64]. From vLLM/sglang MLA backend call sites.
# ---------------------------------------------------------------------------

def _flashinfer_mla_build(inv, env, inputs) -> BenchCase:
    """FlashInfer MLA paged decode (BatchMLAPagedAttentionWrapper). Splits the
    canonical q[B,Sq,N,576]/latent c[B,Lq,576] into nope(512)+rope(64) and pages
    the latent at the deployment's block size. plan(q_indptr, kv_indptr,
    kv_indices, kv_lens,
    num_heads, head_dim_ckv=512, head_dim_kpe=64, page_size, causal, scale,
    q_dtype, kv_dtype); run(q_nope, q_rope, ckv, kpe)."""
    _require_cuda()
    import torch
    import flashinfer

    q, c = inputs["q"], inputs["c"]        # [B,Sq,N,Dqk], [B,Lq,Dqk]
    B, Sq, N, Dqk = q.shape
    Lq = c.shape[1]; Dv = int(inputs["Dv"]); Drope = Dqk - Dv
    dev, dt = q.device, q.dtype
    q_nope = q[..., :Dv].reshape(B * Sq, N, Dv).contiguous()
    q_rope = q[..., Dv:].reshape(B * Sq, N, Drope).contiguous()
    page = resolve_kv_block_size(inv, _FLASHINFER_MLA_BLOCK_SIZES)
    pages_per_seq = (Lq + page - 1) // page
    num_pages = B * pages_per_seq
    padded = pages_per_seq * page
    c_pad = torch.zeros(B, padded, Dqk, dtype=dt, device=dev)
    c_pad[:, :Lq] = c
    ckv = c_pad[..., :Dv].reshape(num_pages, page, Dv).contiguous()
    kpe = c_pad[..., Dv:].reshape(num_pages, page, Drope).contiguous()
    q_indptr = torch.arange(0, B * Sq + 1, Sq, dtype=torch.int32, device=dev)
    kv_indptr = torch.arange(0, (B + 1) * pages_per_seq, pages_per_seq,
                             dtype=torch.int32, device=dev)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=dev)
    kv_lens = torch.full((B,), Lq, dtype=torch.int32, device=dev)
    wbuf = torch.empty(_FLASHINFER_WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchMLAPagedAttentionWrapper(wbuf, backend="auto")
    scale = 1.0 / (Dqk ** 0.5)
    # Causal is the cell's (`refs._causal`): a verify step's queries are the
    # last `Sq` positions of each context, and sglang plans this wrapper with
    # causal=True for target-verify (flashinfer_mla_backend.py, the
    # FlashInferMLAIndicesUpdaterPrefill paged plan) and False only for its
    # one-token decode. Planned False here, a verify cell let every draft
    # token read the drafts after it.
    wrapper.plan(q_indptr, kv_indptr, kv_indices, kv_lens, N, Dv, Drope, page,
                 bool(inputs["causal"]), scale, dt, dt)
    return BenchCase(
        run=lambda: wrapper.run(q_nope, q_rope, ckv, kpe),
        to_canonical=lambda out: out.reshape(B, Sq, N, Dv),
    )


def _cutlass_mla_build(inv, env, inputs) -> BenchCase:
    """CUTLASS MLA decode (sgl_kernel cutlass_mla_decode, SM100). Paged latent
    at the deployment's block size — this kernel serves 128 only — with a
    combined kv_c_and_k_pe_cache[num_blocks, page, 576]."""
    _require_cuda()
    import torch
    from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size

    q, c = inputs["q"], inputs["c"]
    B, Sq, N, Dqk = q.shape
    Lq = c.shape[1]; Dv = int(inputs["Dv"]); Drope = Dqk - Dv
    dev, dt = q.device, q.dtype
    PAGE = resolve_kv_block_size(inv, _CUTLASS_MLA_BLOCK_SIZES)
    q_nope = q[..., :Dv].reshape(B * Sq, N, Dv).contiguous()
    q_pe = q[..., Dv:].reshape(B * Sq, N, Drope).contiguous()
    blocks = (Lq + PAGE - 1) // PAGE
    num_blocks = B * blocks
    c_pad = torch.zeros(B, blocks * PAGE, Dqk, dtype=dt, device=dev)
    c_pad[:, :Lq] = c
    cache = c_pad.reshape(num_blocks, PAGE, Dqk).contiguous()
    page_table = torch.arange(num_blocks, dtype=torch.int32,
                              device=dev).reshape(B, blocks)
    seq_lens = torch.full((B,), Lq, dtype=torch.int32, device=dev)
    ws = torch.empty(cutlass_mla_get_workspace_size(blocks * PAGE, B, num_kv_splits=1),
                     device=dev, dtype=torch.uint8)
    scale = 1.0 / (Dqk ** 0.5)
    return BenchCase(
        run=lambda: cutlass_mla_decode(
            q_nope=q_nope, q_pe=q_pe, kv_c_and_k_pe_cache=cache,
            seq_lens=seq_lens, page_table=page_table, workspace=ws,
            sm_scale=scale, num_kv_splits=1),
        to_canonical=lambda out: out.reshape(B, Sq, N, Dv),
    )


register(Provider(
    library="flashinfer", signature="mla_attention",
    kernel_impl="flashinfer_mla_decode", import_name="flashinfer",
    serves=KernelSignature(signature="mla_attention", phases=("decode",),
                           kv_block_sizes=_FLASHINFER_MLA_BLOCK_SIZES,
                           # The build pages the latent at the QUERY's
                           # dtype and plans `kv_dtype` = it: a bf16 cache
                           # whatever the cell's — so a cell keyed at
                           # another cache dtype is not this launch.
                           kv_format=(PLAIN,),
                           activation=("bf16", "fp16"),
                           dtype_slots=(("kv", ("bf16", "fp16")),)),
    build=_flashinfer_mla_build,
    impl_params=lambda inv: f"_b{resolve_kv_block_size(inv, _FLASHINFER_MLA_BLOCK_SIZES)}",
))
register(Provider(
    library="cutlass_mla", signature="mla_attention",
    kernel_impl="cutlass_mla_decode", import_name="sgl_kernel",


    # The SM100 floor, declared rather than described. As prose in the hint it
    # constrained nothing: on an H100 this kernel was attempted on every MLA
    # decode cell and raised on every one of them.
    min_sm=100,
    serves=KernelSignature(signature="mla_attention", phases=("decode",),
                           # One query token a request: the build hands the
                           # kernel `B·Sq` query rows against `B` sequence
                           # lengths, so a verify step would be read as a
                           # batch of `B·Sq` requests. sglang's CutlassMLA
                           # backend runs it for decode only and sends a
                           # target-verify step to its FlashInfer MLA parent.
                           enum=(("s_query", (1,)),),
                           kv_block_sizes=_CUTLASS_MLA_BLOCK_SIZES,
                           # The build pages the latent at the query's
                           # dtype (`c_pad` at `dt`), a bf16 cache whatever
                           # the cell's; vLLM's CUTLASS MLA takes no scale
                           # either ("does not support scaling for q and
                           # kv_latent", mla/cutlass_mla.py:259-262).
                           kv_format=(PLAIN,),
                           activation=("bf16", "fp16"),
                           dtype_slots=(("kv", ("bf16", "fp16")),)),
    build=_cutlass_mla_build,
    impl_params=lambda inv: f"_b{resolve_kv_block_size(inv, _CUTLASS_MLA_BLOCK_SIZES)}",
))


# ---------------------------------------------------------------------------
# torch SDPA — a no-dependency GQA/MHA baseline that runs on ANY CUDA build
# (Ada included; no nvcc, no Hopper), so `gqa_attention` is always covered even
# when the fused attention libs (fa2/fa3/fa4/flashinfer) won't provision. The
# fused kernels are the envelope; torch SDPA is the floor. Heads-before-sequence
# is declared as `launch_inputs`; enable_gqa lets Nk<N run without repeating KV.
# ---------------------------------------------------------------------------
def _fused_sdpa_backends() -> list:
    """SDPA's backends other than MATH: the ones that do not score in fp32.

    One list for both the question (`_fused_sdpa_takes`) and the timed call
    (`_sdpa_case`), because torch's own order puts MATH AHEAD of cuDNN
    (`torch._C._get_sdp_priority_order()`: flash, efficient, math, cudnn on
    torch 2.8 and 2.11). Asked with MATH excluded, a windowed GQA cell on the
    H100 is taken by cuDNN; timed with torch's default order, the same call
    ran the math path and its fp32 scores, and the launch gate refused it
    ("float32[2,1024,1024]", Mistral-7B, 2026-10-26)."""
    from torch.nn.attention import SDPBackend
    return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION]


def _fused_sdpa_takes(q, k, v, gqa: bool, cargs) -> bool:
    """Whether a fused SDPA backend (`_fused_sdpa_backends`) takes this call
    — asked of torch, by making it with the math backend excluded, rather
    than predicted from the shapes: which calls fall to math depends on the
    device, the head widths, the mask and GQA together (a windowed GQA cell
    does on an RTX 4090: flash takes no mask, efficient no GQA)."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel
    try:
        with sdpa_kernel(_fused_sdpa_backends()):
            F.scaled_dot_product_attention(q, k, v, enable_gqa=gqa, **cargs)
    except torch.OutOfMemoryError:
        raise                  # the driver's OOM outcome, not "no backend"
    except RuntimeError:
        return False
    return True


def _sdpa_case(q, k, v, causal: bool, s_window=None) -> BenchCase:
    """torch's attention over launch-layout q[B,N,Sq,Dq] / k[B,Nk,Lq,Dq] /
    v[B,Nk,Lq,Dv], at the operands' own precision: on the fused backends
    that take the call, under exactly the backends that were asked, or
    else the eager spelling (`_torch_eager_gqa`; a windowed cell past
    `_WINDOWED_DENSE_MAX` walks it in query blocks,
    `_torch_eager_windowed_gqa`) — never SDPA's math path,
    whose fp32 score matrix is another precision's latency."""
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel
    gqa = q.shape[1] != k.shape[1]
    if (s_window is not None and int(s_window) > 0
            and q.shape[2] * k.shape[2] > _WINDOWED_DENSE_MAX):
        # Decided on the shapes, before any allocation: the dense spelling
        # builds the [Sq,Lq] distance matrix and bias ahead of the probe that
        # says whether a fused backend takes the call — 160 GiB at the
        # 131072-token windowed cell, the OOM the queue logged — and the
        # eager fallback behind it would then hold [B,N,Sq,Lq] scores. A
        # windowed cell this size is walked in query blocks instead
        # (`_torch_eager_windowed_gqa`), and no [Sq,Lq] tensor exists.
        return BenchCase(run=_torch_eager_windowed_gqa(q, k, v, int(s_window)))
    cargs = _sdpa_causal_args(q, k, causal, s_window=s_window)
    if not _fused_sdpa_takes(q, k, v, gqa, cargs):
        # The mask is built here, once, so the timed call is the attention
        # alone.
        hidden = (~visible_keys(q.shape[2], k.shape[2], q.device,
                                causal=causal, s_window=s_window)
                  if causal or s_window else None)
        return BenchCase(run=lambda: _torch_eager_gqa(q, k, v, hidden))
    backends = _fused_sdpa_backends()

    def run():
        with sdpa_kernel(backends):
            return F.scaled_dot_product_attention(
                q, k, v, enable_gqa=gqa, **cargs)
    return BenchCase(run=run)


def _torch_eager_gqa(q, k, v, hidden=None):
    """SDPA's math path, at the cell's own dtype.

    Where no fused SDPA backend takes a call (`_fused_sdpa_takes`) SDPA runs
    its math path, which dispatches an fp32 score matrix on a bf16 cell: an
    asymmetric V (a MiMo cell, Dq=192, Dv=128) on a device whose fused
    backends want one last extent, or a windowed GQA cell on a card whose
    flash backend takes no mask. That is another precision's latency, which
    the launch gate rightly refuses. This spelling keeps the score matrix at the operands'
    width by grouping the shared KV heads instead of repeating K/V: Q becomes
    [B, Nkv, G*Sq, Dq], scores become [B, Nkv, G*Sq, Skv], and V broadcasts
    across each group without a material copy.

    `hidden` is the `[Sq, Skv]` bool of keys each query does NOT read
    (`~refs.visible_keys`), built once with the case rather than in every
    timed call; None reads every key.
    """
    import torch
    if q.shape[1] % k.shape[1]:
        raise ValueError(
            f"torch eager GQA needs n_heads % n_kv_heads == 0, got "
            f"{q.shape[1]} % {k.shape[1]}")
    B, H, S, Dq = q.shape
    _B, N, L, Dk = k.shape
    _B2, _N2, _L2, Dv = v.shape
    if Dq != Dk:
        raise ValueError(
            f"torch eager GQA needs Q/K widths to agree, got {Dq}/{Dk}")
    group_rows = H // N
    work = q if q.dtype == k.dtype == v.dtype else torch.promote_types(
        q.dtype, torch.promote_types(k.dtype, v.dtype))
    qq = q.reshape(B, N, group_rows * S, Dq).to(work)
    kk = k.to(work)
    vv = v.to(work)
    scores = torch.matmul(qq, kk.transpose(-2, -1))
    scores = scores.mul_(Dq ** -0.5)
    if hidden is not None:
        scores = scores.reshape(B, N, group_rows, S, L).masked_fill(
            hidden, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(
        probs.reshape(B, N, group_rows * S, L), vv
    ).reshape(B, H, S, Dv)
    return out.to(q.dtype)


#: `Sq * Lq` past which a windowed cell's dense mask spelling is not built:
#: the distance matrix, bool mask and bf16 bias are [Sq, Lq] each — 160 GiB
#: at the 131072-token windowed prefill cell the queue OOMed on. Every
#: windowed cell measured so far is at or under 8192x8192 (2**26, a quarter
#: of this) and keeps the dense path its rows were timed with; past it the
#: floor walks the window in query blocks (`_torch_eager_windowed_gqa`).
_WINDOWED_DENSE_MAX = 1 << 28


def _torch_eager_windowed_gqa(q, k, v, s_window: int):
    """`_torch_eager_gqa` for a windowed cell, walked in query blocks; returns
    the timed call.

    Same bottom-right alignment, group broadcasting and dtype behaviour as
    the dense eager spelling, but a block of rows [s0, s1) scores only the
    keys its window can read — [max(0, Lq−Sq+s0−W+1), Lq−Sq+s1) — so the
    score tensor is [B,N,chunk,chunk+W] and the mask [chunk, chunk+W]; no
    [Sq,Lq] tensor exists. The block is sized to the gather floors' budget
    on the SCORES, the widest tensor this spelling forms.

    Built here, outside the timed call, as the dense spelling builds its
    mask: the output the blocks write into and each block's mask. A block's
    mask depends only on its row count, key count and the offset of its
    first row from its first key, so every interior block shares one.
    """
    import math
    import torch
    if q.shape[1] % k.shape[1]:
        raise ValueError(
            f"torch eager GQA needs n_heads % n_kv_heads == 0, got "
            f"{q.shape[1]} % {k.shape[1]}")
    B, H, S, Dq = q.shape
    _B, N, L, Dk = k.shape
    _B2, _N2, _L2, Dv = v.shape
    if Dq != Dk:
        raise ValueError(
            f"torch eager GQA needs Q/K widths to agree, got {Dq}/{Dk}")
    W = int(s_window)
    group_rows = H // N
    work = q if q.dtype == k.dtype == v.dtype else torch.promote_types(
        q.dtype, torch.promote_types(k.dtype, v.dtype))
    qq = q.reshape(B, N, group_rows, S, Dq).to(work)
    kk = k.to(work)
    vv = v.to(work)
    scale = Dq ** -0.5
    # chunk * (chunk + W) scores a head, within the block budget.
    room = _refs.GATHER_BLOCK_BYTES // max(1, B * H * qq.element_size())
    chunk = max(1, min(S, (math.isqrt(W * W + 4 * room) - W) // 2))
    out = torch.empty(B, N, group_rows, S, Dv, dtype=qq.dtype,
                      device=q.device)
    masks: dict = {}
    blocks = []
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        hi = L - S + s1                      # the last row reads to its own key
        lo = max(0, L - S + s0 - W + 1)      # the first row's window reach
        key = (s1 - s0, hi - lo, L - S + s0 - lo)
        if key not in masks:
            rows, cols, off = key
            dist = (torch.arange(rows, device=q.device, dtype=torch.int32)[:, None]
                    + off
                    - torch.arange(cols, device=q.device, dtype=torch.int32)[None, :])
            masks[key] = (dist < 0) | (dist >= W)
        blocks.append((s0, s1, lo, hi, masks[key]))
    neg = torch.finfo(qq.dtype).min

    def _run():
        for s0, s1, lo, hi, hidden in blocks:
            rows = s1 - s0
            scores = torch.matmul(
                qq[:, :, :, s0:s1].reshape(B, N, group_rows * rows, Dq),
                kk[:, :, lo:hi].transpose(-2, -1))
            scores = scores.mul_(scale).reshape(
                B, N, group_rows, rows, hi - lo).masked_fill_(hidden, neg)
            probs = torch.softmax(scores, dim=-1)
            out[:, :, :, s0:s1] = torch.matmul(
                probs.reshape(B, N, group_rows * rows, hi - lo),
                vv[:, :, lo:hi]).reshape(B, N, group_rows, rows, Dv)
        return out.reshape(B, H, S, Dv).to(q.dtype)

    return _run


def _torch_sdpa_gqa_build(inv: Invocation, env: Env, inputs) -> BenchCase:
    if inputs.get("sinks") is not None:
        raise NotImplementedError(
            "torch SDPA has no attention-sink path; FA/FlashInfer measure it")
    dt, kv_dt = _torch_dtype(inv.dtype), _torch_dtype(inv.kv_dtype)
    return _sdpa_case(inputs["q"].to(dt), inputs["k"].to(kv_dt),
                      inputs["v"].to(kv_dt), bool(inputs["causal"]),
                      s_window=inputs.get("s_window"))


register(Provider(
    library="torch", signature="gqa_attention", kernel_impl="torch_sdpa",
    import_name="torch",
    serves=KernelSignature(signature="gqa_attention",
                           # The dense reference: contiguous K/V, no pages.
                           kv_block_sizes=(),
                           # A windowed mask and a V of its own width. SDPA
                           # has no attention-sink operand, so no sink: the
                           # build raises on one (9,713 lake rows), and every
                           # gpt-oss sink cell reached the GPU while that
                           # was the only place it was said (2026-10-26).
                           variant_dims=("s_window", "d_v_head"),
                           # Whatever float dtype the cell is in; an FP8
                           # cache is dequantised by its per-tensor scale.
                           activation=TORCH_FLOAT,
                           dtype_slots=(("kv", (*TORCH_FLOAT, "fp8_e4m3")),),
                           kv_format=_PLAIN_OR_PER_TENSOR),
    build=_torch_sdpa_gqa_build,
    launch_inputs=_SDPA_LAUNCH,
    launch_output="q",
))
