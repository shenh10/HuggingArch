"""Sparse-attention indexer providers (DSA logits, MiniMax-M3 block-max scores).

Signature `indexer_logits`: logits[t,j] = Σ_h weights[t,h]·ReLU(q[t,h]·k[j]) — the
gate-weighted MQA dot products DeepSeek sparse attention uses to select the
top-k KV tokens. The production kernel is DeepGEMM `fp8_paged_mqa_logits`
(paged, FP8, Blackwell/SM100 — not runnable on Ada), so torch is the always-
available dense implementation that actually measures here; the DeepGEMM
provider is registered so it surfaces in the catalog and runs on a Blackwell
box (`source_build`, like the other DeepGEMM kernels).
"""
from __future__ import annotations

from typing import Any, Dict

from ..env import Env
from ..invocation import Invocation
from . import TORCH_FLOAT, BenchCase, KernelSignature, Launch, Provider, register
from .attention import _require_cuda


#: Bytes the indexer floor's score intermediates may hold — the scores at
#: the cell's dtype, their fp32 upcast, and the fp32 logits. A cell under it
#: keeps the one-shot einsum pair its stored rows were timed with; past it
#: the launch tiles over tokens, each chunk sized to the same bytes. The
#: dense score tensor is [tokens,Hix,Lq] — quadratic in the context — and at
#: the production cell (16384·32·8192) the one-shot pair is 8 GiB bf16 +
#: 16 GiB fp32: a 24 GiB card OOMs.
_INDEXER_ONESHOT_PEAK = 4 << 30


def _torch_indexer_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    import torch
    from .attention import _torch_dtype
    dt = _torch_dtype(inv.dtype if not inv.dtype.startswith("fp8") else "bf16")
    q, k, w = inputs["q"].to(dt), inputs["k"].to(dt), inputs["weights"]
    # Each request's queries (all heads) against its own keys: the cache
    # operand is `[B, s_kv, Dix]` — one per request — so the contraction is
    # one batched GEMM over the requests and the gate sum is over heads.
    batch, keys = k.shape[0], k.shape[1]
    heads = q.shape[1]
    tokens = q.shape[0] // batch
    qb = q.reshape(batch, tokens, heads, q.shape[-1])
    wb = w.reshape(batch, tokens, heads, 1)
    kt = k.transpose(1, 2)                                   # [B,Dix,Lq]

    def _logits(query, gate):
        scores = torch.bmm(query.flatten(1, 2), kt)          # [B,c·Hix,Lq]
        scores = scores.view(batch, -1, heads, keys).relu().float()
        return (scores * gate).sum(2)                        # [B,c,Lq]

    # A score element is held at the cell's width and again as fp32 (no
    # second copy when the cell is fp32: `.float()` returns the tensor).
    per_score = q.element_size() + (0 if q.dtype == torch.float32 else 4)
    per_token = batch * keys * (heads * per_score + 4)
    if tokens * per_token <= _INDEXER_ONESHOT_PEAK:
        def _run():
            return _logits(qb, wb).flatten(0, 1)             # [B·s,Lq]
    else:
        # The same contraction, tiled over each request's tokens: a chunk's
        # scores and their fp32 upcast are the only live intermediates, and
        # the logits are still written fp32 — into a buffer built here, so
        # the timed call allocates nothing of its own.
        out = torch.empty(batch, tokens, keys, dtype=torch.float32,
                          device=q.device)
        chunk = max(1, _INDEXER_ONESHOT_PEAK // per_token)

        def _run():
            for lo in range(0, tokens, chunk):
                hi = min(lo + chunk, tokens)
                out[:, lo:hi] = _logits(qb[:, lo:hi], wb[:, lo:hi])
            return out.flatten(0, 1)
    return BenchCase(run=_run, to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="indexer_logits", kernel_impl="torch_dense_mqa_logits",
    import_name="torch",
    # It writes the logits fp32 (`scores.float()` before the gate sum). The
    # key is read as a plain tensor cast to the cell's dtype — no record's
    # scales — so a cell keyed at a cache record (`kv_format`) is not its.
    serves=KernelSignature(signature="indexer_logits",
                           activation=TORCH_FLOAT, output=("fp32",),
                           # The dense reference reads the candidate K as one
                           # contiguous buffer; no page table.
                           kv_block_sizes=()),
    build=_torch_indexer_build,
))


def _torch_block_score_build(inv: Invocation, env: Env,
                             inputs: Dict[str, Any]) -> BenchCase:
    import torch
    from .attention import _torch_dtype
    from ..refs import causal_block_max
    dt = _torch_dtype(inv.dtype if not inv.dtype.startswith("fp8") else "bf16")
    q, k, block = inputs["q"].to(dt), inputs["k"].to(dt), int(inputs["block"])

    def _run():
        scores = torch.einsum("bshd,btd->bhst", q, k).float()
        return causal_block_max(scores, block)
    return BenchCase(run=_run, to_canonical=lambda o: o)


register(Provider(
    library="torch", signature="indexer_block_score",
    kernel_impl="torch_dense_block_max_scores",
    import_name="torch",
    # Block maxima of fp32 scores (`.float()` before the pool): writes fp32.
    serves=KernelSignature(signature="indexer_block_score",
                           activation=TORCH_FLOAT, output=("fp32",),
                           kv_block_sizes=()),
    build=_torch_block_score_build,
    fused_op="indexer_block_score",
))


# vLLM's prefill score launch `minimax_m3_index_score` (models/minimax_m3/
# common/ops/index_topk.py:674, Triton `_index_block_score_kernel` :82, block
# max at :176, v0.32.0): per-head q·k max-pooled over each 128-key block,
# causal (`off_q >= pos`), writing [num_idx_heads, total_q, max_block] fp32
# off a paged index-K cache. The wrapper is exactly the span's launch — one
# score kernel in, the block scores out — which is why vLLM gets a provider
# and sglang does not: sglang's wrappers (`flash_prefill_with_topk_index` /
# `flash_decode_with_topk_idx`) fold the score launch and the top-k launch
# into one call and return only `(o, topk_idx)`, so the score launch is not
# reachable on its own through the framework's own entry points.
#
# Two neighbours are deliberately NOT served:
#   * vLLM's DECODE score launch `minimax_m3_index_decode_score` (:797) writes
#     force-select sentinels into the init/local blocks (`_decode_index_score_
#     kernel` :409-411, 1e29/1e30) — values the plan's span does not compute,
#     and modelling them would take init_blocks/local_blocks as cell axes.
#     Decode cells keep the torch floor.
#   * Blocks past a query's causal reach are left UNWRITTEN by the kernel
#     (its `hi` bound); `to_canonical` floors them to the oracle's sentinel.
def _vllm_block_score_build(inv: Invocation, env: Env,
                            inputs: Dict[str, Any]) -> BenchCase:
    _require_cuda()
    import torch
    from vllm.models.minimax_m3.common.ops.index_topk import (
        minimax_m3_index_score,
    )
    from .attention import _torch_dtype
    from ..refs import MASKED_BLOCK_SCORE, visible_keys
    dt = _torch_dtype(inv.dtype)
    q, k = inputs["q"].to(dt), inputs["k"].to(dt)
    B, S, H, D = q.shape
    L = k.shape[1]
    block = int(inputs["block"])
    n_blocks = -(-L // block)
    idx_q = q.reshape(B * S, H, D).contiguous()
    # The paged index-K cache as the serving path lays it out: one 128-wide
    # page per block, sequence b's pages contiguous in the pool. Keys past
    # s_kv sit in the last page zeroed — the kernel's causal mask covers them
    # (every real query precedes them).
    cache = k.new_zeros(B * n_blocks, block, D)
    cache.reshape(B, n_blocks * block, D)[:, :L] = k
    block_table = torch.arange(B * n_blocks, dtype=torch.int32,
                               device=q.device).reshape(B, n_blocks)
    cu_q = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=q.device)
    seq_lens = torch.full((B,), L, dtype=torch.int32, device=q.device)
    prefix = torch.full((B,), L - S, dtype=torch.int32, device=q.device)

    def _canon(out):
        # [H, B·S, stride] → canonical [B, H, S, n_blocks]; the stride pad
        # and the blocks past each query's reach are the oracle's sentinel.
        s = out[:, :, :n_blocks].reshape(H, B, S, n_blocks).permute(
            1, 0, 2, 3).float()
        # Block j is unseen when its first key is: the oracle's alignment
        # (`visible_keys`) read at block granularity, not restated here.
        unseen = ~visible_keys(S, L, out.device, causal=True)[:, ::block]
        return s.masked_fill(unseen[None, None], MASKED_BLOCK_SCORE)

    return BenchCase(run=Launch(minimax_m3_index_score, idx_q, cache,
                                block_table, cu_q, seq_lens, prefix,
                                max_query_len=S, max_seq_len=L,
                                num_kv_heads=H),
                     to_canonical=_canon)


register(Provider(
    library="vllm", signature="indexer_block_score",
    kernel_impl="vllm_minimax_m3_index_score",
    import_name="vllm.models.minimax_m3.common.ops.index_topk",
    # The kernel is compiled for exactly the MiniMax-M3 geometry: the block
    # width is the module's SPARSE_BLOCK_SIZE (128) and `tl.arange(0,
    # head_dim)` a power of two.
    # The launch writes fp32 scores (see the note above); bf16 is the dtype
    # MiniMax-M3 serves its index rows at — no other is evidenced.
    serves=KernelSignature(signature="indexer_block_score",
                           activation=("bf16",), output=("fp32",),
                           phases=("prefill",), kv_block_sizes=(),
                           enum=(("s_sparse_block", (128,)),
                                 ("d_indexer_head", (128,)))),
    build=_vllm_block_score_build,
    fused_op="indexer_block_score",
))


def _torch_dsa_topk_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    logits, k = inputs["logits"], int(inputs["k"])
    return BenchCase(
        run=lambda: logits.topk(k, dim=-1).indices,
        to_canonical=lambda o: o,
    )


register(Provider(
    library="torch", signature="dsa_topk_mask",
    kernel_impl="torch_dsa_topk",
    import_name="torch",
    # The launch writes indices, which the cell keys int32 (torch.topk's are
    # int64 — the byte count of the index write is the one difference).
    serves=KernelSignature(signature="dsa_topk_mask", activation=TORCH_FLOAT,
                           output=("int32",)),
    build=_torch_dsa_topk_build,
))


def _deepgemm_indexer_build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
    """SGLang 1b2870d's paged decode logits call, excluding top-k.

    Q scales are folded into float32 gates. The apparent [pages,64,1,132]
    byte cache stores each PAGE's FP8 values followed by its float32 scales,
    not interleaved 132-byte token records. Packing, quantization and the
    schedule are untimed setup; only fp8_paged_mqa_logits is timed.
    """
    _require_cuda()
    import torch
    import deep_gemm
    from ..quant import FORMATS, quantise_groups

    q, k = inputs["q"], inputs["k"]
    batch, length, width = k.shape
    heads = q.shape[1]
    page = int(inv.kv_block_size)
    pages_per_request = (length + page - 1) // page
    padded_length = pages_per_request * page
    q8, qs = quantise_groups(q.reshape(-1, width), FORMATS["fp8_e4m3"],
                            width, "exact", q.dtype, eps=1e-10)
    encoding = "pow2" if inv.kv_format == "fp8_pow2_group128" else "exact"
    k8, ks = quantise_groups(k.reshape(-1, width), FORMATS["fp8_e4m3"],
                            width, encoding, k.dtype, eps=1e-10)
    key_bytes = torch.zeros(batch, padded_length, width, dtype=torch.uint8,
                            device=k.device)
    scales = torch.ones(batch, padded_length, dtype=torch.float32, device=k.device)
    key_bytes[:, :length] = k8.reshape(batch, length, width).view(torch.uint8)
    scales[:, :length] = ks.reshape(batch, length)
    packed = torch.cat((key_bytes.reshape(-1, page * width),
                        scales.reshape(-1, page).view(torch.uint8)), dim=1)
    from ..launch_trace import paged_byte_cache_shape
    cache = packed.reshape(paged_byte_cache_shape(k.shape, page, inv.kv_format))
    table = torch.arange(batch * pages_per_request, dtype=torch.int32,
                         device=k.device).reshape(batch, pages_per_request)
    lengths = torch.full((batch, 1), length, dtype=torch.int32, device=k.device)
    num_sms = torch.cuda.get_device_properties(k.device).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(lengths, page, num_sms)
    weights = (inputs["weights"].float() * qs.reshape(batch, heads)).contiguous()
    query = q8.reshape(batch, 1, heads, width).contiguous()
    return BenchCase(
        run=Launch(deep_gemm.fp8_paged_mqa_logits, query, cache, weights,
                   lengths, table, schedule, padded_length, clean_logits=False),
        to_canonical=lambda out: out.reshape(batch, -1)[:, :length],
    )


register(Provider(
    library="deepgemm", signature="indexer_logits",
    kernel_impl="deepgemm_fp8_paged_mqa_logits",
    import_name="deep_gemm",
    min_sm=100,
    serves=KernelSignature(signature="indexer_logits",
                           activation=("fp8_e4m3",), output=("fp32",),
                           phases=("decode",),
                           enum=(("s_query", (1,)), ("n_indexer_heads", (32,)),
                                 ("d_indexer_head", (128,)), ("kv_block_size", (64,))),
                           max_dims=(("B", 128),),
                           # `fp8_paged_mqa_logits` is paged at page_size=64,
                           # the same fact the build note above states.
                           kv_block_sizes=(64,),
                           # It reads FP8 keys with one F32 scale a 128-wide
                           # key (the 132 B record); whether a writer rounded
                           # that scale to a power of two is the writer's.
                           kv_format=("fp8_group128", "fp8_pow2_group128")),
    build=_deepgemm_indexer_build,
    paged_byte_cache_inputs=(("k", "kv_format"),),
))


# ---------------------------------------------------------------------------
# sglang's DSA indexer preparation, rope-first (GLM-5.x), at sglang v0.5.22.
# `fused_indexer_k_norm_rope_store` and `fused_indexer_q_rope_first_quant`.
# ---------------------------------------------------------------------------

#: The geometry both JIT kernels compile in: a 128-wide head and a 64-wide
#: rotation (`kHeadDim` / `kRopeDim`, indexer_k.cuh:244-245,
#: main_norm_rope.cuh:588-589), bf16 rows. Declared so a cell outside it is
#: declined rather than handed to a kernel whose matcher raises.
_INDEXER_GEOMETRY = (("d_indexer_head", (128,)), ("d_rope", (64,)))

#: Pages the store kernel compiles for: any power of two (`kPageSize`,
#: indexer_k.cuh:351); sglang sizes the index page off the scheduler's
#: `page_size` (`pool.index_page_size`, dsa_indexer.py:1007).
_INDEX_PAGES = (16, 32, 64, 128, 256)

#: One key record in the index-K cache: 128 FP8 values, then (at the page's
#: end) one fp32 scale — `kPageBytes = 132 * kPageSize` (:350).
_INDEX_VALUE_BYTES, _INDEX_SCALE_BYTES = 128, 4

#: The head count beside the indexer key in the fused `wk_weights_proj`
#: row: serving splits `kw` into the key and the head weights
#: (`kw.split([head_dim, n_heads], -1)`, dsa_indexer.py:1070-1071, the
#: projection sized `head_dim + n_heads` at :344), so the key the kernel
#: gets is a strided view — `TensorMatcher(...).with_strides({-1, 1})`
#: admits any batch stride and the kernel carries it
#: (`k_input_stride_batch = k_input.stride(0)`, indexer_k.cuh:416), a
#: runtime argument of the one compiled kernel. The walk routes this span's
#: sglang cells to GLM-5.x, whose indexer has 32 heads
#: (snapshots/zai-org__GLM-5.2.json `index_n_heads`: 32 — DeepSeek-V3.2's
#: 64 would give a 192-wide row read by the same compiled kernel).
_INDEXER_WK_HEADS = 32


def _wk_split_view(k: "Any", heads: int = _INDEXER_WK_HEADS) -> "Any":
    """`k[tokens, d_indexer_head]` as the strided view of the fused
    wk/weights projection row serving passes: one `[tokens, d + heads]`
    buffer, the key its leading columns (`stride(0) == d + heads`, last-dim
    contiguous), not a contiguous copy — the matcher takes both, but only
    the view is the launch serving makes."""
    import torch
    tokens, d = k.shape
    kw = torch.empty(tokens, d + heads, dtype=k.dtype, device=k.device)
    kw[:, :d] = k
    return kw[:, :d]


def _index_cache_rows(cache, slots, page: int):
    """The `[tokens, 128]` keys a launch left in the paged index-K cache, in
    fp32: each record's FP8 values times the scale the kernel wrote for it
    (read back, not re-derived)."""
    import torch
    block, within = slots // page, slots % page
    rows = cache[block]                                   # [tokens, page·132]
    idx = torch.arange(_INDEX_VALUE_BYTES, device=cache.device)
    values = torch.gather(rows, 1, within[:, None] * _INDEX_VALUE_BYTES
                          + idx[None, :]).contiguous().view(torch.float8_e4m3fn)
    base = page * _INDEX_VALUE_BYTES + within * _INDEX_SCALE_BYTES
    sidx = torch.arange(_INDEX_SCALE_BYTES, device=cache.device)
    scale = torch.gather(rows, 1, base[:, None] + sidx[None, :]
                         ).contiguous().view(torch.float32)
    return values.float() * scale


def _sgl_indexer_k_store_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """`fused_k_indexer_norm_rope_store` (kernels/ops/attention/dsa/
    indexer_k.py:65), as `Indexer._fused_k_prepare_and_store` calls it
    (srt/layers/attention/dsa/dsa_indexer.py:1017) on the key slice of the
    fused wk/weights row (`_wk_split_view`): the packed uint8 cache, each
    token's slot, the fp32 LayerNorm gain and bias the kernel's matcher
    requires (indexer_k.cuh:387-394), the fp32 table."""
    from .attention import _require_cuda, _torch_dtype
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.dsa.indexer_k import (
        fused_k_indexer_norm_rope_store,
    )
    k = _wk_split_view(inputs["k"].to(_torch_dtype(inv.dtype)))
    page = int(inputs["kv_block_size"])
    pages = int(inputs["index_cache"].shape[0])
    cache = torch.zeros(pages, page * (_INDEX_VALUE_BYTES + _INDEX_SCALE_BYTES),
                        dtype=torch.uint8, device=k.device)
    slots = torch.arange(k.shape[0], dtype=torch.int64, device=k.device)
    w, b = inputs["k_weight"].float(), inputs["k_bias"].float()
    table = inputs["cos_sin_cache"].float()
    positions = inputs["positions"].to(torch.int64)
    eps = float(inputs["eps"])
    return BenchCase(
        run=lambda: fused_k_indexer_norm_rope_store(
            k, cache, slots, w, b, eps, table, positions, page),
        to_canonical=lambda _o: _index_cache_rows(cache, slots, page))


def _sgl_indexer_q_build(inv: Invocation, env: Env,
                         inputs: Dict[str, Any]) -> BenchCase:
    """`fused_q_indexer_rope_first_quant` (kernels/ops/attention/dsv4/
    elementwise.py:249), as `Indexer._fused_q_prepare_and_store` calls it
    (dsa_indexer.py:1084): the head-split query — serving's own
    `q.contiguous()` (:1084), a no-op on the fresh `wq_b` row it views, and
    a contiguity the kernel hard-requires (`q_input.stride(0) ==
    H·kHeadDim`, main_norm_rope.cuh:774-787) — the head weights as the
    strided trailing slice of the fused wk/weights row (`weight.stride(0)`
    a runtime argument the kernel reads, main_norm_rope.cuh:751-755,
    `:793`), the gate scale, the table, the positions. It returns the FP8
    query and the weights scaled by each head's quantisation scale; the
    canonical query is the FP8 rows times that scale, read back off the
    weights it was folded into (`w · weight_scale · scale`,
    main_norm_rope.cuh:568)."""
    from .attention import _require_cuda, _torch_dtype
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.dsv4.elementwise import (
        fused_q_indexer_rope_first_quant,
    )
    dt = _torch_dtype(inv.dtype)
    q = inputs["q"].to(dt).contiguous()
    # The same fused row the key provider splits: the weights its trailing
    # `n_indexer_heads` columns (dsa_indexer.py:1070-1071).
    w0 = inputs["weights"].to(dt)
    tokens, heads = w0.shape
    d = int(q.shape[-1])
    kw = torch.empty(tokens, d + heads, dtype=dt, device=w0.device)
    kw[:, d:] = w0
    weights = kw[:, d:]
    ws = float(inputs["weight_scale"])
    table = inputs["cos_sin_cache"].float()
    positions = inputs["positions"].to(torch.int64)

    def canonical(out):
        q_fp8, w_out = out
        scale = w_out[..., 0] / (weights.float() * ws)        # [tokens, heads]
        return (q_fp8.float() * scale[..., None]).flatten(1)
    return BenchCase(
        run=lambda: fused_q_indexer_rope_first_quant(q, weights, ws, table,
                                                     positions),
        to_canonical=canonical)


def _rope_first_at(x, i, torch):
    """`x[tokens, heads, d]` with its leading `d_rope` channels rotated in
    fp32 and written back at `x`'s dtype: one rounding per element, as the
    kernels round, while only the rotated slice is ever widened (the floor's
    widest operand stays the cell's)."""
    from ..refs import _rope_rotate
    table = i["cos_sin_cache"]
    rd = int(table.shape[-1])
    out = x.clone()
    out[..., :rd] = _rope_rotate(x[..., :rd].float(), i["positions"], table,
                                 rd, is_neox=False).to(x.dtype)
    return out


def _torch_indexer_k_chain(i, torch):
    """The plan's norm, rotation and re-join at the cell's dtype, then the
    record as the cache keeps it (FP8 values, one scale a key)."""
    import torch.nn.functional as F
    from ..refs import fp8_rows_on_grid
    k = i["k"]
    normed = F.layer_norm(k, (k.shape[-1],), i["k_weight"], i["k_bias"], i["eps"])
    return fp8_rows_on_grid(_rope_first_at(normed[:, None, :], i, torch)[:, 0])


def _torch_indexer_q_chain(i, torch):
    from ..refs import fp8_rows_on_grid
    return fp8_rows_on_grid(_rope_first_at(i["q"], i, torch)).flatten(1)


def _register_indexer_prep() -> None:
    from .norm_act import _cell_floor
    for sig, build, chain, pages, out, record, gran, secondary in (
        # The store writes the 128-value FP8 e4m3 record with one fp32
        # scale a key (`amax / 448`, not rounded to a power of two:
        # indexer_k.cuh:329-350) — `fp8_group128` at a 128-wide key —
        # whatever the cell declares, so a cell whose cache is stored in
        # another record is not this launch's to measure — declared as the
        # launch's OUTPUT and its record (`output_format`), the cached key
        # being the tensor it writes (`RowIdentity.leaf_dtype_env` keys a
        # launch's own cache write there, not under `kv`), not declined in
        # a side channel smoke and the catalog cannot read.
        ("fused_indexer_k_norm_rope_store", _sgl_indexer_k_store_build,
         _torch_indexer_k_chain, _INDEX_PAGES, ("fp8_e4m3",),
         ("fp8_group128",), (), ()),
        # The query leaves the kernel as FP8 e4m3 rows (their scale folded
        # into the head weights): a cell billed at another output dtype is
        # The query leaves the kernel as FP8 e4m3 rows (their scale folded
        # into the head weights): a cell billed at another output dtype is
        # not this launch's to measure. And the query is not the launch's
        # only write — the weights leave it fp32, each head's quantisation
        # scale folded in (`weights_out` allocated fp32, dsv4/
        # elementwise.py:257-270) — which the one-slot output cannot say.
        # One scale per (token, head) row: the cell keys it `per-token`.
        ("fused_indexer_q_rope_first_quant", _sgl_indexer_q_build,
         _torch_indexer_q_chain, None, ("fp8_e4m3",), (),
         ("per-token",), (("weights", "fp32"),)),
    ):
        register(Provider(
            library="sglang", signature=sig, kernel_impl=f"sglang_{sig}",
            import_name=("sglang.kernels.ops.attention.dsa.indexer_k"
                         if pages else
                         "sglang.kernels.ops.attention.dsv4.elementwise"),
            # The K store's `kv` slot, as the cell carries it; a cell that
            # names none reads the activation's (`Invocation.dtype_for`).
            serves=KernelSignature(signature=sig, activation=("bf16",),
                                   dtype_slots=(("kv", ("bf16",)),) if pages else (),
                                   output=out, output_format=record,
                                   activation_granularity=gran,
                                   secondary_outputs=secondary,
                                   enum=_INDEXER_GEOMETRY,
                                   kv_block_sizes=pages),
            build=build, fused_op=sig,
        ))
        register(Provider(
            library="torch", signature=sig, kernel_impl=f"torch_{sig}",
            import_name="torch",
            # The eager chain writes the record as one contiguous row per
            # key; no page is part of what it measured. It is the kernel's
            # record (`fp8_rows_on_grid`: one exact fp32 scale a key).
            # `_cell_floor` runs the chain at the cell's float dtype and
            # writes the FP8 grid the kernel writes (`fp8_rows_on_grid`) —
            # at bf16 and fp32; at fp16 smoke finds both chains off the
            # oracle's grid (rel_err 0.064 / 0.073), so fp16 is not claimed.
            serves=KernelSignature(signature=sig, activation=("bf16", "fp32"),
                                   dtype_slots=((("kv", ("bf16", "fp32")),)
                                                if pages else ()),
                                   output=out, output_format=record,
                                   activation_granularity=gran,
                                   kv_block_sizes=() if pages else None),
            build=_cell_floor(chain), fused_op=sig,
        ))


_register_indexer_prep()


# --- CSA2 / DSA indexer-Q preparation ---------------------------------------

def _indexer_fixed_weight_scale(q) -> float:
    """softmax_scale·n_heads^-1/2, the two constants the plan folds into the
    launch (vLLM attention.py:1117-1118, sglang indexer.py:1173)."""
    return float(q.shape[-2] * q.shape[-1]) ** -0.5


def _decode_mxfp4(payload, packed_scales, tokens: int, heads: int,
                  width: int = 128):
    """Decode the two-nibble payload and packed UE8M0 words both frameworks
    hand DeepGEMM (`fused_indexer_q.py:661-669`; `fp4_indexer_rope.py:109-128`)."""
    import torch
    levels = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
    grid = torch.tensor(levels, dtype=torch.float32, device=payload.device)
    payload = payload.view(torch.uint8).reshape(tokens, heads, width // 2)
    groups = width // 32
    bytes_per_group = 16
    bytes = payload.reshape(tokens, heads, groups, bytes_per_group)
    lo = torch.gather(grid.expand(bytes.shape), -1,
                      (bytes & 0x0F).long())
    hi = torch.gather(grid.expand(bytes.shape), -1,
                      (bytes >> 4).long())
    values = torch.stack((lo, hi), dim=-1).flatten(-2)
    # One int32 per (token, head) holding the head's four UE8M0 bytes
    # little-endian (vLLM hands (T, H), sglang (T*H,)); viewing the whole
    # batch as uint8 first and cropping would keep only the first head's
    # bytes — reshape per (token, head) BEFORE the crop.
    sf = packed_scales.view(torch.uint8).reshape(tokens, heads, -1)
    sf = sf[..., :groups].to(torch.float32)
    scale = torch.exp2(sf - 127.0)
    return values * scale[..., None]


def _hadamard_rotate(x):
    """Apply the reference's orthonormal Sylvester transform to the last
    axis. This is the canonicalizer for a launch that omitted H, not a timed
    step: no extra kernel belongs in `run`."""
    from ..refs import _hadamard_matrix
    import torch
    return torch.matmul(
        x, _hadamard_matrix(x.shape[-1], x.device)) * x.shape[-1] ** -0.5


def _torch_indexer_q_quant_build(inv: Invocation, env: Env,
                                 inputs: Dict[str, Any]) -> BenchCase:
    from ..refs import get_ref
    reference = get_ref(inv.signature).reference
    return BenchCase(run=lambda: reference(inputs), to_canonical=lambda out: out)


def _span_has_hadamard(signature: str) -> bool:
    """Whether the span's own declared graph contains the transform.

    The signature name is an identity, not the switch: the plan says which
    spans have V4's Hadamard and which are V4.1's rope-only writing. Which
    ops a plan absorbs is the pattern's own answer (`NaivePattern.ops`) —
    asked here rather than walked, so the sibling authoring form (whose
    `nodes` is None) answers too.
    """
    from backend.inference.fusion_plans import FUSION_SCHEMES
    return FUSION_SCHEMES[signature].naive_pattern.runs("hadamard_transform")


def _vllm_indexer_q_hadamard_reference(inputs: Dict[str, Any]):
    """vLLM's no-H launch, compared after canonicalizing into the H basis."""
    import torch
    from ..refs import _indexer_q_rope_quant_vllm_reference
    launched = _indexer_q_rope_quant_vllm_reference(inputs)
    tokens, heads = inputs["q"].shape[:2]
    q = launched[:, :-heads].reshape(tokens, heads, -1)
    return torch.cat((_hadamard_rotate(q).reshape(tokens, -1),
                      launched[:, -heads:]), dim=-1)


def _vllm_indexer_q_quant_build(use_fp4: bool, apply_hadamard: bool):
    """The wrapper vLLM calls immediately after its indexer wq_b projection;
    q is the contiguous [tokens, heads, head_dim] view made at the call site
    (deepseek_v4/attention.py:1110-1119, deepseek_v41/:1452-1462)."""
    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        _require_cuda()
        import torch
        from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
            fused_indexer_q_rope_quant)
        from .attention import _torch_dtype
        dt = _torch_dtype(inv.dtype)
        q = inputs["q"].to(dt)
        weights = inputs["weights"].to(dt)
        fixed = _indexer_fixed_weight_scale(q)

        def canonical(out):
            q_out, weights_out = out
            if use_fp4:
                packed, packed_scale = q_out
                decoded = _decode_mxfp4(packed, packed_scale,
                                        *q.shape[:2], q.shape[-1])
            else:
                scale = weights_out.float() / (weights.float() * fixed)
                decoded = q_out.float() * scale.unsqueeze(-1)
            if apply_hadamard:
                # vLLM's V4 launch omits H on q and k. Applying it to the
                # dequantised q restores the reference function without
                # inventing a timed transform; the quantisation order remains
                # visible to the cross-check band.
                decoded = _hadamard_rotate(decoded)
            return torch.cat([decoded.reshape(q.shape[0], -1),
                              weights_out.float()], dim=-1)

        return BenchCase(
            run=Launch(fused_indexer_q_rope_quant,
                       inputs["positions"], q, inputs["cos_sin_cache"],
                       weights, q.shape[-1] ** -0.5,
                       q.shape[-2] ** -0.5, use_fp4=use_fp4),
            to_canonical=canonical,
        )
    return build


def _sgl_indexer_freqs(table, d_rope: int):
    """The cis table as sglang's production call hands it:
    `torch.view_as_real(freqs_cis).flatten(-2)` — [max_pos, d_rope] fp32 with
    real/imag interleaved (scoring.py:125-131). The wrapper JIT-specializes on
    `freqs_cis.shape[-1]` and the kernel matcher requires the interleaved
    float form (`const float*`, trailing dim kRopeDim == 64;
    fp4_indexer_rope.cuh:41, :180-182), so the concat cache's [cos | sin]
    halves are interleaved, not stacked into a complex tensor."""
    import torch
    half = int(d_rope) // 2
    table = table.float()
    return torch.stack((table[:, :half], table[:, half:]),
                       dim=-1).flatten(-2).contiguous()


def _sgl_indexer_cis(table, d_rope: int):
    """The complex cis table C4Indexer owns; its wrapper performs the
    `view_as_real(...).flatten(-2)` conversion itself."""
    import torch
    half = int(d_rope) // 2
    table = table.float()
    return torch.complex(table[:, :half], table[:, half:]).contiguous()


def _sgl_indexer_q_hadamard_build(inv: Invocation, env: Env,
                                  inputs: Dict[str, Any]) -> BenchCase:
    """sglang's default C4 query launch: `compute_q` passes the contiguous
    `[tokens, heads, 128]` wq_b view, the raw weights_proj row, the fixed
    head-weight scale, the model's complex cis table and positions."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.dsv4.elementwise import (
        fused_q_indexer_rope_hadamard_quant,
    )
    from .attention import _torch_dtype
    dt = _torch_dtype(inv.dtype)
    q = inputs["q"].to(dt).contiguous()
    weights = inputs["weights"].to(dt).contiguous()
    freqs = _sgl_indexer_cis(inputs["cos_sin_cache"], inputs["d_rope"])
    fixed = _indexer_fixed_weight_scale(q)

    def canonical(out):
        q_fp8, weights_out = out
        head_weights = weights_out.squeeze(-1).float()
        scale = head_weights / (weights.float() * fixed)
        decoded = q_fp8.float() * scale[..., None]
        return torch.cat([decoded.reshape(q.shape[0], -1),
                          head_weights], dim=-1)

    return BenchCase(
        run=Launch(fused_q_indexer_rope_hadamard_quant,
                   q, weights, fixed, freqs, inputs["positions"]),
        to_canonical=canonical,
    )


def _sgl_indexer_q_split_build(inv: Invocation, env: Env,
                               inputs: Dict[str, Any]) -> BenchCase:
    """The SM100 DeepGEMM decode branch: the production call receives q as the
    contiguous [rows, heads, 128] wq_b view and the raw bf16 weights
    (scoring.py:121-132)."""
    _require_cuda()
    import torch
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
        index_q_rope_pack_weights)
    from .attention import _torch_dtype
    dt = _torch_dtype(inv.dtype)
    q = inputs["q"].to(dt)
    weights = inputs["weights"].to(dt)
    freqs = _sgl_indexer_freqs(inputs["cos_sin_cache"], inputs["d_rope"])

    def canonical(out):
        packed, packed_scale, weights_out = out
        decoded = _decode_mxfp4(packed, packed_scale,
                                *q.shape[:2], q.shape[-1])
        return torch.cat([decoded.reshape(q.shape[0], -1),
                          weights_out.float()], dim=-1)

    return BenchCase(
        run=Launch(index_q_rope_pack_weights, q, freqs, inputs["positions"],
                   weights, _indexer_fixed_weight_scale(q)),
        to_canonical=canonical,
    )


def _register_indexer_q_quant() -> None:
    geometry = (("d_head", (128,)), ("d_rope", (64,)))
    secondary = (("weights", "fp32"),)
    for sig in ("fused_indexer_q_rope_quant",
                "fused_indexer_q_rope_quant_split"):
        vllm_hadamard = _span_has_hadamard(sig)
        register(Provider(
            library="torch", signature=sig,
            kernel_impl=f"torch_{sig}", import_name="torch",
            # The floor runs the signature's own reference, which writes the
            # record the cell keys: fp8 e4m3 with one fp32 scale a head row
            # (`per-token`), or MXFP4 — fp4 e2m1 with one ue8m0 scale per 32
            # (`per-group:32`).
            serves=KernelSignature(signature=sig,
                                   activation=("bf16", "fp16"),
                                   output=("fp8_e4m3", "fp4_e2m1"),
                                   activation_granularity=("per-token",
                                                           "per-group:32"),
                                   enum=geometry),
            build=_torch_indexer_q_quant_build, fused_op=sig,
        ))
        # The kernel name keeps the record's name (`_mxfp4`); the cell keys
        # its element dtype and scale layout, as every activation record.
        for output, record, use_fp4 in (("fp8_e4m3", "fp8_e4m3", False),
                                        ("fp4_e2m1", "mxfp4", True)):
            register(Provider(
                library="vllm", signature=sig,
                kernel_impl=f"vllm_{sig}_{record}", import_name="vllm",
                serves=KernelSignature(
                    signature=sig, activation=("bf16", "fp16"),
                    output=(output,), enum=geometry,
                    activation_granularity=(("per-group:32",)
                                            if use_fp4 else ("per-token",)),
                    secondary_outputs=secondary),
                build=_vllm_indexer_q_quant_build(use_fp4, vllm_hadamard),
                reference=(
                    _vllm_indexer_q_hadamard_reference
                    if vllm_hadamard else None),
                fused_op=sig,
            ))
    # sglang's default C4 kernel computes the graph's Hadamard and uses the
    # continuous FP8 scale in the unsplit oracle above. The FP4 form is an
    # experimental serving flag and is not a default-path provider.
    register(Provider(
        library="sglang", signature="fused_indexer_q_rope_quant",
        kernel_impl="sglang_fused_q_indexer_rope_hadamard_quant",
        import_name="sglang.kernels.ops.attention.dsv4.elementwise",
        serves=KernelSignature(
            signature="fused_indexer_q_rope_quant", activation=("bf16",),
            output=("fp8_e4m3",), enum=geometry,
            activation_granularity=("per-token",),
            secondary_outputs=secondary),
        build=_sgl_indexer_q_hadamard_build,
        fused_op="fused_indexer_q_rope_quant",
    ))
    # The split V4.1 kernel has no transform and is the SM100 decode binding
    # the plan names.
    register(Provider(
        library="sglang", signature="fused_indexer_q_rope_quant_split",
        kernel_impl="sglang_index_q_rope_pack_weights",
        import_name="sglang.kernels.ops.attention.dsv4.fp4_indexer_rope",
        serves=KernelSignature(
            signature="fused_indexer_q_rope_quant_split", activation=("bf16",),
            output=("fp4_e2m1",), activation_granularity=("per-group:32",),
            phases=("decode",), enum=geometry,
            secondary_outputs=secondary),
        min_sm=100, build=_sgl_indexer_q_split_build,
        fused_op="fused_indexer_q_rope_quant_split",
    ))


_register_indexer_q_quant()


# ---------------------------------------------------------------------------
# A Hadamard-rotated index-K cache's query side (GLM-5.3-Flash): the grafted
# `hadamard_transform` on the indexer query, standing alone in sglang and
# fused with the FP8 cast in vLLM (`fused_fwht128_quant`).
#
# sglang (audited at v0.5.22) rotates the query with `rotate_activation`
# (srt/layers/attention/dsa/dsa_indexer_kpool.py:680 / :715 →
# dsa_indexer.py:241), on CUDA the JIT `hadamard_transform(x, scale=n^-0.5)`
# of kernels/ops/quantization/hadamard.py (one launch over the rows of the
# folded `[rows, n]` view, :42-55), and quantises it in a separate `act_quant`
# launch (:1802 / :1807).
#
# vLLM (audited at v0.32.0) runs both as `fwht128_quant_fp8`
# (models/glm5next/nvidia/ops/kpool_compress.py:108): a bf16, contiguous
# `[rows, 128]` query in, FP8 e4m3 + one fp32 pow2 scale a row out
# (:121-133). It writes nothing else, so it serves only a cell that writes
# FP8 — the plan's optional `quant` member present. The graft does not
# materialise that cast yet (fusion_plans/fused_fwht128_quant.yaml), so the
# cell the plan produces today writes bf16 and stays missing here.
# ---------------------------------------------------------------------------

def _rows(inputs: Dict[str, Any], dt: Any) -> Any:
    """The cell's `[rows, n]` operand at the activation dtype, contiguous."""
    x = inputs["x"]
    return x.to(dt).reshape(-1, x.shape[-1]).contiguous()


def _sgl_hadamard_build(inv: Invocation, env: Env,
                        inputs: Dict[str, Any]) -> BenchCase:
    """sglang `rotate_activation`'s CUDA body: the JIT Hadamard over each
    row, scaled n^-1/2."""
    _require_cuda()
    from sglang.kernels.ops.quantization.hadamard import hadamard_transform
    from .attention import _torch_dtype
    x = _rows(inputs, _torch_dtype(inv.dtype))
    return BenchCase(
        run=Launch(hadamard_transform, x, x.shape[-1] ** -0.5),
        to_canonical=lambda out: out.float().reshape(inputs["x"].shape))


def _torch_hadamard_build(inv: Invocation, env: Env,
                          inputs: Dict[str, Any]) -> BenchCase:
    """The op's own function, eager: the row times the Sylvester matrix,
    scaled, written at the activation dtype."""
    import torch
    from ..refs import _hadamard_matrix
    from .attention import _torch_dtype
    dt = _torch_dtype(inv.dtype)
    x = _rows(inputs, dt)
    h = (_hadamard_matrix(x.shape[-1], x.device)
         * x.shape[-1] ** -0.5).to(dt)
    return BenchCase(
        run=Launch(torch.matmul, x, h),
        to_canonical=lambda out: out.float().reshape(inputs["x"].shape))


def _vllm_fwht_quant_build(inv: Invocation, env: Env,
                           inputs: Dict[str, Any]) -> BenchCase:
    """vLLM `fwht128_quant_fp8` as the GLM-5.3 indexer calls it
    (glm5next/common/attention.py:349-350): the query viewed `[rows, 128]`."""
    _require_cuda()
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        fwht128_quant_fp8)
    from .attention import _torch_dtype
    x = _rows(inputs, _torch_dtype(inv.dtype))

    def canonical(out):
        q_fp8, scale = out
        return (q_fp8.float() * scale).reshape(inputs["x"].shape)

    return BenchCase(run=Launch(fwht128_quant_fp8, x),
                     to_canonical=canonical)


def _torch_fwht_quant_build(inv: Invocation, env: Env,
                            inputs: Dict[str, Any]) -> BenchCase:
    """The plan's chain, eager: the rotation, and the FP8 cast when the cell
    writes FP8 (the oracle's own grid, `refs._fwht_quant_reference`)."""
    from ..refs import get_ref
    reference = get_ref("fused_fwht128_quant").reference
    return BenchCase(run=lambda: reference(inputs),
                     to_canonical=lambda out: out.float())


register(Provider(
    library="sglang", signature="hadamard_transform",
    kernel_impl="sglang_hadamard_transform",
    import_name="sglang.kernels.ops.quantization.hadamard",
    # The JIT kernel compiles per dtype; the indexer hands it bf16.
    serves=KernelSignature(signature="hadamard_transform",
                           activation=("bf16",)),
    build=_sgl_hadamard_build,
))
register(Provider(
    library="torch", signature="hadamard_transform",
    kernel_impl="torch_hadamard_transform", import_name="torch",
    serves=KernelSignature(signature="hadamard_transform",
                           activation=TORCH_FLOAT),
    build=_torch_hadamard_build,
))
register(Provider(
    library="vllm", signature="fused_fwht128_quant",
    kernel_impl="vllm_fwht128_quant_fp8",
    import_name="vllm.models.glm5next.nvidia.ops.kpool_compress",
    # The kernel asserts a bf16, contiguous, 128-wide row (:121-123) and
    # writes FP8 e4m3 only.
    serves=KernelSignature(signature="fused_fwht128_quant",
                           activation=("bf16",), output=("fp8_e4m3",),
                           enum=(("d_model", (128,)),)),
    build=_vllm_fwht_quant_build, fused_op="fused_fwht128_quant",
))
register(Provider(
    library="torch", signature="fused_fwht128_quant",
    kernel_impl="torch_fused_fwht128_quant", import_name="torch",
    serves=KernelSignature(signature="fused_fwht128_quant",
                           activation=TORCH_FLOAT,
                           output=("bf16", "fp8_e4m3")),
    build=_torch_fwht_quant_build, fused_op="fused_fwht128_quant",
))
