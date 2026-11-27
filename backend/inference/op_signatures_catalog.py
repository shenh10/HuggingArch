"""Declared op signatures — data, not the registry machinery.

`register_signature` / `OpSignature` / `TensorLayout` live in
`op_signatures.py`. This module is the catalog those functions consume;
importing it populates the registry. Keep declarations here so the
layout math and the list of ops cannot grow into one file again.
"""
from __future__ import annotations

from .fusion_plans import FUSION_SCHEMES
from .op_signatures import (COMM_DOMAIN_AXIS, OpSignature, TensorLayout,
                            register_signature)


register_signature(OpSignature(
    name="gqa_attention",
    binds_ops=('flash_attention',),
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("k", ("B", "s_kv", "n_kv_heads", "d_head"), role="kv"),
        TensorLayout("v", ("B", "s_kv", "n_kv_heads", "d_head"), role="kv"),
    ),
    # The engine lays attention operands head-first:
    # [B, n_heads, s_query, d_head].
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("k", TensorLayout("k", ("B", "n_kv_heads", "s_kv", "d_head"))),
    ),
    # Independent presence flags, none a tensor axis of the default cell:
    #   s_window      sliding-window FA (`window_size=(W-1, 0)`) vs full causal
    #   n_attn_sinks  per-head sink (`s_aux`) vs plain softmax
    #   d_v_head      V width when it is not `d_head` (MiMo partial-RoPE)
    variant_dims=("s_window", "n_attn_sinks", "d_v_head"),
    # The asymmetric-V geometry, selected by `d_v_head` (the one of the three
    # flags above that IS a tensor axis — `variant_selectors`). MiMo's partial
    # RoPE leaves V narrower than Q (128 of 192), and the default layout above
    # calls that last axis `d_head`, so the contract said 192 while the graph's
    # own `shape_in` said 128 and the builder patched the width back on after
    # the fact. Declared here instead, so the width is read from one place.
    variant_inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("k", ("B", "s_kv", "n_kv_heads", "d_head"), role="kv"),
        TensorLayout("v", ("B", "s_kv", "n_kv_heads", "d_v_head"), role="kv"),
    ),
    # No `variant_engine_inputs`: V is deliberately an operand the ENGINE
    # layout does not name (see `stored_operands`), and the identity reads the
    # narrow width off the output instead. Declaring it here would put a
    # `d_v_head` on every GQA cell's key, which is the key every published row
    # already carries.
    dtype_slots=("activation", "kv"),
    doc=(
        "Grouped-query causal attention. Q [tokens, n_heads, d_head], K/V "
        "[tokens, n_kv_heads, d_head] (n_kv_heads<=n_heads, GQA broadcast), "
        "causal mask, softmax "
        "over key positions. Prefill: full seqlen; decode: s_query=1 over "
        "s_kv keys. Binding `s_window` selects the sliding-window kernel "
        "path (FA `window_size` / FlashInfer `window_left`); `n_attn_sinks` "
        "selects a per-head sink in the softmax denominator; `d_v_head` "
        "when V is narrower than Q. A hybrid model's full-attention layers "
        "omit the window; a span that did not absorb a sink softmax "
        "omits the sink."
    ),
))

register_signature(OpSignature(
    name="fused_mla_norm_rope_attn_rope_cast",
    binds_ops=('fused_mla_norm_rope_attn_rope_cast',),
    shape_projection="dsa",
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("c", ("B", "s_kv", "d_head"), role="kv"),
        TensorLayout("indices", ("B", "s_query", "n_topk_keys"),
                     role="metadata"),
        TensorLayout("sinks", ("n_heads",), role="metadata"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    attends_at_most=("n_topk_keys",),
    # d_v_head: the quantised output row's width, the reference's own read.
    # The mega kernel's lora-group count is a worker-side launch parameter,
    # not a cell axis: no harness builder or oracle reads it.
    structural_axes=("d_rope", "d_v_head"),
    dtype_slots=("activation", "output"),
    doc=(
        "DeepInfer's V4.1 SM100 mega attention: partial Q-RoPE, sparse MLA "
        "with per-head sinks, conjugate O-RoPE, and MXFP8 output cast in "
        "one FlashMLA launch. Unlike `mla_sparse_attention`, q arrives "
        "unrotated and the output leaves quantised."
    ),
))

register_signature(OpSignature(
    name="mla_attention",
    binds_ops=('mla_attention',),
    variant_dims=("n_kv_heads",),
    # The default geometry is the absorbed one: ONE shared latent c, and the
    # value is its leading d_v_head columns — so d_v_head is structural (no
    # input tensor spans it). `n_kv_heads` selects the non-absorbed prefill
    # geometry (per-head k/v) — see `variant_dims`.
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("c", ("B", "s_kv", "d_head"), role="kv"),
    ),
    structural_axes=("d_v_head",),
    # Default (absorbed): engine q [B, n_heads, s_query, d_qk] + one shared
    # latent [B, s_kv, d_qk]; the value width is the output's last dim.
    engine_inputs=(
        (0, TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        (1, TensorLayout("c", ("B", "s_kv", "d_head"))),
    ),
    # Prefill materializes per-head K/V (asymmetric Dv). Selected when
    # `n_kv_heads` is bound — the same axis `variant_dims` already names.
    variant_inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("k", ("B", "s_kv", "n_kv_heads", "d_head"), role="kv"),
        TensorLayout("v", ("B", "s_kv", "n_kv_heads", "d_v_head"), role="kv"),
    ),
    variant_engine_inputs=(
        (0, TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        (1, TensorLayout("k", ("B", "n_kv_heads", "s_kv", "d_head"))),
        (2, TensorLayout("v", ("B", "n_kv_heads", "s_kv", "d_v_head"))),
    ),
    output_axes=("d_v_head",),
    dtype_slots=("activation", "kv"),
    doc=(
        "Multi-head latent attention (DeepSeek-style): absorbed latent "
        "KV, decode reads the compressed latent then head-broadcasts. "
        "Distinct kernel family from dense GQA (FlashMLA specializes here)."
    ),
))

register_signature(OpSignature(
    name="mla_sparse_attention",
    binds_ops=('flash_dsa',),
    shape_projection="dsa",
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("c", ("B", "s_kv", "d_head"), role="kv"),
        TensorLayout("indices", ("B", "s_query", "n_topk_keys"),
                     role="metadata"),
    ),
    # The value is the leading d_v_head columns of the shared latent c.
    structural_axes=("d_v_head",),
    # The cache is the score's K operand (the plan's `k`, input 1). Its row
    # axis is what the walk bills a query for reading — the attended keys,
    # `Min(s_kv, k)` of the score op's own top-k budget — so the cell's
    # `s_kv` is read there, by the identity and by the provider that builds
    # the cache (`kernel_bench.refs._carried_axes`) alike. The selection
    # operand is not laid out: its width is the indexer's own selection
    # (GLM-5.3-Flash selects 512 POOLS of 4 keys and attends 2051 keys).
    #
    # The latent geometry is laid out ONLY as stored — `[B, s_kv, d_head]`,
    # no head axis. That is the graph's own answer to "one shared row a
    # key": the walk marks a fan-out of a tensor stored with no head axis
    # (`traffic._Fanout.headless_store`) and the row's stored operands drop
    # it (`stored_operands`), or the graph hands the kernel the stored
    # latent itself (GLM-5.3-Flash prefill, Hy4). A rank-4 K at input 1 is
    # stored per head — the per-head geometry below — whatever widths the
    # spec declares; reading `d_kv_lora` to choose keyed a stale QSA plan's
    # per-head K as a latent and handed FlashMLA a 256-wide head.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("k", TensorLayout("c", ("B", "s_kv", "d_head"))),
    ),
    # The per-head geometry: K and V cached per KV head, each query head
    # reading its group's K and V — top-k GQA (Qwen QSA's split decode,
    # Step-5's region-sparse attention). Same span (`flash_dsa`: score,
    # softmax, apply over the selected keys), another operand list, so it is
    # this signature's variant rather than a second binding of the op, the
    # way `mla_attention` carries its per-head K/V: `n_kv_heads` selects it.
    # K is input 1 (the score's K), V input 3 (the apply's V); the selection
    # (input 2) is not laid out, as above.
    variant_dims=("n_kv_heads",),
    variant_inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("k", ("B", "s_kv", "n_kv_heads", "d_head"), role="kv"),
        TensorLayout("v", ("B", "s_kv", "n_kv_heads", "d_v_head"), role="kv"),
        TensorLayout("indices", ("B", "s_query", "n_topk_keys"),
                     role="metadata"),
    ),
    variant_engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("k", TensorLayout("k", ("B", "n_kv_heads", "s_kv", "d_head"))),
        ("v", TensorLayout("v", ("B", "n_kv_heads", "s_kv", "d_v_head"))),
    ),
    # Each query reads min(s_kv, n_topk_keys) keys, not the whole cache.
    attends_at_most=("n_topk_keys",),
    dtype_slots=("activation", "kv"),
    doc=(
        "Sparse attention over the indexer's selection. Default geometry — "
        "sparse multi-head latent attention (DeepSeek DSA / NSA): the same "
        "absorbed latent geometry as `mla_attention`, but each query attends "
        "only the `n_topk_keys` positions the lightning indexer selected, "
        "gathered out of a cache of `s_kv`. Its own kernel family — FlashMLA "
        "`flash_mla_sparse_fwd` (Hopper/Blackwell only) and FlashInfer "
        "`trtllm_batch_decode_with_kv_cache_mla` — reached through vLLM's "
        "FlashMLASparse / FlashInferMLASparse backends and sglang's NSA "
        "backend, none of which can be served by a dense attention kernel. "
        "Variant (`n_kv_heads` bound) — top-k GQA: K and V cached per KV "
        "head, each query head reading its group's selected rows."
    ),
))

register_signature(OpSignature(
    name="mla_sparse_sink_attention",
    binds_ops=('flash_dsa_sink',),
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("c", ("B", "s_kv", "d_head"), role="kv"),
        TensorLayout("indices", ("B", "s_query", "n_topk_keys"),
                     role="metadata"),
        # The sink is a learned score, not a count: it joins every head's
        # softmax denominator and is then dropped from the value sum.
        TensorLayout("sinks", ("n_heads",), role="weight"),
    ),
    # Hy4's absorbed geometry: the selected latent row's leading d_v_head
    # columns are the value (the same structural fact as plain sparse MLA).
    structural_axes=("d_v_head",),
    shape_projection="dsa",
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        # Hy4's graph broadcasts the one stored latent across the query
        # heads (`mla_dsa_gated_sink_attention`, absorbed as vLLM / sglang
        # serve it); this rank-4 layout says where that head axis is, so the
        # row's stored operands can drop it when the walk marks the store
        # headless. The cell binds only the stored form, `[B, s_kv, d_head]`
        # (`_dsa_identity`): a rank-4 cache the walk did not mark is stored
        # per head, and this signature has no per-head geometry — no cell.
        ("k", TensorLayout("c", ("B", "n_heads", "s_kv", "d_head"))),
        ("k", TensorLayout("c", ("B", "s_kv", "d_head"))),
        ("sinks", TensorLayout("sinks", ("n_heads",))),
        # The apply's value: the same stored latent's leading d_v_head
        # columns, broadcast across the query heads in the graph. Laid out
        # so the row's stored operands can drop that head axis as they do
        # the score's K; undeclared, the row published a per-head
        # [B, n_heads, s_kv, d_v_head] operand no launch reads and the
        # launch gate refused every decode build as measuring a part of it.
        ("v", TensorLayout("v", ("B", "n_heads", "s_kv", "d_v_head"))),
        ("v", TensorLayout("v", ("B", "s_kv", "d_v_head"))),
    ),
    attends_at_most=("n_topk_keys",),
    dtype_slots=("activation", "kv", "weight"),
    doc=(
        "Sparse multi-head latent attention with a per-head learnable sink: "
        "softmax([selected scores, sink]), then apply the resulting weights "
        "to the selected latent rows' leading d_v_head columns and discard "
        "the sink. Hy4 forces the FlashMLA sparse path in both frameworks; "
        "the sink makes this a different function (and a different output "
        "denominator) from plain sparse MLA."
    ),
))

register_signature(OpSignature(
    name="gqa_sparse_gated_attention",
    binds_ops=('flash_dsa_gated',),
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("k", ("B", "s_kv", "n_kv_heads", "d_head"), role="kv"),
        TensorLayout("v", ("B", "s_kv", "n_kv_heads", "d_head"), role="kv"),
        TensorLayout("indices", ("B", "s_query", "n_topk_keys"),
                     role="metadata"),
        TensorLayout("gate", ("B", "s_query", "n_heads", "d_head")),
    ),
    attends_at_most=("n_topk_keys",),
    # Projected like the ungated span (`_dsa_identity`): the keys a query
    # attends are K's rows (`Min(s_kv, k)`), not the spec's budget, which a
    # prefill shorter than it never reaches — keyed at n_topk_keys = 2048
    # over 1024 cached keys, the cell's indices outgrew what any build can
    # select and the run skipped every such job.
    shape_projection="dsa",
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("k", TensorLayout("k", ("B", "n_kv_heads", "s_kv", "d_head"))),
        ("gate", TensorLayout("gate", ("B", "n_heads", "s_query", "d_head"))),
    ),
    dtype_slots=("activation", "kv"),
    doc=(
        "Qwen QSA sparse grouped-query attention with the sigmoid output "
        "gate folded into its unsplit prefill kernel. Each query reads its "
        "selected main-KV rows from a paged cache; the kernel first rounds "
        "the attention result to the activation dtype, then applies "
        "sigmoid(gate) in fp32. GQA heads share KV heads, but unlike plain "
        "GQA the selection and gate are part of the measured function."
    ),
))

register_signature(OpSignature(
    name="gemm",
    binds_ops=('linear', 'qkv_parallel_linear', 'merged_column_parallel_linear', 'mxfp4_fused_linear', 'linear_param', 'linear_external_weight',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_reduce")),
        TensorLayout("w", ("d_out", "d_reduce"), role="weight"),
    ),
    # The engine's activation arrives unfolded as [B, s_query, d_reduce]; the
    # output width `d_out` is the last dim of shape_out (summed over fused ports).
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_reduce"))),
    ),
    output_axes=("d_out",),
    dtype_slots=("activation", "weight"),
    shape_projection="fold_gemm",
    doc=("Dense matmul: x[rows=B·s_query, d_reduce] @ "
         "w[d_out, d_reduce]^T → [rows, d_out]. fp8 engages "
         "DeepGEMM / scaled_mm; bf16 goes cuBLAS/cuBLASLt/CUTLASS."),
))

register_signature(OpSignature(
    name="grouped_gemm",
    binds_ops=('grouped_gemm',),
    inputs=(
        TensorLayout("x", ("n_experts", "s_expert_rows", "d_reduce")),
        TensorLayout("w", ("n_experts", "d_out", "d_reduce"), role="weight"),
    ),
    # The engine hands the rank's routed buffer in the same order: the bank's
    # experts lead, their mean rows next, the reduced width last. The output
    # width is the last dim of shape_out.
    engine_inputs=(
        (0, TensorLayout("x", ("n_experts", "s_expert_rows", "d_reduce"))),
    ),
    output_axes=("d_out",),
    # The routed token total is a sum over the per-expert rows, no tensor's
    # axis.
    structural_axes=("n_tokens",),
    dtype_slots=("activation", "weight"),
    doc=("MoE grouped/masked matmul: n_experts expert weights — the experts "
         "the launch runs, the ones the step's rows reach "
         "(eval.routed.RoutedLaunch.touched_experts) — "
         "w[n_experts,d_out,d_reduce], tokens grouped per expert "
         "x[n_experts,s_expert_rows,d_reduce] → [n_experts,s_expert_rows,d_out]. "
         "A distinct kernel (DeepGEMM m_grouped / fused-MoE), NOT a dense GEMM "
         "of the flattened [n_experts·s_expert_rows,d_reduce]."),
))

register_signature(OpSignature(
    name="fused_moe",
    binds_ops=('fused_moe',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
        TensorLayout("topk_ids", (("B", "s_query"), "n_experts_active"),
                     role="metadata"),
        TensorLayout("topk_weights", (("B", "s_query"), "n_experts_active"),
                     role="metadata"),
        # gate|up packed on the out axis — same split as `silu_and_mul`.
        TensorLayout("w1", ("n_experts", (2, "d_ffn"), "d_model"),
                     role="weight"),
        TensorLayout("w2", ("n_experts", "d_model", "d_ffn"), role="weight"),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_model"))),
        (0, TensorLayout("x", (("B", "s_query"), "d_model"))),
    ),
    dtype_slots=("activation", "weight"),
    doc=("The WHOLE MoE expert block in one kernel: tokens + top-k routing "
         "+ expert weights → gate_up GEMM + SwiGLU + down GEMM + weighted "
         "combine. The coarse fusion boundary (MoE-block counterpart of "
         "flash_attention), distinct from the grouped_gemm pipeline stage. "
         "FlashInfer `cutlass_fused_moe` (single-GPU); DeepGEMM MegaMoE "
         "`fp8_fp4_mega_moe` (EP + symmetric memory)."),
))

register_signature(OpSignature(
    name="moe_permute",
    binds_ops=('permute',),
    inputs=(
        # The dispatch READS the token-ordered rows and WRITES the
        # expert-ordered buffer — a gather is both row counts, so the
        # contract declares both sides: `x` token-ordered (`B·s_query`
        # rows), `idx` one routed source row per (expert, slot)
        # (`n_experts·s_expert_rows` entries into the token side).
        TensorLayout("x", (("B", "s_query"), "d_model")),
        TensorLayout("idx", (("n_experts", "s_expert_rows"),), role="metadata"),
    ),
    # The engine's operand is token-ordered `[B, s_query, d_model]`; the
    # expert-ordered geometry is the side it WRITES, declared as the output.
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_model"))),
        (0, TensorLayout("x", (("B", "s_query"), "d_model"))),
    ),
    engine_outputs=(
        (0, TensorLayout("y", ("n_experts", "s_expert_rows", "d_model"))),
    ),
    # The dispatch writes the rows it reads, in expert order: one element
    # of input 0 per element written (the arch op's `gather_source`). So a
    # quantised operand's record travels whole through it.
    gathers_from=0,
    # The routed token total the buffer holds, no tensor's axis — the count
    # a Kernel Bench curve of the boundary runs along, as grouped_gemm's.
    structural_axes=("n_tokens",),
    dtype_slots=("activation",),
    doc=("MoE token permute/scatter: gather routed tokens into expert-contiguous "
         "order before the grouped GEMM. Memory-bound (no matmul FLOPs), reads "
         "B·s_query token rows and writes n_experts·s_expert_rows routed rows "
         "of width d_model. Fused into the GEMM by triton/DeepEP; a standalone "
         "kernel in the torch-stitched realization. The routed side shares "
         "grouped_gemm's {n_experts,s_expert_rows} convention "
         "(s_expert_rows = padded rows/expert)."),
))

register_signature(OpSignature(
    name="moe_combine",
    binds_ops=('combine',),
    inputs=(
        TensorLayout("x", (("n_experts", "s_expert_rows"), "d_model")),
        TensorLayout("w", (("n_experts", "s_expert_rows"),), role="metadata"),
        TensorLayout("idx", (("n_experts", "s_expert_rows"),), role="metadata"),
    ),
    # The inverse reads the expert-ordered buffer, so the operand geometry is
    # the first operand's: `[n_expert_copies, M, d_model]`. But the launch is
    # a REDUCE back to the token side, and a boundary cell states both sides
    # of the boundary it moves (the fixed half of its byte law is the token
    # side's): the token rows are the OUTPUT's axes, which the input contract
    # does not carry, so they are structural and the engine's output layout
    # says where the walk carries them. The routed token total `n_tokens` is
    # structural too, as on grouped_gemm and the dispatch.
    engine_inputs=(
        (0, TensorLayout("x", ("n_experts", "s_expert_rows", "d_model"))),
    ),
    engine_outputs=(
        (0, TensorLayout("y", ("B", "s_query", "d_model"))),
        (0, TensorLayout("y", (("B", "s_query"), "d_model"))),
    ),
    structural_axes=("B", "s_query", "n_tokens"),
    dtype_slots=("activation",),
    doc=("MoE token combine/unpermute: weighted scatter-add of expert outputs "
         "back to token order. Reads n_experts·s_expert_rows routed rows × "
         "d_model, writes B·s_query token rows. The "
         "counterpart of moe_permute closing the expert block."),
))

register_signature(OpSignature(
    name="gated_delta_rule",
    decay_rank_variant="kda",
    binds_ops=('gated_delta_rule',),
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_lin_k_heads", "d_lin_k_head")),
        TensorLayout("k", ("B", "s_query", "n_lin_k_heads", "d_lin_k_head")),
        TensorLayout("v", ("B", "s_query", "n_lin_v_heads", "d_lin_v_head")),
        # Per-head scalar decay (log space) — KDA's per-channel form is the
        # one axis wider, see that signature.
        TensorLayout("g", ("B", "s_query", "n_lin_v_heads"), role="metadata"),
        # An activation, not a scale: every provider casts it to the cell's
        # activation dtype (`linear_ssm`: `inputs["beta"].to(dt)`) and the
        # walk bills the edge that produces it at that dtype. `metadata` is
        # the weight-SCALE slot, and naming it here made the contract say
        # fp32 for a tensor nothing runs at fp32.
        TensorLayout("beta", ("B", "s_query", "n_lin_v_heads"),
                     role="activation"),
        TensorLayout("initial_state",
                     ("B", "n_lin_v_heads", "d_lin_v_head", "d_lin_k_head"),
                     role="metadata"),
    ),
    # The engine lays the recurrence operands head-first:
    # [B, n_lin_v_heads, s_query, d_lin_v_head].
    # k is not read for dims (same axes as q); v sits at operand 2.
    engine_inputs=(
        (0, TensorLayout("q", ("B", "n_lin_k_heads", "s_query",
                               "d_lin_k_head"))),
        (2, TensorLayout("v", ("B", "n_lin_v_heads", "s_query",
                               "d_lin_v_head"))),
    ),
    dtype_slots=("activation",),
    doc=("Gated delta-rule linear attention (Gated DeltaNet / Qwen3-Next). "
         "Recurrence S_t = g_t·S_{t-1} + β_t·K_tᵀV_t; O_t = Q_t·S_t, with ONE "
         "decay scalar per head. Constant per-layer state — no growing KV. Two "
         "kernels, split by phase: the chunked `chunk_gated_delta_rule` for a "
         "full sequence, and `fused_recurrent_gated_delta_rule` for a single "
         "step. Grouped-value (GVA) when n_lin_k_heads < n_lin_v_heads. "
         "KNOWN GAP, read this before comparing rows across `framework`: the "
         "recurrence above contains NO q/k normalisation, and the two shipped "
         "frameworks now draw that boundary differently — sglang folds the "
         "l2norm into this kernel (`use_qk_l2norm_in_kernel=True`, "
         "gdn_triton.py:212) while vLLM 0.32.0 moved it into the earlier "
         "launch (`rearrange_mixed_qkv(apply_l2norm=True)`, "
         "qwen_gdn_linear_attn.py:1523) and passes False here. Every provider "
         "in this harness measures the norm INSIDE, so a vLLM row on this "
         "signature is one launch wider than vLLM's own launch. "
         "`kernel_impl` tells the rows apart, so a consumer that pins the "
         "implementation is unaffected; one that asks this signature for "
         "\"the framework kernel's time\" is comparing a kernel that "
         "normalises against one that does not. Settling it is a decision "
         "about where this span ends — declare the norm as part of it, or "
         "exclude it from both and price it separately — not a flag to flip, "
         "and either answer moves every row on the signature."),
))

register_signature(OpSignature(
    name="kda",
    binds_ops=('kda',),
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_lin_k_heads", "d_lin_k_head")),
        TensorLayout("k", ("B", "s_query", "n_lin_k_heads", "d_lin_k_head")),
        TensorLayout("v", ("B", "s_query", "n_lin_v_heads", "d_lin_v_head")),
        # Per-CHANNEL decay — the one difference from gated_delta_rule.
        TensorLayout("g", ("B", "s_query", "n_lin_v_heads", "d_lin_k_head"),
                     role="metadata"),
        # An activation, not a scale: every provider casts it to the cell's
        # activation dtype (`linear_ssm`: `inputs["beta"].to(dt)`) and the
        # walk bills the edge that produces it at that dtype. `metadata` is
        # the weight-SCALE slot, and naming it here made the contract say
        # fp32 for a tensor nothing runs at fp32.
        TensorLayout("beta", ("B", "s_query", "n_lin_v_heads"),
                     role="activation"),
        TensorLayout("initial_state",
                     ("B", "n_lin_v_heads", "d_lin_v_head", "d_lin_k_head"),
                     role="metadata"),
    ),
    # Same engine layout as gated_delta_rule:
    # [B, n_lin_v_heads, s_query, d_lin_v_head], v at operand 2.
    engine_inputs=(
        (0, TensorLayout("q", ("B", "n_lin_k_heads", "s_query",
                               "d_lin_k_head"))),
        (2, TensorLayout("v", ("B", "n_lin_v_heads", "s_query",
                               "d_lin_v_head"))),
    ),
    dtype_slots=("activation",),
    doc=("Kimi Delta Attention — delta-rule linear attention with a decay "
         "VECTOR per key channel instead of one scalar per head, so the state "
         "decays anisotropically. Same dims as `gated_delta_rule` but a "
         "separate kernel: vLLM and sglang vendor `chunk_kda` / "
         "`fused_recurrent_kda` beside the Gated-DeltaNet pair, and sglang's "
         "shared decode kernel takes an explicit `is_kda` switch."),
))

# The short conv WITH its SiLU — the fused span `causal_conv1d`
# (fusion_plans/causal_conv1d.yaml: conv1d → silu), which is what both
# frameworks launch (`causal_conv1d_fn` / `causal_conv1d_update` called with
# `activation="silu"`) and what the reference has always computed. The bare
# `conv1d` op is not bound: it is a different function (no activation), and
# the one place a bare conv survives the fusion — sglang's dilated Qwen PLE
# conv, `F.conv1d` then `F.silu` — is not this kernel at all, so measuring it
# here would key a cuDNN conv under a Triton one.
# A Gated DeltaNet DECODE step in one launch — the fused span
# `fused_recurrent_gated_delta_rule` (fusion_plans/…: q/k L2 norm, the forget
# gate g = -exp(A_log)·softplus(a + dt_bias), beta = sigmoid(b) and the
# recurrence). Unlike `gated_delta_rule`, whose kernels take g and beta
# already formed, this launch reads the RAW gate inputs a and b and the two
# per-head parameters, so it is its own cell: the contract is the operands
# the launch reads. Both frameworks' kernel takes one token per sequence
# (`fused_recurrent_gated_delta_rule_packed_decode`).
register_signature(OpSignature(
    name="fused_recurrent_gated_delta_rule",
    binds_ops=('fused_recurrent_gated_delta_rule',),
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_lin_k_heads", "d_lin_k_head")),
        TensorLayout("k", ("B", "s_query", "n_lin_k_heads", "d_lin_k_head")),
        TensorLayout("v", ("B", "s_query", "n_lin_v_heads", "d_lin_v_head")),
        # The raw gate and beta projections, one scalar per value head.
        TensorLayout("a", ("B", "s_query", "n_lin_v_heads")),
        TensorLayout("b", ("B", "s_query", "n_lin_v_heads")),
        TensorLayout("A_log", ("n_lin_v_heads",), role="weight"),
        TensorLayout("dt_bias", ("n_lin_v_heads",), role="weight"),
        TensorLayout("initial_state",
                     ("B", "n_lin_v_heads", "d_lin_v_head", "d_lin_k_head"),
                     role="metadata"),
    ),
    # The fused node's operands, in its members' order: q, k, a, dt_bias,
    # A_log, b (the gate chain) and v last (the recurrence's own input) —
    # [B, heads, s_query, d] for q and v, [B, s_query, HV] for a and b.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_lin_k_heads", "s_query",
                               "d_lin_k_head"))),
        ("dt_raw", TensorLayout("a", ("B", "s_query", "n_lin_v_heads"))),
        ("beta_raw", TensorLayout("b", ("B", "s_query", "n_lin_v_heads"))),
        ("v", TensorLayout("v", ("B", "n_lin_v_heads", "s_query",
                               "d_lin_v_head"))),
    ),
    dtype_slots=("activation",),
    doc=("Gated DeltaNet decode step, one launch: l2norm(q), l2norm(k), "
         "g = -exp(A_log)·softplus(a + dt_bias), beta = sigmoid(b), then "
         "S_t = exp(g)·S_{t-1} + beta·k(v − S_{t-1}k)ᵀ, o = S_t·q. Reads the "
         "post-conv q|k|v, raw a/b, A_log, dt_bias and the recurrent state; "
         "writes o and the state. vLLM / sglang "
         "`fused_recurrent_gated_delta_rule_packed_decode`."),
))

# The KDA decode step in one launch — the fused span `fused_recurrent_kda`
# (fusion_plans/…: the bounded per-channel forget gate
# g = lower_bound·sigmoid(exp(A_log)·(f + dt_bias)) formed from the raw
# low-rank projection f, beta = sigmoid(b), q/k L2 norm and the recurrence).
# Its own cell for the reason the GDN step is: the launch reads raw f and b
# and the per-head rate / per-channel bias, not a formed decay. `A_log` is per
# head and `dt_bias` per channel.
#
# Two plans land on this one cell. GLM-5.3-Flash's checkpoint stores the rate
# per head (`fused_recurrent_kda`); Kimi-K3's stores a `[d_lin_k_head]` vector
# that its graph multiplies in per channel (`fused_recurrent_kda_per_channel`),
# but both frameworks READ it per head — vLLM `A_log + i_h` and an
# `A_log.numel() == H` check (models/kimi_k3/nvidia/ops/third_party/kda/
# fused_recurrent.py:557, :631), sglang `p_A_log` one scalar a head
# (fused_sigmoid_gating_recurrent.py:322) — so the launch computes this
# cell's function on this cell's operands either way. Operands as the fused node emits them:
# q [B, H, S, K], k, f [B, S, HV·K], dt_bias [HV·K], A_log [HV], b [B, S, HV],
# v [B, HV, S, V].
register_signature(OpSignature(
    name="fused_recurrent_kda",
    binds_ops=('fused_recurrent_kda', 'fused_recurrent_kda_per_channel'),
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_lin_k_heads", "d_lin_k_head")),
        TensorLayout("k", ("B", "s_query", "n_lin_k_heads", "d_lin_k_head")),
        TensorLayout("v", ("B", "s_query", "n_lin_v_heads", "d_lin_v_head")),
        # The raw gate projection, one value per key channel of each head,
        # still FLAT as the fused node reads it (`f_b_proj`'s output; the
        # head split `f_unflatten` is inside the span), and its per-channel
        # bias the same flat width.
        TensorLayout("f", ("B", "s_query", ("n_lin_v_heads", "d_lin_k_head"))),
        TensorLayout("b", ("B", "s_query", "n_lin_v_heads")),
        TensorLayout("A_log", ("n_lin_v_heads",), role="weight"),
        TensorLayout("dt_bias", (("n_lin_v_heads", "d_lin_k_head"),),
                     role="weight"),
        TensorLayout("initial_state",
                     ("B", "n_lin_v_heads", "d_lin_v_head", "d_lin_k_head"),
                     role="metadata"),
    ),
    # The fused node's operands, in its members' order: q, k, f, dt_bias,
    # A_log, b, v — q (position 0) and v (position 6) head-first, the raw
    # gate f (position 2) flat; every one checked against its rank and
    # extent when the cell is zipped.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_lin_k_heads", "s_query",
                               "d_lin_k_head"))),
        ("forget_raw", TensorLayout("f", ("B", "s_query",
                               ("n_lin_v_heads", "d_lin_k_head")))),
        ("beta_raw", TensorLayout("b", ("B", "s_query", "n_lin_v_heads"))),
        ("v", TensorLayout("v", ("B", "n_lin_v_heads", "s_query",
                               "d_lin_v_head"))),
    ),
    dtype_slots=("activation",),
    doc=("Kimi Delta Attention decode step, one launch: l2norm(q), "
         "l2norm(k), g = lower_bound·sigmoid(exp(A_log)·(f + dt_bias)) per "
         "key channel, beta = sigmoid(b), then the delta-rule recurrence "
         "with a per-channel decay. vLLM `fused_recurrent_kda(compute_gate="
         "True)` (GLM-5.3-Flash) / sglang `fused_sigmoid_gating_delta_rule_"
         "update(is_kda=True)`."),
))

# Kimi-K3's WHOLE KDA mixer for one decode token, one launch — the fused span
# `fused_kda_decode` (fusion_plans/fused_kda_decode.yaml): the three causal
# short convs with their SiLU on the raw q/k/v projections (each conv state
# shifted in place), q/k L2 norm, the bounded per-channel forget gate from the
# raw low-rank projection, beta = sigmoid(b), the delta-rule recurrence, and
# the gated output RMSNorm (rmsnorm(o)·w·sigmoid(g)). The operands are the
# fused node's (q, k, v, f, dt_bias, A_log, b, g in that order) plus the two
# pools the launch updates and the weights it reads that no node hands over.
# The convs' width rides the conv weights and is read from the spec.
register_signature(OpSignature(
    name="fused_kda_decode",
    binds_ops=('fused_kda_decode',),
    inputs=(
        TensorLayout("q", ("B", "s_query", ("n_lin_k_heads", "d_lin_k_head"))),
        TensorLayout("k", ("B", "s_query", ("n_lin_k_heads", "d_lin_k_head"))),
        TensorLayout("v", ("B", "s_query", ("n_lin_v_heads", "d_lin_v_head"))),
        TensorLayout("f", ("B", "s_query", ("n_lin_v_heads", "d_lin_k_head"))),
        TensorLayout("dt_bias", (("n_lin_v_heads", "d_lin_k_head"),),
                     role="weight"),
        TensorLayout("A_log", ("n_lin_v_heads",), role="weight"),
        TensorLayout("b", ("B", "s_query", "n_lin_v_heads")),
        TensorLayout("g", ("B", "s_query", ("n_lin_v_heads", "d_lin_v_head"))),
        TensorLayout("q_conv_w", (("n_lin_k_heads", "d_lin_k_head"),
                                  "s_conv_kernel"), role="weight"),
        TensorLayout("k_conv_w", (("n_lin_k_heads", "d_lin_k_head"),
                                  "s_conv_kernel"), role="weight"),
        TensorLayout("v_conv_w", (("n_lin_v_heads", "d_lin_v_head"),
                                  "s_conv_kernel"), role="weight"),
        TensorLayout("o_norm_w", ("d_lin_v_head",), role="weight"),
        # The conv pool's taps (`s_conv_kernel − 1` of them, no axis of its
        # own) for every q|k|v channel of each sequence.
        TensorLayout("conv_state",
                     ("B", ("+", (2, "n_lin_k_heads", "d_lin_k_head"),
                            ("n_lin_v_heads", "d_lin_v_head")), None),
                     role="metadata"),
        TensorLayout("initial_state",
                     ("B", "n_lin_v_heads", "d_lin_v_head", "d_lin_k_head"),
                     role="metadata"),
    ),
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "s_query",
                               ("n_lin_k_heads", "d_lin_k_head")))),
        ("v", TensorLayout("v", ("B", "s_query",
                               ("n_lin_v_heads", "d_lin_v_head")))),
        ("beta_raw", TensorLayout("b", ("B", "s_query", "n_lin_v_heads"))),
    ),
    dtype_slots=("activation",),
    doc=("Kimi-K3 KDA decode mixer, one launch: silu(causal_conv1d_update) on "
         "q, k and v, l2norm(q), l2norm(k), g = lower_bound·sigmoid(exp(A_log)"
         "·(f + dt_bias)), beta = sigmoid(b), the delta-rule step, then "
         "rmsnorm(o)·w·sigmoid(g_out) per head. vLLM `_custom_ops."
         "fused_kda_decode` / sglang `kda_fused_decode`. Memory-bound: the "
         "raw projections, both state pools and the output."),
))

# The Gated DeltaNet gates alone, one launch — the fused span
# `fused_gdn_gating` (g = -exp(A_log)·softplus(a + dt_bias), beta =
# sigmoid(b), per value head), which sglang runs before its chunked prefill.
# Two outputs, both fp32 (the plan's `output_dtype`).
register_signature(OpSignature(
    name="fused_gdn_gating",
    binds_ops=('fused_gdn_gating',),
    inputs=(
        # The raw projections, token-major as the kernel takes them.
        TensorLayout("a", (("B", "s_query"), "n_lin_v_heads")),
        TensorLayout("b", (("B", "s_query"), "n_lin_v_heads")),
        TensorLayout("A_log", ("n_lin_v_heads",), role="weight"),
        TensorLayout("dt_bias", ("n_lin_v_heads",), role="weight"),
    ),
    # The fused node's operands: a, dt_bias, A_log, b.
    engine_inputs=(
        ("dt_raw", TensorLayout("a", ("B", "s_query", "n_lin_v_heads"))),
    ),
    dtype_slots=("activation",),
    doc=("Gated DeltaNet gate prep, one launch: g = -exp(A_log)·softplus("
         "a + dt_bias) and beta = sigmoid(b) per value head per token, both "
         "written fp32. sglang `fused_gdn_gating` (prefill). Memory-bound: "
         "a and b in, g and beta out."),
))

# The KDA prefill gates in one launch — the fused span `fused_kda_gate`
# (fusion_plans/fused_kda_gate.yaml), the KDA sibling of `fused_gdn_gating`:
# the bounded per-channel forget gate g = lower_bound·sigmoid(exp(A_log)·
# (f + dt_bias)) formed from the raw low-rank projection f, and beta =
# sigmoid(b), both written fp32. sglang's launch (`kda_gate_chunk_cumsum`)
# goes one step past the gate formation: it chunk-cumsums the gate in place
# (the chunk pipeline's plain `chunk_local_cumsum` launch it replaces) and
# scales by RCP_LN2, the log2 conversion the chunk kernels read the decay in.
# The launch's chunk (64, `chunk_kda_fwd`'s fixed `chunk_size`) and that
# scale are launch constants, not cell axes; the bound rides the inputs
# (`refs.KDA_LOWER_BOUND`), as it does for `fused_recurrent_kda`.
register_signature(OpSignature(
    name="fused_kda_gate",
    binds_ops=('fused_kda_gate',),
    inputs=(
        # The raw gate projection, one value per key channel of each head,
        # still FLAT as the fused node reads it (`f_b_proj`'s output; the
        # head split `f_unflatten` is inside the span), token-major as the
        # kernel's [B, T, H, K] takes it.
        TensorLayout("f", ("B", "s_query", ("n_lin_v_heads", "d_lin_k_head"))),
        TensorLayout("b", ("B", "s_query", "n_lin_v_heads")),
        TensorLayout("A_log", ("n_lin_v_heads",), role="weight"),
        TensorLayout("dt_bias", (("n_lin_v_heads", "d_lin_k_head"),),
                     role="weight"),
    ),
    # The fused node's operands, in its members' order: f, dt_bias, A_log, b.
    # b is read first so `n_lin_v_heads` is bound before f's packed dim is
    # split (A_log is not a layout: Kimi-K3's checkpoint stores it per
    # channel where GLM's is per head — see `fused_recurrent_kda`).
    engine_inputs=(
        ("beta_raw", TensorLayout("b", ("B", "s_query", "n_lin_v_heads"))),
        ("forget_raw", TensorLayout("f", ("B", "s_query",
                               ("n_lin_v_heads", "d_lin_k_head")))),
    ),
    dtype_slots=("activation",),
    doc=("Kimi Delta Attention prefill gate prep, one launch: g = lower_bound"
         "·sigmoid(exp(A_log)·(f + dt_bias)) per key channel, chunk-cumsummed"
         " per 64-token chunk and scaled to log2 (×RCP_LN2), and beta = "
         "sigmoid(b), both written fp32 — the form the chunk kernels read. "
         "sglang `kda_gate_chunk_cumsum` (prefill). Memory-bound: f and b in,"
         " gate and beta out."),
))

register_signature(OpSignature(
    name="fused_gdn_post_conv_prep",
    binds_ops=('fused_gdn_post_conv_prep',),
    inputs=(
        # The convolved q|k|v row, still in the producer's split order.
        TensorLayout("qkv", ("B", "s_query",
                             ("+", (2, "n_lin_k_heads", "d_lin_k_head"),
                              ("n_lin_v_heads", "d_lin_v_head")))),
        TensorLayout("a", ("B", "s_query", "n_lin_v_heads")),
        TensorLayout("b", ("B", "s_query", "n_lin_v_heads")),
        TensorLayout("A_log", ("n_lin_v_heads",), role="weight"),
        TensorLayout("dt_bias", ("n_lin_v_heads",), role="weight"),
    ),
    engine_inputs=(
        ("dt_raw", TensorLayout("a", ("B", "s_query", "n_lin_v_heads"))),
        ("beta_raw", TensorLayout("b", ("B", "s_query", "n_lin_v_heads"))),
    ),
    dtype_slots=("activation", "weight"),
    doc=("Gated DeltaNet prefill preparation in one launch: split the "
         "convolved q|k|v row, l2-normalise q and k, copy v, and form "
         "g=-exp(A_log)·softplus(a+dt_bias) and beta=sigmoid(b). The "
         "l2-norm epsilon is fixed at 1e-6 by the kernel. q is the primary "
         "output; k, v, fp32 g and fp32 beta are secondary writes."),
))

register_signature(OpSignature(
    name="causal_conv1d",
    # The fused launch only: both frameworks run the conv with its SiLU
    # folded in (`fusion_plans/causal_conv1d.yaml`) and the providers below
    # measure THAT kernel. The bare `conv1d` op is NOT bound here — its one
    # remaining site is sglang's dilated Qwen PLE conv, which really is
    # `F.conv1d` then `F.silu`, two launches this signature would misprice.
    binds_ops=('causal_conv1d',),
    inputs=(
        TensorLayout("x", ("B", "d_conv_channels", "s_query")),
        TensorLayout("weight", ("d_conv_channels", "s_conv_kernel"),
                     role="weight"),
        TensorLayout("bias", ("d_conv_channels",), role="weight"),
    ),
    # The conv's tap spacing (the conv1d node's own `dilation`). No tensor
    # carries it, and it chooses the kernel: the frameworks' causal-conv
    # kernels take none, so vLLM runs Qwen PLE's dilated conv in `ple_conv`
    # instead. Without it in the cell that conv (C=10240, width 4 in
    # Qwen3.8-Flash-Next) would share the gated-delta-net conv's cell of the
    # same width and channel count, and be served its measurement.
    structural_axes=("dilation",),
    # Engine is feature-last `[B, s_query, C]`; the kernels (and the
    # contract) want `[B, C, s_query]`. `s_conv_kernel` is on the weight,
    # so identity still reads it from the spec — not a tensor axis of
    # `shape_in`.
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_conv_channels"))),
    ),
    dtype_slots=("activation",),
    doc=("Depthwise causal short convolution over the sequence, width 3–4, one "
         "filter per channel, with the SiLU after it folded in — the mixer "
         "every Mamba / Gated-DeltaNet / KDA layer runs on q/k/v before its "
         "recurrence. `causal_conv1d_fn` over a full sequence; "
         "`causal_conv1d_update` for one step against a conv state ring "
         "buffer, both with `activation=\"silu\"`. `dilation` spaces the taps "
         "(1 for the mixers). Memory-bound: no reuse beyond the kernel width."),
))

register_signature(OpSignature(
    name="conv2d",
    binds_ops=('conv2d',),
    inputs=(
        TensorLayout("x", ("B", "s_conv_grid", "C_in")),
        TensorLayout("weight", ("C_out", "C_in", "kernel_size", "kernel_size"),
                     role="weight"),
        TensorLayout("bias", ("C_out",), role="weight"),
    ),
    # `s_conv_grid` is the flattened H·W square patch grid; the reference
    # reshapes `x` to `[B, C_in, H, W]` for `F.conv2d`. `kernel_size` rides the
    # weight, so it is recovered from the tensor; `stride` is no tensor's axis,
    # so it is the one structural axis the identity reads from the spec. No
    # `engine_inputs`: the encode walk's rows are the grid, flattened
    # (`_conv2d_identity`), not a layout permute.
    output_axes=("C_out",),
    structural_axes=("stride",),
    dtype_slots=("activation", "weight"),
    shape_projection="conv2d",
    doc=("Full (non-depthwise) 2-D convolution over a flattened square patch "
         "grid: a strided k×k kernel (padded k//2) contracts C_in→C_out and "
         "halves each spatial side (output ceil(H/stride) per dim). Step-3.7's "
         "ViT downsamplers — 3×3 stride-2 overlapping convs before the projector."),
))

register_signature(OpSignature(
    name="mamba2_scan",
    binds_ops=('mamba2_scan',),
    inputs=(
        TensorLayout("x", ("B", "s_query", "n_ssm_heads", "d_ssm_head")),
        TensorLayout("dt", ("B", "s_query", "n_ssm_heads"), role="metadata"),
        TensorLayout("A", ("n_ssm_heads",), role="metadata"),
        TensorLayout("B", ("B", "s_query", "n_ssm_groups", "d_ssm_state")),
        TensorLayout("C", ("B", "s_query", "n_ssm_groups", "d_ssm_state")),
        TensorLayout("D", ("n_ssm_heads",), role="metadata"),
        TensorLayout("dt_bias", ("n_ssm_heads",), role="metadata"),
    ),
    # The engine's hidden/B operands are already in contract order; the
    # group/state axes are read from the B operand at position 3.
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "n_ssm_heads", "d_ssm_head"))),
        (3, TensorLayout("B", ("B", "s_query", "n_ssm_groups", "d_ssm_state"))),
    ),
    # The chunk the scan tiles the sequence into. No tensor carries it and it
    # is not a scalar the launch merely takes: the kernel runs
    # `ceil(s_query / ssm_chunk_size)` tiles and the intra-chunk term is
    # quadratic in it, so two chunk lengths are two amounts of work and two
    # measurements. In the identity for that reason. The value is the
    # deployment's, read off the checkpoint config at the fields the
    # framework binding names (`Program.scan_chunk_size`).
    structural_axes=("ssm_chunk_size",),
    dtype_slots=("activation",),
    doc=("Mamba-2 selective state-space scan (mamba_chunk_scan_combined). "
         "Constant per-layer state [n_ssm_heads,d_ssm_head,d_ssm_state], not "
         "growing KV. Nemotron-H / "
         "Zamba2 / Granite-H."),
))

register_signature(OpSignature(
    name="mamba2_rmsnorm_gated",
    binds_ops=('mamba2_rmsnorm_gated',),
    inputs=(
        TensorLayout("x", ("B", "s_query",
                           ("n_ssm_heads", "d_ssm_head"))),
        TensorLayout("gate", ("B", "s_query",
                              ("n_ssm_heads", "d_ssm_head"))),
        TensorLayout("weight", (("n_ssm_heads", "d_ssm_head"),),
                     role="weight"),
    ),
    # One RMS per `n_ssm_groups` group of the full Mamba-2 intermediate
    # width; production's gate is a strided slice of a wider projection row.
    # `d_ssm_state` does not choose the arithmetic, but it does size the B/C
    # columns between gate and dt in serving's packed projection row, and so
    # the stride the kernel is handed.
    structural_axes=("n_ssm_groups", "d_ssm_state"),
    engine_inputs=(
        ("x", TensorLayout("x", ("B", "s_query",
                               ("n_ssm_heads", "d_ssm_head")))),
        ("gate", TensorLayout("gate", ("B", "s_query",
                                  ("n_ssm_heads", "d_ssm_head")))),
    ),
    dtype_slots=("activation", "weight"),
    doc=("Mamba-2's grouped gated RMSNorm, one launch: rmsnorm(x·silu(gate)) "
         "per n_ssm_heads/n_ssm_groups group, then multiply the per-channel "
         "gain. sglang `Mixer2RMSNormGated.forward_cuda`."),
))

# `fused_rmsnorm_quant` is the same launch writing its row already quantised
# (the plan's required `quantise_activation` member): one norm over the same
# operands, told apart from the plain norm by the cell's `output` dtype and
# activation granularity, exactly as `fused_add_rmsnorm`'s optional cast is.
# vLLM's `rms_norm_per_block_quant` / `rms_norm_dynamic_per_token_quant`
# providers below serve those cells; the plain ones serve the BF16 output.
register_signature(OpSignature(
    name="rmsnorm",
    binds_ops=('rmsnorm', 'fused_rmsnorm_quant'),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
        TensorLayout("w", ("d_model",), role="weight"),
    ),
    # The engine's per-token form is [B, s_query, d_model] — the unfolded pair.
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation",),
    shape_projection="fold_rmsnorm",
    doc=("Root-mean-square norm over the last dim (d_model) of "
         "[B·s_query, d_model]. Memory-bound."),
))

register_signature(OpSignature(
    name="fused_add_rmsnorm",
    binds_ops=('fused_add_rmsnorm',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
        TensorLayout("residual", (("B", "s_query"), "d_model")),
        TensorLayout("w", ("d_model",), role="weight"),
    ),
    engine_inputs=(
        ("x", TensorLayout("x", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation",),
    shape_projection="fold_rmsnorm",
    doc=("residual += x; rmsnorm(residual) — one fused kernel over "
         "[B·s_query, d_model]."),
))

# DeepSeek-V4.1's sibling RMSNorm launch in the form that writes q already
# MXFP8: one grid over (padded token rows, q|kv). The q row is normalized,
# rounded at the materialized activation boundary, and quantized in 32-value
# groups with FlashInfer's swizzled UE8M0 scales; the KV row stays in the
# activation dtype.
register_signature(OpSignature(
    name="fused_q_kv_rmsnorm_quant",
    binds_ops=('fused_q_kv_rmsnorm_quant',),
    inputs=(
        TensorLayout("q", (("B", "s_query"), "d_q_lora")),
        TensorLayout("kv", (("B", "s_query"), "d_kv_lora")),
        TensorLayout("q_weight", ("d_q_lora",), role="weight"),
        TensorLayout("kv_weight", ("d_kv_lora",), role="weight"),
    ),
    # `_split_qkv_and_norm` splits the fused projection as qr first, kv
    # second, and hands those strided row views to the wrapper.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "s_query", "d_q_lora"))),
        ("kv", TensorLayout("kv", ("B", "s_query", "d_kv_lora"))),
    ),
    dtype_slots=("activation", "weight", "output"),
    doc=("Two sibling RMSNorms in one launch, writing q as MXFP8 values plus "
         "F8_128x4 UE8M0 scales and kv in its input dtype. vLLM "
         "`fused_q_kv_rmsnorm_quant` (query_quant.py:90; kernel :37, call "
         "deepseek_v41/attention.py:787-793). Memory-bound."),
))

# The linear-attention output norm gated by an activated branch, one launch
# (fusion_plans/rmsnorm_gated.yaml: rmsnorm(x) * act(z), act ∈ {silu,
# sigmoid}). Two gated-norm kernels per framework serve it — FLA's
# `layer_norm_fwd` with `z` (vLLM `RMSNormGated`, sglang `layernorm_gated.RMSNorm`:
# Qwen gated-delta-net, Ling) and the KDA family's `rms_norm_gated` (vLLM /
# sglang `FusedRMSNormGated`: Kimi-K3, GLM-5.3-Flash) — on the same operands.
# Per-head norms (`[B, n_heads, s_query, d_head]`) fold the heads into the
# rows exactly as the plain per-head rmsnorm does (`fold_rmsnorm`), so the
# cell is the per-token one.
#
# Which activation the gate takes (silu or sigmoid) is a kernel constexpr
# that moves no byte and changes no launch, so it is one cell, measured at
# the kernels' own default (`swish`); see the reference.
register_signature(OpSignature(
    name="rmsnorm_gated",
    binds_ops=('rmsnorm_gated',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
        TensorLayout("z", (("B", "s_query"), "d_model")),
        TensorLayout("w", ("d_model",), role="weight"),
    ),
    engine_inputs=(
        ("x", TensorLayout("x", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation",),
    shape_projection="fold_rmsnorm",
    doc=("Gated RMSNorm, norm before gate: rmsnorm(x)·w · act(z) over "
         "[B·s_query, d_model], one Triton launch (FLA `layer_norm_fwd` with "
         "`z` / `rms_norm_gated`). The output norm of every Gated-DeltaNet, "
         "KDA and GLA mixer. Memory-bound: x and z in, y out."),
))

# The MTP block's entry in one launch (fusion_plans/fused_eh_norm.yaml): the
# next token's embedding and the previous hidden state, each RMS-normalised
# with its own gain and written side by side as the `[tokens, 2·d_model]` row
# `eh_proj` reads. Its own cell rather than two `rmsnorm` rows: one launch
# reads both inputs once and writes the concatenated row once.
register_signature(OpSignature(
    name="fused_eh_norm",
    binds_ops=('fused_eh_norm',),
    inputs=(
        TensorLayout("inputs_embeds", (("B", "s_query"), "d_model")),
        TensorLayout("hidden", (("B", "s_query"), "d_model")),
        TensorLayout("enorm_w", ("d_model",), role="weight"),
        TensorLayout("hnorm_w", ("d_model",), role="weight"),
    ),
    # The fused node's operands in the concat's order: the embedding, then
    # the previous hidden state, both `[B, s_query, d_model]`.
    engine_inputs=(
        ("inputs_embeds", TensorLayout("inputs_embeds", ("B", "s_query", "d_model"))),
        ("hidden", TensorLayout("hidden", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation",),
    doc=("MTP entry, one launch: concat(rmsnorm(inputs_embeds)·enorm_w, "
         "rmsnorm(hidden)·hnorm_w) → [B·s_query, 2·d_model]. vLLM "
         "`deepseek_v32.common.kernels.fused_eh_norm` (Triton) / sglang "
         "`kernels.ops.layernorm.fused_eh_norm` (JIT). Memory-bound: two "
         "rows in, one double-width row out."),
))

# The query side's and the key side's norms of one attention, each with its
# own gain, in one launch (fusion_plans/fused_q_kv_rmsnorm.yaml). Two
# geometries of the one pattern, told apart by what the cell carries:
#
#   * the MLA a-norms (default): the q-LoRA norm over `d_q_lora` and the
#     latent-KV norm over `d_kv_lora` (DeepSeek-V4's latent row is its
#     `d_head`, 512 wide — the same slot), one row per token each. Two
#     widths, so two axes: vLLM's grid is (tokens, 2) with a block sized to
#     the wider row.
#   * the per-head Q/K norms (variant): every head of q `[tokens, n_heads,
#     d_head]` and of k `[tokens, n_kv_heads, d_head]` normalised over its
#     `d_head` with one `[d_head]` gain per side — sglang's in-place
#     `fused_inplace_qknorm` (Qwen3 / Qwen3-MoE / GLM-4-MoE).
#
# The per-head axes (`n_heads`, `n_kv_heads`, `d_head`) are on no operand of
# the row form, so their presence is what selects the variant
# (`variant_selectors`); a row cell never carries them, because the walk's
# operand list is rank 3 there and rank 4 here, and the identity zips the
# rank-4 engine layout first.
register_signature(OpSignature(
    name="fused_q_kv_rmsnorm",
    binds_ops=('fused_q_kv_rmsnorm',),
    inputs=(
        TensorLayout("q", (("B", "s_query"), "d_q_lora")),
        TensorLayout("kv", (("B", "s_query"), "d_kv_lora")),
        TensorLayout("q_weight", ("d_q_lora",), role="weight"),
        TensorLayout("kv_weight", ("d_kv_lora",), role="weight"),
    ),
    # Each operand read at the port it fills (`q`, `kv`): both are rank 3,
    # and which comes first in a layer's graph is no part of the contract. A
    # row whose two have unequal token counts (no launch of this kernel: it
    # asserts equal rows) fails to zip and is dropped, named.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "s_query", "d_q_lora"))),
        ("kv", TensorLayout("kv", ("B", "s_query", "d_kv_lora"))),
    ),
    variant_dims=("n_heads", "n_kv_heads", "d_head"),
    # The per-head form, one row of heads per token as the kernel takes it.
    # The names stay the contract's (`kv` is the key side's operand, as the
    # plan's `io_mapping` names it), so one oracle and one floor normalise
    # over the last axis in either geometry.
    variant_inputs=(
        TensorLayout("q", (("B", "s_query"), "n_heads", "d_head")),
        TensorLayout("kv", (("B", "s_query"), "n_kv_heads", "d_head")),
        TensorLayout("q_weight", ("d_head",), role="weight"),
        TensorLayout("kv_weight", ("d_head",), role="weight"),
    ),
    # The head-split q and k, head-first, as the attention leaf carries them
    # (the same operands `gemma_qkv_rmsnorm` reads).
    variant_engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("kv", TensorLayout("kv", ("B", "n_kv_heads", "s_query", "d_head"))),
    ),
    dtype_slots=("activation",),
    doc=("Two sibling RMSNorms in one launch, each written in the activation "
         "dtype: rmsnorm(q)·q_weight and rmsnorm(kv)·kv_weight, over the "
         "last axis. Row form: q over d_q_lora and kv over d_kv_lora — vLLM "
         "`models.common.ops.fused_qk_rmsnorm.fused_q_kv_rmsnorm` "
         "(DeepSeek-V4 / V4.1, Kimi-K3, GLM-5.3-Flash). Per-head form "
         "(n_heads / n_kv_heads / d_head bound): every head of q and of k "
         "over d_head with a [d_head] gain per side — sglang "
         "`kernels.ops.layernorm.norm.fused_inplace_qknorm` (Qwen3, "
         "Qwen3-MoE, GLM-4-MoE), in place. Memory-bound."),
))

# Gemma-4's per-head Q, K and V RMSNorms in one in-place launch
# (fusion_plans/gemma_qkv_rmsnorm.yaml): q and k each with their own gain,
# v with none. The k_eq_v full-attention layers read K's projection as V,
# which the launch reads as a second slice of the same width — the contract's
# own `v`.
register_signature(OpSignature(
    name="gemma_qkv_rmsnorm",
    binds_ops=('gemma_qkv_rmsnorm',),
    inputs=(
        TensorLayout("q", (("B", "s_query"), ("n_heads", "d_head"))),
        TensorLayout("k", (("B", "s_query"), ("n_kv_heads", "d_head"))),
        TensorLayout("v", (("B", "s_query"), ("n_kv_heads", "d_head"))),
        TensorLayout("q_weight", ("d_head",), role="weight"),
        TensorLayout("k_weight", ("d_head",), role="weight"),
    ),
    # The head-split q and k, head-first, as the attention leaf carries them.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("k", TensorLayout("k", ("B", "n_kv_heads", "s_query", "d_head"))),
    ),
    dtype_slots=("activation",),
    doc=("Gemma-4 attention norms, one in-place launch: per head of d_head, "
         "rmsnorm(q)·q_weight, rmsnorm(k)·k_weight and rmsnorm(v) (no gain). "
         "sglang `kernels.ops.layernorm.gemma4_fused_ops.gemma_qkv_rmsnorm`. "
         "Memory-bound: q, k, v in and out."),
))


# Gemma-4's decoder-layer tails, one launch each: the post-FFN norm, the
# residual add and the layer scalar (`gemma_rmsnorm_residual_scalar`); and the
# MoE layers' form, where the dense and the expert branch are each normed, the
# sum normed again, then added and scaled (`gemma_dual_…`). Every operand of
# a row is one `[tokens, d_model]` stream (plus the scalar), so the first one
# carries the whole cell.
register_signature(OpSignature(
    name="gemma_rmsnorm_residual_scalar",
    binds_ops=('gemma_rmsnorm_residual_scalar',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
        TensorLayout("residual", (("B", "s_query"), "d_model")),
        TensorLayout("w", ("d_model",), role="weight"),
        TensorLayout("scalar", (1,), role="weight"),
    ),
    engine_inputs=(
        ("x", TensorLayout("x", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation",),
    doc=("Gemma-4 block tail, one launch: (rmsnorm(x)·w + residual)·scalar "
         "over [B·s_query, d_model]. sglang `gemma4_fused_ops."
         "gemma_rmsnorm_residual_scalar`. Memory-bound."),
))

register_signature(OpSignature(
    name="gemma_dual_rmsnorm_residual_scalar",
    binds_ops=('gemma_dual_rmsnorm_residual_scalar',),
    inputs=(
        TensorLayout("x1", (("B", "s_query"), "d_model")),
        TensorLayout("x2", (("B", "s_query"), "d_model")),
        TensorLayout("residual", (("B", "s_query"), "d_model")),
        TensorLayout("w1", ("d_model",), role="weight"),
        TensorLayout("w2", ("d_model",), role="weight"),
        TensorLayout("w3", ("d_model",), role="weight"),
        TensorLayout("scalar", (1,), role="weight"),
    ),
    engine_inputs=(
        ("mlp_out", TensorLayout("x1", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation",),
    doc=("Gemma-4 MoE block tail, one launch: (rmsnorm(rmsnorm(x1)·w1 + "
         "rmsnorm(x2)·w2)·w3 + residual)·scalar over [B·s_query, d_model]. "
         "sglang `gemma4_fused_ops.gemma_dual_rmsnorm_residual_scalar`. "
         "Memory-bound: three rows in, one out."),
))

# DSA attention's (DeepSeek-V3.2, GLM-5.x) input preparation, one vLLM launch
# (`deepseek_v32.common.kernels.fused_norm_rope`): the q-LoRA a-norm, the
# latent-KV a-norm and the shared rope key's rotation, and on the layers that
# carry their own indexer the indexer key's LayerNorm and rotation too
# (fusion_plans/fused_dsa_qkv_norm_rope{,_indexer}.yaml). One kernel with
# `has_indexer` the switch, two signatures because the two spans read
# different operands and write different bytes — the indexer form reads the
# indexer key and, in the same launch, resets the step's top-k index buffer
# (`n_topk_keys` int32s a token, pid 3 of the grid): no tensor the engine
# hands the span, so a structural axis of the cell.
def _dsa_norm_rope(name: str, indexer: bool, doc: str) -> None:
    inputs = [
        TensorLayout("q", (("B", "s_query"), "d_q_lora")),
        TensorLayout("kv", (("B", "s_query"), "d_kv_lora")),
        TensorLayout("k_pe", (("B", "s_query"), "d_rope")),
        TensorLayout("q_weight", ("d_q_lora",), role="weight"),
        TensorLayout("kv_weight", ("d_kv_lora",), role="weight"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        # The table's first axis is an implementation constant (max_pos).
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ]
    engine = [
        ("q", TensorLayout("q", ("B", "s_query", "d_q_lora"))),
        ("kv", TensorLayout("kv", ("B", "s_query", "d_kv_lora"))),
        ("k_pe", TensorLayout("k_pe", ("B", "s_query", "d_rope"))),
    ]
    if indexer:
        inputs += [
            TensorLayout("ik", (("B", "s_query"), "d_indexer_head")),
            TensorLayout("ik_weight", ("d_indexer_head",), role="weight"),
            TensorLayout("ik_bias", ("d_indexer_head",), role="weight"),
        ]
        engine.append(("index_key", TensorLayout("ik", ("B", "s_query",
                                              "d_indexer_head"))))
    register_signature(OpSignature(
        name=name,
        binds_ops=(name,),
        inputs=tuple(inputs),
        structural_axes=("n_topk_keys",) if indexer else (),
        engine_inputs=tuple(engine),
        dtype_slots=("activation",),
        doc=doc,
    ))


_dsa_norm_rope(
    "fused_dsa_qkv_norm_rope", False,
    "DSA attention prologue on a shared-indexer layer, one launch: "
    "rmsnorm(q)·w over d_q_lora, rmsnorm(kv)·w over d_kv_lora, and the rope "
    "key k_pe rotated GPT-J style. vLLM `fused_norm_rope(has_indexer=False)`. "
    "Memory-bound.")
_dsa_norm_rope(
    "fused_dsa_qkv_norm_rope_indexer", True,
    "The same launch on a layer with its own indexer: also the indexer key "
    "LayerNorm'd (gain and bias) with its leading d_rope channels rotated, "
    "and the step's top-k index buffer reset (n_topk_keys int32 per token). "
    "vLLM `fused_norm_rope(has_indexer=True)`. Memory-bound.")

# sglang's DSA indexer preparation where the indexer ropes the LEADING
# `d_rope` channels (DeepSeek-V3.2-style `Indexer`, GLM-5.x: interleaved),
# one JIT launch each:
#   * the key: LayerNorm, rotate, FP8-quantise (one fp32 scale per key) and
#     write into the paged index-K cache — so the cache is a `role="kv"`
#     operand, and its page an axis of the cell (`reads_paged_cache`);
#   * the query: rotate each head, FP8-quantise (one scale per head row), and
#     fold that scale into the head weights it also reads and writes.
register_signature(OpSignature(
    name="fused_indexer_k_norm_rope_store",
    binds_ops=('fused_indexer_k_norm_rope_store',),
    inputs=(
        TensorLayout("k", (("B", "s_query"), "d_indexer_head")),
        TensorLayout("k_weight", ("d_indexer_head",), role="weight"),
        TensorLayout("k_bias", ("d_indexer_head",), role="weight"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
        # One page is `kv_block_size` key records; how many pages the cache
        # holds is the deployment's allocation, not an axis of this launch.
        TensorLayout("index_cache", (None, "kv_block_size", "d_indexer_head"),
                     role="kv"),
    ),
    # The key the indexer's `wk` writes. The rotation width rides no tensor
    # the fused node reads (the table is not an operand of the graph), so it
    # is the spec's `d_rope` / `d_qk_rope`.
    engine_inputs=(
        (0, TensorLayout("k", ("B", "s_query", "d_indexer_head"))),
    ),
    dtype_slots=("activation", "kv"),
    doc=("DSA indexer key, one launch: LayerNorm(k)·w + b, the leading d_rope "
         "channels rotated (adjacent pairs), FP8 e4m3 with one fp32 scale per "
         "key, written into the paged index-K cache. sglang "
         "`dsa.indexer_k.fused_k_indexer_norm_rope_store`. Memory-bound."),
))

register_signature(OpSignature(
    name="fused_indexer_q_rope_first_quant",
    binds_ops=('fused_indexer_q_rope_first_quant',),
    inputs=(
        TensorLayout("q", (("B", "s_query"), "n_indexer_heads",
                           "d_indexer_head")),
        # The indexer head weights, read and written scaled by each head's
        # quantisation scale — the launch's second write, fp32.
        TensorLayout("weights", (("B", "s_query"), "n_indexer_heads")),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The head-first query the indexer's `wq_b` writes; the weights are not
    # an operand of the plan's span (their scale sits in `indexer_logits`).
    engine_inputs=(
        (0, TensorLayout("q", ("B", "n_indexer_heads", "s_query",
                               "d_indexer_head"))),
    ),
    dtype_slots=("activation",),
    doc=("DSA indexer query, one launch: per head, the leading d_rope channels "
         "rotated (adjacent pairs), the row FP8 e4m3 with its own scale, and "
         "that scale folded into the head weight (w·weight_scale·scale) — the "
         "launch's second output, written fp32. sglang "
         "`dsv4.elementwise.fused_q_indexer_rope_first_quant`. "
         "Memory-bound."),
))

# vLLM's DSA query preparation (`fused_dsa_q_rope_quant`): the main MLA
# query's rope channels rotated, and the indexer query rotated on its leading
# `d_rope` channels and written FP8 with one power-of-two scale per head row,
# one launch (`deepseek_v32.common.kernels.fused_q`). The indexer head
# weights are read and written in the same launch (each head's scale folded
# into them, written fp32) though the plan holds no node for them, as in
# sglang's `fused_indexer_q_rope_first_quant` above. `q_pe` is the rope tail
# of the `q_b_proj` row the walk splits per head, so the nope width beside it
# (`d_qk_nope`) sizes the row stride the kernel reads it at — no tensor of
# the span carries it.
register_signature(OpSignature(
    name="fused_dsa_q_rope_quant",
    binds_ops=('fused_dsa_q_rope_quant',),
    inputs=(
        TensorLayout("q_pe", (("B", "s_query"), "n_heads", "d_rope")),
        TensorLayout("index_q", (("B", "s_query"), "n_indexer_heads",
                                 "d_indexer_head")),
        TensorLayout("weights", (("B", "s_query"), "n_indexer_heads")),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The fused node's operands arrive in the order the engine hands a span's
    # external inputs over — by member uid (`fusion._external_inputs`), not by
    # the plan's `io_mapping` port names. In both DSA attention components the
    # main query's `q_pe_rope` is declared before the indexer's nodes, so the
    # main q_pe is position 0 and the indexer's head-split query position 1.
    # Nothing in this declaration would notice the two being swapped (both
    # are rank 4); `TestDsaPrologue.test_the_plan_reaches_each_span` checks
    # the projected cell against the checkpoint config instead.
    engine_inputs=(
        ("q_pe", TensorLayout("q_pe", ("B", "n_heads", "s_query", "d_rope"))),
        ("index_q", TensorLayout("index_q", ("B", "n_indexer_heads", "s_query",
                                     "d_indexer_head"))),
    ),
    structural_axes=("d_qk_nope",),
    dtype_slots=("activation", "output"),
    doc=("DSA query preparation, one launch: the main query's q_pe rotated "
         "(adjacent pairs) and written at the activation dtype, the indexer "
         "query's leading d_rope channels rotated and the row written FP8 "
         "e4m3 with one power-of-two scale per head, that scale folded into "
         "the head weights (written fp32). vLLM "
         "`deepseek_v32.common.kernels.fused_q`. Memory-bound."),
))

register_signature(OpSignature(
    name="allreduce_fused_rmsnorm",
    binds_ops=('allreduce_fused_rmsnorm',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
        TensorLayout("residual", (("B", "s_query"), "d_model")),
        TensorLayout("w", ("d_model",), role="weight"),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_model"))),
    ),
    # The group the all-reduce reduces over. No tensor carries it, and the
    # per-rank layout is `fused_add_rmsnorm`'s at every group size — but the
    # launch moves about `n_comm_ranks`× the payload over the fabric, so two
    # group sizes are two measurements. It used to ride on the physical key's
    # deployment label; a label is not the launch's group (one deployment has
    # several), and the key no longer carries it.
    structural_axes=("n_comm_ranks",),
    dtype_slots=("activation",),
    shape_projection="fold_rmsnorm",
    doc=("all-reduce(x); residual += x; rmsnorm(residual) — the TP>1 "
         "FlashInfer / sglang kernel. Same per-rank layout as "
         "`fused_add_rmsnorm`, and its own collective group "
         "(`n_comm_ranks`): the fabric traffic, hence the launch, scales "
         "with it."),
))


# The BARE collectives, the ones layout grafts where two neighbours disagree
# about a tensor's placement, plus the pipeline hop. One signature each,
# because they are different kernels with different schedules
# (`ncclAllReduce` is not `ncclAllToAll`) and a framework that fuses one of
# them into a norm does not fuse the others.
#
# A collective has no local math and no layout of its own: what a launch
# costs is the kind, how many elements it moves, the group it moves them
# over, and how that group sits on the mesh. So the cell is the payload
# FLATTENED (`n_comm_elems`, NCCL's `count`) plus `n_comm_ranks` and
# `n_comm_stride` — the same three facts eval priced the row by
# (`comm_payload_shape`, `comm_world`, `comm_base_stride`), read off the
# row by the `collective` projection rather than rebuilt from a deployment
# label. `n_comm_elems` is the whole payload, eval's `T`: where a
# collective's per-rank buffer is a shard of it (all-gather's send,
# reduce-scatter's receive), that shard is `T / n_comm_ranks` by the
# collective's own definition, which is the one NCCL's API also takes.
#
# FLATTENING IS A CLAIM, so here is the claim: one launch of [8, 1, 7168]
# and one of [56, 1, 1024] are the same measurement, because the kernel's
# own parameters are a pointer, a count, a dtype and a communicator — no
# library in this family takes a shape. Three things could break that,
# and the first two are not in this layer:
#   * a payload that is not contiguous would have to be packed first. In
#     this graph that packing is its OWN node with its own row (`permute`,
#     `combine`, the view ops), so what reaches a collective row is the
#     contiguous buffer by construction.
#   * ragged per-peer messages. A real MoE dispatch is an all-to-all-V:
#     peers get unequal slices, and at one total payload a skewed split is
#     slower than an even one. The cell does not say this, and neither does
#     eval's own traffic law ((N-1)/N·T is the balanced ring). The
#     imbalance is in the walk — the busiest rank's rows are the payload's
#     (`ExpectedRoutedLoad.busiest_rank_load_factor`) — so if a measurement
#     ever shows the per-peer split matters it enters as a declared
#     kernel parameter beside the payload (the way `moe_token_distribution`
#     already does for the expert GEMMs), not as a shape axis.
#   * an element width that is not the row's own. A count is a buffer only
#     once something says how wide an element is, and a collective may move
#     a payload that is not its row's activations — DCP all-gathers an fp32
#     log-sum-exp beside bf16 hidden states. The row publishes that width
#     (`comm_payload_dtype_bytes`); `CommLaunch.from_row` refuses a
#     disagreement rather than counting elements at the wrong one, and
#     cannot repair it, because a width does not name a dtype (one byte is
#     fp8 or int8).
#
# Without these, a multi-GPU deployment could never have a measured total:
# the only collective anyone had measured was the fused one, and a
# measured total is all-or-nothing over the step's ops.
register_signature(OpSignature(
    name="all_reduce",
    binds_ops=('all_reduce',),
    shape_projection="collective",
    inputs=(TensorLayout("payload", ("n_comm_elems",)),),
    structural_axes=("n_comm_ranks", "n_comm_stride"),
    variant_dims=(COMM_DOMAIN_AXIS,),
    doc=("all-reduce: every rank ends up holding the sum over the group "
         "(`ncclAllReduce`, or a custom one-shot/two-shot kernel at small "
         "sizes). Grafted where a row-parallel GEMM's partial sums meet a "
         "consumer that needs the whole tensor. Per-rank ring traffic "
         "2·(N-1)/N·T, so it moves about twice what the other collectives "
         "of the same payload do."),
))

register_signature(OpSignature(
    name="all_gather",
    binds_ops=('all_gather',),
    shape_projection="collective",
    inputs=(TensorLayout("payload", ("n_comm_elems",)),),
    structural_axes=("n_comm_ranks", "n_comm_stride"),
    variant_dims=(COMM_DOMAIN_AXIS,),
    doc=("all-gather: each rank contributes its shard and ends up holding "
         "the concatenation (`ncclAllGather`). Grafted where a sharded "
         "tensor meets a consumer that needs it whole — the vocabulary "
         "gather at the LM head is the common one. `n_comm_elems` is the "
         "gathered buffer; one rank sends `n_comm_elems / n_comm_ranks`."),
))

register_signature(OpSignature(
    name="reduce_scatter",
    binds_ops=('reduce_scatter',),
    shape_projection="collective",
    inputs=(TensorLayout("payload", ("n_comm_elems",)),),
    structural_axes=("n_comm_ranks", "n_comm_stride"),
    variant_dims=(COMM_DOMAIN_AXIS,),
    doc=("reduce-scatter: the group sums the payload and each rank keeps "
         "one slice of the result (`ncclReduceScatter`). Grafted where "
         "partial sums meet a consumer that wants them sharded — the "
         "sequence-parallel form of the all-reduce above. `n_comm_elems` "
         "is the payload reduced; one rank keeps "
         "`n_comm_elems / n_comm_ranks`."),
))

register_signature(OpSignature(
    name="all_to_all",
    binds_ops=('all_to_all',),
    shape_projection="collective",
    inputs=(TensorLayout("payload", ("n_comm_elems",)),),
    structural_axes=("n_comm_ranks", "n_comm_stride"),
    variant_dims=(COMM_DOMAIN_AXIS,),
    doc=("all-to-all: every rank sends a distinct slice to every other, "
         "swapping which axis the tensor is sharded on — torch's "
         "`all_to_all_single` with a flat `sendcount`. The EQUAL-slice form, "
         "which is what the sequence-parallel exchange is (Ulysses: the "
         "heads of a sequence-cut tensor for the sequence of a head-cut "
         "one). A MoE dispatch is the ragged form and is its own cell "
         "(`routed_all_to_all`): it sends a token once per distinct "
         "destination its top-k reaches, so its per-peer messages are "
         "unequal and their total is a fraction of the buffer."),
))

register_signature(OpSignature(
    name="routed_all_to_all",
    binds_ops=('routed_all_to_all',),
    shape_projection="collective",
    inputs=(TensorLayout("payload", ("n_comm_elems",)),),
    structural_axes=("n_comm_ranks", "n_comm_stride", "n_experts_active",
                     "n_route_targets", "n_route_experts"),
    doc=("The MoE dispatch's own exchange: DeepEP's dispatch and combine, "
         "which send a token ONCE per distinct destination its top-k picks "
         "land on and fan the copy out at the receiver. Not the signature "
         "above: that one's cell states an equal split of the buffer between "
         "peers, and this one's per-peer messages are ragged and their total "
         "a fraction of it — the fraction the three routing axes give "
         "(`comm_volume.RouteFanout`: `n_experts_active` picks drawn from "
         "`n_route_targets` destinations of which `n_route_experts` hold "
         "weights). A row is this cell when it carries that fan-out, which "
         "is a fact of the graph (`comm_route_fanout`, off the dispatch) and "
         "not of the model's name or the framework's."),
))

register_signature(OpSignature(
    name="send_recv",
    binds_ops=('send_recv',),
    shape_projection="collective",
    inputs=(TensorLayout("payload", ("n_comm_elems",)),),
    structural_axes=("n_comm_ranks", "n_comm_stride"),
    variant_dims=(COMM_DOMAIN_AXIS,),
    doc=("The pipeline hop: one point-to-point send/recv pair handing a "
         "tensor across a pipeline cut (`ncclSend` / `ncclRecv`). Not a "
         "collective over a group — the payload crosses once, whatever the "
         "pipeline's depth — but the group is still part of the cell, "
         "because `n_comm_ranks` with `n_comm_stride` is what decides "
         "whether the peer is in this node or across the fabric. One row "
         "per cut, one launch per step per cut."),
))

register_signature(OpSignature(
    name="silu_and_mul",
    binds_ops=('silu_and_mul',),
    inputs=(
        # The physical second axis is 2·d_ffn (gate|up packed).
        TensorLayout("x", (("B", "s_query"), (2, "d_ffn"))),
    ),
    engine_inputs=(
        ("gate", TensorLayout("x", ("B", "s_query", "d_ffn"))),
    ),
    dtype_slots=("activation",),
    doc="SwiGLU gate: silu(gate) * up over [B·s_query, d_ffn]. Memory-bound.",
))

# Two gated activations whose fused kernel both frameworks ship, beside
# `silu_and_mul`'s: a clamped SwiGLU (gpt-oss, MiniMax-M3) and Kimi-K3's
# SiTU gate. Same packed operand, same boundary — the activation differs,
# which is why each is its own cell rather than a flavour of one.
register_signature(OpSignature(
    name="silu_and_mul_with_clamp",
    binds_ops=('silu_and_mul_with_clamp',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), (2, "d_ffn"))),
    ),
    engine_inputs=(
        ("gate", TensorLayout("x", ("B", "s_query", "d_ffn"))),
    ),
    dtype_slots=("activation",),
    # The bound both branches are clamped to — the one the `clamp` ops this
    # kernel absorbed take (`OpSpec.consts`), as the walk bound it, on the row.
    # vLLM takes it from the quant config's `gemm1_clamp_limit` else the MoE
    # config's `swiglu_limit`, and asserts rather than defaults; DeepSeek-V4
    # and GLM-5.3 publish 10.0, gpt-oss 7.0. Clamping at the wrong bound is
    # a different function — and on standard-normal operands, one that
    # happens to agree with the reference everywhere.
    config_params=(("clamp_limit", "clamp_limit"),),
    doc=("Clamped SwiGLU: silu(clamp(gate)) * clamp(up) over "
         "[B·s_query, d_ffn]. vLLM `torch.ops._C.silu_and_mul_with_clamp` "
         "(layers/fused_moe/activation.py:311). Memory-bound."),
))


register_signature(OpSignature(
    name="situ_and_mul",
    binds_ops=('situ_and_mul',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), (2, "d_ffn"))),
    ),
    engine_inputs=(
        ("gate", TensorLayout("x", ("B", "s_query", "d_ffn"))),
    ),
    dtype_slots=("activation",),
    # Both betas vLLM passes the kernel, as the `situ` op it absorbed
    # declares them: `beta` is mandatory there, `linear_beta` genuinely
    # optional — a non-positive value is the kernel's "unset", which
    # forwards the up branch. Kimi-K3 publishes 4.0 and 25.0.
    config_params=(("situ_beta", "situ_beta"),
                   ("situ_linear_beta", "situ_linear_beta")),
    doc=("SiTU gate (Kimi-K3): situ(gate) * up over [B·s_query, d_ffn], "
         "`situ(x) = beta * tanh(x / beta) * sigmoid(x)`. vLLM "
         "`torch.ops._C.situ_and_mul` (layers/fused_moe/activation.py:372). "
         "Memory-bound."),
))


register_signature(OpSignature(
    name="gelu_and_mul",
    binds_ops=('gelu_and_mul',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), (2, "d_ffn"))),
    ),
    engine_inputs=(
        ("gate", TensorLayout("x", ("B", "s_query", "d_ffn"))),
    ),
    dtype_slots=("activation",),
    doc="GeGLU gate: gelu(gate) * up over [B·s_query, d_ffn]. Same packing "
        "as silu_and_mul, different activation — not the same cell.",
))

register_signature(OpSignature(
    name="rope",
    binds_ops=('fused_qk_rope',),
    inputs=(
        # Canonical q/k are flat [tokens, n_heads·d_rope] /
        # [tokens, n_kv_heads·d_rope] — the second
        # dim packs the head and head-width axes.
        TensorLayout("q", (("B", "s_query"), ("n_heads", "d_rope"))),
        TensorLayout("k", (("B", "s_query"), ("n_kv_heads", "d_rope"))),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        # The cache's first dim is an implementation constant (max_pos), not
        # a cell axis.
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # Engine layouts of the same two operands, keyed by rank: 4-D keeps
    # the head axis, 3-D packs heads×width. Q and K choose independently
    # (MLA: 4-D Q, packed K). At `fused_qk_rope`'s ports; the primitive
    # `rope` op, measured here too under its own name, rotates one operand,
    # read as the query (`primitive_ports`).
    engine_inputs=(
        ("q_pe", TensorLayout("q", ("B", "n_heads", "s_query", "d_rope"))),
        ("q_pe", TensorLayout("q", ("B", "s_query", ("n_heads", "d_rope")))),
        ("k_pe", TensorLayout("k", ("B", "n_kv_heads", "s_query", "d_rope"))),
        ("k_pe", TensorLayout("k", ("B", "s_query", ("n_kv_heads", "d_rope")))),
    ),
    primitive_ports=("q_pe",),
    dtype_slots=("activation",),
    doc=("Rotary position embedding applied to Q and K in one launch. Q "
         "[tokens, n_heads·d_rope], K [tokens, n_kv_heads·d_rope] "
         "(GQA: n_kv_heads<=n_heads); rotate the first "
         "`rotary_dim` of each head by cos/sin selected at each token's "
         "position, pass the rest through. Memory-bound, in-place. vLLM "
         "`_custom_ops.rotary_embedding` / sglang + flashinfer "
         "`apply_rope_with_cos_sin_cache_inplace`."),
))

# Partial rotary embedding on whole heads, one in-place launch — the fused
# spans `fused_qk_rope_partial` (GQA: the FIRST `d_rope` of each q and k head
# of width `d_head` rotate, `partial_rotary_factor` < 1) and
# `fused_qk_rope_partial_q` (MLA: the LAST `d_rope` of each q head rotate,
# the decoupled rope key `k_pe [tokens, d_rope]` rotates whole). The rotary
# kernel is handed the heads where they lie and rotates `d_rope` of each
# `d_head`-strided head, so the head width is an axis of the cell beside the
# rotation width: `rope` (whole-head rotation) cannot key them. Two
# signatures because the two spans read different tensors: the GQA key is
# `n_kv_heads` heads of `d_head`, the MLA key one head of `d_rope`.
register_signature(OpSignature(
    name="rope_partial",
    binds_ops=('fused_qk_rope_partial',),
    inputs=(
        TensorLayout("q", (("B", "s_query"), ("n_heads", "d_head"))),
        TensorLayout("k", (("B", "s_query"), ("n_kv_heads", "d_head"))),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The head-split q and k, head-first. The rotation width is no axis of
    # either (the tail passes through), so it is read from the spec (`d_rope`)
    # and a row whose spec does not name it there is dropped, not guessed.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("k", TensorLayout("k", ("B", "n_kv_heads", "s_query", "d_head"))),
    ),
    dtype_slots=("activation",),
    doc=("Partial RoPE on Q and K in one in-place launch: the first `d_rope` "
         "of every `d_head`-wide head rotated by cos/sin at each token's "
         "position, the rest untouched. vLLM `_custom_ops.rotary_embedding` "
         "on the whole q/k / sglang `apply_rope_with_cos_sin_cache_inplace` "
         "on the `[..., :d_rope]` views. Memory-bound."),
))

register_signature(OpSignature(
    name="rope_partial_q",
    binds_ops=('fused_qk_rope_partial_q',),
    inputs=(
        TensorLayout("q", (("B", "s_query"), ("n_heads", "d_head"))),
        TensorLayout("k_pe", (("B", "s_query"), "d_rope")),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The head-split query and the rope slice of the latent; the rope key's
    # width is the rotation width, so the cell is read off the tensors alone.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("k_pe", TensorLayout("k_pe", ("B", "s_query", "d_rope"))),
    ),
    dtype_slots=("activation",),
    doc=("MLA's RoPE in one in-place launch: the last `d_rope` of every "
         "`d_head`-wide query head (a strided view) and the decoupled rope "
         "key `k_pe [tokens, d_rope]` rotated whole. vLLM "
         "`MultiHeadLatentAttentionWrapper` → `_custom_ops.rotary_embedding` "
         "/ sglang `forward_mha` → `apply_rope_with_cos_sin_cache_inplace`. "
         "Memory-bound."),
))

register_signature(OpSignature(
    name="rope_2d",
    binds_ops=('rope_2d',),
    inputs=(
        # vLLM `ApplyRotaryEmb` takes x [tokens, n_heads, d_rope] (a leading
        # batch axis optional) and the per-token cos/sin rows the tower built
        # from its (h, w) grid, each [tokens, d_rope/2] — the half width is an
        # implementation constant of the table, not a cell axis.
        TensorLayout("x", (("B", "s_query"), "n_heads", "d_rope")),
        TensorLayout("cos", (("B", "s_query"), None), role="metadata"),
        TensorLayout("sin", (("B", "s_query"), None), role="metadata"),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "n_heads", "s_query", "d_rope"))),
    ),
    dtype_slots=("activation",),
    doc=("Two-axis vision RoPE on one tensor: x [tokens, n_heads, d_rope] "
         "rotated NeoX-style over the whole head by per-token cos/sin rows "
         "[tokens, d_rope/2] that the vision tower builds from each patch's "
         "(h, w) grid position. Memory-bound, out-of-place, one launch per "
         "tensor. vLLM `ApplyRotaryEmb` (rotary_embedding/common.py; "
         "vllm-flash-attn Triton `apply_rotary_emb` on CUDA), which the "
         "Qwen2.5-VL / Qwen3-VL / Qwen3.5 and MiMo-V2 omni vision towers "
         "call. Qwen-VL stacks q and k into one call; MiMo calls it once "
         "per tensor."),
))

register_signature(OpSignature(
    name="xdrope",
    binds_ops=('xdrope',),
    inputs=(
        # vLLM `XDRotaryEmbedding.forward_cuda` views each of q and k as
        # [tokens, n_heads, d_rope] and hands it to `ApplyRotaryEmb` with
        # the cos/sin rows it assembled from the four position streams, one
        # `xdrope_section` channel section per stream; the half width is an
        # implementation constant of the table, not a cell axis.
        TensorLayout("x", (("B", "s_query"), "n_heads", "d_rope")),
        TensorLayout("cos", (("B", "s_query"), None), role="metadata"),
        TensorLayout("sin", (("B", "s_query"), None), role="metadata"),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "n_heads", "s_query", "d_rope"))),
    ),
    dtype_slots=("activation",),
    doc=("XD-RoPE on one tensor (HunYuan-VL text): x [tokens, n_heads, "
         "d_rope] rotated NeoX-style over the whole head by per-token cos/sin "
         "rows [tokens, d_rope/2], each `xdrope_section` channel section "
         "taken from its own position stream (p, w, h, t). Memory-bound, "
         "out-of-place, one launch per tensor. vLLM `XDRotaryEmbedding."
         "forward_cuda` (layers/rotary_embedding/xdrope.py), which gathers "
         "the rows and calls `ApplyRotaryEmb` on q and on k (vLLM main "
         "a18c9b56f; v0.32.0 serves HunYuan-VL through its Transformers "
         "backend); sglang v0.5.22 has no HunYuan-VL implementation."),
))

# Multimodal RoPE on whole Q and K heads. The three position streams and the
# three channel sections are parameters of the rotation, not tensor axes; the
# section widths sum to d_rope/2. `mrope_interleaved` selects Qwen3-VL's
# interleaved lane ownership over Qwen3.5's packed sections.
register_signature(OpSignature(
    name="fused_qk_mrope",
    binds_ops=('fused_qk_mrope',),
    inputs=(
        TensorLayout("q", (("B", "s_query"), "n_heads", "d_head")),
        TensorLayout("k", (("B", "s_query"), "n_kv_heads", "d_head")),
        TensorLayout("positions", (3, ("B", "s_query")), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    engine_inputs=(
        ("q_pe", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("k_pe", TensorLayout("k", ("B", "n_kv_heads", "s_query", "d_head"))),
    ),
    structural_axes=("d_rope", "mrope_section_t", "mrope_section_h",
                     "mrope_section_w", "mrope_interleaved"),
    _identity_dims=("B", "s_query", "n_heads", "n_kv_heads", "d_head",
                    "d_rope", "mrope_interleaved"),
    dtype_slots=("activation",),
    doc=("NeoX multimodal RoPE on the first `d_rope` channels of Q and K, one "
         "in-place launch: each channel section reads its temporal, height or "
         "width position stream. vLLM `MRotaryEmbedding.forward_cuda` gathers "
         "three cos/sin rows and calls `triton_mrope` (mrope.py:439-470, "
         "kernel :16); sglang `MRotaryEmbedding.forward_triton` calls "
         "`triton_mrope_fused` (mrope.py:270-291, kernel "
         "rotary_triton.py:13). Memory-bound."),
))


# Qwen's gated attention prologue: one packed [q|gate] per head beside k,
# per-head QK-RMSNorm, partial RoPE, and the gate copied out — one launch
# in both frameworks (vLLM `fused_qk_rmsnorm_rope_gate`,
# model_executor/layers/fused_qk_norm_rope.py:138; sglang's
# `fused_qk_gemma_rmsnorm_rope_gate`). The two spec writings of its norm
# (one node, or unscaled × gain) are two plans on this one cell.
# The head is `d_head` wide and only its first `d_rope` channels rotate
# (Qwen3.5 / Qwen3.8: 256 and 64); the norm spans the whole head. The engine
# hands q|gate head-split, `[B, n_heads, s_query, 2·d_head]`, and k; the
# split-gain writing also hands the two gains as ports of their own
# (`q_weight`, `k_weight`), which the cell does not read.
register_signature(OpSignature(
    name="fused_qk_rmsnorm_rope_gate",
    binds_ops=('fused_qk_rmsnorm_rope_gate',
               'fused_qk_rmsnorm_rope_gate_split_gain'),
    inputs=(
        # Per head [q|gate], as the kernel takes it.
        TensorLayout("q_gate", (("B", "s_query"), ("n_heads", 2, "d_head"))),
        TensorLayout("k", (("B", "s_query"), ("n_kv_heads", "d_head"))),
        TensorLayout("q_norm_w", ("d_head",), role="weight"),
        TensorLayout("k_norm_w", ("d_head",), role="weight"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    engine_inputs=(
        ("q_gate", TensorLayout("q_gate", ("B", "n_heads", "s_query",
                                    (2, "d_head")))),
        ("k", TensorLayout("k", ("B", "n_kv_heads", "s_query", "d_head"))),
    ),
    dtype_slots=("activation",),
    doc=("Qwen's gated attention prologue in one launch: split [q|gate], "
         "per-head RMSNorm of q and k (separate gains), partial RoPE, gate "
         "copied through. vLLM `fused_qk_rmsnorm_rope_gate` / sglang "
         "`fused_qk_gemma_rmsnorm_rope_gate`. Memory-bound."),
))


# The CSA2/DSA indexer's query preparation. Two signatures for the two spec
# writings of the V4 / V4.1 tail (`dsv4_indexer` names rope plus V4's
# Hadamard; `csa2_index_score` spells V4.1's slice → rope → concat), because
# a pattern node matches one spec node. Both bind vLLM's one launch, which
# also quantises q and rescales the head weights.
def _indexer_q_rope_quant(name: str) -> None:
    hadamard = FUSION_SCHEMES[name].naive_pattern.runs("hadamard_transform")
    hadamard_doc = (
        "Hadamard-rotate the head, " if hadamard else "")
    framework_doc = (
        "vLLM `fused_indexer_q_rope_quant` (fused_indexer_q.py:549) writes "
        "FP8 e4m3 with the per-head scale folded into weights, or MXFP4 "
        "with per-32 UE8M0 scales; it "
        "omits the reference Hadamard, a score-preserving deviation its "
        "provider checks against its own no-H oracle and undoes only when "
        "canonicalizing. sglang's V4 default "
        "`fused_q_indexer_rope_hadamard_quant` computes it."
        if hadamard else
        "vLLM `fused_indexer_q_rope_quant` (fused_indexer_q.py:549) and "
        "sglang's V4.1 split form `index_q_rope_pack_weights` "
        "(fp4_indexer_rope.py:91) write the quantised query record."
    )
    register_signature(OpSignature(
        name=name,
        binds_ops=(name,),
        inputs=(
            TensorLayout("q", (("B", "s_query"), "n_heads", "d_head")),
            TensorLayout("weights", (("B", "s_query"), "n_heads")),
            TensorLayout("positions", (("B", "s_query"),), role="metadata"),
            TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
        ),
        # The walk hands q head-split, `[B, n_heads, s_query, d_head]`
        # (`head_split` in both writings), beside the projection's
        # token-major weights.
        engine_inputs=(
            ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
            ("weights", TensorLayout("weights", ("B", "s_query", "n_heads"))),
        ),
        # d_rope is the tail width, while d_head is the axis q already carries.
        structural_axes=("d_rope",),
        dtype_slots=("activation", "output"),
        doc=("Rotate the last `d_rope` channels of each indexer q head, "
             + hadamard_doc
             + "scale "
             "the positive head weights by d_head^-1/2·n_heads^-1/2, and write "
             "q in the launch's output format. " + framework_doc
             + " Memory-bound."),
    ))


_indexer_q_rope_quant("fused_indexer_q_rope_quant")
_indexer_q_rope_quant("fused_indexer_q_rope_quant_split")


# MiniMax-M3's attention prologue. The canonical packed row is [q|k|v]. On
# NVIDIA the sparse layers run the launch WITH its index branch and both
# cache inserts (minimax_m3/nvidia/model.py:625-650; `skip_index_branch`
# defaults False, _custom_ops.py:2587) — that production form is the
# provider's, index tail, slot maps and caches included. The dense layers
# call the same op without the index/cache operands (:374-385), and
# `skip_index_branch=True` is only the AMD reuse layers' form
# (amd/model.py:1032, :1047, :1083). The variant axes below are what tell
# the two geometries apart on a cell.
register_signature(OpSignature(
    name="fused_qk_norm_partial_rope",
    binds_ops=('fused_qk_norm_partial_rope',),
    inputs=(
        TensorLayout("qkv", (("B", "s_query"),
                             ("+", ("n_heads", "d_head"),
                              (2, "n_kv_heads", "d_head")))),
        TensorLayout("q_norm_w", ("d_head",), role="weight"),
        TensorLayout("k_norm_w", ("d_head",), role="weight"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The launch's operands are the plan's ports: the head-split Q and K the
    # norms read (`fused_qk_norm_partial_rope.yaml`: external.q / external.k
    # off `q_split_heads` / `k_split_heads`, `[B, heads, s, d_head]`). The
    # packed [q|k|v] row the kernels take (`inputs` above) is the
    # frameworks' argument, which no graph tensor carries: laid on the `q`
    # port it bound nothing, a rank-3 packing against a rank-4 query.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        ("k", TensorLayout("k", ("B", "n_kv_heads", "s_query", "d_head"))),
    ),
    structural_axes=("d_rope",),
    # The sparse/cache-writing geometry is SELECTED, not sized: a cell
    # carrying `n_index_heads` is the vLLM sparse launch (index branch, both
    # paged caches), a cell without it is the dense [q|k|v] launch sglang
    # runs. As unconditional structural axes the two made the dense cell
    # impossible — smoke's minimal cell carried `n_index_heads`, which the
    # dense provider must refuse. (Planning a sparse row must carry the axis
    # as a leaf fact; no current spec feeds one, so a sparse row that
    # arrives without it keys dense, exactly as before this split.)
    variant_dims=("n_index_heads", "kv_block_size"),
    dtype_slots=("activation", "weight"),
    doc=("Per-head Gemma RMSNorm of Q and K followed by partial NeoX RoPE over "
         "one packed qkv row, in place. vLLM "
         "`fused_minimax_m3_qknorm_rope_kv_insert` (_custom_ops.py:2566; "
         "kernel .cu:124-181), whose sparse form also writes K/V to a paged "
         "cache; sglang `minimax_qknorm_rope` (minimax_qknorm_rope.py:152, "
         "kernel .cuh:41-115). Memory-bound."),
))


# Normalise one activation, then rotate its last `d_rope` channels —
# DeepSeek-V4's query track and its latent-KV row, each one sglang launch
# that reads the row once and writes it once. Two signatures because the
# operand differs: the query is per head and its norm carries no gain (the
# kernel computes the RMS and divides); the KV row is ONE row per token —
# the latent KV, the compressor's pooled entry, the indexer key — and its
# norm reads a weight.
register_signature(OpSignature(
    name="fused_qnorm_rope",
    binds_ops=("fused_qnorm_rope",),
    shape_projection="inv_rope",
    inputs=(
        TensorLayout("x", ("B", "n_heads", "s_query", "d_head")),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    dtype_slots=("activation",),
    doc=("Per-head RMSNorm without a gain, then RoPE over the last `d_rope` "
         "channels, one launch over [B, n_heads, s_query, d_head] (sglang "
         "`fused_q_norm_rope`). Memory-bound: q in, q out."),
))
register_signature(OpSignature(
    name="fused_norm_rope",
    binds_ops=("fused_norm_rope",),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_head")),
        TensorLayout("w", ("d_head",), role="weight"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The row the norm reads, as the graph carries it. Its width is the
    # row's own (the latent's d_head, an indexer key's d_indexer_head); the
    # rotation width rides no tensor the fused node reads, so it is the
    # spec's `d_qk_rope`, as for `fused_indexer_k_norm_rope_store`.
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_head"))),
    ),
    structural_axes=("d_rope",),
    dtype_slots=("activation",),
    doc=("RMSNorm (with gain) then RoPE over the last `d_rope` channels of "
         "one row per token (sglang `fused_k_norm_rope_flashmla` / "
         "`fused_norm_rope_inplace` / `compress_fused_norm_rope_inplace`). "
         "Memory-bound: the row in, the row out."),
))


# DeepSeek-V4's whole attention input preparation, which vLLM runs as ONE
# launch per layer: the query heads normalised and rotated, and the latent-KV
# row rotated, quantised and inserted into the paged sliding-window cache —
# the fused spans `fused_qnorm_rope_kv_insert` and
# `fused_qnorm_rope_kv_insert_no_q_norm`. One signature per span, as
# `gelu_and_mul` sits beside `silu_and_mul`: the same operands and the same
# bytes, a different function (V4.1 norms its query before `wq_b`, so its
# launch skips the norm), and the oracle has to compute the one its own
# kernel does. It is one vLLM op with `apply_q_norm` the flag between them
# (`torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`).
#
# Operands as the kernel takes them, token-major: the query heads, the latent
# row, the cache the row lands in, and the three tables the launch reads that
# no node produces. The cache is a `role="kv"` operand, which is the one
# declaration that says both things about it — the dtype slot its record is
# kept in (which of vLLM's three cache records this launch writes), and that
# the deployment pages it, so the page is an axis of the cell
# (`OpSignature.reads_paged_cache`).
#
# Each activation is declared at the port it fills (`q`, `kv` in the plan's
# `io_mapping`). A launch names the port each operand fills
# (`FusedNode.operand_ports`), so which comes first in a layer's graph — the
# query in four of DeepSeek-V4.1's five row classes, the latent in the
# fifth — is no part of the contract.
def _qnorm_rope_kv_insert(name: str, doc: str) -> None:
    register_signature(OpSignature(
        name=name,
        binds_ops=(name,),
        inputs=(
            TensorLayout("q", (("B", "s_query"), "n_heads", "d_head")),
            TensorLayout("kv", (("B", "s_query"), "d_head")),
            # One paged block is `kv_block_size` token records wide; how many
            # blocks the cache holds is the deployment's allocation and no
            # axis of this launch, which touches one record per token.
            TensorLayout("kv_cache", (None, "kv_block_size", "d_head"),
                         role="kv"),
            TensorLayout("slot_mapping", (("B", "s_query"),), role="metadata"),
            TensorLayout("positions", (("B", "s_query"),), role="metadata"),
            # The table's first axis is an implementation constant (max_pos).
            TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
        ),
        engine_inputs=(
            ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
            ("kv", TensorLayout("kv", ("B", "s_query", "d_head"))),
        ),
        dtype_slots=("activation", "kv"),
        doc=doc,
    ))


_qnorm_rope_kv_insert(
    "fused_qnorm_rope_kv_insert",
    "DeepSeek-V4 attention input preparation in one launch: per-head "
    "RMSNorm of the query without a gain, GPT-J RoPE over the last "
    "`d_rope` channels of every query head and of the latent-KV row, and "
    "that row quantised and written into the paged sliding-window cache at "
    "each token's slot. vLLM "
    "`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` with "
    "`apply_q_norm=True`. Memory-bound: the query in and out, the latent in, "
    "one cache record out.")
_qnorm_rope_kv_insert(
    "fused_qnorm_rope_kv_insert_no_q_norm",
    "The same launch with the query norm off — DeepSeek-V4.1, which norms "
    "`qr` before `wq_b`: GPT-J RoPE over the query heads and the latent-KV "
    "row, and the row quantised into the paged sliding-window cache. vLLM "
    "`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` with "
    "`apply_q_norm=False`. Memory-bound.")


# DeepSeek-V4's attention output, un-rotated before the output projection.
# One kernel reads o, rotates back its last `d_rope` channels with the
# conjugate frequencies, and writes it (vLLM
# `fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, ...)`,
# models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:352 — the FP8 it
# also writes is the precision profile's, not a cell axis).
register_signature(OpSignature(
    name="fused_inv_rope",
    binds_ops=('fused_inv_rope',),
    shape_projection="inv_rope",
    inputs=(
        # Head-first, as the attention leaf emits it. vLLM's kernel takes
        # the same buffer token-first ([num_tokens, num_heads, head_dim]);
        # the provider states that permute (`launch_inputs`) rather than the
        # contract carrying two layouts of one rank.
        TensorLayout("o", ("B", "n_heads", "s_query", "d_head")),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        # The table's first axis is an implementation constant (max_pos).
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    dtype_slots=("activation",),
    doc=("Inverse rotary embedding over the attention output's last "
         "`d_rope` channels, one launch over [tokens, n_heads, d_head] "
         "(vLLM `fused_inv_rope_fp8_quant`). Memory-bound: o in, o out."),
))


# DeepSeek-V4.1's CSA2 compressor tail (fusion_plans/
# fused_save_compress_norm{,_decode}.yaml): the softmax pool of each closed
# group of `compress_ratio` tokens by its gate and the pooled row's RMSNorm
# into the pre-RoPE latent — vLLM's one Triton launch
# `_fused_save_compress_norm_kernel` (models/deepseek_v41/common/ops/
# fused_compress_quant_cache.py:22, from `DeepseekCompressor.forward`,
# deepseek_v41/compressor.py:283), which also saves the open group's fp32
# state rows to the compressor ring. The prefill and decode spellings are
# one cell: the same operands, the same kernel, the token count on
# `s_query` (a decode step's is its `n_verify`, as every decode cell keys
# it). The engine operands are the fused wkv|wgate GEMM row's two halves,
# kv first (the launch reads them side by side as one `[tokens,
# 512·compress_ratio]` fp32 row, which the provider builds as such).
register_signature(OpSignature(
    name="fused_save_compress_norm",
    binds_ops=('fused_save_compress_norm', 'fused_save_compress_norm_decode'),
    inputs=(
        TensorLayout("kv", (("B", "s_query"), "d_head")),
        TensorLayout("score", (("B", "s_query"), "d_head")),
        TensorLayout("norm_weight", ("d_head",), role="weight"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
    ),
    # The group size: no tensor carries it (the launch's row is
    # `d_head`-wide at every ratio); the plan's `view_split` binds it.
    structural_axes=("compress_ratio",),
    engine_inputs=(
        ("kv", TensorLayout("kv", ("B", "s_query", "d_head"))),
        ("score", TensorLayout("score", ("B", "s_query", "d_head"))),
    ),
    dtype_slots=("activation",),
    doc=("CSA2 compressor pool + norm, one launch: per closed group of "
         "compress_ratio tokens, softmax(score) per channel over the group, "
         "the weighted sum of kv, then rmsnorm·w over d_head, written bf16 "
         "at the group's closing row. vLLM `deepseek_v41.common.ops."
         "fused_compress_quant_cache.fused_save_compress_norm` (the launch "
         "also saves the open group's fp32 ring rows). The launch reads "
         "the fp32 row the fused wkv|wgate GEMM emits (`out_dtype="
         "torch.float32`, deepseek_v41/attention.py:954-959); the cell "
         "keys the halves at the walk's activation slot, which bills them "
         "bf16 — the logical billing dtype, not the launch's input "
         "precision (the row's fp32 `kv` slot — the state ring — is the "
         "walk's own). Memory-bound: two rows in, one row out per group."),
))

# DeepInfer's CSA2 compressor pool (fusion_plans/fused_compress_pool_state.yaml):
# the same pool as `fused_save_compress_norm` above without the norm, which
# DeepInfer runs in its store launch (`fused_compress_norm_rope_store`). One
# cell for both phases, the token rows on `s_query`, as above.
register_signature(OpSignature(
    name="fused_compress_pool_state",
    binds_ops=('fused_compress_pool_state',),
    inputs=(
        TensorLayout("kv", (("B", "s_query"), "d_head")),
        TensorLayout("score", (("B", "s_query"), "d_head")),
    ),
    # The group size: no tensor carries it; the plan's `view_split` binds it.
    structural_axes=("compress_ratio",),
    engine_inputs=(
        ("kv", TensorLayout("kv", ("B", "s_query", "d_head"))),
        ("score", TensorLayout("score", ("B", "s_query", "d_head"))),
    ),
    dtype_slots=("activation",),
    doc=("DeepInfer's CSA2 compressor pool and state update, one launch "
         "(`compress_and_update_state_prefill` / `_decode`): per closed "
         "group of compress_ratio tokens, softmax(score) per channel over "
         "the group and the weighted kv sum, written bf16, with the open "
         "group's fp32 [score, kv] rows saved to the state ring. The launch "
         "reads the fp32 row the fused wkv|wgate GEMM emits (DeepInfer "
         "asserts it, v4_modules/compress.py:95); the cell keys the halves "
         "at the walk's activation slot, which bills them bf16 — the same "
         "stated residual as `fused_save_compress_norm`. Memory-bound."),
))


# DeepSeek-V4.1's compressed-KV store (fusion_plans/
# fused_compress_rope_store.yaml): the pooled latent's last `d_rope`
# channels rotated GPT-J style at the group's first position and the row
# published into the paged compressed cache — vLLM's `rope_quant_insert`
# (fused_compress_quant_cache.py:224, from `DeepseekCompressor.insert_cache`,
# deepseek_v41/compressor.py:319). On every arch but SM100 the record is
# V4's: `d_head - d_rope` FP8 e4m3 channels with one UE8M0 exponent per 64,
# a pad byte, and the `d_rope` rotated channels bf16 (584 B a token;
# `_rope_quant_insert_kernel`, :304 — the SM100 MXFP8 record,
# `_use_v41_mxfp8_kv_record` at attention.py:128, is another launch the
# H100 cell never sees). The same launch serves the ratio-1 candidate
# source: every token is a boundary there.
register_signature(OpSignature(
    name="fused_compress_rope_store",
    binds_ops=('fused_compress_rope_store',),
    inputs=(
        # The compressor's pooled rows, one per closed group (at ratio 1,
        # one per token).
        TensorLayout("latent", (("B", "s_slots"), "d_head")),
        # One paged block is `kv_block_size` token records; how many blocks
        # the cache holds is the deployment's allocation, not an axis of
        # this launch, which touches one record per boundary row.
        TensorLayout("kv_cache", (None, "kv_block_size", "d_head"),
                     role="kv"),
        TensorLayout("slot_mapping", (("B", "s_slots"),), role="metadata"),
        TensorLayout("positions", (("B", "s_slots"),), role="metadata"),
        # The table's first axis is an implementation constant (max_pos).
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The group size, read by the boundary gate and the rotation's position
    # law ((position // compress_ratio) · compress_ratio); the plan's
    # `token_pool_compressor` marker binds it.
    structural_axes=("compress_ratio",),
    engine_inputs=(
        (0, TensorLayout("latent", ("B", "s_slots", "d_head"))),
    ),
    dtype_slots=("activation", "kv"),
    doc=("CSA2 compressed-cache store, one launch: the latent's last d_rope "
         "channels rotated GPT-J style at the group's first position, the "
         "row written into the paged compressed cache at each boundary "
         "row's slot (NoPE channels FP8 with per-64 UE8M0 exponents, the "
         "rotated tail bf16). vLLM `deepseek_v41.common.ops."
         "fused_compress_quant_cache.rope_quant_insert`. Memory-bound: the "
         "latent in, one cache record out per group."),
))

# DeepSeek-V4's overlap KV compressor, pool included (fusion_plans/
# fused_overlap_compress_norm_rope_store.yaml): the previous window's Ca
# half and this window's Cb half of the fp32 state rows — kv and
# gate-with-ape alike — softmax-pooled per channel, RMS-normed, the last
# `d_rope` channels rotated GPT-J style at the window's first position,
# and the row written into the paged cache, one launch
# (`DeepseekCompressor`, models/deepseek_v4/compressor.py:475). One span,
# two kernels, keyed by the head width the row itself carries: the main
# attention's 512-wide compressor runs the CuteDSL
# `SparseAttnCompressNormRopeStoreC4Kernel` (nvidia/ops/
# sparse_attn_compress_cutedsl.py:78), the indexer's 128-wide one the
# Triton `_fused_kv_compress_norm_rope_insert_indexer_attn`
# (common/ops/fused_compress_quant_cache.py:711). The window axis is the
# spec's own `Cape`; the launch's compress_ratio is that same number (4).
register_signature(OpSignature(
    name="fused_overlap_compress_norm_rope_store",
    binds_ops=('fused_overlap_compress_norm_rope_store',),
    inputs=(
        # One window's `Cape` state rows, each two halves (Ca|Cb) wide —
        # the engine emits kv and score as `[B, windows, Cape, 2·d_head]`.
        TensorLayout("kv", (("B", "s_slots", "Cape"), (2, "d_head"))),
        TensorLayout("score", (("B", "s_slots", "Cape"), (2, "d_head"))),
        TensorLayout("norm_weight", ("d_head",), role="weight"),
        # The compressed cache, one record per closed window.
        TensorLayout("kv_cache", (None, "kv_block_size", "d_head"),
                     role="kv"),
        TensorLayout("slot_mapping", (("B", "s_slots"),), role="metadata"),
        TensorLayout("positions", (("B", "s_slots"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    engine_inputs=(
        ("kv", TensorLayout("kv", ("B", "s_slots", "Cape", (2, "d_head")))),
        ("score", TensorLayout("score", ("B", "s_slots", "Cape", (2, "d_head")))),
    ),
    dtype_slots=("activation", "kv"),
    doc=("DeepSeek-V4 overlap compressor, one launch: softmax(gate) pool of "
         "the 2·Cape state half-rows of a window (the previous window's Ca, "
         "this window's Cb), rmsnorm·w over d_head, the last d_rope channels "
         "rotated at the window's first position, and the row quantised into "
         "the paged cache (the 584 B MLA record at d_head 512, the FP8 "
         "index-K record at 128). vLLM `SparseAttnCompressNormRopeStore"
         "C4Kernel` / `compress_norm_rope_store_triton(head_dim=128)`. "
         "The launch reads the fp32 state rows (the walk's fp32 `kv` slot "
         "on this row is those); the cell keys the window operands at the "
         "walk's activation slot. Memory-bound: the window rows in, one "
         "cache record out.")),
)

# DeepSeek-V4's no-overlap compressor tail (fusion_plans/
# fused_compress_norm_rope_store.yaml): the pooled rows of the previous
# launch RMS-normed, rotated and written into the paged compressed cache —
# the CuteDSL `SparseAttnNormRopeStoreKernel`
# (sparse_attn_compress_cutedsl.py:1453, from `DeepseekCompressor`,
# compressor.py:475, the second of the two launches the ratio-128
# compressor runs on CUDA). Same store as `fused_compress_rope_store` with
# the norm folded in and the V4 record (584 B a token). The same span is
# DeepSeek-V4.1's index key (`k_norm` .. `k_roped` stored by the indexer's
# `k_cache`): vLLM `indexer_k_norm_rope_store` (models/deepseek_v41/common/
# ops/indexer_k_store.py:29), 128-wide keys at a group of 1 or 2, written as
# the FP8 (132 B) or MXFP4 (68 B) index-key record. Both launches write a
# cached tensor as their output, so the record is the cell's `output`.
register_signature(OpSignature(
    name="fused_compress_norm_rope_store",
    binds_ops=('fused_compress_norm_rope_store',),
    inputs=(
        TensorLayout("kv", (("B", "s_slots"), "d_head")),
        TensorLayout("norm_weight", ("d_head",), role="weight"),
        TensorLayout("kv_cache", (None, "kv_block_size", "d_head"),
                     role="kv"),
        TensorLayout("slot_mapping", (("B", "s_slots"),), role="metadata"),
        TensorLayout("positions", (("B", "s_slots"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The group size (128 for this compressor): the boundary gate and the
    # rotation's position law read it, and the plan's `token_pool_compressor`
    # marker binds it.
    structural_axes=("compress_ratio",),
    engine_inputs=(
        (0, TensorLayout("kv", ("B", "s_slots", "d_head"))),
    ),
    dtype_slots=("activation", "output"),
    doc=("DeepSeek-V4 no-overlap compressor tail, one launch: rmsnorm·w of "
         "each pooled row over d_head, the last d_rope channels rotated "
         "GPT-J style at the group's first position, the row written into "
         "the paged compressed cache (NoPE channels FP8 with per-64 UE8M0 "
         "exponents, the rotated tail bf16). vLLM `SparseAttnNormRopeStore"
         "Kernel`; DeepSeek-V4.1's index key the same way at d_head 128, "
         "written as FP8 with one fp32 scale a key or MXFP4 with UE8M0 per "
         "32 (vLLM `indexer_k_norm_rope_store`). Memory-bound: the pooled "
         "rows in, one record out per group.")),
)

# GLM-5.3-Flash's k-pool indexer compressor (fusion_plans/
# fused_kpool_compress_store.yaml): each completed pool of `compress_ratio`
# index keys is softmax-pooled per channel by its gate with the position
# bias added, Hadamard-rotated, FP8-quantised (one pow2 scale a pool) and
# written into the paged index-K cache — one launch per phase in both
# frameworks (vLLM `kpool_compress_and_write_cache`, models/glm5next/
# nvidia/ops/kpool_compress.py:259, from `SparseAttnIndexerKpool`,
# glm5next/nvidia/sparse_indexer.py:221; sglang
# `kpool_softmax_rotate_write_cache`, srt/layers/attention/dsa/
# kpool_fp8_index.py:699, from `IndexerKPool._compress_write`,
# dsa_indexer_kpool.py:591). The engine operands come in the graph's own
# order — the gate windows, the position-bias table, the key windows.
register_signature(OpSignature(
    name="fused_kpool_compress_store",
    binds_ops=('fused_kpool_compress_store',),
    inputs=(
        TensorLayout("gate", (("B", "s_slots", "compress_ratio"),
                              "d_indexer_head")),
        TensorLayout("keys", (("B", "s_slots", "compress_ratio"),
                              "d_indexer_head")),
        # The learned position bias, one row per window slot.
        TensorLayout("ape", ("compress_ratio", "d_indexer_head"),
                     role="weight"),
        # The paged index-K cache: one pool a record (128 FP8 values and
        # one fp32 scale).
        TensorLayout("kv_cache", (None, "kv_block_size", "d_indexer_head"),
                     role="kv"),
        TensorLayout("slot_mapping", (("B", "s_slots"),), role="metadata"),
    ),
    engine_inputs=(
        ("gate", TensorLayout("gate", ("B", "s_slots", "compress_ratio",
                                  "d_indexer_head"))),
        ("pos_table", TensorLayout("ape", ("compress_ratio", "d_indexer_head"))),
        ("keys", TensorLayout("keys", ("B", "s_slots", "compress_ratio",
                                  "d_indexer_head"))),
    ),
    dtype_slots=("activation", "kv"),
    doc=("GLM k-pool indexer compressor, one launch: per pool of "
         "compress_ratio keys, softmax(gate + ape) per channel over the "
         "pool, the weighted sum of keys rounded to bf16, a Hadamard-128 "
         "rotation (bf16 again), FP8 e4m3 with one pow2 scale per pool, "
         "written into the paged index-K cache. vLLM `kpool_compress_and_"
         "write_cache` / sglang `kpool_softmax_rotate_write_cache`. "
         "Memory-bound: the pool windows in, one record out per pool.")),
)

# A Walsh-Hadamard rotation of each row's last axis (`hadamard_transform`):
# authored in a forward (DeepSeek-V4's indexer, usually absorbed into a fused
# launch) or grafted by a cache format whose `transform` is `hadamard`
# (GLM-5.3-Flash's indexer query, rotated into the cached key's basis —
# `inference/materialize_cast.py`). One per-row kernel over every leading
# axis folded (`fold_rmsnorm`, as a per-head cast folds its heads): sglang
# `rotate_activation` (srt/layers/attention/dsa/dsa_indexer.py:241 →
# kernels/ops/quantization/hadamard.py `hadamard_transform`), called at
# dsa_indexer_kpool.py:680 / :715.
register_signature(OpSignature(
    name="hadamard_transform",
    binds_ops=('hadamard_transform',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation",),
    shape_projection="fold_rmsnorm",
    doc=("Walsh-Hadamard rotation of each row's last axis, scaled by "
         "n^-1/2 (orthonormal): y = (H_n · x)/sqrt(n). n·log2(n) adds a "
         "row; memory-bound — the row in, the rotated row out (sglang "
         "`rotate_activation`)."),
))

# The same rotation and the cast after it in one launch
# (fusion_plans/fused_fwht128_quant.yaml): vLLM `fwht128_quant_fp8`
# (models/glm5next/nvidia/ops/kpool_compress.py:108), rotating GLM-5.3-
# Flash's indexer query into the cached key's basis and writing it FP8 with
# one pow2 scale per 128-wide head. `output` is the dtype the launch writes:
# the plan writes FP8 only where a quantise cast stands on the rotated
# query, which the graft does not materialise yet, so a cell keyed at the
# edge's own dtype is a launch this kernel does not make.
register_signature(OpSignature(
    name="fused_fwht128_quant",
    binds_ops=('fused_fwht128_quant',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation", "output"),
    shape_projection="fold_rmsnorm",
    doc=("Hadamard-128 rotation of each 128-wide row, then FP8 e4m3 with one "
         "pow2 F32 scale a row, in one launch (vLLM `fwht128_quant_fp8`). "
         "Memory-bound: the bf16 row in, FP8 + one scale out."),
))


# Qwen QSA's indexer preparation (fusion_plans/qsa_index_q_norm_rope.yaml
# and qsa_index_k_compress_store.yaml). The query side: per-head Gemma
# RMSNorm (norm · (1 + w), as the spec's own formula states) and the
# partial MRoPE over the first `d_rope` channels, re-joined. The key side:
# each completed block of `compress_ratio` raw keys mean-pooled (fp32 mean
# rounded to the storage dtype, as both kernels state), Gemma-normed,
# rotated at the block's first position NeoX style, and written into the
# compressed cache. One launch each in sglang
# (`qsa_index_q_norm_rope_store` / `qsa_index_k_compress_store`,
# kernels/ops/attention/qsa_indexer.py:69 / :143); in vLLM both are work
# items of the one `qsa_prepare` launch (models/qwen4_exp/nvidia/ops/
# qsa_prepare.py:504, kernel :112 — see the providers for how the two
# cells share it).
register_signature(OpSignature(
    name="qsa_index_q_norm_rope",
    binds_ops=('qsa_index_q_norm_rope',),
    inputs=(
        TensorLayout("q", (("B", "s_query"), "n_indexer_heads",
                           "d_indexer_head")),
        TensorLayout("weight", ("d_indexer_head",), role="weight"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        # The table's first axis is an implementation constant (max_pos).
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The head-split query the indexer's projection emits, head-first.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_indexer_heads", "s_query",
                               "d_indexer_head"))),
    ),
    dtype_slots=("activation",),
    doc=("QSA indexer query prep, one launch: per head of d_indexer_head, "
         "rmsnorm(q)·(1 + w) and the first d_rope channels rotated NeoX "
         "style at the token's position, re-joined. sglang `qsa_index_q_"
         "norm_rope_store` / vLLM `qsa_prepare` (the query work items). "
         "Memory-bound: q in, q out."),
))

register_signature(OpSignature(
    name="qsa_index_k_compress_store",
    binds_ops=('qsa_index_k_compress_store',),
    inputs=(
        TensorLayout("raw_key", (("B", "s_query"), "d_indexer_head")),
        TensorLayout("weight", ("d_indexer_head",), role="weight"),
        # The compressed cache, one row per completed block (vLLM pages
        # it; sglang's is a flat ring — its provider reads it as one
        # contiguous buffer).
        TensorLayout("kv_cache", (None, "kv_block_size", "d_indexer_head"),
                     role="kv"),
        TensorLayout("slot_mapping", (("B", "s_slots"),), role="metadata"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The block size: the plan's `view_split` / `token_pool_compressor`
    # binds it.
    structural_axes=("compress_ratio",),
    engine_inputs=(
        ("raw_key", TensorLayout("raw_key", ("B", "s_query", "d_indexer_head"))),
    ),
    dtype_slots=("activation", "kv"),
    doc=("QSA indexer key compression, one launch: per completed block of "
         "compress_ratio raw keys, the fp32 mean rounded to the storage "
         "dtype, rmsnorm·(1 + w) over d_indexer_head, the first d_rope "
         "channels rotated NeoX style at the block's first position, "
         "written into the compressed cache. sglang `qsa_index_k_compress_"
         "store` / vLLM `qsa_prepare` (the key work items). Memory-bound: "
         "the block's keys in, one cache row out.")),
)


register_signature(OpSignature(
    name="fused_qk_norm_rope",
    binds_ops=('fused_qk_norm_rope',),
    shape_projection="rope",
    inputs=(
        # One packed qkv [tokens, (n_heads+2·n_kv_heads)·d_rope].
        TensorLayout("qkv", (("B", "s_query"),
                             ("+", ("n_heads", "d_rope"),
                              (2, "n_kv_heads", "d_rope")))),
        TensorLayout("q_norm_w", ("d_rope",), role="weight"),
        TensorLayout("k_norm_w", ("d_rope",), role="weight"),
        TensorLayout("positions", (("B", "s_query"),), role="metadata"),
        # The cache's first dim is an implementation constant (max_pos), not
        # a cell axis.
        TensorLayout("cos_sin_cache", (None, "d_rope"), role="metadata"),
    ),
    # The query and the key at the plan's ports, head-shaped or packed
    # heads×width as the graph hands them; the rope projection reads each
    # off its layout (`_rope_identity`).
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_heads", "s_query", "d_rope"))),
        ("q", TensorLayout("q", ("B", "s_query", ("n_heads", "d_rope")))),
        ("k", TensorLayout("k", ("B", "n_kv_heads", "s_query", "d_rope"))),
        ("k", TensorLayout("k", ("B", "s_query", ("n_kv_heads", "d_rope")))),
    ),
    dtype_slots=("activation",),
    doc=("Per-head RMSNorm of Q and K (separate gains) followed by RoPE, "
         "fused over one packed qkv tensor "
         "[tokens, (n_heads+2·n_kv_heads)·d_rope] (V passes "
         "through). sglang `sgl_kernel.fused_qk_norm_rope` (base+YARN) / "
         "vLLM `_custom_ops.fused_qk_norm_rope` (precomputed cos/sin cache)."),
))

register_signature(OpSignature(
    name="moe_gate",
    binds_ops=('moe_fused_gate',),
    shape_projection="moe_gate",
    inputs=(
        TensorLayout("gating", (("B", "s_query"), "n_experts")),
        TensorLayout("bias", ("n_experts",), role="metadata"),
    ),
    # The router's operands at its plan's ports: the activation it scores
    # (its width is the scorer's, no axis of this cell) and the per-expert
    # correction bias, whose length is the experts (`_moe_gate_identity`).
    engine_inputs=(
        ("gating", TensorLayout("gating", ("B", "s_query", None))),
        ("bias", TensorLayout("bias", ("n_experts",))),
    ),
    # The routing widths (top-k experts, group count, top groups) parameterize
    # the selection but are no tensor's axis.
    structural_axes=("n_experts_active", "n_expert_groups", "n_groups_active"),
    # The group structure picks WHICH experts win, not how much work it is, so
    # models that group differently share one measurement of the same router.
    _identity_dims=("B", "s_query", "n_experts", "n_experts_active"),
    dtype_slots=("activation",),
    doc=("Aux-loss-free grouped top-k MoE router (DeepSeek-V3 noaux_tc): "
         "sigmoid(gate[tokens, n_experts]) + correction bias, per-group top-2 sum, "
         "pick `topk_group` groups, mask, top-k experts, gather weights from "
         "the unbiased scores, renormalize + routed scaling. One kernel: "
         "vLLM `_moe_C.grouped_topk` / sglang `moe_fused_gate`. Returns "
         "topk_weights [tokens, topk] + topk_ids (discrete)."),
))

# Ungrouped top-k MoE routers, each one launch from the router logits to the
# top-k ids and weights — the fused spans `moe_topk_softmax`,
# `moe_topk_sigmoid`, `moe_topk_softmax_bias` and `moe_topk_sqrtsoftplus`
# (fusion_plans/…). One signature per span, as `gelu_and_mul` is beside
# `silu_and_mul`: the same operands and bytes, a different scoring function,
# and the oracle has to compute the one the kernel does. The router reads
# `gating [tokens, n_experts]` (and the selection bias where the span adds
# one) and writes `[tokens, n_experts_active]` ids and weights; `top-k` is
# the output's width. `moe_gate` stays the GROUPED router (noaux_tc).
def _moe_topk(name: str, biased: bool, doc: str) -> None:
    inputs = [TensorLayout("gating", (("B", "s_query"), "n_experts"))]
    if biased:
        inputs.append(TensorLayout("bias", ("n_experts",), role="weight"))
    register_signature(OpSignature(
        name=name,
        binds_ops=(name,),
        inputs=tuple(inputs),
        structural_axes=("n_experts_active",),
        # A biased router's launch reads two ports (`gating`, `bias`) and
        # names the one it is keyed by; an unbiased one has one operand.
        engine_inputs=(
            ("gating" if biased else 0,
             TensorLayout("gating", ("B", "s_query", "n_experts"))),
        ),
        output_axes=("n_experts_active",),
        dtype_slots=("activation",),
        doc=doc,
    ))


_moe_topk(
    "moe_topk_softmax", False,
    "Softmax top-k router (Qwen-MoE / Mixtral): p = softmax(gating), "
    "top-k of p, weights gathered from p, renormalised. vLLM "
    "`ops.topk_softmax` (fused_topk) / sglang Triton `moe_fused_gate("
    "scoring_func=\"softmax\")`. Memory-bound.")
_moe_topk(
    "moe_topk_sigmoid", True,
    "Ungrouped sigmoid top-k router with a selection bias: s = sigmoid("
    "gating), top-k of s + bias, weights gathered from s, renormalised, "
    "scaled. vLLM `ops.topk_sigmoid` (fused_topk_bias) / sglang Triton "
    "`moe_fused_gate(scoring_func=\"sigmoid\")`. Memory-bound.")
_moe_topk(
    "moe_topk_softmax_bias", True,
    "Softmax top-k router with a selection bias added after the softmax "
    "(LongCat-Flash): top-k of softmax(gating) + bias, weights gathered from "
    "the unbiased probabilities. vLLM `ops.topk_softmax` with "
    "`e_score_correction_bias` (fused_topk_bias). Memory-bound.")
_moe_topk(
    "moe_topk_sqrtsoftplus", True,
    "DeepSeek-V4 router: s = sqrt(softplus(gating)), top-k of s + bias, "
    "weights gathered from s, renormalised, scaled. vLLM `dsv4_topk` / "
    "`ops.topk_hash_softplus_sqrt` (fused_topk_bias) / sglang Triton "
    "`moe_fused_gate(scoring_func=\"sqrtsoftplus\")`. Memory-bound.")

register_signature(OpSignature(
    name="moe_mega_gate_sqrtsoftplus",
    binds_ops=('moe_mega_gate_sqrtsoftplus',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
        TensorLayout("weight", ("n_experts", "d_model"), role="weight"),
        TensorLayout("bias", ("n_experts",), role="weight"),
    ),
    structural_axes=("n_experts_active",),
    # The walk hands the hidden row and the selection bias; the router
    # weight is the fused node's own parameter, its extents the two axes
    # those operands already name.
    engine_inputs=(
        ("x", TensorLayout("x", ("B", "s_query", "d_model"))),
        ("bias", TensorLayout("bias", ("n_experts",))),
    ),
    output_axes=("n_experts_active",),
    dtype_slots=("activation", "weight"),
    doc=("DeepInfer MegaGate (DeepGEMM `bf16_mega_gate`): the router GEMM "
         "and the whole biased sqrt-softplus top-k / renormalise / scale "
         "chain in one launch, over a BF16 hidden row and a BF16 router "
         "weight. Renormalisation and the routed scale are the oracle's "
         "convention, as on `moe_topk_sqrtsoftplus`. Memory-bound at "
         "V4.1's expert count."),
))

# DeepSeek-V4's hash-routed layers: the experts come from a token-id table,
# the weights from the sqrt-softplus scores — one launch (the fused span
# `moe_topk_hash_sqrtsoftplus`). The table `[n_vocab, n_experts_active]` is
# an operand the launch reads (one row per token), so its extents are cell
# axes; the fused node's operands are gating, input_ids, table.
register_signature(OpSignature(
    name="moe_topk_hash_sqrtsoftplus",
    binds_ops=('moe_topk_hash_sqrtsoftplus',),
    inputs=(
        TensorLayout("gating", (("B", "s_query"), "n_experts")),
        TensorLayout("input_ids", (("B", "s_query"),), role="metadata"),
        TensorLayout("table", ("n_vocab", "n_experts_active"),
                     role="metadata"),
    ),
    engine_inputs=(
        ("gating", TensorLayout("gating", ("B", "s_query", "n_experts"))),
        ("table", TensorLayout("table", ("n_vocab", "n_experts_active"),
                         role="metadata")),
    ),
    dtype_slots=("activation",),
    doc=("DeepSeek-V4 hash router: experts = table[input_id] (top-k fixed "
         "by the token), weights = sqrt(softplus(gating)) gathered at them, "
         "renormalised and scaled. vLLM `ops.topk_hash_softplus_sqrt` with "
         "the table (fused_topk_bias) / sglang `hash_topk` (JIT). "
         "Memory-bound."),
))

register_signature(OpSignature(
    name="fused_engram_gate",
    binds_ops=('fused_engram_gate',),
    inputs=(
        TensorLayout("x", ("B", "s_query",
                           ("n_hidden_copies", "d_model"))),
        TensorLayout("kv", ("B", "s_query",
                            ("+", ("n_hidden_copies", "d_model"),
                             "d_model"))),
        TensorLayout("q_weight", ("n_hidden_copies", "d_model"),
                     role="weight"),
        TensorLayout("k_weight", ("n_hidden_copies", "d_model"),
                     role="weight"),
    ),
    engine_inputs=(
        # The residual stream, at its port; the wkv output beside it is the
        # `kv` port, which the cell does not read.
        ("x", TensorLayout("x", ("B", "s_query",
                               ("n_hidden_copies", "d_model")))),
    ),
    dtype_slots=("activation", "weight"),
    doc=("DeepSeek-V4.1 Engram after its wkv GEMM, one launch: split the "
         "per-copy key and shared value, normalize x and each key, form "
         "the scaled dot gate, and add gate·value to every residual copy. "
         "The signed-sqrt floor is 1e-6. vLLM and sglang run the same "
         "function as a Triton launch."),
))

register_signature(OpSignature(
    name="ple_gate",
    binds_ops=('ple_gate',),
    inputs=(
        TensorLayout("key", ("B", "s_query",
                             ("n_hidden_copies", "d_model"))),
        TensorLayout("value", ("B", "s_query", "d_model")),
        TensorLayout("x", ("B", "s_query",
                           ("n_hidden_copies", "d_model"))),
        TensorLayout("w_key", ("n_hidden_copies", "d_model"),
                     role="weight"),
        TensorLayout("w_query", ("n_hidden_copies", "d_model"),
                     role="weight"),
        TensorLayout("w_conv", ("n_hidden_copies", "d_model"),
                     role="weight"),
    ),
    engine_inputs=(
        ("key", TensorLayout("key", ("B", "s_query",
                                 ("n_hidden_copies", "d_model")))),
        ("query", TensorLayout("x", ("B", "s_query",
                               ("n_hidden_copies", "d_model")))),
        ("value", TensorLayout("value", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation", "weight"),
    doc=("Qwen PLE retrieval gate, one launch: grouped RMSNorm of key and "
         "query, signed-sqrt sigmoid of their scaled dot, the gated value, "
         "and the grouped RMSNorm feeding the short convolution. The launch "
         "writes both the gated residual and the conv input; the signed-"
         "sqrt floor is 1e-6. vLLM `qwen4_exp_ple_gate`."),
))

register_signature(OpSignature(
    name="indexer_logits",
    binds_ops=('indexer_logits',),
    inputs=(
        # The kernel-side operands. Two facts live in this tuple and they
        # are independent: the QUERY's layout (whose per-head form
        # `takeover-idx` declares) and the KEYS' — each request has its
        # own, which is this branch's correction. A later change to either
        # leaves the other as it is.
        TensorLayout("q", (("B", "s_query"), "n_indexer_heads",
                           "d_indexer_head")),
        # Each request's own keys: decode scores B single queries against B
        # caches, not one cache shared by the batch.
        TensorLayout("k", ("B", "s_kv", "d_indexer_head"), role="kv"),
        TensorLayout("weights", (("B", "s_query"), "n_indexer_heads"),
                     role="metadata"),
    ),
    # Graph order is the query per head, the keys, the head weights.
    # Query-side sequence is s_query (decode: the verify window). Indexer-K
    # is the cache, s_kv. Those two axes must not share a name — prefill
    # hides the mix because they are equal; decode is s_query=1 against
    # s_kv=cache. The head-first 4-D query is the same in both writings.
    # At the plan's ports (q, k, w); `indexer_logits` is a primitive op as
    # well, whose three operands are those ports in turn
    # (`primitive_ports`). A launch without the head weights (Qwen3.8's QSA
    # indexer) has no `w`, and its layouts are not read.
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_indexer_heads", "s_query",
                               "d_indexer_head"))),
        ("k", TensorLayout("k", ("B", "s_kv", "d_indexer_head"))),
        ("w", TensorLayout("weights", ("B", "s_query", "n_indexer_heads"))),
        # DSv4 CSA indexer: compressed K after the local compressor (s_kv
        # already the pooled length), broadcast over the heads.
        ("k", TensorLayout("k", ("B", "n_indexer_heads", "s_kv",
                               "d_indexer_head"))),
        # Broadcast gate weights aligned to the score tensor
        # [B, n_indexer_heads, s_query, s_kv] (DSv4 CSA indexer).
        ("w", TensorLayout("weights", ("B", "n_indexer_heads", "s_query",
                                     "s_kv"), role="metadata")),
    ),
    primitive_ports=("q", "k", "w"),
    dtype_slots=("activation",),
    doc=("DeepSeek sparse-attention (DSA) lightning-indexer logits: per "
         "(query token, kv token) a gate-weighted sum over n_indexer_heads "
         "(each of dim Ih) of the dot product between a low-rank query "
         "projection and the indexer-K, used to select the top-k KV tokens "
         "for the main sparse attention. Production kernel is DeepGEMM "
         "`fp8_paged_mqa_logits` (paged, FP8, Blackwell); the torch dense "
         "reference computes it on any GPU."),
))

register_signature(OpSignature(
    name="indexer_block_score",
    binds_ops=('indexer_block_score',),
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_indexer_heads",
                           "d_indexer_head")),
        TensorLayout("k", ("B", "s_kv", "d_indexer_head"), role="kv"),
    ),
    # The block width is the kernel's argument, no tensor's axis.
    structural_axes=("s_sparse_block",),
    # The fused node reads the head-first query and the one index key shown
    # to every index head (a zero-byte head broadcast).
    engine_inputs=(
        ("q", TensorLayout("q", ("B", "n_indexer_heads", "s_query",
                               "d_indexer_head"))),
        ("k", TensorLayout("k", ("B", "n_indexer_heads", "s_kv",
                               "d_indexer_head"))),
    ),
    dtype_slots=("activation",),
    doc=("MiniMax-M3 MSA block-max indexer scores: per index head, query "
         "token and block of s_sparse_block keys, the max of q·k over the "
         "block's keys not after the query (causal; blocks past the query "
         "floor to a sentinel) — [B, n_indexer_heads, s_query, "
         "ceil(s_kv / s_sparse_block)] fp32, the [.., s_kv] score matrix "
         "never landing in HBM. vLLM "
         "`minimax_m3_index_score`, sglang `flash_prefill_with_topk_index`; "
         "the torch dense reference computes it on any GPU."),
))

register_signature(OpSignature(
    name="embedding",
    binds_ops=('embedding',),
    inputs=(
        TensorLayout("ids", ("B", "s_query"), role="metadata"),
        TensorLayout("w", ("n_vocab", "d_model"), role="weight"),
    ),
    engine_inputs=(
        (0, TensorLayout("ids", ("B", "s_query"))),
        # Source leaf: some embeddings arrive as a one-hot / sampled
        # vocab axis [B, s_query, n_vocab] rather than integer ids.
        (0, TensorLayout("ids", ("B", "s_query", "n_vocab"))),
    ),
    output_axes=("d_model",),
    dtype_slots=("weight",),
    doc=("Vocab gather: token ids [B, s_query] index a table "
         "[n_vocab, d_model]. Memory-bound; no matmul."),
))

register_signature(OpSignature(
    name="per_head_gemm",
    binds_ops=('per_head_linear',),
    inputs=(
        TensorLayout("x", ("B", "n_heads", "s_query", "d_reduce")),
        TensorLayout("w", ("n_heads", "d_out", "d_reduce"), role="weight"),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "n_heads", "s_query", "d_reduce"))),
    ),
    output_axes=("d_out",),
    dtype_slots=("activation", "weight"),
    doc=("Per-head dense linear: n_heads independent GEMMs over "
         "[B, n_heads, s_query, d_reduce]. MLA absorbed W_uk / W_uv."),
))

register_signature(OpSignature(
    name="layernorm",
    binds_ops=('layernorm',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
        TensorLayout("w", ("d_model",), role="weight"),
        TensorLayout("b", ("d_model",), role="weight"),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_model"))),
    ),
    dtype_slots=("activation",),
    shape_projection="fold_rmsnorm",
    doc=("Affine layer-norm over the last dim of [B·s_query, d_model] "
         "(mean/var + γ, β). Distinct from rmsnorm."),
))

register_signature(OpSignature(
    name="dsa_topk_mask",
    binds_ops=('dsa_topk_mask',),
    inputs=(
        TensorLayout("logits", ("B", "s_query", "s_kv")),
    ),
    structural_axes=("n_topk_keys",),
    engine_inputs=(
        (0, TensorLayout("logits", ("B", "s_query", "s_kv"))),
    ),
    output_axes=("n_topk_keys",),
    dtype_slots=("activation",),
    doc=("DSA stage-2 top-k over indexer logits: int32 indices "
         "[B, s_query, n_topk_keys] of the selected cache slots."),
))

register_signature(OpSignature(
    name="softmax_act",
    binds_ops=('softmax_act',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "n_experts")),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "n_experts"))),
        (0, TensorLayout("x", (("B", "s_query"), "n_experts"))),
    ),
    dtype_slots=("activation",),
    doc=("Last-dim softmax for an unfused MoE router "
         "[B·s_query, n_experts]."),
))

register_signature(OpSignature(
    name="topk",
    binds_ops=('topk',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "n_experts")),
    ),
    structural_axes=("n_experts_active",),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "n_experts"))),
        (0, TensorLayout("x", (("B", "s_query"), "n_experts"))),
    ),
    output_axes=("n_experts_active",),
    dtype_slots=("activation",),
    doc=("Top-k over the expert axis of an unfused router. "
         "k is n_experts_active."),
))

register_signature(OpSignature(
    name="gather",
    binds_ops=('gather',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "n_experts")),
        TensorLayout("idx", (("B", "s_query"), "n_experts_active"),
                     role="metadata"),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "n_experts"))),
        (0, TensorLayout("x", (("B", "s_query"), "n_experts"))),
        (1, TensorLayout("idx", ("B", "s_query", "n_experts_active"))),
        (1, TensorLayout("idx", (("B", "s_query"), "n_experts_active"))),
    ),
    dtype_slots=("activation",),
    # Writes the scores it picks, unchanged (the arch op's `gather_source`).
    gathers_from=0,
    doc=("Index-gather of router scores at the selected experts."),
))


def _unary_act(name: str, doc: str,
               config_params: tuple = ()) -> None:
    register_signature(OpSignature(
        name=name,
        binds_ops=(name,),
        config_params=config_params,
        inputs=(
            TensorLayout("x", (("B", "s_query"), "d_model")),
        ),
        engine_inputs=(
            (0, TensorLayout("x", ("B", "s_query", "d_model"))),
            (0, TensorLayout("x", (("B", "s_query"), "d_model"))),
        ),
        dtype_slots=("activation",),
        shape_projection="trailing_feature",
        doc=doc,
    ))


_unary_act("silu", "SiLU / Swish, last-dim elementwise.")
_unary_act("gelu", "GELU, last-dim elementwise.")
_unary_act("gelu_tanh", "GELU (tanh approximation), last-dim elementwise.")
_unary_act("sigmoid", "Logistic sigmoid, last-dim elementwise.")
_unary_act("softplus", "Softplus, last-dim elementwise.")
_unary_act("sqrt", "Square root, last-dim elementwise (DSv4 sqrtsoftplus router).")
_unary_act("neg_exp", "exp(−x), last-dim elementwise (GDN A_log).")
_unary_act("relu_squared", "relu(x)², last-dim elementwise (Nemotron-H).")
# Two pointwise activations whose own launch a plan used to drop. The bar for
# declaring one is an AUDITED serving path that launches it alone — not the
# absence of a binding for the span around it, which is our own coverage gap
# and says nothing about what a framework dispatches.
#
#   `tanh`: Gemma 2 / Gemma 4's final-logit soft-cap. sglang fuses it
#   (`softcap_logits`, bound); vLLM 0.32.0 runs the three eager ops divide /
#   tanh / multiply in `LogitsProcessor.forward`
#   (model_executor/layers/logits_processor.py:114-116, cited by
#   `fusion_plans/softcap_logits.yaml`) and binds nothing. The two
#   `scale_const` launches around it were already cells; this one was not.
#
#   `signed_sqrt`: the Qwen3.8 PLE retrieval gate's magnitude. vLLM fuses
#   the whole span (`ple_gate`, bound); sglang 0.5.22 "runs the norms, the
#   gate and the value product as separate ops"
#   (srt/models/qwen4_exp.py:1349-1386, cited by
#   `fusion_plans/ple_gate.yaml`), and its gate line is
#   `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()` (:1377) — the five
#   elementwise launches the op's own cost coefficient prices. DeepSeek-V4.1's
#   engram gate uses the same primitive and is NOT the justification: all
#   three bindings fuse that span (`fused_engram_gate`).
_unary_act("tanh", "tanh(x), last-dim elementwise — the unfused middle of a "
                   "final-logit soft-cap (Gemma 2 / Gemma 4 under vLLM).")
_unary_act("signed_sqrt",
           "sign(x)·sqrt(max(|x|, 1e-6)), last-dim elementwise — the Qwen3.8 "
           "PLE retrieval gate's magnitude, as sglang lowers it.")
# The gate's own beta, the same constant vLLM reads for the fused
# `situ_and_mul` — `tanh(x)·sigmoid(x)` is this function only at beta 1,
# which is no published model's value.
_unary_act("situ", "β·tanh(x/β)·sigmoid(x), last-dim (Kimi-K3 gate).",
           config_params=(("situ_beta", "situ_beta"),))
_unary_act("scale_const", "Elementwise scale by a config-time constant.")
_unary_act("rmsnorm_unscaled", "RMSNorm without learnable gain.")
_unary_act("normalize_sum_one", "x / (Σx + ε) over the last feature axis.")
# One activation rotated at each token's position: MLA's and DSA's rope
# slice, and the inverse rotation DeepSeek-V4 applies to the attention
# output. The rotation is `rope`'s, on one tensor rather than on q and k,
# so the operands are its: the activation, the per-token positions, and the
# cos/sin table the rotation reads. Declared as the lone `x` of a pointwise
# act, no provider could do more than return it — which is what the torch
# baseline did, and the identity-rotation reference agreed with it.
def _rope_emb_act(name: str, doc: str) -> None:
    register_signature(OpSignature(
        name=name,
        binds_ops=(name,),
        inputs=(
            TensorLayout("x", (("B", "s_query"), "d_model")),
            TensorLayout("positions", (("B", "s_query"),), role="metadata"),
            # The table's first axis is an implementation constant
            # (max_pos), not a cell axis — as in `rope`.
            TensorLayout("cos_sin_cache", (None, "d_model"), role="metadata"),
        ),
        engine_inputs=(
            (0, TensorLayout("x", ("B", "s_query", "d_model"))),
            (0, TensorLayout("x", (("B", "s_query"), "d_model"))),
        ),
        dtype_slots=("activation",),
        # A per-head `[B, n_heads, s_query, d]` rotation is the same kernel
        # over B·n_heads·s_query rows, which `trailing_feature` folds into
        # the token count — so the head axis needs no layout of its own.
        shape_projection="trailing_feature",
        doc=doc,
    ))


_rope_emb_act("rope_emb",
              "Rotary embedding on one activation: rotate its last-dim pairs "
              "by cos/sin at each token's position.")
_rope_emb_act("rope_emb_inverse",
              "Inverse rotary embedding (conjugate rotation) on one "
              "activation — undoes `rope_emb` at the same positions.")

# A reduction is not pointwise: its cost is the axis it collapses, and the
# graph collapses any axis (`axis:` on the node — the DSv4 hyper-connection
# copies at 2, the indexer heads at 1, a gate's feature at -1). Measured as
# `[rows, d_reduce, d_model]` summed over the middle axis, `d_model` the
# contiguous extent behind the reduced one (1 when the last axis reduces).
# Declared as a last-axis unary act it was timed reducing the wrong, 4×
# smaller axis while billed for the graph's.
register_signature(OpSignature(
    name="sum_reduce",
    binds_ops=("sum_reduce",),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_reduce", "d_model")),
    ),
    dtype_slots=("activation",),
    shape_projection="reduce_axis",
    doc="Sum over one axis: x[rows, d_reduce, d_model] → y[rows, d_model].",
))


def _binary_act(name: str, doc: str, first=0) -> None:
    """`first`: the operand `a` is — a primitive op's position 0, or the
    port a fused launch reads it at."""
    register_signature(OpSignature(
        name=name,
        binds_ops=(name,),
        inputs=(
            TensorLayout("a", (("B", "s_query"), "d_model")),
            TensorLayout("b", (("B", "s_query"), "d_model")),
        ),
        engine_inputs=(
            (first, TensorLayout("a", ("B", "s_query", "d_model"))),
            (first, TensorLayout("a", (("B", "s_query"), "d_model"))),
        ),
        dtype_slots=("activation",),
        shape_projection="trailing_feature",
        doc=doc,
    ))


_binary_act("mul", "Pointwise multiply of two activations, last-dim aligned.")
_binary_act("add", "Pointwise add of two activations, last-dim aligned.")
# The attention output gate in one elementwise launch
# (fusion_plans/sigmoid_gate_mul.yaml): the same two operands as `mul` and
# the same bytes, a different function — so its own cell, as `gelu_and_mul`
# sits beside `silu_and_mul`. `a` is the attention output, the plan's `x`
# port; `b` the gate.
_binary_act("sigmoid_gate_mul",
            "Attention output gate, one launch: a · sigmoid(b), last-dim "
            "aligned (a the merged attention output, b the gate projection). "
            "sglang `fused_sigmoid_mul` (Qwen3.5 / Qwen4Exp) and "
            "`sigmoid_gate_mul` (HY V4). Memory-bound.", first="x")
# The final-logit soft-cap in one launch (fusion_plans/softcap_logits.yaml):
# `c · tanh(x / c)` over the vocabulary, written fp32. A unary act like the
# others here; the cap `c` moves no byte and is stated on the reference's
# inputs (see `refs`).
_unary_act("softcap_logits",
           "Final-logit soft-cap, one launch: c·tanh(x/c) over the vocabulary "
           "row, written in fp32 (sglang `softcap_to_float32_logits`, Gemma 2 "
           "/ Gemma 4). Memory-bound.")

register_signature(OpSignature(
    name="hash_route",
    binds_ops=('hash_route',),
    inputs=(
        # The lookup key, one token id per token: the reference gathers
        # tid2eid[ids].
        TensorLayout("ids", (("B", "s_query"),), role="metadata"),
        TensorLayout("tid2eid", ("n_vocab", "n_experts_active"),
                     role="metadata"),
    ),
    engine_inputs=(
        (0, TensorLayout("ids", ("B", "s_query"))),
        (0, TensorLayout("ids", (("B", "s_query"),))),
    ),
    # Both operands are integer index tensors (token ids, the table).
    dtype_slots=("metadata",),
    doc=("Hash-MoE router: token id → expert ids via a fixed tid2eid table."),
))

register_signature(OpSignature(
    name="hc_split_sinkhorn",
    binds_ops=('hc_split_sinkhorn',),
    inputs=(
        TensorLayout("mixes", ("B", "s_query", "d_hc_mix")),
        TensorLayout("scale", (3,), role="metadata"),
        # Cast to the mix's own dtype by the provider — an activation.
        TensorLayout("base", ("d_hc_mix",), role="activation"),
    ),
    engine_inputs=(
        (0, TensorLayout("mixes", ("B", "s_query", "d_hc_mix"))),
    ),
    structural_axes=("n_hidden_copies",),
    dtype_slots=("activation",),
    doc=("Hyper-Connection Sinkhorn projection over the mix axis, then split "
         "into pre/post/comb coefficients. DSv4 hc_pre."),
))

register_signature(OpSignature(
    name="hc_mix",
    binds_ops=('hc_mix',),
    inputs=(
        TensorLayout("comb", ("B", "s_query", "n_hidden_copies",
                              "n_hidden_copies")),
        TensorLayout("residual", ("B", "s_query", "n_hidden_copies",
                                  "d_model")),
    ),
    engine_inputs=(
        (1, TensorLayout("residual", ("B", "s_query", "n_hidden_copies",
                                      "d_model"))),
        (0, TensorLayout("comb", ("B", "s_query", "n_hidden_copies",
                                  "n_hidden_copies"))),
    ),
    dtype_slots=("activation",),
    doc=("Hyper-Connection residual mix: comb @ residual over the copy axis. "
         "DSv4 hc_post."),
))

# The cast a quantised matmul runs on its activation operand, as its own
# launch. The frameworks make it one: vLLM quantises in
# `kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:119` and runs the
# GEMM at `:132`; sglang in `srt/layers/quantization/fp8_utils.py:1355` with
# the GEMM after it. Zero FLOPs and non-zero bytes, which is exactly why it
# needs a signature: a row with no provider is UNPLACED, and the measured
# total is all-or-nothing per phase (`test_measured_coverage`), so leaving
# this unbound would take the measured time off every model with an FP8 GEMM.
#
# ONE signature for both directions. The math contract is the same function —
# the activation as the narrow format can hold it — and so is the oracle:
# dequantise(quantise(x)) IS "x on the format's grid", whichever end the
# launch starts from. Which end it starts from is already in the cell (the
# `activation` slot is what it reads, `output` what it writes), so a second
# signature for the inverse would be two names for one piece of math and two
# places to keep its reference in step.
#
# `fold_rmsnorm`: a cast over a per-head stream (`[B, s_query, n_heads,
# d_head]` — DeepSeek's indexer query quant, MLA's decode query) is the same
# per-row kernel over B·s_query·n_heads rows, so the head axis folds into
# the token count exactly as it does for the per-head norms. The contract
# path is tried first, so a rank-3 activation never reaches it.
register_signature(OpSignature(
    name="quantise_activation",
    binds_ops=('quantise_activation',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_model"))),
    ),
    # `output` is the format written, `metadata` the dtype of the scales
    # written beside it, and both change the kernel: an e8m0 scale is a
    # power of two and quantises to other values than an exact fp32 one.
    # The elements that SHARE a scale are the `activation_granularity` slot
    # of the same environment (`ActivationFormat.granularity`), which is not
    # a dtype and so is not named here.
    dtype_slots=("activation", "output", "metadata"),
    shape_projection="fold_rmsnorm",
    doc=("Cast an activation between a wide dtype and a narrow quantised "
         "one at the declared granularity — one dynamic scale per token, or "
         "one per group of channels along the row — writing the elements and "
         "their scales. Zero FLOPs; its cost is the wide side read plus the "
         "narrow side and the scales written. The direction is the cell's: "
         "`activation` is the dtype read, `output` the dtype written, so "
         "quantise and dequantise are two cells of one signature."),
))

register_signature(OpSignature(
    name="gla_simple_recurrence",
    binds_ops=('gla_simple_recurrence',),
    inputs=(
        TensorLayout("q", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("k", ("B", "s_query", "n_heads", "d_head")),
        TensorLayout("v", ("B", "s_query", "n_heads", "d_head")),
        # The per-head decay RATE: S is multiplied by exp(-slope) each step.
        # The graph carries no tensor for it (Ling's `slope_rate` is a
        # non-persisted buffer built from the layer index), but the op's own
        # formula has the decay and every serving kernel takes it: vLLM
        # v0.32.0 `slope_rate=self.tp_slope` (bailing_linear_attn.py:653,
        # :691-703), sglang v0.5.22 `decay_scales=self.tp_slope[...]`
        # (lightning_backend.py:288-301). Its value changes neither bytes nor
        # work, so the reference draws it, and the cross-check then also
        # checks that the decay reaches the kernel the right way round.
        TensorLayout("slope", ("n_heads",), role="weight"),
        # The recurrent state the launch reads and writes back — the same
        # declaration `gated_delta_rule` makes. The walk bills it on this
        # cell (the component's `recurrent_state` entry, carrier
        # `gla_recurrence`); a launch built without it timed a ~2 MB kernel
        # against a ~130 MB numerator. Layout [B, H, K, V], K = V = d_head.
        TensorLayout("initial_state",
                     ("B", "n_heads", "d_head", "d_head"),
                     role="metadata"),
    ),
    # Heads FIRST, as the op's own signature lays its operands out
    # (gla_qknorm_partial_rope_sigmoid_gated_attention.yaml custom_ops:
    # `q: [B, H, s_ctx, d_head]`, which is what `head_split` emits). This
    # used to read the engine tensor as [B, s_query, n_heads, d_head], so
    # the cell swapped the two axes: Ling's tp=1 decode step (64 heads, one
    # token) was keyed and measured as one head over 64 tokens.
    engine_inputs=(
        (0, TensorLayout("q", ("B", "n_heads", "s_query", "d_head"))),
        (2, TensorLayout("v", ("B", "n_heads", "s_query", "d_head"))),
    ),
    dtype_slots=("activation",),
    doc=("Gated linear-attention recurrence (FLA simple_gla / Lightning): "
         "S_t = exp(-slope)·S_{t-1} + kᵀv; o_t = (q/√d)·S_t, one decay "
         "rate per head. Constant state, no growing KV: the launch reads the "
         "per-sequence state [B, H, d, d] and writes the final one back "
         "(vLLM `linear_attention_decode` / lightning prefill, sglang "
         "`seg_la_fwd`, FLA `chunk_simple_gla` / "
         "`fused_recurrent_simple_gla`)."),
))

register_signature(OpSignature(
    name="step3p5_headwise_gate",
    binds_ops=('headwise_gate',),
    inputs=(
        TensorLayout("attn", ("B", "n_heads", "s_query", "d_head")),
        TensorLayout("gate", ("B", "s_query", "n_heads")),
    ),
    engine_inputs=(
        (0, TensorLayout("attn", ("B", "n_heads", "s_query", "d_head"))),
    ),
    dtype_slots=("activation",),
    doc="Per-head scalar gate broadcast over attention output (Step-3.5/3.7).",
))

register_signature(OpSignature(
    name="attn_res_combine",
    binds_ops=('attn_res_combine',),
    inputs=(
        TensorLayout("prefix", ("B", "s_query", "d_model")),
        TensorLayout("stack", ("B", "s_query", "Sar", "d_model")),
    ),
    engine_inputs=(
        (0, TensorLayout("prefix", ("B", "s_query", "d_model"))),
        (1, TensorLayout("stack", ("B", "s_query", "Sar", "d_model"))),
    ),
    dtype_slots=("activation",),
    doc=("Kimi-K3 AttnRes combine: softmax over the history of "
         "block entries plus the running prefix, scored by the call's "
         "`*_res_norm` / `*_res_proj` weights, then a weighted sum. "
         "Bandwidth-bound; torch floor."),
))

register_signature(OpSignature(
    name="position_table_add",
    binds_ops=('position_table_add',),
    inputs=(
        TensorLayout("x", (("B", "s_query"), "d_model")),
        # Cast to the cell's activation dtype by the provider.
        TensorLayout("table", ("d_model",), role="activation"),
    ),
    engine_inputs=(
        (0, TensorLayout("x", ("B", "s_query", "d_model"))),
        (0, TensorLayout("x", (("B", "s_query"), "d_model"))),
    ),
    dtype_slots=("activation",),
    doc="Add a learned position table onto an activation, last-dim aligned.",
))


# ---------------------------------------------------------------------------
# The hyper-connection sublayer boundary — the "mHC seam", one signature per
# launch (`fusion_plans/mhc_{post,pre_gemm,pre_gemm_stream,pre_big_fuse,
# fused_post_pre}.yaml`).
#
# A manifold-hyper-connection model keeps `n_hidden_copies` parallel residual
# copies and mixes them at every sublayer boundary. Both frameworks run that
# boundary as two or three TileLang launches, and each of those launches is
# one of these signatures.
#
# Every one of them reads or writes the `[tokens, n_hidden_copies, d_model]`
# residual, so the cell is `{B, s_query, n_hidden_copies, d_model}` and
# nothing else. The mix vector a pre-mix projection writes is NOT a fifth
# axis: it is `n_hidden_copies² + 2·n_hidden_copies` by construction (the
# cross-mix matrix plus the `pre` and `post` gates), so the contract states it
# as that expression. Declaring it as its own axis would leave the big fuse —
# whose engine operands are the `pre` coefficients and the residual, neither
# of which carries the mix width — unable to bind it, and every one of its
# cells would drop.
#
# The tensors are declared token-flattened, as the frameworks view them
# before every one of these launches.
# ---------------------------------------------------------------------------

#: The step's tokens, packed into one physical dim.
_HC_TOKENS = ("B", "s_query")
#: The width of one token's mix vector: the `n_hidden_copies²` cross-mix
#: coefficients plus two per copy.
_HC_MIX = ("+", ("n_hidden_copies", "n_hidden_copies"), (2, "n_hidden_copies"))
#: One token's residual copies as ONE physical dim — the layout a
#: hyper-connection model that carries its stream flattened uses
#: (`[tokens, n_hidden_copies·d_model]`), and the reduction width of every
#: projection over the copies.
_HC_PACKED = ("n_hidden_copies", "d_model")
#: The multi-copy residual's axes, canonical (tokens folded) and as the
#: engine emits them. Named once because five spans carry this same tensor
#: under four different operand names (`residual`, `x`): the axes are the
#: fact, the name is each span's own.
_HC_COPIES = (_HC_TOKENS, "n_hidden_copies", "d_model")
_HC_COPIES_ENGINE = ("B", "s_query", "n_hidden_copies", "d_model")
#: The residual the seam carries, canonical and as the engine emits it.
_HC_RESIDUAL = TensorLayout("residual", _HC_COPIES)
_HC_RESIDUAL_ENGINE = TensorLayout("residual", _HC_COPIES_ENGINE)
#: The pre-mix projection, `[mix, n_hidden_copies·d_model]`. fp32 in both
#: frameworks' runtime; the cell keys on the activation alone.
_HC_FN = TensorLayout("fn", (_HC_MIX, _HC_PACKED), role="weight")

#: The post-mix's operands: the `post` gate, the finished sublayer output,
#: the cross-mix matrix and the residual, in the order the plan's
#: `io_mapping` lists them. The gate and the sublayer output are named on
#: their own because `ihc_post` — the same spread without a cross-mix —
#: takes those two tensors and this residual, and a second declaration of
#: one tensor is the drift this file exists to avoid.
_HC_POST_GATE = TensorLayout("post", (_HC_TOKENS, "n_hidden_copies"))
_HC_POST_GATE_ENGINE = TensorLayout("post",
                                    ("B", "s_query", "n_hidden_copies"))
_HC_SUBLAYER = TensorLayout("sublayer", (_HC_TOKENS, "d_model"))
_HC_SUBLAYER_ENGINE = TensorLayout("sublayer", ("B", "s_query", "d_model"))
_HC_COMB = TensorLayout(
    "comb", (_HC_TOKENS, "n_hidden_copies", "n_hidden_copies"))
_HC_COMB_ENGINE = TensorLayout(
    "comb", ("B", "s_query", "n_hidden_copies", "n_hidden_copies"))
_HC_POST_OPERANDS = (_HC_POST_GATE, _HC_SUBLAYER, _HC_COMB, _HC_RESIDUAL)
_HC_POST_ENGINE = (
    ("post", _HC_POST_GATE_ENGINE),
    ("sublayer", _HC_SUBLAYER_ENGINE),
    ("comb", _HC_COMB_ENGINE),
    ("residual", _HC_RESIDUAL_ENGINE),
)


register_signature(OpSignature(
    name="mhc_post",
    binds_ops=('mhc_post',),
    inputs=_HC_POST_OPERANDS,
    engine_inputs=_HC_POST_ENGINE,
    dtype_slots=("activation",),
    doc=("Hyper-connection POST-mix of a finished sublayer, one launch: "
         "`y[j,h] = Σᵢ comb[i,j]·residual[i,h] + post[j]·sublayer[h]`, the "
         "sublayer's output spread back over the copies and the copies "
         "cross-mixed. vLLM `mhc_post_tilelang` / sglang `mhc_post`. "
         "Memory-bound: it reads and writes the whole multi-copy residual."),
))

register_signature(OpSignature(
    name="mhc_pre",
    binds_ops=('mhc_pre',),
    # The seam's opening, whole: the multi-copy residual, the mix
    # projection and the affine of the split. `pre` never leaves the
    # launch — it weights the copies on the way out — so it is no operand
    # and no port; `post` and `comb` are ports the post-mix reads.
    inputs=(_HC_RESIDUAL, _HC_FN,
            TensorLayout("scale", (3,), role="metadata"),
            TensorLayout("base", (_HC_MIX,), role="metadata")),
    # Named, not positional: the fusion's operands are its pattern's
    # ports, and a position means a different tensor the day the
    # pattern gains one (`op_signatures` refuses positions here).
    engine_inputs=(("residual", _HC_RESIDUAL_ENGINE),
                   ("mix_weight", _HC_FN)),
    # The copy count is an axis no tensor of the SPLIT carries — the mix
    # vector is one `n²+2n` row per token — and the Sinkhorn rounds are
    # work: the kernel alternates a row and a column normalisation over
    # the `n_hidden_copies²` block that many times. Measured on a 910C
    # die at T=1024, 20 rounds against 100 is 119.2 us against 164.0, so
    # two iteration counts are two measurements and the count is in the
    # identity. The deployment's value is the checkpoint's, read at the
    # fields the framework binding names
    # (`FrameworkBinding.hc_sinkhorn_iters_from`).
    structural_axes=("n_hidden_copies", "sinkhorn_repeat"),
    dtype_slots=("activation",),
    doc=("Hyper-connection PRE-mix, whole, as ONE launch: flatten the "
         "copies, RMS-normalise the flat stream unscaled, project it onto "
         "the mix vector, split that into the `pre` / `post` / `comb` "
         "coefficients with the Sinkhorn projection on the cross-mix, and "
         "reduce the copies under `pre`. Writes the sublayer's input plus "
         "the two coefficient sets its closing post-mix will read. "
         "vLLM-Ascend `torch.ops._C_ascend.npu_hc_pre_v2` (profiler "
         "`HcPre`). The sublayer RMSNorm after it is NOT inside: the "
         "plugin calls `npu_rms_norm` separately and the profiler shows "
         "its own kernel. The GPU frameworks cut this span differently "
         "— `mhc_pre_gemm` plus `mhc_pre_big_fuse`, with the split left "
         "outside as a billed node."),
))

register_signature(OpSignature(
    name="mhc_pre_gemm",
    # `mhc_pre_gemm_stream` is the same launch spelled on a stream that is
    # already flat: DeepSeek-V4.1 carries its copies across block boundaries
    # flattened, so its mix stats have no in-graph `view_merge` leaf
    # (`fusion_plans/mhc_pre_gemm_stream.yaml`). Its residual operand reaches
    # the key zip in that rank-3 form — the second layout of operand 0,
    # selected by rank as every multi-rank declaration is.
    binds_ops=('mhc_pre_gemm', 'mhc_pre_gemm_stream'),
    inputs=(_HC_RESIDUAL, _HC_FN),
    engine_inputs=(("residual", _HC_RESIDUAL_ENGINE),
                   ("residual", TensorLayout("residual",
                                             ("B", "s_query", _HC_PACKED))),
                   ("mix_weight", _HC_FN)),
    dtype_slots=("activation",),
    doc=("Hyper-connection PRE-mix projection, one launch: the flattened "
         "multi-copy residual against the mix projection, scaled by the "
         "residual's own RMS — `mixes = fn @ rms_unscaled(flatten(residual))`. "
         "The kernels write the unnormalised dots plus a per-token sum of "
         "squares and the following big fuse applies the rsqrt, which is the "
         "same function. vLLM `_hc_prenorm_gemm_outputs` (DeepGEMM "
         "`tf32_hc_prenorm_gemm` or its TileLang fallback) / sglang "
         "`mhc_pre_gemm_sqrsum_tilelang`."),
))

register_signature(OpSignature(
    name="mhc_pre_big_fuse",
    binds_ops=('mhc_pre_big_fuse',),
    # The epilogue's own operands: the projection's split-K partials and
    # their sum-of-squares partners, the scale/base of the mix split, the
    # residual it reduces, and the sublayer norm's gain. The `pre`
    # coefficients the plan's naive pattern takes as external are computed
    # INSIDE this launch, so they are an output here, not an input.
    inputs=(
        TensorLayout("mixes", (_HC_TOKENS, _HC_MIX)),
        TensorLayout("sqrsum", (_HC_TOKENS,)),
        TensorLayout("scale", (3,), role="metadata"),
        TensorLayout("base", (_HC_MIX,), role="metadata"),
        _HC_RESIDUAL,
        TensorLayout("norm_weight", ("d_model",), role="weight"),
    ),
    # Engine operand 1 is the residual; operand 0 is the `pre` vector the
    # launch derives for itself.
    engine_inputs=(("residual", _HC_RESIDUAL_ENGINE),),
    dtype_slots=("activation",),
    doc=("Hyper-connection PRE-mix epilogue, one launch: reduce the "
         "projection's split-K partials, apply the rsqrt, the two sigmoid "
         "gates and the Sinkhorn normalisation of the cross-mix, reduce the "
         "copies by the `pre` gate and apply the sublayer RMSNorm — writing "
         "`post`, `comb` and the normalised sublayer input. vLLM "
         "`_MHC_PRE_BIG_FUSE_TILELANG_KERNEL` / sglang "
         "`mhc_pre_big_fuse_with_norm_tilelang`."),
))

register_signature(OpSignature(
    name="mhc_fused_post_pre",
    binds_ops=('mhc_fused_post_pre',),
    inputs=(*_HC_POST_OPERANDS, _HC_FN),
    engine_inputs=(*_HC_POST_ENGINE, ("mix_weight", _HC_FN)),
    dtype_slots=("activation",),
    doc=("A sublayer boundary's post-mix and the next pre-mix projection in "
         "ONE launch — the frameworks' small-token hyper-connection path. "
         "The new residual never leaves registers between the two, which is "
         "the saving; it is still written, because the sublayer's own "
         "post-mix and the next reduce both read it. vLLM "
         "`_MHC_FUSED_TILELANG_KERNEL` / sglang "
         "`mhc_fused_post_pre_fma_tilelang`."),
))

register_signature(OpSignature(
    name="mhc_post_combine_norm",
    binds_ops=('mhc_post_combine_norm',),
    # The post's four operands, the CARRIED `pre` the next sublayer collapses
    # the copies with, and that sublayer's norm gain. No projection: this
    # launch does not compute the mix vector (sglang's stats launch does,
    # `mhc_pre_gemm_stream`), which is what separates it from
    # `mhc_mega_post_pre`.
    inputs=(
        *_HC_POST_OPERANDS,
        TensorLayout("pre", (_HC_TOKENS, "n_hidden_copies"),
                     role="metadata"),
        TensorLayout("norm_weight", ("d_model",), role="weight"),
    ),
    engine_inputs=_HC_POST_ENGINE,
    dtype_slots=("activation",),
    doc=("DeepSeek-V4.1's shifted hyper-connection seam as ONE launch on "
         "Blackwell: the finished sublayer's post-mix, the next sublayer's "
         "collapse of the copies by the carried `pre`, and that sublayer's "
         "RMSNorm. It writes the new multi-copy residual (the next post and "
         "the stats launch both read it) and the normalised sublayer input. "
         "sglang `mhc_post_combine_norm_prefill`."),
))

register_signature(OpSignature(
    name="mhc_mega_post_pre",
    binds_ops=('mhc_mega_post_pre',),
    inputs=(
        *_HC_POST_OPERANDS,
        _HC_FN,
        TensorLayout("pre", (_HC_TOKENS, "n_hidden_copies"),
                     role="metadata"),
        TensorLayout("scale", (3,), role="metadata"),
        TensorLayout("base", (_HC_MIX,), role="metadata"),
        TensorLayout("norm_weight", ("d_model",), role="weight"),
    ),
    engine_inputs=(*_HC_POST_ENGINE, ("mix_weight", _HC_FN)),
    dtype_slots=("activation",),
    doc=("DeepSeek-V4.1's shifted hyper-connection seam in DeepInfer's one "
         "Mega-MHC launch: the previous sublayer's post-mix, the current "
         "coefficient projection, and the carried-pre reduce + sublayer "
         "RMSNorm. It reads the previous `pre` and writes the new residual, "
         "normalised input, mix vector, and next coefficient set; the "
         "coefficient split stays a billed graph node as in the other mHC "
         "spans. deep_gemm.mega_mhc, from DeepInfer "
         "HCLayerMixin._mega_mhc."),
))


# ---------------------------------------------------------------------------
# The other hyper-connection boundaries: the sublayer spread WITHOUT a
# cross-mix (`ihc_post`), the exit-side reduce of the copies to one hidden
# state (`fused_hc_head`), and the three launches of the read/write gates a
# flattened-stream model runs (`hc_silu`, `hc_gate_mix`, `hc_combine_norm`).
#
# Same residual, same axes, and that is the point: every cell here is
# `{B, s_query, n_hidden_copies, d_model}` (the mix activation adds its own
# low-rank width), so the layouts above are reused rather than restated.
#
# Two layouts of the residual, not one, because the models differ in how
# they physically hold it and the kernels take what their model holds:
# DeepSeek-V4 and HY V4 pass `[tokens, copies, width]` (`_HC_COPIES`), while
# Qwen4Exp's hyper-connection carries `[..., copies·width]` throughout
# (`vllm/models/qwen4_exp/nvidia/hyperconnection.py:11-12`) and its kernels
# assert the flat inner stride. The ENGINE emits the split form in both
# cases; `_HC_PACKED` is the canonical (= launch) layout, and the conversion
# between them is a pack, which the declaration already expresses.
# ---------------------------------------------------------------------------

#: One token's residual with the copies flattened, as Qwen4Exp holds it.
_HC_PACKED_STREAM = (_HC_TOKENS, _HC_PACKED)
#: The walk emits the flattened stream two ways, and WHICH ONE depends on
#: whether the graph had reason to split the copies at that node: the read's
#: two operands arrive as `[B, s, copies·width]` (the gate comes straight
#: off a projection, the normed stream off an elementwise `mul`), while the
#: write's residual arrives as `[B, s, copies, width]` (its `add` is per
#: copy). Both are the same canonical tensor; the rank tells them apart,
#: which is exactly how `zip_engine_layouts` selects a layout. Getting this
#: wrong does not misbind — it fails to bind, and every cell of the span
#: drops with "no shape projection".
_HC_PACKED_STREAM_ENGINE = ("B", "s_query", _HC_PACKED)

register_signature(OpSignature(
    name="ihc_post",
    binds_ops=('ihc_post',),
    # The same `post` gate, sublayer output and residual the cross-mixing
    # post-mix takes, minus the cross-mix: this launch spreads and adds, it
    # does not mix the copies with each other.
    inputs=(_HC_POST_GATE, _HC_SUBLAYER, _HC_RESIDUAL),
    engine_inputs=(("post", _HC_POST_GATE_ENGINE), ("y", _HC_SUBLAYER_ENGINE),
                   ("residual", _HC_RESIDUAL_ENGINE)),
    dtype_slots=("activation",),
    doc=("INDEPENDENT hyper-connection post-mix, one launch: "
         "`out[c] = post[c]·sublayer + residual[c]` — a finished sublayer's "
         "output scattered back onto each residual copy by that copy's own "
         "gate, with no cross-mix between copies and no Sinkhorn. Reads the "
         "whole multi-copy residual and writes it again, so it is "
         "memory-bound at `2·copies·width` bytes per token. vLLM "
         "`triton_ihc_post` / sglang `fused_hy4_ihc_post`."),
))

#: The exit reduce's projection, `[n_hidden_copies, n_hidden_copies·d_model]`
#: — one row per copy, contracting the flattened stream. fp32 in both
#: frameworks, which assert it.
_HC_HEAD_FN = TensorLayout("fn", ("n_hidden_copies", _HC_PACKED),
                          role="weight")

register_signature(OpSignature(
    name="fused_hc_head",
    binds_ops=('fused_hc_head',),
    inputs=(TensorLayout("x", _HC_COPIES),
            _HC_HEAD_FN,
            # The affine of the gate, two parameters of the math rather than
            # anything the launch streams: a scalar and one bias per copy.
            TensorLayout("scale", (1,), role="metadata"),
            TensorLayout("base", ("n_hidden_copies",), role="metadata")),
    engine_inputs=(("x", TensorLayout("x", _HC_COPIES_ENGINE)),
                   ("mix_weight", _HC_HEAD_FN)),
    dtype_slots=("activation",),
    doc=("Hyper-connection EXIT reduce, one launch: the multi-copy residual "
         "flattened and RMS-normalised, projected onto `n_hidden_copies` "
         "dots, turned into a per-copy sigmoid gate, and the copies summed "
         "under it — `y = Σ_c sigmoid(mixes_c·scale + base_c)·x_c`. Neither "
         "the dots nor the gate reaches HBM; the launch reads the residual "
         "twice (the norm and the reduce) and writes one hidden state. vLLM "
         "`hc_head_fused_kernel_tilelang` / sglang `fused_hc_head`."),
))

register_signature(OpSignature(
    name="hc_silu",
    binds_ops=('hc_silu',),
    # `d_hc_mix` is the low-rank width of the mix projection, which is what
    # this activation runs on — NOT the residual, which it never touches.
    inputs=(TensorLayout("x", (_HC_TOKENS, "d_hc_mix")),),
    engine_inputs=((0, TensorLayout("x", ("B", "s_query", "d_hc_mix"))),),
    dtype_slots=("activation",),
    doc=("The hyper-connection mix activation, one launch: "
         "`y = silu(x / n_hidden_copies)` on the low-rank vector between the "
         "down and up projections of the read gate. The copy count is a "
         "divisor compiled into the kernel, not an axis — it changes no byte "
         "of this launch (see `kernel_bench.refs`, which states the one "
         "value the oracle and every provider run it at). vLLM `hc_silu`."),
))

register_signature(OpSignature(
    name="hc_gate_mix",
    binds_ops=('hc_gate_mix',),
    # Both tensors have the same shape, so nothing but the port each fills
    # (`gate`, `normed_x` in the plan) tells them apart; the contract lists
    # the gate first, and vLLM takes the stream first.
    inputs=(TensorLayout("gate", _HC_PACKED_STREAM),
            TensorLayout("xn", _HC_PACKED_STREAM)),
    engine_inputs=(("gate", TensorLayout("gate", _HC_PACKED_STREAM_ENGINE)),
                   ("normed_x", TensorLayout("xn", _HC_PACKED_STREAM_ENGINE))),
    dtype_slots=("activation",),
    doc=("The hyper-connection READ, one launch: "
         "`y = (1/hc)·Σ_c sigmoid(gate_c)·xn_c` — the normed stream gated "
         "per copy and averaged over the copies, which is the single hidden "
         "state a sublayer then runs on. Reads two full multi-copy streams "
         "and writes one `d_model` row. vLLM `hc_gate_mix`."),
))

register_signature(OpSignature(
    name="hc_combine_norm",
    binds_ops=('hc_combine_norm',),
    inputs=(TensorLayout("residual", _HC_PACKED_STREAM),
            TensorLayout("block", (_HC_TOKENS, "d_model")),
            TensorLayout("injection", (_HC_TOKENS, "n_hidden_copies")),
            TensorLayout("norm_weight", (_HC_PACKED,), role="weight")),
    # Each operand at its port: the sublayer's output (`block`), the
    # injection logits, the residual (copies on their own axis there) and
    # the gain (`weight`).
    engine_inputs=(("block", TensorLayout("block", ("B", "s_query", "d_model"))),
                   ("injection", TensorLayout("injection",
                                    ("B", "s_query", "n_hidden_copies"))),
                   ("residual", TensorLayout("residual", _HC_COPIES_ENGINE)),
                   ("weight", TensorLayout("norm_weight", (_HC_PACKED,),
                                    role="weight"))),
    dtype_slots=("activation",),
    doc=("The hyper-connection WRITE fused with the next read's grouped "
         "norm, one launch: the sublayer output added onto every residual "
         "copy under `2·sigmoid(injection/hc)`, then each copy "
         "RMS-normalised over its own `d_model` and scaled by `(1 + w)`. "
         "Two writes — the combined stream, because the next write reads it, "
         "and the normalised one. vLLM `hc_combine_norm`."),
))
