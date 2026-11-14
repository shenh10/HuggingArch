"""Primitive op registrations — the OpSpec table itself.

Data, not logic: every `register_op(...)` call is one row of the op
library. `ops.py` imports this module at its bottom to trigger the
registrations; nothing should import it directly.
"""
from __future__ import annotations

from backend.arch.yaml_load import load_yaml
import dataclasses
from typing import Any, Dict, Optional, Sequence

from sympy import Expr, Integer, Rational

from backend.arch.eval.dim import log2 as _log2, value_of

from .dims import overlay_pending
# A proposal patch runs in this module's namespace and declares a recurrent
# op's axis use with it.
from .ops_base import AxisUseDecl as AxisUseDecl
from .ops_base import (Lookup, OP_REGISTRY, OpSpec, ProcessorStep, ShapeKind, TensorDecl, _CANONICAL_DIMS,
                  _dim, _flag, _pairs, _s_kv, _s_pairs, _s_query, _selected_pairs, mask_flags,
                  register_op,
                  resolve_shape_template)
from .promote_op_proposal import read_artifacts

# ---------------------------------------------------------------------------
# Primitive FLOPs / Params formulas
# ---------------------------------------------------------------------------

def _zero(d: Dict[str, Any]) -> Expr:
    return Integer(0)


register_op(OpSpec("linear",
    tp_axis_placement=True,
    label="Linear",
    gemm=True,
    # `in` is the axis the matmul sums over; `out` and every batch axis are
    # separable. A sequence shard passes straight through — a linear is
    # per-token — which is why context parallelism owes nothing here.
    couples_on=("in",),
    # Bias term multiplied by the bias flag (0/1 in numeric mode, Symbol
    # in symbolic mode) — avoids Python truthy on Expr values.
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["in"] * d["out"] + d["B"] * d["s_ctx"] * d["out"] * _flag(d, "bias"),
    params_fn=lambda d: d["in"] * d["out"] + d["out"] * _flag(d, "bias"),
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx", "out"),
    shape_kind="axis_replace", out_key="out", axis_locked=-1,
    weight_shape=("out", "in"),
    # Two tensors, and `params_fn` above is their sum. The bias follows the
    # OUTPUT axis, which is a fact about this op and not a positional rule —
    # `conv1d` biases the first axis of
    # `(channels_out, channels_in, kernel_width)` and `headwise_linear`
    # the first two of `(heads, out, in)`.
    tensors=(
        TensorDecl(".weight", ("out", "in")),
        TensorDecl(".bias", ("out",), when="bias", kind="vector", axis_refs=(0,)),
    ),
    required_dims=("in", "out"),
    # TP sharding modes for Y = XW + b:
    #   column_parallel — weight split along output (col) dim. Input X
    #     is replicated. Output Y is sharded along the output feature
    #     dim. No internal comm; output naturally feeds a row-parallel
    #     consumer or an elementwise op that respects the sharding.
    #   row_parallel — weight split along input (row) dim. Input X
    #     MUST arrive sharded along its last (input feature) dim;
    #     output is a PartialSum that requires inter-op all_reduce
    #     (or reduce_scatter in SP mode) before any non-pointwise
    #     downstream consumption.
    #   replicated — no split; every rank holds full W. Used for tiny
    #     projections (e.g. router gate, MoE shared-expert when the
    #     model author chose to replicate it).

))

# `nn.Parameter`-stored variant of `linear`. Used when the model code
# declares the weight as a naked `self.W = nn.Parameter(torch.empty(out, in))`
# and calls `F.linear(x, self.W)` instead of wrapping it in `nn.Linear`.
# Compute-wise identical to `linear` (same GEMM); the only difference is
# storage convention — the checkpoint name is the bare attribute (no
# `.weight` suffix), so `weight_tensor_suffix=""`. No bias support
# (nn.Parameter has no implicit bias slot — if the model wants one it
# declares another nn.Parameter and uses an explicit `add` in the spec).
#
# Triggered first by V4-Pro Hyper-Connections (`hc_attn_fn`,
# `hc_ffn_fn`, `hc_head_fn`) but applies to any model with this
# storage idiom.
register_op(OpSpec("linear_param",
    gemm=True,
    couples_on=("in",),
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["in"] * d["out"],
    params_fn=lambda d: d["in"] * d["out"],
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx", "out"),
    shape_kind="axis_replace", out_key="out", axis_locked=-1,
    formula="y = W · x",
    description="Linear with nn.Parameter weight (bare checkpoint name, no `.weight`). No bias.",
    weight_shape=("out", "in"),
    weight_tensor_suffix="",
    required_dims=("in", "out"),
))


# Linear with EXTERNAL weight: the weight tensor is owned by an upstream
# `parameter_tensor` node and passed in as the second input, mirroring
# torch's `F.linear(x, w)` functional form. Params=0 (weight isn't this
# op's responsibility). Used to express V4-Pro's
# `mixes = F.linear(x, self.hc_attn_fn)` where the weight lives at
# Block scope as a separate nn.Parameter, not inside an nn.Linear.
register_op(OpSpec("linear_external_weight",
    gemm=True,
    couples_on=("in",),
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["in"] * d["out"],
    params_fn=_zero,
    shape_in=(("B", "s_ctx", "in"), ("out", "in")),
    shape_out=("B", "s_ctx", "out"),
    shape_kind="axis_replace", out_key="out", axis_locked=-1,
    formula="y = W · x  (W passed as input — F.linear(x, W) idiom)",
    description="Linear consuming an externally-owned weight tensor (passed as 2nd input). Params=0; the upstream parameter_tensor owns the weight. Mirrors functional `F.linear(x, w)` where `w` is an explicit argument rather than an nn.Linear's internal weight.",
    required_dims=("in", "out"),
))

# Per-head dense linear: each of n_heads heads carries its own (in, out) weight
# matrix. Params and FLOPs scale by n_heads. Used for MLA absorbed-mode
# absorptions (W_uk and W_uv folded per-head into the Q-side / O-side).
# Runs over the fresh-token axis `S` like a standard `linear`; head-axis
# is the third dimension (post-head_split + transpose convention).
register_op(OpSpec("per_head_linear",
    tp_axis_placement=True,
    gemm=True,
    couples_on=("in",),
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["n_heads"] * d["in"] * d["out"]
                       + d["B"] * d["s_ctx"] * d["n_heads"] * d["out"] * _flag(d, "bias"),
    params_fn=lambda d: d["n_heads"] * d["in"] * d["out"]
                       + d["n_heads"] * d["out"] * _flag(d, "bias"),
    shape_in=("B", "n_heads", "s_ctx", "in"),
    shape_out=("B", "n_heads", "s_ctx", "out"),
    shape_kind="axis_replace", out_key="out", axis_locked=-1,
    formula="y[b,h,s,o] = Σᵢ W[h,o,i] · x[b,h,s,i]  (per-head weights)",
    description="Per-head dense linear (n_heads independent weight matrices). FLOPs and params scale by n_heads.",
    weight_shape=("n_heads", "out", "in"),
    # Two tensors, as for `linear`; `params_fn` above is their sum. The bias
    # follows the (head, out) axes. This declaration was missing while
    # `params_fn` already counted the bias (M4), which is exactly the
    # disagreement `bias_decl()` now makes visible.
    tensors=(
        TensorDecl(".weight", ("n_heads", "out", "in")),
        TensorDecl(".bias", ("n_heads", "out"), when="bias", kind="vector",
                   axis_refs=(0, 1)),
    ),
    required_dims=("n_heads", "in", "out"),
))

register_op(OpSpec("embedding",
    label="Embedding",
    lookup=Lookup(),
    # A lookup is a one-hot contraction over the table: a rank holding a slice
    # of the vocabulary can answer only for the ids that land in it and must
    # emit zeros for the rest — vocab-parallel embedding's partial sum.
    couples_on=("n_vocab",),
    flops_fn=_zero,
    params_fn=lambda d: d["n_vocab"] * d["d_model"],
    # Reads one tensor of token ids — a model input (`model_inputs:`) or a
    # formal its component binds to one — and appends the table's row width:
    # `B` and `s_ctx` are the ids' own two axes.
    shape_in=("B", "s_ctx"),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="explicit",
    weight_shape=("n_vocab", "d_model"),
    # The `(B, S)` input is token IDs — integers, not activations. int32 per
    # vLLM v1's input_ids buffer; see the OpSpec field for the sglang caveat.
    input_dtype="I32",
    required_dims=("d_model", "n_vocab"),
))

# Hash embedding (Svenstrup et al. 2017, "Hash Embeddings"): each token's
# context is hashed into `rows` ids, each one a row of ONE shared table, and the
# rows are gathered. Qwen3.8-Flash-Next's per-layer n-gram embedding
# (modeling_qwen4_exp.py:1080-1180): 16 heads (8 bigram, 8 trigram) hash the
# last 1-2 token ids, add the head's offset, and one `nn.Embedding(320001536,
# 160)` gathers 16 rows per token, flattened to 2560. LongCat-Flash-Lite's
# n-gram embedding is the same lookup (vLLM longcat_flash_ngram.py:85-87,
# 160-165). The hashing is integer arithmetic — no floating-point work — and
# what the caller does with the rows (flatten, project, average) is the
# caller's graph. A lookup (`Lookup`): a step reads `rows` table rows
# per token, under the balanced-lookup convention when the rows are split.
register_op(OpSpec("hash_embedding",
    label="Hash embedding",
    lookup=Lookup(),
    # A row cut of the table answers only the ids that land in it, as for
    # `embedding`.
    couples_on=("n_vocab",),
    flops_fn=_zero,
    params_fn=lambda d: d["n_vocab"] * d["d_model"],
    # Reads the token ids, as `embedding` does, and appends the `rows`
    # lookups per token and the table's row width.
    shape_in=("B", "s_ctx"),
    shape_out=("B", "s_ctx", "rows", "d_model"),
    shape_kind="explicit",
    weight_shape=("n_vocab", "d_model"),
    input_dtype="I32",
    formula="y[b, s, r, :] = table[hash_r(ids[b, ≤s]) , :]   (r < rows; one shared table)",
    description="Hashed multi-row lookup: `rows` ids per token, hashed from the "
                "token's context, gathered from one table "
                "(modeling_qwen4_exp.py:1163-1180). Returned per row; the caller "
                "flattens, projects or averages them.",
    required_dims=("d_model", "n_vocab", "rows"),
))

# The format change a quantised matmul needs on its operand, as the launch
# it is. A framework runs it as its own kernel — vLLM's `QuantFP8` /
# `per_token_group_quant_fp8`, sglang's `fp8_utils.py:1355` — unless a
# neighbour absorbs it, and then it is a FUSION of this op with that
# neighbour (`silu_and_mul` writing fp8, a W8A8 GEMM reading it), which is
# the mechanism that already exists for everything else a framework folds
# together. Before this op there was no node here at all: the cost model
# carried the change as a number on the CONSUMER (`Traffic.convert_bytes`)
# and six plans declared an output contract (`quantises_output`, since
# retired) to say a producer had absorbed it, so the one launch the hardware runs was invisible to everything that
# reads nodes — the inspector, the computation graph, and the measured
# corpus, where the GEMM's row was billed for both launches and 6,196 rows
# came out above the card's bandwidth.
#
# What it writes is NOT a property of the op: the target format is the
# deployment's (`ResolvedPrecision.required_format`), so the node carries
# it (`output_dtype`) exactly as a fused launch does. The op only says the
# shape is unchanged and the arithmetic is two passes over the elements.
register_op(OpSpec("quantise_activation",
    # No `label`: `op_labels()` is mirrored by `rendering._consts.OP_LABEL`
    # and a test holds the two equal, so a label here means a name added to
    # that second table — and the op's own name already reads as what it is.
    # Pointwise per element, EXCEPT the scale: one absmax over each group of
    # `activation_granularity` elements along the row, which couples that
    # group. `dim` is the axis the groups lie along, so a tensor-parallel cut
    # of `dim` cuts groups whole wherever the degree divides the group count
    # — the same coupling a per-tile weight scale has, and the reason this is
    # not `couples_on=()`.
    couples_on=("dim",),
    # On the per-head path too: an absorbed-MLA projection and the DSA
    # indexer's query are cast per head (`[B, n_heads, s_ctx, d_head]`).
    alt_shape_in=(("B", "n_heads", "s_ctx", "dim"),),
    # One absmax pass and one multiply per element. Declared rather than
    # zeroed: it is what the kernel computes. The roofline still reads this
    # node as memory — two FLOPs against at least three bytes moved per
    # element is an intensity below 1 on every card in `gpu_specs.csv` — so
    # declaring it changes no bound while leaving the FLOPs tree true.
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["dim"],
    params_fn=_zero,
    shape_in=("B", "s_ctx", "dim"),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="preserve",
    required_dims=("dim",),
    formula="q = round(x / s),  s = amax(|x| over the group) / max(fmt)",
    description="Cast an activation to the format its consumer's kernel eats "
                "(or back): the narrow elements and their per-group or "
                "per-token scales. vLLM `QuantFP8` / "
                "`per_token_group_quant_fp8`, sglang "
                "`per_token_group_quant_fp8`; measured as the "
                "`quantise_activation` cell.",
))

register_op(OpSpec("rmsnorm",
    tp_axis_placement=True,
    label="RMSNorm",
    # The mean of squares is taken over `dim`, so every output element depends
    # on all of it. Heads and positions are untouched.
    couples_on=("dim",),
    # After a `head_split` the tensor is `[B, n_heads, s_ctx, d_head]`, and
    # every op on the per-head path sees that rank. A second signature rather
    # than a wildcard: `n_heads` is a label, and the label is what says axis 1
    # of the output is axis 1 of the input.
    alt_shape_in=(("B", "n_heads", "s_ctx", "dim"),),
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["dim"],
    params_fn=lambda d: d["dim"],
    shape_in=("B", "s_ctx", "dim"),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="preserve",
    formula="y = x · γ / sqrt(mean(x²) + ε)",
    description="Root-mean-square layer norm (elementwise, last-dim reduce). Llama, Qwen, DSv*, …",
    # Single learnable gain on disk: `<name>.weight` shape (dim,).
    weight_shape=("dim",),
    required_dims=("dim",),
    # Reduction is over the feature dim (last axis = `dim`).
    #   replicate — vanilla TP: input replicated on `dim`. 0 internal comm.
    #   shard_s_ctx   — Megatron-SP: input sharded on S. Norm is per-token
    #               pointwise, so S-sharding is free; cost lands at the
    #               SP↔TP boundary via layout AG/RS.
    # No native mode for sharding along `dim` (the reduction axis):
    # no shipping framework does this, so any policy that hands
    # rmsnorm a Sharded(dim) input forces an upstream all_gather.

))

# Weight-less RMS normalization. Same forward as `rmsnorm` minus the
# `· γ` step at the end — pure `x / sqrt(mean(x²) + ε)`. Used inside
# DeepSeek-V4 `Block.hc_pre`/`hc_head` to normalize the flattened
# `[B, s_ctx, n_hidden_copies*d_model]` activations before the HC mixing Linear, where the
# learnable gain is folded into the downstream `hc_*_fn` weight rather
# than carried as a separate norm parameter, and on every query head in
# DeepSeek-V4 attention (`q *= rsqrt(q.square().mean(-1) + eps)`). Registered as a
# **separate primitive** at the same abstraction level as `rmsnorm` /
# `layernorm`: forward behaviour differs, so it gets its own OpSpec
# rather than a `GraphNode.unscaled` flag.
register_op(OpSpec("rmsnorm_unscaled",
    couples_on=("dim",),
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["dim"],   # same elementwise cost
    params_fn=_zero,                                       # no learnable γ
    shape_in=("B", "s_ctx", "dim"),
    # The per-head path's rank, as for `rmsnorm`: gemma-4's v norm and
    # DeepSeek-V4's query-head norm.
    alt_shape_in=(("B", "n_heads", "s_ctx", "dim"),),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="preserve",
    formula="y = x / sqrt(mean(x²) + ε)  (no learnable gain)",
    description=(
        "RMS normalization without learnable scale. Identical to `rmsnorm` "
        "minus the trailing `· γ` multiply. No on-disk weight."
    ),
    # No `weight_shape` → spec_tree treats this as non-weight-bearing.
    required_dims=("dim",),
))

# L2 normalization: divide by the Euclidean norm of the last axis, not by
# its root-mean-square, so the result is `rmsnorm_unscaled` over
# `sqrt(dim)` — a different function, hence its own primitive. Gated
# DeltaNet and KDA apply it to every q and k head before the delta rule:
# `use_qk_l2norm_in_kernel=True` runs FLA's `l2norm_fwd`
# (`y = x / sqrt(Σx² + 1e-6)`, vLLM's vendored
# `flash_linear_attention/ops/l2norm.py`), and transformers' eager
# fallback runs the same `l2norm` (`modeling_qwen3_next.py`).
# Priced as `rmsnorm_unscaled` is: one pass to accumulate the squares,
# one to scale.
register_op(OpSpec("l2norm",
    couples_on=("dim",),
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["dim"],
    params_fn=_zero,
    shape_in=("B", "s_ctx", "dim"),
    # Linear attention normalizes after the head split, at the per-head
    # rank `[B, n_heads, s_ctx, d_head]`.
    alt_shape_in=(("B", "n_heads", "s_ctx", "dim"),),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="preserve",
    formula="y = x / sqrt(Σ x² + ε)  (no learnable gain)",
    description=(
        "L2 normalization over the last axis, no learnable gain. The q/k "
        "normalization of Gated DeltaNet and KDA (FLA `l2norm_fwd`)."
    ),
    required_dims=("dim",),
))

# The beta of `nn.LayerNorm`, present unless the node says `bias: "0"`
# (`nn.LayerNorm(bias=False)`, or a checkpoint that stores only the gain:
# MiMo-V2.6's vision merger `ln_q`).
_LAYERNORM_BIAS = TensorDecl(".bias", ("dim",), when="bias", when_unset=True,
                             kind="vector", axis_refs=(0,))

register_op(OpSpec("layernorm",
    tp_axis_placement=True,
    label="LayerNorm",
    couples_on=("dim",),
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["dim"],
    params_fn=lambda d: d["dim"] * (1 + _LAYERNORM_BIAS.flag(d)),
    shape_in=("B", "s_ctx", "dim"),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="preserve",
    formula="y = (x − μ) / sqrt(σ² + ε) · γ + β   (β only with a bias)",
    description="Standard layer norm with affine γ and, unless `bias: \"0\"`, β.",
    # Gain `<name>.weight` shape (dim,) — bias is a separate on-disk
    # tensor surfaced via `LeafTensor.has_bias` (set elsewhere because
    # has_bias is per-leaf metadata, not part of the weight shape).
    weight_shape=("dim",),
    # gamma and beta, both `(dim,)`. `params_fn` is this list summed — one
    # fact, stated once, projected twice.
    tensors=(
        TensorDecl(".weight", ("dim",)),
        _LAYERNORM_BIAS,
    ),
    required_dims=("dim",),
    # Same sharding rules as rmsnorm — reduction over feature dim;
    # only `replicate` and `shard_s_ctx` (Megatron-SP) are native.

))

# Dense attention primitives (MHA / GQA / MLA-decompressed family).
#
# FLOPs structure: 2 × s_pairs × d_head, where `s_pairs` is the live
# (query, key) pairs of the matrix the node scores, per batch row and head:
# the phase's dense causal count `s_query·s_past + s_query(s_query+1)/2`
# (`binding.STEP_LAWS`), or what a narrowing node rebinds it to. A
# bidirectional attention (`is_causal: "0"`) scores its whole
# `s_query × s_kv` matrix (`_pairs`). 2× is the explicit matmul coefficient.

register_op(OpSpec("attention_score",
    input_roles=("q", "k", "select"),
    label="Q @ K^T",
    attention_kernel="score",
    # `d_head` is the contraction. The two sequence axes are each separable
    # ALONE — a rank holding all of K can score its own queries — and not
    # separable together: cut both and a rank holds one diagonal block of the
    # score matrix. That is where ring attention starts, and it is not free.
    # The head axes are absent on purpose: Q's `n_heads` and K's `n_kv_heads`
    # are different symbols, and tensor parallelism cuts them in lockstep.
    couples_on=("s_query", "s_kv", "d_head"),
    # Rank is a contract here: a rank-3 query is a mis-wired edge.
    input_rank_is_exact=True,
    flops_fn=lambda d: 2 * d["B"] * d["n_heads"] * _pairs(d) * d["d_head"],
    # Dense attention: every query reads all s_kv cached positions.
    kv_positions_per_query_fn=_s_kv,
    # Dense storage: cache every position; no eviction.
    kv_storage_positions_per_query_fn=_s_kv,
    params_fn=_zero,
    shape_in=(("B", "n_heads", "s_query", "d_head"), ("B", "n_kv_heads", "s_kv", "d_head")),
    shape_out=("B", "n_heads", "s_query", "s_kv"),
    # Q · Kᵀ — specs pass K as [B, n_kv_heads, s_kv, d_head], the transpose is implicit
    # in the op semantics. Output prefix = Q's prefix; last dim = K's
    # sequence axis (s_kv). Doesn't fit the generic `contract` kind because
    # that expects K already transposed. Shape inference reads s_kv from
    # the dim dict so decode / spec / prefix-cached bindings (where s_kv ≠
    # upstream activation length) get the correct attention-matrix shape
    # without editing every attention component yaml.
    shape_kind="explicit",
    formula="A = Q · Kᵀ / sqrt(d_head)",
    description="Scaled dot-product attention scores (pre-softmax). FLOPs scale with s_pairs, the live (query, key) pairs.",
    required_dims=("d_head", "n_heads"),
    # Q · K^T's reduction axis is per-head d_head.
    #   shard_n_heads  — TP shards heads (orthogonal to reduction → free).
    #   shard_s_query  — Q/K sharded along sequence; runs via Ring Attention
    #               (KV blocks rotate around the ring). Cost formula
    #               not yet implemented (ModeNotYetCosted) — fail-fast
    #               on policies that select this until a future commit
    #               wires the ring schedule.
    #   replicate — fallback / no TP.

    # An optional third operand: the selection that gates the read. Most
    # callers wire two and are untouched — `input_kind` defaults to
    # `activation` only for positions this tuple does not name, so declaring
    # the third does not invent one.
    #
    # The edge carries no law: on `dsa_attention_score` the sparsity comes from
    # the `k:` dim, which is why the shared-indexer variant wires no mask at all
    # and is sparse anyway. It does carry a read — the index list is a tensor
    # the kernel loads, billed like Q and K. What the edge states is WHOSE
    # selection this is, and a score that reads a narrowed stream without naming
    # the node that narrowed it leaves that node's output going nowhere. DSv4
    # CSA is the caller: dense over a union whose compressed half the indexer
    # has already cut to `n_topk_keys`.
    input_kinds=("activation", "activation", "mask"),
))

register_op(OpSpec("attention_apply",
    input_roles=("probs", "v"),
    label="Attn @ V",
    attention_kernel="apply",
    # `s_kv` is the contraction against V; `s_query` for the same reason as in
    # `attention_score`.
    couples_on=("s_query", "s_kv"),
    # Rank is a contract here: a rank-3 query is a mis-wired edge.
    input_rank_is_exact=True,
    flops_fn=lambda d: 2 * d["B"] * d["n_heads"] * _pairs(d) * d["d_head"],
    kv_positions_per_query_fn=_s_kv,
    kv_storage_positions_per_query_fn=_s_kv,
    params_fn=_zero,
    shape_in=(("B", "n_heads", "s_query", "s_kv"), ("B", "n_kv_heads", "s_kv", "d_head")),
    shape_out=("B", "n_heads", "s_query", "d_head"),
    shape_kind="contract",
    formula="O = A · V",
    description="Attention-weighted value combine: softmax(QKᵀ) · V. FLOPs scale with s_pairs, the live (query, key) pairs.",
    required_dims=("d_head", "n_heads"),

))

register_op(OpSpec("softmax",
    label="Softmax",
    # Normalization is over `s_kv`: the axis survives the op, but every element
    # of it divides by a sum over all of it. Repairable by an online LSE merge,
    # which is a mode — not transparency.
    couples_on=("s_kv",),
    flops_fn=lambda d: 5 * d["B"] * d["n_heads"] * _pairs(d),
    params_fn=_zero,
    shape_in=("B", "n_heads", "s_query", "s_kv"),
    shape_out=("B", "n_heads", "s_query", "s_kv"),
    shape_kind="preserve",
    formula="p[i] = exp(x[i]) / Σⱼ exp(x[j])",
    description="Row-wise softmax over the last dim (attention scores → weights).",
    required_dims=("n_heads",),
    # Reduction is over the last (softmax) axis s_kv.
    #   shard_n_heads  — TP shards heads (orthogonal to reduction → free).
    #   shard_s_kv     — sharded along softmax axis. Op uses an online-merge
    #               algorithm (running (m, l) state, the FlashAttention /
    #               Ring-Attention LSE trick) so it can natively consume
    #               and emit s_kv-sharded data. internal_comm not yet
    #               costed.
    #   replicate — fallback / no TP.

))

# Dense analogue of `swa_softmax_with_sink`: GPT-OSS full-attention layers
# pass a per-head sink into FA (`s_aux`) even without a window. Same
# denominator trick, unwindowed s_kv. FLOPs = dense softmax + the sink
# term (exp + add + divide per query).
register_op(OpSpec("softmax_with_sink",
    sink=True,
    couples_on=("s_kv",),
    flops_fn=lambda d: (5 * d["B"] * d["n_heads"] * _pairs(d)
                       + 3 * d["B"] * d["n_heads"] * _s_query(d)),
    params_fn=_zero,
    shape_in=(("B", "n_heads", "s_query", "s_kv"), ("n_heads",)),
    shape_out=("B", "n_heads", "s_query", "s_kv"),
    shape_kind="explicit",
    formula="p[i] = exp(x[i]) / (Σⱼ exp(x[j]) + exp(sink))",
    description="Row-wise softmax with a per-head sink-token bias in the denominator. GPT-OSS full-attention layers; the SWA form is swa_softmax_with_sink.",
    required_dims=("n_heads",),
))

# Per-fn elementwise activations. One op per function so the graph
# carries the activation identity in the op name itself (no out-of-band
# `fn:` parameter), each can declare its own per-element FLOPs cost,
# and tooltip formulas are specific. All are `shape_kind=preserve`, so
# rank flows through unchanged. `alt_shape_in` names the axes of a
# further rank an op is actually applied at — declared per op, from its
# real use, because the labels are the axes' identities.
def _act_op(name: str, coeff: int, formula: str, description: str,
            alt_shape_in: tuple = (), config_names: tuple = (),
            consts: tuple = ()) -> None:
    register_op(OpSpec(name,
        flops_fn=lambda d, _c=coeff: _c * d["B"] * d["s_ctx"] * d["dim"],
        params_fn=_zero,
        shape_in=("B", "s_ctx", "dim"),
        alt_shape_in=alt_shape_in,
        shape_out=("B", "s_ctx", "dim"),
        shape_kind="preserve",
        formula=formula,
        description=description,
        required_dims=("dim",),
        is_activation=True,
        config_names=config_names,
        consts=consts,
        # A pointwise function of one element couples along nothing, and
        # saying so once here is what stops any of the ten from forgetting.
        couples_on=(),
        # Pointwise activation: any sharding axis is legal (no
        # reduction along any dim). Modes enumerate the common axes.

    ))

# The per-head frame too: RMSNormGated gates each head of a linear-attention
# output by silu(z) after z's head split (Qwen3-Next / Qwen3.5 Gated DeltaNet).
_act_op("silu",      4, "y = x · sigmoid(x)",                    "SiLU / Swish activation (Llama, Qwen).",
        alt_shape_in=(("B", "n_heads", "s_ctx", "dim"),), config_names=("silu", "swish"))
_act_op("gelu",      8, "y = 0.5·x·(1 + erf(x / sqrt(2)))",       "Gaussian Error Linear Unit, exact (erf) form.",
        config_names=("gelu", "gelu_python"))
# The tanh approximation of GELU is a different function, and the one the
# Gemma family ships (`hidden_activation: gelu_pytorch_tanh`; vLLM
# GeluAndMul(approximate="tanh")). Priced like `gelu`: the per-element cost
# convention does not tell the two approximations apart.
_act_op("gelu_tanh", 8, "y = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))",
        "GELU, tanh approximation (transformers gelu_pytorch_tanh / gelu_new / gelu_fast / gelu_accurate).",
        config_names=("gelu_pytorch_tanh", "gelu_python_tanh", "gelu_new", "gelu_fast",
                      "gelu_accurate"))
# DeepSeek-V4's lightning indexer applies ReLU to the score matrix
# `[B, n_heads, s_query, s_kv]` that `attention_score` produces.
_act_op("relu",      1, "y = max(0, x)",                          "Rectified linear unit.",
        alt_shape_in=(("B", "n_heads", "s_query", "s_kv"),), config_names=("relu",))
_act_op("relu_squared", 2, "y = max(0, x)²",                      "Squared-ReLU (NemotronH ACT2FN[\"relu2\"]): relu then square fused in ONE pass — never materialises the intermediate relu tensor (cf. the eager relu→mul(relu,relu) which round-trips it through HBM twice).",
        config_names=("relu2",))
# DeepSeek-V4's clamped SwiGLU bounds both pre-activations by one limit:
# the gate from above, the up branch on both sides (transformers
# DeepseekV4MLP.forward / DeepseekV4Experts._apply_gate; vLLM
# SiluAndMulWithClamp). One compare-select per element, priced like `relu`:
# the per-element cost convention does not tell a one-sided bound from a
# two-sided one. The bound is the `clamp_limit` the node's frame binds — the
# spec says which config field that is (`OpSpec.consts`).
_act_op("clamp",     1, "y = min(max(x, −clamp_limit), clamp_limit)   (a one-sided clamp drops the lower bound)",
        "Pointwise clamp to a config-time bound (`clamp_limit`; DeepSeek-V4 binds it to `swiglu_limit`).",
        consts=("clamp_limit",))
# Kimi-K3's lower-bounded KDA decay gate applies it per (head, key channel),
# on the head-split `[B, n_heads, s_ctx, d_head]` frame.
# A config-time constant added to every element: HY-V4's hyper-connection
# gates add `hc_eps` after their sigmoid (modeling_hy_v4.py:661, :663, :688).
# The additive twin of `scale_const`; no stored tensor, so not an `add`.
_act_op("add_const", 1, "y = x + c   (c a config-time constant, e.g. hc_eps)",
        "Element-wise addition of a config-time scalar (no learned param).")
_act_op("sigmoid",   4, "y = 1 / (1 + exp(−x))",                  "Logistic sigmoid (gating, routing scores).",
        alt_shape_in=(("B", "n_heads", "s_ctx", "dim"),), config_names=("sigmoid",))
_act_op("softplus",  5, "y = log(1 + exp(x))",                    "Smooth ReLU approximation.")
_act_op("sqrt",      1, "y = sqrt(x)",                            "Pointwise square root. DeepSeek-V4 `sqrtsoftplus` router scoring is softplus followed by sqrt.")
# Sign-preserving square root, as Qwen3.8-Flash-Next's PLE gate writes it
# (modeling_qwen4_exp.py:1247 `gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`):
# five elementwise tensor ops (abs, clamp_min, sqrt, sign, mul) on one element.
# `sqrt` loses the sign and there is no abs/sign primitive to compose it from.
_act_op("signed_sqrt", 5, "y = sign(x) · sqrt(max(|x|, 1e-6))", "Signed square root: sqrt of the magnitude (clamped at 1e-6) with the input's sign. Qwen3.8-Flash-Next PLE gate.")
_act_op("exp",       2, "y = exp(x)",                             "Pointwise exponential.")
_act_op("tanh",      5, "y = (e^x − e^−x) / (e^x + e^−x)",        "Hyperbolic tangent.", config_names=("tanh",))
_act_op("neg_exp",   3, "y = exp(−x)",                            "Pointwise exp(-x), used for gated-delta-net A_log.")
_act_op("situ",      11, "y = situ_beta·tanh(x/situ_beta)·sigmoid(x)", "Kimi-K3 `SituAndMul` gate nonlinearity (β = `situ_beta`): a bounded swish variant. Applied to the gate branch of a gated FFN, then ×up — SwiGLU-shaped, silu↔situ.",
        config_names=("situ",),
        # The gate's β, and the optional β the up branch is softly clipped
        # by in the same launch (`SituAndMul(beta, linear_beta)`). Kimi-K3's
        # spec binds them to `activation_situ_beta` / `_linear_beta` (4.0
        # and 25.0).
        consts=("situ_beta", "situ_linear_beta"))

# Elementwise softmax for routing / selection paths (router scores,
# gate_softmax). Distinct from the attention-side `softmax` op above
# which assumes 4-D `(B, n_heads, s_ctx, s_ctx)` shape — this one is shape-preserving
# under rank propagation, matching how routers and other 1-D softmax
# usages flow.
register_op(OpSpec("softmax_act",
    # `dim` is the NORMALIZED axis in every signature below: the last axis of
    # the rank-3 router form, the window axis of DSv4's rank-4 window form
    # `[B, slots, m, D]`. A call site's `axis:` must land on it — the layout
    # law derives the coupled axis from the matched signature and refuses a
    # node whose `axis:` says otherwise, so the two cannot drift apart.
    couples_on=("dim",),
    # The window form: slots, the normalized window (`dim`), and a feature
    # axis the softmax does not touch. Cost stays per element — the primary
    # template's trailing alignment folds the extra axis in as batch.
    alt_shape_in=(("B", "s_ctx", "dim", "d_head"),),
    flops_fn=lambda d: 5 * d["B"] * d["s_ctx"] * d["dim"],
    params_fn=_zero,
    # A router's scoring nonlinearity, so an activation — registered here
    # rather than through `_act_op` because it normalises over `dim`.
    is_activation=True,
    shape_in=("B", "s_ctx", "dim"),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="preserve",
    formula="p[i] = exp(x[i]) / Σⱼ exp(x[j])   (j runs over the normalized axis)",
    description="Softmax over the `dim` axis of its signature: the last axis "
                "of [B, S, dim] (router scores, gating), or the window axis of "
                "[B, slots, dim=m, D] (DSv4's compressor, declared `axis: 2`). "
                "Layout requires exactly that axis whole; a token or feature "
                "shard is orthogonal. Cost is 5 FLOPs per element.",

    required_dims=("dim",),
))

register_op(OpSpec("add",
    label="+",
    # Pointwise: nothing couples.
    couples_on=(),
    # Pointwise on the per-head path too, where both operands are
    # `[B, n_heads, s_ctx, dim]` — a residual add or a gate multiply inside
    # attention. Labelled rather than wildcarded so the two operands' axes
    # are known to be the SAME index, which is the question a pointwise op
    # exists to answer and the one a rank match cannot.
    alt_shape_in=(((("B", "n_heads", "s_ctx", "dim"),
                    ("B", "n_heads", "s_ctx", "dim"))),),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["dim"],
    params_fn=_zero,
    shape_in=(("B", "s_ctx", "dim"), ("B", "s_ctx", "dim")),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="broadcast",
    required_dims=("dim",),
    # An absent term adds nothing: `moe_out: add [combine, shared_experts]`
    # in a model with no shared expert is `combine`.
    absent_operand_identity="0",
    # Elementwise binary add — both inputs must arrive with the same
    # state; output preserves it. Linear in both operands, so a sum
    # of two partial sums is still a partial sum (lets fusion defer
    # the all_reduce past the residual add).

))

# A running sum over the copies of a repeated producer: `acc += x` once per
# copy, the accumulator being this op's own output. The source writes it as a
# loop of in-place adds into a zero tensor (MiMo-V2.5's speech-code banks,
# modeling_mimo_v2.py `_apply_speech_embeddings`: `out = torch.zeros(...)`;
# `for i in range(audio_channels): out.add_(speech_embeddings[i](codes[..., i]))`),
# so the node repeats with its producer, and copy c adds the producer's
# copy c. One add per element per copy.
register_op(OpSpec("accumulate",
    couples_on=(),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["dim"],
    params_fn=_zero,
    shape_in=("B", "s_ctx", "dim"),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="preserve",
    formula="acc += x   (one copy of x per repeat; acc starts at zero)",
    # `add_` loads the accumulator's current value before it stores the sum.
    reads_own_output=True,
    description="Running sum over the copies of a repeated producer: each "
                "repeat adds one copy into the accumulator.",
    required_dims=("dim",),
))

register_op(OpSpec("mul",
    label="×",
    couples_on=(),
    # Pointwise on the per-head path too, where both operands are
    # `[B, n_heads, s_ctx, dim]` — a residual add or a gate multiply inside
    # attention. Labelled rather than wildcarded so the two operands' axes
    # are known to be the SAME index, which is the question a pointwise op
    # exists to answer and the one a rank match cannot.
    alt_shape_in=(((("B", "n_heads", "s_ctx", "dim"),
                    ("B", "n_heads", "s_ctx", "dim"))),),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["dim"],
    params_fn=_zero,
    shape_in=(("B", "s_ctx", "dim"), ("B", "s_ctx", "dim")),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="broadcast",
    required_dims=("dim",),
    # An absent factor multiplies by one.
    absent_operand_identity="1",
    # Elementwise multiply — same legal sharding as add. A pending sum
    # passes through only when the other factor is the same on every rank
    # (mul is linear in each operand separately, not in both at once);
    # the fold law in `layout/foldlaw.py` states and checks that.

))

# --- Depth-wise attention residuals (Kimi-K3 AttnRes) ----------------------
# `modeling_kimi_linear.py::_forward_attn_residual` / `_apply_attn_res`. Every
# `attn_res_block_size` layers the stack appends its entry to a history
# (`block_residual`), and before each sub-layer the input is a softmax-weighted
# combination of that history and the running prefix sum. The history is the
# stack's `read: all` stream, so its length at each layer comes from the
# stream, not from a constant: these ops read and grow it.
def _history_contract(shapes: Sequence[Sequence[Any]],
                      dims: Dict[str, Any]) -> Optional[str]:
    """A history to append to holds at least one entry: an empty one is
    started with `stack_init`, not appended to."""
    history = shapes[0] if shapes else ()
    if len(history) == 4 and value_of(history[2]) == 0:
        return ("stack_append extends a history of at least one entry; "
                "start one with `stack_init`")
    return None


register_op(OpSpec("stack_init",
    # A copy into a new axis; nothing is summed along anything.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "d_model"),
    shape_out=("B", "s_ctx", "Sar", "d_model"),
    shape_kind="explicit",
    # `torch.cat` allocates: one entry read, one written.
    is_view_op=False,
    input_rank_is_exact=True,
    formula="H = x[:, :, None, :]   (a history of one entry)",
    description="Start a depth-wise history with one entry: "
                "`torch.cat([empty, x.unsqueeze(1)], dim=1)` in "
                "modeling_kimi_linear.py:995-997 at layer 0. A copy — reads "
                "x once, writes one entry.",
))

register_op(OpSpec("stack_append",
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=(("B", "s_ctx", "Sar", "d_model"), ("B", "s_ctx", "d_model")),
    shape_out=("B", "s_ctx", "Sar", "d_model"),
    shape_kind="explicit",
    # `torch.cat` allocates: the m entries and the new one are read, the
    # m + 1 entries written.
    is_view_op=False,
    input_rank_is_exact=True,
    input_contract=_history_contract,
    formula="H' = cat([H, x[:, :, None, :]], dim=2)   (m -> m + 1 entries)",
    description="Append an entry to a depth-wise history of m >= 1 "
                "entries along its history axis: `torch.cat([block_residual, "
                "prefix_sum.unsqueeze(1)], dim=1)` in "
                "modeling_kimi_linear.py:995-997. Not `concat` with an "
                "`axis:` — the op name is the gate. A copy: reads m + 1 "
                "entries, writes m + 1.",
))

register_op(OpSpec("attn_res_combine",
    # Softmax over the Sar+1 candidates, and the score that feeds it
    # contracts `d_model`.
    couples_on=("Sar", "d_model"),
    # RMS-normalise the Sar+1 candidates (the history and the prefix), score
    # each against `norm_w ⊙ proj_w`, softmax, weighted sum: ~3 passes over
    # (Sar+1)·d_model per token. It reads the whole history.
    flops_fn=lambda d: 3 * d["B"] * d["s_ctx"] * (d["Sar"] + 1) * d["d_model"],
    params_fn=_zero,
    shape_in=(("B", "s_ctx", "d_model"), ("B", "s_ctx", "Sar", "d_model"),
              ("d_model",), ("n_res_score", "d_model")),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="preserve",
    input_rank_is_exact=True,
    formula="softmax_i((rms(v_i) ⊙ norm_w ⊙ proj_w)·1) · v,  "
            "v = cat(history, prefix)",
    description="Kimi-K3 `_apply_attn_res` (modeling_kimi_linear.py:1075-"
                "1088): a softmax-weighted combination of the history and "
                "the running prefix. Inputs: prefix [B,S,D], history "
                "[B,S,m,D], and the scoring weights — the `*_res_norm` gain "
                "[D] and the `*_res_proj` row [1,D] — which only score; "
                "they gate nothing. Their tensors are the parameter nodes "
                "wired in; the op owns none.",
))

register_op(OpSpec("rope",
    label="RoPE",
    rotary_position_axes=1,
    # The rotation pairs elements WITHIN `dim` — the same fact the mode list
    # states by omitting `shard_dim`.
    couples_on=("dim",),
    # 3 per rotated element: `x' = x·cos − y·sin`, `y' = y·cos + x·sin` is
    # 4 multiplies and 2 adds per pair (vLLM csrc/libtorch_stable/
    # pos_encoding_kernels.cu:31-32; triton_mrope, ApplyRotaryEmb and sglang
    # rope.cuh the same). cos / sin are read from a cache: bytes.
    flops_fn=lambda d: (
        3 * d["B"] * d["n_heads"] * d["s_ctx"] * d["dim"]
    ),
    params_fn=_zero,
    shape_in=("B", "n_heads", "s_ctx", "dim"),
    shape_out=("B", "n_heads", "s_ctx", "dim"),
    shape_kind="preserve",
    formula="(x_even, x_odd) ← (x_even·cos − x_odd·sin, x_even·sin + x_odd·cos)",
    description=(
        "Rotary position embedding applied per-head over the position axis. "
        "Single-input variant for 4-D `(B, n_heads, s_ctx, dim)` tensors — call once per "
        "tensor (Q and K get their own node). For 3-D head-folded inputs (MLA "
        "partial rope) use `rope_emb` instead."
    ),
    # Rotation pairs elements WITHIN `dim`, so that axis may not be split —
    # and heads and positions are untouched by it, so those may. Each rank
    # rotates its own heads at its own positions with no peer. The absence of a
    # `shard_dim` entry is the load-bearing part of this list.

    required_dims=("dim",),
))

# Two-axis RoPE: a vision tower rotating each patch by its (h, w) grid
# position. The cos/sin row a token reads is built from two position streams
# — vLLM Qwen2.5-VL / Qwen3-VL / Qwen3.5 vision builds `head_dim/2` entries
# per position from the h and w tables concatenated (qwen3_vl.py
# `rot_pos_emb`), MiMo-V2 omni vision likewise (`get_rope(...,
# partial_rotary_factor=0.5)`) — and the rotation covers the whole head. Per
# rotated element the work is `rope`'s, so the geometry, layout law and FLOPs
# are `rope`'s own (derived from its declaration below, so they cannot
# drift). What is its own is the kernel: vLLM runs these through
# `ApplyRotaryEmb` (rotary_embedding/common.py; vllm-flash-attn Triton rotary
# on CUDA), not the text path's `_C.rotary_embedding`, so no q/k fusion plan
# matches it.
register_op(dataclasses.replace(OP_REGISTRY["rope"],
    name="rope_2d",
    label="2-D RoPE",
    rotary_position_axes=2,
    formula=("(x_i, x_j) ← rotation by θ_k · pos_a(k)   "
             "(channel half k reads position stream a ∈ {h, w})"),
    description=(
        "Two-axis rotary position embedding for a vision tower: each patch "
        "is rotated by its (h, w) grid position, half the frequencies from "
        "each axis, over the whole head. 4-D `(B, n_heads, s_ctx, dim)`, "
        "one node per tensor. Same per-element work as `rope`."),
))

# XD-RoPE (HunYuan-VL / HunyuanOCR text decoder). Positions are [4, T] — the
# P / W / H / T streams vLLM's `get_xdrope_input_positions` builds
# (models/hunyuan_vision.py:838-852, `len(xdrope_section)` streams) — and
# `xdrope_section` gives each stream one channel section of the cos/sin row
# (layers/rotary_embedding/xdrope.py:110-115: `cos.split(xdrope_section)`,
# section i from stream i). Per rotated
# element the work is `rope`'s, so the geometry, layout law and FLOPs are
# `rope`'s own (derived below), as for `rope_2d`. What is its own is the
# kernel: vLLM's HunYuan attention calls `get_rope(..., rope_parameters)`
# (models/hunyuan_v1.py:209-213, `q, k = self.rotary_emb(positions, q, k)`
# at :235), which builds `XDRotaryEmbedding` when the rope type is `xdrope`
# (rotary_embedding/__init__.py:231-241); its `forward_cuda` gathers the
# cos/sin rows of the four streams and calls `ApplyRotaryEmb` once on q and
# once on k (xdrope.py:89-138) — no q/k fusion plan covers it, and the
# text path's `_C.rotary_embedding` (`fused_qk_rope`) is not what runs.
# (Those are the native vLLM files of main a18c9b56f; at v0.32.0 vLLM serves
# HunYuan-VL through its Transformers backend, registry.py:735-738, and reads
# `xdrope_section` as `mrope_section`, transformers_utils/config.py:656-680.)
# sglang v0.5.22 has no HunYuan-VL implementation. HF transformers 5.17
# normalises the same config to `mrope_section` with `rope_type: xdrope`
# (configuration_hunyuan_vl.py:176-217); the type decides it, not the key.
register_op(dataclasses.replace(OP_REGISTRY["rope"],
    name="xdrope",
    label="XD-RoPE",
    rotary_position_axes=4,
    rotary_sections_key="xdrope_section",
    formula=("(x_i, x_j) ← rotation by θ_k · pos_a(k)   "
             "(channel section k reads position stream a ∈ {p, w, h, t})"),
    description=(
        "XD rotary position embedding (HunYuan-VL): four position streams "
        "(p, w, h, t), each `xdrope_section` channel section rotated by its "
        "own stream's positions, over the whole head. 4-D `(B, n_heads, "
        "s_ctx, dim)`, one node per tensor. Same per-element work as `rope`."),
))

# View ops — zero FLOPs, zero params, but full-fledged nodes in the graph.
# Matches the convention of ONNX / PyTorch FX / MLIR where slice / concat are
# first-class ops. Keep names operation-oriented (e.g. `q_nope_slice`, not `q_nope`).

register_op(OpSpec("slice",
    # Taking a window of an axis already cut across ranks is a re-partition:
    # which ranks hold the requested range depends on where the cut fell.
    couples_on=("in", "out"),
    # After a `head_split` the tensor is `[B, n_heads, s_ctx, d_head]`, and
    # every op on the per-head path sees that rank. A second signature rather
    # than a wildcard: `n_heads` is a label, and the label is what says axis 1
    # of the output is axis 1 of the input.
    alt_shape_in=(("B", "n_heads", "s_ctx", "in"),),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx", "out"),
    # axis from node.axis (default last). Replaces shape[axis] with dims['out'];
    # a node that says `start` instead keeps the axis from there on
    # (`hs[:, start:]`, the crop of a whole-stream join), see `_dm_slice`.
    shape_kind="axis_replace", out_key="out",
    description="View-op: take a contiguous slice along a dim (zero FLOPs / zero HBM).",
    required_dims=("out",),
    is_view_op=True,
))

# A constant built from another tensor's leading axes: zeros of width
# `out` with a 1 at one position. DeepSeek-V4.1's `make_identity_pre_mix(h,
# hc_mult)` (source/model.py:1159-1163) is this, the one-hot on hidden copy
# 0 that seeds the hyper-connection mix layer 0 reduces with. Only the
# shape of `in` is read (`reads_input_shape_only`); the output is written
# once.
register_op(OpSpec("one_hot",
    # No value of `in` is read, so nothing is summed or gathered along any
    # axis: the output's last axis is written whole on every rank.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx", "out"),
    shape_kind="axis_replace", out_key="out",
    reads_input_shape_only=True,
    formula="y = zeros(B, s_ctx, out); y[:, :, 0] = 1",
    description="A one-hot constant shaped by x's leading axes: zero FLOPs, "
                "zero params, one write of the output.",
    required_dims=("out",),
))

# Opaque local launch emitted by a bound serving-runtime plan. Its work is
# declared once by RuntimeStageWork and grafted into the evaluated layers;
# this primitive provides semantic flow only, never a second cost formula.
register_op(OpSpec("runtime_stage",
    couples_on=(), flops_fn=_zero, params_fn=_zero,
    shape_in=("B", "s_ctx", "in"), shape_out=("B", "s_ctx", "in"),
    shape_kind="preserve",
    description="Local runtime launch with source-declared work and traffic.",
))

register_op(OpSpec("identity",
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    alt_shape_in=(("B", "n_heads", "s_ctx", "in"),),
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx", "in"),
    shape_kind="preserve",
    description="View-op: pass the tensor through unchanged (zero FLOPs / zero params).",
    is_view_op=True,
))

# A one-step shift along a declared axis: `y[.., i, ..] = x[.., i - 1, ..]`,
# with position 0 filled by a constant. DeepSeek-V4's overlap compressor pairs
# each window with the PREVIOUS window's first half (transformers
# DeepseekV4CSACompressor: `new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :D]`,
# window 0 left zero-kv / -inf-gate). Priced as `slice` is: an offset read with
# no arithmetic and no buffer of its own; the reader that materializes the
# result (the compressor's `concat`) pays the copy.
register_op(OpSpec("shift",
    # Position i reads position i - 1 of the SAME axis: a cut of that axis
    # would leave each rank's first position on its neighbour.
    couples_on=(),
    # The compressor shifts its folded `[B, slots, window, D]` frame along
    # the slot axis — a sequence axis folded by the window, hence `s_ctx`.
    alt_shape_in=(("B", "s_ctx", "window", "in"),),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx", "in"),
    shape_kind="preserve",
    formula="y[.., i, ..] = x[.., i − 1, ..] for i ≥ 1; y[.., 0, ..] = fill   (fill a config-time constant)",
    description="View-op: shift the declared `axis:` by one position, filling position 0 with a constant (zero FLOPs / zero HBM; the reader materializes).",
    axis_required=True,
    is_view_op=True,
))

register_op(OpSpec("sample",
    # Serving selects last-N residual positions for the LM head. A view on
    # the residual, not a Binding rewrite: the `s_ctx` slot of this
    # template (axis 1) narrows to `s_sampled`.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx", "in"),
    shape_kind="axis_replace", out_key="s_sampled", axis_locked=1,
    description="View-op: select the sampled residual positions for the output head.",
    is_view_op=True,
))

register_op(OpSpec("concat",
    # It joins along `node.axis` (default the last) and is transparent to
    # every other one: the head shard MLA hands it survives the pack, and only
    # a split of the concatenated axis itself would not.
    couples_on=("in_a", "in_b"),
    # After a `head_split` the tensor is `[B, n_heads, s_ctx, d_head]`, and
    # every op on the per-head path sees that rank. A second signature rather
    # than a wildcard: `n_heads` is a label, and the label is what says axis 1
    # of the output is axis 1 of the input.
    alt_shape_in=(((("B", "n_heads", "s_ctx", "in_a"),
                    ("B", "n_heads", "s_ctx", "in_b"))),),
    flops_fn=_zero,
    params_fn=_zero,
    # 2-input concat is most common (MLA pack of nope+pe); shape templates are
    # illustrative — the engine doesn't strict-check view op shapes.
    shape_in=(("B", "s_ctx", "in_a"), ("B", "s_ctx", "in_b")),
    shape_out=("B", "s_ctx", "out"),
    # axis dim = sum across all inputs along node.axis (default last).
    shape_kind="axis_concat",
    passes_a_lone_operand=True,
    # Concatenation runs along the LAST axis (`node.axis` defaults there, and
    # the shape template names it `in_a` / `in_b`), so a shard on the SEQUENCE
    # is orthogonal to it and survives.
    #
    # What is deliberately not here is the mixed case, and it is a limit of the
    # catalog rather than an open question. MLA packs a head-sharded nope part
    # against a replicated decoupled RoPE key — one operand distributed, one
    # not — and a `ShardMode` names a state per input position, so covering it
    # means writing the cross product of every combination. That is a hole
    # punched per shape, not a declaration. What the `ShardMode` vocabulary
    # is missing is the sentence its docstring already speaks in prose: "an op
    # reducing along an axis OTHER than X is orthogonal to X, natively". Six
    # nodes on the corpus wait on it, and they stay in
    # `Placement.assumptions` until it exists.

    description="Concat along a dim — zero FLOPs but DOES copy HBM bytes "
                "(vLLM's `concat_and_cache_mla` is a real kernel, "
                "`torch.cat` always allocates). is_view_op=False.",
))

# Pure view-split / view-merge ops. Decompose the per-head reshape
# (B,s_ctx,n_heads·d_head) ↔ (B,n_heads,s_ctx,d_head) into two atomic steps:
#   head-split  ≡  view_split (axis -1 → (n_heads, d_head))  →  transpose [0,2,1,3]
#   head-merge  ≡  transpose [0,2,1,3]              →  view_merge (last-2 → out)
# The fused split_heads / merge_heads ops were removed in favor of these.
register_op(OpSpec("view_split",
    # `in` is the composite axis being split; a shard of it does not survive
    # the reshape as a shard of either half.
    couples_on=("in",),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=(),
    shape_out=(),
    shape_kind="explicit",
    description="View-op: split the declared `axis:` into two factor sub-axes "
                "(zero FLOPs / zero HBM), named by the node's dims in order "
                "(outer, inner) — `axis: -1` with (n_heads, d_head) for a head "
                "split. Exact product: the split axis's extent must equal "
                "outer · inner — drop a ragged tail with an explicit upstream "
                "`slice` (as DSv4's compressor slices to `(s_ctx // m) · m` "
                "complete windows) rather than rounding here.",
    required_dims=("n_heads", "d_head"),
    axis_required=True,
    is_view_op=True,
))

register_op(OpSpec("spatial_merge",
    # Patches fold m x m at a time, so the rows regroup into the features: a
    # shard of the rows cuts windows in half. The two axes are POSITIONS of
    # the arriving tensor (`axis_propagation.SPATIAL_MERGE_AXES`), the rows
    # being an encoder's own patches, never a label of the text step.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=(),
    shape_out=(),
    shape_kind="explicit",
    # Pixel-unshuffle over a FLATTENED patch sequence: an m x m spatial
    # window's patches fold into channels, so tokens shrink by m^2 and the
    # channel axis grows by m^2 — [.., S, C] -> [.., S // m^2, C * m^2].
    # Element-conserving by construction, which is why a projector modeled
    # without it still priced correctly; what was missing was the GRAPH
    # saying what the forward does. The sequence axis regroup is not
    # expressible with `view_split` / `view_merge` (both reshape only the
    # trailing axis), so it is its own declared rule rather than a lambda
    # at a call site.
    description="View-op: m x m spatial patch merge on a flattened sequence — "
                "[.., S, C] -> [.., S/m^2, C*m^2] (zero FLOPs / zero HBM).",
    required_dims=("n_spatial_merge",),
    is_view_op=True,
))

# --- Media into rows -------------------------------------------------------
# What a multimodal processor hands an encoder, as the encoder's own rows. An
# image input is `[N, n, H, W]` — the request batch (or the items of another
# input), the items per request, and one item's processed height and width;
# the processor cuts each item into a grid of patches, each one row of
# `channels · temporal · patch²` values, and an item is one attention segment
# (`cu_seqlens`). These are views of what the processor already produced:
# zero FLOPs, no device bytes of their own.


def _extent(value: Any) -> Optional[int]:
    """`value` as a number, or None where the domain is symbolic."""
    number = value_of(value)
    if isinstance(number, Integer):
        return int(number)
    return number if isinstance(number, int) else None


def _patchify_contract(shapes: Sequence[Sequence[Any]],
                       dims: Dict[str, Any]) -> Optional[str]:
    """`[N, n, H, W]`, each side a multiple of the processor's resize
    factor `align` (which every legal side is, so the patch grid is whole)."""
    if len(shapes) != 1 or len(shapes[0]) not in (4, 5):
        return (f"patchify reads one image input [N, n, H, W] or one video "
                f"input [N, n, T, H, W], got ranks "
                f"{[len(shape) for shape in shapes]}")
    align = _extent(dims["align"])
    if align is None:
        return None  # a symbolic factor decides nothing
    if align <= 0:
        return "patchify declares no positive `align`"
    if len(shapes[0]) == 5:
        frames = _extent(shapes[0][2])
        if frames is not None and frames < 1 and _extent(shapes[0][1]):
            return f"a video holds at least one frame, got {frames}"
    for label, side in (("height", shapes[0][-2]), ("width", shapes[0][-1])):
        extent = _extent(side)
        if extent is None or extent % align == 0:
            continue
        low = extent // align * align
        options = [v for v in (low, low + align) if v > 0]
        return (f"the image {label} {extent} is not a multiple of {align}, "
                f"the processor's resize factor; use "
                f"{' or '.join(str(v) for v in options)}")
    return None


register_op(OpSpec("patchify",
    # Axes are regrouped by position (requests x items, then the patch grid),
    # never by a label a shard could follow: `geometry.patchify`.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=(),
    shape_out=(),
    shape_kind="explicit",
    input_rank_is_exact=True,
    input_contract=_patchify_contract,
    formula="y[(r, i), (gh, gw), :] = x[r, i, gh*P:(gh+1)*P, gw*P:(gw+1)*P] "
            "flattened over (channels, temporal, P, P)",
    description="View-op: cut each image item into its patch grid — "
                "[N, n, H, W] -> [N*n, (H/patch)*(W/patch), "
                "channels*temporal*patch^2], one segment per item; a video "
                "[N, n, T, H, W] in ceil(T/temporal) temporal patches (the "
                "last frame repeated), one segment each (`segment: "
                "temporal_patch`) or the whole video one (`segment: item`). "
                "The "
                "processor's work, read as rows (zero FLOPs / zero device "
                "bytes). Sides must be multiples of `align`, the processor's "
                "resize factor.",
    required_dims=("patch", "temporal", "channels", "align"),
    is_view_op=True,
    processor_step=ProcessorStep.CUT,
))


def _pad_rows_contract(shapes: Sequence[Sequence[Any]],
                       dims: Dict[str, Any]) -> Optional[str]:
    if len(shapes) != 1 or len(shapes[0]) < 2:
        return "pad_rows reads one [M, V, ...] tensor"
    rows = _extent(shapes[0][1])
    padded = _extent(dims["rows"])
    if rows is not None and padded is not None and rows > padded:
        return (f"an item holds {rows} rows, more than the {padded} it is "
                f"padded to; give an item that fits (a processor resizes an "
                f"image to fit its padded patch count)")
    return None


register_op(OpSpec("pad_rows",
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=(),
    shape_out=(),
    shape_kind="explicit",
    input_rank_is_exact=True,
    input_contract=_pad_rows_contract,
    formula="y[m, :V] = x[m];  y[m, V:rows] = padding",
    description="View-op: each item's rows padded to `rows` ([M, V, ...] -> "
                "[M, rows, ...]) where a processor pads them; the padding "
                "rows are computed by every op after it (until an `unpad`, "
                "where one strips them).",
    required_dims=("rows",),
    is_view_op=True,
    processor_step=ProcessorStep.PAD,
))


def _unpad_contract(shapes: Sequence[Sequence[Any]],
                    dims: Dict[str, Any]) -> Optional[str]:
    if len(shapes) != 2 or any(len(shape) != 3 for shape in shapes):
        return "unpad reads x [M, R, D] and the unpadded rows [M, V, C]"
    x, ref = ([_extent(e) for e in shape] for shape in shapes)
    if None not in (x[0], ref[0]) and x[0] != ref[0]:
        return f"x and the unpadded rows disagree on items: {x[0]} != {ref[0]}"
    if None not in (x[1], ref[1]) and ref[1] > x[1]:
        return f"{ref[1]} valid rows cannot come out of {x[1]} padded rows"
    return None


register_op(OpSpec("unpad",
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    # The padded rows, and the rows before padding (which say how many of
    # them are real).
    shape_in=(("items", "rows", "width"), ("items", "valid", "patch_width")),
    shape_out=("items", "valid", "width"),
    shape_kind="explicit",
    input_rank_is_exact=True,
    input_contract=_unpad_contract,
    formula="y[m] = x[m, valid(m)]   (the rows the processor did not pad)",
    description="View-op: strip the padding rows `pad_rows` added — x "
                "[M, R, D] keeps the V rows its second input (the rows before "
                "padding, [M, V, C]) holds -> [M, V, D].",
    is_view_op=True,
    processor_step=ProcessorStep.UNPAD,
))

def _group_items_contract(shapes: Sequence[Sequence[Any]],
                          dims: Dict[str, Any]) -> Optional[str]:
    if len(shapes) != 1 or len(shapes[0]) != 3:
        return "group_items reads one [M*k, R, D] tensor"
    items, per, groups = (_extent(shapes[0][0]), _extent(dims["items"]),
                          _extent(dims["groups"]))
    if None not in (items, per, groups) and items != per * groups:
        return (f"{items} items are not {groups} groups of {per}")
    return None


register_op(OpSpec("group_items",
    # The items axis regroups by position (`k` consecutive items into one),
    # never by a label a shard could follow: `geometry.media_rows`.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=(),
    shape_out=(),
    shape_kind="explicit",
    input_rank_is_exact=True,
    input_contract=_group_items_contract,
    formula="y[m, j*R + r, :] = x[m*k + j, r, :]",
    description="View-op: k consecutive items' rows laid end to end as one "
                "item's — [M*k, R, D] -> [M, k*R, D] (an image's crops, "
                "raster order, as one block of rows); `groups` is M and "
                "`items` k, so no item at all (k = 0) still leaves M groups "
                "of no rows. Zero FLOPs / zero device bytes.",
    required_dims=("items", "groups"),
    is_view_op=True,
    processor_step=ProcessorStep.GROUP,
))


register_op(OpSpec("position_table_add",
    # Sample the learned table and add it to the reference in one operation.
    # The source expressions do not materialize an expanded table between a
    # lookup/reindexing and the add, so splitting this into two graph nodes
    # would invent one full-size activation read and write.
    couples_on=(),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["dim"],
    params_fn=_zero,
    # Only the reference has a fixed rank. The table may be [P,D] or
    # [Ph,Pw,D]; an empty second template deliberately contributes no labels
    # to shape-derived FLOPs.
    shape_in=(("B", "s_ctx", "dim"), ()),
    shape_out=(),
    shape_kind="explicit",
    formula="y = reference + map(position_table, reference_positions)",
    description="Add a learned position table to a reference activation, mapping table positions by lookup or reindexing without materializing an expanded intermediate.",
    required_dims=("dim",),
))

# --- Row scatter into the residual stream ---------------------------------
# A multimodal model writes an encoder's output rows into the text stream at
# the placeholder positions: HF `inputs_embeds.masked_scatter(mask, embeds)`
# (replace) and Qwen3-VL's DeepStack `hidden[mask] += embeds_j` (add). Row
# contract: `v[b, n]` goes to the n-th placeholder position of `x[b]`, the
# placeholders taken in sequence order — the row-major order `masked_scatter`
# consumes its source in. The placeholders themselves are not a modelled
# tensor: no mask input, no invented index edge.
#
# Cost is the logical op's full materialization — read `x`, read `v`, write
# `y` through the generic traffic rule — not the source's temporaries (a
# DeepStack `clone` plus a selected read/add/write moves more); fusing it in
# place is the fusion layer's business. Two public ops rather than one with a
# mode: the FLOPs, the reduction law and the layout law differ.


def _known(extent: Any) -> Optional[int]:
    """The extent as a number, or None where the domain is symbolic."""
    value = value_of(extent)
    if isinstance(value, Integer):
        return int(value)
    return value if isinstance(value, int) else None


def _scatter_rows_contract(shapes: Sequence[Sequence[Any]],
                           dims: Dict[str, Any]) -> Optional[str]:
    """`x: [B, S, D]` and one or more `v: [B, N, D]` (each media input's
    rows, one masked scatter writing them all) with ΣN <= S. Rank 3 on
    every side — no implicit flatten of a `[B*N, D]` source, no reliance on
    `B = 1`."""
    if len(shapes) < 2 or any(len(shape) != 3 for shape in shapes):
        return (f"a row scatter reads x [B, S, D] and v [B, N, D], got "
                f"ranks {[len(shape) for shape in shapes]}")
    x, *vs = ([_known(extent) for extent in shape] for shape in shapes)
    for v in vs:
        for axis, label in ((0, "B"), (2, "D")):
            if None not in (x[axis], v[axis]) and x[axis] != v[axis]:
                return (f"x and v disagree on {label}: "
                        f"{x[axis]} != {v[axis]}")
    rows = [v[1] for v in vs]
    if None not in (x[1], *rows) and sum(rows) > x[1]:
        return (f"v writes {sum(rows)} rows into a sequence of {x[1]} "
                f"positions")
    return None


_SCATTER_ROWS_LAYOUT = (
    "Layout: B and D of v must match x's placement (and closure); when x's "
    "sequence is cut over a world W, v's N axis is co-sharded in the same "
    "world as a padded Block, N_rank = ceil(N/W), reached from a replicated v "
    "by a local slice with no communication. This is a uniform-density "
    "co-shard projection: it assumes the placeholder rows spread evenly over "
    "the sequence blocks, not an ownership a mask proves. Any other "
    "placement is refused.")


def _scatter_rows_op(name: str, flops_fn, flops_expr: str, formula: str,
                     description: str) -> None:
    register_op(OpSpec(name,
        # Where v's row n lands depends on where the n-th placeholder falls in
        # the whole sequence, so neither row axis separates on its own: the
        # row axes are POSITIONS of both operands
        # (`axis_propagation.ROW_SCATTER_AXES`), and the dedicated layout
        # handler settles them (see `_SCATTER_ROWS_LAYOUT`).
        couples_on=(),
        flops_fn=flops_fn,
        # `rows` is the op's own slot, bound by the arriving `v`; the
        # canonical display symbols do not carry it.
        flops_expr_override=flops_expr,
        params_fn=_zero,
        shape_in=(("B", "s_ctx", "dim"), ("B", "rows", "dim")),
        shape_out=("B", "s_ctx", "dim"),
        shape_kind="preserve",
        input_rank_is_exact=True,
        input_contract=_scatter_rows_contract,
        # vLLM `_merge_multimodal_embeddings` writes every media input's rows
        # in one masked scatter (models/utils.py:636-673).
        repeats_last_input=True,
        formula=formula,
        description=f"{description} {_SCATTER_ROWS_LAYOUT}",
    ))


_scatter_rows_op(
    "scatter_rows_replace", _zero, "0",
    "y = x;  y[b, p_n] = v[b, n]   (p_n: n-th placeholder of x[b])",
    "Overwrite the placeholder rows of x with the rows of v "
    "(HF `masked_scatter`). Zero FLOPs; bytes read x, read v, write y.")

_scatter_rows_op(
    "scatter_rows_add",
    lambda d: d["B"] * d["rows"] * d["dim"], "B*rows*dim",
    "y = x;  y[b, p_n] += v[b, n]   (p_n: n-th placeholder of x[b])",
    "Add the rows of v onto the placeholder rows of x "
    "(Qwen3-VL DeepStack). B·N·D FLOPs; bytes read x, read v, write y.")


register_op(OpSpec("view_merge",
    # Couples no axis by label: the two axes it merges are POSITIONS of the
    # tensor that arrives (`axis_propagation.VIEW_MERGE_AXES`), and the
    # layout geometry reads them off the DimMap.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=(),
    shape_kind="explicit",
    description="View-op: collapse the last two axes into one, whose width "
                "is their product (zero FLOPs / zero HBM).",
    is_view_op=True,
))

# Broadcast / expand: insert a new axis of size `width` at `axis`, replicating
# the existing tensor along it. Zero FLOPs (memory
# permute under the hood) but materializes the rank lift visibly.
# Used by MLA where the per-layer shared k_pe
# (shape [B, s_ctx, d_qk_rope]) has to
# fan out across all n_heads heads before getting concatenated with k_nope
# (shape [B, s_ctx, n_heads, d_head]); without this op the shape geometry of
# `key_concat` doesn't add up (8192 + 64 ≠ 64·192). Spec writes
# `dims: { in: <inner>, width: <fan_out> }` and node.axis chooses where
# the new axis lands.
#
# The parameter is `width` and not `n_heads` because the axis it inserts is
# whatever the call site names, and `n_heads` is a symbol `dims.yaml`
# classifies as a head count tensor parallelism divides. A key the shard policy
# knows takes the axis's identity away from the binding: the three
# hyper-connection call sites bind `d_model` and `n_hidden_copies` here, and at
# TP=8 the residual width came out 7168/8 and the copies axis collapsed
# 4 → 1, under a key that says "heads" about an axis that holds neither.
# Under `width` — a name the policy does not know — the binding is resolved
# against the per-rank frame, so `width: n_heads` still fans out to this rank's
# head count and `width: d_model` stays whole.
#
# Convention: `axis` is the position in the OUTPUT (numpy
# `np.expand_dims` semantics). For input [B, s_ctx, d_model] (rank 3):
#   axis=-1 → [B, s_ctx, d_model, width]   (append at end)
#   axis=-2 → [B, s_ctx, width, d_model]   (insert before last)
#   axis=-3 → [B, width, s_ctx, d_model]   (insert before second-to-last)
# Default axis is -1 (append).
register_op(OpSpec("broadcast",
    # It inserts a NEW axis; whatever was already sharded keeps its symbol.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=(),
    shape_out=(),
    shape_kind="explicit",
    description="View-op: insert a new axis of size `width` at `axis` (np.expand_dims semantics). "
                "Zero FLOPs / zero HBM (PyTorch `.expand()` is stride=0).",
    required_dims=("width",),
    is_view_op=True,
    # `width` stride-0 copies of the input: a reader loads the input once.
    repeats_input=True,
))


register_op(OpSpec("head_repeat",
    # The head axis changes extent across the op, so an incoming shard of it
    # does not come out the other side describing the same partitioning.
    couples_on=("n_heads",),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=(),
    shape_out=(),
    shape_kind="explicit",
    formula="y = repeat_interleave(x, n_heads // x.shape[1], dim=1)",
    description="Repeat a head-first [B,N,S,D] tensor along its existing head axis to [B,n_heads,S,D].",
    required_dims=("n_heads",),
))

def _rope_emb_op(name: str, label: str, formula: str, description: str,
                 position_axes: int = 1, sections_key: str = "") -> None:
    """`rope_emb`, its inverse and `mrope`: one rotation geometry — two
    directions, and one or three position streams."""
    register_op(OpSpec(name,
        label=label,
        rotary_position_axes=position_axes,
        rotary_sections_key=sections_key,
        is_side_buffer=True,
        couples_on=("dim",),
        # After a `head_split` the tensor is `[B, n_heads, s_ctx, d_head]`, and
        # every op on the per-head path sees that rank. A second signature rather
        # than a wildcard: `n_heads` is a label, and the label is what says axis 1
        # of the output is axis 1 of the input.
        alt_shape_in=(("B", "n_heads", "s_ctx", "dim"),),
        # 3 per rotated element, as `rope` (4 mul + 2 add per pair).
        flops_fn=lambda d: 3 * d["B"] * d["s_ctx"] * d["dim"],
        params_fn=_zero,
        shape_in=("B", "s_ctx", "dim"),
        shape_out=("B", "s_ctx", "dim"),
        shape_kind="preserve",
        formula=formula,
        description=description,
        # Same geometry as `rope`: the rotation pairs elements WITHIN the feature
        # axis, so that one may not be split, while heads and positions are
        # untouched and may. The absence of a `shard_dim` entry is the load-bearing
        # part of this list.

        required_dims=("dim",),
    ))


_rope_emb_op("rope_emb", "RoPE",
    "(x_even, x_odd) ← (x_even·cos − x_odd·sin, x_even·sin + x_odd·cos)",
    "Rotary position embedding (3-D variant — head dim folded into the last axis).")
# The inverse rotation (conjugate frequencies). DeepSeek-V4 rotates the rope
# dims of the attention OUTPUT back before the output projection
# (inference/model.py `apply_rotary_emb(o[..., -rd:], freqs_cis, True)`), and
# both serving frameworks run it as its own kernel (vLLM
# `fused_inv_rope_fp8_quant`, sglang `fused_rope_inplace(inverse=True)`).
_rope_emb_op("rope_emb_inverse", "Inverse RoPE",
    "(x_even, x_odd) ← (x_even·cos + x_odd·sin, −x_even·sin + x_odd·cos)",
    "Inverse rotary position embedding (conjugate rotation) — undoes `rope_emb` at the same positions.")
# Multimodal RoPE (Qwen2-VL "M-RoPE"; Qwen3-VL, Qwen3.5 / Qwen3.8). Positions are
# [3, T] — temporal, height, width — and `mrope_section` assigns each channel
# section (interleaved in Qwen3-VL, `recomposition_frequencies` in
# modeling_qwen3_vl.py:404-431) to one of them. Per rotated element the work
# is a rotation's, so the geometry and FLOPs are `rope_emb`'s, 3-D or 4-D,
# whole-head or a partial slice. What is its own is the kernel: vLLM builds
# `MRotaryEmbedding` whenever the rope parameters carry `mrope_section`
# (rotary_embedding/__init__.py:102-111) and runs `triton_mrope` on q and k in
# one launch (mrope.py:374-381); sglang's Qwen3-MoE attention excludes it
# from the fused qk-norm+rope kernel (qwen3_moe.py:515-530). Four-axis xdrope
# (HunyuanOCR) is a different kernel and is its own op, `xdrope`.
_rope_emb_op("mrope", "M-RoPE",
    "(x_i, x_j) ← rotation by θ_k · pos_a(k)   (channel section k reads position stream a ∈ {t, h, w})",
    "Multimodal rotary position embedding: three position streams (t, h, w), each channel section rotated by its own stream's positions. Same per-element work as RoPE.",
    position_axes=3, sections_key="mrope_section")

# Walsh-Hadamard rotation of the feature axis: `y = (H_n · x) · n^-1/2`,
# H_n the Sylvester Hadamard matrix of order n (a power of two). DeepSeek-V4's
# lightning indexer applies it to its query after the partial RoPE and to its
# compressed key after the compressor's RoPE, before both are quantised
# (reference inference/model.py:247-251 `rotate_activation` =
# `fast_hadamard_transform.hadamard_transform(x, scale=x.size(-1) ** -0.5)`,
# called at :414 and :369). Orthogonal, so scores between two rotated
# operands equal the unrotated ones; what it changes is the value
# distribution the quantiser sees.
#
# Authored, it is a DATAFLOW step of a forward. A cache format's
# `transform: hadamard` (`cache_formats/fp8_pow2_hadamard128.yaml`, the op
# named by `CacheFormat.TRANSFORMS`) is the same rotation done by how a
# serving stack ENCODES a tensor it stores — absent from the forward
# (GLM-5.3-Flash's index keys) — and it is this op too: the serving graph
# grafts one on the edge into the cache and one on the operand the cache's
# readers multiply it with (`inference.materialize_cast`). Never a property
# of the bytes alone.
#
# Cost: the fast transform's n·log2(n) butterfly adds per row, plus the
# n multiplies of the scale. Every output channel reads the whole axis, so
# the axis couples (a cut of it needs the whole row back, as a norm does).
register_op(OpSpec("hadamard_transform",
    label="Hadamard",
    couples_on=("dim",),
    # The per-head frame: the indexer rotates each query head after its split.
    alt_shape_in=(("B", "n_heads", "s_ctx", "dim"),),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["dim"] * (_log2(d["dim"]) + 1),
    params_fn=_zero,
    shape_in=("B", "s_ctx", "dim"),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="preserve",
    formula="y = (H_n · x) / sqrt(n)   (n = dim, a power of two; H_n the Sylvester Hadamard matrix)",
    description=(
        "Walsh-Hadamard rotation of the last axis, scaled by n^-1/2 "
        "(orthonormal). DeepSeek-V4 indexer `rotate_activation`; also "
        "grafted where a cache format's stored `transform` is hadamard."
    ),
    required_dims=("dim",),
))

# Per-layer RoPE inverse-frequency buffer. Not a learnable weight — just
# `1 / θ^(2i/D_head)` materialized once and stored when the modeling code
# registers it with `persistent=True` (older transformers style; LLaMA-2,
# GLM-4), or when whatever exported the checkpoint saved it anyway (Kimi-K2,
# Moonlight: persistent=False, yet serialized). Modern DeepSeek-V3 / Qwen /
# LLaMA-3 checkpoints do not carry it and this op is not needed there.
# `dim` is the element count per layer: the modeling code's formula when that
# code persists it, else the length the checkpoint stores, bound
# `checkpoint.<tensor>[0]`; wrap with `repeat: L` at the model layer.
register_op(OpSpec("rotary_inv_freq_buffer",
    is_side_buffer=True,
    couples_on=(),
    flops_fn=_zero,
    params_fn=lambda d: d["dim"],
    # Stored — it is a real tensor some checkpoints serialise — but not a
    # parameter, so it belongs in the byte accounting and not in the HF total.
    is_buffer=True,
    # No `storage_dtype` declaration, deliberately. This buffer's width is NOT a
    # property of the op: Llama-2 ships it F32 (an older transformers built it
    # at torch's default and never cast it), while glm-4-9b and Kimi-K2 ship the
    # same table at BF16. A per-checkpoint fact belongs in that checkpoint's
    # spec, where `quant_context.overrides` already names tensors literally.
    weight_shape=("dim",),
    # The rotary module registers the table as its `inv_freq` buffer, so
    # that is the name a checkpoint serialises it under.
    weight_tensor_suffix=".inv_freq",
    shape_out=("dim",),
    # Zero-input parameter bank: shape from dim binding alone.
    shape_kind="source",
    formula="inv_freq[i] = θ^(-2i / D_head)",
    description="Precomputed RoPE inverse-frequency buffer (persistent=True).",
    required_dims=("dim",),
))

# Generic learnable parameter vector — zero FLOPs, params = `dim`. Used
# for standalone weights that aren't tied to a projection (e.g. `A_log`,
# `dt_bias`, `e_score_correction_bias`, RMSNorm gain vectors when they
# live outside a norm op). Pair with `repeat:` at the caller to scale
# across layers/heads.
register_op(OpSpec("parameter_vector",
    is_side_buffer=True,
    couples_on=(),
    flops_fn=_zero,
    params_fn=lambda d: d["dim"],
    shape_out=("dim",),
    # Zero-input — shape from dim binding alone.
    shape_kind="source",
    formula="p ∈ R^dim",
    description="Standalone learnable parameter bank (zero FLOPs).",
    weight_shape=("dim",),
    # nn.Parameter — checkpoint stores the bare attribute name with no
    # `.weight` suffix.
    weight_tensor_suffix="",
    required_dims=("dim",),
))


# n-D generalization of parameter_vector. Used for nn.Parameter weights
# whose shape is rank ≥ 2 — e.g. V4-Pro's `Block.hc_attn_fn`
# `nn.Parameter(torch.empty(mix_hc, hc_dim))`. The spec author writes
# `shape: [d_hc_mix, "n_hidden_copies * d_model"]` on the call site; engine resolves each entry
# against dim bindings and stores as `dims["_shape"]`. Params count
# is the product of the resolved dims, matching torch.numel.
def _param_tensor_count(d):
    sh = d.get("_shape") or ()
    n = 1
    for v in sh:
        n = n * v
    return n


register_op(OpSpec("parameter_tensor",
    is_side_buffer=True,
    couples_on=(),
    flops_fn=_zero,
    params_fn=_param_tensor_count,
    shape_out=(),
    # Zero-input source — shape comes from the call-site `shape:` list.
    shape_kind="source",
    params_expr_override="∏ shape",
    formula="p ∈ R^shape  (n-D learnable nn.Parameter)",
    description="Standalone learnable n-D parameter bank — generalization of parameter_vector to rank ≥ 2. Used for V4-Pro `hc_*_fn` weights and similar bare nn.Parameter tensors.",
    weight_shape=("_shape",),
    weight_tensor_suffix="",
    required_dims=(),
))

# --- Generic torch-level selection / reduction ops ---
# `topk` and `gather` mirror torch.topk / torch.gather and show up wherever
# routing or sparse selection happens (MoE routing, DSA index gather, etc.).
# `sum_reduce` and `normalize_sum_one` are the matching shape-reducing /
# normalizing helpers that round out a routing pipeline.

register_op(OpSpec("topk",
    # Ranking is over all of `in`.
    couples_on=("in",),
    # Partial selection: O(in) for the typical in ≫ k case (heap-based partial sort).
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["in"],
    params_fn=_zero,
    # Indices are a SELECTION, not an activation.
    output_kind="mask",
    shape_in=("B", "s_ctx", "in"),
    # Per head: MiniMax-M3's indexer picks each index head's own key blocks
    # (HF `block_scores.topk(...)` over [B, H_idx, S_q, num_key_blocks]).
    alt_shape_in=(("B", "n_heads", "s_ctx", "in"),),
    shape_out=("B", "s_ctx", "k"),
    shape_kind="axis_select", out_key="k",
    formula="indices, values = topk(x, k) along last dim",
    description="Top-k selection over a dim (default last). Returns the k largest values and their indices.",
    # Top-k selects along the SELECTION axis; a token shard is orthogonal to it,
    # so each rank scores its own tokens with no peer. Declared because the
    # alternative is placement assuming it, and an assumption that happens to be
    # right is indistinguishable from one that is not.
    #
    # `k` is the node's; `in` is the ranked axis of the tensor that arrives
    # (`shape_in`), read off the edge like any other label.
    required_dims=("k",),
))

register_op(OpSpec("gather",
    # An index may point anywhere along `in`.
    couples_on=("in",),
    flops_fn=_zero,
    params_fn=_zero,
    # The second input is an index list — a selection, not an activation.
    input_kinds=("activation", "mask"),
    # It reads the `k` it picks of each token's `in`, not all of them —
    # under every deployment, since `k < in` always (a router's top-k
    # gather reads n_experts_active of its n_route_targets scores per
    # token: DeepSeek-V3 8 of 256).
    gather_source=0,
    shape_in=(("B", "s_ctx", "in"), ("B", "s_ctx", "k")),
    shape_out=("B", "s_ctx", "k"),
    # Replaces the gather axis dim with the index count (k). axis from
    # node.axis (default last). Note: only the first (data) input's shape
    # is used; the index input shape governs k.
    shape_kind="axis_replace", out_key="k",
    formula="y[b,s,k] = x[b, s, idx[b,s,k]]",
    description="Index-based gather (zero FLOPs, memory-only).",
    # `k` is what axis_replace writes, so it is required, not merely hoped
    # for: an unbound `k` used to fall through to the enclosing scope and
    # produce whatever width happened to be in it.
    # Indexing runs along the LAST axis (`node.axis` defaults there) and both
    # operands are per-token, so a token shard passes through: each rank gathers
    # its own tokens with its own indices.

    required_dims=("k",),
))

# Permute / combine — MoE token routing boundary ops.
#
# Permute is a memory-only index reordering (zero FLOPs); combine does a
# weighted reduce (genuine MACs over routing weights). Together they bracket
# a per-expert zone.
#
# The two numbers, kept apart because conflating them is what mis-billed
# these ops:
#
#   M = B·s_ctx·n_experts_active·dp_attention / n_route_targets
#       the MEAN rows ONE expert receives. A token visits `n_experts_active`
#       of `n_route_targets` destinations, and the two counts differ only
#       where some destinations hold no weights (LongCat-Flash's 256
#       `nn.Identity()` reserve beside its 512 weighted experts, 768
#       destinations in all).
#   R = M · n_experts
#       the rows the dispatch WRITES — every weighted expert's slice, once.
#       `n_experts` and not `n_route_targets`: a destination with no weights
#       receives no permuted row. It is the same product the comm side
#       already forms (`layout/moe.py`, `global_rows = per_expert * experts`),
#       so there is one owner of it and not two spellings.
#
# `R` is these ops' OWN output geometry: each runs once per layer and each
# touches the whole buffer. `M` is the zone's, and the walk hands it to the
# zone directly (`Walk.dispatch_rows`) rather than leaving it to be read back
# axis 0 — which is what it was, while this template said `M` here.
#
# `R` is an op-slot label resolved by law at the walk, exactly as `M` is
# (`Evaluator._tokens_per_expert`). It is deliberately NOT a `dims.yaml`
# symbol: its `kind:` could only be "a product of other symbols", which is
# not a kind, and a second statement of a quantity `dims.yaml` has already
# decided is a drift waiting to happen. Nor is the product written in the
# slot: `shape_out` carries LABELS — `signature_for` refuses the whole op if
# one of them is not an identifier — so the label goes here and the law goes
# in the engine.
#
# Place a `permute` after the routing pipeline (gate → softmax → topk)
# and a `combine` before the residual add. `permute` writes the tokens laid
# out per expert, `[n_expert_copies, M, d_model]`, and every node between them
# runs on that layout: each expert bank's weights lead with the same
# `n_expert_copies` axis (`shape:`), so a `linear` pairs expert e's matrix
# with slice e of the buffer, and the bank's FLOPs are `E·M` rows' worth —
# the tokens the weighted experts receive.
#
# Under Expert Parallelism (ep_moe > 1) the actual token movement
# between expert-owning ranks is a separate all_to_all node grafted by
# layout — sibling of permute (before) and combine (after). Permute
# itself stays pure-local (just the shape transform).
register_op(OpSpec("permute",
    # Roles, not shapes: the second input is named here but still gets
    # no `shape_in` template (see below) — a template would make the
    # routing index a shape source.
    input_roles=("tokens", "routing"),
    is_dispatch_boundary=True,
    # Each routed row is one token's row: the rank reads the rows its own
    # experts receive, not the step's tokens. Under expert parallelism
    # without a request cut every rank holds every token and routes only
    # `topk/ep` of them per token to itself, so reading the whole stream
    # billed Qwen3-Coder-Next's EP=16 prefill dispatch 134 MB of tokens
    # for the 84 MB of rows it gathers (`gather_source`). Without EP the
    # routed rows outnumber the tokens and the read is the stream, once.
    gather_source=0,
    # The dispatch itself: tokens leave partitioned the way they arrived and
    # come back partitioned by expert. Same tokens, different partitioning — an
    # all_to_all, never a no-op. BOTH token axes are named because either one
    # can be the partitioning it re-cuts, and neither survives it: `B` when
    # DP-attention gave the rank its own requests, `s_ctx` when context
    # parallelism gave it its own positions.
    couples_on=("B", "s_ctx"),
    # `M` divides the tokens by the routing counts, read off the binding
    # (`Evaluator._tokens_per_expert`).
    binding_dims=("n_experts_active", "n_route_targets"),
    flops_fn=_zero,
    params_fn=_zero,
    # Declares ONE input where its graph passes two — `permute: { op:
    # permute, in: [x, gate] }` — so every `permute` node is refused an index
    # signature. Declaring the router input is the obvious fix and is NOT
    # done here: it moves Qwen3-Coder-Next's shape-derived flops by 6.3 GF,
    # because the second template becomes a shape source and the tensor it
    # describes is a routing index, not a value the downstream shapes follow.
    # The arity is wrong AND the fix is not free; both need to be true before
    # this changes.
    shape_in=("B", "s_ctx", "d_model"),
    # The buffer it writes: the bank's experts outermost, each expert's MEAN
    # rows `M` inside, the token width last. `n_expert_copies` is the copies
    # axis — the same symbol every weight of the bank leads with (`shape:`),
    # and the only one expert parallelism divides — so the zone's activations
    # and weights pair expert by expert (`ops.weight_leading_axes`). `M` is a
    # mean: how many rows each expert really receives varies, and what a
    # rank carries at the busiest is `ExpectedRoutedLoad`'s other reading.
    shape_out=("n_expert_copies", "M", "d_model"),
    # (B,S,…) → (E,M,…): the token axes are re-laid out per expert. Not
    # derivable from the input half — which is what EXPLICIT means — so the
    # output labels are written down and `signature_for` reads them; the
    # DimMap is that signature's view, shared with every other op, rather
    # than a hand-written map for these two.
    shape_kind="explicit",
    formula="y[e,m,:] = x[token_for(e, m), :]   (M = B·s_ctx·n_experts_active/n_route_targets, the mean rows per expert)",
    description=(
        "MoE permute: rearrange tokens into per-expert buffers. "
        "Pure memory op (zero FLOPs); under EP the surrounding "
        "all_to_all is grafted by layout."
    ),
    # Under expert parallelism this op IS the re-partition. The stream arrives
    # split by REQUEST — DP-attention gave each rank its own — and leaves split
    # by which rank OWNS the expert each token routed to. Two partitionings of
    # one token set, which is why the exchange has always had to be grafted
    # structurally: a placement carries a kind and an axis and has nowhere to
    # record which partitioning of that axis it is.
    #
    # The `in_state` reads `B` and not `s_ctx`, and the difference is the whole
    # deployment. `s_ctx` is what CONTEXT parallelism cuts, and it says the
    # tokens on this rank are positions of one sequence, which attention may
    # not read across; `B` says they are whole requests, which it may. Written
    # `s_ctx`, this mode matched the residual stream's sequence cut under CP and
    # made expert parallelism borrow context parallelism's repairs — see
    # `layout.propagation.LayoutPropagation.residual_sequence_family`.
    #
    # EP WITHOUT the request cut is a different deployment and derives a
    # different bill, correctly and with no mode of its own: every rank holds
    # every token, so nothing re-partitions on the way in, the rank runs the
    # experts it owns, and each token leaves with a partial sum that the block
    # exit reduces. One all_reduce, no all_to_all — which is what a TP+EP
    # deployment without DP-attention actually runs.
    #
    # Naming the destination axis says it without extending the type. The
    # copies axis `n_expert_copies` is what expert parallelism divides — the
    # glossary calls it "the only thing expert-parallelism shards" — so a mode
    # from the sequence axis to it is precisely "tokens are now grouped by
    # expert owner", and the transition to a consumer expecting its own tokens
    # back derives the all_to_all with no op-intrinsic exception.

    required_dims=("d_model",),
))

register_op(OpSpec("combine",
    input_roles=("expert_out", "routing"),
    is_dispatch_boundary=True,
    # The return leg of `permute`, and coupled for the same reason: it sums
    # each token's experts back into the token.
    couples_on=("n_expert_copies", "M"),
    # One weighted add per element of the expert buffer that arrives: the
    # rows reduced are the routed rows ON THIS RANK, `n_expert_copies · M`
    # read off the first operand (`flops.bind_labels`), which is what the
    # node's bytes already bill. Counting the token side instead
    # (`B·s_ctx·n_experts_active`) billed every token's every slot on each
    # rank — `ep`-fold too many under EP without DP-attention, where a rank
    # holds every token but its own experts produce only its share of the
    # rows. SOL counts the rows this rank has to reduce, the same fact the
    # node's bytes bill. vLLM v0.32.0's default Triton path does not reach
    # that floor: it zeroes the whole buffer and sums all T·k slots
    # (`fused_moe.py:1824`, `moe_sum` at :1850 without `topk_ids`; only the
    # Marlin path, `experts/marlin_moe.py:393`, skips non-local slots).
    flops_fn=lambda d: d["n_expert_copies"] * d["M"] * d["d_model"],   # weighted reduce
    params_fn=_zero,
    # Two templates: the expert buffer and the routing tensor the graph
    # already passes. Last axis of routing is `n_route_width` so
    # `flops_fn` cannot pick it up — that axis is `n_experts_active` or
    # `n_route_targets` depending on whether the spec builds top-K, and
    # naming it either would price combine by the class count.
    shape_in=(("n_expert_copies", "M", "d_model"), ("B", "s_ctx", "n_route_width")),
    shape_out=("B", "s_ctx", "d_model"),
    # Inverse of permute: (E,M,…) → (B,S,…). The expert and per-expert row
    # axes go; the token axes come from the ROUTING operand, which is the one
    # tensor here that still carries them. Both facts are in the templates
    # above — `B` and `s_ctx` appear only in the second — so the DimMap is
    # the signature's view and needs no hand-written map.
    shape_kind="explicit",
    formula="y[b,s,:] = Σ_k routing_weight[b,s,k] · expert_out_k",
    description=(
        "MoE combine: weighted reduce of per-expert outputs back to "
        "(B,s_ctx,d_model). Under EP the all_to_all that returns tokens to their "
        "owners is grafted after this node by layout."
    ),
    # The routing weight is REPLICATED — one scalar per (token, slot),
    # identical on every rank — so scaling a partial-sum expert output by
    # it is linear in the partial operand: `w·(a₁+a₂) = w·a₁ + w·a₂`. The
    # routed sum therefore stays a partial and reduces ONCE at the block
    # exit, alongside the shared expert's own partial through `add`'s
    # partial_sum mode.
    #
    # `mul` carries the same law with its factor checked at the node: it is
    # not linear when BOTH operands are partial, or when the factor varies
    # across the ranks, and the layout tells those apart from a replicated
    # gate (Qwen3-Next's sigmoid on the shared expert).
    #
    # Undeclared, the routed branch reduced at `combine` and the shared
    # branch again at `moe_out`: two all_reduces per MoE block where an
    # implementation performs one. DeepSeek-V2-Lite read 6 against the
    # injector's 4 on exactly this.

    required_dims=("d_model",),
))

# Hash-MoE routing: deterministic token→expert assignment via a fixed
# `tid2eid` hash table (DeepSeek-V4 hash-router variant), the structural
# analog of `topk` for score-based routers: `indices = self.tid2eid[input_ids]`
# (inference/model.py Gate.forward). Reads the token ids `[B, s_ctx]` and the
# table, and emits the per-token expert assignment `[B, s_ctx, n_experts_active]`
# (int32 indices) that `gather` / `permute` / `combine` consume. No learnable
# params; the lookup is O(B·s_ctx·n_experts_active) table reads, one per
# token slot.
register_op(OpSpec("hash_route",
    # The table's rows are the vocabulary: which row a token reads is its id,
    # so a cut of `n_vocab` answers only the ids that land in it.
    couples_on=("n_vocab",),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["n_experts_active"],
    params_fn=_zero,
    # Expert indices, like `topk`'s: a selection that `gather` reads the
    # routing weights at, not an activation.
    output_kind="mask",
    # Two inputs: (input_ids [B,s_ctx], tid2eid [n_vocab, n_experts_active]).
    # The table is a parameter on input 1, and a step reads the rows its
    # token ids select, not the table.
    lookup=Lookup(ids=0, table=1),
    shape_in=(("B", "s_ctx"), ("n_vocab", "n_experts_active")),
    shape_out=("B", "s_ctx", "n_experts_active"),
    shape_kind="explicit",
    input_dtype="I32",
    formula="eid[b,s,k] = tid2eid[ids[b,s], k]",
    description=(
        "Hash-MoE router: deterministic token→expert assignment, the token id's "
        "row of a fixed tid2eid table. Emits (B, s_ctx, n_experts_active) int32 "
        "expert indices at which the routing weights are gathered — the hash "
        "analog of topk for score-based routers."
    ),
    required_dims=("n_experts_active", "n_vocab"),
    output_dtype="I32",
))


register_op(OpSpec("sum_reduce",
    couples_on=("in",),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["in"],
    params_fn=_zero,
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx"),
    # Drops the reduced axis (default last). Honors node.axis.
    shape_kind="explicit",
    formula="y = Σᵢ x[..., i]",
    description="Reduction sum along a dim (default last).",
    required_dims=("in",),
))

register_op(OpSpec("normalize_sum_one",
    # Divides by a sum over all of `in`.
    couples_on=("in",),
    # One pass for sum, one pass for divide.
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["in"],
    params_fn=_zero,
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx", "in"),
    shape_kind="preserve",
    formula="y = x / (Σᵢ xᵢ + ε)",
    description="Sum-to-one normalization. Used for top-K routing weights when norm_topk_prob=True.",
    required_dims=("in",),
))

register_op(OpSpec("scale_const",
    couples_on=(),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["in"],
    params_fn=_zero,
    shape_in=("B", "s_ctx", "in"),
    # The per-head frame, where Kimi-K3's KDA decay gate is scaled by
    # gate_lower_bound after its sigmoid.
    alt_shape_in=(("B", "n_heads", "s_ctx", "in"),),
    shape_out=("B", "s_ctx", "in"),
    shape_kind="preserve",
    formula="y = x · c   (c = config-time constant, e.g. routed_scaling_factor)",
    description="Element-wise scaling by a config-time scalar (no learned param).",
    required_dims=("in",),
))

# --- noaux_tc grouped routing primitives ---
# Two algorithm-specific steps in DeepSeek's `topk_method: noaux_tc` pipeline
# (DSv3, V3.2, Kimi-K2/K2.5, MiMo-V2, Moonlight, GLM-MoE-DSA). Both are
# parameter-free; FLOPs are tiny vs the expert MLP path but they live in the
# graph so the diagram + accounting reflect the true forward.

register_op(OpSpec("group_top2_sum",
    couples_on=("in",),
    # Reshape to (n_expert_groups, in / n_expert_groups), take top-2 per
    # group, and sum.
    # One linear pass over the in elements suffices. The "2" is a fixed
    # algorithm constant from the DeepSeek-V3 noaux_tc paper (§2.1.2) —
    # not config-driven; HF source for DSv3/V3.2/GLM4-MoE/dots1 all
    # literally call `.topk(2, dim=-1)`.
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["in"],
    params_fn=_zero,
    shape_in=("B", "s_ctx", "in"),
    shape_out=("B", "s_ctx", "n_expert_groups"),
    shape_kind="axis_replace", out_key="n_expert_groups", axis_locked=-1,
    formula="g[n] = scores.view(n_expert_groups, in / n_expert_groups)[n].topk(2).sum()   (the 2 is a paper constant)",
    description=(
        "noaux_tc grouped scoring: reshape scores to (n_expert_groups, in / n_expert_groups), "
        "sum the two largest entries per group. n_expert_groups comes from config.n_group. "
        "The literal 2 is fixed by the DeepSeek-V3 paper (not a config hyperparameter)."
    ),
    required_dims=("in",),
))

register_op(OpSpec("group_mask_apply",
    # The mask is per group and `in` is grouped: cutting either one without
    # the other misaligns them.
    couples_on=("in", "n_groups_active"),
    # Build per-expert mask from selected group indices, multiply scores by mask.
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["in"],
    params_fn=_zero,
    # The second input is the selected-group index list — a selection.
    input_kinds=("activation", "mask"),
    shape_in=(("B", "s_ctx", "in"), ("B", "s_ctx", "n_groups_active")),
    shape_out=("B", "s_ctx", "in"),
    shape_kind="preserve",  # output = first input (scores), mask just zeros entries
    formula="mask[i] = 1 if group(i) ∈ selected_groups else 0;   out = scores ⊙ mask",
    description=(
        "noaux_tc grouped masking: zero out experts whose group wasn't picked by group_topk. "
        "n_groups_active = top groups picked per token (config.topk_group)."
    ),
    required_dims=("in",),
))

# The maximum of each of `out` contiguous groups of the last axis.
# DeepSeek-V2 `topk_method: group_limited_greedy` (transformers
# DeepseekV2TopkRouter.forward; vLLM `grouped_topk` without a correction
# bias) scores each expert group by its best expert:
# `group_scores = scores.view(n_group, -1).max(dim=-1).values`, so its router
# binds `out: n_expert_groups`. MiniMax-M3's indexer max-pools each index
# head's scores over key blocks (HF `scores.view(B, H_idx, S_q,
# num_key_blocks, block_size).amax(-1)`), binding `out` to the block count.
# The FLOPs are one compare per input element; the head axis of a per-head
# input is a leading axis the rank-3 template does not cover, so the cost
# binds it off the arriving shape (`eval/flops.py`).
register_op(OpSpec("group_max",
    couples_on=("in",),
    # One pass over the in elements.
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["in"],
    params_fn=_zero,
    shape_in=("B", "s_ctx", "in"),
    alt_shape_in=(("B", "n_heads", "s_ctx", "in"),),
    shape_out=("B", "s_ctx", "out"),
    shape_kind="axis_replace", out_key="out", axis_locked=-1,
    formula="g[n] = x.view(out, in / out)[n].max()",
    description=(
        "Grouped maximum: reshape the last axis to (out, in / out) and take "
        "each group's maximum. Expert-group scoring (DeepSeek-V2 "
        "group_limited_greedy) and key-block pooling (MiniMax-M3's indexer)."
    ),
    required_dims=("in", "out"),
))

# 1D convolution (standard, grouped or depthwise via `groups`).
#   - Full conv (groups=1):   params = C_out · (C_in/groups) · s_conv_kernel (+ bias)
#   - Grouped conv:           the same declaration; each output channel sees
#                             one input group rather than all C_in channels.
#   - Depthwise (groups=C_in=C_out=C): the declared middle weight axis is 1,
#                                      so params = C · s_conv_kernel (+ C bias).
# Bias accounted via `bias` dim (0/1). Element-wise activation (e.g. SiLU)
# applied after the conv is a separate `activation` node.
def _conv1d_weight_elements(d):
    shape = resolve_shape_template(
        ("C_out", "C_in // groups", "kernel_size"), d)
    if not shape:
        return 0
    n = 1
    for extent in shape:
        n = n * extent
    return n

register_op(OpSpec("conv1d",
    tp_axis_placement=True,
    # `s_ctx` because a kernel of width k reads the k-1 tokens before it, so a
    # sequence cut needs a halo. `C_in` because it is the contraction whenever
    # `groups < C_in` — which no call site in the corpus currently is: every
    # conv1d here is the depthwise conv of an SSM mixer (groups == C_in), and
    # those ARE separable along the channel. So this entry is deliberately
    # conservative rather than tight. Separability that holds only for some
    # bindings of `groups` is not a fact about the op, and the field's
    # authoring rule prices the two mistakes differently: over-declaring
    # leaves a recorded assumption over behaviour that is still pass-through,
    # under-declaring divides a cost that was never divisible. If a grouped
    # conv ever lands, it is already covered.
    couples_on=("C_in", "s_ctx"),
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * _conv1d_weight_elements(d)
                       + d["B"] * d["s_ctx"] * d["C_out"] * _flag(d, "bias"),
    params_fn=lambda d: _conv1d_weight_elements(d)
                        + d["C_out"] * _flag(d, "bias"),
    shape_in=("B", "s_ctx", "C_in"),
    shape_out=("B", "s_ctx", "C_out"),
    # Spec dataflow is feature-last [B, S, C]: every SSM / linear-attention
    # mixer that uses conv1d (Mamba-2, gated-delta-net) feeds it [B, S, C]
    # directly — the PyTorch NCL `.transpose(1,2)` around the conv is NOT
    # modeled in the graph. So the channel axis is LAST; replace it
    # C_in → C_out. (Declaring it at axis 1 put C_out in the SEQUENCE slot,
    # injecting `C` spurious tokens that inflated every downstream op's
    # shape — e.g. NemotronH decode rmsnorm/scan at 18432× the real size.)
    shape_kind="axis_replace", out_key="C_out", axis_locked=-1,
    formula="y[b,o,s] = Σ_{i∈group(o),k} w[o,i,k] · x[b,i,s-k]",
    description="1D convolution. Set `C: <channels>` + `groups: <C>` for depthwise; otherwise `C_in` / `C_out` / `groups: 1`.",
    # On-disk: (C_out, C_in/groups, s_conv_kernel). Depthwise (groups==C_in) → middle
    # axis collapses to 1.
    # The one weight shape that genuinely computes, and it still says so as
    # data: `C_in // groups` is 1 for the depthwise convs every SSM / linear-
    # attention mixer runs, C_in for a standard conv.
    #
    # The third axis used to be named `n_experts_active` — the glossary's
    # "top-k experts picked per token" — and all 20 call sites bound it with
    # `n_experts_active: s_conv_kernel`. It worked because a slot name is only
    # a dict key, but a primitive naming its kernel-width slot after an
    # unrelated model concept is the op-slot vocabulary borrowing from the
    # model's, which is the thing the three vocabularies exist to keep apart.
    # `kernel_size` is torch's own name for it, matching this op's other slots
    # (`C_in`, `C_out`, `groups`).
    weight_shape=("C_out", "C_in // groups", "kernel_size"),
    # Params project from the tensors the module declares. In particular the
    # depthwise result comes from the middle weight axis resolving to 1; it is
    # not a second `groups == C` branch kept in the cost formula.
    tensors=(
        TensorDecl(".weight", ("C_out", "C_in // groups", "kernel_size")),
        TensorDecl(".bias", ("C_out",), when="bias", kind="vector", axis_refs=(0,)),
    ),
    # `kernel_size` joins them: flops, params and the weight all read it, so a
    # node that forgets it should fail at load with its own name on the error
    # rather than deep inside a shape function.
    # `dilation` (torch's own name) is read by no cost: a dilated conv does
    # the same work. It is the fact a framework dispatches on — the causal
    # conv kernels that fold the activation in (`causal_conv1d_fn` /
    # `causal_conv1d_update`) take no dilation, so a dilated conv (Qwen PLE,
    # dilation = n-gram size) runs elsewhere — and a binding's `when:` asks it
    # (`eval.fusion.GraphCondition.dims`). Required, so no conv reads as
    # undilated by omission.
    required_dims=("C_out", "groups", "kernel_size", "dilation"),
))


# 2D convolution over a FLATTENED square patch grid. Unlike `conv1d` (depthwise,
# sequence-preserving), this is a strided full conv: each output token contracts
# a `kernel_size x kernel_size` window over all `C_in` channels into `C_out`, and
# a stride > 1 shrinks the grid stride^2-fold (the grid is square, so the
# 2-D halving `52x52 -> 26x26` reads on the flattened axis as `S -> ceil(S /
# stride^2)`). The rows are an image's own patches in the encode phase
# (`patchify`), labelled `s_ctx` here as this op's local name for its row
# axis. Step-3.7-Flash's `understand_projector` runs two of these to compress
# the ViT grid before the text model reads it.
def _conv2d_weight_elements(d):
    shape = resolve_shape_template(
        ("C_out", "C_in", "kernel_size", "kernel_size"), d)
    if not shape:
        return 0
    n = 1
    for extent in shape:
        n = n * extent
    return n


def _conv2d_out_seq(d):
    # ceil(s_ctx / stride^2) on the INPUT rows — the output length, not a
    # bound symbol. `d["s_ctx"]` is this op's row axis as it arrives (already
    # downsampled by an earlier conv), so chaining two convs divides twice.
    return _dim.ceil_div(d["s_ctx"], d["stride"] ** 2)


register_op(OpSpec("conv2d",
    tp_axis_placement=True,
    # `C_in` is the contraction (full, non-depthwise), and `s_ctx` carries the
    # halo a strided kernel reads across window boundaries.
    couples_on=("s_ctx", "C_in"),
    # FLOPs price the OUTPUT tokens `ceil(s_ctx / stride^2)`, each contracting
    # `C_in * kernel^2` inputs into every `C_out` channel (2 per MAC); the bias
    # adds one per output element. Reads the input `s_ctx` + `ceil_div` rather
    # than a separate output dim, so a chained conv divides stagewise.
    flops_fn=lambda d: 2 * d["B"] * _conv2d_out_seq(d) * _conv2d_weight_elements(d)
                       + d["B"] * _conv2d_out_seq(d) * d["C_out"] * _flag(d, "bias"),
    params_fn=lambda d: _conv2d_weight_elements(d)
                        + d["C_out"] * _flag(d, "bias"),
    # Feature-last [B, S, C]. Both the sequence AND the channel change; the
    # output sequence label is deliberately NOT `s_ctx` (a shared label would
    # claim same-index-same-extent and report the input length) — its value
    # comes from the DimMap's `ceil_div`, not a bound symbol.
    shape_in=("B", "s_ctx", "C_in"),
    shape_out=("B", "s_ctx_ds", "C_out"),
    shape_kind="explicit",
    formula="y[b,o,p] = Σ_{i,ky,kx} w[o,i,ky,kx] · x[b,i,stride·p + (ky,kx)]",
    description="2D convolution over a flattened square patch grid; a strided kernel "
                "shrinks the sequence ceil(s_ctx / stride^2). Set C_in / C_out / "
                "kernel_size / stride; owns its (C_out,C_in,k,k) weight and (C_out) bias.",
    weight_shape=("C_out", "C_in", "kernel_size", "kernel_size"),
    tensors=(
        TensorDecl(".weight", ("C_out", "C_in", "kernel_size", "kernel_size")),
        TensorDecl(".bias", ("C_out",), when="bias", kind="vector", axis_refs=(0,)),
    ),
    required_dims=("C_in", "C_out", "kernel_size", "stride"),
))


# Gated linear-attention / delta-rule recurrence core (parameter-free).
# Captures the FLOPs of the gated delta update (q-k-v state matmul +
# scalar gating) used in DeltaNet / GLA / Mamba-2-style linear attention.
# No params of its own — projection params live in upstream `linear` nodes.
#
# FLOPs, from the serving recurrent kernel's own arithmetic (vLLM v0.30.0
# third_party/flash_linear_attention/ops/fused_recurrent.py:125-148, the GDN
# step; models/glm5next/nvidia/ops/third_party/kda/fused_recurrent.py:160-192,
# the KDA step — same body; sglang ships the same FLA kernel). Per token and
# value head, over the [d_lin_v_head, d_lin_k_head] state h:
#   q *= scale                         d_k                      (:162)
#   h *= exp(g)  (GDN g per head,      d_k·d_v   one mul/elem   (:166/:178)
#                 KDA g per key)
#   v -= Σ_k h·k                       2·d_k·d_v readout + d_v  (:180)
#   v *= β                             d_v                      (:190)
#   h += v ⊗ k                         2·d_k·d_v rank-1 write   (:192)
#   o  = Σ_k h·q                       2·d_k·d_v                (FLA :148)
# = 7·d_k·d_v + d_k + 2·d_v per head per token. The gate's own exp (one per
# head, or d_k for KDA) is a transcendental, not counted, as softmax's exp
# is not elsewhere.
#
# This is the recurrent (decode / speculative) form. Prefill runs FLA's
# chunked form (`chunk_gated_delta_rule`, ops/chunk.py with chunk_delta_h /
# chunk_o / chunk_scaled_dot_kkt, chunk C = 64), whose count per token is
# about 6·d_k·d_v + 2·C·(3·d_k + 2·d_v) + C²/3: the inter-chunk state
# update and readout (6·d_k·d_v) plus intra-chunk C×C products (K·Kᵀ, the
# WY transforms, Q·Kᵀ and its product with V) and the triangular solve.
# At d_k = d_v = 128 that is ~181K against the recurrent 115K. The op has no
# prefill/decode form of its own and none of its components declares a
# `decode_variant`, so the recurrent count stands for both phases; the
# chunked prefill is undercounted by ~1.6x at d = 128 (an open item, not a
# per-model case).
#
# The formal parameters are the linear-attention family's own symbols, the
# way `mamba2_scan` takes `n_ssm_heads` / `d_ssm_head`, not generic
# head/width labels another dimension family also binds: a model carrying
# both a vision tower and a linear-attention block (Kimi-K3) would otherwise
# inherit the wrong width from the enclosing frame wherever a call site
# leaves out its explicit `dims:` mapping.
register_op(OpSpec("gated_delta_rule",
    # g decays the state, beta scales how strongly each token writes into it.
    input_roles=("q", "k", "v", "decay", "write"),
    # A recurrence: the state at token t is a function of every token before
    # it. `d_lin_k_head` is contracted into the state matrix.
    couples_on=("s_ctx", "d_lin_k_head"),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["n_lin_v_heads"] * (
        7 * d["d_lin_k_head"] * d["d_lin_v_head"]
        + d["d_lin_k_head"] + 2 * d["d_lin_v_head"]),
    params_fn=_zero,
    # HEAD-FIRST (Q arrives transposed, heads before seq), and rank 4 for the
    # (KDA) decay gate: on Kimi-K3's resolved shapes inputs 0-3 are
    # `(B, n_lin_v_heads, s_ctx, d)`; only the beta gate stays seq-first.
    #
    # Q and K arrive at the KEY head count, which is smaller than the value
    # head count whenever the model groups them (Qwen3-Next: 16 vs 32). The
    # broadcast to value heads happens INSIDE the kernel, not in a tensor
    # before it: vLLM views q/k as `key_dim // tp_size / head_k_dim` heads
    # (`mamba/gdn/qwen_gdn_linear_attn.py`, the `q_contig.view(1, seq_len, -1,
    # head_k_dim)` split) and sglang passes `num_q_heads = num_k_heads =
    # self.num_k_heads // attn_tp_size` (`models/qwen3_next.py`). Neither
    # materialises a repeat. HF's eager reference does
    # (`repeat_interleave(num_v_heads // num_k_heads, dim=2)`), and modelling
    # THAT here would charge every serving deployment a 2x q/k read the
    # kernel it actually runs does not perform.
    #
    # V is value-wide (d_lin_v_head, the output's width). The decay is
    # key-wide: KDA decays each key channel of the state, [B, n_v, s, d_k];
    # GDN's decay is one scalar per head, [B, s, n_v], so the rank-4 template
    # does not describe it and that input does not vote in bind_labels.
    shape_in=(("B", "n_lin_k_heads", "s_ctx", "d_lin_k_head"),
              ("B", "n_lin_k_heads", "s_ctx", "d_lin_k_head"),
              ("B", "n_lin_v_heads", "s_ctx", "d_lin_v_head"),
              ("B", "n_lin_v_heads", "s_ctx", "d_lin_k_head"),
              ("B", "s_ctx", "n_lin_v_heads")),
    # The output is V-shaped, head-first like the inputs: it equals V's shape
    # (shape_in[2]), not Q's, since n_lin_k_heads can be smaller than
    # n_lin_v_heads (Gated DeltaNet).
    shape_out=("B", "n_lin_v_heads", "s_ctx", "d_lin_v_head"),
    shape_kind="explicit",
    # The query scale d_lin_k_head^-0.5 belongs to the op, as 1/sqrt(d_head)
    # belongs to `attention_score`: every kernel takes it as an argument and
    # applies it in-register (FLA `chunk_fwd_o` multiplies the output block
    # by `scale`, `fused_recurrent` multiplies `b_q`), and the default is
    # `k.shape[-1] ** -0.5` in both vLLM's and sglang's FLA `chunk.py`. The q/k
    # L2 normalization is NOT part of the op: it is an `l2norm` node upstream.
    formula="S_t = g_t·S_{t-1} + β_t·K_tᵀV_t;  O_t = (Q_t / sqrt(d_lin_k_head))·S_t",
    description="Gated delta-rule linear recurrence. State [n_lin_v_heads, d_lin_k_head, d_lin_v_head] per layer is cache-not-param, so zero params.",
    required_dims=("d_lin_k_head", "d_lin_v_head",
                   "n_lin_k_heads", "n_lin_v_heads"),
    carries_recurrent_state=True,
))

# Mamba-2 selective state-space scan (`mamba_chunk_scan_combined` /
# `selective_state_update` kernel — Nemotron-H, Zamba2, IBM Granite-H,
# the state-spaces/mamba reference). Zero-param recurrence: per-head
# learnable vectors (A_log, dt_bias, D) are siblings of this op, declared
# separately as `parameter_vector` nodes — same pattern gated_delta_rule
# uses for its A_log / dt_bias siblings inside `gated_delta_net_attention`.
#
# Inputs (positional, mirror of `mamba_chunk_scan_combined` call site):
#   hidden_states   [B, s_ctx, n_ssm_heads, d_ssm_head]   — gated MLP path post conv+silu, reshaped to (heads, head_dim)
#   dt              [B, s_ctx, n_ssm_heads]               — per-head time-step (softplus applied inside kernel)
#   A_log           [n_ssm_heads]                          — log of S4D real init, per-head decay rate
#   B_in            [B, s_ctx, n_ssm_groups, d_ssm_state] — selective input projection grouped over n_ssm_groups
#   C_in            [B, s_ctx, n_ssm_groups, d_ssm_state] — selective output projection grouped over n_ssm_groups
#   D               [n_ssm_heads]                          — per-head skip-connection gain
#   dt_bias         [n_ssm_heads]                          — per-head time-step bias
#
# Output: scan_output [B, s_ctx, n_ssm_heads, d_ssm_head] — same shape as hidden_states.
#
# FLOPs (per token, per head): the chunk-scan kernel does, conceptually,
#   (1) state update    S_t ← exp(-α_t·A) · S_{t-1} + B_t ⊗ x_t  →  2·d_ssm_head·d_ssm_state
#   (2) output read     y_t = (C_t · S_t) + D⊙x_t                →  2·d_ssm_head·d_ssm_state
# Summed over (B, s_ctx, n_ssm_heads):
# 4·B·s_ctx·n_ssm_heads·d_ssm_head·d_ssm_state for the matmul terms
# plus the D⊙x skip: B·s_ctx·n_ssm_heads·d_ssm_head. Mirrors how
# `gated_delta_rule` counts
# `2·B·s_ctx·n_lin_v_heads·d_lin_k_head·d_lin_v_head +
# B·s_ctx·n_lin_v_heads` for its delta recurrence — same shape of
# approximation (chunk-amortized matmul + small per-token additive).
register_op(OpSpec("mamba2_scan",
    # Same shape of argument as `gated_delta_rule`. `n_ssm_heads` and
    # `n_ssm_groups` are absent for the reason `attention_score`'s head axes
    # are: tensor parallelism cuts them together.
    couples_on=("s_ctx", "d_ssm_state"),
    # The state-space dual of linear attention: C reads the state the way a
    # query does, B writes it the way a key does, x is the value, dt decays
    # it. A_log, D and dt_bias are per-head parameters with no upstream
    # branch of their own.
    input_roles=("v", "decay", None, "k", "q", None, None),
    flops_fn=lambda d: 4 * d["B"] * d["s_ctx"] * d["n_ssm_heads"] * d["d_ssm_head"] * d["d_ssm_state"]
                     + d["B"] * d["s_ctx"] * d["n_ssm_heads"] * d["d_ssm_head"],
    params_fn=_zero,
    shape_in=(("B", "s_ctx", "n_ssm_heads", "d_ssm_head"), ("B", "s_ctx", "n_ssm_heads"), ("n_ssm_heads",),
              ("B", "s_ctx", "n_ssm_groups", "d_ssm_state"), ("B", "s_ctx", "n_ssm_groups", "d_ssm_state"),
              ("n_ssm_heads",), ("n_ssm_heads",)),
    shape_out=("B", "s_ctx", "n_ssm_heads", "d_ssm_head"),
    shape_kind="explicit",
    formula="y_t = D⊙x_t + Σ_τ≤t C_t·(exp(-Σα_{τ→t}·A)·B_τ⊗x_τ)",
    description="Mamba-2 selective state-space scan (mamba_chunk_scan_combined kernel). Zero-param recurrence: per-layer state is [n_ssm_heads, d_ssm_head, d_ssm_state], constant in s_ctx, not part of growing KV cache. Per-head learnables (A_log, dt_bias, D) are sibling parameter_vector nodes in the surrounding component.",
    required_dims=("n_ssm_heads", "d_ssm_head", "d_ssm_state", "n_ssm_groups"),
    carries_recurrent_state=True,
))

register_op(OpSpec("headwise_gate",
    # Pointwise per head.
    couples_on=(),
    flops_fn=lambda d: d["B"] * d["s_ctx"] * d["n_heads"] * d["d_head"],
    params_fn=_zero,
    shape_in=(("B", "n_heads", "s_ctx", "d_head"), ("B", "s_ctx", "n_heads")),
    shape_out=("B", "n_heads", "s_ctx", "d_head"),
    shape_kind="preserve",  # output = first input (attn) shape
    formula="y[b,n,s,d] = gate[b,s,n] · attn[b,n,s,d]",
    description="Per-head scalar gate broadcast over attention output (g_proj produces one gate per head; Step-3.5 / Step-3.7).",
    required_dims=("d_head", "n_heads"),
))

# DSA stage 1b — dense indexer logits. One score per (query, cached_pos)
# pair, computed by a single FP8 paged-MQA-logits kernel (vLLM
# `fp8_paged_mqa_logits` / sglang `deep_gemm.fp8_mqa_logits`). Output is
# the DENSE (B, s_query, s_kv) logits tensor — the topk selection that follows
# consumes it from HBM, so this tensor is real and round-trips through
# HBM. Splitting it out from the topk lets the cost model see the FLOPs
# correctly attached to logits compute (this kernel, not topk) and the
# downstream `dsa_topk_mask` output stay n_topk_keys-bounded sparse.
#
# FLOPs per (b, query, cached position, indexer head): 2·d_indexer_head for
# the Q·K dot product, then 1 for the ReLU, 1 for the multiply by the head
# weight and 1 for the add into the sum over n_indexer_heads — 2·d_indexer_head
# + 3 in all. vLLM `fp8_mqa_logits_torch` (rocm_aiter_mla_sparse.py):
# `(score.relu() * weights...).sum(dim=0)`; the Triton kernel
# (triton_fp8_mqa_logits.py) applies the ReLU as `tl.maximum(scores, 0.0)`.
# The ReLU costs 1 as the `relu` activation op does. Each term is per live
# (query, key) pair of the indexer matrix (`s_pairs`).
register_op(OpSpec("indexer_logits",
    dsa=True,
    label="Indexer Logits",
    # An attention score with a head-weighted sum folded in, so the head axis
    # contracts here where in plain attention it would not.
    couples_on=("s_query", "s_kv", "n_indexer_heads", "d_indexer_head"),
    flops_fn=lambda d: d["B"] * _pairs(d) * d["n_indexer_heads"]
                      * (2 * d["d_indexer_head"] + 3),
    params_fn=_zero,
    # The lightning indexer scores the current query against ALL s_kv cached
    # indexer keys (dense O(s_kv) scan) before the top-k selector narrows to
    # n_topk_keys. So the indexer-key cache is READ at full s_kv every decode
    # step and must be STORED in full — the sparse top-k benefit applies only
    # to the main MLA latent (handled by `dsa_attention_score`'s
    # Min(s_kv,n_topk_keys)), NOT to the
    # indexer key. `kv_block_anchor=False`: this carries the indexer-key
    # node's own cardinality (via per-node consumer lookup) but must not
    # become the layer's headline read expr, which stays the top-k MLA path.
    kv_positions_per_query_fn=_s_kv,
    kv_storage_positions_per_query_fn=_s_kv,
    kv_block_anchor=False,
    # The query per head, as the heads were split (`[B, H, s, d]`): the key
    # and every query head are contracted over the same `d_indexer_head`,
    # which is what tells a reader of the declaration which operand is
    # multiplied with the cached key (`cache._paired_operands`) — the one a
    # stored rotation or record is matched on.
    # In the order of `attention_score` — query, keys — and of the fused
    # span a spelled-out score becomes (the catalog's DeepSeek-V4 layouts),
    # then the head weights.
    shape_in=(("B", "n_indexer_heads", "s_query", "d_indexer_head"), ("B", "s_kv", "d_indexer_head"), ("B", "s_query", "n_indexer_heads")),
    shape_out=("B", "s_query", "s_kv"),
    shape_kind="explicit",
    formula="logits[b,s_query,s_kv] = Σ_in weights[b,s_query,in] · ReLU(Q_idx[b,s_query,in,:] · K_idx[b,s_kv,:])",
    description="DSA stage-1b indexer dense logits. Single FP8 paged-MQA-logits kernel in vLLM/sglang; output is the dense (B,s_query,s_kv) fp32 logits the top-k selector reads from HBM.",
    required_dims=("d_indexer_head", "n_indexer_heads"),
    # The logits land in HBM as fp32, whatever the activation dtype: DeepGEMM
    # `fp8_fp4_mqa_logits` / `fp8_paged_mqa_logits` return float32 (vLLM
    # utils/deep_gemm.py:522-552 "Logits tensor of shape [M, N], dtype
    # torch.float32", called at model_executor/layers/sparse_attn_indexer.py
    # :584 / :714; sglang srt/layers/attention/dsa/dsa_indexer.py:1041), and
    # the unweighted QSA form allocates its logits float32 too (vLLM
    # models/qwen4_exp/nvidia/ops/qsa_indexer.py:397-398, :538; sglang
    # srt/layers/attention/qsa/mqa.py:303). A fused `indexer_logits` carries
    # this registration, so the fused launch writes the same width.
    output_dtype="F32",
))

# DSA stage 2 — sparse top-k selection over the dense indexer logits.
# Input: logits (B, s_query, s_kv). Output:
# topk_indices (B, s_query, n_topk_keys) int32 — the n_topk_keys best cached
# positions per query. This is where DSA's first KV bandwidth advantage lives
# in the cost model: the output is n_topk_keys-bounded, not s_kv-bounded — a
# dense (B, n_heads, s_query, s_kv) mask would be 16×–128× larger at long
# context (s_kv=128K, n_topk_keys=2048 → 8 KB/query vs 128 KB/query). The
# real kernel emits int32 indices, NOT a Boolean s_kv-wide mask.
#
# FLOPs: O(B · s_query · s_kv) — radix top-k is bandwidth-bound, not compute-
# bound, and vLLM's `persistent_topk` / sglang's `topk_transform` make a
# single pass over the logits row to extract the n_topk_keys best.
register_op(OpSpec("dsa_topk_mask",
    dsa=True,
    label="DSA Top-K Indices",
    # Top-k is over all of `s_kv`.
    couples_on=("s_kv",),
    flops_fn=lambda d: d["B"] * _pairs(d),
    params_fn=_zero,
    # The output is a SELECTION, not an activation: int32 indices of the K
    # cached positions each query attends to. It is shaped like a small
    # activation — only the kind stops it from being wired as one.
    output_kind="mask",
    shape_in=(("B", "s_query", "s_kv"),),
    shape_out=("B", "s_query", "k"),
    shape_kind="axis_select",
    out_key="k",
    axis_locked=-1,
    formula="topk_indices = TopK_n_topk_keys(logits)  along s_kv axis",
    description="DSA stage-2 top-k selection: emits int32 indices of the n_topk_keys cached positions each query attends to. Output is sparse (B, s_query, n_topk_keys) int32 — the bandwidth saving vs a dense (B, s_query, s_kv) mask is the headline DSA win at long context.",
    required_dims=("k",),
    output_dtype="I32",
))

# --- DSA (DeepSeek Sparse Attention) variants ---
#
# Key difference vs dense attention: the query whose causal context is `c`
# keys attends `min(c, k)` of them — the indexer's top-k over what it may see,
# padded with -1 below k (vLLM `csrc/libtorch_stable/sampler.cu`
# `top_k_per_row`; sglang `dsa/utils.py` `compute_dsa_seqlens = seq_lens.clamp(
# max=index_topk)`). The matrix each op sizes is `s_query × Min(s_kv, k)`, and
# the live pairs in it are that per-query count summed over the step's queries
# (`kv_pairs_per_step_fn`): the walk rebinds the node's `s_pairs` to them. They
# are not the phase's dense pairs — a context past k leaves almost no
# triangle to halve. 2× is the explicit matmul coefficient.

def _topk_pairs(d: Dict[str, Any]) -> Any:
    """Live pairs of a top-`k` selection over the causal context: the query
    whose context is `c` keys attends `min(c, k)` of them (`_selected_pairs`)."""
    return _selected_pairs(d, cap=d["k"])


register_op(OpSpec("dsa_attention_score",
    input_roles=("q", "k", "select"),
    dsa=True,
    label="Q @ K^T (sparse)",
    attention_kernel="score",
    couples_on=("s_query", "s_kv", "d_head"),
    # Rank is a contract here: a rank-3 query is a mis-wired edge.
    input_rank_is_exact=True,
    # sp.Min returns int when fed ints, SymPy Min Expr when fed Symbols
    # — avoids Python's `min()` which compares via __lt__ (raises on Expr).
    flops_fn=lambda d: 2 * d["B"] * d["n_heads"] * _pairs(d) * d["d_head"],
    # DSA sparse-attention: each query attends to at most n_topk_keys (top-k)
    # cached positions. Source of truth for DSA's KV-read benefit.
    kv_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["k"]),
    kv_pairs_per_step_fn=_topk_pairs,
    # DSA storage: the lightning indexer needs to score *every* cached
    # position before picking the top-k, so the cache itself still holds
    # the full s_kv history. Storage benefit is zero; only read traffic
    # shrinks. This is the storage-vs-read asymmetry that distinguishes
    # DSA from SWA.
    kv_storage_positions_per_query_fn=_s_kv,
    params_fn=_zero,
    shape_in=(("B", "n_heads", "s_query", "d_head"), ("B", "n_kv_heads", "s_kv", "d_head")),
    shape_out=("B", "n_heads", "s_query", "s_kv"),
    # Q · K^T has the same implicit key transpose as dense attention_score.
    # The sparse mask changes the number of key positions read, not the key
    # tensor's axis order.
    shape_kind="explicit",
    formula="S = (Q · Kᵀ / √d_head)[mask]",
    required_dims=("d_head", "n_heads", "k"),
    # The third input is the top-k index list the mask gates on — a
    # selection, not an activation (the component wires `dsa_topk_mask`'s
    # output here).
    input_kinds=("activation", "activation", "mask"),
))

register_op(OpSpec("dsa_softmax",
    dsa=True,
    label="Softmax (sparse)",
    couples_on=("s_kv",),
    flops_fn=lambda d: 5 * d["B"] * d["n_heads"] * _pairs(d),
    # Same s_kv-effective handling as the score op: the softmax operates
    # on the sparse score matrix whose last axis is Min(s_kv, n_topk_keys), so the
    # engine must substitute s_kv → Min(s_kv, n_topk_keys) when sizing shape_in /
    # shape_out. Without this, the upstream (score) gives us a correctly
    # sparse-shaped tensor and `_apply_template_axis_override` then
    # overwrites it back to the full s_kv from global dims.
    kv_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["k"]),
    kv_pairs_per_step_fn=_topk_pairs,
    params_fn=_zero,
    shape_in=("B", "n_heads", "s_query", "s_kv"),
    shape_out=("B", "n_heads", "s_query", "s_kv"),
    shape_kind="preserve",
    formula="A = softmax(S) (over n_topk_keys-sparse keys)",
    required_dims=("n_heads", "k"),
))

register_op(OpSpec("dsa_attention_apply",
    input_roles=("probs", "v"),
    dsa=True,
    label="Attn @ V (sparse)",
    attention_kernel="apply",
    couples_on=("s_query", "s_kv"),
    # Rank is a contract here: a rank-3 query is a mis-wired edge.
    input_rank_is_exact=True,
    flops_fn=lambda d: 2 * d["B"] * d["n_heads"] * _pairs(d) * d["d_head"],
    # Parallel to dsa_attention_score: the apply step contracts A · V
    # over the same n_topk_keys-sparse positions the score step picked, so the V
    # cache's s_kv axis is read at Min(s_kv, n_topk_keys) per query. Declaring the fn
    # here lets the engine substitute s_kv → Min(s_kv, n_topk_keys) in shape_in so
    # per-leaf access_bytes / shape rendering reflect the sparse read.
    kv_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["k"]),
    kv_pairs_per_step_fn=_topk_pairs,
    kv_storage_positions_per_query_fn=_s_kv,
    params_fn=_zero,
    shape_in=(("B", "n_heads", "s_query", "s_kv"), ("B", "n_kv_heads", "s_kv", "d_head")),
    shape_out=("B", "n_heads", "s_query", "d_head"),
    shape_kind="contract",
    formula="O = A · V (n_topk_keys-sparse)",
    required_dims=("d_head", "n_heads", "k"),
))

# --- SWA (Sliding Window Attention) variants ---
#
# Key difference vs dense attention: each query attends to at most W keys
# (the window size). Which (query, key) pairs are live is the MASK's fact,
# and the attention declares it (`is_causal`, `is_chunked`;
# `ops_base.mask_flags`), so the count is written here once, from the mask,
# rather than as `s_query · Min(s_kv, W)` scaled by the phase's dense causal
# factor — that factor halves a full lower triangle, and a window that is
# already narrow is not one.
#
#   causal band       query at absolute position p sees min(p + 1, W) keys
#                     (vLLM flash_attn.py: window (W − 1, 0)). Over the
#                     step's queries — the last `s_query` of the L keys the
#                     step reads — that is
#                       [L(L+1) − a(a+1) − m1(m1+1) + m0(m0+1)] / 2,
#                     a = L − s_query, m1 = Max(L − W, 0), m0 = Max(a − W, 0).
#                     L is the K operand's rows (the step's union); the sum
#                     is the same over the union and over the whole context,
#                     because a query past the union's first W − 1 positions
#                     sees exactly W keys either way. W ≥ L gives the dense
#                     causal triangle; s_query = 1 gives Min(L, W).
#   bidirectional     a symmetric band of reach w = (W − 1)/2 — vLLM builds
#   band              a non-causal window symmetric (`_maybe_symmetrize_
#                     window`; MiMo `sliding_window_q = sliding_window_k`).
#                     Over S = s_query positions (an encoder: queries are
#                     the keys): S(2m + 1) − m(m + 1), m = Min(w, S − 1), the
#                     half-windows clipped at both ends of the sequence.
#   chunks            keys cut into fixed chunks of W, a query sees its own
#                     chunk (`is_chunked`; MiMo audio's groups of 4).
#                     Bidirectional: ⌊S/W⌋·W² + r²; causal: the triangles,
#                     [⌊S/W⌋·W(W+1) + r(r+1)] / 2; r = S − W⌊S/W⌋.
#
# The flags select a law by arithmetic, not by a branch, so the symbolic
# binding is the same walk.


def _window_pairs(d: Dict[str, Any]) -> Any:
    """Live (query, key) pairs of one head of a windowed attention."""
    half = Rational(1, 2)
    q, keys, w_size = _s_query(d), _s_kv(d), d["s_window"]
    causal, chunked = mask_flags(d)
    a = keys - q
    m1 = _dim.maximum(keys - w_size, 0)
    m0 = _dim.maximum(a - w_size, 0)
    causal_band = (keys * (keys + 1) - a * (a + 1)
                   - m1 * (m1 + 1) + m0 * (m0 + 1)) * half
    reach = _dim.minimum((w_size - 1) * half, q - 1)
    bidirectional_band = q * (2 * reach + 1) - reach * (reach + 1)
    n = _dim.floor_div(q, w_size)
    r = q - n * w_size
    chunks_bidirectional = n * w_size * w_size + r * r
    chunks_causal = (n * w_size * (w_size + 1) + r * (r + 1)) * half
    return (causal * ((1 - chunked) * causal_band + chunked * chunks_causal)
            + (1 - causal) * ((1 - chunked) * bidirectional_band
                              + chunked * chunks_bidirectional))


def _window_key0_queries(d: Dict[str, Any]) -> Any:
    """Queries whose window contains key 0 — where a key-0 logit bias lands.
    A band of left reach l: the first l + 1 positions (l = w bidirectional,
    W − 1 causal; for a causal step that does not start at position 0 this
    is an upper bound); chunks: the first chunk."""
    half = Rational(1, 2)
    q, w_size = _s_query(d), d["s_window"]
    causal, chunked = mask_flags(d)
    left = (w_size - 1) * causal + (w_size - 1) * half * (1 - causal)
    band = _dim.minimum(q, left + 1)
    return (1 - chunked) * band + chunked * _dim.minimum(q, w_size)


def _window_step_keys(d: Dict[str, Any]) -> Any:
    """Distinct keys a step's `s_query` queries read through a window of W:
    the union reaches W − 1 keys past the step, capped by the context."""
    return _dim.minimum(_s_kv(d), _s_query(d) + d["s_window"] - 1)


register_op(OpSpec("swa_attention_score",
    input_roles=("q", "k"),
    windowed=True,
    attention_kernel="score",
    couples_on=("s_query", "s_kv", "d_head"),
    # Rank is a contract here: a rank-3 query is a mis-wired edge.
    input_rank_is_exact=True,
    cost_pairs_fn=_window_pairs,
    flops_fn=lambda d: 2 * d["B"] * d["n_heads"] * _s_pairs(d) * d["d_head"],
    # Sliding-window attention: each query attends to at most s_window
    # consecutive cached positions.
    kv_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["s_window"]),
    kv_positions_per_step_fn=_window_step_keys,
    # SWA storage: positions outside the window can be evicted (Mistral
    # / Llama-SWA / Gemma — the runtime drops them as the window slides
    # forward), so the per-request KV footprint caps at W independent of
    # context length. This is the storage benefit the FLOPs side already
    # reflects via `Min(s_kv, s_window)` in `flops_fn`.
    kv_storage_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["s_window"]),
    params_fn=_zero,
    # Spec convention (matches dense `attention_score`): both Q and K
    # come in as `(B, n_heads, s_query, d_head)` /
    # `(B, n_kv_heads, s_kv, d_head)` — caller does NOT
    # pre-transpose K. The op semantics treat K's last dim as the
    # contracted dim (d_head) and K's second-to-last as s_kv.
    shape_in=(("B", "n_heads", "s_query", "d_head"), ("B", "n_kv_heads", "s_kv", "d_head")),
    # The score matrix's key axis is `s_attend` — the keys ONE query attends,
    # `Min(s_kv, s_window)` — not `s_kv`, the rows of the K operand. A step's
    # queries together read more keys than any one of them attends, so the
    # two are different extents and carry different labels.
    shape_out=("B", "n_heads", "s_query", "s_attend"),
    shape_kind="explicit",
    formula="A = Q · Kᵀ / sqrt(d_head)  (masked to s_window-token window)",
    description="Sliding-window attention scores: like attention_score but each query attends to at most s_window keys. FLOPs scale with s_pairs, the live pairs of the window band (`_window_pairs`).",
    # Mirrors the dense family exactly — including the fact that its
    # sequence-parallel mode has NO cost formula yet. Copying the states and
    # dropping `internal_comm` is how a ring schedule comes to be priced at
    # zero, which is the thing `ModeNotYetCosted` exists to prevent.
    #
    # A sliding window changes WHICH keys a query attends, not how heads or
    # queries are distributed across ranks, and `required_dims` says so by
    # differing from the dense op's only by `s_window`.

    required_dims=("d_head", "n_heads", "s_window"),
))

register_op(OpSpec("swa_softmax",
    windowed=True,
    couples_on=("s_attend",),
    cost_pairs_fn=_window_pairs,
    flops_fn=lambda d: 5 * d["B"] * d["n_heads"] * _s_pairs(d),
    # Same s_kv-effective handling as the score op: softmax runs over the
    # windowed score matrix whose last axis is Min(s_kv, s_window). See OpSpec
    # `kv_positions_per_query_fn` docstring for why score, softmax, and
    # apply must all declare this — the template-axis override would
    # otherwise re-inflate the windowed shape back to full s_kv.
    kv_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["s_window"]),
    kv_positions_per_step_fn=_window_step_keys,
    params_fn=_zero,
    shape_in=("B", "n_heads", "s_query", "s_attend"),
    shape_out=("B", "n_heads", "s_query", "s_attend"),
    shape_kind="preserve",
    formula="p[i] = exp(x[i]) / Σⱼ exp(x[j])  (j ∈ W-token window)",
    description="Row-wise softmax over the sliding-window-masked attention scores.",
    # Mirrors the dense family exactly — including the fact that its
    # sequence-parallel mode has NO cost formula yet. Copying the states and
    # dropping `internal_comm` is how a ring schedule comes to be priced at
    # zero, which is the thing `ModeNotYetCosted` exists to prevent.
    #
    # A sliding window changes WHICH keys a query attends, not how heads or
    # queries are distributed across ranks, and `required_dims` says so by
    # differing from the dense op's only by `s_window`.

    required_dims=("n_heads", "s_window"),
))

# Sink-aware softmax for SWA. Sink token bias (`attn_sink`, one scalar
# per head) participates in the denominator: each query's softmax
# normalizes over `W` real keys + 1 virtual sink. This prevents the
# softmax from saturating when all in-window keys are uninformative —
# a stabilization trick used by GPT-OSS, DSv4-Pro CSA/HCA/SWA-only,
# and Kimi-K2. The dense (non-SWA) analogue is `softmax_with_sink`.
#
# FLOPs delta vs `swa_softmax`: +B·n_heads·s_query for exp(sink) + add + divide
# (≪ the windowed softmax cost — softmax is
# 5·B·n_heads·s_query·Min(s_kv,s_window); sink
# is rank-0 along the windowed axis so it adds a flat per-token cost).
# Total: 5·B·n_heads·s_query·Min(s_kv,s_window) + 3·B·n_heads·s_query.
register_op(OpSpec("swa_softmax_with_sink",
    sink=True,
    windowed=True,
    couples_on=("s_attend",),
    cost_pairs_fn=_window_pairs,
    flops_fn=lambda d: (5 * d["B"] * d["n_heads"] * _s_pairs(d)
                       + 3 * d["B"] * d["n_heads"] * _s_query(d)),
    # Same windowed s_kv as swa_softmax — see that op for rationale.
    kv_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["s_window"]),
    kv_positions_per_step_fn=_window_step_keys,
    params_fn=_zero,
    # Two inputs: (B, n_heads, s_query, s_kv) scores + (n_heads,) per-head sink bias.
    shape_in=(("B", "n_heads", "s_query", "s_attend"), ("n_heads",)),
    shape_out=("B", "n_heads", "s_query", "s_attend"),
    shape_kind="explicit",
    formula="p[i] = exp(x[i]) / (Σⱼ exp(x[j]) + exp(sink))",
    description="Sliding-window softmax with a per-head sink-token bias absorbed into the denominator. Stabilizes attention when the window has no informative keys.",
    required_dims=("n_heads", "s_window"),
))

# Sliding-window softmax whose per-head term is a LOGIT BIAS on key position
# 0, not a virtual denominator column: A'[h, i, j] = A[h, i, j] + b[h]·[j = 0],
# then softmax over the real window keys. MiMo-V2.5/V2.6 vision blocks
# (vLLM mimo_v2_omni.py MiMoVisionAttention._forward_window_attn:
# `context_attention_fwd(..., sinks=sinks, sinks_bias_key0=True)`; the
# kernel, v1/attention/ops/triton_prefill_attention.py, does
# `qk = tl.where(mask & (pos_k == 0), qk + sink, qk)` and starts the running
# max/denominator empty, so no extra column enters the normalization).
# Not `sink`: nothing joins the denominator, so the span carries no `s_aux`.
# FLOPs: the windowed softmax + one add per (query, head) on key 0.
register_op(OpSpec("swa_softmax_key0_bias",
    windowed=True,
    couples_on=("s_attend",),
    cost_pairs_fn=_window_pairs,
    flops_fn=lambda d: (5 * d["B"] * d["n_heads"] * _s_pairs(d)
                       + d["B"] * d["n_heads"] * _window_key0_queries(d)),
    kv_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["s_window"]),
    kv_positions_per_step_fn=_window_step_keys,
    params_fn=_zero,
    # Two inputs: (B, n_heads, s_query, s_kv) scores + (n_heads,) per-head bias.
    shape_in=(("B", "n_heads", "s_query", "s_attend"), ("n_heads",)),
    shape_out=("B", "n_heads", "s_query", "s_attend"),
    shape_kind="explicit",
    formula="p[i, j] = softmax_j(x[i, j] + b·[j = 0])   (j ∈ W-token window)",
    description="Sliding-window softmax with a per-head bias added to the logit of key position 0 (no extra denominator column).",
    required_dims=("n_heads", "s_window"),
))

register_op(OpSpec("swa_attention_apply",
    input_roles=("probs", "v"),
    windowed=True,
    attention_kernel="apply",
    couples_on=("s_query", "s_attend", "s_kv"),
    # Rank is a contract here: a rank-3 query is a mis-wired edge.
    input_rank_is_exact=True,
    cost_pairs_fn=_window_pairs,
    flops_fn=lambda d: 2 * d["B"] * d["n_heads"] * _s_pairs(d) * d["d_head"],
    # Parallel to swa_attention_score: the apply step contracts A · V
    # over the same in-window positions the score step picked, so the V
    # cache's s_kv axis is read at Min(s_kv, s_window) per query. Declaring the fn
    # here lets the engine substitute s_kv → Min(s_kv, s_window) in shape_in so
    # per-leaf access_bytes / shape rendering reflect the windowed read
    # — without this, hybrid backbones (e.g. MiMo-V2.5: 9 GQA + 39 SWA)
    # inflate decode mem_read ~9× at long context because every SWA
    # apply row claims to read the full s_kv cache.
    kv_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["s_window"]),
    kv_positions_per_step_fn=_window_step_keys,
    params_fn=_zero,
    # The probabilities' key axis is `s_attend` (one query's window), V's rows
    # are `s_kv` (every key the step reads): a banded product, not a plain
    # contraction over one shared label, so the output is declared.
    shape_in=(("B", "n_heads", "s_query", "s_attend"), ("B", "n_kv_heads", "s_kv", "d_head")),
    shape_out=("B", "n_heads", "s_query", "d_head"),
    shape_kind="explicit",
    formula="O = A · V  (A masked to s_window-token window)",
    description="Attention-weighted value combine for SWA: softmax(QKᵀ)·V over the in-window keys. FLOPs scale with s_pairs, the live pairs of the window band (`_window_pairs`).",
    # Mirrors the dense family exactly — including the fact that its
    # sequence-parallel mode has NO cost formula yet. Copying the states and
    # dropping `internal_comm` is how a ring schedule comes to be priced at
    # zero, which is the thing `ModeNotYetCosted` exists to prevent.
    #
    # A sliding window changes WHICH keys a query attends, not how heads or
    # queries are distributed across ranks, and `required_dims` says so by
    # differing from the dense op's only by `s_window`.

    required_dims=("d_head", "n_heads", "s_window"),
))

# --- Time-axis KV compressor (DeepSeek-V4 CSA / HCA) ---
#
# Marks the OUTPUT of a token-pooling KV compressor: `compress_ratio`
# consecutive token positions are pooled into ONE cached latent, so the
# cache holds `ceil(s_kv / compress_ratio)` positions instead of `s_kv`. This
# is the storage-side parallel of SWA's window eviction, but the cap
# *grows* with context (`s_kv / m`) instead of saturating at a constant s_window.
#
# Why this op exists: a CSA/HCA layer caches TWO KV nodes with DIFFERENT
# cardinalities — the SWA `kv_concat` (caps at W) and this compressed
# latent (grows at s_kv/m). The KV-positions math lives on the op (next to
# flops_fn) exactly like swa_/dsa_attention_score, so the per-cached-node
# resolver reads each node's own cardinality instead of forcing both onto
# the single attention-score op's window. One op covers both variants:
#   - storage == ceil(s_kv / m) for both: the pending (incomplete) window
#     owns a cache slot the moment its first token arrives — its rows sit in
#     the fp32 ring state until the window completes and the pooled row lands.
#   - read == floor(s_kv / m) complete windows for HCA (no indexer): the
#     pending slot is reserved but not readable (model.py `kv_cache[:, :end_pos
#     // ratio]`, L271-274); the compute stream the compressor emits this step
#     is the same floor.
#   - CSA (sparse indexer picks top-K of the compressed positions):
#     read == Min(floor(s_kv / m), read_topk) (model.py L427); storage still
#     ceil(s_kv / m) (every compressed position is cached so the indexer can
#     score it).
# `read_topk` is a NODE-LOCAL dim bound only by the CSA wrapper (= index_topk);
# HCA has no indexer and does not bind it → read falls back to the full
# readable count floor(s_kv / m). (Not gated on the global `n_topk_keys` dim, which is
# present model-wide and would wrongly cap the no-indexer HCA branch.)
register_op(OpSpec("token_pool_compressor",
    # A marker on the already-pooled latent: the pooling itself (complete-
    # window slice → view_split → softmax over the window → sum_reduce) is
    # explicit upstream, so this node couples nothing and changes no shape.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "dim"),
    shape_out=("B", "s_ctx", "dim"),
    shape_kind="preserve",
    # `ceil_div`, not `ceiling(a / b)`: the length arrives as a bare sympy
    # symbol and the ratio as a tracked dim, and in that order sympy's own
    # `__truediv__` wins the dispatch — it sympifies the dim through
    # `_sympy_`, keeps the expression half and drops the value. The law then
    # reads `⌈s_kv/compress_ratio⌉` with the ratio still free, and the first
    # consumer to want a number gets `Cannot convert symbols to int`. Invisible
    # while the call site spelled the ratio as a literal (both halves say `4`);
    # the moment it names a dim, it is the whole difference.
    #
    # The ceil is a CAPACITY law, not a shape: a request of s_kv tokens owns
    # ⌈s_kv/m⌉ compressed slots because its partial last window will occupy
    # one when it completes. The compute frame above counts only the complete
    # windows of the current step (⌊S/m⌋).
    kv_storage_positions_per_query_fn=lambda d: _dim.ceil_div(
        _s_kv(d), d["compress_ratio"]),
    # A read sees only COMPLETE windows: the source attends
    # `kv_cache[:, :end_pos // ratio]` and its indexer takes
    # `min(index_topk, end_pos // ratio)` (model.py L271-274, L420, L427).
    # The pending window's slot is reserved (storage) but not yet readable.
    kv_positions_per_query_fn=lambda d: (
        _dim.minimum(_dim.floor_div(_s_kv(d), d["compress_ratio"]), d["read_topk"])
        if d.get("read_topk") is not None
        else _dim.floor_div(_s_kv(d), d["compress_ratio"])
    ),
    # The same law per query, summed over the step's queries: a query whose
    # causal context is `c` keys sees the `⌊c/m⌋` blocks that have closed by
    # its own position — all of them for HCA, the indexer's top `read_topk`
    # for CSA (vLLM `deepseek_v4/common/ops/cache_utils.py` `topk_len =
    # min((pos + 1) // COMPRESS_RATIO, TOP_K)`, `deepseek_v4/sparse_mla.py`
    # `num_compressed = (position + 1) // compress_ratio`; DeepSeek
    # `inference/model.py` `get_compress_topk_idxs`). A prefill query early
    # in the sequence sees few blocks, so the step's pairs are not `s_query`
    # times the last query's read.
    kv_pairs_per_step_fn=lambda d: _selected_pairs(
        d, cap=d.get("read_topk"), ratio=d["compress_ratio"]),
    # The law is the pooled stream's own, so a reader of it (an indexer
    # scoring the pooled keys) reads that many positions — and it is not the
    # layer's headline: the main attention score is.
    kv_stream_marker=True,
    kv_block_anchor=False,
    formula="kv_cached = x   (identity; storage ⌈s_kv / m⌉, read ⌊s_kv / m⌋ compressed positions)",
    description="DSv4 compressed-KV marker: identity on the already-pooled "
                "latent that enters the cache. Zero cost, shape-preserving; "
                "it carries the compressed KV cardinality — storage ⌈s_kv/m⌉ "
                "(the pending window's slot is reserved), read ⌊s_kv/m⌋ "
                "complete windows, capped at read_topk for CSA. The pooling "
                "is the explicit graph upstream of it.",
    required_dims=("compress_ratio",),
))

# The window as a property of the TENSOR rather than of whoever scores it.
#
# `swa_attention_score` declares `Min(s_kv, s_window)` because for a plain SWA
# layer the score IS the only reader, and `cache.py::_anchor` hands that law
# back to the windowed `kv_concat` on the way out. That breaks the moment the
# windowed stream is not the whole story: DeepSeek-V4 cats the window against
# a compressed stream and runs ONE dense attention over the union, so the
# score's cardinality is the union's and the window belongs to the tensor the
# runtime actually evicts from. This op states it there — the dual of how
# compression lives on `token_pool_compressor` rather than on its reader.
#
# `kv_block_anchor=False` for the same reason `indexer_logits` sets it: a real
# law for its own tensor that must not become the layer's headline. Both sides
# are declared, so `_law_for` comes back with read and storage already filled
# and no anchor can overwrite either.
register_op(OpSpec("sliding_window_kv",
    windowed=True,
    # A sequence cut splits the window: which ranks hold the last `s_window`
    # positions depends on where the cut fell. The same sentence `slice` says
    # about the axis it narrows, on the axis this one narrows.
    couples_on=("s_in",),
    # As `slice` does: the MQA latent the window is usually taken of, and the
    # head-major form for a GQA stream windowed after its head split.
    shape_in=("B", "s_in", "d"),
    alt_shape_in=(("B", "n_heads", "s_in", "d"),),
    shape_out=("B", "s_window", "d"),
    flops_fn=_zero,
    params_fn=_zero,
    # Nothing is copied: eviction is which slots the cache still holds, not a
    # tensor written somewhere else.
    is_view_op=True,
    # Second-to-last, so the one declaration reads the MQA latent
    # `[B, s_ctx, d_head]` and a head-major `[B, n_heads, s_ctx, d_head]`
    # alike. The window narrows the shape as well as the law — the cat
    # downstream copies the windowed positions, not the whole context.
    shape_kind="axis_select",
    axis_locked=-2,
    out_key="s_window",
    kv_positions_per_query_fn=lambda d: _dim.minimum(_s_kv(d), d["s_window"]),
    kv_storage_positions_per_query_fn=lambda d: _dim.minimum(
        _s_kv(d), d["s_window"]),
    # The window's own mask: a query whose causal context is `c` keys sees
    # the last `min(c, s_window)` of them, itself included (vLLM
    # `sparse_swa.py` `_compute_swa_indices_and_lens_kernel`: `start_pos =
    # max(pos − window_size + 1, 0)`, `end_pos = pos + 1`; DeepSeek
    # `inference/model.py` `get_window_topk_idxs`). The union hands these
    # pairs to the attention that reads it. The law is a CAUSAL window
    # (`min(c, W)`, the query itself included); an attention that reads the
    # window bidirectionally needs the symmetric band law instead
    # (`_window_pairs`), not these pairs.
    kv_pairs_per_step_fn=lambda d: _selected_pairs(d, cap=d["s_window"]),
    kv_stream_marker=True,
    kv_block_anchor=False,
    formula="kv_cached = kv[-s_window:]  (Min(s_kv, s_window) positions)",
    description="Sliding-window KV marker: the windowed view of a KV stream. "
                "Zero-cost; carries the Min(s_kv, s_window) read and storage "
                "cardinality on the tensor that is evicted from.",
    required_dims=("s_window",),
))

# The whole cache as a KV stream, for a reader that is not an attention
# kernel. A score reads its key cache through its own law; an op upstream of a
# score has none, so the cache length it reads would go unpaid. Qwen3.8-Flash-
# Next's QSA indexer re-pools ALL its cached raw keys per block every step
# (modeling_qwen4_exp.py Qwen4ExpTextQSAIndexer.forward: `raw_keys =
# past_key_values.update_indexer(raw_keys, self.layer_idx)`, then
# `num_complete_blocks = local_visible_indices.shape[-1] // compress_ratio` over
# the visible cache): the pool is the cache read. The dual of
# `sliding_window_kv`, whose view is the last `s_window` positions: this one is
# every position, so its read and storage law is `s_kv` — and so is its
# shape: the reader downstream SPANS the cache, which is how `traffic` knows
# the edge reads the cache rather than this step's rows.
#
# Bounded by `read_topk` where its reader takes only a selection of those
# positions: DeepSeek-V4.1's index sources score only the candidate source's
# `n_topk_blocks · s_sparse_block` positions (sglang `v41_indexer/
# sparse_table.py` `consume_decode`).
# `read_topk` is node-local, as on `token_pool_compressor`: a whole-cache
# view does not bind it. The positions it views are those of the stream that
# arrives (a pooled stream's ⌊s_kv/m⌋, adopted as this node's `s_kv`); the
# step's live pairs are counted over the queries' token contexts, so a
# bounded view states them through the stream's pooling ratio, the
# selection-over-a-pooled-stream count `token_pool_compressor` states.
def _view_read(d: Dict[str, Any]) -> Any:
    cap = d.get("read_topk")
    return _s_kv(d) if cap is None else _dim.minimum(_s_kv(d), cap)


def _view_pairs(d: Dict[str, Any]) -> Any:
    """The pairs a reader of the pooled cache scores: the query whose context
    is `c` tokens sees its `⌊c / compress_ratio⌋` complete blocks, at most
    `read_topk` of them where the read is bounded (`_selected_pairs`)."""
    return _selected_pairs(d, cap=d.get("read_topk"),
                           ratio=d["compress_ratio"])


register_op(OpSpec("kv_cache_view",
    couples_on=("s_in",),
    shape_in=("B", "s_in", "d"),
    alt_shape_in=(("B", "n_heads", "s_in", "d"),),
    shape_out=("B", "s_kv", "d"),
    flops_fn=_zero,
    params_fn=_zero,
    is_view_op=True,
    shape_kind="axis_replace",
    axis_locked=-2,
    out_key="s_kv",
    kv_positions_per_query_fn=_view_read,
    kv_storage_positions_per_query_fn=_s_kv,
    kv_pairs_per_step_fn=_view_pairs,
    kv_stream_marker=True,
    kv_block_anchor=False,
    formula="kv_cached = kv[:s_kv]  (every cached position; Min(s_kv, read_topk) of them read where bounded)",
    description="Whole-cache KV marker: every cached position of a KV stream, "
                "for a reader that is not an attention kernel. Zero-cost; "
                "carries the s_kv read and storage cardinality.",
    required_dims=("s_kv",),
))

# This step's own keys as a KV stream: rows the step computes and attends but
# never writes to a cache. The dual of `kv_cache_view` (every cached position,
# read and stored) and of `sliding_window_kv` (the last `s_window` of them):
# these are read and not stored. DeepSeek-V4.1's DSpark draft block is the
# case (`DSparkAttention.forward`, source/model.py:1057-1066): the block's
# `kv = self.kv_norm(self.wkv(x))` is recomputed from `x` every step and
# joined behind the ring buffer of main-model latents
# (`torch.cat([self.window_kv_cache[:bsz], kv], dim=1)`), and
# `get_dspark_topk_idxs` lists at most its `block_size` rows for every query
# of the block. A `kv_stream_concat` sums its operands' laws, so the
# union of the window and these rows reads `Min(s_kv, s_window) + s_live`
# positions and stores the window's alone.
#
# `s_live` is stated on the node, as `sliding_window_kv` states `s_window`:
# how many rows the step reads is the module's fact (DSpark: one draft block,
# no wider than the rows that arrived), not the walk's query axis. The tensor
# passes through as it arrived; the count rides on the law, which is what
# the union sums and the score reads, as a window's `Min(s_kv, s_window)`
# does. A declared live law
# wider than the tensor that arrived is a structural gap, not a row a
# zero-cost view can create.
register_op(OpSpec("step_kv",
    couples_on=(),
    # The KV stream it marks, as `sliding_window_kv` takes one: the MQA
    # latent, or the head-major form after a head split. Shape preserved.
    shape_in=("B", "s_in", "d"),
    alt_shape_in=(("B", "n_heads", "s_in", "d"),),
    shape_out=("B", "s_in", "d"),
    flops_fn=_zero,
    params_fn=_zero,
    is_view_op=True,
    shape_kind="preserve",
    kv_positions_per_query_fn=lambda d: d["s_live"],
    kv_storage_positions_per_query_fn=lambda d: Integer(0),
    # Every query of the step reads every live row: no causal mask among
    # them (`get_dspark_topk_idxs` lists the whole block for each query).
    kv_pairs_per_step_fn=lambda d: _s_query(d) * d["s_live"],
    kv_stream_marker=True,
    kv_block_anchor=False,
    formula="kv_live = kv  (this step's s_live rows; read, never cached)",
    description="Live KV marker: this step's own keys, read by the step and "
                "never written to a cache. Zero-cost; carries an s_live read "
                "and no storage.",
    required_dims=("s_live",),
))

# The union of two KV streams, on the SEQUENCE axis.
#
# Source: `DeepseekV4Attention.forward` (modeling_deepseek_v4.py:832),
# `kv = torch.cat([kv, compressed_kv], dim=2)` on the `[B, 1, S, d_head]` MQA
# latent — one tensor, then one score, one softmax with one sink, one apply.
#
# Not a flag on `concat`, because the two are different operations. `concat`
# packs FEATURES — MLA's `k_nope ++ k_pe` — where both operands describe the
# same positions and summing their cardinalities would double `s_kv`. Gating
# on the axis would work and would also mean every future `concat` author has
# to know that `axis:` silently changes what the op means. Here the op name is
# the gate, so the gate is exactly the set of nodes that get the behaviour.
register_op(OpSpec("kv_stream_concat",
    # The joined axis IS the sequence, so a sequence cut splits it, and every
    # other axis rides through — the mirror of `concat`, which joins the
    # trailing axis and is transparent to the sequence.
    # The operands are labelled apart, as `concat` labels its two feature
    # widths apart: they are joined precisely because their extents differ,
    # and one label for both would be read as a claim that they agree.
    couples_on=("s_a", "s_b"),
    shape_in=(("B", "s_a", "d"), ("B", "s_b", "d")),
    alt_shape_in=((("B", "n_heads", "s_a", "d"),
                   ("B", "n_heads", "s_b", "d")),),
    shape_out=("B", "s_out", "d"),
    flops_fn=_zero,
    params_fn=_zero,
    # `torch.cat` allocates, the same reason `concat` is not a view op.
    is_view_op=False,
    shape_kind="axis_concat",
    axis_locked=-2,
    kv_stream_union=True,
    formula="kv = cat([kv_window, kv_compressed], dim=-2)",
    description="Join KV streams along the sequence axis. Its cardinality is "
                "the sum of its inputs', and that sum — not s_kv — is what "
                "every op downstream of it reads.",
))

# View-op: zero-FLOP axis reorder (PyTorch `.permute(*axes)` / `.transpose`).
# Per-call-site permutation supplied via the GraphNode's `axes:` field, e.g.
#
#   q_view: { op: view_split, in: q_proj,
#             dims: { n_heads: n_heads, d_head: d_head } }          # (B,s_ctx,n_heads,d_head)
#   q:      { op: transpose,  in: q_view, axes: [0, 2, 1, 3] }     # (B,n_heads,s_ctx,d_head)
#
# This is the canonical (B,s_ctx,n_heads·d_head) →
# (B,n_heads,s_ctx,d_head) chain that replaced the
# fused `split_heads` op. The inverse uses `transpose [0,2,1,3]` followed
# by `view_merge`.
register_op(OpSpec("transpose",
    # A shard names an axis SYMBOL, not a position, so reordering positions
    # carries it along untouched.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    # Templates are placeholders — actual rank comes from the upstream node.
    shape_in=(),
    shape_out=(),
    shape_kind="permute",   # reads node.axes at the call site
    description="View-op: reorder tensor axes (zero FLOPs / zero HBM in fused kernels). "
                "Permutation given by node `axes:`.",
    is_view_op=True,
))

# --- Hyper-Connection split-and-route ---------------------------------------
#
# V4-Pro `Block.hc_pre` calls `hc_split_sinkhorn(mixes, scale, base, hc_mult,
# sinkhorn_iters, eps)` (`inference/kernel.py`). Per token it cuts the
# d_hc_mix = (2 + n_hidden_copies) * n_hidden_copies mix vector into three
# segments, in this order, each with its own `scale` scalar:
#   `pre`  [B,s_ctx,n_hidden_copies] = sigmoid(mix·scale[0] + base) + eps
#          — pre-attn weighted-reduce coefficients (n_hidden_copies → 1)
#   `post` [B,s_ctx,n_hidden_copies] = 2·sigmoid(mix·scale[1] + base)
#          — post-attn additive coefficient (1 → n_hidden_copies, one per copy)
#   `comb` [B,s_ctx,n_hidden_copies,n_hidden_copies]
#          = Sinkhorn(softmax_row(mix·scale[2] + base)) — post-attn mixing
#          matrix; only this segment is projected toward doubly stochastic,
#          by `sinkhorn_iters` (default 20) rounds of row then column
#          normalization over the n_hidden_copies² block.
#
# This is the FIRST true multi-output primitive. Cost per token: the affine
# over all d_hc_mix elements (2·d_hc_mix) plus 20 rounds × (row sum, row div,
# col sum, col div) over n_hidden_copies² (80·n_hidden_copies²); the
# sigmoids and the softmax are a constant number of ops per element and
# fold into those terms.
#
# `dims` keys: { d_hc_mix, n_hidden_copies }; n_hidden_copies is the V4-Pro
# hyper-connection rate (`hc_mult` config). `sinkhorn_iters` doesn't appear in flops because it's
# a compile-time constant (the 80 is 4 ops × the default 20 rounds); specs
# that want it explicit can override with `flops_expr:`.
def _hc_split_sinkhorn_flops(d):
    n = d["n_hidden_copies"]
    return d["B"] * d["s_ctx"] * (2 * d["d_hc_mix"] + 80 * n * n)
register_op(OpSpec("hc_split_sinkhorn",
    # The three segments are cut out of one mix vector; no shard owns a segment.
    couples_on=("d_hc_mix",),
    flops_fn=_hc_split_sinkhorn_flops,
    params_fn=_zero,
    # Three inputs: (mixes [B,s_ctx,d_hc_mix], scale [3], base [d_hc_mix]). shape_in lists
    # the SHAPE TEMPLATE of the FIRST input only (the doc/display path
    # uses it for last-dim port labels). Engine resolves the actual three
    # input shapes from upstream.
    shape_in=("B", "s_ctx", "d_hc_mix"),
    shape_out=("B", "s_ctx", "n_hidden_copies"),     # main "out" port == `pre`
    shape_kind="explicit",
    secondary_outputs={
        "post": ("B", "s_ctx", "n_hidden_copies"),
        "comb": ("B", "s_ctx", "n_hidden_copies", "n_hidden_copies"),
    },
    formula="split mixes·scale + base by n_hidden_copies, n_hidden_copies, n_hidden_copies²; "
            "pre = σ(·) + ε, post = 2σ(·), comb = sinkhorn(softmax(·))",
    description=(
        "Hyper-Connection multi-output kernel: splits the "
        "d_hc_mix=(2+n_hidden_copies)·n_hidden_copies mix axis into "
        "pre [B,s_ctx,n_hidden_copies] (sigmoid) / post [B,s_ctx,n_hidden_copies] "
        "(2·sigmoid) / comb [B,s_ctx,n_hidden_copies,n_hidden_copies] (softmax, "
        "then Sinkhorn toward doubly stochastic). Main `out` port is `pre`; "
        "`post` and `comb` are "
        "secondary outputs referenced as `<node>.post` / `<node>.comb`."
    ),
    required_dims=("d_hc_mix", "n_hidden_copies"),
))


register_op(OpSpec("hc_mix",
    # The mixing matmul contracts the copy axis.
    couples_on=("n_hidden_copies",),
    # y[b,s,i,h] = Σⱼ comb[b,s,i,j] · residual[b,s,j,h]: a per-token batched
    # [n_hidden_copies,n_hidden_copies]·[n_hidden_copies,d_model] matmul
    # contracting the inner n_hidden_copies. FLOPs
    # O(B·s_ctx·n_hidden_copies²·d_model); activation only
    # [B,s_ctx,n_hidden_copies,d_model] — the n_hidden_copies² intermediate is
    # NEVER materialised.
    flops_fn=lambda d: 2 * d["B"] * d["s_ctx"] * d["n_hidden_copies"] * d["n_hidden_copies"] * d["d_model"],
    params_fn=_zero,
    # Two inputs: comb [B,s_ctx,n_hidden_copies,n_hidden_copies], residual
    # [B,s_ctx,n_hidden_copies,d_model]. shape_in lists both
    # templates (last-dim labels drive the doc/display path); the contracted
    # axis is comb's inner n_hidden_copies == residual's leading
    # n_hidden_copies.
    shape_in=(("B", "s_ctx", "n_hidden_copies", "n_hidden_copies"), ("B", "s_ctx", "n_hidden_copies", "d_model")),
    shape_out=("B", "s_ctx", "n_hidden_copies", "d_model"),
    shape_kind="explicit",
    formula="y[b,s,i,h] = Σⱼ comb[b,s,i,j]·residual[b,s,j,h]",
    description=(
        "Hyper-Connection residual mix (DSv4 `mhc_post`): the fused einsum "
        "`...ij,...jh->...ih` over the n_hidden_copies parallel hidden copies — verified against "
        "vLLM `model_executor/kernels/mhc/torch.py::mhc_post_torch` "
        "(`torch.einsum`, NOT the eager `sum(comb.unsqueeze(-1)·residual.unsqueeze(-2))` "
        "outer product). No real impl materialises the [B,s_ctx,n_hidden_copies,n_hidden_copies,d_model] intermediate, "
        "so neither does this op: O(B·s_ctx·n_hidden_copies²·d_model) FLOPs, [B,s_ctx,n_hidden_copies,d_model] activation."
    ),
    required_dims=("n_hidden_copies", "d_model"),
))


# ---------------------------------------------------------------------------
# Communication primitives — materialized from layout redistributes
# ---------------------------------------------------------------------------
#
# These four ops are NEVER referenced from spec yaml directly. They're
# grafted when two consecutive ops have mismatched layouts; the walk
# picks the right primitive from the state difference:
#
#   PartialSum → Replicated       → all_reduce
#   PartialSum → Sharded(axis)    → reduce_scatter
#   Sharded(axis) → Replicated    → all_gather
#   Sharded(axis_a) → Sharded(b)  → all_to_all
#
# All four are zero-FLOPs (pure data movement). Their `access_bytes` is
# the per-rank NCCL-ring traffic; the formula is parametric on the TP
# world size (`W`) which the inference layer binds at runtime from
# `ParallelConfig`. shape_kind="preserve" — input and output have the
# same shape modulo logical layout, only the rank-local view differs.
#
# Approximate per-rank ring formulas (T = tensor element count, dtype
# bytes folded in by access_bytes annotation):
#   all_reduce:     2 · (W-1)/W · T   ≈ 2T  (RS + AG composed)
#   reduce_scatter: (W-1)/W · T       ≈ T
#   all_gather:     (W-1)/W · T       ≈ T
#   all_to_all:     (W-1)/W · T       ≈ T   (each rank sends 1/W to W-1 peers)
#
# We encode the leading-order asymptotic (2T for AR, T for the
# others). The (W-1)/W correction is not applied here — it needs the
# world-size symbol from the inference-side binding. The two-tree ring is the assumed
# implementation; flashinfer / mscclpp / NVLink-SHARP variants get
# distinct kernel_impl labels in framework_bindings, not separate
# primitives here.
#
# `shape_in`/`shape_out` use the canonical post-norm hidden activation
# shape (B, s_ctx, d_model); ops that operate on different shapes (e.g. allreduce
# of a routing-weight vector) will still slot in fine because shape_kind
# is preserve.

register_op(OpSpec("all_reduce",
    reduction=True,
    collective_schedule="hierarchical",
    # A collective couples along nothing LOCALLY — it has no local math, so
    # `()` is the honest answer and the same one every other four collectives
    # give. What keeps it from being READ as transparent is not this field:
    # `is_comm_op` puts it ahead of the transparency question in
    # `eval/transition.py::Sharding.transition` rather than inside it, because
    # the thing being excluded is the question, not the answer.
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "d_model"),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="preserve",
    formula="y = Σ_rank x_rank  (NCCL ring all_reduce)",
    description=(
        "TP all_reduce: collapse a PartialSum back to Replicated. "
        "Per-rank traffic ≈ 2T (= RS + AG bandwidth). Emitted by "
        "layout at row-parallel → replicated boundaries."
    ),
    required_dims=("d_model",),
    is_comm_op=True,
))

register_op(OpSpec("reduce_scatter",
    reduction=True,
    sp_axis_effect="divide",
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "d_model"),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="preserve",
    formula="y_rank = (Σ_rank x_rank).slice(axis)",
    description=(
        "TP reduce_scatter: collapse a PartialSum into a Sharded "
        "result along some axis (typically the sequence dim in "
        "Megatron-SP). Per-rank traffic ≈ T."
    ),
    required_dims=("d_model",),
    is_comm_op=True,
))

register_op(OpSpec("all_gather",
    reduction=False,
    sp_axis_effect="multiply",
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "d_model"),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="preserve",
    formula="y = concat([x_rank for rank in W], axis)",
    description=(
        "TP all_gather: lift a Sharded tensor back to Replicated. "
        "Per-rank traffic ≈ T. Emitted by layout at SP→TP "
        "entry (where TP region needs the full feature view)."
    ),
    required_dims=("d_model",),
    is_comm_op=True,
))

register_op(OpSpec("all_to_all",
    reduction=False,
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "d_model"),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="preserve",
    formula="y_rank[r] = x_r[rank]  (transpose sharding axis)",
    description=(
        "all_to_all: swap a tensor's sharding axis (e.g. from Shard(s_ctx) "
        "to Shard(d_model)). The principal EP comm primitive — MoE dispatch "
        "and combine resolve to this. Per-rank traffic ≈ T."
    ),
    required_dims=("d_model",),
    is_comm_op=True,
))

register_op(OpSpec("send_recv",
    reduction=False,
    # Not a collective over a group's members but one hop between two of
    # them: the stage before a pipeline cut hands a tensor to the stage after
    # it. Its per-rank traffic is the tensor itself, whatever the group size
    # (`comm_volume.ring_factor` reads the schedule).
    collective_schedule="point_to_point",
    couples_on=(),
    flops_fn=_zero,
    params_fn=_zero,
    shape_in=("B", "s_ctx", "d_model"),
    shape_out=("B", "s_ctx", "d_model"),
    shape_kind="preserve",
    formula="y_{stage+1} = x_stage  (point-to-point send / recv)",
    description=(
        "Pipeline send/recv: a tensor live across a pipeline cut leaves the "
        "stage that holds it for the next. Per-rank traffic = T. Emitted "
        "for an appended stream's elements read past the cut that "
        "separates them from their writer."
    ),
    required_dims=("d_model",),
    is_comm_op=True,
))


# --- Guard-ablation experiment hook: HA_STRIP_OPS ---------------------------
# Comma-separated op names to remove from OP_REGISTRY at import, so the spec-gen
# agent's `validate` subprocess (a fresh import) cannot use a model-specific
# primitive and must re-declare it as an in-spec custom_op from the forward
# source. Inert when unset. The oracle re-validates in a process WITHOUT this
# env set, so grading uses the full registry.
import os as _os


def stripped_op_names() -> frozenset:
    """Op names `HA_STRIP_OPS` removes from the registry, parsed once.

    The spec-gen prompt hides the same names from the agent, and read the
    variable itself until this existed — two splits of one comma-separated
    list, which is one spelling change away from an experiment where the
    registry and the prompt disagree about what was withheld."""
    raw = _os.environ.get("HA_STRIP_OPS", "")
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


for _n in stripped_op_names():
    OP_REGISTRY.pop(_n, None)


#: A spec-gen continuation subprocess (never the long-lived API process)
#: points this at a proposal `patch.py` so `validate` sees the pending
#: primitive without splicing `ops_registry.py` on the serving tree.
PENDING_OP_PATCH_ENV = "HA_PENDING_OP_PATCH"


def apply_pending_op_patch(path: str) -> tuple[str, ...]:
    """Exec `OP_REGISTRATIONS` from a proposal patch. Returns names added.

    `register_op` still writes the process-local `OP_REGISTRY`. Callers
    that share a process with serving traffic must not set the env var.
    """
    from pathlib import Path


    patch_path = Path(path)
    if not patch_path.is_file():
        raise FileNotFoundError(f"pending op patch not found: {patch_path}")
    artifacts = read_artifacts(patch_path.parent)
    src = artifacts["registration"] or ""
    fragment = artifacts["registry_entries"]
    if fragment.strip():
        overlay_pending(load_yaml(fragment))
    else:
        overlay_pending(None)
    if not str(src).strip():
        return ()
    before = set(OP_REGISTRY)
    import sympy as sp
    for name in artifacts["symbols"]:
        key = str(name)
        if key not in _CANONICAL_DIMS:
            _CANONICAL_DIMS[key] = sp.Symbol(key)
    exec(compile(src, str(patch_path), "exec"), {**globals(), "ShapeKind": ShapeKind})
    return tuple(sorted(set(OP_REGISTRY) - before))


_pending_patch = _os.environ.get(PENDING_OP_PATCH_ENV)
if _pending_patch:
    apply_pending_op_patch(_pending_patch)
