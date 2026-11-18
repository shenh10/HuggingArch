"""Per-signature canonical inputs + reference implementation.

A signature is the MATH contract; every provider that claims it must compute
the same thing. This module is that contract's executable spec:

  * `make_inputs(inv, device, gen)` — build the CANONICAL input tensors for
    one invocation, every random tensor drawn from `gen`, the cell's own
    generator (`input_generator`, seeded off the measurement key) so a cell
    is the same data on every machine and every rerun. Extents come from the engine's carried `shape_in` (a permutation
    of the contract via `engine_inputs`); the identity scalars fill axes
    no operand holds. Canonical = one logical layout; each provider adapts
    it to its own (varlen / paged / …) inside `build`.
  * `reference(inputs)` — the torch-eager, fp32-accumulated correct output in
    the canonical layout. The cross-check compares each kernel's output
    against THIS, so "fastest" only counts among kernels that agree with it.

torch-guarded: importing this module needs torch, but the small references
run on CPU too (so the mechanism is unit-testable without a GPU). Signatures
without a `reference` (None) simply skip the correctness gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from backend.arch.dtypes import CORPUS_TOKENS
from backend.inference.op_signatures import all_signatures

from .invocation import Invocation


@dataclass
class RefSpec:
    make_inputs: Callable[[Invocation, str, Any], Dict[str, Any]]
    reference: Optional[Callable[[Dict[str, Any]], Any]] = None
    #: Whether this oracle MATERIALIZES the full score matrix (B·heads·Sq·Lq
    #: in fp32) that the kernel it checks never forms. Then its footprint is
    #: QUADRATIC in sequence length while the kernel's is linear, and at the
    #: large end the oracle is the only thing that cannot fit — the flash
    #: kernel runs happily on a cell whose reference needs terabytes. The
    #: budget that decides whether to cross-check a cell
    #: (`driver._reference_footprint_bytes`) asks the oracle here, rather
    #: than keeping its own list of the signatures whose reference does it.
    materializes_scores: bool = False


def input_generator(inv: Invocation, device: str) -> Any:
    """The generator every random input of `inv` is drawn from: one
    `torch.Generator` on `device`, seeded at `Invocation.input_seed`.

    Explicit rather than `torch.manual_seed`: the global RNG is shared with
    whatever else the process does (a provider's own `build`, an allocator
    warmup), so under it the data a cell gets would depend on the order the
    jobs ran in — and that is what made one cell flip its cross-check verdict
    four times in six reruns.

    A `meta` device holds no data and has no generator of its own (torch:
    "META device type not an accelerator"); a meta build records shapes, so
    its generator is a CPU one at the same seed.
    """
    import torch
    rng_device = "cpu" if str(device).startswith("meta") else device
    return torch.Generator(rng_device).manual_seed(inv.input_seed)


_REFS: Dict[str, RefSpec] = {}


#: What a builder's parameters are, in order. The generator is positional and
#: required, so a builder written without it is refused at registration rather
#: than quietly drawing from the global RNG at run time. A builder with no
#: randomness (pure `arange` / `zeros` / `ones`) takes it and ignores it —
#: taking it and not using it is visible, not having it is not.
_BUILDER_PARAMS = ("inv", "device", "gen")


def register_ref(signature: str, spec: RefSpec) -> None:
    """Declare the oracle for one signature. Once: a second registration
    of the same name replaced the first with no word said, which is how
    `situ_and_mul` kept being cross-checked against `tanh(x)·sigmoid(x)`
    (beta fixed at 1) after the beta-carrying reference was written."""
    import inspect
    if signature in _REFS:
        raise ValueError(
            f"reference for {signature!r} is registered twice. The second "
            f"replaces the first silently; edit the one declaration.")
    got = tuple(inspect.signature(spec.make_inputs).parameters)
    if got != _BUILDER_PARAMS:
        raise ValueError(
            f"the input builder for {signature!r} takes {got}; it takes "
            f"{_BUILDER_PARAMS} — `gen` is the cell's generator "
            f"(`input_generator`), and every random tensor is drawn from it "
            f"(a builder with no randomness takes it and ignores it).")
    _REFS[signature] = spec


def get_ref(signature: str) -> Optional[RefSpec]:
    return _REFS.get(signature)


class UnknownTorchDtype(KeyError):
    """A dtype name this table does not know.

    The three copies of this map used to answer a miss with `bfloat16`,
    so an fp8 / mxfp4 cell could be measured or cross-checked as bf16
    and still produce a number.
    """


def torch_dtype(name: str):
    """The one `str → torch.dtype` table for kernel_bench.

    Activation-width names only (`bf16` / `fp16` / `fp32` / the two
    fp8s). Packed weight formats are not torch dtypes and must not
    silently become bf16.
    """
    import torch
    table = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }
    try:
        return table[str(name).lower()]
    except KeyError:
        raise UnknownTorchDtype(
            f"unknown torch dtype {name!r}; known: {sorted(table)}"
        ) from None


def _dtype(name: str):
    return torch_dtype(name)


def _store_dtype(name: str):
    """The torch dtype an operand of `name` is STORED in for the harness.

    The operand's own declared dtype — a cache in fp32 is allocated in
    fp32, which is what its row is billed for — except for a format too
    narrow to hold a reference value, where the canonical input stays
    lossless and each provider quantizes (the convention `_gemm_inputs`
    set for fp8 weights). Allocating a cache at the activation's dtype
    instead is how Gemma-2, which ships in fp32, was billed four bytes an
    element while the harness moved two.
    """
    import torch
    dt = _dtype(name)
    return torch.bfloat16 if dt.is_floating_point and dt.itemsize < 2 else dt


def _operand_dtype(inv: Invocation, name: str):
    """The dtype the operand named `name` is built in — the slot its own
    declared ROLE names, never the dtype the kernel happens to run in.

    `TensorLayout.role` names a `dtype_env` slot (activation / weight / kv
    / metadata) and the cell carries one dtype per slot, so the slot is
    read here once rather than chosen per builder. Choosing per builder is
    how the weight slot went unread: a MoE router gate is stored fp32
    beside bf16 activations, its row is billed four bytes an element
    (`access_bytes_per_rank` = act·2 + w·4, to the byte), and the harness
    allocated two — the Gemma-2 cache bug one operand class over, reporting
    1.19x the bandwidth it moved on 11,582 cells of the corpus.

    Narrow and packed formats keep the convention `_store_dtype` states: an
    fp8 operand is carried lossless and each provider quantizes, and a
    format that is no torch dtype at all (mxfp4, int4 — the packing is the
    provider's business and the cross-check needs real values) is carried
    at the dtype the kernel runs in.
    """
    try:
        return _store_dtype(inv.dtype_for(_layout(inv, name).role
                                          or "activation"))
    except UnknownTorchDtype:
        return _store_dtype(inv.dtype)


def _operand_list(shape_in: Any) -> Optional[list]:
    """`shape_in` as a list of per-operand extents.

    The engine emits a bare tensor for a single operand (`[B, S, H]`) and a
    list of tensors for a fused span (`[Q, K, V]`). Same convention
    `identity_shape_from_shape_in` already reads.
    """
    if not isinstance(shape_in, (list, tuple)) or not shape_in:
        return None
    if isinstance(shape_in[0], (list, tuple)):
        return list(shape_in)
    return [shape_in]


def _zip_carried(engines: Any, operands: list, spec=None) -> Dict[str, int]:
    """One engine-layout declaration, all-or-nothing. Empty if none match."""
    from backend.inference.op_signatures import zip_engine_layouts
    if not engines:
        return {}
    got = zip_engine_layouts(engines, operands, spec=spec or {})
    return got or {}


def _carried_axes(inv: Invocation) -> Dict[str, int]:
    """Contract-axis extents observed on the engine's `shape_in`.

    Empty when the job carries no shapes or the engine layouts of the
    geometry the CELL selects (`OpSignature.binds_variant` on its dims) do
    not match. The cell's key already chose the geometry; the operands only
    give its extents. Trying the variant first whatever the cell said let a
    shorter operand list bind it — `zip_engine_layouts` skips a declared
    position the list does not reach — so GLM-5.3-Flash's latent span
    (Q, latent, selection: no value operand) was built as per-head GQA.
    """
    from backend.inference.op_signatures import get_signature
    sig = get_signature(inv.signature)
    operands = _operand_list(inv.shape_in)
    if sig is None or operands is None:
        return {}
    engines = (sig.variant_engine_inputs if sig.binds_variant(inv.dims)
               else sig.engine_inputs)
    return _zip_carried(engines, operands, spec=inv.dims)


def _dims(inv: Invocation) -> Dict[str, int]:
    """Axis values for constructing canonical inputs.

    The graph's carried activation shape wins on every axis it names —
    `inv.dims` is the lossy corpus-key projection. Axes the engine layout
    does not hold (a GEMM's `d_out`, a conv kernel width, structural
    top-k) stay on the scalars. Returns `inv.dims` itself when nothing
    is carried, so a recording dims mapping still sees the reads.
    """
    carried = _carried_axes(inv)
    if not carried:
        return inv.dims
    return {**inv.dims, **carried}


def _carried_weight(inv: Invocation) -> Optional[tuple]:
    """The param shape the GRAPH stated (`Invocation.shape_weight`), when the
    job carries it. Preferred over a rebuild from the identity scalars: `dims`
    is the lossy projection minted for corpus keys, while this is the shape a
    serving rank actually allocates — per-rank under TP, where the whole-tensor
    rebuild would measure a GEMM no rank runs. `None` on jobs that predate the
    export (or whose op's fused span has no single weight) → the caller
    rebuilds from the scalars as before."""
    if inv.shape_weight is None:
        return None
    return tuple(int(d) for d in inv.shape_weight)


def _layout(inv: Invocation, name: str) -> Any:
    """This invocation's declaration of the operand named `name`.

    The geometry it selects, not the signature's default: an asymmetric-V
    GQA cell lays its V out over `d_v_head` (`OpSignature.layouts_for`).
    Both halves of an operand — its extents and the dtype slot its role
    names — are read off this one lookup, so they cannot disagree about
    which declaration they came from.
    """
    from backend.inference.op_signatures import get_signature
    sig = get_signature(inv.signature)
    if sig is None:
        raise KeyError(f"no signature {inv.signature!r}")
    lay = next((item for item in sig.layouts_for(_dims(inv))
                if item.name == name), None)
    if lay is None:
        raise KeyError(f"{inv.signature} has no input {name!r}")
    return lay


def _shape(inv: Invocation, name: str) -> Tuple[int, ...]:
    """Canonical extent of one named operand, from the signature contract.

    Axis order lives on `OpSignature.inputs`, not in this module. Extents
    come from `_dims` (carried `shape_in` winning over identity scalars);
    a stored weight prefers `shape_weight` when the job carries it.
    """
    lay = _layout(inv, name)
    if lay.role == "weight":
        carried = _carried_weight(inv)
        # Same rank as the contract: the graph's tensor IS the kernel's.
        # Grouped GEMM carries one expert's (N, K); the kernel wants
        # (E, N, K) — ranks differ, so the contract (with E from dims) wins.
        if carried is not None and len(carried) == len(lay.axes):
            return carried
    ext = lay.extents(_dims(inv))
    if any(e is None for e in ext):
        raise ValueError(
            f"{inv.signature}.{name} has a non-cell dim {ext}")
    return tuple(int(e) for e in ext)


def _quant_output_cast(inv: Invocation, width: int):
    """The quantised activation an output-writing span casts to, or None.

    The cell states the cast as three facts — output dtype, scale granularity
    and scale dtype — and `ActivationCast` is the one reader of them. A plain
    output states no granularity, so the existing plain references and their
    bands are unchanged.
    """
    if inv.dtype_env.get("activation_granularity") is None:
        return None
    from .quant import ActivationCast
    return ActivationCast.of(inv, width=int(width))


# ---------------------------------------------------------------------------
# GEMM: x[M,K] @ w[N,K]^T → [M,N]
# ---------------------------------------------------------------------------

def _gemm_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    # Each operand at its own declared slot (`_operand_dtype`): a router
    # gate stored fp32 beside bf16 activations is billed four bytes an
    # element. fp8 can't hold randn directly for the reference comparison,
    # so a narrow format is carried lossless and each provider quantizes.
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=_operand_dtype(inv, "x"),
                         device=device, generator=gen),
        "w": torch.randn(*_shape(inv, "w"), dtype=_operand_dtype(inv, "w"),
                         device=device, generator=gen),
        "output_dtype": inv.dtype_env.get("output") or inv.dtype,
    }


def _gemm_reference(inputs: Dict[str, Any]) -> Any:
    # Oracle in the NATIVE dtype, not fp32: torch matmul accumulates bf16 inputs
    # in fp32 on tensor cores, so the bf16 output is accurate — while using half
    # the memory of an fp32 reference (no fp32 input copies, bf16 output). The
    # cross-check upcasts both sides to fp32 before comparing, and its job is to
    # catch wrong-computation (layout/op), which the per-dtype tolerance already
    # covers — not to audit precision. Anything the kernel fits, the oracle fits.
    #
    # The two slots are read at different widths, the activation slot names
    # the width the cell multiplies at, and the output slot names the width it
    # stores. The oracle follows all three declarations: native operands are
    # promoted to the compute width, then the product is rounded to the edge's
    # declared width, exactly where a promoted kernel narrows as it writes.
    x = inputs["x"]
    w = inputs["w"]
    product = x @ (w if w.dtype == x.dtype else w.to(x.dtype)).T
    return product.to(torch_dtype(inputs["output_dtype"]))


register_ref("gemm", RefSpec(_gemm_inputs, _gemm_reference))


# ---------------------------------------------------------------------------
# grouped_gemm (MoE): E expert weights w[E,N,K], grouped tokens x[E,Sq,K] →
# [E,Sq,N]. Reference is per-expert matmul (bmm) — the grouped/masked kernel
# must agree with running each expert's GEMM independently.
# ---------------------------------------------------------------------------

def _grouped_gemm_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    # The carried weight is ONE expert's (N, K) — the node is a single
    # expert's linear; the grouped kernel runs all E in one launch, so the
    # harness prepends the expert COUNT. A packing, not part of the shape.
    x = torch.randn(*_shape(inv, "x"), dtype=_operand_dtype(inv, "x"),
                    device=device, generator=gen)
    counts = inv.moe_token_counts
    for expert, count in enumerate(counts):
        x[expert, count:] = 0
    return {
        "x": x,
        "w": torch.randn(*_shape(inv, "w"), dtype=_operand_dtype(inv, "w"),
                         device=device, generator=gen),
        "token_counts": counts,
    }


def _grouped_gemm_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    x, w = inputs["x"], inputs["w"]                      # [E,Sq,K], [E,N,K] bf16
    # Per-expert fp32 matmul → bf16 (what a bf16 tensor-core GEMM produces).
    # Chunking over experts avoids materializing the full fp32 batched bmm
    # (peak = one expert, not all E), so large token counts fit in memory.
    outs = [(x[e].float() @ w[e].float().T).to(x.dtype) for e in range(x.shape[0])]
    out = torch.stack(outs)
    for expert, count in enumerate(inputs.get("token_counts", ())):
        out[expert, count:] = 0
    return out                                           # [E,Sq,N]


register_ref("grouped_gemm", RefSpec(_grouped_gemm_inputs, _grouped_gemm_reference))


# ---------------------------------------------------------------------------
# fused_moe: whole expert block. tokens [M,H] + top-k ids/weights +
# w1[E,2I,H] (gate|up) + w2[E,H,I] → [M,H]. Oracle is per-expert SwiGLU
# MLP then weighted reduce — the math FlashInfer's compute_with_experts
# and MegaMoE both claim, not a kernel.
# ---------------------------------------------------------------------------

def _fused_moe_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    x = torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen)
    ids_shape = _shape(inv, "topk_ids")
    n_experts = int(inv.dims["n_experts"])
    n_active = ids_shape[-1]
    m = x.shape[0]
    # Distinct experts per token, round-robin. Deterministic so kernel and
    # oracle see the same routing.
    base = torch.arange(m, device=device).unsqueeze(1)
    offs = torch.arange(n_active, device=device)
    topk_ids = (base + offs) % n_experts
    topk_weights = torch.full((m, n_active), 1.0 / n_active,
                              dtype=torch.float32, device=device)
    return {
        "x": x,
        "topk_ids": topk_ids.to(torch.int32),
        "topk_weights": topk_weights,
        "w1": torch.randn(*_shape(inv, "w1"), dtype=_operand_dtype(inv, "w1"),
                          device=device, generator=gen),
        "w2": torch.randn(*_shape(inv, "w2"), dtype=_operand_dtype(inv, "w2"),
                          device=device, generator=gen),
    }


def _fused_moe_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    import torch.nn.functional as F
    x = inputs["x"].float()
    ids = inputs["topk_ids"].long()
    scales = inputs["topk_weights"].float()
    w1 = inputs["w1"].float()                 # [E, 2I, H] gate|up
    w2 = inputs["w2"].float()                 # [E, H, I]
    m, h = x.shape
    i = w1.shape[1] // 2
    out = x.new_zeros(m, h)
    for e in range(w1.shape[0]):
        mask = ids == e
        if not mask.any():
            continue
        tok, slot = torch.where(mask)
        gate, up = w1[e, :i], w1[e, i:]
        hid = x[tok]
        inter = F.silu(hid @ gate.T) * (hid @ up.T)
        out[tok] += scales[tok, slot, None] * (inter @ w2[e].T)
    return out


register_ref("fused_moe", RefSpec(_fused_moe_inputs, _fused_moe_reference))


# ---------------------------------------------------------------------------
# MoE token dispatch (memory-bound). permute = gather routed tokens into
# expert-contiguous order; combine = weighted unpermute back to token order.
# The dispatch EXPANDS and its inverse REDUCES, and the two row counts are
# different: the step's tokens on one side (B·s_query), one routed copy per
# (token, expert) on the other (n_experts·s_expert_rows). Both contracts name
# both sides of their boundary, so the builders read the contract like every
# other signature's.
#
# Before the two sides had separate names, both were built from the cell's
# axes — which named only the routed one (`s_query` then meant the
# per-expert mean) — so the gather was an equal-length permutation and the
# combine an elementwise scale: 1.50x less traffic than the row's own
# numerator on a DeepSeek-V3.2 prefill cell, 0.63x more on a gpt-oss one —
# the ratio is `(1 + ep/topk)/2`, so it was high enough to break the SOL
# ceiling under EP and silently low without it.
# ---------------------------------------------------------------------------

def _routed_source(inv: Invocation, device: str, gen: Any, tokens: int,
                   routed: int):
    """Which token each routed row came from — `[routed]` into `[tokens]`.

    Many-to-one (a token goes to `topk` experts), and laid out expert by
    expert at the cell's own per-expert counts (`Invocation.
    moe_token_counts`, which is where a measured or declared load imbalance
    already lives): the row map is the routing, so a balanced cell and a
    skewed one are different gathers of the same width.
    """
    import torch
    counts = list(inv.moe_token_counts) or [routed]
    out = torch.empty(routed, dtype=torch.int64, device=device)
    at = 0
    for count in counts:
        if at >= routed:
            break
        take = min(int(count), routed - at)
        if take:
            out[at:at + take] = torch.randint(
                0, max(1, tokens), (take,), device=device, dtype=torch.int64,
                generator=gen)
        at += take
    if at < routed:                      # a cell whose counts under-fill the
        out[at:] = torch.randint(        # padded buffer: the rest are padding
            0, max(1, tokens), (routed - at,), device=device,
            dtype=torch.int64, generator=gen)
    return out


def _moe_permute_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    # Both sides of the gather are the contract's own declaration: the
    # token-ordered `x` it reads and the routed-long `idx` it writes with.
    tokens, width = _shape(inv, "x")
    (routed,) = _shape(inv, "idx")
    x = torch.randn(tokens, width, dtype=_operand_dtype(inv, "x"),
                    device=device, generator=gen)
    return {"x": x, "idx": _routed_source(inv, device, gen, tokens, routed)}


def _moe_permute_reference(inputs: Dict[str, Any]) -> Any:
    return inputs["x"][inputs["idx"]]


register_ref("moe_permute", RefSpec(_moe_permute_inputs, _moe_permute_reference))


def _moe_combine_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    # The reduce's read side is the contract's (`x` and the routing it
    # follows); the token rows it writes back to are the boundary's token
    # side (`Invocation.token_rows`, the walk's `RoutedLaunch.source_rows`).
    routed, width = _shape(inv, "x")
    tokens = inv.token_rows
    return {
        "x": torch.randn(routed, width, dtype=_operand_dtype(inv, "x"),
                         device=device, generator=gen),
        "w": torch.rand(routed, dtype=torch.float32, device=device, generator=gen),
        "idx": _routed_source(inv, device, gen, tokens, routed),
        "tokens": tokens,
    }


def _moe_combine_reference(inputs: Dict[str, Any]) -> Any:
    """The routed copies, weighted, summed back into their token's row — a
    REDUCTION, which is what the boundary is. An elementwise `x * w` moved one
    row out per row in and reduced nothing."""
    import torch
    out = torch.zeros(int(inputs["tokens"]), inputs["x"].shape[-1],
                      dtype=torch.float32, device=inputs["x"].device)
    return out.index_add_(0, inputs["idx"],
                          inputs["x"].float() * inputs["w"][:, None])


register_ref("moe_combine", RefSpec(_moe_combine_inputs, _moe_combine_reference))


# ---------------------------------------------------------------------------
# Linear attention (Gated DeltaNet, Kimi Delta Attention) and Mamba-2.
#
# Canonical inputs follow the layout the kernels document and the frameworks
# call them with — `[B, T, H, D]`, decay in LOG space, state `[N, HV, V, K]`:
#     q [B,T,H,K]   k [B,T,H,K]   v [B,T,HV,V]   beta [B,T,HV]
#     g [B,T,HV]      per-head scalar decay   (Gated DeltaNet)
#     g [B,T,HV,K]    per-channel decay       (KDA)
# HV > H is grouped-value attention (GVA), which the kernels expand internally.
# Both families reject fp32 outright ("please use bfloat16"), so the timed dtype
# is the model's activation dtype and only the ORACLE runs in fp32.
#
# The reference is the recurrence itself, stepped one token at a time in fp32 —
# the definition both the chunked and the fused-recurrent kernel must agree
# with. It is O(T) python, so it is bounded to short sequences; past that it
# raises and the driver measures without a cross-check, which is the same
# arrangement the attention oracle uses at large seq.
# ---------------------------------------------------------------------------

#: What a one-step-at-a-time scan can afford, as the TWO things it
#: actually spends. The loop is `for t in range(T)` and every step is
#: vectorised over the batch inside its einsums, so a batch costs nothing in
#: STEPS; what a batch does cost is the fp32 output and state the scan has to
#: hold, which is where a big one stops fitting.
#:
#: They were one budget spent as `B * T`, which is neither: it refused every
#: DECODE cell this family has — T of 1 to 5, B in the thousands, one Python
#: step — and 6,762 measured rows carry no oracle for a loop that would have
#: run once. The error even said "O(T) python" beside a condition that
#: multiplied B in.
SCAN_MAX_STEPS = 256
#: fp32 elements the oracle may materialise (output + state), about 1 GiB.
SCAN_MAX_ELEMENTS = 1 << 28
#:
#: Public because the TORCH FLOORS that run the same scan declare their own
#: ceiling from it (`providers/linear_ssm.py`): the oracle and the floor are
#: the same `_delta_rule_scan`, so "how many steps is too many" is one fact
#: and must not be two numbers. A floor that kept measuring past it spent
#: 48 seconds on a single launch at 262,144 steps — a figure that is neither
#: an upper bound (the fused kernels are thousands of times faster) nor a
#: lower one, for a row the oracle had already declined to check.


def _refuse_unaffordable_scan(kind: str, *, steps: int, elements: int) -> None:
    """Refuse a cell this oracle cannot pay for, saying which budget ran out.

    One statement for every sequential scan reference (delta rule, its two
    decode forms, Mamba-2's SSD, FLA's simple GLA) rather than the same
    number written five times: a budget copied is a budget that drifts, and
    these five already disagreed with their own error message.
    """
    if steps > SCAN_MAX_STEPS:
        raise ValueError(
            f"{kind} oracle is O(T) python: {steps} steps exceeds the "
            f"{SCAN_MAX_STEPS}-step bound (a batch is free here — it "
            f"is vectorised inside each step)")
    if elements > SCAN_MAX_ELEMENTS:
        raise ValueError(
            f"{kind} oracle would hold {elements:,} fp32 elements, past the "
            f"{SCAN_MAX_ELEMENTS:,} it may: the batch, not the steps, "
            f"is what does not fit")




def _delta_rule_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    if dt is torch.float32:              # the kernels assert against fp32
        dt = torch.bfloat16
    return {
        "q": torch.randn(*_shape(inv, "q"), dtype=dt, device=device, generator=gen),
        "k": torch.randn(*_shape(inv, "k"), dtype=dt, device=device, generator=gen),
        "v": torch.randn(*_shape(inv, "v"), dtype=dt, device=device, generator=gen),
        # log-space decay, strictly negative: g = log σ(·) in production.
        # Rank (per-head vs per-channel) comes from the signature contract.
        "g": -torch.rand(*_shape(inv, "g"), dtype=torch.float32, device=device, generator=gen),
        "beta": torch.rand(*_shape(inv, "beta"), dtype=dt, device=device, generator=gen).sigmoid(),
        "initial_state": torch.zeros(*_shape(inv, "initial_state"),
                                     dtype=torch.float32, device=device),
    }


def _delta_rule_reference(inputs: Dict[str, Any]) -> Any:
    """Sequential fp32 delta-rule recurrence — the definition, not a kernel.

        S_t = diag(exp g_t)·S_{t-1} + β_t · k_t (v_t − S_{t-1} k_t)ᵀ
        o_t = S_t q_t

    The delta term is what separates this family from plain linear attention:
    the update writes the RESIDUAL between the value and what the current state
    already predicts for this key. A per-channel `g` decays each key direction
    of the state separately, which is the whole of KDA's difference.
    """
    q, k, v = inputs["q"].float(), inputs["k"].float(), inputs["v"].float()
    g, beta = inputs["g"].float(), inputs["beta"].float()
    T = q.shape[1]
    _refuse_unaffordable_scan(
        "delta-rule", steps=T,
        elements=inputs["v"].numel() + inputs["initial_state"].numel())
    return _delta_rule_scan(q, k, v, g, beta, inputs["initial_state"].float())


def _delta_rule_scan(q, k, v, g, beta, initial_state, final_state=None):
    """One-token-at-a-time delta-rule. Shared by the oracle and the torch floor.

    `final_state`, when handed, receives the last step's state — the
    write-back a decode launch performs in its pool, which a floor that
    times the step must pay for as well."""
    import torch
    import torch.nn.functional as F
    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    rep = max(1, HV // max(1, H))
    qq = q.repeat_interleave(rep, dim=2) if rep > 1 else q
    kk = k.repeat_interleave(rep, dim=2) if rep > 1 else k
    scale = K ** -0.5
    S = initial_state.clone()
    out = q.new_empty(B, T, HV, V)
    channelwise = g.dim() == 4
    # The state is fp32 and the operands are the kernel's dtype (the
    # kernels keep it there too, `_delta_args`), so each step is taken in
    # the state's dtype — mixing them in the einsum is a type error, which
    # is what the torch floor hit on every fused cell.
    for t in range(T):
        k_t = kk[:, t].to(S.dtype)
        v_t = v[:, t].to(S.dtype)
        decay = g[:, t].exp().to(S.dtype)
        S = S * (decay.unsqueeze(-2) if channelwise else decay[..., None, None])
        pred = torch.einsum("bhvk,bhk->bhv", S, k_t)
        upd = beta[:, t].unsqueeze(-1).to(S.dtype) * (v_t - pred)
        S = S + upd.unsqueeze(-1) * k_t.unsqueeze(-2)
        out[:, t] = torch.einsum("bhvk,bhk->bhv", S,
                                 (qq[:, t] * scale).to(S.dtype)).to(out.dtype)
    if final_state is not None:
        final_state.copy_(S)
    return out


def _gdn_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    return _delta_rule_inputs(inv, device, gen)


def _kda_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    return _delta_rule_inputs(inv, device, gen)


register_ref("gated_delta_rule", RefSpec(_gdn_inputs, _delta_rule_reference))
register_ref("kda", RefSpec(_kda_inputs, _delta_rule_reference))


# The fused decode step: the same recurrence, with the gates formed from the
# raw projections inside the launch. The parameters are drawn the way a
# checkpoint holds them — A_log a log rate (the rate exp(A_log) in (0, 1]),
# dt_bias a small offset — and the raw a/b as projection outputs.

def _gdn_decode_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    if dt is torch.float32:              # the kernels assert against fp32
        dt = torch.bfloat16
    f32 = torch.float32
    return {
        "q": torch.randn(*_shape(inv, "q"), dtype=dt, device=device, generator=gen),
        "k": torch.randn(*_shape(inv, "k"), dtype=dt, device=device, generator=gen),
        "v": torch.randn(*_shape(inv, "v"), dtype=dt, device=device, generator=gen),
        "a": torch.randn(*_shape(inv, "a"), dtype=dt, device=device, generator=gen),
        "b": torch.randn(*_shape(inv, "b"), dtype=dt, device=device, generator=gen),
        "A_log": -torch.rand(*_shape(inv, "A_log"),
                             dtype=_operand_dtype(inv, "A_log"),
                             device=device, generator=gen),
        "dt_bias": 0.1 * torch.randn(*_shape(inv, "dt_bias"),
                                     dtype=_operand_dtype(inv, "dt_bias"),
                                     device=device, generator=gen),
        "initial_state": torch.zeros(*_shape(inv, "initial_state"),
                                     dtype=f32, device=device),
    }


def gdn_gates(a, b, A_log, dt_bias):
    """g = -exp(A_log)·softplus(a + dt_bias), beta = sigmoid(b) — the plan's
    `a_dt`, `softplus`, `rate`, `decay` and `write` nodes, per value head.
    Shared by the decode-step oracle and the gate-chain oracle
    (`fused_gdn_gating`)."""
    import torch
    import torch.nn.functional as F
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    return g, torch.sigmoid(b.float())


def _gdn_decode_reference(inputs: Dict[str, Any]) -> Any:
    q = inputs["q"].float()
    T = q.shape[1]
    _refuse_unaffordable_scan(
        "delta-rule", steps=T,
        elements=inputs["v"].numel() + inputs["initial_state"].numel())
    g, beta = gdn_gates(inputs["a"], inputs["b"], inputs["A_log"],
                        inputs["dt_bias"])
    return _delta_rule_scan(q, inputs["k"].float(), inputs["v"].float(), g,
                            beta, inputs["initial_state"].float())


register_ref("fused_recurrent_gated_delta_rule",
             RefSpec(_gdn_decode_inputs, _gdn_decode_reference))


def _gdn_gating_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "a": torch.randn(*_shape(inv, "a"), dtype=dt, device=device, generator=gen),
        "b": torch.randn(*_shape(inv, "b"), dtype=dt, device=device, generator=gen),
        "A_log": -torch.rand(*_shape(inv, "A_log"),
                             dtype=_operand_dtype(inv, "A_log"),
                             device=device, generator=gen),
        "dt_bias": 0.1 * torch.randn(*_shape(inv, "dt_bias"),
                                     dtype=_operand_dtype(inv, "dt_bias"),
                                     device=device, generator=gen),
    }


def _gdn_gating_reference(inputs: Dict[str, Any]) -> Any:
    """The two outputs stacked `[tokens, 2, HV]` — g then beta — so one
    comparison covers both ports."""
    import torch
    g, beta = gdn_gates(inputs["a"], inputs["b"], inputs["A_log"],
                        inputs["dt_bias"])
    return torch.stack([g, beta], dim=-2)


register_ref("fused_gdn_gating",
             RefSpec(_gdn_gating_inputs, _gdn_gating_reference))


def _fused_gdn_post_conv_prep_inputs(inv: Invocation, device: str,
                                     gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    w32 = _operand_dtype(inv, "A_log")
    d = _dims(inv)
    return {
        "qkv": torch.randn(*_shape(inv, "qkv"), dtype=dt, device=device,
                           generator=gen),
        "a": torch.randn(*_shape(inv, "a"), dtype=dt, device=device,
                         generator=gen),
        "b": torch.randn(*_shape(inv, "b"), dtype=dt, device=device,
                         generator=gen),
        "A_log": -torch.rand(*_shape(inv, "A_log"), dtype=w32,
                             device=device, generator=gen),
        "dt_bias": 0.1 * torch.randn(*_shape(inv, "dt_bias"), dtype=w32,
                                      device=device, generator=gen),
        # The port boundaries are cell axes the flat conv row cannot reveal.
        "H": int(d["n_lin_k_heads"]), "K": int(d["d_lin_k_head"]),
        "HV": int(d["n_lin_v_heads"]), "V": int(d["d_lin_v_head"]),
    }


def gdn_post_conv_prep(inputs: Dict[str, Any]):
    """The five ports `fused_post_conv_prep` writes, in its wrapper order."""
    import torch
    H, K, HV, V = (int(inputs[n]) for n in ("H", "K", "HV", "V"))
    qkv = inputs["qkv"].float()
    q, k, v = qkv.split([H * K, H * K, HV * V], dim=-1)
    norm = lambda x: x / torch.sqrt(x.square().sum(-1, keepdim=True) + 1e-6)
    q = norm(q.reshape(*q.shape[:-1], H, K))
    k = norm(k.reshape(*k.shape[:-1], H, K))
    v = v.reshape(*v.shape[:-1], HV, V)
    g, beta = gdn_gates(inputs["a"], inputs["b"], inputs["A_log"],
                        inputs["dt_bias"])
    return q, k, v, g, beta


def _fused_gdn_post_conv_prep_reference(inputs: Dict[str, Any]) -> Any:
    # The driver cross-checks the primary q port; the complete port tuple is
    # public above and pinned by the provider tests.
    return gdn_post_conv_prep(inputs)[0]


register_ref("fused_gdn_post_conv_prep",
             RefSpec(_fused_gdn_post_conv_prep_inputs,
                     _fused_gdn_post_conv_prep_reference))


#: The bound of KDA's bounded forget gate, g = lower_bound·sigmoid(·). The
#: graph carries it as a `scale_const` with no value on the node, and it moves
#: no byte, so the cell is measured at one stated value: -5.0, what
#: GLM-5.3-Flash publishes (`linear_lower_bound`) and the kernels' own
#: default (vLLM glm5next kda kernels.py `lower_bound=-5.0`). Every provider
#: passes this same value, so the cross-check compares one function.
KDA_LOWER_BOUND = -5.0


def _kda_decode_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    f32 = torch.float32
    return {
        "q": torch.randn(*_shape(inv, "q"), dtype=dt, device=device, generator=gen),
        "k": torch.randn(*_shape(inv, "k"), dtype=dt, device=device, generator=gen),
        "v": torch.randn(*_shape(inv, "v"), dtype=dt, device=device, generator=gen),
        "f": torch.randn(*_shape(inv, "f"), dtype=dt, device=device, generator=gen),
        "b": torch.randn(*_shape(inv, "b"), dtype=dt, device=device, generator=gen),
        "A_log": -torch.rand(*_shape(inv, "A_log"),
                             dtype=_operand_dtype(inv, "A_log"),
                             device=device, generator=gen),
        "dt_bias": 0.1 * torch.randn(*_shape(inv, "dt_bias"),
                                     dtype=_operand_dtype(inv, "dt_bias"),
                                     device=device, generator=gen),
        "initial_state": torch.zeros(*_shape(inv, "initial_state"),
                                     dtype=f32, device=device),
        "lower_bound": KDA_LOWER_BOUND,
    }


def kda_bounded_gates(f, b, A_log, dt_bias, lower_bound):
    """g = lower_bound·sigmoid(exp(A_log)·(f + dt_bias)) per key channel and
    beta = sigmoid(b) — the plan's `f_bias`, `f_split` (the head split),
    `rate`, `g_rate`, `g_squash`, `g_bound` and `write` nodes (the
    transposes move no value). `f [B, T, HV·K]` and `dt_bias [HV·K]` flat, as
    the fused node reads them; `A_log [HV]`. Returns g `[B, T, HV, K]`."""
    import torch
    HV = int(A_log.shape[0])
    x = f.float() + dt_bias.float()                       # f_bias, flat
    x = x.reshape(*x.shape[:-1], HV, -1)                  # f_split
    g = lower_bound * torch.sigmoid(A_log.float().exp()[:, None] * x)
    return g, torch.sigmoid(b.float())


def _kda_decode_reference(inputs: Dict[str, Any]) -> Any:
    q = inputs["q"].float()
    T = q.shape[1]
    _refuse_unaffordable_scan(
        "delta-rule", steps=T,
        elements=inputs["v"].numel() + inputs["initial_state"].numel())
    g, beta = kda_bounded_gates(inputs["f"], inputs["b"], inputs["A_log"],
                                inputs["dt_bias"], inputs["lower_bound"])
    return _delta_rule_scan(q, inputs["k"].float(), inputs["v"].float(), g,
                            beta, inputs["initial_state"].float())


register_ref("fused_recurrent_kda",
             RefSpec(_kda_decode_inputs, _kda_decode_reference))


# ---------------------------------------------------------------------------
# fused_kda_decode: Kimi-K3's whole KDA mixer for one decode token. Raw q/k/v
# through their causal short convs (state taps oldest first, then the new
# token) and SiLU, then the step `fused_recurrent_kda` states, then the gated
# output RMSNorm. Both pools are drawn non-trivial — a zero recurrent state
# would make the forget gate invisible to the cross-check, and a zero conv
# state would leave three of the four taps unread.
# ---------------------------------------------------------------------------

def _kda_mixer_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    out = _kda_decode_inputs(inv, device, gen)
    dt = out["q"].dtype
    w32 = _store_dtype(inv.dtype_for("weight"))
    width = int(_dims(inv)["s_conv_kernel"])
    for n in ("q_conv_w", "k_conv_w", "v_conv_w"):
        out[n] = torch.randn(*_shape(inv, n), dtype=w32, device=device,
                             generator=gen) / width
    out["g"] = torch.randn(*_shape(inv, "g"), dtype=dt, device=device,
                           generator=gen)
    out["o_norm_w"] = torch.randn(*_shape(inv, "o_norm_w"), dtype=w32,
                                  device=device, generator=gen)
    channels = int(out["q"].shape[-1] * 2 + out["v"].shape[-1])
    out["conv_state"] = torch.randn(int(out["q"].shape[0]), channels, width - 1,
                                    dtype=dt, device=device, generator=gen)
    out["initial_state"] = 0.1 * torch.randn(
        *_shape(inv, "initial_state"), dtype=torch.float32, device=device,
        generator=gen)
    out["norm_eps"] = 1e-5
    return out


def kda_short_conv(x: Any, state: Any, weight: Any) -> Any:
    """silu(Σ_w window[w]·weight[:, w]) for one new token: `x [B, 1, C]`,
    `state [B, C, W−1]` the previous taps oldest first, `weight [C, W]`."""
    import torch
    window = torch.cat([state.float(), x.float().transpose(1, 2)], dim=-1)
    return torch.nn.functional.silu(
        (window * weight.float()[None]).sum(-1))[:, None, :]


def _kda_mixer_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's convs, norms, gates, recurrence and gated output norm:
    `[B, 1, n_lin_v_heads, d_lin_v_head]`."""
    import torch
    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    cq, ck = q.shape[-1], k.shape[-1]
    state = inputs["conv_state"]
    qc = kda_short_conv(q, state[:, :cq], inputs["q_conv_w"])
    kc = kda_short_conv(k, state[:, cq:cq + ck], inputs["k_conv_w"])
    vc = kda_short_conv(v, state[:, cq + ck:], inputs["v_conv_w"])
    HV = int(inputs["A_log"].shape[0])
    B = q.shape[0]
    g, beta = kda_bounded_gates(inputs["f"], inputs["b"], inputs["A_log"],
                                inputs["dt_bias"], inputs["lower_bound"])
    heads = (lambda x: x.reshape(B, 1, HV, -1))
    o = _delta_rule_scan(heads(qc), heads(kc), heads(vc), g, beta,
                         inputs["initial_state"].float())
    o = o.float()
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + inputs["norm_eps"])
    return (o * inputs["o_norm_w"].float()
            * torch.sigmoid(heads(inputs["g"].float())))


register_ref("fused_kda_decode",
             RefSpec(_kda_mixer_inputs, _kda_mixer_reference))


# ---------------------------------------------------------------------------
# fused_kda_gate: the KDA prefill gates in one launch — the bounded gate
# `kda_bounded_gates` states, then the two steps sglang's
# `kda_gate_chunk_cumsum` folds in past the gate formation: the chunk-local
# cumsum the chunk pipeline would otherwise run as its own launch, and the
# RCP_LN2 scale that puts the decay in the log2 space the chunk kernels read.
# ---------------------------------------------------------------------------

#: The chunk the launch cumsums within: `chunk_kda_fwd`'s fixed
#: `chunk_size = 64` (sglang fla/kda.py:1167), a launch constant like
#: `KDA_LOWER_BOUND`, not a deployment choice — so no cell axis.
KDA_GATE_CHUNK = 64

#: 1/ln 2 — the scale the launch applies to the cumsummed gate (sglang
#: fla/kda.py:45 `RCP_LN2`, handed over at :1182).
RCP_LN2 = 1.4426950216293335


def kda_gate_chunk_cumsum(f, b, A_log, dt_bias, lower_bound):
    """The launch's two outputs: the bounded gate chunk-cumsummed per
    `KDA_GATE_CHUNK` tokens and scaled to log2, `[B, T, HV, K]` fp32, and
    beta = sigmoid(b) `[B, T, HV]` fp32."""
    import torch
    g, beta = kda_bounded_gates(f, b, A_log, dt_bias, lower_bound)
    T = g.shape[1]
    pad = -T % KDA_GATE_CHUNK
    if pad:
        g = torch.nn.functional.pad(g, (0, 0, 0, 0, 0, pad))
    g = g.unflatten(1, (-1, KDA_GATE_CHUNK)).cumsum(2).flatten(1, 2)
    if pad:
        g = g[:, :T]
    return g * RCP_LN2, beta


def _kda_gate_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "f": torch.randn(*_shape(inv, "f"), dtype=dt, device=device, generator=gen),
        "b": torch.randn(*_shape(inv, "b"), dtype=dt, device=device, generator=gen),
        "A_log": -torch.rand(*_shape(inv, "A_log"),
                             dtype=_operand_dtype(inv, "A_log"),
                             device=device, generator=gen),
        "dt_bias": 0.1 * torch.randn(*_shape(inv, "dt_bias"),
                                     dtype=_operand_dtype(inv, "dt_bias"),
                                     device=device, generator=gen),
        "lower_bound": KDA_LOWER_BOUND,
    }


def _kda_gate_reference(inputs: Dict[str, Any]) -> Any:
    """The launch's two ports side by side — `[B, T, HV, K+1]`: the
    cumsummed log2 gate, then beta as the extra channel, so one comparison
    covers both."""
    import torch
    g, beta = kda_gate_chunk_cumsum(inputs["f"], inputs["b"], inputs["A_log"],
                                    inputs["dt_bias"], inputs["lower_bound"])
    return torch.cat([g, beta.unsqueeze(-1)], dim=-1)


register_ref("fused_kda_gate",
             RefSpec(_kda_gate_inputs, _kda_gate_reference))


# ---------------------------------------------------------------------------
# causal_conv1d: depthwise causal short conv, `[B, S, C]` feature-last with one
# width-`W` filter per channel. Production calls it on the q/k/v projections of
# every Mamba / GDN / KDA layer, so C is heads×head_dim, not d_model.
# ---------------------------------------------------------------------------

def _causal_conv1d_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    weight = torch.randn(*_shape(inv, "weight"),
                         dtype=_operand_dtype(inv, "weight"),
                         device=device, generator=gen)
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen),
        "weight": weight,
        "bias": torch.randn(*_shape(inv, "bias"),
                            dtype=_operand_dtype(inv, "bias"),
                            device=device, generator=gen),
        "W": int(weight.shape[-1]),
        # Tap spacing — a structural axis of the cell (the conv node's own
        # `dilation`), not a tensor's.
        "dilation": int(_dims(inv)["dilation"]),
    }


def _causal_conv1d_reference(inputs: Dict[str, Any]) -> Any:
    """y[c, t] = silu(bias[c] + Σ_j w[c, j]·x[c, t − (W−1−j)·dil]), as an
    explicit fp32 tap sum. Not `F.conv1d`: on a GPU that runs cuDNN, which
    computes an fp32 conv in TF32 by default (`cudnn.allow_tf32`), and the
    oracle then sat further from both frameworks' fp32-accumulating kernels
    than they sit from each other (rel_err 0.0258 for vLLM and sglang alike
    on an H100, cosine 1.00000)."""
    import torch.nn.functional as F
    x = inputs["x"].float()                               # [B,C,S]
    w = inputs["weight"].float()                          # [C,W]
    W = w.shape[-1]
    S = x.shape[-1]
    dil = int(inputs["dilation"])
    xp = F.pad(x, ((W - 1) * dil, 0))                     # causal: left-pad only
    out = inputs["bias"].float()[None, :, None].expand_as(x).clone()
    for j in range(W):
        out += w[None, :, j, None] * xp[:, :, j * dil:j * dil + S]
    return F.silu(out)                                    # the span folds silu in


register_ref("causal_conv1d",
             RefSpec(_causal_conv1d_inputs, _causal_conv1d_reference))


def _mamba2_scan_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen),
        "dt": torch.rand(*_shape(inv, "dt"), dtype=torch.float32, device=device, generator=gen),
        "A": -torch.rand(*_shape(inv, "A"), dtype=torch.float32, device=device, generator=gen),
        "B": torch.randn(*_shape(inv, "B"), dtype=dt, device=device, generator=gen),
        "C": torch.randn(*_shape(inv, "C"), dtype=dt, device=device, generator=gen),
        "D": torch.randn(*_shape(inv, "D"), dtype=torch.float32, device=device, generator=gen),
        "dt_bias": torch.randn(*_shape(inv, "dt_bias"),
                               dtype=torch.float32, device=device, generator=gen),
    }


def _ssd_scan(x, dt, A, B, C, D, dt_bias):
    """Mamba-2 selective scan, one step at a time.

    x [B,S,H,P], dt [B,S,H], A [H], B/C [B,S,G,N], D/dt_bias [H].
    Groups broadcast onto heads. `mamba_chunk_scan_combined` does the
    same recurrence chunked; this is the definition it has to match.
    """
    import torch
    import torch.nn.functional as F
    delta = F.softplus(dt + dt_bias)
    dA = torch.exp(delta * A)
    bsz, seq, n_heads, d_head = x.shape
    n_groups, d_state = B.shape[2], B.shape[3]
    rep = n_heads // max(n_groups, 1)
    Be = B.repeat_interleave(rep, dim=2) if rep > 1 else B
    Ce = C.repeat_interleave(rep, dim=2) if rep > 1 else C
    state = x.new_zeros(bsz, n_heads, d_head, d_state)
    skip = D.reshape(1, n_heads, 1)
    out = x.new_empty(bsz, seq, n_heads, d_head)
    for t in range(seq):
        state = state * dA[:, t].reshape(bsz, n_heads, 1, 1)
        state = state + (
            delta[:, t].reshape(bsz, n_heads, 1, 1)
            * x[:, t].unsqueeze(-1)
            * Be[:, t].unsqueeze(-2)
        )
        out[:, t] = (state * Ce[:, t].unsqueeze(-2)).sum(-1) + skip * x[:, t]
    return out


def _mamba2_scan_reference(inputs: Dict[str, Any]) -> Any:
    x = inputs["x"].float()
    T = x.shape[1]
    _refuse_unaffordable_scan(
        "mamba2", steps=T,
        elements=x.numel() + inputs["B"].numel() + inputs["C"].numel())
    return _ssd_scan(
        x, inputs["dt"].float(), inputs["A"].float(),
        inputs["B"].float(), inputs["C"].float(),
        inputs["D"].float(), inputs["dt_bias"].float(),
    )


register_ref("mamba2_scan", RefSpec(_mamba2_scan_inputs, _mamba2_scan_reference))


def _mamba2_rmsnorm_gated_inputs(inv: Invocation, device: str,
                                 gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device,
                         generator=gen),
        "gate": torch.randn(*_shape(inv, "gate"), dtype=dt, device=device,
                            generator=gen),
        "weight": torch.randn(*_shape(inv, "weight"),
                              dtype=_operand_dtype(inv, "weight"),
                              device=device, generator=gen),
        "group_size": (int(_dims(inv)["n_ssm_heads"])
                       * int(_dims(inv)["d_ssm_head"])
                       // int(_dims(inv)["n_ssm_groups"])),
        "d_ssm_state": int(_dims(inv)["d_ssm_state"]),
        "eps": 1e-6,
    }


def _mamba2_rmsnorm_gated_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    x = (inputs["x"].float()
         * torch.nn.functional.silu(inputs["gate"].float()))
    group = int(inputs["group_size"])
    if x.shape[-1] % group:
        raise ValueError(
            f"Mamba-2 norm width {x.shape[-1]} is not a whole number of "
            f"{group}-wide groups")
    grouped = x.view(*x.shape[:-1], -1, group)
    normed = (grouped * torch.rsqrt(
        grouped.pow(2).mean(-1, keepdim=True) + inputs["eps"])
    ).view_as(x)
    return normed * inputs["weight"].float()


register_ref("mamba2_rmsnorm_gated",
             RefSpec(_mamba2_rmsnorm_gated_inputs,
                     _mamba2_rmsnorm_gated_reference))


# ---------------------------------------------------------------------------
# rmsnorm: x[M,H] * rsqrt(mean(x^2)+eps) * w[H]
# ---------------------------------------------------------------------------

def _rmsnorm_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    x = torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen)
    return {
        "x": x,
        "w": torch.randn(*_shape(inv, "w"), dtype=_operand_dtype(inv, "w"),
                         device=device, generator=gen),
        "eps": 1e-6,
        "cast": _quant_output_cast(inv, x.shape[-1]),
    }


def _rmsnorm_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    x = inputs["x"].float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + inputs["eps"]) * inputs["w"].float()
    return y if inputs.get("cast") is None else inputs["cast"].on_grid(y)


register_ref("rmsnorm", RefSpec(_rmsnorm_inputs, _rmsnorm_reference))


def _fused_add_rmsnorm_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    x = torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen)
    return {
        "x": x,
        "residual": torch.randn(*_shape(inv, "residual"), dtype=dt,
                                device=device, generator=gen),
        "w": torch.randn(*_shape(inv, "w"), dtype=_operand_dtype(inv, "w"),
                         device=device, generator=gen),
        "eps": 1e-6,
        "cast": _quant_output_cast(inv, x.shape[-1]),
    }


def _fused_add_rmsnorm_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    r = (inputs["residual"].float() + inputs["x"].float())
    var = r.pow(2).mean(dim=-1, keepdim=True)
    y = r * torch.rsqrt(var + inputs["eps"]) * inputs["w"].float()
    return y if inputs.get("cast") is None else inputs["cast"].on_grid(y)


register_ref("fused_add_rmsnorm",
             RefSpec(_fused_add_rmsnorm_inputs, _fused_add_rmsnorm_reference))


def _engram_gate_inputs(inv: Invocation, device: str,
                        gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    wdt = _operand_dtype(inv, "q_weight")
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device,
                         generator=gen),
        "kv": torch.randn(*_shape(inv, "kv"), dtype=dt, device=device,
                          generator=gen),
        "q_weight": torch.randn(*_shape(inv, "q_weight"), dtype=wdt,
                                device=device, generator=gen),
        "k_weight": torch.randn(*_shape(inv, "k_weight"), dtype=wdt,
                                device=device, generator=gen),
        "eps": 1e-6,
    }


# A signed square root has TWO audited lowerings in this corpus, and the two
# functions below are TWO SPANS' ORACLES — not two spellings of one formula.
# Each component declares its own, citing its own model source:
#
#   DeepSeek-V4.1 engram (`components/library/dsv41_engram.yaml:130-131`):
#       torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)
#   Qwen3.8 PLE (`components/library/qwen_ple.yaml:93`, and the line sglang
#   runs at srt/models/qwen4_exp.py:1377):
#       gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
#
# They agree at every ordinary value and part at signed zero (`sign(-0.) == 0`
# zeroes the gate; `copysign` keeps the floor's magnitude with a negative
# sign), and they are a different number of elementwise launches — three
# against five, which the `signed_sqrt` op's own cost coefficient prices at
# five. So an oracle or a floor must take the one ITS span runs: using
# `copysign` for the PLE span timed four steps where the framework runs five.
# Picking by span is why neither is called "the" signed sqrt.

def _signed_sqrt_copysign(x, floor: float = 1e-6):
    """DeepSeek-V4.1's engram gate lowering."""
    import torch
    return torch.copysign(x.abs().clamp_min(floor).sqrt(), x)


def _signed_sqrt_times_sign(x, floor: float = 1e-6):
    """Qwen3.8's PLE gate lowering, which sglang runs as written."""
    return x.abs().clamp_min(floor).sqrt() * x.sign()


def _engram_gate_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    x3 = inputs["x"].float().unflatten(-1, (-1, inputs["q_weight"].shape[-1]))
    hc, d = inputs["q_weight"].shape
    key, value = inputs["kv"].float().split([hc * d, d], dim=-1)
    key = key.unflatten(-1, (hc, d))
    h_rms = x3 * torch.rsqrt(x3.pow(2).mean(-1, keepdim=True)
                             + inputs["eps"])
    k_rms = key * torch.rsqrt(key.pow(2).mean(-1, keepdim=True)
                              + inputs["eps"])
    dot = ((h_rms * inputs["q_weight"].float())
           * (k_rms * inputs["k_weight"].float())).sum(-1) * d ** -0.5
    gate = torch.sigmoid(_signed_sqrt_copysign(dot))
    out = x3 + gate.unsqueeze(-1) * value.unsqueeze(-2)
    return out.flatten(-2)


register_ref("fused_engram_gate",
             RefSpec(_engram_gate_inputs, _engram_gate_reference))


def _ple_gate_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    wdt = _operand_dtype(inv, "w_key")
    return {
        "key": torch.randn(*_shape(inv, "key"), dtype=dt, device=device,
                           generator=gen),
        "value": torch.randn(*_shape(inv, "value"), dtype=dt, device=device,
                             generator=gen),
        "x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device,
                         generator=gen),
        "w_key": torch.randn(*_shape(inv, "w_key"), dtype=wdt,
                             device=device, generator=gen),
        "w_query": torch.randn(*_shape(inv, "w_query"), dtype=wdt,
                               device=device, generator=gen),
        "w_conv": torch.randn(*_shape(inv, "w_conv"), dtype=wdt,
                              device=device, generator=gen),
        "eps": 1e-6,
    }


def ple_gate(inputs: Dict[str, Any]):
    """The PLE launch's two ports: gated residual, then conv input."""
    import torch
    d = inputs["value"].shape[-1]
    hc = inputs["x"].shape[-1] // d

    def norm_gain(x, w):
        y = x.float().unflatten(-1, (hc, d))
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True)
                            + inputs["eps"])
        return (y * (1.0 + w.float().view(hc, d))).flatten(-2)

    k = norm_gain(inputs["key"], inputs["w_key"])
    q = norm_gain(inputs["x"], inputs["w_query"])
    dot = (k * q).unflatten(-1, (hc, d)).sum(-1) * d ** -0.5
    gate = torch.sigmoid(_signed_sqrt_times_sign(dot))
    gated = gate.unsqueeze(-1) * inputs["value"].float().unsqueeze(-2)
    c = norm_gain(gated.flatten(-2), inputs["w_conv"])
    return gated.flatten(-2), c


def _ple_gate_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    gated, conv = ple_gate(inputs)
    return torch.stack([gated, conv], dim=-2)


register_ref("ple_gate", RefSpec(_ple_gate_inputs, _ple_gate_reference))


def _mxfp8_rows_on_grid(x):
    """`x` as vLLM's MXFP8 record holds it: FP8 e4m3 values with one UE8M0
    scale per 32-value group (query_quant.py:63-78, the grid FlashInfer's
    F8_128x4 scale layout stores).

    The comparison uses the dequantized values: the canonical contract is the
    normalized row a projection consumes, while the packed scale permutation
    is the provider's launch ABI.
    """
    import torch
    from .quant import FORMATS, quantise_groups
    q, scale = quantise_groups(
        x.float(), FORMATS["mxfp8"], 32, "pow2", x.dtype,
        eps=torch.finfo(torch.float32).tiny)
    return (q.float() * scale.unsqueeze(-1)).reshape(x.shape)


def _q_kv_rmsnorm_quant_inputs(inv: Invocation, device: str,
                               gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    # Read both row widths before drawing either tensor: a malformed cell
    # names both axes it omitted, not just the one before the first raise.
    dims = _dims(inv)
    _ = (dims["d_q_lora"], dims["d_kv_lora"])
    return {
        "q": torch.randn(*_shape(inv, "q"), dtype=dt, device=device,
                         generator=gen),
        "kv": torch.randn(*_shape(inv, "kv"), dtype=dt, device=device,
                          generator=gen),
        "q_weight": torch.randn(*_shape(inv, "q_weight"),
                                dtype=_operand_dtype(inv, "q_weight"),
                                device=device, generator=gen),
        "kv_weight": torch.randn(*_shape(inv, "kv_weight"),
                                 dtype=_operand_dtype(inv, "kv_weight"),
                                 device=device, generator=gen),
        # DeepSeek-V4.1's checkpoint value; the kernel reads it from the RMSNorm
        # module at deepseek_v41/attention.py:792.
        "eps": 1e-20,
    }


def _q_kv_rmsnorm_quant_reference(inputs: Dict[str, Any]) -> Any:
    """The launch's two writes side by side: q on the kernel's MXFP8 grid,
    then the KV row in its own dtype.

    The q normalization is deliberately rounded to the activation dtype before
    quantization (query_quant.py:63-65): that materialized boundary is part of
    the function the launch computes, not an oracle convenience.
    """
    import torch

    def norm(x, w):
        y = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + inputs["eps"]) * w.float()
        return y.to(x.dtype)

    q = _mxfp8_rows_on_grid(norm(inputs["q"], inputs["q_weight"]))
    kv = norm(inputs["kv"], inputs["kv_weight"]).float()
    return torch.cat([q, kv], dim=-1)


register_ref("fused_q_kv_rmsnorm_quant",
             RefSpec(_q_kv_rmsnorm_quant_inputs,
                     _q_kv_rmsnorm_quant_reference))


#: The gate activation the `rmsnorm_gated` cell is measured at. The span
#: takes silu (Qwen gated-delta-net) or sigmoid (Kimi-K3, GLM, Ling), a
#: compile-time switch in every kernel that serves it which reads and writes
#: the same bytes in the same launch — so one cell, at the kernels' own
#: default ("swish" in FLA's `layer_norm_fwd` and `rms_norm_gated`), and the
#: providers pass this same value so the cross-check compares one function.
RMSNORM_GATED_ACTIVATION = "swish"


def _rmsnorm_gated_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen),
        "z": torch.randn(*_shape(inv, "z"), dtype=dt, device=device, generator=gen),
        # The gain at the cell's weight dtype: Qwen3.5 stores its gated-norm
        # gain fp32, GLM bf16 — two cells, each billed for its own.
        "w": torch.randn(*_shape(inv, "w"), dtype=_operand_dtype(inv, "w"),
                         device=device, generator=gen),
        "eps": 1e-6,
        "activation": RMSNORM_GATED_ACTIVATION,
    }


def _rmsnorm_gated_reference(inputs: Dict[str, Any]) -> Any:
    """rmsnorm(x)·w · act(z) — the plan's `norm`, `act`, `gated` nodes."""
    import torch
    import torch.nn.functional as F
    x, z = inputs["x"].float(), inputs["z"].float()
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + inputs["eps"])
    y = y * inputs["w"].float()
    gate = (torch.sigmoid(z) if inputs["activation"] == "sigmoid"
            else F.silu(z))
    return y * gate


register_ref("rmsnorm_gated",
             RefSpec(_rmsnorm_gated_inputs, _rmsnorm_gated_reference))


def rms_normalise(x: Any, w: Any, eps: float) -> Any:
    """`x · rsqrt(mean(x²) + eps) · w` over the last axis, in fp32 — the
    one RMSNorm every oracle below composes, fused-norm and fused-span
    alike (`w=None`: no gain)."""
    import torch
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y if w is None else y * w.float()


def _norm_span_inputs(activations: Tuple[str, ...], gains: Tuple[str, ...],
                      scalar: bool = False):
    """Inputs of a fused norm span: each activation at the cell's dtype,
    each gain at the cell's weight dtype, and — for the Gemma-4 tails — the
    learned layer scalar, drawn positive as a scale is."""
    def make(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
        import torch
        dt = _dtype(inv.dtype)
        wdt = _store_dtype(inv.dtype_for("weight"))
        out: Dict[str, Any] = {
            name: torch.randn(*_shape(inv, name), dtype=dt, device=device,
                              generator=gen) for name in activations}
        out.update({name: torch.randn(*_shape(inv, name), dtype=wdt,
                                      device=device, generator=gen)
                    for name in gains})
        if scalar:
            out["scalar"] = 0.5 + torch.rand(*_shape(inv, "scalar"), dtype=wdt,
                                             device=device, generator=gen)
        out["eps"] = 1e-6
        return out
    return make


def _per_head(x: Any, d_head: int) -> Any:
    """`[tokens, heads·d_head]` viewed per head."""
    return x.reshape(x.shape[0], -1, d_head)


def _fused_eh_norm_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `enorm`, `hnorm`, `concat`: `[tokens, 2·d_model]`, the
    embedding's half first."""
    import torch
    eps = inputs["eps"]
    return torch.cat([rms_normalise(inputs["inputs_embeds"], inputs["enorm_w"], eps),
                      rms_normalise(inputs["hidden"], inputs["hnorm_w"], eps)],
                     dim=-1)


register_ref("fused_eh_norm",
             RefSpec(_norm_span_inputs(("inputs_embeds", "hidden"),
                                       ("enorm_w", "hnorm_w")),
                     _fused_eh_norm_reference))


def _fused_q_kv_rmsnorm_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's two sibling norms over each operand's last axis, written
    flat and side by side per token, the query's first: `[tokens, d_q_lora
    + d_kv_lora]` in the row form, `[tokens, (n_heads + n_kv_heads)·d_head]`
    in the per-head one (each head its own norm, `layouts_for`). One array
    because the cross-check compares one, and a launch that got either half
    wrong is wrong."""
    import torch
    eps = inputs["eps"]
    return torch.cat(
        [rms_normalise(inputs["q"], inputs["q_weight"], eps).flatten(1),
         rms_normalise(inputs["kv"], inputs["kv_weight"], eps).flatten(1)],
        dim=-1)


register_ref("fused_q_kv_rmsnorm",
             RefSpec(_norm_span_inputs(("q", "kv"), ("q_weight", "kv_weight")),
                     _fused_q_kv_rmsnorm_reference))


def _gemma_qkv_rmsnorm_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's three sibling per-head norms, written back flat and side
    by side `[tokens, (n_heads + 2·n_kv_heads)·d_head]`: q·w_q, k·w_k, and v
    with no gain."""
    import torch
    d, eps = int(inputs["q_weight"].shape[0]), inputs["eps"]
    parts = [rms_normalise(_per_head(inputs["q"], d), inputs["q_weight"], eps),
             rms_normalise(_per_head(inputs["k"], d), inputs["k_weight"], eps),
             rms_normalise(_per_head(inputs["v"], d), None, eps)]
    return torch.cat([p.flatten(1) for p in parts], dim=-1)


register_ref("gemma_qkv_rmsnorm",
             RefSpec(_norm_span_inputs(("q", "k", "v"), ("q_weight", "k_weight")),
                     _gemma_qkv_rmsnorm_reference))


def _gemma_rmsnorm_residual_scalar_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `norm`, `add`, `scale`: (rmsnorm(x)·w + residual)·scalar."""
    y = rms_normalise(inputs["x"], inputs["w"], inputs["eps"])
    return (y + inputs["residual"].float()) * inputs["scalar"].float()


register_ref("gemma_rmsnorm_residual_scalar",
             RefSpec(_norm_span_inputs(("x", "residual"), ("w",), scalar=True),
                     _gemma_rmsnorm_residual_scalar_reference))


def _gemma_dual_rmsnorm_residual_scalar_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's two branch norms, their sum normed, then the residual add
    and the scale: (rmsnorm(rmsnorm(x1)·w1 + rmsnorm(x2)·w2)·w3 + residual)
    ·scalar."""
    eps = inputs["eps"]
    branches = (rms_normalise(inputs["x1"], inputs["w1"], eps)
                + rms_normalise(inputs["x2"], inputs["w2"], eps))
    y = rms_normalise(branches, inputs["w3"], eps)
    return (y + inputs["residual"].float()) * inputs["scalar"].float()


register_ref("gemma_dual_rmsnorm_residual_scalar",
             RefSpec(_norm_span_inputs(("x1", "x2", "residual"),
                                       ("w1", "w2", "w3"), scalar=True),
                     _gemma_dual_rmsnorm_residual_scalar_reference))


def _allreduce_fused_rmsnorm_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The same tensors as `fused_add_rmsnorm`, plus the group the collective
    reduces over. The group changes no tensor's shape — every rank holds the
    same per-rank row — so it rides as a scalar the provider reads: a
    launch over `n_comm_ranks` ranks is what a provider has to set up, and
    the cell states it because the fabric traffic scales with it."""
    out = _fused_add_rmsnorm_inputs(inv, device, gen)
    out["n_comm_ranks"] = int(_dims(inv)["n_comm_ranks"])
    return out


register_ref("allreduce_fused_rmsnorm",
             # No reference: on one process the all-reduce is the identity,
             # so a residual+norm oracle would agree with a provider that
             # also omitted the collective — which is how a kernel that
             # never communicated cross-checked green. A provider that runs
             # `n_comm_ranks` processes brings its own.
             RefSpec(_allreduce_fused_rmsnorm_inputs, None))


# ---------------------------------------------------------------------------
# silu_and_mul: silu(gate) * up, input [M, 2I] split into gate|up
# ---------------------------------------------------------------------------

def _silu_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    x = torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen)
    return {"x": x, "I": int(x.shape[-1]) // 2,
            "cast": _quant_output_cast(inv, x.shape[-1] // 2)}


def with_kernel_params(base, *params: str):
    """`base`'s operands plus the scalars the kernel takes beside them, read
    off the cell (`Invocation.kernel_params`, filled from what the model
    publishes — `OpSignature.config_params`) so the reference and the
    provider get them from one place.

    A required scalar the cell does not carry raises. The signature
    declares it required, so a planned cell has it; a cell that does not
    must not be measured at a value chosen here — that is how a clamped
    SwiGLU came to be measured AND cross-checked at 7.0 while every model
    that runs it publishes 10.0, the two sides agreeing about a function
    nobody computes. A name ending in `?` is optional: the kernel has its
    own "unset" for it (SiTU's linear beta, where <= 0 means the up branch
    is passed through)."""
    def inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
        out = base(inv, device, gen)
        for name in params:
            key = name.rstrip("?")
            value = inv.kernel_params.get(key)
            if value is None:
                if name.endswith("?"):
                    continue
                raise KeyError(
                    f"{inv.signature} cell carries no {key!r}: the kernel "
                    f"takes it beside its operands and the model publishes "
                    f"it (OpSignature.config_params). Plan the cell, do not "
                    f"pick a value here.")
            out[key] = float(value)
        return out
    return inputs


def _silu_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    x = inputs["x"].float()
    I = inputs["I"]
    y = torch.nn.functional.silu(x[:, :I]) * x[:, I:]
    return y if inputs.get("cast") is None else inputs["cast"].on_grid(y)


register_ref("silu_and_mul", RefSpec(_silu_inputs, _silu_reference))


def _gelu_and_mul_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    x = inputs["x"].float()
    I = inputs["I"]
    y = torch.nn.functional.gelu(x[:, :I]) * x[:, I:]
    return y if inputs.get("cast") is None else inputs["cast"].on_grid(y)


register_ref("gelu_and_mul", RefSpec(_silu_inputs, _gelu_and_mul_reference))


def _silu_and_mul_with_clamp_reference(inputs: Dict[str, Any]) -> Any:
    """`silu(clamp(gate, max=limit)) * clamp(up, ±limit)` — vLLM's own
    reference for the kernel (layers/fused_moe/activation.py:300-304), at
    alpha 1 / beta 0, which is the form it calls for a packed w13. The
    bound is the model's (`config.swiglu_limit`: 10.0 on DeepSeek-V4 and
    GLM-5.3, 7.0 on gpt-oss), carried on the cell.
    """
    import torch
    x = inputs["x"].float()
    I = inputs["I"]
    limit = float(inputs["clamp_limit"])
    gate = torch.clamp(x[:, :I], max=limit)
    up = torch.clamp(x[:, I:], min=-limit, max=limit)
    y = gate * torch.sigmoid(gate) * up
    return y if inputs.get("cast") is None else inputs["cast"].on_grid(y)


def situ_betas(inputs: Dict[str, Any]) -> tuple:
    """`(beta, linear_beta)` for a SiTU gate, as the KERNEL encodes them:
    the gate's beta is mandatory, and a non-positive linear beta is the
    kernel's "unset", which passes the up branch through (vLLM writes
    `-1.0` for `linear_beta=None`). Declared once because the reference and
    both providers have to agree on the encoding, not just on the value."""
    return (float(inputs["situ_beta"]),
            float(inputs.get("situ_linear_beta", -1.0)))


def _situ_and_mul_reference(inputs: Dict[str, Any]) -> Any:
    """`situ(gate) * up`, Kimi-K3's gate, as vLLM's kernel computes it:
    `situ(x) = beta·tanh(x/beta)·sigmoid(x)`, and the up branch is passed
    through when the linear beta is unset (<= 0) and scaled through
    `tanh(up/linear_beta)·linear_beta` when it is set
    (fused_moe/activation.py:372-379). Kimi-K3 publishes beta 4.0 and
    linear beta 25.0; both ride the cell."""
    import torch
    x = inputs["x"].float()
    I = inputs["I"]
    beta, linear = situ_betas(inputs)
    gate, up = x[:, :I], x[:, I:]
    if linear > 0:
        up = linear * torch.tanh(up / linear)
    return beta * torch.tanh(gate / beta) * torch.sigmoid(gate) * up


register_ref("silu_and_mul_with_clamp",
             RefSpec(with_kernel_params(_silu_inputs, "clamp_limit"),
                     _silu_and_mul_with_clamp_reference))
register_ref("situ_and_mul",
             RefSpec(with_kernel_params(_silu_inputs, "situ_beta",
                                        "situ_linear_beta?"),
                     _situ_and_mul_reference))


# ---------------------------------------------------------------------------
# gqa_attention: canonical Q[B,S,N,D], K/V[B,S,Nk,D], causal. Reference is
# GQA (repeat KV) scaled-dot-product attention → [B,S,N,D].
# ---------------------------------------------------------------------------

def _causal(inv: Invocation) -> bool:
    """Whether a cell's queries are masked causally among themselves: a
    prefill, and a decode step carrying several query tokens (speculative
    verify) — each draft token sits at its own position and must not see the
    drafts after it. vLLM hands a multi-token decode to FlashInfer's paged
    prefill wrapper with causal=True, and FlashMLA's decode takes
    causal=(s_query > 1); a non-causal reference disagreed with both."""
    return inv.phase == "prefill" or int(_dims(inv)["s_query"]) > 1


def visible_keys(Sq: int, Lq: int, device: Any, *, causal: bool,
                 s_window=None) -> Any:
    """Which keys each query reads, as a bool `[Sq, Lq]` (True = attended).

    The one statement of the alignment every attention oracle masks with:
    the `Sq` queries are the LAST `Sq` positions of an `Lq`-long context, so
    query `i` sits at `Lq − Sq + i` and, causal, reads keys `0 .. Lq − Sq + i`
    (bottom-right). A one-token decode therefore reads the whole cache, a
    speculative-verify step of `k` tokens reads `Lq − k + i + 1` keys at token
    `i`, and a prefill with `Sq == Lq` is the plain lower triangle. A sliding
    window `W` further keeps only the last `W` of those keys. Not causal and
    no window: every key."""
    import torch
    # int32 positions: the [Sq,Lq] distance matrix is the widest tensor this
    # mask builds, and every sequence length here is far under 2**31 — at
    # 131072² the int64 spelling alone was 128 GiB of the OOM the queue logged.
    q_pos = torch.arange(Sq, device=device, dtype=torch.int32) + (Lq - Sq)
    k_pos = torch.arange(Lq, device=device, dtype=torch.int32)
    dist = q_pos[:, None] - k_pos[None, :]
    allow = torch.ones(Sq, Lq, dtype=torch.bool, device=device)
    if causal or s_window:
        allow &= dist >= 0
    if s_window and int(s_window) > 0:
        allow &= dist < int(s_window)
    return allow


def _sdpa_causal_args(q, k, causal: bool, s_window=None):
    """`is_causal` / `attn_mask` for SDPA, aligned the way the KERNELS align.

    `is_causal=True` masks TOP-LEFT when Sq != Lq — query 0 sees only key 0.
    FlashAttention (v2 >= 2.1, v3) and flashinfer mask BOTTOM-RIGHT: query 0
    sees keys 0..(Lq-Sq), which is the semantics of chunked prefill, where the
    queries are the LAST Sq positions of a longer context. Measured on an H100:
    FA2 agrees with bottom-right at cosine 0.999997 and with top-left at 0.236.

    `causal_lower_right` rather than a materialized bool mask: an explicit
    [Sq,Lq] mask forces SDPA's MATH backend, which is the O(S^2) path these
    references exist to avoid.

    Sliding-window (`s_window`) is the same band FA takes as
    `window_size=(W-1, 0)`: query i (bottom-right position Lq-Sq+i) attends
    the last W keys up to itself. The band is a dense [Sq, Lq] bias — the
    oracle is not the long-context path; a cell that would OOM here is
    measured without the cross-check.
    """
    Sq, Lq = q.shape[-2], k.shape[-2]
    if s_window is not None and int(s_window) > 0:
        import torch
        allow = visible_keys(Sq, Lq, q.device, causal=True, s_window=s_window)
        bias = torch.zeros(Sq, Lq, dtype=q.dtype, device=q.device)
        bias.masked_fill_(~allow, torch.finfo(q.dtype).min)
        return {"attn_mask": bias, "is_causal": False}
    if causal and Sq != Lq:
        from torch.nn.attention.bias import causal_lower_right
        return {"attn_mask": causal_lower_right(Sq, Lq), "is_causal": False}
    return {"is_causal": causal}


def _gqa_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    out = {
        "q": torch.randn(*_shape(inv, "q"), dtype=dt, device=device, generator=gen),
        "k": torch.randn(*_shape(inv, "k"), dtype=_operand_dtype(inv, "k"),
                         device=device, generator=gen),
        "v": torch.randn(*_shape(inv, "v"), dtype=_operand_dtype(inv, "v"),
                         device=device, generator=gen),
        "causal": _causal(inv),
    }
    # Presence, not `.get`: the recording test treats every `.get` as a
    # cell-axis read, and these are variants — absent on the default cell.
    raw = (inv.kernel_params or {}).get("s_window")
    if raw is None and "s_window" in inv.dims:
        raw = inv.dims["s_window"]
    if raw is not None:
        try:
            w = int(raw)
        except (TypeError, ValueError):
            w = 0
        if w > 0:
            out["s_window"] = w
    has_sinks = bool((inv.kernel_params or {}).get("sinks"))
    if not has_sinks and "n_attn_sinks" in inv.dims:
        has_sinks = True
    if has_sinks:
        n = int(inv.dims["n_heads"]) if "n_heads" in inv.dims else out["q"].shape[2]
        out["sinks"] = torch.randn(n, dtype=dt, device=device, generator=gen)
    return out


def _gqa_reference(inputs: Dict[str, Any]) -> Any:
    # Oracle = torch SDPA in the NATIVE (bf16) dtype, NOT a naive fp32 score
    # materialization. Two reasons, both about memory: (1) the naive path
    # materializes the [B,N,Sq,Lq] score matrix — O(S²), ~30 GB at B16/S4096 —
    # which OOMs at exactly the long-context cells we most want to measure;
    # SDPA's flash / mem-efficient backend is O(S). (2) those backends only
    # engage for bf16/fp16 — an fp32 oracle falls back to the math backend and
    # materializes anyway. bf16 SDPA (fp32 accumulate internally) is accurate
    # enough for a correctness oracle; the dtype tolerance covers the rest.
    #
    # The backend is PINNED rather than left to SDPA's own dispatch. Which
    # kernel computes the oracle is an implementation detail, but letting the
    # dispatcher choose makes the oracle's availability depend on the box: on
    # one H100 image torch reports cuDNN as available and then raises
    # `CUDNN_STATUS_NOT_INITIALIZED` on first use (an image/driver mismatch,
    # same family as that box's libcuda issue). Every `gqa_attention` reference
    # there failed, and the driver's designed response — measure without the
    # cross-check rather than lose the cell — meant the rows landed unverified.
    # A reference that silently stops verifying is worse than one that is slow,
    # so cuDNN is excluded and the O(S) backends are used in preference order,
    # with math only as the last resort.
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = inputs["q"].transpose(1, 2)   # [B,Sq,N,D] → [B,N,Sq,D]
    k = inputs["k"].transpose(1, 2)   # [B,Lq,Nk,D] → [B,Nk,Lq,D]
    v = inputs["v"].transpose(1, 2)

    # `is_causal=True` aligns the mask to the TOP-LEFT when Sq != Lq: query 0
    # would see only key 0. FlashAttention (v2 >= 2.1, v3) and flashinfer align
    # to the BOTTOM-RIGHT — query 0 sees keys 0..(Lq-Sq) — which is the right
    # semantics for chunked prefill, where the queries are the LAST Sq positions
    # of a longer context. Left as `is_causal`, the oracle computed a different
    # mask and rejected the kernels: 226 cells each for fa2 and fa3, every one
    # of them with Sq != Lq and none with Sq == Lq, and the two libraries agreed
    # with each other to the digit (rel_err 2.79, cosine 0.24241) while
    # disagreeing with the reference. When two independent implementations agree
    # and the oracle does not, the oracle is the one to check.
    #
    # `causal_lower_right` rather than a materialized bool mask: an explicit
    # [Sq,Lq] mask forces the MATH backend, which is the O(S²) path this
    # reference exists to avoid.
    cargs = _sdpa_causal_args(
        q, k, bool(inputs["causal"]), s_window=inputs.get("s_window"))
    sinks = inputs.get("sinks")
    if sinks is not None:
        import torch
        if q.shape[1] != k.shape[1]:
            group = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        allow = visible_keys(scores.shape[-2], scores.shape[-1], q.device,
                             causal=bool(inputs["causal"]),
                             s_window=inputs.get("s_window"))
        scores = scores.masked_fill(~allow, float("-inf"))
        # The sink is one more logit a head, in the SAME softmax as the keys
        # and then dropped from the value sum — the form gpt-oss defines and
        # `_mla_sparse_sink_reference` uses. Its exponent is shifted by the
        # row max like every other logit; added unshifted (`exp(sink)` over
        # `Σ exp(s − max)`), the oracle mis-scaled every row whose max was
        # not zero and rejected the sink kernels at cosine ~0.99.
        B, H, Sq, _Lq = scores.shape
        sink = sinks.float().reshape(1, H, 1, 1).expand(B, H, Sq, 1)
        probs = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
        out = torch.matmul(probs.to(v.dtype), v)
        return out.transpose(1, 2)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                      SDPBackend.EFFICIENT_ATTENTION,
                      SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            q, k, v, **cargs,
            enable_gqa=(q.shape[1] != k.shape[1]))
    return out.transpose(1, 2)        # [B,N,Sq,D] → [B,Sq,N,D]


register_ref("gqa_attention", RefSpec(_gqa_inputs, _gqa_reference,
                                      materializes_scores=True))


# ---------------------------------------------------------------------------
# mla_attention (absorbed / decode): canonical q[B,Sq,N,Dqk] + a SINGLE latent
# KV c[B,Lq,Dqk] shared across the N query heads (MQA-like). The value is the
# first Dv dims of that same latent (Dqk = Dv + rope, e.g. 576 = 512 + 64).
# Reference: scores = q·c^T over Dqk, softmax, out = attn·c[...,:Dv] → [B,Sq,N,Dv].
# ---------------------------------------------------------------------------

def _mla_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    from backend.inference.op_signatures import get_signature
    sig = get_signature(inv.signature)
    layouts = sig.layouts_for(_dims(inv)) if sig else ()
    # Each operand at the slot its role names: the latent is the cache, and
    # a cache kept fp32 is billed four bytes an element (`_operand_dtype`).
    out = {lay.name: torch.randn(*_shape(inv, lay.name),
                                 dtype=_operand_dtype(inv, lay.name),
                                 device=device, generator=gen)
           for lay in layouts}
    out["Dv"] = int(_dims(inv)["d_v_head"])
    out["causal"] = _causal(inv)
    return out


def _mla_reference(inputs: Dict[str, Any]) -> Any:
    """Composed from `scaled_dot_product_attention`, not hand-rolled math.

    An oracle written as explicit einsum + softmax is itself code that can be
    wrong, and it is the thing every measured kernel is judged against — so it
    is built out of the torch operator whose semantics ARE the definition,
    exactly like `_gqa_reference`.

    An oracle is allowed to be slow; that is not what this is about. It does
    have to FINISH: the previous form stalled a sweep for 30 minutes on a
    single cell and asked for 144 GiB on another. Both geometries below reduce
    to SDPA, and the absorbed one is arranged so the same O(S²) math does not
    also carry a needless N× memory multiplier.

    Both MLA geometries reduce to SDPA:

    * non-absorbed (prefill) — a plain grouped attention with a NARROWER value
      than query/key (Dqk=192, Dv=128). SDPA takes the asymmetry directly; the
      mem-efficient backend serves it in O(S).
    * absorbed (decode) — one latent `c` shared by every query head, and the
      value is that same latent's first Dv dims. Two steps make it a plain
      single-head attention:
        - the N query heads all read the SAME latent and each (position,
          head) query is independent (a causal mask is per position, so it
          folds with them) — so the head axis folds into the query axis and
          SDPA sees one head, `Sq·N` queries. This is
          what keeps it cheap: leaving N as a head axis against a 1-head KV
          makes the fallback broadcast the latent N times (measured: 144 GiB
          for B=64/N=128/Lq=8192, vs 0.03 GiB folded).
        - `c` is passed as both key and value so the widths are symmetric, and
          the answer is the leading Dv columns of the output — attention is a
          convex combination along Lq, so slicing the output equals combining
          the sliced value columns.
      Dqk=576 exceeds every O(S) backend's head_dim limit (256), so this one
      does run on MATH; folding is what keeps that affordable.

    cuDNN is excluded for the same reason as `_gqa_reference` (see there).
    """
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backends = [SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH]
    Dv = int(inputs["Dv"])
    causal = bool(inputs["causal"])
    if "k" in inputs:                       # non-absorbed: q/k/v, Dv < Dqk
        q = inputs["q"].transpose(1, 2)     # [B,Sq,N,Dqk] → [B,N,Sq,Dqk]
        k = inputs["k"].transpose(1, 2)     # [B,Lq,Nk,Dqk] → [B,Nk,Lq,Dqk]
        v = inputs["v"].transpose(1, 2)     # [B,Lq,Nk,Dv]  → [B,Nk,Lq,Dv]
        # Same alignment the kernels use — see `_gqa_reference`. This is the
        # non-absorbed MLA prefill geometry, so Sq != Lq is the chunked-prefill
        # case and `is_causal` would mask top-left while every kernel masks
        # bottom-right.
        with sdpa_kernel(backends):
            out = F.scaled_dot_product_attention(
                q, k, v, enable_gqa=(q.shape[1] != k.shape[1]),
                **_sdpa_causal_args(q, k, causal))
        return out.transpose(1, 2).contiguous()          # [B,Sq,N,Dv]
    # absorbed: q[B,Sq,N,Dqk] against the single latent c[B,Lq,Dqk].
    q = inputs["q"]
    B, Sq, N, Dqk = q.shape
    c = inputs["c"].unsqueeze(1)                         # [B,1,Lq,Dqk]
    # Fold the head axis into the query axis: every head reads the SAME
    # latent, so `Sq·N` queries against one head is the same computation.
    # Keeping N as a head axis instead would make the fallback broadcast the
    # latent N times — 144 GiB at B=64/N=128/Lq=8192, against 0.03 GiB folded.
    #
    # A speculative-verify step (`Sq` > 1) is causal, and the fold survives
    # it: the mask is per query POSITION, so row `i·N + n` of the folded
    # queries takes position `i`'s row of `visible_keys`. Bottom-right, like
    # the kernels — `is_causal=True` would mask top-left, and did: query 0 of
    # a 2-token verify read ONE key instead of `Lq − 1`, and FlashMLA and
    # FlashInfer both landed at cosine ~0.03 against it on every verify cell.
    mask = None
    if causal:
        mask = visible_keys(Sq, c.shape[-2], q.device, causal=True
                            ).repeat_interleave(N, dim=0)  # [Sq·N, Lq]
    with sdpa_kernel(backends):
        out = F.scaled_dot_product_attention(
            q.reshape(B, 1, Sq * N, Dqk), c, c, attn_mask=mask)
    return out[..., :Dv].reshape(B, Sq, N, Dv).contiguous()


register_ref("mla_attention", RefSpec(_mla_inputs, _mla_reference,
                                      materializes_scores=True))


# ---------------------------------------------------------------------------
# mla_sparse_attention (DeepSeek DSA / NSA): same absorbed latent geometry as
# mla_attention, but each query attends only `n_topk_keys` positions selected
# out of the `s_kv` cache. Canonical inputs add the selection itself as an
# int32 index tensor [B, Sq, topk] — the layout every production kernel takes
# (FlashMLA `flash_mla_sparse_fwd(q, kv, indices, …)`, sglang's NSA backend).
#
# The index set is SCATTERED over the cache rather than a contiguous run: the
# kernel's cost is dominated by gathering scattered rows, and a contiguous
# selection would measure a coalesced read the model never performs. The k-th
# selected row is `k · stride mod s_kv` with the stride the golden-ratio step
# of the cache (a Weyl sequence), bumped to be coprime with `s_kv`: every
# selection is distinct, consecutive picks land ~0.38·s_kv apart, and that
# holds when the selection is the whole cache too — the cell keys the cache at
# the rows a query attends (`_dsa_identity`), so top-k == s_kv is the common
# case, where an evenly spaced stride of `s_kv // topk` degenerates to 1 and
# the "gather" was one sequential read. Each query gets its own offset so no
# two queries share a pattern. Selection is not causally masked — in
# production the indexer's output already encodes causality, and the kernel
# applies no mask of its own.
#
# What this still simplifies: the cache built here IS the attended rows. In
# serving a query picks its top-k out of the whole stored cache (the walk's
# storage positions, `kv_storage_positions_per_query_fn = s_kv`), which no
# cell axis carries — the cell's `s_kv` is the cache operand's rows,
# `Min(s_kv, k)`. So the launch gathers the right rows in a scattered order
# but out of a pool no larger than they are, and across queries of a prefill
# step that pool can sit in L2. Follow-up: materialize the stored length as
# its own axis beside the cache operand in the walk, key the cell by it, and
# draw the selection out of that larger pool.
# ---------------------------------------------------------------------------

#: Bytes one block of a per-query gather may hold before the query axis is
#: walked in blocks: the sparse-attention oracles below spend it on their
#: fp32 gathers, and the torch floors that time the same gather
#: (`providers/attention.py`) on theirs, at the cell's own dtype. Public for
#: the reason `SCAN_MAX_STEPS` is: one fact, not one number per reader.
GATHER_BLOCK_BYTES = 256 * 1024 * 1024


def _scatter_stride(Lq: int) -> int:
    """The golden-ratio step over `Lq` rows, coprime with `Lq` so `k · step
    mod Lq` visits distinct rows for every k < Lq."""
    import math
    if Lq <= 2:
        return 1
    step = max(2, round(Lq * 0.6180339887))
    while math.gcd(step, Lq) != 1:
        step += 1
    return step


def _mla_sparse_indices(B: int, Sq: int, Lq: int, topk: int, device: str):
    import torch
    stride = _scatter_stride(Lq)
    # Each term reduced mod Lq first, so the [Sq, topk] sum is formed once,
    # already int32 (every term < Lq < 2**31), and reduced in place: at the
    # 262144x2048 prefill cell the int64 spelling held three 4 GiB
    # temporaries and the inputs OOMed a 24 GiB card before any launch.
    j = (torch.arange(topk, dtype=torch.int64, device=device) * stride
         % Lq).to(torch.int32)
    s = (torch.arange(Sq, dtype=torch.int64, device=device)
         % Lq).to(torch.int32).unsqueeze(1)
    idx = (j.unsqueeze(0) + s).remainder_(Lq)              # [Sq, topk]
    return idx.unsqueeze(0).expand(B, Sq, topk).contiguous()


def _mla_sparse_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    d = _dims(inv)
    from backend.inference.op_signatures import get_signature
    if get_signature(inv.signature).binds_variant(d):
        # The per-head geometry (`n_kv_heads` bound): top-k GQA, K and V
        # cached per KV head — the gated span's operands without its gate.
        return _gqa_sparse_inputs(inv, device, gen)
    q = torch.randn(*_shape(inv, "q"), dtype=_dtype(inv.dtype), device=device, generator=gen)
    c = torch.randn(*_shape(inv, "c"), dtype=_operand_dtype(inv, "c"),
                    device=device, generator=gen)
    B, Sq, _N, _D = q.shape
    Lq = int(c.shape[1])
    topk = min(int(d["n_topk_keys"]), Lq)
    return {
        "q": q,
        "c": c,
        "indices": _mla_sparse_indices(B, Sq, Lq, topk, device),
        "Dv": int(d["d_v_head"]),
    }


def _mla_sparse_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    if "k" in inputs:
        return _gqa_sparse_core(inputs).contiguous()
    q, c = inputs["q"].float(), inputs["c"].float()
    idx = inputs["indices"].long()                       # [B,Sq,topk]
    B, Sq, _N, Dqk = q.shape
    topk = idx.shape[2]
    Dv = inputs["Dv"]
    scale = 1.0 / (Dqk ** 0.5)
    # Per-query gathers are the whole point of this kernel, but the gathered
    # tensor is [B,Sq,topk,Dqk] — terabytes at prefill shapes. Walk the query
    # axis in blocks sized to a fixed budget so the oracle stays usable at the
    # shapes the kernel actually runs, instead of OOMing into no-cross-check.
    budget = GATHER_BLOCK_BYTES                                 # bytes, fp32
    blk = max(1, min(Sq, budget // max(1, B * topk * Dqk * 4)))
    outs = []
    for s0 in range(0, Sq, blk):
        s1 = min(Sq, s0 + blk)
        ib = idx[:, s0:s1]                                      # [B,blk,topk]
        sel = torch.gather(
            c.unsqueeze(1).expand(B, s1 - s0, c.shape[1], Dqk), 2,
            ib.unsqueeze(-1).expand(B, s1 - s0, topk, Dqk))      # [B,blk,topk,Dqk]
        scores = torch.einsum("bsnd,bstd->bnst", q[:, s0:s1], sel) * scale
        attn = scores.softmax(dim=-1)
        outs.append(torch.einsum("bnst,bstd->bnsd", attn, sel[..., :Dv]))
    out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]  # [B,N,Sq,Dv]
    return out.transpose(1, 2).contiguous()                     # [B,Sq,N,Dv]


register_ref("mla_sparse_attention",
             RefSpec(_mla_sparse_inputs, _mla_sparse_reference,
                     materializes_scores=True))


def _mega_attention_inputs(inv: Invocation, device: str,
                           gen: Any) -> Dict[str, Any]:
    import torch
    out = _mla_sparse_inputs(inv, device, gen)
    B, Sq, N, _D = out["q"].shape
    rd = int(_dims(inv)["d_rope"])
    max_pos = max(B * Sq, 8192)
    table = _rope_cos_sin_cache(
        torch.arange(max_pos, device=device), rd, 10000.0)
    out.update({
        "sinks": 0.1 * torch.randn(N, dtype=torch.float32, device=device,
                                   generator=gen),
        "positions": torch.arange(B * Sq, dtype=torch.int64,
                                  device=device) % max_pos,
        "cos_sin_cache": table,
        "d_rope": rd,
        "output": inv.dtype_for("output"),
    })
    return out


def _mega_attention_reference(inputs: Dict[str, Any]) -> Any:
    """The mega launch's math: rotate q, sparse sink attention, un-rotate o,
    then put the result on the MXFP8 grid the launch writes."""
    import torch
    q = inputs["q"]
    B, Sq, N, D = q.shape
    rd = int(inputs["d_rope"])
    q = q.reshape(B * Sq, N, D)
    nope, rot = q[..., :-rd], q[..., -rd:]
    rot = _rope_rotate(rot, inputs["positions"], inputs["cos_sin_cache"],
                       rd, is_neox=False).to(q.dtype)
    rotated = torch.cat([nope, rot], dim=-1).float().reshape(B, Sq, N, D)
    # The sparse reference has no sink operand; repeat its blocked gather
    # here with the launch's per-head sink logits in the softmax denominator.
    c, idx = inputs["c"].float(), inputs["indices"].long()
    Dv, topk = int(inputs["Dv"]), idx.shape[-1]
    scale = D ** -0.5
    outs = []
    budget = GATHER_BLOCK_BYTES
    block = max(1, budget // max(1, B * topk * D * 4))
    for s0 in range(0, Sq, block):
        s1 = min(Sq, s0 + block)
        sel = torch.gather(
            c.unsqueeze(1).expand(B, s1 - s0, c.shape[1], D), 2,
            idx[:, s0:s1].unsqueeze(-1).expand(B, s1 - s0, topk, D))
        scores = (torch.einsum("bsnd,bstd->bnst",
                               rotated[:, s0:s1], sel) * scale)
        scores = scores + inputs["sinks"].float()[None, :, None, None]
        attn = scores.softmax(dim=-1)
        outs.append(torch.einsum("bnst,bstd->bnsd", attn, sel[..., :Dv])
                    .transpose(1, 2))
    sparse = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
    # Conjugate frequencies: negate the sine half of the shared table.
    half = rd // 2
    inverse_table = torch.cat(
        (inputs["cos_sin_cache"][:, :half],
         -inputs["cos_sin_cache"][:, half:]), dim=-1)
    flat = sparse.reshape(B * Sq, N, D)
    nope, rot = flat[..., :-rd], flat[..., -rd:]
    rot = _rope_rotate(rot, inputs["positions"], inverse_table, rd,
                       is_neox=False).to(q.dtype)
    out = torch.cat([nope, rot], dim=-1).reshape(B, Sq, N, D)
    return _mxfp8_rows_on_grid(out)


register_ref("fused_mla_norm_rope_attn_rope_cast",
             RefSpec(_mega_attention_inputs, _mega_attention_reference))


# ---------------------------------------------------------------------------
# The two remaining sparse-attention spans whose launch is not plain
# `mla_sparse_attention`: Hy4's learnable sink, and Qwen QSA's output gate.
# ---------------------------------------------------------------------------

def _mla_sparse_sink_inputs(inv: Invocation, device: str,
                            gen: Any) -> Dict[str, Any]:
    import torch
    q = torch.randn(*_shape(inv, "q"), dtype=_dtype(inv.dtype),
                    device=device, generator=gen)
    c = torch.randn(*_shape(inv, "c"), dtype=_operand_dtype(inv, "c"),
                    device=device, generator=gen)
    sinks = torch.randn(*_shape(inv, "sinks"),
                        dtype=_operand_dtype(inv, "sinks"),
                        device=device, generator=gen)
    B, Sq, _N, _D = q.shape
    Lq = int(c.shape[1])
    topk = min(int(_dims(inv)["n_topk_keys"]), Lq)
    return {
        "q": q,
        "c": c,
        "indices": _mla_sparse_indices(B, Sq, Lq, topk, device),
        "sinks": sinks,
        "Dv": int(_dims(inv)["d_v_head"]),
    }


def _mla_sparse_sink_reference(inputs: Dict[str, Any]) -> Any:
    """Sink-softmax over the selected latent rows, in fp32 query blocks.

    FlashMLA states the same transform as `out *= exp(lse) /
    (exp(lse) + exp(sink))` (vLLM models/hy_v4/nvidia/flashmla_sparse.py:
    45-52); concatenating the sink score before softmax is that expression
    with one stable denominator.
    """
    import torch
    q, c = inputs["q"].float(), inputs["c"].float()
    idx = inputs["indices"].long()
    sinks = inputs["sinks"].float()
    B, Sq, N, Dqk = q.shape
    topk = idx.shape[2]
    Dv = int(inputs["Dv"])
    scale = 1.0 / (Dqk ** 0.5)
    budget = GATHER_BLOCK_BYTES
    blk = max(1, min(Sq, budget // max(1, B * topk * Dqk * 4)))
    outs = []
    for s0 in range(0, Sq, blk):
        s1 = min(Sq, s0 + blk)
        ib = idx[:, s0:s1]
        sel = torch.gather(
            c.unsqueeze(1).expand(B, s1 - s0, c.shape[1], Dqk), 2,
            ib.unsqueeze(-1).expand(B, s1 - s0, topk, Dqk))
        scores = torch.einsum("bsnd,bstd->bnst", q[:, s0:s1], sel) * scale
        # One learned score a head, then the sink column leaves with zero
        # contribution to the value sum.
        scores = torch.cat([
            scores, sinks.view(1, N, 1, 1).expand(B, N, s1 - s0, 1)], -1)
        attn = scores.softmax(dim=-1)[..., :topk]
        outs.append(torch.einsum("bnst,bstd->bnsd", attn, sel[..., :Dv]))
    out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]
    return out.transpose(1, 2).contiguous()


register_ref("mla_sparse_sink_attention",
             RefSpec(_mla_sparse_sink_inputs, _mla_sparse_sink_reference))


def _gqa_sparse_inputs(inv: Invocation, device: str,
                       gen: Any) -> Dict[str, Any]:
    """Top-k GQA's operands: Q, K and V per KV head, one selection a query
    shared by its heads (the indexer scores keys, not heads)."""
    import torch
    dt = _dtype(inv.dtype)
    q = torch.randn(*_shape(inv, "q"), dtype=dt, device=device, generator=gen)
    k = torch.randn(*_shape(inv, "k"), dtype=_operand_dtype(inv, "k"),
                    device=device, generator=gen)
    v = torch.randn(*_shape(inv, "v"), dtype=_operand_dtype(inv, "v"),
                    device=device, generator=gen)
    B, Sq, _N, _D = q.shape
    Lq = int(k.shape[1])
    topk = min(int(_dims(inv)["n_topk_keys"]), Lq)
    return {"q": q, "k": k, "v": v,
            "indices": _mla_sparse_indices(B, Sq, Lq, topk, device)}


def _gqa_sparse_gated_inputs(inv: Invocation, device: str,
                             gen: Any) -> Dict[str, Any]:
    import torch
    out = _gqa_sparse_inputs(inv, device, gen)
    out["gate"] = torch.randn(*_shape(inv, "gate"), dtype=_dtype(inv.dtype),
                              device=device, generator=gen)
    return out


def _gqa_sparse_core(inputs: Dict[str, Any]) -> Any:
    """Sparse GQA in fp32 query blocks: each query's selected K/V rows of its
    KV head, every query head of the group over them. `[B, Sq, N, Dv]`."""
    import torch
    q, k, v = (inputs[n].float() for n in ("q", "k", "v"))
    idx = inputs["indices"].long()
    B, Sq, N, D = q.shape
    _B, Lq, Nkv, _D = k.shape
    Dv = v.shape[3]
    topk = idx.shape[2]
    scale = 1.0 / (D ** 0.5)
    repeat = N // Nkv
    budget = GATHER_BLOCK_BYTES
    blk = max(1, min(Sq, budget // max(1, B * topk * N * max(D, Dv) * 4)))
    outs = []
    for s0 in range(0, Sq, blk):
        s1 = min(Sq, s0 + blk)
        ib = idx[:, s0:s1]
        ks = torch.gather(
            k.unsqueeze(1).expand(B, s1 - s0, Lq, Nkv, D), 2,
            ib.unsqueeze(-1).unsqueeze(-1).expand(B, s1 - s0, topk, Nkv, D))
        vs = torch.gather(
            v.unsqueeze(1).expand(B, s1 - s0, Lq, Nkv, Dv), 2,
            ib.unsqueeze(-1).unsqueeze(-1).expand(B, s1 - s0, topk, Nkv, Dv))
        # Gather returns [B, block, selected keys, KV heads, D]; put the
        # heads beside Q's before expanding each shared head to its group.
        ks = ks.permute(0, 1, 3, 2, 4).repeat_interleave(repeat, dim=2)
        vs = vs.permute(0, 1, 3, 2, 4).repeat_interleave(repeat, dim=2)
        # Gathered K/V are token-first; the contraction is over each query
        # position's selected keys, with the query's own token left explicit.
        vs = vs.permute(0, 2, 1, 3, 4)
        scores = torch.einsum("bsnd,bsntd->bnst", q[:, s0:s1], ks) * scale
        attn = scores.softmax(dim=-1)
        o = torch.einsum("bnst,bnstd->bnsd", attn, vs)
        outs.append(o.transpose(1, 2))
    return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]


def _gqa_sparse_gated_reference(inputs: Dict[str, Any]) -> Any:
    """Sparse GQA followed by QSA's sigmoid output gate."""
    import torch
    out = _gqa_sparse_core(inputs) * torch.sigmoid(inputs["gate"].float())
    return out.contiguous()


register_ref("gqa_sparse_gated_attention",
             RefSpec(_gqa_sparse_gated_inputs, _gqa_sparse_gated_reference))


# ---------------------------------------------------------------------------
# rope: rotate Q[tokens,N,D] and K[tokens,Nk,D] by cos/sin selected at each
# token's position. Canonical q/k are flat [tokens, N·D] / [tokens, Nk·D] (the
# layout the vLLM/sglang kernels take); rotary_dim == D (full-head rotation,
# the LLaMA/Qwen case). Reference = neox/interleaved rotate of the first
# rotary_dim; the correctness gate compares the concatenated [q‖k] output.
# ---------------------------------------------------------------------------

def _rope_cos_sin_cache(positions: Any, rotary_dim: int, base: float):
    """The fp32 `[cos | sin]` table for `positions` (any shape — mrope's
    three rows included): `[positions.max() + 1, rotary_dim]`, the rows the
    cell's positions reach, derived here once so no input builder sizes it.

    Not serving's shape. Serving hands a kernel the whole
    `[max_position_embeddings, rotary_dim]` table, but a rope launch gathers
    one row per position, so the rows past the last position change no
    launch's work (on the H100 sglang's kernel is bitwise equal over a
    13-row and an 8192-row table). The size is chosen for the launch gate,
    which reads the widest buffer a launch is handed as the precision its
    row prices (`launch_trace.disagreement`): padded to 8192 rows, an fp32
    table outgrew q and k at every decode cell, and a kernel that keeps the
    table fp32 by design — sglang's `apply_rope_with_cos_sin_cache_inplace`
    (srt/layers/rotary_embedding/base.py:112-117: "cache needs to be in
    FP32"), flashinfer's — was refused as "another precision's" latency on
    every model. A serving-sized table would need the gate to stop counting
    auxiliary tables first."""
    import torch
    device = positions.device
    if positions.is_meta:
        # A meta build is shapes only. Positions arrive as `arange`, whose
        # 1-D length is its own last row (`max + 1`), so the table keeps
        # the element count the emitted operand states without reading the
        # data a meta tensor does not carry. Any other positions on meta
        # must state its row count at the caller, not read it here.
        if positions.ndim != 1:
            raise ValueError(
                "cos/sin table on meta positions needs 1-D arange "
                "positions; state the table's rows at the caller")
        rows = int(positions.numel())
    else:
        rows = int(positions.max()) + 1
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2,
                                            dtype=torch.float32, device=device) / rotary_dim))
    t = torch.arange(rows, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)                     # [rows, rotary_dim/2]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [rows, rotary_dim] fp32


def _rope_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    d = _dims(inv)
    q = torch.randn(*_shape(inv, "q"), dtype=_dtype(inv.dtype), device=device, generator=gen)
    k = torch.randn(*_shape(inv, "k"), dtype=q.dtype, device=device, generator=gen)
    N = int(d["n_heads"]); Nk = int(d.get("n_kv_heads", N))
    D = int(d["d_rope"])
    tok = int(q.shape[0])
    positions = torch.arange(tok, device=device)
    return {
        "positions": positions,
        "q": q,
        "k": k,
        "cos_sin_cache": _rope_cos_sin_cache(positions, D, 10000.0),
        "N": N, "Nk": Nk, "D": D, "rotary_dim": D, "is_neox": True,
    }


def _rope_rotate(x, positions, cache, rotary_dim: int, is_neox: bool):
    """x: [tokens, H, D] float → RoPE-rotated first `rotary_dim`, rest passthrough."""
    import torch
    cos = cache[positions][:, : rotary_dim // 2][:, None, :]   # [tokens,1,rd/2]
    sin = cache[positions][:, rotary_dim // 2:][:, None, :]
    xr, xp = x[..., :rotary_dim], x[..., rotary_dim:]
    if is_neox:
        x1, x2 = xr[..., : rotary_dim // 2], xr[..., rotary_dim // 2:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:                                                 # GPT-J interleaved
        x1, x2 = xr[..., 0::2], xr[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2)
    return torch.cat([out, xp], dim=-1)


def _rope_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    N, Nk, D, rd = inputs["N"], inputs["Nk"], inputs["D"], inputs["rotary_dim"]
    pos, cache, neox = inputs["positions"], inputs["cos_sin_cache"], inputs["is_neox"]
    q = _rope_rotate(inputs["q"].float().view(-1, N, D), pos, cache, rd, neox)
    k = _rope_rotate(inputs["k"].float().view(-1, Nk, D), pos, cache, rd, neox)
    return torch.cat([q.reshape(q.shape[0], N * D), k.reshape(k.shape[0], Nk * D)], dim=1)


register_ref("rope", RefSpec(_rope_inputs, _rope_reference))


# ---------------------------------------------------------------------------
# DSA attention's input preparation (`fused_dsa_qkv_norm_rope{,_indexer}`):
# the two a-norms, the rope key's GPT-J rotation, and on an indexer layer
# the indexer key's LayerNorm and rotation of its leading `d_rope` channels.
#
# The indexer key's pairing is a kernel constexpr (`INDEX_ROPE_INTERLEAVE`:
# GLM-5.2 pairs adjacent channels, DeepSeek-V3.2 pairs across halves) that
# reads and writes the same bytes in the same launch — so one cell, measured
# at the NeoX pairing, and stated here once so the oracle and every provider
# compute the one function (the treatment `rmsnorm_gated`'s gate activation
# gets).
# ---------------------------------------------------------------------------

#: The indexer key's rotation pairing the cell is measured at (NeoX).
DSA_INDEX_ROPE_INTERLEAVE = False


def _dsa_norm_rope_inputs(indexer: bool):
    def make(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
        import torch
        dims = _dims(inv)
        dt = _dtype(inv.dtype)
        wdt = _store_dtype(inv.dtype_for("weight"))
        acts = ("q", "kv", "k_pe") + (("ik",) if indexer else ())
        gains = (("q_weight", "kv_weight")
                 + (("ik_weight", "ik_bias") if indexer else ()))
        out: Dict[str, Any] = {
            n: torch.randn(*_shape(inv, n), dtype=dt, device=device,
                           generator=gen) for n in acts}
        out.update({n: torch.randn(*_shape(inv, n), dtype=wdt, device=device,
                                   generator=gen) for n in gains})
        tokens = int(out["q"].shape[0])
        out["positions"] = torch.arange(tokens, dtype=torch.int64,
                                        device=device)
        out["cos_sin_cache"] = _rope_cos_sin_cache(
            out["positions"], int(dims["d_rope"]), 10000.0)
        out["eps"] = 1e-6
        if indexer:
            out["n_topk_keys"] = int(dims["n_topk_keys"])
            out["index_rope_interleave"] = DSA_INDEX_ROPE_INTERLEAVE
        return out
    return make


def _dsa_norm_rope_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's nodes, every port side by side `[tokens, d_q_lora +
    d_kv_lora + d_rope (+ d_indexer_head)]`: the normed query, the normed
    latent, the rotated rope key and — on an indexer layer — the indexer key
    normed and rotated. The top-k buffer reset is a constant fill and is not
    compared."""
    import torch
    import torch.nn.functional as F
    eps = inputs["eps"]
    pos, cache = inputs["positions"], inputs["cos_sin_cache"]
    rd = int(cache.shape[-1])
    parts = [rms_normalise(inputs["q"], inputs["q_weight"], eps),
             rms_normalise(inputs["kv"], inputs["kv_weight"], eps),
             _rope_rotate(inputs["k_pe"].float()[:, None, :], pos, cache, rd,
                          is_neox=False)[:, 0]]
    if "ik" in inputs:
        ik = F.layer_norm(inputs["ik"].float(), (inputs["ik"].shape[-1],),
                          inputs["ik_weight"].float(), inputs["ik_bias"].float(),
                          eps)
        parts.append(_rope_rotate(ik[:, None, :], pos, cache, rd,
                                  is_neox=not inputs["index_rope_interleave"])
                     [:, 0])
    return torch.cat(parts, dim=-1)


def fp8_rows_on_grid(x: Any, eps: float = 1e-4) -> Any:
    """`x` as FP8 e4m3 holds it with ONE exact fp32 scale per last-axis row
    (`max(amax, eps) / 448`) — the recipe sglang's DSA indexer kernels quantise
    with (indexer_k.cuh:330, main_norm_rope.cuh `fused_q_indexer_rope_hadamard_
    quant`). Through `quant.quantise_groups`, the one implementation of a
    grid, so a quantising span's oracle is that grid and not a second opinion
    of it.

    `x` is handed over at its own dtype: `quantise_groups` reads the amax
    there and widens after, the same numbers without an fp32 copy of the
    whole row — a copy the torch floors that call this inside their timed
    chain would otherwise launch as their widest operand."""
    from .quant import FORMATS, quantise_groups
    q, scale = quantise_groups(x, FORMATS["fp8_e4m3"], x.shape[-1],
                               "exact", x.dtype, eps=eps)
    return (q.float() * scale.unsqueeze(2)).reshape(x.shape)


def _indexer_rope_first(x: Any, inputs: Dict[str, Any]) -> Any:
    """`x[tokens, heads, d]` with its LEADING `d_rope` channels rotated,
    adjacent channels paired — both sglang DSA indexer kernels' rotation."""
    cache = inputs["cos_sin_cache"]
    return _rope_rotate(x, inputs["positions"], cache, int(cache.shape[-1]),
                        is_neox=False)


def _indexer_k_store_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dims = _dims(inv)
    dt = _dtype(inv.dtype)
    wdt = _store_dtype(inv.dtype_for("weight"))
    k = torch.randn(*_shape(inv, "k"), dtype=dt, device=device, generator=gen)
    tokens, page = int(k.shape[0]), int(dims["kv_block_size"])
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    return {
        "k": k,
        "k_weight": torch.randn(*_shape(inv, "k_weight"), dtype=wdt,
                                device=device, generator=gen),
        "k_bias": torch.randn(*_shape(inv, "k_bias"), dtype=wdt,
                              device=device, generator=gen),
        # This step's records written in order, as `fused_qnorm_rope_kv_insert`
        # writes its own: a served step writes the same one record per token
        # at whatever slots its sequences are up to.
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, int(dims["d_rope"]), 10000.0),
        "index_cache": torch.zeros(-(-tokens // page), page, int(k.shape[1]),
                                   dtype=_store_dtype(inv.kv_dtype), device=device),
        "kv_block_size": page,
        "eps": 1e-6,
        # The record the oracle writes the key in (`fp8_rows_on_grid`: one
        # F32 scale a 128-wide key), as the cell names it: the comparison
        # is record to record (`driver.compared_records`).
        "output_format": inv.output_format,
    }


def _indexer_k_store_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `norm`, `rope`, `cat`, then the record the cache keeps: the
    key rounded to its activation dtype (the kernel rounds through it before
    the scale, :319-321) and placed on the FP8 row grid. `[tokens,
    d_indexer_head]`, the records in slot order."""
    import torch.nn.functional as F
    k = inputs["k"]
    normed = F.layer_norm(k.float(), (k.shape[-1],), inputs["k_weight"].float(),
                          inputs["k_bias"].float(), inputs["eps"])
    rotated = _indexer_rope_first(normed[:, None, :], inputs)[:, 0]
    return fp8_rows_on_grid(rotated.to(k.dtype))


def _indexer_q_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dims = _dims(inv)
    dt = _dtype(inv.dtype)
    q = torch.randn(*_shape(inv, "q"), dtype=dt, device=device, generator=gen)
    tokens = int(q.shape[0])
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    return {
        "q": q,
        # Positive, so a provider can read each head's scale back off the
        # weight it was folded into (`w · weight_scale · scale`).
        "weights": 0.5 + torch.rand(*_shape(inv, "weights"), dtype=dt,
                                    device=device, generator=gen),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, int(dims["d_rope"]), 10000.0),
        "weight_scale": 1.0,
    }


def _indexer_q_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `q_rope` and `q_cat` per head, then the query as FP8 holds
    it with one scale per head row: `[tokens, n_indexer_heads·d_indexer_head]`.
    The weights' rescale is what that scale is FOR, and is checked by
    reading the scale back, not compared."""
    rotated = _indexer_rope_first(inputs["q"].float(), inputs)
    return fp8_rows_on_grid(rotated).flatten(1)


register_ref("fused_indexer_k_norm_rope_store",
             RefSpec(_indexer_k_store_inputs, _indexer_k_store_reference))
register_ref("fused_indexer_q_rope_first_quant",
             RefSpec(_indexer_q_inputs, _indexer_q_reference))


register_ref("fused_dsa_qkv_norm_rope",
             RefSpec(_dsa_norm_rope_inputs(False), _dsa_norm_rope_reference))
register_ref("fused_dsa_qkv_norm_rope_indexer",
             RefSpec(_dsa_norm_rope_inputs(True), _dsa_norm_rope_reference))


def _rope_partial_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """`rope`'s operands with heads of `d_head` and a rotation of `d_rope`:
    the cache is `d_rope` wide, the heads keep their full width, and the
    oracle (`_rope_reference`) rotates the first `rotary_dim` of each."""
    import torch
    d = _dims(inv)
    q = torch.randn(*_shape(inv, "q"), dtype=_dtype(inv.dtype), device=device, generator=gen)
    k = torch.randn(*_shape(inv, "k"), dtype=q.dtype, device=device, generator=gen)
    D, rd = int(d["d_head"]), int(d["d_rope"])
    tok = int(q.shape[0])
    positions = torch.arange(tok, device=device)
    return {
        "positions": positions,
        "q": q, "k": k,
        "cos_sin_cache": _rope_cos_sin_cache(positions, rd, 10000.0),
        "N": int(d["n_heads"]), "Nk": int(d["n_kv_heads"]), "D": D,
        "rotary_dim": rd, "is_neox": True,
    }


register_ref("rope_partial", RefSpec(_rope_partial_inputs, _rope_reference))


def _rope_partial_q_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    d = _dims(inv)
    q = torch.randn(*_shape(inv, "q"), dtype=_dtype(inv.dtype), device=device, generator=gen)
    k_pe = torch.randn(*_shape(inv, "k_pe"), dtype=q.dtype, device=device, generator=gen)
    rd = int(d["d_rope"])
    tok = int(q.shape[0])
    positions = torch.arange(tok, device=device)
    return {
        "positions": positions,
        "q": q, "k_pe": k_pe,
        "cos_sin_cache": _rope_cos_sin_cache(positions, rd, 10000.0),
        "N": int(d["n_heads"]), "D": int(d["d_head"]), "rotary_dim": rd,
        "is_neox": True,
    }


def rope_partial_q_rotate(q, k_pe, positions, cache, N: int, D: int, rd: int,
                          is_neox: bool):
    """The query's last `rd` of each head rotated, the rope key whole —
    the plan's `q_nope`/`q_pe` slices, the two ropes and `q_cat`.
    Returns `[q ‖ k_pe]` flat in fp32."""
    import torch
    qh = q.float().view(-1, N, D)
    tail = _rope_rotate(qh[..., D - rd:], positions, cache, rd, is_neox)
    q_out = torch.cat([qh[..., :D - rd], tail], dim=-1)
    k_out = _rope_rotate(k_pe.float().view(-1, 1, rd), positions, cache, rd,
                         is_neox)
    return torch.cat([q_out.reshape(q_out.shape[0], N * D),
                      k_out.reshape(k_out.shape[0], rd)], dim=1)


def _rope_partial_q_reference(inputs: Dict[str, Any]) -> Any:
    return rope_partial_q_rotate(inputs["q"], inputs["k_pe"], inputs["positions"],
                                 inputs["cos_sin_cache"], inputs["N"],
                                 inputs["D"], inputs["rotary_dim"],
                                 inputs["is_neox"])


register_ref("rope_partial_q",
             RefSpec(_rope_partial_q_inputs, _rope_partial_q_reference))


def _mrope_sections(inv: Invocation):
    d = _dims(inv)
    sections = tuple(int(d[name]) for name in
                     ("mrope_section_t", "mrope_section_h",
                      "mrope_section_w"))
    rd = int(d["d_rope"])
    if sum(sections) != rd // 2:
        raise ValueError(
            f"mrope sections {sections} must sum to d_rope/2={rd // 2}")
    return sections, int(d["mrope_interleaved"])


def _mrope_axis(sections, interleaved: bool):
    """Which position stream owns each rotary pair, exactly as the kernels'
    masks do: packed thresholds for Qwen3.5, modulo-three lanes for Qwen3-VL."""
    import torch
    t, h, w = sections
    half = t + h + w
    if not interleaved:
        return torch.tensor([0] * t + [1] * h + [2] * w)
    lane = torch.arange(half)
    axis = torch.zeros(half, dtype=torch.int64)
    # The upper bounds are the kernels' own `<= 3*section` tests
    # (vLLM mrope.py:68-74, sglang rotary_triton.py:56-59).
    axis[(lane % 3 == 1) & (lane <= 3 * h)] = 1
    axis[(lane % 3 == 2) & (lane <= 3 * w)] = 2
    return axis


def _qk_mrope_inputs(inv: Invocation, device: str,
                     gen: Any) -> Dict[str, Any]:
    import torch
    d = _dims(inv)
    dt = _dtype(inv.dtype)
    q = torch.randn(*_shape(inv, "q"), dtype=dt, device=device, generator=gen)
    k = torch.randn(*_shape(inv, "k"), dtype=dt, device=device, generator=gen)
    tokens = int(q.shape[0])
    rd = int(d["d_rope"])
    t = torch.arange(tokens, dtype=torch.int64, device=device)
    h = (t // 4)
    w = (t // 16)
    positions = torch.stack((t, h, w))
    sections, interleaved = _mrope_sections(inv)
    return {
        "q": q, "k": k,
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, rd, 10000000.0),
        "sections": sections,
        "interleaved": interleaved,
        "d_rope": rd,
    }


def _qk_mrope_reference(inputs: Dict[str, Any]) -> Any:
    """NeoX-rotate the first `d_rope` channels of q and k, each channel section
    by its own t/h/w stream, then return both writes side by side at the
    activation dtype the in-place kernels store."""
    import torch
    pos, cache = inputs["positions"], inputs["cos_sin_cache"]
    half = int(inputs["d_rope"]) // 2
    lane = _mrope_axis(inputs["sections"], bool(inputs["interleaved"])
                       ).to(pos.device)

    def gather(table):
        streams = torch.stack([table.index_select(0, pos[axis])
                               for axis in range(3)])
        return torch.where(
            lane[None, :] == 0, streams[0],
            torch.where(lane[None, :] == 1, streams[1], streams[2]))

    cos, sin = gather(cache[..., :half].float()), gather(
        cache[..., half:].float())

    def rotate(x):
        y = x.float()
        first, second = y[..., :half], y[..., half:2 * half]
        c, s = cos[:, None, :], sin[:, None, :]
        out = torch.cat((first * c - second * s,
                         second * c + first * s, y[..., 2 * half:]), -1)
        return out.to(x.dtype).float()

    q, k = rotate(inputs["q"]), rotate(inputs["k"])
    return torch.cat([q.reshape(q.shape[0], -1), k.reshape(k.shape[0], -1)],
                     dim=-1)


register_ref("fused_qk_mrope", RefSpec(_qk_mrope_inputs,
                                       _qk_mrope_reference))


def _qk_norm_partial_rope_inputs(inv: Invocation, device: str,
                                 gen: Any) -> Dict[str, Any]:
    import torch
    d = _dims(inv)
    dt = _dtype(inv.dtype)
    qkv = torch.randn(*_shape(inv, "qkv"), dtype=dt, device=device,
                      generator=gen)
    tokens = int(qkv.shape[0])
    positions = torch.arange(tokens, dtype=torch.int64,
                             device=device)
    width = int(qkv.shape[1]) // (int(d["n_heads"]) + 2 * int(d["n_kv_heads"]))
    inputs = {
        "qkv": qkv,
        "q_norm_w": torch.randn(*_shape(inv, "q_norm_w"),
                                dtype=_operand_dtype(inv, "q_norm_w"),
                                device=device, generator=gen),
        "k_norm_w": torch.randn(*_shape(inv, "k_norm_w"),
                                dtype=_operand_dtype(inv, "k_norm_w"),
                                device=device, generator=gen),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, int(d["d_rope"]), 5000000.0),
        "N": int(d["n_heads"]), "Nk": int(d["n_kv_heads"]), "D": width,
        "d_rope": int(d["d_rope"]),
        "eps": 1e-5,
    }
    # The sparse vLLM launch adds two facts the dense oracle does not read:
    # the index branch's learned weights and the slot maps that scatter into
    # freshly allocated pages. They are still part of THIS cell's input batch,
    # so draw them from the cell generator here rather than letting one
    # provider invent a second, unshared batch while it arranges its buffers.
    if "n_index_heads" in d and "kv_block_size" in d:
        blocks = -(-tokens // int(d["kv_block_size"])) + 1
        total = blocks * int(d["kv_block_size"])
        slots = torch.randperm(total, dtype=torch.int64, device=device,
                               generator=gen)
        inputs.update({
            "index_w": torch.randn(2, width, dtype=dt, device=device,
                                   generator=gen),
            "slots": slots[:tokens],
            "index_slots": slots[total - tokens:],
        })
    return inputs


def _qk_norm_partial_rope_reference(inputs: Dict[str, Any]) -> Any:
    """Gemma-normalize Q and K per head, rotate the FIRST `d_rope` channels
    NeoX-style, and leave V untouched.

    Both kernels keep the norm→RoPE chain in fp32 registers and round only the
    final store (vLLM .cu:135-181; sglang .cuh:84-115), so the oracle does not
    invent a bf16 materialization between the two span nodes.
    """
    import torch
    N, Nk, D = inputs["N"], inputs["Nk"], inputs["D"]
    row = inputs["qkv"].float()
    q = row[:, :N * D].view(-1, N, D)
    k = row[:, N * D:(N + Nk) * D].view(-1, Nk, D)

    def norm_rope(x, w):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"])
        y = y * (1.0 + w.float())
        return _rope_rotate(y, inputs["positions"], inputs["cos_sin_cache"],
                            int(inputs["d_rope"]), True).to(
                                inputs["qkv"].dtype).float()

    q, k = norm_rope(q, inputs["q_norm_w"]), norm_rope(k, inputs["k_norm_w"])
    return torch.cat([q.reshape(q.shape[0], -1),
                      k.reshape(k.shape[0], -1)], dim=-1)


register_ref("fused_qk_norm_partial_rope",
             RefSpec(_qk_norm_partial_rope_inputs,
                     _qk_norm_partial_rope_reference))


def _indexer_q_rope_quant_inputs(inv: Invocation, device: str,
                                 gen: Any) -> Dict[str, Any]:
    """The indexer's own operands: raw q heads, positive head weights, and the
    GPT-J tail-rotation table. Positive weights let an FP8 provider recover the
    per-head scale the launch folds into `weights_out`."""
    import torch
    d = _dims(inv)
    dt = _dtype(inv.dtype)
    q = torch.randn(*_shape(inv, "q"), dtype=dt, device=device, generator=gen)
    tokens = int(q.shape[0])
    rd = int(d["d_rope"])
    positions = torch.arange(tokens, dtype=torch.int64,
                             device=device)
    return {
        "q": q,
        "weights": 0.5 + torch.rand(*_shape(inv, "weights"), dtype=dt,
                                    device=device, generator=gen),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, rd, 10000.0),
        "d_rope": rd,
        "output": inv.dtype_for("output"),
    }


def _hadamard_matrix(n: int, device: Any):
    """The Sylvester H_n the fast transform computes, for an oracle small
    enough to materialize it (the benchmarked width is 128)."""
    import torch
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), dim=1),
                       torch.cat((h, -h), dim=1)), dim=0)
    return h


def _indexer_q_rope_quant_reference(inputs: Dict[str, Any]) -> Any:
    """DeepSeek-V4's reference boundary: tail RoPE, a 128-point Hadamard,
    quantisation, and the rescaled weights beside q so one comparison covers
    both writes.

    sglang's default FP8 launch is the exact grid here: it computes in fp32,
    uses the continuous scale `max(1e-4, amax) / 448`, and folds that scale
    into `weights_out` (main_norm_rope.cuh:662-712). vLLM omits H and rounds
    its FP8 scale to a power of two; its provider carries the separate
    no-H oracle below and canonicalizes into this Hadamard basis. MXFP4 keeps
    the per-32 UE8M0 grid.
    """
    import torch
    from .quant import FORMATS, quantise_groups

    q = inputs["q"].float()
    rd = int(inputs["d_rope"])
    nope, rot = q[..., :-rd], q[..., -rd:]
    rot = _rope_rotate(rot, inputs["positions"], inputs["cos_sin_cache"], rd,
                       is_neox=False)
    rotated = torch.cat([nope, rot], dim=-1)
    rotated = torch.matmul(
        rotated, _hadamard_matrix(rotated.shape[-1], rotated.device)
    ) * rotated.shape[-1] ** -0.5
    fixed = 1.0 / torch.sqrt(
        torch.tensor(q.shape[-1] * q.shape[1], dtype=torch.float32))
    weights = inputs["weights"].float() * fixed

    if inputs["output"] == "fp4_e2m1":
        values, scale = quantise_groups(
            rotated, FORMATS["mxfp4"], 32, "pow2", inputs["q"].dtype,
            eps=6.0 * 2.0 ** -126)
        q_out = values.float() * scale.unsqueeze(-1)
    elif inputs["output"] == "fp8_e4m3":
        fmt = FORMATS["fp8_e4m3"]
        values, scale = quantise_groups(
            rotated, fmt, q.shape[-1], "exact", inputs["q"].dtype, eps=1e-4)
        q_out = values.float() * scale.unsqueeze(-1)
        weights = weights * scale.reshape(q.shape[:-1])
    else:
        raise ValueError(
            f"indexer q quantisation has no grid for {inputs['output']!r}")
    return torch.cat([q_out.reshape(q.shape[0], -1), weights], dim=-1)


register_ref("fused_indexer_q_rope_quant",
             RefSpec(_indexer_q_rope_quant_inputs,
                     _indexer_q_rope_quant_reference))


def _indexer_q_rope_quant_vllm_reference(inputs: Dict[str, Any]) -> Any:
    """vLLM's audited V4 deviation, in the shared Hadamard basis used below.

    The launch rounds the ROTATED values through bf16, omits H, rounds the
    FP8 scale to a power of two, and folds that scale into weights
    (fused_indexer_q.py:153-205; MXFP4 uses per-32 UE8M0 scales at
    :392-441). The provider applies H only when canonicalizing, so this
    oracle models the launch as it runs rather than pretending it is the
    reference implementation.
    """
    import dataclasses
    import torch
    from .quant import FORMATS, quantise_groups

    q = inputs["q"].float()
    rd = int(inputs["d_rope"])
    nope, rot = q[..., :-rd], q[..., -rd:]
    rot = _rope_rotate(rot, inputs["positions"], inputs["cos_sin_cache"], rd,
                       is_neox=False).to(torch.bfloat16).float()
    rotated = torch.cat([nope, rot], dim=-1)
    fixed = 1.0 / torch.sqrt(
        torch.tensor(q.shape[-1] * q.shape[1], dtype=torch.float32))
    weights = inputs["weights"].float() * fixed

    if inputs["output"] == "fp4_e2m1":
        values, scale = quantise_groups(
            rotated, FORMATS["mxfp4"], 32, "pow2", inputs["q"].dtype,
            eps=6.0 * 2.0 ** -126)
        q_out = values.float() * scale.unsqueeze(-1)
    elif inputs["output"] == "fp8_e4m3":
        fmt = dataclasses.replace(FORMATS["fp8_e4m3"], pow2_scale=True)
        values, scale = quantise_groups(
            rotated, fmt, q.shape[-1], "exact", inputs["q"].dtype, eps=1e-4)
        q_out = values.float() * scale.unsqueeze(-1)
        weights = weights * scale.reshape(q.shape[:-1])
    else:
        raise ValueError(
            f"indexer q quantisation has no grid for {inputs['output']!r}")
    return torch.cat([q_out.reshape(q.shape[0], -1), weights], dim=-1)


def _indexer_q_rope_quant_split_reference(inputs: Dict[str, Any]) -> Any:
    """sglang's V4.1 split boundary (`index_q_rope_pack_weights`,
    fp4_indexer_rope.cuh:89-144): the same GPT-J tail rotation and bf16
    round-trip on the rotated lanes, then TWO quantisation passes, not one.

    The kernel fake-quants each 32-element block first (floor `6 * 2**-126`,
    `fp4::block_scale`), then packs the fake-quant result with its own amax
    floor — `index_pack_exponent` is `ceil_ue8m0(max(amax/6, 1e-4))`, the
    floor applied past the divide by 6 (`eps=6e-4` on this side of it). The
    wrapper's docstring warns collapsing the two stages is not equivalent
    (fp4_indexer_rope.py:80-83): a block whose fake amax sits below the
    packer's floor is rescaled — and here zeroed — where one pass keeps it.
    The head weights take a bf16 round-trip after the fp32 multiply
    (`weights_out = float(bf16(w * weight_scale))`, .cuh:101-103), which the
    one-stage oracle also skips.

    vLLM's one-stage launch serves this signature too. Its record dequantises
    to the same values except at the packer's floor: the e2m1 grid {2^k,
    1.5·2^k} is closed under doubling up to its peak, so the packer's second
    pass is lossless on the fake-quant grid, and its fp32 weight write sits
    well inside the activation band of the bf16-rounded oracle.
    """
    import dataclasses
    import torch
    from .quant import FORMATS, quantise_groups

    q = inputs["q"].float()
    rd = int(inputs["d_rope"])
    nope, rot = q[..., :-rd], q[..., -rd:]
    rot = _rope_rotate(rot, inputs["positions"], inputs["cos_sin_cache"], rd,
                       is_neox=False).to(torch.bfloat16).float()
    rotated = torch.cat([nope, rot], dim=-1)
    fixed = 1.0 / torch.sqrt(
        torch.tensor(q.shape[-1] * q.shape[1], dtype=torch.float32))
    weights = inputs["weights"].float() * fixed

    if inputs["output"] == "fp4_e2m1":
        fake, _scale = quantise_groups(
            rotated, FORMATS["mxfp4"], 32, "pow2", inputs["q"].dtype,
            eps=6.0 * 2.0 ** -126)
        fake = fake.float() * _scale.unsqueeze(-1)
        values, scale = quantise_groups(
            fake, FORMATS["mxfp4"], 32, "pow2", inputs["q"].dtype,
            eps=6.0e-4)
        q_out = values.float() * scale.unsqueeze(-1)
        weights = weights.to(torch.bfloat16).float()
    elif inputs["output"] == "fp8_e4m3":
        fmt = dataclasses.replace(FORMATS["fp8_e4m3"], pow2_scale=True)
        values, scale = quantise_groups(
            rotated, fmt, q.shape[-1], "exact", inputs["q"].dtype, eps=1e-4)
        q_out = values.float() * scale.unsqueeze(-1)
        weights = weights * scale.reshape(q.shape[:-1])
    else:
        raise ValueError(
            f"indexer q quantisation has no grid for {inputs['output']!r}")
    return torch.cat([q_out.reshape(q.shape[0], -1), weights], dim=-1)


register_ref("fused_indexer_q_rope_quant_split",
             RefSpec(_indexer_q_rope_quant_inputs,
                     _indexer_q_rope_quant_split_reference))


def _rope_2d_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """x [tokens, H, D] and the per-token cos/sin rows [tokens, D/2] a vision
    tower builds from each patch's (h, w): half the frequencies read the row,
    half the column (vLLM qwen3_vl.py `rot_pos_emb`), laid out on the
    smallest square grid that holds the tokens."""
    import math
    import torch
    x = torch.randn(*_shape(inv, "x"), dtype=_dtype(inv.dtype), device=device, generator=gen)
    tok, D = int(x.shape[0]), int(x.shape[-1])
    side = max(1, math.isqrt(tok - 1) + 1) if tok > 1 else 1
    t = torch.arange(tok, device=device)
    quarter = D // 4
    inv_freq = 1.0 / (10000.0 ** (torch.arange(
        quarter, dtype=torch.float32, device=device) / max(quarter, 1)))
    freqs = torch.cat([torch.outer((t // side).float(), inv_freq),
                       torch.outer((t % side).float(), inv_freq)], dim=-1)
    return {"x": x, "cos": freqs.cos(), "sin": freqs.sin()}


def _rope_2d_rotate(x, cos, sin):
    """NeoX rotation of x [tokens, H, D] by cos/sin [tokens, D/2] —
    `ApplyRotaryEmb.forward_static` with `is_neox_style=True`."""
    import torch
    cos, sin = cos[:, None, :], sin[:, None, :]
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def _rope_2d_reference(inputs: Dict[str, Any]) -> Any:
    return _rope_2d_rotate(inputs["x"].float(), inputs["cos"], inputs["sin"])


def _xdrope_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """x [tokens, n_heads, d_rope] and the per-token cos/sin rows
    [tokens, d_rope/2] vLLM's `XDRotaryEmbedding` assembles: each
    `xdrope_section` channel section reads its own position stream (p, w, h,
    t). The rotation is `rope_2d`'s — one NeoX rotation over the whole head
    by per-token rows — so what is its own is where the frequencies come
    from, and four streams over equal sections is the layout the kernel is
    handed."""
    import torch
    x = torch.randn(*_shape(inv, "x"), dtype=_dtype(inv.dtype), device=device, generator=gen)
    tok, d_rope = int(x.shape[0]), int(x.shape[-1])
    half = d_rope // 2
    per = max(1, half // 4)
    t = torch.arange(tok, device=device)
    # Four streams, as HunYuan-VL's text path has them: the position itself
    # and three grid coordinates.
    streams = [t, t % 16, (t // 16) % 16, t // 256]
    cols = []
    for i in range(4):
        n = per if i < 3 else half - 3 * per
        if n <= 0:
            continue
        freq = 1.0 / (10000.0 ** (torch.arange(
            n, dtype=torch.float32, device=device) / max(n, 1)))
        cols.append(torch.outer(streams[i].float(), freq))
    freqs = torch.cat(cols, dim=-1)
    return {"x": x, "cos": freqs.cos(), "sin": freqs.sin()}


register_ref("xdrope", RefSpec(_xdrope_inputs, _rope_2d_reference))


register_ref("rope_2d", RefSpec(_rope_2d_inputs, _rope_2d_reference))


# ---------------------------------------------------------------------------
# fused_qk_norm_rope: per-head RMSNorm (separate Q/K gains) then RoPE, over one
# packed qkv [tokens, (N+2·Nk)·D] (V untouched). Reference reproduces norm+rope
# with a neutral (no-YARN) rope so it matches the kernel called with factor=1.
# ---------------------------------------------------------------------------

def _qk_norm_rope_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    d = _dims(inv)
    dt = _dtype(inv.dtype)
    qkv = torch.randn(*_shape(inv, "qkv"), dtype=dt, device=device, generator=gen)
    N = int(d["n_heads"]); Nk = int(d.get("n_kv_heads", N))
    D = int(d["d_rope"])
    tok = int(qkv.shape[0])
    positions = torch.arange(tok, device=device)
    return {
        "qkv": qkv,
        "q_norm_w": torch.randn(*_shape(inv, "q_norm_w"),
                                dtype=_operand_dtype(inv, "q_norm_w"),
                                device=device, generator=gen),
        "k_norm_w": torch.randn(*_shape(inv, "k_norm_w"),
                                dtype=_operand_dtype(inv, "k_norm_w"),
                                device=device, generator=gen),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, D, 10000.0),
        "N": N, "Nk": Nk, "D": D, "rotary_dim": D, "is_neox": True,
        "base": 10000.0, "eps": 1e-6,
    }


def _qk_norm_rope_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    N, Nk, D = inputs["N"], inputs["Nk"], inputs["D"]
    qkv = inputs["qkv"].float()
    q = qkv[:, : N * D].view(-1, N, D)
    k = qkv[:, N * D: (N + Nk) * D].view(-1, Nk, D)
    v = qkv[:, (N + Nk) * D:]                             # passthrough
    qw, kw = inputs["q_norm_w"].float(), inputs["k_norm_w"].float()

    def _rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"]) * w

    q = _rope_rotate(_rms(q, qw), inputs["positions"], inputs["cos_sin_cache"],
                     inputs["rotary_dim"], inputs["is_neox"])
    k = _rope_rotate(_rms(k, kw), inputs["positions"], inputs["cos_sin_cache"],
                     inputs["rotary_dim"], inputs["is_neox"])
    return torch.cat([q.reshape(q.shape[0], N * D), k.reshape(k.shape[0], Nk * D), v], dim=1)


register_ref("fused_qk_norm_rope",
             RefSpec(_qk_norm_rope_inputs, _qk_norm_rope_reference))


def _qk_gate_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The gated-attention prologue's operands: per head [q|gate] packed, k
    beside it, a gain each, the positions and the rotation table.

    The gains are the norm's EFFECTIVE gain `g` (the spec's `rmsnorm` and
    its `rmsnorm_unscaled · w` writing multiply by it). Both frameworks'
    kernels take the Gemma checkpoint form `g - 1` and add the one back
    (vLLM `norm_beta=1.0`, sglang `WEIGHT_SHIFT`). The gains are drawn on
    [0.5, 2), as trained gains sit near one, which also makes `g - 1` exact
    in the gain's own bf16 / fp16: the providers hand the kernels the
    checkpoint form without a rounding the oracle does not have."""
    import torch
    d = _dims(inv)
    dt = _dtype(inv.dtype)
    q_gate = torch.randn(*_shape(inv, "q_gate"), dtype=dt, device=device, generator=gen)
    k = torch.randn(*_shape(inv, "k"), dtype=dt, device=device, generator=gen)
    N = int(d["n_heads"]); Nk = int(d.get("n_kv_heads", N))
    D = int(d["d_head"]); rd = int(d["d_rope"])
    tok = int(q_gate.shape[0])
    positions = torch.arange(tok, device=device)

    def gain(name):
        g = 1.0 + 0.2 * torch.randn(*_shape(inv, name), device=device,
                                    generator=gen)
        return g.clamp(0.5, 1.99).to(_operand_dtype(inv, name))

    return {
        "q_gate": q_gate, "k": k,
        "q_norm_w": gain("q_norm_w"), "k_norm_w": gain("k_norm_w"),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, rd, 10000.0),
        "N": N, "Nk": Nk, "D": D, "d_rope": rd, "eps": 1e-6,
    }


def _qk_gate_reference(inputs: Dict[str, Any]) -> Any:
    """Split [q|gate] per head, RMSNorm q and k over the whole head with
    their own gains, rotate the first `d_rope` channels NeoX-style (the
    plan's rope slice; the rest pass through), and carry the gate through —
    the three tensors the kernel returns, concatenated so one array
    compares."""
    import torch
    N, Nk, D = inputs["N"], inputs["Nk"], inputs["D"]
    rd = int(inputs["d_rope"])
    qg = inputs["q_gate"].float().view(-1, N, 2, D)
    q, gate = qg[:, :, 0, :], qg[:, :, 1, :]
    k = inputs["k"].float().view(-1, Nk, D)
    eps = inputs["eps"]

    def _rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    pos, cache = inputs["positions"], inputs["cos_sin_cache"]
    q = _rope_rotate(_rms(q, inputs["q_norm_w"].float()), pos, cache, rd, True)
    k = _rope_rotate(_rms(k, inputs["k_norm_w"].float()), pos, cache, rd, True)
    rows = q.shape[0]
    return torch.cat([q.reshape(rows, N * D), k.reshape(rows, Nk * D),
                      gate.reshape(rows, N * D)], dim=1)


register_ref("fused_qk_rmsnorm_rope_gate",
             RefSpec(_qk_gate_inputs, _qk_gate_reference))


# ---------------------------------------------------------------------------
# moe_gate: aux-loss-free grouped top-k router. Kernels emit unsorted
# top-k ids, so the oracle scatters weights onto a dense [tokens, E]
# map — order does not matter. Group structure is a kernel parameter,
# not an identity axis: missing means decline, not a guessed 8/4.
# ---------------------------------------------------------------------------

def _moe_gate_param(inv: Invocation, name: str) -> int:
    """A grouping integer the kernel needs to launch.

    `kernel_params` is the plan's copy of the spec dim; `dims` is what
    smoke / a hand-built invocation binds. Either may carry it; neither
    may invent it.
    """
    raw = inv.kernel_params.get(name)
    if raw is None:
        d = _dims(inv)
        # `in` first: a recording dims mapping logs only the reads that land.
        raw = d[name] if name in d else None  # noqa: SIM401
    if raw is None:
        raise ValueError(
            f"moe_gate requires {name} to launch (no default)")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"moe_gate {name}={value} must be positive")
    return value


def _moe_gate_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The grouped router's inputs: the logits at the dtype the CELL states
    (`_operand_dtype`), the selection bias fp32.

    The logits follow the cell because serving hands this kernel whatever
    its router GEMM wrote. The walk states the two halves of a mixed gate
    apart: a gate stored fp32 (MiMo-V2.x) makes the gate GEMM COMPUTE fp32,
    while the logits it hands the gate keep the edge's declared width — bf16
    wherever the model serves bf16, as a gate stored bf16 (the DeepSeek-V3
    family) also writes. Neither framework's grouped gate restricts the
    dtype (vLLM `fused_grouped_topk`, sglang `moe_fused_gate`), so the cell
    is the width to build: fp32 for all of them measured a kernel at twice
    the logits bytes the step moves.

    The bias stays fp32, for the reason `_moe_topk_inputs` gives: it is the
    kernel's parameter type rather than a guess at how it is stored, and at
    `n_experts` elements it is not what any router row is bandwidth-bound
    on. Its slot here is `metadata` (`op_signatures_catalog`), which an
    unquantised gate's cell does not carry at all."""
    import torch
    gating = torch.randn(*_shape(inv, "gating"),
                         dtype=_operand_dtype(inv, "gating"), device=device,
                         generator=gen)
    E = int(gating.shape[-1])
    n_group = _moe_gate_param(inv, "n_expert_groups")
    if E % n_group != 0:
        raise ValueError(
            f"moe_gate: n_experts={E} is not divisible by "
            f"n_expert_groups={n_group}")
    topk_group = _moe_gate_param(inv, "n_groups_active")
    if topk_group > n_group:
        raise ValueError(
            f"moe_gate: n_groups_active={topk_group} > "
            f"n_expert_groups={n_group}")
    topk = _moe_gate_param(inv, "n_experts_active")
    if topk > E:
        raise ValueError(
            f"moe_gate: n_experts_active={topk} > n_experts={E}")
    return {
        "gating": gating,
        "bias": torch.randn(*_shape(inv, "bias"), dtype=torch.float32, device=device, generator=gen),
        "num_expert_group": n_group,
        "topk_group": topk_group,
        "topk": topk,
    }


def _moe_gate_dense(weights, ids, n_experts):
    """Scatter routing weights onto [tokens, E] so unsorted top-k aligns."""
    weights, ids = weights.float(), ids.long()
    dense = weights.new_zeros(*weights.shape[:-1], n_experts)
    return dense.scatter(-1, ids, weights)


def _moe_gate_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    scores = inputs["gating"].float().sigmoid()
    if scores.dim() > 2:
        scores = scores.reshape(-1, scores.shape[-1])
    n_tok, E = scores.shape
    n_group = int(inputs["num_expert_group"])
    topk_group = int(inputs["topk_group"])
    topk = int(inputs["topk"])
    sfc = scores + inputs["bias"].float().reshape(1, -1)
    group_scores = sfc.view(n_tok, n_group, -1).topk(2, dim=-1).values.sum(-1)
    gidx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=True).indices
    gmask = torch.zeros_like(group_scores).scatter_(1, gidx, 1)
    smask = (gmask.unsqueeze(-1)
             .expand(n_tok, n_group, E // n_group)
             .reshape(n_tok, -1))
    masked = sfc.masked_fill(~smask.bool(), float("-inf"))
    ids = torch.topk(masked, k=topk, dim=-1, sorted=True).indices
    weights = scores.gather(1, ids)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return _moe_gate_dense(weights, ids, E)


register_ref("moe_gate", RefSpec(_moe_gate_inputs, _moe_gate_reference))


# ---------------------------------------------------------------------------
# Ungrouped top-k routers (`moe_topk_*`). The logits are random normal; the
# selection bias, where the span has one, small enough not to swamp them (a
# checkpoint's correction bias steers, it does not dictate). Each builder
# states two values the span's naive pattern leaves to the model, because
# neither moves a byte: whether the weights are renormalised (the plan's
# optional `normalize_sum_one`) and the routed scale (its `scale_const`,
# measured at 1.0 — an in-register multiply). The renormalisation is set to
# what the corpus models of that span run. Providers read both from the
# inputs, so kernel and oracle compute one function. Output: the weights
# scattered onto `[tokens, n_experts]` (`_moe_gate_dense`), so kernels that
# emit unsorted ids compare.
# ---------------------------------------------------------------------------

def _topk_int(inv: Invocation) -> int:
    k = int(_dims(inv)["n_experts_active"])
    if k <= 0:
        raise ValueError(f"{inv.signature}: n_experts_active={k}")
    return k


def _moe_topk_inputs(inv: Invocation, device: str, gen: Any, *,
                     biased: bool, renormalize: bool) -> Dict[str, Any]:
    """The logits at the width the CELL states (`_operand_dtype`), as every
    router oracle builds them.

    The cell is now able to state it. Where the top-k kernel a framework
    dispatches to requires wider logits than the checkpoint implies, its
    binding declares that requirement on the `router` role
    (`FrameworkBinding.edge_dtypes`) and the walk puts it on the edge, so
    this slot already says fp32 for the spans whose kernel needs fp32 —
    vLLM's DeepSeek-V4 router among them. This builder
    used to take a `logits_dtype` override so the sqrt-softplus spans could
    force fp32 against their own cell; that override is gone, because a
    width the oracle picks and the key does not know is a measurement filed
    under the wrong name.
    """
    import torch
    out = {
        "gating": torch.randn(*_shape(inv, "gating"),
                              dtype=_operand_dtype(inv, "gating"),
                              device=device, generator=gen),
        "topk": _topk_int(inv),
        "renormalize": renormalize,
        "routed_scaling_factor": 1.0,
    }
    if biased:
        # fp32 is the KERNEL's parameter type here, not a guess at how the
        # bias is stored: vLLM's `dsv4_topk` takes only fp32
        # (`can_use_dsv4_topk`), and a cell whose weight slot is absent would
        # otherwise hand it the activation's bf16. The selection bias is a
        # vector of `n_experts` — its bytes are not what any router row is
        # bandwidth-bound on, so `_operand_dtype` would trade a measurable
        # launch for an unmeasurable byte.
        out["bias"] = 0.1 * torch.randn(*_shape(inv, "bias"), dtype=torch.float32,
                                        device=device, generator=gen)
    return out


def _route_dense(score, select, topk: int, renormalize: bool, scale: float):
    """Top-k of `select`, weights gathered from `score` — the plans' `topk`,
    `gather`, optional `normalize_sum_one` and `scale_const` nodes."""
    import torch
    ids = torch.topk(select, k=topk, dim=-1, sorted=True).indices
    return _weights_dense(score, ids, renormalize, scale)


def _weights_dense(score, ids, renormalize: bool, scale: float):
    w = score.gather(-1, ids)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return _moe_gate_dense(w * scale, ids, score.shape[-1])


def _softmax_scores(g):
    import torch
    return torch.softmax(g.float(), dim=-1)


def _sigmoid_scores(g):
    import torch
    return torch.sigmoid(g.float())


def _sqrtsoftplus_scores(g):
    import torch
    import torch.nn.functional as F
    return torch.sqrt(F.softplus(g.float()))


def moe_topk_router(score_fn):
    """The oracle of one ungrouped router span: its scoring function, the
    selection bias where the inputs carry one, top-k, gather, the stated
    renormalisation and scale. Also what the torch floors run."""
    def reference(inputs: Dict[str, Any]) -> Any:
        score = score_fn(inputs["gating"])
        select = score + inputs["bias"].float() if "bias" in inputs else score
        return _route_dense(score, select, int(inputs["topk"]),
                            bool(inputs["renormalize"]),
                            float(inputs["routed_scaling_factor"]))
    return reference


def _moe_topk_softmax_inputs(inv: Invocation, device: str, gen: Any):
    # Renormalised: the Qwen-MoE routers (`norm_topk_prob`), every corpus
    # model of the span but DeepSeek-V2-Lite.
    return _moe_topk_inputs(inv, device, gen, biased=False, renormalize=True)


def _moe_topk_sigmoid_inputs(inv: Invocation, device: str, gen: Any):
    # The plan always renormalises (its `normalize_sum_one` is required).
    return _moe_topk_inputs(inv, device, gen, biased=True, renormalize=True)


def _moe_topk_softmax_bias_inputs(inv: Invocation, device: str, gen: Any):
    # Not renormalised: LongCat-Flash, the span's one corpus model.
    return _moe_topk_inputs(inv, device, gen, biased=True, renormalize=False)


def _moe_topk_sqrtsoftplus_inputs(inv: Invocation, device: str, gen: Any):
    """The logits at the width the cell states — which for this span is the
    fp32 its kernel requires, because the walk now holds that fact.

    DeepSeek-V4 ships `ffn.gate.weight` BF16 and both frameworks load it
    BF16, so nothing about the CHECKPOINT makes these logits fp32. The
    framework does: vLLM's router GEMM is built
    `GateLinear(out_dtype=torch.float32)` and the kernel it then dispatches
    exists only at that width — `can_use_dsv4_topk` requires
    `gating_output.dtype == torch.float32`, and `fused_topk_bias` falls
    through to another kernel when it fails. That is declared on the
    `router` role in each framework's binding (`edge_dtypes`), the walk puts
    it on the logits edge, and this builder reads it like any other operand.

    So this span is no longer the one place a router oracle disagrees with
    its cell: the cell says fp32, the walk bills fp32, and the measurement
    is filed under fp32. Building the cell's width is also what keeps the
    RIGHT kernel measured — at bf16 the provider would time the fallback
    under a key promising the dsv4 one."""
    return _moe_topk_inputs(inv, device, gen, biased=True, renormalize=True)


register_ref("moe_topk_softmax",
             RefSpec(_moe_topk_softmax_inputs, moe_topk_router(_softmax_scores)))
register_ref("moe_topk_sigmoid",
             RefSpec(_moe_topk_sigmoid_inputs, moe_topk_router(_sigmoid_scores)))
register_ref("moe_topk_softmax_bias",
             RefSpec(_moe_topk_softmax_bias_inputs,
                     moe_topk_router(_softmax_scores)))
register_ref("moe_topk_sqrtsoftplus",
             RefSpec(_moe_topk_sqrtsoftplus_inputs,
                     moe_topk_router(_sqrtsoftplus_scores)))


def _moe_mega_gate_inputs(inv: Invocation, device: str,
                          gen: Any) -> Dict[str, Any]:
    """Hidden row, BF16 router weight and F32 correction bias. Top-k is the
    cell's `n_experts_active`; renormalisation and a unit routed scale are
    the router oracles' convention (`_moe_topk_inputs`) — the scale is one
    multiply on `[tokens, top-k]` and moves no byte."""
    import torch
    dt = _dtype(inv.dtype)
    tokens, width = _shape(inv, "x")
    return {
        "x": torch.randn(tokens, width, dtype=dt, device=device,
                         generator=gen),
        "weight": torch.randn(*_shape(inv, "weight"), dtype=torch.bfloat16,
                              device=device, generator=gen),
        "bias": 0.1 * torch.randn(*_shape(inv, "bias"),
                                  dtype=torch.float32, device=device,
                                  generator=gen),
        "topk": _topk_int(inv),
        "renormalize": True,
        "routed_scaling_factor": 1.0,
    }


def _moe_mega_gate_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's chain at a unit gate temperature (the launch takes no
    temperature operand): the router GEMM, sqrt(softplus), top-k of the
    biased scores, weights gathered from the unbiased ones."""
    import torch
    gating = inputs["x"].float() @ inputs["weight"].float().t()
    score = torch.sqrt(torch.nn.functional.softplus(gating))
    return _route_dense(score, score + inputs["bias"].float(),
                        int(inputs["topk"]),
                        bool(inputs["renormalize"]),
                        float(inputs["routed_scaling_factor"]))


register_ref("moe_mega_gate_sqrtsoftplus",
             RefSpec(_moe_mega_gate_inputs, _moe_mega_gate_reference))


def _moe_topk_hash_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """DeepSeek-V4 hash routing: the logits at the cell's width (as above),
    token ids, and a `tid2eid` table of k distinct experts per token id (a
    random start and k consecutive experts, mod E — distinct whenever
    k <= E).

    The table is built int32, which is what both frameworks allocate and
    their kernels verify, while the checkpoint ships it I64. That half is
    STILL hard-coded here: the loaded width of a RESIDENT tensor is a
    different object from the width a role's output is written at — it
    belongs in `Representation.intent`, which the walk fixes before a
    binding is in hand — so the walk continues to bill this table 8 bytes
    an element and the row stays declined for want of an I64 corpus token
    (`RowIdentity.leaf_dtype_env`, `docs/inference_cost_model.md` "Stored
    width vs loaded width")."""
    import torch
    out = _moe_topk_inputs(inv, device, gen, biased=False, renormalize=True)
    n_vocab, k = _shape(inv, "table")
    E = int(out["gating"].shape[-1])
    start = torch.randint(0, E, (n_vocab, 1), device=device, generator=gen)
    out["table"] = ((start + torch.arange(k, device=device)) % E).to(torch.int32)
    tokens = int(out["gating"].shape[0])
    out["input_ids"] = torch.randint(0, n_vocab, (tokens,), device=device,
                                     generator=gen)
    return out


def moe_topk_hash_router(inputs: Dict[str, Any]) -> Any:
    """Experts from the table at each token's id, weights from the
    sqrt-softplus scores there."""
    score = _sqrtsoftplus_scores(inputs["gating"])
    ids = inputs["table"].long()[inputs["input_ids"].long()]
    return _weights_dense(score, ids, bool(inputs["renormalize"]),
                          float(inputs["routed_scaling_factor"]))


register_ref("moe_topk_hash_sqrtsoftplus",
             RefSpec(_moe_topk_hash_inputs, moe_topk_hash_router))


# ---------------------------------------------------------------------------
# indexer_logits (DSA lightning indexer): logits[t,j] = Σ_h w[t,h]·ReLU(q[t,h]·k[j]).
# q[B·s_query,Hix,Dix], each request's MQA indexer-K k[B,Lq,Dix], gate
# weights[B·s_query,Hix]; a query scores only its own request's keys.
# Reference = the dense fp32 form the torch provider computes; the DeepGEMM
# fp8 kernel (paged) is compared against it within the fp8 tolerance.
# ---------------------------------------------------------------------------

def _indexer_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    store_dt = torch.bfloat16 if dt.is_floating_point and dt.itemsize < 2 else dt
    return {
        "q": torch.randn(*_shape(inv, "q"), dtype=store_dt, device=device, generator=gen),
        "k": torch.randn(*_shape(inv, "k"), dtype=_operand_dtype(inv, "k"),
                         device=device, generator=gen),
        "weights": torch.rand(*_shape(inv, "weights"),
                              dtype=torch.float32, device=device, generator=gen),
    }


def per_request_mqa_logits(q: Any, k: Any, w: Any) -> Any:
    """`logits[b·s+t, j] = Σ_h w[b·s+t,h]·ReLU(q[b·s+t,h]·k[b,j])` (the
    registered op's formula) at the operands' dtype, the scores widened to
    fp32 before the gate sum: `[B·s_query, Lq]`, each query against its own
    request's keys."""
    import torch
    batch = k.shape[0]
    qb = q.view(batch, -1, *q.shape[1:])                 # [B,s,Hix,Dix]
    scores = torch.einsum("bthd,bjd->bthj", qb, k)       # [B,s,Hix,Lq]
    logits = torch.einsum("bthj,bth->btj", scores.float().relu(),
                          w.view(batch, -1, w.shape[-1]))
    return logits.flatten(0, 1)                          # [B·s,Lq]


def _indexer_reference(inputs: Dict[str, Any]) -> Any:
    return per_request_mqa_logits(inputs["q"].float(), inputs["k"].float(),
                                  inputs["weights"])


register_ref("indexer_logits", RefSpec(_indexer_inputs, _indexer_reference,
                                       materializes_scores=True))


# indexer_block_score (MiniMax-M3 MSA): scores[b,h,s,j] = max over block j of
# q[b,s,h]·k[b,t], blocks of s_sparse_block keys (the last one short), CAUSAL
# lower-right like every implementation of the span — HF masks future keys
# before the amax (modeling_minimax_m3_vl.py `token_future`), vLLM's
# `_index_block_score_kernel` masks `off_q >= pos`, sglang's kernels the same.
# Query s sees keys 0 .. s_kv − s_query + s (a one-token decode sees the whole
# cache); a block the query cannot see floors to a finite sentinel, the value
# an all-`-inf` amax would take, so the comparison stays finite.

#: The floor a fully masked block's score takes — finite, so `-inf − -inf`
#: never reaches the cross-check. vLLM's prefill kernel leaves those blocks
#: unwritten; its provider canonicalizes them to the same value.
MASKED_BLOCK_SCORE = -1e30


def causal_block_max(scores: Any, block: int) -> Any:
    """Block max of a `[.., s_query, s_kv]` score matrix, causal lower-right."""
    import torch
    s_query, s_kv = scores.shape[-2], scores.shape[-1]
    future = ~visible_keys(s_query, s_kv, scores.device, causal=True)
    scores = scores.masked_fill(future, float("-inf"))
    pad = -s_kv % block
    scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    return scores.unflatten(-1, (-1, block)).amax(-1).clamp_min(
        MASKED_BLOCK_SCORE)


def _block_score_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    store_dt = torch.bfloat16 if dt.is_floating_point and dt.itemsize < 2 else dt
    return {
        "q": torch.randn(*_shape(inv, "q"), dtype=store_dt, device=device, generator=gen),
        "k": torch.randn(*_shape(inv, "k"), dtype=_operand_dtype(inv, "k"),
                         device=device, generator=gen),
        "block": int(_dims(inv)["s_sparse_block"]),
    }


def _block_score_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    scores = torch.einsum("bshd,btd->bhst", inputs["q"].float(),
                          inputs["k"].float())
    return causal_block_max(scores, int(inputs["block"]))


register_ref("indexer_block_score",
             RefSpec(_block_score_inputs, _block_score_reference))


def _embedding_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    ids_shape = _shape(inv, "ids")
    n_vocab = int(_dims(inv)["n_vocab"])
    return {
        "ids": torch.randint(0, n_vocab, ids_shape, device=device, generator=gen),
        "w": torch.randn(*_shape(inv, "w"), dtype=_operand_dtype(inv, "w"),
                         device=device, generator=gen),
    }


def _embedding_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.nn.functional.embedding(inputs["ids"], inputs["w"].float())


register_ref("embedding", RefSpec(_embedding_inputs, _embedding_reference))


def _per_head_gemm_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=_operand_dtype(inv, "x"),
                         device=device, generator=gen),
        "w": torch.randn(*_shape(inv, "w"), dtype=_operand_dtype(inv, "w"),
                         device=device, generator=gen),
    }


def _per_head_gemm_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.einsum("bhsk,hnk->bhsn", inputs["x"].float(),
                        inputs["w"].float())


register_ref("per_head_gemm",
             RefSpec(_per_head_gemm_inputs, _per_head_gemm_reference))


def _layernorm_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen),
        "w": torch.randn(*_shape(inv, "w"), dtype=_operand_dtype(inv, "w"),
                         device=device, generator=gen),
        "b": torch.randn(*_shape(inv, "b"), dtype=_operand_dtype(inv, "b"),
                         device=device, generator=gen),
        "eps": 1e-5,
    }


def _layernorm_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.nn.functional.layer_norm(
        inputs["x"].float(), (inputs["x"].shape[-1],),
        inputs["w"].float(), inputs["b"].float(), inputs["eps"])


register_ref("layernorm", RefSpec(_layernorm_inputs, _layernorm_reference))


def _dsa_topk_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "logits": torch.randn(*_shape(inv, "logits"), dtype=dt, device=device, generator=gen),
        "k": int(_dims(inv)["n_topk_keys"]),
    }


def _dsa_topk_reference(inputs: Dict[str, Any]) -> Any:
    return inputs["logits"].float().topk(int(inputs["k"]), dim=-1).indices


register_ref("dsa_topk_mask", RefSpec(_dsa_topk_inputs, _dsa_topk_reference))


def _softmax_act_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {"x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen)}


def _softmax_act_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.nn.functional.softmax(inputs["x"].float(), dim=-1)


register_ref("softmax_act",
             RefSpec(_softmax_act_inputs, _softmax_act_reference))


def _topk_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen),
        "k": int(_dims(inv)["n_experts_active"]),
    }


def _topk_reference(inputs: Dict[str, Any]) -> Any:
    return inputs["x"].float().topk(int(inputs["k"]), dim=-1).indices


register_ref("topk", RefSpec(_topk_inputs, _topk_reference))


def _gather_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    x_shape = _shape(inv, "x")
    idx_shape = _shape(inv, "idx")
    n_experts = int(x_shape[-1])
    return {
        "x": torch.randn(*x_shape, dtype=dt, device=device, generator=gen),
        "idx": torch.randint(0, n_experts, idx_shape, device=device, generator=gen),
    }


def _gather_reference(inputs: Dict[str, Any]) -> Any:
    return inputs["x"].float().gather(-1, inputs["idx"])


register_ref("gather", RefSpec(_gather_inputs, _gather_reference))


def _unary_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {"x": torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen)}


def _silu_unary_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.nn.functional.silu(inputs["x"].float())


def _sigmoid_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.sigmoid(inputs["x"].float())


def _softplus_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.nn.functional.softplus(inputs["x"].float())


def _nonnegative_unary_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """`sqrt` is only real on x >= 0; in the router it reads softplus output."""
    inputs = _unary_inputs(inv, device, gen)
    return {"x": inputs["x"].abs()}


def _sqrt_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.sqrt(inputs["x"].float())


def _neg_exp_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.exp(-inputs["x"].float())


def _tanh_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.tanh(inputs["x"].float())


def _signed_sqrt_reference(inputs: Dict[str, Any]) -> Any:
    """The standalone cell is the PLE gate's node, so it takes the PLE
    lowering (`_signed_sqrt_times_sign`) — the engram span is fused by all
    three bindings and never reaches a standalone cell."""
    return _signed_sqrt_times_sign(inputs["x"].float())


def _gelu_unary_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.nn.functional.gelu(inputs["x"].float())


def _gelu_tanh_unary_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    return torch.nn.functional.gelu(inputs["x"].float(), approximate="tanh")


register_ref("silu", RefSpec(_unary_inputs, _silu_unary_reference))
register_ref("gelu", RefSpec(_unary_inputs, _gelu_unary_reference))
register_ref("gelu_tanh", RefSpec(_unary_inputs, _gelu_tanh_unary_reference))
register_ref("sigmoid", RefSpec(_unary_inputs, _sigmoid_reference))
register_ref("softplus", RefSpec(_unary_inputs, _softplus_reference))
register_ref("sqrt", RefSpec(_nonnegative_unary_inputs, _sqrt_reference))
register_ref("neg_exp", RefSpec(_unary_inputs, _neg_exp_reference))
register_ref("tanh", RefSpec(_unary_inputs, _tanh_reference))
register_ref("signed_sqrt", RefSpec(_unary_inputs, _signed_sqrt_reference))


def _binary_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "a": torch.randn(*_shape(inv, "a"), dtype=dt, device=device, generator=gen),
        "b": torch.randn(*_shape(inv, "b"), dtype=dt, device=device, generator=gen),
    }


def _mul_reference(inputs: Dict[str, Any]) -> Any:
    return inputs["a"].float() * inputs["b"].float()


def _add_reference(inputs: Dict[str, Any]) -> Any:
    return inputs["a"].float() + inputs["b"].float()


register_ref("mul", RefSpec(_binary_inputs, _mul_reference))
register_ref("add", RefSpec(_binary_inputs, _add_reference))


def _sigmoid_gate_mul_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `act` and `gated`: a · sigmoid(b)."""
    import torch
    return inputs["a"].float() * torch.sigmoid(inputs["b"].float())


register_ref("sigmoid_gate_mul",
             RefSpec(_binary_inputs, _sigmoid_gate_mul_reference))


#: The soft-cap every corpus model of the span runs
#: (`final_logit_softcapping`: Gemma-2-9B / 27B and Gemma-4-12B all 30.0). It
#: moves no byte and changes the function, so it rides on the inputs and the
#: oracle and every provider read the one value — the treatment the
#: hyper-connection seam's scalars get. The graph's `scale_const` carries no
#: value of its own to read it from.
SOFTCAP = 30.0


def _softcap_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """Logits on the scale a vocabulary row has before the cap — wide
    enough that tanh saturates on some of them, which is the point of it."""
    out = _unary_inputs(inv, device, gen)
    out["x"] = out["x"] * 20.0
    out["cap"] = SOFTCAP
    return out


def _softcap_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `scale_in`, `tanh`, `scale_out`: c · tanh(x / c)."""
    import torch
    c = float(inputs["cap"])
    return c * torch.tanh(inputs["x"].float() / c)


register_ref("softcap_logits", RefSpec(_softcap_inputs, _softcap_reference))


def _relu_squared_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    x = inputs["x"].float()
    return torch.relu(x) ** 2


def _situ_reference(inputs: Dict[str, Any]) -> Any:
    """`beta·tanh(x/beta)·sigmoid(x)` — Kimi-K3's gate at the beta it
    publishes (4.0). At beta 1 this is a different function that happens to
    be the one `tanh(x)·sigmoid(x)` used to check."""
    import torch
    x = inputs["x"].float()
    beta = float(inputs["situ_beta"])
    return beta * torch.tanh(x / beta) * torch.sigmoid(x)


def _scale_const_reference(inputs: Dict[str, Any]) -> Any:
    return inputs["x"].float() * 2.0


def _rmsnorm_unscaled_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    x = inputs["x"].float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def _sum_reduce_reference(inputs: Dict[str, Any]) -> Any:
    # x[rows, d_reduce, d_model]: the middle axis is the one the graph sums.
    return inputs["x"].float().sum(-2)


def _normalize_sum_one_reference(inputs: Dict[str, Any]) -> Any:
    x = inputs["x"].float()
    return x / (x.sum(-1, keepdim=True) + 1e-6)


def _rope_emb_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """One activation, the position of each of its rows, and the cos/sin
    table the rotation reads — `rope`'s operands for a single tensor."""
    import torch
    rows, d = _shape(inv, "x")
    positions = torch.arange(rows, device=device)
    return {
        "x": torch.randn(rows, d, dtype=_dtype(inv.dtype), device=device, generator=gen),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, int(d), 10000.0),
    }


def _rope_emb_reference(inputs: Dict[str, Any]) -> Any:
    """The rotation itself, at each row's position. GPT-J interleaving:
    the pairs are adjacent (`x[..., 0::2]`, `x[..., 1::2]`), which is what
    DeepSeek's `apply_rotary_emb` does to its rope slice."""
    x = inputs["x"].float()
    return _rope_rotate(x[:, None, :], inputs["positions"],
                        inputs["cos_sin_cache"], x.shape[-1],
                        is_neox=False).squeeze(1)


def _rope_emb_inverse_reference(inputs: Dict[str, Any]) -> Any:
    """The conjugate rotation: the same table read with sin negated."""
    import torch
    cache = inputs["cos_sin_cache"]
    half = cache.shape[-1] // 2
    conj = torch.cat([cache[:, :half], -cache[:, half:]], dim=-1)
    return _rope_emb_reference({**inputs, "cos_sin_cache": conj})


register_ref("relu_squared", RefSpec(_unary_inputs, _relu_squared_reference))
register_ref("situ", RefSpec(with_kernel_params(_unary_inputs, "situ_beta"),
                             _situ_reference))
register_ref("scale_const", RefSpec(_unary_inputs, _scale_const_reference))
register_ref("rmsnorm_unscaled",
             RefSpec(_unary_inputs, _rmsnorm_unscaled_reference))
register_ref("sum_reduce", RefSpec(_unary_inputs, _sum_reduce_reference))
register_ref("normalize_sum_one",
             RefSpec(_unary_inputs, _normalize_sum_one_reference))
def _fused_inv_rope_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The attention output [tokens, n_heads, d_head], each row's position,
    and the cos/sin table — `fused_inv_rope_fp8_quant`'s operands."""
    import torch
    b, n_heads, s_query, d_head = _shape(inv, "o")
    d_rope = int(_dims(inv)["d_rope"])
    tokens = int(b) * int(s_query)
    positions = torch.arange(tokens, device=device)
    out = inv.shape_out
    if inv.dtype_env.get("activation_granularity") is None:
        cast, groups = None, 1
    elif out is not None and len(out) == 4:
        cast = _quant_output_cast(inv, int(out[-1]))
        groups = int(out[1])
    else:
        # The signature's ungrouped canonical output: every head in one row.
        cast = _quant_output_cast(inv, n_heads * d_head)
        groups = 1
    return {
        "o": torch.randn(b, n_heads, s_query, d_head,
                         dtype=_dtype(inv.dtype), device=device, generator=gen),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, d_rope, 10000.0),
        "d_rope": d_rope,
        "n_groups": groups,
        "cast": cast,
    }


def _fused_inv_rope_reference(inputs: Dict[str, Any]) -> Any:
    """The last `d_rope` channels rotated back (conjugate frequencies), the
    rest passed through."""
    import torch
    # [B, n_heads, s_query, d_head] → the kernel's token-major rows.
    o = inputs["o"].float().permute(0, 2, 1, 3).flatten(0, 1)
    rd = int(inputs["d_rope"])
    cache = inputs["cos_sin_cache"]
    half = cache.shape[-1] // 2
    conj = torch.cat([cache[:, :half], -cache[:, half:]], dim=-1)
    head, tail = o[..., :-rd], o[..., -rd:]
    rotated = _rope_rotate(tail, inputs["positions"], conj, rd, is_neox=False)
    row = torch.cat([head, rotated], dim=-1)
    cast = inputs.get("cast")
    if cast is None:
        return row
    n_heads, d_head = row.shape[1], row.shape[2]
    groups = int(inputs["n_groups"])
    if n_heads % groups:
        raise ValueError(
            f"{n_heads} output heads do not divide into {groups} quant groups")
    row = row.reshape(row.shape[0], groups, n_heads // groups,
                      d_head).flatten(0, 1)
    return cast.on_grid(row.reshape(row.shape[0], -1))


def _norm_rope_tables(inv: Invocation, tokens: int, device: str) -> Dict[str, Any]:
    """The positions, cos/sin table, rotation width and eps both norm+rope
    launches take beside their activation."""
    import torch
    d_rope = int(_dims(inv)["d_rope"])
    positions = torch.arange(tokens, device=device)
    return {
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, d_rope, 10000.0),
        "d_rope": d_rope,
        "eps": 1e-6,
    }


def _qnorm_rope_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The query heads [B, n_heads, s_query, d_head] and the rope tables."""
    import torch
    b, n_heads, s_query, d_head = _shape(inv, "x")
    return {
        "x": torch.randn(b, n_heads, s_query, d_head,
                         dtype=_dtype(inv.dtype), device=device, generator=gen),
        **_norm_rope_tables(inv, int(b) * int(s_query), device),
    }


def _row_norm_rope_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """One row per token [B·s_query, d_head], the norm's gain, the tables."""
    import torch
    tokens, width = _shape(inv, "x")
    return {
        "x": torch.randn(tokens, width, dtype=_dtype(inv.dtype),
                         device=device, generator=gen),
        "w": torch.randn(*_shape(inv, "w"), dtype=_operand_dtype(inv, "w"),
                         device=device, generator=gen),
        **_norm_rope_tables(inv, int(tokens), device),
        # The record the launch writes the row in, when the cell states one
        # (`output_format`, `rows_on_record`): sglang's FlashMLA store.
        "output_format": inv.output_format,
    }


def _norm_rope_rows(n: Any, inputs: Dict[str, Any]) -> Any:
    """Normed token rows `[tokens, H, d]` with their last `d_rope` channels
    rotated at each row's position (GPT-J pairs)."""
    import torch
    rd = int(inputs["d_rope"])
    head, tail = n[..., :-rd], n[..., -rd:]
    rotated = _rope_rotate(tail, inputs["positions"],
                           inputs["cos_sin_cache"], rd, is_neox=False)
    return torch.cat([head, rotated], dim=-1)


def _qnorm_rope_reference(inputs: Dict[str, Any]) -> Any:
    """RMSNorm over each head's `d_head` without a gain, then the rotation;
    `[tokens, n_heads, d_head]`, the kernel's token-major rows."""
    import torch
    x = inputs["x"].float()
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"])
    return _norm_rope_rows(n.permute(0, 2, 1, 3).flatten(0, 1), inputs)


def _row_norm_rope_reference(inputs: Dict[str, Any]) -> Any:
    """RMSNorm of each row with its gain, then the rotation; `[tokens, d]`.

    A cell that names the record its launch writes (`output_format`: the
    FlashMLA record sglang's store writes, no bf16 row) is the row as that
    record holds it (`rows_on_record`), the tail rotated from the fp32
    normed row as every writer of that record rotates it."""
    import torch
    x = inputs["x"].float()
    n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs["eps"])
    n = n * inputs["w"].float()
    row = _norm_rope_rows(n[:, None, :], inputs)[:, 0]
    record = inputs.get("output_format")
    if record is None:
        return row
    return rows_on_record(row, record, inputs["x"].dtype,
                          tail=int(inputs["d_rope"]))


register_ref("fused_qnorm_rope",
             RefSpec(_qnorm_rope_inputs, _qnorm_rope_reference))
register_ref("fused_norm_rope",
             RefSpec(_row_norm_rope_inputs, _row_norm_rope_reference))


# ---------------------------------------------------------------------------
# DeepSeek-V4's whole attention input preparation, one launch — the fused
# spans `fused_qnorm_rope_kv_insert` and `..._no_q_norm`. The query heads
# normalised (where the span norms them) and rotated, the latent-KV row
# rotated, and that row written into the paged sliding-window cache at each
# token's slot.
#
# Whether the query is normed is stated on the INPUTS rather than closed over
# twice, as the ungrouped routers state their renormalisation: the oracle and
# every provider read the one value, so kernel and reference compute one
# function. It is a fact of the span (V4.1 norms `qr` before `wq_b`), not of
# the cell, and it moves no byte.
# ---------------------------------------------------------------------------

def _qnorm_rope_kv_insert_inputs(q_norm: bool):
    """The kernel's operands: the query heads and the latent row token-major,
    a fresh paged cache, the slot each token's record lands in, the
    positions, and the cos/sin table.

    The slots are `0..tokens-1` — this step's records written in order into a
    cache that holds nothing else. A served step writes the same one record
    per token at whatever slots its sequences are up to; the addresses differ,
    the launch does not.
    """
    def make(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
        import torch
        tokens, n_heads, d_head = _shape(inv, "q")
        dims = _dims(inv)
        d_rope, page = int(dims["d_rope"]), int(dims["kv_block_size"])
        dt = _dtype(inv.dtype)
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
        return {
            "q": torch.randn(tokens, n_heads, d_head, dtype=dt,
                             device=device, generator=gen),
            "kv": torch.randn(tokens, d_head, dtype=dt, device=device,
                              generator=gen),
            # One `d_head`-wide record per slot, in the record's own element
            # dtype (the cell's `kv` slot). The UE8M0 scales a packed record
            # carries beside it are the format's, not a cell axis — the
            # provider builds the packed layout its kernel takes.
            "kv_cache": torch.zeros(-(-tokens // page), page, d_head,
                                    dtype=_dtype(inv.kv_dtype), device=device),
            "slot_mapping": torch.arange(tokens, dtype=torch.int64,
                                         device=device),
            "positions": positions,
            "cos_sin_cache": _rope_cos_sin_cache(positions, d_rope, 10000.0),
            "d_rope": d_rope,
            "kv_block_size": page,
            "eps": 1e-6,
            "q_norm": q_norm,
            # The record the launch writes the latent row in, when the cell
            # states one (`kv_format`, `rows_on_record`): vLLM's insert
            # writes DeepSeek-V4's FlashMLA record.
            "kv_format": inv.kv_format,
        }
    return make


def _qnorm_rope_kv_insert_reference(inputs: Dict[str, Any]) -> Any:
    """The span's naive chain in fp32: the query heads normalised where this
    span norms them (RMSNorm over `d_head`, no gain), then the last `d_rope`
    channels of every query head and of the latent row rotated GPT-J style at
    the token's position.

    Returned as the launch's two outputs side by side,
    `[tokens, n_heads*d_head + d_head]`, because the cross-check compares one
    array (`driver._cross_check` takes the first element of a tuple) and both
    halves have to be in it: the rotated query is what attention reads next
    and the latent row is what the cache keeps, and a launch that got either
    one wrong is a launch that is wrong.

    A cell that names the record the cache keeps (`kv_format`: V4's
    FlashMLA record, FP8 e4m3 with one UE8M0 per 64 and the rotated tail
    bf16) gets the latent row as that record holds it (`rows_on_record`):
    the unrotated channels taken at the activation dtype and quantised, the
    tail rotated from the fp32 row and rounded once to bf16, as the kernel
    does (.cu:355-420). Kernel and oracle are then compared record to
    record on the activation's band (`driver._xcheck_band`), not an fp32
    row against a quantised one on the loose fp8 band. A cell that names no
    record keeps the fp32 row, banded by its `kv` dtype.
    """
    import torch
    rd = int(inputs["d_rope"])
    pos, cache = inputs["positions"], inputs["cos_sin_cache"]

    def rotate_tail(x):
        """`x[tokens, heads, d_head]` with its last `rd` channels rotated."""
        return torch.cat(
            [x[..., :x.shape[-1] - rd],
             _rope_rotate(x[..., x.shape[-1] - rd:], pos, cache, rd,
                          is_neox=False)], dim=-1)

    q = inputs["q"].float()
    if inputs["q_norm"]:
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + inputs["eps"])
    q = rotate_tail(q)
    kv = rotate_tail(inputs["kv"].float()[:, None, :])
    record = inputs.get("kv_format")
    if record is not None:
        kv = rows_on_record(kv, record, inputs["kv"].dtype, tail=rd)
    return torch.cat([q.flatten(1), kv.flatten(1)], dim=-1)


register_ref("fused_qnorm_rope_kv_insert",
             RefSpec(_qnorm_rope_kv_insert_inputs(True),
                     _qnorm_rope_kv_insert_reference))
register_ref("fused_qnorm_rope_kv_insert_no_q_norm",
             RefSpec(_qnorm_rope_kv_insert_inputs(False),
                     _qnorm_rope_kv_insert_reference))
register_ref("fused_inv_rope",
             RefSpec(_fused_inv_rope_inputs, _fused_inv_rope_reference))


# ---------------------------------------------------------------------------
# The compress/store spans of the KV-compressing attention families
# (DeepSeek-V4 / V4.1, GLM-5.3-Flash, Qwen QSA) — one launch per span,
# audited at vLLM v0.32.0 and sglang v0.5.22.
#
# Two shared conventions:
#   * `rms_normalise` is the one RMSNorm every oracle below composes.
#   * A cell whose launch writes a cached record names it
#     (`output_format`), and its oracle writes the SAME record
#     (`rows_on_record`, the grid read off the registered cache format):
#     kernel and oracle are then compared value for value on the activation
#     dtype's band, not an fp32 row against a quantised one. A launch whose
#     cell names no record (GLM's k-pool store, billed bf16;
#     `fused_qnorm_rope_kv_insert`, whose record is a secondary write the
#     cell states only by its `kv` dtype) keeps the fp32 oracle, banded by
#     that `kv` slot.
# ---------------------------------------------------------------------------

def _boundary_positions(tokens: int, compress_ratio: int, device: str,
                        batch: int = 1) -> Any:
    """The position of each pooled row's closing token, `(i+1)·ratio − 1`,
    restarting per request (`batch` requests of `tokens / batch` rows), as
    serving's positions do.

    The store launches gate on `(position + 1) % compress_ratio == 0` and
    rotate at `(position // ratio) · ratio`, so handing every pooled row its
    closing token's position writes every row and rotates it at its group's
    first token — the positions serving's own rows carry (the pooled row of
    a group belongs to the token that closes the group).
    """
    import torch
    per_req = tokens // batch
    return ((torch.arange(per_req, dtype=torch.int64, device=device) + 1)
            * compress_ratio - 1).repeat(batch)


def _save_compress_norm_inputs(inv: Invocation, device: str,
                               gen: Any) -> Dict[str, Any]:
    """The kernel's model operands: the kv and gate rows token-major and
    the compressor norm's gain. The fp32 state ring, the slot tables and
    the query offsets are the deployment's metadata; the provider builds
    them the way serving does (one request, the ring vLLM sizes)."""
    import torch
    dims = _dims(inv)
    dt = _dtype(inv.dtype)
    wdt = _store_dtype(inv.dtype_for("weight"))
    tokens = int(_shape(inv, "kv")[0])
    return {
        "kv": torch.randn(*_shape(inv, "kv"), dtype=dt, device=device,
                          generator=gen),
        "score": torch.randn(*_shape(inv, "score"), dtype=dt, device=device,
                             generator=gen),
        "norm_weight": torch.randn(*_shape(inv, "norm_weight"), dtype=wdt,
                                   device=device, generator=gen),
        "positions": torch.arange(tokens, dtype=torch.int64, device=device),
        "compress_ratio": int(dims["compress_ratio"]),
        "eps": 1e-6,
    }


def _save_compress_norm_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `smax`, `weighted`, `pooled`, `norm`: per closed group,
    softmax over the group per channel, the weighted sum of kv, then
    rmsnorm·w — `[groups, d_head]` in group (= closing-token) order."""
    import torch
    kv, score = inputs["kv"].float(), inputs["score"].float()
    cr = int(inputs["compress_ratio"])
    groups = kv.shape[0] // cr
    kvg = kv[:groups * cr].view(groups, cr, -1)
    scg = score[:groups * cr].view(groups, cr, -1)
    pooled = (kvg * torch.softmax(scg, dim=1)).sum(dim=1)
    return rms_normalise(pooled, inputs["norm_weight"], inputs["eps"])


register_ref("fused_save_compress_norm",
             RefSpec(_save_compress_norm_inputs,
                     _save_compress_norm_reference))


def _compress_pool_state_inputs(inv: Invocation, device: str,
                                gen: Any) -> Dict[str, Any]:
    """The kv and gate rows token-major, as `fused_save_compress_norm`
    takes them; the fp32 state ring and the sequence tables are the
    deployment's metadata, which the provider builds."""
    import torch
    dt = _dtype(inv.dtype)
    return {
        "kv": torch.randn(*_shape(inv, "kv"), dtype=dt, device=device,
                          generator=gen),
        "score": torch.randn(*_shape(inv, "score"), dtype=dt, device=device,
                             generator=gen),
        "compress_ratio": int(_dims(inv)["compress_ratio"]),
    }


def _compress_pool_state_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `weights`, `weighted`, `pooled_row`: per closed group of
    consecutive rows, softmax over the group per channel and the weighted
    sum of kv — `[groups, d_head]` in group order."""
    import torch
    kv, score = inputs["kv"].float(), inputs["score"].float()
    cr = int(inputs["compress_ratio"])
    groups = kv.shape[0] // cr
    kvg = kv[:groups * cr].view(groups, cr, -1)
    scg = score[:groups * cr].view(groups, cr, -1)
    return (kvg * torch.softmax(scg, dim=1)).sum(dim=1)


register_ref("fused_compress_pool_state",
             RefSpec(_compress_pool_state_inputs,
                     _compress_pool_state_reference))


def _compress_rope_store_inputs(inv: Invocation, device: str,
                                gen: Any) -> Dict[str, Any]:
    """The store's operands: the pooled latent rows, a fresh paged cache in
    the record's element dtype, each row's slot, the closing tokens'
    positions and the cos/sin table."""
    import torch
    dims = _dims(inv)
    dt = _dtype(inv.dtype)
    slots_n, d_head = _shape(inv, "latent")
    page, cr = int(dims["kv_block_size"]), int(dims["compress_ratio"])
    d_rope = int(dims["d_rope"])
    positions = _boundary_positions(slots_n, cr, device,
                                    batch=int(dims["B"]))
    return {
        "latent": torch.randn(slots_n, d_head, dtype=dt, device=device,
                              generator=gen),
        # One `d_head`-wide record per slot, in the record's own element
        # dtype (the cell's `kv` slot; an NVFP4 cell's is carried at the
        # dtype `_operand_dtype` falls back to, as every packed format is).
        # The packed record's scale bytes are the format's, not a cell
        # axis — the provider builds the packed layout its kernel writes.
        "kv_cache": torch.zeros(-(-slots_n // page), page, d_head,
                                dtype=_operand_dtype(inv, "kv_cache"),
                                device=device),
        "slot_mapping": torch.arange(slots_n, dtype=torch.int64,
                                     device=device),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, d_rope, 10000.0),
        "compress_ratio": cr,
        "d_rope": d_rope,
        "kv_block_size": page,
        "eps": 1e-6,
        # The record the launch writes the row in (`rows_on_record`).
        "output_format": inv.output_format,
    }


def _compress_rope_store_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `pe`/`nope` slices, `rope` and `row` concat: the latent
    with its last `d_rope` channels rotated GPT-J style at each row's group
    position — `[slots, d_head]` in slot order; as the record holds it
    when the cell names one (`output_format`, `rows_on_record`)."""
    import torch
    x = inputs["latent"].float()
    rd, cr = int(inputs["d_rope"]), int(inputs["compress_ratio"])
    pos = inputs["positions"] // cr * cr
    rotated = _rope_rotate(x[:, None, -rd:], pos, inputs["cos_sin_cache"],
                           rd, is_neox=False)[:, 0]
    row = torch.cat([x[..., :-rd], rotated], dim=-1)
    record = inputs.get("output_format")
    if record is None:
        return row
    return rows_on_record(row, record, inputs["latent"].dtype, tail=rd)


register_ref("fused_compress_rope_store",
             RefSpec(_compress_rope_store_inputs,
                     _compress_rope_store_reference))


def _compress_norm_rope_store_inputs(inv: Invocation, device: str,
                                     gen: Any) -> Dict[str, Any]:
    """The no-overlap compressor tail's operands: the pooled kv rows the
    previous launch wrote, the compressor norm's gain, and the store's
    cache / slot / position / table operands (as
    `_compress_rope_store_inputs`)."""
    import torch
    dims = _dims(inv)
    dt = _dtype(inv.dtype)
    wdt = _store_dtype(inv.dtype_for("weight"))
    slots_n, d_head = _shape(inv, "kv")
    page, cr = int(dims["kv_block_size"]), int(dims["compress_ratio"])
    d_rope = int(dims["d_rope"])
    positions = _boundary_positions(slots_n, cr, device,
                                    batch=int(dims["B"]))
    return {
        "kv": torch.randn(slots_n, d_head, dtype=dt, device=device,
                          generator=gen),
        "norm_weight": torch.randn(*_shape(inv, "norm_weight"), dtype=wdt,
                                   device=device, generator=gen),
        "kv_cache": torch.zeros(-(-slots_n // page), page, d_head,
                                dtype=_operand_dtype(inv, "kv_cache"),
                                device=device),
        "slot_mapping": torch.arange(slots_n, dtype=torch.int64,
                                     device=device),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, d_rope, 10000.0),
        "compress_ratio": cr,
        "d_rope": d_rope,
        "kv_block_size": page,
        "eps": 1e-6,
        # The record the launch writes the row in, when the cell states one
        # (`output_format`, `rows_on_record`).
        "output_format": inv.output_format,
    }


#: The floor a record writer puts under a group's amax before deriving its
#: scale, by the record's element dtype — measured off the audited writers'
#: sources (`_record_amax_floor` names them). Its keys are dtype tokens,
#: checked against `arch.dtypes` at import, so a misspelt key cannot sit
#: unread while that dtype's records raise.
_RECORD_AMAX_FLOOR: Dict[str, float] = {"fp8_e4m3": 1e-4,
                                        "fp4_e2m1": 6.0 * 2.0 ** -126}
for _element in _RECORD_AMAX_FLOOR:
    if _element not in CORPUS_TOKENS:
        raise ValueError(
            f"record amax floor keyed at {_element!r}, which is not a dtype "
            f"token of arch.dtypes")


def _record_amax_floor(element: str) -> float:
    """The floor a record writer puts under a group's amax before deriving
    its scale, by the record's element dtype — the one fact of a record's
    grid the cache format does not declare, because it is the writers'
    convention and not the storage's (it moves only a group whose amax is
    below it).

    Every audited writer of an FP8 record floors at 1e-4: vLLM's V4 insert
    (csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:384-386), its V4
    Triton stores (models/deepseek_v4/common/ops/fused_compress_quant_cache
    .py:284, :505, :867), V4.1's rope_quant_insert and index key
    (deepseek_v41/common/ops/fused_compress_quant_cache.py:330,
    indexer_k_store.py:242), sglang's FlashMLA store (kernels/jit/csrc/
    deepseek_v4/main_norm_rope.cuh:427). vLLM's two CuteDSL compressor
    stores floor the SCALE at 1e-4 instead (`min_scale`,
    sparse_attn_compress_cutedsl.py:81, :343-346, :1456, :1601-1604) — the
    one writer that deviates, visible only on a group whose amax is under
    0.0448. The FP4 writers floor at `6 · 2^-126`, the smallest normal
    scale (indexer_k_store.py:222).

    An element dtype no audited writer covers raises: a floor invented here
    is a scale no kernel derived, and the oracle would then disagree with a
    correct kernel on every near-zero group.
    """
    try:
        return _RECORD_AMAX_FLOOR[element]
    except KeyError:
        raise ValueError(
            f"no audited writer's amax floor for a {element!r} record; the "
            f"floor is the writers' convention and is read off their "
            f"sources, not guessed") from None


def _on_grid(fmt: Any) -> bool:
    """Whether `rows_on_record` can place a row on cache format `fmt`'s
    grid: a per-group scale, and a write transform (applied by its op's own
    oracle first) only on a row with no unquantised tail."""
    return (fmt.scale is not None and fmt.scale.granularity == "per-group"
            and (fmt.transform_op is None or fmt.tail is None))


def records_on_grid() -> Tuple[Optional[str], ...]:
    """The format-slot declaration (`KernelSignature.output_format` /
    `.kv_format`) of a floor that writes the record the cell names through
    `rows_on_record`, and the row itself where it names none: the plain
    tensor (`benchmark_identity.PLAIN`) and every registry format whose
    grid `rows_on_record` places — read off the registry, so a format added
    there is served or declined by the same rule that writes it."""
    from backend.arch.cache_formats import CACHE_FORMATS
    from backend.inference.benchmark_identity import PLAIN
    return (PLAIN, *sorted(
        name for name, fmt in CACHE_FORMATS.items() if _on_grid(fmt)))


def rows_on_record(row: Any, record: str, act_dtype: Any,
                   tail: Optional[int] = None) -> Any:
    """`row[..., d]` as the cached record `record` (a cache-format name, the
    cell's `output_format`) stores it, decoded back — fp32.

    The grid is the registered format's (`backend/arch/cache_formats/`):
    its element dtype picks the levels (`quant.FORMATS`), its scale's group
    how many channels share one scale, its scale dtype (or `power_of_two`)
    how the scale is held (`quant.activation_scale_encoding`: UE8M0 and a
    power-of-two F32 are both `amax / peak` rounded UP to a power of two,
    as every writer rounds it), and its unquantised tail the last channels
    stored as they are at the tail's dtype (`tail` binds the tail's channel
    count in the cell's frame — the rotated width). Only the amax floor is
    not the format's (`_record_amax_floor`).

    The quantised channels are taken at the activation dtype first: a
    record quantises the activation as the model holds it, which is what
    the reference writes (`act_quant` of the bf16 row) and what vLLM's
    writers do ("cast fp32 → bf16 → fp32 before quant to match reference",
    deepseek_v4/common/ops/fused_compress_quant_cache.py:275-280). The tail
    is rounded once, from `row`, to its own dtype.

    A record whose write rotates the row (`CacheFormat.transform`,
    `fp8_pow2_hadamard128`) is rotated first, by the oracle of the op the
    format names (`CacheFormat.transform_op`, the same op the graph grafts
    on the edge into the cache): the writers rotate the row, then round and
    quantise it (vLLM kpool_compress.py:219, sglang kpool_fp8_index.py:984).

    Through `quant.quantise_groups`, the one implementation of a grid. A
    record whose grid that cannot state (no scale, a scale not per group)
    raises rather than leaving the row off its grid."""
    import torch
    from backend.arch.cache_formats import CACHE_FORMATS
    from backend.arch.dtypes import corpus_token
    from .quant import FORMATS, activation_scale_encoding, quantise_groups
    fmt = CACHE_FORMATS.get(record)
    if fmt is None:
        raise ValueError(f"{record!r} is not a registry cache format")
    if not _on_grid(fmt):
        raise ValueError(f"the record {record!r} has no oracle grid: only a "
                         f"per-group scale, rotating no unquantised tail, "
                         f"is placed on one here")
    if fmt.transform_op is not None:
        row = get_ref(fmt.transform_op).reference({"x": row})
    if fmt.tail is not None:
        if tail is None:
            raise ValueError(f"the record {record!r} keeps its last channels "
                             f"unquantised; the caller binds how many")
        fmt = fmt.bound(int(tail))
    width = int(row.shape[-1])
    kept = fmt.tail_channels(width)
    element = corpus_token(fmt.dtype)
    encoding = ("pow2" if fmt.scale.power_of_two
                else activation_scale_encoding(corpus_token(fmt.scale.dtype)))
    body = row[..., :width - kept].to(act_dtype)
    q, scale = quantise_groups(body, FORMATS[element], fmt.scale.group,
                               encoding, act_dtype,
                               eps=_record_amax_floor(element))
    out = (q.float() * scale.unsqueeze(-1)).reshape(body.shape)
    if not kept:
        return out
    held = row[..., width - kept:].to(
        torch_dtype(corpus_token(fmt.tail.dtype))).float()
    return torch.cat([out, held], dim=-1)


def record_step(record: str) -> float:
    """One rounding step of the record `record`'s element grid, relative
    (`quant.WeightFormat.step`): the most a launch that writes the same
    record may differ from its oracle on one element when the two computed
    the value before the write a hair apart — a fused kernel that keeps the
    normed row in fp32 where the reference rounds it to bf16 (sglang's
    FlashMLA store) lands one step away on the few values that sat at a
    rounding boundary. `driver._xcheck_band` reads it."""
    from backend.arch.cache_formats import CACHE_FORMATS
    from backend.arch.dtypes import corpus_token
    from .quant import FORMATS
    return FORMATS[corpus_token(CACHE_FORMATS[record].dtype)].step


def _compress_norm_rope_store_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `norm`, slices, `rope`, `row`: the pooled row RMS-normed
    with the gain, then its last `d_rope` channels rotated at the group's
    position — `[slots, d_head]` in slot order.

    A cell that names the record its launch writes (`output_format`:
    DeepSeek-V4.1's index key, DeepSeek-V4's FlashMLA record) is the row
    as that record holds it (`rows_on_record`), rounded as the reference
    rounds before writing it: the normed row to the activation dtype, then
    rotated (V4.1's index-key kernel does the same, indexer_k_store.py:162,
    :188-189; V4's CuteDSL store rotates the unrounded fp32 row, which
    moves only the BF16 tail, by at most one BF16 step). A cell with no
    record is the fp32 row."""
    import torch
    record = inputs.get("output_format")
    y = rms_normalise(inputs["kv"], inputs["norm_weight"], inputs["eps"])
    if record is not None:
        y = y.to(inputs["kv"].dtype).float()
    rd, cr = int(inputs["d_rope"]), int(inputs["compress_ratio"])
    pos = inputs["positions"] // cr * cr
    rotated = _rope_rotate(y[:, None, -rd:], pos, inputs["cos_sin_cache"],
                           rd, is_neox=False)[:, 0]
    row = torch.cat([y[..., :-rd], rotated], dim=-1)
    if record is None:
        return row
    return rows_on_record(row, record, inputs["kv"].dtype, tail=rd)


register_ref("fused_compress_norm_rope_store",
             RefSpec(_compress_norm_rope_store_inputs,
                     _compress_norm_rope_store_reference))


def _overlap_compress_inputs(inv: Invocation, device: str,
                             gen: Any) -> Dict[str, Any]:
    """The overlap compressor's operands: the window rows — each window's
    `Cape` positions, two halves (Ca|Cb) wide, kv and gate-with-ape alike —
    the norm's gain, and the store operands. Positions restart per request:
    window w of a request closes at position `(w+1)·Cape − 1`."""
    import torch
    dims = _dims(inv)
    dt = _dtype(inv.dtype)
    wdt = _store_dtype(inv.dtype_for("weight"))
    rows, two_d = _shape(inv, "kv")
    cape, d_head = int(dims["Cape"]), int(dims["d_head"])
    batch, s_slots = int(dims["B"]), int(dims["s_slots"])
    page = int(dims["kv_block_size"])
    d_rope = int(dims["d_rope"])
    windows = batch * s_slots
    positions = ((torch.arange(s_slots, dtype=torch.int64,
                               device=device) + 1) * cape - 1
                 ).repeat(batch)
    return {
        "kv": torch.randn(rows, two_d, dtype=dt, device=device, generator=gen),
        "score": torch.randn(*_shape(inv, "score"), dtype=dt, device=device,
                             generator=gen),
        "norm_weight": torch.randn(*_shape(inv, "norm_weight"), dtype=wdt,
                                   device=device, generator=gen),
        "kv_cache": torch.zeros(-(-windows // page), page, d_head,
                                dtype=_operand_dtype(inv, "kv_cache"),
                                device=device),
        "slot_mapping": torch.arange(windows, dtype=torch.int64,
                                     device=device),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, d_rope, 10000.0),
        "Cape": cape,
        "d_rope": d_rope,
        "kv_block_size": page,
        "windows_per_request": s_slots,
        "eps": 1e-6,
        # The record the launch writes the row in (`rows_on_record`).
        "output_format": inv.output_format,
    }


def _overlap_compress_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's pairs, `smax`, `weighted`, `pooled`, `norm`, `rope`,
    `row`: the softmax pool over the two windows' halves, RMS-normed, the
    last `d_rope` channels rotated at the window's first position —
    `[windows, d_head]` in (request, window) order.

    The shift is PER REQUEST: the kernel gathers a window's rows through
    that request's own block table (`fused_compress_quant_cache.py:204-216`,
    :768-780), so a request's first window has no previous window and pools
    its own Cb half alone — flattening the batch and shifting across it
    would hand request r>0 the previous request's last Ca half, a different
    function at every request boundary.

    A cell that names the record its launch writes (`output_format`) is
    the row as that record holds it (`rows_on_record`)."""
    import torch
    kv, score = inputs["kv"].float(), inputs["score"].float()
    cape, eps = int(inputs["Cape"]), inputs["eps"]
    per_req = int(inputs["windows_per_request"])
    d = kv.shape[1] // 2
    kvw = kv.view(-1, per_req, cape, 2 * d)
    scw = score.view(-1, per_req, cape, 2 * d)
    prev_k = torch.cat([torch.zeros_like(kvw[:, :1, :, :d]),
                        kvw[:, :-1, :, :d]], dim=1)
    prev_s = torch.cat([torch.full_like(scw[:, :1, :, :d], float("-inf")),
                        scw[:, :-1, :, :d]], dim=1)
    pair_k = torch.cat([prev_k, kvw[..., d:]], dim=2)
    pair_s = torch.cat([prev_s, scw[..., d:]], dim=2)
    pooled = (pair_k * torch.softmax(pair_s, dim=2)).sum(dim=2)
    y = rms_normalise(pooled, inputs["norm_weight"], eps).flatten(0, 1)
    rd, cape = int(inputs["d_rope"]), int(inputs["Cape"])
    pos = inputs["positions"] // cape * cape
    rotated = _rope_rotate(y[:, None, -rd:], pos, inputs["cos_sin_cache"],
                           rd, is_neox=False)[:, 0]
    row = torch.cat([y[..., :-rd], rotated], dim=-1)
    record = inputs.get("output_format")
    if record is None:
        return row
    # Rotated from the fp32 normed row, then written: both writers rotate
    # before rounding (sparse_attn_compress_cutedsl.py:296-310;
    # fused_compress_quant_cache.py:856-865).
    return rows_on_record(row, record, inputs["kv"].dtype, tail=rd)


register_ref("fused_overlap_compress_norm_rope_store",
             RefSpec(_overlap_compress_inputs,
                     _overlap_compress_reference))


def hadamard_rotate(x: Any) -> Any:
    """`x [..., n] · H_n / √n` in fp32 — the op `hadamard_transform`'s own
    function (`y = (H_n · x)/sqrt(n)`), whatever the row's width."""
    import torch
    y = x.float()
    return torch.matmul(y, _hadamard_matrix(y.shape[-1], y.device)) \
        * y.shape[-1] ** -0.5


def _hadamard_inputs(inv: Invocation, device: str, gen: Any
                     ) -> Dict[str, Any]:
    """One activation row block `[rows, n]` at the cell's dtype, and the
    dtype the launch writes (`output`)."""
    import torch
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=_dtype(inv.dtype),
                         device=device, generator=gen),
        "output": inv.dtype_for("output"),
    }


def _hadamard_reference(inputs: Dict[str, Any]) -> Any:
    """The rotation of each row, fp32."""
    return hadamard_rotate(inputs["x"])


register_ref("hadamard_transform",
             RefSpec(_hadamard_inputs, _hadamard_reference))


#: The FP8 grid `fwht128_quant_fp8` writes the rotated query on: e4m3, one
#: power-of-two scale per 128-wide row, the amax floored at 1e-4
#: (vLLM models/glm5next/nvidia/ops/kpool_compress.py:100-102).
_FWHT_QUANT_GROUP = 128


def _fwht_quant_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `rotate`, then — when the cell's launch writes FP8 — its
    `quant`: the rotated row rounded to the activation dtype, as the kernel
    rounds it before the quantiser (kpool_compress.py:97), then on the
    pow2-scaled e4m3 grid. A cell writing the activation dtype is the
    rotation alone."""
    from .quant import FORMATS, quantise_groups
    x = inputs["x"]
    rotated = hadamard_rotate(x)
    if inputs["output"] != "fp8_e4m3":
        return rotated
    q, scale = quantise_groups(rotated.to(x.dtype), FORMATS["fp8_e4m3"],
                               _FWHT_QUANT_GROUP, "pow2", x.dtype,
                               eps=_record_amax_floor("fp8_e4m3"))
    return (q.float() * scale.unsqueeze(-1)).reshape(rotated.shape)


register_ref("fused_fwht128_quant",
             RefSpec(_hadamard_inputs, _fwht_quant_reference))


def _kpool_compress_inputs(inv: Invocation, device: str,
                           gen: Any) -> Dict[str, Any]:
    """The pool's operands: the gate and key windows `[pools, ratio, d]`,
    the position-bias table, a fresh paged index-K cache and each pool's
    slot."""
    import torch
    dims = _dims(inv)
    dt = _dtype(inv.dtype)
    page = int(dims["kv_block_size"])
    pools = int(_shape(inv, "slot_mapping")[0])
    return {
        "gate": torch.randn(*_shape(inv, "gate"), dtype=dt, device=device,
                            generator=gen),
        "keys": torch.randn(*_shape(inv, "keys"), dtype=dt, device=device,
                            generator=gen),
        # The learned position bias. The kernels hard-require it fp32
        # (kpool_compress.py:293 and kpool_fp8_index.py:717 assert it);
        # whatever the cell's weight slot declares, the provider hands the
        # kernel the fp32 it asks for.
        "ape": torch.randn(*_shape(inv, "ape"),
                           dtype=_operand_dtype(inv, "ape"), device=device,
                           generator=gen),
        "kv_cache": torch.zeros(-(-pools // page), page,
                                int(dims["d_indexer_head"]),
                                dtype=_operand_dtype(inv, "kv_cache"),
                                device=device),
        "slot_mapping": torch.arange(pools, dtype=torch.int64,
                                     device=device),
        "compress_ratio": int(dims["compress_ratio"]),
        # The record the launch writes the pooled key in (`rows_on_record`).
        "output_format": inv.output_format,
    }


def _kpool_compress_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `pool_logits`, `pool_probs`, `pool_weighted`,
    `pool_keys`: softmax(gate + ape) per channel over the pool, the
    weighted sum of keys rounded to the activation dtype (the kernel's own
    bf16 round-trip, "match the unfused pooled-K path's bf16 precision") —
    `[pools, d_indexer_head]`. A cell that names the record its launch
    writes (`output_format`) is the pooled key as that record holds it
    (`rows_on_record`): rotated exactly when the record's format carries
    the transform — the plan's `rotate` member, grafted from that same
    format — rounded, and on the record's grid. A cell with no record is
    the pooled key."""
    import torch
    dt = inputs["keys"].dtype
    cr, d = int(inputs["compress_ratio"]), int(inputs["keys"].shape[-1])
    gate = inputs["gate"].float().view(-1, cr, d)
    keys = inputs["keys"].float().view(-1, cr, d)
    pooled = (keys * torch.softmax(gate + inputs["ape"].float()[None],
                                   dim=1)).sum(dim=1).to(dt)
    record = inputs.get("output_format")
    if record is None:
        return pooled.float()
    return rows_on_record(pooled, record, dt)


register_ref("fused_kpool_compress_store",
             RefSpec(_kpool_compress_inputs, _kpool_compress_reference))


def _qsa_q_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The indexer query prep's operands: the head-split query, the Gemma
    norm's weight, the per-token positions and the fp32 cos/sin table both
    frameworks' fused-prep gates require (`_use_fused_prep`, sglang
    qsa_indexer.py:126-138)."""
    import torch
    dims = _dims(inv)
    dt = _dtype(inv.dtype)
    wdt = _store_dtype(inv.dtype_for("weight"))
    tokens = int(_shape(inv, "q")[0])
    d_rope = int(dims["d_rope"])
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    return {
        "q": torch.randn(*_shape(inv, "q"), dtype=dt, device=device,
                         generator=gen),
        "weight": torch.randn(*_shape(inv, "weight"), dtype=wdt,
                              device=device, generator=gen),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, d_rope, 10000.0),
        "eps": 1e-6,
    }


def _qsa_q_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `rms`, `norm`, `pe`/`nope` slices, `rope` and `cat`, per
    head: rmsnorm(q)·(1 + w) — the spec's own formula
    (`qwen_qsa_attention.yaml`: `y = norm · (1 + w)`) — then the first
    `d_rope` channels rotated NeoX style at the token's position,
    re-joined; `[tokens, n_indexer_heads·d_indexer_head]`."""
    import torch
    q = inputs["q"].float()
    rd = int(inputs["cos_sin_cache"].shape[-1])
    normed = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + inputs["eps"])
    normed = normed * (1.0 + inputs["weight"].float())
    rotated = _rope_rotate(normed[..., :rd], inputs["positions"],
                           inputs["cos_sin_cache"], rd, is_neox=True)
    return torch.cat([rotated, normed[..., rd:]], dim=-1).flatten(1)


register_ref("qsa_index_q_norm_rope",
             RefSpec(_qsa_q_inputs, _qsa_q_reference))


def _qsa_k_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The key compression's operands: the raw keys token-major, the Gemma
    weight, a fresh compressed cache in the record's dtype, each completed
    block's slot, the per-token positions and the table."""
    import torch
    dims = _dims(inv)
    dt = _dtype(inv.dtype)
    wdt = _store_dtype(inv.dtype_for("weight"))
    tokens, d_head = _shape(inv, "raw_key")
    cr, page = int(dims["compress_ratio"]), int(dims["kv_block_size"])
    # One slot per completed block, as the contract's `slot_mapping` lays
    # them out (`B · s_slots`, the walk's floor(s_query / ratio) a request).
    groups = int(_shape(inv, "slot_mapping")[0])
    d_rope = int(dims["d_rope"])
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    return {
        "raw_key": torch.randn(tokens, d_head, dtype=dt, device=device,
                               generator=gen),
        "weight": torch.randn(*_shape(inv, "weight"), dtype=wdt,
                              device=device, generator=gen),
        "kv_cache": torch.zeros(-(-groups // page), page, d_head,
                                dtype=_operand_dtype(inv, "kv_cache"),
                                device=device),
        "slot_mapping": torch.arange(groups, dtype=torch.int64,
                                     device=device),
        "positions": positions,
        "cos_sin_cache": _rope_cos_sin_cache(positions, d_rope, 10000.0),
        "compress_ratio": cr,
        "d_rope": d_rope,
        "kv_block_size": page,
        "eps": 1e-6,
    }


def _qsa_k_reference(inputs: Dict[str, Any]) -> Any:
    """The plan's `win`, `sum`, `mean`, `rms`, `norm`, `pe`/`nope`, `rope`,
    `cat`: the fp32 block mean rounded to the storage dtype (both kernels
    state it — "match the unfused path's BF16 pooled tensor", sglang
    "rounded to the storage dtype exactly like average_pool_qsa_keys"),
    rmsnorm·(1 + w), the first `d_rope` channels rotated NeoX style at
    the block's first position — `[blocks, d_indexer_head]` in block
    order."""
    import torch
    x, cr = inputs["raw_key"].float(), int(inputs["compress_ratio"])
    d = x.shape[-1]
    pooled = x.view(-1, cr, d).mean(dim=1).to(inputs["raw_key"].dtype).float()
    normed = pooled * torch.rsqrt(pooled.pow(2).mean(-1, keepdim=True)
                                  + inputs["eps"])
    normed = normed * (1.0 + inputs["weight"].float())
    rd = int(inputs["cos_sin_cache"].shape[-1])
    pos = inputs["positions"][::cr]
    rotated = _rope_rotate(normed[:, None, :rd], pos,
                           inputs["cos_sin_cache"], rd, is_neox=True)[:, 0]
    return torch.cat([rotated, normed[:, rd:]], dim=-1)


register_ref("qsa_index_k_compress_store",
             RefSpec(_qsa_k_inputs, _qsa_k_reference))

register_ref("rope_emb", RefSpec(_rope_emb_inputs, _rope_emb_reference))
register_ref("rope_emb_inverse",
             RefSpec(_rope_emb_inputs, _rope_emb_inverse_reference))


def _hash_route_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    table = _shape(inv, "tid2eid")
    n_vocab, n_exp = int(table[0]), int(table[-1])
    ids = torch.randint(0, max(n_vocab, 1), _shape(inv, "ids"),
                        device=device, generator=gen)
    # The table's entries are expert ids, and which ids they are moves no
    # byte and no work of the gather: the cell is the table's shape, so the
    # values are drawn from its own width `k` (`n_experts_active`, so a
    # row's ids are in `[0, k)` and need not be distinct experts — harmless
    # for a gather) rather than from an expert count the cell does not
    # carry.
    return {
        "ids": ids,
        "tid2eid": torch.randint(0, max(n_exp, 1), table, device=device,
                                 generator=gen),
    }


def _hash_route_reference(inputs: Dict[str, Any]) -> Any:
    return inputs["tid2eid"][inputs["ids"].long()].float()


register_ref("hash_route", RefSpec(_hash_route_inputs, _hash_route_reference))


#: KNOWN INPUT-COVERAGE GAP (stated 2026-10-22, not fixed here).
#: Every cell of the hyper-connection seam is built at the IDENTITY
#: parameterisation of the mix split — `scale` all ones, `base` all zeros —
#: so the gates read the projection's dots as they are. That makes the cells
#: readable, and it means THE NON-IDENTITY SCALE/BASE PATH IS EXERCISED BY NO
#: CROSS-CHECK ANYWHERE: a defect that lives only in how a kernel applies a
#: non-unit `scale` or non-zero `base` is invisible to every gate here, on
#: both this span and `mhc_pre_big_fuse`, because no generated input leaves
#: the identity. Same family as the removed `hc_split_sinkhorn` loop, whose
#: `clamp_min` on a signed row sum was reachable only from inputs the
#: generator did draw: what the harness generates decides what it can see.
#: Widening the generator is its own change with its own review.
def _hc_split_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "mixes": torch.randn(*_shape(inv, "mixes"), dtype=dt, device=device, generator=gen),
        "scale": torch.ones(*_shape(inv, "scale"), dtype=torch.float32,
                            device=device),
        "base": torch.zeros(*_shape(inv, "base"), dtype=dt, device=device),
        "n_hidden_copies": int(_dims(inv)["n_hidden_copies"]),
    }


def _hc_split_reference(inputs: Dict[str, Any]) -> Any:
    """The frameworks' split of the mix vector — `hc_split_coefficients`.

    WHAT THIS USED TO BE, and why it had to change. It was:

        x = mixes + base
        for _ in range(8):  x = x / x.sum(-1, keepdim=True)
        return x[..., :n]

    wrong three ways independently of the frameworks. One-sided row
    normalisation is IDEMPOTENT after the first pass — once a row sums to
    one, dividing by its row sum again is the identity — so seven of the
    eight rounds were dead and the whole loop was `(mixes + base) / rowsum`.
    That is not Sinkhorn, which alternates row and column normalisation;
    the alternation is the entire reason the result is doubly stochastic.
    And it returned only the `pre` slice, where this op's own doc says it
    splits into pre/post/comb — so two of the three writes went unchecked.

    What the frameworks compute: sigmoid gates for `pre` and `post`, and a
    softmax plus 20 alternating Sinkhorn rounds for `comb`. The three writes
    are returned side by side, so the one-tensor cross-check covers all of
    them, which is how the epilogue span states the same three.

    The operand is the ALREADY-NORMALISED mix vector: this op's inputs carry
    no sum-of-squares, because the rsqrt belongs to the projection that
    writes the vector, not to this split.
    """
    import torch
    hc = int(inputs["n_hidden_copies"])
    pre, post, comb = hc_split_coefficients(
        inputs["mixes"].float().reshape(-1, inputs["mixes"].shape[-1]),
        inputs["scale"], inputs["base"].float(), hc,
        pre_eps=_HC_EPS, post_mult=_HC_POST_MULT,
        sinkhorn_eps=_HC_EPS, sinkhorn_iters=_HC_SINKHORN_ITERS)
    return torch.cat([pre, post, comb.flatten(1)], dim=-1)


register_ref("hc_split_sinkhorn",
             RefSpec(_hc_split_inputs, _hc_split_reference))


def _hc_mix_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "comb": torch.randn(*_shape(inv, "comb"), dtype=dt, device=device, generator=gen),
        "residual": torch.randn(*_shape(inv, "residual"), dtype=dt,
                                device=device, generator=gen),
    }


def _hc_mix_reference(inputs: Dict[str, Any]) -> Any:
    return (inputs["comb"].float() @ inputs["residual"].float())


register_ref("hc_mix", RefSpec(_hc_mix_inputs, _hc_mix_reference))


def _gla_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "q": torch.randn(*_shape(inv, "q"), dtype=dt, device=device, generator=gen),
        "k": torch.randn(*_shape(inv, "k"), dtype=dt, device=device, generator=gen),
        "v": torch.randn(*_shape(inv, "v"), dtype=dt, device=device, generator=gen),
        # A positive decay rate per head, exp(-slope) in (e^-1, 1) a step.
        # Serving builds it from ALiBi-style slopes and the layer index; the
        # value moves no bytes and no work, so any positive rate is a valid
        # measurement, and a drawn one lets the cross-check see a provider
        # that drops the decay or passes it with the wrong sign.
        "slope": torch.rand(*_shape(inv, "slope"), dtype=torch.float32,
                            device=device, generator=gen),
        # NON-zero, unlike a fresh sequence's: a state of zeros gives the
        # same output whether or not the kernel reads it, and the bug this
        # operand exists for was a launch that never read it. fp32, as the
        # component declares it (`recurrent_state` dtype fp32) and as every
        # kernel keeps it.
        "initial_state": torch.randn(*_shape(inv, "initial_state"),
                                     dtype=torch.float32, device=device,
                                     generator=gen),
    }


def _gla_scan(q, k, v, slope, initial_state, final_state=None):
    """The simple-GLA / Lightning recurrence, one token at a time in fp32:

        S_t = exp(-slope_h)·S_{t-1} + k_tᵀ v_t;   o_t = (q_t/√d)·S_t

    q/k/v are [B, T, H, D], `slope` [H], the state [B, H, K, V] (K-first,
    the layout FLA, vLLM's lightning cache and sglang's seg_la all keep).
    Shared by the oracle and the torch floor; `final_state`, when handed,
    receives the last step's state — the write-back the serving launch
    performs, which a floor timing the step pays for as well."""
    q, k, v = q.float(), k.float(), v.float()
    b, s, h, d = q.shape
    scale = d ** -0.5
    state = initial_state.float().clone()
    out = q.new_empty(b, s, h, v.shape[-1])
    decay = (-slope.float()).exp()[None, :, None, None]
    q = q * scale
    for t in range(s):
        kt, vt, qt = k[:, t], v[:, t], q[:, t]
        state = state * decay + kt.unsqueeze(-1) * vt.unsqueeze(-2)
        out[:, t] = (qt.unsqueeze(-1) * state).sum(-2)
    if final_state is not None:
        final_state.copy_(state)
    return out


def _gla_reference(inputs: Dict[str, Any]) -> Any:
    q = inputs["q"]
    _refuse_unaffordable_scan(
        "simple_gla", steps=int(q.shape[1]),
        elements=inputs["v"].numel() + inputs["initial_state"].numel())
    return _gla_scan(q, inputs["k"], inputs["v"], inputs["slope"],
                     inputs["initial_state"])


register_ref("gla_simple_recurrence", RefSpec(_gla_inputs, _gla_reference))


def _step3p5_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "attn": torch.randn(*_shape(inv, "attn"), dtype=dt, device=device, generator=gen),
        "gate": torch.randn(*_shape(inv, "gate"), dtype=dt, device=device, generator=gen),
    }


def _step3p5_reference(inputs: Dict[str, Any]) -> Any:
    attn, gate = inputs["attn"].float(), inputs["gate"].float()
    # attn [B,H,S,D], gate [B,S,H] → broadcast as [B,H,S,1]
    return attn * gate.transpose(1, 2).unsqueeze(-1)


register_ref("step3p5_headwise_gate",
             RefSpec(_step3p5_inputs, _step3p5_reference))


def _position_table_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    x = torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen)
    table = torch.randn(*_shape(inv, "table"), dtype=dt, device=device, generator=gen)
    return {"x": x, "table": table}


def _position_table_reference(inputs: Dict[str, Any]) -> Any:
    return inputs["x"].float() + inputs["table"].float()


register_ref("position_table_add",
             RefSpec(_position_table_inputs, _position_table_reference))


def _attn_res_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    return {
        "prefix": torch.randn(*_shape(inv, "prefix"), dtype=dt, device=device, generator=gen),
        "stack": torch.randn(*_shape(inv, "stack"), dtype=dt, device=device, generator=gen),
    }


def _attn_res_reference(inputs: Dict[str, Any]) -> Any:
    import torch
    prefix = inputs["prefix"].float()
    stack = inputs["stack"].float()
    cat = torch.cat([stack, prefix.unsqueeze(2)], dim=2)
    x = cat * torch.rsqrt(cat.pow(2).mean(-1, keepdim=True) + 1e-6)
    scores = (x[:, :, -1:, :] * x).sum(-1)
    w = torch.softmax(scores, dim=-1)
    return (w.unsqueeze(-1) * cat).sum(2)


register_ref("attn_res_combine", RefSpec(_attn_res_inputs, _attn_res_reference))


# ---------------------------------------------------------------------------
# conv2d: full (non-depthwise) k×k strided conv over a flattened square patch
# grid. The spec carries x feature-last and flattened, `[B, H·W, C_in]`;
# reshape to `[B, C_in, H, W]` for F.conv2d, then flatten the output grid back
# to `[B, s_ctx_ds, C_out]`. Step-3.7's ViT downsamplers (3×3 stride-2).
# ---------------------------------------------------------------------------

def _conv2d_grid(s_conv_grid: int) -> int:
    """Side H of the square patch grid — `s_conv_grid` is H² (the identity
    keys only a square grid, and MINIMAL_DIMS declares one), so `isqrt` is
    exact."""
    import math
    s = int(s_conv_grid)
    h = math.isqrt(s)
    if h * h != s:
        raise ValueError(f"conv2d s_conv_grid={s} is not a perfect square")
    return h


def _conv2d_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    dt = _dtype(inv.dtype)
    x = torch.randn(*_shape(inv, "x"), dtype=dt, device=device, generator=gen)      # [B, s_conv_grid, C_in]
    weight = torch.randn(*_shape(inv, "weight"),                      # [C_out,C_in,k,k]
                         dtype=_operand_dtype(inv, "weight"),
                         device=device, generator=gen)
    bias = torch.randn(*_shape(inv, "bias"),                          # [C_out]
                       dtype=_operand_dtype(inv, "bias"),
                       device=device, generator=gen)
    return {"x": x, "weight": weight, "bias": bias,
            "stride": int(_dims(inv)["stride"])}


def _conv2d_reference(inputs: Dict[str, Any]) -> Any:
    import torch.nn.functional as F
    x = inputs["x"].float()                      # [B, s_conv_grid, C_in]
    w = inputs["weight"].float()                 # [C_out, C_in, k, k]
    B, s, C_in = x.shape
    H = _conv2d_grid(s)                           # H·W == s exactly
    k = w.shape[-1]
    grid = x.transpose(1, 2).reshape(B, C_in, H, H)   # [B, C_in, H, W]
    # Padded (k//2) strided conv — for ODD k (the ViT downsamplers are all 3×3)
    # this halves each spatial side (out ceil(H/stride) per dim), matching
    # conv2d's cost model rows/stride². Even k with symmetric k//2 padding
    # gives floor(H/s)+1 instead; no even-k conv2d exists in the corpus.
    out = F.conv2d(grid, w, bias=inputs["bias"].float(),
                   stride=inputs["stride"], padding=k // 2)
    Bo, Co, Ho, Wo = out.shape                   # [B, C_out, Ho, Wo]
    return out.reshape(Bo, Co, Ho * Wo).transpose(1, 2)   # [B, s_conv_grid_ds, C_out]


register_ref("conv2d", RefSpec(_conv2d_inputs, _conv2d_reference))


# ---------------------------------------------------------------------------
# The bare collectives: one contiguous payload over a group
# ---------------------------------------------------------------------------

def _collective_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """One payload buffer, the group that moves it, and the group's spacing.

    That is the whole cell, because that is the whole launch: every kernel
    in this family takes a pointer, a count, a dtype and a communicator.
    `n_comm_ranks` and `n_comm_stride` change no tensor's shape — one rank
    holds the same buffer whatever the group is — so they ride as scalars a
    provider reads, the way `allreduce_fused_rmsnorm` already passes its
    group.

    A ROUTED exchange (`routed_all_to_all`) states three more scalars: the
    top-k picks, the destinations they are drawn from and the ones that hold
    weights. They change no shape either, and they are what makes the
    launch the ragged one its cell names — how many distinct destinations a
    token goes to, and so what the per-peer split is. A provider that
    builds the split lists for a DeepEP dispatch reads them; without them it
    could only build the flat equal split, which is the OTHER cell. Passed
    for whichever of them the signature's cell declares, so the flat
    exchange's inputs do not change.

    No oracle. On one process every one of these collectives is the
    identity, so an oracle here would agree with a provider that never
    communicated — which is exactly how a fused all-reduce once
    cross-checked green without a fabric. A provider that brings
    `n_comm_ranks` processes brings its own comparison.
    """
    import torch
    dims = _dims(inv)
    out = {
        "payload": torch.randn(*_shape(inv, "payload"),
                               dtype=_dtype(inv.dtype), device=device, generator=gen),
        "n_comm_ranks": int(dims["n_comm_ranks"]),
        "n_comm_stride": int(dims["n_comm_stride"]),
    }
    # Only what this signature's cell declares: the flat exchange's inputs
    # must not so much as LOOK at a routing axis, or every collective would
    # read dims its cell does not have
    # (`test_cell_dims_are_what_the_reference_reads`).
    from backend.inference.op_signatures import get_signature
    cell = getattr(get_signature(inv.signature), "cell_dims", ())
    for axis in ("n_experts_active", "n_route_targets", "n_route_experts"):
        if axis in cell:
            out[axis] = int(dims[axis])
    return out


# Which signatures those are is the catalog's own declaration — the
# signatures whose cell IS a collective launch (`shape_projection`), asked
# rather than restated as a list of collective names here.
for _sig in all_signatures():
    if _sig.shape_projection == "collective":
        register_ref(_sig.name, RefSpec(_collective_inputs, None))
del _sig


# ---------------------------------------------------------------------------
# The hyper-connection seam (`mhc_post`, `mhc_pre_gemm`, `mhc_pre_big_fuse`,
# `mhc_fused_post_pre`). Every reference here is vLLM's own eager reference
# for these kernels (`model_executor/kernels/mhc/torch.py`, which sglang
# mirrors in `_mhc_post_torch` / `_mhc_pre_torch`), recomputed in fp32.
#
# Six scalars ride on the inputs beside the tensors. None of them moves a
# byte, and every one changes what the kernel computes, so they are stated
# once here and read by the oracle AND by every provider — the treatment the
# MoE routers' `renormalize` and routed scale already get. Their values are
# what all three corpus models of this span run (`config.hc_eps` 1e-6,
# `config.hc_sinkhorn_iters` 20, `rms_norm_eps` 1e-6, and the post multiplier
# 2.0 that both frameworks hard-code — vLLM `self.hc_post_alpha`, sglang's
# `_MHC_POST_MULT_VALUE`).
#
# Two earned input-building details:
#   * `post`, `comb`, `scale`, `base` and the mix projection `fn` are fp32
#     whatever the cell's activation dtype is. Both frameworks hold them in
#     fp32 and assert it (`assert fn.dtype == torch.float32`).
#   * the big fuse's `sqrsum` is the residual's ACTUAL sum of squares, not an
#     independent draw: the launch divides by it to normalise the mix vector,
#     and an unrelated value would put the sigmoid gates outside the range
#     any model runs them in.
# ---------------------------------------------------------------------------

#: What the corpus models of this span run. See the module comment above.
_HC_RMS_EPS = 1e-6
_HC_EPS = 1e-6
_HC_POST_MULT = 2.0
_HC_SINKHORN_ITERS = 20
_HC_NORM_EPS = 1e-6


def _hc_scalars() -> Dict[str, Any]:
    """The six function constants every mHC seam kernel takes."""
    return {
        "rms_eps": _HC_RMS_EPS,
        "hc_pre_eps": _HC_EPS,
        "hc_sinkhorn_eps": _HC_EPS,
        "hc_post_mult_value": _HC_POST_MULT,
        "sinkhorn_repeat": _HC_SINKHORN_ITERS,
        "norm_eps": _HC_NORM_EPS,
    }


def _hc_post_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The post-mix's cell: the sublayer output, the multi-copy residual and
    the two coefficient sets the opening pre-mix published. `comb` is drawn
    row-stochastic, as the Sinkhorn normalisation that produced it leaves it;
    `post` in [0, 2], the range its `2·sigmoid` gate has."""
    import torch
    dt = _dtype(inv.dtype)
    comb = torch.rand(*_shape(inv, "comb"), dtype=torch.float32, device=device,
                      generator=gen)
    out = {
        "post": _HC_POST_MULT * torch.rand(
            *_shape(inv, "post"), dtype=torch.float32, device=device,
            generator=gen),
        "sublayer": torch.randn(*_shape(inv, "sublayer"), dtype=dt,
                                device=device, generator=gen),
        "comb": comb / comb.sum(-1, keepdim=True),
        "residual": torch.randn(*_shape(inv, "residual"), dtype=dt,
                                device=device, generator=gen),
    }
    out.update(_hc_scalars())
    return out


def hc_post_mix(inputs: Dict[str, Any]) -> Any:
    """vLLM `mhc_post_torch`: the finished sublayer output spread back over
    the residual copies by `post`, plus the copies cross-mixed by `comb`.

        y[j,h] = Σᵢ comb[i,j]·residual[i,h] + post[j]·sublayer[h]

    Note which index `comb` contracts: the kernels sum over its FIRST copy
    index (`einsum("...ij,...ih->...jh")`), while the `hc_mix` op's formula
    contracts the second. Same bytes and the same FLOPs either way — the
    oracle computes the kernels'.
    """
    import torch
    comb, residual = inputs["comb"].float(), inputs["residual"].float()
    mixed = torch.einsum("tij,tih->tjh", comb, residual)
    post_term = (inputs["post"].float().unsqueeze(-1)
                 * inputs["sublayer"].float().unsqueeze(-2))
    return mixed + post_term


register_ref("mhc_post", RefSpec(_hc_post_inputs, hc_post_mix))


def _hc_fn(inv: Invocation, device: str, gen: Any) -> Any:
    """The mix projection, fp32 and scaled by 1/√K so the dots it writes are
    O(1) — the range the gates downstream of them are fitted for."""
    import torch
    shape = _shape(inv, "fn")
    return torch.randn(*shape, dtype=torch.float32, device=device,
                       generator=gen) / float(shape[-1]) ** 0.5


def _hc_pre_gemm_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    out = {
        "residual": torch.randn(*_shape(inv, "residual"),
                                dtype=_dtype(inv.dtype), device=device,
                                generator=gen),
        "fn": _hc_fn(inv, device, gen),
    }
    out.update(_hc_scalars())
    return out


def hc_normalised_mixes(residual_flat: Any, fn: Any, rms_eps: float) -> Any:
    """The pre-mix projection's span output: the flattened residual against
    `fn`, scaled by the residual's own root-mean-square.

    `mixes = fn @ rms_unscaled(flatten(residual))`, the plan's chain. The
    kernels write the UNNORMALISED dots plus a per-token sum of squares and
    the following big fuse applies the rsqrt; scaling a row before or after
    the projection is the same function, and this is the form the span
    declares its output in.
    """
    import torch
    x = residual_flat.float()
    dots = x @ fn.float().t()
    scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    return dots * scale


def _hc_pre_gemm_reference(inputs: Dict[str, Any]) -> Any:
    residual = inputs["residual"]
    flat = residual.reshape(residual.shape[0], -1)
    return hc_normalised_mixes(flat, inputs["fn"], float(inputs["rms_eps"]))


register_ref("mhc_pre_gemm",
             RefSpec(_hc_pre_gemm_inputs, _hc_pre_gemm_reference))


def _hc_big_fuse_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The epilogue's cell: the projection's dots and their sum-of-squares
    partner, the scale and base of the mix split, the residual it reduces and
    the sublayer norm's gain. `sqrsum` is the residual's own sum of squares
    (see the module comment); `scale` ones and `base` zeros, the identity
    parameterisation of the split, so the gates read the dots as they are."""
    import torch
    dt = _dtype(inv.dtype)
    residual = torch.randn(*_shape(inv, "residual"), dtype=dt, device=device,
                           generator=gen)
    out = {
        "mixes": torch.randn(*_shape(inv, "mixes"), dtype=torch.float32,
                             device=device, generator=gen),
        "sqrsum": residual.reshape(residual.shape[0], -1).float()
                          .square().sum(-1),
        "scale": torch.ones(*_shape(inv, "scale"), dtype=torch.float32,
                            device=device),
        "base": torch.zeros(*_shape(inv, "base"), dtype=torch.float32,
                            device=device),
        "residual": residual,
        "norm_weight": torch.randn(*_shape(inv, "norm_weight"), dtype=dt,
                                   device=device, generator=gen),
    }
    out.update(_hc_scalars())
    return out


def hc_split_coefficients(m: Any, scale: Any, base: Any, n_copies: int,
                          *, pre_eps: float, post_mult: float,
                          sinkhorn_eps: float, sinkhorn_iters: int):
    """A NORMALISED mix vector split into the `pre` gate, the `post` gate and
    the Sinkhorn-normalised cross-mix — the frameworks' own function
    (vLLM `mhc_pre_torch`, which sglang mirrors in `_mhc_pre_torch`).

    THE ONE PLACE this arithmetic is written. Two spans reach it: the
    hyper-connection epilogue, which applies its rsqrt first and then splits
    (`hc_mix_split`), and the bare `hc_split_sinkhorn` op, whose operand is
    the already-normalised vector. Writing it twice is what let the repo hold
    two readings of one function, which is the defect this consolidates.

    `comb` is doubly stochastic by construction: a softmax over the last
    copy index, then `sinkhorn_iters` rounds that ALTERNATE the row and the
    column normalisation. The alternation is what Sinkhorn means and what
    makes the result doubly stochastic — normalising one side repeatedly is
    idempotent after the first pass and converges to nothing.
    """
    import torch
    hc, eps = n_copies, sinkhorn_eps
    pre = torch.sigmoid(m[:, :hc] * scale[0] + base[:hc]) + pre_eps
    post = (torch.sigmoid(m[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
            * post_mult)
    comb = (m[:, 2 * hc:].view(-1, hc, hc) * scale[2]
            + base[2 * hc:].view(1, hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(int(sinkhorn_iters) - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_mix_split(mixes: Any, sqrsum: Any, scale: Any, base: Any,
                 n_copies: int, reduce_width: int, inputs: Dict[str, Any]):
    """vLLM `mhc_pre_torch`: one token's mix vector split into the `pre` gate,
    the `post` gate and the Sinkhorn-normalised cross-mix.

    `mixes` and `sqrsum` are the projection's writes (split-K partials
    already reduced); the rsqrt that the pre-GEMM span states as its own
    output is applied here, because this is the launch that applies it. The
    split itself is `hc_split_coefficients`.
    """
    import torch
    m = mixes.float() * torch.rsqrt(
        sqrsum.float().unsqueeze(-1) / reduce_width
        + float(inputs["rms_eps"]))
    pre, post, comb = hc_split_coefficients(
        m, scale, base, n_copies,
        pre_eps=float(inputs["hc_pre_eps"]),
        post_mult=float(inputs["hc_post_mult_value"]),
        sinkhorn_eps=float(inputs["hc_sinkhorn_eps"]),
        sinkhorn_iters=int(inputs["sinkhorn_repeat"]))
    return pre, post, comb


def _hc_big_fuse_reference(inputs: Dict[str, Any]) -> Any:
    """The launch's three writes side by side: the normalised sublayer input,
    the `post` gate and the cross-mix. The driver compares one tensor, so the
    three are concatenated along the feature axis — all of them checked, and
    all three are O(1) so one adaptive tolerance fits them."""
    import torch
    residual = inputs["residual"]
    hc, d = int(residual.shape[1]), int(residual.shape[2])
    pre, post, comb = hc_mix_split(
        inputs["mixes"], inputs["sqrsum"], inputs["scale"], inputs["base"],
        hc, hc * d, inputs)
    reduced = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    normed = reduced * torch.rsqrt(
        reduced.square().mean(-1, keepdim=True) + float(inputs["norm_eps"]))
    layer_input = normed * inputs["norm_weight"].float()
    return torch.cat([layer_input, post, comb.flatten(1)], dim=-1)


register_ref("mhc_pre_big_fuse",
             RefSpec(_hc_big_fuse_inputs, _hc_big_fuse_reference))


def _hc_fused_post_pre_inputs(inv: Invocation, device: str, gen: Any
                              ) -> Dict[str, Any]:
    out = _hc_post_inputs(inv, device, gen)
    out["fn"] = _hc_fn(inv, device, gen)
    return out


def _hc_fused_post_pre_reference(inputs: Dict[str, Any]) -> Any:
    """The post-mix, then the next pre-mix projection over its result — the
    span's two ports side by side (`output` and `residual_out`).

    The projection reads the post-mix's output AS THE GRAPH CARRIES IT, in
    the activation dtype, which is what the plan's naive pattern says. The
    fused kernel keeps that value in registers instead, so its mix vector
    differs from this one by a single rounding of the residual — the round
    trip the fusion removes, and well inside the activation dtype's band.
    """
    import torch
    y = hc_post_mix(inputs).to(inputs["residual"].dtype)
    mixes = hc_normalised_mixes(y.reshape(y.shape[0], -1), inputs["fn"],
                                float(inputs["rms_eps"]))
    return torch.cat([mixes, y.flatten(1).float()], dim=-1)


register_ref("mhc_fused_post_pre",
             RefSpec(_hc_fused_post_pre_inputs, _hc_fused_post_pre_reference))


def _hc_mega_post_pre_inputs(inv: Invocation, device: str, gen: Any
                             ) -> Dict[str, Any]:
    """The Mega-MHC cell: the post operands, the projection, the CARRIED
    previous `pre`, and the split/norm metadata the launch consumes.

    `pre` is drawn in the range the sigmoid split leaves it (one to two),
    and `scale` / `base` use the identity parameterisation so the launch's
    new coefficient ports are numerically visible without amplifying the
    mix vector.
    """
    import torch
    out = _hc_post_inputs(inv, device, gen)
    out.update({
        "fn": _hc_fn(inv, device, gen),
        "pre": 1.0 + torch.rand(
            *_shape(inv, "pre"), dtype=torch.float32, device=device,
            generator=gen),
        "scale": torch.ones(*_shape(inv, "scale"), dtype=torch.float32,
                            device=device),
        "base": torch.zeros(*_shape(inv, "base"), dtype=torch.float32,
                            device=device),
        "norm_weight": torch.randn(
            *_shape(inv, "norm_weight"), dtype=_dtype(inv.dtype),
            device=device, generator=gen),
    })
    return out


def _hc_mega_post_pre_reference(inputs: Dict[str, Any]) -> Any:
    """The launch's three canonical writes side by side: normed sublayer
    input, raw current mix vector, and the new multi-copy residual.

    The carried `pre` is the gate the PREVIOUS seam produced. The current
    mix split is also computed by the launch; the graph leaves that split a
    billed node, so this oracle checks the projection port that node reads
    rather than re-stating all three coefficients beside it.
    """
    import torch
    residual = hc_post_mix(inputs).to(inputs["residual"].dtype)
    mixes = hc_normalised_mixes(
        residual.reshape(residual.shape[0], -1), inputs["fn"],
        float(inputs["rms_eps"]))
    reduced = (inputs["pre"].float().unsqueeze(-1)
               * residual.float()).sum(dim=1)
    normed = reduced * torch.rsqrt(
        reduced.square().mean(-1, keepdim=True)
        + float(inputs["norm_eps"]))
    layer_input = normed * inputs["norm_weight"].float()
    return torch.cat([layer_input, mixes, residual.flatten(1).float()],
                     dim=-1)


register_ref("mhc_mega_post_pre",
             RefSpec(_hc_mega_post_pre_inputs,
                     _hc_mega_post_pre_reference))


# quantise_activation: the cast a quantised matmul runs on its activation
# ---------------------------------------------------------------------------

def _quantise_activation_inputs(inv: Invocation, device: str,
                                gen: Any) -> Dict[str, Any]:
    """One activation, at the WIDE end of the cast whichever way it runs.

    For a quantise cell that is the operand the launch reads. For a
    dequantise cell it is not: the launch reads the narrow elements and
    their scales, and the provider quantises this tensor in its own build to
    get them (untimed, as a weight's repack is untimed). Carrying the wide
    tensor in both directions is what makes the two cells share one oracle —
    a dequantise kernel is correct exactly when it returns `x` as the narrow
    format holds it, which is the same thing a quantise kernel's output
    dequantises to.
    """
    import torch
    from .quant import ActivationCast
    cast = ActivationCast.of(inv)
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=_store_dtype(cast.wide),
                         device=device, generator=gen),
        # The cast itself, so a provider and the oracle read ONE reading of
        # the cell rather than each parsing the dtype environment again.
        "cast": cast,
    }


def _quantise_activation_reference(inputs: Dict[str, Any]) -> Any:
    """`x` as the narrow format holds it, at the cell's own granularity and
    scale encoding — quantise then dequantise, which is what the pair of
    directions have in common.

    This is the one place the error should be ~0 rather than within a band:
    a cast kernel and this oracle run the SAME algorithm, so a disagreement
    is a disagreement about the granularity or the scale convention, and
    that is a bug in one of them rather than precision loss to be tolerated.
    """
    return inputs["cast"].on_grid(inputs["x"])


register_ref("quantise_activation",
             RefSpec(_quantise_activation_inputs,
                     _quantise_activation_reference))


# ---------------------------------------------------------------------------
# The rest of the hyper-connection family: the spread with no cross-mix
# (`ihc_post`), the exit reduce (`fused_hc_head`), and the three launches of
# a flattened-stream model's read/write gates (`hc_silu`, `hc_gate_mix`,
# `hc_combine_norm`). Each reference is the kernels' own function recomputed
# in fp32 — the frameworks' eager references where they ship one
# (`hc_head_torch`, sglang `srt/models/deepseek_v4.py:614`), the kernel body
# otherwise.
#
# NONE of them reaches `hc_split_coefficients`, and that is a property of the
# family rather than an omission: the Sinkhorn split exists to produce a
# doubly-stochastic cross-mix, and no launch here has a cross-mix. `ihc_post`
# scatters under one scalar per copy, `hc_gate_mix` averages under a sigmoid
# per element, `fused_hc_head` reduces under a sigmoid per copy. Routing them
# through the seam's split would be a second reading of a function they do
# not compute.
#
# Function constants, stated once and read by the oracle AND every provider,
# as the seam's six are (`_hc_scalars`) and for the same reason — they move
# no byte and they change what the kernel computes, so a provider picking its
# own value would cross-check a function no model runs:
#
#   * the epsilons are DeepSeek-V4-Pro's `rms_norm_eps` / `hc_eps` and
#     Qwen3.8-Flash-Next's `rms_norm_eps`, all 1e-6, which is what
#     `_HC_RMS_EPS` / `_HC_EPS` above already state for this same family.
#   * `hc_silu`'s divisor is the one constant the CELL cannot supply. Its
#     operand is the low-rank mix vector, whose width has nothing to do with
#     the copy count, so `n_hidden_copies` is no axis of that contract and
#     declaring it structurally would key the cell on a number the walk never
#     publishes for this row — every cell would drop. It is therefore stated
#     at the value its one corpus model runs (Qwen3.8-Flash-Next
#     `hc_count = 4`), the treatment `_KDA_LOWER_BOUND` and
#     `_RMSNORM_GATED_ACTIVATION` already get. Every other launch here reads
#     the count off the cell, because there it is a tensor axis.
# ---------------------------------------------------------------------------

#: Qwen3.8-Flash-Next's `hc_count` — the divisor compiled into `hc_silu`.
#: See the module comment above for why this one is stated and the others
#: are read off the cell.
_HC_SILU_COPIES = 4


def _ihc_post_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The spread's cell: the per-copy gate, the finished sublayer output and
    the residual it lands on.

    `post` is fp32 because both kernels assert it is (`assert post.dtype ==
    torch.float32`), and drawn in [0, 2] — the range `hc_magnitude ·
    sigmoid(·) + hc_eps` has at HY V4's magnitude of 2.0.
    """
    import torch
    dt = _dtype(inv.dtype)
    return {
        "post": 2.0 * torch.rand(*_shape(inv, "post"), dtype=torch.float32,
                                 device=device, generator=gen),
        "sublayer": torch.randn(*_shape(inv, "sublayer"), dtype=dt,
                                device=device, generator=gen),
        "residual": torch.randn(*_shape(inv, "residual"), dtype=dt,
                                device=device, generator=gen),
    }


def ihc_post_spread(inputs: Dict[str, Any]) -> Any:
    """`out[c] = post[c]·sublayer + residual[c]` — the kernels' loop over the
    copies (vLLM `_ihc_post_kernel` :183-187, sglang `_hy4_ihc_post_kernel`
    :193-196), as one broadcast.

    No cross-mix: this is the independent hyper-connection, where each copy
    sees only its own gate. The copies' axis is -2, so the gate expands on
    the width and the sublayer output on the copies.
    """
    post = inputs["post"].float().unsqueeze(-1)
    sublayer = inputs["sublayer"].float().unsqueeze(-2)
    return post * sublayer + inputs["residual"].float()


register_ref("ihc_post", RefSpec(_ihc_post_inputs, ihc_post_spread))


def _hc_head_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    """The exit reduce's cell: the multi-copy residual and the gate's affine.

    `fn` is fp32 and scaled by 1/√K, as the seam's projection is and for the
    same reason — the dots it writes feed a sigmoid, which is only
    informative where they are O(1). `scale` ones and `base` zeros is the
    identity parameterisation, so the gate reads the dots as they are.
    """
    import torch
    return {
        "x": torch.randn(*_shape(inv, "x"), dtype=_dtype(inv.dtype),
                         device=device, generator=gen),
        "fn": _hc_fn(inv, device, gen),
        "scale": torch.ones(*_shape(inv, "scale"), dtype=torch.float32,
                            device=device),
        "base": torch.zeros(*_shape(inv, "base"), dtype=torch.float32,
                            device=device),
    }


def hc_head_reduce(inputs: Dict[str, Any]) -> Any:
    """The frameworks' own eager reference for this launch, in fp32:

        x_flat = flatten(x);  rsqrt = 1/√(mean(x_flat²) + rms_eps)
        mixes  = (fn @ x_flat)·rsqrt
        pre    = sigmoid(mixes·scale + base) + hc_eps
        y      = Σ_c pre_c · x_c

    (sglang `hc_head_torch`, deepseek_v4.py:614-628; vLLM computes the same
    chain inside `hc_head_fuse_tilelang`, tilelang_kernels.py:1085-1090.)

    The `+ hc_eps` after the sigmoid is in BOTH kernels and is not
    decoration: it floors the gate away from zero, and leaving it out would
    make the oracle a different function by one epsilon per copy.
    """
    import torch
    x = inputs["x"]
    flat = x.reshape(x.shape[0], -1).float()
    mixes = hc_normalised_mixes(flat, inputs["fn"], _HC_RMS_EPS)
    pre = torch.sigmoid(mixes * inputs["scale"].float()
                        + inputs["base"].float()) + _HC_EPS
    return (pre.unsqueeze(-1) * x.float()).sum(dim=1)


register_ref("fused_hc_head", RefSpec(_hc_head_inputs, hc_head_reduce))


def _hc_silu_inputs(inv: Invocation, device: str, gen: Any) -> Dict[str, Any]:
    import torch
    return {"x": torch.randn(*_shape(inv, "x"), dtype=_dtype(inv.dtype),
                             device=device, generator=gen)}


def hc_silu_activation(inputs: Dict[str, Any]) -> Any:
    """`y = silu(x / n_hidden_copies)`, the kernel's two lines (vLLM
    `_hc_silu_kernel` :100-101: the load is divided by `HC`, then
    `x·sigmoid(x)`). The divisor is inside the activation, not after it —
    silu is not homogeneous, so dividing afterwards is another function."""
    import torch
    x = inputs["x"].float() / _HC_SILU_COPIES
    return x * torch.sigmoid(x)


register_ref("hc_silu", RefSpec(_hc_silu_inputs, hc_silu_activation))


def _hc_gate_mix_inputs(inv: Invocation, device: str, gen: Any
                        ) -> Dict[str, Any]:
    """The read's cell: the gate logits and the normed stream, both the full
    flattened multi-copy row. Both are activation-dtype — the gate arrives
    from the up projection, and the kernel applies the sigmoid itself.

    `n_hidden_copies` rides along because this launch is the one place in
    the family where NO operand carries it: both tensors are the row with
    the copies already flattened into it, and the count reappears only on
    the output. It is the cell's own axis (`hc_gate_mix` declares it), so it
    is READ here rather than stated, and the oracle and every provider take
    it from this one place.
    """
    import torch
    dt = _dtype(inv.dtype)
    return {
        "gate": torch.randn(*_shape(inv, "gate"), dtype=dt, device=device,
                            generator=gen),
        "xn": torch.randn(*_shape(inv, "xn"), dtype=dt, device=device,
                          generator=gen),
        "n_hidden_copies": int(_dims(inv)["n_hidden_copies"]),
    }


def hc_gate_mix_read(inputs: Dict[str, Any]) -> Any:
    """`y = (1/hc)·Σ_c sigmoid(gate_c)·xn_c` — vLLM `_hc_gate_mix_kernel`
    :152-156, which accumulates `sigmoid(g)·x` over the copies and then
    divides by `HC`.

    The accumulation is over the COPY axis of the flattened row, i.e. the
    stream is `[copies, width]`-major (`HC` outer, `HS` inner —
    hyperconnection.py:11-12), so the view splits the LEADING copy index.
    Splitting the trailing one would read a different, interleaved stream and
    still produce a plausible `[tokens, d_model]` answer.
    """
    import torch
    gate, xn = inputs["gate"], inputs["xn"]
    hc = int(inputs["n_hidden_copies"])
    g = gate.float().view(gate.shape[0], hc, -1)
    x = xn.float().view(xn.shape[0], hc, -1)
    return (torch.sigmoid(g) * x).sum(dim=1) / hc


register_ref("hc_gate_mix", RefSpec(_hc_gate_mix_inputs, hc_gate_mix_read))


def _hc_combine_norm_inputs(inv: Invocation, device: str, gen: Any
                            ) -> Dict[str, Any]:
    """The write's cell: the stream it adds onto, the sublayer output, the
    per-copy injection logits and the grouped norm's gain.

    The gain is `d_model`-grouped over a `copies·d_model` tensor — the
    per-branch form this model's checkpoint declares
    (`qwen_hc_full_block.yaml`: `hc_norm.weight` at
    `n_hidden_copies · d_model`) — which is also the layout the kernel's
    `W_SHARED=False` branch reads.
    """
    import torch
    dt = _dtype(inv.dtype)
    return {
        "residual": torch.randn(*_shape(inv, "residual"), dtype=dt,
                                device=device, generator=gen),
        "block": torch.randn(*_shape(inv, "block"), dtype=dt, device=device,
                             generator=gen),
        "injection": torch.randn(*_shape(inv, "injection"), dtype=dt,
                                 device=device, generator=gen),
        "norm_weight": torch.randn(*_shape(inv, "norm_weight"), dtype=dt,
                                   device=device, generator=gen),
    }


def hc_combine_norm_write(inputs: Dict[str, Any]) -> Any:
    """The launch's two writes side by side (the driver compares one tensor):

        out = residual + bcast(block)·2·sigmoid(injection/hc)
        y   = rms_per_copy(out)·(1 + w)

    vLLM `_hc_combine_norm_kernel` :322-345. Three details the kernel fixes
    and a looser reading would get wrong:

      * the injection gate is `2·sigmoid(logit/hc)` — divided BEFORE the
        sigmoid (:322), where dividing after would be another function;
      * `out` is ROUNDED to the activation dtype before the norm reads it
        (:325, "matches the unfused combine -> RMSNorm boundary"), because
        the unfused graph materialises it there;
      * the affine is Gemma's `(1 + w)` (:344, `y += y·w`), which is what
        this model's graph declares too (`formula: "y = norm · (1 + w)"`).

    The norm is per COPY, over that copy's own `d_model`, not over the whole
    flattened row.
    """
    import torch
    residual, block = inputs["residual"], inputs["block"]
    hc = int(inputs["injection"].shape[-1])
    d = int(block.shape[-1])
    gate = 2.0 * torch.sigmoid(inputs["injection"].float() / hc)
    write = block.float().unsqueeze(-2) * gate.unsqueeze(-1)
    out = (residual.float().view(residual.shape[0], hc, d) + write)
    # The boundary the unfused graph writes through: rounded, then normed.
    out = out.to(residual.dtype).float()
    normed = out * torch.rsqrt(out.square().mean(-1, keepdim=True)
                               + _HC_RMS_EPS)
    flat_out = out.reshape(out.shape[0], hc * d)
    y = normed.reshape(out.shape[0], hc * d) * (
        1.0 + inputs["norm_weight"].float())
    return torch.cat([flat_out, y], dim=-1)


register_ref("hc_combine_norm",
             RefSpec(_hc_combine_norm_inputs, hc_combine_norm_write))
