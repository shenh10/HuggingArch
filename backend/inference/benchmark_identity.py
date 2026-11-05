"""Shared helpers that produce the exact identity strings used by both
`huggingarch bench suggest-shapes` and the production `/inference/estimate`
calibration lookup.

The CSV schema treats `spec_hash`, `shape_json`, and `dtype_env_json` as
identity fields with byte-equality match semantics. If two code paths
build those strings differently, calibration lookup will silently miss
every real measurement. This module is the single source of truth so
the template CSV row a contributor authors via `suggest-shapes` is the
same row identity the route looks up at request time.
"""
from __future__ import annotations

from .op_signatures import (
    PAGE_AXIS, OpSignature, get_signature, op_bindings, op_signature,
    output_width,
    spec_get, zip_engine_layouts,
)
from backend.arch.ops import OpSpec
from backend.arch.activation_formats import ACTIVATION_FORMATS
from backend.arch.dtypes import (
    DTYPE_BYTES, UnknownDtype, accepted_dtype_names, corpus_token,
    has_corpus_token,
)
from backend.arch.cache_families import slot_kinds
from backend.arch.cache_formats import CACHE_FORMATS
from backend.arch.dims import call_token_axes
from backend.arch.eval import cache as cache_mod
from backend.arch.eval.dim import algebra, value_of
from backend.arch.eval.refusal import node_context
from backend.arch.eval.routed import RoutedLaunch
from backend.arch.eval.storage import is_expert_axis
from backend.arch.eval.formats import known_dtype_bytes
from backend.arch.spec_migrate import measured_row_migration

import hashlib

import sympy as sp
import math
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property, lru_cache
from typing import (Any, Dict, FrozenSet, List, Optional, Tuple,
                    TYPE_CHECKING)

if TYPE_CHECKING:
    from backend.arch.loader import ResolvedSpec



def compute_spec_hash(resolved: ResolvedSpec) -> str:
    """The resolution's own identity hash (`ResolvedSpec.identity_hash`).

    Named here because the CSV identity field is called `spec_hash` and
    every reader of that column comes through this module; the hash itself
    is the spec's answer, computed where its dims table and config are
    declared, so a row cannot be keyed one way by the planner and another
    way by the lookup.
    """
    return resolved.identity_hash()


class InvalidIdentityJSON(ValueError):
    """A CSV/plan identity JSON field that cannot name a cell.

    In particular, a missing field is not the JSON object ``{}``.  The empty
    environment is a meaningful dtype identity, so treating absence as that
    value can silently re-key an old row onto an unrelated launch.
    """


def canonical_identity_json(value: Any) -> str:
    """One spelling for either row identity JSON field.

    ``shape_json`` and ``dtype_env_json`` are join fields.  Historical CSV or
    plan text may differ only in whitespace or insertion order; once parsed,
    both spellings must map to the same compact, key-sorted text.  A missing,
    null, non-object, or unparsable value is an error rather than ``{}``.
    """
    if value is None:
        raise InvalidIdentityJSON("identity JSON is missing/null")
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError) as exc:
        raise InvalidIdentityJSON(f"invalid identity JSON {value!r}") from exc
    if not isinstance(parsed, dict):
        raise InvalidIdentityJSON(
            f"identity JSON must be an object, got {type(parsed).__name__}")
    try:
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise InvalidIdentityJSON(
            f"identity JSON is not serializable: {exc}") from exc


def physical_cell_of(op: str, shape: Any, dtype_env: Any,
                     phase: str) -> Tuple[str, str, str, str]:
    """The library-agnostic physical cell from row-shaped identity fields.

    This is the canonicalizer shared by a measured CSV row and a legacy plan
    cache.  A current plan does not need this reconstruction: it carries the
    four fields as ``job['physical_key']``, and consumers read that carried
    key instead of deriving a second one.  Both paths meet at this same
    ``(op/signature, shape_json, dtype_env_json, phase)`` spelling.
    """
    return (op, canonical_identity_json(shape),
            canonical_identity_json(dtype_env), phase)


def compute_per_row_shape_json(
    *,
    batch_size: int,
    seq_len: int,
    dim_values: Optional[Dict[str, int]] = None,
    op: str = "",
    spec_id: str = "",
) -> str:
    """Per-leaf `shape_json` for the CSV identity tuple.

    For now, every leaf gets `{B, s_ctx}` plus the resolved-spec dim values
    that show up in the most-frequently-bound symbols (`d_model`, `n_heads`,
    `n_kv_heads`, `d_head`, `n_experts_active`, etc.). Op-specific overrides
    can be added by extending this function later (e.g. attention emits
    `n_heads`/`d_head` but not `d_ffn_expert`);
    keeping the surface uniform first ensures contributors can write
    rows that match what the route generates.

    Returns a compact JSON string (no whitespace) matching what the
    benchmark validator's round-trip rule expects. Dim values are sorted by key
    for byte-stability.
    """
    out: Dict[str, int] = {"B": int(batch_size), "s_ctx": int(seq_len)}
    if dim_values:
        # Forward the dims most commonly used in attention / FFN sizing
        # so the shape_json reflects the actual op shape, not just B/s_ctx.
        # All glossary symbols are forwarded — the validator already
        # restricts to the registry's own symbols, so any
        # extras here either pass validation or get caught at validate
        # time. Sorting keeps the JSON byte-stable.
        for k in sorted(dim_values.keys()):
            v = dim_values[k]
            # Only forward positive ints — the validator rejects 0/neg/float.
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                out[k] = v
    return json.dumps(out, separators=(",", ":"), sort_keys=True)


def compute_per_row_dtype_env_json(
    *,
    weight_dtype: Optional[str],
    activation_dtype: str,
    kv_dtype: Optional[str],
    metadata_dtype: Optional[str],
    output_dtype: str = "bf16",
    weight_granularity: Optional[str] = None,
    activation_granularity: Optional[str] = None,
    output_format: Optional[str] = None,
    kv_format: Optional[str] = None,
) -> str:
    """Per-leaf `dtype_env_json` for the CSV identity tuple.

    Returns a compact JSON string. Every dtype slot but `output` is the
    caller's to state: a default here keyed an FP8 row as a BF16 one
    whenever a caller forgot a slot. Every row's `output` is the walk's
    (`LeafAccess.output_dtype`, `RowIdentity.leaf_dtype_env`), and a row
    that writes a quantised activation (`LeafAccess.output_activation_format`,
    the grafted cast) adds that format's granularity and scale dtype.
    `output` keeps a BF16 default only for a caller that states none.

    A slot that does not apply to the row is None and is then OMITTED
    (BENCHMARKS_CSV.md rule 3), never filled with another tensor's dtype:

      * `weight` — the dtype of the weight the leaf streams; None when it
        streams none (attention, an activation, a collective).
      * `output` — the dtype the leaf writes its primary output at. A leaf
        that writes a cached tensor there writes it in the cache's format,
        so that format's dtype is this slot.
      * `kv` — the dtype of a cache tensor the leaf itself reads, or writes
        on other than its primary output (a launch that inserts K beside
        the query it outputs); None when it touches no other cache.
      * `metadata` — the dtype of that weight's own scales; None when it
        stores none.

    Each is required, so a caller says None on purpose.

    `weight_granularity` (`TensorFormat.weight_granularity`) is present
    exactly when the weight stores scales: a 128x128-block FP8 GEMM and a
    per-channel one are different kernels on the same dtypes. So it never
    comes without `metadata`, and a weight's scales never come without a
    weight.

    `activation_granularity` (`ActivationFormat.granularity`) is the same
    fact on the other operand, for the row whose scales are an
    ACTIVATION's: the cast a quantised matmul runs on its input
    (`quantise_activation`) streams no weight at all, and its `metadata` is
    the dtype of the only thing it writes beside the elements. Per-token,
    per-128-group and per-32-group are three different kernels on one pair
    of dtypes, so it is part of the identity for the same reason the
    weight's is.

    `output_format` (`LeafAccess.output_format`) is the registry name of
    the cache format a leaf writes its primary output in, present exactly
    when that output is a cached tensor stored with scales, a transform or
    an unquantised tail. The `output` dtype is its element dtype and cannot
    tell DeepSeek-V4's FlashMLA record from an FP8 index key with one F32
    scale, or an MXFP4 index key from an NVFP4 one: different launches
    writing different records over one dtype. It must name a registry
    format whose element dtype is `output` (`format_refusal`).

    `kv_format` is the same fact for the cache tensor the `kv` slot names
    (`RowIdentity._cache_tensor`, `CachedTensor.format_name`): the record a
    launch reads, or writes beside its output. vLLM's DeepSeek-V4 q-norm /
    rope / KV-insert launch writes the FlashMLA record next to the query it
    outputs, and FlashMLA's sparse decode reads that record; a per-tensor
    FP8 cache is `kv=fp8_e4m3` too, and another kernel on both sides.
    """
    dtypes = {"output": output_dtype, "kv": kv_dtype}
    for slot, name in (("output_format", output_format),
                       ("kv_format", kv_format)):
        refusal = format_refusal(slot, name, dtypes[FORMAT_SLOTS[slot]])
        if refusal:
            raise ValueError(refusal)
    for granularity, what in ((weight_granularity, "weight"),
                              (activation_granularity, "activation")):
        if granularity is not None and metadata_dtype is None:
            raise ValueError(
                f"the {what} stores scales ({granularity}) but states no "
                "scale dtype; the `metadata` slot has nothing to read")
    if (metadata_dtype is not None and weight_dtype is None
            and activation_granularity is None):
        raise ValueError(
            f"scales ({metadata_dtype}) with no weight and no quantised "
            f"activation they scale")
    env = {slot: value for slot, value in (
        ("activation", activation_dtype), ("output", output_dtype),
        ("weight", weight_dtype), ("kv", kv_dtype),
        ("metadata", metadata_dtype),
        ("weight_granularity", weight_granularity),
        ("activation_granularity", activation_granularity),
        ("output_format", output_format),
        ("kv_format", kv_format)) if value is not None}
    return json.dumps(env, separators=(",", ":"), sort_keys=True)


#: The slots that name how a cached tensor is stored, each with the dtype
#: slot that holds that tensor's element dtype: the primary output a launch
#: writes as a cached record, and the cache it reads or writes beside it.
#: The vocabulary's one definition — the identity producer, the CSV
#: validator (`benchmarks._check_dtype_env_json`) and kernel_bench's
#: declarations, smoke cells and cross-check band read it.
FORMAT_SLOTS: Dict[str, str] = {"output_format": "output", "kv_format": "kv"}

#: What a kernel's declaration of a format slot (`KernelSignature.
#: output_format` / `.kv_format`) names to admit a PLAIN tensor: no scale,
#: no transform, no tail — the cell that states no format in that slot.
#: It is the value the slot reads on such a cell (`dtype_env.get(slot)`),
#: so the gate needs no rule of its own, and no cell can carry it: a cell
#: omits the slot when the tensor is plain, and a stated slot must be a
#: registry name (`format_refusal`; the CSV validator refuses a non-string).
#: Declared beside names it reads "plain or these records" — FA3 reads a
#: bf16 cache or an fp8 one with one per-tensor descale.
PLAIN = None


def format_refusal(slot: str, name: Optional[str],
                   dtype: Optional[str]) -> str:
    """Why format slot `slot` (`FORMAT_SLOTS`) cannot say `name` beside its
    dtype slot's token `dtype`, or "".

    The slot names a registry cache format (`arch/cache_formats/`), and
    that format's element dtype is its dtype slot's token: the two say one
    tensor, so they cannot disagree. The one rule both the identity
    producer (`compute_per_row_dtype_env_json`) and the CSV validator
    (`benchmarks._check_dtype_env_json`) apply."""
    if name is None:
        return ""
    fmt = CACHE_FORMATS.get(name)
    if fmt is None:
        return (f"`{slot}` {name!r} is not a registry cache "
                f"format; named formats are {sorted(CACHE_FORMATS)} "
                f"(backend/arch/cache_formats/)")
    token = corpus_token(fmt.dtype)
    if dtype != token:
        return (f"`{slot}` {name!r} stores {token} elements, "
                f"and the `{FORMAT_SLOTS[slot]}` slot says {dtype!r}")
    return ""



def normalize_dtype_name(name: Optional[str]) -> str:
    """Map UI/API dtype aliases to validator-vocabulary strings.

    The benchmark CSV validator's CORPUS_TOKENS is strict — `fp8` is
    ambiguous (e4m3 vs e5m2) so only the specific variants are
    accepted. The API/UI exposes the shorter `fp8` alias (matching
    vllm's default); without normalization here, the calibration
    overlay's `dtype_env_json` carries the raw alias and gets rejected
    AND fp8-KV lookups can't match any authored CSV row.

    Maps the ambiguous `fp8` → `fp8_e4m3` (vllm/H100 default). An empty or
    unknown name is a ValueError: the callers state their own defaults in
    their signatures, and a name outside the vocabulary must not travel on
    as a token the corpus will never match.
    """
    if not name:
        raise ValueError("dtype name is empty — the caller must state one")
    try:
        return corpus_token(name)
    except UnknownDtype:
        raise ValueError(
            f"unknown dtype name {name!r}; accepted: "
            f"{sorted(accepted_dtype_names())}") from None


def is_known_dtype_name(name: Optional[str]) -> bool:
    """Whether `name` spells a dtype the benchmark corpus has a token for —
    a UI alias or the token itself. Used at API request boundaries to reject
    unknown values with a clear vocabulary error rather than letting them
    fall through to a silent default."""
    return bool(name) and has_corpus_token(name)


# DeepSeek sparse attention (DSA / NSA), the main attention that consumes the
# lightning indexer's selection. It is NOT dense attention: the engine emits
# `shape_in = [Q=[B,n_heads,s_query,d_qk], topk_mask=[B,s_query,s_kv],
# topk_weights=[B,s_query,s_kv]]` —
# the operands after Q are the SELECTION over the candidate cache, not K/V, so
# the KV length lives on the mask's LAST axis (`s_kv`) rather than a head-shaped
# tensor's seq axis. Reading it as dense attention found no 4-D K and produced
# no identity at all, which is how these rows ended up on the whole-step
# fallback.
#
# The kernel attends only the `n_topk_keys` selected positions per query, so the
# identity must carry the selected count BESIDE the candidate length: the work
# scales with topk while the gather's index range (and its cache behaviour)
# scales with `s_kv`. Production dispatch is a dedicated sparse kernel family
# (`flash_mla_sparse_fwd`, `trtllm_batch_decode_with_kv_cache_mla`), never the
# dense GQA path.
# Dense matmul ops: `shape_in` is `[B, s_query, d_reduce]`. The output width
# `d_out` is in `shape_out` when the engine populated it; for the
# FUSED column-parallel linears (qkv / gate_up) the fused node has no single
# `shape_out` (it emits separate q/k/v or gate/up tensors), but the engine
# DID propagate the per-port widths in `shape_outs` — the GEMM's concatenated
# output width is their sum. So `d_out` comes from real propagated shapes, never a
# flops back-calculation.


def _is_gemm_kernel(op: str) -> bool:
    """Whether this row's op is measured by the GEMM kernel signature.
    The kernel side asks its own declaration (the catalog's `binds_ops`);
    the ops side's `OpSpec.gemm` says what the math is, and a per-head
    linear is a GEMM in math but `per_head_gemm` in measurement."""
    return op_signature(op) == "gemm"


def declares_routed_token_total(signature: str) -> bool:
    """Whether this signature's cell carries a routed token TOTAL — the count
    the expert bank's kernels take beside their GEMM dims, which is what the
    planner's MoE token distribution keys on.

    Two declarations already state this, and both are asked rather than
    restated: `dims.yaml` classifies which symbol is a call-token total
    (`call_tokens: total`), and the catalog grows each signature's cell from
    its declared contract (`OpSignature.cell_dims` — `grouped_gemm` declares
    the total as a `structural_axes` entry because it is no tensor's axis).

    It used to read `signature in ("grouped_gemm", "fused_moe")`, and that
    name list had already drifted from the catalog it restated: `fused_moe`'s
    contract declares no token total, so the second member was dead at the
    one call site — and nothing went red, because a list in a function body
    is a table no test and no loader checks.
    """
    sig = get_signature(signature)
    if sig is None:
        return False
    totals = call_token_axes().get("total", ())
    return any(axis in sig.cell_dims for axis in totals)


def _ints(extents: Any) -> Optional[List[int]]:
    """A tensor shape's extents as ints, or None when one is not a number
    (a symbolic extent): such a shape names no physical key."""
    try:
        return [int(x) for x in extents]
    except (TypeError, ValueError):
        return None


def _first_tensor(shape_in: Any) -> Optional[list]:
    """Return the first tensor shape (a list of ints) from a `shape_in`
    that may be a bare `[..]` or a list-of-tensors `[[..], ..]`."""
    if not isinstance(shape_in, (list, tuple)) or not shape_in:
        return None
    head = shape_in[0]
    if isinstance(head, (list, tuple)):   # list-of-tensors
        return list(head) if head and not isinstance(head[0], (list, tuple)) else None
    return list(shape_in)                 # bare tensor


def _fold_per_head_rmsnorm(shape_in: Any) -> Optional[str]:
    """4-D QK-norm `[B, n_heads, s_query, d_head]` → per-token rmsnorm identity.

    One RMS over the last axis for each of B·n_heads·s_query rows — the same
    kernel as `[B, s_query, d_model]`, so it shares that physical key.
    """
    t = _first_tensor(shape_in)
    if t is None or len(t) != 4:
        return None
    try:
        dims = {"B": int(t[0]), "s_query": int(t[1]) * int(t[2]),
                "d_model": int(t[3])}
    except (TypeError, ValueError):
        return None
    if any(v <= 0 for v in dims.values()):
        return None
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _fold_batched_gemm(shape_in: Any, shape_out: Any) -> Optional[str]:
    """4-D batched GEMM `[B, s_query, n_copies, d_reduce]` → dense-gemm identity.

    A linear whose activation still carries a parallel axis — DSv4's MTP
    entry combiner runs `h_proj` on the pre-reduce HC stream
    `[B, s_ctx, n_hidden_copies, d_model]` — multiplies ONE shared weight
    by B·s_query·n_copies rows, the same kernel as the flattened
    `[rows, d_reduce]`. The extra axis folds into the row count, the same
    move `_fold_per_head_rmsnorm` makes for heads. (A per-head weight is
    NOT this: that is `per_head_gemm`, with its own rank-4 layout.)
    """
    t = _first_tensor(shape_in)
    if t is None or len(t) != 4:
        return None
    try:
        dims = {"B": int(t[0]), "s_query": int(t[1]) * int(t[2]),
                "d_reduce": int(t[3]), "d_out": int(output_width(shape_out))}
    except (TypeError, ValueError):
        return None
    if any(v <= 0 for v in dims.values()):
        return None
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _operands(shape_in: Any) -> Optional[list]:
    if not isinstance(shape_in, (list, tuple)) or not shape_in:
        return None
    if isinstance(shape_in[0], (list, tuple)):
        return list(shape_in)
    return [shape_in]


def _zip_engine(engines: Any, shape_in: Any, shape_out: Any,
                output_axes: Tuple[str, ...],
                spec: Optional[Dict[str, Any]] = None,
                ) -> Optional[Dict[str, int]]:
    """Zip one engine-layout declaration onto `shape_in`.

    Packing and permute live on the signature (`TensorLayout` / alternate
    entries at the same operand index). This is only the caller.
    """
    if not engines:
        return None
    operands = _operands(shape_in)
    if operands is None:
        return None
    return zip_engine_layouts(
        engines, operands, spec=spec or {},
        shape_out=shape_out, output_axes=output_axes)


def _zip_sides(sig: OpSignature, engines: Any, shape_in: Any, shape_out: Any,
               spec: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """One engine-layout declaration bound onto the shapes that arrive.

    A kernel's geometry sits on the side the engine carries it. For most
    signatures that is the operands; for the MoE dispatch it is the OUTPUT —
    `permute` reads token-ordered rows and WRITES the `[n_expert_copies, M,
    d_model]` buffer whose geometry the launch is, so the signature declares
    an `engine_outputs` layout instead. Both sides are the same declaration
    zipped the same way, and an axis both name must agree.
    """
    dims: Optional[Dict[str, int]] = None
    for layouts, operands, axes in ((engines, shape_in, sig.output_axes),
                                    (sig.engine_outputs, shape_out, ())):
        if not layouts:
            continue
        bound = _zip_engine(layouts, operands, shape_out, axes, spec=spec)
        if bound is None:
            return None
        if dims is None:
            dims = bound
            continue
        for axis, value in bound.items():
            if dims.setdefault(axis, value) != value:
                return None
    return dims


def _contract_identity(op: str, shape_in: Any, shape_out: Any,
                       spec_dims: Optional[Dict[str, Any]] = None,
                       call_tokens: Optional[int] = None,
                       ) -> Optional[str]:
    """Identity JSON from the signature's declared engine layout.

    The signature names each operand's axes in the engine's order (a
    permutation of its contract, validated at registration); this zips the
    engine's shapes with those names. A variant geometry (MLA prefill's
    per-head K/V) is a second engine set, tried first because it is the
    more specific operand list.

    An axis `dims.yaml` declares a call-token TOTAL (`call_tokens: total` —
    a grouped expert GEMM's `n_tokens`) is no tensor's axis: it is the rows
    one launch presents, which the caller states (`call_tokens`).

    Other axes in `cell_dims` that no engine operand carries (a conv kernel
    width) are filled from `spec_dims`. Missing after that is incomplete
    — None, not a guessed default.

    Variant axes that are not a tensor's axis (`s_window` on GQA) are
    filled from `spec_dims` when present. The caller decides presence:
    passing the model-wide window onto every attention leaf would
    make a hybrid's full-attention layers share the SWA cell.
    """
    sig = get_signature(op_signature(op))
    if sig is None:
        return None
    spec = spec_dims or {}
    layout_axes = sig.layout_axis_names()
    totals = frozenset(call_token_axes().get("total", ()))
    for engines in (sig.variant_engine_inputs, sig.engine_inputs):
        dims = _zip_sides(sig, engines, shape_in, shape_out, spec)
        if dims is None:
            continue
        # The axes of the geometry these operands are (`cell_dims_for`): the
        # variant engine set zips only a variant cell.
        cell_dims = sig.cell_dims_for(dims)
        for axis in cell_dims:
            if axis in dims:
                continue
            if axis in totals:
                if call_tokens is None:
                    break
                dims[axis] = int(call_tokens)
                continue
            raw = spec_get(spec, axis)
            if raw is None:
                break
            try:
                dims[axis] = int(raw)
            except (TypeError, ValueError):
                dims = None
                break
        if dims is None:
            continue
        if any(axis not in dims or dims[axis] <= 0 for axis in cell_dims):
            continue
        # V width when it is not d_head, from this leaf's out (apply's last
        # dim). `spec` holds no variant axis but the leaf's own
        # (`row_invocation_dims` drops the node dims' value for each), so it
        # cannot hand a leaf a width bound model-wide (MiMo binds d_v_head
        # model-wide). Read off the output only: an operand after Q/K is one
        # the signature does not lay out (a V beside its K, a rank-1 sink),
        # and the key reads declared operands alone (`stored_operands`). No
        # row of the corpus lacked the output width when this read stopped.
        v_w = output_width(shape_out)
        d_head = dims.get("d_head")
        # Only a signature that keys a V width has one: a partial rope's
        # output is its q and k ports, whose summed width is no V.
        if (v_w and d_head and v_w != d_head
                and "d_v_head" in sig.identity_dims):
            dims["d_v_head"] = v_w
        for axis in sig.variant_dims:
            if axis in dims or axis in layout_axes:
                continue
            raw = spec.get(axis)
            if raw is None:
                continue
            try:
                val = int(raw)
            except (TypeError, ValueError):
                continue
            if val > 0:
                dims[axis] = val
        return json.dumps(dims, separators=(",", ":"), sort_keys=True)
    return None


@dataclass(frozen=True)
class CommLaunch:
    """What eval published about one collective launch, as a cell reads it.

    A collective has no local math, and its tensor's layout is not its
    kernel's parameter list: `ncclAllReduce` takes a buffer, a count, a
    dtype and a communicator. These fields are the row's own statement of
    the rest (`comm_payload_shape`, `comm_world`, `comm_base_stride`,
    `comm_payload_dtype_bytes`, written by `phases.py`), carried together
    because they arrive together and say nothing apart.
    """
    payload_shape: Tuple[int, ...]
    world: int
    stride: int
    #: Bytes per payload element, as the walk priced the launch. The cell
    #: counts ELEMENTS, so this is what says how wide one is — and it is a
    #: width, not a dtype name: 1 byte is fp8 or int8 and the row does not
    #: say which, which is why this cannot be turned into a dtype
    #: environment and a disagreement is refused instead (`from_row`).
    payload_dtype_bytes: float

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Optional["CommLaunch"]:
        """The launch a row describes, or None for a row that is not one.

        `is_comm` is the row's own flag — the walk's projection of
        `OpSpec.is_comm_op` — so which ops are collectives stays one
        declaration and is not re-tested by name here. A group of one is
        no collective (nothing grafts one, and the fused collective's cell
        refuses it on the same grounds), so it has no launch either.

        A payload whose elements are not the width this row computes at has
        no launch either. The cell is an element count, and the harness
        builds and bills the buffer at the row's activation dtype; a
        collective that moves something else — DCP's log-sum-exp
        all-gather, an fp32 statistic published beside bf16 activations
        (`materialize_rows._LSE_BYTES`) — would be measured at half its
        width and billed at full. The row publishes a WIDTH and not a dtype
        name, and a width cannot name a dtype (1 byte is fp8 or int8), so
        there is no env to route it into: the row drops and names itself.

        A row whose shape the walk could not state as a law in the batch
        has no launch either, and that is the interesting refusal. Every row
        says whether its shape columns are this batch's (`shape_at_batch`:
        the walk stamps True, a re-read stamps what it could read), and the
        collective columns are re-read from the same law beside them
        (`batch_sweep._display_at` off `point_law["comm"]`: payload,
        collective in and out). So a re-read row's payload is its own
        batch's, and the one row to refuse is the one that states no shape
        at this batch: keying off a stale payload would hand two batches one
        cell and bill the smaller one a fraction of the bytes — the
        stale-named-field failure the starved-axis gate hit from the other
        side.
        """
        if not row.get("is_comm") or row.get("shape_at_batch") is False:
            return None
        shape = _ints(row.get("comm_payload_shape"))
        # The group reads through the same "ints or nothing" helper as the
        # payload: one place owns what a non-numeric extent means.
        group = _ints([row.get("comm_world"), row.get("comm_base_stride")])
        if not shape or not group or any(extent <= 0 for extent in shape):
            return None
        world, stride = group
        if world <= 1 or stride <= 0:
            return None
        # The payload's element width, against the width this row computes
        # at. `DTYPE_BYTES` is the declared width column for a caller that
        # already holds a name; an act_dtype outside it (an index or mask
        # type) is no kernel dtype and so no cell either.
        width = row.get("comm_payload_dtype_bytes")
        act = DTYPE_BYTES.get(str(row.get("act_dtype") or "").strip().upper())
        if not width or act is None or float(width) != act:
            return None
        return cls(tuple(shape), world, stride, float(width))

    @property
    def elements(self) -> int:
        """The payload as one buffer — NCCL's `count`. See the catalog's
        collective entries for what this flattening claims."""
        return math.prod(self.payload_shape)


def _collective_identity(comm: Optional[CommLaunch],
                         sig: Optional[OpSignature]) -> Optional[str]:
    """`{n_comm_elems, n_comm_ranks, n_comm_stride}` — the buffer one launch
    moves, the group it moves it over, and that group's spacing on the mesh.

    One axis per published fact, and only the axes the signature's cell
    declares: a signature asking for an axis no comm fact answers gets no
    cell, so the row is dropped and named instead of keyed on a guess.
    """
    if comm is None or sig is None:
        return None
    facts = {"n_comm_elems": comm.elements,
             "n_comm_ranks": comm.world,
             "n_comm_stride": comm.stride}
    dims: Dict[str, int] = {}
    for axis in sig.cell_dims:
        value = facts.get(axis)
        if value is None or value <= 0:
            return None
        dims[axis] = int(value)
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _ceil(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _exact(extent: Any) -> Optional[Fraction]:
    """One extent as the walk's law states it, or None when it states no
    number (a symbolic extent)."""
    if type(extent) is int:
        return Fraction(extent)
    try:
        return Fraction(str(extent))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _whole(shape: Any, exact: Any) -> Any:
    """`shape` as the whole geometry ONE launch runs on.

    The walk publishes two forms of every shape: integer extents, floored,
    and the law's own exact extents beside them. They differ wherever an
    axis is a MEAN — the rows one expert of a bank receives
    (`[n_expert_copies, M, d]`), the window a compressor pools once every
    few tokens — and no kernel runs a fraction of a row, so a cell's extents
    are the law's, rounded UP. An axis that averages below one row is the
    one row the launch still covers; an axis that is genuinely zero stays
    zero, and the row is refused rather than keyed at a geometry it never
    runs.
    """
    operands = _operands(shape)
    exacts = _operands(exact)
    if operands is None:
        return shape
    out = []
    for index, tensor in enumerate(operands):
        law = exacts[index] if exacts is not None and index < len(exacts) else None
        if (not isinstance(tensor, (list, tuple)) or not tensor
                or (law is not None and len(law) != len(tensor))):
            out.append(tensor)
            continue
        row = []
        for axis, extent in enumerate(tensor):
            value = _exact(law[axis] if law is not None else extent)
            row.append(extent if value is None else _ceil(value))
        out.append(row)
    return out if isinstance(shape[0], (list, tuple)) else out[0]


@dataclass(frozen=True)
class MatrixShard:
    """One published weight partition (`weight_shards`, the walk's
    `WeightShard`): the rows `n` and reduction width `k` of the grid its
    scale tiles, unsharded and on one rank, which weight it is packed into
    and its own scale layout. Axis geometry and the optional checkpoint
    block-layout contract are the walk's facts: this consumer does not infer
    which axes are rows, copies, or load-time scale boundaries."""

    pack: Tuple[int, int]
    n: int
    n_local: int
    k: int
    k_local: int
    granularity: Optional[str]
    block_scale_layout: Optional[Dict[str, Any]] = None

    @property
    def cuts_n(self) -> bool:
        return self.n_local != self.n

    @property
    def cuts_k(self) -> bool:
        return self.k_local != self.k

    @classmethod
    def of(cls, shard: Dict[str, Any]) -> "MatrixShard":
        return cls(tuple(shard["pack"]), int(shard["rows"]),
                   int(shard["rows_local"]), int(shard["width"]),
                   int(shard["width_local"]), shard["granularity"],
                   shard.get("block_scale_layout"))


def routed_launch(row: Dict[str, Any]) -> Optional[RoutedLaunch]:
    """The MoE dispatch boundary the row's routed buffer is on, as the walk
    published it (`phases._routed_columns`: `RoutedLaunch.wire`), None for
    a row on no routed buffer."""
    wire = row.get("routed")
    return None if not wire else RoutedLaunch.from_wire(wire)


def _rows(tensor: Any, law: Any) -> Optional[Fraction]:
    """A tensor's rows — its axes but its width — at the law's exact
    extents, None where an axis has no positive value."""
    if not isinstance(tensor, (list, tuple)) or len(tensor) < 2:
        return None
    if not isinstance(law, (list, tuple)) or len(law) != len(tensor):
        law = tensor
    rows = Fraction(1)
    for axis in range(len(tensor) - 1):
        value = _exact(law[axis])
        if value is None or value <= 0:
            return None
        rows *= value
    return rows


def _launch_rows(row: Dict[str, Any]) -> Tuple[Any, Fraction]:
    """`(launched, exact)` rows ONE launch of this row's kernel runs.

    A row on a routed buffer launches the buffer's padded rows
    (`RoutedLaunch.launch_rows`: the experts that receive rows, each at its
    rows rounded up) against the routed rows its law states
    (`RoutedLaunch.routed_rows`) — the routed token total a boundary or
    expert cell keys (`dims.yaml` `call_tokens: total`, `n_tokens`).

    Every other row pads nothing, so its two counts are one: the widest of
    its tensors' rows at the law's exact extents (a compressor's pool that
    runs once every few tokens keeps its law's rows — `RowIdentity.cadence`
    bills it and the plan refuses to build it, `plan._rounded_axes`).
    `launched` is an `int` wherever it is whole, which is every row the plan
    builds.
    """
    launch = routed_launch(row)
    if launch is not None:
        return launch.launch_rows, Fraction(launch.routed_rows)
    sides: List[Fraction] = []
    for shape, exact in ((row.get("shape_in"), row.get("shape_in_exact")),
                         (row.get("shape_out") or row.get("shape_outs"),
                          row.get("shape_out_exact"))):
        for tensor, law in zip(_operands(shape) or [], _operands(exact) or [],
                               strict=False):
            rows = _rows(tensor, law)
            if rows is not None:
                sides.append(rows)
    if not sides:
        return 1, Fraction(1)
    widest = max(sides)
    return (int(widest) if widest.denominator == 1 else widest), widest


def routed_launch_rows(row: Dict[str, Any]) -> Optional[int]:
    """The rows of the routed buffer one launch runs
    (`RoutedLaunch.launch_rows`), or None for a row on no routed buffer —
    the count a cell's call-token total (`n_tokens`) states."""
    launch = routed_launch(row)
    return None if launch is None else launch.launch_rows


def _launched(shape: Any, launch: RoutedLaunch, routed: Any) -> Any:
    """`shape` (already whole, `_whole`) with each routed buffer among its
    operands at the geometry the launch runs: `[n_expert_copies, M, ...]`
    — the layout the dispatch writes and every tensor of its zone keeps,
    experts outermost — becomes the experts that receive rows at their
    padded rows (`RoutedLaunch`). `routed` says which operands are the
    buffer: a set of positions, or True for every one."""
    operands = _operands(shape)
    if operands is None:
        return shape
    out = []
    for index, tensor in enumerate(operands):
        if routed is True or index in routed:
            if (not isinstance(tensor, (list, tuple)) or len(tensor) < 3
                    or int(tensor[0]) != launch.local_experts):
                raise ValueError(
                    f"routed tensor {list(tensor)!r} does not lead with the "
                    f"{launch.local_experts} experts the rank holds")
            tensor = [launch.touched_experts, launch.rows_per_expert,
                      *tensor[2:]]
        out.append(tensor)
    return out if isinstance(shape[0], (list, tuple)) else out[0]


@lru_cache(maxsize=None)
def _names_an_expert_axis(expr: str) -> bool:
    """Whether an axis whose identity is `expr` (`as_wire`'s `expr`) is an
    expert axis (`storage.is_expert_axis`)."""
    return is_expert_axis(algebra(expr))


def _banked(shape: Any, wire: Any, launch: RoutedLaunch,
            skip: Any = frozenset()) -> Any:
    """`shape` with every expert axis of its operands — found by the axis's
    identity in `wire` (`shape_*_dims`), as `layout.physical` finds it — at
    the experts the launch touches (`RoutedLaunch.touched_experts`): an
    expert bank's weight or per-expert parameter (a fused bank's bias,
    `[n_expert_copies, d]`) read beside the routed buffer is the same bank,
    so the launch holds the touched experts' share of it. `skip` names the
    operands already laid out as the routed buffer (`_launched`)."""
    operands, wires = _operands(shape), _operands(wire)
    if operands is None or wires is None:
        return shape
    out = []
    for index, tensor in enumerate(operands):
        axes = wires[index] if index < len(wires) else None
        if (index not in skip and isinstance(tensor, (list, tuple))
                and isinstance(axes, (list, tuple))
                and len(axes) == len(tensor)):
            tensor = [launch.touched_experts
                      if isinstance(axis, dict)
                      and _names_an_expert_axis(str(axis.get("expr")))
                      else extent
                      for extent, axis in zip(tensor, axes, strict=True)]
        out.append(tensor)
    return out if isinstance(shape[0], (list, tuple)) else out[0]


def identity_shape_from_shape_in(
    op: str,
    shape_in: Any,
    *,
    shape_out: Any = None,
    shape_out_exact: Any = None,
    spec_dims: Optional[Dict[str, Any]] = None,
    shape_in_exact: Any = None,
    comm: Optional[CommLaunch] = None,
    routed_rows: Optional[int] = None,
) -> Optional[str]:
    """Derive the calibration identity shape from the engine's per-op
    `shape_in` — the single source of truth that already reflects the phase
    binding (decode collapses the seq axis to s_query=1). Returns a compact
    validator-legal JSON string for a recognized op family, or `None` when
    the op/layout isn't handled (caller falls back to
    `compute_per_row_shape_json`).

    `shape_out` (GEMM only) is the engine's output shape — a single tensor
    OR a list of per-port tensors for a fused multi-output node; see
    `output_width`.

    Emitted in the MEASUREMENT vocabulary, never the engine's own symbols —
    two models that land on the same kernel shape are one physical cell and
    must share one row, which they cannot do while each names the axis after
    its own config. Families:
      * dense attention → `{B, n_heads, n_kv_heads, s_query, s_kv, d_head}`
        (Q=[B,n_heads,s_query,d_head],
        K=[B,n_kv_heads,s_kv,d_head]); `s_query` vs `s_kv` makes prefill and
        decode natively distinct, with no separate `phase` field needed.
        A sliding-window leaf also carries `s_window` (variant, not a
        tensor axis) so it does not share the full-attention cell; a
        sink softmax adds `n_attn_sinks`; a narrower V adds `d_v_head`.
      * norm (per-token) → `{B, s_query, d_model}` from
        `[B, s_query, d_model]`.
      * act / gated-mul → `{B, s_query, d_ffn}` from `[B, s_query, d_ffn]`.
      * dense matmul → `{B, s_query, d_reduce, d_out}`
        (`d_out` = `output_width(shape_out)`).

    A routed row — the MoE dispatch (`permute`), the expert GEMMs
    (`grouped_gemm`) and the inverse (`combine`) — is the contract path too:
    the engine lays every tensor between the dispatch and the combine out
    `[n_expert_copies, M, d]`, so each of those signatures declares that
    layout on the side it carries it (`engine_inputs` / `engine_outputs`)
    and names the experts and their mean rows as `n_experts` /
    `s_expert_rows`. The two boundaries also name the TOKEN side they move
    between (`B` × `s_query`): the dispatch's gather reads it as an operand,
    the combine's reduce writes it as its output, and both row counts are
    the launch's geometry.

    `routed_rows` is the rows of the routed buffer one launch runs
    (`routed_launch_rows`), which no tensor carries as an axis: it is the
    cell's routed token total (`n_tokens`, `dims.yaml` `call_tokens:
    total`) on all three routed signatures. The caller states it because
    which operand is the buffer is the walk's (`LeafAccess.routed_inputs`,
    on the row as `routed`), not something the bare extents here can say —
    a dispatch's first operand is its TOKEN side.
    """
    sig = get_signature(op_signature(op))
    projection = sig.shape_projection if sig is not None else "contract"
    if projection == "collective":
        # A collective's cell comes from what it MOVES, which its operand
        # shape only sometimes is: an all-gather's `shape_in` is one rank's
        # shard while the launch's payload is the gathered buffer. The walk
        # already states the payload it priced, so read that.
        return _collective_identity(comm, sig)
    # One launch's whole geometry, for every projection below: the walk's
    # floored extents are means wherever the law divides, and no kernel runs
    # a fraction of a row.
    shape_in = _whole(shape_in, shape_in_exact)
    shape_out = _whole(shape_out, shape_out_exact)
    # Engine layout is not the kernel layout — count reconstruction, a
    # selection mask, a router whose first operand is d_model not n_experts.
    # Packed/permuted engine tensors are declared on the signature
    # (`engine_inputs`) and inverted by `zip_engine_layouts`, not here.
    if projection == "dsa":
        return _cell_from_spec(sig, _dsa_identity(sig, shape_in, shape_out, spec_dims),
                               spec_dims)
    if projection == "rope":
        return _rope_identity(shape_in, spec_dims)
    if projection == "inv_rope":
        return _inv_rope_identity(shape_in, spec_dims)
    if projection == "moe_gate":
        return _moe_gate_identity(shape_in, shape_out)
    if projection == "conv2d":
        return _conv2d_identity(shape_in, shape_out, spec_dims)
    got = _contract_identity(
        op, shape_in, shape_out, spec_dims=spec_dims,
        call_tokens=routed_rows)
    if got is not None:
        return got
    # Per-head QK-norm is the same rmsnorm kernel over B·n_heads·s_query
    # rows: fold the head axis into the token count so it shares the
    # per-token identity.
    if projection == "fold_rmsnorm":
        return _fold_per_head_rmsnorm(shape_in)
    # A shared-weight linear on a rank-4 stream (DSv4 MTP's `h_proj` over
    # the pre-reduce HC copies) is the dense GEMM over the flattened rows.
    if projection == "fold_gemm":
        return _fold_batched_gemm(shape_in, shape_out)
    if projection == "reduce_axis":
        return _reduce_axis_identity(shape_in, shape_out)
    if projection == "trailing_feature":
        return _trailing_feature_identity(shape_in)
    return None


def _reduce_axis_identity(shape_in: Any, shape_out: Any) -> Optional[str]:
    """`x[*lead, d_reduce, *inner]` → `y[*lead, *inner]` as
    `{B, s_query, d_reduce, d_model}`.

    The reduced axis is the one the output lacks — read off the engine's own
    two shapes, not re-derived from the node's `axis:`. `B` is the leading
    axis, `s_query` the rest of the lead folded, `d_model` the inner extent
    (1 when the last axis reduces). A zero-length axis is an empty reduction,
    not a geometry: no identity, so the row is dropped and named.
    """
    t, o = _first_tensor(shape_in), _first_tensor(shape_out)
    if t is None or o is None or len(o) != len(t) - 1:
        return None
    t, o = _ints(t), _ints(o)
    if t is None or o is None:
        return None
    k = next((i for i in range(len(o)) if t[i] != o[i]), len(o))
    if t[:k] + t[k + 1:] != o or any(x <= 0 for x in t):
        return None
    lead, inner = t[:k], t[k + 1:]
    s_query = 1
    for x in lead[1:]:
        s_query *= x
    d_model = 1
    for x in inner:
        d_model *= x
    dims = {"B": lead[0] if lead else 1, "s_query": s_query,
            "d_reduce": t[k], "d_model": d_model}
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _trailing_feature_identity(shape_in: Any) -> Optional[str]:
    """Last axis is the feature; everything in front is tokens.

    Pointwise ops (silu, mul, add, …) run the same kernel on 3-D `[B,S,D]`
    and 4-D `[B,H,S,D]`; folding heads into `s_query` is the same
    move as per-head RMSNorm. A rank-1 vector is one row of that
    feature. Binary mul/add uses the wider operand.
    """
    ops = _operands(shape_in)
    if not ops:
        t = _first_tensor(shape_in)
        ops = [t] if t is not None else []
    ranked = []
    for t in ops:
        if not isinstance(t, (list, tuple)) or not t:
            continue
        ints = _ints(t)
        if ints is not None:
            ranked.append(ints)
    if not ranked:
        return None
    vals = max(ranked, key=len)
    d_model = vals[-1]
    if d_model <= 0:
        return None
    if len(vals) == 1:
        dims = {"B": 1, "s_query": 1, "d_model": d_model}
    elif len(vals) == 2:
        tokens = max(vals[0], 1)
        dims = {"B": 1, "s_query": tokens, "d_model": d_model}
    else:
        b = max(vals[0], 1)
        mid = 1
        for x in vals[1:-1]:
            mid *= max(int(x), 1)
        dims = {"B": b, "s_query": mid, "d_model": d_model}
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def row_invocation_dims(dims: Optional[Dict[str, Any]] = None,
                        *,
                        signature: Optional[OpSignature],
                        extra: Optional[Dict[str, Any]] = None,
                        s_window: Optional[int] = None,
                        comm_world: Optional[Any] = None) -> Dict[str, Any]:
    """Spec constants for identity projection, with the kernel's variants
    per leaf.

    `dims` is the ROW's own Invocation dims (`node_dims` carried on the
    per-op row), not a model-wide table. Which of them choose WHICH kernel
    runs rather than size it is `signature`'s own declaration
    (`OpSignature.variant_dims`), and those are a fact of the leaf: the
    dims' value for them is dropped and the leaf's own stands — `extra`
    (what `gqa_leaf_variants` read off this span's absorbed ops: a window,
    a sinks flag filled later from `n_heads`) or the leaf's own shapes (a
    narrower V's width, `_cell_from_spec`). A hybrid whose odd layers are
    full attention binds a window model-wide, and its full-attention leaf
    must not share the windowed cell.

    It used to drop a list written here, `{s_window, n_attn_sinks}`, which
    restated the declaration and had already missed its third axis,
    `d_v_head`.
    """
    per_leaf = frozenset(signature.variant_dims if signature is not None else ())
    spec = {k: v for k, v in (dims or {}).items() if k not in per_leaf}
    extra = dict(extra or {})
    if s_window is not None and "s_window" not in extra:
        extra["s_window"] = s_window
    # A fused collective's group, where the row has one. 1 is no group (a
    # local kernel, or a collective of one rank): the signature that asks
    # for it declines such a cell rather than keying it as a collective.
    if comm_world is not None and int(comm_world or 1) > 1:
        extra["n_comm_ranks"] = int(comm_world)
    for key, raw in extra.items():
        if key == "sinks":
            continue
        try:
            val = int(raw)
        except (TypeError, ValueError):
            continue
        if val > 0:
            spec[key] = val
    return spec


def gqa_leaf_variants(analysis: Any) -> Dict[str, Dict[str, Any]]:
    """fused `flash_attention` aggregation_id → per-leaf kernel variants.

    `s_window` only for spans that absorbed an op declared `windowed`;
    `sinks` (True) for spans that absorbed an op declared `sink`. Read off
    the origins' declarations, not a component-name list, an op-name list,
    or the model-wide dim table.
    """
    pre = getattr(analysis, "pre_program", None)
    eval_analysis = getattr(analysis, "analysis", None)
    matches = getattr(analysis, "fusion_matches", None) or ()
    if pre is None or eval_analysis is None:
        return {}
    dims_layer = getattr(eval_analysis, "node_dims", None)
    values = getattr(dims_layer, "values", None) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for match in matches:
        if match.get("fused_op") != "flash_attention":
            continue
        sid = match.get("fused_aggregation_id")
        if not sid:
            continue
        leaf: Dict[str, Any] = {}
        for uid in match.get("matched_uids") or ():
            node = pre.by_uid(uid)
            if node is None:
                continue
            decl = node.declaration
            windowed = isinstance(decl, OpSpec) and decl.windowed
            sink = isinstance(decl, OpSpec) and decl.sink
            nd = values.get(uid) or {}
            if windowed and "s_window" not in leaf:
                raw = nd.get("s_window")
                if raw is not None:
                    try:
                        w = int(value_of(raw))
                    except (TypeError, ValueError):
                        w = 0
                    if w > 0:
                        leaf["s_window"] = w
            if sink:
                leaf["sinks"] = True
        if leaf:
            out[str(sid)] = leaf
    return out


def attach_gqa_sinks(shape_json: Optional[str], has_sinks: bool) -> Optional[str]:
    """Fold per-head sink presence into a GQA identity.

    Width is this leaf's `n_heads` (already per-rank). Presence is the
    variant: a span that did not absorb a `*_softmax_with_sink` omits it.
    """
    if not has_sinks or not shape_json:
        return shape_json
    try:
        dims = json.loads(shape_json)
    except json.JSONDecodeError:
        return shape_json
    heads = dims.get("n_heads")
    try:
        n = int(heads)
    except (TypeError, ValueError):
        return shape_json
    if n <= 0:
        return shape_json
    dims["n_attn_sinks"] = n
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _averaged_fraction(row: Dict[str, Any]) -> Fraction:
    """The share of one launch a call of this row is — 1 for almost every
    row (`RowIdentity.cadence`).

    Read off the ROWS a call presents: each operand's own axes but its
    width, multiplied together at the law's exact extents. A row whose work
    happens once every m calls presents `1/m` of a row — DSv4's KV
    compressor pools a window once every `compress_ratio` decode tokens, so
    its `s_ctx / compress_ratio` axis is a quarter of a row at one token.
    The product and not the axes one at a time, because a mean is only a
    fraction of a row in company with the axes beside it: an expert bank's
    `[n_expert_copies, M, d]` buffer presents `E·M` rows, and 128 experts of
    a quarter row each are 32 whole rows, one launch.

    The operands of one launch present one row count; two different
    fractions are two cadences in one call, which no launch is.
    """
    seen = set()
    for _extents, law in _paired_tensors(row.get("shape_in"),
                                         row.get("shape_in_exact")):
        rows = Fraction(1)
        for extent in law[:-1]:
            value = _exact(extent)
            if value is None:
                rows = Fraction(1)
                break
            rows *= value
        if 0 < rows < 1:
            seen.add(rows)
    if len(seen) > 1:
        raise ValueError(
            f"operands run at different fractions of a call "
            f"{sorted(str(f) for f in seen)}; not one kernel launch")
    return seen.pop() if seen else Fraction(1)


def _paired_tensors(shape_in: Any, exact: Any) -> List[Tuple[list, list]]:
    """`shape_in` beside `shape_in_exact`, tensor by tensor; empty when the
    two do not line up (a row without the exact form)."""
    ops, exs = _operands(shape_in), _operands(exact)
    if not ops or not exs or len(ops) != len(exs):
        return []
    return [(list(t), list(e)) for t, e in zip(ops, exs, strict=True)
            if isinstance(t, (list, tuple)) and isinstance(e, (list, tuple))
            and len(t) == len(e)]


def _fanouts(row: Dict[str, Any]) -> Dict[int, Fraction]:
    """Each input's stride-0 fan-out, by its index in `shape_in`, as the
    walk priced its read (`point_law.in_terms`, `traffic._fanout`): how
    many times the operand repeats what is stored — a `head_broadcast` K
    comes out `n_heads`. Read at the row's own point."""
    law = row.get("point_law") or {}
    point = law.get("point") or {}
    out: Dict[int, Fraction] = {}
    for term in law.get("in_terms") or ():
        value = term["fanout"]
        value = value.get("value") if isinstance(value, dict) else value
        if isinstance(value, sp.Basic) and value.free_symbols:
            value = value.subs({sym: point[sym.name] for sym in value.free_symbols})
        out[int(term["index"])] = Fraction(str(value))
    return out


def _headless_stores(row: Dict[str, Any]) -> FrozenSet[int]:
    """The inputs whose fan-out expanded a tensor stored with NO head axis,
    by their index in `shape_in` (`point_law.in_terms`,
    `traffic._Fanout.headless_store`).

    The ratio beside it answers this only while the view holds more than one
    head: a rank down to one query head sees a fan-out of 1, which is also
    what a per-head store reads, and absorbed MLA's latent was keyed there
    as the per-head K/V of a kernel nobody runs."""
    law = row.get("point_law") or {}
    return frozenset(int(term["index"]) for term in law.get("in_terms") or ()
                     if term.get("headless_store"))


def stored_operands(sig: Optional[OpSignature], shape_in: Any,
                    fanouts: Dict[int, Fraction],
                    headless: FrozenSet[int] = frozenset()) -> List[Any]:
    """The operand lists a launch's key may be read from, in the order to
    try them: the operands as STORED, where the walk's shapes are a
    stride-0 view.

    An operand the engine broadcast over the heads (`head_broadcast`:
    absorbed MLA's latent, an MQA/GQA K/V) has its head extent multiplied
    by its fan-out (`_fanouts`), and the kernel does not read that: it
    takes the K/V once per stored head. Keying the view named another
    kernel — 128 per-head K/V where one latent is stored — and the
    measurement built that kernel. The head axis is the one the
    signature's own layouts declare (`OpSignature.head_axis`).

    WHETHER the store has a head axis is the walk's own answer
    (`headless`, `_headless_stores`), not this function's arithmetic: a
    tensor stored without one has its head axis dropped at any fan-out,
    including the 1 a rank holding a single query head sees. Read off the
    ratio instead, absorbed MLA above 128-way attention tensor parallelism
    came out as per-head K/V — the non-absorbed prefill geometry, whose
    decode kernel does not exist — and every such cell was refused.

    For a store that does have heads, the extent divided by the fan-out is
    the stored count (the KV heads a GQA kernel groups queries over), one
    head included — the axis stays. A fan-out on an operand whose layout declares no head axis (an
    elementwise broadcast) is not a head share and leaves it as it is, as
    does one on an operand the signature lays out not at all (GQA's V
    beside its K): the key is read off the declared operands, and the
    kernel and its cross-check are built from the key and the declared
    operands (`kernel_bench.refs._carried_axes`), so neither reads it.
    No fan-out: the shapes as they are."""
    operands = list(shape_in) if isinstance(shape_in, (list, tuple)) else None
    if sig is None or operands is None:
        return [shape_in]
    stored = list(operands)
    shared = False
    for index in sorted(set(fanouts) | set(headless)):
        if index >= len(operands):
            raise ValueError(
                f"a read term names input {index} of {len(operands)}: the "
                "row's read laws and its shapes describe different launches")
        fanout = fanouts.get(index, Fraction(1))
        # Index -1 is an in-place update's read of its own output.
        if index < 0 or (fanout <= 1 and index not in headless):
            continue
        operand = operands[index]
        if not isinstance(operand, (list, tuple)):
            continue
        axis = sig.head_axis(index, len(operand))
        if axis is None:
            continue
        if index in headless:
            # No head axis is stored, so there is nothing to divide.
            stored[index] = [*operand[:axis], *operand[axis + 1:]]
            shared = True
            continue
        heads = Fraction(int(operand[axis])) / fanout
        if heads.denominator != 1:
            raise ValueError(
                f"input {index}'s fan-out {fanout} does not divide its "
                f"{int(operand[axis])} heads")
        # A store WITH heads keeps its head axis, at one head too: only the
        # walk's headless answer drops it. Squeezing a one-head store to
        # rank 3 handed it to the stored-latent layouts (`mla_attention`'s
        # and the sparse signatures' `c[B, s_kv, d_head]`) — a K kept per KV
        # head keyed as a shared latent, by shape.
        stored[index] = [*operand[:axis], int(heads), *operand[axis + 1:]]
        shared = True
    return [stored] if shared else [shape_in]


def _one_event(row: Dict[str, Any]) -> Tuple[Any, Any]:
    """`(shape_in, shape_out)` of one launch — the row's shapes as the whole
    geometry a launch runs on (`_whole`): the law's own extents, rounded up
    where the walk averages, so a window pooled once every few tokens is
    the rows a kernel allocates. A routed buffer among them is the launch's
    own geometry (`_launched`): the experts that receive rows, each at its
    rows rounded up — not every expert the rank holds, which a grouped
    kernel launches no blocks for when it receives nothing."""
    shape_in = _whole(row.get("shape_in"), row.get("shape_in_exact"))
    shape_out = _whole(row.get("shape_out") or row.get("shape_outs"),
                       row.get("shape_out_exact"))
    launch = routed_launch(row)
    if launch is None:
        return shape_in, shape_out
    wire = row["routed"]
    buffers = frozenset(wire["inputs"])
    shape_in = _banked(_launched(shape_in, launch, buffers),
                       row.get("shape_in_dims"), launch, skip=buffers)
    return (shape_in,
            _launched(shape_out, launch, True) if wire["output"]
            else shape_out)


class RowIdentity:
    """What a per-op row is measured AS — resolved once, per analysis.

    Three producers write this key (the kernel_bench plan, the contributor
    CSV template, the serving overlay's own rows) and the overlay reads it
    back. They must agree byte for byte or the lookup is a permanent silent
    miss, and they did not: the template passed two of the five arguments
    the plan passes, under a comment claiming the two matched by
    construction. So the derivation stops being a call with arguments a
    caller can get wrong and becomes a thing bound to the analysis, which
    answers per row.
    """

    def __init__(self, analysis: Any, *,
                 kv_block_size: Optional[int] = None,
                 ssm_chunk_size: Optional[int] = None) -> None:
        # The serving choices a cell may be keyed on, which the deployment
        # makes and the model does not: the paged-KV page (`PAGE_AXIS`; the
        # framework binding states it and a request may override it) and the
        # Mamba-2 scan chunk (`Program.scan_chunk_size`). Handed to
        # the projection beside the row's own dims, so only a signature
        # whose cell declares the axis reads it (`OpSignature.cell_dims`).
        # None is not stated: a cell that declares the axis then has no
        # identity, rather than one keyed on a guessed value.
        self._served = {axis: int(value) for axis, value in (
            (PAGE_AXIS, kv_block_size), ("ssm_chunk_size", ssm_chunk_size))
            if value is not None}
        self._leaves = gqa_leaf_variants(analysis)
        self._result = analysis
        self._program = analysis.program
        # A row's node, by the aggregation id the row carries. Every
        # invocation of a template node shares the id, and shares what is
        # asked of it here (which component it sits in, whether it is
        # routed), so any one of them answers.
        self._node_by_sid: Dict[str, Any] = {}
        for node in self._program.nodes():
            self._node_by_sid.setdefault(str(node.aggregation_id), node)

    def _node_of(self, row: Dict[str, Any]) -> Optional[Any]:
        """The row's node, or None for a row that is not one (a template
        header carries no aggregation id). A row that names a node the
        program does not have is a refusal, not a key."""
        sid = str(row.get("aggregation_id", "") or "")
        if not sid:
            return None
        node = self._node_by_sid.get(sid)
        if node is None:
            raise ValueError(
                f"per-op row {sid!r} has no node in the program it was "
                "produced from; refusing to key it")
        return node

    def leaf_of(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """This leaf's kernel variants (`s_window`, `sinks`), or empty.

        Per-node, not the model-wide dim table: a hybrid's full-attention
        layers must not inherit the window its sliding layers declare.
        """
        return self._leaves.get(str(row.get("aggregation_id", "") or "")) or {}

    def op_of(self, row: Dict[str, Any], node: Optional[Any]) -> str:
        """The kernel behind the engine's label.

        The engine calls every attention op `flash_attention`; the true
        variant reads off the cache slots its keys come from
        (`KVAccess.key_slots`), the program's routedness of the row's node
        (`_node_of`), and the operand rank — never the node id's spelling.
        """
        if node is None:
            return resolve_op(str(row.get("op", "") or ""), row.get("shape_in"))
        access = self._result.kv_access.get(node)
        return resolve_op(
            str(row.get("op", "") or ""), row.get("shape_in"),
            key_slots=access.key_slots if access is not None else frozenset(),
            routed=self._program.node_is_routed(node))

    def of(self, row: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """`(op, shape_json)` — the row's kernel and its identity shape.

        `shape_json` is None when the row has no projection, and that is a
        refusal rather than a gap to fill. The stand-in on offer is the
        model's whole-step shape (B, the context length, the model's head
        counts); for any op whose kernel geometry differs from that frame it
        is a FABRICATION, naming a kernel the model never runs — DSA
        attention was measured at `d_head = seq_len` that way, 2430 rows of
        it. A caller drops the row and says which projection is missing.

        A refusal raised while keying the row names the row's node.
        """
        op, shape_json, _event = self.of_event(row)
        return op, shape_json

    def of_event(self, row: Dict[str, Any]
                 ) -> Tuple[str, Optional[str], Tuple[Any, Any]]:
        """`of`, and the one launch it keyed the row at (`one_event`)."""
        node = self._node_of(row)
        if node is None:
            return self._keyed(row, None)
        with node_context(node.display_path):
            return self._keyed(row, node)

    @cached_property
    def _resident_caches(self) -> Dict[Any, Any]:
        """Every persisting cached tensor, by reference (`cache.resident`):
        what a row's own cache accesses (`kv_access`) name."""
        return cache_mod.resident(self._result.analysis)

    def carries_state(self, row: Dict[str, Any]) -> bool:
        """Whether this row's leaf is a recurrent state's carrier: the
        kernel that reads and rewrites a state no edge carries
        (`CachedTensor.recurrent`, reached through the row's own cache
        accesses, `kv_access`). The walk bills that state's read and
        write-back on the carrier's row (`traffic._settle_cache_reads`,
        `_settle_state_writes`), so a launch can only be this cell if its
        kernel takes the state (`OpSignature.carries_state`)."""
        node = self._node_of(row)
        access = None if node is None else self._result.kv_access.get(node)
        if access is None:
            return False
        return any(self._resident_caches[ref][2].recurrent
                   for ref in access.reads | access.writes
                   if ref in self._resident_caches)

    def _cache_tensor(self, row: Dict[str, Any]
                      ) -> Tuple[Optional[str], Optional[str]]:
        """`(dtype, format name)` of the cache this row itself reads or
        writes (`KVAccess.reads` / `.writes`) — the key's `kv` and
        `kv_format` — `(None, None)` when it touches no cache —
        less the cached tensor it writes as its primary output, whose
        format is the row's `output` slot (`leaf_dtype_env`): one fact,
        one slot.

        The row's own, not the step's: a projection or a norm reads no
        cache, and keying it with the attention's cache dtype made one BF16
        GEMM two keys across checkpoints.

        An attention whose key and value operands come from the cache says
        which slot kinds they come from (`KVAccess.key_slots`), and only
        those tensors are its cache. Some kernels read several at different
        dtypes (DSA: an FP8 latent beside a BF16 rope key; DSv4.1: an FP8
        window beside an FP4 compressed cache), and the key has one slot.
        That slot is the row's tensor holding the most bytes per stored
        position, the same rule `Analysis.dominant_cache_dtype` applies
        model-wide, here over the row's own tensors only; a tie between
        two dtypes is refused rather than broken.

        The format name is that same tensor's (`CachedTensor.format_name`):
        the record the dtype is the element dtype of. Two tensors that tie
        on bytes at that dtype but are stored two ways are refused for the
        same reason a dtype tie is."""
        node = self._node_of(row)
        access = None if node is None else self._result.kv_access.get(node)
        if access is None:
            return None, None
        where = repr(row.get("aggregation_id"))
        # The tensor the row writes as its primary output is keyed once, as
        # its `output` (`leaf_dtype_env`), not again here.
        published = row.get("output_cache")
        written = frozenset(
            ref for ref in access.writes
            if ref in self._resident_caches and published is not None
            and (self._resident_caches[ref][2].node,
                 self._resident_caches[ref][2].name) == tuple(published))
        refs = access.reads | (access.writes - written)
        absent = sorted(str(ref) for ref in refs
                        if ref not in self._resident_caches)
        if absent:
            raise ValueError(
                f"per-op row {where} accesses cached tensors no persisting "
                f"cache keeps: {absent}")
        tensors = [self._resident_caches[ref][2] for ref in refs]
        if access.key_slots:
            tensors = [t for t in tensors if t.kind in access.key_slots]
        sizes = [(float(value_of(t.data_position_bytes)), t) for t in tensors]
        if not sizes:
            return None, None
        top = max(size for size, _t in sizes)
        lead = sorted({t.dtype for size, t in sizes if size == top})
        if len(lead) > 1:
            raise ValueError(
                f"per-op row {where}: its cached tensors of {lead} hold "
                f"equal bytes per position; its key has one `kv` slot")
        stored = sorted({str(t.format_name) for size, t in sizes
                         if size == top})
        if len(stored) > 1:
            raise ValueError(
                f"per-op row {where}: its cached tensors stored as {stored} "
                f"hold equal bytes per position; its key has one "
                f"`kv_format` slot")
        chosen = next(t for size, t in sizes if size == top)
        return chosen.dtype, chosen.format_name

    def dtype_env(self, row: Dict[str, Any]) -> str:
        """The dtype half of this row's measurement key (`dtype_env_json`).

        The other half is `of`, and the two have the same reason to live
        here. Three producers write this key — the kernel_bench plan, the
        contributor CSV template, the serving overlay — and the overlay
        reads it back; any slot spelled differently is a permanent silent
        miss. The shape half was given one owner after three producers
        drifted on it. The dtype half then drifted the same way and worse:
        the plan stated three of the six slots and re-derived two of them
        from the spec's materialized tensors, so `metadata` defaulted to
        fp32 against the row's real scale dtype and every measured job of
        Kimi-K3, DeepSeek-V4-Flash, gpt-oss-20b and Kimi-K2-Thinking was
        unreachable.

        One leaf's environment, per the design's §4.5: the weight at its own
        resident format, the activation at the dtype its matmul runs at, its
        own scales — a BF16 GEMM inside an FP8 checkpoint is looked up as
        BF16. Every one of those is a fact the WALK put on the row
        (`phases`: `act_dtype`, `weight_dtype`, `metadata_dtype`,
        `weight_granularity`), so this reads them rather than asking the
        spec a second time, and the cache slot is the cache tensor the leaf
        itself accesses (`_cache_tensor`).

        No slot borrows a model-level value. A leaf that streams no weight
        has no `weight` slot (it used to be keyed with its activation's
        dtype); one that touches no cache has no `kv` slot (it used to carry
        the step's widest cache); one whose weight stores no scales has no
        `metadata` slot (it used to carry the step's most common scale
        dtype, so one BF16 GEMM was keyed `ue8m0` inside one checkpoint and
        `fp32` inside another).
        """
        return self.leaf_dtype_env(row)[0]

    def leaf_dtype_env(self, row: Dict[str, Any]) -> Tuple[str, str]:
        """`(dtype_env, declined)`: `dtype_env` as above, and `declined` ""
        when it is this leaf's whole environment, else why it is not — one
        of the leaf's dtypes has no corpus token (a packed U8 bank, an I64
        routing table). Such a row is keyed by the slots that do have one,
        the rest left out, and neither the plan nor the estimate looks it up
        (`calibration_overlay.PointRow.declined`); both sides take this one
        branch, so they agree on its bytes.

        The I64 routing table is NOT waiting for a token in the dtype table.
        DeepSeek-V4 ships `tid2eid` as I64 and that is the width this row
        carries, but both frameworks narrow it at load and launch on what
        they loaded: sglang allocates `torch.int32` and its kernel verifies
        int32 (`srt/layers/moe/hash_topk.py:66-67`,
        `kernels/jit/csrc/deepseek_v4/hash_topk.cuh:24-29`), and vLLM takes
        int64 only under the explicitly chosen native MegaMoE backend
        (`models/deepseek_v4/nvidia/model.py:683`, `:741-743`; `moe_backend`
        defaults to `auto`). Keying the stored width would promise a
        measurement of a table no default deployment allocates — and the
        walk bills that width too, so the gather is billed at 8 bytes a row
        where the default kernel reads 4. The loaded width has to become a
        fact the walk states before this leaf can be keyed or billed right;
        giving I64 a corpus token only hides the first half.

        A framework's declared OUTPUT width is now such a fact
        (`FrameworkBinding.edge_dtypes`, which is why this span's logits key
        fp32), but a RESIDENT tensor's loaded width is a different object —
        `Representation.intent`, fixed while the walk builds its formats and
        so out of reach of a binding answered at serving time. See
        `docs/inference_cost_model.md` "Stored width vs loaded width" for
        what that would take.
        """
        where = repr(row.get("aggregation_id"))
        activation = row.get("act_dtype")
        if activation is None:
            # KNOWN GAP, unchanged here: without a framework binding the
            # walk's access layer has no entry for a collective (comm) node,
            # so its row carries no `act_dtype`, and the model's edge dtype
            # stands in. The fix is the access layer stating it for comm
            # nodes, not a refusal here (that refuses every TP>1 replay).
            activation = self._result.analysis.resolved_precision.model_dtype
        # The leaf's own facts; None where the slot does not apply to it.
        kv, kv_format = self._cache_tensor(row)
        own = {"weight": row.get("weight_dtype"),
               "kv": kv,
               "metadata": row.get("metadata_dtype")}
        granularity = row.get("weight_granularity")
        if granularity is not None and own["metadata"] is None:
            raise ValueError(
                f"per-op row {where} stores scales ({granularity}) but "
                "carries no `metadata_dtype`")
        # What the walk priced the primary output at (`LeafAccess.
        # output_dtype`): every declaration names its dtype (`OpSpec.
        # output_dtype`), so a top-k's int32 indices are keyed `int32`.
        output = row.get("output_dtype")
        # Where the selected launch eats a quantised INPUT format, its
        # granularity is part of the activation slot: FP8 over 32 and FP8
        # over 128 are different kernels at one dtype. The input wins over a
        # quantised output below, because this leaf's launch consumes it.
        input_granularity = row.get("activation_granularity")
        input_scales = row.get("activation_scale_dtype")
        if input_scales is not None and own["metadata"] is None:
            # Where a weight also stores scales, that weight's dtype remains
            # the one `metadata` slot; the activation's own geometry is still
            # stated by `activation_granularity`.
            own["metadata"] = input_scales
        # Where that output is a quantised activation format
        # (`LeafAccess.output_activation_format`, on the row by name — a
        # grafted cast's whole point), the format's own declaration states the rest of the
        # environment: which elements share a scale, and the scales' dtype
        # as `metadata` when the weight stores none. The key has one
        # `metadata` slot, so a leaf whose weight AND output both carry
        # scales is refused rather than keyed with either standing for both.
        name = row.get("output_activation_format")
        written = None if name is None else ACTIVATION_FORMATS[name]
        if written is not None and own["metadata"] is not None:
            raise ValueError(
                f"per-op row {where}: its weight's scales "
                f"({own['metadata']}) and its written activation's "
                f"({written.name}: {written.scale_dtype}) would share one "
                "`metadata` slot")
        scales = None if written is None else written.scale_dtype
        unknown = sorted(f"{slot}={dtype}" for slot, dtype in
                         (("activation", activation), ("output", output),
                          *own.items(), ("metadata", scales))
                         if dtype is not None and not is_known_dtype_name(dtype))
        if f"activation={activation}" in unknown:
            raise ValueError(
                f"per-op row {where}: its activation dtype {activation!r} has "
                "no corpus token, and every key states one")
        tokens = {slot: (normalize_dtype_name(dtype)
                         if dtype is not None and is_known_dtype_name(dtype)
                         else None)
                  for slot, dtype in (*own.items(), ("scales", scales))}
        # How the cached tensor the row outputs is stored, and the cache its
        # `kv` slot names, by registry name (`CachedTensor.format_name`, one
        # rule for both); None for an activation or a cache stored plain. A
        # format the registry does not name (a profile's spelled-out body,
        # carried by its label) has no key, and declines the row the way a
        # dtype without a corpus token does.
        formats = {"output_format": row.get("output_format"),
                   "kv_format": kv_format}
        unnamed = []
        for slot, name in formats.items():
            if name is not None and name not in CACHE_FORMATS:
                what = ("output" if slot == "output_format" else "cache")
                unnamed.append(f"its {what} is stored as {name}, which no "
                               f"registry cache format names")
                formats[slot] = None
        output_token = (normalize_dtype_name(output)
                        if output is not None and is_known_dtype_name(output)
                        else "bf16")
        env = compute_per_row_dtype_env_json(
            weight_dtype=tokens["weight"],
            activation_dtype=normalize_dtype_name(activation),
            kv_dtype=tokens["kv"],
            # The one `metadata` slot holds the weight's scales where the
            # weight stores them, else the written activation's: a norm
            # whose gamma is plain bf16 and whose output a fused launch
            # writes FP8 keys the activation's scales (both at once was
            # refused above).
            metadata_dtype=(tokens["metadata"] if tokens["metadata"]
                            is not None else tokens["scales"]),
            output_dtype=output_token,
            weight_granularity=(granularity if tokens["weight"]
                                and tokens["metadata"] else None),
            activation_granularity=(input_granularity or
                                    (written.granularity
                                     if tokens["scales"] else None)),
            output_format=(formats["output_format"]
                           if f"output={output}" not in unknown else None),
            kv_format=(formats["kv_format"] if tokens["kv"] is not None
                       else None))
        launch_gap = row.get("launch_gap")
        missing = [*([f"precision launch selector gap: {launch_gap}"]
                     if launch_gap is not None else []),
                   *([f"the leaf's dtype environment has no corpus token "
                      f"for {', '.join(unknown)}"] if unknown else []),
                   *unnamed]
        if not missing:
            return env, ""
        return env, (f"{'; '.join(missing)}; keyed by its other slots, which "
                     "is not this leaf's environment")

    @staticmethod
    def one_event(row: Dict[str, Any]) -> Tuple[Any, Any]:
        """`(shape_in, shape_out)` of ONE launch of this row's kernel — the
        geometry a measurement is built at, and the geometry the identity is
        keyed by (`_keyed`). The law's own extents, whole: an axis the walk
        averages is the rows a kernel allocates, rounded up (`_whole`)."""
        return _one_event(row)

    @staticmethod
    def launch_weight(row: Dict[str, Any]) -> Any:
        """The weight ONE launch of `one_event` reads (`shape_weight` as the
        launch runs it). A launch on a routed buffer reads the weights of the
        experts it touches: every axis of the weight whose identity is an
        expert axis (`shape_weight_dims`) is the touched experts (`_banked`);
        a weight with none is the weight as published."""
        weight = row.get("shape_weight")
        launch = routed_launch(row)
        if launch is None or weight is None:
            return weight
        return _banked(weight, row.get("shape_weight_dims"), launch)

    @staticmethod
    def launch_rows(row: Dict[str, Any]) -> Tuple[Any, Fraction]:
        """`(launched, exact)` rows one launch of `one_event` runs
        (`_launch_rows`): what the geometry runs, and what the law states
        for it. Equal unless the row is on a routed buffer."""
        return _launch_rows(row)

    @staticmethod
    def launches(row: Dict[str, Any]) -> int:
        """How many launches of `one_event` one rank runs per step — the
        number the step's numerators are divided by to get one launch's, and
        the number a measured launch multiplies back by. The row's count
        (`count_per_step`): an expert bank is one node per layer whose
        tensors carry its experts as an axis, so one launch of it runs every
        expert the rank holds, and the count is its launches."""
        return max(int(row.get("count_per_step") or 1), 1)

    @staticmethod
    def cadence(row: Dict[str, Any]) -> Fraction:
        """The share of one launch a call of this row is — 1 for almost every
        row.

        A row whose work happens once every m steps presents `1/m` of a row
        per call — DSv4's KV compressor pools a window once every
        `compress_ratio` decode tokens, so its slot axis is
        `s_ctx / compress_ratio`, `1/4` at one token. The cell is one launch
        (`one_event`); the step bills `launches x cadence` of it."""
        return _averaged_fraction(row)

    @staticmethod
    def _node_dims(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The row's own Invocation dims (`node_dims`), which every row the
        walk publishes carries — None for a node the walk bound no dims at
        (a bare collective, whose cell is its payload). The model-wide dim
        table used to stand in for a row without them, and handed the row
        tie-breaker axes of nodes that are not it: a key no request asks
        with. A row without the field was not produced by the walk."""
        if "node_dims" not in row:
            raise ValueError(
                f"per-op row {row.get('aggregation_id')!r} carries no "
                "`node_dims`; its key reads the node's own dims, never the "
                "model's")
        return row["node_dims"]

    def _keyed(self, row: Dict[str, Any], node: Optional[Any]
               ) -> Tuple[str, Optional[str], Tuple[Any, Any]]:
        op = self.op_of(row, node)
        leaf = self.leaf_of(row)
        walked, shape_out = self.one_event(row)
        forms = stored_operands(get_signature(op_signature(op)), walked,
                                _fanouts(row), _headless_stores(row))
        for shape_in in forms:
            shape_json = self._identity(op, row, leaf, shape_in, shape_out,
                                        exact=shape_in is walked)
            if shape_json is not None:
                break
        else:
            shape_in = walked
        return op, attach_gqa_sinks(shape_json, bool(leaf.get("sinks"))), (
            shape_in, shape_out)

    def _identity(self, op: str, row: Dict[str, Any], leaf: Dict[str, Any],
                  shape_in: Any, shape_out: Any, *, exact: bool) -> Optional[str]:
        """The identity shape of `row` read off `shape_in` — the walk's
        shapes (`exact`: their exact extents apply) or a stored form of
        them (`stored_operands`)."""
        # A launch on a routed buffer runs its own geometry (`one_event`),
        # not the law's extents: the experts that receive rows, padded. The
        # law's exact extents would put every expert the rank holds back.
        launched = routed_launch(row) is not None
        return identity_shape_from_shape_in(
            op, shape_in,
            shape_out=shape_out,
            shape_in_exact=(row.get("shape_in_exact")
                            if exact and not launched else None),
            shape_out_exact=None if launched else row.get("shape_out_exact"),
            # Spec constants a tensor shape can't carry (the rope rotation
            # width hides inside heads x width, …) — off the row's own
            # Invocation dims and nowhere else (`_node_dims`).
            spec_dims=row_invocation_dims(
                self._node_dims(row),
                signature=get_signature(op_signature(op)),
                # Only a signature declaring the axis reads it into its cell.
                extra={**leaf, **self._served},
                # The group a fused collective reduces over, as the walk
                # priced it (`RowRoofline` reads the same number). No tensor
                # carries it and the deployment label does not name it —
                # one deployment has several groups.
                comm_world=row.get("comm_world")),
            # A BARE collective's whole cell: it has no local math, so its
            # payload, group and mesh spacing are the launch, and the row
            # states all three. None for every row that is not one.
            comm=CommLaunch.from_row(row),
            # The routed buffer's rows, read off the row's own wire
            # identity — the same count the overlay's padding reads
            # (`launch_rows`), so a key and its geometry name one launch.
            routed_rows=routed_launch_rows(row),
        )


# Rotary embedding, fused over Q and K. The engine hands the two rotated
# tensors flattened as `[B, s_query, heads·d_rope]`, so the head count and the
# rotation width are multiplied together and neither is recoverable from the
# shape alone — `spec_dims` supplies the one that breaks the tie.
# Packed-qkv fused norm+rope still has no engine layout (one concatenated
# operand, not Q/K). fused_qk_rope goes through the signature zip.


def _conv2d_identity(shape_in: Any, shape_out: Any,
                     spec_dims: Optional[Dict[str, Any]]) -> Optional[str]:
    """conv2d identity `{B, s_conv_grid, C_in, C_out, kernel_size, stride}`.

    The encode walk hands the conv each image's patch grid as rows,
    `[segments, rows, C_in]` (`patchify`: one segment per item, `rows` the
    grid at that point of the tower), so `s_conv_grid` is the grid the
    kernel convolves. A kernel is benched over a square grid
    (`kernel_bench.refs._conv2d_grid`); a grid that is not square has no
    cell here and the plan reports it rather than benching another one.
    `C_out` is the output width; `kernel_size` / `stride` are no tensor's
    axis, so they come from the spec. Stride is part of the cell — two
    convs that differ only in stride are different kernels.
    """
    import math
    t = _first_tensor(shape_in)
    if t is None or len(t) != 3:
        return None
    try:
        B, rows, C_in = int(t[0]), int(t[1]), int(t[2])
    except (TypeError, ValueError):
        return None
    C_out = output_width(shape_out)
    sd = spec_dims or {}
    k, stride = sd.get("kernel_size"), sd.get("stride")
    if (B <= 0 or rows <= 0 or C_in <= 0 or not C_out or C_out <= 0
            or not k or not stride or math.isqrt(rows) ** 2 != rows):
        return None
    return json.dumps({"B": B, "s_conv_grid": rows, "C_in": C_in,
                       "C_out": C_out, "kernel_size": int(k),
                       "stride": int(stride)},
                      separators=(",", ":"), sort_keys=True)


def _inv_rope_identity(shape_in: Any,
                       spec_dims: Optional[Dict[str, Any]]) -> Optional[str]:
    """`o[B, n_heads, s_query, d_head]` → `{B, n_heads, s_query, d_head,
    d_rope}`.

    The attention output, whose last `d_rope` channels the kernel rotates
    back. Only the activation is an edge — the positions and the cos/sin
    table are the kernel's own operands and no node produces them — so the
    rotation width comes from the spec (`d_qk_rope`), as `_rope_identity`
    takes it for a packed K.
    """
    o = _first_tensor(shape_in)
    if o is None or len(o) != 4:
        return None
    d_rope = int((spec_dims or {}).get("d_qk_rope") or 0)
    if d_rope <= 0:
        return None
    try:
        dims = {"B": int(o[0]), "n_heads": int(o[1]), "s_query": int(o[2]),
                "d_head": int(o[3]), "d_rope": d_rope}
    except (TypeError, ValueError):
        return None
    if any(v <= 0 for v in dims.values()) or dims["d_rope"] >= dims["d_head"]:
        return None
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _rope_identity(shape_in: Any, spec_dims: Optional[Dict[str, Any]]) -> Optional[str]:
    """Q/K rotary identity `{B, s_query, n_heads, n_kv_heads, d_rope}`.

    Two engine layouts, one cell:

      * packed  `[B, s_query, heads·d_rope]` — heads and width multiplied;
        `spec_dims` supplies `d_rope` (do NOT divide K's width by the
        model's KV-head count: MLA's `k_pe` is one shared rope head).
      * head-shaped `[B, n_heads, s_query, d_rope]` — GQA keeps the head
        axis. Both Q and K are 4-D on Llama/Mistral/GPT-OSS; MLA is 4-D Q
        beside a packed 3-D K. The last dim IS the rotation width.

    The previous projector required K to be 3-D, so every GQA fused rope
    dropped as "no shape projection".
    """
    if not (isinstance(shape_in, (list, tuple)) and len(shape_in) >= 2):
        return None
    q, k = shape_in[0], shape_in[1]
    if not (isinstance(q, (list, tuple)) and isinstance(k, (list, tuple))):
        return None
    if len(q) not in (3, 4) or len(k) not in (3, 4):
        return None
    sd = spec_dims or {}
    try:
        if len(q) == 4:
            B, n_heads, Sq, d_rope = (int(q[0]), int(q[1]),
                                      int(q[2]), int(q[3]))
        else:
            B, Sq, w_q = int(q[0]), int(q[1]), int(q[2])
            width = sd.get("d_rope") or sd.get("d_qk_rope") or sd.get("d_head")
            d_rope = int(width) if width else 0
            if d_rope <= 0 or w_q % d_rope:
                return None
            n_heads = w_q // d_rope
        if len(k) == 4:
            if int(k[0]) != B or int(k[2]) != Sq:
                return None
            n_kv_heads, d_k = int(k[1]), int(k[3])
            if d_k != d_rope:
                return None
        else:
            w_k = int(k[2])
            if d_rope <= 0:
                d_rope = int(sd.get("d_rope") or sd.get("d_qk_rope")
                             or sd.get("d_head") or w_k)
            if d_rope <= 0 or w_k % d_rope:
                return None
            n_kv_heads = w_k // d_rope
    except (TypeError, ValueError):
        return None
    dims = {"B": B, "s_query": Sq, "n_heads": n_heads,
            "n_kv_heads": n_kv_heads, "d_rope": d_rope}
    if any(v <= 0 for v in dims.values()):
        return None
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _moe_gate_identity(shape_in: Any, shape_out: Any) -> Optional[str]:
    """`[x[B,s_query,d_model], bias[n_experts]]` → `{B, s_query, n_experts,
    n_experts_active}`.

    The router scores every expert for every token and keeps the top-k, so those
    four numbers are the whole cost. The group structure (`n_expert_groups` /
    `topk_group` in the noaux_tc variants) selects WHICH experts win, not how
    much work it takes, and is deliberately left out of the identity so the same
    kernel measurement is shared across models that group differently.
    """
    if not (isinstance(shape_in, (list, tuple)) and len(shape_in) >= 2):
        return None
    x, bias = shape_in[0], shape_in[1]
    if not (isinstance(x, (list, tuple)) and isinstance(bias, (list, tuple))):
        return None
    if len(x) != 3 or len(bias) != 1:
        return None
    topk = output_width(shape_out)
    if not topk or topk <= 0:
        return None
    try:
        dims = {"B": int(x[0]), "s_query": int(x[1]),
                "n_experts": int(bias[0]), "n_experts_active": int(topk)}
    except (TypeError, ValueError):
        return None
    if any(v <= 0 for v in dims.values()):
        return None
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _cell_from_spec(sig: OpSignature, shape_json: Optional[str],
                    spec_dims: Optional[Dict[str, Any]]) -> Optional[str]:
    """A projected cell completed with the cell axes it does not carry, read
    off `spec_dims` exactly as the contract path fills them
    (`_contract_identity`): the DSA cell's geometry is projected by hand, and
    its page (`PAGE_AXIS`) is the deployment's, not a tensor's. Missing
    after that is incomplete — None, not a guessed value."""
    if shape_json is None:
        return None
    dims = json.loads(shape_json)
    for axis in sig.cell_dims:
        if axis in dims:
            continue
        raw = spec_get(spec_dims or {}, axis)
        if raw is None or int(raw) <= 0:
            return None
        dims[axis] = int(raw)
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def _dsa_identity(sig: Optional[OpSignature], shape_in: Any, shape_out: Any,
                  spec_dims: Optional[Dict[str, Any]]) -> Optional[str]:
    """A top-k attention span's cell, read through the signature's own
    engine layouts — by operand position and rank, not by searching the
    operand list for a shape that looks right.

    Two geometries, told apart by how the graph stores the cache operand
    (input 1, the score's K), which the layouts encode by rank:

      * stored with no head axis, `c[B, s_kv, d_head]` — one latent row a
        key shared by every head, the value its leading columns (absorbed
        MLA / DSA, `mla_attention`'s geometry): `{B, n_heads, s_query,
        s_kv, n_topk_keys, d_head, d_v_head}`. The walk says which tensors
        are stored headless (`traffic._Fanout.headless_store`) and the row's
        stored operands drop that axis (`stored_operands`).
      * stored per KV head, `k[B, n_kv_heads, s_kv, d_head]` with its V —
        top-k GQA (Qwen QSA, Step-5): the cell adds `n_kv_heads`.

    The spec's latent width (`d_kv_lora`) chose between them before; a stale
    QSA plan carried no `n_kv_heads`, was keyed latent and handed FlashMLA a
    256-wide head ("Invalid d_qk: 256").

    The keys a query attends are the cache operand's rows. The walk sizes
    that operand at the score op's per-query read (`kv_positions_per_query_fn`
    = `Min(s_kv, k)`, with `k` the node's own top-k budget) and bills the
    kernel for exactly those rows, so the cell takes them as both its cache
    and its selection: the measured launch then gathers the rows the
    numerator counts. Two other readings disagreed with that walk and are
    gone. The selection operand's width is the indexer's selection, not the
    keys attended: GLM-5.3-Flash selects `n_topk_keys / compress_ratio`
    POOLS and expands each back to `compress_ratio` keys plus the open tail
    (vLLM `expand_pools_and_append_tail`: `topk + pool_size - 1` = 2051
    token indices a query), so a cell keyed at the 512 pools — or at 288
    when the cache held 1152 keys — timed a quarter of the gather the row
    bills. And the spec's `n_topk_keys` is the indexer's budget: past the
    tail above, and short of a prefill shorter than it, where the gated QSA
    span keyed 2048 over 1024 cached keys and no build could select them.
    It is still required: a kernel without a declared top-k is not
    identifiable as a sparse one.

    The cell's `s_kv` is the attended rows, not the stored cache a query
    selects out of: no cell axis carries that length yet (see
    `kernel_bench.refs._mla_sparse_indices`).
    """
    if sig is None or not isinstance(shape_in, (list, tuple)):
        return None
    Dv = output_width(shape_out)
    if not Dv or Dv <= 0:
        return None
    if int((spec_dims or {}).get("n_topk_keys") or 0) <= 0:
        return None
    bound, laid = None, ()
    for engines in (sig.engine_inputs, sig.variant_engine_inputs):
        got = zip_engine_layouts(engines, shape_in, spec=spec_dims)
        if got and "s_kv" in got:
            bound, laid = got, engines
            break
    if bound is None:
        return None
    # Every operand the geometry lays out must be there:
    # `zip_engine_layouts` skips a declared position the operand list does
    # not reach, so a per-head span missing its V would bind on Q and K.
    if any(pos >= len(shape_in) for pos, _layout in laid):
        return None
    # The latent is the cache stored with no head axis, on every sparse
    # signature: a geometry without KV heads binds input 1 only in that
    # stored form. A signature may lay the broadcast out at rank 4 to say
    # where its head axis is (Hy4's sink span), and `stored_operands` drops
    # that axis where the walk marks the store headless; a rank-4 cache
    # still here was stored per head — not this geometry's.
    if "n_kv_heads" not in bound and len(shape_in[1]) != 3:
        return None
    dims = dict(bound, n_topk_keys=bound["s_kv"])
    cell = sig.cell_dims_for(dims)
    if "n_kv_heads" in dims and dims["n_heads"] % dims["n_kv_heads"]:
        return None
    # The value's width: the output's, where V has a width of its own (the
    # latent's leading columns, or a laid-out V). A geometry that names none
    # (the gated span: V at the head width, its output merged over heads)
    # has no output axis to read.
    if "d_v_head" in cell and dims.setdefault("d_v_head", int(Dv)) != int(Dv):
        return None
    dims = {axis: int(v) for axis, v in dims.items() if axis in cell}
    if any(v <= 0 for v in dims.values()):
        return None
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


# Canonical op name → library-agnostic op SIGNATURE: the catalog's `binds_ops`,
# projected by `op_signatures.op_signature` / `op_bindings`. The harness reads
# the same projection (`kernel_bench.signatures.op_to_signature`) — kernel_bench
# imports backend and never the reverse.
#
# The table used to be a literal here and a mirror there, and the mirror
# carried nine entries this copy did not. The consequence is what the envelope
# key exists to prevent: with `linear`, `qkv_parallel_linear` and
# `merged_column_parallel_linear` all absent here, `to_physical` left three
# names for one piece of math in three different physical slots, and 50 cells
# of the 25k-row H100 corpus were measured twice under two of those names —
# same projected shape, same dtypes, same GPU, same phase — while a node
# labelled `linear` still could not see either measurement.
#
# An op no signature binds is its own signature (a valid, if
# framework-specific, grouping key) — and is simultaneously unmeasurable to
# the harness, which returns None rather than inventing a signature no
# provider serves. That is the visible half: a new op fragments nothing, it
# just cannot be planned until a signature binds it.


def resolve_op(op: str, shape_in: Any = None, *,
               key_slots: FrozenSet[str] = frozenset(),
               routed: bool = False) -> str:
    """The kernel behind the engine's op label, from facts the caller reads
    off the analysis — never from the node id's spelling.

    `key_slots` are the cache slot kinds the row's keys and values come from
    (`KVAccess.key_slots`). An op measured as one of the attention kernel
    families the slot kinds declare (`kernel:` in cache_families.yaml) is
    measured as the family its slots name: a dense span over an MLA latent
    is `mla_attention` — latent KV, asymmetric qk / v dims, FlashMLA's own
    family. Slots naming no family, or two, are refused.
    `routed` is `Program.is_routed`: an expert projection is a `linear` /
    `merged_column_parallel_linear` with a grouped shape and runs the
    GROUPED gemm kernel (DeepGEMM m_grouped), not the dense GEMM. Both the
    write side (plan) and the lookup side (overlay) go through
    `RowIdentity.op_of`, so the physical key agrees by construction.
    """
    kernels = {k.name: k.kernel for k in slot_kinds().values()}
    if key_slots and op_signature(op) in set(kernels.values()):
        named = {kernels.get(kind) for kind in key_slots}
        if None in named or len(named) != 1:
            raise ValueError(
                f"{op!r} reads keys from cache slots {sorted(key_slots)!r}, "
                f"which name kernels {sorted(map(str, named))!r}: no one "
                f"kernel serves this span")
        (kernel,) = named
        if kernel != op_signature(op):
            return _rewrite(op, kernel)
    if _is_gemm_kernel(op) and routed:
        return _rewrite(op, "grouped_gemm")
    # Linear attention splits into two kernel families by the RANK of its decay
    # operand, and the engine's own shape says which: Gated DeltaNet decays the
    # state by one scalar per head (`g [B,n_lin_v_heads,s_query]`), Kimi Delta
    # Attention by a vector per key channel
    # (`g [B,n_lin_v_heads,s_query,d_lin_k_head]`). They are separate Triton kernels
    # in both frameworks — vLLM/sglang vendor `chunk_gated_delta_rule` and
    # `chunk_kda` side by side, and sglang's decode kernel takes an explicit
    # `is_kda` switch. Same dims, so without this split the two would share one
    # physical key and each would be served the other's measurement.
    sig = get_signature(op_signature(op))
    if (sig is not None and sig.decay_rank_variant
            and _has_channelwise_decay(shape_in)):
        return _rewrite(op, sig.decay_rank_variant)
    return op


def _rewrite(was: str, now: str) -> str:
    """A resolution, checked where it happens.

    Graph → operator is MANY-TO-ONE and lossy by design: three linear-shaped
    ops are one `gemm`, a three-node attention span is one `gqa_attention`, and
    which operator a node is can depend on context the op name does not carry
    (an `…expert…` id, a 4-D decay). So a rewrite's target need not be a name
    the graph uses — it is a name in the OPERATOR vocabulary, which is a
    quotient of the graph's, not a subset.

    What must hold is that the quotient is total: a rewrite that lands outside
    the measurable vocabulary has turned a node the harness could have measured
    into a key nothing can serve, and it would do so silently. Enforced here,
    at the one place rewrites are made, rather than as a list of their results
    kept somewhere else and compared by a test."""
    if now not in op_bindings():
        raise KeyError(
            f"resolve_op rewrote {was!r} to {now!r}, which no signature "
            f"measures — bind it on a signature in the catalog or do not rewrite")
    return now


def _has_channelwise_decay(shape_in: Any) -> bool:
    """Whether a delta-rule operand list carries a per-CHANNEL decay (KDA)
    rather than a per-head scalar (Gated DeltaNet). The decay is the operand
    that follows q/k/v; KDA's has the key-dim axis, so it is 4-D."""
    if not (isinstance(shape_in, (list, tuple)) and len(shape_in) >= 4):
        return False
    g = shape_in[3]
    return isinstance(g, (list, tuple)) and len(g) == 4


def project_shape_for_op(op: str, shape_json: str) -> str:
    """Project `shape_json` onto the axes that identify op `op`'s measurement.

    The shared per-row shape_json carries ALL model dims (`d_model`, `d_ffn`,
    `n_layers`, `n_vocab`, …), but a gqa_attention kernel's timing depends only
    on (B, s_query, s_kv, n_heads, n_kv_heads, d_head) — the
    rest is model-specific noise that would stop the same physical kernel
    measured from two different models from matching. The axes are the
    SIGNATURE's, not the op name's: `linear` and `merged_column_parallel_linear`
    are one gemm and must project identically or they do not share a slot.

    Byte-stable (compact, sorted). Unchanged for an unmeasurable op, and for a
    row that does not carry every identifying axis — see `OpSignature.project`.
    """
    sig = get_signature(op_signature(op))
    # An op with no registered signature is not measurable, so nothing will
    # ever look it up by a projected key; leaving it whole is the honest answer.
    return sig.project(shape_json) if sig is not None else shape_json


def current_shape(op: str, shape_json: str) -> str:
    """A measured shape (canonical JSON text) in the current identity.

    A row measured before an identity change still has the old shape: the
    retired letters of the dim-symbol rename (`B, K, N, Sq`), or a grouped
    expert GEMM's shape without the routed `n_tokens` (schema v6). The
    measured-row migration (`spec_migrate.measured_row_migration`, the table
    the corpus CSVs were re-keyed from) renames the one and derives the
    other, so the row's physical shape and workload are the ones a current
    row of the same kernel problem has. A current shape is returned
    unchanged; a retired axis beside its current name with another value is
    refused, not resolved."""
    dims = json.loads(shape_json)
    migration = measured_row_migration(op_signature(op))
    renames, derived = migration["rename"], migration["derived"]
    if not any(axis in renames for axis in dims) and all(axis in dims for axis in derived):
        return shape_json
    out: dict = {}
    for axis, value in dims.items():
        # Renames chain (Sq → s_query → s_expert_rows): the letter renames
        # landed on names a later identity change moved again. Follow the
        # chain so the oldest spelling reaches the current one.
        name = axis
        seen = {name}
        while name in renames:
            name = renames[name]
            if name in seen:
                raise ValueError(
                    f"{op}: the measured-row rename of {axis!r} cycles "
                    f"through {name!r}")
            seen.add(name)
        if name in out and out[name] != value:
            raise ValueError(
                f"{op}: shape {shape_json} names {name!r} twice with different "
                f"values ({out[name]} and {value}) — a retired axis beside its "
                f"current name")
        out[name] = value
    for axis, factors in derived.items():
        if axis not in out and all(isinstance(out.get(f), int) for f in factors):
            out[axis] = math.prod(out[f] for f in factors)
    if out == dims:
        # Nothing to rename and no derivable axis: an old boundary shape
        # whose factors are not there to multiply stays the text it was.
        return shape_json
    return json.dumps(out, sort_keys=True, separators=(",", ":"))


def workload_coordinates(phase: str, shape_json: Any) -> Tuple[Optional[int], Optional[int]]:
    """The workload a measured row belongs to: `(batch, context)`.

    `batch` is the shape's `B`. `context` is the length the phase scales
    with — `s_query` in prefill (the tokens processed), `s_kv` in decode
    (the tokens attended); see `phase_binding`. A shape without that axis
    (decode rmsnorm / linear have no `s_kv`) is the same at every context:
    its context is `0`, which no real length takes (upload validation
    rejects non-positive dims). A shape without `B` — `grouped_gemm`'s
    per-expert token count scales with `B·s` but does not say which `B` —
    or a phase with no single context (`unified`) has no coordinates:
    `(None, None)`, not a guessed placement.
    """
    dims = json.loads(shape_json) if isinstance(shape_json, str) else (shape_json or {})
    context_axis = {"prefill": "s_query", "decode": "s_kv"}.get(phase)
    batch = dims.get("B")
    if context_axis is None or not isinstance(batch, int):
        return None, None
    context = dims.get(context_axis)
    return batch, (context if isinstance(context, int) else 0)


def workload_hash(
    *,
    shape_json: str,
    weight_dtype: Optional[str],
    activation_dtype: str,
    kv_dtype: Optional[str],
    output_dtype: str,
    metadata_dtype: Optional[str],
    weight_granularity: Optional[str],
    activation_granularity: Optional[str],
    output_format: Optional[str],
    kv_format: Optional[str],
) -> str:
    """SHA256 prefix over the workload shape — collapses `shape_json` +
    `dtype_env_json` into one opaque token.

    Used as one of four identity dimensions in `SlimBenchmarkKey`
    (the rest: spec_hash, hardware_hash, op_kernel_hash). Folding the
    workload facets (shape + all dtype slots) into one hash means
    contributors can't typo one slot's JSON without the hash diverging —
    the lookup either matches or misses cleanly.

    `shape_json` is the caller's own per-op identity shape, and it is
    required. It used to be optional, reconstructing `{B,s_ctx,dims}` from a
    batch and a sequence length when omitted — a slim key that silently
    disagreed with the wide and physical keys built from the same row, since
    those carry `s_query`/`s_kv`. dtype canonicalization is unaffected.

    A slot that does not apply to the row (`weight` / `kv` / `metadata`
    None) is omitted from the hashed environment, exactly as the row's
    `dtype_env_json` omits it. So is
    `weight_granularity`, and it is hashed: a 128x128-block FP8 GEMM and a
    per-channel one are two kernels on the same dtypes, and two rows apart
    only in granularity folded onto one slim key. `activation_granularity`
    is hashed for the same reason on the other operand — a per-token cast
    and a per-128-group one are two kernels between the same two dtypes —
    and `output_format` / `kv_format` for the cached record a launch
    writes or reads: two records over one element dtype are two launches.
    """
    shape = shape_json
    payload = {
        "shape": shape,
        "dtype_env": compute_per_row_dtype_env_json(
            weight_dtype=(None if weight_dtype is None
                          else normalize_dtype_name(weight_dtype)),
            activation_dtype=normalize_dtype_name(activation_dtype),
            kv_dtype=(None if kv_dtype is None
                      else normalize_dtype_name(kv_dtype)),
            output_dtype=normalize_dtype_name(output_dtype),
            metadata_dtype=(None if metadata_dtype is None
                            else normalize_dtype_name(metadata_dtype)),
            weight_granularity=weight_granularity,
            activation_granularity=activation_granularity,
            output_format=output_format,
            kv_format=kv_format,
        ),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def hardware_hash(
    *,
    gpu: str,
    framework: str,
    framework_version: str,
    phase: str,
    parallel: str,
) -> str:
    """SHA256 prefix over the deployment-context tuple. Collapses the
    five "how it's served" facets into one opaque token used as one
    of four `SlimBenchmarkKey` dimensions.
    """
    payload = {
        "gpu": gpu, "framework": framework,
        "framework_version": str(framework_version),
        "phase": phase, "parallel": parallel,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def op_kernel_hash(
    *,
    node_spec_id: str,
    op: str,
    kernel_impl: str,
) -> str:
    """SHA256 prefix over (node_spec_id, op, kernel_impl) — identifies
    WHICH leaf within the spec, WHAT primitive it computes, and WHICH
    kernel implementation runs. Op is derivable from
    (spec_hash, node_spec_id) given the spec on hand, but folding it
    in here means lookups don't require the spec to be loaded.
    """
    payload = {
        "node_spec_id": node_spec_id,
        "op": op,
        "kernel_impl": kernel_impl,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def scale_dtype_to_bpe(name: Optional[str], *, default: float = 4.0) -> float:
    """Bytes per element of a weight-scale dtype NAME (`ue8m0`, `fp32`), from
    `arch.dtypes`. An unknown name raises there; only an absent name takes
    `default`, which is the caller's stated assumption for a deployment that
    did not say."""
    if not name:
        return default
    return known_dtype_bytes(str(name).strip())
