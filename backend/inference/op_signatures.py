"""Op signatures — the math contract an op-class satisfies, and its axes.

A signature is the ANCHOR (not any library's kernel): it names what an op
computes and which dims/dtypes parameterize it. Many providers (FA3,
FlashInfer, FlashMLA …) claim to implement one signature, and the numeric
cross-check asserts they compute the same thing so their latencies compare.

Pure metadata: no torch, no GPU. It lives in `backend` rather than in
`kernel_bench` because BOTH sides need it and the dependency runs one way —
kernel_bench imports backend, never the reverse. The harness reads it to build
cells and pick provider pools; the cost model reads it to project a measured
row onto its physical identity.

TWO AXIS LISTS, ONE OBJECT
--------------------------
They used to be two module-level tables in two packages, keyed differently
(`kernel_bench.signatures.required_dims` by signature, `benchmark_identity.
_OP_RELEVANT_DIMS` by OP), and they disagreed about ten of the nineteen
signatures. `gemm` declared `n_experts_active, n_heads` — MoE axes, copied
from a neighbour — and neither `d_reduce` nor `d_out`, so the smoke test built
every GEMM cell with no GEMM dimension in it and nothing failed. The
projection table, meanwhile, had no gemm entry at all.

They are not one list, and merging them would be its own error:

  `cell_dims`     what you must bind to BUILD a legal cell. DERIVED from the
                  signature's shape contract — the declared canonical input
                  layout (`inputs`) plus the structural axes the math takes
                  that are no tensor's axis — never written a second time.
                  `test_cell_dims_are_what_the_reference_reads` runs each
                  reference (and, for an axis it does not read, each
                  provider's build) against a recording dims mapping and
                  asserts they consult exactly the derived axes, so a
                  contract that misstates the cell fails the moment it is
                  written. The paged-KV page (`PAGE_AXIS`) is the one axis
                  no builder need read: it is in the cell of every signature
                  that reads a paged cache because the deployment pages it.
  `variant_dims`  axes whose PRESENCE selects a geometry rather than sizing a
                  tensor, so a minimal cell must NOT bind them by default.
                  `mla_attention` reads `n_kv_heads` to choose between the
                  non-absorbed prefill form (per-head k/v) and the absorbed
                  decode one (a single shared latent); binding it
                  unconditionally builds the prefill geometry and the decode
                  kernel then asks for a latent that is not there.
                  `gqa_attention` reads `s_window` the same way: present is
                  sliding-window FA (`window_size`), absent is full causal.
                  Binding the window on every GQA cell would make a hybrid
                  model's full-attention layers share the SWA measurement.
                  Recorded by the reference like any other read, which is why
                  the two are distinguished by declaration and not by watching.
  `identity_dims` what IDENTIFIES a measurement physically — the envelope
                  projection. A subset of `cell_dims`: an axis needed to
                  construct an input is not necessarily one the cost depends
                  on. `moe_gate` is the case that forces the distinction — the
                  group structure (`n_expert_groups`, `n_groups_active`)
                  chooses WHICH experts win, not how much work it is, so it is
                  deliberately outside the identity and models that group
                  differently share one measurement.

Defaults to `cell_dims` plus the variant axes that select a geometry. The one
conditional exception is the page: it belongs to the static identity only when
the contract itself reads a paged cache. A write-only variant may still name it
as a variant axis (and `project` keeps it whenever that variant is present)
without making the signature's default dense cell speak a page it never touches.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import (Any, Dict, FrozenSet, List, Mapping,
                    Optional, Sequence, Tuple, Union)

from backend.arch.dims import get as dim_symbol
from backend.inference.fusion_plans import FUSION_SCHEMES


@dataclass(frozen=True)
class TensorLayout:
    """One canonical input tensor of a signature's measurement cell.

    `axes` has one entry per tensor dim; the entry language:

      "d_reduce"            a cell axis — the dim's extent is that axis's value
      ("n_heads", "d_rope") a tuple packs several axes into one physical dim
                            ([B·s_query, d_reduce] reads as
                            (("B", "s_query"), "d_reduce"));
                            ints multiply in (the gate|up ×2 of silu_and_mul
                            is (2, "d_ffn"))
      ("+", a, b, ...)      a sum of entries — the
                            (n_heads+2·n_kv_heads)·d_head of a packed qkv
      None                  a dim that is no cell axis at all (a cache sized
                            by an implementation constant), listed so the
                            tensor is declared and its rank is checked

    `cell_dims` derives from exactly the named axes here plus the
    signature's structural axes. `role` names the dtype_env slot the
    tensor is bound from.
    """
    name: str
    axes: Tuple[Any, ...]
    role: str = "activation"      # activation / weight / kv / metadata

    def extents(self, dims: Dict[str, int]) -> Tuple[Optional[int], ...]:
        """The physical shape this declaration implies under `dims`
        (None for a non-cell dim)."""
        return tuple(_extent(ax, dims) for ax in self.axes)

    def axis_names(self) -> Tuple[str, ...]:
        """Every named axis of this tensor (packed entries included), in
        declaration order."""
        return tuple(a for ax in self.axes for a in _named_axes(ax))

    def head_axes(self) -> Tuple[str, ...]:
        """The named axes of this tensor that `dims.yaml` classes
        `head_count` (packed entries included), in declaration order."""
        return tuple(a for ax in self.axes for a in _named_axes(ax)
                     if (sym := dim_symbol(a)) is not None
                     and sym.kind == "head_count")


def _extent(entry: Any, dims: Dict[str, int]) -> Optional[int]:
    if entry is None:
        return None
    if isinstance(entry, int):
        return entry
    if isinstance(entry, str):
        return int(dims[entry])
    if entry and entry[0] == "+":
        return sum(_extent(e, dims) for e in entry[1:])
    prod = 1
    for e in entry:
        v = _extent(e, dims)
        if v is None:
            return None
        prod *= v
    return prod


def _named_axes(entry: Any) -> Tuple[str, ...]:
    if isinstance(entry, str):
        return (entry,)
    if isinstance(entry, tuple):
        out: list = []
        for e in entry:
            if e == "+":
                continue
            out.extend(_named_axes(e))
        return tuple(out)
    return ()


def _dim_atoms(ax: Any) -> Optional[Tuple[Any, ...]]:
    """Atomic factors of one physical dim, or None if it is not permute/pack.

    A named axis, an int, or None is one atom. A pack `("n_heads", "d_rope")`
    is several. A sum (`("+", …)`) is a concat, not a pack — converting it
    would be a split, which is not unique from the layout alone.
    """
    if ax is None or isinstance(ax, (int, str)):
        return (ax,)
    if not isinstance(ax, tuple) or not ax:
        return None
    if ax[0] == "+":
        return None
    out: list = []
    for part in ax:
        if part is None or isinstance(part, (int, str)):
            out.append(part)
        else:
            return None
    return tuple(out)


def _atoms(axes: Sequence[Any]) -> Optional[Tuple[Any, ...]]:
    out: list = []
    for ax in axes:
        parts = _dim_atoms(ax)
        if parts is None:
            return None
        out.extend(parts)
    return tuple(out)


def layout_conversion(src: TensorLayout, dst: TensorLayout
                      ) -> Optional[Tuple[int, ...]]:
    """Permutation of `src`'s flattened atoms onto `dst`'s, or None.

    Unique when the two layouts name the same atoms once each — a permute,
    a pack, or both. Concat, a duplicate atom, or a different atom set is
    not a conversion of this kind; the kernel's `build` still owns those.
    """
    a = _atoms(src.axes)
    b = _atoms(dst.axes)
    if a is None or b is None or len(a) != len(b):
        return None
    if len(set(a)) != len(a) or set(a) != set(b):
        return None
    return tuple(a.index(x) for x in b)


def _flatten_grouping(grouping: Sequence[Any]) -> Optional[Tuple[int, ...]]:
    out: list = []
    for g in grouping:
        if isinstance(g, int):
            out.append(g)
            continue
        if not isinstance(g, tuple) or not g:
            return None
        inner = _flatten_grouping(g)
        if inner is None:
            return None
        out.extend(inner)
    return tuple(out)


def layout_from_grouping(src: TensorLayout, grouping: Sequence[Any]
                         ) -> Optional[TensorLayout]:
    """`src` with physical dims reordered/packed by index.

    `grouping` is a permutation of `range(rank)`, nested tuples pack
    adjacent axes: `(0, 2, 1, 3)` swaps, `((0, 1), 2, 3)` packs. Axis
    names come from `src` — a launch declaration does not restate them.
    """
    flat = _flatten_grouping(grouping)
    n = len(src.axes)
    if flat is None or sorted(flat) != list(range(n)):
        return None

    def take(g: Any) -> Any:
        if isinstance(g, int):
            return src.axes[g]
        return tuple(take(x) for x in g)

    return TensorLayout(src.name, tuple(take(g) for g in grouping),
                        role=src.role)


def unpacked_shape(axes: Sequence[Any], physical: Sequence[int],
                   dims: Mapping[str, int]) -> Tuple[int, ...]:
    """Physical extents expanded so each atom is its own dim.

    A packed physical dim splits when all but at most one factor are
    known (an int, or a name already in `dims`). Two unknowns is not a
    unique split — raise, do not guess.
    """
    if len(physical) != len(axes):
        raise ValueError(
            f"rank {len(physical)} does not match layout rank {len(axes)}")
    out: list = []
    for ax, ext in zip(axes, physical, strict=True):
        parts = _dim_atoms(ax)
        if parts is None:
            raise ValueError(f"cannot unpack {ax!r}")
        extent = int(ext)
        if len(parts) == 1:
            out.append(extent)
            continue
        sizes: list = []
        unknown = None
        known = 1
        for i, p in enumerate(parts):
            if isinstance(p, int):
                sizes.append(p)
                known *= p
            elif isinstance(p, str) and p in dims:
                v = int(dims[p])
                sizes.append(v)
                known *= v
            else:
                if unknown is not None:
                    raise ValueError(
                        f"cannot unpack {ax!r}: more than one unknown factor")
                unknown = i
                sizes.append(None)
        if unknown is None:
            if known != extent:
                raise ValueError(
                    f"packed {ax!r} wants product {known}, tensor has {extent}")
        else:
            if known <= 0 or extent % known:
                raise ValueError(
                    f"packed {ax!r} does not divide extent {extent}")
            sizes[unknown] = extent // known
        out.extend(int(s) for s in sizes)
    return tuple(out)


def packed_shape(axes: Sequence[Any], unpacked: Sequence[int]
                 ) -> Tuple[int, ...]:
    """Collapse adjacent unpacked extents back into `axes`' physical dims."""
    out: list = []
    i = 0
    for ax in axes:
        parts = _dim_atoms(ax)
        if parts is None:
            raise ValueError(f"cannot pack {ax!r}")
        prod = 1
        for _ in parts:
            prod *= int(unpacked[i])
            i += 1
        out.append(prod)
    if i != len(unpacked):
        raise ValueError(
            f"packed {i} of {len(unpacked)} unpacked extents")
    return tuple(out)


# Spec YAML may spell a contract axis by another registered name. Only
# aliases that mean the same quantity (MLA's rope half is `d_qk_rope`).
_SPEC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "d_rope": ("d_qk_rope",),
    "s_conv_kernel": ("s_ssm_conv_kernel",),
}


def spec_get(spec: Mapping[str, Any], axis: str) -> Any:
    """`spec[axis]`, or a declared alias. Missing is None, not a guess."""
    if axis in spec:
        return spec[axis]
    for alt in _SPEC_ALIASES.get(axis, ()):
        if alt in spec:
            return spec[alt]
    return None


def bind_layout(layout: TensorLayout, extents: Sequence[Any],
                dims: Dict[str, int],
                spec: Optional[Mapping[str, Any]] = None) -> bool:
    """Write `layout`'s axes into `dims` from one physical shape.

    The engine tensor may permute the contract, pack several axes into one
    dim (`("n_heads", "d_rope")`), or both. Packed dims invert when all
    but one factor are already bound (from another axis or `spec`).
    More than one unknown factor is incomplete — False, not a guess.
    Inconsistent with an already-bound axis is False.
    """
    spec = spec or {}
    if not isinstance(extents, (list, tuple)) or len(extents) != len(layout.axes):
        return False
    try:
        values = tuple(int(e) for e in extents)
    except (TypeError, ValueError):
        return False
    if any(v <= 0 for v in values):
        return False
    pending: list = []
    for ax, ext in zip(layout.axes, values, strict=True):
        if ax is None:
            continue
        if isinstance(ax, int):
            if ext != ax:
                return False
            continue
        if isinstance(ax, str):
            if ax in dims:
                if dims[ax] != ext:
                    return False
            else:
                dims[ax] = ext
            continue
        pending.append((ax, ext))
    return all(_bind_compound(ax, ext, dims, spec) for ax, ext in pending)


def _bind_compound(entry: Any, ext: int, dims: Dict[str, int],
                   spec: Mapping[str, Any]) -> bool:
    if not isinstance(entry, tuple) or not entry:
        return False
    if entry[0] == "+":
        total = 0
        for part in entry[1:]:
            v = _extent(part, dims)
            if v is None:
                return False
            total += v
        return total == ext
    prod = 1
    unknown: list = []
    for part in entry:
        if isinstance(part, int):
            prod *= part
            continue
        if isinstance(part, str):
            if part in dims:
                prod *= dims[part]
                continue
            unknown.append(part)
            continue
        v = _extent(part, dims)
        if v is None:
            return False
        prod *= v
    if not unknown:
        return prod == ext
    if prod <= 0 or ext % prod:
        return False
    rest = ext // prod
    # Spec may split a packed dim (d_rope in heads×width). Do not bind
    # every spec key: a model-level n_kv_heads sitting in spec_dims is
    # not this tensor's factor (that split produced d_rope=1). Fill from
    # the last packed name first (the width) until one unknown remains.
    if len(unknown) > 1:
        for name in reversed(list(unknown)):
            if len(unknown) == 1:
                break
            raw = spec_get(spec, name)
            if raw is None:
                continue
            try:
                val = int(raw)
            except (TypeError, ValueError):
                continue
            if val <= 0 or rest % val:
                continue
            dims[name] = val
            rest //= val
            unknown.remove(name)
    if len(unknown) != 1:
        return False
    dims[unknown[0]] = rest
    return True


def zip_engine_layouts(
    engines: Sequence[Tuple[int, TensorLayout]],
    operands: Sequence[Any],
    *,
    spec: Optional[Mapping[str, Any]] = None,
    shape_out: Any = None,
    output_axes: Tuple[str, ...] = (),
) -> Optional[Dict[str, int]]:
    """Bind one engine-layout declaration onto operand extents.

    Layouts are keyed by `(operand position, rank)`. Rank is an observable
    of the tensor, so the layout is selected, not searched. Two layouts
    of the same rank at one position are a registration error.
    """
    if not engines:
        return None
    spec = spec or {}
    if any(isinstance(pos, str) for pos, _layout in engines):
        raise ValueError(
            "engine layouts keyed by port are read through the row's "
            "operand ports (`OpSignature.at_ports`), not zipped as declared")
    # One operand can fill two ports (an MLA latent that is K and V): both
    # layouts of its rank bind it, and must agree.
    by_key: Dict[Tuple[int, int], List[TensorLayout]] = {}
    positions: list = []
    for pos, layout in engines:
        key = (pos, len(layout.axes))
        by_key.setdefault(key, []).append(layout)
        if pos not in positions:
            positions.append(pos)
    dims: Dict[str, int] = {}
    bound = False
    for pos in positions:
        if pos >= len(operands):
            # A layout for an operand the engine did not emit — skip,
            # do not demand a tensor that is not there.
            continue
        t = operands[pos]
        if not isinstance(t, (list, tuple)):
            return None
        layouts = by_key.get((pos, len(t)))
        if layouts is None:
            return None
        for layout in layouts:
            if not bind_layout(layout, t, dims, spec):
                return None
        bound = True
    if not bound:
        return None
    for axis in output_axes:
        w = output_width(shape_out)
        if axis in dims:
            if w and w > 0 and dims[axis] != w:
                return None
            continue
        if not w or w <= 0:
            return None
        dims[axis] = int(w)
    if not dims or any(v <= 0 for v in dims.values()):
        return None
    return dims


def output_width(shape_out: Any) -> Optional[int]:
    """The op's output feature width `d_out` from the engine's output shape.

    `shape_out` is either a single tensor `[.., d_out]` (o_proj, down_proj) or a
    list of per-port tensors `[[..,w0],[..,w1],..]` (a fused node whose
    absorbed siblings emit separate outputs — qkv → q/k/v, gate_up → gate/up).
    For the multi-port case the concatenated GEMM width is the sum of the
    ports. One channel, both layouts — no reverse-engineering from flops.
    """
    if not isinstance(shape_out, (list, tuple)) or not shape_out:
        return None
    try:
        if isinstance(shape_out[0], (list, tuple)):     # per-port list
            return sum(int(p[-1]) for p in shape_out if p)
        return int(shape_out[-1])                        # single tensor
    except (TypeError, ValueError, IndexError):
        return None


def _factor_run(entry: Any, wanted: set) -> Tuple[str, ...]:
    """The `wanted` names an axis entry multiplies together — a name, or a
    product tuple of entries — or () when there are none, or when one of them
    is a term of a sum (`("+", …)`), whose extent is not a product."""
    if isinstance(entry, str):
        return (entry,) if entry in wanted else ()
    if not isinstance(entry, tuple):
        return ()
    if "+" in entry:
        return ()
    out: list = []
    for e in entry:
        if isinstance(e, tuple) and "+" in e and wanted.intersection(_named_axes(e)):
            return ()
        out.extend(_factor_run(e, wanted))
    return tuple(out)


#: The paged-KV page, as an axis of the cell of every signature that reads a
#: paged cache (`OpSignature.reads_paged_cache`). Its value is the
#: deployment's (`FrameworkBinding.kv_block_size`, or the request's
#: override), handed to the projection beside the spec's dims
#: (`RowIdentity`); it is not a `dims.yaml` symbol, because no model binds
#: it.
PAGE_AXIS = "kv_block_size"

#: How far a collective's group physically reaches, when it reaches past
#: one chassis (`comm_cost.comm_reach`: `scaleup` / `scaleout`). A VARIANT
#: axis on every collective signature, so it identifies by presence and a
#: group inside one host — the default, and every row measured so far —
#: keys exactly as it did before the axis existed. Like `PAGE_AXIS` it is
#: no `dims.yaml` symbol: the deployment's placement states it, not the
#: model. Its one derivation is `RowIdentity`, off the card's domain sizes
#: and the group's own world and stride.
COMM_DOMAIN_AXIS = "n_comm_domain"


def _contract_axes(inputs: Tuple[TensorLayout, ...],
                   structural: Tuple[str, ...]) -> Tuple[str, ...]:
    """The cell's axes, derived from the declared layout: first appearance
    across the input tensors, then the structural axes (parameters of the
    math that are no tensor's axis — a token total, a top-k width)."""
    out: list = []
    for lay in inputs:
        for ax in lay.axes:
            for a in _named_axes(ax):
                if a not in out:
                    out.append(a)
    for a in structural:
        if a not in out:
            out.append(a)
    return tuple(out)


@dataclass(frozen=True)
class OpSignature:
    name: str
    # The measurement contract: the canonical input layout, and the axes the
    # math takes that are no tensor's axis. This is the ground truth —
    # `cell_dims` below is its projection, and `refs.make_inputs` is checked
    # against it (not the other way round).
    inputs: Tuple[TensorLayout, ...] = ()
    structural_axes: Tuple[str, ...] = ()
    # dtype_env slots that materially affect the kernel (drive dtype).
    dtype_slots: Tuple[str, ...] = ("activation",)
    # Human description of the math contract (what the reference computes).
    doc: str = ""
    # Axes whose presence selects a geometry — see the module docstring.
    variant_dims: Tuple[str, ...] = ()
    # Canonical layouts used when every `variant_dims` axis is bound.
    # Empty = the default `inputs` is the only geometry.
    variant_inputs: Tuple[TensorLayout, ...] = ()
    # Engine-order layouts for that geometry, same contract as `engine_inputs`.
    variant_engine_inputs: Tuple[Tuple[Union[int, str], TensorLayout], ...] = ()
    # Axes that identify a measurement. () means "declared unprojected" — see
    # `project`. None (the default) means cell + variant, except that the page
    # joins only a contract that reads a cache. A write-only variant page is
    # still projected when that variant is present; it just cannot make the
    # default geometry claim a page it never touches.
    _identity_dims: Optional[Tuple[str, ...]] = None
    #: The engine ops measured under this signature — the kernel side's own
    #: declaration of what it measures (the ops side's `OpSpec.gemm` says
    #: what the math is; a per-head linear is a GEMM in math and
    #: `per_head_gemm` in measurement). An op is bound by at most one
    #: signature; `op_signature` is the projection.
    binds_ops: Tuple[str, ...] = ()
    #: How a row's engine shapes become the measurement cell: `"contract"`
    #: zips the engine layouts declared above, which is what almost every
    #: signature does; the others name a projection whose geometry is not a
    #: layout permutation at all (the DSA indexer's selection mask, the
    #: packed-qkv rope's folded head width, the MoE gate's reconstruction, a
    #: collective's published payload).
    shape_projection: str = "contract"
    #: The sibling signature a row of this op is measured under when its
    #: decay operand is per-channel (rank 4) rather than per-head — Gated
    #: DeltaNet's `kda`. Two Triton kernels with the same dims; without the
    #: split they would share one physical key.
    decay_rank_variant: str = ""
    # Engine-side operand layouts for identity projection: (operand, layout
    # of that tensor as the ENGINE emits it). The operand is a position for
    # a primitive op, and for a fused launch of two or more input ports the
    # PORT it fills (`NaivePattern.input_ports`, the plan's `io_mapping`):
    # which tensor stands where in a launch's operand list depends on
    # which ports that graph fills — an optional mask present or not, one
    # latent that is both K and V — so a row's operands are read by the
    # ports it names (`at_ports`), never by a position written here. Same axis language as
    # `inputs` — a name, a packed tuple, a sum — so a permute or a pack is
    # a declaration, not a second projector. An operand may have several
    # layouts only when they differ in rank; zip looks up `(position, rank)`
    # and does not search. Registration rejects two layouts of the same
    # rank at the same position.
    engine_inputs: Tuple[Tuple[Union[int, str], TensorLayout], ...] = ()
    #: For a port-keyed signature a primitive op is also measured under: the
    #: port each of the primitive's operands is, by position. A primitive
    #: has no ports, and `rope` (one operand, a query or a key) and
    #: `indexer_logits` (q, k, the head weights) share their cells with a
    #: plan's launch; the join of the two vocabularies is stated once, here,
    #: and checked against the primitive's arity and the plans' required
    #: ports (`_check_operand_keys`).
    primitive_ports: Tuple[str, ...] = ()
    # The same declaration for the engine's OUTPUT tensors, for a kernel
    # whose geometry is on the side it writes: `permute` reads token-ordered
    # rows and writes the `[n_expert_copies, M, d_model]` buffer the launch
    # is, so its cell is zipped off the output.
    engine_outputs: Tuple[Tuple[int, TensorLayout], ...] = ()
    # Axes read from the output's last-dim width (a GEMM's d_out, MLA's
    # d_v_head). At most one today.
    output_axes: Tuple[str, ...] = ()
    #: Axes that cap the keys one call attends: it reads min(context, these)
    #: — a sparse attention gathers `n_topk_keys` of `s_kv` — so the context
    #: axis alone is not what a call reads.
    attends_at_most: Tuple[str, ...] = ()
    #: The input position whose activation this launch's output is an index
    #: gather of — one of its elements per element written, in another order
    #: (`moe_permute`'s routed rows, `gather`'s per-token pick). None for
    #: every launch that computes its output.
    #:
    #: What it decides on this side: whether a QUANTISED operand's record
    #: travels whole. A launch that writes its input's elements unchanged
    #: moves the record — the elements and the scale each row carries — so a
    #: cell naming one format on both dtype slots is one record moved
    #: (`kernel_bench.quant.ActivationCast.moved_by`); a launch that
    #: computes cannot be measured that way, however equal its slots, and
    #: stays a refusal until a provider declares what it computes. The same
    #: fact is the arch op's `OpSpec.gather_source`, and
    #: `tests/backend/inference/test_gathering_signatures.py` holds the two
    #: to each other — this side may not look an op up by name
    #: (`tests/meta/test_eval_boundary.py`).
    gathers_from: Optional[int] = None
    #: Function constants this signature's kernels take beside their
    #: operands, and which each is: `(kernel_param, constant symbol)`, naming
    #: a `constant`-kind symbol an absorbed op declares (`OpSpec.consts`)
    #: and the walk bound at that node onto the row (`consts`). The rename
    #: is here because the kernel's argument name is the kernel's business,
    #: while the FACT that this model clamps at 10.0 is the graph's, and is
    #: read, not re-derived.
    #:
    #: These are not axes — they change what the kernel COMPUTES, not how
    #: big it is, so they are `Invocation.kernel_params` rather than
    #: `dims`. They are declared because a reference that picks its own
    #: value cross-checks a function no model runs: Kimi-K3's SiTU gate is
    #: `beta·tanh(x/beta)·sigmoid(x)` at beta 4 with a linear beta of 25,
    #: and at the reference's own beta of 1 the kernel and the oracle
    #: agreed with each other about a third function. vLLM reads them the
    #: same way and refuses the launch when one is missing ("SITU requires
    #: activation_situ_beta from FusedMoEConfig"), so a cell whose model
    #: binds none is a cell this kernel does not run — the provider
    #: declines it (`KernelSignature.requires_params`) instead of
    #: defaulting.
    config_params: Tuple[Tuple[str, str], ...] = ()

    def params_from_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        """The kernel parameters this signature's cells carry, read off a
        per-op row: the constants of `config_params`, off its `consts`, and
        the axes of its cell that are no part of its identity (`cell_dims`
        outside `identity_dims` — `moe_gate`'s expert grouping, which picks
        WHICH experts win, so models that group differently share one
        measurement), off the node's own dims (`node_dims`, as its frame
        bound them). A value the model does not bind is absent, so the
        provider that needs it declines the cell."""
        consts = row.get("consts") or {}
        out: Dict[str, Any] = {param: float(consts[symbol])
                               for param, symbol in self.config_params
                               if consts.get(symbol) is not None}
        bound = row["node_dims"]
        if bound is not None:
            identity = set(self.identity_dims)
            out.update({axis: bound[axis] for axis in self.cell_dims
                        if axis not in identity
                        and bound.get(axis) is not None})
        return out

    @property
    def has_contract(self) -> bool:
        return bool(self.inputs or self.structural_axes)

    @property
    def reads_paged_cache(self) -> bool:
        """Whether this signature's kernels read a cache the deployment
        pages — so the page size is part of what was measured.

        Derived from the contract, not listed: a kernel reads the cache iff
        it declares an operand with `role="kv"`, which is the same
        declaration the dtype environment's `kv` slot comes from. A list of
        paged signature names beside it would be the second statement of
        one fact, and it drifted the moment it existed — `kernel_bench.plan`
        kept three attention names and left out DSA's lightning indexer,
        which reads the same paged K.
        """
        return any(t.role == "kv"
                   for t in (self.inputs + self.variant_inputs))

    def keyed_by_port(self) -> bool:
        """Whether this signature names its operands by port (a fused
        launch of several input ports) rather than by position."""
        return any(isinstance(key, str) for key, _layout in
                   (*self.engine_inputs, *self.variant_engine_inputs))

    def at_ports(self, ports: Optional[Sequence[Sequence[str]]], *,
                 op: Optional[str] = None) -> "OpSignature":
        """This signature with its operands placed where one row's are:
        each port-keyed layout at the position of the operand that fills
        that port (`ports`, the row's `operand_ports`: per operand, the
        ports it fills in plan order). An operand that fills several ports
        (one latent that is a span's K and its V) is read as the first: its
        extents are that port's tensor, and a later port's layout, which
        reads the same storage as another tensor, is not placed; nor is a
        layout on a port no operand fills (an optional port the graph does
        not have: Qwen3.8's indexer weighs no heads). A port the launch's
        plan (`op`) requires that no operand fills — not even as the second
        port of a shared tensor — is no launch of the plan, refused
        (`NaivePattern.required_ports`): a per-head sparse span that lost
        its V would otherwise bind its geometry on Q and K. Position-keyed
        signatures are returned as they are.

        An operand list that names no ports is refused, loudly — a job
        written before launches named their operands would be read by where
        its tensors happen to stand. A primitive op's row names its operands
        by `primitive_ports` (`benchmark_identity.operand_ports_of`)."""
        if not self.keyed_by_port():
            return self
        if ports is None:
            raise ValueError(
                f"signature {self.name!r} reads its operands by port, and "
                f"this row or job names none (`operand_ports`); a launch's "
                f"operand order is no contract — regenerate it from the "
                f"current plan")
        where: Dict[str, int] = {}
        filled: set = set()
        for position, names in enumerate(ports):
            names = [str(name) for name in names]
            if names:
                where.setdefault(names[0], position)
            filled.update(names)
        scheme = FUSION_SCHEMES.get(op) if op is not None else None
        if scheme is not None:
            missing = sorted(scheme.naive_pattern.required_ports() - filled)
            if missing:
                raise ValueError(
                    f"{op!r}: its launch names no operand for the port(s) "
                    f"{missing} its plan requires (ports named: "
                    f"{[list(p) for p in ports]})")

        def placed(engines):
            return tuple((key if not isinstance(key, str) else where[key],
                          layout) for key, layout in engines
                         if not isinstance(key, str) or key in where)

        return dataclasses.replace(
            self, engine_inputs=placed(self.engine_inputs),
            variant_engine_inputs=placed(self.variant_engine_inputs))

    def head_axis(self, position: int, rank: int) -> Optional[int]:
        """Where operand `position`, a tensor of `rank` dims, carries its head
        axis in the engine's layout — the axis `dims.yaml` classes
        `head_count` — as the signature's engine layout for that operand
        declares it (selected by position and rank, as `zip_engine_layouts`
        selects it), or None when it declares none.

        The axis a stride-0 head fan-out repeats: a K/V the engine
        broadcasts to every query head (`head_broadcast`) is stored once per
        KV head, and the kernel takes the stored count — GQA shares a head
        across a group of queries, absorbed MLA one latent across all of
        them. Derived from the layouts, not listed per signature. Two layouts
        of one operand that disagree on it are a declaration error."""
        if self.keyed_by_port():
            raise ValueError(
                f"signature {self.name!r} names its operands by port; ask "
                f"the signature placed at a row's ports (`at_ports`)")
        found = set()
        for pos, layout in (*self.engine_inputs, *self.variant_engine_inputs):
            if pos != position or len(layout.axes) != rank:
                continue
            for index, axis in enumerate(layout.axes):
                sym = dim_symbol(axis) if isinstance(axis, str) else None
                if sym is not None and sym.kind == "head_count":
                    found.add(index)
        if len(found) > 1:
            raise ValueError(
                f"signature {self.name!r}: the rank-{rank} layouts of operand "
                f"{position} put its head axis at {sorted(found)}")
        return next(iter(found)) if found else None

    @property
    def cell_dims(self) -> Tuple[str, ...]:
        """Axes an invocation must bind — derived from the declared contract,
        never written twice.

        A signature that reads a paged cache (`reads_paged_cache`) also has
        the page (`PAGE_AXIS`) among its structural axes: a paged kernel's
        time depends on the page the deployment allocates the cache in, so
        two pages are two cells. Derived from the same `role="kv"` operand
        rather than written into each catalog entry, so a new attention
        signature cannot leave it out."""
        return self._axes_of(self.inputs)

    def _axes_of(self, inputs: Tuple[TensorLayout, ...]) -> Tuple[str, ...]:
        paged = ((PAGE_AXIS,) if self.reads_paged_cache
                 and PAGE_AXIS not in self.structural_axes else ())
        return _contract_axes(inputs, self.structural_axes + paged)

    def cell_dims_for(self, dims: Any) -> Tuple[str, ...]:
        """The axes a cell of the geometry `dims` selects must bind
        (`layouts_for`): `cell_dims` for the default geometry, the variant
        layout's own axes for the variant one.

        Derived the same way from the layout that geometry declares. Every
        variant before `fused_q_kv_rmsnorm`'s per-head form only ADDED axes
        to the default (MLA's per-head K/V, GQA's narrow V), so `cell_dims`
        happened to be a subset of either; the per-head norms share only the
        token axes with the MLA a-norm rows, and asking a per-head cell for a
        `d_q_lora` it has no tensor of is a cell that can never be complete.
        """
        if not self.binds_variant(dims):
            return self.cell_dims
        return self._axes_of(self.variant_inputs)

    @property
    def identity_dims(self) -> Tuple[str, ...]:
        if self._identity_dims is not None:
            return self._identity_dims
        variants = tuple(
            axis for axis in self.variant_dims
            if axis != PAGE_AXIS or self.reads_paged_cache
        )
        return self.cell_dims + variants

    def token_runs(self, factors: Sequence[str], context: Sequence[str] = (),
                   batch: Sequence[str] = ()) -> Tuple[Tuple[str, ...], ...]:
        """How the contract lays out the token `factors` (dims.yaml
        `call_tokens: factor`): each tensor axis that folds factors into one
        (a GEMM's rows `(B, s_query)` — one dimension whose extent is their
        product), as those factors; then each factor the contract keeps on an
        axis of its own (attention's `B`, then `s_query`), alone; in layout
        order. The first run a shape has is what one call's work grows with:
        a folded run is a single dimension, a separate axis is not
        interchangeable with the others. An axis on which a factor is a term
        of a sum (`("+", …)`) is no run: its extent is not their product.

        A contract with a `context` axis (the keys attended, `s_kv`) runs
        first along the `batch` axes it has times that axis — the KV tokens
        one call reads (decode attention: batch × context); its `s_query`
        stays a separate run, so prefill and decode keep apart. A contract
        whose attended keys are capped (`attends_at_most`) reads a minimum,
        which no run is: it has no context run."""
        named = set(self.cell_dims)
        ctx = tuple(a for a in batch if a in named) + tuple(a for a in context if a in named)
        lead = [ctx] if any(a in named for a in context) and not self.attends_at_most else []
        wanted = set(factors)
        folded: list = []
        alone: list = []
        for layout in self.inputs:
            for axis in layout.axes:
                names = _factor_run(axis, wanted)
                runs = folded if isinstance(axis, tuple) else alone
                if names and names not in runs:
                    runs.append(names)
        return tuple(lead + folded + [r for r in alone if not any(r[0] in f for f in folded)])

    def layout_axis_names(self) -> FrozenSet[str]:
        """Axes that appear on a declared tensor (canonical or engine).

        A variant dim in this set is selected by the operand list (MLA's
        `n_kv_heads`). One that is not — GQA's `s_window` — is a mask
        parameter: it never shows up on Q/K/V, so identity fills it from
        spec only when the caller says this node is windowed.
        """
        names: set = set()
        for lay in (*self.inputs, *self.variant_inputs):
            for ax in lay.axes:
                names.update(_named_axes(ax))
        for engines in (self.engine_inputs, self.variant_engine_inputs,
                        self.engine_outputs):
            for _pos, lay in engines:
                for ax in lay.axes:
                    names.update(_named_axes(ax))
        names.update(self.structural_axes)
        names.update(self.output_axes)
        return frozenset(names)

    @property
    def variant_selectors(self) -> Tuple[str, ...]:
        """The variant axes that SELECT the variant geometry: those its own
        operand list names.

        Derived, because `variant_dims` is not one list of selectors — it is
        every axis whose presence is a fact of the leaf rather than a size,
        and the two kinds are told apart by whether a variant operand spans
        them (the distinction `layout_axis_names` already draws). GQA carries
        three: `s_window` and `n_attn_sinks` are softmax parameters that
        change no operand's shape, while `d_v_head` is the width of V. Asking
        for all three at once meant the asymmetric-V layout could never be
        selected — no cell binds a window AND a sink AND a narrow V — so
        every MiMo attention cell was compared against a 192-wide V and
        refused.
        """
        if not self.variant_inputs:
            return ()
        named: set = set()
        for lay in self.variant_inputs:
            for ax in lay.axes:
                named.update(_named_axes(ax))
        return tuple(a for a in self.variant_dims if a in named)

    def binds_variant(self, dims: Any) -> bool:
        """True when `dims` selects the variant geometry.

        Presence, not value: `n_kv_heads` on an MLA cell is the non-absorbed
        prefill form; an absorbed cell does not carry the key at all.
        Layouts only swap when `variant_inputs` is declared — a mask
        parameter like `s_window` does not change Q/K/V rank, and is not a
        selector (`variant_selectors`).
        """
        selectors = self.variant_selectors
        if not selectors:
            return False
        try:
            return all(axis in dims for axis in selectors)
        except TypeError:
            return False

    def layouts_for(self, dims: Any) -> Tuple[TensorLayout, ...]:
        return self.variant_inputs if self.binds_variant(dims) else self.inputs

    def project(self, shape_json: str) -> str:
        """`shape_json` reduced to the identifying axes, or unchanged.

        Unchanged when the shape does not carry all of them, which is not a
        courtesy — it is the difference between a cross-model key and a
        catastrophe. The shape that arrives here not speaking the axis names
        is the whole-step MODEL one, `{B, s_ctx, d_model, n_heads, n_kv_heads,
        d_head, d_ffn, …}`, which shares exactly `B` with what `gemm`
        declares. Keeping "whichever axes are present" would key it on its
        batch size alone and hand back the first GEMM measurement of any
        width. Kept whole, it matches nothing but itself.

        There is one vocabulary and it is the contract's. `s_ctx` — the
        whole request length, from before the phase split — used to be
        accepted here as a second spelling of `s_query`, back when the
        estimate side keyed off the model-wide shape. It bought nothing
        even then: the projected dict keeps the key it FOUND, never renaming
        (an `s_ctx` estimate matching an `s_query` row would be matching
        numbers that mean something else in decode), so an aliased shape
        projected to a key no measured row can carry. It only made the
        fabrication look well-formed. Both sides now derive the key
        from `shape_in` through the signature and refuse when they cannot.
        """
        if self._identity_dims is not None:
            required, optional = self._identity_dims, ()
        else:
            required, optional = self.cell_dims, self.variant_dims
        if not required and not optional:
            return shape_json
        # A shape cell that does not parse is a broken row, not a shape that
        # "matches nothing but itself": it raises.
        dims = json.loads(shape_json)
        if self._identity_dims is None and isinstance(dims, dict):
            # The axes of the geometry this cell is (`cell_dims_for`).
            required = self.cell_dims_for(dims)
        out = {}
        for axis in required:
            if axis not in dims:
                return shape_json
            out[axis] = dims[axis]
        # Variant axes identify by PRESENCE. Requiring them would make every
        # dense GQA shape fail open (and keep model-level noise in the key);
        # omitting a bound `s_window` would let a hybrid's full-attn leaf
        # share the sliding-window measurement.
        for axis in optional:
            if axis not in out and axis in dims:
                out[axis] = dims[axis]
        return json.dumps(out, separators=(",", ":"), sort_keys=True)


_SIGNATURES: Dict[str, OpSignature] = {}
_OP_TO_SIGNATURE: Dict[str, str] = {}


def _check_operand_keys(sig: OpSignature) -> None:
    """A signature names a fused launch's operands by the ports its plan
    declares (`NaivePattern.input_ports`). Refused:

      * a port a bound plan does not declare, or one on a single-port
        launch — its one operand is position 0, which a job of any age
        names alike;
      * a position on a launch of two or more ports, whose operand order is
        no contract (a launch lists the tensors that fill its ports, in
        port order, and which are there is the graph's);
      * a port on a collective's launch, read by the one flow it moves
        (`NaivePattern.collective`, which the launch carries as
        `FusedNode.is_comm_op`);
      * `primitive_ports` naming a port a bound plan does not declare, or
        on a signature read by position — a primitive op has no ports, so a
        port-keyed signature it is measured under states which port each of
        its operands is, once, here."""
    keys = [key for key, _layout in (*sig.engine_inputs,
                                     *sig.variant_engine_inputs)]
    named = sorted({key for key in keys if isinstance(key, str)})
    positional = sorted({key for key in keys if not isinstance(key, str)})
    if sig.primitive_ports and not named:
        raise ValueError(
            f"{sig.name}: `primitive_ports` names the operands of a "
            f"signature read by position")
    unported = [op for op in sig.binds_ops if op not in FUSION_SCHEMES]
    if named and unported and not sig.primitive_ports:
        raise ValueError(
            f"{sig.name}: reads its operands by port and measures the "
            f"primitive op(s) {unported}, which have none; state each of "
            f"their operands' port (`primitive_ports`)")
    for op in sig.binds_ops:
        scheme = FUSION_SCHEMES.get(op)
        if scheme is None:
            continue
        pattern = scheme.naive_pattern
        ports_of_op = pattern.input_ports()
        if pattern.collective() or len(ports_of_op) == 1:
            if named:
                raise ValueError(
                    f"{sig.name}: {op!r} has one operand — "
                    f"{'the flow its collective moves' if pattern.collective() else 'its one port'}"
                    f" — which is position 0 at any age; not ports {named}")
            continue
        if positional:
            raise ValueError(
                f"{sig.name}: operand(s) {positional} of {op!r} named by "
                f"position, but {op!r} has the input ports "
                f"{list(ports_of_op)}; name the port")
        for key in (*named, *sig.primitive_ports):
            if key not in ports_of_op:
                raise ValueError(
                    f"{sig.name}: port {key!r} is no input port of {op!r} "
                    f"(its ports: {list(ports_of_op)})")


def register_signature(sig: OpSignature) -> None:
    # The engine-side layout is a permutation of the contract, or it is a
    # second layout — the failure mode this declaration exists to end.
    # A variant geometry may name `variant_dims` (they are not in cell_dims
    # so a minimal cell does not bind them).
    cell = set(sig.cell_dims)
    allowed = cell | set(sig.variant_dims)
    for engines in (sig.engine_inputs, sig.variant_engine_inputs,
                    sig.engine_outputs):
        seen_rank: Dict[Tuple[int, int], str] = {}
        for pos, layout in engines:
            for ax in layout.axes:
                for name in _named_axes(ax):
                    if name not in allowed:
                        raise ValueError(
                            f"{sig.name}: engine layout entries must be "
                            f"contract axes, got {name!r} in {ax!r}")
            key = (pos, len(layout.axes))
            if key in seen_rank:
                raise ValueError(
                    f"{sig.name}: operand {pos} has two engine layouts "
                    f"of rank {key[1]} — rank must select a unique packing")
            seen_rank[key] = layout.name
    _check_operand_keys(sig)
    for ax in sig.output_axes:
        if ax not in cell:
            raise ValueError(
                f"{sig.name}: output axis {ax!r} is not a contract axis")
    for ax in sig.attends_at_most:
        if ax not in cell:
            raise ValueError(
                f"{sig.name}: attends_at_most axis {ax!r} is not a contract axis")
    if sig.variant_engine_inputs and not sig.variant_inputs:
        raise ValueError(
            f"{sig.name}: variant_engine_inputs requires variant_inputs")
    # A variant layout no axis selects is a geometry nothing can ever reach:
    # the cells it describes are compared against the DEFAULT layout and
    # refused. GQA's narrow V spent the corpus in that state, one level up —
    # the layout was missing rather than unselectable.
    if sig.variant_inputs and not sig.variant_selectors:
        raise ValueError(
            f"{sig.name}: variant_inputs spans none of variant_dims "
            f"{sig.variant_dims}, so no cell selects it")
    # One name, registered once. The ownership check below catches two
    # signatures claiming one op, but `owner == sig.name` lets a SECOND
    # registration of the same name through — and the last one wins
    # silently: `situ_and_mul` was declared twice, and the stale entry
    # (no beta in its doc, no params required) replaced the live one.
    if sig.name in _SIGNATURES:
        raise ValueError(
            f"signature {sig.name!r} is registered twice. The second "
            f"registration replaces the first with no word said; a "
            f"signature that needs to change is edited where it is "
            f"declared.")
    for op in sig.binds_ops:
        owner = _OP_TO_SIGNATURE.get(op)
        if owner is not None and owner != sig.name:
            raise ValueError(
                f"op {op!r} is bound by two signatures: {owner!r} and "
                f"{sig.name!r}")
        _OP_TO_SIGNATURE[op] = sig.name
    _SIGNATURES[sig.name] = sig


def op_bindings() -> Dict[str, str]:
    """Every bound op → its signature, as the catalog declared it."""
    return dict(_OP_TO_SIGNATURE)


def op_signature(op: str) -> str:
    """The signature an engine op is measured under — the projection of the
    catalog's `binds_ops`. An op no signature binds is its own name (the
    plan then reports it as dropped: no registered signature)."""
    return _OP_TO_SIGNATURE.get(op, op)


def get_signature(name: str) -> Optional[OpSignature]:
    return _SIGNATURES.get(name)


def all_signatures() -> Tuple[OpSignature, ...]:
    return tuple(_SIGNATURES.values())


from . import op_signatures_catalog as _catalog  # noqa: F401
