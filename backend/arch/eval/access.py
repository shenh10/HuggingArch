"""The access rows: per-leaf and fused HBM bytes in the canonical sid
space, priced off the traffic layer.

A derived fact of the walk, like `traffic` it prices: the layout package's
per-rank cost reads it, so it sits in eval's core rather than in the
`projection` presentation package (which would put presentation on the
composition root's import path)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

from .dim import (as_extent, as_number, bound_or_folded, expr_of, floored,
                  ratio, same_law, value_of)
from . import traffic as traffic_mod
from .traffic import declared_input
from .formats import segments_of
from ..ops import DISPATCH_OP, is_dispatch_boundary, is_identity_source
from .cache import main_port
from .launch import FusedNode
from ..ops_base import declaration_is_gemm
from .layer import Layer
from backend.code_trace.memo import memo_property


def _int(value: Any) -> int:
    """A whole count, with an absent cost reading as zero.

    The rounding itself is `as_extent`'s — one convention, named where it is
    taken. What is local here is only that a `None` cost is a zero and not a
    missing number: an op that declares no parameters HAS none.
    """
    return 0 if value_of(value) is None else as_extent(value, rounding="floor")


@dataclass
class LeafAccess:
    """One leaf's HBM access bytes — the row shape the per-op roofline and
    the popovers consume.

    The cache a leaf reads where no edge carries it (`Traffic.kv_reads`) is
    part of `act_in_bytes_per_rank`; `kv_read_bytes_per_rank` names that
    share, so a consumer that wants the step's cache read sums it rather
    than re-deriving it.

    The weight term is the derivation's: `Representation.resident` per
    tensor, scheme metadata included. A serving cast is an intent on
    the walk, so this row does not reprice. `weight_dtype_bytes` on the
    row is the EFFECTIVE width (bytes over the elements streamed), display
    only.
    """

    full: int
    per_rank: int
    weight_bytes: int = 0
    act_in_bytes: int = 0
    act_out_bytes: int = 0
    metadata_bytes: int = 0
    #: The quantised copy of its own input a matmul's kernel needs and the
    #: edge does not carry — written once, read back once (§3.7 of the
    #: precision design). Part of `full` / `per_rank` like every other term.
    convert_bytes: int = 0
    convert_bytes_per_rank: int = 0
    #: The per-rank conversion as an extent, for reading the row at another
    #: batch (`act_in_extent_per_rank`'s sibling).
    convert_extent_per_rank: Any = 0
    #: What the conversion is, for display; empty when there is none.
    conversion: str = ""
    #: The leaf's own resident weight dtype and scale dtype (names), the
    #: dtype environment a measured row for THIS leaf was taken in.
    weight_dtype: Optional[str] = None
    scale_dtype: Optional[str] = None
    #: Which of that weight's elements share one scale — part of the same
    #: environment: a block-scaled kernel and a per-channel one are different
    #: measurements of one dtype.
    weight_granularity: Optional[str] = None
    #: The stored-tensor nodes whose matrices the leaf's kernel multiplies,
    #: grouped by the weight they are packed into (`weight_packs`) — what
    #: `Analysis.weight_shards` describes partition by partition.
    weight_packs: Tuple[Tuple[int, ...], ...] = ()
    #: Every stored tensor the leaf's kernel streams, one rank's shape of it
    #: and its resident dtype, and the input it arrives on where a dataflow
    #: edge threads it in (`StreamedWeight`): the weight operands, which the
    #: activation terms do not name — a weight is billed as a weight read.
    weights: Tuple["StreamedWeight", ...] = ()
    #: The quantised activation format (`ActivationFormat`) this leaf writes
    #: its primary output in (`Traffic.written_formats[""]`) — a grafted
    #: cast's; the absorbed-quantise branch that would state a fused
    #: launch's is unreached under the current graft order (see
    #: `Traffic.written_formats`). The rest of the same environment: what
    #: the launch writes, which elements share a scale, and the dtype of
    #: those scales. None where the leaf writes its output as the edge
    #: carries it. Only the PRIMARY port's: the key has one `output` slot,
    #: and a quantised secondary port is not keyed (`docs/known_limitations.md`
    #: §3.5).
    output_activation_format: Optional[Any] = None
    #: Whether the deployment CHOSE `act_dtype` (a compute pin, or a weight
    #: rule of its own) rather than reading it off the checkpoint. A card
    #: with no peak column for a chosen dtype refuses the estimate; one the
    #: checkpoint ships runs weight-only there (`pipeline.leaf_peak`).
    act_dtype_chosen: bool = False
    #: The granularity of the quantised input format the selected launch
    #: eats (`ActivationFormat.granularity`), if it eats one.
    activation_granularity: Optional[str] = None
    #: That input format's scale dtype, needed to key its scales when the
    #: leaf streams no quantised weight.
    activation_scale_dtype: Optional[str] = None
    #: Every MMA family in the selected launch's declared vocabulary
    #: (`ResolvedPrecision.mma_dtypes`), `act_dtype` first. A normal launch
    #: needs all of them; a promotion-ladder launch (the next field) permits
    #: the audited subset and pricing chooses its widest supported rung.
    mma_dtypes: Tuple[str, ...] = ()
    #: Whether the selected launch explicitly declared that subset semantics.
    promotion_ladder: bool = False
    #: Why ordered framework-launch candidates were skipped. The first is
    #: also the launch gap when no candidate was viable; `launch_gap` states
    #: that outcome directly.
    launch_gaps: Tuple[str, ...] = ()
    launch_gap: Optional[str] = None
    #: The dtype the input edge carries — what the matmul runs at in that
    #: weight-only fallback.
    edge_dtype: Optional[str] = None
    weight_bytes_per_rank: int = 0
    act_in_bytes_per_rank: int = 0
    act_out_bytes_per_rank: int = 0
    metadata_bytes_per_rank: int = 0
    #: Of `act_in_bytes_per_rank`, the bytes read from the cache.
    kv_read_bytes_per_rank: int = 0
    params: int = 0
    quant_metadata_params: int = 0
    #: Elements this row streams as weight — its own and any it borrows
    #: from an identity source. The denominator of `weight_dtype_bytes`.
    weight_elems: int = 0
    weight_dtype_bytes: float = 0.0
    act_dtype_bytes: float = 0.0
    #: Activation dtype **name** when the deployment stated one. Peak
    #: FLOPs dispatch reads this, not `act_dtype_bytes`.
    act_dtype: Optional[str] = None
    scale_dtype_bytes: float = 4.0
    act_in_dtype_bytes: float = 0.0
    act_out_dtype_bytes: float = 0.0
    #: The element dtype the leaf writes its primary output at, as the
    #: derivation priced it (`traffic._PortWrite.dtype`): the activation
    #: dtype, a launch's declared one, or — for the leaf that writes a
    #: cached tensor — that tensor's format dtype.
    output_dtype: Optional[str] = None
    #: The cached tensor that primary output IS (`eval.cache.landed_by`),
    #: None for an activation: the tensor this leaf writes in its format.
    output_cached: Any = None
    #: What the popover prints its params and shapes from: the per-rank
    #: cost and shape, and the substitution record the params were priced
    #: from. The text (`params_sym` … `shape_out_vals`) is rendered off it
    #: where it is read — at a point, the numbers a walk there prints.
    display: Optional["AccessDisplay"] = None
    #: The per-rank activation reads and writes as the traffic derivation
    #: built them — a `Dim` over the axes they are made of, whose value the
    #: two integer columns above truncate — so a row can read them at
    #: another batch.
    act_in_extent_per_rank: Any = 0
    act_out_extent_per_rank: Any = 0
    #: `kv_read_bytes_per_rank` as an extent, for the same reading.
    kv_read_extent_per_rank: Any = 0
    #: A lookup leaf's per-rank weight read before its cap
    #: (`traffic.TableRead`): the read grows with the lookups until it
    #: covers the table. None on every other leaf.
    table_read_per_rank: Any = None
    #: The per-rank activation read, term by term, as `traffic` priced it
    #: (`Traffic.in_edges`): `(input index, fan-out, width, elements)` per
    #: edge, `elements` None where the edge still reads its input's span over
    #: the fan-out (the display prints the shape then) and the number read
    #: where the cache law took the span over. Index -1 is an in-place
    #: update's read of its own output. The popover's activation-read term is
    #: built from these, so it states the bytes the row charges.
    in_terms_per_rank: Tuple["ReadTerm", ...] = ()
    #: The per-rank activation write, term by term (`Traffic.out_terms`):
    #: `(port, elements, width, spanned)` per written tensor, in the order of
    #: `shape_out_syms` (the main output, then each port). `elements` is read
    #: where `spanned` is false — a fused boundary that rewrote only a region
    #: of what it publishes (`Shape.written_elements`) — and the published
    #: shape is printed where it is true, exactly as on the read side. The
    #: popover's activation-write term is built from these, so it states the
    #: bytes the row charges.
    out_terms_per_rank: Tuple["WriteTerm", ...] = ()
    #: The inputs (`Shape.ins` positions) that are a MoE dispatch's routed
    #: buffer, `[n_expert_copies, M, d]`, and whether this leaf writes one
    #: (`_writes_routed`): what a launch pads when it rounds each expert's
    #: mean rows up (`routed.RoutedLaunch`). Empty / False outside a MoE
    #: block, and on the token side of its boundaries — a dispatch reads
    #: tokens and writes the buffer, a combine reads the buffer and writes
    #: tokens.
    routed_inputs: Tuple[int, ...] = ()
    routed_output: bool = False
    #: Of `per_rank`, the bytes on those routed buffers — their reads, their
    #: writes and the quantised copy of a routed operand — as an extent: the
    #: part of the leaf's activation traffic that moves with the routed rows
    #: and nothing else does.
    routed_extent_per_rank: Any = 0

    @property
    def output_format(self) -> Optional[str]:
        """How the cached tensor this leaf writes as its primary output
        (`output_cached`) is stored, by its registry name
        (`CachedTensor.format_name`, the one naming rule); None for an
        activation and for a cache stored plain."""
        if self.output_cached is None:
            return None
        return self.output_cached.format_name

    def input_dtypes(self, arity: int) -> Tuple[Optional[str], ...]:
        """Each input's element dtype by its `Shape.ins` position, as its
        read was billed (`in_terms_per_rank`) — or, for a stored tensor an
        edge threads in, its resident dtype (`weights`); None for an input
        nothing prices (a view moves nothing)."""
        read: Dict[int, Optional[str]] = {
            weight.input: weight.dtype for weight in self.weights
            if weight.input is not None}
        for term in self.in_terms_per_rank:
            read.setdefault(term.index, term.dtype)
        return tuple(read.get(i) for i in range(arity))

    def port_dtype(self, port: str) -> Optional[str]:
        """The element dtype written on `port` ("" the primary output), as
        the write was billed (`out_terms_per_rank`); None where nothing is
        written there."""
        return next((term.dtype for term in self.out_terms_per_rank
                     if term.port == port), None)

    @memo_property
    def _text(self) -> Dict[str, Any]:
        if self.display is None:
            return {"params_sym": None, "params_val": None,
                    "shape_in_syms": (), "shape_in_vals": (),
                    "shape_out_syms": (), "shape_out_vals": ()}
        return _formula_display(self.display.cost, self.display.shape,
                                self.display.bindings)

    # The popover's text (`_formula_display`): the params and each input
    # and written tensor's per-rank shape, symbolic and substituted.
    params_sym = property(lambda self: self._text["params_sym"])
    params_val = property(lambda self: self._text["params_val"])
    #: One entry per `Shape.ins`.
    shape_in_syms = property(lambda self: self._text["shape_in_syms"])
    shape_in_vals = property(lambda self: self._text["shape_in_vals"])
    #: The main output, then each port.
    shape_out_syms = property(lambda self: self._text["shape_out_syms"])
    shape_out_vals = property(lambda self: self._text["shape_out_vals"])


class AccessDisplay(NamedTuple):
    """What a leaf's popover text is printed from (`_formula_display`)."""
    cost: Any
    shape: Any
    bindings: Optional[Mapping[Any, Any]]


def _formula_display(cost: Any, shape: Any,
                     bindings: Optional[Mapping[Any, Any]] = None) -> Dict[str, Optional[str]]:
    """The formula-popover strings, from ONE side's cost and shape.

    Callers pass the PER-RANK side: the house convention has both popover
    lines (symbolic and substituted) in per-rank sharded shape — the
    symbolic line carries the per-rank symbols with no trailing `/tp` (the
    shard substitution already ate the divisor) and the substituted line
    carries each axis VALUE inline (`1*8*1*576`, the 8 being the sharded
    head count), which is what lets a reader see where the parallelism
    landed. The params symbol line is omitted when it collapsed to the
    exact same text as the value line — the renderer then falls back to its
    numeric form rather than printing one string twice.

    `bindings` is the substitution record the physical derivation priced the
    params figure from (`physical_cost_plan`: each stored tensor's
    global axis identity → its runtime extent), so the params term renders
    as the product it was computed with (`1536*1536`) rather than the folded
    element count. The bound form is kept only when it reproduces the params
    value exactly.
    """
    def _axes(dims: Any, *, symbolic: bool) -> Optional[str]:
        parts = []
        for axis in (dims or ()):
            if symbolic:
                parts.append(str(expr_of(axis)))
            else:
                parts.append(str(as_number(axis)))
        return "*".join(parts) if parts else None

    ins = tuple(shape.ins) if shape is not None else ()
    params_val = (
        bound_or_folded(cost.params, bindings, _int(cost.params))
        if cost is not None else None)
    params_sym = str(expr_of(cost.params)) if cost is not None else None
    if params_sym == params_val:
        params_sym = None
    in_syms = tuple(_axes(dims, symbolic=True) for dims in ins)
    in_vals = tuple(_axes(dims, symbolic=False) for dims in ins)
    written = (((shape.out, *tuple((shape.ports or {}).values())))
               if shape is not None else ())
    out_syms = tuple(_axes(dims, symbolic=True) for dims in written)
    out_vals = tuple(_axes(dims, symbolic=False) for dims in written)
    return {
        "params_sym": params_sym,
        "params_val": params_val if params_sym else None,
        "shape_in_syms": in_syms,
        "shape_in_vals": in_vals,
        "shape_out_syms": out_syms,
        "shape_out_vals": out_vals,
    }


def _wrap_expr(s: str) -> str:
    return f"({s})" if any(c in s for c in "+-") else s


def _fmt_dtype_bytes(b: float) -> str:
    if b <= 0:
        return "?"
    if b == int(b):
        return f"{int(b)}"
    return f"{b:g}"


def params_bytes_formula(
    cost: Any, bindings: Optional[Mapping[Any, Any]] = None, *,
    dt_w: float, meta_bytes: float = 0,
    meta_grids: Sequence[Tuple[Sequence[int], float]] = (),
) -> Dict[str, str]:
    """Leaf stored-bytes identity: `params * dt_w` [+ meta].

    Same granularity as `_access_bytes_formula_for_row`'s weight term
    (`d_model*(d_kv_lora + d_qk_rope)*dt_w`, values `7168*576*1`) — never
    the folded `bytes_full` placeholder.
    """
    params_sym = str(expr_of(cost.params)) if cost is not None else "params"
    params_val = (
        bound_or_folded(cost.params, bindings, _int(cost.params))
        if cost is not None else "0")
    dt = _fmt_dtype_bytes(dt_w)
    symbolic = f"{_wrap_expr(params_sym)}*dt_w"
    substituted = f"{_wrap_expr(params_val)}*{dt}"
    if meta_grids:
        # The format's scale tensors, each its grid times its element
        # width (`12*56*4`): the metadata as arithmetic, not a total.
        symbolic += " + scale_grid*scale_bytes"
        substituted += " + " + " + ".join(
            "*".join(str(int(d)) for d in shape) + f"*{_fmt_dtype_bytes(width)}"
            for shape, width in meta_grids)
    elif meta_bytes:
        meta = int(meta_bytes) if float(meta_bytes) == int(meta_bytes) else meta_bytes
        symbolic += " + meta"
        substituted += f" + {meta}"
    return {
        "symbolic": symbolic,
        "substituted": substituted,
        "note": "dt_w = weight dtype bytes",
    }


def access_layer_from_layers(
    analysis: Any,
    *,
    program: Any,
    global_traffic: Any,
    global_cost: Any,
    global_shapes: Any,
    local_traffic: Any,
    local_cost: Any,
    local_shapes: Any,
    scale_dtype_bytes: float,
    local_params_bindings: Optional[Mapping[int, Any]] = None,
) -> Layer[LeafAccess]:
    """Join complete traffic/cost/shape layers into per-leaf access rows."""
    global_rows = _leaf_traffic_rows(
        analysis,
        traffic_layer=global_traffic,
        cost_layer=global_cost,
        shape_layer=global_shapes,
        program=program,
    )
    rank_rows = _leaf_traffic_rows(
        analysis,
        traffic_layer=local_traffic,
        cost_layer=local_cost,
        shape_layer=local_shapes,
        program=program,
        params_bindings=local_params_bindings,
    )
    return Layer(program, {
        uid: _access_row(
            row,
            rank_rows.get(uid),
            scale_dtype_bytes=float(scale_dtype_bytes),
        )
        for uid, row in global_rows.items()
    })



class _TrafficRow(NamedTuple):
    """One leaf's traffic, priced final by the derivation.

    No repricing happens downstream: the precision profile's activation
    table decided each edge's width, so `derive` already priced each side at
    the width this row will report — index-declared sides at their own
    declared width, everything else at the producing leaf's.
    """

    act_in: Any
    act_out: Any
    #: The width the ACTIVATION portions were priced at (the role's) —
    #: display metadata, not a repricing input.
    act_width: float
    #: Stored-tensor reads as the derivation priced them — resident
    #: format, scheme metadata included. Final for owners and for alias
    #: readers alike (a tied LM head streams the embedding's table).
    weight: Any
    metadata: Any
    #: Of `act_in`, the cache read (`Traffic.kv_reads`).
    kv: Any
    cost: Any
    shape: Any
    node: Any
    #: Elements behind `weight`: own plus borrowed.
    weight_elems: Any = 0
    #: `LeafAccess.weight_packs`.
    weight_packs: Tuple[Tuple[int, ...], ...] = ()
    #: The dtype whose peak column the leaf's matmul runs at
    #: (`ResolvedPrecision.compute_dtype`).
    act_dtype: Optional[str] = None
    #: The substitution record the params figure was priced from (identity →
    #: runtime extent), for the popover's substituted line.
    params_bindings: Optional[Mapping[Any, Any]] = None
    #: `act_in` / `act_out` as the derivation built them (`Dim`s whose
    #: values the two floats are).
    act_in_extent: Any = 0
    act_out_extent: Any = 0
    #: `kv` as the derivation built it (like `act_in_extent`).
    kv_extent: Any = 0
    table: Any = None
    #: The input operand's own quantised copy, when the kernel needs one the
    #: edge does not carry (`Traffic.convert_bytes`).
    convert: Any = 0
    #: The same, as the derivation built it, so a row can be read at another
    #: batch (like `act_in_extent`).
    convert_extent: Any = 0
    #: What that conversion is, for the row that shows it
    #: (`ResolvedPrecision.conversion().label()`); empty when there is none.
    conversion: str = ""
    #: Whether the deployment chose `act_dtype`
    #: (`ResolvedPrecision.compute_dtype_is_chosen`).
    act_dtype_chosen: bool = False
    #: The quantised input granularity the selected launch eats, if any.
    activation_granularity: Optional[str] = None
    #: That input format's scale dtype.
    activation_scale_dtype: Optional[str] = None
    #: The MMA families the matmul needs a column for
    #: (`ResolvedPrecision.mma_dtypes`), its compute dtype first.
    mma_dtypes: Tuple[str, ...] = ()
    #: Whether those families are an audited promotion ladder rather than a
    #: set that must all enter the instruction together.
    promotion_ladder: bool = False
    #: Selector skips (`LeafAccess.launch_gaps`), carried unchanged.
    launch_gaps: Tuple[str, ...] = ()
    launch_gap: Optional[str] = None
    #: The dtype the leaf's input edge carries. The dtype its matmul runs at
    #: where the card has no MMA for the quantised one and the kernel falls
    #: back to weight-only.
    edge_dtype: Optional[str] = None
    #: The resident dtype of the widest stored tensor this leaf streams, and
    #: the dtype of its quant scales — the leaf's own, so a measured row is
    #: looked up in the dtype environment IT ran in rather than the step's
    #: dominant one.
    weight_dtype: Optional[str] = None
    scale_dtype: Optional[str] = None
    #: Which of that weight's elements share one scale
    #: (`TensorFormat.weight_granularity`), None when it stores none.
    weight_granularity: Optional[str] = None
    #: The quantised format the leaf writes its primary output in
    #: (`Traffic.written_formats`), None when it writes none.
    output_activation_format: Optional[Any] = None
    #: Every stored tensor the leaf streams (`StreamedWeight`).
    weights: Tuple["StreamedWeight", ...] = ()
    #: The derivation's per-edge activation terms (`Traffic.in_edges`) and
    #: per-port write terms (`Traffic.out_terms`).
    in_edges: Tuple[Any, ...] = ()
    out_terms: Tuple[Any, ...] = ()
    #: `LeafAccess.routed_inputs` / `routed_output` / `routed_extent_per_rank`
    #: of this derivation.
    routed_inputs: Tuple[int, ...] = ()
    routed_output: bool = False
    routed_extent: Any = 0


def _writes_routed(graph: Any, node: Any, port: str = "") -> bool:
    """Whether `node` writes a MoE dispatch's routed buffer on `port`.

    The dispatch (`permute`) writes it, and so does every leaf of its zone
    (`Program.in_zone`) up to the `combine`, which writes tokens back — but
    not a stored tensor's source in the zone (an expert bias,
    `is_identity_source`), which is a weight. A fused launch writes, on a
    port, what the member it publishes there writes; a component its output
    leaf's; a collective moves its source's tensor, so a route exchange
    carries the buffer it was handed."""
    if node is None:
        return False
    if isinstance(node, FusedNode):
        member = next((graph.by_uid(uid) for uid, published
                       in node.output_ports.items()
                       if main_port(published) == main_port(port)), None)
        return _writes_routed(graph, member)
    if node.provenance == "comm":
        refs = tuple(node.inputs_obj or ())
        return (len(refs) == 1
                and _writes_routed(graph, refs[0].src, refs[0].port or ""))
    if node.children:
        leaf, resolved = graph.output_leaf(node, port or "out")
        return _writes_routed(graph, leaf, main_port(resolved))
    if node.op == DISPATCH_OP:
        return True
    if is_dispatch_boundary(node.declaration) or is_identity_source(node.op):
        return False
    return graph.in_zone(node)


def _routed_terms(graph: Any, node: Any, t: Any
                  ) -> Tuple[Tuple[int, ...], bool, Any]:
    """`(routed inputs, writes routed, routed bytes)` of one leaf's traffic
    `t`: which of its read terms read a routed buffer, whether its writes
    land one, and the bytes of both, with the quantised copy of a routed
    operand (`Traffic.cast`) — every term off the derivation's own
    records."""
    inputs = tuple(dict.fromkeys(
        edge.index for edge in t.in_edges
        if edge.producer is not None and edge.index >= 0
        and _writes_routed(graph, graph.by_uid(edge.producer), edge.port)))
    writes = [term for term in t.out_terms
              if term.cached is None
              and _writes_routed(graph, node, term.port)]
    extent: Any = 0
    for edge in t.in_edges:
        if edge.index in inputs:
            extent = extent + edge.elements * edge.width
    for term in writes:
        extent = extent + term.elements * term.width
    if t.cast is not None and t.cast.index in inputs:
        extent = extent + t.convert_bytes
    return inputs, bool(writes), extent


def routed_launches(program: Any, access: Layer["LeafAccess"],
                    launches: Mapping[int, Any]) -> Layer[Any]:
    """Each leaf of `program` that reads or writes a routed buffer
    (`LeafAccess.routed_inputs` / `routed_output`), mapped to the launch it
    runs on the rank (`Analysis.routed`, `launches`, by leaf uid): the one
    record a launch's padded geometry is read off. A fused launch runs what
    its members run — they are one kernel on one buffer, so they must agree.
    A node grafted after the walk is in `launches` only if its grafter
    named the leaf whose tensor it carries."""
    out: Dict[int, Any] = {}
    for uid, ab in access.values.items():
        if not (ab.routed_inputs or ab.routed_output):
            continue
        node = program.by_uid(uid)
        if node.provenance == "comm":
            # A route exchange moves the rows a rank receives, whole
            # (`layout.materialize_flow.routed_flow_shape`); it is a
            # collective keyed by its payload, not a grouped launch padded
            # per expert.
            continue
        members = (node.origins if isinstance(node, FusedNode)
                   else (node.uid,))
        found = [launches[member] for member in members
                 if member in launches]
        if not found or any(other != found[0] for other in found[1:]):
            raise ValueError(
                f"{node.display_path}: it moves a routed buffer, and its "
                f"members run {len(found)} routed launches that are not "
                f"one ({found!r})")
        out[uid] = found[0]
    return Layer(program, out)


def _scale_dtype(widest: Optional[Any]) -> Optional[str]:
    """The dtype of the scales of the tensor `traffic.widest_format` picked, None
    when it stores none.

    What the scale HOLDS, not the slot it is shipped in: MXFP4's scale is a
    UE8M0 exponent and safetensors has no header spelling for one, so the
    scheme declares the slot (`dtype: U8`) and what fills it
    (`logical_dtype: F8_E8M0FNU`) as two facts — `StoredTensor.
    logical_dtype`. Reading the slot handed `U8` on as the scale's dtype,
    and U8 is a container with no corpus token at all, so every gpt-oss
    expert GEMM was declined for a dtype environment the corpus could not
    spell. The slot is still what the BYTES are counted at; it is not what
    the kernel computes with.

    The row's weight dtype, granularity and scale dtype describe ONE tensor.
    Taking the first scale met among a fused span's members instead paired
    one member's weight dtype with another's scale, and which member came
    first decided it. Within the one format, the scheme's declaration order
    decides: its first `kind: scale` sibling is the scale its granularity
    names (NVFP4's per-16 `.weight_scale`, not the per-tensor
    `.weight_scale_2`)."""
    if widest is None:
        return None
    return next((tensor.logical_dtype or tensor.dtype
                 for tensor in widest.metadata
                 if tensor.sibling_kind == "scale"), None)


def _members(node: Any) -> Tuple[int, ...]:
    """The walked leaves a row stands for: a fused node's origins, else
    itself — whose resident weight formats decide its matmul dtype."""
    return node.origins if isinstance(node, FusedNode) else (node.uid,)


def weight_packs(program: Any, node: Any, own: Any, borrowed: Any
                 ) -> Tuple[Tuple[int, ...], ...]:
    """The stored-tensor nodes whose matrices a leaf's kernel multiplies,
    grouped by the weight they are packed into.

    Asked of the nodes it streams (`traffic.streamed_sources`, the answer its
    bytes are billed from) and of the walked leaves it stands for: a member
    whose declaration multiplies its weight (`declaration_is_gemm`) is read
    as a matrix, and so is every other node it streams — a parameter
    threaded in along an edge, a tie's target — when some member does. A
    member that is no GEMM (a convolution's taps, a norm's scale) streams
    bytes but is no matrix a GEMM loader tiles. A plan that merges sibling
    projections (`FusedNode.merges_siblings`) packs its members' matrices
    into one weight; every other node is a weight of its own.
    """
    members = _members(node)
    gemm = tuple(member for member in members
                 if declaration_is_gemm(program.by_uid(member).declaration))
    if not gemm:
        return ()
    merged = isinstance(node, FusedNode) and node.merges_siblings
    packs = ((gemm,) if merged else tuple((member,) for member in gemm))
    others = tuple(dict.fromkeys(
        source for source in (*own, *borrowed) if source not in members))
    return packs + tuple((source,) for source in others)


def _leaf_traffic_rows(analysis: Any, *,
                       traffic_layer: Any = None,
                       cost_layer: Any = None,
                       shape_layer: Any = None,
                       program: Any = None,
                       params_bindings: Optional[Mapping[int, Any]] = None,
                       ) -> Dict[int, _TrafficRow]:
    layer = (
        traffic_mod.derive(analysis)
        if traffic_layer is None else traffic_layer
    )
    costs = analysis.cost if cost_layer is None else cost_layer
    shapes = analysis.shapes if shape_layer is None else shape_layer
    graph = analysis.program if program is None else program
    precision = analysis.resolved_precision
    walked_formats = analysis.formats.values
    # The same question traffic asks before it bills these bytes, asked of
    # the same function — so the dtype on the row describes the tensor the
    # bytes on the row came from (`traffic.streamed_sources`).
    read_aliases = graph.read_aliases or {}
    borrowed_by = traffic_mod._borrowed_sources(graph)
    held = _held_tensors(analysis)
    out: Dict[int, _TrafficRow] = {}
    for uid, t in layer.values.items():
        node = graph.by_uid(uid)
        if node is None or node.children:
            continue
        cost = costs.values.get(uid)
        # The weight half of the row's dtype environment describes the
        # tensors this leaf STREAMS, which is not always the ones it owns.
        own_srcs, borrowed_srcs = traffic_mod.streamed_sources(
            node, uid, members=_members(node), read_aliases=read_aliases,
            borrowed_by=borrowed_by)
        # The formats it multiplies by: its members' own and the tensors
        # they read through a tie, as traffic asks for its conversion.
        reps = segments_of(walked_formats,
                           *dict.fromkeys((*_members(node), *own_srcs)))
        streamed = traffic_mod.streamed_formats(
            walked_formats, own_srcs, borrowed_srcs)
        widest = traffic_mod.widest_format(streamed)
        input_format = precision.required_format(
            node.role, graph.stack_of(node), reps)
        out[uid] = _TrafficRow(
            act_dtype_chosen=precision.compute_dtype_is_chosen(
                node.role, graph.stack_of(node), reps),
            activation_granularity=(None if input_format is None
                                    else input_format.granularity),
            activation_scale_dtype=(None if input_format is None
                                    else input_format.scale_dtype),
            edge_dtype=precision.edge_dtype(node.role, graph.stack_of(node)),
            weight_dtype=(widest.logical_dtype if widest is not None else None),
            weight_granularity=(widest.weight_granularity
                                if widest is not None else None),
            scale_dtype=_scale_dtype(widest),
            output_activation_format=t.written_formats.get(""),
            convert=value_of(t.convert_bytes),
            convert_extent=t.convert_bytes,
            conversion=precision.conversion_label(
                node.role, graph.stack_of(node), reps,
            ) if not same_law(t.convert_bytes, 0) else "",
            act_in=value_of(t.act_reads) + value_of(t.kv_reads),
            act_out=value_of(t.act_writes),
            act_in_extent=t.act_reads + t.kv_reads,
            act_out_extent=t.act_writes,
            table=t.table,
            act_width=precision.edge_bytes(node.role, graph.stack_of(node)),
            act_dtype=precision.compute_dtype(
                node.role, graph.stack_of(node), reps,
                operand=t.operand_dtype),
            mma_dtypes=precision.mma_dtypes(
                node.role, graph.stack_of(node), reps,
                operand=t.operand_dtype),
            promotion_ladder=precision.launch_promotion_ladder(
                node.role, graph.stack_of(node), reps),
            launch_gaps=precision.launch_gaps(
                node.role, graph.stack_of(node), reps),
            launch_gap=precision.launch_gap(
                node.role, graph.stack_of(node), reps),
            weight=value_of(t.weight_reads) + value_of(t.borrowed_weight_reads),
            metadata=(value_of(t.metadata_reads)
                      + value_of(t.borrowed_metadata_reads)),
            kv=value_of(t.kv_reads),
            kv_extent=t.kv_reads,
            cost=cost,
            shape=shapes.values.get(uid),
            node=node,
            weight_elems=(value_of(t.weight_elems)
                          + value_of(t.borrowed_weight_elems)),
            params_bindings=(
                params_bindings.get(uid) if params_bindings else None),
            in_edges=tuple(t.in_edges),
            out_terms=tuple(t.out_terms),
            **dict(zip(("routed_inputs", "routed_output", "routed_extent"),
                       _routed_terms(graph, node, t), strict=True)),
            weight_packs=weight_packs(graph, node, own_srcs, borrowed_srcs),
            weights=streamed_weights(node, own_srcs, borrowed_srcs, held),
        )
    return out


class StreamedWeight(NamedTuple):
    """One stored tensor a leaf's kernel streams: its name, the shape one
    rank holds of it, its resident dtype (the widest of its layer segments,
    as `widest_format` describes a leaf), and the input it arrives on —
    None for a tensor the leaf owns (or reads through a tie)."""
    name: str
    shape: Tuple[Any, ...]
    dtype: Optional[str]
    input: Optional[int] = None


def _held_tensors(analysis: Any) -> Dict[int, Dict[int, list]]:
    """`Analysis.local_tensors()` by node and tensor index: each stored
    tensor's layer segments as this rank holds them."""
    held: Dict[int, Dict[int, list]] = {}
    for row in analysis.local_tensors():
        held.setdefault(row.uid, {}).setdefault(row.tensor_index, []).append(row)
    return held


def streamed_weights(node: Any, own: Mapping[int, Any],
                     borrowed: Sequence[int],
                     held: Mapping[int, Mapping[int, list]]
                     ) -> Tuple[StreamedWeight, ...]:
    """The stored tensors `node` streams (`traffic.streamed_sources`: the
    ones billed to it), each as `StreamedWeight`. A borrowed tensor names
    the input edge that threads it in — the edge `_borrowed_sources` found
    it on, by its position in the node's inputs."""
    arrives = {ref.src.uid: index
               for index, ref in enumerate(node.inputs_obj or ())
               if ref.src is not None and is_identity_source(ref.src.op)}
    out = []
    for source in (*own, *(b for b in borrowed if b not in own)):
        for segments in held.get(source, {}).values():
            widest = traffic_mod.widest_format(
                tuple(row.rep for row in segments if row.rep is not None))
            tensor = segments[0].tensor
            out.append(StreamedWeight(
                name=tensor.logical_name,
                shape=tuple(tensor.shape),
                dtype=widest.logical_dtype if widest is not None else None,
                input=None if source in own else arrives.get(source)))
    return tuple(out)


class ReadTerm(NamedTuple):
    """One input edge's activation read, as the derivation priced it
    (`traffic._EdgeRead`): which input, the fan-out its span was divided
    by, the width, the elements read, and whether those elements are still
    the input's span over the fan-out — at a point where the cache law took
    the span over they are the elements the edge read instead.

    `headless_store` says the fan-out expanded a tensor stored with no head
    axis at all, which its ratio no longer says at one head per rank.
    `dtype` is the element dtype `width` is the bytes of."""
    index: int
    fanout: Any
    width: Any
    elements: Any
    spanned: Any
    headless_store: bool = False
    dtype: Optional[str] = None


def _in_terms(row: _TrafficRow) -> Tuple[ReadTerm, ...]:
    """The row's activation-read terms, off the derivation's own edge
    records."""
    return tuple(ReadTerm(int(edge.index), value_of(edge.fanout),
                          value_of(edge.width), value_of(edge.elements),
                          edge.spanned, edge.headless_store, edge.dtype)
                 for edge in row.in_edges)


class WriteTerm(NamedTuple):
    """One written tensor's activation write, as the derivation priced it
    (`traffic._PortWrite`): the port it lands on ("" the main output), the
    elements written, the width, and whether those elements are still the
    whole of what that port publishes — at a fused boundary that rewrote
    only a region of its output they are the region. `dtype` is the element
    dtype written."""
    port: str
    elements: Any
    width: Any
    spanned: Any
    dtype: Optional[str] = None


def _out_terms(row: _TrafficRow) -> Tuple[WriteTerm, ...]:
    """The row's activation-write terms, off the derivation's own port
    records."""
    return tuple(WriteTerm(term.port, value_of(term.elements),
                           value_of(term.width), term.spanned, term.dtype)
                 for term in row.out_terms)


def _access_row(row: _TrafficRow, rank: Optional[_TrafficRow], *,
                scale_dtype_bytes: float) -> LeafAccess:
    def _terms(r: _TrafficRow) -> Tuple[Any, ...]:
        # Priced FINAL by the derivation — resident format unless the
        # deployment's cast applied. No repricing here, for weights just
        # as for activations. Whole bytes, a law where they grow with the
        # workload.
        return (floored(r.weight), floored(r.metadata),
                floored(r.act_in), floored(r.act_out), floored(r.convert))

    weight, meta, act_in, act_out, convert = _terms(row)
    extents = rank if rank is not None else row
    if rank is not None:
        r_weight, r_meta, r_act_in, r_act_out, r_convert = _terms(rank)
        display = AccessDisplay(rank.cost, rank.shape, rank.params_bindings)
    else:
        (r_weight, r_meta, r_act_in, r_act_out,
         r_convert) = weight, meta, act_in, act_out, convert
        display = AccessDisplay(row.cost, row.shape, row.params_bindings)
    params = int(row.cost.params) if row.cost is not None else 0
    qmeta = int(row.cost.quant_metadata_params or 0) \
        if row.cost is not None else 0
    # Per-side display dtypes: an index-declared side renders at its OWN
    # declared width — read off the op, the same declaration the pricing
    # used — and every other side at the role's activation width.
    act_dtype_bytes = row.act_width
    # The primary write at the width the derivation priced it — a cached
    # tensor's format, a launch's declared dtype — read off its own term.
    main = next((term for term in row.out_terms if term.port == ""), None)
    out_dt_display = (value_of(main.width) if main is not None
                      else act_dtype_bytes)
    # The primary read likewise, off the derivation's own first edge record
    # (`_in_terms`): what the operand edge was priced at — a collective's or
    # a dispatch's narrow copy, an index-declared input's own width. A row
    # the derivation records no read for (a view moves nothing) shows its
    # declared input, else the activation width.
    first = next((edge for edge in row.in_edges if edge.index >= 0), None)
    if first is not None:
        in_dt_display = value_of(first.width)
    else:
        in_override = declared_input(row.node)
        in_dt_display = (in_override.width if in_override is not None
                         else act_dtype_bytes)
    return LeafAccess(
        full=weight + meta + act_in + act_out + convert,
        per_rank=r_weight + r_meta + r_act_in + r_act_out + r_convert,
        weight_bytes=weight, act_in_bytes=act_in,
        act_out_bytes=act_out, metadata_bytes=meta,
        convert_bytes=convert, convert_bytes_per_rank=r_convert,
        convert_extent_per_rank=extents.convert_extent,
        conversion=row.conversion,
        weight_dtype=row.weight_dtype, scale_dtype=row.scale_dtype,
        weight_granularity=row.weight_granularity,
        weight_packs=row.weight_packs,
        weights=row.weights,
        output_activation_format=row.output_activation_format,
        act_dtype_chosen=row.act_dtype_chosen, edge_dtype=row.edge_dtype,
        activation_granularity=row.activation_granularity,
        activation_scale_dtype=row.activation_scale_dtype,
        mma_dtypes=row.mma_dtypes,
        promotion_ladder=row.promotion_ladder,
        launch_gaps=row.launch_gaps,
        launch_gap=row.launch_gap,
        weight_bytes_per_rank=r_weight,
        act_in_bytes_per_rank=r_act_in,
        act_out_bytes_per_rank=r_act_out,
        metadata_bytes_per_rank=r_meta,
        kv_read_bytes_per_rank=floored(
            (rank if rank is not None else row).kv),
        params=params, quant_metadata_params=qmeta,
        weight_elems=floored(row.weight_elems),
        weight_dtype_bytes=ratio(weight, row.weight_elems, empty=0.0),
        act_dtype_bytes=act_dtype_bytes,
        act_dtype=row.act_dtype,
        scale_dtype_bytes=scale_dtype_bytes,
        act_in_dtype_bytes=in_dt_display,
        act_out_dtype_bytes=out_dt_display,
        output_dtype=None if main is None else main.dtype,
        output_cached=None if main is None else main.cached,
        act_in_extent_per_rank=extents.act_in_extent,
        act_out_extent_per_rank=extents.act_out_extent,
        kv_read_extent_per_rank=extents.kv_extent,
        table_read_per_rank=extents.table,
        in_terms_per_rank=_in_terms(extents),
        out_terms_per_rank=_out_terms(extents),
        routed_inputs=extents.routed_inputs,
        routed_output=extents.routed_output,
        routed_extent_per_rank=extents.routed_extent,
        display=display,
    )
