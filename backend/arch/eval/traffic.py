"""`Traffic` — bytes moved to and from HBM, derived rather than propagated.

Why derived
-----------
Every other layer answers a question about a node in isolation and is built as
the walk descends. Traffic is a pure function of layers that already exist —
shapes, storage, formats — plus the deployment's precision, so it is computed from
whatever graph you are holding rather than carried across changes to it.

That is not a convenience. It is why a fusion's saving needs no arithmetic: the
fused node's traffic is computed from ITS boundary, which no longer includes the
intermediates, and the saving is simply `Σ pre − Σ post` — no "subtract the
eliminated buffer" logic exists to get wrong. A traffic value that had to be
`rebase`d across a rewrite would need a combine that is neither a sum (flash
attention exists precisely to not materialise the score matrix) nor a
projection, and there is no such thing.

The four terms, and the fifth
-----------------------------
    weight_reads      the parameters, at their storage format
    metadata_reads    the scales that come with them
    act_reads         what the node's inputs carry
    act_writes        what its output carries
    kv_reads          the cache, where an edge could not carry it

The fifth is separate because the cache is not in the dataflow: no node
produces it. A decode step's K/V leaves emit one token while attention reads
`s_kv` of them, so the cache read can never be an edge weight on the graph.

It is a COMPLEMENT, not a second copy. An edge whose span its source
materialised in full — a prefill attending what it just wrote — carries every
cache byte in `act_reads`, and lands nothing in `kv_reads`; an edge that spans
more than its source produced is reading the cache, so the cache's law
replaces its price entirely and lands there instead. Either way each byte is
counted once, which is why the split needs no phase case and why both terms go
into the per-op rows downstream.

A recurrent state is the one cache no edge carries in either direction: its
carrier reads it where the step continues a sequence (`s_past` > 0 — decode,
a prefix-cached prefill) and writes it back every step, a fresh prefill
included. The read lands in `kv_reads`; the write is a term of the carrier's
`act_writes` at the state's own format (`_state_writes`).

What makes it derivable anyway is that the cache IS declared, and `eval.cache`
is where that declaration is assembled: which nodes a component keeps, at what
dtype, and under which positions law each of them grows. This module asks it
for the read side at the length the walk ran at, and does not do the
arithmetic itself.

That separation is not filing. Doing it here would mean summing the cached
nodes into one per-token figure and multiplying by `s_kv` — correct for a dense
cache, and the same dense answer for a sliding window, a compressed latent and
a recurrent state, none of which grow that way. The unit has to be the tensor,
and once it is, the layer holding two tensors under two different laws stops
being a case.

Dtypes
------
Weights get theirs from `Representation.resident` — what the deployment holds,
which defaults to what the checkpoint saved and differs when a precision
profile converts it. An edge's width is the producing leaf's, as the profile's
activation table decides it — the model's own dtype when no rule selects it —
and a cached tensor's is its own format (`eval.cache`), on both sides: the
leaf that writes it (`cache.landed_by`) writes the format's record — data,
unquantised tail and the scales riding each position — and every same-step
reader of that edge, through views, joins and the launch that absorbed the
writer, reads it at that size (`declared_output`), which is the size the
cache law bills a later step's read of it at. Every one of them is
`Analysis.resolved_precision`'s answer; nothing here holds a width.

A leaf that multiplies a stored weight held wider than its activation —
DeepSeek-V4's F32 hyper-connection `hc_*_fn`, an F32 router gate — COMPUTES
in the weight's dtype (`weight_promotion`): the type promotion of a mixed
BF16 x F32 product, which decides the peak column its FLOPs are timed at
and the MMA family its cell is keyed on. The weight is the one the leaf
streams (`streamed_formats`), at its resident format, so the checkpoint's
declaration decides it and a framework that upcasts a BF16 weight at load
does not.

It does NOT widen what the leaf writes. The width of a tensor is its
edge's declaration; a promotion is about the arithmetic, and the kernels
narrow in registers as they store (both frameworks' iHC post allocate
their output at the sub-block operand's dtype). Writing the product wide
would send fp32 down the graph after it, and the graph would then need a
cast node to bring it back that no kernel launches.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from typing import (Any, Dict, FrozenSet, Iterable, Iterator, List,
                    Mapping, Optional, Sequence, Tuple)

import sympy as sp

from ..dims import get as dim_symbol
from ..dtypes import dtype_bytes, is_unquantized
from ..ops import (OP_REGISTRY, ShapeKind, attention_kernel,
                   DISPATCH_OP, declaration_is_gemm, gather_source,
                   is_identity_source, passes_a_lone_operand,
                   reads_input_shape_only, reads_own_output, repeats_input)
from . import cache
from .cache import (Edges,
                    ContextBranch,
                    ContextCoverage,
                    context_sources as _context_sources,
                    covered_context_source as _covered_source,
                    is_view as _is_view,
                    materialized as _materialized)
from .formats import segments_of
from .tensor_rows import pair_stored
from .dim import (algebra, Dim, ceil_div, exact, exact_div, expr_of, minimum,
                  ratio, same_law, value_of, names_of, where)
from .launch import FusedNode, score_matrix_owner
from .layer import Layer, Rewrite
from .refusal import SpecRefusal
from .values import Extent, product
from .node import Identified


@dataclass(frozen=True)
class TableRead:
    """A lookup leaf's weight read before its cap: the table it gathers
    from (bytes and scales, as the leaf holds it), the lookups one step
    makes and the table's global row count. The read is
    `min(table, table x lookups / rows)` (`_table_read_fractions`)."""
    weight: Extent
    metadata: Extent
    lookups: Extent
    rows: Extent
    #: The table is a parameter the leaf reads off an edge
    #: (`OpSpec.lookup` with a table port), billed as a borrowed read.
    borrowed: bool = False

    def scaled(self, count: Extent) -> "TableRead":
        """`count` copies, each gathering its own rows. A borrowed table is
        its source's one copy, whatever the reader's count."""
        if count == 1 or self.borrowed:
            return self
        return replace(self, weight=self.weight * count,
                       metadata=self.metadata * count)


@dataclass(frozen=True)
class Traffic:
    """HBM bytes one node moves in one step."""

    weight_reads: Extent = 0
    metadata_reads: Extent = 0
    act_reads: Extent = 0
    act_writes: Extent = 0
    #: The activation read no edge price could express: cache-resident bytes
    #: whose size follows the cache's own growth law rather than the reader's
    #: shape. Split out from `act_reads` because it is derived differently, not
    #: because it is a different kind of byte — `total` adds it and so does the
    #: per-op projection.
    #
    # It is the COMPLEMENT of the clip in `derive`, not a second copy of the
    # cache: an edge whose span its source materialised in full carries every
    # cache byte in `act_reads` already, and lands zero here. A decode step
    # lands the whole cache here; a fresh prefill lands none of it; a chunked
    # prefill lands the prefix it did not just write. Read it as "cache reads
    # the shapes do not show", never as "all cache reads".
    kv_reads: Extent = 0
    #: The quantised copy of its own input operand a matmul's kernel needs
    #: and the edge does not carry: written by the quantise and read back by
    #: the GEMM (`ResolvedPrecision.conversion`, the design's §3.7). Zero on
    #: every leaf whose kernel eats the edge as it arrives, and zero again
    #: once a framework's fused launch writes the edge in that format
    #: (`absorbs_consumer_quant`) — one format change, one price, whichever
    #: node pays it.
    convert_bytes: Extent = 0
    #: Weight bytes this node reads out of an IDENTITY SOURCE — a node whose
    #: output is the tensor it stores (`arity() == 0 and has_weight() and not
    #: is_buffer`: `parameter_tensor`, `parameter_vector`). Such a node moves
    #: nothing itself; whoever consumes its output is the kernel that streams
    #: the bytes, which is what "the read bills whoever moves the bytes" means.
    #
    # Held apart from `weight_reads`: the bytes are the source's tensor,
    # streamed by this reader (gpt-oss's packed expert bias, read by the GEMM
    # it is threaded into). The row adds the two (`access._leaf_traffic_rows`).
    borrowed_weight_reads: Extent = 0
    #: The borrowed tensors' quant-metadata (scale) bytes, held apart from
    #: `metadata_reads` the same way.
    borrowed_metadata_reads: Extent = 0
    #: Elements behind `weight_reads` / `borrowed_weight_reads`, kept beside
    #: them so a consumer wanting the EFFECTIVE width divides bytes by what
    #: was actually streamed. It used to divide by the row's `params`, which
    #: equals this only while a row streams exactly the parameters it owns —
    #: false both ways once a weight is billed to the kernel that moves it:
    #: the parameter row owns params and streams nothing; the reader streams
    #: elements it does not own.
    weight_elems: Extent = 0
    borrowed_weight_elems: Extent = 0
    #: A lookup leaf's `weight_reads` / `metadata_reads` before the cap, so
    #: the read can be stated at another step's lookups. None on every other
    #: node, and on a sum of nodes.
    table: Optional[TableRead] = None
    #: The terms `act_reads` is the sum of, one per input edge read (and one
    #: for an in-place update's own output): `elements * width` each. What a
    #: display of this node's activation read prints — the same fan-out,
    #: clip and width the price used, not a second reading of the shapes.
    #: Empty on a sum of nodes.
    in_edges: Tuple["_EdgeRead", ...] = ()
    #: The terms `act_writes` is the sum of, one per tensor written
    #: (`_PortWrite`), at the width the price used — a launch that writes its
    #: output quantised is billed, and displayed, at the narrow width. What a
    #: display of this node's activation write prints, the same way `in_edges`
    #: serves the read. Empty on a sum of nodes.
    out_terms: Tuple["_PortWrite", ...] = ()
    #: The dtype this leaf's kernel computes on, as it was billed: for an
    #: elementwise op its operands' promoted type (`promoted_operand`), for
    #: any other its first input edge's (`in_edges[0]`, the operand a
    #: matmul's own format applies to), promoted with the weight a matmul
    #: multiplies where that is wider (`weight_promotion`). None where it
    #: reads no data operand —
    #: none at all, or only the indices its op declares (`OpSpec.
    #: input_dtype`) — and on a sum of nodes. The measurement key's
    #: activation slot reads this (`ResolvedPrecision.compute_dtype`).
    operand_dtype: Optional[str] = None
    #: The standalone format change this leaf's input needs, when no
    #: producer absorbed it (`CastLaunch`) — one launch, which
    #: `convert_bytes` prices and nothing else could take apart. None where
    #: the kernel eats the edge as it arrives, where a framework's fused
    #: launch wrote it narrow, and on a sum of nodes.
    cast: Optional["CastLaunch"] = None
    #: The quantised activation format (`ActivationFormat`) this leaf WRITES
    #: on each output port that carries one, by port (`""` the primary):
    #: `Written.format` of the walk that prices the write (`declared_output`).
    #: Two writers state one — a grafted cast (`inference.materialize_casts`)
    #: declares the format it was grafted for on its node
    #: (`SpecIRNode.output_format`), and a fused launch that took a cast in
    #: writes what that member declares, on the port it publishes the member
    #: at (`_written_formats`) — and an element copy of that record (a MoE
    #: dispatch, the exchange after it) hands it on. A measured row's dtype
    #: environment reads the primary port's (`LeafAccess.output_format`).
    #: Empty everywhere else, and on a sum of nodes.
    written_formats: Mapping[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> Extent:
        return (self.weight_reads + self.metadata_reads + self.act_reads
                + self.act_writes + self.kv_reads + self.convert_bytes
                + self.borrowed_weight_reads + self.borrowed_metadata_reads)

    def __add__(self, other: "Traffic") -> "Traffic":
        return Traffic(
            weight_reads=self.weight_reads + other.weight_reads,
            metadata_reads=self.metadata_reads + other.metadata_reads,
            act_reads=self.act_reads + other.act_reads,
            act_writes=self.act_writes + other.act_writes,
            kv_reads=self.kv_reads + other.kv_reads,
            convert_bytes=self.convert_bytes + other.convert_bytes,
            borrowed_weight_reads=(self.borrowed_weight_reads
                                   + other.borrowed_weight_reads),
            borrowed_metadata_reads=(self.borrowed_metadata_reads
                                     + other.borrowed_metadata_reads),
            weight_elems=self.weight_elems + other.weight_elems,
            borrowed_weight_elems=(self.borrowed_weight_elems
                                   + other.borrowed_weight_elems),
        )

    @staticmethod
    def sum(parts: Iterable["Traffic"]) -> "Traffic":
        out = Traffic()
        for part in parts:
            out = out + part
        return out

    def scaled(self, count: Extent) -> "Traffic":
        """`count` copies of this node's traffic.

        Every term alike, which is where this differs from `Cost` — that one
        scales params by the stored count and flops by the active one, because a
        MoE stores more experts than it runs. Bytes have no such split: a step
        moves what it moves. Two combines over the same roots, which is why
        `Program.contributions` hands out the count rather than applying it.
        """
        if count == 1:
            return self
        return Traffic(
            weight_reads=self.weight_reads * count,
            metadata_reads=self.metadata_reads * count,
            act_reads=self.act_reads * count,
            act_writes=self.act_writes * count,
            kv_reads=self.kv_reads * count,
            convert_bytes=self.convert_bytes * count,
            borrowed_weight_reads=self.borrowed_weight_reads * count,
            borrowed_metadata_reads=self.borrowed_metadata_reads * count,
            weight_elems=self.weight_elems * count,
            borrowed_weight_elems=self.borrowed_weight_elems * count,
            table=None if self.table is None else self.table.scaled(count),
        )

    def intensity(self, flops: Extent) -> Extent:
        """FLOPs per byte. Which side of the roofline a node sits on, and so
        which of two implementations of the same function is faster — the
        question neither FLOPs nor bytes answers alone. Zero where the node
        moves nothing."""
        return ratio(flops, self.total, empty=0)


@dataclass(frozen=True)
class TrafficGeometry:
    """Physical layers supplied by a complete production projection.

    Workload laws, graph identity and attribution remain on ``Analysis``.
    This bundle replaces only the physical inputs traffic reads, so a caller
    cannot mix a local shape layer with global storage or cache laws.
    """

    shapes: Layer[Any]
    storage: Layer[Any]
    formats: Layer[Any]
    cache: Layer[Any]
    #: The same program's shapes unsplit — `shapes` before any deployment
    #: cut them (`shapes` itself when it is unsplit). Whether a reader reads
    #: the cache is asked here (`_spans_in_whole_frame`): per rank, a reader
    #: and the tensor it reads can be cut differently. Replacing `shapes`'
    #: program replaces it too.
    whole_shapes: Layer[Any]
    #: Whole-cache view uid -> how many times fewer positions that view's
    #: invocation owns than the tensor landed on ITS edge in this program.
    #: The divisor is a ratio of two shape facts, not a property of the view
    #: alone: before Ulysses' all-gather both frames are sequence-cut, while
    #: after it the landed tensor is whole and the invocation is cut.  It is
    #: a fact of `shapes`' program: replacing `shapes` replaces it too.
    cache_view_frame_divisor: Mapping[int, Extent] = field(
        default_factory=dict)
    # Runtime operand replicas can differ from checkpoint alias ownership.
    read_aliases: Optional[Mapping[int, Mapping[int, Any]]] = None

    @staticmethod
    def _axis_identity(axis: Any, *, tracked_only: bool = False) -> Optional[str]:
        if tracked_only and not isinstance(axis, Dim):
            return None
        return sp.srepr(algebra(expr_of(axis)))

    @classmethod
    def _axes_identity(
        cls,
        axes: Any,
        *,
        tracked_only: bool = False,
    ) -> Optional[tuple]:
        if axes is None:
            return None
        return tuple(
            cls._axis_identity(axis, tracked_only=tracked_only)
            for axis in axes
        )

    @classmethod
    def _shape_structure(cls, shape: Any) -> tuple:
        return (
            tuple(
                cls._axes_identity(axes, tracked_only=True)
                for axes in shape.ins
            ),
            cls._axes_identity(shape.out, tracked_only=True),
            tuple(
                (name, cls._axes_identity(axes, tracked_only=True))
                for name, axes in sorted(shape.ports.items())
            ),
        )

    @classmethod
    def _stored_tensor_identity(cls, tensor: Any) -> tuple:
        return (
            tensor.module,
            tensor.suffix,
            tensor.logical_suffix,
            tensor.kind,
            tensor.dtype,
            tensor.axis_refs,
            cls._axes_identity(tensor.shape),
        )

    @classmethod
    def _format_identity(cls, reps: Any) -> tuple:
        """One stored tensor's formats, over every layer segment it has.

        A tensor whose node stores it one way across its layers is one
        segment, which is almost every tensor; a layer-scoped
        `quant_context.overrides` row makes it several, and then all of them
        are the identity. Asking for "the" format would raise here, and
        comparing one of them would let a projection drop a segment and still
        look unchanged."""
        def _tensor_format(fmt: Any) -> Any:
            if fmt is None:
                return None
            return (
                replace(fmt, metadata=()),
                tuple(cls._stored_tensor_identity(row) for row in fmt.metadata),
            )

        return tuple(
            (segment.layers, _tensor_format(segment.saved),
             _tensor_format(segment.intent))
            for segment in reps)

    @staticmethod
    def _cache_identity(entry: Any) -> tuple:
        def _law(value: Any) -> str:
            return sp.srepr(algebra(value))

        return (
            entry.component,
            entry.label,
            entry.variant,
            entry.persists,
            tuple(
                (
                    tensor.node,
                    tensor.name,
                    tensor.dtype,
                    _law(tensor.stored),
                    _law(tensor.read),
                )
                for tensor in entry.tensors
            ),
        )

    def validate(self, analysis: Any, *, program: Any = None) -> None:
        target = analysis.program if program is None else program
        if self.shapes.program is not target:
            raise ValueError(
                "traffic geometry shapes belong to a different Program"
            )
        if self.whole_shapes.program is not target:
            raise ValueError(
                "traffic geometry whole shapes belong to a different Program"
            )
        for label, layer in (
            ("storage", self.storage),
            ("formats", self.formats),
            ("cache", self.cache),
        ):
            if layer.program not in (analysis.program, target):
                raise ValueError(
                    f"traffic geometry {label} belongs to a different Program"
                )
        if target is analysis.program:
            if set(self.shapes.values) != set(analysis.shapes.values):
                raise ValueError("traffic geometry must project every shape owner")
            for uid, source in analysis.shapes.values.items():
                if self._shape_structure(
                    self.shapes.values[uid]
                ) != self._shape_structure(source):
                    raise ValueError(
                        "traffic geometry changed shape axis identity for "
                        f"owner {uid}"
                    )
        if set(self.storage.values) != set(analysis.storage.values):
            raise ValueError("traffic geometry must project every storage owner")
        if set(self.formats.values) != set(analysis.formats.values):
            raise ValueError("traffic geometry must project every format owner")
        for uid, source_tensors in analysis.storage.values.items():
            local_tensors = self.storage.values[uid]
            source_formats = analysis.formats.values[uid]
            local_formats = self.formats.values[uid]
            cardinalities = {
                len(source_tensors),
                len(local_tensors),
                len(source_formats),
                len(local_formats),
            }
            if len(cardinalities) != 1:
                raise ValueError(
                    f"traffic geometry changed storage cardinality for owner {uid}"
                )
            if tuple(map(self._stored_tensor_identity, local_tensors)) != tuple(
                map(self._stored_tensor_identity, source_tensors)
            ):
                raise ValueError(
                    f"traffic geometry changed storage tensor identity for owner {uid}"
                )
            if tuple(map(self._format_identity, local_formats)) != tuple(
                map(self._format_identity, source_formats)
            ):
                raise ValueError(
                    f"traffic geometry changed format identity for owner {uid}"
                )
        expected_cache = cache.derive(analysis)
        if set(self.cache.values) != set(expected_cache.values):
            raise ValueError("traffic geometry must project every cache owner")
        for uid, source in expected_cache.values.items():
            if self._cache_identity(self.cache.values[uid]) != self._cache_identity(
                source
            ):
                raise ValueError(
                    f"traffic geometry changed cache tensor identity for owner {uid}"
                )


def declared_input(node: Identified, program: Any = None,
                   launch_of: Optional[Mapping[int, Any]] = None,
                   cached: Optional[Mapping[int, Any]] = None,
                   shapes: Any = None) -> Optional["Written"]:
    """How the op declares its input side is written, `None` where it
    carries real activations; the output side is `declared_output`.

    An `embedding` reads int32 token IDs; a `dsa_topk_mask` writes int32
    indices. Neither side is an activation, so pricing it at the deployment's
    activation dtype counts bytes the kernel never moves at that width. The
    declaration lives on the op — `OpSpec.input_dtype` / `output_dtype` —
    because which tensors are indices is a fact about the operation, not
    about the deployment.

    The input override exists for tensors that enter from OUTSIDE the graph
    (token IDs have no producer node). For a within-graph edge the output
    declaration is the authoritative one, and `derive` prices a reader's edge
    by its producer's declaration — the width is a property of the tensor,
    not of whoever reads it.
    """
    from ..ops import OP_REGISTRY

    if node.provenance == "comm":
        # A collective moves one tensor: it reads what it writes.
        return declared_output(node, program, launch_of, cached=cached,
                               shapes=shapes)
    spec = OP_REGISTRY.get(node.op or "")
    if spec is None or spec.input_dtype is None:
        return None
    return Written(float(dtype_bytes(spec.input_dtype)), spec.input_dtype)


@dataclass(frozen=True)
class Written:
    """How a node's output is written where that is not the deployment's
    activation width: the bytes per element, the element dtype they are the
    bytes of (scales and an unquantised tail included where the tensor is a
    cached record, scales where it is a quantised activation record), and
    the cached tensor it is when it is one
    (`eval.cache.landed_by`, or a recurrent state's write-back,
    `_settle_state_writes`)."""

    width: Any
    dtype: str
    cached: Any = None
    #: The quantised activation format (`ActivationFormat`) the elements are
    #: in, None for any other tensor: the record a grafted cast, or a launch
    #: that took one in, writes (`_written_formats`), and every element copy
    #: that hands it on (`ElementCopy`). `width` already counts its scales,
    #: over the tensor that holds them (`_record_width`).
    format: Any = None


@dataclass
class TensorReaders:
    """Who reads a node's output: the graph's edges, and what reads a tensor
    beyond them.

    One question, asked wherever a quantised write moves across an element
    copy — the cast graft moving a cast above it (`inference.
    materialize_cast`) and fusion reading a cast through views
    (`fusion.ViewChains`): the copy may change what it writes only if the
    readers in hand are ALL that read it. Beyond the edges, a tensor is read
    when it is cached (`cache.cached_node_uids`) or carries context
    (`cache.context_sources`: a KV marker view reads the cache's positions
    past its edge), a stream a later layer reads (`SpecIRTree.
    stream_writes`), a block template's output, a model-level node (stages
    read those by name: an aux stack's `inputs:`, a handoff), or a
    component's published output (`_published`)."""

    program: Any
    #: Node uids whose output something reads past the graph's edges, as
    #: given; None to find them on `graph` when first asked (`pinned`).
    pinned_uids: Optional[FrozenSet[int]] = None
    analysis: Any = None
    graph: Any = None
    rewrite: Any = None
    _by_source: Optional[Dict[Tuple[int, str], List[Tuple[Any, int]]]] = None

    @classmethod
    def of(cls, analysis: Any, program: Any, *, graph: Any = None,
           rewrite: Any = None) -> "TensorReaders":
        """Over `program`, with the beyond-edge reads found on `graph`
        (default `program`): the graph `analysis` walked or a graft of it —
        the comm graft's collectives carry a cached tensor's context on, so
        they are its readers past the edges too — which `rewrite` made from
        the walked one (the cast graft's bindings of edges no node writes,
        `CastGraft.rewrite`; None for none)."""
        return cls(program, analysis=analysis,
                   graph=program if graph is None else graph,
                   rewrite=Rewrite() if rewrite is None else rewrite)

    @property
    def pinned(self) -> FrozenSet[int]:
        """Node uids whose output something reads past the graph's edges."""
        if self.pinned_uids is None:
            graph = self.graph
            self.pinned_uids = frozenset(
                set(cache.cached_node_uids(graph))
                | set(_context_sources(
                    Edges(self.analysis, graph, self.rewrite)))
                | set(graph.ir.stream_writes.values())
                | {root.uid for template in graph.blocks
                   for root in template.output_roots()}
                | {node.uid for node in graph.ir.top_level})
        return self.pinned_uids

    def edges_from(self, node: Any, port: str = "") -> List[Tuple[Any, int]]:
        """`(reader, input index)` of every edge reading output `port` of
        `node`."""
        if self._by_source is None:
            self._by_source = {}
            for other in self.program.nodes():
                for index, ref in enumerate(other.inputs_obj or ()):
                    if ref.src is not None:
                        self._by_source.setdefault(
                            (ref.src.uid, cache.main_port(ref.port)),
                            []).append((other, index))
        return self._by_source.get((node.uid, cache.main_port(port)), [])

    def only(self, node: Any, port: str,
             readers: Iterable[Tuple[Any, int]]) -> bool:
        """Whether `readers` (`(leaf or component, input index)`) are all
        that read output `port` of `node`. An edge counts as theirs when it
        is one of them, or is read by a node of their region — a component
        entry enclosing a reader, or a body node inside a component reader,
        which pass the same tensor on (CLAUDE.md §2.8)."""
        if node.uid in self.pinned:
            return False
        ancestor = node.parent
        while ancestor is not None:
            if node in self._published(ancestor):
                return False
            ancestor = ancestor.parent
        listed = {(reader.uid, index) for reader, index in readers}
        region: set = set()
        for reader, _index in readers:
            up = reader.parent
            while up is not None:
                region.add(up.uid)
                up = up.parent
            region.update(n.uid for n in _below(reader))
        return all((other.uid, index) in listed or other.uid in region
                   for other, index in self.edges_from(node, port))

    def _published(self, component: Any) -> Tuple[Any, ...]:
        """The leaves `component` publishes as its outputs, primary and
        named."""
        ports = ("out", *dict(component.secondary_outputs or {}))
        return tuple(self.program.output_leaf(component, port)[0]
                     for port in ports)


def _below(node: Any) -> Iterator[Any]:
    for child in node.children or ():
        yield child
        yield from _below(child)


@dataclass(frozen=True)
class ElementCopy:
    """What a node that only COPIES elements copies, read off declarations.

    A copy writes elements of its data inputs unchanged and computes
    nothing: a view (`OpSpec.is_view_op`, or a component made only of
    views), a collective that does not reduce (`is_comm_op` and not
    `reduction`: all_to_all, all_gather, send_recv), an index gather of its
    `gather_source` (a MoE dispatch's `permute` is one), and a join
    (`AXIS_CONCAT`). Everything else computes, a reducing collective
    included (it sums).

    Two facts of a copy decide what a quantised record does across it
    (`keeps_rows`, and whether it is exact, which `commutes` combines with
    it), and both come from the op's dim correspondence (`axis_propagation.dim_map`)
    over the walked shapes, or, for a collective (whose declaration is
    shape-preserving and whose per-rank geometry is its event's), from its
    per-rank shapes:

      * `keeps_rows`: the output's innermost axis IS the data input's
        innermost axis, at the same extent — the axis a per-group scale
        lies along. Copying whole rows copies each row's scales with it,
        so quantising before the copy is quantising after it.
      * exact: it writes exactly the elements it is handed (no narrowing,
        no repetition). A view must be (a narrowed view would make a cast
        moved above it quantise elements nobody reads); a gather may
        repeat rows (`permute` copies each token `topk` times, and
        quantising the token once is the point), a collective moves what it
        moves.

    One answer, asked by the width walk (`declared_output`: a copy writes
    what it reads) and by the cast graft's canonicalisation (a cast commutes
    above a copy that keeps rows, `inference.materialize_cast`)."""

    #: The inputs (`inputs_obj` positions) whose elements it writes.
    inputs: Tuple[int, ...]
    keeps_rows: bool
    #: Whether a cast may move above it: it keeps rows, has one data input,
    #: and moves no fewer elements than it is handed (a view exactly them).
    commutes: bool
    #: Whether it is the tensor it reads, re-indexed or moved (a view, a
    #: collective) rather than a new tensor made of copies (a gather, a
    #: join): a cached tensor stays that tensor across the former only.
    same_tensor: bool

    @classmethod
    def of(cls, node: Identified, shapes: Any = None
           ) -> Optional["ElementCopy"]:
        """The copy `node` is, None when it computes. `shapes`: the walk's
        shape layer over the graph `node` is in (`Layer`), without which
        neither row fact can be read and both are False."""
        refs = tuple(node.inputs_obj or ())
        if not refs:
            return None
        spec = OP_REGISTRY.get(node.op or "")
        values = {} if shapes is None else shapes.values
        if node.provenance == "comm":
            if spec is None or spec.reduction or len(refs) != 1:
                return None
            shape = values.get(node.uid)
            keeps = (shape is not None and len(shape.ins) == 1
                     and bool(shape.ins[0]) and bool(shape.out)
                     and same_law(shape.ins[0][-1], shape.out[-1]))
            return cls((0,), keeps, keeps, True)
        if _is_view(node):
            leaves = ([node] if not node.children
                      else [leaf for leaf in _leaves_of(node)])
            keeps = all(_keeps_rows(leaf, 0, values.get(leaf.uid))
                        for leaf in leaves)
            exact_ = all(_moves_what_it_is_handed(values.get(leaf.uid))
                         for leaf in leaves)
            return cls((0,), keeps, keeps and exact_, True)
        if spec is None or node.children:
            return None
        if spec.gather_source is not None and spec.gather_source < len(refs):
            index = spec.gather_source
            shape = values.get(node.uid)
            keeps = _keeps_rows(node, index, shape)
            return cls((index,), keeps, keeps, False)
        if spec.shape_kind is ShapeKind.AXIS_CONCAT:
            inputs = tuple(range(len(refs)))
            shape = values.get(node.uid)
            keeps = all(_keeps_rows(node, i, shape) for i in inputs)
            # One cast above a join would be one per operand: not a move
            # of one launch.
            return cls(inputs, keeps, False, False)
        return None


def _leaves_of(node: Identified) -> Iterable[Identified]:
    for child in node.children or ():
        if child.children:
            yield from _leaves_of(child)
        else:
            yield child


def _keeps_rows(node: Identified, index: int, shape: Any) -> bool:
    """Whether `node`'s output innermost axis is a copy of its input
    `index`'s innermost axis at the same extent (`ElementCopy.keeps_rows`),
    read off its dim correspondence over its walked shape."""
    from ..axis_propagation import dim_map

    if shape is None or not shape.out or index >= len(shape.ins):
        return False
    handed = tuple(shape.ins[index] or ())
    if not handed:
        return False
    sources = dim_map(node.op, node, in_shapes=shape.ins)
    if not sources or len(sources) != len(shape.out):
        return False
    last = sources[-1]
    return (last.kind == "from"
            and (index, len(handed) - 1) in tuple(last.refs)
            and same_law(handed[-1], shape.out[-1]))


def _moves_what_it_is_handed(shape: Any) -> bool:
    if shape is None or not shape.out or len(shape.ins) != 1:
        return False
    return same_law(product(shape.ins[0]), product(shape.out))


def declared_width(node: Identified, program: Any = None,
                   launch_of: Optional[Mapping[int, Any]] = None,
                   port: str = "",
                   cached: Optional[Mapping[int, Any]] = None,
                   shapes: Any = None) -> Optional[float]:
    """The width a node writes its output at where that is not the
    deployment's activation width, else None (`declared_output`'s width)."""
    written = declared_output(node, program, launch_of, port, cached, shapes)
    return None if written is None else written.width


def payload_width(node: Identified, program: Any, shapes: Any,
                  payload: Sequence[Extent]) -> Optional[float]:
    """Bytes per element a collective moving `node`'s output carries over
    its payload buffer `payload` (the per-rank tensor it moves), where that
    is not the deployment's activation width, else None.

    The walk says what the elements are (`declared_output` over `shapes`);
    a quantised record's width is read by the one law (`_format_width`)
    over the payload itself, whose innermost extent sets the share of a
    partial scale group. The one answer for a collective's payload width —
    when it is grafted (`layout.materialize_rows.output_width`) and when the
    cast graft changes what the edge holds
    (`inference.materialize_cast.exchange_widths`)."""
    held = declared_output(node, program, shapes=shapes)
    if held is None:
        return None
    if held.format is None:
        return float(value_of(held.width))
    return float(value_of(_format_width(held.format, _elements(payload),
                                        _inner_of(payload))))


def declared_output(node: Identified, program: Any = None,
                    launch_of: Optional[Mapping[int, Any]] = None,
                    port: str = "",
                    cached: Optional[Mapping[int, Any]] = None,
                    shapes: Any = None) -> Optional[Written]:
    """How a node writes its output where that is not the deployment's
    activation width, else None.

    Declared where the tensor is made: the op's registration
    (`OpSpec.output_dtype` — DSA's int32 top-k, the fp32
    indexer logits), a fused launch's plan (`output_dtype` — MiniMax-M3's
    fp32 block scores), and the cache: the leaf that writes a cached tensor
    (`cached`, `eval.cache.landed_by`) writes it in the tensor's own format,
    scales and unquantised tail included (`CachedTensor.written_width`), so
    its write and every same-step read of it are that size — the size the
    cache law bills the read of it at a later step. A component publishes
    its output leaf's tensor, a fused launch publishes, on a port its plan
    states nothing for, the tensor of the member it publishes there, an
    element copy — a view, a collective, an index gather, a join
    (`ElementCopy`) — writes what it reads (`_copied`), and an
    elementwise op (`PRESERVE` / `BROADCAST` shape kind) writes its
    operands' promoted type: the widest width among them, an operand at the
    activation width counting as that. sglang's in-place `div_` over QSA's
    fp32 logits (`qsa/mqa.py:322`) writes fp32, whichever side the constant
    is.

    `launch_of` maps a node a rewrite absorbed to the launch that writes its
    output now: a reader's edge still names the member, and the tensor on it
    is the launch's — on the port the launch publishes that member's output
    at, whose declared width (`FusedNode.output_dtype`) is the tensor's.
    `port` is the port of `node` itself asked for (`""` the primary).

    A quantised activation record — a grafted cast's output, or the port
    of a launch that took one in (`_written_formats`) — is its elements and
    the scales beside them, priced over the tensor on that port; `shapes`
    (the walk's shape layer) says what that tensor is (`_record_width`)."""
    from ..ops import OP_REGISTRY, ShapeKind

    def written(node: Identified, port: str = "", *,
                inside: bool = False) -> Optional[Written]:
        # `inside`: resolving a member of the launch already asked for. Its
        # own reads are of other members, never of the launch's ports.
        if not inside and launch_of and node.uid in launch_of:
            launch = launch_of[node.uid]
            if node.uid not in launch.output_ports:
                raise ValueError(
                    f"node {node.uid} is read outside the launch "
                    f"{launch.op!r} that absorbed it, which publishes no "
                    f"port for it")
            port = launch.output_ports[node.uid]
            node = launch
        fmt = _written_formats(node, program).get(port or "")
        if fmt is not None:
            # A quantised activation record: a grafted cast declares the
            # whole format it writes (`SpecIRNode.output_format`), and a
            # launch that took one in writes it on the port it publishes
            # the cast at. Its elements AND the scales beside them, over the
            # tensor on that port — the size the pre-graft record of the
            # same launch bills (`_settle_conversions`), by the same law.
            return Written(_record_width(node, port, fmt, shapes), fmt.dtype,
                           format=fmt)
        declared = node.output_dtype or {}
        if port in declared:
            return Written(float(dtype_bytes(declared[port])),
                           declared[port])
        if isinstance(node, FusedNode):
            # The plan states nothing for this port: the tensor on it is
            # the member's that the launch publishes there, when that
            # member's own declaration says how it is written.
            want = port or ""
            for member_uid, published in node.output_ports.items():
                member = (program.by_uid(member_uid)
                          if program is not None else None)
                if (published or "") == want and member is not None:
                    found = written(member, inside=True)
                    if found is not None:
                        return found
        if port and declared:
            return None             # a named port the plan left undeclared
        if node.children and program is not None:
            # The port says WHICH of a component's outputs is read: resolving
            # it as the primary would name the wrong leaf (a hyper-connection
            # pre-mix publishes its reduced stream on `out` and its two mix
            # coefficients on `post` / `comb`, and they are different leaves).
            leaf, resolved = program.output_leaf(node, port or "out")
            return written(leaf, "" if resolved in ("", "out") else resolved,
                           inside=inside)
        if cached and not port and node.uid in cached:
            tensor = cached[node.uid]
            return Written(tensor.written_width, tensor.dtype, tensor)
        refs = tuple(node.inputs_obj or ())
        spec = OP_REGISTRY.get(node.op or "")
        if node.provenance == "comm" and len(refs) != 1:
            raise ValueError(
                f"communication node {node.uid} has no unique tensor "
                f"source")
        if spec is not None and spec.output_dtype is not None:
            return Written(float(dtype_bytes(spec.output_dtype)),
                           spec.output_dtype)
        copy = ElementCopy.of(node, shapes)
        if copy is not None or node.provenance == "comm":
            # An element copy writes what it reads (`ElementCopy`). A
            # reducing collective is no copy, but sums in the type it is
            # handed, which is the same answer for a plain dtype.
            inputs = (0,) if copy is None else copy.inputs
            return _copied(node, port, copy, [
                None if refs[i].src is None
                else written(refs[i].src, _port_of(refs[i]), inside=inside)
                for i in inputs], shapes)
        if (spec is not None
                and spec.shape_kind in (ShapeKind.PRESERVE, ShapeKind.BROADCAST)):
            return _promoted(written(ref.src, inside=inside)
                             for ref in refs if ref.src is not None)
        return None

    return written(node, port)


def _port_of(ref: Any) -> str:
    """The producer port an input ref reads, `""` for the primary."""
    return "" if ref.port in (None, "out") else ref.port


def _copied(node: Identified, port: str, copy: Optional[ElementCopy],
            parts: Sequence[Optional[Written]], shapes: Any
            ) -> Optional[Written]:
    """What an element copy writes: what it reads (`ElementCopy`).

    A quantised record is handed on as that format, its width read again
    over the copy's own output (`_record_width`), so a scale group the copy
    regroups is never inherited from its input's row. A record can only be
    handed on whole: every part in that one format, and its rows kept — a
    copy holding parts in different records, or regrouping a record's rows,
    holds no one record, and that is refused rather than priced at a width
    no tensor has. Plain parts written one way are written that way; parts
    written differently, or at the activation width, leave the copy at the
    activation width. A copy that is a new tensor (a gather, a join) is not
    the cached tensor it may read."""
    formats = {None if part is None else part.format for part in parts}
    if formats != {None}:
        fmt = next(f for f in formats if f is not None)
        if len(formats) != 1 or copy is None or not copy.keeps_rows:
            held = [None if part is None or part.format is None
                    else part.format.name for part in parts]
            raise ValueError(
                f"{node.display_path}: copies {held} "
                + ("without keeping their rows" if len(formats) == 1
                   else "held in different formats")
                + "; a quantised record is handed on only whole, every "
                "part in that format and its rows kept")
        return Written(_record_width(node, port, fmt, shapes), fmt.dtype,
                       format=fmt)
    first = parts[0] if parts else None
    if first is None or not all(
            part is not None and same_law(part.width, first.width)
            and part.dtype == first.dtype for part in parts[1:]):
        return None
    if copy is not None and not copy.same_tensor:
        return replace(first, cached=None)
    return first


def operand_dtype(node: Identified, edge_reads: Sequence["_EdgeRead"],
                  program: Any = None,
                  launch_of: Optional[Mapping[int, Any]] = None,
                  cached: Optional[Mapping[int, Any]] = None,
                  shapes: Any = None, *,
                  weight: Optional[Written] = None) -> Optional[str]:
    """The dtype a leaf's kernel computes on, as it was billed
    (`Traffic.operand_dtype`).

    An elementwise op (`PRESERVE` / `BROADCAST`) computes in its operands'
    promoted type (`promoted_operand`) — `add(x_bf16, logits_fp32)` runs in
    fp32 whichever side the logits are. Any other op computes on its first
    input edge (`edge_reads[0]`), the operand its declaration puts first
    and a matmul's own format applies to — promoted with `weight`, the
    weight it multiplies where that is held wider (`weight_promotion`): a
    BF16 row times an F32 matrix is an F32 product. None where the op
    declares its inputs as indices (`OpSpec.input_dtype` — an embedding's
    token IDs are an address, not what the gather computes on), where it
    reads nothing, and where an elementwise op's operands are all at the
    activation dtype: the caller's answer for "no data operand" is the
    role's dtype."""
    spec = OP_REGISTRY.get(node.op or "")
    if spec is not None and spec.input_dtype is not None:
        return None
    if spec is not None and spec.shape_kind in (ShapeKind.PRESERVE,
                                                ShapeKind.BROADCAST):
        widest = promoted_operand(node, program, launch_of, cached,
                                  shapes)
        return None if widest is None else widest.dtype
    operand = edge_reads[0].dtype if edge_reads else None
    if weight is not None and (
            operand is None
            or dtype_bytes(weight.dtype) > dtype_bytes(operand)):
        return weight.dtype
    return operand


def _promoted(operands: Iterable[Optional[Written]]) -> Optional[Written]:
    """Torch's type promotion over an elementwise op's operands: the widest
    of the declared ones, None (the activation width) where none is."""
    return max((w for w in operands if w is not None),
               key=lambda w: w.width, default=None)


def promoted_operand(node: Identified, program: Any = None,
                     launch_of: Optional[Mapping[int, Any]] = None,
                     cached: Optional[Mapping[int, Any]] = None,
                     shapes: Any = None) -> Optional[Written]:
    """The type an elementwise op (`PRESERVE` / `BROADCAST`) computes in: its
    operands' promoted type, by the rule its output is written at
    (`declared_output`) — but of the operands alone, so a node that declares
    its own output (a grafted cast writing FP8) still reads what arrives.
    None where every operand is at the activation width."""
    return _promoted(declared_output(ref.src, program, launch_of,
                                     cached=cached, shapes=shapes)
                     for ref in tuple(node.inputs_obj or ())
                     if ref.src is not None)


@dataclass(frozen=True)
class _Fanout:
    """A stride-0 repetition, with the axis identity it repeats.

    `total` is the element ratio a consumer saw before frames changed.  The
    consumer that finally reads the repeated tensor may not be in that frame:
    Ulysses repartitions a head broadcast's sequence and heads, so the rank
    which held every output head now holds a slice of them.  `head` and
    `head_names` retain the one structural fact needed there — that THIS part
    of the ratio repeated a `head_count` axis.  A repetition along any other
    axis keeps the element ratio; in particular, a tensor with an unchanged
    head axis that is broadcast over keys is not clamped by its heads.

    `headless_store` is the other structural fact, and a ratio cannot carry
    it: whether the tensor the view expanded declares NO head axis at all
    (absorbed MLA's single latent, shown to every query head) as against one
    whose heads the view repeats a group at a time (GQA's KV heads).  At one
    head per rank the two are the same number — the ratio is 1 either way —
    and a reader rebuilding the stored shape owes them opposite answers, so
    the fact is kept here instead of inferred back out of an extent.
    """

    total: Extent = 1
    head: Extent = 1
    head_names: FrozenSet[str] = frozenset()
    headless_store: bool = False

    @staticmethod
    def of(total: Extent = 1, head: Extent = 1,
           head_names: FrozenSet[str] = frozenset(),
           headless_store: bool = False) -> "_Fanout":
        if not head_names:
            head = 1
        return _Fanout(total=total, head=head, head_names=head_names,
                       headless_store=headless_store)

    def __mul__(self, other: "_Fanout") -> "_Fanout":
        names = self.head_names | other.head_names
        head = self.head * other.head
        return _Fanout(total=self.total * other.total, head=head,
                       head_names=names if head != 1 else frozenset(),
                       headless_store=(self.headless_store
                                       or other.headless_store))

    def in_frame(self, dims: Optional[Iterable[Extent]]) -> Extent:
        """The repetition in the consumer's operand frame.

        Head repetition is capped by the matching head axes that consumer
        holds: a rank holding `h_C` heads reads at least one stored head, so
        it sees at most `min(r, h_C)` copies.  Repetition on another axis,
        including an elementwise broadcast over keys, keeps its full ratio.
        """
        heads = _head_extent(dims, self.head_names)
        if heads is None:
            return self.total
        return ratio(self.total, self.head, empty=1) * minimum(
            self.head, heads)


def _head_axes(dims: Optional[Iterable[Extent]]
               ) -> Tuple[Tuple[Extent, FrozenSet[str]], ...]:
    """The axes a tensor declares as head counts, with their identities."""
    out = []
    for axis in dims or ():
        names = names_of(axis)
        if names and all((dim_symbol(name) or _NotHead).kind == "head_count"
                         for name in names):
            out.append((axis, names))
    return tuple(out)


class _NotHead:
    kind = ""


def _carries_heads(dims: Optional[Iterable[Extent]]) -> bool:
    """Whether a tensor holds its heads at all, in any form.

    Wider than `_head_axes`, which asks for an axis that IS a head count: a
    packed `n_heads*d_head` carries its heads too, and a view that splits
    that axis apart is handing on a tensor whose heads are stored.
    """
    return any(any((dim_symbol(name) or _NotHead).kind == "head_count"
                   for name in names_of(axis))
               for axis in dims or ())


def _head_extent(dims: Optional[Iterable[Extent]],
                 wanted: FrozenSet[str]) -> Optional[Extent]:
    """The product of head-count axes matching `wanted`; None when unknown.

    The match is asked of each axis's own identity (`names_of`) rather than
    of the copy `_head_axes` carries beside it: which axis this is, is a
    question about identity and never about magnitude, and the extent is
    where that answer lives.
    """
    axes = [axis for axis, _names in _head_axes(dims)
            if names_of(axis) & wanted]
    return product(axes) if axes else None


def _own_fanout(own: Any, inner: Any) -> _Fanout:
    """One view's repetition, decomposed by axis identity.

    The element ratio is still the total.  The head share is read from the
    axes the registry itself calls `head_count`, not from their names or this
    op's name, so a broadcast over another axis that merely coexists with a
    head axis is not mistaken for a head fan-out.

    A view that gives a head axis to a tensor which does not carry its heads
    at all repeats a whole store rather than a group of its heads:
    `headless_store`, which the extents stop telling apart once the view
    holds a single head.  Carrying them is asked of the store's axis
    identities and not of its rank (`_carries_heads`), because a view that
    merely splits a packed `n_heads*d_head` apart also hands on a rank-3
    tensor with no head AXIS, and its heads are stored.
    """
    if own is None or inner is None:
        return _Fanout()
    total = ratio(_elements(own.out), _elements(inner.out), empty=1)
    outer = _head_axes(own.out)
    inner_axes = _head_axes(inner.out)
    if not outer:
        return _Fanout(total=total)
    head = ratio(product(axis for axis, _names in outer),
                 product(axis for axis, _names in inner_axes), empty=1)
    names = frozenset().union(*(names for _axis, names in outer))
    return _Fanout.of(total, head, names,
                      headless_store=not _carries_heads(inner.out))


def _fanout(node: Optional[Identified], shapes: Mapping[int, Any],
            seen: Optional[frozenset] = None, *, edges: Edges) -> Extent:
    """How many times a producer's bytes are repeated in the view a consumer
    sees, following the chain back.

    A `head_broadcast` is a compile-time stride-0 expansion: the consumer's
    shape carries `[B, n_heads, s_kv, d_kv_lora+d_qk_rope]` but the kernel
    loads the latent once and fans `n_heads` out in registers. Counting the
    post-broadcast shape is how absorbed MLA came out reading `n_heads` times the bytes it really does — and absorbed
    MLA is the form every system uses at decode, so the model said the wrong one
    was cheaper.

    Four rules govern that normalization:
    function is the single declared source for them:

      * a view-only COMPONENT contributes its own fan-out, times whatever its
        input already had. The wrapper is where a repetition is visible as a
        ratio: `head_broadcast` comes out `n_heads`, its two siblings come out
        1, and none of the three is named here;
      * a view PRIMITIVE that declares `repeats_input` (`broadcast`, a
        stride-0 `.expand()`) contributes its ratio the same way. The ratio
        alone cannot say it is a fan-out — over a `slice` it is a narrowing,
        the clip rule's business below, and over a `kv_cache_view` it is the
        cache span, the cache law's — so the declaration does. DeepSeek-V4's
        indexer broadcasts its [B, S, H] head weights to the score's
        [B, H, S, T] for the `mul`; the fused MQA-logits kernel, like the
        unfused multiply, reads the H weights per token once, not T times;
      * `concat` takes the MINIMUM across branches. It declares
        `is_view_op=False` because "torch.cat always allocates", and yet the
        walker recurses through it: when EVERY branch is a broadcast the kernel
        does not concatenate at all, and when only some are — prefill MLA
        concatenates a real head-split with a broadcast rope — nothing may be
        dropped. Minimum is the fan-out reading of that intersection;
      * a declared view op passes through.

    This compatibility face answers in the producer's own frame.  The edge
    price asks `_fanout_factor(...).in_frame(consumer)` instead, because a
    context-parallel interior can repartition the frame between them.

    A ref naming a tensor rather than a node is followed to the producer the
    walk bound to it (`cache.Edges`), like any other edge.
    """
    return _fanout_factor(node, shapes, seen, edges=edges).total


def _fanout_factor(node: Optional[Identified], shapes: Mapping[int, Any],
                   seen: Optional[frozenset] = None, *,
                   edges: Edges) -> _Fanout:
    """`_fanout`, with the repeated axis retained for the reader's frame."""
    if node is None:
        return _Fanout()
    seen = (seen or frozenset())
    if node.uid in seen:
        return _Fanout()
    seen = seen | {node.uid}
    op = node.op or ""
    refs = tuple(node.inputs_obj or ())
    sources = [edges.source(node, index) for index in range(len(refs))]

    def _of(src: Optional[Identified]) -> _Fanout:
        return (_fanout_factor(src, shapes, seen, edges=edges)
                if src is not None else _Fanout())

    if op == "concat":
        if not sources or any(s is None for s in sources):
            return _Fanout()
        factors = [_of(s) for s in sources]
        # The least total over EVERY branch, so the accumulator is the extent
        # a branch's total is, from the first branch's total on — as the head
        # fold below is written.  Seeded with the `_Fanout` instead, the
        # accumulator was one type on the first round and another on the
        # rest: a two-way join folds once and never reads the difference, a
        # three-way one asks an extent for `.total` and has no answer.
        total = functools.reduce(
            lambda least, factor: where(sp.Lt, (factor.total, least),
                                        factor.total, least),
            factors[1:], factors[0].total)
        # Only a head repetition common to every branch may be clamped.  A
        # concat of a real branch with a broadcast one has minimum total one
        # and no repeated bytes to normalize; a mixed-axis minimum has no
        # sound componentwise reading, so it remains scalar.
        common = all(factor.head_names and factor.head_names ==
                     factors[0].head_names for factor in factors)
        if common:
            head = functools.reduce(
                lambda least, factor: minimum(least, factor.head),
                factors[1:], factors[0].head)
            return _Fanout.of(
                total, head, factors[0].head_names,
                # The join is one headless store only where every branch
                # is: concatenating a real head-split with a broadcast
                # leaves a tensor whose heads are stored.
                headless_store=all(factor.headless_store
                                   for factor in factors))
        return _Fanout(total=total)

    from ..op_names import view_only_components
    if op in view_only_components():
        own = shapes.get(node.uid)
        inner = shapes.get(sources[0].uid) if sources and sources[0] else None
        factor = _own_fanout(own, inner)
        return factor * _of(sources[0] if sources else None)

    if _is_view(node):
        factor = _Fanout()
        if repeats_input(op):
            own = shapes.get(node.uid)
            inner = shapes.get(sources[0].uid) if sources and sources[0] else None
            if own is not None and inner is not None:
                factor = _own_fanout(own, inner)
        return factor * _of(sources[0] if sources else None)
    return _Fanout()


def _always(*_operands: Any) -> Any:
    return sp.true


def _is_none(count: Any) -> Any:
    return sp.Eq(count, 0)


def _is_one(count: Any) -> Any:
    return sp.Eq(count, 1)


def _repeats_nothing(factor: Any) -> Any:
    """A fan-out that divides nothing: one copy per row, or no rows."""
    return sp.Or(sp.Eq(factor, 0), sp.Eq(factor, 1))


def _elements(dims: Optional[Iterable[Extent]]) -> Extent:
    """One implementation, in `values`. Kept as a name here because the call
    sites read better with it, and deleted the day they stop needing one."""
    return product(dims)


def _wider(a: Extent, b: Extent) -> Extent:
    """The larger of two spans on one stream.

    A formula (`where`), not a comparison this function takes itself: at a
    point, whichever span is wider there, with its own identity; over the
    points, the cases. What this once did was catch the `TypeError` of an
    undecidable `b > a` and keep `a` — the same shape of silent pick that
    priced a decode step's KV read at zero, on the same cache-read path:
    two incomparable symbolic spans billed the first and disagreed with the
    numeric walk at every point where the second was wider."""
    return where(sp.Lt, (a, b), b, a)


@dataclass(frozen=True)
class _ContextPart:
    """One sequence branch's share of one actual reader edge."""

    branch: ContextBranch
    #: The input elements belonging to this branch, after fan-out division.
    elements: Extent
    #: The authoritative rows this branch contributes to the reader's span.
    rows: Extent


@dataclass(frozen=True)
class _ContextSplit:
    """A positional context coverage resolved against one reader edge."""

    parts: Tuple[_ContextPart, ...]

    @property
    def seeds(self) -> FrozenSet[int]:
        return frozenset(
            seed for part in self.parts for seed in part.branch.seeds)


def _split_context(elements: Extent, coverage: ContextCoverage,
                   ) -> Optional[_ContextSplit]:
    """Split a reader's actual elements by declared sequence-branch rows.

    Only a disjoint cache/live union takes this path: its branch row counts
    add, so `rows / Σrows` is the one declared share of every non-row axis,
    and its live share cannot be guessed or allowed to fall back to a whole
    cache law. Every other coverage keeps the whole-edge rule — in particular
    an all-cache positional union, whose seeds are each read at their own law
    whether or not `KVLength` states each branch separately. A co-positional
    feature join likewise keeps the whole-edge rule: its branches own
    different channels of the same rows, not different positions, and the
    reader reads each of their seeds.
    """
    if not coverage.mixed:
        return None
    if any(branch.rows is None for branch in coverage.branches):
        raise ValueError(
            "a mixed positional context stream has a branch whose "
            "authoritative rows are unstated")
    total = sum(branch.rows for branch in coverage.branches)
    parts = tuple(
        _ContextPart(
            branch=branch,
            elements=ratio(elements * branch.rows, total, empty=0),
            rows=branch.rows)
        for branch in coverage.branches)
    return _ContextSplit(parts=parts)


def derive(analysis: Any, *,
           geometry: Optional[TrafficGeometry] = None,
           program: Any = None,
           rewrite: Any = None) -> Layer[Traffic]:
    """`Layer[Traffic]` over every node the analysis covers.

    Reads the layers that already exist and adds nothing to the walk. Run it
    again after a graph change and the difference is the change's effect —
    which is the whole reason it is a derivation and not a propagation.

    A module function rather than a method on `Traffic`, and the distinction is
    the one this package draws everywhere else: `Reconciliation.of` and
    `Placement` build ONE value out of their arguments, so the constructor
    belongs on the type. This builds a `Layer` over a graph, and the type it
    holds is incidental to that — putting it on `Traffic` would say the layer
    belongs to its element.
    """
    program = analysis.program if program is None else program
    precision = analysis.resolved_precision
    if geometry is not None:
        geometry.validate(analysis, program=program)
    shapes = analysis.shapes if geometry is None else geometry.shapes
    whole_shapes = shapes if geometry is None else geometry.whole_shapes
    storage = analysis.storage if geometry is None else geometry.storage
    formats = analysis.formats if geometry is None else geometry.formats
    cache_layer = None if geometry is None else geometry.cache
    out: Dict[int, Traffic] = {}

    if cache_layer is None:
        cache_layer = cache.derive(analysis)
    cache_laws = _kv_reads(analysis, cache_layer=cache_layer)
    paired = cache.paired_records(analysis)
    cache_view_frame_divisor = (
        {} if geometry is None else geometry.cache_view_frame_divisor)
    gathered_laws = _gathered_reads(analysis, cache_layer=cache_layer)
    state_carriers = frozenset(
        tensor.node for entry in cache_layer.values.values()
        for tensor in entry.tensors if tensor.recurrent)
    effective_rewrite = analysis.rewrite if rewrite is None else rewrite
    origins = dict(effective_rewrite.origins or {})
    # The launch that writes each absorbed node's output now: a reader's
    # edge still names the member (`declared_width`).
    launch_of = {member: program.by_uid(fused)
                 for fused, members in origins.items() for member in members
                 if program.by_uid(fused) is not None}
    read_aliases = (geometry.read_aliases if geometry is not None and geometry.read_aliases is not None
                    else program.read_aliases or {})
    borrowed_by = _borrowed_sources(program)
    # Each walked matmul whose weight is held wider than its activation —
    # the walk's own layer (`Analysis.weight_promotions`). Read for the
    # COMPUTE dtype alone (`operand_dtype` below); no width question asks
    # it, because a promotion is about the arithmetic and not about what
    # the launch stores.
    promoted = analysis.weight_promotions
    table_reads = _table_read_fractions(analysis, program, shapes, origins)
    # A reader in a later layer names the stream a writer published its
    # cached tensor under (YOCO / CLA shared KV), not the node: its layers'
    # `reads_from` name the producers, as `cache.kv_access` traces them.
    edges = Edges(analysis, program, effective_rewrite)
    context_srcs = _context_sources(edges)

    # The declared cache nodes. A reader's ref names that child
    # (`v_split_heads`), so the walk stops on the same uid the ledger
    # bills. Membership — is this edge's tensor cache-resident — is
    # `context_srcs`, which is that set closed over re-arrangement.
    cached_uids = cache.cached_node_uids(program)
    # The leaf that writes each cached tensor, so its write and the edge out
    # of it are at the tensor's own format (`declared_output`).
    writers = cache.landed_by(analysis, cache_layer)

    # Which leaf reads which cache, filled by the walk and priced after it,
    # when the gathering kernels' query shares are known: each leaf pays its
    # own read of every cache it reads (`_settle_cache_reads`).
    reads_cache: Dict[int, Dict[int, Extent]] = {}
    # Leaf uid -> the share of the step's queries this copy of it scores,
    # for each attention kernel that gathers the selection it reads
    # (`_gathers`). Its cache read is its queries' pairs, not one pass.
    gatherers: Dict[int, Extent] = {}
    # Leaf uid -> (the format change its kernel needs, the input edge it
    # needs it on). Settled into bytes after the walk.
    conversions: Dict[int, Tuple[Any, "_EdgeRead"]] = {}
    # Leaf uid -> what each of its input edges read, in input order.
    edge_reads_by_uid: Dict[int, List["_EdgeRead"]] = {}
    components: List[Any] = []

    # Only the nodes that run are billed. A rewrite keeps the shape rows of
    # members it absorbed whose output is still read outside the match
    # (`fusion.rebase_shapes`) so the walks below can follow a ref to them,
    # but their bytes are the fused node's. Skipped here, inside the walk,
    # rather than filtered after it: a kept row that reads a cache would
    # otherwise be billed a second read of it in `_settle_cache_reads`.
    live = {node.uid for node in program.nodes()}

    for uid, shape in shapes.values.items():
        node = program.by_uid(uid)
        if node is None or uid not in live:
            continue
        if node.children:
            # A component's traffic is its children's, and ONLY its
            # children's. The cache it declares is billed to the leaf that
            # reads it — a component adds nothing of its own, so nothing it
            # holds can be lost by a consumer that walks leaves. Deferred
            # rather than summed here because those leaves are not finished
            # until the cache shares are settled; the order is kept, so a
            # component still sees the sub-components it contains.
            components.append(node)
            continue

        # A fused node reads what its origins read, and that includes any
        # parameter threaded into one of them. Its OWN uid is in the union,
        # not an `or` alternative to the members: once a match is absorbed
        # the members are gone from `FusedProgram.nodes()`, and the edge to
        # the parameter survives on the fused node's `inputs_obj` (that is
        # what `_external_inputs` is for). Looking only at members would
        # lose gpt-oss's expert bias and every fused sink the moment a
        # framework's fusion plan applies.
        own, borrowed = streamed_sources(
            node, uid, members=(origins.get(uid) or (uid,)),
            read_aliases=read_aliases, borrowed_by=borrowed_by)
        (weight_bytes, metadata_bytes, borrowed_bytes, borrowed_metadata,
         weight_elems, borrowed_elems, table) = _weight_traffic(
            node, uid, own=own, storage=storage, formats=formats,
            borrowed=borrowed, table_reads=table_reads)

        # A VIEW moves nothing. `transpose`, `view_split`, `view_merge`,
        # `slice` and `broadcast` all declare `is_view_op`, and the declaration
        # means the tensor is re-indexed rather than rewritten — no HBM traffic
        # at all. Charging them a full read and write would make every
        # `head_split` and `head_merge` cost as much as the projection feeding
        # it.
        #
        # This generalises the fan-out rule in `_activation_reads` rather
        # than sitting beside it: a broadcast is a view whose fan-out
        # matters to its CONSUMER, and a view's own traffic is zero either
        # way.
        # An identity source moves nothing at all — not a read, and not a
        # write either. Its output IS the tensor it stores, already
        # resident; billing `_elements(shape.out) * act` every step would
        # just move the ghost kernel from the weight column to the
        # activation-write one, and `access.py` would carry it into
        # `access_bytes` and the per-op `t_sol` all the same.
        if is_identity_source(node.op):
            out[uid] = Traffic()
            continue

        if _is_view(node) or (
                passes_a_lone_operand(node.op or "")
                and sum(1 for dims in shape.ins
                        if not same_law(_elements(dims), 0)) == 1):
            # A view, or a join with one operand that is not empty by
            # declaration (handed on as it is). A join whose operands are
            # empty at some points only moves their elements, which come
            # to nothing there.
            out[uid] = Traffic(weight_reads=weight_bytes,
                               metadata_reads=metadata_bytes,
                               borrowed_weight_reads=borrowed_bytes,
                               borrowed_metadata_reads=borrowed_metadata,
                               weight_elems=weight_elems,
                               borrowed_weight_elems=borrowed_elems,
                               table=table)
            continue

        act = precision.edge_bytes(node.role, program.stack_of(node))
        in_override = declared_input(node, program, launch_of,
                                     cached=writers, shapes=shapes)
        main_written = declared_output(node, program, launch_of,
                                       cached=writers, shapes=shapes)
        gathers = _gathers(analysis, origins.get(uid) or (uid,))
        if reads_input_shape_only(node.op or ""):
            # A `*_like` constant: no input element moves.
            reads, seeds_read, edge_reads = 0, None, ()
        else:
            reads, seeds_read, edge_reads = _activation_reads(
                node, uid, shape, in_override=in_override,
                precision=precision, program=program, edges=edges,
                shapes=shapes, cached_uids=cached_uids,
                context_srcs=context_srcs, cache_laws=cache_laws,
                gathered=gathered_laws if gathers is True else None,
                launch_of=launch_of,
                cache_view_frame_divisor=cache_view_frame_divisor,
                whole_shapes=whole_shapes.values, writers=writers)
        # The format this leaf's kernel eats, and what it costs when the edge
        # does not already carry it. Settled after the walk: whether a
        # framework's fused launch wrote the edge in that format depends on
        # what else reads that edge (§2.3 invariant 3).
        edge_reads_by_uid[uid] = edge_reads
        # The formats it multiplies by: its members' own and the tensors
        # they read through a tie (`streamed_sources`), as billed above.
        # Only a launch that multiplies its weights eats its operand in the
        # format they declare: a lookup gathers rows of its table by id
        # (DeepSeek-V4.1's FP8 Engram table reads I32 token ids) and needs
        # no copy of its ids, whatever the table is stored as.
        multiplies = multiplies_weight(program, origins.get(uid) or (uid,))
        wanted = precision.conversion(
            node.role, program.stack_of(node),
            read_representations(formats, own, uid) if multiplies else ())
        operand = edge_reads[0] if edge_reads else None
        # The operand a kernel multiplies a quantised cached tensor with,
        # quantised to what its reader declares (`cache.paired_records`):
        # the same conversion, required by the cached operand rather than
        # by a weight.
        for (reader, index), target in paired.items():
            if reader != uid:
                continue
            if wanted is not None:
                raise ValueError(
                    f"{node.display_path}: two conversions required — its "
                    f"weights' and the cached operand's on input {index}")
            operand = _read_of_input(edge_reads, index)
            if operand is None:
                raise ValueError(
                    f"{node.display_path}: input {index}, which the cached "
                    f"operand it multiplies wants quantised, was not read")
            wanted = precision.conversion_to(
                target, node.role, program.stack_of(node))
        if wanted is not None and operand is not None:
            billed = _conversion_to_bill(
                wanted, operand, tuple(node.inputs_obj or ()),
                program=program, launch_of=launch_of, writers=writers,
                shapes=shapes)
            if billed is not None:
                conversions[uid] = (billed, operand)

        # Writes are the PRIMARY output plus every secondary port: a split's
        # three ports are three tensors landing in HBM, and counting only the
        # main one under-billed every multi-output op by its side channels
        # (V4's hyper-connection split writes pre, post AND the mixing
        # matrix).
        # Each port at the elements the launch WRITES there, which is its
        # tensor except where a fused boundary publishes a region nothing
        # inside it wrote (`Shape.written_elements`). Each term says which
        # of the two it is (`_PortWrite.spanned`), so the display prices
        # what the row charges and not the tensor it publishes.
        main_w = act if main_written is None else main_written.width
        main_elems = shape.written_elements()
        writes = main_elems * main_w
        out_terms = [_PortWrite(
            "", main_elems, main_w, same_law(main_elems,
                                             _elements(shape.out)),
            dtype=(precision.edge_dtype(node.role, program.stack_of(node))
                   if main_written is None else main_written.dtype),
            cached=None if main_written is None else main_written.cached)]
        if reads_own_output(node.op or ""):
            # An in-place update reads the tensor it then writes.
            reads = reads + main_elems * main_w
            edge_reads = (*edge_reads, _EdgeRead(
                producer=uid, port="", elements=main_elems,
                width=main_w, dtype=out_terms[0].dtype,
                inner=_inner_of(shape.out), index=-1))
        port_formats = (
            {} if main_written is None or main_written.format is None
            else {"": main_written.format})
        for port, port_dims in (shape.ports or {}).items():
            port_written = declared_output(node, program, port=port,
                                           cached=writers, shapes=shapes)
            if port_written is not None and port_written.format is not None:
                port_formats[port] = port_written.format
            if port_written is None:
                port_written = Written(
                    produced_width(node, port, precision, program),
                    produced_dtype(node, port, precision, program))
            port_w = port_written.width
            port_elems = shape.written_elements(port)
            writes = writes + port_elems * port_w
            out_terms.append(_PortWrite(port, port_elems, port_w,
                                        same_law(port_elems,
                                                 _elements(port_dims)),
                                        dtype=port_written.dtype,
                                        cached=port_written.cached))
        if seeds_read:
            reads_cache[uid] = seeds_read
            if gathers is True and any(seed in gathered_laws
                                       for seed in seeds_read):
                gatherers[uid] = _query_share(analysis, node, shapes, program)
        out[uid] = Traffic(
            weight_reads=weight_bytes,
            metadata_reads=metadata_bytes,
            act_reads=reads,
            act_writes=writes,
            borrowed_weight_reads=borrowed_bytes,
            borrowed_metadata_reads=borrowed_metadata,
            weight_elems=weight_elems,
            borrowed_weight_elems=borrowed_elems,
            table=table,
            in_edges=tuple(edge_reads),
            out_terms=tuple(out_terms),
            # The format each port's tensor is in, as the width walk
            # resolved it (`declared_output`): what a grafted cast or a
            # launch that took one in declares, and what an element copy of
            # that record hands on.
            written_formats=port_formats,
            # What this leaf COMPUTES at, where a weight widens it: the
            # widest promotion among the walked leaves this row stands for,
            # read off the walk's table (`Analysis.weight_promotions`)
            # rather than derived again here.
            #
            # The union over members, because a launch that absorbed an F32
            # matmul computes in F32 whichever member it publishes — the
            # instruction is the widest its members need. What it STORES is
            # a different question, answered by the declarations alone
            # (`declared_output`), which no promotion reaches —
            # which is why they read the table differently and both say so.
            operand_dtype=operand_dtype(
                node, edge_reads, program, launch_of, writers, shapes,
                weight=_promoted(
                    promoted.get(member)
                    for member in (origins.get(uid) or (uid,)))),
        )

    _settle_conversions(out, conversions, program, origins)
    _settle_gathered_live_reads(out, analysis=analysis, program=program,
                                shapes=shapes)
    launch_uid = {member: launch.uid for member, launch in launch_of.items()}
    _settle_cache_reads(out, reads_cache, cache_laws, state_carriers,
                        launch_of=launch_uid,
                        gathered_laws=gathered_laws, gatherers=gatherers)
    _settle_state_writes(out, _state_writes(analysis, cache_layer=cache_layer),
                         launch_of=launch_uid)

    for node in components:
        out[node.uid] = Traffic.sum(out[c.uid] for c in node.children
                                    if c.uid in out)
    return Layer(program, out)



def read_representations(formats: Layer[Any], own: Mapping[int, Any],
                         uid: int) -> Tuple[Any, ...]:
    """The stored representations a leaf's matmul format is read from: its
    own and the tensors it streams through a tie (`streamed_sources`'s
    `own`), and the leaf itself. One answer for every consumer that asks
    what a reader eats — the conversion `derive` prices and the fusion
    condition `readers_eat` (`reader_format`)."""
    return segments_of(formats.values, *dict.fromkeys((*own, uid)))


def reader_format(analysis: Any, node: Identified) -> Optional[Any]:
    """The activation format the leaf `node`'s kernel eats on its input
    (`ResolvedPrecision.required_format`), from the same representations
    `derive` prices its conversion from; None when it takes the edge as it
    arrives."""
    program = analysis.program
    own, _borrowed = streamed_sources(
        node, node.uid, members=(node.uid,),
        read_aliases=program.read_aliases or {},
        borrowed_by=_borrowed_sources(program))
    return analysis.resolved_precision.required_format(
        node.role, program.stack_of(node),
        read_representations(analysis.formats, own, node.uid))


def streamed_sources(node: Identified, uid: int, *, members, read_aliases,
                     borrowed_by) -> Tuple[Dict[int, Any], Tuple[int, ...]]:
    """`(own, borrowed)` — every stored-tensor node whose bytes this leaf
    moves, the reader's own and the ones threaded into it.

    `members` are the walked leaves this row stands for: a fused node's
    origins, else itself. The caller says, because the two callers hold the
    rewrite differently (the traffic walk has the origins map; the access
    projection is handed a fused program whose nodes carry their own).

    `own` is what the leaf (or, fused, each of its origins) stores, plus
    whatever each of those aliases (`Program.read_aliases`: a tied LM head
    streams the embedding's table), each with the share of it read (1 but
    for a tie to part of a tensor). `borrowed` is the parameters threaded in
    along a dataflow edge (`_borrowed_sources`: `F.linear(x, W)`, a bias
    `add`, a sink softmax) — the source's tensor, read once by this reader.

    One owner, because two consumers ask it: `_weight_traffic` bills these
    bytes, and the access projection reads the DTYPE of the same tensors
    (`projection.access._leaf_traffic_rows`). Asked twice, the answers drift
    — a tied LM head had its bytes billed and its dtype missing.
    """
    own: Dict[int, Any] = {}
    # An identity source reads nothing: its output IS the tensor, and the
    # consumer at the other end of the edge is what streams it.
    if not is_identity_source(node.op):
        for member in members:
            own[member] = 1
            for source, share in read_aliases.get(member, {}).items():
                own[source] = own.get(source, 0) + share
    # A fused node's OWN uid is in the scan beside its origins: once a match
    # is absorbed the members are gone from `FusedProgram.nodes()` and the
    # edge to the parameter survives on the fused node's `inputs_obj`.
    borrowed = tuple(dict.fromkeys(
        src for member in (*members, uid)
        for src in borrowed_by.get(member, ())))
    return own, borrowed


def widest_format(reps: Tuple[Any, ...]) -> Optional[Any]:
    """The resident format of the widest tensor a leaf streams. The GEMM's own
    weight, where it has one: a projection streams one matrix, and a fused
    span streams the widest of the ones it absorbed. Its dtype and its scale
    granularity are read off this one format, so they describe one tensor.

    Takes layer segments (`segments_of`), so a tensor stored wider on some of
    its node's layers is described by the widest of them."""
    formats = [rep.resident for rep in reps if rep.resident is not None]
    if not formats:
        return None
    return max(formats, key=lambda f: dtype_bytes(f.logical_dtype))


def streamed_formats(walked_formats: Mapping[int, Any], own: Iterable[int],
                     borrowed: Iterable[int]) -> Tuple[Any, ...]:
    """The resident formats of the tensors whose dtype describes this leaf's
    matmul: the ones it OWNS, and the ones it only borrows when it owns none.

    A leaf can multiply by a weight it does not store — `F.linear(x, W)`
    threads the matrix in along a dataflow edge, and a tied LM head streams
    the embedding's table. `derive` already bills those bytes
    (`borrowed_weight_reads`), so the row said how much it moved while
    saying nothing about WHAT: `weight_dtype` was None and every consumer
    of the dtype environment fell back to the activation's. An FP8 tied
    head was then keyed, and measured, as a BF16 one.

    Own wins where there is one, rather than the two sets being pooled:
    `widest_format` picks by width, and gpt-oss's expert bank borrows a
    BF16 bias beside its own FP4 weight — pooled, the bias would name the
    kernel. A leaf that owns a tensor is described by what it owns. The one
    answer for the row's weight dtype (`projection.access`) and for the
    weight a matmul's product is promoted by (`weight_promotion`).
    """
    for sources in (own, borrowed):
        reps = segments_of(walked_formats, *sources)
        if any(rep.resident is not None for rep in reps):
            return reps
    return ()


def multiplies_weight(program: Any, members: Iterable[int]) -> bool:
    """Whether a leaf standing for `members` (a fused node's origins, else
    itself) multiplies a stored weight as a matrix (`declaration_is_gemm`)."""
    return any(declaration_is_gemm(program.by_uid(member).declaration)
               for member in members if program.by_uid(member) is not None)


def weight_promotion(node: Identified, own: Iterable[int],
                     borrowed: Iterable[int], *, multiplies: bool,
                     formats: Mapping[int, Any], precision: Any,
                     program: Any) -> Optional[Written]:
    """The dtype a matmul COMPUTES in where the weight it multiplies is
    held wider than its activation, else None: the weight's.

    The type promotion of a mixed product. A BF16 row times an F32 matrix
    is an F32 multiply — its operand upcast as it is loaded, its FLOPs
    timed at the F32 peak, and its cell keyed on the F32 MMA family:
    DeepSeek-V4's hyper-connection mixes (`hc_*_fn`, F32 on disk) and an
    F32 router gate. A matmul (`multiplies`) only: an op that merely reads
    a wide parameter — a sink in a softmax, a selection bias, a positional
    table — computes on the operand it is handed.

    What the launch STORES is not this. Both frameworks' kernels narrow in
    registers as they write (vLLM `triton_ihc.py:297-299`, sglang
    `hy4_ihc.py:325` allocate at the sub-block operand's dtype), so the
    output keeps its edge's declared width and no cast node is grafted for
    a narrowing no kernel launches. Returning a width here and letting
    `declared_output` read it sent F32 down the graph after every promoted
    matmul, which cost Hy4's prefill 14% of a step in tensors the
    hardware never writes.

    The weight is the one the leaf streams (`streamed_formats`), at its
    RESIDENT format: what the checkpoint declares unless a precision
    profile converts it. A weight stored BF16 stays BF16 here however a
    framework loads it (GLM-5.3-Flash's `hc_*_fn`). Only weights that
    store values as-is rank by width (`is_unquantized`): a quantised
    format is dequantised into the activation's dtype, never wider."""
    if not multiplies:
        return None
    widest = widest_format(streamed_formats(formats, own, borrowed))
    if widest is None or not is_unquantized(widest.logical_dtype):
        return None
    width = float(dtype_bytes(widest.logical_dtype))
    if width <= precision.edge_bytes(node.role, program.stack_of(node)):
        return None
    return Written(width, widest.logical_dtype)


def weight_promotions(program: Any, formats: Mapping[int, Any],
                      precision: Any) -> Dict[int, Written]:
    """Every walked leaf whose matmul `weight_promotion` computes wide, and
    the dtype it computes in — one table of it, keyed by walked uid. Read
    for a row's compute dtype (`Traffic.operand_dtype`, and through it
    `ResolvedPrecision.compute_dtype` and the peak column); no width
    question reads it.

    Promotion compares a weight's format against its edge, so it is a
    question about ONE resolution and both of its sides have to come from
    that resolution: `Analysis.weight_promotions` asks it of the resident
    formats and the profile's edges, `Analysis.shipped_weight_promotions`
    of the saved formats and the edges the checkpoint ships. Mixing them —
    the resident table read beside saved formats — reports the profile's
    own widening as something the checkpoint already did.
    """
    read_aliases = program.read_aliases or {}
    borrowed_by = _borrowed_sources(program)
    out: Dict[int, Written] = {}
    for node in program.nodes():
        if node.children:
            continue
        own, borrowed = streamed_sources(
            node, node.uid, members=(node.uid,), read_aliases=read_aliases,
            borrowed_by=borrowed_by)
        held = weight_promotion(
            node, own, borrowed,
            multiplies=multiplies_weight(program, (node.uid,)),
            formats=formats, precision=precision, program=program)
        if held is not None:
            out[node.uid] = held
    return out


def _borrowed_sources(program) -> Dict[int, Tuple[int, ...]]:
    """Reader uid -> the identity sources whose tensors it reads.

    `F.linear(x, W)`, a bias `add`, a sink softmax, a positional table:
    the weight arrives along a dataflow edge instead of being owned, and
    the node at the other end declares data rather than transforming it.

    Component wrappers are skipped. They repeat their body's `inputs:` and
    have no cost of their own, so counting them would bill the same read
    twice — the same reason `Traffic.sum` builds a component out of its
    children rather than pricing it directly.
    """
    out: Dict[int, List[int]] = {}
    for node in program.nodes():
        if node.is_component:
            continue
        for ref in (node.inputs_obj or ()):
            src = ref.src
            if src is not None and is_identity_source(src.op):
                out.setdefault(node.uid, []).append(src.uid)
    return {uid: tuple(srcs) for uid, srcs in out.items()}


def _table_read_fractions(analysis: Any, program: Any, shapes: Layer[Any],
                          origins: Mapping[int, Tuple[int, ...]]
                          ) -> Dict[int, Tuple[Extent, Extent, int]]:
    """Per lookup leaf, `(N, R, table)`: the lookups one step makes, the
    table's global row count, and the uid holding the table, so its read is
    `min(T, T · N / R)`. Where the table is comes from `Program.lookups`:
    the leaf's own weight (an embedding) or the stored parameter on the
    input port its `OpSpec.lookup` names (DeepSeek-V4 `hash_route` gathers
    `tid2eid[input_ids]`, model.py `Gate.forward`); the law is the same.

    A lookup gathers one row per lookup; it does
    not stream the table (vLLM `F.embedding` in `vocab_parallel_embedding.py`,
    sglang `fused_vocab_parallel_embedding`). Its read is the table's bytes on
    this rank times `min(1, N / R)`: N lookups — the output's elements over its
    last axis, which is the row width — and R the table's GLOBAL row count,
    the leading axis of its declared weight. One expression covers every
    placement the table's `weight_axes` can select:

      * unsplit, or split along the row WIDTH (Qwen4Exp `colwise_gather_output`):
        each rank reads one (narrow) row per lookup, and its table is all R
        rows, so `T_rank · N / R` is exactly `N` rows of the rank's width;
      * split along the ROWS (`VocabParallelEmbedding`): which rank a lookup
        hits depends on the token id, which no workload carries. The rank's
        share of the lookups is taken as the share of real rows it holds — the
        balanced-lookup convention `OpSpec.lookup` declares — so the
        figure is the LONG-RUN AVERAGE expected useful bytes per rank, not
        any one step's physical bytes on the busiest rank. The deviation from
        that step-wise maximum is at most `N · b_row_rank · (1 − R_rank / R)`
        per rank per step (KB-scale at decode).

    `min(1, ·)` caps the read at the table: each row is read at most once.
    Lookups are counted, not distinct ids, so a repeated token reads its row
    again — an upper bound on DRAM traffic. A tied head that streams the same
    table (`Program.read_aliases`) is not a lookup and reads it whole.

    A fused kernel that absorbs a lookup gathers the same rows (DeepSeek-V4's
    hash router reads `tid2eid[input_ids]` inside the top-k launch): the
    count comes from the member's own shape row, which the rewrite keeps
    (`fusion.kept_rows`), and the read bills the launch. One lookup per
    launch: two would need a second table slot on the entry.
    """
    sites = analysis.program.lookups
    launch: Dict[int, int] = {}
    for fused, members in origins.items():
        absorbed = [m for m in members if m in sites]
        if len(absorbed) > 1:
            raise ValueError(
                f"a fused kernel absorbed {len(absorbed)} lookups "
                f"{sorted(analysis.program.by_uid(m).name for m in absorbed)}; "
                "a launch bills one table's gathered rows")
        if absorbed:
            launch[absorbed[0]] = fused
    out: Dict[int, Tuple[Extent, Extent, int]] = {}
    for uid, site in sites.items():
        node = analysis.program.by_uid(uid)
        if program.by_uid(uid) is None and uid not in launch:
            continue
        shape = shapes.values.get(uid)
        if shape is None or shape.out is None:
            # Not in this walk (an encoder the step does not run).
            continue
        where = f"{node.display_path} ({node.op})"
        stored = analysis.storage.values.get(site.table) or ()
        if not stored:
            raise ValueError(
                f"{where} gathers rows from a table that stores nothing")
        table = stored[0].shape
        width = analysis.shapes.values[uid].out[-1]
        if len(table) < 2 or value_of(table[-1]) != value_of(width):
            raise ValueError(
                f"{where} gathers rows of width {value_of(width)} from a "
                f"table of shape {[value_of(d) for d in table]}; a table is "
                "[rows, ..., row width]")
        # The rows one step gathers: the output's elements over its row
        # width, which the check above just matched against the table's. Both
        # come off the one shape, so this divides exactly.
        lookups = exact_div(product(shape.out), shape.out[-1])
        out[launch.get(uid, uid)] = (lookups, table[0], site.table)
    return out


def _weight_traffic(node: Identified, uid: int, *, own, storage, formats,
                    borrowed=(), table_reads=None
                    ) -> Tuple[Extent, Extent, Extent, Extent, Extent, Extent,
                               Optional[TableRead]]:
    """Weight + quant-metadata read bytes for one leaf.

    A node may READ stored tensors it does not own: a fused kernel reads
    its origins' weights (the rewrite says whose), and a tied LM head
    streams the embedding's table (`Program.read_aliases`). Storage stays
    keyed on the single stored copy; the read bills whoever moves it.
    Which sources those are is `streamed_sources`, asked by the caller.

    Bytes come from `Representation.resident`. A serving cast is an
    intent policy on the walk, not a second width asked here.
    """
    def _bytes(source: int) -> Tuple[Extent, Extent, Extent]:
        w: Extent = 0
        m: Extent = 0
        e: Extent = 0
        for row in pair_stored(
            source,
            storage.values.get(source),
            formats.values.get(source),
        ):
            if row.rep is None:
                continue
            # One copy of the tensor, which `pair_stored` bills per layer
            # segment: `row.count` is that segment's share of a copy (1 for
            # a tensor its node's layers all store alike), so the sum over
            # segments is what moving one copy costs — exact once the caller
            # multiplies by the reader's own copies.
            fmt = row.rep.resident
            w = w + fmt.weight_bytes_for(row.tensor.elements) * row.count
            m = m + fmt.metadata_bytes * row.count
            e = e + row.tensor.elements * row.count
        return w, m, e

    weight_bytes: Extent = 0
    metadata_bytes: Extent = 0
    borrowed_bytes: Extent = 0
    borrowed_metadata: Extent = 0
    weight_elems: Extent = 0
    borrowed_elems: Extent = 0
    table: Optional[TableRead] = None
    lookup = (table_reads or {}).get(uid)

    def _gathered(source: int, w: Extent, m: Extent, e: Extent,
                  borrowed: bool) -> Tuple[Extent, Extent, Extent]:
        # A lookup reads the rows it gathers, not its table; an alias of the
        # same table read by a projection (`read_aliases`) streams it whole.
        nonlocal table
        if lookup is None or source != lookup[2]:
            return w, m, e
        lookups, table_rows, _ = lookup
        table = TableRead(w, m, lookups, table_rows, borrowed)
        # The share of the table's bytes that is the gathered rows is a
        # proportion, so `ratio` — a table with no rows has nothing to gather.
        return tuple(minimum(x, ratio(x * lookups, table_rows, empty=0))
                     for x in (w, m, e))

    for source, share in own.items():
        read = _bytes(source) if share == 1 else tuple(
            x * share for x in _bytes(source))
        w, m, e = _gathered(source, *read, False)
        weight_bytes = weight_bytes + w
        metadata_bytes = metadata_bytes + m
        weight_elems = weight_elems + e
    # Union, not another branch of the `or` above: a fused kernel reads its
    # origins' weights AND any parameter threaded into it, and a short
    # circuit would drop the second the moment the first exists.
    for source in borrowed:
        w, m, e = _gathered(source, *_bytes(source), True)
        borrowed_bytes = borrowed_bytes + w
        borrowed_metadata = borrowed_metadata + m
        borrowed_elems = borrowed_elems + e
    return (weight_bytes, metadata_bytes, borrowed_bytes, borrowed_metadata,
            weight_elems, borrowed_elems, table)


def _cache_view_frame_divisor(source: Optional[Identified],
                              divisors: Mapping[int, Extent], *,
                              cached_uids: FrozenSet[int],
                              edges: Edges) -> Extent:
    """The landed/view frame divisor on one edge's view-only chain, or one.

    The search stops at the first op that materialises a tensor, or at a
    cached node, which lands its own tensor (`cache.materialized`, the same
    landing `fresh` reads).  Thus an
    edge downstream of a projection cannot borrow a cache view behind that
    projection, while a broadcast re-indexing a view can.  No seed-to-view
    provenance is consulted and no branches are joined here.
    """
    seen: set = set()
    while source is not None and source.uid not in seen:
        seen.add(source.uid)
        if source.uid in cached_uids:
            return 1
        if source.uid in divisors:
            return divisors[source.uid]
        if not _is_view(source):
            return 1
        source = edges.source(source, 0) if source.inputs_obj else None
    return 1


def _net_of_fanout(elements: Extent, factor: Extent) -> Extent:
    """`elements` with the copies a fan-out of `factor` made taken back out
    (exactly: the span contains them); a unit or empty fan-out divides
    nothing."""
    return where(_repeats_nothing, (factor,), elements,
                 exact_div(elements,
                           where(_repeats_nothing, (factor,), 1, factor)))


def _spans_in_whole_frame(uid: int, index: int, src: Identified,
                          landed: Identified, whole: Mapping[int, Any], *,
                          edges: Edges) -> Tuple[Extent, Extent]:
    """`(fresh, span)` of one cache-pedigree edge in the unsplit frame: the
    rows the landed tensor made this step and the rows the reader spans
    through this edge, each net of the fan-out its chain introduced."""
    reader, made = whole.get(uid), whole.get(landed.uid)
    if reader is None or made is None or made.out is None \
            or index >= len(reader.ins):
        raise ValueError(
            f"uid {uid} reads cache-resident uid {landed.uid}, and the "
            f"unsplit frame has no shape for the pair; whether it reads the "
            f"cache has no frame to be asked in")
    dims = reader.ins[index]
    span = _net_of_fanout(
        _elements(dims),
        _fanout_factor(src, whole, edges=edges).in_frame(dims))
    fresh = _net_of_fanout(_elements(made.out),
                           _fanout(landed, whole, edges=edges))
    return fresh, span


def _activation_reads(node: Identified, uid: int, shape, *, in_override,
                      precision, program, edges: Edges, shapes, cached_uids,
                      context_srcs, cache_laws: Mapping[int, Extent],
                      gathered: Optional[Mapping[int, Extent]] = None,
                      launch_of: Optional[Mapping[int, Any]] = None,
                      cache_view_frame_divisor: Mapping[int, Extent] = {},
                      whole_shapes: Mapping[int, Any],
                      writers: Optional[Mapping[int, Any]] = None,
                      ) -> Tuple[Extent, Dict[int, Extent],
                                 List["_EdgeRead"]]:
    """Activation read bytes + the caches this node reads.

    Returns (reads, seeds_read, edge_reads): the dense-edge price, per seed
    whether (1) or not (0) this node reads that cached tensor at the point,
    to be priced at the cache's law after the walk, and what each input
    edge contributed —
    its producer, the elements read and the width they were priced at — so a
    framework's fused quantise can reprice the one edge it changed.

    `gathered` is given when this node is an attention kernel that gathers
    its selection (`_gathers`): the selected rows of every cache it reaches
    that states one (`CachedTensor.gathered_bytes`) are read out of the
    cache query by query, so the edge is the cache's even where it spans no
    more than this step wrote — a fresh prefill's rows land in the cache and
    are gathered back from it, each once per query that selects it."""
    # Each input divided by the fan-out its producer chain introduced (a
    # broadcast repeats bytes the kernel reads once), priced at the WIDTH
    # THE EDGE CARRIES: the producer's role, or the deployment default
    # when there is no producer. A tensor's width is decided where it is
    # made; a reader prices what arrives, not what it would have
    # preferred. `qk_score` reads the int32 mask `dsa_topk_mask`
    # emitted at int32, whatever the deployment's activation dtype is.
    # Second gate. `derive` already bills an identity source nothing, and
    # `Evaluator._inputs` hands one no input, but a price is not the
    # place to discover that an edge should not exist.
    if is_identity_source(node.op):
        return 0, {}
    refs = tuple(node.inputs_obj or ())
    reads: Extent = 0
    # The seeds this node reads, and whether it reads each at the point.
    # Unioned over its edges rather than summed per edge: a fused attention
    # takes K and V by two refs and a separate one by a third, and the cache
    # each of them reaches is read once by the kernel however many refs
    # point into it — so the flag is the MAX over its edges, not their sum.
    seeds_read: Dict[int, Extent] = {}
    edge_reads: List[_EdgeRead] = []
    picked_from = gather_source(node.op or "")
    for index, dims in enumerate(shape.ins):
        # What the launch reads of this input: the tensor it is handed, less
        # any region it only passed a view over (`Shape.read_elements`).
        elements = shape.read_elements(index)
        # Whether that is still the whole of this input, which decides
        # whether a display may print the input's shape for this term
        # (`_EdgeRead.spanned`): a boundary that passed a region through
        # prices the region, and the shape would state a charge the row
        # does not make. Asked before the fan-out divides the count, so it
        # is a question about the tensor and not about the repetition.
        moves_whole = same_law(elements, _elements(dims))
        # A ref naming an inner stream reads a producer its layers' `reads_from`
        # name. Each layer's kernel reads one of them, and they keep one
        # cache law (`_one_cache_law`), so the first prices the read: the law
        # is charged once per kernel call, not once per producer.
        src = None
        if index < len(refs):
            found = edges.sources(node, index)
            if len(found) > 1:
                _one_cache_law(node, found, cached_uids, context_srcs,
                               cache_laws, edges)
            src = found[0] if found else None
        # A weight arriving on an edge is a WEIGHT read, priced at the
        # width it is stored in, and `_weight_traffic` has already billed
        # it to this node. Pricing it here as well counted the same tensor
        # twice under two names and at two widths — DeepSeek-V4's F32
        # hyper-connection matrix showed up as 458752 weight bytes on the
        # parameter node and 229376 activation bytes here, because an
        # activation edge is priced at the activation dtype.
        if src is not None and is_identity_source(src.op):
            continue
        # An edge out of a CONTEXT SOURCE reads cache-resident data.
        # Its act price is what the source materialised THIS step (a
        # decode K/V leaf emits one token); the attended-`s_kv` bulk the
        # reader's shape spans belongs to the cache law alone —
        # `kv_reads` bills it under each tensor's own growth rule
        # (windowed, latent, constant-state), which a dense edge price
        # cannot even express (a 4K window at 128K context is 32x).
        # Pricing the edge at its span counted those bytes twice. The
        # swap is all-or-nothing per edge and there is no phase case in
        # it: in a fresh prefill the two lengths coincide, so nothing is
        # clipped and nothing is owed. Edges with no cache pedigree keep
        # their span: a mismatch against the producer there is a modeling
        # error and must stay visible, not be quietly clipped.
        repetition = _fanout_factor(src, shapes.values, edges=edges)
        headless_store = repetition.headless_store
        factor = repetition.in_frame(dims)
        # A unit fan-out divides nothing, and an empty producer repeats
        # nothing.
        divided_by = where(_repeats_nothing, (factor,), 1, factor)
        # The span CONTAINS the fan-out's copies, so taking them back out
        # divides exactly (`exact_div`, which raises if a fan-out ever
        # appears that is not a factor of the span it made). A bare `/`
        # answered this differently per phase: a prefill span is a law in
        # `s_input` and sympy kept it exact, a decode span is a count and
        # `int / int` is a float.
        elements = where(_repeats_nothing, (factor,), elements,
                         exact_div(elements, divided_by))
        # Main's exact division (a fan-out that is not a factor of the span
        # it made is a bug, not a float) with this branch's answer to WHAT
        # the launch spans: a kernel that writes a new buffer crosses the
        # whole of it, while one that rotates lanes in place crosses the
        # lanes (`moves_whole`, from the plan's output contract).
        spanned: Any = moves_whole
        landed = _materialized(src, cached_uids, edges=edges)
        covered = _covered_source(src, context_srcs, edges)
        if covered is None and landed is not None \
                and landed.uid in context_srcs:
            covered = landed
        if covered is not None:
            coverage = context_srcs[covered.uid]
            split = _split_context(elements, coverage)
            if split is not None and all(
                    cache.producer_fresh_rows(
                        program.by_uid(part.branch.producer), shapes)
                    is not None for part in split.parts):
                # A mixed sequence union is settled branch by branch. A live
                # branch has no seed and stays activation for the rows its
                # `KVLength` declares. A cache branch at a
                # true consumer is replaced by the cache law. At a join inside
                # the coverage it keeps this step's fresh producer rows, so a
                # mixed union bills its cache suffix once as an activation and
                # once — downstream — as persistent-cache rows.
                branch_records: List[_EdgeBranchRead] = []
                taken: Extent = 0
                # Both flags below are Python truths, not magnitudes, and
                # the magnitude question in this block is `where`'s: whether
                # a branch's fresh rows fall short of the rows it carries is
                # asked of the values as the cases they are (`more`), and
                # these two only choose WHICH condition to ask of them.
                # `gathering` is a membership in the gathered mapping's KEYS
                # beside an `is not None`; `reads_cache_branch` is the
                # emptiness of a branch's `seeds`, declared `FrozenSet[int]`.
                # So no expression reaches either truth test, and
                # `tests/meta/test_symbolic_decisions.py` holds that
                # exemption with the same reason.
                gathering = gathered is not None and any(
                    seed in gathered for seed in split.seeds)
                frame_divisor = _cache_view_frame_divisor(
                    src, cache_view_frame_divisor,
                    cached_uids=cached_uids, edges=edges)
                for part in split.parts:
                    producer = program.by_uid(part.branch.producer)
                    # `exact(frame_divisor)`: the divisor is the view's cut
                    # over the landed tensor's cut and is built as an exact
                    # `Rational` (`physical_cost._cache_view_frame_divisors`),
                    # except on the paths that return the literal `1`. Read
                    # exactly, the quotient is SymPy's in either domain; `/`
                    # on the bare `1` turned a decode row count into a float.
                    fresh_rows = (
                        cache.producer_fresh_rows(producer, shapes)
                        / exact(frame_divisor))
                    reads_cache_branch = bool(part.branch.seeds)
                    cache_reads = (_always if gathering and reads_cache_branch
                                   else sp.Lt)
                    more = (fresh_rows, part.rows)
                    if reads_cache_branch:
                        fresh_elements = ratio(
                            part.elements * fresh_rows, part.rows, empty=0)
                        branch_elements = where(
                            cache_reads, more,
                            fresh_elements if uid in context_srcs else 0,
                            part.elements)
                        if uid not in context_srcs:
                            for seed in part.branch.seeds:
                                seeds_read[seed] = _wider(
                                    seeds_read.get(seed, 0),
                                    where(cache_reads, more, 1, 0))
                    else:
                        # A live marker's declared rows are the law; the
                        # union reader always spans the rows that arrived.
                        branch_elements = part.elements
                    branch_records.append(_EdgeBranchRead(
                        producer=part.branch.producer,
                        seeds=part.branch.seeds,
                        rows=part.rows,
                        law_node=part.branch.law_node,
                        elements=branch_elements,
                        spanned=same_law(branch_elements, part.elements),
                        gathered=(gathered is not None
                                  and not part.branch.seeds)))
                    taken = taken + branch_elements
                branch = in_override
                if branch is None and src is not None:
                    branch = declared_output(
                        src, program, launch_of, cached=writers,
                        shapes=shapes)
                if branch is None:
                    port = refs[index].port if index < len(refs) else ""
                    branch = Written(
                        produced_width(src, port, precision, program),
                        produced_dtype(src, port, precision, program))
                branch_width = branch.width
                reads = reads + taken * branch_width
                edge_reads.append(_EdgeRead(
                    producer=None if src is None else src.uid,
                    port=(refs[index].port if index < len(refs) else ""),
                    elements=taken, width=branch_width, dtype=branch.dtype,
                    inner=(dims[-1] if dims else None),
                    index=index, fanout=divided_by,
                    headless_store=headless_store,
                    spanned=all(record.spanned
                                for record in branch_records),
                    branches=tuple(branch_records)))
                continue
            if split is not None:
                # The complement of the branch-by-branch settlement above:
                # a mixed union with a producer whose fresh rows are unknown.
                raise ValueError(
                    "a mixed positional context stream has a branch producer "
                    "whose fresh sequence rows are unstated")
            # The reader SPANS more than the source materialised: what
            # it is reading is the cache, not this step's output, and
            # the cache is priced by its own law. So the law REPLACES
            # this edge's price rather than topping it up — the fresh
            # row was written to the cache and comes back as part of
            # the same `s_kv` positions, and charging it here as well
            # would count it twice. A decode reader spans the whole
            # context, so this binds; a prefill that attends exactly
            # what it just wrote spans no more than it produced, so it
            # does not, and the edge keeps its span. Both are single
            # counts of the same bytes under different names, which is
            # why the rule needs no phase case.
            #
            # `fresh` is net of the repetition the chain introduced — a
            # concat of two head-broadcasts lands `n_heads` stride-0 copies of
            # one fresh row — so the comparison is between tensors, not
            # between fan-outs. Outside the pedigree a span/production
            # mismatch is a modeling error and stays loud.
            produced = shapes.values.get(landed.uid)
            if produced is not None and produced.out is not None:
                fresh = _elements(produced.out)
                own = _fanout(landed, shapes.values, edges=edges)
                fresh = where(_repeats_nothing, (own,), fresh,
                              exact_div(fresh, where(_repeats_nothing,
                                                     (own,), 1, own)))
                # A whole-cache view can hand one reader only its resident
                # positions even though the producer landed a different
                # frame — local before Ulysses' gather, whole after it.  Put
                # `fresh` in that reader's frame by the landed/view ratio on
                # this edge; `elements` is already there.  This is the same
                # comparison and the same operands, not a second cache rule,
                # and one edge never inherits another edge's view.
                fresh = fresh / exact(_cache_view_frame_divisor(
                    src, cache_view_frame_divisor,
                    cached_uids=cached_uids, edges=edges))
                # Whether this reader spans more than its producer made
                # this step decides whether the step is billed a cache read,
                # and a decode step's cache read is most of what it moves.
                # It differs between points — a prefill attends what it
                # wrote, a decode step the whole context — so it is written
                # as the cases (`where`), never chosen here. No fresh row at
                # all still binds: a pooling cache whose step completes no
                # window (DeepSeek-V4.1's ratio-2 compressor at decode) emits
                # nothing, and its reader, spanning the pooled rows, is
                # reading the cache. A kernel that gathers its selection
                # reads the cache at every point.
                # `gathering` is a membership in the gathered mapping's KEYS
                # beside an `is not None` — a Python truth, and not one of
                # the magnitudes `where` is about to be asked about.
                gathering = gathered is not None and any(
                    seed in gathered
                    for seed in context_srcs[landed.uid].seeds)
                # Asked in the UNSPLIT frame (`whole_shapes`): whether the
                # step reads the cache is a fact of the workload, not of how
                # a deployment cuts it. Per rank, the reader and the tensor
                # that landed can be cut differently — a prefill
                # context-parallel rank lands its own slice of the sequence
                # and its reader takes every rank's K, gathered by a
                # collective the pre-communication program does not hold —
                # and comparing those two frames read a fresh prefill as a
                # cache read before the collective was grafted and not after
                # it. The magnitudes (`fresh`, `elements`) stay this frame's.
                spans_more = _spans_in_whole_frame(
                    uid, index, src, landed,
                    whole_shapes, edges=edges)
                reads_it = _always if gathering else sp.Lt
                if uid in context_srcs:
                    # A node INSIDE the pedigree is not the consumer but a
                    # link in the chain: a concat re-indexes what it was
                    # handed, so it is charged the fresh rows where the
                    # cache is read, and the law goes to whoever reads the
                    # concat. Billing every hop would charge one cache once
                    # per hop.
                    taken = where(reads_it, spans_more, fresh, elements)
                else:
                    # Whether this edge reads the cache at all, per seed it
                    # reaches. HOW MUCH it reads is the seed's law, priced
                    # in `_settle_cache_reads`, and not this span: the span
                    # is in the reader's frame, and under a context split a
                    # reader that gathers K/V is in a different frame from
                    # the producer that cut the sequence, so comparing spans
                    # across readers compared two cuts. Which part of a row a
                    # reader takes is which seeds it reaches: a concat's
                    # branches are separate seeds, so absorbed MLA's score
                    # reaches the 512-wide latent and the 64-wide positional
                    # half, and its apply the latent alone.
                    for seed in context_srcs[landed.uid].seeds:
                        seeds_read[seed] = _wider(
                            seeds_read.get(seed, 0),
                            where(reads_it, spans_more, 1, 0))
                    taken = where(reads_it, spans_more, 0, elements)
                # Where the law takes the span over the term prints the
                # elements; where it does not, whatever the boundary
                # already said about this input stands.
                spanned = where(reads_it, spans_more, False, spanned)
                elements = taken
        if index == picked_from:
            # An index gather reads what it picks of this input: one of its
            # elements per element written, and no element it does not pick
            # (`OpSpec.gather_source`). The rows of a dispatch a rank's own
            # experts receive, not the step's whole token stream.
            written = shape.written_elements()
            spanned = where(sp.Lt, (written, elements), False, spanned)
            elements = minimum(elements, written)
        elif node.op == DISPATCH_OP and index == 1 and shape.out:
            # Its routing operand is the INDICES the gather follows, one per
            # routed row — not the router's dense candidate scores. The graph
            # hands this node the wide score tensor it came from; the launch
            # reads only the selected row numbers it expands into the output.
            rows = exact_div(shape.written_elements(), shape.out[-1])
            spanned = where(sp.Lt, (rows, elements), False, spanned)
            elements = minimum(elements, rows)
        carried = in_override
        port = refs[index].port if index < len(refs) else ""
        if carried is None and src is not None:
            carried = declared_output(src, program, launch_of,
                                      "" if port == "out" else port,
                                      cached=writers, shapes=shapes)
        if carried is None:
            carried = Written(produced_width(src, port, precision, program),
                              produced_dtype(src, port, precision, program))
        width = carried.width
        reads = reads + elements * width
        # One record per edge, in input order: the first is the operand a
        # matmul's own format applies to (as every other reading of "the
        # input" here does — `shape.ins[0]`, the access row's
        # `act_in_elems`), and the rest say who else reads their producer.
        edge_reads.append(_EdgeRead(
            producer=None if src is None else src.uid,
            port=(refs[index].port if index < len(refs) else ""),
            elements=elements, width=width, dtype=carried.dtype,
            inner=(dims[-1] if dims else None),
            index=index, fanout=divided_by, headless_store=headless_store,
            spanned=spanned))
    return reads, seeds_read, edge_reads


@dataclass(frozen=True)
class _EdgeRead:
    """What one input edge contributed to a leaf's activation read: the
    producer it came from, the elements read after fan-out, the width they
    were priced at and the tensor's innermost extent (a per-token scale is
    one per row)."""

    producer: Optional[int]
    #: The producer's output port this edge read ("" for the main output).
    #: A launch publishing several tensors quantises the one its consumers
    #: read, not all of them.
    port: str
    elements: Extent
    width: float
    #: The dtype `width` is the bytes of (`Written.dtype`): the name the
    #: measurement key states for this operand, so the key and the bill read
    #: one declaration (`Traffic.operand_dtype`).
    dtype: str
    inner: Any
    #: Which input of the reader (`Shape.ins` position); -1 for the reader's
    #: own output, read back by an in-place update.
    index: int = 0
    #: The fan-out the input's span was divided by (`_fanout`): the stride-0
    #: repetition its chain of views introduced.
    fanout: Extent = 1
    #: Whether that repetition gave a head axis to a tensor stored without
    #: one (`_Fanout.headless_store`) — absorbed MLA's latent, not GQA's
    #: grouped KV heads. The ratio stops saying so at one head per rank.
    headless_store: bool = False
    #: Whether `elements` is still the input's span over `fanout`. False when
    #: the cache law took the span over (the edge prices the rows this step
    #: wrote, or nothing) — what is read then is not the input's shape.
    spanned: bool = True
    #: The sequence branches that made `elements`, with each branch's actual
    #: reach. Internal to the derivation: the public access row still exposes
    #: one term per input edge, while this records why a mixed edge is partly
    #: activation and partly cache.
    branches: Tuple["_EdgeBranchRead", ...] = ()


@dataclass(frozen=True)
class CastLaunch:
    """A format change on one leaf's input that NO producer absorbed.

    Which is to say: a launch of its own. The frameworks run it as one —
    vLLM quantises at `BlockScaledMMLinearKernel.py:119` and calls the GEMM
    at `:132`, sglang at `fp8_utils.py:1355` — and `kernel_bench` measures
    it as the `quantise_activation` cell. `Traffic.convert_bytes` prices it,
    but as a number inside the CONSUMER's traffic, so everything downstream
    reads one node where the hardware runs two: the benchmark row bills the
    GEMM for both and comes out at 1.7-2.7x the card's bandwidth, which is
    how 6,196 measured rows were refused by the SOL gate.

    This is the fact those readers need instead, published where the
    pre-graft walk prices the conversion (`_settle_conversions`) rather than
    re-derived from a byte count nobody can take apart. It is the CONSUMER's
    record: the serving graph grafts one `quantise_activation` node per
    tensor and format from these (`inference.materialize_casts`), and where
    that node stands and which launch takes it in are the graft's and
    fusion's, not this record's.

    The three bytes are the split of what the consumer's pre-graft row
    carries: `wide_read` is the operand as the edge holds it — the quantise
    kernel's read, priced on the consumer's `act_reads` because the
    consumer is the only node there is — and `narrow_write` + `narrow_read`
    are `convert_bytes`, the copy written once and streamed back once.
    """

    #: The leaf whose kernel needs the operand in another format.
    consumer: int
    #: Which of its inputs (`Shape.ins` position).
    index: int
    #: The launch and port that wrote the edge, None for a model input.
    producer: Optional[Tuple[int, str]]
    #: `quantise`, `requantise` or `dequantise` (`precision.Conversion.kind`).
    kind: str
    #: The format written, None when the cast lands on a plain dtype.
    target: Any
    #: The element dtype written.
    target_dtype: str
    #: Bytes the cast launch READS: the operand as the edge holds it.
    wide_read: Extent
    #: Bytes it writes: the narrow copy and its scales.
    narrow_write: Extent
    #: Bytes the consumer then streams back in.
    narrow_read: Extent

    @property
    def convert_bytes(self) -> Extent:
        """What `Traffic.convert_bytes` holds for this cast: the narrow copy
        written once and read once. Stated here so the two cannot drift."""
        return self.narrow_write + self.narrow_read


@dataclass(frozen=True)
class _PortWrite:
    """What one written tensor contributed to a leaf's activation write: the
    output port it lands on ("" the main output), the elements written and
    the width they were priced at. A recurrent state's write-back, which no
    port publishes, is a term named for the state, `cached` on it and never
    `spanned` (`_settle_state_writes`).

    `spanned` is `_EdgeRead.spanned`'s question on this side: whether
    `elements` is still the whole of what that port publishes. A fused
    boundary that only rewrote a region of its output publishes more than
    its launch wrote (`Shape.written_elements`), and a display that printed
    the published shape there would state a charge the row does not make —
    which is the one thing the access popover must never do
    (`test_access_formula_states_the_charge`). The display prints the
    region's elements instead, exactly as it does for a read the cache law
    took over."""

    port: str
    elements: Extent
    width: Extent
    spanned: Any = True
    #: The element dtype written: the activation dtype, a declared dtype, a
    #: cached tensor's format dtype.
    dtype: Optional[str] = None
    #: The cached tensor this write lands (`eval.cache.landed_by`), None
    #: for an activation.
    cached: Any = None


@dataclass(frozen=True)
class _EdgeBranchRead:
    """One sequence branch's contribution to an `_EdgeRead`.

    A gathered live branch is initially recorded at its one-pass size; the
    post-walk settlement replaces that read with this reader's share of the
    query-by-row gather, as cache laws are settled across their readers.
    """

    producer: Optional[int]
    seeds: FrozenSet[int]
    rows: Extent
    elements: Extent
    spanned: bool
    #: Whether the evaluator says this reader gathers the branch. Set only on
    #: a live branch; cache branches settle through their cache laws instead.
    gathered: bool = False
    #: The stream marker that stated `rows` (and a live gather's pairs).
    law_node: Optional[int] = None


def _one_cache_law(reader: Identified, sources, cached_uids, context_srcs,
                   cache_laws: Mapping[int, Extent], edges: Edges) -> None:
    """The producers a reader of an inner stream reads over its layers keep
    one cache law each — what its kernel reads per call is one of them. The
    walk holds their values to one shape, layout and KV length; this holds
    the bytes, which the cache's dtype also decides."""
    laws = []
    for source in sources:
        landed = _materialized(source, cached_uids, edges=edges)
        seeds = (context_srcs[landed.uid].seeds
                 if landed is not None and landed.uid in context_srcs
                 else frozenset())
        laws.append(sum(cache_laws.get(seed, 0) for seed in seeds))
    if any(expr_of(law) != expr_of(laws[0]) for law in laws[1:]):
        raise SpecRefusal(
            f"{reader.display_path!r} reads an inner stream from layers "
            f"whose caches read {laws} bytes per step; one template's layers "
            f"read one cache law",
            declaration="reads_from",
            fix=("give the layers that read differing caches templates of "
                 "their own: one template per source cache"))


def produced_width(src, port, precision: Any, program) -> float:
    """The width of the tensor on an edge: the producing leaf's, as the
    precision profile's activation table decides it (`edge_bytes`) — the
    bytes of `produced_dtype`.

    A fused node may publish several members; the port picks the member.
    No producer (block input, token ids) is the model's own width.
    """
    return float(dtype_bytes(produced_dtype(src, port, precision, program)))


def produced_dtype(src, port, precision: Any, program) -> str:
    """The dtype `produced_width` is the bytes of: the producing leaf's
    activation dtype (`ResolvedPrecision.edge_dtype`)."""
    if src is None:
        return precision.edge_dtype(None, "")
    ports = src.output_ports if isinstance(src, FusedNode) else {}
    want = port or "out"
    for origin, name in ports.items():
        if (name or "out") == want:
            member = program.by_uid(origin)
            if member is None:
                return precision.edge_dtype(None, "")
            return precision.edge_dtype(member.role, program.stack_of(member))
    return precision.edge_dtype(src.role, program.stack_of(src))


def _holds(held: Optional[Written], wanted: Any, writer: Any) -> bool:
    """Whether an edge written as `held` by `writer` is already in the form
    a conversion `wanted` would make: the quantised format it targets, or,
    for a cast onto a plain dtype (a dequantise), a cast's write of that
    dtype."""
    if held is None:
        return False
    if wanted.target is not None:
        return held.format == wanted.target
    # A plain dtype is no format to compare: an edge declared at the target
    # dtype by its op (an fp32 logit) is not a dequantised copy of the
    # quantised operand the conversion starts from. Only a grafted cast's
    # write of that dtype is the conversion already made.
    return (held.format is None and held.dtype == wanted.target_dtype
            and writer.provenance == "cast")


def _read_of_input(edge_reads: Sequence["_EdgeRead"],
                   index: int) -> Optional["_EdgeRead"]:
    """The edge read that came in on the reader's input `index`
    (`_EdgeRead.index`), or None where that input was not read."""
    return next((edge for edge in edge_reads if edge.index == index), None)


def _ref_read_by(operand: "_EdgeRead",
                 input_refs: Sequence[Any]) -> Optional[Any]:
    """The input ref `operand` came in on, or None where its position is not
    one of the reader's declared inputs (`_EdgeRead.index` is -1 for the
    reader's own output, read back by an in-place update)."""
    return (input_refs[operand.index]
            if 0 <= operand.index < len(input_refs) else None)


def _conversion_to_bill(wanted: Any, operand: "_EdgeRead",
                        input_refs: Sequence[Any], *, program: Any,
                        launch_of: Any, writers: Any, shapes: Any) -> Any:
    """The conversion this leaf pays for on `operand` — `wanted`, or None
    where the edge already holds it.

    An edge that already holds the format wanted needs no conversion: a
    grafted cast made it (`inference.materialize_casts`), or a copy of one
    hands it on (a dispatch and its exchange move the copy made on the
    tokens), or a launch that took a cast in wrote it. Billing the consumer
    as well would count the one format change twice. A grafted
    stored-transform op (a rotation) writes no format and is no conversion.

    Everything decided here is Python metadata — an operand's position
    among the reader's inputs, and whether the writer's declared output is
    already the target format (`_holds`) — so it is asked here, of plain
    types, and not among the walk's extents."""
    ref = _ref_read_by(operand, input_refs)
    writer = None if ref is None else ref.src
    held = (None if writer is None else declared_output(
        writer, program, launch_of, _port_of(ref), cached=writers,
        shapes=shapes))
    return None if _holds(held, wanted, writer) else wanted


def _settle_conversions(out: Dict[int, Traffic],
                        conversions: Dict[int, Tuple[Any, "_EdgeRead"]],
                        program: Any,
                        origins: Mapping[int, Tuple[int, ...]]) -> None:
    """Price each leaf's input conversion as the launch that makes it.

    Invariant 3 of the design: one format change is one price. A
    conversion still on a consumer here is one nothing upstream made — the
    serving graph grafts each as a `quantise_activation` node before fusion
    (`inference.materialize_cast`), and a fused launch that writes its
    output quantised is a plan naming that node as a member, so the cast
    is in the launch's bytes by topology and no consumer carries it. What
    is left is recorded as a launch (`CastLaunch`) for the graft to read.

    The producer an edge names is the node the WALK ran, which a rewrite may
    since have absorbed into a launch (`origins`), so each edge is resolved
    to the launch that actually writes it.
    """
    launch_of = {member: fused for fused, members in origins.items()
                 for member in members}

    def written_by(read: "_EdgeRead") -> Optional[Tuple[int, str]]:
        """The launch and port that writes what this edge reads, None for an
        edge with no producer (a model input, a block's own input)."""
        if read.producer is None:
            return None
        launch = launch_of.get(read.producer)
        if launch is None:
            return (read.producer, read.port)
        node = program.by_uid(launch)
        if node is None:
            return (read.producer, read.port)
        # The launch renames its members' outputs to its own ports.
        return (launch, node.output_ports.get(read.producer, read.port))

    for uid, (conversion, edge) in conversions.items():
        traffic = out.get(uid)
        if traffic is None:
            continue
        tensor = written_by(edge)
        # The copy is written once and read back once, each at the width
        # the record it lands as has: a quantised format's elements and
        # scales by the one law the grafted cast's write and its readers
        # are priced at (`_format_width`, through `_record_width`), so the
        # launch is one price before the graft and after it; a cast onto a
        # plain dtype at that dtype's.
        width = (float(dtype_bytes(conversion.target_dtype))
                 if conversion.target is None
                 else _format_width(conversion.target, edge.elements,
                                    edge.inner))
        narrow = edge.elements * width
        launch = CastLaunch(
            consumer=uid, index=edge.index,
            producer=tensor,
            kind=conversion.kind, target=conversion.target,
            target_dtype=conversion.target_dtype,
            wide_read=edge.elements * edge.width,
            narrow_write=narrow, narrow_read=narrow)
        out[uid] = replace(traffic, convert_bytes=launch.convert_bytes,
                           cast=launch)


def _written_formats(node: Any, program: Any) -> Dict[str, Any]:
    """The quantised formats a leaf writes, by output port: a grafted
    cast's own declaration (`SpecIRNode.output_format`), and for a fused
    launch each member's that it publishes, on the port it publishes the
    member at — one statement, read wherever the tensor lands."""

    if isinstance(node, FusedNode):
        if program is None:
            raise ValueError(
                f"launch {node.op!r}: which of its members' formats it "
                f"writes is read off its members, and no program was given "
                f"to find them in")
        out: Dict[str, Any] = {}
        for uid, port in node.output_ports.items():
            member = program.by_uid(uid)
            if member is not None:
                found = _written_formats(member, program).get("")
                if found is not None:
                    out[port or ""] = found
        return out
    if node.provenance == "cast" and node.output_format is not None:
        return {"": node.output_format}
    return {}


def _record_width(node: Any, port: str, fmt: Any, shapes: Any) -> Extent:
    """Bytes per element of the quantised record `node` writes on `port`
    in `fmt`: `_format_width` over the tensor the walk put on that port,
    in the scale tile the launch declares when it pads one
    (`FusedNode.output_scale_tile`)."""

    shape = None if shapes is None else shapes.values.get(node.uid)
    dims = None if shape is None else shape.axes_for(port)
    if dims is None:
        raise ValueError(
            f"node {node.uid} ({node.op!r}) writes a {fmt.name} record on "
            f"port {port or 'out'!r}, and no walked shape says what tensor "
            f"that is: its scales cannot be priced")
    tile = node.output_scale_tile if isinstance(node, FusedNode) else None
    return _format_width(fmt, _elements(dims), _inner_of(dims), tile)


def _inner_of(dims: Optional[Iterable[Extent]]) -> Any:
    axes = tuple(dims or ())
    return axes[-1] if axes else None


def _format_width(fmt: Any, elements: Extent, inner: Any,
                  scale_tile: Any = None) -> Extent:
    """Bytes per element of a quantised activation tensor: its data plus the
    share of its scales — `ceil(inner / group)` per row (a partial last
    group still takes one), or one per row for a per-token organisation.

    `scale_tile` is the scale sidecar's storage tile as (rows, groups) when
    the producing launch pads it — FlashInfer's F8_128x4 (vLLM
    `mxfp8_scale_bytes`, query_quant.py:31-33) rounds the token rows to 128
    and the 32-value groups to 4 and stores the padded positions as zeros,
    so a 64-token cell writes twice the scale bytes the plain per-group
    share prices. The fused node that writes such a record declares the
    tile (`FusionOpSpec.output_scale_tile`).

    The one law for an activation record's width: the write of a grafted
    cast and of a launch that took one in (`_record_width`), every read of
    either, and the pre-graft record of the same launch
    (`_settle_conversions`)."""
    per_element = float(dtype_bytes(fmt.dtype))
    if inner is None:
        raise ValueError(
            f"a {fmt.name} record's scales lie along its rows, and the "
            f"tensor has no innermost extent to count them by")
    # The rows are the elements over the row width, both read off the one
    # shape, so the division is exact by construction (`exact_div`).
    rows = exact_div(elements, inner)
    if fmt.scale_granularity == "per-group":
        # One scale per group of a row, a partial last group included: the
        # quantiser stores ceil(inner / group) of them per row.
        groups = ceil_div(inner, fmt.group)
        if scale_tile:
            # The launch's storage tile pads the counted scales: the rows
            # and the groups each rounded up to the tile.
            rows_tile, groups_tile = scale_tile
            rows = ceil_div(rows, rows_tile) * rows_tile
            groups = ceil_div(groups, groups_tile) * groups_tile
        scales = rows * groups * float(dtype_bytes(fmt.scale_dtype))
        return per_element + ratio(scales, elements, empty=0)
    scales = rows * float(dtype_bytes(fmt.scale_dtype))
    return per_element + ratio(scales, elements, empty=0)


def _settle_gathered_live_reads(out: Dict[int, Traffic], *, analysis: Any,
                                program: Any, shapes: Any) -> None:
    """Turn gathered live branches into one query-by-row read.

    The walk records a live branch as it records every activation: the rows
    the edge carries. A reader the evaluator says gathers (`Analysis.gathered`)
    loads those rows once per query, and each such kernel pays its own
    gather, as each reader of a cache pays its own read
    (`_settle_cache_reads`): unfused, the score and the apply are two
    launches that each load the rows. Several edges of one kernel into one
    branch are one gather, allocated over those edges in proportion to what
    each spans. Settled after the walk; it stays activation traffic because
    these rows are not cached.
    """
    groups: Dict[Tuple[int, int], List[Tuple[
        int, "_EdgeRead", "_EdgeBranchRead", Extent, Extent]]] = {}
    for uid, traffic in out.items():
        if not any(branch.gathered
                   for edge in traffic.in_edges
                   for branch in edge.branches):
            continue
        node = program.by_uid(uid)
        if node is None:
            continue
        for edge in traffic.in_edges:
            for branch in edge.branches:
                # `_EdgeBranchRead.gathered` is declared `bool` and is set
                # from one (`gathered is not None and not …seeds` in
                # `_activation_reads`): whether the evaluator says this
                # reader gathers the branch, which no workload changes. The
                # magnitudes of the branch are `elements` and `rows` below,
                # and they are multiplied, never decided. So no expression
                # reaches this truth test, and
                # `tests/meta/test_symbolic_decisions.py` holds that
                # exemption with the same reason.
                if not branch.gathered:
                    continue
                span = branch.elements * edge.width
                law_node = branch.law_node or branch.producer
                length = analysis.kv_lengths.get(law_node)
                if length is None or length.pairs is None:
                    raise ValueError(
                        "a gathered live stream has no KVLength.pairs on its "
                        "marker; its physical read has no law")
                # The cache branch's `gathered_bytes()` is pairs × bytes per
                # row. A live marker has no cache format, so the row's bytes
                # come from this edge and the local reader takes its share of
                # the step's queries. Multiplying the edge first keeps the
                # expression in the walk's bound frame.
                selected_queries = (
                    ratio(length.pairs, branch.rows, empty=0)
                    * _query_share(analysis, node, shapes, program))
                law = span * selected_queries
                entry = (uid, edge, branch, span, law)
                groups.setdefault((law_node, uid), []).append(entry)

    for entries in groups.values():
        law = entries[0][4]
        if any(expr_of(other) != expr_of(law)
               for other in (entry[4] for entry in entries[1:])):
            raise ValueError(
                "one kernel gathers one live stream at different "
                "query-by-row sizes on its edges; its read has no one law to "
                "allocate over them")
        total = sum(entry[3] for entry in entries)
        count = len(entries)
        for entry in entries:
            uid, edge, branch, span, _law = entry
            share = where(_is_one, (count,), law, ratio(
                law * span, total, empty=ratio(law, count, empty=0)))
            allocated = ratio(share, edge.width, empty=0)
            branches = tuple(
                replace(other, elements=allocated,
                        spanned=same_law(allocated, other.elements))
                if other is branch else other
                for other in edge.branches)
            edges = tuple(
                replace(other, branches=branches) if other is edge else other
                for other in out[uid].in_edges)
            out[uid] = replace(
                out[uid],
                act_reads=out[uid].act_reads - span + share,
                in_edges=edges)


def _settle_cache_reads(out: Dict[int, Traffic],
                        reads_cache: Dict[int, Dict[int, Extent]],
                        cache_laws: Dict[int, Extent],
                        state_carriers: FrozenSet[int], *,
                        gathered_laws: Optional[Mapping[int, Extent]] = None,
                        gatherers: Optional[Mapping[int, Extent]] = None,
                        launch_of: Optional[Mapping[int, int]] = None,
                        ) -> None:
    """Bill every kernel that reads a cache its own read of it.

    Runs after the walk, when each gathering kernel's share of the step's
    queries is known.

    A kernel reads the cache from HBM itself. Two kernels that read one
    cached tensor — attention's score and its apply, run unfused — are two
    launches, and each streams the tensor through its own pass: two reads,
    each billed to its own row, not one read split between them. Fusion is
    what saves the second read: a framework kernel that scores and applies
    in one launch is one leaf after the rewrite, it reads the tensor once,
    and the difference between the two programs is that saving, priced
    where it happens (`fusion_delta`). Several edges of one kernel into one
    tensor (K and V refs into one latent) are still one read: the walk's
    flag is the max over the kernel's edges (`_activation_reads`).

    What a kernel reads of a tensor is the tensor's law. A streaming kernel
    reads its `read` positions once (`cache_laws`). A kernel that gathers its
    selection (`gatherers`, with the share of the step's queries this copy of
    it scores) reads, for each of its queries, every live pair's row of a
    tensor whose law states one (`gathered_laws`), at the tensor's row
    width. Which part of a concatenated row a kernel takes is which seeds it
    reaches, each billed at its own law.

    Once per kernel per step, so a cache shared across layers (YOCO / CLA:
    a later layer attends the KV an earlier one `stream_out`ed) is stored
    once and read by every layer's kernel that attends it."""
    gathered_laws = gathered_laws or {}
    gatherers = gatherers or {}
    readers: Dict[int, Dict[int, Extent]] = {}
    for uid, seeds in reads_cache.items():
        for seed, reads in seeds.items():
            readers.setdefault(seed, {})[uid] = reads

    # A RECURRENT STATE is read by no edge. A `state:` cache entry says so in
    # as many words — "no graph node outputs it" — so the per-edge walk in
    # `derive`, which finds a cache read by following an edge back to what
    # produced it, finds nothing. Decode is that read almost entirely for a
    # Mamba or DeltaNet layer, and it was once free: Nemotron-3-Ultra's
    # mixer, 4.3 MB a step a layer, absent. Each state names the kernel that
    # streams it, its law is keyed on that carrier, and the carrier is its
    # one reader — the scan for the recurrent matrix, the causal conv for
    # its window. A carrier a fused kernel absorbed is read by that launch
    # (`launch_of`, member → fused uid): the law stays keyed on the carrier,
    # the row it lands on is the launch's.
    launch_of = launch_of or {}
    for seed in state_carriers:
        if seed in readers:
            raise ValueError(
                f"uid {seed} carries a recurrent state and is also a cached "
                f"node some edge reads; one tensor's read would be billed "
                f"twice")
        reader = launch_of.get(seed, seed)
        if reader not in out:
            raise ValueError(
                f"uid {seed} carries a recurrent state but has no traffic row "
                f"to bill its read to")
        readers[seed] = {reader: 1}
    for seed, flags in readers.items():
        gathered = gathered_laws.get(seed)
        for uid, reads in flags.items():
            # Which law is a membership and an `is not None`, not a
            # magnitude: whether this kernel gathers is the evaluator's.
            if gathered is not None and uid in gatherers:
                law = gathered * gatherers[uid]
            else:
                law = cache_laws.get(seed)
            if law is None or same_law(law, 0):
                continue
            # Whether a kernel reads the tensor can differ between points (a
            # reader spans the cache at decode and only its own rows at a
            # fresh prefill), so the read is the cases: nothing where it
            # does not read, its read where it does.
            read = where(_is_none, (reads,), 0, law)
            out[uid] = replace(out[uid], kv_reads=out[uid].kv_reads + read)


def _kv_reads(analysis: Any, *,
              cache_layer: Optional[Layer[Any]] = None) -> Dict[int, Extent]:
    """Cache bytes read per step, per CACHED NODE.

    A step reads, for every request in flight, whatever each cached tensor's own
    law says it reads at this cache length. That law is `eval.cache`'s subject
    and is asked of it rather than assumed here: this function previously summed
    the cached nodes into one per-token figure and multiplied by `s_kv`, which is
    the dense answer and is the dense answer for a sliding window too — 32x too
    large for a 4K window at 128K context, and the right shape of number.

    Keyed on the node the tensor is kept AT, not on the component that declared
    it, because these bytes have to reach a reader. `eval.cache` keys its
    entries by component — the cache is a property of the attention layer, two
    of whose nodes share one window — and a component is exactly what the
    per-op projection skips (`if node.children: continue`). Billed to the
    component, the whole cache read landed on a row nothing downstream reads,
    which is how a 4K-context MiMo step reported 128 GB of traffic while 213 GB
    of it sat on the parent. The node key is what `_context_sources` traces
    back to, so the caller can put each law on the leaf that reads it.
    """
    layer = cache.derive(analysis) if cache_layer is None else cache_layer
    # A walk that keeps no cache — an encode walk, which runs no language
    # step and binds no `s_kv` — reads none.
    if not layer.values:
        return {}
    cache_len = cache.workload_dim(analysis, "s_kv")
    batch = cache.workload_dim(analysis, "B")
    if same_law(cache_len, 0):
        return {}
    out: Dict[int, Extent] = {}
    for entry in layer.values.values():
        for tensor in entry.tensors:
            # No cache, no read at the points where the step attends no
            # position. A recurrent state holds the positions BEFORE the
            # step folded in, so it is read where there are some: decode,
            # and a prefill continuing a cached prefix (`s_past`, vLLM
            # `num_computed_tokens`, sglang `extend_prefix_lens` — its
            # `has_initial_states`). A fresh prefill is billed no read and
            # one write (`_state_writes`). Stated assumption: some kernels
            # gather the empty slot anyway at a fresh prefill (vLLM GDN,
            # qwen_gdn_linear_attn.py:1599, gathers then zeroes; sglang's
            # GDN/KDA and vLLM's Ling read it unconditionally) — at most one
            # state per request per layer, invisible next to a prefill's
            # weight and activation streams at SOL, so it is not billed;
            # telling the frameworks apart would first need that declared
            # as a fact of the binding.
            held = (cache.workload_dim(analysis, "s_past") if tensor.recurrent
                    else cache_len)
            read = where(_is_none, (held,), 0,
                         tensor.bytes_at(cache_len, read=True))
            if same_law(read, 0):
                continue
            # ``CachedTensor.bytes_at`` owns the position projection. This is
            # crucial for a hybrid component whose K/V positions are spread by
            # DCP while its recurrent state is replicated on every rank.
            out[tensor.node] = out.get(tensor.node, 0) + read * batch
    return out


def _state_writes(analysis: Any, *,
                  cache_layer: Optional[Layer[Any]] = None
                  ) -> Dict[int, Tuple[Tuple[cache.CachedTensor, Extent], ...]]:
    """The recurrent states each carrier writes back per step, keyed on the
    carrier: `(tensor, elements written)` — every request's state, as many
    times as the carrier reads it (`CachedTensor.read`, the entry's
    `reads:`), at the points where the step runs at all.

    A state's carrier "reads and rewrites it every step" (the cache entry's
    declaration): a scan folds the step's tokens into the matrix and stores
    it back in its slot, and a fresh prefill stores the state its last
    token leaves. No graph node outputs it, so no edge carries the write,
    exactly as none carries the read (`_kv_reads`); the carrier's row
    bills it as a written tensor at the state's own format
    (`_settle_state_writes`)."""
    layer = cache.derive(analysis) if cache_layer is None else cache_layer
    if not layer.values:
        return {}
    cache_len = cache.workload_dim(analysis, "s_kv")
    if same_law(cache_len, 0):
        return {}
    batch = cache.workload_dim(analysis, "B")
    out: Dict[int, List[Tuple[cache.CachedTensor, Extent]]] = {}
    for entry in layer.values.values():
        for tensor in entry.tensors:
            if not tensor.recurrent:
                continue
            elements = where(_is_none, (cache_len,), 0,
                             tensor.elements
                             * tensor.positions(cache_len, read=True)
                             * batch)
            if same_law(elements, 0):
                continue
            out.setdefault(tensor.node, []).append((tensor, elements))
    return {uid: tuple(writes) for uid, writes in out.items()}


def _settle_state_writes(
        out: Dict[int, Traffic],
        writes: Mapping[int, Tuple[Tuple[Any, Extent], ...]], *,
        launch_of: Optional[Mapping[int, int]] = None) -> None:
    """Bill each recurrent state's write-back to the launch that runs its
    carrier — the carrier's own row, or the fused launch that absorbed it,
    as for its read (`_settle_cache_reads`). One write term per state, at
    the width its format lands it (`CachedTensor.written_width`), named for
    the state and marked as the cached tensor it is."""
    launch_of = launch_of or {}
    for carrier, states in writes.items():
        row = launch_of.get(carrier, carrier)
        if row not in out:
            raise ValueError(
                f"uid {carrier} carries a recurrent state but has no traffic "
                f"row to bill its write to")
        traffic = out[row]
        terms = tuple(_PortWrite(tensor.name, elements, tensor.written_width,
                                 spanned=False, dtype=tensor.dtype,
                                 cached=tensor)
                      for tensor, elements in states)
        out[row] = replace(
            traffic,
            act_writes=traffic.act_writes + sum(
                term.elements * term.width for term in terms),
            out_terms=traffic.out_terms + terms)


def _gathered_reads(analysis: Any, *,
                    cache_layer: Optional[Layer[Any]] = None
                    ) -> Dict[int, Extent]:
    """Cache bytes a step's selection reads, per CACHED NODE, for the
    tensors whose law states one (`CachedTensor.gathered_bytes`): every
    live (query, key) pair's row, for every request in flight. What an
    attention kernel that gathers its selection reads of them (`_gathers`)
    in place of `_kv_reads`' one pass; at one query a step the two agree."""
    out: Dict[int, Extent] = {}
    layer = cache.derive(analysis) if cache_layer is None else cache_layer
    for entry in layer.values.values():
        for tensor in entry.tensors:
            read = tensor.gathered_bytes()
            if read is None:
                continue
            batch = cache.workload_dim(analysis, "B")
            out[tensor.node] = out.get(tensor.node, 0) + read * batch
    return out


def _gathers(analysis: Any, members: Iterable[int]) -> bool:
    """Whether this leaf — or, fused, one of the leaves it stands for —
    reads its keys through a per-query index list (`Analysis.gathered`): a
    top-k attention, or an attention over a union of index lists.

    Such a kernel loads each query's listed rows by index: FlashMLA sparse,
    vLLM `_gqa_sparse_fwd_kernel` and sglang `_sparse_gqa_chunk_prefill` run
    a grid over queries, so a row two queries list is read twice. Every
    other reader of a cache streams it once — a dense or windowed attention
    tiles its context, an indexer scores every pooled key."""
    return any(member in analysis.gathered for member in members)


def _query_rows(analysis: Any, node: Any, shapes: Any,
                program: Any) -> Extent:
    """The query rows this copy of a reader scores (`Shape.queries`).

    A fused kernel's matrix is the attention kernel it absorbed
    (`fusion.score_matrix_owner`)."""

    member = score_matrix_owner(node, program, shapes)
    if member is None:
        raise ValueError(
            f"{node.display_path}: a fused kernel that gathers its "
            f"selection absorbed no attention kernel whose score matrix "
            f"says which of the step's queries it scores")
    shape = shapes.values.get(member.uid)
    rows = (shape.queries(attention_kernel(member.op))
            if shape is not None else None)
    if rows is None:
        raise ValueError(
            "an attention kernel that gathers its selection has no "
            "`s_query` axis to say which of the step's queries it scores")
    return rows


def _query_share(analysis: Any, node: Any, shapes: Any,
                 program: Any) -> Extent:
    """The share of the step's queries this copy of a reader scores.

    A prefill context-parallel rank that cuts the sequence (PCP) holds
    `s_query / cp` of them and gathers only their pairs; one that repartitions
    heads (Ulysses) scores every query for its heads. One where the reader's
    queries are the step's, which is every unsplit walk.
    """
    # A ratio of two counts is a `Rational`, and `exact` on both halves
    # keeps it one: a prefill rank's counts are laws in `s_input` where
    # sympy's `/` is exact, a decode step's are the literal `n_verify` where
    # `128 / 128` is the float `1.0` — the same split `layout.physical.
    # pairs_share` published two different numbers for.
    return exact(_query_rows(analysis, node, shapes, program)) / exact(
        cache.workload_dim(analysis, "s_query"))
