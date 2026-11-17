"""The KV cache — per cached tensor, as a function of the cache length.

Why this is not a per-token scalar
----------------------------------
The cache is the one footprint that is not linear in the sequence, and writing
it as `per_token_bytes x s_kv` assumes it is. Three of the four classes in the
shipped corpus break that assumption, and each breaks it differently:

    dense MLA / GQA     s_kv                        positions, per token
    sliding window      Min(s_kv, s_window)         stops growing at the window
    time-compressed     ceil(s_kv / m)              grows m times slower
    linear / SSM        1                           a fixed matrix, s_kv-independent

A single scalar can only carry the first. The other three come out as the dense
answer scaled — for a 4K window at 128K context, 32x too large — and the error
is invisible in a total because it is the right shape of number.

So the unit here is the TENSOR, not the layer: `Cache` holds one `CachedTensor`
per node the component declared, each with its own positions law and its own
dtype, and a byte count is that list summed at a length. A hybrid layer that
caches an SWA `kv_concat` alongside a compressed latent is then not a special
case — it is two entries whose laws differ, which is what it is.

Where the laws come from
------------------------
`OpSpec.kv_positions_per_query_fn` / `kv_storage_positions_per_query_fn`, which
the ops already declare beside the `flops_fn` counting the same positions. The
walk records them per node in `CACHE_LEN` (see `Analysis.cache_positions`); this
module only decides WHICH op speaks for a given cached tensor, in four steps:

    1. the node's own op, if it declares a law
    2. its output op, if the node is a sub-component
    3. an op that CONSUMES it — a windowed `kv_concat` has no law of its own,
       and the window belongs to the score that reads it
    4. the component's anchor — the main attention score, for every cache whose
       nodes all share one cardinality

Read and stored fall back independently, because an op may describe one side and
not the other, and defaulting the silent side to the other one is how a DSA cache
would come to be billed at its top-K read instead of the full cache it keeps.

A law describes how many positions are kept, never how wide one is. The width
of a row is read off the tensor's named axes (`_per_position_elements`): a
compressor emitting one row per complete window of `m` tokens has a sequence
axis of `floor(s_ctx/m)`, still an axis made of `s_ctx`, so the row is the same
product of the other axes whether the tensor compresses or evicts.

The quant-scale laws
--------------------
A quantised cache costs its data plus its scales, and the scales are not a flat
overhead: a format distributes them per tensor / per channel (static, billed
once — `CachedTensor.static_scale_bytes`) or per token-head, per channel
group, per tile (dynamic, riding each stored position). A dynamic scale is part
of the position it rides: `CachedTensor.position_bytes` is data plus dynamic
scales, and every byte figure — `bytes_at`, stored or read, and everything
summed from it — multiplies that one size, never the data alone. The format is the tensor's own (`CachedTensor.format`, a
`cache_formats.CacheFormat`): the precision profile's choice for its slot
kind, else the format the spec declares, else the model's dtype unscaled.
The structure the scales distribute over is the walk's: each `CachedTensor`
carries its channel axis's extent (`inner`) and the rows beside it (`rows`),
both read off one shape. A windowed tensor's scales then stop at its window
because they multiply its own `positions`, not because a consumer remembered
to cap them.

What is deliberately absent
---------------------------
A `kv_window` summary field, and any other per-layer scalar standing in for the
list. Reading a window off a summary is how a component whose real window was
declared per node came to report a 64K window it does not have, and the summary
cannot represent the hybrid layer at all. The list is the representation; a
consumer that wants one number asks for one AT a length.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import (Any, Dict, FrozenSet, List, Optional,
                    Mapping, Tuple)

import sympy as sp

from .. import cache_families, dims
from ..activation_formats import ACTIVATION_FORMATS
from ..dims import sequence_axes
from .dim import (algebra, eval_dim_expr, is_zero, names_of, same_law,
                  substitute, value_of, ratio, where)
from ..block_synthesis import binding_of
from ..cache_formats import CacheFormat
from ..precision_profile import PrecisionError
from .layer import Layer
from .node import Identified
from ..ops import OP_REGISTRY
from .op_ref import KVPositions
from .program import BlockStage
from .refusal import SpecRefusal
from .values import CACHE_LEN, Extent, product

def positions_at(law: Any, cache_len: int) -> int:
    """A KV law evaluated at one cache length. Failure raises, never falls back."""
    length = int(cache_len)
    if law is None:
        return length
    expr = law if isinstance(law, sp.Basic) else algebra(law)
    result = substitute(expr, CACHE_LEN, length)
    try:
        return int(value_of(result))
    except (TypeError, ValueError) as exc:
        leftover = sorted(names_of(result))
        raise ValueError(
            f"KV law {law!r} still free in {leftover} "
            f"after substituting cache length {length}"
        ) from exc


@dataclass(frozen=True)
class CacheGeometry:
    """Physical tensor geometry and position ownership for cache derivation.

    Cache laws remain properties of the source ``Analysis``.  A production
    projection may provide different, explicitly derived tensor shapes,
    recurrent-state axes (``{state name: (axis, ...)}`` per component, the
    shape of ``Analysis.cache_state``) and a position-owner degree without
    manufacturing a partially local Analysis.
    """

    shapes: Layer[Any]
    state_shapes: Layer[Any]
    positions_over: int = 1
    #: The cached tensors — `(component uid, node uid, name)` — whose
    #: per-rank width this geometry narrowed from the global one: what
    #: tensor parallelism divides of the cache in this deployment, measured
    #: by projecting it (`layout.physical_cache.cache_geometry`).
    tp_divided: FrozenSet[Tuple[int, int, str]] = frozenset()
    # Components whose striped owners reserve a final partial position slot.
    padded_position_components: FrozenSet[int] = frozenset()

    def __post_init__(self) -> None:
        if self.positions_over <= 0:
            raise ValueError(
                "cache position ownership degree must be positive, got "
                f"{self.positions_over}"
            )


@dataclass(frozen=True)
class CachedTensor:
    """One tensor a layer keeps, and how much of it exists at a given length.

    `elements` is per POSITION and per copy of the layer — the node's resolved
    output with the batch and sequence axes divided out. `stored` and `read` are
    expressions in `CACHE_LEN`, and they are separate because they disagree:
    DSA keeps every position so its indexer can score them and reads top-K of
    what it kept.
    """

    node: int
    name: str
    elements: Extent
    #: How this tensor is stored in this deployment: the precision profile's
    #: format for its slot kind, else `declared`, else the model's dtype with
    #: no scale. Set when the cache is derived (`_per_call`); None only on a
    #: tensor still being assembled.
    format: Optional[CacheFormat] = None
    #: The format the spec declares for the slot (the HF forward's), None
    #: when it declares none.
    declared: Optional[CacheFormat] = None
    #: What the slot holds — the cache entry's declared `kind:` (`kv`,
    #: `latent`, `recurrent_state`, … ; `slot_kinds:` in
    #: cache_families.yaml). A deployment's cache precision is chosen per kind.
    kind: Optional[str] = None
    stored: Any = CACHE_LEN
    read: Any = CACHE_LEN
    #: The live (query, key) pairs one step's selection reads of this tensor,
    #: per batch row, at the walk's own length — the `read` law summed over
    #: the step's queries (`KVPositions.pairs`), stated where the law that
    #: speaks for the tensor is a selection (a top-k, a window, a pooled
    #: stream). None when it states none: the step's readers stream the
    #: tensor's `read` positions once. Who reads the pairs rather than the
    #: positions is the reader's fact (`gathered_bytes`).
    pairs: Any = None
    #: The innermost (channel) axis's extent, per rank — what the quant-scale
    #: conventions group over — and the rows beside it: the product of every
    #: other axis that is neither the batch nor the sequence (a K tensor's
    #: heads). `elements` is the flat per-position count and cannot answer
    #: "rows × channel" on its own, so both are read off the one shape at
    #: derivation, by axis identity — a cached node's resolved output, a
    #: recurrent state's declared axes. None when the walk bound no shape, or
    #: bound one symbolically — scale math is asked numerically, and an
    #: untracked split declines it rather than guessing one row.
    inner: Optional[Extent] = None
    rows: Optional[int] = None
    #: How many ranks divide this tensor's POSITION axis. Width sharding is
    #: already reflected in ``elements``; this is only the ownership of the
    #: positions substituted by the consumer. Recurrent state has no position
    #: axis and therefore always keeps the default value of one.
    positions_over: int = 1
    #: A recurrent state: one fixed tensor per request that no graph node
    #: outputs, shaped by its own declared axes. `node` is then its declared
    #: `carrier` — the leaf whose kernel reads and rewrites it each step, and
    #: so the one it is billed and drawn on.
    recurrent: bool = False
    # Invocation-only projected read law; stored/read retain graph identity.
    runtime_read_positions: Any = None
    position_padding: bool = False

    def __post_init__(self) -> None:
        if self.positions_over <= 0:
            raise ValueError(
                "cached tensor position ownership degree must be positive, got "
                f"{self.positions_over}"
            )

    @property
    def storage_law(self) -> Any:
        """Global-equivalent occupied positions, including owner padding."""
        law = self.stored if self.stored is not None else CACHE_LEN
        if self.position_padding and not self.recurrent:
            return sp.ceiling(algebra(law) / self.positions_over) * self.positions_over
        return law

    def positions(self, cache_len: Extent, *, read: bool = False) -> Extent:
        """The law, at a length. Substitution rather than a second evaluation of
        the op's lambda: the lambda closed over the frame's other dims when the
        walk recorded it, and re-running it here would need that frame back."""
        expr = self.read if read else self.storage_law
        if read and self.runtime_read_positions is not None:
            expr = self.runtime_read_positions
        if expr is None:
            return cache_len
        return substitute(expr, CACHE_LEN, cache_len)

    @property
    def dtype(self) -> str:
        """The element dtype the tensor is stored at."""
        return self.resolved_format.dtype

    @property
    def resolved_format(self) -> CacheFormat:
        if self.format is None:
            raise ValueError(f"cached tensor {self.name!r} has no format: "
                             f"read it off a derived cache")
        return self.format

    @property
    def format_name(self) -> Optional[str]:
        """How this tensor is stored, by its registry name
        (`cache_formats.CacheFormat.name`): the scales' organisation, their
        dtype, a write transform and an unquantised tail, none of which the
        element dtype (`dtype`) can say. DeepSeek-V4's FlashMLA record and
        an FP8 index key with one F32 scale are both FP8 E4M3.

        None for a tensor stored plain (no scale, transform or tail): its
        dtype is then the whole of the format. A format the registry does
        not name — a precision profile may spell one out — is given by its
        label (`CacheFormat.label`), which is no registry name, so a
        consumer that keys by name refuses it rather than reading it as some
        other format. The one naming rule for every launch that touches the
        tensor: the leaf that writes it as its output
        (`LeafAccess.output_format`) and every row that reads it or writes
        it beside its output (the measurement key's `kv_format`)."""
        fmt = self.resolved_format
        if fmt == type(fmt).plain(fmt.dtype):
            return None
        return fmt.name or fmt.label()

    @property
    def element_bytes(self) -> float:
        """Mean bytes per element: the format's width, or with an
        unquantised tail both widths over this tensor's row."""
        return self.resolved_format.bytes_per_element(self.inner)

    @property
    def data_position_bytes(self) -> Extent:
        """The element bytes of one stored position of one copy — the data
        half of `position_bytes`, for a display that shows the split."""
        return self.elements * self.element_bytes

    @property
    def scale_position_bytes(self) -> float:
        """The dynamic-scale bytes riding one stored position of one copy —
        the other half of `position_bytes`. Zero for an unscaled format and
        for one whose scales are static (`static_scale_bytes`)."""
        if self.resolved_format.scale is None:
            return 0.0
        dynamic, _static = self.scale_overhead()
        return dynamic * self.resolved_format.scale_bytes

    @property
    def position_bytes(self) -> Extent:
        """What one stored position of one copy occupies: its data and the
        dynamic scales its format writes beside it. The one per-position
        size every cache figure multiplies — storage, a request's bytes,
        the per-token slope, the decode read — so none of them can count
        the data and forget the scales a kernel stores and reads with it."""
        scale = self.scale_position_bytes
        return (self.data_position_bytes + scale if scale
                else self.data_position_bytes)

    @property
    def written_width(self) -> Extent:
        """Bytes per element of this tensor as its writer lands it: one
        position's `position_bytes` — data, an unquantised tail and the
        dynamic scales riding it — over the elements that position holds.
        The width an edge carrying the tensor is priced at, so a step's
        write of it and its decode read are the same size per position."""
        return ratio(self.position_bytes, self.elements,
                     empty=self.element_bytes)

    @property
    def static_scale_bytes(self) -> float:
        """Scale bytes one copy holds once, whatever the length (per-tensor
        and per-channel scales): not a position's, so never in `bytes_at`."""
        if self.resolved_format.scale is None:
            return 0.0
        _dynamic, static = self.scale_overhead()
        return static * self.resolved_format.scale_bytes

    def bytes_at(self, cache_len: Extent, *, read: bool = False) -> Extent:
        """Stored bytes one copy holds (or reads per query) at `cache_len`:
        `position_bytes` at the law's positions, data and dynamic scales
        together. Static scales are `static_scale_bytes`."""
        return (self.position_bytes
                * self.positions(cache_len, read=read)
                / self.positions_over)

    def gathered_bytes(self) -> Optional[Extent]:
        """Bytes one copy hands an attention kernel that gathers its
        selection, per batch row, in one step: `position_bytes` at each live
        pair. A sparse attention kernel reads each query's selected rows by
        index (FlashMLA sparse, vLLM `_gqa_sparse_fwd_kernel`, sglang
        `_sparse_gqa_chunk_prefill` — a grid over queries), so two queries
        selecting one row read it twice. Rows are shared across queries only
        by a prompt several requests share, read once by a shared-prefix
        kernel — a deployment the engine does not model, so for distinct
        requests this is exact and under a shared prefix it is an upper bound.
        At one query a step it is the `read` law's bytes. None when the law
        states no selection."""
        if self.pairs is None:
            return None
        return self.position_bytes * self.pairs / self.positions_over

    @property
    def is_growing(self) -> bool:
        """Whether the storage law still has the length free in it — a window
        grows until its window, a recurrent state never does."""
        return bool(algebra(self.stored).free_symbols & {CACHE_LEN})

    def scale_overhead(self) -> Tuple[float, float]:
        """`(dynamic_per_position, static)` scale COUNTS for one copy of this
        tensor under its own format (`CacheFormat.scale_counts`): zero for an
        unscaled format. Their bytes are `scale_position_bytes` and
        `static_scale_bytes`."""
        return self.resolved_format.scale_counts(
            float(value_of(self.elements)), self.inner, self.rows)


@dataclass(frozen=True)
class Cache:
    """What one attention layer keeps, per copy of that layer.

    Per COPY, like every other value the walk produces: a model total is this
    joined against `Analysis.instances`, not baked in here. `label` and `variant`
    are carried because the KV class decides how the cache shards — an MLA latent
    is replicated across tensor-parallel ranks while a GQA layer's heads split —
    and a consumer asking that question should not have to re-find the component.
    """

    component: int
    tensors: Tuple[CachedTensor, ...] = ()
    label: Optional[str] = None
    variant: Optional[str] = None
    #: Whether this cache survives between decode steps, and so belongs in a
    #: serving memory budget.
    #
    # An encoder's attention allocates a cache inside one forward pass and drops
    # it: a vision or audio tower runs once over an image, not once per token,
    # and billing its cache per request adds a gigabyte to Step-3.7-Flash that
    # no server ever holds. A backbone layer and a speculative head both keep
    # theirs across steps. The distinction is whether the stack attaches to the
    # token stream, which its `inputs:` declares — the same thing
    # `Stack.reads_hidden_state` reads, asked here rather than restated, so a
    # new kind of head is classified once.
    persists: bool = True

    def bytes_at(self, cache_len: Extent, *, read: bool = False) -> Extent:
        """Bytes one copy of this layer holds at `cache_len` — or reads per
        query, which is the same sum under the other law.

        `cache_len` is the GLOBAL context length, always: it is what a request
        asked for, not what a rank happens to hold, and asking the caller to
        pre-divide it would put the sequence shard somewhere every caller has to
        remember. Each tensor applies its own ``positions_over`` exactly once;
        a recurrent state in the same component has no position axis to divide.
        """
        return sum(
            (tensor.bytes_at(cache_len, read=read)
             for tensor in self.tensors),
            0,
        )

    def groups(self) -> List[Tuple[Any, Extent, List[str]]]:
        """`(positions law, bytes per position, names)`, merged where the law
        and the width agree.

        The merge key is the law and the byte count, never the names: K and V
        coincide on most models and are two entries with different extents on
        the ones where they do not, so assuming the pair would report one of
        them twice.
        """
        buckets: Dict[Tuple[str, Any], Tuple[Any, Extent, List[str]]] = {}
        for tensor in self.tensors:
            per_position = tensor.position_bytes / tensor.positions_over
            key = (sp.srepr(algebra(tensor.storage_law)), per_position)
            if key in buckets:
                law, total, names = buckets[key]
                buckets[key] = (law, total + per_position,
                                [*names, tensor.name])
            else:
                buckets[key] = (tensor.storage_law, per_position, [tensor.name])
        return list(buckets.values())


# -- derivation ---------------------------------------------------------------


def derive(
    analysis: Any,
    *,
    geometry: Optional[CacheGeometry] = None,
) -> Layer[Cache]:
    """`Layer[Cache]` over every component that declares one.

    Keyed on the COMPONENT's uid rather than on the cached nodes', because the
    cache is a property of the attention layer: two of its nodes share one
    window, a sub-component's cache belongs to the parent that invoked it, and a
    recurrent state has no node at all.
    """
    entries, absorbed = _per_call(analysis, geometry)
    # A sub-component's cache belongs to the layer that invoked it and is
    # already among that layer's tensors. Leaving it keyed on its own uid as
    # well would double every byte of it — and the sub-caches are the sparse
    # indexers, which is where an unexplained factor of two is least visible.
    return Layer(analysis.program, {uid: entry for uid, entry in entries.items()
                                    if uid not in absorbed})


def by_call(analysis: Any) -> Layer[Cache]:
    """What each component call keeps, sub-components included.

    The entries `derive` counts from, before it keys a sub-component's cache on
    the layer that invoked it: a question about the sub-component itself — what
    DSv4's indexer keeps — is asked of its own call, and reads the same tensors
    named from it.
    """
    entries, _absorbed = _per_call(analysis, None)
    return Layer(analysis.program, entries)


def _per_call(
    analysis: Any,
    geometry: Optional[CacheGeometry],
) -> Tuple[Dict[int, Cache], set]:
    """Every component call's own cache, and the calls whose cache a parent
    already holds among its tensors."""
    program = analysis.program
    shapes = analysis.shapes if geometry is None else geometry.shapes
    states = analysis.cache_state if geometry is None else geometry.state_shapes
    if geometry is not None:
        if shapes.program is not program or states.program is not program:
            raise ValueError(
                "cache geometry belongs to a different Program than its Analysis"
            )
        declared = analysis.cache_state.values
        if (set(states.values) != set(declared)
                or any(set(states.values[uid]) != set(declared[uid])
                       for uid in declared)):
            raise ValueError(
                "cache geometry must project every recurrent state exactly once"
            )
    positions_over = (
        _sequence_shard(analysis)
        if geometry is None else geometry.positions_over
    )
    out: Dict[int, Cache] = {}
    absorbed: set = set()
    for node in program.language_nodes():
        if not node.children:
            continue
        entry = _of_component(
            analysis,
            node,
            absorbed,
            shapes=shapes,
            state_shapes=states,
            positions_over=positions_over,
        )
        if entry is not None and entry.tensors:
            out[node.uid] = entry
    precision = analysis.resolved_precision
    for uid, entry in out.items():
        node = program.by_uid(uid)
        stack = program.stack_of(node)
        out[uid] = replace(entry, tensors=tuple(
            replace(t, format=bind_format(
                precision.cache_format(t.kind, stack, t.declared),
                analysis, node), position_padding=(geometry is not None
                    and uid in geometry.padded_position_components and not t.recurrent))
            for t in entry.tensors))
    return out, absorbed


def bind_format(fmt: CacheFormat, analysis: Any,
                node: Identified) -> CacheFormat:
    """`fmt` with an unquantised tail's channel count resolved in the frame
    the walk bound for the component that keeps the tensor."""
    if fmt.tail is None or not isinstance(fmt.tail.channels, str):
        return fmt
    frame = analysis.node_dims.get(node) or {}
    return fmt.bound(int(value_of(eval_dim_expr(fmt.tail.channels, frame))))


def identity(program: Any, node: Any) -> Optional[Cache]:
    """Family and compute flavor of this call, without geometry.

    `derive` fills tensors. Classification reads `label` / `variant` here.
    Those two were copied onto the IR node at emit; this does not
    `get_block` / `program.component`.
    """
    del program
    label = node.kv_label
    variant = node.attn_variant
    if label is None and variant is None:
        return None
    return Cache(
        component=int(node.uid),
        label=label,
        variant=variant,
    )


def cache_of_component(analysis: Any, name: str) -> Optional[Cache]:
    """This component's `Cache` on an already-isolated analysis, or None.

    Empty when the isolated program has no cache entry for `name`. Does not
    peek the recipe graph to decide emptiness.
    """
    call = next((node for node in analysis.program.nodes()
                 if node.children
                 and binding_of(analysis.program.spec, node.op, name)), None)
    if call is None:
        return None
    return derive(analysis).get(call)


def of_component(analysis: Any, name: str) -> Tuple[CachedTensor, ...]:
    """Cached tensors of one component on an already-isolated analysis.

    A full-model `derive` absorbs a library indexer into the attention that
    invoked it. Axioms judge the component's own `kv_cached:` declaration, so
    the caller runs it as a one-stage model — `analyze_isolated`, the same
    isolation `project_block` uses — whose block calls it once, and this is
    `derive`'s entry for that call. Capacity still reads `derive` on the real
    walk; this is that walk, on one component.
    """
    program = analysis.program
    call = next((node for node in program.nodes()
                 if node.children
                 and binding_of(program.spec, node.op, name)), None)
    if call is None:
        raise ValueError(
            f"isolating {name!r} produced a program that does not call it; "
            f"its cache cannot be read and must not be counted as empty")
    if not call.kv_cached and not call.kv_states:
        return ()
    entry = derive(analysis).get(call)
    return entry.tensors if entry is not None else ()


def _state_carriers(component: Identified) -> Dict[str, Any]:
    """State name → the child node declared to carry it."""
    by_name = {child.name: child for child in component.children}
    return {e.state: by_name[e.carrier] for e in component.kv_states}


def state_carrier_uids(analysis: Any) -> FrozenSet[int]:
    """The nodes carrying a recurrent state, by uid. A kernel rewrite that
    absorbs one must keep its shape: the state's per-rank axes are projected
    through it, each following the carrier's kernel axis of the same
    identity (`layout.physical_cache`)."""
    program = analysis.program
    return frozenset(
        carrier.uid
        for uid in analysis.cache_state.values
        for carrier in _state_carriers(program.by_uid(uid)).values())


def cached_node_uids(program: Any) -> FrozenSet[int]:
    """The producing tensors of every `cache.entries` declaration, by uid.

    Same resolution `derive` uses for `CachedTensor.node`: a declared child
    that is itself a cached component contributes *its* producing tensors
    (the CSA `indexer` keeps compressed keys inside; its output is a list
    of positions and is not a cache); a child with no cache of its own is
    the tensor, even when that child is a wrapper — GQA publishes V as
    `v_split_heads`, and that is the name `attn_apply` reads. Callers
    that need "which nodes are the cache" — the write endpoint, traffic
    seeds, fusion preserve — read this set, not the declaration names.
    The names are lookup keys into the graph.

    A graph rewrite must keep these nodes shape-readable: fused qk-rope
    may swallow `k_pe_rope`, but the cache the fused kernel writes is
    unchanged, and `derive` still has to bill it.
    """
    out: set = set()
    for node in program.language_nodes():
        if node.children:
            out.update(_producing_uids(program, node))
    return frozenset(out)


def _producing_uids(program: Any, node: Any) -> set:
    """Uids `derive` would put on `CachedTensor.node` for this component.

    Recurses through a declared child that has its own cache; stops at a
    child that does not. The declared child is the identity even when it
    is a wrapper — geometry is read off its output leaf elsewhere. Must
    stay in lockstep with `_of_component`'s inner/declared split — the
    identity test in `test_eval_cache.py` is the join.
    """
    if not node.children:
        return set()
    declared = list(node.kv_cached)
    if not declared:
        return set()
    by_name = {c.name: c for c in node.children}
    out: set = set()
    for name in declared:
        child = by_name.get(name)
        if child is None:
            continue
        inner = _producing_uids(program, child)
        if inner:
            out.update(inner)
        else:
            out.add(child.uid)
    return out


def _of_component(
    analysis: Any,
    node: Identified,
    absorbed: Optional[set] = None,
    *,
    shapes: Layer[Any],
    state_shapes: Layer[Any],
    positions_over: int,
) -> Optional[Cache]:
    program = analysis.program
    declared = list(node.kv_cached)
    state = state_shapes.get(node)
    if not declared and state is None:
        return None

    entries = {entry.name: entry for entry in node.kv_entries}
    overrides = _declared_formats(analysis, entries.values())
    by_name = {c.name: c for c in node.children}
    anchor = _anchor(analysis, node)
    tensors: List[CachedTensor] = []

    for name in declared:
        child = by_name.get(name)
        if child is None:
            raise MissingCachedNode(
                f"{node.op!r} declares `cache.entries`: [{name!r}] and has no "
                f"such node — the declaration names the graph, so one of the "
                f"two is a typo. Present: {sorted(by_name)}"
            )
        # A cached entry that is itself a component with a cache contributes
        # that component's own tensors: DSv4's indexer keeps a separate
        # compressed-key buffer that is reachable from nowhere else, since the
        # indexer's only output is a list of positions.
        inner = _of_component(
            analysis,
            child,
            absorbed,
            shapes=shapes,
            state_shapes=state_shapes,
            positions_over=positions_over,
        )
        if inner is not None and inner.tensors:
            if absorbed is not None:
                absorbed.add(child.uid)
            # A format the caller wrote on this entry is the default for every
            # tensor of the sub-cache that does not carry its own: the
            # indexer stores its compressor's latent at fp8, while the
            # compressor's fp32 states keep theirs. The entry's kind is what
            # the sub-cache's per-token tensors hold HERE — the compressor's
            # pooled rows are an index key under the indexer — and its states
            # keep their own kinds.
            tensors.extend(
                CachedTensor(node=t.node, name=f"{name}.{t.name}",
                             elements=t.elements,
                             declared=t.declared if t.declared is not None
                             else overrides.get(name),
                             kind=t.kind if t.recurrent
                             else entries[name].kind,
                             stored=t.stored, read=t.read, pairs=t.pairs,
                             inner=t.inner,
                             rows=t.rows, positions_over=t.positions_over,
                             recurrent=t.recurrent)
                for t in inner.tensors)
            continue
        law = _law_for(analysis, node, child, anchor)
        # Identity is the declared child. A wrapper has no axes of its
        # own — `v_split_heads` publishes `permute` — so the row width
        # and the channel split are read off the leaf.
        geometry = child
        if child.children:
            leaf, _port = program.output_leaf(child)
            if shapes.get(leaf) is not None:
                geometry = leaf
        elements = _per_position_elements(geometry, shapes=shapes)
        if elements is None:
            raise MissingCachedNode(
                f"{node.op}.{name} (op={child.op!r}) is declared cached and the "
                f"walk resolved no shape for it, so there is no row to bill. "
                f"Skipping it would drop a whole tensor out of the cache and "
                f"leave a model that still fits in memory."
            )
        inner, rows = _row_split(shapes.get(geometry).out)
        tensors.append(CachedTensor(
            node=child.uid, name=name, elements=elements,
            declared=overrides.get(name),
            kind=entries[name].kind,
            stored=law.stored if law.stored is not None else CACHE_LEN,
            read=law.read if law.read is not None else CACHE_LEN,
            pairs=law.pairs,
            inner=inner, rows=rows,
            positions_over=positions_over,
        ))

    carriers = _state_carriers(node)
    reads = analysis.cache_state_reads.values.get(node.uid) or {}
    for name, axes in (state or {}).items():
        # A recurrent state is billed once per request and does not grow, which
        # is the whole of what makes a linear-attention layer cheap at long
        # context. No output in the graph IS the state, so it is named for
        # what it is and kept at the node declared to carry it. Its size and
        # its rows × innermost axis come from the axes it declares.
        inner, rows = _row_split(axes)
        tensors.append(CachedTensor(
            node=carriers[name].uid, name=name,
            elements=product(axes),
            declared=overrides.get(name),
            kind=entries[name].kind,
            stored=sp.Integer(1), read=reads.get(name, sp.Integer(1)),
            inner=inner, rows=rows,
            positions_over=1,
            recurrent=True,
        ))

    return Cache(
        component=node.uid, tensors=tuple(tensors),
        label=node.kv_label,
        variant=node.attn_variant,
        persists=_persists(program, node),
    )


def _declared_formats(analysis: Any, entries: Any
                      ) -> Dict[str, CacheFormat]:
    """Slot name → the format its cache entry declares, resolved against
    this checkpoint (`CacheEntry.declared_format`): a named or spelled-out
    `format:`, a dtype as written, a config binding to the field's value,
    `?? model` to the precision the model states for its own tensors.
    Slots that declare none are absent: they ship at the model's dtype."""
    program = analysis.program
    config = program.resolved.config

    def model_dtype() -> Optional[str]:
        return analysis.resolved_precision.model_dtype

    out: Dict[str, CacheFormat] = {}
    for entry in entries:
        fmt = entry.declared_format(config, model_dtype)
        if fmt is not None:
            out[entry.name] = fmt
    return out


def _row_split(axes: Tuple[Any, ...]
               ) -> Tuple[Optional[int], Optional[int]]:
    """`(inner, rows)` of one cached tensor, per rank — what the quant-scale
    conventions group over: the innermost (channel) axis's extent, and the
    product of every other axis that is neither the batch nor the sequence.

    Both from the one shape — a cached node's resolved output, a recurrent
    state's declared axes — by axis identity and by multiplication: a row
    count divided back out of the flat element count is an answer for
    whatever shape the division assumed. `(None, None)` when there is no
    shape, or it is bound symbolically: scale math is only ever asked
    numerically, so a symbolic extent is recorded as untracked rather than
    coerced."""
    if not axes:
        return None, None
    token = dims.step_token_axes()
    rest = [axis for axis in axes[:-1] if not names_of(axis) & token]
    try:
        inner = int(value_of(axes[-1]))
        rows = 1
        for axis in rest:
            rows *= int(value_of(axis))
    except (TypeError, ValueError):
        return None, None
    return inner, rows


def _sequence_shard(analysis: Any) -> int:
    """How many ranks the deployment spreads cache positions over.

    Prefer the layout rule's cache family — that is the DCP group the
    walk actually carried. No rule, or no cache family, is one rank.
    """
    rule = analysis.rule
    family = None if rule is None else rule.cache_family
    if family is None:
        return 1
    return max(1, int(family.group_size))


def _persists(program: Any, node: Identified) -> bool:
    """Whether the stack this component sits in keeps its cache between steps.

    The backbone does. A head that predicts from the backbone's hidden state
    does, and `Stack.reads_hidden_state` is already the question "does it read
    that state" — answered by whether the stack names a node it attaches to.
    An encoder reads an image or a waveform, names nothing, and so neither
    reads that state nor persists anything.
    """
    name = program.stack_of(node)
    for stack in program.stacks:
        if stack.name == name:
            return not stack.is_aux or stack.reads_hidden_state
    return True


# -- who reads and writes the cache ------------------------------------------


def is_view(node: Identified) -> bool:
    """Whether the op re-indexes rather than rewrites. Declared, not guessed.

    A COMPONENT counts when every node inside it does — `head_broadcast` is a
    `broadcast` and a `transpose`, `head_merge` a `transpose` and a
    `view_merge`. A ref may point at either the wrapper or its leaves, and a
    walk that stops at the wrapper has stopped at a name rather than at a
    tensor. `is_view_name` is the one classifier.
    """
    from ..op_names import is_view_name

    return is_view_name(node.op or "")


def main_port(port: Optional[str]) -> str:
    """A ref's port with the main output spelled one way ("" for `out`)."""
    return "" if port in (None, "", "out") else port


class Edges:
    """What each node input of a graph reads, for the cache and traffic
    derivations.

    An input with an edge (`src`) reads the node the edge names: the graph's
    own edge — the tree's, or one a rewrite drew (a fused node's input, a
    collective spliced in before its reader) — except an edge onto an
    encoder's handed-over output, which the walk reads as free. An input
    that names a tensor
    — a component formal, a stage, a stream — has no edge in the graph; it
    reads what the walk bound to it (`Analysis.input_reference`): the
    producing nodes over the reading stage's iterations, in iteration
    order. Across a rewrite, a fused node's named input is the input of the
    member it carries (`Rewrite.inputs`), and a producer the rewrite
    absorbed is the node that absorbed it. Nothing is resolved by name
    here."""

    def __init__(self, analysis: Any, program: Any = None,
                 rewrite: Any = None) -> None:
        self.analysis = analysis
        self.program = analysis.program if program is None else program
        self.rewrite = analysis.rewrite if rewrite is None else rewrite
        self._table = analysis.reference_table
        self._walked_uids = frozenset(
            node.uid for node in self._table.program.nodes())
        self._absorbed_by = {
            old: new for new, olds in self.rewrite.origins.items()
            for old in olds if old in self.rewrite.dropped}

    def covers(self, node: Identified) -> bool:
        """Whether `node` runs in this analysis: the walk evaluated it, or a
        rewrite made it (a fused node, a collective spliced in). A node of
        the walked program the walk did not evaluate — an encoder's, in the
        language model's step — reads nothing here."""
        return (self._table.covers(node.uid)
                or node.uid not in self._walked_uids)

    def sources(self, reader: Identified, index: int
                ) -> Tuple[Identified, ...]:
        """The nodes input `index` of `reader` reads."""
        seen: Dict[int, Identified] = {}
        for node, _port in self.reads(reader, index):
            seen.setdefault(node.uid, node)
        return tuple(seen.values())

    def reads(self, reader: Identified, index: int
              ) -> Tuple[Tuple[Identified, str], ...]:
        """`(node, port)` for each tensor input `index` of `reader` reads —
        `sources` with the producer's output port ("" for the main one),
        for a question about one output of a node that publishes several."""
        ref = tuple(reader.inputs_obj or ())[index]
        if ref.src is not None:
            # An encoder's output the language model reads is handed over
            # (`Program.handed_exits`): nothing this walk produced it.
            if ref.src.uid in self._table.program.handed_exits:
                return ()
            return ((ref.src, main_port(ref.port)),)
        key = (reader.uid, index)
        if not self._table.covers(reader.uid):
            if key not in self.rewrite.inputs:
                raise ValueError(
                    f"{reader.display_path}: input {index} names "
                    f"{ref.external_sid!r}, and neither the walk nor a "
                    f"rewrite bound it")
            key = self.rewrite.inputs[key]
        reference = self._table.at(*key)
        found_reads: Dict[Tuple[int, str], None] = {}
        for _iteration, found in reference.table:
            for source in found:
                uid = self._absorbed_by.get(source.occurrence.uid,
                                            source.occurrence.uid)
                found_reads[(uid, main_port(source.port))] = None
        return tuple((self.program.by_uid(uid), port)
                     for uid, port in found_reads)

    def absorbed(self) -> Tuple[Identified, ...]:
        """The walked nodes a rewrite absorbed into a launch, in walk order.
        Not nodes of the rewritten graph, but each still names the tensor it
        made, and an absorbed member whose output the launch exposes is read
        by name (`FusedNode.output_ports`): a reader's edge onto it is an
        edge onto that tensor."""
        walked = self._table.program
        return tuple(node for node in walked.nodes()
                     if node.uid in self.rewrite.dropped)

    def source(self, reader: Identified, index: int) -> Optional[Identified]:
        """The one node input `index` of `reader` prices as: the first of
        `sources` — a stream's producers over the reading layers keep one
        shape, layout and cache law (`Evaluator._stream_value`), so any one
        stands for each read."""
        found = self.sources(reader, index)
        return found[0] if found else None


def context_sources(edges: Edges, seeds: Optional[FrozenSet[int]] = None
                    ) -> Dict[int, "ContextCoverage"]:
    """Nodes whose output carries authoritative context, each mapped to its
    branch-preserving coverage: the `kv_cached` seeds, live KV markers, their
    re-arrangements, a positional sequence union, a co-positional feature join,
    and a collective that moves any of those between ranks.

    This is the identity `traffic`'s pricing rule and `kv_access` both turn
    on, and it is a
    PEDIGREE, not arithmetic: an edge either reads this step's activations
    or it reads the cache, and which one is decided by where the tensor
    came from — never by comparing sizes. The closure exists because the
    declaration sits on the published child (`kv_a_layernorm`,
    `k_pe_rope`, `v_split_heads`) while the attention op may read them
    through a broadcast-and-concat chain; the chain re-indexes
    cache-resident data without making anything new, so membership
    propagates through it.

    The ancestry is carried branch by branch rather than flattened because a
    positional union can be cache on one axis-extent and live on the next.
    Membership answers "which branches reach the cache"; each branch's rows
    answer how much of the reader's authoritative span belongs to that law;
    and the seed set answers WHICH cache replaces those rows. The three are
    asked at opposite ends of the same chain, one at the union the reader
    points at, the other at the `k_pe_rope` five hops upstream.

    `seeds` defaults to `cached_node_uids(program)`. `kv_access` passes the
    tensors of the caches it resolved, so a reader is traced to the same
    references it records. A ref naming a tensor instead of a `src` is
    traced to the producers the walk bound to it (`Edges`), so a view of a
    cached tensor another layer published is traced as well. Every branch's
    row count is the evaluator's `KVLength`; this function never re-derives
    a cache law from an op name.
    """
    from ..op_names import is_join_name

    program = edges.program
    if seeds is None:
        seeds = cached_node_uids(program)
    if not seeds:
        return {}
    pedigree: Dict[int, ContextCoverage] = {
        uid: ContextCoverage((
            ContextBranch(seeds=frozenset((uid,)),
                          # The evaluator's `KVLength` is authoritative when
                          # it has already stated one (a pooled latent); a
                          # plain projection leaves this unstated rather than
                          # inventing a cache law from its fresh shape.
                          rows=_kv_rows(edges.analysis, program.by_uid(uid)),
                          producer=uid, law_node=uid),), disjoint=True)
        for uid in seeds}
    # Every node, wrappers included: a reader's ref may point at the
    # `head_broadcast` COMPONENT rather than its leaf children (`_fanout`
    # crosses the same wrappers by op name), and the admission rule below
    # already limits membership to the re-arranging ops. Across a rewrite,
    # the members a launch absorbed too: a launch that joins two cached
    # tensors and exposes the join (Kimi-K3's MLA decode writes the latent
    # row `[kv_latent | k_pe]` beside its query) hands its readers the same
    # cache-resident rows the unfused join did, and they read it by the
    # member's name.
    nodes = [node for node in program.nodes() if edges.covers(node)]
    nodes.extend(edges.absorbed())
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node.uid in pedigree:
                continue
            if node.children:
                # A ref may name the wrapper rather than the tensor it
                # publishes, and the wrapper IS that tensor: DeepSeek-V4's
                # attention reads `compressor`, whose cache is declared on
                # its `pooled_out` leaf.
                leaf, _port = program.output_leaf(node)
                if leaf.uid in pedigree:
                    pedigree[node.uid] = pedigree[leaf.uid]
                    changed = True
                    continue
            refs = tuple(node.inputs_obj or ())
            sources = [edges.sources(node, index)
                       for index in range(len(refs))]
            if not sources or not all(sources):
                continue
            spec = OP_REGISTRY.get(node.op or "")
            if spec is not None and spec.kv_stream_union:
                if not all(s.uid in pedigree for found in sources
                           for s in found):
                    continue
                branches: List[ContextBranch] = []
                # Positional concatenation, in declared input order. A source
                # that resolved to several stream producers is co-positional
                # (this call reads one of them), not another sequence branch.
                for found in sources:
                    if len(found) == 1:
                        branches.extend(pedigree[found[0].uid].branches)
                        continue
                    rows = _one_row(pedigree[found[0].uid])
                    branches.extend(
                        replace(branch, rows=rows)
                        for source in found for branch in
                        pedigree[source.uid].branches)
                coverage = ContextCoverage(tuple(branches), disjoint=True)
            elif is_join_name(node.op or ""):
                if not all(s.uid in pedigree for found in sources
                           for s in found):
                    continue
                # A feature JOIN reads co-positional branches: each input
                # describes the same rows, so seeds (and feature ownership)
                # stay separate while row counts do not add.
                rows = _one_row(pedigree[sources[0][0].uid])
                coverage = ContextCoverage(tuple(
                    replace(branch, rows=rows)
                    for found in sources for source in found
                    for branch in pedigree[source.uid].branches),
                    disjoint=False)
            elif is_view(node) or _restates_kv(node) or _moves(node):
                # Re-indexing of ONE tensor. Only the first source is the
                # tensor; a slice's bounds are not cache reads however they
                # were computed. A stream's producers are its layers' one
                # tensor each, so all of them. A KV stream marker over cached
                # rows (`kv_stream_marker`: a pooled latent, a window) states
                # their law and moves nothing, so its output is those rows:
                # DeepSeek-V4.1's CSA2 reads its compressed KV through one
                # and its window through another, and the score past them
                # is the cache's reader. A collective that MOVES its operand
                # (`_moves`) is the same rows on another rank: a prefill
                # context-parallel rank all-gathers the latent it cached
                # before the attention reads it.
                landed_sources = []
                for source in sources[0]:
                    landed = materialized(source, seeds, edges=edges,
                                          follow_moves=True)
                    if landed is not None and landed.children:
                        # A component's output leaf is the tensor it lands;
                        # the wrapper's UID is only namespace for that leaf.
                        landed, _port = program.output_leaf(landed)
                    landed_sources.append(landed)
                if _restates_kv(node) and not any(
                        landed is not None and landed.uid in seeds
                        for landed in landed_sources):
                    # A live marker has no cache seed, but its rows are still
                    # first-class context: a mixed union must preserve this
                    # branch and traffic must bill it as activation.
                    # The marker states the branch law; the rows themselves
                    # were landed by its input. Structural checks compare that
                    # law with the input's actual output, not with the
                    # zero-cost marker that repeats it.
                    live_producer = next(
                        (landed.uid for landed in landed_sources
                         if landed is not None), None)
                    coverage = ContextCoverage((
                        ContextBranch(seeds=frozenset(),
                                      rows=_kv_rows(edges.analysis, node),
                                      producer=live_producer,
                                      law_node=node.uid),), disjoint=True)
                    pedigree[node.uid] = coverage
                    changed = True
                    continue
                if not all(s.uid in pedigree for s in sources[0]):
                    continue
                found = sources[0]
                if len(found) == 1:
                    coverage = pedigree[found[0].uid]
                    if _restates_kv(node):
                        rows = _kv_rows(edges.analysis, node)
                        coverage = ContextCoverage(tuple(
                            # Keep the branch's landed producer: a marker's
                            # output shape can already state the reader's
                            # cache span, which is not its fresh-row count.
                            replace(branch, rows=rows, law_node=node.uid)
                            for branch in coverage.branches), disjoint=True)
                else:
                    # The several producers an inner stream names all hold
                    # this call's rows; their branches describe alternative
                    # producers, not additional positions.
                    rows = _one_row(pedigree[found[0].uid])
                    coverage = ContextCoverage(tuple(
                        replace(branch, rows=rows)
                        for source in found
                        for branch in pedigree[source.uid].branches),
                        disjoint=False)
            else:
                continue
            pedigree[node.uid] = coverage
            changed = True
    return pedigree


def _kv_rows(analysis: Any, node: Identified) -> Any:
    """The authoritative rows the evaluator says this tensor carries."""
    length = analysis.kv_lengths.get(node.uid)
    return None if length is None else length.read


def _one_row(coverage: ContextCoverage) -> Any:
    """The one row count a set of co-positional branches shares."""
    if not coverage.branches:
        return 0
    rows = coverage.branches[0].rows
    if any(branch.rows != rows for branch in coverage.branches[1:]):
        raise ValueError(
            "co-positional context branches declare different row counts: "
            + ", ".join(repr(branch.rows) for branch in coverage.branches))
    return rows


def producer_fresh_rows(node: Optional[Identified], shapes: Any) -> Any:
    """The fresh sequence rows a producer's actual output lands.

    Row identity is read from the dimension registry (the one classification
    of what a symbol counts), not from an op name. This is deliberately a
    producer fact for structural checks and branch clipping; authoritative
    reader coverage comes from `KVLength`.
    """
    if node is None:
        return None
    shape = shapes.get(node.uid)
    axes = tuple(extent for extent in (shape.out if shape else ())
                 if names_of(extent) & sequence_axes())
    if not axes:
        return None
    if len(axes) > 1:
        raise ValueError(
            f"{node.display_path}: a context producer has multiple sequence "
            f"axes {axes!r}, so its fresh row count is not one fact")
    return axes[0]


def covered_context_source(node: Optional[Identified],
                           coverage: Mapping[int, ContextCoverage],
                           edges: Edges) -> Optional[Identified]:
    """The nearest actual source that carries context coverage.

    Unlike `materialized`, this stops at a KV stream marker. A marker may be a
    view of a cache seed, but its `KVLength` — not the seed's fresh shape — is
    the branch fact a mixed union needs.
    """
    seen: set = set()
    while node is not None and node.uid not in seen:
        seen.add(node.uid)
        if node.uid in coverage:
            return node
        if node.children:
            node, _port = edges.program.output_leaf(node)
            continue
        # Do not walk backwards through a kernel: coverage is a propagation
        # fact, and a reader of a newly computed tensor does not inherit the
        # cache that tensor's operand once came from.
        if not (is_view(node) or _restates_kv(node) or _moves(node)):
            return None
        node = edges.source(node, 0) if node.inputs_obj else None
    return None


@dataclass(frozen=True)
class ContextBranch:
    """One positional branch of a context stream.

    `seeds` is empty for a live branch. `producer` is where the branch's fresh
    rows land (the seed, marker, or collective input that made the branch), so
    structural checks can compare declared rows with actual producer rows
    without billing anything. `law_node` is the stream marker that stated
    those rows (and, for a gathered live marker, its pairs); it is the
    producer unless a view restated the branch.
    """

    seeds: FrozenSet[int]
    rows: Any
    producer: Optional[int] = None
    law_node: Optional[int] = None


@dataclass(frozen=True)
class ContextCoverage:
    """Branch-preserving coverage of an authoritative context stream.

    `disjoint` is True only for a sequence union, where branch rows add. A
    feature join is co-positional: its branches describe different feature
    parts of the same rows, so adding their row counts would count positions
    once per feature.
    """

    branches: Tuple[ContextBranch, ...]
    disjoint: bool

    @property
    def seeds(self) -> FrozenSet[int]:
        return frozenset(
            seed for branch in self.branches for seed in branch.seeds)

    @property
    def mixed(self) -> bool:
        """A sequence union of cache and live branches: some branch holds a
        seed and some branch holds none, so the reader's span splits between
        cache laws and this step's activations."""
        return bool(self.disjoint and self.seeds
                    and any(not branch.seeds for branch in self.branches))

    @property
    def rows(self) -> Any:
        if not self.branches:
            return 0
        if any(branch.rows is None for branch in self.branches):
            return None
        return (sum(branch.rows for branch in self.branches)
                if self.disjoint else self.branches[0].rows)

@dataclass(frozen=True)
class ContextSpanGap:
    """One authoritative context branch a producer's fresh rows do not cover.

    A seeded branch is covered only by the read law of one of its persistent
    seeds, not by the seed's name alone. A live branch has nowhere else to
    come from and must be fresh.
    """

    reader: int
    reader_display: str
    input_index: int
    producer: int
    producer_display: str
    rows: Any
    fresh_rows: Any
    uncovered: Any
    seeds: FrozenSet[int] = frozenset()


def _input_rows(dims: Any) -> Any:
    axes = tuple(extent for extent in (dims or ())
                 if names_of(extent) & sequence_axes())
    if len(axes) > 1:
        raise ValueError(
            f"an input carries multiple sequence axes {axes!r}, so its "
            "authoritative row count is not one fact")
    return axes[0] if axes else None


def context_span_gaps(analysis: Any) -> Tuple[ContextSpanGap, ...]:
    """Uncovered excess in every leaf's authoritative K/V input spans.

    This is a structural fact from the one walk: rewritten edges, actual
    shapes, branch coverage, and the evaluator's `KVLength`. It is not a
    traffic total and does not use `KVAccess.reads` or `key_slots` as a proxy.
    """
    program = analysis.program
    edges = Edges(analysis)
    coverage = context_sources(edges)
    cached = cached_node_uids(program)
    laws: Dict[int, Any] = {}
    cache_len = workload_dim(analysis, "s_kv")
    for _holder, _entry, tensor in resident(analysis).values():
        positions = ratio(tensor.positions(cache_len, read=True),
                          tensor.positions_over, empty=0)
        laws[tensor.node] = (positions if tensor.node not in laws
                             else where(sp.Gt, (positions,
                                                laws[tensor.node]),
                                        positions, laws[tensor.node]))

    def _gap(reader: Identified, index: int, producer: Identified,
             rows: Any, fresh: Any,
             seeds: FrozenSet[int] = frozenset()) -> Optional[ContextSpanGap]:
        if rows is None:
            return None
        covered = sum(laws.get(seed, 0) for seed in seeds)
        fresh_rows = value_of(fresh) if fresh is not None else 0
        cached_rows = value_of(covered)
        # Fresh and cached rows cover different positions: a chunked-prefill
        # SWA reader attends its fresh suffix and the cached `window - 1`
        # rows before it. Taking the wider of the two would call that
        # intersection-shaped union a gap; the sum is the available-row upper
        # bound, so it cannot hide a genuinely missing live or cache row.
        available = fresh_rows + cached_rows
        excess = value_of(rows) - available
        uncovered = where(sp.Gt, (excess, 0), excess, 0)
        # WHETHER THIS READER IS COVERED IS ANSWERED OR REFUSED, never read
        # at the point the walk happens to hold. It is a magnitude question
        # — is nothing left over? — so `is_zero` asks it: a number answers,
        # a law in the workload's symbols is refused (`Undecided`).
        #
        # The refusal is the whole of this choice, so: `uncovered` is
        # deliberately the cases it is, and those cases ARE the per-point
        # answer, which is why the record carries them — but what is decided
        # here is whether there is a RECORD, and the tuple this returns is
        # the walk's, read unchanged at every batch and context, so `where`
        # cannot write "no record at these points and one at those". Taking
        # the identity instead (`same_law(uncovered, 0)`, which is what
        # comparing the value halves came to) keeps a record wherever SymPy
        # merely cannot prove coverage: a symbolic prefill of DeepSeek-V4.1
        # reported 22 gaps and a Gemma-2 chunked prefill 2, every one
        # reading `uncovered = 0`, where the walk made at that same point
        # reported none — a law disagreeing with every point it answers.
        # And `require` is not this function's to call: coverage is a fact
        # about the spec and the rewritten graph, not a condition on the
        # admitted workload, and a post-walk reader may not add conditions
        # to the domain of a walk that is already over.
        #
        # So a symbolic walk is told this reader's coverage is not decidable
        # from the law rather than handed a list that is wrong. Making it
        # decidable is the spec's or the engine's job: state the covering
        # rows so `rows - (fresh + cached)` settles.
        if is_zero(uncovered):
            return None
        return ContextSpanGap(
            reader=reader.uid, reader_display=reader.display_path,
            input_index=index, producer=producer.uid,
            producer_display=producer.display_path, rows=rows,
            fresh_rows=fresh, uncovered=uncovered, seeds=seeds)

    out: List[ContextSpanGap] = []
    for reader in program.nodes():
        if reader.children or not edges.covers(reader):
            continue
        shape = analysis.shapes.get(reader.uid)
        for index, refs in enumerate(() if shape is None else shape.ins):
            source = edges.source(reader, index)
            if source is None:
                continue
            covered = covered_context_source(source, coverage, edges)
            found = (coverage.get(covered.uid) if covered is not None
                     else None)
            if found is None:
                roles = _roles(reader)
                if (index >= len(roles) or roles[index] not in ("k", "v")):
                    continue
            rows = _input_rows(refs)
            if rows is None:
                continue
            if found is not None and found.disjoint and all(
                    branch.rows is not None for branch in found.branches):
                for branch in found.branches:
                    producer = program.by_uid(branch.producer)
                    out.append(_gap(
                        reader, index, producer, branch.rows,
                        producer_fresh_rows(producer, analysis.shapes),
                        branch.seeds))
                continue
            producer = (program.by_uid(found.branches[0].producer)
                        if found is not None and found.branches
                        else materialized(source, cached, edges=edges))
            out.append(_gap(
                reader, index, producer,
                found.rows if found is not None and found.rows is not None
                else rows,
                producer_fresh_rows(producer, analysis.shapes),
                found.seeds if found is not None else frozenset()))
    return tuple(gap for gap in out if gap is not None)


def _moves(node: Identified) -> bool:
    """Whether the node is a collective that moves its operand between ranks
    rather than reducing it (`OpSpec.reduction` is False — declared by every
    collective, by nothing else)."""
    spec = OP_REGISTRY.get(node.op or "")
    return spec is not None and spec.reduction is False


def _restates_kv(node: Identified) -> bool:
    """Whether the node is a KV stream marker (`OpSpec.kv_stream_marker`):
    an op that declares the positions law of the stream it emits."""
    spec = OP_REGISTRY.get(node.op or "")
    return spec is not None and spec.kv_stream_marker


def materialized(node: Optional[Identified],
                 cached: FrozenSet[int] = frozenset(), *,
                 edges: Edges, follow_moves: bool = False
                 ) -> Optional[Identified]:
    """The first producer in the chain that actually LANDS a tensor — views
    and broadcasts re-index what somebody else materialised. This is the node
    whose cache declaration (or not) decides how an edge is priced; the
    fan-out walker follows the same chain for the same reason.

    A CACHED node lands one whatever its op: GQA declares `v_split_heads`, a
    head reshape, and to cache a view is precisely to write it down. Walking
    past it reaches the projection that made it, which carries no declaration,
    and the cache then bills nothing — half of GQA's cache traffic.

    With `follow_moves`, cross a non-reducing collective too: it lands the
    same operand on another rank rather than making new context, exactly as
    `covered_context_source` crosses it. The live-marker check needs that
    chain so a cached window past an all-gather or all-to-all is not mistaken
    for live rows."""
    seen: set = set()
    while node is not None and node.uid not in seen:
        seen.add(node.uid)
        if node.uid in cached or not (
                is_view(node) or (follow_moves and _moves(node))):
            return node
        node = edges.source(node, 0) if node.inputs_obj else None
    return node


def landed_by(analysis: Any, layer: Layer["Cache"]
              ) -> Dict[int, CachedTensor]:
    """The leaf that writes each per-position cached tensor, by uid, mapped
    to the tensor — so the write and every same-step read of that leaf's
    output are priced at the tensor's own format (`traffic.declared_width`).

    The declared cached node is the tensor; the leaf that WRITES it is the
    first one at or above it that is not a view (`is_view`, the predicate
    `traffic` bills nothing for): GQA declares `v_split_heads`, a head
    reshape, and the bytes are the projection's. A wrapper is its output
    leaf. The walk stops at a view that does not hand on its whole input
    (a slice of a packed projection): the leaf above it writes more than
    the cache keeps, so no leaf writes this tensor alone and none is
    returned for it.

    A recurrent state is no node's output and is skipped. Two tensors one
    leaf writes must agree on their format; they cannot, being one tensor,
    disagree, and a declaration that makes them is refused.

    Over `analysis`'s own walked program, whatever program a caller then
    prices: the uids are the walked leaves, which a rewrite's `launch_of`
    maps onto the launch that absorbed them."""
    program = analysis.program
    shapes = analysis.shapes.values
    edges = Edges(analysis)
    out: Dict[int, CachedTensor] = {}
    for entry in layer.values.values():
        for tensor in entry.tensors:
            if tensor.recurrent:
                continue
            node = program.by_uid(tensor.node)
            while node is not None:
                if node.children:
                    node, _port = program.output_leaf(node)
                    continue
                if not is_view(node) or not node.inputs_obj:
                    break
                shape = shapes.get(node.uid)
                if (shape is None or not shape.ins
                        or not same_law(product(shape.out),
                                        product(shape.ins[0]))):
                    node = None
                    break
                node = edges.source(node, 0)
            if node is None:
                continue
            held = out.setdefault(node.uid, tensor)
            if held.resolved_format != tensor.resolved_format:
                raise ValueError(
                    f"{node.display_path} writes the cached tensors "
                    f"{held.name!r} ({held.resolved_format.label()}) and "
                    f"{tensor.name!r} ({tensor.resolved_format.label()}); "
                    f"one tensor is stored in one format")
    return out


@dataclass(frozen=True)
class CacheRef:
    """One cached tensor: the component keeping it and the name `derive`
    gives it (`indexer.k_norm` for a sub-component's). Two references are the
    same tensor exactly when these agree, whichever node holds them."""

    component: int
    tensor: str

    def __str__(self) -> str:
        return f"{self.component}:{self.tensor}"


@dataclass(frozen=True)
class KVAccess:
    """Which cached tensors one node writes and reads — a structural fact,
    not a byte count. How much a step moves is `traffic`'s answer and depends
    on the phase; this says only that the node's input or output IS the
    cache.

    `publishes` marks a node whose OUTPUT is cache-resident data: a declared
    cached node, or a pure re-arrangement of one (a view, a broadcast, a
    join of nothing but cache). Its out-edges carry the cache; it is not a
    reader, because it only moves what it was handed.
    """

    writes: FrozenSet[CacheRef] = frozenset()
    reads: FrozenSet[CacheRef] = frozenset()
    publishes: bool = False
    #: The slot kinds (`CachedTensor.kind`) the node's key and value
    #: operands come from, on an op declaring `k` / `v` input roles: the
    #: cache they read, or that the component keeping it computes them from
    #: (a dense MLA prefill decompresses its keys from the latent).
    key_slots: FrozenSet[str] = frozenset()

    @staticmethod
    def union(parts: Tuple["KVAccess", ...]) -> "KVAccess":
        """What a node standing for several does: a fused kernel writes and
        reads what its members did. The combine `Layer.rebase` needs."""
        return KVAccess(
            writes=frozenset().union(*(p.writes for p in parts)),
            reads=frozenset().union(*(p.reads for p in parts)),
            publishes=any(p.publishes for p in parts),
            key_slots=frozenset().union(*(p.key_slots for p in parts)),
        )


def resident(analysis: Any, *, geometry: Optional[CacheGeometry] = None
             ) -> Dict[CacheRef, Tuple[Any, Cache, CachedTensor]]:
    """Every tensor a persisting cache keeps, by reference, with the
    component node that keeps it.

    `derive` over the component nodes. An encoder's cache (`persists`
    false) is dropped: it lives for one forward pass and nothing reads it
    later.
    """
    out: Dict[CacheRef, Tuple[Any, Cache, CachedTensor]] = {}
    for holder, entry in _holders(analysis, geometry):
        if not entry.persists:
            continue
        for tensor in entry.tensors:
            out[CacheRef(entry.component, tensor.name)] = (
                holder, entry, tensor)
    return out


def kv_access(analysis: Any) -> Layer[KVAccess]:
    """Who writes and who reads each persisting cached tensor, by node uid.

    WRITE: `CachedTensor.node` — the declared cached node is the write
    endpoint (`cached_node_uids`). A recurrent state has no such node; its
    declared carrier writes it, and reads it.

    READ: a leaf with an input edge whose source, traced back to the tensor
    that lands it (`materialized`), is cache-resident data
    (`context_sources`). The nodes inside that set — views, broadcasts,
    joins of cache — only move the data along and are not readers; they are
    marked `publishes`.

    Structural, like the pedigree it is built on: a fresh prefill's
    attention reads the cache it just wrote, so it is a reader, even though
    `traffic` bills those bytes on the edge rather than as `kv_reads`. That
    is why `traffic`'s cache readers are a subset of these and not the same
    set.

    A cached node published to later layers through an inner stream (YOCO /
    CLA shared KV) is read by a node with no `src`: the reader resolves the
    stream's name in its frame. Its layers' `reads_from` name the layers
    whose cache it reads, and a ref naming the stream is traced through each
    producer the walk bound to it (`Edges`): the reader reads the union of
    their caches.
    """
    program = analysis.program
    trace = _ReadTrace.of(analysis)
    edges, by_seed, kept, pedigree, kinds = (
        trace.edges, trace.by_seed, trace.kept, trace.pedigree, trace.kinds)
    writes: Dict[int, set] = {}
    reads: Dict[int, set] = {}
    for _holder, entry in _holders(analysis, None):
        if not entry.persists:
            continue
        for tensor in entry.tensors:
            ref = CacheRef(entry.component, tensor.name)
            writes.setdefault(tensor.node, set()).add(ref)
            if tensor.recurrent:
                reads.setdefault(tensor.node, set()).add(ref)
    for uid, operands in trace.operand_reads().items():
        reads.setdefault(uid, set()).update(*operands.values())

    publishing = {
        uid for uid, origin in pedigree.items()
        if any(ref in kept for seed in origin.seeds
               for ref in by_seed[seed])}

    def key_slots(node: Any) -> FrozenSet[str]:
        """Up from a kernel's `k` / `v` operands to the cache-resident data
        they come from, not past another kernel."""
        found, seen = set(), set()
        todo = [source for index, role in enumerate(_roles(node))
                if role in ("k", "v") for source in edges.sources(node, index)]
        while todo:
            current = todo.pop()
            if current.children:
                current, _port = program.output_leaf(current)
            if current.uid in seen or _roles(current):
                continue
            seen.add(current.uid)
            if current.uid in pedigree:
                found.update(kinds[ref]
                             for seed in pedigree[current.uid].seeds
                             for ref in by_seed[seed] if ref in kept)
                continue
            todo.extend(source for index in range(len(current.inputs_obj or ()))
                        for source in edges.sources(current, index))
        return frozenset(kind for kind in found if kind is not None)

    keyed = {node.uid: key_slots(node) for node in program.nodes()
             if not node.children and edges.covers(node)
             and {"k", "v"} & set(_roles(node))}
    values: Dict[int, KVAccess] = {}
    for uid in set(writes) | set(reads) | publishing | set(keyed):
        access = KVAccess(writes=frozenset(writes.get(uid, ())),
                          reads=frozenset(reads.get(uid, ())),
                          publishes=uid in publishing,
                          key_slots=keyed.get(uid, frozenset()))
        if access.writes or access.reads or access.publishes \
                or access.key_slots:
            values[uid] = access
    return Layer(program, values)


@dataclass(frozen=True)
class _ReadTrace:
    """The cache-resident data of one walked program: every per-position
    cached tensor by the node that lands it (`by_seed`), which of them
    persist (`kept`), and the pedigree of every edge that carries one
    (`context_sources`)."""

    program: Any
    edges: "Edges"
    by_seed: Mapping[int, List[CacheRef]]
    kept: FrozenSet[CacheRef]
    kinds: Mapping[CacheRef, Optional[str]]
    pedigree: Mapping[int, Any]

    @classmethod
    def of(cls, analysis: Any) -> "_ReadTrace":
        edges = Edges(analysis)
        by_seed: Dict[int, List[CacheRef]] = {}
        kept: set = set()
        kinds: Dict[CacheRef, Optional[str]] = {}
        for _holder, entry in _holders(analysis, None):
            for tensor in entry.tensors:
                ref = CacheRef(entry.component, tensor.name)
                kinds[ref] = tensor.kind
                if entry.persists:
                    kept.add(ref)
                if not tensor.recurrent:
                    by_seed.setdefault(tensor.node, []).append(ref)
        return cls(analysis.program, edges, by_seed, frozenset(kept), kinds,
                   context_sources(edges, frozenset(by_seed)))

    def operand_reads(self) -> Dict[int, Dict[int, FrozenSet[CacheRef]]]:
        """`{reader uid: {input index: persisting tensors that input
        carries}}` — a leaf with an input edge whose source, traced back to
        the tensor that lands it (`materialized`), is cache-resident data.
        The nodes inside the pedigree only move the data along and are not
        readers."""
        edges, pedigree, seeds = self.edges, self.pedigree, frozenset(
            self.by_seed)
        out: Dict[int, Dict[int, FrozenSet[CacheRef]]] = {}
        for node in self.program.nodes():
            if (node.children or node.uid in pedigree
                    or not edges.covers(node)):
                continue
            for index in range(len(node.inputs_obj or ())):
                carried: set = set()
                for source in edges.sources(node, index):
                    # A re-arrangement already carries every cache it
                    # re-indexes (a stream's view, all the producers its
                    # layers read); anything else is traced to the tensor
                    # that lands it.
                    origin = pedigree.get(source.uid)
                    if origin is None:
                        landed = materialized(source, seeds, edges=edges)
                        if landed is None or landed.uid not in pedigree:
                            continue
                        origin = pedigree[landed.uid]
                    for seed in origin.seeds:
                        carried.update(ref for ref in self.by_seed[seed]
                                       if ref in self.kept)
                if carried:
                    out.setdefault(node.uid, {})[index] = frozenset(carried)
        return out


def _input_templates(declaration: Any) -> Tuple[Tuple[Any, ...], ...]:
    """The op's declared input templates, one per input."""
    shape_in = tuple(declaration.shape_in or ())
    if shape_in and isinstance(shape_in[0], (tuple, list)):
        return tuple(tuple(t or ()) for t in shape_in)
    return (shape_in,)


def _paired_operands(reader: Any, index: int) -> Tuple[int, ...]:
    """The inputs `reader` multiplies its input `index` with along that
    input's innermost axis: by the op's declaration, the inputs whose
    declared innermost axis is the same one and which no output keeps (a
    contraction — `attention_score`'s q and k over `d_head`). Empty when
    the input's innermost axis is not contracted."""
    declaration = reader.declaration
    if declaration is None:
        return ()
    templates = _input_templates(declaration)
    if index >= len(templates) or not templates[index]:
        return ()
    axis = templates[index][-1]
    if axis in tuple(declaration.shape_out or ()):
        return ()
    return tuple(j for j, template in enumerate(templates)
                 if j != index and template and template[-1] == axis
                 and j < len(reader.inputs_obj or ()))


def _refuse_transform(tensor: Any, message: str) -> Any:
    fmt = tensor.resolved_format
    return PrecisionError([{
        "path": f"cache[{tensor.kind}]",
        "message": (f"{tensor.name} is stored {fmt.label()}: {message}"),
        "fix": ("store this slot in a format with no `transform`, or "
                "author the rotation in the component's forward"),
    }])


def transform_edges(analysis: Any) -> Dict[Tuple[int, int], str]:
    """`{(consumer uid, input index): op}` — every edge a cache format's
    stored transform (`CacheFormat.transform_op`) puts its op on: into each
    store marker that lands a tensor whose format rotates it (the rotation
    is of what is stored), and on the paired operand of each product that
    reads one (`_paired_operands`: the product is only right in one basis,
    so the fresh operand is rotated into the cached one's). Empty for a
    model with no transform format. A cache whose writer or readers cannot
    carry the rotation that way is refused (`PrecisionError`): the format
    is not one this graph can store."""
    program = analysis.program
    layer = derive(analysis)
    edges: Dict[Tuple[int, int], str] = {}
    transformed: Dict[Any, Any] = {}
    kept = resident(analysis)
    for _holder, _entry, tensor in kept.values():
        if tensor.recurrent or tensor.resolved_format.transform_op is None:
            continue
        transformed[(tensor.node, tensor.name)] = tensor
    if not transformed:
        return edges
    for writer, tensor in sorted(landed_by(analysis, layer).items()):
        op = tensor.resolved_format.transform_op
        if op is None or (tensor.node, tensor.name) not in transformed:
            continue
        node = program.by_uid(writer)
        if (node.declaration is None or not node.declaration.kv_stream_marker
                or not node.inputs_obj):
            raise _refuse_transform(tensor, (
                f"its write rotates it, and {node.display_path} ({node.op}) "
                f"computes the stored tensor rather than marking it, so no "
                f"edge of the graph carries the rotated values into the "
                f"cache"))
        edges[(writer, 0)] = op
    for reader, operands in sorted(
            _ReadTrace.of(analysis).operand_reads().items()):
        node = program.by_uid(reader)
        for index, refs in sorted(operands.items()):
            for ref in sorted(refs, key=str):
                tensor = kept[ref][2]
                op = tensor.resolved_format.transform_op
                if op is None:
                    continue
                paired = _paired_operands(node, index)
                if len(paired) == 1 and any(
                        kept[other][2].resolved_format.transform_op == op
                        for other in operands.get(paired[0], ())):
                    continue            # both operands stored rotated
                if len(paired) != 1:
                    raise _refuse_transform(tensor, (
                        f"{node.display_path} ({node.op}) reads it on input "
                        f"{index} and does not multiply it with exactly one "
                        f"other operand along its innermost axis, so it "
                        f"cannot run in the rotated basis"))
                if (reader, paired[0]) in edges and \
                        edges[(reader, paired[0])] != op:
                    raise _refuse_transform(tensor, (
                        f"{node.display_path} would rotate input "
                        f"{paired[0]} by two transforms"))
                edges[(reader, paired[0])] = op
    return edges


def paired_records(analysis: Any) -> Dict[Tuple[int, int], Any]:
    """`{(reader uid, input index): ActivationFormat}` — every operand a
    reader of a quantised cached tensor quantises before it multiplies the
    two: the operand paired with the cached one (`_paired_operands`), for a
    tensor whose slot kind declares what its readers do to it
    (`SlotKind.paired_operand`, by the element dtype the tensor is stored
    at). The requirement is the reader's, as a quantised weight's is its
    matmul's (`ResolvedPrecision.required_format`): the walk prices it as a
    conversion and the cast graft materialises it. Empty for a model whose
    cached tensors no such reader multiplies quantised. A reader that does
    not multiply the tensor with exactly one operand, or a stored dtype the
    kind names no format for, is refused (`PrecisionError`)."""
    kinds = cache_families.slot_kinds()
    kept = resident(analysis)
    wanted = {ref for ref, (_holder, _entry, tensor) in kept.items()
              if not tensor.recurrent and tensor.kind in kinds
              and kinds[tensor.kind].paired_operand
              and tensor.resolved_format.scale is not None}
    out: Dict[Tuple[int, int], Any] = {}
    if not wanted:
        return out
    program = analysis.program
    for reader, operands in sorted(
            _ReadTrace.of(analysis).operand_reads().items()):
        node = program.by_uid(reader)
        if node.provenance == "fusion":
            # The requirement is a leaf's of the walked graph, where the cast
            # graft reads it; a launch the fusion rewrite made from such
            # leaves reads what the graft already put on their operand.
            continue
        for index, refs in sorted(operands.items()):
            for ref in sorted(refs & wanted, key=str):
                tensor = kept[ref][2]
                fmt = tensor.resolved_format
                name = kinds[tensor.kind].paired_operand.get(fmt.dtype)
                if name is None:
                    raise _refuse_paired(tensor, (
                        f"its readers quantise the operand they multiply it "
                        f"with to its element type, and slot kind "
                        f"{tensor.kind!r} names no format for {fmt.dtype}"))
                paired = _paired_operands(node, index)
                if len(paired) != 1:
                    raise _refuse_paired(tensor, (
                        f"{node.display_path} ({node.op}) reads it on input "
                        f"{index} and does not multiply it with exactly one "
                        f"other operand along its innermost axis"))
                target = ACTIVATION_FORMATS[name]
                if out.setdefault((reader, paired[0]), target) != target:
                    raise _refuse_paired(tensor, (
                        f"{node.display_path} would quantise input "
                        f"{paired[0]} to two formats"))
    return out


def _refuse_paired(tensor: Any, message: str) -> Any:
    return PrecisionError([{
        "path": f"cache[{tensor.kind}]",
        "message": (f"{tensor.name} is stored "
                    f"{tensor.resolved_format.label()}: {message}"),
        "fix": "store this slot in a format its readers can multiply",
    }])


def _roles(node: Identified) -> Tuple[Optional[str], ...]:
    """The operand roles the node's op declares (`OpSpec.input_roles`)."""
    return tuple(node.declaration.input_roles or ()) \
        if node.declaration is not None else ()


def _holders(analysis: Any, geometry: Optional[CacheGeometry]
             ) -> List[Tuple[Any, Cache]]:
    """`(component node, Cache)` for every component keeping a cache."""
    program = analysis.program
    layer = derive(analysis, geometry=geometry)
    return [(program.by_uid(uid), entry)
            for uid, entry in layer.values.items()]


class MissingCachedNode(KeyError):
    """A `cache.entries` slot naming a node the component does not have.

    Raised rather than skipped: a cache declared and not found is an unbilled
    cache, and a model whose KV comes out too small by one tensor still fits in
    memory and still reads as an answer.
    """


def _per_position_elements(
    node: Identified,
    *,
    shapes: Layer[Any],
) -> Optional[Extent]:
    """One cache row's width: the node's output without the batch and the
    sequence axes, asked by identity.

    Not a slice of the leading axes: a K tensor is
    `[B, n_kv_heads, s_ctx, d_head]`, so dropping two leading entries drops the
    head count and the cache comes out `n_kv_heads` times too small. And not a
    division by the workload's `B·s_ctx`: how many rows a node emitted is not
    always the token count. A time-compressing producer writes one row per
    complete window of `m` tokens — its own output sequence axis is
    `floor(s_ctx/m)` — and that is still an axis MADE OF `s_ctx`, so the row
    is the product of the other axes without knowing that it compresses.

    An axis with no name has no identity to drop, and is refused: the walk
    owes that name, and dividing the workload back out downstream answers for
    a shape it assumed.
    """
    shape = shapes.get(node)
    if shape is None or not shape.out:
        return None
    unnamed = [i for i, names in enumerate(shape.names()) if not names]
    if unnamed:
        raise ValueError(
            f"{node.display_path}: cached tensor {list(shape.out)!r} has "
            f"unnamed axis/axes {unnamed}; a cache row is the axes that are "
            f"neither the batch nor the sequence, asked by name")
    return shape.without(*sorted(dims.step_token_axes()))


def _law_for(analysis: Any, component: Identified, node: Identified,
             anchor: Any) -> Any:
    """Which op speaks for this tensor's cardinality.

    Four steps, and each exists because a real cache has no law of its own at
    the step before. A windowed `kv_concat` is a plain concatenation — the
    window is the score's, and the score is its consumer. A compressed latent
    carries its own `ceil(s_kv/m)` and must NOT inherit the layer's. The two sides
    fall back independently, because an op that states a read and no storage has
    said nothing about storage.
    """
    own = analysis.cache_positions.get(node)
    if own is None and node.children:
        own = _output_law(analysis, node)
    if own is None:
        own = _consumer_law(analysis, component, node)
    if own is None:
        own = _scoring_law(analysis, component, node)
    read = own.read if own is not None else None
    stored = own.stored if own is not None else None
    # The pairs are the read law summed over the step's queries, so they
    # come from whichever law gave the read.
    pairs = own.pairs if own is not None else None
    if anchor is not None:
        if read is None:
            read = anchor.read
            pairs = anchor.pairs
        if stored is None:
            stored = anchor.stored
    return KVPositions(read=read, stored=stored, pairs=pairs)


def _output_law(analysis: Any, node: Identified) -> Any:
    """A sub-component's law is its declared output op's; None when that op
    declares none, and the caller asks the tensor's readers
    (`_consumer_law`)."""
    # `output_names`: the declared outputs as this call resolved them — a
    # switched-off `when:` output already names the step it reads through.
    by_name = {c.name: c for c in node.children}
    for name in node.output_names:
        child = by_name.get(name)
        if child is not None:
            law = analysis.cache_positions.get(child)
            if law is not None:
                return law
    return None


def _consumer_law(analysis: Any, component: Identified,
                  node: Identified) -> Any:
    """The law of the ops that READ this tensor.

    A cache node is usually produced by something that says nothing about
    cardinality and consumed by the attention score that does. Anchor status is
    irrelevant here: a full-scan indexer is not the layer's headline and is
    exactly the right answer for the tensor it scans.

    A reader through a re-indexing is still the reader: absorbed MLA scores
    its latent through a `head_broadcast`, which moves nothing and says
    nothing about cardinality, so the walk continues past views to the ops
    that do.

    Every reader reached is asked, and each side of the law (read, storage)
    is the one its readers that state it agree on. Readers that state
    different laws for one tensor leave no answer to pick — declaration
    order is not one — so that is refused; the tensor then states its own
    law (a KV marker op such as `sliding_window_kv` / `token_pool_compressor`
    / `kv_cache_view` on it).
    """
    laws: List[Tuple[Identified, Any]] = []
    reached = {node.uid}
    frontier = {node.uid}
    while frontier:
        views = set()
        for sibling in component.children:
            if sibling.uid in reached:
                continue
            for ref in sibling.inputs_obj:
                if ref.src is not None and ref.src.uid in frontier:
                    law = analysis.cache_positions.get(sibling)
                    if law is not None:
                        laws.append((sibling, law))
                    elif is_view(sibling):
                        views.add(sibling.uid)
                    reached.add(sibling.uid)
                    break
        frontier = views
    if not laws:
        return None

    def agreed(side: str, values: List[Tuple[Identified, Any]]) -> Any:
        stated = [(reader, value) for reader, value in values
                  if value is not None]
        if not stated:
            return None
        first = algebra(stated[0][1])
        disagree = [(reader, value) for reader, value in stated[1:]
                    if algebra(value) != first]
        if disagree:
            said = "; ".join(f"{reader.display_path} says {value}"
                             for reader, value in stated)
            raise SpecRefusal(
                f"{node.display_path}: its readers state different {side} "
                f"laws for one cached tensor ({said})",
                declaration="cache.entries",
                fix=(f"state the law on the cached tensor itself: put the KV "
                     f"marker op that says how many positions it holds "
                     f"(`sliding_window_kv`, `token_pool_compressor`, "
                     f"`kv_cache_view`) between {node.name!r} and its "
                     f"readers"),
                node=node.display_path)
        return stated[0][1]

    return KVPositions(
        read=agreed("read", [(r, law.read) for r, law in laws]),
        stored=agreed("stored", [(r, law.stored) for r, law in laws]),
        pairs=agreed("pairs", [(r, law.pairs) for r, law in laws]))


def _scoring_law(analysis: Any, component: Identified,
                 node: Identified) -> Any:
    """The law of the attention kernels the tensor is scored by, through
    any op of the component: MLA's latent reaches its score only through
    `kv_b_proj`, which is no view, so `_consumer_law` stops before it.

    Without this step the tensor took the component's anchor — the first
    kernel a breadth-first search meets — so a latent was priced by
    whichever score the component happened to list first: GLM-5.3-Flash
    lists its indexer's dense score before the DSA score that reads the
    latent's top-k, and its latent came out read whole. Only the kernels
    that state themselves block anchors (`is_block_anchor`) speak, each
    side (read, stored) where those that state it agree; a side they leave
    open, or disagree on, the anchor answers as before."""
    reached = {node.uid}
    frontier = {node.uid}
    laws: List[Any] = []
    while frontier:
        nxt = set()
        for sibling in component.children:
            if sibling.uid in reached:
                continue
            if any(ref.src is not None and ref.src.uid in frontier
                   for ref in (sibling.inputs_obj or ())):
                reached.add(sibling.uid)
                law = analysis.cache_positions.get(sibling)
                if law is not None and law.is_block_anchor:
                    laws.append(law)
                else:
                    nxt.add(sibling.uid)
        frontier = nxt
    def agreed(values: List[Any]) -> Any:
        stated = [v for v in values if v is not None]
        if not stated or any(algebra(v) != algebra(stated[0])
                             for v in stated[1:]):
            return None
        return stated[0]

    read = agreed([law.read for law in laws])
    stored = agreed([law.stored for law in laws])
    if read is None and stored is None:
        return None
    return KVPositions(read=read, stored=stored,
                       pairs=agreed([law.pairs for law in laws])
                       if read is not None else None)


def _anchor(analysis: Any, component: Identified) -> Any:
    """The component's headline cardinality — the main attention score.

    Searched over the whole subtree rather than the direct children, because the
    score often sits inside a wrapper the component invokes. An op that declared
    itself out (`kv_block_anchor=False`) is skipped: it has a real law for its
    own tensor and must not become the layer's.
    """
    stack: List[Identified] = list(component.children)
    while stack:
        node = stack.pop(0)
        law = analysis.cache_positions.get(node)
        if law is not None and law.is_block_anchor:
            return law
        stack.extend(node.children)
    return None


def workload_dim(analysis: Any, symbol: str) -> Extent:
    """A workload fact, from the binding the analysis was run against. The
    cache length is not in the graph — no node produces it — so there is
    nowhere else it could come from. A walk that bound no such fact has no
    answer to give, and is refused rather than handed a stand-in number."""
    binding = analysis.binding
    if binding is None or symbol not in binding:
        raise ValueError(
            f"the analysis was run against no binding of {symbol!r}, so the "
            f"workload fact it names is unknown")
    return binding[symbol]


def holds_every_position(tensor: CachedTensor) -> bool:
    """Whether a cached tensor depends on every position of the sequence: a
    recurrent state folds them all in, and a storage law that grows without
    bound as the length does (`s_kv`, `ceil(s_kv / m)`) keeps one per
    position or per group. A window (`Min(s_kv, s_window)`) keeps only the
    last positions, and is the one law that does not."""
    return tensor.recurrent or law_grows(tensor.stored)


def law_grows(law: Any) -> bool:
    """Whether a positions law (stored or read) grows without bound as the
    cache length does. A law never shrinks as the length grows, so it is
    bounded exactly when two lengths past any window give the same count;
    a law that still has another symbol in it cannot be shown bounded and
    is taken to grow. No length in it at all is a fixed count."""
    expr = algebra(CACHE_LEN if law is None else law)
    if CACHE_LEN not in expr.free_symbols:
        return False
    far = sp.Integer(10) ** 15
    return sp.simplify(expr.subs(CACHE_LEN, 2 * far)
                       - expr.subs(CACHE_LEN, far)) != 0


@dataclass(frozen=True)
class PrefillReach:
    """Which positions of a long prompt each layer runs on in prefill.

    `counts`: per accounting root (`Program.contributions`), how many of its
    layers run on EVERY prompt position. `full_from`: the prompt length from
    which the other layers run on a fixed number of last positions, so
    that past it each further prompt token touches only the layers
    `counts` keeps (0 when every layer runs on every position)."""
    counts: Dict[int, int]
    full_from: int


def _bounded_positions(tensor: CachedTensor, *, read: bool) -> int:
    """How many positions a bounded (windowed) tensor keeps or reads once
    the length is past its window."""
    law = tensor.read if read else tensor.stored
    return positions_at(law if law is not None else CACHE_LEN, 10 ** 15)


def prefill_reach(analysis: Any) -> PrefillReach:
    """Which layers process every prompt token in prefill, and from which
    prompt length the rest run on the prompt's last positions only.

    A layer's output at a prompt position is needed only by what reads it:
    a cache that outlives the step, or a layer that needs it. So a layer
    must run on every position when it keeps a tensor that depends on
    every position (`holds_every_position`), or when a layer reading its
    output must. Any other layer keeps only a window, so it needs its
    input at the positions its readers need plus how far its window reaches
    back, and at least the positions its window keeps: a DeepSeek-V4.1 layer
    past layer 20 keeps its 128-token window and reads layer 20's compressed
    KV, so past a few thousand prompt tokens each further token runs only
    through layers 0..20 — the README's 8B activated in prefill against 16B
    in decode. The sampled head reads one position.

    The edges are the declared ones: the stream a stack carries from each
    layer to the next (`Stage.carry`), and each stage's authored inputs
    (`Stage.upstream`). An aux stack that reads the backbone by a stream
    rather than by its inputs (`Stack.upstream` empty) is answered for every
    backbone layer: its full need marks them all, its windowed need is held
    by each. Encoder stacks are not the language model's prefill: their
    roots count 0."""
    program = analysis.program
    calls = by_call(analysis).values

    def tensors_under(roots) -> List[CachedTensor]:
        out: List[CachedTensor] = []
        pending = list(roots)
        while pending:
            node = pending.pop()
            entry = calls.get(node.uid)
            if entry is not None:
                out.extend(entry.tensors)
            pending.extend(node.children or ())
        return out

    # One slot per op stage and per layer of a repeated template, in walk
    # order, each with the slots it reads.
    slots: List[Tuple[Any, Optional[int]]] = []
    reads: List[List[int]] = []
    by_name: Dict[Tuple[str, str], int] = {}
    stream_read: set = set()

    def upstream(stack: str, stage: Any) -> List[int]:
        found = (by_name.get((stack, n), by_name.get(("", n)))
                 for n in stage.upstream)
        return [i for i in found if i is not None]

    for stack in (st for st in program.stacks if not st.is_encoder):
        if stack.is_aux and not stack.upstream:
            stream_read.add(stack.name)
        stages = stack.walk_order()
        placed: set = set()
        for stage in stages:
            if not isinstance(stage, BlockStage):
                by_name[(stack.name, stage.name)] = len(slots)
                reads.append(upstream(stack.name, stage))
                slots.append((stage, None))
                continue
            owner = stage.carry.owner if stage.carry is not None else id(stage)
            if owner in placed:
                continue
            placed.add(owner)
            chain = sorted(
                ((layer, st) for st in stages if isinstance(st, BlockStage)
                 and (st is stage or (st.carry is not None
                                      and st.carry.owner == owner))
                 for layer in st.template.layer_indices),
                key=lambda pair: pair[0])
            previous: Optional[int] = None
            for layer, st in chain:
                reads.append([previous] if previous is not None
                             else upstream(stack.name, st))
                previous = len(slots)
                slots.append((st, layer))
                by_name[(stack.name, st.name)] = previous

    # Per slot: None when it runs on every position, else the number of
    # last positions it needs its input at. Walked back from the readers,
    # which come later in walk order; a slot no one reads is a sink and
    # yields the one sampled position.
    held = [tensors_under(stage.roots) for stage, _layer in slots]
    every = [any(holds_every_position(t) for t in tensors) for tensors in held]
    out_need: List[Any] = [0] * len(slots)   # 0: not read yet
    need: List[Optional[int]] = [None] * len(slots)
    readers = [0] * len(slots)
    for sources in reads:
        for source in sources:
            readers[source] += 1
    for index in range(len(slots) - 1, -1, -1):
        wanted = out_need[index] if readers[index] else 1
        if every[index] or wanted is None:
            need[index] = None
        else:
            keep = max((_bounded_positions(t, read=False)
                        for t in held[index]), default=0)
            reach = max((_bounded_positions(t, read=True) - 1
                         for t in held[index]), default=0)
            need[index] = max(wanted + reach, keep)
        for source in reads[index]:
            current = out_need[source]
            out_need[source] = (None if need[index] is None or current is None
                                else max(current, need[index]))
    aux_need = [need[i] for i, (stage, _l) in enumerate(slots)
                if stage.stack in stream_read and not reads[i]]
    if aux_need:
        full = any(n is None for n in aux_need)
        widest = max((n for n in aux_need if n is not None), default=0)
        for i, (stage, layer) in enumerate(slots):
            if stage.stack == "" and layer is not None and need[i] is not None:
                need[i] = None if full else max(need[i], widest)
        # Re-walk the backbone so the slots before a newly full one follow.
        for index in range(len(slots) - 1, -1, -1):
            if need[index] is None:
                for source in reads[index]:
                    need[source] = None

    counts: Dict[int, int] = {c.node.uid: 0 for c in program.contributions}
    for (stage, _layer), n in zip(slots, need, strict=True):
        for root in stage.roots:
            if root.uid in counts:
                counts[root.uid] += int(n is None)
    full_from = max((n for n in need if n is not None and n > 1), default=0)
    return PrefillReach(counts=counts, full_from=full_from)
