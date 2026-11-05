"""`Analysis` — one walk's output, and the joins consumers ask of it.

The container is layers over the program's nodes (`cost`, `shapes`,
`storage`, `formats`, `instances`, …), produced by `Evaluator.run` and read
by everything downstream. The methods are the JOINS — `tensors()` zips
storage with formats and counts, `by_role`/`by_stack`/`total` group cost by
facts that ride other layers — kept on the container so attribution is a
`zip`/group-by over layers that already exist, never a second traversal.

Separate from `evaluator` because the two change for different reasons: the
machine changes when the walk learns an op or a scope rule, this changes when
a consumer needs a new join. Nothing here traverses the tree.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction
from functools import cached_property
from typing import (Any, Callable, ClassVar, Dict, FrozenSet, Iterable, Iterator, List,
                    Mapping, NamedTuple, Optional, Sequence, Tuple)

import sympy as sp

from .binding import Binding
from . import cache as cache_mod
from .dim import Dim, assuming, exact as dim_exact, exact_div, value_of
from .dim import least_of as dim_least_of, point_value
from . import formats as formats_mod
from .precision import AS_SHIPPED, OUTSIDE_ROLES, ResolvedPrecision
from . import storage as storage_mod
from .tensor_rows import TensorRow, pair_stored
from . import traffic as traffic_mod
from .copies import CopyFactor
from .layer import Layer, Rewrite
from .occurrence import OccurrenceGraph, Reference, ReferenceTable
from .program import Program
from ..cache_formats import CacheFormat
from ..ops import ProcessorStep
from ..dtypes import UnknownDtype, canonical, corpus_token, display_name
from .routed import RoutedLaunch
from .rule import Rule
from .values import Cost, Shape
from .. import dims as dims_mod



def _joined(labels: Mapping[str, float]) -> str:
    """One role's weight formats as the precision panel names them: the one
    it has, or all of them joined. A role holding two is a fact about the
    model, so both are named rather than the heavier one standing for the
    role (which is what `dominant_weight_dtype` does, and why a
    three-layer BF16 tail could go unmentioned)."""
    return (next(iter(labels)) if len(labels) == 1
            else " + ".join(sorted(labels)))


def _format_label(fmt: Any) -> str:
    """A weight format as the precision panel names it: the logical dtype,
    and the scheme that stores it when one does."""
    try:
        name = display_name(fmt.logical_dtype)
    except UnknownDtype:
        name = fmt.logical_dtype
    if fmt.scheme is not None:
        name = f"{name} ({fmt.scheme.name})"
    if fmt.activation_format:
        name = f"{name} · act {fmt.activation_format}"
    return name


class WeightShard(NamedTuple):
    """One partition of a matrix a kernel multiplies, as the grid a weight's
    scale tiles: its rows and its reduction width, as the checkpoint holds
    them and as one rank holds them.

    Read off the placement (`Rule.local_axis_segments`), through the one
    reading of a stored tensor as a matrix (`StoredTensor.matrix_dims`): a
    world cut what differs between the two, a per-head bank's heads are rows,
    an expert bank's experts are copies. A packed tensor (`gate|up` along
    one output axis) is one partition per segment, and `pack` says which
    partitions one weight concatenates — a packed tensor's segments, and the
    matrices a sibling-merging fusion multiplies as one (`parallel:`
    plans) — keyed with the tensor's layer segment, since a layer carve-out
    is its own weight. `granularity` is this tensor's own scale layout
    (`TensorFormat.weight_granularity`), None when it stores no scale.
    `block_scale_layout` carries an explicit checkpoint chunk/load contract,
    with runtime TP and the full fused rank extent beside logical segments.
    """

    pack: Tuple[int, int]
    rows: Any
    rows_local: Any
    width: Any
    width_local: Any
    granularity: Optional[str]
    block_scale_layout: Optional[Dict[str, Any]] = None


def _dtype_name(row: TensorRow) -> str:
    """The dtype a byte total groups `row` under. Every stored tensor
    declares one (its format's, or its own); a row without one is a hole in
    the storage join, so it raises rather than filing under a nameless
    bucket."""
    name = row.logical_dtype or row.storage_dtype
    if not name:
        raise ValueError(f"{row.name}: stored tensor has no dtype")
    return name.upper()


class OwnCopies(NamedTuple):
    """One node's OWN multipliers, as the walk resolved them in the node's
    frame (`Evaluator._copies`) — not cumulative: every enclosing `repeat:`
    is its ancestor's own count, and `Analysis.instances` is their product.

    `stored` is the node's `repeat:`, `active` the copies that run per token
    (its `active:`, else all of them). An expert bank has no copies here: its
    experts are an axis of its tensors.
    """

    stored: Any
    active: Any

    @staticmethod
    def fused(parts: Tuple["OwnCopies", ...]) -> Optional["OwnCopies"]:
        """A fused node's own copies: its members', when they agree. Members
        that run at different multipliers leave the fused node with no own
        count at all — there is no one number — and a reader that asks for
        it is refused (`Analysis.own_copies_of`)."""
        first = parts[0]
        return first if all(part == first for part in parts[1:]) else None


@dataclass(frozen=True)
class AttentionFamily:
    """One attention family of the backbone (`Analysis.attention_families`):
    the cache family its attention call declares (`kv_label`, with the
    compute flavor `variant` — DSA, CSA), how many layers run it, and the
    model-level dims its body reads (`Program.reads`) with the values the
    walk bound them to at the call (`node_dims`).

    `dims` holds only what the call reads: an MLA call reads `d_kv_lora`,
    `d_qk_nope`, `d_v_head` and no KV-head count; a hybrid's linear
    attention reads its own heads. `rebound` names the dims a layer
    assignment or an enclosing call sets for these layers instead of the
    model's binding."""

    label: Optional[str]
    variant: Optional[str]
    layers: int
    dims: Tuple[Tuple[str, int], ...]
    rebound: FrozenSet[str]

    @property
    def name(self) -> str:
        return self.variant or self.label or ""

    def as_dict(self) -> Dict[str, Any]:
        """The family as the analyzer's summary carries it."""
        return {"name": self.name, "layers": self.layers,
                "dims": [{"symbol": symbol, "value": value}
                         for symbol, value in self.dims],
                "rebound": sorted(self.rebound)}


@dataclass(frozen=True)
class MoeFacts:
    """What one routed expert pool declares, as numbers (`Analysis.moe_pools`;
    `Analysis.moe` is the backbone's).

    `n_expert_copies` is the expert AXIS, the one symbol expert parallelism
    divides; `n_experts` is how many experts hold weights and is never
    divided; `n_route_targets` is the dispatch denominator, the destinations
    the top-k draws from (all three default to the same number).
    `active_copies` is what the bank runs per token, `n_experts_active` what
    the router picks; they differ when the destinations include experts no
    node stands for (LongCat's zero-compute experts: 12 picked, 8 run). `d_ffn_expert` is None when the
    banks bind different widths.

    A model may route over more than one pool: DeepSeek-V4.1's draft stages
    route over 128 experts, top-3, while its backbone routes over 384, top-6
    (model.py:142-149). Each pool's numbers are the ones bound where its
    banks sit, and expert parallelism divides each pool by its own count."""

    n_expert_copies: int
    active_copies: int
    n_experts: int
    n_experts_active: int
    n_route_targets: int
    d_ffn_expert: Optional[int]
    n_shared_experts: int


class MediaRows(NamedTuple):
    """How one media input arrives at its encoder in an encode walk: the
    tensor the encoder's first computing op reads, read off the walk's own
    shapes. `segments` is the leading axis — one per item (an image cut by
    `patchify` is one attention segment per item, `segment: item` or one
    temporal patch of a still image; an audio clip is one item) — and
    `rows` the rows each segment is computed at: the item's own rows
    (`valid_rows`, the patch grid or the clip's frames) or, where the spec
    pads them (`pad_rows`), the padded count."""

    name: str
    segments: int
    rows: int
    valid_rows: int


@dataclass
class Analysis:
    """One walk's output: layers over the program's nodes."""

    program: Program
    cost: Layer[Cost]
    shapes: Layer[Shape]
    # The binding this was evaluated against. An analysis is of a program AT a
    # binding, and a consumer that needs a workload fact the graph does not
    # carry — the cache length a step reads, the batch in flight — has nowhere
    # else to get it.
    binding: Optional[Binding] = None
    # Sparse physical kernel geometry authored by a transition plan when raw
    # local extents cannot be represented by ``(global shape, layout)`` alone
    # (for example DCP's gathered head flow).  Ordinary nodes derive locally
    # from ``shapes`` and leave this layer empty.  Kept after ``binding`` to
    # preserve the DTO's original first four positional fields.
    kernel_shapes: Layer[Shape] = field(default_factory=Layer)
    # Semantic role — attention / ffn_dense / ffn_moe / router / … — propagated
    # down from the nearest roled component by `iter_tree`. A layer rather than a
    # field on the cost, because it is a different KIND of fact: cost is derived
    # by the walk, role is read off the graph, and attribution
    # (`attention_params` vs `ffn_params`) is then a `zip` instead of a
    # classification pass that has to re-descend the tree.
    roles: Layer[Optional[str]] = field(default_factory=Layer)
    # How each weight-bearing node is stored. Sideband — resolved by lookup
    # from the node's role and the model's `quant_context`, never propagated
    # along an edge. Partial by construction: a node with no parameters has
    # nothing stored and simply is not in it.
    # What each node puts on disk, asked of the op that owns it. The weight's
    # own two dims come from here, which is what makes a quant scheme's
    # `in`/`out` the weight's rather than whatever the enclosing scope bound.
    storage: Layer[Tuple[storage_mod.StoredTensor, ...]] = field(
        default_factory=Layer)
    formats: Layer[Tuple[formats_mod.Representation, ...]] = field(
        default_factory=Layer)
    # What the checkpoint ships for a tied node beside the weight it reads
    # (`tied_weight.checkpoint_copy`): `uid → (stored tensors, formats)`.
    # Kept out of `storage` because serving skips the copy at load — every
    # deployment reader of `storage` (layout, traffic, weight residency)
    # therefore sees the tie, and only `checkpoint_tensors()` sees the file.
    checkpoint_copies: Layer[Tuple[Tuple[storage_mod.StoredTensor, ...],
                                   Tuple[formats_mod.Representation, ...]]] = field(
        default_factory=Layer)
    # How many physical copies of each node the model holds. Cumulative —
    # block count times every enclosing `repeat:`.
    #
    # A layer rather than a property of the program, because the count is
    # RESOLVED against the binding this analysis ran at: a per-rank walk divides
    # the expert-copies axis, so "how many experts are here" has a different
    # answer per rank and only the walk knows which one it asked. That is also
    # why every per-node value the walk produces is per-COPY: a derivation added
    # afterwards — `Traffic`, storage bytes — reaches a model total by joining
    # against this.
    instances: Layer[Any] = field(default_factory=Layer)
    # The share of a node one token runs — the fraction a consumer asks for
    # when it wants "how much of this leaf's resident weight does a token
    # read": the copies that run over the copies held (a shared-expert bank's
    # `active:` under its `repeat:`), times the share of ONE copy a token
    # runs where a leaf runs only part of itself (a routed bank's
    # `n_experts_active / n_route_targets` of its expert axis,
    # `Evaluator._routed_share`). 1 on almost every node, and exact.
    #
    # One statement, because the share used to have two carriers: it was
    # applied to `active_params` AND multiplied into a published copy COUNT,
    # which made that count a count times a fraction. Both readers of the
    # count then divided the fraction back out — the pipeline to publish
    # `active_ratio_per_token`, the playground's runtime table to feed
    # `row_roofline.weight_demand` — so the one thing anybody wanted was the
    # thing neither could read.
    active_share: Layer[Any] = field(default_factory=Layer)
    # Each node's OWN multipliers (`OwnCopies`), the factors `instances` is a
    # product of. Kept because the product cannot be divided back into them:
    # a per-rank projection that needs one node's own execution count reads
    # it here rather than dividing a cumulative count by its parent's.
    own_copies: Layer[OwnCopies] = field(default_factory=Layer)
    # The dim bindings each node declared for itself (`dims: {in: …, out: …}`).
    # Small — a handful of keys per node, not the whole environment — and
    # kept because a later pass that has to re-evaluate anything about a node
    # otherwise has to re-walk the tree to find out what it resolved against.
    # Retained explicitly so later derivations never need a second walk.
    node_dims: Layer[Mapping[str, Any]] = field(default_factory=Layer)
    # What a node's op rebinds in its own frame, apart from what the spec
    # wrote there: the narrowed `s_kv` and the live pairs `s_pairs` of a
    # node that scores a matrix of its own (a top-k, a window, a pooled
    # stream, a union, or one only narrowed), `s_pairs` as its definition
    # (`binding.Definition`). A reader of names follows the frame through
    # it (`module_bench.frame_names`); a formula's where column prints the
    # definition (`phase_formulas.FormulaDefs`).
    node_rebinds: Layer[Mapping[str, Any]] = field(default_factory=Layer)
    # Rank-law readings already projected under this analysis's rule.  The
    # key is made beside the projection itself, from the exact semantic inputs
    # the remaining computation reads.
    _rank_definition_readings: Dict[Tuple[Any, ...], Any] = field(
        default_factory=dict, init=False, repr=False, compare=False)
    _physical_shape_reading: Optional[Any] = field(
        default=None, init=False, repr=False, compare=False)
    # The function constants each leaf's op takes (`OpSpec.consts`: a
    # clamp's bound, SiTU's betas), `{symbol: value}`, as the node's own
    # frame bound them — the spec's binding, rebound on the way down, so a
    # vision tower's clamp has the tower's bound. Partial by construction: a
    # node whose op takes none is not in it, and neither is one whose frame
    # binds none (absent, never defaulted). A fused node carries the
    # constants its origins took (`rebase_analysis`, `union_consts`).
    consts: Layer[Mapping[str, float]] = field(default_factory=Layer)
    # Nodes the walk refused, `uid → reason`. Empty on a fully-modelled spec;
    # a populated map is the honest form of partial coverage, and the oracle
    # reads it to know which nodes to exclude from comparison rather than
    # counting them as agreement.
    # Where every tensor lives, from the same walk and the same edges as the
    # shapes. A walk with a layout rule writes `EdgeState` here.
    states: Layer[Any] = field(default_factory=Layer)
    # Per-PORT states, an overlay `states` cannot carry (a Layer is uid-keyed
    # and a node has one row): a rule with port-granular states — the layout
    # propagation, whose secondary outputs carry their own layouts —
    # records {(uid, port): state} here. Where a writer fills both, ("out")
    # must duplicate `states[uid]`; the writer asserts that, so the two
    # views cannot disagree silently.
    port_states: Mapping[Tuple[int, str], Any] = field(default_factory=dict)
    # Per-consumer input states after the transition rule has stated what the
    # kernel requires.  Producer output state is not an equivalent substitute:
    # a redistribution exists precisely when the two differ.  Kept as an input-
    # port layer because inputs are keyed by (consumer uid, input index), while
    # output states remain uid/port keyed above.
    input_states: Mapping[Tuple[int, int], Any] = field(default_factory=dict)
    # `KVPositions` in `CACHE_LEN`, per node whose op declares one, and the
    # per-request recurrent state axes (`{state name: (axis, ...)}`), per
    # component that declares any. Both are what `eval.cache` assembles a
    # cache footprint out of; both have to be produced HERE because resolving
    # either needs the frame — a window length rebound per template, a state
    # axis written against its carrier's dims.
    cache_positions: Layer[Any] = field(default_factory=Layer)
    #: The nodes that read their keys through a per-query index list — a
    #: top-k attention's selection, an attention over a `kv_stream_concat`
    #: union of index lists. Such a kernel gathers each listed row from the
    #: cache, query by query, at the cached tensor's live pairs
    #: (`CachedTensor.pairs`, `traffic`); a node absent here streams what it
    #: reads.
    gathered: FrozenSet[int] = frozenset()
    #: What each tensor's output carries in KV positions (`KVLength`), keyed
    #: on the producing node. `cache_positions` says what an op's own law
    #: declares; this says what an edge downstream actually holds, including
    #: a sequence union's summed cardinality and a live marker's read-only
    #: rows. Context coverage is a projection of this layer, never a second
    #: walk over op names.
    kv_lengths: Layer[Any] = field(default_factory=Layer)
    cache_state: Layer[Any] = field(default_factory=Layer)
    #: Reads a step makes of each state, where its entry declares `reads:`
    #: (once otherwise) — resolved in the carrier's frame, like its axes.
    cache_state_reads: Layer[Any] = field(default_factory=Layer)
    # The substitution each node's FLOPs evaluation used — identity → value,
    # recorded by the walk at derivation time. Formula displays bind this
    # table; nothing downstream re-derives it.
    flops_bindings: Layer[Mapping[Any, Any]] = field(default_factory=Layer)
    # Same, for the params figure (`op.params(dims, ...)`).
    params_bindings: Layer[Mapping[Any, Any]] = field(default_factory=Layer)
    redistributes: Tuple[Any, ...] = ()
    #: The conditions on the point this answer's formulas hold under
    #: (`dim.Assumption`), in the order they arose: empty on a walk of
    #: numbers, which checks each on the spot. `admit` checks a point against
    #: them, and a derivation off the walk adds its own (`assumed`).
    assumptions: Tuple[Any, ...] = ()
    #: The counts the walk's conditions are decided over (`dim.Domain`),
    #: for a reader that writes conditions of its own under them
    #: (`assumed`).
    domain: Any = None
    # Each appended stream's elements as the walk published them, in order:
    # the producing `(uid, port)` and the state the value was handed on in.
    # A transfer of an element — across a pipeline cut — moves that value.
    appended: Mapping[str, Tuple[Tuple[Tuple[int, str], Any], ...]] = field(
        default_factory=dict)
    # The state each stage exit is handed to its readers in, settled at the
    # boundary (`Evaluator._closed`), keyed on the producing `(uid, port)`.
    # A value with no entry is handed on in its port state. A transfer of
    # the value — across a pipeline cut — moves it in this state.
    handed_states: Mapping[Tuple[int, str], Any] = field(default_factory=dict)
    # What each block stage bound its template's formals to, by block
    # invocation (`Program.blocks` ordinal): the tensors a stage hands the
    # template it runs, as a component call node's are its `shapes.ins`.
    stage_inputs: Mapping[int, Tuple[Tuple[Any, ...], ...]] = field(
        default_factory=dict)
    # What each of those formals reads, by block invocation: the
    # `occurrence.Reference` the walk bound beside the tensor, as a component
    # call's are its `input_references`.
    stage_references: Mapping[int, Tuple[Any, ...]] = field(
        default_factory=dict)
    # The element of each appended stream a block stage read (`read: each`),
    # by block invocation and stream name: the tensor its producer in
    # another stack appended, as the stage was handed it.
    stage_elements: Mapping[int, Mapping[str, Tuple[Any, ...]]] = field(
        default_factory=dict)
    # The frame each block stage's body ran in, by block invocation: the
    # stage's binding with its assignment's `dims:` over it and the
    # `derived:` names they feed derived again (`Scope.frame`). What
    # a block binds a dim to is read here — the walk's own answer, not a
    # frame opened again by a reader, nor one node's frame standing in for
    # the block's.
    stage_frames: Mapping[int, Binding] = field(default_factory=dict)
    # What the walk bound to each input of each node it evaluated
    # (`occurrence.ReferenceTable`, over the walked program): the producing
    # occurrences at each iteration of the reading stage. A component call's
    # inputs are its `in:` refs, then the streams its body reads with no
    # edge naming them (`Program.declared_reads`). None on an analysis
    # carried across a rewrite: the table is its walk's (`walked`), read
    # through `input_reference`.
    input_references: Optional[ReferenceTable] = None
    # Per stage name, the `(uid, port)` an `inputs:` naming it reads: its
    # last iteration's exit, as the walk handed it on.
    stage_outputs: Mapping[str, Tuple[int, str]] = field(default_factory=dict)
    # The cache law each published inner stream carries, keyed on the node
    # that published it: the node whose op stated the length (None for a
    # union's) and whether the publisher is a cache entry. What a call that
    # reads the stream is handed with it (`transforms.CallSite`).
    stream_laws: Mapping[int, Tuple[Optional[int], bool]] = field(
        default_factory=dict)
    #: the transition rule the walk consulted. Defaults to the identity
    #: `Rule` (one rank, nothing distributed); the evaluator overwrites it
    #: with the rule the walk actually ran under.
    rule: Rule = field(default_factory=Rule)
    #: The deployment's precision profile read against this program
    #: (`precision.ResolvedPrecision`) — what the walk priced weights at, and
    #: what every later layer asks for an edge's dtype, a matmul's dtype and
    #: a cached tensor's format. None only on an analysis assembled outside
    #: the evaluator; `resolved_precision` answers as shipped there.
    precision: Any = None
    # What the graph transforms applied before this walk did, so a layer
    # computed against the ORIGINAL graph can still be carried across. Identity
    # when nothing was rewritten.
    rewrite: Rewrite = field(default_factory=Rewrite)
    #: The walk's own analysis a rewrite started from (`rebase_analysis`),
    #: None on a walk's own analysis. Facts keyed on the walked nodes that a
    #: rewrite merges without changing — the expert pools (`moe_pools`):
    #: a kernel fusion merges nodes, not pools — are read there.
    walked: Optional["Analysis"] = None

    def at(self, point: Any, *,
           structure: Tuple[type, ...] = ()) -> "Analysis":
        """This walk's numbers at `point` (a `Point`, or its counts
        `{symbol name: number}`): the analysis a walk of the workload there
        has, read off this one.

        A point the walk does not admit is refused first (`admit`), with the
        refusal a walk there raises. Then every value is read at the point —
        each layer, the binding, the layout states, whatever a record of the
        walk holds — and nothing else is touched: the program, its nodes,
        and every object none of whose values reads the workload stay the
        same objects. The result carries no conditions, because it is no
        longer a law: it is that point's walk.

        `structure` names the graph types beside `Program` the walk
        holds (a fused program), which no point changes. Counts the point
        leaves free (`Point.free`) stay laws in the read walk, and the
        conditions that read them stay its conditions.
        """
        if not isinstance(point, Point):
            point = Point(point)
        held = tuple(assumption for assumption in self.assumptions
                     if _reads(assumption.condition, point.free))
        for assumption in self.assumptions:
            if assumption not in held:
                assumption.at(point.counts)
        reader = _PointReading(point, structure=structure)
        read = reader.read(self)
        # A condition on the caller's counts stays one, read at the point:
        # its refusal reads its numbers there too.
        read.assumptions = tuple(
            dataclasses.replace(
                reader.read(assumption),
                refusal=lambda at, refusal=assumption.refusal: refusal(
                    lambda expr: at(reader.read(expr))))
            for assumption in held)
        return read

    def admit(self, point: Mapping[str, Any]) -> None:
        """Refuse `point` (`{symbol name: number}`): the first condition of
        the walk's formulas it breaks raises that condition's own
        refusal."""
        for assumption in self.assumptions:
            assumption.at(point)

    @contextmanager
    def assumed(self) -> Iterator[Any]:
        """The scope a reader derives from this walk under: the walk's
        conditions, to which the ones the reader states on the point are
        added. Leaving the scope writes them back, so the conditions a
        reader's numbers rest on are the conditions `admit` checks."""
        with assuming(self.assumptions, self.domain) as ledger:
            yield ledger
        self.assumptions = tuple(ledger.held)

    @property
    def resolved_precision(self) -> Any:
        """`precision`, or as shipped for an analysis no evaluator made."""
        if self.precision is None:
            self.precision = ResolvedPrecision(
                AS_SHIPPED, self.walked.program if self.walked is not None
                else self.program)
        return self.precision

    @cached_property
    def weight_promotions(self) -> Mapping[int, Any]:
        """Every walked leaf whose matmul computes in its weight's dtype
        because the weight is held wider than the activation
        (`traffic.weight_promotion`) — the dtype, not a width: what the
        leaf stores keeps its edge's declaration.

        A layer of this walk, produced once here, because §2.7 gives eval
        the production and the derivation that reads it is handed a graph
        rather than an `Analysis`.

        The storage ⋈ formats join it reads (`tensor_rows.pair_stored`) is
        owned beside the row, so traffic can be imported here at module level
        without either layer importing its consumer or producer."""
        return traffic_mod.weight_promotions(
            self.program, self.formats.values, self.resolved_precision)

    @cached_property
    def shipped_precision(self) -> Any:
        """This walk read against the profile the checkpoint SHIPS with.

        Not a second resolver and not a second walk: the same
        `ResolvedPrecision` over the same program and the same framework
        launches, with `AS_SHIPPED` in place of the deployment's profile.
        What a profile changed is a difference between two resolutions —
        the weight a rule converted, the edge it narrowed, the pin it put on
        a matmul — and asking one resolution about the other's formats
        answers neither question (`precision_effective`)."""
        precision = self.resolved_precision
        # The launches carry their selection environment with them: without
        # the card's peak families `_launch_is_supported` reads the
        # resolution as a cardless fixture and admits a candidate this card
        # cannot run, so the as-shipped column would answer for a launch the
        # priced row never takes.
        return ResolvedPrecision(AS_SHIPPED, precision.program).with_launches(
            precision.launches,
            peak_families=precision.launch_peak_families,
            unavailable_launches=precision.unavailable_launches)

    @cached_property
    def shipped_weight_promotions(self) -> Mapping[int, Any]:
        """`weight_promotions` as the checkpoint ships it: the saved formats
        of every leaf, against the edges `shipped_precision` gives them.

        The promotion a FILE carries — MiniMax's F32 router gate times a
        BF16 row — as opposed to one a precision profile created by holding
        a weight wider than the checkpoint does."""
        return traffic_mod.weight_promotions(
            self.program,
            {uid: tuple(reps.as_saved() for reps in entries)
             for uid, entries in self.formats.values.items()},
            self.shipped_precision)

    def precision_effective(self) -> Dict[str, Any]:
        """What this walk's precision profile is and what it changed: per
        weight role, the edges, per matmul role and per cache slot kind, the
        as-shipped value, the effective value, where each came from and
        whether it changed; and the rules that select nothing here.

        Read off the walk's own layers — `tensors()` for weights, the cache
        layer for slots — never a second derivation. The one source of the
        precision panel and the response's `precision` block."""

        precision = self.resolved_precision
        profile = precision.profile
        program = self.program
        effective: List[Dict[str, Any]] = []

        def _row(category: str, selector: str, shipped: str, actual: str,
                 source: str) -> None:
            effective.append({"category": category, "selector": selector,
                              "as_shipped": shipped, "effective": actual,
                              "source": source, "changed": shipped != actual})

        # Weights: per canonical role, the describable tensors' dominant
        # saved and resident formats, by bytes.
        saved_by: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        resident_by: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        touched: set = set()
        # Roles whose weights the FRAMEWORK narrows rather than the profile:
        # a checkpoint stored wider than 16 bits is loaded at the dtype both
        # frameworks serve it at (`dtypes.served_dtype`). A separate set
        # because the source column answers "who decided this", and
        # answering "profile" for a run with an empty weight table would
        # send a reader looking for a rule that is not there.
        served: set = set()
        weight_points: set = set()
        for row in self._weight_rows():
            if row.rep is None or not row.tensor.as_dims():
                continue
            node = program.by_uid(row.uid)
            role = precision.role_value(node.role if node else None)
            stack = precision.stack_value(program.stack_of(node)
                                          if node is not None else "")
            weight_points.add((role, stack))
            elements = float(value_of(row.elements)) * float(
                value_of(row.count))
            saved = row.rep.saved
            resident = row.rep.resident
            saved_by[role][_format_label(saved)] += elements
            resident_by[role][_format_label(resident)] += elements
            if row.rep.intent is not None:
                (touched if row.rep.intent.chosen_by_profile
                 else served).add(role)
        for role in sorted(saved_by):
            # Both sides the same way: a role stored at two formats is named
            # at both. Taking the majority hid exactly the tail that makes a
            # role worth looking at — Step-5's three BF16 expert layers
            # beside 85 FP8 ones — and it hid it on the side the reader
            # trusts most, the one the checkpoint states. Whether a profile
            # touched the role decides the SOURCE column, not whether the
            # several formats a role holds are all named.
            shipped = _joined(saved_by[role])
            actual = _joined(resident_by[role])
            _row("weights", role, shipped, actual,
                 "profile" if role in touched
                 else "framework" if role in served else "checkpoint")

        # Edges: the dtype the checkpoint's own tensors are stored at, what
        # the framework loads them at, and what the activation table makes
        # of each role that produces one.
        shipped_edge = display_name(precision.shipped_model_dtype)
        served_edge = display_name(precision.model_dtype)
        # Per role where the profile's activation table makes the roles
        # differ, one row otherwise. A width a deployed framework's kernel
        # requires (`FrameworkBinding.edge_dtypes`) does NOT reach this
        # panel: it is materialized after the walk, while every published
        # route into `precision_effective` builds a framework-free one. That
        # gap is named in `docs/inference_cost_model.md` beside the other
        # fact the walk cannot see before it runs; it closes when a
        # framework's facts are resolved with the profile inside the walk,
        # not by this layer deriving them a second way.
        if profile.activations:
            for role in sorted({precision.role_value(n.role)
                                for n in program.nodes() if not n.children}):
                actual = display_name(precision.edge_dtype(
                    None if role == OUTSIDE_ROLES else role))
                _row("activations", role, shipped_edge, actual, "profile")
        else:
            stated = precision.model_dtype_source or "config"
            _row("activations", "all", shipped_edge, served_edge,
                 stated if precision.served_as_stored else "framework")

        # Matmuls: the peak column each role's matmuls run at, and the one
        # they would run at as the model ships — which for an FP8 checkpoint
        # is already FP8, because its own kernel quantises the activation
        # (`QuantScheme.activation`). Each column asks the question the
        # priced row asks (`compute_dtype` — compute pin, launch, the
        # operand promoted by a wider-held weight) of its OWN resolution:
        # this walk's for the effective column, `shipped_precision` and the
        # saved formats for the as-shipped one. Two resolutions rather than
        # one over two sets of formats, because a promotion is a weight
        # compared against its edge and both sides of that comparison move
        # with the profile: the resident table beside saved formats reports
        # a profile's own widening (a rule holding Llama's BF16 weights in
        # F32) as something the checkpoint already did, and never sees a
        # promotion the checkpoint has and the profile removed. The row is
        # the profile's only where the profile chose the width (a compute
        # pin, or a weight rule of its own, `compute_dtype_is_chosen`).
        compute_by: Dict[str, set] = defaultdict(set)
        shipped_compute: Dict[str, set] = defaultdict(set)
        chosen_roles: set = set()
        promotions = self.weight_promotions
        shipped_promotions = self.shipped_weight_promotions
        shipped_precision = self.shipped_precision
        for uid in self.formats.values:
            node = program.by_uid(uid)
            if node is None or node.children:
                continue
            # Every layer segment of every tensor: what a kernel multiplies
            # by is a question about formats, so it takes all of them.
            reps = formats_mod.segments_of(self.formats.values, uid)
            saved = tuple(rep.as_saved() for rep in reps)
            promotion = promotions.get(uid)
            shipped_promotion = shipped_promotions.get(uid)
            # A leaf enters the panel when EITHER side has something to say:
            # a profile that quantises a promoted weight takes the promotion
            # away, and that is exactly the change worth showing.
            if not (promotion is not None or shipped_promotion is not None
                    or any(rep.resident.activation_format for rep in reps)
                    or any(rep.resident.activation_format for rep in saved)):
                continue
            role = precision.role_value(node.role)
            stack = program.stack_of(node)
            # A wide unquantized weight does not carry an activation format,
            # so a resolution alone would fall back to the edge. Its
            # promotion table is the one answer to what that matmul's operand
            # is (`Traffic.operand_dtype`); each column hands its own to the
            # same resolution the row's `act_dtype` comes from.
            compute_by[role].add(precision.compute_dtype(
                node.role, stack, reps,
                operand=promotion.dtype if promotion is not None else None))
            shipped_compute[role].add(shipped_precision.compute_dtype(
                node.role, stack, saved,
                operand=(shipped_promotion.dtype
                         if shipped_promotion is not None else None)))
            if precision.compute_dtype_is_chosen(node.role, stack, reps):
                chosen_roles.add(role)
        for role in sorted(compute_by):
            actual = " + ".join(display_name(d) for d in sorted(
                compute_by[role]))
            shipped = " + ".join(display_name(d) for d in sorted(
                shipped_compute[role]))
            _row("compute", role, shipped, actual,
                 "profile" if role in chosen_roles else "checkpoint")

        # Cache: per slot kind present, shipped vs effective format.
        kinds: Dict[str, Tuple[set, set, set]] = {}
        for uid, entry in cache_mod.derive(self).values.items():
            keeper = self.program.by_uid(uid)
            for tensor in entry.tensors:
                if tensor.kind is None:
                    continue
                shipped, actual, sources = kinds.setdefault(
                    tensor.kind, (set(), set(), set()))
                ship = cache_mod.bind_format(
                    precision.shipped_cache_format(tensor.kind,
                                                   tensor.declared),
                    self, keeper)
                shipped.add(ship.label())
                actual.add(tensor.resolved_format.label())
                sources.add("profile" if tensor.resolved_format != ship
                            else "spec" if ship != CacheFormat.plain(
                                precision.model_dtype)
                            else "model")
        for kind in sorted(kinds):
            shipped, actual, sources = kinds[kind]
            _row("cache", kind, " + ".join(sorted(shipped)),
                 " + ".join(sorted(actual)),
                 "profile" if "profile" in sources else sorted(sources)[0])

        inapplicable: List[Dict[str, str]] = [
            {"rule": rule.describe(),
             "reason": "selects no weight matrix of this model"}
            for rule in profile.weights
            if not any(rule.selects({"roles": r, "stacks": st})
                       for r, st in weight_points)]
        for rule in profile.cache:
            dims = rule.dims()
            if not dims["slots"] & set(kinds):
                inapplicable.append({
                    "rule": rule.describe(),
                    "reason": (f"this model keeps no "
                               f"{' / '.join(sorted(dims['slots']))} cache")})
        return {
            **profile.to_json(),
            "overridden": any(row["changed"] for row in effective),
            "effective": effective,
            "inapplicable": inapplicable,
        }

    @cached_property
    def replica_cost(self) -> Layer[Cost]:
        """`cost` as ONE data-parallel replica's step: each node's FLOPs at
        `dp_attention = 1`, everything else as it is.

        `cost` is each region's own whole work, and the regions do not agree
        on what that is under DP-attention: an attention replica runs its own
        `B·s` tokens, while the expert bank they share receives every
        replica's (`Evaluator._tokens_per_expert` bins `M` with
        `dp_attention`). Summed, that is neither one replica's step nor the
        deployment's — DeepSeek-V3 at DP8/EP8 came out `A + 8X`. A replica
        is the unit every other count here is per (the step's latency, the
        batch, a direct estimate's table), so this is the unsharded figure
        to quote; `cost` keeps the bank's own whole, which the per-rank
        projection and the route's measure preservation read.

        Read off each leaf's FLOPs law, where the replica degree stays a
        symbol (`M`'s identity; the degree is the one `dims.yaml` declares
        `bound_per_rank`): the law at 1 over the law, at what the leaf's frame
        bound its dims to — `1/dp_attention` on a bank's own leaves. A
        component is its cost less what its children's pooling added, at its
        own copy multiple, as the walk built its cost from theirs."""
        out: Dict[Any, Cost] = {}
        # The degrees a replica is one of: the worlds whose axes arrive at one
        # rank's extent (`dims.World.bound_per_rank` — DP-attention).
        replicas = {world.degree for world in dims_mod.world_by_label().values()
                    if world.bound_per_rank}

        def replica(uid: Any) -> Cost:
            if uid in out:
                return out[uid]
            cost = self.cost.values[uid]
            node = self.program.by_uid(uid)
            if node is None:
                raise ValueError(f"node {uid} has a cost and is not in the program")
            if node.children:
                # A component's cost is its costed children's times its own
                # copies (`Evaluator` applies `own_copies` to every node), so
                # its replica is its cost less what its children's pooling
                # added, at the same multiple. A child with no cost adds
                # nothing to either side: its work stays at the cost's, never
                # summed short to zero.
                own = self.own_copies_of(node)
                children = [c.uid for c in node.children
                            if c.uid in self.cost.values]
                pooled = Cost.combine(
                    [self.cost.values[c] for c in children]).flops - Cost.combine(
                    [replica(c) for c in children]).flops
                excess = Cost(flops=pooled).scaled(storage=own.stored,
                                                   active=own.active).flops
                out[uid] = replace(cost, flops=cost.flops - excess)
                return out[uid]
            out[uid] = cost
            law = cost.flops.expr
            for symbol in law.free_symbols:
                if symbol.name not in replicas:
                    continue
                # The law at one replica over the law, read at what this
                # leaf's frame bound each of its dims to.
                at_one = law.subs(symbol, 1)
                ratio = sp.simplify(at_one / law)
                frame = self.node_dims.values.get(uid) or {}
                try:
                    share = ratio.subs({
                        sym: value_of(frame[sym.name] if sym.name in frame
                                      else self.binding.raw(sym.name))
                        for sym in ratio.free_symbols})
                except KeyError as exc:
                    raise ValueError(
                        f"node {uid}: one replica's share of its FLOPs law "
                        f"{law} reads {exc.args[0]!r}, which this walk did "
                        f"not bind") from None
                held = out[uid]
                out[uid] = replace(held, flops=Dim(at_one, held.flops.value * share))
                law = at_one
            return out[uid]

        for uid in self.cost.values:
            replica(uid)
        return Layer(self.program, out)

    @cached_property
    def occurrence_graph(self) -> OccurrenceGraph:
        """Every edge of the walked program's forward, at occurrence
        granularity, as the walk bound it (`input_references`). A rewritten
        analysis reads its walk's (`walked`): the graph is over the walked
        program (`OccurrenceGraph.program`), not the rewritten one."""
        return OccurrenceGraph.of(self.reference_table)

    @property
    def reference_table(self) -> ReferenceTable:
        """The walk's `ReferenceTable`: this analysis's own, or, carried
        across a rewrite, its walk's (`walked`)."""
        table = (self.walked if self.walked is not None
                 else self).input_references
        if table is None:
            raise ValueError(
                f"{self.program.spec.name}: this analysis was not produced by "
                f"a walk and holds no references")
        return table

    def input_reference(self, uid: int, index: int) -> Reference:
        """What input `index` of walked node `uid` reads, as the walk bound
        it. A node the walk did not evaluate — one a rewrite made — is
        refused."""
        return self.reference_table.at(uid, index)

    def model_input_read(self, uid: int, index: int) -> Optional[str]:
        """The model input input `index` of walked node `uid` reads, as the
        walk bound it; None when it reads anything else."""
        return self.input_reference(uid, index).model_input

    def own_copies_of(self, node: Any) -> OwnCopies:
        """`node`'s own multipliers, or a refusal: a node the walk did not
        count — or a fused node whose members run at different multipliers
        — has no own count, and defaulting it to one copy is a silent
        under-count."""
        copies = self.own_copies.values.get(node.uid)
        if copies is None:
            raise ValueError(
                f"{node.display_path}: no own copy count in this analysis")
        return copies

    def block_frame(self, template: Any) -> Mapping[str, Any]:
        """`{dim: value}` of the frame the nodes `template` lays out
        (`BlockTemplate.shown`) ran in, as this walk bound it: the stage's
        frame (`stage_frames`), or — where what is shown is the body of an
        isolated component's site call — the frame that call opened
        (`node_dims` of the call, its own `dims:` over the stage's). The one
        answer to "what does this block bind a dim to"; a reader neither
        opens the frame again nor takes one node's frame for the block's.
        Empty for a block this walk did not run (an encoder's, outside its
        encode phase), which binds nothing here as it costs nothing here."""
        call = template.site_call
        if call is not None:
            return self.node_dims.values.get(call.uid, {})
        frame = self.stage_frames.get(self.program.block_index(template))
        return frame.flat() if frame is not None else {}

    @property
    def moe_pools(self) -> Tuple[MoeFacts, ...]:
        """Every routed expert pool the walk evaluated, backbone first: the
        distinct numbers its banks run at (`MoeFacts`). Empty for a model
        with no dispatch zone, even one that binds `n_experts` (a dense spec
        may).

        The routed nodes whose stored weights lead with the expert axis
        (`shape: [n_expert_copies, ...]`) are the expert bank: that axis's
        extent is the copies expert parallelism divides (never the routing
        count), and a token's routing touches `n_experts_active /
        n_route_targets` of them. The routing
        counts (`n_experts`, `n_experts_active`, `n_route_targets`), the
        shared-expert count and the expert width (`d_ffn_expert`) are the
        spec's registered dims as the walk bound them in the bank's frame
        (`node_dims`) — LongCat's 768 route targets include 256 zero-compute
        experts no node stands for, and Kimi-K3 fuses its two shared experts
        into one node of twice the width, so neither is a count of nodes.
        The width is None when banks of one pool bind different values
        (never averaged).

        Banks bound to the same numbers are one pool — one bank per layer of
        it — and banks bound to different numbers are different pools, which
        is why the numbers are read in each bank's own frame and not off the
        model's global dims.
        """
        return tuple(self._moe_pool_stacks)

    @property
    def moe(self) -> Optional[MoeFacts]:
        """The backbone's expert pool (`moe_pools`) — the routing a model
        card states as its expert count and top-k. None for a model with no
        dispatch zone; the only pool when the backbone routes over none
        (every routed bank sits in an auxiliary stack)."""
        pools = self.moe_pools
        if not pools:
            return None
        backbone = [pool for pool, stacks in self._moe_pool_stacks.items()
                    if "" in stacks]
        if len(backbone) > 1:
            raise ValueError(
                f"the backbone's routed nodes disagree on their expert copies "
                f"({sorted(p.n_expert_copies for p in backbone)}) or active "
                f"copies ({sorted(p.active_copies for p in backbone)})")
        if backbone:
            return backbone[0]
        if len(pools) > 1:
            raise ValueError(
                "the model routes over several expert pools and none of them "
                "is the backbone's")
        return pools[0]

    @cached_property
    def routed(self) -> Dict[int, RoutedLaunch]:
        """Each MoE dispatch's boundary on the rank this analysis sits on
        (`RoutedLaunch`: the tokens the rank holds, the rows routed onto its
        experts, its experts, and the padded launch they make), by the
        dispatch's uid. Which dispatch a node's buffer came from is
        `Program.dispatch_of`.

        Per rank, so the answer is the rule's, as `local_tensors` is: the
        identity `Rule` reads the walk's own shapes, `LayoutPropagation`
        the same tensors at its rank."""
        return self.rule.routed(self)

    def media_rows(self) -> Dict[str, "MediaRows"]:
        """Each media input this walk's encoder reads, as it arrives
        (`MediaRows`): the rows a processor's cut (`processor_step: cut`,
        `patchify`) makes of an image (else the input itself), padded where
        a pad (`processor_step: pad`) reads them — the input itself, or the
        cut's rows along its `in:` edge. An encode walk's fact; a
        language-model walk reads no media and has none.

        Every reader of one input is held to the same answer, and a cut
        read by pads of different row counts has no one answer: both are
        refused rather than settled by walk order."""
        program = self.program
        walked = {name for name, entry in program.walked_inputs.items()
                  if entry.encoder}
        if not walked:
            return {}
        nodes = list(program.walked_nodes())
        out: Dict[str, MediaRows] = {}
        for node in nodes:
            for index in range(len(node.inputs_obj or ())):
                name = self.model_input_read(node.uid, index)
                if name not in walked:
                    continue
                shape = self.shapes.values[node.uid]
                step = node.processor_step
                base = (shape.out if step is ProcessorStep.CUT
                        else shape.ins[index])
                if step is ProcessorStep.PAD:
                    rows = shape.out[1]
                elif step is ProcessorStep.CUT:
                    rows = self._padded_rows(node, nodes, base[1])
                else:
                    rows = base[1]
                found = MediaRows(
                    name=name, segments=int(value_of(base[0])),
                    rows=int(value_of(rows)),
                    valid_rows=int(value_of(base[1])))
                if out.setdefault(name, found) != found:
                    raise ValueError(
                        f"media input {name!r} arrives at its encoder as "
                        f"{out[name]} and, at {node.display_path}, as "
                        f"{found}; one input arrives one way")
        return out

    def _padded_rows(self, cut: Any, nodes: Sequence[Any], rows: Any) -> Any:
        """The rows each item of `cut`'s output is computed at: the count
        the pads reading it on their `in:` edge pad to, or its own `rows`
        where no pad reads it."""
        padded = {int(value_of(self.shapes.values[node.uid].out[1]))
                  for node in nodes if node.processor_step is ProcessorStep.PAD
                  and any(ref.src is cut for ref in node.inputs_obj or ())}
        if len(padded) > 1:
            raise ValueError(
                f"{cut.display_path}: the pads reading its rows pad them to "
                f"{sorted(padded)}; an item's rows are padded to one count")
        return padded.pop() if padded else rows

    @cached_property
    def attention_families(self) -> Tuple[AttentionFamily, ...]:
        """The backbone's attention families (`AttentionFamily`), in layer
        group order; groups whose call has the same family and reads the
        same dims at the same values are one family. Each dim's value is
        the one the walk bound in the call's frame (`node_dims`). The
        model's width (`d_model`), which every call reads, is the
        headline's, not a family's."""
        if self.walked is not None:
            return self.walked.attention_families
        program = self.program
        declared = list(program.resolved.dim_values)
        families: Dict[Tuple[Any, ...], AttentionFamily] = {}
        for template in program.stack_blocks():
            call = template.attention_call()
            if call is None:
                continue
            frame = self.node_dims.values.get(call.uid)
            if frame is None:
                raise ValueError(
                    f"{call.display_path}: an attention call this walk bound "
                    "no frame for")
            reads = program.reads(call)
            # The frame holds the call's own `dims:` over the enclosing
            # binding; a name it binds to anything but itself is not the
            # model's dim there.
            shadowed = {name for name, bound in (call.dims_raw or {}).items()
                        if name in reads and str(bound).strip() != name}
            if shadowed:
                raise ValueError(
                    f"{call.display_path}: its own `dims:` rebind "
                    f"{sorted(shadowed)}, which its body also reads under the "
                    "model's name; its frame has no value for the model's")
            rebound = set(template.dim_overrides)
            cursor = call.parent
            while cursor is not None:
                rebound |= set(cursor.dims_raw or {})
                cursor = cursor.parent
            dims = []
            for name in declared:
                if name not in reads or name == "d_model":
                    continue
                value = value_of(frame.get(name))
                if value is not None:
                    dims.append((name, int(value)))
            rebound &= {name for name, _ in dims}
            key = (call.kv_label, call.attn_variant, tuple(dims),
                   frozenset(rebound))
            found = families.get(key)
            families[key] = AttentionFamily(
                label=call.kv_label, variant=call.attn_variant,
                layers=template.count + (found.layers if found else 0),
                dims=tuple(dims), rebound=frozenset(rebound))
        return tuple(families.values())

    @cached_property
    def _moe_pool_stacks(self) -> Mapping[MoeFacts, frozenset]:
        """`{pool: stacks its banks sit in}`, backbone's pool first, then
        in walk order."""
        if self.walked is not None:
            return self.walked._moe_pool_stacks
        program = self.program
        if not program.has_dispatch_zone:
            return {}
        banks = [n for n in program.walked_nodes()
                 if program.is_routed(n.uid)
                 and any(storage_mod.leads_with_expert_axis(t)
                         for t in self.storage.values.get(n.uid) or ())]
        if not banks:
            raise ValueError(
                "the model has a dispatch zone but no routed node stores a "
                "weight led by the expert axis (`shape: [n_expert_copies, "
                "...]`) — no expert bank to read the copies from")
        grouped: Dict[Tuple[Any, ...], List[Any]] = {}
        for n in banks:
            frame = self.node_dims.values.get(n.uid)
            if frame is None:
                raise ValueError(
                    f"{n.display_path}: an expert bank this walk bound no "
                    "frame for")
            # The bank's experts are its weights' leading axis, and a token
            # runs `n_experts_active / n_route_targets` of them.
            (lead,) = {int(value_of(t.shape[0]))
                       for t in self.storage.values[n.uid]
                       if storage_mod.leads_with_expert_axis(t)}
            routing = []
            for name in ("n_experts", "n_experts_active", "n_route_targets"):
                value = value_of(frame.get(name))
                if value is None:
                    raise ValueError(
                        f"the model routes tokens to experts but binds no "
                        f"`{name}` dim")
                routing.append(int(value))
            shared = value_of(frame.get("n_shared_experts"))
            width = value_of(frame.get("d_ffn_expert"))
            _experts, top_k, targets = routing
            # Refused, not floored, when a token would run a fraction of an
            # expert (`exact_div`).
            active = int(value_of(exact_div(lead * top_k, targets)))
            key = (lead, active, *routing, int(shared or 0))
            grouped.setdefault(key, []).append(
                (n, None if width is None else int(width)))
        pools: Dict[MoeFacts, set] = {}
        for (copies, active, experts, top_k, targets, shared), members \
                in grouped.items():
            widths = {width for _n, width in members}
            pool = MoeFacts(
                n_expert_copies=copies,
                active_copies=active,
                n_experts=experts,
                n_experts_active=top_k,
                n_route_targets=targets,
                d_ffn_expert=widths.pop() if len(widths) == 1 else None,
                n_shared_experts=shared,
            )
            pools.setdefault(pool, set()).update(
                program.stack_of(n) for n, _width in members)
        ordered = sorted(pools.items(), key=lambda item: "" not in item[1])
        return {pool: frozenset(stacks) for pool, stacks in ordered}

    def layer_multiplicity(self) -> Layer[int]:
        """How many layers of each accounting contribution the model runs:
        every root at its count (`Program.multiplicity`). One pipeline
        stage's share is `layout.stages.StageView.rows`."""
        return self.program.multiplicity()

    def rank_definitions(
        self, definitions: Mapping[str, Any], global_shape: Shape,
        local_shape: Shape, node: Any, *,
        query_shapes: Optional[Tuple[Shape, Shape]] = None,
        query_node: Any = None,
    ) -> Dict[str, Any]:
        """This analysis's rank-law reading, computed once for exact inputs.

        An `Analysis` owns the rule a rank projection reads, and the rule
        answers it (`Rule.rank_definitions`: none under the identity rule).  The projection
        keys each law by the law, its rank bindings and its cut query names;
        global/local/query faces have already been consumed into those
        bindings.  Repeated physical/serving/liveness readers of one cell
        share this reading rather than re-deriving equal symbolic shapes.
        """
        return self.rule.rank_definitions(
            self, definitions, global_shape, local_shape, node,
            query_shapes=query_shapes, query_node=query_node,
            cache=self._rank_definition_readings)

    def physical_shapes(self) -> Any:
        """This analysis's per-rank semantic shapes, derived once.

        Physical, cache, access and serving readers can each ask for the
        projection; all of them read the same analysis and must see the same
        shape objects.  Keeping the projection here removes both repeated
        SymPy work and equal-but-distinct shape copies that would fragment
        downstream identity caches. The rule answers it (`Rule.rank_shapes`:
        the walk's own shapes under the identity rule)."""
        if self._physical_shape_reading is None:
            self._physical_shape_reading = self.rule.rank_shapes(self)
        return self._physical_shape_reading

    def recount(
        self,
        counts: Layer[Any],
        layer_multiplicity: Layer[int],
    ) -> Layer[int]:
        """Scale a node count layer by one resolved layer selection.

        ``counts`` may be evaluator instances, physical artifact copies, or any
        other cumulative node count.  The pipeline-layer ratio is applied here
        once with exact integer checks; consumers do not know the
        ``kept/source`` formula.
        """
        if counts.program is not self.program:
            raise ValueError("count layer belongs to another Program")
        if layer_multiplicity.program is not self.program:
            raise ValueError("layer multiplicity belongs to another Program")
        source = self.layer_multiplicity()
        values: Dict[int, int] = {}
        for uid, raw in counts.values.items():
            source_count = source.values.get(uid)
            kept = layer_multiplicity.values.get(uid)
            if source_count is None or kept is None:
                raise ValueError(f"count layer uid {uid} has no layer multiplicity")
            value = int(value_of(raw))
            if source_count == 0:
                if value or kept:
                    raise ValueError(
                        f"zero-layer node uid {uid} has count {value} and keeps "
                        f"{kept} layers"
                    )
                values[uid] = 0
                continue
            numerator = value * kept
            if numerator % source_count:
                node = self.program.by_uid(uid)
                raise ValueError(
                    f"{node.display_path}: stage count {value} * {kept} / "
                    f"{source_count} is not integral"
                )
            values[uid] = numerator // source_count
        return Layer(self.program, values)

    def tensors_of(self, uid: int) -> Iterator[TensorRow]:
        """The `tensors()` join for one node."""
        yield from pair_stored(
            uid,
            self.storage.values.get(uid),
            self.formats.values.get(uid),
            self.instances.values.get(uid, 1),
        )

    def tensors(self) -> Iterator[TensorRow]:
        """Every tensor the model stores, as a `TensorRow`.

        The one place `storage` and `formats` are zipped, and the one place the
        instance count is attached. Both are alignment facts that a consumer
        would otherwise have to know: that the two layers' tuples are index-
        parallel, and that a per-node value is per-COPY. Getting either wrong is
        silent — a bias paired with its weight's format, an expert bank counted
        once — so it is done here and not in each caller.
        """
        for uid in self.storage.values:
            yield from self.tensors_of(uid)

    def checkpoint_copy_rows(self) -> Iterator[TensorRow]:
        """The stored copies of tied weights the checkpoint carries and
        serving skips, joined like `tensors()` — and, like
        `checkpoint_tensors()`, at the format the FILE has: a copy serving
        never loads has no loaded width to report."""
        for uid, (stored, reps) in self.checkpoint_copies.values.items():
            for row in pair_stored(uid, stored, reps,
                                   self.instances.values.get(uid, 1)):
                yield (row if row.rep is None
                       else row._replace(rep=row.rep.as_saved()))

    def checkpoint_tensors(self) -> Iterator[TensorRow]:
        """Every tensor the checkpoint FILE holds: what the model stores
        (`tensors()`) plus the declared redundant copies of tied weights,
        with a tensor the file splits (`stored_split`) as its parts.

        `tensors()` is what a deployment holds; this is what the file holds.
        They differ by `checkpoint_copy_rows()`, the splits — and by the
        FORMAT each row carries: a row here drops its `intent`, so every
        byte below is the one the file has. That is the whole point of the
        projection and it used to be true only by accident, because nothing
        gave an as-shipped walk an intent. gemma-2 does: its weights are
        saved F32 and loaded BF16 (`dtypes.served_dtype`), and with the
        resident format on these rows the byte gate would have compared the
        loaded width against the stored one and called a correct spec wrong
        by a factor of two.
        """
        rows = [row if row.rep is None else row._replace(
            rep=row.rep.as_saved()) for row in self.tensors()]
        # How many TENSORS the node stores, not how many rows: a tensor the
        # node stores at several formats across its layers is several rows of
        # one tensor, and `stored_split` is about siblings, not segments.
        per_uid: Dict[int, set] = {}
        for row in rows:
            per_uid.setdefault(row.uid, set()).add(row.tensor_index)
        for row in rows:
            yield from self._stored_parts(row, len(per_uid[row.uid]))
        yield from self.checkpoint_copy_rows()

    def _stored_parts(self, row: TensorRow, rows_of_node: int
                      ) -> Iterator[TensorRow]:
        """`row` as the file stores it: itself, or — under the node's
        `stored_split` — its `parts` consecutive blocks along the split axis,
        each named `<module>.<segment>` + the suffix.

        The split axis is an axis of the op's logical `weight_shape`: the
        table is cut there first, and each block's payload is then laid out
        in the checkpoint's axis order (`stored_axes`: file axis i is
        logical axis `stored_axes[i]`), so a block row has the name and the
        shape its file tensor has."""
        node = self.program.by_uid(row.uid)
        declared = node.stored_split
        if declared is None or row.is_metadata or row.is_buffer:
            yield row
            return
        split, parts = declared
        if rows_of_node > 1 or row.metadata:
            raise ValueError(
                f"{row.name}: `stored_split` is declared on a weight that "
                f"stores sibling tensors ({rows_of_node} stored rows, "
                f"{len(row.metadata)} format metadata). The split of a "
                f"sibling is not declared; `stored_split` applies to a "
                f"weight stored alone")
        logical = list(row.tensor.shape)
        logical[split.axis] = logical[split.axis] // parts
        order = node.stored_axes or tuple(range(len(logical)))
        if sorted(order) != list(range(len(logical))):
            raise ValueError(
                f"{row.name}: `stored_axes` {list(order)!r} is not a "
                f"permutation of the {len(logical)} axes `stored_split` cuts")
        shape = [logical[axis] for axis in order]
        for index in range(parts):
            tensor = replace(
                row.tensor,
                module=f"{row.tensor.module}.{split.segment_name(index)}",
                shape=tuple(shape))
            yield TensorRow(row.uid, tensor, row.rep, row.count,
                            row.tensor_index, row.segment_index)

    def tensor_index(self) -> Dict[str, TensorRow]:
        """`{aggregation_id → stored tensor}`, from the same join.

        The flat keyspace the renderer and the validate matcher share:
        renderer-dict node ids and report ids are aggregation_ids, so both
        do a direct equality lookup instead of re-walking `tensors()` and
        re-deriving "which row is the stored one". Metadata siblings and
        buffers are not weights and stay out; the first row wins where one
        node stores several primaries.
        """
        index: Dict[str, TensorRow] = {}
        for row in self.tensors():
            if row.is_metadata or row.is_buffer:
                continue
            agg = self.program.by_uid(row.uid).aggregation_id
            if agg and agg not in index:
                index[agg] = row
        return index

    def iter_checkpoint(self) -> Iterator[TensorRow]:
        """What the checkpoint FILE holds: `checkpoint_tensors()` plus the
        metadata siblings its SAVED formats imply.

        Metadata is not a graph node — it hangs off the format — but the
        checkpoint lists it as its own entry. Matcher, packing, names and
        dumps read this projection rather than a second fusion.
        """
        yield from self._with_metadata(self.checkpoint_tensors())

    def iter_resident(self) -> Iterator[TensorRow]:
        """What this deployment HOLDS: `tensors()` plus the metadata
        siblings its resident formats imply.

        The same flattening as `iter_checkpoint()`, read off the other side
        of `Representation`, and the pair exists because the two answers are
        not the same tensor list: a profile that quantises a BF16 weight
        invents scales no file has, and a checkpoint stored wider than the
        framework loads it (gemma-2's F32) holds bytes no rank does. Asking
        the file's list for a deployment's bytes is how a weight rule came
        to move nothing at all.
        """
        yield from self._with_metadata(self.tensors())

    def _with_metadata(self, rows: Iterable[TensorRow]
                       ) -> Iterator[TensorRow]:
        """Each row, then its format's metadata siblings as rows of their
        own — the flattening both projections above share."""
        for row in rows:
            yield row
            for meta in row.metadata:
                fmt = formats_mod.TensorFormat(storage_dtype=meta.dtype or "F32")
                # A sibling is the same module as its weight, under the
                # path this row of the weight is stored at.
                meta = replace(meta, module=row.tensor.module)
                yield TensorRow(
                    row.uid, meta, formats_mod.Representation(saved=fmt), row.count,
                    row.tensor_index, row.segment_index,
                )

    def _weight_rows(self) -> Iterator[TensorRow]:
        """The quantizable weights: every stored tensor that is neither a
        format's metadata nor a load-time buffer. Excluded by what it IS
        (`StorageKind`), not by who declared its op — a spec-declared op's
        weight (gpt-oss's MXFP4 expert banks, the largest tensors in the
        model) is a weight, and leaving it out read the model as uniform
        BF16."""
        for row in self.tensors():
            if row.is_metadata or row.is_buffer:
                continue
            yield row

    def checkpoint_param_bytes(self, *, without_encoders: bool = False) -> int:
        """On-disk bytes of learnable weights.

        Metadata siblings and buffers are not parameters; a buffer the spec
        declares is one the checkpoint serialises, and its bytes are
        `checkpoint_buffer_bytes`. `without_encoders` leaves out the encoder
        stacks' tensors (`Stack.is_encoder`): the language model's share.
        """
        program = self.program
        encoders = frozenset(stack.name for stack in program.stacks
                             if stack.is_encoder) if without_encoders else ()
        total = 0
        for row in self.checkpoint_tensors():
            if row.is_metadata or row.is_buffer:
                continue
            if encoders and program.stack_of(program.by_uid(row.uid)) in encoders:
                continue
            total += int(row.storage_bytes) * int(value_of(row.count))
        return total

    def checkpoint_buffer_bytes(self) -> int:
        """On-disk bytes of the precomputed buffers the spec declares (a
        serialised `inv_freq` table): not parameters, and not a quant
        scheme's sidecar either — in the header they sit with the weights."""
        return sum(int(row.storage_bytes) * int(value_of(row.count))
                   for row in self.checkpoint_tensors() if row.is_buffer)

    def stored_bytes_by_stack(self) -> Dict[str, int]:
        """Bytes the model stores per stack (`tensors()`, format metadata
        included), at this walk's precision: what loading a stack costs,
        so a deployment that leaves one out can say how much it left."""
        program = self.program
        out: Dict[str, int] = {}
        for row in self.tensors():
            stack = program.stack_of(program.by_uid(row.uid))
            out[stack] = (out.get(stack, 0)
                          + int(row.storage_bytes) * int(value_of(row.count)))
        return out

    def weight_bytes_by_dtype(self) -> Dict[str, int]:
        """RESIDENT weight bytes per logical dtype name — the quantizable
        weights only (`_weight_rows`: no scales, no buffers), so a
        mixed-precision checkpoint says how much of it each dtype holds.

        The width a rank holds, not the one the file stores: `_weight_rows`
        reads `tensors()`, whose format is `Representation.resident`. These
        are the figures the precision label on the model-config panel is
        built from (`rendering._quant`), and that label should say what is
        being run — gemma-2 reads BF16 here against an F32 checkpoint. The
        file's own per-dtype bytes are `iter_checkpoint`'s."""
        return self._bytes_by(self._weight_rows(), _dtype_name)

    def metadata_bytes_by_dtype(self) -> Dict[str, int]:
        """On-disk bytes of the quantisation metadata (scales, zero points)
        per logical dtype — every metadata row `iter_checkpoint` lists, each
        at its own width. A checkpoint mixing 1-byte ue8m0 and 4-byte F32
        scales sums each at its own size; no count is multiplied by one
        dominant width.

        On-disk and it means it, unlike the two around it: `iter_checkpoint`
        is the FILE's list, so a profile's invented scales are not in here
        and the file's own are, whatever the deployment runs."""
        return self._bytes_by(
            (row for row in self.iter_checkpoint() if row.is_metadata),
            _dtype_name)

    def weight_bytes_by_method(self) -> Dict[str, int]:
        """RESIDENT weight bytes per quant method covering them (the
        tensor's format's scheme); weights no scheme covers are left out.
        Which scheme holds most of a mixed checkpoint, by the same join."""
        out = self._bytes_by(self._weight_rows(), lambda row: row.method or "")
        out.pop("", None)
        return out

    @staticmethod
    def _bytes_by(rows: Iterable[TensorRow],
                  key: Callable[[TensorRow], str]) -> Dict[str, int]:
        """On-disk bytes of `rows` grouped by `key`, every copy counted."""
        totals: Dict[str, int] = {}
        for row in rows:
            mult = int(value_of(row.count))
            if mult <= 0:
                continue
            k = key(row)
            totals[k] = totals.get(k, 0) + int(row.storage_bytes) * mult
        return totals

    def dominant_weight_dtype(self) -> Optional[str]:
        """Logical dtype name that holds the most on-disk weight bytes."""
        bytes_by_dtype = self.weight_bytes_by_dtype()
        if not bytes_by_dtype:
            return None
        return max(bytes_by_dtype.items(), key=lambda kv: kv[1])[0]

    def dominant_cache_dtype(self) -> str:
        """The element dtype of the cached tensor holding the most bytes per
        stored position — the cache dtype a single-dtype lookup keys on.
        The model's own dtype when nothing is cached."""
        best: Tuple[float, str] = (-1.0, self.resolved_precision.model_dtype)
        for entry in cache_mod.derive(self).values.values():
            for tensor in entry.tensors:
                size = float(value_of(tensor.data_position_bytes))
                if tensor.is_growing and size > best[0]:
                    best = (size, tensor.dtype)
        return best[1]

    def dominant_weight_dtype_bytes(self) -> Optional[float]:
        """Bytes per logical element of the dominant weight dtype."""
        name = self.dominant_weight_dtype()
        if name is None:
            return None
        return formats_mod.known_dtype_bytes(name)

    def weight_dtype_is_uniform(self) -> bool:
        """True when every quantizable weight shares one logical dtype.

        FP32 norms/biases are not a quantization target and do not make a
        model mixed-precision for serving overrides.
        """
        seen: set = set()
        for row in self._weight_rows():
            dt = (row.logical_dtype or row.storage_dtype or "").upper()
            if canonical(dt) == "F32":
                continue
            seen.add(dt)
            if len(seen) > 1:
                return False
        return True

    def dominant_weight_scale_dtype(self) -> Optional[str]:
        """Most common scale-sibling dtype, in the runtime vocabulary.

        Reads the sibling's LOGICAL dtype — what the scale is, not the slot it
        is stored in — so an MXFP4 scale declared `dtype: U8` with
        `logical_dtype: F8_E8M0FNU` answers `ue8m0`. A scale dtype with no
        runtime token is an error, not a lower-cased slot name that no
        calibration row will ever match.
        """
        counts: Dict[str, int] = {}
        for row in self.iter_checkpoint():
            # The scheme said which sibling is the scale when it declared it
            # (`kind: scale`); the tensor carries that answer.
            if not row.is_metadata or row.tensor.sibling_kind != "scale":
                continue
            name = row.name or ""
            dtype = (row.logical_dtype or "").upper()
            try:
                runtime = corpus_token(dtype)
            except UnknownDtype:
                raise ValueError(
                    f"scale tensor {name!r} is {dtype!r}, which has no runtime "
                    f"dtype token; declare `logical_dtype:` on its scheme "
                    f"sibling") from None
            mult = int(value_of(row.count))
            if mult <= 0:
                continue
            counts[runtime] = counts.get(runtime, 0) + mult
        if not counts:
            return None
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    @cached_property
    def load_refusals(self) -> Tuple[str, ...]:
        """Why this deployment's selected loaders cannot materialize a rank."""
        return tuple(dict.fromkeys(
            rep.resident.load_refusal
            for tensors in self.formats.values.values()
            for reps in tensors
            for rep in reps
            if rep.resident.load_refusal is not None))

    def with_load_formats(self, loaders: Mapping[str, str]) -> "Analysis":
        """This walk with the resident representations its loader creates.

        The checkpoint side of every representation is preserved. The caller
        resolves each layout kind to a framework's loader and passes only that
        plain mapping; metadata, residency and rank projection share one format.
        """
        values = {}
        changed = False
        for uid, tensors in self.storage.values.items():
            formats = []
            for index, (stored, reps) in enumerate(
                    zip(tensors, self.formats.values[uid], strict=True)):
                segments = []
                for segment_index, rep in enumerate(reps):
                    contract = rep.resident.block_scale_layout
                    loaded = rep.resident
                    loader = (loaders.get(contract.kind)
                              if contract is not None else None)
                    if loader is not None:
                        axes = self._axis_segments[(uid, index, segment_index)]
                        local_stored = replace(stored, shape=tuple(
                            sum(local for _, local in parts) for parts in axes))
                        loaded = rep.resident.for_loader(
                            stored, local_stored, loader=loader,
                            runtime_tp=self.rule.world_degree(contract.world))
                    changed |= loaded is not rep.resident
                    segments.append(rep if loaded is rep.resident
                                    else replace(rep, intent=loaded))
                formats.append(formats_mod.Representations.of(*segments))
            values[uid] = tuple(formats)
        return (replace(self, formats=Layer(self.program, values))
                if changed else self)

    @cached_property
    def _axis_segments(self) -> Dict[Any, Any]:
        return self.rule.local_axis_segments(self)

    def weight_shards(self, packs: Sequence[Sequence[int]]
                      ) -> Tuple[WeightShard, ...]:
        """The matrices a kernel multiplies, partition by partition, beside
        one rank's extents of them (`WeightShard`).

        `packs` groups the nodes whose matrices one launch reads: each group
        a set of nodes whose weights it concatenates into one (a
        sibling-merging fusion's origins), every other node a group of its
        own. A matrix is a stored tensor a GEMM contracts — `MATRIX` or
        `HEADWISE`; vectors, tables and buffers are none, and a format's
        metadata follows its weight.
        """
        out = []
        for group, members in enumerate(packs):
            for uid in members:
                for row in self.tensors_of(uid):
                    if row.tensor.kind not in (storage_mod.StorageKind.MATRIX,
                                               storage_mod.StorageKind.HEADWISE):
                        continue
                    key = (uid, row.tensor_index, row.segment_index)
                    axes = self._axis_segments.get(key)
                    if axes is None or len(axes) < 2:
                        raise ValueError(
                            f"node {uid}: stored tensor {row.tensor_index} has "
                            f"no rank-local partitions")
                    if any(len(parts) > 1 for parts in (*axes[:-2], axes[-1])):
                        raise ValueError(
                            f"node {uid}: stored tensor {row.tensor_index} "
                            f"packs partitions along an axis that is not its "
                            f"matrix's rows; no scale grid is read that way")
                    lead = tuple(local for ((_g, local),) in axes[:-2])
                    ((_width, width_local),) = axes[-1]
                    resident = (row.rep.resident if row.rep is not None
                                else None)
                    granularity = (resident.weight_granularity
                                   if resident is not None else None)
                    load_layout = None
                    if resident is not None and resident.block_scale_layout is not None:
                        contract = resident.block_scale_layout
                        held = replace(row.tensor, shape=(
                            *lead, sum(local for _, local in axes[-2]), width_local))
                        _, rows_held = held.matrix_dims
                        load_layout = {
                            **contract.model_dump(),
                            "loader": resident.loader,
                            "load_refusal": resident.load_refusal,
                            "runtime_tp": self.rule.world_degree(contract.world),
                            "rows_local": rows_held,
                        }
                    for size, local in axes[-2]:
                        whole = replace(row.tensor, shape=(
                            *row.tensor.shape[:-2], size, row.tensor.shape[-1]))
                        held = replace(row.tensor,
                                       shape=(*lead, local, width_local))
                        width, rows = whole.matrix_dims
                        width_held, rows_held = held.matrix_dims
                        out.append(WeightShard(
                            (group, row.segment_index), rows, rows_held,
                            width, width_held, granularity, load_layout))
        return tuple(out)

    def local_tensors(self) -> Iterator["TensorRow"]:
        """The same join as `tensors()`, at the rank this analysis sits on.

        `tensors()` stays the logical face. The per-rank answer
        is the rule's: `LayoutPropagation` projects each stored tensor onto
        the runtime shape and copy count one rank holds, and the identity
        `Rule` answers with the global join. The contract is the rule's so
        that this container never imports the layout package to reach it.
        """
        yield from self.rule.local_tensors(self)

    @cached_property
    def copy_factors(self) -> Layer[Tuple[CopyFactor, ...]]:
        """The named counts each weight leaf is repeated by, outermost first,
        whose product is its `multiplicity`.

        A leaf's multiplicity is `n_layers x 256 experts` and the storage
        tooltip prints it that way, so somebody has to say which count is
        which. The consumer used to decide by substring-matching a dim
        symbol's name (`"experts" if "expert" in raw`), which is right about
        `n_expert_copies` by accident and calls every layer loop "copies".
        Here each factor cites the declaration that named it: the symbol, its
        `dims.yaml` `kind`, and its `unit` — the display noun, declared in
        the registry because `kind` cannot carry it.

        Total over weight leaves: `()` is "not repeated", so an absent entry
        means "not a weight leaf" and a leaf whose factors went missing is a
        refusal rather than a leaf reported as repeated once.
        """
        return self.program.copy_factors

    def by_role(self) -> Dict[Optional[str], Cost]:
        """Cost attributed by semantic role.

        The engine does this with a recursive classifier that re-walks the result
        tree deciding, per node, whether it belongs to the attention or FFN
        bucket. With role as its own layer it is a join and a group-by: no second
        traversal, and a node with no role simply is not in the layer rather than
        being classified as "neither" by falling through a chain of tests.

        `leaves()` is the whole of the "do not double-count" rule — a component
        and its children both carry the role — and it says so once, on the layer,
        instead of as a `continue` inside the loop this used to be.
        """
        return self.cost.leaves().group_by(self.roles, Cost.combine)

    def _contributions(self, cost: Optional[Layer[Cost]] = None
                       ) -> Iterator[Tuple[str, Cost]]:
        """Each accounting root's cost, scaled by how many the model holds, with
        the stack it belongs to.

        The multiplication is `Cost`'s own: params scale with the STORAGE count
        and flops with the ACTIVE one, and for a block root those coincide
        because every layer using a template both exists and runs. Handing the
        count out and applying it here is what lets `Traffic` reach a model
        total from the same roots with a different combine.
        """
        layer = self.cost if cost is None else cost
        for part in self.program.contributions:
            value = layer.values.get(part.node.uid)
            if value is None:
                continue
            yield part.stack, (value if part.count == 1
                               else value.scaled(storage=part.count,
                                                 active=part.count))

    def by_stack(self, *, replica: bool = False) -> Dict[str, Cost]:
        """Cost per stack — the main backbone under `""`, each aux stack under
        its own name; with `replica`, one data-parallel replica's
        (`replica_cost`).

        The old result projection carried three auxiliary totals as a side
        channel, which answered one question (main vs aux) and no others: it
        could not say how much was the vision tower and how much
        the MTP head. With stacks in the program this is a group-by, and "aux"
        is a filter over the result rather than a second accounting.
        """
        out: Dict[str, Cost] = {}
        for stack, value in self._contributions(
                self.replica_cost if replica else None):
            out[stack] = out.get(stack, Cost()) + value
        return out

    def main_total(self) -> Cost:
        """The main backbone alone: no MTP head, no encoder tower. What a
        model card counts as the model's parameters."""
        return self.by_stack().get("", Cost())

    def aux_total(self) -> Cost:
        """Everything not on the main backbone."""
        return Cost.combine([c for name, c in self.by_stack().items() if name])

    def total(self) -> Cost:
        """Model total — the same roots as `by_stack`, without the key.

        Roots ONLY: a component's cost already includes its children, so summing
        every node would double-count the whole tree. And a block's roots are
        evaluated once and multiplied by the number of layers using the
        template, because the walk deliberately does not evaluate a template 28
        times — every one of those layers has identical geometry, so the
        repetition is a multiplier on the result rather than a reason to redo
        the work.

        Both rules now live in `Program.contributions`, so this and `by_stack`
        cannot drift apart on either of them.
        """
        return Cost.combine([value for _stack, value in self._contributions()])

    def total_for(self, layer_multiplicity: Layer[int]) -> Cost:
        """Model cost under an already-resolved layer selection."""
        if layer_multiplicity.program is not self.program:
            raise ValueError("layer multiplicity belongs to another Program")
        values = []
        for contribution in self.program.contributions:
            value = self.cost.values.get(contribution.node.uid)
            if value is None:
                continue
            count = layer_multiplicity.values[contribution.node.uid]
            values.append(
                value if count == 1 else value.scaled(
                    storage=count,
                    active=count,
                )
            )
        return Cost.combine(values)


@dataclass
class Point:
    """One point a walk is read at, and how: the workload's counts there
    (`counts`, `{symbol name: number}`) and the caller's own counts it
    leaves free (`free`: the read walk stays a law in them). It keeps what its
    readings have read (`seen`, `numbers`, `identities`): an object reached
    by two readings at the point — the walk, read for the step and again
    inside the serving layers built off it — is read once."""

    counts: Mapping[str, Any]
    free: FrozenSet[str] = frozenset()
    seen: Dict[int, Tuple[Any, Any]] = field(default_factory=dict)
    numbers: Dict[Any, Any] = field(default_factory=dict)
    identities: Dict[Any, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.counts = dict(self.counts)

    def read(self, value: Any, *, structure: Tuple[type, ...] = ()) -> Any:
        """`value` — anything a walk's laws are held in: an `Analysis`, a
        layer, a record built off one — with every law read here
        (`_PointReading`). `structure` names further types that hold the
        graph and no law (a caller's trees), returned as themselves. The
        caller admits the point first (`Analysis.admit`)."""
        return _PointReading(self, structure=structure).read(value)

    def identity(self, identity: Any) -> Any:
        """An extent's identity (`Dim.expr`) read here: the workload's
        counts in it put in, the names kept — for a reader that holds the
        identity apart from the walk (a row's FLOPs formula,
        `batch_sweep._flops_identity_at`)."""
        return _PointReading(self)._identity(identity)

    def laws(self, book: Any, laws: Iterable[Any]) -> Dict[Any, Any]:
        """`laws` — a cell's, compiled in its `book` (`row_roofline.
        LawBook`) — read here, all in one compiled call: each law's exact
        value, an `int` or a `Fraction`, by the law."""
        ordered = tuple(dict.fromkeys(laws))
        names = tuple(sorted(self.counts))
        values = book.compiled(ordered, names)(
            *(point_value(self.counts[name]) for name in names))
        return dict(zip(ordered, values, strict=True))

    @staticmethod
    def laws_by_point(book: Any, needs: Iterable[Tuple[Mapping[str, Any], Any]]
                      ) -> Callable[[Mapping[str, Any]], Dict[Any, Any]]:
        """Laws, each with the point it is read at (`needs`), read once per
        point, each point's in one call (`laws`): what a point's laws come
        to, asked by the point."""
        grouped: Dict[Tuple[Tuple[str, Any], ...], List[Any]] = {}
        for at, law in needs:
            grouped.setdefault(tuple(sorted(at.items())), []).append(law)
        read = {key: Point(dict(key)).laws(book, laws)
                for key, laws in grouped.items()}
        return lambda at: read[tuple(sorted(at.items()))]

    def admit(self, cell: Any) -> None:
        """Refuse this point where a walk there would refuse it
        (`Analysis.admit`): the walk `cell` (a step's, `pipeline.
        AnalysisResult`) at its own point, with the counts this point sets
        moved."""
        cell.laws.analysis.admit({**cell.point, **self.counts})


def is_count(symbol: Any) -> bool:
    """Whether a symbol of a walk's expression is one of the workload's
    inputs — a count declared whole and non-negative, or a ratio declared
    with its floor (`dim.least_of`) — rather than a name."""
    return dim_least_of(symbol) is not None


def _reads(condition: Any, names: FrozenSet[str]) -> bool:
    """Whether a condition reads any of `names`."""
    return bool(names) and any(
        symbol.name in names
        for symbol in condition.free_symbols)


class _PointReading:
    """Every value of a symbolic walk read at one point (`Analysis.at`).

    One rule, applied to whatever a walk's records hold: an expression in
    the workload's counts is that expression with the point's numbers put
    in — an exact SymPy number, as every value the walk computes is exact
    (`dim._exact_values`): a whole count divided by another stays the
    fraction it is — and a container or a record holding one is the same
    container or record holding the number. Anything holding none is returned as itself, so a
    node, a program and a spec keep their identity, and an object reached
    twice is read once (a layer and the walk that made it may share one).

    A count the point does not bind is refused: a number read off a walk
    with a free count left in it is no number. Values memoised on a record
    (`functools.cached_property`) are dropped from the read copy rather
    than read: they are derived, and derive again from the read fields.
    """

    _ATOMS = (str, bytes, bool, int, float, type(None), type, range,
              enum.Enum, types.FunctionType, types.BuiltinFunctionType,
              types.MethodType)

    def __init__(self, point: Point, *,
                 structure: Tuple[type, ...] = ()) -> None:
        self.point = point.counts
        self.free = point.free
        self.structure = (Program, *structure)
        self._seen = point.seen
        self._numbers = point.numbers
        self._identities = point.identities

    #: The atoms' own types, asked first by identity: most of what a walk
    #: records is a number, a name or None, and an exact type lookup is the
    #: cheap way to say so.
    _ATOM_TYPES = frozenset((str, bytes, bool, int, float, Fraction,
                             type(None)))

    def read(self, value: Any) -> Any:
        kind = type(value)
        if kind in self._ATOM_TYPES:
            return value
        if kind is Dim:
            # The commonest record of all, read in place: an extent whose
            # value is a number already is itself. Its identity is read too:
            # an extent no dim names carries its value as its identity, and
            # at a point that is the number a walk there names it by.
            inner, named = value.value, value.expr
            number = (inner if type(inner) in self._ATOM_TYPES
                      else self.read(inner))
            identity = (self._identity(named) if isinstance(named, sp.Basic)
                        else named)
            if number is inner and identity is named:
                return value
            # One axis reached twice is one axis read: two shapes that held
            # the same extent hold the same one at the point.
            key = id(value)
            if key in self._seen:
                return self._seen[key][1]
            out = Dim(identity, number)
            self._seen[key] = (value, out)
            return out
        if isinstance(value, sp.Basic):
            return self._number(value)
        if isinstance(value, self._ATOMS):
            return value
        key = id(value)
        if key in self._seen:
            return self._seen[key][1]
        out = self._read(value)
        self._seen[key] = (value, out)
        return out

    def _number(self, expr: Any) -> Any:
        if expr.is_Number or (expr.is_Symbol and not is_count(expr)):
            # A number, or a name (`n_heads`) the point does not bind: the
            # workload's inputs are declared, the names are not.
            return expr
        found = self._numbers.get(expr)
        if found is None:
            found = self._numbers[expr] = self._at(expr)
        return found

    def _identity(self, expr: Any) -> Any:
        """An extent's identity at the point: the workload's counts in it
        read, the names kept — a name is what the axis is, whatever number
        the point gives it."""
        found = self._identities.get(expr)
        if found is None:
            found = self._identities[expr] = (
                expr if expr.is_Number or (expr.is_Symbol and not is_count(expr))
                else self._at(expr))
        return found

    def _at(self, expr: Any) -> Any:
        """`expr` with the point's counts put in, rebuilt from the leaves
        up. A `Min` or `Max` whose arguments came to numbers is the number
        it picks: SymPy would reach the same one, through an argument
        analysis that costs more than the rest of the reading."""
        if expr.is_Symbol:
            count = is_count(expr)
            if expr.name in self.point and count:
                return dim_exact(self.point[expr.name])
            if not count or expr.name in self.free:
                return expr
            raise ValueError(
                f"the point {self.point} binds no number for "
                f"{expr.name!r}, which the walk's values read")
        if not expr.args:
            return expr
        args = tuple(self._at(arg) for arg in expr.args)
        if all(new is old for new, old in zip(args, expr.args, strict=True)):
            return expr
        if expr.func in (sp.Max, sp.Min) and all(
                arg.is_Number for arg in args):
            return (max if expr.func is sp.Max else min)(args)
        return expr.func(*args)

    def _read(self, value: Any) -> Any:
        if isinstance(value, self.structure):
            # The graph: structure, which no point changes.
            return value
        if isinstance(value, Analysis):
            # A walk read at a point is that point's walk: its numbers rest
            # on no condition.
            read = self._record(value)
            if read is value:
                read = copy.copy(value)
            read.assumptions = ()
            return read
        atoms, read = self._ATOM_TYPES, self.read
        if isinstance(value, tuple):
            items = [item if type(item) in atoms else read(item)
                     for item in value]
            if all(new is old for new, old in zip(items, value, strict=True)):
                return value
            # A named tuple (the only tuple subclass a walk records) is
            # rebuilt as its own type.
            return (tuple(items) if type(value) is tuple
                    else type(value)._make(items))
        if isinstance(value, list):
            items = [item if type(item) in atoms else read(item)
                     for item in value]
            return (value if all(new is old for new, old
                                 in zip(items, value, strict=True)) else items)
        if isinstance(value, (set, frozenset)):
            pairs = [(item, self.read(item)) for item in value]
            if all(new is old for old, new in pairs):
                return value
            return type(value)(new for _, new in pairs)
        if isinstance(value, (dict, types.MappingProxyType)):
            # A key is a name for what it maps (an identity), read as one.
            pairs = [(k, v, k if type(k) in atoms
                      else self._identity(k) if isinstance(k, sp.Basic)
                      else read(k),
                      v if type(v) in atoms else read(v))
                     for k, v in value.items()]
            if all(nk is k and nv is v for k, v, nk, nv in pairs):
                return value
            mapping = {nk: nv for _, _, nk, nv in pairs}
            if isinstance(value, types.MappingProxyType):
                return types.MappingProxyType(mapping)
            if type(value) is not dict:
                out = type(value)()
                out.update(mapping)
                return out
            return mapping
        if isinstance(value, Layer):
            values = self.read(value.values)
            return (value if values is value.values
                    else Layer(value.program, values))
        return self._record(value)

    #: The number types a record field may declare, and the Python number a
    #: law read at a point is in that field.
    _DECLARED: ClassVar[Dict[str, type]] = {"int": int, "float": float}

    @classmethod
    def _declared(cls, record: Any, spec: Any, read: Any) -> Any:
        """A field read at the point, as the type the record declares for
        it: a byte count declared `int` is a Python `int` there, as it is in
        a walk of numbers, and a fraction in a field declared whole is
        refused rather than truncated."""
        kind = cls._DECLARED.get(spec.type) if isinstance(spec.type, str) \
            else None
        if kind is None or not isinstance(read, sp.Basic) or not read.is_Number:
            return read
        if kind is int and not read.is_Integer:
            raise ValueError(
                f"{type(record).__name__}.{spec.name} is declared whole and "
                f"reads {read} at the point")
        return kind(read)

    def _record(self, value: Any) -> Any:
        """A record — a dataclass, which is what every layer value, state
        and walk record is: the same object when none of its fields reads
        the workload, else a copy with those fields read. Anything else is
        returned as itself; a free count left inside one surfaces where a
        consumer asks it for a number."""
        if not dataclasses.is_dataclass(value) or isinstance(value, type):
            return value
        # A cycle back to this record reads it as it is; the fields that do
        # read the workload are still replaced on the copy below.
        self._seen[id(value)] = (value, value)
        changed = {}
        atoms = self._ATOM_TYPES
        for spec in dataclasses.fields(value):
            field_value = object.__getattribute__(value, spec.name)
            if type(field_value) in atoms:
                continue
            read = self.read(field_value)
            if read is not field_value:
                changed[spec.name] = self._declared(value, spec, read)
        if not changed:
            return value
        out = copy.copy(value)
        for name, read in changed.items():
            object.__setattr__(out, name, read)
        # Values memoised off the fields (`functools.cached_property`) were
        # derived from the symbolic ones; they derive again from the read.
        memoised = [name for klass in type(out).__mro__
                    for name, attribute in vars(klass).items()
                    if isinstance(attribute, cached_property)]
        if memoised:
            memo = vars(out)
            for name in memoised:
                memo.pop(name, None)
        return out
