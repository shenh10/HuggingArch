"""Transforms — and the one place they compose.

There are exactly three kinds of EFFECT, told apart by what they change:

    Binding      the dim values
    Graph        the node set
    Annotation   a layer, changing nothing already computed

A transform is a set of effects, not one of them, and conflating the two is a
mistake worth naming because I made it here first. `Phase` and `Parallelism` are
filed below as bindings; both are really Binding + Graph:

    Workload      Binding
    Phase         Binding + Graph   decode reads MLA's KV in latent space —
                                    `mla_attention_absorbed` is a different
                                    subgraph, not the same one at s_query=1
    Parallelism   Binding + Graph   collectives. NO component declares an
                                    all_reduce; every one is derived from the
                                    layout walk
    Framework     Graph             fusion
    Measured      Annotation

`Phase`'s graph half is implemented (`substitute`). `Parallelism`'s
communication is derived by the layout walk on `analyze()`, not materialised
as graph nodes. Saying they ARE bindings would make the gap invisible, which
is the actual damage a wrong taxonomy does.

That partition is the whole design. It says which transforms can be reordered
(bindings — they are frames, and a frame does not care what came before it),
which one has to hand out a correspondence (a graph rewrite, or every existing
layer silently refers to nodes that are gone), and which need neither (an
annotation is keyed by uid and joins by identity).

Quantisation is deliberately NOT in that last row, though it looks like it
belongs there. `Formats` is sideband in the sense that it does not propagate
along edges — but a weight's metadata footprint is part of its cost, and cost is
computed by the walk, so folding it in afterwards never reaches a total (a
component's cost is combined from its children DURING the walk). It is a
resolver the walk consults, like the op registry, not something applied to the
result. `Measured` is the real annotation: it sits beside the theory and
overwrites nothing, which is exactly why it can be applied last.

The engine has no such partition, and pays for it three times. Per-rank geometry
is a hook installed on the instance that has to fire between two other steps.
The phase binding is applied by writing into `engine.dims` from
`inference/phases.py` — a caller reaching into the evaluator's state, which is
possible only because the evaluator has state. And fusion produces a new graph
whose relationship to the old one exists only as diagnostic strings, so nothing
computed before the fusion can be carried across it.

Why the phase math lives here
-----------------------------
`s_query`, `s_kv` and `s_pairs` are properties of a WORKLOAD against a model's
dims: arithmetic over sequence lengths, with no hardware in it. They currently
live in `inference/phases.py`, one layer above, which is exactly why the only
way to apply them is to mutate the evaluator's dims from outside. Put them where
they belong and applying a phase is constructing a binding.

`Deployment` is the composition site
------------------------------------
    Deployment(program, workload=..., config=..., fusions=[...])

One object holding what varies between runs, and one `analyze()` that applies
the bindings, then the graph rewrites, then the annotations, in that order —
because a rewrite has to know the shapes it is fusing, and an annotation has to
key on the graph that survived. `compute_model(batch, seq, ...)` grows a keyword
argument per question anyone asks; this grows a field, and the fields compose.
"""
from __future__ import annotations
from backend.arch.runtime_profile import RuntimeProfile

from fractions import Fraction
from dataclasses import dataclass, field, fields, replace
from typing import (Any, ClassVar, Dict, FrozenSet, List,
                    Mapping, Optional, Protocol, Sequence, Tuple)

import sympy as sp

from .. import dims
from ..block_synthesis import binding_of
from ..schema import (BlockGraph, GraphNode, ModelLayer, StreamSpec,
                      resolve_block_input_ports)
from ..sharding import (
    DECODE as _DECODE, ENCODE as _ENCODE, PREFILL as _PREFILL,
    degree_in_phase,
)
from .binding import STEP_LAWS, Binding, Definition
from .cache import cached_node_uids
from .formats import FormatPolicy, Formats
from .precision import AS_SHIPPED, ResolvedPrecision
from .analysis import Point
from .fusion import FusedProgram, rebase_analysis
from .layer import Rewrite
from .op_ref import StreamLaw
from .program import HandedTensor, Program, Stack
from .refusal import EngineGap, SpecRefusal
from .storage import module_path
from .traffic import produced_width
from .evaluator import Evaluator
from .dim import (Dim, Domain, declared_workload, eval_dim_expr, expr_of,
                  maximum, minimum, names_of, nearest, ratio_symbol, value_of)
from ..loader import ResolvedSpec, get_default_loader
from ..ops import OP_REGISTRY

# This module is eval's composition root — `Deployment.analyze()` is where a
# walk meets its layout rule — so it is the ONE eval module that imports the
# layout package. Nothing else under `eval/` may (the rule's projection
# hooks carry the per-rank answers back instead).
from ..layout.attention_plan import (
    IndivisibleHeadRepartition,
    legal_cp_degrees,
    replicated_encoders,
)
from ..layout.parallel_config import ParallelConfig
from ..layout.propagation import LayoutPropagation


# -- the three kinds ------------------------------------------------------

class GraphTransform(Protocol):
    """Changes the NODE SET, and must say how the new nodes relate to the old.

    Returning the graph alone is not enough: every layer computed so far is
    keyed by uid, and after a rewrite those keys refer to nodes the graph no
    longer contains. `Rewrite` is what lets a layer cross. `analysis` is the
    walk `program` was evaluated in, for a rewrite whose condition reads it.
    """

    def rewrite(self, program: Program,
                analysis: Any) -> Tuple[Program, Rewrite]: ...


class Annotation(Protocol):
    """Adds a layer keyed by uid, deriving nothing from the graph's edges.

    A weight's dtype follows from its role and the model's quant context; a
    measured latency follows from an external corpus. Neither propagates, so
    neither needs an ordering or a rewrite — they need identity, which is what
    the uid is for.
    """

    def annotate(self, program: Program, analysis: Any) -> Any: ...


# -- binding transforms ---------------------------------------------------

@dataclass(frozen=True)
class Workload:
    """Batch and sequence, as a binding.

    `s_prefix_cached` and `s_verify` are part of the workload rather than of the
    phase: a prefix cache is a property of the request mix, and speculative
    verification width is a property of how the request is served. The phase
    then decides what those mean for `s_query` and `s_kv`.
    """

    # Each count a sweep varies names the symbol one walk answers every
    # point with (`symbols`): the batch and the verify width at least one,
    # the prompt, the output and the cached prefix possibly zero.
    batch: int = field(default=1, metadata={
        "symbol": sp.Symbol("B", integer=True, positive=True)})
    input_tokens: int = field(default=1, metadata={
        "symbol": sp.Symbol("s_input", integer=True, nonnegative=True)})
    output_tokens: int = field(default=0, metadata={
        "symbol": sp.Symbol("s_output", integer=True, nonnegative=True)})
    s_prefix_cached: int = field(default=0, metadata={
        "symbol": sp.Symbol("s_prefix", integer=True, nonnegative=True)})
    s_verify: int = field(default=1, metadata={
        "symbol": sp.Symbol("s_verify", integer=True, positive=True)})
    #: The routing skew of a batch: the hottest expert's rows over the mean
    #: expert's (max/mean, what a routing trace measures). A ratio of at
    #: least one; the layout reads it as the busiest expert-parallel rank's
    #: rows (`ExpectedRoutedLoad.busiest_rank_load_factor`).
    ep_load_factor: Any = field(default=1, metadata={
        "symbol": ratio_symbol("rho", 1)})
    dcp_selected_load_factor: Any = field(default=1, metadata={
        "symbol": ratio_symbol("rho_dcp", 1)})
    dcp_failed_owners: Any = field(default=0, metadata={
        "symbol": sp.Symbol("dcp_failed_owners", integer=True, nonnegative=True)})
    logit_positions: Optional[int] = None
    #: The media each request carries this step, by model input name:
    #: `{"pixel_values": {"n_img": 2, "img_h": 672, "img_w": 672}}` — each
    #: of the input's axes the config does not fix
    #: (`ModelInputSpec.axes`). An input not named carries nothing. Only
    #: the encode phase reads it: an encoder runs over its inputs here, and
    #: the language model's step is `input_tokens` whatever the media.
    media: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Every input a sweep varies, held to its own declaration — at
        construction, because that is the one place both bindings pass.

        The declaration is the symbol the field names above, and the door
        asks IT (`dim.declared_workload`, `dim.Domain.refusal`) rather than
        a second table of bounds: a width declared `positive` takes no
        zero, a count declared `nonnegative` takes no negative, a count
        declared `integer` takes no half request, and a ratio takes nothing
        below its floor. Symbolised, the same declaration is what a law is
        written for — `Max(1, s_verify)` folds to `s_verify` because the
        symbol says the width is positive — so holding the number to it is
        what makes the two bindings one walk. It was not held: a verify
        width of 0 was accepted here and rewritten to 1 four screens later
        by a `maximum` the symbolic domain had already folded away, so the
        two halves answered a workload nobody asked for differently.

        The inputs are named here beside the fields they are, rather than
        walked by name: a list of field names would be a second spelling of
        the dataclass, and the two would drift. `media` is an encoder's
        axes rather than a count a law is written in (`ModelInputSpec`),
        and `logit_positions` names no symbol — no declaration, nothing to
        hold it to."""
        symbols = self.symbols()
        for what, value in (("batch", self.batch),
                            ("input_tokens", self.input_tokens),
                            ("output_tokens", self.output_tokens),
                            ("s_prefix_cached", self.s_prefix_cached),
                            ("s_verify", self.s_verify),
                            ("ep_load_factor", self.ep_load_factor),
                            ("dcp_selected_load_factor", self.dcp_selected_load_factor),
                            ("dcp_failed_owners", self.dcp_failed_owners),
                            ("logit_positions", self.logit_positions)):
            declared_workload(value, f"workload {what}", symbols.get(what))

    @classmethod
    def symbols(cls) -> Dict[str, sp.Symbol]:
        """The workload's counts as the symbols one walk answers every point
        with, by the field each stands for. Every law a walk writes reads
        these, and a point binds them by name."""
        return {spec.name: spec.metadata["symbol"] for spec in fields(cls)
                if "symbol" in spec.metadata}

    @staticmethod
    def walked_as_number(what: str, value: Any) -> bool:
        """Whether count `what` at `value` is walked as that number rather
        than a symbol: only a verify width of exactly the number 1, the step
        that does not speculate (`symbolic`). The one statement of "does this
        step speculate": a cell keys on its negation (`step_eval.StepCells`)
        and a reader refuses to move a count it answers true for
        (`batch_sweep.rows_at`). A SymPy number or a bool is not that number
        (`symbolic` refuses a SymPy number for a whole count)."""
        return (what == "s_verify"
                and not isinstance(value, (bool, sp.Basic))
                and value == 1)

    def stated_point(self) -> Dict[str, Any]:
        """Every count of this workload by its symbol's name: the point
        `symbolic` reads its walk at, plus the counts the walk takes as
        numbers (`walked_as_number`). A reader that evaluates a law at the
        step's numbers by name (a per-op row's roofline) reads this."""
        _workload, point = self.symbolic()
        symbols = self.symbols()
        return {**point, **{symbols[what].name: value
                            for what, value in (("s_verify", self.s_verify),)
                            if self.walked_as_number(what, value)}}

    def symbolic(self) -> Tuple["Workload", Dict[str, int]]:
        """This workload as one walk answers it at every point, and the
        point that reads that walk back as this workload.

        Each input a sweep varies becomes its own symbol — `Phase.resolve`
        reads them through different laws, and two inputs behind one symbol
        would hide a law that reads the wrong one — declared the whole count
        it is: the batch and the verify width are at least one (a step serves
        a request and verifies a token), the prompt, the output and the
        cached prefix may be zero. `logit_positions` is where the sampler
        reads, a fact of the phase and the verify width that whoever sets it
        derives (`step_eval.sampled_logit_positions`); `media` are an
        encoder's, numbers by construction (`ModelInputSpec.media_values`).
        Both are kept as given.

        A verify width of exactly 1 — a step that does not speculate — is
        kept the number 1 too, not the symbol: whether a step speculates is
        a fact of the walk a framework's dispatch reads (vLLM runs Kimi-K3's
        whole-mixer decode kernel only without speculation), so a
        non-speculative step and a speculative one are two walks
        (`step_eval.StepCells` keys its cells on it). Every other count, and
        a verify width above 1, is a symbol."""
        symbols = self.symbols()
        domain = Domain(symbols.values())
        counts = tuple((what, value, symbols[what]) for what, value in (
            ("batch", self.batch), ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("s_prefix_cached", self.s_prefix_cached),
            ("s_verify", self.s_verify),
            ("ep_load_factor", self.ep_load_factor),
            ("dcp_selected_load_factor", self.dcp_selected_load_factor),
            ("dcp_failed_owners", self.dcp_failed_owners)))
        point = {}
        stated = {}
        for what, value, symbol in counts:
            if self.walked_as_number(what, value):
                # No speculation: a fact of the walk, which it walks as the
                # number; no point binds it (`stated_point` still names it).
                stated[what] = value
                continue
            if isinstance(value, sp.Basic) and value.free_symbols:
                # A count the caller states as a law of its own stays that
                # law: the walk is over it, and no point binds it.
                stated[what] = value
                continue
            if not symbol.is_integer and isinstance(
                    value, (int, float, Fraction, sp.Rational)) \
                    and not isinstance(value, bool):
                # A ratio is read back at the exact number it was stated as.
                exact_value = Fraction(str(value))
                value = (int(exact_value) if exact_value.denominator == 1
                         else sp.Rational(exact_value.numerator,
                                          exact_value.denominator))
            # A whole count is a Python int: a SymPy number is a number
            # stated as an expression, and the walk is over expressions.
            exact = int if symbol.is_integer else (int, sp.Rational)
            if not isinstance(value, exact) or isinstance(value, bool):
                raise ValueError(
                    f"workload {what} is {value!r}; a symbolic walk reads a "
                    f"workload of numbers back at its point")
            # The symbol's declaration is the count's domain (`dim.Domain`):
            # every law of the walk holds on it, and only on it.
            refusal = domain.refusal(symbol, value)
            if refusal is not None:
                raise ValueError(f"workload {what} is {value}; {refusal}")
            point[symbol.name] = value
        return (replace(self, **{what: stated.get(what, symbol)
                                 for what, _, symbol in counts}),
                point)

    def bind(self, binding: Binding) -> Binding:
        """Only what the phase cannot decide. The activation length is the
        phase's business — see `Phase.resolve`.

        Nothing is held here: a workload exists only if it passed the door
        (`__post_init__`), and a second check on the way to the binding
        would be a second place for the answer to drift."""
        out = binding.child(B=self.batch)
        if self.logit_positions is not None:
            out = out.child(s_sampled=self.logit_positions)
        return out


#: Re-exported, not redefined: `sharding.degree_in_phase` branches on the phase
#: (the sequence world cuts the residual stream in prefill and nothing in
#: decode), so the word is owned one layer down and every importer here keeps
#: working.
PREFILL = _PREFILL
DECODE = _DECODE
ENCODE = _ENCODE


@dataclass(frozen=True)
class Phase:
    """`(s_query, s_kv, s_past, s_pairs)` for one phase of one workload.

    Four workload shapes, and `s_pairs` counts the EXACT live pairs of the
    attention matrix in each — not half of it. The lower triangle of an
    `s_query x s_query` block holds `s_query(s_query+1)/2` pairs, the
    diagonal included, and every query also sees the `s_past` keys before
    the step.

      prefill, no prefix     s_query = input_tokens,                 s_kv = input_tokens
      prefill, prefix-cached s_query = input_tokens - prefix_tokens,
                             s_kv = input_tokens; every uncached query attends
                             all `prefix_tokens`
                             unconditionally, plus the triangle of the
                             `s_query x s_query`
                             tail — both sum into the pairs
      decode, AR             s_query = 1,        s_kv = s_cache, s_pairs = s_kv
      decode, speculative    s_query = n_verify,
                             s_kv = s_cache + n_verify - 1; the first query
                             sees the `s_cache` keys an AR step sees and each
                             later one a key more, so the intra-window block
                             has a causal saving and cross-cache attention
                             does not

    In every shape the step scores `s_pairs = s_query · s_past +
    s_query(s_query+1)/2` pairs (`binding.STEP_LAWS`), and an attention's
    cost is written in them. A node that scores a matrix of its own rebinds
    `s_kv` and `s_pairs` in its frame.

    `s_past = s_kv − s_query` is bound beside them: the keys before the step's
    first query, so the query at step offset `i` has a causal context of
    `s_past + i + 1` keys. It is its own name because `s_kv` is not stable
    inside the walk — a window, a top-k or a stream union narrows it per node —
    and an op that sums a per-query law over the step's queries (top-k
    attention: `min(context, k)` each) needs where those queries sit.

    The activation length is `s_query`. Op templates still write `s_ctx` on
    the residual sequence — an op has no prefill/decode — and Binding
    reads that name as `s_query`. Decode's identity on that axis is
    `n_verify`. A decode step left at the prompt length costs 4096x too
    much everywhere except attention.
    """

    kind: str = PREFILL
    # Serving-level number, so it is supplied rather than derived: the average
    # cache a decode step reads is a function of the request mix.
    cache_len: Optional[int] = None
    #: The encoder stack an `encode` phase runs; nothing else names one.
    stack: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.kind == ENCODE) != (self.stack is not None):
            raise ValueError(
                "an encode phase runs one encoder stack (`stack=`), and only "
                "an encode phase names one")

    @staticmethod
    def step_symbols() -> Tuple[str, ...]:
        """The symbols a language-model phase binds (`resolve`)."""
        return tuple(Phase().resolve(Workload()))

    def substitute(self, program: Program) -> Program:
        """The graph half: rebind each component to the form this phase runs.

        At decode an MLA attention reads its KV in latent space — a different
        subgraph, shipped as `mla_attention_absorbed`, not the same graph at
        `s_query=1`. `BlockGraph.decode_variant` says which component is whose decode
        form; `phase_form` says which phases run it (a sparse attention's
        prefill too); `rebind_decode_variants` below does the reading.

        Applies BEFORE the walk, unlike fusion, and the difference is not
        arbitrary: an absorbed attention has a cost of its own that only
        evaluating it produces, whereas a fused op's cost is its parts'. That is
        the rule for which side of the walk a graph transform belongs on.

        What is stored does not change with the form: a form that runs a
        stored tensor differently stores at its address (`module:`), so the
        checkpoint a phase's walk lists is the file's in every phase.
        """
        if self.kind == ENCODE:
            return program.encoding(self.stack)
        rebound = rebind_decode_variants(program.resolved, self.kind)
        site = program.resolved.site
        if site is not None and self.kind == DECODE:
            # An isolated component is handed what the model's decode walk
            # hands it at its call (`CallSite.formals`).
            held = rebound if rebound is not None else program.resolved
            rebound = ResolvedSpec(held.spec, dict(held.dim_values),
                                   held.config, dict(held.view_values),
                                   site=replace(site, phase=DECODE),
                                   declared_by=held.declared_by,
                                   checkpoint=held.checkpoint)
        return program if rebound is None else Program.of(rebound)

    def resolve(self, workload: Workload) -> Dict[str, Any]:
        """The step's sequence symbols. A prompt is `input_tokens`; a
        decode step's default cache holds that prompt and half the
        output."""
        # Dual-domain throughout (`dim.maximum` / `minimum` / `nearest`): the
        # batch and the lengths may be integer symbols, and then every extent
        # below is the expression a numeric binding would evaluate.
        #
        # Each `maximum` below is a law about a quantity that can genuinely
        # be zero — a prompt may be empty, a prompt may be wholly cached, a
        # decode step's average cache may round to nothing — and none of
        # them compensates for a declaration. The two that did were
        # removed: the cached prefix cannot be negative and the verify
        # width cannot be zero, both because the workload's door says so
        # (`Workload.__post_init__`), and a `max` for a case the door
        # refuses is dead in the numeric domain and already folded away in
        # the symbolic one.
        prompt = workload.input_tokens
        if self.kind == PREFILL:
            s_in = maximum(1, prompt)
            s_prefix = minimum(workload.s_prefix_cached, s_in)
            sq = maximum(1, s_in - s_prefix)
            lq = s_in
            k = 1
        elif self.kind == DECODE:
            k = workload.s_verify
            cache = self.cache_len
            if cache is None:
                # `/ 2`, not `/ 2.0`: half of an int is the same float it
                # always was, and half of a symbol is an exact `s/2` rather
                # than a `0.5*s` whose floor no longer has a denominator to
                # read the law one residue at a time (`dim.signs`).
                cache = (prompt + workload.output_tokens / 2)
            sq = k
            lq = maximum(1, nearest(cache)) + k - 1
        else:
            raise ValueError(
                f"phase {self.kind!r} binds no language-model step")
        # Prefill: n_verify is unused (the prompt is the query axis). Decode:
        # s_query's identity is n_verify, so the walk's activation axis
        # carries that name into every formula.
        n_verify = k
        query = (Dim.symbol("n_verify", sq) if self.kind == DECODE
                 else Dim.symbol("s_query", sq))
        # The step's derived extents are the laws the phase declares
        # (`binding.STEP_LAWS`): the keys before its first query, and its
        # live (query, key) pairs — the query at offset i sees
        # `s_past + i + 1` keys.
        step = {"s_query": sq, "s_kv": lq}
        for name, law in STEP_LAWS.items():
            step[name] = Definition.of(law, step).value
        return {"s_query": query, "s_kv": lq, "s_past": step["s_past"],
                "s_pairs": step["s_pairs"], "n_verify": n_verify}

    def bind(self, binding: Binding, workload: Workload) -> Binding:
        """`s_kv` is declared authoritative here, and here is where it belongs.

        The KV cache is not in the dataflow: in decode the K/V leaves emit only
        the new token while attention reads `s_kv` positions, so copying the
        producer's extent under-counts that op's traffic by roughly `s_kv`. What
        makes the axis authoritative is a property of the WORKLOAD — there is a
        cache — which is what a phase is. The evaluator carried it as a
        hardcoded symbol name instead.
        """
        declared_workload(self.cache_len, "phase cache_len")
        return binding.child_map(
            self.resolve(workload)).with_authoritative("s_kv")



def tp_divides() -> FrozenSet[str]:
    """The symbols tensor parallelism divides — the axes whose world is on
    the `tp` mesh AND has a deployment field to set its degree. A world
    without one (DSA's indexer: both frameworks build it whole on every rank)
    runs at 1 for every deployment that exists, so its axes are whole by
    arithmetic and this set says so without the registry having to pretend
    they are unsplittable.

    A projection of `arch/dims.yaml` and of the `dim_symbols:` of the specs
    loaded now, read when asked: a copy taken at import would not see a
    symbol a spec declared after it, and that symbol's tensors would never
    shard."""
    return dims.divided_by("tp")


@dataclass(frozen=True)
class Parallelism:
    """Parallel degrees over the atomic dims they divide.

    Which symbol each degree divides is the whole content, and it is a property
    of the parallelism rather than of any op: tensor parallelism splits attention
    heads and FFN intermediates, expert parallelism splits the expert AXIS
    (`n_expert_copies`) and never the dispatch denominator
    (`n_route_targets`, also the gate's width), context
    parallelism splits the sequence.

    Data parallelism appears in no policy on purpose. It adds replicas, so it
    changes throughput and leaves every per-rank shape alone. A degree that
    does not divide its dim is not rewritten into DP here: the layout
    either declares `pad_to` or refuses at construction.

    The communication is DERIVED, and it is not filed here. The layout
    walk emits a `Redistribute` wherever what arrives is not what an op
    can consume. No component declares an all_reduce and none needs to.

    What is NOT done is materialising those collectives as graph NODES, and two
    things were learned trying:

      * `Rewrite` cannot express a pure insertion. It maps each new node to the
        origins it was built FROM, so a node inserted beside a producer either
        takes that producer's cost (double-counting it) or takes nothing. An
        insertion is `Rewrite()` plus an explicit `Layer.with_value` per layer,
        which works but is a different operation than the one the protocol
        names.
      * A collective may not want to be a node. It has no parameters, no FLOPs
        and no output of its own — it moves a tensor that already exists — so
        what an ANALYSIS needs from it is a size and a world, which is what it
        now has. Layout materializes collectives as presentation nodes
        when a renderer needs a row in an op table; the analysis itself
        only needs a size and a world.
    """

    tp: int = 1
    ep: int = 1
    cp: int = 1
    #: Decode context parallelism — a SUBGROUP of the `tp` ranks, so it splits
    #: what `dims.in_world("kv_shard")` names while adding no ranks of its own
    #: to `ParallelConfig.cluster_size` (which is where rank counting lives:
    #: this layer holds shapes, not a mesh). See `dims.World.subgroup_of`.
    dcp: int = 1
    dp: int = 1
    #: DP-attention — the request cut, and the one data parallelism that is a
    #: SHARD rather than a replica. `dp` above adds whole model copies and so
    #: owns no axis (see the class docstring); this one owns `B`, which is why
    #: it has a world (`attn_data`, which a placement calls by its degree field
    #: — see `dims.World.label`) and a layout of its own, and `dp` has neither.
    #: Owns without dividing: `B` reaches the walk already at one rank's
    #: extent, so the shard is a label here and a number nowhere — see
    #: `dims.World.bound_per_rank` and `policy`. Kept apart from `dp` for the
    #: same reason `cp` and `dcp` are: they are two deployments, and
    #: `ParallelConfig` carries them as two fields.
    dp_attention: int = 1
    #: `{world label: degree}` — the deployment's answer for every world in
    #: the registry, as `ParallelConfig` resolved it.
    #:
    #: The four scalars above are the HAND-WRITTEN surface: a test says
    #: `Parallelism(tp=8)` and means "every TP world at 8". A real deployment
    #: runs its three TP worlds at three widths and cannot say so in one
    #: integer, so it fills this instead and `degree_of_world` reads it first.
    #: Until it existed, `Placement.communication` priced every TP collective
    #: of every production config at group size 1 — `tp` is the only place it
    #: could look, and a converted config leaves `tp` at its default.
    #:
    #: A per-SYMBOL table used to sit beside this one, filled from the same
    #: config by the same resolver. It said nothing this cannot: a world's
    #: degree is one number, and `ParallelConfig` has no way to express two
    #: axes of one world at different widths — `dims.yaml` gives a world one
    #: degree field. What it did have was a place in the resolution order, and
    #: both bugs at this level were arguments about where: whether the uniform
    #: `tp` outranked it (it must not) and whether an unset `cp` did (it must
    #: not either). Deriving the axis from the world instead deletes the
    #: question.
    world_degrees: Mapping[str, int] = field(default_factory=dict)

    #: `{world label: the scalar field that speaks for it}`. The four worlds
    #: a hand-written `Parallelism` can name one at a time; `tp` is absent
    #: because it speaks for all three TP worlds at once and so is not a
    #: label. Written out rather than derived: which shorthand a test types is
    #: a fact about THIS class, and the registry has no opinion on it.
    _SHORTHAND: ClassVar[Dict[str, str]] = {
        "ep_moe": "ep", "cp": "cp", "dcp": "dcp",
        "dp_attention": "dp_attention"}

    @classmethod
    def role_of(cls, symbol: str) -> str:
        """Which MESH owns a dim symbol, whatever this deployment's degrees
        happen to be.

        `worlds` answers the same question for one deployment and drops every
        symbol whose degree is 1; this is the standing fact, and placement needs
        it to read an op's mode catalog. A catalog names its axes abstractly —
        `shard_s_ctx` says `s_ctx`, `shard_n_heads` says `n_heads`, `shard_dim` says
        `dim` — so deciding whether a concrete `d_ffn` shard can enter a mode
        means comparing what KIND of axis each is, and the parallelism that
        divides an axis is exactly that classification. Nothing else declares
        it: the dims glossary is prose for tooltips.

        Read off the symbol's world, which is where the fact lives. An axis
        the registry does not know — a catalog's abstract `dim`, a symbol a
        caller invented — answers `tp`, because that is the mesh a shape
        divides under and the modes being matched are shape modes.
        """
        world = dims.world_of(symbol)
        return world.mesh if world is not None else "tp"

    @classmethod
    def world_of(cls, symbol: str) -> Optional[str]:
        """Which WORLD divides a dim symbol, under the name a placement gives
        it — `dims.World.label`. `None` for a symbol no world splits.

        The companion of `role_of`, and the two answer different questions on
        purpose. A mode CATALOG is matched by mesh, because it names its axes
        abstractly and only the kind of split has to agree; a PLACEMENT is
        labelled by world, because the label is what a reader turns back into
        a degree and two worlds share the `cp` mesh at different degrees.
        Answering the second question with the first is what priced a `dcp`
        combine at `cp`'s degree — 1, in the common tp x dcp deployment.

        Unlike `role_of` there is no fallback: a symbol the registry does not
        know is not in a world, and inventing one puts a placement on a group
        no deployment can size. `worlds` skips those.
        """
        world = dims.world_of(symbol)
        return world.label if world is not None else None

    def degree_of_world(self, world: str, *, phase: str) -> int:
        """What degree the named world runs at here, in this phase.

        The one place a world's degree is read on the eval side — the mirror of
        `ParallelConfig.degree_of_world`, which is the one place it is read on
        the deployment side. Everything downstream (which symbols `worlds`
        lists, what `policy` divides an axis by, what
        `Placement.communication` prices a collective at) folds through this,
        so the three cannot come to disagree about how wide a group is.

        Three ways to have said it, read most-specific first, because each one
        below is a coarser statement standing in for the one above.

        `world_degrees` is the RESOLVED table and names this world exactly.
        `ep`/`cp`/`dcp`/`dp_attention` each name one world too — they ARE its
        label — so they come next. `tp` is last because it is the coarsest:
        one integer standing in for all three TP worlds, and it must not
        shadow either statement above it.

        Only the resolved table speaks at 1. A SHORTHAND left at its default is
        not a deployment saying "this world runs at 1" — it is a deployment
        that did not use that knob, and reading it as a statement is what let
        an unset `cp` shadow a deployment that had already split `s_ctx` by 2:
        `policy` divided the axis and `degree_of_world` answered 1, so any
        collective on that world priced at zero bytes. `world_degrees` is
        unguarded because a table that names a world names it deliberately.

        `tp` also reaches only worlds that have a degree FIELD. A field-less
        world is one no knob names — the DSA indexer, which vLLM and sglang
        both build whole on every rank — and letting the uniform shorthand
        carry it would shard a tensor no implementation shards.
        `world_degrees` is how such a world is given a degree.

        A world nothing names runs at 1, which is what makes "replicated"
        arithmetic rather than a declaration.
        """
        entry = dims.world_by_label().get(world)
        shorthand = int(getattr(self, self._SHORTHAND[world], 1) or 1) \
            if world in self._SHORTHAND else 1
        if world in self.world_degrees:
            degree = self.world_degrees[world]
        elif shorthand > 1:
            degree = shorthand
        elif entry is not None and entry.mesh == "tp" and entry.degree:
            degree = self.tp
        else:
            degree = 1
        # Through `degree_in_phase` rather than reading `phases:` here, because
        # `LayoutPropagation.of` cuts the residual stream through that same
        # call: a world live for the shapes but not for the residual stream would
        # divide an axis nothing divided and price collectives for a group
        # doing identical work.
        return degree_in_phase(world, degree, phase)

    def degree_of_symbol(self, symbol: str, *, phase: str) -> int:
        """What THIS axis is divided by, per rank, in this phase.

        Its world's degree, and nothing else. An axis is divided by the group
        that owns it, so per-axis is not a finer STATEMENT than per-world — it
        is the same statement projected through the registry, and a projection
        kept as a second field is a place for the two to disagree.

        Written once because `worlds` and `policy` both need it and must not
        disagree — `worlds` says a symbol is divided, `policy` says by how
        much, and a symbol listed by one at a degree the other reads as 1 is a
        shard the placement algebra propagates and the shapes never made.
        """
        return self.degree_of_world(self.world_of(symbol) or "", phase=phase)

    def worlds(self, program: Optional[Program] = None) -> Dict[str, str]:
        """Symbol → the world that divides it, labelled the way a placement
        labels one.

        The same table `policy` walks, keeping the world instead of throwing it
        away for the degree. Per-rank SHAPE only needs the degree, so `policy`
        drops the world; placement needs it, because a collective runs inside one
        subgroup and a tensor sharded on `tp_attention` and on `cp` at once needs
        one per world. Both read this so the two cannot disagree about which
        parallelism owns a symbol.

        One loop over the registry's own symbol→world map, and no branch — and
        no mesh list — that names a world. The four this replaced each
        hard-coded a mesh name, so `dcp`'s axes came out labelled `cp` and
        `ep_moe`'s came out `ep`, and `Placement.closure` — which checks a
        placement's world against the values here — rejected every `dcp` shard
        as belonging to a group the deployment does not have. 46 of them on
        Qwen3-30B-A3B at dcp=4.

        Listed in EITHER phase, unlike `policy`, which answers for the one it
        was asked about. Ownership does not move between prefill and decode —
        only the degree does — and a caller reading this table is asking which
        parallelism owns a symbol, not whether this step divides it. Dropping
        `cp`'s axes in decode would leave the decode walk unable to name the
        world a prefill-sharded tensor arrives on.
        """
        return {
            symbol: world
            for symbol, world in dims.world_labels().items()
            if any(self.degree_of_symbol(symbol, phase=p) > 1
                   for p in ("prefill", "decode"))
        }

    def as_parallel_config(self):
        """This shorthand, as the `ParallelConfig` the layout rule uses.

        Hand-written `Parallelism(tp=8)` means every TP world at 8.
        A config that already filled `world_degrees` keeps those widths.
        Sequence-parallel and the CP scheme are not on this surface; they
        stay at `ParallelConfig` defaults unless the caller passed a
        config through `Deployment`.
        """
        resolved = dict(self.world_degrees or {})
        tp = max(1, int(self.tp or 1))

        def _degree(label: str, fallback: int) -> int:
            if label in resolved:
                return max(1, int(resolved[label] or 1))
            return fallback

        return ParallelConfig(
            tp_attention=_degree("tp_attention", tp),
            tp_dense_ffn=_degree("tp_dense_ffn", tp),
            tp_moe_expert=_degree("tp_moe_expert", tp),
            ep_moe=_degree("ep_moe", max(1, int(self.ep or 1))),
            cp=_degree("cp", max(1, int(self.cp or 1))),
            dcp=_degree("dcp", max(1, int(self.dcp or 1))),
            dp_attention=_degree(
                "dp_attention", max(1, int(self.dp_attention or 1))),
            dp_replicas=max(1, int(self.dp or 1)),
            world_degrees=resolved,
        )

    def policy(self, program: Optional[Program] = None,
               *, phase: str) -> Mapping[str, int]:
        """What degree each symbol is cut by, per rank, in this phase.

        A table for the layout rule, not a divisor applied to the binding.
        Extents stay global; a non-divisible cut is refused when the
        `Block` is constructed, unless it declared `pad_to`.

        Every degree asked the same way, of the same `degree_of_world` the
        placement side asks — so the shapes and the collectives cannot come to
        disagree about how wide a group is or which worlds are live in this
        phase. Today `cp` is the only world the phase changes, and it changes
        to 1: PCP splits a prefill chunk and replicates the decode tokens over
        the same group. That restriction is declared on the world in
        `dims.yaml`, not written here.

        The `kv_shard` axes needed a branch of their own while worlds were
        labelled by mesh, because `kv_shard` and `sequence` are both `cp` and
        run at different degrees — asking the mesh got one of them the other's
        number. Labelled by world they answer for themselves: `s_kv` is in
        `dcp`, and its degree REPLACES rather than multiplies `cp`'s because
        the cache is replicated across the PCP group (`attention/pcp.py`
        all_gathers the prefill K/V before the cache write) and cut by `dcp`
        alone — which is now a statement of what one world's degree is, not a
        special case in the walk.

        A world whose axes arrive already at one rank's extent is listed by
        `worlds` and absent here: it owns the axis — placement still labels
        the shard, and the MoE seam still reads a token cut off it — but there
        is nothing left to divide. `dims.World.bound_per_rank` carries which,
        and today it is `attn_data` alone.
        """
        by_rank = {label for label, world in dims.world_by_label().items()
                   if world.bound_per_rank}
        return {
            symbol: self.degree_of_symbol(symbol, phase=phase)
            for symbol, world in self.worlds(program).items()
            if world not in by_rank
        }


def _decode_form(resolved, name: str) -> Any:
    """A component the resolved spec pruned away: its decode form.

    `components` carries only the transitive closure of what the model graph
    references, and nothing references the absorbed twin until a decode phase
    asks for it. Asked of the loader that resolved the spec, which knows the
    drafts the spec opted into (a draft's decode form is a draft); the
    singleton when the spec was assembled by hand. Never a fresh
    `SpecLoader()`: it re-parses every YAML in the library (~2.5s) to read one
    cached component, and the swap happens per sequence in a sweep.
    """
    loader = (resolved.declared_by if resolved.declared_by is not None
              else get_default_loader())
    return loader.decode_form(resolved.spec.name, name)


def splice_sampled_head(resolved):
    """Insert a zero-FLOP `sample` view immediately before every output head.

    Sampling last-N is a view the head's `in:` should see. The linear then
    propagates; Binding `s_ctx` stays the activation length. Returns `None`
    when the model has no output head. One clone per resolution, held by
    it: every walk that splices it and every tree built off the splice
    (`Cell.pre_tree`) read the same spec, and with it one spec tree.
    """
    if resolved._sampled_head_cache is None:
        resolved._sampled_head_cache = (_spliced_sampled_head(resolved),)
    return resolved._sampled_head_cache[0]


def _spliced_sampled_head(resolved):

    def splice(layers):
        out = []
        changed = False
        for layer in layers:
            if layer.layers:
                inner, inner_changed = splice(layer.layers)
                out.append(layer.model_copy(update={"layers": inner})
                           if inner_changed else layer)
                changed = changed or inner_changed
                continue
            if layer.boundary_role == "output_head":
                # Splice the view onto the edge the head actually reads. The
                # sample takes over the head's `inputs` (its explicit `in:`, or
                # the sequential predecessor when it has none), and the head now
                # reads the sample. Just inserting a preceding sibling was a
                # no-op for a head with an explicit `in:` — kimi_k3's lm_head
                # reads `language_model.model.norm` directly, so it kept running
                # the whole prefill instead of the single sampled position.
                sampled_name = f"{layer.name}__sampled"
                out.append(ModelLayer(
                    name=sampled_name,
                    op="sample",
                    inputs=layer.inputs,
                ))
                out.append(layer.model_copy(update={"inputs": [sampled_name]}))
                changed = True
            else:
                out.append(layer)
        return out, changed

    def splice_graph(graph):
        """The same view inside a component: an auxiliary stack's own head
        (an MTP layer's `shared_head.head`) is as much a logits projection
        as the model's, and serving runs it at the sampled positions too."""
        heads = [name for name, node in graph.graph.items()
                 if node.boundary_role == "output_head"]
        if not heads:
            return None
        nodes: Dict[str, GraphNode] = {}
        for name, node in graph.graph.items():
            if name in heads:
                sampled_name = f"{name}__sampled"
                nodes[sampled_name] = GraphNode(op="sample",
                                                inputs=list(node.inputs))
                nodes[name] = node.model_copy(
                    update={"inputs": [sampled_name]})
            else:
                nodes[name] = node
        out = graph.model_copy(update={"graph": nodes})
        resolve_block_input_ports(out, resolved.spec.custom_ops,
                                  resolved.spec.components)
        return out

    def splice_graphs(graphs):
        replaced = {name: spliced for name, graph in (graphs or {}).items()
                    if (spliced := splice_graph(graph)) is not None}
        return {**graphs, **replaced} if replaced else None

    spec = resolved.spec
    new_model, changed = splice(list(spec.model))
    update: Dict[str, Any] = {"model": new_model}
    blocks = splice_graphs(spec.blocks)
    if blocks is not None:
        update["blocks"] = blocks
    components = splice_graphs(spec.components)
    if components is not None:
        update["components"] = components
    if not changed and len(update) == 1:
        return None
    return resolved.with_spec(spec.model_copy(update=update))


def phase_form(graph: BlockGraph, phase: str) -> Optional[str]:
    """The component `phase` runs in place of `graph`: its declared decode
    form (`decode_variant`) in the phases the component says run it
    (`variant_phases`: decode, and prefill too for a sparse MLA attention),
    or None for `graph` itself.

    The declaration follows what serving runs. A dense MLA prefills
    expanded — K/V through `kv_b_proj`, attended MHA — as vLLM and SGLang
    prefill it. A DSA attention prefills absorbed, MQA over the latent,
    through the sparse MLA kernel decode uses: vLLM's sparse MLA layer hands
    every prefill token to `forward_mqa`
    (`model_executor/layers/attention/mla_attention.py:964-968, :1108`;
    `v1/attention/backends/mla/flashmla_sparse.py`), SGLang does beyond
    `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` = 2048 keys
    (`srt/layers/attention/dsa_backend.py:3592-3633`, `srt/environ.py:1672`).
    Not modelled: the short-prompt dense-MHA paths both keep — SGLang's at
    or below 2048 keys, vLLM's `_use_dense_mha_prefill` at or below top-k
    keys (`sparse_mla_attention.py:130-138`) and its masked-MHA thresholds.
    """
    form = graph.decode_variant
    if not form or phase not in graph.variant_phases:
        return None
    return form


def rebind_decode_variants(resolved, phase: str = DECODE):
    """Clone `resolved` with every component rebound to the form `phase`
    runs (`phase_form`); `None` when no component has another one.

    Which components have a decode twin is a spec fact, not a fact this
    module knows: the component YAML says `decode_variant:
    mla_attention_absorbed`, and reading it is the whole reason the field
    exists. Both the phase transform and serving pipeline need the same rebind,
    so it is written once here
    rather than as one loop each — two loops over the same declaration are
    exactly how the hardcoded map this replaced came to exist.

    The rebind lands on the cloned resolved spec, which is the immutable input
    shared by Program construction and structural serialization.

    The spec is CLONED, never mutated. `SpecLoader` caches one
    `ResolvedSpec` per model at module level, so every request shares it.
    This was once a mutate-and-restore helper: the API serves through
    `asyncio.to_thread`, so a concurrent prefill (`/inference/estimate`)
    constructed inside a decode caller's mid-swap window snapshotted the
    absorbed components into its own Program, surfacing absorbed-MLA
    ops in prefill's `flops_by_module`.
    """
    spec = resolved.spec
    # Same order as `ArchSpec.get_block`: inline blocks win a name clash.
    names: List[str] = []
    if spec.blocks:
        names.extend(spec.blocks)
    if spec.components:
        names.extend(n for n in spec.components if n not in names)
    rebound: Dict[str, Any] = {}
    for name in names:
        graph = spec.get_block(name)
        twin_name = phase_form(graph, phase)
        if not twin_name:
            continue
        try:
            twin = spec.get_block(twin_name)
        except KeyError:
            twin = _decode_form(resolved, twin_name)
        if twin is None:
            raise ValueError(
                f"{name}: declares `decode_variant: {twin_name}`, but no "
                f"component of that name exists in the spec or the library. "
                f"Skipping it ran decode on the prefill graph.")
        rebound[name] = twin
        rebound[twin_name] = twin
    if not rebound:
        return None
    new_c = dict(spec.components or {})
    new_b = dict(spec.blocks or {})
    for key, twin in rebound.items():
        if spec.blocks and key in spec.blocks:
            new_b[key] = twin
        else:
            new_c[key] = twin
    # A twin's graph may invoke library components the prefill closure never
    # pulled in — nothing references the twin until a decode phase binds it
    # (`_decode_form`). Add the twins' transitive component deps, or the
    # decode walk meets a leaf with no declaration.
    loader = resolved.declared_by or get_default_loader()
    seen = set(new_c) | set(new_b) | set(OP_REGISTRY) | set(spec.custom_ops or {})
    pending = list(rebound.values())
    while pending:
        graph = pending.pop()
        for node in graph.graph.values():
            refs = [node.op, *(p for p in (node.params or {}).values()
                               if isinstance(p, str))]
            for ref in refs:
                if not isinstance(ref, str) or ref in seen:
                    continue
                dep = loader.decode_form(spec.name, ref)
                if dep is None:
                    continue
                seen.add(ref)
                new_c[ref] = dep
                pending.append(dep)
    return resolved.with_spec(
        spec.model_copy(update={"components": new_c, "blocks": new_b}))


# -- the composition site -------------------------------------------------

@dataclass(frozen=True)
class Deployment:
    """A model, a workload, a phase, a mesh, and any graph rewrites.

    Everything that varies between runs, in one object, with `analyze()` as the
    only entry point. The order is fixed and the reason is structural: bindings
    first because a rewrite needs resolved shapes to fuse against, rewrites
    second because an annotation must key on the graph that survived, and
    annotations last.
    """

    program: Program
    workload: Workload = field(default_factory=Workload)
    phase: Phase = field(default_factory=Phase)
    #: The serving mesh. `None` is one rank. `.on(tp=8)` writes this
    #: from the `Parallelism` shorthand.
    config: Any = None
    fusions: Sequence[Any] = ()
    annotations: Sequence[Any] = ()
    #: Additional run-scoped dimension bindings. These are immutable frames,
    #: not mutations of the evaluator, and intentionally apply after phase
    #: resolution so callers can provide authoritative phase-specific dims.
    bindings: Mapping[str, Any] = field(default_factory=dict)
    #: The precision profile (`precision.PrecisionProfile`): every weight,
    #: edge and cache slot no rule of it selects runs as the model ships.
    #: `None` is `AS_SHIPPED` — any other profile is a claim about a
    #: deployment choice somebody made.
    precision: Optional[Any] = None

    runtime_profile: RuntimeProfile = RuntimeProfile.DECLARED

    @property
    def specializes_verification(self) -> bool:
        """Whether this runtime requires a finite causal query block."""
        return (self.runtime_profile == RuntimeProfile.DECLARED
                and self.phase.kind == DECODE and self.config is not None
                and self.config.dcp > 1
                and bool(self.program.sparse_dcp_components))

    def binding(self) -> Binding:
        """Workload and phase only — the layout rule keeps extents global.

        The language model's phases bind its step (`Phase.bind`) from the
        workload's tokens alone; no media reaches it. The encode
        phase binds the batch and the media axes of the encoder's own inputs
        and nothing of the language model's step: an encoder node that reads
        `s_ctx` has nothing to read.

        Layout scalars that are not cuts come from the mesh
        (`ParallelConfig.layout_binds`). An explicit `with_dims`
        still wins.
        """
        out = self.program.base_binding
        out = self.workload.bind(out)
        bindings = dict(self.bindings)
        if self.phase.kind == ENCODE:
            program = self.program.encoding(self.phase.stack)
            for entry in program.walked_inputs.values():
                # An encoder that runs another's weights walks beside it
                # (`Stack.reruns`) at no item of that one's: this step is
                # its own media's.
                own = entry.stack == self.phase.stack or (
                    entry.per_item_of is not None
                    and program.model_inputs[entry.per_item_of].stack
                    == self.phase.stack)
                out = out.child_map(
                    entry.media_values(self.media if own else {}))
            # A run-scoped binding of the language model's step is no
            # encoder's: the encode phase binds none of it.
            for name in Phase.step_symbols():
                bindings.pop(name, None)
        else:
            out = self.phase.bind(out, self.workload)
        if bindings:
            out = out.child_map(bindings)
        if self.config is not None:
            extras = {
                name: value
                for name, value in self.config.layout_binds().items()
                if name not in (self.bindings or {})
            }
            if extras:
                out = out.child_map(extras)
        return _declared(out)

    def analyze(self) -> Any:
        """Bindings, then the walk, then the graph rewrites, then annotations.

        The walk uses `LayoutPropagation`: extents stay global and layout
        rides beside them. Rank-local numbers are derived from layout,
        not by dividing the binding.

        The rewrites come AFTER the walk, which is the opposite of what a graph
        transform usually implies and follows from what a fusion plan declares:
        nothing. The registry carries no cost arithmetic on purpose — a fused
        op's cost is its parts' — so the parts must be evaluated for the fused
        node to have a value at all, and there is no op spec for
        `fused_add_rmsnorm` to evaluate directly. Each layer then crosses with
        its own combine.

        The walk is over the workload's counts as symbols
        (`Workload.symbolic`): a workload of numbers is that walk read at its
        point (`Analysis.at`), so the numbers of every point come from the
        one walk that is every point's, and the walk decides nothing about
        them.
        """
        over, point, free = self._over_counts()
        analysis = over._walk(over.binding())
        return (analysis.at(Point(point, free=free),
                                structure=(FusedProgram,))
                if point else analysis)

    def _over_counts(self) -> Tuple["Deployment", Dict[str, int],
                                    FrozenSet[str]]:
        """This deployment with its workload's counts as symbols
        (`Workload.symbolic`), the point that reads its walk back as this
        one, and the names of the counts the caller states as symbols of
        its own — in the workload, a dim binding or a phase's cache — which
        the point leaves free."""
        stated = (self.workload.batch, self.workload.input_tokens,
                  self.workload.output_tokens, self.workload.s_prefix_cached,
                  self.workload.s_verify, self.workload.ep_load_factor,
                  self.workload.dcp_selected_load_factor,
                  self.workload.dcp_failed_owners, self.phase.cache_len,
                  *self.bindings.values())
        free = frozenset(
            symbol.name for value in stated
            if isinstance(value_of(value), sp.Basic)
            for symbol in value_of(value).free_symbols)
        workload, point = self.workload.symbolic()
        if self.specializes_verification:
            # Sparse verification has ragged per-query ownership; retain its
            # finite width when compiling the context/batch laws. Serving
            # cells include this width in their key for the same reason.
            width = value_of(self.workload.s_verify)
            if not isinstance(width, (int, sp.Integer)):
                raise ValueError('sparse DCP requires a fixed verification width')
            workload = replace(workload, s_verify=int(width))
        return replace(self, workload=workload), point, free

    # -- the media a step carries -------------------------------------------

    @property
    def media(self) -> Mapping[str, Mapping[str, int]]:
        """`Workload.media`, held to the spec: every name a media input of
        this model, and an input whose items sit inside another's
        (`per_item_of`) given exactly when that one is — a request's crops
        are stated with its images, none included. An input the spec does
        not walk (`unmodeled`) is refused any item, with its reason."""
        media = dict(self.workload.media)
        inputs = self.program.model_inputs
        mixed = sorted(name for name, given in media.items()
                       if isinstance(given, (list, tuple)))
        if mixed:
            # Items of different sizes run packed in one tower call; a walk
            # binds one size per input, and the one call is not expressible
            # in one walk yet.
            raise EngineGap(
                f"{self.program.spec.name}: media {mixed} give items of "
                f"different sizes; one request states one size per input "
                f"here (items of different sizes in one call are not "
                f"modelled yet) — state each size as its own request",
                capability="media.mixed_sizes")
        unknown = sorted(name for name in media
                         if name not in inputs or inputs[name].kind == "tokens")
        if unknown:
            raise ValueError(
                f"{self.program.spec.name}: media {unknown} name no media "
                f"input of this model; its media inputs are "
                f"{sorted(n for n, e in inputs.items() if e.kind != 'tokens')}")
        for name in media:
            parent = inputs[name].per_item_of
            if parent is not None and parent not in media:
                raise ValueError(
                    f"{self.program.spec.name}: media gives {name!r}, whose "
                    f"items belong to each item of {parent!r}, and no "
                    f"{parent!r}; give both")
        for name, entry in inputs.items():
            if entry.per_item_of in media and name not in media:
                raise ValueError(
                    f"{self.program.spec.name}: media gives "
                    f"{entry.per_item_of!r} and not {name!r}, whose items "
                    f"belong to each of its items; state them too "
                    f"({ {a: '<n>' for a in entry.axes if a not in entry.fixed} }"
                    f", 0 when there are none)")
            if entry.unmodeled is not None and name in media:
                values = entry.media_values(media)
                if values[entry.axes[0]]:
                    raise EngineGap(
                        f"{self.program.spec.name}: media gives "
                        f"{values[entry.axes[0]]} item(s) of {name!r}, "
                        f"which the spec declares and the engine does not "
                        f"walk yet: {entry.unmodeled}",
                        capability=f"model_inputs.{name}")
        return media

    def encoders(self) -> Tuple["Deployment", ...]:
        """One encode deployment per encoder stack of the model, with this
        deployment's workload, mesh and weights."""
        return tuple(
            replace(self, phase=Phase(kind=ENCODE, stack=stack.name),
                    fusions=(), annotations=())
            for stack in self.program.stacks if stack.is_encoder)

    def handed_over(self, media: Optional[Mapping[str, Any]] = None
                    ) -> Dict[str, Tuple[HandedTensor, ...]]:
        """What each encoder output the language model reads holds for one
        request carrying `media` (this deployment's when None): one
        `HandedTensor` per tensor, read off that encoder's encode walk at one
        request (`_handed_by`). An encoder none of whose inputs `media` names
        hands over zero rows. Every edge onto a domain's prompt positions
        carries the rows its defining edge does."""
        media = self.media if media is None else media
        out: Dict[str, Tuple[HandedTensor, ...]] = {}
        for stack in {h.stack for h in self.program.handoffs}:
            out.update(_handed_by(self.program, stack, media,
                                  self.precision))
        rows = {h.name: sum(t.rows for t in out[h.name])
                for h in self.program.handoffs}
        for handoff in self.program.handoffs:
            if handoff.defines:
                continue
            (defining,) = [h for h in self.program.handoffs
                           if h.defines and h.domain == handoff.domain]
            for tensor in out[handoff.name]:
                if tensor.rows != rows[defining.name]:
                    raise SpecRefusal(
                        f"{self.program.spec.name}: {handoff.name!r} adds "
                        f"{tensor.rows} rows onto the prompt positions of "
                        f"{handoff.domain!r}, and {defining.name!r} defines "
                        f"{rows[defining.name]} of them",
                        declaration="prompt_slots",
                        fix="an edge with `defines: false` lands on the "
                            "positions its domain's defining edge fills; read "
                            "the encoder output that carries as many rows")
        return out

    def media_tokens(self) -> Dict[str, int]:
        """The prompt positions one request's media fill, per media input
        this deployment's media names: the rows its defining edge hands over
        (`prompt_slots`). An encoder's fact, reported beside its own
        accounting; the language model's step is `input_tokens` and reads
        none of it."""
        if not (self.program.handoffs and self.workload.media):
            return {}
        handed = self.handed_over()
        return {handoff.domain: sum(t.rows for t in handed[handoff.name])
                for handoff in self.program.handoffs
                if handoff.defines and handoff.domain in self.media}

    def checkpoint(self) -> Any:
        """The language model's walk with every encoder walked beside it,
        each in its own encode binding (`Program.CHECKPOINT`): one Analysis
        over every tensor the checkpoint holds, for a question about the
        checkpoint — its tensor names, its stored and resident weights. The
        language model's numbers in it are `analyze()`'s; the encoders run
        on this deployment's media (none by default), so their cost is the
        encode phase's at that media and is no part of the step's."""
        over, point, free = self._over_counts()
        analysis = over._walk(over.binding(), checkpoint=True)
        return (analysis.at(Point(point, free=free),
                            structure=(FusedProgram,))
                if point else analysis)

    def _walk(self, binding: Binding, *, checkpoint: bool = False) -> Any:
        # A walk is over the workload's counts as symbols: a count given as
        # a number is read at its point after the walk (`analyze`), never
        # walked as one.
        numbers = sorted(
            what for what, count in (
                ("batch", self.workload.batch),
                ("input_tokens", self.workload.input_tokens),
                ("output_tokens", self.workload.output_tokens),
                ("s_prefix_cached", self.workload.s_prefix_cached),
                ("s_verify", self.workload.s_verify),
                ("ep_load_factor", self.workload.ep_load_factor),
                ("dcp_selected_load_factor", self.workload.dcp_selected_load_factor),
                ("dcp_failed_owners", self.workload.dcp_failed_owners))
            if not (isinstance(count, sp.Basic) and count.free_symbols)
            and not Workload.walked_as_number(what, count))
        if numbers == ['s_verify'] and self.specializes_verification:
            # The sparse runtime plan is specialized to a finite verify
            # width; its query-pair laws still vary with context and batch.
            numbers = []
        if numbers:
            raise TypeError(
                f"a walk of the workload's counts {numbers} as numbers: "
                f"walk them as symbols and read the walk at the point "
                f"(`Deployment.analyze`)")
        program = self.phase.substitute(self.program)
        # The encoders' outputs, as the language model's step reads them:
        # the edges are the graph's, and they carry no rows. Media are the
        # encode phase's own accounting; no step of the language model
        # holds them, whatever the workload's media.
        handed = self.handed_over({}) if self.phase.kind != ENCODE else {}
        if self.workload.logit_positions is not None:
            spliced = splice_sampled_head(program.resolved)
            if spliced is not None:
                program = Program.of(spliced)
        if checkpoint:
            program = program.checkpoint()
        # Joins of whole streams are nodes of the graph the walk prices,
        # like the decode forms above: written before the walk.
        program = program.lowered()
        producers = set()
        for uid in cached_node_uids(program):
            node = program.by_uid(uid)
            producers.add(
                program.output_leaf(node)[0].uid
                if node is not None and node.children else uid)
        parallel = self._layout_parallel(program)
        rule = LayoutPropagation.of(
            parallel, self.phase.kind,
            cached_producers=frozenset(producers),
            encoder_nodes={uid: program.stack_of(program.by_uid(uid))
                           for uid in program.encoder_uids},
            replicated_encoders=replicated_encoders(
                program, parallel, self.phase.kind),
            expert_load=self.workload.ep_load_factor,
            dcp_selected_load_factor=self.workload.dcp_selected_load_factor,
            dcp_failed_owners=self.workload.dcp_failed_owners,
            program=program, runtime_profile=self.runtime_profile)
        encoder_bindings = (
            {enc.phase.stack: enc.binding() for enc in self.encoders()}
            if checkpoint else {})
        try:
            analysis = Evaluator(
                program,
                precision=ResolvedPrecision(self.profile, program),
                rule=rule,
                workload=self.workload,
                handed=handed,
            ).run(binding, encoder_bindings=encoder_bindings)
        except IndivisibleHeadRepartition as refusal:
            # The degrees this model does take, solved over its walk with
            # no sequence world: the set a picker lists, on the refusal.
            raise refusal.solved(legal_cp_degrees(
                Deployment(self.program).analyze(), rule.parallel)) from None
        for transform in self.fusions:
            program, step = transform.rewrite(program, analysis)
            analysis = rebase_analysis(analysis, program, step)

        for annotation in self.annotations:
            analysis = annotation.annotate(program, analysis)
        return analysis

    # -- variations ------------------------------------------------------

    def _layout_parallel(self, program: Program):
        """The mesh the layout rule uses, valid on its own and for
        `program`."""
        parallel = self.config if self.config is not None else ParallelConfig()
        errors = parallel.validate()
        if errors:
            raise ValueError(
                "layout rule requires a valid deployment: "
                + "; ".join(errors))
        return parallel

    def at(self, **workload: Any) -> "Deployment":
        return replace(self, workload=replace(self.workload, **workload))

    def with_dims(self, **bindings: Any) -> "Deployment":
        """Return a deployment with additional run-scoped dim bindings."""
        return replace(self, bindings={**self.bindings, **bindings})

    def during(self, kind: str, cache_len: Optional[int] = None,
               stack: Optional[str] = None) -> "Deployment":
        return replace(self, phase=Phase(kind=kind, cache_len=cache_len,
                                         stack=stack))

    def on(self, **degrees: Any) -> "Deployment":
        """Shorthand mesh. `on(tp=8)` is every TP world at 8."""
        return replace(self, config=Parallelism(**degrees).as_parallel_config())

    @property
    def profile(self) -> Any:
        """`precision`, as shipped when unset."""
        if self.precision is None:
            return AS_SHIPPED
        return self.precision

    def with_precision(self, profile: Any) -> "Deployment":
        """The same deployment under another precision profile."""
        return replace(self, precision=profile)

    def fused(self, *transforms: Any) -> "Deployment":
        return replace(self, fusions=tuple(self.fusions) + transforms)


def _declared(binding: Binding) -> Binding:
    """`binding`, with every dim in it held to what a walk can reason about.

    The door, and it is on the ASSEMBLED frame rather than on each route in,
    because the routes are several and a door per route is a door someone
    adds a seventh route beside: `at` writes the workload, `during` the
    phase's cache length, `with_dims` a run-scoped overlay that is applied
    AFTER the phase (so `with_dims(s_ctx=…)` reaches the walk without
    passing any workload check), the encode branch binds an encoder's media
    axes and skips the workload entirely, and the mesh adds its layout
    scalars. Every one of them ends here, so this is the only place that can
    hold all of them.

    What it holds them to is `dim.declared_workload`: a dim bound to an
    expression must be made of symbols that say they are whole counts. A
    bare `Symbol(name)` says nothing, and what it costs is not an error at
    the door but a comparison four screens into the layout algebra that
    cannot be taken — `physical_axis_partition_audit`'s `extent <= 0` on
    `Symbol('S')`, which is how this hole was found.
    """
    for name, value in binding.flat().items():
        declared_workload(value, f"dim {name}")
    return binding


def _handed_by(program: Program, stack: str,
               media: Mapping[str, Mapping[str, Any]],
               precision: Any) -> Dict[str, Tuple[HandedTensor, ...]]:
    """What encoder `stack` hands the language model for one request's
    `media`: the encode walk at one request, its outputs read where the
    language model reads them — a stage's exit, or each element of an
    appended stream — as `rows × width`."""
    own = {name: dict(media[name]) for name, entry
           in program.model_inputs.items()
           if entry.stack == stack and name in media}
    # The items of those the spec does not walk, stated with them.
    own.update({name: dict(media[name]) for name, entry
                in program.model_inputs.items()
                if entry.unmodeled is not None and entry.per_item_of in own
                and name in media})
    analysis = Deployment(
        program, workload=Workload(batch=1, media=own),
        phase=Phase(kind=ENCODE, stack=stack), precision=precision).analyze()
    walked = analysis.program
    out: Dict[str, Tuple[HandedTensor, ...]] = {}
    for handoff in walked.handoffs:
        if handoff.stack != stack:
            continue
        if handoff.stream:
            # A stream no iteration appends to hands nothing over.
            producers = [producer for producer, _state
                         in analysis.appended.get(handoff.name, ())]
        else:
            producers = [analysis.stage_outputs[handoff.name]]
        tensors = []
        for uid, port in producers:
            shape = analysis.shapes.values[uid]
            axes = shape.out if port == "out" else shape.ports[port]
            rows = 1
            for extent in axes[:-1]:
                rows *= int(value_of(extent))
            tensors.append(HandedTensor(
                rows=rows, width=int(value_of(axes[-1])),
                value_bytes=produced_width(
                    walked.by_uid(uid), port, analysis.resolved_precision,
                    walked)))
        out[handoff.name] = tuple(tensors)
    return out


# -- isolating one component ---------------------------------------------
#
# `CallSite` / `isolate_resolved` / `analyze_isolated` assemble a one-stage
# `Deployment` for a component outside its model. Projecting the result into
# a report is the presentation package's (`projection.result.project_model`
# and friends), which composes on top of this module.

@dataclass(frozen=True)
class CallSite:
    """One call of a component in a model, as the model makes it.

    Isolating a component lifts its graph out of the block that calls it,
    and the call supplies what the graph alone does not have. Where its
    tensors live in the checkpoint (`layers.X.attn.`), the role it holds
    there (`attention`) and the checkpoint's saved formats (`quant_context`
    and native dtype): a quant rule reads all three — a role picks the
    scheme, and an override is keyed on the on-disk name (DeepSeek-V4 keeps
    `attn.compressor.wkv` BF16 inside an FP8 attention). What it is handed
    (`formals`), the frame it is called from (`frame`) and the `dims:` the
    call itself binds (`binds`): the model's walk at that call, which is the
    only place a formal has a shape.

    The call itself is the last fact, and the one the cache needs: a
    component's `cache:` declaration belongs to the node that invokes it,
    which is where the walk records the recurrent state and where
    `eval.cache` finds what the component keeps. An isolated spec therefore
    runs the component the way a model block does — one node calling it,
    with the call's own `dims:` — in the block `block` names.

    A call is a component call node of the model's graph or a stage running
    a block template. Every call of a model is listed by `calls`; a site is
    one of them, chosen by `of`.
    """

    address: str
    role: Optional[str]
    stack: str
    #: The checkpoint's saved formats; None for the host spec's own.
    formats: Optional[FormatPolicy]
    #: The model whose checkpoint this is.
    model_id: Optional[str] = None
    #: Whether the component is the one the model calls here (True), or
    #: takes the place of the component called here (False).
    invoked: bool = True
    #: The component called here. `isolate_resolved` sets it to the one it
    #: isolates.
    component: Optional[str] = None
    #: The kind of encoder stack the call sits in (`vision`, `audio`), whose
    #: cache lives one forward pass (`Stack.is_encoder`); None on the
    #: backbone or a head, whose cache persists.
    encoder: Optional[str] = None
    #: The call's address in the model: the block invocation
    #: (`Program.blocks` ordinal) and the call node's path in it —
    #: `3/self_attn`; a stage running the template is the ordinal alone,
    #: a call outside every block `/path`.
    call: str = ""
    #: The block invocation (`Program.blocks` ordinal) the call runs in;
    #: None for a call outside every block.
    invocation: Optional[int] = None
    #: The call node's path in its block (`SpecIRNode.display_path`); ""
    #: for a stage, which runs the template itself.
    path: str = ""
    #: The call in words: the stack, its layers, the node's path.
    label: str = ""
    #: What the call hands the component in each phase the model's walk
    #: makes it (`prefill`, `decode`; an encoder's call `encode`), one
    #: tensor per formal, in order: each
    #: axis an expression in the model's dims and the workload's symbols
    #: (`B`, `s_ctx`), read off that walk at the call (`Analysis.shapes` of
    #: the call node, `Analysis.stage_inputs` of a stage). The isolated walk
    #: values them in its own frame.
    handed: Tuple[Tuple[str, Tuple[Tuple[Any, ...], ...]], ...] = ()
    #: Per phase, the element of each appended stream a stage call read
    #: (`read: each`; `Analysis.stage_elements`) — a tensor another stack
    #: appended, which no isolation of the call produces — by stream name,
    #: every axis an expression the isolated walk values in its own frame.
    handed_elements: Tuple[Tuple[str, Tuple[Tuple[str, Tuple[Any, ...]],
                                            ...]], ...] = ()
    #: The phase the isolated walk runs, whose tensors `formals` are;
    #: `Phase.substitute` sets decode.
    phase: str = PREFILL
    #: The frame the call is made from, as far as it differs from the
    #: model's: the dims its layer variant (`assignments[].dims`) and its
    #: enclosing calls rebind, at the values the model's walk bound there.
    frame: Tuple[Tuple[str, Any], ...] = ()
    #: The call's own `dims:` as the spec writes them, in the calling
    #: frame's symbols (`n_heads: n_kv_heads`).
    binds: Tuple[Tuple[str, str], ...] = ()
    #: The call's own `params:` as the spec writes them — the slot bindings
    #: of a component call written inline (`attn: {op: csa2_attention,
    #: params: {compressor: …}}`). Empty for a call through a `<param>`
    #: placeholder, whose bindings live in the synthesized component its
    #: `component` names instead. The isolated walk rebuilds the call and
    #: must re-apply them: without them the component's `<param>` ops stay
    #: unbound.
    params: Tuple[Tuple[str, str], ...] = ()
    #: The model's declarations of the inner streams the called component
    #: reads at this call (`_read_stream_declarations`), by stream name. A
    #: read stream is handed over by the call like a formal is, so it is
    #: the calling model's fact: a component benched inside another host
    #: (a drop-in) reads its streams as its own model declares them. Not
    #: hashed: a declaration is a schema object.
    streams: Tuple[Tuple[str, Any], ...] = field(default=(), hash=False)
    #: Per phase, the inner streams the call's layers read
    #: (`Program.block_stream_reads`), each with the cache law it carries
    #: there (`StreamLaw`; None where it carries none), by stream name: the
    #: law the layer that publishes it states (`Analysis.stream_laws`), as
    #: the model hands it to the call with the tensor.
    handed_streams: Tuple[Tuple[str, Tuple[Tuple[str, Optional[StreamLaw]],
                                           ...]], ...] = ()
    #: An encoder's call is sized by its encoder's media: the axes a caller
    #: gives (`ModelInputSpec.axes` the config does not fix) and those the
    #: config fixes, at their values. Empty for a call no encoder makes.
    media_axes: Tuple[str, ...] = ()
    media_fixed: Tuple[Tuple[str, int], ...] = ()

    def dims_in(self, resolved: Any) -> Dict[str, Any]:
        """The dims this call is made under in model `resolved`: the
        model's, with the call's frame (`frame`) over them — what its
        isolation is resolved at (`isolate_resolved`), before the call's
        own `dims:` (`binds`), which its walk binds at the call node."""
        return {**resolved.dim_values, **dict(self.frame)}

    @staticmethod
    def stage_call(invocation: int) -> str:
        """The address of the call a stage makes: the block invocation
        (`Program.blocks` ordinal) it runs, alone."""
        return str(invocation)

    @property
    def is_stage(self) -> bool:
        """Whether the call is a stage running a block template (a layer),
        not a component call node inside one."""
        return self.invocation is not None and not self.path

    @property
    def block(self) -> str:
        """The block an isolated spec runs this call in: one node, invoking
        `component`, the stage's only template. A site that does not yet
        name its component has no such block (`isolate_resolved` names it)."""
        if self.component is None:
            raise ValueError(
                "this call site names no component, so it runs in no block; "
                "isolate_resolved sets the component it isolates")
        return f"{self.component}_site"

    @property
    def formals(self) -> Tuple[Tuple[Any, ...], ...]:
        """What the call hands the component in `phase`."""
        handed = dict(self.handed)
        if self.phase not in handed and ENCODE in handed:
            raise ValueError(
                f"{self.call or 'this call site'} is a call of encoder stack "
                f"{self.stack!r}, made in its encode phase, where it is "
                f"handed {self._described(handed[ENCODE])} — sized by the "
                f"media's own axes {list(self.media_axes)}; a language-model "
                f"step makes no call of an encoder. Evaluate it in its encode "
                f"phase with those axes given")
        if self.phase not in handed:
            raise ValueError(
                f"{self.call or 'this call site'} hands {self.component!r} "
                f"nothing at {self.phase}: the model's {self.phase} walk does "
                f"not make this call")
        return handed[self.phase]

    @property
    def stream_laws(self) -> Dict[str, Optional[StreamLaw]]:
        """The inner streams the call reads in `phase`, each with the cache
        law it carries there (`handed_streams`)."""
        return dict(dict(self.handed_streams).get(self.phase, ()))

    @property
    def elements(self) -> Dict[str, Tuple[Any, ...]]:
        """The appended streams' elements the call read in `phase`
        (`handed_elements`)."""
        return dict(dict(self.handed_elements).get(self.phase, ()))

    def encoding(self, media: Optional[Mapping[str, Any]]
                 ) -> Tuple["CallSite", Dict[str, int]]:
        """This encoder call as its encode phase makes it, at the media
        sizes `media` gives (`media_axes`, each a whole number): the site
        whose formals are the encode walk's, and the dims an isolated walk
        of it binds — the given axes and the ones the config fixes. A call
        no encoder makes, or media that miss an axis or name another, is
        refused with the axes it takes."""
        if ENCODE not in dict(self.handed):
            raise ValueError(
                f"{self.call or 'this call site'} is no encoder's call; it "
                f"is evaluated at a language-model step, not with media")
        given = dict(media or {})
        missing = [axis for axis in self.media_axes if axis not in given]
        unknown = sorted(set(given) - set(self.media_axes))
        bad = sorted(axis for axis, value in given.items()
                     if isinstance(value, bool) or not isinstance(value, int)
                     or value < 0)
        if missing or unknown or bad:
            raise ValueError(
                f"{self.call or 'this call site'} is a call of encoder stack "
                f"{self.stack!r}, evaluated in its encode phase at the "
                f"media's own sizes: give {list(self.media_axes)} as whole "
                f"numbers"
                + (f" (missing {missing})" if missing else "")
                + (f" (unknown {unknown})" if unknown else "")
                + (f" (not whole numbers: {bad})" if bad else ""))
        return (replace(self, phase=ENCODE),
                {**given, **dict(self.media_fixed)})

    def stream_reads(self, resolved: Any) -> Dict[str, Any]:
        """The declarations of the model-wide streams the component reads at
        this call: all the stack the call runs in tells its isolation,
        besides an encoder's (`encoder`). The component the model calls
        here reads them as the calling model declares them (`streams`),
        whichever host it is evaluated in. A foreign component taking the
        call's place reads the host's: the call's own `params:` decide
        which components its `<param>` slots resolve to — a shared-KV
        consumer reads `compress_kv` only under the binding that reads
        it — and the foreign one has no claim on them."""
        if self.invoked:
            return dict(self.streams)
        return _read_stream_declarations(
            resolved, self.component, self.stack,
            # A foreign component taking the call's place has no claim on
            # the called component's slot bindings.
            params=dict(self.params) if self.invoked else None)

    @classmethod
    def of(cls, resolved: Any, component: str, call: Optional[str] = None,
           model_id: Optional[str] = None) -> "CallSite":
        """The call of `component` in model `resolved` that `call` names.

        `call` is one of `calls(resolved)`. When it calls another component,
        `component` takes that one's place there: what the call hands over
        is still the model's. Without `call`, the component's only call;
        a component the model calls more than once, or not at all, has no
        call to read its formals from until the caller names one."""
        calls = cls.calls(resolved)
        if call is not None:
            found = next((site for site in calls if site.call == call), None)
            if found is None:
                raise ValueError(
                    f"{resolved.spec.name}: no call {call!r}. Its calls: "
                    + "; ".join(f"{site.call} ({site.component}, "
                                f"{site.label})" for site in calls))
            if not binding_of(resolved.spec, found.component, component):
                # The component takes the place of the one called here: the
                # call's slot bindings belong to the component that holds
                # them, and the foreign one has no claim on them.
                found = replace(found, component=component, invoked=False,
                                params=())
            return replace(found, model_id=model_id)
        own = [site for site in calls
               if binding_of(resolved.spec, site.component, component)]
        if len(own) == 1:
            return replace(own[0], model_id=model_id)
        if not own:
            raise ValueError(
                f"{resolved.spec.name} does not call {component!r}, so "
                f"nothing hands it a tensor. Name the call it takes the "
                f"place of (`call`)")
        raise ValueError(
            f"{resolved.spec.name} calls {component!r} {len(own)} times; "
            f"name the call to evaluate it at (`call`): "
            + "; ".join(site.described() for site in own))

    @property
    def handed_where_made(self) -> Tuple[Tuple[Any, ...], ...]:
        """What the call hands over in the phase that makes it: a prefill
        step's for the language model's calls, the encode phase's for an
        encoder's."""
        handed = dict(self.handed)
        return handed.get(PREFILL, handed.get(ENCODE, ()))

    @property
    def where_made(self) -> "CallSite":
        """This call in the phase that makes it: an encoder's in its encode
        phase, the language model's as it is."""
        return replace(self, phase=ENCODE) if self.encoder is not None else self

    @staticmethod
    def _described(formals: Tuple[Tuple[Any, ...], ...]) -> str:
        return ", ".join("[" + ", ".join(str(axis) for axis in formal) + "]"
                         for formal in formals)

    def described(self) -> str:
        """The call, where it is and what it hands over, as a message
        lists it: `3/self_attn (backbone · layers 3–60 · self_attn)
        [B, s_ctx, d_model]`."""
        return (f"{self.call} ({self.label}) "
                f"{self._described(self.handed_where_made)}")

    @classmethod
    def calls(cls, resolved: Any) -> Tuple["CallSite", ...]:
        """Every call model `resolved` makes: each stage, and each component
        call node of a block that has layers, with what the model's prefill
        and decode walks hand it there. Memoized on the resolution, like
        its IR."""
        if resolved._call_sites_cache is not None:
            return resolved._call_sites_cache
        base = Deployment(Program.of(resolved)).at(batch=1, input_tokens=1)
        steps = {phase: base.during(phase) for phase in (PREFILL, DECODE)}
        walks = {phase: step.analyze() for phase, step in steps.items()}
        # An encoder's calls are made in its own phase: each encoder's
        # encode walk (with no media item, so what it hands a call is an
        # expression in its media axes).
        encodes = tuple(encode.analyze() for encode in base.encoders())
        analysis = walks[PREFILL]
        program = analysis.program
        formats = Formats.of(program).saved
        encoders = {stack.name: stack.aux_kind for stack in program.stacks
                    if stack.is_encoder}
        # Each encoder's media axes, which its calls are sized by: the ones
        # a request gives, and the ones the config fixes.
        media_axes: Dict[str, List[str]] = {}
        media_fixed: Dict[str, List[Tuple[str, int]]] = {}
        for entry in program.model_inputs.values():
            if entry.stack is None or entry.unmodeled is not None:
                continue
            media_axes.setdefault(entry.stack, []).extend(
                axis for axis in entry.axes if axis not in entry.fixed)
            media_fixed.setdefault(entry.stack, []).extend(
                (axis, int(value_of(value))) for axis, value
                in entry.fixed.items())
        # Each walk's tensors at each call, by where the call is: a phase
        # rebinds components to their decode forms, so its nodes are not
        # the prefill program's.
        tensors: Dict[str, Dict[Tuple[Optional[int], str], Any]] = {}
        for phase, walk in (*walks.items(),
                            *((ENCODE, encode) for encode in encodes)):
            at = tensors.setdefault(phase, {})
            at.update(((index, ""), formals)
                      for index, formals in walk.stage_inputs.items())
            for node in walk.program.nodes():
                shape = walk.shapes.values.get(node.uid)
                if node.is_component and shape is not None:
                    at[(walk.program.block_of(node.uid),
                        node.display_path)] = shape.ins
        # The dims each walk bound at a node, the encoders' in their walks.
        node_dims: Dict[int, Any] = dict(analysis.node_dims.values)
        for encode in encodes:
            node_dims.update(encode.node_dims.values)
        model_dims = set(resolved.dim_values)

        def producer_dims(walk: Any, formal: Tuple[Any, ...],
                          reference: Any) -> Dict[Any, Any]:
            """What the nodes that bound the names `formal` reads, and the
            walk's own binding does not, bound them to — the spec's text, or,
            for a value the engine bound with no text behind it, the value
            half the walk holds for it there (`node_dims`) — as a
            substitution in the names the call's frame binds. Each name is looked for upstream from the producers
            `reference` names, through their own inputs, to the node that
            binds it (a `pad_rows` keeps the `temporal` its `patchify`
            bound). A name its producers bind differently is left out —
            nothing says which one the call reads."""
            free = {name for axis in formal for name in names_of(axis)}
            # What the walk binds everywhere — the model's dims, the
            # workload, the phase — the isolated walk binds too.
            names_in_scope = model_dims | set(walk.binding.flat())
            free -= names_in_scope
            found: Dict[str, set] = {}
            pending = [source.occurrence.uid
                       for _iteration, sources in reference.table
                       for source in sources]
            seen: set = set()
            while pending and free - set(found):
                uid = pending.pop()
                if uid in seen:
                    continue
                seen.add(uid)
                bound = walk.node_dims.values.get(uid) or {}
                node = walk.program.by_uid(uid)
                raw = {name: str(text)
                       for name, text in node.dims_raw.items()}
                if node.dim_raw is not None or node.dim is not None:
                    raw["dim"] = str(node.dim_raw or node.dim)
                for name in free & set(bound):
                    found.setdefault(name, set()).add(
                        raw[name] if name in raw
                        else str(value_of(bound[name])))
                if walk.input_references.covers(uid):
                    pending.extend(
                        source.occurrence.uid
                        for i in range(walk.input_references.arity[uid])
                        for _iteration, sources
                        in walk.input_references.at(uid, i).table
                        for source in sources)
            symbols = {name: sp.Symbol(name) for name in names_in_scope}
            return {sp.Symbol(name): eval_dim_expr(texts.pop(), symbols)
                    for name, texts in found.items() if len(texts) == 1}

        # Each walk's producer dims at each call, by where the call is.
        given: Dict[str, Dict[Tuple[Optional[int], str], Any]] = {}
        for phase, walk in (*walks.items(),
                            *((ENCODE, encode) for encode in encodes)):
            at = given.setdefault(phase, {})
            for index, references in walk.stage_references.items():
                at[(index, "")] = tuple(
                    producer_dims(walk, formal, reference)
                    for formal, reference in zip(
                        walk.stage_inputs[index], references, strict=True))
            for node in walk.program.nodes():
                shape = walk.shapes.values.get(node.uid)
                if (node.is_component and shape is not None
                        and walk.input_references.covers(node.uid)):
                    at[(walk.program.block_of(node.uid),
                        node.display_path)] = tuple(
                        producer_dims(walk, formal,
                                      walk.input_references.at(node.uid, i))
                        for i, formal in enumerate(shape.ins))

        def handed(index: Optional[int], path: str):
            # A name a formal's producer bound on its own node means nothing
            # in the callee's frame (where `dim` is the callee's own): the
            # call hands the formal with it written out as the producer
            # bound it.
            return tuple(
                (phase, tuple(
                    tuple(expr_of(axis).xreplace(local)
                          if isinstance(expr_of(axis), sp.Basic)
                          else expr_of(axis) for axis in formal)
                    for formal, local in zip(
                        at[(index, path)],
                        given[phase].get((index, path))
                        or tuple({} for _ in at[(index, path)]),
                        strict=True)))
                for phase, at in tensors.items() if (index, path) in at)

        def handed_elements(index: int):
            return tuple(
                (phase, tuple(sorted(
                    (name, tuple(expr_of(axis) for axis in axes))
                    for name, axes in walk.stage_elements[index].items())))
                for phase, walk in (*walks.items(),
                                    *((ENCODE, encode) for encode in encodes))
                if index in walk.stage_elements)

        def stream_law(walk: Any, name: str, producer: int
                       ) -> Optional[StreamLaw]:
            """The law stream `name` carries out of `producer`, as the node
            that stated it binds its dims: those it declares, and those its
            frame — the layer variant and every enclosing call — rebinds, at
            the values the walk bound there; a declared one as the
            expression it is, with the frame's names written out."""
            if producer not in walk.stream_laws:
                return None
            uid, cached = walk.stream_laws[producer]
            if uid is None:
                return StreamLaw(stream=name, op=None, cached=cached, dims=(),
                                 node=walk.program.by_uid(producer).display_path)
            node = walk.program.by_uid(uid)
            bound = walk.node_dims.values[uid]
            index = walk.program.block_of(uid)
            rebound = (set(walk.program.blocks[index].dim_overrides)
                       if index is not None else set())
            caller = node.parent
            while caller is not None:
                rebound |= set(caller.dims)
                if caller.dim is not None or caller.dim_raw is not None:
                    rebound.add("dim")
                caller = caller.parent
            frame = {dim: value_of(bound[dim]) for dim in sorted(rebound)
                     if dim in bound}
            written = {sp.Symbol(dim): value for dim, value in frame.items()}
            declared = set(node.dims_raw) | (
                {"dim"} if node.dim is not None or node.dim_raw is not None
                else set())
            stated = {dim: expr_of(bound[dim]).xreplace(written)
                      if isinstance(expr_of(bound[dim]), sp.Basic)
                      else expr_of(bound[dim])
                      for dim in sorted(declared) if dim in bound}
            return StreamLaw(stream=name, op=node.op, cached=cached,
                             node=node.display_path,
                             dims=tuple({**frame, **stated}.items()))

        def handed_streams(index: int):
            # An encoder's block reads its streams in its own encode walk.
            at: Dict[str, Dict[str, Optional[StreamLaw]]] = {}
            for phase, walk in (*walks.items(),
                                *((ENCODE, encode) for encode in encodes)):
                for name, producers in walk.program.block_stream_reads(
                        index).items():
                    at.setdefault(phase, {})[name] = stream_law(
                        walk, name, producers[0])
            return tuple((phase, tuple(sorted(laws.items())))
                         for phase, laws in at.items())

        out: List[CallSite] = []
        for index, template in enumerate(program.blocks):
            if template.count == 0:
                continue
            stack = program.stack_of_block(index)
            meta = template.meta
            out.append(cls(
                address=template.checkpoint_prefix,
                role=resolved.spec.get_block(meta.template_name).role,
                stack=stack, formats=formats, component=meta.template_name,
                encoder=encoders.get(stack), call=cls.stage_call(index),
                media_axes=tuple(media_axes.get(stack, ())),
                media_fixed=tuple(media_fixed.get(stack, ())),
                invocation=index, label=_layers_label(stack, meta),
                handed=handed(index, ""),
                handed_elements=handed_elements(index),
                handed_streams=handed_streams(index),
                streams=tuple(sorted(_read_stream_declarations(
                    resolved, meta.template_name, stack).items())),
                frame=tuple(sorted(
                    (name, value_of(value))
                    for name, value in template.dim_overrides.items()))))
        for node in program.nodes():
            if not node.is_component:
                continue
            index = program.block_of(node.uid)
            meta = program.blocks[index].meta if index is not None else None
            if index is not None and program.blocks[index].count == 0:
                continue
            stack = program.stack_of(node)
            path = module_path(node, program.checkpoint_prefix(node))
            # The calling frame: the layer variant's rebinding, and each
            # enclosing call's `dims:`, at the values the walk bound where
            # the call is made — its caller's frame, or the block's.
            block = program.blocks[index] if index is not None else None
            rebound = set(block.dim_overrides) if block is not None else set()
            caller = node.parent
            while caller is not None:
                rebound |= set(caller.dims)
                if caller.dim is not None or caller.dim_raw is not None:
                    rebound.add("dim")
                caller = caller.parent
            if node.parent is not None:
                bound = node_dims[node.parent.uid]
            else:
                bound = block.dim_overrides if block is not None else {}
            own = {name: str(node.dims_raw.get(name, value))
                   for name, value in node.dims.items()}
            if node.dim_raw is not None or node.dim is not None:
                own["dim"] = str(node.dim_raw or node.dim)
            out.append(cls(
                address=f"{path}." if path else "", role=node.role,
                stack=stack, formats=formats, component=node.op,
                encoder=encoders.get(stack),
                media_axes=tuple(media_axes.get(stack, ())),
                media_fixed=tuple(media_fixed.get(stack, ())),
                call=(f"{index}/{node.display_path}" if index is not None
                      else f"/{node.display_path}"),
                invocation=index, path=node.display_path,
                label=(_layers_label(stack, meta) if meta is not None
                       else Stack.display_name(stack))
                + f" · {node.display_path}",
                handed=handed(index, node.display_path),
                handed_streams=(handed_streams(index) if index is not None
                                else ()),
                frame=tuple(sorted((name, value_of(bound[name]))
                                   for name in rebound)),
                binds=tuple(sorted(own.items())),
                params=node.call_params,
                streams=tuple(sorted(_read_stream_declarations(
                    resolved, node.op, stack,
                    params=dict(node.call_params)).items()))))
        resolved._call_sites_cache = tuple(out)
        return resolved._call_sites_cache


def _layers_label(stack: str, meta: Any) -> str:
    """The layers a block invocation runs at, in its stack: `backbone ·
    layers 3–60`; a few at most when they do not run contiguously."""
    if meta.layer_indices is not None:
        layers = sorted(meta.layer_indices)
    else:
        layers = list(range(int(meta.count)))
    if len(layers) == 1:
        span = f"layer {layers[0]}"
    elif layers == list(range(layers[0], layers[-1] + 1)):
        span = f"layers {layers[0]}–{layers[-1]}"
    elif len(layers) <= 4:
        span = "layers " + ", ".join(str(n) for n in layers)
    else:
        span = (f"{len(layers)} layers {layers[0]}, {layers[1]}, …, "
                f"{layers[-1]}")
    return f"{Stack.display_name(stack)} · {span}"


def isolate_resolved(resolved: Any, template: str,
                     dim_overrides: Optional[Dict[str, Any]] = None,
                     site: Optional[CallSite] = None):
    """Run one block/component as a one-stage model, same library and dims.

    `project_block` and cache isolation share this so a component is not
    wrapped two different ways. The stage is a block holding one node that
    calls the component (`CallSite.block`), as a model block calls it, so
    what the component declares on its invocation — its cache, its recurrent
    state — enters the program the way it does in the model. `site` is the
    model's call it runs at (`CallSite.of`; the component's only call when
    None): its tensors take that address, role and checkpoint formats, so
    they price as they do inside that model; the isolation is made from the
    call's frame (`CallSite.frame`) with the call's own `dims:`
    (`CallSite.binds`), and the stage receives what the call hands the
    component (`CallSite.formals`), as a model receives its inputs.
    """

    site = site if site is not None else CallSite.of(resolved, template)
    if (site.invoked and site.component != template
            and binding_of(resolved.spec, site.component, template)):
        # The model's binding of `template` at this call is a synthesized
        # specialization: its slot bindings live in the name
        # (`csa2_attention__compressor-…`). Isolating the bare component
        # here would leave its `<param>` ops unbound.
        template = site.component
    site = replace(site, component=template)
    dims = site.dims_in(resolved)
    for name, value in (dim_overrides or {}).items():
        dims[name] = (eval_dim_expr(str(value), dims)
                      if isinstance(value, str) else value)
    spec = resolved.spec
    body = spec.get_block(template)
    if not site.invoked and body.open_slots:
        raise ValueError(
            f"{site.call} calls another component, and {template!r} leaves "
            f"its slots {body.open_slots} to its caller: a call of another "
            f"component fills none of them. Evaluate {template!r} at one of "
            f"the model's own calls of it")
    taken = set(spec.all_block_names())
    if site.block in taken:
        raise ValueError(
            f"isolating {template!r}: the spec already has a block or "
            f"component named {site.block!r}")
    inputs = list(body.inputs)
    if len(inputs) != len(site.formals):
        raise ValueError(
            f"isolating {template!r} at {site.described()}: the call hands "
            f"over {len(site.formals)} tensors and {template!r} takes "
            f"{len(inputs)} ({inputs})")
    binds = dict(site.binds)
    call = BlockGraph(
        inputs=inputs,
        graph={template: GraphNode(op=template, inputs=inputs,
                                   role=site.role, module=None,
                                   dim=binds.pop("dim", None), dims=binds,
                                   # The model's own call of the component
                                   # keeps its slot bindings; a foreign
                                   # component taking the call's place has
                                   # no claim on them.
                                   params=(dict(site.params)
                                           if site.invoked else {}))},
        outputs=[template])
    # The model-wide streams the component reads come from producers outside
    # the isolation. The stage declares them as the model does, so the walk
    # binds the reads, and the evaluator seeds them at their declared shape
    # (`Evaluator._isolated_stream`).
    # A stack that declares streams is a repeated one; the isolation is one
    # iteration of it.
    # A stream's `seed` names a layer of the model's stack that writes it at
    # the stack's entry; the isolation's one layer is never that one, and
    # the evaluator seeds the read itself, so the declaration drops it.
    reads = {stream: decl.model_copy(update={"seed": None})
             for stream, decl in site.stream_reads(resolved).items()}
    # An appended stream's element the call read (`read: each`) was appended
    # by another stack: the stage takes it at the shape the model handed the
    # call, as it takes its formals.
    for stream, axes in site.elements.items():
        reads.setdefault(stream, StreamSpec(
            shape=[str(axis) for axis in axes]))
    name = site.stack or template
    if name in inputs:
        raise ValueError(
            f"isolating {template!r}: its formal {name!r} names the stage")
    stage = ModelLayer(name=name, template=site.block, inputs=inputs,
                       **({"repeat": "1", "streams": reads} if reads else {}))
    if site.encoder is not None:
        # An encoder's call keeps its encoder stack: that is what says its
        # cache lives one forward pass and nothing reads it later.
        stage = ModelLayer(name=site.stack, group=True,
                           aux_kind=site.encoder, layers=[stage])
    isolated_spec = spec.model_copy(update={
        "model": [stage],
        # What the stage receives is the call's (`Program.model_inputs`
        # reads `site.formals`), not the host's model inputs.
        "model_inputs": {},
        "blocks": {**(spec.blocks or {}), site.block: call},
    })
    return ResolvedSpec(isolated_spec, dims, resolved.config,
                        dict(resolved.view_values), site=site,
                        declared_by=resolved.declared_by,
                        checkpoint=resolved.checkpoint)


def _read_stream_declarations(resolved: Any, template: str,
                               stack: Optional[str] = None,
                               params: Optional[Dict[str, str]] = None
                               ) -> Dict[str, Any]:
    """The host spec's declarations of the inner streams `template` reads,
    itself or through a component it calls: the declaring stack's, and of
    two stacks declaring one name, the one the call site runs in.

    `params` is the call's own slot bindings: a `<param>` op resolves
    through them (over the graph's declared defaults), the same frame the
    walk's `_build_frame` pushes — a bare parameterized component read
    through no binding would report the union of nothing its slots read."""
    spec = resolved.spec
    declared: Dict[str, Dict[Tuple[str, ...], Any]] = {}
    for (path, name), decl in spec.stream_names().inner.items():
        declared.setdefault(name, {})[path] = decl
    graphs = {**(spec.components or {}), **(spec.blocks or {})}
    out: Dict[str, Any] = {}
    seen: set = set()
    pending = [(template, dict(params or {}))]
    while pending:
        name, bindings = pending.pop()
        graph = graphs.get(name)
        if graph is None:
            continue
        frame = {pname: str(pspec.default)
                 for pname, pspec in (graph.params or {}).items()
                 if pspec.default is not None}
        frame.update(bindings)
        key = (name, tuple(sorted(frame.items())))
        if key in seen:
            continue
        seen.add(key)
        for stream in graph.reads or ():
            found = declared.get(stream, {})
            if len(found) > 1:
                found = {path: decl for path, decl in found.items()
                         if path[-1] == stack}
            if len(found) == 1:
                (out[stream],) = found.values()
        for node in graph.graph.values():
            op = node.op
            if (isinstance(op, str) and op.startswith("<")
                    and op.endswith(">")):
                op = frame.get(op[1:-1])
            if op:
                pending.append((op, {k: str(v)
                                     for k, v in (node.params or {}).items()}))
    return out


def analyze_isolated(resolved: Any, template: str,
                     dim_overrides: Optional[Dict[str, Any]] = None,
                     site: Optional[CallSite] = None, *,
                     precision: Any = None):
    """Evaluate one component as a one-stage model.

    Cache isolation and axiom checks share this so they do not each
    construct a `Deployment`. `get_block` runs first: a name the spec
    does not have must raise before the walk.
    """
    resolved.spec.get_block(template)
    return Deployment(Program.of(
        isolate_resolved(resolved, template, dim_overrides, site)
    ), precision=precision).at(batch=1, input_tokens=1).analyze()
