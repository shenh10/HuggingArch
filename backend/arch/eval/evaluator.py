"""`Evaluator` — the walk, with no mutable state.

`SpecEngine` is an object whose methods communicate through attributes on `self`:
the dim binding, four frame stacks, two per-rank hooks. `Evaluator` holds only the
`Program`, and everything scoped travels as a parameter. What comes out is
`Layer`s rather than one twenty-nine-field record, so a consumer that wants cost
does not also receive ten rendered display strings.

Coverage
--------
Primitives and spec-scoped custom ops alike (one `OpRef` protocol), components,
expert-bank multipliers, MoE dispatch narrowing, source ops, secondary ports, the
sparse-attention effective `s_kv`, and both value domains — numeric and symbolic
are the same walk against different bindings.

Where an op that reads the cache (it declares a KV law) names in its `shape_in`
a symbol the binding marked authoritative, the node's extent wins over the
upstream tensor. That set is only `s_kv`: decode attention reads the cache
while the K/V leaves emit the new token. Every other op reads the tensor that
arrives. Do not widen it to `s_query` or `s_ctx` — those follow arriving
tensors. `Unsupported` remains
for a node this walk cannot model at all — raised, never approximated, because an
oracle cannot tell "not implemented" from "implemented wrong" if the gap returns a
plausible number.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType
from typing import (Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple)

import sympy as sp

import backend.arch.dims as dim_registry

from .refusal import SpecRefusal, node_context
from .analysis import Analysis, OwnCopies
from .binding import Binding
from .dim import (Dim, Domain, ReadTracking, algebra, apply, assuming, eval_dim_expr,
                  expr_of, ratio, require, same_law,
                  minimum, names_of, value_of)
from . import formats as formats_mod
from .precision import AS_SHIPPED, ResolvedPrecision
from . import transition as transition_mod
from .flops import (bind_labels, flops_over_operands,
                    record_binding)


from ..ops import DISPATCH_OP
from .layer import Layer
from .node import Evaluable
from .occurrence import Occurrence, Reference, ReferenceTable, Source
from .op_ref import KVLength, OpRef, StreamLaw
from .program import (HISTORY_AXIS, BlockStage, Carry, OpStage, Program,
                      Stack, Stage)
from .rule import Rule
from .scope import Carried, Scope
from .storage import is_expert_axis, leads_with_expert_axis
from .values import CACHE_LEN, Cost, Shape, product


@dataclass(frozen=True)
class CarryEntry:
    """What formal 0 of a carrying stage reads (`_eval_link`): at each of
    its iterations, the previous iteration's exit — or, at the chain's
    first iteration, what the stack starts from (`start`). The exits are
    known once the whole chain has walked; `Walk.into_analysis` resolves
    this into the `Reference` the read is."""

    key: Tuple[str, str]
    iterations: Tuple[int, ...]
    start: Reference
    #: The model input the stack starts from, when this stage holds the
    #: chain's first iteration and the start is one.
    model_input: Optional[str] = None

    def resolve(self, exits: Sequence[Tuple[Tuple[int, ...], int]]
                ) -> Reference:
        chain = sorted((i, uid) for iterations, uid in exits
                       for i in iterations)
        order = [i for i, _uid in chain]
        table = []
        for iteration in self.iterations:
            at = order.index(iteration)
            if at:
                previous, uid = chain[at - 1]
                found: Tuple[Source, ...] = (
                    Source(Occurrence(uid, previous), "out", "carry"),)
            else:
                found = self.start.via("carry").at(iteration)
            table.append((iteration, found))
        return Reference(table=tuple(table), model_input=self.model_input)


def _share(active: Any, stored: Any) -> Any:
    """`active / stored`, exact in both value domains.

    The copies a node holds and the copies that run are counts; the SHARE of
    itself a step runs is their quotient, and it is read as a number (a
    saturation fraction, a per-token weight demand). Kept exact — a
    `Fraction` where the walk is numeric, the quotient of the two expressions
    where it is symbolic — because the fraction is multiplied by byte counts
    before anything rounds.
    """
    return apply(lambda a, b: a / b, Fraction, active, stored)


@dataclass
class Walk:
    """The maps one run fills in, gathered into one object.

    Every one of these is `{uid: something the walk decided}`, they are all
    written by the same recursion, and they all become a `Layer` on the
    `Analysis` at the end. Passing them as seven positional dicts made the
    signature the reason not to add an eighth — which is how the per-node copy
    count came to be read back off the IR node instead of resolved here, where
    the binding that decides it is in scope.

    Mutable and short-lived, unlike everything else in this package: it exists
    between `run` entering and `run` returning, and `Analysis` is built from it.
    """

    costs: Dict[int, Cost] = field(default_factory=dict)
    shapes: Dict[int, Shape] = field(default_factory=dict)
    node_dims: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # What a node's op rebinds in its own frame (`Analysis.node_rebinds`).
    node_rebinds: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # The function constants each leaf's op takes, as its frame binds them.
    consts: Dict[int, Dict[str, float]] = field(default_factory=dict)
    formats: Dict[int, Any] = field(default_factory=dict)
    storage: Dict[int, Any] = field(default_factory=dict)
    # A tied node's own tensor the checkpoint ships anyway
    # (`tied_weight.checkpoint_copy`): `(stored tensors, formats)`, kept apart
    # from `storage` because serving never loads it.
    checkpoint_copies: Dict[int, Any] = field(default_factory=dict)
    # Physical copies of each node the model holds. Cumulative — block count
    # times every enclosing `repeat:` — and resolved against the frame's
    # binding, so a per-rank walk yields per-rank counts.
    instances: Dict[int, Any] = field(default_factory=dict)
    # The share of each node one token runs (`Analysis.active_share`).
    active_share: Dict[int, Any] = field(default_factory=dict)
    # Each node's own multipliers, of which the two above are products.
    own_copies: Dict[int, OwnCopies] = field(default_factory=dict)
    # Where each tensor LIVES, threaded along the same edges as its shape.
    # A second traversal answering this independently is free to disagree with
    # the first.
    states: Dict[int, Any] = field(default_factory=dict)
    # Port-granular view of `states` — see `Analysis.port_states`. BOTH maps
    # are written only through `set_state` and read only through `state_at`,
    # which is what keeps them one fact: the primary port mirrors into the
    # uid-keyed map every uid-keyed reader already uses, in the same call.
    port_states: Dict[Tuple[int, str], Any] = field(default_factory=dict)
    # What each primitive's physical kernel consumes.  Usually this is the
    # state after its edge transition; a persistent-state input may retain the
    # SSA edge only as logical identity while reading a separately placed
    # runtime buffer (see transition.InputRequirements).
    input_states: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    kernel_shapes: Dict[int, Shape] = field(default_factory=dict)
    redistributes: List[Any] = field(default_factory=list)
    # What each op said about the cache it touches, in `CACHE_LEN` rather than
    # at this walk's cache length. Sparse — only the handful of ops per model
    # that declare a cardinality at all.
    cache_positions: Dict[int, Any] = field(default_factory=dict)
    # The substitution each node's FLOPs evaluation used — identity → value,
    # collected by `flops_from_shape`'s `record`. Display binds THIS table
    # rather than reconstructing the values afterwards.
    flops_bindings: Dict[int, Dict[Any, Any]] = field(default_factory=dict)
    # Same, for the params figure (`op.params(dims, ...)`) — the node's own
    # dim binding as evaluated.
    params_bindings: Dict[int, Dict[Any, Any]] = field(default_factory=dict)
    # How many KV positions each tensor holds, at this walk's cache length and
    # with it held free — `KVLength`, keyed on the node that produced the
    # tensor. `cache_positions` is what an op said; this is what its output
    # carries to whoever reads it, which is a different question the moment a
    # `kv_stream_concat` joins two KV streams into one.
    kv_lengths: Dict[int, Any] = field(default_factory=dict)
    # The nodes that read their keys through a per-query index list — a
    # selecting op's own pairs (`kv_pairs_per_step_fn`), or a
    # `kv_stream_concat` union's. Their kernels gather those rows query by
    # query (`traffic`); the pairs are the cached tensor's
    # (`CachedTensor.pairs`).
    gathered: Set[int] = field(default_factory=set)
    # A recurrent component's per-request state sizes, `{state name:
    # elements}` on the component's uid.
    cache_state: Dict[int, Any] = field(default_factory=dict)
    # How many times a step reads each of those states, `{state name: extent}`
    # on the component's uid (`reads:`, once when absent).
    cache_state_reads: Dict[int, Any] = field(default_factory=dict)
    # Inner stream values, keyed on the node that produced each: what a
    # `stream_out` of a node published. A reader takes the values of the
    # producers `Program.block_stream_reads` names for it; which producer that
    # is, is the reader's declaration, never this map's order.
    stream_values: Dict[int, Carried] = field(default_factory=dict)
    # The cache law each published inner stream carries, keyed on the node
    # that published it: the node whose op stated the length
    # (`KVLength.stated_by`; None for a union's) and whether the publishing
    # node is a cache entry (`_published_length`). What an isolated call of
    # a reader is handed with the stream (`op_ref.StreamLaw`).
    stream_laws: Dict[int, Tuple[Optional[int], bool]] = field(
        default_factory=dict)
    # In an isolated walk, the inner streams the component reads that the
    # model publishes to no layer of its call: seeded at their declared
    # shape, with no law (`_isolated_stream`).
    unpublished: Set[str] = field(default_factory=set)
    # What each reading template was seeded with, per stream, and the
    # producers it reads: a producer that publishes after the reader walked
    # is held to the seeded value (`Evaluator._publish_stream_outs`).
    stream_seeds: Dict[str, List[Tuple[FrozenSet[int], Carried, str]]] = \
        field(default_factory=dict)
    # Published appended streams: each element, in order. A stack that
    # appends writes them once all its stages have walked; a stream with no
    # elements is not published.
    appended: Dict[str, Tuple[Carried, ...]] = field(default_factory=dict)
    # Streams local to a stack: the value each carrying stack
    # (`StreamSpec.carries`) hands to its next stage, keyed by
    # `(declaring layer, stream name)` — two stacks that both call theirs
    # `hidden` hold two values.
    carried: Dict[Tuple[str, str], Carried] = field(default_factory=dict)
    # A stage's output as the stage hands it on — settled at the boundary by
    # `_closed` — keyed on the producing port. A later stage's node whose ref
    # resolved to that producer reads this, not the unsettled output: reading
    # the raw one re-prices the reduction the boundary already recorded.
    handed_on: Dict[Tuple[int, str], Carried] = field(default_factory=dict)
    # What each block stage binds its template's formals to, by block
    # invocation (`Program.blocks` ordinal) (`Analysis.stage_inputs`).
    stage_inputs: Dict[int, Tuple[Tuple[Any, ...], ...]] = field(
        default_factory=dict)
    # What each of those formals reads, as the walk bound it: a `Reference`,
    # or a carried stream's `CarryEntry` (`Analysis.stage_references`).
    stage_references: Dict[int, Tuple[Any, ...]] = field(default_factory=dict)
    # The appended streams' elements each block stage read (`read: each`),
    # by stream name (`Analysis.stage_elements`).
    stage_elements: Dict[int, Dict[str, Tuple[Any, ...]]] = field(
        default_factory=dict)
    # The frame each block stage's body ran in, by block invocation
    # (`Analysis.stage_frames`).
    stage_frames: Dict[int, Binding] = field(default_factory=dict)
    # What each node input reads, as the walk bound it: a `Reference`, or a
    # carried stream's `CarryEntry`, resolved once its chain has walked.
    input_references: Dict[Tuple[int, int], Any] = field(default_factory=dict)
    # How many inputs the walk bound per evaluated node.
    input_arity: Dict[int, int] = field(default_factory=dict)
    # Per carried stream (`Carry.key`), each stage's iterations and the node
    # its exit is.
    chain_exits: Dict[Tuple[str, str], List[Tuple[Tuple[int, ...], int]]] = \
        field(default_factory=dict)
    # Per stage name, the `(uid, port)` a later `inputs:` naming it reads:
    # its last iteration's exit.
    stage_outputs: Dict[str, Tuple[int, str]] = field(default_factory=dict)

    PRIMARY_PORT = "out"

    def set_state(self, uid: int, state: Any, *, port: str = "out") -> None:
        """The single write path for a node's distributed state.

        The primary port mirrors into `states[uid]` here — one call, so the
        uid-keyed view and the port-keyed view cannot disagree. A rule with
        port-granular states writes its secondary ports with `port=...`;
        a rule that only has one output writes the primary.
        """
        self.port_states[(uid, port)] = state
        if port == self.PRIMARY_PORT:
            self.states[uid] = state

    def state_at(self, uid: int, *, port: str = "out") -> Any:
        """The single read path — a port with no state of its own reads the
        producer's primary, which is what every secondary port used to
        do when the walk stored one state per node."""
        if (uid, port) in self.port_states:
            return self.port_states[(uid, port)]
        return self.states.get(uid)

    def into_analysis(self, program: Program, binding: Binding) -> "Analysis":
        """The maps, as layers over `program`.

        Here rather than at the end of `run` because the correspondence between
        a map and the layer it becomes is a fact about `Walk` — adding a map and
        forgetting to carry it out was a live possibility while `run` listed all
        eight by hand, and the count is the thing that grows.

        `roles` is read off the graph rather than out of a map: the walk does
        not decide a role, `iter_tree` already propagated it, so producing it
        here keeps the walk from carrying a ninth dict that only ever copies.
        """
        return Analysis(
            program=program,
            binding=binding,
            cost=Layer(program, self.costs),
            shapes=Layer(program, self.shapes),
            roles=Layer(program, {n.uid: n.role for n in program.nodes()
                                  if n.role}),
            storage=Layer(program, self.storage),
            formats=Layer(program, self.formats),
            checkpoint_copies=Layer(program, self.checkpoint_copies),
            node_dims=Layer(program, self.node_dims),
            node_rebinds=Layer(program, self.node_rebinds),
            consts=Layer(program, self.consts),
            instances=Layer(program, self.instances),
            active_share=Layer(program, self.active_share),
            own_copies=Layer(program, self.own_copies),
            states=Layer(program, self.states),
            port_states=dict(self.port_states),
            input_states=dict(self.input_states),
            kernel_shapes=Layer(program, self.kernel_shapes),
            cache_positions=Layer(program, self.cache_positions),
            gathered=frozenset(self.gathered),
            kv_lengths=Layer(program, self.kv_lengths),
            flops_bindings=Layer(program, self.flops_bindings),
            params_bindings=Layer(program, self.params_bindings),
            cache_state=Layer(program, self.cache_state),
            cache_state_reads=Layer(program, self.cache_state_reads),
            redistributes=tuple(self.redistributes),
            assumptions=(),
            appended={name: tuple((element.producer, element.state)
                                  for element in elements)
                      for name, elements in self.appended.items()},
            handed_states={value: closed.state
                           for value, closed in self.handed_on.items()},
            stage_inputs=dict(self.stage_inputs),
            stage_elements={index: dict(read)
                            for index, read in self.stage_elements.items()},
            stage_frames=dict(self.stage_frames),
            stage_references={
                index: tuple(
                    found.resolve(self.chain_exits[found.key])
                    if isinstance(found, CarryEntry) else found
                    for found in found_all)
                for index, found_all in self.stage_references.items()},
            input_references=ReferenceTable(
                program=program, arity=self.input_arity,
                references={
                    key: (found.resolve(self.chain_exits[found.key])
                          if isinstance(found, CarryEntry) else found)
                    for key, found in self.input_references.items()}),
            stage_outputs=dict(self.stage_outputs),
            stream_laws=dict(self.stream_laws),
        )

class Evaluator:
    """Stateless over the walk; holds only the program."""

    def __init__(self, program: Program, *,
                 precision: Any = None,
                 rule: Rule,
                 workload: Any,
                 handed: Mapping[str, Tuple[Any, ...]] = MappingProxyType({})
                 ) -> None:
        self.program = program
        # What each encoder output the language model reads holds, per
        # request (`Program.handoffs`, `HandedTensor`): the encoders are
        # another phase's walk, so their outputs arrive here like inputs.
        self.handed = dict(handed)
        # Nodes of encoder stacks: walked in the encode phase's binding,
        # which binds no language-model step.
        self._encoders = (program.encoder_uids
                          if program.resolved.site is None else frozenset())
        # How this deployment distributes tensors. Required: the evaluator
        # must not know the layout package exists, so the rule is built by
        # the caller — `Deployment.analyze` for production, an explicit
        # `LayoutPropagation.of(ParallelConfig(), phase)` for a single-rank
        # walk in tests.
        self.rule = rule
        # The `Workload` the binding was made from.
        self.workload = workload
        # Cost formulas read extents off the tensors that arrived.
        # Sparse ops rebind `s_kv` in `dims` first (`Min(s_kv, s_window)` /
        # top-k); `_binding_authoritative` then writes that effective length
        # onto the input-shape axis the template names `s_kv`, so
        # `flops_from_shape` and the formula see the same count. The window
        # is that authoritative axis, not a second field on the tensor.
        # The deployment's precision profile, read against this program
        # (`precision.ResolvedPrecision`). Both halves of "what dtype is this
        # tensor" come through it: what the checkpoint holds, which the spec
        # declares, and what this deployment runs at, the profile's weight
        # table — None for a model served as it was saved.
        if precision is None:
            precision = ResolvedPrecision(AS_SHIPPED, program)
        self.precision = precision
        self._formats = formats_mod.Formats.of(program,
                                               precision.weight_intent)
        # The iteration space a pipeline partition cuts is whole at any pp:
        # each repeated layer's templates cover its iterations exactly once.
        _ = program.iteration_space
        # A join is a node the walk evaluates, not a binding it improvises:
        # the program it walks has them in its graph (`Program.lowered`).
        if program.joined_streams and not program.is_lowered:
            raise ValueError(
                f"{program.spec.name}: stream(s) "
                f"{sorted(program.joined_streams)} are read joined, and the "
                f"program has not lowered the joins to nodes; walk "
                f"`Program.lowered()`")

    # -- entry point -----------------------------------------------------

    def run(self, binding: Binding,
            encoder_bindings: Mapping[str, Binding] = MappingProxyType({})
            ) -> Analysis:
        """`_run`, recording the conditions its formulas hold under.

        A walk of numbers checks every condition on the spot and records
        nothing. A walk of the workload's symbols records, on
        `Analysis.assumptions`, each condition on the point it states
        (`dim.require`); a reader checks a point against them
        (`Analysis.admit`). A condition is decided where it is written only
        over the counts the walk's frames bind, as the door admitted them
        (`transforms._declared`; `dim.Domain`) — the workload's among them —
        which the walk keeps for its readers (`Analysis.domain`)."""
        domain = Domain(value for frame in (binding, *encoder_bindings.values())
                        for value in frame.flat().values())
        with assuming(domain=domain) as assumed:
            analysis = self._run(binding, encoder_bindings)
        analysis.assumptions = tuple(assumed.held)
        analysis.domain = domain
        return analysis

    def _run(self, binding: Binding,
             encoder_bindings: Mapping[str, Binding] = MappingProxyType({})
             ) -> Analysis:
        """Evaluate every node against `binding`.

        `binding` carries the workload (`B`, `s_query`, `s_kv`, `s_past`,
        `s_pairs`). Extents stay global; per-rank geometry is the
        layout rule's, not a divisor on this frame. A program that walks an
        encoder beside the language model (`Program.CHECKPOINT`) walks it in
        the encode phase's binding, `encoder_bindings[stack]`.
        """
        if self.program.resolved.site is not None and not self.rule.is_trivial:
            raise ValueError(
                f"{self.program.spec.name}: an isolated component runs on one "
                f"rank. Its call hands it the shapes of its formals "
                f"(`CallSite.formals`), not where they live on a mesh — that "
                f"is the model's walk on the mesh; analyze the whole model "
                f"there and read the call")
        walk = Walk()
        scope = Scope(binding=binding)
        scopes = {stack.name: Scope(binding=encoder_bindings.get(stack.name,
                                                                 binding))
                  for stack in self.program.walk_stacks if stack.is_encoder}

        # What each stage hands on, by the name later `inputs:` read it by.
        # A stage's formals bind its authored `inputs:` in order; a stack
        # that carries a stream binds formal 0 to the stream's current value
        # instead, of which `inputs[0]` is only the initial one. The model's
        # inputs are there before any stage runs. The walk only looks names
        # up — it does not invent a previous stage.
        produced_by: Dict[str, Carried] = {
            name: self._model_input(entry, scopes.get(entry.stack, scope))
            for name, entry in self.program.walked_inputs.items()}
        # The language model reads what the encoders hand over; an encoder
        # reads none of it.
        handoffs = (self.program.handoffs
                    if self.program.walks_language_model else ())
        missing = sorted(h.name for h in handoffs
                         if h.name not in self.handed)
        if missing:
            raise ValueError(
                f"{self.program.spec.name}: the language model reads encoder "
                f"output(s) {missing}, and the walk was handed none; analyze "
                f"through `Deployment`, which runs the encoders first")
        for handoff in handoffs:
            tensors = tuple(self._handed(handoff.name, tensor, scope)
                            for tensor in self.handed[handoff.name])
            if handoff.stream:
                # A stream with no elements is not published (`_append`).
                if tensors:
                    walk.appended[handoff.name] = tensors
            else:
                (produced_by[handoff.name],) = tensors
        # Refuses a reader that does not pair before anything is walked.
        _ = self.program.stream_pairings
        for stack in self.program.walk_stacks:
            # An encoder walked beside the language model, in its own phase.
            here = scopes.get(stack.name, scope)
            # A stage's redistributions are listed where the stage is
            # declared, not where the chain walks it: a record's position
            # names its collective (`commN`), and a name should not move
            # because variants are walked in iteration order.
            records: Dict[int, List[Any]] = {}
            exits: Dict[int, Optional[Carried]] = {}
            for stage in stack.walk_order():
                start = len(walk.redistributes)
                declared = stage.upstream
                finals = self._final_inner_streams(stage, walk)
                missing = [n for n in declared
                           if n not in produced_by and n not in walk.appended
                           and n not in finals]
                if missing:
                    raise ValueError(
                        f"{self.program.spec.name}: stage {stage.name!r} "
                        f"declares `inputs: {list(declared)}`, and "
                        f"{missing} named nothing evaluated before it. A "
                        f"name that resolves to nothing means the stage "
                        f"reads a tensor that does not exist.")
                joined = self._joined(stage, walk)
                inbounds = [
                    produced_by[n] if n in produced_by
                    else finals[n] if n in finals
                    else joined.get(n) if (isinstance(stage, OpStage)
                                           and stage.node.axis is not None)
                    else self._appended_input(stage, n, walk, here)
                    for n in declared]
                layer_stages = [st for st in stack.stages
                                if st.layer_path == stage.layer_path]
                prefix = self._prefix(stage, layer_stages, exits,
                                      produced_by, here)
                if stage.carry is not None:
                    out_shape = self._eval_link(
                        stack, stage, stage.carry, inbounds, here, walk,
                        prefix=prefix, joined=joined)
                else:
                    entry = inbounds[0] if inbounds else None
                    out_shape = self._eval_stage(
                        stage, entry, here, walk, inbounds=inbounds,
                        prefix=prefix, joined=joined)
                    if isinstance(stage, OpStage):
                        self._refuse_view_selecting_an_iteration(
                            stage, inbounds, out_shape, walk)
                        if stage.node.uid in self.program.stream_seed_nodes:
                            # An inner stream's `seed`: what its readers
                            # read at the stack's entry.
                            walk.stream_values[stage.node.uid] = out_shape
                if out_shape is not None:
                    produced_by[stage.name] = out_shape
                exits[id(stage)] = out_shape
                records[id(stage)] = walk.redistributes[start:]
                del walk.redistributes[start:]
                if all(id(st) in exits for st in layer_stages):
                    self._name_final(layer_stages, exits, produced_by, walk)
                    self._append(layer_stages, exits, produced_by, walk)
            for stage in stack.stages:
                walk.redistributes.extend(records[id(stage)])

        analysis = walk.into_analysis(self.program, scope.binding)
        analysis.rule = self.rule
        analysis.precision = self.precision
        analysis = self.rule.close_analysis(analysis)
        return analysis

    def _name_final(self, stages: Sequence[Stage],
                    exits: Mapping[int, Optional[Carried]],
                    produced_by: Dict[str, Carried], walk: Walk) -> None:
        """Once every stage a model layer expanded into has walked, what a
        later `inputs:` naming the layer reads: its last iteration's exit
        (a carried stack's stream after its last iteration), or the op's."""
        name = stages[0].name
        if name not in produced_by:
            return
        if isinstance(stages[0], OpStage):
            (stage,) = stages
            iteration: Optional[int] = None
        else:
            iteration, stage = max(
                ((i, st) for st in stages
                 for i in st.template.layer_indices),
                key=lambda pair: pair[0])
        out = exits[id(stage)]
        if out is None:
            raise ValueError(
                f"{self.program.spec.name}: {name!r} is read by its name, and "
                f"its last iteration ({iteration}, template "
                f"{stage.template.name!r}) produced no tensor")
        uid, port = out.producer
        walk.stage_outputs[name] = (uid, port)
        produced_by[name] = replace(
            produced_by[name], origin=Reference.fixed(
                (Source(Occurrence(uid, iteration), port, "named"),)))

    @staticmethod
    def _free_names(values: Sequence[Any], binding: Binding) -> set:
        """The names `values` read that `binding` does not bind."""
        names: set = set()
        for value in values:
            for symbol in algebra(value_of(value)).free_symbols:
                if str(symbol) not in binding:
                    names.add(str(symbol))
        return names

    def _unbound_in_encoder(self, node: Evaluable,
                            names: set) -> SpecRefusal:
        """An encoder node reading a name its phase does not bind: the
        language model's step (`s_ctx`, `s_kv`, …) above all, which an
        encoder never sees."""
        stack = self.program.stack_of(node)
        inputs = sorted(name for name, entry in self.program.model_inputs.items()
                        if entry.stack == stack)
        return SpecRefusal(
            f"{self.program.spec.name}: {node.display_path} ({node.op}) "
            f"reads {sorted(names)}, which the encode phase does not bind; "
            f"an encoder's rows come from its own input axes "
            f"({', '.join(f'model_inputs.{n}.axes' for n in inputs)}), never "
            f"from the language model's step",
            declaration="dims",
            fix="write the extent from the encoder's input axes and the "
                "patch grid `patchify` declares, not from s_ctx / s_kv",
            node=node.display_path)

    def _model_input(self, entry: Any, scope: Scope) -> Carried:
        """A model input as it arrives, placed where the rule puts a model
        input of that structure. The token ids are `[B, s_ctx]`. An image is
        `[N, n, H, W]` and an audio clip `[N·n, s]` (`[N·n, s, width]` with a
        width): `N` the requests — times the items of the input whose items
        hold these (`per_item_of`) — and the rest the axes the workload gives
        (`Workload.media`, bound by `Deployment.binding`). An isolated
        component's formal arrives at the shape its call hands over, valued
        in this frame."""
        binding = scope.binding
        if entry.shape is not None:
            flat = scope.flat_dims()
            unbound = sorted(
                {name for axis in entry.shape for name in names_of(axis)}
                - set(flat))
            if unbound:
                raise ValueError(
                    f"{self.program.spec.name}: its call hands formal "
                    f"{entry.name!r} over as {list(entry.shape)}, and "
                    f"{unbound} is bound by none of the model's dims, the "
                    f"workload or the call's frame — the model's walk binds "
                    f"it only around that call; evaluate the component "
                    f"that makes the call")
            axes: Tuple[Any, ...] = tuple(
                eval_dim_expr(str(axis), flat) for axis in entry.shape)
        elif entry.kind == "tokens":
            axes = (binding["B"], binding["s_ctx"])
        else:
            lead = binding["B"]
            parent = entry.per_item_of
            while parent is not None:
                holder = self.program.model_inputs[parent]
                lead = lead * binding[holder.axes[0]]
                parent = holder.per_item_of
            values = tuple(binding[axis] for axis in entry.axes)
            axes = ((lead, *values) if entry.kind in ("image", "video")
                    else (lead * values[0], *values[1:]))
            if entry.width is not None:
                axes = (*axes, eval_dim_expr(entry.width, scope.flat_dims()))
        state = (None if self.rule.is_trivial
                 else self.rule.model_input_state(
                     entry.name, axes, encoder=entry.encoder))
        return Carried(shape=Shape(out=axes), state=state,
                       origin=Reference.fixed(model_input=entry.name))

    def _handed(self, name: str, tensor: Any, scope: Scope) -> Carried:
        """An encoder's output as the language model receives it: `[B, rows,
        width]`, the rows one request's items hand over, laid out as an
        encoder's rows are — whole, the batch with the request."""
        axes = (scope.binding["B"], tensor.rows, tensor.width)
        state = (None if self.rule.is_trivial
                 else self.rule.model_input_state(name, axes, encoder=True))
        # Nothing this walk evaluates produced it.
        return Carried(shape=Shape(out=axes), state=state,
                       origin=Reference.fixed())

    def _eval_link(self, stack: Stack, stage: BlockStage, carry: Carry,
                   inbounds: Sequence[Carried], scope: Scope,
                   walk: Walk, *, prefix: Mapping[str, Carried],
                   joined: Mapping[str, Carried]) -> Optional[Carried]:
        """Evaluate one stage of a stack that carries a stream, and hand the
        stream on.

        The stages of such a stack form a chain in `Stack.walk_order`: the
        first enters with `inputs[0]`, each later one with its predecessor's
        exit. Each stage is still evaluated once and multiplied by its
        count, which is only the same as walking every iteration when every
        occurrence of a stage enters with the state it was evaluated at. So
        a stage that occurs more than once — repeated, interleaved with
        another variant, or inside a group that repeats the whole chain —
        must exit with the state it entered with (`_closed` holds it to that
        fixed point). Only a stage that occurs exactly once may change the
        state, and then no other stage may occur on both sides of it, or
        that stage's later occurrences would enter with a state it was
        never evaluated at.
        """
        if not inbounds:
            raise ValueError(
                f"{self.program.spec.name}: stack {stage.name!r} carries "
                f"stream {carry.stream!r} but names no `inputs:` to start "
                f"it from")
        first = carry.key not in walk.carried
        template = stage.template
        indices = template.layer_indices
        start = inbounds[0].origin
        entry = replace(
            walk.carried.get(carry.key, inbounds[0]),
            origin=CarryEntry(key=carry.key, iterations=indices, start=start,
                              model_input=(start.model_input if first
                                           else None)))
        # Occurs more than once: a count above one, a chain the enclosing
        # group repeats, or iterations with another variant between them.
        repeats = (template.count > 1
                   or template.outer_repeat > 1
                   or max(indices) - min(indices) + 1 != len(indices))
        out = self._eval_stage(
            stage, entry, scope, walk,
            inbounds=[entry, *inbounds[1:]],
            fixed_point=repeats, close=True,
            prefix=prefix,
            joined=joined)
        if out is None:
            return None
        walk.chain_exits.setdefault(carry.key, []).append(
            (indices, out.producer[0]))
        if not repeats and out.state != entry.state:
            other = stack.spanning(stage)
            if other is not None:
                (at,) = indices
                span = other.template.layer_indices
                raise NotImplementedError(
                    f"{self.program.spec.name}: stage {stage.name!r} "
                    f"(template {template.name!r}) changes the layout of "
                    f"stream {carry.stream!r} at iteration {at}, which lies "
                    f"inside the iterations {min(span)}..{max(span)} of "
                    f"template {other.template.name!r}; that template's "
                    f"occurrences after iteration {at} would enter with a "
                    f"state it was not evaluated at")
        walk.carried[carry.key] = out
        return out

    def _append(self, stages: Sequence[Stage],
                exits: Mapping[int, Optional[Carried]],
                produced_by: Mapping[str, Carried], walk: Walk) -> None:
        """Publish the streams a model layer appends to, once every stage
        it expanded into (`stages`) has walked. A stream with no elements
        is not published: nothing reads it, and a name that resolved would
        let a reader bind something else."""
        paths = {tuple(st.template.meta.layer_path)
                 for st in stages if isinstance(st, BlockStage)}
        for name, stream in self.program.ir.streams.items():
            if stream.layer_path not in paths:
                continue
            elements = self._elements(stages, exits, produced_by, name)
            if elements:
                walk.appended[name] = elements

    def _elements(self, stages: Sequence[Stage],
                  exits: Mapping[int, Optional[Carried]],
                  produced_by: Mapping[str, Carried], name: str,
                  count: Optional[int] = None) -> Tuple[Carried, ...]:
        """The first `count` elements of stream `name` (all of them when
        None), from the stages its model layer expanded into.

        Element j is what the stack's `from` held after iteration `t_j`:
        the exit of the stage whose iterations include `t_j` — for a
        carried stream, closed at the boundary — or, for -1, the stack's
        input as it received it. A stage is walked once, so every
        iteration of it exits with one tensor; the elements are told apart
        by the pairing tables, not here. The stack's own `read: all`
        readers ask for a prefix mid-chain, and a prefix is only ever
        appended by iterations the chain has already walked
        (`AppendedStream.visible`); an element whose stage has not walked
        is a broken invariant, not a value to guess."""
        stream = self.program.ir.streams[name]
        first = stages[0]
        elements: List[Carried] = []
        for t in stream.iterations[:count]:
            if t < 0:
                elements.append(produced_by[first.upstream[0]])
                continue
            (stage,) = [st for st in stages
                        if t in st.template.layer_indices]
            if id(stage) not in exits:
                raise ValueError(
                    f"{self.program.spec.name}: element of stream {name!r} "
                    f"after iteration {t} is read before stage "
                    f"{stage.name!r} (template {stage.template.name!r}) "
                    f"walked")
            value = exits[id(stage)]
            if value is None:
                raise ValueError(
                    f"{self.program.spec.name}: {first.name!r} appends "
                    f"to stream {name!r} after iteration {t}, which "
                    f"produced no tensor")
            uid, port = value.producer
            elements.append(replace(value, origin=Reference.fixed(
                (Source(Occurrence(uid, t), port, "element"),))))
        return tuple(elements)

    def _prefix(self, stage: Stage, stages: Sequence[Stage],
                exits: Mapping[int, Optional[Carried]],
                produced_by: Mapping[str, Carried],
                scope: Scope) -> Dict[str, Carried]:
        """The `read: all` values `stage` reads: for each stream its
        template reads the prefix of, the m elements every layer it stands
        for sees (`BlockTemplate.visible`), as one history."""
        if not isinstance(stage, BlockStage):
            return {}
        return {name: self._history(
                    self._elements(stages, exits, produced_by, name, m),
                    name, stage, scope)
                for name, m in stage.template.visible.items()}

    def _refuse_view_selecting_an_iteration(
            self, stage: OpStage, inbounds: Sequence[Carried],
            out: Optional[Carried], walk: Walk) -> None:
        """A model-level view of a stack's name that stands for an earlier
        iteration.

        A stack that carries a stream is read by its name as its final value:
        the carried stream after its last iteration, a tensor with no
        iteration axis. A view op (`slice`, `identity`, …) of that value that
        passes it on unchanged, or that declares an `in:` extent the axis it
        narrows does not have, selects nothing from it; what it was written
        to stand for — `hidden_states[i]` of an earlier layer — is an
        appended stream. Refused here, where the extents are known; a real
        narrowing of the final value (a CLS crop along the sequence) is not.
        """
        op = OpRef.resolve(stage.node.op, self.program, axis=stage.node.axis)
        if op is None or not op.is_view_op or out is None or out.shape is None:
            return
        # A view that narrows to an extent the engine supplies (`sample`'s
        # `s_sampled`, spliced before every output head) is the analysis's
        # own view, not a selection anyone wrote.
        narrows_to = op.spec.out_key and dim_registry.get(op.spec.out_key)
        if narrows_to and narrows_to.engine_supplied:
            return
        carried = {reads.name: reads.carried
                   for reads in self.program.ir.stack_reads.values()
                   if reads.carried is not None}
        for name, value in zip(stage.upstream, inbounds, strict=True):
            if name not in carried or value is None or value.shape is None:
                continue
            arriving = tuple(value.shape.out or ())
            unchanged = (tuple(map(expr_of, out.shape.out or ()))
                         == tuple(map(expr_of, arriving)))
            declared = walk.node_dims.get(stage.node.uid, {}).get("in")
            narrowed = arriving[-1 if stage.node.axis is None
                                else stage.node.axis] if arriving else None
            misdeclared = (op.shape_kind == "axis_replace"
                           and "in" in stage.node.dims_raw
                           and narrowed is not None
                           and value_of(declared) != value_of(narrowed))
            if not (unchanged or misdeclared):
                continue
            what = ("hands its value on unchanged" if unchanged else
                    f"declares `in: {stage.node.dims_raw['in']}` "
                    f"({value_of(declared)}), but the axis it narrows is "
                    f"{expr_of(narrowed)} ({value_of(narrowed)})")
            raise ValueError(
                f"{self.program.spec.name}: model layer {stage.name!r} "
                f"(`{stage.node.op}`) reads stack {name!r} by name and "
                f"{what}. A stack's name reads its final value, the carried "
                f"stream {carried[name]!r} after its last iteration, which "
                f"has no iteration axis to select from. To read the value "
                f"after chosen iterations, declare an appended stream on "
                f"{name!r} — `<stream>: {{ from: {carried[name]}, after: "
                f"[<i>, …], index: hidden_states | block, src: \"<forward "
                f"line>\" }}` (HF `hidden_states[-2]` is `after: [-2], index: "
                f"hidden_states`) — and name `<stream>` in the reader's "
                f"`inputs:` (with `axis: -1` to join several elements); see "
                f"the schema section \"Appended streams: values a stack hands "
                f"on after chosen iterations\". To read the final value, name "
                f"{name!r} in the reader's `inputs:` and drop "
                f"{stage.name!r}.")

    def _final_inner_streams(self, stage: Stage,
                             walk: Walk) -> Dict[str, Carried]:
        """The inner streams a model-level op names in `inputs:`, each
        bound to its final value: the write of the last layer of its stack
        that writes it (`Program.final_inner_streams`)."""
        return {name: replace(
                    walk.stream_values[final.uid],
                    origin=Reference.fixed((Source(final, "out", "named"),)))
                for name, final in self.program.final_inner_streams(stage).items()}

    def _appended_input(self, stage: Stage, name: str,
                        walk: Walk, scope: Scope) -> Carried:
        """What a stage that names appended stream `name` in `inputs:`
        binds: a repeated stack's iteration its paired element, a
        model-level op the whole stream."""
        if isinstance(stage, BlockStage):
            return self._element(stage, name, walk)
        return self._history(walk.appended[name], name, stage, scope)

    def _joined(self, stage: Stage, walk: Walk) -> Dict[str, Carried]:
        """The elements the joins in `stage` read, by the names its lowered
        graph gives them (`Program.joined_elements`), and a stream of one
        element by its own name as well: that read is direct.

        A join takes the k elements in order and requires what `concat`
        requires of them: one rank, and every axis but the joined one
        equal. The walk has already published every element (a reader's
        stack follows the writer's), so an element that is missing is a
        broken invariant, not a value to guess."""
        reads = self.program.concat_reads(stage)
        if not reads:
            return {}
        for _uid, name, axis in reads:
            elements = walk.appended[name]
            ranks = {len(e.shape.out) for e in elements}
            if len(ranks) != 1:
                raise ValueError(
                    f"{self.program.spec.name}: stage {stage.name!r} joins "
                    f"stream {name!r}, whose elements have ranks "
                    f"{sorted(ranks)}; a concat joins tensors of one rank")
            (rank,) = ranks
            if not -rank <= axis < rank:
                raise ValueError(
                    f"{self.program.spec.name}: stage {stage.name!r} joins "
                    f"stream {name!r} along axis {axis}, outside "
                    f"its elements' rank {rank}")
            at = axis % rank
            first = elements[0].shape.out
            crop = self.program.ir.streams[name].slice
            if crop is not None:
                crop_axis, crop_start = crop
            if crop is not None:
                def uncropped(at, name=name, crop_axis=crop_axis,
                              crop_start=crop_start, elements=elements):
                    shapes = [[at(value_of(d)) for d in e.shape.out]
                              for e in elements]
                    return ValueError(
                        f"{self.program.spec.name}: stream {name!r} "
                        f"publishes each element from {crop_start} on along "
                        f"axis {crop_axis}, which its elements {shapes} do "
                        f"not hold")

                if not -rank <= crop_axis < rank:
                    raise uncropped(lambda expr: expr)
                # Each element must reach past the crop: a condition on the
                # point where its extent grows with the workload.
                for element in elements:
                    require(sp.Gt(value_of(element.shape.out[crop_axis]),
                                  crop_start), uncropped)
            for j, element in enumerate(elements[1:], 1):
                apart = [i for i in range(rank) if i != at and value_of(
                    element.shape.out[i]) != value_of(first[i])]
                if apart:
                    raise ValueError(
                        f"{self.program.spec.name}: stage {stage.name!r} "
                        f"joins stream {name!r} along axis {axis}, "
                        f"but element {j} {list(element.shape.out)} differs "
                        f"from element 0 {list(first)} on axis {apart[0]}; "
                        f"a concat needs every other axis equal")
        return {ref: replace(walk.appended[name][j],
                             origin=walk.appended[name][j].origin.via("join"))
                for ref, (name, j)
                in self.program.joined_elements(stage).items()}

    def _history(self, elements: Sequence[Carried], name: str, stage: Stage,
                 scope: Scope) -> Carried:
        """Elements `[B, S, D]` read at once, as `[B, S, m, D]`.

        One tensor per element would be one value per iteration; a reader
        takes them together, and the history axis is bound as
        `HISTORY_AXIS` = m — the derived group's own binding, so the axis
        carries the symbol its group binds. The elements are one tensor
        repeated, as every entry of a closed carried stream is: a history
        of differently shaped or placed entries has no single layout, and
        is refused rather than laid out as its first."""
        first = elements[0]
        if any(e.shape.out != first.shape.out or e.state != first.state
               for e in elements[1:]):
            raise NotImplementedError(
                f"{self.program.spec.name}: stage {stage.name!r} reads "
                f"{len(elements)} elements of stream {name!r} that differ "
                f"in shape or layout")
        axis = scope.child_map(
            {HISTORY_AXIS: len(elements)}).binding[HISTORY_AXIS]
        entry = first.shape.out
        state = (None if self.rule.is_trivial
                 else self.rule.history_state(first.state))
        return Carried(shape=Shape(out=(*entry[:-1], axis, entry[-1])),
                       state=state, origin=Reference.fixed(tuple(
                           found for e in elements
                           for found in e.origin.via("history").at(None))))

    def _element(self, stage: Stage, name: str, walk: Walk) -> Carried:
        """The element of appended stream `name` that `stage` reads.

        `stage` is evaluated once for all its iterations, and each reads the
        element `Program.paired_element` pairs it with, which the read's
        reference keeps per iteration. One evaluation stands for
        them all only when those elements are one tensor, as they are when
        the appending stack is one template.
        """
        elements = walk.appended[name]
        indices = stage.template.layer_indices
        values = [elements[self.program.paired_element(name, stage, i)]
                  for i in indices]
        if any(value != values[0] for value in values[1:]):
            raise NotImplementedError(
                f"{self.program.spec.name}: stage {stage.name!r} is "
                f"evaluated once but reads elements of stream {name!r} that "
                f"differ in shape or layout")
        return replace(values[0], origin=Reference(table=tuple(
            (i, value.origin.via("element").at(None))
            for i, value in zip(indices, values, strict=True))))

    def _isolated_stream(self, stage: Stage, name: str,
                         scope: Scope, walk: Walk) -> Carried:
        """A stream read inside an isolated component (`ResolvedSpec.site`
        set: one component run as a one-stage model). Its producer is
        outside the isolation by construction, so the read takes the shape
        the model's stack declares for the stream, under this stage's dims,
        and the cache law the stream carries where the model hands it to
        this call (`CallSite.stream_laws`), stated at this walk's workload.
        In a whole model the producer's tensor is the edge and a missing one
        stays a missing input."""
        shapes = (dict(stage.template.meta.reads.shapes)
                  if isinstance(stage, BlockStage) else {})
        axes = shapes.get(name)
        if not axes:
            raise ValueError(
                f"{self.program.spec.name}: component reads stream {name!r}, "
                f"whose producer is outside this isolated evaluation, and "
                f"the stack declaring it gives no `shape:`. Declare the "
                f"stream's shape on that stack's `streams:` entry "
                f"(`{name}: {{ shape: [B, ...] }}`) so the component can be "
                f"evaluated alone.")
        laws = self.program.resolved.site.stream_laws
        if name not in laws:
            # The model publishes the stream to no layer of this call (a
            # component taking the place of one that does not read it): the
            # tensor is the declared shape, and no law comes with it.
            walk.unpublished.add(name)
        flat = scope.flat_dims()
        return Carried(shape=Shape(out=tuple(
            eval_dim_expr(str(axis), flat) for axis in axes)),
            origin=Reference.fixed(),
            kv_length=(None if laws.get(name) is None
                       else self._handed_length(laws[name], flat)))

    def _handed_length(self, law: StreamLaw,
                       flat: Mapping[str, Any]) -> KVLength:
        """The length a stream carries into an isolated call: its law
        (`StreamLaw`) stated by the publishing node's op as the walk states
        it at that node (`_stated_length`) and published as it publishes it
        (`_published_length`), at this walk's workload under the publisher's
        dims."""
        if law.op is None:
            raise ValueError(
                f"{self.program.spec.name}: stream {law.stream!r} carries the "
                f"summed length of a union of KV streams ({law.node}); no "
                f"one node states it, so an isolated call cannot be handed "
                f"its law. Evaluate the model whole")
        op = OpRef.resolve(law.op, self.program)
        dims = {**flat, **{name: eval_dim_expr(str(expr), flat)
                           for name, expr in law.dims}}
        positions = op.kv_positions(dims, self._expr)
        symbolic = op.kv_positions({**dims, "s_kv": CACHE_LEN}, self._expr)
        if positions is None or positions.read is None:
            raise ValueError(
                f"{self.program.spec.name}: stream {law.stream!r} carries the "
                f"law of {law.node} ({law.op}), whose op states no length at "
                f"this walk's workload")
        return self._published_length(
            self._stated_length(None, op, positions,
                                symbolic.read if symbolic.read is not None
                                else positions.read, encoder=False),
            law.cached)

    def _publish_stream_outs(self, stream_out, nodes, walk: Walk,
                             cached: Sequence[str] = ()) -> None:
        """Publish each `stream_out` node's tensor as the value it produced.

        If that node is also a `cache.entries` slot, the stream carries the
        cache law (`dictates=True`) so its readers adopt it as `s_kv`; so
        does a node whose output already dictates its length (a KV stream
        marker: `kv_cache_view`, `token_pool_compressor`), as a `src` edge
        out of it would. Storage stays on the writer; readers do not declare
        the cache. A reader seeded before this producer walked is held to
        what it was seeded with (`_stream_value`).
        """
        if not stream_out:
            return
        cached_set = set(cached)
        found = {n.name: n for n in nodes or ()}
        for sname, target in stream_out:
            node = found.get(str(target))
            if node is None:
                continue
            cached = str(target) in cached_set
            kv = self._published_length(walk.kv_lengths.get(node.uid),
                                        cached)
            if kv is not None:
                walk.stream_laws[node.uid] = (kv.stated_by, cached)
            value = Carried(
                shape=walk.shapes.get(node.uid),
                state=walk.state_at(node.uid),
                kv_length=kv,
                producer=(node.uid, "out"),
            )
            walk.stream_values[node.uid] = value
            for producers, seeded, reader in walk.stream_seeds.get(sname, ()):
                if node.uid in producers:
                    self._same_stream_value(sname, reader, seeded, value)

    @staticmethod
    def _published_length(length: Optional[KVLength],
                          cached: bool) -> Optional[KVLength]:
        """What a `stream_out` of a node whose output carries `length`
        publishes with the tensor: a cache entry's law, which its readers
        adopt as their `s_kv`; a length the node's output already dictates
        (a KV stream marker's, handed on through views); nothing else."""
        if length is not None and cached:
            return replace(length, dictates=True)
        if length is not None and length.dictates:
            return length
        return None

    def _stream_value(self, stage: BlockStage, name: str,
                      producers: Sequence[int], walk: Walk) -> Carried:
        """The inner stream value a template reads: what the producers its
        layers' `reads_from` resolve to published (`Program.block_stream_reads`).

        The template is evaluated once for all its layers, so one value
        stands for every producer they read: the producers that have walked
        must agree, and one that walks later is held to it when it
        publishes. The first of them in layer order precedes the template's
        first layer, so it has walked."""
        values = [walk.stream_values[uid] for uid in producers
                  if uid in walk.stream_values]
        if not values:
            raise ValueError(
                f"{stage.name}: stream {name!r} has no producer "
                f"tensor yet")
        for value in values[1:]:
            self._same_stream_value(name, stage.template.name, values[0],
                                    value)
        walk.stream_seeds.setdefault(name, []).append(
            (frozenset(producers), values[0], stage.template.name))
        return replace(values[0], origin=self.program.inner_stream_reference(
            stage.template, name))

    def _same_stream_value(self, name: str, reader: str, seeded: Carried,
                           value: Carried) -> None:
        # What a reader reads is the tensor: its shape, layout and cache
        # law. The shapes of the producer's own inputs are the producer's.
        def read(carried: Carried):
            out = carried.shape.out if carried.shape is not None else None
            return (out, carried.state, carried.kv_length)
        if read(value) == read(seeded):
            return
        raise SpecRefusal(
            f"{self.program.spec.name}: template {reader!r} reads stream "
            f"{name!r} from layers whose values differ — "
            f"{seeded.shape} {seeded.state} {seeded.kv_length} against "
            f"{value.shape} {value.state} {value.kv_length}; a template is "
            f"evaluated once for all its layers, so they read one shape, "
            f"layout and cache law",
            declaration="reads_from",
            fix=("give the layers that read differing sources templates of "
                 "their own: one template per source shape. Two assignments "
                 "of one template hold its weights twice, so each group "
                 "binds its own adapter template over the shared body"))

    def _eval_stage(self, stage: Stage, entry: Optional[Carried],
                    scope: Scope, walk: Walk, *,
                    inbounds: Sequence[Carried] = (),
                    fixed_point: bool = False, close: bool = False,
                    prefix: Mapping[str, Carried],
                    joined: Mapping[str, Carried],
                    ) -> Optional[Carried]:
        """Evaluate one stage and return its output as it is handed on.

        `inbounds` bind the stage's formals in order; `entry` is the one on
        formal 0. `joined` binds the elements the stage's joins read
        (`_joined`); like a stream, it reaches into every component. A
        block stage's output is settled at its exit when `close` (the
        stream its stack carries), and then held to `entry`'s state when
        `fixed_point`. A top-level op stage has no
        interface: its exit only finalizes pending values (the vocabulary
        embedding's reduction).
        """
        if isinstance(stage, BlockStage):
            template = stage.template
            # Per-template dim overrides. A hybrid model rebinds dims per block
            # VARIANT — `n_kv_heads: n_kv_heads_swa` on the sliding-window
            # assignment, a different head count on gemma-4's local layers —
            # and `BlockMeta.dim_overrides` is where the assignment resolution
            # put them. Not applying them evaluates every variant against the
            # backbone's dims, which is one mechanism behind three models'
            # disagreements: gemma-4 at half the heads and four times the KV
            # heads, Step-3.7 at 1.5x, MiMo-V2.5's fused QKV short by its SWA
            # KV share.
            scope = scope.frame(template.dim_overrides,
                                template.meta.dim_overrides_raw)
            walk.stage_frames[self.program.block_index(template)] = \
                scope.binding
            # Streams: the producer tensor is the edge. An inner stream is
            # read from the producers the template's `reads_from` resolve to;
            # a reader whose producer has not walked yet is a missing input,
            # not a yaml shape pretending to be live. The frame holds only
            # the streams this stage can see; a carried stream is not among
            # them — it arrives on formal 0.
            inner = self.program.block_stream_reads(
                self.program.block_index(template))
            _, needed = self.program.component_stream_roles(template)
            if inner or needed or joined:
                seeded: Dict[str, Carried] = dict(joined)
                for sname, producers in inner.items():
                    seeded[sname] = self._stream_value(
                        stage, sname, producers, walk)
                unread = []
                for sname in needed:
                    if sname in seeded:
                        continue
                    if sname in prefix:
                        seeded[sname] = prefix[sname]
                        continue
                    if sname in template.meta.reads.each:
                        seeded[sname] = self._element(stage, sname, walk)
                        walk.stage_elements.setdefault(
                            self.program.block_index(template), {})[sname] = \
                            tuple(seeded[sname].shape.out)
                        continue
                    if self.program.resolved.site is not None:
                        seeded[sname] = self._isolated_stream(
                            stage, sname, scope, walk)
                    else:
                        unread.append(sname)
                if unread:
                    raise ValueError(
                        f"{stage.name}: stream {unread[0]!r} has no "
                        f"producer tensor yet")
                scope = replace(scope, streams=seeded)
            names = template.input_names()
            # The stage's `inputs:` bind the body's formals one to one, in
            # order; `ArchSpec._check_stack_call_arity` refused any other
            # count at load, so a mismatch here is a broken invariant, not a
            # stage to bind partially.
            aliases = dict(zip(names, inbounds, strict=True)) if inbounds else {}
            walk.stage_inputs[self.program.block_index(template)] = tuple(
                tuple(value.shape.out) for value in inbounds)
            walk.stage_references[self.program.block_index(template)] = tuple(
                value.origin for value in inbounds)
            inner = scope.entering(aliases) if aliases else scope
            # The template is evaluated ONCE and its result multiplied by the
            # number of layers using it, so every node beneath here exists that
            # many times. Seeding the frame is the same statement `total()`
            # makes about cost, made once for everything derived instead.
            inner = inner.repeated(template.count, template.count)
        else:
            inner = scope
            if stage.upstream and inbounds:
                # A stream the op joins is read through its join, which
                # binds the elements by their own names; the stream's name
                # has a value only when it is one element, read directly.
                inner = replace(inner, aliases={
                    **dict(inner.aliases),
                    **{name: value for name, value in zip(
                        stage.upstream, inbounds, strict=True)
                       if value is not None},
                })
            if joined:
                inner = replace(inner, streams={**inner.streams, **joined})
        for node in stage.roots:
            self._eval(node, inner, walk)
        if isinstance(stage, BlockStage):
            self._publish_stream_outs(
                stage.template.meta.stream_out, stage.template.roots, walk,
                cached=stage.template.meta.kv_cached)
        output_roots = (
            stage.template.output_roots()
            if isinstance(stage, BlockStage) else stage.roots)
        produced = [n for n in output_roots
                    if n.uid in walk.shapes and walk.shapes[n.uid].out]
        if not produced:
            return None
        last = produced[-1]
        out = Carried(shape=walk.shapes[last.uid],
                      state=walk.state_at(last.uid),
                      producer=(last.uid, "out"))
        if isinstance(stage, OpStage):
            if stage.node.boundary_role == "output_head":
                return out
            return self._closed(stage, out, last, walk)
        if not close:
            return out
        interface = entry.state if fixed_point else None
        return self._closed(stage, out, last, walk, interface=interface)

    def _closed(self, stage: Stage, out: Carried, last: Evaluable,
                walk: Walk, *, interface: Any = None) -> Carried:
        """A stage's output settles its pending layout at the stage boundary.

        For a block stage this is its stack's carried stream: the main
        activation, which a tensor-parallel exit leaves partial. The
        inner streams carry their producer's state as is.

        A block template is evaluated ONCE and multiplied, so the edge from its
        output into the next layer's input is not in the graph — and that is
        where an FFN's row-parallel partial gets reduced. Without closing it the
        attention exit's reduction is found, because its consumer is inside the
        same block, and the FFN exit's is not: half the communication of every
        dense tensor-parallel model.
        """
        rule = self.rule
        if rule.is_trivial or out.state is None:
            return out
        # A repeated stage supplies both sides of its interface directly.
        # The uid is attribution for a loud fixed-point failure; it is not
        # used as a proxy for the entering state.
        stage_uid = (stage.node.uid if isinstance(stage, OpStage)
                     else stage.roots[0].uid)
        closure = rule.close_stage(
            out.state, interface=interface, stage_uid=stage_uid)
        if closure is None:
            return out
        settled, moved = closure
        walk.redistributes.append(transition_mod.Redistribute(
            node=last.uid, input_index=-1, producer=last.uid,
            arrived=out.state, required=settled, collectives=moved,
            site=transition_mod.RedistributeSite.STAGE_CLOSE))
        closed = Carried(shape=out.shape, state=settled, producer=out.producer)
        walk.handed_on[out.producer] = closed
        return closed

    def _tokens_per_expert(self, node: Evaluable, op: OpRef, binding: Binding,
                           batch: Any, seq: Any) -> Any:
        """`M = B·s_ctx·n_experts_active·dp_attention/n_route_targets`, the
        per-expert effective batch. `B` and `s_ctx` are the token tensor that
        arrived at `node` (a `permute`); the routing counts stay on the
        binding.

        Under DP-attention with expert parallelism each attention replica runs
        its own `B·s_ctx` tokens, then the expert all-to-all pools tokens from
        every replica onto every rank's experts, so a rank's experts see
        `B·s_ctx·dp_attention` tokens. Dropping `dp_attention` under-counts
        MoE compute by exactly that factor. Unbound, it is 1 — no
        DP-attention — and bound to anything but a positive integer it is
        refused.

        EXACTLY, because `M` is the AVERAGE tokens per expert and the bank's
        total is `n_experts x M`: any rounding here multiplies by `n_experts`.
        A ceiling overstates the total whenever `n_route_targets` does not
        divide `B·s_ctx·n_experts_active·dp_attention` (31% on Kimi-K3 at B=1,
        s_ctx=128), and a `max(1, .)` floor bills every expert one token at
        small decode batches where most experts receive none. A fractional
        `M` is the honest per-copy average; a consumer that needs the padded
        integer geometry of a real grouped kernel (kernel_bench) applies its
        own kernel's padding convention at its own boundary.

        The denominator is `n_route_targets` — how many destinations the
        top-k chooses among — and not `n_experts`, which counts the ones that
        hold weights. They differ by LongCat-Flash's 256 `nn.Identity()`
        experts. Read through `raw` on purpose: expert parallelism shards the
        copies axis `n_expert_copies`, never the denominator, and reading it
        sharded here would inflate `M` by the EP degree.

        `sympy.Rational`, not float: exactness is the point, and it
        normalises to a plain Integer whenever `e` divides. A bare
        `Symbol("M")` kept the derivation out of the expression: 660 of
        10,872 shapes then disagreed with the numeric walk.
        """
        def _raw(name: str) -> Any:
            try:
                return binding.raw(name)
            except KeyError:
                return None

        # The routing counts are the ones the op declares it reads off its
        # binding (`OpSpec.binding_dims`), which a static reader of the graph
        # sees too; a name missing from that declaration fails here loudly.
        routing = {name: _raw(name) for name in op.binding_dims}
        k, e = routing["n_experts_active"], routing["n_route_targets"]
        g = _raw("dp_attention")
        if g is None:
            g = 1
        elif (degree := algebra(value_of(g))).is_number and not (
                degree.is_integer and degree >= 1):
            raise ValueError(
                f"{node.display_path} ({node.op}): `dp_attention` is bound to "
                f"{value_of(g)!r}; it is a positive integer degree")
        missing = [name for name, value in (
            ("B", batch), ("s_ctx", seq), ("n_experts_active", k),
            ("n_route_targets", e)) if value is None]
        if missing:
            raise ValueError(
                f"{node.display_path} ({node.op}): the tokens per expert need "
                f"{missing}, which neither the arriving token tensor nor the "
                f"routing binding resolves")
        if not value_of(e):
            raise ValueError(
                f"{node.display_path} ({node.op}): `n_route_targets` is "
                f"{value_of(e)!r}; a router chooses among at least one "
                f"destination")
        values = [value_of(v) for v in (batch, seq, k, g, e)]
        if all(isinstance(v, int) for v in values):
            bv, sv, kv, gv, ev = values
            exact: Any = sp.Rational(bv * sv * kv * gv, ev)
            if any(isinstance(v, Dim) for v in (batch, seq, k, g, e)):
                return Dim(
                    (expr_of(k) * expr_of(g) * expr_of(batch) * expr_of(seq))
                    / expr_of(e), exact)
            return exact
        return (k * g * batch * seq) / e

    # -- the walk --------------------------------------------------------

    def _eval(self, node: Evaluable, scope: Scope, walk: Walk) -> None:
        """Evaluate `node` and everything under it, writing into the layer maps.

        The dispatch zone's routing ratio is decided here and spans the whole
        subtree: a node inside the zone and every component it recurses into
        run `n_experts_active / n_route_targets` of their stored copies per
        token. It is a child frame, and the parent frame is untouched by
        construction.

        The node's copy count is decided here too, for the same reason and once:
        a `repeat:` scales this node's cost AND everything under it, so it is a
        property of the frame the subtree is evaluated in rather than something
        each level re-reads off the IR.
        """
        inner = scope
        if node.in_dispatch_zone:
            # No narrowing: the tensors in the zone are `[n_expert_copies, M,
            # ·]` as the dispatch wrote them, so every node reads the token
            # axes off what arrives. What the zone does decide for
            # everything below is the routing ratio — `n_experts_active` of
            # `n_route_targets` stored copies run per token — read raw: the
            # denominator is global, and expert parallelism shards the copies
            # axis, never it. It counts DESTINATIONS, so a bank whose targets
            # include zero-compute experts runs the weighted fraction of its
            # copies per token. Both are bound: the zone's `permute` already
            # refused a bank without them (`_tokens_per_expert`).
            inner = inner.routed(inner.binding.get("n_experts_active"),
                                 inner.binding.raw("n_route_targets"))

        own = self._copies(node, inner)
        stored, active = own.stored, own.active
        inner = inner.repeated(stored, active)
        walk.instances[node.uid] = inner.copies[0]
        # The copies that run, as a SHARE of the copies held (`Scope.copies`
        # is held-then-running). A leaf that also runs only part of each copy
        # multiplies its own share in below.
        walk.active_share[node.uid] = _share(inner.copies[1], inner.copies[0])
        walk.own_copies[node.uid] = own

        if node.is_component or node.children:
            cost = self._eval_component(node, inner, walk)
        else:
            try:
                cost, shape = self._eval_primitive(node, inner, walk)
            except KeyError as exc:
                if node.uid not in self._encoders:
                    raise
                raise self._unbound_in_encoder(
                    node, {str(exc.args[0])}) from exc
            if node.uid in self._encoders:
                unbound = self._free_names(
                    (*shape.out, cost.flops), inner.binding)
                if unbound:
                    raise self._unbound_in_encoder(node, unbound)
            walk.shapes[node.uid] = shape
        # One place the multiplier is applied, for both kinds of node. Only the
        # node's OWN count: a component's cost is its children's, and each of
        # them already carries theirs.
        walk.costs[node.uid] = (
            cost if stored == 1 and active == 1
            else cost.scaled(storage=stored, active=active)
        )

    def _eval_component(self, node: Evaluable, scope: Scope,
                        walk: Walk) -> Cost:
        # A call site's `dims:` are already resolved on the IR node, so this is a
        # child frame over values — no expression evaluation, no `_resolve_dim`.
        own = self._node_dims_override(node, scope)
        inner = scope.frame(own, node.dims_raw)
        # The frame the component runs in, as a primitive's is recorded: the
        # enclosing binding with its own `dims:` over it. Its own dims alone
        # read against the model's global binding lose every rebinding
        # between the model and this call — a block assignment's
        # `n_experts: n_pool_experts` among them, and the expert axis and
        # denominator its frame derives again from it, which size the bank's
        # leading axis and its rows under this call.
        walk.node_dims[node.uid] = inner.flat_dims()
        inner = inner.calling(self._callee_aliases(node, scope, walk))
        for child in node.children:
            self._eval(child, inner, walk)
        self._record_cache_state(node, walk)
        self._publish_stream_outs(
            node.stream_out, node.children, walk, cached=node.kv_cached)
        self._carry_kv_length(node, walk)
        caller_shapes = self._input_shapes(
            node, scope, self._inputs(node, scope, walk),
        )
        # The streams its body reads inside a kernel no edge shows are
        # inputs of the call too, after its `in:` refs.
        refs = len(node.inputs_obj or ())
        declared = self.program.declared_reads(node.uid)
        walk.input_arity[node.uid] = refs + len(declared)
        for index, name in enumerate(declared, refs):
            # A name the body reads under `reads:` is a stream, whatever
            # alias of the caller's frame shares its spelling.
            found = scope.stream(name)
            if found is None and self.program.resolved.site is None:
                raise ValueError(
                    f"{node.display_path} reads stream {name!r} inside its "
                    f"body, which this stage does not see")
            # An isolated component's stream comes from outside the
            # isolation: nothing this walk produced it.
            walk.input_references[(node.uid, index)] = (
                found.origin if found is not None else Reference.fixed())
        def _output_shape(port: str):
            producer, producer_port = self.program.output_leaf(node, port)
            inner_shape = walk.shapes.get(producer.uid)
            if inner_shape is None:
                return None
            return (
                inner_shape.out
                if producer_port == "out"
                else inner_shape.ports.get(producer_port)
            )

        secondary = tuple(port for port, _ref in node.secondary_outputs)
        main = _output_shape("out")
        if main is not None:
            shape = Shape(out=main,
                          ins=tuple(tuple(s) for s in caller_shapes))
            for port in secondary:
                dims = _output_shape(port)
                if dims is not None:
                    shape = shape.with_port(port, dims)
            walk.shapes[node.uid] = shape

        output_ports = ("out", *secondary)
        for port in output_ports:
            producer, producer_port = self.program.output_leaf(node, port)
            inner_state = walk.state_at(producer.uid, port=producer_port)
            if inner_state is not None:
                walk.set_state(node.uid, inner_state, port=port)
        return Cost.combine([walk.costs[c.uid] for c in node.children
                             if c.uid in walk.costs])

    def _record_cache_state(self, node: Evaluable, walk: Walk) -> None:
        """A recurrent component's states, each resolved in the frame of the
        node that carries it.

        `cache.entries` states declare their axes — a linear attention or
        SSM holds fixed tensors per request rather than a row per token — and
        each names its `carrier`, the kernel that reads and rewrites it. The
        carrier's frame is the component's (the call site's `dims:` already
        applied) plus the carrier's own `dims:`, so a conv state written
        `[C, kernel_size]` reads the conv's own channel count.
        Resolving from the root binding instead would read the model's
        `d_head` where the call site rebound it, and a state size is exactly
        the kind of number that stays plausible when it is wrong.

        Recorded on the component's uid, as `{state name: axes}` in
        declaration order, which is where `eval.cache` looks for it: the
        states belong to the component, and no node's output IS a state.
        """
        states = node.kv_states
        if not states:
            return
        by_name = {child.name: child for child in node.children}
        walk.cache_state[node.uid] = {
            entry.state: tuple(
                self._expr(axis, walk.node_dims[by_name[entry.carrier].uid])
                for axis in entry.shape)
            for entry in states
        }
        walk.cache_state_reads[node.uid] = {
            entry.state: self._expr(
                entry.reads, walk.node_dims[by_name[entry.carrier].uid])
            for entry in states if entry.reads is not None
        }

    def _callee_aliases(self, node: Evaluable, scope: Scope,
                        walk: Walk) -> Dict[str, Carried]:
        """Bind the component body's `inputs:` names to the caller's tensors.

        The body's unresolved refs (`x`, `residual`) name these, and they name
        the CALLER's tensor — not whatever the previous sibling produced. That
        distinction is the whole reason this exists, and it is why placement
        rides this walk instead of repeating it: a second traversal resolved a
        residual `x` to the preceding attention, so the add saw two partials
        where one operand is the replicated hidden state.

        Shape and state together — a component's cost comes from its children,
        but both of those cross the boundary.
        """
        out: Dict[str, Carried] = {}
        # One to one: a component's call sites are held to its formals at
        # load (`ArchSpec._check_component_call_arity` for graph nodes,
        # `_check_model_op_arity` for model-level `op:` layers).
        for name, ref, carried in zip(node.formals, node.inputs_obj or (),
                                      self._inputs(node, scope, walk),
                                      strict=True):
            if ref.src is not None and ref.port and carried.shape is not None:
                # A port edge hands the formal that output's tensor.
                selected = carried.shape.ports.get(ref.port)
                carried = replace(carried, shape=(
                    Shape(out=selected) if selected is not None else None))
            out[name] = carried
            # Expanded IR refs retain the caller-side actual name while the
            # component schema names the callee-side formal. Both identify the
            # same edge at this boundary, so expose both keys in the new frame.
            # The actual name wins a collision: an expanded child reads that
            # name, while the colliding formal belongs to a different input.
            # HC post is the canonical case -- formal ``x`` is the fresh
            # branch, but caller-side ``x`` names the rank-4 residual input.
            if ref.external_sid:
                out[ref.external_sid] = carried
        return out

    # -- primitives ------------------------------------------------------

    def _eval_primitive(self, node: Evaluable, scope: Scope,
                        walk: Walk) -> Tuple[Cost, Shape]:
        op = OpRef.resolve(node.op, self.program, axis=node.axis)
        if op is None:
            # Evaluation-time and LOUD, not a skip: a node the walk cannot
            # price is a subgraph silently absent from every layer, which is
            # the one failure no downstream instrumentation can see. The
            # schema stays permissive on purpose — a spec may be authored
            # before its op lands — and this is where the debt comes due.
            raise ValueError(
                f"unknown op {node.op!r} at uid {node.uid} — every node the "
                "walk reaches must resolve in OP_REGISTRY")

        own = self._node_dims_override(node, scope)
        if own:
            walk.node_dims[node.uid] = own
        dims = (scope.child_map(own).flat_dims() if own
                else scope.flat_dims())
        op.refuse_a_shape_it_cannot_store(node)
        if node.shape:
            # `shape:` is read the way `dims:` is (`_node_dims_override`):
            # from what the spec wrote, under this walk's binding. The
            # build-time integers kept a `parameter_tensor`'s
            # `n_hidden_copies` a 4 under the symbolic binding while the
            # activations it multiplies carried the symbol.
            dims["_shape"] = ([eval_dim_expr(s, dims) for s in node.shape_raw]
                              if node.shape_raw else list(node.shape))
        walk.node_dims[node.uid] = dict(dims)
        # The function constants the op takes (`OpSpec.consts`), as THIS
        # node's frame binds them: the spec's binding, rebound by any call
        # site or the node's own `dims:` on the way down — so a vision
        # tower's clamp reads the tower's bound. One the frame leaves
        # unbound is absent, and so is one this binding holds as a symbol
        # (the symbolic walk has no number to give).
        consts = {name: float(value) for name, value in (
            (name, value_of(dims[name])) for name in op.consts
            if name in dims)
            if isinstance(value, (int, float, sp.Number))}
        if consts:
            walk.consts[node.uid] = consts

        # Sparse attention: the op declares how many KV positions a query really
        # attends (`Min(s_kv, s_window)` for a sliding window,
        # `Min(s_kv, n_topk_keys)` for top-k), and
        # substituting it here makes shape, FLOPs and byte traffic all read the
        # same effective length. Without it a windowed layer reports touching the
        # whole cache.
        #
        # The same declaration is recorded with `s_kv` held symbolic, BEFORE the
        # substitution — a fn reading `s_kv` after it has been narrowed answers
        # `Min(Min(s_kv, s_window), s_window)` — because a capacity question
        # asks for the whole curve and this walk ran at one point on it.
        #
        # A KV stream that arrives already knowing its own length overrides
        # `s_kv` before any of that: `attention_score` reading the union of a
        # window and a compressed stream reads neither the context length nor
        # either operand's, and its own law — "whatever `s_kv` is" — has no way
        # to say so. A node that spells `s_kv` in its own `dims:` keeps what it
        # wrote, so an explicit binding still wins over an inherited one.
        carried = self._inputs(node, scope, walk)
        self.program.check_model_input_reader(
            node, walk.input_references, walk.input_arity)
        # A union sums its operands' lengths (`_record_kv_length`); it does
        # not adopt one of them.
        arriving = (None if op.kv_stream_union
                    else self._arriving_kv_length(node, walk, carried))
        # Whether this node's `s_kv` is narrower than the frame's: its
        # `s_pairs` is then its own (below).
        narrowed = False
        if arriving is not None and "s_kv" not in own:
            dims = {**dims, "s_kv": arriving.read}
            walk.node_dims[node.uid] = dict(dims)
            free = arriving.symbolic
            narrowed = True
        else:
            arriving = None
            free = CACHE_LEN
        # An encoder's attention reads the rows that arrive, one segment's
        # worth: its query and key lengths are the arriving tensors' (no step
        # binds them), and it keeps no cache for a later step.
        encoding = node.uid in self._encoders
        if encoding:
            arrived = [c.shape.out if c.shape is not None else ()
                       for c in carried]
            labels, _batch = bind_labels(op, arrived,
                                         require_consistent_labels=False)
            dims = {**dims, **{name: labels[name]
                               for name in ("s_query", "s_kv")
                               if name in labels and name not in own}}
            walk.node_dims[node.uid] = dict(dims)
        with node_context(node.display_path):
            positions = op.kv_positions(dims, self._expr)
            symbolic = (op.kv_positions({**dims, "s_kv": free}, self._expr)
                        if positions is not None else None)
        if positions is not None:
            if not encoding:
                walk.cache_positions[node.uid] = symbolic
            if positions.read is not None:
                # Two counts, two labels. `s_attend` is what one query
                # attends — the key axis of a windowed score matrix.
                # `s_kv` is what the step reads — the K / V operands' rows,
                # the union over the step's queries. They coincide for a
                # decode query and for dense attention; a prefill step over
                # a sliding window reads every key it reaches once. The
                # per-query count is recorded on the node too, so the
                # per-rank projection resolves the label the walk resolved.
                # `_law` returns value_of so CACHE_LEN still substitutes.
                # A dense law that did not narrow must keep the walk's
                # s_kv identity — otherwise attention_score folds 2*s_kv
                # into a coefficient while softmax/apply still carry s_kv,
                # and the fused formula no longer lines up with values.
                new_kv = (positions.step if positions.step is not None
                          else positions.read)
                prior_kv = dims.get("s_kv")
                if (isinstance(prior_kv, Dim)
                        and not isinstance(new_kv, Dim)
                        and same_law(prior_kv, new_kv)):
                    new_kv = prior_kv
                elif not isinstance(new_kv, Dim) and new_kv is not None:
                    new_kv = Dim.symbol("s_kv", new_kv)
                if new_kv is not prior_kv:
                    narrowed = True
                dims = {**dims, "s_kv": new_kv,
                        "s_attend": positions.read}
                walk.node_dims[node.uid]["s_attend"] = positions.read
        # The live (query, key) pairs of the matrix this node scores,
        # `s_query × s_kv` with `s_kv` as narrowed above. The phase binds the
        # dense causal count over the whole context (`binding.STEP_LAWS`);
        # this node's are its own where its op declares them — a banded
        # attention's own band (`cost_pairs`), a top-k selection, the stream
        # it adopted its length from (a window, a pooled stream, a union of
        # them, summed). Each is rebound in the node's frame with its
        # definition beside it (`node_rebinds`); the cost is written in
        # `s_pairs`. A node that narrowed its keys with no such declaration
        # has no pairs to price: its frame binds none, and an op that reads
        # them is refused (`PhaseBindingMissing`) rather than handed the
        # dense count of a matrix it does not score.
        stated = (positions if positions is not None
                  and positions.pairs is not None
                  else arriving if arriving is not None
                  and arriving.pairs is not None else None)
        selected = stated.named if stated is not None else None
        pairs = op.cost_pairs(dims)
        if pairs is None:
            pairs = selected
        # Pairs that come to the frame's own at every point ARE the frame's
        # `s_pairs`, and keep its identity, as a dense `s_kv` law above
        # keeps the walk's `s_kv`: a per-token cache read (`kv_cache_view`
        # at `compress_ratio: "1"`) states the dense causal triangle, and
        # rebinding it would print that triangle as a pooled law in the
        # ratio. One law, two spellings: their difference expands to zero.
        frame_pairs = dims.get("s_pairs")
        if (pairs is not None and frame_pairs is not None
                and same_law(sp.expand(algebra(value_of(frame_pairs))
                                       - algebra(pairs.value)), 0)):
            pairs = None
            narrowed = False
        if pairs is None and narrowed:
            dims = dict(dims)
            dims.pop("s_pairs", None)
            walk.node_dims[node.uid].pop("s_pairs", None)
        if pairs is not None:
            s_pairs = Dim(sp.Symbol("s_pairs"), pairs.value)
            dims = {**dims, "s_pairs": s_pairs}
            walk.node_dims[node.uid]["s_pairs"] = s_pairs
            walk.node_rebinds[node.uid] = {"s_kv": dims["s_kv"],
                                           "s_pairs": pairs}
        s_kv_law = _narrowed_s_kv(positions, op)
        if s_kv_law is not None:
            walk.node_rebinds.setdefault(node.uid, {})["s_kv"] = s_kv_law
        if stated is not None:
            # Pairs are a per-query index list where the op selects them
            # itself (a top-k attention; a stream marker only states its
            # stream's) or reads a union the source builds as the joined
            # index lists of its streams (DeepSeek-V4 `get_window_topk_idxs`
            # + `get_compress_topk_idxs`, vLLM `combine_topk_swa_indices`).
            # A reader of one stream's mask — an indexer scoring every
            # pooled key — scores a dense causal band of it, and streams it.
            # A view on the way re-indexes the list and reads nothing.
            # DSpark is the measured case, not a traffic-side exception:
            # `get_dspark_topk_idxs` expands the same window+block index
            # list for every draft query, and the TileLang kernel launches
            # one program per query and loads `kv[by, idxs[i]]` per listed
            # index (inference/model.py:1021-1029, inference/kernel.py:
            # 312-365). Whether such a reader streams or gathers is therefore
            # answered here, in `Analysis.gathered`; downstream pricing only
            # reads that fact.
            # Booleans throughout (`is`), never an extent's truth value.
            selects = (positions is not None and positions.pairs is not None
                       and op.kv_stream_marker is False)
            joined = arriving is not None and arriving.listed is True
            if op.is_view_op is False and (selects or joined) is True:
                walk.gathered.add(node.uid)
        self._record_kv_length(node, op, walk, arriving, positions)

        input_ports = self._binding_authoritative(
            self._primitive_input_shapes(node, carried),
            op, dims, scope.binding, reads_cache=positions is not None)
        in_shapes = [shape for shape in input_ports if shape]
        if not in_shapes and op.arity() != 0:
            # A source (`arity() == 0`) has nothing arriving. Any other node
            # reads what its inputs name: a node that names inputs and got no
            # shape for any of them is a broken edge, and one that names none
            # reads a tensor it never said which.
            if node.inputs_obj:
                raise ValueError(
                    f"{node.display_path} ({node.op}): declares "
                    f"{len(node.inputs_obj)} input(s) and none of them "
                    f"resolved to a shape")
            raise SpecRefusal(
                f"{node.display_path} ({node.op}) reads a tensor and does "
                f"not say which",
                declaration="in",
                fix="name what it reads in `inputs:` / `in:` — an earlier "
                    "layer, or a name in `model_inputs:`",
                node=node.display_path)
        violation = op.input_contract_violation(input_ports, dims)
        if violation is not None:
            # The op's contract is the engine's statement of what it accepts;
            # the tensors that arrive are the spec's choice of operands.
            raise SpecRefusal(
                f"{node.display_path} ({node.op}): {violation}",
                declaration="in",
                fix=f"give `{node.op}` operands of the shapes it accepts, "
                    "or the op the forward source applies to these tensors",
                node=node.display_path)
        # Shape inference consumes the live tensors.  FLOP signatures are
        # port-positional, so an absent earlier operand must remain an empty
        # slot rather than shifting every later tensor onto the wrong template.
        flop_inputs = input_ports
        if node.op == DISPATCH_OP:
            # `M` has one source: the token tensor that ARRIVED here, which
            # is not the workload's `B·s_ctx` once a slice, a window or a
            # stream has narrowed it on the way. It accepts `(B, s_ctx, d)`
            # and nothing else — a trailing-aligned leftover is not `B`.
            token = input_ports[0] if input_ports else None
            if not token or len(token) != 3:
                raise ValueError(
                    f"{node.display_path} ({node.op}): dispatches a "
                    f"`(B, s_ctx, d)` token tensor, and "
                    f"{list(token or ())!r} arrived")
            labels, _leftover = bind_labels(op, in_shapes)
            moe_m = self._tokens_per_expert(
                node, op, scope.binding, labels.get("B"), labels.get("s_ctx"))
            # `M` is the per-expert axis of the buffer this node writes,
            # `[n_expert_copies, M, d_model]`: the bank's copies outermost,
            # each expert's mean rows inside it. One law, bound once; every
            # node of the zone reads it off the tensor that arrives, and the
            # comm side reads it off this binding (`layout/moe.py`).
            dims = {**dims, "M": moe_m}
            walk.node_dims[node.uid] = dict(dims)
        tracked_dims = ReadTracking(dims)
        params = op.params(tracked_dims, self._expr)
        # An op owns parameters unless its parameter law is zero; a
        # parameter count never reads the workload (`same_law`).
        owns_params = not same_law(params, 0)
        if owns_params:
            # The substitution the params figure was evaluated with: the
            # dims the op's parameter projection actually read, identity →
            # value. Display binds this table; nothing downstream re-derives
            # it.
            params_record: Dict[Any, Any] = {}
            for name in sorted(tracked_dims.read):
                if name in dims:
                    record_binding(params_record, dims[name])
            # And every symbol of the parameter law at the value this
            # binding gives it, so the values line can name each factor
            # (`128 × (128 + 64)`) rather than the product a tracked axis
            # carried (`24576`). `bind_record` keeps the per-symbol line
            # only when it reproduces the figure; a rank whose shard is not
            # the binding's value falls back to the axis products.
            for symbol in sorted(algebra(expr_of(params)).free_symbols,
                                 key=str):
                value = scope.binding.get(symbol.name)
                if (symbol not in params_record and value is not None
                        and algebra(value_of(value)).is_number):
                    params_record[symbol] = value_of(value)
            if params_record:
                walk.params_bindings[node.uid] = params_record
        owns_weight = node.owns_weight
        reps: Tuple[Any, ...] = ()
        tensors: Tuple[Any, ...] = ()
        if owns_params and (owns_weight or node.stores_checkpoint_copy):
            # The on-disk name, composed the one way there is — asked of the
            # op, because the suffix it ends in is the op's own declaration.
            module = op.module_path(node, self.program.checkpoint_prefix(node))
            tensors = self.program.as_stored(
                node, op.stored_tensors(dims, params, module, self._expr))
            # The scheme's `in`/`out` come from the stored tensor, so they are
            # the weight's own dims by construction. Where the op owns nothing
            # resolvable there is nothing to bind, and the format falls back to
            # the default dtype rather than resolving against another tensor's
            # extents. Per TENSOR, not per node: a scheme that quantises a
            # weight leaves its bias at the default dtype, and asking once per
            # node cannot express that.
            reps = tuple(
                self._formats.resolve(t, node.role,
                                      node_dims=dims,
                                      stack=self.program.stack_of(node),
                                      components=node.enclosing_components,
                                      # The layers this node runs at. A
                                      # weight's dtype may be declared per
                                      # layer (`select.layers`), and the
                                      # template is walked once for all of
                                      # them, so the answer is per layer
                                      # segment rather than per node.
                                      layers=self.program.occurrence_layers(
                                          node.uid))
                for t in tensors)
            if reps:
                # The scheme's half of the name, applied HERE rather than at
                # `tensors()`. A tensor in `walk.storage` is then already
                # called what the checkpoint calls it, and reading that layer
                # directly — which two call sites and the name golden all did —
                # is no longer a way to get an incomplete answer.
                # Asked of the tensor, over all its layer segments: a
                # segment stores the same tensor another way, and the name
                # is the tensor's (`Representations.weight_suffix`).
                tensors = tuple(
                    replace(t, suffix=r.weight_suffix)
                    if t.suffix == ".weight" and r is not None
                    and r.weight_suffix != ".weight" else t
                    for t, r in zip(tensors, reps, strict=False))
            if owns_weight:
                if tensors:
                    walk.storage[node.uid] = tensors
                if reps:
                    walk.formats[node.uid] = reps
            else:
                # The file's copy of a tied weight: stored bytes, no
                # parameters, no deployed residency, no metadata elements.
                if tensors:
                    walk.checkpoint_copies[node.uid] = (tensors, reps)
                tensors, reps = (), ()
        if node.host_offload is not None and not tensors:
            raise SpecRefusal(
                f"{node.display_path} ({node.op}) declares `host_offload` and "
                "stores no weight, so there is nothing to keep in host memory",
                declaration="host_offload",
                fix="declare `host_offload` on the node that stores the "
                    "weight (its table), not on this one",
                node=node.display_path)
        # Flops derive from the propagated shapes — the same geometry bytes
        # and storage already read — with the dim environment answering only
        # the labels no tensor carries (`out` on a projection, a window
        # length). One source: an edge the graph gets wrong is a number the
        # suite sees, not a disagreement for a fork to adjudicate. The
        # adjudication that retired the fork found every divergence was a
        # bug on one side — three graph miswirings the environment reading
        # papered over with declared dims, and per-head elementwise ops that
        # same reading undercounted by exactly the head axis.
        flops_record: Dict[Any, Any] = {}
        flops = flops_over_operands(
            op, dims, flop_inputs,
            lambda d: op.flops(d, self._expr),
            record=flops_record)
        if flops_record:
            walk.flops_bindings[node.uid] = flops_record
        # A non-learnable buffer is STORED but is not a parameter, so it keeps
        # its `StoredTensor` — bytes still count it — and its elements move out
        # of `params`. Which tensor is a buffer is the op's declaration
        # (`StorageKind.BUFFER`); `_is_quant_meta` cannot substitute for it,
        # because that one also answers True for quantisation scales and so
        # cannot tell the two apart.
        # A tensor the checkpoint does not count is excluded by its KIND, not
        # by a second field on the cost. It keeps its `StoredTensor`, so every
        # byte it occupies is still counted where bytes are counted.
        counted = (0 if not owns_weight
                   or any(not t.is_parameter for t in tensors) else params)
        out = op.shape_out(in_shapes, dims, node.axis, node.axes, self._expr,
                           node=node)
        if not out:
            # Every ShapeKind produces a tensor; an empty answer is the
            # DimMap failing to infer one. Carrying `out=None` on let flops
            # fall back to the dim environment while traffic priced nothing.
            raise ValueError(
                f"{node.display_path} ({node.op}): output shape did not "
                f"resolve from inputs {input_ports!r}")
        # The share of itself a token runs at this leaf, stated once: the
        # parameters it touches take it, and the published share is it.
        routed = self._routed_share(node, tensors, scope)
        if routed is not None:
            walk.active_share[node.uid] = (
                walk.active_share[node.uid] * _share(*routed))
        cost = Cost(flops=flops, params=counted,
                    active_params=self._active_params(
                        node, counted, tensors, input_ports, out, routed),
                    quant_metadata_params=sum(
                        (r.metadata_elements for r in reps), 0))
        shape = Shape(
            out=tuple(out),
            ins=tuple(tuple(s) if s else () for s in input_ports),
        )
        for port, port_shape in op.secondary_ports(
                in_shapes, dims, node.axis, self._expr).items():
            shape = shape.with_port(port, port_shape)
        self._place(
            node,
            carried,
            owns_params,
            walk,
            input_ports,
            node_dims=dims,
            stored_tensors=tensors,
            weight_shape=tuple(op.spec.weight_shape or ()),
            out_shape=tuple(out) if out else (),
            op_spec=op.spec,
        )
        # A tensor that adopted a dictated length carries it on as long as it
        # still has that KV axis — an elementwise op or a reduction over
        # another axis of a score — so the selection ranking it prices the
        # pairs the stream states (`_record_kv_length` recorded only ops
        # whose template names the axis).
        if (arriving is not None and node.uid not in walk.kv_lengths
                and any(isinstance(axis, Dim) and axis.is_("s_kv")
                        and same_law(axis, arriving.read)
                        for axis in shape.out or ())):
            walk.kv_lengths[node.uid] = arriving
        return cost, shape

    def _place(self, node: Evaluable, carried: List[Carried],
               uses_weight: bool, walk: Walk,
               in_shapes: Sequence[Optional[Sequence[Any]]], *,
               node_dims: Mapping[str, Any],
               stored_tensors: Sequence[Any],
               weight_shape: Sequence[Any],
               out_shape: Sequence[Any] = (),
               op_spec: Any = None) -> None:
        """Where this node's output lives, and what its inputs cost to get here.

        On the same edges the shape just travelled, resolved by the same code —
        which is the whole reason this is a method of the walk and not a pass
        over the finished graph. Whether the node owns a WEIGHT is handed to the
        rule rather than looked up afterwards, because tied weights still have
        contraction geometry even though they do not create a second stored
        tensor. The arriving SHAPES travel with them for the same reason: an
        op's `couples_on` names slots, and only the tensors say which axis is
        in a given slot on this edge.
        """
        rule = self.rule
        if rule.is_trivial:
            return
        arrived = [item.state for item in carried]
        with node_context(node.display_path):
            requirements, produced = rule.transition(
                node,
                arrived,
                uses_weight,
                in_shapes,
                node_dims=node_dims,
                stored_tensors=stored_tensors,
                weight_shape=weight_shape,
                out_shape=out_shape,
                op_spec=op_spec,
            )
        if isinstance(requirements, transition_mod.InputRequirements):
            edge_required = requirements.edges
            kernel_required = requirements.kernels
            geometry = requirements.geometry
            if len(edge_required) != len(in_shapes):
                raise ValueError(
                    f"{node.display_path}: sharding transition returned "
                    f"{len(edge_required)} input states for "
                    f"{len(in_shapes)} consumer ports"
                )
        else:
            edge_required = kernel_required = requirements
            geometry = None
        if geometry is not None:
            walk.kernel_shapes[node.uid] = Shape(
                out=geometry.output,
                ins=tuple(axes or () for axes in geometry.inputs),
                pair_share=geometry.pair_share,
            )
        for index, state in enumerate(kernel_required):
            walk.input_states[(node.uid, index)] = state
        outputs = (produced if isinstance(produced, transition_mod.OutputStates)
                   else transition_mod.OutputStates(produced))
        out_state = outputs.out
        refs = tuple(node.inputs_obj or ())
        for index, (got, need) in enumerate(
            zip(arrived, edge_required, strict=False)
        ):
            if got is None or got == need:
                continue
            collectives = rule.lower_diff(
                got, need, node=node, input_index=index
            )
            if not collectives:
                continue
            # A ref into the caller's scope has no `src`; the carried tensor
            # still knows which node produced it.
            producer = carried[index].producer if index < len(carried) else None
            if index >= len(refs) and producer is not None:
                # No ref at all: the stage's sequential inbound (a stack entry
                # reading the previous stage's output). That edge is not in the
                # graph, so there is no consumer input to rewrite; the move
                # happens as the previous stage hands over — a stage boundary,
                # recorded where `_closed` records one.
                walk.redistributes.append(transition_mod.Redistribute(
                    node=producer[0], input_index=-1, producer=producer[0],
                    arrived=got, required=need, collectives=collectives,
                    site=transition_mod.RedistributeSite.STAGE_CLOSE))
                continue
            walk.redistributes.append(transition_mod.Redistribute(
                node=node.uid, input_index=index,
                producer=producer[0] if producer is not None else -1,
                arrived=got, required=need,
                collectives=collectives,
                producer_port=producer[1] if producer is not None else "out"))
        for side, entered, moved in rule.internal(
                kernel_required, out_state, node=node):
            dependency = None
            if side == "in":
                source_ref = refs[0] if refs else None
                producer_uid = (
                    source_ref.src.uid if source_ref is not None
                    and source_ref.src is not None else node.uid)
                producer_port = (
                    source_ref.port or "out" if source_ref is not None
                    else "out")
                for index in range(len(walk.redistributes) - 1, -1, -1):
                    earlier = walk.redistributes[index]
                    if (earlier.node == node.uid
                            and earlier.input_index == 0
                            and earlier.producer == producer_uid
                            and earlier.producer_port == producer_port
                            and earlier.required == entered
                            and earlier.collectives):
                        dependency = earlier
                        break
            edge = transition_mod.Redistribute(
                node=node.uid, input_index=0 if side == "in" else -1,
                producer=node.uid, arrived=entered, required=out_state,
                collectives=moved,
                site=(transition_mod.RedistributeSite.INTERNAL_IN
                      if side == "in"
                      else transition_mod.RedistributeSite.INTERNAL_OUT))
            src = refs[0].src if side == "in" and refs else None
            if src is not None:
                edge = replace(
                    edge,
                    producer=src.uid,
                    producer_port=refs[0].port or "out",
                )
            edge = replace(edge, dependency=dependency)
            walk.redistributes.append(edge)
        walk.set_state(node.uid, out_state)
        for port, state in outputs.secondary:
            walk.set_state(node.uid, state, port=port)

    def _inputs(self, node: Evaluable, scope: Scope,
                walk: Walk) -> List[Carried]:
        """Every declared input, resolved ONCE — tensor and placement together.

        Positionally aligned with `inputs_obj`, including the entries whose
        shape could not be resolved, because a state has to line up with the
        port it arrived at even when the shape did not. `_input_shapes` then
        filters, preserving the behaviour the oracle pins.
        """
        refs = tuple(node.inputs_obj or ())
        # A zero-input op (a parameter, a buffer) declares no inputs. Every
        # other node names its upstream — on the ModelLayer, where the name
        # (a stage, a stream, a model input) is bound as an alias on the
        # stage, or as a graph edge. A source
        # reads nothing even when a caller wired an edge to it: handing it a
        # tensor gave it a `shape.ins` the size of the hidden state, which
        # every reader of that shape then believed — traffic priced an
        # activation read, and the layout walk saw an arriving edge where the
        # placement comes from `weight_axes`. The question is the op's
        # declared arity, asked of the declaration.
        op = OpRef.resolve(node.op, self.program)
        if op is not None and op.arity() == 0:
            walk.input_arity[node.uid] = 0
            return []
        out: List[Carried] = []
        for ref in refs:
            handed = (self.program.handed_exits.get(ref.src.uid)
                      if ref.src is not None else None)
            if handed is not None:
                # An encoder's output, read through its handoff.
                out.append(scope.resolve(handed))
                continue
            if ref.src is not None:
                port = ref.port or "out"
                handed_on = walk.handed_on.get((ref.src.uid, port))
                if handed_on is None:
                    handed_on = Carried(
                        shape=walk.shapes.get(ref.src.uid),
                        state=walk.state_at(ref.src.uid, port=port),
                        kv_length=walk.kv_lengths.get(ref.src.uid),
                        producer=(ref.src.uid, port))
                out.append(replace(handed_on,
                                   origin=self._src_reference(node, ref)))
                continue
            found = scope.read(ref)
            if found is not None:
                out.append(found)
            else:
                raise ValueError(
                    f"{node.display_path} ({node.op}) reads "
                    f"{ref.external_sid!r}, which nothing binds here: not a "
                    f"formal its caller passed, not a stream this stage "
                    f"sees, not a stage's `inputs:` name. Name a tensor that "
                    f"exists where it is read — an earlier layer, a formal "
                    f"of the component, or a name in `model_inputs:`")
        for index, carried in enumerate(out):
            walk.input_references[(node.uid, index)] = carried.origin
        walk.input_arity.setdefault(node.uid, len(out))
        return out

    def _src_reference(self, node: Evaluable, ref: Any) -> Reference:
        """What a `src` edge reads: in a block template, the same iteration
        of the sibling; at a model-level op, the op, run once."""
        stage, _root = self.program.evaluating_stage(ref.src.uid)
        port = ref.port or "out"
        if not isinstance(stage, BlockStage):
            return Reference.fixed((Source(Occurrence(ref.src.uid, None),
                                           port, "src"),))
        if stage is not self.program.evaluating_stage(node.uid)[0]:
            raise ValueError(
                f"{self.program.spec.name}: {node.display_path} reads "
                f"{self.program.where(ref.src.uid)} by `src` from another "
                f"block stage; no iteration pairs the two")
        return Reference.sibling(ref.src.uid, port)

    @staticmethod
    def _arriving_kv_length(node: Evaluable, walk: Walk,
                            carried: Sequence[Carried] = ()
                            ) -> Optional[KVLength]:
        """The KV cardinality this node's inputs bring with them, if any does.

        A `kv_stream_concat` union sets `dictates`, so does a stream marker
        (`kv_stream_marker`: a pooled latent, a window), and so does a stack
        stream whose producer cached the tensor (`Carried.kv_length`).
        Every other law-declaring op records its length for a union to sum
        and says nothing to its own consumers.
        """
        found: Optional[KVLength] = None
        refs = tuple(node.inputs_obj or ())
        for index, ref in enumerate(refs):
            length = None
            if ref.src is not None:
                length = walk.kv_lengths.get(ref.src.uid)
            elif index < len(carried):
                length = carried[index].kv_length
            if length is None or not length.dictates:
                continue
            if found is not None and str(found.read) != str(length.read):
                # Two unions of different length meeting at one op is a
                # mis-wired graph, and picking either silently mis-sizes every
                # derivation downstream — the same reason the unknown op above
                # raises rather than skipping.
                raise ValueError(
                    f"node {node.name!r} reads two KV unions of different "
                    f"length: {found.read} and {length.read}")
            found = length
        return found

    def _record_kv_length(self, node: Evaluable, op: OpRef, walk: Walk,
                          arriving: Optional[KVLength],
                          positions: Any) -> None:
        """What this node's output carries onward.

        Four cases, and the third is the whole propagation. A union sums its
        operands. A stream marker's law is its output's length, so it
        dictates that. A tensor that adopted a dictated length keeps carrying it
        as long as it is still THAT tensor — through a pure re-arrangement, or
        through an op whose output still has the KV axis — and stops at the
        apply that contracts the axis away, which is where the number ceases to
        describe anything. Anything else records its own law, unadvertised, so
        that a union upstream of it has something to sum.
        """
        if op.kv_stream_union:
            parts = [walk.kv_lengths.get(ref.src.uid)
                     for ref in (node.inputs_obj or ())
                     if ref.src is not None]
            if not parts or any(part is None for part in parts):
                unpublished = (
                    f" This isolated call reads "
                    f"{sorted(walk.unpublished)}, which the model publishes "
                    f"to no layer of the call, so no cache law came with "
                    f"them: evaluate the component at a call whose layers "
                    f"read them" if walk.unpublished else "")
                raise ValueError(
                    f"node {node.name!r} joins KV streams, but not every "
                    "operand declares a cardinality — a union whose operands "
                    "are silent has no length to give the attention that "
                    "reads it." + unpublished)
            read = parts[0].read
            symbolic = parts[0].symbolic
            for part in parts[1:]:
                read = read + part.read
                symbolic = symbolic + part.symbolic
            # The union's live pairs are its streams' — each query sees its
            # window AND its pooled blocks — and only a stream that states
            # its own mask can say how many. One that does not, beside one
            # that does, leaves the sum unknowable; that is a graph to
            # repair, not a fraction to guess.
            masked = [part.pairs is not None for part in parts]
            if any(masked) and not all(masked):
                raise ValueError(
                    f"node {node.name!r} joins KV streams of which only some "
                    "declare their live pairs (`kv_pairs_per_step_fn`) — the "
                    "attention over the union cannot price one stream's mask "
                    "and guess the other's")
            pairs = named = None
            if all(masked):
                pairs, named = parts[0].pairs, parts[0].named
                for part in parts[1:]:
                    pairs = pairs + part.pairs
                    named = (None if named is None or part.named is None
                             else named + part.named)
            walk.kv_lengths[node.uid] = KVLength(read=read, symbolic=symbolic,
                                                 dictates=True, pairs=pairs,
                                                 named=named,
                                                 listed=pairs is not None)
            return
        own = None
        if positions is not None and positions.read is not None:
            recorded = walk.cache_positions.get(node.uid)
            # In an encoder the length a score reads is its segment's rows,
            # which no binding states: its softmax and apply take it from
            # the score, as a stream marker's readers do.
            own = self._stated_length(
                node.uid, op, positions,
                recorded.read if recorded is not None
                and recorded.read is not None else positions.read,
                encoder=node.uid in self._encoders)
        if own is not None and own.dictates:
            walk.kv_lengths[node.uid] = own
            return
        if arriving is not None and (
                op.is_view_op or "s_kv" in op.shape_out_template):
            walk.kv_lengths[node.uid] = arriving
            return
        if own is not None:
            walk.kv_lengths[node.uid] = own

    @staticmethod
    def _stated_length(uid: Optional[int], op: OpRef, positions: Any,
                       symbolic: Any, *, encoder: bool) -> KVLength:
        """The length node `uid`'s op states for its output
        (`OpRef.kv_positions`: `positions` at this walk, `symbolic` with the
        cache length held free). A stream marker's — and an encoder's
        attention's, whose readers take its segment's rows — dictates its
        readers' `s_kv`; any other is recorded for a union to sum."""
        return KVLength(read=positions.read, symbolic=symbolic,
                        dictates=op.kv_stream_marker or encoder,
                        pairs=positions.pairs, named=positions.named,
                        stated_by=uid)

    def _carry_kv_length(self, node: Evaluable, walk: Walk) -> None:
        """A component's output carries what its output leaf produced, and
        nothing else: a wrapper (`head_broadcast`) hands on its input's
        length because its children read that input as a formal that
        carries it (`Carried.kv_length`) and record it as views do, and a
        component whose leaf made a new tensor — an indexer gathering
        positions — hands on none. Inheriting the inputs' length at the
        boundary instead gave such a component the length of whichever
        input happened to carry one.
        """
        leaf, _ = self.program.output_leaf(node, "out")
        length = (walk.kv_lengths.get(leaf.uid) if leaf is not None else None)
        if length is not None:
            walk.kv_lengths[node.uid] = length

    def _input_shapes(self, node: Evaluable, scope: Scope,
                      carried: List[Carried]) -> List[List[Any]]:
        """Upstream shapes, in the node's declared input order.

        Two resolution paths, and the second is the whole reason `Scope` carries
        aliases: an object ref reads the producer's layer entry by uid, while a
        ref `iter_tree` left unresolved names an enclosing scope's input and reads
        the frame. A port ref takes the producer's secondary output rather than
        its main one — conflating those is what left a V4-Pro broadcast with no
        shape at all.
        """
        refs = tuple(node.inputs_obj or ())
        out: List[List[Any]] = []
        for index, item in enumerate(carried):
            found = item.shape
            if found is None:
                continue
            port = (refs[index].port or "out") if index < len(refs) else "out"
            dims = found.ports.get(port) if port != "out" else found.out
            if dims:
                out.append(list(dims))
        return out

    @staticmethod
    def _primitive_input_shapes(node: Evaluable, carried: List[Carried],
                                ) -> List[Optional[List[Any]]]:
        """The arriving shape at each input port, by position; a port nothing
        arrived at stays None so later ports keep their positions."""
        refs = tuple(node.inputs_obj or ())
        out: List[Optional[List[Any]]] = []
        for index, item in enumerate(carried):
            found = item.shape
            port = (refs[index].port or "out") if index < len(refs) else "out"
            shape = (found.ports.get(port) if found is not None and port != "out"
                     else found.out if found is not None else None)
            out.append(list(shape) if shape else None)
        return out

    # -- shared helpers --------------------------------------------------

    def _binding_authoritative(
        self,
        in_shapes: List[Optional[List[Any]]],
        op: OpRef,
        dims: Mapping[str, Any],
        binding: Binding,
        *,
        reads_cache: bool,
    ) -> List[Optional[List[Any]]]:
        """Where an op that reads the cache names, in its `shape_in`, a symbol
        the binding declared AUTHORITATIVE, take the node's value for it
        rather than the producer's extent. Which symbols those are is declared
        by whoever bound them — the phase, since a cache of length `s_kv` that
        no node produces is a fact about the workload and not about the graph.

        An op reads the cache when it declares how many cache positions a
        query touches (`OpSpec.kv_positions_per_query_fn`: a score, an apply,
        a stream marker such as `kv_cache_view`). For it the template says
        something the dataflow cannot: in decode the K/V leaves emit only the
        NEW token, so their shape carries `s_ctx = 1`, while the op reads the
        whole cache. Its `dims["s_kv"]` is the length its law gives — the
        arriving stream's, narrowed to `Min(s_kv, s_query + s_window − 1)` for
        a window or `Min(s_kv, n_topk_keys)` for a top-k — so the
        substitution gives what the kernel really loads.

        Every other op reads the rows that arrive (`in:` is the edge): a
        top-k over pooled block scores ranks the scores its producer made,
        whatever the cache holds.

        Positional: the template's axis names line up index-for-index with the
        resolved shape, and only positions whose declared name is authoritative are
        touched.
        """
        template = op.shape_in_template
        if not template or not in_shapes or not reads_cache:
            return in_shapes
        wanted = {a for a in binding.authoritative if a in dims}
        if not wanted:
            return in_shapes
        # One template per input port when the op declares several, else the
        # single template applies to input 0 only.
        per_port = (list(template)
                    if template and isinstance(template[0], (list, tuple))
                    else [template])
        def _bound(name: Any) -> Any:
            # The frame is tracked; the binding already carries identity.
            return dims[str(name)]

        out: List[Optional[List[Any]]] = []
        for idx, shape in enumerate(in_shapes):
            if not shape:
                out.append(shape)
                continue
            axes = per_port[idx] if idx < len(per_port) else None
            if not axes or len(axes) != len(shape):
                out.append(shape)
                continue
            out.append([_bound(name) if str(name) in wanted else extent
                        for name, extent in zip(axes, shape, strict=False)])
        return out

    @staticmethod
    def _expr(expr: Any, dims: Mapping[str, Any]) -> Any:
        """Resolve a dim expression written in YAML against a dims map.

        The one place a custom op's `flops:` / `params:` / shape template is
        evaluated, so it cannot drift from the dims the primitive path uses —
        they are the same dict, passed in. Passed through unmaterialised:
        `eval_dim_expr` looks up only the expression's own free symbols, so a
        ReadTracking map sees exactly the reads this expression made.
        """
        return eval_dim_expr(str(expr), dims)

    def _active_params(self, node: Evaluable, counted: Any, tensors: Any,
                       input_ports: Sequence[Any], out: Tuple[Any, ...],
                       routed: Optional[Tuple[Any, Any]]) -> Any:
        """The parameters one token touches at this leaf.

        `routed` is the share of its own weights a token runs
        (`_routed_share`), None where it runs all of them; applied here
        exactly — an integer when it divides, the `Rational` average
        otherwise.

        A leaf that gathers rows from its own table (`Program.lookups`)
        touches the rows it reads, not the table: `min(1, r / R)` of it, `r`
        the rows one token reads — the output's lookups (its elements over
        the row width) over the ids that arrive — and `R` the table's rows.
        The law is the one traffic bills the read by
        (`traffic._table_read_fractions`), per token instead of per step.
        DeepSeek-V4.1's Engram reads 24 rows of a 384M-row table per token.

        The model's input embedding (`boundary_role: input_embedding`) is
        the one table counted whole: that is how a model's activated
        parameters are reported, and the head that shares or mirrors it
        reads every row anyway. Every other leaf touches all it holds."""
        if routed is not None:
            num, den = routed
            counted = apply(
                lambda c, n, d: c * n / d,
                lambda c, n, d: (c * n // d if not c * n % d
                                 else sp.Rational(c * n, d)),
                counted, num, den)
        site = self.program.lookups.get(node.uid)
        if (site is None or site.table != node.uid or not tensors
                or node.boundary_role == "input_embedding"):
            return counted
        ids = input_ports[site.ids] if site.ids < len(input_ports) else None
        if ids is None or len(out) <= len(ids):
            raise ValueError(
                f"{node.display_path} ({node.op}) gathers into "
                f"{list(out)} from ids {ids}; a lookup's output is its ids' "
                "axes, the rows each id reads, then the row width")
        # The axes between the ids' and the row width are the rows one id
        # reads (an embedding has none: one row). Read off the shape rather
        # than divided out of the step's token count, which an encoder the
        # workload gives no input has at zero.
        extra = out[len(ids):-1]
        rows_per_token = product(extra) if extra else 1
        # A proportion, so `ratio`: a table with no rows is read by nobody.
        # `/` was a float here and not even a faithful one — one row of a
        # 151552-row table came out `6.598395270270271e-06`, a non-dyadic
        # value that then multiplied the leaf's parameter count.
        return counted * minimum(
            1, ratio(rows_per_token, tensors[0].shape[0], empty=0))

    @staticmethod
    def _routed_share(node: Evaluable, tensors: Any,
                      scope: Scope) -> Optional[Tuple[Any, Any]]:
        """`(active, total)` — the share of ITS OWN weights a token runs at
        this leaf — or None where a token runs all of them.

        A weight whose leading axis is the experts is touched
        `n_experts_active / n_route_targets` of it per token inside a
        dispatch zone (`Scope.active_ratio`, which keeps the pair rather than
        a quotient so the multiplication stays exact). A leaf whose weights
        all lead with the experts takes the share whole; one that mixed
        expert-led and plain weights would need a per-tensor answer no spec
        has asked for, and is refused.

        Stated once and read twice: `_active_params` multiplies the
        parameters a token touches by it, and `Analysis.active_share`
        publishes it as the share itself. It used to be applied to the
        parameters AND multiplied into the published copy COUNT, which made
        that count a count times a fraction — and the two readers of it both
        divided the fraction back out to get at the share.
        """
        params = [t for t in tensors if t.is_parameter]
        led = [t for t in params if leads_with_expert_axis(t)]
        if not led or scope.active_ratio is None:
            return None
        if len(led) != len(params):
            raise ValueError(
                f"{node.display_path} ({node.op}): stores weights both led by "
                "the expert axis and not, and the share of them a token runs "
                "is one number per leaf")
        return scope.active_ratio

    @staticmethod
    def _node_dims_override(
        node: Evaluable, scope: "Scope",
    ) -> Dict[str, Any]:
        """A node's own `dims:` / `dim:`, resolved against the frame.

        Resolved from `dims_raw`, the expression the spec author wrote — NOT from
        the sibling `dims` field. That field is a build-time best effort and it is
        not reliably resolved: a Llama `q_proj` carries `dims_raw["bias"] = "Qb"`
        alongside `dims["bias"] = Symbol("Qb")`, even though `Qb` is bound to 0 and
        grounds immediately. Trusting it sends a Symbol into `params_fn`, and the
        linear reports `3072*Qb + 9437184` params instead of an int — one
        unresolved flag turns the whole model total symbolic.

        The engine never hits this because it walks `BlockGraph`, whose `dims:` are
        the raw strings, and re-resolves per node. So of that dual channel the
        "resolved" side is the stale one, and nothing numeric may read it.

        One frame, the node's own, and the expression is read in the vocabulary
        it was written in. `dims: {d_ffn: d_ffn_expert}` says the callee's
        `d_ffn` is this caller's `d_ffn_expert`. Extents stay global; layout
        names the cut, so the right-hand axis is not divided here.
        """
        flat: Optional[Dict[str, Any]] = None

        def _resolve(value: Any) -> Any:
            nonlocal flat
            if not isinstance(value, str):
                return value
            if flat is None:
                flat = scope.flat_dims()
            return eval_dim_expr(value, flat)

        raw = node.dims_raw or {}
        over: Dict[str, Any] = {}
        for key, val in (node.dims or {}).items():
            over[key] = _resolve(raw[key]) if key in raw else val
        if node.dim is not None or node.dim_raw is not None:
            over["dim"] = (_resolve(node.dim_raw) if node.dim_raw
                           else node.dim)
        return over

    def _copies(self, node: Evaluable, scope: Scope) -> OwnCopies:
        """The node's OWN multipliers: `repeat:` for what is stored, `active:`
        for what runs per token.

        Own, not cumulative — `multiplicity` on the IR node is the product
        through ancestors, and applying it at every level would re-apply every
        enclosing repeat. The cumulative count is what the FRAME carries.

        One resolution, two consumers. The cost multiplier and the storage count
        are the same statement about the same copies, and reading them from two
        places is how a per-rank walk came to report a rank holding every
        expert while charging it for a fraction of the FLOPs.

        A node that declares no `active:` runs every copy it stores. An expert
        bank has no copies here at all: its experts are an axis of its
        weights (`shape: [n_expert_copies, ...]`) and of the activation it
        reads, and the share of that axis a token runs is a fact about the
        tensors (`_active_params`), not a copy count.
        """
        storage = self._count(node.repeat, node.repeat_raw, scope)
        if node.repeat is not None and (scope.active_ratio is not None
                                        or is_expert_axis(storage)):
            raise ValueError(
                f"{node.display_path} ({node.op}): declares `repeat: "
                f"{node.repeat_raw}` over a routed expert bank. A bank's "
                "experts are the leading axis of its weights (`shape: "
                "[n_expert_copies, ...]`, the expert templates in "
                "components/library), and a token runs the router's share of "
                "that axis; a repeated copy has no declared share")
        active = (self._count(node.active, node.active_raw, scope)
                  if node.active is not None else storage)
        return OwnCopies(storage, active)

    def _count(self, resolved: Optional[int], raw: Optional[str],
               scope: Scope) -> Any:
        """A declared count, resolved against THIS frame's binding.

        `iter_tree` already resolved `repeat:` — against the spec's global dims,
        which is the only binding it has. That answer is right for the model and
        wrong for a rank: expert parallelism divides the copies axis, so a
        per-rank frame binds a smaller number under the same symbol and the
        baked integer cannot see it. Re-resolving the expression the spec WROTE
        is what makes the count follow the frame.

        The resolved integer is NOT kept as a fallback. `spec_tree` sets the two
        forms together or sets neither, so a count with no expression is a
        contract violation and not a case. And the fallback was not inert: in
        SYMBOLIC mode `n_expert_copies` is a `Symbol`, the guard rejected it as
        "not an int", and the walk quietly substituted the concrete count — so a
        symbolic model total carried a literal expert count inside it. Measuring
        the fallback numerically said it fired zero times, which is exactly how
        a fallback hides the domain it is wrong in.

        A non-integer is therefore an ANSWER, not a failure: this walk runs over
        two value domains and a count is an expression in both. What is refused
        is a count that resolves to nothing usable — a non-positive integer.

        A count that will not resolve makes every number UNDER that node
        wrong too, and there is no per-node skip channel to soften it into:
        it crashes, loudly, at the node that owed the answer.
        """
        if resolved is None:
            return 1
        if not raw:
            raise ValueError(
                f"a resolved count of {resolved} with no expression behind it; "
                f"`repeat` and `repeat_raw` are set together or not at all")
        value = self._expr(raw, scope.flat_dims())
        if isinstance(value, bool) or (isinstance(value, int) and value <= 0):
            raise ValueError(
                f"count {raw!r} resolved to {value!r} at this binding")
        return value


def _narrowed_s_kv(positions: Any, op: OpRef) -> Optional[Any]:
    """The law the op narrowed `s_kv` by, or None where the node narrowed
    nothing a reader of its `s_kv` must re-evaluate.

    It is the definition in the node's names, each bound to what the frame
    bound it to BEFORE the narrowing (`KVPositions.step_named`):
    `Min(s_kv, s_query + s_window − 1)` at the frame's own context length.
    This walk is global, so it evaluates the law at the step's whole query
    count; a rank that holds part of the queries reaches fewer keys, and
    the per-rank projection re-evaluates this law at its own extents
    (`layout.physical.rank_law_shape`) — the narrowed value is no input to
    it, so a law that is no idempotent cap (a pooled stream's `⌊s_kv/m⌋`)
    is not applied twice.

    There is a law to re-evaluate where it says something in the node's own
    names — not a dense op's bare `s_kv`, nor a length an arriving stream
    fixed in the workload's (a pooled or joined stream's) — and where the
    node's tensors carry the narrowed `s_kv` (its templates name it: the
    K/V operands `_binding_authoritative` writes it on). A node that does
    not (a windowed softmax, whose axis is `s_attend`) keeps the context as
    its `s_kv`, and has nothing it defines. Every question here is asked of
    names and laws, never of a magnitude: which symbols the law reads is a
    set of names, and `same_law` compares expressions."""
    defined = (positions.step_named if positions is not None
               and positions.read is not None else None)
    if defined is None or not defined.bindings:
        return None
    reads = {symbol.name for symbol in defined.law.free_symbols}
    if not reads.issubset(defined.bindings):
        return None
    if same_law(defined.law, sp.Symbol("s_kv")):
        return None
    return defined if "s_kv" in _template_names(op) else None


def _template_names(op: OpRef) -> set:
    """Every axis name `op`'s input and output templates write, whether it
    declares one input template or one per port."""
    names: set = set()
    for template in (op.shape_in_template, op.shape_out_template):
        for item in template or ():
            if isinstance(item, (list, tuple)):
                names.update(str(name) for name in item)
            else:
                names.add(str(item))
    return names
