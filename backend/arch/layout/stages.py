"""Pipeline stages as placement: where every occurrence runs, what crosses
each cut, and what one stage's rank runs and holds.

`pp` cuts the backbone's iteration space into contiguous blocks
(`stage_layer_ranges`, the canonical balanced partition). Every other node's
stage follows from the dataflow the walk bound (`Analysis.occurrence_graph`):

- a unit whose every external input is free, or itself pulled, is pulled to
  its earliest consumer (the embedding, an encoder tower and what merges it);
- any other unit runs after its latest input (the final norm and head, an MTP
  head);
- an edge whose two ends sit on different stages is a hop: one `send_recv` per
  cut it crosses, sent once however many sinks read the value past that cut
  (`StageCrossing`, layer 1). Each sink's own input transition is the record
  the walk already made for it (layer 2), on the sink's stage.

`StageView` projects one stage three ways, each within its own scope: the
rows it times, the weights it holds, the cache it keeps.

A stage is named across analyses by its index: each analysis (a prefill and
a decode step walk different programs) places its own program and projects
the same index.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import (Any, Dict, FrozenSet, Hashable, Iterator, List, Mapping,
                    Optional, Set, Tuple)

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.dim import minimum, ratio
from backend.arch.eval.layer import Layer
from backend.arch.eval.occurrence import (Occurrence, OccurrenceGraph,
                                          StageCrossing)
from backend.arch.eval.program import BlockStage, OpStage, Program, Stage
from backend.arch.eval.transition import Redistribute, RedistributeSite
from backend.arch.ops import is_identity_source

from .parallel_config import stage_layer_ranges
from backend.code_trace.memo import memo_property


@dataclass(frozen=True)
class StagePlacement:
    """The one stage every computed occurrence of `program` runs on, at
    `pp` stages.

    The partitioned iteration space is the concatenation, in walk order, of
    the repeated model layers of every stack that is not aux (`Stack.is_aux`,
    a declared boolean; no kind is read). The placement units are: each
    iteration of that space; each aux stack, whole; and each stage root run
    once with everything under it. Parameters (identity sources) are free
    endpoints: readable on any stage, so an edge from one is never a hop.

    Two linear passes over the unit graph place the rest (`_stages`). A
    cycle among units is a structural error at any `pp`; an edge read on an
    earlier stage than it is written is an engine bug.
    """

    graph: OccurrenceGraph
    pp: int

    def __post_init__(self) -> None:
        if self.pp < 1:
            raise ValueError(f"pp must be at least 1, got {self.pp}")

    @classmethod
    def of(cls, analysis: Any, pp: Optional[int] = None) -> "StagePlacement":
        """The placement of the program `analysis` walked, by the edges the
        walk bound, at `pp` stages — when None, the stages its rule's
        pipeline world declares (one without one)."""
        if pp is None:
            family = analysis.rule.pipeline_family
            pp = 1 if family is None else family.group_size
        return cls(analysis.occurrence_graph, pp)

    @property
    def program(self) -> Program:
        return self.graph.program

    # -- the partition -----------------------------------------------------

    @property
    def _partitioned(self) -> Mapping[Tuple[int, ...], Tuple[int, int]]:
        """`layer_path` → `(offset, depth)` in the partitioned
        iteration space (`Program.iteration_space`, checked whole when the
        program is walked, at any `pp`)."""
        return self.program.iteration_space

    @memo_property
    def n_iterations(self) -> int:
        """The length of the partitioned iteration space."""
        return sum(depth for _offset, depth in self._partitioned.values())

    @memo_property
    def ranges(self) -> Tuple[range, ...]:
        """The global iterations each stage holds (`stage_layer_ranges`)."""
        return tuple(stage_layer_ranges(self.n_iterations, self.pp))

    @memo_property
    def _stage_of_global(self) -> Tuple[int, ...]:
        return tuple(stage for stage, held in enumerate(self.ranges)
                     for _ in held)

    def partitioned(self, stage: Stage) -> bool:
        return stage.layer_path in self._partitioned \
            and isinstance(stage, BlockStage)

    def iteration_stage(self, stage: Stage, iteration: int) -> int:
        """The stage iteration `iteration` of a partitioned stage runs on."""
        offset, _depth = self._partitioned[stage.layer_path]
        return self._stage_of_global[offset + iteration]

    # -- units -------------------------------------------------------------

    @memo_property
    def _aux(self) -> FrozenSet[str]:
        return frozenset(stack.name for stack in self.program.stacks
                         if stack.is_aux)

    def unit(self, occurrence: Occurrence) -> Hashable:
        """The placement unit an occurrence belongs to."""
        stage, root = self.program.evaluating_stage(occurrence.uid)
        if self.partitioned(stage):
            return ("iteration", stage.layer_path, occurrence.iteration)
        if stage.stack in self._aux:
            return ("stack", stage.stack)
        return ("root", root.uid)

    def is_free(self, uid: int) -> bool:
        """A free endpoint: a parameter, readable on any stage."""
        return is_identity_source(self.program.by_uid(uid).op or "")

    def _occurrence_edges(self) -> Iterator[Tuple[Occurrence, int,
                                                  Occurrence, str]]:
        """Every edge as `(sink, input, source, port)`: the explicit pairs,
        and a template-local edge only where its ends can be two units (a
        stage run once outside an aux stack)."""
        for edge in self.graph.pairs:
            yield edge.sink, edge.input_index, edge.source, edge.port
        for sink, index, _k, source, port in self.graph.local:
            stage, _root = self.program.evaluating_stage(sink)
            if self.partitioned(stage) or stage.stack in self._aux:
                continue
            for iteration in self.program.stage_iterations(stage):
                yield (Occurrence(sink, iteration), index,
                       Occurrence(source, iteration), port)

    @memo_property
    def _unit_graph(self) -> Tuple[Tuple[Hashable, ...],
                                   Mapping[Hashable, Set[Hashable]],
                                   Mapping[Hashable, Set[Hashable]],
                                   Mapping[Hashable, Set[Hashable]]]:
        """`(units in topological order, inputs, free inputs, consumers)`.
        `inputs` are the units a unit reads a computed value from; `free`
        those it reads only parameters from; `consumers` every unit reading
        it."""
        order_key: Dict[Hashable, Tuple] = {}
        for uid, iterations in self.program.occurrence_domains():
            stage, _root = self.program.evaluating_stage(uid)
            position = self.program.walk_position(stage)
            for iteration in iterations:
                unit = self.unit(Occurrence(uid, iteration))
                key = (position, -1 if iteration is None else iteration, uid)
                if unit not in order_key or key < order_key[unit]:
                    order_key[unit] = key
        inputs: Dict[Hashable, Set[Hashable]] = {u: set() for u in order_key}
        free: Dict[Hashable, Set[Hashable]] = {u: set() for u in order_key}
        consumers: Dict[Hashable, Set[Hashable]] = {u: set()
                                                    for u in order_key}
        for sink, _index, source, _port in self._occurrence_edges():
            into, out_of = self.unit(sink), self.unit(source)
            if into == out_of:
                continue
            consumers[out_of].add(into)
            (free if self.is_free(source.uid) else inputs)[into].add(out_of)
        pending = {u: len(inputs[u] | free[u]) for u in order_key}
        ready = sorted((u for u, n in pending.items() if not n),
                       key=order_key.__getitem__)
        order: List[Hashable] = []
        while ready:
            unit = ready.pop(0)
            order.append(unit)
            for consumer in consumers[unit]:
                pending[consumer] -= 1
                if not pending[consumer]:
                    ready.append(consumer)
            ready.sort(key=order_key.__getitem__)
        if len(order) != len(order_key):
            stuck = sorted((u for u in order_key if pending[u]),
                           key=order_key.__getitem__)
            raise ValueError(
                f"{self.program.spec.name}: placement units {stuck[:4]} "
                f"read each other in a cycle; a unit is placed whole, so no "
                f"stage order runs it")
        return tuple(order), inputs, free, consumers

    @memo_property
    def pulled(self) -> FrozenSet[Hashable]:
        """The pull set: units outside the partition whose every input is
        free or pulled (the least fixed point, in topological order)."""
        order, inputs, _free, _consumers = self._unit_graph
        out: Set[Hashable] = set()
        for unit in order:
            if unit[0] != "iteration" and inputs[unit] <= out:
                out.add(unit)
        return frozenset(out)

    @memo_property
    def _stages(self) -> Mapping[Hashable, int]:
        """Each unit's stage. Forward, in topological order: an iteration
        on its partition's stage, any other unit outside the pull set after
        its latest computed input. Backward, over the pull set: the earliest
        consumer's stage; with no consumer, the last stage for a model output
        and the first otherwise."""
        order, inputs, _free, consumers = self._unit_graph
        stages: Dict[Hashable, int] = {}
        for unit in order:
            if unit[0] == "iteration":
                _kind, layer, iteration = unit
                offset, _depth = self._partitioned[layer]
                stages[unit] = self._stage_of_global[offset + iteration]
            elif unit not in self.pulled:
                stages[unit] = max(stages[source] for source in inputs[unit]
                                   if source not in self.pulled)
        for unit in reversed(order):
            if unit not in self.pulled:
                continue
            if consumers[unit]:
                stages[unit] = min(stages[c] for c in consumers[unit])
            else:
                stages[unit] = (self.pp - 1 if self._is_output(unit) else 0)
        return stages

    def _is_output(self, unit: Hashable) -> bool:
        """Whether a unit holds a model-output endpoint, where logits are
        sampled (`boundary_role: output_head`)."""
        return any(self.program.by_uid(uid).boundary_role == "output_head"
                   for uid, iterations in self.program.occurrence_domains()
                   for iteration in iterations
                   if self.unit(Occurrence(uid, iteration)) == unit)

    def stage(self, occurrence: Occurrence) -> int:
        """The stage an occurrence runs on."""
        return self._stages[self.unit(occurrence)]

    def stage_running(self, stage: Stage, iteration: Optional[int]) -> int:
        """The pipeline stage the walk's `stage` runs `iteration` on — the
        unit it places, read off the stage itself: an iteration of a
        partitioned block, a whole aux stack, or a model-level op's root.
        Any node of a graph built over the walk that the stage evaluates
        (a collective or cast a pass grafted into it) runs there too."""
        if self.partitioned(stage):
            return self.iteration_stage(stage, iteration)
        if stage.stack in self._aux:
            return self._stages[("stack", stage.stack)]
        if not isinstance(stage, OpStage):
            raise ValueError(
                f"stage {stage.name!r} is no partitioned block, aux stack or "
                f"model-level op; no unit places it")
        return self._stages[("root", stage.node.uid)]

    # -- crossings -----------------------------------------------------------

    @memo_property
    def crossings(self) -> Tuple[StageCrossing, ...]:
        """Every hop (layer 1): per computed value read on a later stage,
        one per cut between its stage and its sink's, deduplicated over
        sinks by `(value, source iteration, cut)`. Asserts that no edge is
        read on an earlier stage than it is written."""
        sinks: Dict[Tuple[Tuple[int, str], Optional[int], int],
                    List[Tuple[Occurrence, int]]] = {}
        for sink, index, source, port in self._occurrence_edges():
            if self.is_free(source.uid):
                continue
            start, end = self.stage(source), self.stage(sink)
            if start > end:
                raise AssertionError(
                    f"{self.program.spec.name}: {sink} (stage {end}) reads "
                    f"{source} (stage {start}); placement ran a reader "
                    f"before its producer")
            for cut in range(start, end):
                sinks.setdefault(((source.uid, port), source.iteration, cut),
                                 []).append((sink, index))
        return tuple(
            StageCrossing(producer=producer, source_iteration=iteration,
                          cut=cut, sinks=tuple(found),
                          serves=frozenset(
                              self.program.evaluating_stage(sink.uid)[0].stack
                              for sink, _index in found))
            for (producer, iteration, cut), found in sorted(
                sinks.items(), key=lambda item: (
                    item[0][2], item[0][0], -1 if item[0][1] is None
                    else item[0][1])))

    @memo_property
    def backbone(self) -> FrozenSet[str]:
        """The default scope of every projection: the stacks that are not
        aux."""
        return frozenset(stack.name for stack in self.program.stacks
                         if not stack.is_aux)

    def records(self, analysis: Any) -> List[Redistribute]:
        """Every hop as the transition record a cut is priced from: one
        `send_recv` of the value in the state it is handed on in, the hop
        attached (`Redistribute.crossing`).

        A hop of a cache-resident value — a cache one layer keeps and later
        layers read through `reads_from` — carries what those readers read
        of the cache this step (`CacheRead`), not the producer's output:
        the reader's stage receives that span from the writer's every step
        and keeps no copy."""
        family = analysis.rule.pipeline_family
        if family is None:
            return []
        reads = self._cache_reads(analysis)
        out: List[Redistribute] = []
        for crossing in self.crossings:
            found = reads.get(crossing.producer[0])
            if found is not None:
                crossing = replace(crossing, cache_read=found)
            uid, port = crossing.producer
            state = self.handed_state(analysis, crossing.producer)
            out.append(Redistribute(
                node=uid, input_index=-1, producer=uid, producer_port=port,
                arrived=state, required=state,
                collectives=((family.id, "send_recv"),),
                site=RedistributeSite.STAGE_CROSSING, crossing=crossing))
        return out

    def _cache_reads(self, analysis: Any) -> Dict[int, "CacheRead"]:
        """Per hop producer whose output is cache-resident data (the
        pedigree `traffic` prices a cache read by: `cache.materialized`,
        `cache.context_sources`), what one step reads of each cached
        tensor behind it: `min(stored(s_kv), s_query · read(s_kv))`
        positions per request — the read law at decode, the whole span a
        fresh prefill wrote — of the tensor's per-rank elements, at its
        cache dtype."""
        program = self.program
        producers = {crossing.producer[0] for crossing in self.crossings}
        if not producers:
            return {}
        edges = cache_mod.Edges(analysis)
        resident = cache_mod.context_sources(edges)
        if not resident:
            return {}
        seeds = cache_mod.cached_node_uids(program)
        tensors: Dict[int, List[Any]] = {}
        for entry in cache_mod.derive(analysis).values.values():
            for tensor in entry.tensors:
                if not tensor.recurrent:
                    tensors.setdefault(tensor.node, []).append(tensor)
        cache_len = cache_mod.workload_dim(analysis, "s_kv")
        queries = cache_mod.workload_dim(analysis, "s_query")
        batch = cache_mod.workload_dim(analysis, "B")
        out: Dict[int, CacheRead] = {}
        for uid in producers:
            landed = cache_mod.materialized(program.by_uid(uid), seeds,
                                            edges=edges)
            if landed is None or landed.uid not in resident:
                continue
            spans = []
            for seed in sorted(resident[landed.uid].seeds):
                if seed not in tensors:
                    raise ValueError(
                        f"{program.where(uid)} crosses a pipeline cut as "
                        f"cache-resident data of {program.where(seed)}, "
                        f"which keeps no cached tensor")
                for tensor in tensors[seed]:
                    positions = minimum(
                        tensor.positions(cache_len),
                        queries * tensor.positions(cache_len, read=True))
                    # A rank's share of the positions is a proportion, not
                    # a declared-exact division: the live length is the
                    # request's and need not be a multiple of the group that
                    # owns the positions. `ratio` keeps it exact in both
                    # domains where `/` on two counts was a float.
                    spans.append((ratio(positions, tensor.positions_over,
                                        empty=0),
                                  tensor.elements, tensor.element_bytes))
            out[uid] = CacheRead(batch=batch, spans=tuple(spans))
        return out

    @staticmethod
    def handed_state(analysis: Any, value: Tuple[int, str]) -> Any:
        """The state `value` is handed to its readers in. A rule that
        distributes nothing records none."""
        if analysis.rule.is_trivial:
            return None
        found = analysis.handed_states.get(value)
        if found is not None:
            return found
        uid, _port = value
        return analysis.port_states.get(value, analysis.states.values.get(uid))

    def repairs(self, analysis: Any, crossing: StageCrossing
                ) -> Tuple[Tuple[Occurrence, Tuple[Any, ...]], ...]:
        """Layer 2: each sink of a hop, with the input transitions the walk
        recorded for it (none when the handed state is what it requires).
        They are pp's no more than they are the walk's: a repair lands on
        its sink's stage."""
        by_input: Dict[Tuple[int, int], List[Any]] = {}
        for record in analysis.redistributes:
            if record.crossing is None:
                by_input.setdefault((record.node, record.input_index),
                                    []).append(record)
        return tuple((sink, tuple(by_input.get((sink.uid, index), ())))
                     for sink, index in crossing.sinks)

    # -- views -------------------------------------------------------------

    def view(self, stage: int) -> "StageView":
        if not 0 <= stage < self.pp:
            raise ValueError(f"stage {stage} of {self.pp}")
        return StageView(self, stage)

    @memo_property
    def _block_stages(self) -> Mapping[int, Stage]:
        """Block ordinal (`Program.blocks`) → the stage evaluating it."""
        index = {id(template): i
                 for i, template in enumerate(self.program.blocks)}
        return {index[id(stage.template)]: stage
                for stack in self.program.walk_stacks for stage in stack.stages
                if isinstance(stage, BlockStage)}

    def block_counts(self, stage: Optional[int]) -> Dict[int, int]:
        """Per block ordinal (`Program.blocks`), how many of its layers run
        on `stage` (all of them when None): a partitioned block its
        iterations there, any other its count when its unit is placed
        there."""
        out: Dict[int, int] = {}
        for index, block in self._block_stages.items():
            template = block.template
            if self.partitioned(block):
                out[index] = sum(1 for i in template.layer_indices
                                 if stage is None
                                 or self.iteration_stage(block, i) == stage)
                continue
            root = template.roots[0].uid
            placed = self.stage(Occurrence(root, template.layer_indices[0]))
            out[index] = (template.count
                          if stage is None or placed == stage else 0)
        return out

    def stored_counts(self, stage: Optional[int]) -> Dict[int, int]:
        """Per block ordinal that holds a looped stack's stored layers (a
        pass-1 invocation, the owner its later passes read), how many of
        those stored layers have an execution of any pass on `stage`. A
        stage loads the modules of the layers it runs, so a stored layer
        whose two passes sit on two stages is resident on both."""
        program = self.program
        out: Dict[int, int] = {}
        for index, block in self._block_stages.items():
            meta = block.template.meta
            if (meta.pass_ != 1 or meta.loop_gate
                    or not self.partitioned(block)):
                continue
            schedule = program.execution_schedule(block.layer_path)
            if not schedule.is_looped:
                continue
            stored = {schedule.executions[o].stored
                      for o in block.template.layer_indices}
            out[index] = sum(
                1 for layer in stored
                if any(stage is None or self.iteration_stage(block, o) == stage
                       for o in schedule.ordinals_of(layer)))
        return out

    def contribution_counts(self, stage: Optional[int],
                            scope: FrozenSet[str]) -> Dict[int, int]:
        """How many of each accounting root of `scope`'s stacks run on
        `stage` (all stages when None)."""
        blocks = self.block_counts(stage)
        out: Dict[int, int] = {}
        for contribution in self.program.contributions:
            node = contribution.node
            if contribution.stack not in scope:
                kept = 0
            elif contribution.block_index is not None:
                kept = blocks[contribution.block_index]
            else:
                placed = self.stage(Occurrence(node.uid, None))
                kept = (int(contribution.count)
                        if stage is None or placed == stage else 0)
            out[node.uid] = kept
        return out

    @memo_property
    def free_reads(self) -> Mapping[int, FrozenSet[int]]:
        """Per stage, the parameters a computed occurrence there reads."""
        out: Dict[int, Set[int]] = {}
        for sink, _index, source, _port in self._occurrence_edges():
            if self.is_free(source.uid):
                out.setdefault(self.stage(sink), set()).add(source.uid)
        return {stage: frozenset(uids) for stage, uids in out.items()}


@dataclass(frozen=True)
class CacheRead:
    """What a pipeline cut carries in one step for a cache-resident value
    (`StagePlacement._cache_reads`): for each of `batch` requests, per
    cached tensor behind the value, `(positions, elements per position on
    this rank, bytes per element)`."""

    batch: Any
    spans: Tuple[Tuple[Any, Any, float], ...]

    def payload(self) -> Tuple[List[Any], float]:
        """The send's typed shape and element width: `[batch, positions,
        elements]` for one tensor; tensors of one width laid end to end,
        `[batch, Σ positions × elements]`, for several."""
        widths = {width for _rows, _elements, width in self.spans}
        if len(widths) != 1:
            raise ValueError(
                f"a cache-resident value crossing a pipeline cut joins "
                f"cached tensors of widths {sorted(widths)}; one send has "
                f"one element width")
        (width,) = widths
        if len(self.spans) == 1:
            ((rows, elements, _width),) = self.spans
            return [self.batch, rows, elements], width
        return [self.batch, sum(rows * elements for rows, elements, _width
                                in self.spans)], width


@dataclass(frozen=True)
class StageView:
    """What one stage's rank runs and holds, three ways over one placement,
    each in its own scope (a set of stack names; `StagePlacement.backbone`
    is the default of each):

    - `rows(timing_scope)`: the compute and comm rows it times;
    - `weights(weight_scope)`: the weights it holds;
    - `cache(cache_scope)`: the cache it keeps.

    No scope widens another."""

    placement: StagePlacement
    stage: int

    def _layer(self, counts: Mapping[int, int]) -> Layer:
        return self.placement.program.multiplicity(counts)

    def describe(self) -> str:
        """`stage 2, layers 30–45 (16 of 61)`: which rank the numbers beside
        it belong to."""
        held = self.placement.ranges[self.stage]
        span = (f"layers {held.start}–{held.stop - 1}" if len(held)
                else "no layers")
        return (f"stage {self.stage}, {span} "
                f"({len(held)} of {self.placement.n_iterations})")

    def rows(self, timing_scope: FrozenSet[str]) -> Layer:
        """How many of each node this stage runs, within `timing_scope`."""
        return self._layer(self.placement.contribution_counts(
            self.stage, timing_scope))

    def cache(self, cache_scope: FrozenSet[str]) -> Layer:
        """How many of each cache-holding node this stage keeps, within
        `cache_scope`."""
        return self._layer(self.placement.contribution_counts(
            self.stage, cache_scope))

    def weights(self, weight_scope: FrozenSet[str]) -> Layer:
        """The weights this stage holds, within `weight_scope`: what runs
        here, plus every parameter it reads and every table it reads through
        `tied_weight` (a residency fact, not an edge), once."""
        placement = self.placement
        program = placement.program
        counts = placement.contribution_counts(self.stage, weight_scope)
        everywhere = placement.contribution_counts(None, weight_scope)
        held = self._layer(counts).values
        resident: Set[int] = set(placement.free_reads.get(self.stage, ()))
        for reader, tied in program.read_aliases.items():
            if held.get(reader):
                resident.update(tied)
        roots = {c.node.uid for c in program.contributions}
        for uid in resident:
            if uid in roots and not counts.get(uid):
                counts[uid] = everywhere[uid]
        # A looped stack's later pass stores nothing and reads its owner's
        # layers: the owner is resident for exactly the stored layers any
        # pass on this stage runs. That count is the answer for a looped
        # owner — it replaces the whole-owner promotion above, which a
        # later pass's tied reads would otherwise have triggered.
        stored = placement.stored_counts(self.stage)
        for contribution in program.contributions:
            index = contribution.block_index
            if (index in stored and contribution.stack in weight_scope):
                counts[contribution.node.uid] = stored[index]
        return self._layer(counts)

    def crossings(self, timing_scope: FrozenSet[str]
                  ) -> Tuple[StageCrossing, ...]:
        """The hops this stage sends: cut `stage`, serving `timing_scope`."""
        return tuple(crossing for crossing in self.placement.crossings
                     if crossing.cut == self.stage
                     and crossing.serves & timing_scope)
