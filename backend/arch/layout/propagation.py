"""Layout rule the evaluator asks at each node: required vs produced `EdgeState`."""
from __future__ import annotations

from ..ops import is_identity_source
from .zone import COMBINE_OP, DISPATCH_OP

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from functools import cached_property
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .parallel_config import ParallelConfig


from backend.arch import dims
from backend.arch.schema.custom_ops import CustomOpSpec
from backend.arch.eval.analysis import TensorRow
from backend.arch.eval.dim import (algebra, exact, require_equal,
                                   require_multiple, same_law, value_of)
from backend.arch.eval.refusal import SpecRefusal
from backend.arch.eval.routed import RoutedLaunch
from backend.arch.eval.rule import Rule
from backend.arch.eval.transition import InputRequirements, KernelGeometry
from backend.arch.eval.values import Shape

from .attention_plan import (
    AttentionPlan,
    HeadRepartitionKVRegion,
    ATTENTION_PLANS,
    AttentionForm,
    head_repartition_kv_cache_region,
)
from .edge import EdgeState
from .axis_use import AxisUse
from .foldlaw import (
    CONSTANT,
    GROUP_INVARIANT,
    HOMOMORPHISM_REGISTRY,
    INDEX_EQUIVARIANT,
    ReductionHomomorphism,
    may_defer_reduction,
)
from .groups import ProcessGroupFamily, same
from .layout import (
    AxisLayout,
    Block,
    Flattened,
    IndivisibleSequence,
    Product,
    Replicated,
    Segmented,
    UniformFiberError,
    Unknown,
    aligned,
    axis_fully_replicated,
    physical_local_extent,
    sharded,
)
from .resolver import families_for, layout_for_symbol, pipeline_family_for
from .transfer_table import (
    CONTRACT,
    CellResult,
    Derived,
    transfer,
)
from .value import (COMPLETE, SUM, Partial,
                    ValueProvenance)
from .workload import ExpectedRoutedLoad
from .close import close_analysis as _close_analysis, close_stage as _close_stage
from backend.arch.eval.storage import is_expert_axis

from .persistent import STORED
from .lower import cache_write_endpoint as _cache_write, lower_diff as _price
from .persistent import physical_artifacts, resident_weights
from .physical import (_output_state, local_dims, physical_shapes,
                       rank_definitions)
from .weight_endpoints import (
    PlacementContext,
    WeightGeometry,
    _eval_extent,
)


@dataclass(frozen=True)
class LogicalAxisOwner:
    """A logical owner whose raw extent is not necessarily physically cut.

    ``bound_per_rank`` distinguishes an input extent the caller already bound
    locally (request B) from a semantic singleton selected out of a global
    extent (sampled position). Both carry ownership in the layout algebra; only
    the former may retain a raw extent greater than one.
    """

    name: str
    layout: AxisLayout
    keep_at_model_output: bool
    bound_per_rank: bool = False

    def can_project(self, extent: Any) -> bool:
        value = value_of(extent)
        if self.bound_per_rank:
            # A whole, non-negative count — a number, or an integer symbol
            # of the workload.
            whole = algebra(value)
            return bool(whole.is_integer and whole.is_nonnegative)
        return value == 1


@dataclass(frozen=True)
class _WeightPlacement:
    """Ordered physical axes plus the unambiguous semantic slot lookup.

    Buffer shapes may repeat one dimension expression on distinct axes (a
    square convolution kernel is the canonical case).  Consumers such as a
    linear address axes by semantic slot, while buffer propagation needs the
    original order; one slot-keyed dictionary cannot represent both facts.

    Checkpoint axes, segment terms and per-axis symbols are the rest of
    the same ``axis_endpoints`` call. Persistent instantiation reads them
    instead of building a second PlacementContext and asking again. Every
    tuple is in the op's weight-shape order, checkpoint axes included;
    ``WeightGeometry.to_checkpoint`` reorders them for the checkpoint.
    """

    axes: Tuple[AxisLayout, ...]
    slots: Mapping[str, AxisLayout]
    ambiguous_slots: frozenset[str] = frozenset()
    checkpoint_axes: Tuple[AxisLayout, ...] = ()
    axis_segments: Tuple[Tuple[str, ...], ...] = ()
    axis_segment_names: Tuple[Tuple[str, ...], ...] = ()
    axis_symbols: Tuple[Tuple[Tuple[str, int], ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))

    def require(self, slot: str, where: str) -> AxisLayout:
        """Look up one semantic weight slot, or name why it cannot be used."""
        if slot in self.ambiguous_slots:
            raise ValueError(
                f"{where}: semantic weight slot {slot!r} occurs "
                "on multiple axes and cannot be addressed by name")
        layout = self.slots.get(slot)
        if layout is None:
            raise ValueError(
                f"{where}: weight slot {slot!r} not among "
                f"declared axes {sorted(self.slots)}")
        return layout


def propagate_step(step: Any) -> Any:
    """The analysis a serving step already ran. Does not evaluate again."""
    return step.analysis


#: Data-input slot names per op, for `may_defer_reduction`'s full slot
#: picture. Only ops whose registered law this rule can actually feed —
#: an op absent here never defers, which is the always-correct default.
@dataclass(frozen=True)
class LayoutPropagation(Rule):
    """The transition rule the walk consults.

    States are `EdgeState`s. Collectives come from the layout/value
    difference. A repeated block closes when its exit state equals the
    state that entered it.
    """

    parallel: Any
    #: Required, by keyword: the base `Rule` declares the attribute (None —
    #: the identity rule walks no named phase), and a plain annotation here
    #: would inherit that None as a default.
    phase: str = field(kw_only=True)
    #: world label -> ProcessGroupFamily, for every world above degree one.
    families: Mapping[str, Any]
    #: The stage's full rank set — what Replicated means relative to.
    universe: frozenset
    #: uid -> {weight slot: AxisLayout}, from the occurrence `weight_axes`
    #: declarations resolved through the phase-3 endpoint path.
    weights: Mapping[int, _WeightPlacement]
    #: uid -> the node's resolved dim environment (the global walk's
    #: `node_dims`), read for geometry factors and node facts only.
    env: Mapping[int, Mapping[str, Any]]
    #: The prefill residual stream's sequence cut (cp).
    resident_sequence_family: Optional[ProcessGroupFamily] = None
    #: The world that owns the residual stream's sequence axis. CP in prefill,
    #: or attention TP when sequence parallelism is selected.
    residual_sequence_family: Optional[ProcessGroupFamily] = None
    #: The decode KV cache's storage contract family (dcp).
    cache_family: Optional[ProcessGroupFamily] = None
    #: Request ownership (DP-attention) is a logical B-axis cut. The phase-4
    #: evaluator still binds B per attention replica, so the layout carries the
    #: ownership now and reconciliation globalizes that one extent by this
    #: family's degree until Binding migrates in phase 5.
    request_family: Optional[ProcessGroupFamily] = None
    #: Expert collection / routing family. None means this graph may contain a
    #: syntactic MoE component but the deployment has no expert partition.
    route_family: Optional[ProcessGroupFamily] = None
    #: Shape-bearing leaf of every cache entry. Prefill CP cuts the
    #: computation of those tensors, not their storage: they leave the
    #: producer whole on the resident family. Cache identity (the
    #: declared child a reader names) may be a wrapper around this leaf.
    cached_producers: FrozenSet[int] = frozenset()
    #: Whole-cache reads whose main-attention K/V pedigree Ulysses computes
    #: on resident rows before the attention entry exchanges that cut for
    #: heads. Structural provenance from the reverse dataflow walk.
    head_repartition_kv_views: FrozenSet[int] = frozenset()
    #: The row-wise invocation kernels between those views and main attention.
    #: Their persistent SSA/memory shape stays whole; their kernel geometry
    #: alone takes the resident sequence cut.
    head_repartition_kv_producers: FrozenSet[int] = frozenset()
    #: uid -> encoder stack (`Stack.is_encoder`) of every encoder node: laid
    #: out by `encoder_view`, where the text model's sequence worlds do not
    #: exist, or by `replicated_encoder_view` (`view_of`).
    encoder_nodes: Mapping[int, str] = field(default_factory=dict)
    #: Encoder stacks whose query heads the attention TP degree does not
    #: divide (`attention_plan.replicated_encoders`): every weight of one is
    #: whole on every rank.
    replicated_encoders: FrozenSet[str] = frozenset()
    #: Meshes (`dims.yaml` `worlds: mesh`) that do not exist for the nodes
    #: this rule places: an axis or weight placed on a world of one is whole.
    #: `cp` in `encoder_view`, whose token axis is its modality's own; `tp`
    #: too in `replicated_encoder_view`.
    absent_meshes: FrozenSet[str] = frozenset()
    #: The batch's routing skew (`Workload.ep_load_factor`): the hottest
    #: expert's rows over the mean's — `rho` in a walk of the workload's
    #: symbols, a number at a point. Dispatch reads it into the routed
    #: rows' evidence (`ExpectedRoutedLoad`).
    expert_load: Any = 1
    #: uid -> the kernel's cut output for a producer published whole.
    _producer_gathers: Dict[int, EdgeState] = field(default_factory=dict)
    #: uid -> the op declaration the walk handed this rule for the node. A
    #: spec-declared op's family and axis tags are read off it by the
    #: handler, the way a registered op's are read off the module tables.
    declarations: Dict[int, Any] = field(default_factory=dict)
    #: Ulysses / pcp / layersplit / DCP machine. Filled from the families when omitted
    #: so test constructors stay short.
    attention: Any = None
    #: The `pp` world (`resolver.pipeline_family_for`): who a stage hands a value
    #: to across a cut. None at one stage.
    pipeline_family: Optional[ProcessGroupFamily] = None
    #: The Program this rule walks, for the structural questions a handler
    #: asks of it (`Program.output_leaf`: which child a component's port
    #: names). Handed in by the composition root (`Deployment.analyze`).
    program: Any = None

    def world_degree(self, world: str) -> int:
        """The deployment's degree of `world` in this walk's phase
        (`ParallelConfig.degree_of_world`); a rule built without a
        deployment distributes nothing."""
        if self.parallel is None:
            return 1
        return self.parallel.degree_of_world(world, phase=self.phase)

    def __post_init__(self) -> None:
        if self.attention is None:
            object.__setattr__(
                self, "attention", AttentionPlan.from_rule(self))

    @classmethod
    def of(cls, parallel: ParallelConfig, phase: str, *,
           cached_producers: FrozenSet[int] = frozenset(),
           encoder_nodes: Mapping[int, str] = MappingProxyType({}),
           replicated_encoders: FrozenSet[str] = frozenset(),
           expert_load: Any = 1,
           program: Any = None,
           ) -> "LayoutPropagation":
        """Build a rule whose node facts are filled by the one evaluator walk."""
        families = families_for(parallel, phase)
        cp_form = (ATTENTION_PLANS.get(
            (str(parallel.cp_scheme), phase))
            if phase == "prefill" else None)
        kv_region = (
            head_repartition_kv_cache_region(program, parallel)
            if (cp_form is AttentionForm.HEAD_REPARTITION
                and parallel.degree_of_world("cp", phase=phase) > 1)
            else HeadRepartitionKVRegion()
        )
        universe = frozenset(range(parallel.per_stage_size))
        cp_degree = parallel.degree_of_world("cp", phase=phase)
        dcp_degree = parallel.degree_of_world("dcp", phase=phase)
        live_label = "cp" if phase == "prefill" else "dcp"
        live_degree = cp_degree if phase == "prefill" else dcp_degree
        resident = None
        cache = families.get("dcp") if dcp_degree > 1 else None
        if live_degree > 1:
            family = families.get(live_label)
            if family is None:
                raise ValueError(
                    f"sequence world {live_label!r} runs at degree "
                    f"{live_degree} but resolved no family")
            if phase == "prefill":
                resident = family
        for refusal in parallel.refusals(phase):
            raise refusal
        residual = resident
        if bool(parallel.sequence_parallel):
            # `refusals` guarantees no resident cut competes for the
            # residual sequence axis and that the family exists.
            residual = families[parallel.SEQUENCE_PARALLEL_WORLD]
        return cls(parallel=parallel, phase=phase, families=families,
                   universe=universe, weights={}, env={},
                   resident_sequence_family=resident,
                   residual_sequence_family=residual,
                   cache_family=cache,
                   request_family=families.get("dp_attention"),
                   route_family=families.get("ep_moe"),
                   cached_producers=cached_producers,
                   head_repartition_kv_views=kv_region.views,
                   head_repartition_kv_producers=kv_region.producers,
                   encoder_nodes=dict(encoder_nodes),
                   replicated_encoders=replicated_encoders,
                   expert_load=expert_load,
                   pipeline_family=pipeline_family_for(parallel),
                   program=program)

    @cached_property
    def encoder_view(self) -> "LayoutPropagation":
        """This rule as an encoder stack sees it (`Stack.is_encoder`).

        An encoder's token axis is its modality's own sequence — patches, audio
        frames — not the text context, so the text model's sequence worlds do
        not exist in it: no context-parallel or sequence-parallel cut of the
        residual, no KV-cache sharding, no attention execution plan (the
        encoder attends over its own tokens whole). What remains is what the
        encoder's own declarations place — tensor-parallel weights — and the
        stage it sits in. The per-node facts (`env`, `weights`,
        `declarations`) are the same maps, filled by the same walk.
        """
        if not self.encoder_nodes:
            return self
        return replace(
            self, resident_sequence_family=None,
            residual_sequence_family=None, cache_family=None,
            encoder_nodes={}, absent_meshes=self.absent_meshes | {"cp"},
            attention=AttentionPlan(
                name=None, form=None, phase=self.phase, universe=self.universe,
                resident=None, cache=None,
                attention_tp=(self.families or {}).get("tp_attention"),
                parallel=self.parallel, families=self.families or {}))

    @cached_property
    def replicated_encoder_view(self) -> "LayoutPropagation":
        """`encoder_view` with no tensor-parallel world either: every weight
        of the stack whole on every rank, attention, MLP and projector alike.
        An encoder whose query heads the attention TP degree does not divide
        runs this way — vLLM runs the tower data parallel rather than split
        its heads unevenly (`vision.is_vit_use_data_parallel`)."""
        view = self.encoder_view
        return replace(
            view, absent_meshes=view.absent_meshes | {"tp"},
            attention=replace(view.attention, attention_tp=None))

    def view_of(self, uid: int) -> "LayoutPropagation":
        """The view node `uid` is laid out in: its encoder stack's, else
        this rule."""
        stack = self.encoder_nodes.get(uid)
        if stack is None:
            return self
        if stack in self.replicated_encoders:
            return self.replicated_encoder_view
        return self.encoder_view

    @property
    def placement_context(self) -> PlacementContext:
        """The stored-axis placement inputs of the nodes this view places."""
        return PlacementContext(
            pc=self.parallel, phase=self.phase, families=self.families,
            scope=self.universe, absent_meshes=self.absent_meshes)

    # -- what the evaluator and handlers call ----------------------------

    def symbol_layout(self, symbol: str, extent: int) -> Optional[AxisLayout]:
        """The declared placement of an axis named by one dim symbol, as the
        nodes this rule places see it: none (replicated) when the symbol's
        world is a text sequence world this view does not have — an encoder's
        `s_ctx` counts its own patches or frames, which no context-parallel
        world cuts."""
        world = dims.world_of(symbol)
        if world is not None and world.mesh in self.absent_meshes:
            return None
        return layout_for_symbol(symbol, extent, self.parallel, self.phase,
                                 dict(self.families))

    @property
    def is_trivial(self) -> bool:
        """Nothing is distributed, so nothing needs deciding."""
        return not self.families

    # -- what an Analysis asks of the rule --------------------------------
    # The per-rank projections `Analysis.local_tensors` /
    # `eval.capacity.weights` delegate to these `Rule` overrides: the answers
    # are layout knowledge, and eval never imports this package.

    def local_tensors(self, analysis: Any) -> Any:
        """The `Analysis.tensors()` join, projected to this rule's rank."""
        local = {
            (row.node_uid, row.tensor_index, row.segment_index): row
            for row in physical_artifacts(analysis)
            if row.surface == STORED
        }
        for uid in analysis.storage.values:
            for row in analysis.tensors_of(uid):
                tensor, count, rep = row.tensor, row.count, row.rep
                # Joined on the row's own tensor AND layer segment, not on
                # its position in the rows: a tensor stored at several
                # formats across its node's layers is several rows at one
                # tensor index, and each takes its own segment's placement —
                # whose `runtime_instances` is already that segment's copies
                # on this rank.
                physical = local.get(
                    (uid, row.tensor_index, row.segment_index))
                if physical is not None:
                    tensor = replace(tensor, shape=tuple(physical.runtime_shape))
                    count = physical.runtime_instances
                yield TensorRow(uid, tensor, rep, count, row.tensor_index,
                                row.segment_index)

    def rank_shapes(self, analysis: Any) -> Any:
        """The walk's shapes projected to this rule's rank
        (`physical.physical_shapes`)."""
        return physical_shapes(analysis)

    def rank_definitions(self, analysis: Any, definitions: Any,
                         global_shape: Any, local_shape: Any, node: Any, *,
                         query_shapes: Any = None, query_node: Any = None,
                         cache: Any = None) -> Dict[str, Any]:
        """A node's laws at this rule's rank (`physical.rank_definitions`)."""
        return rank_definitions(
            definitions, global_shape, local_shape, self, node,
            query_shapes=query_shapes, query_node=query_node, cache=cache)

    def local_axis_segments(self, analysis: Any) -> Dict[Any, Any]:
        """Each stored tensor's axes as this rank holds their partitions
        (`Rule.local_axis_segments`), off the placement itself: a
        `Segmented` axis is one partition per segment, each cut as its own
        layout cuts it; any other axis is one partition, the tensor's global
        extent beside the rank's."""
        out: Dict[Any, Any] = {}
        for row in physical_artifacts(analysis, stored_only=True):
            if row.surface != STORED:
                continue
            layout = row.runtime_layout
            axes = []
            for index, extent in enumerate(row.global_shape):
                axis = layout.axes[index] if layout is not None else None
                if isinstance(axis, Segmented):
                    axes.append(tuple(
                        (size, physical_local_extent(child, size))
                        for _offset, size, child in axis.segments))
                else:
                    axes.append(((extent, row.runtime_shape[index]),))
            out[(row.node_uid, row.tensor_index, row.segment_index)] = tuple(axes)
        return out

    def resident_weights(self, analysis: Any) -> Any:
        """The `eval.capacity.Weights` ledger of this rule's rank."""
        return resident_weights(physical_artifacts(analysis))

    def routed(self, analysis: Any) -> Dict[int, RoutedLaunch]:
        """Each dispatch's boundary at this rule's rank: its token operand
        as the rank holds it and the buffer it writes there — the buffer's
        rows the busiest rank's share of the routed-load evidence
        (`ExpectedRoutedLoad`), exactly as the per-rank shapes read them
        (`physical._local_dims`). A rule that distributes nothing records
        no states, and its rank is the walk's: the identity answer."""
        if self.is_trivial:
            return super().routed(analysis)
        out: Dict[int, RoutedLaunch] = {}
        for uid, shape in analysis.shapes.values.items():
            node = analysis.program.by_uid(uid)
            if node is None or node.op != DISPATCH_OP:
                continue
            where = node.display_path
            token = local_dims(shape.ins[0], analysis.input_states[(uid, 0)],
                                self, f"{where}.in[0]")
            buffer = local_dims(shape.out, _output_state(analysis, uid, "out"),
                                 self, f"{where}.out")
            out[uid] = RoutedLaunch.of(uid, token, buffer)
        return out

    def _prepare_node(
        self,
        node: Any,
        node_dims: Mapping[str, Any],
        stored_tensors: Sequence[Any],
        weight_shape: Sequence[Any],
    ) -> None:
        """Record node-local placement facts during the evaluator walk."""
        env = self.env
        weights = self.weights
        if not isinstance(env, dict) or not isinstance(weights, dict):
            raise TypeError("LayoutPropagation node facts require mutable maps")

        resolved = dict(node_dims)
        previous = env.get(node.uid)
        if previous is not None:
            if resolved and dict(previous) != resolved:
                raise ValueError(
                    f"{node.display_path}: one evaluator walk produced "
                    "different node bindings for the same uid"
                )
            resolved = dict(previous)
        else:
            env[node.uid] = resolved

        context = self.placement_context

        if node.weight_axes is None:
            return

        # Every per-axis record below is in the op's weight-shape order;
        # the checkpoint order is read back through the same geometry.
        geometry = WeightGeometry.of(
            node, tuple(weight_shape), tuple(stored_tensors))
        ordered: List[AxisLayout] = []
        checkpoint_axes: List[AxisLayout] = []
        segments: List[Tuple[str, ...]] = []
        segment_names: List[Tuple[str, ...]] = []
        axis_symbols: List[Tuple[Tuple[str, int], ...]] = []
        per_slot: Dict[str, AxisLayout] = {}
        ambiguous_slots: set[str] = set()
        for slot, (source, contract) in zip(
                geometry.slots, geometry.runtime_axes(), strict=True):
            extent = _eval_extent(source, resolved)
            checkpoint, runtime, terms, names, symbols = (
                context.axis_endpoints(
                    node, source, contract, extent, resolved
                )
            )
            _require_supported_axis(node, slot, runtime)
            ordered.append(runtime)
            checkpoint_axes.append(checkpoint)
            segments.append(terms)
            segment_names.append(names)
            axis_symbols.append(symbols)
            if slot in per_slot:
                ambiguous_slots.add(slot)
                per_slot.pop(slot)
            elif slot not in ambiguous_slots:
                per_slot[slot] = runtime
        weights[node.uid] = _WeightPlacement(
            tuple(ordered), per_slot, frozenset(ambiguous_slots),
            checkpoint_axes=tuple(checkpoint_axes),
            axis_segments=tuple(segments),
            axis_segment_names=tuple(segment_names),
            axis_symbols=tuple(axis_symbols),
        )

    def transition(self, node: Any,
                   arrived: Sequence[Optional[EdgeState]],
                   uses_weight: bool,
                   in_shapes: Sequence[Optional[Sequence[Any]]],
                   *,
                   node_dims: Optional[Mapping[str, Any]] = None,
                   stored_tensors: Sequence[Any] = (),
                   weight_shape: Sequence[Any] = (),
                   out_shape: Sequence[Any] = (),
                   op_spec: Any = None,
                   ) -> Tuple[List[EdgeState], EdgeState]:
        """What this node requires of each input, and what it produces.

        Dispatch is a closed table for registered ops; a spec-declared op is
        dispatched by the family its declaration puts it in
        (`custom_handler`). An op outside both raises rather than passing
        through: silence here is a zero that looks like an answer.
        """
        view = self.view_of(node.uid)
        if view is not self:
            return view.transition(
                node, arrived, uses_weight, in_shapes, node_dims=node_dims,
                stored_tensors=stored_tensors, weight_shape=weight_shape,
                out_shape=out_shape, op_spec=op_spec)
        op_name = node.op or ""
        self._prepare_node(
            node,
            node_dims or {},
            stored_tensors,
            weight_shape,
        )
        if op_spec is not None:
            self.declarations[node.uid] = op_spec
        handler = layout_handler(op_name, op_spec)
        if handler is None:
            raise SpecRefusal(
                f"op {op_name!r} at {node.display_path!r} "
                "has no layout handler — the dispatch is closed; "
                "extend it before running this corpus",
                declaration="op", fix=layout_gap(op_name, op_spec),
                node=node.display_path)
        # A `weight_axes` declaration places a stored tensor; an op that
        # declares no `weight_shape` (primitive or custom — the walk hands
        # this rule the op's own declaration) has nothing to place, and
        # passing it through would be a modeling gap read as an answer.
        if node.weight_axes is not None and not weight_shape:
            raise ValueError(
                f"{node.display_path}: weight_axes declared on op "
                f"{op_name!r}, which declares no stored weight")
        for state in arrived:
            if state is not None and not isinstance(state, EdgeState):
                raise ValueError(
                    f"{node.display_path}: foreign state {state!r} in the "
                    "layout rule")
        for state, shape in zip(arrived, in_shapes, strict=False):
            if state is not None and state.workload is not None:
                self._validate_routed_state(node, state, shape)
        effective = self._settle_arrivals(op_name, node, list(arrived))
        requirements, produced = handler(self, node, effective, in_shapes)
        return self._finalize_cached_producer(
            node, requirements, produced, in_shapes, out_shape)

    def _finalize_cached_producer(self, node: Any, requirements: Any,
                                  produced: Any,
                                  in_shapes: Sequence[Optional[Sequence[Any]]],
                                  out_shape: Sequence[Any]) -> Tuple[Any, Any]:
        """Prefill CP cuts computing a cached tensor, not storing it.

        Each rank projects its own positions; one all_gather publishes the
        axis whole. Window, union, attention and the cache write then find
        it whole. Not gated on the attention plan: Ulysses still
        repartitions Q at the kernel; persistent KV layout is one law.
        """
        if (node.uid not in self.cached_producers
                or not out_shape
                or not isinstance(produced, EdgeState)):
            return requirements, produced
        position = Shape(out=tuple(out_shape), ins=()).index_of_any(
            dims.query_axes())
        if position is None or position >= len(produced.layout):
            return requirements, produced
        family = self.resident_sequence_family
        axis = produced.layout[position]
        if family is None or not (isinstance(axis, Block)
                                  and axis.is_sharded_on(family)):
            return requirements, produced
        if isinstance(requirements, InputRequirements):
            if requirements.geometry is not None:
                raise ValueError(
                    f"{node.display_path}: a cached producer that declares "
                    "its own kernel geometry is not modelled")
            edges, kernels = requirements.edges, requirements.kernels
        else:
            edges = kernels = tuple(requirements)
        published = self.attention.gather_resident_axis(
            node, produced, position, "the cached tensor")
        geometry = KernelGeometry(
            inputs=tuple(
                None if state is None or shape is None
                else self.attention.kernel_local_axes(
                    node, tuple(shape), state, "cached producer input",
                    self.project_local_extent)
                for state, shape in zip(kernels, in_shapes, strict=True)),
            output=self.attention.kernel_local_axes(
                node, tuple(out_shape), produced, "cached producer output",
                self.project_local_extent),
        )
        self._producer_gathers[node.uid] = produced
        return (InputRequirements(edges=tuple(edges), kernels=tuple(kernels),
                                  geometry=geometry),
                published)

    def internal(self, required: Sequence[EdgeState], produced: EdgeState, *,
                 node: Any = None,
                 ) -> List[Tuple[str, EdgeState, Tuple[Tuple[str, str], ...]]]:
        """Collectives that happen inside a node, not on an input edge.

        Ulysses restore, MoE dispatch / combine, and the cached-producer
        gather.
        """
        op_name = node.op if node is not None else None
        if node is not None and node.uid in self._producer_gathers:
            entered = self._producer_gathers.pop(node.uid)
            return [("out", entered,
                     ((self.resident_sequence_family.id, "all_gather"),))]
        if op_name == DISPATCH_OP and self.route_family is not None:
            if len(required) != 2 or self.route_owner_inclusion(required[0]):
                return []
            return [("out", required[0],
                     ((self.route_family.id, "all_to_all"),))]
        if op_name == COMBINE_OP and self.route_family is not None:
            if len(required) != 2 or self.route_owner_inclusion(required[1]):
                return []
            return [("in", required[0],
                     ((self.route_family.id, "all_to_all"),))]
        return self.attention.ulysses_exit(required, produced, node)

    def _settle_arrivals(self, op_name: str, node: Any,
                         states: List[Optional[EdgeState]]
                         ) -> List[Optional[EdgeState]]:
        """Arriving partials that cannot defer through this op finalize.

        The finalized form becomes the required state for that input, so
        the walk materializes the all_reduce on exactly this edge — the
        residual add under dense TP is the canonical case. The default is
        `requires_complete`, which is always correct: it can only place the
        reduction earlier than the framework's real schedule.
        """
        pending = [s is not None and s.carries_partial() for s in states]
        if not any(pending):
            return states
        residual = self._residual_sequence_add(op_name, states, pending)
        if residual is not None:
            return residual
        law = HOMOMORPHISM_REGISTRY.get(op_name)
        if law is not None and self._defers(node, law, states):
            return states
        return [s.finalized() if carries else s
                for s, carries in zip(states, pending, strict=True)]

    def _residual_sequence_add(
            self, op_name: str, states: Sequence[Optional[EdgeState]],
            pending: Sequence[bool],
            ) -> Optional[List[Optional[EdgeState]]]:
        """Dense TP residual: the sequence-cut operand finalizes the SUM debt."""
        if (op_name != "add"
                or self.residual_sequence_family is None
                or sum(pending) != 1):
            return None
        debt_index = pending.index(True)
        debt = states[debt_index]
        complete = next(
            (state for index, state in enumerate(states)
             if index != debt_index and state is not None
             and state.value is None),
            None,
        )
        family = self.residual_sequence_family
        if complete is None or debt.value.reducer != SUM \
                or not same(debt.value.group, family) \
                or len(debt.layout) != len(complete.layout):
            return None
        differences = [
            axis for axis, (got, need) in enumerate(zip(
                debt.layout, complete.layout, strict=True
            )) if got != need
        ]
        if differences != [1] \
                or not isinstance(debt.layout[1], Replicated) \
                or not isinstance(complete.layout[1], Block) \
                or not same(complete.layout[1].group, family):
            return None
        out = list(states)
        out[debt_index] = EdgeState(
            complete.layout, None, debt.workload, debt.provenance,
        )
        return out

    def _defers(self, node: Any, law: ReductionHomomorphism,
                states: Sequence[Optional[EdgeState]]) -> bool:
        # The law names its own operands; a node whose arity it does not
        # describe, or with an unresolved input, does not defer. A
        # commutative law may name them in any order (a pending sum at
        # either operand of `mul`); one naming that holds suffices.
        if any(s is None for s in states):
            return False
        facts = frozenset(
            {"bias_absent"} if self._bias_absent(node) else ())
        partials = [state.value for state in states
                    if state is not None and state.value is not None]
        pending_group = partials[0].group if partials else None
        return any(
            self._defers_as(node, law, slots, states, facts, pending_group)
            for slots in law.namings(len(states)))

    def _defers_as(self, node: Any, law: ReductionHomomorphism,
                   slots: Sequence[str],
                   states: Sequence[EdgeState], facts: frozenset,
                   pending_group: Optional[ProcessGroupFamily]) -> bool:
        inputs = tuple(
            (slot, state.value if state.value is not None else COMPLETE)
            for slot, state in zip(slots, states, strict=True))
        by_slot = dict(zip(slots, states, strict=True))

        def coefficient_ok(slot: str, cls: str) -> bool:
            if cls in (CONSTANT, INDEX_EQUIVARIANT):
                return True
            if cls == GROUP_INVARIANT:
                # A real coefficient operand is invariant when none of its
                # static cuts varies along the pending group. An implicit
                # weight follows the same rule: a TP-head weight is invariant
                # over an orthogonal DCP merge even though it is not globally
                # replicated.
                operand = by_slot.get(slot)
                if operand is not None:
                    return (operand.value is None
                            and self._state_group_invariant(
                                operand, pending_group))
                placement = self.weights.get(node.uid)
                return placement is not None and all(
                    self._axis_group_invariant(axis, pending_group)
                    for axis in placement.axes)
            return False

        return may_defer_reduction(law, inputs, coefficient_ok, facts)

    def _bias_absent(self, node: Any) -> bool:
        bias = (self.env.get(node.uid) or {}).get("bias")
        return bias is None or not value_of(bias)

    # -- shared pieces ----------------------------------------------------

    def replicated(self) -> Replicated:
        return Replicated(scope=self.universe)

    def request_owner(self) -> Optional[LogicalAxisOwner]:
        """Who owns the batch axis: the request (DP-attention) family, whose
        cut is a label, not a division — `B` is already one rank's extent
        when the world says so (`dims.World.bound_per_rank`)."""
        if self.request_family is None:
            return None
        world = dims.world_of("B")
        return LogicalAxisOwner(
            "request_B", sharded(self.request_family), True,
            bound_per_rank=bool(world is not None and world.bound_per_rank))

    def logical_axis_owner(self, extent: Any) -> Optional[LogicalAxisOwner]:
        """Return the declared logical ownership cut for a tracked axis.

        Logical ownership is distinct from physically slicing the raw bound
        extent.  Declarations live here so generic shape transitions can ask
        the contract without learning about a particular graph or op.
        """
        owners = {
            "B": self.request_owner(),
            # The sampled position lives on whichever family owns the residual
            # stream's sequence axis — `resident_sequence_family` under prefill
            # CP, but `residual_sequence_family` (= tp_attention) under
            # sequence parallelism, where resident is None. Keying this on the
            # resident family alone left the sampled singleton unowned under SP,
            # so a 1-position output head fell through to a physical `S / tp`
            # cut and raised "extent 1 not divisible by tp". The residual
            # family is the resident family whenever SP is off, so this covers
            # both. (`lm_head` in prefill computes only the last position — the
            # sample view narrows to s_sampled=1; that singleton is owned, not
            # cut, until a boundary materializes its collective.)
            "s_sampled": LogicalAxisOwner(
                "sampled_position",
                sharded(self.residual_sequence_family),
                False,
            ) if self.residual_sequence_family is not None else None,
        }
        is_exact = getattr(extent, "is_", None)
        if not callable(is_exact):
            return None
        for name, owner in owners.items():
            if owner is not None and is_exact(name):
                return owner
        return None

    def project_local_extent(self, extent: Any, layout: AxisLayout) -> Any:
        """Project one tracked extent under the logical-owner contract.

        Ownership is checked before ordinary block arithmetic. A per-rank
        bound extent can be divisible by its owner group while still already
        being local; dividing first would silently apply the same deployment
        twice.
        """
        value = value_of(extent)
        if same_law(value, 0):
            # An empty tensor everywhere; one empty at some points only is
            # projected below, and comes to zero there.
            return 0
        owner = self.logical_axis_owner(extent)
        if owner is not None and owner.layout == layout:
            if not owner.can_project(extent):
                raise ValueError(
                    f"logical axis owner {owner.name!r} cannot project "
                    f"extent {value!r}"
                )
            return value
        return physical_local_extent(layout, value)

    @staticmethod
    def value_provenance(node: Any, **operand_ports: int
                          ) -> Optional[ValueProvenance]:
        """Bind semantic operand roles to this node's resolved input edges."""
        refs = tuple(node.inputs_obj)
        if not refs:
            # Direct transition tests may exercise the algebra without a
            # resolved Program node. No lineage is better than invented
            # lineage; an online lowering that needs it will refuse later.
            return None
        operands = []
        for role, port in operand_ports.items():
            ref = refs[port] if 0 <= port < len(refs) else None
            if ref is None or ref.src is None:
                raise ValueError(
                    f"{node.display_path}: provenance role {role!r} has no "
                    f"resolved input at port {port}")
            operands.append((role, ref.src.uid))
        return ValueProvenance(node.uid, tuple(operands))

    def model_input_state(self, name: str, shape: Sequence[Any], *,
                          encoder: bool) -> EdgeState:
        """Where model input `name`, `[batch, rows, width?]`, lives as it
        arrives.

        Each axis by its place in that structure: the batch is the request
        owner's (`request_owner`), the rows are the residual stream's sequence
        cut of the view that reads them, and a row's width is whole. An
        encoder input is read in `encoder_view`, where the text model's
        sequence worlds do not exist, so its rows arrive whole.
        """
        view = self.encoder_view if encoder else self
        owner = view.request_owner()
        batch = owner.layout if owner is not None else view.replicated()
        rows: AxisLayout = view.replicated()
        family = view.residual_sequence_family
        if family is not None:
            require_multiple(
                shape[1], family.group_size,
                lambda extent: IndivisibleSequence(
                    name, extent, family.id, family.group_size))
            rows = sharded(family)
        width = tuple(view.replicated() for _ in shape[2:])
        return EdgeState((batch, rows, *width))

    def _axis_group_invariant(
            self, axis: AxisLayout,
            group: Optional[ProcessGroupFamily]) -> bool:
        if group is None:
            return True
        for owners in self._axis_owner_cells(axis):
            for rank in owners:
                if not frozenset(group.group_of(rank)) <= owners:
                    return False
        return True

    def _axis_owner_cells(self, axis: AxisLayout) -> Tuple[frozenset, ...]:
        """Owner sets for every structurally distinct shard of one axis.

        A Block pattern repeats over every group in its family, so one logical
        shard is the union of the matching replica cell from every group. This
        is the owner-set form needed by FoldLaw coefficient invariance and the
        routed seam; group orthogonality alone loses replicated-cell inclusion.
        """
        if isinstance(axis, Replicated):
            return (axis.scope,)
        if isinstance(axis, Block):
            return tuple(
                frozenset(group[coordinate]
                          for group in axis.group.groups
                          for coordinate in replica_cell)
                for replica_cell in axis.replica_cells)
        if isinstance(axis, Segmented):
            return tuple(
                owners
                for _offset, _extent, child in axis.segments
                for owners in self._axis_owner_cells(child))
        if isinstance(axis, Product):
            cells = (frozenset(axis.cuts[0].group.ranks()),)
            for cut in axis.cuts:
                cells = tuple(
                    left & right
                    for left in cells
                    for right in self._axis_owner_cells(cut)
                    if left & right)
            return cells
        if isinstance(axis, Flattened):
            cells = (frozenset(self.universe),)
            for factor in axis.factors:
                cells = tuple(
                    left & right
                    for left in cells
                    for right in self._axis_owner_cells(factor)
                    if left & right)
            return cells
        raise TypeError(axis)

    def _state_group_invariant(
            self, state: EdgeState,
            group: Optional[ProcessGroupFamily]) -> bool:
        return all(self._axis_group_invariant(axis, group)
                   for axis in state.layout)

    def route_owner_inclusion(self, token_state: EdgeState) -> bool:
        """All route destinations already own each token exactly when the
        token-domain axes are invariant while a rank moves inside the EP
        group. This is owner inclusion in closed form for the symmetric Block
        layouts: same/refining cuts fail; orthogonal cuts are preserved."""
        if self.route_family is None:
            return True
        if len(token_state.layout) < 2:
            raise ValueError("route operand has no (B, s_ctx) axes")
        return all(self._axis_group_invariant(axis, self.route_family)
                   for axis in token_state.layout[:2])

    def route_preserved_cuts(
            self, token_state: EdgeState) -> Tuple[Block, ...]:
        """Static token cuts invariant inside a route group survive flatten."""
        cuts: List[Block] = []
        for axis in token_state.layout[:2]:
            if isinstance(axis, Replicated):
                continue
            if isinstance(axis, Block):
                candidates = (axis,)
            elif isinstance(axis, Product):
                candidates = axis.cuts
            else:
                raise ValueError(
                    "route token axes support only Replicated, Block, or "
                    f"Product layouts, got {axis!r}")
            for cut in candidates:
                if self._axis_group_invariant(cut, self.route_family):
                    if cut.pad_to is not None:
                        raise ValueError(
                            "a padded token cut cannot be preserved on the "
                            "routed-rows axis")
                    cuts.append(cut)
        return tuple(cuts)


    def _validate_routed_state(self, node: Any, state: EdgeState,
                               shape: Optional[Sequence[Any]]) -> None:
        """A routed tensor carries one expert axis — the axis whose identity
        is `n_expert_copies` (`storage.is_expert_axis`), never a position —
        placed the way the routed-load evidence says (cut over the route
        group, or whole with no expert parallelism), and right after it the
        per-expert rows, whose physical extent after every preserved static
        cut is the evidence's expected mean. This catches a lost or
        reordered axis at the first consumer instead of letting `combine`
        read an unrelated axis as experts or rows.
        """
        if not shape or len(shape) != len(state.layout):
            raise ValueError(
                f"{node.display_path}: a routed tensor's shape and layout "
                "disagree in rank")
        experts = [index for index, extent in enumerate(shape)
                   if is_expert_axis(extent)]
        if len(experts) != 1 or experts[0] + 1 >= len(shape):
            raise ValueError(
                f"{node.display_path}: a routed tensor carries exactly one "
                f"expert axis (`n_expert_copies`) followed by its rows; "
                f"{list(shape)!r} has {len(experts)}")
        axis = experts[0]
        workload = state.workload
        expected_axis = (workload.expert_shard_axis
                         if workload.expert_shard_axis is not None
                         else self.replicated())
        self.align(node, state.layout[axis], expected_axis, "expert axis")
        row = state.layout[axis + 1]
        if isinstance(row, Replicated):
            shards = 1
        elif isinstance(row, Block) and row.pad_to is None:
            shards = row.num_shards
        elif isinstance(row, Product) and all(
                cut.pad_to is None for cut in row.cuts):
            shards = 1
            for cut in row.cuts:
                shards *= cut.num_shards
        else:
            raise ValueError(
                f"{node.display_path}: routed rows axis has unsupported "
                f"layout {row!r}")
        physical_rows = exact(shape[axis + 1]) / shards
        expected = workload.expected_rows_per_expert
        require_equal(
            physical_rows, expected,
            lambda at: ValueError(
                f"{node.display_path}: routed rows must follow the expert "
                f"axis; layout/shape imply {at(physical_rows)} rows per "
                f"expert but ExpectedRoutedLoad records {at(expected)}"))

    @staticmethod
    def produced_state(node: Any, layout: Sequence[AxisLayout],
                        value: Optional[Partial],
                        workload: Optional[ExpectedRoutedLoad]) -> EdgeState:
        """Construct a result while naming routed/static diagonal failures."""
        try:
            return EdgeState(tuple(layout), value, workload)
        except UniformFiberError as exc:
            if workload is not None:
                raise ValueError(
                    f"{node.display_path}: preserved routed-row cut "
                    "conflicts with a tensor-parallel cut on another axis "
                    "(diagonal ownership has no tensor boundary)") from exc
            raise

    def declaration(self, node: Any) -> Any:
        """The op declaration the walk handed this rule for `node`."""
        spec = self.declarations.get(node.uid)
        if spec is None:
            raise ValueError(
                f"{node.display_path}: the walk handed no op declaration for "
                f"{node.op!r}; a spec-declared op is laid out by its "
                "declaration, not by its name")
        return spec

    def weight_batch_axes(self, node: Any, core: int) -> Tuple[AxisLayout, ...]:
        """The placements of a weight's axes ahead of its last `core` —
        the batch axes a weight declared with leading axes carries
        (`ops.weight_leading_axes`). Empty for the op's own weight."""
        placement = self.weights.get(node.uid)
        if placement is None:
            return ()
        return tuple(placement.axes[:max(len(placement.axes) - core, 0)])

    def weight(self, node: Any, slot: str) -> AxisLayout:
        placement = self.weights.get(node.uid)
        if placement is None:
            raise ValueError(
                f"{node.display_path}: weight-bearing op with no "
                "weight_axes declaration — occurrence declarations are the "
                "only placement source (DECLARATION_MISSING)")
        return placement.require(slot, node.display_path)

    def sole(self, node: Any, states: Sequence[Optional[EdgeState]]
              ) -> Tuple[int, EdgeState]:
        """The one activation state arriving at a single-input law.

        An edge out of an IDENTITY SOURCE (`OpSpec.is_identity_source` —
        `parameter_tensor`, `parameter_vector`) does not count. What
        arrives there is a weight, and a weight's placement is declared by
        `weight_axes` on the node that reads it, not propagated as the
        activation's edge state. `mxfp4_fused_linear` takes gpt-oss's
        expert bias as a second operand and is still a one-activation law;
        so is a sink softmax, and so is any bias `add` written that way.
        """
        refs = tuple(node.inputs_obj)
        activations, weights = [], []
        for i, state in enumerate(states):
            if state is None:
                continue
            src = refs[i].src if i < len(refs) else None
            target = (weights if src is not None and is_identity_source(getattr(src, "op", ""))
                      else activations)
            target.append((i, state))
        # An activation wins when there is one. Where there is none, the
        # node's only operand is the parameter — `neg_exp` over Mamba's
        # `A_log` is a whole op derived from a weight — and that state is
        # the one the law follows. Both are "the one arriving state"; the
        # distinction only matters when an op takes both.
        known = activations or weights
        if len(known) != 1:
            raise ValueError(
                f"{node.display_path}: expected exactly one arriving "
                f"state, got {len(known)} of {len(states)} inputs")
        return known[0]

    def cell(self, node: Any, what: str, result: CellResult) -> Derived:
        return result.require(what, node.display_path)

    def contract(self, node: Any, axis: AxisLayout, use: AxisUse,
                 what: str) -> Optional[Partial]:
        """Reduce along `axis`. Replicated is free; otherwise a Partial(sum).

        A plan-owned or unregistered statistic has no generic accumulator
        and must be handled by its execution plan.
        """
        if not isinstance(axis, Replicated) and \
                not use.admits_deferred_reduction:
            raise NotImplementedError(
                f"{node.display_path}: sharded {what} is owned by "
                f"execution plan {use.plan or '<none>'} / statistic "
                f"{use.param!r} and has no generic accumulator")
        cell = self.cell(node, what,
                         transfer(CONTRACT, (axis,), reducer=SUM))
        pending = cell.pending_reduction
        return (pending.with_provenance(node.uid)
                if pending is not None else None)

    def whole_axis(self, node: Any, state: EdgeState, index: int,
                   use: Optional[AxisUse] = None, what: str = ""
                   ) -> EdgeState:
        """Require axis `index` whole. The gather is priced by the walk.

        An axis_use tag with a `plan` belongs to that plan, not to a
        gather. `use` may be None when the whole-axis requirement is the
        consumer's own law (a KV union reading whole positions) rather
        than a tagged coupling.
        """
        axis = state.layout[index]
        if axis_fully_replicated(axis):
            return state
        if use is not None and use.plan:
            raise NotImplementedError(
                f"{node.display_path}: sharded {what} needs the "
                f"{use.plan!r} execution plan's online form")
        fixed = list(state.layout)
        fixed[index] = self.replicated()
        return EdgeState(
            tuple(fixed), state.value, state.workload, state.provenance)

    def align(self, node: Any, left: AxisLayout, right: AxisLayout,
              what: str) -> None:
        aligned(left, right, where=node.display_path, what=what)

    def lower_diff(self, arrived: EdgeState, required: EdgeState, *,
                   node: Any = None, input_index: Optional[int] = None,
                   ) -> Tuple[Tuple[str, str], ...]:
        return _price(self, arrived, required, node=node,
                      input_index=input_index)

    def close_stage(self, state: EdgeState, *,
                    interface: Optional[EdgeState] = None,
                    stage_uid: Optional[int] = None,
                    ) -> Optional[Tuple[EdgeState,
                                        Tuple[Tuple[str, str], ...]]]:
        return _close_stage(self, state, interface=interface,
                            stage_uid=stage_uid)

    def close_analysis(self, analysis: Any) -> Any:
        return _close_analysis(self, analysis)

    def history_state(self, entry: EdgeState) -> EdgeState:
        return _geometry.history_of(self, entry)

    def cache_write_endpoint(self, node: Any, arrived: EdgeState, shape: Any
                             ) -> Tuple[EdgeState,
                                        Tuple[Tuple[str, str], ...]]:
        return _cache_write(self, node, arrived, shape)


def _require_supported_axis(node: Any, slot: str, layout: AxisLayout) -> None:
    """Declared weight axes are Replicated, Block, or Segmented of those."""
    if isinstance(layout, Unknown):
        raise ValueError(
            f"{node.display_path} weight axis {slot!r}: Unknown layout")
    if isinstance(layout, (Replicated, Block)):
        return
    if isinstance(layout, Segmented):
        for _offset, _extent, child in layout.segments:
            _require_supported_axis(node, slot, child)
        return
    raise ValueError(
        f"{node.display_path} weight axis {slot!r}: "
        f"{type(layout).__name__} weight layouts (nested cuts) are outside "
        "scope")


from . import attention as _attention
from . import geometry as _geometry
from . import linear as _linear
from . import moe as _moe
from . import recurrent as _recurrent

#: op -> transition handler for REGISTERED ops. Components are not in this
#: table: the walk uses the last node's state inside the component. Nor are
#: spec-declared ops: `custom_handler` dispatches them by family.
_HANDLERS = {
    "embedding": _linear.embedding,
    "hash_embedding": _linear.hash_embedding,
    "linear": _linear.linear,
    "linear_param": _linear.linear,
    "linear_external_weight": _linear.linear_external_weight,
    "per_head_linear": _linear.per_head_linear,
    "sum_reduce": _linear.sum_reduce,
    "token_pool_compressor": _linear.unary,
    "hc_split_sinkhorn": _moe.hc_split_sinkhorn,
    "hc_mix": _moe.hc_mix,
    "conv1d": _recurrent.conv1d,
    "conv2d": _recurrent.conv2d,
    "mamba2_scan": _recurrent.mamba2_scan,
    "gated_delta_rule": _recurrent.gated_delta_rule,
    "headwise_gate": _recurrent.headwise_gate,
    "rmsnorm": _linear.rmsnorm,
    "rmsnorm_unscaled": _linear.rmsnorm_unscaled,
    "l2norm": _linear.l2norm,
    # Every output channel mixes the whole trailing axis and there is no
    # affine: the distributed dependency of an unscaled norm.
    "hadamard_transform": _linear.rmsnorm_unscaled,
    "layernorm": _linear.layernorm,
    "softmax": _attention.softmax,
    "softmax_with_sink": _attention.softmax_with_sink,
    "swa_softmax": _attention.softmax,
    "swa_softmax_with_sink": _attention.softmax_with_sink,
    "swa_softmax_key0_bias": _attention.softmax_with_sink,
    "dsa_softmax": _attention.softmax,
    "softmax_act": _linear.normalize_axis,
    "normalize_sum_one": _linear.normalize_axis,
    "rope": _linear.rope,
    "rope_2d": _linear.rope,
    "xdrope": _linear.rope,
    "rope_emb": _linear.rope_emb,
    "rope_emb_inverse": _linear.rope_emb,
    "mrope": _linear.rope_emb,
    "silu": _linear.unary,
    "gelu": _linear.unary,
    "gelu_tanh": _linear.unary,
    "relu": _linear.unary,
    "relu_squared": _linear.unary,
    "add_const": _linear.unary,
    "clamp": _linear.unary,
    "sigmoid": _linear.unary,
    "softplus": _linear.unary,
    "sqrt": _linear.unary,
    "signed_sqrt": _linear.unary,
    "exp": _linear.unary,
    "tanh": _linear.unary,
    "neg_exp": _linear.unary,
    "situ": _linear.unary,
    "scale_const": _linear.unary,
    "mul": _linear.elementwise,
    "add": _linear.elementwise,
    "accumulate": _linear.unary,
    "attn_res_combine": _attention.attn_res_combine,
    "stack_init": _geometry.stack_init,
    "stack_append": _geometry.stack_append,
    "position_table_add": _attention.position_table_add,
    "attention_score": _attention.attention_score,
    "swa_attention_score": _attention.attention_score,
    "dsa_attention_score": _attention.dsa_attention_score,
    "attention_apply": _attention.attention_apply,
    "swa_attention_apply": _attention.attention_apply,
    "dsa_attention_apply": _attention.attention_apply,
    "indexer_logits": _attention.indexer_logits,
    "dsa_topk_mask": _attention.dsa_topk_mask,
    "topk": _moe.topk,
    "gather": _moe.gather,
    "group_top2_sum": _moe.group_top2_sum,
    "group_mask_apply": _moe.group_mask_apply,
    "group_max": _moe.group_top2_sum,
    "hash_route": _moe.hash_route,
    DISPATCH_OP: _moe.permute,
    COMBINE_OP: _moe.combine,
    "transpose": _geometry.transpose,
    "broadcast": _geometry.broadcast,
    "concat": _geometry.concat,
    "scatter_rows_replace": _geometry.scatter_rows,
    "scatter_rows_add": _geometry.scatter_rows,
    "kv_stream_concat": _geometry.kv_stream_concat,
    "sliding_window_kv": _geometry.sequence_window,
    "kv_cache_view": _geometry.sequence_window,
    "step_kv": _geometry.sequence_window,
    "view_split": _geometry.view_split,
    "view_merge": _geometry.view_merge,
    "spatial_merge": _geometry.spatial_merge,
    "patchify": _geometry.media_rows,
    "pad_rows": _geometry.media_rows,
    "group_items": _geometry.media_rows,
    "unpad": _geometry.media_rows,
    "slice": _geometry.slice_axis,
    "one_hot": _geometry.one_hot,
    "shift": _geometry.shift_axis,
    "sample": _linear.unary,  # layout pass-through; extent change is the DimMap
    "identity": _linear.unary,
    "rotary_inv_freq_buffer": _linear.buffer,
    "parameter_vector": _linear.buffer,
    "parameter_tensor": _linear.buffer,
}


def family_handler(family: Optional[str]):
    """The transition handler of a layout family (`ops_base.LAYOUT_FAMILIES`):
    a `[Out, In]` weight is a linear map, a recurrent state a head-wise
    recurrence. Any other family — a custom op's `geometry` — has no generic
    law, and a node of it reaching the walk is a modeling gap."""
    if family == "linear":
        return _linear.custom_linear
    if family == "recurrent":
        return _recurrent.headwise_recurrence
    return None


def custom_handler(spec: CustomOpSpec):
    """The handler a spec-declared op follows, from its declaration
    (`CustomOpSpec.layout_family`)."""
    return family_handler(spec.layout_family)


def layout_handler(op_name: str, op_spec: Any = None):
    """The transition handler `transition` dispatches `op_name` to, or None.

    A registered op with a hand-written law is looked up in the closed
    table; any other op — a spec-declared one, or a registered one an op
    proposal added — by the layout family its declaration states. This is
    the dispatch the walk runs and the one validate's dispatch scan asks, so
    the two cannot disagree about which ops a deployment can lay out."""
    handler = _HANDLERS.get(op_name)
    if handler is None and op_spec is not None:
        handler = family_handler(op_spec.layout_family)
    return handler


def layout_gap(op_name: str, op_spec: Any = None) -> Optional[str]:
    """Why the walk cannot lay `op_name` out and what a spec writes
    instead; None when `layout_handler` resolves it."""
    if layout_handler(op_name, op_spec) is not None:
        return None
    if isinstance(op_spec, CustomOpSpec):
        return (
            f"custom op {op_name!r} is neither a linear map (one `[Out, In]` "
            "weight contracting its operand's last axis, `shape_kind: "
            "axis_replace`) nor a recurrence (`carries_recurrent_state: "
            "true` with `axis_use`), the two custom-op families the layout "
            "can place; every other custom op has no layout law. Express "
            "the step with registered ops (a view such as `slice` / "
            "`view_split` / `broadcast`, a pointwise or elementwise op, a "
            "norm), or declare it in one of those two families")
    if op_spec is not None and op_spec.is_comm_op:
        return (
            f"op {op_name!r} is a collective; collectives come from the "
            "layout, never from the spec graph. Remove the node and wire its "
            "consumer to its producer")
    return (
        f"op {op_name!r} has no layout law: no hand-written handler and no "
        "declared layout family. Express the step with registered ops that "
        "have one; an op proposal declares `layout_family=\"linear\"` (one "
        "`[Out, In]` weight) or `layout_family=\"recurrent\"` (a head-wise "
        "recurrence, with its `axis_use`) on its OpSpec, and a spec "
        "`custom_ops` entry declares the same families by its weight or "
        "`carries_recurrent_state`")
