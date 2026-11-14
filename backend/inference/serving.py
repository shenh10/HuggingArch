"""The serving layers a cell's walk is read through (`ServingLayers`):
built once per framework binding, pipeline stage and timed stacks by the
cell that owns them (`step_eval.Cell.serving`), off its walk and per-rank
projections — the graph rewrite, communication nodes and framework fusion,
every layer following each rewrite with its own combine law. Below the
cell, so the cell holds its one builder (`build_serving_layers`); the
reading at a point and the roofline rows are `pipeline`'s.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.analysis import Point
from backend.arch.eval.binding import STEP_LAWS, Definition
from backend.arch.eval.dim import same_law, value_of
from backend.arch.eval.flops import merge_bindings
from backend.arch.eval.fusion import (
    Framework,
    FusedProgram,
    fusion_matches,
    kept_rows,
    materialize_tree,
    rebase_shapes,
    union_consts,
)
from backend.arch.eval.layer import Layer, Rewrite
from backend.arch.eval.access import LeafAccess, routed_launches
from backend.arch.eval.values import Cost, Shape
from backend.arch.layout.physical import physical_kernel_shapes
from backend.arch.layout.physical_cost import PhysicalProjection
from backend.arch.layout.stages import StagePlacement
from backend.arch.spec_tree import SpecIRTree

from .framework_bindings import FrameworkBinding
from .fusion_plans import FUSION_SCHEMES
from .materialize_cast import (CastGraft, cast_facts, exchange_widths,
                              materialize_casts)
from .benchmark_identity import scale_dtype_to_bpe
from .row_roofline import LawBook


@dataclass(frozen=True)
class ServingRewrite:
    """Comm injection, then the activation casts, then framework fusion.

    Pipeline and the activation peak both start from this stretch.
    Each still rebases its own layers (cost, access, instances).
    """

    pre_program: Any
    program: Any
    rewrite: Rewrite
    materialized: Any
    #: The casts grafted between comm and fusion (`materialize_casts`); the
    #: pre-fusion program is its. Empty on a deployment whose matmuls all
    #: eat their edges as they arrive.
    casts: CastGraft
    fusion_matches: Tuple[Mapping[str, Any], ...]


def rewrite_serving_graph(
    analysis: Any,
    *,
    framework_binding: Optional[FrameworkBinding] = None,
    stage: Any = None,
    timing_scope: Optional[FrozenSet[str]] = None,
    local_shapes: Any = None,
) -> ServingRewrite:
    """`analysis → materialize_comm → materialize_casts → Framework.rewrite`.

    Peak and pipeline call this once each, with their own analysis
    (different batch/seq). The binding is the fact that says which
    kernels fire; there is no second matcher.

    `local_shapes` is the walk's per-rank kernel shape layer
    (`physical_kernel_shapes`), read by fusion where a launch takes in a
    cast across views (`fusion.ViewChains`): a scale group is the rank's.
    Derived here when the caller holds none.
    """
    from backend.arch.layout.materializer import materialize_comm_tree

    materialized = materialize_comm_tree(
        analysis,
        stage=stage,
        timing_scope=timing_scope,
    )
    casts = materialize_casts(analysis, materialized)
    # The graft can change what a collective moves (the exchange after a
    # MoE dispatch carries the quantised tokens): its payload is read again
    # off the grafted graph, so the wire and the HBM sides agree.
    materialized = materialized.with_payload_widths(
        exchange_widths(analysis, casts, materialized))
    pre_program = casts.program
    if framework_binding is None:
        program, rewrite = pre_program, Rewrite()
    else:
        program, rewrite = Framework.of(
            framework_binding, FUSION_SCHEMES).rewrite(
                pre_program, analysis,
                (physical_kernel_shapes(analysis) if local_shapes is None
                 else local_shapes), casts.rewrite)
    # What a cast on a model input reads is its first reader's input
    # (`CastGraft.rewrite`), across fusion as before it.
    rewrite = casts.rewrite.then(rewrite)
    matches = fusion_matches(pre_program, program, rewrite, FUSION_SCHEMES)
    return ServingRewrite(
        pre_program=pre_program,
        program=program,
        rewrite=rewrite,
        materialized=materialized,
        casts=casts,
        fusion_matches=matches,
    )


def _on_program(layer: Layer[Any], program: Any, extra: Layer[Any] | None = None
                ) -> Layer[Any]:
    values = dict(layer.values)
    if extra is not None:
        overlap = set(values) & set(extra.values)
        if overlap:
            raise ValueError(
                f"layer extension collides on UIDs {sorted(overlap)!r}"
            )
        values.update(extra.values)
    return Layer(program, values)


def _same(parts: Tuple[Any, ...]) -> Any:
    if not parts:
        raise ValueError("cannot combine an empty layer origin")
    first = parts[0]
    if any(value != first for value in parts[1:]):
        raise ValueError(f"fusion spans unequal per-step facts {parts!r}")
    return first


def _node_definitions(analysis: Any) -> Layer[Any]:
    """The declared definition of each name a node's frame binds by one
    (`binding.Definition`): what the node's op rebinds (`Analysis.
    node_rebinds`), and else each step extent the phase declares
    (`binding.STEP_LAWS`), formed once over the phase's own binding — so a
    frame that narrowed `s_kv` still shows `s_past` as the phase's
    `s_kv − s_query`. A definition whose value is not the value the frame
    binds the name to is a contradiction and is refused. A definition that
    reads the name it defines narrows the frame's own value — a window's
    `s_kv = Min(s_kv, s_query + s_window − 1)` over the context the frame
    binds (`KVPositions.step_named`) — so what the frame must bind is the
    value the law read, not the one it came to."""
    binding = analysis.binding
    phase: Dict[str, Any] = {}
    for name, law in STEP_LAWS.items():
        if name in binding:
            phase[name] = Definition.of(law, binding)
    out: Dict[int, Dict[str, Any]] = {}
    for uid, frame in analysis.node_dims.values.items():
        rebound = analysis.node_rebinds.values.get(uid) or {}
        defs = {name: value for name, value in rebound.items()
                if isinstance(value, Definition)}
        for name, definition in phase.items():
            if name not in defs and name in (frame or {}):
                defs[name] = definition
        for name, definition in defs.items():
            narrows = name in {symbol.name for symbol
                               in definition.law.free_symbols}
            stated = (definition.bindings[name] if narrows
                      else definition.value)
            if not same_law(stated, frame[name]):
                raise ValueError(
                    f"{analysis.program.by_uid(uid).display_path}: `{name}` "
                    f"is declared {definition.law} = {definition.value}"
                    f"{f' at {name} = {stated}' if narrows else ''}, and "
                    f"its frame binds it to {value_of(frame[name])}")
        if defs:
            out[uid] = defs
    return Layer(analysis.program, out)


def _first(parts: Tuple[Any, ...]) -> Any:
    if not parts:
        raise ValueError("cannot combine an empty layer origin")
    return parts[0]


def _sum(parts: Tuple[Any, ...]) -> Any:
    return sum(parts)


def _ratios(active_share: Layer[Any], program: Any,
            comm_instances: Layer[int]) -> Layer[float]:
    """`Analysis.active_share` as the float the rows carry.

    The share is the walk's (`Evaluator._routed_share` times the copies that
    run over the copies held); this only reads it. A grafted collective runs
    whole, and the walk never saw it, so it comes in at 1.
    """
    values = {uid: float(value_of(share))
              for uid, share in active_share.values.items()}
    values.update({uid: 1.0 for uid in comm_instances.values})
    return Layer(program, values)


def _untimed_uids(program: Any, timed: FrozenSet[str],
                  comm_meta: Layer[Any]) -> frozenset[int]:
    """The nodes of stacks that are not timed. A hop across a pipeline cut
    is drawn in its producer's stack but timed by the stacks it serves, and
    the materializer kept only the hops that serve a timed one: an
    encoder's element read by a decoder layer past a cut is a backbone
    row."""
    hops = {uid for uid, meta in comm_meta.values.items()
            if meta.get("cut") is not None}
    return frozenset(
        node.uid for node in program.nodes()
        if program.stack_of(node) not in timed and node.uid not in hops
    )


@dataclass
class ServingLayers:
    """Every layer one step's rows are read from: the walk's physical
    projections, the program with its collectives and the framework's
    fusions, and each layer rebased across them. Built off the walk of the
    workload's symbols (`StepEvaluation.walk`), so each value is a law in
    them; `at` reads them all at one point. One per walk: the points of a
    cell share it."""

    analysis: Any
    pre_tree: Any
    materialized: Any
    pre_program: Any
    program: Any
    rewrite: Rewrite
    fusion_matches: Tuple[Mapping[str, Any], ...]
    post_tree: Any
    pre_cost: Layer[Cost]
    cost: Layer[Cost]
    pre_full_cost: Layer[Cost]
    full_cost: Layer[Cost]
    #: `full_cost` as one data-parallel replica's step
    #: (`Analysis.replica_cost`): the unsharded figure a table quotes.
    replica_cost: Layer[Cost]
    pre_shapes: Layer[Shape]
    shapes: Layer[Shape]
    pre_full_shapes: Layer[Shape]
    full_shapes: Layer[Shape]
    pre_access: Layer[LeafAccess]
    access: Layer[LeafAccess]
    pre_kv_access: Layer[Any]
    kv_access: Layer[Any]
    pre_instances: Layer[int]
    instances: Layer[int]
    pre_ratios: Layer[float]
    ratios: Layer[float]
    comm_bytes: Layer[Any]
    comm_meta: Layer[Any]
    pre_node_dims: Layer[Any]
    node_dims: Layer[Any]
    #: The definitions of the names each node's op rebinds in its frame
    #: (`Analysis.node_rebinds`: `{name: binding.Definition}`), before and
    #: after the rewrite; a fused kernel's are its first origin's, as its
    #: dims are.
    pre_definitions: Layer[Any]
    definitions: Layer[Any]
    pre_consts: Layer[Any]
    consts: Layer[Any]
    pre_flops_bindings: Layer[Any]
    flops_bindings: Layer[Any]
    untimed: FrozenSet[int]
    dispatch: FrozenSet[int]
    pre_members: FrozenSet[int]
    members: FrozenSet[int]
    #: The buffers the access records priced, after the rewrite and before
    #: it (`physical_access_shapes`): a fused attention streams the
    #: rank-local arriving input, not the kernel's gathered one.
    access_shapes: Layer[Shape]
    pre_access_shapes: Layer[Shape]
    #: Each leaf on a MoE dispatch's routed buffer, after the rewrite and
    #: before it, with that dispatch's boundary on the rank
    #: (`access.routed_launches`).
    pre_routed: Layer[Any]
    routed: Layer[Any]
    #: What a leaf that runs on the step's tokens runs on: the step's rows
    #: (`row_roofline.weight_demand`).
    step_rows: Any
    #: The book every law of the cell is written into and compiled in
    #: (`step_eval.Cell.laws`).
    book: Any
    #: The Ulysses K/V decompression region on the SERVED graph: the rows
    #: the layout rule named (`LayoutPropagation.head_repartition_kv_producers`,
    #: which run on the resident sequence cut before the head exchange), plus
    #: the casts grafted onto their operands — a cast runs on the rows of the
    #: leaf it stands before, so it is in the region exactly when that leaf
    #: is, as a graft is in a MoE dispatch zone exactly when its consumer is.
    kv_region: FrozenSet[int] = frozenset()
    #: The framework's all-reduce dispatch, by collective op
    #: (`FrameworkBinding.collectives`): the rungs each collective row is
    #: priced over (`comm_cost.collective_ladder`). Empty without a binding:
    #: NCCL's schedule prices every collective.
    collectives: Mapping[str, Tuple[Any, ...]] = field(default_factory=dict)
    #: Each row's laws as the row carries them, after the rewrite and before
    #: it, by `(uid, whether its operand conversion is dropped on the
    #: card)`: the same at every point of the cell, so written once, by the
    #: first point that prices the row (`phases.compute_per_op_timing`).
    written: Dict[Any, Dict[str, Any]] = field(default_factory=dict)
    pre_written: Dict[Any, Dict[str, Any]] = field(default_factory=dict)

    def at(self, point: Point) -> "LayersAt":
        """The layers read at `point`, which the walk — and every condition
        a reader of it recorded — must admit: each layer as a reader asks
        for it (`LayersAt`)."""
        self.analysis.admit(point.counts)
        return LayersAt(self, point)


@dataclass
class LayersAt:
    """`ServingLayers` read at one point, layer by layer as a reader asks:
    a layer no reader of the point asks for — the rows' laws the cell
    writes once (`written`), which stay laws — is not read there. Each
    layer is read once (`Point.read`), and two layers that hold one object
    hold the same one read (the point's memo)."""

    _layers: ServingLayers
    _point: Point
    _read: Dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        layers = vars(self._layers)
        if name.startswith("_") or name not in layers:
            raise AttributeError(name)
        read = self._read.get(name, _UNREAD)
        if read is _UNREAD:
            read = self._read[name] = self._point.read(
                layers[name], structure=(SpecIRTree, FusedProgram, LawBook))
        return read


#: A layer `LayersAt` has not read yet.
_UNREAD = object()


def build_serving_layers(cell: Any, *,
                          framework_binding: Optional[FrameworkBinding],
                          stage: Optional[int],
                          timed: FrozenSet[str]) -> ServingLayers:
    """A cell's serving layers for one reader configuration
    (`Cell.serving`), off its walk, its per-rank projections
    (`Cell.physical`) and into its book of laws (`Cell.laws`). The readers
    run under the walk's conditions (`Analysis.assumed`): a condition one of
    them states on the point joins the ones the point is checked against."""
    analysis = cell.built_walk()
    launches = (framework_binding.precision_launches_for()
                if framework_binding is not None else ())
    unavailable_launches = (
        framework_binding.unavailable_precision_launches_for()
        if framework_binding is not None else ())
    # The output widths the kernels this framework dispatches to require of
    # a declared role (`FrameworkBinding.edge_dtypes_for`), with both halves
    # of each `when:` already answered against this deployment and this walk.
    # Resolved here with the launches, and before them in `edge_dtype`'s
    # order, because an edge is what a launch's operand arrives as.
    required = (framework_binding.edge_dtypes_for(
        analysis.program, analysis) if framework_binding is not None else ())
    # One condition, read twice: a launch resolution happens when the
    # framework took one, or when a candidate it declares went unavailable on
    # a card whose capability table participates (an audit gap, never a
    # fallback). Written once so the outer gate and the call below cannot
    # drift apart: with `required` alone pushing the gate open, a second
    # spelling here resolved launches for a cardless walk and invented
    # selector gaps nothing had answered.
    apply_launches = bool(launches) or bool(
        unavailable_launches and framework_binding.deployment.peak_families)
    loaders = (framework_binding.weight_loaders_for(analysis.program, analysis)
               if framework_binding is not None else {})
    loaded = analysis.with_load_formats(loaders)
    load_changed = loaded is not analysis
    analysis = loaded
    if apply_launches or required or load_changed:
        # The binding has already answered each launch's `when:` for this
        # deployment. Materialize the ordered candidates before access reads
        # matmul dtypes: the walk's scheme says what a checkpoint allows,
        # while the card's peak columns and this list say what the selected
        # launch feeds to hardware.
        precision = analysis.resolved_precision
        if apply_launches:
            precision = precision.with_launches(
                launches,
                peak_families=framework_binding.deployment.peak_families,
                unavailable_launches=unavailable_launches)
        if required:
            precision = precision.with_required_edge_dtypes(required)
        analysis = replace(analysis, precision=precision)
        projection = PhysicalProjection(analysis)
    else:
        projection = cell.physical()
    pre_tree = cell._pre_tree
    with analysis.assumed():
        from backend.arch.layout.physical import (
            local_definitions, local_node_dims, pairs_share)
        from backend.arch.eval.launch import score_matrix_owner
        from backend.arch.layout.physical_cost import (
            local_symbol_extents,
            physical_access,
            physical_access_after_rewrite,
        )

        # The weights' dominant scale dtype, for the per-row display width.
        scale_dtype_bytes = scale_dtype_to_bpe(
            analysis.dominant_weight_scale_dtype())
        view = (None if stage is None
                else StagePlacement.of(analysis).view(stage))
        layer_multiplicity = (
            analysis.layer_multiplicity() if view is None
            else view.rows(analysis.program.included_stack_names(
                include_aux=True)))
        selected_instances = analysis.recount(
            analysis.instances, layer_multiplicity
        )
        local_shapes = projection.kernel_shapes
        local_access_shapes = projection.access_shapes
        cost_plan = projection.cost_plan
        local_cost = cost_plan.cost

        serving = rewrite_serving_graph(
            analysis,
            framework_binding=framework_binding,
            stage=view,
            timing_scope=timed,
            local_shapes=local_shapes,
        )
        materialized = serving.materialized
        casts = serving.casts
        pre_program = serving.pre_program
        program = serving.program
        rewrite = serving.rewrite
        matches = serving.fusion_matches
        pre_cost = _on_program(local_cost, pre_program, materialized.cost)
        pre_full_cost = _on_program(
            analysis.cost, pre_program, materialized.cost
        )
        pre_shapes = _on_program(
            local_shapes, pre_program, materialized.shapes
        )
        pre_access_shapes = _on_program(
            local_access_shapes, pre_program, materialized.shapes
        )
        pre_full_shapes = _on_program(
            analysis.shapes, pre_program, materialized.shapes
        )
        pre_instances = _on_program(
            selected_instances, pre_program, materialized.instances
        )
        cast_layers = (
            cast_facts(
                casts,
                local_shapes=pre_shapes,
                full_shapes=pre_full_shapes,
                instances=pre_instances,
            )
            if casts else None
        )
        if cast_layers is not None:
            pre_cost = _on_program(
                pre_cost, pre_program, cast_layers.local_cost)
            pre_full_cost = _on_program(
                pre_full_cost, pre_program, cast_layers.full_cost)
            pre_shapes = _on_program(
                pre_shapes, pre_program, cast_layers.local_shapes)
            pre_access_shapes = _on_program(
                pre_access_shapes, pre_program, cast_layers.local_shapes)
            pre_full_shapes = _on_program(
                pre_full_shapes, pre_program, cast_layers.full_shapes)
            pre_instances = _on_program(
                pre_instances, pre_program, cast_layers.instances)

        if casts:
            # The graft changed the graph the access derivation reads: the
            # cast is a node with its own read and write, and the consumer's
            # conversion number is gone from its row. Deriving over the
            # analysis's program would know neither. An identity rewrite
            # takes this same derivation below.
            pre_access = physical_access_after_rewrite(
                analysis,
                program=pre_program,
                rewrite=casts.rewrite,
                full_cost=pre_full_cost,
                physical_cost=pre_cost,
                full_shapes=pre_full_shapes,
                physical_shapes=pre_access_shapes,
                scale_dtype_bytes=scale_dtype_bytes,
                projection=projection,
            )
        else:
            base_access = physical_access(
                analysis,
                scale_dtype_bytes=scale_dtype_bytes,
                projection=projection,
            )
            pre_access = _on_program(base_access, pre_program)
        pre_kv_access = _on_program(cache_mod.kv_access(analysis), pre_program)

        cost = pre_cost.rebase(rewrite, program, Cost.combine)
        kv_access = pre_kv_access.rebase(
            rewrite, program, cache_mod.KVAccess.union)
        full_cost = pre_full_cost.rebase(rewrite, program, Cost.combine)
        replica_cost = _on_program(
            analysis.replica_cost, pre_program, materialized.cost
        ).rebase(rewrite, program, Cost.combine)
        # The substitution records ride the same rebase as the costs they
        # describe: a fused node's record is the union of its origins'.
        flops_bindings = _on_program(
            cost_plan.flops_bindings, pre_program,
            None if cast_layers is None else cast_layers.flops_bindings,
        ).rebase(rewrite, program, merge_bindings)
        cached = kept_rows(analysis)
        shapes = rebase_shapes(
            pre_program,
            pre_shapes,
            program,
            rewrite,
            preserve=frozenset(cached),
        )
        access_shapes = rebase_shapes(
            pre_program,
            pre_access_shapes,
            program,
            rewrite,
            preserve=frozenset(cached),
        )
        full_shapes = rebase_shapes(
            pre_program,
            pre_full_shapes,
            program,
            rewrite,
            preserve=frozenset(cached),
        )
        instances = pre_instances.rebase(rewrite, program, _same)
        pre_ratios = _ratios(
            _on_program(analysis.active_share, pre_program),
            pre_program,
            materialized.instances,
        )
        ratios = pre_ratios.rebase(rewrite, program, min)

        if rewrite.is_identity():
            access = Layer(program, dict(pre_access.values))
        else:
            access = physical_access_after_rewrite(
                analysis,
                program=program,
                rewrite=rewrite,
                full_cost=full_cost,
                physical_cost=cost,
                full_shapes=full_shapes,
                physical_shapes=access_shapes,
                scale_dtype_bytes=scale_dtype_bytes,
                projection=projection,
            )

        # Each leaf on a routed buffer, with its dispatch's boundary on the
        # rank (`Analysis.routed`): what its padded launch is read off.
        pre_routed = routed_launches(pre_program, pre_access, analysis.routed)
        routed = routed_launches(program, access, analysis.routed)

        comm_bytes = materialized.comm_bytes.rebase(rewrite, program, _sum)
        comm_meta = materialized.comm_meta.rebase(rewrite, program, _first)
        pre_node_dims = _on_program(analysis.node_dims, pre_program)
        pre_definitions = _on_program(_node_definitions(analysis),
                                      pre_program)
        # The function constants each launch takes: a fused kernel's are the
        # ones its origins took (`union_consts`).
        pre_consts = _on_program(analysis.consts, pre_program)
        consts = pre_consts.rebase(rewrite, program, union_consts)
        # Binding stays global on Analysis; layout names the cut. Project
        # here so the timing export evaluates weight templates on the same
        # per-rank slot dims FLOPs already uses. Using the unsharded (N, K)
        # next to a sharded `shape_in` is what killed F.linear on TP linears.
        # Symbols that exist only as factors of a merged axis are answered by
        # the weight-axis contracts' proven per-rank extents. Each frame is
        # projected on the node it belongs to, before the rewrite: a fused
        # node's frame is its first origin's (`_first`), and only that
        # origin's axes say what its names are on a rank — the fused node's
        # surfaces join members that run at different per-rank extents of
        # one name (under CP a GDN prep's gates are on the rank's tokens,
        # its q/k/v on the gathered sequence).
        local_symbols_by_uid = local_symbol_extents(analysis)

        def _projected(graph, dims_layer, globals_, locals_):
            projected_dims = {}
            for uid, dims in dims_layer.values.items():
                if dims is None:
                    projected_dims[uid] = None
                    continue
                node = graph.by_uid(uid)
                global_shape = globals_.values.get(uid)
                local_shape = locals_.values.get(uid)
                if node is None or global_shape is None or local_shape is None:
                    # A node's dims are per rank only off its two shapes; a
                    # node that has dims and not both shapes would publish
                    # the global ones under a per-rank name.
                    raise ValueError(
                        f"node {uid} "
                        f"({'not in the graph' if node is None else node.display_path}) "
                        f"has dims but no "
                        f"{'global' if global_shape is None else 'per-rank'} "
                        f"shape to project them by")
                projected_dims[uid] = local_node_dims(
                    node, dims, global_shape, local_shape,
                    local_symbols=local_symbols_by_uid.get(uid), program=graph,
                    pairs_share=_pairs_share(graph, node, globals_, locals_))
            return Layer(graph, projected_dims)

        def _pairs_share(graph, node, globals_, locals_):
            # The share of the score matrix's cells off the matrix's owner
            # (a fused kernel's absorbed attention kernel), the matrix the
            # FLOPs and the gathered pairs are read off.
            owner = score_matrix_owner(node, graph, locals_)
            if owner is None:
                return None
            return pairs_share(owner.op, globals_.values[owner.uid],
                               locals_.values[owner.uid])

        projected_pre_dims = _projected(pre_program, pre_node_dims,
                                        pre_full_shapes, pre_shapes)
        node_dims = projected_pre_dims.rebase(rewrite, program, _first)

        def _projected_definitions(graph, layer, globals_, locals_):
            # Per rank beside the frames `_projected` makes: a definition
            # explains the number its frame binds (`local_definitions`).
            def projected(uid: int, defs: Any):
                owner = score_matrix_owner(graph.by_uid(uid), graph, locals_)
                query_shapes = (
                    None if owner is None
                    else (globals_.values[owner.uid],
                          locals_.values[owner.uid]))
                return local_definitions(
                    defs, share=_pairs_share(graph, graph.by_uid(uid),
                                             globals_, locals_),
                    global_shape=globals_.values[uid],
                    local_shape=locals_.values[uid],
                    analysis=analysis, node=graph.by_uid(uid),
                    query_shapes=query_shapes, query_node=owner)

            return Layer(graph, {
                uid: projected(uid, defs)
                for uid, defs in layer.values.items()})

        pre_definitions = _projected_definitions(
            pre_program, pre_definitions, pre_full_shapes, pre_shapes)
        definitions = pre_definitions.rebase(rewrite, program, _first)

        untimed = set(_untimed_uids(pre_program, timed, materialized.comm_meta))
        for fused_uid, origins in rewrite.origins.items():
            if any(uid in untimed for uid in origins):
                untimed.add(fused_uid)

    # What a leaf that runs on the step's tokens runs on: the walk's batch
    # times its query block; an encode step has no query block, and its
    # tokens are its requests.
    binding = analysis.binding
    step_tokens = value_of(binding["B"]) * (
        value_of(binding["s_query"]) if "s_query" in binding else 1)
    # The cell's one book, for the rows after the rewrite and before it (a
    # fused kernel's absorbed parts read their laws off the rows before)
    # and for every other reader of the cell.
    book = cell.laws
    pre_flops_bindings = _on_program(
        cost_plan.flops_bindings, pre_program,
        None if cast_layers is None else cast_layers.flops_bindings)
    return ServingLayers(
        analysis=analysis, pre_tree=pre_tree, materialized=materialized,
        pre_program=pre_program, program=program, rewrite=rewrite,
        fusion_matches=matches,
        post_tree=materialize_tree(pre_program, program, rewrite),
        pre_cost=pre_cost, cost=cost, pre_full_cost=pre_full_cost,
        full_cost=full_cost, replica_cost=replica_cost,
        pre_shapes=pre_shapes, shapes=shapes,
        pre_full_shapes=pre_full_shapes, full_shapes=full_shapes,
        pre_access=pre_access, access=access, pre_kv_access=pre_kv_access,
        kv_access=kv_access, pre_instances=pre_instances,
        instances=instances, pre_ratios=pre_ratios, ratios=ratios,
        comm_bytes=comm_bytes, comm_meta=comm_meta,
        pre_node_dims=projected_pre_dims,
        node_dims=node_dims, pre_definitions=pre_definitions,
        definitions=definitions, pre_consts=pre_consts, consts=consts,
        pre_flops_bindings=pre_flops_bindings,
        flops_bindings=flops_bindings, untimed=frozenset(untimed),
        dispatch=program.routed_uids,
        pre_members=pre_program.dispatch_members,
        members=program.dispatch_members,
        access_shapes=access_shapes, pre_access_shapes=pre_access_shapes,
        pre_routed=pre_routed, routed=routed,
        step_rows=step_tokens, book=book,
        kv_region=frozenset(analysis.rule.head_repartition_kv_producers) | {
            site.node.uid for site in (casts.sites if casts else ())
            if site.consumer in analysis.rule.head_repartition_kv_producers},
        collectives=(framework_binding.collectives_for(
                         analysis.program, analysis)
                     if framework_binding is not None else {}))
