"""Physical FLOPs, traffic, instances and access derived from one Analysis."""
from __future__ import annotations

from dataclasses import replace
from math import prod
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

import sympy as sp

from backend.arch import dims as dim_registry
from backend.arch.eval.refusal import node_context
from backend.arch.eval.analysis import Analysis
from backend.arch.eval import cache as cache_mod
from backend.arch.eval import formats as formats_mod
from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.dim import (algebra, Dim, expr_of, is_whole, is_zero,
                                   same_law, value_of, varies)
from backend.arch.eval.flops import (flops_over_operands,
                                     merge_bindings, record_binding)
from backend.arch.eval.layer import Layer
from backend.arch.eval.op_ref import OpRef
from backend.arch.eval.access import (
    LeafAccess,
    access_layer_from_layers,
)
from backend.arch.eval.values import Cost, Shape

from .persistent import (RESIDENT_METADATA, STORED, PhysicalArtifact,
                         physical_artifacts)
from .weight_endpoints import local_symbol_extents
from .physical import (
    _eval_expr,
    _tracked_shape,
    local_node_dims,
    pairs_share,
    physical_access_shapes,
    physical_axis_partition_audit,
    physical_kernel_shapes,
)
from .physical_cache import cache_geometry
from backend.code_trace.memo import memo_property


class PhysicalProjection:
    """One analysis's per-rank projections, each derived once: its physical
    and kernel shapes, the buffers its kernels read, its cache and traffic
    geometry, its traffic and its cost plan. Each derivation below reads the
    others off one projection, so a reader that asks for several (the
    serving layers) derives each once; a reader that asks for one alone
    gets a projection of its own."""

    def __init__(self, analysis: Analysis, *,
                 artifacts: Optional[Tuple[PhysicalArtifact, ...]] = None
                 ) -> None:
        self.analysis = analysis
        self._artifacts = artifacts

    @memo_property
    def artifacts(self) -> Tuple[PhysicalArtifact, ...]:
        """What each rank stores (`persistent.physical_artifacts`)."""
        return (physical_artifacts(self.analysis) if self._artifacts is None
                else self._artifacts)

    @memo_property
    def shapes(self) -> Layer[Shape]:
        """Per-rank tensor shapes (`physical.physical_shapes`)."""
        return self.analysis.physical_shapes()

    @memo_property
    def kernel_shapes(self) -> Layer[Shape]:
        """Per-rank kernel geometry (`physical.physical_kernel_shapes`)."""
        return physical_kernel_shapes(self.analysis, self.shapes)

    @memo_property
    def access_shapes(self) -> Layer[Shape]:
        """The buffers the kernels read (`physical.physical_access_shapes`)."""
        return physical_access_shapes(self.analysis, self.kernel_shapes)

    @memo_property
    def cache_geometry(self) -> cache_mod.CacheGeometry:
        """Per-rank cache geometry (`physical_cache.cache_geometry`)."""
        return cache_geometry(self.analysis, shapes=self.shapes,
                              kernel_shapes=self.kernel_shapes)

    @memo_property
    def traffic_geometry(self) -> traffic_mod.TrafficGeometry:
        """`physical_traffic_geometry`."""
        return physical_traffic_geometry(self.analysis, projection=self)

    @memo_property
    def traffic(self) -> Layer[Any]:
        """`physical_traffic`."""
        return traffic_mod.derive(self.analysis,
                                  geometry=self.traffic_geometry)

    @memo_property
    def cost_plan(self) -> "PhysicalCostPlan":
        """`physical_cost_plan`."""
        return physical_cost_plan(self.analysis, projection=self)


def _projection(analysis: Analysis,
                artifacts: Optional[Tuple[PhysicalArtifact, ...]],
                projection: Optional[PhysicalProjection]
                ) -> PhysicalProjection:
    """The caller's projection of `analysis`, or one of its own."""
    if projection is not None:
        if projection.analysis is not analysis:
            raise ValueError("a projection of another analysis")
        return projection
    return PhysicalProjection(analysis, artifacts=artifacts)


def physical_flops(
    analysis: Analysis,
    kernel_shapes: Layer[Shape] | None = None,
) -> Layer[Any]:
    """Physical per-rank FLOPs derived without another evaluator walk.

    Parameters, resident artifacts and copy counts remain independent layers;
    this function changes only FLOPs and returns a regular ``Layer``.
    """
    values, _bindings = _physical_flops_and_bindings(analysis, kernel_shapes)
    return Layer(analysis.program, values)


def _physical_flops_and_bindings(
    analysis: Analysis,
    kernel_shapes: Layer[Shape] | None = None,
) -> Tuple[Dict[int, Any], Dict[int, Dict[Any, Any]]]:
    """The flops values plus the substitution record each evaluation used."""
    if analysis.rule.is_trivial:
        values = {
            uid: cost.flops for uid, cost in analysis.cost.values.items()
        }
        bindings = {
            uid: dict(record)
            for uid, record in analysis.flops_bindings.values.items()
        }
        return values, bindings
    shapes = (
        kernel_shapes
        if kernel_shapes is not None
        else physical_kernel_shapes(analysis)
    )
    return _project_flops(analysis, shapes)


def physical_traffic_geometry(
    analysis: Analysis,
    *,
    artifacts: Optional[Tuple[PhysicalArtifact, ...]] = None,
    projection: Optional[PhysicalProjection] = None,
) -> traffic_mod.TrafficGeometry:
    """Physical shapes, storage formats and cache geometry for HBM traffic."""
    proj = _projection(analysis, artifacts, projection)
    rows = proj.artifacts
    stored = {
        (row.node_uid, row.tensor_index, row.segment_index): row
        for row in rows
        if row.surface == STORED
    }
    resident_metadata = {
        (row.node_uid, row.tensor_index, row.segment_index,
         row.metadata_index): row
        for row in rows
        if row.surface == RESIDENT_METADATA
    }
    storage = {}
    formats = {}

    for uid, tensors in analysis.storage.values.items():
        local_tensors = []
        local_reps = []
        reps = tuple(analysis.formats.values.get(uid) or ())
        if len(tensors) != len(reps):
            raise ValueError(
                f"storage owner uid {uid} has {len(tensors)} tensors and "
                f"{len(reps)} formats"
            )
        for index, (tensor, reps_of_tensor) in enumerate(
                zip(tensors, reps, strict=True)):
            shapes = None
            local_segments = []
            # Each layer segment of the tensor carries its own format, so
            # each has its own artifact and its own projected metadata. The
            # tensor's rank-local SHAPE is one — every layer holds the same
            # tensor and is cut the same way — and the segments agree on it.
            for segment_index, rep in enumerate(reps_of_tensor):
                row = stored.get((uid, index, segment_index))
                if row is None:
                    raise ValueError(
                        f"storage artifact {(uid, index, segment_index)!r} "
                        f"is missing")
                shapes = row.runtime_shape
                resident = rep.resident
                projected_metadata = []
                for metadata_index, metadata in enumerate(resident.metadata):
                    key = (uid, index, segment_index, metadata_index)
                    metadata_row = resident_metadata.get(key)
                    if metadata_row is None:
                        raise ValueError(
                            f"resident metadata artifact {key!r} is missing"
                        )
                    projected_metadata.append(replace(
                        metadata,
                        shape=_tracked_shape(
                            metadata.shape, metadata_row.runtime_shape
                        ),
                    ))
                local_resident = replace(
                    resident,
                    metadata=tuple(projected_metadata),
                )
                local_segments.append(
                    replace(rep, saved=local_resident) if rep.intent is None
                    else replace(rep, intent=local_resident))
            local_tensors.append(replace(
                tensor,
                shape=_tracked_shape(tensor.shape, shapes),
            ))
            local_reps.append(formats_mod.Representations(
                tuple(local_segments)))
        storage[uid] = tuple(local_tensors)
        formats[uid] = tuple(local_reps)

    geometry = proj.cache_geometry
    cache = _invocation_local_cache(
        analysis, geometry, invocation_shapes=proj.access_shapes)
    frame_divisors = _cache_view_frame_divisors(
        analysis, program=analysis.program, rewrite=analysis.rewrite,
        whole_shapes=analysis.shapes, invocation_shapes=proj.access_shapes)
    from .runtime_weights import runtime_read_aliases
    aliases = runtime_read_aliases(analysis, rows) if analysis.rule.sparse_dcp.plans else None
    return traffic_mod.TrafficGeometry(
        shapes=proj.access_shapes,
        storage=Layer(analysis.program, storage),
        formats=Layer(analysis.program, formats),
        cache=cache,
        whole_shapes=analysis.shapes,
        cache_view_frame_divisor=frame_divisors,
        read_aliases=aliases,
    )


def _invocation_local_cache(analysis: Analysis,
                            geometry: cache_mod.CacheGeometry, *,
                            invocation_shapes: Layer[Any]
                            ) -> Layer[Any]:
    """Cache laws in the invocation frame a Ulysses reader actually runs.

    Persistent cache storage stays whole on the resident sequence family; a
    head-repartition whole-cache view is nevertheless the one invocation seam
    that reads only its resident rows and decompresses them before the
    attention all-to-all.  The structural view set comes from the layout walk.
    Its semantic and access shapes say how many positions that reader owns,
    by axis kind, so the cache law is divided without touching capacity's
    persistent cache geometry.  The landed/view frame ratio is derived
    separately from the shape pair of whichever tensor lands on the view's
    edge in the program being priced.
    """
    layer = cache_mod.derive(analysis, geometry=geometry)
    if analysis.rule.sparse_dcp.plans:
        from .runtime_weights import sparse_dcp_cache_reads
        layer = sparse_dcp_cache_reads(analysis, layer)
    views = analysis.rule.head_repartition_kv_views
    if not views:
        return layer
    context_sources = cache_mod.context_sources(
        cache_mod.Edges(analysis, analysis.program))
    views_by_seed: Dict[int, set] = {}
    view_ownership: Dict[int, int] = {}
    for view in views:
        view_ownership[view] = _view_ownership(
            analysis, invocation_shapes, view)
        coverage = context_sources.get(view)
        for seed in (coverage.seeds if coverage is not None else frozenset()):
            views_by_seed.setdefault(seed, set()).add(view)
    uncut = {
        view for view in views
        if _view_ownership(analysis, invocation_shapes, view) <= 1
    }
    if uncut:
        raise ValueError(
            "a head-repartition cache view has no resident invocation cut: "
            f"{sorted(uncut)}")
    values = dict(layer.values)
    for component_uid, entry in layer.values.items():
        tensors = []
        changed = False
        for tensor in entry.tensors:
            if tensor.node not in views_by_seed:
                tensors.append(tensor)
                continue
            ownerships = {
                view_ownership[view]
                for view in views_by_seed[tensor.node]
            }
            if len(ownerships) != 1:
                raise ValueError(
                    f"cache seed uid {tensor.node} reaches views with "
                    f"different row ownership: {sorted(ownerships)}")
            positions_over = tensor.positions_over * next(iter(ownerships))
            tensors.append(replace(tensor, positions_over=positions_over))
            changed = changed or positions_over != tensor.positions_over
        if changed:
            values[component_uid] = replace(entry, tensors=tuple(tensors))
    represented = {
        tensor.node
        for entry in values.values()
        for tensor in entry.tensors
    }
    if set(views_by_seed) - represented:
        raise ValueError(
            "head-repartition cache views name seeds the cache layer does "
            f"not hold: {sorted(set(views_by_seed) - represented)}")
    return Layer(analysis.program, values)


def _cache_view_frame_divisors(
        analysis: Analysis, *, program: Any, rewrite: Any,
        whole_shapes: Layer[Any], invocation_shapes: Layer[Any]
) -> Dict[int, Any]:
    """The view invocation's sequence cut over its landed tensor's cut.

    A view's sequence cut is only half the frame fact.  The tensor landing on
    its edge may already be sequence-cut (the pre-comm program) or whole (the
    program after Ulysses' all-gather).  Comparing each with its whole-shape
    counterpart gives the ratio traffic must divide fresh rows by; following
    the view edge makes the landed side a path fact rather than a producer
    class or a constant attached to the view.
    """
    views = analysis.rule.head_repartition_kv_views
    if not views:
        return {}
    edges = cache_mod.Edges(analysis, program, rewrite)
    # Land where traffic's `fresh` lands: a cached node lands its own
    # tensor, so both sides walk the same `materialized` chain.
    cached = cache_mod.cached_node_uids(program)
    divisors: Dict[int, Any] = {}
    for view_uid in views:
        view = program.by_uid(view_uid)
        view_divisor = _view_ownership(
            analysis, invocation_shapes, view_uid)
        landed = cache_mod.materialized(view, cached, edges=edges)
        if landed is None:
            raise ValueError(
                f"head-repartition cache view uid {view_uid} has no landed "
                "producer")
        whole = whole_shapes.values.get(landed.uid)
        local = invocation_shapes.values.get(landed.uid)
        if whole is None or local is None:
            raise ValueError(
                f"head-repartition cache landed uid {landed.uid} has no "
                "whole/invocation shape pair")
        landed_divisor = _sequence_ownership(whole.out, local.out)
        divisors[view_uid] = sp.cancel(
            sp.Rational(view_divisor, landed_divisor))
    return divisors


def _sequence_ownership(global_axes: Any, local_axes: Any) -> int:
    """Position ownership implied by one semantic sequence axis being cut."""
    # Which axis is the sequence one is a question about identity, answered
    # by the registry's declaration of the symbol the axis carries
    # (`dims.kinds`, as `materialize_flow.axis_of_kind` reads it).
    kinds = dim_registry.kinds()
    positions = [
        index for index, axis in enumerate(global_axes or ())
        if isinstance(axis, Dim)
        and kinds.get(str(expr_of(axis))) == "sequence"
    ]
    if (len(positions) != 1 or len(global_axes or ()) != len(local_axes or ())
            or any(local_axes[index] is None for index in positions)):
        return 1
    index = positions[0]
    whole, local = global_axes[index], local_axes[index]
    # Whether the axis is cut at all is a question about identity too: one
    # law, the same at every point (`same_law`). Dividing and testing the
    # quotient against 1 asked it of the value halves, which agree at the
    # walked point for two laws that do not agree anywhere else.
    if same_law(whole, local):
        return 1
    ownership = sp.cancel(algebra(value_of(whole)) / algebra(value_of(local)))
    # One rank's share of a sequence axis is a deployment degree — the same
    # number at every workload — so a quotient still carrying the workload's
    # symbols is refused here (`varies`) rather than read at the walked
    # point, and the number is held to a whole count of positions
    # (`is_whole`) above zero (`is_zero`; both extents are nonnegative
    # counts, so zero is the only degenerate value the quotient takes).
    if varies(ownership) or not is_whole(ownership) or is_zero(ownership):
        raise ValueError(
            f"a head-repartition cache sequence axis {whole!r} does not "
            f"divide into local rows {local!r}: one rank would own "
            f"{ownership} positions")
    return int(ownership)


def _view_ownership(analysis: Analysis, invocation_shapes: Layer[Any],
                    view: int) -> int:
    persistent = analysis.shapes.values.get(view)
    invocation = invocation_shapes.values.get(view)
    if persistent is None or invocation is None:
        raise ValueError(
            f"head-repartition cache view uid {view} has no persistent/"
            "invocation shape pair")
    return _sequence_ownership(persistent.out, invocation.out)


def physical_traffic(
    analysis: Analysis,
    *,
    artifacts: Optional[Tuple[PhysicalArtifact, ...]] = None,
    projection: Optional[PhysicalProjection] = None,
) -> Layer[Any]:
    """Per-kernel physical HBM traffic, derived without evaluator replay."""
    return _projection(analysis, artifacts, projection).traffic


class PhysicalCostPlan(NamedTuple):
    """What `physical_cost_plan` derives in one pass.

    `flops_bindings` / `params_bindings` are the substitution records the
    physical derivation itself used — identity → per-rank value — so a
    formula display binds the derivation's own numbers instead of
    reconstructing them from shapes afterwards.
    """

    cost: Layer[Cost]
    flops_bindings: Layer[Dict[Any, Any]]
    params_bindings: Layer[Dict[Any, Any]]


def physical_cost_plan(
    analysis: Analysis,
    *,
    artifacts: Optional[Tuple[PhysicalArtifact, ...]] = None,
    projection: Optional[PhysicalProjection] = None,
) -> PhysicalCostPlan:
    """Physical per-kernel cost on one rank.

    A weight-bearing leaf's parameters are its stored tensors' rank-local
    elements times the copies the walk counted (`Analysis.own_copies`; a copy
    count is never sharded — what a rank holds less of is an AXIS, the
    expert axis of a bank among them). Its active parameters keep the walk's
    own share of them (`cost.active_params / cost.params`), which is where a
    dispatch zone's `n_experts_active / n_route_targets` already lives.
    """
    proj = _projection(analysis, artifacts, projection)
    rows = proj.artifacts
    primaries: dict[int, list[Any]] = {}
    for row in rows:
        if row.surface == STORED and row.is_parameter:
            primaries.setdefault(row.node_uid, []).append(row)

    costs = {}
    params_bindings: Dict[int, Dict[Any, Any]] = {}
    flops_values, flops_bindings = _physical_flops_and_bindings(
        analysis, None if analysis.rule.is_trivial else proj.kernel_shapes)
    for uid, cost in analysis.cost.values.items():
        flops = flops_values.get(uid, cost.flops)
        owned = primaries.get(uid)
        if not owned:
            costs[uid] = replace(cost, flops=flops)
            continue
        node = analysis.program.by_uid(uid)
        own = analysis.own_copies_of(node)
        per_copy = sum(prod(row.runtime_shape) for row in owned)
        params = algebra(value_of(own.stored)) * sp.Integer(per_copy)
        if not params.is_integer:
            raise ValueError(
                f"{node.display_path}: projected parameters {params!r} are "
                "not an integer"
            )
        params = int(params)
        # The substitution the params figure was priced from: each stored
        # tensor's global axis identity bound to its runtime extent.
        stored = analysis.storage.values.get(uid) or ()
        record: Dict[Any, Any] = {}
        for row in owned:
            tensor = stored[row.tensor_index]
            for axis in _tracked_shape(tensor.shape, row.runtime_shape):
                record_binding(record, axis)
        if record:
            params_bindings[uid] = record
        active = sp.Integer(params) * _active_share(node, cost)
        local_params = (
            Dim(expr_of(cost.params), params)
            if isinstance(cost.params, Dim) else params
        )
        local_active = (
            Dim(expr_of(cost.active_params), active)
            if isinstance(cost.active_params, Dim) else active
        )
        costs[uid] = replace(
            cost,
            flops=flops,
            params=local_params,
            active_params=local_active,
        )
    return PhysicalCostPlan(
        cost=Layer(analysis.program, costs),
        flops_bindings=Layer(analysis.program, flops_bindings),
        params_bindings=Layer(analysis.program, params_bindings),
    )


def _active_share(node: Any, cost: Cost) -> Any:
    """The fraction of a leaf's parameters one token touches, as the walk
    priced it: exact, and refused rather than guessed when the walk left it
    symbolic."""
    params = algebra(value_of(cost.params))
    active = algebra(value_of(cost.active_params))
    if params.free_symbols or active.free_symbols:
        raise ValueError(
            f"{node.display_path}: an ungrounded active-parameter share "
            f"{active!r} / {params!r}")
    return sp.Rational(active) / sp.Rational(params)


def physical_access(
    analysis: Analysis,
    *,
    artifacts: Optional[Tuple[PhysicalArtifact, ...]] = None,
    scale_dtype_bytes: float,
    projection: Optional[PhysicalProjection] = None,
) -> Layer[LeafAccess]:
    """Per-leaf HBM access on the analysis's physical layers."""
    proj = _projection(analysis, artifacts, projection)
    traffic = proj.traffic
    plan = proj.cost_plan
    global_traffic = traffic_mod.derive(analysis)
    return access_layer_from_layers(
        analysis,
        program=analysis.program,
        global_traffic=global_traffic,
        global_cost=analysis.cost,
        global_shapes=analysis.shapes,
        local_traffic=traffic,
        local_cost=plan.cost,
        local_shapes=proj.access_shapes,
        scale_dtype_bytes=scale_dtype_bytes,
        local_params_bindings=plan.params_bindings.values,
    )


def physical_access_after_rewrite(
    analysis: Analysis,
    *,
    program: Any,
    rewrite: Any,
    full_cost: Layer[Cost],
    physical_cost: Layer[Cost],
    full_shapes: Layer[Shape],
    physical_shapes: Layer[Shape],
    artifacts: Optional[Tuple[PhysicalArtifact, ...]] = None,
    scale_dtype_bytes: float,
    projection: Optional[PhysicalProjection] = None,
) -> Layer[LeafAccess]:
    """HBM access after a graph rewrite, without constructing an Analysis."""
    proj = _projection(analysis, artifacts, projection)
    base_geometry = proj.traffic_geometry
    global_geometry = traffic_mod.TrafficGeometry(
        shapes=full_shapes,
        storage=analysis.storage,
        formats=analysis.formats,
        cache=cache_mod.derive(analysis),
        whole_shapes=full_shapes,
    )
    local_geometry = replace(
        base_geometry,
        shapes=physical_shapes,
        whole_shapes=full_shapes,
        cache_view_frame_divisor=_cache_view_frame_divisors(
            analysis, program=program, rewrite=rewrite,
            whole_shapes=full_shapes, invocation_shapes=physical_shapes),
    )
    global_traffic = traffic_mod.derive(
        analysis,
        geometry=global_geometry,
        program=program,
        rewrite=rewrite,
    )
    local_traffic = traffic_mod.derive(
        analysis,
        geometry=local_geometry,
        program=program,
        rewrite=rewrite,
    )
    plan = proj.cost_plan
    # A fused node's params substitution record is the union of its absorbed
    # siblings' records; a key the siblings disagree on is dropped.
    rebased_params_bindings: Dict[int, Dict[Any, Any]] = dict(
        plan.params_bindings.values)
    for fused_uid, members in rewrite.origins.items():
        member_records = [
            plan.params_bindings.values[uid]
            for uid in members
            if uid in plan.params_bindings.values
        ]
        if member_records:
            rebased_params_bindings[fused_uid] = merge_bindings(member_records)
    return access_layer_from_layers(
        analysis,
        program=program,
        global_traffic=global_traffic,
        global_cost=full_cost,
        global_shapes=full_shapes,
        local_traffic=local_traffic,
        local_cost=physical_cost,
        local_shapes=physical_shapes,
        scale_dtype_bytes=scale_dtype_bytes,
        local_params_bindings=rebased_params_bindings,
    )


def _project_flops(
    analysis: Analysis, local_shapes: Layer[Shape]
) -> Tuple[Dict[int, Any], Dict[int, Dict[Any, Any]]]:
    """Re-evaluate leaves locally, then close component costs bottom-up.

    Returns the flops values and, per leaf, the substitution record the
    local evaluation used (`flops_from_shape`'s `record`) — the per-rank
    values the physical formula's symbols actually stood for.
    """
    projected: Dict[int, Any] = {}
    bindings: Dict[int, Dict[Any, Any]] = {}
    # Per-rank symbol values proven by weight-axis contracts, and the shard
    # degree each model symbol is under. An env-tier read whose symbol is
    # sharded must have been answered by the shape axes or the facts; keeping
    # the global binding is the double-source divergence, and it raises
    # rather than passing.
    symbol_facts = local_symbol_extents(analysis)
    shard_degrees = {
        audit.symbol: audit.applied
        for audit in physical_axis_partition_audit(analysis)
    }

    def env_guard(node: Any, projected_names: set):
        def guard(name: str, extent: Any) -> None:
            if name in projected_names:
                return
            identity = expr_of(extent)
            if not isinstance(identity, sp.Basic):
                return
            free = identity.free_symbols
            sharded = sorted(
                sym.name for sym in free
                if shard_degrees.get(sym.name, 1) > 1
            )
            if sharded:
                raise ValueError(
                    f"{node.display_path}: the local FLOPs evaluation read "
                    f"env dim {name!r} at its global value, but its "
                    f"symbol(s) {sharded!r} are sharded under this "
                    f"deployment and neither the node's shapes nor its "
                    f"weight-axis contracts provide a per-rank extent"
                )
        return guard

    def visit(node: Any) -> Any:
        children = [
            child for child in node.children
            if child.uid in analysis.cost.values
        ]
        if children:
            local_base = Cost.combine([
                Cost(flops=visit(child)) for child in children
            ]).flops
        else:
            global_shape = analysis.shapes.values[node.uid]
            local_shape = local_shapes.values[node.uid]
            facts = symbol_facts.get(node.uid, {})
            projected_names: set = set()
            record: Dict[Any, Any] = {}
            with node_context(node.display_path):
                local_base = _primitive_flops(
                    analysis, node, local_shape,
                    local_node_dims(
                        node,
                        analysis.node_dims.values.get(node.uid, {}),
                        global_shape,
                        local_shape,
                        local_symbols=facts,
                        projected_names=projected_names,
                        pairs_share=pairs_share(
                            node.op, global_shape, local_shape),
                    ),
                    record=record,
                    env_guard=env_guard(node, projected_names),
                )
            if record:
                bindings[node.uid] = record
        projected[node.uid] = local_base * algebra(
            value_of(analysis.own_copies_of(node).active))
        return projected[node.uid]

    for node in analysis.program.nodes():
        if node.parent is None and node.uid in analysis.cost.values:
            visit(node)
    missing = set(analysis.cost.values) - set(projected)
    if missing:
        raise ValueError(
            f"local FLOPs projection did not reach node UIDs {sorted(missing)!r}"
        )
    return projected, bindings


def _primitive_flops(
    analysis: Analysis,
    node: Any,
    shape: Shape,
    dims: Mapping[str, Any],
    *,
    record: Optional[Dict[Any, Any]] = None,
    env_guard: Optional[Any] = None,
) -> Any:
    op = OpRef.resolve(node.op or "", analysis.program, axis=node.axis)
    if op is None:
        raise ValueError(
            f"primitive {node.display_path!r} has unknown op {node.op!r}"
        )
    return flops_over_operands(
        op,
        dims,
        shape.ins,
        lambda derived: op.flops(derived, _eval_expr),
        require_consistent_labels=True,
        record=record,
        env_guard=env_guard,
    )
