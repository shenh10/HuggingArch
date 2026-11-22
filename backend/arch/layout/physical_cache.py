"""Physical cache geometry derived from one layout-carrying Analysis."""
from __future__ import annotations

from dataclasses import replace
from typing import (TYPE_CHECKING, Any, Dict, FrozenSet, Mapping, Optional,
                    Sequence, Tuple)

from backend.arch.eval.analysis import Analysis
from backend.arch.eval.refusal import node_context
from backend.arch.eval import cache as cache_mod
from backend.arch.eval.cache_view import CacheView, cache_view_from_analysis
from backend.arch.eval.dim import algebra, Dim, value_of
from backend.arch.eval.layer import Layer
from backend.arch.eval.values import Shape
from backend.arch import dims as dim_registry

from .physical import (
    _eval_expr,
    _shape_axes,
    local_node_dims,
    pairs_share,
    physical_kernel_shapes,
)

if TYPE_CHECKING:
    from .propagation import LayoutPropagation


def cache_geometry(
    analysis: Analysis,
    *,
    shapes: Optional[Layer[Shape]] = None,
    kernel_shapes: Optional[Layer[Shape]] = None,
) -> cache_mod.CacheGeometry:
    """Physical per-rank cache geometry from the same analysis.

    Cache cardinality laws remain on ``analysis``. Tensor widths come from
    ``physical_shapes``, recurrent-state widths from the declared physical
    kernel carrier, and persistent position ownership from the typed DCP cache
    family; no per-rank evaluator or partial Analysis is constructed.

    Which cached tensors tensor parallelism narrowed is measured here, once,
    by comparing the global and the projected widths, and carried on the
    geometry (`CacheGeometry.tp_divided`). `shapes` / `kernel_shapes` are
    the analysis's physical and kernel shapes, where the caller already
    holds them (`physical_cost.PhysicalProjection`).
    """
    if shapes is None:
        shapes = analysis.physical_shapes()
    sharding = analysis.rule
    if kernel_shapes is None:
        kernel_shapes = physical_kernel_shapes(analysis, shapes)
    state_shapes = _project_cache_states(analysis, kernel_shapes)
    positions_over = (
        sharding.cache_family.group_size
        if sharding.cache_family is not None else 1
    )
    geometry = cache_mod.CacheGeometry(
        shapes=shapes,
        state_shapes=state_shapes,
        positions_over=positions_over,
        padded_position_components=frozenset(
            plan.component.uid for plan in sharding.sparse_dcp.plans),
    )
    global_cache = cache_mod.derive(analysis)
    local_cache = cache_mod.derive(analysis, geometry=geometry)
    global_elements = _cache_elements(global_cache)
    local_elements = _cache_elements(local_cache)
    if set(global_elements) != set(local_elements):
        raise ValueError(
            "physical cache geometry changed the declared cache tensor set"
        )
    width_changed = {
        key for key in global_elements
        if value_of(global_elements[key]) != value_of(local_elements[key])
    }
    tp_owners = tuple(sorted(
        world.label
        for world in dim_registry.worlds().values()
        if world.mesh == "tp" and world.label in sharding.families
    ))
    if width_changed and not tp_owners:
        raise ValueError(
            "cache tensor widths changed without an active TP owner: "
            f"{sorted(width_changed)!r}"
        )
    return replace(geometry, tp_divided=frozenset(width_changed))


def physical_cache_view(
    analysis: Analysis,
    *,
    cache_scope: Optional[FrozenSet[str]] = None,
) -> CacheView:
    """Per-rank cache accounting derived from physical cache geometry, for
    the stacks of `cache_scope` (the backbone when None)."""
    view = cache_view_from_analysis(
        analysis,
        geometry=cache_geometry(analysis),
        cache_scope=cache_scope,
    )
    parallel = analysis.rule.parallel
    if parallel is None:
        return view
    return view.on_layer_split(parallel.layer_split_over)


def whole_model_cache_view(
    analysis: Analysis,
    *,
    cache_scope: Optional[FrozenSet[str]] = None,
) -> CacheView:
    """Whole-model cache accounting for comparison and reporting, for the
    stacks of `cache_scope` (the backbone when None)."""
    return cache_view_from_analysis(
        analysis,
        cache_scope=cache_scope,
    )


def _cache_elements(layer: Layer[Any]) -> Dict[Tuple[int, int, str], Any]:
    out = {}
    for component_uid, entry in layer.values.items():
        for tensor in entry.tensors:
            key = (component_uid, tensor.node, tensor.name)
            if key in out:
                raise ValueError(f"duplicate cache tensor identity {key!r}")
            out[key] = tensor.elements
    return out


def _project_cache_states(
    analysis: Analysis,
    kernel_shapes: Layer[Shape],
) -> Layer[Any]:
    """Evaluate each recurrent state's declared axes in its carrier's
    physical binding.

    Per state, because each is the carrier's own: the scan's matrix shards
    with the scan's heads, a conv's window with the conv's channels.
    """
    projected = {}
    for component_uid, global_states in analysis.cache_state.values.items():
        component = analysis.program.by_uid(component_uid)
        if component is None:
            raise ValueError(
                f"cache state layer names unknown component uid {component_uid}"
            )
        with node_context(component.display_path):
            declared = {e.state: e for e in component.kv_states}
            if set(declared) != set(global_states):
                raise ValueError(
                    f"{component.display_path}: cache state {sorted(global_states)} "
                    f"does not match the declared states {sorted(declared)}"
                )
            by_name = {child.name: child for child in component.children}
            local_states = {}
            for name, global_axes in global_states.items():
                entry = declared[name]
                carrier = by_name.get(entry.carrier)
                if carrier is None:
                    raise ValueError(
                        f"{component.display_path}: recurrent state {name!r} "
                        f"names carrier {entry.carrier!r}, which is not among its "
                        f"nodes"
                    )
                with node_context(carrier.display_path):
                    local_states[name] = _project_state(
                        analysis, kernel_shapes, component, carrier, name,
                        entry.shape, global_axes)
        projected[component_uid] = local_states
    return Layer(analysis.program, projected)


def _project_state(
    analysis: Analysis,
    kernel_shapes: Layer[Shape],
    component: Any,
    carrier: Any,
    name: str,
    declared: Sequence[Any],
    global_axes: Tuple[Any, ...],
) -> Tuple[Any, ...]:
    """One state's per-rank axes, from its carrier's kernel shape: each
    declared axis evaluated in the carrier's per-rank binding, so an axis
    follows the kernel axis of the same identity."""
    global_shape = analysis.shapes.values.get(carrier.uid)
    local_shape = kernel_shapes.values.get(carrier.uid)
    binding = analysis.node_dims.values.get(carrier.uid)
    if global_shape is None or local_shape is None or binding is None:
        raise ValueError(
            f"{carrier.display_path}: recurrent-state carrier has no "
            "complete shape/binding geometry"
        )
    _validate_cache_state_dependencies(
        carrier,
        declared,
        binding,
        global_shape,
        local_shape,
        analysis.rule,
    )
    local_binding = local_node_dims(
        carrier,
        binding,
        global_shape,
        local_shape,
        pairs_share=pairs_share(carrier.op, global_shape, local_shape),
    )
    local_axes = tuple(_eval_expr(axis, local_binding) for axis in declared)
    for index, (axis, local, global_) in enumerate(
            zip(declared, local_axes, global_axes, strict=True)):
        local_value = value_of(local)
        if (
            getattr(algebra(local_value), "free_symbols", None)
            or local_value <= 0
        ):
            raise ValueError(
                f"{component.display_path}: recurrent state {name!r} axis "
                f"{index} `{axis}`, evaluated in carrier "
                f"{carrier.display_path}'s per-rank binding, is not a "
                f"positive concrete extent: {local!r}"
            )
        if value_of(global_) <= 0:
            raise ValueError(
                f"{component.display_path}: recurrent state {name!r} axis "
                f"{index} `{axis}` must be positive, got {global_!r}"
            )
    return local_axes


def _validate_cache_state_dependencies(
    carrier: Any,
    declared: Sequence[Any],
    binding: Mapping[str, Any],
    global_shape: Shape,
    local_shape: Shape,
    sharding: LayoutPropagation,
) -> None:
    """Require every live shardable symbol of a state axis to have physical
    provenance.

    A registry capability does not say that every occurrence is sharded.  It
    does say that, when its world is live, a state axis cannot inherit the
    global binding for that symbol without proving which occurrence owns it.
    The recurrent kernel's tracked axes are that proof; their propagated
    layout decides whether this particular occurrence is local or replicated.
    """
    global_exprs = {
        axis.expr
        for axis in _shape_axes(global_shape, outputs=True)
        if isinstance(axis, Dim)
    } | {
        axis.expr
        for axis in _shape_axes(global_shape, outputs=False)
        if isinstance(axis, Dim)
    }
    local_exprs = {
        axis.expr
        for axis in _shape_axes(local_shape, outputs=True)
        if isinstance(axis, Dim)
    } | {
        axis.expr
        for axis in _shape_axes(local_shape, outputs=False)
        if isinstance(axis, Dim)
    }
    for index, free in sorted(
            {(index, free) for index, axis in enumerate(declared)
             for free in algebra(axis).free_symbols},
            key=lambda pair: (pair[0], str(pair[1]))):
        name = str(free)
        bound = binding.get(name)
        identity_names = (
            bound.names if isinstance(bound, Dim) else frozenset({name})
        )
        active_worlds = tuple(
            sorted({
                world.label
                for identity in identity_names
                for world in dim_registry.capabilities_of(identity)
                if world.label in sharding.families
            })
        )
        if not active_worlds:
            continue
        if not isinstance(bound, Dim):
            raise ValueError(
                f"{carrier.display_path}: recurrent-state axis {index} reads "
                f"{name!r}, live in {active_worlds!r}, but it has no tracked "
                f"Dim binding"
            )
        if bound.expr not in global_exprs or bound.expr not in local_exprs:
            raise ValueError(
                f"{carrier.display_path}: recurrent-state axis {index} "
                f"reads {name!r}, live in {active_worlds!r}, but its identity "
                f"{bound.expr!r} is not carried by the physical kernel shape. "
                f"Write this axis as the expression the carrier's output "
                f"carries that head count in, so the state shards with the "
                f"kernel."
            )
