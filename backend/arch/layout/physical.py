"""Physical layers derived from one layout-carrying ``Analysis``.

The propagator deliberately evaluates the global geometry: layout is carried
beside a tensor and no dimension is divided on read.  Runtime consumers need a
numeric local shape, so this module is the single reconciliation boundary from
``(global extent, layout)`` to the physical extent one rank allocates.

The functions return existing core types.  There is no second result object:
``Analysis`` remains the sole evaluation result and every physical layer keeps
its ``Program`` identity.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import sympy as sp
from typing import (TYPE_CHECKING, Any, Dict, Iterator, Mapping, Optional,
                    Sequence, Tuple)

from backend.arch.eval.analysis import Analysis
from backend.arch.eval.dim import (Dim, eval_dim_expr, exact, expr_of,
                                   require_equal, same_law, value_of, varies)
from backend.arch.eval.flops import bind_labels
from backend.arch.eval.op_ref import OpRef
from backend.arch.eval.layer import Layer
from backend.arch.eval.values import Shape
from backend.arch import dims as dim_registry
from backend.arch.eval.binding import Definition
from backend.arch.ops import attention_kernel

from .attention import ATTENTION_KERNEL_OPS
from .attention_plan import ATTENTION_PLANS
from .layout import Block, Product, Replicated
from .edge import EdgeState
from .resolver import layout_for_symbol
from backend.arch.eval.storage import is_expert_axis

if TYPE_CHECKING:
    from .propagation import LayoutPropagation


@dataclass(frozen=True)
class AxisPartitionAudit:
    """Requested and applied partition for one declared model axis."""

    symbol: str
    extent: int
    requested: int
    applied: int
    #: The degree that splits the axis, as a deployment names it (`cp`,
    #: `tp_attention`: `World.label`).
    degree: str = ""


def physical_axis_partition_audit(
    analysis: Analysis,
) -> Tuple[AxisPartitionAudit, ...]:
    """Resolve the partitions that produced this analysis's local shapes."""
    sharding = analysis.rule
    if sharding.is_trivial:
        return ()
    out = []
    for symbol in sorted(analysis.binding.symbols()):
        world = dim_registry.world_of(symbol)
        if world is None or world.bound_per_rank:
            continue
        family = sharding.families.get(world.label)
        if family is None:
            continue
        extent = value_of(analysis.binding.raw(symbol))
        if same_law(extent, 0):
            # An axis the model leaves empty partitions nothing.
            continue
        plans = sharding.sparse_dcp
        if world.label == "dcp" and plans.plans:
            # Striped ownership permits a final partial shard. This audit
            # reports the owner count, not an equal contiguous partition.
            out.append(AxisPartitionAudit(
                symbol=symbol, extent=extent, requested=family.group_size,
                applied=family.group_size, degree=world.label))
            continue
        layout = layout_for_symbol(
            symbol,
            extent,
            sharding.parallel,
            sharding.phase,
            dict(sharding.families),
        )
        if isinstance(layout, Block):
            applied = layout.num_shards
        elif layout is None or isinstance(layout, Replicated):
            applied = 1
        else:
            raise ValueError(
                f"axis {symbol!r} resolved unsupported declaration layout "
                f"{layout!r}"
            )
        out.append(AxisPartitionAudit(
            symbol=symbol,
            extent=extent,
            requested=family.group_size,
            applied=applied,
            degree=world.label,
        ))
    return tuple(out)


def physical_axis_partitions(analysis: Analysis) -> Mapping[str, int]:
    """Named model axes physically partitioned in this analysis."""
    return {
        row.symbol: row.applied
        for row in physical_axis_partition_audit(analysis)
        if row.applied > 1
    }


def _projected_shapes(analysis: Analysis, *,
                      persistent_head_repartition_kv: bool
                      ) -> Layer[Shape]:
    """Physical per-rank tensor shapes derived from one analysis.

    ``outputs`` mirrors semantic ``Analysis.shapes``.  Primitive inputs are
    projected through the state the physical kernel consumes; invocation-only
    extent changes remain in ``physical_kernel_shapes`` rather than changing the SSA
    tensor.  Usually the state follows an edge redistribution, while a
    persistent buffer may have independent placement and retain only logical
    identity on its SSA edge.  Component boundaries have no transition of
    their own; their input shapes remain the global structural boundary and
    are not kernel rows.
    """
    sharding = analysis.rule
    if sharding.is_trivial:
        return analysis.shapes

    def persistent_state(uid: int, state: EdgeState) -> EdgeState:
        """A persistent tensor's state, minus this invocation's row cut.

        Ulysses cuts a whole-cache view and its decompression chain only for
        the invocation that ends at the attention all-to-all.  The SSA tensor
        and cache storage remain persistent: remove exactly that resident
        sequence cut, while every other placement (notably attention TP's
        head cut) still projects the shape.
        """
        if not (persistent_head_repartition_kv
                and uid in analysis.rule.head_repartition_kv_producers):
            return state
        family = analysis.rule.resident_sequence_family
        if family is None:
            return state
        layout = tuple(
            Replicated(axis.group.ranks())
            if isinstance(axis, Block) and axis.is_sharded_on(family)
            else axis
            for axis in state.layout
        )
        return replace(state, layout=layout)

    def project(uid: int, node: Any, shape: Shape) -> Shape:
        outputs = {"out": shape.out, **dict(shape.ports)}
        local_outputs = {}
        for port, dims in outputs.items():
            if dims is None:
                local_outputs[port] = None
                continue
            state = persistent_state(
                uid, _output_state(analysis, uid, port))
            local_outputs[port] = local_dims(
                dims, state, sharding, f"{node.display_path}.{port}",
            )

        local_inputs = []
        for index, dims in enumerate(shape.ins):
            key = (uid, index)
            state = analysis.input_states.get(key)
            if state is None:
                if key not in analysis.input_states:
                    if not node.children:
                        raise ValueError(
                            f"{node.display_path} input {index} has a shape but no "
                            "consumer-required layout state"
                        )
                    local_inputs.append(tuple(dims))
                    continue

                # A present None is an evaluator-retained optional input, not
                # a physical full-size tensor.  When it has the result's rank,
                # its structural shape follows the sole live branch/output.
                output_state = persistent_state(
                    uid, _output_state(analysis, uid, "out"))
                if len(dims) != len(output_state.layout):
                    local_inputs.append(tuple(dims))
                    continue
                state = output_state
                if node.children:
                    raise ValueError(
                        f"{node.display_path} component input {index} has an "
                        "optional state at a physical boundary"
                    )
            local_inputs.append(local_dims(
                dims,
                persistent_state(uid, state),
                sharding,
                f"{node.display_path}.input[{index}]",
            ))

        # No `moved`: that is a fused boundary's region (`values.Moved`), and
        # the rows projected here are the walk's own nodes, which publish
        # exactly what they write. The serving path rebases these into fused
        # boundaries afterwards (`traffic.derive` with a rewrite, which calls
        # `fusion.rebase_shapes`), and the region is computed there from
        # these per-rank rows — so a projection that carried one would be
        # carrying it from a node that cannot have had it.
        return Shape(
            out=local_outputs.pop("out"),
            ins=tuple(local_inputs),
            ports=local_outputs,
        )

    projected = {}
    for uid, shape in analysis.shapes.values.items():
        node = analysis.program.by_uid(uid)
        if node is None:
            raise ValueError(f"shape layer names unknown node uid {uid}")
        local = project(uid, node, shape)
        # An axis whose extent is a law the walk evaluated at the global
        # extents is this rank's law, then its own layout's cut.
        at_rank = rank_law_shape(analysis, node, shape, local)
        projected[uid] = (local if at_rank is shape
                          else project(uid, node, at_rank))
    return Layer(analysis.program, projected)


def physical_shapes(analysis: Analysis) -> Layer[Shape]:
    """Physical per-rank tensor shapes derived from one analysis.

    A Ulysses K/V decompression producer has two legitimate shapes: its kernel
    computes resident rows, while its persistent SSA/cache shape remains whole
    on the invocation's sequence axis.  The persistent face is still per-rank:
    every other placement (attention TP's heads included) is projected.  The
    kernel face is the overlay in `physical_kernel_shapes`; cache storage does
    not inherit that compute cut.
    """
    if analysis.rule.is_trivial:
        return analysis.shapes
    return _projected_shapes(
        analysis, persistent_head_repartition_kv=True)


def _validate_kernel_shape(
    node: Any, semantic: Shape, physical: Shape
) -> None:
    """A physical override may change extents, never tensor identities."""
    if semantic.ports or physical.ports:
        raise ValueError(
            f"{node.display_path}: primitive kernel geometry does not support "
            "secondary output ports"
        )
    surfaces = (
        ("out", semantic.out, physical.out),
        *(
            (f"input[{index}]", global_axes, local_axes)
            for index, (global_axes, local_axes) in enumerate(
                zip(semantic.ins, physical.ins, strict=False)
            )
        ),
    )
    if len(semantic.ins) != len(physical.ins):
        raise ValueError(
            f"{node.display_path}: semantic/kernel input arity "
            f"{len(semantic.ins)} != {len(physical.ins)}"
        )
    for label, global_axes, local_axes in surfaces:
        if global_axes is None or local_axes is None:
            if global_axes != local_axes:
                raise ValueError(
                    f"{node.display_path}.{label}: semantic/kernel presence "
                    "differs"
                )
            continue
        if len(global_axes) != len(local_axes):
            raise ValueError(
                f"{node.display_path}.{label}: semantic/kernel rank "
                f"{len(global_axes)} != {len(local_axes)}"
            )
        for global_axis, local_axis in zip(
            global_axes, local_axes, strict=True
        ):
            if isinstance(global_axis, Dim) and (
                not isinstance(local_axis, Dim)
                or expr_of(local_axis) != expr_of(global_axis)
            ):
                raise ValueError(
                    f"{node.display_path}.{label}: kernel axis "
                    f"{local_axis!r} lost identity {expr_of(global_axis)!r}"
                )


def physical_kernel_shapes(
    analysis: Analysis,
    semantic_shapes: Layer[Shape] | None = None,
) -> Layer[Shape]:
    """Overlay invocation-only kernel geometry on physical tensor shapes."""
    shapes = (semantic_shapes if semantic_shapes is not None
              else analysis.physical_shapes())
    if analysis.rule.head_repartition_kv_producers:
        invocation = _projected_shapes(
            analysis, persistent_head_repartition_kv=False)
        values = dict(shapes.values)
        values.update({
            uid: invocation.values[uid]
            for uid in analysis.rule.head_repartition_kv_producers
            if uid in invocation.values
        })
        shapes = Layer(analysis.program, values)
    return _kernel_shape_layer(analysis, shapes)


def physical_access_shapes(
    analysis: Analysis,
    kernel_shapes: Layer[Shape] | None = None,
) -> Layer[Shape]:
    """Per-rank buffers read at prefill attention redistribution seams.

    Kernel geometry describes the logical input after an all-gather or
    all-to-all.  A fused attention kernel streams the rank-local arriving
    buffer while the materialized collective owns the movement and its output
    traffic.  Keep that distinction as a derived shape layer instead of
    changing either the propagated edge requirement or the kernel shape.
    `kernel_shapes` are the analysis's (`physical_kernel_shapes`), where the
    caller already holds them.
    """
    shapes = (kernel_shapes if kernel_shapes is not None
              else physical_kernel_shapes(analysis))
    family = analysis.rule.resident_sequence_family
    if family is None:
        return shapes

    values = dict(shapes.values)
    for record in analysis.redistributes:
        if record.input_index < 0 or not any(
            world == family.id for world, _collective in record.collectives
        ):
            continue
        node = analysis.program.by_uid(record.node)
        if node is None or node.op not in ATTENTION_KERNEL_OPS:
            continue
        global_shape = analysis.shapes.values.get(node.uid)
        local_shape = values.get(node.uid)
        if global_shape is None or local_shape is None:
            raise ValueError(
                f"{node.display_path}: attention access shape is missing"
            )
        index = record.input_index
        if index >= len(global_shape.ins) or index >= len(local_shape.ins):
            raise ValueError(
                f"{node.display_path}: redistribution input {index} is "
                "outside its shape"
            )
        inputs = list(local_shape.ins)
        inputs[index] = local_dims(
            rank_law_shape(analysis, node, global_shape,
                           local_shape).ins[index],
            record.arrived,
            analysis.rule,
            f"{node.display_path}.access_input[{index}]",
        )
        # One input's dims, in place: `moved` is keyed by input POSITION
        # (`values.Moved`) and no position moves here, so carrying it over is
        # carrying it for the same tensors. (These are the walk's own rows,
        # which carry none — see `physical_shapes`.)
        values[node.uid] = replace(local_shape, ins=tuple(inputs))
    return Layer(analysis.program, values)


def _tracked_shape(global_shape: Any, physical_shape: Any) -> tuple[Any, ...]:
    if len(global_shape) != len(physical_shape):
        raise ValueError(
            f"global/physical artifact ranks differ: {len(global_shape)} and "
            f"{len(physical_shape)}"
        )
    return tuple(
        Dim(expr_of(global_axis), int(local_axis))
        for global_axis, local_axis in zip(
            global_shape, physical_shape, strict=True
        )
    )


def _kernel_shape_layer(
    analysis: Analysis,
    semantic_shapes: Layer[Shape],
) -> Layer[Shape]:
    """Overlay invocation-only kernel geometry on local semantic tensors."""
    kernel_shapes = dict(semantic_shapes.values)
    for uid, physical in analysis.kernel_shapes.values.items():
        node = analysis.program.by_uid(uid)
        if node is None:
            raise ValueError(
                f"kernel shape layer names unknown node uid {uid}"
            )
        if node.children:
            raise ValueError(
                f"{node.display_path}: component cannot declare primitive "
                "kernel geometry"
            )
        semantic = analysis.shapes.values.get(uid)
        if semantic is None:
            raise ValueError(
                f"{node.display_path}: kernel geometry has no semantic shape"
            )
        _validate_kernel_shape(node, semantic, physical)
        kernel_shapes[uid] = physical
    return Layer(analysis.program, kernel_shapes)


def _output_state(analysis: Analysis, uid: int, port: str) -> EdgeState:
    state = analysis.port_states.get((uid, port))
    if state is None and port == "out":
        state = analysis.states.values.get(uid)
    if not isinstance(state, EdgeState):
        raise ValueError(f"node {uid} port {port!r} has no layout output state")
    return state


def local_dims(
    dims: Sequence[Any],
    state: EdgeState,
    sharding: LayoutPropagation,
    label: str,
) -> Tuple[Any, ...]:
    if len(dims) != len(state.layout):
        raise ValueError(
            f"{label}: shape rank {len(dims)} does not match layout rank "
            f"{len(state.layout)}"
        )
    if state.workload is not None:
        # The rows axis of a routed tensor — the one after its expert axis
        # (`n_expert_copies`, asked of the identity) — is the MEAN rows one
        # expert receives, read off the routed-load evidence rather than cut
        # by index arithmetic: a static cut of it (CP's token blocks) splits
        # the mean exactly, rational or odd. The expert axis is cut like any
        # other axis, by its layout. The rank priced is the busiest one: its
        # rows are the mean times the batch's load factor
        # (`busiest_rank_load_factor`, 1 where every rank holds every
        # expert), so what a routed tensor costs a rank is read off them.
        rows_at = next(index for index, dim in enumerate(dims)
                       if is_expert_axis(dim)) + 1
        rows = value_of(state.workload.expected_rows_per_expert
                        * state.workload.busiest_rank_load_factor)
        return tuple(
            (Dim(expr_of(dim), rows) if isinstance(dim, Dim) else rows)
            if index == rows_at else local_extent(dim, layout, sharding)
            for index, (dim, layout) in enumerate(
                zip(dims, state.layout, strict=True)))
    return tuple(
        local_extent(dim, layout, sharding)
        for dim, layout in zip(dims, state.layout, strict=True)
    )


def _proportional(layout: Any) -> bool:
    """Whether one rank's extent is the global extent over a constant: whole,
    an exact cut, or a product of exact cuts."""
    if isinstance(layout, Block):
        return layout.pad_to is None
    return isinstance(layout, (Replicated, Product))


def local_extent(
    extent: Any, layout: Any, sharding: LayoutPropagation
) -> Any:
    """One per-rank extent, including a declared logical singleton owner.

    A normal divisible extent uses the physical layout algebra. A logical
    owner may instead declare that the raw extent is already local (request
    ``B``) or is a selected semantic singleton (sampled output position). It
    retains that local extent until the corresponding boundary materializes
    its collective; unnamed unit extents and derived expressions receive no
    exemption.
    """
    global_value = value_of(extent)
    if isinstance(extent, Dim) and not _proportional(layout):
        # The per-rank value of a padded or segmented cut is not its global
        # value over a constant, so an axis that grows with the request
        # batch would leave the batch-linear reading of every row through
        # it (`RowRoofline`) silently wrong at any other batch.
        batch = {name for name, kind in dim_registry.kinds().items()
                 if kind == "batch"}
        if extent.names & batch:
            raise ValueError(
                f"axis {extent!r} grows with the request batch and is cut "
                f"by {layout!r}, whose per-rank extent is not proportional "
                "to the global one")
    # An inactive branch may carry a well-typed empty tensor.  It has no
    # ownership indices to partition and allocates zero elements on every
    # rank; the positive-extent precondition of the layout index algebra does
    # not apply to this projection. An extent that is zero at some points
    # only is projected by the algebra, which gives zero there.
    if same_law(global_value, 0):
        local_value = 0
    else:
        local_value = sharding.project_local_extent(extent, layout)
    if isinstance(extent, Dim):
        return Dim(expr_of(extent), local_value)
    return local_value


def _eval_expr(expr: Any, dims: Mapping[str, Any]) -> Any:
    return eval_dim_expr(str(expr), dims)


def _shape_axes(
    shape: Shape, *, outputs: bool
) -> Iterator[Any]:
    surfaces = (
        (shape.out, *shape.ports.values()) if outputs else shape.ins
    )
    for axes in surfaces:
        if axes is not None:
            yield from axes


def _surfaces(shape: Shape) -> Tuple[Tuple[Any, ...], ...]:
    return tuple((shape.out, *shape.ports.values(), *shape.ins))


def _bare_name(axis: Any) -> Optional[str]:
    if isinstance(axis, Dim) and isinstance(axis.expr, sp.Symbol):
        return axis.expr.name
    return None


def _proven_extents(global_shape: Shape, local_shape: Shape, node: Any,
                    name: str, law: Any, *,
                    query_shapes: Optional[Tuple[Shape, Shape]] = None,
                    query_node: Any = None) -> Tuple[Dict[str, Any], set]:
    """Each name an axis of the law's own faces IS (`Dim.expr` a bare
    symbol), at its per-rank extent there; and the names two of those axes
    give different per-rank extents, which have no one per-rank value.

    The faces are the ones that declare the defined name — for an attention
    law, the K/V operand carrying `s_kv` — plus, when the law reads a query
    count, the score matrix's rows.  A node's other faces are not evidence:
    Ulysses leaves an attention kernel's boundary sequence-cut while the
    matrix holds every query for the rank's heads, and mixing the two makes
    the query count look ambiguous when the law's own face says it is whole.
    `query_shapes` and `query_node` name that matrix's owner for a fused
    launch, whose own boundary is not the matrix.
    """
    faces = [
        (global_axes, local_axes)
        for global_axes, local_axes in zip(
            _surfaces(global_shape), _surfaces(local_shape), strict=True)
        if global_axes is not None and local_axes is not None
        if any(_bare_name(axis) == name for axis in global_axes)
    ]
    kernel = attention_kernel((query_node if query_node is not None
                               else node).op)
    if faces and any(symbol.name in dim_registry.query_axes()
                     for symbol in law.free_symbols):
        query_global, query_local = (query_shapes
                                     if query_shapes is not None
                                     else (global_shape, local_shape))
        faces.append((query_global._score_rows(kernel),
                      query_local._score_rows(kernel)))
    proven: Dict[str, Any] = {}
    ambiguous: set = set()
    for global_axes, local_axes in faces:
        for global_axis, local_axis in zip(
                global_axes, local_axes, strict=True):
            axis_name = _bare_name(global_axis)
            if axis_name is None:
                continue
            local = value_of(local_axis)
            if axis_name in proven and not same_law(proven[axis_name], local):
                ambiguous.add(axis_name)
            proven.setdefault(axis_name, local)
    return proven, ambiguous


def rank_definitions(definitions: Optional[Mapping[str, Any]],
                     global_shape: Shape, local_shape: Shape,
                     sharding: Any, node: Any, *,
                     query_shapes: Optional[Tuple[Shape, Shape]] = None,
                     query_node: Any = None,
                     cache: Optional[Dict[Tuple[Any, ...], Definition]] = None,
                     ) -> Dict[str, Definition]:
    """Each of a node's definitions (`binding.Definition`, a law of its own
    in the names its frame binds) that comes to another value at this rank,
    as the definition it is at this rank; the rest are absent.

    A law is re-evaluated where a name it reads, other than the name it
    defines, has a per-rank extent on the law's own faces different from
    the one it was bound to (`_proven_extents`). The defined name keeps
    its binding — the frame's value BEFORE the walk narrowed it — so the law is
    applied once, whatever it is (a pooled stream's `⌊s_kv/m⌋` is no
    idempotent cap). Only a name some axis of the node IS is projected:
    the definition then sizes a tensor this rank holds.

    How a cut query count enters it is the attention form's statement of
    which queries a rank holds (`AttentionForm.query_runs`): under prefill
    CP, chunks `i` and `2r − 1 − i` of `2r` equal chunks, `r` the degree
    (the balanced split vLLM and sglang run). A step law reads as a suffix
    reach — `law(s_kv, s_query)` is how many keys the last `s_query`
    queries of a `s_kv`-key context reach, the keys just before the
    context's end (`OpSpec.kv_positions_per_step_fn`; a window's
    `Min(s_kv, s_query + s_window − 1)`). So chunk `j` of `h = q/2`
    queries, ending at `e_j = s_kv − r·q + (j + 1)·h`, reaches the interval
    of `L_j = law(e_j, h)` keys ending at `e_j`, clipped at the context's
    start by the law itself; rank `i` reads the union of its two chunks'
    intervals, `L_a + L_b − Max(0, Min(L_a, e_a − e_b + L_b))` (`a < b`);
    and the engine prices the balanced average, the mean over the `r`
    ranks — the same average `pairs_share` prices FLOPs at:

        (1/r) · Σ_i |reach(i) ∪ reach(2r − 1 − i)|

    That is the direct count of the keys each rank's queries reach,
    averaged, for chunks of exactly `h` queries; a framework's integer
    chunks (`AttentionForm.query_runs`) differ from them by under one query
    each. A cut name that counts no queries takes its rank's extent. The
    definition is written in the rank's names (`s_query` the rank's
    queries, `s_kv` the whole context), bound to the rank's extents, and
    comes to its value there, so a formula that prints it still reads the
    number beside it. A count of score pairs is not projected here: it
    takes `pairs_share`."""
    out: Dict[str, Definition] = {}
    laws = {name: definition for name, definition
            in (definitions or {}).items()
            if isinstance(definition, Definition)
            and not _counts_score_pairs(name)}
    if not laws:
        return out
    queries = dim_registry.query_axes()
    for name, definition in laws.items():
        proven, ambiguous = _proven_extents(
            global_shape, local_shape, node, name, definition.law,
            query_shapes=query_shapes, query_node=query_node)
        if name not in proven:
            # An extent no tensor of the node carries (the phase's `s_past`
            # read at a node): it sizes nothing this rank holds.
            continue
        reads = {symbol.name for symbol in definition.law.free_symbols} - {name}
        cut = {symbol for symbol in reads
               if symbol in proven and symbol in definition.bindings
               and not same_law(proven[symbol], definition.bindings[symbol])}
        if not cut:
            continue
        if cut & ambiguous:
            raise ValueError(
                f"{node.display_path}: `{name}` is the law "
                f"{definition.law}, and {sorted(cut & ambiguous)!r} have no "
                "one per-rank extent on the node's axes")
        bindings = dict(definition.bindings)
        for symbol in cut:
            bindings[symbol] = proven[symbol]
        cut_queries = sorted(cut & queries)
        reading = (name, definition.law,
                   tuple(sorted(bindings.items())), tuple(cut_queries))
        if cache is not None and reading in cache:
            out[name] = cache[reading]
            continue
        law = (definition.law if not cut_queries else _rank_reach(
            definition, name, cut_queries, proven, sharding, node))
        value = law.xreplace({sp.Symbol(symbol): exact(bound)
                              for symbol, bound in bindings.items()})
        out[name] = Definition(
            law, bindings, int(value) if isinstance(value, sp.Integer) else value)
        if cache is not None:
            cache[reading] = out[name]
    return out


def _rank_reach(definition: Definition, name: str, cut_queries: Sequence[str],
                proven: Mapping[str, Any], sharding: Any, node: Any) -> Any:
    """`definition`'s step law as the mean over ranks of the keys a rank's
    query chunks reach (`rank_definitions`), in the rank's names."""
    if len(cut_queries) != 1:
        raise ValueError(
            f"{node.display_path}: `{name}` = {definition.law} reads "
            f"{cut_queries!r}, several query counts the rank holds part of; "
            "a step's queries are one count")
    query = cut_queries[0]
    degree = sp.cancel(exact(definition.bindings[query]) / exact(proven[query]))
    form = ATTENTION_PLANS.get(
        (str(sharding.parallel.cp_scheme), sharding.phase))
    runs = (None if form is None or not isinstance(degree, sp.Integer)
            else form.query_runs(0, int(degree)))
    if runs is None:
        raise ValueError(
            f"{node.display_path}: `{name}` = {definition.law} reads "
            f"`{query}`, cut {degree} ways here, and the attention form "
            f"{form} states no split of the queries over ranks")
    ranks = int(degree)
    context, q = sp.Symbol(name), sp.Symbol(query)
    chunk = q / len(runs)

    def reach(j: int) -> Tuple[Any, Any]:
        end = context - ranks * q + (j + 1) * chunk
        return end, definition.law.xreplace({context: end, q: chunk})

    with sp.evaluate(False):
        total = sp.Integer(0)
        for rank in range(ranks):
            # The earlier chunk first: chunk ends grow with the index.
            (e_a, l_a), (e_b, l_b) = (
                reach(j) for j in sorted(form.query_runs(rank, ranks)))
            total += l_a + l_b - sp.Max(0, sp.Min(l_a, e_a - e_b + l_b))
        out = total / ranks
    return out


def rank_law_shape(analysis: Analysis, node: Any, global_shape: Shape,
                   local_shape: Shape) -> Shape:
    """`global_shape` with each axis whose extent is a law of the node's
    own (`Analysis.node_rebinds`: a `binding.Definition`) re-evaluated at
    this rank's extents — the global extents a rank's layout then cuts.

    The walk is global: it evaluates a law at the step's whole extents and
    the layout carries the cut beside the result. That is exact for a law
    linear in a cut symbol and wrong for any other. A sliding window's
    step reads `Min(s_kv, s_query + s_window − 1)` keys
    (`ops_registry._window_step_keys`); at the global `s_query` that is the
    whole context, and the K/V axis it is written on is whole on every
    prefill CP rank (each gathers the cache), so the layout alone left every
    rank reading every key while its queries reach about `1/cp` of them.

    The rank's law is `rank_definitions` (the rule and the split it is
    evaluated over are stated there). The axis's own layout then cuts the
    result like any other extent (`local_extent`): a decode cache cut by
    DCP divides the window's keys after the law, never the context before
    it (`Min(S, k)/dcp`, not `Min(S/dcp, k)`).

    A count of score pairs is not projected here: it takes `pairs_share`
    (`local_node_dims`). Returns `global_shape` itself where nothing moves.
    """
    rebound = analysis.node_rebinds.values.get(node.uid) or {}
    values = {name: definition.value for name, definition in
              analysis.rank_definitions(
                  rebound, global_shape, local_shape, node).items()}
    if not values:
        return global_shape

    def at_rank(axes: Optional[Sequence[Any]]) -> Optional[Tuple[Any, ...]]:
        if axes is None:
            return None
        return tuple(
            Dim(axis.expr, values[axis.expr.name])
            if isinstance(axis, Dim) and isinstance(axis.expr, sp.Symbol)
            and axis.expr.name in values else axis
            for axis in axes)

    return replace(
        global_shape,
        out=at_rank(global_shape.out),
        ins=tuple(at_rank(axes) for axes in global_shape.ins),
        ports={port: at_rank(axes)
               for port, axes in global_shape.ports.items()})


def pairs_share(op: Optional[str], global_shape: Shape,
                local_shape: Shape) -> Optional[Any]:
    """The share of its score matrix's (query, key) cells (`Shape.pairs`)
    a rank holds for a node of `op` — `op` and the two shapes are the
    matrix's owner's (`fusion.score_matrix_owner`): PCP cuts the rows, DCP
    the keys, Ulysses neither (it cuts heads), and a DSA indexer's rows
    follow the sequence cut under either prefill scheme. None for a node
    whose matrix has no query and key axis: nothing to scale."""
    if local_shape.pair_share is not None:
        return local_shape.pair_share
    if op == "dsa_topk_mask" and global_shape.ins and local_shape.ins:
        # A selector scans its input logits, not the padded K-slot output.
        # Sparse DCP narrows the scan even though K remains unchanged.
        before, after = global_shape.ins[0], local_shape.ins[0]
        return sp.cancel(exact(after[-2] * after[-1]) /
                         exact(before[-2] * before[-1]))
    kernel = attention_kernel(op)
    global_pairs = global_shape.pairs(kernel)
    local_pairs = local_shape.pairs(kernel)
    if global_pairs is None or local_pairs is None:
        return None
    # A ratio of two counts is a `Rational`, and `exact` keeps it one. A
    # bare `/` did not: a decode step's query axis is the literal
    # `n_verify`, so its pair counts are plain Python ints, and `128 / 128`
    # is the float `1.0`. That coefficient rode into the decode law —
    # `1.0*Max(1, s_input + floor(s_output/2 + 1/2))` — so the walk's own
    # `s_pairs` came out `Float(4224)` and published as the string
    # "4224.00000000000" where the same law read at a point published the
    # integer 4224. Prefill's counts are expressions in `s_input`, so
    # sympy's `/` kept them exact and only one of the two phases was wrong.
    # An odd cut is the louder case: a third of the pairs was
    # 0.3333333333333333 going into the per-rank pair count and the FLOPs
    # read off it.
    share = exact(local_pairs) / exact(global_pairs)
    # A cut divides its axis evenly (layout refuses one that does not), so
    # the ratio of the two laws is a number: `(s_kv/2) / s_kv` written
    # out term by term must still read as 1/2, or the definition it scales
    # would read workload names it has no binding for.
    return sp.cancel(share)


def _counts_score_pairs(name: str) -> bool:
    symbol = dim_registry.get(name)
    return symbol is not None and symbol.score_pairs


def local_definitions(definitions: Optional[Mapping[str, Any]], *,
                      analysis: Any, share: Optional[Any],
                      global_shape: Shape, local_shape: Shape,
                      node: Any,
                      query_shapes: Optional[Tuple[Shape, Shape]] = None,
                      query_node: Any = None) -> Optional[Dict[str, Any]]:
    """A node's declared definitions (`binding.Definition`) at this rank,
    beside the frame `local_node_dims` projects with the same `share`
    (`pairs_share`) and shapes, each still coming to the number it
    explains:

      * a count of score pairs, scaled by `share`;
      * a law of an extent the node's tensors carry, as the rank's law
        (`rank_definitions`) and then the cut the axis's own layout makes
        (`rank_law_shape`): the factor from the law's value to the axis's
        per-rank extent — `1/dcp` for a decode cache cut by DCP, 1 for
        prefill CP's gathered K/V. A cut that is no constant factor of the
        law cannot be written as one and is refused;
      * every other one as declared."""
    if not definitions:
        return None if definitions is None else dict(definitions)
    at_rank = analysis.rank_definitions(
        definitions, global_shape, local_shape, node,
        query_shapes=query_shapes, query_node=query_node)
    carried = {}
    for global_axes, local_axes in zip(_surfaces(global_shape),
                                       _surfaces(local_shape), strict=True):
        for global_axis, local_axis in zip(global_axes or (),
                                           local_axes or (), strict=True):
            name = _bare_name(global_axis)
            if name is not None:
                carried.setdefault(name, value_of(local_axis))
    out: Dict[str, Any] = {}
    for name, definition in definitions.items():
        if share is not None and _counts_score_pairs(name):
            # Physical ownership ratios can read workload symbols (rounded
            # local scans / skewed selection), not only constant cuts.
            extra = {s.name: s for s in exact(share).free_symbols}
            out[name] = Definition(definition.law * share,
                                   {**extra, **definition.bindings},
                                   definition.value * share)
            continue
        definition = at_rank.get(name, definition)
        if (isinstance(definition, Definition) and name in carried
                and not same_law(definition.value, 0)):
            factor = sp.cancel(exact(carried[name]) / exact(definition.value))
            if varies(factor):
                # Explicit runtime geometry may be a bounded, nonuniform
                # ownership law rather than a constant tensor partition.
                # Publish that law in workload names, not a fake fixed cut.
                plan = (None if analysis.rule.is_trivial else
                        analysis.rule.sparse_dcp.for_node(node))
                if plan is not None and plan.decode:
                    law = exact(carried[name])
                    out[name] = Definition(
                        law, {s.name: s for s in law.free_symbols}, law)
                    continue
                raise ValueError(
                    f"{node.display_path}: `{name}` = {definition.law} comes "
                    f"to {definition.value}, and the rank's axis holds "
                    f"{carried[name]}, which is no constant share of it")
            if not same_law(factor, 1):
                definition = Definition(definition.law * factor,
                                        definition.bindings,
                                        definition.value * factor)
        out[name] = definition
    return out


def local_node_dims(
    node: Any,
    dims: Mapping[str, Any],
    global_shape: Shape,
    local_shape: Shape,
    *,
    local_symbols: Optional[Mapping[str, Optional[int]]] = None,
    projected_names: Optional[set] = None,
    program: Any = None,
    pairs_share: Optional[Any],
) -> Dict[str, Any]:
    """Project a node binding through the same tracked axes as its shapes.

    ``flops_from_shape`` binds labels carried by kernel inputs.  A projection's
    ``out`` width and other labels no input carries must follow
    the output layout as well.  ``Dim.expr`` is the shared identity: it maps a
    composite binding such as ``out=d_head*n_heads`` to the corresponding
    global/local output axis without knowing the op or the dim name.

    A dim whose identity matches no shape axis — a bare ``n_heads`` that
    exists only as a FACTOR of a merged axis — keeps the global value unless
    ``local_symbols`` answers it. ``local_symbols`` carries per-rank symbol
    extents already proven by the node's weight-axis contracts
    (``WeightEndpoint.runtime_symbol_extents``); a dim whose free symbols it
    covers fully is re-evaluated under them, so the per-rank value comes from
    the proven fact rather than from the unsharded environment. A symbol the
    facts map to None has no one per-rank value (it sizes two axes split
    differently); a binding that needs it and that no axis or input slot
    answers is refused rather than kept global.

    A slot the node declares in ``dims:`` that the op's input template
    names (the node's op in `program`, when given: a linear's ``in``) takes
    the extent of the input axis it labels, the precedence
    ``flops_from_shape`` binds with. Identity alone cannot place it when one
    symbol sizes both ends — a square projection split on its output only.

    A count of the pairs the node's score matrix holds (`DimSymbol.
    score_pairs`) is no axis of any shape; it takes ``pairs_share``, the
    share of the matrix's cells this rank holds (`pairs_share`, off the
    matrix's owner — a fused kernel's absorbed attention kernel). Required:
    a caller that forgot it would bill a rank every pair of the step.

    ``projected_names``, when given, collects the names whose value this
    projection actually rewrote — from a shape axis or from the facts — so a
    caller guarding the evaluation can tell "kept the global value" apart
    from "per-rank value, arrived by projection".
    """
    def index(*, outputs: bool) -> Dict[Any, list[Any]]:
        global_axes = tuple(_shape_axes(global_shape, outputs=outputs))
        local_axes = tuple(_shape_axes(local_shape, outputs=outputs))
        if len(global_axes) != len(local_axes):
            raise ValueError(
                f"{node.display_path}: global/local shape surfaces have "
                f"{len(global_axes)} and {len(local_axes)} axes"
            )
        by_expr: Dict[Any, list[Any]] = {}
        for global_axis, local_axis in zip(
            global_axes, local_axes, strict=True
        ):
            if not isinstance(global_axis, Dim):
                continue
            if (
                not isinstance(local_axis, Dim)
                or local_axis.expr != global_axis.expr
            ):
                raise ValueError(
                    f"{node.display_path}: local axis {local_axis!r} lost "
                    f"global identity {global_axis.expr!r}"
                )
            by_expr.setdefault(global_axis.expr, []).append(local_axis)
        return by_expr

    output_by_expr = index(outputs=True)
    input_by_expr = index(outputs=False)

    projected = dict(dims)
    ambiguous: set = set()

    def local(name: str, value: Dim) -> Any:
        """`value` at this rank: the extent its identity has on one of the
        node's axes, else the per-rank values of its symbols; `value` as it
        is when neither answers, and the binding marked ambiguous when a
        symbol of it has no one per-rank value."""
        candidates = output_by_expr.get(value.expr) or input_by_expr.get(
            value.expr, ()
        )
        if not candidates:
            if local_symbols:
                symbols = value.expr.free_symbols
                if any(sym.name in local_symbols
                       and local_symbols[sym.name] is None
                       for sym in symbols):
                    ambiguous.add(name)
                elif symbols and all(
                    sym.name in local_symbols for sym in symbols
                ):
                    resolved = {
                        sym.name: local_symbols[sym.name] for sym in symbols
                    }
                    if projected_names is not None:
                        projected_names.add(name)
                    return Dim(
                        value.expr,
                        eval_dim_expr(str(value.expr), resolved),
                    )
            return value
        first = value_of(candidates[0])
        for candidate in candidates[1:]:
            if same_law(candidate, first):
                continue
            require_equal(candidate, first, lambda at: ValueError(
                f"{node.display_path}: binding {name!r} with identity "
                f"{value.expr!r} maps to conflicting local extents "
                f"{[at(value_of(c)) for c in candidates]!r}"))
        if projected_names is not None:
            projected_names.add(name)
        return Dim(value.expr, first)

    for name, value in dims.items():
        if isinstance(value, Dim):
            projected[name] = local(name, value)
        elif name == "_shape" and isinstance(value, (list, tuple)):
            # A declared tensor shape (`shape:`) is a list of extents, each
            # projected as a binding of its own would be.
            projected[name] = [
                local(name, extent) if isinstance(extent, Dim) else extent
                for extent in value]
    if pairs_share is not None:
        for name, value in dims.items():
            if isinstance(value, Dim) and _counts_score_pairs(name):
                projected[name] = Dim(value.expr,
                                      value_of(value) * pairs_share)
                if projected_names is not None:
                    projected_names.add(name)
    # A fused kernel has no declaration of its own; its slots are its
    # origins'.
    op = (None if program is None or node.declaration is None
          else OpRef.resolve(node.op, program, axis=node.axis))
    if op is not None:
        labels, _batch = bind_labels(
            op, local_shape.ins, require_consistent_labels=False)
        slots = set(node.dims_raw or ())
        for name, value in dims.items():
            if isinstance(value, Dim) and name in labels and name in slots:
                projected[name] = Dim(value.expr, value_of(labels[name]))
                ambiguous.discard(name)
                if projected_names is not None:
                    projected_names.add(name)
    if ambiguous:
        raise ValueError(
            f"{node.display_path}: bindings {sorted(ambiguous)!r} need a "
            "symbol that sizes two of the node's axes split differently, so "
            "it has no one per-rank value, and no shape axis or input slot "
            "of the node gives theirs")
    return projected
