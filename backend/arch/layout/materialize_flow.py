"""Project global extents + layouts to per-rank flow shapes."""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from backend.arch import dims as dim_registry
from backend.arch.eval.dim import (Dim, ceil_div, expr_of, value_of,
                                   whole_count)

from .materialize_comm import RewriteTarget
from .physical import local_dims, local_extent


def _output_axes(shape: Any, port: str = "out"):
    if shape is None:
        return None
    method = getattr(shape, "axes_for", None)
    if callable(method):
        return method(port)
    key = port or "out"
    if key == "out":
        return getattr(shape, "out", None)
    ports = getattr(shape, "ports", None) or {}
    return ports.get(key)


def local_shape(mat: Any, uid: int, state: Any, port: str = "out"
) -> List[Any]:
    shape = mat.analysis.shapes.values.get(uid)
    axes = _output_axes(shape, port)
    if axes is None:
        raise ValueError(
            f"{mat.analysis.program.by_uid(uid).display_path}: no global shape for port "
            f"{port!r} on the analysis"
        )
    if len(state.layout) != len(axes):
        raise ValueError(
            f"{mat.analysis.program.by_uid(uid).display_path}.{port}: state rank "
            f"{len(state.layout)} does not match shape rank {len(axes)}"
        )
    dims = list(local_dims(
        axes, state, mat.analysis.rule,
        f"{mat.analysis.program.by_uid(uid).display_path}.{port}"))
    return dims

def local_input_shape(mat: Any, uid: int, input_index: int, state: Any
) -> List[Any]:
    shape = mat.analysis.shapes.values.get(uid)
    axes = None
    if shape is not None and 0 <= input_index < len(shape.ins):
        axes = shape.ins[input_index]
    if axes is None:
        raise ValueError(
            f"{mat.analysis.program.by_uid(uid).display_path}: no global input shape at "
            f"position {input_index} on the analysis"
        )
    if len(state.layout) != len(axes):
        raise ValueError(
            f"{mat.analysis.program.by_uid(uid).display_path}.input[{input_index}]: state "
            f"rank {len(state.layout)} does not match shape rank "
            f"{len(axes)}"
        )
    dims = list(local_dims(
        axes, state, mat.analysis.rule,
        f"{mat.analysis.program.by_uid(uid).display_path}.input[{input_index}]"))
    return dims

def physical_extent(mat: Any, extent: Any, layout: Any) -> Any:
    """One rank's extent of a moved buffer, typed: the layout's local extent
    of the global axis — a `Dim` wherever the axis is tracked — whose value
    must be a nonnegative integer."""
    local = local_extent(extent, layout, mat.analysis.rule)
    whole_count(local, lambda at: ValueError(
        f"physical communication extent must be a nonnegative "
        f"integer, got {at(value_of(local))!r}"))
    return local

def routed_flow_shape(mat: Any,
    uid: int,
    state: Any,
    port: str = "out",
) -> List[Any]:
    """Materialize a routed collective's buffer as one packed per-rank row
    buffer, `[E_local · M, D]`, read off the routed state it moves.

    The routed tensor is `[n_expert_copies, M, D]`; a rank sends and receives
    its own experts' rows. `M` is the state's own routed-load evidence
    (`physical.local_dims`: the mean times the busiest rank's factor, times
    the rows a gather stacked, `ExpectedRoutedLoad.row_replication`) and may
    be rational — the mean of row counts that differ per expert — and so may
    the rank's total `E_local · M`: what an attention replica sends is whole
    (`B·s·k`), but under attention tensor parallelism a replica is `t`
    ranks, and the `B·s·k·dp/ep = B·s·k/t` rows a rank receives on average
    need not be (one token, top-6, `t = 4`: 1.5). A rank's buffer holds whole
    rows — the next whole row above its expectation — and a gathered buffer
    is the gathered ranks' whole-row buffers stacked, so the ceiling is
    taken per stacked copy. Per-expert padding belongs to the kernel
    boundary that declares it (`RoutedLaunch.launch_rows`).
    """
    shape = mat.analysis.shapes.values.get(uid)
    axes = _output_axes(shape, port)
    where = f"{mat.analysis.program.by_uid(uid).display_path}.{port}"
    if axes is None or len(axes) != len(state.layout):
        raise ValueError(
            f"{where}: routed communication needs the routed tensor's shape "
            "and layout")
    if state.workload is None:
        raise ValueError(
            f"{where}: routed communication needs the routed state's "
            "load evidence")
    expert = axis_of_kind(mat, uid, "expert_copies", port)
    if expert + 3 != len(axes):
        raise ValueError(
            f"{where}: routed communication moves an `[n_expert_copies, M, "
            f"D]` buffer, and {list(axes)!r} does not end in one")
    local = local_dims(axes, state, mat.analysis.rule, where)
    exact_rows = value_of(local[expert]) * value_of(local[expert + 1])
    copies = state.workload.row_replication
    total = ceil_div(exact_rows, copies) * copies
    mean = axes[expert + 1]
    rows = (Dim(expr_of(axes[expert]) * expr_of(mean) * copies, total)
            if isinstance(mean, Dim) else total)
    feature = physical_extent(mat, axes[-1], state.layout[-1])
    return [rows, feature]

def rewrite_flow_shape(mat: Any, targets: Sequence[RewriteTarget]
) -> List[Any]:
    """The exact tensor shape every rewritten consumer input accepts."""
    projected: List[Tuple[RewriteTarget, List[int]]] = []
    structural: List[RewriteTarget] = []
    for target in targets:
        state = mat.analysis.input_states.get(
            (target.node_uid, target.input_index)
        )
        if state is None:
            node = mat.analysis.program.by_uid(target.node_uid)
            if not node.children:
                raise ValueError(
                    f"{node.display_path} input {target.input_index} has "
                    "no consumer-required state"
                )
            structural.append(target)
            continue
        projected.append((
            target,
            local_input_shape(mat,
                target.node_uid, target.input_index, state
            ),
        ))
    if not projected:
        raise ValueError("rewrite targets expose no physical kernel input")
    expected = projected[0][1]
    mismatched = [
        (target, shape)
        for target, shape in projected
        if shape != expected
    ]
    if mismatched:
        raise ValueError(
            f"rewrite targets disagree on physical input shape: "
            f"{projected!r}"
        )
    canonical_state = mat.analysis.input_states[
        (projected[0][0].node_uid, projected[0][0].input_index)
    ]
    for target in structural:
        shape = local_input_shape(mat,
            target.node_uid, target.input_index, canonical_state
        )
        if shape != expected:
            raise ValueError(
                f"structural rewrite target {target!r} projects to "
                f"{shape!r}, not kernel input {expected!r}"
            )
    return expected

def model_output_flow_shape(mat: Any, uid: int, state: Any, port: str
) -> Tuple[List[Any], Tuple[Tuple[int, str], ...]]:
    """Project an output, preserving the sampled-position owner token."""
    shape = mat.analysis.shapes.values.get(uid)
    axes = _output_axes(shape, port)
    if axes is None or len(axes) != len(state.layout):
        return local_shape(mat, uid, state, port), ()
    logical_axis_owner = getattr(
        mat.analysis.rule, "logical_axis_owner", None
    )
    if not callable(logical_axis_owner):
        raise TypeError("the sharding rule exposes no logical-axis law")
    out: List[Any] = []
    logical_owners: List[Tuple[int, str]] = []
    for axis, (extent, layout) in enumerate(
        zip(axes, state.layout, strict=True)
    ):
        owner = logical_axis_owner(extent)
        out.append(physical_extent(mat, extent, layout))
        if (
            owner is not None
            and layout == owner.layout
            and owner.can_project(extent)
        ):
            logical_owners.append((axis, owner.name))
    return out, tuple(logical_owners)

def axis_of_kind(mat: Any, uid: int, kind: str, port: str = "out") -> int:
    """Locate one axis by the shared semantic dimension registry."""
    shape = mat.analysis.shapes.values.get(uid)
    axes = _output_axes(shape, port)
    kinds = dim_registry.kinds()
    matches = [
        axis
        for axis, extent in enumerate(axes or ())
        if any(kinds.get(name) == kind for name in getattr(extent, "names", ()))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{mat.analysis.program.by_uid(uid).display_path}: expected exactly one "
            f"{kind!r} axis, got {matches!r} in {tuple(axes or ())!r}"
        )
    return matches[0]
