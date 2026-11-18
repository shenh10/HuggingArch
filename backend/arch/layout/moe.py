"""Expert dispatch, combine, and router layout handlers."""
from __future__ import annotations

from .zone import COMBINE_OP, DISPATCH_OP

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from .propagation import LayoutPropagation

from backend.arch.eval.dim import Dim, exact, expr_of, require_equal, value_of
from backend.arch.eval.transition import OutputStates

from .axis_use import AXIS_USE
from .comm_volume import RouteFanout
from .edge import EdgeState
from .linear import normalize_axis
from .groups import same
from .layout import axis_fully_replicated, sharded
from .value import SUM, Partial, join_orthogonal_partials
from .workload import ExpectedRoutedLoad


def _pair_rank3(rule: LayoutPropagation, node: Any,
                states: Sequence[Optional[EdgeState]],
                what: str) -> Tuple[EdgeState, EdgeState]:
    del rule
    if len(states) != 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: {what} expects two resolved inputs")
    left, right = states
    if len(left.layout) != 3 or len(right.layout) != 3:
        raise ValueError(
            f"{node.display_path}: {what} expects rank-3 inputs, got "
            f"{len(left.layout)} and {len(right.layout)}")
    return left, right


def topk(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
         in_shapes: Sequence[Any]
         ) -> Tuple[List[EdgeState], EdgeState]:
    required, state = normalize_axis(rule, node, states, in_shapes)
    return required, EdgeState(
        (*state.layout[:-1], rule.replicated()), state.value,
        state.workload)


def group_top2_sum(rule: LayoutPropagation, node: Any,
                   states: List[Optional[EdgeState]],
                   in_shapes: Sequence[Any]
                   ) -> Tuple[List[EdgeState], EdgeState]:
    required, state = normalize_axis(rule, node, states, in_shapes)
    return required, EdgeState(
        (*state.layout[:-1], rule.replicated()), state.value,
        state.workload)


def gather(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
           in_shapes: Sequence[Any]
           ) -> Tuple[List[EdgeState], EdgeState]:
    del in_shapes
    data, index = _pair_rank3(rule, node, states, "gather")
    for axis, (left, right) in enumerate(
            zip(data.layout[:-1], index.layout[:-1], strict=True)):
        rule.align(node, left, right, f"token axis {axis}")
    if not axis_fully_replicated(data.layout[-1]) \
            or not axis_fully_replicated(index.layout[-1]):
        raise ValueError(
            f"{node.display_path}: gather selection/data axes must be "
            "whole")
    return list(states), EdgeState(
        (*data.layout[:-1], rule.replicated()), data.value,
        data.workload)


def group_mask_apply(rule: LayoutPropagation, node: Any,
                     states: List[Optional[EdgeState]],
                     in_shapes: Sequence[Any]
                     ) -> Tuple[List[EdgeState], EdgeState]:
    del in_shapes
    scores, selected = _pair_rank3(
        rule, node, states, "group_mask_apply")
    for axis, (left, right) in enumerate(
            zip(scores.layout[:-1], selected.layout[:-1], strict=True)):
        rule.align(node, left, right, f"token axis {axis}")
    if not axis_fully_replicated(scores.layout[-1]) \
            or not axis_fully_replicated(selected.layout[-1]):
        raise ValueError(
            f"{node.display_path}: grouped routing axes must be whole")
    return list(states), scores


def hash_route(rule: LayoutPropagation, node: Any,
               states: List[Optional[EdgeState]],
               in_shapes: Sequence[Any]
               ) -> Tuple[List[EdgeState], EdgeState]:
    """Look up each token's expert ids: `tid2eid[input_ids]`. The ids' axes
    carry through as they arrived; the table is held whole — every id may
    land on any row — and the expert slots it appends are whole."""
    if len(states) != 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: hash_route expects token ids and a "
            "lookup table")
    if len(in_shapes) != 2 or len(in_shapes[0]) != 2 \
            or len(in_shapes[1]) != 2:
        raise ValueError(
            f"{node.display_path}: hash_route expects rank-2 token ids "
            "and a rank-2 [n_vocab, n_experts_active] lookup table")
    ids, table = states
    if table.value is not None or table.workload is not None:
        raise ValueError(
            f"{node.display_path}: hash lookup table must be complete "
            "and carry no routed workload")
    binding = rule.env.get(node.uid) or {}
    active = binding.get("n_experts_active")
    vocab = binding.get("n_vocab")
    if active is None or vocab is None \
            or [int(value_of(d)) for d in in_shapes[1]] != [
                int(value_of(vocab)), int(value_of(active))]:
        raise ValueError(
            f"{node.display_path}: lookup-table shape does not match "
            "[n_vocab, n_experts_active]")
    table_need = rule.whole_axis(
        node, table, 0, AXIS_USE["hash_route"]["n_vocab"],
        "token-to-expert lookup table")
    table_need = rule.whole_axis(
        node, table_need, 1, None, "token-to-expert lookup row")
    return [ids, table_need], EdgeState(
        (*ids.layout, rule.replicated()), None, ids.workload)


def permute(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
            in_shapes: Sequence[Any]
            ) -> Tuple[List[EdgeState], EdgeState]:
    """Dispatch `(B, s_ctx, D)` into the `[n_expert_copies, M, D]` buffer.

    The expert axis is cut over the route group — a rank holds its own
    experts' rows, as it holds their weights. The two token axes become the
    per-expert MEAN rows `M`: cuts invariant inside a route group survive on
    that axis, the other cuts are absorbed by dispatch. Runtime row imbalance
    is not a static axis layout, so the expected load travels beside the
    state until combine (`ExpectedRoutedLoad`: the mean, and the busiest
    rank's).

    The same node binding says how the dispatch FANS OUT (`RouteFanout`:
    the picks, the destinations they are drawn from, and the ones that hold
    weights), which is what the exchange of those rows is priced by — a
    token crosses the fabric once per distinct destination its picks land
    on, not once per pick.
    """
    del in_shapes
    token, routing = _pair_rank3(rule, node, states, DISPATCH_OP)
    if token.value is not None or routing.value is not None:
        raise ValueError(
            f"{node.display_path}: dispatch requires complete token and "
            "routing operands")
    if token.workload is not None or routing.workload is not None:
        raise ValueError(
            f"{node.display_path}: nested routed zones are not allowed")
    for axis, (left, right) in enumerate(
            zip(token.layout[:2], routing.layout[:2], strict=True)):
        rule.align(node, left, right, f"token axis {axis}")
    if not axis_fully_replicated(routing.layout[-1]):
        raise ValueError(
            f"{node.display_path}: routing selection axis must be whole")

    routed_axis, preserved_cells = rule.routed_rows_layout(token)
    # Expected routed load travels on the routed rows in every deployment:
    # it is what the rows axis MEANS (a mean count per expert), with or
    # without a route family to exchange over.
    # `M`, the mean rows per expert, is the walk's: it bound it on this
    # node from the token tensor that arrived and the routing counts
    # (`Evaluator._tokens_per_expert`, the DP-attention gather included),
    # and handed the same node binding here. Every weighted expert receives
    # that mean, so the rows arriving at the bank are `M` per expert —
    # `n_route_targets - n_experts` destinations hold no weights (LongCat's
    # zero-compute reserve) and receive none of them.
    binding = rule.env.get(node.uid) or {}
    try:
        # The experts the buffer's axis 0 holds — `permute`'s declared
        # `n_expert_copies`, the bank's own count.
        experts = int(value_of(binding["n_expert_copies"]))
        mean = binding["M"]
        # The same binding says how the dispatch fans out: how many
        # destinations the top-k draws from, and how many it picks. Read
        # here, where the routed rows are born, because the exchange that
        # moves them cannot ask the router afterwards.
        picks = int(value_of(binding["n_experts_active"]))
        targets = int(value_of(binding["n_route_targets"]))
    except KeyError as exc:
        raise ValueError(
            f"{node.display_path}: routed workload dimension "
            f"{exc.args[0]!r} is unbound") from exc
    # The mean keeps its expression: the rows a route exchange moves are
    # read off it at any batch (`RowRoofline`), its value exact.
    per_expert = (Dim(expr_of(mean), exact(mean))
                  if isinstance(mean, Dim) else exact(mean))
    global_rows = per_expert * experts
    routed_rows = global_rows / preserved_cells
    # Without expert parallelism every rank holds every expert; the mean
    # still travels, so the rows axis is read as the average it is (a static
    # cut of it — CP's token blocks — splits the mean exactly).
    route = rule.route_family
    expert_axis = sharded(route) if route is not None else None
    workload = ExpectedRoutedLoad(
        group=route,
        expert_shard_axis=expert_axis,
        routed_rows_in_cell=routed_rows,
        num_experts=experts,
        load_assumption_id="moe_routing_balance",
        fanout=RouteFanout(picks=picks, targets=targets, placed=experts),
        expert_load_factor=rule.expert_load,
        dispatched_rows=routed_axis)
    if not workload.is_measure_preserving():
        raise AssertionError(
            f"{node.display_path}: routed-load measure was not "
            "preserved")

    produced = EdgeState(
        (expert_axis if expert_axis is not None else rule.replicated(),
         routed_axis, token.layout[-1]), None, workload)
    return list(states), produced


def combine(rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
            in_shapes: Sequence[Any]
            ) -> Tuple[List[EdgeState], EdgeState]:
    """Inverse dispatch and weighted expert reduction.

    With request-owner inclusion the expert ranks already hold every
    token and combine creates a pending EP sum. Without inclusion, the
    inverse all-to-all returns complete values to request owners and no
    EP Partial is minted. An orthogonal TP expert Partial survives and
    joins the EP sum when both reductions are pending.
    """
    del in_shapes
    expert, routing = _pair_rank3(rule, node, states, COMBINE_OP)
    if routing.value is not None or routing.workload is not None:
        raise ValueError(
            f"{node.display_path}: routing operand must be complete and "
            "outside the routed zone")
    if rule.route_family is not None:
        if expert.workload is None:
            raise ValueError(
                f"{node.display_path}: routed expert result lost its "
                "ExpectedRoutedLoad evidence")
        if not same(expert.workload.group, rule.route_family):
            raise ValueError(
                f"{node.display_path}: routed workload belongs to "
                f"{expert.workload.group.id!r}, not route family "
                f"{rule.route_family.id!r}")
    elif expert.workload is not None and expert.workload.group is not None:
        raise ValueError(
            f"{node.display_path}: routed workload names route family "
            f"{expert.workload.group.id!r}, which is not live")

    required = list(states)
    # The inverse dispatch takes the rows back as the dispatch left them —
    # the rows axis the dispatch published on its load evidence. Rows a
    # layout gathered inside the zone go back to that cut — a pending sum
    # over the gathered ranks is then a reduce_scatter, a complete value a
    # local slice — and anything the exchange moves is a rank's own. Where
    # nothing was gathered the rows are already that cut.
    rows = (expert.workload.dispatched_rows
            if expert.workload is not None else None)
    if rows is None:
        raise ValueError(
            f"{node.display_path}: routed expert result carries no rows "
            "axis from its dispatch (ExpectedRoutedLoad.dispatched_rows)")
    if expert.layout[1] != rows:
        expert = EdgeState(
            (expert.layout[0], rows, expert.layout[2]), expert.value,
            expert.workload.as_dispatched()
            if expert.workload is not None else None,
            expert.provenance)
        required[0] = expert
    inclusion = (rule.route_family is not None
                 and rule.route_owner_inclusion(routing))
    out_value = expert.value
    if inclusion:
        routed_sum = Partial(
            SUM, rule.route_family).with_provenance(node.uid)
        if out_value is None:
            out_value = routed_sum
        else:
            try:
                out_value = join_orthogonal_partials(
                    out_value, routed_sum).with_provenance(node.uid)
            except ValueError:
                # Two non-orthogonal debts cannot inhabit one Partial.
                # Finalize the expert debt on the input edge, then mint
                # the route-family sum at the inverse dispatch.
                required[0] = expert.finalized()
                out_value = routed_sum
    return required, EdgeState(
        (*routing.layout[:2], expert.layout[-1]), out_value, None)


def hc_split_sinkhorn(
        rule: LayoutPropagation, node: Any, states: List[Optional[EdgeState]],
        in_shapes: Sequence[Any],
        ) -> Tuple[List[EdgeState], OutputStates]:
    """Normalize a mix vector and publish its three HC tensor ports."""
    ranks = (3, 1, 1)
    if len(states) != len(ranks) or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: hc_split_sinkhorn expects mixes, "
            "scale, and base inputs")
    if len(in_shapes) != len(ranks) or any(
            shape is None or len(shape) != rank
            for shape, rank in zip(in_shapes, ranks, strict=True)):
        raise ValueError(
            f"{node.display_path}: hc_split_sinkhorn input ranks must be "
            f"{ranks!r}")
    mixes, scale, base = states
    mix_extent = int(value_of(in_shapes[0][-1]))
    if int(value_of(in_shapes[1][0])) != 3 \
            or int(value_of(in_shapes[2][0])) != mix_extent:
        raise ValueError(
            f"{node.display_path}: scale/base shapes do not match the "
            "HC mix vector")
    binding = rule.env.get(node.uid) or {}
    copies = binding.get("n_hidden_copies")
    if copies is None:
        raise ValueError(
            f"{node.display_path}: n_hidden_copies is not resolved")
    copies = int(value_of(copies))
    if mix_extent != copies * (copies + 2):
        raise ValueError(
            f"{node.display_path}: d_hc_mix={mix_extent} does not equal "
            f"n_hidden_copies*(n_hidden_copies+2)={copies * (copies + 2)}")
    if scale.value is not None or base.value is not None \
            or scale.workload is not None or base.workload is not None:
        raise ValueError(
            f"{node.display_path}: HC scale/base parameters must be "
            "complete and carry no routed workload")

    use = AXIS_USE["hc_split_sinkhorn"]["d_hc_mix"]
    required = [
        rule.whole_axis(node, mixes, 2, use,
                         "d_hc_mix (sinkhorn normalization)"),
        rule.whole_axis(node, scale, 0, use,
                         "scale coefficients"),
        rule.whole_axis(node, base, 0, use,
                         "d_hc_mix base coefficients"),
    ]
    copy_axis = rule.replicated()
    leading = required[0].layout[:2]
    vector = EdgeState((*leading, copy_axis), None, mixes.workload)
    matrix = EdgeState(
        (*leading, copy_axis, copy_axis), None, mixes.workload)
    return required, OutputStates(
        vector, (("post", vector), ("comb", matrix)))


def hc_mix(rule: LayoutPropagation, node: Any,
           states: List[Optional[EdgeState]],
           in_shapes: Sequence[Any]
           ) -> Tuple[List[EdgeState], EdgeState]:
    """Batched ``[..., i, j] x [..., j, h]`` HC contraction."""
    if len(states) != 2 or any(state is None for state in states):
        raise ValueError(
            f"{node.display_path}: hc_mix expects comb and residual "
            "inputs")
    if len(in_shapes) != 2 or any(
            shape is None or len(shape) != 4 for shape in in_shapes):
        raise ValueError(
            f"{node.display_path}: hc_mix expects two rank-4 inputs")
    comb, residual = states
    for left, right in ((in_shapes[0][0], in_shapes[1][0]),
                        (in_shapes[0][1], in_shapes[1][1]),
                        (in_shapes[0][3], in_shapes[1][2])):
        require_equal(
            left, right,
            lambda at, left=left, right=right: ValueError(
                f"{node.display_path}: hc_mix batch, sequence, or contracted "
                f"extents disagree ({at(value_of(left))} vs "
                f"{at(value_of(right))})"))
    rule.align(node, comb.layout[0], residual.layout[0], "B")
    rule.align(node, comb.layout[1], residual.layout[1], "s_ctx")
    rule.align(
        node, comb.layout[3], residual.layout[2],
        "contracted n_hidden_copies")
    if comb.workload != residual.workload:
        raise ValueError(
            f"{node.display_path}: hc_mix inputs carry different "
            "workload evidence")
    pending = rule.contract(
        node, comb.layout[3], AXIS_USE["hc_mix"]["n_hidden_copies"],
        "n_hidden_copies contraction")
    return list(states), EdgeState(
        (comb.layout[0], comb.layout[1], comb.layout[2],
         residual.layout[3]),
        pending, residual.workload)
