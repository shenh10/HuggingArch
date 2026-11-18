"""Ordered MoE collectives read one routed-load record and chain correctly."""
from __future__ import annotations

import json
import math

import pytest
from fractions import Fraction
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.materializer import collect_events
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step
from tests.snapshots import shared_loader

SNAP = PROJECT_ROOT / "backend" / "arch" / "snapshots" / "Qwen__Qwen3-30B-A3B.json"


#: The workload every case here runs: prefill, 2 requests of 128 tokens a
#: request rank; decode, 16 requests a request rank, one token each.
_PREFILL = dict(phase="prefill", batch=2, input_tokens=128, output_tokens=0)
_DECODE = dict(phase="decode", batch=16, input_tokens=128, output_tokens=8)


def _step(phase="prefill", precision=None, **parallel):
    ep_load_factor = parallel.pop("ep_load_factor", 1)
    config = json.loads(SNAP.read_text())["config"]
    resolved = shared_loader().match(config, model_id="Qwen/Qwen3-30B-A3B")
    step = evaluate_step(
        resolved, ParallelConfig(**parallel),
        **(_PREFILL if phase == "prefill" else _DECODE),
        ep_load_factor=ep_load_factor, precision=precision)
    return step


def _events(**parallel):
    return collect_events(_step(**parallel).analysis)


def test_expert_tp_reduction_then_route_exchange_is_one_ordered_chain():
    """The expert-TP partial sum goes back to the rows as dispatched — a
    reduce_scatter over the gathered ranks — and the inverse exchange moves
    exactly what it hands out, ordered after it."""
    events = _events(tp_moe_expert=2, ep_moe=2, dp_attention=4)
    reduction = next(event for event in events if (
        event.op == "reduce_scatter" and event.world == "tp_moe_expert"))
    exchange = next(event for event in events if (
        event.op == "all_to_all" and event.world == "ep_moe"
        and event.source_event == reduction.event_id))

    assert len(exchange.flow_in_shape) == 2
    assert exchange.flow_in_shape == reduction.flow_out_shape
    assert exchange.flow_out_shape == reduction.flow_out_shape
    assert exchange.payload_bytes * reduction.group_size \
        == reduction.payload_bytes


def test_routing_skew_reads_one_busiest_rank_record():
    balanced_step = _step(tp_moe_expert=2, ep_moe=2, dp_attention=4)
    skewed_step = _step(tp_moe_expert=2, ep_moe=2, dp_attention=4,
                        ep_load_factor=2.0)
    balanced = collect_events(balanced_step.analysis)
    skewed = collect_events(skewed_step.analysis)

    def payloads(events):
        return {
            event.event_id: event.payload_bytes
            for event in events
            if event.world in ("tp_moe_expert", "ep_moe")
        }

    left, right = payloads(balanced), payloads(skewed)
    assert set(left) == set(right)
    loads = [
        state.workload for state in skewed_step.analysis.states.values.values()
        if state.workload is not None
    ]
    load = loads[0]
    factor = Fraction(str(load.busiest_rank_load_factor))
    assert factor != 1
    by_id = {event.event_id: event for event in balanced}
    for key in left:
        # Each rank's buffer is whole rows at its own dispatched share; a
        # gather (or its inverse reduce_scatter) moves several ranks' of them.
        rows = min(int(by_id[key].flow_in_shape[0]),
                   int(by_id[key].flow_out_shape[0]))
        expected = math.ceil(rows * factor) / rows
        assert right[key] == expected * left[key]


def test_two_world_token_cut_exchanges_like_one_world():
    """A token cut made of TWO worlds (`cp` and `dp_attention`, which
    together tile `ep_moe` without either equalling it) crosses the MoE seam
    as the same all_to_all pair a one-world cut does, each routed row moved
    once. The oracle is arithmetic off the config, not the layout algebra: a
    rank holds `B * s / cp` tokens (B per request rank), each routed to
    `num_experts_per_tok` experts as a `hidden_size` row of BF16. Red when
    the dispatch reads the two-world cut as replicated (a free split) and
    the return all_gathers the expert shard instead."""
    config = json.loads(SNAP.read_text())["config"]
    row_bytes = config["num_experts_per_tok"] * config["hidden_size"] * 2
    batch, seq, ep = 2, 128, 4
    for parallel in (dict(cp=2, dp_attention=2, ep_moe=ep),
                     dict(dp_attention=ep, ep_moe=ep),
                     dict(cp=ep, ep_moe=ep)):
        seam = [event for event in _events(**parallel)
                if event.world == "ep_moe"]
        tokens = batch * seq // parallel.get("cp", 1)
        assert sorted(event.op for event in seam) == ["all_to_all"] * 2, (
            parallel, [(event.op, event.world) for event in seam])
        for event in seam:
            assert event.group_size == ep
            assert event.payload_bytes == tokens * row_bytes, (
                parallel, event.payload_bytes, tokens * row_bytes)


def _own_rows(parallel, phase, rho):
    """The routed rows one rank holds after the exchange, from the config
    alone: its `T` tokens (attention runs at tensor parallelism 1, so a
    rank's tokens are its own: `B * s / cp` in prefill, `B` in decode)
    times `num_experts_per_tok`, on the busiest rank — the one holding the
    hottest expert at measured skew `rho` beside `E/ep - 1` average ones —
    in whole rows."""
    config = json.loads(SNAP.read_text())["config"]
    workload = _PREFILL if phase == "prefill" else _DECODE
    tokens = (workload["batch"] * workload["input_tokens"]
              // parallel.get("cp", 1) if phase == "prefill"
              else workload["batch"])
    experts, ep = config["num_experts"], parallel.get("ep_moe", 1)
    local = experts // ep
    rho = Fraction(str(rho))
    busiest = (Fraction(1) if local == experts
               else (rho + (local - 1) * (experts - rho)
                     / Fraction(experts - 1)) / local)
    return math.ceil(tokens * config["num_experts_per_tok"] * busiest)


def _expected_seam(parallel, phase="prefill", rho=1):
    """Every collective of one MoE layer's routed zone, per rank, from the
    config alone — no layout algebra. The expert bank is cut
    `tp_moe_expert x ep_moe` over the ranks the tokens are cut over:

      * dispatch: all_to_all over `ep_moe` of a rank's own routed rows;
      * every rank of an expert's tensor-parallel group needs every row
        routed to that expert: `tp_moe_expert` ranks' rows, one all_gather
        over `tp_moe_expert`;
      * the partial sums go back the same way: reduce_scatter over
        `tp_moe_expert`, then the inverse all_to_all of a rank's own rows.

    Each rank's buffer is whole rows, so a gathered buffer is
    `tp_moe_expert` whole-row buffers."""
    config = json.loads(SNAP.read_text())["config"]
    own = _own_rows(parallel, phase, rho) * config["hidden_size"] * 2
    tp, ep = parallel.get("tp_moe_expert", 1), parallel.get("ep_moe", 1)
    seam = []
    if ep > 1:
        seam.append(("all_to_all", "ep_moe", ep, own))
    if tp > 1:
        seam += [("all_gather", "tp_moe_expert", tp, tp * own),
                 ("reduce_scatter", "tp_moe_expert", tp, tp * own)]
    if ep > 1:
        seam.append(("all_to_all", "ep_moe", ep, own))
    return seam


def _zone_events(phase="prefill", rho=1, **parallel):
    """Every event attributed to a node of the MoE layer's routed zone, on
    any world (`Program.dispatch_of` of its attribution node) — so an extra
    collective on `cp` or a cross-replica world shows up too."""
    analysis = _step(phase=phase, ep_load_factor=rho, **parallel).analysis
    program = analysis.program
    return [(event.op, event.world, event.group_size, event.payload_bytes)
            for event in collect_events(analysis)
            if program.dispatch_of(program.by_uid(event.attribution_uid))
            is not None]


@pytest.mark.parametrize("parallel", [
    # The token cut (dp_attention over all 8 ranks) spans the 2-rank route
    # groups only partly: the exchange absorbs its ep part, the tp_moe_expert
    # part survives and is gathered. First: red when the dispatch absorbs the
    # whole cut (one all_to_all of 4x the rank's rows, no gather).
    dict(tp_moe_expert=4, ep_moe=2, dp_attention=8),
    dict(tp_moe_expert=2, ep_moe=4, dp_attention=8),
    dict(tp_moe_expert=4, ep_moe=2, cp=8),
])
def test_routed_zone_collectives_match_config_arithmetic(parallel):
    assert _zone_events(**parallel) == _expected_seam(parallel)


def test_a_route_invariant_cut_beside_an_absorbed_one_gathers_over_expert_tp():
    """`cp` alone is invariant inside the route groups, `dp_attention` is
    not; the rows a rank holds after the exchange are its route group's
    tokens only, so the gather runs over `tp_moe_expert`. It used to keep
    the `cp` cut whole and gather over `cp` a buffer that did not grow —
    an invalid collective, raised at materialization."""
    parallel = dict(tp_moe_expert=4, ep_moe=2, cp=2, dp_attention=4)
    assert _zone_events(**parallel) == _expected_seam(parallel)


def test_without_expert_parallelism_the_partial_sums_reduce_scatter_back():
    """No exchange: the rows each rank dispatched are gathered over
    `tp_moe_expert` and the partial sums go back to them by reduce_scatter
    (it used to be an all_reduce of the gathered rows, twice the bytes)."""
    parallel = dict(tp_moe_expert=4, dp_attention=4)
    assert _zone_events(**parallel) == _expected_seam(parallel)


def test_a_product_token_cut_reduce_scatters_back_onto_the_product():
    """The same with the token cut a product of two worlds (`cp` x
    `dp_attention` over `tp_moe_expert`'s ranks): the return is one
    reduce_scatter, recognised by the product's owner cells — it used to be
    an all_reduce plus a local slice."""
    parallel = dict(tp_moe_expert=4, cp=2, dp_attention=2)
    assert _zone_events(**parallel) == _expected_seam(parallel)


@pytest.mark.parametrize("phase, rho", [("prefill", 2.0), ("decode", 1)])
def test_routed_zone_skew_and_decode_match_config_arithmetic(phase, rho):
    parallel = dict(tp_moe_expert=4, ep_moe=2, dp_attention=8)
    assert _zone_events(phase=phase, rho=rho, **parallel) \
        == _expected_seam(parallel, phase, rho)


def test_attention_tensor_parallelism_gathers_over_the_ranks_with_other_rows():
    """`tp_attention` = 2 under `tp_moe_expert` = 4: the two attention-TP
    ranks of a pair hold the same tokens, so after the exchange a rank's
    expert-TP group holds two distinct row shards, each on two ranks. The
    gather runs over the ranks holding the OTHER shard (the cross-replica
    group of size 2) and doubles the buffer, read off the routed rows like
    its neighbours (the busiest rank's, in whole rows).

    The return pins today's form: an all_reduce over all of `tp_moe_expert`
    and a local slice. The cheaper form — reduce_scatter over the size-2
    group, then all_reduce over the pair — is two chained collectives on one
    edge, which the lowering does not yet build."""
    parallel = dict(tp_attention=2, tp_dense_ffn=2, dp_attention=4,
                    tp_moe_expert=4, ep_moe=2)
    config = json.loads(SNAP.read_text())["config"]
    # At a measured skew: the gather's buffer is the busiest rank's whole
    # rows, as on every other collective of the zone.
    own = _own_rows(parallel, "prefill", 2.0) * config["hidden_size"] * 2
    assert _zone_events(rho=2.0, **parallel) == [
        ("all_to_all", "ep_moe", 2, own),
        ("all_gather", "tp_moe_expert^cross", 2, 2 * own),
        ("all_reduce", "tp_moe_expert", 4, 2 * own),
        ("all_to_all", "ep_moe", 2, own),
    ]


def test_every_routed_launch_runs_its_own_input_on_the_serving_path():
    """What the kernel geometry is read off (`ServingLayers.pre_routed`,
    one `RoutedLaunch` per routed leaf) against what the serving layers say
    each leaf is handed per rank (`pre_shapes`) — the dispatch's buffer for
    the dispatch, the routed operand for every other leaf — including the
    FP8 activation cast grafted before `down_proj` after the walk, unfused
    (no framework). Past the gather every leaf runs `tp_moe_expert` ranks'
    rows; the cast is one of them."""
    import sympy as sp

    from backend.arch.eval.dim import value_of
    from backend.arch.eval.precision import PrecisionProfile
    from backend.inference.pipeline import serving_layers

    parallel = dict(tp_moe_expert=4, ep_moe=2, dp_attention=8)
    step = _step(precision=PrecisionProfile.from_request(
        {"preset": "w_fp8_kv_fp8"}), **parallel)
    layers = serving_layers(step, framework_binding=None, stage=None,
                            timed=step.walk.program.included_stack_names(
                                include_aux=False))
    program = layers.pre_program
    casts = []
    for uid, launch in layers.pre_routed.values.items():
        node = program.by_uid(uid)
        shape = layers.pre_shapes.values[uid]
        buffer = shape.out if node.op == "permute" else shape.ins[0]
        held = sp.Integer(1)
        for extent in buffer[:-1]:
            held = held * value_of(extent)
        assert sp.simplify(value_of(launch.routed_rows) - held) == 0, (
            node.display_path, launch.routed_rows, held)
        if node.op == "quantise_activation":
            casts.append(node)
    assert casts, "no grafted activation cast is on the routed buffer"
    config = json.loads(SNAP.read_text())["config"]
    own = 2 * 128 * config["num_experts_per_tok"]
    point = {"B": 2, "s_input": 128, "s_prefix": 0, "rho": 1}
    for node in casts:
        rows = sp.sympify(
            value_of(layers.pre_routed.values[node.uid].routed_rows))
        at = rows.subs({symbol: point[symbol.name]
                        for symbol in rows.free_symbols})
        assert at == parallel["tp_moe_expert"] * own


def test_a_whole_axis_cut_into_a_product_is_free_and_a_sum_scatters_onto_it():
    """`lower`: Replicated -> a product cut is a local slice; a pending sum
    over `tp_moe_expert` finalized onto a product cut that owns exactly
    `tp_moe_expert`'s coordinate fibers is ONE reduce_scatter (recognised by
    membership, not by spelling); onto a cut owning coarser cells it is
    not."""
    from backend.arch.layout.edge import EdgeState
    from backend.arch.layout.layout import Replicated, product, sharded
    from backend.arch.layout.lower import axis_collective, lower_diff
    from backend.arch.layout.value import SUM, Partial

    rule = _step(tp_moe_expert=4, cp=2, dp_attention=2).analysis.rule
    families = rule.families
    whole = Replicated(scope=rule.universe)
    fine = product((sharded(families["dp_attention"]),
                    sharded(families["cp"])))
    coarse = sharded(families["cp"])
    pending = Partial(SUM, families["tp_moe_expert"])

    assert axis_collective(rule, whole, fine, EdgeState((whole, whole))) \
        is None
    assert lower_diff(rule, EdgeState((whole, whole), pending),
                      EdgeState((fine, whole))) \
        == (("tp_moe_expert", "reduce_scatter"),)
    assert lower_diff(rule, EdgeState((whole, whole), pending),
                      EdgeState((coarse, whole))) \
        == (("tp_moe_expert", "all_reduce"),)
