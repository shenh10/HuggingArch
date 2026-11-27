"""One dispatch zone's measurement keys agree on the rows one expert runs.

A MoE layer is keyed as three kinds of launch — the dispatch (`permute`),
the expert GEMMs (`grouped_gemm`) and the inverse (`combine`) — and every
one of them names the same pair: the experts the step's rows reach on
this rank (`n_experts`) and the rows one of them runs (`s_expert_rows`).
Every tensor between `permute` and `combine` is `[E_local, M, width]`, so
all three read the pair off that buffer through the ONE key path every
other signature uses: the engine layout each declares, on the side it
carries it (`grouped_gemm` / `moe_combine` read it as operand 0, `permute`
WRITES it, so it declares `engine_outputs`). The two boundaries also name
the token side they move between (`B` × `s_query`): `permute` reads it as
an operand, `combine` writes it as its output.

A mean is not a geometry, so the cell is the whole rows a launch runs: every
axis rounded up, and the routed token total (`n_tokens`, `dims.yaml`
`call_tokens: total`) the product of those padded extents — 51 routed rows
over 64 experts reach 51 of them at one row each, so the launch runs 51
(`eval.routed.RoutedLaunch`, on the row as `routed`).
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from tests.snapshots import STRUCTURAL_SNAPSHOT_IDS, resolve_snapshot

from backend.inference.benchmark_identity import (
    RowIdentity,
    identity_shape_from_shape_in,
    routed_launch_rows,
)
from backend.inference.op_signatures import get_signature
from backend.inference.parallel_config import ParallelConfig

_ROUTED_OPS = ("permute", "grouped_gemm", "combine")


def _routed_row(op, ins, ins_exact, routed_in, out, out_exact, routed_out,
                routed_rows, experts, bytes_per_row=0, weight_rows=None):
    """A per-op row as the walk publishes one on a routed buffer: its shapes
    and its `routed` record (`phases._routed_columns`) — the dispatch's
    boundary on the rank (`RoutedLaunch.wire`), which of the row's tensors
    are the buffer (`routed_in` the read side's indices, `routed_out`
    whether it writes one) and its bytes per routed row."""
    return {"aggregation_id": f"zone.{op}", "moe_block_member": True,
            "shape_in": ins, "shape_in_exact": ins_exact,
            "shape_out": out, "shape_out_exact": out_exact,
            "routed": {"dispatch": 1, "source_rows": 1,
                       "routed_rows": routed_rows, "local_experts": experts,
                       "inputs": list(routed_in), "output": routed_out,
                       "bytes_per_row": bytes_per_row,
                       "weight_rows_per_row": weight_rows}}


def _keyed(op, row):
    """The row's cell, keyed off the launch it runs (`one_event`)."""
    shape_in, shape_out = RowIdentity.one_event(row)
    return json.loads(identity_shape_from_shape_in(
        op, shape_in, shape_out=shape_out,
        routed_rows=routed_launch_rows(row)))


def _zone_keys(stem: str, batch: int, seq: int, phase: str = "prefill",
               **parallel):
    from backend.inference.framework_bindings import get_binding
    from backend.inference.pipeline import analyze

    resolved = resolve_snapshot(stem)
    walk = analyze(resolved, parallel=ParallelConfig(**parallel),
                   framework_binding=get_binding("vllm"), batch_size=batch,
                   avg_input_tokens=seq,
                   avg_output_tokens=16 if phase == "decode" else 0,
                   phase=phase, breakdown=False)
    identity = RowIdentity(walk)
    keys = {}
    for row in walk.per_op_timing:
        if not row.get("moe_block_member"):
            continue
        op, shape_json = identity.of(row)
        if op in _ROUTED_OPS:
            keys.setdefault(op, set()).add(shape_json)
    return keys


@pytest.mark.parametrize("stem,batch,seq,parallel,expected", [
    # 256 tokens x top-8 over 128 experts: 16 rows each.
    ("Qwen__Qwen3-30B-A3B", 1, 256, {}, (128, 16)),
    # 8 x 2048 tokens x top-8 over 128 experts, a rank holding 16 of them.
    ("Qwen__Qwen3-30B-A3B", 8, 2048,
     {"ep_moe": 8, "dp_attention": 8}, (16, 8192)),
    # The packed bank: the copy endpoint sits on the GEMM leaf itself.
    # 512 tokens x top-4 over 32 experts, a rank holding 8.
    ("openai__gpt-oss-20b", 1, 512, {"ep_moe": 4}, (8, 64)),
])
def test_every_launch_of_one_zone_names_one_expert_load(
        stem, batch, seq, parallel, expected):
    keys = _zone_keys(stem, batch, seq, **parallel)
    assert set(keys) == set(_ROUTED_OPS), keys
    for op, shapes in keys.items():
        assert None not in shapes, f"{op}: a routed row went unkeyed"
        pairs = {(d["n_experts"], d["s_expert_rows"])
                 for d in map(json.loads, shapes)}
        assert pairs == {expected}, (
            f"{stem} {parallel}: {op} keys (n_experts, s_expert_rows) as "
            f"{sorted(pairs)}, the zone's expert load is {expected}")
    # All three state the routed total the buffer holds — the axis a Kernel
    # Bench curve of any of them runs along.
    totals = {op: {(d["n_tokens"], d["n_experts"] * d["s_expert_rows"])
                   for d in map(json.loads, shapes)}
              for op, shapes in keys.items()}
    assert len(set(map(frozenset, totals.values()))) == 1, totals
    assert all(t == p for pairs in totals.values() for t, p in pairs), totals
    # And the two boundaries name the same TOKEN side: the dispatch's read
    # is the combine's write.
    tokens = {op: {(d["B"], d["s_query"]) for d in map(json.loads, shapes)}
              for op, shapes in keys.items() if op in ("permute", "combine")}
    assert len(tokens) == 2 and len(set(map(frozenset, tokens.values()))) == 1, \
        tokens


@pytest.mark.parametrize("stem,batch,parallel,rows", [
    # DeepSeek-V3 at B=1: 8 rows routed over 256 experts. The three
    # launches of the zone run those 8 rows, not one per expert in the bank.
    ("deepseek-ai__DeepSeek-V3", 1, {}, 8),
    # Qwen3-30B-A3B at B=4: 32 rows over 128 experts.
    ("Qwen__Qwen3-30B-A3B", 4, {}, 32),
])
def test_a_small_decode_runs_the_rows_it_routes(stem, batch, parallel, rows):
    keys = _zone_keys(stem, batch, 256, "decode", **parallel)
    assert set(keys) == set(_ROUTED_OPS), keys
    for op, shapes in keys.items():
        totals = {json.loads(shape)["n_tokens"] for shape in shapes}
        assert totals == {rows}, (op, totals)


class TestTheRoutedCellIsTheContractPath:
    """The three routed signatures are keyed by their declared engine
    layout, like every other signature — no projection of their own."""

    def test_no_routed_signature_declares_its_own_projection(self):
        for name in ("grouped_gemm", "moe_permute", "moe_combine"):
            sig = get_signature(name)
            assert sig.shape_projection == "contract", (
                f"{name} projects its shape itself; the `[E, M, d]` buffer is "
                "an engine layout like any other")
            assert sig.engine_inputs or sig.engine_outputs, (
                f"{name} declares no engine layout, so nothing says which "
                "axis of the arriving shape is the experts")

    def test_the_dispatch_reads_the_buffer_it_writes(self):
        # `permute`'s operands are token-ordered; the launch is the
        # expert-ordered buffer it produces, which is its `engine_outputs`.
        # Both sides are the cell's: the token rows it reads and the routed
        # rows it writes.
        key = identity_shape_from_shape_in(
            "permute", [[1, 2048, 2048], [1, 2048, 6]],
            shape_out=[64, 192, 2048], routed_rows=64 * 192)
        assert json.loads(key) == {"B": 1, "s_query": 2048, "n_experts": 64,
                                   "s_expert_rows": 192, "d_model": 2048,
                                   "n_tokens": 64 * 192}

    def test_the_expert_gemm_reads_the_buffer_and_the_output_width(self):
        key = identity_shape_from_shape_in(
            "grouped_gemm", [64, 192, 2048], shape_out=[64, 192, 7168],
            routed_rows=64 * 192)
        assert json.loads(key) == {"n_experts": 64, "n_tokens": 64 * 192,
                                   "s_expert_rows": 192, "d_reduce": 2048,
                                   "d_out": 7168}

    def test_the_launch_pads_only_the_experts_that_receive_rows(self):
        # 51 routed rows over 64 experts: a grouped kernel launches no blocks
        # for an expert that receives nothing, so the launch is the 51
        # experts that do, one row each — not 64 experts padded to one row.
        row = _routed_row("grouped_gemm", [[64, 0, 2048]],
                          [[64, "51/64", 2048]], [0], [64, 0, 7168],
                          [64, "51/64", 7168], True, 51, 64)
        assert RowIdentity.launch_rows(row) == (51, Fraction(51))
        assert routed_launch_rows(row) == 51
        assert _keyed("grouped_gemm", row) == {
            "n_experts": 51, "n_tokens": 51, "s_expert_rows": 1,
            "d_reduce": 2048, "d_out": 7168}

    @pytest.mark.parametrize("op,ins,ins_exact,routed_in,out,out_exact,routed_out", [
        # The dispatch's FIRST operand is the token side (2048 rows); the
        # buffer it writes is 64 experts x ceil(51/2) = 26 rows.
        ("permute", [[1, 2048, 2048], [1, 2048, 6]],
         [[1, 2048, 2048], [1, 2048, 6]], [], [64, 25, 2048],
         [64, "51/2", 2048], True),
        # The combine reads the buffer and writes the token side back.
        ("combine", [[64, 25, 2048], [1, 2048, 6]],
         [[64, "51/2", 2048], [1, 2048, 6]], [0], [1, 2048, 2048],
         [1, 2048, 2048], False),
    ])
    def test_a_boundary_total_is_its_routed_buffer_not_its_first_operand(
            self, op, ins, ins_exact, routed_in, out, out_exact, routed_out):
        row = _routed_row(op, ins, ins_exact, routed_in, out, out_exact,
                          routed_out, 64 * 51 // 2, 64)
        assert routed_launch_rows(row) == 64 * 26
        key = _keyed(op, row)
        assert key["n_tokens"] == key["n_experts"] * key["s_expert_rows"] == 64 * 26
        assert (key["B"], key["s_query"]) == (1, 2048)

    def test_a_row_with_no_routed_buffer_states_no_routed_total(self):
        row = {"shape_in": [1, 2048, 2048], "shape_in_exact": [1, 2048, 2048],
               "shape_in_dims": [{"expr": "B", "value": 1},
                                 {"expr": "s_query", "value": 2048},
                                 {"expr": "d_model", "value": 2048}],
               "shape_out": [1, 2048, 2048]}
        assert routed_launch_rows(row) is None

    def test_a_fractional_mean_above_one_row_is_rounded_up(self):
        # 1.14 rows per expert is not a geometry: a launch allocates 2.
        key = identity_shape_from_shape_in(
            "combine", [[8, 1, 2048], [1, 64, 6]],
            shape_in_exact=[["8", "8/7", "2048"], ["1", "64", "6"]],
            shape_out=[1, 64, 2048], routed_rows=16)
        assert json.loads(key) == {"B": 1, "s_query": 64, "n_experts": 8,
                                   "s_expert_rows": 2, "d_model": 2048,
                                   "n_tokens": 16}

    @pytest.mark.parametrize("extents,law,cadence", [
        # 128 experts at a quarter of a row each are 32 whole rows: the bank
        # runs once per call, which is what the key's `launches` bills.
        ([128, 0, 2048], [128, "1/4", 2048], Fraction(1)),
        # One slot at a quarter of a row is a launch every fourth call —
        # DSv4's KV compressor pooling a window once every four tokens.
        ([1, 0, 2048], [1, "1/4", 2048], Fraction(1, 4)),
    ])
    def test_the_cadence_is_the_rows_a_call_presents(self, extents, law,
                                                     cadence):
        assert RowIdentity.cadence(
            {"shape_in": [extents], "shape_in_exact": [law]}) == cadence

    def test_a_zero_extent_is_no_cell(self):
        # Rounding up is what lifts a mean below one row to the one row a
        # launch covers; an extent that is genuinely zero is not a mean, and
        # a tensor nothing runs on is a refusal rather than a cell at one.
        assert identity_shape_from_shape_in(
            "combine", [[8, 16, 0], [1, 64, 6]], shape_out=[1, 64, 0]) is None
        assert identity_shape_from_shape_in(
            "combine", [[8, 0, 2048], [1, 0, 6]], shape_out=[1, 0, 2048]) is None


class TestOnlyTheRoutedBufferIsPadded:
    """The padding rule is the routed buffer's and nothing else's: the
    tensors the walk says are the buffer (`row["routed"]`), at the launch's
    experts and rows (`RoutedLaunch`). A row's dispatch-zone membership
    does not widen anything else: a pool's cadence inside the zone is still
    refused by name, and the numerators at the launched geometry move only
    the buffer's bytes and the touched experts' weights."""

    @staticmethod
    def _row(**extra):
        routed = [{"expr": "n_expert_copies", "value": 32},
                  {"expr": "B*n_experts_active/n_route_targets", "value": "5/4"},
                  {"expr": "d_model", "value": 7168}]
        pooled = [{"expr": "B", "value": 1},
                  {"expr": "s_ctx/compress_ratio", "value": "3/2"},
                  {"expr": "d_model", "value": 7168}]
        row = _routed_row("grouped_gemm", [[32, 1, 7168], [1, 1, 7168]],
                          [[32, "5/4", 7168], [1, "3/2", 7168]], [0],
                          [32, 1, 7168], [32, "5/4", 7168], True, 40, 32,
                          bytes_per_row=4 * 7168, weight_rows="1/32")
        row.update(aggregation_id="block|zone.leaf",
                   shape_in_dims=[routed, pooled], shape_out_dims=routed,
                   **extra)
        return row

    def test_a_non_routed_fractional_axis_in_the_zone_is_refused(self):
        from kernel_bench.plan import _rounded_axes, _starved_axes
        row = self._row()
        assert _starved_axes(row) == []
        assert _rounded_axes(row) == ["s_ctx/compress_ratio=3/2"]

    def test_the_launch_pads_the_routed_buffer_alone(self):
        row = self._row()
        # 40 rows over 32 experts: every expert receives rows, each padded
        # to ceil(40/32) = 2.
        assert RowIdentity.launch_rows(row) == (64, Fraction(40))
        shape_in, shape_out = RowIdentity.one_event(row)
        assert shape_in == [[32, 2, 7168], [1, 2, 7168]]
        assert shape_out == [32, 2, 7168]

    def test_the_geometry_numerator_moves_the_buffer_and_the_weights(self):
        from backend.inference.calibration_overlay import _at_geometry
        # One launch a step; its law: 40 rows at 4*7168 bytes each on the
        # buffer, 1000 bytes off it, and the walk's saturating read of
        # 20 experts' weights of the 32 held (100 bytes each).
        row = self._row(flops_per_rank=40 * 10, count_per_step=1,
                        access_bytes_per_rank=40 * 4 * 7168 + 1000 + 2000,
                        weight_read_bytes_per_rank=2000,
                        weight_bytes_total_per_rank=3200)
        flops, access, delta = _at_geometry(row, billed=1.0)
        assert flops == 64 * 10
        # The buffer at 64 rows, the token side unmoved, and the touched
        # experts' weights (all 32 here) instead of the walk's read.
        assert access == 64 * 4 * 7168 + 1000 + 3200
        # The padding it added, stated rather than left to subtraction.
        assert delta == access - (40 * 4 * 7168 + 1000 + 2000)

    def test_a_fractional_decode_launch_reads_the_experts_it_touches(self):
        from backend.inference.calibration_overlay import _at_geometry
        # 5/2 rows over 32 experts: the walk's mean reads 2.5 experts'
        # weights (100 bytes each); the launch touches 3 at one row each,
        # and the weight law re-read at those 3 rows reads 3 experts.
        row = _routed_row("grouped_gemm", [[32, 0, 64]], [[32, "5/64", 64]],
                          [0], [32, 0, 64], [32, "5/64", 64], True, "5/2",
                          32, bytes_per_row=256, weight_rows="1/32")
        row.update(flops_per_rank=0, count_per_step=1,
                   access_bytes_per_rank=640 + 250,
                   weight_read_bytes_per_rank=250,
                   weight_bytes_total_per_rank=3200)
        _flops, access, delta = _at_geometry(row, billed=1.0)
        assert access == 3 * 256 + 3 * 100
        assert delta == access - (640 + 250)

    def test_a_row_on_no_routed_buffer_keeps_the_walks_numerators(self):
        from backend.inference.calibration_overlay import _at_geometry
        row = {"flops_per_rank": 12, "access_bytes_per_rank": 34,
               "shape_in": [[1, 1, 8]], "shape_in_exact": [[1, "3/2", 8]]}
        # Nothing padded: the walk's numerators and a zero delta.
        assert _at_geometry(row, billed=2.0) == (6.0, 17.0, 0.0)


@pytest.mark.parametrize("experts,mean,rows", [
    # Below a row: the rows actually routed, one per expert reached.
    (256, Fraction(1, 32), 8),
    (64, Fraction(51, 64), 51),
    # From a row up: every expert, padded to the next whole row.
    (256, Fraction(1), 256),
    (256, Fraction(3, 2), 512),
    # A fractional total below one row per expert still runs its rows.
    (8, Fraction(1, 3), 3),
])
def test_an_expert_that_receives_nothing_runs_nothing(experts, mean, rows):
    from backend.arch.eval.routed import RoutedLaunch
    launch = RoutedLaunch(dispatch=1, source_rows=1,
                          routed_rows=experts * mean, local_experts=experts)
    assert launch.launch_rows == rows


class TestTheLaunchHoldsOneBank:
    """The weights and per-expert parameters a routed launch reads are the
    bank over the experts its rows reach (`RoutedLaunch.touched_experts`),
    each found by its axis identity (`shape_weight_dims`, `shape_in_dims`),
    not by where the matrix sits."""

    @staticmethod
    def _row():
        def wire(tensor, expert_axis):
            return [{"expr": "n_expert_copies" if axis == expert_axis
                     else f"a{axis}", "value": value}
                    for axis, value in enumerate(tensor)]
        x = [32, "1/8", 2880]
        bias = [32, 5760]
        row = _routed_row("grouped_gemm", [[32, 0, 2880], bias], [x, bias],
                          [0], [32, 0, 5760], [32, "1/8", 5760], True, 4, 32)
        row.update(shape_in_dims=[wire(x, 0), wire(bias, 0)],
                   shape_out_dims=wire([32, "1/8", 5760], 0),
                   shape_weight=[32, 5760, 2880],
                   shape_weight_dims=wire([32, 5760, 2880], 0))
        return row

    def test_the_weight_and_the_bias_lead_with_the_experts_reached(self):
        # gpt-oss at B=1: 4 rows over 32 experts reach 4 of them.
        row = self._row()
        assert RowIdentity.launch_weight(row) == [4, 5760, 2880]
        shape_in, _out = RowIdentity.one_event(row)
        assert shape_in == [[4, 1, 2880], [4, 5760]]

    def test_a_weight_with_no_expert_axis_is_the_weight_published(self):
        row = self._row()
        row["shape_weight_dims"] = [{"expr": f"a{axis}", "value": v}
                                    for axis, v in enumerate(row["shape_weight"])]
        assert RowIdentity.launch_weight(row) == [32, 5760, 2880]


def test_the_identity_rule_and_an_unsplit_layout_rank_agree():
    """Where nothing splits the experts or the tokens (attention tensor
    parallelism alone), a rank's dispatch boundary is the walk's own: the
    identity `Rule` reading the walk's shapes and `LayoutPropagation`
    reading the same tensors at its rank give one record."""
    from backend.arch.eval.dim import value_of
    from backend.arch.eval.rule import Rule
    from backend.inference.pipeline import analyze

    walk = analyze(resolve_snapshot("Qwen__Qwen3-30B-A3B"),
                   parallel=ParallelConfig(tp_attention=2, tp_dense_ffn=2),
                   batch_size=8, avg_input_tokens=256, avg_output_tokens=16,
                   phase="decode", breakdown=False).analysis
    assert walk.routed

    def numbers(records):
        return {uid: (value_of(r.source_rows), value_of(r.routed_rows),
                      r.local_experts) for uid, r in records.items()}
    assert numbers(walk.routed) == numbers(Rule().routed(walk))


def test_a_body_with_two_dispatches_is_refused():
    """A zone is one dispatch's: a body holding two cannot say which one a
    node's buffer came from, and says so rather than taking the first."""
    from types import SimpleNamespace
    from backend.arch.eval.program import Program

    body = SimpleNamespace(parent=None, children=[])
    first, second = (SimpleNamespace(uid=uid, op="permute", parent=body,
                                     children=[], display_path=f"p{uid}")
                     for uid in (1, 2))
    member = SimpleNamespace(uid=3, op="linear", parent=body, children=[],
                             display_path="bank")
    body.children = [first, second, member]
    graph = SimpleNamespace(_scope_of=lambda node: node.parent.children)
    with pytest.raises(ValueError, match="2 dispatches"):
        Program.dispatch_of(graph, member)


@pytest.mark.regression
@pytest.mark.parametrize("stem", STRUCTURAL_SNAPSHOT_IDS)
def test_structural_moe_models_routed_launch_is_one_cell(stem):
    """Every structural MoE family: a small decode (fewer routed rows than
    experts, the path that launches only the experts the rows reach) keys
    each routed launch off its `RoutedLaunch` without refusal — its routed
    buffers lead with the experts and their rows, its banks' expert axes are
        found — and every launch of the zone names one expert load whose total
        is its experts times its rows."""
    from tests.snapshots import shared_program
    if not shared_program(stem).has_dispatch_zone:
        pytest.skip("no dispatch zone")
    keys = _zone_keys(stem, 1, 256, "decode")
    assert set(keys) == set(_ROUTED_OPS), keys
    for op, shapes in keys.items():
        assert None not in shapes, f"{op}: a routed row went unkeyed"
        for dims in map(json.loads, shapes):
            assert dims["n_tokens"] == dims["n_experts"] * dims["s_expert_rows"], \
                (op, dims)
