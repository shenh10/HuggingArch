"""The timed call is read off the dispatcher, and compared to the cell.

Three provider bugs measured something other than what their rows priced —
equal-length operands for a launch that expands rows, one token out of a
cell billed for several, fp32 operands on a row billed in bf16 — and each
was found by hand, months later. All three are visible in what the launch
dispatched, so the harness now looks: `kernel_bench.launch_trace` records
the ops of the one call the cross-check already makes, writes them into the
row as its `launch_signature`, and refuses a row whose launch is not the
cell's.

Both historical cases are asserted here as the gate's own behaviour, with
no provider edited and nothing declared.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch")
import torch

from kernel_bench.launch_trace import Operand, Trace, disagreement, record


def _elements(*shapes):
    out = []
    for shape in shapes:
        count = 1
        for extent in shape:
            count *= extent
        out.append(count)
    return out


class TestWhatTheTraceSees:
    def test_one_call_is_one_launch(self):
        a, b = torch.randn(32, 64), torch.randn(64, 16)
        with record() as recorded:
            a @ b
        assert len(recorded.trace.work) == 1
        assert recorded.trace.describe() == (
            "aten.mm(float32[32,64], float32[64,16])")

    def test_a_composite_floor_says_how_many_launches_it_is(self):
        # The torch baselines are deliberately composite, and the row now
        # says so: the fused kernel they bound is ONE launch.
        x = torch.randn(16, 128)
        with record() as recorded:
            torch.nn.functional.silu(x[:, :64]) * x[:, 64:]
        assert recorded.trace.describe().startswith("2 launches:")

    def test_views_and_allocations_are_not_launches(self):
        x = torch.randn(8, 8)
        with record() as recorded:
            x.t().reshape(64).unsqueeze(0)
        assert recorded.trace.work == []
        assert recorded.trace.describe() == ""

    def test_a_tensor_list_is_operands(self):
        # `aten.cat` takes its inputs as one list. A partial-rope floor
        # reads the half of each head it passes through only there, so a
        # recorder that skipped lists saw a third of the head.
        rotated = torch.randn(16, 4, 64, dtype=torch.bfloat16)
        passed = torch.randn(16, 4, 128, dtype=torch.bfloat16)
        with record() as recorded:
            torch.cat([rotated, passed], dim=-1)
        # The joined output is the widest side now that both are recorded.
        assert str(recorded.trace.widest) == "bfloat16[16,4,192]"

    def test_an_allocation_off_a_tensor_is_not_a_launch(self):
        # vLLM's Triton block-scaled GEMM allocates its output with
        # `A.new_empty(...)` and then launches a kernel the dispatcher
        # cannot see; the allocation is not the launch.
        a = torch.randn(1024, 512, dtype=torch.bfloat16)
        with record() as recorded:
            a.new_empty((1024, 2048))
            a.new_zeros((4,))
        assert recorded.trace.work == []

    def test_nothing_observed_is_not_an_empty_launch(self):
        # A kernel behind a pybind entry point dispatches nothing. The row
        # keeps whatever `Launch` said and the gate asks nothing.
        assert disagreement(Trace(), declared_dtypes=["bf16"],
                            declared_elements=[1024]) is None


class TestTheGateCatchesWhatItWasWrittenFor:
    def test_a_launch_at_another_precision_is_refused(self):
        # The MoE combine as it was: fp32 operands on both sides of a cell
        # the walk bills in bf16, so the latency was 2.4x the row's bytes.
        out = torch.zeros(1024, 256, dtype=torch.float32)
        index = torch.randint(0, 1024, (8192,))
        source = torch.randn(8192, 256, dtype=torch.float32)
        with record() as recorded:
            torch.index_add(out, 0, index, source)
        says = disagreement(
            recorded.trace, declared_dtypes=["bf16"],
            declared_elements=_elements((8192, 256), (1024, 256)))
        assert says is not None and "float32" in says

    def test_the_same_launch_at_the_cells_precision_passes(self):
        out = torch.zeros(1024, 256, dtype=torch.bfloat16)
        index = torch.randint(0, 1024, (8192,))
        source = torch.randn(8192, 256, dtype=torch.bfloat16)
        with record() as recorded:
            torch.index_add(out, 0, index, source)
        assert disagreement(
            recorded.trace, declared_dtypes=["bf16"],
            declared_elements=_elements((8192, 256), (1024, 256))) is None

    def test_a_launch_over_part_of_the_cell_is_refused(self):
        # The conv1d decode build as it was: one token handed to the kernel
        # out of a cell billed for five.
        x = torch.randn(8, 1024, 5, dtype=torch.bfloat16)
        with record() as recorded:
            x[:, :, -1].contiguous().mul(2.0)
        says = disagreement(recorded.trace, declared_dtypes=["bf16"],
                            declared_elements=_elements((8, 1024, 5)))
        assert says is not None and "part of the cell" in says

    def test_a_narrow_kernels_fp32_scales_are_not_a_precision_claim(self):
        # Only the widest operand answers for the precision: every fp8
        # kernel carries fp32 scales, int32 offsets and a page table beside
        # its payload.
        trace = Trace(calls=[])
        from kernel_bench.launch_trace import Call
        trace.calls.append(Call(
            "aten._scaled_mm",
            (Operand("float8_e4m3fn", (512, 512)),
             Operand("float8_e4m3fn", (512, 512)),
             Operand("float32", ()), Operand("float32", ()))))
        assert disagreement(trace, declared_dtypes=["fp8_e4m3", "bf16"],
                            declared_elements=[512 * 512]) is None

    def test_fp32_before_a_named_record_is_not_a_precision_claim(self):
        # A record writer may keep its row in fp32 until it lands on the
        # record's grid. The cross-check reads that landing through the
        # record's own rounding step; the gate must not reject the journey.
        row = torch.randn(128, 512, dtype=torch.bfloat16)
        recorded = record()
        recorded.run(lambda: torch.cat(
            (row[:, :448], row[:, 448:].float()), dim=-1))
        assert disagreement(
            recorded.trace, declared_dtypes=["bf16", "fp8_e4m3"],
            declared_elements=_elements((128, 512)),
            records=("fp8_ue8m0_group64_bf16_rope_tail",)) is None

    def test_a_padded_launch_is_not_a_partial_one(self):
        # Rounding the rows up to a tile moves MORE than the cell, which is
        # the direction this never complains about.
        trace = Trace(calls=[])
        from kernel_bench.launch_trace import Call
        trace.calls.append(Call("aten.mm", (
            Operand("bfloat16", (520, 2048)), Operand("bfloat16", (2048, 512)))))
        assert disagreement(trace, declared_dtypes=["bf16"],
                            declared_elements=[512 * 2048]) is None

class TestTheGateReadsTheCellsOwnSides:
    """Four ways the gate refused a launch that was the whole cell, each one
    the census found by the thousand (2026-10-24: ~4.9k H100, ~8.5k 4090
    cells with no row), each next to the partial launch it must still catch.
    """

    def test_a_gemm_whose_output_outgrows_its_inputs_is_the_whole_cell(self):
        # cuBLAS linear at prefill / large-batch decode, embedding, per-head
        # bmm: the walk publishes the output, and the output is not among
        # aten.mm's arguments. Measured before the fix: aten.mm([1024,256],
        # [256,4096]) refused against the [1024,4096] output it produced.
        a, b = torch.randn(128, 16), torch.randn(16, 1024)
        recorded = record()
        recorded.run(lambda: a @ b)
        assert disagreement(
            recorded.trace, declared_dtypes=["fp32"],
            declared_elements=_elements((128, 16), (16, 1024), (128, 1024))
        ) is None

    def test_an_unexpanded_output_is_still_a_part(self):
        # The MoE dispatch as it was: equal-length rows where the cell
        # expands them four-fold. Counting outputs must not let it through.
        x, idx = torch.randn(64, 32), torch.randint(0, 64, (64,))
        recorded = record()
        recorded.run(lambda: torch.index_select(x, 0, idx))
        says = disagreement(recorded.trace, declared_dtypes=["fp32"],
                            declared_elements=_elements((64, 32), (256, 32)))
        assert says is not None and "part of the cell" in says

    def test_a_kernel_aten_never_saw_is_not_judged_by_its_allocation(self):
        # vLLM's Triton block-scaled GEMM: aten sees `A.new_empty(...)` and
        # nothing else, and that template (the [1, K] activation at decode)
        # was judged against the [N, K] weight. The kernel itself is out of
        # the dispatcher's sight, like any Triton or pybind launch.
        from torch.utils._python_dispatch import _disable_current_modes
        a, b = torch.randn(1, 256), torch.randn(256, 4096)

        def launch():
            out = a.new_empty((1, 4096))
            with _disable_current_modes():
                torch.mm(a, b, out=out)          # the launch aten cannot see
            return out

        recorded = record()
        recorded.run(launch)
        assert recorded.trace.saw_result is False
        assert not recorded.trace.observed
        assert disagreement(
            recorded.trace, declared_dtypes=["fp32"],
            declared_elements=_elements((1, 256), (256, 4096), (1, 4096))
        ) is None

    def test_a_visible_partial_launch_is_still_refused_through_run(self):
        x = torch.randn(8, 1024, 5, dtype=torch.bfloat16)
        recorded = record()
        recorded.run(lambda: x[:, :, -1].contiguous().mul(2.0))
        assert recorded.trace.saw_result is True
        says = disagreement(recorded.trace, declared_dtypes=["bf16"],
                            declared_elements=_elements((8, 1024, 5)))
        assert says is not None and "part of the cell" in says

    def test_a_loop_over_every_expert_covers_the_stacked_weight(self):
        # The torch ragged floor: one mm per expert, each over `w[e]`. Every
        # single operand is one expert; together they are the weight.
        E, K, N = 16, 32, 64
        x, w = torch.randn(E, 1, K), torch.randn(E, K, N)
        recorded = record()
        recorded.run(lambda: [x[e] @ w[e] for e in range(E)])
        assert disagreement(
            recorded.trace, declared_dtypes=["fp32"],
            declared_elements=_elements((E, 1, K), (E, K, N), (E, 1, N))
        ) is None

    def test_a_loop_over_some_experts_is_a_part(self):
        E, K, N = 16, 32, 64
        x, w = torch.randn(E, 1, K), torch.randn(E, K, N)
        recorded = record()
        recorded.run(lambda: [x[e] @ w[e] for e in range(E // 4)])
        says = disagreement(
            recorded.trace, declared_dtypes=["fp32"],
            declared_elements=_elements((E, 1, K), (E, K, N), (E, 1, N)))
        assert says is not None and "part of the cell" in says

    def test_a_kernel_striding_over_the_rope_lanes_is_handed_q(self):
        # vLLM's partial rope is handed `q[..., D - rd:]` — the strided view
        # mla.py passes — and rotates it in place; the walk publishes q whole
        # for that reason. Read as its view's extent it was a third of q.
        q = torch.randn(1024, 8, 192, dtype=torch.bfloat16)
        tail = q[..., 192 - 64:]
        recorded = record()
        recorded.run(lambda: tail.mul_(2.0))
        assert disagreement(recorded.trace, declared_dtypes=["bf16"],
                            declared_elements=_elements((1024, 8, 192))) is None


class TestARowViewIsHandedItsRows:
    """A view of the leading lanes of each row (sglang's rope hands its
    kernel `q.view(T, -1, head_size)[..., :rotary_dim]`) covers its rows to
    their pitch; a compact copy of the same lanes is still only those
    lanes."""

    def test_one_row_of_a_wider_head_counts_the_head(self):
        q = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
        with record() as recorded:
            q[..., :64].mul_(2)
        assert recorded.trace.widest.elements == 256

    def test_a_copy_of_the_lanes_is_still_the_lanes(self):
        q = torch.zeros(1, 1, 256, dtype=torch.bfloat16)
        lanes = q[..., :64].clone()
        with record() as recorded:
            lanes.mul_(2)
        assert recorded.trace.widest.elements == 64


class _Op:
    """An op as the dispatch mode sees it: its schema is all `_unreturnable`
    reads."""

    def __init__(self, schema):
        self._schema = torch._C.parse_schema(schema)


class TestAResultTheTraceCannotHandBack:
    """vLLM's FA3 forward returns `Tensor[]` holding two undefined tensors
    (`out, softmax_lse, _, _`). torch cannot convert those `None`s back out
    of a dispatch mode, so every traced call of it raised pybind's "Unable
    to cast Python instance of type <class 'list'>", recorded as the
    kernel's `error` (gpt-oss, 2026-10-26)."""

    def test_an_undefined_tensor_in_a_tensor_list_is_unreturnable(self):
        from kernel_bench.launch_trace import _unreturnable
        t = torch.zeros(2)
        fa3 = _Op("kbtest::fwd(Tensor q) -> Tensor[]")
        assert _unreturnable(fa3, [t, t, None, None])
        assert not _unreturnable(fa3, [t, t])
        assert not _unreturnable(
            _Op("kbtest::fwd(Tensor q) -> Tensor?[]"), [t, None])
        assert _unreturnable(
            _Op("kbtest::fwd(Tensor q) -> (Tensor, Tensor[])"), (t, [None]))

    def test_the_launch_is_still_checked_and_judged(self, monkeypatch):
        # The driver makes the call again untraced and keeps the trace of
        # the launch for the gate, instead of recording an `error`.
        import kernel_bench.driver as driver
        from kernel_bench import launch_trace
        from kernel_bench.env import Env
        from tests.kernel_bench.test_launch_gate_floors import CASES

        def run(self, fn):
            with self:
                fn()
            raise launch_trace.Unobservable("cannot hand the result back")
        monkeypatch.setattr(launch_trace.record, "run", run)
        impl, job = CASES[0]
        original = driver.applicable
        monkeypatch.setattr(driver, "applicable", lambda inv, env: [
            p for p in original(inv, env) if p.kernel_impl == impl])

        # `**_kw`: the driver hands `time_callable` an `align=` for a
        # group's barrier, and this stub raises before any of it matters.
        def stop_before_timing(call, **_kw):
            raise driver.TooSlowToMeasure("gate passed")
        monkeypatch.setattr(driver, "time_callable", stop_before_timing)
        result = driver.run_jobs(
            [job], env=Env("2.11", None, None, None, None,
                           libs={"torch": "2.11"}),
            date="2026-10-26", contributor="test")
        assert len(result.failures) == 1, result.failures
        assert "gate passed" in result.failures[0], result.failures
        assert result.unobserved == []


class TestWhereTheDoorIsBlind:
    """A span with no observable launch is not a span the door guarded.

    `disagreement` answers None both when the trace agrees and when there is
    no trace, which is correct for the gate (nothing observed is not an
    empty launch) and useless for coverage: it cannot say which spans the
    door has ever judged. `tools.launch_blind_spots` answers that from the
    rows, and the subtle part is what it must NOT count.
    """

    def _rows(self, *specs):
        out = []
        for op, framework, date, launch in specs:
            out.append({"op": op, "framework": framework, "date": date,
                        "schema_version": "14", "launch_signature": launch})
        return out

    def _write(self, tmp_path, rows):
        import csv
        path = tmp_path / "results" / "m" / "x.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return [str(path)]

    def test_a_row_measured_before_the_recorder_is_not_evidence(self, tmp_path):
        # The migration relabels an old row to the CURRENT schema and fills
        # the new column empty, so on a migrated corpus every row claims to
        # be current. Judging by schema version would then put the whole
        # corpus in the blind column and the answer would be about the
        # column's age. The date is the only honest filter.
        from tools.launch_blind_spots import tally

        rows = self._rows(
            ("linear", "vllm", "2026-10-01", ""),      # predates the recorder
            ("linear", "vllm", "2026-10-22", "aten.mm(bfloat16[8,8])"),
        )
        seen, too_old = tally(self._write(tmp_path, rows))
        assert too_old == 1
        assert seen[("linear", "vllm")] == [1, 1]

    def test_a_baseline_being_traced_does_not_cover_the_framework(self, tmp_path):
        # torch baselines dispatch eager ops and are always traced, so an op
        # can look guarded while every framework row in it is unjudged —
        # which is exactly `gated_delta_rule`, where both frameworks run a
        # compiled kernel behind a pybind entry point.
        from tools.launch_blind_spots import tally

        rows = self._rows(
            ("gated_delta_rule", "torch", "2026-10-22", "22 launches: …"),
            ("gated_delta_rule", "vllm", "2026-10-22", ""),
            ("gated_delta_rule", "sglang", "2026-10-22", ""),
        )
        seen, _ = tally(self._write(tmp_path, rows))
        assert seen[("gated_delta_rule", "torch")] == [1, 1]
        assert seen[("gated_delta_rule", "vllm")] == [1, 0]
        assert seen[("gated_delta_rule", "sglang")] == [1, 0]

    def test_the_run_records_what_the_door_could_not_judge(self):
        # "The door agreed" and "the door never looked" used to be the same
        # branch; the run now counts the second.
        from kernel_bench.driver import RunResult

        assert RunResult().unobserved == []
