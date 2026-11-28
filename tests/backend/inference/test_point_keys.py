"""`calibration_overlay.point_keys` — the one derivation of what a per-op row
looks up and launches at a point (docs/auto_sweep.md §5.1).

Two kinds of check:
  * its output, on a real walk: one `PointRow` per row, the key and the
    launch read off the row, and a row re-read at another batch keyed as a
    walk at that batch keys it;
  * its callers: the estimate's measured overlay, the shape rescale's readers
    and the Kernel Bench plan ask it, and none of them re-derives a key.
"""
from __future__ import annotations

from fractions import Fraction

import ast
import functools
import inspect
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from tests.snapshots import resolve_snapshot

from backend.inference import calibration_overlay
from backend.arch.eval.analysis import Point
from backend.inference.batch_sweep import rows_at
from backend.inference.comm_cost import Fabric


def _at(walk, rows, batch):
    """`rows` read at `batch` off the walk's laws, for their keys alone:
    on no card and no fabric."""
    return rows_at(
        rows, Point({"B": batch}), cell=walk, peak_flops=None,
        peak_bw=None, fabric=Fabric(None, None, None), breakdown=False)
from backend.inference.benchmark_identity import RowIdentity
from backend.inference.calibration_overlay import (
    NoShapeLaw,
    WorkloadShape,
    build_deployment_context,
    point_keys,
)
from backend.inference.op_signatures import op_signature
from backend.inference.parallel_config import ParallelConfig

MODEL = "meta-llama/Llama-3.2-3B"


def _walk(batch: int):
    from backend.inference.framework_bindings import get_binding
    from backend.inference.pipeline import analyze
    resolved = resolve_snapshot(MODEL)
    walk = analyze(resolved, parallel=ParallelConfig(),
                   framework_binding=get_binding("vllm"), batch_size=batch,
                   avg_input_tokens=512, avg_output_tokens=0, phase="prefill", breakdown=False)
    return resolved, walk


def _points(resolved, walk, rows):
    workload = WorkloadShape.of_walk(walk, seq_len=512)
    deployment = build_deployment_context(
        framework="vllm", framework_binding=None, parallel=ParallelConfig(),
        gpu_sku="H100-SXM", phase="prefill")
    return point_keys(walk, rows, workload, deployment,
                      model_id=MODEL, spec_hash="h")


@pytest.fixture(scope="module")
def walked():
    return _walk(16)


class TestPointRow:
    def test_one_point_per_row_read_off_the_row(self, walked):
        resolved, walk = walked
        rows = walk.per_op_timing
        points = _points(resolved, walk, rows)
        assert len(points) == len(rows)
        identity = RowIdentity(walk)
        assert any(p.key is not None for p in points)
        for row, p in zip(rows, points, strict=True):
            assert p.uid == row.get("uid")
            assert p.aggregation_id == str(row.get("aggregation_id") or "")
            assert (p.op, p.shape_json) == identity.of(row)
            # No shape, no key; a key is the row's physical op at this point.
            assert (p.key is None) == (p.shape_json is None)
            if p.key is not None:
                assert p.key.signature == op_signature(p.op)
                assert p.key.dtype_env_json == p.dtype_env_json
                assert (p.key.gpu, p.key.phase) == ("H100-SXM", "prefill")
            assert p.count == int(row.get("count_per_step", 1))
            assert isinstance(p.launches, int)
            assert (p.launches, p.cadence) == (
                identity.launches(row), identity.cadence(row))
            billed = float(max(p.launches, 1) * p.cadence)
            assert p.flops_per_launch == float(row["flops_per_rank"]) / billed
            assert p.access_bytes_per_launch == (
                float(row["access_bytes_per_rank"]) / billed)
            assert p.t_sol_s == float(row["t_sol_s"])
            # Zero cost is the row's own numerators, not its op's name.
            assert p.zero_cost == (not any(row.get(f) for f in (
                "flops_per_rank", "access_bytes_per_rank", "comm_bytes_per_rank")))
            assert (p.launch_shape_in, p.launch_shape_out or None) == (
                identity.one_event(row)[0],
                identity.one_event(row)[1] or row.get("shape_outs"))
            assert p.leaf == identity.leaf_of(row)

    def test_a_row_read_at_another_batch_is_keyed_as_a_walk_there(self, walked):
        resolved, walk = walked
        direct_resolved, direct = _walk(4)
        at_4 = _points(resolved, walk, _at(walk, walk.per_op_timing, 4))
        there = _points(direct_resolved, direct, direct.per_op_timing)

        def view(points):
            return [(p.uid, p.op, p.signature, p.shape_json, p.key,
                     p.launch_shape_in, p.launch_shape_out, p.count, p.launches,
                     p.cadence)
                    for p in points]
        assert view(at_4) == view(there)

    def test_a_measurable_row_with_no_shape_law_refuses(self, walked):
        resolved, walk = walked
        rows = [dict(r) for r in walk.per_op_timing]
        measurable = next(i for i, p in enumerate(_points(resolved, walk, rows))
                          if p.signature is not None)
        rows[measurable]["shape_law"] = None
        with pytest.raises(NoShapeLaw):
            _points(resolved, walk, _at(walk, rows, 4))


# ---------------------------------------------------------------------------
# The callers ask point_keys, and nothing else derives a key.
# ---------------------------------------------------------------------------


def _calls(tree: ast.AST) -> set:
    """Every name called in `tree`, by its last segment (`a.b.c()` → `c`)."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            out.add(fn.attr if isinstance(fn, ast.Attribute)
                    else getattr(fn, "id", None))
    return out


#: What deriving a row's identity, key or launch is made of. `point_keys`
#: is the one function that may call these. The two dtype/shape cell writers
#: are in here because the previous seam passed a scan without them: the
#: Kernel Bench plan called `point_keys` and then rebuilt the dtype half with
#: `compute_per_row_dtype_env_json`, which is half of the measurement key.
_DERIVATION = {"RowIdentity", "BenchmarkKey", "one_event", "of_event",
               "launches", "cadence", "leaf_of", "params_from_row", "leaf_dtype_env",
               "compute_per_row_dtype_env_json", "compute_per_row_shape_json",
               "keyed_shape"}


def _functions(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


class TestCallSites:
    def test_the_overlay_keys_through_point_keys(self):
        # The estimate's walk (`OverlayWalk.finish`) and a standalone apply
        # (`CalibrationOverlay.apply`) state each row's key the one way.
        def calls(method):
            return _calls(ast.parse(textwrap.dedent(inspect.getsource(method))))
        assert "point_keys" in calls(calibration_overlay.CalibrationOverlay.apply)
        assert "apply" in calls(calibration_overlay.OverlayWalk.finish)

    def test_the_plan_builds_jobs_through_point_keys(self):
        from kernel_bench import plan
        tree = ast.parse(inspect.getsource(plan))
        assert "point_keys" in _calls(tree)
        assert not _calls(tree) & _DERIVATION, sorted(_calls(tree) & _DERIVATION)

    def test_the_driver_reads_the_key_the_plan_carried(self):
        """The harness does not assemble a key either: the job carries it
        (`Invocation.physical_key`) and the run stamps only its card."""
        from kernel_bench.driver import physical_key_for
        calls = _calls(ast.parse(textwrap.dedent(inspect.getsource(physical_key_for))))
        assert "PhysicalKey" in calls
        assert not calls & {"project_shape_for_op", "op_signature", "dumps"}

    def test_every_shape_rescale_reader_keys_through_point_keys(self):
        """A reader of the shape rescale derives no key of its own; a sweep
        point's rows are keyed, when a reader asks, by `PendingEstimate.
        keyed_at` — through `point_keys`."""
        from backend.inference.capacity import PendingEstimate
        readers = []
        for root in ("backend", "kernel_bench"):
            for path in sorted((PROJECT_ROOT / root).rglob("*.py")):
                for fn in _functions(path):
                    if fn.name == "rows_at":
                        continue
                    if "rows_at" in _calls(fn):
                        readers.append(f"{path.relative_to(PROJECT_ROOT)}:{fn.name}")
                        assert not _calls(fn) & _DERIVATION, readers[-1]
        assert readers, "no reader of the shape rescale found"
        keyed_at = ast.parse(textwrap.dedent(inspect.getsource(PendingEstimate.keyed_at)))
        assert "point_keys" in _calls(keyed_at)

    def test_only_point_keys_derives_in_the_overlay(self):
        path = Path(calibration_overlay.__file__)
        for fn in _functions(path):
            if fn.name == "point_keys":
                continue
            if fn.name == "point_row_of":
                # Exactly these vocabulary calls, no more: a new one is a
                # second road growing (and a removed one this test's
                # prompt to shrink the set).
                assert _calls(fn) & _DERIVATION == {
                    "BenchmarkKey", "cadence", "keyed_shape",
                    "params_from_row"}, sorted(_calls(fn) & _DERIVATION)
                continue
            # The structure half's capture is as much the single
            # derivation as the values half that replays it: exactly these
            # vocabulary calls, and nothing else may call any (a second
            # road would grow here first).
            if fn.name == "_capture_row_template":
                assert _calls(fn) & _DERIVATION == {
                    "launches", "leaf_dtype_env",
                    "leaf_of"}, sorted(_calls(fn) & _DERIVATION)
                continue
            # A nested function is walked inside its parent too; judge each
            # by its own body.
            assert not _calls(fn) & _DERIVATION, (fn.name, sorted(_calls(fn) & _DERIVATION))

    def test_no_consumer_derives_a_key_cell_of_its_own(self):
        """Every reader of a row's identity asks `point_keys` and states
        nothing itself — the plan and the contributor template included, which
        are the write side: a row stored under a key the estimate does not ask
        with is a measurement nobody reads. The equality is gated for real in
        `tests/kernel_bench/test_plan_keys_are_the_estimates.py`; this keeps a
        new reader from growing its own derivation."""
        import backend.cli_bench as cli_bench
        from kernel_bench import plan as plan_mod
        for module in (plan_mod, cli_bench):
            path = Path(module.__file__)
            assert "point_keys" in _calls(ast.parse(path.read_text(encoding="utf-8")))
            for fn in _functions(path):
                assert not _calls(fn) & _DERIVATION, (
                    path.name, fn.name, sorted(_calls(fn) & _DERIVATION))


# ---------------------------------------------------------------------------
# Billing today: a row's launches are its count per step and its cadence the
# averaged fraction it runs at (a DSv4 compressor's once-every-m steps, a MoE
# expert's share of the routed rows). How many expert launches a MoE call
# runs is being re-stated by the MoE workstream; these pins move with it.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _decode_points(model: str, batch: int):
    from backend.inference.framework_bindings import get_binding
    from backend.inference.pipeline import analyze
    resolved = resolve_snapshot(model)
    walk = analyze(resolved, parallel=ParallelConfig(),
                   framework_binding=get_binding("vllm"), batch_size=batch,
                   phase="decode",
                   avg_input_tokens=2048, avg_output_tokens=0, breakdown=False)
    workload = WorkloadShape.of_walk(walk, seq_len=1)
    deployment = build_deployment_context(
        framework="vllm", framework_binding=None, parallel=ParallelConfig(),
        gpu_sku="H100-SXM", phase="decode")
    points = point_keys(walk, walk.per_op_timing, workload, deployment,
                        model_id=model, spec_hash="h")
    return walk, points


def _extents(shape):
    if isinstance(shape, (list, tuple)):
        return [e for item in shape for e in _extents(item)]
    return [] if shape is None else [shape]


class TestTodaysBilling:
    """`launches x cadence` is what the step bills one launch's time by: the
    row's launches (`count_per_step`), times the share of one launch a call
    is. A DSv4 compressor's cadence is such a share; a MoE dispatch-zone
    row's fractional axis is not — it is the per-expert mean `M` beside the
    expert axis, one launch runs the rank's whole `[E_local, M, d]` bank, and
    `E_local x M` rows are whole (`RowIdentity.cadence`)."""

    def test_a_compressor_launch_is_a_cadence(self):
        walk, points = _decode_points("deepseek-ai/DeepSeek-V4-Flash", 1)
        cadenced = [(row, p) for row, p in zip(walk.per_op_timing, points, strict=True)
                    if p.cadence != 1 and not row.get("moe_block_member")]
        assert cadenced
        for _row, p in cadenced:
            assert p.launches == p.count and 0 < p.cadence < 1

    @pytest.mark.parametrize("model,batch", [("Qwen/Qwen3-30B-A3B", 1),
                                             ("moonshotai/Kimi-K3", 8)])
    def test_every_row_bills_its_launches_times_its_share(self, model, batch):
        from backend.inference.benchmark_identity import _averaged_fraction
        walk, points = _decode_points(model, batch)
        zone = [(row, p) for row, p in zip(walk.per_op_timing, points, strict=True)
                if row.get("moe_block_member")]
        # The decode mean is below one row per expert, which is the reading
        # that must not become a share of a launch.
        assert any(0 < Fraction(str(x)) < 1 for row, _p in zone
                   for x in _extents(row.get("shape_in_exact")))
        for row, p in zip(walk.per_op_timing, points, strict=True):
            assert p.launches == int(row["count_per_step"])
            in_zone = bool(row.get("moe_block_member"))
            assert p.cadence == (1 if in_zone else _averaged_fraction(row))
            assert p.billed == pytest.approx(float(p.launches * p.cadence))
