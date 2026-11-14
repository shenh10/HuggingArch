"""A failed attempt is a row of the shared table, not an absence.

Before schema 15 a cross-check rejection, a provider exception, a launch too
slow to time, a dead process and a stalled one all left NO row: "nobody has
tried this" and "this was tried and broke" read the same, and every pass paid
to rediscover each failure. Now each is a row whose `outcome` names the
class (`mismatch`, `error`, `timeout`, `crash`) with an `outcome_detail`, and
no latency.

Each class has a test here that goes red if its write regresses to "drop
it", plus the three properties that make writing them safe: the estimate
never indexes one, a failure covers only the implementation version on its
identity, and `hbench report` counts every cell × implementation in one of
nine classes.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.inference import benchmarks as bm

torch = pytest.importorskip("torch")

from kernel_bench import providers as P
from kernel_bench.driver import CudaContextLost, run_jobs
from kernel_bench.env import Env
from kernel_bench.providers import TORCH_FLOAT, BenchCase, KernelSignature, Provider
from tests.kernel_bench.test_plan_candidates import judged_on
#: The one-env manifest the run-all tests fake (`dependency_manifest`).
_FAKE = {"fake": {"source_build": False}}


def _job(**over) -> dict:
    job = {
        "signature": "gemm", "op": "linear",
        "dims": {"B": 1, "s_query": 4, "d_reduce": 64, "d_out": 64},
        "dtype_env": {"weight": "fp32", "activation": "fp32", "kv": "fp32",
                      "output": "fp32", "metadata": "fp32"},
        "phase": "decode", "tp": 1,
        "flops_per_rank": 1.0e6, "access_bytes_per_rank": 1.0e4,
        "model": "m", "spec_hash": "h", "node_spec_id": "block|linear",
        "parallel": "tp=1", "count_per_step": 1,
    }
    job.update(over)
    job.setdefault("physical_key", [
        job["signature"],
        json.dumps(job["dims"], sort_keys=True, separators=(",", ":")),
        json.dumps(job["dtype_env"], sort_keys=True, separators=(",", ":")),
        job["phase"]])
    return job


ENV = Env("2.11", False, None, None, None, libs={"torch": "2.11"})


def _provider(build, impl="t_fail") -> Provider:
    return Provider(library="torch", signature="gemm", kernel_impl=impl,
                    import_name="torch", build=build,
                    serves=KernelSignature(signature="gemm", weight=TORCH_FLOAT,
                                           activation=TORCH_FLOAT))


@pytest.fixture
def only():
    """Run with exactly the given providers registered."""
    P.discover()
    saved = list(P._PROVIDERS)

    def use(*provs):
        P._PROVIDERS[:] = list(provs)
    yield use
    P._PROVIDERS[:] = saved


def _right(inv, env, inputs):
    x, w = inputs["x"], inputs["w"]
    return BenchCase(run=lambda: x @ w.T, to_canonical=lambda o: o)


def _run(**kw):
    written = []
    res = run_jobs([_job()], env=ENV, date="2026-10-24", contributor="@t",
                   on_row=written.append, **kw)
    return res, written


def _only_row(written, outcome):
    rows = [r for r in written if bm.outcome_of(r) == outcome]
    assert len(rows) == 1, [(r["kernel_impl"], r["outcome"]) for r in written]
    row = rows[0]
    bm.validate_row(row)                    # what `run` persists, it gates
    assert not bm.is_measurement(row)
    assert row["latency_us"] == "" and row["p50_us"] == ""
    assert row["outcome_detail"]
    return row


# ---------------------------------------------------------------- the writes

def test_a_cross_check_rejection_is_a_mismatch_row_with_its_verdict(only):
    def wrong(inv, env, inputs):
        x, w = inputs["x"], inputs["w"]
        return BenchCase(run=lambda: 1.5 * (x @ w.T), to_canonical=lambda o: o)
    only(_provider(wrong))
    res, written = _run(cross_check=True)
    row = _only_row(written, "mismatch")
    assert row["kernel_impl"] == "t_fail"
    # The verdict's basis rides on the row, so the rejection can be audited.
    assert float(row["xcheck_typical"]) > bm_band_typical()
    assert row["xcheck_worst"] and row["cross_check_cosine"]
    assert "cross-check FAILED" in row["outcome_detail"]
    assert row["cross_check"] == "unknown"
    assert not [r for r in res.rows if bm.is_measurement(r)]


def bm_band_typical() -> float:
    from kernel_bench.driver import _XCHECK_BANDS
    return _XCHECK_BANDS["fp32"].typical


def test_a_provider_that_raises_is_an_error_row(only):
    def boom(inv, env, inputs):
        raise RuntimeError("kernel launch failed: bad layout")
    only(_provider(boom))
    _res, written = _run(cross_check=False)
    row = _only_row(written, "error")
    assert row["outcome_detail"] == "RuntimeError: kernel launch failed: bad layout"


def test_a_runtime_unsupported_is_an_error_not_a_decline(only):
    """The provider accepted the cell and then could not serve it: its
    declaration missed a case, which is a fact to record, not a TODO."""
    def unsupported(inv, env, inputs):
        raise NotImplementedError("this head size is not compiled")
    only(_provider(unsupported))
    _res, written = _run(cross_check=False)
    assert "NotImplementedError" in _only_row(written, "error")["outcome_detail"]


def test_a_launch_too_slow_to_time_is_a_timeout_row(only, monkeypatch):
    import kernel_bench.driver as drv

    def too_slow(call, **kw):
        raise drv.TooSlowToMeasure("one launch takes 99.0s")
    monkeypatch.setattr(drv, "time_callable", too_slow)
    only(_provider(_right))
    _res, written = _run(cross_check=False)
    assert "99.0s" in _only_row(written, "timeout")["outcome_detail"]


def test_a_lost_cuda_context_stops_the_run_and_names_the_implementation(
        only, monkeypatch):
    """Recording an `error` and carrying on would blame every later cell for
    a context it did not break; the run stops instead, with the provider
    still named in flight for the supervisor to record as a `crash`."""
    import kernel_bench.driver as drv

    def boom(inv, env, inputs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")
    monkeypatch.setattr(drv, "_context_lost", lambda env: True)
    only(_provider(boom))
    seen = []
    with pytest.raises(CudaContextLost):
        run_jobs([_job()], env=ENV, date="2026-10-24", contributor="@t",
                 cross_check=False, on_attempt=seen.append)
    assert seen and seen[-1] and seen[-1][0]["kernel_impl"] == "t_fail"


def test_in_flight_is_cleared_as_soon_as_the_row_lands(only):
    """Otherwise a death in the next job's shared stage — or a kill between
    the row's write and the end of the provider's loop — would be recorded
    against an implementation that already has its row."""
    only(_provider(_right))
    seen, events = [], []

    def attempt(rows):
        seen.append(rows)
        events.append(("attempt", len(rows)))
    run_jobs([_job()], env=ENV, date="2026-10-24", contributor="@t",
             cross_check=False, on_attempt=attempt,
             on_row=lambda r: events.append(("row", bm.outcome_of(r))))
    assert len(seen[0]) == 1 and seen[-1] == []
    assert events[events.index(("row", "measured")) + 1] == ("attempt", 0)


def test_a_launch_that_is_not_the_cells_is_a_mismatch_row(only, monkeypatch):
    """The launch gate refuses a row for the same reason a cross-check does
    — it would price something else than it measured — and the verdict
    depends only on the provider and the cell, so it is recorded."""
    import kernel_bench.driver as drv
    monkeypatch.setattr(drv.launch_trace, "disagreement",
                        lambda trace, **kw: "the widest operand is bf16, the cell says fp32")
    monkeypatch.setattr(drv.launch_trace, "record", _ObservedRecorder)
    only(_provider(_right))
    _res, written = _run(cross_check=True)
    row = _only_row(written, "mismatch")
    assert row["outcome_detail"].startswith("launch gate: the widest operand is bf16")


class _ObservedRecorder:
    """A launch recorder that saw work, so the gate is asked at all."""

    class _Trace:
        work = (1,)
        saw_result = True
        observed = True

        @staticmethod
        def describe() -> str:
            return "aten.mm(bf16[4, 64], bf16[64, 64])"

    def __init__(self):
        self.trace = self._Trace()

    def run(self, fn):
        return fn()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("stage", ["reference", "grid_reference"])
def test_a_context_lost_in_the_shared_stage_names_no_implementation(
        only, monkeypatch, stage):
    """The oracle and a weight grid are shared work: a lost context there is
    the harness's. Swallowed, it surfaced at the next provider's launch and
    that provider was recorded as the crash."""
    import kernel_bench.driver as drv
    from kernel_bench import refs

    real = refs.get_ref

    class Spec:
        def __init__(self, spec):
            self._spec = spec

        def __getattr__(self, name):
            return getattr(self._spec, name)

        calls = 0

        def reference(self, inputs):
            # The shared oracle is the first call; the grid's own is the next.
            Spec.calls += 1
            if stage == "grid_reference" and Spec.calls == 1:
                return self._spec.reference(inputs)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(refs, "get_ref", lambda sig: Spec(real(sig)))
    monkeypatch.setattr(drv, "_context_lost", lambda env: True)
    monkeypatch.setattr(drv, "_reference_will_not_fit", lambda *a: None)
    prov = _provider(_right)
    if stage == "grid_reference":
        monkeypatch.setattr(drv._quant, "effective_weights",
                            lambda inputs, *a, **kw: dict(inputs))
        prov.weight_group = 16
    only(prov)
    seen = []
    with pytest.raises(CudaContextLost):
        run_jobs([_job()], env=ENV, date="2026-10-24", contributor="@t",
                 cross_check=True, on_attempt=seen.append)
    assert all(rows == [] for rows in seen), seen


def test_each_provider_records_under_its_own_identity(only):
    """The provider loop used to read `impl_id` / `version` left over from
    the selection loop — the LAST provider's — so the first one's failure
    was recorded under the second one's name."""
    def boom(inv, env, inputs):
        raise RuntimeError("first fails")
    only(_provider(boom, impl="t_first"), _provider(_right, impl="t_second"))
    _res, written = _run(cross_check=False)
    assert {(r["kernel_impl"], bm.outcome_of(r)) for r in written} == {
        ("t_first", "error"), ("t_second", "measured")}


# ------------------------------------------------------- crash and timeout

def _quarantine_run(tmp_path, monkeypatch, stopped_by, in_flight_impl="t_fail"):
    """`run-all` over one env whose first `run` dies with an implementation
    in flight; returns the output CSV's rows and the second run's jobs."""
    from kernel_bench import supervisor
    from kernel_bench.cli import main
    from kernel_bench.emit import IncrementalCsvWriter, build_refusal_row
    from kernel_bench.invocation import Invocation
    from kernel_bench.plan import write_jobs_jsonl

    jobs = [{**job, "candidates": {"t_fail": "fake"}, "judged_on": judged_on(manifest=_FAKE, envs_dir=tmp_path)} for job in (
        _job(), _job(node_spec_id="block|other",
                     dims={"B": 1, "s_query": 4, "d_reduce": 64, "d_out": 128}))]
    jobs_path = tmp_path / "jobs.jsonl"
    write_jobs_jsonl(jobs, str(jobs_path))
    monkeypatch.setattr(P, "dependency_manifest", lambda: _FAKE)
    template = build_refusal_row(
        Invocation.from_job(jobs[0]), _provider(_right, impl=in_flight_impl), ENV,
        outcome="crash", date="2026-10-24", contributor="@t",
        impl_id=in_flight_impl, framework_version="2.11", detail="in flight")
    calls, given = [], []

    def fake_run(command, **kwargs):
        if "--progress-file" not in command:
            return supervisor.SupervisedResult(0, None, 0.1)
        calls.append(command)
        given.append(Path(command[command.index("--jobs") + 1]).read_text().splitlines())
        progress = command[command.index("--progress-file") + 1]
        output = command[command.index("-o") + 1]
        if len(calls) == 1:
            with IncrementalCsvWriter(output):      # the child opened its file
                pass
            supervisor.write_progress(progress, 0, jobs[0], in_flight=[template])
            return supervisor.SupervisedResult(-9 if stopped_by else 1, stopped_by, 1.0)
        return supervisor.SupervisedResult(0, None, 0.1)

    monkeypatch.setattr(supervisor, "run_command", fake_run)
    main(["run-all", "--jobs", str(jobs_path), "--envs-dir", str(tmp_path),
          "-o", str(tmp_path / "results"), "--stall-timeout", "1s"])
    with open(tmp_path / "results" / "fake.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows, given[1]


def test_a_process_that_dies_records_a_crash_for_the_implementation_in_flight(
        tmp_path, monkeypatch):
    rows, rest = _quarantine_run(tmp_path, monkeypatch, stopped_by=None)
    assert [(r["kernel_impl"], r["outcome"]) for r in rows] == [("t_fail", "crash")]
    bm.validate_row(rows[0])
    assert "exited 1" in rows[0]["outcome_detail"]
    # The job goes back in: its other implementations still get measured,
    # and the crashed one is skipped by the row it now has.
    assert len(rest) == 2


def test_a_stalled_process_records_a_timeout(tmp_path, monkeypatch):
    rows, _rest = _quarantine_run(
        tmp_path, monkeypatch, stopped_by="output file made no progress for 1s")
    assert [r["outcome"] for r in rows] == ["timeout"]
    assert "no progress" in rows[0]["outcome_detail"]


def test_a_death_naming_no_implementation_records_nothing(tmp_path, monkeypatch):
    """A job's shared stage (inputs, the oracle) is the harness's, not any
    kernel's: no row, and the job is dropped as before."""
    from kernel_bench import supervisor
    real = supervisor.write_progress

    def no_in_flight(path, index, job, in_flight=()):
        real(path, index, job)
    monkeypatch.setattr(supervisor, "write_progress", no_in_flight)
    rows, rest = _quarantine_run(tmp_path, monkeypatch, stopped_by=None)
    assert rows == [] and len(rest) == 1


# ----------------------------------------------------------- what is safe

def _failure_row(**over) -> dict:
    row = {c: "" for c in bm.ALL_COLUMNS}
    row.update(
        model="m/x", spec_hash="h", phase="decode", parallel="tp=1",
        framework="torch", framework_version="2.11", gpu="H100-SXM",
        op="linear", node_spec_id="n",
        shape_json=json.dumps({"B": 1, "s_query": 4, "d_reduce": 64, "d_out": 64}),
        dtype_env_json=json.dumps({"activation": "bf16", "output": "bf16"}),
        kernel_impl="k", schema_version=str(bm.KNOWN_SCHEMA_VERSION),
        contributor="@t", date="2026-10-24", cross_check="unknown",
        outcome="error", outcome_detail="RuntimeError: boom")
    row.update(over)
    return row


def test_a_schema_14_row_merges_as_a_request_row():
    """`OPTIONAL_COLUMNS` lets an older file omit `outcome_detail`; the
    request path re-emits rows with every current column, and raised
    `KeyError` on one that did not carry it."""
    from backend.inference.op_calibration import (CalibrationRegistry,
                                                  merge_validated_rows)
    from tests.corpus_row import row as corpus_row
    old = corpus_row()
    old.pop("outcome_detail")
    old["schema_version"] = "14"
    registry = merge_validated_rows(CalibrationRegistry(), [old],
                                    source="test", origin="mine")
    assert len(registry.rows_by_key) == 1


def test_a_kernel_with_no_body_is_declared_disabled_not_left_to_raise():
    """A `build` that always raises `NotImplementedError` would write an
    `error` row on every cell it is offered."""
    from kernel_bench.providers import all_providers
    by_impl = {p.kernel_impl: p for p in all_providers()}
    for impl in ("deepgemm_fp8_fp4_mega_moe",):
        assert by_impl[impl].disabled_reason, impl


def test_implemented_paged_indexer_is_offered_at_its_audited_geometry():
    """The paged indexer has a real body; MegaMoE remains a stub.

    Layout and numerical checks of this body live in test_paged_indexer_build.
    This regression protects the catalog/driver contract from disabling the
    implementation just to keep an obsolete stub fixture green.
    """
    from kernel_bench.invocation import Invocation
    from kernel_bench.providers import all_providers
    from kernel_bench.providers.indexer import _deepgemm_indexer_build
    provider = next(p for p in all_providers()
                    if p.kernel_impl == "deepgemm_fp8_paged_mqa_logits")
    inv = Invocation(signature="indexer_logits", op="indexer_logits",
                     dims={"B": 1, "s_query": 1, "s_kv": 257,
                           "n_indexer_heads": 32, "d_indexer_head": 128,
                           "kv_block_size": 64},
                     dtype_env={"activation": "fp8_e4m3", "kv": "fp8_e4m3",
                                "kv_format": "fp8_group128", "output": "fp32"},
                     phase="decode", tp=8)
    env = Env("2.14.1", True, "NVIDIA B300 SXM6 AC", "B300", 8000,
              sm="sm_103", libs={"deep_gemm": "0.2.8"}, library="deepgemm")
    assert not provider.disabled_reason
    assert provider.build is _deepgemm_indexer_build
    assert list(provider.declines(inv, env)) == []


class TestTheSchema:
    def test_an_attempt_must_say_why(self):
        with pytest.raises(bm.BenchmarkValidationError, match="outcome_detail"):
            bm.validate_row(_failure_row(outcome_detail=""))

    def test_a_measurement_has_nothing_to_explain(self):
        with pytest.raises(bm.BenchmarkValidationError, match="outcome_detail"):
            bm.validate_row(_failure_row(
                outcome="measured", latency_us="5", p50_us="5", p95_us="6",
                samples="60", count_per_step="1", achieved_tflops="1",
                achieved_bw_gbps="0", peak_tflops_at_test="900",
                peak_bw_at_test="3350"))

    def test_only_a_mismatch_carries_a_comparison(self):
        bm.validate_row(_failure_row(outcome="mismatch", outcome_detail="x",
                                     xcheck_typical="0.5", xcheck_worst="2"))
        with pytest.raises(bm.BenchmarkValidationError, match="ran no comparison"):
            bm.validate_row(_failure_row(xcheck_typical="0.5"))

    def test_every_other_non_measurement_must_say_why(self):
        for outcome in ("mismatch", "error", "crash", "timeout"):
            with pytest.raises(bm.BenchmarkValidationError, match="outcome_detail"):
                bm.validate_row(_failure_row(outcome=outcome, outcome_detail=""))

    def test_an_oom_need_not_say_why(self):
        bm.validate_row(_failure_row(outcome="oom", outcome_detail=""))

    def test_the_detail_is_one_short_line(self):
        with pytest.raises(bm.BenchmarkValidationError, match="one line"):
            bm.validate_row(_failure_row(outcome_detail="a\nb"))
        with pytest.raises(bm.BenchmarkValidationError, match="one line"):
            bm.validate_row(_failure_row(outcome_detail="x" * (bm.OUTCOME_DETAIL_MAX + 1)))


def test_the_estimate_never_indexes_a_failure():
    from backend.inference.op_calibration import PhysicalKey, registry_from_lookup
    row = dict(_failure_row(), origin="shared")
    key = PhysicalKey(signature="gemm", shape_json=row["shape_json"],
                      dtype_env_json=row["dtype_env_json"], gpu="H100-SXM",
                      phase="decode")
    for outcome in ("mismatch", "error", "crash", "timeout"):
        row["outcome"] = outcome
        assert dict(registry_from_lookup([key], [[row]]).rows_by_key) == {}


@pytest.mark.parametrize("version, skipped", [("2.10", False), ("2.11", True)])
def test_a_failure_covers_only_its_own_implementation_version(
        only, tmp_path, version, skipped):
    """`framework_version` is identity: a failure recorded at 2.10 says
    nothing about 2.11, so the cell is tried again there — and at the same
    version it counts as tried, so the sweep does not rediscover it."""
    from kernel_bench.emit import (IncrementalCsvWriter, build_refusal_row,
                                   existing_run_keys)
    from kernel_bench.invocation import Invocation

    prov = _provider(_right)
    only(prov)
    row = build_refusal_row(Invocation.from_job(_job()), prov, ENV,
                            outcome="error", date="2026-10-24", contributor="@t",
                            impl_id="t_fail", framework_version=version,
                            detail="RuntimeError: boom")
    path = tmp_path / "old.csv"
    with IncrementalCsvWriter(str(path)) as writer:
        assert writer.write(row) is None
    res = run_jobs([_job()], env=ENV, date="2026-10-24", contributor="@t",
                   cross_check=False, measured_keys=existing_run_keys(str(path)))
    measured = [r for r in res.rows if bm.is_measurement(r)]
    assert (not measured) == skipped, (res.skipped, res.failures)


def test_report_counts_every_cell_and_implementation_in_nine_classes(
        tmp_path, capsys, only):
    from kernel_bench.cli import main, report_statuses
    from kernel_bench.driver import physical_key_for
    from kernel_bench.emit import IncrementalCsvWriter, build_refusal_row
    from kernel_bench.invocation import Invocation
    from kernel_bench.plan import write_jobs_jsonl

    assert report_statuses() == (
        "passed", "unverified", "oom", "mismatch", "error", "crash", "timeout",
        "not_supported", "pending")
    env = Env(None, False, None, "H100-SXM", None)
    a, b = _provider(_right, impl="t_a"), _provider(_right, impl="t_b")
    declined = Provider(library="torch", signature="gemm", kernel_impl="t_no",
                        import_name="torch", build=_right,
                        serves=KernelSignature(signature="gemm", weight=("fp8_e4m3",),
                                               activation=TORCH_FLOAT))
    only(a, b, declined)
    jobs = [_job(), _job(dims={"B": 1, "s_query": 4, "d_reduce": 64, "d_out": 128})]
    write_jobs_jsonl(jobs, str(tmp_path / "jobs.jsonl"))
    inv = Invocation.from_job(jobs[0])
    assert physical_key_for(inv, env)
    rows = [build_refusal_row(inv, a, env, outcome="timeout", date="2026-10-24",
                              contributor="@t", impl_id="t_a",
                              framework_version="2.11", detail="slow")]
    with IncrementalCsvWriter(str(tmp_path / "r.csv")) as writer:
        for r in rows:
            writer.write(r)
    assert main(["report", "--jobs", str(tmp_path / "jobs.jsonl"),
                 "--results", str(tmp_path / "r.csv"), "--gpu", "H100-SXM"]) == 0
    out = capsys.readouterr().out
    assert ("6 cell × implementation: passed 0 | unverified 0 | oom 0 | "
            "mismatch 0 | error 0 | crash 0 | timeout 1 | not_supported 2 | "
            "pending 3") in out, out
