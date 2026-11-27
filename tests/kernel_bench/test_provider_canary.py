"""A run attempts each launch path's smallest cell first, and stops a path
whose first attempt measured nothing.

When an installed release moves or drops what a provider calls, every cell
of that provider fails the same way, one at a time, each after its inputs
and oracle were built (`cannot import name 'fused_grouped_topk'` ~10,000
times a pass). The canary is the path's smallest pending cell, attempted as
an ordinary attempt when the path first comes up: one row records it, the
path's other cells are not attempted, and the run says so. A passing path
runs as before, so a failure particular to one cell is still recorded.
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
from tests.kernel_bench.test_plan_candidates import judged_on
#: The one-env manifest the run-all tests fake (`dependency_manifest`).
_FAKE = {"fake": {"source_build": False}}
from kernel_bench.cli import EXIT_PATHS_DISABLED
from kernel_bench.driver import run_jobs
from kernel_bench.env import Env
from kernel_bench.providers import TORCH_FLOAT, BenchCase, KernelSignature, Provider

ENV = Env("2.11", None, None, None, None, libs={"torch": "2.11"})
CANONICAL = {"sort_keys": True, "separators": (",", ":")}

#: Plan order on purpose: the smallest cell is neither first nor last.
WIDTHS = (64, 128, 32, 256)


def _job(d_out: int, phase: str = "decode") -> dict:
    dims = {"B": 1, "s_query": 4, "d_reduce": 64, "d_out": d_out}
    dtype_env = {"weight": "fp32", "activation": "fp32", "kv": "fp32",
                 "output": "fp32", "metadata": "fp32"}
    return {
        "signature": "gemm", "op": "linear", "dims": dims,
        "dtype_env": dtype_env, "phase": phase, "tp": 1,
        "shape_in": [1, 4, 64], "shape_out": [1, 4, d_out],
        "flops_per_rank": 1.0e6, "access_bytes_per_rank": 1.0e4,
        "model": "m", "spec_hash": "h", "node_spec_id": f"block|linear{d_out}",
        "parallel": "tp=1", "count_per_step": 1,
        "physical_key": ["gemm", json.dumps(dims, **CANONICAL),
                         json.dumps(dtype_env, **CANONICAL), phase],
    }


def _jobs(phase: str = "decode") -> list:
    return [_job(w, phase) for w in WIDTHS]


@pytest.fixture
def only():
    """Run with exactly the given providers registered."""
    P.discover()
    saved = list(P._PROVIDERS)

    def use(*provs):
        P._PROVIDERS[:] = list(provs)
    yield use
    P._PROVIDERS[:] = saved


class Counting:
    """A build that counts its calls and fails where `fails(inv)` says."""

    def __init__(self, fails=lambda inv: None):
        self.fails = fails
        self.built = []

    def __call__(self, inv, env, inputs):
        self.built.append((inv.phase, inv.dims["d_out"]))
        error = self.fails(inv)
        if error is not None:
            raise error
        x, w = inputs["x"], inputs["w"]
        return BenchCase(run=lambda: x @ w.T, to_canonical=lambda o: o)


def _provider(build, impl="t_canary") -> Provider:
    return Provider(library="torch", signature="gemm", kernel_impl=impl,
                    import_name="torch", build=build,
                    serves=KernelSignature(signature="gemm", weight=TORCH_FLOAT,
                                           activation=TORCH_FLOAT))


def _run(jobs, **kw):
    written = []
    res = run_jobs(jobs, env=ENV, date="2026-10-25", contributor="@t",
                   on_row=written.append, **kw)
    return res, written


def _outcomes(written):
    return sorted((json.loads(r["shape_json"])["d_out"], bm.outcome_of(r))
                  for r in written)


def test_a_provider_whose_import_moved_is_tried_once_and_stopped(only):
    build = Counting(lambda inv: ImportError(
        "cannot import name 'fused_grouped_topk' from 'vllm'"))
    only(_provider(build))
    disabled = []
    res, written = _run(_jobs(), cross_check=True,
                        on_disable=lambda path, why: disabled.append((path, why)))
    # One attempt, on the smallest cell, and one row recording it.
    assert build.built == [("decode", 32)]
    assert _outcomes(written) == [(32, "error")]
    bm.validate_row(written[0])
    assert "fused_grouped_topk" in written[0]["outcome_detail"]
    # Said once, with the reason, and the cells held back are counted.
    assert len(res.disabled) == 1 and disabled == list(res.disabled.items())
    (path, why), = res.disabled.items()
    assert path.startswith("torch:t_canary gemm decode")
    assert "error" in why and "fused_grouped_topk" in why
    assert res.not_attempted == 3


def test_a_passing_provider_is_measured_once_per_cell(only):
    build = Counting()
    only(_provider(build))
    res, written = _run(_jobs(), cross_check=True)
    assert _outcomes(written) == [(w, "measured") for w in sorted(WIDTHS)]
    assert all(r["cross_check"] == "passed" for r in written)
    # The canary IS the smallest cell's attempt: nothing is built twice.
    assert sorted(build.built) == sorted(("decode", w) for w in WIDTHS)
    assert build.built[0] == ("decode", 32)
    assert res.disabled == {} and res.not_attempted == 0


def test_a_cell_that_fails_after_a_passing_canary_is_still_recorded(only):
    build = Counting(lambda inv: RuntimeError("expected scalar type BFloat16 "
                                              "but found Float")
                     if inv.dims["d_out"] == 128 else None)
    only(_provider(build))
    res, written = _run(_jobs(), cross_check=True)
    assert _outcomes(written) == [(32, "measured"), (64, "measured"),
                                  (128, "error"), (256, "measured")]
    assert res.disabled == {} and res.not_attempted == 0


def test_a_broken_path_does_not_stop_another_path_of_the_provider(only):
    """A path is one implementation at one phase and precision: a prefill
    build that broke says nothing about the decode build."""
    build = Counting(lambda inv: TypeError("prefill signature changed")
                     if inv.phase == "prefill" else None)
    only(_provider(build))
    res, written = _run(_jobs("prefill") + _jobs("decode"), cross_check=False)
    rows = sorted((r["phase"], bm.outcome_of(r)) for r in written)
    assert rows == [("decode", "measured")] * 4 + [("prefill", "error")]
    assert len(res.disabled) == 1 and res.not_attempted == 3


def test_a_canary_ahead_of_its_job_is_named_on_the_gpu_without_moving_the_resume_point(only):
    """The sidecar resumes at the job in hand and names the canary's job as
    the one running, so a death there blames the canary's cell and loses
    none of the jobs between them."""
    only(_provider(Counting()))
    jobs = _jobs()
    starts = []
    run_jobs(jobs, env=ENV, date="2026-10-25", contributor="@t",
             cross_check=False,
             on_job_start=lambda i, job: starts.append((i, job["dims"]["d_out"])))
    assert starts == [(0, 64), (0, 32), (0, 64), (1, 128), (2, 32), (3, 256)]


def test_run_all_drops_the_job_named_on_the_gpu_and_lists_disabled_paths(
        tmp_path, monkeypatch, capsys):
    """A death no row could take the blame for drops the job the sidecar
    names, not the one at its resume index; and the paths the child's
    canaries disabled reach `run-all`'s summary."""
    from kernel_bench import supervisor
    from kernel_bench.cli import main
    from kernel_bench.emit import IncrementalCsvWriter
    from kernel_bench.plan import write_jobs_jsonl

    jobs = _jobs()
    jobs = [{**job, "candidates": {"t": "fake"}, "judged_on": judged_on(manifest=_FAKE, envs_dir=tmp_path)} for job in jobs]
    jobs_path = tmp_path / "jobs.jsonl"
    write_jobs_jsonl(jobs, str(jobs_path))
    monkeypatch.setattr(P, "dependency_manifest", lambda: _FAKE)
    given = []

    def fake_run(command, **kwargs):
        if "--progress-file" not in command:
            return supervisor.SupervisedResult(0, None, 0.1)
        given.append([json.loads(line)["dims"]["d_out"] for line in Path(
            command[command.index("--jobs") + 1]).read_text().splitlines()])
        progress = command[command.index("--progress-file") + 1]
        if len(given) == 1:
            with IncrementalCsvWriter(command[command.index("-o") + 1]):
                pass
            supervisor.write_progress(
                progress, 0, jobs[2],
                disabled={"torch:t gemm decode [..]": "canary: error — boom"})
            return supervisor.SupervisedResult(1, None, 1.0)
        return supervisor.SupervisedResult(0, None, 0.1)

    monkeypatch.setattr(supervisor, "run_command", fake_run)
    code = main(["run-all", "--jobs", str(jobs_path), "--envs-dir", str(tmp_path),
                 "-o", str(tmp_path / "results"), "--stall-timeout", "1s"])
    assert code == 1                    # a quarantine fails the env, as before
    assert given[1] == [64, 128, 256]
    assert "fake: disabled torch:t gemm decode [..] — canary: error — boom" in (
        capsys.readouterr().out)
    with open(tmp_path / "results" / "fake.csv", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == []


@pytest.mark.parametrize("outcome", ["mismatch", "timeout"])
def test_a_canary_that_measured_nothing_disables_its_path(only, monkeypatch, outcome):
    """Any row that is not a measurement is the canary failing — not only
    an `error`. Except `oom` (below)."""
    import kernel_bench.driver as drv

    def wrong(inv, env, inputs):
        x, w = inputs["x"], inputs["w"]
        return BenchCase(run=lambda: 1.5 * (x @ w.T), to_canonical=lambda o: o)

    def too_slow(call, **kw):
        raise drv.TooSlowToMeasure("one launch takes 99.0s")

    build = {"mismatch": wrong, "timeout": Counting()}[outcome]
    if outcome == "timeout":
        monkeypatch.setattr(drv, "time_callable", too_slow)
    only(_provider(build))
    res, written = _run(_jobs(), cross_check=True)
    assert _outcomes(written) == [(32, outcome)]
    assert len(res.disabled) == 1 and res.not_attempted == 3


def test_a_canary_that_ran_out_of_memory_leaves_its_path_running(
        only, monkeypatch):
    """An `oom` is the cell's size, not the path failing: the larger cells
    are attempted and each records its own, in the same run. Disabling on
    it left them pending, one more recorded per pass, and the model never
    done (the H100's 2.6M-row grouped-GEMM prefill cells, 2026-10-26)."""
    import kernel_bench.driver as drv

    def no_room(inv, env, inputs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(drv, "_is_out_of_memory", lambda e, env=None: True)
    monkeypatch.setattr(drv, "_fits_after_release", lambda env, error: False)
    only(_provider(no_room))
    res, written = _run(_jobs(), cross_check=True)
    assert _outcomes(written) == [(w, "oom") for w in sorted(WIDTHS)]
    assert res.disabled == {} and res.not_attempted == 0


def test_a_canary_with_no_row_of_its_own_leaves_the_path_running(only, monkeypatch):
    """No row, no verdict: the canary cell's inputs failing to build is the
    harness's failure, not the provider's."""
    from kernel_bench import refs
    real = refs.get_ref

    class Spec:
        def __init__(self, spec):
            self._spec = spec

        def __getattr__(self, name):
            return getattr(self._spec, name)

        def make_inputs(self, inv, device, generator):
            if inv.dims["d_out"] == 32:
                raise ValueError("no builder for this cell")
            return self._spec.make_inputs(inv, device, generator)

    monkeypatch.setattr(refs, "get_ref", lambda sig: Spec(real(sig)))
    only(_provider(Counting()))
    res, written = _run(_jobs(), cross_check=False)
    assert _outcomes(written) == [(64, "measured"), (128, "measured"),
                                  (256, "measured")]
    assert res.disabled == {} and res.not_attempted == 0


def test_a_run_that_disabled_a_path_exits_not_done(only, monkeypatch, tmp_path):
    """`run` reached the end, but cells were held back: it exits
    `EXIT_PATHS_DISABLED`, which `run-all` passes on and the queue does not
    mark `.done`."""
    import kernel_bench.env as env_mod
    from kernel_bench.cli import main
    from kernel_bench.plan import write_jobs_jsonl

    monkeypatch.setattr(env_mod, "detect_env", lambda: ENV)
    only(_provider(Counting(lambda inv: ImportError("cannot import name 'x'"))))
    jobs_path = tmp_path / "jobs.jsonl"
    write_jobs_jsonl(_jobs(), str(jobs_path))
    code = main(["run", "--jobs", str(jobs_path), "-o", str(tmp_path / "out.csv"),
                 "--no-gpu-lock"])
    assert code == EXIT_PATHS_DISABLED


def test_the_queue_does_not_mark_a_model_with_a_disabled_path_done(tmp_path):
    from kernel_bench.queue import Workspace, run_once
    space = Workspace(tmp_path).prepare()
    space.plan_of("org/m").write_text(json.dumps(_job(32)) + "\n")
    assert run_once(space, "org/m", spawn=lambda cmd: EXIT_PATHS_DISABLED) == (
        EXIT_PATHS_DISABLED)
    assert not space.done_marker("org/m").exists()


def test_run_all_passes_a_disabled_path_on_as_not_done(tmp_path, monkeypatch, capsys):
    """A child that reached the end with a path disabled is not a crash (no
    quarantine, no restart) and not done: `run-all` exits with it."""
    from kernel_bench import supervisor
    from kernel_bench.cli import main
    from kernel_bench.emit import IncrementalCsvWriter
    from kernel_bench.plan import write_jobs_jsonl

    jobs = _jobs()
    jobs = [{**job, "candidates": {"t": "fake"}, "judged_on": judged_on(manifest=_FAKE, envs_dir=tmp_path)} for job in jobs]
    jobs_path = tmp_path / "jobs.jsonl"
    write_jobs_jsonl(jobs, str(jobs_path))
    monkeypatch.setattr(P, "dependency_manifest", lambda: _FAKE)
    calls = []

    def fake_run(command, **kwargs):
        if "--progress-file" not in command:
            return supervisor.SupervisedResult(0, None, 0.1)
        calls.append(command)
        with IncrementalCsvWriter(command[command.index("-o") + 1]):
            pass
        supervisor.write_progress(
            command[command.index("--progress-file") + 1], 3, jobs[3],
            disabled={"torch:t gemm decode [..]": "canary: error — boom"})
        return supervisor.SupervisedResult(EXIT_PATHS_DISABLED, None, 1.0)

    monkeypatch.setattr(supervisor, "run_command", fake_run)
    code = main(["run-all", "--jobs", str(jobs_path), "--envs-dir", str(tmp_path),
                 "-o", str(tmp_path / "results"), "--stall-timeout", "1s"])
    out = capsys.readouterr().out
    assert code == EXIT_PATHS_DISABLED and len(calls) == 1
    assert "ran 1 ok: ['fake']" in out and "not done: 1 launch path(s)" in out
