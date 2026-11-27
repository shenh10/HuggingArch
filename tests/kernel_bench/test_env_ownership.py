"""One provider, one env: a run measures each provider in the env built for
its library, not in every env that can import it.

Every env carries torch and the sglang / vllm envs carry flashinfer, and the
implementation identity includes `framework_version` — so when every env
measured whatever it could import, each env's torch read as a new
implementation and one cell's `torch_add` was timed three times in three
minutes (2.8.0+cu128, 2.13.0+cu130, 2.14.1+cu130). An env owns the providers
registered under its library (`Provider.library`), the set its
`envs/<lib>.txt` lists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("torch")

from kernel_bench import providers as P
from kernel_bench.env import Env
from kernel_bench.invocation import Invocation
from kernel_bench.providers import TORCH_FLOAT, BenchCase, KernelSignature, Provider

from tests.kernel_bench.test_failure_outcomes import _job
from tests.kernel_bench.test_plan_candidates import judged_on


def _gemm(inv, env, inputs):
    x, w = inputs["x"], inputs["w"]
    return BenchCase(run=lambda: x @ w.T, to_canonical=lambda o: o)


def _provider(library: str, impl: str) -> Provider:
    # Every one imports torch: importable in every env, as a torch floor is
    # and as flashinfer is in the sglang env.
    return Provider(library=library, signature="gemm", kernel_impl=impl,
                    import_name="torch", build=_gemm,
                    serves=KernelSignature(signature="gemm", weight=TORCH_FLOAT,
                                           activation=TORCH_FLOAT))


FLOOR = _provider("torch", "t_floor")
FI = _provider("flashinfer", "fi_gemm")
SGL = _provider("sglang", "sgl_gemm")


@pytest.fixture
def registered():
    P.discover()
    saved = list(P._PROVIDERS)
    P._PROVIDERS[:] = [FLOOR, FI, SGL]
    yield
    P._PROVIDERS[:] = saved


def _env(library, torch="2.8"):
    return Env(torch, None, None, None, None, libs={"torch": torch},
               library=library)


def test_an_env_measures_only_the_providers_of_its_library(registered):
    inv = Invocation.from_job(_job())
    assert [p.kernel_impl for p in P.applicable(inv, _env("sglang"))] == ["sgl_gemm"]
    assert [p.kernel_impl for p in P.applicable(inv, _env("torch"))] == ["t_floor"]
    # An env nobody built for one library (a bare `run`) still measures
    # everything it can import.
    assert {p.kernel_impl for p in P.applicable(inv, _env(None))} == {
        "t_floor", "fi_gemm", "sgl_gemm"}


def test_run_all_measures_a_torch_floor_in_one_env_only(
        registered, tmp_path, monkeypatch):
    """Three envs, each with its own torch release, every provider importable
    in all three: each provider is timed exactly once, in its own env."""
    from kernel_bench import driver
    from kernel_bench import env as env_mod
    from kernel_bench import supervisor
    from kernel_bench.cli import _cmd_run, main
    from kernel_bench.plan import write_jobs_jsonl

    info = {"source_build": False, "build_hint": None,
            "no_build_isolation": None}
    manifest = {"torch": info, "flashinfer": info, "sglang": info}
    monkeypatch.setattr(P, "dependency_manifest", lambda: {
        lib: dict(i) for lib, i in manifest.items()})
    jobs_path = tmp_path / "jobs.jsonl"
    # Planned for all three envs, so run-all starts each of them.
    write_jobs_jsonl([{**_job(), "candidates": {
        "fi_gemm": "flashinfer", "sgl_gemm": "sglang", "t_floor": "torch"},
        "judged_on": judged_on(manifest=manifest, envs_dir=tmp_path)}],
        str(jobs_path))
    torch_of = {"flashinfer": "2.13", "sglang": "2.14.1", "torch": "2.8"}
    current, timed = [None], []

    def run_jobs(jobs, *, env, **kwargs):
        # What the driver times: every applicable provider of each job.
        for job in jobs:
            timed.extend((current[0], p.kernel_impl, env.torch_version)
                         for p in P.applicable(Invocation.from_job(job), env))
        return driver.RunResult()

    def child(command, **kwargs):
        if "--progress-file" not in command:            # the env resolve
            return supervisor.SupervisedResult(0, None, 0.1)
        argv = command[command.index("kernel_bench") + 2:]
        current[0] = Path(argv[argv.index("-o") + 1]).stem
        # This process stands in for that env's interpreter: its own torch
        # release, every provider importable.
        version = torch_of[current[0]]
        monkeypatch.setattr(env_mod, "detect_env", lambda: Env(
            version, None, None, None, None, libs={"torch": version}))
        return supervisor.SupervisedResult(
            _cmd_run([*argv, "--no-gpu-lock"]), None, 0.1)

    monkeypatch.setattr(driver, "run_jobs", run_jobs)
    monkeypatch.setattr(supervisor, "run_command", child)
    assert main(["run-all", "--jobs", str(jobs_path), "--envs-dir",
                 str(tmp_path), "-o", str(tmp_path / "results"),
                 "--no-skip-existing", "--stall-timeout", "0"]) == 0
    assert sorted(timed) == [("flashinfer", "fi_gemm", "2.13"),
                             ("sglang", "sgl_gemm", "2.14.1"),
                             ("torch", "t_floor", "2.8")]


def test_each_generated_env_file_tells_run_its_library():
    """The "measure just this env" line every pip env file carries is the
    other way to drive one env; without `--library` it measures every
    importable provider again — the duplication run-all no longer has."""
    files = P.render_env_files()
    runnable = {name: text for name, text in files.items()
                if "-- python -m kernel_bench run " in text}
    assert "torch.txt" in runnable and "sglang.txt" in runnable
    for name, text in runnable.items():
        assert f"--library {name.removesuffix('.txt')} " in text, name


def test_run_refuses_a_library_the_registry_does_not_know(tmp_path, capsys):
    """A typo owns no provider: measuring nothing and exiting 0 would leave a
    header-only file the queue reads as this env done."""
    from kernel_bench.cli import main
    from kernel_bench.plan import write_jobs_jsonl

    jobs_path = tmp_path / "jobs.jsonl"
    write_jobs_jsonl([_job()], str(jobs_path))
    out = tmp_path / "out.csv"
    rc = main(["run", "--jobs", str(jobs_path), "-o", str(out),
               "--library", "flashinfr", "--no-gpu-lock"])
    assert rc != 0
    assert not out.exists()
    assert "flashinfr" in capsys.readouterr().err
