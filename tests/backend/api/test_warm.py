"""`python -m backend.api.warm` fills the existing gallery cache.

Stubs the frontier build — this pins the CLI wiring (exit codes, the GPU
list), not the estimator.
"""
from __future__ import annotations

import os

import pytest

from backend.api import warm
from backend.inference.sweep_grid import GALLERY_GPUS, GALLERY_LANDING_GPU


@pytest.fixture(autouse=True)
def _no_renice(monkeypatch):
    """Keep the pytest process at its own priority; the env the warm writes
    for the pool size is restored at teardown."""
    monkeypatch.setenv("HA_WARM_NICE", "0")
    monkeypatch.setenv("HA_SWEEP_POOL_WORKERS", "0")
    monkeypatch.setenv("HA_WARM_WORKERS", "0")


@pytest.fixture(autouse=True)
def _facts_calls(monkeypatch):
    """Stub the model-facts list; returns how often a run asked for it."""
    calls = []

    async def fake_facts():
        calls.append("facts")
        return {"models": [{"model_id": "m", "error": None,
                            "total_params": 7}], "errors_count": 0}

    monkeypatch.setattr(warm, "build_model_facts", fake_facts)
    return calls


@pytest.fixture(autouse=True)
def _sweep_calls(monkeypatch):
    """Stub the Scaling warm; records one entry per run."""
    calls = []

    async def fake_sweeps(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(warm, "warm_sweeps", fake_sweeps)
    return calls


@pytest.fixture(autouse=True)
def _frontier_calls(monkeypatch):
    """Stub the decode frontier; returns the GPUs a run built it for."""
    gpus = []

    async def fake_frontier(view, model_params):
        gpus.append((view.gpu_sku, dict(model_params)))
        return _OK

    monkeypatch.setattr(warm, "warm_frontier", fake_frontier)
    return gpus


_OK = {"models": [{"model_id": "m", "error": None}], "errors_count": 0}
_FAIL = {"models": [{"model_id": "m", "error": "boom"}], "errors_count": 1}


def test_default_warms_both_surfaces_and_the_gallery_landing_first(
        _frontier_calls, _sweep_calls):
    """With no surface flag, the warm fills the shared SOL surface: every
    Gallery card and the Scaling requests. The Gallery's own series
    (`GALLERY_GPUS`), not the wider sweep one: the Kernel Bench card has no
    Gallery frontier to fill."""
    assert warm.main([]) == 0
    assert _frontier_calls[0][0] == GALLERY_LANDING_GPU
    assert sorted(gpu for gpu, _params in _frontier_calls) == sorted(GALLERY_GPUS)
    assert all(gpu != "RTX-4090" for gpu, _params in _frontier_calls)
    assert len(_sweep_calls) == 1


def test_gpu_flag_replaces_the_sku_list(_frontier_calls):
    assert warm.main(["--gpu", "H100-SXM", "--gpu", "H200"]) == 0
    assert [gpu for gpu, _params in _frontier_calls] == ["H100-SXM", "H200"]


def test_all_gpus_does_not_widen_the_gallery_frontier(_frontier_calls):
    # The frontier serves the gallery's GPUs alone; --all-gpus is the sweeps'.
    assert warm.main(["--all-gpus"]) == 0
    assert sorted(gpu for gpu, _params in _frontier_calls) == sorted(GALLERY_GPUS)
    # Selecting a Gallery scope opts out of Scaling unless --sweeps says it.
    # The all-GPU widening itself is tested below with the sweeps stub.


def test_the_gallery_flag_warms_the_gallery_alone(
        _frontier_calls, _sweep_calls):
    assert warm.main(["--gallery"]) == 0
    assert sorted(gpu for gpu, _params in _frontier_calls) == sorted(
        GALLERY_GPUS)
    assert _sweep_calls == []


def test_all_gpus_sweeps_use_the_spec_table(monkeypatch, _frontier_calls):
    class _S:
        def __init__(self, sku):
            self.sku = sku

    seen = []

    async def fake_sweeps(*, models, gpus):
        seen.append(gpus)
        return 1

    monkeypatch.setattr(
        "backend.inference.gpu_specs.load_gpu_specs",
        lambda: [_S("B200"), _S("H100-SXM")],
    )
    monkeypatch.setattr(warm, "warm_sweeps", fake_sweeps)
    assert warm.main(["--sweeps", "--all-gpus"]) == 0
    assert seen == [["B200", "H100-SXM"]]
    assert _frontier_calls == []


def test_warm_all_failed_is_nonzero(monkeypatch):
    async def fake_frontier(view, model_params):
        return _FAIL

    monkeypatch.setattr(warm, "warm_frontier", fake_frontier)
    assert warm.main([]) == 1


def test_landing_warms_the_homepage_default(_frontier_calls):
    assert warm.main(["--landing"]) == 0
    assert [gpu for gpu, _params in _frontier_calls] == [GALLERY_LANDING_GPU]


def test_landing_and_sweeps_run_both(monkeypatch, _frontier_calls):
    order = []

    async def fake_sweeps(**_kwargs):
        order.append("sweeps")
        return 1

    monkeypatch.setattr(warm, "warm_sweeps", fake_sweeps)
    assert warm.main(["--landing", "--sweeps"]) == 0
    assert [gpu for gpu, _params in _frontier_calls] == [GALLERY_LANDING_GPU]
    assert order == ["sweeps"]


def test_pidfile_holds_the_pid_while_running_and_is_removed(monkeypatch, tmp_path):
    pidfile = tmp_path / "warm.pid"
    during = []

    async def fake_frontier(view, model_params):
        during.append(pidfile.read_text().strip())
        return _OK

    monkeypatch.setattr(warm, "warm_frontier", fake_frontier)
    assert warm.main(["--landing", "--pidfile", str(pidfile)]) == 0
    assert during == [str(os.getpid())]
    assert not pidfile.exists()


def test_workers_flag_sizes_this_process_pool(monkeypatch):
    from backend.measure import pool as _sweep_pool

    sizes = []

    async def fake_frontier(view, model_params):
        sizes.append(_sweep_pool._max_workers())
        return _OK

    monkeypatch.setattr(warm, "warm_frontier", fake_frontier)
    assert warm.main(["--landing", "--workers", "3"]) == 0
    assert sizes == [min(3, os.cpu_count() or 1)]


def test_entry_point_binds_cache_dirs_from_dotenv(tmp_path):
    """`python -m backend.warm_caches` loads `.env` before `backend.api` binds
    the cache directories, so a directory set only in `.env` is the one the
    warm writes — the same one the server (which loads `.env` first) reads."""
    import pathlib
    import subprocess
    import sys

    repo = pathlib.Path(__file__).resolve().parents[3]
    gallery = tmp_path / "from-dotenv-gallery"
    sol = tmp_path / "from-dotenv-sol-pool"
    sweep = tmp_path / "from-dotenv-sweep"
    (tmp_path / ".env").write_text(
        f"HA_GALLERY_CACHE_DIR={gallery}\n"
        f"HA_SOL_POOL_DIR={sol}\n"
        f"HA_SWEEP_CACHE_DIR={sweep}\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("HA_GALLERY_CACHE_DIR", "HA_SOL_POOL_DIR",
                       "HA_SWEEP_CACHE_DIR",
                        "PYTEST_CURRENT_TEST")}
    env["PYTHONPATH"] = str(repo)
    out = subprocess.run(
        [sys.executable, "-m", "backend.warm_caches", "--show-config"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
        check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert f"gallery cache dir: {gallery}" in out.stdout
    assert f"SOL pool dir:      {sol}" in out.stdout
    assert f"sweep cache dir:   {sweep}" in out.stdout


def test_a_gallery_warm_builds_the_decode_frontier_once_per_gpu(_frontier_calls):
    """The frontier spans every layout of a node, so a gallery warm builds
    it once per GPU."""
    assert warm.main(["--gpu", "H100-SXM", "--gpu", "H200"]) == 0
    assert [gpu for gpu, _params in _frontier_calls] == ["H100-SXM", "H200"]


def test_every_gallery_warm_fills_the_model_facts_first(monkeypatch, _facts_calls):
    """The facts half of the page (`/gallery/models`) is warmed with any
    gallery warm, before the frontier; a sweeps-only run leaves it."""
    assert warm.main([]) == 0
    assert _facts_calls == ["facts"]

    assert warm.main(["--landing"]) == 0
    assert _facts_calls == ["facts", "facts"]

    async def fake_sweeps(**_kwargs):
        return 1

    monkeypatch.setattr(warm, "warm_sweeps", fake_sweeps)
    assert warm.main(["--sweeps"]) == 0
    assert _facts_calls == ["facts", "facts"]


def test_the_warm_pool_is_its_own_knob(monkeypatch):
    """The warm's pool is budgeted apart from the API's (one container):
    `HA_WARM_WORKERS`, not `HA_SWEEP_POOL_WORKERS`."""
    monkeypatch.setenv("HA_SWEEP_POOL_WORKERS", "6")
    monkeypatch.setenv("HA_WARM_WORKERS", "1")
    assert warm.WarmProcess.default_workers() == 1


def test_an_offline_warm_takes_up_to_32_workers_and_leaves_two_cores(monkeypatch):
    """Unset, the warm's pool is as wide as the box allows up to 32, two
    cores left to everything else."""
    monkeypatch.delenv("HA_WARM_WORKERS", raising=False)
    monkeypatch.setattr(warm.os, "cpu_count", lambda: 112)
    assert warm.WarmProcess.default_workers() == 32
    monkeypatch.setattr(warm.os, "cpu_count", lambda: 8)
    assert warm.WarmProcess.default_workers() == 6


def test_the_default_warm_pool_uses_the_wide_offline_budget(monkeypatch):
    """The offline warm is the only heavy process on its machine, so its
    default reaches 32 workers rather than the API's conservative default."""
    monkeypatch.delenv("HA_WARM_WORKERS", raising=False)
    monkeypatch.setattr(warm.os, "cpu_count", lambda: 112)
    assert warm.WarmProcess.default_workers() == 32


def test_the_frontier_receives_the_model_facts_parameter_costs(
        _frontier_calls):
    """Layout order reads the parameter facts the warm just produced."""
    assert warm.main(["--landing"]) == 0
    assert _frontier_calls == [(GALLERY_LANDING_GPU, {"m": 7})]


def test_the_warm_runs_its_jobs_in_its_pool(monkeypatch):
    """With workers, every heavy job of the warm runs in its pool, as in the
    server."""
    from backend.api import offload
    seen = {}
    monkeypatch.setattr(offload, "serve_requests_off_process",
                        lambda deadline_s=None: seen.setdefault("served", deadline_s) or True)
    monkeypatch.setattr(offload, "stop_serving", lambda: seen.setdefault("stopped", True))
    with warm.WarmProcess(nice=0, workers=2, pidfile=None):
        pass
    assert "served" in seen and seen["stopped"]
