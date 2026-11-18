"""A Scaling sweep's SOL and measured answers (docs/auto_sweep.md §7).

The SOL answer (`sol_only`, or a theory body) is cached without a scope and
without measurement-only request fields. A measured answer is never cached:
each request computes its measures, makes ONE Kernel Bench lookup and folds
in the process that assembles it, so no Kernel Bench write invalidates
anything. The canonical SOL pool holds layout laws only — never a sweep's
expanded point payload.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from backend.measure.cache import KeyedCache, MemoryBackend
from backend.inference.sol_pool import SolPool
from tests.snapshots import snapshot_config
from tests.sweep_parity import FixtureBench, diff

MODEL = "meta-llama/Llama-3.2-3B"


@pytest.fixture
def sweeps(monkeypatch, tmp_path):
    """The real sweep compute, offline, on a fresh in-memory cache; counts
    the cells that run."""
    snapshot_config(MODEL)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    import backend.api.sweep_runtime as rt
    monkeypatch.setattr(rt, "fetch_config", lambda m, overlay_dir=None: snapshot_config(m))
    monkeypatch.setattr(rt, "_SWEEP", KeyedCache("sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(rt, "_SWEEP_STEP", KeyedCache("sweep-step", MemoryBackend()))
    monkeypatch.setattr(rt, "sharedSolPool", SolPool(tmp_path / "sol-pool"))
    cells = {"n": 0}
    for name in ("_PARALLEL", "_ESTIMATE"):
        sweep = getattr(rt, name)
        real = sweep.steps

        def counted(cell, prices, _real=real):
            cells["n"] += 1
            yield from _real(cell, prices)
        monkeypatch.setattr(rt, name, dataclasses.replace(sweep, steps=counted))
    return rt, cells


def _estimate_req(scope, mode="latest", **kw):
    from backend.api._models import BatchPoint, InferenceEstimateSweepRequest
    body = dict(model_id=MODEL, gpu_sku="H100-SXM", seq_lens=[1024, 4096],
                batch_grid=[BatchPoint(key="b1", prefill_batch_size=1, decode_batch_size=1),
                            BatchPoint(key="max", prefill_batch_size=-1,
                                       decode_batch_size=None)],
                avg_output_tokens=256, selection_mode=mode)
    return InferenceEstimateSweepRequest(**{**body, **kw}).model_copy(
        update={"measurement_scope": scope})


def _parallel_req(scope, mode="latest", **kw):
    from backend.api._models import InferenceParallelSweepRequest
    body = dict(model_id=MODEL, gpu_sku="", contexts=[4096], modes=["tp_tp"],
                gpu_skus=["H100-SXM", "B200"], avg_output_tokens=256, selection_mode=mode)
    return InferenceParallelSweepRequest(**{**body, **kw}).model_copy(
        update={"measurement_scope": scope})


def _measured(out: dict) -> list:
    return [(p["phases"]["decode"]["t_measured_s"], p["phases"]["decode"]["coverage"])
            for pts in out["results_by_batch"].values() for p in pts]


def test_a_measured_answer_is_made_live_each_request(sweeps):
    rt, cells = sweeps
    with FixtureBench() as bench:
        first = asyncio.run(rt.compute_estimate_sweep_cached(_estimate_req(bench.scope)))
        assert cells["n"] == 1 and bench.lookups == 1      # one walk, one lookup
        assert rt.sweep_cache_key("sweep", _estimate_req(bench.scope)) is None
        # The same request again: not stored, so it measures and looks up again.
        again = asyncio.run(rt.compute_estimate_sweep_cached(_estimate_req(bench.scope)))
        assert cells["n"] == 2 and bench.lookups == 2
        assert _measured(again) == _measured(first)
        # The fold ran: every point states what it measured.
        assert all(cov and cov["measured"] > 0 for _, cov in _measured(first))
        # Kernel Bench changes (a new shared snapshot): the measured answer follows it.
        from dataclasses import replace
        moved = replace(bench.scope, shared_snapshot=2)
        assert (rt.sweep_theory_identity("sweep", _estimate_req(moved)).key
                == rt.sweep_theory_identity("sweep", _estimate_req(bench.scope)).key)
        asyncio.run(rt.compute_estimate_sweep_cached(_estimate_req(moved)))
        assert cells["n"] == 3 and bench.lookups == 3
        # Another selection mode measures live too.
        asyncio.run(rt.compute_estimate_sweep_cached(_estimate_req(bench.scope, "envelope")))
        assert cells["n"] == 4 and bench.lookups == 4
        # A theory request has its own cache identity and reads no lake.
        theory = asyncio.run(rt.compute_estimate_sweep_cached(
            _estimate_req(bench.scope, "theoretical_only")))
        assert cells["n"] == 5 and bench.lookups == 4
        assert all(t is None for t, _ in _measured(theory))


@pytest.mark.parametrize("kind", ["sweep", "parallel"])
def test_the_sol_answer_seeds_the_measured_one_and_is_the_theory_answer(
        sweeps, kind, monkeypatch, tmp_path):
    """The panel's first POST (`sol_only`, its mode measured) reads no
    measurement and stores its own answer. The measured POST after it
    computes its measures live rather than turning that response payload into
    a measured cache. Its answer is the theory body's answer, byte for byte —
    computed the other way round (a theory request on a fresh cache)."""
    rt, cells = sweeps
    req = _estimate_req if kind == "sweep" else _parallel_req
    compute = (rt.compute_estimate_sweep_cached if kind == "sweep"
               else rt.compute_parallel_sweep_cached)
    with FixtureBench() as bench:
        sol = asyncio.run(compute(req(bench.scope, sol_only=True)))
        ran = cells["n"]
        assert ran >= 1 and bench.lookups == 0
        again = asyncio.run(compute(req(bench.scope, "envelope", sol_only=True)))
        assert cells["n"] == ran and bench.lookups == 0 and again == sol
        measured = asyncio.run(compute(req(bench.scope)))
        assert cells["n"] > ran and bench.lookups == 1
        assert measured != sol                 # the fold wrote a measured track
    monkeypatch.setattr(rt, "_SWEEP", KeyedCache("sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(rt, "sharedSolPool", SolPool(tmp_path / "sol-theory"))
    theory = asyncio.run(compute(req(None, "theoretical_only")))
    assert theory == sol
    # Computed afresh, a SOL request carrying posted rows and a scope reads
    # none of them: the same answer, no lookup.
    monkeypatch.setattr(rt, "_SWEEP", KeyedCache("sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(rt, "sharedSolPool", SolPool(tmp_path / "sol-rows"))
    with FixtureBench() as bench:
        rows = asyncio.run(compute(req(bench.scope, "latest", sol_only=True,
                                       calibration_rows=[{"op": "x"}])))
        assert rows == sol and bench.lookups == 0


def test_a_theory_request_leaves_no_theory_half_a_measured_one_would_read(sweeps):
    """A theory request asks no point for its rows' keys, so the theory half
    it computed answers no measured request: it is not stored, and a
    measured request after it runs the cell and reads what it measures."""
    rt, cells = sweeps
    with FixtureBench() as bench:
        asyncio.run(rt.compute_estimate_sweep_cached(
            _estimate_req(bench.scope, "theoretical_only")))
        assert cells["n"] == 1 and bench.lookups == 0
        measured = asyncio.run(rt.compute_estimate_sweep_cached(_estimate_req(bench.scope)))
        assert cells["n"] == 2 and bench.lookups == 1
        # It read what it measures: every point states rows measured. (Its
        # total stays None unless every op is measured — all or nothing — and
        # the fixture measures some of the step's ops, not all.)
        assert all(cov and cov["measured"] > 0 for _, cov in _measured(measured))


def test_a_direct_sweep_never_writes_its_points_to_the_law_pool(sweeps):
    """The canonical pool is laws, not a second response cache: even a cell
    outside every canonical anchor leaves no sweep record behind."""
    rt, cells = sweeps
    request = _estimate_req(None, "theoretical_only")
    first = asyncio.run(rt.compute_estimate_sweep_cached(request))
    assert cells["n"] == 1
    assert rt.sharedSolPool.recordIds(
        rt.sweep_theory_identity("sweep", request).key) is None
    assert diff(rt._SWEEP.get(rt.sweep_cache_key("sweep", request)), first) == []


def test_an_image_warm_does_not_bake_response_documents(sweeps, monkeypatch):
    """`HA_SOL_POOL_ONLY` is the build's disk budget: bake canonical laws, not
    request-shaped sweep answers."""
    rt, _cells = sweeps
    request = _estimate_req(None, "theoretical_only")
    monkeypatch.setattr(rt, "poolOnly", True)
    assert asyncio.run(rt.compute_estimate_sweep_cached(request))
    identity = rt.sweep_theory_identity("sweep", request)
    assert rt.sharedSolPool.recordIds(identity.key) is None
    assert rt._SWEEP.get(identity.key) is None
    assert rt._SWEEP.get(rt.sweep_cache_key("sweep", request)) is None


def test_the_parallel_sweep_looks_up_once_for_every_layout_and_card(sweeps):
    rt, cells = sweeps
    with FixtureBench() as bench:
        out = asyncio.run(rt.compute_parallel_sweep_cached(_parallel_req(bench.scope)))
    assert cells["n"] > 1 and bench.lookups == 1
    assert [p for s in out["series"] for p in s["points"] if p["feasible"]]
    # The one lookup reached every card of every layout: each point of the
    # sweep's cloud carries the prefill coverage that fold wrote.
    priced = [p for p in out["throughput_cloud"] if "refused" not in p]
    assert priced and all(p["prefill_coverage"]["total"] > 0 for p in priced)


def test_every_axis_that_makes_points_is_in_the_theory_key(monkeypatch):
    """Two series of cards, two context grids, two batch grids, two mode
    lists: two theory keys each. The measurement-only fields are not in it."""
    import backend.api.sweep_runtime as rt
    from backend.api._models import BatchPoint
    monkeypatch.setattr(rt, "result_version", lambda r: "rv")
    monkeypatch.setattr(rt, "resolved_spec_for", lambda m: object())
    key = lambda kind, r: rt.sweep_theory_identity(kind, r).key
    assert key("parallel", _parallel_req(None)) != key(
        "parallel", _parallel_req(None, gpu_skus=["H100-SXM"]))
    assert key("parallel", _parallel_req(None)) != key(
        "parallel", _parallel_req(None, modes=["tp_tp", "dp_ep"]))
    assert key("sweep", _estimate_req(None)) != key(
        "sweep", _estimate_req(None, seq_lens=[1024]))
    assert key("sweep", _estimate_req(None)) != key("sweep", _estimate_req(
        None, batch_grid=[BatchPoint(key="b1", prefill_batch_size=1, decode_batch_size=1)]))
    for kind, req in (("sweep", _estimate_req), ("parallel", _parallel_req)):
        base = key(kind, req(None))
        assert key(kind, req(None, "envelope")) == base
        assert key(kind, req(None, "theoretical_only")) == base
        assert key(kind, req(None, calibration_rows=[{"x": 1}])) == base
