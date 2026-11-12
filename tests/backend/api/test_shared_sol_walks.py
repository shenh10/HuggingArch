"""One canonical SOL walk record serving every public surface."""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from backend.code_trace.provider import recording
from backend.measure.cache import KeyedCache, MemoryBackend
from backend.inference.sol_data import ServingWalkFacts, StepLawKey
from backend.inference.sol_pool import SolPool
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import snapshot_config
from tests.sweep_parity import diff, report


MODEL = "meta-llama/Llama-3.2-3B"
PARALLEL = ParallelConfig(
    tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2).to_dict()


def _record_name(gallery, view, layout):
    """The producer's record name, derived through the same key owner."""
    stored = gallery._Stored(MODEL, gallery._resolve_one(MODEL)[2])
    walk_key = stored.serving_key(view)
    return StepLawKey.of_layout(walk_key, layout).record_name



@pytest.fixture(autouse=True)
def _one_computation():
    """Each test drives the worker side of a Gallery job directly — the
    fold, the read, a bundle written — so each is one recorded computation,
    as the job it stands in for is (`measure.pool.submit` runs every job in
    one: `code_trace.provider.run_recorded`). What it keeps is stored with
    the code it ran."""
    with recording():
        yield

@pytest.fixture
def shared(monkeypatch, tmp_path):
    """A Gallery-produced canonical walk and a one-cell Scaling request."""
    snapshot_config(MODEL)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    import backend.api.gallery_frontier as gallery
    import backend.api.sweep_runtime as scaling

    pool = SolPool(tmp_path / "sol-pool")
    monkeypatch.setattr(gallery, "sharedSolPool", pool)
    monkeypatch.setattr(scaling, "sharedSolPool", pool)
    monkeypatch.setattr(scaling, "_SWEEP", KeyedCache(
        "sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(scaling, "_SWEEP_STEP", KeyedCache(
        "sweep-step", MemoryBackend()))
    view = gallery.FrontierView("H100-SXM")
    cards = view.walk_cards()
    law = gallery._layout_law(
        (MODEL, cards, 0.9, gallery._GALLERY_PRECISION.canonical(), PARALLEL))
    stored = gallery._Stored(MODEL, gallery._resolve_one(MODEL)[2])
    record_name = _record_name(gallery, view, PARALLEL)
    assert gallery.sharedSolPool.write(
        stored.walk_identity(view),
        [(record_name,
          {"parallel": PARALLEL, "walk": law})],
        metadata={
            "cards": cards,
            "layoutsByCard": {sku: [PARALLEL] for sku in cards},
            "facts": ServingWalkFacts.producer().to_json(),
            "recordSchema": gallery.walkRecordSchema,
        })

    calls = {"cells": 0}
    realPlan = scaling._PARALLEL.plan
    realSteps = scaling._PARALLEL.steps

    def oneCell(req):
        full = realPlan(req)
        return dataclasses.replace(full, cells=full.cells[:1])

    def counted(cell, prices):
        calls["cells"] += 1
        yield from realSteps(cell, prices)

    monkeypatch.setattr(scaling, "_PARALLEL", dataclasses.replace(
        scaling._PARALLEL, plan=oneCell, steps=counted))
    return gallery, scaling, pool, calls


def _request(scaling, **changes):
    from backend.api._models import InferenceParallelSweepRequest

    body = {
        "model_id": MODEL,
        "gpu_sku": "",
        "context_len": 4096,
        "modes": ["tp_tp"],
        "gpu_skus": ["H100-SXM"],
        "avg_output_tokens": 256,
        "gpu_memory_fraction": 0.9,
        "selection_mode": "envelope",
        "sol_only": True,
    }
    return InferenceParallelSweepRequest(**{**body, **changes})


def _estimateRequest(scaling, **changes):
    from backend.api._models import BatchPoint, InferenceEstimateSweepRequest
    from backend.inference.sweep_grid import (
        ROUTINE_SEQ_LENS,
        SCALING_BATCH_GRID,
    )

    body = {
        "model_id": MODEL,
        "gpu_sku": "H100-SXM",
        "parallel": PARALLEL,
        "seq_lens": list(ROUTINE_SEQ_LENS),
        "batch_grid": [BatchPoint(**point) for point in SCALING_BATCH_GRID],
        "avg_output_tokens": 256,
        "gpu_memory_fraction": 0.9,
        "selection_mode": "envelope",
        "sol_only": True,
    }
    return InferenceEstimateSweepRequest(**{**body, **changes})


SERVED_SEQ_LENS = [1024, 4096, 16384]


def assertProjection(got, want, path="point"):
    """Every field the projection serves matches the complete estimate."""
    if isinstance(got, dict):
        assert set(got) == set(want), (
            f"{path}: projection fields {sorted(got)} != estimate fields "
            f"{sorted(want)}")
        for name in got:
            assertProjection(got[name], want[name], f"{path}.{name}")
        return
    if isinstance(got, (int, float)) and not isinstance(got, bool):
        assert got == pytest.approx(want, rel=1e-12, abs=0), path
    else:
        assert got == want


def assertProjectionPoint(got, want):
    assertProjection(got, want)


def test_scaling_projects_the_gallery_record_without_a_cell(shared, monkeypatch):
    _gallery, scaling, pool, calls = shared
    req = _request(scaling)

    reused = asyncio.run(scaling.compute_parallel_sweep_cached(req))
    assert calls["cells"] == 0
    assert pool.recordIds(
        scaling.sweep_theory_identity("parallel", req).key) is None

    monkeypatch.setattr(scaling, "_SWEEP", KeyedCache(
        "sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(scaling, "sharedSolPool", SolPool(
        pool.root.parent / "empty-pool"))
    direct = asyncio.run(scaling.compute_parallel_sweep_cached(req))
    assert calls["cells"] == 1
    assert scaling.sharedSolPool.recordIds(
        scaling.sweep_theory_identity("parallel", req).key) is None
    differences = diff(reused, direct)
    assert not differences, report(differences, limit=80)


def test_a_point_outside_the_producer_anchor_runs_its_own_cell(shared):
    _gallery, scaling, _pool, _calls = shared
    req = _request(scaling, context_len=6144)
    assert scaling._sharedWalkResults(
        scaling._PARALLEL, scaling._PARALLEL.plan(req)) == {}


def test_the_canonical_bundle_holds_every_scaling_layout(shared):
    """Scaling's ladder does not become direct cells beside the Gallery pool."""
    _gallery, scaling, _pool, _calls = shared
    assert asyncio.run(_gallery.keepSharedWalk(MODEL, "H100-SXM"))
    req = _request(scaling, context_len=1024, gpu_skus=["H100-SXM"])
    plan = scaling._PARALLEL.plan(req)
    assert len(plan.cells) >= 1
    assert set(scaling._sharedWalkResults(
        scaling._PARALLEL, plan)) == set(range(len(plan.cells)))


def test_a_decode_batch_outside_the_public_grid_is_substituted(shared):
    from backend.api._models import BatchPoint

    _gallery, scaling, _pool, _calls = shared
    req = _estimateRequest(
        scaling, seq_lens=[4096],
        batch_grid=[BatchPoint(key="b3", prefill_batch_size=1,
                               decode_batch_size=3)])
    assert set(scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(req))) == {0}


def test_nondefault_floors_are_substituted(shared):
    """Floors are law inputs, substituted by the one request owner."""
    _gallery, scaling, _pool, _calls = shared
    estimate = _estimateRequest(scaling, comm_latency_us=3.0)
    assert set(scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(estimate))) == {0}
    parallel = _request(scaling, launch_floor_us=2.0)
    assert set(scaling._sharedWalkResults(
        scaling._PARALLEL, scaling._parallel_plan(parallel))) == {0}


def test_a_paired_prefill_batch_is_substituted(shared):
    """Prefill batching is a law input, not a second walk."""
    from backend.api._models import BatchPoint

    _gallery, scaling, _pool, _calls = shared
    req = _estimateRequest(
        scaling,
        batch_grid=[BatchPoint(key="paired", prefill_batch_size=16,
                               decode_batch_size=16)])
    assert set(scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(req))) == {0}


def test_context_batch_projects_the_law_without_a_cell(shared, monkeypatch):
    _gallery, scaling, _pool, calls = shared
    calls["estimateCells"] = 0
    realSteps = scaling._ESTIMATE.steps

    def counted(cell, prices):
        calls["estimateCells"] += 1
        yield from realSteps(cell, prices)

    monkeypatch.setattr(scaling, "_ESTIMATE", dataclasses.replace(
        scaling._ESTIMATE, steps=counted))
    originalPool = scaling.sharedSolPool
    for req in (_estimateRequest(scaling),
                _estimateRequest(scaling, comm_latency_us=3.0)):
        before = calls["estimateCells"]
        reused = asyncio.run(scaling.compute_estimate_sweep_cached(req))
        assert calls["estimateCells"] == before

        monkeypatch.setattr(scaling, "_SWEEP", KeyedCache(
            "sweep", MemoryBackend(), single_flight=True))
        monkeypatch.setattr(scaling, "_SWEEP_STEP", KeyedCache(
            "sweep-step", MemoryBackend()))
        monkeypatch.setattr(scaling, "sharedSolPool", SolPool(
            originalPool.root.parent / "empty-estimate-pool"))
        direct = asyncio.run(scaling.compute_estimate_sweep_cached(req))
        monkeypatch.setattr(scaling, "sharedSolPool", originalPool)
        assert calls["estimateCells"] == before + 1
        for key, points in reused["results_by_batch"].items():
            for got, want in zip(points, direct["results_by_batch"][key],
                                 strict=True):
                assertProjectionPoint(got, want)
    assert calls["estimateCells"] == 2


def test_compare_rows_read_the_same_context_batch_projection(shared):
    _gallery, scaling, _pool, _calls = shared
    requests = [_estimateRequest(scaling, gpu_sku=gpu)
                for gpu in ("H100-SXM", "H200")]
    projections = [scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(req)) for req in requests]
    assert all(set(projection) == {0} for projection in projections)


def test_gallery_and_scaling_name_the_same_producer_anchor(shared):
    from backend.api.estimate_env import resolve_estimate_environment
    from backend.inference.sol_data import ServingWalkFacts

    gallery, _scaling, _pool, _calls = shared
    req = gallery._layout_request(
        MODEL, "H100-SXM", 0.9, gallery._GALLERY_PRECISION.canonical(), PARALLEL)
    assert (ServingWalkFacts.from_environment(resolve_estimate_environment(req))
            == ServingWalkFacts.producer())


def test_gallery_and_scaling_compile_the_same_law_graph_once(
        shared, monkeypatch):
    """A page subset and a Scaling ladder read one compiled law graph."""
    import backend.inference.sol_law as law

    gallery, scaling, _pool, _calls = shared
    view = gallery.FrontierView("H100-SXM")
    stored = gallery._Stored(MODEL, gallery._resolve_one(MODEL)[2])
    walk = stored.shared_walk(view)
    assert walk is not None
    # The compiler the arithmetic mode selects (`_compiledLaw`): both the
    # prefill's evaluation and the decode's floor terms are built with it.
    compiled = law._compiledLaw
    compiled.cache_clear()
    # The decode step is read off its floor terms (`_FloorTerms`), whose
    # program is compiled once per law like the prefill's graph.
    law._floorTerms.cache_clear()
    before = compiled.cache_info().misses
    gallery._restamp(MODEL, walk, "H100-SXM", view)
    assert compiled.cache_info().misses - before == 2  # prefill and decode
    restamped = gallery._restamp(MODEL, walk, "H100-SXM", view)
    assert "_laws" not in restamped
    assert "_law" not in restamped["_theory"][0]

    req = _estimateRequest(scaling, gpu_sku="H100-SXM")
    assert set(scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(req))) == {0}
    assert compiled.cache_info().misses - before == 2

    monkeypatch.setattr(
        law.LawGraph, "evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(
            "a serving entry re-evaluated its law")))
    assert gallery._at_view(restamped["_theory"][0], view) is not None

    nondefault = gallery.FrontierView("H100-SXM", gpu_memory_fraction=.8)
    gallery._restamp(MODEL, walk, "H100-SXM", nondefault)
    assert compiled.cache_info().misses - before == 2


def test_an_unserved_public_context_is_read_from_its_step_law(shared):
    """A serving projection owns an unserved point's status, not its time."""
    from backend.inference.sweep_grid import ROUTINE_SEQ_LENS

    _gallery, scaling, _pool, _calls = shared
    req = _estimateRequest(scaling, seq_lens=list(ROUTINE_SEQ_LENS))
    reused = scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(req))
    assert set(reused) == {0}
    rows = list(reused[0])
    assert len(rows) == len(ROUTINE_SEQ_LENS)
    assert all(len(row) == 4 for row in rows)
    assert [row[0]["sol"]["beyond_max_context"] for row in rows] == [
        context >= 262144 for context in ROUTINE_SEQ_LENS]


def test_an_infeasible_rtx_context_has_no_step_on_either_path(
        monkeypatch, tmp_path):
    """RTX-4090 TP2 at 1M context: no B=1 fallback, on either estimate path."""
    from backend.api._models import BatchPoint, InferenceEstimateSweepRequest

    snapshot_config(MODEL)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    import backend.api.gallery_frontier as gallery
    import backend.api.sweep_runtime as scaling

    gpu = "RTX-4090"
    pool = SolPool(tmp_path / "producer-pool")
    monkeypatch.setattr(gallery, "sharedSolPool", pool)
    monkeypatch.setattr(scaling, "sharedSolPool", pool)
    monkeypatch.setattr(scaling, "_SWEEP", KeyedCache(
        "sweep", MemoryBackend(), single_flight=True))
    monkeypatch.setattr(scaling, "_SWEEP_STEP", KeyedCache(
        "sweep-step", MemoryBackend()))
    view = gallery.FrontierView(gpu)
    cards = view.walk_cards()
    law = gallery._layout_law(
        (MODEL, cards, 0.9, gallery._GALLERY_PRECISION.canonical(), PARALLEL))
    stored = gallery._Stored(MODEL, gallery._resolve_one(MODEL)[2])
    record_name = _record_name(gallery, view, PARALLEL)
    assert pool.write(
        stored.walk_identity(view),
        [(record_name, {"parallel": PARALLEL, "walk": law})],
        metadata={
            "cards": cards,
            "layoutsByCard": {sku: [PARALLEL] for sku in cards},
            "facts": ServingWalkFacts.producer().to_json(),
            "recordSchema": gallery.walkRecordSchema,
        })

    for floor_change in ({}, {"comm_latency_us": 3.0}):
        req = InferenceEstimateSweepRequest(
            model_id=MODEL, gpu_sku=gpu, parallel=PARALLEL,
            seq_lens=[1048576],
            batch_grid=[
                BatchPoint(key="b1", prefill_batch_size=1,
                           decode_batch_size=1),
                BatchPoint(key="max", prefill_batch_size=1,
                           decode_batch_size=None),
            ],
            avg_output_tokens=256, gpu_memory_fraction=0.9,
            selection_mode="envelope", sol_only=True, **floor_change)
        reused = scaling._sharedWalkResults(
            scaling._ESTIMATE, scaling._estimate_plan(req))
        assert set(reused) == {0}
        fast = [point["sol"] for row in reused[0] for point in row]

        monkeypatch.setattr(scaling, "sharedSolPool", SolPool(
            pool.root.parent / "empty-pool"))
        monkeypatch.setattr(scaling, "_SWEEP", KeyedCache(
            "sweep", MemoryBackend(), single_flight=True))
        monkeypatch.setattr(scaling, "_SWEEP_STEP", KeyedCache(
            "sweep-step", MemoryBackend()))
        direct_response = asyncio.run(
            scaling.compute_estimate_sweep_cached(req))
        monkeypatch.setattr(scaling, "sharedSolPool", pool)
        direct = [point for points
                  in direct_response["results_by_batch"].values()
                  for point in points]

        assert len(fast) == len(direct) == 2
        for got, want in zip(fast, direct, strict=True):
            assertProjectionPoint(got, want)
            assert got["max_decode_batch_per_gpu"] == 0
            reason = want["infeasibility_reason"]
            assert reason
            assert got["beyond_max_context"] is True
            for phase in got["phases"].values():
                assert phase["batch"] is None
                assert "t_sol_s" not in phase
                assert phase["time_unpublished_reason"] == reason


def test_the_producer_bundle_covers_every_parallelism_cell(shared):
    from backend.inference.parallel_sweep import ALL_MODES, config_for_degree
    from backend.inference.sweep_grid import SCALING_DEGREES

    gallery, scaling, _pool, _calls = shared
    resolved = gallery._resolve_one(MODEL)[1]
    view = gallery.FrontierView("H100-SXM")
    produced = {_record_name(gallery, view, layout) for layout in
                gallery._walk_plan(resolved, view, scaling=True)["layouts"]}
    req = _request(scaling, gpu_skus=None)
    plan = scaling._parallel_plan(req)
    assert plan.cells
    assert {_record_name(gallery, view, cell.parallel)
            for cell in plan.cells} <= produced
    assert {_record_name(gallery, view, config_for_degree(mode, degree))
            for mode in ALL_MODES
            for degree in SCALING_DEGREES} <= produced


def test_the_gallery_plan_does_not_pay_for_scaling_tail_rungs(shared):
    gallery, _scaling, _pool, _calls = shared
    resolved = gallery._resolve_one(MODEL)[1]
    view = gallery.FrontierView("H100-SXM")
    frontier = gallery._walk_plan(resolved, view)
    scaling = gallery._walk_plan(resolved, view, scaling=True)
    assert all(gallery.ParallelConfig.from_dict(layout).cluster_size <= 64
               for layout in frontier["layouts"])
    frontier_ids = {_record_name(gallery, view, layout)
                    for layout in frontier["layouts"]}
    scaling_ids = {_record_name(gallery, view, layout)
                   for layout in scaling["layouts"]}
    assert frontier_ids < scaling_ids


def test_a_card_of_another_class_reads_its_own_class_bundle(shared, monkeypatch):
    """A Hopper bundle — even one whose manifest claims B200 among its cards
    — is not B200's to read: B200 (compute capability 10.0) is another walk
    class, whose rows differ, and it names its own class's bundle. With none
    kept for it the cell runs rather than pricing B200 off Hopper rows; a
    series of Hopper cards alone still projects."""
    gallery, scaling, pool, _calls = shared
    claims = SolPool(pool.root.parent / "claims-b200")
    monkeypatch.setattr(gallery, "sharedSolPool", claims)
    monkeypatch.setattr(scaling, "sharedSolPool", claims)
    stored = gallery._Stored(MODEL, gallery._resolve_one(MODEL)[2])
    hopper = gallery.FrontierView("H100-SXM")
    record_name = _record_name(gallery, hopper, PARALLEL)
    item = pool.record(stored.walk_identity(hopper).key,
                       record_name)
    cards = [*hopper.walk_cards(), "B200"]
    assert claims.write(
        stored.walk_identity(hopper),
        [(record_name, item)],
        metadata={
            "cards": cards,
            "layoutsByCard": {sku: [PARALLEL] for sku in cards},
            "facts": ServingWalkFacts.producer().to_json(),
            "recordSchema": gallery.walkRecordSchema,
        })
    same = _request(scaling, gpu_skus=["H100-SXM", "H200"])
    assert set(scaling._sharedWalkResults(
        scaling._PARALLEL, scaling._PARALLEL.plan(same))) == {0}
    mixed = _request(scaling, gpu_skus=["H100-SXM", "B200"])
    assert scaling._sharedWalkResults(
        scaling._PARALLEL, scaling._PARALLEL.plan(mixed)) == {}
    blackwell = _estimateRequest(scaling, gpu_sku="B200")
    assert scaling._sharedWalkResults(
        scaling._ESTIMATE, scaling._estimate_plan(blackwell)) == {}
    # The Gallery reads B200's own class's bundle, not this one.
    assert stored.shared_walk(gallery.FrontierView("B200")) is None
    assert stored.shared_walk(hopper) is not None
