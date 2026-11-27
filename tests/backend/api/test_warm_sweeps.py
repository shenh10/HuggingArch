"""Warm/serve parity — the death-gate for the sweep warm job.

Two independent axes are pinned:

1. **Key parity against the frontend's literal body.** `warm_sweeps` must write
   the SAME cache key a first visitor's Scaling-panel POST produces. The panel
   body is hardcoded here from `web/src/pages/ScalingPanel.tsx` (fetch body) +
   `web/src/pages/Inference.tsx` (seeded state defaults); if warm's request
   builder or the panel body drift on any keyed field, the keys diverge and this
   test fails — which is exactly the production-miss the reviewers caught.

2. **Hit, not recompute.** After warming, serving the same request returns the
   warmed value without recomputing.

Offline: the spec fingerprint resolves from a snapshot, the expensive sweep is
stubbed to a marker, Supabase is off, and the cache is a fresh in-memory tier.
"""
from __future__ import annotations

import asyncio

import pytest

from backend.measure.cache import KeyedCache, MemoryBackend
from backend.inference.sol_pool import SolPool
from tests.snapshots import snapshot_config

MODEL = "Qwen/Qwen3-30B-A3B"  # MoE → EP modes produce real series; bf16 weights


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Offline + isolated: real spec fingerprint from the snapshot, a fresh
    in-memory sweep cache, Supabase off, a deterministic weight-bytes seed, and
    a stubbed (fast, counted) compute."""
    snapshot_config(MODEL)  # skips when the snapshot is not checked out
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    import backend.api.routes.inference as inf
    import backend.api.sweep_runtime as sweep_rt
    import backend.api.warm as warm

    async def fakeSharedWalks(_models, _gpus):
        return 0

    monkeypatch.setattr(warm, "ensureSharedWalks", fakeSharedWalks)

    # Real key path, no network: fingerprint resolves from the snapshot config.
    # The key builder (`sweep_runtime.resolved_spec_for`) reads its own import.
    monkeypatch.setattr(sweep_rt, "fetch_config",
                        lambda m, overlay_dir=None: snapshot_config(MODEL))

    # Isolate the cache from ~/.cache / Supabase. Compute lives in
    # sweep_runtime (shared with the route), so the patch lands there.
    cache = KeyedCache("sweep", MemoryBackend(), single_flight=True)
    monkeypatch.setattr(sweep_rt, "_SWEEP", cache)
    monkeypatch.setattr(inf, "_SWEEP", cache)
    monkeypatch.setattr(sweep_rt, "sharedSolPool", SolPool(tmp_path / "sol-pool"))

    calls = {"n": 0}

    class _FakeParallelPlan:
        """A sweep with no cells whose answer is a marker; counts plans."""
        cells = ()

        @classmethod
        def of(cls, *, modes, contexts, batch, gpu_skus, base_env,
               ep_load_factor=1.0):
            calls["n"] += 1
            plan = cls()
            plan.modes, plan.contexts = list(modes), list(contexts)
            return plan

        def measurement(self):
            return {"selection_mode": "theoretical_only"}

        def assemble(self, results):
            return {"results": ["marker"], "modes": self.modes,
                    "contexts": self.contexts}

    monkeypatch.setattr(sweep_rt, "ParallelSweepPlan", _FakeParallelPlan)

    class _FakeCapacityPlan:
        @classmethod
        def of(cls, *, seq_lens, batch_grid=None, **_kw):
            calls["sweep"] = calls.get("sweep", 0) + 1
            plan = cls()
            plan.seq_lens, plan.gridded = list(seq_lens), bool(batch_grid)
            return plan

        def measurement(self):
            return {"selection_mode": "theoretical_only"}

        def cells(self):
            return []

        def shape(self, results):
            if self.gridded:
                return {"max": [{"marker": "batch"} for _ in self.seq_lens]}
            return [{"marker": "kv"} for _ in self.seq_lens]

        def throughput_cloud(self, results):
            return []

    monkeypatch.setattr(sweep_rt, "CapacitySweepPlan", _FakeCapacityPlan)
    monkeypatch.setattr(sweep_rt, "resolve_estimate_environment", lambda req: {"env": "stub"})
    return inf, warm, calls


def _frontend_first_visitor_body(model_id: str) -> dict:
    """The literal SOL POST body the Scaling panel sends on first load — mirrored
    from web/src/pages/ScalingPanel.tsx (fetch body) + Inference.tsx (seeded
    defaults: frac 0.9, selection 'envelope', and Plan's output length
    `outTok` = 256, the routine grid's; precision as shipped, which the panel sends as no
    `precision` at all). Its contexts are the routine grid (`SWEEP_SEQ_LENS`),
    whichever context its dropdown shows. The panel does NOT send gpu_skus;
    framework defaults to '' → not sent. gpu_sku is a real SKU but is dropped
    from the parallel key."""
    return {
        "model_id": model_id,
        "gpu_sku": "H100-SXM",
        "contexts": [1024, 2048, 4096, 8192, 16384, 32768, 131072, 262144,
                     1048576],
        "modes": ["dp_ep", "cp_ep", "tp_ep", "tp_tp"],
        "gpu_memory_fraction": 0.9,
        "avg_output_tokens": 256,
        "selection_mode": "envelope",
        "sol_only": True,
    }


def _frontend_batch_body(model_id: str, gpu: str, mode: str, degree: int) -> dict:
    """Literal Context × Batch SOL POST from ScalingPanel.tsx + compare.ts grids."""
    from backend.inference.sweep_grid import (
        ROUTINE_SEQ_LENS,
        SCALING_PANEL_BATCH_GRID,
    )
    return {
        "model_id": model_id,
        "gpu_sku": gpu,
        "seq_lens": list(ROUTINE_SEQ_LENS),
        "batch_grid": [dict(bp) for bp in SCALING_PANEL_BATCH_GRID],
        "avg_output_tokens": 256,
        "parallel_preset": {"mode": mode, "degree": degree},
        "gpu_memory_fraction": 0.9,
        "selection_mode": "envelope",
        "sol_only": True,
    }


def test_warm_request_matches_frontend_first_visitor_key(isolated):
    """Anti-drift anchor: warm's request key == the frontend first-visitor key."""
    inf, warm, _ = isolated
    from backend.api._models import InferenceParallelSweepRequest

    warm_req = warm.default_scaling_sweep_request(MODEL)
    frontend_req = InferenceParallelSweepRequest(
        **_frontend_first_visitor_body(MODEL)
    )
    warm_key = inf._sweep_cache_key("parallel", warm_req)
    frontend_key = inf._sweep_cache_key("parallel", frontend_req)
    assert warm_key is not None
    assert warm_key == frontend_key, (
        "warm and the Scaling panel disagree on a keyed field — a warmed entry "
        "would land on a key production traffic never reads"
    )


def test_scaling_warm_produces_each_missing_card_class_once(monkeypatch):
    """The producer fills canonical walks before projecting Scaling cells."""
    import backend.api.warm as warm

    calls = []

    async def fakeKeep(model, gpu):
        calls.append((model, gpu))
        return True

    monkeypatch.setattr(warm, "keepSharedWalk", fakeKeep)
    assert asyncio.run(warm.ensureSharedWalks(
        ["test/Model"], ["RTX-4090", "B200", "GB200"])) == 2
    assert calls == [("test/Model", "RTX-4090"), ("test/Model", "GB200")]


def test_warm_batch_request_matches_frontend_first_visitor_key(isolated):
    inf, warm, _ = isolated
    from backend.api._models import InferenceEstimateSweepRequest

    warm_req = warm.default_scaling_batch_request(MODEL, "B200", "dp_ep", 8)
    frontend_req = InferenceEstimateSweepRequest(
        **_frontend_batch_body(MODEL, "B200", "dp_ep", 8)
    )
    warm_key = inf._sweep_cache_key("sweep", warm_req)
    frontend_key = inf._sweep_cache_key("sweep", frontend_req)
    assert warm_key is not None
    assert warm_key == frontend_key, (
        "warm and the Scaling panel disagree on a keyed batch-sweep field"
    )


def test_warm_then_serve_is_a_hit_not_a_recompute(isolated):
    _inf, warm, calls = isolated
    import backend.api.sweep_runtime as sweep_rt
    from backend.api._models import (
        InferenceEstimateSweepRequest,
        InferenceParallelSweepRequest,
    )

    ok = asyncio.run(warm.warm_sweeps(
        models=[MODEL], gpus=["B200"], modes=["dp_ep"], degrees=[8],
    ))
    assert ok == 2  # one parallel + one batch
    assert calls["n"] == 1
    assert calls["sweep"] == 1

    frontend_req = InferenceParallelSweepRequest(
        **_frontend_first_visitor_body(MODEL)
    )
    # A stored answer starts no work, so the rate limiter (which counts
    # starts, `offload.set_start_hook`) is not charged for serving it.
    import backend.api.offload as offload
    starts = []
    offload.set_start_hook(lambda: starts.append(1))
    try:
        served = asyncio.run(sweep_rt.compute_parallel_sweep_cached(frontend_req))
    finally:
        offload.set_start_hook(None)
    assert starts == []
    assert served["results"] == ["marker"]
    assert served["code_version"]
    assert calls["n"] == 1

    batch_req = InferenceEstimateSweepRequest(
        **_frontend_batch_body(MODEL, "B200", "dp_ep", 8)
    )
    batch_served = asyncio.run(sweep_rt.compute_estimate_sweep_cached(batch_req))
    assert "results_by_batch" in batch_served
    assert calls["sweep"] == 1


def test_a_long_tail_post_does_not_stop_the_next_post(isolated, monkeypatch):
    """The pool's next work is submitted before the current POST completes.

    A POST is a submitter whose cells occupy the pool, not one indivisible pool
    job.  The old outer loop awaited it before submitting the next POST, so one
    long-tail cell left every other worker idle.
    """
    _inf, warm, _calls = isolated
    second_started = asyncio.Event()

    async def fake_parallel(_req):
        try:
            await asyncio.wait_for(second_started.wait(), timeout=1)
        except TimeoutError:
            raise AssertionError(
                "the second POST waited for the long tail") from None
        return {"results": ["slow"]}

    async def fake_batch(_req):
        second_started.set()
        return {"results_by_batch": ["next"]}

    monkeypatch.setattr(warm, "compute_parallel_sweep_cached", fake_parallel)
    monkeypatch.setattr(warm, "compute_estimate_sweep_cached", fake_batch)
    monkeypatch.setattr(warm, "scaling_jobs", lambda *_args, **_kwargs: [
        ("parallel", "slow", object()),
        ("batch", "next", object()),
    ])

    assert asyncio.run(warm.warm_sweeps(models=[MODEL])) == 2


def test_pool_only_warm_bakes_laws_without_response_documents(isolated, monkeypatch):
    """An image build keeps canonical SOL laws, not request-shaped answers."""
    _inf, warm, calls = isolated

    async def laws(_models, _gpus):
        return 1

    monkeypatch.setattr(warm, "ensureSharedWalks", laws)
    monkeypatch.setattr(warm, "poolOnly", True)
    assert asyncio.run(warm.warm_sweeps(models=[MODEL])) == 1
    assert calls["n"] == 0
    assert calls.get("sweep", 0) == 0


def test_parallel_jobs_include_default_model_once_with_every_context(monkeypatch):
    """The default landing model (DeepSeek-V3) is warmed even if it is not in
    the featured set — the single most common first hit — with ONE parallel
    sweep that prices the routine context grid (the decode cloud's contexts,
    `ROUTINE_SEQ_LENS`), not one sweep per dropdown context."""
    import backend.api.warm as warm
    from backend.inference.sweep_grid import DEFAULT_INFER_MODEL, ROUTINE_SEQ_LENS

    monkeypatch.setattr(
        "backend.api.gallery_overview.featured_examples",
        lambda *a, **k: ["org/Featured-A", DEFAULT_INFER_MODEL, "org/Featured-B"],
    )
    jobs = [(label, req) for kind, label, req in warm.scaling_jobs(
        gpus=["B200"], modes=["dp_ep"], degrees=[8]) if kind == "parallel"]
    models = [req.model_id for _, req in jobs]
    assert models[0] == DEFAULT_INFER_MODEL              # default first
    assert models.count(DEFAULT_INFER_MODEL) == 1        # deduped (once)
    for _, req in jobs:
        assert req.contexts == list(ROUTINE_SEQ_LENS)
        assert 65536 not in req.contexts                 # no unmeasured 64K rung
        assert 1048576 in req.contexts                   # upper bound is 1M


def test_scaling_jobs_cover_parallel_and_batch(monkeypatch):
    """`--sweeps` fills every Scaling dropdown: one parallel sweep per model
    (every context inside) plus GPU × parallel × degree (Context × Batch).
    KV storage reuses the batch payload, so it is not a third kind."""
    import backend.api.warm as warm
    from backend.inference.parallel_sweep import ALL_MODES, default_sweep_gpus
    from backend.inference.sweep_grid import SCALING_DEGREES

    jobs = warm.scaling_jobs(models=[MODEL])
    kinds = {k for k, _, _ in jobs}
    assert kinds == {"parallel", "batch"}
    n_mid = 1
    assert sum(1 for k, _, _ in jobs if k == "parallel") == n_mid
    assert sum(1 for k, _, _ in jobs if k == "batch") == (
        # The series a warm defaults to is the default framework's own
        # cards (`default_sweep_gpus`): the sheet spans two devices now,
        # and warming vLLM onto an Ascend card is `FrameworkNotOnDevice`.
        n_mid * len(default_sweep_gpus()) * len(ALL_MODES) * len(SCALING_DEGREES)
    )


def test_only_an_answer_that_reads_no_measurement_is_keyed(monkeypatch):
    """The SOL answer is keyed on the theory's identity alone: whatever the
    lake holds, whatever the mode, a `sol_only` body and a `theoretical_only`
    one are one entry. An answer that reads measurements has no key — it is
    made live from the theory half — and neither has one whose model has no
    spec now."""
    import backend.api.routes.inference as inf
    import backend.api.sweep_runtime as sweep_rt
    from backend.api._models import InferenceEstimateSweepRequest, InferenceParallelSweepRequest
    from backend.measure.scope import MeasurementScope

    monkeypatch.setattr(sweep_rt, "result_version", lambda r: "rv")
    monkeypatch.setattr(sweep_rt, "resolved_spec_for", lambda m: object())

    def scoped(r, shared):
        return r.model_copy(update={"measurement_scope": MeasurementScope(
            None, "lake", 100, shared, "rules")})

    def req(mode=None, shared=1, **kw):
        if mode:
            kw["selection_mode"] = mode
        return scoped(InferenceEstimateSweepRequest(
            model_id="x/y", gpu_sku="H100-SXM", seq_lens=[4096], **kw), shared)

    k_sol = inf._sweep_cache_key("sweep", req("theoretical_only"))
    assert k_sol is not None
    for mode in (None, "latest", "envelope", "theoretical_only"):
        for shared in (1, 2):
            assert inf._sweep_cache_key("sweep", req(mode, shared, sol_only=True)) == k_sol
    for mode in (None, "latest", "envelope"):
        assert inf._sweep_cache_key("sweep", req(mode)) is None
    # Posted rows reach no cached answer: the SOL answer reads none.
    assert inf._sweep_cache_key("sweep", req("latest", sol_only=True,
                                             calibration_rows=[{"op": "x"}])) == k_sol

    def par(mode, shared, **kw):
        return scoped(InferenceParallelSweepRequest(
            model_id="x/y", gpu_sku="", contexts=[4096], modes=["dp_ep"],
            selection_mode=mode, **kw), shared)
    assert inf._sweep_cache_key("parallel", par("latest", 1)) is None
    assert inf._sweep_cache_key("parallel", par("latest", 1, sol_only=True)) == inf._sweep_cache_key(
        "parallel", par("theoretical_only", 2))

    # Unresolvable spec → no key at all (computed uncached, never shared).
    monkeypatch.setattr(sweep_rt, "resolved_spec_for", lambda m: None)
    assert inf._sweep_cache_key("sweep", req("theoretical_only")) is None


def test_parallel_key_carries_every_field_a_point_reads(monkeypatch):
    """The parallel sweep's key drops only the fields each point sets for
    itself. The output length sizes every request's KV (decode batch ceiling
    94 vs 33 for Qwen3-8B on an H100 at 4K in, 256 vs 8192 out), so two
    requests that differ only there must not share an entry; the dropped
    fields must not split one."""
    import backend.api.sweep_runtime as sweep_rt
    from backend.api._models import InferenceParallelSweepRequest

    monkeypatch.setattr(sweep_rt, "result_version", lambda r: "rv")
    monkeypatch.setattr(sweep_rt, "resolved_spec_for", lambda m: object())

    def key(**kw):
        body = {"model_id": "x/y", "gpu_sku": "H100-SXM", "contexts": [4096],
                "modes": ["tp_tp"], "selection_mode": "theoretical_only", **kw}
        return sweep_rt.sweep_cache_key(
            "parallel", InferenceParallelSweepRequest(**body))

    assert key(avg_output_tokens=256) != key(avg_output_tokens=8192)
    base = key(avg_output_tokens=256)
    assert key(avg_output_tokens=256, gpu_sku="B200", num_gpus=8,
               avg_input_tokens=65536, parallel={"tp_attention": 4}) == base


def test_routing_imbalance_keys_both_sweeps_and_reaches_every_point(monkeypatch):
    """ρ = `ep_load_factor` moves every MoE point's number, so two sweeps that
    differ only in ρ never share a cached result — the parallel sweep carries
    it as a request field (its `parallel` is per-point and dropped from the
    key), the batch sweep inside its preset — and each point is estimated
    under it, a workload input beside the deployment."""
    import backend.api.sweep_runtime as sweep_rt
    from backend.api._models import (
        InferenceEstimateSweepRequest, InferenceParallelSweepRequest)

    monkeypatch.setattr(sweep_rt, "result_version", lambda r: "rv")
    monkeypatch.setattr(sweep_rt, "resolved_spec_for", lambda m: object())

    def par(**kw):
        return InferenceParallelSweepRequest(
            model_id="x/y", gpu_sku="H100-SXM", contexts=[4096],
            modes=["dp_ep"], selection_mode="theoretical_only", **kw)

    assert (sweep_rt.sweep_cache_key("parallel", par(ep_load_factor=1.5))
            != sweep_rt.sweep_cache_key("parallel", par()))
    assert (sweep_rt.sweep_cache_key("parallel", par(ep_load_factor=1.0))
            == sweep_rt.sweep_cache_key("parallel", par()))
    with pytest.raises(ValueError):
        par(ep_load_factor=0.9)

    def batch(preset):
        return InferenceEstimateSweepRequest(
            model_id="x/y", gpu_sku="H100-SXM", seq_lens=[4096],
            selection_mode="theoretical_only", parallel_preset=preset)

    assert (sweep_rt.sweep_cache_key("sweep", batch(
                {"mode": "dp_ep", "degree": 8, "ep_load_factor": 1.5}))
            != sweep_rt.sweep_cache_key("sweep", batch(
                {"mode": "dp_ep", "degree": 8})))
    # An explicit balanced ρ is the omitted one: same deployment, same entry.
    assert (sweep_rt.sweep_cache_key("sweep", batch(
                {"mode": "dp_ep", "degree": 8, "ep_load_factor": 1.0}))
            == sweep_rt.sweep_cache_key("sweep", batch(
                {"mode": "dp_ep", "degree": 8})))
    assert (sweep_rt.sweep_cache_key("sweep", batch(
                {"mode": "dp_ep", "degree": 8, "ep_load_factor": "1.5"}))
            == sweep_rt.sweep_cache_key("sweep", batch(
                {"mode": "dp_ep", "degree": 8, "ep_load_factor": 1.5})))
    for bad in (0.9, "abc"):
        with pytest.raises(ValueError):
            batch({"mode": "dp_ep", "degree": 8, "ep_load_factor": bad})

    # The compute side: every point is estimated under the request's ρ,
    # beside its deployment rather than inside it.
    seen = {}
    monkeypatch.setattr(sweep_rt, "resolve_estimate_environment", lambda req: {})

    class _SeenParallel:
        @staticmethod
        def of(**kw):
            seen["parallel"] = kw["ep_load_factor"]

    class _SeenCapacity:
        @staticmethod
        def of(**kw):
            seen["preset"] = kw["parallel"]
            seen["preset_rho"] = kw["ep_load_factor"]

    monkeypatch.setattr(sweep_rt, "ParallelSweepPlan", _SeenParallel)
    monkeypatch.setattr(sweep_rt, "CapacitySweepPlan", _SeenCapacity)
    sweep_rt._parallel_plan(par(ep_load_factor=1.5))
    sweep_rt._estimate_plan(
        batch({"mode": "dp_ep", "degree": 8, "ep_load_factor": 1.5}))
    assert seen["parallel"] == 1.5
    assert seen["preset_rho"] == 1.5
    assert seen["preset"].ep_moe == 8


def test_a_preset_and_no_precision_share_one_sweep_key(isolated):
    """The sweep key holds a precision profile by its key: as shipped named
    and as shipped implied are one entry, and another profile is another."""
    inf, _warm, _ = isolated
    from backend.api._models import InferenceParallelSweepRequest
    body = _frontend_first_visitor_body(MODEL)
    plain = InferenceParallelSweepRequest(**body)
    named = InferenceParallelSweepRequest(
        **body, precision={"preset": "as_shipped"})
    fp8 = InferenceParallelSweepRequest(**body, precision={"preset": "kv_fp8"})
    key = inf._sweep_cache_key
    assert key("parallel", plain) == key("parallel", named)
    assert key("parallel", plain) != key("parallel", fp8)
