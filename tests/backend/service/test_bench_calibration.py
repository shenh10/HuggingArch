"""Estimates read Kernel Bench, end to end (docs/measured_calibration.md
§7): the HA API resolves the measurement scope (the caller as its owner
with `calibrate_with_mine`, the lake's snapshot), the estimate looks its
physical ops up in the Kernel Bench service — here in process, over a local
lake — and its per-op rows carry what was measured. A new upload is used by
the very next estimate; one user's rows never calibrate another's estimate;
the lookup path gives the numbers the corpus files gave; the service down is
a 503 for estimates and for /api/bench/*, and nothing else."""
from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

import tests.corpus_row as _corpus
from backend.measure import lookup as measurements
from backend.inference.framework_bindings import DEFAULT_FRAMEWORK
from backend.service import bench_proxy

from .conftest import BENCH_SECRET, as_user, make_user_jwt

ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
ADMIN = "33333333-3333-4333-8333-333333333333"
MODEL = "meta-llama/Llama-3.2-1B"
_BODY = {"model_id": MODEL, "gpu_sku": "H100-SXM", "avg_input_tokens": 1024,
         "avg_output_tokens": 128}


def _jwt(sub: str) -> dict:
    return {"Authorization": f"Bearer {make_user_jwt(sub=sub, email=f'{sub[:4]}@example.com')}"}


@pytest.fixture
def wired(monkeypatch, bench_app, bench_lake):
    """The HA API's estimator client and its proxy, both to the service in
    process."""
    monkeypatch.setenv("HUGGINGARCH_BENCH_URL", "http://bench.internal")
    measurements.forget_rows()
    monkeypatch.setattr(measurements, "_clients", {os.getpid(): TestClient(
        bench_app, base_url="http://bench.internal", client=("127.0.0.1", 1))})
    monkeypatch.setattr(bench_proxy, "_clients", type(bench_proxy._clients)())
    monkeypatch.setattr(bench_proxy, "transport", httpx.ASGITransport(app=bench_app))
    return TestClient(bench_app, client=("127.0.0.1", 1),
                      headers={"X-HA-Internal": BENCH_SECRET})


def _estimate(client, headers=None, **body) -> dict:
    r = client.post("/inference/estimate", json={**_BODY, **body}, headers=headers or {})
    assert r.status_code == 200, r.text
    return r.json()


def _measured(est: dict) -> dict:
    """aggregation_id → (t_measured_s, origin, contributor) of the prefill
    rows a measurement was selected for (a zero-cost row is measured at 0
    with no measurement behind it)."""
    return {r["aggregation_id"]: (r["t_measured_s"], r["source_badge"].get("origin"),
                                  r["source_badge"].get("contributor"))
            for r in est["per_op_timing"]
            if r.get("source") in ("measured", "measured_anomalous")}


@pytest.fixture(scope="module")
def measurable_rows():
    """Two of the model's prefill per-op rows as measured CSV rows (the
    estimate's own identity bytes), the default framework's."""
    from tests.snapshots import resolve_snapshot, snapshot_config
    from backend.inference.capacity import estimate_capacity
    est = estimate_capacity(config=snapshot_config(MODEL), resolved=resolve_snapshot(MODEL),
                            gpu_sku="H100-SXM", model_id=MODEL, avg_input_tokens=1024,
                            avg_output_tokens=128, selection_mode="latest")
    rows = [r for r in est["per_op_timing"] if r.get("shape_json")][:2]
    return [{"aggregation_id": r["aggregation_id"],
             "row": _corpus.row(op=r["op"], shape_json=r["shape_json"],
                                dtype_env_json=r["dtype_env_json"], gpu="H100-SXM",
                                phase="prefill", framework=DEFAULT_FRAMEWORK,
                                kernel_impl="kb_kernel")} for r in rows]


def _upload(service, user, rows, *, public=False):
    r = service.post("/api/bench/upload", headers=as_user(user, admin=public),
                     json={"csv_text": _corpus.to_csv(rows), "make_public": public})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_new_upload_is_used_by_the_very_next_estimate(client, wired, measurable_rows):
    (a, b) = measurable_rows
    mine = {"calibrate_with_mine": True}
    assert _measured(_estimate(client, _jwt(ALICE), **mine)) == {}
    _upload(wired, ALICE, [{**a["row"], "latency_us": "11.0", "contributor": "alice"}])
    got = _measured(_estimate(client, _jwt(ALICE), **mine))
    assert got[a["aggregation_id"]][1:] == ("mine", "alice")
    # Bob, byte-identical body; anyone not asking: not hers.
    assert _measured(_estimate(client, _jwt(BOB), **mine)) == {}
    assert _measured(_estimate(client, _jwt(ALICE))) == {}
    assert _measured(_estimate(client)) == {}
    # A shared row is everyone's, from the next estimate on.
    _upload(wired, ADMIN, [{**b["row"], "latency_us": "22.0", "contributor": "pub"}], public=True)
    for headers, body in (({}, {}), (_jwt(BOB), mine), (_jwt(ALICE), mine)):
        assert _measured(_estimate(client, headers, **body))[b["aggregation_id"]][1:] == (
            "shared", "pub")


def test_the_response_says_what_it_read(client, wired, measurable_rows, monkeypatch):
    est = _estimate(client)
    assert est["measurements"] == "kernel_bench" and len(est["measurement_scope"]) == 16
    theory = _estimate(client, prefer_measured=False)
    assert theory["measurements"] == "none"
    monkeypatch.delenv("HUGGINGARCH_BENCH_URL")
    assert _estimate(client)["measurements"] == "none"


def test_the_lookup_gives_the_corpus_files_numbers(client, wired, measurable_rows, monkeypatch):
    """The same rows read from Kernel Bench and posted with the request (the
    corpus files' path: validated, indexed by their own projection): the
    same measured rows, to the bit."""
    rows = [{**m["row"], "latency_us": lat} for m, lat in zip(measurable_rows, ("13.0", "29.0"),
                                                               strict=True)]
    _upload(wired, ADMIN, rows, public=True)
    looked_up = _estimate(client)
    monkeypatch.delenv("HUGGINGARCH_BENCH_URL")
    posted = _estimate(client, calibration_rows=rows)
    # Posted rows are the caller's own (`origin` mine); everything else is
    # the same.
    def unowned(value):
        if isinstance(value, dict):
            return {k: unowned(v) for k, v in value.items() if k != "origin"}
        if isinstance(value, list):
            return [unowned(v) for v in value]
        return value
    a, b = _measured(looked_up), _measured(posted)
    assert len(a) >= 2 and {k: (t, c) for k, (t, _o, c) in a.items()} == {
        k: (t, c) for k, (t, _o, c) in b.items()}
    for key in ("per_op_timing", "per_op_timing_decode", "phases"):
        assert unowned(looked_up[key]) == unowned(posted[key]), key


def test_the_service_down_is_503_for_estimates_and_bench_only(client, wired, bench_gate):
    client.get("/health")
    _estimate(client)                                   # the lake serves
    bench_gate.phase, bench_gate.reason = "unavailable", "not the authoritative lake"
    r = client.post("/inference/estimate", json=_BODY)
    assert r.status_code == 503 and r.json()["reason"] == "measurements_unavailable"
    assert "not the authoritative lake" in r.json()["detail"]
    assert client.get("/api/bench/browse").status_code == 503
    assert _estimate(client, prefer_measured=False)["measurements"] == "none"
    assert client.get("/inference/gpus").status_code == 200
    assert client.get("/health").status_code == 200


def test_uploads_and_promotions_never_move_a_theory_key(wired, monkeypatch, measurable_rows):
    """A theory-only result reads no measurements, so no key it is cached or
    joined under carries a scope element: a private upload, a public upload
        and a promotion leave every theory key — the estimate's single-flight,
        both sweeps' cache keys, the gallery's SOL entry — as it
        was, while the measured keys of the same requests (including a
        speculative Gallery view's K-aware fold) move
    (docs/measured_calibration.md §7.1)."""
    import asyncio

    from backend.api import _access
    from backend.api import sweep_runtime as S
    from backend.api.gallery_frontier import FrontierView, _Stored, _view_scope
    from backend.api._models import (InferenceEstimateRequest, InferenceEstimateSweepRequest,
                                     InferenceParallelSweepRequest)
    from backend.api.offload import call_key
    from backend.api.routes import inference as R

    monkeypatch.setattr(S, "resolved_spec_for", lambda m: object())
    monkeypatch.setattr(S, "result_version", lambda r: "rv")

    def keys(**mode):
        est = asyncio.run(_access.with_measurement_scope(
            InferenceEstimateRequest(**_BODY, **mode)))
        out = [call_key(R._estimate_payload, (est,), {})]
        for kind, req in (("sweep", InferenceEstimateSweepRequest(**_BODY, seq_lens=[1024], **mode)),
                          ("parallel", InferenceParallelSweepRequest(
                              **_BODY, modes=["tp_tp"], context_len=1024, **mode))):
            req = asyncio.run(_access.with_measurement_scope(req))
            # A measured answer has no cache key (made live); its join does.
            key = S.sweep_cache_key(kind, req)
            out += ([] if key is None else [key]) + [call_key(S._run_sweep, (req,), {})]
        return out

    def frontier_keys():
        # The frontier entry is theory-only: its points name the pool records
        # where each view's measured keys live. A speculative drawing is now a
        # measured consumer and is checked separately below.
        return [_Stored(MODEL, "rv").entry_key(FrontierView("B200"))]

    def speculative_key():
        view = FrontierView("B200", draft_tokens=1)
        return _Stored(MODEL, "rv").view_key(view, *_view_scope(view))

    frontier0 = frontier_keys()
    speculative0 = speculative_key()
    theory0, measured0 = keys(prefer_measured=False), keys()
    (a, b) = measurable_rows
    _upload(wired, ALICE, [{**a["row"], "contributor": "alice"}])            # private
    up = _upload(wired, ALICE, [{**b["row"], "contributor": "alice", "gpu": "H200"}])
    assert wired.post(f"/api/bench/{up['upload_id']}/contribute",
                      headers=as_user(ALICE)).status_code == 200
    assert wired.post("/api/bench/promote", headers=as_user(ADMIN, admin=True), json={
        "sets": [{"owner": ALICE, "kind": "upload", "id": up["upload_id"]}]}).status_code == 200
    _upload(wired, ADMIN, [{**b["row"], "contributor": "pub"}], public=True)  # public
    assert keys(prefer_measured=False) == theory0
    assert frontier_keys() == frontier0
    assert speculative_key() != speculative0
    measured1 = keys()
    assert all(x != y for x, y in zip(measured1, measured0, strict=True))
