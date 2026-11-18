"""The measurement scope at the API (docs/measured_calibration.md §7.1):
resolved in the server process — the owner from the request's credentials
when it asks `calibrate_with_mine`, the lake's version from the Kernel Bench
service — before anything is offloaded, joined or keyed; the client can
never name it; it separates every identity a measured result goes under;
theory reads no measurements and asks the service nothing. The owner
resolver is the service layer's (`backend.service.ha_api`); here a
stand-in, and the service a stand-in transport, so these tests hold for the
core alone."""
from __future__ import annotations

import os

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.api import _access
from backend.api import sweep_runtime as _sweeps
from backend.api._models import (
    CALIBRATION_ROWS_MAX,
    InferenceEstimateRequest,
    InferenceEstimateSweepRequest,
    InferenceParallelSweepRequest,
)
from backend.api.offload import call_key
from backend.api.routes import inference as _routes
from backend.measure import lookup as measurements
from backend.measure.scope import NONE, MeasurementScope

ALICE, BOB = "a" * 8 + "-1111-4111-8111-111111111111", "b" * 8 + "-2222-4222-8222-222222222222"
_BODY = {"model_id": "m/x", "gpu_sku": "H100-SXM", "calibrate_with_mine": True}


def _scope(owner, snapshot=7, shared=5):
    return MeasurementScope(owner=owner, lake_id="lake-1", snapshot=snapshot,
                            shared_snapshot=shared, rules_rev="r1",
                            snapshot_content=f"t{snapshot}", shared_snapshot_content=f"t{shared}")


def _resolver(authorization, cookie):
    """Bearer `alice` / `bob`; none: no session."""
    if authorization is None and cookie is None:
        return None
    who = (authorization or "").removeprefix("Bearer ") or cookie
    if who == "expired":
        raise HTTPException(status_code=401, detail="Session expired")
    return {"alice": ALICE, "bob": BOB}[who]


@pytest.fixture
def client():
    from backend.api import app
    return TestClient(app)


@pytest.fixture
def resolver(monkeypatch):
    monkeypatch.setattr(_access, "_owner_resolver", _resolver)


@pytest.fixture
def service(monkeypatch):
    """A stand-in Kernel Bench service answering `/internal/bench/version`;
    records what was asked."""
    asked: list = []

    def handler(request):
        asked.append(request.url.path)
        assert request.headers["x-ha-internal"] == "s3cret"
        return httpx.Response(200, json={"lake_id": "lake-1", "snapshot": 7,
                                         "shared_snapshot": 5, "rules_rev": "r1",
                                         "snapshot_content": "t7", "shared_snapshot_content": "t5"})
    monkeypatch.setenv("HUGGINGARCH_BENCH_URL", "http://bench.internal")
    monkeypatch.setenv("HUGGINGARCH_BENCH_SECRET", "s3cret")
    monkeypatch.setattr(measurements, "_clients", {
        os.getpid(): httpx.Client(transport=httpx.MockTransport(handler))})
    return asked


@pytest.fixture
def seen(monkeypatch):
    """What the offloaded estimate received."""
    got: list = []

    def payload(req):
        got.append(req)
        return {"scope": req.scope().to_json()}
    monkeypatch.setattr(_routes, "_estimate_payload", payload)
    return got


def test_without_the_service_layer_it_is_501(client, monkeypatch, seen):
    monkeypatch.setattr(_access, "_owner_resolver", None)
    r = client.post("/inference/estimate", json=_BODY, headers={"Authorization": "Bearer alice"})
    assert r.status_code == 501 and "needs the service" in r.json()["detail"]
    assert seen == []


def test_without_a_session_it_is_401(client, resolver, seen):
    r = client.post("/inference/estimate", json=_BODY)
    assert r.status_code == 401 and "sign in" in r.json()["detail"]
    r = client.post("/inference/estimate", json=_BODY, headers={"Authorization": "Bearer expired"})
    assert r.status_code == 401
    assert seen == []


def test_the_scope_is_resolved_before_the_work(client, resolver, service, seen):
    for who, owner in (("alice", ALICE), ("bob", BOB)):
        r = client.post("/inference/estimate", json=_BODY,
                        headers={"Authorization": f"Bearer {who}"})
        assert r.status_code == 200, r.text
        assert r.json()["scope"] == _scope(owner).to_json()
    # Not asked: the shared rows, whoever calls.
    r = client.post("/inference/estimate", json={**_BODY, "calibrate_with_mine": False},
                    headers={"Authorization": "Bearer alice"})
    assert r.json()["scope"] == _scope(None).to_json()
    assert service == ["/internal/bench/version"] * 3


def test_theory_asks_the_service_nothing(client, resolver, service, seen):
    for body in ({**_BODY, "prefer_measured": False},
                 {**_BODY, "calibrate_with_mine": False, "selection_mode": "theoretical_only"}):
        r = client.post("/inference/estimate", json=body, headers={"Authorization": "Bearer alice"})
        assert r.status_code == 200 and r.json()["scope"] == NONE.to_json(), r.text
    assert service == []


def test_no_service_configured_reads_none(client, resolver, seen, monkeypatch):
    monkeypatch.delenv("HUGGINGARCH_BENCH_URL", raising=False)
    r = client.post("/inference/estimate", json=_BODY, headers={"Authorization": "Bearer alice"})
    assert r.status_code == 200 and r.json()["scope"] == NONE.to_json()


def test_the_service_down_is_503_and_nothing_runs(client, resolver, seen, monkeypatch):
    def down(request):
        raise httpx.ConnectError("refused")
    monkeypatch.setenv("HUGGINGARCH_BENCH_URL", "http://bench.internal")
    monkeypatch.setattr(measurements, "_clients", {
        os.getpid(): httpx.Client(transport=httpx.MockTransport(down))})
    r = client.post("/inference/estimate", json={**_BODY, "calibrate_with_mine": False})
    assert r.status_code == 503 and r.json()["reason"] == "measurements_unavailable"
    assert "measurements unavailable" in r.json()["detail"]
    assert seen == []
    # Theory still answers.
    r = client.post("/inference/estimate",
                    json={**_BODY, "calibrate_with_mine": False, "prefer_measured": False})
    assert r.status_code == 200


def test_a_body_cannot_name_a_scope(client, resolver, seen):
    for spelled in ({"owner": ALICE, "lake_id": "x", "snapshot": 1}, None, "alice"):
        r = client.post("/inference/estimate", json={**_BODY, "measurement_scope": spelled},
                        headers={"Authorization": "Bearer bob"})
        assert r.status_code == 422, r.text
    assert seen == []
    assert "measurement_scope" not in InferenceEstimateRequest.model_json_schema()["properties"]
    req = InferenceEstimateRequest(gpu_sku="H100").model_copy(
        update={"measurement_scope": _scope(ALICE)})
    assert "measurement_scope" not in req.model_dump() and "measurement_scope" not in req.model_dump_json()


def test_posted_rows_are_bounded(client, seen):
    rows = [{}] * (CALIBRATION_ROWS_MAX + 1)
    r = client.post("/inference/estimate", json={**_BODY, "calibrate_with_mine": False,
                                                 "calibration_rows": rows})
    assert r.status_code == 422 and "20000" in r.text
    assert seen == []


def _scoped(cls, scope, **extra):
    return cls(**_BODY, **extra).model_copy(update={"measurement_scope": scope})


def test_two_callers_with_identical_bodies_are_two_calls():
    a, b = (_scoped(InferenceEstimateRequest, _scope(o)) for o in (ALICE, BOB))
    assert a.model_dump() == b.model_dump()
    assert call_key(_routes._estimate_payload, (a,), {}) != call_key(_routes._estimate_payload, (b,), {})
    same = _scoped(InferenceEstimateRequest, _scope(ALICE))
    assert call_key(_routes._estimate_payload, (a,), {}) == call_key(_routes._estimate_payload, (same,), {})


@pytest.fixture
def keyable(monkeypatch):
    """Sweep keys for a model id without resolving a real spec."""
    monkeypatch.setattr(_sweeps, "resolved_spec_for", lambda mid: object())
    monkeypatch.setattr(_sweeps, "result_version", lambda resolved: "rv")


@pytest.mark.parametrize("kind, cls, extra", [
    ("sweep", InferenceEstimateSweepRequest, {"seq_lens": [1024]}),
    ("parallel", InferenceParallelSweepRequest, {"modes": ["tp_tp"], "contexts": [1024],
                                                 "gpu_skus": ["H100-SXM"]}),
])
def test_sweep_answers_with_rows_are_never_stored_and_joins_separate_callers(
        keyable, kind, cls, extra):
    """A sweep that reads anyone's rows is made live, never stored — so no
    caller's rows reach a cache another reads; in flight, each caller joins
    only their own. The SOL answer is one entry for every caller."""
    a, b, anon = (_scoped(cls, _scope(o), **extra) for o in (ALICE, BOB, None))
    assert {_sweeps.sweep_cache_key(kind, r) for r in (a, b, anon)} == {None}
    assert _sweeps.sweep_cache_key(kind, cls(**{**_BODY, "calibrate_with_mine": False},
                                             calibration_rows=[{}], **extra)) is None
    assert call_key(_sweeps._run_sweep, (a,), {}) != call_key(_sweeps._run_sweep, (b,), {})
    assert call_key(_sweeps._run_sweep, (a,), {}) == call_key(
        _sweeps._run_sweep, (_scoped(cls, _scope(ALICE), **extra),), {})
    sol = {_sweeps.sweep_cache_key(kind, r.model_copy(update={"sol_only": True}))
           for r in (a, b, anon)}
    assert len(sol) == 1 and None not in sol


@pytest.mark.parametrize("kind, cls, extra", [
    ("sweep", InferenceEstimateSweepRequest, {"seq_lens": [1024]}),
    ("parallel", InferenceParallelSweepRequest, {"modes": ["tp_tp"], "contexts": [1024],
                                                 "gpu_skus": ["H100-SXM"]}),
])
def test_theory_keys_carry_no_scope(keyable, kind, cls, extra):
    theory = {**extra, "prefer_measured": False}
    keys = {_sweeps.sweep_cache_key(kind, _scoped(cls, s, **theory))
            for s in (NONE, _scope(None), _scope(ALICE))}
    assert len(keys) == 1
    plain = cls(**_BODY, **theory)          # no scope set: byte-stable
    assert _sweeps.sweep_cache_key(kind, plain) in keys


@pytest.mark.parametrize("path, name, extra", [
    ("/inference/estimate-sweep", "estimate_sweep_response", {"seq_lens": [1024]}),
    ("/inference/estimate-parallel-sweep", "parallel_sweep_response",
     {"modes": ["tp_tp"], "contexts": [1024], "gpu_skus": ["H100-SXM"]}),
])
def test_both_sweeps_resolve_before_the_work(client, resolver, service, monkeypatch,
                                             path, name, extra):
    got: list = []

    async def respond(req):
        got.append(req)
        return {"owner": req.scope().owner}
    monkeypatch.setattr(_routes, name, respond)
    r = client.post(path, json={**_BODY, **extra}, headers={"Authorization": "Bearer alice"})
    assert r.status_code == 200 and r.json() == {"owner": ALICE}, r.text
    assert client.post(path, json={**_BODY, **extra}).status_code == 401
    monkeypatch.setattr(_access, "_owner_resolver", None)
    assert client.post(path, json={**_BODY, **extra},
                       headers={"Authorization": "Bearer alice"}).status_code == 501
    assert len(got) == 1
