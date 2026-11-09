"""Restart drain: `/health/drain` stops spec-job admissions atomically and
lists what a restart would still interrupt, for the local supervisor only."""
import threading

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.api import _spec_preflight as preflight
from backend.spec_worker import runtime as rt

LOCAL = ("127.0.0.1", 50000)


@pytest.fixture(autouse=True)
def _clean_runtime(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setattr(rt, "_model_locks", {})
    monkeypatch.setattr(rt, "_slot_held", {})
    monkeypatch.setattr(rt, "_lock_epoch", {})
    monkeypatch.setattr(rt, "_model_procs", {})
    monkeypatch.setattr(rt, "_jobs", {})
    monkeypatch.setattr(rt, "_job_store", {})
    monkeypatch.setattr(rt, "_job_threads", {})
    monkeypatch.setattr(rt, "semaphore", threading.BoundedSemaphore(4))
    monkeypatch.setattr(rt, "_draining", False)
    monkeypatch.setattr(rt, "_persist_begin", lambda *a, **k: None)
    monkeypatch.setattr(rt, "_persist_finish", lambda *a, **k: None)
    yield


def _drain(method="post", client=LOCAL, headers=None):
    c = TestClient(app, client=client)
    return getattr(c, method)("/health/drain", headers=headers or {})


def test_job_admitted_before_drain_is_waited_on():
    assert rt.acquire("org/early") == "ok"         # admitted, still preparing
    state = _drain().json()
    assert state["draining"] is True
    assert [j["run_key"] for j in state["in_flight"]] == ["org/early"]
    rt.release("org/early")
    assert _drain("get").json()["in_flight"] == []


def test_job_admitted_after_drain_is_refused():
    _drain()
    with pytest.raises(rt.ServerDraining):
        rt.acquire("org/late")
    assert _drain("get").json()["in_flight"] == []
    assert rt.semaphore.acquire(blocking=False)    # the slot was handed back
    rt.semaphore.release()
    _drain("delete")
    assert rt.acquire("org/late") == "ok"          # admissions resume


def test_running_job_thread_is_in_flight_until_it_finishes():
    gate = threading.Event()
    assert rt.acquire("org/run") == "ok"

    def run_fn():
        gate.wait(5)
        rt.release("org/run")
        yield {"type": "done", "success": True, "message": "ok"}

    rt.start_job("org/run", run_fn, model_id="org/run", kind="generate")
    _drain()
    inflight = _drain("get").json()["in_flight"]
    assert [(j["run_key"], j["kind"]) for j in inflight] == [("org/run", "generate")]
    gate.set()
    for _ in range(100):
        if not rt.in_flight():
            break
        threading.Event().wait(0.05)
    assert rt.in_flight() == []


def test_admission_during_drain_answers_503(host_credentials, monkeypatch):
    """Through a real route: the claim raises, the app answers 503."""
    monkeypatch.setattr(preflight, "agent_cli_refusal", lambda: None)
    monkeypatch.setattr(preflight, "probe_repo", lambda mid: ({}, None))
    monkeypatch.setattr(preflight, "known_spec", lambda *a, **k: None)
    _drain()
    r = TestClient(app).post("/generate-spec", json={"model_id": "org/new"})
    assert r.status_code == 503
    assert r.json()["status"] == "restarting"
    assert "restarting" in r.json()["message"].lower()


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
def test_loopback_supervisor_is_served(host):
    assert _drain("get", client=(host, 50000)).status_code == 200


def test_tunnel_relayed_or_remote_caller_is_refused():
    assert _drain(headers={"Cf-Connecting-Ip": "1.2.3.4"}).status_code == 404
    assert _drain(headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 404
    assert _drain(client=("8.8.8.8", 50000)).status_code == 404
    assert rt.drain_state()["draining"] is False
