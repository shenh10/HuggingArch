"""Auth + dispatch for the agent-review routes (user + admin)."""
from __future__ import annotations

import pytest

import backend.spec_worker as _sw
from backend.service import spec_generations as _gens
from backend.service import agent_credentials as _creds
from .conftest import mock_admins_no, mock_admins_yes


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


def _stub_job(monkeypatch):
    monkeypatch.setattr(_sw, "acquire", lambda k: "ok")
    monkeypatch.setattr(_sw, "start_job", lambda k, fn, **kw: None)


def test_user_review_anonymous_401(client):
    r = client.post("/api/account/generations/rec-1/review")
    assert r.status_code == 401


def test_user_review_not_owner_403(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "get", lambda rid: {"id": rid, "model_id": "m", "user_id": "someone-else"})
    r = client.post("/api/account/generations/rec-1/review", headers=_auth(user_token))
    assert r.status_code == 403


def test_user_review_missing_record_404(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "get", lambda rid: None)
    r = client.post("/api/account/generations/rec-1/review", headers=_auth(user_token))
    assert r.status_code == 404


def test_user_review_no_credentials(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "get", lambda rid: {"id": rid, "model_id": "m", "user_id": "user-uuid-1"})
    monkeypatch.setattr(_creds, "resolve_env", lambda uid, **kw: None)
    r = client.post("/api/account/generations/rec-1/review", headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["status"] == "no_credentials"


def test_user_review_started(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "get", lambda rid: {"id": rid, "model_id": "m", "user_id": "user-uuid-1",
                                                   "spec_yaml": "x", "new_component_yamls": []})
    monkeypatch.setattr(_creds, "resolve_env", lambda uid, **kw: {"agent_cli": "claude", "env": {"K": "v"}})
    _stub_job(monkeypatch)
    r = client.post("/api/account/generations/rec-1/review", headers=_auth(user_token))
    body = r.json()
    assert r.status_code == 200 and body["status"] == "started"
    assert body["run_key"] == "review|user-uuid-1|rec-1"


def test_user_review_prep_reports_incomplete_bundle(client, user_token, monkeypatch):
    record = {"id": "rec-1", "model_id": "meta/Foo", "user_id": "user-uuid-1",
              "spec_yaml": "match: {model_id: meta/Foo}\ndrafts: [missing]\n",
              "new_component_yamls": []}
    monkeypatch.setattr(_gens, "get", lambda rid: record)
    monkeypatch.setattr(_creds, "resolve_env",
                        lambda uid, **kw: {"agent_cli": "claude", "env": {}})
    monkeypatch.setattr(_sw, "acquire", lambda k: "ok")
    jobs = []
    monkeypatch.setattr(_sw, "start_job", lambda k, fn, **kw: jobs.append(fn))

    r = client.post("/api/account/generations/rec-1/review",
                    headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["status"] == "started"
    events = list(jobs[0]())
    done = events[-1]
    assert done["type"] == "done" and done["success"] is False
    assert done["error_kind"] == "spec_bundle_incomplete"
    assert "draft component 'missing'" in done["message"]


def test_user_review_prep_reports_unmigratable_as_spec_invalid(
        client, user_token, monkeypatch):
    record = {"id": "rec-1", "model_id": "meta/Foo", "user_id": "user-uuid-1",
              "spec_yaml": (
                  "role: attention\nkv_cached: [k, v]\n"
                  "graph: {x: {op: identity, in: [x]}}\n"
              ),
              "new_component_yamls": []}
    monkeypatch.setattr(_gens, "get", lambda rid: record)
    monkeypatch.setattr(_creds, "resolve_env",
                        lambda uid, **kw: {"agent_cli": "claude", "env": {}})
    monkeypatch.setattr(_sw, "acquire", lambda k: "ok")
    jobs = []
    monkeypatch.setattr(_sw, "start_job", lambda k, fn, **kw: jobs.append(fn))

    r = client.post("/api/account/generations/rec-1/review",
                    headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["status"] == "started"
    events = list(jobs[0]())
    done = events[-1]
    assert done["type"] == "done" and done["success"] is False
    assert done["error_kind"] == "spec_invalid"
    assert "errors" in done


def test_user_review_quota_exceeded(client, user_token, monkeypatch):
    monkeypatch.setattr(_gens, "get", lambda rid: {"id": rid, "model_id": "m", "user_id": "user-uuid-1",
                                                   "spec_yaml": "x", "new_component_yamls": []})
    monkeypatch.setattr(_creds, "resolve_env", lambda uid, **kw: {"agent_cli": "claude", "env": {"K": "v"}})
    import backend.service.routes.spec as _spec
    monkeypatch.setattr(_spec, "_quota_ok", lambda uid: False)  # over the cap
    # acquire must NOT be reached when the quota gate rejects
    monkeypatch.setattr(_sw, "acquire", lambda k: (_ for _ in ()).throw(AssertionError("acquire before quota")))
    r = client.post("/api/account/generations/rec-1/review", headers=_auth(user_token))
    assert r.status_code == 200 and r.json()["status"] == "quota_exceeded"


def test_admin_review_non_admin_403(client, user_token, monkeypatch, httpx_mock):
    mock_admins_no(httpx_mock, "user-uuid-1")
    r = client.post("/api/admin/spec-generations/rec-1/review", headers=_auth(user_token))
    assert r.status_code == 403


def test_admin_review_started_host_fallback(host_credentials, client, admin_token, monkeypatch, httpx_mock):
    mock_admins_yes(httpx_mock, "admin-uuid-1")
    monkeypatch.setattr(_gens, "get", lambda rid: {"id": rid, "model_id": "m", "user_id": "someone",
                                                   "spec_yaml": "x", "new_component_yamls": []})
    monkeypatch.setattr(_creds, "resolve_env", lambda uid, **kw: None)  # admin → host fallback allowed
    _stub_job(monkeypatch)
    r = client.post("/api/admin/spec-generations/rec-1/review", headers=_auth(admin_token))
    body = r.json()
    assert r.status_code == 200 and body["status"] == "started"
    assert body["run_key"] == "review|admin-uuid-1|rec-1"


def test_review_status_builds_scoped_key(client, user_token, monkeypatch):
    seen = {}
    monkeypatch.setattr(_sw, "job_status", lambda k, c: seen.setdefault("key", k) or {"state": "idle"})
    r = client.get("/api/spec/review-status?record_id=rec-9&cursor=0", headers=_auth(user_token))
    assert r.status_code == 200
    assert seen["key"] == "review|user-uuid-1|rec-9"


def test_review_and_fix_prep_report_unavailable_forward_source(monkeypatch):
    """A prep that cannot obtain the modeling source ends the job with the
    typed error (kind + retryable) and frees the slot; no agent starts."""
    import backend.service.routes.spec as _spec
    from backend.spec_worker import record_jobs
    record = {"id": "rec-1", "model_id": "org/m", "spec_yaml": "x",
              "new_component_yamls": []}
    monkeypatch.setattr(_sw, "acquire", lambda k: "ok")
    released = []
    monkeypatch.setattr(_sw, "release", released.append)
    jobs = []
    monkeypatch.setattr(_sw, "start_job", lambda k, fn, **kw: jobs.append(fn))
    monkeypatch.setattr(record_jobs, "record_bundle", lambda *a: None)
    monkeypatch.setattr(record_jobs, "fix_seed_bundle", lambda record: None)

    def unavailable(*a, **k):
        raise _sw.ForwardSourceUnavailable("no modeling source", retryable=False)
    monkeypatch.setattr(_sw, "prepare_review", unavailable)
    monkeypatch.setattr(_sw, "prepare_fix", unavailable)
    monkeypatch.setattr(_sw, "stream_review", lambda *a, **k: pytest.fail("agent ran"))
    monkeypatch.setattr(_sw, "stream_fix", lambda *a, **k: pytest.fail("agent ran"))

    _spec.launch_review(record, reviewer_id="u", reviewer_role="admin")
    _spec.launch_fix(record, fixer_id="u", fixer_role="admin", review_report="r")
    for job in jobs:
        done = list(job())[-1]
        assert done["success"] is False
        assert done["error_kind"] == "forward_source_missing"
        assert done["retryable"] is False
    assert released == ["review|u|rec-1", "fix|u|rec-1"]
