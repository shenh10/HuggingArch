"""Every spec-run route refuses with the same body for the same reason.

Four routes start a generate/refine run — the core `/generate-spec` and
`/refine-spec` (anonymous, single-user deployments) and the signed-in
`/api/spec/generate` and `/api/spec/refine`. Each case drives one route's
handler into one refusal and pins the exact body. The bodies are literal, so
a shared pre-flight that drifted from any route's old answer shows up here.
"""
from __future__ import annotations

import asyncio

import pytest

ROUTES = ["core_generate", "core_refine", "service_generate", "service_refine"]
CORE = ROUTES[:2]
SERVICE = ROUTES[2:]
MODEL = "org/m"


@pytest.fixture
def world(monkeypatch, tmp_path):
    """A run that would start: agent on PATH, a stored key, no HF config
    (so no `cached` short-circuit), a free lock. Cases break one piece."""
    from backend import spec_worker
    from backend.service import agent_credentials as creds
    from backend.service import auth
    from backend.service.routes import spec as service_spec

    started = []
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    # The host's own claude credential: the core routes and an admin with no
    # stored key run on it.
    monkeypatch.setenv("SPEC_AGENT_CLI", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-key")
    monkeypatch.setattr(spec_worker, "get_agent_binary", lambda: "claude")
    monkeypatch.setattr("shutil.which", lambda b: f"/usr/bin/{b}")
    monkeypatch.setattr(spec_worker, "acquire", lambda key: "ok")
    monkeypatch.setattr(spec_worker, "release", lambda key: None)
    monkeypatch.setattr(spec_worker, "start_job", lambda key, fn, **kw: started.append(key))
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **k: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(creds, "resolve_env",
                        lambda uid, **kw: {"agent_cli": "claude", "env": {"K": "v"}})
    monkeypatch.setattr(auth, "is_admin", lambda uid: False)
    monkeypatch.setattr(service_spec, "_peruser_active", lambda: False)
    monkeypatch.setattr(service_spec, "_quota_hits", {})
    monkeypatch.setattr("backend.spec_worker.overlay.load_seed_spec_yaml",
                        lambda mid: "name: seeded\n")
    return started


def _status_body(out):
    """Handlers return a dict on 200, or a JSONResponse on 4xx refusals."""
    if hasattr(out, "body") and hasattr(out, "status_code"):
        import json
        return out.status_code, json.loads(out.body)
    return 200, out


def _call(route: str, monkeypatch):
    from backend.api._models import RefineSpecRequest
    refine = RefineSpecRequest(model_id=MODEL, spec_storage_bytes=1,
                               hf_storage_bytes=2, storage_diff=1)
    if route in CORE:
        # The core routes serve only a deployment without user auth.
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        from backend.api.routes import spec as core
        from backend.api._models import GenerateSpecRequest
        if route == "core_generate":
            return asyncio.run(core.generate_spec(GenerateSpecRequest(model_id=MODEL)))
        return asyncio.run(core.refine_spec(refine))
    from backend.service.routes import spec as service
    if route == "service_generate":
        return service.service_generate_spec(service.GenerateSpecRequest(model_id=MODEL),
                                             user={"sub": "u"})
    return service.service_refine_spec(refine, user={"sub": "u"})


@pytest.mark.parametrize("route", ROUTES)
def test_the_run_starts(world, monkeypatch, route):
    out = _call(route, monkeypatch)
    assert out["status"] == "started"
    want = f"refine|{MODEL}" if "refine" in route else MODEL
    assert world == [want]


@pytest.mark.parametrize("route", ROUTES)
def test_invalid_agent_configuration(world, monkeypatch, route):
    from backend import spec_worker

    def bad():
        raise ValueError("no such agent")
    monkeypatch.setattr(spec_worker, "get_agent_binary", bad)
    assert _call(route, monkeypatch) == {
        "success": False,
        "message": "Invalid coding agent configuration: no such agent"}
    assert world == []


@pytest.mark.parametrize("route", ROUTES)
def test_agent_cli_not_installed(world, monkeypatch, route):
    monkeypatch.setattr("shutil.which", lambda b: None)
    assert _call(route, monkeypatch) == {
        "success": False, "message": "Coding agent CLI not installed"}
    assert world == []


@pytest.mark.parametrize("route", ROUTES)
def test_lock_duplicate(world, monkeypatch, route):
    from backend import spec_worker
    monkeypatch.setattr(spec_worker, "acquire", lambda key: "duplicate")
    assert _call(route, monkeypatch) == {
        "success": False, "status": "duplicate",
        "message": f"Already analyzing {MODEL}"}
    assert world == []


@pytest.mark.parametrize("route", ROUTES)
def test_lock_overloaded(world, monkeypatch, route):
    from backend import spec_worker
    monkeypatch.setattr(spec_worker, "acquire", lambda key: "overloaded")
    assert _call(route, monkeypatch) == {
        "success": False, "status": "overloaded",
        "message": "Too many concurrent tasks, please retry later"}
    assert world == []


@pytest.mark.parametrize("route", ["core_refine", "service_refine"])
def test_refine_without_seed_is_refused(world, monkeypatch, route):
    monkeypatch.setattr("backend.spec_worker.overlay.load_seed_spec_yaml",
                        lambda mid: None)
    status, body = _status_body(_call(route, monkeypatch))
    assert status == 409
    assert body["success"] is False
    assert body["status"] == "no_spec"
    assert "generate" in body["message"].lower()
    assert world == []


def _hf_error(kind):
    import httpx2  # the hub's HTTP client; its errors subclass httpx2.HTTPError
    if issubclass(kind, httpx2.HTTPError):
        resp = httpx2.Response(404, request=httpx2.Request("GET", "https://hf.co"))
        return kind("probe", response=resp)
    return kind("probe")


@pytest.mark.parametrize("route", ["core_generate", "service_generate"])
def test_a_missing_repo_does_not_start_the_agent(world, monkeypatch, route):
    from huggingface_hub import errors as hf_errors
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **k: (_ for _ in ()).throw(_hf_error(hf_errors.RepositoryNotFoundError)))
    out = _call(route, monkeypatch)
    assert out["success"] is False
    assert out["status"] == "repo_unavailable"
    assert "does not exist on Hugging Face" in out["message"]
    assert world == []


@pytest.mark.parametrize("route", ["core_generate", "service_generate"])
def test_a_gated_repo_does_not_start_the_agent(world, monkeypatch, route):
    from huggingface_hub import errors as hf_errors
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **k: (_ for _ in ()).throw(_hf_error(hf_errors.GatedRepoError)))
    out = _call(route, monkeypatch)
    assert out["success"] is False
    assert out["status"] == "repo_unavailable"
    assert "gated" in out["message"]
    assert "does not exist" not in out["message"]
    assert world == []


@pytest.mark.parametrize("route", ["core_generate", "service_generate"])
def test_an_invalid_repo_id_does_not_start_the_agent(world, monkeypatch, route):
    from huggingface_hub import errors as hf_errors
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **k: (_ for _ in ()).throw(hf_errors.HFValidationError("bad id")))
    out = _call(route, monkeypatch)
    assert out["success"] is False
    assert out["status"] == "repo_unavailable"
    assert "not a valid Hugging Face repo id" in out["message"]
    assert world == []


@pytest.mark.parametrize("route", ["core_generate", "service_generate"])
def test_a_repo_without_config_json_does_not_start_the_agent(world, monkeypatch, route):
    from huggingface_hub import errors as hf_errors
    calls = []

    def _download(**k):
        calls.append(k)
        raise _hf_error(hf_errors.EntryNotFoundError)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    out = _call(route, monkeypatch)
    assert out["success"] is False
    assert out["status"] == "repo_unavailable"
    assert "no config.json" in out["message"]
    assert world == []
    assert len(calls) == 1


@pytest.mark.parametrize("route, verb", [("service_generate", "generate"),
                                         ("service_refine", "refine")])
def test_no_credentials(world, monkeypatch, route, verb):
    from backend.service import agent_credentials as creds
    monkeypatch.setattr(creds, "resolve_env", lambda uid, **kw: None)
    assert _call(route, monkeypatch) == {
        "success": False, "status": "no_credentials",
        "message": f"Add your own agent API key in account settings to {verb} specs."}
    assert world == []


@pytest.mark.parametrize("route", SERVICE)
def test_undecryptable_credentials(world, monkeypatch, route):
    from backend.service import agent_credentials as creds

    def undecryptable(uid, **kw):
        raise creds.CredentialDecryptError("x")
    monkeypatch.setattr(creds, "resolve_env", undecryptable)
    assert _call(route, monkeypatch) == {
        "success": False, "status": "credential_undecryptable",
        "message": "Your saved agent API key could not be decrypted — please re-enter it in account settings."}
    assert world == []


@pytest.mark.parametrize("route", SERVICE)
def test_admin_undecryptable_key_is_not_replaced_by_the_host(world, monkeypatch, route):
    """An admin's stored key that this server cannot decrypt is refused like
    anyone's — never silently swapped for the host's credentials."""
    from backend.service import agent_credentials as creds
    from backend.service import auth

    def undecryptable(uid, **kw):
        raise creds.CredentialDecryptError("x")
    monkeypatch.setattr(auth, "is_admin", lambda uid: True)
    monkeypatch.setattr(creds, "resolve_env", undecryptable)
    assert _call(route, monkeypatch)["status"] == "credential_undecryptable"
    assert world == []


@pytest.mark.parametrize("route", SERVICE)
def test_admin_without_a_key_runs_on_host_credentials(world, monkeypatch, route):
    from backend.service import agent_credentials as creds
    from backend.service import auth
    monkeypatch.setattr(auth, "is_admin", lambda uid: True)
    monkeypatch.setattr(creds, "resolve_env", lambda uid, **kw: None)
    assert _call(route, monkeypatch)["status"] == "started"


@pytest.mark.parametrize("route", ["service_generate", "core_generate",
                                   "core_refine", "service_refine"])
def test_host_run_without_host_credentials(world, monkeypatch, route):
    """A run on the host's credentials when the host has none is refused
    before it starts, not left to the agent CLI's "please log in"."""
    from backend.service import agent_credentials as creds
    from backend.service import auth
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(auth, "is_admin", lambda uid: True)
    monkeypatch.setattr(creds, "resolve_env", lambda uid, **kw: None)
    out = _call(route, monkeypatch)
    assert out["status"] == "host_credentials_missing"
    assert "ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN" in out["message"]
    assert world == []


@pytest.mark.parametrize("route", SERVICE)
def test_quota_exceeded_before_the_lock(world, monkeypatch, route):
    from backend import spec_worker
    from backend.service.routes import spec as service
    monkeypatch.setattr(service, "_quota_room", lambda uid: False)
    monkeypatch.setattr(spec_worker, "acquire",
                        lambda key: pytest.fail("lock taken over quota"))
    assert _call(route, monkeypatch) == {
        "success": False, "status": "quota_exceeded",
        "message": f"Generation limit reached ({service._QUOTA_MAX}/hour). Try again later."}
    assert world == []


@pytest.mark.parametrize("route", ["core_generate", "service_generate"])
def test_a_known_model_is_cached(world, monkeypatch, route, tmp_path):
    from types import SimpleNamespace

    from backend.arch import loader
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **k: str(cfg))
    monkeypatch.setattr(loader.SpecLoader, "match",
                        lambda self, config, model_id=None:
                        SimpleNamespace(spec=SimpleNamespace(name="known")))
    assert _call(route, monkeypatch) == {
        "success": True, "spec_name": "known", "status": "cached"}
    assert world == []


@pytest.mark.parametrize("route, task", [("service_generate", "generate"),
                                         ("service_refine", "refine")])
def test_the_run_resolves_its_task_model(world, monkeypatch, route, task):
    from backend.service import agent_credentials as creds
    seen = []

    def resolve(uid, agent_cli=None, *, task=None):
        seen.append(task)
        return {"agent_cli": "claude", "env": {"K": "v"}}
    monkeypatch.setattr(creds, "resolve_env", resolve)
    assert _call(route, monkeypatch)["status"] == "started"
    assert seen == [task]
