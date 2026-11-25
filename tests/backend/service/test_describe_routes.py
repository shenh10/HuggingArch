"""A model described in words, agent-free: the turn routes hand the agent
the right prompt and workspace, a hand edit is judged before it is kept,
and the Playground serves the caller's own model.

The model is Llama-3.2-1B's spec and config stored as `custom:tiny` with no
checkpoint, the shape a described model is kept in.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SOURCE = "meta-llama__Llama-3.2-1B"
MODEL = "custom:tiny"


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _spec(model_id: str = MODEL) -> str:
    return (PROJECT_ROOT / "backend/arch/models" / f"{_SOURCE}.yaml").read_text(
    ).replace("model_id: [meta-llama/Llama-3.2-1B]", f"model_id: [{model_id}]")


def _config() -> dict:
    return json.loads((PROJECT_ROOT / "backend/arch/snapshots"
                       / f"{_SOURCE}.json").read_text())["config"]


def _broken_spec() -> str:
    """llama.yaml whose lm_head takes d_ffn while d_model reaches it."""
    return ((PROJECT_ROOT / "backend/arch/models/llama.yaml").read_text()
            .replace("  model_type: [llama]", f"  model_id: [{MODEL}]")
            .replace("dims: { in: d_model, out: n_vocab }",
                     "dims: { in: d_ffn, out: n_vocab }")
            .replace(", tied_weight: { to: embed_tokens, when: "
                     "config.tie_word_embeddings } ", ""))


def _row() -> dict:
    return {"id": "rec-1", "model_id": MODEL, "status": "generated",
            "revision": 3,
            "spec_yaml": _spec(), "new_component_yamls": [],
            "snapshot_json": {"model_id": MODEL, "config": _config(),
                              "description": "Llama-3.2-1B, kept"}}


@pytest.fixture
def described(monkeypatch, tmp_path):
    """Per-user mode over private roots, the caller holding `custom:tiny`."""
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path / "user_specs"))
    monkeypatch.setattr("backend.service.auth.is_admin", lambda uid: False)
    from backend.spec_worker import runtime
    monkeypatch.setattr(runtime, "_supabase_lock_enabled", lambda: False)
    runtime._model_locks.clear()
    rows = {MODEL: _row()}
    monkeypatch.setattr("backend.spec_worker.records.serving_generation",
                        lambda uid, mid: rows.get(mid))
    monkeypatch.setattr(
        "backend.service.spec_generations.serving_generation",
        lambda uid, mid: rows.get(mid))
    recorded = []
    monkeypatch.setattr("backend.spec_worker.records.record",
                        lambda **kw: recorded.append(kw) or {"outcome": "updated"})
    return rows, recorded


def test_a_broken_hand_edit_is_refused_and_the_model_stands(
        client, user_token, described):
    _rows, recorded = described
    r = client.post("/api/spec/describe/save", headers=_auth(user_token),
                    json={"name": "tiny", "spec_yaml": _broken_spec(),
                          "config": _config()})
    body = r.json()
    assert body["success"] is False, body
    assert any("its weight takes" in e for e in body["errors"])
    assert recorded == []


def test_a_sound_hand_edit_is_kept(client, user_token, described):
    _rows, recorded = described
    config = {**_config(), "num_hidden_layers": 8}
    r = client.post("/api/spec/describe/save", headers=_auth(user_token),
                    json={"name": "tiny", "spec_yaml": _spec(), "config": config})
    body = r.json()
    assert body["success"] is True, body
    assert "checkpoint bytes" in body["checks"]["skipped"]
    (kept,) = recorded
    assert kept["model_id"] == MODEL
    assert kept["snapshot_json"]["config"]["num_hidden_layers"] == 8


def test_the_playground_serves_the_callers_own_model(
        client, user_token, described):
    r = client.get("/api/account/playground/model-skeleton",
                   params={"model_id": MODEL}, headers=_auth(user_token))
    body = r.json()
    assert body["success"] is True, body
    assert body["n_layers"] == 16
    assert body["served_from"] == "user_generation"
    assert body["stored_without_checkpoint"] is True


def test_an_edit_made_on_an_older_revision_is_refused(
        client, user_token, described):
    _rows, recorded = described
    r = client.post("/api/spec/describe/save", headers=_auth(user_token),
                    json={"name": "tiny", "spec_yaml": _spec(),
                          "config": _config(), "revision": 2})
    assert r.status_code == 409 and r.json()["status"] == "stale"
    assert recorded == []


def test_an_edit_waits_for_a_running_turn(client, user_token, described,
                                          monkeypatch):
    monkeypatch.setattr("backend.spec_worker.acquire", lambda key: "duplicate")
    r = client.post("/api/spec/describe/save", headers=_auth(user_token),
                    json={"name": "tiny", "spec_yaml": _spec(), "config": _config()})
    assert r.status_code == 409 and r.json()["status"] == "busy"


def _started(monkeypatch):
    """Capture the job a turn starts instead of running an agent."""
    started = {}
    monkeypatch.setattr("backend.service.routes.spec._preflight",
                        lambda uid, admin, verb: (None, None))
    monkeypatch.setattr("backend.service.routes.describe._preflight",
                        lambda uid, admin, verb: (None, None))
    monkeypatch.setattr("backend.service.routes.describe._claim_run",
                        lambda *a: None)
    monkeypatch.setattr("backend.spec_worker.get_agent_cli", lambda: "claude")
    monkeypatch.setattr(
        "backend.spec_worker.generation_job",
        lambda model_id, prep_fn, **kw: started.update(
            model_id=model_id, prep_fn=prep_fn, **kw) or (lambda: None))
    monkeypatch.setattr("backend.spec_worker.start_job",
                        lambda run_key, job, **kw: started.update(run_key=run_key))
    monkeypatch.setattr(
        "backend.service.agent_credentials.fallback_envs",
        lambda *a, **k: {})
    return started


def test_a_first_turn_writes_the_model_from_its_description(
        client, user_token, described, monkeypatch):
    started = _started(monkeypatch)
    r = client.post("/api/spec/describe", headers=_auth(user_token), json={
        "name": "fresh", "description": "a 12-layer GQA model, hidden 1024"})
    assert r.json()["status"] == "started", r.json()
    prep = started["prep_fn"]()
    assert "a 12-layer GQA model, hidden 1024" in prep["prompt"]
    assert "snapshots/custom:fresh.json" in prep["prompt"]
    assert "models/custom:fresh.yaml" in prep["prompt"]


def test_a_name_already_kept_is_not_overwritten_by_a_first_turn(
        client, user_token, described, monkeypatch):
    _started(monkeypatch)
    r = client.post("/api/spec/describe", headers=_auth(user_token), json={
        "name": "tiny", "description": "anything"})
    assert r.status_code == 409
    assert r.json()["status"] == "name_taken"


def test_a_follow_up_starts_from_the_model_as_stored(
        client, user_token, described, monkeypatch):
    started = _started(monkeypatch)
    r = client.post("/api/spec/describe", headers=_auth(user_token), json={
        "name": "tiny", "message": "use 8 layers"})
    assert r.json()["status"] == "started", r.json()
    leaf = Path(started["target_root"])
    assert (leaf / "models" / f"{MODEL}.yaml").is_file()
    snapshot = json.loads((leaf / "snapshots" / f"{MODEL}.json").read_text())
    assert snapshot["config"]["num_hidden_layers"] == 16
    prep = started["prep_fn"]()
    assert "> use 8 layers" in prep["prompt"]
    assert "passes validation" in prep["prompt"]
    # the description is the stored model's, not the browser's
    assert "Llama-3.2-1B, kept" in prep["prompt"]
    assert prep["snapshot_fields"]["description"] == "Llama-3.2-1B, kept"
