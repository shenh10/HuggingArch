"""End-to-end per-user isolation (HA_PERUSER_SPECGEN=1), agent-free.

Proves the risky new behavior without a real agent run:
  • run-key isolation — two users generating the SAME model don't collide on
    the lock; the same user's duplicate still serializes.
  • serving precedence — /api/spec/analyze serves a PUBLIC (repo/disk) spec
    when one matches (never consulting the user's private store), and falls
    back to the signed-in user's OWN spec_generations spec on a public miss.

The agent write→harvest half is covered by the overlay unit tests + the
(flag-off) pipeline; here we pin the isolation + routing the flag switches on.
"""
from __future__ import annotations

import threading

from backend.arch.loader import SpecLoadError



def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Run-key isolation (the "users no longer block each other" fix)
# ---------------------------------------------------------------------------
def test_run_key_isolation(monkeypatch):
    from backend.spec_worker import runtime
    # In-memory lock path (no Supabase) + a roomy semaphore so concurrency
    # isn't the thing under test.
    monkeypatch.setattr(runtime, "_supabase_lock_enabled", lambda: False)
    monkeypatch.setattr(runtime, "semaphore", threading.BoundedSemaphore(8))
    runtime._model_locks.clear()

    assert runtime.acquire("userA|meta/Foo") == "ok"
    # Different user, SAME model → must NOT be blocked (the whole point).
    assert runtime.acquire("userB|meta/Foo") == "ok"
    # Same user + model again → still serialized (one run at a time per user).
    assert runtime.acquire("userA|meta/Foo") == "duplicate"

    runtime.release("userA|meta/Foo")
    runtime.release("userB|meta/Foo")
    # After release, the user can start again.
    assert runtime.acquire("userA|meta/Foo") == "ok"
    runtime.release("userA|meta/Foo")


# ---------------------------------------------------------------------------
# Serving precedence via /api/spec/analyze
# ---------------------------------------------------------------------------
def _patch_build_response(monkeypatch):
    # Minimal build_response so the route doesn't need a real analyzer result.
    monkeypatch.setattr(
        "backend.api.routes.analyze.build_response",
        lambda req, result, elapsed_ms, overlay_dir=None: {
            "success": True,
            "spec_matched": result.get("spec_matched"),
            "overlay_dir": overlay_dir,
        },
    )


def test_analyze_public_spec_wins(client, user_token, monkeypatch):
    """A repo/disk spec match is served as-is — the user's private store is
    never consulted (repo precedence)."""
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    _patch_build_response(monkeypatch)
    monkeypatch.setattr("backend.analyzer.analyze_model",
                        lambda **kw: {"success": True, "spec_matched": True})
    consulted = []
    monkeypatch.setattr("backend.service.spec_generations.serving_generation",
                        lambda uid, mid: consulted.append((uid, mid)))

    r = client.post("/api/spec/analyze", json={"model_id": "meta/Foo"},
                    headers=_auth(user_token))
    assert r.status_code == 200
    body = r.json()
    assert body["spec_matched"] is True
    assert body["overlay_dir"] is None          # public path, no overlay
    assert consulted == []                        # private store never touched


def test_analyze_falls_back_to_user_spec(client, user_token, monkeypatch):
    """On a public miss, a signed-in user is served their OWN latest spec."""
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    _patch_build_response(monkeypatch)
    seen_overlays = []

    def fake_analyze(**kw):
        ov = kw.get("overlay_dir")
        seen_overlays.append(ov)
        # public (no overlay) misses — the analyzer refuses; the user overlay matches.
        if ov is None:
            raise ValueError("No spec matches 'meta/Foo'. Add a spec … or run /generate-spec first.")
        return {"success": True, "spec_matched": True}

    monkeypatch.setattr("backend.analyzer.analyze_model", fake_analyze)
    monkeypatch.setattr("backend.service.spec_generations.serving_generation",
                        lambda uid, mid: {"status": "generated", "spec_yaml": "name: x\n", "new_component_yamls": []})
    monkeypatch.setattr("backend.spec_worker.overlay.materialize_user_spec",
                        lambda uid, mid, bundle, snapshot_json=None: "/tmp/ov")

    r = client.post("/api/spec/analyze", json={"model_id": "meta/Foo"},
                    headers=_auth(user_token))
    assert r.status_code == 200
    body = r.json()
    assert body["spec_matched"] is True
    assert body.get("served_from") == "user_generation"
    assert seen_overlays == [None, "/tmp/ov"]     # public first, then overlay


def test_analyze_incomplete_bundle_is_typed_and_writes_no_overlay(
        client, user_token, tmp_path, monkeypatch):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path))
    calls = []

    def fake_analyze(**kw):
        calls.append(kw.get("overlay_dir"))
        if kw.get("overlay_dir") is None:
            raise ValueError("No spec matches 'meta/Foo'.")
        return {"success": True}

    monkeypatch.setattr("backend.analyzer.analyze_model", fake_analyze)
    monkeypatch.setattr(
        "backend.service.spec_generations.serving_generation",
        lambda uid, mid: {"status": "generated", 
            "spec_yaml": "match: {model_id: meta/Foo}\ndrafts: [missing]\n",
            "new_component_yamls": [],
        })

    r = client.post("/api/spec/analyze", json={"model_id": "meta/Foo"},
                    headers=_auth(user_token))

    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error_kind"] == "spec_bundle_incomplete"
    assert "draft component 'missing'" in body["message"]
    assert body["errors"]
    assert calls == [None]
    assert not (tmp_path / "user-uuid-1" / "meta__Foo").exists()


def test_analyze_malformed_draft_is_typed_and_writes_no_overlay(
        client, user_token, tmp_path, monkeypatch):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path))
    calls = []

    def fake_analyze(**kw):
        calls.append(kw.get("overlay_dir"))
        if kw.get("overlay_dir") is None:
            raise ValueError("No spec matches 'meta/Foo'.")
        return {"success": True}

    monkeypatch.setattr("backend.analyzer.analyze_model", fake_analyze)
    monkeypatch.setattr(
        "backend.service.spec_generations.serving_generation",
        lambda uid, mid: {"status": "generated", 
            "spec_yaml": "match: {model_id: meta/Foo}\ndrafts: [bad]\n",
            "new_component_yamls": [{
                "path": "backend/arch/components/drafts/bad.yaml",
                "body": "role: attention\ngraph: [not-a-map]\n",
            }],
        })

    r = client.post("/api/spec/analyze", json={"model_id": "meta/Foo"},
                    headers=_auth(user_token))

    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error_kind"] == "spec_bundle_incomplete"
    assert "draft component 'bad'" in body["message"]
    assert "not parseable" in body["message"]
    assert calls == [None]
    assert not (tmp_path / "user-uuid-1" / "meta__Foo").exists()


def test_analyze_draft_conflict_is_an_actionable_spec_error(
        client, user_token, tmp_path, monkeypatch):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(tmp_path))
    conflict_path = tmp_path / "user-uuid-1" / "meta__Foo" / "components" / "drafts" / "old.yaml"
    conflict_path.parent.mkdir(parents=True)

    def fake_analyze(**kw):
        if kw.get("overlay_dir") is None:
            raise ValueError("No spec matches 'meta/Foo'.")
        raise SpecLoadError(
            "draft component 'old' has the same name as the shared component "
            "but a different structure; rename this draft and update the "
            "target spec", path=str(conflict_path))

    monkeypatch.setattr("backend.analyzer.analyze_model", fake_analyze)
    monkeypatch.setattr(
        "backend.service.spec_generations.serving_generation",
        lambda uid, mid: {"status": "generated", 
            "spec_yaml": "match: {model_id: meta/Foo}\n",
            "new_component_yamls": [],
        })
    monkeypatch.setattr(
        "backend.spec_worker.overlay.materialize_user_spec",
        lambda uid, mid, bundle, snapshot_json=None: str(conflict_path.parents[2]))

    r = client.post("/api/spec/analyze", json={"model_id": "meta/Foo"},
                    headers=_auth(user_token))

    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error_kind"] == "spec_invalid"
    assert "Your saved spec for this model no longer validates" in body["message"]
    assert "rename this draft" in body["message"]


def test_analyze_anonymous_never_serves_private(client, monkeypatch):
    """No token → the private fallback is never reached even with the flag on."""
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    _patch_build_response(monkeypatch)
    def public_miss(**kw):
        raise ValueError("No spec matches 'meta/Foo'. Add a spec … or run /generate-spec first.")

    monkeypatch.setattr("backend.analyzer.analyze_model", public_miss)
    consulted = []
    monkeypatch.setattr("backend.service.spec_generations.serving_generation",
                        lambda uid, mid: consulted.append(1))

    r = client.post("/api/spec/analyze", json={"model_id": "meta/Foo"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False and "generate-spec" in body["message"]
    assert consulted == []                         # anon never hits private store


def test_analyze_spec_load_error_is_a_spec_error(client, user_token, monkeypatch, tmp_path):
    """A spec that exists but fails validation must surface as a spec error
    (success:false + error_kind + actionable message), not as an unhandled
    500 the frontend would misread as 'backend unreachable'."""
    from backend.arch.loader import SpecLoadError
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    _patch_build_response(monkeypatch)

    user_root = tmp_path / "user_specs"
    monkeypatch.setenv("HA_USER_SPECS_ROOT", str(user_root))
    broken = user_root / "u" / "M" / "models" / "M.yaml"

    def boom(**kw):
        # The loader names the file; the route asks WHERE it lives, not
        # whether the message happens to mention `user_specs`.
        raise SpecLoadError(
            f"spec for model_type ['foo'] exists at {broken} "
            "but failed to load: 1 validation error", path=str(broken))

    monkeypatch.setattr("backend.analyzer.analyze_model", boom)

    r = client.post("/api/spec/analyze", json={"model_id": "meta/Foo"},
                    headers=_auth(user_token))
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error_kind"] == "spec_invalid"
    assert "no longer validates" in body["message"]


# ---------------------------------------------------------------------------
# Phase 2: anonymous core /generate-spec gated in multi-user mode
# ---------------------------------------------------------------------------
def test_anon_generate_disabled_in_multiuser(client, monkeypatch):
    monkeypatch.setattr("backend.supabase_env.jwt_verification_available", lambda: True)  # multi-user
    r = client.post("/generate-spec", json={"model_id": "meta/Foo"})
    assert r.status_code == 200
    assert r.json().get("status") == "disabled"    # no job, no token spend


def test_anon_generate_allowed_selfhost(client, monkeypatch, tmp_path):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr("backend.supabase_env.jwt_verification_available", lambda: False)  # self-host
    r = client.post("/generate-spec", json={"model_id": "meta/Foo"})
    # Not blocked by the gate — proceeds (and fails later on e.g. missing agent
    # CLI), but the status is NEVER 'disabled'.
    assert r.json().get("status") != "disabled"


# ---------------------------------------------------------------------------
# Phase 3: the global cached_specs table never reaches a signed-in user
# ---------------------------------------------------------------------------
def test_cached_specs_hook_refuses_in_multiuser(monkeypatch):
    """In multi-user mode the miss hook must NEVER materialize a globally
    cached (cross-user) spec — only repo YAML + the user's own overlay reach
    a user. The refusal happens before Supabase is consulted."""
    from backend.spec_worker import shared_spec_miss
    fetched = []
    monkeypatch.setattr("backend.spec_worker.shared_spec_miss.fetch",
                        lambda mid: fetched.append(mid) or {"spec_yaml": "name: x\n"})
    monkeypatch.setattr("backend.supabase_env.jwt_verification_available", lambda: True)
    assert shared_spec_miss.materialize_cached_spec("meta/Foo") is None
    assert fetched == []


def test_cached_specs_hook_refuses_the_global_cache_when_the_service_env_is_pinned(monkeypatch):
    """With the service environment pinned (Supabase URL set) the global
    cache is refused before any fetch — the switch is an environment read."""
    from backend.spec_worker import shared_spec_miss
    fetched = []
    monkeypatch.setattr("backend.spec_worker.shared_spec_miss.fetch",
                        lambda mid: fetched.append(mid) or {"spec_yaml": "name: x\n"})

    # The multi-user switch is an environment read, not an auth-backend
    # call: there is no "backend down" to fail closed against. With the
    # service env pinned (URL set) the global cache is refused before any
    # fetch; the fetch stub must stay untouched.
    from backend import supabase_env
    assert supabase_env.jwt_verification_available()
    assert shared_spec_miss.materialize_cached_spec("meta/Foo") is None
    assert fetched == []


def test_loader_has_no_service_import():
    """The loader reaches the cache only through the registered hook."""
    import inspect
    import backend.arch.loader as loader_mod
    assert "backend.service" not in inspect.getsource(loader_mod)


# ---------------------------------------------------------------------------
# Per-user refine (POST /api/spec/refine)
# ---------------------------------------------------------------------------
def _refine_body(mid="meta/Foo"):
    return {"model_id": mid, "spec_storage_bytes": 10, "hf_storage_bytes": 12,
            "storage_diff": 2}


def test_refine_requires_login(client, monkeypatch):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    r = client.post("/api/spec/refine", json=_refine_body())
    assert r.status_code == 401  # require_user, no token


def test_refine_no_spec_to_refine(host_credentials, client, user_token, monkeypatch):
    """A user with no prior generation has nothing to refine — no job started."""
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    import backend.service.routes.spec as S
    monkeypatch.setattr(S.spec_worker, "get_agent_binary", lambda: "claude")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/claude")
    monkeypatch.setattr(S._auth, "is_admin", lambda uid: True)  # skip BYOK gate
    monkeypatch.setattr(S._creds, "resolve_env", lambda uid, **kw: None)
    monkeypatch.setattr("backend.service.spec_generations.serving_generation",
                        lambda uid, mid: None)
    started = []
    monkeypatch.setattr(S.spec_worker, "start_job", lambda *a, **k: started.append(a))

    monkeypatch.setattr("backend.spec_worker.overlay.load_seed_spec_yaml",
                        lambda mid: None)
    r = client.post("/api/spec/refine", json=_refine_body(), headers=_auth(user_token))
    assert r.status_code == 409
    body = r.json()
    assert body.get("status") == "no_spec"
    assert "generate" in (body.get("message") or "").lower()
    assert started == []  # never acquired a run / spent tokens


def test_refine_seeds_overlay_and_starts(host_credentials, client, user_token, tmp_path, monkeypatch):
    """The user's stored spec is seeded into a run overlay and a keyed job
    starts against it (agent-free — start_job stubbed)."""
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    import backend.service.routes.spec as S
    monkeypatch.setattr(S.spec_worker, "get_agent_binary", lambda: "claude")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/claude")
    monkeypatch.setattr(S._auth, "is_admin", lambda uid: True)
    monkeypatch.setattr(S._creds, "resolve_env", lambda uid, **kw: None)
    monkeypatch.setattr("backend.service.spec_generations.serving_generation",
                        lambda uid, mid: {"status": "generated", "spec_yaml": "name: foo\n", "new_component_yamls": []})
    monkeypatch.setattr(S.spec_worker, "acquire", lambda k: "ok")
    captured = {}
    monkeypatch.setattr(S.spec_worker, "start_job",
                        lambda run_key, fn, **kw: captured.update(run_key=run_key, **kw))

    r = client.post("/api/spec/refine", json=_refine_body(), headers=_auth(user_token))
    assert r.status_code == 200
    assert r.json().get("status") == "started"
    assert captured["run_key"] == "user-uuid-1|refine|meta/Foo"
    assert captured.get("kind") == "refine"
    assert captured.get("model_id") == "meta/Foo"
    # The current spec was materialized into the run overlay for the agent.
    seeded = (tmp_path / "runs" / "user-uuid-1" / "refine__meta__Foo"
              / "models" / "meta__Foo.yaml")
    assert seeded.read_text() == "name: foo\n"


def test_refine_job_carries_the_seed_bundle_to_publication(host_credentials, client, user_token, tmp_path, monkeypatch):
    """The prompt prep result must carry the same normalized seed to publish."""
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    from backend.spec_worker.bundle import BundleFile
    import backend.service.routes.spec as S
    proposal = BundleFile(
        "backend/arch/components/library/removed_attention.yaml",
        "role: attention\n", formal_body_sha256="3" * 64)
    monkeypatch.setattr(S.spec_worker, "get_agent_binary", lambda: "claude")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/claude")
    monkeypatch.setattr(S._auth, "is_admin", lambda uid: True)
    monkeypatch.setattr(S._creds, "resolve_env", lambda uid, **kw: None)
    monkeypatch.setattr(
        "backend.service.spec_generations.serving_generation",
        lambda uid, mid: {"status": "generated", 
            "spec_yaml": "match: {model_id: meta/Foo}\n",
            "new_component_yamls": [proposal.as_json()],
        })
    monkeypatch.setattr(S.spec_worker, "acquire", lambda k: "ok")
    captured = {}

    def fake_prepare_refine_prompt(*_args, seed_bundle=None, **_kwargs):
        return {"prompt": "", "target_root": None,
                "seeded_spec": True, "seed_bundle": seed_bundle}

    def fake_generation_job(_model_id, prep_fn, **_kwargs):
        def _run():
            prepared = prep_fn()
            captured["seed_bundle"] = prepared.get("seed_bundle")
            yield {"type": "done", "success": True}
        return _run

    monkeypatch.setattr(S.spec_worker, "prepare_refine_prompt",
                        fake_prepare_refine_prompt)
    monkeypatch.setattr(S.spec_worker, "generation_job", fake_generation_job)
    monkeypatch.setattr(S.spec_worker, "start_job",
                        lambda run_key, fn, **_kw: captured.update(
                            run_key=run_key, job=fn))

    r = client.post("/api/spec/refine", json=_refine_body(),
                    headers=_auth(user_token))

    assert r.status_code == 200 and r.json().get("status") == "started"
    assert captured["run_key"] == "user-uuid-1|refine|meta/Foo"
    list(captured["job"]())
    assert captured["seed_bundle"] is not None
    assert captured["seed_bundle"].proposal_files == (proposal,)


def test_refine_peruser_off_takes_global_path(host_credentials, client, user_token, tmp_path, monkeypatch):
    """Auth ON but per-user isolation explicitly OFF → refine mirrors generate's
    global path: keyed by model_id (no user overlay owner), not a special
    'unavailable'. The write root is still a kind-isolated `__anonymous__` leaf.
    """
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "0")
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    import backend.service.routes.spec as S
    monkeypatch.setattr(S.spec_worker, "get_agent_binary", lambda: "claude")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/claude")
    monkeypatch.setattr(S._auth, "is_admin", lambda uid: True)
    monkeypatch.setattr(S._creds, "resolve_env", lambda uid, **kw: None)
    monkeypatch.setattr(S.spec_worker, "acquire", lambda k: "ok")
    monkeypatch.setattr("backend.spec_worker.overlay.load_seed_spec_yaml",
                        lambda mid: "name: foo\n")
    captured = {}
    monkeypatch.setattr(S.spec_worker, "start_job",
                        lambda run_key, fn, **kw: captured.update(run_key=run_key, **kw))
    r = client.post("/api/spec/refine", json=_refine_body(), headers=_auth(user_token))
    assert r.status_code == 200 and r.json().get("status") == "started"
    assert captured["run_key"] == "refine|meta/Foo"
    ref_root = tmp_path / "runs" / "__anonymous__" / "refine__meta__Foo"
    assert ref_root.is_dir()


def test_refine_stop_uses_refine_run_key(client, user_token, monkeypatch):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    import backend.service.routes.spec as S
    stopped = []
    monkeypatch.setattr(S.spec_worker, "stop", lambda key: stopped.append(key) or True)
    r = client.post("/api/spec/stop",
                    json={"model_id": "meta/Foo", "kind": "refine"},
                    headers=_auth(user_token))
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True
    assert stopped == ["user-uuid-1|refine|meta/Foo"]


def test_stop_rejects_unknown_kind(client, user_token, monkeypatch):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    r = client.post("/api/spec/stop",
                    json={"model_id": "meta/Foo", "kind": "other"},
                    headers=_auth(user_token))
    assert r.status_code == 422


def test_stop_review_uses_review_run_key(client, user_token, monkeypatch):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    import backend.service.routes.spec as S
    stopped = []
    monkeypatch.setattr(S.spec_worker, "stop", lambda key: stopped.append(key) or True)
    r = client.post("/api/spec/stop",
                    json={"kind": "review", "record_id": "rec-1", "model_id": ""},
                    headers=_auth(user_token))
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True
    assert stopped == ["review|user-uuid-1|rec-1"]


def test_stop_fix_requires_record_id(client, user_token):
    r = client.post("/api/spec/stop",
                    json={"kind": "fix"},
                    headers=_auth(user_token))
    assert r.status_code == 422


def test_in_flight_generate_and_refine_use_different_overlay_roots(host_credentials, client, user_token, tmp_path, monkeypatch):
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "1")
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    import backend.service.routes.spec as S
    monkeypatch.setattr(S.spec_worker, "get_agent_binary", lambda: "claude")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/claude")
    monkeypatch.setattr(S._auth, "is_admin", lambda uid: True)
    monkeypatch.setattr(S._creds, "resolve_env", lambda uid, **kw: None)
    monkeypatch.setattr(S.spec_worker, "acquire", lambda k: "ok")
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **k: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr("backend.service.spec_generations.serving_generation",
                        lambda uid, mid: {"status": "generated", "spec_yaml": "name: foo\n",
                                          "new_component_yamls": []})
    roots = []
    monkeypatch.setattr(S.spec_worker, "start_job",
                        lambda run_key, fn, **kw: roots.append(kw.get("kind")))
    gen = client.post("/api/spec/generate", json={"model_id": "meta/Foo"},
                      headers=_auth(user_token))
    ref = client.post("/api/spec/refine", json=_refine_body(),
                      headers=_auth(user_token))
    assert gen.status_code == 200 and gen.json().get("status") == "started"
    assert ref.status_code == 200 and ref.json().get("status") == "started"
    gen_root = tmp_path / "runs" / "user-uuid-1" / "generate__meta__Foo"
    ref_root = tmp_path / "runs" / "user-uuid-1" / "refine__meta__Foo"
    assert gen_root.is_dir() and ref_root.is_dir()
    assert gen_root.resolve() != ref_root.resolve()
    assert (ref_root / ".agent-home").is_dir()
    assert (gen_root / ".agent-home").is_dir()


def test_peruser_off_in_flight_generate_and_refine_use_different_overlay_roots(host_credentials, client, user_token, tmp_path, monkeypatch):
    """HA_PERUSER_SPECGEN=0 still isolates generate vs refine write roots."""
    monkeypatch.setenv("HA_PERUSER_SPECGEN", "0")
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    import backend.service.routes.spec as S
    monkeypatch.setattr(S.spec_worker, "get_agent_binary", lambda: "claude")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/claude")
    monkeypatch.setattr(S._auth, "is_admin", lambda uid: True)
    monkeypatch.setattr(S._creds, "resolve_env", lambda uid, **kw: None)
    monkeypatch.setattr(S.spec_worker, "acquire", lambda k: "ok")
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **k: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr("backend.spec_worker.overlay.load_seed_spec_yaml",
                        lambda mid: "name: foo\n")
    monkeypatch.setattr(S.spec_worker, "start_job", lambda run_key, fn, **kw: None)
    gen = client.post("/api/spec/generate", json={"model_id": "meta/Foo"},
                      headers=_auth(user_token))
    ref = client.post("/api/spec/refine", json=_refine_body(),
                      headers=_auth(user_token))
    assert gen.status_code == 200 and gen.json().get("status") == "started"
    assert ref.status_code == 200 and ref.json().get("status") == "started"
    gen_root = tmp_path / "runs" / "__anonymous__" / "generate__meta__Foo"
    ref_root = tmp_path / "runs" / "__anonymous__" / "refine__meta__Foo"
    assert gen_root.is_dir() and ref_root.is_dir()
    assert gen_root.resolve() != ref_root.resolve()
    assert (gen_root / ".agent-home").is_dir()
    assert (ref_root / ".agent-home").is_dir()


def test_custom_in_flight_generate_and_refine_use_different_overlay_roots(host_credentials, client, tmp_path, monkeypatch):
    """Core custom generate/refine write to distinct `__anonymous__` leaves,
    not the shared custom-cache spec.yaml."""
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setattr("backend.supabase_env.jwt_verification_available",
                        lambda: False)
    import backend.api.routes.spec as core
    monkeypatch.setattr(core.spec_worker, "get_agent_binary", lambda: "claude")
    monkeypatch.setattr("shutil.which", lambda _b: "/usr/bin/claude")
    monkeypatch.setattr(core.spec_worker, "acquire", lambda k: "ok")
    monkeypatch.setattr(core, "_load_custom", lambda name: {"config": {}})
    monkeypatch.setattr("backend.spec_worker.overlay.load_seed_spec_yaml",
                        lambda mid: "match:\n  model_type: [toy]\n")
    monkeypatch.setattr(core.spec_worker, "start_job", lambda run_key, fn, **kw: None)
    model_id = "custom:s8a_r3_toy"
    gen = client.post("/generate-spec", json={"model_id": model_id})
    ref = client.post("/refine-spec", json=_refine_body(model_id))
    assert gen.status_code == 200 and gen.json().get("status") == "started"
    assert ref.status_code == 200 and ref.json().get("status") == "started"
    gen_root = tmp_path / "runs" / "__anonymous__" / "generate__custom:s8a_r3_toy"
    ref_root = tmp_path / "runs" / "__anonymous__" / "refine__custom:s8a_r3_toy"
    assert gen_root.is_dir() and ref_root.is_dir()
    assert gen_root.resolve() != ref_root.resolve()
    assert (gen_root / "models").is_dir()
    assert (ref_root / "models").is_dir()


def test_cached_specs_hook_consults_cache_on_selfhost(monkeypatch):
    """Self-host (auth off) still consults the cross-VM cache; a miss is None."""
    from backend.spec_worker import shared_spec_miss
    fetched = []
    monkeypatch.setattr("backend.spec_worker.shared_spec_miss.fetch",
                        lambda mid: fetched.append(mid) or None)
    monkeypatch.setattr("backend.supabase_env.jwt_verification_available", lambda: False)
    assert shared_spec_miss.materialize_cached_spec("meta/Foo") is None
    assert fetched == ["meta/Foo"]
