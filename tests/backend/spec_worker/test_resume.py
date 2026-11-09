"""Continuing a failed generation from the draft its leaf kept."""
from __future__ import annotations

import json

import pytest

from backend.spec_worker import overlay
from backend.spec_worker import resume as RS
from backend.spec_worker import worker as W

MODEL = "meta/Foo"


def _failed_generate_leaf(tmp_path, monkeypatch, *, session="sess-1",
                          agent_cli="claude"):
    """A generate leaf as a failed run leaves it: its draft, a draft
    component, the agent's session transcript and the run's end record."""
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    leaf = overlay.new_run_overlay("user-a", MODEL, kind="generate")
    (leaf / "models" / "meta__Foo.yaml").write_text(
        "match: {model_id: [meta/Foo]}\n", encoding="utf-8")
    (leaf / "components" / "drafts" / "foo_attn.yaml").write_text(
        "role: attention\n", encoding="utf-8")
    transcript = leaf / ".agent-home" / ".claude" / "projects" / "-leaf"
    transcript.mkdir(parents=True)
    (transcript / f"{session}.jsonl").write_text("{}\n", encoding="utf-8")
    overlay.RunEnd(success=False, message="An agent round exceeded its budget",
                   timed_out=True, session_id=session, agent_cli=agent_cli,
                   ended_at=1.0).write(leaf)
    (leaf / ".tmp" / "scratch").write_text("x", encoding="utf-8")
    return leaf


def test_resume_mode_keeps_the_draft_and_the_agent_session(tmp_path, monkeypatch):
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    resumed = overlay.allocate_run_overlay(
        MODEL, kind="generate", user_id="user-a", resume=True)
    assert resumed == str(leaf)
    assert (leaf / "models" / "meta__Foo.yaml").is_file()
    assert (leaf / "components" / "drafts" / "foo_attn.yaml").is_file()
    assert list((leaf / ".agent-home" / ".claude" / "projects").glob("*/sess-1.jsonl"))
    assert overlay.RunEnd.read(leaf).session_id == "sess-1"
    # Temp files do not carry over.
    assert (leaf / ".tmp").is_dir() and list((leaf / ".tmp").iterdir()) == []


def test_a_new_generate_over_a_kept_draft_is_refused(tmp_path, monkeypatch):
    """Start over is the user's choice: a plain generate leaves the draft."""
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    with pytest.raises(overlay.DraftKept) as raised:
        overlay.allocate_run_overlay(MODEL, kind="generate", user_id="user-a")
    refusal = raised.value.refusal()
    assert refusal["status"] == "draft_kept" and refusal["resumable"] is True
    assert (leaf / "models" / "meta__Foo.yaml").is_file()
    assert list((leaf / ".agent-home").rglob("*.jsonl"))


def test_a_draft_left_by_a_successful_run_is_not_kept(tmp_path, monkeypatch):
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    overlay.RunEnd(success=True, message="", timed_out=False, session_id="s",
                   agent_cli="claude", ended_at=2.0).write(leaf)
    assert not overlay.has_resumable_draft(leaf)
    overlay.allocate_run_overlay(MODEL, kind="generate", user_id="user-a")
    assert list((leaf / "models").glob("*")) == []


def test_start_over_clears_the_draft_and_session(tmp_path, monkeypatch):
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    overlay.allocate_run_overlay(MODEL, kind="generate", user_id="user-a",
                                 discard_draft=True)
    assert list((leaf / "models").glob("*")) == []
    assert list((leaf / ".agent-home").iterdir()) == []
    assert overlay.RunEnd.read(leaf) is None


def test_resume_without_a_draft_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    with pytest.raises(overlay.NoResumableDraft) as raised:
        overlay.allocate_run_overlay(
            MODEL, kind="generate", user_id="user-a", resume=True)
    refusal = raised.value.refusal()
    assert refusal["status"] == "no_draft" and refusal["success"] is False
    assert "no earlier draft" in refusal["message"]
    # An empty leaf left by a fresh run is no draft either.
    overlay.new_run_overlay("user-a", MODEL, kind="generate")
    with pytest.raises(overlay.NoResumableDraft):
        overlay.allocate_run_overlay(
            MODEL, kind="generate", user_id="user-a", resume=True)


def test_drafts_without_the_model_spec_are_a_resumable_draft(tmp_path, monkeypatch):
    """A long run stopped mid-way often has components but no model spec."""
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    (leaf / "models" / "meta__Foo.yaml").unlink()
    assert overlay.has_resumable_draft(leaf)
    assert overlay.allocate_run_overlay(
        MODEL, kind="generate", user_id="user-a", resume=True) == str(leaf)
    (leaf / "components" / "drafts" / "foo_attn.yaml").unlink()
    assert not overlay.has_resumable_draft(leaf)


def test_resume_refuses_a_draft_pruned_while_it_is_prepared(tmp_path, monkeypatch):
    """GC can remove the leaf between the check and the prepare; the prepare
    recreates empty dirs, which must not start as a continuation."""
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    real_prepare = overlay._prepare

    def prepare_after_gc(path, **kw):
        import shutil
        shutil.rmtree(path)
        return real_prepare(path, **kw)

    monkeypatch.setattr(overlay, "_prepare", prepare_after_gc)
    with pytest.raises(overlay.NoResumableDraft):
        overlay.allocate_run_overlay(
            MODEL, kind="generate", user_id="user-a", resume=True)
    assert not leaf.joinpath("models", "meta__Foo.yaml").exists()


def test_only_generate_resumes(tmp_path, monkeypatch):
    _failed_generate_leaf(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        overlay.allocate_run_overlay(
            MODEL, kind="refine", user_id="user-a", resume=True)


def _stub_verdict(monkeypatch, downgrade):
    monkeypatch.setattr(W, "_verify_spec_output",
                        lambda *a, **k: (downgrade, None))


def test_prepare_resume_reenters_the_recorded_session(tmp_path, monkeypatch):
    """A session resume needs neither the task prompt nor a Hub fetch: the
    original prep is not run."""
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    _stub_verdict(monkeypatch, W.Downgrade(
        "validation", "generated spec failed validation: bad dim", ("bad dim",)))

    def base_prep():
        raise AssertionError("a session resume must not run the original prep")

    prep = RS.prepare_resume(
        MODEL, target_root=str(leaf), agent_cli="claude", base_prep=base_prep)
    assert prep["resume_session"] == "sess-1"
    assert prep["target_root"] == str(leaf)
    assert prep["seeded_spec"] is True
    assert "ORIGINAL TASK" not in prep["prompt"]
    assert "reached its time budget" in prep["prompt"]
    assert "`models/meta__Foo.yaml`" in prep["prompt"]
    assert "`components/drafts/foo_attn.yaml`" in prep["prompt"]
    assert "- bad dim" in prep["prompt"]
    assert "agent session" in prep["prefetch_logs"][-1]


def test_a_full_session_is_not_reentered_and_a_timed_out_one_is(
        tmp_path, monkeypatch):
    from backend.agents.base import AgentEnd
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)

    def ended(agent_end):
        # The leaf records the word the `done` wrote; `resume` reads it.
        return overlay.RunEnd(success=False, message="m", timed_out=False,
                              session_id="sess-1", agent_cli="claude",
                              ended_at=1.0, agent_end=agent_end.value)

    assert RS.resumable_session(
        ended(AgentEnd.TIMED_OUT), "claude", str(leaf)) == "sess-1"
    assert RS.resumable_session(
        ended(AgentEnd.CONTEXT_EXHAUSTED), "claude", str(leaf)) is None
    # The ending survives the leaf's record.
    ended(AgentEnd.CONTEXT_EXHAUSTED).write(leaf)
    end = overlay.RunEnd.read(leaf)
    assert end.agent_end == AgentEnd.CONTEXT_EXHAUSTED.value
    _stub_verdict(monkeypatch, None)
    prep = RS.prepare_resume(
        MODEL, target_root=str(leaf), agent_cli="claude",
        base_prep=lambda: {"prompt": "ORIGINAL TASK", "target_root": str(leaf)})
    assert prep["resume_session"] is None
    assert prep["prompt"].startswith("ORIGINAL TASK")
    assert "new agent session" in prep["prefetch_logs"][-1]


def test_a_record_written_before_the_agent_end_field_still_loads(
        tmp_path, monkeypatch):
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    (leaf / overlay.RUN_END_FILE).write_text(json.dumps({
        "success": False, "message": "m", "timed_out": True,
        "session_id": "sess-1", "agent_cli": "claude", "ended_at": 1.0}),
        encoding="utf-8")
    end = overlay.RunEnd.read(leaf)
    assert end is not None and end.agent_end is None
    assert RS.resumable_session(end, "claude", str(leaf)) == "sess-1"


def test_the_continuation_names_a_missing_model_spec(tmp_path, monkeypatch):
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    (leaf / "models" / "meta__Foo.yaml").unlink()
    _stub_verdict(monkeypatch, W.Downgrade("crash", "must not be called"))
    prompt = RS.continuation_prompt(MODEL, str(leaf), overlay.RunEnd.read(leaf))
    assert "`models/meta__Foo.yaml` is not written yet" in prompt
    assert "`components/drafts/foo_attn.yaml`" in prompt
    assert "must not be called" not in prompt


@pytest.mark.parametrize("change", ["other_cli", "no_transcript"])
def test_prepare_resume_cold_starts_with_the_task_and_the_draft(
        tmp_path, monkeypatch, change):
    leaf = _failed_generate_leaf(
        tmp_path, monkeypatch,
        agent_cli="codex" if change == "other_cli" else "claude")
    if change == "no_transcript":
        for path in (leaf / ".agent-home").rglob("*.jsonl"):
            path.unlink()
    _stub_verdict(monkeypatch, None)
    prep = RS.prepare_resume(
        MODEL, target_root=str(leaf), agent_cli="claude",
        base_prep=lambda: {"prompt": "ORIGINAL TASK", "target_root": str(leaf)})
    assert prep["resume_session"] is None
    assert prep["prompt"].startswith("ORIGINAL TASK")
    assert "passes validation as it stands" in prep["prompt"]
    assert "`models/meta__Foo.yaml`" in prep["prompt"]


def _job_events(monkeypatch, target_root, done, prep=None):
    def fake_stream(model_id, prep_result, *, run_key=None, failover=None):
        prep_result.update(_session_id="sess-9", _agent_cli="claude")
        yield f"data: {json.dumps(done)}\n\n"

    monkeypatch.setattr(W, "stream_agent", fake_stream)
    return list(W.generation_job(
        MODEL, prep or (lambda: {"prompt": "p", "target_root": target_root}),
        run_key="u|meta/Foo", target_root=target_root)())


def test_a_failed_done_says_whether_the_draft_can_be_continued(tmp_path, monkeypatch):
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    failed = {"type": "done", "success": False, "message": "stopped",
              "timed_out": True}
    assert _job_events(monkeypatch, str(leaf), failed)[-1]["resumable"] is True
    end = overlay.RunEnd.read(leaf)
    assert (end.session_id, end.agent_cli, end.timed_out, end.success) == (
        "sess-9", "claude", True, False)

    (leaf / "models" / "meta__Foo.yaml").unlink()
    assert _job_events(monkeypatch, str(leaf), failed)[-1]["resumable"] is True
    (leaf / "components" / "drafts" / "foo_attn.yaml").unlink()
    assert _job_events(monkeypatch, str(leaf), failed)[-1]["resumable"] is False


def test_a_run_records_its_session_before_it_ends(tmp_path, monkeypatch):
    """A restart can kill the job thread before any `done`: the session is
    on record as soon as it is known."""
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)

    def fake_stream(model_id, prep_result, *, run_key=None, failover=None):
        prep_result.update(_session_id="sess-live", _agent_cli="claude")
        yield 'data: {"type": "log", "text": "working"}\n\n'
        yield 'data: {"type": "log", "text": "still working"}\n\n'

    monkeypatch.setattr(W, "stream_agent", fake_stream)
    job = W.generation_job(MODEL, lambda: {"prompt": "p"}, run_key="k",
                           target_root=str(leaf))()
    for event in job:
        if event.get("text") == "working":
            break
    end = overlay.RunEnd.read(leaf)
    assert (end.session_id, end.success) == ("sess-live", False)
    assert overlay.has_resumable_draft(leaf)


def test_a_crashed_run_is_recorded_unfinished(tmp_path, monkeypatch):
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    overlay.RunEnd(success=True, message="", timed_out=False, session_id="old",
                   agent_cli="claude", ended_at=1.0).write(leaf)

    def fake_stream(model_id, prep_result, *, run_key=None, failover=None):
        prep_result.update(_session_id="sess-c", _agent_cli="claude")
        yield 'data: {"type": "log", "text": "working"}\n\n'
        raise RuntimeError("boom")

    monkeypatch.setattr(W, "stream_agent", fake_stream)
    with pytest.raises(RuntimeError):
        list(W.generation_job(MODEL, lambda: {"prompt": "p"}, run_key="k",
                              target_root=str(leaf))())
    end = overlay.RunEnd.read(leaf)
    assert (end.session_id, end.success) == ("sess-c", False)


def test_a_successful_done_carries_no_resumable_flag(tmp_path, monkeypatch):
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    ok = {"type": "done", "success": True}
    assert _job_events(monkeypatch, str(leaf), ok)[-1] == ok
    assert overlay.RunEnd.read(leaf).success is True


def test_a_failed_prep_on_a_continued_draft_is_resumable(tmp_path, monkeypatch):
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    monkeypatch.setattr(W, "release", lambda key: None)

    def prep():
        raise OSError("hf down")

    done = _job_events(monkeypatch, str(leaf), {}, prep=prep)[-1]
    assert done["success"] is False and done["resumable"] is True
    # No agent ran: the earlier run's session stays on record.
    assert overlay.RunEnd.read(leaf).session_id == "sess-1"


def test_a_refine_leaf_is_never_marked_resumable(tmp_path, monkeypatch):
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    refine = overlay.new_run_overlay("user-a", MODEL, kind="refine")
    (refine / "models" / "meta__Foo.yaml").write_text("match: {}\n", encoding="utf-8")
    failed = {"type": "done", "success": False, "message": "x"}
    assert _job_events(monkeypatch, str(refine), failed)[-1] == failed
    assert overlay.RunEnd.read(refine) is None


def test_service_resume_without_a_draft_refuses_and_refunds(tmp_path, monkeypatch):
    from backend.service.routes import spec as routes
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    released, refunded, started = [], [], []
    monkeypatch.setattr(routes, "_peruser_active", lambda: True)
    monkeypatch.setattr(routes.spec_worker, "acquire", lambda key: "ok")
    monkeypatch.setattr(routes.spec_worker, "release", released.append)
    monkeypatch.setattr(routes.spec_worker, "start_job",
                        lambda *a, **k: started.append(a))
    monkeypatch.setattr(routes, "_quota_room", lambda user_id: True)
    monkeypatch.setattr(routes, "_quota_ok", lambda user_id: True)
    monkeypatch.setattr(routes, "_quota_refund", refunded.append)
    monkeypatch.setattr(routes._creds, "resolve_env",
                        lambda uid, **kw: {"agent_cli": "claude", "env": {}})
    monkeypatch.setattr(routes.spec_worker, "get_agent_binary", lambda: "sh")
    monkeypatch.setattr(routes._auth, "is_admin", lambda uid: False)
    monkeypatch.setattr(routes.preflight, "probe_repo", lambda mid: (None, None))
    out = routes.service_generate_spec(
        routes.GenerateSpecRequest(model_id=MODEL, resume=True), user={"sub": "u"})
    assert out.status_code == 409
    assert json.loads(out.body)["status"] == "no_draft"
    assert released == ["u|meta/Foo"] and refunded == ["u"] and started == []


def test_service_resume_runs_the_continuation_prep(tmp_path, monkeypatch):
    from backend.service.routes import spec as routes
    leaf = _failed_generate_leaf(tmp_path, monkeypatch)
    # The leaf belongs to the signed-in user "user-a".
    captured = {}
    monkeypatch.setattr(routes, "_peruser_active", lambda: True)
    monkeypatch.setattr(routes.spec_worker, "acquire", lambda key: "ok")
    monkeypatch.setattr(routes, "_quota_room", lambda user_id: True)
    monkeypatch.setattr(routes, "_quota_ok", lambda user_id: True)
    monkeypatch.setattr(routes._creds, "resolve_env",
                        lambda uid, *a, **kw: {"agent_cli": "claude", "env": {}})
    monkeypatch.setattr(routes.spec_worker, "get_agent_binary", lambda: "sh")
    monkeypatch.setattr(routes._auth, "is_admin", lambda uid: False)
    monkeypatch.setattr(routes.preflight, "probe_repo", lambda mid: (None, None))
    monkeypatch.setattr(routes.spec_worker, "prepare_generate_prompt",
                        lambda mid, target_root=None: {"prompt": "ORIGINAL",
                                                       "target_root": target_root})
    monkeypatch.setattr(routes.spec_worker, "prepare_resume",
                        lambda mid, **kw: captured.update(kw, base=kw["base_prep"]()) or {})

    def fake_generation_job(model_id, prep_fn, **kw):
        captured["target_root"] = kw["target_root"]
        captured["prep_fn"] = prep_fn
        captured["failover"] = kw["failover"]
        return lambda: iter(())

    monkeypatch.setattr(routes.spec_worker, "generation_job", fake_generation_job)
    monkeypatch.setattr(routes.spec_worker, "start_job", lambda *a, **k: None)
    out = routes.service_generate_spec(
        routes.GenerateSpecRequest(model_id=MODEL, resume=True),
        user={"sub": "user-a"})
    assert out["status"] == "started"
    assert captured["target_root"] == str(leaf)
    prep = captured["prep_fn"]()
    assert captured["agent_cli"] == "claude"
    assert captured["base"]["prompt"] == "ORIGINAL"
    # The run may fail over to the user's own keys for the other CLIs, and a
    # cold start there takes the original task prep, not the continuation.
    assert prep["agent_fallbacks"] == {"codex": {}, "kimi": {}}
    assert captured["failover"]._task_prep()["prompt"] == "ORIGINAL"
    # The draft survived the allocation.
    assert (leaf / "models" / "meta__Foo.yaml").is_file()


def test_core_resume_without_a_draft_refuses_and_releases(host_credentials, tmp_path, monkeypatch):
    import asyncio
    from backend.api.routes import spec as core
    from backend.api._models import GenerateSpecRequest
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    released, started = [], []
    monkeypatch.setattr(core.supabase_env, "jwt_verification_available", lambda: False)
    monkeypatch.setattr(core.preflight, "agent_cli_refusal", lambda: None)
    monkeypatch.setattr(core.preflight, "probe_repo", lambda mid: (None, None))
    monkeypatch.setattr(core.spec_worker, "acquire", lambda key: "ok")
    monkeypatch.setattr(core.spec_worker, "release", released.append)
    monkeypatch.setattr(core.spec_worker, "start_job",
                        lambda *a, **k: started.append(a))
    out = asyncio.run(core.generate_spec(
        GenerateSpecRequest(model_id=MODEL, resume=True)))
    assert out.status_code == 409
    assert json.loads(out.body)["status"] == "no_draft"
    assert released == [MODEL] and started == []


@pytest.mark.parametrize("status,expect", [
    ({"state": "interrupted", "success": False}, True),
    ({"state": "idle"}, True),
    ({"state": "done", "success": False}, True),
    ({"state": "done", "success": True}, None),
    ({"state": "running", "success": None}, None),
])
def test_the_status_states_a_kept_draft_whenever_no_run_is_live(
        tmp_path, monkeypatch, status, expect):
    from backend.api import _spec_preflight as preflight
    _failed_generate_leaf(tmp_path, monkeypatch)
    out = preflight.with_resumable(status, MODEL, "generate", user_id="user-a")
    assert out.get("resumable") is expect
    assert "resumable" not in preflight.with_resumable(
        status, MODEL, "refine", user_id="user-a")
    # Another user's leaf is not this caller's draft.
    if expect:
        assert preflight.with_resumable(
            status, MODEL, "generate", user_id="user-b")["resumable"] is False


def test_service_status_offers_continue_after_an_interrupt(tmp_path, monkeypatch):
    from backend.service.routes import spec as routes
    _failed_generate_leaf(tmp_path, monkeypatch)
    monkeypatch.setattr(routes, "_peruser_active", lambda: True)
    monkeypatch.setattr(routes.spec_worker, "job_status", lambda key, cursor: {
        "state": "interrupted", "success": False,
        "message": "Interrupted by a server restart."})
    out = routes.service_spec_status(MODEL, user={"sub": "user-a"})
    assert out["state"] == "interrupted" and out["resumable"] is True


def test_core_new_generate_over_a_kept_draft_needs_start_over(host_credentials, tmp_path, monkeypatch):
    import asyncio
    from backend.api.routes import spec as core
    from backend.api._models import GenerateSpecRequest
    monkeypatch.setenv("HA_SPEC_RUNS_ROOT", str(tmp_path / "runs"))
    leaf = overlay.new_run_overlay(overlay.ANONYMOUS_OWNER, MODEL)
    (leaf / "components" / "drafts" / "foo_attn.yaml").write_text(
        "role: attention\n", encoding="utf-8")
    released, started = [], []
    monkeypatch.setattr(core.supabase_env, "jwt_verification_available", lambda: False)
    monkeypatch.setattr(core.preflight, "agent_cli_refusal", lambda: None)
    monkeypatch.setattr(core.preflight, "probe_repo", lambda mid: (None, None))
    monkeypatch.setattr(core.spec_worker, "acquire", lambda key: "ok")
    monkeypatch.setattr(core.spec_worker, "release", released.append)
    monkeypatch.setattr(core.spec_worker, "start_job",
                        lambda *a, **k: started.append(a))
    out = asyncio.run(core.generate_spec(GenerateSpecRequest(model_id=MODEL)))
    body = json.loads(out.body)
    assert out.status_code == 409
    assert body["status"] == "draft_kept" and body["resumable"] is True
    assert released == [MODEL] and started == []
    assert (leaf / "components" / "drafts" / "foo_attn.yaml").is_file()

    out = asyncio.run(core.generate_spec(
        GenerateSpecRequest(model_id=MODEL, discard_draft=True)))
    assert out["status"] == "started" and len(started) == 1
    assert not (leaf / "components" / "drafts" / "foo_attn.yaml").exists()


def test_claude_finds_a_session_by_its_exact_name(tmp_path):
    from backend.agents.claude import ClaudeBackend
    backend = ClaudeBackend()
    projects = tmp_path / ".agent-home" / ".claude" / "projects" / "-leaf"
    projects.mkdir(parents=True)
    (projects / "sess-1.jsonl").write_text("{}\n", encoding="utf-8")
    assert backend.has_session("sess-1", str(tmp_path))
    assert not backend.has_session("sess-2", str(tmp_path))
    assert not backend.has_session("sess*", str(tmp_path))
    assert not backend.has_session("*", str(tmp_path))
    assert not backend.has_session("../-leaf/sess-1", str(tmp_path))
    assert not backend.has_session("", str(tmp_path))
    assert not backend.has_session("sess-1", str(tmp_path / "other"))
