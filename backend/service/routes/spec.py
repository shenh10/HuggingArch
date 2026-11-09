"""Service-layer spec-generation route (Phase 1 of multi-user spec-gen).

  POST /api/spec/generate  — kick off an attributed spec run for a signed-in
                             user. Returns {status:"started"}; the browser
                             then polls GET /api/spec/status, which derives
                             the private run key from the caller's JWT.

Why a separate route from the core `POST /generate-spec`
(`backend/api/routes/spec.py`): the core route is anonymous and carries no
user identity, and the core↔service split (tests/meta/test_service_layout.py)
forbids the core importing auth. This route lives in the service layer, gates
on the caller's JWT, and threads their id (`sub`) into the run so the
post-success hook can attribute a `spec_generations` record to them.

Gate: any signed-in user, but a non-admin must bring their own key (BYOK) —
they can't spend the host's tokens. Untrusted runs are hardened: the agent
gets a scrubbed env (no server secrets), drops to an unprivileged OS user in
the container (HA_AGENT_USER), and each user is capped to a per-hour quota.
Full microVM isolation is deliberately not used (see the plan); these
process-level controls turn "trivial total breach" into "bounded abuse."

Progress polling uses the service `/api/spec/status` route. The raw core
`/spec-status` route remains available to local self-hosts and administrators,
but regular service users never supply or read opaque run keys directly.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend import spec_worker, supabase_env
from backend.arch.loader import SpecLoadError, get_default_loader
from backend.api import _spec_preflight as preflight
from backend.api._app import app
from backend.api.offload import offload_response
from backend.api._models import AnalyzeRequest, RefineSpecRequest, StopSpecRequest
from backend.service import auth as _auth
from backend.service import agent_credentials as _creds
from backend.service import admission as _adm
from backend.service import spec_generation_reviews as _reviews
from backend.service.record_recheck import RecordRecheck
from backend.service.spec_pr import DryRunAdmission, SpecPRJob
from backend.spec_worker.bundle import SpecBundleError, normalize_error_payload
from backend.spec_worker import overlay as _overlay
from backend.spec_worker import record_jobs
# The record jobs' shared orchestration, re-exported under the names the
# account / admin routes and their tests already import from here. One
# definition each; the route layer only adds its quota + HTTP translation.
from backend.spec_worker.record_jobs import (  # noqa: F401 — re-export
    RecordJob,
    fix_run_key,
    fix_seed_bundle,
    pending_refusal,
    record_bundle as _normalize_record_bundle,
    review_run_key,
)
from backend.spec_worker.outcome import (
    PENDING,
    GenerationStatus,
    RunErrorKind,
    vocabulary,
)
from backend.arch.validate._analysable import (
    ENGINE_KIND,
    SPEC_KIND,
    Refusal,
    refusal_finding,
)
from backend.fingerprint import engine_version


# Per-user generation quota: an in-memory sliding window of start times,
# capping how many runs a single user can kick off per hour. Bounds the
# abuse surface (a user can't enqueue 100 junk models); the per-VM semaphore
# already caps *concurrency*. Set SPEC_GEN_HOURLY_QUOTA=0 to disable.
_QUOTA_MAX = int(os.environ.get("SPEC_GEN_HOURLY_QUOTA", "10") or "10")
_QUOTA_WINDOW = 3600.0
_quota_lock = threading.Lock()
_quota_hits: dict[str, deque] = {}


def _quota_room(user_id: str) -> bool:
    """True if the user is under their hourly cap. Records nothing — the early
    check before any lock or HF work; `_quota_ok` records the start once the
    run has its lock."""
    if _QUOTA_MAX <= 0:
        return True
    cutoff = time.time() - _QUOTA_WINDOW
    with _quota_lock:
        return sum(1 for t in _quota_hits.get(user_id, ()) if t >= cutoff) < _QUOTA_MAX


def _quota_ok(user_id: str) -> bool:
    """True if the user is under their hourly cap; records this start if so."""
    if _QUOTA_MAX <= 0:
        return True
    now = time.time()
    cutoff = now - _QUOTA_WINDOW
    with _quota_lock:
        dq = _quota_hits.setdefault(user_id, deque())
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= _QUOTA_MAX:
            return False
        dq.append(now)
        return True


def _quota_refund(user_id: str) -> None:
    """Give back the start `_quota_ok` just recorded (the run did not start)."""
    with _quota_lock:
        dq = _quota_hits.get(user_id)
        if dq:
            dq.pop()


def _peruser_active() -> bool:
    """Per-user private spec-gen is active when explicitly enabled via
    `HA_PERUSER_SPECGEN`, or — BY DEFAULT — on any MULTI-USER deployment (auth
    configured). Self-host / single-user (no auth) uses the reserved
    `__anonymous__` overlay owner (kind-isolated leaves, then land to
    shared tree / cached_specs). Explicit 0/1 wins.

    Computed per-request (not a global env mutation) so it can't leak across
    processes/tests; generate/refine always get a `target_root`, and this
    policy only chooses the overlay owner (user vs `__anonymous__`)."""
    v = os.environ.get("HA_PERUSER_SPECGEN", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return supabase_env.jwt_verification_available()


class GenerateSpecRequest(BaseModel):
    model_id: str
    # Continue the draft an earlier failed generation left, instead of
    # starting over (`spec_worker.prepare_resume`).
    resume: bool = False
    # Start over even though an unfinished run kept a draft; without it such
    # a request is refused with 409 `draft_kept`.
    discard_draft: bool = False


def _bundle_failure(exc: BaseException, *, model_id: str) -> dict:
    """Shared adapter: SpecBundleError → spec_bundle_incomplete, SpecLoadError → spec_invalid."""
    return normalize_error_payload(exc, model_id=model_id)


def _run_stamp(user_id: str, agent_cli=None, agent_env=None) -> dict:
    """What a service run adds to `prep_result`: the requesting user (so the
    post-success hook writes an attributed `spec_generations` record) and, for
    a BYOK run, the chosen agent and its decrypted credential env (both absent
    → the host's configured agent and allowlisted auth)."""
    stamp = {"user_id": user_id}
    if agent_cli:
        stamp["agent_cli"] = agent_cli
    if agent_env:
        stamp["agent_env"] = agent_env
    return stamp


def _claim_run(run_key: str, model_id: str, user_id: str,
               is_admin: bool) -> dict | None:
    """Take the run lock, then spend the user's quota; the refusal response,
    or None once the run is theirs. The quota is spent only for a run that
    starts: a duplicate or overloaded click costs nothing."""
    refused = preflight.claim(run_key, model_id)
    if refused:
        return refused
    if not is_admin and not _quota_ok(user_id):
        spec_worker.release(run_key)
        return {"success": False, "status": "quota_exceeded",
                "message": f"Generation limit reached ({_QUOTA_MAX}/hour). Try again later."}
    return None


def _preflight(user_id: str, is_admin: bool, verb: str) -> tuple[dict | None, dict | None]:
    """The checks a signed-in generate/refine makes before any lock or HF
    work: `(resolved credentials, None)` to go on, `(None, refusal)` to stop.

    The core route's agent-CLI check, then the BYOK gate: resolve the
    caller's own agent + credential env, with the model the credential pins
    for this run (`verb` is its task key: generate / refine). A non-admin with no stored key is
    rejected — the run spends the user's key, not the host's. An admin with
    no stored key may run with the selected host agent's allowlisted
    credentials (resolved None) when the host has them; a stored key that
    cannot be decrypted is refused for everyone. Last, the per-user
    hourly quota (admins exempt), checked without recording a start."""
    refused = preflight.agent_cli_refusal()
    if refused:
        return None, refused
    try:
        resolved = _creds.resolve_env(user_id, task=verb)
    except _creds.CredentialDecryptError:
        # Stored but undecryptable: the truth (re-enter), not the misleading
        # "add a key" — and never the host's credentials instead, admin or
        # not: the caller chose their own key.
        return None, {
            "success": False, "status": "credential_undecryptable",
            "message": "Your saved agent API key could not be decrypted — please re-enter it in account settings.",
        }
    except _creds.CredentialRefusal as exc:
        # None of the caller's keys can be chosen, or they could not be
        # read: never the host's instead.
        return None, exc.refusal(verb)
    if resolved is None and not is_admin:
        return None, {
            "success": False, "status": "no_credentials",
            "message": f"Add your own agent API key in account settings to {verb} specs.",
        }
    if resolved is None:
        refused = preflight.host_credentials_refusal()
        if refused:
            return None, refused
    if not is_admin and not _quota_room(user_id):
        return None, {
            "success": False, "status": "quota_exceeded",
            "message": f"Generation limit reached ({_QUOTA_MAX}/hour). Try again later.",
        }
    return resolved, None


@app.post("/api/spec/generate")
def service_generate_spec(
    req: GenerateSpecRequest,
    user: dict = Depends(_auth.require_user),
):
    """Trigger an attributed spec-gen run for a not-in-library model.

    Open to any signed-in user, but a non-admin MUST bring their own key
    (BYOK) — they can't spend the host's tokens. The user's key is injected
    into a SCRUBBED subprocess env (AgentBackend._merge_env), so a run against
    a malicious HF repo can't exfiltrate server secrets. Admins may fall back
    to the host's own agent credentials when they haven't configured a key.

    Same pre-flight as the core route (agent-CLI present, spec not already
    known), then acquire the per-model lock and start the background job. The
    POST returns immediately; the browser polls GET /api/spec/status?model_id=...
    """
    user_id = user.get("sub")
    is_admin = _auth.is_admin(user_id)

    resolved, refused = _preflight(user_id, is_admin, "generate")
    if refused:
        return refused

    # Already-supported model → nothing to generate. The core route's
    # HF config lookup + SpecLoader.match short-circuit.
    config, refused = preflight.probe_repo(req.model_id)
    if refused:
        return refused
    cached = preflight.known_spec(req.model_id, config=config)
    if cached:
        return cached

    byok_cli = resolved["agent_cli"] if resolved else None
    byok_env = resolved["env"] if resolved else None

    # Per-user isolation (opt-in via HA_PERUSER_SPECGEN). Re-key the run to
    # (user_id, model_id) so users don't block each other, and redirect the
    # agent's writes to a private overlay so the generation stays out of the
    # shared tree + global cached_specs. Off → the same kind-isolated overlay
    # under the reserved `__anonymous__` owner (S4 §2.5), not the shared tree.
    peruser = _peruser_active()
    run_key = preflight.job_run_key(
        req.model_id, kind="generate", user_id=user_id if peruser else "")
    refused = _claim_run(run_key, req.model_id, user_id, is_admin)
    if refused:
        return refused
    # The overlay is prepared only once this run holds the lock: preparing it
    # clears the leaf's agent home and tmp, which a running agent is using.
    try:
        target_root = _overlay.allocate_run_overlay(
            req.model_id, kind="generate",
            user_id=user_id if peruser else "", resume=req.resume,
            discard_draft=req.discard_draft)
    except _overlay.RunLeafRefusal as exc:
        spec_worker.release(run_key)
        if not is_admin:
            _quota_refund(user_id)
        return JSONResponse(status_code=409, content=exc.refusal())
    except BaseException:
        spec_worker.release(run_key)
        if not is_admin:
            _quota_refund(user_id)
        raise

    run_cli = byok_cli or spec_worker.get_agent_cli()
    task_prep = lambda: spec_worker.prepare_generate_prompt(
        req.model_id, target_root=target_root)
    run_prep = task_prep
    if req.resume:
        run_prep = lambda: spec_worker.prepare_resume(
            req.model_id, target_root=target_root, agent_cli=run_cli,
            base_prep=task_prep)

    def prep_fn() -> dict:
        # In the job thread, off the request path: the backends this run may
        # fail over to, with credentials resolved as its own were.
        prep = run_prep()
        prep["agent_fallbacks"] = _creds.fallback_envs(
            user_id, run_cli, task="generate", host_allowed=is_admin)
        return prep

    spec_worker.start_job(
        run_key,
        spec_worker.generation_job(req.model_id, prep_fn, run_key=run_key,
                                   stamp=_run_stamp(user_id, byok_cli, byok_env),
                                   target_root=target_root,
                                   failover=spec_worker.Failover(
                                       req.model_id, target_root=target_root,
                                       task_prep=task_prep if req.resume else None)),
        model_id=req.model_id, user_id=user_id, kind="generate",
        agent_cli=byok_cli, agent_env=byok_env,
    )
    # `run_key` is this user's job key: GET /api/spec/status rebuilds it from
    # the JWT, and the admin-only raw /spec-status accepts it directly.
    return {"status": "started", "model_id": req.model_id, "run_key": run_key}


def _resolved_spec_yaml(model_id: str):
    """Source YAML of the model's currently-resolved (repo/disk/custom) spec.
    Shared with core so generate/refine seed the same overlay identity file."""
    return _overlay.load_seed_spec_yaml(model_id)


@app.post("/api/spec/refine")
def service_refine_spec(
    req: RefineSpecRequest,
    user: dict = Depends(_auth.require_user),
):
    """Refine a signed-in user's OWN generated spec (BYOK), symmetric to
    /api/spec/generate. The user's latest spec_generations spec is seeded into a
    private run overlay so the agent reads + edits THAT (never the shared tree),
    and the refined result harvests back to a new spec_generations row. Refining
    a public/repo spec is intentionally not offered here — global specs change
    only via PR.

    Only reached when the user explicitly clicks Refine: analyze never
    auto-fires this, so a run (which spends the user's tokens) is always opt-in.
    """
    user_id = user.get("sub")
    is_admin = _auth.is_admin(user_id)

    # A private refinement of a library model would be archived the moment it
    # lands: analyze serves the library spec, never it. Refused before any
    # credential or quota check, so the answer names the real reason.
    peruser = _peruser_active()
    if peruser:
        library_spec = library_spec_for(req.model_id)
        if library_spec:
            return superseded_refusal(req.model_id, library_spec, "Refine")

    resolved, refused = _preflight(user_id, is_admin, "refine")
    if refused:
        return refused

    # Isolation mirrors generate's: every refine gets its own kind leaf.
    # Per-user run → private overlay seeded with the user's own generation
    # (else a fork of the resolved repo/disk spec). Per-user-off / self-host
    # → `__anonymous__/refine__*` seeded from the shared/custom spec, never
    # the in-flight generate leaf.
    seed_yaml = None
    seed_comps = None
    seed_bundle = None
    if peruser:
        from backend.service import spec_generations
        try:
            row = spec_generations.serving_generation(user_id, req.model_id)
        except spec_generations.SpecGenerationsUnavailable:
            return _records_unavailable()
        if row and row.get("spec_yaml") and row.get("status") in spec_generations.SERVED:
            seed_yaml, seed_comps = row["spec_yaml"], row.get("new_component_yamls")
        else:
            # No per-user generation — fork the model's currently-resolved (repo
            # or local-disk) spec so the user refines a PRIVATE copy. Covers the
            # common case where the served spec came from disk, not the user's
            # own generation.
            seed_yaml = _resolved_spec_yaml(req.model_id)
        run_key = preflight.job_run_key(
            req.model_id, kind="refine", user_id=user_id)
    else:
        seed_yaml = _resolved_spec_yaml(req.model_id)
        run_key = preflight.job_run_key(req.model_id, kind="refine")
    if not seed_yaml:
        return preflight.no_seed_refusal()

    try:
        seed_bundle = _normalize_record_bundle(
            req.model_id, seed_yaml, seed_comps)
    except (SpecBundleError, SpecLoadError) as exc:
        return _bundle_failure(exc, model_id=req.model_id)

    refused = _claim_run(run_key, req.model_id, user_id, is_admin)
    if refused:
        return refused
    # Seeding writes the leaf the running agent edits: only with the lock.
    try:
        target_root = _overlay.allocate_run_overlay(
            req.model_id, kind="refine",
            user_id=user_id if peruser else "",
            seed_bundle=seed_bundle)
    except BaseException:
        spec_worker.release(run_key)
        if not is_admin:
            _quota_refund(user_id)
        raise

    byok_cli = resolved["agent_cli"] if resolved else None
    byok_env = resolved["env"] if resolved else None
    prep_fn = lambda: spec_worker.prepare_refine_prompt(
        req.model_id, req.spec_storage_bytes, req.hf_storage_bytes,
        req.storage_diff, target_root=target_root, seed_bundle=seed_bundle)
    spec_worker.start_job(
        run_key,
        spec_worker.generation_job(req.model_id, prep_fn, run_key=run_key,
                                   stamp=_run_stamp(user_id, byok_cli, byok_env),
                                   target_root=target_root),
        model_id=req.model_id, user_id=user_id, kind="refine",
        agent_cli=byok_cli, agent_env=byok_env,
    )
    return {"status": "started", "model_id": req.model_id, "run_key": run_key}


@app.get("/api/spec/status")
def service_spec_status(
    model_id: str,
    cursor: int = 0,
    kind: Literal["generate", "refine"] = "generate",
    user: dict = Depends(_auth.require_user),
):
    """Reattach to a signed-in user's in-flight run after a page reload, when
    the browser no longer holds the `run_key`. Rebuilds the composite key from
    the caller's JWT (peruser mode) or falls back to `model_id`. `kind`
    selects generate vs refine (they no longer share a lock)."""
    user_id = user.get("sub")
    owner = user_id if (_peruser_active() and user_id) else ""
    run_key = preflight.job_run_key(model_id, kind=kind, user_id=owner)
    return preflight.with_resumable(
        spec_worker.job_status(run_key, cursor), model_id, kind, user_id=owner)


@app.post("/api/spec/stop")
def service_stop_spec(
    req: StopSpecRequest,
    user: dict = Depends(_auth.require_user),
):
    """Stop a signed-in user's in-flight run, rebuilding the per-user run_key
    from the JWT (mirrors the core /stop-spec for the anonymous path).
    Review/fix jobs use `record_id`; generate/refine use `model_id`."""
    user_id = user.get("sub")
    if req.kind in ("review", "fix"):
        if not req.record_id:
            raise HTTPException(status_code=422, detail="record_id required to stop a review or fix")
        run_key = (review_run_key(user_id, req.record_id) if req.kind == "review"
                   else fix_run_key(user_id, req.record_id))
    else:
        run_key = preflight.job_run_key(
            req.model_id, kind=req.kind,
            user_id=user_id if (_peruser_active() and user_id) else "")
    killed = spec_worker.stop(run_key)
    return {"success": killed,
            "message": "Stopped" if killed else "No running task found"}


def _busy_or(started: dict) -> dict:
    """A record job's answer, with the server's own `overloaded` raised as a
    503 — the one translation an HTTP caller needs."""
    if started.get("status") == "overloaded":
        raise HTTPException(status_code=503, detail=started["message"])
    return started


def launch_review(record: dict, *, reviewer_id: str, reviewer_role: str,
                  agent_cli=None, agent_env=None) -> dict:
    """Start an async agent review of a generation record. Shared by the user
    (account) and admin routes; each supplies its own auth + resolved creds.

    The orchestration is `record_jobs.launch_review`, shared with a local
    run whose record has no database behind it. This adds what only a
    service run has: the per-user hourly quota, a 503 for an overloaded
    server, and the dry-run admission a PASS starts."""
    if record.get("status") in PENDING:
        return pending_refusal(record, "Review")

    # A review is a full agent run — same window as generate/refine so a user
    # can't parallel-fill the agent slots across different records or spam.
    # Admins are exempt (they may also fall back to host creds).
    if reviewer_role == "user" and not _quota_ok(reviewer_id):
        return {"status": "quota_exceeded",
                "message": f"Review limit reached ({_QUOTA_MAX}/hour). Try again later."}

    # A PASS starts the record's admission. It never changes the verdict:
    # the review row is already written, and the admission only writes its
    # own column on it.
    return _busy_or(record_jobs.launch_review(
        record, reviewer_id=reviewer_id, reviewer_role=reviewer_role,
        agent_cli=agent_cli, agent_env=agent_env,
        on_pass=lambda row: launch_admission(record, review_id=row["id"])))


def mark_engine_changed(records: list) -> None:
    """`engine_changed` on every engine-blocked record: the engine this
    server runs differs from the one that blocked it, so a re-check may now
    pass."""
    blocked = [rec for rec in records
               if rec.get("status") == GenerationStatus.BLOCKED_BY_ENGINE]
    if not blocked:
        return
    current = engine_version()
    for rec in blocked:
        rec["engine_changed"] = rec.get("blocked_engine_version") != current


def recheck_run_key(actor_id: str, record_id: str) -> str:
    return RecordJob.RECHECK.run_key(actor_id, record_id)


def launch_recheck(record: dict, *, actor_id: str) -> dict:
    """Start a deterministic re-check of a pending record (`RecordRecheck`)
    as `actor_id` (owner or admin). The key only de-duplicates re-checks of
    one record; correctness against every other writer is the revision
    condition on the write. Poll `/api/spec/recheck-status`."""
    if record.get("status") not in PENDING:
        return {"status": "not_pending", "message": (
            "Only an engine-blocked record or one that needs repair can be "
            "re-checked.")}
    run_key = recheck_run_key(actor_id, record["id"])
    status = spec_worker.acquire(run_key)
    if status == "duplicate":
        return {"status": "already_running", "run_key": run_key}
    if status == "overloaded":
        raise HTTPException(status_code=503,
                            detail="Server busy — try the re-check again shortly.")
    job = RecordRecheck(record["id"], actor_id=actor_id)
    _record_job(run_key, {**record, "user_id": actor_id}, job.run)
    return {"status": "started", "run_key": run_key}


@app.get("/api/spec/recheck-status")
def service_recheck_status(
    record_id: str,
    cursor: int = 0,
    user: dict = Depends(_auth.require_user),
):
    """Poll a re-check this caller (owner or admin) started. Same shape as
    /api/spec/status; the `done` event carries `recheck` (`RecheckOutcome`)."""
    user_id = _auth.uid(user, detail="No user_id in token")
    return spec_worker.job_status(recheck_run_key(user_id, record_id), cursor)


@app.get("/api/spec/generation-vocabulary")
def service_generation_vocabulary():
    """The closed vocabularies a generation record and a run end in
    (`backend.spec_worker.outcome`): the frontend builds its filters and
    badge keys from these values."""
    return vocabulary()


def admission_run_key(record: dict) -> str:
    """One admission per record, keyed by its owner."""
    return RecordJob.ADMISSION.run_key(record.get("user_id") or "", record["id"])


def pr_run_key(record: dict) -> str:
    """One Open PR / update-pr per record at a time."""
    return RecordJob.PR.run_key(record.get("user_id") or "", record["id"])


def _record_job(run_key: str, record: dict, body) -> None:
    """Run `body()` (a generator of events) as a record job holding
    `run_key`, and release it when the job ends."""
    def _run_fn():
        try:
            yield from body()
        finally:
            spec_worker.release(run_key)

    spec_worker.start_job(run_key, _run_fn, model_id=record["model_id"],
                          user_id=record.get("user_id"),
                          kind=run_key.split("|", 1)[0])


def launch_admission(record: dict, *, review_id: str) -> dict:
    """Start the dry-run admission of `record` for its PASS review
    `review_id` (ADMIT §4.1). The report lands on that review row."""
    run_key = admission_run_key(record)
    status = spec_worker.acquire(run_key)
    if status != "ok":
        if status == "overloaded":
            _reviews.set_admission(review_id, _adm.envelope(
                "failed", mode=_adm.DRY_RUN,
                error="the server was busy; run the admission again"))
        return {"status": "already_running" if status == "duplicate" else "overloaded",
                "run_key": run_key}
    job = DryRunAdmission(record, review_id, run_key=run_key)
    _record_job(run_key, record, job.run)
    return {"status": "started", "run_key": run_key}


def launch_spec_pr(record: dict, review: dict, *, update: bool,
                   admin_override: bool = False, contributor: Optional[str] = None,
                   note: str = "") -> dict:
    """Start Open PR (or update-pr) for `record` on its PASS review `review`
    (ADMIT §4.3). Poll `/api/account/generations/{id}/pr-status`."""
    run_key = pr_run_key(record)
    status = spec_worker.acquire(run_key)
    if status == "duplicate":
        return {"status": "already_running", "run_key": run_key}
    if status == "overloaded":
        raise HTTPException(status_code=503, detail="Server busy — try again shortly.")
    job = SpecPRJob(record, review, owner_id=record.get("user_id") or "", update=update,
                    admin_override=admin_override, contributor=contributor, note=note,
                    run_key=run_key)
    _record_job(run_key, record, job.run)
    return {"status": "started", "run_key": run_key}


@app.get("/api/spec/review-status")
def service_review_status(
    record_id: str,
    cursor: int = 0,
    user: dict = Depends(_auth.require_user),
):
    """Poll a review run started by this caller (user OR admin). The run_key is
    rebuilt from the caller's JWT `sub` + record_id, so each reviewer polls
    their own review. Same response shape as /api/spec/status."""
    user_id = _auth.uid(user, detail="No user_id in token")
    return spec_worker.job_status(review_run_key(user_id, record_id), cursor)


def launch_fix(record: dict, *, fixer_id: str, fixer_role: str,
               review_report: str, agent_cli=None, agent_env=None) -> dict:
    """Start an async agent FIX of a generation record: the agent repairs the
    defects named in `review_report` (the latest NO_PASS review; empty when
    the caller opened Fix on a record's load refusals), editing only this
    record's own draft, then the fixed bundle is written back to the record.
    Shared by the user (account) and admin routes.

    The orchestration is `record_jobs.launch_fix`, shared with a local run;
    this adds the per-user hourly quota and the 503."""
    if record.get("status") == GenerationStatus.BLOCKED_BY_ENGINE:
        return pending_refusal(record, "Agent Fix")

    if fixer_role == "user" and not _quota_ok(fixer_id):
        return {"status": "quota_exceeded",
                "message": f"Fix limit reached ({_QUOTA_MAX}/hour). Try again later."}

    return _busy_or(record_jobs.launch_fix(
        record, fixer_id=fixer_id, fixer_role=fixer_role,
        review_report=review_report, agent_cli=agent_cli, agent_env=agent_env))


@app.get("/api/spec/fix-status")
def service_fix_status(
    record_id: str,
    cursor: int = 0,
    user: dict = Depends(_auth.require_user),
):
    """Poll a fix run started by this caller (user OR admin). run_key is rebuilt
    from the caller's JWT `sub` + record_id. Same shape as /api/spec/status."""
    user_id = _auth.uid(user, detail="No user_id in token")
    return spec_worker.job_status(fix_run_key(user_id, record_id), cursor)


def library_spec_for(model_id: str) -> str | None:
    """Name of the library spec `service_analyze` serves for `model_id`, or
    None when analyze would fall through to the caller's private record.

    The lookup is analyze's own: the shared default loader's
    `spec_for_model_id` (the identity half of `SpecLoader.match`). That is
    the repo tree, the runtime custom-model cache, and whatever the loader's
    spec-miss hook attaches — the global generated-spec cache, which the hook
    only answers on a single-user deployment; on a multi-user deployment the
    library is the repo tree. A library spec that claims the id but fails to
    load still counts: analyze answers it with a spec error and never falls
    through to the private record, so it is named by its file."""
    try:
        spec = get_default_loader().spec_for_model_id(model_id)
    except SpecLoadError as err:
        return Path(err.path).stem if err.path else model_id
    return spec.name if spec is not None else None


def library_standing(record: dict) -> dict:
    """A user's generation record against the library, as the two fields the
    account and admin views carry: `library_spec`, the library spec analyze
    serves for the record's model (None when it serves the record), and
    `superseded_by`, that spec unless the record is `merged` — a merged
    record is where its library spec came from, not superseded. Either way a
    record with a `library_spec` is not what the product uses, so its own
    review verdict is history, not the model's standing."""
    return _standing(record, library_spec_for(record.get("model_id") or ""))


def add_library_standing(records: list) -> None:
    """`library_standing` on every record of a list, in place, with one
    library lookup per distinct model id."""
    library = {model_id: library_spec_for(model_id)
               for model_id in {r.get("model_id") or "" for r in records}}
    for record in records:
        record.update(_standing(record, library[record.get("model_id") or ""]))


def _standing(record: dict, library_spec: str | None) -> dict:
    return {
        "library_spec": library_spec,
        "superseded_by": None if record.get("status") == "merged" else library_spec,
    }


def superseded_by(record: dict) -> str | None:
    """The library spec that supersedes a user's generation record, or None.
    Analyze never serves a record whose model has a library spec, so the
    record is archived: listed apart, with no Review / Fix / regenerate."""
    return library_standing(record)["superseded_by"]


def superseded_refusal(model_id: str, library_spec: str, action: str) -> dict:
    """The answer to `action` on a record (or model) a library spec
    supersedes."""
    return {
        "success": False, "status": "superseded", "superseded_by": library_spec,
        "message": (f"{model_id} has a library spec ({library_spec}); the "
                    "Inspector serves it instead of your generated spec, "
                    f"which is archived. {action} is not available on it — "
                    "changes to a library spec go through a pull request."),
    }


def _spec_load_failure(err: SpecLoadError) -> dict:
    """Turn a loader refusal into a structured, actionable SPEC error. A
    broken spec is not a server outage — surfaced as success:false (which the
    Inspector renders on its error card) instead of a bare 500 that the
    frontend would misread as 'backend unreachable'."""
    msg = str(err)
    if err.path is not None and _overlay.is_user_spec_path(err.path):
        msg = ("Your saved spec for this model no longer validates against the "
               "current schema — regenerate it from the Inspector. "
               f"Details: {msg}")
    return {"success": False, "error_kind": RunErrorKind.SPEC_INVALID.value, "message": msg}


@app.post("/api/spec/analyze")
async def service_analyze(
    req: AnalyzeRequest,
    user: dict = Depends(_auth.get_current_user),
):
    """User-aware analyze. The PUBLIC spec (repo/disk + global cache) always
    wins — repo precedence preserved. On a miss, serve the signed-in user's
    OWN latest private generated spec for this model (materialized server-side
    into an overlay; the path is never client-supplied). Anonymous callers or
    peruser-off → identical to the core /analyze."""
    user_id = user.get("sub") if user else None
    return await offload_response(_service_analyze_payload, req, user_id)


def _service_analyze_payload(req: AnalyzeRequest, user_id: Optional[str]) -> dict:
    """The `/api/spec/analyze` answer; runs in a compute worker."""
    from backend.analyzer import NoSpecMatch, analyze_model
    from backend.api.routes.analyze import build_response
    from backend.service import spec_generations
    t0 = time.time()
    try:
        try:
            result = analyze_model(
                model_id=req.model_id, batch_size=req.batch_size,
                seq_len=req.seq_len, dtype=req.dtype, trust_remote_code=False,
            )
        except ValueError as public_miss:
            # No public spec matches. The analyzer no longer answers with
            # a config-only guess; a signed-in user's own stored spec is
            # the one fallback, and without it the refusal (which points
            # at /generate-spec) is the answer.
            # `no_spec`: neither the library nor a record of the
            # caller's has a spec — the one case the page generates one.
            miss = {"success": False, "message": str(public_miss),
                    **({"no_spec": True}
                       if isinstance(public_miss, NoSpecMatch) else {})}
            if not (user_id and _peruser_active()):
                return miss
            try:
                row = spec_generations.serving_generation(
                    user_id, req.model_id)
            except spec_generations.SpecGenerationsUnavailable:
                # Not knowing is not "no spec": nothing may generate.
                return _records_unavailable()
            if not (row and row.get("spec_yaml")):
                return miss
            served = _served_record(row)
            if row.get("status") not in spec_generations.SERVED:
                # The user's only record of the model needs repair or
                # was rejected: not analysed, and not a missing spec.
                return {"success": False, **served, "message": (
                    "Your saved spec for this model needs repair before "
                    "the Inspector can analyse it." if row.get("status")
                    == GenerationStatus.NEEDS_REPAIR else
                    "Your saved spec for this model was rejected.")}
            try:
                bundle = _normalize_record_bundle(
                    req.model_id, row["spec_yaml"],
                    row.get("new_component_yamls"))
            except (SpecBundleError, SpecLoadError) as exc:
                fail = _bundle_failure(exc, model_id=req.model_id)
                fail.update(served)
                return fail
            ov = _overlay.materialize_user_spec(
                user_id, req.model_id, bundle, row.get("snapshot_json"))
            try:
                priv = analyze_model(
                    model_id=req.model_id, batch_size=req.batch_size,
                    seq_len=req.seq_len, dtype=req.dtype, overlay_dir=ov,
                )
                if not priv.get("success"):
                    return {"success": False,
                            "message": priv.get("error", "Unknown error"),
                            **served}
                resp = build_response(
                    req, priv, (time.time() - t0) * 1000)
            except (ValueError, KeyError, NotImplementedError) as broken:
                return {**_analysis_refusal(broken), **served}
            resp.update(served)
            return resp
        if not result.get("success"):
            raise HTTPException(
                status_code=404,
                detail=result.get("error", "Unknown error"),
            )
        resp = build_response(req, result, (time.time() - t0) * 1000)
        if user_id and _peruser_active():
            # The library spec answered, so a private Refine of it is
            # refused (`service_refine_spec`); the Inspector reads this.
            resp["refine_offered"] = False
        return resp
    except HTTPException:
        raise
    except SpecLoadError as e:
        return _spec_load_failure(e)
    except (ValueError, KeyError, NotImplementedError) as e:
        # A spec that loads but does not walk (a dim, a broadcast, a
        # capability the engine lacks) is an outcome, same as public
        # `/analyze`.
        return _analysis_refusal(e)


def _records_unavailable() -> dict:
    """The answer when the caller's saved specs could not be read."""
    return {"success": False, "status": "records_unavailable",
            "message": "Could not read your saved specs. Try again in a moment."}


def _served_record(row: dict) -> dict:
    """What an analyze served from the caller's own record says about it:
    where it came from, the record's status, and — for an engine-blocked
    record — the deployments and stages where validate met each gap."""
    out = {"served_from": "user_generation", "record_status": row.get("status"),
           "record_id": row.get("id")}
    if row.get("status") == GenerationStatus.BLOCKED_BY_ENGINE:
        out["engine_gaps"] = row.get("engine_gaps") or []
    return out


def _analysis_refusal(exc: BaseException) -> dict:
    """An analysis that refused, as an outcome classified by who resolves it
    (`Refusal.of`, the owner validate judges a refusal by), never by its
    text:

    - the engine (`EngineGap`, `NotImplementedError`): `blocked_by_engine`
      and the capability and node — the spec is right;
    - the spec writer (`SpecRefusal`): `spec_invalid` with the repair, for
      Refine or Agent Fix;
    - no declared owner: `analysis_refusal_unattributed`, a failure.

    None of them is a missing spec: a page never answers one with a new
    generation."""
    refusal = Refusal.of(exc, deployment="the requested analysis",
                         stage="analyze")
    out = {"success": False, "message": str(exc), "node": refusal.node}
    if refusal.kind == ENGINE_KIND:
        out["error_kind"] = RunErrorKind.BLOCKED_BY_ENGINE.value
        out["engine_gap"] = {"capability": refusal.capability,
                             "node": refusal.node}
    elif refusal.kind == SPEC_KIND:
        out["error_kind"] = RunErrorKind.SPEC_INVALID.value
        out["errors"] = [refusal_finding([refusal]).detail]
        out["fixable"] = True
    else:
        out["error_kind"] = RunErrorKind.ANALYSIS_REFUSAL_UNATTRIBUTED.value
    return out
