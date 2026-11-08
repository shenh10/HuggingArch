"""Self-service account endpoints.

  POST   /api/account/avatar   — multipart upload of avatar image.
                                  Backend re-uploads to Storage with
                                  service-role + writes profiles.avatar_url.
  DELETE /api/account/avatar   — clears avatar_url + removes the
                                  storage object.

Note: data export and self-delete endpoints were intentionally removed.
Surfacing destructive / data-portability actions to end users requires
a public privacy policy + a confirmation flow (e.g. email re-verify)
that we don't have yet. Re-introduce only when those are in place.
Admin-side "delete user" still exists at /api/admin/users/:id/delete.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

from backend.api._app import app
from backend.service import auth as _auth
from backend.service import rate_limit as _rate_limit
from backend.service import agent_credentials as _creds
from backend.service import crypto as _crypto
from backend.service import spec_generations as _gens
from backend.service import spec_generation_logs as _genlogs
from backend.service import spec_generation_reviews as _reviews
from backend.service import github as _github
from backend.service.routes.spec import (
    RecordJob,
    admission_run_key,
    fix_seed_bundle,
    launch_admission,
    launch_fix,
    launch_recheck,
    launch_review,
    mark_engine_changed,
    pending_refusal,
    launch_spec_pr,
    add_library_standing,
    library_standing,
    pr_run_key,
    superseded_by,
    superseded_refusal,
)
from backend import spec_worker
from backend.arch.loader import SpecLoadError
from backend.spec_worker.bundle import (
    FIX_PATH,
    BlockedBundle,
    SpecBundleError,
    is_repairable,
    record_fingerprint,
)
from backend.spec_worker.self_register import stamp_match_model_id as _stamp_match_model_id
from backend.spec_worker.outcome import GenerationStatus, RunErrorKind
from backend.service.routes.admin import _contributor_for
from backend.service import personal_tokens as _pat
from backend.service import profiles as _profiles
from backend.arch.snapshots import release_dates_for as _release_dates_for


def _bundle_invalid(exc: SpecBundleError) -> HTTPException:
    repair = (f"Agent Fix can repair it ({FIX_PATH})." if is_repairable(exc)
              else "Regenerate it from the Inspector.")
    return HTTPException(
        status_code=409,
        detail={
            "error_kind": RunErrorKind.BUNDLE_INVALID.value,
            "message": (
                "The saved bundle must be repaired before a PR can be opened. "
                + repair
            ),
            "errors": exc.errors,
        },
    )


def _pr_data_error(exc: BaseException) -> HTTPException:
    """Bundle reference errors and schema/load errors are data, not 502."""
    if isinstance(exc, SpecBundleError):
        return _bundle_invalid(exc)
    if isinstance(exc, SpecLoadError):
        repair = (f"Agent Fix can repair it ({FIX_PATH})." if is_repairable(exc)
                  else "Regenerate it from the Inspector.")
        return HTTPException(
            status_code=409,
            detail={
                "error_kind": RunErrorKind.SPEC_INVALID.value,
                "message": (
                    "The saved spec no longer validates against the current "
                    "schema. " + repair
                ),
                "errors": [str(exc)],
            },
        )
    raise TypeError(f"unexpected PR data error {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Avatar upload / delete
#
# Storage layout: bucket=avatars, path=<user_id>/<unix_ts>.<ext>. The
# user_id prefix is namespacing only — the bucket's RLS blocks all
# client writes, so the path is enforced server-side. Old objects are
# removed lazily when a new avatar lands; if the cleanup call fails,
# the user just has dangling files (cheap, no security issue).
# ---------------------------------------------------------------------------

_AVATAR_BUCKET = "avatars"
_AVATAR_MAX_BYTES = 2 * 1024 * 1024
_AVATAR_MIME_EXT = {
    "image/png":  "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif":  "gif",
}


def _avatar_object_path(user_id: str, ext: str) -> str:
    return f"{user_id}/{int(time.time())}.{ext}"


def _avatar_public_url(object_path: str) -> str:
    base = (_auth._supabase_url() or "").rstrip("/")
    return f"{base}/storage/v1/object/public/{_AVATAR_BUCKET}/{object_path}"


def _list_avatar_objects(user_id: str) -> list[str]:
    """Return existing object names under <user_id>/ so we can delete
    them after a new upload. Storage list endpoint is recursive when we
    pass a prefix; we filter to direct children only (the format we
    write)."""
    url = _auth._supabase_url()
    key = _auth._service_role_key()
    if not (url and key):
        return []
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{url.rstrip('/')}/storage/v1/object/list/{_AVATAR_BUCKET}",
                json={"prefix": f"{user_id}/", "limit": 100},
                headers={"apikey": key, "Authorization": f"Bearer {key}"},
            )
        if resp.status_code != 200:
            return []
        items = resp.json() or []
        return [f"{user_id}/{it['name']}" for it in items if it.get("name")]
    except httpx.RequestError:
        return []


def _delete_avatar_objects(paths: list[str]) -> None:
    if not paths:
        return
    url = _auth._supabase_url()
    key = _auth._service_role_key()
    if not (url and key):
        return
    try:
        with httpx.Client(timeout=10.0) as client:
            client.request(
                "DELETE",
                f"{url.rstrip('/')}/storage/v1/object/{_AVATAR_BUCKET}",
                json={"prefixes": paths},
                headers={"apikey": key, "Authorization": f"Bearer {key}"},
            )
    except httpx.RequestError:
        pass  # best-effort cleanup


def _patch_profile_avatar_url(user_id: str, value: str | None) -> None:
    """PATCH profiles.avatar_url for a single user via service-role."""
    url = _auth._supabase_url()
    key = _auth._service_role_key()
    if not (url and key):
        raise HTTPException(status_code=503, detail="Service-role key not configured")
    body = {"avatar_url": value}
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.patch(
                f"{url.rstrip('/')}/rest/v1/profiles",
                params={"id": f"eq.{user_id}"},
                json=body,
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
        if r.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to update profile.avatar_url: {r.status_code} {r.text}",
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Profile update errored: {e}") from e


@app.post("/api/account/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    user: dict = Depends(_auth.require_user),
):
    """Upload one image as the caller's avatar. Validates size and MIME
    server-side (Storage bucket also caps at 2 MB / allow-listed MIME,
    defense-in-depth). On success, writes the resulting public URL to
    profiles.avatar_url and removes any previously stored avatar
    objects under <user_id>/. Returns the new URL.
    """
    _rate_limit.starts_work()       # each upload is its own Storage relay
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")

    content_type = (file.content_type or "").lower()
    ext = _AVATAR_MIME_EXT.get(content_type)
    if not ext:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type {content_type!r}. "
                   f"Allowed: {sorted(_AVATAR_MIME_EXT)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(data) > _AVATAR_MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)} B); max {_AVATAR_MAX_BYTES} B",
        )

    url = _auth._supabase_url()
    key = _auth._service_role_key()
    if not (url and key):
        raise HTTPException(status_code=503, detail="Service-role key not configured")

    object_path = _avatar_object_path(user_id, ext)

    # Capture the existing-object list BEFORE upload so the new object
    # we just wrote doesn't get cleaned up by a concurrent call.
    existing = await asyncio.to_thread(_list_avatar_objects, user_id)

    # Upload via the Storage REST endpoint. POST with x-upsert:true is
    # equivalent to PUT — we generate fresh paths so no conflict, but
    # this also covers the rare case where someone stamps the same
    # second twice.
    def _upload():
        with httpx.Client(timeout=15.0) as client:
            return client.post(
                f"{url.rstrip('/')}/storage/v1/object/{_AVATAR_BUCKET}/{object_path}",
                content=data,
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": content_type,
                    "x-upsert": "true",
                    "Cache-Control": "max-age=31536000, immutable",
                },
            )
    try:
        r = await asyncio.to_thread(_upload)
        if r.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Storage upload failed: {r.status_code} {r.text}",
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Storage call errored: {e}") from e

    public_url = _avatar_public_url(object_path)
    await asyncio.to_thread(_patch_profile_avatar_url, user_id, public_url)

    # Best-effort cleanup of older objects.
    await asyncio.to_thread(
        _delete_avatar_objects, [p for p in existing if p != object_path])

    return {"avatar_url": public_url}


@app.delete("/api/account/avatar")
def delete_avatar(user: dict = Depends(_auth.require_user)):
    """Clear avatar_url + drop every avatar object under <user_id>/.
    Idempotent: succeeds even when nothing's stored."""
    _rate_limit.starts_work()       # each delete is its own Storage relay
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    _patch_profile_avatar_url(user_id, None)
    _delete_avatar_objects(_list_avatar_objects(user_id))
    return {"avatar_url": None}


# ---------------------------------------------------------------------------
# Profile fields (username / full name / bio / GitHub handle)
#
# The `authenticated` role has no direct DML on profiles (migration 0008), so
# the account page can't upsert the row client-side — it posts here and the
# backend writes with service-role, exactly like the avatar flow.
# ---------------------------------------------------------------------------

@app.patch("/api/account/profile")
def update_profile(
    req: _profiles.EditableProfile,
    user: dict = Depends(_auth.require_user),
):
    """Save the caller's own editable profile fields. A non-empty GitHub handle
    is validated (so it can't reach a commit trailer malformed) and stored
    without its leading '@'; an empty field clears the column."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")

    gh_raw = (req.github_handle or "").strip().lstrip("@")
    github_handle = None
    if gh_raw:
        github_handle = _profiles.normalize_handle(gh_raw)
        if github_handle is None:
            raise HTTPException(status_code=400, detail="Invalid GitHub handle")

    try:
        ok = _profiles.set_profile(user_id, _profiles.EditableProfile(
            username=(req.username or "").strip() or None,
            full_name=(req.full_name or "").strip() or None,
            bio=(req.bio or "").strip() or None,
            github_handle=github_handle,
        ))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=503, detail="Account service not configured")
    return {"ok": True}


# ---------------------------------------------------------------------------
# BYOK agent credentials (Phase 2)
# ---------------------------------------------------------------------------
# A user stores their own LLM key + endpoint so spec generation runs on their
# account. The key is encrypted at rest (backend.service.crypto) and never
# returned to the client — GET only reports which agents are configured.

class AgentCredentialRequest(BaseModel):
    agent_cli: str
    api_key: str
    base_url: str | None = None
    model: str | None = None
    # Which config of this agent this is (''/None = the unnamed one); several
    # named configs of the same agent coexist (migration 0000).
    name: str | None = None
    # Per-task model overrides, task key -> model id ("" = the default model).
    # None leaves the stored map as it is.
    task_models: dict[str, str | None] | None = None
    is_default: bool = False


class TaskRoutingRequest(BaseModel):
    task: str
    # 'agent_cli' or 'agent_cli:name'; blank/None clears the task's routing.
    agent_cli: str | None = None


@app.get("/api/account/agent-credentials")
def list_agent_credentials(user: dict = Depends(_auth.require_user)):
    """Sanitized list of the caller's configured agents. Never includes the
    key or its ciphertext — only agent_cli / base_url / model / task_models /
    is_default / has_key. `agent_tasks` are the runs a model can be chosen
    for; `task_models_enabled` is False until migration 0036 is deployed."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    view = _creds.listing(user_id)
    return {
        "items": view["items"],
        "byok_enabled": _creds.enabled(),
        "agent_tasks": [{"key": t.value, "label": t.label} for t in _creds.AgentTask],
        "task_models_enabled": view["task_models_enabled"],
        "names_enabled": view.get("names_enabled", False),
        # Site-wide per-task defaults (migration 0039) — the UI shows them as
        # each task row's effective default when the credential pins nothing.
        "task_defaults": _creds.fetch_task_defaults(),
        # The caller's per-task agent routing (migration 0040).
        "task_agents": _creds.fetch_task_routing(user_id),
    }


@app.put("/api/account/agent-task-routing")
def put_agent_task_routing(
    req: TaskRoutingRequest,
    user: dict = Depends(_auth.require_user),
):
    """Route one agent task to one of the caller's credentials — an
    'agent_cli' or 'agent_cli:name' target (blank clears it, falling back to
    the default credential)."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    try:
        _creds.put_task_routing(user_id, req.task, (req.agent_cli or "").strip() or None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except _creds.TaskModelsUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"ok": True}


@app.put("/api/account/agent-credentials")
def put_agent_credential(
    req: AgentCredentialRequest,
    user: dict = Depends(_auth.require_user),
):
    """Store (or replace) the caller's credential for one agent. The key is
    encrypted before it touches the database."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    if req.agent_cli not in _creds.VALID_AGENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent: {req.agent_cli!r}. One of {list(_creds.VALID_AGENTS)}.",
        )
    base_url = (req.base_url or "").strip() or None
    model = (req.model or "").strip() or None
    api_key = (req.api_key or "").strip()
    try:
        name = _creds.normalize_name(req.name)
        task_models = (None if req.task_models is None
                       else _creds.normalize_task_models(req.task_models))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    try:
        if not api_key:
            # Editing an existing credential: change endpoint / model / default
            # while KEEPING the stored key. Fails (→ 400) if none exists yet.
            updated = _creds.update_fields(
                user_id=user_id,
                agent_cli=req.agent_cli,
                name=name,
                base_url=base_url,
                model=model,
                is_default=req.is_default,
                task_models=task_models,
            )
            if not updated:
                raise HTTPException(
                    status_code=400,
                    detail="Enter an API key to add this agent (blank only edits an existing one).",
                )
        else:
            _creds.upsert(
                user_id=user_id,
                agent_cli=req.agent_cli,
                name=name,
                api_key=api_key,
                base_url=base_url,
                model=model,
                is_default=req.is_default,
                task_models=task_models,
            )
    except HTTPException:
        raise
    except (_creds.TaskModelsUnavailable, _creds.CredentialNamesUnavailable) as e:
        raise HTTPException(status_code=503, detail=f"Could not save credential: {e}.") from None
    except _crypto.CredentialsKeyMissing:
        raise HTTPException(
            status_code=503,
            detail="BYOK is not enabled on this server (HA_CREDS_ENC_KEY unset).",
        ) from None
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not save credential: {e}") from e
    return {"saved": True, "agent_cli": req.agent_cli, "name": name,
            "is_default": req.is_default}


@app.post("/api/account/agent-credentials/{agent_cli}/default")
def set_default_credential(
    agent_cli: str,
    name: str = Query(""),
    user: dict = Depends(_auth.require_user),
):
    """Make one of the caller's existing credentials their default (clears any
    other). Key-free — no re-entry needed. `name` picks among several configs
    of the same agent (migration 0000)."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    if agent_cli not in _creds.VALID_AGENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent: {agent_cli!r}. One of {list(_creds.VALID_AGENTS)}.",
        )
    try:
        ok = _creds.set_default(user_id=user_id, agent_cli=agent_cli, name=name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not set default: {e}") from e
    if not ok:
        raise HTTPException(status_code=404, detail="No such credential to set as default.")
    return {"saved": True, "agent_cli": agent_cli, "is_default": True}


@app.delete("/api/account/agent-credentials/{agent_cli}")
def delete_agent_credential(
    agent_cli: str,
    name: str = Query(""),
    user: dict = Depends(_auth.require_user),
):
    """Remove one of the caller's stored credentials (the unnamed config
    unless `name` says otherwise)."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    ok = _creds.delete(user_id, agent_cli, name=name)
    return {"deleted": ok, "agent_cli": agent_cli}


class AgentModelsRequest(BaseModel):
    agent_cli: str
    base_url: str | None = None
    api_key: str | None = None
    # Which stored config to borrow the key from when api_key is blank.
    name: str | None = None


@app.post("/api/account/agent-credentials/models")
def list_agent_models(
    req: AgentModelsRequest,
    user: dict = Depends(_auth.require_user),
):
    """Proxy the provider's /v1/models so the account page can offer the
    endpoint's real model ids. Uses the api_key from the request, or falls back
    to the caller's stored credential when the field is blank (editing an
    existing credential shows no key). The key is used in-flight only — never
    logged or stored."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    if req.agent_cli not in _creds.VALID_AGENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent: {req.agent_cli!r}. One of {list(_creds.VALID_AGENTS)}.",
        )
    api_key = (req.api_key or "").strip()
    base_url = (req.base_url or "").strip() or None
    if not api_key:
        try:
            saved = _creds.saved_key_and_base(user_id, req.agent_cli,
                                              name=req.name or "")
        except _creds.CredentialDecryptError as e:
            # The key is stored but undecryptable — tell the truth ("re-enter")
            # instead of the misleading "save this credential first".
            raise HTTPException(
                status_code=409,
                detail="Your saved credential could not be decrypted — please re-enter your API key.",
            ) from e
        if not saved:
            # `reason` is what the page branches on (it asks for a key
            # instead of showing an error); the text is for people.
            return JSONResponse(status_code=400, content={
                "detail": "Enter an API key (or save this credential first) to load models.",
                "reason": "key_required",
            })
        api_key, saved_base = saved
        base_url = base_url or saved_base
    try:
        models = _creds.fetch_agent_models(req.agent_cli, api_key, base_url)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"Could not load models: {e}") from e
    return {"models": models}


@app.post("/api/account/agent-credentials/{agent_cli}/test")
def test_agent_credential(
    agent_cli: str,
    name: str = Query(""),
    user: dict = Depends(_auth.require_user),
):
    """Probe the caller's stored credential against its endpoint — reachability
    and auth, via /v1/models (the same call the model list uses). Always 200
    with `ok`: the probe ran; `ok: false` means the CREDENTIAL failed, with a
    user-safe `detail`. The key is decrypted server-side and used in-flight
    only. `name` picks among several configs of the same agent."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    if agent_cli not in _creds.VALID_AGENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent: {agent_cli!r}. One of {list(_creds.VALID_AGENTS)}.",
        )
    try:
        saved = _creds.saved_key_and_base(user_id, agent_cli, name=name)
    except _creds.CredentialDecryptError as e:
        raise HTTPException(
            status_code=409,
            detail="Your saved credential could not be decrypted — please re-enter your API key.",
        ) from e
    if not saved:
        raise HTTPException(status_code=404, detail="No such credential to test.")
    api_key, base_url = saved
    t0 = time.monotonic()
    try:
        models = _creds.fetch_agent_models(agent_cli, api_key, base_url)
    except ValueError as e:
        return {"ok": False, "detail": str(e),
                "latency_ms": int((time.monotonic() - t0) * 1000)}
    # Listing models proves reachability + auth, nothing about GENERATING:
    # run a 1-token completion too, on each model a run of this credential
    # would use (`run_model`, the rule `resolve_env` applies per task). A
    # task left to the endpoint's default model names none to probe.
    cred = next((c for c in _creds.list_for_user(user_id)
                 if c.get("agent_cli") == agent_cli
                 and (c.get("name") or "") == (name or "")), None) or {}
    defaults = _creds.fetch_task_defaults()
    run_models = [m for m in dict.fromkeys(
        _creds.run_model(cred, agent_cli, task, defaults)
        for task in _creds.AgentTask.known()) if m] or [None]
    probes = [_creds.probe_agent_generation(agent_cli, api_key, base_url, m)
              for m in run_models]
    failed = [p for p in probes if not p.get("ok")]
    generation = probes[0] if len(probes) == 1 else {
        "ok": not failed,
        "model": ", ".join(str(p.get("model")) for p in probes),
        "detail": "; ".join(f"{p.get('model')}: {p.get('detail')}" for p in failed),
    }
    return {"ok": True, "models": len(models),
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "generation": generation}


# ---------------------------------------------------------------------------
# My spec generations (review / fix / open PR)
# ---------------------------------------------------------------------------

def _ts(s: Optional[str]) -> datetime:
    """ISO string → aware datetime for ordering run logs; unparsable sorts
    first (epoch)."""
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def _log_ref(log: dict) -> dict:
    return {
        "id": log["id"],
        "success": log.get("success"),
        "created_at": log.get("created_at"),
    }


def _paired_log_ids(items: list) -> set:
    """Every log id already hanging off a generation (produce + fix + review)."""
    ids: set = set()
    for g in items:
        if g.get("log_id"):
            ids.add(g["log_id"])
        for key in ("fix_logs", "review_logs", "job_logs"):
            for row in g.get(key) or []:
                if row.get("id"):
                    ids.add(row["id"])
    return ids


#: where a record job's log hangs off its record; admission and PR jobs
#: leave a log row but no transcript worth listing (`job_logs`)
_JOB_LOG_KEYS = {RecordJob.FIX: "fix_logs", RecordJob.REVIEW: "review_logs"}


def _attach_run_logs(items: list, logs: list) -> None:
    """Pair each generation with the run log that produced it (`log_id` /
    `log_success`) and with later fix/review transcripts (`fix_logs` /
    `review_logs`, newest first).

    A log names its record: the run stamps `record_id` on its log row when
    it writes a record (migration 0005), and a fix/review/admission/PR run's
    key is `{job}|{user}|{record_id}`. A record's produce log is the newest
    generate/refine log that names it — a regenerated record keeps one row,
    so the runs behind its earlier contents name it too and stay listed as
    their own rows. A log that names no record (a failed run, a run logged
    before 0005) pairs with nothing; deleting a record deletes only logs
    that name it."""
    by_id = {it["id"]: it for it in items if it.get("id")}
    produce: dict = {}
    jobs: list = []
    for log in logs:
        job = RecordJob.of_log(log)
        if job is not None:
            jobs.append(log)
            continue
        rec_id = log.get("record_id")
        if rec_id in by_id:
            best = produce.get(rec_id)
            if best is None or _ts(log.get("created_at")) > _ts(best.get("created_at")):
                produce[rec_id] = log
    for rec_id, log in produce.items():
        by_id[rec_id]["log_id"] = log["id"]
        by_id[rec_id]["log_success"] = log.get("success")

    jobs.sort(key=lambda l: _ts(l.get("created_at")), reverse=True)
    for log in jobs:
        rec = by_id.get(log.get("record_id") or RecordJob.record_of_log(log) or "")
        if rec is None:
            continue
        key = _JOB_LOG_KEYS.get(RecordJob.of_log(log), "job_logs")
        rec.setdefault(key, []).append(_log_ref(log))


@app.get("/api/account/generations")
def list_my_generations(
    superseded: bool = False,
    user: dict = Depends(_auth.require_user),
):
    """The caller's own spec-generation records (summary), for the account
    'My spec generations' view. Each item carries `log_id` for its producing
    run's log, plus `fix_logs` / `review_logs` when those jobs have
    transcripts (see _attach_run_logs); each review log names the
    `verdict` recorded for its review. Logs that paired with no spec —
    failed generate runs produce no generation record — come back as
    `orphan_logs` so the page manages EVERYTHING in the one list.

    Records a library spec supersedes (`superseded_by`) are archived: the
    default view leaves them out, and `?superseded=true` lists only them.
    Orphan logs split the same way: a log of a model the library now serves
    is archived with that model's records, so a model is listed in one place.
    Every item and log also names its `library_spec` (`library_standing`):
    a merged record whose model the library now serves is in the library,
    not superseded."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    items = _gens.list_for_user(user_id)
    logs = _genlogs.list_for_user(user_id, limit=1000)
    # Pair logs over every record first: an archived record's logs are its
    # own, not orphans of the default view.
    _attach_run_logs(items, logs)
    paired = _paired_log_ids(items)
    orphan_logs = [l for l in logs if l.get("id") not in paired]
    add_library_standing(items + orphan_logs)
    items = [it for it in items if bool(it["superseded_by"]) == superseded]
    orphan_logs = [l for l in orphan_logs if bool(l["superseded_by"]) == superseded]
    record_ids = [g["id"] for g in items if g.get("id")]
    verdicts = _reviews.latest_verdicts_for_records(record_ids)
    by_log = _reviews.verdicts_by_log(record_ids)
    for it in items:
        if verdicts.get(it.get("id")):
            it["latest_verdict"] = verdicts[it["id"]]
        # A review log is the agent's transcript; the verdict recorded for
        # it can differ from the one the agent wrote, so the list says it.
        for ref in it.get("review_logs") or []:
            ref["verdict"] = by_log.get(ref["id"])
    dates = _release_dates_for(
        [it.get("model_id") for it in items]
        + [l.get("model_id") for l in orphan_logs]
    )
    for it in items:
        it["release_date"] = dates.get(it.get("model_id"))
    mark_engine_changed(items)
    for log in orphan_logs:
        log["release_date"] = dates.get(log.get("model_id"))
    # Poll-on-view (no webhook): reconcile any still-pending contributed PR so
    # the user sees merged/closed on their own page, not just when an admin
    # opens the record. Bounded to `contributed` — the only pollable state —
    # and best-effort (a GitHub blip leaves the row unchanged).
    for it in items:
        if it.get("pr_url") and it.get("status") == "contributed":
            new = _gens.sync_pr_status(it)
            if new:
                it["status"] = new
    return {"items": items, "orphan_logs": orphan_logs}


@app.post("/api/account/generations/{record_id}/recheck")
def recheck_my_generation(
    record_id: str,
    user: dict = Depends(_auth.require_user),
):
    """Re-check one of the caller's own pending records against the current
    engine (`launch_recheck`). Returns {status:"started", run_key}; poll
    GET /api/spec/recheck-status."""
    user_id, rec = _owned_record(record_id, user)
    return launch_recheck(rec, actor_id=user_id)


@app.post("/api/account/generations/{record_id}/review")
def review_my_generation(
    record_id: str,
    user: dict = Depends(_auth.require_user),
):
    """Start an agent review of the caller's own generation. Uses the caller's
    BYOK creds (mirrors generation); a non-admin with no stored key is rejected.
    Returns {status:"started", run_key}; poll GET /api/spec/review-status."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    rec = _gens.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="No such generation")
    if rec.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your generation")
    library_spec = superseded_by(rec)
    if library_spec:
        return superseded_refusal(rec["model_id"], library_spec, "Review")
    try:
        resolved = _creds.resolve_env(user_id, task="review")
    except _creds.CredentialDecryptError:
        return {"status": "credential_undecryptable",
                "message": "Your saved agent API key could not be decrypted — please re-enter it in account settings."}
    except _creds.CredentialRefusal as exc:
        return exc.refusal("run a review")
    if resolved is None:
        return {"status": "no_credentials",
                "message": "Add your own agent API key in account settings to run a review."}
    return launch_review(rec, reviewer_id=user_id, reviewer_role="user",
                         agent_cli=resolved["agent_cli"], agent_env=resolved["env"])


@app.get("/api/account/generations/{record_id}/reviews")
def list_my_generation_reviews(
    record_id: str,
    user: dict = Depends(_auth.require_user),
):
    """Reviews of a generation the caller owns (their own + any admin's).
    Each row says whether it judged the bundle the record holds now
    (`bundle_current`): an admission report on a row that did not is stale."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    rows = _reviews.list_for_record_owner(record_id, user_id)
    if rows:
        try:
            rec = _gens.get_for_user(record_id, user_id)
        except _gens.SpecGenerationsUnavailable:
            rec = None
        current = record_fingerprint(rec) if rec else None
        for row in rows:
            row["bundle_current"] = bool(current) and row.get("bundle_fingerprint") == current
    return {"items": rows}


@app.get("/api/account/generations/{record_id}")
def get_my_generation(
    record_id: str,
    user: dict = Depends(_auth.require_user),
):
    """Full payload for one of the caller's OWN generations — spec_yaml,
    snapshot, new component yamls, status, PR link — so the account page can show
    a user the actual content of their spec (the list view omits the heavy
    payload). Owner-scoped: a miss or a non-owned record both answer 404 (never
    leaks another user's record). Poll-on-view reconciles a still-pending
    contributed PR to merged/closed, same as the list route."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    try:
        rec = _gens.get_for_user(record_id, user_id)
    except _gens.SpecGenerationsUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Could not load this generation. Retry in a moment.",
        ) from None
    if not rec:
        raise HTTPException(status_code=404, detail="No such generation")
    if rec.get("pr_url") and rec.get("status") == "contributed":
        new = _gens.sync_pr_status(rec)
        if new:
            rec["status"] = new
    rec.update(library_standing(rec))
    mark_engine_changed([rec])
    return rec


@app.delete("/api/account/generations/{record_id}")
def delete_my_generation(
    record_id: str,
    user: dict = Depends(_auth.require_user),
):
    """Delete one of the caller's own generations. `generated` / `rejected` /
    `closed` records are deletable, and so is a `contributed` record whose PR
    was never opened; `merged` records and ones with an open PR stay.
    Reviews of the record cascade. Scoped to the caller's own row."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    # Run logs are bound to the spec: find the produce + fix + review logs
    # that name this record, then drop them with it — otherwise deleting a
    # spec would leave those transcripts behind as orphan logs. Reuses the
    # exact pairing the list view shows; a log that names no record is never
    # deleted here.
    items = _gens.list_for_user(user_id)
    logs = _genlogs.list_for_user(user_id, limit=1000)
    _attach_run_logs(items, logs)
    target = next((it for it in items if it.get("id") == record_id), None)
    paired_ids = _paired_log_ids([target] if target else [])
    if not _gens.delete_for_user(record_id, user_id):
        raise HTTPException(
            status_code=404,
            detail="No such generation — or it has an open or merged PR "
                   "(those records can't be deleted).",
        )
    for log_id in paired_ids:
        _genlogs.delete_for_user(log_id, user_id)
    return {"deleted": True, "id": record_id}


def _opens_on_refusals(rec: dict) -> bool:
    """Whether Fix opens this record on its load refusals: a repairable
    refusal stops it from loading, so no review of it can exist. A record
    that loads, or whose refusal is not repairable (`RefusedRecord`), does
    not."""
    return isinstance(fix_seed_bundle(rec), BlockedBundle)


@app.post("/api/account/generations/{record_id}/fix")
def fix_my_generation(
    record_id: str,
    user: dict = Depends(_auth.require_user),
):
    """Start an agent FIX of the caller's own generation: repair the defects the
    latest NO_PASS review named, editing only this record's own draft, and write
    the fixed bundle back. Uses the caller's BYOK creds. Returns
    {status:"started", run_key}; poll GET /api/spec/fix-status.

    A record that a repairable refusal stops from loading cannot be reviewed,
    so it opens on those refusals instead: Fix stages it as saved and they
    are its work order; a `needs_repair` record opens on the errors its
    re-check found. An engine-blocked record has nothing to fix
    ({status:"blocked_by_engine"}). Any other record needs its latest review
    to be NO_PASS; {status:"no_review"} otherwise."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    rec = _gens.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="No such generation")
    if rec.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your generation")
    library_spec = superseded_by(rec)
    if library_spec:
        return superseded_refusal(rec["model_id"], library_spec, "Agent Fix")
    if rec.get("status") == GenerationStatus.BLOCKED_BY_ENGINE:
        return pending_refusal(rec, "Agent Fix")
    rows = _reviews.list_for_record_owner(record_id, user_id)
    if rec.get("status") == GenerationStatus.NEEDS_REPAIR:
        # The re-check's errors are the work order (`launch_fix`).
        report = ""
    elif rows and str(rows[0].get("verdict")) == "no_pass":
        report = rows[0].get("report") or ""
    elif _opens_on_refusals(rec):
        report = ""
    else:
        return {"status": "no_review",
                "message": "Run a review first — the latest review must be NO_PASS to fix."}
    try:
        resolved = _creds.resolve_env(user_id, task="fix")
    except _creds.CredentialDecryptError:
        return {"status": "credential_undecryptable",
                "message": "Your saved agent API key could not be decrypted — please re-enter it in account settings."}
    except _creds.CredentialRefusal as exc:
        return exc.refusal("run a fix")
    if resolved is None:
        return {"status": "no_credentials",
                "message": "Add your own agent API key in account settings to run a fix."}
    return launch_fix(rec, fixer_id=user_id, fixer_role="user", review_report=report,
                      agent_cli=resolved["agent_cli"], agent_env=resolved["env"])


def _latest_review_is_pass(rows: list, record: dict) -> bool:
    """A PR may be opened only once the spec's latest review is PASS for the
    bundle the record holds now. Rows arrive newest-first (owner-scoped); an
    older PASS behind a newer NO_PASS does not count, and a PASS of another
    bundle — the spec edited since, or a review from before reviews were
    bound to a bundle — does not apply."""
    return bool(rows and str(rows[0].get("verdict")) == "pass"
                and rows[0].get("bundle_fingerprint")
                == record_fingerprint(record))


class CreatePRBody(BaseModel):
    #: an admin opens a draft even when the corpus checks fail, to put the
    #: rule fix in the same PR (ADMIT §2.3)
    draft_anyway: bool = False


def _owned_record(record_id: str, user: dict) -> tuple[str, dict]:
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    rec = _gens.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="No such generation")
    if rec.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Not your generation")
    return user_id, rec


def _pass_review(record_id: str, user_id: str, rec: dict) -> Optional[dict]:
    """The record's latest review when it is a PASS of the bundle the record
    holds now; None otherwise."""
    rows = _reviews.list_for_record_owner(record_id, user_id)
    return rows[0] if _latest_review_is_pass(rows, rec) else None


def _check_bundle(rec: dict) -> None:
    """A malformed or incomplete bundle answers 409 before any job starts."""
    try:
        _github.check_spec_bundle(
            rec["model_id"],
            _stamp_match_model_id(rec.get("spec_yaml") or "", rec["model_id"]),
            rec.get("new_component_yamls") or [])
    except (SpecBundleError, SpecLoadError) as e:
        raise _pr_data_error(e) from None


@app.post("/api/account/generations/{record_id}/create-pr")
def create_pr_my_generation(
    record_id: str,
    body: Optional[CreatePRBody] = None,
    user: dict = Depends(_auth.require_user),
):
    """Owner opens the PR contributing their own generation to the library —
    self-service, gated on a PASS review (the abuse guard: you can only propose a
    spec that passed review). The App bot pushes the commits, credited to the
    author. Admins see + review + merge on GitHub; they no longer have to open
    it. {status:"needs_pass"} when the latest review isn't PASS.

    Opening runs the corpus admission against current main first (minutes),
    so this starts a job: {status:"started", run_key}; poll
    GET .../pr-status. The PR carries the record's files and the corpus rows
    the admission wrote; it is a draft when rows wait on a maintainer's
    decision, and it does not open when the admission is blocked or could
    not run (an admin may open a draft anyway past failed corpus checks)."""
    user_id, rec = _owned_record(record_id, user)
    # Lifecycle guard (distinct from the PASS quality gate): a rejected / merged /
    # closed record is done — do not open a PR (and never flip a reject back to
    # contributed). The UI hides the button here; this enforces it at the API.
    if rec.get("status") in ("rejected", "merged", "closed"):
        raise HTTPException(status_code=403,
                            detail=f"This record is {rec.get('status')} — it cannot open a PR.")
    if rec.get("pr_url"):
        return {"pr_url": rec["pr_url"], "status": rec.get("status"), "already": True}
    library_spec = superseded_by(rec)
    if library_spec:
        return superseded_refusal(rec["model_id"], library_spec, "Opening a PR")
    review = _pass_review(record_id, user_id, rec)
    if review is None:
        return {"status": "needs_pass",
                "message": "Run a review and get PASS for the spec as it is now "
                           "before opening a PR."}
    if not _github.enabled():
        raise HTTPException(status_code=503, detail="Spec-PR is not configured.")
    _check_bundle(rec)
    admin_override = bool(body and body.draft_anyway) and _auth.is_admin(user_id)
    return launch_spec_pr(rec, review, update=False, admin_override=admin_override,
                          contributor=_contributor_for(user_id),
                          note="Contributed by the author via HuggingArch.")


@app.post("/api/account/generations/{record_id}/rebase-pr")
@app.post("/api/account/generations/{record_id}/update-pr")
def update_pr_my_generation(
    record_id: str,
    user: dict = Depends(_auth.require_user),
):
    """Owner pushes their record's current bundle to its open PR as ONE commit
    on current main (a rebase, not a stack), recomputing the corpus admission
    there. Same deterministic branch, so the SAME PR updates. Gated on a PASS
    review so a still-failing fix is never shipped. A maintainer's decisions
    written on the branch are carried to the new main; any other change to
    the branch means a maintainer took the PR over, and update-pr stops.
    ``rebase-pr`` is the same action under a UI name. Starts a job like
    create-pr."""
    user_id, rec = _owned_record(record_id, user)
    if rec.get("status") in ("rejected", "merged", "closed"):
        raise HTTPException(status_code=403,
                            detail=f"This record is {rec.get('status')} — it cannot update a PR.")
    if not rec.get("pr_url"):
        return {"status": "no_pr", "message": "No PR yet — open one first."}
    review = _pass_review(record_id, user_id, rec)
    if review is None:
        return {"status": "needs_pass",
                "message": "The latest review must be PASS before updating the PR."}
    if not _github.enabled():
        raise HTTPException(status_code=503, detail="Spec-PR is not configured.")
    _check_bundle(rec)
    return launch_spec_pr(rec, review, update=True,
                          contributor=_contributor_for(user_id),
                          note="Updated by the author via HuggingArch.")


@app.get("/api/account/generations/{record_id}/pr-status")
def pr_status_my_generation(
    record_id: str,
    cursor: int = 0,
    user: dict = Depends(_auth.require_user),
):
    """Poll the record's Open PR / update-pr job. Same shape as
    /api/spec/status; the final `done` event carries `pr_url`, `code` and
    `admission_status`."""
    user_id = _auth.uid(user, detail="No user_id in token")
    return spec_worker.job_status(pr_run_key({"id": record_id, "user_id": user_id}), cursor)


@app.post("/api/account/generations/{record_id}/admission")
def rerun_admission_my_generation(
    record_id: str,
    user: dict = Depends(_auth.require_user),
):
    """Run the record's dry-run admission again on its latest PASS review
    (after an `unavailable` report, or when main moved)."""
    user_id, rec = _owned_record(record_id, user)
    review = _pass_review(record_id, user_id, rec)
    if review is None:
        return {"status": "needs_pass",
                "message": "The admission follows a PASS of the spec as it is now."}
    return launch_admission(rec, review_id=review["id"])


@app.get("/api/account/generations/{record_id}/admission-status")
def admission_status_my_generation(
    record_id: str,
    cursor: int = 0,
    user: dict = Depends(_auth.require_user),
):
    user_id = _auth.uid(user, detail="No user_id in token")
    return spec_worker.job_status(
        admission_run_key({"id": record_id, "user_id": user_id}), cursor)


@app.get("/api/account/gen-logs")
def list_my_gen_logs(user: dict = Depends(_auth.require_user)):
    """The caller's own spec-gen RUN logs (summary, newest first) — every run
    incl. failures, not just cache-worthy generations."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    return {"items": _genlogs.list_for_user(user_id)}


@app.get("/api/account/gen-logs/{log_id}")
def get_my_gen_log(log_id: str, user: dict = Depends(_auth.require_user)):
    """Full log body for one of the caller's own runs (owner-scoped)."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    row = _genlogs.get(log_id, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="Log not found")
    return row


@app.delete("/api/account/gen-logs/{log_id}")
def delete_my_gen_log(log_id: str, user: dict = Depends(_auth.require_user)):
    """Delete one of the caller's own run logs (owner-scoped) — the unified
    generations list manages orphan logs (failed runs) inline, so they get
    the same delete path as spec records."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    if not _genlogs.delete_for_user(log_id, user_id):
        raise HTTPException(status_code=404, detail="Log not found")
    return {"deleted": True, "id": log_id}


class GithubHandleBody(BaseModel):
    github_handle: str | None = None


@app.get("/api/account/github-handle")
def get_account_github_handle(user: dict = Depends(_auth.require_user)):
    """The caller's stored GitHub handle (used to credit contributed-spec PRs)."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    return {"github_handle": _profiles.get_github_handle(user_id)}


@app.put("/api/account/github-handle")
def set_account_github_handle(
    body: GithubHandleBody,
    user: dict = Depends(_auth.require_user),
):
    """Set (or clear, with null/empty) the caller's GitHub handle. When a user
    later contributes a generated spec, the PR credits them with a linkable
    Co-authored-by trailer built from this handle."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    raw = (body.github_handle or "").strip()
    handle = _profiles.normalize_handle(raw) if raw else None
    if raw and handle is None:
        raise HTTPException(
            status_code=422,
            detail="Invalid GitHub handle (1–39 chars: letters, digits, single hyphens).",
        )
    try:
        ok = _profiles.set_github_handle(user_id, handle)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=503, detail="Profile store not configured")
    return {"github_handle": handle}


# ---------------------------------------------------------------------------
# Personal access tokens (PAT) — for headless `hbench upload` from a GPU box.
# The plaintext token is returned exactly once (on issue); afterwards only
# name + prefix + timestamps are ever shown.
# ---------------------------------------------------------------------------
class TokenIssueRequest(BaseModel):
    name: str = ""


@app.get("/api/account/tokens")
def list_access_tokens(user: dict = Depends(_auth.require_user)):
    """The caller's personal access tokens (metadata only, never the hash)."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    return {"tokens": _pat.list_for_user(user_id)}


@app.post("/api/account/tokens")
def issue_access_token(body: TokenIssueRequest,
                             user: dict = Depends(_auth.require_user)):
    """Mint a token. The response carries the PLAINTEXT `token` ONCE — the
    server only stores its hash and can never show it again."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    try:
        issued = _pat.issue(user_id=user_id, name=(body.name or "").strip()[:80])
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    if issued is None:
        raise HTTPException(status_code=503, detail="Token store not configured")
    return issued


@app.delete("/api/account/tokens/{token_id}")
def revoke_access_token(token_id: str,
                              user: dict = Depends(_auth.require_user)):
    """Revoke one of the caller's tokens (owner-scoped)."""
    user_id = user.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="No user_id in token")
    ok = _pat.revoke(token_id=token_id, user_id=user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Token not found")
    return {"revoked": True}
