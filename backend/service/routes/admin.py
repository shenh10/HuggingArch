"""Admin REST endpoints.

  POST /api/admin/users/invite            — invite a user by email
                                             (Supabase sends a sign-up link)
  GET  /api/admin/users/:user_id/summary  — drill-down stats
  GET  /api/admin/users/:user_id/events   — paginated analyze_logs
  POST /api/admin/users/:user_id/delete   — cascade delete from public
                                             tables + auth.users
  POST /api/admin/update-model-stack      — dispatch the update-transformers
                                             workflow (relock + gate + PR)

The admin page (web/src/pages/Admin.tsx) calls these with the user's
JWT (admin-gated by the SQL functions). The DELETE / invite flows need the service-role key
because they touch auth.users via the Auth Admin API.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import Depends, HTTPException
from pydantic import BaseModel
import httpx

from backend import supabase_rest
from backend.api._app import app
from backend.api._spec_preflight import host_credentials_refusal
from backend.service import auth as _auth
from backend.service import spec_generations as _gens
from backend.service import agent_credentials as _creds
from backend.service.routes.spec import (
    launch_recheck,
    launch_review,
    mark_engine_changed,
    add_library_standing,
    library_standing,
    superseded_by,
    superseded_refusal,
)
from backend.service import github as _github
from backend.service import profiles as _profiles
from backend.spec_worker.bundle import record_fingerprint
from backend.spec_worker.outcome import PENDING, GenerationStatus, RunErrorKind

logger = logging.getLogger(__name__)


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class InviteUserRequest(BaseModel):
    email: str
    username: str | None = None


@app.post("/api/admin/users/invite")
def admin_invite_user(
    req: InviteUserRequest,
    admin: dict = Depends(_auth.require_admin),
):
    """Send an invite email to `email`. Supabase mints a magic-link
    sign-up URL; the recipient clicks it, sets a password, and lands
    in the app already signed in. The frontend's profile-create hook
    fires on first session, so the user shows up in the admin list
    immediately after they accept.

    `username` (optional) is stashed in user_metadata so it pre-fills
    the profile on first login.
    """
    email = (req.email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Invalid email")

    url = _auth._supabase_url()
    key = _auth._service_role_key()
    if not (url and key):
        raise HTTPException(status_code=503, detail="Service-role key not configured")

    payload: dict = {"email": email}
    if req.username:
        payload["data"] = {"username": req.username.strip()}

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{url.rstrip('/')}/auth/v1/admin/invite",
                json=payload,
                headers=supabase_rest.service_headers(key),
            )
        if resp.status_code >= 400:
            # Surface Supabase's error string for the dashboard to show.
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        body = resp.json() if resp.content else {}
        return {"invited": True, "email": email, "user_id": body.get("id")}
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Auth Admin invite errored: {e}") from e


@app.get("/api/admin/users/{user_id}/summary")
def admin_user_summary(user_id: str, token: str = Depends(_auth.admin_token)):
    """Headline stats for one user — drives the dashboard drill-down
    card. Wraps the get_user_summary() SQL function via the user's JWT
    so RLS is honored (and anyone calling this without admin gets a
    Postgres-level reject inside the function)."""
    return _call_admin_rpc(
        "get_user_summary", {"p_user_id": user_id}, user_jwt=token
    )


@app.get("/api/admin/users/{user_id}/events")
def admin_user_events(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    token: str = Depends(_auth.admin_token),
):
    """Last N analyze_logs for the given user (newest first)."""
    return _call_admin_rpc(
        "get_user_analyze_logs",
        {"p_user_id": user_id, "p_limit": limit, "p_offset": offset},
        user_jwt=token,
    )


@app.post("/api/admin/users/{user_id}/delete")
def admin_delete_user(
    user_id: str,
    admin: dict = Depends(_auth.require_admin),
    token: str = Depends(_auth.admin_token),
):
    """Two-step: cascade-delete from public tables, then purge
    auth.users via the Supabase Auth Admin API. Returns the user_id
    for client confirmation.

    Why two calls: SECURITY DEFINER on a public function can't reach
    auth.users (Supabase reserves it). The Auth Admin API is the only
    sanctioned path. We do the cascade-delete first so a partial
    failure can't leave dangling rows referencing a now-gone auth.user.
    """
    if user_id == admin.get("sub"):
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    # Step 1: Postgres-side cascade. delete_user_data() raises on the
    # last-admin guard; surface that as 400 to the dashboard.
    try:
        _call_admin_rpc("delete_user_data", {"p_user_id": user_id}, user_jwt=token)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Public-side delete failed: {e}") from e
    # The public rows (their admins row among them) are gone: whatever the
    # auth step below does, this process stops treating them as an admin.
    _auth.forget_admin(user_id)

    # Step 2: drop auth.users. If this fails, the user is already gone
    # from the public schema — they can't sign in again because no
    # profile row exists, but their auth.users entry persists. Surface
    # the partial-success state so the admin can retry.
    url = _auth._supabase_url()
    key = _auth._service_role_key()
    if not (url and key):
        raise HTTPException(status_code=503, detail="Service-role key not configured")

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(
                f"{url.rstrip('/')}/auth/v1/admin/users/{user_id}",
                headers=supabase_rest.service_headers(key, json_body=False),
            )
        if resp.status_code not in (200, 204):
            raise HTTPException(
                status_code=502,
                detail=f"Auth Admin delete failed ({resp.status_code}): {resp.text}",
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Auth Admin call errored: {e}") from e

    return {"deleted": True, "user_id": user_id}


# ---------------------------------------------------------------------------
# Cached specs (Phase B: admin "Cached Models" tab)
# ---------------------------------------------------------------------------
# Reads from public.cached_specs (the cross-VM spec cache populated by
# spec_worker after a successful agent run). Lets admin see what's
# stored, drill into the YAML/components for review, and delete cache
# rows. Promotion-to-PR is Phase C — not in this batch.

@app.get("/api/admin/cached-specs")
def admin_cached_specs_list(
    limit: int = 50,
    offset: int = 0,
    sort: str = "hits",  # 'hits' | 'recent' | 'oldest'
    token: str = Depends(_auth.admin_token),
):
    """Paginated list of cached specs. Returns summary fields only —
    spec_yaml / new_component_yamls payloads are heavy and only loaded
    on detail click (the drawer)."""
    if sort not in ("hits", "recent", "oldest"):
        raise HTTPException(status_code=400, detail=f"Invalid sort: {sort!r}")
    rows = _call_admin_rpc(
        "get_cached_specs",
        {"p_limit": limit, "p_offset": offset, "p_sort": sort},
        user_jwt=token,
    )
    total = _call_admin_rpc("get_cached_specs_count", {}, user_jwt=token)
    return {"items": rows or [], "total": total or 0}


@app.get("/api/admin/cached-specs/{model_id:path}")
def admin_cached_spec_detail(
    model_id: str,
    token: str = Depends(_auth.admin_token),
):
    """Full payload for one cached spec: yaml, snapshot, every new
    component yaml. Powers the drawer's preview panes.

    `:path` converter so model_ids containing '/' (the common case —
    'meta-llama/Llama-3') aren't URL-decoded into 404s."""
    rows = _call_admin_rpc(
        "get_cached_spec_detail", {"p_model_id": model_id}, user_jwt=token,
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No cached spec for {model_id}")
    return rows[0]


@app.delete("/api/admin/cached-specs/{model_id:path}")
def admin_delete_cached_spec(
    model_id: str,
    token: str = Depends(_auth.admin_token),
):
    """Drop one cached spec row. The model's git-tracked spec (if any)
    is unaffected — this only purges the cross-VM cache layer.
    Subsequent requests for the same model_id either fall through to a
    built-in spec or trigger a fresh agent run (a cache miss)."""
    deleted = _call_admin_rpc(
        "delete_cached_spec", {"p_model_id": model_id}, user_jwt=token,
    )
    return {"deleted": bool(deleted), "model_id": model_id}


# ---------------------------------------------------------------------------
# Spec generations (Phase 1: admin "Spec generations" tab)
# ---------------------------------------------------------------------------
# Reads from public.spec_generations (the per-user append log written by the
# service /api/spec/generate route after a successful run). Unlike cached_specs
# (one row per model), the same model can appear multiple times — one row per
# user who generated it. The author opens the PR after a PASS review.

def _vocabulary_value(enum_cls, value: Optional[str], name: str) -> Optional[str]:
    """`value` when it is one of `enum_cls`'s values (None passes: no
    filter); 400 otherwise, like an unknown sort."""
    if value is None or value == "":
        return None
    if value not in {member.value for member in enum_cls}:
        raise HTTPException(status_code=400, detail=f"Invalid {name}: {value!r}")
    return value


@app.get("/api/admin/spec-generations")
def admin_spec_generations_list(
    limit: int = 50,
    offset: int = 0,
    sort: str = "recent",  # 'recent' | 'oldest' | 'model'
    status: Optional[str] = None,
    token: str = Depends(_auth.admin_token),
):
    """Paginated list of per-user spec generations, optionally only one
    `status` (`GenerationStatus`). Summary fields only — the spec_yaml /
    component payloads load on detail click. Each row names its
    `library_spec` / `superseded_by` (`library_standing`), and an
    engine-blocked row whether the engine changed since (`engine_changed`)."""
    if sort not in ("recent", "oldest", "model"):
        raise HTTPException(status_code=400, detail=f"Invalid sort: {sort!r}")
    status = _vocabulary_value(GenerationStatus, status, "status")
    rows = _call_admin_rpc(
        "get_spec_generations",
        {"p_limit": limit, "p_offset": offset, "p_sort": sort,
         "p_status": status},
        user_jwt=token,
    )
    total = _call_admin_rpc("get_spec_generations_count",
                            {"p_status": status}, user_jwt=token)
    rows = rows or []
    add_library_standing(rows)
    mark_engine_changed(rows)
    return {"items": rows, "total": total or 0}


@app.get("/api/admin/spec-generation-runs")
def admin_spec_generation_runs(
    limit: int = 50,
    offset: int = 0,
    error_kind: Optional[str] = None,
    token: str = Depends(_auth.admin_token),
):
    """Spec runs from their log rows, newest first, optionally only one
    `error_kind` (`RunErrorKind`). The one place a run without a record
    shows: an anonymous or failed-to-record blocked run, and a run left with
    only refusals whose raise site declared no owner."""
    error_kind = _vocabulary_value(RunErrorKind, error_kind, "error_kind")
    rows = _call_admin_rpc(
        "get_spec_generation_runs",
        {"p_limit": limit, "p_offset": offset, "p_error_kind": error_kind},
        user_jwt=token,
    )
    total = _call_admin_rpc("get_spec_generation_runs_count",
                            {"p_error_kind": error_kind}, user_jwt=token)
    return {"items": rows or [], "total": total or 0}


@app.get("/api/admin/spec-generations/{record_id}")
def admin_spec_generation_detail(
    record_id: str,
    token: str = Depends(_auth.admin_token),
):
    """Full payload for one generation record (by uuid): yaml, snapshot,
    new component yamls, and its `library_standing`. Powers the drawer's
    preview panes.

    Poll-on-view merge detection: if the record has an open PR, check its state
    and flip the record to `merged` once the PR lands (no webhook infra)."""
    rows = _call_admin_rpc(
        "get_spec_generation_detail", {"p_id": record_id}, user_jwt=token,
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No generation record {record_id}")
    row = rows[0]
    # Poll-on-view: reconcile the PR state (→ merged / closed) with no webhook.
    new = _gens.sync_pr_status(row)
    if new:
        row["status"] = new
    row.update(library_standing(row))
    mark_engine_changed([row])
    return row


def _contributor_for(user_id: Optional[str]) -> Optional[str]:
    """Best contributor string for the PR: a linkable ``handle <noreply>`` when
    the user has set a GitHub handle, else the internal user id (body credit
    only). Never raises — credit is best-effort."""
    if not user_id:
        return None
    try:
        handle = _profiles.get_github_handle(user_id)
        if handle:
            return _github.coauthor_for_handle(handle) or user_id
    except Exception:
        # Credit is best-effort — never block PR creation on a lookup error.
        logger.warning("contributor handle lookup failed", exc_info=True)
    return user_id


@app.post("/api/admin/spec-generations/{record_id}/reject")
def admin_reject_spec(
    record_id: str,
    admin: dict = Depends(_auth.require_admin),
):
    """Mark a generation record as rejected (won't be contributed). A pending
    record (engine-blocked / needs repair) is discarded instead: it is not a
    candidate, and turned `rejected` it would become a second working
    candidate beside the user's own."""
    rec = _gens.get(record_id)
    if rec and rec.get("status") in PENDING:
        if not _gens.discard_pending(record_id):
            raise HTTPException(status_code=409, detail="Could not discard this record; reload it")
        return {"rejected": True, "discarded": True, "id": record_id}
    ok = _gens.set_status(record_id, GenerationStatus.REJECTED.value)
    if not ok:
        raise HTTPException(status_code=400, detail="Could not reject this record")
    return {"rejected": True, "id": record_id}


@app.post("/api/admin/spec-generations/{record_id}/recheck")
def admin_recheck_spec(
    record_id: str,
    admin: dict = Depends(_auth.require_admin),
):
    """Re-check any pending record against the current engine
    (`launch_recheck`); poll GET /api/spec/recheck-status."""
    rec = _gens.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="No such generation")
    return launch_recheck(rec, actor_id=admin.get("sub"))


@app.post("/api/admin/spec-generations/{record_id}/review")
def admin_review_spec(
    record_id: str,
    admin: dict = Depends(_auth.require_admin),
):
    """Start an agent review of any generation (admin). Uses the admin's own
    BYOK creds if set, else the host's credentials when it has them. Returns
    {status:"started", run_key}; poll GET /api/spec/review-status."""
    admin_id = admin.get("sub")
    rec = _gens.get(record_id)
    if not rec:
        raise HTTPException(status_code=404, detail="No such generation")
    library_spec = superseded_by(rec)
    if library_spec:
        return superseded_refusal(rec["model_id"], library_spec, "Review")
    try:
        resolved = _creds.resolve_env(admin_id, task="review")
    except _creds.CredentialDecryptError:
        # The admin's own key, unreadable: never the host's instead.
        return {"status": "credential_undecryptable",
                "message": "Your saved agent API key could not be decrypted — please re-enter it in account settings."}
    except _creds.CredentialRefusal as exc:
        # The admin's own keys cannot be chosen or read: not the host's.
        return exc.refusal("run a review")
    if resolved is None:
        refused = host_credentials_refusal()
        if refused:
            return {"status": refused["status"], "message": refused["message"]}
    return launch_review(
        rec, reviewer_id=admin_id, reviewer_role="admin",
        agent_cli=(resolved or {}).get("agent_cli"),
        agent_env=(resolved or {}).get("env"))


@app.get("/api/admin/spec-generations/{record_id}/reviews")
def admin_list_spec_reviews(
    record_id: str,
    token: str = Depends(_auth.admin_token),
):
    """All reviews of a record (any reviewer), for the admin drawer. Before
    migration 0038 the RPC returns rows without `admission`; those are
    marked `admission_unavailable` so the drawer says so. Each row says
    whether it judged the bundle the record holds now (`bundle_current`)."""
    rows = _call_admin_rpc(
        "get_spec_generation_reviews", {"p_record_id": record_id}, user_jwt=token) or []
    rec = _gens.get(record_id) if rows else None
    current = record_fingerprint(rec) if rec else None
    for row in rows:
        if "admission" not in row:
            row["admission_unavailable"] = True
        row["bundle_current"] = bool(current) and row.get("bundle_fingerprint") == current
    return {"items": rows}


# ---------------------------------------------------------------------------
# Model-stack refresh (admin "Update transformers" button)
# ---------------------------------------------------------------------------

@app.post("/api/admin/update-model-stack")
def admin_update_model_stack(admin: dict = Depends(_auth.require_admin)):
    """Fire the update-transformers GitHub Actions workflow NOW instead of
    waiting for the weekly cron. The workflow re-locks the model-loading
    stack, runs the golden + test gates, and opens a PR — nothing on this
    deployment changes until a human merges that PR and deploy ships it.
    Returns the Actions page URL so the dashboard can link to the run."""
    try:
        url = _github.trigger_model_stack_update()
    except _github.GitHubNotConfigured:
        raise HTTPException(
            status_code=503,
            detail="GitHub auth not configured (HA_GITHUB_APP_* or "
                   "HA_GITHUB_TOKEN).") from None
    except httpx.HTTPStatusError as e:
        # 403 here almost always means the App/PAT lacks Actions: write —
        # say so explicitly; that's the one permission the spec-PR flow
        # didn't already require.
        raise HTTPException(
            status_code=502,
            detail=f"GitHub dispatch failed ({e.response.status_code}): "
                   f"{e.response.text}") from e
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502, detail=f"GitHub dispatch errored: {e}") from e
    return {"triggered": True, "workflow_url": url}


def _call_admin_rpc(fn_name: str, params: dict, *, user_jwt: str):
    """Invoke a Postgres function via the REST RPC endpoint as the
    admin user. Passes:

      • `apikey: <anon>`               — required for REST entry
      • `Authorization: Bearer <jwt>`  — the admin's user JWT, so
                                          `auth.uid()` inside the SQL
                                          function resolves to their id

    The SQL function then runs `IF NOT public.is_admin(auth.uid())` as
    a real defense-in-depth check, NOT the all-pass-or-all-fail joke
    you get when this header carries the service-role key (which has
    no identity → `auth.uid()` is NULL → `is_admin(NULL)` rejects every
    call, silently breaking the entire admin Cached-Specs panel).

    Service-role is still used elsewhere (auth admin invite, table-
    level operations that need RLS bypass) — but for SECURITY DEFINER
    functions whose own body gates on user identity, we forward the
    user's JWT.
    """
    url = _auth._supabase_url()
    anon = _auth._anon_key()
    if not (url and anon):
        raise HTTPException(status_code=503, detail="Supabase anon key not configured")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{url.rstrip('/')}/rest/v1/rpc/{fn_name}",
                json=params,
                headers={
                    "apikey": anon,
                    "Authorization": f"Bearer {user_jwt}",
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=resp.status_code, detail=resp.text or fn_name
            )
        return resp.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Supabase RPC errored: {e}") from e


class AgentTaskDefaultRequest(BaseModel):
    agent_cli: str
    task: str
    model: str | None = None  # blank/None clears the row


@app.get("/api/admin/agent-task-defaults")
def admin_get_task_defaults(admin: dict = Depends(_auth.require_admin)):
    """The site-wide per-task default models (migration 0039) plus the
    vocabulary the editor needs (agents, tasks)."""
    return {
        "defaults": _creds.fetch_task_defaults(),
        "agents": list(_creds.VALID_AGENTS),
        "tasks": [{"key": t.value, "label": t.label} for t in _creds.AgentTask],
    }


@app.put("/api/admin/agent-task-defaults")
def admin_put_task_default(
    req: AgentTaskDefaultRequest,
    admin: dict = Depends(_auth.require_admin),
):
    """Set or clear one (agent_cli, task) site default. A run resolves its
    model as: the credential's per-task pin → this table → the credential's
    model → the endpoint default."""
    try:
        _creds.put_task_default(
            agent_cli=req.agent_cli, task=req.task, model=req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"ok": True}
