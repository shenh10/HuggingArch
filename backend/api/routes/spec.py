"""Spec-generation endpoints — wraps `backend.spec_worker.stream_agent`
into HTTP routes.

  • POST /generate-spec   — start a fresh spec run as a background job;
                            returns `{status: "started"}`. `resume: true`
                            continues the draft a failed run left (409
                            `no_draft` when there is none); a new run over
                            a kept draft needs `discard_draft: true` (409
                            `draft_kept` otherwise).
  • POST /refine-spec     — start a background job that fixes a
                            storage-bytes-diff > 0 spec; returns
                            `{status: "started"}`.
  • GET  /spec-status     — poll a job's state and buffered events.
  • POST /stop-spec       — kill the running agent for a model.

Both /generate-spec and /refine-spec acquire a per-model lock + global
semaphore via spec_worker.acquire(); duplicate requests for the same
model_id return immediately with `status: "duplicate"`.
"""
from __future__ import annotations


from typing import Literal

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse

from backend import spec_worker
from backend.arch.loader import SpecLoadError, SpecLoader
from backend import supabase_env
from backend.custom_id import is_custom_id, strip_prefix, model_path
from backend.custom_model import load as _load_custom
from backend.spec_worker import overlay as _overlay
from backend.spec_worker.bundle import (
    FormalSource,
    SpecBundleError,
    normalize_bundle,
)

from .. import _spec_preflight as preflight
from .._access import require_admin_if_configured
from .._app import app
from .._models import GenerateSpecRequest, RefineSpecRequest
from backend.spec_worker.outcome import RunErrorKind


# ---------------------------------------------------------------------------
# Anonymous spec-gen gate. In a MULTI-USER deployment the anonymous
# /generate-spec + /refine-spec routes are disabled: generation must go
# through the authed service route (BYOK), so an anonymous caller can neither
# burn a host key nor write the shared repo tree. "Multi-user" is
# `supabase_env.jwt_verification_available()` — the one declared switch, read
# here directly (the core may import `backend.supabase_env`; the injection
# hook this replaced existed because it could not import `service.auth`, and
# its `except Exception: return False` failed OPEN).
# ---------------------------------------------------------------------------


_ANON_DISABLED_RESPONSE = {
    "success": False, "status": "disabled",
    "message": "Sign in to generate specs on this deployment.",
}


@app.post("/generate-spec")
async def generate_spec(req: GenerateSpecRequest):
    """Trigger the configured agent CLI to generate a spec for an unknown model."""
    if supabase_env.jwt_verification_available():
        return _ANON_DISABLED_RESPONSE
    refused = preflight.agent_cli_refusal() or preflight.host_credentials_refusal()
    if refused:
        return refused

    # Custom-model branch: ground truth + target path come from the local
    # ~/.huggingarch/custom_models/<name>/ cache, not HF.
    is_custom = is_custom_id(req.model_id)
    loader = SpecLoader()
    if is_custom:
        name = strip_prefix(req.model_id)
        try:
            _load_custom(name)
        except FileNotFoundError:
            return {
                "success": False,
                "message": (
                    f"Custom model not registered: {req.model_id}. "
                    "Run `python -m backend.custom_model register` first."
                ),
            }
        # Already has a spec.yaml of its own → nothing to do. The custom
        # model's spec is the file in its directory (the same one
        # `validate._resolve_spec` loads), not whatever `match()` finds for
        # its model_type.
        spec_path = model_path(name) / "spec.yaml"
        existing = loader.load_model_spec(str(spec_path)) if spec_path.is_file() else None
        if existing is not None:
            return {"success": True, "spec_name": existing.name, "status": "cached"}
    else:
        config, refused = preflight.probe_repo(req.model_id)
        if refused:
            return refused
        cached = preflight.known_spec(req.model_id, loader, config=config)
        if cached:
            return cached

    refused = preflight.claim(req.model_id, req.model_id)
    if refused:
        return refused

    run_key = req.model_id
    try:
        target_root = _overlay.allocate_run_overlay(
            req.model_id, kind="generate", resume=req.resume,
            discard_draft=req.discard_draft)
    except _overlay.RunLeafRefusal as exc:
        spec_worker.release(run_key)
        return JSONResponse(status_code=409, content=exc.refusal())
    except BaseException:
        spec_worker.release(run_key)
        raise

    if is_custom:
        prep_fn = lambda: spec_worker.prepare_custom_generate_prompt(
            strip_prefix(req.model_id), target_root=target_root)
    else:
        prep_fn = lambda: spec_worker.prepare_generate_prompt(
            req.model_id, target_root=target_root)
    task_prep = None
    if req.resume:
        task_prep = prep_fn
        prep_fn = lambda: spec_worker.prepare_resume(
            req.model_id, target_root=target_root,
            agent_cli=spec_worker.get_agent_cli(), base_prep=task_prep)
    spec_worker.start_job(
        run_key, spec_worker.generation_job(
            req.model_id, prep_fn, run_key=run_key, target_root=target_root,
            failover=spec_worker.Failover(
                req.model_id, target_root=target_root, task_prep=task_prep)),
        model_id=req.model_id, kind="generate",
    )
    return {"status": "started", "model_id": req.model_id}


@app.post("/refine-spec")
async def refine_spec(req: RefineSpecRequest):
    """Trigger the configured agent CLI to refine a spec with params discrepancy."""
    if supabase_env.jwt_verification_available():
        return _ANON_DISABLED_RESPONSE
    refused = preflight.agent_cli_refusal() or preflight.host_credentials_refusal()
    if refused:
        return refused

    seed_yaml = _overlay.load_seed_spec_yaml(req.model_id)
    if not seed_yaml:
        return preflight.no_seed_refusal()

    try:
        seed_bundle = normalize_bundle(
            req.model_id, seed_yaml, [], FormalSource.from_deployment())
    except SpecBundleError as exc:
        first = exc.errors[0] if exc.errors else str(exc)
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "error_kind": RunErrorKind.SPEC_BUNDLE_INCOMPLETE.value,
                "status": "bundle_incomplete",
                "message": (
                    f"The current spec for {req.model_id} is incomplete: "
                    f"{first}. Regenerate it before refining."),
                "errors": exc.errors,
            },
        )
    except SpecLoadError as exc:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "error_kind": RunErrorKind.SPEC_INVALID.value,
                "status": "spec_invalid",
                "message": (
                    f"The current spec for {req.model_id} no longer validates "
                    "against the current schema. Regenerate it from the "
                    f"Inspector. Details: {exc}"),
                "errors": [str(exc)],
            },
        )

    run_key = preflight.job_run_key(req.model_id, kind="refine")
    refused = preflight.claim(run_key, req.model_id)
    if refused:
        return refused

    try:
        target_root = _overlay.allocate_run_overlay(
            req.model_id, kind="refine", seed_bundle=seed_bundle)
    except BaseException:
        spec_worker.release(run_key)
        raise

    prep_fn = lambda: spec_worker.prepare_refine_prompt(
        req.model_id,
        req.spec_storage_bytes,
        req.hf_storage_bytes,
        req.storage_diff,
        target_root=target_root,
        seed_bundle=seed_bundle,
    )
    spec_worker.start_job(
        run_key, spec_worker.generation_job(
            req.model_id, prep_fn, run_key=run_key, target_root=target_root),
        model_id=req.model_id, kind="refine",
    )
    return {"status": "started", "model_id": req.model_id, "run_key": run_key}


@app.post(
    "/stop-spec",
    dependencies=[Depends(require_admin_if_configured)],
)
async def stop_spec(req: dict):
    """Stop a running spec worker. Keyed by the opaque `run_key` when the
    caller has one, else falls back to `model_id`. In service mode this raw-key
    endpoint is admin-only; regular users use `/api/spec/stop`, which derives
    their key from the JWT. Local self-host mode remains open."""
    try:
        kind = preflight.JobKind(req.get("kind") or "generate")
    except ValueError:
        raise HTTPException(status_code=422, detail="kind must be generate or refine") from None
    key = req.get("run_key") or preflight.job_run_key(
        req.get("model_id", ""), kind=kind)
    if not key:
        return {"success": False, "message": "model_id required"}
    killed = spec_worker.stop(key)
    return {
        "success": killed,
        "message": "Stopped" if killed else "No running task found",
    }


@app.get(
    "/spec-status",
    dependencies=[Depends(require_admin_if_configured)],
)
async def spec_status(model_id: str = "", cursor: int = 0, run_key: str = "",
                      kind: Literal["generate", "refine"] = "generate"):
    """Poll a background generate/refine job. Returns state plus any events
    buffered after `cursor` (pass back the returned `next_cursor`). The
    generate/refine POSTs return `{status: "started"}` immediately and the run
    proceeds server-side; the browser polls this endpoint in local self-host
    mode. Keyed by the opaque `run_key` when provided, else `model_id` plus
    `kind` (generate vs refine use different keys). In service mode it is
    admin-only; regular users poll `/api/spec/status`. `{state: "idle"}`
    means no job exists for this key."""
    status = spec_worker.job_status(
        run_key or preflight.job_run_key(model_id, kind=kind), cursor)
    return preflight.with_resumable(status, model_id, kind)
