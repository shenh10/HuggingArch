"""A model described in words: written by an agent as the caller's own
custom model, then refined turn by turn or edited by hand.

  • POST /api/spec/describe        — first turn (`description`, optional
                                     `base_model_id`) or a follow-up
                                     (`message`); starts a background run and
                                     returns `{status: "started"}`. Progress
                                     is the generate run's: GET
                                     /api/spec/status?model_id=custom:<name>.
  • GET  /api/spec/describe/files  — the stored spec, config and revision.
  • POST /api/spec/describe/save   — a hand edit: validated like a run's
                                     output, stored only when it passes.
  • GET  /api/spec/describe/mine   — the caller's described models.

The model is `custom:<name>`, stored without a checkpoint
(`hf.metadata.stored_without_checkpoint`): validate judges it with every
check that reads the spec alone. It is kept as the caller's generation
record (`records`), its config and description in the record's snapshot, so
the Inspector and the Playground resolve it the way they resolve any record
of the caller's (`user_model_overlay`). Per-user mode only: the record, the
run and its workspace are all keyed by the caller.
"""
from __future__ import annotations

import asyncio
import functools
import re
import shutil
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend import spec_worker
from backend.api import _spec_preflight as preflight
from backend.api._app import app
from backend.api._models import ComposeModelRequest
from backend.arch.loader import SpecLoadError, get_default_loader
from backend.custom_id import registered_on_host
from backend.service import auth as _auth
from backend.service import agent_credentials as _creds
from backend.service import rate_limit as _rate_limit
from backend.spec_worker import overlay as _overlay
from backend.spec_worker import records
from backend.spec_worker.bundle import SpecBundleError

from .spec import (
    _bundle_failure,
    _claim_run,
    _normalize_record_bundle,
    _peruser_active,
    _preflight,
    _quota_refund,
    _records_unavailable,
    _run_stamp,
    user_model_overlay,
)

#: A custom model name: what `custom:<name>` may carry on disk and in a URL.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class DescribeModelRequest(BaseModel):
    name: str
    # The description the model is written from, on the first turn. A
    # follow-up reads it from the stored record.
    description: Optional[str] = None
    base_model_id: Optional[str] = None
    # A follow-up turn's change request; absent on the first turn.
    message: Optional[str] = None


class SaveDescribedModelRequest(BaseModel):
    name: str
    spec_yaml: str
    config: dict
    # The record revision the edit was made on (`/files`); a save over a
    # model that changed since is refused, not merged.
    revision: Optional[int] = None


class _Refused(Exception):
    """An answer a route gives from inside a helper."""

    def __init__(self, response):
        super().__init__()
        self.response = response


def _answers_refusals(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _Refused as refused:
            return refused.response
    return run


def _model_id(name: str) -> str:
    name = (name or "").strip()
    if not _NAME.fullmatch(name):
        raise HTTPException(status_code=400, detail=(
            "A model name is 1-64 letters, digits, '.', '_' or '-', starting "
            "with a letter or digit."))
    return f"custom:{name}"


def _caller(user: dict) -> str:
    """The caller, who owns the record, the run and its workspace."""
    if not _peruser_active():
        raise _Refused({"success": False, "status": "per_user_off", "message": (
            "Describing a model keeps it as your own; this deployment runs "
            "spec generation without per-user records.")})
    return user.get("sub")


def _record(user_id: str, model_id: str) -> Optional[dict]:
    from backend.service import spec_generations
    try:
        return records.serving_generation(user_id, model_id)
    except spec_generations.SpecGenerationsUnavailable:
        raise _Refused(_records_unavailable()) from None


@app.post("/api/spec/describe")
@_answers_refusals
def describe_model(req: DescribeModelRequest,
                   user: dict = Depends(_auth.require_user)):
    """Start a turn on a model described in words. The first turn writes it
    from `description` (and `base_model_id`, a library model it builds on);
    a turn with `message` changes the model as stored — including any hand
    edit — re-entering the agent session when it can."""
    user_id = _caller(user)
    is_admin = _auth.is_admin(user_id)
    model_id = _model_id(req.name)
    name = model_id.removeprefix("custom:")
    if registered_on_host(model_id):
        # A custom model registered on this host is the host's: its dir
        # answers for that id (`hf.metadata.fetch_config`).
        return JSONResponse(status_code=409, content={
            "success": False, "status": "name_taken",
            "message": f"{model_id} is a model registered on this server. "
                       f"Pick another name."})
    row = _record(user_id, model_id)
    follow_up = req.message is not None
    if follow_up:
        if not (row and row.get("spec_yaml")):
            return JSONResponse(status_code=404, content={
                "success": False, "status": "no_model",
                "message": f"You have no model {model_id} to change yet."})
        kept = row.get("snapshot_json") or {}
        description = kept.get("description") or ""
        base_model_id = kept.get("base_model_id")
        try:
            bundle = _normalize_record_bundle(
                model_id, row["spec_yaml"], row.get("new_component_yamls"))
        except (SpecBundleError, SpecLoadError) as exc:
            return _bundle_failure(exc, model_id=model_id)
    else:
        if row:
            return JSONResponse(status_code=409, content={
                "success": False, "status": "name_taken",
                "message": f"You already have {model_id}. Pick another name, "
                           f"or ask for a change to it."})
        description = (req.description or "").strip()
        if not description:
            raise HTTPException(status_code=400, detail="Describe the model.")
        base_model_id = req.base_model_id or None
        if base_model_id and get_default_loader().spec_for_model_id(
                base_model_id) is None:
            return {"success": False, "status": "unknown_base", "message": (
                f"{base_model_id} is not a model this server serves. Describe "
                f"the model without it, or pick one from the library.")}

    resolved, refused = _preflight(user_id, is_admin, "generate")
    if refused:
        return refused
    run_key = preflight.job_run_key(model_id, kind="generate", user_id=user_id)
    refused = _claim_run(run_key, model_id, user_id, is_admin)
    if refused:
        return refused
    try:
        if follow_up:
            target_root = _overlay.continue_from_record(
                user_id, model_id, bundle, row.get("snapshot_json"))
        else:
            target_root = _overlay.allocate_run_overlay(
                model_id, kind="generate", user_id=user_id, discard_draft=True)
    except BaseException:
        spec_worker.release(run_key)
        if not is_admin:
            _quota_refund(user_id)
        raise

    byok_cli = resolved["agent_cli"] if resolved else None
    byok_env = resolved["env"] if resolved else None
    run_cli = byok_cli or spec_worker.get_agent_cli()
    task_prep = lambda: spec_worker.prepare_describe_prompt(
        name, description, target_root=target_root,
        base_model_id=base_model_id)
    run_prep = failover_prep = task_prep
    if follow_up:
        run_prep = lambda: spec_worker.prepare_resume(
            model_id, target_root=target_root, agent_cli=run_cli,
            base_prep=task_prep, request=req.message)

        def failover_prep() -> dict:
            # A backend this turn fails over to cold-starts: it needs the
            # description and the change request, not the description alone.
            from backend.spec_worker.resume import continuation_prompt
            prep = task_prep()
            prep["prompt"] = (
                f"{prep['prompt'].rstrip()}\n\n---\n\n"
                f"{continuation_prompt(model_id, target_root, None, req.message)}")
            return prep

    def prep_fn() -> dict:
        prep = run_prep()
        prep["agent_fallbacks"] = _creds.fallback_envs(
            user_id, run_cli, task="generate", host_allowed=is_admin)
        prep["snapshot_fields"] = {"description": description,
                                   "base_model_id": base_model_id}
        return prep

    spec_worker.start_job(
        run_key,
        spec_worker.generation_job(model_id, prep_fn, run_key=run_key,
                                   stamp=_run_stamp(user_id, byok_cli, byok_env),
                                   target_root=target_root,
                                   failover=spec_worker.Failover(
                                       model_id, target_root=target_root,
                                       task_prep=failover_prep)),
        model_id=model_id, user_id=user_id, kind="generate",
        agent_cli=byok_cli, agent_env=byok_env,
    )
    return {"status": "started", "model_id": model_id, "run_key": run_key}


@app.get("/api/spec/describe/files")
@_answers_refusals
def described_model_files(name: str, user: dict = Depends(_auth.require_user)):
    """The stored spec, config and revision of one of the caller's models."""
    model_id = _model_id(name)
    row = _record(_caller(user), model_id)
    if not (row and row.get("spec_yaml")):
        raise HTTPException(status_code=404, detail=f"No model {model_id}.")
    kept = row.get("snapshot_json") or {}
    return {"success": True, "model_id": model_id,
            "spec_yaml": row["spec_yaml"], "config": kept.get("config") or {},
            "description": kept.get("description"),
            "revision": row.get("revision"),
            "record_status": row.get("status")}


@app.post("/api/spec/describe/save")
async def save_described_model(req: SaveDescribedModelRequest,
                               user: dict = Depends(_auth.require_user)):
    """A hand edit of a described model. Judged by the same validate a run's
    output is, in a throwaway overlay; stored as the model only when it
    passes, otherwise the errors come back and the stored model stands.
    Holds the model's run lock, so an edit and an agent turn never write
    the record over one another."""
    _rate_limit.starts_work()       # each save runs one validate subprocess
    return await asyncio.to_thread(_save, req, user)


@_answers_refusals
def _save(req: SaveDescribedModelRequest, user: dict):
    from backend.spec_worker.worker import _invoke_validate

    user_id = _caller(user)
    model_id = _model_id(req.name)
    run_key = preflight.job_run_key(model_id, kind="generate", user_id=user_id)
    claimed = spec_worker.acquire(run_key)
    if claimed != "ok":
        return JSONResponse(status_code=409 if claimed == "duplicate" else 503, content={
            "success": False, "status": "busy" if claimed == "duplicate" else claimed,
            "message": ("An agent turn on this model is running; save after it ends."
                        if claimed == "duplicate" else
                        "The server is busy; try saving again in a moment.")})
    scratch = None
    try:
        row = _record(user_id, model_id)
        if not (row and row.get("spec_yaml")):
            raise HTTPException(status_code=404, detail=f"No model {model_id}.")
        if req.revision is not None and row.get("revision") != req.revision:
            return JSONResponse(status_code=409, content={
                "success": False, "status": "stale",
                "message": "The model changed since you opened it (an agent "
                           "turn or another save). Reload the files and edit again."})
        try:
            bundle = _normalize_record_bundle(
                model_id, req.spec_yaml, row.get("new_component_yamls"))
        except (SpecBundleError, SpecLoadError) as exc:
            return _bundle_failure(exc, model_id=model_id)
        kept = row.get("snapshot_json") or {}
        snapshot = {"model_id": model_id, "config": req.config,
                    **{k: kept[k] for k in ("description", "base_model_id")
                       if k in kept}}
        scratch = _overlay.scratch_user_spec(user_id, model_id, bundle, snapshot)
        try:
            verdict = _invoke_validate(model_id, None, scratch)
        except Exception as exc:  # infrastructure: config / subprocess
            return {"success": False, "status": "validate_failed",
                    "message": f"Validation could not run: {exc}"}
        errors = list(verdict.get("errors") or [])
        if errors:
            return {"success": False, "status": "invalid", "errors": errors,
                    "checks": verdict.get("checks")}
        recorded = records.record(
            model_id=model_id, user_id=user_id, spec_yaml=req.spec_yaml,
            snapshot_json=snapshot,
            new_component_yamls=row.get("new_component_yamls"))
        return {"success": True, "model_id": model_id,
                "checks": verdict.get("checks"), "record": recorded}
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
        spec_worker.release(run_key)


@app.get("/api/spec/describe/mine")
@_answers_refusals
def my_described_models(user: dict = Depends(_auth.require_user)):
    """The caller's custom models, newest first."""
    rows = records.list_for_owner(_caller(user))
    return {"models": [
        {"model_id": r.get("model_id"), "status": r.get("status"),
         "updated_at": r.get("updated_at") or r.get("created_at")}
        for r in rows if str(r.get("model_id") or "").startswith("custom:")]}


# ---------------------------------------------------------------------------
# The Playground for a signed-in caller. The core playground routes are
# anonymous and serve the library; these resolve the caller's own record
# first (`user_model_overlay`, as the Inspector's analyze does) and hand its
# serving overlay to the same functions. A model the caller has no record of
# is served as the core routes serve it.
# ---------------------------------------------------------------------------


def _caller_model(user_id: str, model_id: str):
    """`(overlay, served, refusal)` for the model a signed-in caller opens in
    the Playground: the library's when the library serves it — as analyze
    answers, the library first — else the caller's own record
    (`user_model_overlay`)."""
    if get_default_loader().spec_for_model_id(model_id) is not None:
        return None, {}, None
    return user_model_overlay(user_id, model_id)


def _owned_skeleton(user_id: str, model_id: str) -> dict:
    from backend.playground.compose import model_skeleton
    ov, served, refusal = _caller_model(user_id, model_id)
    if refusal is not None:
        return refusal
    try:
        return {"success": True, **model_skeleton(model_id, ov), **served}
    except (ValueError, KeyError) as exc:
        return {"success": False, "message": str(exc)}


def _owned_compose(user_id: str, req: ComposeModelRequest) -> dict:
    from backend.playground.compose import compose_groups
    ov, _served, refusal = _caller_model(user_id, req.base_model_id)
    if refusal is not None:
        return refusal
    try:
        return compose_groups(
            base_model_id=req.base_model_id,
            groups=[g.model_dump() for g in req.groups],
            dim_overrides=dict(req.dim_overrides),
            batch=req.batch_size,
            input_tokens=req.seq_len,
            precision=req.precision_profile(),
            kv_cache_lens=list(req.kv_cache_lens),
            gpu_sku=req.gpu_sku,
            overlay_dir=ov,
        )
    except (ValueError, KeyError) as exc:
        return {"success": False, "message": str(exc)}


@app.get("/api/account/playground/model-skeleton")
async def account_model_skeleton(model_id: str,
                                 user: dict = Depends(_auth.require_user)):
    from backend.api.offload import offload
    return await offload(_owned_skeleton, user.get("sub"), model_id)


@app.post("/api/account/playground/compose-model")
async def account_compose_model(req: ComposeModelRequest,
                                user: dict = Depends(_auth.require_user)):
    from backend.api.offload import offload
    return await offload(_owned_compose, user.get("sub"), req)
