"""The checks every spec-run route makes before it starts a run.

The core routes (`/generate-spec`, `/refine-spec`) and the signed-in service
routes (`/api/spec/generate`, `/api/spec/refine`) refuse a run for the same
reasons with the same responses: the agent CLI is misconfigured or missing,
a run on the host's credentials has none to use,
the Hugging Face repo is missing / gated / has no config, the model already
has a spec, or its run lock is taken. Each check returns the refusal body,
or None to go on.

Workers are reached through the `spec_worker` package (not its submodules)
so a test that stubs `spec_worker.acquire` / `get_agent_binary` stubs every
route at once.
"""
from __future__ import annotations

import json
import shutil
from enum import Enum
from typing import Optional

from fastapi.responses import JSONResponse

from backend import spec_worker
from backend.agents.base import HostCredentialsMissing, host_has_credentials
from backend.analyzer import classify_config_fetch_failure
from backend.arch.loader import SpecLoader
from backend.arch.snapshots import load_config as load_snapshot_config
from backend.hf.task_gate import unsupported_task_refusal
from backend.spec_worker import overlay as _overlay
from backend.custom_id import is_custom_id


def agent_cli_refusal() -> Optional[dict]:
    """Refusal when the configured coding agent is invalid or its CLI is not
    on PATH."""
    try:
        agent_binary = spec_worker.get_agent_binary()
    except ValueError as e:
        return {"success": False, "message": f"Invalid coding agent configuration: {e}"}
    if not shutil.which(agent_binary):
        return {"success": False, "message": "Coding agent CLI not installed"}
    return None


def host_credentials_refusal() -> Optional[dict]:
    """Refusal when a run on the host's own credentials could not
    authenticate: the server's environment holds no credential for the
    configured agent (the agent's private HOME has no login state to fall
    back on — `AgentBackend._run_cwd_and_env` refuses the same run at spawn)."""
    cli = spec_worker.get_agent_cli()
    if host_has_credentials(cli):
        return None
    return {"success": False, "status": "host_credentials_missing",
            "message": str(HostCredentialsMissing(cli))}


def probe_repo(model_id: str) -> tuple[Optional[dict], Optional[dict]]:
    """One config fetch for generate preflight.

    Returns `(config, None)` when `config.json` is readable (snapshot first,
    then Hub), `(None, refusal)` for a permanent repo failure (missing,
    gated, invalid id, no config.json), and `(None, None)` for a custom id
    or a network blip the run can still retry.

    Classifies the same exception the download raised, so generate does not
    probe Hub three times and a missing `config.json` cannot 500 before the
    refusal. Custom models are not HF repos; their own path already refuses
    an unregistered id.
    """
    if is_custom_id(model_id):
        return None, None
    config = load_snapshot_config(model_id)
    if config is not None:
        return config, None
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception as exc:
        kind, message = classify_config_fetch_failure(model_id, cause=exc)
        if kind not in ("invalid_id", "gated", "not_found", "no_config"):
            return None, None
        return None, {
            "success": False, "status": "repo_unavailable", "message": message,
        }
    # The config is readable, but the hub task may still be outside what a
    # spec run can describe (a diffusion repo with a root config.json) —
    # refuse before the agent run starts.
    refusal = unsupported_task_refusal(model_id)
    if refusal:
        return None, {
            "success": False, "status": "repo_unavailable", "message": refusal,
        }
    return config, None


def repo_refusal(model_id: str) -> Optional[dict]:
    """Refusal when the Hugging Face repo cannot be read — missing, gated,
    an invalid id, or no config.json. None when the config is reachable,
    or when the failure is a network blip the run can still retry."""
    _, refused = probe_repo(model_id)
    return refused


def known_spec(model_id: str, loader: Optional[SpecLoader] = None,
               config: Optional[dict] = None) -> Optional[dict]:
    """The `cached` answer when the HF model's config already matches a spec;
    None when it does not, or when no config was obtained (the run then
    decides). `config` is the dict `probe_repo` already fetched — this
    never downloads. `loader` is the caller's when it has built one
    already; otherwise one is built here, and a loader that fails to
    build (OSError / ValueError) counts as no match."""
    if config is None:
        return None
    try:
        existing = (loader if loader is not None else SpecLoader()).match(
            config, model_id=model_id)
        if existing:
            return {"success": True, "spec_name": existing.spec.name, "status": "cached"}
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return None


class JobKind(str, Enum):
    """The run kinds a lock/job key tells apart. Refine owns its own key so
    an in-flight refine cannot refuse regenerate (and vice versa)."""

    GENERATE = "generate"
    REFINE = "refine"


def job_run_key(model_id: str, *, kind: str = "generate", user_id: str = "") -> str:
    """Lock/job key for a generate or refine run. Refine is a different key
    so an in-flight refine cannot refuse regenerate (and vice versa).
    `kind` is a closed vocabulary — unknown values raise."""
    try:
        kind = JobKind(kind)
    except ValueError:
        raise ValueError(
            f"unknown job kind {kind!r}; expected generate or refine"
        ) from None
    leaf = f"refine|{model_id}" if kind is JobKind.REFINE else model_id
    return f"{user_id}|{leaf}" if user_id else leaf


def with_resumable(status: dict, model_id: str, kind: str, *,
                   user_id: str = "") -> dict:
    """A generate job's status with `resumable`: whether the caller's
    generate leaf keeps a draft a continued generation picks up. Stated
    whenever no run is in flight and none succeeded — after a failed `done`,
    an interrupted run, a crash, or with no job at all (a page reload, a
    server restart) — so the entry point outlives the browser that saw the
    run fail."""
    if (kind != JobKind.GENERATE.value or not model_id
            or status.get("state") == "running" or status.get("success")):
        return status
    leaf = _overlay.generate_leaf(user_id, model_id)
    return {**status, "resumable": _overlay.has_resumable_draft(leaf)}


def claim(run_key: str, model_id: str) -> Optional[dict]:
    """Take the run lock for `run_key`; the refusal when another run holds
    it or every slot is busy, None once it is this caller's."""
    status = spec_worker.acquire(run_key)
    if status == "duplicate":
        return {"success": False, "status": "duplicate",
                "message": f"Already analyzing {model_id}"}
    if status == "overloaded":
        return {"success": False, "status": "overloaded",
                "message": "Too many concurrent tasks, please retry later"}
    return None


def no_seed_refusal() -> JSONResponse:
    """409 when a refine finds no spec on disk or in the cache to edit: the
    model has to be generated first."""
    return JSONResponse(status_code=409, content={
        "success": False,
        "status": "no_spec",
        "message": "No spec found for this model — generate one first.",
    })
