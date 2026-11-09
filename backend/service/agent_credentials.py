"""Per-user BYOK agent credentials (Phase 2).

Stores each user's own LLM key + endpoint (encrypted) and resolves them
into the environment a per-job agent subprocess needs. Wraps
public.user_agent_credentials (migration 0013; named configs per
(user, agent_cli, name) since migration 0000) via the service-role REST
endpoint — the row's `api_key_enc` is ciphertext produced by
backend.service.crypto; plaintext never leaves this module except when
building the subprocess env for a run the user themselves triggered.

Service-layer module: the OSS core never imports this. The spec-worker
run is handed a plain `env` dict (already resolved here by the service
route), so no core code touches credentials.
"""
from __future__ import annotations

import logging

from backend import supabase_env, supabase_rest
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import httpx

from backend.agents.base import AGENT_ENV_VARS, host_has_credentials

from . import crypto


logger = logging.getLogger(__name__)


class CredentialDecryptError(Exception):
    """A stored credential exists but could not be decrypted.

    Distinct from "no credential stored": the ciphertext is present but the
    server can't recover it — typically a rotated or missing
    `HA_CREDS_ENC_KEY`. Surfaced (not swallowed) so operators see a systemic
    key problem instead of every user's key silently looking absent.
    """


class CredentialRefusal(Exception):
    """No credential of the user's can be used for this run, and the host's
    must not be used instead: the host's credentials stand in only when the
    user is confirmed to have none stored. `refusal(verb)` is the answer a
    route gives instead of starting the run (admins included)."""
    status = ""
    #: the refusal's message; `{verb}` names what the user was doing
    template = ""

    def refusal(self, verb: str) -> dict:
        return {"success": False, "status": self.status,
                "message": self.template.format(verb=verb)}


class CredentialChoiceRequired(CredentialRefusal):
    """The user has stored credentials but none is the one to use: several,
    none the default, and no task routing or exact config names one. The
    user picks a default."""
    status = "credential_choice_required"
    template = ("You have several saved agent credentials and none is the "
                "default. Mark one as default (or route this task to one) in "
                "account settings to {verb}.")


class CredentialsUnavailable(CredentialRefusal):
    """The user's stored credentials could not be read (Supabase errored):
    not knowing is not "none stored"."""
    status = "credentials_unavailable"
    template = ("Could not read your saved agent credentials. Try again in a "
                "moment to {verb}.")


class CredentialModelRequired(CredentialRefusal):
    """The credential points its agent at a custom endpoint but names no
    model for this run. The CLI's built-in default model belongs to the
    vendor's own endpoint (Codex's is an OpenAI model), so another endpoint
    rejects it — refuse the run instead of sending a model it cannot serve."""
    status = "credential_model_required"
    template = ("Your saved agent credential sets a custom endpoint but no "
                "model. Pick a model for it in account settings to {verb}.")


def require_model_for_endpoint(base_url: Optional[str],
                               model: Optional[str]) -> None:
    """A credential with a custom endpoint must name its model: blank means
    the agent CLI's own default, which only the vendor's endpoint serves."""
    if (base_url or "").strip() and not (model or "").strip():
        raise ValueError("A custom endpoint needs a model: pick the model "
                         "this endpoint serves.")


VALID_AGENTS = tuple(AGENT_ENV_VARS.keys())


class AgentTask(str, Enum):
    """The agent runs that spend a user's credential, valued as the keys of
    the credential's `task_models` (migration 0036). Each run may pin its own
    model; a task with no entry runs on the site-wide task default
    (`agent_task_defaults`, migration 0039) when one is set, else the
    credential's `model` (or the endpoint default when that is unset too)."""

    GENERATE = "generate"
    REFINE = "refine"
    REVIEW = "review"
    FIX = "fix"

    @property
    def label(self) -> str:
        """How the account page names the run."""
        return "Agent Fix" if self is AgentTask.FIX else self.value.capitalize()

    @classmethod
    def known(cls) -> list[str]:
        """Every task key, in the order the account page lists them."""
        return [t.value for t in cls]


_TASK_MODELS = "task_models"
_NAME = "name"
_OPTIONAL_COLS = (_NAME, _TASK_MODELS)


def _lacks_column(r: httpx.Response, col: str) -> bool:
    """PostgREST's answer to a request naming a column that is not deployed
    yet (42703 on select, PGRST204 on write)."""
    return r.status_code == 400 and col in (r.text or "")


def _order_without_missing(order: str, allowed: set[str]) -> str:
    """Keep only order columns that this try's select still names.

    PostgREST 42703s an `order=` on a column that is not deployed, the same
    as a missing select column. Listing asks for `agent_cli,name`; when
    `name` is stripped from select the order must drop it too, or the retry
    400s and the UI shows an empty key list while the rows are still there."""
    return ",".join(
        part.strip() for part in (order or "").split(",")
        if part.strip().split(".")[0] in allowed
    )


def _get_rows(client, url: str, key: str, params: dict,
              select: str) -> tuple[httpx.Response, set]:
    """GET `select` plus the optional columns (`name`, `task_models`),
    retrying without whichever the database does not have yet. Returns
    (response, the optional columns that exist)."""
    cols = list(_OPTIONAL_COLS)
    while True:
        q = {**params, "select": ",".join([select, *cols])}
        if "order" in q:
            order = _order_without_missing(
                q["order"], set(select.split(",")) | set(cols))
            if order:
                q["order"] = order
            else:
                q.pop("order", None)
        r = client.get(_table_url(url),
                       params=q,
                       headers=supabase_rest.service_headers(key))
        missing = next((c for c in cols if _lacks_column(r, c)), None)
        if missing is None:
            return r, set(cols)
        cols.remove(missing)


def _write(send, body: dict, task_models: Optional[dict],
           name: Optional[str] = None) -> httpx.Response:
    """Send a row write, carrying `task_models` when the caller set it (None
    leaves the stored map untouched) and `name` likewise. A write that only
    CLEARS an undeployed optional column goes through without it; one that
    sets a value raises (TaskModelsUnavailable / CredentialNamesUnavailable)
    rather than silently dropping the user's choice."""
    if task_models is not None:
        body = {**body, _TASK_MODELS: task_models or None}
    if name is not None:
        body = {**body, _NAME: name}
    r = send(body)
    for col, value, exc, migration in (
        (_NAME, name, CredentialNamesUnavailable, "0000"),
        (_TASK_MODELS, task_models, TaskModelsUnavailable, "0036"),
    ):
        if r.status_code >= 400 and col in body and _lacks_column(r, col):
            if value:
                raise exc(f"column {col!r} is not enabled on this server yet "
                          f"(migration {migration})")
            body = {k: v for k, v in body.items() if k != col}
            r = send(body)
    return r


class TaskModelsUnavailable(Exception):
    """Per-task models were asked to be stored, but the `task_models` column
    (migration 0036) is not deployed on this database."""


class CredentialNamesUnavailable(Exception):
    """A named credential was asked to be stored, but the `name` column
    (migration 0000) is not deployed on this database."""


def normalize_name(value: Optional[str]) -> str:
    """A credential's name as stored: trimmed, '' = the unnamed config. ':'
    is rejected because routing targets are stored as 'agent_cli:name'."""
    name = (value or "").strip()
    if ":" in name:
        raise ValueError("credential name must not contain ':'")
    return name


def normalize_task_models(value: Optional[dict]) -> dict:
    """The per-task model map as stored: known task keys only, blank models
    dropped (blank = "use the credential's default model"). Raises ValueError
    naming any key outside AgentTask."""
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError("task_models must be an object of task -> model")
    unknown = sorted(k for k in value if k not in AgentTask.known())
    if unknown:
        raise ValueError(
            f"Unknown task(s) {unknown}. One of {AgentTask.known()}.")
    out = {}
    for task, model in value.items():
        if model is not None and not isinstance(model, str):
            raise ValueError(f"task_models[{task!r}] must be a model id string")
        model = (model or "").strip()
        if model:
            out[task] = model
    return out


def _stored_task_models(row: dict) -> dict:
    """A row's per-task models as read back: tolerant of NULL, a column that
    is not deployed, or keys no longer in AgentTask (all mean "no override")."""
    raw = row.get(_TASK_MODELS)
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items()
            if k in AgentTask.known() and isinstance(v, str) and v.strip()}


def _stored_name(row: dict) -> str:
    """A row's credential name as read back: tolerant of NULL or a column
    that is not deployed (both mean the unnamed config)."""
    return (row.get(_NAME) or "").strip()


def _supabase() -> tuple[Optional[str], Optional[str]]:
    return (
        supabase_env.url(),
        supabase_env.service_role_key(),
    )


def enabled() -> bool:
    """BYOK is usable only when both Supabase (storage) and an encryption
    key (secret at rest) are configured."""
    url, key = _supabase()
    return bool(url and key and crypto.enabled())


def _table_url(url: str) -> str:
    return f"{url}/rest/v1/user_agent_credentials"


# ---------------------------------------------------------------------------
# Site-wide per-task defaults (migration 0039) — admin-managed policy answering
# "which model does each run use when the user hasn't pinned one". Read per
# resolve_env / listing call: job launches are rare, the table is tiny, and a
# cache would make an admin's edit not take effect immediately.
# ---------------------------------------------------------------------------


def _defaults_url(url: str) -> str:
    return f"{url}/rest/v1/agent_task_defaults"


def fetch_task_defaults() -> dict:
    """`{agent_cli: {task: model}}` from public.agent_task_defaults. Empty on
    an unconfigured/unreachable store or before migration 0039 is deployed —
    a missing table never breaks a run, it just means "no site policy"."""
    url, key = _supabase()
    if not (url and key):
        return {}
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(_defaults_url(url), params={"select": "agent_cli,task,model"},
                           headers=supabase_rest.service_headers(key))
    except httpx.RequestError as e:
        logger.warning(f"agent_task_defaults read failed: {e.__class__.__name__}")
        return {}
    if r.status_code >= 400:
        if r.status_code != 404:
            logger.warning(f"agent_task_defaults read: HTTP {r.status_code}")
        return {}
    out: dict = {}
    for row in r.json() or []:
        cli, task, model = row.get("agent_cli"), row.get("task"), row.get("model")
        if cli in AGENT_ENV_VARS and task in AgentTask.known() and (model or "").strip():
            out.setdefault(cli, {})[task] = model.strip()
    return out


def put_task_default(*, agent_cli: str, task: str, model: Optional[str]) -> None:
    """Set (or, with a blank model, clear) the site default for one
    (agent_cli, task). Raises ValueError on unknown agent/task, RuntimeError
    on misconfiguration or a hard Supabase failure (an admin action — surface
    it, don't swallow)."""
    if agent_cli not in AGENT_ENV_VARS:
        raise ValueError(f"Unknown agent_cli: {agent_cli!r}")
    if task not in AgentTask.known():
        raise ValueError(f"Unknown agent task: {task!r}")
    url, key = _supabase()
    if not (url and key):
        raise RuntimeError("Supabase service-role not configured")
    now = datetime.now(timezone.utc).isoformat()
    with httpx.Client(timeout=10.0) as client:
        if not (model or "").strip():
            r = client.delete(
                _defaults_url(url),
                params={"agent_cli": f"eq.{agent_cli}", "task": f"eq.{task}"},
                headers=supabase_rest.service_headers(key, prefer="return=minimal"),
            )
        else:
            r = client.post(
                _defaults_url(url),
                params={"on_conflict": "agent_cli,task"},
                headers=supabase_rest.service_headers(
                    key, prefer="resolution=merge-duplicates,return=minimal"),
                json={"agent_cli": agent_cli, "task": task,
                      "model": model.strip(), "updated_at": now},
            )
        if r.status_code >= 400:
            raise RuntimeError(f"task default write failed: {r.status_code} {r.text[:200]}")


def _env_for(agent_cli: str, api_key: str, base_url: Optional[str],
             model: Optional[str] = None) -> dict:
    names = AGENT_ENV_VARS[agent_cli]
    key_var, url_var, model_var = names.key, names.base_url, names.model
    env = {key_var: api_key}
    if base_url:
        env[url_var] = base_url
    if model:
        env[model_var] = model
    return env


# ---------------------------------------------------------------------------
# Per-user task → agent routing (migration 0040): which of the user's
# credentials runs each task when the caller names no agent. Stored as a
# jsonb map on the user's profiles row.
# ---------------------------------------------------------------------------

_ROUTING = "agent_task_routing"


def _profiles_url(url: str) -> str:
    return f"{url}/rest/v1/profiles"


def _lacks_routing(r: httpx.Response) -> bool:
    return r.status_code == 400 and _ROUTING in (r.text or "")


def fetch_task_routing(user_id: str) -> dict:
    """`{task: agent_cli}` for one user; {} when unset, unconfigured, or
    migration 0040 is not deployed. Unknown tasks/agents are dropped."""
    url, key = _supabase()
    if not (url and key):
        return {}
    with httpx.Client(timeout=5.0) as client:
        r = client.get(_profiles_url(url),
                       params={"id": f"eq.{user_id}", "select": _ROUTING},
                       headers=supabase_rest.service_headers(key))
    if r.status_code >= 400:
        if not _lacks_routing(r):
            logger.warning(f"task routing read: HTTP {r.status_code}")
        return {}
    rows = r.json() or []
    raw = rows[0].get(_ROUTING) if rows else None
    if not isinstance(raw, dict):
        return {}
    return {t: a for t, a in raw.items()
            if t in AgentTask.known() and parse_target(a) is not None}


def parse_target(value) -> Optional[tuple[str, str]]:
    """A routing target as stored: 'agent_cli' (any config of it) or
    'agent_cli:name' (one named config). None when it names nothing valid."""
    if not isinstance(value, str):
        return None
    cli, sep, name = value.partition(":")
    if cli not in AGENT_ENV_VARS or (sep and (not name or ":" in name)):
        return None
    return cli, name


def put_task_routing(user_id: str, task: str, target: Optional[str]) -> None:
    """Route one task to one of the user's credentials — 'agent_cli' or
    'agent_cli:name' (blank clears the entry). Raises ValueError on an unknown
    task/agent or when the user has no matching credential, RuntimeError on a
    hard store failure."""
    if task not in AgentTask.known():
        raise ValueError(f"Unknown agent task: {task!r}")
    parsed = parse_target((target or "").strip()) if target else None
    if target and parsed is None:
        raise ValueError(f"Unknown routing target: {target!r}")
    url, key = _supabase()
    if not (url and key):
        raise RuntimeError("Supabase service-role not configured")
    routing = fetch_task_routing(user_id)
    if parsed:
        cli, name = parsed
        # Routing to a credential the user never configured would silently
        # fall back at resolve time — refuse it instead.
        if not any(c.get("agent_cli") == cli and _stored_name(c) == name
                   for c in list_for_user(user_id)):
            label = f"{cli}:{name}" if name else cli
            raise ValueError(f"No {label!r} credential stored — add the key first")
        routing[task] = f"{cli}:{name}" if name else cli
    else:
        routing.pop(task, None)
    with httpx.Client(timeout=10.0) as client:
        r = client.patch(
            _profiles_url(url),
            params={"id": f"eq.{user_id}"},
            json={_ROUTING: routing or None},
            headers=supabase_rest.service_headers(key, prefer="return=minimal"),
        )
    if r.status_code >= 400:
        if _lacks_routing(r) and parsed:
            raise TaskModelsUnavailable(
                "task routing is not enabled on this server yet (migration 0040)")
        raise RuntimeError(f"task routing write failed: {r.status_code} {r.text[:200]}")


def upsert(
    *,
    user_id: str,
    agent_cli: str,
    api_key: str,
    base_url: Optional[str],
    is_default: bool,
    model: Optional[str] = None,
    task_models: Optional[dict] = None,
    name: str = "",
) -> None:
    """Encrypt + store one credential (one row per user / agent / name).
    Raises on misconfiguration or a hard Supabase failure so the route can
    surface it (this is a user action, not a best-effort background write)."""
    if agent_cli not in AGENT_ENV_VARS:
        raise ValueError(f"Unknown agent_cli: {agent_cli!r}")
    name = normalize_name(name)
    require_model_for_endpoint(base_url, model)
    if task_models is not None:
        task_models = normalize_task_models(task_models)
    url, key = _supabase()
    if not (url and key):
        raise RuntimeError("Supabase service-role not configured")

    api_key_enc = crypto.encrypt(api_key)  # raises CredentialsKeyMissing if unset
    now = datetime.now(timezone.utc).isoformat()

    with httpx.Client(timeout=10.0) as client:
        # Only one default per user: clear the others first (the partial
        # unique index is the hard backstop).
        if is_default:
            r = client.patch(
                _table_url(url),
                params={"user_id": f"eq.{user_id}", "is_default": "eq.true"},
                headers=supabase_rest.service_headers(key, prefer="return=minimal"),
                json={"is_default": False, "updated_at": now},
            )
            if r.status_code >= 400:
                raise RuntimeError(f"clear-default failed: {r.status_code} {r.text[:200]}")

        # Upsert (unique on user_id, agent_cli, name). A column left out of
        # the body keeps its stored value on conflict.
        r = _write(lambda body: client.post(
            _table_url(url),
            params={"on_conflict": "user_id,agent_cli,name"},
            headers=supabase_rest.service_headers(key, prefer="resolution=merge-duplicates,return=minimal"),
            json=body,
        ), {
            "user_id": user_id,
            "agent_cli": agent_cli,
            "api_key_enc": api_key_enc,
            "base_url": base_url or None,
            "model": model or None,
            "is_default": is_default,
            "updated_at": now,
        }, task_models, name=name)
        if r.status_code >= 400:
            raise RuntimeError(f"credential upsert failed: {r.status_code} {r.text[:200]}")


def update_fields(
    *,
    user_id: str,
    agent_cli: str,
    base_url: Optional[str],
    model: Optional[str],
    is_default: bool,
    task_models: Optional[dict] = None,
    name: str = "",
) -> bool:
    """Edit an EXISTING credential's endpoint / model / default flag (and, when
    `task_models` is given, its per-task models; None leaves them) WITHOUT
    touching the stored key (so a user can retarget a proxy or pick a model
    without re-pasting their secret). PATCH-only: returns False when no row
    exists for (user_id, agent_cli, name), which the route turns into "a key
    is required to create a new credential"."""
    if agent_cli not in AGENT_ENV_VARS:
        raise ValueError(f"Unknown agent_cli: {agent_cli!r}")
    name = normalize_name(name)
    require_model_for_endpoint(base_url, model)
    if task_models is not None:
        task_models = normalize_task_models(task_models)
    url, key = _supabase()
    if not (url and key):
        raise RuntimeError("Supabase service-role not configured")
    now = datetime.now(timezone.utc).isoformat()
    with httpx.Client(timeout=10.0) as client:
        if is_default:
            _clear_other_defaults(client, url, key, user_id, agent_cli, name, now)
        r = _write(lambda body: client.patch(
            _table_url(url),
            params={"user_id": f"eq.{user_id}", "agent_cli": f"eq.{agent_cli}",
                    "name": f"eq.{name}"},
            headers=supabase_rest.service_headers(key, prefer="return=representation"),
            json=body,
        ), {
            "base_url": base_url or None,
            "model": model or None,
            "is_default": is_default,
            "updated_at": now,
        }, task_models)
        if r.status_code >= 400:
            raise RuntimeError(f"credential update failed: {r.status_code} {r.text[:200]}")
        return bool(r.json() or [])


def set_default(*, user_id: str, agent_cli: str, name: str = "") -> bool:
    """Make one existing credential the user's default (clearing any other).
    Key-free. Returns False when the row doesn't exist."""
    if agent_cli not in AGENT_ENV_VARS:
        raise ValueError(f"Unknown agent_cli: {agent_cli!r}")
    name = normalize_name(name)
    url, key = _supabase()
    if not (url and key):
        raise RuntimeError("Supabase service-role not configured")
    now = datetime.now(timezone.utc).isoformat()
    with httpx.Client(timeout=10.0) as client:
        _clear_other_defaults(client, url, key, user_id, agent_cli, name, now)
        r = client.patch(
            _table_url(url),
            params={"user_id": f"eq.{user_id}", "agent_cli": f"eq.{agent_cli}",
                    "name": f"eq.{name}"},
            headers=supabase_rest.service_headers(key, prefer="return=representation"),
            json={"is_default": True, "updated_at": now},
        )
        if r.status_code >= 400:
            raise RuntimeError(f"set-default failed: {r.status_code} {r.text[:200]}")
        return bool(r.json() or [])


def _clear_other_defaults(client, url, key, user_id: str, agent_cli: str,
                          name: str, now: str) -> None:
    """Clear is_default on the user's OTHER rows (anything but the exact
    target), so setting a new default can't self-clobber regardless of
    statement order."""
    r = client.patch(
        _table_url(url),
        params={
            "user_id": f"eq.{user_id}",
            "is_default": "eq.true",
            "or": f"(agent_cli.neq.{agent_cli},name.neq.{name})",
        },
        headers=supabase_rest.service_headers(key, prefer="return=minimal"),
        json={"is_default": False, "updated_at": now},
    )
    if r.status_code >= 400:
        raise RuntimeError(f"clear-default failed: {r.status_code} {r.text[:200]}")


def listing(user_id: str) -> dict:
    """Sanitized view for the account UI — NEVER returns the key or its
    ciphertext, only which credentials are configured and how.
    `task_models_enabled` / `names_enabled` say whether per-task models
    (migration 0036) / named credentials (migration 0000) can be stored here;
    each item's `task_models` / `name` degrade to {} / '' when undeployed."""
    url, key = _supabase()
    if not (url and key):
        return {"items": [], "task_models_enabled": False, "names_enabled": False}
    with httpx.Client(timeout=10.0) as client:
        r, cols = _get_rows(
            client, url, key,
            {"user_id": f"eq.{user_id}", "order": "agent_cli,name"},
            "agent_cli,base_url,model,is_default,updated_at")
    if r.status_code >= 400:
        logger.warning(f"list creds failed: {r.status_code} {r.text[:200]}")
        return {"items": [], "task_models_enabled": False, "names_enabled": False}
    return {"items": [{
        "agent_cli": row.get("agent_cli"),
        "name": _stored_name(row),
        "base_url": row.get("base_url"),
        "model": row.get("model"),
        "task_models": _stored_task_models(row),
        "is_default": bool(row.get("is_default")),
        "has_key": True,
        "updated_at": row.get("updated_at"),
    } for row in (r.json() or [])],
        "task_models_enabled": _TASK_MODELS in cols,
        "names_enabled": _NAME in cols}


def list_for_user(user_id: str) -> list[dict]:
    """The sanitized credential items of `listing`."""
    return listing(user_id)["items"]


def delete(user_id: str, agent_cli: str, name: str = "") -> bool:
    name = normalize_name(name)
    url, key = _supabase()
    if not (url and key):
        return False
    with httpx.Client(timeout=10.0) as client:
        r = client.delete(
            _table_url(url),
            params={"user_id": f"eq.{user_id}", "agent_cli": f"eq.{agent_cli}",
                    "name": f"eq.{name}"},
            headers=supabase_rest.service_headers(key, prefer="return=minimal"),
        )
    return r.status_code < 400


def run_model(row: dict, agent_cli: str, task: Optional[str],
              defaults: Optional[dict] = None) -> Optional[str]:
    """The model a run of `task` uses on this stored credential: its pin for
    the task, else the site-wide task default (`defaults`, read when not
    given), else the credential's own `model`; None means the endpoint
    default."""
    model = _stored_task_models(row).get(task) if task else None
    if model is None and task:
        if defaults is None:
            defaults = fetch_task_defaults()
        model = defaults.get(agent_cli, {}).get(task)
    return model or row.get("model") or None


def _pick(rows: list[dict], agent_cli: Optional[str] = None,
          name: Optional[str] = None) -> Optional[dict]:
    """The row to use among `rows`: filtered to the agent (and the exact
    config when `name` is given); among several candidates the default row
    wins, else the unnamed config."""
    cand = [r for r in rows
            if (agent_cli is None or r.get("agent_cli") == agent_cli)
            and (name is None or _stored_name(r) == name)]
    if len(cand) <= 1:
        return cand[0] if cand else None
    row = next((r for r in cand if r.get("is_default")), None)
    if row is None:
        row = next((r for r in cand if not _stored_name(r)), None)
    return row


def resolve_env(user_id: str, agent_cli: Optional[str] = None, *,
                name: Optional[str] = None,
                task: Optional[str] = None) -> Optional[dict]:
    """Resolve the credential to inject into a run this user triggered.

    Returns `{"agent_cli": <cli>, "env": {KEY_VAR: ..., BASE_VAR?: ...,
    MODEL_VAR?: ...}}` for the specified `agent_cli` (the exact config when
    `name` is given, else its default/unnamed one), else the credential the
    user routes this task to (migration 0040, a 'agent_cli[:name]' target),
    else the user's default credential (or sole one); None if the user has no
    stored credential — in which case an admin's run falls back to the host's
    own agent credentials. A user with stored credentials of which none can be
    chosen (several, no default, nothing routed) gets
    `CredentialChoiceRequired`, never the host's; a read that fails raises
    `CredentialsUnavailable` (an outage is not "none stored").

    `task` (an AgentTask value) names the run: its model is the one the
    credential pins for that task, else the site-wide task default
    (`agent_task_defaults`, migration 0039), else the credential's own
    `model`, else the endpoint default.

    Raises `CredentialDecryptError` when a credential IS stored but can't be
    decrypted — distinct from None ("none stored") so the caller can tell the
    user the truth (re-enter) rather than "add a key", while an admin can
    still choose to fall back to host credentials.
    """
    if task is not None and task not in AgentTask.known():
        raise ValueError(f"Unknown agent task: {task!r}")
    url, key = _supabase()
    if not (url and key):
        return None
    params = {"user_id": f"eq.{user_id}"}
    if agent_cli:
        params["agent_cli"] = f"eq.{agent_cli}"
    try:
        with httpx.Client(timeout=10.0) as client:
            r, _ = _get_rows(client, url, key, params,
                             "agent_cli,api_key_enc,base_url,model,is_default")
    except httpx.RequestError as e:
        logger.warning(f"resolve creds errored: {e}")
        raise CredentialsUnavailable(str(e)) from e
    if r.status_code >= 400:
        logger.warning(f"resolve creds failed: {r.status_code} {r.text[:200]}")
        raise CredentialsUnavailable(f"credentials read {r.status_code}")
    rows = r.json() or []
    if not rows:
        return None
    # Prefer the explicit agent/config match, else the task's routed
    # credential (migration 0040), else the default, else the sole row. The
    # routing read is skipped when one credential is all there is — it could
    # only name that same credential or a deleted one (which falls back
    # anyway).
    row = None
    if agent_cli:
        row = _pick(rows, agent_cli, normalize_name(name) if name else None)
    else:
        if task and len(rows) > 1:
            routed = parse_target(fetch_task_routing(user_id).get(task))
            if routed:
                cli_t, name_t = routed
                row = _pick(rows, cli_t, name_t or None)
        if row is None:
            row = next((x for x in rows if x.get("is_default")), None)
        if row is None and len(rows) == 1:
            row = rows[0]
    if row is None and name:
        return None  # the named config is not stored
    if row is None:
        raise CredentialChoiceRequired(
            f"{len(rows)} stored credentials and none chosen for this run")
    cli = row["agent_cli"]
    try:
        api_key = crypto.decrypt(row["api_key_enc"])
    except Exception as e:
        # A stored credential exists but can't be decrypted (rotated/missing
        # HA_CREDS_ENC_KEY). Don't return None — that's indistinguishable from
        # "no credential", so a non-admin would be told to "add your key" (it
        # is added, just unreadable) and a systemic rotation would be silent.
        # Surface it: loud for operators, distinct for the caller.
        logger.error(
            "credential decrypt failed for user %s / %s — check HA_CREDS_ENC_KEY "
            "(rotation?): %s", user_id, cli, e,
        )
        raise CredentialDecryptError("saved credential could not be decrypted") from e
    model = run_model(row, cli, task)
    if row.get("base_url") and not model:
        # Rows saved before the write-time rule (require_model_for_endpoint).
        raise CredentialModelRequired(
            f"{cli} credential has endpoint {row['base_url']!r} and no model")
    return {
        "agent_cli": cli,
        "env": _env_for(cli, api_key, row.get("base_url"), model),
    }


def fallback_envs(user_id: str, run_cli: str, *, task: str,
                  host_allowed: bool) -> dict[str, Optional[dict]]:
    """The credentials a run may fail over to when its agent's service is
    unavailable: for every other agent CLI, the env `resolve_env` gives this
    user for it, else — only when `host_allowed` (an admin, as in the run's
    own preflight) — None, meaning the host's own credentials for that CLI
    when the host has them. A CLI with neither is left out; a stored
    credential that cannot be chosen or read is left out too (never the
    host's instead). The order is the worker's (`limits.failover_order`)."""
    fallbacks: dict[str, Optional[dict]] = {}
    for cli in AGENT_ENV_VARS:
        if cli == run_cli:
            continue
        try:
            resolved = resolve_env(user_id, cli, task=task)
        except (CredentialRefusal, CredentialDecryptError) as exc:
            logger.info("failover: %s credential for %s not usable: %s",
                        cli, user_id, exc)
            continue
        if resolved is not None:
            fallbacks[cli] = resolved["env"]
        elif host_allowed and host_has_credentials(cli):
            fallbacks[cli] = None
    return fallbacks


# ---------------------------------------------------------------------------
# List models — proxy the provider's /v1/models so the account page offers the
# endpoint's REAL model ids instead of a hand-maintained guess list.
# ---------------------------------------------------------------------------

_MODELS_DEFAULT_BASE = {
    "claude": "https://api.anthropic.com",
    "codex": "https://api.openai.com",
    "kimi": "https://api.moonshot.cn",
}


def fetch_agent_models(agent_cli: str, api_key: str,
                       base_url: Optional[str] = None) -> list[str]:
    """Query the provider's `/v1/models` with `api_key`; return sorted model ids.

    Absorbs the base-url `/v1` inconsistency (OpenAI / Moonshot bases usually
    include `/v1`, Anthropic's does not). Raises ValueError with a user-safe
    message on any failure (bad key, unreachable endpoint, unparseable body).
    """
    if agent_cli not in AGENT_ENV_VARS:
        raise ValueError(f"unknown agent {agent_cli!r}")
    if not (api_key or "").strip():
        raise ValueError("no API key to query models with")
    base = (base_url or _MODELS_DEFAULT_BASE[agent_cli]).rstrip("/")
    models_url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    if agent_cli == "claude":
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    else:
        headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(models_url, headers=headers)
    except httpx.RequestError as e:
        raise ValueError(f"could not reach the endpoint ({e.__class__.__name__})") from e
    if r.status_code in (401, 403):
        raise ValueError("the endpoint rejected this key (auth error)")
    if r.status_code >= 400:
        raise ValueError(f"the endpoint returned HTTP {r.status_code}")
    try:
        body = r.json()
    except Exception as e:
        raise ValueError("the endpoint did not return JSON") from e
    if not isinstance(body, dict):
        raise ValueError("the endpoint JSON was not an object")
    # Some gateways (Zhipu `/api/v1/models`) answer HTTP 200 with
    # `{success: false, msg: …}` and no `data`. That is an error, not an
    # empty catalog.
    if body.get("success") is False or body.get("error"):
        err = body.get("error")
        msg = (body.get("msg") or body.get("message")
               or (err.get("message") if isinstance(err, dict) else None)
               or "endpoint error")
        raise ValueError(str(msg))
    rows = body.get("data") or body.get("models") or []
    if isinstance(rows, dict):
        rows = rows.get("data") or rows.get("list") or []
    ids = sorted({
        (m if isinstance(m, str) else (m.get("id") or m.get("name")))
        for m in rows
        if (isinstance(m, str) and m)
        or (isinstance(m, dict) and (m.get("id") or m.get("name")))
    })
    return ids


def probe_agent_generation(agent_cli: str, api_key: str,
                           base_url: Optional[str], model: Optional[str]) -> dict:
    """A minimal 1-token completion — the model list proves the endpoint
    answers and the key lists models, but says nothing about GENERATING:
    the model id may be an alias the endpoint doesn't chat with, the key may
    lack generation rights, the protocol may mismatch. Returns
    `{ok, model, detail, latency_ms}`; never raises."""
    import time as _time
    if not (model or "").strip():
        return {"ok": False, "model": None,
                "detail": "no model configured on the credential — nothing to probe"}
    base = (base_url or _MODELS_DEFAULT_BASE[agent_cli]).rstrip("/")
    t0 = _time.monotonic()
    if agent_cli == "claude":
        url = f"{base}/v1/messages"
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        body = {"model": model, "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}]}
    else:
        url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        body = {"model": model, "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}]}
    try:
        with httpx.Client(timeout=20.0) as client:
            r = client.post(url, headers=headers, json=body)
    except httpx.RequestError as e:
        return {"ok": False, "model": model,
                "detail": f"could not reach the endpoint ({e.__class__.__name__})",
                "latency_ms": int((_time.monotonic() - t0) * 1000)}
    latency_ms = int((_time.monotonic() - t0) * 1000)
    if r.status_code in (401, 403):
        return {"ok": False, "model": model,
                "detail": "the endpoint rejected this key for generation (auth error)",
                "latency_ms": latency_ms}
    if r.status_code == 404:
        return {"ok": False, "model": model,
                "detail": f"the endpoint has no model {model!r}",
                "latency_ms": latency_ms}
    if r.status_code >= 400:
        return {"ok": False, "model": model,
                "detail": f"the endpoint returned HTTP {r.status_code}",
                "latency_ms": latency_ms}
    return {"ok": True, "model": model, "latency_ms": latency_ms}


def saved_key_and_base(user_id: str, agent_cli: str, name: str = ""):
    """Decrypt the user's stored (api_key, base_url) for one credential.

    Returns None when there is no stored credential. Raises
    `CredentialDecryptError` when a credential exists but can't be decrypted —
    the caller must surface that distinctly, not treat it as "no credential".

    Lets `list models` work while EDITING an existing credential, where the key
    field is intentionally blank (we never echo a stored key back to the client).
    """
    url, key = _supabase()
    if not (url and key):
        return None
    params = {
        "user_id": f"eq.{user_id}",
        "agent_cli": f"eq.{agent_cli}",
        "name": f"eq.{normalize_name(name)}",
        "select": "api_key_enc,base_url",
    }
    with httpx.Client(timeout=10.0) as client:
        r = client.get(_table_url(url), params=params, headers=supabase_rest.service_headers(key))
    if r.status_code >= 400:
        return None
    rows = r.json() or []
    if not rows:
        return None
    try:
        return crypto.decrypt(rows[0]["api_key_enc"]), rows[0].get("base_url")
    except Exception as e:
        # The credential EXISTS (rows is non-empty) but can't be decrypted —
        # NOT the same as "no credential". A systemic cause (rotated/missing
        # HA_CREDS_ENC_KEY) would otherwise make every user's saved key look
        # absent. Surface it: loud for operators, distinct for the caller.
        logger.error(
            "credential decrypt failed for user %s / %s — check HA_CREDS_ENC_KEY "
            "(rotation?): %s", user_id, agent_cli, e,
        )
        raise CredentialDecryptError("saved credential could not be decrypted") from e
