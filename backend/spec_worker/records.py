"""Where a run's generation record is kept.

A generation record is ONE per (owner, model): the spec bundle a run
produced, the snapshot it was judged against, where it stands
(`GenerationStatus`), and the id that review, fix and PR are pointed at.
Every later step of the pipeline is keyed on one, so a run that leaves none
leaves those steps nothing to act on.

Two deployments keep that same record in two places:

    Supabase    `backend.service.spec_generations` — the hosted,
                multi-user tier: the conditional write RPCs, one row per
                (user, model) plus at most one pending row beside it.
    host-local  one JSON file per (owner, model) under
                `~/.huggingarch/spec_records/<owner>/<model>/record.json`,
                for a deployment with no database — a self-host, or a
                local offline run.

This module is the only place that chooses, so every caller asks one
question ("the record of this (owner, model)") and the deployment decides
who answers. The two are backends of one record, not two notions of one:
the field names, the status vocabulary (`backend.spec_worker.outcome`) and
the outcome words (`RecordOutcome`) are the ones the RPCs already use, and
the local write keeps the same rule `record_generation` keeps — the working
candidate is overwritten in place.

Why the chooser lives here and not inside `spec_generations`: `backend.service`
is an OPTIONAL package (a no-service build drops it, see
tests/meta/test_service_layout.py) and core may import it only lazily, under
a `supabase_env.configured()` guard. A chooser inside `spec_generations`
would be unreachable in exactly the deployment that needs the local backend.
This is the arrangement `shared_spec_miss` already uses for the shared spec
cache, whose service-side module says so in as many words.

The local owner is the reserved `__anonymous__` overlay owner
(`overlay.run_overlay_owner`) — the name this repo already gives "the single
user of a deployment with no auth". There is no second identity.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend import supabase_env
from backend.spec_worker.outcome import MUTABLE, GenerationStatus, RecordOutcome
from backend.spec_worker.overlay import _atomic_write_text, _safe, run_overlay_owner

logger = logging.getLogger(__name__)

#: One writer at a time over the local store. A read-then-overwrite decides
#: on the record already there, exactly as the RPC's advisory lock does.
_store_lock = threading.Lock()

_RECORD_FILE = "record.json"


def _hosted():
    """The Supabase backend (`backend.service.spec_generations`), or None when
    this deployment has no database and the local store answers. The one lazy,
    service-guarded import this module needs: `backend.service` is the optional
    half of the repo, so core may reach it only this way
    (tests/meta/test_service_layout.py)."""
    if not supabase_env.configured():
        return None
    from backend.service import spec_generations
    return spec_generations


def spec_records_root() -> Path:
    """Where the local backend keeps records — a sibling of the run overlays
    and the local shared-spec cache under `~/.huggingarch`, NOT a file inside
    a run overlay: overlay leaves are pure cache and get garbage-collected."""
    value = os.environ.get("HA_SPEC_RECORDS_ROOT")
    return Path(value) if value else Path.home() / ".huggingarch" / "spec_records"


def _leaf(owner: str, model_id: str) -> Path:
    root = spec_records_root().resolve()
    leaf = (root / _safe(owner) / _safe(model_id)).resolve()
    if not leaf.is_relative_to(root):
        raise ValueError(
            f"spec record path for {owner!r}/{model_id!r} escapes its root")
    return leaf


def local_record_id(owner: str, model_id: str) -> str:
    """The local record's id. A record IS its (owner, model), and the local
    store holds one per pair, so the id is derived from the pair: a re-run
    overwrites the record it already had rather than orphaning the reviews
    and fixes pointed at the old id. A uuid5 so the value is a UUID like the
    hosted tier's."""
    return str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"https://huggingarch.local/spec_records/{owner}/{model_id}"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_leaf(leaf: Path) -> Optional[dict]:
    path = leaf / _RECORD_FILE
    if not path.is_file():
        return None  # this (owner, model) has no record — not an error
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning(f"unreadable spec record {path}: {exc}")
        return None
    if not isinstance(row, dict) or not (row.get("spec_yaml") or "").strip():
        logger.warning(f"spec record {path} holds no spec")
        return None
    return row


def _write_leaf(leaf: Path, row: dict) -> None:
    leaf.mkdir(parents=True, exist_ok=True, mode=0o700)
    leaf.chmod(0o700)
    _atomic_write_text(leaf / _RECORD_FILE,
                       json.dumps(row, indent=2, ensure_ascii=False) + "\n")


def _next_revision(standing: Optional[dict]) -> int:
    """What the hosted tier's `spec_generations_bump_revision` trigger does:
    a fresh record starts at 0, every rewrite of one adds 1. The optimistic
    concurrency check a pending transition makes reads this."""
    if standing is None:
        return 0
    return int(standing.get("revision") or 0) + 1


def _iter_local() -> list[dict]:
    root = spec_records_root()
    if not root.is_dir():
        return []
    rows = []
    for path in sorted(root.glob(f"*/*/{_RECORD_FILE}")):
        row = _read_leaf(path.parent)
        if row is not None:
            rows.append(row)
    return rows


def record(
    *,
    model_id: str,
    user_id: str,
    spec_yaml: str,
    snapshot_json: Optional[dict] = None,
    new_component_yamls: Optional[list[dict]] = None,
    agent_cli: Optional[str] = None,
    agent_model: Optional[str] = None,
    run_started_at: Optional[str] = None,
) -> dict:
    """Record a successful run's spec as this (owner, model)'s working
    candidate, and return the record outcome (`done.record`).

    Hosted: `spec_generations.record` — unchanged, including its `none` for a
    run that carries no user. Local: the one record of (owner, model) is
    overwritten in place as `generated`, the same rule `record_generation`
    applies to the newest MUTABLE row.
    """
    hosted = _hosted()
    if hosted is not None:
        return hosted.record(
            model_id=model_id, user_id=user_id, spec_yaml=spec_yaml,
            snapshot_json=snapshot_json,
            new_component_yamls=new_component_yamls,
            agent_cli=agent_cli, agent_model=agent_model,
            run_started_at=run_started_at)
    owner = run_overlay_owner(user_id)
    try:
        with _store_lock:
            leaf = _leaf(owner, model_id)
            standing = _read_leaf(leaf)
            if standing is not None and standing.get("status") not in MUTABLE:
                # The RPC inserts the candidate BESIDE a frozen or pending
                # row; the local store has one slot per (owner, model), so
                # it keeps what stands and says it did not write.
                return {"outcome": RecordOutcome.FAILED.value,
                        "reason": (
                            f"the local record of {model_id} stands as "
                            f"{standing.get('status')}; it is not overwritten")}
            record_id = local_record_id(owner, model_id)
            _write_leaf(leaf, {
                "id": record_id,
                "user_id": owner,
                "model_id": model_id,
                "spec_yaml": spec_yaml,
                "snapshot_json": snapshot_json or {},
                "new_component_yamls": new_component_yamls or [],
                "agent_cli": agent_cli,
                "agent_model": agent_model,
                "status": GenerationStatus.GENERATED.value,
                "pr_url": None,
                "created_at": _now(),
                "run_started_at": run_started_at or _now(),
                "revision": _next_revision(standing),
                "engine_gaps": None,
                "repairable_errors": None,
                "blocked_engine_version": None,
            })
    except OSError as exc:  # storage must not break the run's verdict
        logger.warning(f"local spec record write failed for {model_id}: {exc}")
        return {"outcome": RecordOutcome.FAILED.value, "reason": str(exc)[:200]}
    return {"outcome": RecordOutcome.RECORDED.value, "id": record_id}


def get(record_id: str, *, strict: bool = False) -> Optional[dict]:
    """One record by id (full payload), or None on a miss. `strict` is the
    hosted tier's "tell an outage from a miss" (`SpecGenerationsUnavailable`);
    the local store reads a file, where a miss is a miss."""
    hosted = _hosted()
    if hosted is not None:
        return hosted.get(record_id, strict=strict)
    return next((row for row in _iter_local() if row.get("id") == record_id),
                None)


def serving_generation(user_id: str, model_id: str) -> Optional[dict]:
    """The (owner, model)'s record — what review or fix is launched against
    when the caller knows the model and not the id. Hosted: the ranked read
    (`spec_generations.serving_generation`, which raises on an outage rather
    than reporting "no record"). Local: the one record of the pair."""
    hosted = _hosted()
    if hosted is not None:
        return hosted.serving_generation(user_id, model_id)
    return _read_leaf(_leaf(run_overlay_owner(user_id), model_id))


def list_for_owner(user_id: str) -> list[dict]:
    """The (owner)'s records, newest first, without their bundles. Hosted:
    `spec_generations.list_for_user`; local: the owner's record files."""
    hosted = _hosted()
    if hosted is not None:
        return hosted.list_for_user(user_id)
    owner = run_overlay_owner(user_id)
    rows = [{k: v for k, v in row.items()
             if k not in ("spec_yaml", "new_component_yamls", "snapshot_json")}
            for row in _iter_local()
            if row.get("user_id") in (user_id, owner)]
    return sorted(rows, key=lambda r: r.get("created_at") or "", reverse=True)


def update_spec(
    record_id: str,
    *,
    spec_yaml: str,
    new_component_yamls: list,
    user_id: Optional[str] = None,
    snapshot_json: Optional[dict] = None,
) -> bool:
    """Replace a record's spec bundle IN PLACE — the agent-fix write-back.
    True only when a record was really rewritten. As in the hosted PATCH, a
    pending record is never matched (its fix lands by `promote_pending`) and
    `user_id`, when given, scopes the write to that owner."""
    hosted = _hosted()
    if hosted is not None:
        return hosted.update_spec(
            record_id, spec_yaml=spec_yaml,
            new_component_yamls=new_component_yamls, user_id=user_id,
            snapshot_json=snapshot_json)
    try:
        with _store_lock:
            standing = next(
                (row for row in _iter_local() if row.get("id") == record_id),
                None)
            if standing is None or standing.get("status") not in MUTABLE:
                return False
            if user_id and standing.get("user_id") != run_overlay_owner(user_id):
                return False
            row = {**standing,
                   "spec_yaml": spec_yaml,
                   "new_component_yamls": new_component_yamls or [],
                   "revision": _next_revision(standing)}
            if snapshot_json:
                row["snapshot_json"] = snapshot_json
            _write_leaf(_leaf(standing["user_id"], standing["model_id"]), row)
    except OSError as exc:
        logger.warning(f"local spec record update failed for {record_id}: {exc}")
        return False
    return True


def record_pending(**fields) -> dict:
    """Keep an engine-blocked run's spec as this (owner, model)'s pending
    record (`record_pending_generation`), beside — never over — its working
    candidate. The local store has one slot per (owner, model) and no second
    one to put a pending record in, so it keeps none and the outcome is
    `none`: the spec stays in the run's workspace, as it does today for any
    run that cannot be recorded."""
    hosted = _hosted()
    if hosted is not None:
        return hosted.record_pending(**fields)
    return {"outcome": RecordOutcome.NONE.value}


def promote_pending(record_id: str, **fields) -> dict:
    """Make a pending record the working candidate (`transition_pending_generation`).
    The local store writes no pending record — nothing there is ever pending —
    so there is nothing to promote and the outcome is `none`."""
    hosted = _hosted()
    if hosted is not None:
        return hosted.promote_pending(record_id, **fields)
    return {"outcome": RecordOutcome.NONE.value}


__all__ = [
    "get",
    "local_record_id",
    "promote_pending",
    "record",
    "record_pending",
    "serving_generation",
    "spec_records_root",
    "update_spec",
]
