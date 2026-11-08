"""Kernel Bench HTTP routes, served by the Kernel Bench service
(`backend.service.bench.server`, docs/measured_calibration.md §5.3):
measured-benchmark upload, the batch pages, review, the Kernel Bench page's
views, and the measurements estimates read.

The measurements live in a DuckLake (`backend/service/bench`, docs/measured_calibration.md §4): `store`
writes, `sets` and `browse` read, `lookup` answers estimates, `lake` holds
the process's DuckDB instance, opened from the environment
(BENCH_LAKE_CATALOG, BENCH_LAKE_DATA, …), behind `serving.Gate`: until the
lake is open, imported if asked, verified and holds the lease, every lake
endpoint answers 503 with the reason (a missing setting, the import
running, not the authoritative lake, …); writes also while the startup
backup is pending or the volume is short of space. A committed write pokes
the backup. The store knows no users: this module resolves who may see or
change what — from the identity the HA API forwards (`identity`) — and
passes explicit owners and set references.

  POST   /api/bench/upload                  validate a CSV, store it as one private batch
                                            (an admin may publish it straight to shared)
  GET    /api/bench/list                    the caller's runs and uploads, with their batches
  GET    /api/bench/summary                 one set's counts and facets
  GET    /api/bench/rows                    one page of a set (run_id or upload_id required)
  GET    /api/bench/calibration             the caller's rows: how many calibrate, how many not, why
  GET    /api/bench/overlap                 one set against the shared corpus per op shape
  POST   /api/bench/rows/delete             delete some of the caller's rows
  DELETE /api/bench/run/{run_id}            delete a run (all its batches)
  DELETE /api/bench/{upload_id}             delete an upload
  POST   /api/bench/run/{run_id}/contribute nominate a run's private batches
  POST   /api/bench/{upload_id}/contribute  nominate an upload's private batch
  GET    /api/bench/nominations             (admin) nominated batches
  POST   /api/bench/promote                 (admin) nominated → shared
  POST   /api/bench/reject                  (admin) nominated → rejected
  GET    /api/bench/providers               the measurement catalog (no auth)
  GET    /api/bench/browse                  one page of one Kernel Bench view
  POST   /internal/bench/lookup             the candidate rows of an estimate's keys
  GET    /internal/bench/version            the lake, its snapshot, the annotation rules

Estimates never read the lake: they ask `/internal/bench/lookup` for the
rows of the physical ops they price (`lookup.lookup`), at the snapshot
`/internal/bench/version` named when the API resolved their scope.

The Kernel Bench service's composition root for `kernel_bench` is
`annotations.py` (tests/meta/test_structure_debt.py): it owns
`row_annotations`, the kernel catalog (`registered_kernels`) and the revision
of those rules (`rules_rev`). Routes re-export those facts and hand the
annotations to the gate (every input, in memory, at startup) and to uploads.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
import weakref
import time
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Mapping, Optional, Sequence

import duckdb
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.inference import benchmarks as _benchmarks
from backend.measure.lookup import WireKey
from backend.measure.scope import MeasurementScope
from backend.inference.op_calibration import measured_utilization, node_label
from backend.service import profiles as _profiles
from backend.service.bench.annotations import (
    registered_kernels,
    row_annotations,
)
from backend.service.bench import browse as _browse
from backend.service.bench import identity as _identity
from backend.service.bench import lake as _lake_mod
from backend.service.bench import lookup as _lookup
from backend.service.bench import maintain as _maintain
from backend.service.bench.rules_revision import rules_rev
from backend.service.bench import serving as _serving
from backend.service.bench import sets as _sets
from backend.service.bench import store as _store
from backend.service.bench.store import SetRef

_LOG = logging.getLogger(__name__)

router = APIRouter()

# 200k is "someone POSTed a dump", not a kernel_bench run (one sweep CSV is
# tens of thousands of rows).
_MAX_ROWS_PER_UPLOAD = 200_000
_ROWS_PAGE_MAX = 1000
_DELETE_IDS_MAX = 5000


# ---------------------------------------------------------------------------
# The lake, the maintainer, the shared rows' revision.
# ---------------------------------------------------------------------------

#: The snapshot the shared rows last changed at (`lookup.SharedRevision`).
shared_revision = _lookup.SharedRevision()
#: Validator verdicts for owner calibration, keyed by row content.
verdicts = _lookup.Verdicts()
#: What a lookup selects from, per pinned snapshot (`lookup.Selection`).
selection = _lookup.Selection()
#: The cells lookups send, the one place they read them (`lookup.Cells`):
#: built before the lake serves (`ready`), then following it.
cells = _lookup.Cells()
#: What lookups have answered, for /health (`lookup.LookupStats`).
lookup_stats = _lookup.LookupStats()


def _serving_started(lake: _lake_mod.Lake, backup, *, read_only: bool = False) -> None:
    """Once the lake serves: the shared rows' revision starts at the
    snapshot now; the lookup's cells file follows the lake; the maintainer
    (lake maintenance; a backup every cycle) starts."""
    shared_revision.current(lake)
    cells.follow(lake)
    if not read_only:
        _maintain.start(lake, backup)


gate = _serving.Gate(annotate=row_annotations, ready=cells.refresh,
                     on_serving=_serving_started)


def health() -> dict:
    """/health's lake detail: the gate's state, the backup's lag and newest
    recoverable snapshot, free space."""
    return gate.health()


def start() -> None:
    """At server start, off the event loop: `serving.Gate.start` (open,
    import when asked into a new lake, lease, annotations, startup backup,
    then `_serving_started`) and the annotation rules' revision."""
    def run() -> None:
        gate.start()
        try:
            rules_rev()
        except Exception:
            _LOG.exception("Kernel Bench: the annotation rules' revision failed")
    threading.Thread(target=run, name="bench-startup", daemon=True).start()


def _lake() -> _lake_mod.Lake:
    """The process's lake for a read (raises `LakeUnavailable` — 503 —
    naming why it does not serve)."""
    return gate.reader()


@contextmanager
def _writing() -> Iterator[_lake_mod.Lake]:
    """The lake for a write (`LakeUnavailable` — 503 — when writes are
    refused); after the block the backup follows. A write that changes the
    shared rows says so inside the block (`shared_revision.changed`)."""
    lake = gate.writer()
    try:
        with shared_revision.writing():
            yield lake
    finally:
        gate.wrote()


def lake_unavailable(_request, exc: Exception):
    """No lake (a setting missing, the catalog or the bucket unreachable) or
    a write that lost to concurrent commits three times: one 503 naming it."""
    return JSONResponse(status_code=503,
                        content={"detail": f"Kernel Bench store unavailable: {exc}"})


# ---------------------------------------------------------------------------
# Who may read or change which set.
# ---------------------------------------------------------------------------

def set_reader(user: dict, owner: Optional[str]) -> str:
    """Whose set a caller reading one set reads: their own, or — for an
    admin reviewing a contributor's nomination — the named `owner`'s."""
    caller = _identity.uid(user)
    if not owner or owner == caller:
        return caller
    if not _identity.is_admin(user):
        raise HTTPException(status_code=403, detail="not the owner of that set")
    return owner


def _upload_id(value: str) -> str:
    """An upload id in the form the store keeps (lowercase UUID text)."""
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{value!r} is not an upload id") from None


def _ref(owner: str, kind: Literal["run", "upload"], set_id: str) -> SetRef:
    set_id = set_id.strip()
    if not set_id:
        raise HTTPException(status_code=400, detail=f"{kind} id must be non-empty")
    return SetRef(owner, kind, _upload_id(set_id) if kind == "upload" else set_id)


def _read_ref(user: dict, owner: Optional[str], run_id: Optional[str],
              upload_id: Optional[str]) -> SetRef:
    """The set a read names: a run (`run_id`) or an upload (`upload_id`) of
    the caller's — an admin may name another `owner`."""
    if not run_id and not upload_id:
        raise HTTPException(status_code=400, detail="run_id or upload_id required")
    reader = set_reader(user, owner)
    return _ref(reader, "run", run_id) if run_id else _ref(reader, "upload", upload_id)


def _as_row(r: dict) -> dict:
    """A measured row as the pages read it: its columns plus the backend's
    MFU / MBU / roofline check and the node's display name."""
    return {**r, **measured_utilization(r), "node_label": node_label(r.get("node_spec_id"))}


# ---------------------------------------------------------------------------
# Upload.
# ---------------------------------------------------------------------------

class BenchUploadBody(BaseModel):
    csv_text: str
    label: Optional[str] = None
    # Admins (public.admins) may publish straight to the SHARED corpus,
    # skipping nominate → promote (`hbench upload --public`); anyone else
    # asking to is refused.
    make_public: bool = False


def run_of(rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """The run a file's rows belong to (their `run_id` column; None when
    empty): one batch is one upload of one run, or of none. Raises
    ValueError for a file holding rows of several runs."""
    runs = {(r.get("run_id") or "").strip() for r in rows}
    if len(runs) > 1:
        raise ValueError(
            f"the file holds rows of {len(runs)} runs "
            f"({', '.join(sorted(r or '(none)' for r in runs))}); upload one file per run_id")
    return next(iter(runs), "") or None


def _sol_refusal(rows: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """The first row that beats a peak it records, for a 422."""
    from kernel_bench.emit import above_sol
    for line, row in enumerate(rows, start=2):
        why = above_sol(row)
        if why is not None:
            return f"row {line}: {why}"
    return None


def _restamp_refusal(lake: _lake_mod.Lake, *, owner: str,
                     patches: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """The first patch that would leave its row beating a recorded peak."""
    from kernel_bench.emit import above_sol
    ids = [row["id"] for row in patches]
    existing = lake.read(
        "SELECT m.id::VARCHAR AS id, m.outcome::VARCHAR AS outcome, "
        "m.peak_tflops_at_test, m.peak_bw_at_test, "
        "m.achieved_tflops, m.achieved_bw_gbps "
        "FROM lake.measurements m JOIN lake.batches b USING (batch_id) "
        "WHERE m.id IN (SELECT unnest(?::UUID[])) AND b.owner = ?::UUID",
        [ids, owner])
    by_id = {row["id"]: row for row in existing}
    for patch in patches:
        row = by_id.get(patch["id"])
        if row is None:
            continue
        row = {**row, **{column: patch[column] for column in _store.ROOFLINE_COLUMNS
                         if patch[column] is not None}}
        why = above_sol(row)
        if why is not None:
            return f"row {patch['id']}: {why}"
    return None


@router.post("/api/bench/upload")
def bench_upload(body: BenchUploadBody, user: dict = Depends(_identity.require_user)):
    """Validate a benchmark CSV (`validate_text` and the SOL gate) and store
    its rows as one batch of the caller's. Rows whose 12-column identity the
    caller already has are skipped (`duplicates`). The API accepts a personal
    access token here (`hbench upload`) as well as a session.

    Returns `{upload_id, row_count, duplicates, anomalous_count, label,
    shared}`; `row_count` counts the rows stored."""
    user_id = _identity.uid(user)
    try:
        _header, rows = _benchmarks.validate_text(body.csv_text, source="upload")
    except _benchmarks.BenchmarkValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    if not rows:
        raise HTTPException(status_code=422, detail="CSV has no data rows")
    why = _sol_refusal(rows)
    if why is not None:
        raise HTTPException(status_code=422, detail=why)
    if len(rows) > _MAX_ROWS_PER_UPLOAD:
        raise HTTPException(
            status_code=413,
            detail=f"upload has {len(rows)} rows; max is {_MAX_ROWS_PER_UPLOAD}")
    try:
        run_id = run_of(rows)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None
    if body.make_public and not _identity.is_admin(user):
        raise HTTPException(
            status_code=403, detail="publishing to the shared corpus requires admin")
    label = (body.label or "").strip() or None
    try:
        with _writing() as lake:
            res = _store.upload(
                lake, owner=user_id, upload_id=str(uuid.uuid4()), run_id=run_id,
                label=label, rows=rows, annotate=row_annotations,
                status="shared" if body.make_public else "private",
                decided_by=user_id if body.make_public else None)
            if body.make_public and res.inserted:
                shared_revision.changed(lake)
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"provider catalog unavailable, rows cannot be annotated: {e}") from e
    return {
        "upload_id": res.upload_id,
        "row_count": res.inserted,
        "duplicates": res.duplicates,
        # A heads-up only: rows faster than a peak they recorded are kept.
        "anomalous_count": sum(measured_utilization(r)["roof_status"] not in (None, "ok")
                               for r in rows),
        "label": label,
        "shared": body.make_public,
    }


# ---------------------------------------------------------------------------
# The batch pages.
# ---------------------------------------------------------------------------

@router.get("/api/bench/list")
def bench_list(user: dict = Depends(_identity.require_user)):
    """The caller's sets — runs (batches sharing a `run_id`) and uploads
    without one — newest first, each with its batches and their status
    (`sets.list_sets`)."""
    user_id = _identity.uid(user)
    return {"uploads": _sets.list_sets(_lake(), user_id)}


@router.get("/api/bench/summary")
def bench_summary(
    upload_id: Optional[str] = None,
    run_id: Optional[str] = None,
    owner: Optional[str] = None,
    user: dict = Depends(_identity.require_user),
):
    """One set of the caller's (an admin: `owner`'s): row count, models,
    ops, GPUs, frameworks, phases, and how many rows beat the roofline."""
    return _sets.summary(_lake(), _read_ref(user, owner, run_id, upload_id))


@router.get("/api/bench/rows")
def bench_rows(
    upload_id: Optional[str] = None,
    run_id: Optional[str] = None,
    owner: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
    columns: Literal["all", "corpus"] = "all",
    since: Optional[str] = None,
    after: Optional[str] = None,
    before: Optional[str] = None,
    user: dict = Depends(_identity.require_user),
):
    """One page of the set `upload_id` / `run_id` names (an admin may name
    another `owner`) — offset paging (`total` on the first page only) or
    keyset paging over row ids (`since` ≤ id, `after` < id, id < `before`;
    `next` is the id to pass as `after`). `columns=corpus` returns a row as
    the corpus states it (the CSV columns, its batch, MFU / MBU).

    Without a set: 400. Every row of the caller's was the browser's
    calibration overlay; the server now calibrates with them itself
    (`calibrate_with_mine`, `/api/bench/calibration`)."""
    if not (run_id or upload_id):
        raise HTTPException(
            status_code=400,
            detail="run_id or upload_id required (calibrating with all your rows: "
                   "send calibrate_with_mine with the estimate)")
    if limit < 1 or limit > _ROWS_PAGE_MAX:
        raise HTTPException(status_code=400, detail=f"limit must be 1..{_ROWS_PAGE_MAX}")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")
    bounds = {"since": since, "after": after, "before": before}
    for name, value in bounds.items():
        if value is not None:
            try:
                uuid.UUID(value)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"{name} must be a row id") from None
    page = _sets.rows(_lake(), _read_ref(user, owner, run_id, upload_id), limit=limit,
                      offset=offset, columns=columns, **bounds)
    return {**page, "rows": [_as_row(r) for r in page["rows"]]}


@router.get("/api/bench/calibration")
def bench_calibration(user: dict = Depends(_identity.require_user)):
    """The caller's rows as their estimates read them (`lookup.owner_report`,
    the lookup's own gate): how many calibrate (`rows_used`), how many the
    current validator refuses and why (`rows_rejected`, `rejected_reasons`,
    `first_rejected`), at the lake's snapshot now — and the digest of the
    scope an estimate calibrating with them now goes under
    (`measurement_scope`), reported for display: the browser caches no
    measured result (its in-tab cache holds theory only).
    Always the caller's own; there is no owner parameter."""
    owner = _identity.uid(user)
    lake = _lake()
    scope = MeasurementScope.from_version(version_of(lake), owner)
    snapshot = scope.snapshot
    return {**_lookup.owner_report(lake, owner, snapshot=snapshot, verdicts=verdicts),
            "snapshot": snapshot, "measurement_scope": scope.digest}


@router.get("/api/bench/overlap")
def bench_overlap(
    upload_id: Optional[str] = None,
    run_id: Optional[str] = None,
    owner: Optional[str] = None,
    limit: int = 200,
    user: dict = Depends(_identity.require_user),
):
    """The set against the shared corpus, per op shape: how many of its op
    shapes shared already has, the new ones, and where it is faster.
    Computed now; the admin reads it in review, right before promoting."""
    ref = _read_ref(user, owner, run_id, upload_id)
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be 1..500")
    return _sets.overlap(_lake(), ref, limit=limit)


# ---------------------------------------------------------------------------
# Deleting (the caller's own).
# ---------------------------------------------------------------------------

class BenchDeleteRowsBody(BaseModel):
    ids: list[str]


@router.post("/api/bench/rows/delete")
def bench_delete_rows(body: BenchDeleteRowsBody, user: dict = Depends(_identity.require_user)):
    """Delete selected rows of the caller's (ids are row UUIDs); a batch
    left without rows goes with them. Returns how many were deleted."""
    user_id = _identity.uid(user)
    ids = [i.strip() for i in body.ids if i and i.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="ids must be non-empty")
    if len(ids) > _DELETE_IDS_MAX:
        raise HTTPException(status_code=413, detail=f"at most {_DELETE_IDS_MAX} ids per call")
    for i in ids:
        try:
            uuid.UUID(i)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"{i!r} is not a row id") from None
    with _writing() as lake:
        shared = _sets.any_shared(lake, user_id, ids)
        out = {"deleted": _store.delete_rows(lake, owner=user_id, ids=ids)}
        if shared and out["deleted"]:
            shared_revision.changed(lake)
    return out


class BenchRestampRow(BaseModel):
    """A row id and the roofline columns to set (`store.ROOFLINE_COLUMNS`);
    a column left out keeps its value."""
    id: str
    peak_tflops_at_test: float | None = None
    peak_bw_at_test: float | None = None
    achieved_tflops: float | None = None
    achieved_bw_gbps: float | None = None


class BenchRestampBody(BaseModel):
    rows: list[BenchRestampRow]


@router.post("/api/bench/rows/restamp")
def bench_restamp_rows(body: BenchRestampBody,
                       user: dict = Depends(_identity.require_user)):
    """Correct the roofline columns of the caller's rows (`store.restamp`):
    the recorded PEAKS when the GPU table is corrected, the ACHIEVED rates
    when the walk corrects a launch's numerators. In place, so a nominated
    or shared row keeps its status.

    The latency is never touched here — a card does not get faster because a
    sheet was wrong, which is the other half of a peak correction
    (`op_calibration` compensates for a moved peak by scaling the measured
    latency, so until the rows are restamped every estimate reading them is
    wrong the other way).
    """
    user_id = _identity.uid(user)
    rows = [r.model_dump() for r in body.rows]
    if not rows:
        raise HTTPException(status_code=400, detail="rows must be non-empty")
    if len(rows) > _DELETE_IDS_MAX:
        raise HTTPException(
            status_code=413,
            detail=f"at most {_DELETE_IDS_MAX} rows per call")
    for row in rows:
        try:
            uuid.UUID(row["id"])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"{row['id']!r} is not a row id") from None
        given = [c for c in _store.ROOFLINE_COLUMNS if row[c] is not None]
        if not given:
            raise HTTPException(
                status_code=400,
                detail=f"row {row['id']} sets none of {list(_store.ROOFLINE_COLUMNS)}")
        for column in given:
            # A peak is a positive rate; an achieved rate may be 0 (a gather
            # or a permute does no FLOPs), never negative or not a number.
            ok = row[column] > 0 if column.startswith("peak_") else row[column] >= 0
            if not ok:
                raise HTTPException(
                    status_code=400,
                    detail=f"{column} must be "
                           f"{'positive' if column.startswith('peak_') else 'non-negative'}")
    with _writing() as lake:
        shared = _sets.any_shared(lake, user_id, [r["id"] for r in rows])
        why = _restamp_refusal(lake, owner=user_id, patches=rows)
        if why is not None:
            raise HTTPException(status_code=422, detail=why)
        out = {"updated": _store.restamp(lake, owner=user_id, rows=rows)}
        if shared and out["updated"]:
            shared_revision.changed(lake)
    return out


def _delete_set(ref: SetRef) -> dict:
    with _writing() as lake:
        shared = any(b["status"] == "shared" for b in _store.batches_of(lake, ref))
        out = {"deleted": bool(_store.delete_set(lake, ref))}
        if shared and out["deleted"]:
            shared_revision.changed(lake)
    return out


@router.delete("/api/bench/run/{run_id}")
def bench_delete_run(run_id: str, user: dict = Depends(_identity.require_user)):
    """Delete every batch of one run of the caller's, and their rows."""
    user_id = _identity.uid(user)
    return {"run_id": run_id, **_delete_set(_ref(user_id, "run", run_id))}


@router.delete("/api/bench/{upload_id}")
def bench_delete(upload_id: str, user: dict = Depends(_identity.require_user)):
    """Delete one upload of the caller's, and its rows."""
    user_id = _identity.uid(user)
    return {"upload_id": upload_id, **_delete_set(_ref(user_id, "upload", upload_id))}


# ---------------------------------------------------------------------------
# Review: the owner nominates, an admin promotes or rejects. Review state is
# the batch's status (store.py); no row changes.
# ---------------------------------------------------------------------------

def _nominate(ref: SetRef, within_peak: bool) -> int:
    """Nominate the set's private batches — with `within_peak`, offering only
    the rows within the peaks they recorded. 404 when none is private."""
    with _writing() as lake:
        changed = _store.set_status(lake, [ref], from_status=["private"],
                                    to_status="nominated", within_peak=within_peak)
    if not changed:
        raise HTTPException(status_code=404, detail=f"no private batch in that {ref.kind}")
    return len(changed)


@router.post("/api/bench/run/{run_id}/contribute")
def bench_contribute_run(run_id: str, within_peak: bool = False,
                         user: dict = Depends(_identity.require_user)):
    """Nominate a run's private batches for the shared corpus."""
    user_id = _identity.uid(user)
    n = _nominate(_ref(user_id, "run", run_id), within_peak)
    return {"run_id": run_id, "nominated": True, "batches": n}


@router.post("/api/bench/{upload_id}/contribute")
def bench_contribute(upload_id: str, within_peak: bool = False,
                     user: dict = Depends(_identity.require_user)):
    """Nominate one private upload of the caller's for the shared corpus; an
    admin then promotes or rejects it."""
    user_id = _identity.uid(user)
    n = _nominate(_ref(user_id, "upload", upload_id), within_peak)
    return {"upload_id": upload_id, "nominated": True, "batches": n}


@router.get("/api/bench/nominations")
def bench_nominations(user: dict = Depends(_identity.require_admin)):
    """ADMIN review list: nominated batches, one entry per upload, newest
    nomination first (`sets.nominations`), each naming its `submitter`."""
    noms = _sets.nominations(_lake())
    people = _profiles.who(n["owner"] for n in noms)
    return {"nominations": [{**n, "submitter": people.get(n["owner"]) or {}} for n in noms]}


class BenchSetRef(BaseModel):
    """A run or an upload of one owner, as the workspace and the review
    list name it (run ids are only unique within an owner)."""
    owner: str
    kind: Literal["run", "upload"]
    id: str

    def ref(self) -> SetRef:
        try:
            owner = str(uuid.UUID(self.owner))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"{self.owner!r} is not an owner id") from None
        return _ref(owner, self.kind, self.id)


class BenchReviewBody(BaseModel):
    sets: list[BenchSetRef]


def _decide(body: BenchReviewBody, to_status: Literal["shared", "rejected"],
            admin: str) -> list[dict]:
    """Move the named sets' nominated batches to `to_status` (any owner's:
    the admin reviews contributors' sets), in one transaction; returns per
    set how many batches moved."""
    if not body.sets:
        raise HTTPException(status_code=400, detail="no sets named")
    refs = [s.ref() for s in body.sets]
    with _writing() as lake:
        changed = _store.set_status(lake, refs, from_status=["nominated"],
                                    to_status=to_status, decided_by=admin)
        if changed and to_status == "shared":
            shared_revision.changed(lake)

    def names(ref: SetRef, b: dict) -> bool:
        set_id = b["run_id"] if ref.kind == "run" else str(b["upload_id"])
        return str(b["owner"]) == ref.owner and set_id == ref.id
    return [{"owner": r.owner, "kind": r.kind, "id": r.id,
             "batches": sum(names(r, b) for b in changed)} for r in refs]


@router.post("/api/bench/promote")
def bench_promote(body: BenchReviewBody, user: dict = Depends(_identity.require_admin)):
    """ADMIN: share the named sets' nominated batches — of a batch nominated
    within peak, only its rows within peak are shared, in one transaction.
    How a set overlaps the shared corpus is read in review (`/api/bench/overlap`), before the admin decides."""
    promoted = _decide(body, "shared", _identity.uid(user))
    if not any(p["batches"] for p in promoted):
        raise HTTPException(status_code=404, detail="no nominated batch in those sets")
    return {"promoted": promoted}


@router.post("/api/bench/reject")
def bench_reject(body: BenchReviewBody, user: dict = Depends(_identity.require_admin)):
    """ADMIN: reject the named sets' nominated batches (they stay their
    owners', calibrating their own estimates; the shared rows do not
    change)."""
    return {"rejected": _decide(body, "rejected", _identity.uid(user))}


# ---------------------------------------------------------------------------
# The measurement catalog.
# ---------------------------------------------------------------------------

@router.get("/api/bench/providers")
def bench_providers():
    """The measurement catalog: which libraries this harness can bench, which
    kernels each ships, and what each kernel DECLARES it can serve.

    Published from the registry rather than a hand-maintained list, so the
    surface cannot drift from what would actually run. Grouped by library
    because that is how the libraries are installed and how availability is
    decided — one import gate per library, not per kernel.

    `capabilities` is `None` for a kernel still on a hand-written gate. That is
    reported as unknown rather than filled in: a kernel whose scope was never
    declared has not promised anything, and showing an empty constraint list
    would read as "serves everything".

    No auth — this describes the harness, not anyone's measurements.
    """
    try:
        from kernel_bench.providers import all_providers, get_library
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"provider registry unavailable: {type(e).__name__}: {e}",
        ) from e

    libraries: dict = {}
    for p in all_providers():
        _lib = get_library(p.library)
        lib = libraries.setdefault(p.library, {
            "library": p.library,
            # The import gate is per-KERNEL (sglang's rope kernels gate on
            # `sgl_kernel`, its conv kernels on `sglang`), so this collects
            # them rather than taking whichever registered first.
            "import_names": [],
            "provisioning": "source_build" if _lib.source_build else "pip",
            "pip": " ".join(_lib.pip),
            "build_hint": _lib.build_hint,
            "kernels": [],
        })
        if p.import_name not in lib["import_names"]:
            lib["import_names"].append(p.import_name)
        lib["kernels"].append({
            "kernel_impl": p.kernel_impl,
            "signature": p.signature,
            "fused_op": p.fused_op or "",
            "min_sm": p.min_sm,
            "capabilities": p.capabilities(),
        })

    out = sorted(libraries.values(), key=lambda d: d["library"])
    for lib in out:
        lib["kernels"].sort(key=lambda d: d["kernel_impl"])
    declared = sum(1 for lib in out for k in lib["kernels"]
                   if k["capabilities"] is not None)
    total = sum(len(lib["kernels"]) for lib in out)
    return {"libraries": out, "kernel_count": total, "declared_count": declared}


# ---------------------------------------------------------------------------
# The Kernel Bench page.
# ---------------------------------------------------------------------------

def _coverage(out: dict) -> dict:
    """Rows per registered kernel, grouped by library — every registered
    kernel listed, measured or not — and the rows no kernel claims."""
    measured = out.pop("kernels", {}) or {}
    return {**out, "libraries": {
        lib: {k: int(measured.get(k, 0)) for k in kernels}
        for lib, kernels in registered_kernels().items()}}


@router.get("/api/bench/browse")
def bench_browse(
    view: Literal["envelope", "rows", "workload", "curves", "coverage"] = "envelope",
    source: Literal["", "shared", "mine"] = "",
    run: str = "",
    upload: str = "",
    owner: str = "",
    model: str = "",
    signature: str = "",
    gpu: str = "",
    framework: str = "",
    dtype: str = "",
    phase: str = "",
    kernel: str = "",
    outcome: str = "",
    check: Literal["", "over", "ok"] = "",
    key: str = "",
    batch: Optional[int] = None,
    context: Optional[int] = None,
    sort: str = "",
    dir: Literal["", "asc", "desc"] = "",
    page: int = 0,
    page_size: int = 50,
    user: Optional[dict] = Depends(_identity.get_current_user),
):
    """One page of one Kernel Bench view (`browse.browse`) over the shared
    corpus and the caller's own rows (`source` narrows to one; `run` /
    `upload` to one set of the caller's — an admin reviewing a nomination
    names its `owner`; `key` to one envelope's rows; `outcome` to the rows
    recording one outcome — measurements when unnamed, the only one the
    latency views answer, or `all` on the rows view: measurements and
    failures together). Signed out, only the
    shared corpus is in view. A request beyond a view's bounds is refused
    with a message naming the bound (400)."""
    user_id = _identity.uid(user) if user else None
    in_set = bool(run or upload)
    if (source == "mine" or in_set) and user_id is None:
        raise HTTPException(status_code=401, detail="sign in to see your own measurements")
    filters: dict = {"model": model, "signature": signature, "gpu": gpu,
                     "framework": framework, "dtype": dtype, "phase": phase,
                     "kernel": kernel, "outcome": outcome, "check": check,
                     "batch": batch, "context": context}
    if key:
        try:
            filters["key"] = json.loads(key)
        except ValueError:
            raise HTTPException(status_code=400, detail="key is not JSON") from None
    if view == "curves":
        filters["call_tokens"] = _browse.call_tokens()
    q = _browse.Query(
        view=view, caller=user_id, source=source, run=run,
        upload=_upload_id(upload) if upload else "",
        set_owner=set_reader(user, owner) if in_set else None,
        filters=filters, sort=sort, desc={"": None, "asc": False, "desc": True}[dir],
        page=page, page_size=page_size)
    try:
        out = _browse.browse(_lake(), q)
    except _browse.Refused as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    if view in ("rows", "workload"):
        out["items"] = [_as_row(r) for r in out.get("items") or []]
    elif view == "envelope":
        out["items"] = [{**e, "winner": _as_row(e["winner"])} for e in out.get("items") or []]
    elif view == "coverage":
        out = _coverage(out)
    # Plain JSON already (`browse._plain`): skip FastAPI's encoder, ~200 ms
    # on the largest answers.
    return JSONResponse(content=out)


# ---------------------------------------------------------------------------
# The measurements estimates read (internal: the API and its compute workers).
# ---------------------------------------------------------------------------

class LookupBody(BaseModel):
    #: The physical ops an estimate prices (`measurements.WireKey`).
    keys: list[WireKey] = Field(max_length=_lookup.MAX_KEYS)
    #: The owner whose rows join the shared ones (`calibrate_with_mine`).
    owner: Optional[str] = None
    #: The snapshot to answer at (`/internal/bench/version`); None: now.
    snapshot: Optional[int] = None


_lake_ids: "weakref.WeakKeyDictionary[_lake_mod.Lake, str]" = weakref.WeakKeyDictionary()


def _lake_id(lake: _lake_mod.Lake) -> str:
    """The lake's id (`lake_meta`, written once when the lake is made),
    read once per opened lake."""
    got = _lake_ids.get(lake)
    if got is None:
        got = _lake_ids[lake] = _store.meta(lake)["lake_id"]
    return got


@router.post("/internal/bench/lookup")
def bench_lookup(body: LookupBody):
    """The candidate rows of each key (`lookup.lookup`), at `snapshot`:
    `{lake_id, snapshot, columns, rows: [[row cells…] per key],
    excluded_mine}`. A snapshot that cannot be read (expired) is 503, never
    an empty answer. Row validity is the write path's and the registered
    data-rules migration's fact, not a per-row read-path check."""
    owner = None
    if body.owner:
        try:
            owner = str(uuid.UUID(body.owner))
        except ValueError:
            raise HTTPException(status_code=400, detail="owner is not a user id") from None
    lake = _lake()
    snapshot = lake.snapshot_id() if body.snapshot is None else int(body.snapshot)
    keys = [dict(k) for k in body.keys]
    started = time.perf_counter()
    try:
        out = _lookup.lookup(lake, keys, owner=owner, snapshot=snapshot,
                             selection=selection, cells=cells, stats=lookup_stats)
    except _lookup.TooManyRows as e:
        raise HTTPException(status_code=503, detail=str(e)) from None
    except duckdb.Error as e:
        raise _lake_mod.LakeUnavailable(f"snapshot {snapshot} cannot be read: {e}") from e
    return JSONResponse(content={
        "lake_id": _lake_id(lake), "snapshot": snapshot,
        "service_s": round(time.perf_counter() - started, 6), **out})


def version_of(lake: _lake_mod.Lake) -> dict:
    """What a measured estimate read from `lake` is a function of, now: the
    lake, its snapshot (the lookup's pin), the snapshot its shared rows last
    changed at — each with the identity of the lake's content at it
    (`Lake.content_id`), which tells divergent copies of one lake apart —
    and the annotation rules' revision (`rules_rev`)."""
    (snapshot, content), (shared, shared_content) = shared_revision.current(lake)
    return {"lake_id": _lake_id(lake), "snapshot": snapshot, "snapshot_content": content,
            "shared_snapshot": shared, "shared_snapshot_content": shared_content,
            "rules_rev": rules_rev()}


@router.get("/internal/bench/version")
def bench_version():
    """`version_of` the service's lake."""
    return version_of(_lake())
