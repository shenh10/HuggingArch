"""The batch pages over the lake: a user's sets, one set's summary and rows,
the set against the shared corpus, the admin's nominations. Same response shapes as the PostgREST
paths they replace (backend/service/bench/routes.py), with batches as the
review unit (a run lists its batches).

Review state lives on batches (docs/measured_calibration.md §6.1), not on
rows. The fields that described the old per-row review are derived from the
batches' status:

  * a set's `is_public` — some batch of it is `shared`;
  * a set's `nominated` — some batch of it is `nominated`;
  * a set's `status` — what is left to do with it, first match of
    `private` (some batch still to nominate: "contribute a run" nominates
    exactly those) > `nominated` > `shared` > `rejected`; each batch's own
    status is in `batches`;
  * a row's `is_public` — the row is shared (`store.SHARED_ROW`); its
    `nominated_at` — its batch's, when the batch offers the row (nominated
    or shared, and within its peaks if the batch was nominated that way).

Times are ISO-8601 text in UTC, as PostgREST wrote them; ids are text; the
JSON cells (`shape_json`, `dtype_env_json`) come back as objects, as the
JSONB columns did.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterable, Optional, get_args

from backend.inference import benchmarks as _benchmarks
from backend.service.bench.lake import Lake
from backend.service.bench.store import (ANNOT, IS_MEASUREMENT,
    JSON_COLUMNS, MBU, MFU, OUTCOME, SHARED_ROW, WITHIN_PEAK, SetRef, Status)

if TYPE_CHECKING:
    from backend.service.bench.derived import Derived

# A row its batch offers to the shared corpus once nominated.
_OFFERED = f"(NOT b.within_peak OR {WITHIN_PEAK})"
_EXPORT_CHUNK = 50_000


def _iso(column: str) -> str:
    """A stored time (TIMESTAMP in UTC, store.py) as ISO-8601 text, the
    form PostgREST returned."""
    return f"strftime({column}, '%Y-%m-%dT%H:%M:%S.%f+00:00')"


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


_CORPUS = ", ".join(f"m.{_q(c)}" for c in _benchmarks.ALL_COLUMNS)
# `columns=corpus`: a measured row as the corpus states it, its batch, MFU / MBU.
_CORPUS_SELECT = (f"{_CORPUS}, m.id::VARCHAR AS id, b.upload_id::VARCHAR AS upload_id, "
                  f"{MFU} AS mfu, {MBU} AS mbu")
# `columns=all`: what `select=*,mfu,mbu` on `user_benchmarks` returned, the
# review fields derived from the batch (module docstring), and its status.
_ALL_SELECT = (f"{_CORPUS_SELECT}, b.owner::VARCHAR AS user_id, b.label, "
               f"{_iso('m.created_at')} AS created_at, {SHARED_ROW} AS is_public, "
               f"CASE WHEN b.status IN ('nominated', 'shared') AND {_OFFERED} "
               f"THEN {_iso('b.nominated_at')} END AS nominated_at, b.status")


def _select(columns: str) -> str:
    """The SELECT list of a set's rows: `corpus` (the corpus columns) or
    `all` (with the batch's owner and label)."""
    if columns == "corpus":
        return _CORPUS_SELECT
    if columns == "all":
        return _ALL_SELECT
    raise ValueError(f"columns is one of all, corpus, not {columns!r}")

_SET_BATCHES = "m.batch_id IN (SELECT batch_id FROM lake.batches WHERE {where})"


def _decoded(row: dict) -> dict:
    """A measured row with its JSON cells as objects."""
    for c in JSON_COLUMNS:
        if row.get(c) is not None:
            row[c] = json.loads(row[c])
    return row


# --------------------------------------------------------------------------
# Summaries: one row per batch, merged per set.
# --------------------------------------------------------------------------

_TEXT_FACETS = (("frameworks", "framework"), ("gpus", "gpu"), ("models", "model"),
                ("ops", "op"), ("phases", "phase"))
# A batch summary's counts, each with the SQL that counts it (over m, b);
# `_merged` adds exactly these up.
_COUNT_SQL = (
    ("mfu_over", f"count(*) FILTER (WHERE {MFU} > 1)"),
    ("mbu_over", f"count(*) FILTER (WHERE {MBU} > 1)"),
    ("anomalous", f"count(m.id) FILTER (WHERE NOT {WITHIN_PEAK})"),
    ("checked", f"count(*) FILTER (WHERE {MFU} IS NOT NULL OR {MBU} IS NOT NULL)"),
    # `row_count` counts every row of the batch; these two say how it
    # divides. Without them a set of 10,000 rows of which 400 are recorded
    # refusals reads as 10,000 measurements, and `row_count` stops being the
    # sum of the states beside it. `count(m.id)`: a batch with no matching
    # row joins as one all-NULL row, whose OUTCOME coalesces to measured.
    ("measured", f"count(m.id) FILTER (WHERE {IS_MEASUREMENT})"),
    ("refused", f"count(m.id) FILTER (WHERE NOT {IS_MEASUREMENT})"),
    # `refused` by what happened instead. A batch whose rows all failed has
    # nothing to check (checked = 0); these let the batch pages say WHY —
    # the Check column reads them rather than showing a bare "–".
    ("refused_oom", f"count(*) FILTER (WHERE {OUTCOME} = 'oom')"),
    ("refused_mismatch", f"count(*) FILTER (WHERE {OUTCOME} = 'mismatch')"),
    ("refused_error", f"count(*) FILTER (WHERE {OUTCOME} = 'error')"),
    ("refused_crash", f"count(*) FILTER (WHERE {OUTCOME} = 'crash')"),
    ("refused_timeout", f"count(*) FILTER (WHERE {OUTCOME} = 'timeout')"),
)

_BATCH_SUMMARY = (
    "SELECT b.owner::VARCHAR AS owner, b.upload_id::VARCHAR AS upload_id, b.piece, "
    f"b.run_id, b.label, {_iso('b.created_at')} AS created_at, b.status, b.within_peak, "
    f"{_iso('b.nominated_at')} AS nominated_at, {_iso('b.decided_at')} AS decided_at, "
    "count(m.id) AS row_count, "
    + ", ".join(f"list(DISTINCT m.{col}) FILTER (WHERE m.{col} <> '') AS {key}"
                for key, col in _TEXT_FACETS)
    + "".join(f", {sql} AS {name}" for name, sql in _COUNT_SQL)
    + " FROM lake.batches b LEFT JOIN lake.measurements m"
    " ON m.batch_id = b.batch_id AND {rows}"
    " WHERE {where}"
    # created_at, upload_id, piece (ISO text sorts as the time does)
    " GROUP BY ALL ORDER BY 6, 2, 3")


def _batch_summaries(lake: Lake, where: str, params: list, *,
                     rows: str = "TRUE") -> list[dict]:
    """Each batch matching `where` (over alias b) with the summary of its
    rows matching `rows` (over m, b), earliest first."""
    return lake.read(_BATCH_SUMMARY.format(where=where, rows=rows), params)


def _merged(batches: Iterable[dict]) -> dict:
    """The summary of several batches' rows (the shape `_summary_from_set`
    gave a `user_benchmark_sets` row)."""
    out: dict[str, Any] = {"row_count": 0, **{k: set() for k, _ in _TEXT_FACETS},
                           **{name: 0 for name, _sql in _COUNT_SQL}}
    for b in batches:
        out["row_count"] += b["row_count"]
        for k, _ in _TEXT_FACETS:
            out[k].update(b[k] or ())
        for k, _sql in _COUNT_SQL:
            out[k] += b[k]
    return {**out, **{k: sorted(out[k]) for k, _ in _TEXT_FACETS}}


def _batch_entry(b: dict) -> dict:
    return {**_merged([b]), **{k: b[k] for k in (
        "upload_id", "piece", "label", "created_at", "status", "within_peak",
        "nominated_at", "decided_at")}}


def list_sets(lake: Lake, owner: str) -> list[dict]:
    """`/api/bench/list`: the owner's runs and standalone uploads, newest
    first, each with its batches and their status.

    A set is a run (its batches share a `run_id`) or an upload without one.
    Its `upload_id` and `label` are its earliest batch's (by `created_at`,
    then `upload_id`, then `piece`), `created_at` the earliest; the summary
    (`row_count`, frameworks … `checked`) is over all its batches' rows;
    `is_public` / `nominated` / `status` are derived (module docstring)."""
    sets: dict[tuple, list[dict]] = {}
    for b in _batch_summaries(lake, "b.owner = ?::UUID", [owner]):
        key = ("run", b["run_id"]) if b["run_id"] else ("upload", b["upload_id"])
        sets.setdefault(key, []).append(b)
    out = []
    for (kind, sid), batches in sets.items():
        first = batches[0]
        statuses = {b["status"] for b in batches}
        out.append({
            **_merged(batches),
            "id": sid,
            "kind": kind,
            "run_id": sid if kind == "run" else None,
            "upload_id": first["upload_id"],
            "label": first["label"],
            "created_at": first["created_at"],
            "is_public": "shared" in statuses,
            "nominated": "nominated" in statuses,
            "status": next(s for s in get_args(Status) if s in statuses),
            "batches": [_batch_entry(b) for b in batches],
        })
    out.sort(key=lambda s: s["created_at"], reverse=True)
    return out


def _in_set(ref: SetRef) -> tuple[str, list]:
    """The predicate over b naming the set's batches (`SetRef.where`)."""
    where, params = ref.where()
    return f"b.batch_id IN (SELECT batch_id FROM lake.batches WHERE {where})", params


def summary(lake: Lake, ref: SetRef) -> dict:
    """`/api/bench/summary`: counts, frameworks, gpus, models, ops, phases,
    over-peak counts of one set, over all its batches' rows (a set with no
    batches: zero rows, empty lists)."""
    where, params = _in_set(ref)
    return _merged(_batch_summaries(lake, where, params))


# --------------------------------------------------------------------------
# Rows.
# --------------------------------------------------------------------------

def rows(lake: Lake, ref: SetRef, *, limit: int, offset: int = 0,
         after: Optional[str] = None, before: Optional[str] = None,
         since: Optional[str] = None, columns: str = "all",
         count: bool = True) -> dict:
    """`/api/bench/rows` for one set: a page in row-id order.

    Keyset paging when any bound is given (`since` ≤ id, `after` < id,
    id < `before`; ids are row UUIDs): `{rows, limit, next}`, `next` the id
    to pass as `after` for the following page, None on the last. Otherwise
    offset paging: `{rows, total, offset, limit}`, `total` counted only on
    the first page (`offset` 0, and `count`), None after. `columns` is `all`
    or `corpus` (module constants `_ALL_SELECT` / `_CORPUS_SELECT`); both
    carry `mfu` / `mbu`."""
    select = _select(columns)
    where, params = _in_set(ref)
    bounds = [(op, v) for op, v in ((">=", since), (">", after), ("<", before)) if v is not None]
    cond = "".join(f" AND m.id {op} ?::UUID" for op, _ in bounds)
    page = lake.read(
        f"SELECT {select} FROM lake.measurements m JOIN lake.batches b "
        f"ON b.batch_id = m.batch_id WHERE {where}{cond} ORDER BY m.id LIMIT ?"
        + ("" if bounds else " OFFSET ?"),
        [*params, *(v for _, v in bounds), limit, *(() if bounds else (offset,))])
    page = [_decoded(r) for r in page]
    if bounds:
        return {"rows": page, "limit": limit,
                "next": page[-1]["id"] if len(page) == limit else None}
    total = None
    if offset == 0 and count:
        total = lake.read(
            "SELECT count(*) AS n FROM lake.measurements m JOIN lake.batches b "
            f"ON b.batch_id = m.batch_id WHERE {where}", params)[0]["n"]
    return {"rows": page, "total": total, "offset": offset, "limit": limit}


# --------------------------------------------------------------------------
# The set against the shared corpus.
# --------------------------------------------------------------------------

# One op shape of a scope: its envelope key (phys from the derived file's `annot`,
# env, phase — placeholders, with no annotations, are in no envelope), the
# fastest row by (latency, id).
_OP_SHAPES = (
    "SELECT m.phase, a.phys, m.env, any_value(a.signature) AS signature, "
    "any_value(a.physical_shape) AS physical_shape, any_value(m.gpu) AS gpu, "
    "min(m.latency_us) AS latency_us, "
    "arg_min(m.framework, (m.latency_us, m.id)) AS framework, "
    "arg_min(m.kernel_impl, (m.latency_us, m.id)) AS kernel_impl, count(*) AS n "
    "FROM lake.measurements m JOIN lake.batches b ON b.batch_id = m.batch_id "
    f"JOIN {ANNOT} a ON a.input_id = m.input_id {{keys}} "
    # Only a measurement can be the fastest row: a recorded refusal has no
    # latency, and left in it both inflated `n` and — for an op shape whose
    # every row is a refusal — produced a winner with no latency at all,
    # which the page renders as 0.0us.
    f"WHERE a.phys IS NOT NULL AND {IS_MEASUREMENT} AND {{where}} "
    "GROUP BY m.phase, a.phys, m.env")


def overlap(lake: Lake, ref: SetRef, derived: "Derived", *, limit: int = 200) -> dict:
    """`/api/bench/overlap`: the set against the shared corpus per op shape
    (the fields of the Postgres bench_set_overlap, 0016). An op shape is an
    envelope key (`signature`, `physical_shape`, dtypes, gpu, phase — not
    the deployment that cut the shape); per op shape the set's fastest row against shared's.
    `pending` (op shapes not yet indexed) is gone: nothing is indexed, the
    counts are always complete. `limit` bounds `in_shared_list` (1..500)."""
    derived.reach(lake, lake.snapshot_id())      # its annotations hold every input
    where, params = _in_set(ref)
    mine = _OP_SHAPES.format(keys="", where=where)
    shared = _OP_SHAPES.format(
        keys="JOIN mine k ON k.phase = m.phase AND k.phys = a.phys AND k.env = m.env",
        where=SHARED_ROW)
    c = lake.read(
        f"WITH mine AS ({mine}), shared AS ({shared}) "
        "SELECT mine.signature, mine.physical_shape, mine.gpu, mine.phase, mine.latency_us, "
        "mine.framework, mine.kernel_impl, shared.latency_us AS shared_latency_us, "
        "shared.framework AS shared_framework, shared.kernel_impl AS shared_kernel_impl, "
        "shared.n AS shared_candidates "
        "FROM mine LEFT JOIN shared USING (phase, phys, env)", params)
    in_shared = [r for r in c if r["shared_latency_us"] is not None]
    faster = [r for r in in_shared if r["latency_us"] < r["shared_latency_us"]]
    by_op: dict[str, dict] = {}
    for r in c:
        o = by_op.setdefault(r["signature"], {"op": r["signature"], "op_shapes": 0,
                                              "in_shared": 0, "faster": 0})
        o["op_shapes"] += 1
        o["in_shared"] += r["shared_latency_us"] is not None
    for r in faster:
        by_op[r["signature"]]["faster"] += 1
    in_shared.sort(key=lambda r: (r["latency_us"] / r["shared_latency_us"], r["signature"],
                                  r["physical_shape"], r["gpu"], r["phase"]))
    return {
        "op_shapes": len(c),
        "in_shared": len(in_shared),
        "new": len(c) - len(in_shared),
        "faster_than_shared": len(faster),
        "by_op": sorted(by_op.values(), key=lambda o: (-o["op_shapes"], o["op"])),
        "in_shared_list": in_shared[:max(1, min(limit, 500))],
    }


# --------------------------------------------------------------------------
# Review.
# --------------------------------------------------------------------------

def nominations(lake: Lake) -> list[dict]:
    """`/api/bench/nominations`: nominated batches, newest nomination first,
    one entry per upload (its nominated pieces), named as the set reference
    promote / reject take (`owner`, `kind` = upload, `id` = the upload id).
    The summary counts the rows the batch offers (within its peaks when it
    was nominated that way — what promote would share); `nominated_at` is
    the latest of its pieces'; `run_id` names the run it belongs to; the
    route adds the owners' profiles."""
    grouped: dict[tuple, list[dict]] = {}
    for b in _batch_summaries(lake, "b.status = 'nominated'", [], rows=_OFFERED):
        grouped.setdefault((b["owner"], b["upload_id"]), []).append(b)
    out = [{**_merged(batches), "owner": owner, "kind": "upload", "id": upload_id,
            "run_id": batches[0]["run_id"], "label": batches[0]["label"],
            "nominated_at": max(b["nominated_at"] for b in batches),
            "batches": [_batch_entry(b) for b in batches]}
           for (owner, upload_id), batches in grouped.items()]
    out.sort(key=lambda n: n["nominated_at"], reverse=True)
    return out


# --------------------------------------------------------------------------
# Deleting.
# --------------------------------------------------------------------------

def any_shared(lake: Lake, owner: str, ids: list[str]) -> bool:
    """Whether any of the owner's rows `ids` (UUID text) is a shared row —
    deleting them changes the shared rows (`lookup.SharedRevision`)."""
    if not ids:
        return False
    return bool(lake.read(
        "SELECT EXISTS (SELECT 1 FROM lake.measurements m JOIN lake.batches b "
        "ON b.batch_id = m.batch_id WHERE b.owner = ?::UUID "
        f"AND m.id IN (SELECT unnest(?::UUID[])) AND {SHARED_ROW}) AS s",
        [owner, list(ids)])[0]["s"])
