"""The measurements an estimate reads (docs/measured_calibration.md §7.2).

`lookup` answers `POST /internal/bench/lookup`: for each requested physical
key (signature, physical shape, dtype environment, gpu, phase), the candidate
rows — every shared row, and every row of the named owner, whose input's
in-memory annotations (`annot`, store.py: the current `physical_identity` of
the current shape) carry the key's signature and physical shape and whose
own dtype environment, gpu and phase equal the key's. Equality on the text
fields, no hash, no other preselection. One projection decides membership —
the service's annotations — and the client indexes a row under the key that
selected it, never re-projecting its stored shape (a legacy row whose shape
spells an old axis is found under the current key).

Pinned: the rows are read at one lake snapshot, `measurements` and `batches`
both `AT (VERSION => n)` (held in memory per snapshot, `Selection`), so an
answer always matches the identity it was asked for. The annotations are joined as they are now: inputs are never
deleted, so every input of an old snapshot has its entry, and an entry of an
input added later joins no row of the old snapshot (tests pin both).

Validity is owned by the write path and the registered data-rules migration,
not by this hot read path. Upload validates every new row; when the validator
changes, the registered migration updates or removes the rows it now refuses
before `lake_meta.data_rules` reaches the live revision and the lake serves.
Lookup sends the selected rows without re-running Python's validator for
every row in every process. `/api/bench/calibration` still reports an owner's
rows under the current validator; it is a page over that owner's rows, not
the estimate's hot path.

Rows travel as the CSV cells the estimator's registry reads
(`op_calibration.ROW_COLUMNS`: the identity and the measurement — the text
`validate_text` reads, the cells the shared-corpus file carried), each cell
an index into the answer's table of distinct texts, in `columns` order, with
the row's `origin` (`shared` / `mine`). A row both shared and the owner's is sent
twice, once per origin: the client indexes shared rows first and the
owner's over them, as the corpus files were.

`SharedRevision` names the snapshot the shared rows last changed at, the
identity a shared estimate is cached under: a private upload moves the
snapshot, not the shared rows.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Optional, Sequence

from backend.inference import benchmarks as _benchmarks
from backend.inference.op_calibration import ROW_COLUMNS
from backend.measure.lookup import WireKey
from backend.service.bench.lake import Lake
from backend.service.bench.store import SHARED_ROW, WITHIN_PEAK, canonical_json

_LOG = logging.getLogger(__name__)

#: The most keys one lookup may name (an estimate asks a few hundred).
MAX_KEYS = 20_000
#: What a lookup sends per row, in order: the columns the estimator's
#: registry reads (`op_calibration.ROW_COLUMNS`), then its origin. The rest
#: of a row (provenance the estimate never reads) stays in the lake.
COLUMNS: tuple[str, ...] = (*ROW_COLUMNS, "origin")
#: Where the identity and the key's own fields sit in a held row's cells.
_IDENTITY = len(_benchmarks.IDENTITY_COLUMNS)
if ROW_COLUMNS[:_IDENTITY] != _benchmarks.IDENTITY_COLUMNS:
    raise AssertionError("ROW_COLUMNS starts with the 12-column identity")
_DTYPE, _GPU, _PHASE = (ROW_COLUMNS.index(c) for c in ("dtype_env_json", "gpu", "phase"))
#: The measurement columns the pinned query selects a row by and decides
#: whether it is shared with.
_NARROW = ("id, input_id, dtype_env_json, gpu, phase, batch_id, achieved_tflops, "
           "achieved_bw_gbps, peak_tflops_at_test, peak_bw_at_test")

_REJECTIONS_KEPT = 20


#: The most candidate rows one lookup answers. Past it the lookup is refused
#: (`TooManyRows`, a 503) before any wide column is read or any row
#: assembled — never truncated: a partial candidate set would change what
#: an estimate picks. At the live lake's density (~4 rows a key) a few
#: hundred keys are ~1.4k rows; at the 2.2M-row scale envelope's (~175) 300
#: keys are ~45k, and 3,000 keys were 526k rows, 2.3–2.6 GB in the service.
#: The lasting cure is fewer candidates per key (a candidate reducer owned
#: by the calibration layer and run by the service; not built —
#: docs/measured_calibration.md §7.2).
MAX_ROWS = 50_000


class TooManyRows(RuntimeError):
    """A lookup whose candidates exceed `MAX_ROWS` (the route answers 503)."""


class LookupStats:
    """What lookups have answered since the start, for `/health`: how many,
    keys and candidate rows asked, rows per key (mean, the largest one key
    had), the largest answer, and how many were refused (`TooManyRows`)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.lookups = self.keys = self.rows = self.refused = 0
        self.max_rows_per_key = self.max_rows = 0

    def record(self, keys: int, per_key: Sequence[int], refused: bool) -> None:
        rows = sum(per_key)
        with self._lock:
            self.lookups += 1
            self.keys += keys
            self.rows += rows
            self.refused += refused
            self.max_rows_per_key = max(self.max_rows_per_key, max(per_key, default=0))
            self.max_rows = max(self.max_rows, rows)
        (_LOG.warning if refused else _LOG.info)(
            "Kernel Bench lookup: %d keys, %d candidate rows (most %d for one key)%s",
            keys, rows, max(per_key, default=0), "; refused, over MAX_ROWS" if refused else "")

    def report(self) -> dict:
        with self._lock:
            return {"lookups": self.lookups, "keys": self.keys, "rows": self.rows,
                    "rows_per_key": round(self.rows / self.keys, 1) if self.keys else None,
                    "max_rows_per_key": self.max_rows_per_key, "max_rows": self.max_rows,
                    "refused": self.refused, "max_rows_allowed": MAX_ROWS}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


_CELLS = ", ".join(f"m.{_q(c)}" for c in _benchmarks.ALL_COLUMNS)


def _keys_sql() -> str:
    """The posted keys as a relation: each `WireKey` (every field text) with
    its position, `idx`."""
    types = {"idx": "INTEGER", **dict.fromkeys(WireKey.__annotations__, "VARCHAR")}
    return (f"(SELECT unnest(from_json(?::JSON, '{json.dumps([types])}'), "
            "recursive := true))")


_KEYS = _keys_sql()


def _cell(v: Any) -> str:
    """A stored value as the CSV cell the validator reads ("" for NULL)."""
    return "" if v is None else str(v)


class Verdicts:
    """Each row's cells and its verdict under this process's validator, by
    row id (the verdict None when the row passes `validate_row`), bounded.
    Rows are never edited and the validator changes only with the code, so
    an entry holds for the process's life; a lookup reads the lake's wide
    columns only for rows it does not hold (the narrow join that selects
    rows is most of a warm lookup). A row is held as the cells a lookup
    sends (`ROW_COLUMNS`, interned: a shape or a dtype env is one string
    however many rows repeat it) and its verdict. Lookup asks for cells with
    `validate=False`; owner reporting asks for the current verdict."""

    def __init__(self, max_entries: int = 120_000):
        self._max = max_entries
        self._memo: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, row_id: str) -> Optional[tuple]:
        """`(cells, verdict)` of a row held, else None."""
        with self._lock:
            got = self._memo.get(row_id)
            if got is not None:
                self._memo.move_to_end(row_id)
            return got

    def put(self, row_id: str, cells: Mapping[str, str], *,
            validate: bool = True) -> tuple:
        """Hold a row read from the lake (all its cells, `ALL_COLUMNS`): the
        cells a lookup sends, and its verdict when the caller asked for one."""
        why = None
        if validate:
            try:
                _benchmarks.validate_row(dict(cells))
            except _benchmarks.BenchmarkValidationError as e:
                why = str(e)
        entry = (tuple(sys.intern(cells[c]) for c in ROW_COLUMNS), why)
        with self._lock:
            self._memo[row_id] = entry
            while len(self._memo) > self._max:
                self._memo.popitem(last=False)
        return entry

    def of(self, row_id: str, cells: Mapping[str, str]) -> Optional[str]:
        """The row's verdict (held, or computed from `cells` and held)."""
        got = self.get(row_id)
        return (got if got is not None else self.put(row_id, cells))[1]


class _Gate:
    """`RowCheck` over a stream with remembered per-row verdicts: the row
    passes `validate_row` and its 12-column identity is new to the stream."""

    def __init__(self) -> None:
        self.seen: dict[tuple, str] = {}

    def reason(self, row_id: str, entry: tuple) -> Optional[str]:
        cells, why = entry
        if why is not None:
            return why
        # Identity AND outcome, as `RowCheck` and the upload's own match do:
        # a cell may carry a measurement and the record of having been
        # refused on this card, and keying on identity alone made the two
        # fight — whichever the row order happened to put first survived.
        outcome = dict(zip(COLUMNS, cells, strict=False)).get("outcome") or "measured"
        ident = (*cells[:_IDENTITY], outcome or "measured")
        first = self.seen.get(ident)
        if first is not None:
            return (f"duplicate identity tuple (first seen as row {first}); rows "
                    "whose 12 identity columns and `outcome` are all equal must "
                    "be deduplicated by the contributor.")
        self.seen[ident] = row_id
        return None


def _entries(lake: Lake, found: Sequence[Mapping[str, Any]], snapshot: int,
             verdicts: Verdicts) -> dict:
    """Every selected row as a held entry, reading the lake's wide columns
    (at `snapshot`) only for the rows not held. The read is narrowed by the
    rows' inputs as literals, which prune the lake's files by their input
    ranges; a row-id list does not (it scanned every wide column)."""
    out = {}
    missing: dict[str, int] = {}
    for r in found:
        row_id = r["id"]
        if row_id in out or row_id in missing:
            continue
        got = verdicts.get(row_id)
        if got is None:
            missing[row_id] = int(r["input_id"])
        else:
            out[row_id] = got
    if missing:
        inputs = ",".join(str(i) for i in sorted(set(missing.values())))
        for r in lake.read(
                f"SELECT {_CELLS}, m.id::VARCHAR AS id FROM lake.measurements m "
                f"AT (VERSION => ?) WHERE m.input_id IN ({inputs})", [snapshot]):
            if r["id"] in missing:
                out[r["id"]] = verdicts.put(
                    r["id"], {c: _cell(r[c]) for c in _benchmarks.ALL_COLUMNS},
                    validate=False)
    return out


class Selection:
    """A memory-budgeted copy of what a lookup selects from, for the newest
    snapshot asked.

    The table `sel_<n>_<k>` (the DuckDB instance's own in-memory database)
    holds, per measurement at snapshot `n` (`AT (VERSION => n)`), only what
    narrows it to a key: its input, its `env` (the hash of its dtype env and
    gpu the lake already keeps), its phase, its id, its batch and whether it
    is within its recorded peaks. The batch's status and owner are read
    pinned at query time (`batches` is small), and every row's dtype env,
    gpu and phase are checked as text against the key afterwards, so the
    hash narrows and text equality decides.

    * One table, for the newest snapshot a lookup asked: a lookup at an
      older pin, or a lake whose table would exceed `BUDGET_BYTES`, takes
      the exact pinned query over the lake's files instead (`None`).
    * Lifetime: a lookup holds the table while it reads it (`use`); a newer
      snapshot's table replaces it for new lookups, and the old one is
      dropped when its last reader lets go — never under a reader.
    * Cost (docs/measured_calibration.md §7.2): ~66 bytes a row; the local
      lake's 71k rows ~5 MB, built in ~15 ms when a write moves the
      snapshot. The 2.2M-row scale copy would take ~145 MB, over the
      budget: there every lookup is the pinned query, whose time is the
      rows it returns (at that scale ~280 per key), not the selection."""

    #: Bytes a held table may take (at ~66 bytes a row, ~500k rows).
    BUDGET_BYTES = 32 * 2**20
    ROW_BYTES = 66

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._built = 0
        # per lake: [snapshot, name, readers] of the current table, and the
        # retired ones still read: {name: readers}.
        self._current: "weakref.WeakKeyDictionary[Lake, list]" = weakref.WeakKeyDictionary()
        self._retired: "weakref.WeakKeyDictionary[Lake, dict]" = weakref.WeakKeyDictionary()

    def _build(self, lake: Lake, snapshot: int) -> Optional[str]:
        rows = lake.read("SELECT count(*) AS n FROM lake.measurements AT (VERSION => ?)",
                         [snapshot])[0]["n"]
        if rows * self.ROW_BYTES > self.BUDGET_BYTES:
            return None
        self._built += 1
        name = f"sel_{int(snapshot)}_{self._built}"
        lake.transact(lambda cur: cur.execute(
            f"CREATE TABLE {name} AS SELECT m.input_id, m.env, m.phase, m.id, m.batch_id, "
            f"{WITHIN_PEAK} AS within_peak FROM lake.measurements m AT (VERSION => ?)",
            [snapshot]))
        return name

    def _drop(self, lake: Lake, name: str) -> None:
        lake.transact(lambda cur: cur.execute(f"DROP TABLE IF EXISTS {name}"))

    @contextmanager
    def use(self, lake: Lake, snapshot: int) -> Iterator[Optional[str]]:
        """The held table of `snapshot` for the block (built when it is newer
        than the held one), or None: take the pinned query."""
        name = None
        with self._lock:
            cur = self._current.get(lake)
            if cur is None or snapshot > cur[0]:
                built = self._build(lake, snapshot)
                if cur is not None:
                    if cur[2]:
                        self._retired.setdefault(lake, {})[cur[1]] = cur[2]
                    elif cur[1] is not None:
                        self._drop(lake, cur[1])
                cur = self._current[lake] = [snapshot, built, 0]
            if cur[0] == snapshot and cur[1] is not None:
                name = cur[1]
                cur[2] += 1
        try:
            yield name
        finally:
            if name is not None:
                with self._lock:
                    cur = self._current.get(lake)
                    if cur is not None and cur[1] == name:
                        cur[2] -= 1
                    else:
                        retired = self._retired.get(lake, {})
                        retired[name] -= 1
                        if not retired[name]:
                            del retired[name]
                            self._drop(lake, name)


def _select(lake: Lake, records: list, owner: Optional[str], snapshot: int,
            table: Optional[str]) -> list[dict]:
    """`(idx, input_id, id, shared, mine)` of every row a key selects at `snapshot`:
    from the held table, or by the pinned query."""
    shared = "(b.status = 'shared' AND (NOT b.within_peak OR {within}))"
    mine = "coalesce(b.owner = ?::UUID, false)"
    if table is not None:
        sh = shared.format(within="s.within_peak")
        return lake.read(
            "WITH ki AS (SELECT k.idx, k.phase, a.input_id, "
            "md5_number_upper(k.dtype_env_json || chr(31) || k.gpu) AS env "
            f"FROM {_KEYS} k JOIN annot a ON a.signature = k.signature "
            "AND a.physical_shape = k.physical_shape) "
            f"SELECT ki.idx, ki.input_id, s.id::VARCHAR AS id, {sh} AS shared, {mine} AS mine "
            f"FROM ki JOIN {table} s ON s.input_id = ki.input_id AND s.env = ki.env "
            "AND s.phase = ki.phase "
            "JOIN lake.batches b AT (VERSION => ?) ON b.batch_id = s.batch_id "
            f"WHERE {sh} OR {mine} ORDER BY ki.idx, s.id",
            [json.dumps(records), owner, snapshot, owner])
    # The keys' inputs first (`annot`, in memory), then only those inputs'
    # measurements, narrow: left to the planner, the join scanned them all.
    return lake.read(
        "WITH ki AS MATERIALIZED (SELECT k.idx, k.dtype_env_json, k.gpu, k.phase, a.input_id "
        f"FROM {_KEYS} k JOIN annot a ON a.signature = k.signature "
        "AND a.physical_shape = k.physical_shape), "
        f"m AS MATERIALIZED (SELECT {_NARROW} FROM lake.measurements AT (VERSION => ?) "
        "WHERE input_id IN (SELECT input_id FROM ki)) "
        f"SELECT ki.idx, ki.input_id, m.id::VARCHAR AS id, {SHARED_ROW} AS shared, {mine} AS mine "
        "FROM ki JOIN m ON m.input_id = ki.input_id AND m.dtype_env_json = ki.dtype_env_json "
        "AND m.gpu = ki.gpu AND m.phase = ki.phase "
        "JOIN lake.batches b AT (VERSION => ?) ON b.batch_id = m.batch_id "
        f"WHERE {SHARED_ROW} OR {mine} ORDER BY ki.idx, m.id",
        [json.dumps(records), snapshot, owner, snapshot, owner])


def lookup(lake: Lake, keys: Sequence[Mapping[str, str]], *, owner: Optional[str],
           snapshot: int, verdicts: Verdicts, selection: Selection,
           stats: Optional[LookupStats] = None) -> dict:
    """The candidate rows of each key at `snapshot` (module docstring):
    `{columns, values, rows: [[[value index…] per row] per key],
    excluded_mine}` — a row's cells are indices into `values`, the distinct
    cell texts of the answer (a shape, a dtype env, a framework repeats
    across rows; each is sent once)."""
    if len(keys) > MAX_KEYS:
        raise ValueError(f"a lookup names at most {MAX_KEYS} keys, got {len(keys)}")
    records = [{"idx": i, **{f: str(k[f]) for f in WireKey.__annotations__}}
               for i, k in enumerate(keys)]
    for r in records:
        r["dtype_env_json"] = canonical_json(r["dtype_env_json"])
    out: list[list[list[int]]] = [[] for _ in keys]
    if not records:
        return {"columns": list(COLUMNS), "values": [], "rows": out, "excluded_mine": 0}
    # Which rows: ids and origin (`Selection`, or the pinned query) — the
    # wide columns come from the held rows (`Verdicts`), read from the lake
    # only for rows not held yet.
    with selection.use(lake, snapshot) as table:
        found = _select(lake, records, owner, snapshot, table)
    per_key = [0] * len(keys)
    for r in found:
        per_key[r["idx"]] += 1
    refused = len(found) > MAX_ROWS
    if stats is not None:
        stats.record(len(keys), per_key, refused)
    if refused:
        raise TooManyRows(
            f"{len(found)} candidate rows for {len(keys)} keys, over the lookup's "
            f"bound of {MAX_ROWS}: ask fewer keys")
    entries = _entries(lake, found, snapshot, verdicts)
    shared_rows: list[list[list[int]]] = [[] for _ in keys]
    mine_rows: list[list[list[int]]] = [[] for _ in keys]
    excluded = 0
    values: dict[str, int] = {}

    def cells_of(entry: tuple, origin: str) -> list[int]:
        return [values.setdefault(v, len(values)) for v in (*entry[0], origin)]
    for r in found:
        entry = entries[r["id"]]
        key = records[r["idx"]]
        # The text equality the hash narrowed to (the row's own cells).
        if (entry[0][_DTYPE], entry[0][_GPU], entry[0][_PHASE]) != (
                key["dtype_env_json"], key["gpu"], key["phase"]):
            continue
        if r["shared"]:
            shared_rows[r["idx"]].append(cells_of(entry, "shared"))
        if r["mine"]:
            mine_rows[r["idx"]].append(cells_of(entry, "mine"))
    for i in range(len(keys)):
        out[i] = shared_rows[i] + mine_rows[i]
    return {"columns": list(COLUMNS), "values": list(values), "rows": out,
            "excluded_mine": excluded}


def owner_report(lake: Lake, owner: str, *, snapshot: int, verdicts: Verdicts) -> dict:
    """`/api/bench/calibration`: of the owner's rows at `snapshot`, how many
    calibrate their estimates (`rows_used`), how many are valid but record
    an outcome rather than a measurement (`rows_not_measured`: an OOM or a
    failed attempt — the lookup sends them, the estimate's registry skips
    them, `op_calibration._index_row`), and how many the current validator
    refuses, why (the first `_REJECTIONS_KEPT` rows and distinct reasons) —
    the same gate the lookup applies to them."""
    gate = _Gate()
    outcome_at = ROW_COLUMNS.index("outcome")
    used = not_measured = rejected = 0
    first: list[dict] = []
    reasons: dict[str, int] = {}
    for r in lake.read(
            f"SELECT {_CELLS}, m.id::VARCHAR AS id FROM lake.measurements m "
            "AT (VERSION => ?) JOIN lake.batches b AT (VERSION => ?) "
            "ON b.batch_id = m.batch_id WHERE b.owner = ?::UUID ORDER BY m.id",
            [snapshot, snapshot, owner]):
        entry = verdicts.get(r["id"]) or verdicts.put(
            r["id"], {c: _cell(r[c]) for c in _benchmarks.ALL_COLUMNS})
        why = gate.reason(r["id"], entry)
        if why is None:
            if _benchmarks.is_measurement({"outcome": entry[0][outcome_at]}):
                used += 1
            else:
                not_measured += 1
            continue
        rejected += 1
        if len(first) < _REJECTIONS_KEPT:
            first.append({"id": r["id"], "reason": why})
        if why in reasons or len(reasons) < _REJECTIONS_KEPT:
            reasons[why] = reasons.get(why, 0) + 1
    return {"rows_used": used, "rows_not_measured": not_measured,
            "rows_rejected": rejected, "first_rejected": first,
            "rejected_reasons": [{"reason": k, "rows": n} for k, n in reasons.items()]}


#: A snapshot: its id and the identity of the lake's content at it
#: (`Lake.content_id`).
Snapshot = tuple[int, str]


def snapshot_now(lake: Lake) -> Snapshot:
    """The lake's current snapshot and its content identity. The identity
    names the content, not only the number: two copies of a lake (a restore,
    a copy to another machine or beside itself) share their history's ids,
    and once they diverge the same id is different content."""
    snapshot = lake.snapshot_id()
    return snapshot, lake.content_id(snapshot)


class SharedRevision:
    """The snapshot the shared rows last changed at.

    Every write runs inside `writing()`; one that changed the shared rows
    calls `changed()` before it leaves. `current(lake)` answers `(snapshot,
    shared_snapshot)`, each `(id, content identity)`: the lake's snapshot now,
    and a snapshot at or before it whose shared rows are the shared rows now
    — the last recorded change when no write is running, else the snapshot
    itself (a write in flight may have committed a shared change not
    recorded yet). Forgotten at a restart: the first answer after one is
    the snapshot then."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = 0
        self._rev: Optional[Snapshot] = None

    @contextmanager
    def writing(self) -> Iterator[None]:
        with self._lock:
            self._pending += 1
        try:
            yield
        finally:
            with self._lock:
                self._pending -= 1

    def changed(self, lake: Lake) -> None:
        """The write running now changed the shared rows (it has committed)."""
        snap = snapshot_now(lake)
        with self._lock:
            self._rev = snap if self._rev is None else max(self._rev, snap)

    def current(self, lake: Lake) -> tuple[Snapshot, Snapshot]:
        snap = snapshot_now(lake)
        with self._lock:
            if self._rev is None:
                self._rev = snap
            if self._pending:
                return snap, snap
            return snap, min(self._rev, snap)
