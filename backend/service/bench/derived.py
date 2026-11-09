"""The lake's derived file: everything derived from one lake snapshot that a
read serves from (docs/measured_calibration.md §7.2).

One DuckDB file in the lake's local cache directory (`path`:
BENCH_LAKE_CACHE, else `<lake dir>/tmp`, root-only), attached to the lake's
instance (its memory limit, threads, spill) as `store.DERIVED`; DuckDB locks
it against a second process. It holds:

  rows         the lookup's cells (`CELL_COLUMNS`) of every measurement,
               each version with the snapshots it is the row at —
               `begin_snapshot` inclusive, `end_snapshot` exclusive (NULL:
               still the row) — as DuckLake keeps the rows themselves, so a
               lookup pinned at any snapshot from `base` reads it exactly
  annot        one entry per input: its annotations under the annotation
               rules (`annotate`; a placeholder's are NULL) and `phys`
               (`store.ANNOT`; every reader joins it)
  batches      the lake's batches (`browse.BATCHES`)
  browse_rows  the browse row of every measurement (`browse.ROWS`,
               `browse.rows_select`)
  browse_groups the browse rows counted by batch and filter columns
               (`browse.GROUPS`, `browse.groups_select`), what facets and
               coverage read
  placement    each physical shape's place on a curve (`browse.PLACEMENT`)
  meta         `oldest` and `newest` (base and at: the snapshots it
               answers), the content identity of `at` (`Lake.content_id`),
               `columns` (the form of every table above, `Derived.form`) and
               `rules` (the annotation rules' revision it was derived under)

annot, batches, browse_rows, browse_groups and placement are the lake at `at`; the rows'
versions answer every pin from `base` to `at`.

Why. Every one of these is a function of one snapshot (and the annotation
rules), and computing it per request or per process start was the cost:
a cold lookup's cells read every data file's wide columns; the browse views
read the lake's files and joined two million annotations on every request
(envelope 2.4 s, curves 11 s on 4.28M rows: the page's requests crossed the
API's 30 s read limit when fired together); and the annotations were
recomputed for every input at every start (85 s for 2M inputs), with the
service unavailable throughout.

One builder (`refresh`), the one writer of the file, the only reader of the
lake on behalf of a derived table:

  * the file is this lake's at `at` (its `columns` and `rules` are this
    code's; `at` is the snapshot this process last brought it to, or its
    content identity still matches the lake's at `at`): nothing to do when
    `at` is the lake's snapshot; otherwise the lake's change feed
    (`ducklake_table_changes`) from `at` is applied — changed measurements'
    versions closed and opened at the snapshots their changes committed,
    their browse rows replaced, new inputs annotated, the batches re-read;
  * any other file — none yet, from before a form change, under other
    annotation rules, a restored or diverged copy — is rebuilt at the lake's
    snapshot, and the reason logged.

After either, the file's tables are checked against the lake at `at` (a row
count each; `Inconsistent` when one differs — never served), versions
ended at or before the lake's oldest snapshot are pruned, and `meta`
records what the file answers. The whole refresh is one transaction: a
reader sees the file before it or after it, never between.

Reads. A read first asks the file to reach a snapshot (`reach`): a lookup
its pin, a browse request the lake's snapshot when it starts. Behind it,
the read wakes the follower thread and waits for it, at most `CATCH_UP_S`;
past that it fails (`LakeUnavailable`, a 503) — it never reads the lake's
files instead. A pin before `base` is refused the same way. The service
builds the file before it serves (`serving.Gate`'s `ready`), so `base` is at
most the snapshot it started at and every pin a client got from it since is
answerable.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import threading
import time
import weakref
from typing import Any, Callable, Mapping, Optional, Sequence

from backend.inference.op_calibration import ROW_COLUMNS
from backend.measure.lookup import TIMEOUT_S as _CLIENT_TIMEOUT_S
from backend.service.bench import browse as _browse
from backend.service.bench import store
from backend.service.bench.lake import Lake, LakeUnavailable

_LOG = logging.getLogger(__name__)

#: The cells a lookup sends of a row (`op_calibration.ROW_COLUMNS`).
CELL_COLUMNS = ROW_COLUMNS


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


#: The cells, over the lake's measurement alias m.
CELLS_SQL = ", ".join(f"m.{_q(c)}" for c in CELL_COLUMNS)
_CELLS = ", ".join(_q(c) for c in CELL_COLUMNS)

A = store.DERIVED
ROWS, ANNOT = f"{A}.rows", store.ANNOT
BATCHES, BROWSE, PLACEMENT = _browse.BATCHES, _browse.ROWS, _browse.PLACEMENT
GROUPS = _browse.GROUPS
#: Inputs annotated, and placement entries written, per statement (one JSON
#: text DuckDB parses whole: its memory grows with the batch, not the lake).
BATCH = 50_000


class Inconsistent(RuntimeError):
    """A refreshed table does not hold what the lake holds at its snapshot."""


class Derived:
    """The lake's derived file (module docstring): its one builder
    (`refresh`), the follower thread that keeps it at the lake's snapshot
    (`follow`, `poke`), and the wait a read takes for it (`reach`).

    `annotate`: an input's annotations (`store.Annotate`; the routes pass
    `annotations.row_annotations`). `rules`: the revision of the rules
    `annotate` and the curves' placement follow (the routes pass
    `rules_revision.rules_rev`). `tokens`: what a shape is placed on a
    curve by (`browse.call_tokens`)."""

    #: The longest a read waits for the file to reach its snapshot: the
    #: lookup client's own floor (`measure.lookup.TIMEOUT_S`) — a lookup
    #: that waits longer has already timed out at the client.
    CATCH_UP_S = _CLIENT_TIMEOUT_S

    def __init__(self, *, annotate: store.Annotate, rules: Callable[[], str],
                 tokens: Callable[[], dict] = _browse.call_tokens) -> None:
        self._annotate, self._rules, self._tokens = annotate, rules, tokens
        self._form: Optional[str] = None
        self._lock = threading.Lock()                 # one builder at a time
        self._moved = threading.Condition()
        # per lake: (base, at) of the file as this process last left it.
        self._held: "weakref.WeakKeyDictionary[Lake, tuple[int, int]]" = \
            weakref.WeakKeyDictionary()
        self._wake: "weakref.WeakKeyDictionary[Lake, threading.Event]" = \
            weakref.WeakKeyDictionary()

    def form(self) -> str:
        """The form of every derived table, and of what the tables not
        covered by the annotation rules (`rules`) are computed from: the
        lookup's cells, the annotations' columns, the browse row's
        quantities and their SQL, the groups — and the placement, a Python
        function (`browse._Placement`, its source hashed as `rules_rev`
        hashes its owners') of the declared token axes (`tokens`, the
        `call_tokens()` value). A file of another form is rebuilt; computed
        once per process (the code and the declarations do not change
        under it)."""
        if self._form is None:
            self._form = hashlib.sha256(json.dumps({
                "cells": list(CELL_COLUMNS), "annot": store.ANNOT_COLUMNS,
                "browse": _browse.rows_select("m"), "groups": _browse.groups_select(),
                "placement": [_browse.PLACEMENT_TYPES, inspect.getsource(_browse._Placement),
                              self._tokens()],
            }, sort_keys=True).encode()).hexdigest()[:16]
        return self._form

    @staticmethod
    def path(lake: Lake) -> str:
        return os.path.join(lake.config.spill_dir, "derived.duckdb")

    def _attach(self, lake: Lake) -> None:
        path = self.path(lake).replace("'", "''")

        def attach(cur: Any) -> None:
            cur.execute(f"ATTACH IF NOT EXISTS '{path}' AS {A}")
            cur.execute(f"CREATE TABLE IF NOT EXISTS {A}.meta (oldest BIGINT, newest BIGINT, "
                        "content VARCHAR, columns VARCHAR, rules VARCHAR)")
        lake.transact(attach)

    # ---- reading

    def held(self, lake: Lake) -> Optional[tuple[int, int]]:
        """(base, at) of the file as this process last left it."""
        with self._moved:
            return self._held.get(lake)

    def reach(self, lake: Lake, snapshot: int) -> None:
        """Return once the file answers `snapshot`; wake the follower and wait
        (`CATCH_UP_S`) when it is behind; refuse a pin before its base."""
        deadline = time.monotonic() + self.CATCH_UP_S
        self.follow(lake)
        with self._moved:
            while True:
                held = self._held.get(lake)
                if held is not None and held[0] <= snapshot <= held[1]:
                    return
                if held is not None and snapshot < held[0]:
                    raise LakeUnavailable(
                        f"snapshot {snapshot} is older than the derived file holds "
                        f"(from snapshot {held[0]})")
                self.poke(lake)
                left = deadline - time.monotonic()
                if left <= 0:
                    raise LakeUnavailable(
                        f"the derived file did not reach snapshot {snapshot} within "
                        f"{self.CATCH_UP_S:.0f} s (it holds "
                        f"{'nothing yet' if held is None else held[1]})")
                self._moved.wait(left)

    def cells(self, lake: Lake, ids: Sequence[str], snapshot: int) -> dict[str, tuple]:
        """The cells of every one of `ids` at `snapshot`, each a tuple in
        `CELL_COLUMNS` order (a NULL as ""); `LakeUnavailable` when the file
        cannot answer `snapshot` in time, or lacks one of the rows."""
        if not ids:
            return {}
        self.reach(lake, snapshot)
        # Ids are the lake's own UUID text; literals let the scan filter by
        # them (a joined list is planned as a hash join over the whole
        # table, several times slower). The meta check is in the statement,
        # so a refresh committing meanwhile is seen whole or not at all.
        rows = lake.read(
            f"SELECT r.id::VARCHAR AS id, {', '.join(f'r.{_q(c)}' for c in CELL_COLUMNS)} "
            f"FROM {ROWS} r, {A}.meta m WHERE m.oldest <= ? AND m.newest >= ? "
            "AND m.columns = ? AND r.begin_snapshot <= ? "
            "AND (r.end_snapshot IS NULL OR r.end_snapshot > ?) "
            f"AND r.id IN ({','.join(repr(str(i)) for i in ids)})",
            [snapshot, snapshot, self.form(), snapshot, snapshot])
        out = {r["id"]: tuple("" if r[c] is None else str(r[c]) for c in CELL_COLUMNS)
               for r in rows}
        if len(out) < len(set(ids)):
            raise LakeUnavailable(
                f"the derived file's cells at snapshot {snapshot} lack "
                f"{len(set(ids)) - len(out)} selected rows")
        return out

    # ---- following the lake

    def poke(self, lake: Lake) -> None:
        """Wake the follower (`follow`); nothing when none follows the lake."""
        wake = self._wake.get(lake)
        if wake is not None:
            wake.set()

    def follow(self, lake: Lake) -> None:
        """Keep the file at the lake's snapshot from a daemon thread: one
        refresh per poke (pokes during a refresh coalesce); a failed refresh
        is logged and tried again on the next poke."""
        with self._moved:
            if lake in self._wake:
                return
            wake = self._wake[lake] = threading.Event()
        ref = weakref.ref(lake)

        def run() -> None:
            while wake.wait():
                wake.clear()
                lk = ref()
                if lk is None:
                    return
                try:
                    self.refresh(lk)
                except Exception:
                    _LOG.exception("Kernel Bench derived file: refresh failed; reads "
                                   "behind it fail until the next one")
                del lk
        threading.Thread(target=run, name="bench-derived", daemon=True).start()

    # ---- the builder

    def refresh(self, lake: Lake) -> dict:
        """Bring the file to the lake's current snapshot: `{snapshot, how,
        rows}` — how `current` (nothing to do), `changes` (the change feed
        since `at` applied; `rows`: changed measurements) or `rebuilt`
        (`rows`: measurements)."""
        with self._lock:
            self._attach(lake)
            target = lake.snapshot_id()
            content = lake.content_id(target)
            rules = self._rules()
            meta = lake.read(f"SELECT oldest AS base, newest AS at, content, columns, rules "
                             f"FROM {A}.meta")
            held = meta[0] if meta else None
            oldest = int(lake.read("SELECT min(snapshot_id) AS n FROM lake.snapshots()")[0]["n"])
            why = self._why_rebuild(lake, held, target, rules, oldest)
            t0 = time.monotonic()
            if why is None and held["at"] == target:
                how, rows, base = "current", 0, held["base"]
            else:
                lake.require_free("refreshing the derived file")
                if why is None:
                    how, base = "changes", held["base"]
                    step = lambda cur: self._apply_changes(lake, cur, held["at"], target)
                else:
                    _LOG.warning("Kernel Bench derived file: rebuilding at snapshot %d: %s",
                                 target, why)
                    how, base = "rebuilt", target
                    step = lambda cur: self._rebuild(lake, cur, target)
                base = max(base, min(oldest, target))

                def write(cur: Any) -> int:
                    n = step(cur)
                    self._check(cur, target)
                    self._settle(cur, base, target, content, rules)
                    return n
                rows = lake.transact(write)
            with self._moved:
                self._held[lake] = (base, target)
                self._moved.notify_all()
            if how != "current":
                _LOG.info("Kernel Bench derived file at snapshot %d (from %d): %s "
                          "(%d rows) in %.1f s", target, base, how, rows,
                          time.monotonic() - t0)
            return {"snapshot": target, "how": how, "rows": rows}

    def _why_rebuild(self, lake: Lake, held: Optional[Mapping[str, Any]], target: int,
                     rules: str, oldest: int) -> Optional[str]:
        """Why the file cannot be followed from where it is, or None."""
        if held is None:
            return "no derived file yet"
        if held["columns"] != self.form():
            return "the derived tables' form changed"
        if held["rules"] != rules:
            return f"the annotation rules changed ({held['rules']} -> {rules})"
        if held["at"] > target:
            return f"the file is at snapshot {held['at']}, past the lake's {target}"
        if held["at"] < oldest:
            return (f"the file is at snapshot {held['at']}, which the lake no longer keeps "
                    f"(its oldest is {oldest}): no change feed reaches it")
        if (self._held.get(lake, (None, None))[1] != held["at"]
                and held["content"] != lake.content_id(held["at"])):
            return f"the file is of another history than this lake's at snapshot {held['at']}"
        return None

    def _rebuild(self, lake: Lake, cur: Any, snapshot: int) -> int:
        cur.execute(f"CREATE OR REPLACE TABLE {ROWS} AS SELECT m.id, "
                    "?::BIGINT AS begin_snapshot, NULL::BIGINT AS end_snapshot, "
                    f"{CELLS_SQL} FROM lake.measurements m AT (VERSION => ?)",
                    [snapshot, snapshot])
        cur.execute(f"CREATE OR REPLACE TABLE {ANNOT} ({store.ANNOT_COLUMNS})")
        cur.execute(f"CREATE OR REPLACE TABLE {PLACEMENT} ("
                    + ", ".join(f"{c} {t}" for c, t in _browse.PLACEMENT_TYPES.items()) + ")")
        self._annotate_new(lake, cur, f"SELECT * FROM lake.inputs AT (VERSION => {int(snapshot)})")
        self._batches(cur, snapshot)
        self._place_new(cur)
        cur.execute(f"CREATE OR REPLACE TABLE {BROWSE} AS "
                    + _browse.rows_select(f"(SELECT * FROM lake.measurements AT (VERSION => {int(snapshot)}))")
                    + " ORDER BY envelope")
        self._groups(cur)
        return cur.execute(f"SELECT count(*) FROM {ROWS}").fetchone()[0]

    def _apply_changes(self, lake: Lake, cur: Any, at: int, target: int) -> int:
        """The change feed (at, target] applied. Cells: per changed id, its
        state at each snapshot that changed it — present with its cells
        after an insert or an update's post-image, absent after a delete —
        closes the version open before and opens one until the id's next
        change. Annotations: the inputs inserted annotated, those deleted
        dropped. Browse rows: each changed id's replaced by its row at
        `target`, if it has one."""
        cur.execute(
            f"CREATE OR REPLACE TABLE {A}.changes AS SELECT m.id, m.snapshot_id AS s, "
            "m.change_type IN ('insert', 'update_postimage') AS present, "
            f"{CELLS_SQL} FROM ducklake_table_changes('lake', 'main', 'measurements', ?, ?) m "
            "QUALIFY row_number() OVER (PARTITION BY m.id, m.snapshot_id "
            "ORDER BY present DESC) = 1", [at + 1, target])
        cur.execute(
            f"UPDATE {ROWS} AS r SET end_snapshot = c.first FROM (SELECT id, "
            f"min(s) AS first FROM {A}.changes GROUP BY id) c "
            "WHERE r.id = c.id AND r.end_snapshot IS NULL")
        cur.execute(
            f"INSERT INTO {ROWS} SELECT id, s, e, {_CELLS} FROM (SELECT *, "
            f"lead(s) OVER (PARTITION BY id ORDER BY s) AS e FROM {A}.changes) "
            "WHERE present")
        inputs = (f"FROM ducklake_table_changes('lake', 'main', 'inputs', {int(at) + 1}, "
                  f"{int(target)})")
        cur.execute(f"DELETE FROM {ANNOT} WHERE input_id IN (SELECT input_id {inputs} "
                    "WHERE change_type = 'delete')")
        self._annotate_new(lake, cur, f"SELECT * {inputs} WHERE change_type = 'insert'")
        self._batches(cur, target)
        self._place_new(cur)
        cur.execute(f'DELETE FROM {BROWSE} WHERE "id" IN (SELECT id FROM {A}.changes)')
        cur.execute(f"INSERT INTO {BROWSE} " + _browse.rows_select(
            f"(SELECT * FROM lake.measurements AT (VERSION => {int(target)}) "
            f"WHERE id IN (SELECT id FROM {A}.changes))"))
        self._groups(cur)
        n = cur.execute(f"SELECT count(DISTINCT id) FROM {A}.changes").fetchone()[0]
        cur.execute(f"DROP TABLE {A}.changes")
        return n

    @staticmethod
    def _groups(cur: Any) -> None:
        cur.execute(f"CREATE OR REPLACE TABLE {GROUPS} AS {_browse.groups_select()}")

    def _batches(self, cur: Any, snapshot: int) -> None:
        cur.execute(f"CREATE OR REPLACE TABLE {BATCHES} AS SELECT * FROM lake.batches "
                    "AT (VERSION => ?)", [snapshot])

    def _annotate_new(self, lake: Lake, cur: Any, inputs: str) -> None:
        """Annotate the inputs `inputs` selects that `annot` lacks, in
        annotation order (`op`, `shape_json`: the routes' annotator memoizes
        the physical half while duplicates are adjacent), `BATCH` at a time.
        An input `annotate` refuses gets an entry with every annotation NULL
        (a placeholder); a catalog that cannot load (`ImportError`) fails
        the refresh."""
        names = list(store.ANNOT_ENTRY)
        keys = ", ".join(f"i.{k}" for k in store.INPUT_KEY)
        pending = self._pending(cur, f"SELECT i.input_id, {keys} FROM ({inputs}) i "
                                f"ANTI JOIN {ANNOT} a USING (input_id)",
                                "op, shape_json, phase, kernel_impl")
        refused: list = []
        for chunk in pending:
            entries = [{"input_id": input_id, **self._annotation(
                dict(zip(store.INPUT_KEY, k, strict=True)), refused)} for input_id, *k in chunk]
            cur.execute(f"INSERT INTO {ANNOT} SELECT {', '.join(names)}, {store.PHYS} FROM "
                        f"{store.records(store.ANNOT_ENTRY)}", [json.dumps(entries)])
        if refused:
            k, e = refused[0]
            _LOG.warning("bench: %d inputs cannot be annotated (placeholders), e.g. %s: %s: %s",
                         len(refused), k, type(e).__name__, e)

    def _annotation(self, key: Mapping[str, Any], refused: list) -> dict:
        try:
            ann = self._annotate(key)
        except ImportError:
            raise
        except Exception as e:
            refused.append((key, e))
            return dict.fromkeys(store.ANNOTATIONS)
        return {a: ann[a] for a in store.ANNOTATIONS}

    def _place_new(self, cur: Any) -> None:
        """Place every physical shape of `annot` the placement lacks."""
        tokens = self._tokens()
        place = _browse._Placement(tokens["runs"], list(tokens["total"]), list(tokens["factor"]))
        for chunk in self._pending(
                cur, f"SELECT DISTINCT a.phys, a.signature, a.physical_shape FROM {ANNOT} a "
                f"WHERE a.phys IS NOT NULL AND a.phys NOT IN (SELECT phys FROM {PLACEMENT})",
                "phys"):
            cur.execute(f"INSERT INTO {PLACEMENT} SELECT {', '.join(_browse.PLACEMENT_TYPES)} "
                        f"FROM {store.records(_browse.PLACEMENT_TYPES)}",
                        [json.dumps([place.entry(*r) for r in chunk])])

    @staticmethod
    def _pending(cur: Any, select: str, order: str):
        """The rows of `select` in `order`, `BATCH` at a time — numbered into
        a work table first, since the insert each batch feeds runs on the
        same cursor (one transaction) as the read."""
        cur.execute(f"CREATE OR REPLACE TABLE {A}.pending AS SELECT row_number() OVER "
                    f"(ORDER BY {order}) AS n, * FROM ({select}) ORDER BY n")
        total = cur.execute(f"SELECT count(*) FROM {A}.pending").fetchone()[0]
        for lo in range(0, total, BATCH):
            yield [r[1:] for r in cur.execute(
                f"SELECT * FROM {A}.pending WHERE n > ? AND n <= ? ORDER BY n",
                [lo, lo + BATCH]).fetchall()]
        cur.execute(f"DROP TABLE {A}.pending")

    def _check(self, cur: Any, at: int) -> None:
        """Each refreshed table against the lake at `at`: as many entries as
        the lake has inputs, measurements, batches; a version of every
        measurement live at `at`; a placement of every physical shape."""
        n = int(at)
        want = {t: cur.execute(f"SELECT count(*) FROM lake.{t} AT (VERSION => {n})").fetchone()[0]
                for t in ("inputs", "measurements", "batches")}
        got = {
            "annot": (cur.execute(f"SELECT count(*), count(DISTINCT input_id) FROM {ANNOT}").fetchone(),
                      (want["inputs"],) * 2),
            "browse_rows": (cur.execute(f'SELECT count(*), count(DISTINCT "id") FROM {BROWSE}').fetchone(),
                            (want["measurements"],) * 2),
            "browse_groups": (cur.execute(f"SELECT sum(n) FROM {GROUPS}").fetchone(),
                              (want["measurements"] or None,)),
            "batches": (cur.execute(f"SELECT count(*) FROM {BATCHES}").fetchone(), (want["batches"],)),
            "rows": (cur.execute(f"SELECT count(*), count(DISTINCT id) FROM {ROWS} WHERE "
                                 f"begin_snapshot <= {n} AND (end_snapshot IS NULL OR "
                                 f"end_snapshot > {n})").fetchone(), (want["measurements"],) * 2),
            "placement": (cur.execute(f"SELECT count(DISTINCT a.phys) FILTER (WHERE p.phys IS NULL) "
                                      f"FROM {ANNOT} a LEFT JOIN {PLACEMENT} p USING (phys) "
                                      "WHERE a.phys IS NOT NULL").fetchone(), (0,)),
        }
        bad = {t: (tuple(g), w) for t, (g, w) in got.items() if tuple(g) != w}
        if bad:
            raise Inconsistent(f"the derived file at snapshot {n} does not hold the lake: "
                               + "; ".join(f"{t} has {g}, the lake {w}" for t, (g, w) in bad.items()))

    def _settle(self, cur: Any, base: int, at: int, content: str, rules: str) -> None:
        """Prune the versions no pin from `base` reads; record what the file
        answers."""
        cur.execute(f"DELETE FROM {ROWS} WHERE end_snapshot <= ?", [base])
        cur.execute(f"DELETE FROM {A}.meta")
        cur.execute(f"INSERT INTO {A}.meta VALUES (?, ?, ?, ?, ?)",
                    [base, at, content, self.form(), rules])

