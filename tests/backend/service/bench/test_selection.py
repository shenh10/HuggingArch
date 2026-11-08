"""The lookup's memory-budgeted selection table (`lookup.Selection`,
docs/measured_calibration.md §7.2): one table, for the newest snapshot
asked; an older pin or a lake over its configured selection limit
(BENCH_LAKE_SELECTION_LIMIT) takes the pinned query, with the same answer; a
table is never dropped under a lookup reading it."""
from __future__ import annotations

import copy
import threading
import uuid

import pytest

from backend.service.bench import lookup as LK
from backend.service.bench import store
from backend.service.bench.lake import Lake, LakeConfig, LakeUnavailable


def _upload(lake, annotate, owner, rows, *, status="private"):
    store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()), run_id=None, label=None,
                 rows=rows, annotate=annotate, status=status,
                 decided_by=owner if status == "shared" else None)
    return lake.snapshot_id()


def _key(annotate, row) -> dict:
    ann = annotate(row)
    return {"signature": ann["signature"], "physical_shape": ann["physical_shape"],
            "dtype_env_json": row["dtype_env_json"], "gpu": row["gpu"], "phase": row["phase"]}


def _tables(lake) -> set:
    return {r["name"] for r in lake.read(
        "SELECT table_name AS name FROM duckdb_tables() WHERE table_name LIKE 'sel_%'")}


def _limited(lake, selection_bytes: int):
    """Another handle on the same lake whose selection limit is
    `selection_bytes` (as a lake opened with that BENCH_LAKE_SELECTION_LIMIT)."""
    handle = copy.copy(lake)
    handle.selection_bytes = selection_bytes
    return handle


def _open(root, **over) -> Lake:
    lk = Lake.open(LakeConfig(catalog=str(root / "catalog.ducklake"),
                              data=str(root / "data"), **over))
    store.ensure_schema(lk)
    return lk


def test_the_held_tables_budget_is_the_lakes_configured_limit(tmp_path, annotate, make_rows,
                                                               new_owner, monkeypatch):
    """A table over the old fixed 32 MiB budget is held when the lake's
    configured limit holds it, and one within 32 MiB is not when the
    configured limit does not: the budget is the lake's, not the module's."""
    rows = make_rows(3)
    # Each row priced at 16 MiB: the three rows' table (48 MiB) is over
    # the old fixed budget and within a 64 MiB configured limit.
    monkeypatch.setattr(LK.Selection, "ROW_BYTES", 16 * 2**20)
    for limit, held in (("64MiB", True), ("32MiB", False)):
        lk = _open(tmp_path / limit, selection_limit=limit)
        try:
            store.annotate_all(lk, annotate)
            snap = _upload(lk, annotate, new_owner(), rows, status="shared")
            with LK.Selection().use(lk, snap) as table:
                assert (table is not None) is held, limit
        finally:
            lk.close()


def test_a_selection_limit_not_below_the_memory_limit_is_refused(tmp_path):
    with pytest.raises(LakeUnavailable, match="BENCH_LAKE_SELECTION_LIMIT"):
        Lake.open(LakeConfig(catalog=str(tmp_path / "catalog.ducklake"),
                             data=str(tmp_path / "data"),
                             memory_limit="256MB", selection_limit="256MB"))


def test_the_held_table_answers_as_the_pinned_query(lake, cells, annotate, make_rows,
                                                    new_owner):
    alice, admin = new_owner(), new_owner()
    rows = make_rows(3)
    s1 = _upload(lake, annotate, admin, rows[:2], status="shared")
    s2 = _upload(lake, annotate, alice, rows[1:])
    keys = [_key(annotate, r) for r in rows]
    held, pinned = LK.Selection(), LK.Selection()
    unheld = _limited(lake, 0)
    cells.refresh(unheld)
    for owner in (None, alice):
        for snap in (s2, s1):
            a = LK.lookup(lake, keys, owner=owner, snapshot=snap, selection=held,
                          cells=cells)
            b = LK.lookup(unheld, keys, owner=owner, snapshot=snap, selection=pinned,
                          cells=cells)
            assert a == b and a["rows"] != [[], [], []]
    assert len(_tables(lake)) == 1 and pinned._current[unheld][1] is None


def test_two_selections_on_one_lake_get_distinct_table_names(
        lake, annotate, make_rows, new_owner):
    owner = new_owner()
    snap = _upload(lake, annotate, owner, make_rows(1), status="shared")
    first, second = LK.Selection(), LK.Selection()
    # A copied handle is still the same DuckDB connection and name space.
    same_connection = _limited(lake, lake.selection_bytes)
    with first.use(lake, snap) as one, second.use(same_connection, snap) as two:
        assert one is not None and two is not None and one != two
        assert lake.read(f"SELECT count(*) AS n FROM {one}")[0]["n"] == 1
        assert lake.read(f"SELECT count(*) AS n FROM {two}")[0]["n"] == 1


def test_a_table_is_never_dropped_under_its_reader(lake, annotate, make_rows, new_owner):
    """A lookup holding snapshot 2's table while two newer snapshots' tables
    are built reads its table to the end; it is dropped once released."""
    owner = new_owner()
    rows = make_rows(3)
    snaps = [_upload(lake, annotate, owner, [r], status="shared") for r in rows]
    sel = LK.Selection()
    entered, release = threading.Event(), threading.Event()
    got: dict = {}

    def reader():
        with sel.use(lake, snaps[0]) as table:
            got["table"] = table
            entered.set()
            release.wait(10)
            got["rows"] = lake.read(f"SELECT count(*) AS n FROM {table}")[0]["n"]

    t = threading.Thread(target=reader)
    t.start()
    assert entered.wait(10)
    for snap in snaps[1:]:                      # two newer snapshots replace it
        with sel.use(lake, snap) as table:
            assert table is not None
    assert got["table"] in _tables(lake)
    release.set()
    t.join(10)
    assert got["rows"] == 1
    assert got["table"] not in _tables(lake) and len(_tables(lake)) == 1


def test_an_older_pin_takes_the_pinned_query(lake, annotate, make_rows, new_owner):
    owner = new_owner()
    rows = make_rows(2)
    s1 = _upload(lake, annotate, owner, rows[:1], status="shared")
    s2 = _upload(lake, annotate, owner, rows[1:], status="shared")
    sel = LK.Selection()
    with sel.use(lake, s2) as table:
        assert table is not None
    with sel.use(lake, s1) as table:
        assert table is None


@pytest.mark.parametrize("held", [False, True])
def test_concurrent_lookups_at_moving_snapshots_never_fail(lake, cells, annotate, make_rows,
                                                           new_owner, held):
    owner = new_owner()
    rows = make_rows(6)
    snaps = [_upload(lake, annotate, owner, [r], status="shared") for r in rows]
    keys = [_key(annotate, r) for r in rows]
    sel = LK.Selection()
    if not held:
        lake = _limited(lake, 0)
    cells.refresh(lake)
    errors: list = []

    def run(snap):
        try:
            for _ in range(5):
                out = LK.lookup(lake, keys, owner=None, snapshot=snap,
                                selection=sel, cells=cells)
                assert sum(map(len, out["rows"])) == snaps.index(snap) + 1
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=run, args=(s,)) for s in snaps]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert len(_tables(lake)) <= 1
