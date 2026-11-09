"""The lake's derived file (`derived.Derived`, docs/measured_calibration.md
§7.2) as a lookup reads it: the one place a lookup reads its rows' cells. No lookup reads the
lake's wide columns; a file behind the lookup's pin is caught up before the
lookup answers, and one that cannot be caught up in time fails the lookup
(503: measurements unavailable) rather than sending it to the lake's files.
The file keeps each row's versions with the snapshots they are live at, so
every pin from its base answers exactly what the lake holds at that pin."""
from __future__ import annotations

import threading
import uuid

import pytest

from backend.inference.op_calibration import ROW_COLUMNS
from backend.service.bench import derived as D
from backend.service.bench import lookup as LK
from backend.service.bench import store
from backend.service.bench.lake import Lake, LakeConfig, LakeUnavailable
from tests.backend.service.bench.conftest import derive, make_annotator


def _upload(lake, annotate, owner, rows, *, status="shared"):
    store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()), run_id=None, label=None,
                 rows=rows, status=status,
                 decided_by=owner if status == "shared" else None)
    return lake.snapshot_id()


def _key(annotate, row) -> dict:
    ann = annotate(row)
    return {"signature": ann["signature"], "physical_shape": ann["physical_shape"],
            "dtype_env_json": row["dtype_env_json"], "gpu": row["gpu"], "phase": row["phase"]}


_SELECTION = LK.Selection()


def _lookup(lake, keys, snap, derived, owner=None):
    return LK.lookup(lake, keys, owner=owner, snapshot=snap, selection=_SELECTION,
                     derived=derived)


def _file_at(lake, snap) -> dict:
    """The versions the file holds live at `snap`."""
    return {r["id"]: tuple(LK._cell(r[c]) for c in ROW_COLUMNS) for r in lake.read(
        f"SELECT id::VARCHAR AS id, * EXCLUDE (id) FROM {D.ROWS} WHERE begin_snapshot <= ? "
        "AND (end_snapshot IS NULL OR end_snapshot > ?)", [snap, snap])}


def _lake_at(lake, snap) -> dict:
    """The second route: every row's cells read from the lake itself."""
    return {r["id"]: tuple(LK._cell(r[c]) for c in ROW_COLUMNS) for r in lake.read(
        f"SELECT id::VARCHAR AS id, {D.CELLS_SQL} FROM lake.measurements m "
        "AT (VERSION => ?)", [snap])}


def test_no_lookup_reads_the_lakes_wide_columns(lake, derived, annotate, make_rows,
                                                new_owner, monkeypatch):
    """Not with the file current, not behind it (the catch-up is the
    builder's, not the lookup's), not at an older pin."""
    owner = new_owner()
    rows = make_rows(4)
    s1 = _upload(lake, annotate, owner, rows[:2])
    keys = [_key(annotate, r) for r in rows]
    wide: list[str] = []
    real = lake.read

    def read(sql, params=()):
        if "lake.measurements" in sql and '"shape_json"' in sql:
            wide.append(sql)
        return real(sql, params)
    monkeypatch.setattr(lake, "read", read)
    first = _lookup(lake, keys, s1, derived)
    s2 = _upload(lake, annotate, owner, rows[2:], status="private")
    now = _lookup(lake, keys, s2, derived, owner=owner)
    assert _lookup(lake, keys, s1, derived) == first
    assert wide == []
    assert sum(map(len, first["rows"])) == 2
    assert sum(map(len, now["rows"])) == 6       # the owner's shared rows twice


def _cell_of(out, key_index: int, column: str) -> str:
    """One returned row's cell, decoded from the lookup's value table."""
    row = out["rows"][key_index][0]
    return out["values"][row[out["columns"].index(column)]]


def test_a_restamped_row_is_answered_with_its_cells_at_each_pin(
        lake, derived, annotate, make_rows, new_owner):
    """A row id names different content before and after `store.restamp`:
    the file, behind the restamp, catches up before answering the new pin
    and still answers the old pin with the old cells."""
    owner = new_owner()
    rows = make_rows(1)
    before = _upload(lake, annotate, owner, rows)
    key = _key(annotate, rows[0])
    assert _cell_of(_lookup(lake, [key], before, derived), 0,
                    "achieved_tflops") == rows[0]["achieved_tflops"]
    row_id = next(iter(_lake_at(lake, before)))
    store.restamp(lake, owner=owner, rows=[{"id": row_id, "achieved_tflops": "1234.5"}])
    after = lake.snapshot_id()
    assert _cell_of(_lookup(lake, [key], after, derived), 0, "achieved_tflops") == "1234.5"
    assert _cell_of(_lookup(lake, [key], before, derived), 0,
                    "achieved_tflops") == rows[0]["achieved_tflops"]


def test_a_behind_file_is_caught_up_before_the_lookup_answers(lake, derived, annotate,
                                                              make_rows, new_owner):
    owner = new_owner()
    rows = make_rows(3)
    _upload(lake, annotate, owner, rows[:1])
    derived.refresh(lake)
    s2 = _upload(lake, annotate, owner, rows[1:])
    assert derived.held(lake)[1] < s2
    out = _lookup(lake, [_key(annotate, r) for r in rows], s2, derived)
    assert [len(r) for r in out["rows"]] == [1, 1, 1]
    assert derived.held(lake)[1] >= s2


def test_a_catch_up_that_cannot_finish_fails_the_lookup(lake, derived, annotate, make_rows,
                                                        new_owner, monkeypatch):
    owner = new_owner()
    rows = make_rows(2)
    s1 = _upload(lake, annotate, owner, rows[:1])
    s2 = _upload(lake, annotate, owner, rows[1:])

    def broken(lk):
        raise LakeUnavailable("the builder is down")
    monkeypatch.setattr(derived, "refresh", broken)
    monkeypatch.setattr(D.Derived, "CATCH_UP_S", 0.3)
    keys = [_key(annotate, r) for r in rows]
    with pytest.raises(LakeUnavailable, match="did not reach snapshot"):
        _lookup(lake, keys, s2, derived)
    del s1


def test_a_pin_before_the_files_base_is_refused(lake, annotate, make_rows, new_owner):
    owner = new_owner()
    rows = make_rows(2)
    s1 = _upload(lake, annotate, owner, rows[:1])
    _upload(lake, annotate, owner, rows[1:])
    derived = derive(lake, annotate, rules="test")  # built after s1: base is s2
    with pytest.raises(LakeUnavailable, match="older than"):
        _lookup(lake, [_key(annotate, rows[0])], s1, derived)


def test_every_pin_from_the_base_holds_what_the_lake_holds(lake, derived, annotate,
                                                           make_rows, new_owner):
    """Inserts, a delete and an in-place restamp, followed by the change
    feed: at every snapshot the file answers, its live versions are the
    lake's rows at that snapshot."""
    alice = new_owner()
    rows = make_rows(6)
    snaps = [_upload(lake, annotate, alice, rows[:3])]
    derived.refresh(lake)
    snaps.append(_upload(lake, annotate, alice, rows[3:], status="private"))
    ids = sorted(_lake_at(lake, snaps[-1]))
    store.delete_rows(lake, owner=alice, ids=[ids[0]])
    snaps.append(lake.snapshot_id())
    store.restamp(lake, owner=alice, rows=[{"id": ids[1], "achieved_tflops": 1.25}])
    snaps.append(lake.snapshot_id())
    got = derived.refresh(lake)
    assert got["how"] == "changes" and got["snapshot"] == snaps[-1]
    base, at = derived.held(lake)
    assert base <= snaps[0] and at == snaps[-1]
    for snap in range(base, at + 1):
        assert _file_at(lake, snap) == _lake_at(lake, snap), snap
    assert _file_at(lake, at)[ids[1]][ROW_COLUMNS.index("achieved_tflops")] == "1.25"
    assert derived.refresh(lake)["how"] == "current"


def test_a_reopened_lake_resumes_from_the_file_it_left(tmp_path, make_rows, new_owner):
    annotate = make_annotator()
    config = LakeConfig(catalog=str(tmp_path / "catalog.ducklake"),
                        data=str(tmp_path / "data"), cache_dir=str(tmp_path / "cache"))
    owner = new_owner()
    rows = make_rows(3)
    lk = Lake.open(config)
    store.ensure_schema(lk)
    _upload(lk, annotate, owner, rows[:2])
    assert D.Derived(annotate=annotate, rules=lambda: "r1").refresh(lk)["how"] == "rebuilt"
    lk.close()
    lk = Lake.open(config)
    try:
        snap = _upload(lk, annotate, owner, rows[2:])
        # A new process: the file's history is checked by content, then followed.
        again = D.Derived(annotate=annotate, rules=lambda: "r1")
        assert again.refresh(lk) == {"snapshot": snap, "how": "changes", "rows": 1}
        assert _file_at(lk, snap) == _lake_at(lk, snap)
        # Other annotation rules: the file is rebuilt, never followed.
        assert D.Derived(annotate=annotate, rules=lambda: "r2").refresh(lk)["how"] == "rebuilt"
    finally:
        lk.close()


@pytest.mark.parametrize("n", [4])
def test_lookups_read_while_the_file_refreshes(lake, derived, annotate, make_rows,
                                               new_owner, n):
    owner = new_owner()
    rows = make_rows(n)
    snaps = [_upload(lake, annotate, owner, [r]) for r in rows]
    keys = [_key(annotate, r) for r in rows]
    errors: list = []

    def run():
        try:
            for _ in range(5):
                out = _lookup(lake, keys, snaps[-1], derived)
                assert sum(map(len, out["rows"])) == n
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=run) for _ in range(3)]
    for t in threads:
        t.start()
    derived.refresh(lake)
    for t in threads:
        t.join(30)
    assert errors == []
