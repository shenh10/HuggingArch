"""The maintainer (backend/service/bench/maintain.py) on a local lake; its
backups go to a local directory (the backup protocol: test_backup.py)."""
from __future__ import annotations

import threading
import uuid

import pytest

from backend.service.bench import maintain, store
from backend.service.bench.backup import Backup, LocalBucket

_TABLES = {"batches": "batch_id", "inputs": "input_id", "measurements": "id"}


def _up(lake, owner, rows, annotate, **kw):
    return store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()), run_id="r",
                        label="l", rows=rows, **kw)


def _dump(lake, at=None):
    v = f" AT (VERSION => {at})" if at is not None else ""
    return {t: lake.read(f"SELECT * FROM lake.{t}{v} ORDER BY {k}") for t, k in _TABLES.items()}


# --- the maintainer lock -------------------------------------------------------------

def test_maintainer_lock_is_exclusive(lake, make_rows, annotate, new_owner):
    _up(lake, new_owner(), make_rows(2), annotate)
    results = []
    with lake.maintainer_lock() as held:
        assert held
        # the same process, another thread: the lock is taken, nothing runs
        t = threading.Thread(target=lambda: results.append(maintain.maintain_once(lake)))
        t.start()
        t.join()
        assert maintain.maintain_once(lake) == {"maintainer": False}
    assert results == [{"maintainer": False}]
    assert maintain.maintain_once(lake, housekeep=False) == {"maintainer": True}


# --- DuckLake maintenance ---------------------------------------------------------------

def test_housekeeping_runs_on_a_local_lake(lake, make_rows, annotate, new_owner):
    owner = new_owner()
    for k in range(4):            # four small files
        _up(lake, owner, make_rows(40 * (k + 1)), annotate)
    want = _dump(lake)
    did = maintain.maintain_once(lake)
    assert did["maintainer"] and did["merged"] >= 1
    # nothing is older than 7 days: no snapshot expires, no file goes
    assert did["expired"] == did["cleaned"] == did["orphaned"] == 0
    assert _dump(lake) == want


# --- backups in the cycle ----------------------------------------------------------------

def test_maintain_once_backs_up_and_holds_the_backup_lock_while_housekeeping(
        tmp_path, lake, make_rows, annotate, new_owner, monkeypatch):
    backup = Backup(LocalBucket(str(tmp_path / "bucket")))
    _up(lake, new_owner(), make_rows(20), annotate)
    held = []
    real = maintain.housekeeping

    def housekeeping(lk, expire=True):
        held.append((lk._process_locks["bench:backup"].locked(), expire))
        return real(lk, expire=expire)
    monkeypatch.setattr(maintain, "housekeeping", housekeeping)
    did = maintain.maintain_once(lake, backup=backup)
    assert held == [(True, True)]
    assert did["backup"]["snapshot"] == lake.snapshot_id() and "gc" in did
    did = maintain.maintain_once(lake, backup=backup, housekeep=False)
    assert did["backup"] is None and "gc" not in did          # nothing new


def test_expiry_and_retention_wait_while_the_backup_is_behind(
        tmp_path, lake, make_rows, annotate, new_owner, monkeypatch):
    backup = Backup(LocalBucket(str(tmp_path / "bucket")))
    _up(lake, new_owner(), make_rows(20), annotate)
    monkeypatch.setattr(backup, "run", lambda lk: None)      # a backup that does not catch up
    monkeypatch.setattr(backup, "gc", lambda *a: pytest.fail("no retention while behind"))
    did = maintain.maintain_once(lake, backup=backup)
    assert did["expiry_held"] == "the backup is behind"
    assert "merged" in did and not {"expired", "cleaned", "orphaned"} & set(did)


def test_maintenance_needs_free_space(lake, monkeypatch):
    from collections import namedtuple

    from backend.service.bench import lake as lake_mod
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(lake_mod.shutil, "disk_usage", lambda p: usage(0, 0, 10 * 2**20))
    with pytest.raises(lake_mod.LakeUnavailable, match="maintenance refused"):
        maintain.maintain_once(lake)


# --- the loop -----------------------------------------------------------------------------

def test_start_is_off_by_switch(lake, monkeypatch):
    monkeypatch.setenv("HUGGINGARCH_BENCH_MAINTENANCE", "off")
    assert maintain.start(lake, None) is None


def test_loop_paces_and_backs_off(lake, monkeypatch):
    now = [0.0]
    calls = []

    def fake_once(lk, *, backup, housekeep):
        calls.append(housekeep)
        if len(calls) in (2, 3):
            raise RuntimeError("boom")
        now[0] += 20.0                    # the cycle took 20 s
        return {"maintainer": True}
    monkeypatch.setattr(maintain, "maintain_once", fake_once)
    loop = maintain.Loop(lambda: lake, None, clock=lambda: now[0])
    assert loop.step() == 60.0            # 3 × 20 s
    assert loop.step() == 30.0 and loop.step() == 60.0     # backoff doubles
    assert loop.step() == 60.0            # recovered; backoff reset
    assert loop.backoff == maintain.MIN_REST
    assert calls == [True, False, False, False]    # housekeeping at most hourly
    now[0] += maintain.HOUSEKEEPING_EVERY
    loop.step()
    assert calls[-1] is True


# --- the content identity across a merge ---------------------------------------------

def test_content_id_tells_snapshots_apart_inside_a_merged_file(lake):
    """`ducklake_merge_adjacent_files` merges files written at several
    snapshots into one file that keeps the oldest one's begin_snapshot (and
    a `partial_max`); a read pinned at an earlier snapshot sees only that
    snapshot's part of it. The live catalog rows are then the same at each
    of those snapshots, so `Lake.content_id` must tell them apart by the
    snapshot itself wherever the visible rows differ. (One table only, so
    nothing else in the lake changes between the snapshots.)"""
    lake.transact(lambda cur: cur.execute("CREATE TABLE lake.probe (i BIGINT)"))
    snaps = []
    for k in range(3):                                     # three data files
        lake.transact(lambda cur, k=k: cur.execute(
            f"INSERT INTO lake.probe SELECT range FROM range({k * 300}, {(k + 1) * 300})"))
        snaps.append(lake.snapshot_id())
    assert maintain.housekeeping(lake, expire=False)["merged"] > 0
    m = "__ducklake_metadata_lake.main"
    t = lake.read(f"SELECT table_id FROM {m}.ducklake_table WHERE table_name = 'probe'")[0][
        "table_id"]
    files = lake.read(f"SELECT path, begin_snapshot AS b, partial_max AS p FROM "
                      f"{m}.ducklake_data_file WHERE table_id = {t} AND end_snapshot IS NULL")
    assert [(f["b"], f["p"]) for f in files] == [(snaps[0], snaps[2])]   # one merged file
    counts = [lake.read(f"SELECT count(*) AS n FROM lake.probe AT (VERSION => {s})")[0]["n"]
              for s in snaps]
    assert counts == [300, 600, 900]
    lake._content_ids.clear()                              # recomputed after the merge
    assert len({lake.content_id(s) for s in snaps}) == 3


def test_inlined_deletions_are_none_before_the_first_and_read_after(lake):
    """DuckLake makes `ducklake_inlined_delete_<table>` at a table's first
    inlined deletion: before it the answer is none; after it, the deleted
    rows; another catalog error is not swallowed."""
    import duckdb

    lake.transact(lambda cur: cur.execute("CREATE TABLE lake.probe (i BIGINT)"))
    lake.transact(lambda cur: cur.execute("INSERT INTO lake.probe SELECT range FROM range(100)"))
    m = "__ducklake_metadata_lake.main"
    t = int(lake.read(f"SELECT table_id FROM {m}.ducklake_table WHERE table_name = 'probe'")[0][
        "table_id"])
    assert lake._inlined_deletions(t, lake.snapshot_id()) == []
    before = lake.content_id(lake.snapshot_id())
    lake.transact(lambda cur: cur.execute("DELETE FROM lake.probe WHERE i = 5"))
    assert len(lake._inlined_deletions(t, lake.snapshot_id())) == 1
    assert lake.content_id(lake.snapshot_id()) != before

    def other(*_a, **_k):
        raise duckdb.CatalogException("Catalog Error: Schema with name x does not exist!")
    lake.read = other
    with pytest.raises(duckdb.CatalogException, match="Schema"):
        lake._inlined_deletions(t, 1)
