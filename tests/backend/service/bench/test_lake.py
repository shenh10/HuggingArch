"""lake.py: configuration, reads, transactions with conflict retry, and the
owner / maintainer locks (process locks on a file catalog, Postgres advisory
locks on a Postgres catalog)."""
from __future__ import annotations

import os
import stat
import threading
import time
from pathlib import Path

import duckdb
import pytest

from backend.service.bench import lake as lake_mod
from backend.service.bench.lake import (CommitConflict, Lake, LakeConfig, LakeUnavailable,
                                        is_commit_conflict)
from tests.backend.service.bench.pg import advisory_locks, scratch_database

_ENV = ("BENCH_LAKE_CATALOG", "BENCH_LAKE_DATA", "BENCH_LAKE_CACHE",
        "BENCH_LAKE_S3_ENDPOINT", "BENCH_LAKE_S3_REGION", "BENCH_LAKE_S3_KEY",
        "BENCH_LAKE_S3_SECRET", "BENCH_LAKE_MEMORY_LIMIT", "BENCH_LAKE_SELECTION_LIMIT")


@pytest.fixture
def clean_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# --- configuration ---------------------------------------------------------

def test_from_env_requires_catalog_and_data(clean_env):
    with pytest.raises(LakeUnavailable):
        LakeConfig.from_env()
    clean_env.setenv("BENCH_LAKE_CATALOG", "/x/c.ducklake")
    with pytest.raises(LakeUnavailable):
        LakeConfig.from_env()


def test_from_env_local_and_postgres(clean_env):
    clean_env.setenv("BENCH_LAKE_CATALOG", "/x/c.ducklake")
    clean_env.setenv("BENCH_LAKE_DATA", "/x/data")
    c = LakeConfig.from_env()
    assert (c.catalog, c.data, c.s3, c.postgres_dsn) == ("/x/c.ducklake", "/x/data", None, None)
    assert (c.spill_dir, c.staging_dir) == ("/x/tmp", "/x/staging")
    assert {"/x", "/x/data", "/x/tmp", "/x/staging"} == set(c.private_dirs())
    clean_env.setenv("BENCH_LAKE_CATALOG", "postgres:host=db dbname=x")
    clean_env.setenv("BENCH_LAKE_CACHE", "/x/cache")
    c = LakeConfig.from_env()
    assert c.postgres_dsn == "host=db dbname=x" and c.cache_dir == "/x/cache"


def test_from_env_rejects_unknown_catalog(clean_env):
    clean_env.setenv("BENCH_LAKE_CATALOG", "sqlite:/x/c.db")
    clean_env.setenv("BENCH_LAKE_DATA", "/x/data")
    with pytest.raises(LakeUnavailable, match="ducklake"):
        LakeConfig.from_env()


def test_from_env_s3_needs_credentials(clean_env):
    clean_env.setenv("BENCH_LAKE_CATALOG", "/x/c.ducklake")
    clean_env.setenv("BENCH_LAKE_DATA", "s3://bench-lake/lake/")
    clean_env.setenv("BENCH_LAKE_S3_ENDPOINT", "https://s3.example.com")
    with pytest.raises(LakeUnavailable, match="BENCH_LAKE_S3_KEY"):
        LakeConfig.from_env()
    for k, v in (("REGION", "r"), ("KEY", "k"), ("SECRET", "s")):
        clean_env.setenv(f"BENCH_LAKE_S3_{k}", v)
    assert LakeConfig.from_env().s3 == {"endpoint": "https://s3.example.com",
                                        "region": "r", "key": "k", "secret": "s"}


def test_the_api_workers_are_not_the_lakes(clean_env):
    """WEB_CONCURRENCY sizes the HA API; the lake lives in the Kernel Bench
    service, one process whatever the API runs (docs/measured_calibration.md §5.1)."""
    clean_env.setenv("BENCH_LAKE_CATALOG", "/x/c.ducklake")
    clean_env.setenv("BENCH_LAKE_DATA", "/x/data")
    clean_env.setenv("WEB_CONCURRENCY", "4")
    assert LakeConfig.from_env().memory_limit == "384MB"


def test_the_lake_memory_limit_comes_from_env(clean_env):
    clean_env.setenv("BENCH_LAKE_CATALOG", "/x/c.ducklake")
    clean_env.setenv("BENCH_LAKE_DATA", "/x/data")
    clean_env.setenv("BENCH_LAKE_MEMORY_LIMIT", "2GB")
    assert LakeConfig.from_env().memory_limit == "2GB"


def test_the_selection_limit_comes_from_env(clean_env):
    clean_env.setenv("BENCH_LAKE_CATALOG", "/x/c.ducklake")
    clean_env.setenv("BENCH_LAKE_DATA", "/x/data")
    assert LakeConfig.from_env().selection_limit == "32MiB"
    clean_env.setenv("BENCH_LAKE_SELECTION_LIMIT", "512MiB")
    assert LakeConfig.from_env().selection_limit == "512MiB"


# --- the root-only lake directory (docs/measured_calibration.md §4.5) ---

def test_open_creates_the_lake_directories_root_only(tmp_path):
    root = tmp_path / "lake"
    lk = Lake.open(LakeConfig(catalog=str(root / "catalog.ducklake"), data=str(root / "data")))
    try:
        for d in (root, root / "data", root / "tmp", root / "staging"):
            assert stat.S_IMODE(os.stat(d).st_mode) == 0o700, d
        spill = lk.read("SELECT value FROM duckdb_settings() WHERE name = 'temp_directory'")
        assert spill == [{"value": str(root / "tmp" / "spill")}]
    finally:
        lk.close()


@pytest.mark.parametrize("which", ["root", "data", "tmp"])
def test_open_refuses_a_directory_open_to_group_or_other(tmp_path, which):
    root = tmp_path / "lake"
    config = LakeConfig(catalog=str(root / "catalog.ducklake"), data=str(root / "data"))
    Lake.open(config).close()
    target = {"root": root, "data": root / "data", "tmp": root / "tmp"}[which]
    for mode in (0o750, 0o705, 0o701):
        os.chmod(target, mode)
        with pytest.raises(LakeUnavailable, match="0700"):
            Lake.open(config)
    os.chmod(target, 0o700)
    Lake.open(config).close()


def test_open_refuses_a_directory_of_another_user(tmp_path, monkeypatch):
    root = tmp_path / "lake"
    config = LakeConfig(catalog=str(root / "catalog.ducklake"), data=str(root / "data"))
    Lake.open(config).close()
    monkeypatch.setattr(lake_mod.os, "geteuid", lambda: os.stat(root).st_uid + 1)
    with pytest.raises(LakeUnavailable, match="owned by uid"):
        Lake.open(config)


def test_the_agent_environment_carries_no_lake_setting(monkeypatch):
    """The spec agents' environment (an allowlist) names no BENCH_LAKE_*
    setting — no lake path, no backup credential — and a scrubbed
    environment drops them even when the server has them."""
    from backend.agents import base
    assert not [k for k in base._SAFE_ENV_ALLOWLIST if k.upper().startswith("BENCH_LAKE")]
    for k in ("BENCH_LAKE_CATALOG", "BENCH_LAKE_DATA", "BENCH_LAKE_BACKUP_DATA",
              "BENCH_LAKE_BACKUP_S3_KEY", "BENCH_LAKE_BACKUP_S3_SECRET"):
        monkeypatch.setenv(k, "x")
    assert not [k for k in base.build_scrubbed_env() if k.startswith("BENCH_LAKE")]


def test_the_image_keeps_the_lake_root_only():
    """The image's entrypoint chowns only the agent's directories, makes the
    lake directory root-only, and starts the Kernel Bench service — the one
    process that opens the lake — beside the API."""
    root = Path(__file__).resolve().parents[4]
    text = (root / "Dockerfile").read_text()
    entry = (root / "scripts" / "entrypoint.sh").read_text()
    assert "chown -R specagent /app/.cache " not in text and "--workers" not in entry
    assert 'CMD ["bash", "/app/scripts/entrypoint.sh"]' in text
    assert 'chmod 0700 "$cache/bench_lake"' in entry
    for env in ("HF_HOME", "HUGGINGARCH_HF_META_CACHE_DIR", "HA_SPEC_RUNS_ROOT",
                "HA_USER_SPECS_ROOT", "HA_ADMISSION_CACHE_DIR"):
        assert f"${{{env}:-}}" in entry, env
    assert "python -m backend.service.bench.server" in entry
    assert "BENCH_LAKE_CATALOG=/app/.cache/bench_lake/catalog.ducklake" in text
    # No corpus copy of the lake is exported anywhere (docs/measured_calibration.md §7.2).
    assert "HUGGINGARCH_CORPUS_DIR" not in text


def test_open_unreachable_is_unavailable(tmp_path):
    with pytest.raises(LakeUnavailable):
        Lake.open(LakeConfig(catalog="postgres:host=/nonexistent port=1 dbname=x",
                             data=str(tmp_path / "d"), cache_dir=str(tmp_path / "c")))


def test_get_lake_is_a_singleton(clean_env, tmp_path):
    clean_env.setenv("BENCH_LAKE_CATALOG", str(tmp_path / "c.ducklake"))
    clean_env.setenv("BENCH_LAKE_DATA", str(tmp_path / "data"))
    clean_env.setenv("BENCH_LAKE_CACHE", str(tmp_path / "cache"))
    clean_env.setattr(lake_mod, "_LAKE", None)
    a = lake_mod.get_lake()
    try:
        assert lake_mod.get_lake() is a
    finally:
        a.close()


def test_settings_applied(lake):
    s = {r["name"]: r["value"] for r in lake.read(
        "SELECT name, value FROM duckdb_settings() WHERE name IN "
        "('threads', 'preserve_insertion_order', 'TimeZone')")}
    assert s == {"threads": "4", "preserve_insertion_order": "false", "TimeZone": "UTC"}
    opts = {r["option_name"]: r["value"] for r in lake.read("SELECT * FROM lake.options()")}
    assert opts["target_file_size"] == "32000000"


# --- reads and transactions ------------------------------------------------

def test_read_returns_dicts_with_params(lake):
    assert lake.read("SELECT ? AS a, ?::INT + 1 AS b", ["x", 1]) == [{"a": "x", "b": 2}]


def test_reads_are_bounded_by_the_semaphore(lake):
    seen, peak, guard = [0], [0], threading.Lock()
    real = lake._reads

    class Counting:
        def __enter__(self):
            real.acquire()
            with guard:
                seen[0] += 1
                peak[0] = max(peak[0], seen[0])
            time.sleep(0.05)

        def __exit__(self, *a):
            with guard:
                seen[0] -= 1
            real.release()

    lake._reads = Counting()
    threads = [threading.Thread(target=lake.read, args=("SELECT 1",)) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == lake.config.max_concurrent_reads == 2


def _table(lake):
    lake.transact(lambda cur: cur.execute("CREATE TABLE lake.t (a INTEGER)"))
    lake.transact(lambda cur: cur.execute("INSERT INTO lake.t VALUES (1)"))


def test_transact_commits_and_snapshot_advances(lake):
    before = lake.snapshot_id()
    _table(lake)
    assert lake.snapshot_id() > before
    assert lake.read("SELECT a FROM lake.t") == [{"a": 1}]


def test_transact_rolls_back_on_error(lake):
    _table(lake)

    def fail(cur):
        cur.execute("INSERT INTO lake.t VALUES (2)")
        raise KeyError("boom")
    with pytest.raises(KeyError):
        lake.transact(fail)
    assert lake.read("SELECT a FROM lake.t") == [{"a": 1}]


def _concurrent_delete(lake):
    """Commit a delete of lake.t on another cursor, while the caller's
    transaction is open — a genuine DuckLake commit conflict for it."""
    other = lake._con.cursor()
    other.execute("DELETE FROM lake.t WHERE a = 1")
    other.close()


def test_a_real_conflict_is_a_transaction_exception(lake):
    _table(lake)
    a = lake._con.cursor()
    a.execute("BEGIN")
    a.execute("INSERT INTO lake.t VALUES (3)")
    _concurrent_delete(lake)
    with pytest.raises(duckdb.TransactionException) as e:
        a.execute("COMMIT")
    assert is_commit_conflict(e.value)
    assert not is_commit_conflict(duckdb.TransactionException("cannot rollback"))


def test_transact_retries_a_conflict(lake, monkeypatch):
    monkeypatch.setattr(lake_mod.time, "sleep", lambda s: None)
    _table(lake)
    calls = []

    def fn(cur):
        calls.append(1)
        cur.execute("INSERT INTO lake.t VALUES (5)")
        if len(calls) == 1:
            _concurrent_delete(lake)
        return len(calls)
    assert lake.transact(fn) == 2
    assert lake.read("SELECT a FROM lake.t ORDER BY a") == [{"a": 5}]


def test_transact_gives_up_after_three_conflicts(lake, monkeypatch):
    sleeps = []
    monkeypatch.setattr(lake_mod.time, "sleep", sleeps.append)
    _table(lake)
    calls = []

    def fn(cur):
        calls.append(1)
        cur.execute("INSERT INTO lake.t VALUES (5)")
        other = lake._con.cursor()
        other.execute("INSERT INTO lake.t VALUES (7)")
        other.execute("DELETE FROM lake.t WHERE a = 7")
        other.close()
    with pytest.raises(CommitConflict):
        lake.transact(fn)
    assert len(calls) == 3 and len(sleeps) == 2 and all(s > 0 for s in sleeps)


def test_transact_does_not_retry_other_errors(lake):
    calls = []

    def fn(cur):
        calls.append(1)
        cur.execute("SELECT * FROM lake.missing_table")
    with pytest.raises(duckdb.CatalogException):
        lake.transact(fn)
    assert calls == [1]


# --- locks on a file catalog ------------------------------------------------

def _overlaps(lake, owners):
    """Run a 0.1 s critical section per owner in threads; return the peak
    number of sections inside at once."""
    inside, peak, guard = [0], [0], threading.Lock()

    def work(owner):
        with lake.owner_lock(owner):
            with guard:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            time.sleep(0.1)
            with guard:
                inside[0] -= 1
    ts = [threading.Thread(target=work, args=(o,)) for o in owners]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return peak[0]


def test_owner_lock_serializes_one_owner_only(lake):
    assert _overlaps(lake, ["a", "a", "a"]) == 1
    assert _overlaps(lake, ["a", "b", "c"]) == 3


def test_owner_lock_released_on_error(lake):
    with pytest.raises(RuntimeError), lake.owner_lock("a"):
        raise RuntimeError
    assert _overlaps(lake, ["a"]) == 1


def test_maintainer_lock_is_exclusive_and_non_blocking(lake):
    with lake.maintainer_lock() as first, lake.maintainer_lock() as second:
        assert (first, second) == (True, False)
    with lake.maintainer_lock() as again:
        assert again


# --- locks on a Postgres catalog ---------------------------------------------

@pytest.fixture
def pg_config(tmp_path):
    with scratch_database("bench_lake_test_lake") as dsn:
        yield LakeConfig(catalog=f"postgres:{dsn}", data=str(tmp_path / "data"),
                         cache_dir=str(tmp_path / "cache"))


def test_postgres_owner_lock_holds_across_instances(pg_config):
    """Two DuckDB instances stand for two API processes: the second blocks on
    the owner the first holds, until the first releases; the lock is a
    Postgres advisory lock while held and gone after."""
    a, b = Lake.open(pg_config), Lake.open(pg_config)
    try:
        got = threading.Event()

        def second():
            with b.owner_lock("owner-1"):
                got.set()
        with a.owner_lock("owner-1"):
            t = threading.Thread(target=second)
            t.start()
            assert not got.wait(0.5)
            locks = advisory_locks("bench_lake_test_lake")
            assert [g for _, g in locks] == [True, False]
            with b.owner_lock("owner-2"):      # another owner is not blocked
                pass
        assert got.wait(5)
        t.join()
        assert advisory_locks("bench_lake_test_lake") == []
    finally:
        a.close()
        b.close()


def test_postgres_owner_lock_holds_across_a_lake_commit(pg_config):
    a, b = Lake.open(pg_config), Lake.open(pg_config)
    try:
        first = a.snapshot_id()
        a.transact(lambda cur: cur.execute("CREATE TABLE lake.t (a INTEGER)"))
        assert b.snapshot_id() == a.snapshot_id() > first
        with a.owner_lock("o"):
            a.transact(lambda cur: cur.execute("INSERT INTO lake.t VALUES (1)"))
            assert [g for _, g in advisory_locks("bench_lake_test_lake")] == [True]
            assert b.read("SELECT count(*) AS n FROM lake.t") == [{"n": 1}]
        assert advisory_locks("bench_lake_test_lake") == []
    finally:
        a.close()
        b.close()


def test_postgres_maintainer_lock_is_one_process(pg_config):
    a, b = Lake.open(pg_config), Lake.open(pg_config)
    try:
        with a.maintainer_lock() as ha, b.maintainer_lock() as hb:
            assert (ha, hb) == (True, False)
        with b.maintainer_lock() as hb:
            assert hb
    finally:
        a.close()
        b.close()
