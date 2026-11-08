"""The DuckDB instance attached to the Kernel Bench lake.

Configuration (environment):
  BENCH_LAKE_CATALOG   DuckLake catalog: a local file path ending in
                       `.ducklake` (production, docs/measured_calibration.md §4.1: the
                       volume's `/app/.cache/bench_lake/catalog.ducklake`;
                       tests, development), or `postgres:<libpq dsn>`
                       (schema `bench_lake`; supported for a multi-machine
                       deployment, not used)
  BENCH_LAKE_DATA      data path: a local directory (production: the
                       volume's `/app/.cache/bench_lake/data/`) or
                       `s3://bench-lake/lake/`
  BENCH_LAKE_S3_ENDPOINT / _REGION / _KEY / _SECRET   for an s3:// data path
  BENCH_LAKE_CACHE     local directory for spill and the remote-file cache
                       (default: `tmp/` beside a file catalog)
  BENCH_LAKE_MEMORY_LIMIT  DuckDB memory limit (default: 384MB)
  BENCH_LAKE_SELECTION_LIMIT  the share of that memory the lookup's held
                       selection table may take (`lookup.Selection`; default
                       32MiB, ~460k measurements). A lake whose table would
                       exceed it is looked up by the pinned query over its
                       files. DuckDB's size grammar, as the memory limit; it
                       must be below the memory limit (`Lake.open` refuses
                       otherwise)

One `Lake` per process, and one process (docs/measured_calibration.md §5.1): the Kernel Bench
service (`server`) runs exactly one, and DuckDB locks a file catalog against
a second. Settings (docs/measured_calibration.md §4.3): threads 4,
memory_limit BENCH_LAKE_MEMORY_LIMIT (default 384MB) — of which the held
selection table may take BENCH_LAKE_SELECTION_LIMIT —
preserve_insertion_order false, target_file_size 32 MB
(Supabase Storage refuses objects over 50 MB). At most two analytical reads
run at once (a process-wide semaphore); each request uses its own cursor.

The lake's local directories — a file catalog's directory, a local data
directory, the spill directory — are root-only (docs/measured_calibration.md §4.5): `Lake.open`
creates a missing one with mode 0700 and refuses (`LakeUnavailable`) one not
owned by the process user or granting any group / other permission. The
untrusted spec agents run as another user and must not read or change the
measurements.

Locks. With a Postgres catalog a lock is a Postgres advisory lock taken on
its own connection: a cursor that opens a DuckDB transaction and takes
`pg_advisory_xact_lock` through the `postgres` extension (a second attach of
the catalog database, `bench_lock`). Inside a DuckDB transaction the extension
pins one Postgres connection, so the lock is held by that connection until the
cursor rolls back — across the lake transaction, which runs on another cursor
and another catalog connection. Outside a transaction the extension pools its
connections and a session lock taken through `postgres_execute` does not stay
held (tested), which is why the transaction-scoped form is used: it cannot be
returned to the pool still held, and a crashed process releases it with its
connection. With a file catalog only one process can open the lake (DuckDB
locks the file), so a process lock is the same guarantee.

Lock order: owner → inputs → (commit). The backup lock and the maintainer
lock are never taken while a write lock is held; maintenance takes
maintainer → backup.

Free space. Every write, maintenance run and backup staging first checks
the lake's volume (`require_free`): under `MIN_FREE_BYTES` it refuses
(`LakeUnavailable`).

Commit conflicts. DuckLake raises `duckdb.TransactionException` from COMMIT
("Failed to commit DuckLake transaction. Transaction conflict - …") when a
concurrent commit touched what this one changes — e.g. a delete against an
insert into the same table; two appends do not conflict. A failed COMMIT has
already rolled the transaction back.

Content identity (`content_id`). A snapshot id names a commit only within
one lake's history: a lake restored or copied (to another machine, or beside
itself) keeps its id and its history, and once the copies diverge one
snapshot id — even one commit time — is two different contents. What a
read pinned at snapshot `n` sees is a function of `n` and the catalog rows
live at `n`: the data and delete files (DuckLake names each file it writes
uniquely and never rewrites one), the inlined rows and inlined deletions
(small writes kept in the catalog, by value), and the tables and columns.
`n` itself is needed: `ducklake_merge_adjacent_files` merges files written
at several snapshots into one file that keeps the oldest one's
`begin_snapshot` (with a `partial_max`), and a read pinned at an earlier
snapshot filters the rows inside it by their own snapshot — the live
catalog rows are then the same at every snapshot the file spans, the
visible rows are not. `content_id(n)` hashes `n` and those rows, so equal
ids mean equal row and table content at that snapshot (what a pinned
lookup reads) — not equal namespace metadata: schemas, tags, views and
macros are not hashed, and no lookup reads them. Divergent copies at one
snapshot differ by construction (their diverging commits wrote different
files, or different inlined rows); copies holding the same rows at the same
snapshot share it; one lake at two snapshot numbers never does, even with
the same rows (a cache miss, never a wrong answer). It reads only the
catalog — tens of rows here — and is memoized per snapshot.
"""
from __future__ import annotations

import hashlib
import os
import random
import shutil
import stat
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Sequence, TypeVar

import duckdb

T = TypeVar("T")

_ATTEMPTS = 3
_CONTENT_IDS_KEPT = 256              # snapshots whose content_id is memoized
MIN_FREE_BYTES = 500 * 1024 * 1024   # writes, maintenance and backups stop below this (docs/measured_calibration.md §4.3)


@dataclass(frozen=True)
class LakeConfig:
    catalog: str            # "postgres:<dsn>" or a path to a .ducklake file
    data: str               # "s3://…/" or a local directory
    s3: Optional[dict] = None   # endpoint, region, key, secret
    cache_dir: str = ""     # BENCH_LAKE_CACHE; "" → beside the catalog (`spill_dir`)
    threads: int = 4
    memory_limit: str = "384MB"  # BENCH_LAKE_MEMORY_LIMIT
    selection_limit: str = "32MiB"  # BENCH_LAKE_SELECTION_LIMIT
    max_concurrent_reads: int = 2

    @property
    def postgres_dsn(self) -> Optional[str]:
        """The catalog database's libpq dsn, for a Postgres catalog."""
        return self.catalog[len("postgres:"):] if self.catalog.startswith("postgres:") else None

    @property
    def root(self) -> Optional[str]:
        """The lake's own local directory — a file catalog's directory —
        holding `tmp/` (spill) and `staging/` (backups) beside the catalog
        and, in production, `data/`. None for a Postgres catalog."""
        if self.postgres_dsn is not None:
            return None
        return os.path.dirname(os.path.abspath(self.catalog))

    def _beside(self, name: str) -> str:
        if self.cache_dir:
            return os.path.join(self.cache_dir, name) if name != "tmp" else self.cache_dir
        if self.root is None:
            raise LakeUnavailable("BENCH_LAKE_CACHE must be set for a Postgres catalog")
        return os.path.join(self.root, name)

    @property
    def spill_dir(self) -> str:
        """DuckDB's spill: BENCH_LAKE_CACHE, else
        `<lake dir>/tmp`."""
        return self._beside("tmp")

    @property
    def staging_dir(self) -> str:
        """Where a backup stages its catalog copy and a restore its
        downloads: `<lake dir>/staging` (inside BENCH_LAKE_CACHE when set)."""
        return self._beside("staging")

    def private_dirs(self) -> list[str]:
        """Every local directory of the lake, each root-only (`private_dir`)."""
        dirs = [self.root] if self.root is not None else []
        if not self.data.startswith("s3://"):
            dirs.append(os.path.abspath(self.data))
        dirs += [os.path.abspath(self.spill_dir), os.path.abspath(self.staging_dir)]
        return list(dict.fromkeys(dirs))

    @classmethod
    def from_env(cls) -> "LakeConfig":
        catalog = os.environ.get("BENCH_LAKE_CATALOG", "").strip()
        data = os.environ.get("BENCH_LAKE_DATA", "").strip()
        if not catalog or not data:
            raise LakeUnavailable("BENCH_LAKE_CATALOG and BENCH_LAKE_DATA must be set")
        if not (catalog.startswith("postgres:") or catalog.endswith(".ducklake")):
            raise LakeUnavailable(
                f"BENCH_LAKE_CATALOG must be `postgres:<dsn>` or a .ducklake path, got {catalog!r}")
        s3 = None
        if data.startswith("s3://"):
            s3 = {k: os.environ.get(f"BENCH_LAKE_S3_{k.upper()}", "").strip()
                  for k in ("endpoint", "region", "key", "secret")}
            missing = [f"BENCH_LAKE_S3_{k.upper()}" for k, v in s3.items() if not v]
            if missing:
                raise LakeUnavailable(f"an s3:// data path needs {', '.join(missing)}")
        cache = os.environ.get("BENCH_LAKE_CACHE", "").strip()
        memory_limit = os.environ.get("BENCH_LAKE_MEMORY_LIMIT", "").strip()
        selection_limit = os.environ.get("BENCH_LAKE_SELECTION_LIMIT", "").strip()
        overrides = {"cache_dir": cache, "memory_limit": memory_limit,
                     "selection_limit": selection_limit}
        return cls(catalog=catalog, data=data, s3=s3,
                   **{name: value for name, value in overrides.items() if value})


class LakeUnavailable(RuntimeError):
    """The lake cannot be reached or is not configured (routes answer 503)."""


class CommitConflict(RuntimeError):
    """A write transaction lost to a concurrent commit three times."""


def is_commit_conflict(e: BaseException) -> bool:
    """Whether `e` is DuckLake refusing a commit for a concurrent one."""
    return isinstance(e, duckdb.TransactionException) and "conflict" in str(e).lower()


def _sql_text(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def private_dir(path: str) -> None:
    """Create `path` with mode 0700 when missing; refuse (`LakeUnavailable`)
    a symlink, or a directory not owned by this process's user or granting
    any group / other permission (docs/measured_calibration.md §4.5)."""
    if os.path.islink(path):
        raise LakeUnavailable(f"lake directory {path} is a symlink")
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)            # makedirs' mode is masked by the umask
    st = os.stat(path)
    if st.st_uid != os.geteuid():
        raise LakeUnavailable(f"lake directory {path} is owned by uid {st.st_uid}, "
                              f"not this process's ({os.geteuid()})")
    if st.st_mode & 0o077:
        raise LakeUnavailable(f"lake directory {path} has mode {stat.S_IMODE(st.st_mode):04o}; "
                              "it must grant nothing to group or other (0700)")


def _set_lake_option(con: duckdb.DuckDBPyConnection, name: str, value: str,
                     stored: str) -> None:
    """Set a DuckLake option (a persistent catalog entry) unless it already
    reads `stored`. Processes opening the lake at once race on the write
    ("could not serialize access" from a Postgres catalog); the loser is fine
    when the winner stored the same value."""
    def current() -> Optional[str]:
        row = con.execute("SELECT value FROM lake.options() WHERE option_name = ?",
                          [name]).fetchone()
        return row[0] if row else None
    if current() == stored:
        return
    try:
        con.execute(f"CALL lake.set_option({_sql_text(name)}, {_sql_text(value)})")
    except duckdb.Error:
        if current() != stored:
            raise


def _advisory_key(name: str) -> int:
    """A stable signed 64-bit advisory-lock key for `name`."""
    return int.from_bytes(hashlib.md5(name.encode()).digest()[:8], "big", signed=True)


class Lake:
    """The process's handle on the lake."""

    def __init__(self, con: duckdb.DuckDBPyConnection, config: LakeConfig):
        self._con = con
        self.config = config
        #: `config.selection_limit` in bytes, parsed by DuckDB (the memory
        #: limit's own grammar).
        self.selection_bytes: int = con.execute(
            "SELECT parse_formatted_bytes(?)", [config.selection_limit]).fetchone()[0]
        self._reads = threading.Semaphore(config.max_concurrent_reads)
        self._process_locks: dict[str, threading.Lock] = {}
        self._process_locks_guard = threading.Lock()
        self._content_ids: dict[int, str] = {}

    @classmethod
    def open(cls, config: LakeConfig) -> "Lake":
        """Attach the lake and apply the settings. The tables are the
        store's (`store.ensure_schema`, applied once at startup by the
        composition root); the lake knows no schema."""
        for d in config.private_dirs():
            private_dir(d)
        try:
            con = duckdb.connect(config={
                "threads": config.threads,
                "memory_limit": config.memory_limit,
                "preserve_insertion_order": False,
                "TimeZone": "UTC",
                "temp_directory": os.path.join(config.spill_dir, "spill"),
            })
            con.execute("INSTALL ducklake; LOAD ducklake")
            dsn = config.postgres_dsn
            if dsn is not None:
                con.execute("INSTALL postgres; LOAD postgres")
            if config.data.startswith("s3://"):
                con.execute("INSTALL httpfs; LOAD httpfs")
                s3 = config.s3 or {}
                endpoint = s3["endpoint"]
                use_ssl = not endpoint.startswith("http://")
                endpoint = endpoint.split("://", 1)[-1].rstrip("/")
                con.execute(
                    "CREATE OR REPLACE SECRET bench_lake_s3 (TYPE s3, KEY_ID ?, SECRET ?, "
                    "REGION ?, ENDPOINT ?, URL_STYLE 'path', USE_SSL ?)",
                    [s3["key"], s3["secret"], s3["region"], endpoint, use_ssl])
            # ATTACH takes no prepared parameters: the catalog and the data
            # path are operator configuration, quoted as SQL literals.
            if dsn is not None:
                con.execute(
                    f"ATTACH {_sql_text('ducklake:postgres:' + dsn)} AS lake "
                    f"(DATA_PATH {_sql_text(config.data)}, METADATA_SCHEMA 'bench_lake')")
                con.execute(f"ATTACH {_sql_text(dsn)} AS bench_lock (TYPE postgres)")
            else:
                con.execute(f"ATTACH {_sql_text('ducklake:' + config.catalog)} AS lake "
                            f"(DATA_PATH {_sql_text(config.data)})")
            _set_lake_option(con, "target_file_size", "32MB", "32000000")
            lake = cls(con, config)
            memory = con.execute("SELECT parse_formatted_bytes(?)",
                                 [config.memory_limit]).fetchone()[0]
        except duckdb.Error as e:
            raise LakeUnavailable(f"cannot attach the bench lake: {e}") from e
        if lake.selection_bytes >= memory:
            con.close()
            raise LakeUnavailable(
                f"BENCH_LAKE_SELECTION_LIMIT ({config.selection_limit}) must be below "
                f"BENCH_LAKE_MEMORY_LIMIT ({config.memory_limit}): the held selection "
                "table lives in that memory")
        return lake

    def read(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        """One read on a fresh cursor, under the read semaphore; rows as
        dicts. SQL is built by `browse` / `sets` from whitelisted names only;
        values always go in `params`."""
        with self._reads:
            cur = self._con.cursor()
            try:
                cur.execute(sql, list(params))
                names = [d[0] for d in cur.description]
                return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]
            finally:
                cur.close()

    def read_batches(self, sql: str, params: Sequence[Any] = (),
                     *, batch_rows: int) -> Iterator[list[dict]]:
        """`read` without materializing the answer: `fetchmany(batch_rows)`
        lists of row dicts, one read cursor held for the generator's life.
        A bounded caller's peak is bounded by its batch, not the lake."""
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        with self._reads:
            cur = self._con.cursor()
            try:
                cur.execute(sql, list(params))
                names = [d[0] for d in cur.description]
                while rows := cur.fetchmany(batch_rows):
                    yield [dict(zip(names, row, strict=True))
                           for row in rows]
            finally:
                cur.close()

    def transact(self, fn: Callable[[Any], T]) -> T:
        """Run `fn(cursor)` in one lake transaction (BEGIN … COMMIT); on a
        commit conflict roll back and run `fn` again, up to 3 attempts with
        jitter, then raise `CommitConflict`. `fn` must be idempotent."""
        last: Optional[BaseException] = None
        for attempt in range(1, _ATTEMPTS + 1):
            cur = self._con.cursor()
            try:
                cur.execute("BEGIN")
                try:
                    out = fn(cur)
                except BaseException:
                    cur.execute("ROLLBACK")
                    raise
                cur.execute("COMMIT")      # a failed COMMIT has rolled back
                return out
            except duckdb.Error as e:
                if not is_commit_conflict(e):
                    raise
                last = e
            finally:
                cur.close()
            if attempt < _ATTEMPTS:
                time.sleep(random.uniform(0.05, 0.2) * attempt)
        raise CommitConflict(f"lost to concurrent commits {_ATTEMPTS} times: {last}") from last

    @contextmanager
    def _advisory(self, name: str, *, wait: bool) -> Iterator[bool]:
        """Hold the lock `name` for the block (see the module docstring);
        with `wait=False` yield whether it was free instead of waiting."""
        if self.config.postgres_dsn is None:
            with self._process_locks_guard:
                lock = self._process_locks.setdefault(name, threading.Lock())
            held = lock.acquire(blocking=wait)
            try:
                yield held
            finally:
                if held:
                    lock.release()
            return
        fn = "pg_advisory_xact_lock" if wait else "pg_try_advisory_xact_lock"
        cur = self._con.cursor()
        try:
            cur.execute("BEGIN")
            try:
                held = cur.execute(
                    f"SELECT ok FROM postgres_query('bench_lock', "
                    f"'SELECT {fn}({_advisory_key(name)})::text AS ok')").fetchone()[0]
                yield held in ("true", "")    # the blocking form returns void
            finally:
                cur.execute("ROLLBACK")        # ends the Postgres transaction: unlocks
        finally:
            cur.close()

    @contextmanager
    def owner_lock(self, owner: str) -> Iterator[None]:
        """Serialize one owner's uploads across processes: an advisory lock
        on a dedicated connection to the catalog database (Postgres catalog),
        a process lock otherwise. Held across the identity anti-join and the
        commit."""
        with self._advisory(f"bench:owner:{owner}", wait=True):
            yield

    @contextmanager
    def inputs_lock(self) -> Iterator[None]:
        """Serialize registering new `inputs` entries across processes, so
        two uploads (of any owners) never both add the same input (nor its
        in-memory annotations, `store.annotate_all`). Taken after
        `owner_lock` when an upload brings inputs the lake lacks."""
        with self._advisory("bench:inputs", wait=True):
            yield

    @contextmanager
    def maintainer_lock(self) -> Iterator[bool]:
        """Try to become the one maintainer (non-blocking advisory lock);
        yields whether this process holds it."""
        with self._advisory("bench:maintainer", wait=False) as held:
            yield held

    @contextmanager
    def backup_lock(self) -> Iterator[None]:
        """Serialize backups with each other and with DuckLake maintenance
        (which rewrites and deletes files a backup may be reading); the
        maintainer takes it inside `maintainer_lock`."""
        with self._advisory("bench:backup", wait=True):
            yield

    def free_bytes(self) -> Optional[int]:
        """Free space on the lake's volume; None when it has no local
        directory."""
        where = self.config.root or (None if self.config.data.startswith("s3://")
                                     else self.config.data)
        return None if where is None else shutil.disk_usage(where).free

    def require_free(self, what: str) -> None:
        """Refuse `what` (a write, maintenance, a backup) with
        `LakeUnavailable` when the volume has under `MIN_FREE_BYTES` free."""
        free = self.free_bytes()
        if free is not None and free < MIN_FREE_BYTES:
            raise LakeUnavailable(f"the lake's volume has {free // 2**20} MB free, under "
                                  f"{MIN_FREE_BYTES // 2**20} MB: {what} refused")

    def snapshot_id(self) -> int:
        """The lake's current snapshot — a cache key for anything derived
        from the lake (a lookup's pin, the measurement scope)."""
        return self.read("SELECT id FROM lake.current_snapshot()")[0]["id"]

    def _catalog(self) -> str:
        """The schema of the lake's DuckLake catalog tables, as attached."""
        schema = "bench_lake" if self.config.postgres_dsn is not None else "main"
        return f"__ducklake_metadata_lake.{schema}"

    def content_id(self, snapshot: int) -> str:
        """The identity of the lake's row and table content at `snapshot`
        (the module docstring): a hash of the snapshot and the catalog rows
        live at it."""
        got = self._content_ids.get(snapshot)
        if got is not None:
            return got
        m = self._catalog()
        n = int(snapshot)
        live = f"begin_snapshot <= {n} AND (end_snapshot IS NULL OR end_snapshot > {n})"
        lines = [f"snapshot|{n}"]          # a merged file's visible part (docstring)
        lines += [f"{r['k']}|{r['v']}" for r in self.read(
            f"SELECT 'table' AS k, CAST(r AS VARCHAR) AS v FROM (SELECT * EXCLUDE "
            f"(begin_snapshot, end_snapshot) FROM {m}.ducklake_table WHERE {live}) r "
            f"UNION ALL SELECT 'column', CAST(r AS VARCHAR) FROM (SELECT * EXCLUDE "
            f"(begin_snapshot, end_snapshot) FROM {m}.ducklake_column WHERE {live}) r "
            f"UNION ALL SELECT 'data', table_id || '/' || path FROM {m}.ducklake_data_file "
            f"WHERE {live} UNION ALL SELECT 'delete', table_id || '/' || path "
            f"FROM {m}.ducklake_delete_file WHERE {live}")]
        for t in self.read(f"SELECT table_id, table_name FROM {m}.ducklake_inlined_data_tables"):
            lines += [f"inlined {t['table_id']}|{r['v']}" for r in self.read(
                f"SELECT CAST(r AS VARCHAR) AS v FROM (SELECT * EXCLUDE (begin_snapshot, "
                f"end_snapshot) FROM {m}.{t['table_name']} WHERE {live}) r")]
        for t in self.read(f"SELECT DISTINCT table_id FROM {m}.ducklake_table"):
            lines += [f"inlined delete {t['table_id']}|{v}"
                      for v in self._inlined_deletions(int(t["table_id"]), n)]
        got = hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()[:32]
        with self._process_locks_guard:
            if len(self._content_ids) >= _CONTENT_IDS_KEPT:
                self._content_ids.pop(next(iter(self._content_ids)))
            self._content_ids[snapshot] = got
        return got

    def _inlined_deletions(self, table_id: int, snapshot: int) -> list[str]:
        """The inlined deletions of rows in `table_id`'s data files live at
        `snapshot`, as `file_id/row_id`. DuckLake keeps them in
        `ducklake_inlined_delete_<table_id>`, made at the table's first
        inlined deletion and listed nowhere: its catalog database is hidden
        from `duckdb_tables()` and `information_schema`, and a second attach
        of a file catalog is refused (DuckDB 1.5.6), so its existence cannot
        be asked first. The binder's refusal to name exactly that table is
        the answer "none yet"; any other error is raised."""
        table = f"ducklake_inlined_delete_{table_id}"
        try:
            rows = self.read(f"SELECT file_id || '/' || row_id AS v FROM "
                             f"{self._catalog()}.{table} WHERE begin_snapshot <= {int(snapshot)}")
        except duckdb.CatalogException as e:
            if f"Table with name {table} does not exist" not in str(e):
                raise
            return []
        return [r["v"] for r in rows]

    def close(self) -> None:
        self._con.close()


_LAKE: Optional[Lake] = None
_LAKE_GUARD = threading.Lock()


def get_lake() -> Lake:
    """The process-wide lake, opened from the environment on first use
    (raises `LakeUnavailable` when unconfigured or unreachable)."""
    global _LAKE
    with _LAKE_GUARD:
        if _LAKE is None:
            _LAKE = Lake.open(LakeConfig.from_env())
        return _LAKE
