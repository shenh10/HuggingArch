"""Backups of the Kernel Bench lake (docs/measured_calibration.md §4.7):
they leave the machine, into their own bucket with their own key.

Configuration (environment):
  BENCH_LAKE_BACKUP_DATA   `s3://bench-lake-backup/<prefix>` or a local directory
  BENCH_LAKE_BACKUP_S3_ENDPOINT / _REGION / _KEY / _SECRET   for s3:// — an
                           access key of the backup bucket only, not the
                           service-role key
  BENCH_LAKE_BACKUP=off     development only: no backups, and nothing waits
                           for them
Backups are mandatory and fail-closed (docs/measured_calibration.md §4.7): writes are
refused while the backup is unconfigured (BENCH_LAKE_BACKUP_DATA unset and
no explicit `off`), failing, or more than `MAX_LAG` / `MAX_COMMITS` behind
(`Backup.refusal`); DuckLake expiry / cleanup and the retention below wait
while it is behind (maintain.py). /health reports the newest recoverable
snapshot and the lag.

Layout under the backup root:
  data/<sha256>                    every data and delete file a snapshot's
                                   catalog references, by content; never
                                   overwritten (an existing key is skipped)
  snapshots/v<v>/catalog.ducklake  the catalog copy at lake snapshot v
  snapshots/v<v>/manifest.json     the snapshot: the catalog object and every
                                   data object (path in the data directory,
                                   size, sha256), row counts at v
  current                          {"snapshot": v, "manifest": key}

One backup (`Backup.run`), under the lake's process-wide backup lock (the
maintainer holds it while DuckLake maintenance rewrites and deletes files):
  1. the catalog at a snapshot v: `COPY FROM DATABASE` of the attached
     catalog into a staging file, one transaction (a commit landing meanwhile
     is wholly in it or not); v is the copy's latest snapshot. No CHECKPOINT:
     DuckLake's CHECKPOINT compacts and rewrites files;
  2. the files the copy references, uploaded first under `data/<sha256>`;
  3. then the catalog object, then the manifest — the snapshot exists once
     its manifest does; a manifest is never overwritten (a catalog object
     without a manifest is an interrupted backup and may be replaced);
  4. `current` last.
It runs after every committed write (`Trigger`: coalesced, off the request
path), every maintainer cycle, and at startup before writes are accepted.
A write is acknowledged after the local commit; its backup follows within
seconds (RPO: losing the volume in that window loses those writes; a crash
loses nothing — startup backs up what `current` lacks).

Retention (`Backup.gc`, by manifests, never by listing alone): every
manifest of the last 7 days, the last one of each day for 30 days, and the
one `current` names; then every data object no kept manifest names goes.

Restore (`restore`): `current` (or a named snapshot) → manifest → objects,
each checked against its sha256, at the exact paths; the catalog pointed at
the new data directory; opened; row counts checked against the manifest.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol
from urllib.parse import quote, urlsplit

import duckdb
import httpx

from backend.service.bench import store
from backend.service.bench.lake import Lake, LakeConfig, _sql_text, private_dir

_LOG = logging.getLogger(__name__)

MAX_LAG = _dt.timedelta(minutes=5)      # writes wait when the oldest unbacked write is older
MAX_COMMITS = 20                        # … or more writes than this are not backed up
KEEP_ALL = _dt.timedelta(days=7)        # every manifest this recent is kept
KEEP_DAILY = _dt.timedelta(days=30)     # then the last of each day, this far back
_METADATA = "__ducklake_metadata_lake"  # DuckLake's catalog database of the attach `lake`
_MANIFEST = re.compile(r"^snapshots/v(\d+)/manifest\.json$")


# --------------------------------------------------------------------- bucket

class Bucket(Protocol):
    """Object storage, keys relative to the backup root ('/'-separated)."""

    def put(self, key: str, path: Path) -> None: ...
    def get(self, key: str, path: Path) -> None: ...
    def list(self, prefix: str) -> list[str]: ...
    def delete(self, key: str) -> None: ...


class LocalBucket:
    """A directory standing in for a bucket (development, tests)."""

    def __init__(self, root: str):
        self.root = Path(root)

    def put(self, key: str, path: Path) -> None:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(path, tmp)
        os.replace(tmp, dest)

    def get(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / key, path)

    def list(self, prefix: str) -> list[str]:
        """Keys starting with `prefix`, as S3 lists them."""
        base = self.root / prefix.rpartition("/")[0]
        if not base.is_dir():
            return []
        keys = (p.relative_to(self.root).as_posix() for p in base.rglob("*")
                if p.is_file() and not p.name.endswith(".part"))
        return sorted(k for k in keys if k.startswith(prefix))

    def delete(self, key: str) -> None:
        (self.root / key).unlink(missing_ok=True)


def sigv4_headers(method: str, url: str, headers: dict[str, str], payload_hash: str, *,
                  key: str, secret: str, region: str, now: _dt.datetime) -> dict[str, str]:
    """AWS Signature V4 for an S3 request: `headers` (which must include
    `host`) plus `x-amz-date`, `x-amz-content-sha256` and `Authorization`.
    `url` is the path with its query string, already encoded."""
    path, _, query = url.partition("?")
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    day = stamp[:8]
    hdrs = {k.lower(): v.strip() for k, v in headers.items()}
    hdrs["x-amz-date"] = stamp
    hdrs["x-amz-content-sha256"] = payload_hash
    params = sorted(tuple(p.split("=", 1)) if "=" in p else (p, "")
                    for p in query.split("&") if p)
    signed = ";".join(sorted(hdrs))
    canonical = "\n".join([
        method, path or "/", "&".join(f"{k}={v}" for k, v in params),
        "".join(f"{k}:{hdrs[k]}\n" for k in sorted(hdrs)), signed, payload_hash])
    scope = f"{day}/{region}/s3/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", stamp, scope,
                         hashlib.sha256(canonical.encode()).hexdigest()])
    k = f"AWS4{secret}".encode()
    for part in (day, region, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    hdrs["authorization"] = (f"AWS4-HMAC-SHA256 Credential={key}/{scope}, "
                             f"SignedHeaders={signed}, Signature={signature}")
    return hdrs


def _q(s: str) -> str:
    return quote(s, safe="-_.~")


class S3Bucket:
    """An S3-compatible bucket (Cloudflare R2, Supabase Storage) over
    path-style requests signed with SigV4. `prefix` is the backup root
    inside the bucket."""

    def __init__(self, *, endpoint: str, region: str, key: str, secret: str,
                 bucket: str, prefix: str = "", client: Optional[httpx.Client] = None):
        endpoint = endpoint.rstrip("/")
        if "://" not in endpoint:
            endpoint = "https://" + endpoint
        parts = urlsplit(endpoint)
        # The endpoint may carry a path (Supabase Storage: /storage/v1/s3):
        # it is part of every request's path and of the signed canonical URI;
        # the host header is the network location alone.
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self.host = parts.netloc
        self.base = parts.path.rstrip("/")
        self.region, self.key, self.secret = region, key, secret
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client or httpx.Client(timeout=120.0)

    def _object(self, key: str) -> str:
        full = f"{self.prefix}/{key}" if self.prefix else key
        return f"/{_q(self.bucket)}/" + "/".join(_q(p) for p in full.split("/"))

    def _send(self, method: str, url: str, body: bytes = b"", stream: bool = False):
        url = self.base + url
        headers = sigv4_headers(method, url, {"host": self.host},
                                hashlib.sha256(body).hexdigest(), key=self.key,
                                secret=self.secret, region=self.region,
                                now=_dt.datetime.now(_dt.timezone.utc))
        headers.pop("host")
        req = self.client.build_request(method, self.origin + url, headers=headers,
                                        content=body or None)
        r = self.client.send(req, stream=stream)
        if r.status_code >= 300 and not (method == "DELETE" and r.status_code == 404):
            text = r.read().decode(errors="replace")[:300]
            r.close()
            raise RuntimeError(f"S3 {method} {url}: {r.status_code} {text}")
        return r

    def put(self, key: str, path: Path) -> None:
        self._send("PUT", self._object(key), path.read_bytes()).close()

    def get(self, key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        r = self._send("GET", self._object(key), stream=True)
        try:
            with open(path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        finally:
            r.close()

    def list(self, prefix: str) -> list[str]:
        root = f"{self.prefix}/" if self.prefix else ""
        out: list[str] = []
        token: Optional[str] = None
        ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
        while True:
            q = {"list-type": "2", "prefix": root + prefix}
            if token:
                q["continuation-token"] = token
            url = f"/{_q(self.bucket)}?" + "&".join(f"{_q(k)}={_q(v)}" for k, v in sorted(q.items()))
            r = self._send("GET", url)
            tree = ET.fromstring(r.content)
            out.extend(c.findtext(f"{ns}Key")[len(root):] for c in tree.iter(f"{ns}Contents"))
            if tree.findtext(f"{ns}IsTruncated") != "true":
                return sorted(out)
            token = tree.findtext(f"{ns}NextContinuationToken")

    def delete(self, key: str) -> None:
        self._send("DELETE", self._object(key)).close()


@dataclass(frozen=True)
class BackupConfig:
    root: str                      # s3://bucket/prefix or a local directory
    s3: Optional[dict] = None      # endpoint, region, key, secret

    @classmethod
    def from_env(cls) -> Optional["BackupConfig"]:
        """None when BENCH_LAKE_BACKUP_DATA is unset (backups disabled)."""
        root = os.environ.get("BENCH_LAKE_BACKUP_DATA", "").strip()
        if not root:
            return None
        s3 = None
        if root.startswith("s3://"):
            s3 = {k: os.environ.get(f"BENCH_LAKE_BACKUP_S3_{k.upper()}", "").strip()
                  for k in ("endpoint", "region", "key", "secret")}
            missing = [f"BENCH_LAKE_BACKUP_S3_{k.upper()}" for k, v in s3.items() if not v]
            if missing:
                raise RuntimeError(f"an s3:// backup location needs {', '.join(missing)}")
        return cls(root=root, s3=s3)

    def bucket(self) -> Bucket:
        if self.s3 is None:
            return LocalBucket(self.root)
        bucket, _, prefix = self.root[len("s3://"):].partition("/")
        return S3Bucket(bucket=bucket, prefix=prefix, **self.s3)


# ------------------------------------------------------------------ helpers

def _statements(lake: Lake, *statements: str) -> list:
    """Statements on one lake cursor in one transaction; the last one's rows."""
    def go(cur):
        out = []
        for s in statements:
            out = cur.execute(s).fetchall()
        return out
    return lake.transact(go)


def _referenced_files(catalog_copy: Path) -> list[str]:
    """The data and delete files a catalog copy references, as paths
    relative to its data path."""
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH {_sql_text(str(catalog_copy))} AS m (READ_ONLY)")
        rows = con.execute(
            "WITH t AS (SELECT DISTINCT t.table_id, CASE WHEN t.path_is_relative THEN"
            " (CASE WHEN s.path_is_relative THEN s.path ELSE NULL END) || t.path END AS dir"
            " FROM m.ducklake_table t JOIN m.ducklake_schema s USING (schema_id)),"
            " f AS (SELECT table_id, path, path_is_relative FROM m.ducklake_data_file"
            " UNION ALL SELECT table_id, path, path_is_relative FROM m.ducklake_delete_file)"
            " SELECT DISTINCT f.path_is_relative AND t.dir IS NOT NULL AS rel,"
            " CASE WHEN f.path_is_relative THEN t.dir || f.path ELSE f.path END"
            " FROM f JOIN t USING (table_id) ORDER BY 2").fetchall()
    finally:
        con.close()
    absolute = [p for rel, p in rows if not rel]
    if absolute:
        raise RuntimeError(f"the catalog references files outside its data path: {absolute[:3]}")
    return [p for _, p in rows]


def _latest_snapshot(catalog: Path) -> int:
    """The latest snapshot a catalog file holds."""
    con = duckdb.connect()
    try:
        con.execute(f"ATTACH {_sql_text(str(catalog))} AS m (READ_ONLY)")
        return int(con.execute("SELECT max(snapshot_id) FROM m.ducklake_snapshot").fetchone()[0])
    finally:
        con.close()


def _set_data_path(catalog: Path, data: str) -> None:
    con = duckdb.connect(str(catalog))
    try:
        con.execute("UPDATE ducklake_metadata SET value = ? WHERE key = 'data_path'",
                    [data.rstrip("/") + "/"])
    finally:
        con.close()


# ------------------------------------------------------------------ backups

def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_key(v: int, name: str) -> str:
    return f"snapshots/v{int(v)}/{name}"


class Backup:
    """The lake's backups into one bucket (None: not configured — every run
    fails, so writes wait), and their state for the gate and /health."""

    def __init__(self, bucket: Optional[Bucket]):
        self.bucket = bucket
        self._guard = threading.Lock()
        self._hashes: dict[tuple, str] = {}        # (path, size, mtime_ns) → sha256
        self._manifests: dict[str, dict] = {}      # manifest key → manifest
        self.backed_up: Optional[int] = None       # the snapshot `current` names
        self.succeeded = False                     # a run has succeeded in this process
        self.pending_since: Optional[_dt.datetime] = None   # oldest write not backed up
        self.pending_writes = 0                    # writes not backed up
        self.last_success_at: Optional[_dt.datetime] = None
        self.last_error: Optional[str] = None
        self.last_error_at: Optional[_dt.datetime] = None

    @staticmethod
    def supported(lake: Lake) -> bool:
        """Backups copy a file catalog and local data files (docs/measured_calibration.md §4.1)."""
        return lake.config.postgres_dsn is None and not lake.config.data.startswith("s3://")

    # ---- state

    def note_write(self) -> None:
        """A write committed: the backup is behind until the next run."""
        with self._guard:
            self.pending_writes += 1
            if self.pending_since is None:
                self.pending_since = _now()

    def refusal(self) -> Optional[str]:
        """Why writes must wait for the backup, None when they need not."""
        now = _now()
        with self._guard:
            if self.bucket is None:
                return ("backups are not configured (BENCH_LAKE_BACKUP_DATA); "
                        "BENCH_LAKE_BACKUP=off opts out in development")
            if self.last_error is not None:
                return f"the backup is failing ({self.last_error})"
            if not self.succeeded:
                return "the startup backup has not completed"
            if self.pending_since is not None and now - self.pending_since > MAX_LAG:
                return f"the backup is more than {MAX_LAG.seconds // 60} minutes behind"
            if self.pending_writes > MAX_COMMITS:
                return f"the backup is more than {MAX_COMMITS} writes behind"
        return None

    def status(self) -> dict:
        """The newest recoverable snapshot, the lag (the dynamic RPO) and
        the last error, for /health."""
        now = _now()
        with self._guard:
            return {
                "configured": self.bucket is not None,
                "newest_recoverable_snapshot": (self.backed_up if self.backed_up is not None
                                                and self.backed_up >= 0 else None),
                "lag_seconds": (round((now - self.pending_since).total_seconds(), 1)
                                if self.pending_since else 0.0),
                "writes_behind": self.pending_writes,
                "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
                "last_error": self.last_error,
                "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
            }

    def current(self) -> Optional[dict]:
        """The `current` pointer, None when there is none yet."""
        if self.bucket is None:
            raise RuntimeError("backups are not configured (BENCH_LAKE_BACKUP_DATA is not set)")
        if "current" not in self.bucket.list("current"):
            return None
        return self._read_json("current")

    def _read_json(self, key: str) -> dict:
        fd, name = tempfile.mkstemp(prefix="bench-backup-", suffix=".json")
        os.close(fd)
        try:
            self.bucket.get(key, Path(name))
            return json.loads(Path(name).read_text())
        finally:
            os.unlink(name)

    def _manifest(self, key: str) -> dict:
        if key not in self._manifests:
            self._manifests[key] = self._read_json(key)
        return self._manifests[key]

    # ---- one backup

    def behind(self, lake: Lake) -> bool:
        """Whether `current` names an older snapshot than the lake's
        (always, when backups are not configured)."""
        if self.bucket is None:
            return True
        if self.backed_up is None:
            cur = self.current()
            self.backed_up = int(cur["snapshot"]) if cur else -1
        return self.backed_up < lake.snapshot_id()

    def run(self, lake: Lake) -> Optional[dict]:
        """Back the lake up to its current snapshot (the protocol in the
        module docstring); None when `current` already names it. Records
        success or the error (and re-raises it)."""
        with self._guard:
            covered = self.pending_writes      # the writes this run will hold
        try:
            with lake.backup_lock():
                out = self._run(lake) if self.behind(lake) else None
        except Exception as e:
            with self._guard:
                self.last_error = f"{type(e).__name__}: {e}"[:500]
                self.last_error_at = _now()
            _LOG.exception("bench lake backup failed")
            raise
        with self._guard:
            self.succeeded = True
            self.last_success_at = _now()
            self.last_error = None
            self.pending_writes = max(0, self.pending_writes - covered)
            if self.pending_writes == 0:
                self.pending_since = None
        if out:
            _LOG.info("bench lake backup: %s", out)
        return out

    def _run(self, lake: Lake) -> dict:
        if self.bucket is None:
            raise RuntimeError("backups are not configured (BENCH_LAKE_BACKUP_DATA is not set)")
        if not self.supported(lake):
            raise RuntimeError("backups need a file catalog and a local data directory")
        lake.require_free("backup staging")
        staging = Path(lake.config.staging_dir)
        private_dir(str(staging))
        copy = staging / f"catalog-{uuid.uuid4().hex}.ducklake"
        alias = f"bench_backup_{uuid.uuid4().hex[:8]}"
        try:
            _statements(lake, f"ATTACH {_sql_text(str(copy))} AS {alias}",
                 f"COPY FROM DATABASE {_METADATA} TO {alias}")
            _statements(lake, f"DETACH {alias}")
            v = _latest_snapshot(copy)
            manifest_key = _snapshot_key(v, "manifest.json")
            uploaded = 0
            if not self.bucket.list(manifest_key):
                data = Path(lake.config.data)
                files = []
                for rel in _referenced_files(copy):
                    path = data / rel
                    files.append({"path": rel, "size": path.stat().st_size,
                                  "sha256": self._hash(path)})
                have = set(self.bucket.list("data/"))
                for f in files:                     # 2. data first
                    key = f"data/{f['sha256']}"
                    if key not in have:
                        self.bucket.put(key, data / f["path"])
                        have.add(key)
                        uploaded += 1
                counts = {t: int(_statements(lake, f"SELECT count(*) FROM lake.{t} AT (VERSION => {v})")[0][0])
                          for t in store.TABLES}
                catalog_key = _snapshot_key(v, "catalog.ducklake")
                self.bucket.put(catalog_key, copy)  # 3. the catalog, then the manifest
                manifest = {"snapshot": v, "created_at": _now().isoformat(),
                            "catalog": {"key": catalog_key, "size": copy.stat().st_size,
                                        "sha256": _sha256(copy)},
                            "files": files, "counts": counts}
                self._put_json(staging, manifest_key, manifest)
                self._manifests[manifest_key] = manifest
            self._put_json(staging, "current", {"snapshot": v, "manifest": manifest_key})  # 4.
            self.backed_up = v
            return {"snapshot": v, "uploaded": uploaded}
        finally:
            for p in (copy, Path(str(copy) + ".wal")):
                p.unlink(missing_ok=True)

    def _hash(self, path: Path) -> str:
        st = path.stat()
        k = (str(path), st.st_size, st.st_mtime_ns)
        if k not in self._hashes:
            self._hashes[k] = _sha256(path)
        return self._hashes[k]

    def _put_json(self, staging: Path, key: str, value: dict) -> None:
        tmp = staging / f"{uuid.uuid4().hex}.json"
        try:
            tmp.write_text(json.dumps(value, sort_keys=True))
            self.bucket.put(key, tmp)
        finally:
            tmp.unlink(missing_ok=True)

    # ---- retention

    def gc(self, lake: Lake, now: Optional[_dt.datetime] = None) -> dict:
        """Apply the retention (module docstring) under the backup lock;
        returns what went."""
        now = now or _now()
        with lake.backup_lock():
            keys = self.bucket.list("snapshots/")
            manifests = {int(m.group(1)): k for k in keys if (m := _MANIFEST.match(k))}
            cur = self.current()
            dated = {v: _dt.datetime.fromisoformat(self._manifest(k)["created_at"])
                     for v, k in manifests.items()}
            keep = {v for v, t in dated.items() if now - t <= KEEP_ALL}
            last_of_day: dict[_dt.date, int] = {}
            for v, t in dated.items():
                if now - t <= KEEP_DAILY:
                    d = t.astimezone(_dt.timezone.utc).date()
                    last_of_day[d] = max(v, last_of_day.get(d, v))
            keep |= set(last_of_day.values())
            if cur:
                keep.add(int(cur["snapshot"]))
            gone = sorted(set(manifests) - keep)
            for v in gone:                       # the manifest first: then it is no snapshot
                self.bucket.delete(manifests[v])
                self._manifests.pop(manifests[v], None)
            kept_prefixes = {f"snapshots/v{v}/" for v in keep}
            for k in keys:                       # catalogs of dropped or interrupted snapshots
                if not _MANIFEST.match(k) and not any(k.startswith(p) for p in kept_prefixes):
                    self.bucket.delete(k)
            referenced = {f"data/{f['sha256']}" for v in keep if v in manifests
                          for f in self._manifest(manifests[v])["files"]}
            orphans = [k for k in self.bucket.list("data/") if k not in referenced]
            for k in orphans:
                self.bucket.delete(k)
        return {"snapshots_removed": gone, "data_removed": len(orphans)}


class Trigger:
    """Backs the lake up off the request path: `poke()` after a committed
    write; pokes during a run coalesce into one more run; a failed run is
    retried with exponential backoff (5 s … 300 s)."""

    MIN_RETRY = 5.0
    MAX_RETRY = 300.0

    def __init__(self, backup: Backup, lake: Lake):
        self.backup, self.lake = backup, lake
        self._event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def poke(self) -> None:
        self.backup.note_write()
        self._event.set()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="bench-lake-backup", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        retry = self.MIN_RETRY
        while True:
            self._event.wait()
            self._event.clear()
            try:
                self.backup.run(self.lake)
            except Exception:                 # recorded by run(); tried again later
                time.sleep(retry)
                retry = min(retry * 2, self.MAX_RETRY)
                self._event.set()
            else:
                retry = self.MIN_RETRY


# ------------------------------------------------------------------ restore

def restore(bucket: Bucket, dest: str, *, snapshot: Optional[int] = None) -> LakeConfig:
    """Restore the snapshot `current` names (or `snapshot`) into the empty
    directory `dest`: `catalog.ducklake` and `data/` at the paths the
    manifest names, every object checked against its sha256; the catalog
    pointed at `dest/data/`; opened, and its row counts checked against the
    manifest. Returns the config to open the restored lake with.

    `dest` must be empty — a sanity check, not a policy: the service makes
    nothing of a lake's in a directory before a catalog has attached there
    (`Lake.open`), so anything present is state someone left, named in the
    refusal. The catalog is downloaded under a working name and takes its
    own name last, once the restored lake has opened and its counts match:
    a `catalog.ducklake` is always a verified restore, and a failed restore
    leaves pieces the next one refuses, never a lake the next start opens."""
    root = Path(dest)
    present = sorted(e.name for e in root.iterdir()) if root.exists() else []
    if present:
        raise RuntimeError(f"restore needs an empty directory, {dest} holds "
                           f"{', '.join(present)}")
    backup = Backup(bucket)
    if snapshot is None:
        cur = backup.current()
        if cur is None:
            raise RuntimeError("the backup has no `current` pointer")
        key = cur["manifest"]
    else:
        key = _snapshot_key(snapshot, "manifest.json")
    manifest = backup._read_json(key)
    private_dir(str(root))

    def fetch(obj_key: str, sha: str, path: Path) -> None:
        bucket.get(obj_key, path)
        got = _sha256(path)
        if got != sha:
            raise RuntimeError(f"{obj_key}: sha256 {got}, the manifest says {sha}")

    catalog = root / "catalog.ducklake"
    restoring = root / "catalog.ducklake.restoring"
    fetch(manifest["catalog"]["key"], manifest["catalog"]["sha256"], restoring)
    data = root / "data"
    private_dir(str(data))
    for f in manifest["files"]:
        target = (data / f["path"]).resolve()
        if data.resolve() not in target.parents:
            raise RuntimeError(f"the manifest names a path outside the data directory: {f['path']}")
        fetch(f"data/{f['sha256']}", f["sha256"], target)
    _set_data_path(restoring, str(data))
    if _latest_snapshot(restoring) != int(manifest["snapshot"]):
        raise RuntimeError("the catalog object is not at the manifest's snapshot")
    lake = Lake.open(LakeConfig(catalog=str(restoring), data=str(data)))
    try:
        for t in store.TABLES:
            n = lake.read(f"SELECT count(*) AS n FROM lake.{t}")[0]["n"]
            if n != manifest["counts"][t]:
                raise RuntimeError(f"restored lake.{t} has {n} rows, the manifest {manifest['counts'][t]}")
    finally:
        lake.close()
    wal = restoring.with_name(restoring.name + ".wal")
    if wal.exists():
        raise RuntimeError(f"the restored catalog left a write-ahead log after closing: {wal}")
    os.replace(restoring, catalog)
    return LakeConfig(catalog=str(catalog), data=str(data))
