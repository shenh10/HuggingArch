"""The backup protocol (backend/service/bench/backup.py, docs/measured_calibration.md §4.7) on a
local lake: the bucket is a local directory; the S3 client is checked
against AWS's published signature and a mock transport."""
from __future__ import annotations

import datetime as _dt
import json
import threading
import time
import uuid

import httpx
import pytest

from backend.service.bench import backup as backup_mod
from backend.service.bench import maintain, store
from backend.service.bench.backup import Backup, LocalBucket, S3Bucket, Trigger, restore, sigv4_headers
from backend.service.bench.lake import Lake

_TABLES = {"batches": "batch_id", "inputs": "input_id", "measurements": "id"}


class RecordingBucket(LocalBucket):
    """A local bucket that records every put, in order."""

    def __init__(self, root):
        super().__init__(root)
        self.puts: list[str] = []

    def put(self, key, path):
        self.puts.append(key)
        super().put(key, path)


def _up(lake, owner, rows, annotate):
    return store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()), run_id="r",
                        label="l", rows=rows)


def _dump(lake):
    return {t: lake.read(f"SELECT * FROM lake.{t} ORDER BY {k}") for t, k in _TABLES.items()}


def _json(bucket, key):
    return json.loads((bucket.root / key).read_text())


def test_data_first_then_catalog_then_manifest_then_current(tmp_path, lake, make_rows, annotate,
                                                            new_owner):
    bucket = RecordingBucket(str(tmp_path / "bucket"))
    _up(lake, new_owner(), make_rows(20), annotate)      # over 10 rows: a data file, not inlined
    out = Backup(bucket).run(lake)
    v = lake.snapshot_id()
    manifest = _json(bucket, f"snapshots/v{v}/manifest.json")
    files = manifest["files"]
    assert len(files) == 2                               # inputs and measurements
    assert out == {"snapshot": v, "uploaded": 2}
    assert bucket.puts == [f"data/{f['sha256']}" for f in files] + [
        f"snapshots/v{v}/catalog.ducklake", f"snapshots/v{v}/manifest.json", "current"]
    for f in files:
        data_file = tmp_path / "data" / f["path"]
        assert f["sha256"] == backup_mod._sha256(data_file) and f["size"] == data_file.stat().st_size
    assert manifest["counts"] == {"batches": 1, "inputs": 20, "measurements": 20}
    assert manifest["catalog"]["sha256"] == backup_mod._sha256(
        bucket.root / f"snapshots/v{v}/catalog.ducklake")
    assert _json(bucket, "current") == {"snapshot": v, "manifest": f"snapshots/v{v}/manifest.json"}
    # nothing written since: nothing to do
    bucket.puts.clear()
    assert Backup(bucket).run(lake) is None and bucket.puts == []


def test_data_objects_are_content_addressed_and_manifests_never_overwritten(
        tmp_path, lake, make_rows, annotate, new_owner):
    bucket = RecordingBucket(str(tmp_path / "bucket"))
    b = Backup(bucket)
    _up(lake, new_owner(), make_rows(20), annotate)
    b.run(lake)
    first = [k for k in bucket.puts if k.startswith("data/")]
    bucket.puts.clear()
    _up(lake, new_owner(), make_rows(30, gpu="H200"), annotate)
    out = b.run(lake)
    v = lake.snapshot_id()
    assert out["uploaded"] >= 1                          # only the new files
    assert not set(first) & set(bucket.puts)
    assert set(first) <= {f"data/{f['sha256']}" for f in
                          _json(bucket, f"snapshots/v{v}/manifest.json")["files"]}
    # `current` lost (an interrupted backup): the next run republishes it
    # without rewriting the snapshot's objects
    (bucket.root / "current").unlink()
    bucket.puts.clear()
    assert Backup(bucket).run(lake) == {"snapshot": lake.snapshot_id(), "uploaded": 0}
    assert bucket.puts == ["current"]


def test_restore_round_trip_and_checks(tmp_path, lake, make_rows, annotate, new_owner):
    bucket = LocalBucket(str(tmp_path / "bucket"))
    b = Backup(bucket)
    owner = new_owner()
    a = _up(lake, owner, make_rows(20), annotate)
    b.run(lake)
    early = lake.snapshot_id()
    early_dump = _dump(lake)
    for _ in range(3):
        _up(lake, new_owner(), make_rows(30), annotate)
    maintain.housekeeping(lake)          # merged files: the catalog drops the ones it merged
    store.delete_rows(lake, owner=owner, ids=[r["id"] for r in lake.read(
        "SELECT id::VARCHAR AS id FROM lake.measurements WHERE batch_id = ? LIMIT 12",
        [a.batch_id])])                  # a delete file
    b.run(lake)
    want = _dump(lake)
    config = restore(bucket, str(tmp_path / "restored"))
    restored = Lake.open(config)
    try:
        assert _dump(restored) == want
    finally:
        restored.close()
    with pytest.raises(RuntimeError, match="empty directory"):
        restore(bucket, str(tmp_path / "restored"))
    # an older snapshot, by name
    old = Lake.open(restore(bucket, str(tmp_path / "early"), snapshot=early))
    try:
        assert _dump(old) == early_dump
    finally:
        old.close()
    # a damaged object is refused
    manifest = _json(bucket, _json(bucket, "current")["manifest"])
    (bucket.root / f"data/{manifest['files'][0]['sha256']}").write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="sha256"):
        restore(bucket, str(tmp_path / "damaged"))


def test_a_failed_restore_leaves_no_catalog_and_the_next_refuses_naming_it(
        tmp_path, lake, make_rows, annotate, new_owner):
    """The catalog takes its name only once the restored lake opened and its
    counts matched: a restore failing its check leaves no `catalog.ducklake`
    (which the next start would open as a lake), and the next restore
    refuses the directory, naming what is in it."""
    bucket = LocalBucket(str(tmp_path / "bucket"))
    _up(lake, new_owner(), make_rows(20), annotate)
    Backup(bucket).run(lake)
    key = _json(bucket, "current")["manifest"]
    manifest = _json(bucket, key)
    manifest["counts"]["measurements"] += 1
    (bucket.root / key).write_text(json.dumps(manifest))
    dest = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="rows"):
        restore(bucket, str(dest))
    assert not (dest / "catalog.ducklake").exists()
    with pytest.raises(RuntimeError, match=r"empty directory.*catalog\.ducklake\.restoring"):
        restore(bucket, str(dest))


def test_gc_keeps_by_manifest(tmp_path, lake, make_rows, annotate, new_owner, monkeypatch):
    bucket = LocalBucket(str(tmp_path / "bucket"))
    b = Backup(bucket)
    now = _dt.datetime(2026, 10, 9, 12, tzinfo=_dt.timezone.utc)
    ages = [_dt.timedelta(days=40), _dt.timedelta(days=20, hours=2), _dt.timedelta(days=20),
            _dt.timedelta(days=10), _dt.timedelta(days=3), _dt.timedelta(hours=1)]
    versions = []
    for k, age in enumerate(ages):
        monkeypatch.setattr(backup_mod, "_now", lambda age=age: now - age)
        _up(lake, new_owner(), make_rows(20, gpu=f"G{k}"), annotate)
        versions.append(b.run(lake)["snapshot"])
    # an interrupted backup (a catalog object, no manifest) and an orphan data object
    (bucket.root / "snapshots/v9999").mkdir()
    (bucket.root / "snapshots/v9999/catalog.ducklake").write_bytes(b"x")
    (bucket.root / "data/deadbeef").write_bytes(b"x")
    monkeypatch.setattr(backup_mod, "_now", lambda: now)
    out = b.gc(lake, now)
    # 40 days: gone; day 20 keeps its last; 10 days, 3 days, 1 hour (current) kept
    assert out["snapshots_removed"] == [versions[0], versions[1]]
    kept = {int(k.split("/")[1][1:]) for k in bucket.list("snapshots/")}
    assert kept == set(versions[2:])
    assert out["data_removed"] == 1 and not (bucket.root / "data/deadbeef").exists()
    for v in kept:                       # every kept snapshot is whole
        m = _json(bucket, f"snapshots/v{v}/manifest.json")
        assert (bucket.root / m["catalog"]["key"]).exists()
        assert all((bucket.root / f"data/{f['sha256']}").exists() for f in m["files"])


def test_status_reports_lag_and_the_last_error(tmp_path, lake, make_rows, annotate, new_owner,
                                               monkeypatch):
    b = Backup(LocalBucket(str(tmp_path / "bucket")))
    assert b.status()["lag_seconds"] == 0.0 and b.status()["last_error"] is None
    _up(lake, new_owner(), make_rows(2), annotate)
    b.note_write()
    time.sleep(0.05)
    assert b.status()["lag_seconds"] > 0
    monkeypatch.setattr(b.bucket, "put", lambda k, p: (_ for _ in ()).throw(OSError("bucket down")))
    with pytest.raises(OSError):
        b.run(lake)
    st = b.status()
    assert "bucket down" in st["last_error"] and st["lag_seconds"] > 0 and not b.succeeded
    monkeypatch.undo()
    b.run(lake)
    st = b.status()
    assert st["last_error"] is None and st["lag_seconds"] == 0.0 and b.succeeded
    assert st["newest_recoverable_snapshot"] == lake.snapshot_id()


def test_writes_wait_for_a_backup_unconfigured_failing_or_behind(tmp_path, lake, make_rows,
                                                                 annotate, new_owner, monkeypatch):
    assert "not configured" in Backup(None).refusal()
    with pytest.raises(RuntimeError, match="not configured"):
        Backup(None).run(lake)
    b = Backup(LocalBucket(str(tmp_path / "bucket")))
    assert "startup backup" in b.refusal()               # no run yet in this process
    b.run(lake)
    assert b.refusal() is None
    for _ in range(backup_mod.MAX_COMMITS):
        b.note_write()
    assert b.refusal() is None
    b.note_write()
    assert f"more than {backup_mod.MAX_COMMITS} writes" in b.refusal()
    _up(lake, new_owner(), make_rows(2), annotate)
    b.run(lake)
    assert b.refusal() is None and b.status()["writes_behind"] == 0
    b.note_write()
    later = backup_mod._now() + backup_mod.MAX_LAG + _dt.timedelta(seconds=1)
    monkeypatch.setattr(backup_mod, "_now", lambda: later)
    assert "5 minutes behind" in b.refusal()


def test_backup_staging_needs_free_space(tmp_path, lake, monkeypatch):
    from collections import namedtuple

    from backend.service.bench import lake as lake_mod
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(lake_mod.shutil, "disk_usage", lambda p: usage(0, 0, 10 * 2**20))
    b = Backup(LocalBucket(str(tmp_path / "bucket")))
    with pytest.raises(Exception, match="backup staging refused"):
        b.run(lake)
    assert "backup staging refused" in b.refusal()


def test_trigger_coalesces_and_retries(lake, monkeypatch):
    runs: list[int] = []
    gate = threading.Event()
    done = threading.Event()

    class Fake(Backup):
        def run(self, lk):
            runs.append(len(runs))
            if len(runs) == 1:
                gate.wait(5)             # the first run is slow: pokes meanwhile coalesce
            if len(runs) == 2:
                raise RuntimeError("transient")
            if len(runs) >= 3:
                done.set()

    monkeypatch.setattr(Trigger, "MIN_RETRY", 0.01)
    t = Trigger(Fake(LocalBucket("/nonexistent")), lake)
    t.start()
    t.poke()
    time.sleep(0.1)
    for _ in range(5):
        t.poke()
    gate.set()
    assert done.wait(5)
    time.sleep(0.1)
    assert len(runs) == 3                # one, one coalesced (failed), its retry


# --- the S3 client -----------------------------------------------------------------------

def test_sigv4_matches_the_aws_example():
    # "Example: GET Object", AWS S3 API reference (Signature Version 4).
    h = sigv4_headers(
        "GET", "/test.txt", {"host": "examplebucket.s3.amazonaws.com", "range": "bytes=0-9"},
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        key="AKIAIOSFODNN7EXAMPLE", secret="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1", now=_dt.datetime(2013, 5, 24, tzinfo=_dt.timezone.utc))
    assert h["authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")


def test_s3_bucket_requests(tmp_path):
    seen = []
    pages = {
        None: ("<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
               "<IsTruncated>true</IsTruncated><NextContinuationToken>t/2</NextContinuationToken>"
               "<Contents><Key>root/data/a</Key></Contents></ListBucketResult>"),
        "t/2": ("<ListBucketResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">"
                "<IsTruncated>false</IsTruncated>"
                "<Contents><Key>root/data/b c</Key></Contents></ListBucketResult>"),
    }

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.raw_path.decode(), req.headers.get("authorization", "")))
        if req.method == "GET" and req.url.params.get("list-type") == "2":
            assert req.url.params["prefix"] == "root/data/"
            return httpx.Response(200, text=pages[req.url.params.get("continuation-token")])
        if req.method == "GET":
            return httpx.Response(200, content=b"payload")
        return httpx.Response(204 if req.method == "DELETE" else 200)

    b = S3Bucket(endpoint="https://s3.example.test", region="auto", key="k", secret="s",
                 bucket="bench-lake-backup", prefix="root/",
                 client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert b.list("data/") == ["data/a", "data/b c"]
    src = tmp_path / "f"
    src.write_bytes(b"payload")
    b.put("data/b c", src)
    b.get("data/b c", tmp_path / "g")
    b.delete("data/b c")
    assert (tmp_path / "g").read_bytes() == b"payload"
    methods = [(m, p) for m, p, _ in seen[2:]]
    assert methods == [(m, "/bench-lake-backup/root/data/b%20c") for m in ("PUT", "GET", "DELETE")]
    assert all(a.startswith("AWS4-HMAC-SHA256 Credential=k/") for _, _, a in seen)


def test_s3_endpoint_with_a_path_signs_the_full_path(tmp_path):
    """Supabase Storage's S3 endpoint carries a path (`/storage/v1/s3`): it
    belongs to every request path and to the signed canonical URI, and the
    host header is the network location alone (the live bucket refused the
    signature while the path was folded into the host)."""
    import datetime as dt

    from backend.service.bench.backup import sigv4_headers
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, content=b"x")

    b = S3Bucket(endpoint="https://ref.storage.supabase.co/storage/v1/s3", region="ap-northeast-1",
                 key="k", secret="s", bucket="bench-lake-backup", prefix="",
                 client=httpx.Client(transport=httpx.MockTransport(handler)))
    b.get("current", tmp_path / "c")
    req = seen[0]
    assert req.url.host == "ref.storage.supabase.co"
    assert req.url.raw_path.decode() == "/storage/v1/s3/bench-lake-backup/current"
    stamp = dt.datetime.strptime(req.headers["x-amz-date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)
    want = sigv4_headers("GET", "/storage/v1/s3/bench-lake-backup/current",
                         {"host": "ref.storage.supabase.co"}, req.headers["x-amz-content-sha256"],
                         key="k", secret="s", region="ap-northeast-1", now=stamp)
    assert req.headers["authorization"] == want["authorization"]
