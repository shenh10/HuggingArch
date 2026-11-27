"""The sweep cache's release artifact, end to end against a stand-in for
Supabase Storage (`_Storage`: the object GET and the resumable TUS upload,
nothing else): what a warm machine publishes, a machine with an empty cache
directory and no trace bodies pulls — and then serves as cache hits. A
tampered archive fails the pull and installs nothing."""
from __future__ import annotations

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import ClassVar

import pytest

from backend import release_artifacts
from backend.fingerprint import cache_engine_version
from backend.code_trace.provider import TRACES
from backend.measure.cache import DiskBackend, KeyedCache
from backend.sweep_cache_artifact import SWEEP_CACHE


class _Storage(BaseHTTPRequestHandler):
    objects: ClassVar[dict[str, bytes]] = {}
    uploads: ClassVar[dict[str, dict]] = {}
    fail_next_patch = 0
    fail_next_get = 0
    fail_next_index_get = 0
    patches = 0

    def log_message(self, *args):
        return None

    def _reply(self, code, body=b"", headers=None):
        self.send_response(code)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        prefix = "/storage/v1/object/"
        name = self.path[len(prefix):]
        if not self.path.startswith(prefix) or name not in self.objects:
            self._reply(400, b'{"code":"NoSuchKey"}')
            return
        if type(self).fail_next_index_get and self.path.endswith("/index.json"):
            type(self).fail_next_index_get -= 1
            self._reply(500, b"transient")
            return
        if type(self).fail_next_get and ".part-" in self.path:
            type(self).fail_next_get -= 1
            self._reply(500, b"transient")
            return
        self._reply(200, self.objects[name])

    def do_POST(self):
        assert self.path == "/storage/v1/upload/resumable"
        self._body()
        meta = dict(item.split(" ") for item in
                    self.headers["Upload-Metadata"].split(","))
        meta = {k: base64.b64decode(v).decode() for k, v in meta.items()}
        uid = str(len(self.uploads))
        self.uploads[uid] = {
            "name": f"{meta['bucketName']}/{meta['objectName']}",
            "length": int(self.headers["Upload-Length"]), "data": b""}
        self._reply(201, headers={
            "Location": f"/storage/v1/upload/resumable/{uid}"})

    def _upload(self):
        return self.uploads[self.path.rsplit("/", 1)[1]]

    def do_HEAD(self):
        self._reply(200, headers={
            "Upload-Offset": str(len(self._upload()["data"]))})

    def do_PATCH(self):
        upload, data = self._upload(), self._body()
        type(self).patches += 1
        if self.fail_next_patch:
            type(self).fail_next_patch -= 1
            self._reply(500, b"transient")
            return
        if int(self.headers["Upload-Offset"]) != len(upload["data"]):
            self._reply(409, b"offset")
            return
        upload["data"] += data
        if len(upload["data"]) == upload["length"]:
            self.objects[upload["name"]] = upload["data"]
        self._reply(204, headers={
            "Upload-Offset": str(len(upload["data"]))})


@pytest.fixture
def storage(monkeypatch):
    _Storage.objects, _Storage.uploads = {}, {}
    _Storage.fail_next_patch = 0
    _Storage.fail_next_get, _Storage.fail_next_index_get = 0, 0
    _Storage.patches = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Storage)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
                 "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("SUPABASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    monkeypatch.setenv("HA_ARTIFACT_BUCKET", "sol-pool")
    monkeypatch.setenv("HA_SWEEP_CACHE_ARTIFACT", "")
    # Several PATCHes per archive, and a retry that does not wait.
    monkeypatch.setattr(release_artifacts, "TUS_CHUNK", 4096)
    monkeypatch.setattr(release_artifacts.time, "sleep", lambda s: None)
    yield _Storage
    server.shutdown()


def _warm(root, n=40):
    """A warm machine's sweep directory: answers stored by `KeyedCache`
    (each with its trace), and a step that must not ship."""
    cache = KeyedCache("sweep", DiskBackend(root))
    keys = [f"{i:064x}" for i in range(n)]
    for i, key in enumerate(keys):
        cache.get_or_compute(key, lambda i=i: {
            "points": [[i, j, j * 0.5] for j in range(60)]})
    (root / "steps").mkdir()
    (root / "steps" / "unfinished.json").write_text("{}", encoding="utf-8")
    return keys


def test_projection_entries_never_enter_the_release_artifact(tmp_path):
    """The shared rows are a machine-local derived tier, not an answer file."""
    source = tmp_path / "warm"
    keys = _warm(source, n=1)
    projection = DiskBackend(source / "projection", compress=True)
    projection.put("rows", {"points": keys})

    files = SWEEP_CACHE.files(source)

    assert (source / "projection" / "rows.json.gz").is_file()
    assert all(file.relative_to(source).parts[0] != "projection"
               for file in files)


def test_installing_an_artifact_clears_the_machine_local_projection(
        tmp_path, monkeypatch, storage):
    source = tmp_path / "warm"
    _warm(source, n=1)
    path = SWEEP_CACHE.publish(source)
    assert path.startswith("sweep-cache-v1/")

    root = _fresh_machine(tmp_path, monkeypatch, "projection-fresh")
    projection = SWEEP_CACHE.disk("projection")
    projection.put("rows", {"points": []})
    assert (root / "projection" / "rows.json.gz").is_file()

    assert SWEEP_CACHE.pull(timeout=30) == 0
    assert not (root / "projection" / "rows.json.gz").exists()


def _fresh_machine(tmp_path, monkeypatch, name):
    """An empty cache directory and a trace store that has never seen the
    publishing machine's traces."""
    root = tmp_path / name / "sweep-cache"
    monkeypatch.setenv("HA_SWEEP_CACHE_DIR", str(root))
    monkeypatch.setattr(TRACES, "directory", str(tmp_path / name / "traces"))
    monkeypatch.setattr(TRACES, "_mem", {})
    return root


def test_publish_then_pull_serves_the_published_keys(
        tmp_path, monkeypatch, storage):
    source = tmp_path / "warm"
    keys = _warm(source)
    storage.fail_next_patch = 1
    path = SWEEP_CACHE.publish(source)
    assert path.startswith("sweep-cache-v1/") and path.endswith(".tar.gz")
    assert storage.patches > 3  # resumable, in chunks, one retried

    root = _fresh_machine(tmp_path, monkeypatch, "fresh")
    reader = KeyedCache("sweep", DiskBackend(root))
    assert reader.get(keys[0]) is None  # cold before the pull
    assert SWEEP_CACHE.pull(timeout=30) == 0

    for i, key in enumerate(keys):
        assert reader.get(key) == {
            "points": [[i, j, j * 0.5] for j in range(60)]}
    assert not (root / "steps").exists()
    marker = json.loads((root / ".bootstrap.json").read_text())
    assert marker["artifact"] == path and marker["files"] == len(keys)
    # A second start finds it installed and downloads nothing.
    gets = len(storage.objects)
    assert SWEEP_CACHE.installed(root, path, marker["sha256"])
    assert SWEEP_CACHE.pull(timeout=30) == 0 and len(storage.objects) == gets


def test_an_archive_above_the_object_limit_publishes_ordered_parts(
        tmp_path, monkeypatch, storage):
    """The storage limit is transport, not a new artifact format.

    Index and pin still name one digest-addressed tar.gz.  Only the bytes move
    as ordered parts, and a fresh machine transparently reassembles and checks
    the whole archive before installing it.
    """
    source = tmp_path / "warm"
    keys = _warm(source)
    monkeypatch.setenv("HA_ARTIFACT_OBJECT_MAX_BYTES", "4096")
    path = SWEEP_CACHE.publish(source)

    index = json.loads(storage.objects[
        f"sol-pool/sweep-cache-v1/{cache_engine_version()}"
        "/index.json"])
    entry = index["artifacts"][0]
    assert index["schema"] == release_artifacts.SHARDED_INDEX_SCHEMA
    assert entry["path"] == path and len(entry["parts"]) > 1
    assert f"sol-pool/{path}" not in storage.objects
    assert all(part["size"] <= 4096 for part in entry["parts"])
    whole = b"".join(storage.objects[f"sol-pool/{part['path']}"]
                     for part in entry["parts"])
    assert hashlib.sha256(whole).hexdigest() == entry["sha256"]

    root = _fresh_machine(tmp_path, monkeypatch, "sharded-fresh")
    reader = KeyedCache("sweep", DiskBackend(root))
    assert SWEEP_CACHE.pull(timeout=30) == 0
    assert all(reader.get(key) is not None for key in keys)
    marker = json.loads((root / ".bootstrap.json").read_text())
    assert marker["artifact"] == path and marker["sha256"] == entry["sha256"]


def test_transient_part_downloads_are_retried(
        tmp_path, monkeypatch, storage):
    source = tmp_path / "warm"
    keys = _warm(source, n=30)
    monkeypatch.setenv("HA_ARTIFACT_OBJECT_MAX_BYTES", "4096")
    SWEEP_CACHE.publish(source)
    storage.fail_next_get = 2

    root = _fresh_machine(tmp_path, monkeypatch, "transient-download")
    assert SWEEP_CACHE.pull(timeout=30) == 0
    reader = KeyedCache("sweep", DiskBackend(root))
    assert all(reader.get(key) is not None for key in keys)


def test_a_transient_index_download_is_retried(
        tmp_path, monkeypatch, storage):
    source = tmp_path / "warm"
    keys = _warm(source, n=3)
    SWEEP_CACHE.publish(source)
    storage.fail_next_index_get = 1

    root = _fresh_machine(tmp_path, monkeypatch, "transient-index")
    assert SWEEP_CACHE.pull(timeout=30) == 0
    reader = KeyedCache("sweep", DiskBackend(root))
    assert all(reader.get(key) is not None for key in keys)


def test_a_corrupt_part_fails_before_install(tmp_path, monkeypatch, storage):
    source = tmp_path / "warm"
    _warm(source, n=30)
    monkeypatch.setenv("HA_ARTIFACT_OBJECT_MAX_BYTES", "4096")
    SWEEP_CACHE.publish(source)
    index_name = (
        f"sol-pool/sweep-cache-v1/{cache_engine_version()}"
        "/index.json")
    part = json.loads(storage.objects[index_name])["artifacts"][0]["parts"][0]
    data = bytearray(storage.objects[f"sol-pool/{part['path']}"])
    data[-1] ^= 0xFF
    storage.objects[f"sol-pool/{part['path']}"] = bytes(data)

    root = _fresh_machine(tmp_path, monkeypatch, "corrupt-part")
    with pytest.raises(ValueError, match="artifact part digest mismatch"):
        SWEEP_CACHE.pull(timeout=30)
    assert not root.exists() or not list(root.glob("*.json"))


def test_a_digest_mismatch_fails_closed(tmp_path, monkeypatch, storage):
    source = tmp_path / "warm"
    keys = _warm(source, n=3)
    path = SWEEP_CACHE.publish(source)
    name = f"sol-pool/{path}"
    tampered = bytearray(storage.objects[name])
    tampered[len(tampered) // 2] ^= 0xFF
    storage.objects[name] = bytes(tampered)

    root = _fresh_machine(tmp_path, monkeypatch, "fresh")
    with pytest.raises(ValueError, match="digest mismatch"):
        SWEEP_CACHE.pull(timeout=30)
    assert KeyedCache("sweep", DiskBackend(root)).get(keys[0]) is None
    assert not root.exists() or not list(root.glob("*.json"))
