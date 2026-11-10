import gzip
import json
import tarfile

import pytest

from backend.code_trace.provider import TRACES
from backend.gallery_cache_artifact import GALLERY_CACHE
from backend.fingerprint import engine_version
from backend.measure.cache import KeyedCache


def test_safe_archive_rejects_non_json(tmp_path):
    archive = tmp_path / "cache.tar.gz"
    source = tmp_path / "source" / "gallery" / "entry.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source.parent, arcname="gallery")
    assert GALLERY_CACHE.check(archive) == 1

    bad = tmp_path / "bad.tar.gz"
    payload = tmp_path / "bad-source" / "gallery" / "entry.txt"
    payload.parent.mkdir(parents=True)
    payload.write_text("x", encoding="utf-8")
    with tarfile.open(bad, "w:gz") as tar:
        tar.add(payload.parent, arcname="gallery")
    try:
        GALLERY_CACHE.check(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("non-JSON archive accepted")


def test_install_merges_artifact_into_runtime_cache(tmp_path):
    archive = tmp_path / "cache.tar.gz"
    source = tmp_path / "source" / "gallery"
    source.mkdir(parents=True)
    (source / "facts.json").write_text("{}", encoding="utf-8")
    (source / "frontier").mkdir()
    (source / "frontier" / "entry.json.gz").write_bytes(gzip.compress(b"{}"))
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="gallery")

    root = tmp_path / "runtime-gallery"
    (root / "runtime-only").mkdir(parents=True)
    (root / "runtime-only" / "local.json").write_text("{}", encoding="utf-8")
    (root / "frontier-views").mkdir()
    (root / "frontier-views" / "stale.json").write_text("{}", encoding="utf-8")
    installed = GALLERY_CACHE.install(
        archive, root, "gallery-cache-v2/test/cache.tar.gz", "digest")

    assert installed == 2
    assert (root / "facts.json").exists()
    assert (root / "frontier" / "entry.json.gz").exists()
    assert (root / "runtime-only" / "local.json").exists()
    assert not (root / "frontier-views" / "stale.json").exists()
    marker = json.loads((root / ".bootstrap.json").read_text(encoding="utf-8"))
    assert marker == {
        "artifact": "gallery-cache-v2/test/cache.tar.gz",
        "sha256": "digest",
        "files": 2,
        "traces": [],
    }
    assert GALLERY_CACHE.installed(root, "gallery-cache-v2/test/cache.tar.gz", "digest")


def test_a_member_outside_its_tiers_format_is_refused(tmp_path):
    """A frontier entry is the frontier tier's file (`.json.gz`); a plain
    `.json` there — what a v1 archive held — is not an entry of this
    artifact."""
    source = tmp_path / "source" / "gallery"
    (source / "frontier").mkdir(parents=True)
    (source / "facts.json").write_text("{}", encoding="utf-8")
    (source / "frontier" / "entry.json").write_text("{}", encoding="utf-8")
    archive = tmp_path / "v1.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="gallery")
    with pytest.raises(ValueError, match="declared tier"):
        GALLERY_CACHE.check(archive)


def test_publish_then_install_serves_compressed_frontier_entries(
        tmp_path, monkeypatch):
    """What a warm machine's tiers hold, a fresh machine (no trace bodies)
    installs and serves: the frontier entries ship as the very gzip files
    the warm machine wrote, and the build finds their traces through the
    tier's own backend."""
    source = tmp_path / "warm" / "gallery"
    facts = KeyedCache("gallery", GALLERY_CACHE.entries[0].disk(source))
    frontier_tier = next(t for t in GALLERY_CACHE.entries
                         if t.path == "frontier")
    assert frontier_tier.compress
    frontier = KeyedCache("gallery-frontier", frontier_tier.disk(source))
    values = {f"{i:064x}": {"points": [[i, j, j * 0.25] for j in range(200)]}
              for i in range(3)}
    for key, value in values.items():
        frontier.get_or_compute(key, lambda value=value: value)
    facts.get_or_compute("f" * 64, lambda: {"model": "m"})
    views = tmp_path / "warm" / "gallery" / "frontier-views"
    views.mkdir()
    (views / "view.json").write_text("{}", encoding="utf-8")

    archive = tmp_path / "artifact.tar.gz"
    assert GALLERY_CACHE.build(source, archive) == 4
    assert GALLERY_CACHE.check(archive) == 4
    with tarfile.open(archive, "r:gz") as tar:
        members = {m.name: tar.extractfile(m).read()
                   for m in tar.getmembers() if m.isfile()}
    for key in values:
        name = f"gallery/frontier/{key}.json.gz"
        assert members[name] == (source / "frontier" / f"{key}.json.gz"
                                 ).read_bytes()
    assert not any("frontier-views" in name for name in members)
    assert sum(name.startswith("gallery/code-traces/") for name in members) >= 1

    root = tmp_path / "fresh" / "gallery"
    monkeypatch.setattr(TRACES, "directory", str(tmp_path / "fresh-traces"))
    monkeypatch.setattr(TRACES, "_mem", {})
    reader = KeyedCache("gallery-frontier", frontier_tier.disk(root))
    assert reader.get(next(iter(values))) is None
    assert GALLERY_CACHE.install(archive, root, "a.tar.gz", "sha") == 4
    for key, value in values.items():
        assert reader.get(key) == value
    assert GALLERY_CACHE.installed(root, "a.tar.gz", "sha")


def test_index_selects_an_immutable_digest_addressed_artifact():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({
                "schema": 1,
                "engine_version": engine_version(),
                "artifacts": [{
                    "path": "gallery-cache-v1/engine/digest.tar.gz",
                    "sha256": "digest",
                }],
            }).encode()

    from backend.release_artifacts import selected_artifact

    assert selected_artifact(
        "gallery-cache-v1/engine/index.json", engine_version(),
        "https://supabase", "key", "sol-pool", 30,
        request=lambda url, key, timeout: Response()) == (
        "gallery-cache-v1/engine/digest.tar.gz", "digest")


def test_index_for_another_engine_is_refused():
    from backend.release_artifacts import selected_artifact

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"schema":1,"engine_version":"other","artifacts":[]}'

    try:
        selected_artifact(
            "gallery-cache-v1/engine/index.json", engine_version(),
            "https://supabase", "key", "sol-pool", 30,
            request=lambda url, key, timeout: Response())
    except ValueError as error:
        assert "another engine version" in str(error)
    else:
        raise AssertionError("a foreign engine index was accepted")
