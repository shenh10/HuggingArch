import json
import tarfile

from backend.gallery_cache_artifact import GALLERY_CACHE
from backend.fingerprint import engine_version


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
    (source / "frontier" / "entry.json").write_text("{}", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="gallery")

    root = tmp_path / "runtime-gallery"
    (root / "runtime-only").mkdir(parents=True)
    (root / "runtime-only" / "local.json").write_text("{}", encoding="utf-8")
    (root / "frontier-views").mkdir()
    (root / "frontier-views" / "stale.json").write_text("{}", encoding="utf-8")
    installed = GALLERY_CACHE.install(
        archive, root, "gallery-cache-v1/test/cache.tar.gz", "digest")

    assert installed == 2
    assert (root / "facts.json").exists()
    assert (root / "frontier" / "entry.json").exists()
    assert (root / "runtime-only" / "local.json").exists()
    assert not (root / "frontier-views" / "stale.json").exists()
    marker = json.loads((root / ".bootstrap.json").read_text(encoding="utf-8"))
    assert marker == {
        "artifact": "gallery-cache-v1/test/cache.tar.gz",
        "sha256": "digest",
        "files": 2,
        "traces": [],
    }
    assert GALLERY_CACHE.installed(root, "gallery-cache-v1/test/cache.tar.gz", "digest")


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
