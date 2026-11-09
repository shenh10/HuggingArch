import json
import tarfile

from backend.release_artifacts import digest as _digest
from backend.sol_pool_artifact import _has_bootstrap, _unpack


def test_digest_reads_hash_object_by_chunks(tmp_path):
    archive = tmp_path / "pool.tar.gz"
    archive.write_bytes(b"0123456789" * 100000)
    assert _digest(archive) == _digest(archive)


def test_unpack_replaces_a_partial_runtime_pool(tmp_path, monkeypatch):
    archive = tmp_path / "pool.tar.gz"
    bundle = tmp_path / "canonical" / "ab" / "bundle-id"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "canonical", arcname="sol")

    root = tmp_path / "sol-pool"
    partial = root / "zz" / "runtime-bundle"
    partial.mkdir(parents=True)
    (partial / "manifest.json").write_text("{}", encoding="utf-8")

    _unpack(archive, root, "serving-law-v1/test/pool.tar.gz", "digest")

    manifests = list(root.glob("*/*/manifest.json"))
    assert len(manifests) == 1
    assert manifests[0].parent.name == "bundle-id"
    marker = json.loads((root / ".bootstrap.json").read_text(encoding="utf-8"))
    assert marker == {
        "artifact": "serving-law-v1/test/pool.tar.gz",
        "sha256": "digest",
        "manifests": 1,
    }
    assert _has_bootstrap(
        root, "serving-law-v1/test/pool.tar.gz", "digest")
    assert not _has_bootstrap(root, "serving-law-v1/test/pool.tar.gz", "other")
