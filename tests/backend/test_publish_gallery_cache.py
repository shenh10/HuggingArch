import gzip
import tarfile

from scripts.publish_gallery_cache import build


def test_gallery_artifact_excludes_measurement_views(tmp_path):
    root = tmp_path / "gallery"
    (root / "frontier").mkdir(parents=True)
    (root / "frontier-views").mkdir()
    (root / "facts.json").write_text("{}", encoding="utf-8")
    (root / "frontier" / "entry.json.gz").write_bytes(gzip.compress(b"{}"))
    (root / "frontier-views" / "stale.json").write_text("{}", encoding="utf-8")
    archive = tmp_path / "cache.tar.gz"

    assert build(root, archive) == 2
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "gallery/facts.json" in names
    assert "gallery/frontier/entry.json.gz" in names
    assert all(name != "gallery/frontier-views/stale.json" for name in names)
