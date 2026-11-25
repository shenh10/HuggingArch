"""A file that exists but is corrupt must raise, not be silently conflated
with an absent file.

Covers the two infra-file readers hardened in the silent_except C-cluster:
`snapshots.load_snapshot` (prefetched config/metadata store) and
`hf.metadata.fetch_metadata` (custom-model tensor index). A missing file is
still a legitimate, quiet "absent" — only corruption is loud.
"""

import pytest


def test_load_snapshot_raises_on_corrupt_file(tmp_path, monkeypatch):
    from backend.arch import snapshots

    bad = tmp_path / "snap.json"
    bad.write_text("{ this is not: valid json ]")
    monkeypatch.setattr(snapshots, "snapshot_path", lambda model_id: bad)
    # No overlay in play.
    monkeypatch.delenv("HA_SPEC_OVERLAY_DIR", raising=False)

    with pytest.raises(ValueError, match="unreadable"):
        snapshots.load_snapshot("whatever")


def test_load_snapshot_absent_file_is_quiet(tmp_path, monkeypatch):
    from backend.arch import snapshots

    missing = tmp_path / "nope.json"
    monkeypatch.setattr(snapshots, "snapshot_path", lambda model_id: missing)
    monkeypatch.delenv("HA_SPEC_OVERLAY_DIR", raising=False)

    assert snapshots.load_snapshot("whatever") is None


def test_fetch_metadata_raises_on_corrupt_custom_index(tmp_path, monkeypatch):
    from backend.hf import metadata

    record = tmp_path / "checkpoint_header.json"
    record.write_text("{ broken ]")
    monkeypatch.setattr(metadata, "is_custom_id", lambda mid: True)
    monkeypatch.setattr(metadata, "strip_prefix", lambda mid: mid)
    monkeypatch.setattr(metadata, "model_path", lambda mid: tmp_path)
    monkeypatch.setattr(metadata, "registered_on_host", lambda mid: True)

    with pytest.raises(ValueError, match="corrupt"):
        metadata.fetch_metadata("custom:whatever")


def test_fetch_metadata_absent_custom_index_is_quiet(tmp_path, monkeypatch):
    from backend.hf import metadata

    monkeypatch.setattr(metadata, "is_custom_id", lambda mid: True)
    monkeypatch.setattr(metadata, "strip_prefix", lambda mid: mid)
    monkeypatch.setattr(metadata, "model_path", lambda mid: tmp_path)  # no file
    monkeypatch.setattr(metadata, "registered_on_host", lambda mid: True)

    out = metadata.fetch_metadata("custom:whatever")
    assert out == {}
