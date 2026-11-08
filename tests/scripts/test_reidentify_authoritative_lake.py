from __future__ import annotations

import json
import uuid

import tools.reidentify_authoritative_lake as reidentify
from backend.service.bench import data_rules, store
from backend.service.bench.lake import Lake, LakeConfig


def _open(path) -> Lake:
    return Lake.open(LakeConfig(catalog=str(path / "catalog.ducklake"),
                                data=str(path / "data")))


def test_a_replacement_copy_gets_a_new_authoritative_identity(tmp_path, capsys):
    source = _open(tmp_path / "source")
    old_id = str(uuid.uuid4())
    try:
        store.ensure_schema(source, lake_id=old_id, import_state="verified",
                            lake_role="authority")
        data_rules.initialize(source)
    finally:
        source.close()
    new_id = uuid.uuid4()

    rc = reidentify.main([
        "--source-catalog", str(tmp_path / "source" / "catalog.ducklake"),
        "--source-data", str(tmp_path / "source" / "data"),
        "--destination", str(tmp_path / "replacement"),
        "--lake-id", str(new_id)])

    assert rc == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["old_lake_id"] == old_id and answer["lake_id"] == str(new_id)
    copy = _open(tmp_path / "replacement")
    try:
        meta = store.meta(copy)
        assert meta["lake_id"] == str(new_id)
        assert meta["lake_role"] == "authority"
        assert meta["supersedes_lake_id"] == old_id
    finally:
        copy.close()
    # The source lineage is untouched.
    original = _open(tmp_path / "source")
    try:
        assert store.meta(original)["lake_id"] == old_id
    finally:
        original.close()


def test_reusing_the_source_identity_is_refused_before_a_copy_is_made(tmp_path):
    source = _open(tmp_path / "source")
    old_id = str(uuid.uuid4())
    try:
        store.ensure_schema(source, lake_id=old_id, import_state="verified",
                            lake_role="authority")
        data_rules.initialize(source)
    finally:
        source.close()
    try:
        reidentify.main([
            "--source-catalog", str(tmp_path / "source" / "catalog.ducklake"),
            "--source-data", str(tmp_path / "source" / "data"),
            "--destination", str(tmp_path / "replacement"),
            "--lake-id", old_id])
    except SystemExit as e:
        assert "new lake_id" in str(e)
    else:
        raise AssertionError("the same identity was accepted")
    assert not (tmp_path / "replacement").exists()
