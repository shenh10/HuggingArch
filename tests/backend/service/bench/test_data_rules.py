"""The authoritative lake's registered data-format migrations."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from backend.service.bench import data_rules, store


def test_the_registered_chain_reaches_the_live_rule_code():
    """A key or validator edit must be registered, not left to fork a lake."""
    from backend.service.bench.rules_revision import rules_rev

    manifest = json.loads(
        (Path(data_rules.__file__).parent / "data_rules_version.json").read_text())
    chain = manifest["migrations"]
    assert chain
    assert [step["from"] for step in chain[1:]] == [step["to"] for step in chain[:-1]]
    assert data_rules.registered_version() == chain[-1]["to"] == rules_rev()
    for step in chain:
        if step.get("rekey_plan"):
            assert (Path(data_rules.__file__).parent / "migrations"
                    / step["rekey_plan"]).is_file()


def test_an_unknown_data_rules_version_refuses_to_migrate(lake):
    store.set_meta(lake, data_rules="from-the-future")
    assert not data_rules.known_version("from-the-future")
    with pytest.raises(data_rules.DataRulesError, match="outside"):
        data_rules.migrate(lake)
    assert store.meta(lake)["data_rules"] == "from-the-future"


def test_the_legacy_migration_is_idempotent_and_removes_refused_shared_rows(
        lake, make_rows, annotate, new_owner):
    rows = make_rows(2)
    rows[0]["op"] = "removed_op"
    owner = new_owner()
    store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()),
                 run_id="r", label=None, rows=rows,
                 status="shared")
    store.upload(lake, owner=new_owner(), upload_id=str(uuid.uuid4()),
                 run_id="r", label=None, rows=[rows[1]],
                 status="shared")
    rows_now = lake.read("SELECT id, op FROM lake.measurements")
    kept_id = next(row["id"] for row in rows_now if row["op"] != "removed_op")

    result = data_rules.migrate(lake)
    meta = store.meta(lake)
    assert result.version == data_rules.registered_version()
    assert meta["data_rules"] == data_rules.registered_version()
    assert json.loads(meta["data_rules_migration"]) == result.report()
    assert result.dropped_shared_rows == 2
    assert result.reasons["duplicate shared row identity"] == 1
    assert sum(result.reasons.values()) == 2
    kept = lake.read("SELECT id FROM lake.measurements ORDER BY id")
    assert [row["id"] for row in kept] == [kept_id]
    assert lake.read("SELECT count(*) AS n FROM lake.batches")[0]["n"] == 1

    again = data_rules.migrate(lake)
    assert (again.rekeyed_rows, again.env_rekeyed_rows,
            again.dropped_shared_rows) == (0, 0, 0)
    assert len(lake.read("SELECT id FROM lake.measurements")) == 1


def test_a_current_lake_is_not_rewritten(lake):
    data_rules.initialize(lake)
    before = lake.read("SELECT * FROM lake.measurements")
    result = data_rules.migrate(lake)
    assert result.version == data_rules.registered_version()
    assert result.report()["rekeyed_rows"] == 0
    assert lake.read("SELECT * FROM lake.measurements") == before


def test_paged_storage_identity_migration_preserves_failed_attempts(
        lake, make_rows, new_owner, monkeypatch):
    """A now-supported storage layout does not retroactively create a timing.

    Preserve both a measured row and the old paged-indexer mismatch, including
    their ids and evidence. Only the lake revision changes; derived annotations
    rebuild independently under the new provider catalog.
    """
    from backend.inference import benchmarks

    rows = make_rows(1)
    failed = {column: "" for column in benchmarks.ALL_COLUMNS}
    failed.update(
        model="moonshotai/Kimi-K3.1-Pro", spec_hash="test", phase="decode",
        parallel="tp=8", framework="deepgemm", framework_version="0.2.8",
        gpu="B300", op="indexer_logits", node_spec_id="indexer.logits",
        shape_json=json.dumps({"B": 1, "s_query": 1, "s_kv": 257,
                               "n_indexer_heads": 32, "d_indexer_head": 128,
                               "kv_block_size": 64}, sort_keys=True),
        dtype_env_json=json.dumps({"activation": "fp8_e4m3", "kv": "fp8_e4m3",
                                   "kv_format": "fp8_group128", "output": "fp32"},
                                  sort_keys=True),
        kernel_impl="deepgemm_fp8_paged_mqa_logits",
        schema_version=str(benchmarks.KNOWN_SCHEMA_VERSION), contributor="test",
        date="2026-11-13", cross_check="unknown", outcome="mismatch",
        outcome_detail="launch gate: uint8 packed cache was not declared",
        launch_signature="fp8_paged_mqa_logits(uint8[5,64,1,132])")
    benchmarks.validate_row(failed)
    rows.append(failed)
    store.upload(lake, owner=new_owner(), upload_id=str(uuid.uuid4()),
                 run_id="old-storage-check", label=None, rows=rows, status="shared")
    before = lake.read("SELECT * FROM lake.measurements ORDER BY id")
    step = next(s for s in data_rules._MIGRATIONS
                if s["from"] == "a2b93a8d956f7699")
    assert step["reason"].startswith("Identity migration:")
    assert not any(step.get(name) for name in (
        "rekey_plan", "rekey_env", "validate_rows", "deduplicate_shared_rows"))

    def no_row_mutation(*args, **kwargs):
        pytest.fail("identity migration must not rewrite or promote stored attempts")

    monkeypatch.setattr(data_rules, "rekey_apply", no_row_mutation)
    monkeypatch.setattr(data_rules, "_invalid_rows", no_row_mutation)
    monkeypatch.setattr(data_rules, "_duplicate_shared_rows", no_row_mutation)
    monkeypatch.setattr(store, "rekey_env", no_row_mutation)
    store.set_meta(lake, data_rules=step["from"])
    result = data_rules.migrate(lake)
    assert result.version == data_rules.registered_version()
    assert result.report() == {"rekeyed_rows": 0, "env_rekeyed_rows": 0,
                               "dropped_shared_rows": 0, "reasons": {}}
    assert lake.read("SELECT * FROM lake.measurements ORDER BY id") == before
    assert sum(row["outcome"] == "mismatch" for row in before) == 1
