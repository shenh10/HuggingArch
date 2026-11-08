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
                 run_id="r", label=None, rows=rows, annotate=annotate,
                 status="shared")
    store.upload(lake, owner=new_owner(), upload_id=str(uuid.uuid4()),
                 run_id="r", label=None, rows=[rows[1]], annotate=annotate,
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
