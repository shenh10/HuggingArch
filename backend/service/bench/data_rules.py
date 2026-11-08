r"""Registered data-format migrations for the authoritative measurement lake.

The format of a lake row has one owner: the key and validator code named by
`rules_revision.rules_rev`.  A deployment cannot describe that revision by hand; it
registers the chain of migrations it knows how to apply in
`data_rules_version.json`.  The registration's final target is compared with
the live revision by a test, so changing a key, a validator, or shared-data
table without registering what to do with an older lake is red before that
deployment can fork the authoritative lake.

The legacy migration is deliberately materialized rather than recomputed at
startup.  Its expensive half is the precomputed per-leaf dtype-env rewrite
(`migrations/prod_rekey_plan.json`), made by the same `rekey_lake` planner
that would otherwise spend twenty-five minutes replaying models while the
service starts.  The bundle carries the plan with the deployment; applying it
is idempotent.  The migration then runs today's row validator over shared
rows and removes the rows it refuses.  `lake_meta.data_rules` is written only
after every step has committed, so an interrupted start runs the same chain
again instead of half-describing the lake.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from backend.inference import benchmarks
from backend.service.bench import store
from kernel_bench import rekey_lake

_LOG = logging.getLogger(__name__)
_HERE = Path(__file__).parent
LEGACY_VERSION = "legacy"
_MANIFEST = json.loads((_HERE / "data_rules_version.json").read_text())
_MIGRATIONS = tuple(_MANIFEST["migrations"])
_SCAN_ROWS = 5_000
_REKEY_ROWS = 25
rekey_apply = rekey_lake.apply


class DataRulesError(RuntimeError):
    """The lake's data format cannot be brought to the registered revision."""


@dataclass(frozen=True)
class MigrationResult:
    """What one startup migration changed, for the health answer."""

    version: str
    rekeyed_rows: int
    env_rekeyed_rows: int
    dropped_shared_rows: int
    reasons: Mapping[str, int]

    def report(self) -> dict[str, Any]:
        return {
            "rekeyed_rows": self.rekeyed_rows,
            "env_rekeyed_rows": self.env_rekeyed_rows,
            "dropped_shared_rows": self.dropped_shared_rows,
            "reasons": dict(self.reasons),
        }


def registered_version() -> str:
    """The data-rules revision this deployment's migration chain reaches."""
    return str(_MIGRATIONS[-1]["to"])


def lake_version(lake: Any) -> str:
    """The data-rules revision a lake says it holds (legacy before this owner)."""
    return str(store.meta(lake).get("data_rules") or LEGACY_VERSION)


def known_version(version: str) -> bool:
    """Whether a lake at this revision has a registered path to the target."""
    return version == registered_version() or any(
        step["from"] == version for step in _MIGRATIONS)


def initialize(lake: Any) -> None:
    """Stamp a brand-new lake with the only revision this code can write."""
    store.set_meta(lake, data_rules=registered_version())


def _invalid_rows(lake: Any) -> tuple[int, Counter[str]]:
    """Remove rows today's validator refuses.

    The validator is the only authority on “invalid”; this function owns no
    second list of suspicious ops or shapes.  Rows are read in bounded pages
    so a future validator refusal cannot make startup memory grow with the
    lake.  `store.delete_rows` is used, rather than a private DELETE, so an
    emptied upload batch goes with its rows exactly as an owner deletion
    takes it.
    """
    cells = ", ".join(f'm."{c}" AS "{c}"' for c in benchmarks.ALL_COLUMNS)
    dropped = 0
    reasons: Counter[str] = Counter()
    sql = (f"SELECT m.id::VARCHAR AS id, b.owner::VARCHAR AS owner, {cells} "
           "FROM lake.measurements m JOIN lake.batches b USING (batch_id) "
           "WHERE true")
    for page in lake.read_batches(sql, batch_rows=_SCAN_ROWS):
        invalid: dict[str, list[str]] = {}
        for row in page:
            try:
                benchmarks.validate_row(
                    {c: "" if row[c] is None else str(row[c])
                     for c in benchmarks.ALL_COLUMNS})
            except benchmarks.BenchmarkValidationError as e:
                reason = str(e).split(";", 1)[0][:160]
                invalid.setdefault(row["owner"], []).append(row["id"])
                reasons[reason] += 1
        for owner, ids in invalid.items():
            dropped += store.delete_rows(lake, owner=owner, ids=ids)
    return dropped, reasons


def _duplicate_shared_rows(lake: Any) -> int:
    """Remove later effective-shared repeats of one row identity.

    Upload deduplicates within an owner, and promotion can still accept the
    same public row from another contributor.  The read path must not carry
    that stream check forever: a migration makes the shared corpus duplicate-
    free once, deterministically keeping the earliest upload batch and id.
    """
    identity = ", ".join(f'm."{c}"' for c in benchmarks.IDENTITY_COLUMNS)
    outcome = "coalesce(nullif(m.outcome, ''), 'measured')"
    dropped = 0
    query = (
        "WITH shared AS (SELECT m.id, b.owner, row_number() OVER ("
        f"PARTITION BY {identity}, {outcome} "
        "ORDER BY b.created_at, b.upload_id, m.id) AS n "
        "FROM lake.measurements m JOIN lake.batches b USING (batch_id) "
        f"WHERE {store.SHARED_ROW}) "
        "SELECT id::VARCHAR AS id, owner::VARCHAR AS owner FROM shared WHERE n > 1")
    for page in lake.read_batches(query, batch_rows=_SCAN_ROWS):
        by_owner: dict[str, list[str]] = {}
        for row in page:
            by_owner.setdefault(row["owner"], []).append(row["id"])
        for owner, ids in by_owner.items():
            dropped += store.delete_rows(lake, owner=owner, ids=ids)
    return dropped


def _apply_registered(step: Mapping[str, Any], lake: Any) -> MigrationResult:
    rekeyed = 0
    if step.get("rekey_plan"):
        path = _HERE / "migrations" / str(step["rekey_plan"])
        rewrites = [(dict(group), new)
                    for group, new in json.loads(path.read_text())]
        # `rekey_lake.apply` owns the rewrite SQL. Applying its plan in bounded
        # slices keeps the whole-lake UPDATE inside the service's DuckDB memory
        # budget; an interrupted slice simply matches no rows when replayed.
        for start in range(0, len(rewrites), _REKEY_ROWS):
            rekeyed += int(rekey_apply(
                lake, rewrites[start:start + _REKEY_ROWS]))
    env_rekeyed = int(store.rekey_env(lake)) if step.get("rekey_env") else 0
    dropped, reasons = (_invalid_rows(lake)
                        if step.get("validate_rows") else (0, Counter()))
    if step.get("deduplicate_shared_rows"):
        duplicates = _duplicate_shared_rows(lake)
        dropped += duplicates
        if duplicates:
            reasons["duplicate shared row identity"] = duplicates
    return MigrationResult(str(step["to"]), rekeyed, env_rekeyed, dropped, reasons)


def migrate(lake: Any) -> MigrationResult:
    """Bring a lake to the registered data-rules revision, or refuse it."""
    version = lake_version(lake)
    total = MigrationResult(version, 0, 0, 0, Counter())
    while version != registered_version():
        step = next((s for s in _MIGRATIONS if s["from"] == version), None)
        if step is None:
            raise DataRulesError(
                f"the lake's data rules are {version!r}, outside this "
                f"deployment's registered chain to {registered_version()!r}")
        result = _apply_registered(step, lake)
        total = MigrationResult(
            result.version, total.rekeyed_rows + result.rekeyed_rows,
            total.env_rekeyed_rows + result.env_rekeyed_rows,
            total.dropped_shared_rows + result.dropped_shared_rows,
            total.reasons + result.reasons)
        version = result.version

    store.set_meta(
        lake, data_rules=registered_version(),
        data_rules_migration=json.dumps(
            total.report(), sort_keys=True, separators=(",", ":")))
    if total.rekeyed_rows or total.env_rekeyed_rows or total.dropped_shared_rows:
        _LOG.warning(
            "bench lake data rules migrated to %s: %d dtype rows, %d env rows, "
            "%d invalid shared rows", total.version, total.rekeyed_rows,
            total.env_rekeyed_rows, total.dropped_shared_rows)
    return total
