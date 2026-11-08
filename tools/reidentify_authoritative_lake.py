#!/usr/bin/env python3
"""Make a replacement measurement lake into a new authoritative lineage.

The candidate is copied from an existing lake, but a copy keeps the old
``lake_id`` and therefore must not be written as though it were the same
history.  This tool writes a fresh UUID and ``lake_role=authority`` into the
copy (never the source), optionally applies the registered data-rules
migration, and reports the identity to use when moving the lease.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.service.bench import data_rules, store
from backend.service.bench.backup import _set_data_path
from backend.service.bench.lake import Lake, LakeConfig


def _copy(source_catalog: Path, source_data: Path, destination: Path) -> None:
    destination_catalog = destination / "catalog.ducklake"
    destination_data = destination / "data"
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    try:
        shutil.copy2(source_catalog, destination_catalog)
        shutil.copytree(source_data, destination_data)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    _set_data_path(destination_catalog, str(destination_data))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-catalog", required=True, type=Path)
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--lake-id", type=uuid.UUID,
                        help="replacement UUID (default: generated)")
    parser.add_argument("--migrate", action="store_true",
                        help="apply registered data-rules migrations to the copy")
    args = parser.parse_args(argv)
    new_id = args.lake_id or uuid.uuid4()

    source = Lake.open(LakeConfig(catalog=str(args.source_catalog),
                                  data=str(args.source_data)))
    try:
        old_meta = store.meta(source)
        old_id = old_meta.get("lake_id")
        rows = int(source.read("SELECT count(*) AS n FROM lake.measurements")[0]["n"])
    finally:
        source.close()
    if not old_id:
        raise SystemExit("the source lake has no lake_id")
    if str(new_id) == old_id:
        raise SystemExit("the replacement lake must have a new lake_id")
    if old_meta.get("lake_role") == "development":
        raise SystemExit("a development lake is not an authoritative replacement")

    _copy(args.source_catalog, args.source_data, args.destination)
    lake = Lake.open(LakeConfig(catalog=str(args.destination / "catalog.ducklake"),
                                data=str(args.destination / "data")))
    try:
        store.set_meta(lake, lake_id=str(new_id), lake_role="authority",
                       supersedes_lake_id=old_id)
        if args.migrate:
            result = data_rules.migrate(lake)
            rules = {"version": result.version, **result.report()}
        else:
            rules = {"version": data_rules.lake_version(lake)}
        if rules["version"] != data_rules.registered_version():
            raise SystemExit(
                f"the replacement copy's data rules are {rules['version']!r}, not "
                f"{data_rules.registered_version()!r}; rerun with --migrate")
        answer = {
            "destination": str(args.destination),
            "old_lake_id": old_id,
            "lake_id": str(new_id),
            "rows": rows,
            "data_rules": rules,
        }
    finally:
        lake.close()
    print(json.dumps(answer, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
