#!/usr/bin/env python3
"""Import a directory of benchmark CSVs into the SHARED corpus of the Kernel
Bench lake, owned by an admin user.

Each file is validated by the CLI/CI gate (`validate_text`) and stored as
one batch of the owner's, already shared — exactly what an admin's
`hbench upload --public` stores (`store.upload`; the service's derived
file annotates the new inputs when it reaches the upload). A file holds one run (its rows' `run_id`).

Idempotent: a row whose 12-column identity the owner already holds is
skipped, and a file whose every row is such a duplicate writes no batch, so
a re-run stores nothing twice.

Usage (the lake from the environment, as the API reads it):
  BENCH_LAKE_CATALOG=... BENCH_LAKE_DATA=... [BENCH_LAKE_S3_*=...] \
    uv run python scripts/seed_shared_corpus.py --owner <admin-user-uuid> --dir <dir of CSVs>

The owner is the admin's uuid: `select id from auth.users where email = '…';`
With a file catalog only one process can open the lake: run it while the API
is stopped (the API exports the shared corpus for the estimator at startup).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.inference import benchmarks as _benchmarks
from backend.service.bench import lake as _lake
from backend.service.bench import store as _store
from backend.service.bench.routes import run_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True, help="admin user UUID (row owner)")
    ap.add_argument("--dir", required=True, help="a directory of corpus CSVs")
    args = ap.parse_args()
    owner = str(uuid.UUID(args.owner))

    files = sorted(glob.glob(os.path.join(args.dir, "**", "*.csv"), recursive=True))
    if not files:
        print(f"no CSVs under {args.dir}", file=sys.stderr)
        return 1

    lake = _lake.get_lake()
    _store.ensure_schema(lake)
    stored = skipped = 0
    for path in files:
        with open(path, encoding="utf-8") as f:
            _header, rows = _benchmarks.validate_text(f.read(), source=path)
        try:
            run_id = run_of(rows)
        except ValueError as e:
            print(f"FAIL {path}: {e}", file=sys.stderr)
            return 1
        res = _store.upload(lake, owner=owner, upload_id=str(uuid.uuid4()), run_id=run_id,
                            label=f"shared:{os.path.basename(path)}", rows=rows,
                            status="shared", decided_by=owner)
        stored += res.inserted
        skipped += res.duplicates
        print(f"{res.inserted:5d} rows from {path}"
              + (f" ({res.duplicates} already present)" if res.duplicates else ""))
    print(f"done — {stored} rows shared (owner {owner}); {skipped} already present were skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
