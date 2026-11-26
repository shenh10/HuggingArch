#!/usr/bin/env python3
"""Upload measured GPU results into the authoritative HuggingArch lake.

The bench service on port 8100 is a machine-local trust boundary, not the
authoritative corpus.  This client therefore uses the public API and a
personal access token (`hbench upload`): upload private, nominate within the
peaks each row records, then promote.  The token's account must be an admin
for the final call; using one admin token keeps the owner and reviewer facts
on the server instead of forging `X-HA-*` headers.

The upload preparation is unchanged from the local uploader: a CSV is split
by `(model, run_id)`, achieved rates are restamped from the plan the queue
holds now, and every set passes `hbench check` before it leaves the box.
The server remains the owner of deduplication.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from urllib.parse import quote, urlparse
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kernel_bench.check import check_paths, format_report
from kernel_bench.emit import restamp_rows
from kernel_bench.plan import read_jobs_jsonl


def _token(args: argparse.Namespace) -> str:
    token = (args.token or "").strip()
    if not token and args.token_file:
        token = Path(os.path.expanduser(args.token_file)).read_text(
            encoding="utf-8").strip()
    if not token:
        raise SystemExit(
            "no authoritative-lake token: pass --token, set "
            "HUGGINGARCH_TOKEN, or point HUGGINGARCH_TOKEN_FILE at a file")
    return token


class AuthoritativeLake:
    def __init__(self, base: str, token: str):
        self.base = base.rstrip("/")
        self.token = token

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.token}",
                     "User-Agent": "huggingarch-authoritative-upload/1"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {e.code} from {path}: {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"cannot reach the authoritative lake at {path}: {e}") from e

    def upload(self, text: str, label: str) -> dict[str, Any]:
        return self.post("/api/bench/upload", {
            "csv_text": text, "label": label, "make_public": False})

    def share_within_peak(self, owner: str, run_id: str) -> int:
        """Nominate and promote every private batch of the RUN, within peak;
        the number of batches promoted. By run, not by upload: an upload
        whose promote never ran (the process stopped between the two
        calls) is deduplicated to 0 rows when sent again, so promoting
        only what this call uploaded would leave that batch private for
        good. A 404 is the run having nothing in that state any more."""
        try:
            self.post(f"/api/bench/run/{quote(run_id, safe='')}/contribute"
                      "?within_peak=true", {})
        except RuntimeError as e:
            if not str(e).startswith("HTTP 404"):
                raise
        try:
            answer = self.post("/api/bench/promote", {"sets": [
                {"owner": owner, "kind": "run", "id": run_id}]})
        except RuntimeError as e:
            if str(e).startswith("HTTP 404"):
                return 0
            raise
        return sum(p["batches"] for p in answer["promoted"])


def _plan_for(path: Path) -> Path:
    return path.parent.parent.parent / "jobs" / f"{path.parent.name}.jsonl"


def _not_local(base: str) -> str:
    """Refuse the machine-local trust boundary even if an old env names it."""
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"{base!r} is not an HTTP(S) deployment URL")
    if parsed.hostname == "localhost" or parsed.hostname.startswith("127."):
        raise ValueError(
            "the authoritative lake is a public API, not 127.0.0.1:8100")
    return base


def _prepared(path: Path, chunk_rows: int
              ) -> list[tuple[tuple[str, str], int, list[dict[str, str]], str]]:
    """One CSV as `(model, run_id)` sets, restamped from the queue's plan,
    each in numbered parts of at most `chunk_rows` rows."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        by_set: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            by_set[(row["model"], row["run_id"])].append(row)
    plan = _plan_for(path)
    if plan.exists():
        jobs = read_jobs_jsonl(str(plan))
        for key, rows in list(by_set.items()):
            new_rows, changed, missed = restamp_rows(rows, jobs)
            by_set[key] = new_rows
            if changed:
                print(f"  restamped {changed} rows of {key[0]} {key[1]} against "
                      f"{plan.name} ({missed} not in the plan)", flush=True)
    out = []
    for key, rows in sorted(by_set.items()):
        # A run goes up in parts of at most `chunk_rows`: the public API sits
        # behind an edge proxy that cuts a request at 100 s (HTTP 524), and an
        # 8k-row set already took longer. A run may span several uploads —
        # each is its own upload_id, batches share the run_id.
        for start in range(0, len(rows), chunk_rows):
            part = rows[start:start + chunk_rows]
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(part)
            out.append((key, start // chunk_rows, part, buf.getvalue()))
    return out


def _checked(text: str, label: str) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", encoding="utf-8",
                                     delete=False) as tmp:
        tmp.write(text)
        name = tmp.name
    try:
        report = check_paths(name)
        print(format_report(report))
        return report.ok()
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label_prefix")
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--host", default=os.environ.get("HUGGINGARCH_AUTHORITATIVE_URL")
                        or os.environ.get("HUGGINGARCH_HOST", "https://huggingarch.com"))
    parser.add_argument("--token", default=os.environ.get("HUGGINGARCH_TOKEN"))
    parser.add_argument("--token-file",
                        default=os.environ.get("HUGGINGARCH_TOKEN_FILE"))
    parser.add_argument("--owner", default=os.environ.get("HUGGINGARCH_BENCH_OWNER"),
                        help="contributor user id (required unless --private)")
    parser.add_argument("--chunk-rows", type=int, default=2000,
                        help="rows per upload request (default 2000)")
    parser.add_argument("--private", action="store_true",
                        help="upload only; do not nominate or promote")
    args = parser.parse_args(argv)
    owner = (args.owner or "").strip()
    if not args.private and not owner:
        parser.error("sharing needs --owner or HUGGINGARCH_BENCH_OWNER")
    lake = AuthoritativeLake(_not_local(args.host), _token(args))
    rc = 0
    runs: list[str] = []
    for path in args.csv:
        for (model, run_id), part, rows, text in _prepared(path, args.chunk_rows):
            label = (f"{args.label_prefix}-{model.replace('/', '_')}-"
                     f"{path.stem}-{run_id}" + (f"-part{part}" if part else ""))
            if not _checked(text, label):
                print(f"✗ {label}: check failed", file=sys.stderr)
                rc = 1
                continue
            try:
                answer = lake.upload(text, label)
            except RuntimeError as e:
                print(f"✗ {label}: {e}", file=sys.stderr)
                rc = 1
                continue
            if run_id not in runs:
                runs.append(run_id)
            print(f"{label}: {len(rows)} rows → {answer.get('row_count')} rows, "
                  f"id={answer.get('upload_id')}", flush=True)
    if args.private:
        return rc
    # Shared once per run, after all of its parts are up, whatever each part
    # added: what is shared is the run's state on the server, not this call's.
    for run_id in runs:
        try:
            n = lake.share_within_peak(owner, run_id)
        except (RuntimeError, KeyError, TypeError) as e:
            print(f"✗ share {run_id}: {e}", file=sys.stderr)
            rc = 1
            continue
        print(f"shared within peak: run {run_id} ({n} batches)", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
