"""The Kernel Bench maintainer: one process at a time (lake.maintainer_lock),
paced, behind HUGGINGARCH_BENCH_MAINTENANCE (`off`: nothing runs).

Each cycle (refused, like a write, when the lake's volume is short of
space — `Lake.require_free`):
  1. a backup when the lake moved past what `current` names
     (backup.Backup.run);
  2. housekeeping, at most hourly: DuckLake maintenance — merge adjacent
     small files, expire snapshots older than 7 days, delete the files those
     snapshots alone referenced and orphaned files, both older than 7 days
     (longer than any query and the backup cycle) — under the lake's backup
     lock (maintainer → backup), so no backup reads a file it rewrites; then
     the backups' retention (backup.Backup.gc). While the backup is behind,
     only the merge runs: expiry, cleanup and the retention wait
     (docs/measured_calibration.md §4.7); then a backup of what the merge wrote.

The loop (`start`) waits 300 s before its first cycle, rests three times
each cycle's duration (at least 30 s), and backs off exponentially up to
600 s on errors. The routes module starts it with the lake and the backup
once the lake serves (`serving.Gate`'s `on_serving`). The derived file
(annotations, lookup cells, browse tables) is not maintained here: its own
refresh follows the lake (`derived.Derived`).
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from backend.service.bench.backup import Backup
from backend.service.bench.lake import Lake

_LOG = logging.getLogger(__name__)

RETAIN = "7 days"                # snapshots and files kept for readers and backups
HOUSEKEEPING_EVERY = 3600.0      # seconds between DuckLake maintenance passes
FIRST_CYCLE_AFTER = 300.0
REST_FACTOR = 3.0
MIN_REST = 30.0
MAX_BACKOFF = 600.0


def housekeeping(lake: Lake, *, expire: bool = True) -> dict:
    """DuckLake maintenance; returns how many files each step touched.
    `expire=False`: only the merge (nothing is deleted)."""
    steps = {
        "merged": "SELECT count(*) FROM ducklake_merge_adjacent_files('lake')",
        "expired": "SELECT count(*) FROM ducklake_expire_snapshots('lake', "
                   f"older_than => now() - INTERVAL '{RETAIN}')",
        "cleaned": "SELECT count(*) FROM ducklake_cleanup_old_files('lake', "
                   f"older_than => now() - INTERVAL '{RETAIN}')",
        "orphaned": "SELECT count(*) FROM ducklake_delete_orphaned_files('lake', "
                    f"older_than => now() - INTERVAL '{RETAIN}')",
    }
    if not expire:
        steps = {"merged": steps["merged"]}
    return {k: int(lake.transact(lambda cur, sql=sql: cur.execute(sql).fetchone()[0]))
            for k, sql in steps.items()}


def maintain_once(lake: Lake, *, backup: Optional[Backup] = None, housekeep: bool = True,
                  now: Optional[_dt.datetime] = None) -> dict:
    """One cycle, if this process is the maintainer (`maintainer: False`
    and nothing done otherwise)."""
    with lake.maintainer_lock() as held:
        if not held:
            return {"maintainer": False}
        lake.require_free("maintenance")
        did: dict[str, Any] = {"maintainer": True}
        if backup is not None:
            did["backup"] = backup.run(lake)
        if housekeep:
            with lake.backup_lock():
                caught_up = backup is None or not backup.behind(lake)
                did.update(housekeeping(lake, expire=caught_up))
            if backup is not None:
                if caught_up:
                    did["gc"] = backup.gc(lake, now)
                else:
                    did["expiry_held"] = "the backup is behind"
                if did["merged"]:
                    backup.run(lake)
        return did


class Loop:
    """The paced loop's state; `step` runs one cycle and returns how long to
    rest before the next."""

    def __init__(self, lake: Callable[[], Lake], backup: Optional[Backup],
                 clock: Callable[[], float] = time.monotonic):
        self.lake = lake
        self.backup, self.clock = backup, clock
        self.last_housekeeping: Optional[float] = None
        self.backoff = MIN_REST

    def step(self) -> float:
        t0 = self.clock()
        due = self.last_housekeeping is None or t0 - self.last_housekeeping >= HOUSEKEEPING_EVERY
        try:
            did = maintain_once(self.lake(), backup=self.backup, housekeep=due)
        except Exception:
            # Reported, then retried — later each time it fails again.
            _LOG.exception("bench lake maintenance failed; next try in %d s", self.backoff)
            rest, self.backoff = self.backoff, min(self.backoff * 2, MAX_BACKOFF)
            return rest
        self.backoff = MIN_REST
        if due and did.get("maintainer"):
            self.last_housekeeping = t0
        if any(v for k, v in did.items() if k != "maintainer"):
            _LOG.info("bench lake maintenance: %s", did)
        return max(REST_FACTOR * (self.clock() - t0), MIN_REST)


def start(lake: Lake, backup: Optional[Backup]) -> Optional[threading.Thread]:
    """Start the maintainer thread. Returns the thread, or None when
    HUGGINGARCH_BENCH_MAINTENANCE=off."""
    if os.environ.get("HUGGINGARCH_BENCH_MAINTENANCE", "on") == "off":
        _LOG.warning("bench lake maintenance is off (HUGGINGARCH_BENCH_MAINTENANCE=off)")
        return None
    loop = Loop(lambda: lake, backup)

    def run() -> None:
        time.sleep(FIRST_CYCLE_AFTER)
        while True:
            time.sleep(loop.step())

    t = threading.Thread(target=run, name="bench-lake-maintainer", daemon=True)
    t.start()
    return t
