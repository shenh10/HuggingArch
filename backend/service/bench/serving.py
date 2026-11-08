"""Whether this process may serve the Kernel Bench lake, and may write it
(docs/measured_calibration.md §5.2).

`Gate.start` (at server start; on first use when nothing started it, as
under pytest) walks the startup once:

  1. open the lake (`lake.get_lake`: one worker, root-only directories);
  2. a brand-new lake (no table yet) is decided before anything is written:
     when a lake already holds the lease it is refused — an empty volume is
     restored from backup, never filled again — otherwise its tables are
     created with a new `lake_id` and `import_state` `none`;
  3. a lake left `importing` by the one-time import from the Postgres
     tables (those tables are gone, migration 0026) can never
     finish: refused — restore it from backup;
  4. decide the copy's role: a lake created (or claimed) with the lease is the
     one authority; a lake created without the lease is a development lake
     with its own id; an existing lake without the lease that does not say it
     is development is a restored copy and serves reads only — it never gets
     a second writable history under an authoritative lake_id;
  5. claim the lease (`bench_lake_claim`, migration 0023) with the lake's
     `lake_id` and this machine's id (FLY_MACHINE_ID; BENCH_LAKE_MACHINE in
     development): 'claimed' / 'ok' serve; 'refused' — another lake, or this
     lake on another machine (a cloned volume) — does not, and neither does
     an unreachable lease (fail closed; retried). Only an operator moves the
     lease to another machine (`bench_lake_transfer`);
  6. annotate every input in memory (`store.annotate_all` with the
     `annotate` the routes pass: only they may import kernel_bench) — the
     grouping labels are computed, never stored;
  7. backups (mandatory; BENCH_LAKE_BACKUP=off opts out in development):
     back up what `current` lacks, then a `backup.Trigger` follows every
     write;
  8. serve: `on_serving(lake, backup)` (the routes start the shared rows' revision
     and start the maintainer).

Until step 7, and whenever a step refuses, every lake endpoint answers 503
with the reason (`reader` raises `LakeUnavailable`). Writes are refused too
while the backup refuses them (`Backup.refusal`: unconfigured, failing, or
too far behind) or the lake's volume is short of space
(`Lake.require_free`), both checked before every write (`writer`). A step
that failed for a reason that may pass (the lake or the lease unreachable)
is retried on a request after `RETRY_AFTER` seconds.

`BENCH_LAKE_LEASE=off` never turns an authoritative copy into a writable
local lake. It permits writes only to a lake whose `lake_meta.lake_role` says
`development` (and such a lake can never claim the production lease); every
other unleased copy is read-only.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import Callable, Optional

import httpx

from backend import supabase_env, supabase_rest
from backend.service.bench import data_rules, store
from backend.service.bench import backup as _backup
from backend.service.bench.backup import Backup, BackupConfig, Trigger
from backend.service.bench.lake import Lake, LakeConfig, LakeUnavailable, get_lake

_LOG = logging.getLogger(__name__)

RETRY_AFTER = 30.0                     # seconds before a transient startup failure is retried


class LeaseUnreachable(RuntimeError):
    """The lease could not be read or claimed (network, Supabase, the
    migration not applied)."""


class Lease:
    """`public.bench_lake_lease` over PostgREST, as the service role."""

    def __init__(self, url: str, key: str, client: Optional[httpx.Client] = None):
        self.base = url.rstrip("/") + "/rest/v1"
        self.headers = supabase_rest.service_headers(key)
        self.client = client or httpx.Client(timeout=10.0)

    @classmethod
    def from_env(cls) -> Optional["Lease"]:
        """None when BENCH_LAKE_LEASE=off; raises `LeaseUnreachable` when
        Supabase is not configured."""
        if os.environ.get("BENCH_LAKE_LEASE", "on").strip() == "off":
            return None
        if not supabase_env.configured():
            raise LeaseUnreachable("Supabase is not configured (SUPABASE_URL and the "
                                   "service-role key), so the lake's lease cannot be claimed")
        return cls(*supabase_env.service_role())

    def _send(self, method: str, path: str, **kw) -> httpx.Response:
        try:
            r = self.client.request(method, f"{self.base}/{path}", headers=self.headers, **kw)
        except httpx.HTTPError as e:
            raise LeaseUnreachable(f"the lease is unreachable: {e}") from e
        if r.status_code >= 400:
            raise LeaseUnreachable(f"the lease answered {r.status_code}: {r.text[:200]}")
        return r

    def holder(self) -> Optional[str]:
        """The lake_id the lease names, None when no lake holds it."""
        rows = self._send("GET", "bench_lake_lease", params={"select": "lake_id"}).json()
        return str(rows[0]["lake_id"]) if rows else None

    def claim(self, lake_id: str, machine: str) -> str:
        """'claimed' (no lake held it), 'ok' (this lake on this machine
        holds it) or 'refused' (another lake, or another machine)."""
        answer = self._send("POST", "rpc/bench_lake_claim",
                            json={"p_lake": lake_id, "p_machine": machine}).json()
        if answer not in ("claimed", "ok", "refused"):
            raise LeaseUnreachable(f"bench_lake_claim answered {answer!r}")
        return answer


def machine() -> str:
    """This machine's id for the lease: FLY_MACHINE_ID, or BENCH_LAKE_MACHINE
    (development); none is a refusal, never a guess."""
    m = (os.environ.get("FLY_MACHINE_ID") or os.environ.get("BENCH_LAKE_MACHINE") or "").strip()
    if not m:
        raise _Refused("no machine id for the lake's lease: FLY_MACHINE_ID is not set "
                       "(BENCH_LAKE_MACHINE names one in development)")
    return m


def restore_from_env() -> Optional[str]:
    """With HUGGINGARCH_BENCH_RESTORE=on and no lake on the volume yet, fill
    the lake's directory from the backup's `current` snapshot
    (`backup.restore`: every object checked against its sha256, the row
    counts against the manifest) before anything opens it — how a lake built
    elsewhere (the cutover's offline import) or a lost volume comes back.
    Returns what was restored, or None when there was nothing to do. Refuses
    (`RuntimeError`) a partly filled directory or a backup without
    `current`: never a silent empty lake."""
    if os.environ.get("HUGGINGARCH_BENCH_RESTORE", "off").strip() != "on":
        return None
    config = LakeConfig.from_env()
    if config.root is None or os.path.exists(config.catalog):
        return None
    if os.path.abspath(config.data) != os.path.join(os.path.abspath(config.root), "data"):
        raise RuntimeError("a restore puts the data in <lake dir>/data; BENCH_LAKE_DATA names "
                           f"{config.data}")
    backup_config = BackupConfig.from_env()
    if backup_config is None:
        raise RuntimeError("HUGGINGARCH_BENCH_RESTORE=on needs BENCH_LAKE_BACKUP_DATA")
    restored = _backup.restore(backup_config.bucket(), config.root)
    return restored.catalog


class _Refused(Exception):
    """A startup step refused; `transient` ones are retried."""

    def __init__(self, reason: str, *, transient: bool = False):
        super().__init__(reason)
        self.transient = transient


class _Degraded(Exception):
    """Startup put the lake in a named degraded state; retrying may help."""

    def __init__(self, reason: str, *, transient: bool = True):
        super().__init__(reason)
        self.transient = transient


class Gate:
    """The process's Kernel Bench lake behind the startup checks."""

    def __init__(self, *, annotate: store.Annotate, open_lake: Callable[[], Lake] = get_lake,
                 lease: Callable[[], Optional[Lease]] = Lease.from_env,
                 backup: Callable[[], Optional[BackupConfig]] = BackupConfig.from_env,
                 on_serving: Callable[..., None] = lambda lake, backup, read_only=False: None,
                 restore: Callable[[], Optional[str]] = restore_from_env,
                 clock: Callable[[], float] = time.monotonic):
        self._open, self._lease, self._backup_config = open_lake, lease, backup
        self._annotate = annotate
        self._restore = restore
        self._on_serving, self._clock = on_serving, clock
        self._guard = threading.Lock()
        self.phase = "new"         # new | starting | serving | degraded | unavailable
        self.reason = ""
        self.lease_state = "unknown"
        self.data_rules = {"state": "unknown",
                           "registered": data_rules.registered_version()}
        self.read_only = False
        self._retry_at: Optional[float] = None
        self.lake: Optional[Lake] = None
        self.backup: Optional[Backup] = None
        self._trigger: Optional[Trigger] = None

    # ---- startup

    def start(self) -> None:
        """Run the startup unless it ran (or runs) already, or a transient
        failure's retry is not due yet."""
        with self._guard:
            due = self.phase == "new" or (
                self._retry_at is not None and self._clock() >= self._retry_at)
            if not due:
                return
            self.phase, self.reason, self._retry_at = "starting", "the Kernel Bench store is starting", None
        try:
            self._startup()
        except _Refused as e:
            self._unavailable(str(e), transient=e.transient)
        except (LakeUnavailable, LeaseUnreachable) as e:
            self._unavailable(str(e), transient=True)
        except _Degraded as e:
            self._degraded(str(e), transient=e.transient)
        except Exception as e:
            _LOG.exception("bench lake startup failed")
            self._unavailable(f"startup failed: {type(e).__name__}: {e}", transient=True)

    def _unavailable(self, reason: str, *, transient: bool) -> None:
        (_LOG.warning if transient else _LOG.error)("Kernel Bench store unavailable: %s", reason)
        with self._guard:
            self.phase, self.reason = "unavailable", reason
            self._retry_at = self._clock() + RETRY_AFTER if transient else None

    def _degraded(self, reason: str, *, transient: bool) -> None:
        _LOG.error("Kernel Bench store degraded: %s", reason)
        with self._guard:
            self.phase, self.reason = "degraded", reason
            self._retry_at = self._clock() + RETRY_AFTER if transient else None

    def _set_data_rules(self, state: str, *, lake: str | None = None,
                        migration: dict | None = None, error: str | None = None,
                        phase: tuple[str, str] | None = None) -> None:
        """Publish the migration owner's in-memory answer without reading the lake."""
        with self._guard:
            self.data_rules = {
                "state": state,
                "registered": data_rules.registered_version(),
                **({"lake": lake} if lake is not None else {}),
                **({"migration": migration} if migration is not None else {}),
                **({"error": error} if error is not None else {}),
            }
            if phase is not None:
                self.phase, self.reason = phase

    def _startup(self) -> None:
        try:
            restored = self._restore()
        except Exception as e:
            raise _Refused(f"the restore from backup failed: {type(e).__name__}: {e}") from e
        if restored:
            _LOG.warning("bench lake restored from backup into %s", restored)
        lake = self._open()
        lease = self._lease()
        here = machine() if lease is not None else None     # before anything is written
        if lease is None:
            self.lease_state = "off"
            _LOG.warning("bench lake lease is OFF (BENCH_LAKE_LEASE=off): development only")
        if not store.has_tables(lake):
            # Brand-new: decided before anything is written.
            if lease is not None:
                holder = lease.holder()
                if holder is not None:
                    raise _Refused(f"not the authoritative lake (an empty lake; the lease names lake "
                                   f"{holder}) — restore it from backup")
            store.ensure_schema(
                lake, lake_id=str(uuid.uuid4()), import_state="none",
                lake_role="authority" if lease is not None else "development")
            data_rules.initialize(lake)
        meta = store.meta(lake)
        if "lake_id" not in meta:
            # No identity means this copy cannot share an authoritative id.
            store.set_meta(lake, lake_id=str(uuid.uuid4()), import_state="none",
                           lake_role="development")
            meta = store.meta(lake)
        if meta.get("import_state") == "importing":
            raise _Refused("the import from the Postgres tables into this lake never finished, "
                "and those tables are gone — restore the lake from backup")
        role = meta.get("lake_role")
        if lease is not None:
            if role == "development":
                raise _Refused("a development lake cannot claim the authoritative lease")
            self._claim(lease, meta["lake_id"], here)
            if role is None:
                store.set_meta(lake, lake_role="authority")
                role = "authority"
        elif role != "development":
            # An older restored copy has no role. It may be the authoritative
            # lake's backup, so the only safe assumption is read-only.
            self.read_only = True
            _LOG.warning("bench lake lease is OFF and lake_role is %r: serving read-only",
                         role or "unset")
        before = data_rules.lake_version(lake)
        if not data_rules.known_version(before):
            reason = (f"the lake's data rules are {before!r}, outside this "
                      f"deployment's registered chain")
            self._set_data_rules("failed", lake=before, error=reason)
            raise _Degraded(f"the lake's data rules did not migrate: {reason}",
                            transient=False)
        if before != data_rules.registered_version():
            self._set_data_rules(
                "behind", lake=before,
                phase=("degraded", f"the lake's data rules are {before}; migrating"))
            if self.read_only:
                reason = (f"the read-only copy's data rules are {before}, not "
                          f"{data_rules.registered_version()}; restore a current backup "
                          "or start the authoritative lease holder")
                self._set_data_rules("failed", lake=before, error=reason,
                                     phase=("degraded", reason))
                raise _Degraded(f"the lake's data rules did not migrate: {reason}",
                                transient=False)
        # A lake written before annotations moved to memory (docs/measured_calibration.md §4.2) still
        # has them as `inputs` columns, and an upload's insert of the key
        # alone fails against them: drop them (idempotent; after the lease,
        # like every other write).
        if self.read_only:
            self._set_data_rules("current", lake=before)
        else:
            store.ensure_schema(lake)
            try:
                migrated = data_rules.migrate(lake)
            except Exception as e:
                self._set_data_rules("failed", lake=before,
                                     error=f"{type(e).__name__}: {e}")
                raise _Degraded(
                    f"the lake's data rules did not migrate: {type(e).__name__}: {e}",
                    transient=not isinstance(e, data_rules.DataRulesError)) from e
            if before == data_rules.registered_version():
                self._set_data_rules("current", lake=before)
            else:
                self._set_data_rules(
                    "current", lake=migrated.version, migration=migrated.report())
        store.annotate_all(lake, self._annotate)
        if not self.read_only:
            self._start_backups(lake)
        with self._guard:
            self.lake, self.phase, self.reason = lake, "serving", ""
        self._on_serving(lake, self.backup, read_only=self.read_only)

    def _claim(self, lease: Lease, lake_id: str, here: str) -> None:
        try:
            answer = lease.claim(lake_id, here)
        except LeaseUnreachable:
            self.lease_state = "unreachable"
            raise
        self.lease_state = answer
        if answer == "refused":
            raise _Refused(f"not the authoritative lake (lake {lake_id} on machine {here}; the "
                           "lease names another lake or machine) — restore from backup, or "
                           "move the lease with bench_lake_transfer")

    def _start_backups(self, lake: Lake) -> None:
        if os.environ.get("BENCH_LAKE_BACKUP", "on").strip() == "off":
            _LOG.warning("bench lake backups are OFF (BENCH_LAKE_BACKUP=off): development only")
            return
        config = self._backup_config()
        self.backup = Backup(config.bucket() if config is not None else None)
        if config is None:
            _LOG.error("bench lake backups are not configured (BENCH_LAKE_BACKUP_DATA): "
                       "writes are refused")
            return
        self._trigger = Trigger(self.backup, lake)
        try:
            self.backup.run(lake)        # what `current` lacks, before any write
        except Exception:
            self._trigger.poke()         # recorded; writes wait for a run to succeed
        self._trigger.start()

    # ---- serving

    def reader(self) -> Lake:
        """The lake, or `LakeUnavailable` naming why not."""
        self.start()
        with self._guard:
            if self.phase == "serving" and self.lake is not None:
                return self.lake
            raise LakeUnavailable(self.reason or "the Kernel Bench store is starting")

    def writer(self) -> Lake:
        """The lake for a write, or `LakeUnavailable`: not serving, the
        backup refusing (unconfigured, failing, behind), or the volume
        short of space."""
        lake = self.reader()
        if self.read_only:
            raise LakeUnavailable(
                "writes are refused: this copy of lake "
                f"{store.meta(lake).get('lake_id', '?')} is read-only; only the lease holder may write")
        refusal = self.backup.refusal() if self.backup is not None else None
        if refusal:
            raise LakeUnavailable(f"writes are refused: {refusal}")
        lake.require_free("writes")
        return lake

    def wrote(self) -> None:
        """A write committed: back it up (coalesced, off the request path)."""
        if self._trigger is not None:
            self._trigger.poke()

    def health(self) -> dict:
        """For /health's detail: never raises."""
        out = {"state": self.phase, "reason": self.reason or None, "lease": self.lease_state,
               "read_only": self.read_only,
               "backup": self.backup.status() if self.backup is not None else "off",
               "data_rules": self.data_rules}
        try:
            free = self.lake.free_bytes() if self.lake is not None else None
            out["free_mb"] = None if free is None else free // 2**20
        except OSError as e:
            out["free_mb"] = f"unknown: {e}"
        return out

    def close(self) -> None:
        """Tests: forget the lake (the caller closes it)."""
        with self._guard:
            self.phase, self.lake, self.backup, self._trigger = "new", None, None, None
            self.read_only = False
            self.data_rules = {"state": "unknown",
                               "registered": data_rules.registered_version()}
