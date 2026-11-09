"""The engine-versioned pool of SOL walk records.

The pool is deliberately below every page payload. A record is one neutral
piece of theory — a Gallery class-walk layout, a Scaling sweep cell, a
Compare projection input — and an identity names the walk that produced it.
Consumers read the records they need and project them; they never store their
response shape here.

The durable form is spill-friendly. Each identity has a small manifest and
index, while the records themselves are appended to JSON Lines shards. The
index carries ``(shard, offset, length, digest)`` per record, so a reader
seeks one record instead of parsing a whole bundle. Only the records a
consumer asks for enter Python objects.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import threading
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from backend.measure.scope import ResultIdentity, make_key
from backend.code_trace.memo import memoize
from backend.code_trace.provider import TRACES, Trace, provider

logger = logging.getLogger(__name__)


#: v5: every record names its trace (`backend.code_trace`).
poolSchemaVersion = 5
defaultShardBytes = 8 << 20


@dataclass(frozen=True)
class RecordSlot:
    shard: int
    offset: int
    length: int
    digest: str


def _reopen(root: str, hotRecords: int, hotBytes: int,
            shardBytes: int) -> "SolPool":
    """`SolPool.__reduce__`'s constructor: the pool at `root`, as cut."""
    return SolPool(root, hotRecords=hotRecords, hotBytes=hotBytes,
                   shardBytes=shardBytes)


class SolPool:
    """An immutable, seekable local store of SOL records."""

    def __init__(self, root: os.PathLike[str] | str, *,
                 hotRecords: int = 256, hotBytes: int = 8 << 20,
                 shardBytes: int = defaultShardBytes):
        self.root = pathlib.Path(root)
        self.shardBytes = int(shardBytes)
        self._hot: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._hotSizes: dict[tuple[str, str], int] = {}
        self._hotBytesTarget = int(hotBytes)
        self._hotRecordsTarget = int(hotRecords)
        self._lock = threading.Lock()

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        """A pool crosses a process boundary as where it is and how it is
        cut: the receiving process opens the same directory with a lock and
        a hot set of its own, which are this process's, not the pool's. So
        a `RecordView` a worker hands back stays a seekable view instead of
        failing to pickle (the measured Scaling sweep's 500)."""
        return (_reopen, (str(self.root), self._hotRecordsTarget,
                          self._hotBytesTarget, self.shardBytes))

    def write(self, identity: ResultIdentity,
              records: Iterable[tuple[str, Any]], *,
              metadata: Optional[Mapping[str, Any]] = None,
              traces: Optional[Mapping[str, Trace]] = None) -> bool:
        """Write one bundle, each record with the trace of the code that
        computed it (`traces`, by record id; a record without one takes what
        the computation open around this call has run so far). An existing
        bundle whose every record is still valid wins; one with a stale
        record is replaced — the caller recomputed the stale ones and
        handed the valid ones back (`recordTrace`).

        Records are serialised one at a time. A producer may
        hold its result in memory, but the store never turns it into one large
        JSON document and never asks a reader to parse a neighbouring record
        to reach this one.
        """
        directory = self._directory(identity.key)
        fallback: Optional[Trace] = None
        try:
            directory.parent.mkdir(parents=True, exist_ok=True)
            lock = directory.with_name(directory.name + ".writer")
            # Cross-process single-writer. A crashed writer leaves a stale
            # lock; its age makes it removable rather than wedging the pool.
            try:
                lock.mkdir()
            except FileExistsError:
                try:
                    stale = (time.time() - lock.stat().st_mtime) > 600
                except OSError:
                    stale = False
                if not stale:
                    return False
                lock.rmdir()
                lock.mkdir()
            output: Any = None
            staging = directory.with_name(
                f"{directory.name}.{os.getpid()}.{threading.get_ident()}.new")
            try:
                if self._manifest(identity.key) is not None and self._allValid(
                        identity.key):
                    return True
                staging.mkdir()
                slots: dict[str, list[Any]] = {}
                recordTraces: dict[str, str] = {}
                bodies: dict[str, Any] = {}
                shards: list[pathlib.Path] = []
                shardNo = -1
                target = 0
                seen: set[str] = set()
                for recordId, value in records:
                    if recordId in seen:
                        raise ValueError(f"duplicate SOL record id {recordId!r}")
                    seen.add(recordId)
                    trace = (traces or {}).get(recordId)
                    if trace is None:
                        if fallback is None:
                            fallback = provider().snapshot()
                        if fallback is None:
                            message = ("an SOL record names no trace and no "
                                       "computation is open to take one from")
                            # Strict (tests): loud, past the write's own
                            # failure-is-a-miss handling below.
                            raise (RuntimeError if provider().strict
                                   else ValueError)(message)
                        trace = fallback
                    recordTraces[recordId] = TRACES.put(trace)
                    bodies[trace.id] = trace.to_json()
                    line = json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, default=str)
                    encoded = line.encode("utf-8")
                    if output is None or target + len(encoded) > self.shardBytes:
                        if output is not None:
                            output.close()
                        shardNo += 1
                        target = 0
                        shards.append(
                            staging / f"records-{shardNo:05d}.jsonl")
                        output = shards[-1].open("wb")
                    offset = target
                    output.write(encoded)
                    target += len(encoded)
                    slots[recordId] = [
                        shardNo, offset, len(encoded),
                        hashlib.sha256(encoded).hexdigest()]
                if output is None:
                    raise ValueError("an SOL bundle states at least one record")
                output.close()
                output = None

                # The trace bodies travel with the bundle: a bundle unpacked
                # from an artifact is checked on a machine that never
                # recorded them.
                self._atomic_json(staging / "traces.json", bodies)
                self._atomic_json(staging / "index.json", {
                    "schema": poolSchemaVersion,
                    "records": slots,
                    "traces": recordTraces,
                    "order": list(slots),
                    "record_count": len(slots),
                })
                manifest = {
                    "schema": poolSchemaVersion,
                    "identity": {
                        "kind": identity.kind,
                        "version": identity.version,
                        "content": identity.content,
                        "scope": identity.scope.key,
                        "reads": identity.reads,
                    },
                    "key": identity.key,
                    "record_count": len(slots),
                    "shard_count": len(shards),
                    "metadata": dict(metadata or {}),
                }
                self._atomic_json(staging / "manifest.json", manifest)
                self._swapIn(staging, directory)
                with self._lock:
                    for cacheKey in [k for k in self._hot if k[0] == identity.key]:
                        del self._hot[cacheKey]
                        del self._hotSizes[cacheKey]
                return True
            finally:
                if output is not None:
                    output.close()
                shutil.rmtree(staging, ignore_errors=True)
                lock.rmdir()
        except (OSError, TypeError, ValueError):
            logger.debug("SOL pool write failed for %s", identity.key,
                         exc_info=True)
            return False

    @staticmethod
    def _swapIn(staging: pathlib.Path, directory: pathlib.Path) -> None:
        """`staging` becomes the bundle at `directory`, replacing one there.
        A reader between the two renames finds no bundle: a miss."""
        if directory.exists():
            retired = directory.with_name(
                f"{directory.name}.{os.getpid()}.{threading.get_ident()}.old")
            directory.rename(retired)
            staging.rename(directory)
            shutil.rmtree(retired, ignore_errors=True)
        else:
            staging.rename(directory)

    def recordTrace(self, key: str, recordId: str) -> Optional[Trace]:
        """The trace a record was stored with, or None."""
        index = self._index(key)
        tid = index and index.get("traces", {}).get(recordId)
        if not tid:
            return None
        trace = TRACES.get(tid)
        if trace is None:
            body = self._bodies(key).get(tid)
            if body is None:
                return None
            trace = Trace.from_json(body)
            if trace.id != tid:
                return None
            TRACES.put(trace)
        return trace

    def _allValid(self, key: str) -> bool:
        index = self._index(key)
        p = provider()
        return index is not None and all(
            (trace := self.recordTrace(key, recordId)) is not None
            and p.valid(trace)
            for recordId in index["order"])

    def _bodies(self, key: str) -> dict[str, Any]:
        try:
            return json.loads((self._directory(key) / "traces.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    def manifest(self, key: str) -> Optional[dict[str, Any]]:
        return self._manifest(key)

    def recordIds(self, key: str) -> Optional[tuple[str, ...]]:
        index = self._index(key)
        return None if index is None else tuple(index["order"])

    def records(self, key: str) -> Iterator[tuple[str, Any]]:
        """Every record, one object at a time, in written order."""
        index = self._index(key)
        if index is None:
            return
        for recordId in index["order"]:
            value = self.record(key, recordId)
            if value is not None:
                yield recordId, value

    def record(self, key: str, recordId: str) -> Any:
        """One record, seeking its byte range; missing, corrupt, or computed
        by code the running code no longer matches (its trace) is a miss. A
        record read hands its trace to the computations open around the
        read: what they read depends on the code that computed it."""
        trace = self.recordTrace(key, recordId)
        p = provider()
        if trace is None or not p.valid(trace):
            return None
        p.merge(trace)
        cacheKey = (key, recordId)
        with self._lock:
            if cacheKey in self._hot:
                self._hot.move_to_end(cacheKey)
                return self._hot[cacheKey]
        index = self._index(key)
        slot = index and index["records"].get(recordId)
        if slot is None:
            return None
        try:
            recordSlot = RecordSlot(*slot)
            with (self._directory(key) / f"records-{recordSlot.shard:05d}.jsonl").open(
                    "rb") as shard:
                shard.seek(recordSlot.offset)
                encoded = shard.read(recordSlot.length)
            if hashlib.sha256(encoded).hexdigest() != recordSlot.digest:
                return None
            value = json.loads(encoded)
        except (OSError, ValueError, TypeError):
            logger.debug("SOL pool record read failed for %s/%s",
                         key, recordId, exc_info=True)
            return None
        self._keep(cacheKey, value, recordSlot.length)
        return value

    def sequence(self, key: str) -> Optional["RecordView"]:
        """The bundle's records as a lazy `Sequence`.

        Some consumers (a sweep's assembler) take a sequence but never hold all
        of it at once. This view gives them one record at a time from disk;
        callers that must mutate every record still materialise it explicitly.
        """
        index = self._index(key)
        if index is None:
            return None
        return RecordView(self, key, tuple(index["order"]))

    def nbytes(self, key: str) -> Optional[int]:
        try:
            return sum(path.stat().st_size for path in self._directory(key).glob("*")
                       if path.is_file() and path.name != ".writer")
        except OSError:
            logger.debug("SOL pool size failed for %s", key, exc_info=True)
            return None

    def clear(self) -> None:
        with self._lock:
            self._hot.clear()
            self._hotSizes.clear()

    def _directory(self, key: str) -> pathlib.Path:
        return self.root / key[:2] / key

    def _manifest(self, key: str) -> Optional[dict[str, Any]]:
        try:
            value = json.loads(
                (self._directory(key) / "manifest.json").read_text(encoding="utf-8"))
            return value if value.get("schema") == poolSchemaVersion else None
        except (OSError, ValueError, TypeError):
            logger.debug("SOL pool manifest read failed for %s", key,
                         exc_info=True)
            return None

    def _index(self, key: str) -> Optional[dict[str, Any]]:
        try:
            value = json.loads(
                (self._directory(key) / "index.json").read_text(encoding="utf-8"))
            return value if value.get("schema") == poolSchemaVersion else None
        except (OSError, ValueError, TypeError):
            logger.debug("SOL pool index read failed for %s", key,
                         exc_info=True)
            return None

    def _keep(self, cacheKey: tuple[str, str], value: Any, size: int) -> None:
        with self._lock:
            self._hot[cacheKey] = value
            self._hotSizes[cacheKey] = size
            while (len(self._hot) > self._hotRecordsTarget
                   or sum(self._hotSizes.values()) > self._hotBytesTarget):
                oldest = next(iter(self._hot))
                del self._hot[oldest]
                del self._hotSizes[oldest]

    @staticmethod
    def _atomic_json(path: pathlib.Path, value: Any) -> None:
        temporary = path.with_suffix(
            f".{os.getpid()}.{threading.get_ident()}.tmp")
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, default=str) + "\n",
            encoding="utf-8")
        os.replace(temporary, path)


class RecordView(Sequence[Any]):
    """A seekable, ordered view of one pool bundle."""

    def __init__(self, pool: SolPool, key: str, recordIds: tuple[str, ...]):
        self.pool = pool
        self.key = key
        self.recordIds = recordIds

    def __len__(self) -> int:
        return len(self.recordIds)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self.pool.record(self.key, recordId)
                    for recordId in self.recordIds[index]]
        return self.pool.record(self.key, self.recordIds[index])


@memoize(maxsize=1)
def defaultPool() -> SolPool:
    """The process's local SOL pool, configured by ``HA_SOL_POOL_DIR``."""
    root = os.environ.get(
        "HA_SOL_POOL_DIR",
        pathlib.Path.home() / ".cache" / "huggingarch" / "sol-pool",
    )
    return SolPool(root)


def recordId(*parts: Any) -> str:
    """One record's stable id inside an already-identified bundle."""
    return make_key("sol-record", *parts)
