"""Result caches: a keyed cache over pluggable backends, with single-flight
and a remote tier bound from outside.

Every cached result — the Scaling sweeps, the gallery's entries, the
supported-models list — is held here; no caller hand-rolls a dict, a lock
and an eviction rule:

  * backends: `MemoryBackend` (LRU), `DiskBackend` (content-addressed JSON
    files, oldest-by-mtime evicted by a file-count and a byte budget),
    `TieredBackend` (memory in front of disk), `WriteBehindBackend`
    (a slow store's writes queued on `WRITE_BEHIND`);
  * `KeyedCache` — a keyed cache over a backend, with optional single-flight
    (concurrent misses for one key compute once) and an optional remote tier
    gated per call (`remote_ok`). The remote tier is bound after
    construction (`set_remote`): the service layer binds its Supabase store,
    so this module stays DB-agnostic.

What a cache key is lives beside it: `scope.make_key` (the one key hash) and
`scope.ResultIdentity` (what an estimate's result is a function of); a key is
version-folded by its caller (`backend.fingerprint.result_version` /
`cache_engine_version`), so a data edit lands on a fresh key and no
invalidation pass runs.

The code a value was computed by is the value's own: every entry is stored
with the id of its trace (`backend.code_trace.provider`), the trace body
once beside it (`TRACES`, and the remote tier for a shareable entry). A read
returns a value only while the code this process runs still satisfies its
trace, and hands the trace to every recording open around the read — a hit
names the code it did not run. A stale entry reads as a miss, and the
recomputed value overwrites it. Under `GlobalHash` the trace is the code
hash the key already carries, so every stored entry is valid. Stdlib only
(and `backend.code_trace`, which is).
"""

from __future__ import annotations

import collections
import gc
import json
import logging
import os
import pathlib
import pickle
import queue
import re
import sys
import threading
import types
from typing import Any, Callable, Optional

from backend.code_trace.provider import TRACES, Trace, provider, recording

_log = logging.getLogger(__name__)

#: The field of a stored entry naming its trace; the value is beside it.
TRACE_FIELD = "__code_trace__"
#: Where a trace body is kept in a remote tier, by id.
REMOTE_TRACE_PREFIX = "code-trace/"


# ---------------------------------------------------------------------------
# Backends. A backend is any object with get(key) / put(key, value) / clear().
# A durable backend behind a hot one (`TieredBackend`) also has nbytes(key):
# a stored value's JSON byte length without reading it, or None — what the
# hot tier is told when the value is promoted.
# ---------------------------------------------------------------------------


def private_copy(value: Any) -> Any:
    """`value` as a caller may mutate it: a memory tier hands out (and keeps)
    the stored object itself, so a caller that writes into a value it got
    from a cache, or put into one, writes into a copy. A pickle round trip —
    the copy `copy.deepcopy` makes, several times faster on the millions of
    small objects a sweep's points are."""
    return pickle.loads(pickle.dumps(value, pickle.HIGHEST_PROTOCOL))


class MemoryBackend:
    """Bounded LRU, in-process. Fast; lost on restart. Thread-safe.

    Bounded by entry count, and with `max_bytes` also by size: a value's
    size is its JSON byte length — the `nbytes` the caller passes when the
    tier behind knows it (`TieredBackend`: a file's length), else measured
    once by serialising it (values must then be JSON-serializable, as
    `DiskBackend`'s are) — and the least recently used entries are evicted
    until the kept ones fit. A value larger than the whole budget is not
    kept in memory at all — a tier behind this one still holds it. The
    budget counts JSON bytes, not the Python objects a value is held as,
    which are several times larger: a caller sizes it with that factor in
    mind, per process that holds one."""

    def __init__(self, max_entries: int = 256, max_bytes: Optional[int] = None,
                 *, size_of: Optional[Callable[[Any], int]] = None):
        self._max = max_entries
        self._max_bytes = max_bytes
        self._size_of = size_of or _json_size_of
        self._store: "collections.OrderedDict[str, Any]" = collections.OrderedDict()
        self._sizes: dict[str, int] = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)  # LRU: mark most-recently used
                return self._store[key]
            return None

    def put(self, key: str, value: Any, nbytes: Optional[int] = None) -> None:
        size = 0
        if self._max_bytes is not None:
            size = int(nbytes) if nbytes is not None else self._size_of(value)
        with self._lock:
            self._drop(key)
            if self._max_bytes is not None and size > self._max_bytes:
                return
            self._store[key] = value
            self._sizes[key] = size
            self._bytes += size
            while len(self._store) > self._max or (
                    self._max_bytes is not None and self._bytes > self._max_bytes):
                # Evict least-recently used.
                self._drop(next(iter(self._store)))

    def resize(self, key: str, value: Any,
               nbytes: Optional[int] = None) -> bool:
        """Re-measure a mutable value already in this tier.

        A working value may grow after it is stored (`WALK_CELLS` builds its
        walk lazily). Its owner re-accounts it when a reader has finished
        building; if it no longer fits, it is evicted while the caller keeps
        the reference it already holds.
        """
        if self._max_bytes is None:
            return key in self._store
        size = int(nbytes) if nbytes is not None else self._size_of(value)
        with self._lock:
            if key not in self._store:
                return False
            if size > self._max_bytes:
                self._drop(key)
                return False
            self._bytes += size - self._sizes[key]
            self._sizes[key] = size
            while len(self._store) > self._max or (
                    self._max_bytes is not None
                    and self._bytes > self._max_bytes):
                evicted = next(iter(self._store))
                self._drop(evicted)
                if evicted == key:
                    return False
            return True

    def _drop(self, key: str) -> None:
        if key in self._store:
            del self._store[key]
            self._bytes -= self._sizes.pop(key)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._sizes.clear()
            self._bytes = 0


def _json_size_of(value: Any) -> int:
    return len(json.dumps(value).encode())


def deep_size_of(value: Any) -> int:
    """The resident bytes an object graph attributes to one cache value.

    Shared children are counted once within a value and process-global code
    (modules, classes, functions, and builtins) is not charged to an entry.
    This is the size function for in-process caches whose values are working
    objects rather than JSON payloads; the bytes still feed the same
    `MemoryBackend.max_bytes` budget and LRU eviction as JSON sizes.
    """
    shared_code = (
        type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
        types.MethodType, types.WrapperDescriptorType,
        types.MethodWrapperType, types.BuiltinMethodType,
    )
    seen: set[int] = set()
    stack: list[Any] = [value]
    total = 0
    while stack:
        obj = stack.pop()
        if isinstance(obj, shared_code) or id(obj) in seen:
            continue
        seen.add(id(obj))
        try:
            total += sys.getsizeof(obj)
        except TypeError:
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        else:
            slots = getattr(obj, "__slots__", None)
            if slots is not None:
                for name in slots:
                    try:
                        stack.append(getattr(obj, name))
                    except AttributeError:
                        pass
            else:
                stack.extend(gc.get_referents(obj))
    return total


_SIZE_UNITS = {"": 1, "b": 1, "k": 1 << 10, "kib": 1 << 10, "kb": 1 << 10,
               "m": 1 << 20, "mib": 1 << 20, "mb": 1 << 20,
               "g": 1 << 30, "gib": 1 << 30, "gb": 1 << 30}


def env_bytes(name: str, default: int) -> int:
    """A deployment's byte budget from the environment: a plain byte count
    or a size (`512MB`, `1.5GiB`, `2G`; the decimal spellings read 1024-based,
    the way an operator means them beside a disk size), `default` when unset.
    A malformed value is refused, not guessed: a mistyped budget stops the
    process at startup rather than silently becoming the default.

    A disk tier's budget is a fact about the volume it fills — production's
    cache volume is shared with the lake and the sol pool, a dev box's is not
    — so it is the operator's knob, with a default sized for production.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)", raw)
    if m is None or m.group(2).lower() not in _SIZE_UNITS:
        raise ValueError(f"{name}={raw!r}: expected bytes or a size like 512MB")
    return int(float(m.group(1)) * _SIZE_UNITS[m.group(2).lower()])


class DiskBackend:
    """Content-addressed JSON files under a directory, capped by file count
    and by total bytes (oldest-by-mtime evicted, the count as `max_files`
    and the bytes as `max_bytes`, `None` for no byte bound). Persists across
    restarts. Best-effort: a read or write failure never raises — the
    endpoint just recomputes.

    Values must be JSON-serializable. A changed key lands on a fresh file and
    the old one is orphaned, so the cap bounds orphan growth (e.g. one set per
    data edit); an entry whose trace went stale is replaced under its own key,
    atomically. The byte bound is what the count alone could not give: an
    entry is 21 KB or 150 MB by the value it holds, so a count of files says
    nothing about the disk it fills. One entry larger than the byte bound is
    kept alone (the newest file is never the one pruned): the bound caps
    growth at one oversized entry rather than a write-then-delete cycle."""

    def __init__(self, directory: os.PathLike | str, max_files: int = 2000,
                 max_bytes: Optional[int] = None):
        self._dir = pathlib.Path(directory)
        self._max = max_files
        self._max_bytes = max_bytes

    def get(self, key: str) -> Optional[Any]:
        try:
            with open(self._dir / f"{key}.json") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def nbytes(self, key: str) -> Optional[int]:
        """The stored value's JSON byte length — its file's — without
        reading it; None when there is no such file."""
        try:
            return (self._dir / f"{key}.json").stat().st_size
        except OSError:
            return None

    def put(self, key: str, value: Any) -> None:
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{key}.json"
            # Atomic write-tmp+rename: a reader in another process (the server
            # while `backend.api.warm` fills the same directory) sees either
            # no file or the whole file. The tmp name is unique per writer
            # thread so two writers of one key never share a half-written tmp.
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            with open(tmp, "w") as f:
                json.dump(value, f)
            os.replace(tmp, path)
            self._prune()
        except (OSError, TypeError, ValueError):
            pass

    def _prune(self) -> None:
        try:
            # One stat pass serves both bounds; the prune runs after every
            # write, so it is already O(files) and the sizes are free.
            files = []
            for p in self._dir.glob("*.json"):
                try:
                    st = p.stat()
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, p))
            files.sort()
            total = sum(s for _, s, _ in files)
            while len(files) > 1 and (
                    len(files) > self._max
                    or (self._max_bytes is not None and total > self._max_bytes)):
                _, size, p = files.pop(0)
                p.unlink(missing_ok=True)
                total -= size
        except OSError:
            pass

    def clear(self) -> None:
        try:
            for p in self._dir.glob("*.json"):
                p.unlink(missing_ok=True)
        except OSError:
            pass


class TieredBackend:
    """Hot memory (`MemoryBackend`) in front of a durable backend that knows
    its values' sizes (`DiskBackend`).

    ``get`` checks memory, then durable (and promotes a durable hit into
    memory). ``put`` writes both, durable first. The durable tier's size of
    a value (`nbytes`, a file's length) is passed to the hot tier with it,
    so a byte-bounded hot tier never serialises a value again to size it —
    one larger than its budget is turned away at once. A code/spec change
    lands on a new key, so neither tier needs an invalidation pass.
    """

    def __init__(self, hot: Any, cold: Any):
        self._hot = hot
        self._cold = cold

    def get(self, key: str) -> Optional[Any]:
        hit = self._hot.get(key)
        if hit is not None:
            return hit
        hit = self._cold.get(key)
        if hit is not None:
            self._promote(key, hit)
        return hit

    def put(self, key: str, value: Any) -> None:
        self._cold.put(key, value)
        self._promote(key, value)

    def _promote(self, key: str, value: Any) -> None:
        # None (the durable write failed) leaves the hot tier to size it.
        self._hot.put(key, value, nbytes=self._cold.nbytes(key))

    def clear(self) -> None:
        self._hot.clear()
        self._cold.clear()


class WriteBehindQueue:
    """One daemon thread that performs backend writes off the caller's thread.

    A request that just computed a value must not wait on a slow durable
    store (a Supabase upsert that times out held the request for the whole
    timeout). Writes queue here and the caller returns. The queue is bounded:
    when the store is down and writes pile up, new ones are dropped and
    logged — a cache write is an optimization, never a backlog to replay.
    A batch job that must see its writes land before exiting calls
    ``drain`` (the warm entry point does).
    """

    def __init__(self, max_pending: int = 256):
        self._q: "queue.Queue[tuple[Any, str, Any]]" = queue.Queue(max_pending)
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()

    def submit(self, backend: Any, key: str, value: Any) -> None:
        self._ensure_worker()
        try:
            self._q.put_nowait((backend, key, value))
        except queue.Full:
            _log.warning("write-behind queue full; dropping cache write %s", key[:12])

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Block until every queued write has been attempted. True when the
        queue emptied within ``timeout`` (None waits indefinitely)."""
        done = threading.Event()

        def _join() -> None:
            self._q.join()
            done.set()

        threading.Thread(target=_join, daemon=True).start()
        return done.wait(timeout)

    def _ensure_worker(self) -> None:
        if self._thread is not None:
            return
        with self._start_lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="cache-write-behind", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            backend, key, value = self._q.get()
            try:
                backend.put(key, value)
            except Exception as exc:  # a failed cache write is a dropped write
                _log.warning("write-behind put %s failed: %s", key[:12], exc)
            finally:
                self._q.task_done()


# The process's write-behind queue: every `WriteBehindBackend` built without
# an explicit queue shares it, so one `drain()` flushes them all.
WRITE_BEHIND = WriteBehindQueue()


class WriteBehindBackend:
    """A backend whose ``put`` is queued on a `WriteBehindQueue` instead of
    run inline. ``get`` / ``clear`` pass straight through. Wrap the slow
    durable tier with it (the service layer wraps the Supabase remote) so
    the request that computed a value never blocks on storing it."""

    def __init__(self, inner: Any, writes: Optional[WriteBehindQueue] = None):
        self._inner = inner
        self._writes = writes if writes is not None else WRITE_BEHIND

    def get(self, key: str) -> Optional[Any]:
        return self._inner.get(key)

    def put(self, key: str, value: Any) -> None:
        self._writes.submit(self._inner, key, value)

    def clear(self) -> None:
        self._inner.clear()


# ---------------------------------------------------------------------------
# The cache.
# ---------------------------------------------------------------------------


class KeyedCache:
    """A keyed cache over a local backend, with optional single-flight and an
    optional shared ``remote`` tier gated per request.

    With ``single_flight=True`` a per-key lock ensures two concurrent misses for
    the same key compute the value once (the second waits, then reads what the
    first stored) — the classic double-checked pattern the gallery needs so a
    cold model isn't rendered N times under concurrent page loads.

    ``remote`` (any get/put/clear backend, e.g. a Supabase-backed store) sits
    BEHIND the local backend: a local miss falls through to it and a remote hit
    is promoted into the local tiers. It is gated by a per-call ``remote_ok`` so
    a result that must NOT be shared across requests (e.g. one that folds in a
    user's private calibration) stays on the local tiers only — those requests
    neither read nor write the remote tier. Unbound (``remote=None``, e.g. OSS
    self-host without Supabase) → local-only, exactly as before.
    """

    def __init__(
        self,
        name: str,
        backend: Any,
        *,
        single_flight: bool = False,
        remote: Any = None,
    ):
        self.name = name
        self._backend = backend
        self._remote = remote
        self._single_flight = single_flight
        #: One lock per key a miss is being computed for, with how many
        #: callers hold or wait on it; dropped when the last one leaves, so
        #: the table holds only keys in flight.
        self._key_locks: dict[str, list] = {}
        self._locks_guard = threading.Lock()

    def set_remote(self, remote: Any) -> None:
        """Bind the shared remote tier AFTER construction. Lets a core cache be
        built remote-less (this module stays DB-agnostic) and the service
        layer inject a Supabase backend at import — the same inversion the
        loader uses via ``set_spec_miss_hook``. Unbound → local-only."""
        self._remote = remote

    def get(self, key: str, *, remote_ok: bool = False) -> Optional[Any]:
        """Local-tier read; with ``remote_ok`` also falls through to the
        remote tier (and promotes a remote hit local). Same fail-closed
        default as ``get_or_compute``. None for a missing entry and for one
        whose trace the running code no longer satisfies."""
        return self._local_or_remote(key, remote_ok)

    def put(
        self,
        key: str,
        value: Any,
        *,
        trace: Optional[Trace] = None,
        remote_ok: bool = False,
        remote_if: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        """Store `value`, computed by the code `trace` names — by default
        what the computation open around this call (`code_trace.provider.
        recording`) has run so far. With no computation open there is no
        telling what computed the value, and it is not stored."""
        if trace is None:
            p = provider()
            trace = p.snapshot()
            if trace is None:
                message = (f"cache {self.name}: {key} put outside any "
                           f"recorded computation (code_trace.provider."
                           f"recording); nothing says what computed it")
                if p.strict:
                    raise RuntimeError(message)
                _log.warning("%s; not stored", message)
                return
        self._store(key, value, trace, remote_ok, remote_if)

    def resize(self, key: str, value: Any) -> bool:
        """Re-account a mutable local value; see `MemoryBackend.resize`."""
        resize = getattr(self._backend, "resize", None)
        if resize is None:
            return False
        # A local entry is a value plus its trace id. Re-account that whole
        # entry — not a second, unwrapped representation of the mutable value.
        stored = self._backend.get(key)
        if not (isinstance(stored, dict) and TRACE_FIELD in stored
                and stored.get("value") is value):
            return False
        return bool(resize(key, stored))

    def _local_or_remote(self, key: str, remote_ok: bool) -> Optional[Any]:
        """Local hit, else (when shareable) a remote hit promoted into local;
        either only while its trace holds."""
        hit = self._backend.get(key)
        if hit is not None:
            return self._open(hit, remote_ok)
        if remote_ok and self._remote is not None:
            rhit = self._remote.get(key)
            if rhit is not None:
                value = self._open(rhit, remote_ok)
                if value is not None:
                    self._backend.put(key, rhit)  # promote: the next hit is local
                return value
        return None

    def _open(self, stored: Any, remote_ok: bool) -> Optional[Any]:
        """The value of a stored entry, or None when its trace is unknown
        here or stale (an entry from before traces is neither: a miss)."""
        if not (isinstance(stored, dict) and TRACE_FIELD in stored):
            return None
        trace = self._trace(stored[TRACE_FIELD], remote_ok)
        p = provider()
        if trace is None or not p.valid(trace):
            return None
        p.merge(trace)
        return stored["value"]

    def _trace(self, tid: str, remote_ok: bool) -> Optional[Trace]:
        trace = TRACES.get(tid)
        if trace is None and remote_ok and self._remote is not None:
            body = self._remote.get(REMOTE_TRACE_PREFIX + tid)
            if body is not None:
                try:
                    trace = Trace.from_json(body)
                except (TypeError, ValueError):
                    return None
                if trace.id != tid:
                    return None
                TRACES.put(trace)
        return trace

    def _store(self, key: str, value: Any, trace: Trace, remote_ok: bool,
               remote_if: Optional[Callable[[Any], bool]] = None) -> None:
        tid = TRACES.put(trace)
        stored = {TRACE_FIELD: tid, "value": value}
        self._backend.put(key, stored)
        if (
            remote_ok
            and self._remote is not None
            and (remote_if is None or remote_if(value))
        ):
            self._remote.put(REMOTE_TRACE_PREFIX + tid, trace.to_json())
            self._remote.put(key, stored)

    def get_or_compute(
        self,
        key: str,
        compute: Callable[[], Any],
        *,
        store_if: Optional[Callable[[Any], bool]] = None,
        remote_ok: bool = False,
        remote_if: Optional[Callable[[Any], bool]] = None,
    ) -> Any:
        """Return the cached value for ``key`` or compute + store it. ``store_if``
        gates whether a freshly-computed value is persisted (e.g. don't cache an
        error entry) — it is always returned regardless.

        ``remote_ok`` defaults to False (fail-closed): a caller must opt IN to the
        shared remote tier per request, so a forgotten flag keeps a result local
        rather than silently publishing it to the global table. ``remote_ok=True``
        reads and writes the remote tier (a shareable result). ``remote_if``
        further gates just the remote WRITE (the local store always happens):
        a value one machine produced but others must not inherit — e.g. an
        error entry from a transient failure — stays local."""
        hit = self._local_or_remote(key, remote_ok)
        if hit is not None:
            return hit
        if not self._single_flight:
            return self._compute(key, compute, store_if, remote_ok, remote_if)
        with self._locks_guard:
            entry = self._key_locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        try:
            with entry[0]:
                hit = self._local_or_remote(key, remote_ok)  # filled while we waited
                if hit is not None:
                    return hit
                return self._compute(key, compute, store_if, remote_ok,
                                     remote_if)
        finally:
            with self._locks_guard:
                entry[1] -= 1
                if entry[1] == 0:
                    del self._key_locks[key]

    def _compute(self, key: str, compute: Callable[[], Any],
                 store_if: Optional[Callable[[Any], bool]], remote_ok: bool,
                 remote_if: Optional[Callable[[Any], bool]]) -> Any:
        """`compute()` as one recorded computation, stored with its trace."""
        with recording() as rec:
            value = compute()
        if store_if is None or store_if(value):
            self._store(key, value, rec["trace"], remote_ok, remote_if)
        return value

    def clear(self) -> None:
        self._backend.clear()
        if self._remote is not None:
            self._remote.clear()
