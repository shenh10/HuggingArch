"""Persistent process pool for CPU-bound work.

The analysis is pure Python, so threads cannot overlap it, and a thread
that runs it holds the GIL the server's event loop needs: with cold caches
a handful of heavy requests starved `/health` past Cloudflare's 100 s.
This pool is where that work runs, in two shapes:

  * `submit(fn, ...)` — one call, one job. The API server hands each heavy
    request to the pool this way (`backend.api.offload`), and each cell of
    a sweep it serves, so its process only routes requests and moves
    results.
  * `run_parallel(fn, items)` / `map_cells` — the cells of a sweep (a seq
    sweep, a parallel-layout sweep), mapped across workers in order. A
    process that drives its sweeps itself (the out-of-process cache warm,
    the CLI) fans them out this way.

A worker never builds a pool of its own: inside a worker `pool_enabled()`
is False and a sweep called there runs its cells serially.

A job is one task to the executor, so a job of several steps has no way to
say it is moving — the future resolves once, at the end. `send_progress`
is that way: a job tagged by its submitter (`submit(progress_tag=…)`)
sends how far it has got on a queue the workers inherit, and the process
that submitted it reads the ticks with `drain_progress`. One channel, so
there is one answer to "how far did this job get" wherever the job ran.

The worker function is the caller's. It must be a top-level picklable
callable — `spawn` re-imports it in the child. Equality with the serial
path is the caller's contract, not this module's.

`spawn` start method (not fork): the pool is created lazily from inside
a request already running on starlette's threadpool, and forking a
multi-threaded process risks a child deadlock on an inherited lock.
The pool is persistent, so the one-time spawn/import cost is amortized
across all sweeps.
"""
from __future__ import annotations

import importlib
import logging
import multiprocessing as _mp
import os
import pickle
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from backend.arch.loader import default_loader_generation, invalidate_default_loader
from backend.code_trace.provider import current, provider, run_recorded

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_R = TypeVar("_R")

_pool: Optional[ProcessPoolExecutor] = None
# Serializes pool creation so two concurrent first-sweep requests
# don't build two pools (and pay the spawn/import storm twice).
_pool_lock = threading.Lock()
# Frozen by shutdown(freeze=True) so teardown cannot race a request
# that is still building the pool as the process exits.
_create_ok = True
# Modules every worker imports, and calls it makes, before its first job
# (`configure_workers`).
_worker_imports: Tuple[str, ...] = ()
_worker_prepare: Tuple[Callable[[], Any], ...] = ()
# True inside a pool worker: a worker runs sweep cells serially, never in a
# pool of its own.
_in_worker = False
# The parent's default-loader generation this worker last matched against.
_seen_loader_generation: Optional[int] = None
# The progress channel (module docstring): built with the pool and inherited
# by every worker (`_init_worker`), so one queue carries every job's ticks.
# None here means no pool was built in this process, or none in the worker's
# parent — a tick is then dropped.
_progress_q: Optional[Any] = None
# Worker-side: what its submitter tagged the job it is running
# (`submit(progress_tag=…)`), or None for a job nobody waits on the progress
# of. Thread-local would say nothing more: a worker runs one job at a time.
_progress_tag: Any = None


def configure_workers(*, imports: Tuple[str, ...] = (),
                      prepare: Tuple[Callable[[], Any], ...] = ()) -> None:
    """What each worker does when it starts, before its first job: import
    `imports`, then call each of `prepare` (top-level callables).

    The API server imports `backend.api` — importing it registers what the
    server's own import registers (the spec-miss hook, the shared remote
    cache tiers), so a job run in a worker reads specs and caches the way
    it would have in the server — and builds the spec loader, which every
    job reads. Takes effect for the next pool built."""
    global _worker_imports, _worker_prepare
    _worker_imports = tuple(imports)
    _worker_prepare = tuple(prepare)


def _die_with_parent() -> None:
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        if libc.prctl(pr_set_pdeathsig, signal.SIGKILL) == 0 and os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)
    except (OSError, AttributeError) as exc:
        logger.warning("compute worker %d cannot die with its parent: %s",
                       os.getpid(), exc)


def _init_worker(imports: Tuple[str, ...] = (),
                 prepare: Tuple[Callable[[], Any], ...] = (),
                 loader_generation: Optional[int] = None,
                 progress_q: Optional[Any] = None) -> None:
    """Start a worker: mark it one, make it die with its parent, then the
    configured imports and preparation (`configure_workers`).

    ProcessPoolExecutor workers survive a parent that exits without join
    (failed bind, crash) and get reparented to PID 1; the kernel signal
    closes that hole on Linux.

    `loader_generation` is the parent's spec-loader generation when the
    pool was built: the loader `prepare` parsed answers for it, so the
    first job that carries another generation drops it (`_run_job`). A
    worker spawned after a refine parsed the newer specs already and drops
    them once more — a reload, never a stale read.

    `progress_q` is the parent's progress channel, inherited here (it can
    be handed over only at spawn): what `send_progress` writes to."""
    global _in_worker, _seen_loader_generation, _progress_q
    _in_worker = True
    _progress_q = progress_q
    _die_with_parent()
    for name in imports:
        importlib.import_module(name)
    for fn in prepare:
        fn()
    _seen_loader_generation = loader_generation


_CGROUP_CPU_MAX = "/sys/fs/cgroup/cpu.max"


def _cgroup_cpu_quota(path: str = _CGROUP_CPU_MAX) -> Optional[int]:
    """The CPUs a cgroup v2 quota grants this process (`cpu.max`:
    "<quota> <period>", rounded up), or None when unlimited or unreadable.
    A container can see every core of the host in its affinity mask and
    still be allowed only a few cores' worth of time; a pool sized by the
    mask then runs more workers than it has CPU for, and each job takes
    that many times longer."""
    try:
        with open(path) as f:
            quota, period = f.read().split()[:2]
    except (OSError, ValueError):
        return None
    if quota == "max":
        return None
    try:
        q, p = int(quota), int(period)
    except ValueError:
        return None
    if q <= 0 or p <= 0:
        return None
    return max(1, -(-q // p))


def _usable_cpus() -> int:
    """The CPUs this process may run on: its affinity mask (which a
    container's cpuset narrows), capped by its cgroup CPU quota — not the
    machine's count."""
    try:
        cpus = len(os.sched_getaffinity(0)) or 1
    except AttributeError:  # no affinity call on this platform
        cpus = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    return cpus if quota is None else min(cpus, quota)


def _max_workers() -> int:
    """Pool size: `HA_SWEEP_POOL_WORKERS` when set (the out-of-process warm
    job sets a small one so it leaves cores to the server), else
    min(16, usable cpus - 2) (`_usable_cpus`). In the server every heavy request and every
    sweep cell is a job of this pool, so its size is the server's compute
    parallelism."""
    cpu = _usable_cpus()
    raw = os.environ.get("HA_SWEEP_POOL_WORKERS", "").strip()
    if raw:
        return max(0, min(int(raw), cpu))
    return max(0, min(16, cpu - 2))


def worker_count() -> int:
    """How many workers the pool runs (or would run)."""
    return _max_workers()


def in_worker() -> bool:
    """Whether this process is one of the pool's workers (each holds its own
    copy of every per-process cache, so a cache sized for the server is
    sized smaller here)."""
    return _in_worker


def in_test_run() -> bool:
    """Under pytest: tests monkeypatch in-process (the amortization tests
    assert preparation helpers run exactly once, only true on the serial
    path), so nothing goes to the pool unless a test asks for it."""
    return "PYTEST_CURRENT_TEST" in os.environ


def _pool_allowed() -> bool:
    """OFF under pytest, when explicitly disabled, and inside a worker."""
    if os.environ.get("HUGGINGARCH_NO_SWEEP_POOL"):
        return False
    if in_test_run():
        return False
    return not _in_worker


def pool_enabled() -> bool:
    """Whether a sweep fans its cells out (`run_parallel`): the pool is
    allowed and the box has the cores for two workers to win."""
    return _pool_allowed() and _max_workers() >= 2


def jobs_enabled() -> bool:
    """Whether `submit` has a pool to run on: allowed, one worker or more."""
    return _pool_allowed() and _max_workers() >= 1


def _get_pool(min_workers: int = 2,
              honor_freeze: bool = True) -> Optional[ProcessPoolExecutor]:
    global _pool, _progress_q
    if (honor_freeze and not _create_ok) or _in_worker:
        return None
    workers = _max_workers()
    if workers < min_workers:
        return None
    if _pool is not None:
        return _pool
    with _pool_lock:
        if honor_freeze and not _create_ok:
            return None
        if _pool is not None:  # built while we waited on the lock
            return _pool
        try:
            ctx = _mp.get_context("spawn")
            # The progress channel is built with the pool: a queue can be
            # handed to another process only as it is spawned, so it goes in
            # the initializer's arguments and every worker of this pool
            # inherits the one queue.
            queue = ctx.Queue()
            _pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=ctx, initializer=_init_worker,
                initargs=(_worker_imports, _worker_prepare,
                          default_loader_generation(), queue),
            )
            _progress_q = queue
        except Exception:
            _pool = None
    return _pool


def send_progress(done: int, total: int) -> None:
    """From inside a job: `done` of `total` steps finished, carried to the
    process that submitted it (`drain_progress`). A no-op in a job whose
    submitter did not tag it (`submit(progress_tag=…)`) and outside a
    worker — a caller in the submitting process has the work in hand and
    needs no channel.

    Best-effort and never blocking: a tick is a hint about work that is
    still running, so a full or closed queue drops it rather than hold the
    job up."""
    if _progress_q is None or _progress_tag is None:
        return
    try:
        _progress_q.put_nowait((_progress_tag, int(done), int(total)))
    except Exception as exc:
        logger.debug("progress tick dropped: %s", exc)


def drain_progress() -> List[Tuple[Any, int, int]]:
    """Every tick the jobs submitted from this process have sent since the
    last drain — `(progress_tag, done, total)`, oldest first. Non-blocking;
    empty when no pool was built here."""
    out: List[Tuple[Any, int, int]] = []
    if _progress_q is None:
        return out
    while True:
        try:
            out.append(_progress_q.get_nowait())
        except Exception:  # Empty, or a queue the pool's teardown closed
            return out


def run_parallel(fn: Callable[[_T], _R], items: List[_T]) -> List[_R]:
    """Map `fn` across `items` in worker processes, same order.

    Raises if the pool is unavailable or any item fails, so the caller
    can fall back to the serial path. `fn` must be a top-level function
    the spawned child can import; the items must be picklable.

    An explicit call: it bypasses the `pool_enabled` gate (callers gate
    themselves) and the server-teardown freeze, which guards the request
    path (`submit`) as a server exits.
    """
    pool = _get_pool(honor_freeze=False)
    if pool is None:
        raise RuntimeError("sweep pool unavailable")
    # Submit in order, collect by index — `.result()` re-raises any
    # worker exception here so the caller's try/except can fall back.
    inherited, requester = provider().snapshot(), current()
    try:
        futures = [pool.submit(run_recorded, inherited, fn, item)
                   for item in items]
        out = []
        for f in futures:
            value, trace = f.result()
            provider().merge(trace, into=requester)
            out.append(value)
        return out
    except BrokenProcessPool:
        # A worker died (OOM kill, segfault) and the executor refuses all
        # further work. Drop it so the next call builds a fresh pool.
        _discard(pool)
        raise


def submit(fn: Callable[..., _R], /, *args: Any,
           progress_tag: Any = None, **kwargs: Any) -> "Future[_R]":
    """Run `fn(*args, **kwargs)` as one job in a worker; the future resolves
    to its return value or raises its exception (the exception's own class,
    `_Portable`).

    The job first matches the worker's spec loader to this process's
    (`default_loader_generation`): a spec refined here is the spec a
    worker reads next. Raises RuntimeError when there is no pool. A pool a
    dead worker broke is dropped, so the next job builds a fresh one.

    `progress_tag` names this job on the progress channel: what it sends
    while it runs (`send_progress`) comes back under that tag
    (`drain_progress`). Untagged, a job's ticks are dropped. Keyword-only,
    so it is this module's argument and not one of `fn`'s."""
    pool = _get_pool(min_workers=1)
    if pool is None:
        raise RuntimeError("compute pool unavailable")
    # The job is a computation of its own in the worker; it starts from what
    # the requester had computed when it handed the job over (its arguments
    # came from there), and its trace comes back to the requester's
    # computations — not to every computation this process has open.
    inherited, requester = provider().snapshot(), current()
    try:
        fut = pool.submit(_run_job, default_loader_generation(), fn, args,
                          kwargs, progress_tag, inherited)
    except BrokenProcessPool:
        _discard(pool)
        raise
    out: "Future[_R]" = Future()

    def _settle(done: Future) -> None:
        if done.cancelled():
            out.cancel()
            return
        exc = done.exception()
        if exc is not None:
            if isinstance(exc, BrokenProcessPool):
                _discard(pool)
            # The worker's failure still owns the pool's state even when the
            # caller has cancelled its bridge; otherwise one cancelled request
            # hides a broken pool from every later one.
            if out.cancelled():
                return
            out.set_exception(exc)
            return
        # A caller may cancel the bridge while the worker job is already
        # running.  The bridge's state is this side's fact; a late worker
        # answer must not overwrite it.
        if out.cancelled():
            return
        value, trace = done.result()
        provider().merge(trace, into=requester)
        out.set_result(value)

    fut.add_done_callback(_settle)
    return out


def _ready() -> int:
    return os.getpid()


def prestart() -> None:
    """Start every worker now, each importing and parsing what a job needs,
    so the first requests do not wait on it. A no-op without a pool."""
    if not jobs_enabled():
        return
    pool = _get_pool(min_workers=1)
    if pool is None:
        return
    # One job per worker slot: no worker is idle yet, so each one spawns.
    for _ in range(_max_workers()):
        pool.submit(_ready)


def _run_job(loader_generation: int, fn: Callable[..., _R],
             args: tuple, kwargs: dict, progress_tag: Any = None,
             inherited: Any = None) -> Tuple[_R, Any]:
    """Worker side of `submit`: the job, under the tag its ticks travel
    under (`send_progress`), as one recorded computation
    (`code_trace.provider.run_recorded`) — its value and its trace."""
    global _seen_loader_generation, _progress_tag
    if loader_generation != _seen_loader_generation:
        # The loader this worker holds answered for another generation (or
        # for none it knows of): parse the specs again.
        invalidate_default_loader()
        _seen_loader_generation = loader_generation
    _progress_tag = progress_tag
    try:
        return run_recorded(inherited, fn, *args, **kwargs)
    except Exception as exc:
        try:
            pickle.loads(pickle.dumps(exc))
        except Exception:
            raise _Portable(exc) from None
        raise
    finally:
        _progress_tag = None


def _rebuild_exception(cls: type, args: tuple, state: dict) -> BaseException:
    exc = cls.__new__(cls, *args)
    exc.args = args
    exc.__dict__.update(state)
    return exc


class _Portable(Exception):
    """Carries an exception whose class cannot be rebuilt from its `args`
    (an `__init__` with its own signature, e.g. `IndivisibleHeadRepartition`)
    across the process boundary. Unpickles as the original class with the
    original args and attributes, so the caller's `except` clauses see the
    exception the job raised."""

    def __init__(self, exc: BaseException):
        super().__init__(str(exc))
        self._reduced = (_rebuild_exception,
                         (type(exc), exc.args, dict(vars(exc))))
        try:
            pickle.dumps(self._reduced)
        except Exception:
            self._reduced = (RuntimeError, (f"{type(exc).__name__}: {exc}",))

    def __reduce__(self):
        return self._reduced


def map_cells(fn: Callable[[_T], _R], items: List[_T]) -> List[_R]:
    """`fn` over `items`, in order: across the pool when this process fans
    sweeps out (`pool_enabled`), else here. Any pool failure (unavailable, a
    crashed worker) falls through to the serial path; a cell's own failure
    is the cell's to isolate."""
    if pool_enabled() and len(items) > 1:
        try:
            return run_parallel(fn, items)
        except Exception as exc:
            logger.warning("sweep pool failed (%s); running %d cells here",
                           exc, len(items))
    return [fn(item) for item in items]


def _discard(pool: ProcessPoolExecutor) -> None:
    """Forget `pool` if it is still the live one, then shut it down. Its
    progress channel goes with it: the workers that could write to it are
    gone, and the next pool builds its own."""
    global _pool, _progress_q
    with _pool_lock:
        if _pool is pool:
            _pool = None
            _progress_q = None
    pool.shutdown(wait=False, cancel_futures=True)


def shutdown(*, freeze: bool = False) -> None:
    """Tear down the pool. `freeze=True` (FastAPI lifespan teardown) also
    blocks a racing request from building a replacement as the process
    exits."""
    global _pool, _create_ok, _progress_q
    if freeze:
        _create_ok = False
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
    _progress_q = None
