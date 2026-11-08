"""Heavy request work off the event loop, answered within a deadline.

The server's event loop must stay free to route requests: `/health`, the
static pages and every light endpoint run on it. A heavy endpoint (an
analysis, an estimate, a sweep, the gallery) hands its work here instead:

  * `run_off_loop(fn, ...)` runs one call in the compute pool
    (`backend.measure.pool.submit`) when this process serves
    requests (`serve_requests_off_process`, set by the server's lifespan),
    else in a thread — the warm job, the CLI and the tests, which fan
    sweeps out across the pool themselves or patch in-process.
  * `within_deadline(key, start)` answers a request with the work `start`
    begins, or, past the deadline, with `StillComputing` (a 503 the client
    retries). The work goes on; a request for the same `key` joins it, and
    a result that finished after its caller gave up is held for a while so
    the retry finds it. The process's caches (disk, remote) are written by
    the worker that computed the value, so a retry that lands after the
    hold also reads it there. Work no request has waited on for
    `UNWATCHED_S` is cancelled (`_Work.leave`): a client that went
    elsewhere — another GPU on the gallery — leaves no work behind to
    compete with the work it now waits on; what that work already kept in
    the caches stays there for the next request. Work of several jobs says
    how far it got (`report_progress`); the 503 carries it, and how many
    compute jobs the work has finished (`jobs_done`), so a waiting client
    can say more than "still computing" and tell work that moves from work
    that is stuck. `report_progress` is the one entry point whether the
    steps are counted here or inside a compute worker: a worker's tick
    travels on the pool's progress channel (`pool.send_progress`) and is
    folded into the same work here (`_fold_progress`), so work whose whole
    schedule is one job — a (context × batch) sweep is one walk, hence one
    cell — still reports movement.
  * `offload(fn, ...)` is the two together for one call, keyed on the call;
    `offload_response` also renders the JSON response there.
  * `map_off_loop(fn, items)` runs the cells of a sweep, one compute job
    each, so a sweep spreads across the workers. With `progress=True` it
    counts the finished cells into the work's progress (`report_progress`),
    so a retry's 503 says how many of the sweep's cells are done.
  * The cells of every request share one set of slots (`_cell_slots`), and
    a free slot goes to the most urgent waiter, not the oldest
    (`PrioritySlots`): work a client polled most recently first — the
    view on screen before a view the client left — and within one work the
    items the client named first (`within_deadline(..., hint=)`: the models
    on screen, the one it picked). A single job (`offload`, an estimate)
    takes no slot, so the cells never hold up the request a user waits on.
  * Work that yields results one item at a time publishes them as it goes
    (`report_partial`); the 503 carries the items a client does not have yet
    (`within_deadline(..., cursor=)`), so a page draws each as it lands
    instead of waiting for the slowest.
  * Whether a request starts work or joins work already running (or held)
    is decided here, in `within_deadline`, and nowhere else. Whoever meters
    the work clients start — the service's rate limiter — is told of each
    start (`set_start_hook`) and may refuse it by raising; a join, and the
    retries a waiting page sends while its work runs, are not starts.

`fn` is a top-level function and its arguments are picklable — the pool
runs it in another process.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import itertools
import logging
import os
import pickle
import time
import weakref
from concurrent.futures.process import BrokenProcessPool
from typing import (Any, AsyncIterator, Awaitable, Callable, Dict, List,
                    Optional, Sequence, Tuple, TypeVar)

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from backend.api._models import InferenceEstimateRequest
from backend.measure.scope import of_request
from backend.arch.loader import default_loader_generation, get_default_loader
from backend.code_trace.provider import seal as code_trace_seal
from backend.fingerprint import engine_code_version, engine_version, shared_data_version
from backend.measure import pool

_T = TypeVar("_T")
logger = logging.getLogger(__name__)

# How long one request waits on its work before it answers "still
# computing" with the progress so far. Cloudflare drops a response that takes
# 100 s, the ceiling; the progress a page shows moves once per deadline, so
# the wait is short. The result is not later for it: work that finishes
# between two asks is held (`_HOLD_S`) and the next ask, `RETRY_AFTER_S`
# away, takes it.
DEADLINE_S = float(os.environ.get("HA_REQUEST_DEADLINE_S", "20"))
RETRY_AFTER_S = 5
# Work that publishes as it goes (`report_partial`) answers a waiting client
# as soon as it has something new rather than at the deadline — after this
# grace, so items landing together go out together — and asks it back
# sooner: its next item may be a second away.
_NEWS_GRACE_S = 0.5
NEWS_RETRY_AFTER_S = 1
# How long a result whose caller timed out stays for the retry, and how many.
_HOLD_S = 600.0
_HOLD_MAX = 32
# How long running work goes on with no request waiting on it before it is
# cancelled. Well past a client's retry interval (`RETRY_AFTER_S`): a page
# in a background tab has its timers throttled to about once a minute.
UNWATCHED_S = float(os.environ.get("HA_UNWATCHED_WORK_S", "120"))

_off_process = False
_deadline_s: Optional[float] = None
#: Told, in the request's own task, that the request is about to start new
#: work (`within_deadline`); raising refuses the work, which is then never
#: begun. Set by the service layer (`set_start_hook`); None meters nothing.
_start_hook: Optional[Callable[[], None]] = None


class StillComputing(Exception):
    """The request's work did not finish within the deadline; it is still
    running, and the same request will join it. `progress` is how far it
    got, `(done, total)` jobs, when the work reports it; `jobs_done` how
    many compute jobs it has finished — it grows while the work moves, even
    between two steps of its `progress`."""

    reason = "still_computing"

    def __init__(self, message: str = "",
                 progress: Optional[Tuple[int, int]] = None,
                 jobs_done: Optional[int] = None,
                 partial: Optional[Dict[str, Any]] = None,
                 retry_after_s: Optional[float] = None):
        super().__init__(message)
        self.progress = progress
        self.jobs_done = jobs_done
        #: How soon to ask again, when sooner than `RETRY_AFTER_S`: the work
        #: answered early because it published something new.
        self.retry_after_s = retry_after_s
        #: The items the work has published that the client does not have
        #: yet (`report_partial`): `{"tag", "count", "head", "items"}`, or
        #: None for work that publishes none.
        self.partial = partial


class SpecChanged(StillComputing):
    """A spec was refined while a request of several jobs ran, so its jobs
    may have read two versions of it. Nothing is stored or served; the same
    request, sent again, starts over on the new spec. Clients treat it as
    `still_computing`."""


class ComputeUnavailable(Exception):
    """The compute pool cannot run the request: switched off, not buildable,
    or its worker died (the pool is rebuilt). The request can be sent
    again."""

    reason = "compute_unavailable"


def spec_generation() -> int:
    """The spec version this server process answers with: the default
    loader's generation (`invalidate_default_loader` moves it). Part of
    every request's identity, so a request sent after a refine never joins
    work begun before it."""
    return default_loader_generation()


def ensure_spec_generation(generation: int) -> None:
    """Raise `SpecChanged` when the spec moved since `generation` — called
    by a request of several jobs before it stores or serves their result."""
    if default_loader_generation() != generation:
        raise SpecChanged(
            "A model spec changed while this request was computing; retry "
            "to compute it on the new spec.")


def serve_requests_off_process(*, deadline_s: Optional[float] = DEADLINE_S) -> None:
    """Called once by the server's lifespan: heavy work runs in the compute
    pool and each request is answered within `deadline_s`. Workers import
    `backend.api` so their hooks and cache tiers are the server's, and parse
    the spec library and hash the engine (every cache key reads it, ~5 s
    each) before their first job."""
    global _off_process, _deadline_s
    pool.configure_workers(
        imports=("backend.api",),
        prepare=(code_trace_seal, get_default_loader, engine_code_version,
                 shared_data_version, engine_version))
    _off_process = True
    _deadline_s = deadline_s


def stop_serving() -> None:
    """The server's lifespan is over: back to in-process work, no deadline."""
    global _off_process, _deadline_s
    pool.configure_workers()
    _off_process = False
    _deadline_s = None


def prestart_workers_later(delay_s: float = 5.0) -> None:
    """Start the compute workers `delay_s` after the server comes up. Not in
    the lifespan itself: it runs before the port binds, and a failed bind
    exits before this fires."""
    if _off_process:
        asyncio.get_running_loop().call_later(delay_s, pool.prestart)


def set_start_hook(hook: Optional[Callable[[], None]]) -> None:
    """Who is told that a request starts new work (`_start_hook`)."""
    global _start_hook
    _start_hook = hook


def set_deadline(deadline_s: Optional[float]) -> None:
    """The request deadline; None waits for the work however long it takes."""
    global _deadline_s
    _deadline_s = deadline_s


async def run_off_loop(fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """`fn(*args, **kwargs)` in the compute pool (or, outside the server, a
    thread), awaited without blocking the loop.

    In the server there is no thread to fall back to: heavy work on the
    server's threads is the outage this module exists to prevent. A pool
    that is switched off, cannot be built, or lost its worker answers
    `ComputeUnavailable` (503)."""
    if not _off_process:
        out = await asyncio.to_thread(fn, *args, **kwargs)
        _job_finished()
        return out
    if not pool.jobs_enabled():
        raise ComputeUnavailable(
            "The compute pool is switched off on this server "
            "(HUGGINGARCH_NO_SWEEP_POOL / HA_SWEEP_POOL_WORKERS=0).")
    try:
        # Tagged with the work this job runs for, so what it reports while
        # it runs (`report_progress` inside the worker) is folded into that
        # work and not another request's.
        fut = pool.submit(fn, *args, progress_tag=_reporting_tag(), **kwargs)
    except (RuntimeError, OSError, BrokenProcessPool) as exc:
        raise ComputeUnavailable(
            f"The compute pool could not take this request ({exc}); "
            "send it again.") from exc
    try:
        out = await asyncio.wrap_future(fut)
    except BrokenProcessPool as exc:
        raise ComputeUnavailable(
            "The worker computing this request stopped; send it again."
        ) from exc
    _job_finished()
    return out


class PrioritySlots:
    """`n` slots, each granted to the most urgent waiter.

    A waiter's urgency is a zero-argument callable read when a slot is
    granted, not when the waiter arrived: a client that switches view or
    picks another model reorders what is already queued. Grants happen on
    the next turn of the loop, so waiters that arrive together — the models
    of one frontier, gathered at once — are ranked together rather than the
    first come taking the free slots. Ties go to the earlier waiter, so
    without urgencies this is a FIFO semaphore."""

    def __init__(self, n: int) -> None:
        self._free = n
        self._seq = itertools.count()
        self._waiters: List[Tuple[Callable[[], Any], int, "asyncio.Future[None]"]] = []
        self._granting = False

    @contextlib.asynccontextmanager
    async def hold(self, urgency: Callable[[], Any]) -> AsyncIterator[None]:
        fut: "asyncio.Future[None]" = asyncio.get_running_loop().create_future()
        entry = (urgency, next(self._seq), fut)
        self._waiters.append(entry)
        self._schedule()
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self._release()        # granted, then cancelled: pass it on
            elif entry in self._waiters:
                self._waiters.remove(entry)
            raise
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        self._free += 1
        self._schedule()

    def _schedule(self) -> None:
        if not self._granting:
            self._granting = True
            asyncio.get_running_loop().call_soon(self._grant)

    def _grant(self) -> None:
        self._granting = False
        self._waiters = [w for w in self._waiters if not w[2].done()]
        while self._free > 0 and self._waiters:
            nxt = min(self._waiters, key=lambda w: (w[0](), w[1]))
            self._waiters.remove(nxt)
            self._free -= 1
            nxt[2].set_result(None)


def urgency(name: Optional[str] = None) -> Callable[[], Any]:
    """How urgent a slot is for the work this task runs for, read when a
    slot frees: work asked for more recently first (the view a client is
    on), then `name`'s place in the order the client asked for (`hint`).
    Outside a request (the warm job) every waiter ties and order is FIFO."""
    work = _current_work.get()
    if work is None:
        return lambda: (0.0, 0)

    def read() -> Tuple[float, int]:
        rank = len(work.hint)
        if name is not None:
            rank = work.hint.get(name, rank)
        return (-work.asked_at, rank)

    return read


#: Per event loop: the slots sweep cells hold while in the pool
#: (`_cell_slots`).
_cell_slot_tables: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, PrioritySlots]" = (
    weakref.WeakKeyDictionary())


def _cell_slots() -> PrioritySlots:
    """Every worker but one, for sweep cells. A sweep is hundreds of cells;
    handed to the pool together they fill its one FIFO queue, and a request
    that is a single job — a one-second lookup, an estimate — waits behind
    every one of them (after a restart `/supported-models` sat behind the
    Scaling sweeps). Capped, the cells leave a worker to the single jobs;
    a one-worker pool still bounds the wait to one cell. A freed slot goes
    to the most urgent cell (`PrioritySlots`, `urgency`)."""
    loop = asyncio.get_running_loop()
    slots = _cell_slot_tables.get(loop)
    if slots is None:
        slots = _cell_slot_tables[loop] = PrioritySlots(
            max(1, pool.worker_count() - 1))
    return slots


async def run_ranked(fn: Callable[..., _T], /, *args: Any,
                     name: Optional[str] = None) -> _T:
    """`run_off_loop` for one job of a batch — a step of one model of the
    gallery — in a cell slot at `name`'s urgency (`urgency`), so a batch's
    steps queue with its cells and leave the single jobs their worker."""
    if not _off_process:
        return await run_off_loop(fn, *args)
    async with _cell_slots().hold(urgency(name)):
        return await run_off_loop(fn, *args)


async def map_off_loop(fn: Callable[[Any], _T], items: List[Any],
                       *, progress: bool = False,
                       name: Optional[str] = None,
                       on_result: Optional[Callable[[int, _T], None]] = None
                       ) -> List[_T]:
    """`fn` over `items`, in order — the cells of a sweep. In the server each
    cell is its own compute job, so one sweep spreads across the workers
    beside every other request's jobs, in at most every worker but one at a
    time (`_cell_slots`); elsewhere the cells go to this process's own
    fan-out (`pool.map_cells`) on a thread. With `progress` the finished
    cells are counted into the work's progress (`report_progress`), so a
    caller that misses the deadline reads "k of n cells done" in its 503;
    the count is per cell in the server, start/end elsewhere (a thread's
    serial map has no between to report). `name` is what the client's order
    knows these cells by (`urgency`) — the model they belong to.
    `on_result(index, result)` is told of each cell as it finishes, in the
    server (a thread's serial map has no between: it is told of none)."""
    if _off_process:
        slots = _cell_slots()
        done = 0
        total = len(items)
        if progress:
            report_progress(0, total)

        rank = urgency(name)

        async def cell(index: int, item: Any) -> _T:
            nonlocal done
            async with slots.hold(rank):
                out = await run_off_loop(fn, item)
            if progress:
                done += 1
                report_progress(done, total)
            if on_result is not None:
                on_result(index, out)
            return out

        return list(await asyncio.gather(
            *(cell(i, item) for i, item in enumerate(items))))
    if progress:
        report_progress(0, len(items))
        out = await asyncio.to_thread(pool.map_cells, fn, items)
        report_progress(len(items), len(items))
        return out
    return await asyncio.to_thread(pool.map_cells, fn, items)


class _Work:
    """One key's work: the task, until when its result is held, how far it
    got (`report_progress`), how many compute jobs it finished, and how many
    requests wait on it now."""

    __slots__ = ("__weakref__", "abandoned", "asked_at", "held_until", "hint",
                 "jobs_done", "left_at", "news", "partial_head",
                 "partial_items", "progress", "tag", "task", "waiting")

    def __init__(self) -> None:
        self.task: "asyncio.Task[Any]"
        self.abandoned = False
        self.held_until: Optional[float] = None
        self.progress: Optional[Tuple[int, int]] = None
        self.jobs_done = 0
        self.waiting = 0
        self.left_at: Optional[float] = None
        #: When a request last asked for this work: the more recent, the
        #: more urgent its cells (`urgency`).
        self.asked_at = time.monotonic()
        #: The client's order of this work's items, by name (`urgency`).
        self.hint: Dict[str, int] = {}
        #: What the work published so far (`report_partial`).
        self.partial_head: Optional[Dict[str, Any]] = None
        self.partial_items: List[Any] = []
        #: Set, and replaced, each time the work publishes (`report_partial`):
        #: a request waiting on the work answers then (`_result_or_news`).
        self.news = asyncio.Event()
        #: What this work's jobs are tagged with on the pool's progress
        #: channel (`_reporting`). An int, so it crosses to a worker and
        #: back by value and holds no reference to the work itself.
        self.tag = next(_work_tags)
        _reporting[self.tag] = self

    def leave(self, table: Dict[str, "_Work"], key: str) -> None:
        """A request stops waiting on the work (answered, timed out, or
        gone). When it was the last and the work is still running, the work
        is cancelled `UNWATCHED_S` later unless a request has come back to
        it meanwhile — the retry of a client still waiting joins it long
        before."""
        self.waiting -= 1
        if self.waiting or self.task.done():
            return
        left_at = self.left_at = time.monotonic()

        def drop() -> None:
            if (table.get(key) is self and not self.waiting
                    and self.left_at == left_at and not self.task.done()):
                logger.info("offload: cancelling %s — no request has waited "
                            "on it for %.0f s", key, UNWATCHED_S)
                # Forgotten first: a cancelled task is not done until it
                # unwinds, and a retry arriving meanwhile must start new
                # work, not join this one and inherit its CancelledError.
                del table[key]
                self.task.cancel()

        asyncio.get_running_loop().call_later(UNWATCHED_S, drop)


# The work the running task belongs to: set while `within_deadline` creates
# the task, so the task (and every task it gathers) carries it in its context.
_current_work: "contextvars.ContextVar[Optional[_Work]]" = contextvars.ContextVar(
    "offload_current_work", default=None)

#: Work a compute worker can report into, by its tag (`_Work.tag`). Weak:
#: a tick for work that has finished or been cancelled finds nothing and is
#: dropped, and the entry goes when the work does.
_work_tags = itertools.count(1)
_reporting: "weakref.WeakValueDictionary[int, _Work]" = weakref.WeakValueDictionary()


def _reporting_tag() -> Optional[int]:
    """The tag a job submitted now reports under: the work this task runs
    for, or None outside `within_deadline` (the warm job, the CLI — nobody
    is waiting on a 503)."""
    work = _current_work.get()
    return None if work is None else work.tag


def _fold_progress() -> None:
    """Fold what the compute workers reported into the work each tick
    belongs to (`pool.drain_progress`). Called where this process reads or
    writes the ledger — a 503's progress, a finished job — so the channel is
    emptied often enough to hold only the ticks of the jobs running now, and
    no background task polls it."""
    for tag, done, total in pool.drain_progress():
        work = _reporting.get(tag)
        if work is not None:
            work.progress = (done, total)


def _job_finished() -> None:
    """Count a finished compute job into the work this task runs for."""
    _fold_progress()
    work = _current_work.get()
    if work is not None:
        work.jobs_done += 1


def report_progress(done: int, total: int) -> None:
    """Record that the work this task runs for has `done` of `total` steps
    finished; a request that misses the deadline gets it in its 503.

    The one entry point wherever the steps are counted. In this process the
    work is in hand. Inside a compute worker it is not — the work belongs to
    the server that submitted the job — so the tick travels there on the
    pool's progress channel (`pool.send_progress`, folded by
    `_fold_progress`); that is how work whose whole schedule is one job
    reports movement. A no-op in neither (the CLI, an untagged job)."""
    work = _current_work.get()
    if work is None:
        pool.send_progress(int(done), int(total))
        return
    work.progress = (int(done), int(total))


def report_partial(item: Any = None, *,
                   head: Optional[Dict[str, Any]] = None) -> None:
    """Publish one finished item of the work this task runs for — and, once,
    the `head` every item is drawn under (the page's frame: its axes, its
    config) — for a request that misses the deadline to take with its 503
    (`within_deadline(..., cursor=)`). A no-op outside a request (the warm
    job, the CLI: nobody draws a partial answer)."""
    work = _current_work.get()
    if work is None:
        return
    if head is not None:
        work.partial_head = head
    if item is not None:
        work.partial_items.append(item)
    published, work.news = work.news, asyncio.Event()
    published.set()


class _News(Exception):
    """The work published something the waiting client does not have."""


async def _result_or_news(work: "_Work", deadline_s: float,
                          have: Optional[int]) -> Any:
    """The work's result; or `_News` once it has published what the client
    lacks — at once when it already holds more items than the client's
    `have`, else when it next publishes — after `_NEWS_GRACE_S`, so what
    lands together goes out together; or `asyncio.TimeoutError` at the
    deadline. The work goes on either way."""
    loop = asyncio.get_running_loop()
    end = loop.time() + deadline_s
    result = asyncio.shield(work.task)
    try:
        behind = have is not None and len(work.partial_items) > have
        if not behind:
            news = asyncio.ensure_future(work.news.wait())
            try:
                done, _ = await asyncio.wait(
                    {result, news}, timeout=max(0.0, end - loop.time()),
                    return_when=asyncio.FIRST_COMPLETED)
            finally:
                news.cancel()
            if result in done:
                return result.result()
            if not done:
                raise asyncio.TimeoutError
        grace = max(0.0, min(_NEWS_GRACE_S, end - loop.time()))
        done, _ = await asyncio.wait({result}, timeout=grace)
        if done:
            return result.result()
        raise _News
    finally:
        if not result.done():
            result.cancel()       # the shield only: the work goes on


# Per event loop: a task belongs to the loop that runs it (a test client
# starts a loop of its own).
_tables: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[str, _Work]]" = (
    weakref.WeakKeyDictionary())


def _table() -> Dict[str, _Work]:
    loop = asyncio.get_running_loop()
    table = _tables.get(loop)
    if table is None:
        table = _tables[loop] = {}
    return table


def _settle(table: Dict[str, _Work], key: str, work: _Work) -> None:
    """The work is done: forget it, or hold a successful result whose caller
    gave up so the retry finds it."""
    if table.get(key) is not work:
        return
    task = work.task
    if not work.abandoned or task.cancelled() or task.exception() is not None:
        del table[key]
        return
    work.held_until = time.monotonic() + _HOLD_S
    held = sorted((w.held_until, k) for k, w in table.items()
                  if w.held_until is not None)
    for _until, k in held[: max(0, len(held) - _HOLD_MAX)]:
        del table[k]


def _expire(table: Dict[str, _Work]) -> None:
    now = time.monotonic()
    for k in [k for k, w in table.items()
              if w.held_until is not None and w.held_until < now]:
        del table[k]


def _own(value: _T) -> _T:
    """One caller's copy of a shared result. A `Response` is sent by each
    caller that joined the work, and sending one is not read-only: the gzip
    middleware rewrites its headers in place (`Content-Encoding`, the
    compressed `Content-Length`), so a second send of the same object
    declares the first send's length over the raw body and the client sees
    a truncated response. Each caller gets the bytes under headers of its
    own."""
    if not isinstance(value, Response) or not hasattr(value, "body"):
        return value
    own = Response(content=value.body, status_code=value.status_code)
    own.raw_headers = list(value.raw_headers)
    return own  # type: ignore[return-value]


async def within_deadline(key: str, start: Callable[[], Awaitable[_T]], *,
                          hint: Optional[Sequence[str]] = None,
                          cursor: Optional[Tuple[int, int]] = None) -> _T:
    """The result of the work `start()` begins for `key` — joined if that
    work is already running or held — or `StillComputing` past the deadline.

    Asking makes the work the most recently asked for (`urgency`); `hint`,
    when given, is the order the client wants its items in. `cursor` is
    `(tag, count)`: the work and how many of its published items the client
    already has; the 503 carries the rest (all of them when the tag is not
    this work's — the work was cancelled and begun again).

    Beginning the work is a start (`_start_hook`, which may refuse it);
    joining it is not. Work asked for by other work (`_current_work` is
    set: a step inside a build) is that work's, not a request's start."""
    table = _table()
    _expire(table)
    work = table.get(key)
    if work is None:
        if _start_hook is not None and _current_work.get() is None:
            _start_hook()
        work = _Work()
        token = _current_work.set(work)
        try:
            work.task = asyncio.ensure_future(start())
        finally:
            _current_work.reset(token)
        table[key] = work
        work.task.add_done_callback(lambda _t, w=work: _settle(table, key, w))
    work.asked_at = time.monotonic()
    if hint is not None:
        work.hint = {name: i for i, name in enumerate(hint)}
    work.waiting += 1
    try:
        if _deadline_s is None:
            return _own(await asyncio.shield(work.task))
        have = (None if cursor is None
                else cursor[1] if cursor[0] == work.tag else 0)
        try:
            return _own(await _result_or_news(work, _deadline_s, have))
        except (asyncio.TimeoutError, _News) as why:
            work.abandoned = True
            # The ledger is read now, so take what the workers reported
            # since the last fold first.
            _fold_progress()
            partial = None
            if work.partial_head is not None or work.partial_items:
                have = (cursor[1] if cursor is not None
                        and cursor[0] == work.tag else 0)
                partial = {"tag": work.tag,
                           "count": len(work.partial_items),
                           "head": work.partial_head,
                           "items": work.partial_items[have:]}
            raise StillComputing(
                "Still computing — the result is not ready yet. Retry in a "
                "few seconds; the computation continues while it is asked "
                "for.",
                progress=work.progress, jobs_done=work.jobs_done,
                partial=partial,
                retry_after_s=(NEWS_RETRY_AFTER_S if isinstance(why, _News)
                               else None)) from None
    finally:
        work.leave(table, key)


def _call_part(arg: Any) -> Any:
    """What a call argument contributes to its call key: an estimate request
    stands for its result's identity (`measure.scope.of_request` — the one
    answer to what identifies an estimate's result), anything else for
    itself."""
    if isinstance(arg, InferenceEstimateRequest):
        return ("estimate-request", type(arg).__name__,
                of_request("call", arg, None).key)
    return arg


def call_key(fn: Callable[..., Any], args: Tuple[Any, ...],
             kwargs: Dict[str, Any]) -> str:
    """The identity of one call in this process: the spec generation, the
    function object and its pickled arguments (an estimate request by its
    result identity, `_call_part`). After a refine the same call is a new
    key: it neither joins nor is served work begun on the old spec."""
    raw = pickle.dumps((tuple(_call_part(a) for a in args),
                        sorted((k, _call_part(v)) for k, v in kwargs.items())), protocol=4)
    return (f"g{spec_generation()}:{id(fn):x}:"
            f"{hashlib.sha256(raw).hexdigest()}")


async def offload(fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """One heavy call for one request: off the loop, within the deadline,
    joined with an identical call already running."""
    return await within_deadline(
        call_key(fn, args, kwargs),
        lambda: run_off_loop(fn, *args, **kwargs))


def render_json(value: Any) -> Response:
    """The response FastAPI makes of a route's return value (a `Response`
    passes as it is), made by the caller. Encoding a payload of a few MB is
    a second of pure-Python work; done where the payload was computed, the
    server process passes bytes on."""
    if isinstance(value, Response):
        return value
    return JSONResponse(jsonable_encoder(value))


def _rendered(fn: Callable[..., Any], args: Tuple[Any, ...],
              kwargs: Dict[str, Any]) -> Response:
    return render_json(fn(*args, **kwargs))


async def offload_response(fn: Callable[..., Any], /, *args: Any,
                           **kwargs: Any) -> Response:
    """`offload` for a route that returns the call's result as it is: the
    JSON response is rendered in the worker too (`render_json`)."""
    return await within_deadline(
        "response:" + call_key(fn, args, kwargs),
        lambda: run_off_loop(_rendered, fn, args, kwargs))
