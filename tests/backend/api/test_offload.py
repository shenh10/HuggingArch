"""Heavy request work runs off the event loop (`backend.api.offload`).

The incident this pins: with cold caches, heavy analysis on the server's own
threads held the GIL and `/health` stopped answering. A heavy handler now runs
in a compute-pool worker — another process — and is answered within a
deadline: past it the client gets a 503 it retries, and the retry joins the
work already running.
"""
from __future__ import annotations

import asyncio
import os
import pickle
import threading
import time

import pytest
from fastapi.testclient import TestClient

import backend.api  # noqa: F401  # registers routes
from backend.api import offload
from backend.api._app import app
from backend.arch.layout.attention_plan import IndivisibleHeadRepartition
from backend.measure import pool as _sweep_pool

_CALLS: list = []


def burn(seconds: float) -> int:
    """Pure-Python CPU for `seconds`, the shape of an analysis: holds the GIL
    of whatever process runs it. Returns that process's pid."""
    end = time.monotonic() + seconds
    n = 0
    while time.monotonic() < end:
        n += 1
    return os.getpid()


def sleep_and_count(seconds: float) -> str:
    _CALLS.append(seconds)
    time.sleep(seconds)
    return "done"


def sleep_then_payload(seconds: float) -> dict:
    time.sleep(seconds)
    return {"rows": [f"row {i}" for i in range(500)]}


def refuse_head_repartition() -> None:
    raise IndivisibleHeadRepartition("attn", "q", 5, 2, (1, 5))


def steps_then_wait(steps: int, pause: float, tail: float) -> str:
    """One job of several steps — the shape of a (context × batch) sweep's
    cell, which is the whole sweep because one walk serves every point.
    Reports each step from whatever process runs it."""
    for done in range(1, steps + 1):
        time.sleep(pause)
        offload.report_progress(done, steps)
    time.sleep(tail)
    return "done"


@pytest.fixture
def slow_routes():
    """Two throwaway heavy routes on the real app, removed afterwards."""
    @app.get("/__test/burn")
    async def _burn(seconds: float):
        return {"pid": await offload.offload(burn, seconds)}

    @app.get("/__test/payload")
    async def _payload(seconds: float):
        return await offload.offload_response(sleep_then_payload, seconds)

    @app.get("/__test/sleep")
    async def _sleep(seconds: float):
        return {"out": await offload.offload(sleep_and_count, seconds)}

    added = [r for r in app.router.routes
             if getattr(r, "path", "").startswith("/__test/")]
    yield
    for route in added:
        app.router.routes.remove(route)


@pytest.fixture
def process_pool(monkeypatch):
    """The server's mode — heavy work in the compute pool — under pytest,
    which otherwise keeps the app in-process: two workers that import
    nothing extra, torn down after."""
    _sweep_pool.shutdown()
    monkeypatch.setattr(_sweep_pool, "_pool_allowed", lambda: True)
    monkeypatch.setattr(_sweep_pool, "_max_workers", lambda: 2)
    monkeypatch.setattr(_sweep_pool, "_create_ok", True)
    monkeypatch.setattr(offload, "_off_process", True)
    monkeypatch.setattr(offload, "_deadline_s", None)
    yield
    _sweep_pool.shutdown()


def test_slow_handler_does_not_block_health(slow_routes, process_pool):
    """While a handler burns CPU in the pool, `/health` answers at once, and
    the burn ran in another process. One client, one event loop: the
    health checks wait on the same loop the slow request is on."""
    with TestClient(app) as client:
        # Pay the worker spawn before timing anything.
        assert client.get("/__test/burn", params={"seconds": 0.01}).status_code == 200
        out: dict = {}

        def slow():
            out["resp"] = client.get("/__test/burn", params={"seconds": 3.0})

        t = threading.Thread(target=slow)
        t.start()
        time.sleep(0.3)
        health = []
        while t.is_alive():
            t0 = time.monotonic()
            assert client.get("/health").status_code == 200
            health.append(time.monotonic() - t0)
            time.sleep(0.1)
        t.join()
    assert len(health) >= 10, health
    assert max(health) < 0.5, health
    resp = out["resp"]
    assert resp.status_code == 200
    assert resp.json()["pid"] != os.getpid()


def test_past_deadline_is_a_retryable_503_that_joins_the_work(slow_routes, monkeypatch):
    """Past the deadline the request gets a 503 naming `still_computing`
    with a Retry-After; the work goes on, and the retry is answered from
    it — the function ran once."""
    _CALLS.clear()
    monkeypatch.setattr(offload, "_deadline_s", 0.3)
    with TestClient(app) as client:
        first = client.get("/__test/sleep", params={"seconds": 1.0})
        assert first.status_code == 503
        assert first.json()["reason"] == "still_computing"
        assert int(first.headers["Retry-After"]) > 0
        time.sleep(1.2)
        again = client.get("/__test/sleep", params={"seconds": 1.0})
        assert again.status_code == 200
        assert again.json() == {"out": "done"}
    assert _CALLS == [1.0]


def test_a_held_response_is_whole_for_every_caller(slow_routes, monkeypatch):
    """A rendered response held for the retries is sent once per caller,
    each through the gzip middleware: every caller gets the whole body.
    (Sharing one `Response` object sent the second caller the first one's
    compressed Content-Length over the raw body — a truncated response.)"""
    monkeypatch.setattr(offload, "_deadline_s", 0.3)
    want = sleep_then_payload(0)
    with TestClient(app) as client:
        first = client.get("/__test/payload", params={"seconds": 1.0})
        assert first.status_code == 503
        time.sleep(1.2)
        for _ in range(3):
            again = client.get("/__test/payload", params={"seconds": 1.0},
                               headers={"Accept-Encoding": "gzip"})
            assert again.status_code == 200
            assert again.headers.get("content-encoding") == "gzip"
            assert again.json() == want


def test_job_exception_crosses_the_process_boundary_as_itself():
    """An exception whose `__init__` cannot be called with its `args`
    (the route maps `IndivisibleHeadRepartition` to its own 400) arrives
    in the server as its own class, attributes and message."""
    with pytest.raises(Exception) as info:
        _sweep_pool._run_job(0, refuse_head_repartition, (), {})
    rebuilt = pickle.loads(pickle.dumps(info.value))
    assert type(rebuilt) is IndivisibleHeadRepartition
    assert rebuilt.legal_cp_degrees == (1, 5)
    assert rebuilt.degree == 2
    assert "legal CP degrees: 1, 5" in str(rebuilt)


@pytest.fixture
def dropped(monkeypatch):
    """The worker-side loader drops `_run_job` makes, counted."""
    out: list = []
    monkeypatch.setattr(_sweep_pool, "invalidate_default_loader",
                        lambda: out.append(True))
    monkeypatch.setattr(_sweep_pool, "_seen_loader_generation", None)
    return out


def test_worker_drops_its_spec_loader_when_the_servers_moves(dropped):
    """A spec refined in the server (which drops its default loader) is the
    spec the worker reads next: the job carries the server's loader
    generation and the worker drops its own loader when it moves."""
    _sweep_pool._seen_loader_generation = 3
    _sweep_pool._run_job(3, int, (), {})
    assert dropped == []
    _sweep_pool._run_job(4, int, (), {})
    assert dropped == [True]


def test_a_worker_with_no_known_generation_drops_its_loader(dropped):
    """A worker that cannot say which generation its loader answers for
    drops it at its first job, whatever that job carries."""
    _sweep_pool._run_job(1, int, (), {})
    assert dropped == [True]


def test_a_prestarted_worker_drops_the_loader_it_parsed_on_a_refine(
        dropped, monkeypatch):
    """A worker parses the specs when it starts, for the generation the pool
    was built at. A refine before its first job moves the server's
    generation, and that first job drops the parsed loader."""
    monkeypatch.setattr(_sweep_pool, "_in_worker", False)
    monkeypatch.setattr(_sweep_pool, "_die_with_parent", lambda: None)
    _sweep_pool._init_worker((), (), 7)
    _sweep_pool._run_job(7, int, (), {})
    assert dropped == []
    _sweep_pool._run_job(8, int, (), {})
    assert dropped == [True]


def test_a_request_after_a_refine_does_not_join_work_begun_before(monkeypatch):
    """The request identity carries the spec generation: the same request,
    sent after a refine, starts new work instead of joining (or being
    served) the old spec's."""
    import asyncio
    _CALLS.clear()
    generation = [0]
    monkeypatch.setattr(offload, "default_loader_generation",
                        lambda: generation[0])

    async def scenario():
        offload.set_deadline(0.15)
        with pytest.raises(offload.StillComputing):
            await offload.offload(sleep_and_count, 0.4)
        generation[0] += 1
        offload.set_deadline(None)
        return await offload.offload(sleep_and_count, 0.4)

    monkeypatch.setattr(offload, "_deadline_s", None)
    assert asyncio.run(scenario()) == "done"
    assert _CALLS == [0.4, 0.4]


def test_a_request_spanning_a_refine_is_not_served(monkeypatch):
    """A request of several jobs pins the generation it started at; a
    refine before it finishes is a retryable `still_computing`, not a
    result mixing two versions of a spec."""
    generation = [5]
    monkeypatch.setattr(offload, "default_loader_generation",
                        lambda: generation[0])
    pinned = offload.spec_generation()
    offload.ensure_spec_generation(pinned)
    generation[0] += 1
    with pytest.raises(offload.SpecChanged) as info:
        offload.ensure_spec_generation(pinned)
    assert info.value.reason == "still_computing"


def test_server_without_a_pool_answers_503_not_a_thread(slow_routes, monkeypatch):
    """In the server there is no thread to fall back to: with the pool
    switched off a heavy request is a 503 naming the reason, and the work
    never runs on the server's threads."""
    _CALLS.clear()
    monkeypatch.setattr(offload, "_off_process", True)
    monkeypatch.setattr(_sweep_pool, "jobs_enabled", lambda: False)
    resp = TestClient(app).get("/__test/sleep", params={"seconds": 0.01})
    assert resp.status_code == 503
    assert resp.json()["reason"] == "compute_unavailable"
    assert _CALLS == []


def test_server_whose_pool_cannot_be_built_answers_503(slow_routes, monkeypatch):
    _CALLS.clear()
    monkeypatch.setattr(offload, "_off_process", True)
    monkeypatch.setattr(_sweep_pool, "jobs_enabled", lambda: True)
    monkeypatch.setattr(_sweep_pool, "_get_pool", lambda *a, **k: None)
    resp = TestClient(app).get("/__test/sleep", params={"seconds": 0.01})
    assert resp.status_code == 503
    assert resp.json()["reason"] == "compute_unavailable"
    assert _CALLS == []


def test_a_503_says_how_far_the_work_got(monkeypatch):
    """Work of several jobs reports its progress (`report_progress`); a
    request that misses the deadline gets it in the 503 body, so a waiting
    page can say "12 / 60" rather than only "still computing". Work that
    reports nothing sends no `progress` field."""
    import asyncio

    async def three_jobs():
        for done in range(3):
            offload.report_progress(done, 3)
            if done == 1:
                await asyncio.sleep(1.0)
        return {"out": "done"}

    @app.get("/__test/progress")
    async def _progress():
        return await offload.within_deadline("test-progress", three_jobs)

    @app.get("/__test/silent")
    async def _silent():
        return {"out": await offload.offload(sleep_and_count, 1.0)}

    monkeypatch.setattr(offload, "_deadline_s", 0.2)
    try:
        with TestClient(app) as client:
            resp = client.get("/__test/progress")
            assert resp.status_code == 503
            body = resp.json()
            assert body["reason"] == "still_computing"
            assert body["progress"] == {"done": 1, "total": 3}
            silent = client.get("/__test/silent")
            assert silent.status_code == 503
            assert "progress" not in silent.json()
    finally:
        app.router.routes[:] = [r for r in app.router.routes
                                if getattr(r, "path", "") not in
                                ("/__test/progress", "/__test/silent")]


def test_report_progress_outside_a_request_is_a_no_op():
    offload.report_progress(1, 2)


def test_one_jobs_steps_reach_the_work_from_the_worker(process_pool, monkeypatch):
    """Work whose whole schedule is ONE compute job still reports movement.

    A (context × batch) sweep is one walk and therefore one cell, so the
    cells the server schedules go 0 of 1 to 1 of 1 and nothing moves for
    the sweep's whole run — half an hour on a cold large model, past the
    page's inactivity budget, which then gives up on work that is only
    slow. The steps inside the job are the movement: `report_progress`
    called in the worker travels on the pool's progress channel
    (`pool.send_progress`) and lands on the work this request waits on."""
    @app.get("/__test/worker-steps")
    async def _steps():
        return await offload.within_deadline(
            "test-worker-steps",
            lambda: offload.run_off_loop(steps_then_wait, 6, 0.3, 0.6))

    monkeypatch.setattr(offload, "_deadline_s", 0.2)
    try:
        with TestClient(app) as client:
            # Start the workers (the pool spawns lazily) so what follows is
            # the job's timing, not the spawn's.
            warm = [_sweep_pool.submit(sleep_and_count, 0.01) for _ in range(2)]
            assert [job.result() for job in warm] == ["done", "done"]
            seen = []
            for _ in range(8):
                resp = client.get("/__test/worker-steps")
                if resp.status_code == 200:
                    break
                assert resp.json()["reason"] == "still_computing"
                seen.append(resp.json().get("progress"))
                time.sleep(0.5)
            else:
                raise AssertionError(f"never answered; progress {seen}")
            assert resp.json() == "done"
    finally:
        app.router.routes[:] = [r for r in app.router.routes
                                if getattr(r, "path", "") != "/__test/worker-steps"]
    counts = [p["done"] for p in seen if p]
    # The one job's count moved while it ran — the whole point — and it
    # only ever went forward, out of six steps.
    assert counts, f"the job's steps never reached the work: {seen}"
    assert counts == sorted(counts) and counts[-1] > counts[0], counts
    assert all(p["total"] == 6 for p in seen if p), seen


def test_each_works_steps_reach_that_work_only(process_pool, monkeypatch):
    """Two requests computing at once each read their own work's steps: a
    tick carries the tag of the work whose job sent it, so one page's
    progress is never another's (the totals here, 6 and 3, tell them
    apart)."""
    import asyncio

    async def scenario():
        async def work(key, steps):
            return await offload.within_deadline(
                key, lambda: offload.run_off_loop(
                    steps_then_wait, steps, 0.2, 0.2))

        tasks = {key: asyncio.ensure_future(work(key, steps))
                 for key, steps in (("steps-six", 6), ("steps-three", 3))}
        # Every total each work was ever seen reporting, while they run.
        totals: dict = {}
        while not all(task.done() for task in tasks.values()):
            await asyncio.sleep(0.02)
            offload._fold_progress()
            for key, running in offload._table().items():
                if running.progress is not None:
                    totals.setdefault(key, set()).add(running.progress[1])
        await asyncio.gather(*tasks.values())
        return totals

    monkeypatch.setattr(offload, "_deadline_s", None)
    # Each work only ever reported its own job's step count.
    assert asyncio.run(scenario()) == {"steps-six": {6}, "steps-three": {3}}


def test_a_maps_cells_report_their_progress(process_pool, monkeypatch):
    """A `map_off_loop(..., progress=True)` work counts its finished cells:
    the 503 of a request that misses the deadline says how many of the
    sweep's cells are done, growing as cells land, and the retry that lands
    after the last one gets the results. Runs the server branch (cells as
    pool jobs; the count is a closure the event loop increments as each
    cell's job resolves)."""
    @app.get("/__test/cells")
    async def _cells():
        return await offload.within_deadline(
            "test-cell-progress",
            lambda: offload.map_off_loop(sleep_and_count, [1.0] * 3,
                                         progress=True))

    monkeypatch.setattr(offload, "_deadline_s", 0.3)
    try:
        with TestClient(app) as client:
            # Start both workers (the pool spawns lazily), so the timing
            # below is the cells', not the spawn's.
            warm = [_sweep_pool.submit(sleep_and_count, 0.01) for _ in range(2)]
            assert [job.result() for job in warm] == ["done", "done"]
            first = client.get("/__test/cells")
            assert first.status_code == 503
            assert first.json()["progress"] == {"done": 0, "total": 3}
            # One cell (~1 s, cells run one at a time — one slot for two
            # workers) is done when the second request joins the work.
            time.sleep(1.4)
            second = client.get("/__test/cells")
            assert second.status_code == 503
            progress = second.json()["progress"]
            assert progress["total"] == 3 and progress["done"] >= 1
            time.sleep(2.4)
            third = client.get("/__test/cells")
            assert third.status_code == 200
            assert third.json() == ["done"] * 3
    finally:
        app.router.routes[:] = [r for r in app.router.routes
                                if getattr(r, "path", "") != "/__test/cells"]


def test_a_long_tail_cell_does_not_stop_the_next_cell(monkeypatch):
    """A plan submits every cell before collecting any result.

    This is the cell level `ParallelSweepPlan` reaches through
    `map_off_loop`: one long-tail cell must not stop later cells from taking
    freed slots.  (The warm's former POST-level loop had that defect; the
    plan's cell map must not grow one.)
    """
    async def scenario():
        second_started = asyncio.Event()
        slots = offload.PrioritySlots(2)

        async def fake_run_off_loop(_fn, item):
            if item != "slow":
                second_started.set()
                return item
            try:
                await asyncio.wait_for(second_started.wait(), timeout=0.5)
            except TimeoutError:
                raise AssertionError(
                    "the next cell waited for the long tail") from None
            return item

        monkeypatch.setattr(offload, "_off_process", True)
        monkeypatch.setattr(offload, "run_off_loop", fake_run_off_loop)
        monkeypatch.setattr(offload, "_cell_slots", lambda: slots)
        return await offload.map_off_loop(
            lambda item: item, ["slow", "next", "last"])

    assert asyncio.run(scenario()) == ["slow", "next", "last"]


def test_a_maps_cells_report_nothing_by_default(monkeypatch):
    """`progress` defaults off: a `map_off_loop` work's 503 carries no
    `progress` field, and the finished work still answers the retry. This
    runs the non-server branch (a serial map on a thread — under pytest the
    pool stays off), which reports (0, n) / (n, n) only when asked; the
    per-cell counting of the server branch is the test above."""
    @app.get("/__test/cells-silent")
    async def _cells_silent():
        return await offload.within_deadline(
            "test-cells-silent",
            lambda: offload.map_off_loop(sleep_and_count, [1.0, 1.0]))

    monkeypatch.setattr(offload, "_deadline_s", 0.2)
    try:
        with TestClient(app) as client:
            resp = client.get("/__test/cells-silent")
            assert resp.status_code == 503
            assert "progress" not in resp.json()
            time.sleep(2.3)
            again = client.get("/__test/cells-silent")
            assert again.status_code == 200
            assert again.json() == ["done", "done"]
    finally:
        app.router.routes[:] = [r for r in app.router.routes
                                if getattr(r, "path", "") != "/__test/cells-silent"]


def test_a_single_job_does_not_wait_behind_a_sweeps_cells(process_pool):
    """A sweep's cells hold every worker but one (`offload._cell_slots`), so
    a request that is one job runs beside them instead of behind them. Two
    workers, as production runs: six 0.8 s cells all queued ahead would make
    the single job wait ~2.4 s; with one worker left to it, it does not."""
    import asyncio

    async def scenario():
        sweep = asyncio.ensure_future(
            offload.map_off_loop(sleep_and_count, [0.8] * 6))
        await asyncio.sleep(0.3)        # the cells are in the pool first
        start = time.monotonic()
        await offload.run_off_loop(sleep_and_count, 0.0)
        waited = time.monotonic() - start
        assert await sweep == ["done"] * 6
        return waited

    # Start both workers (two jobs at once: the pool spawns lazily), so the
    # measurement is the queue, not a spawn.
    warm = [_sweep_pool.submit(sleep_and_count, 0.5) for _ in range(2)]
    assert [job.result() for job in warm] == ["done", "done"]
    waited = asyncio.run(scenario())
    assert waited < 1.0, waited


def test_a_retry_while_the_cancel_unwinds_starts_over(monkeypatch):
    """The unwatched timer cancels the work, but cancelling only asks: the
    task is not done until it next runs, and work unwinding (a `finally`,
    an awaited cleanup) stays undone longer. A retry of the same request in
    that window must start new work — not join the dying task and be
    answered with its `CancelledError`."""
    import asyncio
    starts: list = []

    async def slow():
        starts.append(1)
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)        # the unwinding window
            raise
        return "done"

    async def quick():
        starts.append(2)
        return "fresh"

    async def scenario():
        with pytest.raises(offload.StillComputing):
            await offload.within_deadline("test-unwinding", slow)
        await asyncio.sleep(0.15)           # the timer fired; still unwinding
        monkeypatch.setattr(offload, "_deadline_s", None)
        return await offload.within_deadline("test-unwinding", quick)

    monkeypatch.setattr(offload, "_deadline_s", 0.05)
    monkeypatch.setattr(offload, "UNWATCHED_S", 0.05)
    assert asyncio.run(scenario()) == "fresh"
    assert starts == [1, 2]


def test_work_no_request_waits_on_is_cancelled(monkeypatch):
    """A client that stops asking — the gallery user who clicked another
    GPU — leaves no work behind: once no request has waited on it for
    `UNWATCHED_S`, the work is cancelled, so it stops taking compute from
    the work its client waits on now. The same request sent later starts
    over (reading whatever the cancelled work had kept in the caches)."""
    import asyncio
    starts: list = []
    cancelled: list = []

    async def slow():
        starts.append(1)
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise
        return "done"

    async def scenario():
        with pytest.raises(offload.StillComputing):
            await offload.within_deadline("test-unwatched", slow)
        await asyncio.sleep(0.5)
        assert cancelled == [1]
        assert "test-unwatched" not in offload._table()
        with pytest.raises(offload.StillComputing):
            await offload.within_deadline("test-unwatched", slow)
        assert len(starts) == 2

    monkeypatch.setattr(offload, "_deadline_s", 0.1)
    monkeypatch.setattr(offload, "UNWATCHED_S", 0.2)
    asyncio.run(scenario())


def test_work_a_client_keeps_asking_for_runs_to_its_end(monkeypatch):
    """The join-on-retry property survives the cancel: a client whose
    retries come back within `UNWATCHED_S` keeps the work alive past many
    deadlines, joins it each time, and is answered by the one run."""
    import asyncio
    starts: list = []

    async def slow():
        starts.append(1)
        await asyncio.sleep(1.0)
        return "done"

    async def scenario():
        for _ in range(50):
            try:
                return await offload.within_deadline("test-watched", slow)
            except offload.StillComputing:
                await asyncio.sleep(0.1)    # the client's retry interval
        raise AssertionError("never answered")

    monkeypatch.setattr(offload, "_deadline_s", 0.1)
    monkeypatch.setattr(offload, "UNWATCHED_S", 0.3)
    assert asyncio.run(scenario()) == "done"
    assert starts == [1]


def test_a_503_counts_the_jobs_the_work_finished(monkeypatch):
    """Between two steps of its reported progress — one model of a cold
    gallery frontier is dozens of layout jobs — work still moves: the 503
    carries how many compute jobs it has finished (`jobs_done`), so a
    client can tell work that moves from work that is stuck."""
    import asyncio

    async def jobs():
        offload.report_progress(0, 1)
        for _ in range(3):
            await offload.run_off_loop(sleep_and_count, 0.05)
        await asyncio.sleep(1.0)
        return "done"

    async def scenario():
        with pytest.raises(offload.StillComputing) as info:
            await offload.within_deadline("test-jobs-done", jobs)
        return info.value

    monkeypatch.setattr(offload, "_deadline_s", 0.5)
    got = asyncio.run(scenario())
    assert got.progress == (0, 1)
    assert got.jobs_done == 3

    @app.get("/__test/jobs-done")
    async def _jobs_done():
        return await offload.within_deadline("test-jobs-done-route", jobs)

    try:
        with TestClient(app) as client:
            body = client.get("/__test/jobs-done").json()
        assert body["reason"] == "still_computing"
        assert body["jobs_done"] == 3
        assert body["progress"] == {"done": 0, "total": 1}
    finally:
        app.router.routes[:] = [r for r in app.router.routes
                                if getattr(r, "path", "") != "/__test/jobs-done"]


def test_a_freed_slot_goes_to_the_most_urgent_waiter():
    """`PrioritySlots` grants a freed slot by urgency, read when the slot
    frees — a waiter whose urgency changed while it queued is reordered —
    and ties keep arrival order. A cancelled waiter gives up its place, and
    one cancelled just after it was granted passes the slot on."""
    import asyncio

    async def scenario():
        slots = offload.PrioritySlots(1)
        urgency = {"a": 3, "b": 2, "c": 1, "d": 1}
        order: list = []
        release = asyncio.Event()

        async def blocker():
            async with slots.hold(lambda: 0):
                await release.wait()

        async def waiter(name):
            async with slots.hold(lambda: urgency[name]):
                order.append(name)

        held = asyncio.ensure_future(blocker())
        await asyncio.sleep(0)
        tasks = {n: asyncio.ensure_future(waiter(n)) for n in "abcd"}
        await asyncio.sleep(0)
        urgency["a"] = 0                  # re-ranked while it queued
        tasks["c"].cancel()               # gives up its place
        release.set()
        await held
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        return order

    assert asyncio.run(scenario()) == ["a", "d", "b"]


def test_the_view_asked_for_last_takes_the_next_slot(monkeypatch):
    """Two works wait on one slot: the one a client asked for more recently
    goes first — the GPU a gallery client switched to before the one it left
    — and asking again makes a work the most recent again."""
    import asyncio

    async def scenario():
        slots = offload.PrioritySlots(1)
        order: list = []
        release = asyncio.Event()

        async def blocker():
            async with slots.hold(lambda: 0):
                await release.wait()

        def work(name):
            async def run():
                async with slots.hold(offload.urgency()):
                    order.append(name)
                return name
            return run

        async def ask(key, name):
            with pytest.raises(offload.StillComputing):
                await offload.within_deadline(key, work(name))

        held = asyncio.ensure_future(blocker())
        await asyncio.sleep(0)
        for key, name in asks:
            await ask(f"{key}-{len(asks)}", name)
        release.set()
        await held
        for _ in range(5):
            await asyncio.sleep(0)
        return order

    monkeypatch.setattr(offload, "_deadline_s", 0.01)
    # The client moved from view a to view b: b first (FIFO would say a).
    asks = [("view-a", "a"), ("view-b", "b")]
    assert asyncio.run(scenario()) == ["b", "a"]
    # ... and came back to a: a is the most recent again.
    asks = [("view-a", "a"), ("view-b", "b"), ("view-a", "a")]
    assert asyncio.run(scenario()) == ["a", "b"]


def test_the_clients_order_ranks_the_items_of_one_work(monkeypatch):
    """Within one work, the items the client named first (`hint`: the model
    it picked, then those on screen) take the slots first; an item it did
    not name comes after every named one."""
    import asyncio

    async def scenario():
        slots = offload.PrioritySlots(1)
        order: list = []
        release = asyncio.Event()

        async def blocker():
            async with slots.hold(lambda: 0):
                await release.wait()

        async def run():
            async def item(name):
                async with slots.hold(offload.urgency(name)):
                    order.append(name)
            await asyncio.gather(*(item(n) for n in ("m1", "m2", "m3")))
            return order

        held = asyncio.ensure_future(blocker())
        await asyncio.sleep(0)
        with pytest.raises(offload.StillComputing):
            await offload.within_deadline("ranked", run, hint=["m3", "m2"])
        release.set()
        await held
        offload.set_deadline(None)
        return await offload.within_deadline("ranked", run)

    monkeypatch.setattr(offload, "_deadline_s", 0.01)
    try:
        assert asyncio.run(scenario()) == ["m3", "m2", "m1"]
    finally:
        offload.set_deadline(None)


def test_a_503_carries_the_published_items_the_client_lacks(monkeypatch):
    """Work that publishes items as they land (`report_partial`) puts in
    each 503 its head and the items past the client's cursor — all of them
    for a cursor of another work — so a page draws each item once."""
    import asyncio

    async def publishing():
        offload.report_partial(head={"seq_lens": [1024]})
        offload.report_partial({"model_id": "m1"})
        offload.report_partial({"model_id": "m2"})
        await asyncio.sleep(1.0)
        return {"done": True}

    async def scenario():
        got = []
        for cursor in (None, "same", (0, 1)):
            with pytest.raises(offload.StillComputing) as info:
                await offload.within_deadline(
                    "partial", publishing,
                    cursor=(None if cursor is None else
                            (got[0]["tag"], 1) if cursor == "same" else cursor))
            got.append(info.value.partial)
        return got

    monkeypatch.setattr(offload, "_deadline_s", 0.05)
    first, mine, foreign = asyncio.run(scenario())
    assert first["head"] == {"seq_lens": [1024]}
    assert first["count"] == 2
    assert [i["model_id"] for i in first["items"]] == ["m1", "m2"]
    assert mine["tag"] == first["tag"]
    assert [i["model_id"] for i in mine["items"]] == ["m2"]
    assert [i["model_id"] for i in foreign["items"]] == ["m1", "m2"]

    @app.get("/__test/partial")
    async def _partial():
        return await offload.within_deadline("partial-route", publishing)

    try:
        with TestClient(app) as client:
            body = client.get("/__test/partial").json()
        assert body["reason"] == "still_computing"
        assert body["partial"]["head"] == {"seq_lens": [1024]}
        assert [i["model_id"] for i in body["partial"]["items"]] == ["m1", "m2"]
    finally:
        app.router.routes[:] = [r for r in app.router.routes
                                if getattr(r, "path", "") != "/__test/partial"]
    offload.report_partial({"model_id": "nobody"})   # outside a request: no-op


def test_work_that_publishes_answers_a_waiting_client_without_waiting_for_the_deadline(
        monkeypatch):
    """A long poll: a request waiting on work that publishes (`report_partial`)
    answers as soon as something new lands — after a short grace — and asks
    to be sent again in a second; a client already behind the work's items
    is answered at once; with nothing new it waits for the deadline."""
    import asyncio

    async def publishes_late():
        await asyncio.sleep(0.1)
        offload.report_partial({"model_id": "m1"})
        await asyncio.sleep(10)
        return "done"

    async def silent():
        await asyncio.sleep(10)
        return "done"

    async def timed(key, start, **kw):
        t0 = time.monotonic()
        with pytest.raises(offload.StillComputing) as info:
            await offload.within_deadline(key, start, **kw)
        return time.monotonic() - t0, info.value

    async def scenario():
        early, first = await timed("news", publishes_late)
        behind, again = await timed("news", publishes_late,
                                    cursor=(first.partial["tag"], 0))
        quiet, nothing = await timed("quiet", silent)
        return early, first, behind, again, quiet, nothing

    monkeypatch.setattr(offload, "_deadline_s", 3.0)
    early, first, behind, again, quiet, nothing = asyncio.run(scenario())
    assert early < 1.5, early
    assert first.retry_after_s == offload.NEWS_RETRY_AFTER_S
    assert [i["model_id"] for i in first.partial["items"]] == ["m1"]
    assert behind < 1.0, behind
    assert [i["model_id"] for i in again.partial["items"]] == ["m1"]
    assert quiet >= 2.9 and nothing.retry_after_s is None


def test_an_offline_rank_orders_waiters_outside_a_request():
    """Outside a request the rank a task set (`offline_rank`) orders the
    slot's waiters, in the task and every task it starts: a batch ranked
    later takes a slot only when no earlier one waits — even when its
    waiters queued first."""
    import asyncio

    async def scenario():
        slots = offload.PrioritySlots(1)
        order: list = []
        release = asyncio.Event()

        async def blocker():
            async with slots.hold(lambda: 0):
                await release.wait()

        async def batch(rank, name):
            offload.offline_rank(rank)

            async def job(i):
                async with slots.hold(offload.urgency()):
                    order.append(f"{name}{i}")
            await asyncio.gather(*(asyncio.create_task(job(i))
                                   for i in range(2)))

        held = asyncio.ensure_future(blocker())
        await asyncio.sleep(0)
        late = asyncio.create_task(batch(1, "late"))
        await asyncio.sleep(0)
        early = asyncio.create_task(batch(0, "early"))
        for _ in range(3):
            await asyncio.sleep(0)
        release.set()
        await asyncio.gather(held, late, early)
        return order

    assert asyncio.run(scenario()) == ["early0", "early1", "late0", "late1"]


def test_a_process_that_answers_no_request_keeps_no_worker(monkeypatch):
    """The cells' slots are the pool's workers less those the process keeps
    for single jobs: one for the server, none for an offline warm."""
    import asyncio

    monkeypatch.setattr(offload.pool, "worker_count", lambda: 12)
    monkeypatch.setattr(offload.pool, "configure_workers", lambda **_kw: None)

    def slots() -> int:
        async def read():
            return offload._cell_slots()._free
        return asyncio.run(read())

    try:
        offload.serve_requests_off_process(deadline_s=None)
        assert slots() == 11
        offload.run_off_process()
        assert slots() == 12
        assert offload._deadline_s is None
    finally:
        offload.stop_serving()
    assert offload._answers_requests
