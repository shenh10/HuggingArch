"""The persistent process pool (`backend/measure/pool.py`): the pytest
gate, the failure a caller falls back on, a broken executor replaced, and
the worker count read from the cgroup quota and the environment. The sweeps
run across it are tested in `tests/backend/inference/test_sweep_pool.py`."""
from __future__ import annotations

import unittest
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from unittest import mock

from backend.measure import pool as _sweep_pool


class TestSweepPoolGate(unittest.TestCase):
    def test_pool_disabled_under_pytest(self):
        """`PYTEST_CURRENT_TEST` is set during pytest, so the pool gate is
        OFF — guarantees the preparation-amortization tests (which monkeypatch
        in-process and count == 1) keep exercising the serial path."""
        self.assertFalse(_sweep_pool.pool_enabled())


class TestPoolUnavailable(unittest.TestCase):
    def test_run_parallel_raises_without_pool(self):
        # Force no pool, confirm run_parallel signals failure (so the
        # caller's try/except can fall through).
        orig = _sweep_pool._get_pool
        _sweep_pool._get_pool = lambda *a, **k: None
        try:
            with self.assertRaises(RuntimeError):
                _sweep_pool.run_parallel(abs, [1])
        finally:
            _sweep_pool._get_pool = orig


class _BrokenExecutor:
    """An executor whose worker died: every submit raises, as the real one does."""

    def __init__(self):
        self.shut_down = False

    def submit(self, fn, *args, **kwargs):
        raise BrokenProcessPool("a worker was killed")

    def shutdown(self, wait=True, cancel_futures=False):
        self.shut_down = True


class _InlineExecutor:
    """A healthy executor that runs submits in-process."""

    def __init__(self, **_kw):
        pass

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        fut.set_result(fn(*args, **kwargs))
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


class TestBrokenPoolRecovery(unittest.TestCase):
    """A crashed worker breaks its executor for good. The pool must be
    replaced on the next call, not reused forever."""

    def test_broken_pool_is_replaced_on_next_run(self):
        broken = _BrokenExecutor()
        with mock.patch.object(_sweep_pool, "_pool", broken), \
                mock.patch.object(_sweep_pool, "_create_ok", True), \
                mock.patch.object(_sweep_pool, "_max_workers", lambda: 2), \
                mock.patch.object(_sweep_pool, "ProcessPoolExecutor", _InlineExecutor):
            with self.assertRaises(BrokenProcessPool):
                _sweep_pool.run_parallel(abs, [-1, -2])
            self.assertTrue(broken.shut_down)
            self.assertIsNone(_sweep_pool._pool)
            self.assertEqual(_sweep_pool.run_parallel(abs, [-1, -2]), [1, 2])
            self.assertIsInstance(_sweep_pool._pool, _InlineExecutor)


def test_a_late_worker_answer_does_not_overwrite_a_cancelled_bridge(monkeypatch):
    """A caller may cancel its submitted work while the worker is running."""
    class RunningFuture:
        callback = None

        def add_done_callback(self, callback):
            self.callback = callback

    class RunningExecutor:
        worker = None

        def submit(self, *_args, **_kwargs):
            self.worker = RunningFuture()
            return self.worker        # running: no answer yet

    class Provider:
        def snapshot(self):
            return None

        def merge(self, *_args, **_kwargs):
            pass

    executor = RunningExecutor()
    monkeypatch.setattr(_sweep_pool, "_get_pool", lambda *_a, **_k: executor)
    provider = Provider()
    monkeypatch.setattr(_sweep_pool, "provider", lambda *_a, **_k: provider)
    out = _sweep_pool.submit(lambda value: value, 42)
    assert out.cancel()

    done = Future()
    done.set_result((42, None))
    executor.worker.callback(done)    # formerly InvalidStateError
    assert out.cancelled()


def test_cgroup_cpu_quota_caps_the_pool(tmp_path, monkeypatch):
    """A container whose affinity mask lists every host core but whose
    cgroup grants a few cores' time sizes its pool by the quota: a pool of
    16 on a 12-core quota runs every job slower, not more of them."""
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("1200000 100000\n")
    assert _sweep_pool._cgroup_cpu_quota(str(cpu_max)) == 12
    cpu_max.write_text("150000 100000\n")
    assert _sweep_pool._cgroup_cpu_quota(str(cpu_max)) == 2  # rounded up
    cpu_max.write_text("max 100000\n")
    assert _sweep_pool._cgroup_cpu_quota(str(cpu_max)) is None
    assert _sweep_pool._cgroup_cpu_quota(str(tmp_path / "absent")) is None

    monkeypatch.setattr(_sweep_pool, "_cgroup_cpu_quota", lambda: 12)
    monkeypatch.setattr(_sweep_pool.os, "sched_getaffinity",
                        lambda _pid: set(range(112)), raising=False)
    monkeypatch.delenv("HA_SWEEP_POOL_WORKERS", raising=False)
    assert _sweep_pool._usable_cpus() == 12
    assert _sweep_pool._max_workers() == 10
