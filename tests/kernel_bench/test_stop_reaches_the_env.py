"""A stop reaches the env that is measuring.

`queue` runs `run-all`, which runs each provider env in its own session
(`supervisor.run_command`, `start_new_session=True`). A SIGTERM to the queue
used to stop the queue alone: `_spawn` passed only Ctrl-C to its child, and
`run-all` exited on SIGTERM without unwinding, so the env kept measuring on
the card with nobody watching it — found 2026-11-27 when a stop left two
envs timing kernels after their queues were gone.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class _Process:
    def __init__(self):
        self.returncode = None
        self.stopped = False

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


def test_a_stopped_supervisor_stops_the_env_it_watches():
    from kernel_bench.supervisor import supervise_process
    process = _Process()

    def stop(target):
        target.stopped = True
        target.returncode = -15

    def sleep(_seconds):
        raise SystemExit(143)                 # run-all's SIGTERM, unwinding

    with pytest.raises(SystemExit):
        supervise_process(process, output_path="unused.csv", poll_interval_s=1,
                          clock=lambda: 0.0, sleep=sleep,
                          mtime_ns=lambda path: None, stop_process=stop)
    assert process.stopped


def test_a_stopped_queue_stops_its_run(monkeypatch):
    from kernel_bench import queue

    class Child:
        terminated = False

        def __init__(self, cmd):
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise queue.Stopped("signal 15")
            return -15

        def terminate(self):
            Child.terminated = True

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    with pytest.raises(queue.Stopped):
        queue._spawn(["run-all"])
    assert Child.terminated


def test_sigterm_to_the_supervisor_takes_its_session_down(tmp_path):
    # Real processes: the watched child is in its own session, as an env is.
    pidfile = tmp_path / "child.pid"
    script = textwrap.dedent(f"""
        import signal, sys
        sys.path.insert(0, {str(ROOT)!r})
        from kernel_bench.supervisor import supervise_process
        import subprocess
        def _stop(signum, _frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, _stop)
        child = subprocess.Popen(["sleep", "60"], start_new_session=True)
        open({str(pidfile)!r}, "w").write(str(child.pid))
        supervise_process(child, output_path={str(tmp_path / "out.csv")!r},
                          poll_interval_s=0.1)
    """)
    parent = subprocess.Popen([sys.executable, "-c", script])
    deadline = time.time() + 20
    while not pidfile.exists() or not pidfile.read_text():
        assert time.time() < deadline, "the supervised child never started"
        time.sleep(0.05)
    child = int(pidfile.read_text())
    parent.send_signal(signal.SIGTERM)
    parent.wait(timeout=20)
    deadline = time.time() + 10
    while True:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        assert time.time() < deadline, "the env outlived its supervisor"
        time.sleep(0.05)
