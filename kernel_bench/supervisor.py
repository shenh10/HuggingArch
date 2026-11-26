"""Wall-clock and progress supervision for provider environment processes."""
from __future__ import annotations

import os
import json
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


def parse_duration(value: str) -> float:
    """Parse seconds or a value suffixed with s, m, or h.

    The suffixes and what each is worth in seconds are stated once, and the
    pattern that accepts them is built from that table — so a suffix this
    matches is always one it can convert, and neither half can grow a unit
    the other does not know.
    """
    scale = {"s": 1.0, "m": 60.0, "h": 3600.0, "": 1.0}
    pattern = r"^\s*(\d+(?:\.\d+)?)\s*(%s)\s*$" % "|".join(scale)
    match = re.match(pattern, value, re.IGNORECASE)
    if not match:
        raise ValueError(f"invalid duration {value!r}; use e.g. 30s, 15m, or 2h")
    seconds = float(match.group(1)) * scale[match.group(2).lower()]
    if seconds < 0:
        raise ValueError(f"duration must be non-negative: {value!r}")
    return seconds


def write_progress(path: str, index: int, job: Mapping[str, object],
                   in_flight: Sequence[Mapping[str, str]] = (),
                   disabled: Optional[Mapping[str, str]] = None) -> None:
    """Atomically publish where a restarted run resumes (`index`), the job
    whose process may currently own the GPU (`job` — the job at `index`,
    except while a launch path's canary runs ahead of its own), and the
    implementation of it in flight: the row that records it if the
    process dies now (`driver.run_jobs`'s `on_attempt`), outcome left to the
    supervisor, which is the one that knows whether it was killed for
    making no progress or died by itself (`read_in_flight`). `disabled` is
    the launch paths this run's canaries switched off, with why
    (`read_disabled`)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(
        {"index": index, "job": job, "in_flight": list(in_flight),
         "disabled": dict(disabled or {})}) + "\n")
    temporary.replace(target)


def read_disabled(path: str) -> dict[str, str]:
    """The launch paths the run's canaries disabled, path → why; `{}` when
    none were (or there is no readable sidecar)."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    found = payload.get("disabled") if isinstance(payload, dict) else None
    if not isinstance(found, dict):
        return {}
    return {str(k): str(v) for k, v in found.items()}


def read_in_flight(path: str) -> list[dict]:
    """The rows naming the implementation in flight when the sidecar was
    last written; `[]` when none was (between providers, in a job's shared
    stage, a sidecar that predates the field, or none at all)."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("in_flight") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [dict(r) for r in rows if isinstance(r, dict)]


class NoProgress(ValueError):
    """There is no sidecar: the child never started a job.

    A different fact from a sidecar that cannot be READ, and the difference
    is the whole diagnosis. `write_progress` publishes atomically and the
    parent unlinks the file before each attempt, so an absent sidecar is not
    a lost race or a half-written file — it says the process exited before
    its first job, and the cause is therefore upstream of the jobs
    altogether. The live case: every env of every model was dying in the
    `--measured-in` read (one duplicated row in another model's results),
    which is before `run_jobs` is even called, while the log said "cannot
    isolate failed job" — a sentence that sent two readers to the sidecar
    instead of to the crash printed above it.
    """


def read_progress(path: str) -> tuple[int, dict]:
    if not Path(path).exists():
        raise NoProgress(
            f"no progress sidecar at {path}: no job had started")
    try:
        payload = json.loads(Path(path).read_text())
        index = payload["index"]
        job = payload["job"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid progress sidecar {path}: {exc}") from exc
    if not isinstance(index, int) or index < 0 or not isinstance(job, dict):
        raise ValueError(f"invalid progress sidecar {path}: {payload!r}")
    return index, job


@dataclass(frozen=True)
class SupervisedResult:
    returncode: int
    stopped_by: Optional[str]
    elapsed_s: float


def _mtime_ns(path: str) -> Optional[int]:
    try:
        return Path(path).stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _stop_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def supervise_process(
    process,
    *,
    output_path: str,
    budget_s: float = 0,
    stall_timeout_s: float = 0,
    poll_interval_s: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    mtime_ns: Callable[[str], Optional[int]] = _mtime_ns,
    stop_process: Callable[[object], None] = _stop_process_group,
) -> SupervisedResult:
    """Wait for a process while enforcing wall-clock and file-progress limits.
    Whatever stops the wait — a limit, or this process being stopped — stops
    the process too: it runs in its own session, so nothing else reaches it."""
    try:
        return _supervise(process, output_path=output_path, budget_s=budget_s,
                          stall_timeout_s=stall_timeout_s,
                          poll_interval_s=poll_interval_s, clock=clock,
                          sleep=sleep, mtime_ns=mtime_ns,
                          stop_process=stop_process)
    except BaseException:
        if process.poll() is None:
            stop_process(process)
        raise


def _supervise(process, *, output_path, budget_s, stall_timeout_s,
               poll_interval_s, clock, sleep, mtime_ns, stop_process
               ) -> SupervisedResult:
    started = clock()
    last_progress = started
    last_mtime = mtime_ns(output_path)

    while True:
        returncode = process.poll()
        now = clock()
        if returncode is not None:
            return SupervisedResult(returncode, None, now - started)

        current_mtime = mtime_ns(output_path)
        if current_mtime != last_mtime:
            last_mtime = current_mtime
            last_progress = now

        reason = None
        if budget_s > 0 and now - started >= budget_s:
            reason = f"wall-clock budget {budget_s:g}s exhausted"
        elif stall_timeout_s > 0 and now - last_progress >= stall_timeout_s:
            reason = (f"output file made no progress for "
                      f"{stall_timeout_s:g}s: {output_path}")
        if reason is not None:
            stop_process(process)
            returncode = process.poll()
            if returncode is None:
                returncode = process.wait()
            return SupervisedResult(returncode, reason, clock() - started)

        sleep(poll_interval_s)


def run_command(
    command: Sequence[str],
    *,
    output_path: str,
    env: Mapping[str, str],
    cwd: str,
    budget_s: float = 0,
    stall_timeout_s: float = 0,
) -> SupervisedResult:
    process = subprocess.Popen(
        list(command), env=dict(env), cwd=cwd, start_new_session=True)
    return supervise_process(
        process, output_path=output_path, budget_s=budget_s,
        stall_timeout_s=stall_timeout_s)
