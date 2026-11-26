"""The measuring loop a box serves: one model at a time, oldest first.

A box with a plan for a model measures it, writes a `.done` beside the
results, and takes the next plan. That is the whole loop, and it ran for
three days as a shell script on one machine's local disk — a disk a node
reallocation wipes, taking the only copy of how this box measures with it.

Two things in it are not obvious and are the reason it is code:

CROSS-MODEL DEDUP
    Every run is given every OTHER model's results as `--measured-in`.
    The physical key carries no model name, so a cell another model
    already measured is this model's cell: skipping it is correct, and
    not skipping it is hours of GPU time spent re-measuring the same
    geometry under a different model's name. The operator's own
    `--measured-in` (an export of what the corpus already holds) is
    merged in the same way.

STOPPING
    The loop runs `run-all` as a child and waits. SIGTERM stops both,
    without the process-group kill the shell version needed — which took
    the loop down along with the run every time, so a restart was two
    commands and a guess about which of them had actually died.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence

#: `org/model` ⇄ `org__model`. The repo's own spelling, as
#: `backend/arch/snapshots/` writes it: reversible, because a single
#: underscore is not (model names contain one).
SLUG_SEPARATOR = "__"


def slug_of(model_id: str) -> str:
    return model_id.replace("/", SLUG_SEPARATOR, 1)


def model_of(slug: str) -> str:
    return slug.replace(SLUG_SEPARATOR, "/", 1)


@dataclass(frozen=True)
class Workspace:
    """Where a box keeps the queue. One place, derived from `work_dir`."""

    root: Path

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def log(self) -> Path:
        return self.root / "queue.log"

    @property
    def pidfile(self) -> Path:
        return self.root / "queue.pid"

    def plan_of(self, model_id: str) -> Path:
        return self.jobs / f"{slug_of(model_id)}.jsonl"

    def results_of(self, model_id: str) -> Path:
        return self.results / slug_of(model_id)

    def done_marker(self, model_id: str) -> Path:
        return self.results_of(model_id) / ".done"

    def prepare(self) -> "Workspace":
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)
        return self

    def say(self, line: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.log.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {line}\n")
        print(f"{stamp} {line}", flush=True)


def pending(space: Workspace) -> List[str]:
    """Models with a plan and no `.done`, oldest plan first.

    Oldest first because a plan written earlier has been waiting longer,
    and because `enqueue` hands plans over as they land: the order they
    were asked for is the order they run.
    """
    plans = sorted(space.jobs.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    out = []
    for plan in plans:
        model = model_of(plan.stem)
        if not space.done_marker(model).exists():
            out.append(model)
    return out


def measured_in_for(space: Workspace, model_id: str,
                    extra: Sequence[str] = ()) -> List[str]:
    """Every result CSV that counts as already measured for `model_id`.

    Its own directory is included — a run resumed after a crash must not
    re-measure what it already wrote — and so is every other model's,
    which is the cross-model dedup. `extra` is whatever the operator
    names (an export of the corpus), and it comes last so a reader can
    see where it entered.
    """
    mine = space.results_of(model_id)
    out: List[str] = [str(csv) for csv in sorted(mine.glob("*.csv"))]
    for other in sorted(space.results.iterdir()) if space.results.is_dir() else []:
        if not other.is_dir() or other == mine:
            continue
        out.extend(str(csv) for csv in sorted(other.glob("*.csv")))
    out.extend(str(path) for path in extra)
    return out


def run_all_command(space: Workspace, model_id: str, *,
                    box: Optional[str] = None,
                    budget: str = "6h",
                    stall_timeout: str = "15m",
                    contributor: Optional[str] = None,
                    measured_in: Sequence[str] = (),
                    python: Optional[str] = None,
                    envs_dir: Optional[str] = None) -> List[str]:
    """The `run-all` invocation for one model — built here so `--dry-run`
    can print exactly what would run. `envs_dir` is the one the plans were
    judged against: the run must start the same env files."""
    cmd = [sys.executable, "-m", "kernel_bench", "run-all",
           "--jobs", str(space.plan_of(model_id)),
           "-o", str(space.results_of(model_id)) + os.sep,
           "--budget", budget,
           "--stall-timeout", stall_timeout]
    if envs_dir:
        cmd += ["--envs-dir", str(envs_dir)]
    if box:
        cmd += ["--box", box]
    if python:
        cmd += ["--python", python]
    if contributor:
        cmd += ["--contributor", contributor]
    for path in measured_in_for(space, model_id, measured_in):
        cmd += ["--measured-in", path]
    return cmd


class Stopped(Exception):
    """SIGTERM reached the loop."""


def run_once(space: Workspace, model_id: str, *, dry_run: bool = False,
             spawn: Optional[Callable[[List[str]], int]] = None,
             **kwargs) -> int:
    """Measure one model and mark it done. Returns run-all's exit code."""
    space.results_of(model_id).mkdir(parents=True, exist_ok=True)
    cmd = run_all_command(space, model_id, **kwargs)
    jobs = sum(1 for _ in space.plan_of(model_id).open(encoding="utf-8"))
    if dry_run:
        print(f"{model_id} ({jobs} jobs)\n  " + " ".join(cmd))
        return 0
    space.say(f"start {model_id} ({jobs} jobs)")
    code = (spawn or _spawn)(cmd)
    space.say(f"end {model_id} rc={code}")
    # Done means the run REACHED THE END, not that every cell measured: a
    # run that finishes with providers refusing cells has measured
    # everything this box can, and leaving it unmarked would loop on it
    # forever. A run that could not start is a different thing — the queue
    # refused 61 models in a second once, because it handed `run-all` a box
    # name where a path belongs — so a non-zero exit does not mark it, and
    # the caller stands the model aside for this pass instead of spinning.
    # Nor does a run that reached the end with a launch path disabled by
    # its canary (`cli.EXIT_PATHS_DISABLED`): that path's cells were never
    # attempted, and a `.done` would leave them pending for good.
    if code == 0:
        space.done_marker(model_id).touch()
    return code


def _spawn(cmd: List[str]) -> int:
    """Run the child and wait, passing a stop on to it."""
    child = subprocess.Popen(cmd)
    try:
        return child.wait()
    except BaseException:
        # A stop of any kind — Ctrl-C, or the SIGTERM `serve` turns into
        # `Stopped` — goes to the child, which passes it to its env. Only
        # `KeyboardInterrupt` used to, so a SIGTERM'd queue left its run
        # measuring on the card with nobody watching it.
        child.terminate()
        child.wait(timeout=60)
        raise


def serve(space: Workspace, *, poll: float = 120.0, once: bool = False,
          dry_run: bool = False, stop_when_idle: bool = False,
          **kwargs) -> int:
    """Measure until the queue is empty (`once`) or until stopped.

    `stop_when_idle` returns instead of waiting for new plans — what a
    test wants, and what `--once` means for a single model.
    """
    space.prepare()
    stop = {"now": False}

    def _stop(signum, _frame):
        stop["now"] = True
        raise Stopped(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _stop)
        except ValueError:
            pass            # not the main thread (a test)
    space.pidfile.write_text(f"{os.getpid()}\n", encoding="utf-8")
    # Models whose run did not reach the end, THIS PASS. They keep their
    # place in the queue but do not hold it: the loop moves on rather than
    # spinning on the same failure, and the set is emptied when the pass
    # rolls over, so "this pass" is literally one pass.
    #
    # It used to live for the whole process, which made "a restart retries
    # them" the only retry there was — and that is not a small difference,
    # because `run_once` deliberately returns non-zero for a run that DID
    # reach the end: `cli.EXIT_PATHS_DISABLED` (4) means a launch path was
    # switched off by its canary, so those cells were never attempted and
    # the model must be asked again rather than marked done. Two readings of
    # "ask again next pass" met here, and the loop's won: on 2026-10-31 a
    # queue 1 day 2 h old had every remaining model in this set (rc=4 is
    # non-zero, and most runs end that way by design), so `todo` was empty
    # with 14 plans pending and it slept for three hours in a 30-second poll.
    # Nothing was down, nothing logged, and the GPU was idle.
    #
    # Clearing on rollover retries a genuinely failing model once per pass
    # rather than never, which is a retry with the poll interval as its
    # backoff, not the spin the original comment set out to avoid — and it is
    # the only way a canary that starts passing, a provider that ships, or a
    # plan that is rewritten gets picked up without someone restarting the
    # process by hand.
    aside: set = set()
    try:
        while not stop["now"]:
            todo = [m for m in pending(space) if m not in aside]
            if not todo:
                if once or stop_when_idle:
                    return 0
                # The pass is over: whatever was stood aside gets another
                # turn after the poll.
                if aside:
                    space.say(f"pass over, retrying {len(aside)} stood aside")
                    aside = set()
                time.sleep(poll)
                continue
            if run_once(space, todo[0], dry_run=dry_run, **kwargs) != 0:
                aside.add(todo[0])
            if once:
                return 0
    except Stopped as exc:
        space.say(f"stopped ({exc})")
    finally:
        space.pidfile.unlink(missing_ok=True)
    return 0


# ---------------------------------------------------------------------------
# Handing plans to the queue
# ---------------------------------------------------------------------------

def corpus_models(repo_root: Path) -> List[str]:
    """Every model the spec corpus knows, by its snapshot file."""
    snapshots = repo_root / "backend" / "arch" / "snapshots"
    return sorted(model_of(path.stem) for path in snapshots.glob("*.json"))


def unmeasured(space: Workspace, models: Sequence[str]) -> List[str]:
    """Those with no results on this box yet."""
    return [m for m in models if not space.done_marker(m).exists()]


def hand_over(space: Workspace, model_id: str, plan: Path) -> None:
    """Move a finished plan into the queue and let the model be measured
    again. Written elsewhere and moved, so the queue never reads a plan
    that is still being written."""
    from .plan import gaps_path, unplaced_path

    space.prepare()
    target = space.plan_of(model_id)
    plan.replace(target)
    # The plan's gaps travel with it, and a re-plan with none leaves none.
    staged_gaps, gaps = Path(gaps_path(str(plan))), Path(gaps_path(str(target)))
    if staged_gaps.exists():
        staged_gaps.replace(gaps)
    else:
        gaps.unlink(missing_ok=True)
    # Structural rows are not a queue input, but they are part of the plan's
    # report denominator and travel beside it like provider gaps.
    staged_unplaced = Path(unplaced_path(str(plan)))
    target_unplaced = Path(unplaced_path(str(target)))
    if staged_unplaced.exists():
        staged_unplaced.replace(target_unplaced)
    else:
        target_unplaced.unlink(missing_ok=True)
    space.done_marker(model_id).unlink(missing_ok=True)
