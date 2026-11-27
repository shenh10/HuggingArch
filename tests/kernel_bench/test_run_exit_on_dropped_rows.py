"""A row the device measured and the harness could not persist exits non-zero.

The failure this pins was silent. A collective group measured two cells,
printed `measured 2 rows`, and wrote a header and nothing else: each child
inherited no card, so every row carried an empty `peak_bw_at_test` and
validation refused it (`group._spawn`, fixed). Losing EVERY row already
exited `EXIT_NO_VALID_ROWS`. Losing SOME of them returned 0 — the same
defect, one surviving row away from reading as success.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from kernel_bench.cli import (EXIT_NO_VALID_ROWS, EXIT_PATHS_DISABLED,
                              EXIT_ROWS_DROPPED, run_exit_code)


def test_a_clean_run_exits_zero():
    assert run_exit_code(measured=True, persisted=True, dropped=0,
                         disabled=False) == 0


def test_losing_some_rows_is_not_success():
    """The regression. Rows on disk AND rows refused used to be exit 0."""
    assert run_exit_code(measured=True, persisted=True, dropped=1,
                         disabled=False) == EXIT_ROWS_DROPPED


def test_losing_every_row_still_says_so():
    assert run_exit_code(measured=True, persisted=False, dropped=2,
                         disabled=False) == EXIT_NO_VALID_ROWS


def test_an_env_with_no_applicable_provider_is_not_a_fault():
    """Measuring nothing is a declaration, not a failure: a box that runs
    no provider of this library has nothing to say about these cells."""
    assert run_exit_code(measured=False, persisted=False, dropped=0,
                         disabled=False) == 0


def test_a_dropped_row_outranks_a_disabled_path():
    """Both mean "reached the end", but only one says a measurement was
    lost; reporting the path would hide the row."""
    assert run_exit_code(measured=True, persisted=True, dropped=1,
                         disabled=True) == EXIT_ROWS_DROPPED


def test_a_disabled_path_alone_keeps_its_own_code():
    assert run_exit_code(measured=True, persisted=True, dropped=0,
                         disabled=True) == EXIT_PATHS_DISABLED


def test_the_three_end_of_run_exits_are_distinct():
    """Each says a different thing, so `run-all` clears them separately."""
    codes = [EXIT_NO_VALID_ROWS, EXIT_PATHS_DISABLED, EXIT_ROWS_DROPPED]
    assert len(set(codes)) == 3, codes
    assert all(c != 0 for c in codes), codes


def test_run_all_clears_the_dropped_row_exit():
    """Reaching the end is not a crash.

    `run-all` reads an unrecognised non-zero exit as a crash: it
    quarantines the job in flight and fails the env, so the queue never
    marks the model done. A code meaning "finished, with drops" has to be
    cleared the way the other two already are.
    """
    import inspect

    from kernel_bench import cli
    src = inspect.getsource(cli)
    for name in ("EXIT_NO_VALID_ROWS", "EXIT_PATHS_DISABLED",
                 "EXIT_ROWS_DROPPED"):
        assert f"if rc == {name}:" in src, f"run-all does not clear {name}"
