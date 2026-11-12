"""The functions provider excludes storage infrastructure from traces."""
from __future__ import annotations

import os
import sys

from backend.code_trace.provider import TRACED_EXCLUDE, TracedFunctions


def test_sol_pool_is_excluded_from_traced_trees():
    """The SOL record store is infrastructure, not computation: its code
    runs when a result is read through it, but including it in every
    trace means every storage edit invalidates every record."""
    assert "backend.inference.sol_pool" in TRACED_EXCLUDE


def test_import_trees_skips_excluded_modules():
    provider = TracedFunctions(canonical=False, strict=False)
    failed = provider._import_trees()
    assert not failed  # everything that should import did


def test_excluded_files_resolve_to_real_paths():
    provider = TracedFunctions(canonical=False, strict=False)
    import backend.inference.sol_pool as sol_pool
    assert os.path.abspath(sol_pool.__file__) in provider._excluded_files
