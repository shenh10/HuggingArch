"""`/health`'s memory report: the number an OOM alert watches."""
from __future__ import annotations

import pytest

from backend.api._app import _memory_report


def test_the_report_names_the_kernels_own_available_estimate():
    report = _memory_report()
    if "error" in report:                      # a non-Linux dev box
        pytest.skip(report["error"])
    assert report["total_mb"] > 0
    assert 0 <= report["available_mb"] <= report["total_mb"]
    assert 0.0 <= report["used_pct"] <= 100.0
