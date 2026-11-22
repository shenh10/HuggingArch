"""The release entry point keeps every producer on the deployed trace mode."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.release import _trace_mode
from tools.deploy_env import HOST

ROOT = Path(__file__).resolve().parents[2]


def _host(tmp_path, trace: str) -> Path:
    """A copy of the production host's files with HA_CODE_TRACE set."""
    shutil.copytree(ROOT / HOST, tmp_path / HOST)
    runtime = tmp_path / HOST / "runtime.env"
    lines = [f"HA_CODE_TRACE={trace}" if line.startswith("HA_CODE_TRACE=") else line
             for line in runtime.read_text(encoding="utf-8").splitlines()]
    runtime.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_the_production_host_is_the_single_trace_mode_source(tmp_path):
    assert _trace_mode(_host(tmp_path, "functions")) == "functions"
    assert _trace_mode(_host(tmp_path / "g", "global")) == "global"


def test_an_unknown_trace_mode_is_refused_before_warm_or_publish(tmp_path):
    with pytest.raises(ValueError, match="HA_CODE_TRACE"):
        _trace_mode(_host(tmp_path, "sidecar"))
