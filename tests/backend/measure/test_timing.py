"""The warm's optional JSONL timing log."""
from __future__ import annotations

import json

import backend.measure.timing as timing


def test_timing_is_disabled_without_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "_PATH", "")
    timing.log_timing("event", value=1)
    assert not list(tmp_path.iterdir())


def test_timing_writes_one_json_line(tmp_path, monkeypatch):
    path = tmp_path / "timing.jsonl"
    monkeypatch.setattr(timing, "_PATH", str(path))
    timing.log_timing("event", model="org/model", elapsed_s=1.25)
    assert path.exists()
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["event"] == "event"
    assert record["model"] == "org/model"
    assert record["elapsed_s"] == 1.25
