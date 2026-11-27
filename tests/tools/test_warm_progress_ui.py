"""The warm progress dashboard's summary and incremental reader."""
from __future__ import annotations

from tools.warm_progress_ui import TimingState, summarize


def test_incremental_reading_skips_a_partial_last_line(tmp_path):
    path = tmp_path / "timing.jsonl"
    path.write_text('{"event":"warm_start"}\n{"event":"read"', encoding="utf-8")
    state = TimingState(path)
    assert len(state.snapshot()) == 1
    with path.open("a", encoding="utf-8") as stream:
        stream.write(',"gpu":"H100","elapsed_s":1}\n')
    assert [r["event"] for r in state.snapshot()] == ["warm_start", "read"]


def test_summary_counts_progress_by_gpu():
    records = [
        {"event": "warm_start", "engine": "e", "workers": 10},
        {"event": "gpu_start", "gpu": "H100", "models": 2, "layouts": 3},
        {"event": "read", "gpu": "H100", "model_id": "m", "elapsed_s": 1},
        {"event": "layout_compute", "gpu": "H100", "model_id": "m",
         "layout_index": 0, "elapsed_s": 2},
        {"event": "layout", "gpu": "H100", "model_id": "m",
         "layout_index": 0, "elapsed_s": 3},
        {"event": "model", "gpu": "H100", "model_id": "m"},
        {"event": "gpu_end", "gpu": "H100"},
    ]
    summary = summarize(records)
    assert summary["overall"]["models"] == 1
    assert summary["overall"]["layouts"] == 1
    assert summary["gpus"][0]["total_models"] == 2
    assert summary["gpus"][0]["total_layouts"] == 3
    assert summary["gpus"][0]["done"] is True
    assert summary["top_models"][0]["layout_compute_s"] == 2
