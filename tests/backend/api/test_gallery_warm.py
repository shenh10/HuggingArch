"""The offline Gallery warm orders layouts, not model names."""
from __future__ import annotations

import asyncio

import backend.api.gallery_frontier as f
from backend.inference.parallel_config import ParallelConfig
from backend.measure.scope import NONE as NO_MEASUREMENTS


def test_layout_work_is_ordered_by_parameter_count_times_gpus():
    walks = [
        ("a/model", {"cards": ["B200", "B300"],
                     "layouts": [ParallelConfig().to_dict(),
                                 ParallelConfig(tp_attention=4).to_dict()]}),
        ("b/model", {"cards": ["B200", "B300"],
                     "layouts": [ParallelConfig(tp_attention=2).to_dict()]}),
    ]
    works = f._warm_layout_works(
        walks, f.FrontierView("B200"), {"a/model": 2, "b/model": 10})
    assert [(work.model_id, work.cost, work.layout_index)
            for work in works] == [
        ("b/model", 20, 0),
        ("a/model", 8, 1),
        ("a/model", 2, 0),
    ]


def test_all_models_layouts_share_one_queue_and_finish_per_model(monkeypatch):
    """Every missing layout is submitted before any model entry is assembled;
    a model finishes only after all of its layouts have landed."""
    plans = {
        "a/model": {"cards": ["B200", "B300"],
                    "layouts": [ParallelConfig().to_dict(),
                                ParallelConfig(tp_attention=4).to_dict()]},
        "b/model": {"cards": ["B200", "B300"],
                    "layouts": [ParallelConfig(tp_attention=2).to_dict()]},
    }
    monkeypatch.setattr(f, "iter_snapshot_ids", lambda: list(plans))
    monkeypatch.setattr(f, "_view_scope",
                        lambda _view: (NO_MEASUREMENTS, "none"))

    async def run_off(fn, *args):
        if fn is f._view_scope:
            return (NO_MEASUREMENTS, "none")
        if fn is f._frontier_read:
            return {"walk": plans[args[0]]}
        raise AssertionError(fn)

    events: list[tuple[str, str]] = []

    async def run_ranked(fn, *args, name=None):
        if fn is f._layout_job:
            work = args[0]
            events.append(("layout", work.model_id))
            return {"layout": work.layout_index}, object()
        if fn is f._finish_frontier:
            events.append(("finish", name))
            return {"model_id": name, "built": True}
        raise AssertionError(fn)

    monkeypatch.setattr(f, "run_off_loop", run_off)
    monkeypatch.setattr(f, "run_ranked", run_ranked)
    monkeypatch.setattr(f, "report_partial", lambda **_kwargs: None)
    monkeypatch.setattr(f, "report_progress", lambda *_args: None)
    out = asyncio.run(f.warm_frontier(
        f.FrontierView("B200"), {"a/model": 2, "b/model": 10}))
    assert [model["model_id"] for model in out["models"]] == [
        "a/model", "b/model"]
    assert [event for event in events if event[0] == "layout"] == [
        ("layout", "b/model"), ("layout", "a/model"), ("layout", "a/model")]
    last_a = max(i for i, event in enumerate(events)
                 if event == ("layout", "a/model"))
    assert events.index(("finish", "a/model")) > last_a
