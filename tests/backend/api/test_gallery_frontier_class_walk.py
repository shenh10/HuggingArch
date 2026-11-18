"""Canonical serving laws replace Gallery's second ClassSweep walk."""
from __future__ import annotations

import pytest
import math

import backend.api.gallery_frontier as f
from backend.api.gallery_overview import _grid_points
from backend.api.estimate_env import resolve_estimate_environment
from backend.arch.eval.precision import AS_SHIPPED, PRESETS
from backend.inference.parallel_config import ParallelConfig
from backend.inference.parallel_sweep import (DEFAULT_SWEEP_GPUS, class_cards,
                                             frontier_layouts, sweep_caps)
from backend.inference.defaults import GPU_MEMORY_FRACTION_DEFAULT
from backend.inference.sweep_grid import FRONTIER_MAX_GPUS, GALLERY_SEQ_LENS
from tests.snapshots import resolve_snapshot

SEQ_LENS = list(GALLERY_SEQ_LENS)


def _same(got, want, path="priced"):
    """Whole public shape: same keys, non-numbers equal, numbers to 1e-12."""
    if isinstance(got, dict):
        assert set(got) == set(want), f"{path}: {sorted(got)} != {sorted(want)}"
        for key in got:
            if key == "timings":
                continue
            _same(got[key], want[key], f"{path}.{key}")
        return
    if isinstance(got, list):
        assert len(got) == len(want), path
        for i, (one, other) in enumerate(zip(got, want, strict=True)):
            _same(one, other, f"{path}[{i}]")
        return
    if isinstance(got, (int, float)) and not isinstance(got, bool):
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=0), path
    else:
        assert got == want, path


@pytest.mark.parametrize("gpu", ["H100-SXM", "B200", "GB200"])
def test_each_card_prices_the_layouts_its_own_frontier_enumerates(gpu):
    resolved = resolve_snapshot("Qwen/Qwen3-30B-A3B")
    plan = f._walk_plan(resolved, f.FrontierView(gpu))
    caps = sweep_caps({}, resolved=resolved)
    # The kept bundle carries every card of the class the Scaling series
    # (every card of the sheet) prices, not only the Gallery's.
    assert plan["cards"] == class_cards(gpu, DEFAULT_SWEEP_GPUS) and gpu in plan["cards"]
    for sku in plan["cards"]:
        own = [cfg.to_dict() for cfg in frontier_layouts(caps, sku)]
        assert plan["galleryLayoutsByCard"][sku] == own
        assert plan["layoutsByCard"][sku] == plan["layouts"]
        assert all(p in plan["layouts"] for p in own)
    assert all(ParallelConfig.from_dict(cfg).cluster_size <= FRONTIER_MAX_GPUS
               for own in plan["galleryLayoutsByCard"].values()
               for cfg in own)
    assert all(ParallelConfig.from_dict(cfg).cluster_size <= FRONTIER_MAX_GPUS
               for cfg in plan["layouts"])
    assert len(plan["layouts"]) == len({str(sorted(p.items()))
                                        for p in plan["layouts"]})


def test_a_view_at_another_hbm_fraction_walks_its_own_card_alone():
    view = f.FrontierView("H100-SXM", 0.8)
    assert not view.keeps_walk() and view.walk_cards() == ["H100-SXM"]
    assert f.FrontierView("H100-SXM").walk_cards() == class_cards(
        "H100-SXM", DEFAULT_SWEEP_GPUS)


def test_a_dtype_the_class_has_no_peak_for_is_refused_in_the_record():
    model = "meta-llama/Llama-3.2-1B"
    precision = PRESETS["w_nvfp4_w4a4_kv_fp8"].canonical()
    record = f._layout_law((model, ["H100-SXM"], .9, precision, {}))
    assert "no_peak" in record


LAW_CASES = [
    ("H100-SXM", "deepseek-ai/DeepSeek-V4-Flash",
     {"tp_attention": 8, "tp_dense_ffn": 8, "ep_moe": 8}),
    ("B200", "deepseek-ai/DeepSeek-V4-Flash",
     {"tp_attention": 8, "tp_dense_ffn": 8, "ep_moe": 8}),
    ("H100-SXM", "meta-llama/Llama-3.2-3B", {"tp_attention": 2,
                                            "tp_dense_ffn": 2}),
]

#: What identifies a per-op row for the row-set comparison: the row, its op
#: and every precision it is priced at (§1.5: compare the set, its order
#: and its dtypes, not only the time).
_ROW_FIELDS = ("uid", "op", "act_dtype", "weight_dtype", "output_dtype",
               "output_format")


def _own_walk(model: str, sku: str, parallel: dict, precision: dict) -> dict:
    req = f._layout_request(model, sku, GPU_MEMORY_FRACTION_DEFAULT, precision,
                            parallel)
    grid = [{"key": "b1", "prefill_batch_size": 1, "decode_batch_size": 1}]
    return _grid_points(resolve_estimate_environment(req),
                        seq_lens=SEQ_LENS[:1], batch_grid=grid)["b1"][0]["sol"]


def _row_set(per: dict) -> dict:
    return {phase: [tuple(r.get(k) for k in _ROW_FIELDS) for r in per[key]]
            for phase, key in (("prefill", "per_op_timing"),
                               ("decode", "per_op_timing_decode"))}


@pytest.mark.regression
@pytest.mark.parametrize("gpu,model,parallel", LAW_CASES)
def test_the_law_a_card_is_priced_by_is_its_own_classs_walk(gpu, model,
                                                            parallel):
    """The oracle for the canonical law road (`_walk_plan` → `_layout_law` →
    `servingLawPoint`): the card the view's law was walked on runs the same
    per-op rows as the card itself — same rows, same order, same dtypes —
    and the law priced on the card is the card's own single estimate."""
    from backend.inference.card import Card
    from backend.inference.sol_law import servingLawPoint
    from backend.inference.sweep_grid import GALLERY_OUTPUT_TOKENS

    precision = AS_SHIPPED.canonical()
    view = f.FrontierView(gpu)
    plan = f._walk_plan(resolve_snapshot(model), view)
    record = f._layout_law((model, plan["cards"], GPU_MEMORY_FRACTION_DEFAULT,
                            precision, parallel))
    assert "phases" in record, record
    own = _own_walk(model, gpu, parallel, precision)
    walked = (own if record["walked_on"] == gpu else
              _own_walk(model, record["walked_on"], parallel, precision))
    assert _row_set(walked) == _row_set(own), (record["walked_on"], gpu)
    served = servingLawPoint(
        record, card=Card.of(gpu), context=SEQ_LENS[0],
        outputTokens=GALLERY_OUTPUT_TOKENS, decodeBatch=1, prefillBatch=1,
        comm_latency_us=view.comm_latency_us,
        launch_floor_us=view.launch_floor_us)
    for phase in ("prefill", "decode"):
        assert served["phases"][phase]["t_sol_s"] == pytest.approx(
            own["phases"][phase]["t_sol_s"], rel=1e-9, abs=0), phase

def test_a_class_law_priced_on_each_card_is_that_cards_own_law():
    """The bundle law is one walk, but each card still prices its own entry:
    projecting the class record on a card equals projecting that card's own
    record — its capacity and its card binding — for the whole point shape."""
    model = "meta-llama/Llama-3.2-1B"
    resolve_snapshot(model)
    cards = class_cards("B200")
    precision = AS_SHIPPED.canonical()
    walked = f._layout_law((model, cards, GPU_MEMORY_FRACTION_DEFAULT,
                            precision, {}))
    assert "phases" in walked, walked
    for sku in cards:
        fresh = f._layout_law((model, [sku], GPU_MEMORY_FRACTION_DEFAULT,
                               precision, {}))
        assert "phases" in fresh, (sku, fresh)
        _same(f._restamp_layout(walked, sku, ParallelConfig()),
              f._restamp_layout(fresh, sku, ParallelConfig()), sku)
