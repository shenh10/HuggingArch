"""A preparation that cannot load its weights refuses before phase walks."""
from __future__ import annotations

import pytest

from backend.api.gallery_frontier import (
    FrontierView,
    _GALLERY_PRECISION,
    _layout_law,
    _resolve_one,
    _walk_plan,
    layout_label,
)
from backend.api._models import InferenceEstimateRequest
from backend.api.estimate_env import resolve_estimate_environment
from backend.arch.eval.precision import AS_SHIPPED
from backend.inference import capacity
from backend.inference.capacity import CapacityKnobs, _capacity_pending, _capacity_preparation
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import snapshot_config


MODEL = "moonshotai/Kimi-K3"

# The pre-refactor producer's refused set, captured layout by layout. Reasons
# are compared to the complete capacity answer rather than pinned through
# rounded GiB strings: near the boundary two values rounded to 126.76 and
# 126.90 can still have the exact ordering the budget computed.
EXPECTED_REFUSALS = {
    "TP1",
    "EP2·DP2",
    "TP2·EP2",
    "CP2·EP2",
    "TP2",
    "EP4·DP4",
    "TP2·EP4·DP2",
    "TP4·EP4",
    "CP4·EP4",
    "TP4",
    "EP8·DP8",
    "TP2·EP8·DP4",
    "TP4·EP8·DP2",
    "TP8·EP8",
    "CP8·EP8",
    "TP8",
    "EP16·DP16",
    "TP2·EP16·DP8",
    "CP16·EP16",
    "EP32·DP32",
    "CP32·EP32",
    "EP64·DP64",
    "CP64·EP64",
}
EXPECTED_VALID = {
    "TP4·EP16·DP4", "TP8·EP16·DP2", "TP16",
    "TP2·EP32·DP16", "TP4·EP32·DP8", "TP8·EP32·DP4", "TP32",
    "TP2·EP64·DP32", "TP4·EP64·DP16", "TP8·EP64·DP8", "TP64",
}


@pytest.mark.regression  # walks all 23 impossible Kimi layouts
def test_kimi_weight_refusals_keep_their_old_reasons(monkeypatch):
    snapshot_config(MODEL)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    resolved = _resolve_one(MODEL)[1]
    plan = _walk_plan(resolved, FrontierView("H100-SXM"), scaling=False)
    layouts = [ParallelConfig.from_dict(value) for value in plan["layouts"]]
    labels = [layout_label(layout) for layout in layouts]

    assert set(labels) == EXPECTED_REFUSALS | EXPECTED_VALID
    for layout, label in zip(layouts, labels, strict=True):
        if label not in EXPECTED_REFUSALS:
            continue
        record = _layout_law((
            MODEL, plan["cards"], .9, _GALLERY_PRECISION.canonical(), layout))
        reason = record.get("refused")
        assert reason is not None
        assert reason.startswith("weights (")
        assert "on H200 at this parallel config." in reason
        assert reason.endswith("Raise TP/EP or use a smaller-weight model.")


def test_a_weight_refusal_finishes_without_phase_walks(monkeypatch):
    """The preparation's refusal is the one every estimate consumer reads."""
    snapshot_config(MODEL)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    request = InferenceEstimateRequest(
        model_id=MODEL,
        gpu_sku="H200",
        parallel=ParallelConfig().to_dict(),
        precision={"canonical": AS_SHIPPED.canonical()},
        avg_input_tokens=1024,
        avg_output_tokens=256,
        selection_mode="theoretical_only",
    )
    environment = resolve_estimate_environment(request)
    knobs = CapacityKnobs.from_kwargs(environment, envelope=frozenset({
        "config", "gpu_sku", "avg_output_tokens", "prefill_batch_size",
        "decode_batch_size", "prefill_chunk_size"}))
    preparation = _capacity_preparation(
        config=environment["config"], gpu_sku="H200", knobs=knobs)
    assert preparation.refusal_reason

    def no_phase_walk(**_kwargs):
        raise AssertionError("a weight-refused preparation walked a phase")

    monkeypatch.setattr(capacity, "overlay_walk", no_phase_walk)
    pending = _capacity_pending(
        preparation,
        avg_input_tokens=1024,
        avg_output_tokens=256,
        breakdown=False,
    )
    result = pending.finish()
    assert result["max_batch_per_gpu"] == 0
    assert "per_op_timing" not in result
    assert result["phases"]["prefill"]["batch"] is None
    assert result["phases"]["decode"]["batch"] is None

    # What a refusal does NOT suspend: the budget facts. A rank that cannot
    # load its weights is still asked which modules those weights are and how
    # many bytes each holds — that is the answer the Memory card draws and the
    # question the user is asking when a deployment refuses.
    #
    # Checked by the second route rather than by the field: sum the published
    # roots' own per-rank bytes and compare with the per-rank total published
    # beside them. Reading `total_bytes_per_rank` alone cannot fail, because a
    # second producer of that number is exactly the defect — the refusal path
    # once hand-built a `weight_breakdown` whose tree was empty and whose total
    # came from the resident-weight ledger instead of the tree, so the same
    # field meant "weights the rank holds in HBM" when the card refused and
    # "weights the tree places on the rank" when it did not (they part by
    # 1.68x on DeepSeek-V4.1-Flash, which keeps tables off the device).
    wb = result["weight_breakdown"]
    roots = list(wb["groups"]) + list(wb["model_level"])
    assert roots, "a refused card published no weight tree"
    assert sum(int(root["params"] or 0) for root in roots) > 0
    assert (sum(float(root["bytes_per_rank"] or 0.0) for root in roots)
            + float(wb["unattributed_scale_bytes"] or 0.0)
            == pytest.approx(wb["total_bytes_per_rank"], rel=1e-12))


def test_a_draft_weight_refusal_is_not_read_as_capacity_data(monkeypatch):
    """A drafted record that cannot load says `refused`; it has no capacity
    table for the caller to move onto a plain record."""
    model = "inclusionAI/Ling-2.6-1T"
    snapshot_config(model)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    plan = _walk_plan(
        _resolve_one(model)[1], FrontierView("H100-SXM"), scaling=False)
    layout = next(
        ParallelConfig.from_dict(value) for value in plan["layouts"]
        if layout_label(ParallelConfig.from_dict(value)) == "TP4·EP8·DP2")
    record = _layout_law((
        model, plan["cards"], .9, _GALLERY_PRECISION.canonical(), layout))
    assert set(record) == {"refused"}
    assert record["refused"].startswith("weights (")
    assert " + recurrent state (" in record["refused"]
