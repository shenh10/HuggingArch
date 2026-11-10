"""The measured total is all or nothing (docs/auto_sweep.md §5.4).

Against the Kernel Bench stand-in (`tests/sweep_parity.FixtureBench`): a
phase has a measured time only when every per-op row has one; a zero-cost row
is measured at 0; every other row is in exactly one coverage state, and the
phase's `coverage` lists the keys or ops in each.
"""
from __future__ import annotations

import pytest

from backend.inference.calibration_overlay import zero_cost
from backend.inference.parallel_config import ParallelConfig
from backend.measure.auto_sweep import (DECLINED, MEASURED, MISSING, STATES,
                                        UNPLACED)
from tests.sweep_parity import Deployment, FixtureBench

LLAMA = "meta-llama/Llama-3.2-1B"
CTX = 4096
ROWS = {"prefill": "per_op_timing", "decode": "per_op_timing_decode"}


def _estimate(bench: FixtureBench, *, mode: str = "latest", parallel=None) -> dict:
    dep = Deployment(LLAMA, parallel=parallel, selection_mode=mode,
                     measurement_scope=bench.scope)
    return dep.direct(avg_input_tokens=CTX, prefill_batch=1, decode_batch=4)


def test_every_row_measured_is_a_measured_total():
    with FixtureBench() as bench:
        # Envelope keeps this fold test about all-or-nothing measured totals;
        # latest's declaration semantics have their own cross-layer test.
        est = _estimate(bench, mode="envelope")
    for phase, lst in ROWS.items():
        ph, rows = est["phases"][phase], est[lst]
        assert all(r["measured_state"] in STATES for r in rows)
        zero = [r for r in rows if zero_cost(r)]
        assert zero, "Llama has zero-cost view rows"
        from backend.arch.ops_registry import OP_REGISTRY
        declared_views = {op for op, spec in OP_REGISTRY.items()
                          if getattr(spec, "is_view_op", False)}
        walked_views = [r for r in rows if r.get("op") in declared_views]
        assert walked_views and all(zero_cost(r) for r in walked_views)
        assert all(r["t_measured_s"] == 0.0 and r["measured_state"] == MEASURED
                   for r in zero)
        assert all(r["measured_reason"] == "zero_launch" for r in zero)
        assert ph["coverage"] == {"measured": len(rows), "total": len(rows)}
        assert ph["t_measured_s"] == pytest.approx(sum(r["t_measured_s"] for r in rows),
                                                   rel=1e-12)
        assert ph["throughput"]["measured"] is not None


def _policy(key):
    if key.signature == "gemm":
        return "missing"
    return "measured"


def test_one_row_unmeasured_is_no_measured_total_and_says_why():
    with FixtureBench(_policy) as bench:
        est = _estimate(bench)
    for phase, lst in ROWS.items():
        ph, rows = est["phases"][phase], est[lst]
        assert ph["t_measured_s"] is None
        assert ph["throughput"]["measured"] is None
        by_state = {s: [r for r in rows if r["measured_state"] == s] for s in STATES}
        cov = ph["coverage"]
        assert cov["measured"] == len(by_state[MEASURED]) and cov["total"] == len(rows)
        assert by_state[MISSING]
        assert set(cov["states"]["missing"]) == {r["measured_key"] for r in by_state[MISSING]}
        # The per-op table shows every measured row's own time, whatever the
        # coverage; a missing or ambiguous row shows none.
        assert all(r["t_measured_s"] is not None for r in by_state[MEASURED])
        assert all(r["t_measured_s"] is None for r in by_state[MISSING])


def test_the_envelope_reads_an_ambiguous_key():
    """Two kernels at one key leave `latest` without a pick; the envelope
    takes the faster."""
    def policy(key):
        return "ambiguous" if "rmsnorm" in key.signature else "measured"
    with FixtureBench(policy) as bench:
        latest = _estimate(bench)
    with FixtureBench(policy) as bench:
        envelope = _estimate(bench, mode="envelope")
    assert latest["phases"]["decode"]["t_measured_s"] is None
    assert envelope["phases"]["decode"]["t_measured_s"] is not None
    assert envelope["phases"]["decode"]["coverage"]["measured"] == len(
        envelope["per_op_timing_decode"])


def test_a_leaf_whose_dtype_has_no_corpus_token_is_declined_with_its_reason():
    """A leaf whose own dtype environment has no corpus token is keyed by
    slots that are not this leaf's, so it is not looked up: the row is
    declined, says why, and the step, with every other row measured, has no
    measured total.

    No leaf of the corpus is declined today. DeepSeek-V4's hash router — an
    integer table no measurement dtype covers — is one fused launch since
    78f5e2d35 and is dropped before its dtypes are read; gpt-oss's expert
    banks are keyed by their format since 96c2108e3. So the decline is
    stated here on one leaf, where both sides read it
    (`RowIdentity.leaf_dtype_env`; the plan's half is
    `test_a_declined_row_is_not_planned`)."""
    from unittest import mock
    from backend.inference.benchmark_identity import RowIdentity
    target, real = "post_attention_layernorm", RowIdentity.leaf_dtype_env

    def declining(identity, row):
        env, why = real(identity, row)
        if str(row.get("aggregation_id", "")).endswith(target):
            return env, "the leaf's dtype environment has no corpus token for weight=I64"
        return env, why

    with FixtureBench() as bench, mock.patch.object(
            RowIdentity, "leaf_dtype_env", declining):
        est = _estimate(bench)
    for phase, lst in ROWS.items():
        ph, rows = est["phases"][phase], est[lst]
        declined = [r for r in rows if r["measured_state"] == DECLINED]
        assert declined, phase
        assert all(str(r["aggregation_id"]).endswith(target) for r in declined)
        assert all("no corpus token" in r["measured_reason"] and r["t_measured_s"] is None
                   for r in declined)
        assert ph["t_measured_s"] is None
        assert {d["key"] for d in ph["coverage"]["states"][DECLINED]} == {
            r["measured_key"] for r in declined}


@pytest.mark.regression
def test_a_pipeline_step_waits_for_every_stage(monkeypatch):
    """pp=2: only the bound stage's rows are keyed, so the step has no
    measured time and its coverage names the other stage.

    A foreign stage whose aggregate cost is zero is still measured@0 with
    `zero_launch`; a nonzero one is the closed `pipeline_stage` unplaced row.
    Constructing this real measure covers both cases and used to reject the
    zero term when its structured reason was blank.
    """
    from backend.inference import batch_sweep
    from backend.measure.auto_sweep import ZERO_LAUNCH
    real_measure = batch_sweep.PointMeasure
    measures = []

    class CapturingMeasure(real_measure):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            measures.append(self)

    monkeypatch.setattr(batch_sweep, "PointMeasure", CapturingMeasure)
    with FixtureBench() as bench:
        est = _estimate(bench, parallel=ParallelConfig(pp=2))

    zero_foreign_stages = [
        (label, reason, aggregation_id)
        for measure in measures
        for label, reason, aggregation_id, zero in zip(
            measure.labels, measure.reasons, measure.aggregation_ids,
            measure.formula.zero_cost, strict=True)
        if zero and aggregation_id.startswith("pipeline stage")
    ]
    assert zero_foreign_stages
    assert all(reason == ZERO_LAUNCH for _label, reason, _agg
               in zero_foreign_stages)

    for phase in ROWS:
        ph = est["phases"][phase]
        coverage = ph["coverage"]
        foreign = f"pipeline stage {1 - est['memory']['pp_stage']}"
        assert ph["t_measured_s"] is None
        assert coverage["states"][UNPLACED] == [foreign]
        assert coverage["unplaced_details"]
        assert all(detail == {"op": foreign, "aggregation_id": foreign,
                              "reason": "pipeline_stage"}
                   for detail in coverage["unplaced_details"])
        assert coverage["measured"] < coverage["total"]


def test_parallel_sweep_points_carry_their_coverage():
    from backend.inference.parallel_sweep import estimate_parallel_sweep
    with FixtureBench(_policy) as bench:
        env = Deployment(LLAMA, selection_mode="latest", measurement_scope=bench.scope).env()
        out = estimate_parallel_sweep(mode="tp_tp", context_len=CTX, gpu_skus=["H100-SXM"],
                                      base_env=env)
    pts = out["series"][0]["points"]
    assert pts
    for p in pts:
        if not p["feasible"]:
            continue
        for phase in ROWS:
            cov = p[f"{phase}_measured_coverage"]
            assert cov["measured"] < cov["total"] and cov["states"]["missing"]
            assert p[f"{phase}_t_measured_s"] is None
            assert p[f"{phase}_tps_per_gpu_measured"] is None


def test_partial_total_is_each_row_measured_or_its_sol():
    """The partial track (`Folded.partial`): the step with every measured row
    at its measured time and every other row at its own SOL time. Checked by
    a second route — adding the per-op rows up directly, which at pp=1 with
    no draft is the step — against the backend's SOL-total-minus-shares."""
    with FixtureBench(_policy) as bench:
        est = _estimate(bench, mode="envelope")
    for phase, lst in ROWS.items():
        ph, rows = est["phases"][phase], est[lst]
        blend = sum(r["t_measured_s"] if r["measured_state"] == MEASURED
                    else r["t_op_s"] for r in rows)
        real = sum(r["t_measured_s"] for r in rows
                   if r["measured_state"] == MEASURED)
        assert ph["t_measured_s"] is None
        assert ph["t_partial_s"] == pytest.approx(blend, rel=1e-9)
        assert 0 < ph["partial_measured_share"] < 1
        assert ph["partial_measured_share"] == pytest.approx(real / blend, rel=1e-9)
        assert ph["throughput"]["partial"] is not None


def test_partial_total_at_the_ends_is_sol_or_measured():
    """Nothing measured: the blend is the SOL total, share 0. Everything
    measured: it is the measured total, share 1."""
    with FixtureBench(lambda key: "missing") as bench:
        none = _estimate(bench, mode="envelope")
    with FixtureBench() as bench:
        full = _estimate(bench, mode="envelope")
    for phase in ROWS:
        ph = none["phases"][phase]
        assert ph["t_partial_s"] == pytest.approx(ph["t_sol_s"], rel=1e-9)
        assert ph["partial_measured_share"] == 0
        ph = full["phases"][phase]
        assert ph["t_partial_s"] == pytest.approx(ph["t_measured_s"], rel=1e-9)
        assert ph["partial_measured_share"] == pytest.approx(1.0)


def test_a_chunked_prefill_blends_its_terms_as_every_chunk_runs_them():
    """A long prefill runs in chunks, and its measure is the chunks' terms
    folded (`batch_sweep.fold_chunk_point_measures`), each term's SOL
    carried with its launches. Checked by complementary coverage: with the
    gemm terms measured in one run and every other term in another, the two
    partial totals add to the SOL total plus the measured total — each term
    is SOL in exactly one and measured in exactly the other — which holds
    only if every term's SOL survives the fold at its own weight."""
    def run(policy):
        with FixtureBench(policy) as bench:
            from backend.inference.capacity import estimate_capacity
            dep = Deployment(LLAMA, selection_mode="envelope",
                             measurement_scope=bench.scope)
            est = estimate_capacity(
                **dep.env(), avg_input_tokens=20000, prefill_batch_size=1,
                decode_batch_size=4, prefill_chunk_size=8192)
            assert len(est["phases"]["prefill"]["prefill_chunk_map"]
                       ["chunks"]) > 1, "the prefill runs in chunks"
            return est["phases"]["prefill"]
    gemm = run(lambda key: "measured" if key.signature == "gemm" else "missing")
    rest = run(lambda key: "missing" if key.signature == "gemm" else "measured")
    full = run(lambda key: "measured")
    assert full["t_measured_s"] is not None
    assert 0 < gemm["partial_measured_share"] < 1
    assert 0 < rest["partial_measured_share"] < 1
    assert gemm["t_partial_s"] + rest["t_partial_s"] == pytest.approx(
        full["t_sol_s"] + full["t_measured_s"], rel=1e-9)
