"""A phase cell's points — prompt, cached prefix, verify width, batch — are
one walk read at each (`plan.build_jobs_over_points`): every point plans
exactly the jobs a walk there plans, keys, shapes and the numerators the
SOL gate divides by."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.inference.framework_bindings import (ServingDeployment,
                                                  get_binding)
from backend.inference.parallel_config import ParallelConfig
from kernel_bench.plan import PlanPoint, build_jobs, build_jobs_over_points
from tests.snapshots import resolve_snapshot

MODEL = "Qwen/Qwen3-8B"
TP2 = ParallelConfig(tp_attention=2, tp_dense_ffn=2)


def _keyed(jobs):
    return {json.dumps([j["signature"], j["dims"], j["dtype_env"],
                        j["node_spec_id"], j["count_per_step"],
                        j["kernel_params"]], sort_keys=True): j
            for j in jobs}


@pytest.mark.parametrize("phase, points", [
    ("prefill", [PlanPoint(8192, 0, 1, 1), PlanPoint(8192, 7168, 1, 1),
                 PlanPoint(2048, 0, 1, 4)]),
    ("decode", [PlanPoint(8192, 0, 1, 1), PlanPoint(8192, 0, 3, 1),
                PlanPoint(2048, 0, 1, 16)]),
])
def test_each_point_of_the_one_walk_plans_what_a_walk_there_does(phase, points):
    from backend.arch.precision_profile import AS_SHIPPED
    from backend.inference.walk_registry import WALK_CELLS, shared_walk_cells

    resolved = resolve_snapshot(MODEL)
    kw = dict(model_id=MODEL, phase=phase, avg_output_tokens=256,
              parallel=TP2, resolved=resolved)
    WALK_CELLS.clear()
    family = shared_walk_cells(resolved, TP2, AS_SHIPPED, model_id=MODEL)
    swept = build_jobs_over_points(points=points, **kw)
    assert shared_walk_cells(resolved, TP2, AS_SHIPPED,
                             model_id=MODEL) is family
    for point in points:
        want = _keyed(build_jobs(
            batch_size=point.batch, seq_len=point.seq_len,
            s_prefix_cached=point.s_prefix_cached, s_verify=point.s_verify,
            **kw)[0])
        got = _keyed(swept[point][0])
        assert set(got) == set(want), point
        for key, job in want.items():
            for field in ("flops_per_rank", "access_bytes_per_rank"):
                assert got[key][field] == pytest.approx(
                    job[field], rel=1e-9), (point, field)


def test_routing_skew_is_a_plan_point_coordinate(monkeypatch):
    """ρ is point-time: the plan binds it when it reads the shared walk."""
    from backend.inference import step_eval

    seen = []
    real = step_eval.evaluate_step

    def spy(*args, **kwargs):
        seen.append(kwargs["ep_load_factor"])
        return real(*args, **kwargs)

    monkeypatch.setattr(step_eval, "evaluate_step", spy)
    point = PlanPoint(128, 0, 1, 1, 1.5)
    assert point.counts("prefill")["rho"] == 1.5
    build_jobs_over_points(
        model_id=MODEL, points=[point], phase="prefill",
        avg_output_tokens=0, parallel=TP2,
        resolved=resolve_snapshot(MODEL), gpu="H100-SXM")
    assert seen == [1.5]


def test_a_cardless_plan_says_it_is_not_a_box_plan():
    """`gpu=None` remains a shape-inspection mode, but only visibly.

    Its mixed-precision dtype key has no framework launch answer and must
    never be mistaken for a queue plan. A deployment-answered binding is the
    production side of that boundary: refusing it without the card prevents
    the old pre-fallback key from leaking through a real caller.
    """
    resolved = resolve_snapshot(MODEL)
    jobs, _dropped = build_jobs(
        model_id=MODEL, batch_size=1, seq_len=128, phase="prefill",
        avg_output_tokens=0, resolved=resolved)
    assert jobs
    assert {job.get("plan_scope") for job in jobs} == {"cardless-shapes"}

    binding = get_binding("vllm").deployed(ServingDeployment.at(
        gpu_sku="H100-SXM", activation_dtype="BF16"))
    with pytest.raises(ValueError, match="names that card"):
        build_jobs(
            model_id=MODEL, batch_size=1, seq_len=128, phase="prefill",
            avg_output_tokens=0, resolved=resolved, binding=binding)


def test_a_refused_chunk_point_is_dropped_without_replacing_the_walk():
    """DCP may admit an endpoint's first chunk and refuse a later prefix.

    The refused launch is named and empty.  It must not raise away the whole
    one-walk call, and especially must not fall back to the endpoint's
    full-sequence geometry.
    """
    from kernel_bench.plan import _Forward, _read_group

    parallel = ParallelConfig(tp_attention=4, dcp=2)
    first = PlanPoint(8192, 0, 1, 1)
    refused = PlanPoint(16384, 8192, 1, 1)
    walked = []

    def analyze(point):
        walked.append(point)
        return [[_Forward(object())]]

    def jobs_over_ranks(_ranks, _rows_of, _step_seq):
        return ([{"signature": "gemm"}], [], [])

    out = _read_group([first, refused], analyze, jobs_over_ranks,
                      lambda point: 1, parallel, "prefill")

    assert walked == [first]
    assert out[first] == ([{"signature": "gemm"}], [], [])
    assert out[refused] == ([], [
            "prefill point seq_len=16384 s_prefix_cached=8192 batch=1: "
            "prefill with a cached prefix under a position-sharded cache "
            "(dcp>1) is not modelled: the prefix lives 1/dcp per rank and "
            "the kernel needs it whole. Set dcp=1 or pass no prefix."
        ], [])


def test_the_anchor_is_canonical_not_caller_order(monkeypatch):
    """Two callers listing the same points ask for the same walk family."""
    from backend.inference import batch_sweep
    from kernel_bench.plan import _Forward, _read_group

    class FakeAnalysis:
        def __init__(self):
            self.per_op_timing = ()
            self.point = {}
            self.laws = SimpleNamespace(analysis=SimpleNamespace(
                admit=lambda counts: None))

    high = PlanPoint(8192, 0, 1, 1)
    low_small = PlanPoint(1024, 0, 1, 1)
    anchor = PlanPoint(1024, 0, 1, 64)
    walked = []

    def analyze(point):
        walked.append(point)
        return [[_Forward(FakeAnalysis())]]

    def jobs_over_ranks(_ranks, rows_of, _step_seq):
        return ([{"rows": rows_of(_Forward(FakeAnalysis()))}], [], [])

    monkeypatch.setattr(
        batch_sweep, "rows_at",
        lambda rows, at, **kwargs: {"counts": dict(at.counts)})
    out = _read_group([high, anchor, low_small], analyze, jobs_over_ranks,
                      lambda point: 1, ParallelConfig(), "prefill")
    assert walked == [anchor]
    assert out[anchor][0] == [{"rows": ()}]
    assert out[high][0] == [{"rows": {"counts": high.counts("prefill")}}]
