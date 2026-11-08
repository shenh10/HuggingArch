"""tools/post_deploy_smoke.py's assertions over canned responses: a deploy
whose lookups return zero rows must go red, a healthy one green. The canned
shapes are production's (2026-11-08): browse rows, estimate per-op rows with
`zero_cost` / `measured_state` / `measured_key`, frontier models, /health."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.post_deploy_smoke import (
    Report,
    candidate_groups,
    check_estimate,
    check_frontier,
    check_health,
    check_revisit,
    lake_keys_in_estimate,
)

_ENV = {"activation": "bf16", "kv": "bf16", "metadata": "fp32",
        "output": "bf16", "weight": "bf16"}
_LM_HEAD = '{"B":325,"d_out":151936,"d_reduce":2048,"s_query":1}'
_NORM = '{"B":325,"d_model":2048,"s_query":1}'


def _browse_rows() -> list:
    return [
        {"model": "m/a", "gpu": "H100-SXM", "phase": "decode", "outcome": "measured",
         "signature": "gemm", "physical_shape": _LM_HEAD, "dtype_env_json": _ENV},
        {"model": "m/a", "gpu": "H100-SXM", "phase": "decode", "outcome": "measured",
         "signature": "rmsnorm", "physical_shape": _NORM, "dtype_env_json": _ENV},
        # Not a decode measurement: never a candidate.
        {"model": "m/a", "gpu": "H100-SXM", "phase": "prefill", "outcome": "measured",
         "signature": "gemm", "physical_shape": '{"B":1,"d_out":8,"d_reduce":8,"s_query":64}',
         "dtype_env_json": _ENV},
    ]


def _estimate(lookups_answer: bool) -> dict:
    state = "measured" if lookups_answer else "missing"
    t = 1e-5 if lookups_answer else None
    per_op = [
        # Zero-cost rows are "measured" whatever the lake holds.
        {"op": "view_split", "zero_cost": True, "measured_state": "measured",
         "measured_key": "view_split", "t_measured_s": 0.0},
        {"op": "linear", "zero_cost": False, "measured_state": state, "t_measured_s": t,
         "measured_key": f"gemm {_LM_HEAD}", "act_dtype": "BF16", "weight_dtype": "BF16",
         "output_dtype": "BF16"},
        {"op": "rmsnorm", "zero_cost": False, "measured_state": state, "t_measured_s": t,
         "measured_key": f"rmsnorm {_NORM}", "act_dtype": "BF16", "weight_dtype": "BF16",
         "output_dtype": "BF16"},
        {"op": "add", "zero_cost": False, "measured_state": "missing", "t_measured_s": None,
         "measured_key": 'add {"B":325,"d_model":2048,"s_query":1}', "act_dtype": "BF16"},
    ]
    measured = sum(r["measured_state"] == "measured" for r in per_op)
    return {"per_op_timing_decode": per_op,
            "phases": {"decode": {"coverage": {"measured": measured, "total": len(per_op)}}}}


def _frontier(lookups_answer: bool) -> dict:
    points = [{"tps_user": 50.0, "tps_gpu": 900.0}] if lookups_answer else []
    return {"config": {"gpu_sku": "H100-SXM"}, "models": [
        {"model_id": "m/a", "measured_track": {"state": "reads"},
         "frontier": {"1024": [{"tps_user": 60.0, "tps_gpu": 1000.0}]},
         "frontier_measured": {"1024": points, "4096": []}},
        {"model_id": "m/b", "measured_track": {"state": "reads"},
         "frontier_measured": {"1024": []}},
    ]}


def _health(lookups_answer: bool) -> dict:
    return {"status": "ok", "detail": {"kernel_bench": {"state": "up", "lookups": {
        "lookups": 61, "keys": 125700, "rows": 4100 if lookups_answer else 0}}}}


def _run(lookups_answer: bool) -> Report:
    rep = Report(out=lambda _s: None)
    groups = candidate_groups(_browse_rows())
    assert [g[0] for g in groups] == [("m/a", "H100-SXM", "", 325)]
    keys = groups[0][1]
    est = _estimate(lookups_answer)
    hits = lake_keys_in_estimate(keys, est)
    # The lake's two keys, not the zero-cost row nor the key the lake lacks.
    assert sorted(r["op"] for r in hits) == ["linear", "rmsnorm"]
    check_estimate(rep, "estimate", hits, est)
    check_frontier(rep, "H100-SXM", 200, _frontier(lookups_answer))
    check_revisit(rep, "H100-SXM", 200, 0.4)
    check_health(rep, _health(lookups_answer))
    return rep


def test_lookups_returning_zero_rows_fail_every_content_check():
    """Production's state for weeks: everything answers 200, the tracks say
    "reads", 63 models — and no lookup returns a row. Each content check is
    red on its own, and the zero-cost rows' "measured" does not mask it."""
    rep = _run(lookups_answer=False)
    joined = "\n".join(rep.fails)
    assert len(rep.fails) == 4, joined
    assert "0/2 per-op rows" in joined
    assert "at least one non-zero-cost row" in joined
    assert "draws measured points" in joined
    assert "rows=0" in joined


def test_healthy_deploy_passes():
    rep = _run(lookups_answer=True)
    assert rep.fails == []


def test_engine_error_and_slow_revisit_fail():
    rep = Report(out=lambda _s: None)
    body = _frontier(True)
    body["models"][1]["measured_track"] = {"state": "engine_error"}
    check_frontier(rep, "H100-SXM", 200, body)
    check_revisit(rep, "H100-SXM", 200, 7.0)
    check_frontier(rep, "H100-SXM", 503, {"retry_after_s": 5})
    assert any("engine_error" in f for f in rep.fails)
    assert any("revisit" in f for f in rep.fails)
    assert any("answered 200 (got 503)" in f for f in rep.fails)


def test_dtype_disagreement_is_not_a_lake_hit():
    """A lake row measured in bf16 says nothing about an fp8 GEMM's key."""
    keys = {f"gemm {_LM_HEAD}": _ENV}
    est = {"per_op_timing_decode": [
        {"zero_cost": False, "measured_state": "missing", "measured_key": f"gemm {_LM_HEAD}",
         "act_dtype": "FP8", "weight_dtype": "FP8", "output_dtype": "BF16"}]}
    assert lake_keys_in_estimate(keys, est) == []
    assert json.loads(_LM_HEAD)["B"] == 325


def test_a_measurement_is_reproduced_under_its_own_deployment():
    """A shared row was measured at the per-rank shape its plan's deployment
    cut; the estimate must name that deployment, or a TP>1 row is never
    reproduced and the gate goes red on a healthy lake."""
    from tools.post_deploy_smoke import parallel_of
    assert parallel_of("tp_attention=4,tp_dense_ffn=4,ep_moe=8") == {
        "tp_attention": 4, "tp_dense_ffn": 4, "ep_moe": 8}
    assert parallel_of("") == {}
    groups = candidate_groups([{
        "model": "m/a", "gpu": "H100-SXM", "phase": "decode", "outcome": "measured",
        "parallel": "tp_attention=4", "signature": "gemm",
        "physical_shape": '{"B":8,"d_out":1,"d_reduce":1,"s_query":1}',
        "dtype_env_json": {"activation": "bf16"}}])
    assert groups[0][0] == ("m/a", "H100-SXM", "tp_attention=4", 8)
