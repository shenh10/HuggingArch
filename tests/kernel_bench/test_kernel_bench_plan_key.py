"""The measurement key has one owner, dtype half included.

Three producers write this key — the kernel_bench plan, the contributor CSV
template, the serving overlay's own rows — and the overlay reads it back. Any
slot spelled differently is not a wrong number, it is a permanent silent
lookup miss: the measurement exists in the lake and no estimate can ever
reach it.

The shape half was given one owner (`RowIdentity.of`) after three producers
drifted on it. The dtype half then drifted the same way and worse. The plan
passed three of the six slots to `compute_per_row_dtype_env_json` and
re-derived two more from the spec's materialized tensors, while the overlay
read the row's own fields — so `metadata` was written `fp32` against the
row's real scale dtype, and on four models EVERY measured job was
unreachable: Kimi-K3 (21 op classes), DeepSeek-V4-Flash (28), gpt-oss-20b
(15), Kimi-K2-Thinking (14), and a quarter of DeepSeek-V3.2's.

So this test does not check that one function returns what it returns. It
runs the two real entry points — `plan.build_jobs` and the serving
`overlay_walk` — over the same model, phase and batch, and demands the key
bytes agree per node. That is the assertion that was false.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.analyzer import fetch_config
from backend.arch.loader import SpecLoader
from backend.inference.calibration_overlay import overlay_walk
from backend.inference.framework_bindings import DEFAULT_FRAMEWORK, get_binding
from backend.inference.parallel_config import ParallelConfig
from kernel_bench.plan import build_jobs

#: The four models whose every measured job was unreachable, plus the one
#: that was a quarter unreachable, plus a dense unquantized control that was
#: already fine and must stay fine.
MODELS = (
    "meta-llama/Llama-3.2-3B",
    "openai/gpt-oss-20b",
    "deepseek-ai/DeepSeek-V3.2",
    "moonshotai/Kimi-K2-Thinking",
    "deepseek-ai/DeepSeek-V4-Flash",
    "moonshotai/Kimi-K3",
)

SEQ = 4096
PHASE = "decode"


def _plan_keys(model_id: str, resolved) -> dict:
    jobs, _dropped = build_jobs(avg_output_tokens=0, model_id=model_id, batch_size=1, seq_len=SEQ,
                                phase=PHASE, resolved=resolved)
    return {j["node_spec_id"]: json.dumps(j["dtype_env"], sort_keys=True,
                                          separators=(",", ":"))
            for j in jobs}


def _query_keys(model_id: str, resolved) -> dict:
    """The keys the serving estimate asks the lake for, through the same
    entry point `/inference/estimate` uses. Every row states its key
    (`calibration_overlay.identity_columns`) whether or not the request goes
    on to look a measurement up, so the walk needs no selection mode."""
    walk = overlay_walk(
        resolved=resolved, model_id=model_id, parallel=ParallelConfig(),
        phase=PHASE, avg_input_tokens=SEQ, avg_output_tokens=0, max_batch=1,
        framework=DEFAULT_FRAMEWORK, framework_binding=get_binding(DEFAULT_FRAMEWORK),
        gpu_sku="H100-SXM", gpu=None, s_prefix_cached=0, s_verify=1,
        precision=None, peak_flops=1e15, peak_bw=3.35e12,
        peak_intra_bw=None, peak_inter_bw=None, intra_bw_label="",
        inter_bw_label="", comm_latency_s=0.0, launch_floor_s=0.0,
        breakdown=False)
    rows, *_ = walk.finish()
    return {str(r.get("aggregation_id", "") or ""): r.get("dtype_env_json")
            for r in rows}


@pytest.mark.regression
class TestOneOwnerForTheDtypeHalfOfTheKey(unittest.TestCase):
    def test_the_two_producers_agree_byte_for_byte(self):
        for model_id in MODELS:
            with self.subTest(model=model_id):
                config = fetch_config(model_id)
                resolved = SpecLoader().match(config, model_id=model_id)
                self.assertIsNotNone(resolved, model_id)
                plan = _plan_keys(model_id, resolved)
                query = _query_keys(model_id, resolved)
                joined = [sid for sid in plan
                          if query.get(sid) is not None]
                # A plan with no row in common with the estimate would pass
                # the comparison below vacuously.
                self.assertTrue(joined, f"{model_id}: no row joins the two sides")
                bad = {sid: (plan[sid], query[sid])
                       for sid in joined if plan[sid] != query[sid]}
                self.assertEqual(
                    bad, {},
                    f"{model_id}: {len(bad)} of {len(joined)} measured rows "
                    f"carry a dtype key the estimate will never ask for")


class TestTheGemmWeightPredicateIsGone(unittest.TestCase):
    """`is_weight_gemm_signature` was a list of signature names kept beside
    the kernel catalog, and its only job was to decide whether the plan
    should re-derive a weight dtype from the spec. Nothing re-derives one
    now — the walk puts it on the row — so the list has no reader, and a
    list of names with no reader is the next drift."""

    def test_the_predicate_is_not_reintroduced(self):
        import backend.inference.benchmark_identity as bi
        self.assertFalse(
            hasattr(bi, "is_weight_gemm_signature"),
            "the plan reads the row's own weight dtype; a second list of "
            "which signatures have one is the drift this fix removed")


if __name__ == "__main__":
    unittest.main()


class TestANumeratorBelongsToItsDeployment(unittest.TestCase):
    """A cell says what was measured; it does not say under which deployment.

    That is deliberate — a physical key is library- and model-agnostic so one
    model's measurement can serve another's identical cell — but the
    numerator is not. It depends on how many launches of the node the step
    bills and on how the deployment shards. On 2026-11-27 a 4090 row for
    Qwen3.8-2.4T-A95B (tp=256, count_per_step=69) was priced by an H100 job
    for Qwen3-Coder-Next (ep=64, count_per_step=36) because their cells
    matched; 276,824,320 / 12,582,912 = exactly 22.0, and the restated
    bandwidth came out at 1510% of the card's peak with nothing in the row
    or the job to say the two were not the same measurement.
    """

    def _job(self, **over):
        job = {"physical_key": ["silu_and_mul",
                                '{"B":1,"d_ffn":2048,"s_query":1}',
                                '{"activation":"bf16","output":"bf16"}',
                                "decode"],
               "flops_per_rank": 4096.0, "access_bytes_per_rank": 24576.0,
               "count_per_step": 36, "parallel": "tp_attention=4"}
        job.update(over)
        return job

    def _row(self, **over):
        row = {"op": "silu_and_mul", "phase": "decode",
               "shape_json": '{"B":1,"d_ffn":2048,"s_query":1}',
               "dtype_env_json": '{"activation":"bf16","output":"bf16"}',
               "p50_us": "10.0", "achieved_tflops": "0.1000",
               "achieved_bw_gbps": "1.0000",
               "count_per_step": "36", "parallel": "tp_attention=4"}
        row.update(over)
        return row

    def test_a_job_from_another_deployment_does_not_price_this_row(self):
        from kernel_bench.emit import restamp_rows

        for field, theirs in (("count_per_step", 69),
                              ("parallel", "tp_attention=256")):
            with self.subTest(field=field):
                result = restamp_rows([self._row()],
                                      [self._job(**{field: theirs})])
                self.assertEqual(result.changed, 0,
                                 "the row was restated by another "
                                 "deployment's numerator")
                self.assertEqual(len(result.unmatched), 1)
                self.assertIn(field, result.unmatched[0].reason)
                self.assertEqual(result.rows[0]["achieved_bw_gbps"], "1.0000")

    def test_the_same_deployment_still_prices_it(self):
        """The control: without this the test above would pass on a restamp
        that refused everything."""
        from kernel_bench.emit import restamp_rows

        result = restamp_rows([self._row()], [self._job()])
        self.assertEqual(result.unmatched, [])
        self.assertEqual(result.changed, 1)
        self.assertNotEqual(result.rows[0]["achieved_bw_gbps"], "1.0000")

    def test_a_plan_that_states_neither_field_still_prices_it(self):
        """An older plan carries neither field. Refusing those rows would
        retire measurements over a fact the plan that priced them never had."""
        from kernel_bench.emit import restamp_rows

        job = self._job()
        del job["count_per_step"], job["parallel"]
        result = restamp_rows([self._row()], [job])
        self.assertEqual(result.unmatched, [])
        self.assertEqual(result.changed, 1)
