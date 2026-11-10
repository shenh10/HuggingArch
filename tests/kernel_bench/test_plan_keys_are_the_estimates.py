"""The write side asks the key the read side asks with.

A measurement is useful only if the key it is stored under is the key an
estimate looks it up by. Both are stated by one derivation
(`calibration_overlay.point_keys`), and this is the gate over that claim: for
one (model, batch, context, phase, layout), every job the Kernel Bench plan
exports carries the measurement key the estimate's own overlay states for the
same node — signature, projected shape, dtype environment and phase, the four
fields that are not the run's card.

It is a real gate rather than an AST scan because the previous seam passed a
scan: the plan called `point_keys` and then rebuilt the dtype half from the
spec's quant scheme, which agreed on a BF16 model and disagreed on every row
of a quantized one (gpt-oss: `metadata` read `fp32` where the walk says
`ue8m0`; DSv4-Flash: an FP4 expert's activation read `bf16` where the walk
says `fp8_e4m3`). Those rows would be measured under a key no reader asks
with.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import pipeline
from backend.inference.calibration_overlay import (
    WorkloadShape,
    build_deployment_context,
    point_keys,
)
from backend.inference.framework_bindings import DEFAULT_FRAMEWORK, get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import phase_binding
from backend.arch.eval.analysis import Point
from backend.inference.capacity import (
    CapacityKnobs,
    _capacity_pending,
    _capacity_preparation,
    _point_rows,
)
from backend.api.estimate_env import resolve_estimate_environment
from backend.api._models import InferenceEstimateRequest
from kernel_bench.plan import build_jobs
from tests.snapshots import resolve_snapshot

BATCH = 8
SEQ = 2048


def _estimate_keys(model: str, phase: str) -> dict:
    """What the estimate's overlay asks for, per node: the walk at the same
    point, keyed by the one derivation (`calibration_overlay.point_keys`,
    which is what `CalibrationOverlay.apply` calls)."""
    resolved = resolve_snapshot(model)
    binding = get_binding(DEFAULT_FRAMEWORK)
    bind = phase_binding(phase=phase, avg_input_tokens=SEQ, avg_output_tokens=0)
    walk = pipeline.analyze(
        resolved, parallel=ParallelConfig(), framework_binding=binding,
        batch_size=BATCH,
        avg_input_tokens=SEQ, avg_output_tokens=0,
        phase=phase,
        include_aux_timing=True, breakdown=False)
    points = point_keys(
        walk, walk.per_op_timing,
        WorkloadShape.of_walk(walk, seq_len=max(1, int(bind["s_query"]))),
        build_deployment_context(framework=binding.framework,
                                 framework_binding=binding,
                                 parallel=ParallelConfig(), gpu_sku="", phase=phase),
        model_id=model, spec_hash="x")
    return {p.aggregation_id: p for p in points if p.key is not None}


@pytest.mark.regression          # walks three snapshots
class TestTheTwoSidesAskOneKey(unittest.TestCase):
    #: A BF16 model, an FP8 one (block scales), an FP4 one (packed banks):
    #: the three dtype environments the two derivations used to disagree on.
    MODELS = (("meta-llama/Llama-3.2-3B", "decode"),
              ("deepseek-ai/DeepSeek-V3", "prefill"),
              ("openai/gpt-oss-20b", "decode"))

    def test_every_job_carries_the_key_the_estimate_looks_up(self):
        for model, phase in self.MODELS:
            with self.subTest(model=model):
                jobs, _dropped = build_jobs(avg_output_tokens=0,
                    model_id=model, batch_size=BATCH, seq_len=SEQ, phase=phase,
                    resolved=resolve_snapshot(model), parallel=ParallelConfig())
                asked = _estimate_keys(model, phase)
                self.assertTrue(jobs)
                for job in jobs:
                    node = job["node_spec_id"]
                    point = asked.get(node)
                    self.assertIsNotNone(
                        point, f"{model}: the plan exports {node} and the "
                        f"estimate states no key for it")
                    self.assertEqual(
                        [point.key.signature, point.key.shape_json,
                         point.key.dtype_env_json, point.key.phase],
                        job["physical_key"],
                        f"{model} [{node}]: the job's key is not the one the "
                        f"estimate asks with")

    def test_the_job_shape_and_dtype_env_are_that_key(self):
        """The job's own `dims` / `dtype_env` are the key's, so a provider
        builds the kernel the key names."""
        for model, phase in self.MODELS:
            with self.subTest(model=model):
                jobs, _dropped = build_jobs(avg_output_tokens=0,
                    model_id=model, batch_size=BATCH, seq_len=SEQ, phase=phase,
                    resolved=resolve_snapshot(model), parallel=ParallelConfig())
                asked = _estimate_keys(model, phase)
                for job in jobs:
                    point = asked[job["node_spec_id"]]
                    self.assertEqual(job["dtype_env"],
                                     json.loads(point.dtype_env_json))
                    self.assertEqual(job["dims"], json.loads(point.shape_json))

    def test_a_speculative_plan_carries_verify_and_draft_keys(self):
        """K's MTP forward is walked at s_query=1, not at the K+1 verify."""
        from kernel_bench.plan import PlanPoint, build_jobs_with_unplaced
        model = "Qwen/Qwen3.5-35B-A3B"
        resolved = resolve_snapshot(model)
        jobs, _dropped, _records = build_jobs_with_unplaced(
            model_id=model, phase="decode", avg_output_tokens=0,
            points=[PlanPoint(SEQ, 0, 2, BATCH)],
            resolved=resolved, parallel=ParallelConfig(), gpu="H100-SXM")
        request = InferenceEstimateRequest(
            model_id=model, gpu_sku="H100-SXM", avg_input_tokens=SEQ,
            avg_output_tokens=0, decode_batch_size=BATCH,
            speculative_draft_tokens=1, speculative_draft="recursive",
            selection_mode="theoretical_only")
        environment = resolve_estimate_environment(request)
        knobs = CapacityKnobs.from_kwargs(environment, envelope=frozenset({
            "config", "gpu_sku", "avg_output_tokens", "prefill_batch_size",
            "decode_batch_size", "prefill_chunk_size"}))
        preparation = _capacity_preparation(
            config=environment["config"], gpu_sku="H100-SXM", knobs=knobs)
        pending = _capacity_pending(
            preparation, avg_input_tokens=SEQ, avg_output_tokens=0,
            prefill_batch_override=1, decode_batch_override=BATCH,
            breakdown=False, fold_prefill_chunks=False, prefill_capacity=False)
        answer = pending.finish()
        moved = _point_rows(
            answer, preparation,
            {"per_op_timing_decode": BATCH, "per_op_timing_draft": BATCH},
            pending, Point({}), breakdown=False, costs=False)
        expected = set()
        for name, walk in pending.walk_of_rows.items():
            if name not in ("per_op_timing_decode", "per_op_timing_draft"):
                continue
            rows = moved[name]
            binding = preparation.framework_binding
            for point in point_keys(
                    walk.analysis, rows, walk.workload,
                    build_deployment_context(
                        framework=binding.framework, framework_binding=binding,
                        parallel=ParallelConfig(), gpu_sku="", phase="decode"),
                    model_id=model, spec_hash=resolved.identity_hash()):
                if point.key is not None:
                    expected.add((point.key.signature, point.key.shape_json,
                                  point.key.dtype_env_json, point.key.phase,
                                  point.aggregation_id))
        got = {(*job["physical_key"], job["node_spec_id"]) for job in jobs}
        self.assertTrue(expected)
        self.assertEqual(got, expected)

    def test_a_declined_row_is_not_planned(self):
        """A leaf whose own dtype the corpus has no token for is keyed by the
        STEP's environment, which is not the leaf's — so the estimate
        declines it and the plan must not export it. A row measured under a
        key nobody asks with is GPU time spent twice.

        No leaf of the corpus is declined today, so the decline is stated
        here on one leaf, where the plan reads it
        (`RowIdentity.leaf_dtype_env`). The two that used to be: gpt-oss's
        expert banks, once read as their `U8` containers and now keyed by
        their format; and DeepSeek-V4's hash router, one fused launch
        (`moe_topk_hash_sqrtsoftplus`) whose I64 `tid2eid` is the width the
        CHECKPOINT ships and not the one either framework loads — see
        `RowIdentity.leaf_dtype_env` for why a corpus token for I64 is not
        what that leaf is waiting for."""
        from unittest import mock
        from backend.inference.benchmark_identity import RowIdentity
        model, target = "deepseek-ai/DeepSeek-V4-Flash", "ffn.gate.weight"
        real = RowIdentity.leaf_dtype_env

        def declining(identity, row):
            env, why = real(identity, row)
            if str(row.get("aggregation_id", "")).endswith(target):
                return env, "the leaf's dtype environment has no corpus token for weight=I64"
            return env, why

        with mock.patch.object(RowIdentity, "leaf_dtype_env", declining):
            jobs, dropped = build_jobs(avg_output_tokens=0,
                model_id=model, batch_size=BATCH, seq_len=SEQ, phase="decode",
                resolved=resolve_snapshot(model), parallel=ParallelConfig())
        declined = [d for d in dropped if "no corpus token" in d]
        self.assertTrue(declined, "the declined leaf is reported as dropped")
        # The hash router's fused launch carries the same phrase for its
        # I64 tid2eid (the known case the docstring above explains), so the
        # names asserted here are the mocked target's alone.
        known = "ffn.topk_scale__dsv4_moe_hash"
        named = {d.split("[", 1)[1].split("]", 1)[0] for d in declined
                 if not d.split("[", 1)[1].split("]", 1)[0].endswith(known)}
        self.assertTrue(all(n.endswith(target) for n in named), named)
        self.assertFalse(named & {job["node_spec_id"] for job in jobs})

        # And a packed format is no longer one of them.
        oss, oss_dropped = build_jobs(avg_output_tokens=0,
            model_id="openai/gpt-oss-20b", batch_size=BATCH, seq_len=SEQ,
            phase="decode", resolved=resolve_snapshot("openai/gpt-oss-20b"),
            parallel=ParallelConfig())
        self.assertEqual([d for d in oss_dropped if "no corpus token" in d], [])
        self.assertEqual(
            {job["dtype_env"]["weight"] for job in oss
             if job["signature"] == "grouped_gemm"}, {"mxfp4"})


if __name__ == "__main__":
    unittest.main()
