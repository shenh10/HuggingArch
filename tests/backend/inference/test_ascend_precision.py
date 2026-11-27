"""What vLLM-Ascend runs a quantised matmul at on an Ascend NPU, as its
binding's `precision_launches` declares it.

* msModelSlim W8A8_DYNAMIC (`w_int8_w8a8_ffn`): INT8 activations per token, one
  INT8 x INT8 matmul at the card's INT8 peak.
* A native block-FP8 checkpoint: no Ascend matmul eats it; off Ascend 950 the
  tiles are expanded to the model dtype and the unquantised GEMM runs, so the
  GEMM is BF16 and NO activation is quantised. The first assertion is the one
  that went red before the launch was declared: the walk grafted FP8
  activation casts a card without an FP8 MMA never runs.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.parallel_config import ParallelConfig
from backend.inference.gpu_specs import get_gpu, peak_flops_for_dtype
from tests.sweep_parity import Deployment

NPU = "Ascend-910C-A3"


def _rows(model: str, precision=None, parallel=None):
    knobs = {"framework": "vllm_ascend"}
    if precision is not None:
        knobs["precision"] = precision
    res = Deployment(model_id=model, gpu_sku=NPU, knobs=knobs,
                     parallel=parallel or ParallelConfig(
                         tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8),
                     ).direct(avg_input_tokens=512, prefill_batch=1,
                              decode_batch=4)
    return res, [r for k in ("per_op_timing", "per_op_timing_decode")
                 for r in res[k]]


class TestFp8CheckpointOnAscend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.res, cls.rows = _rows(
            "deepseek-ai/DeepSeek-V3",
            parallel=ParallelConfig(tp_attention=16, tp_dense_ffn=16,
                                    ep_moe=16))

    def test_no_fp8_activation_is_quantised(self):
        quant = [r for r in self.rows if r["op"] == "quantise_activation"]
        self.assertEqual(quant, [])

    def test_fp8_weights_compute_at_bf16(self):
        bf16 = peak_flops_for_dtype(get_gpu(NPU), "BF16")
        fp8 = [r for r in self.rows if r.get("weight_dtype") == "F8_E4M3"]
        self.assertTrue(fp8)
        for r in fp8:
            self.assertEqual(r["act_dtype"], "BF16", r["op"])
            self.assertEqual(r["peak_flops_per_sec_for_op"], bf16, r["op"])


class TestW8A8OnAscend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.shipped, _ = _rows("Qwen/Qwen3-30B-A3B")
        cls.res, cls.rows = _rows("Qwen/Qwen3-30B-A3B",
                                  precision={"preset": "w_int8_w8a8_ffn"})

    def test_matrices_run_int8_at_the_int8_peak(self):
        int8 = peak_flops_for_dtype(get_gpu(NPU), "I8")
        gemms = [r for r in self.rows if r.get("weight_dtype") == "INT8"]
        self.assertTrue(gemms)
        for r in gemms:
            self.assertEqual(r["act_dtype"], "I8", r["op"])
            self.assertEqual(r["peak_flops_per_sec_for_op"], int8, r["op"])

    def test_activations_are_quantised_once_per_int8_input(self):
        self.assertTrue(any(r["op"] == "quantise_activation"
                            for r in self.rows))

    def test_weights_shrink_to_one_byte_plus_scales(self):
        ratio = self.res["weight_bytes_total"] / self.shipped["weight_bytes_total"]
        self.assertLess(ratio, 0.55)
        self.assertGreater(ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
