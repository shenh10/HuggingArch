"""A framework runs on the devices its binding declares (`devices:`), and a
deployment on any other device is refused where a binding meets a card
(`FrameworkBinding.deployed`).

The first case is the one that goes red if the refusal is lost: vLLM on an
Ascend NPU used to be answered, pricing CUDA fusions on a card that cannot
launch them, with nothing failing.
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.framework_bindings import (
    FRAMEWORK_BINDINGS, FrameworkNotOnDevice, ServingDeployment,
    _load_framework_bindings_dir, get_binding)
from backend.inference.gpu_specs import load_gpu_specs

NPU_CARD = "Ascend-910C-A3"
CUDA_CARD = "H100-SXM"


def _at(sku: str) -> ServingDeployment:
    return ServingDeployment.at(gpu_sku=sku, activation_dtype="BF16")


class TestDeployRefusesOtherDevices(unittest.TestCase):

    def test_cuda_framework_on_npu_card_is_refused(self):
        with self.assertRaises(FrameworkNotOnDevice) as ctx:
            get_binding("vllm").deployed(_at(NPU_CARD))
        self.assertIn("npu", str(ctx.exception))
        # A ValueError, so the API answers it 400.
        self.assertIsInstance(ctx.exception, ValueError)

    def test_npu_framework_on_cuda_card_is_refused(self):
        with self.assertRaises(FrameworkNotOnDevice):
            get_binding("vllm_ascend").deployed(_at(CUDA_CARD))

    def test_each_framework_deploys_on_its_own_devices(self):
        for name, binding in FRAMEWORK_BINDINGS.items():
            for card in load_gpu_specs():
                if card.device in binding.devices:
                    binding.deployed(_at(card.sku))  # no raise
                else:
                    with self.assertRaises(FrameworkNotOnDevice,
                                           msg=f"{name} on {card.sku}"):
                        binding.deployed(_at(card.sku))

    def test_unknown_device_is_not_refused(self):
        # No card described: the registry's own undeployed state.
        get_binding("vllm").deployed(ServingDeployment())


class TestShippedBindingsDeclareDevices(unittest.TestCase):

    def test_every_binding_declares_sheet_devices(self):
        sheet = {g.device for g in load_gpu_specs()}
        for name, binding in FRAMEWORK_BINDINGS.items():
            self.assertTrue(binding.devices, name)
            self.assertLessEqual(set(binding.devices), sheet, name)


_MINIMAL = """\
framework: {name}
audited_at: {{ repo: a/b, tag: v1, commit: "{commit}" }}
{devices}
kv_block_size: 16
ssm_chunk_size: {{ from_config: [chunk_size] }}
{extra}
"""


class TestLoaderRefuses(unittest.TestCase):

    def _load(self, devices: str, extra: str = ""):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "fw.yaml"), "w") as f:
                f.write(_MINIMAL.format(name="fw", commit="0" * 40,
                                        devices=devices,
                                        extra=textwrap.dedent(extra)))
            return _load_framework_bindings_dir(d)

    def test_accepts_a_declared_sheet_device(self):
        self.assertEqual(self._load("devices: [npu]")["fw"].devices, ("npu",))

    def test_missing_devices_is_refused(self):
        with self.assertRaisesRegex(ValueError, "devices"):
            self._load("")

    def test_device_no_card_has_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no row of gpu_specs.csv"):
            self._load("devices: [tpu]")

    def test_entry_device_outside_devices_is_refused(self):
        extra = """\
        op_implementations:
          fused_add_rmsnorm:
            source_anchor: "x/y.py:1"
            when: { device: [cuda] }
        """
        with self.assertRaisesRegex(ValueError, "outside the framework"):
            self._load("devices: [npu]", extra)


class TestEstimateOnNpuCard(unittest.TestCase):
    """End to end through `estimate_capacity`, the path a request takes."""

    def _estimate(self, framework: str):
        from backend.arch.layout.parallel_config import ParallelConfig
        from tests.sweep_parity import Deployment
        return Deployment(
            model_id="meta-llama/Llama-3.2-3B",
            parallel=ParallelConfig(tp_attention=8, tp_dense_ffn=8),
            gpu_sku=NPU_CARD, knobs={"framework": framework},
        ).direct(avg_input_tokens=512, prefill_batch=1, decode_batch=4)

    def test_cuda_framework_estimate_is_refused(self):
        with self.assertRaises(FrameworkNotOnDevice):
            self._estimate("vllm")

    def test_npu_framework_prices_only_its_own_fusions(self):
        # Every fused row on the NPU card is a fusion vLLM-Ascend declares;
        # none of vLLM's CUDA-only kernels reaches it.
        from backend.inference.fusion_plans import FUSION_SCHEMES
        own = set(get_binding("vllm_ascend").fused_ops())
        cuda_only = set(get_binding("vllm").fused_ops()) - own
        res = self._estimate("vllm_ascend")
        for phase in ("per_op_timing", "per_op_timing_decode"):
            rows = res[phase]
            self.assertTrue(rows, phase)
            fused = {r["op"] for r in rows if r["op"] in FUSION_SCHEMES}
            self.assertTrue(fused, phase)
            self.assertLessEqual(fused, own, phase)
            self.assertFalse(fused & cuda_only, phase)


if __name__ == "__main__":
    unittest.main()
