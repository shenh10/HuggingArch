"""The timing harness names its device, and says which clock timed a row.

Before this, `timing.py` asked `torch.cuda.is_available()`: an Ascend box
answered no, tensors landed on the CPU, and the run produced wall-clock
rows wearing the card's SKU. These pin the three facts that prevent it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.benchmarks import TIMING_METHODS
from kernel_bench.device import DEVICES, NoDeviceBackend, backend_named
from kernel_bench.env import Env
from kernel_bench.providers import Provider
from kernel_bench.timing import _method


class TestEnvNamesOneDevice(unittest.TestCase):

    def test_cuda_available_is_derived_not_stored(self):
        # One stored fact. Two fields answering "which device" could differ,
        # and the difference is silent: a CPU number still looks like a row.
        self.assertTrue(Env(None, "cuda", None, None, None).cuda_available)
        self.assertFalse(Env(None, "npu", None, None, None).cuda_available)
        self.assertFalse(Env(None, None, None, None, None).cuda_available)

    def test_accelerated_asks_any_device_not_cuda(self):
        for device in DEVICES:
            self.assertTrue(Env(None, device, None, None, None).accelerated)
        self.assertFalse(Env(None, None, None, None, None).accelerated)

    def test_a_device_outside_the_vocabulary_is_refused(self):
        # The field was a boolean, so a stale `device=True` would read as
        # "some device" while no backend could serve it.
        for bad in (True, False, "gpu", "rocm"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Env(None, bad, None, None, None)


class TestKernelFloorsAreCudas(unittest.TestCase):

    def test_a_min_sm_provider_is_refused_off_cuda(self):
        # `sm` is CUDA's notion, so an unmet floor used to ADMIT off CUDA:
        # an Ascend box has no `sm`, so every CUDA-only kernel passed this
        # gate and only the import gate stood between it and a measurement.
        hopper = Provider(library="x", signature="gemm", kernel_impl="k",
                          import_name="x", build=lambda *a, **k: None,
                          min_sm="90")
        self.assertTrue(hopper._sm_ok(Env(None, "cuda", None, None, None,
                                          sm="sm_90")))
        self.assertFalse(hopper._sm_ok(Env(None, "cuda", None, None, None,
                                           sm="sm_89")))
        self.assertFalse(hopper._sm_ok(Env(None, "npu", None, None, None)))

    def test_a_provider_without_a_floor_is_judged_by_its_import(self):
        anywhere = Provider(library="x", signature="gemm", kernel_impl="k",
                            import_name="x", build=lambda *a, **k: None)
        self.assertTrue(anywhere._sm_ok(Env(None, "npu", None, None, None)))


class TestTimingMethodNamesTheGuard(unittest.TestCase):

    def test_every_method_this_module_can_record_is_in_the_vocabulary(self):
        sweep = object()
        for name in (_method(None, None, 0, None),
                     _method(None, sweep, 1, None),
                     _method(None, sweep, 0, True),
                     _method(None, sweep, 0, False)):
            self.assertIn(name, TIMING_METHODS, name)

    def test_a_spin_held_row_is_not_a_sweep_held_one(self):
        # Two devices reaching a number by different routes must not claim
        # the same method: a reader comparing an Ascend row with an NVIDIA
        # one has to see which kept the queue full.
        sweep = object()
        self.assertEqual(_method(None, sweep, 1_000, None), "l2_drain")
        self.assertEqual(_method(None, sweep, 0, True), "l2_drain_sweep_hold")
        self.assertEqual(_method(None, sweep, 0, False), "l2_drain_unheld")


class TestDeviceBackend(unittest.TestCase):

    def test_an_unknown_device_is_refused_not_guessed(self):
        with self.assertRaises(NoDeviceBackend):
            backend_named("rocm")

    def test_cuda_backend_names_its_tensor_device(self):
        self.assertEqual(backend_named("cuda").tensor_device, "cuda")


if __name__ == "__main__":
    unittest.main()
