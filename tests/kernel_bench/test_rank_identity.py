"""Rank identity, world form, align hook, NPU OOM and the per-rank oracle.

The harness ran one process, so every collective was the identity of one
buffer. These pin the facts a multi-process group will read: the rank is
on the invocation, the group size is a declared execution form, each
sample waits before its windows, an NPU out-of-memory releases the
device, and the oracle reconstructs every rank from `Invocation.seed_for_rank`
rather than mixing `seed + rank` inside `refs`.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def _env(**over):
    from kernel_bench.env import Env
    fields = dict(torch_version="2.11", device=None, device_name=None,
                  gpu_sku=None, peak_bw_gbps=None)
    fields.update(over)
    return Env(**fields)


def _comm(rank=0, world=4, n=8, signature="all_reduce"):
    from kernel_bench.invocation import Invocation
    dims = {"n_comm_elems": n, "n_comm_ranks": world, "n_comm_stride": 1}
    dtype_env = {"activation": "fp32", "output": "fp32"}
    canonical = {"sort_keys": True, "separators": (",", ":")}
    return Invocation(
        signature=signature, op=signature, dims=dims, dtype_env=dtype_env,
        phase="prefill", tp=1, rank=rank,
        physical_key=(signature, json.dumps(dims, **canonical),
                      json.dumps(dtype_env, **canonical), "prefill"))


class TestTheProcessKnowsItsRank(unittest.TestCase):
    def test_a_bare_env_is_a_single_process(self):
        env = _env()
        self.assertEqual(env.local_rank, 0)
        self.assertEqual(env.world_size, 1)

    def test_env_int_reads_the_torchrun_variables(self):
        import os
        from unittest.mock import patch
        from kernel_bench.env import _env_int
        with patch.dict(os.environ, {"LOCAL_RANK": "3", "WORLD_SIZE": "8"}):
            self.assertEqual(_env_int("LOCAL_RANK", 0), 3)
            self.assertEqual(_env_int("WORLD_SIZE", 1), 8)
        with patch.dict(os.environ, {"LOCAL_RANK": "nope"}):
            self.assertEqual(_env_int("LOCAL_RANK", 0), 0)

    def test_a_stated_rank_is_kept(self):
        env = _env(local_rank=3, world_size=8)
        self.assertEqual(env.local_rank, 3)
        self.assertEqual(env.world_size, 8)

    def test_the_device_backend_can_bind_a_die(self):
        from kernel_bench.device import DeviceBackend
        self.assertTrue(hasattr(DeviceBackend, "set_device"))
        self.assertTrue(hasattr(DeviceBackend, "device_count"))
        self.assertTrue(hasattr(DeviceBackend, "current_device"))

    def test_from_job_carries_rank(self):
        from kernel_bench.invocation import Invocation
        inv = Invocation.from_job({
            "op": "all_reduce", "signature": "all_reduce",
            "dims": {"n_comm_elems": 8, "n_comm_ranks": 2, "n_comm_stride": 1},
            "dtype_env": {"activation": "bf16"}, "phase": "prefill",
            "rank": 3,
        })
        self.assertEqual(inv.rank, 3)

    def test_a_job_without_rank_is_rank_zero(self):
        from kernel_bench.invocation import Invocation
        inv = Invocation.from_job({
            "op": "linear", "signature": "gemm",
            "dims": {"B": 1, "s_query": 4, "d_reduce": 8, "d_out": 8},
            "dtype_env": {"activation": "bf16"}, "phase": "prefill",
        })
        self.assertEqual(inv.rank, 0)


class TestRankSeedsAreTheInvocations(unittest.TestCase):
    def test_two_ranks_of_one_cell_draw_differently(self):
        a, b = _comm(rank=0), _comm(rank=1)
        self.assertEqual(a.input_seed, b.input_seed)
        self.assertNotEqual(a.rank_seed, b.rank_seed)
        self.assertEqual(a.seed_for_rank(1), b.rank_seed)
        self.assertEqual(a.at_rank(1).rank, 1)

    def test_refs_does_not_mix_seed_and_rank_itself(self):
        source = (PROJECT_ROOT / "kernel_bench" / "refs.py").read_text()
        self.assertIn("seed_for_rank", source)
        self.assertNotIn("input_seed +", source)
        self.assertNotIn("input_seed+", source)


class TestTheAlignHookRunsOutsideTheWindows(unittest.TestCase):
    def test_every_sample_aligns_once(self):
        from unittest.mock import patch
        from kernel_bench.timing import time_callable
        hits = []

        def align():
            hits.append(1)

        with patch("kernel_bench.timing.detect_backend", return_value=None):
            stats = time_callable(lambda: None, warmup=2, iters=5, align=align)
        self.assertEqual(len(hits), 5)
        self.assertEqual(stats.samples, 5)

    def test_no_align_is_the_old_clock(self):
        from unittest.mock import patch
        from kernel_bench.timing import time_callable
        with patch("kernel_bench.timing.detect_backend", return_value=None):
            stats = time_callable(lambda: None, warmup=1, iters=3)
        self.assertEqual(stats.samples, 3)
        self.assertEqual(stats.timing_method, "wall_clock")


class TestACollectiveNamesItsWorld(unittest.TestCase):
    def test_hccl_providers_declare_the_group_axis(self):
        from kernel_bench.providers import all_providers
        hccl = [p for p in all_providers()
                if p.kernel_impl.startswith("hccl_")]
        self.assertEqual(len(hccl), 4, [p.kernel_impl for p in hccl])
        for p in hccl:
            self.assertEqual(p.world, "n_comm_ranks", p.kernel_impl)
            self.assertFalse(p.disabled_reason, p.kernel_impl)

    def test_an_enabled_collective_without_world_is_refused(self):
        from kernel_bench.providers import Provider, _check_execution_world
        p = Provider(library="torch", signature="all_reduce",
                     kernel_impl="probe_all_reduce", import_name="torch",
                     build=lambda *a, **k: None)
        with self.assertRaises(ValueError) as caught:
            _check_execution_world(p)
        self.assertIn("world", str(caught.exception))

    def test_a_world_axis_must_belong_to_the_signature(self):
        from kernel_bench.providers import Provider, _check_execution_world
        p = Provider(library="torch", signature="all_reduce",
                     kernel_impl="probe_bad_world", import_name="torch",
                     build=lambda *a, **k: None, world="B")
        with self.assertRaises(ValueError) as caught:
            _check_execution_world(p)
        self.assertIn("structural axis", str(caught.exception))

    def test_a_named_world_on_a_disabled_kernel_is_kept(self):
        from kernel_bench.providers import Provider, _check_execution_world
        p = Provider(library="torch", signature="all_reduce",
                     kernel_impl="probe_named", import_name="torch",
                     build=lambda *a, **k: None, world="n_comm_ranks",
                     disabled_reason="one process")
        _check_execution_world(p)


class TestTheOracleReconstructsEveryRank(unittest.TestCase):
    def test_all_reduce_is_the_sum_across_ranks(self):
        pytest.importorskip("torch")
        import torch
        from kernel_bench.refs import get_ref, input_generator
        spec = get_ref("all_reduce")
        self.assertIsNotNone(spec.reference)
        payloads = []
        for rank in range(4):
            inv = _comm(rank=rank)
            got = spec.make_inputs(inv, "cpu", input_generator(inv, "cpu"))
            payloads.append(got["payload"].float())
        self.assertFalse(torch.equal(payloads[0], payloads[1]))
        inv0 = _comm(rank=0)
        inputs = spec.make_inputs(inv0, "cpu", input_generator(inv0, "cpu"))
        out = spec.reference(inputs).float()
        self.assertTrue(torch.allclose(out, sum(payloads)))

    def test_two_ranks_hold_different_payloads_of_the_same_cell(self):
        pytest.importorskip("torch")
        import torch
        from kernel_bench.refs import get_ref, input_generator
        spec = get_ref("all_reduce")
        a = spec.make_inputs(_comm(0), "cpu",
                             input_generator(_comm(0), "cpu"))["payload"]
        b = spec.make_inputs(_comm(1), "cpu",
                             input_generator(_comm(1), "cpu"))["payload"]
        self.assertFalse(torch.equal(a, b))
        self.assertEqual(tuple(a.shape), tuple(b.shape))

    def test_all_gather_concatenates_shards(self):
        pytest.importorskip("torch")
        import torch
        from kernel_bench.refs import get_ref, input_generator
        spec = get_ref("all_gather")
        world, n = 4, 16
        shards = []
        for rank in range(world):
            inv = _comm(rank=rank, world=world, n=n, signature="all_gather")
            payload = spec.make_inputs(
                inv, "cpu", input_generator(inv, "cpu"))["payload"]
            chunk = n // world
            shards.append(payload.reshape(-1)[rank * chunk:(rank + 1) * chunk])
        inv0 = _comm(rank=0, world=world, n=n, signature="all_gather")
        inputs = spec.make_inputs(inv0, "cpu", input_generator(inv0, "cpu"))
        out = spec.reference(inputs)
        self.assertTrue(torch.equal(out.reshape(-1), torch.cat(shards)))


class TestNpuOomReleasesTheDevice(unittest.TestCase):
    def test_an_npu_out_of_memory_is_the_card_refusing(self):
        from kernel_bench.driver import _is_out_of_memory
        npu = _env(device="npu", device_name="Ascend910C", gpu_sku="910C-A3")
        self.assertTrue(_is_out_of_memory(RuntimeError("NPU out of memory"), npu))
        cpu = _env(device=None, device_name="cpu")
        self.assertFalse(_is_out_of_memory(RuntimeError("out of memory"), cpu))

    def test_release_on_a_host_box_is_a_no_op(self):
        from kernel_bench.driver import _release_device
        _release_device(_env(device=None))

    def test_fits_after_release_asks_the_device_backend(self):
        from unittest import mock
        import torch
        from kernel_bench.driver import _fits_after_release
        gpu = _env(device="cuda", device_name="H100", gpu_sku="H100-SXM")
        info = (41 << 30, 79 << 30)
        error = RuntimeError(
            "CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a "
            "total capacity of 79.18 GiB of which 38.35 GiB is free.")
        with mock.patch.object(torch.cuda, "synchronize"), \
                mock.patch.object(torch.cuda, "empty_cache"), \
                mock.patch.object(torch.cuda, "mem_get_info",
                                  return_value=info):
            self.assertTrue(_fits_after_release(gpu, error))
