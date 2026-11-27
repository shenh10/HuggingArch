"""The tensors a reference builds are the cell the row bills.

`cell_check` compares builders against the signature contract; these tests
pin the two contract facts that used to be restated (or missed) in builders:
which variant axes select a geometry, and which dtype slot each operand reads.
"""
from __future__ import annotations

import os
import json
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.inference.op_signatures import get_signature
from kernel_bench.cell_check import complaints
from kernel_bench.invocation import Invocation
from kernel_bench.refs import get_ref, input_generator


class TestVariantSelectors(unittest.TestCase):
    def test_a_mask_flag_does_not_select_gqas_narrow_v(self):
        sig = get_signature("gqa_attention")
        windowed = {"s_window": 512}
        self.assertFalse(sig.binds_variant(windowed))
        self.assertIs(sig.layouts_for(windowed), sig.inputs)

    def test_the_v_width_selects_gqas_narrow_v_without_the_masks(self):
        sig = get_signature("gqa_attention")
        narrow = {"d_v_head": 128}
        self.assertTrue(sig.binds_variant(narrow))
        self.assertIs(sig.layouts_for(narrow), sig.variant_inputs)
        self.assertEqual(
            sig.layouts_for(narrow)[2].extents(
                {"B": 1, "s_kv": 1024, "n_kv_heads": 1, "d_v_head": 128}),
            (1, 1024, 1, 128))


class TestBuiltOperandSlots(unittest.TestCase):
    def _build(self, inv):
        return get_ref(inv.signature).make_inputs(
            inv, "cpu", input_generator(inv, "cpu"))

    def test_a_router_weight_is_built_at_its_own_slot(self):
        inv = Invocation(
            signature="gemm", op="linear", phase="prefill", tp=1,
            dims={"B": 1, "s_query": 8, "d_out": 4, "d_reduce": 6},
            dtype_env={"activation": "bf16", "weight": "fp32",
                       "output": "bf16"})
        inv = replace(inv, physical_key=("gemm", "{}", "{}", "prefill"))
        inputs = self._build(inv)
        self.assertEqual(str(inputs["x"].dtype), "torch.bfloat16")
        self.assertEqual(str(inputs["w"].dtype), "torch.float32")
        self.assertEqual(complaints(inv, inputs), [])

    def test_narrow_v_and_the_kvs_own_dtype_come_from_one_contract(self):
        inv = Invocation(
            signature="gqa_attention", op="flash_attention",
            phase="prefill", tp=1,
            dims={"B": 1, "s_query": 8, "s_kv": 8, "n_heads": 2,
                  "n_kv_heads": 1, "d_head": 4, "d_v_head": 2,
                  "kv_block_size": 16},
            dtype_env={"activation": "bf16", "kv": "fp32",
                       "output": "bf16"})
        inv = replace(inv, physical_key=(
            "gqa_attention", json.dumps(inv.dims, sort_keys=True),
            json.dumps(inv.dtype_env, sort_keys=True), "prefill"))
        inputs = self._build(inv)
        self.assertEqual(tuple(inputs["q"].shape), (1, 8, 2, 4))
        self.assertEqual(tuple(inputs["k"].shape), (1, 8, 1, 4))
        self.assertEqual(tuple(inputs["v"].shape), (1, 8, 1, 2))
        self.assertEqual(str(inputs["k"].dtype), "torch.float32")
        self.assertEqual(str(inputs["v"].dtype), "torch.float32")
        self.assertEqual(complaints(inv, inputs), [])


class TestTorchAsymmetricGqaFloor(unittest.TestCase):
    def test_the_fallback_keeps_the_launch_at_the_cells_dtype(self):
        import torch.nn.functional as F
        from kernel_bench.env import Env
        from kernel_bench.launch_trace import disagreement, record
        from kernel_bench.providers import all_providers

        inv = Invocation(
            signature="gqa_attention", op="flash_attention",
            phase="prefill", tp=1,
            dims={"B": 1, "s_query": 8, "s_kv": 8, "n_heads": 2,
                  "n_kv_heads": 1, "d_head": 4, "d_v_head": 2,
                  "kv_block_size": 16},
            dtype_env={"activation": "bf16", "kv": "bf16",
                       "output": "bf16"},
            physical_key=("gqa_attention", "{}", "{}", "prefill"))
        inputs = get_ref("gqa_attention").make_inputs(
            inv, "cpu", input_generator(inv, "cpu"))
        provider = next(p for p in all_providers()
                        if p.kernel_impl == "torch_sdpa")
        launched = provider.convert_inputs(inv, inputs)
        case = provider.build(
            inv, Env("2.11", None, None, None, None,
                     libs={"torch": "2.11"}), launched)
        with record() as traced:
            out = case.run()
        canonical = provider.wrap_output(inv, case.to_canonical)(out)
        self.assertEqual(tuple(canonical.shape), (1, 8, 2, 2))
        self.assertEqual(str(canonical.dtype), "torch.bfloat16")
        reference = get_ref("gqa_attention").reference(inputs).flatten()
        cosine = F.cosine_similarity(
            canonical.flatten().float(), reference.float(), dim=0).item()
        self.assertGreater(cosine, 0.999)
        self.assertIsNone(disagreement(
            traced.trace,
            declared_dtypes=(inv.dtype, inv.kv_dtype),
            declared_elements=tuple(
                tensor.numel() for tensor in
                (inputs["q"], inputs["k"], inputs["v"]))))


if __name__ == "__main__":
    unittest.main()
