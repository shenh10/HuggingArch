"""A reader's row is its own terms plus the borrowed ones, each once.

gpt-oss's expert GEMM reads its own `[E, out, in]` weight and the bias
threaded into it, one packed tensor it borrows. `borrowed_weight_reads` and
`borrowed_metadata_reads` are their own terms so a borrowed tensor's bytes
— its scales included — land on the reader's row exactly once.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.access import _leaf_traffic_rows, _scale_dtype
from backend.arch.eval.storage import StoredTensor
from backend.arch.eval.traffic import Traffic, widest_format


class TestBorrowedTermsAddOnce(unittest.TestCase):
    def test_metadata_and_weight_terms(self):
        uid = 7
        from backend.arch.ops import OP_REGISTRY
        node = SimpleNamespace(uid=uid, children=(), role="ffn", op="linear",
                               is_component=False, inputs_obj=(),
                               declaration=OP_REGISTRY["linear"])
        program = SimpleNamespace(by_uid=lambda u: node if u == uid else None,
                                  stack_of=lambda n: "",
                                  # The row's dtype is read off the tensors the
                                  # leaf streams, which the program names
                                  # (`traffic.streamed_sources`): this stub owns
                                  # its one tensor and aliases nothing.
                                  read_aliases={}, nodes=lambda: (node,))
        precision = SimpleNamespace(
            edge_bytes=lambda role, stack: 2.0,
            compute_dtype=lambda role, stack, formats, operand=None: "BF16",
            required_format=lambda role, stack, formats: None,
            # A BF16 leaf needs the one column it computes in: nothing here
            # quantises an activation operand, so there is no narrower
            # operand to need a second MMA family for.
            mma_dtypes=lambda role, stack, formats, operand=None: ("BF16",),
            launch_promotion_ladder=lambda role, stack, formats: False,
            launch_gaps=lambda role, stack, formats: (),
            launch_gap=lambda role, stack, formats: None,
            compute_dtype_is_chosen=lambda role, stack, formats: False,
            conversion_label=lambda role, stack, formats: "",
            edge_dtype=lambda role, stack: "BF16")
        analysis = SimpleNamespace(
            cost=SimpleNamespace(values={}),
            shapes=SimpleNamespace(values={}),
            formats=SimpleNamespace(values={}),
            resolved_precision=precision,
            program=program,
            local_tensors=lambda: iter(()))
        layer = SimpleNamespace(values={uid: Traffic(
            weight_reads=100, metadata_reads=10,
            borrowed_weight_reads=7, borrowed_metadata_reads=3)})
        rows = _leaf_traffic_rows(
            analysis, traffic_layer=layer,
            program=program)
        self.assertEqual(rows[uid].weight, 100 + 7)
        self.assertEqual(rows[uid].metadata, 10 + 3)
        # A GEMM's own matrix is a weight of its own.
        self.assertEqual(rows[uid].weight_packs, ((uid,),))


class TestScaleDtypeDescribesTheWidestTensor(unittest.TestCase):
    """A fused span's row reports one tensor: the weight dtype, its
    granularity and its scale dtype are all the widest member's. The first
    scale met among the members paired a BF16 weight with an FP8 member's
    scale, and the members' order decided which."""

    @staticmethod
    def _rep(dtype, scale=None):
        # The scale is a real `StoredTensor`: `_scale_dtype` reads what the
        # slot holds (`logical_dtype`) before the slot (`dtype`) since
        # 96c2108e3, and a hand-rolled stub without that field stopped being
        # the thing it stood for.
        metadata = (() if scale is None else
                    (StoredTensor(module="m", suffix=".scale", dtype=scale,
                                  sibling_kind="scale"),))
        return SimpleNamespace(resident=SimpleNamespace(
            logical_dtype=dtype, metadata=metadata))

    def test_order_does_not_decide(self):
        bf16, fp8 = self._rep("BF16"), self._rep("F8_E4M3", scale="F32")
        for reps in ((bf16, fp8), (fp8, bf16)):
            self.assertIsNone(_scale_dtype(widest_format(reps)))

    def test_the_widest_tensors_own_scale(self):
        wide = self._rep("BF16", scale="BF16")
        narrow = self._rep("F8_E4M3", scale="F32")
        for reps in ((wide, narrow), (narrow, wide)):
            self.assertEqual(_scale_dtype(widest_format(reps)), "BF16")

    def test_the_scale_is_what_its_slot_holds(self):
        # MXFP4's UE8M0 exponent ships in a U8 slot; the scale's dtype is the
        # exponent, not the container.
        scale = StoredTensor(module="m", suffix="_scales", dtype="U8",
                             logical_dtype="F8_E8M0FNU", sibling_kind="scale")
        rep = SimpleNamespace(resident=SimpleNamespace(
            logical_dtype="MXFP4", metadata=(scale,)))
        self.assertEqual(_scale_dtype(widest_format((rep,))), "F8_E8M0FNU")


if __name__ == "__main__":
    unittest.main()
