"""An exported axis carries a JSON value, whatever the binding made of it.

`as_wire` is the boundary between the tracked walk and every consumer that
reads shapes without rebuilding identity — the per-op rows the API returns,
and the kernel_bench plan. Past it there is no sympy, so the number has to
become an int, or a string when it is not one.

Not-one is the ordinary MoE case, not an exotic one: dispatching a decode
step's single token over `n_experts` leaves the expert bank
`n_experts_active*s_ctx/n_experts` of a token — `1/32` on DeepSeek-V3.1.
A `Rational` that reaches FastAPI's encoder is a 500 for the whole request,
so this is the difference between the inference page working for a MoE model
and not working at all.
"""
from __future__ import annotations

import json
import sys
import unittest

from tests.snapshots import snapshot_config
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import as_extent, as_number, as_wire
from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig

# The corpus's cheapest model with a fractional expert-token axis.
MOE = (PROJECT_ROOT / "backend" / "arch" / "snapshots"
       / "Qwen__Qwen3-30B-A3B.json")


def _resolved_moe():
    """Resolve the spec once so capacity never re-matches (single entry)."""
    from tests.snapshots import shared_loader
    return shared_loader().match(
        json.loads(MOE.read_text())["config"],
        model_id=MOE.stem.replace("__", "/", 1))


class TestAsWireIsJsonNative(unittest.TestCase):

    def test_an_integral_extent_is_an_int(self):
        import sympy as sp
        wire = as_wire(sp.Integer(4096) / sp.Integer(8))
        self.assertEqual(wire["value"], 512)
        self.assertIsInstance(wire["value"], int)

    def test_a_fractional_extent_is_its_exact_string(self):
        """`int()` succeeds on a `Rational` and simply disagrees with it, so
        an is-it-integral test that only guards against `int()` RAISING lets
        the object straight through."""
        import sympy as sp
        wire = as_wire(sp.Rational(1, 32))
        self.assertEqual(wire["value"], "1/32")

    def test_an_unbound_extent_is_its_name(self):
        import sympy as sp
        self.assertEqual(as_wire(sp.Symbol("s_kv"))["value"], "s_kv")


class TestAnExtentNamesItsRounding(unittest.TestCase):
    """`as_extent` is the other reading of the same axis: a whole geometry.

    It exists because the conversion it replaces was a bare `int()`, which
    both picks a rounding and hides that a rounding was picked. The exact
    value and the allocated geometry are different questions about one axis,
    and a fractional `M` is where they visibly disagree.
    """

    def test_a_fractional_average_floors_to_an_empty_bank(self):
        import sympy as sp
        self.assertEqual(as_extent(sp.Rational(1, 32), rounding="floor"), 0)

    def test_the_same_axis_ceils_to_a_row(self):
        import sympy as sp
        self.assertEqual(as_extent(sp.Rational(1, 32), rounding="ceil"), 1)

    def test_an_unnamed_convention_is_refused(self):
        with self.assertRaises(ValueError):
            as_extent(4, rounding="nearest")

    def test_an_unbound_axis_has_no_geometry(self):
        """A symbol is a legitimate `as_number` answer and not a legitimate
        extent: nothing can be allocated for it."""
        import sympy as sp
        self.assertEqual(as_number(sp.Symbol("s_kv")), "s_kv")
        with self.assertRaises(TypeError):
            as_extent(sp.Symbol("s_kv"), rounding="floor")


class TestTheWalkSpellsTheQueryAxisSCtx(unittest.TestCase):
    """The exported names are the producer's, and a consumer must use them.

    The walk spells the per-step query extent `s_ctx`, never `s_query`, and
    the name already carries the phase: prefill exports the prompt length,
    decode exports 1. A reader that binds measurement axes by name has to
    ask for the spelling that arrives.

    Pinned because a reader once asked for `s_query`, got it on no row of
    any model, and silently fell back for its whole life. Nothing went red:
    its unit test handed it `{"expr": "s_query"}` by hand, so the mechanism
    was proven while the wiring was never connected.
    """

    def setUp(self):
        snapshot_config(MOE.name)
        self.result = estimate_capacity(
             config=json.loads(MOE.read_text())["config"],
            resolved=_resolved_moe(),
            gpu_sku="H100-SXM", num_gpus=8,
            avg_input_tokens=4096, avg_output_tokens=256,
            parallel=ParallelConfig(),
        )

    def _exprs(self, column):
        for row in self.result.get(column) or []:
            for tensor in (row.get("shape_in_dims") or ()):
                axes = tensor if isinstance(tensor, (list, tuple)) else [tensor]
                for axis in axes:
                    if isinstance(axis, dict) and "expr" in axis:
                        yield row.get("op"), str(axis["expr"]), axis["value"]

    def test_the_query_axis_is_spelled_s_query(self):
        names = {e for _, e, _ in self._exprs("per_op_timing")}
        self.assertIn("s_query", names)

    def test_decode_query_axis_is_n_verify(self):
        names = {e for _, e, _ in self._exprs("per_op_timing_decode")}
        self.assertIn("n_verify", names)

    def test_the_spelling_carries_the_phase(self):
        """Same op, both phases: the axis is the STEP's query extent, so a
        name-bound reader gets a decode row that is genuinely decode."""
        def values(column, name):
            return {v for op, e, v in self._exprs(column)
                    if e == name and op == "rmsnorm"}

        prefill = values("per_op_timing", "s_query")
        decode = values("per_op_timing_decode", "n_verify")
        self.assertTrue(prefill and decode, "no rmsnorm row in one of the phases")
        self.assertTrue(all(v > 1 for v in prefill), prefill)
        self.assertEqual(decode, {1})


class TestTheEstimateResponseEncodes(unittest.TestCase):
    """The per-op rows of a MoE estimate survive the route's encoder.

    The unit tests above pin the conversion; this pins that the conversion is
    on the path the API actually takes, for the shape that actually occurs.
    """

    def setUp(self):
        snapshot_config(MOE.name)
        # An explicit non-power-of-two batch: the `max` sentinel now
        # resolves to the rung (a power of two), which makes every
        # expert-token axis whole — the fractional encoding this test
        # guards against rotting is exercised by a batch that divides
        # unevenly (`docs/sweep_card_class.md` §2.C).
        self.result = estimate_capacity(
             config=json.loads(MOE.read_text())["config"],
            resolved=_resolved_moe(),
            gpu_sku="H100-SXM", num_gpus=8,
            avg_input_tokens=4096, avg_output_tokens=256,
            parallel=ParallelConfig(), decode_batch_size=100,
        )

    def _axes(self):
        out = []

        def walk(obj):
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item)
            elif isinstance(obj, dict):
                if "expr" in obj and "value" in obj:
                    out.append(obj)
                for item in obj.values():
                    walk(item)

        for column in ("per_op_timing", "per_op_timing_decode"):
            walk(self.result.get(column) or [])
        return out

    def test_every_axis_is_int_or_string(self):
        axes = self._axes()
        self.assertTrue(axes, "no exported axes to check")
        for axis in axes:
            with self.subTest(expr=axis["expr"]):
                self.assertIsInstance(axis["value"], (int, str))

    def test_the_expert_token_axis_is_among_them_and_fractional(self):
        """Guards the guard: a corpus where every axis happened to be whole
        would let the conversion rot without a red test."""
        fractional = [a for a in self._axes() if isinstance(a["value"], str)
                      and "/" in a["value"]]
        self.assertTrue(fractional,
                        "expected a fractional expert-token axis")

    def test_the_response_is_encodable(self):
        from fastapi.encoders import jsonable_encoder
        from backend.inference.module_rows import served
        jsonable_encoder(served(self.result))


if __name__ == "__main__":
    unittest.main()
