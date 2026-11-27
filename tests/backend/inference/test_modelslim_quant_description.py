"""What a checkpoint quantised is the checkpoint's to say, not a preset's.

GLM-5.3-Flash ships twice. The official release is block-FP8 across
attention and FFN; an msModelSlim export for Ascend quantised the FFN to
INT8 and left everything else BF16. Their `config.json` files are
byte-identical, so nothing in the config tells them apart — the difference
is which tensors carry a quantised format, which `quant_context` is the
one place to say.

These assert that `quant_context.per_role` can express that export, by
building it and walking it. The export is NOT registered as a model id of
its own: its tensors are also NAMED differently (`attn_hc.base` where the
official checkpoint writes `hc_attn_base`, and 349 header entries the
official spec does not declare), so a spec for it is a spec written
against its own names rather than this one with two lines changed. That
is its own piece of work; what is settled here is the quantisation
question, which needed no engine change at all.

Read off the real export at /data3/GLM-5.3-Flash-w8a8 (82 shards,
113,353 tensors, headers read directly):

  * 111,843 `W8A8_DYNAMIC` entries, every one an expert's gate/up/down;
  * 27 more in the first three layers — the dense FFNs that
    `first_k_dense_replace: 3` leaves before the MoE stack starts;
  * every other tensor `FLOAT`, including all 957 attention ones.

Of the 100 distinct tensor-path shapes in its description file, NONE
carries both labels: the quantisation boundary falls on roles, which is
why two `per_role` rows are the whole of it.
"""
from __future__ import annotations

import collections
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.parallel_config import ParallelConfig
from backend.arch.loader import SpecLoader
from tests.snapshots import snapshot_config
from tests.sweep_parity import Deployment

MODEL = "zai-org/GLM-5.3-Flash"
PARALLEL = dict(tp_attention=8, tp_dense_ffn=8, ep_moe=8)


def _resolved():
    """A fresh resolve, so an injected `quant_context` cannot leak into
    another test through the shared loader's cache."""
    return SpecLoader().match(snapshot_config(MODEL), model_id=MODEL)


def _as_the_export_quantised_it(resolved):
    """The export's topology on this spec: the FFN roles INT8, the rest
    left at the width the checkpoint stores."""
    quant = resolved.spec.quant_context
    entry = type(next(iter(quant.per_role.values())))
    w8a8 = entry(method="modelslim-w8a8-dynamic")
    quant.per_role = {**{role: entry(method="none") for role in quant.per_role},
                      "ffn_dense": w8a8, "ffn_moe": w8a8}
    return resolved


def _estimate(resolved=None, preset=None):
    knobs = {"framework": "vllm_ascend"}
    if preset:
        knobs["precision"] = {"preset": preset}
    if resolved is not None:
        knobs["resolved"] = resolved
    return Deployment(model_id=MODEL, gpu_sku="Ascend-910C-A3", knobs=knobs,
                      parallel=ParallelConfig(**PARALLEL),
                      ).direct(avg_input_tokens=2048, prefill_batch=1,
                               decode_batch=16)


def _weight_widths(result):
    out = collections.defaultdict(collections.Counter)
    for row in result.get("per_op_timing") or []:
        if row.get("weight_dtype"):
            out[row.get("role") or "?"][row["weight_dtype"]] += 1
    return out


def _casts(result):
    return sum(1 for row in (result.get("per_op_timing") or [])
               if row.get("op") == "quantise_activation")


class TestPerRoleCanExpressTheExport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.export = _estimate(_as_the_export_quantised_it(_resolved()))
        cls.widths = _weight_widths(cls.export)

    def test_both_ffn_roles_turn_int8(self):
        for role in ("ffn_moe", "ffn_dense"):
            self.assertEqual(set(self.widths[role]), {"INT8"},
                             f"{role}: {dict(self.widths[role])}")

    def test_the_attention_stack_stays_as_stored(self):
        for role in ("attention", "attention_linear", "attention_kv_up",
                     "attention_indexer"):
            widths = set(self.widths[role])
            self.assertNotIn("INT8", widths, f"{role}: {widths}")


class TestItIsNeitherNeighbour(unittest.TestCase):
    """The two answers this one is not."""

    def test_the_official_release_quantises_the_ffn_to_fp8(self):
        official = _weight_widths(_estimate())
        self.assertEqual(set(official["ffn_moe"]), {"F8_E4M3"})
        export = _weight_widths(
            _estimate(_as_the_export_quantised_it(_resolved())))
        self.assertEqual(set(export["ffn_moe"]), {"INT8"})

    def test_the_preset_reaches_the_same_roles_the_export_did(self):
        """`w_int8_w8a8_ffn` states the two rules this export states.

        The preset it replaced selected `weight_class`, which turned the
        attention linears INT8 too — a width no msModelSlim export writes.
        This asserts the stored topology and the selectable one now agree
        role by role, so a reader who picks the preset is reading this
        checkpoint and not a hypothetical full-W8A8 one.
        """
        preset = _weight_widths(_estimate(preset="w_int8_w8a8_ffn"))
        export = _weight_widths(_estimate(_as_the_export_quantised_it(_resolved())))
        for role in sorted(set(preset) | set(export)):
            self.assertEqual(set(preset[role]), set(export[role]), role)

    def test_the_preset_also_states_the_activation_the_injection_cannot(self):
        """Where the two legitimately differ, and why the preset is the
        truer one.

        `quant_context.per_role` names a weight `method` and has nowhere to
        say what the matmul then eats, so injecting it leaves BF16
        activations — a BF16 MMA with INT8 weights. W8A8_DYNAMIC is not
        that: `npu_quant_matmul_w8a8`, the kernel this binding dispatches
        to, takes INT8 activations (`kernel_bench/providers/ascend.py`
        serves `activation=("int8",)`), and the dynamic quantise in front
        of it is a launch of its own. The preset says both, so it grafts
        those casts and prices the matmul on the INT8 peak.
        """
        preset = _estimate(preset="w_int8_w8a8_ffn")
        export = _estimate(_as_the_export_quantised_it(_resolved()))
        self.assertGreater(_casts(preset), _casts(export))
        # The INT8 peak is twice the BF16 one, so the same INT8-weight
        # matmuls price strictly faster when the activations are INT8 too.
        def _ffn_matmul_ms(result):
            return sum(float(r.get("t_sol_s") or 0) * 1000
                       for r in (result.get("per_op_timing") or [])
                       if r.get("weight_dtype") == "INT8")
        self.assertLess(_ffn_matmul_ms(preset), _ffn_matmul_ms(export),
                        "INT8 activations must price below BF16 ones")


if __name__ == "__main__":
    unittest.main()
