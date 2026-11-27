"""A measured Ascend row reaches the estimate that asked for it.

The write side (`kernel_bench`) and the read side (`calibration_overlay`)
are different programs, so a row measured under one key and looked up under
another is invisible — and invisible without failing, which is the whole
reason `point_keys` is the single derivation both read. These assert the
round trip end to end, on rows measured on a real 910C die:

    sweep -> jobs -> run on the device -> CSV -> overlay -> a row's measured
    column

The corpus is `tests/golden/ascend_vllm_ascend.csv`, 789 rows measured on
one die of an Atlas A3 node inside the
`quay.io/ascend/vllm-ascend:v0.30.0-a3` container (the release the binding
is audited at), through the harness's own `kernel_bench run`.
"""
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.parallel_config import ParallelConfig
from tests.sweep_parity import Deployment

CORPUS = PROJECT_ROOT / "tests" / "golden" / "ascend_vllm_ascend.csv"
NPU = "Ascend-910C-A3"
MODEL = "Qwen/Qwen3-30B-A3B"
#: The sheet's ceilings, which no measured row may exceed.
CEILINGS = {"bf16": 248.4, "int8": 496.8, "bw": 1600.0}


def _rows():
    with open(CORPUS) as fh:
        return list(csv.DictReader(fh))


#: The deployment the corpus was measured at. A measured row is keyed on
#: its per-rank shape, so an estimate at another batch, length or layout
#: asks for a different cell and finds nothing — correctly. Read off the
#: corpus rather than chosen: its densest point is dp_attention=2 / ep_moe=2
#: at one request of 8192 tokens.
CORPUS_PARALLEL = dict(dp_attention=2, ep_moe=2)
CORPUS_SEQ = 8192


def _estimate(rows=None):
    """The estimate, with or without a corpus mounted.

    `selection_mode` is the deployment's own field, not a knob: it decides
    whether a measured lookup happens at all (`_resolve_overlay_context`).
    """
    knobs = {"framework": "vllm_ascend"}
    if rows is not None:
        knobs["calibration_rows"] = rows
    return Deployment(
        model_id=MODEL, gpu_sku=NPU, knobs=knobs,
        selection_mode="latest" if rows is not None else "theoretical_only",
        parallel=ParallelConfig(**CORPUS_PARALLEL),
    ).direct(avg_input_tokens=CORPUS_SEQ, prefill_batch=1, decode_batch=1)


class TestTheCorpusIsWhatTheDeviceMeasured(unittest.TestCase):

    def test_every_row_is_an_ascend_row_from_the_audited_release(self):
        """Every row was measured on this card, by a library this binding
        dispatches to.

        `framework` is not "which deployment did this" — it is which
        LIBRARY'S provider measured the launch. On an NPU an unquantised
        bf16 linear runs torch's own `F.linear`, which dispatches to
        `aclnnMatmul_MatMulV3Common_MatMulV3` (`vllm_ascend/ops/
        linear.py:49-54`; read off the Ascend profiler, not assumed), so
        the binding declares `library: torch` for it and the 462 rows it
        produced are named `torch`. Asserting one fixed name would have
        said those rows came from somewhere else.

        Derived from the declarations rather than listed here: a second
        list would be free to drift from the binding, which is the whole
        failure mode these rows exist to rule out.
        """
        from backend.inference.framework_bindings import get_binding
        binding = get_binding("vllm_ascend")
        declared = {entry.library for entry in binding.kernel_dispatches}
        self.assertIn("vllm_ascend", declared)
        rows = _rows()
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r["gpu"], NPU, r["op"])
            self.assertIn(r["framework"], declared, r["op"])

    def test_no_row_was_timed_on_the_host_clock(self):
        # `wall_clock` is what a CPU fallback records. One such row in an
        # Ascend corpus means the run never reached the device, which is
        # the defect `device.py` exists to prevent — and it would not
        # otherwise fail anything.
        for r in _rows():
            self.assertNotEqual(r.get("timing_method"), "wall_clock", r["op"])

    def test_no_row_claims_more_than_the_card(self):
        # The sheet's cells are upper bounds (`gpu_specs` docstring); a row
        # above one means either the cell is under the hardware or the row
        # counted work the kernel did not do. The harness drops such rows at
        # write time, so this pins that none survived.
        for r in _rows():
            peak = float(r.get("peak_tflops_at_test") or 0)
            if peak:
                self.assertLessEqual(peak, CEILINGS["int8"], r["op"])


class TestTheCorpusSpansMoreThanOneShape(unittest.TestCase):
    """What the corpus covers, asserted rather than assumed.

    A corpus of one model at one phase proves the lookup works for that
    cell and says nothing about the rest. These pin the span the rows
    actually have, so a future run that silently narrows it goes red.
    """

    def test_it_holds_more_than_one_model(self):
        models = {r["model"] for r in _rows()}
        self.assertIn("zai-org/GLM-5.3-Flash", models)
        self.assertGreaterEqual(len(models), 3, sorted(models))

    def test_it_has_not_shrunk(self):
        # A row count is the one thing a re-measurement can quietly lose:
        # a run that measures fewer cells still writes a valid CSV.
        self.assertGreaterEqual(len(_rows()), 789)

    def test_it_holds_both_phases(self):
        phases = {r["phase"] for r in _rows()}
        self.assertEqual(phases, {"prefill", "decode"}, sorted(phases))

    def test_the_decode_rows_read_the_plugins_page(self):
        # vLLM-Ascend allocates at 128 (`vllm_ascend.yaml kv_block_size`),
        # and a paged kernel's time depends on the page, so a decode
        # attention row keyed at another page would be a different cell.
        import json
        pages = {json.loads(r["shape_json"]).get("kv_block_size")
                 for r in _rows()
                 if r["phase"] == "decode" and "attention" in r["op"]}
        self.assertTrue(pages <= {128, None}, sorted(p for p in pages if p))


class TestTheOverlayFindsThem(unittest.TestCase):
    """The round trip: what the device measured reaches the estimate."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _rows()
        cls.sol = _estimate()
        cls.measured = _estimate(rows=cls.rows)

    def _per_op(self, result):
        return [r for phase in ("per_op_timing", "per_op_timing_decode")
                for r in (result.get(phase) or [])]

    def test_the_theory_track_does_not_move_when_a_corpus_is_mounted(self):
        """Measured sits BESIDE the roofline and never overwrites it.

        The two tracks answer different questions — "how fast could this
        card run it" and "how fast did this kernel run it" — and a reader
        comparing them needs the first to be the same number it was without
        a corpus.
        """
        theory = [(r["op"], r.get("t_sol_s")) for r in self._per_op(self.sol)]
        with_rows = [(r["op"], r.get("t_sol_s"))
                     for r in self._per_op(self.measured)]
        self.assertEqual(theory, with_rows)

    def test_a_measured_row_reaches_the_op_that_asked_for_it(self):
        # The assertion that catches a key drift: the corpus was written by
        # `kernel_bench run` from `point_keys`, and the estimate reads the
        # same function. If the two ever derive differently, this is empty
        # and nothing else fails.
        hit = [r for r in self._per_op(self.measured)
               if r.get("t_measured_s")]
        self.assertTrue(
            hit, "no measured row reached the estimate — the write side and "
                 "the read side derived different keys")

    def test_a_measured_time_never_beats_the_roofline(self):
        for r in self._per_op(self.measured):
            t_m, t_s = r.get("t_measured_s"), r.get("t_sol_s")
            if t_m and t_s:
                self.assertGreaterEqual(
                    t_m, t_s * 0.999,
                    f"{r['op']}: measured {t_m} beats SOL {t_s}")

    def test_the_collectives_still_price_on_theory(self):
        # Their providers are registered and disabled (one process cannot
        # measure a group), so these rows carry a roofline and no measured
        # time — which is the modelled fact, not a gap.
        comm = [r for r in self._per_op(self.measured) if r.get("is_comm")]
        self.assertTrue(comm)
        for r in comm:
            self.assertTrue(r.get("t_sol_s"), r["op"])


if __name__ == "__main__":
    unittest.main()
