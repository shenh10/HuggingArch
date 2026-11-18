"""The two halves of a step's HBM traffic, and the step that has no duration.

Three contracts, all of which were violated by a literal:

  • `mem_write_bytes` was the literal 0. A leaf's `access_bytes_per_rank` is
    `weight + metadata + act_in + act_out` — its whole traffic, both
    directions — so the write was never missing, it was published under a
    field named "read". On a Llama-3.2-3B prefill that is 39% of the step.
  • The frontend rebuilds the phase at a user-chosen batch from the per-op
    rows. It can only publish a true write half if the row carries one, so
    the row's per-request write must fold back to the phase's at the batch
    the phase was computed at.
  • A phase with no capacity for a single request returned 0.0 seconds. Zero
    is a duration, and every consumer divides tokens by it; the contract for
    a time that was not computed is absence plus a reason, which is what the
    pp>1 gate already does.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.capacity import estimate_capacity, withheld_phase

SNAP_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

_PHASE_ROWS = (("prefill", "per_op_timing"), ("decode", "per_op_timing_decode"))


def _run(model: str, **kw):
    from tests.snapshots import shared_loader
    cfg = json.loads((SNAP_DIR / f"{model}.json").read_text())["config"]
    resolved = shared_loader().match(cfg, model_id=model.replace("__", "/", 1))
    return estimate_capacity(config=cfg,
                             gpu_sku="H200", resolved=resolved,
                             num_gpus=8, avg_input_tokens=4096,
                             avg_output_tokens=256, **kw)


class TestTrafficSplit(unittest.TestCase):
    """Read and write are two names for halves of one fold."""

    MODELS = ("meta-llama__Llama-3.2-3B", "Qwen__Qwen3-30B-A3B")

    def test_read_plus_write_is_the_total(self):
        """The split renames halves of a sum; it must not move the sum. If it
        did, the arithmetic intensity printed beside them would move too."""
        for model in self.MODELS:
            res = _run(model)
            for phase, _ in _PHASE_ROWS:
                ph = res["phases"][phase]
                with self.subTest(model=model, phase=phase):
                    self.assertEqual(
                        ph["mem_read_bytes"] + ph["mem_write_bytes"],
                        ph["total_bytes"])

    def test_write_is_the_sum_of_the_leaves_act_out(self):
        """The phase's write is a fold over the same rows the breakdown shows,
        so a leaf and the summary card cannot disagree about who wrote what."""
        for model in self.MODELS:
            res = _run(model)
            for phase, key in _PHASE_ROWS:
                rows = res.get(key) or []
                ph = res["phases"][phase]
                if not rows:
                    continue
                with self.subTest(model=model, phase=phase):
                    self.assertEqual(
                        sum(int(r.get("act_out_bytes_per_rank") or 0)
                            for r in rows),
                        ph["mem_write_bytes"])

    def test_prefill_writes_a_large_share_and_decode_a_small_one(self):
        """Not a magic-number check — a direction check. Prefill moves
        activations and elementwise leaves write a third to two thirds of what
        they touch; decode re-reads resident weights and KV and writes almost
        nothing. A split that came back symmetric in both phases would mean
        the write side was tracking the wrong quantity."""
        res = _run("meta-llama__Llama-3.2-3B")
        share = {}
        for phase, _ in _PHASE_ROWS:
            ph = res["phases"][phase]
            share[phase] = ph["mem_write_bytes"] / ph["total_bytes"]
        self.assertGreater(share["prefill"], 0.15, share)
        self.assertLess(share["decode"], 0.05, share)

    def test_the_rows_writes_sum_to_the_phases(self):
        """The phase's write bytes are its rows' writes summed: the summary
        card and the table beside it are one step."""
        for model in self.MODELS:
            res = _run(model)
            for phase, key in _PHASE_ROWS:
                rows = res.get(key) or []
                ph = res["phases"][phase]
                if not rows:
                    continue
                with self.subTest(model=model, phase=phase):
                    self.assertEqual(
                        sum(int(r.get("act_out_bytes_per_rank") or 0)
                            for r in rows),
                        ph["mem_write_bytes"])

    def test_no_leaf_writes_more_than_it_touches(self):
        """`act_out` is a term inside `access_bytes_per_rank`, not a quantity
        beside it. A row where the part exceeds the whole means the two were
        computed from different shapes."""
        for model in self.MODELS:
            res = _run(model)
            for phase, key in _PHASE_ROWS:
                for r in (res.get(key) or []):
                    with self.subTest(model=model, phase=phase,
                                      op=r.get("op")):
                        self.assertLessEqual(
                            int(r.get("act_out_bytes_per_rank") or 0),
                            int(r.get("access_bytes_per_rank") or 0))


class TestNoCapacityWithholdsTime(unittest.TestCase):
    """DeepSeek-V3 on 8xH200 with no tensor parallel does not fit: 626.92 GB
    of weights per rank against 126.90 GB of usable HBM. No batch runs, so
    there is no step: each phase is its name, no batch, and why — no time,
    no counts, no table, no per-op rows (`capacity.without_step`).

    This used to describe the step at one request, times withheld, so a
    model too big for every deployment could still be read step by step.
    That published a batch the deployment does not admit and quoted a B=1
    step the card cannot run; a deployment that holds the weights is where
    the step is read."""

    MODEL = "deepseek-ai__DeepSeek-V3"

    def setUp(self):
        self.res = _run(self.MODEL)
        self.ph = self.res["phases"]["decode"]
        if not (self.ph.get("time_unpublished_reason") or "").strip():
            self.skipTest("this deployment now fits; pick a tighter one")

    def test_the_phase_is_its_name_and_its_reason(self):
        self.assertEqual(self.ph, withheld_phase(
            "decode", self.res["infeasibility_reason"]))

    def test_the_absence_carries_a_reason(self):
        self.assertTrue((self.ph.get("time_unpublished_reason") or "").strip())

    def test_no_rows_are_published(self):
        for rows in ("per_op_timing", "per_op_timing_decode",
                     "per_op_timing_draft"):
            with self.subTest(rows=rows):
                self.assertNotIn(rows, self.res)

    def test_the_admitted_batch_is_the_one_that_says_zero(self):
        """The verdict lives where it belongs: capacity says how many requests
        fit, and it says none."""
        self.assertEqual(int(self.res["max_batch_per_gpu"] or 0), 0)
