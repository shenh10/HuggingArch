"""Under pp, the published memory picture describes exactly one stage.

A pipeline stage is a set of layers, so it has its own weights AND its own
cache, and the two are not the same fraction of the model. Which stage limits
the deployment therefore has more than one answer, and the code asks the
question twice on purpose:

* `_capacity_preparation` ranks stages by how many cache TOKENS they can hold.
  That question is workload-free, which is why a sweep can reuse one
  preparation across every sequence length.
* `capacity_ceilings` re-ranks by how many whole REQUESTS fit at the length it
  was actually asked about. A hybrid backbone's stages differ in how many of
  their layers are windowed, and a request's bytes are its growth per token
  times its length PLUS a per-request floor (a full window is held whatever
  the length). The token ranking sees only the growth; the request ranking
  sees the floor too, and on short requests the floor is most of the bill —
  MiMo-V2.5 at pp=8 reported 3770 concurrent requests where its worst stage
  could hold 2182.

Both questions are legitimate. The defect was that the SECOND answer stayed in
locals: `max_batch` came from it while the weights, the budget, the cache
tokens and the weight tree printed beside it came from the first. The two
argmins agree on most of the corpus, which is why it survived — they disagree
on 23 of 468 (model, pp, length) points, and where they do the gap is not
rounding. MiMo-V2-Flash at pp=4 published 75.76 GiB of weights next to a batch
computed against a rank holding 69.95.

So what this pins is not "the stage choice is X". It is that every per-rank
number on one response comes from ONE stage, and that the response says which.
A budget from one rank divided by a footprint from another is not any rank's
answer, and it is not visibly wrong from either side.
"""
from __future__ import annotations

import json
import sys
import unittest

import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

#: (model, pp, tp, avg_in, avg_out). The first three are measured points where
#: the two argmins DISAGREE, so they exercise the seam rather than a case where
#: coherence is free. The last two are controls: a pp where they agree, and
#: pp=1 where the question does not arise.
#:
#: The MiMo-V2-Flash and Step-3.7-Flash points are SHORT requests (384 tokens)
#: since KV per token became the cache's growth per token of length (d82c17baa).
#: Before it, the token ranking billed every sliding-window layer per token, so
#: it disagreed with the request ranking at 4096 — where the windows had long
#: capped. Now the token ranking leaves the windows out (they are a per-request
#: floor), which is right for long requests, and the two part where the floor
#: is most of a request's bytes: at or under one window. Measured on H200,
#: tp=1: MiMo-V2-Flash pp=4 (window 128) binds stage 3, not 0, up to a
#: 640-token request; Step-3.7-Flash pp=8 (window 512) binds stage 2, not 3,
#: up to 512. 384 sits inside both.
CASES = (
    ("XiaomiMiMo__MiMo-V2-Flash", 4, 1, 256, 128),
    ("stepfun-ai__Step-3.7-Flash", 8, 1, 256, 128),
    ("google__gemma-2-9b", 4, 1, 1024, 128),
    ("deepseek-ai__DeepSeek-V3", 2, 8, 4096, 256),
    ("meta-llama__Llama-3.2-3B", 1, 8, 4096, 256),
)

#: The three points above that were measured to disagree. Kept separate so a
#: refactor that made the two argmins coincide — collapsing the seam instead of
#: closing it — cannot pass this file by making every case trivial.
DISAGREEING = {
    ("XiaomiMiMo__MiMo-V2-Flash", 4),
    ("stepfun-ai__Step-3.7-Flash", 8),
    ("google__gemma-2-9b", 4),
}


def _run(model: str, pp: int, tp: int, avg_in: int, avg_out: int):
    from backend.inference.capacity import (
        CapacityKnobs, _capacity_preparation, capacity_ceilings,
        estimate_capacity)
    from backend.inference.parallel_config import ParallelConfig

    config = json.loads(
        (SNAPSHOT_DIR / f"{model}.json").read_text())["config"]
    from tests.snapshots import shared_loader
    resolved = shared_loader().match(config, model_id=model.replace("__", "/", 1))
    parallel = ParallelConfig(tp_attention=tp, tp_dense_ffn=tp,
                              tp_moe_expert=tp, pp=pp)
    kwargs = dict(config=config, resolved=resolved, gpu_sku="H200",
                  num_gpus=pp * tp,
                  avg_input_tokens=avg_in, avg_output_tokens=avg_out,
                  parallel=parallel)
    prep = _capacity_preparation(
        config=config, gpu_sku="H200",
        knobs=CapacityKnobs(resolved=resolved,
                            num_gpus=pp * tp, parallel=parallel))
    ceil = capacity_ceilings(prep, avg_input_tokens=avg_in,
                             avg_output_tokens=avg_out)
    return prep, ceil, estimate_capacity(**kwargs)


@pytest.mark.regression  # minutes-long
class TestPipelineStageCoherence(unittest.TestCase):

    maxDiff = None

    def test_every_per_rank_number_comes_from_the_named_stage(self):
        for model, pp, tp, avg_in, avg_out in CASES:
            with self.subTest(model=model, pp=pp):
                prep, ceil, res = _run(model, pp, tp, avg_in, avg_out)
                stage = res["memory"]["pp_stage"]
                self.assertEqual(
                    stage, ceil.pp_stage,
                    "the response names a stage the ceilings layer did not bind")
                self.assertEqual(
                    (res["weight_breakdown"].get("pipeline") or {}).get("stage"),
                    stage,
                    "the weight TREE describes a different stage than the "
                    "scalars quoted beside it, so the module breakdown does "
                    "not add up to the total printed above it")
                self.assertAlmostEqual(
                    res["memory"]["weight_bytes_per_gpu"],
                    res["weight_breakdown"]["total_bytes_per_rank"], delta=1.0)
                self.assertEqual(res["max_tokens_per_gpu"], ceil.max_tokens)
                self.assertEqual(res["max_batch_per_gpu"], ceil.max_batch)
                self.assertEqual(res["memory"]["fits"], ceil.fits)

                if prep.pp_stages:
                    by_stage = {s.stage: s.weight_bytes for s in prep.pp_stages}
                    self.assertAlmostEqual(
                        res["memory"]["weight_bytes_per_gpu"], by_stage[stage],
                        delta=1.0,
                        msg="the published weight is not the named stage's")
                    self.assertAlmostEqual(
                        res["memory"]["weight_bytes_per_gpu_heaviest_stage"],
                        max(by_stage.values()), delta=1.0)

    def test_the_compute_is_billed_on_the_named_stage(self):
        """The rows and the phase FLOPs are the rank the memory describes.

        On the points where the two rankings disagree, the stage-restricted
        compute must be the stage the response names, not the workload-free
        one the preparation started from. The evidence moved when pp>1
        started publishing the pipeline-step time: `time_unpublished_reason`
        no longer names the stage (times are published), so the bound stage
        rides `pipeline_time.stage` instead.
        """
        for model, pp, tp, avg_in, avg_out in CASES:
            if (model, pp) not in DISAGREEING:
                continue
            with self.subTest(model=model, pp=pp):
                prep, ceil, res = _run(model, pp, tp, avg_in, avg_out)
                named = res["memory"]["pp_stage"]
                self.assertEqual(named, ceil.pp_stage)
                self.assertNotEqual(named, prep.pp_bound.stage)
                for phase in ("prefill", "decode"):
                    pt = res["phases"][phase].get("pipeline_time")
                    self.assertIsNotNone(pt, "pp>1 publishes pipeline_time")
                    self.assertEqual(pt["stage"], named,
                                     f"{phase}: compute billed on stage "
                                     f"{pt['stage']}, memory names {named}")

    def test_the_seam_is_still_reachable(self):
        """The disagreeing points must still disagree.

        Coherence is trivial to hold by making the two questions the same
        question, and that would be a regression wearing a green test: the
        workload-free ranking exists so a sweep can reuse one preparation, and
        the workload-aware one exists because a per-request floor (a full
        window) is invisible to a ranking by growth per token.
        """
        for model, pp, tp, avg_in, avg_out in CASES:
            if (model, pp) not in DISAGREEING:
                continue
            with self.subTest(model=model, pp=pp):
                prep, ceil, _ = _run(model, pp, tp, avg_in, avg_out)
                self.assertNotEqual(
                    prep.pp_bound.stage, ceil.pp_stage,
                    "this point no longer exercises the seam — either the "
                    "stage ranking changed meaning, or the two questions "
                    "collapsed into one. Find a point that still separates "
                    "them, or delete the second question deliberately.")

    def test_the_deployment_total_does_not_follow_the_stage(self):
        """`weight_bytes_total` is the whole model, on every stage's rank.

        Re-binding moves the per-RANK picture. A model does not get smaller
        because a different rank is the one that runs out first, and the
        rebuild of the weight tree is the easiest place to lose that.
        """
        for model, pp, tp, avg_in, avg_out in CASES:
            with self.subTest(model=model, pp=pp):
                prep, _, res = _run(model, pp, tp, avg_in, avg_out)
                self.assertAlmostEqual(res["weight_bytes_total"],
                                       prep.weight_bytes_total, delta=1.0)


@pytest.mark.regression  # minutes-long
class TestEveryCardBillsItsOwnStage(unittest.TestCase):
    """The parallel sweep prices one layout on several cards from one walk.

    A card with less HBM can run out on a different stage: MiMo-V2-Flash at
    pp=4 binds stage 0 on an H200 and stage 3 on an H20. The H20's rows must
    then be stage 3's — the rows a direct estimate on the H20 publishes — and
    not the H200's stage 0 rows re-priced under the H20's peaks. (The pair
    used to be H200/H100-SXM, but the H100-SXM no longer holds the weights:
    74.65 GiB against 72.00 GiB of usable HBM, and a card that admits no
    request publishes no rows to compare.)
    """

    MODEL = "XiaomiMiMo__MiMo-V2-Flash"
    CARDS = ("H200", "H20")

    def test_a_second_card_on_another_stage_gets_that_stages_rows(self):
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig
        from tests.sweep_parity import layout_at
        from tests.snapshots import shared_loader

        config = json.loads(
            (SNAPSHOT_DIR / f"{self.MODEL}.json").read_text())["config"]
        resolved = shared_loader().match(
            config, model_id=self.MODEL.replace("__", "/", 1))
        common = dict(config=config, resolved=resolved,
                      parallel=ParallelConfig(pp=4), num_gpus=4,
                      avg_input_tokens=4096, avg_output_tokens=256,
                      prefill_batch_size=1, decode_batch_size=8)
        stamped = layout_at({**common, "gpu_skus": list(self.CARDS)})
        direct = {sku: estimate_capacity(gpu_sku=sku, **common)
                  for sku in self.CARDS}
        stages = {sku: d["memory"]["pp_stage"] for sku, d in direct.items()}
        self.assertNotEqual(
            stages[self.CARDS[0]], stages[self.CARDS[1]],
            "the two cards no longer bind different stages, so this point "
            "does not exercise the re-pricing seam; find one that does")
        for sku in self.CARDS:
            for key in ("per_op_timing", "per_op_timing_decode"):
                with self.subTest(sku=sku, key=key):
                    got = {r["aggregation_id"]: self._facts(r)
                           for r in stamped[sku]["sol"][key]}
                    want = {r["aggregation_id"]: self._facts(r)
                            for r in direct[sku][key]}
                    self.assertEqual(got, want)

    @staticmethod
    def _facts(row):
        """What a row bills on its stage: the work, how often, the payload."""
        return (row["flops_per_rank"], row["count_per_step"],
                row["weight_read_bytes_per_rank"], row["comm_tensor_bytes"])


if __name__ == "__main__":
    unittest.main()
