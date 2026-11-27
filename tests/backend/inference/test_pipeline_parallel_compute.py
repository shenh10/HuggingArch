"""A pipeline stage computes its own layers, and the clock says so.

`test_pipeline_parallel_cache` pinned the MEMORY half: a stage holds its own
layers' weights and its own layers' cache. The compute half was still whole.
A rank at `pp=8` was reported holding an eighth of the parameters and running
all twenty-eight layers — a per-rank utilisation eight times the truth, and a
per-op table that listed every op in the model under the label of one rank.

Two things close it, and this file pins both because each is a different kind
of claim:

  * the counts. A stage's view (`layout.stages.StageView.rows`) is the one
    fold of pipeline ownership, and FLOPs, per-op rows and communication all
    read that layer;
  * the clock. A stage's step is not the model's step: the model's is the
    stages in sequence plus the fill/drain bubble and the inter-stage handoff,
    and neither is modelled. `t_sol_s` is read downstream as TTFT, as TPOT and
    as the denominator of every tokens/s, so publishing a stage's wall time
    there would divide the true latency by `pp`. It is withheld instead, and
    it says why — including when the phase folds the per-op rows, whose sum is
    a real quantity (that stage's wall time) and still not the one `t_sol_s`
    means.

What is NOT asserted is any relation between the compute ratio and the memory
ratio. On a hybrid stack there is none: Qwen3-Coder-Next's binding stage holds
2 of 12 full-attention layers and 4 of 36 linear ones, so its cache shrinks by
6.34x while its parameters shrink by 7.59x, and both are right. The claim that
survives a hybrid is the literal one — the same stage, named once.
"""
from __future__ import annotations

import json
import sys
import unittest

import pytest
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.dim import value_of
from backend.arch.layout.stages import StagePlacement
from backend.arch.loader import get_default_loader
from backend.inference import pipeline as _pipeline
from backend.inference.capacity import (
    CapacityKnobs,
    _capacity_preparation,
    capacity_ceilings,
    estimate_capacity,
)
from backend.inference.phases import (
    PhaseInputs,
    _flops_rows,
    _rows_total,
    estimate_phase,
)
from backend.inference.invariants import _inv_phase_closure
from backend.inference.parallel_config import ParallelConfig
from backend.inference.step_eval import evaluate_step

SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"

#: One of each shape the restriction has to survive. Llama is homogeneous;
#: gpt-oss alternates SWA and GQA by declared layer index, so a stage can hold
#: zero of a template; GLM-5.1 declares one MoE template twice, which is why
#: the counts are keyed by block ordinal and not by name; Qwen3-Coder-Next is
#: the hybrid whose stages differ in mix as well as in size.
_MODELS = (
    "meta-llama__Llama-3.2-3B",
    "openai__gpt-oss-20b",
    "zai-org__GLM-5.1",
    "Qwen__Qwen3-Coder-Next",
)


@lru_cache(maxsize=None)
def _config(name: str) -> dict:
    return json.loads((SNAPSHOT_DIR / f"{name}.json").read_text())["config"]


@lru_cache(maxsize=None)
def _resolved(name: str):
    return get_default_loader().match(
        _config(name), model_id=name.replace("__", "/", 1))


@lru_cache(maxsize=None)
def _step(name: str):
    return evaluate_step(
        _resolved(name), ParallelConfig(), phase="prefill",
        batch=1, input_tokens=128, output_tokens=0,
    )


@lru_cache(maxsize=None)
def _placement(name: str, pp: int):
    return StagePlacement.of(_step(name).analysis, pp)


def _rows(name: str, pp: int, stage: int):
    """The stage's counts of every node, every stack."""
    program = _step(name).analysis.program
    return _placement(name, pp).view(stage).rows(
        program.included_stack_names(include_aux=True))


#: GLM-5.1 holds 173 GiB a rank at pp=8 — no H200 stage of eight admits a
#: request, and a card that admits none publishes no clock to quote. Its
#: point is priced at tp=2, the smallest cut that holds it (92 GiB a rank).
_PP_EXTRA_TP = {"zai-org__GLM-5.1": 2}


def _pp_parallel(name: str, pp: int) -> ParallelConfig:
    tp = _PP_EXTRA_TP.get(name, 1)
    return ParallelConfig(pp=pp, tp_attention=tp, tp_dense_ffn=tp,
                          tp_moe_expert=tp)


@lru_cache(maxsize=None)
def _estimate(name: str, pp: int) -> dict:
    return estimate_capacity(
         config=_config(name), gpu_sku="H200", resolved=_resolved(name),
        selection_mode="theoretical_only",
        avg_input_tokens=8192, avg_output_tokens=256,
        parallel=_pp_parallel(name, pp))


class StageRestrictionIsASubtraction(unittest.TestCase):
    """Pipeline ownership changes multiplicity, never geometry or cost."""

    def test_the_stages_layer_flops_sum_back_to_the_model(self):
        """The claim a subtraction has to answer for.

        Each stage's block FLOPs are its layers'; summed over stages they are
        the model's, once. Keyed by block ordinal, so GLM-5.1's two
        invocations of one MoE template are two entries — keyed by name each
        would take the name's whole share and the stages would sum to nearly
        twice the model, which is the bug this mirrors on the memory side.
        """
        for name in _MODELS:
            with self.subTest(name):
                analysis = _step(name).analysis
                whole_counts = analysis.layer_multiplicity()
                blocks = [c for c in analysis.program.contributions
                          if c.block_index is not None]
                whole = sum(
                    float(value_of(analysis.cost.values[c.node.uid].flops))
                    * whole_counts.values[c.node.uid] for c in blocks)
                staged = sum(
                    float(value_of(analysis.cost.values[c.node.uid].flops))
                    * _rows(name, 8, stage).values[c.node.uid]
                    for stage in range(8) for c in blocks)
                self.assertAlmostEqual(staged / whole, 1.0, places=9)

    def test_an_aux_stack_counts_whole_on_its_own_stage(self):
        """An aux stack has its own layer index space: the partition does
        not cut it, and it is placed whole (GLM-5.1's MTP head reads the
        norm, so it runs on the last stage, all of it, and nowhere else)."""
        analysis = _step("zai-org__GLM-5.1").analysis
        aux = [c for c in analysis.program.contributions
               if c.block_index is not None and c.stack == "mtp"]
        self.assertTrue(aux, "model has no aux stack to test with")
        whole = analysis.layer_multiplicity()
        for contribution in aux:
            uid = contribution.node.uid
            self.assertEqual(
                [_rows("zai-org__GLM-5.1", 8, s).values[uid]
                 for s in range(8)],
                [0] * 7 + [whole.values[uid]])

    def test_the_totals_move_with_the_counts(self):
        """The phase rows equal the selected contribution-root fold."""
        for name in _MODELS:
            with self.subTest(name):
                step = _step(name)
                analysis = step.analysis
                multiplicity = _rows(name, 8, 0)
                included = analysis.program.included_stack_names()
                rebuilt = sum(
                    round(
                        float(value_of(analysis.cost.values[c.node.uid].flops))
                        * multiplicity.values[c.node.uid]
                    )
                    for c in analysis.program.contributions
                    if c.stack in included
                )
                self.assertEqual(
                    _rows_total(_flops_rows(step, multiplicity)), rebuilt
                )


class PerOpRowsFollowTheStage(unittest.TestCase):
    """The other consumer of the counts, restricted through the same call."""

    def test_the_per_op_flops_shrink_by_the_layer_ratio(self):
        """The selected multiplicity scales per-op compute and communication.

        The ratio it produces is the layer ratio, not `pp`.
        """
        name = "meta-llama__Llama-3.2-3B"
        resolved = _resolved(name)

        def total(stage):
            a = _pipeline.analyze(resolved, parallel=ParallelConfig(pp=8),
                                  batch_size=1, avg_input_tokens=128, avg_output_tokens=0, stage=stage, breakdown=False)
            return sum(float(r.get("flops_per_rank") or 0)
                       for r in a.per_op_timing if r.get("template_name"))

        # 28 layers over 8 stages is 3,3,3,4,4,4,4,3 — stage 0 holds 3.
        self.assertEqual(len(_placement(name, 8).ranges[0]), 3)
        whole, stage = total(None), total(0)
        self.assertAlmostEqual(whole / stage, 28 / 3, places=6)

    def test_the_row_set_is_the_stages_ops(self):
        """A template a stage holds none of contributes nothing.

        gpt-oss alternates SWA and GQA by declared layer index, so its per-op
        rows are the same op classes on every stage; what changes is the
        count each is charged. The rows themselves are structural and stay —
        this pins that the restriction is a re-count and not a filter, which
        is why `per_op_timing` keeps its rows at `pp=8` — beside the rows a
        stage adds: its send across the exit cut and the seam-carried add
        that cut keeps beside the hop.
        """
        name = "meta-llama__Llama-3.2-3B"
        one = _estimate(name, 1)
        many = _estimate(name, 8)
        sends = [r for r in many["per_op_timing"]
                 if r.get("comm_op") == "send_recv"]
        seam_adds = [
            r for r in many["per_op_timing"]
            if r.get("op") == "add"
            and r.get("display_path") == "out__block"
            and r.get("spec_id") is None
        ]
        self.assertLessEqual(len(sends), 2)
        self.assertEqual(len(seam_adds), 1)
        self.assertEqual(len(one["per_op_timing"]),
                         len(many["per_op_timing"])
                         - len(sends) - len(seam_adds))
        self.assertLess(
            sum(int(r.get("flops_per_rank") or 0)
                for r in many["per_op_timing"]),
            sum(int(r.get("flops_per_rank") or 0)
                for r in one["per_op_timing"]))


class ModelLevelStepsRunOnTheirStage(unittest.TestCase):
    """The embedding runs on the first stage and the final norm and LM head
    on the last (`StagePlacement`), and so do their collectives: the
    vocab-parallel embedding's all_reduce and the head's vocab all_gather.
    Summed over the stages each is paid once, as at `pp = 1`."""

    def _rows(self, stage):
        name = "meta-llama__Llama-3.2-3B"
        parallel = ParallelConfig(pp=2, tp_attention=4, tp_dense_ffn=4)
        a = _pipeline.analyze(
            _resolved(name), parallel=parallel, batch_size=1, avg_input_tokens=128, avg_output_tokens=0,
            stage=stage, breakdown=False)
        return {r["aggregation_id"]: r for r in a.per_op_timing}

    def test_each_runs_once_on_its_stage(self):
        def paid(rows):
            return {
                "embed_tokens.vocab_all_reduce":
                    rows["embed_tokens.vocab_all_reduce"]["comm_bytes_per_rank"],
                "lm_head.vocab_all_gather":
                    rows["lm_head.vocab_all_gather"]["comm_bytes_per_rank"],
                "lm_head": rows["lm_head"].get("flops_per_rank") or 0,
                "norm": rows["norm"].get("flops_per_rank") or 0,
            }

        whole = paid(self._rows(None))
        first, last = paid(self._rows(0)), paid(self._rows(1))
        self.assertTrue(all(whole.values()), whole)
        self.assertEqual(first, {**whole, "lm_head.vocab_all_gather": 0,
                                 "lm_head": 0, "norm": 0})
        self.assertEqual(last, {**whole, "embed_tokens.vocab_all_reduce": 0})


class AFrameworkSelectionRefusesNothing(unittest.TestCase):
    """Naming a framework changes how a value is dispatched, never whether
    the model is estimated: sglang has no PP path for Qwen3-Next
    (qwen3_next.py:1011 asserts first and last rank), and its pp = 2
    estimate is still priced."""

    def test_the_estimate_and_the_per_op_walk_price_it(self):
        name = "Qwen__Qwen3-Coder-Next"
        result = estimate_capacity(
            config=_config(name),
            gpu_sku="H200", resolved=_resolved(name),
            selection_mode="theoretical_only", framework="sglang",
            avg_input_tokens=1024, avg_output_tokens=64,
            parallel=ParallelConfig(pp=2))
        self.assertIsNotNone(result["memory"]["pp_stage"])
        from backend.inference.framework_bindings import get_binding
        rows = _pipeline.analyze(
            _resolved(name), parallel=ParallelConfig(pp=2), batch_size=1,
            avg_input_tokens=128, avg_output_tokens=0,
            framework_binding=get_binding("sglang"), breakdown=False).per_op_timing
        self.assertTrue(any(r.get("comm_op") == "send_recv" for r in rows))


class ThePipelineClock(unittest.TestCase):
    """pp>1 publishes the pipeline-step time: Σ every stage's roofline plus
    the inter-stage hops, with the fill/drain bubble deliberately unmodelled
    (a steady-state throughput artifact, not a roofline term)."""

    def test_pp_1_publishes_its_time(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase):
                p = _estimate("meta-llama__Llama-3.2-3B", 1)["phases"][phase]
                self.assertIsNotNone(p["t_sol_s"])
                self.assertGreater(p["t_sol_s"], 0)
                self.assertEqual(p["time_unpublished_reason"], "")

    def test_pp_8_publishes_the_pipeline_sum_and_says_what_it_is(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase):
                p = _estimate("meta-llama__Llama-3.2-3B", 8)["phases"][phase]
                for field in ("t_compute_s", "t_memory_s", "t_comm_s",
                              "t_sol_s"):
                    self.assertIsNotNone(p.get(field), field)
                    self.assertGreater(p[field], 0, field)
                # Measured coverage is per-stage; a full-model measured
                # number would mix bases, so it stays unpublished.
                self.assertIsNone(p["t_measured_s"])
                self.assertEqual(p["time_unpublished_reason"], "")
                pt = p["pipeline_time"]
                self.assertEqual(pt["stages"], 8)
                self.assertEqual(p["t_sol_s"], pt["t_sol"])
                self.assertIn("bubble is not modelled",
                              p["formulas"]["t_sol"]["note"])

    def test_the_capacity_account_beside_it_is_unaffected(self):
        """Publishing the pipeline clock changes nothing per-rank.

        fits / max_tokens / weights / KV are per-rank questions with per-rank
        answers, and `pp` is exactly the knob that improves them.
        """
        one = _estimate("meta-llama__Llama-3.2-3B", 1)
        many = _estimate("meta-llama__Llama-3.2-3B", 8)
        self.assertLess(many["weight_breakdown"]["total_bytes_per_rank"],
                        one["weight_breakdown"]["total_bytes_per_rank"])
        self.assertLess(many["kv_cache"]["per_request_kv_bytes"],
                        one["kv_cache"]["per_request_kv_bytes"])
        self.assertGreater(many["phases"]["decode"]["batch"],
                           one["phases"]["decode"]["batch"])

    def test_folding_the_per_op_rows_does_not_put_the_time_back(self):
        """Without a pipeline sum the old rule still holds: the phase folds
        the per-op rows for its bytes, but a stage's fold is not published
        as the model's step time."""
        from tests.snapshots import shared_loader
        from backend.analyzer import fetch_config
        cfg = fetch_config("meta-llama/Llama-3.2-3B")
        spec = shared_loader().match(cfg, model_id="meta-llama/Llama-3.2-3B")
        rows = [{"access_bytes_per_rank": 10, "t_compute_s": 1.0,
                 "t_memory_s": 2.0, "t_comm_s": 0.5, "t_launch_s": 0.0,
                 "t_op_s": 2.0,
                 "t_measured_s": None}]
        base = dict(avg_input_tokens=128, avg_output_tokens=8,
                    weight_bytes_per_rank=1e6, per_token_kv_bytes_per_rank=0,
                    static_kv_bytes_per_rank=0, resolved_spec=spec)
        kw = dict(phase="decode", batch=1, graph_rows=rows)

        withheld = estimate_phase(PhaseInputs(stage=0, **base), **kw)
        self.assertIsNone(withheld.t_sol_s)
        self.assertIn("pipeline parallelism",
                      withheld.formulas["t_sol"]["note"])
        # And the memory side of the same call still lands.
        self.assertEqual(withheld.mem_read_bytes, 10)

        published = estimate_phase(PhaseInputs(**base), **kw)
        self.assertEqual(published.t_sol_s, 2.0)
        self.assertEqual(published.mem_read_bytes, 10)

        # With the sums supplied, the same fold publishes the pipeline model.
        piped = estimate_phase(
            PhaseInputs(stage=0, **base),
            pipeline_time={"t_compute": 4.0, "t_memory": 6.0, "t_comm": 1.0,
                           "t_sol": 8.0, "stages": 2,
                           "t_sol_by_bound": {"compute": 2.0, "memory": 5.0,
                                              "comm": 1.0},
                           # Every stage's rows; only stage 0's were keyed.
                           "stage_laws": [[{}], [{}, {}]]},
            **kw)
        self.assertEqual(piped.t_sol_s, 8.0)
        self.assertEqual(sum(piped.t_sol_by_bound.values()), piped.t_sol_s)
        self.assertIsNone(piped.t_measured_s)
        # Nothing is read here: the phase states an unread track over every
        # term of the step, the other stage's two included, and the measured
        # fold names that stage `unplaced` (`batch_sweep.step_measure`).
        self.assertEqual(piped.coverage, {"measured": 0, "total": 3})
        self.assertIn("bubble is not modelled",
                      piped.formulas["t_sol"]["note"])

    def test_phase_closure_reports_the_pipeline_total_instead(self):
        """At pp>1 the displayed total is the Σ-stages model; the bound
        stage's rows close against their own per_stage entry, and the
        per-stage sums plus the draft close against the published total."""
        res = _estimate("meta-llama__Llama-3.2-3B", 8)
        results = _inv_phase_closure(res)
        self.assertTrue(results)
        for r in results:
            self.assertTrue(r.passed, r.detail)
            self.assertIn("Σ stages + draft", r.detail)

    def test_pp_time_is_the_pp1_time_plus_the_hops(self):
        """Pinned batch: with no TP, the only comm rows are the pipeline
        hops, so the pp=2 step must equal the single-rank step plus their
        transfer time and the cut's one extra launch."""
        from tests.snapshots import shared_loader
        from backend.analyzer import fetch_config
        from backend.inference.capacity import estimate_capacity
        from backend.inference.parallel_config import ParallelConfig
        cfg = fetch_config("meta-llama/Llama-3.2-3B")
        spec = shared_loader().match(cfg, model_id="meta-llama/Llama-3.2-3B")
        kw = dict(config=cfg, resolved=spec, gpu_sku="H100-SXM",
                  avg_input_tokens=1024, avg_output_tokens=128,
                  decode_batch_size=64,
                  selection_mode="theoretical_only")
        one = estimate_capacity(**kw, num_gpus=1, parallel=ParallelConfig())
        two = estimate_capacity(**kw, num_gpus=2, parallel=ParallelConfig(pp=2))
        t1 = one["phases"]["decode"]["t_sol_s"]
        p2 = two["phases"]["decode"]
        hops = p2["pipeline_time"]["t_comm"]
        launch_delta = (
            p2["t_sol_by_bound"]["launch"]
            - one["phases"]["decode"]["t_sol_by_bound"]["launch"]
        )
        self.assertGreater(hops, 0)
        self.assertAlmostEqual(p2["t_sol_s"], t1 + hops + launch_delta,
                               delta=1e-6 * max(t1, 1e-9))


class TheAbsenceTravels(unittest.TestCase):
    """Downstream of the phase, `or 0.0` is how a refusal becomes a number."""

    def test_disaggregation_reports_absence_rather_than_zero(self):
        """The rate match divides a decode wall-time by a prefill one. With
        `or 0.0` the prefill term is zero, the ratio is a division by zero,
        and `prefill_time_per_request_s: 0.0` is a claim of instantaneous
        prefill. The answer is absent instead, and says whose time was
        missing and why.
        """
        from backend.inference.disagg import disagg_handoff

        gpu = {"rdma_gbps": 400.0}
        withheld = {
            "kv_cache": {"per_request_kv_bytes": 1 << 30}, "gpu": gpu,
            "phases": {
                "prefill": {"t_sol_s": None,
                            "time_unpublished_reason": "because pp"},
                "decode": {"t_sol_s": None,
                           "time_unpublished_reason": "because pp"}}}
        rm = disagg_handoff(
            withheld, withheld,
            kv_transfer_bw_gbps=None, avg_output_tokens=256)["rate_match"]
        self.assertIsNone(rm["prefill_time_per_request_s"])
        self.assertIsNone(rm["decode_time_per_request_s"])
        self.assertIsNone(rm["decode_to_prefill_time_ratio"])
        self.assertIn("because pp", rm["unavailable_reason"])

        published = {
            "kv_cache": {"per_request_kv_bytes": 1 << 30}, "gpu": gpu,
            "phases": {"prefill": {"t_sol_s": 0.5},
                       "decode": {"t_sol_s": 0.01}}}
        rm = disagg_handoff(
            published, published,
            kv_transfer_bw_gbps=None, avg_output_tokens=256)["rate_match"]
        self.assertAlmostEqual(rm["decode_time_per_request_s"], 2.56)
        self.assertAlmostEqual(rm["decode_to_prefill_time_ratio"], 5.12)
        self.assertEqual(rm["unavailable_reason"], "")
        # The KV handoff is a bytes-over-bandwidth question and has an answer
        # either way — withholding a time is not withholding the transfer.
        self.assertGreater(
            disagg_handoff(
            withheld, withheld,
            kv_transfer_bw_gbps=None, avg_output_tokens=256)
            ["kv_transfer"]["handoff_latency_s"], 0)

    def test_kv_handoff_is_bytes_over_decimal_gbs(self):
        """CSV `rdma_gbps` is GB/s, so handoff = bytes / (gbps · 1e9).

        A `/8` (treating the column as Gbit/s) made this 8× too slow.
        NVLink is intra-node and is not the default handoff fabric.
        """
        from backend.inference.disagg import disagg_handoff

        kv = 8e9
        gbps = 50.0
        stub = {
            "kv_cache": {"per_request_kv_bytes": kv},
            "gpu": {"rdma_gbps": gbps},
            "phases": {"prefill": {"t_sol_s": 1.0}, "decode": {"t_sol_s": 1.0}},
        }
        kv_xfer = disagg_handoff(
            stub, stub, kv_transfer_bw_gbps=None, avg_output_tokens=1,
        )["kv_transfer"]
        self.assertAlmostEqual(kv_xfer["handoff_latency_s"], kv / (gbps * 1e9))
        self.assertEqual(kv_xfer["link_bw_gbps"], gbps)

        over = disagg_handoff(
            stub, stub, kv_transfer_bw_gbps=25.0, avg_output_tokens=1,
        )["kv_transfer"]
        self.assertAlmostEqual(over["handoff_latency_s"], kv / (25.0 * 1e9))

        # No RDMA column: the scale-up link and PCIe are intra-node fabrics and never
        # carry a cross-node handoff, so there is no latency to report.
        no_rdma = {
            **stub,
            "gpu": {"scaleup_gbps": 450.0, "scaleup_fabric": "NVLink",
                    "pcie_gbps": 64.0},
        }
        absent = disagg_handoff(
            stub, no_rdma, kv_transfer_bw_gbps=None, avg_output_tokens=1,
        )["kv_transfer"]
        self.assertIsNone(absent["handoff_latency_s"])
        # ... and no link of 0 GB/s either: the bandwidth is absent, with
        # the column that would state it named beside it.
        self.assertIsNone(absent["link_bw_gbps"])
        self.assertIn("rdma_gbps", absent["unavailable_reason"])
        self.assertEqual(kv_xfer["unavailable_reason"], "")


class WhichLinkAHandoffCrosses(unittest.TestCase):
    """The prefill and decode clusters are separate machines, so the KV
    handoff always leaves the chassis. On an Ascend supernode it does not
    thereby leave the scale-up fabric: 768 dies reach each other over HCCS
    across machines. The handoff used to be priced on `rdma_gbps` wherever
    it happened, which put a transfer between two machines of one
    supernode on the NIC that only carries traffic LEAVING the supernode --
    25 GB/s where the link is 196."""

    def _cluster(self, sku, ranks, **gpu):
        from backend.inference.gpu_specs import get_gpu
        return {"kv_cache": {"per_request_kv_bytes": 8e9},
                "gpu": {"sku": sku, "rdma_gbps": get_gpu(sku).rdma_gbps,
                        **gpu},
                "cluster_size": ranks,
                "phases": {"prefill": {"t_sol_s": 1.0},
                           "decode": {"t_sol_s": 1.0}}}

    def _handoff(self, prefill, decode, bw=None):
        from backend.inference.disagg import disagg_handoff
        return disagg_handoff(prefill, decode, kv_transfer_bw_gbps=bw,
                              avg_output_tokens=1)["kv_transfer"]

    def test_two_machines_of_one_supernode_stay_on_scale_up(self):
        a = "Ascend-910C-A3"
        kv = self._handoff(self._cluster(a, 16), self._cluster(a, 16))
        self.assertEqual((kv["link"], kv["link_bw_gbps"]), ("HCCS", 196.0))
        self.assertAlmostEqual(kv["handoff_latency_s"],
                               kv["per_request_kv_bytes"] / 196e9)
        self.assertEqual(kv["unavailable_reason"], "")

    def test_the_domain_is_filled_in_whole_machines(self):
        """768 dies are 48 chassis and a chassis is not shared between the
        two clusters, so each is counted at the machines it occupies: 753
        dies take 48 of them and leave no room for a 49th, though 753 + 1
        is under 768."""
        a = "Ascend-910C-A3"
        for prefill, decode, inside in ((752, 16, True), (16, 752, True),
                                        (384, 384, True), (753, 1, False),
                                        (768, 1, False), (385, 385, False),
                                        (1, 768, False)):
            kv = self._handoff(self._cluster(a, prefill),
                               self._cluster(a, decode))
            with self.subTest(prefill=prefill, decode=decode):
                if inside:
                    self.assertEqual(kv["link_bw_gbps"], 196.0)
                else:
                    self.assertEqual(kv["link"], "RDMA")
                    self.assertNotEqual(kv["link_bw_gbps"], 196.0)

    def test_a_card_with_no_supernode_hands_off_over_its_nic(self):
        """Two TP4 clusters on 8-GPU H200 nodes are 8 ranks, which is one
        node's worth, and are still two machines: NVLink does not join
        them. A rule that only asked whether the ranks fit the domain would
        put this handoff on 450 GB/s."""
        from backend.inference.gpu_specs import get_gpu, load_gpu_specs
        kv = self._handoff(self._cluster("H200", 4), self._cluster("H200", 4))
        self.assertEqual((kv["link"], kv["link_bw_gbps"]),
                         ("RDMA", get_gpu("H200").rdma_gbps))
        for gpu in load_gpu_specs():
            if gpu.sku == "Ascend-910C-A3":
                continue
            for ranks in (1, 4, 36):
                kv = self._handoff(self._cluster(gpu.sku, ranks),
                                   self._cluster(gpu.sku, ranks))
                with self.subTest(sku=gpu.sku, ranks=ranks):
                    self.assertEqual((kv["link"], kv["link_bw_gbps"]),
                                     ("RDMA", gpu.rdma_gbps))

    def test_clusters_on_different_cards_share_no_scale_up_fabric(self):
        kv = self._handoff(self._cluster("H200", 8),
                           self._cluster("Ascend-910C-A3", 16))
        self.assertEqual(kv["link"], "RDMA")

    def test_a_link_the_card_does_not_state_is_absent_not_zero(self):
        """Past the supernode the handoff needs the NIC. With that column
        blank there is nothing to divide by: no latency, no 0 GB/s link, no
        exception -- and the reason names the column."""
        a = "Ascend-910C-A3"
        kv = self._handoff(self._cluster(a, 768, rdma_gbps=None),
                           self._cluster(a, 16, rdma_gbps=None))
        self.assertIsNone(kv["handoff_latency_s"])
        self.assertIsNone(kv["link_bw_gbps"])
        self.assertIn("rdma_gbps", kv["unavailable_reason"])
        self.assertIn("kv_transfer_bw_gbps", kv["unavailable_reason"])
        # Inside the supernode that column is never read.
        kv = self._handoff(self._cluster(a, 16, rdma_gbps=None),
                           self._cluster(a, 16, rdma_gbps=None))
        self.assertEqual(kv["link_bw_gbps"], 196.0)

    def test_a_stated_link_is_the_link(self):
        """The request's `kv_transfer_bw_gbps` outranks the placement both
        ways: a measured 81.8 GB/s replaces the 196 GB/s rating inside the
        supernode, and a stated link prices a handoff whose card states
        none."""
        a = "Ascend-910C-A3"
        kv = self._handoff(self._cluster(a, 16), self._cluster(a, 16),
                           bw=81.8)
        self.assertEqual((kv["link"], kv["link_bw_gbps"]), ("stated", 81.8))
        self.assertAlmostEqual(kv["handoff_latency_s"],
                               kv["per_request_kv_bytes"] / 81.8e9)
        kv = self._handoff(self._cluster(a, 768, rdma_gbps=None),
                           self._cluster(a, 16, rdma_gbps=None), bw=25.0)
        self.assertAlmostEqual(kv["handoff_latency_s"],
                               kv["per_request_kv_bytes"] / 25e9)
        self.assertEqual(kv["unavailable_reason"], "")


class OneStageNamedOnce(unittest.TestCase):
    """Memory and compute answer for the same rank."""

    @pytest.mark.regression  # minutes-long
    def test_the_phase_quotes_the_stage_the_memory_was_sized_on(self):
        """Not a ratio test — a hybrid stack breaks every ratio identity,
        because the cache-heaviest stage holds more than its share of
        full-attention layers and its parameter count says nothing about
        that. The claim is the literal one: the stage the phase quotes back
        in its refusal is the stage the ceilings layer bound
        (`CapacityCeilings.stage`), which is also what the published weight
        tree and the cache reader were restricted by.
        """
        for name in _MODELS:
            with self.subTest(name):
                prep = _capacity_preparation(
                    config=_config(name), gpu_sku="H200",
                    knobs=CapacityKnobs(
                        resolved=_resolved(name),
                        selection_mode="theoretical_only",
                        parallel=_pp_parallel(name, 8)))
                ceil = capacity_ceilings(prep, avg_input_tokens=8192,
                                         avg_output_tokens=256)
                self.assertIsNotNone(ceil.stage)
                phases = _estimate(name, 8)["phases"]
                for phase in ("prefill", "decode"):
                    p = phases[phase]
                    if p["t_sol_s"] is None:
                        continue  # no admitted batch: the phase never ran
                    # The stage evidence moved with the published clock:
                    # the bound stage rides pipeline_time.stage.
                    self.assertEqual(p["pipeline_time"]["stage"],
                                     ceil.pp_stage)


if __name__ == "__main__":
    unittest.main()
