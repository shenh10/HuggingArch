"""Parallelism-scaling sweep — config enumeration + per-GPU throughput.

The Scaling panel's DP-EP / CP-EP / TP-EP / TP-TP charts sweep the deployment
degree X (num_gpus == X, dp_replicas=1), one series per GPU type, reporting
per-GPU throughput. Every enumerated config must be `validate()`-clean, an EP
degree must divide the experts, and whether the model deploys at X is the
layout's answer; EP modes must be empty for dense models.
"""
from __future__ import annotations

from typing import ClassVar

import json
import sys
import unittest

import pytest
from unittest import mock
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.phases import phase_throughput
from backend.inference.parallel_sweep import (
    ALL_MODES, EP_MODES, parallel_configs_for_mode, sweep_caps, is_moe,
    estimate_parallel_sweep, _per_gpu_tput, legal_parallel_degrees,
    layout_verdict, frontier_layouts, layout_label, config_for_degree,
    decode_frontier, SweepCaps,
)
from backend.inference.sweep_grid import SWEEP_MAX_GPUS
from backend.inference.card import Card
from backend.inference.parallel_config import ParallelConfig

SNAP = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _cfg(name):
    return json.loads((SNAP / name).read_text())["config"]


class TestConfigEnumeration(unittest.TestCase):
    def setUp(self):
        self.dsv3 = _cfg("deepseek-ai__DeepSeek-V3.json")
        self.dims = sweep_caps(self.dsv3, model_id="deepseek-ai/DeepSeek-V3")  # E=256 copies, 128 heads

    def test_every_config_valid_and_num_gpus_equals_x(self):
        for mode in ALL_MODES:
            for x, cfg in parallel_configs_for_mode(mode, self.dims):
                self.assertEqual(cfg.validate(), [], f"{mode}@{x} invalid: {cfg}")
                self.assertEqual(cfg.cluster_size, x,
                                 f"{mode}@{x}: num_gpus {cfg.cluster_size} != {x}")
                self.assertEqual(cfg.dp_replicas, 1)

    def test_ep_modes_cap_at_num_experts(self):
        # E=256 → DP-EP X tops out at 256.
        xs = [x for x, _ in parallel_configs_for_mode("dp_ep", self.dims)]
        self.assertEqual(max(xs), 256)
        self.assertTrue(all(x <= 256 for x in xs))

    def test_tensor_parallel_degrees_run_to_the_enumeration_bound(self):
        # Heads do not cap the ladder: past them the layout replicates, and
        # which degrees deploy is the layout's answer. TP-TP runs to the
        # bound; TP-EP still stops where the 256 expert copies stop dividing.
        xs = [x for x, _ in parallel_configs_for_mode("tp_tp", self.dims)]
        self.assertEqual(max(xs), SWEEP_MAX_GPUS)
        xs = [x for x, _ in parallel_configs_for_mode("tp_ep", self.dims)]
        self.assertEqual(max(xs), 256)

    def test_dp_ep_layout(self):
        _, cfg = parallel_configs_for_mode("dp_ep", self.dims)[-1]
        self.assertEqual((cfg.dp_attention, cfg.ep_moe, cfg.tp_attention, cfg.cp),
                         (256, 256, 1, 1))


class _Pool:
    def __init__(self, copies):
        self.n_expert_copies = copies


class TestFrontierLayouts(unittest.TestCase):
    """The deployments a decode frontier prices (`frontier_layouts`)."""

    def test_a_moe_splits_attention_tp_within_the_node_and_dp_across(self):
        caps = SweepCaps(moe_enabled=True, moe_pools=(_Pool(256),))
        labels = [layout_label(c) for c in frontier_layouts(caps, "B300", x_max=16)]
        self.assertEqual(labels, [
            "TP1",
            "EP2·DP2", "TP2·EP2", "CP2·EP2", "TP2",
            "EP4·DP4", "TP2·EP4·DP2", "TP4·EP4", "CP4·EP4", "TP4",
            "EP8·DP8", "TP2·EP8·DP4", "TP4·EP8·DP2", "TP8·EP8", "CP8·EP8", "TP8",
            "EP16·DP16", "TP2·EP16·DP8", "TP4·EP16·DP4", "TP8·EP16·DP2",
            "CP16·EP16", "TP16"])

    def test_its_ends_are_the_scaling_panels_dp_ep_and_tp_ep(self):
        caps = SweepCaps(moe_enabled=True, moe_pools=(_Pool(256),))
        layouts = frontier_layouts(caps, "B300", x_max=8)
        for mode in ("dp_ep", "tp_ep", "cp_ep", "tp_tp"):
            self.assertIn(config_for_degree(mode, 8), layouts, mode)
        for cfg in layouts:
            self.assertEqual(cfg.validate(), [])

    def test_a_dense_model_is_tensor_parallel_only(self):
        labels = [layout_label(c) for c in frontier_layouts(
            SweepCaps(moe_enabled=False), "B300", x_max=64)]
        self.assertEqual(labels, ["TP1", "TP2", "TP4", "TP8", "TP16", "TP32", "TP64"])

    def test_attention_tp_stops_at_the_gpus_node(self):
        caps = SweepCaps(moe_enabled=True, moe_pools=(_Pool(256),))
        tps = {g: max(c.tp_attention for c in frontier_layouts(caps, g, x_max=64)
                      if c.ep_moe > 1 and c.cp == 1)
               for g in ("B300", "GB200")}
        self.assertEqual(tps, {"B300": 8, "GB200": 64})

    def test_an_ep_degree_that_leaves_a_pool_fractional_is_skipped(self):
        caps = SweepCaps(moe_enabled=True, moe_pools=(_Pool(12),))
        eps = {c.ep_moe for c in frontier_layouts(caps, "B300", x_max=16)}
        self.assertEqual(eps, {1, 2, 4})


class TestTheLayoutDecidesTheLadder(unittest.TestCase):
    """A degree deploys when the layout walk an estimate starts from accepts
    it; a refused degree carries the layout's reason, and a degree past the
    enumeration bound is a different answer from a refused one."""

    def _resolved(self, stem):
        from tests.snapshots import shared_loader
        return shared_loader().match(_cfg(f"{stem}.json"),
                                     model_id=stem.replace("__", "/", 1))

    def test_cp_ep_runs_the_default_scheme_uncut_by_heads(self):
        """CP-EP runs the default prefill scheme, pcp, which gathers context
        rather than repartitioning heads: Qwen3-30B's 32 query heads bound
        nothing, and the ladder runs to its 128 expert copies. (A head
        repartition's limit is pinned in test_legal_cp / test_attention_plan_cp.)"""
        ladder = legal_parallel_degrees(
            self._resolved("Qwen__Qwen3-30B-A3B"), "cp_ep")
        self.assertEqual([r["x"] for r in ladder["accepted"]],
                         [2, 4, 8, 16, 32, 64, 128])
        self.assertEqual(ladder["refused"], [])
        # EP stops at the 128 copies, inside the bound: nothing past it.
        self.assertIsNone(ladder["beyond_cap"])

    def test_a_tp_degree_past_an_axis_deploys_with_its_replication_named(self):
        """The step's layout accepts a TP degree that some axis no longer
        divides, and names each such axis: its ranks hold copies."""
        for stem in ("Qwen__Qwen3-VL-235B-A22B-Instruct",
                     "moonshotai__Kimi-K2.5"):
            ladder = legal_parallel_degrees(self._resolved(stem), "tp_tp")
            with self.subTest(model=stem):
                at = {r["x"]: r for r in ladder["accepted"]}
                self.assertIn(32, at)
                self.assertTrue(at[32]["warnings"])
                # Said in the deployment's words (af468c39b): what the GPUs
                # hold instead of the split asked for.
                for warning in at[32]["warnings"]:
                    self.assertRegex(warning, "hold copies|keeps all of")
                self.assertEqual(ladder["beyond_cap"], {"cap": SWEEP_MAX_GPUS})

    def test_the_sampled_positions_are_not_a_cp_replication_warning(self):
        """`s_sampled` is `whole` in dims.yaml: the sampled rows live on the
        rank that owns them, no degree cuts them, so a CP degree is never
        "asked to split" them and an accepted one carries no warning."""
        ladder = legal_parallel_degrees(
            self._resolved("deepseek-ai__DeepSeek-V3"), "cp_ep")
        self.assertTrue(ladder["accepted"])
        for row in ladder["accepted"]:
            with self.subTest(x=row["x"]):
                self.assertEqual(row["warnings"], [])

    def test_a_degree_past_the_heads_replicates_and_says_so(self):
        from backend.inference.parallel_sweep import config_for_degree
        warnings = layout_verdict(self._resolved("moonshotai__Kimi-K3"),
                                  config_for_degree("tp_tp", 32))
        self.assertTrue(any("hold copies" in w or "keeps all of" in w
                            for w in warnings), warnings)


class TestMoEGating(unittest.TestCase):
    def test_dense_is_not_moe_ep_modes_empty(self):
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        self.assertFalse(is_moe(llama, model_id="meta-llama/Llama-3.2-3B"))
        env = _base_env(llama, "meta-llama/Llama-3.2-3B")
        for mode in EP_MODES:
            r = estimate_parallel_sweep(mode=mode, context_len=1024,
                                        gpu_skus=["B300"], base_env=env)
            self.assertFalse(r["moe_enabled"])
            self.assertEqual(r["series"], [])

    def test_dense_tp_tp_runs(self):
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        env = _base_env(llama, "meta-llama/Llama-3.2-3B")
        r = estimate_parallel_sweep(mode="tp_tp", context_len=1024,
                                    gpu_skus=["B300"], base_env=env)
        pts = [p for p in r["series"][0]["points"] if p["feasible"]]
        self.assertTrue(pts)
        self.assertTrue(all(p["decode_tps_per_gpu"] > 0 for p in pts))

    def test_fixed_batch_is_honored_not_silently_max(self):
        # Regression: a pinned per-GPU batch must NOT return the same per-GPU
        # throughput as "max" — _stamp_gpu_on_result used to rescale every point
        # to the capacity max, ignoring the requested batch.
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        kw = dict(mode="tp_tp", context_len=1024, gpu_skus=["B300"])
        r_max = estimate_parallel_sweep(
            batch="max", base_env=_base_env(llama, "meta-llama/Llama-3.2-3B"), **kw)
        r_one = estimate_parallel_sweep(
            batch=1, base_env=_base_env(llama, "meta-llama/Llama-3.2-3B"), **kw)

        def first_decode(r):
            pts = [p for p in r["series"][0]["points"] if p["feasible"]]
            self.assertTrue(pts)
            return pts[0]["decode_tps_per_gpu"]
        self.assertNotEqual(first_decode(r_max), first_decode(r_one))


class TestPerGpuThroughput(unittest.TestCase):
    _DECODE: ClassVar[dict] = {"t_sol_s": 0.05,
               "binding": {"effective_tokens_per_step": 1000,
                           "effective_tokens_formula": {"symbolic": "B × 1",
                                                        "substituted": "1000 × 1"}}}

    def test_per_gpu_formula(self):
        # per-GPU = (effective_tokens / t_sol) × utm / GPUs, stated once by
        # `phase_throughput` — the payload the sweep reads.
        rates = phase_throughput(self._DECODE, unique_tokens_multiplier=8, num_gpus=8)
        self.assertAlmostEqual(rates["sol"]["per_gpu"], (1000 / 0.05) * 8 / 8)
        # OOM phase → None.
        self.assertIsNone(phase_throughput({"t_sol_s": 0}, unique_tokens_multiplier=1,
                                           num_gpus=1)["sol"])

    def test_a_phase_without_its_payload_has_no_rate(self):
        # No second formula in the sweep: a phase that carries no
        # `throughput` payload is not recomputed there.
        self.assertIsNone(_per_gpu_tput({"phases": {"decode": dict(self._DECODE)}}, "decode"))

    def test_attached_throughput_is_not_recomputed(self):
        r = {"phases": {"decode": {
                "t_sol_s": 0.05,
                "binding": {"effective_tokens_per_step": 1000},
                "throughput": {"sol": {"per_gpu": 123.0}},
             }},
             "unique_tokens_multiplier": 8, "num_gpus": 8}
        self.assertEqual(_per_gpu_tput(r, "decode"), 123.0)

    def test_point_measured_track_is_never_the_sol_value(self):
        """Measured is a track beside theory: a phase with a SOL rate and no
        measured step time ships a SOL value and a None measured value."""
        from backend.inference.parallel_sweep import _point
        r = {"phases": {"decode": {"t_sol_s": 0.05, "t_measured_s": None,
                                   "throughput": {"sol": {"per_gpu": 123.0},
                                                  "measured": None}}},
             "max_decode_batch_per_gpu": 4}
        pt = _point(8, ParallelConfig(), r)
        self.assertEqual(pt["decode_tps_per_gpu"], 123.0)
        self.assertIsNone(pt["decode_tps_per_gpu_measured"])
        self.assertIsNone(pt["decode_t_measured_s"])

    def test_infeasible_point_carries_the_capacity_reason(self):
        from backend.inference.parallel_sweep import _point
        reason = "weights exceed usable HBM."
        pt = _point(2, ParallelConfig(), {"phases": {}, "max_decode_batch_per_gpu": 0,
                        "infeasibility_reason": reason})
        self.assertFalse(pt["feasible"])
        self.assertEqual(pt["reason"], reason)

    @pytest.mark.regression  # minutes-long
    def test_dsv3_dp_ep_series_shape(self):
        env = _base_env(_cfg("deepseek-ai__DeepSeek-V3.json"), "deepseek-ai/DeepSeek-V3")
        r = estimate_parallel_sweep(mode="dp_ep", context_len=1024,
                                    gpu_skus=["B300"], base_env=env)
        self.assertTrue(r["moe_enabled"])
        feas = [p for p in r["series"][0]["points"] if p["feasible"]]
        self.assertTrue(feas)
        # per-GPU decode throughput rises off the OOM floor then declines at scale.
        decs = [p["decode_tps_per_gpu"] for p in feas]
        self.assertEqual(max(decs), decs[decs.index(max(decs))])  # has a peak
        self.assertLess(decs[-1], max(decs))                       # declines by the top
        # The OOM floor says why, in the capacity layer's words: the
        # plain sentence it composes while the weights, the static KV and
        # the card's usable bytes are still in scope (`capacity`), which
        # the page shows as it is.
        oom = [p for p in r["series"][0]["points"] if not p["feasible"]]
        self.assertTrue(oom)
        for point in oom:
            self.assertRegex(point["reason"],
                             r"^weights \(.* GiB\).* exceed usable HBM "
                             r"\(.* GiB\) on B300 at this parallel config\. "
                             r"Raise TP/EP or use a smaller-weight model\.$")

    @pytest.mark.regression  # minutes-long
    def test_points_carry_hover_facts_batch_and_step_time(self):
        """Each point ships the batch its phase ran at and its step times —
        the hover tooltip says why per-GPU throughput saturates (B and t grow
        together at the max-batch operating point) instead of a stuck number."""
        env = _base_env(_cfg("deepseek-ai__DeepSeek-V3.json"), "deepseek-ai/DeepSeek-V3")
        r = estimate_parallel_sweep(mode="dp_ep", context_len=1024,
                                    gpu_skus=["B300"], base_env=env)
        feas = [p for p in r["series"][0]["points"] if p["feasible"]]
        self.assertTrue(feas)
        for p in feas:
            self.assertGreater(p["decode_batch_per_gpu"], 0)
            self.assertGreater(p["decode_t_sol_s"], 0)
            # dp_ep: per-GPU rate × step time == the per-GPU batch shipped.
            self.assertAlmostEqual(p["decode_tps_per_gpu"] * p["decode_t_sol_s"],
                                   p["decode_batch_per_gpu"], places=3)


class TestMeasuredTrackOnARealEstimate(unittest.TestCase):
    """With no measurement in play (theoretical-only lookup), a real estimate
    has no measured time anywhere: the phases, their throughput, and the
    sweep point all say None — never the SOL value under the measured name."""

    def test_theory_only_estimate_has_no_measured_track(self):
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.parallel_sweep import _estimate_layout, _point
        from tests.snapshots import shared_loader

        config = _cfg("Qwen__Qwen3-8B.json")
        resolved = shared_loader().match(config, model_id="Qwen/Qwen3-8B")
        stamped = _estimate_layout(dict(
            config=config, resolved=resolved, parallel=ParallelConfig(),
            num_gpus=1, avg_input_tokens=4096, avg_output_tokens=256,
            selection_mode="theoretical_only", gpu_skus=["H100-SXM"]))
        est = stamped["H100-SXM"]["sol"]
        for phase in ("prefill", "decode"):
            ph = est["phases"][phase]
            self.assertIsNotNone(ph["t_sol_s"])
            self.assertIsNone(ph["t_measured_s"], phase)
            self.assertEqual(ph["measured_rows"], 0)
            self.assertGreater(ph["total_rows"], 0)
            self.assertIsNone(ph["throughput"]["measured"], phase)
        pt = _point(1, ParallelConfig(), est)
        self.assertGreater(pt["decode_tps_per_gpu"], 0)
        self.assertIsNone(pt["decode_tps_per_gpu_measured"])
        self.assertIsNone(pt["prefill_tps_per_gpu_measured"])


class TestMultiModeBatch(unittest.TestCase):
    """Several axes in one call equal N single-mode calls, env resolved once."""

    def test_dense_batches_ep_skip_and_tp(self):
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        env = _base_env(llama, "meta-llama/Llama-3.2-3B")
        kw = dict(context_len=1024, gpu_skus=["B300"], base_env=env)
        multi = estimate_parallel_sweep(modes=["dp_ep", "tp_tp"], **kw)
        self.assertEqual(set(multi["by_mode"]), {"dp_ep", "tp_tp"})
        self.assertFalse(multi["by_mode"]["dp_ep"]["moe_enabled"])
        self.assertEqual(multi["by_mode"]["dp_ep"]["series"], [])
        single_tp = estimate_parallel_sweep(mode="tp_tp", **kw)
        self.assertEqual(multi["by_mode"]["tp_tp"]["series"], single_tp["series"])
        self.assertEqual(multi["by_mode"]["tp_tp"]["moe_enabled"], single_tp["moe_enabled"])
        # Single-mode callers still see the flat shape.
        self.assertEqual(single_tp["mode"], "tp_tp")
        self.assertTrue(single_tp["series"])

    def test_forwards_request_selection_mode(self):
        """The Scaling panel's Measured toggle reads t_measured_s; the sweep
        must honour the request's selection_mode instead of forcing theory."""
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        env = _base_env(llama, "meta-llama/Llama-3.2-3B")
        env["selection_mode"] = "latest"
        captured = []

        def fake(payload):
            captured.append(payload.get("selection_mode"))
            return {"__error__": "skip"}

        with mock.patch("backend.measure.pool.pool_enabled", return_value=False), \
             mock.patch("backend.inference.parallel_sweep._estimate_layout_safe", fake):
            estimate_parallel_sweep(mode="tp_tp", context_len=1024,
                                    gpu_skus=["B300"], base_env=env)
        self.assertTrue(captured)
        self.assertTrue(all(s == "latest" for s in captured))

    def test_unknown_mode_raises(self):
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        env = _base_env(llama, "meta-llama/Llama-3.2-3B")
        with self.assertRaises(ValueError):
            estimate_parallel_sweep(modes=["nope"], context_len=1024,
                                    gpu_skus=["B300"], base_env=env)

    def test_two_gpus_match_per_sku_singles(self):
        """Walk once per card class, restamp peaks within it — must equal a
        dedicated call per SKU. H100-SXM and B300 are different classes
        (compute capability 9.0 and 10.3), so this is also the check that
        neither card is priced off the other's walk."""
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        env = _base_env(llama, "meta-llama/Llama-3.2-3B")
        kw = dict(mode="tp_tp", context_len=1024, base_env=env)
        both = estimate_parallel_sweep(gpu_skus=["H100-SXM", "B300"], **kw)
        h100 = estimate_parallel_sweep(gpu_skus=["H100-SXM"], **kw)
        b300 = estimate_parallel_sweep(gpu_skus=["B300"], **kw)
        by = {s["gpu_sku"]: s["points"] for s in both["series"]}
        self.assertEqual(by["H100-SXM"], h100["series"][0]["points"])
        self.assertEqual(by["B300"], b300["series"][0]["points"])

    def test_points_past_the_published_context_say_so(self):
        # Every card's point, the restamped ones included, carries the
        # model's published context and whether this request runs past it.
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        env = _base_env(llama, "meta-llama/Llama-3.2-3B")
        for ctx, past in ((131072, False), (262144, True)):
            r = estimate_parallel_sweep(mode="tp_tp", context_len=ctx,
                                        gpu_skus=["H100-SXM", "B300"],
                                        base_env=env)
            pts = [p for s in r["series"] for p in s["points"]]
            self.assertTrue(pts)
            for p in pts:
                self.assertEqual(p["beyond_max_context"], past, (ctx, p["x"]))
                self.assertEqual(p["max_context_tokens"], 131072)


class TestDecodeFrontier(unittest.TestCase):
    """The Scaling panel's decode point cloud — every layout of the sweep at
    every per-GPU decode batch of its ladder, read off the layout's walk —
    and the Pareto frontiers drawn as views of it (`decode_frontier`)."""

    def test_a_view_holds_its_dimensions_and_fronts_each_group(self):
        def pt(gpu, mode, n, c, user, per_gpu):
            return {"mode": mode, "num_gpus": n, "gpu_sku": gpu,
                    "context": 1024, "batch_key": f"b{c}",
                    "draft_tokens": 0, "acceptance": 1.0,
                    "ep_load_factor": 1.0, "concurrent_requests": c,
                    "tok_s_per_user": user, "tok_s_per_gpu": per_gpu}
        cloud = [pt("B300", "dp_ep", 8, 64, 100.0, 50.0),
                 pt("B300", "dp_ep", 8, 128, 80.0, 90.0),
                 pt("B300", "dp_ep", 8, 256, 70.0, 85.0),   # dominated
                 pt("B300", "tp_tp", 8, 64, 120.0, 40.0),
                 pt("B300", "dp_ep", 16, 64, 130.0, 30.0),
                 pt("H200", "dp_ep", 8, 64, 60.0, 20.0)]
        v = decode_frontier(cloud, fix={"gpu_sku": ["B300"], "num_gpus": [8]},
                            group_by=["mode"])
        fronts = {s["group"]["mode"]: [(p["concurrent_requests"]) for p in s["frontier"]]
                  for s in v["series"]}
        self.assertEqual(fronts, {"dp_ep": [64, 128], "tp_tp": [64]})
        # A control offers what the OTHER held dimensions admit: GPU counts
        # B300 has, cards that have an 8-GPU layout.
        self.assertEqual(v["values"]["num_gpus"], [8, 16])
        self.assertEqual(v["values"]["gpu_sku"], ["B300", "H200"])
        self.assertEqual(v["values"]["batch_key"], ["b64", "b128", "b256"])
        # A dimension holds a set: every point at any of its values is drawn.
        both = decode_frontier(cloud, fix={"num_gpus": [8, 16], "mode": ["dp_ep"]},
                               group_by=["gpu_sku", "num_gpus"])
        self.assertEqual({(s["group"]["gpu_sku"], s["group"]["num_gpus"])
                          for s in both["series"]},
                         {("B300", 8), ("B300", 16), ("H200", 8)})
        with self.assertRaises(ValueError):
            decode_frontier(cloud, fix={"layout": ["x"]}, group_by=[])

    def test_a_frontier_point_is_the_sweep_pinned_at_its_batch(self):
        # A second route: the same layout swept with its batch pinned (an
        # independent `_point_batches` override) prices the same decode.
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        kw = dict(mode="tp_tp", context_len=1024, gpu_skus=["B300"])
        env = _base_env(llama, "meta-llama/Llama-3.2-3B")
        cloud = estimate_parallel_sweep(base_env=env, **kw)["decode_cloud"]
        (line,) = decode_frontier(cloud, fix={}, group_by=["gpu_sku"])["series"]
        front = line["frontier"]
        self.assertGreater(len(front), 2)
        for pt in front[:3]:
            pinned = estimate_parallel_sweep(base_env=env, batch=pt["batch"], **kw)
            (got,) = [p for p in pinned["series"][0]["points"] if p["x"] == pt["x"]]
            self.assertEqual(got["decode_batch_per_gpu"], pt["batch"])
            self.assertAlmostEqual(got["decode_tps_per_gpu"], pt["tok_s_per_gpu"], places=6)
            self.assertAlmostEqual(got["decode_t_sol_s"], pt["tpot_s"], places=12)

    def test_cloud_concurrency_and_per_layout_fronts(self):
        dsv3 = _cfg("deepseek-ai__DeepSeek-V3.json")
        env = _base_env(dsv3, "deepseek-ai/DeepSeek-V3")
        r = estimate_parallel_sweep(modes=["dp_ep", "tp_tp"], context_len=1024,
                                    gpu_skus=["B300"], base_env=env)
        cloud = r["decode_cloud"]
        self.assertEqual({p["mode"] for p in cloud}, {"dp_ep", "tp_tp"})
        for p in cloud:
            # A second route to the requests in flight: cluster tokens/s over
            # one request's tokens/s (plain decode, one token a step).
            self.assertAlmostEqual(
                p["concurrent_requests"],
                p["tok_s_per_gpu"] * p["num_gpus"] / p["tok_s_per_user"], places=6)
        # The card's envelope over every mode: each of its points is on its
        # own layout's front, and no point of the cloud beats one on both axes.
        (env_line,) = decode_frontier(cloud, fix={}, group_by=["gpu_sku"])["series"]
        envelope = env_line["frontier"]
        self.assertTrue(envelope)
        key = lambda p: (p["mode"], p["x"], p["batch"])
        per_layout = {key(p) for s in decode_frontier(
            cloud, fix={}, group_by=["mode", "num_gpus"])["series"]
            for p in s["frontier"]}
        self.assertTrue({key(p) for p in envelope} <= per_layout)
        for e in envelope:
            self.assertFalse(any(p["tok_s_per_user"] > e["tok_s_per_user"]
                                 and p["tok_s_per_gpu"] > e["tok_s_per_gpu"]
                                 for p in cloud), e)


def _base_env(config, model_id):
    return dict(
        config=config, gpu_sku="B300", gpu_memory_fraction=0.9,
        num_gpus=8, avg_output_tokens=256, parallel=None,
        selection_mode="theoretical_only", model_id=model_id,
    )


class TestSweepRobustness(unittest.TestCase):
    """A single invalid (mode, degree) point must not sink the whole sweep."""

    def test_a_card_that_cannot_time_the_format_drops_only_its_own_series(self):
        # NVFP4 W4A4 computes in FP4, which Hopper / Ada cards list no peak
        # for. Every layout used to fail on the series' first card and the
        # whole mode read "unavailable" ("TP does not apply to this model"),
        # though the Blackwell cards price it.
        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        env = {**_base_env(llama, "meta-llama/Llama-3.2-3B"),
               "precision": {"preset": "w_nvfp4_w4a4_kv_fp8"}}
        r = estimate_parallel_sweep(mode="tp_tp", context_len=1024,
                                    gpu_skus=["H100-SXM", "B300"], base_env=env)
        self.assertNotIn("unavailable", r)
        by = {s["gpu_sku"]: s["points"] for s in r["series"]}
        self.assertTrue(all("FP4" in p["error"] for p in by["H100-SXM"]))
        self.assertTrue(any(p["feasible"] and p["decode_tps_per_gpu"] > 0
                            for p in by["B300"]))

    def test_ep_degrees_divide_expert_copies(self):
        # Step-3.7 has 288 expert copies (= 32·9). EP degrees that do not divide
        # 288 (64/128/256) are not deployments and must be dropped from the
        # ladder, not attempted and crashed at layout time.
        caps = sweep_caps(_cfg("stepfun-ai__Step-3.7-Flash.json"),
                          model_id="stepfun-ai/Step-3.7-Flash")
        self.assertEqual([p.n_expert_copies for p in caps.moe_pools], [288])
        for mode in EP_MODES:
            xs = [x for x, _ in parallel_configs_for_mode(mode, caps)]
            self.assertTrue(all(288 % x == 0 for x in xs),
                            f"{mode}: {xs} carries a non-divisor of 288")
            self.assertNotIn(64, xs)   # 288 % 64 == 32
            self.assertNotIn(128, xs)

    def test_restamping_a_prepared_walk_onto_another_card_class_is_refused(self):
        """Within a card class a prepared walk restamps; across one it is
        refused, naming both classes.

        The pair matters in that order: the refusal is the property that was
        missing, and a silent wrong number is what it cost. H100-SXM and
        H200 are one class (compute capability 9.0, same peak columns) and
        B200 is another (10.0, which also lists FP4), so the same
        preparation answers both halves."""
        from backend.inference import capacity as cap

        llama = _cfg("meta-llama__Llama-3.2-3B.json")
        prep = cap._capacity_preparation(
            config=llama, gpu_sku="H100-SXM",
            knobs=cap.CapacityKnobs(model_id="meta-llama/Llama-3.2-3B"))
        with self.assertRaises(ValueError) as refused:
            cap._restamp_gpu(prep, Card.of("B200"))
        said = str(refused.exception)
        for part in ("H100-SXM", "B200", "compute_capability=9.0",
                     "compute_capability=10.0"):
            self.assertIn(part, said)
        # ... and the same walk still reads on its own class's other cards.
        self.assertEqual(cap._restamp_gpu(prep, Card.of("H200")).gpu_sku, "H200")

    def test_moving_a_walks_rows_onto_another_card_class_is_refused(self):
        """The OTHER road a walk takes to a card: its per-op rows, re-read
        at a point on that card (`capacity._point_rows`).

        `_restamp_gpu` checks the card a PREPARATION moves to; the rows
        moved here come from a `PendingEstimate`'s walk, which is a
        different fact, and the pair is what this refuses. Unchecked, it
        priced DeepSeek-V4-Flash's B200 prefill off an H100-SXM walk — 270
        rows where B200's own walk has 274, the routed activation at BF16
        where that card runs it in FP8. Both halves again: the refusal, and
        the re-pricing within one class that every sweep depends on."""
        from backend.arch.eval.analysis import Point
        from backend.inference import capacity as cap

        llama = _cfg("meta-llama__Llama-3.2-1B.json")
        knobs = cap.CapacityKnobs(model_id="meta-llama/Llama-3.2-1B")
        prep = cap._capacity_preparation(config=llama, gpu_sku="H100-SXM",
                                         knobs=knobs)
        pending = cap._capacity_pending(
            prep, avg_input_tokens=256, avg_output_tokens=16,
            prefill_batch_override=1, decode_batch_override=1, breakdown=False)
        base = pending.finish()
        batches = {"per_op_timing": 1, "per_op_timing_decode": 1,
                   "per_op_timing_draft": 1}
        other_class = cap._capacity_preparation(config=llama, gpu_sku="B200",
                                                knobs=knobs)
        with self.assertRaises(ValueError) as refused:
            cap._point_rows(base, other_class, batches, pending, Point({}),
                            breakdown=False)
        said = str(refused.exception)
        for part in ("H100-SXM", "B200", "compute_capability=9.0",
                     "compute_capability=10.0"):
            self.assertIn(part, said)
        moved = cap._point_rows(base, cap._restamp_gpu(prep, Card.of("H200")), batches,
                                pending, Point({}), breakdown=False)
        self.assertEqual(
            [len(moved[k]) for k in batches],
            [len(base.get(k) or []) for k in batches])

    def test_one_road_refuses_both_an_off_class_and_an_off_stage_card(self):
        """`PendingEstimate.priced_on` is the one road from a walk to a
        card's numbers, and it asks BOTH facts a walk is a function of.

        The class and the bound pipeline stage used to be refused on
        different roads — the class where the rows moved
        (`_point_rows`), the stage where the result was stamped
        (`_stamp_gpu_on_result`) — so a caller that took one road asked
        only one question. A walk names the card it was taken on
        (`walked_on`) and the stage its ceilings bound, so one method can
        ask both."""
        from backend.inference import capacity as cap
        from backend.inference.card import Card
        from backend.arch.eval.analysis import Point

        llama = _cfg("meta-llama__Llama-3.2-1B.json")
        knobs = cap.CapacityKnobs(model_id="meta-llama/Llama-3.2-1B")
        prep = cap._capacity_preparation(config=llama, gpu_sku="H100-SXM",
                                         knobs=knobs)
        pending = cap._capacity_pending(
            prep, avg_input_tokens=256, avg_output_tokens=16,
            prefill_batch_override=1, decode_batch_override=1, breakdown=False)
        other_class = cap._capacity_preparation(config=llama, gpu_sku="B200",
                                                knobs=knobs)
        # Its own class and its own stage: priced.
        same_class = cap._restamp_gpu(prep, Card.of("H200"))
        pending.priced_on(same_class)

        class OwnStage:
            pp_stage = pending.ceilings.pp_stage

            def refuse_unless_one_stage(self, rows_stage, _gpu_sku):
                return None

        pending.priced_on(same_class, OwnStage())
        # Another class: refused, naming both classes.
        with self.assertRaises(ValueError) as refused:
            pending.priced_on(other_class)
        for part in ("H100-SXM", "B200", "compute_capability=9.0",
                     "compute_capability=10.0"):
            self.assertIn(part, str(refused.exception))
        # The result-stamping road cannot bypass that check by handing in rows
        # directly: it takes the walk's IDENTITY beside the rows' laws.
        with self.assertRaises(ValueError) as stamped:
            cap._stamp_gpu_on_result(
                pending.finish(), other_class, pending.ceilings,
                pending.walked_on, batches={}, rows={}, walks={},
                point=Point({}), breakdown=False)
        self.assertIn("compute_capability=10.0", str(stamped.exception))
        # Its own class, another bound stage: refused too, on the same road.
        stage = pending.ceilings.pp_stage

        class OtherStage:
            pp_stage = 1 if stage != 1 else 2

            def refuse_unless_one_stage(self, rows_stage, gpu_sku):
                assert rows_stage != self.pp_stage
                raise ValueError(
                    f"{gpu_sku}: the rows to re-price are pipeline stage "
                    f"{rows_stage}'s, and this card's ceilings bind stage "
                    f"{self.pp_stage}")

        with self.assertRaises(ValueError) as off_stage:
            pending.priced_on(same_class, OtherStage())
        self.assertIn("pipeline stage", str(off_stage.exception))

    def test_a_result_with_no_prefill_rows_is_not_stamped_from_a_prefill_walk(self):
        """Small-batch Scaling can carry decode alone: its row lists say
        which phases exist, so the stamper must not ask every walk by name."""
        from backend.arch.eval.analysis import Point
        from backend.inference import capacity as cap

        llama = _cfg("meta-llama__Llama-3.2-1B.json")
        knobs = cap.CapacityKnobs(model_id="meta-llama/Llama-3.2-1B")
        prep = cap._capacity_preparation(config=llama, gpu_sku="H100-SXM",
                                         knobs=knobs)
        pending = cap._capacity_pending(
            prep, avg_input_tokens=256, avg_output_tokens=16,
            prefill_batch_override=1, decode_batch_override=1, breakdown=False)
        base = pending.finish()
        rows = {"per_op_timing": [], "per_op_timing_decode":
                base["per_op_timing_decode"], "per_op_timing_draft": []}
        walks = {"per_op_timing_decode": pending.walk_of_rows[
            "per_op_timing_decode"]}
        out = cap._stamp_gpu_on_result(
            base, prep, pending.ceilings, pending.walked_on,
            batches={"per_op_timing": 1, "per_op_timing_decode": 1,
                     "per_op_timing_draft": 1},
            rows=rows, walks=walks, point=Point({}), breakdown=False)
        self.assertEqual(list(out["phases"]), ["decode"])
        self.assertTrue(out["phases"]["decode"])

    def test_a_layout_walks_once_per_card_class(self):
        """`_estimate_layout` over the default series walks each card CLASS
        once — on the class's roomiest card (`class_cards`) — and not once
        per card, nor once for the series.

        Counting the walks is the check that survives a model whose graph
        happens not to depend on the capability: anchoring the series on one
        card again would show up here as a single walk, whatever the numbers
        come out as."""
        import backend.inference.parallel_sweep as ps
        from backend.inference.framework_bindings import card_class
        from backend.inference.gpu_specs import get_gpu

        series = list(ps.DEFAULT_SWEEP_GPUS)
        walked = []
        real = ps._capacity_preparation

        def spy(*, config, gpu_sku, knobs):
            walked.append(gpu_sku)
            return real(config=config, gpu_sku=gpu_sku, knobs=knobs)

        llama = _cfg("meta-llama__Llama-3.2-1B.json")
        payload = {
            "config": llama, "model_id": "meta-llama/Llama-3.2-1B",
            "gpu_skus": series, "avg_input_tokens": 1024,
            "avg_output_tokens": 128, "parallel": ParallelConfig(),
            "num_gpus": 1, "selection_mode": "theoretical_only",
        }
        with mock.patch.object(ps, "_capacity_preparation", spy):
            out = ps._estimate_layout(payload)
        self.assertEqual(sorted(out), sorted(series))
        classes = {card_class(get_gpu(s)) for s in series}
        self.assertEqual(len(walked), len(classes))
        self.assertEqual({card_class(get_gpu(w)) for w in walked}, classes)
        # Each class walked on the card `class_cards` names, not on
        # whichever card of it the series happens to list first.
        self.assertEqual(sorted(walked),
                         sorted({ps.class_cards(s, series)[0] for s in series}))

    def test_estimate_layout_safe_isolates_a_crash(self):
        import backend.inference.parallel_sweep as ps

        def boom(_payload):
            raise ValueError("kaboom")

        with mock.patch.object(ps, "_estimate_layout", boom):
            out = ps._estimate_layout_safe({"gpu_skus": ["B300"]})
        self.assertEqual(out, {ps._LAYOUT_ERROR_KEY: "kaboom"})

    @pytest.mark.regression  # minutes-long
    def test_a_wholly_failing_mode_collapses_to_unavailable(self):
        # A mode whose every point fails (a recurrent mixer / vision under CP)
        # comes back as an empty series carrying the reason — and the OTHER
        # modes still report, rather than the whole sweep raising.
        import backend.inference.parallel_sweep as ps
        env = _base_env(_cfg("deepseek-ai__DeepSeek-V3.json"), "deepseek-ai/DeepSeek-V3")
        real = ps._estimate_layout

        def fail_cp(payload):
            cfg = payload["parallel"]
            if getattr(cfg, "cp", 1) and cfg.cp > 1:
                raise ValueError("boom-cp")
            return real(payload)

        # Serial path so the in-process monkeypatch is seen (pool workers spawn).
        with mock.patch.object(ps, "_estimate_layout", fail_cp), \
             mock.patch("backend.measure.pool.pool_enabled", lambda: False):
            r = ps.estimate_parallel_sweep(
                modes=["dp_ep", "cp_ep"], context_len=1024,
                gpu_skus=["B300"], base_env=env)
        bm = r["by_mode"]
        self.assertEqual(bm["cp_ep"]["series"], [])
        self.assertIn("boom-cp", bm["cp_ep"].get("unavailable", ""))
        self.assertTrue(any(s["points"] for s in bm["dp_ep"]["series"]),
                        "dp_ep must still report when cp_ep failed")


if __name__ == "__main__":
    unittest.main()
