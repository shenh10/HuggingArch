"""Tests for the parallelism comm cost — the collective per-rank
traffic formulas and their end-to-end wiring into the per-op roofline.
"""
from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.comm_volume import RouteFanout, ring_factor
from backend.inference.comm_cost import (
    COMM_OPS, COMM_REACH_HOST, COMM_REACH_SCALEOUT, COMM_REACH_SCALEUP,
    Fabric, PlacementLevel, collective_latency,
    collective_per_rank_bytes, collective_phases, comm_reach,
    members_per_node, placement_levels, price_collective, routed_shares,
)
from backend.inference.op_signatures import COMM_DOMAIN_AXIS


class TestCollectiveFormulas(unittest.TestCase):
    def test_world_one_is_zero(self):
        # No peers → nothing crosses the interconnect.
        for op in COMM_OPS:
            self.assertEqual(collective_per_rank_bytes(op, 1, 1000), 0.0)

    def test_ring_factors(self):
        # all_reduce = 2(N-1)/N · bytes; the others = (N-1)/N · bytes.
        N, B = 8, 1000.0
        self.assertAlmostEqual(
            collective_per_rank_bytes("all_reduce", N, B), 2 * 7 / 8 * B)
        for op in ("reduce_scatter", "all_gather", "all_to_all"):
            self.assertAlmostEqual(
                collective_per_rank_bytes(op, N, B), 7 / 8 * B)

    def test_all_reduce_is_rs_plus_ag(self):
        # The headline SP equivalence: a ring all_reduce moves exactly as
        # much as a reduce_scatter + all_gather.
        N, B = 8, 1000.0
        ar = collective_per_rank_bytes("all_reduce", N, B)
        rs = collective_per_rank_bytes("reduce_scatter", N, B)
        ag = collective_per_rank_bytes("all_gather", N, B)
        self.assertAlmostEqual(ar, rs + ag)

    def test_non_comm_op_zero(self):
        self.assertEqual(collective_per_rank_bytes("linear", 8, 1000), 0.0)


class TestCollectivePhases(unittest.TestCase):
    """Hierarchy-aware decomposition. Whose contract: at most one phase
    when the collective fits inside a single NVLink domain; for cross-
    domain all_reduce the NCCL/Megatron 3-phase schedule
    (intra-RS + inter-AR-on-shard + intra-AG)."""

    def test_intra_only_single_phase(self):
        # world fits inside one node → single intra-fabric phase.
        ph = collective_phases("all_reduce", 8, 1024, gpus_per_node=8)
        self.assertEqual(len(ph), 1)
        self.assertEqual(ph[0]["name"], "all_reduce")
        self.assertEqual(ph[0]["fabric"], "intra")
        self.assertEqual(ph[0]["world"], 8)
        # Ring AR per-rank = 2(N-1)/N · T.
        self.assertAlmostEqual(ph[0]["bytes_per_rank"], 2 * 7 / 8 * 1024)

    def test_cross_node_all_reduce_is_three_phase(self):
        # world=16 > gpus_per_node=8, world % gpus_per_node == 0:
        # 3-phase hierarchical AR.
        T = 8192.0
        ph = collective_phases("all_reduce", 16, T, gpus_per_node=8)
        self.assertEqual([p["name"] for p in ph],
                         ["reduce_scatter", "all_reduce", "all_gather"])
        self.assertEqual([p["fabric"] for p in ph],
                         ["intra", "inter", "intra"])
        self.assertEqual([p["world"] for p in ph], [8, 2, 8])
        # Phase 1 (intra RS on full T): (8-1)/8 · T.
        self.assertAlmostEqual(ph[0]["bytes_per_rank"], 7 / 8 * T)
        # Phase 2 (inter AR on shard T/N_intra): 2(N_inter-1)/N_inter · T/8.
        self.assertAlmostEqual(ph[1]["bytes_per_rank"], 2 * 1 / 2 * (T / 8))
        # Phase 3 (intra AG on full T): (8-1)/8 · T.
        self.assertAlmostEqual(ph[2]["bytes_per_rank"], 7 / 8 * T)

    def test_cross_node_non_ar_falls_back_to_flat_ring_on_inter(self):
        # RS / AG / A2A don't have a widely-deployed hierarchical
        # decomposition in NCCL — they take a single inter-fabric ring.
        for op in ("reduce_scatter", "all_gather", "all_to_all"):
            ph = collective_phases(op, 16, 8192, gpus_per_node=8)
            self.assertEqual(len(ph), 1)
            self.assertEqual(ph[0]["fabric"], "inter")
            self.assertEqual(ph[0]["world"], 16)
            self.assertAlmostEqual(ph[0]["bytes_per_rank"], 15 / 16 * 8192)

    def test_non_divisible_world_falls_back_to_flat_ring(self):
        # When world is not a multiple of gpus_per_node (e.g. EP=12 on
        # 8-GPU nodes) the hierarchical decomposition isn't well-defined;
        # fall back to a single inter-fabric ring.
        ph = collective_phases("all_reduce", 12, 8192, gpus_per_node=8)
        self.assertEqual(len(ph), 1)
        self.assertEqual(ph[0]["fabric"], "inter")
        self.assertEqual(ph[0]["world"], 12)

    def test_world_one_returns_empty(self):
        # No peers → no phases at all.
        for op in COMM_OPS:
            self.assertEqual(collective_phases(op, 1, 1000, gpus_per_node=8), [])

    # --- base_stride (mesh topology of composed parallelism) -------------
    def test_base_stride_1_matches_legacy_size_only(self):
        # base_stride defaults to 1; passing it explicitly must reproduce
        # the size-only behavior byte-for-byte across the fabric cases.
        for op in ("all_reduce", "reduce_scatter", "all_gather", "all_to_all"):
            for n in (4, 8, 16, 12):
                self.assertEqual(
                    collective_phases(op, n, 8192, gpus_per_node=8),
                    collective_phases(op, n, 8192, gpus_per_node=8, base_stride=1),
                    f"{op} world={n} drifted when base_stride=1 passed explicitly",
                )

    def test_outer_small_axis_takes_the_inter_fabric(self):
        # THE FIX: cp=4 composed over tp_attention=8 on an 8-GPU node.
        # Each cp member is a whole TP inner_group (8 ranks) away → one member
        # per node → fully cross-node, even though 4 <= gpus_per_node=8.
        # Size-only logic mis-prices this as a single intra ring.
        intra = collective_phases("all_to_all", 4, 8192, gpus_per_node=8)
        self.assertEqual(intra[0]["fabric"], "intra")  # legacy (wrong) verdict
        topo = collective_phases(
            "all_to_all", 4, 8192, gpus_per_node=8, base_stride=8)
        self.assertEqual(len(topo), 1)
        self.assertEqual(topo[0]["fabric"], "inter")   # corrected verdict
        self.assertEqual(topo[0]["world"], 4)

    def test_inner_composed_axis_stays_intra(self):
        # ep=2 nested inside tp_moe_expert=4 on an 8-GPU node: the two
        # experts × 4-way TP = 8 ranks = exactly one node, so the EP
        # collective is node-local → intra (matches the DSv3 golden case).
        ph = collective_phases(
            "all_to_all", 2, 4096, gpus_per_node=8, base_stride=4)
        self.assertEqual(len(ph), 1)
        self.assertEqual(ph[0]["fabric"], "intra")

    def test_straddling_ar_uses_per_node_group_in_hierarchy(self):
        # AR over world=4 with base_stride=4 on g=8: members_per_node =
        # 8//4 = 2, so the group straddles (2 nodes × 2 members). The
        # hierarchical AR's intra phases run on the 2-member per-node
        # group, NOT on gpus_per_node.
        ph = collective_phases(
            "all_reduce", 4, 16384, gpus_per_node=8, base_stride=4)
        self.assertEqual([p["fabric"] for p in ph], ["intra", "inter", "intra"])
        self.assertEqual(ph[0]["world"], 2)   # n_intra = members_per_node
        self.assertEqual(ph[1]["world"], 2)   # n_inter = world / n_intra

    def test_a_pipeline_send_is_one_hop_of_the_whole_tensor(self):
        # A send across a pipeline cut moves the sender's tensor once,
        # whatever the pp degree: intra-node when every stage's counterpart
        # (base_stride ranks apart) shares the node, inter otherwise.
        self.assertEqual(collective_per_rank_bytes("send_recv", 4, 1000.0),
                         1000.0)
        (near,) = collective_phases(
            "send_recv", 4, 1000.0, gpus_per_node=8, base_stride=2)
        self.assertEqual((near["fabric"], near["bytes_per_rank"]),
                         ("intra", 1000.0))
        (far,) = collective_phases(
            "send_recv", 4, 1000.0, gpus_per_node=8, base_stride=4)
        self.assertEqual((far["fabric"], far["bytes_per_rank"]),
                         ("inter", 1000.0))


class TestCommCostInPipeline(unittest.TestCase):
    """End-to-end: analyze() attaches a t_comm rooftop term to the comm
    nodes, scaled by the layer count and the right TP world size."""

    @classmethod
    def setUpClass(cls):
        from backend.analyzer import fetch_config
        from tests.snapshots import shared_loader
        from backend.inference.framework_bindings import get_binding
        cls.resolved = shared_loader().match(fetch_config("meta-llama/Llama-2-7b-hf"), model_id="meta-llama/Llama-2-7b-hf")
        cls.binding = get_binding("vllm")

    def _rows(self, **pc_kwargs):
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        res = analyze(
            self.resolved, parallel=ParallelConfig(**pc_kwargs),
            framework_binding=self.binding, batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            interconnect_bw_bytes_per_sec=900e9,
            breakdown=True,
        )
        return res.per_op_timing

    def test_tp_comm_has_time_and_scales_by_layers(self):
        # Per-layer collectives only — the vocab-parallel embed all_reduce /
        # lm_head all_gather are top-level (count_per_step=1, once per forward),
        # not multiplied through the 32 layers, so they're scoped out of the
        # "scales by layers" assertion below.
        #
        # The walk is a 4096-token prefill, and vLLM's AR+RMSNorm fusion runs
        # only where its workspace holds the step's rows, which the binding
        # bills as decode (`allreduce_fused_rmsnorm`'s `phase`): every TP
        # all_reduce of a llama layer runs bare here, two per layer.
        rows = [r for r in self._rows(tp_attention=8, tp_dense_ffn=8)
                if r.get("comm_bytes_per_rank", 0) > 0
                and "vocab_" not in str(r.get("aggregation_id", ""))]
        self.assertTrue(rows)
        self.assertTrue(all(r["comm_op"] == "all_reduce" for r in rows))
        # llama has 32 layers and two TP all_reduces in each.
        self.assertEqual(sum(r["count_per_step"] for r in rows), 2 * 32)
        self.assertEqual({r["op"] for r in rows}, {"all_reduce"})
        self.assertEqual(sorted(r["count_per_step"] for r in rows), [32, 32])
        per_launch = [r["t_comm_s"] / r["count_per_step"] for r in rows]
        for r in rows:
            self.assertGreater(r["t_comm_s"], 0)
        # Same bytes per launch everywhere: the time is multiplied through
        # the layers, whichever row (in-layer or seam) carries the launch.
        for t in per_launch:
            self.assertAlmostEqual(t / per_launch[0], 1.0, places=9)

    def test_every_framework_prices_one_collective_alike(self):
        """The envelope asks the card, not the framework: on an H100 node
        the same all-reduce (world, instances and payload) costs the same
        under vLLM and sglang, in time, wire bytes and chosen algorithm."""
        from backend.inference.framework_bindings import get_binding
        from backend.inference.gpu_specs import get_gpu
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        gpu = get_gpu("H100-SXM")
        priced = {}
        for framework in ("vllm", "sglang"):
            res = analyze(
                self.resolved, parallel=ParallelConfig(tp_attention=8,
                                                       tp_dense_ffn=8),
                framework_binding=get_binding(framework), batch_size=1,
                avg_input_tokens=4096, avg_output_tokens=0,
                peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
                gpu=gpu, intra_bw_bytes_per_sec=450e9,
                inter_bw_bytes_per_sec=50e9, comm_latency_s=2e-6,
                breakdown=True)
            priced[framework] = {
                (c["op"], c["world"], c["count"], c["payload_bytes"]):
                    (r["t_comm_s"], r["comm_bytes_per_rank"], c["algorithm"])
                for r in res.per_op_timing
                for c in [r.get("comm_choice")] if c}
        shared = set(priced["vllm"]) & set(priced["sglang"])
        self.assertTrue(shared)
        for key in shared:
            self.assertEqual(priced["vllm"][key], priced["sglang"][key], key)
        # A 4096-token prefill all-reduce of Llama-2's 4096-wide residual is
        # 32 MiB: NVLS on this switch.
        self.assertIn("nvls", {v[2] for v in priced["vllm"].values()})

    def test_the_published_choice_is_the_rows_price(self):
        """Second route on a real walk: the candidate `comm_choice` names is
        the row's own price — its `t_s` the row's `t_comm_s` and its wire
        bytes the row's `comm_bytes_per_rank` — inside an NVLS node, across
        two HGX nodes (the hierarchical schedule) and on a PCIe card."""
        from backend.inference import gpu_specs as gs
        from backend.inference.framework_bindings import get_binding
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        seen = set()
        for card, world, want in (("H100-SXM", 16, {"nccl"}),
                                  ("H100-PCIe", 4, {"nccl"}),
                                  ("B300", 8, {"nvls"})):
            gpu = gs.get_gpu(card)
            res = analyze(
                self.resolved, parallel=ParallelConfig(
                    tp_attention=world, tp_dense_ffn=world),
                framework_binding=get_binding("vllm"), batch_size=1,
                avg_input_tokens=4096, avg_output_tokens=0,
                peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
                gpu=gpu,
                intra_bw_bytes_per_sec=gs.peak_intra_bw_bytes_per_sec(gpu),
                inter_bw_bytes_per_sec=gs.peak_inter_bw_bytes_per_sec(gpu),
                comm_latency_s=2e-6, breakdown=True)
            rows = [r for r in res.per_op_timing if r.get("comm_choice")]
            self.assertTrue(rows, card)
            for r in rows:
                choice = r["comm_choice"]
                chosen = next(c for c in choice["candidates"]
                              if c["algorithm"] == choice["algorithm"])
                self.assertAlmostEqual(chosen["t_s"], r["t_comm_s"],
                                       delta=1e-12 * max(1.0, r["t_comm_s"]))
                self.assertAlmostEqual(chosen["wire_bytes_per_rank"],
                                       r["comm_bytes_per_rank"], delta=1e-6)
                if choice["op"] == "all_reduce":
                    seen.add((card, choice["algorithm"],
                              choice["fabric"]["nodes_spanned"]))
            self.assertEqual({a for c, a, _n in seen if c == card}, want, card)
        self.assertIn(("H100-SXM", "nccl", 2), seen)

    def test_no_comm_no_cost_at_tp1(self):
        rows = self._rows(tp_attention=1, tp_dense_ffn=1)
        self.assertFalse(any(r.get("comm_bytes_per_rank", 0) > 0 for r in rows))

    def _dsv3_comm(self, **pc_kwargs):
        from backend.analyzer import fetch_config
        from tests.snapshots import shared_loader
        from backend.inference.framework_bindings import get_binding
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        resolved = shared_loader().match(fetch_config("deepseek-ai/DeepSeek-V3"), model_id="deepseek-ai/DeepSeek-V3")
        res = analyze(
            resolved, parallel=ParallelConfig(**pc_kwargs),
            framework_binding=get_binding("vllm"),
            batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            interconnect_bw_bytes_per_sec=900e9,
            breakdown=True,
        )
        return res.per_op_timing

    def test_ep_derives_all_to_all_when_the_resting_cut_splits_tokens(self):
        # Expert parallelism's all_to_all pair is a state transition, not an
        # op-intrinsic graft: it exists because the tokens arrive grouped by
        # REQUEST (DP-attention) and the expert bank wants them grouped by
        # OWNER, one tensor cut two ways over one process group. Both legs flow
        # through the generic comm cost path.
        rows = self._dsv3_comm(ep_moe=8, dp_attention=8)
        a2a_rows = [r for r in rows if r["op"] == "all_to_all"
                    and r.get("comm_bytes_per_rank", 0) > 0]
        # One from the permute boundary, one from the combine boundary.
        self.assertGreaterEqual(len(a2a_rows), 2)
        for r in a2a_rows:
            self.assertGreater(r["t_comm_s"], 0)
        # Permute itself should carry zero FLOPs / zero comm bytes —
        # purely a shape op.
        permute_rows = [r for r in rows if r["op"] == "permute"]
        self.assertTrue(permute_rows, "expected permute rows in DSv3 MoE breakdown")
        for r in permute_rows:
            self.assertEqual(r.get("comm_bytes_per_rank", 0), 0)

    def test_ep_without_a_token_cut_reduces_instead_of_exchanging(self):
        # The same degree of expert parallelism on a deployment that does NOT
        # cut tokens is a different bill, and the difference is not a special
        # case: every rank holds every token, so nothing re-partitions on the
        # way in, and each token leaves the bank with a partial sum over the
        # experts its rank owns. One reduction, no exchange — what a TP+EP
        # deployment without DP-attention actually runs.
        rows = self._dsv3_comm(ep_moe=8)
        self.assertFalse(any(r["op"] == "all_to_all"
                             and r.get("comm_bytes_per_rank", 0) > 0
                             for r in rows))
        reduced = [r for r in rows if r["op"] == "all_reduce"
                   and r.get("comm_bytes_per_rank", 0) > 0]
        self.assertTrue(reduced, "the routed partial has to reduce somewhere")
        for r in reduced:
            self.assertGreater(r["t_comm_s"], 0)

    def test_ep_one_derives_nothing(self):
        # ep=1 owns every expert, so neither regime has anything to pay for.
        rows = self._dsv3_comm(ep_moe=1)
        self.assertFalse(any(r.get("comm_bytes_per_rank", 0) > 0 for r in rows))

    def test_no_interconnect_bw_degrades_to_zero_time(self):
        # Without an interconnect bandwidth, comm bytes are still sized
        # but t_comm is 0 (structural-only, the pre-cost behavior).
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        res = analyze(
            self.resolved, parallel=ParallelConfig(tp_attention=8),
            framework_binding=self.binding, batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            interconnect_bw_bytes_per_sec=None,
            breakdown=True,
        )
        comm = [r for r in res.per_op_timing if r.get("comm_bytes_per_rank", 0) > 0]
        self.assertTrue(comm)
        self.assertTrue(all(r["t_comm_s"] == 0 for r in comm))


class TestCommBandwidthLaneSplit(unittest.TestCase):
    """End-to-end check on the intra (NVLink) vs inter (RDMA) bandwidth
    selection: a TP collective whose world fits inside one node rides
    the intra fabric; once world exceeds gpus_per_node the collective
    crosses node boundaries and takes the inter fabric (slower → larger
    t_comm at the same per-rank bytes)."""

    @classmethod
    def setUpClass(cls):
        from backend.analyzer import fetch_config
        from tests.snapshots import shared_loader
        from backend.inference.framework_bindings import get_binding
        cls.resolved = shared_loader().match(fetch_config("meta-llama/Llama-2-7b-hf"), model_id="meta-llama/Llama-2-7b-hf")
        cls.binding = get_binding("vllm")

    def _rows(self, **pc_kwargs):
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        # H100-SXM-like split, per-direction unidirectional GB/s:
        # NVLink 4 intra (450 GB/s/dir), RDMA CX-7 inter (50 GB/s/dir),
        # 8 GPUs per node. Picking these as fixed numbers keeps the
        # test independent of the CSV (so a CSV update can't silently
        # break the assertion).
        res = analyze(
            self.resolved, parallel=ParallelConfig(**pc_kwargs),
            framework_binding=self.binding, batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            intra_bw_bytes_per_sec=450e9,
            inter_bw_bytes_per_sec=50e9,
            gpus_per_node=8,
            intra_bw_label="NVLink",
            inter_bw_label="RDMA",
            breakdown=True,
        )
        return res.per_op_timing

    def test_tp8_takes_the_intra_nvlink_fabric(self):
        # TP=8 ≤ gpus_per_node=8 → single intra-only NVLink phase. The
        # bottleneck reduces to that phase, so the row-level label and
        # bandwidth match the intra fabric and t_comm = per_rank_bytes / intra_bw.
        rows = [r for r in self._rows(tp_attention=8, tp_dense_ffn=8)
                if r.get("comm_bytes_per_rank", 0) > 0
                and r["op"] == "all_reduce"]
        self.assertTrue(rows, "expected an all_reduce row at TP=8")
        for r in rows:
            self.assertEqual(r["comm_bw_label"], "NVLink")
            self.assertEqual(r["comm_bw_bytes_per_sec"], 450e9)
            # Single-phase intra-only case: t_comm == per_rank_bytes / intra_bw.
            self.assertAlmostEqual(
                r["t_comm_s"], r["comm_bytes_per_rank"] / 450e9,
                places=12,
            )
            # Single-phase rows collapse `comm_phases` to an empty list —
            # frontend uses that to skip the per-phase breakdown popover.
            self.assertEqual(r["comm_phases"], [])

    def test_tp16_crosses_nodes_uses_hierarchical_ar(self):
        # TP=16 > gpus_per_node=8 → cross-node all_reduce. NCCL /
        # Megatron run a 3-phase hierarchical schedule (intra-RS +
        # inter-AR on the shard + intra-AG) rather than a flat 16-rank
        # ring on the inter fabric; the row carries `comm_phases` with
        # the three sub-collectives and t_comm sums their per-phase
        # times. Catches the pre-fix `flat_ring(T) / inter_bw` model,
        # which materially under-counts intra-node hops.
        rows = [r for r in self._rows(tp_attention=16, tp_dense_ffn=16)
                if r.get("comm_bytes_per_rank", 0) > 0
                and r["op"] == "all_reduce"]
        self.assertTrue(rows, "expected an all_reduce row at TP=16")
        for r in rows:
            phases = r["comm_phases"]
            self.assertEqual(
                len(phases), 3,
                f"expected 3 hierarchical phases, got {phases}",
            )
            self.assertEqual(
                [p["name"] for p in phases],
                ["reduce_scatter", "all_reduce", "all_gather"],
            )
            self.assertEqual(
                [p["fabric"] for p in phases],
                ["intra", "inter", "intra"],
            )
            # Intra phases span one node (8 ranks); the inter phase
            # spans the cross-node domain (16/8 = 2 nodes).
            self.assertEqual([p["world"] for p in phases], [8, 2, 8])
            # Row total t_comm == Σ phase times — the whole point of
            # the hierarchical model.
            expected_t = sum(p["t_phase_s"] for p in phases)
            self.assertAlmostEqual(r["t_comm_s"], expected_t, places=12)
            # Bottleneck phase (largest t_phase_s) determines the
            # row-level summary bandwidth + label. Under realistic
            # H100 ratios (450 GB/s intra vs 50 GB/s inter) the inter
            # phase is the bottleneck → RDMA.
            bottleneck = max(phases, key=lambda p: p["t_phase_s"])
            self.assertEqual(r["comm_bw_label"], bottleneck["fabric_label"])
            self.assertEqual(
                r["comm_bw_bytes_per_sec"], bottleneck["bw_bytes_per_sec"],
            )
            self.assertEqual(r["comm_bw_label"], "RDMA")
            self.assertEqual(r["comm_bw_bytes_per_sec"], 50e9)

    def test_legacy_kwarg_still_works(self):
        # Pre-split callers used a single `interconnect_bw_bytes_per_sec`
        # kwarg. The alias must keep working: it mirrors to both intra
        # and inter so t_comm matches the legacy single-fabric behavior.
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        res = analyze(
            self.resolved, parallel=ParallelConfig(tp_attention=8),
            framework_binding=self.binding, batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            interconnect_bw_bytes_per_sec=450e9,
            breakdown=True,
        )
        rows = [r for r in res.per_op_timing
                if r.get("comm_bytes_per_rank", 0) > 0]
        self.assertTrue(rows)
        for r in rows:
            self.assertGreater(r["t_comm_s"], 0)
            # Legacy alias mirrors to both fabrics → whichever one the
            # world size picks, the BW is the same 450e9.
            self.assertEqual(r["comm_bw_bytes_per_sec"], 450e9)

    def test_comm_world_one_short_circuits(self):
        # TP=1 → no comm rows at all (no peers). Defensive: even if
        # something synthesized a comm node with world=1, t_comm would
        # be zero and the row would carry empty bandwidth label.
        rows = self._rows(tp_attention=1, tp_dense_ffn=1)
        for r in rows:
            self.assertEqual(r.get("comm_bytes_per_rank", 0), 0)
            self.assertEqual(r.get("t_comm_s", 0), 0)
            self.assertEqual(r.get("comm_bw_label", ""), "")

    def test_per_op_rows_carry_shape_out(self):
        # Every leaf row in per_op_timing must surface its tensor shape
        # so the Per-Module Breakdown can render a Shape column. For
        # compute leaves this comes from the physical shape layer; for
        # materialized communication rows it comes from the typed payload
        # shape carried in the communication metadata layer.
        # Regression guard: a previous version forgot the comm fallback,
        # leaving AR rows with shape=None — the most identifying fact
        # about a collective was invisible.
        rows = self._rows(tp_attention=8, tp_dense_ffn=8)
        # Compute leaves: at least one linear must have a populated shape.
        compute_rows = [r for r in rows if r["op"] not in (
            "all_reduce", "reduce_scatter", "all_gather", "all_to_all",
            "permute", "allreduce_fused_rmsnorm",
        )]
        with_shape = [r for r in compute_rows if r.get("shape_out")]
        self.assertTrue(with_shape, "no compute leaf carries shape_out")
        for r in with_shape:
            self.assertIsInstance(r["shape_out"], list)
            self.assertTrue(all(isinstance(d, int) for d in r["shape_out"]))

        # Comm rows: AR rows in llama TP=8 must carry the producer's
        # shape (decompressed [B, S, H] for the FFN-exit reduce).
        comm_rows = [r for r in rows
                     if r["op"] == "all_reduce"
                     and r.get("comm_bytes_per_rank", 0) > 0]
        self.assertTrue(comm_rows, "expected at least one all_reduce row")
        for r in comm_rows:
            self.assertIsNotNone(
                r.get("shape_out"),
                f"all_reduce row {r['aggregation_id']!r} missing shape_out — "
                f"comm_meta tensor_shape fallback broken",
            )
            self.assertIsInstance(r["shape_out"], list)
            self.assertTrue(all(isinstance(d, int) for d in r["shape_out"]))

    def test_comm_bytes_formula_carries_symbolic_and_substituted(self):
        # The popover-grade formula is a {symbolic, substituted} dict so
        # the frontend can render the coefficient·T form alongside the
        # substituted numbers as first-class lines. `T` is EXPANDED to its
        # axis·dtype product — `2*(N-1)/N * (...·dt_a)` symbolic, with the
        # concrete dims·dtype on the substituted line — so the bytes are
        # traceable to the tensor shape, not an opaque T. `*` (not `·`) for
        # the coefficient so it's copy-paste safe back into Python / sympy.
        rows = [r for r in self._rows(tp_attention=8, tp_dense_ffn=8)
                if r.get("comm_bytes_per_rank", 0) > 0
                and r["op"] == "all_reduce"]
        self.assertTrue(rows)
        for r in rows:
            fmt = r["comm_bytes_formula"]
            self.assertIsInstance(fmt, dict)
            # coefficient kept; T expanded to a parenthesized axis·dt_a product.
            self.assertIn("2*(N-1)/N * (", fmt["symbolic"])
            self.assertIn("dt_a", fmt["symbolic"])
            self.assertIn("2*7/8", fmt["substituted"])
            self.assertEqual(r["comm_world"], 8)


if __name__ == "__main__":
    unittest.main()


class TestCollectiveLatency(unittest.TestCase):
    """A collective's fixed time is the link latency times the traversals
    its schedule waits on in sequence (`comm_volume.latency_hops`), added
    over the phases of a hierarchical schedule."""

    ALPHA = 2e-6

    def _latency(self, op, world, count=1, gpus_per_node=8, stride=1,
                 rdma=0.0):
        fabric = Fabric(gpus_per_node, 4.5e11, 5e10, "NVLink", "RDMA",
                        self.ALPHA, inter_latency=rdma)
        return collective_latency(op=op, world=world, count_per_step=count,
                                  fabric=fabric, base_stride=stride)

    def test_each_collective_waits_on_its_schedules_hops(self):
        from backend.arch.layout.comm_volume import latency_hops
        self.assertEqual(latency_hops("all_reduce", 8), 6)    # RS + AG, 3 rounds each
        self.assertEqual(latency_hops("reduce_scatter", 8), 3)
        self.assertEqual(latency_hops("all_gather", 5), 3)    # rounds round up
        self.assertEqual(latency_hops("all_to_all", 64), 6)   # Bruck
        self.assertEqual(latency_hops("send_recv", 2), 1)
        self.assertEqual(latency_hops("all_reduce", 1), 0)

    def test_a_hierarchical_all_reduce_adds_its_phases_hops(self):
        # 16 ranks over 2 nodes: intra RS (3) + inter AR of 2 (2) + intra AG (3).
        self.assertAlmostEqual(self._latency("all_reduce", 16), 8 * self.ALPHA)
        # Within a node: one flat all-reduce of 8.
        self.assertAlmostEqual(self._latency("all_reduce", 8), 6 * self.ALPHA)

    def test_a_hop_across_nodes_waits_on_the_cards_rdma_latency_too(self):
        # 16 ranks over 2 nodes: the inter AR's 2 hops each wait α + RDMA,
        # the intra RS and AG's 3 + 3 wait α alone.
        rdma = 4.4e-6
        self.assertAlmostEqual(self._latency("all_reduce", 16, rdma=rdma),
                               8 * self.ALPHA + 2 * rdma)
        # Inside a node no hop crosses: the RDMA latency is not paid.
        self.assertAlmostEqual(self._latency("all_reduce", 8, rdma=rdma),
                               6 * self.ALPHA)
        # A world that is not whole nodes runs one inter ring: every hop
        # crosses (Bruck, ceil(log2 12) = 4).
        self.assertAlmostEqual(self._latency("all_to_all", 12, rdma=rdma),
                               4 * (self.ALPHA + rdma))

    def test_the_sheet_states_the_rdma_hop_for_every_card_with_rdma(self):
        from backend.inference.gpu_specs import (
            inter_hop_latency_s, load_gpu_specs, peak_inter_bw_bytes_per_sec)
        for gpu in load_gpu_specs():
            fabric = Fabric.of(gpu, self.ALPHA)
            if peak_inter_bw_bytes_per_sec(gpu):
                self.assertGreater(inter_hop_latency_s(gpu), 0, gpu.sku)
                self.assertGreater(fabric.hop_latency("inter"),
                                   fabric.hop_latency("intra"), gpu.sku)
            self.assertEqual(fabric.hop_latency("intra"), self.ALPHA, gpu.sku)
            self.assertEqual(fabric.without_latency().hop_latency("inter"),
                             0.0, gpu.sku)

    def test_every_instance_pays_it_and_bytes_do_not_move_it(self):
        self.assertAlmostEqual(self._latency("all_to_all", 16, count=58),
                               58 * 4 * self.ALPHA)
        t_small, _ = price_collective(
            op="all_reduce", world=8, tensor_bytes=1.0, count_per_step=1,
            fabric=Fabric(8, 4.5e11, 5e10, "NVLink", "RDMA", self.ALPHA),
            base_stride=1)
        self.assertAlmostEqual(t_small, 6 * self.ALPHA, delta=1e-10)


class TestTheEnvelopePicksTheAllReduce(unittest.TestCase):
    """A collective is priced as its envelope (`comm_cost.Envelope`): the
    fastest, at its payload, of the algorithms the card's fabric can run —
    one-shot (one traversal, a rank receives (N-1)T) and two-shot (two,
    2(N-1)/N T) inside an NVLink domain, NVLS (two, (N-1)/N T) where the
    switch reduces, NCCL's schedule everywhere. Which ones the fabric runs
    is a hardware fact and nothing else; no framework is asked."""

    ALPHA = 2e-6

    def _envelope(self, card, world, stride=1, op="all_reduce", count=1):
        from backend.inference.comm_cost import CollectiveSignature
        from backend.inference.gpu_specs import get_gpu
        fabric = Fabric.of(get_gpu(card), self.ALPHA)
        return fabric, CollectiveSignature(op, world, count, stride).envelope(
            fabric)

    def _one(self, card, world, payload, stride=1):
        fabric, envelope = self._envelope(card, world, stride)
        chosen = envelope.choose(payload)
        t, rows = price_collective(
            op="all_reduce", world=world, tensor_bytes=payload,
            count_per_step=1, fabric=fabric, base_stride=stride,
            algorithm=chosen.algorithm)
        self.assertAlmostEqual(t, chosen.t(payload), delta=1e-15)
        return t, [(r["algorithm"], r["name"], r["world"], r["hops"])
                   for r in rows]

    def test_a_small_payload_is_one_shot_on_an_nvlink_card(self):
        # Kimi-K3 / DeepSeek-V3 at B=1: 7168 x 2 B.
        for card, bw in (("H100-SXM", 450e9), ("B300", 900e9),
                         ("A100-80G-SXM", 300e9)):
            t, rows = self._one(card, 8, 14336.0)
            self.assertEqual(rows, [("one_shot", "all_reduce", 8, 1)], card)
            # By hand: one traversal, and every rank receives 7 payloads.
            self.assertAlmostEqual(t, self.ALPHA + 7 * 14336 / bw, delta=1e-15)

    def test_a_large_payload_is_nvls_where_the_switch_reduces(self):
        T = 14680064.0   # 1024 tokens x 7168 x 2 B
        for card, bw in (("H100-SXM", 450e9), ("B300", 900e9)):
            t, rows = self._one(card, 8, T)
            self.assertEqual(rows, [("nvls", "all_reduce", 8, 2)], card)
            self.assertAlmostEqual(t, 2 * self.ALPHA + 7 / 8 * T / bw,
                                   delta=1e-12)

    def test_the_switch_is_where_one_shot_and_nvls_cross(self):
        # one-shot α + 7T/bw against NVLS 2α + 7T/(8 bw): equal at
        # T = α bw / 6.125, the fastest on either side of it.
        cross = self.ALPHA * 450e9 / 6.125
        self.assertEqual(self._one("H100-SXM", 8, cross * 0.99)[1][0][0],
                         "one_shot")
        self.assertEqual(self._one("H100-SXM", 8, cross * 1.01)[1][0][0],
                         "nvls")

    def test_nvlink_without_a_reducing_switch_is_two_shot(self):
        # A100's NVSwitch does not reduce in the network: past one-shot it
        # is two-shot — NCCL's ring moves the same bytes in more hops.
        T = 14680064.0
        t, rows = self._one("A100-80G-SXM", 8, T)
        self.assertEqual(rows, [("two_shot", "all_reduce", 8, 2)])
        self.assertAlmostEqual(t, 2 * self.ALPHA + 1.75 * T / 300e9,
                               delta=1e-12)
        _fabric, envelope = self._envelope("A100-80G-SXM", 8)
        nvls = next(c for c in envelope.candidates if c.algorithm == "nvls")
        self.assertEqual(nvls.refusal, "card has no NVLink multicast (NVLS)")

    def test_without_an_nvlink_domain_it_is_nccls_schedule(self):
        # PCIe cards: no peer-to-peer NVLink, no switch — every payload runs
        # NCCL's schedule, its log-depth hops and the ring's bytes.
        for card, world in (("H100-PCIe", 2), ("H100-PCIe", 4),
                            ("RTX-4090", 2), ("RTX-4090", 8)):
            for payload in (14336.0, 14680064.0):
                _t, rows = self._one(card, world, payload)
                self.assertEqual(rows, [("nccl", "all_reduce", world,
                                         2 * (world - 1).bit_length())],
                                 (card, world, payload))
        _fabric, envelope = self._envelope("H100-PCIe", 4)
        self.assertEqual(
            [(c.algorithm, c.refusal) for c in envelope.candidates],
            [("one_shot", "card has no NVLink domain (PCIe)"),
             ("two_shot", "card has no NVLink domain (PCIe)"),
             ("nvls", "card has no NVLink domain (PCIe)"),
             ("nccl", None)])

    def test_across_hgx_nodes_the_hierarchy_stays(self):
        # TP16 on 8-GPU HGX nodes: NCCL's three phases, each its own hops.
        _t, rows = self._one("H100-SXM", 16, 14336.0)
        self.assertEqual(rows, [("nccl", "reduce_scatter", 8, 3),
                                ("nccl", "all_reduce", 2, 2),
                                ("nccl", "all_gather", 8, 3)])
        _fabric, envelope = self._envelope("H100-SXM", 16)
        self.assertEqual(
            [c.refusal for c in envelope.candidates[:-1]],
            [f"the group spans 2 nodes; {a} runs inside one NVLink domain"
             for a in ("one-shot", "two-shot", "nvls")])
        # An outer group strided across nodes is one member per node.
        _t, rows = self._one("H100-SXM", 4, 14336.0, stride=8)
        self.assertEqual(rows, [("nccl", "all_reduce", 4, 4)])

    def test_an_nvl72_rack_is_one_domain_at_any_group_size(self):
        # GB200 NVL72: a 16-, 32- or 72-rank group stays inside the rack's
        # NVLink domain, so the switch's algorithms run there — no
        # framework's table of group sizes is asked.
        for world in (16, 32, 72):
            self.assertEqual(self._one("GB200", world, 7168.0)[1],
                             [("one_shot", "all_reduce", world, 1)], world)
            self.assertEqual(self._one("GB200", world, 14680064.0)[1],
                             [("nvls", "all_reduce", world, 2)], world)

    def test_other_collectives_run_nccls_schedule(self):
        for op in ("all_gather", "reduce_scatter", "all_to_all"):
            _fabric, envelope = self._envelope("B300", 8, op=op)
            self.assertEqual([c.algorithm for c in envelope.candidates],
                             ["nccl"], op)

    def test_a_fabric_with_no_card_runs_nccl(self):
        # Bandwidths alone admit no NVLink algorithm: the textbook schedule.
        from backend.inference.comm_cost import CollectiveSignature
        fabric = Fabric(8, 4.5e11, 5e10, "NVLink", "RDMA", self.ALPHA)
        envelope = CollectiveSignature("all_reduce", 8, 1, 1).envelope(fabric)
        self.assertEqual([c.algorithm for c in envelope.candidates
                          if c.feasible], ["nccl"])
        self.assertEqual(envelope.choose(14336.0).algorithm, "nccl")
        # A refused algorithm has no price: it never finishes, so the law's
        # minimum over every candidate is the minimum over the feasible
        # ones, whatever order the algorithms are declared in.
        self.assertEqual([latency == math.inf for _rate, latency
                          in envelope.law_values()],
                         [True, True, True, False])

    def test_the_choice_is_published_from_the_same_envelope(self):
        """`comm_choice`: the chosen algorithm is the fastest feasible
        candidate, its time and wire bytes are the ones the row prices, and
        every feasible candidate's curve is its own law on the log grid."""
        from backend.inference.comm_cost import CURVE_PAYLOADS
        fabric, envelope = self._envelope("H100-SXM", 8, count=3)
        payload = 229376.0
        choice = envelope.choice(payload)
        self.assertEqual(choice["algorithm"], "nvls")
        self.assertEqual(choice["fabric"], {"link": "NVLink", "nvls": True,
                                            "gpus_per_node": 8,
                                            "nodes_spanned": 1})
        feasible = [c for c in choice["candidates"] if c["feasible"]]
        self.assertEqual(min(feasible, key=lambda c: c["t_s"])["algorithm"],
                         "nvls")
        chosen = next(c for c in feasible if c["algorithm"] == "nvls")
        t, rows = price_collective(
            op="all_reduce", world=8, tensor_bytes=payload, count_per_step=3,
            fabric=fabric, base_stride=1, algorithm="nvls")
        self.assertAlmostEqual(chosen["t_s"], t, delta=1e-15)
        self.assertEqual(chosen["wire_bytes_per_rank"],
                         sum(r["bytes_per_rank"] for r in rows))
        self.assertEqual(chosen["wire_bytes_per_rank"], 3 * 7 / 8 * payload)
        self.assertEqual([p for p, _t in chosen["curve"]],
                         list(CURVE_PAYLOADS))
        self.assertEqual(len(CURVE_PAYLOADS), 25)
        # The envelope on the same grid is the pointwise minimum of the
        # feasible curves, with the algorithm that sets it.
        for k, (p, t, algorithm) in enumerate(choice["envelope_curve"]):
            best = min(feasible, key=lambda c: c["curve"][k][1])
            self.assertEqual((p, t, algorithm),
                             (best["curve"][k][0], best["curve"][k][1],
                              best["algorithm"]))
        self.assertEqual({row[2] for row in choice["envelope_curve"]},
                         {"one_shot", "nvls"})
        self.assertIn("12.5%", chosen["note"])
        # What each algorithm is, from its declaration: a declared
        # algorithm's stages, and NCCL's from its phases on this placement.
        self.assertEqual([st["kind"] for st in chosen["stages"]],
                         ["switch_reduce", "switch_multicast"])
        self.assertTrue(all(c["description"] for c in choice["candidates"]))
        self.assertEqual(choice["ranks"], list(range(8)))
        fabric, envelope = self._envelope("H100-SXM", 16)
        nccl = envelope.choice(payload)["candidates"][-1]
        self.assertEqual([st["kind"] for st in nccl["stages"]],
                         ["reduce_scatter", "inter_node", "all_gather"])
        self.assertEqual((CURVE_PAYLOADS[0], CURVE_PAYLOADS[-1]),
                         (1 << 10, 1 << 30))


class TestRoutedExchangeLevels(unittest.TestCase):
    """A routed exchange's bytes are the exchange's; the card's levels only
    say which fabric they cross.

    The total is "how many distinct destination RANKS does a token reach",
    which needs no card fact and is what the layer publishes
    (`comm_volume.ring_factor`). `placement_levels` partitions it: a level
    whose destination is a node counts its own destinations (the RDMA
    sends), the level whose destination is a rank takes the remainder. So
    the phases always sum to the row's own bytes, and a group spread over
    more nodes cannot become cheaper — only slower.

    What is here is the law's own properties, asserted: the partition, the
    degeneracies, the hop counts, which levels a card makes. The
    independent derivations of the same numbers are the enumeration in
    `tests/backend/arch/layout/test_route_fanout.py` and the simulation in
    `tests/backend/inference/test_route_dedup_oracle.py`.

    The degeneracies fall out of that one law: a level with one destination
    is not a level, so a single node has only the exchange inside it and a
    group with one member per node only the one between nodes.
    """

    #: DeepSeek-V3's bank: 256 experts, every one weighted, top-8.
    F = RouteFanout(picks=8, targets=256, placed=256)
    T = 121110528.0

    def _phases(self, world, gpus_per_node, base_stride=1, fanout=None):
        return collective_phases(
            "all_to_all", world, self.T, gpus_per_node=gpus_per_node,
            base_stride=base_stride,
            fanout=self.F if fanout is None else fanout)

    def _shares(self, world, gpus_per_node, base_stride=1, fanout=None):
        fanout = self.F if fanout is None else fanout
        return routed_shares(fanout, world,
                             placement_levels(world, gpus_per_node,
                                              base_stride))

    # -- the total, and its partition ------------------------------------

    def test_the_levels_partition_the_exchanges_own_bytes(self):
        """The conservation ASSERTION, over every placement a card can make.

        Not an oracle: both sides go through `RouteFanout.remote_factor`,
        so this is the structural property stated as a test — the levels
        add up to the exchange's total — and not a second derivation of it.
        The second routes are elsewhere and are two: the enumeration over
        every top-k draw (`test_route_fanout`) and the Monte Carlo over
        real rank placements (`test_route_dedup_oracle`).

        It is worth having anyway, because the property it states is the
        one the first form of this model broke: a term charged to nobody
        (the receiving node's fan-out) fails here at every placement, over
        every routing shape, in a second.
        """
        for picks, targets, placed in ((8, 256, 256), (1, 256, 256),
                                       (8, 8, 8), (3, 24, 16),
                                       (6, 384, 384), (2, 128, 128)):
            fanout = RouteFanout(picks=picks, targets=targets, placed=placed)
            for world in (2, 4, 8, 16, 32, 64, 128):
                if placed % world:
                    continue
                for node, stride in ((4, 1), (8, 1), (8, 2), (72, 1),
                                     (72, 2), (8, 8)):
                    shares = self._shares(world, node, stride, fanout)
                    self.assertAlmostEqual(
                        sum(share for _level, share in shares),
                        float(ring_factor("all_to_all", world, fanout)),
                        places=12,
                        msg=f"k={picks} T={targets} E={placed} world={world} "
                            f"node={node} stride={stride}")

    def test_the_phases_sum_to_the_bytes_the_row_publishes(self):
        """The same conservation read where a user sees it: the per-phase
        rows a popover adds up against the per-rank bytes the layer states
        for the same collective. These disagreed by 1.56x."""
        for world, node in ((8, 8), (8, 4), (32, 8), (64, 8), (16, 4)):
            phases = self._phases(world, node)
            self.assertAlmostEqual(
                sum(ph["bytes_per_rank"] for ph in phases),
                collective_per_rank_bytes("all_to_all", world, self.T,
                                          self.F),
                delta=1e-6 * self.T, msg=f"world={world} node={node}")

    def test_a_worse_topology_is_never_cheaper(self):
        """One group of eight, one node of eight against two nodes of four:
        the same bytes, more of them on the slower fabric. It used to cost
        0.579 of the payload on one node and 0.373 on two."""
        one, two = self._shares(8, 8), self._shares(8, 4)
        self.assertAlmostEqual(sum(s for _l, s in one),
                               sum(s for _l, s in two), places=12)
        self.assertEqual([level.fabric for level, _s in two],
                         ["intra", "inter"])
        self.assertGreater(dict((l.fabric, s) for l, s in two)["inter"], 0.0)

    # -- the levels a card makes -----------------------------------------

    def test_a_single_node_has_one_level_and_the_flat_hop_count(self):
        """Every member in one node: the exchange between nodes has one
        destination, so there is no second phase — and deduping inside the
        node forwards nothing further, so the hops are the flat
        `ceil(log2 world)` they were."""
        self.assertEqual(placement_levels(8, 8, 1),
                         (PlacementLevel("intra", 8, 1),))
        (ph,) = self._phases(8, 8)
        self.assertEqual((ph["fabric"], ph["world"], ph["hops"]),
                         ("intra", 8, 3))
        self.assertEqual(
            ph["hops"],
            collective_phases("all_to_all", 8, self.T,
                              gpus_per_node=8)[0]["hops"])
        self.assertAlmostEqual(ph["bytes_per_rank"] / self.T,
                               float(self.F.remote_factor(8)))

    def test_one_member_per_node_has_only_the_node_level(self):
        """The group's stride puts every member on its own node: there is
        no exchange inside a node, and the one phase carries the whole
        total on the inter fabric."""
        self.assertEqual(placement_levels(8, 8, 8),
                         (PlacementLevel("inter", 8, 1),))
        (ph,) = self._phases(8, 8, base_stride=8)
        self.assertEqual((ph["fabric"], ph["world"], ph["hops"]),
                         ("inter", 8, 3))
        self.assertAlmostEqual(ph["bytes_per_rank"] / self.T,
                               float(self.F.remote_factor(8)))

    def test_both_levels_split_the_hops_and_the_bytes(self):
        """Four nodes of eight: the ranks inside a node over NVLink, the
        nodes over RDMA, and the hops are the two exchanges' —
        `ceil(log2 4) + ceil(log2 8)`, the flat `ceil(log2 32)`."""
        intra, inter = self._phases(32, 8)
        self.assertEqual((intra["fabric"], intra["world"]), ("intra", 8))
        self.assertEqual((inter["fabric"], inter["world"]), ("inter", 4))
        self.assertEqual(intra["hops"] + inter["hops"], 5)
        # The node level counts its own destinations; the rank level takes
        # what is left of the exchange's total.
        self.assertAlmostEqual(
            inter["bytes_per_rank"] / self.T,
            float(self.F.remote_factor(32, scope_members=32, dest_members=8)))
        self.assertAlmostEqual(
            (intra["bytes_per_rank"] + inter["bytes_per_rank"]) / self.T,
            float(self.F.remote_factor(32)))

    def test_a_world_that_is_not_whole_nodes_is_one_level(self):
        """64 ranks of an NVL72 node, two ranks of expert tensor
        parallelism beneath them: 36 of the group's members fit in a node
        and 64 is not a whole number of those, so no node level's
        destination count applies. One level then, carrying the whole
        total on the inter fabric — pessimistic in the FABRIC, and not in
        the bytes."""
        self.assertEqual(placement_levels(64, 72, 2),
                         (PlacementLevel("inter", 64, 1),))
        (ph,) = self._phases(64, 72, base_stride=2)
        self.assertEqual((ph["fabric"], ph["world"]), ("inter", 64))
        self.assertAlmostEqual(ph["bytes_per_rank"] / self.T,
                               float(self.F.remote_factor(64)))

    def test_without_a_fan_out_nothing_changes(self):
        """A Ulysses CP exchange carries no fan-out and is priced exactly
        as a flat all-to-all — bit for bit the phases of an unrouted
        world."""
        for world, node, stride in ((8, 8, 1), (32, 8, 1), (12, 8, 1),
                                    (4, 8, 8)):
            self.assertEqual(
                collective_phases("all_to_all", world, self.T,
                                  gpus_per_node=node, base_stride=stride,
                                  fanout=None),
                collective_phases("all_to_all", world, self.T,
                                  gpus_per_node=node, base_stride=stride),
                f"world={world} node={node} stride={stride}")

    def test_nothing_to_dedup_is_the_flat_ring_at_both_levels(self):
        """The two degeneracies, end to end. One pick per token, and every
        pick to every destination: in both there is nothing to dedup, so
        the levels must sum to the flat ring's bytes exactly — and they do
        at every placement, not only on one node.

        This is what a model that charged the receiving node's fan-out to
        nobody could not do: with `k = T = E` it came out 1.75x cheaper
        than the ring.
        """
        for fanout in (RouteFanout(picks=1, targets=256, placed=256),
                       RouteFanout(picks=8, targets=8, placed=8),
                       RouteFanout(picks=4, targets=4, placed=4)):
            for world, node in ((8, 8), (8, 4), (32, 8), (16, 4), (64, 8)):
                if fanout.placed % world:
                    continue
                phases = self._phases(world, node, fanout=fanout)
                self.assertAlmostEqual(
                    sum(ph["bytes_per_rank"] for ph in phases) / self.T,
                    (world - 1) / world, places=12,
                    msg=f"{fanout} world={world} node={node}")
                self.assertEqual(
                    sum(ph["hops"] for ph in phases),
                    sum(ph["hops"] for ph in collective_phases(
                        "all_to_all", world, self.T, gpus_per_node=node)),
                    f"{fanout} world={world} node={node}")

    def test_the_levels_are_read_off_the_card(self):
        """The same group on a 4-GPU node is placed differently: two nodes
        of four instead of one node of eight. The levels are computed where
        the card is known, so a row re-read on another card re-levels."""
        self.assertEqual(placement_levels(8, 4, 1),
                         (PlacementLevel("intra", 4, 1),
                          PlacementLevel("inter", 8, 4)))
        self.assertEqual(len(self._phases(8, 4)), 2)
        self.assertEqual(len(self._phases(8, 8)), 1)

    def test_a_routed_exchange_runs_nccls_schedule(self):
        """A routed exchange is an all-to-all, which only NCCL's schedule
        runs (`comm_volume.algorithms_for`): its envelope is that one
        candidate at every placement, and asking the fabric about an
        all-reduce algorithm for it is an error, not a refusal."""
        from backend.inference.comm_cost import CollectiveSignature
        for world, node, stride in ((8, 8, 1), (32, 8, 1), (64, 72, 2)):
            fabric = Fabric(node, 4.5e11, 5e10, nvlink=True, nvls=True)
            envelope = CollectiveSignature(
                "all_to_all", world, 1, stride, self.F).envelope(fabric)
            self.assertEqual([c.algorithm for c in envelope.candidates],
                             ["nccl"])
            with self.assertRaises(ValueError):
                fabric.refusal("one_shot", op="all_to_all", world=world,
                               base_stride=stride)

    def test_members_per_node_has_one_statement(self):
        """The envelope's feasibility, the fabric split and the levels read
        one number (`members_per_node`), not three spellings of it."""
        for world, node, stride, want in ((8, 8, 1, 8), (8, 8, 2, 4),
                                          (32, 8, 1, 8), (4, 8, 8, 1),
                                          (8, 72, 1, 8), (2, 8, 4, 2)):
            self.assertEqual(members_per_node(world, node, stride), want,
                             f"{world}/{node}/{stride}")


class TestCardFabricHasOneStatement(unittest.TestCase):
    """A card's fabric is `Fabric.of` and nothing else.

    Three pricing roads used to build one from six loose preparation fields
    plus a `topology` splat (`capacity._point_rows`,
    `_stamp_gpu_on_result`, `_chunk_point_rows_and_phase`); they now ask
    `_CapacityPreparation.fabric`. This pins the two readings equal on
    every row of the sheet, so the consolidation is checked against the
    form it replaced rather than against itself.
    """

    def test_of_equals_the_loose_field_assembly(self):
        from backend.inference import gpu_specs as gs
        from backend.inference.comm_cost import topology
        latency = 1.5e-6
        for gpu in gs.load_gpu_specs():
            was = Fabric(
                gs.gpus_per_node(gpu),
                gs.peak_intra_bw_bytes_per_sec(gpu),
                gs.peak_inter_bw_bytes_per_sec(gpu),
                gs.intra_bw_label(gpu), gs.inter_bw_label(gpu),
                latency,
                scaleup_domain=gs.ranks_per_scaleup_domain(gpu),
                )._replace(**topology(gpu))
            self.assertEqual(Fabric.of(gpu, latency), was, gpu.sku)


class TheReachOfAGroup(unittest.TestCase):
    """A collective's group spans one chassis, a scale-up domain, or more
    than one — and a measurement of each must answer only for its own.

    Before this, a sheet row meant one chassis and a group either fitted it
    or did not. A supernode ends that: on an Atlas A3, 32 ranks leave the
    16-die chassis and are still on the scale-up fabric, a domain away in
    bandwidth from the RoCE the two-valued reading prices them at.
    """

    def test_a_group_inside_one_chassis_is_host(self):
        for world in (2, 8, 16):
            self.assertEqual(
                comm_reach(world, base_stride=1, ranks_per_host=16),
                COMM_REACH_HOST, world)

    def test_without_a_wider_domain_leaving_the_chassis_is_scaleout(self):
        """What the two-valued reading has always said, unchanged: a card
        declaring no domain past its chassis has nothing in between."""
        self.assertEqual(
            comm_reach(32, base_stride=1, ranks_per_host=16),
            COMM_REACH_SCALEOUT)

    def test_a_declared_supernode_puts_a_tier_in_between(self):
        def reach(n):
            return comm_reach(n, base_stride=1, ranks_per_host=16,
                              ranks_per_scaleup_domain=384)
        self.assertEqual(reach(16), COMM_REACH_HOST)
        self.assertEqual(reach(32), COMM_REACH_SCALEUP)
        self.assertEqual(reach(384), COMM_REACH_SCALEUP)
        self.assertEqual(reach(385), COMM_REACH_SCALEOUT)

    def test_stride_spreads_a_small_group_past_its_chassis(self):
        """An outer axis composed over an inner one is spaced out: four
        ranks at stride 8 touch four chassis of 16, not one."""
        self.assertEqual(
            comm_reach(4, base_stride=8, ranks_per_host=16,
                       ranks_per_scaleup_domain=384),
            COMM_REACH_SCALEUP)

    def test_the_price_and_the_key_divide_the_same_way(self):
        """`collective_phases` and `comm_reach` must not disagree about
        where a group sits: one says intra, the other says host, and they
        are the same sentence. Both read `members_within`."""
        for world, stride, g in ((8, 1, 16), (32, 1, 16), (4, 8, 16),
                                 (16, 1, 8), (12, 1, 16), (2, 1, 16)):
            phases = collective_phases("all_reduce", world, 1 << 20,
                                       gpus_per_node=g, base_stride=stride)
            priced_all_intra = bool(phases) and all(
                p["fabric"] == "intra" for p in phases)
            keyed_host = comm_reach(world, base_stride=stride,
                                    ranks_per_host=g) == COMM_REACH_HOST
            self.assertEqual(priced_all_intra, keyed_host,
                             f"world={world} stride={stride} g={g}")


class WhatACollectiveCellIsKeyedOn(unittest.TestCase):
    """The reach belongs in the key, and belongs there by its ABSENCE for
    the common case."""

    HOST = '{"n_comm_elems":4194304,"n_comm_ranks":8,"n_comm_stride":1}'

    @staticmethod
    def _sig(name="all_reduce"):
        import backend.inference.op_signatures_catalog  # noqa: F401
        from backend.inference.op_signatures import get_signature
        return get_signature(name)

    def _keyed(self, reach, name="all_reduce"):
        import json
        return self._sig(name).project(json.dumps(
            {**json.loads(self.HOST), COMM_DOMAIN_AXIS: int(reach)},
            separators=(",", ":"), sort_keys=True))

    def test_an_intra_host_row_keys_exactly_as_it_did(self):
        """Every collective row measured so far is intra-host, so this is
        the whole backward-compatibility claim: the axis costs them
        nothing and no corpus row is rewritten to gain it."""
        self.assertEqual(self._sig().project(self.HOST), self.HOST)

    def test_the_same_payload_and_group_at_two_reaches_are_two_keys(self):
        self.assertNotEqual(self._keyed(COMM_REACH_SCALEUP),
                            self._sig().project(self.HOST))

    def test_scaleup_and_scaleout_do_not_share_a_key_either(self):
        """The middle tier is the point: a supernode measurement must not
        answer for a group that left the supernode."""
        self.assertNotEqual(self._keyed(COMM_REACH_SCALEUP),
                            self._keyed(COMM_REACH_SCALEOUT))

    def test_every_collective_signature_declares_the_axis(self):
        """A signature that forgot it would go on sharing one key across
        fabrics, which is the defect this axis exists to end."""
        for name in ("all_reduce", "all_gather", "reduce_scatter",
                     "all_to_all", "send_recv"):
            self.assertIn(COMM_DOMAIN_AXIS, self._sig(name).variant_dims, name)

    def test_the_axis_is_not_a_cell_dim(self):
        """It identifies by presence. In `cell_dims` it would be REQUIRED,
        and every existing row — none of which carries it — would stop
        keying at all."""
        for name in ("all_reduce", "send_recv"):
            self.assertNotIn(COMM_DOMAIN_AXIS, self._sig(name).cell_dims, name)

    def test_a_crossing_lookup_misses_a_corpus_of_host_rows(self):
        """The property the axis exists for, asserted on the real lookup
        and not on two strings being unequal.

        An absent axis is the DEFAULT (host), never a wildcard matching
        every reach. So a group that left the chassis must not be handed
        an intra-host row just because that row's key omits the axis the
        asking key carries — the difference between those two readings is
        a cross-fabric collective priced on HCCS.

        The corpus now holds rows of BOTH reaches, so this asks the
        question at a reach nothing was measured at (`scaleout`) as well
        as at one that was (`scaleup`): the first must miss every row, and
        the second must not be answered by the chassis rows that share its
        payload and world.
        """
        import csv
        from backend.inference.op_calibration import (
            CalibrationRegistry, BenchmarkKey, merge_validated_rows,
            physical_time_for)
        path = PROJECT_ROOT / "tests/golden/ascend_vllm_ascend.csv"
        with open(path, newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["op"] in COMM_OPS]
        registry = merge_validated_rows(
            CalibrationRegistry(), [dict(r) for r in rows],
            source="<test>", origin="shared")

        import json

        def timing(key):
            return physical_time_for(
                registry, key,
                current_peak_tflops_at=lambda *_a, **_k: 248.4,
                current_peak_bw_gbps=1600.0)

        def reach_of(row):
            return json.loads(row["shape_json"]).get(COMM_DOMAIN_AXIS)

        host_rows = [r for r in rows if reach_of(r) is None]
        host_keys = [BenchmarkKey.from_csv_row(r).to_physical()
                     for r in host_rows]
        self.assertTrue(host_keys, "no intra-host collective rows to ask about")

        # The corpus answers for the reach each row was taken at ...
        hit = [k for k in host_keys if timing(k) is not None]
        self.assertTrue(hit, "no intra-host collective row answered at all, "
                             "so a miss below would prove nothing")

        # ... and a chassis row never answers for a group that left it.
        # `scaleout` is asked too: nothing has been measured there, so
        # every one of these must miss outright.
        for reach in (COMM_REACH_SCALEUP, COMM_REACH_SCALEOUT):
            for key in host_keys:
                crossing = replace(key, shape_json=json.dumps(
                    {**json.loads(key.shape_json), COMM_DOMAIN_AXIS: int(reach)},
                    separators=(",", ":"), sort_keys=True))
                self.assertEqual(
                    registry.get_physical(crossing), [],
                    f"reach {reach} found a row keyed for another reach")
                self.assertIsNone(
                    timing(crossing),
                    f"reach {reach} was handed a measurement of another")

    def test_the_measured_corpus_keys_each_row_at_its_own_reach(self):
        """What the corpus says about where its rows were taken.

        The chassis rows must key exactly as they did before the axis
        existed -- they are the backward-compatibility claim, and not one
        of them was re-measured to gain it. The rows taken across two
        chassis of the supernode must carry `scaleup`, because a reader
        that cannot tell them apart would answer a cross-fabric
        deployment with a chassis measurement.
        """
        import csv
        import json
        from backend.inference.op_calibration import BenchmarkKey
        path = PROJECT_ROOT / "tests/golden/ascend_vllm_ascend.csv"
        with open(path, newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["op"] in COMM_OPS]
        self.assertTrue(rows, "no collective rows to check")

        chassis = crossing = 0
        for row in rows:
            stated = json.loads(row["shape_json"]).get(COMM_DOMAIN_AXIS)
            keyed = BenchmarkKey.from_csv_row(row).to_physical().shape_json
            if stated is None:
                chassis += 1
                self.assertNotIn(COMM_DOMAIN_AXIS, keyed, row["op"])
            else:
                crossing += 1
                self.assertEqual(stated, COMM_REACH_SCALEUP, row["op"])
                self.assertIn(f'"{COMM_DOMAIN_AXIS}":{COMM_REACH_SCALEUP}',
                              keyed, row["op"])
        self.assertTrue(chassis, "the chassis rows are the compatibility claim")
        self.assertTrue(crossing, "no row records a group that crossed "
                                  "machines, so nothing pins the axis on "
                                  "real data")


if __name__ == "__main__":
    unittest.main()


class ASupernodeIsScaleUpAcrossMachines(unittest.TestCase):
    """Both halves of the question, on the sheet row that answers it.

    `gpus_per_node` is the CHASSIS and `ranks_per_scaleup_domain` is the
    fast fabric's reach. They coincide on every card that has no
    supernode, and on an Atlas A3 they do not: 16 against 768. Getting
    that wrong is not a crash, it is 32 ranks priced on a 25 GB/s NIC
    that carried none of their traffic.
    """

    def _ascend(self):
        from backend.inference.gpu_specs import get_gpu
        gpu = get_gpu("Ascend-910C-A3")
        self.assertIsNotNone(gpu, "the Ascend row is what this tests")
        return gpu

    def test_the_sheet_separates_the_chassis_from_the_domain(self):
        from backend.inference.gpu_specs import (gpus_per_node,
                                                 ranks_per_scaleup_domain)
        gpu = self._ascend()
        self.assertEqual(gpus_per_node(gpu), 16)
        self.assertEqual(ranks_per_scaleup_domain(gpu), 768)

    def test_the_lookup_tiers_a_crossing_group_as_scaleup(self):
        """A measured row keyed `scaleup` is only found if the estimate
        asks for `scaleup`."""
        from backend.inference.gpu_specs import (gpus_per_node,
                                                 ranks_per_scaleup_domain)
        gpu = self._ascend()
        host, domain = gpus_per_node(gpu), ranks_per_scaleup_domain(gpu)
        for world, want in ((8, COMM_REACH_HOST), (16, COMM_REACH_HOST),
                            (32, COMM_REACH_SCALEUP),
                            (64, COMM_REACH_SCALEUP),
                            (768, COMM_REACH_SCALEUP),
                            (769, COMM_REACH_SCALEOUT),
                            (1024, COMM_REACH_SCALEOUT)):
            self.assertEqual(
                comm_reach(world, base_stride=1, ranks_per_host=host,
                           ranks_per_scaleup_domain=domain),
                want, f"world={world}")

    def test_a_world_with_no_measurement_is_priced_on_the_fast_fabric(self):
        """The half a reach code cannot cover.

        64 and 96 ranks are not in the corpus and will be priced from the
        roofline. Inside the supernode that traffic never touches a NIC,
        so pricing it on `rdma_gbps` would be wrong by the better part of
        an order of magnitude -- and silently, since a number comes back
        either way.
        """
        gpu = self._ascend()
        fabric = Fabric.of(gpu, 2e-6)
        for world in (16, 32, 64, 96, 768):
            _t, phases = price_collective(
                op="all_reduce", world=world, tensor_bytes=1 << 23,
                count_per_step=1, fabric=fabric, base_stride=1)
            labels = {p["fabric_label"] for p in phases}
            self.assertEqual(labels, {"HCCS"}, f"world={world} left the "
                                               f"scale-up fabric: {labels}")

    def test_past_the_domain_it_is_the_nic(self):
        from backend.inference.comm_cost import collective_phases
        from backend.inference.gpu_specs import NoFabricBandwidth
        fabric = Fabric.of(self._ascend(), 2e-6)
        phases = collective_phases(
            op="all_reduce", world=1024, tensor_bytes=float(1 << 23),
            gpus_per_node=fabric.scaleup_ranks(), base_stride=1)
        self.assertIn("inter", {p["fabric"] for p in phases})
        # ... which this row does not describe: its RoCE ports have never
        # carried traffic, so the cell is empty and the group is refused
        # rather than priced on a nameplate.
        with self.assertRaises(NoFabricBandwidth):
            price_collective(
                op="all_reduce", world=1024, tensor_bytes=1 << 23,
                count_per_step=1, fabric=fabric, base_stride=1)

    def test_the_hops_are_read_off_the_split_the_bytes_are(self):
        """A collective's time has two halves -- bytes over a bandwidth and
        hops at a latency -- and the row publishes the envelope's sum of
        them while its phase rows itemise the same collective. Both halves
        have to ask the one question, "did this phase leave the fast
        fabric", of the one width.

        They did not: the bytes were split at the 768-rank domain and the
        hops at the 16-die chassis, so 32 ranks moved their bytes over HCCS
        and still waited on an RDMA hop per traversal. The published time of
        an all-to-all there was 36.94us against phase rows summing to
        14.94us. Pricing the bytes alone, as the test above does, could not
        see it.

        The card is the 910C row with its NIC stated. The two splits charge
        differently only where a hop between domains costs more than one
        inside, and the sheet's 910C row states no RDMA hop at all -- on it
        the old split and this one agree, and a test on it passes against
        the bug it was written for.
        """
        from backend.inference.comm_cost import CollectiveSignature
        fabric = Fabric.of(replace(self._ascend(), rdma_gbps=25.0,
                                   rdma_hop_latency_us=4.4), 2e-6)
        self.assertGreater(fabric.inter_latency, 0)
        payload = float(1 << 20)
        for op in ("all_to_all", "all_reduce"):
            for world in (32, 64, 768):
                t_rows, rows = price_collective(
                    op=op, world=world, tensor_bytes=payload,
                    count_per_step=1, fabric=fabric, base_stride=1)
                hops = collective_latency(
                    op=op, world=world, count_per_step=1, fabric=fabric,
                    base_stride=1)
                self.assertAlmostEqual(
                    hops, sum(r["latency_s"] for r in rows), places=12,
                    msg=f"{op} world={world}: hops split elsewhere")
                self.assertAlmostEqual(
                    t_rows, sum(r["t_phase_s"] for r in rows), places=12)
                published = CollectiveSignature(
                    op=op, world=world, count=1, stride=1).envelope(
                        fabric).choose(payload)
                self.assertEqual(published.algorithm, "nccl")
                self.assertAlmostEqual(
                    published.t(payload), t_rows, places=12,
                    msg=f"{op} world={world}: the row's time is not its "
                        "phase rows' sum")

    def test_a_card_with_no_supernode_is_unchanged(self):
        """Every other row on the sheet leaves the column empty, and must
        read exactly as it did: this is what keeps the GPU baseline
        byte-identical."""
        from backend.inference import gpu_specs as gs
        for gpu in gs.load_gpu_specs():
            if gpu.sku == "Ascend-910C-A3":
                continue
            self.assertEqual(
                gs.ranks_per_scaleup_domain(gpu), gs.gpus_per_node(gpu),
                f"{gpu.sku} gained a domain it never declared")


class APublishedCollectiveIsItsPhasesSum(unittest.TestCase):
    """A collective row publishes one time (`t_comm_s`, the row law's `Min`
    over the envelope's candidates) and itemises the same collective as
    phase rows (`RowRoofline.comm`, the phase loop in `price_collective`).
    They are two derivations of one number by different code, so each is
    the other's oracle: on every collective row of a real walk, the
    published time is the sum of its phases' times.

    `TestCommBandwidthLaneSplit` asserts the same sum, at a link latency of
    0 on a card with no supernode, which is where the two derivations
    cannot disagree: they differed only in where a hop was charged, and a
    free hop costs the same everywhere. So the walk here pays the
    estimate's own link latency, and the first card is a supernode whose
    hops between domains cost more than those inside -- the 910C row with
    its NIC stated. Against the split 9504f28 removed, every collective row
    of that walk fails.

    Single-phase rows publish no breakdown (`comm_phases` is empty), so
    the phases are re-read from the row's own signature and payload rather
    than taken off the row."""

    def _walk(self, gpu, framework, world):
        from backend.analyzer import fetch_config
        from backend.inference import gpu_specs as gs
        from backend.inference.framework_bindings import get_binding
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        from tests.snapshots import shared_loader
        model = "meta-llama/Llama-2-7b-hf"
        fabric = Fabric.of(gpu, 2e-6)
        rows = analyze(
            shared_loader().match(fetch_config(model), model_id=model),
            parallel=ParallelConfig(tp_attention=world, tp_dense_ffn=world),
            framework_binding=get_binding(framework), batch_size=1,
            avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15,
            peak_hbm_bw_bytes_per_sec=gs.peak_hbm_bw(gpu), gpu=gpu,
            intra_bw_bytes_per_sec=fabric.intra_bw,
            inter_bw_bytes_per_sec=fabric.inter_bw,
            gpus_per_node=fabric.gpus_per_node,
            intra_bw_label=fabric.intra_label,
            inter_bw_label=fabric.inter_label,
            comm_latency_s=fabric.step_latency, breakdown=True,
        ).per_op_timing
        return rows, fabric

    def _assert_rows_are_their_phases(self, gpu, framework, world):
        from backend.inference.comm_cost import CollectiveSignature
        from backend.inference.row_roofline import RowRoofline
        rows, fabric = self._walk(gpu, framework, world)
        checked = 0
        for row in rows:
            sig = CollectiveSignature.of(row)
            if sig is None:
                continue
            _t, phases = RowRoofline.comm(
                sig, tensor_bytes=row["comm_tensor_bytes"], fabric=fabric)
            with self.subTest(sku=gpu.sku, op=row["op"], world=world):
                self.assertTrue(phases)
                self.assertAlmostEqual(
                    row["t_comm_s"], sum(p["t_phase_s"] for p in phases),
                    delta=1e-9 * row["t_comm_s"])
            checked += 1
        self.assertGreater(checked, 0, f"{gpu.sku}: no collective to check")

    def test_on_a_supernode_whose_hops_between_domains_cost_more(self):
        from backend.inference import gpu_specs as gs
        nic = replace(gs.get_gpu("Ascend-910C-A3"), rdma_gbps=25.0,
                      rdma_hop_latency_us=4.4)
        self._assert_rows_are_their_phases(nic, "vllm_ascend", 32)

    def test_on_the_sheet_as_it_stands(self):
        """The rows published today: the 910C inside its supernode, and a
        group that leaves an H100 node over RDMA in three phases."""
        from backend.inference import gpu_specs as gs
        self._assert_rows_are_their_phases(
            gs.get_gpu("Ascend-910C-A3"), "vllm_ascend", 32)
        self._assert_rows_are_their_phases(gs.get_gpu("H100-SXM"), "vllm", 16)


class AFabricTheCardDoesNotStateIsNotFree(unittest.TestCase):
    """A phase's time is `bytes / bw` plus its hops. A card whose sheet row
    leaves a fabric's bandwidth blank used to have the division skipped, so
    a group that left the machine moved its bytes in zero seconds and paid
    only the hop latency: a cross-node all-reduce of 8 MiB priced at 4us.
    A number came back, lower than any real link gives, with nothing beside
    it to say a column was empty."""

    def _card(self, sku, **blank):
        from backend.inference import gpu_specs as gs
        return replace(gs.get_gpu(sku), **blank)

    def test_a_group_that_leaves_the_stated_fabric_is_refused(self):
        from backend.inference.comm_cost import CollectiveSignature
        from backend.inference.gpu_specs import NoFabricBandwidth
        gpu = self._card("Ascend-910C-A3", rdma_gbps=None,
                         rdma_hop_latency_us=None)
        fabric = Fabric.of(gpu, 2e-6)
        self.assertIsNone(fabric.inter_bw)
        for op in ("all_reduce", "all_to_all"):
            with self.assertRaises(NoFabricBandwidth) as refused:
                price_collective(
                    op=op, world=1024, tensor_bytes=1 << 23,
                    count_per_step=1, fabric=fabric, base_stride=1)
            self.assertEqual(refused.exception.field, "rdma_gbps")
            self.assertIn("Ascend-910C-A3", str(refused.exception))
            # The row's published time is the envelope's, built before any
            # payload is known: it is refused there too, not left at zero
            # seconds per byte.
            with self.assertRaises(NoFabricBandwidth):
                CollectiveSignature(op=op, world=1024, count=1,
                                    stride=1).envelope(fabric)

    def test_what_stays_on_a_stated_fabric_is_priced_as_before(self):
        """The refusal is for the fabric a phase crosses, not for the card:
        blanking the NIC changes nothing a group inside the supernode
        pays."""
        from backend.inference import gpu_specs as gs
        stated = Fabric.of(gs.get_gpu("Ascend-910C-A3"), 2e-6)
        blank = Fabric.of(self._card("Ascend-910C-A3", rdma_gbps=None,
                                     rdma_hop_latency_us=None), 2e-6)
        for world in (8, 16, 32, 64, 768):
            for op in ("all_reduce", "all_to_all"):
                kw = dict(op=op, world=world, tensor_bytes=1 << 23,
                          count_per_step=1, base_stride=1)
                self.assertEqual(price_collective(fabric=blank, **kw),
                                 price_collective(fabric=stated, **kw),
                                 f"{op} world={world}")

    def test_the_fabric_inside_the_domain_is_held_to_the_same_rule(self):
        from backend.inference.gpu_specs import NoFabricBandwidth
        gpu = self._card("H100-SXM", scaleup_gbps=None, scaleup_fabric=None,
                         pcie_gbps=None, nvls="no")
        with self.assertRaises(NoFabricBandwidth) as refused:
            price_collective(
                op="all_reduce", world=8, tensor_bytes=1 << 23,
                count_per_step=1, fabric=Fabric.of(gpu, 2e-6), base_stride=1)
        self.assertEqual(refused.exception.field, "scaleup_gbps")

    def test_no_bandwidth_handed_over_is_still_a_request_for_bytes(self):
        """A fabric handed neither bandwidth is a caller that asked for no
        times (`row_roofline.LawBook.row_at`), with or without a card: a
        kernel plan and a fusion walk pass the card for its dispatch and no
        interconnect, which is how `phases` builds the second fabric below.
        An absent bandwidth there is "not asked", and the phases carry
        their bytes and hops as they did. Refusing it took every plan built
        on an H100 down with a complaint about a cell the sheet fills."""
        from backend.inference import gpu_specs as gs
        from backend.inference.comm_cost import topology
        bare = Fabric(8, None, None, step_latency=2e-6)
        for fabric in (bare, bare._replace(**topology(gs.get_gpu("H100-SXM")))):
            t, phases = price_collective(
                op="all_reduce", world=16, tensor_bytes=1 << 23,
                count_per_step=1, fabric=fabric, base_stride=1)
            self.assertEqual({p["bw_bytes_per_sec"] for p in phases}, {0.0})
            self.assertAlmostEqual(t, sum(p["latency_s"] for p in phases))

    def test_which_cards_on_the_sheet_leave_a_fabric_unstated(self):
        """Every row states the fabric inside its scale-up domain, and
        every row but one the link between domains -- so the refusal moves
        no number on any other card. The one is the Ascend 910C, whose NIC
        has never been observed carrying traffic."""
        from backend.inference import gpu_specs as gs
        from backend.inference.gpu_specs import NoFabricBandwidth
        for gpu in gs.load_gpu_specs():
            fabric = Fabric.of(gpu, 2e-6)
            with self.subTest(sku=gpu.sku):
                self.assertEqual(fabric.sku, gpu.sku)
                self.assertTrue(fabric.bandwidth("intra"))
                if gpu.sku == "Ascend-910C-A3":
                    with self.assertRaises(NoFabricBandwidth):
                        fabric.bandwidth("inter")
                else:
                    self.assertTrue(fabric.bandwidth("inter"))


class TestTheScaleUpDomainReachesTheEstimate(unittest.TestCase):
    """The sheet's `ranks_per_scaleup_domain` has to arrive at the walk that
    prices a deployment, not merely at `Fabric.of`.

    It did not, once: `Fabric.of` passed the width and the phase walk built
    its fabric by hand without it, so every Ascend group wider than one
    16-die chassis was split at `gpus_per_node` and its middle phase priced
    as RDMA with a 4.4us hop -- the error the column exists to remove, on
    the one path an estimate actually takes. A test that hands a hand-made
    `Fabric` to `price_collective` cannot see this; it proves what the
    pricing does GIVEN a fabric, and what broke was which fabric arrives.
    So this drives `topology()`, the one both construction sites share.
    """

    def test_topology_carries_the_domain_width(self):
        from backend.inference import gpu_specs as gs
        from backend.inference.comm_cost import Fabric, topology

        ascend = gs.get_gpu("Ascend-910C-A3")
        self.assertEqual(topology(ascend).get("scaleup_domain"), 768)
        # Both construction sites agree, because both read `topology`.
        self.assertEqual(Fabric.of(ascend).scaleup_domain, 768)

        # A card that declares no supernode reads as its own chassis, which
        # is what `None` meant before: no card changes behaviour here.
        for gpu in gs.load_gpu_specs():
            if gpu.sku == "Ascend-910C-A3":
                continue
            self.assertEqual(topology(gpu).get("scaleup_domain"),
                             gs.gpus_per_node(gpu),
                             f"{gpu.sku} would split somewhere new")

    def test_a_group_inside_the_supernode_crosses_no_rdma(self):
        """32 Ascend dies span two chassis and stay in one scale-up domain,
        so the whole collective is HCCS: 174.5 GB/s measured, against the
        25 GB/s an RDMA split would have charged."""
        from backend.inference.comm_cost import (Fabric, collective_phases,
                                                 topology)
        from backend.inference.gpu_specs import get_gpu

        fabric = Fabric.of(get_gpu("Ascend-910C-A3"))
        self.assertEqual(fabric.gpus_per_node, 16)

        for world in (8, 16, 32, 64, 768):
            phases = collective_phases(
                op="all_reduce", world=world, tensor_bytes=float(1 << 20),
                gpus_per_node=int(fabric.scaleup_domain),
                base_stride=1, algorithm="nccl")
            crossings = [p for p in phases if p.get("fabric") == "inter"]
            self.assertEqual(
                crossings, [],
                f"world={world} is inside the 768-rank domain and must not "
                f"cross: {crossings}")

        # Past the domain it must split, or the fabric would be infinite.
        beyond = collective_phases(
            op="all_reduce", world=1024, tensor_bytes=float(1 << 20),
            gpus_per_node=int(fabric.scaleup_domain),
            base_stride=1, algorithm="nccl")
        self.assertTrue(
            [p for p in beyond if p.get("fabric") == "inter"],
            "a group wider than the supernode has to leave it")

        # The width is what splits it, not the chassis: splitting at 16
        # would have crossed at 32.
        at_chassis = collective_phases(
            op="all_reduce", world=32, tensor_bytes=float(1 << 20),
            gpus_per_node=int(fabric.gpus_per_node),
            base_stride=1, algorithm="nccl")
        self.assertTrue(
            [p for p in at_chassis if p.get("fabric") == "inter"],
            "this is the behaviour the domain width replaces; if it stopped "
            "crossing, the test above no longer discriminates")
        _ = topology
