"""One member of a fusion may publish two of its tensors.

`hc_split_sinkhorn` writes three: `pre`, its main output, read by the
weighted reduce beside it, and `post` / `comb`, read by the post-mix outside
the component. A launch that absorbs the whole pre-mix — the Ascend `HcPre`
kernel does — has to publish the last two, and `FusedNode.output_ports` keys
a port by its member alone, so the second port silently overwrote the first.
`secondary_ports` carries a member's published secondary outputs, keyed by
the member and the port of its own a reader names.

Falsification targets, on GLM-5.3-Flash's real walk with the plan the
Ascend binding will carry:
  - both ports are published, each under its own name, and `output_ports`
    still holds the member whose main output is the launch's;
  - the outside readers (`hc_combine`'s broadcast of `post`, its `hc_mix`
    of `comb`) are rewired onto the launch at those ports;
  - the boundary shape carries both tensors and the traffic bills them;
  - leaf cost survives the rewrite exactly;
  - every plan that publishes only `<node>.out` — all of them before this
    field existed, the GPU bindings included — has an empty
    `secondary_ports`, on this very model.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval import dim as dim_mod
from backend.arch.eval import traffic
from backend.arch.eval.fusion import (FusedNode, Fusion, Framework,
                                      rebase_analysis)
from backend.arch.eval.transforms import Deployment
from backend.arch.schema.fusion import FusionOpSpec
from backend.inference.framework_bindings import get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES
from tests.snapshots import shared_program, shared_walk

MODEL = "zai-org__GLM-5.3-Flash"

#: The whole of `hc_pre` as one launch — what `_C_ascend.npu_hc_pre_v2`
#: runs (profiler `HcPre`): the flatten, the unscaled norm, the mix
#: projection, the Sinkhorn split and the weighted reduce. Not the sublayer
#: RMSNorm after it, which the device runs as its own kernel.
MHC_PRE = FusionOpSpec(naive_pattern={
    "nodes": {
        "flat":       {"op": "view_merge",             "inputs": ["external.x"]},
        "norm":       {"op": "rmsnorm_unscaled",       "inputs": ["flat.out"]},
        "mixes":      {"op": "linear_external_weight", "inputs": ["norm.out", "external.fn"]},
        "split":      {"op": "hc_split_sinkhorn",      "inputs": ["mixes.out", "external.scale", "external.base"]},
        "w_expanded": {"op": "broadcast",              "inputs": ["split.out"]},
        "weighted":   {"op": "mul",                    "inputs": ["external.x", "w_expanded.out"]},
        "reduced":    {"op": "sum_reduce",             "inputs": ["weighted.out"]},
    },
    "io_mapping": {
        "x": "external.x", "fn": "external.fn",
        "scale": "external.scale", "base": "external.base",
        "output": "reduced.out",
        "post": "split.post",
        "comb": "split.comb",
    },
})


def _fusion() -> Fusion:
    return Fusion(scheme_name="mhc_pre_test", scheme=MHC_PRE,
                  crosses="component")


def _deployment() -> Deployment:
    return Deployment(shared_program(MODEL)).at(batch=1, input_tokens=128)


def _leaf_totals(analysis) -> tuple:
    flops = params = 0
    for uid, cost in analysis.cost.values.items():
        if getattr(analysis.program.by_uid(uid), "children", None):
            continue
        flops += dim_mod.value_of(cost.flops)
        params += dim_mod.value_of(cost.params)
    return flops, params


def _elements(dims) -> int:
    out = 1
    for d in dims:
        out *= int(dim_mod.value_of(d))
    return out


class TestOneMemberTwoPorts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.naive = shared_walk(_deployment())
        cls.fused_program, cls.rewrite = _fusion().rewrite(
            cls.naive.program, cls.naive)
        cls.launches = [n for n in cls.fused_program.nodes()
                        if isinstance(n, FusedNode)]
        cls.splits = {n.uid for n in cls.naive.program.nodes()
                      if n.op == "hc_split_sinkhorn"}

    def test_the_plan_matches_every_pre_mix(self):
        self.assertFalse(self.rewrite.is_identity())
        # every hc_split_sinkhorn is absorbed: no match was refused for
        # the two outside readers of its post / comb
        self.assertTrue(self.splits <= set(self.rewrite.dropped))
        self.assertEqual(len(self.launches), len(self.splits))

    def test_both_secondary_ports_are_published_by_their_names(self):
        for launch in self.launches:
            (split,) = [uid for uid in launch.origins if uid in self.splits]
            self.assertEqual(launch.secondary_ports,
                             {(split, "post"): "post", (split, "comb"): "comb"})
            # the main output is the reduce's, and split's `pre` stays inside
            self.assertEqual(list(launch.output_ports.values()), [""])
            self.assertNotIn(split, launch.output_ports)
            self.assertEqual(launch.published_port(split, "post"), "post")
            self.assertEqual(launch.published_port(split, "comb"), "comb")
            self.assertIsNone(launch.published_port(split, ""))
            self.assertEqual(
                sorted(p for _u, _o, p in launch.publishers()),
                ["", "comb", "post"])

    def test_a_reader_of_either_port_resolves_through_the_launch(self):
        """The post-mix names the pre-mix COMPONENT (`attn_hc.comb`), which
        survives the rewrite, so the edge keeps that producer. The
        retargeting happens where the tensor is priced: traffic resolves the
        component to its writing leaf, the leaf to the launch that absorbed
        it, and the leaf's OWN port to the one the launch publishes it at.

        Both ports are checked, and they are checked differently because the
        graph reads them differently. `comb` feeds `hc_mix`, which is priced
        and carries a read edge, so the edge itself is the evidence. `post`
        feeds a `broadcast`, a view that records no read edge at all (it
        does not in the unfused walk either) — so the evidence there is the
        resolution itself: asking how that leaf's `post` is written, with
        the launch in hand, has to answer with the launch's `post` port.

        Keyed by member alone, the second port overwrote the first and this
        resolution raised "publishes no port for it".
        """
        rebased = rebase_analysis(self.naive, self.fused_program, self.rewrite)
        billed = traffic.derive(rebased)
        by_launch = {uid: launch for launch in self.launches
                     for uid in launch.origins}

        # -- comb: a priced read edge, resolved and billed at its own size --
        comb_reads = 0
        for node in self.fused_program.nodes():
            row = billed.values.get(node.uid)
            if row is None:
                continue
            for edge in row.in_edges:
                if edge.producer is None:
                    continue
                producer = self.naive.program.by_uid(edge.producer)
                if producer is None:
                    continue
                leaf, port = self.naive.program.output_leaf(
                    producer, edge.port or "out")
                if leaf.uid not in self.splits:
                    continue
                self.assertEqual(port, "comb")
                self.assertEqual(node.op, "hc_mix")
                self.assertEqual(
                    by_launch[leaf.uid].published_port(leaf.uid, port), "comb")
                self.assertEqual(
                    int(dim_mod.value_of(edge.elements)),
                    _elements(self.naive.shapes.values[leaf.uid].ports["comb"]))
                comb_reads += 1
        self.assertEqual(comb_reads, len(self.launches))

        # -- post: no read edge (its consumer is a view), so ask traffic
        #    how that port is written with the launch in hand --
        for launch in self.launches:
            (split,) = [uid for uid in launch.origins if uid in self.splits]
            launch_of = {uid: launch for uid in launch.origins}
            leaf = self.naive.program.by_uid(split)
            for port in ("post", "comb"):
                # Resolving must REACH the launch. A plain tensor answers
                # None here — "written at the deployment's activation
                # width" — which is the common case and what the caller
                # falls back on; the failure this catches is the raise.
                traffic.declared_output(
                    leaf, program=self.fused_program, launch_of=launch_of,
                    port=port, shapes=rebased.shapes)
                # and the port names the member it says it does
                self.assertEqual(
                    traffic.produced_dtype(launch, port, rebased.precision,
                                           self.naive.program),
                    traffic.produced_dtype(leaf, port, rebased.precision,
                                           self.naive.program))
            with self.assertRaises(ValueError):
                # `pre` stays inside the launch: nothing publishes it, and
                # a reader of it would be an edge the rewrite cut.
                traffic.declared_output(
                    leaf, program=self.fused_program, launch_of=launch_of,
                    port="", shapes=rebased.shapes)

    def test_the_boundary_carries_both_tensors_and_traffic_bills_them(self):
        rebased = rebase_analysis(self.naive, self.fused_program, self.rewrite)
        billed = traffic.derive(rebased)
        for launch in self.launches:
            (split,) = [uid for uid in launch.origins if uid in self.splits]
            before = self.naive.shapes.values[split]
            shape = rebased.shapes.values[launch.uid]
            self.assertEqual(shape.ports["post"], before.ports["post"])
            self.assertEqual(shape.ports["comb"], before.ports["comb"])
            self.assertNotIn("", shape.ports)
            row = billed.values[launch.uid]
            written = {term.port: term for term in row.out_terms}
            self.assertEqual(set(written), {"", "post", "comb"})
            self.assertEqual(int(dim_mod.value_of(written["comb"].elements)),
                             _elements(before.ports["comb"]))
            self.assertEqual(int(dim_mod.value_of(written["post"].elements)),
                             _elements(before.ports["post"]))

    def test_leaf_cost_survives_the_rewrite(self):
        d = _deployment()
        self.assertEqual(_leaf_totals(shared_walk(d)),
                         _leaf_totals(shared_walk(d.fused(_fusion()))))


class TestPlansPublishingOnlyMainOutputsAreUntouched(unittest.TestCase):
    def test_gpu_bindings_have_no_secondary_port_on_this_model(self):
        for name in ("vllm", "sglang"):
            framework = Framework.of(get_binding(name), FUSION_SCHEMES)
            fused = shared_walk(_deployment().fused(framework))
            launches = [n for n in fused.program.nodes()
                        if isinstance(n, FusedNode)]
            self.assertTrue(launches, name)
            for launch in launches:
                self.assertEqual(launch.secondary_ports, {}, (name, launch.op))
                self.assertEqual(
                    launch.publishers(),
                    tuple((uid, "", port)
                          for uid, port in launch.output_ports.items()))


if __name__ == "__main__":
    unittest.main()
