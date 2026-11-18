"""Direct regression net for how the mesh hands its layout to the layout
groups: `spans` (each world's stride and size) resolved into explicit
`ProcessGroupFamily` members, whose relations (`same` / `refines` /
`orthogonal` / `overlaps`) are the only statement of how two worlds relate,
plus `subgroup_stride` and its companion `subgroup_degrees`.

The mesh once stated those relations a second time by stride arithmetic
(`alias_map`, an equivalence on spans, and `contains`, a containment test).
No consumer read them after the layout groups took over, and their documented
soft spot — the dp×cp×ep token cut falling in three alias classes — was a
fact about them, not about the engine, which asks by membership and prices
that seam as the all_to_all pair (`tests/backend/arch/layout/
test_moe_ordered_collectives.py::test_two_world_token_cut_exchanges_like_one_world`).
The cases they pinned are kept here, asked of the families instead."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.groups import orthogonal, overlaps, refines, same
from backend.arch.layout.resolver import families_for
from backend.inference import mesh
from backend.inference.parallel_config import ParallelConfig


def _families(**kw) -> dict:
    pc = ParallelConfig(**kw)
    pc.validate()
    return families_for(pc, "prefill")


class TestFamilyRelations(unittest.TestCase):
    def test_three_tp_worlds_at_equal_degree_are_one_group(self):
        # Stride 1 in each region, same size: the same ranks. This is the
        # fact that lets sequence parallelism meet attention and FFN without
        # a spurious re-shard per block.
        f = _families(tp_attention=8, tp_dense_ffn=8, tp_moe_expert=8)
        self.assertTrue(same(f["tp_dense_ffn"], f["tp_attention"]))
        self.assertTrue(same(f["tp_moe_expert"], f["tp_attention"]))

    def test_tp_worlds_at_different_degrees_are_different_groups(self):
        f = _families(tp_attention=8, tp_dense_ffn=4, cp=2,
                      tp_moe_expert=8)
        self.assertFalse(same(f["tp_attention"], f["tp_dense_ffn"]))
        self.assertTrue(refines(f["tp_dense_ffn"], f["tp_attention"]))

    def test_moe_seam_request_cut_is_the_expert_world(self):
        f = _families(dp_attention=8, ep_moe=8)
        self.assertTrue(same(f["ep_moe"], f["dp_attention"]))

    def test_moe_seam_position_cut_is_the_expert_world(self):
        # The REGION_AXES["moe"] no-cp comment, as behavior.
        f = _families(cp=8, ep_moe=8)
        self.assertTrue(same(f["ep_moe"], f["cp"]))

    def test_contained_cut_refines_without_being_the_same(self):
        # cp spans (2, 4) inside ep_moe's (1, 8).
        f = _families(tp_attention=2, cp=4, ep_moe=8)
        self.assertFalse(same(f["cp"], f["ep_moe"]))
        self.assertTrue(refines(f["cp"], f["ep_moe"]))
        self.assertFalse(refines(f["ep_moe"], f["cp"]))

    def test_dp_cp_ep_token_cut_tiles_the_expert_world(self):
        # Both token worlds lie inside the expert world, neither inside the
        # other, and together they tile it: the shape a span equivalence
        # split into three classes.
        f = _families(cp=2, dp_attention=2, ep_moe=4,
                      tp_attention=2, tp_dense_ffn=2, tp_moe_expert=2)
        self.assertTrue(refines(f["cp"], f["ep_moe"]))
        self.assertTrue(refines(f["dp_attention"], f["ep_moe"]))
        self.assertFalse(refines(f["cp"], f["dp_attention"]))
        self.assertFalse(refines(f["dp_attention"], f["cp"]))
        self.assertTrue(orthogonal(f["cp"], f["dp_attention"]))

    def test_expert_tp_wider_than_attention_tp_crosses_the_token_cut(self):
        # tp_attention=1 against tp_moe_expert=4: ep_moe sits at stride 4
        # while the token worlds start at stride 1, so an expert group holds
        # ranks with different tokens without lying inside either token
        # cut — `cp` crosses it, and it lies inside `dp_attention` rather
        # than the other way round.
        f = _families(cp=2, dp_attention=4, tp_moe_expert=4, ep_moe=2)
        self.assertTrue(orthogonal(f["cp"], f["ep_moe"]))
        self.assertTrue(refines(f["ep_moe"], f["dp_attention"]))
        self.assertFalse(refines(f["dp_attention"], f["ep_moe"]))
        self.assertFalse(overlaps(f["dp_attention"], f["ep_moe"]))

    def test_axis_at_degree_one_has_no_family(self):
        f = _families(dp_attention=8, ep_moe=8)
        self.assertNotIn("cp", f)
        self.assertNotIn("cp", mesh.spans(mesh.axis_degrees(
            ParallelConfig(dp_attention=8, ep_moe=8))))


class TestSubgroupStrideAndDegrees(unittest.TestCase):
    def test_dcp_inside_tp_reuses_tp_ranks_at_stride_one(self):
        # No pcp: dcp partitions tp_attention's ranks directly.
        pc = ParallelConfig(tp_attention=8, dcp=8)
        self.assertEqual(mesh.subgroup_stride(pc, "dcp"), 1)
        self.assertEqual(mesh.subgroup_degrees(pc, "dcp"), (1, 2, 4, 8))

    def test_dcp_fills_cp_before_tp(self):
        # dcp == pcp: the group is one PCP column, spaced by the tp width
        # below it — vLLM's transpose-then-reshape, as a stride.
        pc = ParallelConfig(tp_attention=4, cp=2, dcp=2)
        self.assertEqual(mesh.subgroup_stride(pc, "dcp"), 4)
        self.assertEqual(mesh.subgroup_degrees(pc, "dcp"), (1, 2, 8))

    def test_dcp_swallowing_cp_spills_into_tp(self):
        # dcp == tp*pcp: the group swallows the PCP column whole and runs
        # contiguous through TP.
        pc = ParallelConfig(tp_attention=4, cp=2, dcp=8)
        self.assertEqual(mesh.subgroup_stride(pc, "dcp"), 1)
        self.assertEqual(mesh.subgroup_degrees(pc, "dcp"), (1, 2, 8))


if __name__ == "__main__":
    unittest.main()
