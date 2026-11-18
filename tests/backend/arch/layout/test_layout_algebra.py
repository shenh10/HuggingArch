"""Property tests for the layout closed vocabularies — the semantic
appendix's oracle, run on small exhaustive configurations. Each class
discharges one proof obligation: constructor O/I/P relations and unit
laws (including the no-bypass rule: raw dataclasses reject ill-formed and
non-canonical construction), well-formedness => coverage/boundedness/
symmetry, the cross_replica_family derived family, ReductionHomomorphism merge-homomorphism
gates with per-law counterexamples, and conservation under padding
(per-part deduplicated)."""
from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch import layout
from backend.arch.eval import dim
from backend.arch.layout.groups import ProcessGroupFamily


def fam(id, *groups):
    return ProcessGroupFamily(id=id, groups=tuple(tuple(g) for g in groups))


TP8 = fam("tp", range(8))                       # one group of eight
TP8x2 = fam("tp2", range(8), range(8, 16))      # two replica groups
DP2 = fam("dp", (0, 4), (1, 5), (2, 6), (3, 7))  # orthogonal to TP4
TP4 = fam("tp4", (0, 1, 2, 3), (4, 5, 6, 7))
PAIRS = tuple((2 * i, 2 * i + 1) for i in range(4))
SINGLETONS8 = tuple((i,) for i in range(8))

#: Two workload counts as the walk declares them (`transforms.Workload`):
#: one that may be zero (a prompt, a cached prefix) and one that is at least
#: one (the batch, the verify width).
WHOLE_COUNT = sp.Symbol("s_prefix", integer=True, nonnegative=True)
POSITIVE_COUNT = sp.Symbol("s_ctx", integer=True, positive=True)


class TestGroupRelations(unittest.TestCase):
    def test_membership_equality_ignores_id_and_listing_order(self):
        a = fam("a", (0, 1), (2, 3))
        b = fam("b", (2, 3), (0, 1))
        self.assertTrue(layout.same(a, b))
        self.assertEqual(a, b)                    # dataclass-level too
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(layout.sharded(a), layout.sharded(b))  # layouts follow

    def test_member_order_is_semantic(self):
        self.assertFalse(layout.same(fam("a", (0, 1)), fam("b", (1, 0))))

    def test_subgroup_refines(self):
        dcp = fam("dcp", (0, 1), (2, 3), (4, 5), (6, 7))
        self.assertTrue(layout.refines(dcp, TP8))
        self.assertFalse(layout.refines(TP8, dcp))

    def test_orthogonal_by_transposition(self):
        self.assertTrue(layout.orthogonal(TP4, DP2))
        self.assertFalse(layout.orthogonal(TP4, TP4))

    def test_interleaved_two_cycles_are_not_orthogonal(self):
        # codex's counterexample: pairwise coordinate pairs are distinct in
        # every group-intersection singleton, but the six-rank component
        # sees only two distinct pairs.
        a = fam("a", (0, 1), (2, 3), (4, 5))
        b = fam("b", (1, 2), (3, 4), (5, 0))
        self.assertFalse(layout.orthogonal(a, b))

    def test_partial_overlap_is_its_own_relation(self):
        a = fam("a", (0, 1, 2))
        b = fam("b", (2, 3, 4))
        self.assertTrue(layout.overlaps(a, b))
        self.assertFalse(layout.overlaps(a, fam("c", (3, 4, 5))))


class TestConstructionIsTheOnlyDoor(unittest.TestCase):
    def test_raw_single_cell_block_rejected(self):
        with self.assertRaises(ValueError):
            layout.Block(group=TP8, replica_cells=(tuple(range(8)),))

    def test_raw_noncovering_cells_rejected(self):
        with self.assertRaises(ValueError):
            layout.Block(group=TP8, replica_cells=((0, 1), (2, 3)))

    def test_pad_to_must_be_a_positive_integer(self):
        for pad_to in (0, True, 1.5):
            with self.subTest(pad_to=pad_to), self.assertRaises(ValueError):
                layout.block(TP8, SINGLETONS8, pad_to=pad_to)

    def test_raw_single_segment_rejected(self):
        with self.assertRaises(ValueError):
            layout.Segmented(segments=((0, 8, layout.sharded(TP8)),))

    def test_raw_negative_or_gapped_segments_rejected(self):
        b = layout.sharded(TP8)
        with self.assertRaises(ValueError):
            layout.Segmented(segments=((0, 8, b), (9, 8, b)))
        with self.assertRaises(ValueError):
            layout.Segmented(segments=((0, 8, b), (8, 0, b)))

    def test_segment_extents_are_integral_and_valid_for_each_child(self):
        rep = layout.replicated(TP8.ranks())
        for segments in (
            ((0, True, rep), (1, True, rep)),
            ((0, 1.5, rep), (1.5, 1.5, rep)),
        ):
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                layout.segmented(segments)

        block = layout.sharded(TP8)
        for segments in (
            ((0, 3, block),),
            ((0, 3, block), (3, 3, block)),
        ):
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                layout.segmented(segments)

    def test_raw_product_rejects_padded_replicated_and_nonorthogonal(self):
        padded = layout.block(TP4, tuple((i,) for i in range(4)), pad_to=2)
        with self.assertRaises(ValueError):
            layout.Product(cuts=(padded, layout.sharded(DP2)))
        with self.assertRaises(ValueError):
            layout.Product(cuts=(layout.replicated(TP8.ranks()), layout.sharded(DP2)))
        with self.assertRaises(ValueError):
            layout.Product(cuts=(layout.sharded(TP8), layout.block(TP8, PAIRS)))


class TestCanonicalForm(unittest.TestCase):
    def test_single_cell_block_is_replicated(self):
        self.assertEqual(layout.block(TP8, (tuple(range(8)),)),
                 layout.replicated(TP8.ranks()))

    def test_padded_single_cell_block_stays(self):
        self.assertIsInstance(
            layout.block(TP8, (tuple(range(8)),), pad_to=4), layout.Block)

    def test_single_segment_reduces_to_child(self):
        child = layout.sharded(TP8)
        self.assertEqual(layout.segmented([(0, 64, child)]), child)

    def test_single_cut_product_reduces(self):
        cut = layout.sharded(TP4)
        self.assertEqual(layout.product([cut]), cut)

    def test_isomorphic_segments_do_not_reduce(self):
        b = layout.sharded(TP8)
        seg = layout.segmented([(0, 32, b), (32, 32, b)])
        self.assertIsInstance(seg, layout.Segmented)
        self.assertNotEqual(seg, b)

    def test_product_order_is_semantic(self):
        a, b = layout.sharded(TP4), layout.sharded(DP2)
        self.assertNotEqual(layout.product([a, b]), layout.product([b, a]))


class TestExtentsAndOwners(unittest.TestCase):
    def test_every_division_is_exact_or_loud(self):
        with self.assertRaises(ValueError):
            layout.physical_local_extent(layout.sharded(TP8), 12)
        p = layout.product([layout.sharded(TP4), layout.sharded(DP2)])
        with self.assertRaises(ValueError):
            layout.logical_local_extent(p, 6)      # 6 // 2 // 4 is not exact
        with self.assertRaises(ValueError):
            layout.owner_ranks(p, 2, 6, TP8.ranks())

    def test_unknown_is_rejected_at_every_consumer(self):
        u = layout.Unknown("test")
        for fn in (lambda: layout.logical_local_extent(u, 8),
                   lambda: layout.physical_local_extent(u, 8),
                   lambda: layout.owner_ranks(u, 0, 8, TP8.ranks()),
                   lambda: layout.local_index(u, 0, 0, 8, TP8.ranks()),
                   lambda: layout.slice_axis(u, 0, 8, 8)):
            with self.assertRaises(ValueError):
                fn()

    #: Extents of every shape the non-negativity gate meets, each with what
    #: the walk must do with it: hold it (the relation is `true`), write it
    #: down as a condition of the walk, or refuse it here.
    NON_NEGATIVITY = (
        (128, "held"), (0, "held"), (sp.Integer(7168), "held"),
        (sp.S.One, "held"), (sp.oo, "held"),
        (POSITIVE_COUNT, "held"),                       # declared positive
        (sp.floor(POSITIVE_COUNT / 4), "held"),
        (sp.Max(POSITIVE_COUNT, 1) - 1, "held"),
        (POSITIVE_COUNT - 1, "held"),                   # a count less one
        (WHOLE_COUNT + POSITIVE_COUNT - 1, "recorded"),  # the decode `s_kv`
        (POSITIVE_COUNT - 4096, "recorded"),
        (sp.Symbol("undeclared"), "recorded"),
        (-5, "refused"), (sp.Rational(-1, 2), "refused"),
        (-POSITIVE_COUNT, "refused"), (-sp.oo, "refused"),
    )

    def test_non_negativity_is_held_recorded_or_refused(self):
        """A count is non-negative by its declaration, or by a condition the
        walk writes down, or it is refused — and which of the three never
        depends on whether the extent is a number."""
        for extent, verdict in self.NON_NEGATIVITY:
            with self.subTest(extent=str(extent), verdict=verdict), \
                    dim.assuming(domain=dim.Domain(
                        [WHOLE_COUNT, POSITIVE_COUNT])) as ledger:
                if verdict == "refused":
                    with self.assertRaises(ValueError):
                        layout.physical_local_extent(
                            layout.Replicated(TP8.ranks()), extent)
                    continue
                layout.physical_local_extent(
                    layout.Replicated(TP8.ranks()), extent)
                written = [str(a.condition) for a in ledger.held]
                self.assertEqual(
                    written,
                    [] if verdict == "held" else [f"{extent} >= 0"])

    def test_the_gate_agrees_with_the_relation_it_stands_for(self):
        """The gate holds an extent without writing anything down exactly
        where `Ge(extent, 0)` is already `true`.

        It reads the assumption that relation consults rather than building
        the relation (`layout._assume_non_negative`), so the two must answer
        alike on every extent — a SymPy whose `is_ge` stopped agreeing with
        its own `is_extended_nonnegative` would be a gate that quietly
        stopped gating.
        """
        for extent, _verdict in self.NON_NEGATIVITY:
            with self.subTest(extent=str(extent)):
                settled = sp.sympify(
                    extent, strict=True).is_extended_nonnegative
                self.assertEqual(bool(settled),
                                 sp.Ge(extent, 0) is sp.true)

    def test_owner_bounds_are_checked(self):
        b = layout.sharded(TP8)
        for bad in (-1, 8):
            with self.assertRaises((IndexError, ValueError)):
                layout.owner_ranks(b, bad, 8, TP8.ranks())
        with self.assertRaises(ValueError):
            layout.owner_ranks(b, 0, 0, TP8.ranks())

    def test_padded_block_extents_and_dedup_conservation(self):
        b = layout.block(TP4, tuple((i,) for i in range(4)), pad_to=1)
        self.assertEqual(layout.physical_local_extent(b, 10), 3)
        self.assertEqual(b.valid_counts_per_shard(10), (3, 3, 3, 1))
        self.assertEqual(sum(b.valid_counts_per_shard(10)), 10)
        with self.assertRaises(ValueError):
            layout.logical_local_extent(b, 10)
        # Replicated replica_cells do not double the conservation sum: per PART.
        rb = layout.block(TP8, PAIRS, pad_to=1)
        self.assertEqual(sum(rb.valid_counts_per_shard(10)), 10)

    def test_replicated_cell_owners(self):
        b = layout.block(TP8, PAIRS)
        self.assertEqual(layout.owner_ranks(b, 0, 8, TP8.ranks()), frozenset({0, 1}))
        self.assertEqual(layout.owner_ranks(b, 7, 8, TP8.ranks()), frozenset({6, 7}))

    def test_every_index_has_owners(self):
        for axis, extent in [
            (layout.sharded(TP8), 16),
            (layout.block(TP8, PAIRS), 8),
            (layout.product([layout.sharded(TP4), layout.sharded(DP2)]), 8),
            (layout.segmented([(0, 8, layout.sharded(TP8)),
                           (8, 4, layout.block(TP8, PAIRS))]), 12),
        ]:
            for i in range(extent):
                self.assertTrue(layout.owner_ranks(axis, i, extent, TP8.ranks()))

    def test_product_owner_is_the_intersection(self):
        p = layout.product([layout.sharded(TP4), layout.sharded(DP2)])
        seen = set()
        for i in range(8):
            o = layout.owner_ranks(p, i, 8, TP8.ranks())
            self.assertEqual(len(o), 1)
            seen |= o
        self.assertEqual(seen, TP8.ranks())

    def test_segmented_extents_sum(self):
        seg = layout.segmented([(0, 32, layout.sharded(TP8)),
                            (32, 16, layout.block(TP8, PAIRS))])
        self.assertEqual(layout.logical_local_extent(seg, 48), 32 // 8 + 16 // 4)


class TestLocalOffsetOracle(unittest.TestCase):
    """The appendix's I(r, i): defined iff owner, injective per rank,
    bounded by physical local — checked exhaustively on small layouts."""

    LAYOUTS = None

    def _layouts(self):
        return [
            (layout.replicated(TP8.ranks()), 6),
            (layout.sharded(TP8), 16),
            (layout.block(TP8, PAIRS), 8),
            (layout.product([layout.sharded(TP4), layout.sharded(DP2)]), 16),
            (layout.segmented([(0, 8, layout.sharded(TP8)),
                           (8, 4, layout.block(TP8, PAIRS))]), 12),
        ]

    def test_defined_iff_owner_injective_and_bounded(self):
        for axis, extent in self._layouts():
            phys = layout.physical_local_extent(axis, extent)
            for r in sorted(TP8.ranks()):
                seen = {}
                for i in range(extent):
                    off = layout.local_index(axis, r, i, extent, TP8.ranks())
                    owned = r in layout.owner_ranks(axis, i, extent, TP8.ranks())
                    self.assertEqual(off is not None, owned)
                    if off is not None:
                        self.assertTrue(0 <= off < phys)
                        self.assertNotIn(off, seen)   # injective per rank
                        seen[off] = i


class TestSliceRules(unittest.TestCase):
    def test_replicated_slices_freely(self):
        rep = layout.replicated(TP8.ranks())
        self.assertIs(layout.slice_axis(rep, 2, 5, 8), rep)

    def test_block_proper_subinterval_rejected(self):
        b = layout.sharded(TP8)
        with self.assertRaises(ValueError):
            layout.slice_axis(b, 0, 8, 16)
        self.assertEqual(layout.slice_axis(b, 0, 16, 16), b)

    def test_segmented_slices_per_segment(self):
        q, kv = layout.sharded(TP8), layout.block(TP8, PAIRS)
        fused = layout.segmented([(0, 32, q), (32, 8, kv), (40, 8, kv)])
        self.assertEqual(layout.slice_axis(fused, 0, 32, 48), q)    # Q slice
        self.assertEqual(layout.slice_axis(fused, 32, 40, 48), kv)  # K slice
        both = layout.slice_axis(fused, 32, 48, 48)                 # K+V
        self.assertIsInstance(both, layout.Segmented)
        with self.assertRaises(ValueError):
            layout.slice_axis(fused, 0, 30, 48)   # cuts into Q's block


class TestUniformFiber(unittest.TestCase):
    def test_multi_replica_family_is_per_component(self):
        # One legal cut over a two-group family must pass: counting over
        # the flat rank union would merge the replicas into a spurious
        # fiber of two.
        self.assertTrue(layout.is_uniform_surjection([layout.sharded(TP8x2)]))
        layout.tensor_layout([layout.sharded(TP8x2)])

    def test_gqa_replication_is_surjective_fiber_two(self):
        self.assertTrue(layout.is_uniform_surjection([layout.block(TP8, PAIRS)]))
        with self.assertRaises(ValueError):
            layout.tensor_layout([layout.sharded(TP8), layout.block(TP8, PAIRS)])

    def test_fused_qkv_segments_are_alternatives_not_joint(self):
        fused = layout.segmented([
            (0, 32, layout.sharded(TP8)),
            (32, 8, layout.block(TP8, PAIRS)),
            (40, 8, layout.block(TP8, PAIRS)),
        ])
        layout.tensor_layout([fused])               # must not raise

    def test_cross_axis_same_group_diagonal_rejected(self):
        with self.assertRaises(ValueError):
            layout.tensor_layout([layout.sharded(TP8), layout.sharded(TP8)])

    def test_cross_axis_orthogonal_accepted(self):
        layout.tensor_layout([layout.sharded(TP4), layout.sharded(DP2)])

    def test_three_cut_joint_fails_where_every_pair_passes(self):
        # xor construction on eight ranks (one shared group, replica_cells by
        # x1^x2, x2^x3, x1^x3): every PAIR of cuts tiles 2x2 with equal
        # fiber 2 — a legal two-axis layout — but the third label is
        # determined by the other two, so the JOINT image is 4 where 8 is
        # demanded. Only the tensor-level joint check can see this.
        g8 = fam("g8", range(8))

        def xor_cut(bit_a, bit_b):
            part0 = tuple(r for r in range(8)
                          if ((r >> bit_a) ^ (r >> bit_b)) & 1 == 0)
            part1 = tuple(r for r in range(8) if r not in part0)
            return layout.block(g8, (part0, part1))

        cuts = [xor_cut(2, 1), xor_cut(1, 0), xor_cut(2, 0)]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertTrue(layout.is_uniform_surjection([cuts[i], cuts[j]]))
        self.assertFalse(layout.is_uniform_surjection(cuts))

    def test_padded_single_cell_never_bridges_nor_changes_verdicts(self):
        pad_only = layout.block(TP8, (tuple(range(8)),), pad_to=4)
        tl = layout.tensor_layout([pad_only, layout.sharded(TP8)])    # passes
        # Locked structurally, not just by verdict: the padding-only Block
        # contributes NO cut to any alternative — its group can never
        # bridge joint_components.
        self.assertEqual(tl.cut_alternatives()[0], [[]])
        with self.assertRaises(ValueError):                    # still fails
            layout.tensor_layout([pad_only, layout.sharded(TP8), layout.sharded(TP8)])

    def test_raw_tensor_layout_is_no_bypass(self):
        with self.assertRaises(ValueError):
            layout.TensorLayout(axes=(layout.sharded(TP8), layout.sharded(TP8)))
        with self.assertRaises(ValueError):
            layout.TensorLayout(axes=(layout.sharded(TP8),), buffer_order=(0, 0))
        with self.assertRaises(ValueError):
            layout.TensorLayout(axes=(layout.sharded(TP8),), buffer_order=(1,))

    def test_segmented_extent_must_match_its_tiling(self):
        seg = layout.segmented([(0, 8, layout.sharded(TP8)),
                            (8, 4, layout.block(TP8, PAIRS))])
        for fn in (lambda: layout.logical_local_extent(seg, 8),
                   lambda: layout.physical_local_extent(seg, 16),
                   lambda: layout.owner_ranks(seg, 0, 8, TP8.ranks()),
                   lambda: layout.local_index(seg, 0, 0, 8, TP8.ranks()),
                   lambda: layout.slice_axis(seg, 0, 8, 8)):
            with self.assertRaises(ValueError):
                fn()

    def test_mismatched_scopes_fail(self):
        self.assertFalse(layout.is_uniform_surjection(
            [layout.sharded(TP8), layout.sharded(TP8x2)]))


class TestCrossReplicaFamily(unittest.TestCase):
    def test_m_by_r_groups_of_size_shards_covering_all(self):
        b = layout.block(TP8x2, PAIRS)
        derived = layout.cross_replica_family(b.group, b.replica_cells)
        self.assertEqual(len(derived.groups), 2 * 2)          # m*r
        self.assertEqual(derived.group_size, 4)               # num_shards
        self.assertEqual(derived.ranks(), TP8x2.ranks())      # disjoint cover
        self.assertIn((0, 2, 4, 6), derived.groups)
        self.assertIn((1, 3, 5, 7), derived.groups)


class TestValueAndJoin(unittest.TestCase):
    def test_unknown_reducer_and_foreign_schema_are_loud(self):
        with self.assertRaises(ValueError):
            layout.Partial(reducer="topk", group=TP8)
        with self.assertRaises(ValueError):
            layout.Partial(reducer=layout.SUM, group=TP8, schema=("wrong",))

    def test_online_attention_schema_defaults(self):
        p = layout.Partial(reducer=layout.ONLINE_ATTENTION, group=TP8)
        self.assertEqual(p.schema,
                         ("running_max", "normalizer", "weighted_value"))

    def test_provenance_is_structured_but_not_partial_identity(self):
        attributed = layout.Partial(
            reducer=layout.SUM, group=TP8).with_provenance(
                7, input=3, weight=5)
        plain = layout.Partial(reducer=layout.SUM, group=TP8)
        self.assertEqual(attributed, plain)
        self.assertEqual(attributed.provenance.producer, 7)
        self.assertEqual(attributed.provenance.operand("input"), 3)
        with self.assertRaisesRegex(ValueError, "no operand 'missing'"):
            attributed.provenance.operand("missing")
        with self.assertRaisesRegex(ValueError, "duplicate.*role"):
            layout.ValueProvenance(7, (("input", 3), ("input", 4)))

    def test_orthogonal_partials_join(self):
        joined = layout.join_orthogonal_partials(
            layout.Partial(reducer=layout.SUM, group=TP4),
            layout.Partial(reducer=layout.SUM, group=DP2))
        self.assertEqual(joined.group.ranks(), TP8.ranks())
        self.assertEqual(len(joined.group.groups), 1)

    def test_join_is_composable_with_a_third_family(self):
        # codex's counterexample: rank-sorted join groups lose the
        # (coord_A, coord_B) product structure and falsely fail
        # orthogonality against a third independent family. Coordinate-
        # ordered join keeps the composition closed.
        A = fam("A", (0, 1), (2, 3), (4, 5), (6, 7))
        B = fam("B", (2, 0), (1, 3), (4, 6), (5, 7))
        C = fam("C", (6, 0), (1, 7), (2, 4), (3, 5))
        self.assertTrue(layout.orthogonal(A, B))
        self.assertTrue(layout.orthogonal(A, C))
        self.assertTrue(layout.orthogonal(B, C))
        ab = layout.join_orthogonal_partials(
            layout.Partial(reducer=layout.SUM, group=A),
            layout.Partial(reducer=layout.SUM, group=B))
        abc = layout.join_orthogonal_partials(
            ab, layout.Partial(reducer=layout.SUM, group=C))
        self.assertEqual(abc.group.ranks(), frozenset(range(8)))
        self.assertEqual(len(abc.group.groups), 1)

    def test_non_orthogonal_partials_refuse(self):
        with self.assertRaises(ValueError):
            layout.join_orthogonal_partials(layout.Partial(reducer=layout.SUM, group=TP8),
                             layout.Partial(reducer=layout.SUM, group=TP8))
        with self.assertRaises(ValueError):
            layout.join_orthogonal_partials(layout.Partial(reducer=layout.SUM, group=TP4),
                             layout.Partial(reducer=layout.MAX, group=DP2))


class TestExpectedRoutedLoad(unittest.TestCase):
    def _wl(self, **kw):
        # The fan-out places the bank's own experts (one bank, one count),
        # so it follows `num_experts` unless a case states its own.
        experts = max(int(kw.get("num_experts", 64)), 1)
        args = dict(group=TP8, expert_shard_axis=layout.sharded(TP8),
                    routed_rows_in_cell=Fraction(4096), num_experts=64,
                    load_assumption_id="balanced",
                    fanout=layout.RouteFanout(picks=min(8, experts),
                                              targets=experts,
                                              placed=experts))
        args.update(kw)
        return layout.ExpectedRoutedLoad(**args)

    def test_exact_rational_mean_and_conservation(self):
        wl = self._wl()
        self.assertEqual(wl.expected_rows_per_expert, Fraction(4096, 64))
        self.assertTrue(wl.is_measure_preserving())
        odd = self._wl(routed_rows_in_cell=Fraction(1000), num_experts=48)
        self.assertEqual(odd.expected_rows_per_expert, Fraction(1000, 48))  # no ceil
        self.assertTrue(odd.is_measure_preserving())

    def test_preconditions_are_loud(self):
        with self.assertRaises(ValueError):
            self._wl(expert_shard_axis=layout.block(TP8, PAIRS))       # replicated
        with self.assertRaises(ValueError):
            self._wl(expert_shard_axis=layout.sharded(TP4))            # wrong family
        with self.assertRaises(ValueError):
            self._wl(num_experts=0)
        with self.assertRaises(ValueError):
            self._wl(num_experts=12)                        # 12 % 8 != 0
        with self.assertRaises(ValueError):
            self._wl(routed_rows_in_cell=Fraction(-1))
        with self.assertRaises(ValueError):
            self._wl(expert_load_factor=Fraction(1, 2))     # below the mean
        with self.assertRaises(ValueError):
            self._wl(expert_load_factor=65)                 # > all the rows
        with self.assertRaises(ValueError):
            # The fan-out places another bank's experts: one bank, one count.
            self._wl(fanout=layout.RouteFanout(picks=8, targets=32, placed=32))
        with self.assertRaises(ValueError):
            self._wl(fanout=None)                           # required

    def test_busiest_rank_dilutes_the_hot_expert_over_its_rank(self):
        """rho is per expert; a rank of k experts holding the hot one carries
        `(rho + (k-1)(E-rho)/(E-1)) / k` of the mean rank — the rows are
        conserved, so the others hold less than the mean."""
        wl = self._wl(expert_load_factor=10)               # E=64, 8 per rank
        self.assertEqual(wl.busiest_rank_load_factor,
                         (10 + 7 * Fraction(54, 63)) / 8)
        self.assertEqual(wl.routed_row_capacity_per_rank,
                         wl.expected_rows_on_rank(8)
                         * wl.busiest_rank_load_factor)
        # Balanced: no rank is busier.
        self.assertEqual(self._wl().busiest_rank_load_factor, 1)
        # Every rank holds every expert: a hot expert makes no rank busier.
        whole = self._wl(group=None, expert_shard_axis=None,
                         expert_load_factor=10)
        self.assertEqual(whole.busiest_rank_load_factor, 1)
        # Every row on one expert: its rank holds all of them, EP x the mean.
        self.assertEqual(
            self._wl(expert_load_factor=64).busiest_rank_load_factor, 8)


class TestReductionHomomorphism(unittest.TestCase):
    def setUp(self):
        self.p = layout.Partial(reducer=layout.SUM, group=TP8)
        self.yes = lambda s, c: True

    def test_law_table_speaks_the_real_op_vocabulary(self):
        from backend.arch.ops import OP_REGISTRY
        for name in layout.HOMOMORPHISM_REGISTRY:
            self.assertIn(name, OP_REGISTRY,
                          f"law keyed to unknown op {name!r}")

    def test_linear_requires_bias_absent(self):
        # Real arity: the weight is an implicit parameter, not a data
        # input — its invariance arrives as a parameter condition.
        law = layout.HOMOMORPHISM_REGISTRY["linear"]
        inputs = (("in", self.p),)
        self.assertTrue(layout.may_defer_reduction(law, inputs, self.yes,
                                    frozenset({"bias_absent"})))
        self.assertFalse(layout.may_defer_reduction(law, inputs, self.yes, frozenset()))

    def test_linear_needs_invariant_weight(self):
        law = layout.HOMOMORPHISM_REGISTRY["linear"]
        inputs = (("in", self.p),)
        self.assertFalse(layout.may_defer_reduction(law, inputs, lambda s, c: False,
                                     frozenset({"bias_absent"})))

    def test_combine_needs_invariant_gate(self):
        # Real arity: (expert_out, routing) — the routing operand IS the
        # gate stream, matching the two-input seam contract.
        law = layout.HOMOMORPHISM_REGISTRY["combine"]
        inputs = (("expert_out", self.p), ("routing", layout.COMPLETE))
        self.assertTrue(layout.may_defer_reduction(law, inputs, self.yes))
        routing_cut = lambda s, c: s != "routing"
        self.assertFalse(layout.may_defer_reduction(law, inputs, routing_cut))

    def test_add_partial_plus_complete_rejected(self):
        law = layout.HOMOMORPHISM_REGISTRY["add"]
        self.assertFalse(layout.may_defer_reduction(
            law, (("a", self.p), ("b", layout.COMPLETE)), self.yes))

    def test_add_matching_partials_merge(self):
        law = layout.HOMOMORPHISM_REGISTRY["add"]
        self.assertTrue(layout.may_defer_reduction(
            law, (("a", self.p), ("b", self.p)), self.yes))

    def test_mixed_pending_states_never_merge(self):
        law = layout.HOMOMORPHISM_REGISTRY["add"]
        other = layout.Partial(reducer=layout.SUM, group=TP4)
        self.assertFalse(layout.may_defer_reduction(
            law, (("a", self.p), ("b", other)), self.yes))

    def test_mul_folds_one_pending_operand_by_an_invariant_factor(self):
        law = layout.HOMOMORPHISM_REGISTRY["mul"]
        # Commutative: the pending sum may sit at either operand.
        self.assertEqual(set(law.namings(2)),
                         {("x", "factor"), ("factor", "x")})
        self.assertTrue(layout.may_defer_reduction(
            law, (("x", self.p), ("factor", layout.COMPLETE)), self.yes))
        self.assertTrue(layout.may_defer_reduction(
            law, (("factor", layout.COMPLETE), ("x", self.p)), self.yes))

    def test_mul_of_two_pending_sums_finalizes(self):
        # (sum a)(sum b) != sum ab, under any naming.
        law = layout.HOMOMORPHISM_REGISTRY["mul"]
        for slots in law.namings(2):
            self.assertFalse(layout.may_defer_reduction(
                law, tuple((slot, self.p) for slot in slots), self.yes))

    def test_mul_by_a_cut_factor_finalizes(self):
        law = layout.HOMOMORPHISM_REGISTRY["mul"]
        factor_cut = lambda s, c: s != "factor"
        self.assertFalse(layout.may_defer_reduction(
            law, (("x", self.p), ("factor", layout.COMPLETE)), factor_cut))

    def test_a_non_commutative_law_names_its_slots_once(self):
        self.assertEqual(layout.HOMOMORPHISM_REGISTRY["combine"].namings(2),
                         (("expert_out", "routing"),))
        self.assertEqual(layout.HOMOMORPHISM_REGISTRY["linear"].namings(2), ())

    def test_concat_wildcard_requires_every_input_pending(self):
        law = layout.HOMOMORPHISM_REGISTRY["concat"]
        self.assertTrue(layout.may_defer_reduction(
            law, (("x0", self.p), ("x1", self.p)), self.yes))
        self.assertFalse(layout.may_defer_reduction(
            law, (("x0", self.p), ("x1", layout.COMPLETE)), self.yes))

    def test_partial_preserving_view_moves_any_merge_kind(self):
        law = layout.HOMOMORPHISM_REGISTRY["slice"]
        p_max = layout.Partial(reducer=layout.MAX, group=TP8)
        self.assertTrue(layout.may_defer_reduction(law, (("in", p_max),), self.yes))

    def test_undeclared_operand_blocks_passthrough(self):
        law = layout.HOMOMORPHISM_REGISTRY["linear"]
        inputs = (("in", self.p), ("mystery", layout.COMPLETE))
        self.assertFalse(layout.may_defer_reduction(law, inputs, self.yes,
                                     frozenset({"bias_absent"})))

    def test_external_weight_is_a_data_input(self):
        # linear_external_weight's weight arrives as the second DATA
        # input (hc_pre wires a computed tensor in), so it must be
        # present and its invariance is a coefficient condition.
        law = layout.HOMOMORPHISM_REGISTRY["linear_external_weight"]
        inputs = (("in", self.p), ("weight", layout.COMPLETE))
        self.assertTrue(layout.may_defer_reduction(law, inputs, self.yes,
                                               frozenset({"bias_absent"})))
        with self.assertRaises(ValueError):
            layout.may_defer_reduction(law, (("in", self.p),), self.yes,
                                   frozenset({"bias_absent"}))

    def test_registry_keys_equal_law_ops(self):
        for key, law in layout.HOMOMORPHISM_REGISTRY.items():
            self.assertEqual(key, law.op)

    def test_unknown_value_is_loud_in_deferral_predicate(self):
        law = layout.HOMOMORPHISM_REGISTRY["slice"]
        with self.assertRaises(ValueError):
            layout.may_defer_reduction(law, (("in", layout.UnknownValue("t")),), self.yes)
        # Caller bugs raise even when `required_node_facts` would reject: a broken
        # call must never masquerade as a mathematical False.
        linear = layout.HOMOMORPHISM_REGISTRY["linear"]
        with self.assertRaises(ValueError):
            layout.may_defer_reduction(linear, (("in", layout.UnknownValue("t")),),
                                   self.yes, frozenset())  # facts missing too
        combine = layout.HOMOMORPHISM_REGISTRY["combine"]
        with self.assertRaises(ValueError):
            layout.may_defer_reduction(combine, (("expert_out", self.p),),
                                   self.yes)   # routing operand omitted

    def test_partial_slot_picture_is_a_caller_bug(self):
        law = layout.HOMOMORPHISM_REGISTRY["combine"]
        with self.assertRaises(ValueError):
            layout.may_defer_reduction(law, (("expert_out", self.p),), self.yes)

    def test_no_law_for_nonlinearities(self):
        for op in ("silu", "relu", "rmsnorm", "softmax"):
            self.assertNotIn(op, layout.HOMOMORPHISM_REGISTRY)


if __name__ == "__main__":
    unittest.main()
