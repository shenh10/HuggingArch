"""The sweep engine (`backend/measure/auto_sweep.py`): the all-or-nothing
measured track and its coverage, the measured formula, substitution at a
point, a point's measure folded over a selection, and one walk per cell."""
from __future__ import annotations

from dataclasses import replace

import pytest

from backend.measure import auto_sweep as A
from backend.measure.lookup import Key


def _key(sig: str, shape: str = '{"m":1}') -> Key:
    return Key(sig, shape, '{"activation":"bf16"}', "H100-SXM", "decode")


class TestMeasuredTrack:
    def test_a_state_names_a_key_once_and_counts_every_row(self):
        """One missing key over three layers is one thing to measure and
        three rows not measured: the counts add up to the rows the coverage
        leaves out (the page once read "76 / 84 measured; missing 3 ·
        unplaced 2")."""
        track = A.MeasuredTrack.of(
            [(A.MEASURED, "a", "")] + [(A.MISSING, "k", "")] * 3
            + [(A.UNPLACED, "op", "")] * 2)
        cov = track.coverage()
        assert cov["states"] == {"missing": ["k"], "unplaced": ["op"]}
        assert cov["rows_by_state"] == {"missing": 3, "unplaced": 2}
        assert sum(cov["rows_by_state"].values()) == cov["total"] - cov["measured"]

    def test_a_total_only_when_every_row_is_measured(self):
        full = A.MeasuredTrack.of([(A.MEASURED, "a", ""), (A.MEASURED, "z", "")])
        assert full.coverage() == {"measured": 2, "total": 2}
        assert full.with_total(1.0).t_s == 1.0
        part = A.MeasuredTrack.of([(A.MEASURED, "a", ""), (A.MISSING, "k", "")])
        assert part.t_s is None
        assert part.coverage() == {"measured": 1, "total": 2, "states": {"missing": ["k"]},
                                   "rows_by_state": {"missing": 1}}
        with pytest.raises(ValueError, match="full coverage"):
            part.with_total(1.0)

    def test_no_rows_is_no_total(self):
        assert A.MeasuredTrack.of([]).t_s is None

    def test_each_state_is_listed_by_what_names_it(self):
        t = A.MeasuredTrack.of([
            (A.MISSING, "k1", ""), (A.AMBIGUOUS, "k2", ""),
            (A.UNPLACED, "sample", ""), (A.DECLINED, "k3", "u8 has no token"),
            (A.MISSING, "k1", "")])
        assert t.coverage()["states"] == {
            "missing": ["k1"], "ambiguous": ["k2"], "unplaced": ["sample"],
            "declined": [{"key": "k3", "reason": "u8 has no token"}]}
        assert (t.measured_rows, t.total_rows) == (0, 5)

    def test_rows_nothing_was_looked_up_for_are_in_no_state(self):
        t = A.MeasuredTrack.unread(3)
        assert t.coverage() == {"measured": 0, "total": 3}
        assert t.t_s is None

    def test_a_measured_total_says_how_much_of_it_nobody_checked(self):
        """A row with no oracle is still the best number there is, and a
        reader has to be able to tell.

        40,591 of 2,469,869 measured rows carry `cross_check=no_reference`:
        the fp32 oracle would not build for that cell, so the kernel's
        latency was recorded unverified. For the biggest attention shapes
        that is unavoidable — an O(S²) score matrix in fp32 does not fit
        beside the kernel on one card — so refusing those rows would drop
        the only measurement they have and fall back to the roofline, which
        is further from the truth. The per-row badge has carried the
        verdict all along; nothing added it up, so a total built on
        unchecked numbers read exactly like one built on checked ones.
        """
        track = A.MeasuredTrack.of(
            [(A.MEASURED, "a", ""), (A.MEASURED, "b", ""),
             (A.MISSING, "k", "")], unchecked=1)
        assert track.coverage() == {
            "measured": 2, "total": 3, "unchecked": 1,
            "states": {"missing": ["k"]}, "rows_by_state": {"missing": 1}}
        # Silent when every measured row was checked: the absence of the
        # key is the ordinary case, not a zero to read past.
        assert "unchecked" not in A.MeasuredTrack.of(
            [(A.MEASURED, "a", "")]).coverage()

    def test_more_unchecked_than_measured_is_refused(self):
        """A row that was not measured has no verdict to be unchecked."""
        with pytest.raises(ValueError, match="has no verdict"):
            A.MeasuredTrack.of([(A.MISSING, "k", "")], unchecked=1)

    def test_an_unknown_state_is_refused(self):
        with pytest.raises(ValueError, match="unknown coverage state"):
            A.MeasuredTrack.of([("guessed", "a", "")])


class TestSubstitution:
    def test_law_text_at_a_point(self):
        assert A.evaluate("2*B + Max(B, 3)", {"B": 5}) == 15
        assert A.evaluate("floor(B/2) + ceiling(B/2)", {"B": 7}) == 7

    def test_a_symbol_the_point_does_not_bind_raises(self):
        with pytest.raises(ValueError, match="does not bind"):
            A.evaluate("B*PEAK", {"B": 1})

    def test_a_cells_laws_compile_once_in_the_book_that_owns_them(self):
        """Compiled laws belong to the cell that wrote them
        (`row_roofline.LawBook`), not to the process: a book compiles a law
        once for every point of its cell, and another book starts empty."""
        from backend.inference.row_roofline import LawBook
        book = LawBook()
        law = book.read("3*B + 1")
        for b in range(1, 20):
            assert book.evaluate(law, {"B": b}) == 3 * b + 1
        assert len(book._compiled) == 1
        assert LawBook()._compiled == {}


class TestMeasuredFormula:
    def test_the_formula_binds_launches_and_times_per_term(self):
        f = A.MeasuredFormula("T_0*n_0 + 2*T_1*n_1", (False, False, True))
        assert f.at([3, 1, 0], [0.5, 0.25, None]) == pytest.approx(3 * 0.5 + 2 * 0.25)

    def test_all_zero_cost_is_zero(self):
        f = A.MeasuredFormula("0", (True,))
        assert f.at([0], [None]) == 0.0

    def test_exact_until_a_float_enters(self):
        from fractions import Fraction
        assert A.evaluate("B/3 + B/6", {"B": 2}) == Fraction(1)
        assert A.evaluate("Piecewise((B, B > 2), (0, True))", {"B": 3}) == 3


class TestPointMeasure:
    def _measure(self) -> A.PointMeasure:
        keys = (_key("gemm"), None, _key("rmsnorm"), None, None)
        return A.PointMeasure(
            keys=keys, launches=(28.0, 1.0, 57.0, 1.0, 1.0),
            unkeyed=(None, A.MEASURED, None, A.UNPLACED, A.DECLINED),
            labels=("gemm", "sample", "rmsnorm", "custom_op", "embed"),
            reasons=("", "", "", "", "u8 has no corpus token"),
            formula=A.MeasuredFormula("T_0*n_0 + T_2*n_2 + T_3*n_3 + T_4*n_4",
                                      (False, True, False, False, False)),
            context={"gpu": "H100-SXM"})

    def test_rows_without_a_key_and_zero_cost_rows_need_no_lookup(self):
        m = self._measure()
        assert m.key_set() == [_key("gemm"), _key("rmsnorm")]

    def test_fold_reports_every_state_and_no_total(self):
        m = self._measure()
        seen = []

        def select(key, ctx, label, read):
            seen.append((key.signature, ctx["gpu"]))
            return ((2e-6, A.MEASURED, {"by": label}) if key.signature == "gemm"
                    else (None, A.AMBIGUOUS, None))
        folded = m.fold(select)
        track, per_row = folded.track, folded.rows
        assert track.t_s is None
        assert track.coverage() == {"measured": 2, "total": 5, "states": {
            "ambiguous": [A.key_label(_key("rmsnorm"))], "unplaced": ["custom_op"],
            "declined": [{"key": "embed", "reason": "u8 has no corpus token"}]},
            "rows_by_state": {"ambiguous": 1, "unplaced": 1, "declined": 1}}
        assert [t for _s, t, _l, _d in per_row] == [pytest.approx(56e-6), 0.0, None, None, None]
        assert [s for s, *_ in per_row] == [A.MEASURED, A.MEASURED, A.AMBIGUOUS, A.UNPLACED,
                                           A.DECLINED]
        assert per_row[0][3] == {"by": "gemm"}
        assert seen == [("gemm", "H100-SXM"), ("rmsnorm", "H100-SXM")]

    def test_a_complete_point_has_the_formulas_total(self):
        keys = (_key("gemm"), None)
        m = A.PointMeasure(keys, (28.0, 1.0), (None, A.MEASURED), ("gemm", "sample"),
                           ("", ""), A.MeasuredFormula("2*T_0*n_0", (False, True)))
        folded = m.fold(lambda k, c, label, read: (1e-6, A.MEASURED, None))
        assert folded.track.t_s == pytest.approx(2 * 28e-6)
        assert [t for _s, t, _l, _d in folded.rows] == [pytest.approx(28e-6), 0.0]

    def test_a_part_is_the_formulas_share_and_only_when_all_of_it_is_measured(self):
        # A step of one main row and a draft row run in 3 forwards: the
        # draft's share is the formula's own term, not a re-added sum.
        keys = (_key("main"), _key("draft"))
        m = A.PointMeasure(keys, (2.0, 1.0), (None, None), ("main", "draft"), ("", ""),
                           A.MeasuredFormula("T_0*n_0 + 3*T_1*n_1", (False, False)))
        folded = m.fold(lambda k, c, label, read: (1e-6, A.MEASURED, None))
        assert folded.subtotal([1]) == pytest.approx(3e-6)
        assert folded.track.t_s == pytest.approx(5e-6)
        missing = m.fold(lambda k, c, label, read: (
            (1e-6, A.MEASURED, None) if label == "draft" else (None, A.MISSING, None)))
        assert missing.subtotal([1]) == pytest.approx(3e-6)
        assert missing.subtotal([0, 1]) is None and missing.track.t_s is None

    def test_a_selection_must_say_what_it_found(self):
        m = self._measure()
        with pytest.raises(ValueError, match="selection"):
            m.fold(lambda k, c, label, read: (None, A.MEASURED, None))

    def test_plain_data_round_trip(self):
        m = self._measure()
        assert A.PointMeasure.from_json(m.to_json()) == m

    def test_inconsistent_rows_are_refused(self):
        with pytest.raises(ValueError, match="zero-cost"):
            A.PointMeasure((None,), (1.0,), (A.UNPLACED,), ("x",), ("",),
                           A.MeasuredFormula("0", (True,)))
        with pytest.raises(ValueError, match="a key, or a state"):
            A.PointMeasure((_key("gemm"),), (1.0,), (A.UNPLACED,), ("x",), ("",),
                           A.MeasuredFormula("T_0*n_0", (False,)))


def test_sweep_keys_are_the_union_in_order():
    a = A.PointMeasure((_key("a"), _key("b")), (1.0, 1.0), (None, None), ("a", "b"),
                       ("", ""), A.MeasuredFormula("T_0*n_0 + T_1*n_1", (False, False)))
    b = A.PointMeasure((_key("b"), _key("c")), (1.0, 1.0), (None, None), ("b", "c"),
                       ("", ""), A.MeasuredFormula("T_0*n_0 + T_1*n_1", (False, False)))
    assert A.sweep_keys([a, b]) == [_key("a"), _key("b"), _key("c")]


def _double(cell: A.Cell) -> list:
    return [cell.walk * p for p in cell.points]


def test_one_walk_per_cell_in_order():
    cells = [A.Cell(2, (1, 2)), A.Cell(3, (1,))]
    calls = []

    def mapper(fn, items):
        calls.append(len(items))
        return [fn(c) for c in items]
    assert A.walk_cells(_double, cells, mapper=mapper) == [[2, 4], [3]]
    assert calls == [2]


def test_the_partial_swaps_each_measured_term_s_sol_for_its_measurement():
    """`Folded.partial`: the SOL total with each measured term's weighted
    SOL swapped for its measured share; a measure that states no SOL for
    a measured term (a pool point) leaves the blend to its reader."""
    from backend.measure.auto_sweep import Folded, MeasuredFormula, PointMeasure
    from backend.measure.lookup import Key
    key = Key("gemm", "{}", "{}", "H100", "decode")
    measure = PointMeasure(
        (key, key), (1.0, 2.0), (None, None), ("a", "b"), ("", ""),
        MeasuredFormula("n_0*T_0 + 3*n_1*T_1", (False, False)),
        sol=(0.010, 0.004))
    folded = Folded(measure, (("measured", 0.012, "a", None),
                              ("missing", None, "b", None)), (0.012, None))
    t_sol = 0.010 + 3 * 0.004
    got = folded.partial(t_sol)
    assert got["t_s"] == pytest.approx(0.012 + 3 * 0.004)
    assert got["measured_part_s"] == pytest.approx(0.012)
    assert folded.measured_terms == (0,)
    pool = Folded(replace(measure, sol=()), folded.rows, folded.times)
    assert pool.partial(t_sol) is None
    assert pool.measured_part_s == pytest.approx(0.012)
