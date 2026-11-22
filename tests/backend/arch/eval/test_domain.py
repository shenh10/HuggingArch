"""The workload domain a walk decides its conditions over (`dim.Domain`):
the counts its frame declares, and nothing else. A condition the domain
decides is that answer; one it does not is kept for the point; a name the
domain does not hold is anything, whatever its SymPy assumptions say."""
from __future__ import annotations

from fractions import Fraction

import sympy as sp

import pytest

from backend.arch.eval.dim import (Domain, _under, assuming, ratio_symbol,
                                   where)
from backend.arch.eval.transforms import Workload
from backend.arch.law_text import Cases

SYMBOLS = Workload.symbols()
B, S_IN, S_OUT = SYMBOLS["batch"], SYMBOLS["input_tokens"], SYMBOLS["output_tokens"]
DOMAIN = Domain(SYMBOLS.values())


def test_the_domain_is_the_workloads_declaration():
    """The batch and the verify width at least one; the prompt, the output
    and the cached prefix zero or more; the routing skew, a ratio, at
    least one."""
    assert DOMAIN.least == {B: 1, SYMBOLS["s_verify"]: 1, S_IN: 0, S_OUT: 0,
                            SYMBOLS["s_prefix_cached"]: 0,
                            SYMBOLS["ep_load_factor"]: 1,
                            SYMBOLS["dcp_selected_load_factor"]: 1,
                            SYMBOLS["dcp_failed_owners"]: 0}


def test_a_ratio_input_holds_from_its_floor():
    """`rho` is no whole count: its floor is its declaration."""
    rho = SYMBOLS["ep_load_factor"]
    assert DOMAIN.settled(sp.Ge(rho, 1)) is sp.true
    assert DOMAIN.settled(sp.Lt(rho, 1)) is sp.false
    assert DOMAIN.refusal(rho, sp.Rational(1, 2)) is not None
    assert DOMAIN.refusal(rho, sp.Rational(3, 2)) is None


def test_what_holds_everywhere_is_true():
    assert DOMAIN.settled(sp.Lt(0, B)) is sp.true
    assert DOMAIN.settled(sp.Ge(B + S_IN, 1)) is sp.true
    assert DOMAIN.settled(sp.Ne(B, 0)) is sp.true


def test_what_holds_nowhere_is_false():
    assert DOMAIN.settled(sp.Lt(B, 1)) is sp.false
    assert DOMAIN.settled(sp.Eq(B * (S_IN + 1), 0)) is sp.false


def test_what_the_point_decides_is_kept():
    """`s_input < 1` holds at an empty prompt and not at a long one."""
    truth = sp.Lt(S_IN, 1)
    assert DOMAIN.settled(truth) == truth
    assert DOMAIN.bounds(S_IN - S_OUT) == (-sp.oo, sp.oo)


def test_cases_against_a_number_answer_through_their_operands():
    """Both operands below the number: true. One above, one below: the
    case's own condition (or its negation)."""
    condition = sp.Lt(S_IN, 1)
    cases = Cases(condition, sp.Integer(0), B + 4)
    assert DOMAIN.settled(sp.Lt(cases, B + 10)) == sp.Lt(cases, B + 10)
    assert DOMAIN.settled(sp.Ge(cases, 0)) is sp.true
    assert DOMAIN.settled(sp.Lt(cases, 1)) == condition
    assert DOMAIN.settled(sp.Ge(cases, 1)) == DOMAIN.settled(sp.Not(condition))
    assert DOMAIN.bounds(cases) == (0, sp.oo)


def test_floor_and_ceiling_bound_their_argument_s_bounds():
    assert DOMAIN.bounds(sp.floor(S_IN / 2)) == (0, sp.oo)
    assert DOMAIN.bounds(sp.ceiling(B / 2)) == (1, sp.oo)
    assert DOMAIN.settled(sp.Gt(sp.ceiling(B / 2), 0)) is sp.true
    assert DOMAIN.settled(sp.Gt(sp.floor(S_IN / 2), 0)) == sp.Gt(sp.floor(S_IN / 2), 0)


def test_a_name_outside_the_domain_is_anything():
    """A symbol declared a positive whole number but not one of the walk's
    counts decides nothing: the domain is what the frame declares, not what
    a symbol says of itself."""
    other = sp.Symbol("x", integer=True, positive=True)
    assert DOMAIN.bounds(other) == (-sp.oo, sp.oo)
    assert DOMAIN.settled(sp.Gt(other, 0)) == sp.Gt(other, 0)
    assert Domain().settled(sp.Gt(B, 0)) == sp.Gt(B, 0)


def test_under_puts_a_known_condition_in_and_leaves_the_rest():
    condition = sp.Lt(S_IN, 1)
    assert _under(Cases(condition, S_OUT, B), condition, sp.true) == S_OUT
    assert _under(Cases(condition, S_OUT, B), condition, sp.false) == B
    unrelated = B + S_OUT
    assert _under(unrelated, condition, sp.true) is unrelated


def test_a_walk_s_condition_is_decided_over_its_ledger_s_domain():
    """`where` asks the walk's ledger (`assuming`): over the workload's
    domain `B + ⌊B/2⌋` is never below one (SymPy alone does not say so);
    with no domain it is the cases."""
    def below_one(b):
        return sp.Lt(b + sp.floor(b / 2), 1)
    with assuming(domain=DOMAIN):
        assert where(below_one, (B,), S_IN, S_OUT) == S_OUT
    with assuming():
        assert isinstance(where(below_one, (B,), S_IN, S_OUT), Cases)


def test_a_ratio_is_declared_once_and_above_zero():
    """A name is one symbol (SymPy's are equal by name and assumptions), so
    a second floor would move the domain the laws already written in it
    hold on. Re-declaring the same floor is the same declaration."""
    # Equal, not identical: SymPy's symbol cache may be cleared between two
    # calls, and `_RATIO_FLOORS` is keyed by equality.
    assert ratio_symbol("rho", 1) == SYMBOLS["ep_load_factor"]
    with pytest.raises(ValueError, match="already declared from 1"):
        ratio_symbol("rho", 2)
    assert DOMAIN.least[SYMBOLS["ep_load_factor"]] == 1
    with pytest.raises(ValueError, match="above zero"):
        ratio_symbol("rho_zero_floor", 0)


def test_a_ratio_point_is_read_at_the_ratio_it_is():
    """A law read at `rho = 3/2` is read there: reading the point as a whole
    count would answer at a workload nobody asked for, and the two readers
    of a point make one conversion (`dim.point_value`)."""
    from backend.arch.eval.analysis import Point
    from backend.inference.row_roofline import LawBook

    rho = SYMBOLS["ep_load_factor"]
    book = LawBook()
    at = Point({"rho": sp.Rational(3, 2)})
    assert at.laws(book, (10 * rho,))[10 * rho] == Fraction(15)
    by_point = Point.laws_by_point(book, [({"rho": sp.Rational(3, 2)},
                                           10 * rho)])
    assert by_point({"rho": sp.Rational(3, 2)})[10 * rho] == Fraction(15)


def test_rows_are_rounded_after_the_skew_is_applied_not_before():
    """A ratio multiplies the rows and the PRODUCT is rounded, at the one
    boundary that rounds (a launch's whole-row buffer). Rounding the ratio
    first is a different number: 3 rows at `rho = 4/3` is 4 rows, and a
    `rho` read as a whole count would leave 3 — a whole row apart."""
    from backend.arch.eval.analysis import Point
    from backend.inference.row_roofline import LawBook

    rho = SYMBOLS["ep_load_factor"]
    rows, skew = 3, sp.Rational(4, 3)
    law = sp.ceiling(rows * rho)
    at = Point({"rho": skew}).laws(LawBook(), (law,))[law]
    assert at == Fraction(4)
    assert at != rows * int(skew)
