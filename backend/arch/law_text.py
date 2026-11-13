"""Reading an authored expression: which names are axes, which are operators.

A spec author writes lengths and laws as text — `d_model // n_heads`,
`Min(s_window, s_kv)`, `n_lin_v_heads * d_lin_k_head`. Turning that text into
algebra is one decision, and it used to be made in two dozen places by
calling `sympy.sympify` directly, which hands the text to SymPy's whole
namespace: `E` is Euler's number there, `I` the imaginary unit, and `beta`,
`gamma`, `zeta` are special functions. No declared dim collides with one
today, but nothing stopped one from being declared — Kimi-K3 already
binds `situ_beta` (a `constant` symbol, dims.yaml).

This module sits at the bottom of `arch`; its only package dependency is
the closed runtime-law registry (which has no architecture imports), so both `eval.dim` (which builds these laws) and `arch.schema`
(which sits below `eval` and compares authored axis text) read through it
rather than each writing the rule out. `tests/backend/arch/test_dim_symbols.py` asserts
the rule; `tests/meta/test_structure_debt.py` asserts there is one reader.
"""
from __future__ import annotations

import keyword
import re
from typing import Any, Callable, Dict, Mapping

import sympy as sp
from sympy.core.logic import fuzzy_and

from .runtime_laws import RUNTIME_LAW_OPERATORS

#: Any name in an authored expression.
_NAME = re.compile(r"[A-Za-z_]\w*")
#: A name the expression APPLIES.
_APPLIED = re.compile(r"([A-Za-z_]\w*)\s*\(")

#: The operators a law may use, and nothing else.
#:
#: A law is not arbitrary SymPy. It is BUILT by `eval.dim`'s `ceiling`,
#: `minimum`, `maximum`, `floor_div` and `ceil_div`, and printing one back is
#: where these names come from: `Min(128, s_kv)` is what `minimum(128, s_kv)`
#: looks like as text. So the vocabulary is declared, rather than inferred
#: from whether a name happens to be followed by a bracket.
#:
#: Everything else in a law is an axis. Applying a name that is not an
#: operator raises, instead of quietly meaning whatever SymPy has under that
#: name.
LAW_OPERATORS: Dict[str, Any] = {
    "Min": sp.Min,
    "Max": sp.Max,
    "ceiling": sp.ceiling,
    "floor": sp.floor,
}


class Cases(sp.Function):
    """`then` where `condition` holds, else `otherwise`: the value a law
    takes at each point, written as the cases it is (`where`).

    SymPy's own `Piecewise` would do, but its constructor folds a condition
    that holds cases of its own (a reader count summed over readers that
    read at some points only) and simplifies it, at a cost greater than the
    walk's. These cases are kept as written; a condition that is a number's
    truth — every condition, once the point's numbers are in (`Analysis.at`)
    — picks its operand."""

    nargs = 3

    @classmethod
    def eval(cls, condition, then, otherwise):
        if condition is sp.true:
            return then
        if condition is sp.false:
            return otherwise
        return None

    @staticmethod
    def _both(a: Any, b: Any) -> Any:
        """A fact of the cases where both operands share it."""
        return a if a == b else None

    def chosen(self, truth: Any) -> Any:
        """The operand the condition's truth picks."""
        return self.args[1] if truth is sp.true else self.args[2]

    def _eval_is_extended_real(self):
        _condition, then, otherwise = self.args
        return self._both(then.is_extended_real, otherwise.is_extended_real)

    def _eval_is_finite(self):
        _condition, then, otherwise = self.args
        return self._both(then.is_finite, otherwise.is_finite)

    def _eval_is_integer(self):
        _condition, then, otherwise = self.args
        return self._both(then.is_integer, otherwise.is_integer)

    def _eval_is_extended_nonnegative(self):
        _condition, then, otherwise = self.args
        return self._both(then.is_extended_nonnegative,
                          otherwise.is_extended_nonnegative)

    def _eval_is_extended_positive(self):
        _condition, then, otherwise = self.args
        return self._both(then.is_extended_positive,
                          otherwise.is_extended_positive)

    def _eval_is_zero(self):
        _condition, then, otherwise = self.args
        return self._both(then.is_zero, otherwise.is_zero)


class Quotient(sp.Function):
    """`part / whole`, and `empty` where `condition` holds — the points
    where `whole` is zero (`eval.dim.ratio`): a share of nothing, a fan-out
    over an empty input. The quotient is taken only where `whole` is not
    zero, so it is a real, finite number wherever it is taken when its two
    operands are; SymPy, which cannot see the condition, would not know a
    quotient by a possibly-zero divisor to be either."""

    nargs = 4

    @classmethod
    def eval(cls, condition, empty, part, whole):
        if condition is sp.true:
            return empty
        if condition is sp.false:
            return part / whole
        return None

    def chosen(self, truth: Any) -> Any:
        """The operand the condition's truth picks."""
        _condition, empty, part, whole = self.args
        return empty if truth is sp.true else part / whole

    def _eval_is_extended_real(self):
        _condition, empty, part, whole = self.args
        return fuzzy_and((empty.is_extended_real, part.is_extended_real,
                          whole.is_extended_real))

    def _eval_is_finite(self):
        _condition, empty, part, whole = self.args
        return fuzzy_and((empty.is_finite, part.is_finite, whole.is_finite))

    def _eval_is_extended_nonnegative(self):
        _condition, empty, part, whole = self.args
        return fuzzy_and((empty.is_extended_nonnegative,
                          part.is_extended_nonnegative,
                          whole.is_extended_nonnegative))


def read_law(text: str, *, operators: Mapping[str, Any] = LAW_OPERATORS,
             symbol: Callable[[str], Any] = sp.Symbol) -> Any:
    """Authored expression text as a SymPy expression.

    `//` is left for SymPy to read, which makes it floor division — the
    meaning the author wrote, and the reading `eval.dim.eval_dim_expr` gives
    a binding's text too.

    `operators` is the vocabulary the text may apply; an author writes in
    `LAW_OPERATORS`. A law the inference layer writes and reads back (a
    step's time by the term that set it) applies `Piecewise` too, with
    `True` as its last condition: a Python keyword is a literal, never an
    axis. `symbol` makes each axis's symbol from its name: a law read back
    reads its names with the facts it was written over.
    """
    body = str(text)
    applied = set(_APPLIED.findall(body))
    unknown = sorted(applied - set(operators))
    if unknown:
        raise ValueError(
            f"expression {text!r} applies {unknown!r}, which the law language "
            f"does not have. Its operators are {sorted(operators)}; every "
            "other name is an axis.")
    # Every name that is not one of those operators is an axis, bound as a
    # Symbol so SymPy's namespace cannot answer for it. The dim registry is
    # not consulted: that would make this module depend on `arch.dims` and
    # put a cycle through `arch`, and a name the registry has not heard of is
    # still a name.
    local: Dict[str, Any] = {
        name: symbol(name) for name in set(_NAME.findall(body))
        if name not in operators and not keyword.iskeyword(name)
    }
    local.update({name: operators[name] for name in applied})
    from sympy.parsing.sympy_parser import parse_expr
    try:
        return parse_expr(body, local_dict=local, evaluate=True)
    except Exception as exc:
        raise ValueError(f"cannot read expression {text!r}: {exc}") from exc


def as_written(extremum: Any, *args: Any) -> Any:
    """`Min` / `Max` of laws built as written. SymPy's own compare their
    arguments over the symbols' assumptions to drop the ones that can never
    win — at more cost than the rest of a row, and with a refusal where it
    cannot tell (an argument holding cases); the law reads the same at every
    point either way. Numbers are compared, as numbers."""
    args = tuple(sp.sympify(arg) for arg in args)
    if all(arg.is_number for arg in args):
        return extremum(*args)
    return extremum(*args, evaluate=False)


def read_walk_law(text: str, *,
                  symbol: Callable[[str], Any] = sp.Symbol) -> Any:
    """A law the walk wrote, read back (`read_law`). Its operators are an
    author's and the walk's own: the cases of a value that differs between
    points (`Cases`, `eval.dim.where`), the comparisons and the remainder
    their conditions are written with, a quotient taken where its divisor
    is not zero (`Quotient`, `eval.dim.ratio`), and the `Piecewise` a
    step's time by the term that set it is written with."""
    return read_law(text, operators={
        **LAW_OPERATORS,
        # As the walk wrote them (`as_written`): read back, a `Max` of laws
        # is not compared again.
        "Max": lambda *args: as_written(sp.Max, *args),
        "Min": lambda *args: as_written(sp.Min, *args),
        "Cases": Cases, "Quotient": Quotient,
        **RUNTIME_LAW_OPERATORS,
        "Piecewise": sp.Piecewise,
        "Mod": sp.Mod, "Eq": sp.Eq, "Ne": sp.Ne}, symbol=symbol)
