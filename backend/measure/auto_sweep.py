"""Sweeps: one walk per cell, substitution per point, one Kernel Bench lookup
per sweep (docs/auto_sweep.md).

A sweep asks one estimate question at many points. A *cell* is what changes
the program the engine walks (model, parallel layout, context, phase,
pipeline stage); walking it is the expensive part, so it happens once. A
*point* inside a cell differs only in what the cell's laws take as symbols
(the batch, the card), so it is priced by substituting into them. Every point
names the Kernel Bench keys of its rows; the sweep looks the union up once.

This module is the engine and knows nothing about models. What a cell's walk
produces, what the roofline law is, which key a row has and how a measured
row is selected are the estimator's (`backend.inference`), which calls in
with them; the boundary is plain data — law text, numbers, key tuples — so a
cell can be walked in a worker process and the lookup made in the API's.

  * `Cell` — one walk and the points it prices; `walk_cells` maps a walk over
    cells (the pool's `map_cells`, or the caller's mapper: the API passes
    its deadline-bound one).
  * `evaluate` / `compiled` — law text at a point's binding, compiled once
    per text, exactly (the one compiler: the estimator's row and step laws
    compile here too).
  * The measured side: `PointMeasure` (one phase of one point before any
    measurement: each row's key or why it has none, its launches per step,
    its structural weight), `sweep_keys` (the union of every point's keys),
    `look_up` (the one lookup), `MeasuredFormula` (the measured total
    Σ w_k·n_k·T_k as law text, written by the estimator's one step
    aggregation, read at one point or over a matrix of points) and
    `MeasuredTrack` (the fold and its coverage).

The measured total is all or nothing. Each row is in exactly one coverage
state (`STATES`); the total exists only when every row is `measured`, and a
point that has none reports how many rows are measured and which keys or ops
are in each other state. A row with zero cost — its own FLOPs, access bytes
and comm bytes all 0 — is measured at 0: it is not a symbol and never
missing.
"""
from __future__ import annotations

from collections.abc import Mapping as ABCMapping
import math
from functools import lru_cache
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, ClassVar, Dict, Iterable,
                    List, Mapping, Optional, Sequence, Tuple)

import sympy as sp

from fractions import Fraction

from sympy.printing.lambdarepr import LambdaPrinter

from backend.arch.law_text import read_walk_law

from . import lookup as _lookup
from .lookup import Key
from .pool import map_cells
from .scope import MeasurementScope

# ---------------------------------------------------------------------------
# Coverage states: one per row.
# ---------------------------------------------------------------------------

#: A measurement was selected for the row's key — or the row costs nothing.
MEASURED = "measured"
#: The row has a key and a declared exact implementation, but Kernel Bench has
#: no candidate for that identity.
MISSING = "missing"
#: `latest` has no unique framework dispatch (undeclared, inactive, or
#: multi-active). Not a corpus gap to measure: declare/pin one, or read the
#: envelope.
AMBIGUOUS = "ambiguous"
#: The row has no key: no measurement signature, or no measurable shape.
UNPLACED = "unplaced"
#: The key could not be stated exactly (a dtype the corpus has no token for,
#: an identity the walk could not state); shown with its reason.
DECLINED = "declined"

#: The coverage vocabulary — the one definition; every row's state is one of
#: these.
STATES = (MEASURED, MISSING, AMBIGUOUS, UNPLACED, DECLINED)

# Why a row without a lookup key has no key. This vocabulary is closed: the
# frontend and reports consume these names without parsing display text.
# `zero_launch` is not an unplaced reason in the coverage state sense, but it
# rides PointMeasure.reasons as the closed per-row explanation of measured@0.
ZERO_LAUNCH = "zero_launch"
DECLINED_REASON = "declined"
MISSING_SIGNATURE = "missing_signature"
MISSING_PROJECTION = "missing_projection"
# Pipeline stages outside the rank's own walk are structural unplaced rows.
# They are deliberately not called a missing signature or projection.
PIPELINE_STAGE = "pipeline_stage"
UNPLACED_REASONS = (MISSING_SIGNATURE, MISSING_PROJECTION, PIPELINE_STAGE)


@dataclass(frozen=True)
class MeasuredTrack:
    """The measured track of a set of rows: its coverage, and its total —
    which the one measured formula gives (`MeasureMatrix.fold`), and which
    exists only at full coverage.

    The coverage counts rows, not launches, and lists the rows in each state
    other than `measured`: `missing` / `ambiguous` by key, `unplaced` by op,
    `declined` by key with its reason. Without every row measured there is
    no total: a sum that put the roofline in for the rows Kernel Bench has
    not measured would be SOL under the measured name."""

    measured_rows: int = 0
    total_rows: int = 0
    missing: Tuple[str, ...] = ()
    ambiguous: Tuple[str, ...] = ()
    unplaced: Tuple[str, ...] = ()
    declined: Tuple[Tuple[str, str], ...] = ()
    #: The same unplaced rows as `unplaced`, without collapsing rows that share
    #: an op name. Each detail carries the row's closed reason and aggregation
    #: id; `states.unplaced` remains the existing string[] contract.
    unplaced_details: Tuple[Dict[str, str], ...] = ()
    #: How many of `measured_rows` rest on a measurement NOBODY CHECKED —
    #: a corpus row whose `cross_check` is not `passed`, almost always
    #: `no_reference`: the fp32 oracle could not be built for that cell, so
    #: the kernel's number was recorded unverified. 40,591 rows of the
    #: 2,469,869 measured are in that state (1.6%), and for the biggest
    #: attention shapes it is unavoidable — an O(S²) fp32 score matrix does
    #: not fit beside the kernel on one card.
    #:
    #: Counted because the alternative readings are both wrong. Refusing
    #: those rows would drop the only measurements the biggest cells have
    #: and fall back to the roofline, which is further from the truth than
    #: an unverified measurement. Saying nothing leaves a reader unable to
    #: tell a total built on checked numbers from one built on unchecked
    #: ones — the per-row badge has carried the verdict all along, and
    #: nothing added it up.
    unchecked_rows: int = 0
    #: Rows per non-measured state. The lists above name each key (or op)
    #: once, however many rows share it — the thing to measure — so their
    #: lengths do not add up to `total_rows - measured_rows`; these do.
    rows_by_state: Tuple[Tuple[str, int], ...] = ()
    #: The formula's total, set only at full coverage.
    total_s: Optional[float] = None

    @classmethod
    def of(cls, rows: Iterable[Tuple[str, str, str]], *,
           unchecked: int = 0,
           aggregation_ids: Sequence[str] = ()) -> "MeasuredTrack":
        """The coverage of `(state, label, reason)` per row: `label` its key
        (or op, when unplaced), `reason` its closed unplaced reason or why it
        was declined.

        `unchecked` is how many of the measured rows carry a cross-check
        verdict other than `passed`. The caller counts it because the
        caller is what holds each row's verdict (`Folded.track` reads the
        selection's badge); this records it.
        """
        n, measured = 0, 0
        lists: Dict[str, List[Any]] = {s: [] for s in STATES if s != MEASURED}
        details: List[Dict[str, str]] = []
        rows = list(rows)
        with_ids = len(aggregation_ids) > 0
        if with_ids and len(aggregation_ids) != sum(1 for _ in rows):
            raise ValueError("a coverage detail states every row once")
        for i, (state, label, reason) in enumerate(rows):
            if state not in STATES:
                raise ValueError(f"unknown coverage state {state!r}; one of {STATES}")
            n += 1
            if state == MEASURED:
                measured += 1
            elif state == DECLINED:
                lists[DECLINED].append((label, reason))
            elif state == UNPLACED:
                # Empty/None is the pre-structured form. A producer that can
                # name the row's aggregation must also name its closed reason.
                allowed_legacy = ("", None) if not with_ids else ()
                if reason not in (*allowed_legacy, *UNPLACED_REASONS):
                    raise ValueError(
                        f"unknown unplaced reason {reason!r}; one of {UNPLACED_REASONS}")
                lists[UNPLACED].append(label)
                if with_ids:
                    details.append({"op": label, "aggregation_id": aggregation_ids[i],
                                    "reason": reason})
            else:
                lists[state].append(label)
        if unchecked > measured:
            raise ValueError(
                f"{unchecked} unchecked rows of {measured} measured: a row "
                f"that was not measured has no verdict to be unchecked")
        return cls(measured, n, *(tuple(dict.fromkeys(lists[s]))
                                  for s in (MISSING, AMBIGUOUS, UNPLACED, DECLINED)),
                   unchecked_rows=int(unchecked),
                   unplaced_details=tuple(details),
                   rows_by_state=tuple((s, len(lists[s])) for s in lists
                                       if lists[s]))

    @classmethod
    def unread(cls, rows: int) -> "MeasuredTrack":
        """`rows` rows nothing was looked up for (a theory estimate, or a
        point before its sweep's lookup): no row in any state."""
        return cls(0, int(rows))

    def with_total(self, total_s: float) -> "MeasuredTrack":
        if not self.complete:
            raise ValueError("a measured total exists only at full coverage")
        return replace(self, total_s=float(total_s))

    @property
    def complete(self) -> bool:
        return self.total_rows > 0 and self.measured_rows == self.total_rows

    @property
    def t_s(self) -> Optional[float]:
        """The measured total, at full coverage only."""
        return self.total_s if self.complete else None

    def coverage(self) -> Dict[str, Any]:
        """The coverage as a response states it: `{measured, total, states}`
        (`states` holds only the non-empty lists; `declined` as
        `{key, reason}`). `unplaced_details` is additive and parallel to
        `states.unplaced`; consumers must not parse the latter's op strings."""
        states: Dict[str, Any] = {}
        for name, items in ((MISSING, self.missing), (AMBIGUOUS, self.ambiguous),
                            (UNPLACED, self.unplaced)):
            if items:
                states[name] = list(items)
        if self.declined:
            states[DECLINED] = [{"key": k, "reason": r} for k, r in self.declined]
        return {"measured": self.measured_rows, "total": self.total_rows,
                **({"unchecked": self.unchecked_rows}
                   if self.unchecked_rows else {}),
                **({"unplaced_details": [dict(d) for d in self.unplaced_details]}
                   if self.unplaced_details else {}),
                **({"states": states, "rows_by_state": dict(self.rows_by_state)}
                   if states else {})}


# ---------------------------------------------------------------------------
# Substitution: law text at a point.
# ---------------------------------------------------------------------------


class _ExactPrinter(LambdaPrinter):
    """Prints a law's rationals as `Fraction`s, so a compiled law adds and
    floors exactly and turns into a float only where a float value (a card's
    rate, a measured time) enters it. Terms print in the order the
    expression holds them (`order="none"`): an exact sum is the same sum in
    any order, and sorting them for print costs more than the rest of the
    compile."""

    _default_settings: ClassVar[dict] = {**LambdaPrinter._default_settings, "order": "none"}

    def _print_Rational(self, expr: sp.Rational) -> str:
        return f"Fraction({expr.p}, {expr.q})"

    def _print_Cases(self, expr: Any) -> str:
        # Only the case the point selects is evaluated: the other may
        # divide by what is zero there (`dim.ratio`).
        condition, then, otherwise = expr.args
        return (f"(({self._print(then)}) if ({self._print(condition)}) "
                f"else ({self._print(otherwise)}))")

    def _print_Quotient(self, expr: Any) -> str:
        # The quotient is taken only where its divisor is not zero.
        condition, empty, part, whole = expr.args
        return (f"(({self._print(empty)}) if ({self._print(condition)}) "
                f"else (_law_div({self._print(part)}, "
                f"{self._print(whole)})))")


def _each_law_once(exprs: Sequence[Any]
                   ) -> Tuple[List[Tuple[sp.Symbol, Any]], List[Any]]:
    """The laws of one compiled list, each computed once (`lambdify`'s
    `cse` hook): a law is its own variable, and a later law that holds an
    earlier one whole — a row's step time holds its FLOPs and bytes — reads
    that variable rather than printing and computing it again. Only whole
    laws of the list are named, never a part of one: each is computed
    anyway, in the list's order, so no part of a case is computed where
    the case does not select it (`_print_Cases`)."""
    named: Dict[Any, sp.Symbol] = {}
    assigned: List[Tuple[sp.Symbol, Any]] = []
    out: List[Any] = []
    for i, expr in enumerate(exprs):
        if not isinstance(expr, sp.Basic) or expr.is_Atom:
            out.append(expr)
            continue
        if expr in named:
            out.append(named[expr])
            continue
        variable = sp.Symbol(f"_law{i}")
        if named:
            # Rebuilt as written: a `Max` of a variable is not compared again.
            with sp.evaluate(False):
                expr = expr.xreplace(named)
        assigned.append((variable, expr))
        named[expr] = variable
        out.append(variable)
    return assigned, out


def compile_law(exprs: Sequence[Any], names: Tuple[str, ...]
                ) -> Callable[..., Any]:
    """Laws compiled over `names`, in that order, exactly (`_ExactPrinter`):
    one function returning their values in order — every point of a sweep
    substituting into it. A name a law reads that is not in `names`
    raises."""
    by_name = {sym.name: sym for expr in exprs for sym in expr.free_symbols}
    stray = sorted(set(by_name) - set(names))
    if stray:
        raise ValueError(f"laws read {stray}, which the point does not bind "
                         f"(it binds {sorted(names)})")
    return sp.lambdify([by_name.get(n, sp.Symbol(n)) for n in names],
                       list(exprs),
                       modules=[{"math": math, "Fraction": Fraction,
                                 "_law_div": exact_div}],
                       printer=_ExactPrinter,
                       cse=_each_law_once)


class _FloatPrinter(_ExactPrinter):
    """EXPERIMENT (exp/float-projection): the same law graph, evaluated in
    machine floats — a rational prints as one division, `_law_div` is the
    plain `/`. Only `_print_Rational` changes: `Cases`/`Quotient` keep their
    select-only-the-taken-arm shape (that is semantics, not precision)."""

    def _print_Rational(self, expr: sp.Rational) -> str:
        return f"({expr.p}/{expr.q})"


def _eachLawInOrder(exprs: Sequence[Any]
                    ) -> Tuple[List[Tuple[sp.Symbol, Any]], List[Any]]:
    """Like `_each_law_once`, but every expression is assigned (even a repeat
    or an atom), because a CSE graph's later expressions read the variables
    the earlier ones define."""
    assigned: List[Tuple[sp.Symbol, Any]] = []
    out: List[Any] = []
    for index, expr in enumerate(exprs):
        variable = sp.Symbol(f"_law{index}")
        assigned.append((variable, expr))
        out.append(variable)
    return assigned, out


def compile_law_float(exprs: Sequence[Any], names: Tuple[str, ...]
                      ) -> Callable[..., Any]:
    """EXPERIMENT: `compile_law`'s float twin — same graph, same names,
    float arithmetic. For measuring the projection's exact-arithmetic cost,
    not for shipping."""
    by_name = {sym.name: sym for expr in exprs for sym in expr.free_symbols}
    return sp.lambdify([by_name.get(n, sp.Symbol(n)) for n in names],
                       list(exprs),
                       modules=[{"math": math, "_law_div": lambda a, b: a / b}],
                       printer=_FloatPrinter,
                       cse=_eachLawInOrder)


def exact_div(part: Any, whole: Any) -> Any:
    """A quotient of two whole numbers stays exact in compiled law text.

    Python's ``/`` always makes a float, even for ``int / int``; a law's
    `Quotient` is exact until a card's float rate enters it, like every other
    rational in the law language.
    """
    if isinstance(part, int) and isinstance(whole, int):
        return Fraction(part, whole)
    return part / whole


# Compatibility for readers that learned the runtime helper by its old name.
_exact_div = exact_div


@lru_cache(maxsize=8)
def _compiled_law(text: str, names: Tuple[str, ...]) -> Any:
    return compile_law([read_walk_law(text)], names)


def evaluate(text: str, binding: Mapping[str, Any]) -> Any:
    """Law `text` at a point: every symbol it reads bound by `binding`
    (name → number). Exact where no float enters."""
    names = tuple(sorted(binding))
    return _compiled_law(str(text), names)(
        *(binding[n] for n in names))[0]


# ---------------------------------------------------------------------------
# The measured side.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasuredFormula:
    """A measured total over its terms as law text, Σ_k w_k·n_k·T_k — the
    estimator writes it (one aggregation with the SOL total; the weights,
    a draft's forwards or a pipeline's stages, are in the text) — and which
    terms cost nothing (0 in the text, no symbol). `n_k` (launches per step)
    and `T_k` (one launch's measured time) are bound per point."""

    text: str
    zero_cost: Tuple[bool, ...]

    def at(self, launches: Sequence[float], times: Sequence[Optional[float]],
           terms: Optional[Iterable[int]] = None) -> float:
        """The total at a point — or, with `terms`, the same formula's share
        of those terms alone (every other term's time bound to 0): one
        formula, one compiled form, whatever part of it is asked."""
        only = None if terms is None else set(terms)
        binding: Dict[str, float] = {}
        for k, z in enumerate(self.zero_cost):
            if not z:
                binding[f"n_{k}"] = float(launches[k])
                binding[f"T_{k}"] = (0.0 if only is not None and k not in only
                                     else float(times[k]))
        return float(evaluate(self.text, binding)) if binding else 0.0

    def at_many(self, launches: Any, times: Any) -> Any:
        """The same law at many points, with two-dimensional bindings.

        The expression is the one declaration `at` reads; only its bindings
        are columns. This is the record-level fold's total, not a second
        Gallery formula.
        """
        count = len(self.zero_cost)
        if not isinstance(launches, ABCMapping):
            launches = {
                f"n_{i}": launches[:, i] for i in range(count)}
        if not isinstance(times, ABCMapping):
            times = {f"T_{i}": times[:, i] for i in range(count)}
        return evaluate(self.text, {**launches, **times})


def blend(t_sol: float, sol_measured: float, measured_part_s: float
          ) -> Dict[str, float]:
    """The partial track — its one definition: the step's SOL total
    `t_sol` with the measured terms' SOL (`sol_measured`, the same terms
    the SOL total weights) swapped for their measured share `measured_part_s`.
    The SOL total and the measured formula are one aggregation
    (`StepSum.total` over `t_op` and over `n·T`), so a term's two shares
    are interchangeable. `t_s` is the blend, `measured_share` the part of
    it that is measured. Whoever holds the SOL states `sol_measured`: a
    walk's priced rows (`Folded.partial`), or a pool point's floor form at
    the reader's floors (`row_roofline.FloorForm.terms_at`)."""
    t_s = float(t_sol) - float(sol_measured) + float(measured_part_s)
    return {"t_s": t_s, "measured_part_s": float(measured_part_s),
            "measured_share": (float(measured_part_s) / t_s
                               if t_s > 0 else 0.0)}


def partial_of(formula: "MeasuredFormula", measured_terms: Sequence[int],
               measured_part_s: float, sol: Sequence[Optional[float]],
               t_sol: Optional[float]) -> Optional[Dict[str, float]]:
    """The partial track (`blend`) off a measure's own SOL leaves
    (`PointMeasure.sol`), weighted by `formula` as the SOL total weights
    them: the one reading both folds (`Folded`, `MatrixFold`) share. None
    when the step has no SOL total, or the measure states no SOL — a pool
    point, whose SOL is a function of the floors its reader picks
    (`row_roofline.FloorForm.terms_at`). At full coverage the blend is the
    measured total by construction: the SOL total is its terms' sum."""
    if t_sol is None or not sol:
        return None
    live = [k for k in measured_terms if not formula.zero_cost[k]]
    unpriced = [k for k in live if k >= len(sol) or sol[k] is None]
    if unpriced:
        raise ValueError(
            f"measured terms {unpriced[:5]} have no SOL leaf: a measured "
            "term is a priced row of this walk")
    width = len(formula.zero_cost)
    leaves = [0.0 if v is None else float(v) for v in sol]
    leaves += [0.0] * (width - len(leaves))
    return blend(t_sol, formula.at([1.0] * width, leaves, live),
                 measured_part_s)


def key_label(key: Key) -> str:
    """A key as the coverage lists it."""
    return f"{key.signature} {key.shape_json}"


@dataclass(frozen=True)
class PointMeasure:
    """One phase of one point before measurement — plain data a worker hands
    back: per row its lookup key (None when it has none), its launches per
    step, and what it is when it is not looked up (`measured` for a zero-cost
    row, `unplaced` / `declined` for a row with no key, with `labels` /
    `reasons` saying which); the phase's formula; and what the selection of a
    measured row reads (`context`: the strategy, the framework, the card —
    the estimator's, opaque here)."""

    keys: Tuple[Optional[Key], ...]
    launches: Tuple[float, ...]
    unkeyed: Tuple[Optional[str], ...]
    labels: Tuple[str, ...]
    reasons: Tuple[str, ...]
    formula: MeasuredFormula
    context: Any = None
    #: What the selection reads of each row beside its key (the estimator's,
    #: opaque here; plain data), or empty when it reads nothing.
    inputs: Tuple[Any, ...] = ()
    #: Per-row selection contexts. Empty uses `context` for every row; a
    #: chunk fold retains an unsafe duplicate-key group whose contexts differ.
    contexts: Tuple[Any, ...] = ()
    #: The per-op row's aggregation id, kept parallel to every other row field.
    #: Empty means an older serialized measure; it has no structured details.
    aggregation_ids: Tuple[str, ...] = ()
    #: Each term's SOL time at the point — one occurrence, before the
    #: formula's weight, the leaf the SOL total adds (`StepSum.total` over
    #: `t_op`) — or None where the producer has none (a pipeline stage not
    #: walked here). Empty when the producer states none: a pool point,
    #: whose SOL is a function of the floors its reader picks
    #: (`row_roofline.FloorForm.terms_at`).
    sol: Tuple[Optional[float], ...] = ()

    def __post_init__(self) -> None:
        n = len(self.keys)
        if self.inputs and len(self.inputs) != n:
            raise ValueError("a point measure states every field once per row")
        if self.contexts and len(self.contexts) != n:
            raise ValueError("a point measure states every field once per row")
        if not (len(self.launches) == len(self.unkeyed) == len(self.labels)
                == len(self.reasons) == len(self.formula.zero_cost) == n):
            raise ValueError("a point measure states every field once per row")
        if self.aggregation_ids and len(self.aggregation_ids) != n:
            raise ValueError("a point measure states every field once per row")
        if self.sol and len(self.sol) != n:
            raise ValueError("a point measure states every field once per row")
        for k, u, z in zip(self.keys, self.unkeyed, self.formula.zero_cost, strict=True):
            if z and u != MEASURED:
                raise ValueError("a zero-cost row is measured at 0")
            if not z and (k is None) == (u is None):
                raise ValueError("a row has a key, or a state saying why it has none")
        for _key, state, reason, zero in zip(
                self.keys, self.unkeyed, self.reasons,
                self.formula.zero_cost, strict=True):
            structured = bool(self.aggregation_ids)
            if zero and reason not in (*(("",) if not structured else ()),
                                       ZERO_LAUNCH):
                raise ValueError("a zero-cost row says why it was measured at 0")
            if (state == UNPLACED
                    and reason not in (*(("",) if not structured else ()),
                                       *UNPLACED_REASONS)):
                raise ValueError(
                    f"unknown unplaced reason {reason!r}; one of {UNPLACED_REASONS}")

    def key_set(self) -> List[Key]:
        """The keys this point looks up."""
        return [k for k, z in zip(self.keys, self.formula.zero_cost, strict=True)
                if k is not None and not z]

    def fold(self, select: Callable[[Key, Any, str, Any], Tuple[Optional[float], str, Any]]
             ) -> Any:
        """The measurement at this point, as one row of a `MeasureMatrix`.

        `select(key, context,
        label, inputs)` is the estimator's selection over the sweep's one
        lookup (`label` the row's own, which a kernel pin reads; `inputs`
        what it reads of the row): one launch's time, `measured` and what it
        says of the measurement (`detail`, opaque), or None, `missing` /
        `ambiguous` and None."""
        from backend.measure.measure_matrix import MeasureMatrix
        return MeasureMatrix.of_measures([self]).fold(select)[0]

    def to_json(self) -> Dict[str, Any]:
        """Plain data, compact: every string (key fields, labels) once in a
        table, each key and label an index into it — a point's keys repeat
        their card, phase and dtype environment on every row."""
        table: Dict[str, int] = {}

        def ix(text: str) -> int:
            return table.setdefault(text, len(table))
        return {"strings": None,  # filled below, after every index is taken
                "keys": [[ix(f) for f in k] if k is not None else None for k in self.keys],
                "labels": [ix(label) for label in self.labels],
                "launches": list(self.launches), "unkeyed": list(self.unkeyed),
                "reasons": list(self.reasons),
                **({"aggregation_ids": list(self.aggregation_ids)}
                   if self.aggregation_ids else {}),
                "zero_cost": list(self.formula.zero_cost),
                "formula": self.formula.text, "context": self.context,
                "inputs": list(self.inputs),
                **({"contexts": list(self.contexts)}
                   if self.contexts else {}),
                **({"sol": list(self.sol)} if self.sol else {})} | {"strings": list(table)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "PointMeasure":
        strings = d["strings"]
        return cls(tuple(Key(*(strings[i] for i in k)) if k is not None else None
                         for k in d["keys"]),
                   tuple(float(n) for n in d["launches"]), tuple(d["unkeyed"]),
                   tuple(strings[i] for i in d["labels"]), tuple(d["reasons"]),
                   MeasuredFormula(d["formula"], tuple(d["zero_cost"])),
                   d.get("context"), tuple(d.get("inputs") or ()),
                   tuple(d.get("contexts") or ()),
                   tuple(d.get("aggregation_ids") or ()),
                   tuple(None if v is None else float(v)
                         for v in d.get("sol") or ()))


@dataclass(frozen=True)
class Folded:
    """A point measure at one point: per term `(state, measured time per
    step or None, label, detail)` — `detail` the selection's word on a
    measured row, the reason on a declined one — and the totals the one
    formula gives: the whole (`track`) and any part (`subtotal`)."""

    measure: PointMeasure
    rows: Tuple[Tuple[str, Optional[float], str, Any], ...]
    times: Tuple[Optional[float], ...]

    @property
    def track(self) -> MeasuredTrack:
        """The coverage of every term, and the formula's total at full
        coverage."""
        # The verdict rides in the selection's badge, which is where the
        # row's own `cross_check` arrives (`op_calibration`'s measured
        # badge). A measured row with no badge is one nothing was looked up
        # for — a zero-cost term — and has nothing to be unchecked.
        unchecked = sum(
            1 for state, _t, _label, detail in self.rows
            if state == MEASURED and isinstance(detail, dict)
            and ((detail.get("source_badge") or {}).get("cross_check")
                 or "passed") != "passed")
        track = MeasuredTrack.of(
            ((state, label, detail if state in (DECLINED, UNPLACED) else "")
             for state, _t, label, detail in self.rows),
            unchecked=unchecked,
            aggregation_ids=self.measure.aggregation_ids)
        if not track.complete:
            return track
        return track.with_total(self.measure.formula.at(self.measure.launches, self.times))

    @property
    def measured_terms(self) -> Tuple[int, ...]:
        """The terms with a measured time (a zero-cost term among them)."""
        return tuple(k for k, row in enumerate(self.rows) if row[0] == MEASURED)

    @property
    def measured_part_s(self) -> float:
        """The formula's share of the measured terms: what the partial
        track puts in place of their SOL."""
        return self.measure.formula.at(self.measure.launches, self.times,
                                       self.measured_terms)

    def partial(self, t_sol: Optional[float]) -> Optional[Dict[str, float]]:
        """The partial track off the measure's own SOL leaves
        (`partial_of`)."""
        return partial_of(self.measure.formula, self.measured_terms,
                          self.measured_part_s, self.measure.sol, t_sol)

    def subtotal(self, terms: Iterable[int]) -> Optional[float]:
        """The formula's share of `terms` — a draft's forwards, a module's
        subtree — only when every one of them is measured; else None."""
        terms = list(terms)
        if not terms or any(self.rows[k][0] != MEASURED for k in terms):
            return None
        return self.measure.formula.at(self.measure.launches, self.times, terms)


def sweep_keys(measures: Iterable[PointMeasure]) -> List[Key]:
    """Every key the sweep's points look up, each once, in first-seen order."""
    return list(dict.fromkeys(k for m in measures for k in m.key_set()))


def look_up(scope: MeasurementScope, keys: Sequence[Key]) -> _lookup.Lookup:
    """The sweep's one lookup: the candidate rows of every key under `scope`
    (`measure.lookup.lookup`; `NONE` asks nothing)."""
    return _lookup.lookup(scope, keys)


# ---------------------------------------------------------------------------
# Cells.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One walk and the points it prices: `walk` is the plain data the
    estimator's walk reads (picklable — it may run in a worker), `points`
    the coordinates substituted inside it."""

    walk: Any
    points: Tuple[Any, ...] = field(default=())


def walk_cells(walk: Callable[[Cell], Any], cells: Sequence[Cell], *,
               mapper: Callable[[Callable[[Cell], Any], List[Cell]], List[Any]] = map_cells
               ) -> List[Any]:
    """One walk per cell, in cell order: `walk` mapped over `cells` — across
    the process pool by default (`pool.map_cells`), or by the caller's
    mapper. `walk` must be a top-level function."""
    return list(mapper(walk, list(cells)))
