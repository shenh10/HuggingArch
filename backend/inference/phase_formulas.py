"""Popover formulas for a phase's per-op rows.

These strings are display, not the roofline. They live next to `phases`
because inference must not import rendering (§2.7); the API is the
composition root that may hand them to a page.
"""
from __future__ import annotations
import copy
import re
from dataclasses import dataclass, field
from types import MappingProxyType

import sympy as sp
from sympy.printing.str import StrPrinter

from backend.arch.eval.dim import as_number, bind_record, exact as exact_number
from backend.arch.law_text import Cases, Quotient, read_walk_law
from backend.arch.layout.comm_volume import ring_factor, ring_factor_symbolic
from backend.inference.intensity import arithmetic_intensity, intensity_formula

from typing import (Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence,
                    TYPE_CHECKING, Tuple)


if TYPE_CHECKING:
    from backend.arch.eval.access import LeafAccess


#: A Σ-children `substituted` line stays scannable: terms sorted
#: large→small and capped; the Σ and the total keep counting everything the
#: cap dropped, and the cap itself is said in the generic `note` field.
MAX_COMPONENT_TERMS = 20


def _finalize_parts(parts: List[tuple]) -> tuple:
    ordered = sorted(parts, key=lambda p: p[1], reverse=True)
    if len(ordered) <= MAX_COMPONENT_TERMS:
        return ordered, 0
    return ordered[:MAX_COMPONENT_TERMS], len(ordered) - MAX_COMPONENT_TERMS


def _substituted_sum(shown: List[tuple], total: float, fmt: Any) -> str:
    """The generic formula line every other formula already uses:
    `a(x) + b(y) … = total`."""
    return (" + ".join(f"{name}({fmt(value)})" for name, value in shown)
            + f" = {fmt(total)}")


#: A name as a formula line spells it.
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


@dataclass(frozen=True)
class FormulaDefs:
    """The intermediate quantities a per-op row's popover names, at one
    point: each name its symbolic line reads whose law (the row's node dims,
    `point_law["dims"]`) is neither the name's own symbol nor a number —
    `s_kv`, `n_verify`, a phase's derived extents — and the cache read the
    access line calls `kv` (the row's `kv_read` law, one instance's share).
    Each is shown as `name = <its law with the point's counts bound,
    unevaluated> = value` (`dim.bound_formula_text`): the page shows numbers,
    never a law with a variable in it. A name the symbolic line does not
    read, a config dim (a number) and a workload count itself (its own
    symbol) are not listed.

    `value` reads a law at the point (the cell's book, or a reading of the
    row's laws there); `point` is the point's counts by name."""

    dims: Optional[Mapping[str, Any]]
    point: Mapping[str, int]
    value: Any
    kv_law: Any = None
    kv_total: int = 0
    count: int = 1
    #: The row's declared definitions (`point_law["definitions"]`:
    #: `{name: {law, bindings}}`), each name's law in names and what each
    #: of those names is bound to there.
    definitions: Optional[Mapping[str, Any]] = None

    def attach(self, formula: Dict[str, Any]) -> Dict[str, Any]:
        """`formula` with its `defs` beside its lines, where it names any:
        each `{name, formula, value}`, the formula its law at the point
        folded to its branch (`_BranchFold`), kept only where it comes to
        the value and is more than the number itself."""
        defs: List[Dict[str, Any]] = []
        for name in dict.fromkeys(_NAME.findall(formula.get("symbolic") or "")):
            if name == "kv":
                law, count = self.kv_law, max(int(self.count or 1), 1)
                if not (isinstance(law, sp.Basic) and law.free_symbols
                        and self.kv_total):
                    continue
                if count != 1:
                    law = sp.Mul(law, sp.Rational(1, count), evaluate=False)
                value = sp.Rational(int(self.kv_total), count)
            else:
                law = (self.dims or {}).get(name)
                # A number is a config dim, and a single symbol is a count
                # or another name for one (`n_verify` is `s_verify`):
                # neither is an intermediate quantity.
                if (not isinstance(law, sp.Basic) or not law.free_symbols
                        or law.is_Symbol):
                    continue
                value = self.value(law)
            # The law with the point's counts put in, folded; none where a
            # name the point does not bind is left.
            folded = (None if any(symbol.name not in self.point
                                  for symbol in law.free_symbols)
                      else _Folding(self.point).fold(law))
            if (folded is None or folded.number
                    or folded.value != exact_number(value)):
                continue
            number = as_number(value)
            entry = {"name": name, "formula": folded.text(outer=True),
                     "value": (f"{number:,d}" if isinstance(number, int)
                               else str(number))}
            entry.update(self._defined(name, value))
            defs.append(entry)
        if defs:
            formula["defs"] = defs
        return formula

    def _defined(self, name: str, value: Any) -> Dict[str, str]:
        """A declared name's definition and that definition with its names'
        values at the point put in, kept as written (`bind_keep_products`):
        `{definition, formula}`; nothing for a name with no declared
        definition. A definition that reads a name it does not bind, or
        that does not come to `value` at the point, contradicts the number
        beside it and is refused."""
        declared = (self.definitions or {}).get(name)
        if declared is None:
            return {}
        # Physical projection can compose an authored definition with a
        # workload-dependent share (e.g. sparse DCP's guarded quotient).
        # These are serialized walk laws, including their conditional terms.
        law = read_walk_law(declared["law"])
        values = {}
        for symbol in law.free_symbols:
            bound = declared["bindings"].get(symbol.name)
            if bound is None:
                raise ValueError(
                    f"`{name}` is declared {declared['law']}, which reads "
                    f"`{symbol.name}`, and the declaration binds no value "
                    f"for it")
            values[symbol.name] = (as_number(bound["value"])
                                   if isinstance(bound, dict)
                                   else self.value(bound))
        exact = {name: exact_number(number)
                 for name, number in values.items()}
        reached = law.xreplace({symbol: exact[symbol.name]
                                for symbol in law.free_symbols})
        if reached != exact_number(value):
            raise ValueError(
                f"`{name}` is declared {declared['law']}, which comes to "
                f"{reached} at this point, and its value is {value}")
        return {"definition": _readable(law),
                "formula": _readable(_BoundPrinter(exact).doprint(law))}


@dataclass
class _BranchFold:
    """A law at a point, its arithmetic folded and its branches kept: what
    a definition's line shows (`FormulaDefs`). Folded from the inside out,
    every sum and product of numbers is one number; a branch — `Min`,
    `Max`, `floor`, `ceiling`, the cases of a value — is its name over its
    arguments' numbers where it is the whole law or a term of it, so the
    reader sees which side was taken, and its number deeper in. A product
    that holds a branch keeps its factors, as numbers (`text`)."""

    _BRANCHES: ClassVar[Mapping[Any, str]] = MappingProxyType(
        {sp.Min: "Min", sp.Max: "Max", sp.floor: "floor", sp.ceiling: "ceiling"})

    value: Any
    branch: Optional[Tuple[str, List[str]]] = None
    terms: Optional[List["_BranchFold"]] = None
    factors: Optional[List["_BranchFold"]] = None
    #: The cases this is the taken operand of, shown where they are the
    #: whole law.
    cases: Optional[Tuple[str, List[str]]] = None

    @property
    def number(self) -> bool:
        return self.branch is None and self.terms is None and self.factors is None

    def text(self, *, outer: bool = False) -> str:
        """The line: the whole law's branch as its name over its numbers;
        a sum, its terms — a term that is a branch shown as one, so a cap
        added to a window reads `128 + Min(512, 1024)`; a product, its
        factors' numbers. Below that every part is its number."""
        shown = (self.cases or self.branch) if outer else None
        if shown is not None:
            name, args = shown
            return f"{name}({', '.join(args)})"
        if not outer:
            return _text_of(self.value)
        if self.terms is not None:
            out = ""
            for i, term in enumerate(self.terms):
                if i and term.number and term.value < 0:
                    out += f" - {_text_of(-term.value)}"
                    continue
                piece = term.text(outer=term.branch is not None or term.cases is not None)
                out += piece if not i else f" + {piece}"
            return out
        if self.factors is not None:
            top: List[str] = []
            below: List[str] = []
            for factor in self.factors:
                if factor.value.is_Rational and factor.value.p == 1 and factor.number:
                    below.append(_text_of(factor.value.q))
                    continue
                top.append(_text_of(factor.value))
            line = "·".join(top) or "1"
            return line + "".join(f"/{q}" for q in below)
        return _text_of(self.value)


@dataclass
class _Folding:
    """One law folded at one point (`FormulaDefs`): each subexpression
    folded once — a law shares its subtrees — with the point's counts put
    in where its symbols are."""

    point: Mapping[str, int]
    _done: Dict[Any, _BranchFold] = field(default_factory=dict)

    def fold(self, node: Any) -> _BranchFold:
        done = self._done.get(node)
        if done is None:
            done = self._done[node] = self._fold(node)
        return done

    def _fold(self, node: Any) -> _BranchFold:
        cls = _BranchFold
        if node.is_Symbol:
            return cls(exact_number(self.point[node.name]))
        if node.is_Number:
            return cls(exact_number(node))
        if node.func in cls._BRANCHES:
            args = [self.fold(arg).value for arg in node.args]
            return cls(node.func(*args), branch=(cls._BRANCHES[node.func],
                                                 [_text_of(a) for a in args]))
        if isinstance(node, (Cases, Quotient)):
            condition = node.args[0]
            truth = self._truth(condition)
            chosen = self.fold(node.chosen(truth))
            if isinstance(node, Quotient):
                return chosen
            # Part of a law, the cases are the operand they take; the whole
            # law, they are the cases, so the side taken shows.
            then, otherwise = (self.fold(node.args[1]).value,
                               self.fold(node.args[2]).value)
            shown = copy.copy(chosen)
            shown.cases = ("Cases", [self._condition_text(condition),
                                     _text_of(then), _text_of(otherwise)])
            return shown
        if node.is_Add:
            parts = [self.fold(arg) for arg in node.args]
            constant = sum((p.value for p in parts if p.number), sp.Integer(0))
            rest = [p for p in parts if not p.number]
            if not rest:
                return cls(constant)
            if constant != 0:
                rest.append(cls(constant))
            if len(rest) == 1:
                return rest[0]
            return cls(sum((p.value for p in rest), sp.Integer(0)), terms=rest)
        if node.is_Mul:
            parts: List[_BranchFold] = []
            for arg in node.args:
                folded = self.fold(arg)
                parts.extend(folded.factors if folded.factors is not None else [folded])
            value = sp.Integer(1)
            for p in parts:
                value *= p.value
            if all(p.number for p in parts):
                return cls(value)
            kept = [p for p in parts if not (p.number and p.value == 1)]
            if len(kept) == 1:
                return kept[0]
            return cls(value, factors=kept)
        # Anything else — a power, a function the branch list does not
        # name — is the number it comes to.
        args = [self.fold(arg).value for arg in node.args]
        return cls(exact_number(node.func(*args)))

    def _truth(self, condition: Any) -> Any:
        if isinstance(condition, sp.core.relational.Relational):
            return condition.func(self.fold(condition.lhs).value,
                                  self.fold(condition.rhs).value)
        if isinstance(condition, (sp.And, sp.Or, sp.Not)):
            return condition.func(*(self._truth(arg) for arg in condition.args))
        return condition

    def _condition_text(self, condition: Any) -> str:
        if isinstance(condition, sp.core.relational.Relational):
            return (f"{_text_of(self.fold(condition.lhs).value)} "
                    f"{condition.rel_op} {_text_of(self.fold(condition.rhs).value)}")
        return str(condition)


class _BoundPrinter(StrPrinter):
    """A law printed as written with each name's number in its place: the
    definition's own order and grouping, numbers where the names were."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        super().__init__()
        self._values = values

    def _print_Symbol(self, expr: Any) -> str:
        return _text_of(self._values[expr.name])


def _readable(law: Any) -> str:
    """A definition's line as the where column prints it: `·` for a
    product, `^` for a power."""
    return str(law).replace("**", "^").replace("*", "·")


def _text_of(value: Any) -> str:
    """A number as a definition's line spells it: whole, or the fraction.

    The walk's numbers are exact already — an `int` or a `Fraction`
    (`Point.laws`) — so getting one into a SymPy number is `dim.exact`, a
    conversion. It is NOT `nsimplify`, which is the opposite direction:
    guessing a closed form back out of an approximation. Handed `1/32` with
    nothing to guess, `nsimplify` still runs mpmath's integer-relation
    search to rediscover the denominator it was given."""
    value = exact_number(value)
    if value.is_Integer:
        return str(int(value))
    if value.is_Rational:
        return f"{value.p}/{value.q}"
    return str(value)


def _flops_formula_for_row(
    flops_expr: Any, flops_value: Any,
    record: Optional[Dict[Any, Any]], *, cnt: int, result_value: int,
) -> Dict[str, Any]:
    """Build the `{symbolic, substituted, result}` tooltip for a
    per-op row's prefill FLOPs cell.

    `flops_expr` / `flops_value` are one instance's FLOPs, its identity
    and its exact value. The substituted line binds the expression with the
    substitution record the FLOPS evaluation itself used
    (`flops_from_shape`'s `record`, keyed by identity — bare symbols and
    composites like `n_heads*d_head` alike), so it reads
    `2*4*8192*3072*1024`, not the folded `206158430208`. The bound form is
    displayed only when it reproduces the value exactly; without a record,
    or on a mismatch, the folded number stands. The layer multiplier is
    surfaced separately. A row re-read at another batch passes the same
    three at that batch (`batch_sweep`), so its line is this one there.
    """
    pieces_sym = []
    pieces_val = []
    if flops_value:
        bound = bind_record(flops_expr, record, flops_value)
        pieces_sym.append(f"({flops_expr})")
        pieces_val.append(f"({bound if bound is not None else flops_value})")
    per_layer_flops = result_value // max(int(cnt or 1), 1) if cnt else result_value
    out: Dict[str, Any] = {
        "symbolic": " ".join(pieces_sym) if pieces_sym else "",
        "substituted": " ".join(pieces_val) if pieces_val else "",
        "result": _fmt_si(result_value, " FLOPs"),
    }
    if cnt and cnt != 1:
        out["per_layer_result"] = _fmt_si(per_layer_flops, " FLOPs")
        out["cnt"] = int(cnt)
        out["cnt_unit"] = "layers"
    return out


def access_display_facts(ab: LeafAccess) -> Dict[str, Any]:
    """What a leaf's bytes popover prints off its access record
    (`_access_bytes_formula_for_row`) that does not move with the batch, as
    plain data a per-op row carries (`point_law`). The shape values are the
    point's and are passed beside it. A term whose shape is all literal
    axes prints its element count, which then cannot move either; one with
    neither prints its bytes whole, and only that term's bytes are kept —
    the walk's."""
    in_dt = float(ab.act_in_dtype_bytes or ab.act_dtype_bytes)
    out_dt = float(ab.act_out_dtype_bytes or ab.act_dtype_bytes)
    # A lookup table's weight read is the rows its lookups gather, not the
    # table its params count: the term prints the bytes it read.
    gathered = ab.table_read_per_rank is not None
    # Every term is the PER-RANK one the row charges: the shapes and params
    # print per-rank values; a term with no per-rank structure to print
    # (no symbolic params, a scale count) prints the per-rank bytes rather
    # than a global count the rank does not hold.
    structured = bool(ab.params_sym) and not gathered
    return {
        "weight": bool(ab.weight_bytes),
        "params_sym": ab.params_sym if structured else None,
        "params_val": ab.params_val,
        "weight_dtype_bytes": float(ab.weight_dtype_bytes),
        "weight_bytes": None if structured else ab.weight_bytes_per_rank,
        # A lookup table's weight read is the rows its lookups gather
        # (`table_read_per_rank`): it moves with the batch, and a point
        # prints its row law's weight read there, not the walk's bytes.
        "weight_gathered": ab.table_read_per_rank is not None,
        "act_in": bool(ab.act_in_bytes), "in_dt": in_dt,
        "act_out": bool(ab.act_out_bytes), "out_dt": out_dt,
        "metadata": bool(ab.metadata_bytes_per_rank),
        "metadata_bytes": ab.metadata_bytes_per_rank,
        # The activation read as `traffic` priced it, term by term
        # (`LeafAccess.in_terms_per_rank`), with the cache read and the
        # operand conversion it is charged beside.
        "in_terms": [[term.index, float(term.fanout), float(term.width),
                      None if term.spanned else float(term.elements)]
                     for term in ab.in_terms_per_rank],
        "shape_in_syms": list(ab.shape_in_syms),
        # The activation write as `traffic` priced it, term by term
        # (`LeafAccess.out_terms_per_rank`): each written tensor's width, and
        # its elements where they are no longer the whole of what the port
        # publishes (a fused boundary that rewrote a region of its output).
        "out_terms": [[float(term.width),
                       None if term.spanned else float(term.elements)]
                      for term in ab.out_terms_per_rank],
        "shape_out_syms": list(ab.shape_out_syms),
        "kv_bytes": ab.kv_read_bytes_per_rank,
        "convert_bytes": ab.convert_bytes_per_rank,
    }


def _num(value: float) -> str:
    """A number as a formula prints it: whole when it is whole, else in
    full (a narrow width's scale share, 1 + 4/128 = 1.03125)."""
    return f"{int(value):,d}" if float(value) == int(value) else repr(float(value))


def act_in_from_terms(facts: Mapping[str, Any],
                       shape_in_vals: Optional[Sequence[Optional[str]]],
                       shape_out_vals: Optional[Sequence[Optional[str]]]
                       ) -> Tuple[str, str]:
    """The activation-read term off the derivation's own edge records
    (`facts["in_terms"]`): per edge, its input's shape over the fan-out the
    price divided by, at the width it was priced at — or, where the cache law
    took the edge's span over, the elements it read. The cache read follows,
    so the term states the bytes charged."""
    def _wrap(text: str) -> str:
        return f"({text})" if any(c in text for c in "+-/") else text

    syms = facts.get("shape_in_syms") or []
    vals = list(shape_in_vals or ())
    in_dt = facts["in_dt"]
    terms_sym: List[str] = []
    terms_val: List[str] = []
    for index, fanout, width, elements in facts["in_terms"]:
        dt_sym = "dt_a" if width == in_dt else _num(width)
        if elements is not None:
            if not elements:
                continue
            terms_sym.append(f"rows_in{index}*{dt_sym}")
            terms_val.append(f"{_num(elements)}*{_num(width)}")
            continue
        # Index -1 is an in-place update's read of its own (main) output.
        side_syms, side_vals, at, name = (
            (facts.get("shape_out_syms") or [], list(shape_out_vals or ()),
             0, "Πshape_out") if index < 0
            else (syms, vals, index, f"Πshape_in{index}"))
        sym = (side_syms[at] if at < len(side_syms) and side_syms[at]
               else name)
        val = (side_vals[at] if at < len(side_vals) and side_vals[at]
               else sym)
        div = "" if fanout == 1 else f"/{_num(fanout)}"
        terms_sym.append(f"{_wrap(sym)}{div}*{dt_sym}")
        terms_val.append(f"{_wrap(val)}{div}*{_num(width)}")
    sym = " + ".join(terms_sym)
    val = " + ".join(terms_val)
    if facts.get("kv_bytes"):
        sym = f"{sym} + kv" if sym else "kv"
        val = f"{val} + {_num(facts['kv_bytes'])}" if val else _num(facts["kv_bytes"])
    return sym, val


def act_out_from_terms(facts: Mapping[str, Any],
                        shape_out_vals: Optional[Sequence[Optional[str]]]
                        ) -> Tuple[str, str]:
    """The activation-write term off the derivation's own write terms
    (`facts["out_terms"]`): each written tensor at the width it was priced
    at — a launch that writes its output quantised at the narrow width — or,
    where the launch rewrote only a region of what that port publishes, the
    elements it wrote there."""
    def _wrap(text: str) -> str:
        return f"({text})" if any(c in text for c in "+-/") else text

    syms = facts.get("shape_out_syms") or []
    vals = list(shape_out_vals or ())
    terms_sym: List[str] = []
    terms_val: List[str] = []
    for index, (width, elements) in enumerate(facts["out_terms"]):
        dt_sym = "dt_a" if width == facts["out_dt"] else _num(width)
        if elements is not None:
            # A fused boundary publishing more than its launch wrote: the
            # term is the region, as it is for a read the cache law took
            # over. The published shape would state bytes nobody moved.
            if not elements or not width:
                continue
            terms_sym.append(f"rows_out{index}*{dt_sym}")
            terms_val.append(f"{_num(elements)}*{_num(width)}")
            continue
        sym = syms[index] if index < len(syms) else None
        if not sym or not width:
            continue
        val = vals[index] if index < len(vals) and vals[index] else sym
        terms_sym.append(f"{_wrap(sym)}*{dt_sym}")
        terms_val.append(f"{_wrap(val)}*{_num(width)}")
    return " + ".join(terms_sym), " + ".join(terms_val)


def _access_bytes_formula_for_row(
    facts: Optional[Mapping[str, Any]], *, cnt: int, result_value: int,
    shape_in_vals: Optional[Sequence[Optional[str]]] = None,
    shape_out_vals: Optional[Sequence[Optional[str]]] = None,
    stored_read: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the `{symbolic, substituted, per_layer_result, cnt, result}`
    tooltip for a per-op row's `access_bytes` cell, from the leaf's
    `access_display_facts` and the point's shape values (`a*b*c`, each axis
    exact).

    The symbolic lines and values come straight from the per-rank forward-
    transform node (`params_val`, `shape_in_vals`, `shape_out_vals` already
    substitute per-rank dims). So each component renders `symbolic (full
    dims) = per-rank values × dtype` directly — no `/ div`, no shard-symbol
    rewrite; the per-rank geometry is already in the numbers. Column- vs
    row-parallel replication falls out naturally (the replicated term's
    per-rank value equals its full value).

    Per-layer roll-up (`× cnt`) is emitted as a separate stage so the
    reader sees how a single leaf's bytes get multiplied across the block
    stack.
    """
    if facts is None:
        return {"symbolic": "", "substituted": "", "result": _fmt_si(result_value, " B")}
    def _fmt_dt(b: float) -> str:
        if b <= 0:
            return "?"
        if b == int(b):
            return f"{int(b)}"
        return f"{b:g}"

    def _wrap(s: str) -> str:
        return f"({s})" if any(c in s for c in "+-") else s

    parts_sym: List[str] = []
    parts_val: List[str] = []
    if stored_read is not None:
        # The row's stored read is its saturating law's —
        # `min(rows x read_per_row, weights)`, the weights and their scales
        # (`RowRoofline.stored_read`) — not the leaf's weights once: one
        # layer's share of what the law read.
        if stored_read:
            parts_sym.append("min(rows*w_row, weights)")
            parts_val.append(_num(stored_read / max(int(cnt or 1), 1)))
    elif facts["weight"]:
        if facts["params_sym"]:
            s = f"{_wrap(facts['params_sym'])}*dt_w"
            v = (f"{_wrap(facts['params_val'] or facts['params_sym'])}"
                 f"*{_fmt_dt(facts['weight_dtype_bytes'])}")
        else:
            s = "weights"
            v = f"{facts['weight_bytes']:,d}"
        parts_sym.append(s)
        parts_val.append(v)
    if facts["act_in"] or facts.get("kv_bytes"):
        s, v = act_in_from_terms(facts, shape_in_vals, shape_out_vals)
        if s:
            parts_sym.append(s)
            parts_val.append(v)
    if facts["act_out"]:
        s, v = act_out_from_terms(facts, shape_out_vals)
        if s:
            parts_sym.append(s)
            parts_val.append(v)
    if facts.get("convert_bytes"):
        # The quantised copy of the input operand the kernel needs and the
        # edge does not carry: written, then read back (`Traffic.convert_bytes`).
        parts_sym.append("convert")
        parts_val.append(_num(facts["convert_bytes"]))
    if facts["metadata"] and stored_read is None:
        s = "metadata"
        v = f"{facts['metadata_bytes']:,d}"
        parts_sym.append(s)
        parts_val.append(v)
    # Fold adjacent identical terms (act_in and act_out often share one
    # shape+dtype): `X + X` reads as `2*X`. Only an exact match on BOTH the
    # symbolic and the value side folds — a replicated input next to a
    # sharded output stays two terms.
    merged_sym: List[str] = []
    merged_val: List[str] = []
    pairs = list(zip(parts_sym, parts_val, strict=True))
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1] == pairs[i]:
            j += 1
        run = j - i + 1
        s, v = pairs[i]
        merged_sym.append(f"{run}*{_wrap(s)}" if run > 1 else s)
        merged_val.append(f"{run}*{_wrap(v)}" if run > 1 else v)
        i = j + 1
    sym = " + ".join(merged_sym) or "0"
    val = " + ".join(merged_val) or "0"
    # Per-layer subtotal = full result / cnt. Pulled out as a separate
    # stage so the formula reads as `(weights + act_in + act_out) = X per layer`
    # then `× n_layers = step total` instead of cramming everything onto
    # one line with ambiguous precedence.
    per_layer_value = result_value // max(int(cnt or 1), 1) if cnt else result_value
    out: Dict[str, Any] = {
        "symbolic": sym,
        "substituted": val,
        "result": _fmt_si(result_value, " B"),
    }
    dt_glossary = " · ".join(
        f"{dt} = {label}"
        for dt, label in (("dt_w", "weight dtype bytes"),
                          ("dt_a", "activation dtype bytes"),
                          ("dt_s", "scale dtype bytes"))
        if dt in sym
    )
    if dt_glossary:
        out["note"] = dt_glossary
    if cnt and cnt != 1:
        out["per_layer_result"] = _fmt_si(per_layer_value, " B")
        out["cnt"] = int(cnt)
        out["cnt_unit"] = "layers"
    return out


def _access_breakdown_note(
    *,
    weight_bytes_step_pr: int = 0,
    act_in_bytes_step_pr: int = 0,
    act_out_bytes_step_pr: int = 0,
    meta_bytes_step_pr: int = 0,
) -> str:
    """`access_bytes = weights=X + act_in=Y + …` — the decomposition of the
    figure t_memory divides, so the note's parts sum to the substituted
    line's numerator. "" when every part is zero."""
    parts = []
    if weight_bytes_step_pr:
        parts.append(f"weights={_fmt_si(weight_bytes_step_pr, 'B')}")
    if act_in_bytes_step_pr:
        parts.append(f"act_in={_fmt_si(act_in_bytes_step_pr, 'B')}")
    if act_out_bytes_step_pr:
        parts.append(f"act_out={_fmt_si(act_out_bytes_step_pr, 'B')}")
    if meta_bytes_step_pr:
        parts.append(f"metadata={_fmt_si(meta_bytes_step_pr, 'B')}")
    return "access_bytes = " + " + ".join(parts) if parts else ""


def _t_compute_formula_for_row(
    *,
    t_compute_s: float,
    flops_per_rank: int,
    peak_flops_per_sec: Optional[float],
    act_dtype: str = "",
) -> Optional[Dict[str, Any]]:
    """The t_compute cell's popover: `FLOPs / peak_flops`, divided by the peak
    THIS leaf's time was priced at (`peak_flops_per_sec_for_op` — an FP8
    expert leaf's, not the dashboard's dominant dtype's), so the substituted
    line reproduces the cell. A routed leaf's FLOPs are the busiest rank's
    already (`layout.physical`). None where the row runs no compute."""
    pf = peak_flops_per_sec or 0.0
    if not t_compute_s or t_compute_s <= 0 or not pf:
        return None
    note = f"peak {_fmt_si(int(pf), 'FLOP/s')}"
    if act_dtype:
        note += f" ({act_dtype})"
    return {
        "symbolic": "t_compute = FLOPs / peak_flops",
        "substituted": f"{flops_per_rank:,d} / {pf:.3g}",
        "result": _fmt_time(t_compute_s),
        "note": note,
    }


def _t_memory_formula_for_row(
    *,
    t_memory_s: float,
    access_per_rank: int,
    peak_hbm_bw_bytes_per_sec: Optional[float],
    weight_bytes_step_pr: int = 0,
    act_in_bytes_step_pr: int = 0,
    act_out_bytes_step_pr: int = 0,
    meta_bytes_step_pr: int = 0,
) -> Optional[Dict[str, Any]]:
    """The t_memory cell's popover: `access_bytes / peak_bw`, with the same
    access decomposition t_op's note carries so the divided figure reads as
    the sum of its parts. None where the row moves nothing."""
    pb = peak_hbm_bw_bytes_per_sec or 0.0
    if not t_memory_s or t_memory_s <= 0 or not pb:
        return None
    note = _access_breakdown_note(
        weight_bytes_step_pr=weight_bytes_step_pr,
        act_in_bytes_step_pr=act_in_bytes_step_pr,
        act_out_bytes_step_pr=act_out_bytes_step_pr,
        meta_bytes_step_pr=meta_bytes_step_pr)
    return {
        "symbolic": "t_memory = access_bytes / peak_bw",
        "substituted": f"{access_per_rank:,d} / {pb:.3g}",
        "result": _fmt_time(t_memory_s),
        "note": note,
    }


def _t_op_formula_for_row(
    *,
    t_compute_s: float, t_memory_s: float,
    flops_per_rank: int, access_per_rank: int,
    peak_flops_per_sec: Optional[float],
    peak_hbm_bw_bytes_per_sec: Optional[float],
    weight_bytes_step_pr: int = 0,
    act_in_bytes_step_pr: int = 0,
    act_out_bytes_step_pr: int = 0,
    meta_bytes_step_pr: int = 0,
    t_comm_s: float = 0.0,
    comm_bytes_per_rank: float = 0.0,
    comm_bw_bytes_per_sec: float = 0.0,
    comm_bw_label: str = "",
    comm_bytes_formula: Optional[Dict[str, Any]] = None,
    t_launch_s: float = 0.0,
    launch_floor_s: float = 0.0,
    cnt: int = 1,
) -> Dict[str, Any]:
    """Build the `t_op` tooltip explaining which roofline term wins.

    Four-way roofline (`row_roofline.BOUNDS`): max(t_compute, t_memory,
    t_comm, t_launch). A term a row does not have is dropped from the
    formula: a row with no collective has no t_comm (a pure
    compute/memory row collapses to the two-way max plus its launch
    floor), and a collective has no t_launch — its floor is its own link
    latency, inside t_comm. A row that launches a kernel adds the launch
    floor, `n x launch_floor`: a step of tiny kernels is launch-bound
    however few bytes they move. For fused-collective rows (e.g.
    allreduce_fused_rmsnorm) the comm term typically dominates and the
    result must reflect that — without this the cell value (= true t_op,
    the max of every term) silently disagreed with the tooltip's stated
    result (= two-way max).

    The `note` field carries the access_bytes breakdown
    (weights + act_in + act_out + metadata) so users see what dominates
    memory traffic — surfaced here instead of as a separate column to
    keep the breakdown table compact. When t_comm is in play, the comm
    derivation (per-rank bytes + chosen fabric) is appended too so
    the popover is self-explanatory.
    """
    pf = peak_flops_per_sec or 0.0
    pb = peak_hbm_bw_bytes_per_sec or 0.0
    has_comm = bool(t_comm_s and t_comm_s > 0)
    # Pick the dominating term — surface it as the "{x}-bound" label.
    terms = {"compute": t_compute_s, "memory": t_memory_s}
    if has_comm:
        terms["comm"] = t_comm_s
    has_launch = bool(t_launch_s and t_launch_s > 0)
    if has_launch:
        terms["launch"] = t_launch_s
    bound = max(terms.items(), key=lambda kv: kv[1])[0]
    # Note line uses per-step component fields (cnt baked in, post-
    # MoE-saturation) so that `weights + act + metadata` sums to the
    # exact `access_bytes` value the substituted line divides by peak_bw.
    # Before this scale fix, note showed per-call `ab.X` while substituted
    # used per-step `access_per_rank` — for a leaf with cnt=58 the
    # substituted said `12,486,039,552 B` while the note added up to
    # ≈ 215 MB. Visually contradictory; user reported "公式和结果不等".
    note_parts: List[str] = []
    access_note = _access_breakdown_note(
        weight_bytes_step_pr=weight_bytes_step_pr,
        act_in_bytes_step_pr=act_in_bytes_step_pr,
        act_out_bytes_step_pr=act_out_bytes_step_pr,
        meta_bytes_step_pr=meta_bytes_step_pr)
    if access_note:
        note_parts.append(access_note)
    if has_comm and comm_bytes_formula and comm_bytes_formula.get("substituted"):
        note_parts.append(
            f"comm bytes_per_rank = {comm_bytes_formula['substituted']}"
        )
    sym = "max(FLOPs / peak_flops, access_bytes / peak_bw"
    sub = (
        f"max({flops_per_rank:,d} / {pf:.3g}, "
        f"{access_per_rank:,d} / {pb:.3g}"
    )
    if has_comm:
        sym += ", comm_bytes / comm_bw"
        fabric = f" ({comm_bw_label})" if comm_bw_label else ""
        sub += (f", {comm_bytes_per_rank:,.0f} / "
                f"{comm_bw_bytes_per_sec:.3g}{fabric}")
    if has_launch:
        sym += ", n × launch_floor"
        sub += f", {int(cnt):,d} × {_fmt_time(launch_floor_s)}"
    sym += ")"
    sub += f") = {bound}-bound"
    return {
        "symbolic": sym,
        "substituted": sub,
        "result": _fmt_time(max(t_compute_s, t_memory_s, t_comm_s or 0.0,
                                t_launch_s or 0.0)),
        "note": "  ·  ".join(note_parts),
    }


def _t_comm_formula_for_row(
    *,
    t_comm_s: float,
    comm_bytes_per_rank: float,
    comm_bw_bytes_per_sec: float,
    comm_bw_label: str,
    comm_world: int,
    comm_bytes_formula: Optional[Dict[str, Any]],
    comm_phases: List[Dict[str, Any]],
    cnt: int,
    link_latency_s: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """The `t_comm` tooltip of a collective row: `bytes / bandwidth
    (fabric) + n × hops × link_latency = time`, and for a hierarchical
    collective (intra RS + inter AR + intra AG) one subterm per phase, each
    at one invocation (`cnt` invocations make the row) with its own hops.
    The hops are each phase's, by the algorithm that runs it (`algorithm`
    and `hops` on the phase rows `price_collective` returns); the note names
    a one-phase collective's algorithm. `link_latency_s` is one
    traversal's.
    None for a row that moves nothing."""
    if not comm_bw_label or not comm_bw_bytes_per_sec:
        return None
    if comm_bytes_per_rank <= 0 and t_comm_s <= 0:
        return None

    def bw(v: float) -> str:
        return f"{_fmt_si(v, 'B')}/s" if v and v > 0 else "bandwidth"

    note = f"world={comm_world}" if comm_world else None
    if note and len(comm_phases) == 1 and comm_phases[0].get("algorithm"):
        note += f", {comm_phases[0]['algorithm'].replace('_', '-')}"
    per = max(int(cnt or 1), 1)
    out: Dict[str, Any]
    if len(comm_phases) > 1:
        out = {
            "symbolic": "t_comm = " + " + ".join(
                f"t_{p.get('name') or 'phase'}" for p in comm_phases),
            "substituted": "  +  ".join(
                f"({p.get('substituted') or 'bytes'}) / "
                f"{bw(p.get('bw_bytes_per_sec') or 0)}" for p in comm_phases),
            "result": _fmt_time(t_comm_s),
            "subterms": [{
                "label": (f"{p.get('name') or 'phase'} "
                          f"({p.get('fabric_label') or p.get('fabric')}, "
                          f"world={p.get('world')})"),
                "symbolic": (f"({p.get('symbolic') or 'bytes'}) / bandwidth"
                             + (" + hops × link_latency" if link_latency_s else "")),
                "substituted": (f"({p.get('substituted') or 'bytes'}) / "
                                f"{bw(p.get('bw_bytes_per_sec') or 0)}"
                                + (f" + {p.get('hops', 0)} × {_fmt_time(link_latency_s)}"
                                   if link_latency_s else "")),
                "result": (
                    f"{_fmt_si((p.get('bytes_per_rank') or 0) / per, ' B')} / "
                    f"{bw(p.get('bw_bytes_per_sec') or 0)}"
                    + (f" + {_fmt_time(p['latency_s'] / per)}"
                       if p.get("latency_s") else "")
                    + f" = {_fmt_time((p.get('t_phase_s') or 0) / per)}"),
            } for p in comm_phases],
        }
    elif comm_bytes_formula and comm_bytes_formula.get("symbolic"):
        out = {
            "symbolic": f"t_comm = ({comm_bytes_formula['symbolic']}) / bandwidth",
            "substituted": (f"({comm_bytes_formula.get('substituted')}) / "
                            f"{bw(comm_bw_bytes_per_sec)} ({comm_bw_label})"),
            "result": _fmt_time(t_comm_s),
        }
    else:
        out = {
            "symbolic": "t_comm = bytes_per_rank / bandwidth",
            "substituted": (f"{_fmt_si(comm_bytes_per_rank, ' B')} / "
                            f"{bw(comm_bw_bytes_per_sec)} ({comm_bw_label})"),
            "result": _fmt_time(t_comm_s),
        }
    hops = sum(int(p.get("hops") or 0) for p in comm_phases)
    if link_latency_s > 0 and hops:
        # The fixed part: every instance waits on its schedule's hops, each a
        # link latency (`comm_cost.collective_latency`).
        out["symbolic"] += " + n × hops × link_latency"
        out["substituted"] += (f"  +  {per} × {hops} × "
                               f"{_fmt_time(link_latency_s)}")
    if note:
        out["note"] = note
    return out


def _collective_bytes_formula(
    *, op: str, world: int, tensor_bytes: float, cnt: int,
    phases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Per-rank bytes derivation for a comm row, returned as the
    `{symbolic, substituted}` pair the frontend popover renders as its
    primary formula line.

    Mirrors `comm_cost.collective_per_rank_bytes` literally so the
    tooltip explains exactly what the timing pass divided by bandwidth:
        all_reduce      → 2·(N-1)/N · T  (= RS + AG)
        reduce_scatter  →   (N-1)/N · T
        all_gather      →   (N-1)/N · T
        all_to_all      →   (N-1)/N · T

    When `phases` is multi-entry (hierarchical AR), both lines sum the
    phases term-for-term, each prefixed with its collective primitive so
    BOTH levels show: `reduce_scatter [(N_intra-1)/N_intra * T] + ...`
    symbolic, `reduce_scatter [7/8 * T] + ...` substituted. Per-phase fabric
    bandwidth and time decomposition live in `comm_phases` for the
    frontend subterms.

    `cnt` is the per-step layer-count multiplier — folded in only when
    > 1 so single-occurrence comms read cleanly. Empty op / N≤1 / 0 B
    short-circuits to `{}` so callers can branch on truthiness without
    having to discriminate a sentinel string.
    """
    n = int(world or 0)
    if n <= 1 or tensor_bytes <= 0 or not op:
        return {}
    if phases:
        # Compose the formula from the phases `collective_phases` already
        # built (T expanded to its axis·dtype product at the source). Single
        # collective → that phase's formula verbatim (the row label already
        # names the op). Hierarchical AR → sum the phases, each prefixed
        # with its primitive so BOTH levels show:
        #   reduce_scatter
        #   [(N_intra-1)/N_intra * (B·s_ctx·d_model·dt_a)] + all_reduce [...]
        # Generic — reads each phase's own name/symbolic/substituted, no
        # per-op branching, no duplicated coefficient logic.
        if len(phases) == 1:
            sym = phases[0]["symbolic"]
            sub = phases[0]["substituted"]
        else:
            sym = " + ".join(f"{ph['name']} [{ph['symbolic']}]" for ph in phases)
            sub = " + ".join(f"{ph['name']} [{ph['substituted']}]" for ph in phases)
        out: Dict[str, Any] = {"symbolic": sym, "substituted": sub}
        if cnt and cnt != 1:
            total_per_layer_bytes = int(sum(
                ph["bytes_per_rank"] for ph in phases
            ) / cnt)
            out["per_layer_result"] = _fmt_si(total_per_layer_bytes, " B")
            out["cnt"] = int(cnt)
            out["cnt_unit"] = "layers"
        return out
    # Fallback when no phase breakdown was supplied (degenerate caller):
    # the bare coefficient·T form.
    coeff_sym = ring_factor_symbolic(op)
    coeff_sub = ring_factor_symbolic(op, n)
    bytes_str = _fmt_si(int(tensor_bytes), " B")
    # Per-layer per-rank collective bytes; the layer multiplier `× cnt`
    # is split off as a separate popover stage so the formula reads
    # `(N-1)/N * T  → × n_layers  → step total` instead of cramming the
    # layer rollup into the same line as the coefficient × tensor product.
    sym = f"{coeff_sym} * T"
    sub = f"{coeff_sub} * {bytes_str}"
    out_single: Dict[str, Any] = {"symbolic": sym, "substituted": sub}
    if cnt and cnt != 1:
        coeff_num = ring_factor(op, n)
        per_layer_bytes = int(coeff_num * float(tensor_bytes))
        out_single["per_layer_result"] = _fmt_si(per_layer_bytes, " B")
        out_single["cnt"] = int(cnt)
        out_single["cnt_unit"] = "layers"
    return out_single


#: A step's SOL time as its popovers write it: every leaf's four-term
#: roofline (`row_roofline.BOUNDS`), summed — rows serial, no overlap.
STEP_SOL_LAW = "Σ_leaf max(t_compute, t_memory, t_comm, t_launch)"
#: What a step's per-bound sums are, beside its SOL time.
STEP_SOL_BOUND_NOTE = ("per-bound sums, not addends — each leaf counts only "
                       "its winning bound")


def phase_roofline_formulas(
    *,
    flops: Dict[str, str],
    flops_per_step: float,
    mem_read: float,
    mem_write: float,
    t_sol: Optional[float] = None,
    t_compute: Optional[float] = None,
    t_memory: Optional[float] = None,
    t_comm: Optional[float] = None,
    unpublished_reason: str = "",
    t_sol_context: str = "",
) -> Dict[str, Any]:
    """The phase table's `formulas` object — display, not the times.

    When `unpublished_reason` is set (pp>1, or the deployment admits no
    request) the t_sol popover explains the blank cell; the counts
    beside it are unchanged. `t_sol_context` leads the t_sol note when the
    published time has a basis the caller must state (pp: one microbatch's
    latency across stages).
    """
    out: Dict[str, Any] = {
        "flops": flops,
        "mem_read": {
            "symbolic": "Σ leaf (weight + metadata + act_in)  (per-op "
                        "breakdown)",
            "substituted": f"= {int(mem_read):,d} bytes",
        },
        "mem_write": {
            "symbolic": "Σ leaf.act_out  (per-op breakdown)",
            "substituted": f"= {int(mem_write):,d} bytes",
        },
        "arithmetic_intensity": intensity_formula(
            flops_per_step, mem_read + mem_write),
    }
    if unpublished_reason:
        out["t_sol"] = {
            "symbolic": "not published",
            "substituted": "—",
            "note": unpublished_reason,
        }
        return out
    out["t_sol"] = {
        "symbolic": STEP_SOL_LAW,
        "substituted": f"= {t_sol:.6g} s",
        "note": (
            (f"{t_sol_context} · " if t_sol_context else "")
            + f"{STEP_SOL_BOUND_NOTE}: Σt_compute={t_compute:.6g}s · "
            f"Σt_memory={t_memory:.6g}s · Σt_comm={t_comm:.6g}s "
            f"(sums exclude aux stacks: MTP/vision/audio)"
        ),
    }
    return out


def attach_parent_formulas(
    rows: List[Dict[str, Any]],
    *,
    value_key: str,
    formula_key: str,
    format_value: Optional[Any] = None,
) -> None:
    """Fill formula popovers on parent rows as Σ immediate children.

    Leaves keep the per-op formula they already have. A parent that
    already has `formula_key` is left alone. This does not add rows:
    invariants that sum `depth == 0` still see the same numbers.
    """
    fmt = format_value or (lambda n: _fmt_si(int(n), ""))
    n = len(rows)
    for i, row in enumerate(rows):
        if row.get(formula_key):
            continue
        parent_depth = int(row.get("depth") or 0)
        if i + 1 >= n or int(rows[i + 1].get("depth") or 0) <= parent_depth:
            continue
        kids: List[Dict[str, Any]] = []
        for nxt in rows[i + 1:]:
            child_depth = int(nxt.get("depth") or 0)
            if child_depth <= parent_depth:
                break
            if child_depth == parent_depth + 1:
                kids.append(nxt)
        parts: List[tuple] = []
        total = 0.0
        for child in kids:
            value = float(child.get(value_key) or 0)
            if value <= 0:
                continue
            parts.append((child.get("name") or child.get("op") or "?", value))
            total += value
        if not parts:
            continue
        shown, hidden = _finalize_parts(parts)
        entry = {
            "symbolic": (
                f"Σ {len(parts)} component"
                + ("s" if len(parts) != 1 else "")
            ),
            "substituted": _substituted_sum(shown, total, fmt),
            "result": fmt(total),
        }
        if hidden:
            entry["note"] = f"+ {hidden} smaller terms"
        row[formula_key] = entry


def _phase_top_components(
    rows: List[Dict[str, Any]], value_key: str, fmt: Any,
) -> List[tuple]:
    """The fold's depth-0 rows as `(name, value)` — what a phase-level Σ
    actually sums."""
    parts: List[tuple] = []
    for row in rows:
        if int(row.get("depth") or 0) != 0:
            continue
        v = float(row.get(value_key) or 0)
        if v <= 0:
            continue
        name = (row.get("template_name") or row.get("name")
                or row.get("op") or "?")
        parts.append((name, v))
    return parts


def attach_phase_component_formulas(
    phase: Dict[str, Any], rows: List[Dict[str, Any]],
) -> None:
    """Expand the phase-level Σ placeholders (the Summary rows) into the
    fold's depth-0 component sum, so the FLOPs / t_SOL popovers
    read like the module rows' own `Σ components` popovers instead of a bare
    `Σ …` with only a total. mem_read / mem_write are NOT expanded: a per-op
    row's access bytes are read+write combined and the fold carries no
    act_out column, so a tree sum could not reconcile with the phase's reads
    figure — those two keep their total-only substituted line."""
    formulas = phase.get("formulas") or {}
    # The fold's depth-0 rows are the phase's terms: their per-rank FLOPs
    # and rolled-up times add up to the step's (the draft is a row of the
    # fold). A pipelined step's time is every stage's plus the hops
    # (`pipeline_time`) while the fold is the bound stage's rows — that
    # total has no expansion over this fold, so it keeps its own line.
    spec = [("flops", "flops_per_rank", phase.get("flops_per_step"),
             lambda n: _fmt_si(int(n), ""))]
    if not phase.get("pipeline_time"):
        spec.append(("t_sol", "t_op_agg", phase.get("t_sol_s"), _fmt_time))
    for key, value_key, total, fmt in spec:
        entry = formulas.get(key)
        if not isinstance(entry, dict) or not entry:
            continue
        if total is None or not total > 0:
            continue
        parts = _phase_top_components(rows, value_key, fmt)
        if not parts:
            continue
        shown, hidden = _finalize_parts(parts)
        entry["substituted"] = _substituted_sum(shown, float(total), fmt)
        entry["result"] = fmt(float(total))
        if hidden:
            prior = entry.get("note")
            entry["note"] = " · ".join(
                ([prior] if prior else []) + [f"+ {hidden} smaller terms"])


def _rollup_column(
    rows: List[Dict[str, Any]], self_key: str, agg_key: str,
) -> None:
    """Each row's `agg_key` is its `self_key` plus every descendant's.

    A row that does not PUBLISH `self_key` is not a row that publishes zero,
    and the difference is load-bearing here. `batch_sweep` drops the fields
    it cannot re-read at another batch rather than carrying the walked
    batch's value, so a row can legitimately arrive without one. The
    motivating case is `comm_bytes_per_rank` on the one collective whose
    payload is not its typed flow: an all-gather widens an axis, the walk
    states no shape law for it (`phases._walked`), and the per-rank volume
    is the ring factor times a payload that has no law at this batch — so
    there is nothing to state. Summing that as 0.0 put a zero communication
    total beside a non-zero `t_comm_s`, which is the same
    fabricated-plausible-number failure as pricing an unknown width at 0
    bytes.

    So an absent value makes the aggregate absent (`None`) for that row and
    every ancestor of it: a subtree total that silently omits one of its
    children is worse than a subtree total that says it cannot be computed.
    """
    for row in rows:
        row[agg_key] = 0.0
    unpublished: set = set()
    stack: List[tuple] = []
    for i, row in enumerate(rows):
        depth = int(row.get("depth") or 0)
        while stack and stack[-1][1] >= depth:
            stack.pop()
        if self_key in row and row.get(self_key) is None:
            # This row and every ancestor can no longer state a total.
            unpublished.add(i)
            unpublished.update(idx for idx, _ in stack)
            stack.append((i, depth))
            continue
        value = float(row.get(self_key) or 0)
        if value > 0:
            row[agg_key] += value
            for idx, _ in stack:
                rows[idx][agg_key] += value
        stack.append((i, depth))
    for i in unpublished:
        rows[i][agg_key] = None


def table_total(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """What a per-op table's rows add up to — its Total row's work and
    theory columns.

    The rows are a flat per-op list or the module fold over it
    (`physical_module_rows`): either way each op's own cost sits on exactly
    one row, so times and bytes add over every row. A fold parent's FLOPs
    are its subtree's sum, so FLOPs add over the top level (a flat list is
    all top level). Time is the rows' `t_op_s`, the same column a phase's
    `t_sol_s` sums. The measured total is not among them: it is the one
    measured formula's (`auto_sweep.MeasuredTrack`, stated on the phase),
    all or nothing, never a sum of rows made here."""
    rows = list(rows)
    timed = [r for r in rows if r.get("t_op_s") is not None]
    # A priced row always states its collective bytes (0 for a kernel that
    # moves none). One that does not is not a row that moves zero bytes, so
    # the column has no total. The fold's rollups go unpublished the same
    # way (`phase_formulas._rollup_column`).
    comm_unpublished = any(r.get("comm_bytes_per_rank") is None for r in timed)
    return {
        "flops_per_rank": sum(int(r.get("flops_per_rank") or 0)
                              for r in rows
                              if int(r.get("depth") or 0) == 0),
        "access_bytes_per_rank": sum(
            float(r.get("access_bytes_per_rank") or 0.0) for r in rows),
        "comm_bytes_per_rank": (None if comm_unpublished else sum(
            float(r.get("comm_bytes_per_rank") or 0.0) for r in rows)),
        "t_sol_s": (sum(float(r["t_op_s"]) for r in timed)
                    if timed else None),
    }


def enrich_module_breakdown(phase: Dict[str, Any]) -> None:
    """Roll up the per-op table's time/bytes columns and fill parent
    popovers for t_op / access / comm.

    The rows arrive complete: `module_rows.physical_module_rows` walks the
    grafted + fused tree, so collectives and fused kernels are native rows
    and timing is already merged by uid. What remains here is structural —
    subtree rollups and parent formulas read any row list. No row carries
    both a parent and its own double-count: the tree IS the accounting.
    """
    rows = phase.get("flops_by_module") or []
    # The table's Total row, summed once here over the same rows the table
    # draws — the page reads it instead of adding its own columns.
    phase["op_table_total"] = table_total(rows)
    if not rows:
        return
    _rollup_column(rows, "t_op_s", "t_op_agg")
    _rollup_column(rows, "access_bytes_per_rank", "access_bytes_agg")
    _rollup_column(rows, "comm_bytes_per_rank", "comm_bytes_agg")
    # Subtree rollups for the roofline's component columns (compute,
    # memory, comm; a row's launch floor stays its own), so aggregator rows
    # (self_attn / mlp / template headers) can show what their descendants
    # exercise instead of a blank self value. The measured share is not summed
    # here: it is the one measured formula's (`batch_sweep._measured_table`).
    _rollup_column(rows, "t_compute_s", "t_compute_agg")
    _rollup_column(rows, "t_memory_s", "t_memory_agg")
    _rollup_column(rows, "t_comm_s", "t_comm_agg")
    bytes_fmt = lambda n: _fmt_si(int(n), "B")
    value_key = (
        "flops_per_rank"
        if any(r.get("flops_per_rank") is not None for r in rows)
        else "flops_full"
    )
    # Each parent popover as `(column it sums, popover, format)`.
    popovers = ((value_key, "flops_formula", None),
                ("t_op_agg", "t_op_formula", _fmt_time),
                ("access_bytes_agg", "access_bytes_formula", bytes_fmt),
                ("comm_bytes_agg", "comm_bytes_formula", bytes_fmt),
                ("t_compute_agg", "t_compute_formula", _fmt_time),
                ("t_memory_agg", "t_memory_formula", _fmt_time))
    n = len(rows)
    for i, row in enumerate(rows):
        # Arithmetic intensity of what the row shows: its (subtree) FLOPs
        # over its (subtree) access bytes. A parent has no winning bound —
        # its subtree's leaves each have their own — so it publishes none,
        # and its popovers are its children's sums, written below: one the
        # row carries from an earlier enrich (a re-read at another batch)
        # is that enrich's.
        row["arithmetic_intensity"] = arithmetic_intensity(
            row.get("flops_per_rank") or 0, row["access_bytes_agg"])
        row["arithmetic_intensity_formula"] = intensity_formula(
            row.get("flops_per_rank") or 0, row["access_bytes_agg"])
        if i + 1 < n and int(rows[i + 1].get("depth") or 0) > int(
                row.get("depth") or 0):
            for _, formula_key, _ in popovers:
                row.pop(formula_key, None)
            row["bound"] = None
    for column, formula_key, fmt in popovers:
        attach_parent_formulas(rows, value_key=column, formula_key=formula_key,
                               format_value=fmt)
    attach_phase_component_formulas(phase, rows)


def _fmt_si(n: int, suffix: str = "") -> str:
    """Compact SI-prefixed display of an integer (e.g. `1.23 GB`)."""
    extra = suffix.strip()
    # Letter units concatenate onto the prefix (`1.23 KB`); words keep a
    # space (`1.23 K FLOPs`).
    letter = bool(extra) and extra.isalpha() and extra.isupper() and len(extra) <= 2
    if not n:
        return f"0 {extra}" if extra else "0"
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            body = f"{n / scale:.2f} {unit}"
            if not extra:
                return body
            return f"{body}{extra}" if letter else f"{body} {extra}"
    return f"{n} {extra}" if extra else str(n)


def _fmt_time(s: float) -> str:
    if not s or s <= 0:
        return "—"
    if s < 1e-3:
        return f"{s * 1e6:.1f} µs"
    if s < 1.0:
        return f"{s * 1e3:.1f} ms"
    return f"{s:.2f} s"
