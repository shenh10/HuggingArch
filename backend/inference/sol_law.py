"""Canonical SOL laws: one walked formula, every serving point substituted.

This layer is deliberately smaller than an estimate: it carries a phase's
SOL account and time as SymPy laws over workload and card symbols, with the
common subexpressions named once. Per-op rows, module tables, shapes and
display formulas belong to an explicit breakdown request and are not stored
here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import sympy as sp

from backend.arch.law_text import read_walk_law
from backend.arch.eval.analysis import Point
from backend.measure.auto_sweep import (
    _ExactPrinter, exact_div, MeasuredFormula, PointMeasure,
)
from backend.measure.lookup import Key

from .capacity import (
    CapacityKnobs,
    capacity_ceilings,
    _restamp_gpu,
    _capacity_pending,
    _capacity_preparation,
    _point_rows,
    point_served,
    without_step,
)
from .card import Card
from .comm_cost import CollectiveRung, collective_ladder
from .gpu_specs import peak_flops_for_dtype, peak_hbm_bw
from .phases import Speculation, phase_binding, phase_throughput, step_tokens
from .parallel_config import ParallelConfig
from .prefill_geometry import PrefillChunkSchedule, PrefillEndpointGeometry
from backend.measure.measure_matrix import MeasureMatrix
from .row_roofline import FloorForm, StepSum, floor_of, law_symbol, rung_symbols
from .sol_data import walkRecordSchema as servingLawSchema
from .calibration_overlay import point_keys
from .serving_view import servingCapacityView
from backend.code_trace.memo import memoize
from .sweep_grid import (
    FRONTIER_BATCH_GRID,
    FRONTIER_DRAFT_TOKENS,
    GALLERY_SEQ_LENS,
)
from .batch_sweep import point_measures


_lawOutputs = {
    "t_sol_s": "t_op",
}
_MEASURE_POINTS_SCHEMA = "measure-points-v7"

# One draft law serves every speculative K: the request says which draft stack
# exists, while this symbol says how many forwards the point runs.
draftForwardsLaw = sp.Symbol("s_verify") - 1


def _csePreservingExtrema(values: Sequence[sp.Expr]):
    """CSE without asking SymPy to compare the arms of a `Max`/`Min`.

    A roofline extremum is written as-written: its arms can hold `Cases`, so
    SymPy cannot order them until every axis is numeric. CSE rebuilds function
    applications by calling their constructors again, and ``Max(...)``'s
    constructor tries that comparison. An inert application therefore stands in
    for the extremum's *constructor* while CSE runs; its arguments stay visible,
    so the shape and byte subterms repeated by many extrema are still named
    once. The restored graph still prints the law language's as-written
    ``Max`` / ``Min``.
    """
    extrema: dict[sp.Expr, sp.Expr] = {}

    def mask(expr: sp.Expr) -> sp.Expr:
        if not isinstance(expr, sp.Basic) or not expr.args:
            return expr
        args = tuple(mask(arg) for arg in expr.args)
        if isinstance(expr, sp.Max):
            inert = _InertMaximum(*args)
        elif isinstance(expr, sp.Min):
            inert = _InertMinimum(*args)
        else:
            if not args:
                return expr
            with sp.evaluate(False):
                return expr.func(*args)
        extrema[inert] = expr
        return inert

    masked = [mask(value) for value in values]
    definitions, reduced = sp.cse(
        masked, symbols=sp.numbered_symbols("u"), order="canonical", list=True)

    def restore(expr: sp.Expr) -> sp.Expr:
        if expr in extrema:
            return extrema[expr]
        if isinstance(expr, _InertMaximum):
            args = tuple(restore(arg) for arg in expr.args)
            with sp.evaluate(False):
                return sp.Max(*args)
        if isinstance(expr, _InertMinimum):
            args = tuple(restore(arg) for arg in expr.args)
            with sp.evaluate(False):
                return sp.Min(*args)
        if not isinstance(expr, sp.Basic) or not expr.args:
            return expr
        args = tuple(restore(arg) for arg in expr.args)
        if all(new is old for new, old in zip(args, expr.args, strict=True)):
            return expr
        return expr.func(*args)

    return ([(symbol, restore(value)) for symbol, value in definitions],
            [restore(value) for value in reduced])


class _InertMaximum(sp.Function):
    """`Max` during CSE: an application, never a comparison."""

    @classmethod
    def eval(cls, *_args):
        return None


class _InertMinimum(sp.Function):
    """`Min` during CSE: an application, never a comparison."""

    @classmethod
    def eval(cls, *_args):
        return None


def _withoutITE(expr: sp.Expr) -> sp.Expr:
    """Rewrite SymPy's printed ITE as the law language's Piecewise."""
    if not isinstance(expr, sp.Basic) or not expr.args:
        return expr
    old_args = expr.args
    args = [_withoutITE(arg) for arg in old_args]
    if expr.func == sp.ITE:
        return sp.Piecewise((args[1], args[0]), (args[2], sp.true))
    if any(new is not old for new, old in zip(args, old_args, strict=True)):
        return expr.func(*args)
    return expr


def _replaceKeepingShape(expr: sp.Expr, rule: Mapping[sp.Expr, sp.Expr]):
    """Substitute without rebuilding an unevaluated parent eagerly.

    A graph is as-written: `Max` may hold `Cases` that are not orderable until
    a point is numeric. Plain `xreplace` rebuilds changed parents with global
    evaluation on; this walks the tree and constructs changed parents under
    `evaluate(False)`.
    """
    if expr in rule:
        return rule[expr]
    if not isinstance(expr, sp.Basic) or not expr.args:
        return expr
    args = tuple(_replaceKeepingShape(arg, rule) for arg in expr.args)
    if all(new is old for new, old in zip(args, expr.args, strict=True)):
        return expr
    with sp.evaluate(False):
        return expr.func(*args)


@memoize(maxsize=32768)
def _read(text: str, local: frozenset[str]) -> sp.Expr:
    def symbol(name: str) -> sp.Symbol:
        if name in local:
            return sp.Symbol(name)
        return law_symbol(name)

    return read_walk_law(text, symbol=symbol)


@memoize(maxsize=1024)
def _compiledGraph(inputs: tuple[str, ...],
                   definitions: tuple[tuple[str, str], ...],
                   outputs: tuple[tuple[str, str], ...]):
    """One CSE graph as one exact program, compiled once.

    Each definition becomes an assignment to the variable every later
    expression reads. A point no longer interprets the graph node by node;
    it substitutes into the same immutable program.
    """
    localNames: set[str] = set()
    byDefinition: dict[sp.Expr, sp.Expr] = {}
    exprs: list[sp.Expr] = []
    internalNames: set[str] = set()
    for index, (name, text) in enumerate(definitions):
        expr = _replaceKeepingShape(
            _read(text, frozenset(localNames)), byDefinition)
        variable = sp.Symbol(f"_law{index}")
        byDefinition[sp.Symbol(name)] = variable
        internalNames.add(variable.name)
        localNames.add(name)
        exprs.append(expr)
    for _name, text in outputs:
        exprs.append(_replaceKeepingShape(
            _read(text, frozenset(localNames)), byDefinition))
    by_name = {symbol.name: symbol for expr in exprs
               for symbol in expr.free_symbols}
    stray = sorted(set(by_name) - set(inputs) - internalNames)
    if stray:
        raise ValueError(f"laws read {stray}, which the point does not bind "
                         f"(it binds {sorted(inputs)})")
    return sp.lambdify(
        [by_name.get(name, sp.Symbol(name)) for name in inputs],
        list(exprs),
        modules=[{"math": math, "Fraction": Fraction,
                  "_law_div": exact_div}],
        printer=_ExactPrinter,
        cse=_eachLawInOrder)


def _eachLawInOrder(exprs: Sequence[Any]):
    """Assign every graph expression in order, even an atom or a repeat.

    A CSE graph's later expressions read the variables this gives the earlier
    ones; unlike a list of independent laws, an expression cannot be deduped
    away because its variable is itself a fact of the program.
    """
    assigned = []
    out = []
    for index, expr in enumerate(exprs):
        variable = sp.Symbol(f"_law{index}")
        assigned.append((variable, expr))
        out.append(variable)
    return assigned, out


@dataclass(frozen=True)
class LawGraph:
    """A CSE graph: external inputs, local definitions, named outputs."""

    inputs: tuple[str, ...]
    definitions: tuple[tuple[str, str], ...]
    outputs: dict[str, str]

    @classmethod
    def of(cls, exprs: Mapping[str, sp.Expr]) -> "LawGraph":
        values = [_withoutITE(exprs[name]) for name in _lawOutputs]
        replacements, reduced = _csePreservingExtrema(values)
        outputs = {
            name: sp.sstr(value)
            for name, value in zip(_lawOutputs, reduced, strict=True)
        }
        inputs = tuple(sorted(
            {symbol.name for expr in values for symbol in expr.free_symbols}))
        definitions = tuple((symbol.name, sp.sstr(expr))
                            for symbol, expr in replacements)
        return cls(inputs, definitions, outputs)

    def to_json(self) -> dict[str, Any]:
        return {"inputs": list(self.inputs),
                "definitions": [[name, text] for name, text in self.definitions],
                "outputs": self.outputs,
                }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "LawGraph":
        if "program" in value:
            raise ValueError("a law graph carries law text only")
        return cls(tuple(value["inputs"]),
                   tuple((name, text) for name, text in value["definitions"]),
                   dict(value["outputs"]))

    def floorForm(self, point: Mapping[str, Any],
                  floorCard: "FloorCard", name: str = "t_sol_s") -> FloorForm:
        """Output `name` at `point` on a card as a function of the two
        floors (`_FloorTerms`): the one form every floor is substituted
        into."""
        return _floorTerms(self.inputs, self.definitions,
                           self.outputs[name]).form(
            point, floorCard.zero, floorCard.unit, self.inputs)

    def evaluate(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        program = _compiledGraph(
            self.inputs, self.definitions, tuple(self.outputs.items()))
        values = program(*(binding[name] for name in self.inputs))
        firstOutput = len(self.definitions)
        if len(values) != firstOutput + len(self.outputs):
            raise ValueError("a law program does not return its graph outputs")
        return dict(zip(
            self.outputs,
            values[firstOutput:firstOutput + len(self.outputs)],
            strict=True))


@dataclass(frozen=True)
class _FloorTerms:
    """A step law's terms by the floor each waits on, compiled once per law.

    A step's time is `Σ_k w_k x Max(arms_k)` (`StepSum`, `build_row_exprs`):
    each arm is a row's compute, memory, collective or launch time, and
    only the last two read a floor — a collective's fixed time is its link
    traversals times the link latency (`COMM_LATENCY_S_<collective>`), a
    launch is `LAUNCH_S`. This is that sum read off the law's own CSE graph,
    every weight and arm an output of one program over the graph's
    definitions; `FloorForm` is it at one point with the floors left free.
    Nothing here is a second statement of the law: a law whose floors sit
    anywhere but an affine arm of a top-level term is refused."""
    program: Any
    firstOutput: int
    #: Floor-free terms: (weight output, value output).
    constants: tuple[tuple[int, int], ...]
    #: Per max term: its weight output and per arm (output, floor or None).
    terms: tuple[tuple[int, tuple[tuple[int, Optional[str]], ...]], ...]

    def form(self, point: Mapping[str, Any], zero: Mapping[str, float],
             unit: Mapping[str, float], inputs: Sequence[str]) -> FloorForm:
        """The step's `FloorForm` at `point` on the card `zero` / `unit`
        bind with both floors 0 and 1 s: each arm read at the two (it is
        affine in its one floor), weighted."""
        at0 = self.program(*({**point, **zero}[n] for n in inputs))
        at1 = self.program(*({**point, **unit}[n] for n in inputs))
        base = self.firstOutput

        def read(values: Sequence[Any], index: int) -> float:
            return float(values[base + index])

        constant = 0.0
        for weight, value in self.constants:
            constant += read(at0, weight) * read(at0, value)
        terms: list[tuple[float, float, float, float]] = []
        for weightIndex, arms in self.terms:
            w = read(at0, weightIndex)
            if w < 0:
                raise ValueError(
                    f"a step term is weighted {w}: a max of its arms is not "
                    "the arms' max weighted")
            a = b = c = n = 0.0
            for index, floor in arms:
                v0, v1 = read(at0, index), read(at1, index)
                if floor is None:
                    a = max(a, w * v0)
                elif floor == "latency":
                    b, c = w * v0, w * (v1 - v0)
                else:
                    if v0 != 0:
                        raise ValueError(
                            f"a launch arm is {v0} s with no launch floor")
                    n = w * v1
            if c == 0 and n == 0:
                constant += max(a, b)
            else:
                terms.append((a, b, c, n))
        return FloorForm.collapsed(constant, terms)


@memoize(maxsize=1024)
def _floorTerms(inputs: tuple[str, ...],
                definitions: tuple[tuple[str, str], ...],
                output: str) -> _FloorTerms:
    """`output`'s law split into `_FloorTerms` (its docstring)."""
    exprs: dict[str, sp.Expr] = {}
    local: set[str] = set()
    for name, text in definitions:
        exprs[name] = _read(text, frozenset(local))
        local.add(name)
    floorsOf: dict[str, frozenset[str]] = {}

    def floors(expr: sp.Expr) -> frozenset[str]:
        out: set[str] = set()
        for symbol in expr.free_symbols:
            if symbol.name in exprs:
                if symbol.name not in floorsOf:
                    floorsOf[symbol.name] = floors(exprs[symbol.name])
                out |= floorsOf[symbol.name]
            else:
                kind = floor_of(symbol.name)
                if kind is not None:
                    out.add(kind)
        return frozenset(out)

    def affine(expr: sp.Expr) -> bool:
        """Whether `expr` is affine in the floors it reads: sums, products
        with one floored factor, cases over floor-free conditions."""
        if not floors(expr):
            return True
        if isinstance(expr, sp.Symbol):
            return (affine(exprs[expr.name]) if expr.name in exprs
                    else floor_of(expr.name) is not None)
        if isinstance(expr, sp.Add):
            return all(affine(arg) for arg in expr.args)
        if isinstance(expr, sp.Mul):
            floored = [arg for arg in expr.args if floors(arg)]
            return len(floored) == 1 and affine(floored[0])
        if isinstance(expr, sp.Piecewise):
            return all(not floors(cond) and affine(value)
                       for value, cond in expr.args)
        return False

    outputs: list[str] = []

    def emit(expr: Any) -> int:
        outputs.append(sp.sstr(expr))
        return len(outputs) - 1

    constants: list[tuple[int, int]] = []
    terms: list[tuple[int, tuple[tuple[int, Optional[str]], ...]]] = []

    def arm(expr: sp.Expr) -> tuple[int, Optional[str]]:
        kinds = floors(expr)
        if len(kinds) > 1 or not affine(expr):
            raise ValueError(
                f"a step term's arm is not affine in one floor: {expr}")
        return emit(expr), next(iter(kinds), None)

    def split(expr: sp.Expr, weight: tuple[sp.Expr, ...]) -> None:
        w = sp.Mul(*weight) if weight else sp.Integer(1)
        if not floors(expr):
            constants.append((emit(w), emit(expr)))
        elif isinstance(expr, sp.Symbol) and expr.name in exprs:
            split(exprs[expr.name], weight)
        elif isinstance(expr, sp.Add):
            for term in expr.args:
                split(term, weight)
        elif isinstance(expr, sp.Mul):
            floored = [arg for arg in expr.args if floors(arg)]
            if len(floored) != 1:
                raise ValueError(
                    f"a step term multiplies {len(floored)} floored factors")
            split(floored[0], (*weight, *(
                arg for arg in expr.args if arg is not floored[0])))
        elif isinstance(expr, sp.Max):
            arms = tuple(arm(a) for a in expr.args)
            if sum(1 for _i, kind in arms if kind == "latency") > 1 or sum(
                    1 for _i, kind in arms if kind == "launch_floor") > 1:
                raise ValueError(
                    "a step term waits on one collective and one launch")
            terms.append((emit(w), arms))
        else:
            # A term whose max SymPy folded to its one floored arm.
            terms.append((emit(w), (arm(expr),)))

    split(_read(output, frozenset(local)), ())
    program = _compiledGraph(
        inputs, definitions,
        tuple((f"_floor{i}", text) for i, text in enumerate(outputs)))
    return _FloorTerms(program, len(definitions), tuple(constants),
                       tuple(terms))


def _collectives(rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not row.get("is_comm") or int(row.get("comm_world") or 1) <= 1:
            continue
        signature = json.dumps({
            "op": row["comm_op"], "world": int(row["comm_world"]),
            "count": int(row["count_per_step"]),
            "stride": int(row["comm_base_stride"]),
            "rungs": [dict(rung) for rung in row.get("comm_rungs") or ()],
        }, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            out.append(signature)
    return tuple(out)


def _measurePointsToJson(
        items: Sequence[tuple[str, int, int, int, PointMeasure]]
) -> dict[str, Any]:
    """One decode grid's concrete lookups as ordinary JSON tables.

    Every distinct string and physical key is named once. Launches and term
    metadata are properties of the step's rows, held by a term variant rather
    than repeated at every point. A point names only the looked-up terms and
    the key entry each reads.
    """
    strings: list[str] = []
    at_string: dict[str, int] = {}
    contexts: list[Any] = []
    at_context: dict[str, int] = {}
    variants: list[dict[str, Any]] = []
    at_variant: dict[str, int] = {}
    keys: list[list[int]] = []
    at_key: dict[tuple[Any, ...], int] = {}
    anchors: list[Any] = []
    at_anchor: dict[str, int] = {}
    dispatches: list[Any] = []
    at_dispatch: dict[str, int] = {}
    shapeAxes: list[str] = []
    at_axis: dict[str, int] = {}
    shapeTemplates: list[list[int]] = []
    at_template: dict[tuple[int, ...], int] = {}
    shapes: list[list[Any]] = []
    at_shape: dict[str, int] = {}

    def text(value: str) -> int:
        got = at_string.get(value)
        if got is None:
            got = at_string[value] = len(strings)
            strings.append(value)
        return got

    def indexed(table: list[Any], at: dict[str, int], value: Any) -> int:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        got = at.get(canonical)
        if got is None:
            got = at[canonical] = len(table)
            table.append(value)
        return got

    def variant(measure: PointMeasure) -> int:
        value = {
            "unkeyed": [text(v) for v in measure.unkeyed],
            "labels": [text(v) for v in measure.labels],
            "reasons": [text(v) for v in measure.reasons],
            "zero_cost": list(measure.formula.zero_cost),
            "launches": list(measure.launches),
            "formula": text(measure.formula.text),
        }
        return indexed(variants, at_variant, value)

    def key(value: Any, read: Any, *, cardless: bool) -> int:
        if value is None or read is None:
            raise ValueError("a looked-up term states its key and selection read")
        decoded = json.loads(value.shape_json)
        template = tuple(indexed(shapeAxes, at_axis, axis)
                         for axis in decoded)
        templateIndex = at_template.get(template)
        if templateIndex is None:
            templateIndex = at_template[template] = len(shapeTemplates)
            shapeTemplates.append(list(template))
        shapeRow = [templateIndex, *(decoded[axis] for axis in decoded)]
        canonicalShape = json.dumps(decoded, sort_keys=True,
                                    separators=(",", ":"))
        if canonicalShape != value.shape_json:
            raise ValueError("a physical key's shape is not canonical JSON")
        shape = at_shape.get(canonicalShape)
        if shape is None:
            shape = at_shape[canonicalShape] = len(shapes)
            shapes.append(shapeRow)
        fields = [text(value.signature), shape, text(value.dtype_env_json),
                  text("" if cardless else value.gpu), text(value.phase)]
        anchor = indexed(anchors, at_anchor, read.get("anchor"))
        dispatch = indexed(dispatches, at_dispatch, read.get("dispatch"))
        canonical = (*fields, anchor, dispatch)
        got = at_key.get(canonical)
        if got is None:
            got = at_key[canonical] = len(keys)
            keys.append([*fields, anchor, dispatch])
        return got

    def withoutCard(measure: PointMeasure) -> PointMeasure:
        return replace(
            measure,
            keys=tuple(None if value is None else value._replace(gpu="")
                       for value in measure.keys),
            context={**(measure.context or {}), "gpu": ""})

    grouped: dict[tuple[int, int, int], list[tuple[str, PointMeasure]]] = {}
    for gpu, context, batch, draft_tokens, measure in items:
        grouped.setdefault((int(context), int(batch), int(draft_tokens)), []).append(
            (gpu, measure))
    points = []
    for (context, batch, draft_tokens), atPoint in grouped.items():
        firstGpu, first = atPoint[0]
        normalized = withoutCard(first)
        cardless = all(
            withoutCard(measure) == normalized
            for _gpu, measure in atPoint[1:])
        representatives = [(firstGpu, first)] if cardless else atPoint
        for gpu, measure in representatives:
            keyed = []
            for i, (physical, read) in enumerate(
                    zip(measure.keys, measure.inputs, strict=True)):
                if physical is None or measure.formula.zero_cost[i]:
                    continue
                keyed.append((i, key(physical, read, cardless=cardless)))
            point = {
                "context": context, "batch": batch,
                "draftTokens": draft_tokens,
                "variant": variant(measure),
                "contextIndex": indexed(
                    contexts, at_context, withoutCard(measure).context
                    if cardless else measure.context),
                "keyedTerms": [term for term, _key in keyed],
                "keyedKeys": [entry for _term, entry in keyed],
            }
            if not cardless:
                point["gpu"] = gpu
            points.append(point)
    return {"schema": _MEASURE_POINTS_SCHEMA, "strings": strings,
            "contexts": contexts, "variants": variants,
            "keyColumns": [list(column) for column in zip(*keys, strict=True)]
            if keys else [],
            "shapeAxes": shapeAxes, "shapeTemplates": shapeTemplates,
            "shapes": shapes,
            "anchors": anchors,
            "dispatches": dispatches, "points": points,
            "cardless": all("gpu" not in point for point in points)}


def _measurePointAt(table: Mapping[str, Any], *, gpu_sku: str,
                    context: int, batch: int, draft_tokens: int = 0,
                    point: Any = None,
                    as_object: bool = False,
                    shared: Optional[Dict[Any, Any]] = None
                    ) -> dict[str, Any]:
    """One stored table row as the `PointMeasure` a fold reads.

    `point` is the already-located row; a one-point caller may omit it.
    """
    if point is None:
        for candidate in table["points"]:
            if (candidate.get("gpu", gpu_sku) == gpu_sku
                    and int(candidate["context"]) == int(context)
                    and int(candidate["batch"]) == int(batch)
                    and int(candidate.get("draftTokens", 0)) == int(draft_tokens)):
                point = candidate
                break
    if point is None:
        raise KeyError(f"the serving law has no measured point for {gpu_sku} "
                       f"at {context} tokens and batch {batch}")

    strings = table["strings"]
    cardless = bool(table.get("cardless"))
    stored = table["variants"][point["variant"]]
    count = len(stored["zero_cost"])
    physical: list[Any] = [None] * count
    inputs: list[Any] = [None] * count
    for term, entry in zip(point["keyedTerms"], point["keyedKeys"],
                           strict=True):
        storedKey = [column[entry] for column in table["keyColumns"]]
        cacheKey = (gpu_sku, tuple(storedKey))
        if shared is not None and cacheKey in shared:
            physical[term], inputs[term] = shared[cacheKey]
        else:
            shapeRow = table["shapes"][storedKey[1]]
            template = table["shapeTemplates"][shapeRow[0]]
            shape = {table["shapeAxes"][axis]: value
                     for axis, value in zip(template, shapeRow[1:],
                                            strict=True)}
            shapeJson = json.dumps(shape, sort_keys=True,
                                   separators=(",", ":"))
            fields = [strings[storedKey[0]], shapeJson,
                      strings[storedKey[2]], strings[storedKey[3]],
                      strings[storedKey[4]]]
            if cardless:
                fields[3] = gpu_sku
            rowInput = {
                "anchor": table["anchors"][storedKey[5]],
                "dispatch": table["dispatches"][storedKey[6]],
            }
            physical[term], inputs[term] = Key(*fields), rowInput
            if shared is not None:
                shared[cacheKey] = (physical[term], inputs[term])
    selectionContext = dict(table["contexts"][point["contextIndex"]])
    if cardless:
        selectionContext["gpu"] = gpu_sku
    measureObject = PointMeasure(
        tuple(physical), tuple(float(v) for v in stored["launches"]),
        tuple(strings[i] for i in stored["unkeyed"]),
        tuple(strings[i] for i in stored["labels"]),
        tuple(strings[i] for i in stored["reasons"]),
        MeasuredFormula(strings[stored["formula"]],
                        tuple(stored["zero_cost"])),
        selectionContext, tuple(inputs), (), (),
    )
    # Gallery/Scaling SOL points carry no per-op row table. The measured fold
    # updates their phase totals and coverage; term-to-row mapping is
    # breakdown-only and is not a pool fact.
    return {"measure": measureObject if as_object
            else measureObject.to_json(),
            "terms": [], "phase": "decode"}


def servingMeasurePoint(record: Mapping[str, Any], *, card: Card,
                        context: int, batch: int,
                        draft_tokens: int = 0) -> dict[str, Any]:
    """A record's stored decode measurement at one public point.

    The producer derived every key through `calibration_overlay.point_keys`.
    This read is a table lookup; it neither replays a law nor walks.
    """
    table = record.get("phases", {}).get("decode", {}).get("measurePoints")
    if table is None:
        raise ValueError("the serving law carries no measured points")
    if table.get("schema") != _MEASURE_POINTS_SCHEMA:
        raise ValueError(
            f"a serving law measure table must be {_MEASURE_POINTS_SCHEMA}, "
            f"got {table.get('schema')!r}")
    return _measurePointAt(
        table, gpu_sku=card.sku, context=int(context), batch=int(batch),
        draft_tokens=int(draft_tokens))


def servingMeasureMatrix(record: Mapping[str, Any], *, card: Card,
                         points: Sequence[Tuple[int, int, int]]) -> MeasureMatrix:
    """A record's measured points as the matrix its fold reads."""
    table = record.get("phases", {}).get("decode", {}).get("measurePoints")
    if table is None:
        raise ValueError("the serving law carries no measured points")
    if table.get("schema") != _MEASURE_POINTS_SCHEMA:
        raise ValueError(
            f"a serving law measure table must be {_MEASURE_POINTS_SCHEMA}, "
            f"got {table.get('schema')!r}")
    return MeasureMatrix.of_record(table, card=card, wanted=points)


def _measurePointTables(pending: Any, preparation: Any,
                        base: Mapping[str, Any],
                        plain_capacity: Sequence[Mapping[str, Any]],
                        draft_capacity: Optional[Sequence[Mapping[str, Any]]] = None, *,
                        cards: Sequence[str],
                        contexts: Sequence[int],
                        batchGrid: Sequence[Mapping[str, Any]]
                        ) -> Dict[str, dict[str, Any]]:
    """Every public point's `PointMeasure`, derived once while the walk is live.

    Each point uses the same road a plan and a live estimate use:
    `capacity._point_rows` moves the walked rows to the point and card, then
    `calibration_overlay.point_keys` derives what the lake is asked. The
    resulting `PointMeasure`s are plain data; a pool reader never replays a
    symbolic row key. The public pool carries the decode measured track:
    prefill and custom plan points walk and ask `point_keys` there.
    """
    base_phases = base.get("phases") or {}
    base = {name: rows for name, rows in base.items()
            if name in ("per_op_timing_decode", "per_op_timing_draft")}
    all_facts = {"plain": plain_capacity, "draft": draft_capacity or plain_capacity}
    byPhase: Dict[str, list[tuple[str, int, int, int, PointMeasure]]] = {}
    speculative = preparation.draft is not None
    draft_phase = dict(base_phases.get("decode") or {})
    plain_phase = {key: value for key, value in draft_phase.items()
                   if key != "speculative_draft"}
    if speculative and "speculative_draft" not in draft_phase:
        raise ValueError("the walked decode phase does not state its draft")
    ks = (0, *FRONTIER_DRAFT_TOKENS) if speculative else (0,)
    for context in contexts:
        for k in ks:
            facts = {(str(fact["gpu_sku"]), int(fact["context"])): fact
                     for fact in all_facts["plain" if k == 0 else "draft"]}
            cardFacts = [(gpu_sku, facts.get((gpu_sku, int(context))))
                         for gpu_sku in cards]
            cardFacts = [pair for pair in cardFacts if pair[1] is not None]
            if not cardFacts:
                continue
            wanted = {int(batchPoint["decode_batch_size"])
                      for batchPoint in batchGrid
                      if batchPoint["decode_batch_size"] is not None}
            wanted.update(int(pair[1]["max_decode_batch"]) for pair in cardFacts)
            phase = plain_phase if k == 0 else {
                **draft_phase,
                "speculative_draft": {
                    **draft_phase["speculative_draft"],
                    "draft_tokens": k, "forwards": k,
                },
            }
            atPreparation = replace(preparation, s_verify=k + 1)
            onCard = _restamp_gpu(atPreparation, Card.of(cards[0]))
            for decodeBatch in sorted(wanted):
                if decodeBatch <= 0:
                    continue
                if not any(point_served(
                        max_batch_per_gpu=fact["max_decode_batch"],
                        beyond_max_context=fact["beyond_max_context"],
                        requests=decodeBatch or 1)
                        for _gpu_sku, fact in cardFacts):
                    continue
                batches = {
                    "per_op_timing_decode": decodeBatch,
                    "per_op_timing_draft": decodeBatch,
                }
                rows = _point_rows(
                    base, onCard, batches, pending,
                    Point({"s_input": int(context)}), breakdown=False,
                    costs=False)
                for gpu_sku, fact in cardFacts:
                    if not point_served(
                            max_batch_per_gpu=fact["max_decode_batch"],
                            beyond_max_context=fact["beyond_max_context"],
                            requests=decodeBatch or 1):
                        continue
                    keyed: Dict[str, list[Any]] = {}
                    for name, walk in pending.walk_of_rows.items():
                        if name not in base:
                            continue
                        keyed[name] = point_keys(
                            walk.analysis, rows.get(name) or [], walk.workload,
                            replace(walk.deployment, gpu_sku=gpu_sku),
                            model_id=walk.model_id, spec_hash=walk.spec_hash)
                    measures = point_measures(
                        pending, rows, keyed, {"decode": phase},
                        gpu_sku=gpu_sku, as_objects=True)
                    for phase_name, measure in measures.items():
                        byPhase.setdefault(phase_name, []).append(
                            (gpu_sku, int(context), decodeBatch, k,
                             measure["measure"]))
    return {phase: _measurePointsToJson(items)
            for phase, items in byPhase.items()}



def _phase(rows: list[Mapping[str, Any]],
           *, mla_absorbed: bool = False,
           draft_rows: Iterable[Mapping[str, Any]] = (),
           draft_forwards: int = 0) -> dict[str, Any]:
    """One phase's law and every collective its summed terms name.

    Draft rows are terms of this same step (`StepSum.of`): their collectives
    enter the phase law at their own count — commonly 3, while the backbone
    rows name 1, 8 or 39 — so the sidecar must collect them here too. The law
    and its binding signatures are one fact of `_phase`, never of a reader.
    """
    draft_rows = tuple(draft_rows)
    step = StepSum.of(rows, draft_rows=draft_rows,
                      draft_forwards=draft_forwards)
    graph = LawGraph.of({
        wire: step.law(name) for wire, name in _lawOutputs.items()
    })
    return {"law": graph.to_json(),
            "collectives": list(_collectives((*rows, *draft_rows))),
            "terms": len(rows) + len(draft_rows),
            "mla_absorbed": bool(mla_absorbed)}


def _capacityFacts(preparation: Any, outputTokens: int, *,
                   cards: Sequence[str], contexts: Sequence[int],
                   prefillChunkSize: Optional[int] = None
                   ) -> list[dict[str, Any]]:
    """One preparation's public capacity facts, by card and context.

    This is the only construction; `servingLawCapacity` publishes it, and a
    speculative serving record receives the plain answer as `publicCapacity`
    rather than recomputing it.
    """
    if preparation.refusal_reason is not None:
        def refusedOn(cardSku: str) -> Any:
            return _restamp_gpu(preparation, Card.of(cardSku))

        return [{
            "gpu_sku": cardSku, "context": int(context),
            "max_decode_batch": 0, "max_prefill_batch": 0,
            "beyond_max_context": int(context) > int(
                preparation.max_context or 0),
            "infeasibility_reason": preparation.refusal_reason,
            **servingCapacityView(refusedOn(cardSku), capacity_ceilings(
                refusedOn(cardSku), avg_input_tokens=int(context),
                avg_output_tokens=int(outputTokens),
                prefill_chunk_size=prefillChunkSize)),
        } for cardSku in cards for context in contexts]
    out: list[dict[str, Any]] = []
    for cardSku in cards:
        onCard = _restamp_gpu(preparation, Card.of(cardSku))
        for context in contexts:
            ceilings = capacity_ceilings(
                onCard, avg_input_tokens=int(context),
                avg_output_tokens=int(outputTokens),
                prefill_chunk_size=prefillChunkSize,
                prefill_capacity=True)
            out.append({
                "gpu_sku": cardSku, "context": int(context),
                "max_decode_batch": int(ceilings.max_batch),
                "max_prefill_batch": int(ceilings.max_prefill_batch),
                "beyond_max_context": bool(ceilings.beyond_max_context),
                "infeasibility_reason": ceilings.infeasibility_reason,
                **servingCapacityView(onCard, ceilings),
            })
    return out


def servingLawRecord(environment: Mapping[str, Any], *, gpu_sku: str,
                     anchorContext: int, outputTokens: int,
                     parallel: Any,
                     capacityCards: Sequence[str] = (),
                     capacityContexts: Sequence[int] = (),
                     draft_forwards: Any = None,
                     measureContexts: Sequence[int] | None = None,
                     measureBatchGrid: Sequence[Mapping[str, Any]] | None = None,
                     publicCapacity: Sequence[Mapping[str, Any]] | None = None,
                     ) -> dict[str, Any]:
    """Walk one layout once and keep its SOL laws, no expanded points."""
    knobs = CapacityKnobs.from_kwargs(environment, envelope=frozenset({
        "config", "gpu_sku", "avg_output_tokens", "prefill_batch_size",
        "decode_batch_size", "prefill_chunk_size"}))
    preparation = _capacity_preparation(
        config=environment["config"], gpu_sku=gpu_sku, knobs=knobs)
    if preparation.refusal_reason is not None:
        return {"refused": preparation.refusal_reason}

    capacity = (list(publicCapacity) if publicCapacity is not None else
                _capacityFacts(preparation, outputTokens,
                               cards=capacityCards or (gpu_sku,),
                               contexts=capacityContexts or (1024,),
                               prefillChunkSize=environment.get(
                                   "prefill_chunk_size")))
    if all(int(fact["max_decode_batch"] or 0) <= 0
           and int(fact["max_prefill_batch"] or 0) <= 0
           for fact in capacity):
        return {"refused": next(
            fact.get("infeasibility_reason") or "layout admits no request"
            for fact in capacity)}

    pending = _capacity_pending(
        preparation,
        avg_input_tokens=anchorContext,
        avg_output_tokens=outputTokens,
        prefill_batch_override=1,
        decode_batch_override=1,
        breakdown=False,
        fold_prefill_chunks=False,
        prefill_capacity=False,
    )
    prefillWalk = pending.walk_of_rows["per_op_timing"]
    decodeWalk = pending.walk_of_rows["per_op_timing_decode"]
    prefillRows = prefillWalk.analysis.per_op_timing
    decodeRows = decodeWalk.analysis.per_op_timing
    draftRows = (
        (pending.walk_of_rows.get("per_op_timing_draft")
         and pending.walk_of_rows["per_op_timing_draft"].analysis.per_op_timing)
        or ())

    draftForwards = (0 if preparation.draft is None
                     else preparation.draft.forwards)
    if draft_forwards is not None:
        draftForwards = draft_forwards
    phases = {
        "prefill": _phase(
            prefillRows,
            mla_absorbed=bool(prefillWalk.step_eval.mla_absorbed)),
        "decode": _phase(
            decodeRows,
            mla_absorbed=bool(decodeWalk.step_eval.mla_absorbed),
            draft_rows=draftRows,
                         draft_forwards=draftForwards),
    }
    base = pending.finish()
    draftCapacity = (
        None if preparation.draft is None else _capacityFacts(
            preparation, outputTokens,
            cards=capacityCards or (gpu_sku,),
            contexts=capacityContexts or (1024,),
            prefillChunkSize=environment.get("prefill_chunk_size")))
    # K=0 and every public speculative K are point views over this one walk,
    # each keyed through the same `point_keys` road a plan uses. The table's
    # K field is the only difference between the views; its strings, shapes
    # and physical keys remain record-level deduplicated facts.
    measures = _measurePointTables(
        pending, preparation, base, capacity, draftCapacity,
        cards=capacityCards or (gpu_sku,),
        contexts=GALLERY_SEQ_LENS if measureContexts is None
        else measureContexts,
        batchGrid=FRONTIER_BATCH_GRID if measureBatchGrid is None
        else measureBatchGrid)
    for name, table in measures.items():
        phases[name]["measurePoints"] = table
    # The record's producer is the boundary at which the lazily built walk is
    # complete; re-account its shared family before publishing the record.
    if preparation.account_cells is not None:
        preparation.account_cells()
    return {
        "schema": servingLawSchema,
        "walked_on": gpu_sku,
        "framework": str(knobs.framework or ""),
        "parallel": preparation.parallel.to_dict(),
        "speculative": preparation.draft is not None,
        "gpu_memory_fraction": knobs.gpu_memory_fraction,
        "max_context_tokens": preparation.max_context,
        "parallel_warnings": preparation.wb.get("parallel_warnings", []),
        "phases": phases,
        "capacity": capacity,
        **({} if draftCapacity is None else {"draft_capacity": draftCapacity}),
    }


def servingLawCapacity(environment: Mapping[str, Any], *, gpu_sku: str,
                       parallel: Any,
                       capacityCards: Sequence[str] = (),
                       capacityContexts: Sequence[int] = ()) -> list[Any]:
    """The plain (non-speculative) capacity facts without a phase law.

    A speculative record's decode law already covers K=0 through its
    `s_verify` symbol, but its preparation also loads the draft stack. Capacity
    keeps both answers: this plain one and the drafted one
    (`servingLawRecord.draft_capacity`)."""
    knobs = CapacityKnobs.from_kwargs(environment, envelope=frozenset({
        "config", "gpu_sku", "avg_output_tokens", "prefill_batch_size",
        "decode_batch_size", "prefill_chunk_size"}))
    preparation = _capacity_preparation(
        config=environment["config"], gpu_sku=gpu_sku, knobs=knobs)
    return _capacityFacts(
        preparation, int(environment["avg_output_tokens"]),
        cards=capacityCards or (gpu_sku,),
        contexts=capacityContexts or (1024,),
        prefillChunkSize=environment.get("prefill_chunk_size"))

def _capacityAt(record: Mapping[str, Any], card: Card, context: int):
    for item in record.get("capacity") or ():
        if item["gpu_sku"] == card.sku and item["context"] == context:
            return item
    raise KeyError(
        f"the serving law has no capacity fact for {card.sku} at {context}")


def servingLawPhaseView(record: Mapping[str, Any], name: str, *, batch: int,
                        time: float, context: int, outputTokens: int,
                        verifyWidth: int, prefixCached: int, draftTokens: int,
                        acceptanceRate: float, groups: int,
                        num_gpus: int) -> dict[str, Any]:
    phaseBinding = phase_binding(
        phase=name, avg_input_tokens=context,
        avg_output_tokens=outputTokens, s_prefix_cached=prefixCached,
        s_verify=verifyWidth)
    tokensPerStep = float(phaseBinding["tokens_per_step"])
    if name == "decode" and draftTokens:
        effectiveTokens, effectiveFormula = step_tokens(
            "decode", batch, tokensPerStep,
            Speculation(draftTokens, acceptanceRate))
    else:
        effectiveTokens = tokensPerStep * batch
        effectiveFormula = {
            "symbolic": (
                "B × s_query"
                if name == "prefill" or draftTokens else "B × 1"),
            "substituted": f"{batch} × {tokensPerStep:g}",
        }
    phase = {
        "phase": name,
        "batch": batch,
        "binding": {
            **{key: value for key, value in phaseBinding.items()
               if key != "phase"},
            "effective_tokens_per_step": effectiveTokens,
            "effective_tokens_formula": effectiveFormula,
            "speculative_draft_tokens": (
                draftTokens if name == "decode" else 0),
            "acceptance_rate": (
                acceptanceRate if name == "decode" else 1.0),
            "mla_absorbed": bool(
                record["phases"][name].get("mla_absorbed")),
        },
        "t_sol_s": time,
        "time_unpublished_reason": "",
        "t_measured_s": None,
        "coverage": None,
    }
    phase["throughput"] = phase_throughput(
        phase, unique_tokens_multiplier=groups, num_gpus=num_gpus)
    return phase


def _servingLawPoint(record: Mapping[str, Any], *, facts: Mapping[str, Any],
                     cardValues: Mapping[str, float],
                     floorCard: "FloorCard",
                     graphs: Mapping[str, LawGraph], context: int,
                     outputTokens: int, decodeBatch: int | None,
                     prefillBatch: int | None, prefixCached: int,
                     verifyWidth: int,
                     draftTokens: int, acceptanceRate: float,
                     prefillTimes: dict[tuple[int, int, int, int], float],
                     floorForms: bool = False) -> dict:
    decodeB = (int(decodeBatch) if decodeBatch is not None
               else int(facts["max_decode_batch"]))
    prefillB = (int(prefillBatch) if prefillBatch is not None
               else int(facts["max_prefill_batch"]))
    phases = {}
    parallel = ParallelConfig.from_dict(record["parallel"])
    groups = parallel.dp_attention * parallel.dp_replicas
    num_gpus = parallel.cluster_size
    for name, batch in (("prefill", prefillB), ("decode", decodeB)):
        graph = graphs[name]
        point = {
            **cardValues, "B": max(1, batch), "s_input": context,
            "s_output": outputTokens, "s_prefix": int(prefixCached),
            "s_verify": verifyWidth,
            "rho": 1,
        }
        if name == "prefill":
            key = (int(context), int(prefixCached), prefillB,
                   int(verifyWidth))
            if key not in prefillTimes:
                schedule = PrefillChunkSchedule.materialize(
                    PrefillEndpointGeometry(
                        int(context), int(prefixCached), 1, max(1, batch)),
                    prefill_chunk_size=8192)
                t = 0.0
                for chunk in schedule.chunks:
                    repeat = (schedule.endpoint_batch // chunk.launch_batch
                              if len(schedule.chunks) > 1 else 1)
                    chunkTime = float(graph.evaluate({
                        **point, "B": chunk.launch_batch,
                        "s_input": chunk.end, "s_prefix": chunk.start,
                    })["t_sol_s"])
                    t += chunkTime * repeat
                prefillTimes[key] = t
            t = prefillTimes[key]
        else:
            # The decode step is its floor form at the request's floors:
            # one road to the time, whichever floors a reader substitutes.
            form = graph.floorForm(point, floorCard)
            t = form.at(*floorCard.floors_s)
        phases[name] = servingLawPhaseView(
            record, name, batch=batch, time=t, context=context,
            outputTokens=outputTokens, verifyWidth=verifyWidth,
            prefixCached=int(prefixCached),
            draftTokens=draftTokens, acceptanceRate=acceptanceRate,
            groups=groups, num_gpus=num_gpus)
        if name == "decode" and floorForms:
            phases[name]["floors"] = form.to_json()
    return {
        "avg_input_tokens": context,
        "avg_output_tokens": outputTokens,
        "max_batch_per_gpu": facts["max_decode_batch"],
        "max_decode_batch_per_gpu": facts["max_decode_batch"],
        "max_prefill_batch_per_gpu": facts["max_prefill_batch"],
        "max_decode_batch_total": facts.get("max_decode_batch_total"),
        "max_prefill_batch_total": facts.get("max_prefill_batch_total"),
        "max_tokens_per_gpu": facts.get("max_tokens_per_gpu"),
        "max_tokens_total": facts.get("max_tokens_total"),
        "num_gpus": num_gpus,
        "cluster_size": parallel.cluster_size,
        "parallel": record["parallel"],
        "gpu_memory_fraction": record.get("gpu_memory_fraction"),
        "beyond_max_context": facts["beyond_max_context"],
        "infeasibility_reason": facts["infeasibility_reason"],
        "max_context_tokens": record.get("max_context_tokens"),
        "parallel_warnings": list(record.get("parallel_warnings") or []),
        "memory": facts.get("memory"),
        "kv_cache": facts.get("kv_cache"),
        "phases": phases,
    }


def servingLawPoint(record: Mapping[str, Any], *, card: Card,
                    context: int, outputTokens: int,
                    decodeBatch: int | None = None,
                    prefillBatch: int | None = 1,
                    prefixCached: int = 0,
                    verifyWidth: int = 1,
                    draftTokens: int = 0,
                    acceptanceRate: float = 1.0,
                    comm_latency_us: float, launch_floor_us: float) -> dict:
    """One served point on `card`, derived from the law and the card's
    capacity facts. `card` must walk as the record's walk card does
    (`cardBinding`, the one place a card enters a law)."""
    draftCapacity = (
        {"capacity": record["draft_capacity"]} if draftTokens
        and "draft_capacity" in record else record)
    facts = _capacityAt(draftCapacity, card, context)
    return _servingLawPoint(
        record, facts=facts,
        cardValues=cardBinding(
            record, card, comm_latency_us=comm_latency_us,
            launch_floor_us=launch_floor_us),
        floorCard=FloorCard.of(
            record, card, comm_latency_us=comm_latency_us,
            launch_floor_us=launch_floor_us),
        graphs={name: LawGraph.from_json(record["phases"][name]["law"])
                for name in ("prefill", "decode")},
        context=context, outputTokens=outputTokens, decodeBatch=decodeBatch,
        prefillBatch=prefillBatch, prefixCached=prefixCached,
        verifyWidth=verifyWidth,
        draftTokens=draftTokens, acceptanceRate=acceptanceRate,
        prefillTimes={})


def servingLawCapacityFacts(record: Mapping[str, Any], *, card: Card,
                            contexts: Sequence[int],
                            draftTokens: int = 0) -> dict[int, dict[str, Any]]:
    """The capacity facts a serving projection needs, by context.

    The law is not read here: this is the cheap existence half of a
    projection, asked before any point is priced.
    """
    capacityRecord = (
        {"capacity": record["draft_capacity"]} if draftTokens
        and "draft_capacity" in record else record)
    return {int(context): dict(_capacityAt(capacityRecord, card, int(context)))
            for context in contexts}


def servingLawPoints(record: Mapping[str, Any], *, card: Card,
                     contexts: Sequence[int], decodeBatches: Sequence[int | None],
                     outputTokens: int,
                     prefillBatches: int | Sequence[int | None] = 1,
                     prefixCached: int | Sequence[int] = 0,
                     verifyWidth: int | Sequence[int] = 1,
                     draftTokens: int = 0,
                     acceptanceRate: float = 1.0,
                     comm_latency_us: float, launch_floor_us: float,
                     servedOnly: bool = False,
                     floorForms: bool = False
                     ) -> list[list[dict | None]]:
    """A context-by-batch projection of one serving-law record.

    This is not a second walk: it reads the same two graphs as
    ``servingLawPoint``. The grid only changes what is shared — the card
    binding, each context's capacity fact and each context's prefill chunk
    schedule — and binds each context's prefix and verify width. With
    ``servedOnly``, capacity is asked before the law is evaluated; a missing
    capacity fact is an absent point, as a reader of one point already
    treats its exception as absence. With ``floorForms``, each point's
    decode phase also publishes its step as a function of the two floors
    (`floors`, `FloorForm.to_json`), the form its `t_sol_s` is read off.
    """
    cardValues = cardBinding(
        record, card, comm_latency_us=comm_latency_us,
        launch_floor_us=launch_floor_us)
    floorCard = FloorCard.of(record, card, comm_latency_us=comm_latency_us,
                             launch_floor_us=launch_floor_us)
    if (not isinstance(prefillBatches, int)
            and len(prefillBatches) != len(decodeBatches)):
        raise ValueError("prefillBatches must name one batch per decode batch")
    if isinstance(prefixCached, int):
        prefixes = [prefixCached] * len(contexts)
    elif len(prefixCached) == len(contexts):
        prefixes = list(prefixCached)
    else:
        raise ValueError("prefixCached must name one prefix per context")
    if isinstance(verifyWidth, int):
        widths = [verifyWidth] * len(contexts)
    elif len(verifyWidth) == len(contexts):
        widths = list(verifyWidth)
    else:
        raise ValueError("verifyWidth must name one width per context")
    graphs = {name: LawGraph.from_json(record["phases"][name]["law"])
              for name in ("prefill", "decode")}
    prefillTimes: dict[tuple[int, int, int, int], float] = {}
    out: list[list[dict | None]] = []
    for context_index, context in enumerate(contexts):
        capacityRecord = (
            {"capacity": record["draft_capacity"]} if draftTokens
            and "draft_capacity" in record else record)
        prefixCached = int(prefixes[context_index])
        verifyWidth = int(widths[context_index])
        try:
            facts = _capacityAt(capacityRecord, card, int(context))
        except KeyError:
            if servedOnly:
                out.append([None] * len(decodeBatches))
                continue
            raise
        row: list[dict | None] = []
        for i, decodeBatch in enumerate(decodeBatches):
            requests = int(decodeBatch) if decodeBatch is not None else 1
            if servedOnly and not point_served(
                    max_batch_per_gpu=facts["max_decode_batch"],
                    beyond_max_context=bool(facts["beyond_max_context"]),
                    requests=requests):
                row.append(None)
                continue
            prefillBatch = (
                prefillBatches if isinstance(prefillBatches, int)
                else prefillBatches[i])
            point = _servingLawPoint(
                record, facts=facts, cardValues=cardValues,
                floorCard=floorCard,
                graphs=graphs, context=int(context), outputTokens=outputTokens,
                decodeBatch=decodeBatch, prefillBatch=prefillBatch,
                prefixCached=prefixCached, verifyWidth=verifyWidth,
                draftTokens=draftTokens,
                acceptanceRate=acceptanceRate, prefillTimes=prefillTimes,
                floorForms=floorForms)
            row.append(without_step(point, ())
                       if facts.get("infeasibility_reason") else point)
        out.append(row)
    return out


@dataclass(frozen=True)
class FloorCard:
    """A card's binding of a law with the two floors apart: the card's
    values at both floors 0 (`zero`) and at both 1 s (`unit`) — a
    collective's fixed time is linear in the link latency
    (`comm_cost.collective_latency`), a launch is `LAUNCH_S` — and the
    floors a point is read at (`floors_s`: link latency, launch floor)."""
    zero: Mapping[str, float]
    unit: Mapping[str, float]
    floors_s: tuple[float, float]

    @classmethod
    def of(cls, record: Mapping[str, Any], card: Card, *,
           comm_latency_us: float, launch_floor_us: float) -> "FloorCard":
        oneSecondUs = 1e6
        return cls(
            cardBinding(record, card, comm_latency_us=0.0,
                        launch_floor_us=0.0),
            cardBinding(record, card, comm_latency_us=oneSecondUs,
                        launch_floor_us=oneSecondUs),
            (comm_latency_us * 1e-6, launch_floor_us * 1e-6))


def cardBinding(record: Mapping[str, Any], card: Card, *,
                comm_latency_us: float, launch_floor_us: float,
                dtypes: Sequence[str] = ()) -> dict[str, float]:
    """The card half of a law graph's inputs — `card`'s peaks, bandwidth and
    fabric — for a card of the class the record's rows were walked on
    (`walked_on`), refused for any other (`Card.refuse_unless_one_class`).

    The one door a card enters a law through, so the rule is asked here and
    not by each projection: a law's terms are one walk's rows, and a card of
    another class runs other rows (another kernel, another activation
    format). Its symbols would bind and the number would look right."""
    Card.of(record["walked_on"]).refuse_unless_one_class(card)
    gpu = card.spec
    fabric = card.fabric(comm_latency_us * 1e-6)
    out: dict[str, float] = {
        "PEAK_BW": float(peak_hbm_bw(gpu) or math.inf),
        "COMM_S_PER_BYTE": 0.0,
        "COMM_LATENCY_S": 0.0,
        "LAUNCH_S": launch_floor_us * 1e-6,
    }
    wanted = set(dtypes)
    for phase in record["phases"].values():
        graph = LawGraph.from_json(phase["law"])
        for name in graph.inputs:
            if name == "PEAK_FLOPS":
                out[name] = float(peak_flops_for_dtype(gpu, "") or math.inf)
            elif name.startswith("PEAK_FLOPS_"):
                dtype = name[len("PEAK_FLOPS_"):]
                wanted.add(dtype)
                out[name] = float(
                    peak_flops_for_dtype(gpu, dtype) or math.inf)

    def collectiveValues(signature: str) -> None:
        facts = json.loads(signature)
        rungs = tuple(CollectiveRung.from_dict(rung)
                      for rung in facts.get("rungs") or ())
        ladder = collective_ladder(
            op=facts["op"], world=int(facts["world"]),
            count_per_step=int(facts["count"]),
            base_stride=int(facts["stride"]), fabric=fabric, rungs=rungs)
        base = (f"_{facts['op']}_{int(facts['world'])}_"
                f"{int(facts['count'])}_{int(facts['stride'])}")
        names = (f"COMM_S_PER_BYTE{base}", f"COMM_LATENCY_S{base}")
        values = (ladder[-1].s_per_byte, ladder[-1].latency_s)
        for symbol, value in zip(names, values, strict=True):
            out[symbol] = float(value)
        for index, segment in enumerate(ladder[:-1]):
            names = tuple(
                f"{symbol.name[:-len(f'_r{index}')] if symbol.name.endswith(f'_r{index}') else symbol.name}{base}_r{index}"
                for symbol in rung_symbols(index))
            values = (segment.max_bytes, segment.s_per_byte,
                      segment.latency_s)
            for symbol, value in zip(names, values, strict=True):
                out[symbol] = float(value)

    seen: set[str] = set()
    for phase in record["phases"].values():
        for signature in phase["collectives"]:
            if signature not in seen:
                seen.add(signature)
                collectiveValues(signature)
    return out
