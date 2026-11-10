"""A phase and its per-op rows re-evaluated at another batch or on another
card.

Each per-op row carries its laws over the workload and the card — every
workload count (batch, prompt, cached prefix, output, verify width) and the
card's peaks: its roofline (`roofline_expr`, `row_roofline.build_row_exprs`),
its shape (`shape_law`) and the rest of what it publishes that moves with the
point (`point_law`: the point the row's numbers are at,
its node's dims, its FLOPs formula's expression and substitution record, its
access formula's display facts, a collective's payload and shapes). The
rescale substitutes into them per row — no re-walk, no overlay re-run — and
prints what it publishes through the same functions the walk printed its own
row with (`row_roofline.point_columns`, `phase_formulas`), so a row re-read at
a point is the row a walk there emits. The parallel sweep re-prices each
deployment at each card's max batch here; the web client reads the results
and does not recompute the roofline.

What substitutes, and what is rendered after it:

  * laws, read by substitution — every number a row publishes (FLOPs, each
    byte column, the collective's payload and per-rank volume, the four
    times), its shapes (per-rank, exact, unsharded, wire dims, a
    collective's payload and buffer shapes, a fused kernel's absorbed
    parts'), its node's dims and its FLOPs formula's substitution record;
    a phase's totals are the step's sum of those (`row_roofline.StepSum`);
  * a tail, rendered from the substituted numbers by the function the walk
    renders its own row with — the term that set a time (an argmax; as a law
    it is `Piecewise`, `StepSum.bound_law`), arithmetic intensity, every
    popover line and the phase note (text printed from numbers), a
    collective's per-phase rows (how many phases, and on which link, is the
    fabric's topology; the time is linear in the payload and substitutes as
    `COMM_S_PER_BYTE` beside its fixed `COMM_LATENCY_S`), throughput (tokens over the point's time), a table
    parent's sums; the unsharded `flops_full` keeps its linear factor (the
    full cost carries no law on the row).

What a re-read row does not carry is its measurement identity
(`shape_json`, `dtype_env_json`, `signature`): at another batch the key is
the point's, and the sweep pipeline states it by calling `point_keys` on the
re-read row. Until then those fields are absent — not None — so a row that
was not keyed is never mistaken for one keyed to nothing.
"""
from __future__ import annotations

import time

from backend.arch.eval.analysis import Point
from backend.arch.eval.dim import as_extent, as_number
from backend.arch.eval.transforms import Workload
from backend.arch.layout.materialize_comm import payload_substituted
from backend.arch.law_text import read_walk_law

from .comm_cost import Fabric
from .declaration_groups import declaration_terms
from .intensity import arithmetic_intensity
from .module_rows import TableIdentity, shape_columns, table_row
from .calibration_overlay import (_resolve_overlay_context, key_columns_at, missed_badge,
                                  dispatch_source_anchor)
from .gpu_specs import get_gpu, peak_flops_for_dtype, peak_hbm_bw
from .op_calibration import (PhysicalKey, merge_validated_rows, registry_from_lookup,
                             select_time)
from .phases import Speculation, phase_note, phase_throughput, step_tokens
from .phase_formulas import (FormulaDefs, _access_bytes_formula_for_row, _flops_formula_for_row,
                             _rollup_column, enrich_module_breakdown,
                             phase_roofline_formulas)
from .row_roofline import (LawBook, RowCard, RowRoofline,
                           RowTimes, StepSum, _exact_number, bare_row,
                           point_columns)
from .speculative_draft import SpeculativeDraft
from .chunk_terms import ChunkLaunchTerm, merge_chunk_launch_terms

from backend.measure.auto_sweep import (DECLINED, MEASURED, PIPELINE_STAGE, UNPLACED,
                                        ZERO_LAUNCH, MeasuredFormula, MeasuredTrack,
                                        PointMeasure, look_up)
from backend.measure.lookup import Key
from backend.measure.measure_matrix import MeasureMatrix, MatrixFold
from backend.measure.scope import NONE as NO_MEASUREMENTS

from dataclasses import asdict, replace
from typing import (Any, Dict, Iterable, Iterator, List, Mapping,
                    Optional, Sequence, Tuple)

import sympy as sp
from backend.code_trace.memo import memoize


def _tensors(read: Mapping[Any, Any], laws: List[List[Any]]
             ) -> List[List[sp.Rational]]:
    return [[read[_t] for _t in axes] for axes in laws]


def _one_or_many(tensors: List[Any]) -> Any:
    return tensors[0] if len(tensors) == 1 else tensors


def _floor(axes: List[Any]) -> List[int]:
    return [as_extent(v, rounding="floor") for v in axes]


def _shape_columns_at(read: Mapping[str, Any], row: Mapping[str, Any]
                      ) -> Dict[str, Any]:
    """The row's shape columns read at a point (`read`) off its `shape_law`. A row
    without a law keeps no shape at all — its shape columns are None and it
    says so (`shape_at_batch: False`) — so a reader that needs one refuses
    rather than reading the walked point's."""
    law = row.get("shape_law")
    if law is None:
        return dict(shape_in=None, shape_in_exact=None, shape_in_dims=None,
                    shape_in_full=None, shape_out=None, shape_out_exact=None,
                    shape_out_dims=None, shape_out_full=None, shape_outs=None,
                    shape_at_batch=False)
    ins = _tensors(read, law["in"])
    out = [read[_t] for _t in law["out"]] if law["out"] is not None else None
    ports = _tensors(read, law["ports"]) if law["ports"] else None
    shape_in = _one_or_many([_floor(t) for t in ins]) if ins else None
    shape_out = _floor(out) if out is not None else None
    in_full = (_one_or_many([_floor(t) for t in _tensors(read, law["in_full"])])
               if law["in_full"] else None)
    out_full = (_floor([read[_t] for _t in law["out_full"]])
                if law["out_full"] is not None else None)

    # The wire dims (`{expr, value}` per axis) are the walked shape's, every
    # input of it: a collective's columns are its typed flow, its wire dims
    # the walked axes (`point_law["wire"]`).
    wire = row["point_law"]["wire"]

    def named(dims: Any, laws: Any) -> Any:
        return [{**d, "value": as_number(v)}
                for d, v in zip(dims, [read[_t] for _t in laws], strict=True)]

    in_dims, out_dims = row.get("shape_in_dims"), row.get("shape_out_dims")
    out_wire = (None if wire is None else wire["out"]
                if wire["out"] is not None else wire.get("ports") or None)
    out_many = wire is not None and wire["out"] is None and out_wire is not None
    return {
        "shape_in": shape_in,
        # The exact extents are the walked shape's, every input of it (a
        # collective's floored columns are its typed flow).
        "shape_in_exact": (_one_or_many([[as_number(read[_t]) for _t in t] for t in wire["in"]])
                           if wire is not None and wire["in"] else None),
        "shape_in_dims": (None if in_dims is None or wire is None
                          else named(in_dims, wire["in"][0]) if len(wire["in"]) == 1
                          else [named(d, t) for d, t in
                                zip(in_dims, wire["in"], strict=True)]),
        # The unsharded shape only where the shard differs from it, as the
        # walk publishes it.
        "shape_in_full": (in_full if in_full and shape_in and in_full != shape_in
                          else None),
        "shape_out": shape_out,
        # Of the one output, or of every port in `shape_outs`' order, as
        # the walk publishes them (`phases`).
        "shape_out_exact": (None if out_wire is None else
                            [[as_number(read[_t]) for _t in t]
                             for t in out_wire]
                            if out_many else
                            [as_number(read[_t]) for _t in out_wire]),
        "shape_out_dims": (None if out_dims is None or out_wire is None
                           else [named(d, t) for d, t in
                                 zip(out_dims, out_wire, strict=True)]
                           if out_many else named(out_dims, out_wire)),
        "shape_out_full": (out_full if out_full and shape_out
                           and out_full != shape_out else None),
        "shape_outs": [_floor(p) for p in ports] if ports else None,
        "shape_at_batch": True,
    }


def _shape_text(tensors: List[List[sp.Rational]]) -> Optional[str]:
    """`a*b*c + d*e`: tensors' exact extents as the access popover prints
    them (`projection.access._formula_display`)."""
    rendered = ["*".join(str(as_number(v)) for v in t) for t in tensors]
    rendered = [r for r in rendered if r]
    return " + ".join(rendered) if rendered else None


def _law_at(read: Mapping[str, Any], law: Any) -> Any:
    """A value a row carries as its law (read at the point) or as itself: a
    number (`{"value": n}`, `phases._row_laws`) or a truth."""
    if isinstance(law, sp.Basic):
        return read[law]
    if isinstance(law, dict):
        return law["value"]
    return law


def _access_at(access: Optional[Mapping[str, Any]],
               terms: Optional[List[Mapping[str, Any]]], read: Mapping[str, Any],
               at: Any, cnt: int,
               written: Optional[List[Mapping[str, Any]]] = None
               ) -> Optional[Mapping[str, Any]]:
    """A row's access facts at a point: the walk's, except what moves with
    the workload — the activation-read terms (`terms`, their laws: whether
    the cache law takes an input's span over is the point's), the
    activation-write terms (`written`: the region a fused boundary rewrote
    grows with the workload like the tensor it publishes), and a lookup
    table's weight bytes — the rows its lookups gather move with the batch
    — which are one instance's share of the row law's weight read there."""
    if not access:
        return access
    # The cache read and the operand conversion are one instance's share of
    # the row law's at this point, like a gathered table's weight read.
    out = {**access,
           "kv_bytes": at.kv_read / max(cnt, 1),
           "convert_bytes": at.convert / max(cnt, 1)}
    if terms is not None:
        out["in_terms"] = [
            [index, float(_law_at(read, term["fanout"])), width,
             None if _law_at(read, term["spanned"])
             else float(_law_at(read, term["elements"]))]
            for (index, _fanout, width, _elements), term
            in zip(access["in_terms"], terms, strict=True)]
    if written is not None:
        out["out_terms"] = [
            [width, None if _law_at(read, term["spanned"])
             else float(_law_at(read, term["elements"]))]
            for (width, _elements), term
            in zip(access["out_terms"], written, strict=True)]
    if access.get("weight_gathered") and access.get("weight_bytes") is not None:
        out["weight_bytes"] = int(at.weight_read // max(cnt, 1))
    return out


def _display_at(book: LawBook, read: Mapping[str, Any], row: Mapping[str, Any],
                at: Any, flops_law: str, point_at: Mapping[str, int]) -> Dict[str, Any]:
    """What the row shows at a point (`read`), off its `point_law`, beside
    its numbers and key columns: its FLOPs and access popovers, a
    collective's buffer shapes, a fused kernel's absorbed parts' shapes.
    `flops_law` is the FLOPs identity there, and
    `point_at` the point's counts."""
    point = row["point_law"]
    cnt = int(row["count_per_step"])
    out: Dict[str, Any] = {}
    flops = point["flops"]
    record = flops["record"]
    # The popovers' intermediate quantities at this point (`FormulaDefs`),
    # off the row's own laws read here.
    defs = FormulaDefs(dims=point["dims"], definitions=point["definitions"],
                       point=point_at,
                       value=read.__getitem__,
                       kv_law=row["roofline_expr"]["kv_read"],
                       kv_total=at.kv_read, count=cnt)
    out["flops_formula"] = defs.attach(_flops_formula_for_row(
        flops_law,
        read[flops["one"]],
        None if record is None else
        {key: read[law] for key, law in record.items()},
        cnt=cnt, result_value=at.flops))
    # The access popover's shape values at this point: its inputs as the
    # access record priced them (`access_in`), its outputs off the walked
    # shape (`projection.access._formula_display`).
    wire = point["wire"]
    ins = (_tensors(read, point["access_in"])
           if point["access_in"] is not None else [])
    access = _access_at(point["access"], point["in_terms"], read, at, cnt,
                        point["out_terms"])
    out["access_bytes_formula"] = defs.attach(_access_bytes_formula_for_row(
        access,
        shape_in_vals=[_shape_text([t]) for t in ins],
        # One entry per tensor written, the main output first (None where
        # the kernel has none, a parallel fusion's), as the walk lists them.
        shape_out_vals=([None if wire is None or wire["out"] is None
                         else _shape_text([[read[_t] for _t in wire["out"]]])]
                        + ([] if wire is None else
                           [_shape_text([t]) for t in _tensors(read, wire["ports"])])),
        stored_read=((at.weight_read + at.metadata)
                     if point.get("saturating") else None),
        cnt=cnt, result_value=at.access))
    comm = point["comm"]
    if comm is not None:
        out["comm_collective_in_shape"] = [
            int(v) for v in [read[_t] for _t in comm["collective_in"]]]
        out["comm_collective_out_shape"] = [
            int(v) for v in [read[_t] for _t in comm["collective_out"]]]
    badge = row.get("source_badge") or {}
    if badge.get("absorbed"):
        out["source_badge"] = {**badge, "absorbed": [
            {**part, "shape_out": (None if part["shape_out_law"] is None
                                   else _floor([read[_t] for _t in part["shape_out_law"]]))}
            for part in badge["absorbed"]]}
    return out


def _point_laws(row: Mapping[str, Any], *, breakdown: bool) -> Iterator[Any]:
    """Every law in the workload's counts a row carries beside its roofline
    that a reading at a point reads: its shape's (`shape_law`), and what its
    key is read off (`point_law`: the walked shape, the node's dims, a
    collective's payload) and its access facts' read terms — and, with
    `breakdown`, what it shows (the FLOPs formula's per-instance law and
    record, the access popover's input shapes, a collective's buffers, a
    fused kernel's absorbed parts' outputs) and what its key is read off
    (`calibration_overlay.key_columns_at`)."""
    law = row.get("shape_law")
    if law is not None:
        for key in ("in", "ports", "in_full"):
            for axes in law[key] or ():
                yield from axes
        for key in ("out", "out_full"):
            if law[key] is not None:
                yield from law[key]
    point = row["point_law"]
    wire = point["wire"]
    if wire is not None:
        for axes in (*wire["in"], *wire["ports"]):
            yield from axes
        if wire["out"] is not None:
            yield from wire["out"]
    comm = point["comm"]
    yield from (point.get("routed") or {}).values()
    for key in ("in_terms", "out_terms"):
        for term in point[key] or ():
            for law in term.values():
                if isinstance(law, sp.Basic):
                    yield law
    if not breakdown:
        return
    for value in (point["dims"] or {}).values():
        if not isinstance(value, dict):
            yield value
    for definition in (point["definitions"] or {}).values():
        for value in definition["bindings"].values():
            if not isinstance(value, dict):
                yield value
    if comm is not None:
        yield from comm["payload"]
    for axes in point["access_in"] or ():
        yield from axes
    flops = point["flops"]
    yield flops["one"]
    yield from (flops["record"] or {}).values()
    if comm is not None:
        for key in ("collective_in", "collective_out"):
            yield from comm[key]
    for part in (row.get("source_badge") or {}).get("absorbed") or ():
        if part["shape_out_law"] is not None:
            yield from part["shape_out_law"]


def rows_at(rows: List[Dict[str, Any]], point: Point, *, cell: Any,
            peak_flops: Optional[float], peak_bw: Optional[float],
            fabric: Fabric, breakdown: bool,
            launch_s: float = 0.0, costs: bool = True) -> List[Dict[str, Any]]:
    """Each row read at the point `point` moves it to — the workload counts
    it sets by name (`{"B": 8}`, `{"s_input": 4096, "B": 1}`), the rest the
    row's own — on the card `peak_*`, `fabric` and `launch_s` describe: its
    numbers there (`point_columns`), and — where the point is not the one
    the row was walked at — its shapes, every axis its `shape_law` there, so a reader
    that keys kernels by them reads the rows a walk there would have emitted
    (`calibration_overlay.point_keys`, which refuses a measurable row
    without a law, `shape_at_batch: False`), and the absence of its
    measurement identity (module docstring). With `breakdown` the row also
    shows what a walk there shows (its popovers, a collective's per-phase
    rows and buffer shapes, a fused kernel's absorbed parts' shapes, what
    its key is read off); without it, it carries none of what it showed at
    the walked point, and its key columns are read where it is keyed
    (`calibration_overlay.key_columns`). The
    numbers are the same either way. `costs=False` is the same existing road
    with the row's SOL numerators and times left at the walk's anchor: a key
    table needs the moved shapes, dtype, dispatch and launch geometry, not a
    second per-point cost evaluation. `cell` is the walk the rows come from
    (`pipeline.AnalysisResult`), whose book holds their laws compiled."""
    book = cell.laws.book
    # The row's laws are in every workload count; another point is the
    # row's own with the point's counts moved — except a count its walk took
    # as a number (`Workload.walked_as_number`: a verify width of 1), which
    # no law reads, so moving it would read the row at a point it was not
    # walked for. That is another walk, refused here.
    count_of = {symbol.name: what
                for what, symbol in Workload.symbols().items()}
    for r in rows:
        own = r["point_law"]["point"]
        moved = sorted(
            name for name, value in point.counts.items()
            if name in own and own[name] != value
            and Workload.walked_as_number(count_of.get(name, name), own[name]))
        if moved:
            raise ValueError(
                f"rows walked with {moved} as numbers cannot be read at "
                f"{ {n: point.counts[n] for n in moved} }: a step there is "
                f"another walk")
    at_of = [{**r["point_law"]["point"], **point.counts} for r in rows]
    asks = [at != r["point_law"]["point"]
            or (breakdown and r["point_law"]["comm"] is not None)
            for r, at in zip(rows, at_of, strict=True)]
    # The laws of the rows that read any, each point's in one call.
    reading = Point.laws_by_point(book, (
        (at, law) for r, at, ask in zip(rows, at_of, asks, strict=True) if ask
        for law in _point_laws(r, breakdown=breakdown)))
    out: List[Dict[str, Any]] = []
    for r, point_at, ask in zip(rows, at_of, asks, strict=True):
        point = r["point_law"]
        comm = point["comm"]
        moved = point_at != point["point"]
        read = reading(point_at) if ask else None
        # Without breakdown, what the row showed at the walked point is not
        # this point's.
        row = dict(r) if breakdown else bare_row(r)
        if costs:
            card = RowCard.of(r, peak_flops=peak_flops, peak_bw=peak_bw,
                              fabric=fabric, launch_s=launch_s)
            at = book.row_at(r, point_at, card)
            row.update(point_columns(
                at, r, card, breakdown=breakdown,
                t_sym=comm["symbolic"] if comm and breakdown else None,
                t_sub=(payload_substituted([read[law] for law in comm["payload"]],
                                           comm["dtype_bytes"])
                       if comm and breakdown else None)))
        else:
            row["t_sol_s"] = 0.0
        if moved:
            row.update(_shape_columns_at(read, r))
            if point.get("routed") is not None:
                # The routed boundary's rows there (`phases._routed_columns`);
                # its bytes per routed row are the same at every point.
                row["routed"] = {**r["routed"], **{
                    key: as_number(read[law])
                    for key, law in point["routed"].items()}}
            # The FLOPs identity there: the walk's own read at the point where
            # it reads the workload's counts, else every point's.
            flops_law = (point["flops"]["law"] if point["flops"]["identity"] is None
                         else Point(point_at).identity(point["flops"]["identity"]))
            if breakdown:
                row.update(key_columns_at(read, r))
                row.update(_display_at(book, read, row, at, flops_law, point_at))
            else:
                # Read where the row is keyed (`calibration_overlay.
                # key_columns`), by a reader that asks for its key.
                row.pop("node_dims", None)
                if comm is not None:
                    row.pop("comm_payload_shape", None)
            # The laws are the workload's; the row now stands at `point_at`.
            row["point_law"] = {
                **point, "point": point_at,
                "flops": {**point["flops"], "law": flops_law},
                "access": (
                    _access_at(point["access"], point["in_terms"],
                               read, at, int(r["count_per_step"]),
                               point["out_terms"])
                    if costs else point["access"])}
            for key in ("shape_json", "dtype_env_json", "signature"):
                row.pop(key, None)
            # A measurement is of one key: at another batch the row has
            # none until its own key is looked up (`apply_measured`).
            row["t_measured_s"] = None
            for key in ("measured_state", "measured_key", "measured_reason"):
                row.pop(key, None)
        out.append(row)
    return out


def _stage_row(book: LawBook, law: Mapping[str, Any],
               counts: Mapping[str, int], card: RowCard) -> Dict[str, Any]:
    """A pipeline stage's row (its law only, `row_law`) at a point: the
    numbers the step adds."""
    at = book.row_at(law, {**law["point"], **counts}, card)
    return {"flops_per_rank": at.flops, "access_bytes_per_rank": at.access,
            "act_out_bytes_per_rank": at.act_out,
            "t_compute_s": at.times.t_compute, "t_memory_s": at.times.t_memory,
            "t_comm_s": at.times.t_comm, "t_launch_s": at.times.t_launch,
            "t_op_s": at.times.t_op}


def _speculation_of(binding: Dict[str, Any]) -> Speculation:
    """The speculation a phase's binding states, read the way the phase
    itself read it.

    `is None`, not `or`: an acceptance rate of 0.0 is a LEGAL binding — every
    draft token rejected, so a verify forward keeps the one token the step
    would have decoded anyway and `1 + α·K` is 1. Read with `or`, that zero
    became the default 1.0, every token accepted, and the two readers of one
    binding disagreed by `1 + K`: at K=3 and batch 8 the phase published 8
    tok/step where this re-read published 32. The phase reads the field with
    no fallback at all (`phases.py`, `inputs.acceptance_rate`); a binding
    that states no rate is the unspeculated default.
    """
    alpha = binding.get("acceptance_rate")
    return Speculation.verifying(binding.get("s_verify") or 1,
                                 1.0 if alpha is None else alpha)
def phase_at(phase: Dict[str, Any], point_rows: List[Dict[str, Any]],
             point: Point, peak_flops: Optional[float],
             peak_bw: Optional[float], fabric: Fabric, *, cell: Any,
             breakdown: bool,
             draft_rows: Optional[List[Dict[str, Any]]] = None,
             launch_s: float = 0.0,
             ) -> Dict[str, Any]:
    """The phase at the point `point` moves it to (its batch, `B`) on the
    card, summed from its rows there (`point_rows`, and a draft's one
    forward's, `draft_rows`) — read at the point already (`rows_at`), as a
    sweep's point reads them once (`capacity._point_rows`). `cell` is the
    walk the phase's rows come from, whose book holds their laws and the
    other pipeline stages' and the module table's. With
    `breakdown` the phase shows its rows' table there (`flops_by_module`,
    its `op_table_total`) and the formulas written from it, as the walk
    does; without it, it carries none of the walked point's — its numbers
    the same either way."""
    if not phase or not phase.get("batch"):
        return phase
    book, counts = cell.laws.book, point.counts
    batch = int(counts["B"])
    # pp>1 publishes the pipeline step: Σ every stage + its exit hop. `rows`
    # are the bound stage's (the per-rank account); the step's times come
    # from every stage's row laws, re-priced here the same way.
    pipeline = phase.get("pipeline_time")
    # A phase whose step time was withheld (no admissible batch on its card)
    # stays withheld here: its bytes and FLOPs re-price, its time does not.
    withheld = phase.get("time_unpublished_reason") or ""

    stages = None
    if pipeline is not None:
        stages = [
            point_rows if s == pipeline["stage"] else
            [_stage_row(book, law, counts,
                        RowCard.of(law, peak_flops=peak_flops,
                                   peak_bw=peak_bw, fabric=fabric,
                                   launch_s=launch_s))
             for law in laws]
            for s, laws in enumerate(pipeline["stage_laws"])]

    # A counted speculative draft is part of the step at every batch: its one
    # forward's rows rescale at the draft's own tokens per sequence, and the
    # step runs `forwards` of them.
    draft_wire = phase.get("speculative_draft")
    draft_cost = None
    if draft_wire:
        if not draft_rows:
            raise ValueError(
                "the decode phase counts a speculative draft but its rows "
                "were not passed to the rescale")
        draft = SpeculativeDraft.from_wire(draft_wire)
        draft_cost = draft.price(draft_rows)
    step = StepSum.of(point_rows, stages=stages,
                      rank_stage=pipeline["stage"] if pipeline else None,
                      draft_rows=draft_cost.rows if draft_cost else (),
                      draft_forwards=draft_cost.draft.forwards if draft_cost else 0)

    def column(key: str, *, account: bool = False) -> Any:
        return step.total(lambda t: t.row[key], account=account)

    total_flops = column("flops_per_rank", account=True)
    total_access = column("access_bytes_per_rank", account=True)
    total_write = column("act_out_bytes_per_rank", account=True)
    total_compute = column("t_compute_s")
    total_memory = column("t_memory_s")
    total_comm = column("t_comm_s")
    total_op = column("t_op_s")
    by_bound = step.by_bound(lambda t: RowTimes.of_row(t.row))
    per_stage = ([StepSum.of(stage_rows).total(lambda t: t.row["t_op_s"])
                  for stage_rows in stages] if stages is not None else [])

    # Nothing is measured here: every term of the step waits for the
    # point's measured fold (`apply_measured`), which states the track.
    measured = MeasuredTrack.unread(len(step.terms))

    ai = arithmetic_intensity(total_flops, total_access)
    binding = phase.get("binding") or {}
    speculation = _speculation_of(binding)
    effective, effective_formula = step_tokens(
        phase["phase"], batch, binding["tokens_per_step"], speculation)

    # The table's leaves are the per-op rows, joined by uid — so the table's
    # bytes, times, shapes and popovers at this point are the re-read
    # leaves', not the anchor's carried over. An untimed row (a wrapper)
    # re-reads its shapes off its own law; every row its unsharded FLOPs
    # off its law (`TableIdentity.flops_law`, one call's, times its calls).
    point_leaf = {r.get("uid"): r for r in point_rows if r.get("uid") is not None}
    modules = (phase.get("flops_by_module") or []) if breakdown else []
    # The module table's laws at the point, each point's read once: an
    # untimed row's shape laws (`shape_law`), every row's unsharded and
    # per-rank FLOPs (`TableIdentity.flops_law`) — no others: a leaf's
    # shapes are its row's, and a counted draft's row its price's.
    needs: List[Tuple[Mapping[str, int], Any]] = []
    for m in modules:
        uid, law = m.get("uid"), m.get("shape_law")
        if ((uid is None or uid not in point_leaf)
                and not (m.get("op") == "<speculative_draft>" and draft_cost is not None)
                and law is not None
                and {**law["point"], **counts} != law["point"]):
            at = {**law["point"], **counts}
            needs.extend((at, text) for key in ("in", "in_full")
                         for axes in law[key] for text in axes)
            needs.extend((at, text) for key in ("out", "out_full")
                         if law[key] is not None for text in law[key])
        flops = m.get("flops_law")
        if flops is not None:
            needs.extend(({**flops["point"], **counts}, text)
                         for text in (flops["full"], flops.get("rank"))
                         if text is not None)
    read = Point.laws_by_point(book, needs)
    rescaled_modules: List[Dict[str, Any]] = []
    for m in modules:
        m = dict(m)
        uid = m.get("uid")
        if uid is not None and uid in point_leaf:
            m = table_row(point_leaf[uid], TableIdentity.of(m))
            m["flops_law"] = {**m["flops_law"],
                              "point": point_leaf[uid]["point_law"]["point"]}
        elif m.get("op") == "<speculative_draft>" and draft_cost is not None:
            # The draft's one row, at this point; its unsharded FLOPs off
            # its law, below, like every other row's.
            m.update({key: value for key, value in draft_cost.module_row().items()
                      if key not in ("flops_full", "flops_law")})
        elif (m.get("shape_law") is not None
              and {**m["shape_law"]["point"], **counts}
              != m["shape_law"]["point"]):
            law = m["shape_law"]
            point_at = {**law["point"], **counts}
            for key in ("shape_in", "shape_out", "shape_in_full", "shape_out_full"):
                m.pop(key, None)
            at = read(point_at)

            def axes_at(axes: Sequence[Any]) -> List[Any]:
                return [_exact_number(v) for v in [at[_t] for _t in axes]]

            m.update(shape_columns(
                [axes_at(axes) for axes in law["in"]],
                axes_at(law["out"]) if law["out"] is not None else None,
                [axes_at(axes) for axes in law["in_full"]],
                axes_at(law["out_full"]) if law["out_full"] is not None else None))
            m["shape_law"] = {**law, "point": point_at}
        flops = m.get("flops_law")
        if flops is not None:
            flops_at = {**flops["point"], **counts}
            at = read(flops_at)
            m["flops_full"] = round(float(at[flops["full"]])
                                    * int(flops["calls"]))
            if flops.get("rank") is not None:
                m["flops_per_rank"] = round(
                    float(at[flops["rank"]]) * int(flops["calls"]))
            m["flops_law"] = {**flops, "point": flops_at}
        rescaled_modules.append(m)
    # Parents were Σ-children by construction; re-sum them over the
    # rescaled leaves, then rebuild the subtree rollups and parent
    # popovers for this batch.
    if rescaled_modules:
        for key in ("flops_per_rank", "flops_full"):
            _resum_module_parents(rescaled_modules, key)
    if draft_cost is not None:
        phase = {**phase, "speculative_draft": draft_cost.wire(
            step_t_sol=None if withheld else total_op,
            times_published=not withheld)}
    t_sol_context = (
        f"pp={pipeline.get('stages')}: one microbatch's latency — "
        f"Σ every stage's roofline + inter-stage hops; the steady-state "
        f"pipeline bubble is not modelled" if pipeline is not None else "")
    anchor_flops = (phase.get("formulas") or {}).get("flops") or {}
    out = {
        **({key: value for key, value in phase.items()
            if key not in ("flops_by_module", "formulas", "op_table_total")}
           if not breakdown else phase),
        "batch": batch,
        "note": phase_note(phase["phase"], phase["seq_effective"], batch,
                           phase["note_detail"]),
        "flops_per_step": total_flops,
        "mem_read_bytes": total_access - total_write,
        "mem_write_bytes": total_write,
        "total_bytes": total_access,
        "arithmetic_intensity": ai,
        "t_compute_s": total_compute,
        "t_memory_s": total_memory,
        "t_comm_s": total_comm,
        "t_sol_s": total_op,
        "t_sol_by_bound": by_bound,
        "t_sol_by_bound_share": RowRoofline.shares(by_bound),
        # Measured coverage is the bound stage's alone: at pp no measured
        # step is published (the anchor's rule).
        "t_measured_s": measured.t_s,
        "measured_rows": measured.measured_rows,
        "total_rows": measured.total_rows,
        "coverage": measured.coverage(),
        **({"flops_by_module": rescaled_modules} if breakdown else {}),
        "binding": {**binding, "effective_tokens_per_step": effective,
                    "effective_tokens_formula": effective_formula},
        # The sums the times were published from, at this batch / card —
        # draft included, as at the anchor.
        **({"pipeline_time": {
            **pipeline, "per_stage": per_stage,
            "t_compute": total_compute, "t_memory": total_memory,
            "t_comm": total_comm, "t_sol": total_op,
            "t_sol_by_bound": dict(by_bound)}}
           if pipeline is not None else {}),
        # The popovers the walk writes, written at this point by the same
        # function; the module table's expansion below fills the FLOPs and
        # t_sol lines as it does at the walk.
        **({"formulas": phase_roofline_formulas(
            flops={key: anchor_flops[key] for key in ("symbolic", "substituted")
                   if key in anchor_flops},
            flops_per_step=total_flops,
            mem_read=total_access - total_write, mem_write=total_write,
            t_sol=None if withheld else total_op,
            t_compute=total_compute, t_memory=total_memory, t_comm=total_comm,
            unpublished_reason=withheld, t_sol_context=t_sol_context)}
           if breakdown else {}),
    }
    if withheld:
        # Withheld before the table is enriched, as the walk withholds a
        # phase before its table is (`capacity._capacity_pending`): the
        # enrichment writes no time a withheld step does not have.
        out.update({
            "t_compute_s": None, "t_memory_s": None, "t_comm_s": None,
            "t_sol_s": None, "t_sol_by_bound": None,
            "t_sol_by_bound_share": None, "t_measured_s": None,
        })
        if pipeline is not None:
            out["pipeline_time"] = {
                key: value for key, value in pipeline.items()
                if key in ("stages", "stage", "stage_laws")}
    if breakdown and rescaled_modules:
        # Rebuild this batch's subtree rollups, parent popovers and the
        # phase-formula component sums over the rescaled rows.
        enrich_module_breakdown(out)
    if "throughput" in phase:
        # Tokens over this point's time, for the same groups and GPUs.
        out["throughput"] = phase_throughput(
            out, unique_tokens_multiplier=phase["throughput"]["groups"],
            num_gpus=phase["throughput"]["num_gpus"])
    return out


def _resum_module_parents(rows: List[Dict[str, Any]], key: str) -> None:
    """A parent's `key` is its subtree's sum (the post-tree builder's
    convention) — recomputed after a batch rescale rewrites the leaves.
    The parent rows' own entries are the ANCHOR batch's stale sums: zero
    them before rolling, or they are counted a second time."""
    tmp = "__swept_subtree_sum"
    parents = set()
    for i, row in enumerate(rows):
        if i + 1 < len(rows) and int(
                rows[i + 1].get("depth") or 0) > int(row.get("depth") or 0):
            parents.add(i)
            row[key] = 0
    _rollup_column(rows, key, tmp)
    for i, row in enumerate(rows):
        if i in parents:
            row[key] = row[tmp]
        row.pop(tmp, None)


def step_measure(step: StepSum, keyed: Mapping[int, Any], where: Mapping[int, str], *,
                 gpu_sku: str, framework: str, binding: Any,
                 written: Dict[Any, Tuple[str, Any]],
                 include_cadence: bool = False,
                 as_object: bool = False) -> Dict[str, Any]:
    """One measured total's plain data: the `auto_sweep.PointMeasure` of
    `step` — its formula, written by the step's one aggregation
    (`StepSum.measured_law`), and per term its key at the point, launches,
    and why it has none — and where each term's row lives (`[list, uid]`).

    `keyed` holds each walked row's `PointRow` by the row's id, `where` the
    row list it sits in. A term with no `PointRow` is a pipeline stage not
    walked here: unplaced, named by its stage. `written` holds the formulas
    of the steps already measured, by the step's shape (each term's weight,
    place and whether it costs nothing): the points of one walk share it,
    and their formula is one text. With `as_object`, `measure` is the
    `PointMeasure` itself: an in-process producer (the serving-law key table)
    writes it directly, without a JSON round trip."""
    shape = (step.rank_stage, tuple(
        (t.weight, t.stage, t.draft, t.zero_cost()) for t in step.terms))
    if shape not in written:
        written[shape] = step.measured_law()
    text, _symbols = written[shape]
    keys, launches, unkeyed, labels = ([] for _ in range(4))
    reasons, aggregation_ids, inputs, zero, terms = ([] for _ in range(5))
    cadences = []
    for t in step.terms:
        z = t.zero_cost()
        point = keyed.get(id(t.row))
        zero.append(z)
        if point is None:
            keys.append(None)
            launches.append(0.0)
            unkeyed.append(MEASURED if z else UNPLACED)
            labels.append(f"pipeline stage {t.stage}")
            reasons.append(ZERO_LAUNCH if z else PIPELINE_STAGE)
            aggregation_ids.append(f"pipeline stage {t.stage}")
            inputs.append(None)
            terms.append(None)
            cadences.append(None)
            continue
        looked_up = not z and not point.declined and point.key is not None
        keys.append(Key(**asdict(point.key)) if looked_up else None)
        launches.append(point.billed)
        unkeyed.append(MEASURED if z else DECLINED if point.declined
                       else UNPLACED if point.key is None else None)
        labels.append(point.op)
        reasons.append(point.placement_reason if z else
                       point.declined if point.declined else
                       point.placement_reason)
        aggregation_ids.append(point.aggregation_id)
        cadences.append(point.cadence)
        # A measured latency is of the cell's launch, which on a routed
        # buffer is the padded geometry (`RoutedLaunch`): its gate divides
        # the numerators AT that geometry, the ones the plan wrote with the
        # measurement (`PointRow.geometry_flops`), not the law's rows.
        anchor = dispatch_source_anchor(binding, point.dispatch)
        inputs.append({"flops": int(point.geometry_flops or 0),
                       "access": int(point.geometry_access_bytes or 0),
                       "anchor": anchor,
                       "dispatch": (None if point.dispatch is None else
                                    point.dispatch.as_dict())}
                      if looked_up else None)
        terms.append([where[id(t.row)], t.row.get("uid")])
    measure = PointMeasure(
        tuple(keys), tuple(launches), tuple(unkeyed), tuple(labels), tuple(reasons),
        MeasuredFormula(text, tuple(zero)),
        context={"gpu": gpu_sku, "framework": framework},
        inputs=tuple(inputs),
        aggregation_ids=tuple(aggregation_ids))
    return {"measure": measure if as_object else measure.to_json(),
            "terms": terms,
        **({"cadences": tuple(cadences)} if include_cadence else {})}


def point_measures(pending: Any, rows: Mapping[str, List[Dict[str, Any]]],
                   keyed: Mapping[str, List[Any]], phases: Mapping[str, Any], *,
                   gpu_sku: str, include_cadence: bool = False,
                   as_objects: bool = False
                   ) -> Dict[str, Dict[str, Any]]:
    """Each phase of one point, before measurement, as plain data
    (`step_measure`): one measured total per phase — its rows, a pipeline's
    other stages (unplaced: only this rank's stage was walked here) and a
    speculative draft's forwards, as the step adds them up (`StepSum`).
    Theory: nothing here depends on what is measured or how it is selected,
    so a cached point serves every selection mode and every Kernel Bench
    state.

    `rows` are the point's rows and `keyed` their `PointRow`s per list
    (`capacity._point_rows`, or an estimate's own walk), `phases` its
    phases."""
    walks = pending.walk_of_rows
    by_id = {id(r): pt for name in keyed for r, pt in zip(rows.get(name) or [], keyed[name],
                                                          strict=True)}
    where = {id(r): name for name in keyed for r in rows.get(name) or []}
    out: Dict[str, Dict[str, Any]] = {}
    for phase, name in (("prefill", "per_op_timing"), ("decode", "per_op_timing_decode")):
        if name not in walks or not phases.get(phase):
            continue
        ph = phases[phase]
        pipeline = ph.get("pipeline_time")
        draft = ph.get("speculative_draft")
        main = rows.get(name) or []
        step = StepSum.of(
            main if pipeline is None else (),
            stages=(None if pipeline is None else
                    [main if s == pipeline["stage"] else laws
                     for s, laws in enumerate(pipeline["stage_laws"])]),
            rank_stage=pipeline["stage"] if pipeline is not None else None,
            draft_rows=(rows.get("per_op_timing_draft") or []) if draft else (),
            draft_forwards=int(draft["forwards"]) if draft else 0)
        walk = walks[name]
        out[phase] = {**step_measure(step, by_id, where, gpu_sku=gpu_sku,
                                     framework=walk.deployment.framework,
                                     binding=walk.deployment.framework_binding,
                                     written=pending.measured_formulas,
                                     include_cadence=include_cadence,
                                     as_object=as_objects),
                      "phase": phase}
    return out


def _measured_formula_weight(formula: MeasuredFormula, index: int,
                             count: int) -> sp.Expr:
    """The structural coefficient of term ``index`` in a measured formula.

    Every other launch time is bound to zero and this term's launch and time to
    one.  The value stays symbolic (a Rational pipeline/draft weight, say), so
    duplicate-key merging compares the mathematical law rather than a float
    rendering of it.
    """
    if formula.zero_cost[index]:
        return sp.Integer(0)
    return _formula_term_weight(formula.text, index, count)


@memoize(maxsize=8)
def _read_formula(text: str) -> sp.Expr:
    """A measured formula's text, read once (`read_walk_law`): the chunks of
    one endpoint, and the endpoints of a sweep, write the same text.

    The cache is small because a parsed formula is big — a
    DeepSeek-V3-scale step sum is 2 545 terms over 38 KB of text and
    retains 12.8 MiB of sympy nodes — and because nothing asks for more
    than a handful at a time: a point's weights are read term by term off
    one text (`step_measure` walks one step's terms), so the asks for one
    text are consecutive. Measured over six deployments of three models in
    one process (gpt-oss-20b at EP2 and TP2, Llama-3.2-1B plain, at PP2 and
    at a verify width of 4, Qwen3-30B-A3B at EP2; two contexts x six batch
    points each, chunked): 97 asks, 4 distinct texts, 93 hits — the same
    hits at maxsize 64, 8, 4 and even 1, so the misses are the distinct
    texts and no size above 1 buys anything."""
    return read_walk_law(text)


@memoize(maxsize=1 << 16)
def _formula_term_weight(text: str, index: int, count: int) -> sp.Expr:
    """`_measured_formula_weight` of a term that costs something: a function
    of the text, the term and the term count alone."""
    weight = _read_formula(text).xreplace({
        **{sp.Symbol(f"n_{k}"): sp.Integer(1) for k in range(count)},
        **{sp.Symbol(f"T_{k}"): sp.Integer(1 if k == index else 0)
           for k in range(count)}})
    if weight.free_symbols:
        raise ValueError(
            f"a measured formula has a stray symbol beside its launch terms: {text}")
    return weight


def _same_fold_law(left: ChunkLaunchTerm, right: ChunkLaunchTerm) -> bool:
    return (left.formula_weight == right.formula_weight
            and left.zero_cost == right.zero_cost
            and left.inputs == right.inputs
            and left.context == right.context
            and left.cadence == right.cadence)


def fold_chunk_point_measures(chunk_map: Mapping[str, Any],
                              chunks: Sequence[Mapping[str, Any]]
                              ) -> Dict[str, Any]:
    """Fold one endpoint's per-chunk ``step_measure`` records.

    ``chunk_map`` is the endpoint record emitted beside a chunked campaign
    (``stats['prefill_chunk_map']``), not the globally deduped job list.  Each
    chunk supplies its own exact ``PointMeasure``; only explicit chunk records
    participate.  A one-chunk map is returned unchanged, preserving today's
    short-context behavior.

    For a multi-chunk endpoint, each term becomes a backend-owned
    ``ChunkLaunchTerm``.  Endpoint repetition is read from the map's endpoint
    batch divided by that chunk's launch batch.  ``merge_chunk_launch_terms``
    then adds launches only under the exact-key/weight/input/context/cadence
    invariant.  Context-dependent rows keep their distinct keys and therefore
    remain separate terms.  No endpoint-sized or old whole-sequence key is read.
    """
    metadata = list(chunk_map.get("chunks") or ())
    if not metadata:
        raise ValueError("a chunk fold needs at least one mapped chunk")
    if len(metadata) == 1:
        if len(chunks) != 1:
            raise ValueError("a one-chunk endpoint has exactly one chunk measure")
        return dict(chunks[0])
    if len(chunks) != len(metadata):
        raise ValueError(
            f"the chunk map has {len(metadata)} chunks, but {len(chunks)} "
            "point measures were supplied")
    expected = list(range(len(metadata)))
    if [int(chunk.get("index", -1)) for chunk in metadata] != expected:
        raise ValueError("chunk measures must follow the endpoint chunk map exactly")

    endpoint_batch = int(chunk_map["endpoint"]["batch"])
    records: List[Tuple[ChunkLaunchTerm, Mapping[str, Any], int, int]] = []
    for chunk_index, (metadata_chunk, measure_chunk) in enumerate(
            zip(metadata, chunks, strict=True)):
        measure = PointMeasure.from_json(measure_chunk["measure"])
        term_rows = list(measure_chunk.get("terms") or ())
        if len(term_rows) != len(measure.keys):
            raise ValueError("a chunk measure states each term's row exactly once")
        cadences = measure_chunk.get("cadences")
        if cadences is None or len(cadences) != len(measure.keys):
            raise ValueError(
                "a multi-chunk fold states each term's cadence provenance")
        launch_batch = int(metadata_chunk["launch_batch"])
        if endpoint_batch < 1 or launch_batch < 1 or endpoint_batch % launch_batch:
            raise ValueError(
                "endpoint batch must be a whole number of chunk launch batches "
                f"(endpoint={endpoint_batch}, launch={launch_batch})")
        repeat = endpoint_batch // launch_batch
        context = measure.context
        for term_index, key in enumerate(measure.keys):
            records.append((
                ChunkLaunchTerm(
                    # A row without a physical key gets a private identity: it
                    # is preserved for all-or-nothing coverage, never merged.
                    key=(("unkeyed", chunk_index, term_index),)
                    if key is None else key,
                    launches=float(measure.launches[term_index]) * repeat,
                    formula_weight=_measured_formula_weight(
                        measure.formula, term_index, len(measure.keys)),
                    zero_cost=bool(measure.formula.zero_cost[term_index]),
                    inputs=(measure.inputs[term_index] if measure.inputs else None),
                    context=(measure.contexts[term_index]
                             if measure.contexts else context),
                    cadence=cadences[term_index]),
                measure_chunk, term_index, term_index))

    merged = merge_chunk_launch_terms(term for term, *_ in records)
    keys, launches, unkeyed, labels = ([] for _ in range(4))
    reasons, aggregation_ids, inputs, zero, terms = ([] for _ in range(5))
    text_parts: List[str] = []
    contexts = []
    # Match each result back to the first source record.  Unsafe groups retain
    # their own records term-for-term; safe groups intentionally collapse to
    # the first graph-row provenance.
    used = set()
    for out_index, out in enumerate(merged):
        source = next(
            (record for record in records
             if id(record) not in used and record[0].key == out.key
             and _same_fold_law(record[0], out)),
            None)
        if source is None:
            raise ValueError("a merged chunk term lost its source provenance")
        used.add(id(source))
        _term, measure_chunk, measure_term, _record_index = source
        measure = PointMeasure.from_json(measure_chunk["measure"])
        keys.append(None if measure.keys[measure_term] is None else out.key)
        launches.append(out.launches)
        unkeyed.append(measure.unkeyed[measure_term])
        labels.append(measure.labels[measure_term])
        reasons.append(measure.reasons[measure_term])
        aggregation_ids.append(
            measure.aggregation_ids[measure_term]
            if measure.aggregation_ids else "")
        inputs.append(out.inputs)
        zero.append(out.zero_cost)
        terms.append(measure_chunk["terms"][measure_term])
        contexts.append(out.context)
        if not out.zero_cost:
            base = f"n_{out_index}*T_{out_index}"
            text_parts.append(base if out.formula_weight == 1
                              else f"({sp.sstr(out.formula_weight)})*{base}")
    return {
        "measure": PointMeasure(
            tuple(keys), tuple(launches), tuple(unkeyed), tuple(labels),
            tuple(reasons), MeasuredFormula(" + ".join(text_parts) or "0",
                                            tuple(zero)),
            context=contexts[0] if contexts else None,
            inputs=tuple(inputs), contexts=tuple(contexts),
            aggregation_ids=tuple(aggregation_ids)).to_json(),
        "terms": terms,
    }


def measures_on_card(measures: Mapping[str, Mapping[str, Any]],
                     gpu_sku: str) -> Dict[str, Dict[str, Any]]:
    """A point's measures (`point_measures`) as the same rows on `gpu_sku`
    state them: every key's card (`Key.gpu`) and the selection's card
    (`step_measure`'s context) are `gpu_sku`'s; the shapes, dtype
    environments, launches and formula are the rows' own, which a card of
    the walk's class (`framework_bindings.card_class`) walks alike."""
    out: Dict[str, Dict[str, Any]] = {}
    for phase, m in measures.items():
        pm = PointMeasure.from_json(m["measure"])
        pm = replace(pm, keys=tuple(None if k is None else k._replace(gpu=gpu_sku)
                                    for k in pm.keys),
                     context={**(pm.context or {}), "gpu": gpu_sku},
                     contexts=tuple(
                         {**(context or {}), "gpu": gpu_sku}
                         for context in pm.contexts))
        out[phase] = {**m, "measure": pm.to_json()}
    return out


def selector(registry: Any, *, strategy: str,
             kernel_mix: Optional[Mapping[str, str]] = None,
             timings: Optional[Dict[str, Any]] = None) -> Any:
    """The measured selection over a registry built from one lookup:
    `select(key, context, op, read) -> (one launch's time or None, state,
    the measurement's source and badge)`, through `op_calibration.
    select_time` with the request's strategy and kernel pins, the context's
    framework, on the context's card, with the row's anomaly numerators and
    source anchor (`read`)."""
    gpus: Dict[str, Any] = {}

    def select(key: Key, context: Mapping[str, Any], op: str,
               read: Mapping[str, Any]) -> Tuple[Optional[float], str, Any]:
        started = time.perf_counter()
        gpu = gpus.setdefault(context["gpu"], get_gpu(context["gpu"]))
        pin = (kernel_mix or {}).get(op) or None
        dispatch = read.get("dispatch")
        # An audited vLLM anchor cannot prove a user's arbitrary pin. It is
        # carried only when the pin names the active declared identity.
        anchor = read["anchor"]
        if (pin is not None
                and not (dispatch and dispatch.get("state") == "declared"
                         and dispatch.get("impl_id") == pin)):
            anchor = None
        measured, state = select_time(
            registry, PhysicalKey(*key), strategy=strategy,
            framework=context["framework"],
            kernel_impl=pin,
            kernel_dispatch=read.get("dispatch"),
            current_peak_tflops_at=lambda dtype: (peak_flops_for_dtype(gpu, dtype) or 0.0) / 1e12,
            current_peak_bw_gbps=(peak_hbm_bw(gpu) or 0.0) / 1e9,
            flops_for_anomaly=(None if read.get("flops") is None
                               else float(read["flops"])),
            access_bytes_for_anomaly=(None if read.get("access") is None
                                      else float(read["access"])),
            source_anchor=anchor)
        if measured is None:
            if timings is not None:
                timings["selector_s"] = (timings.get("selector_s", 0.0)
                                         + time.perf_counter() - started)
            return None, state, None
        if timings is not None:
            timings["selector_s"] = (timings.get("selector_s", 0.0)
                                     + time.perf_counter() - started)
        return measured.t_op_s, state, {"source": measured.source,
                                        "source_badge": measured.source_badge,
                                        "utilization": measured.utilization}
    return select


def _measured_row(row: Mapping[str, Any],
                  folded: Tuple[str, Optional[float], str, Any],
                  indices: Optional[Sequence[int]] = None,
                  folded_all: Any = None) -> Dict[str, Any]:
    """`row` with what its measurement at the point says, as an estimate's
    rows state it: the time, its coverage state and the key (or op) the state
    names, and the measurement's source, badge and MFU / MBU — or, with none selected,
    its theory classification and the miss reason
    (`calibration_overlay.missed_badge`); a declined row its reason."""
    if indices is None or len(indices) == 1:
        st, t, label, detail = folded
    else:
        whole = folded_all
        rows = [whole.rows[i] for i in indices]
        unmeasured = [entry for entry in rows if entry[0] != MEASURED]
        labels = list(dict.fromkeys(entry[2] for entry in rows))
        label = labels[0] if len(labels) == 1 else ", ".join(str(x) for x in labels)
        if unmeasured:
            st, detail = unmeasured[0][0], unmeasured[0][3]
            t = None
        else:
            st = MEASURED
            t = whole.subtotal(indices)
        detail = next((entry[3] for entry in rows
                       if isinstance(entry[3], dict)), None)
    out = {**row, "t_measured_s": t, "measured_state": st, "measured_key": label}
    if isinstance(detail, str):
        out["measured_reason"] = detail
    if st == MEASURED and isinstance(detail, dict):
        util = detail.get("utilization") or {}
        out.update(source=detail["source"], source_badge=detail["source_badge"],
                   measured_mfu=util.get("mfu"), measured_mbu=util.get("mbu"))
    else:
        out["source_badge"] = {**(row.get("source_badge") or {}), **missed_badge()}
    return out


def terms_by_uid(uids: Iterable[Any]) -> Dict[Any, List[int]]:
    """Each uid's term indices, in term order: the one index every measured
    reader takes (`_measured_row`, `_measured_table`, `Folded.subtotal`). A
    uid may price several terms, so its value is always a list."""
    out: Dict[Any, List[int]] = {}
    for k, uid in enumerate(uids):
        if uid is not None:
            out.setdefault(uid, []).append(k)
    return out


def _terms_of_rows(rows: Sequence[Mapping[str, Any]], indices: Iterable[int],
                   terms: Mapping[Any, Sequence[int]]) -> List[int]:
    """The terms under `rows[indices]`, flat — what `Folded.subtotal` sums."""
    return [k for i in indices for k in terms.get(rows[i].get("uid"), ())]


def _measured_table(modules: List[Dict[str, Any]],
                    term_of_uid: Mapping[Any, Sequence[int]],
                    folded: Any) -> List[Dict[str, Any]]:
    """A module table with its leaves' measured times (`_measured_row`) and
    each row's `t_measured_agg`: the one formula's share of the terms under
    it (`Folded.subtotal`) — none unless every one of them is measured."""
    out = [(_measured_row(m, folded.rows[term_of_uid[m["uid"]][0]],
                          term_of_uid[m["uid"]], folded)
            if m.get("uid") in term_of_uid else dict(m)) for m in modules]
    for i, m in enumerate(out):
        depth = int(m.get("depth") or 0)
        under = [] if m.get("uid") not in term_of_uid else list(term_of_uid[m["uid"]])
        for child in out[i + 1:]:
            if int(child.get("depth") or 0) <= depth:
                break
            if child.get("uid") in term_of_uid:
                under.extend(term_of_uid[child["uid"]])
        m["t_measured_agg"] = folded.subtotal(under)
    return out


def _row_sol(rows: Sequence[Mapping[str, Any]],
             terms: Mapping[Any, Sequence[int]]
             ) -> List[Tuple[Sequence[int], float]]:
    """Each folded row's terms and its SOL time (`t_op_s`), what
    `Folded.partial` blends."""
    return [(terms[r["uid"]], float(r.get("t_op_s") or 0.0))
            for r in rows if r.get("uid") in terms]


def apply_measured(point: Dict[str, Any], measures: Mapping[str, Mapping[str, Any]],
                   folds: Mapping[str, MatrixFold]) -> Dict[str, Any]:
    """`point` with its measured track, from the record-level matrix row.

    Each phase's fold is the same matrix path whether it has one Inference
    point or is one row of a Gallery record; this function only unfolds the
    row into the response shapes that point asks for."""
    for phase, m in measures.items():
        folded = folds[phase]
        track = folded.track
        by_list = {name: terms_by_uid(None if t is None or t[0] != name else t[1]
                                      for t in m["terms"])
                   for name in dict.fromkeys(t[0] for t in m["terms"] if t is not None)}
        if m.get("encode") is not None:
            entry = point["encode"]["stacks"][m["encode"]]
            terms = by_list.get("per_op_timing", {})
            entry["per_op_timing"] = [
                _measured_row(r, folded.rows[terms[r.get("uid")][0]],
                              terms[r.get("uid")], folded)
                              if r.get("uid") in terms else r
                              for r in entry["per_op_timing"]]
            entry["flops_by_module"] = _measured_table(entry.get("flops_by_module") or [],
                                                       terms, folded)
            entry["per_op_timing_measured_by_declaration"] = {
                    parent: folded.subtotal(
                        _terms_of_rows(entry["per_op_timing"], indices, terms))
                for parent, indices in declaration_terms(entry["per_op_timing"])}
            partial = folded.partial(_row_sol(entry["per_op_timing"], terms),
                                     entry.get("t_sol"))
            entry.update(t_measured=track.t_s, coverage=track.coverage(),
                         t_partial=None if partial is None else partial["t_s"],
                         partial_measured_share=(None if partial is None
                                                 else partial["measured_share"]))
            continue
        for name, terms in by_list.items():
            if point.get(name):
                point[name] = [
                    _measured_row(r, folded.rows[terms[r.get("uid")][0]],
                                  terms[r.get("uid")], folded)
                               if r.get("uid") in terms else r for r in point[name]]
                point[f"{name}_measured_by_declaration"] = {
                    parent: folded.subtotal(
                        _terms_of_rows(point[name], indices, terms))
                    for parent, indices in declaration_terms(point[name])}
        ph = point["phases"][m["phase"]]
        withheld = bool(ph.get("time_unpublished_reason"))
        main = "per_op_timing" if m["phase"] == "prefill" else "per_op_timing_decode"
        draft_terms = [index
                       for indices in by_list.get("per_op_timing_draft", {}).values()
                       for index in indices]
        draft = ph.get("speculative_draft")
        draft_share = None
        if draft and draft_terms:
            draft_share = None if withheld else folded.subtotal(draft_terms)
            forwards = int(draft["forwards"])
            draft["t_measured_s"] = draft_share
            draft["per_forward"]["t_measured_s"] = (None if draft_share is None
                                                    else draft_share / forwards)
        partial = None if withheld else folded.partial(
            (row for name, terms in by_list.items()
             for row in _row_sol(point.get(name) or [], terms)),
            ph.get("t_sol_s"))
        ph.update({"t_measured_s": None if withheld else track.t_s,
                   "t_partial_s": None if partial is None else partial["t_s"],
                   "partial_measured_share": (None if partial is None
                                              else partial["measured_share"]),
                   "measured_rows": track.measured_rows, "total_rows": track.total_rows,
                   "coverage": track.coverage()})
        if "flops_by_module" in ph:
            modules = _measured_table(ph["flops_by_module"],
                                      by_list.get(main, {}), folded)
            for row in modules:
                if row.get("op") == "<speculative_draft>":
                    row["t_measured_s"] = row["t_measured_agg"] = draft_share
            ph["flops_by_module"] = modules
        if "throughput" in ph:
            ph["throughput"] = phase_throughput(
                ph, unique_tokens_multiplier=ph["throughput"]["groups"],
                num_gpus=ph["throughput"]["num_gpus"])
    return point


def _measured_selector_for(matrix: MeasureMatrix, *, selection_mode: str,
                           measurement_scope: Any = None,
                           calibration_rows: Optional[list] = None,
                           kernel_mix: Optional[Mapping[str, str]] = None,
                           timings: Optional[Dict[str, Any]] = None) -> Any:
    """The selection over a matrix's distinct key/context/op/read questions."""
    _fw, _binding, strategy, reads = _resolve_overlay_context(
        framework=None, selection_mode=selection_mode,
        measurement_scope=measurement_scope, calibration_rows=calibration_rows)
    if not reads or not len(matrix.formulas):
        return None
    started = time.perf_counter()
    keys = [key for key in matrix.keys if key is not None]
    if timings is not None:
        timings["key_count"] = len(keys)
        timings["key_collect_s"] = time.perf_counter() - started
    started = time.perf_counter()
    found = look_up(measurement_scope if measurement_scope is not None else NO_MEASUREMENTS,
                    keys)
    if timings is not None:
        timings["lookup_s"] = time.perf_counter() - started
        timings["lookup_service_s"] = found.service_s
        timings["lookup_network_s"] = max(
            0.0, found.request_s - found.service_s)
    started = time.perf_counter()
    registry = registry_from_lookup([PhysicalKey(*key) for key in found.keys],
                                    found.rows)
    if calibration_rows:
        registry = merge_validated_rows(registry, calibration_rows,
                                        source="request")
    got = selector(registry, strategy=strategy,
                   kernel_mix=kernel_mix if selection_mode == "kernel_mix" else None,
                   timings=timings)
    if timings is not None:
        timings["selector_setup_s"] = time.perf_counter() - started
    return got


def measured_selector(measures: List[PointMeasure], *, selection_mode: str,
                      measurement_scope: Any = None,
                      calibration_rows: Optional[list] = None,
                      kernel_mix: Optional[Mapping[str, str]] = None,
                      timings: Optional[Dict[str, Any]] = None) -> Any:
    """Compatibility for a caller with `PointMeasure`s: their one matrix."""
    return _measured_selector_for(
        MeasureMatrix.of_measures(measures), selection_mode=selection_mode,
        measurement_scope=measurement_scope, calibration_rows=calibration_rows,
        kernel_mix=kernel_mix, timings=timings)


def measure_points(points: List[Dict[str, Any]], *, selection_mode: str,
                   measurement_scope: Any = None,
                   calibration_rows: Optional[list] = None,
                   kernel_mix: Optional[Mapping[str, str]] = None,
                   timings: Optional[Dict[str, Any]] = None) -> None:
    """Fold every point's measured totals through one record-level matrix.

    A canonical Gallery reference names a row of an already-record matrix; a
    live Inference `PointMeasure` becomes a one-row matrix here. Selection,
    total and coverage are therefore the same path at both granularities.
    """
    started = time.perf_counter()
    direct: list[PointMeasure] = []
    pending: list[tuple[dict, str, dict, int, Optional[MeasureMatrix]]] = []
    for pt in points:
        for phase, m in (pt.get("measures") or {}).items():
            record_matrix = m.get("matrix")
            if record_matrix is None:
                measure = m["measure"]
                direct.append(measure if isinstance(measure, PointMeasure)
                              else PointMeasure.from_json(measure))
                pending.append((pt, phase, m, len(direct) - 1, None))
            else:
                row = int(m["row"])
                pending.append((pt, phase, m, row, record_matrix))
    # Direct measures share one matrix; each canonical record contributes its
    # own once. Keep global row offsets while concatenating those groups.
    groups: list[MeasureMatrix] = []
    offsets: dict[int, int] = {}
    if direct:
        direct_matrix = MeasureMatrix.of_measures(direct)
        groups.append(direct_matrix)
        offsets[id(direct_matrix)] = 0
    seen: dict[int, MeasureMatrix] = {}
    for _pt, _phase, _m, _row, record_matrix in pending:
        if record_matrix is not None and id(record_matrix) not in seen:
            seen[id(record_matrix)] = record_matrix
            groups.append(record_matrix)
            offsets[id(record_matrix)] = -1
    matrix = MeasureMatrix.empty()
    before = 0
    for _i, group in enumerate(groups):
        group_id = id(group)
        if offsets.get(group_id) is None or offsets[group_id] == -1:
            offsets[group_id] = before
        matrix = matrix.concat(group)
        before = len(matrix.formulas)
    if timings is not None:
        timings["point_measure_read_s"] = time.perf_counter() - started
        timings["matrix_rows"] = len(matrix.formulas)
    select = _measured_selector_for(
        matrix, selection_mode=selection_mode,
        measurement_scope=measurement_scope, calibration_rows=calibration_rows,
        kernel_mix=kernel_mix, timings=timings)
    if select is None:
        return
    started = time.perf_counter()
    folds = matrix.fold(select)
    for pt, phase, m, row, record_matrix in pending:
        if record_matrix is not None:
            row += offsets[id(record_matrix)]
        apply_measured(pt["sol"], {phase: m}, {phase: folds[row]})
    if timings is not None:
        timings["apply_measured_s"] = time.perf_counter() - started
