"""Views — Pareto frontiers and lines — drawn from a point cloud.

A chart of "points x a few dimensions" is served as one cached cloud and
views of it: some dimensions held at a value, the rest grouped into lines,
each line its Pareto front or its points along an X. The cloud's producer
declares its dimensions; the view is the one place a front is picked.

A point carries BOTH serving phases (`cloud_point`): the decode rung it is
and the prefill of the same deployment. Which phase a view draws is the
view's `phase`; which (phase, form) pairs this cloud can answer at all is
derived from the cloud itself (`view_forms`), not declared per phase.
"""
from __future__ import annotations

from itertools import product
from typing import (Any, Dict, List, Literal, Mapping, Optional, Sequence,
                    Tuple, get_args)


#: Every coordinate a point of any throughput cloud states, whatever producer made
#: it. A producer may add a provenance tag (`row`, for a Compare request), but
#: it may not omit one of these: a view written for one cloud must be askable of
#: another. `ep_load_factor` is here even though it is deliberately NOT in
#: `measure.scope.of_walk` — rho is substituted at the point. `layout` is the
#: deployment itself (`layout_label`): a line grouped by it is one deployment
#: through the batches it serves, which `mode` is not — the one-GPU point is on
#: every mode's ladder, and the Gallery's mixed TP·EP·DP layouts are all
#: `custom`.
PointDim = Literal[
    "mode", "num_gpus", "layout", "gpu_sku", "context", "batch_key",
    "draft_tokens", "acceptance", "ep_load_factor",
    #: The decode-context selected-position skew (`Workload.rho_dcp`) and the
    #: candidate-certificate fallback scenario (`Workload.dcp_failed_owners`).
    #: Both are workload inputs a walk's laws read, so a point states them
    #: beside `ep_load_factor`; the swept grid spans over neither (a sweep
    #: varies topology, not routing or the owner scenario).
    "rho_dcp", "dcp_failed_owners",
]
POINT_DIMS: Tuple[str, ...] = get_args(PointDim)

#: The coordinates a coordinate's value fixes: a deployment (`layout`) is
#: one mode at one GPU count, in every cloud (`point_coordinates` derives the
#: three from the one `ParallelConfig`). Not the other way round — the
#: Gallery's mixed TP·EP·DP layouts are all mode `custom` at one count. A
#: view reads it twice: a line along an axis does not hold what fixes the
#: axis (`view_forms`), and a group by a coordinate states what it fixes.
POINT_FIXES: Mapping[str, Tuple[str, ...]] = {"layout": ("mode", "num_gpus")}

#: Why a grid point has no values. `beyond_max_context` is not here: the
#: existing chart extrapolates past a model's published context and flags it,
#: so those points still have values.
PointRefusal = Literal[
    "layout", "capacity", "card", "not_enumerated", "error",
]
POINT_REFUSALS: Tuple[str, ...] = get_args(PointRefusal)


def _control_order(value: Any) -> Tuple[Any, Any, str]:
    """A control value's display order.

    Numeric coordinates sort numerically. Batch rungs keep grid order
    (`b1, b2, … b1024, max`) rather than lexicographic order, which would put
    `b1024` before `b128`.
    """
    if isinstance(value, str):
        body = value[1:]
        if value.startswith("b") and body.isdigit():
            return 1, int(body), ""
        if value == "max":
            return 1, float("inf"), ""
        return 2, value, ""
    return 0, float(value), ""


def absent_point(coordinates: Mapping[str, Any], refusal: str,
                 reason: str) -> dict:
    """A grid point the deployment cannot serve, with every coordinate and why.

    Absence is data. Dropping the point makes a refusal indistinguishable from
    a rung nobody asked for, which is why a chart region used to be empty
    without a reason.
    """
    unknown = [name for name in coordinates if name not in POINT_DIMS]
    if unknown:
        raise ValueError(f"unknown point coordinate(s) {unknown}; one of "
                         f"{POINT_DIMS}")
    if refusal not in POINT_REFUSALS:
        raise ValueError(f"unknown point refusal {refusal!r}; one of "
                         f"{POINT_REFUSALS}")
    missing = [name for name in POINT_DIMS if name not in coordinates]
    if missing:
        raise ValueError(f"an absent point still states every coordinate; "
                         f"missing {missing}")
    return {**coordinates, "refused": refusal, "reason": reason}


#: The one-GPU deployment's mode coordinate: no parallelism at all.
SINGLE_GPU_MODE = "tp1"


def parallel_mode(parallel: Any) -> str:
    """A deployment's point-cloud mode coordinate — one per deployment, in
    every cloud.

    The four Scaling modes are their canonical `ParallelConfig` shapes. A
    one-GPU deployment is `SINGLE_GPU_MODE`; anything else is `custom`
    rather than a nearest-mode lie. That the one-GPU deployment is also the
    first rung of every mode's ladder is a view rule (`mode_ladders`), not a
    second coordinate.
    """
    if parallel is None:
        return SINGLE_GPU_MODE
    degree = max(1, int(getattr(parallel, "cluster_size", 1) or 1))
    if degree == 1:
        return SINGLE_GPU_MODE
    fields = {
        name: int(getattr(parallel, name, 1) or 1)
        for name in ("dp_attention", "cp", "tp_attention", "tp_dense_ffn",
                     "tp_moe_expert", "ep_moe")
    }
    if (fields["dp_attention"] == degree and fields["ep_moe"] == degree
            and fields["cp"] == 1 and fields["tp_attention"] == 1):
        return "dp_ep"
    if (fields["cp"] == degree and fields["ep_moe"] == degree
            and fields["dp_attention"] == 1
            and fields["tp_attention"] == 1):
        return "cp_ep"
    if (fields["tp_attention"] == degree
            and fields["tp_dense_ffn"] == degree
            and fields["ep_moe"] == degree):
        return "tp_ep"
    if (fields["tp_attention"] == degree
            and fields["tp_dense_ffn"] == degree
            and fields["tp_moe_expert"] == degree
            and fields["ep_moe"] == 1):
        return "tp_tp"
    return "custom"


def layout_label(parallel: Any) -> str:
    """How a chart names a deployment, and the point-cloud `layout`
    coordinate — so one name is one deployment. The degrees above one,
    short enough to sit beside a point (`TP8`, `TP8·EP64·DP8`, `EP32·DP32`;
    one GPU is `TP1`); the GPU count is said beside it.

    The short form leaves out what the common deployments agree on: the
    dense FFN split as attention is, and each expert split as attention is
    when the experts are not spread (`ep_moe` 1) and whole when they are.
    Anything else is named after it — `FFN-TP`, `MoE-TP`, `DCP`, `PP`, data
    replicas `REP`, sequence parallelism `SP`, a context-parallel scheme
    other than PCP, a world's own degree — so two deployments never share a
    name (`test_point_cloud`)."""
    tp = parallel.tp_attention
    cp = f"CP{parallel.cp}" + (
        f"({parallel.cp_scheme.value})"
        if parallel.cp_scheme.value != "pcp" else "")
    parts = [part for part, n in (
        (f"TP{tp}", tp), (cp, parallel.cp),
        (f"EP{parallel.ep_moe}", parallel.ep_moe),
        (f"DP{parallel.dp_attention}", parallel.dp_attention)) if n > 1]
    expert_tp = tp if parallel.ep_moe == 1 else 1
    parts += [part for part, named in (
        (f"FFN-TP{parallel.tp_dense_ffn}", parallel.tp_dense_ffn != tp),
        (f"MoE-TP{parallel.tp_moe_expert}",
         parallel.tp_moe_expert != expert_tp),
        (f"DCP{parallel.dcp}", parallel.dcp > 1),
        (f"PP{parallel.pp}", parallel.pp > 1),
        (f"REP{parallel.dp_replicas}", parallel.dp_replicas > 1),
        ("SP", bool(parallel.sequence_parallel))) if named]
    parts += [f"{world}{n}" for world, n in
              sorted((parallel.world_degrees or {}).items()) if n > 1]
    return "·".join(parts) or "TP1"


def mode_ladders(cloud: Sequence[dict]) -> Dict[Any, Tuple[Any, ...]]:
    """Which modes' ladders each point's mode coordinate belongs to, for a
    view to hold and group by: every mode its own, and the one-GPU
    deployment (`SINGLE_GPU_MODE`) the first rung of every mode that has a
    ladder of its own — a priced point past one GPU in this cloud. A mode
    with none (every degree past one refused, or a dense model's expert
    modes) grows no one-point line; with no ladder at all the one-GPU
    deployment stays its own mode.

    The one statement of the rule, read by every view (`_view_groups`), so
    holding a mode at one GPU finds the one-GPU point and a front over
    several modes draws it once."""
    ladders = tuple(dict.fromkeys(
        p["mode"] for p in cloud
        if p["mode"] != SINGLE_GPU_MODE and "refused" not in p
        and int(p.get("num_gpus") or 0) > 1))
    out: Dict[Any, Tuple[Any, ...]] = {
        p["mode"]: (p["mode"],) for p in cloud}
    if SINGLE_GPU_MODE in out and ladders:
        out[SINGLE_GPU_MODE] = ladders
    return out


def point_coords(cloud: Sequence[dict], dims: Sequence[str]):
    """How a point's coordinate on each dimension reads for holding,
    grouping, and asking whether a phase moves along an axis: its own
    value, except `mode`, where the one-GPU deployment belongs to every
    mode's ladder (`mode_ladders`).

    The one statement of that rule. `_view_groups` holds and groups by it,
    and `view_forms` asks by it — read on the raw coordinate instead, a
    mode whose only priced degrees are one GPU (stamped `tp1`) and one
    more landed in two groups of one X apiece, and "vs GPUs" was disabled
    with the untrue sentence "decode does not move with num_gpus here"."""
    ladders = mode_ladders(cloud) if "mode" in dims else {}

    def at(p: dict, d: str) -> Tuple[Any, ...]:
        return ladders[p[d]] if d == "mode" else (p[d],)

    return at


#: Each phase a cloud point can be drawn by, with its two axes per track:
#: tokens/s one request sees and tokens/s per GPU, both higher-is-better.
#: Decode's pair carries no phase in its name — it is the pair the cloud
#: carried before a point had two phases, and the Gallery and Compare pages
#: read it by those names; prefill's is named for the phase beside it.
#: This is the vocabulary's one definition: `cloud_point` writes these
#: fields, every view reads them, and the request model reads the phases
#: (`api._models.ParallelFrontierRequest`).
AXES = {
    "decode": {"sol": ("tok_s_per_user", "tok_s_per_gpu"),
               "measured": ("tok_s_per_user_measured", "tok_s_per_gpu_measured"),
               "partial": ("tok_s_per_user_partial", "tok_s_per_gpu_partial")},
    "prefill": {"sol": ("prefill_tok_s_per_user", "prefill_tok_s_per_gpu"),
                "measured": ("prefill_tok_s_per_user_measured",
                             "prefill_tok_s_per_gpu_measured"),
                "partial": ("prefill_tok_s_per_user_partial",
                            "prefill_tok_s_per_gpu_partial")},
}
CloudPhase = Literal["decode", "prefill"]
PHASES: Tuple[str, ...] = get_args(CloudPhase)


def _phase_rates(per: dict, phase: str) -> dict:
    """One phase's rates on every track, in the point's flat vocabulary
    (`AXES`): the rate one request sees and the rate per GPU, both read off
    the phase's published `throughput` (`phases.phase_throughput`) — no
    quotient is taken here. Empty when the phase published no SOL time."""
    body = (per.get("phases") or {}).get(phase) or {}
    published = body.get("throughput") or {}
    if not (published.get("sol") or {}).get("per_request"):
        return {}
    out: dict = {}
    for track, (user, gpu) in AXES[phase].items():
        rates = published.get(track)
        if not rates or not rates.get("per_request"):
            continue
        out[user] = float(rates["per_request"])
        out[gpu] = float(rates["per_gpu"])
    return out


def coverage_field(phase: str) -> str:
    """Where a point states how much of that phase's step is measured
    (`auto_sweep.MeasuredTrack.coverage`). One rule, no table: a page draws
    a phase's measured rate only where its coverage is complete, and the
    view tells it which field to read (`_view_payload`).

    A phase a point prices but never measures states no coverage at all —
    the field is simply not there, which a page reads as "not complete".
    The cloud's decode rungs are that case: a rung keys no measurement
    (`parallel_sweep._decode_ladder`), so the only road that could state a
    row count would be the walk's, and a point projected from a canonical
    record has no such road — stating it on one road alone is exactly the
    second producer this module exists to avoid."""
    return f"{phase}_coverage"


def cloud_point(per: dict) -> Optional[dict]:
    """One cloud point's values from an estimate: each phase's rates
    (`_phase_rates`) plus the facts a hover names, or None when the DECODE
    phase published no time — a cloud's rungs are decode batches, so a
    point with no decode step is no point of it (the producer turns that
    into an absence). Measured and partial rates ride beside SOL where the
    step published those times. A point whose estimate has no prefill phase
    simply carries no prefill fields; a view of that phase then says so
    (`view_forms`). The cloud's producer tags the point with what it is a
    point of."""
    dec = (per.get("phases") or {}).get("decode") or {}
    decode = _phase_rates(per, "decode")
    if not decode:
        return None
    rates = dec["throughput"]["sol"]
    out = {
        **decode,
        "tpot_s": float(rates["time_per_output_token_s"]),
        "batch": dec.get("batch"),
        "concurrent_requests": dec["throughput"]["concurrent_requests"],
    }
    meas = (dec.get("throughput") or {}).get("measured")
    if meas and meas.get("time_per_output_token_s"):
        out["tpot_measured_s"] = float(meas["time_per_output_token_s"])
    # The partial track (`Folded.partial`): measured rows measured, the rest
    # at SOL, with how much of the step's time is measured.
    part = (dec.get("throughput") or {}).get("partial")
    if part and part.get("time_per_output_token_s"):
        out["tpot_partial_s"] = float(part["time_per_output_token_s"])
        out["partial_measured_share"] = dec.get("partial_measured_share")
    pre = (per.get("phases") or {}).get("prefill") or {}
    prefill = _phase_rates(per, "prefill")
    if prefill:
        # Beside prefill's rates, what its hover names: the batch it ran at
        # (one request on this panel's points) and its step time — the
        # request's time to first token — on each track it published, and
        # how much of the step is measured, so a page draws the measured
        # rate only where every row of it is measured.
        out.update(prefill)
        out["prefill_batch"] = pre.get("batch")
        out["prefill_t_s"] = pre.get("t_sol_s")
        out["prefill_t_measured_s"] = pre.get("t_measured_s")
        out[coverage_field("prefill")] = pre.get("coverage")
    return out


def pareto_front(points: List[dict], *, phase: str = "decode",
                 track: str = "sol") -> List[dict]:
    """The points no other point beats on both of the phase's axes on
    `track`, most interactive first. A point with no value there is not
    drawn."""
    user, gpu = AXES[phase][track]
    ranked = sorted((p for p in points if p.get(user) is not None
                     and p.get(gpu) is not None),
                    key=lambda p: (-p[user], -p[gpu]))
    front: List[dict] = []
    for p in ranked:
        if not front or p[gpu] > front[-1][gpu]:
            front.append(p)
    return front


def _view_groups(cloud: Sequence[dict], *, dims: Sequence[str],
                 fix: Mapping[str, Sequence[Any]], group_by: Sequence[str]
                 ) -> Tuple[Dict[Tuple[Any, ...], List[dict]],
                            Dict[str, List[Any]]]:
    """What every view of a cloud starts from: the points whose `fix`
    dimensions each hold one of the given values, grouped by `group_by` (in
    cloud order), and `values` — each dimension's values among the points
    every OTHER held dimension admits, what a control for that dimension can
    offer without emptying the chart. `dims` are the dimensions the cloud's
    producer declares; a view naming another is refused, and every view
    publishes them (`dims`) — what a page may hold or group by.

    A point's `mode` is read through `mode_ladders`: the one-GPU deployment
    is held by, and grouped into, every mode with a ladder of its own —
    once per group, so no front or line draws it twice."""
    for d in (*fix, *group_by):
        if d not in dims:
            raise ValueError(f"unknown cloud dimension {d!r}; one of {tuple(dims)}")
    coords = point_coords(cloud, dims)

    def admits(p: dict, skip: Optional[str] = None) -> bool:
        return all(any(v in vs for v in coords(p, d))
                   for d, vs in fix.items() if d != skip)

    groups: Dict[Tuple[Any, ...], List[dict]] = {}
    for p in cloud:
        if not admits(p):
            continue
        keys: List[Tuple[Any, ...]] = [()]
        for d in group_by:
            held = fix.get(d)
            keys = [(*k, v) for k in keys for v in coords(p, d)
                    if held is None or v in held]
        for k in keys:
            groups.setdefault(k, []).append(p)
    values = {d: sorted({v for p in cloud if admits(p, d)
                         for v in coords(p, d)},
                        key=_control_order)
              for d in dims}
    return groups, values


#: What a line view of a cloud can put on X, and the coordinate each X is
#: the point's value of: the deployment's GPU count and the prompt length
#: are coordinates themselves; `batch` is the point's ACTUAL per-GPU decode
#: batch (`cloud_point`), so the `max` rung lands on its ceiling and a
#: ladder rung equal to that ceiling is the same X — its coordinate is
#: `batch_key`. The pairing is what `view_forms` holds still to ask whether
#: a phase moves along an axis. The vocabulary's one definition; the request
#: model reads it (`api._models.ParallelFrontierRequest`).
LineAxis = Literal["num_gpus", "batch", "context"]
LINE_AXES: Dict[str, str] = {"num_gpus": "num_gpus", "batch": "batch_key",
                             "context": "context"}

#: The two views of one cloud: each line its Pareto front (`frontier_view`),
#: or its points along an X of `LINE_AXES` (`line_view`).
ViewForm = Literal["frontier", "line"]

#: A view's form as one name, as a page switches it: the Pareto front, or a
#: line along one of `LINE_AXES`. `view_forms` answers for each of these.
FORM_KEYS: Tuple[str, ...] = ("frontier", *LINE_AXES)


def view_forms(cloud: Sequence[dict], *, track: str = "sol"
               ) -> Dict[str, Dict[str, Optional[str]]]:
    """Which (phase, form) views THIS cloud can answer, and why not where it
    cannot: `{phase: {form: None | reason}}` over `PHASES` × `FORM_KEYS`.

    Derived from the cloud, never declared per phase. A form is not a view
    of this cloud when

      * no point prices the phase at all (a cloud whose points carry no
        prefill), or
      * its X has one value here (nothing to run a line along), or
      * the phase's rate does not MOVE along it: holding every other
        coordinate still, the rate is the same at every value of the axis'
        coordinate. That is how "prefill runs one request per point" shows
        up — prefill is stamped on every decode rung of the parallel
        sweep's cloud, so a prefill line along the decode batch would be
        flat — without any consumer naming the phase.

    The Pareto front needs no axis, so it is a view whenever the phase is
    priced. A page reads this to disable the forms it cannot draw, with the
    reason; a view asked for one anyway is still answered — it is
    uninformative, not wrong."""
    # One coordinate tuple per point for every phase and axis to read: the
    # key a group is held by is this tuple without the axis' own coordinate
    # and without every coordinate that fixes it (`POINT_FIXES`: holding a
    # deployment holds its GPU count). Read through `point_coords`, the
    # same expansion every view groups by — a point on several modes'
    # ladders is in each of them.
    at = point_coords(cloud, POINT_DIMS)
    priced = [(coords, p) for p in cloud if "refused" not in p
              for coords in product(*(at(p, d) for d in POINT_DIMS))]
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for phase in PHASES:
        gpu = AXES[phase][track][1]
        drawn = [(coords, p) for coords, p in priced if p.get(gpu) is not None]
        if not drawn:
            out[phase] = {form: f"no point of this sweep prices {phase}"
                          for form in FORM_KEYS}
            continue
        forms: Dict[str, Optional[str]] = {"frontier": None}
        for x, coordinate in LINE_AXES.items():
            at = [(coords, p) for coords, p in drawn if p.get(x) is not None]
            if len({p[x] for _coords, p in at}) < 2:
                forms[x] = f"every point of this sweep is at one {x}"
                continue
            # Everything but the axis' own coordinate held still: the rate
            # at each X of that group, the best where a group has several
            # points at one X (the line's own rule).
            kept = [i for i, d in enumerate(POINT_DIMS) if d != coordinate
                    and coordinate not in POINT_FIXES.get(d, ())]
            groups: Dict[Tuple[Any, ...], Dict[Any, float]] = {}
            for coords, p in at:
                best = groups.setdefault(tuple(coords[i] for i in kept), {})
                if p[x] not in best or p[gpu] > best[p[x]]:
                    best[p[x]] = p[gpu]
            moves = any(len(set(rates.values())) > 1
                        for rates in groups.values() if len(rates) > 1)
            forms[x] = None if moves else (
                f"{phase} does not move with {x} here: every point of it is "
                f"the same at every {x}")
        out[phase] = forms
    return out


def _view_payload(cloud: Sequence[dict], *, dims: Sequence[str],
                  values: Mapping[str, List[Any]], phase: str,
                  track: str) -> dict:
    """What every view publishes beside its series: the controls' values,
    the dimensions a page may hold or group by, the phase drawn and the
    fields its axes and its coverage are on (`AXES`, `coverage_field`, so no
    page restates the vocabulary), and which (phase, form) views this cloud
    answers (`view_forms`)."""
    return {
        "values": dict(values),
        "dims": list(dims),
        # What each dimension's value fixes, so a page does not run a line
        # along an axis by a dimension that fixes it (one point a line).
        "fixes": {d: list(f) for d, f in POINT_FIXES.items() if d in dims},
        "phase": phase,
        # `sol` is the phase's definition and is always said; the tracks
        # beside it only where this cloud carries them, so a page draws
        # what is there rather than asking for a track no point has (the
        # decode rungs' measured one, which no fold writes).
        "axes": {**{name: list(pair) for name, pair in AXES[phase].items()
                    if name == "sol"
                    or any(p.get(pair[0]) is not None for p in cloud)},
                 "coverage": coverage_field(phase)},
        "forms": view_forms(cloud, track=track),
    }


def _fixed_by(group_by: Sequence[str], pts: Sequence[dict]) -> Dict[str, Any]:
    """What a group's coordinates fix (`POINT_FIXES`), stated beside them:
    a group by deployment names its mode and GPU count, so a page styles
    the line from its group rather than from one of its points. Every point
    of the group holds the one value, or the declaration is wrong."""
    out: Dict[str, Any] = {}
    for d in group_by:
        for f in POINT_FIXES.get(d, ()):
            seen = {p[f] for p in pts}
            if len(seen) != 1:
                raise ValueError(f"{d} fixes {f}, yet one group of it holds "
                                 f"{sorted(seen, key=str)}")
            out[f] = seen.pop()
    return out


def frontier_view(cloud: Sequence[dict], *, dims: Sequence[str],
                  fix: Mapping[str, Sequence[Any]], group_by: Sequence[str],
                  phase: str = "decode", track: str = "sol",
                  envelope_by: Optional[Sequence[str]] = None) -> dict:
    """The points of `cloud` held and grouped as `_view_groups` says, one
    line per group, each line its `pareto_front` on the phase's axes on
    `track`, with the controls' `values`.

    With `envelope_by`, the view also publishes `envelopes`: the same held
    points grouped by those coarser dimensions, each group's front — one
    per card over every deployment drawn, with `["gpu_sku"]`. An envelope is
    over whole lines, so its dimensions are among `group_by`."""
    def fronts(by: Sequence[str]) -> Tuple[List[dict], Dict[str, List[Any]]]:
        groups, values = _view_groups(cloud, dims=dims, fix=fix, group_by=by)
        return [{"group": {**_fixed_by(by, pts), **dict(zip(by, k, strict=True))},
                 "frontier": pareto_front(pts, phase=phase, track=track)}
                for k, pts in groups.items()], values

    series, values = fronts(group_by)
    out: Dict[str, Any] = {"series": series}
    if envelope_by is not None:
        loose = [d for d in envelope_by if d not in group_by]
        if loose:
            raise ValueError(f"an envelope is over whole lines; {loose} are "
                             f"not among the lines' dimensions {list(group_by)}")
        out["envelopes"], _values = fronts(envelope_by)
    return {**out, **_view_payload(cloud, dims=dims, values=values,
                                   phase=phase, track=track)}


def line_view(cloud: Sequence[dict], *, dims: Sequence[str], x: str,
              fix: Mapping[str, Sequence[Any]], group_by: Sequence[str],
              phase: str = "decode", track: str = "sol") -> dict:
    """The points of `cloud` held and grouped as `frontier_view`'s are, each
    group a line along `x` (`LINE_AXES`) of the phase's tokens/s per GPU on
    `track`.

    A group can hold several points at one X — the dimensions neither held
    nor grouped (several contexts, modes or batches at one GPU count; two
    batch keys at one batch). The line keeps the one with the most tokens/s
    per GPU, and the point keeps every coordinate and its layout, so a page
    can say which deployment it is. A point with no value on the phase's
    track (an absent one) is not drawn. Whether a line along `x` says
    anything about this phase is `view_forms`' answer, published beside the
    series rather than refused here."""
    if x not in LINE_AXES:
        raise ValueError(f"unknown line axis {x!r}; one of {tuple(LINE_AXES)}")
    if x != "batch" and x not in dims:
        raise ValueError(f"line axis {x!r} is not a dimension of this cloud")
    gpu = AXES[phase][track][1]
    groups, values = _view_groups(cloud, dims=dims, fix=fix, group_by=group_by)
    series = []
    for k, pts in groups.items():
        best: Dict[Any, dict] = {}
        for p in pts:
            at = p.get(x)
            if at is None or p.get(gpu) is None:
                continue
            if at not in best or p[gpu] > best[at][gpu]:
                best[at] = p
        series.append({"group": dict(zip(group_by, k, strict=True)),
                       "points": [best[at] for at in sorted(best)]})
    return {"series": series,
            **_view_payload(cloud, dims=dims, values=values, phase=phase,
                            track=track)}
