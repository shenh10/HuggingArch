"""Pareto frontiers drawn from a point cloud.

A chart of "points x a few dimensions" is served as one cached cloud and
views of it: some dimensions held at a value, the rest grouped into lines,
each line its Pareto front. The cloud's producer declares its dimensions;
the view is the one place a front is picked.
"""
from __future__ import annotations

from typing import (Any, Dict, List, Literal, Mapping, Optional, Sequence,
                    Tuple, get_args)


#: Every coordinate a point of any decode cloud states, whatever producer made
#: it. A producer may add a provenance tag (`row`, for a Compare request), but
#: it may not omit one of these: a view written for one cloud must be askable of
#: another. `ep_load_factor` is here even though it is deliberately NOT in
#: `measure.scope.of_walk` — rho is substituted at the point.
PointDim = Literal[
    "mode", "num_gpus", "gpu_sku", "context", "batch_key", "draft_tokens",
    "acceptance", "ep_load_factor",
]
POINT_DIMS: Tuple[str, ...] = get_args(PointDim)

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


def parallel_mode(parallel: Any) -> str:
    """A deployment's point-cloud mode coordinate.

    The four Scaling modes are their canonical `ParallelConfig` shapes. A
    one-GPU deployment is `tp1`; anything else is `custom` rather than a
    nearest-mode lie.
    """
    if parallel is None:
        return "tp1"
    degree = max(1, int(getattr(parallel, "cluster_size", 1) or 1))
    if degree == 1:
        return "tp1"
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


#: A decode point's two axes, both higher-is-better: tokens/s one request
#: sees and tokens/s per GPU, on one track (`decode_point`).
AXES = {"sol": ("tok_s_per_user", "tok_s_per_gpu"),
        "measured": ("tok_s_per_user_measured", "tok_s_per_gpu_measured"),
        "partial": ("tok_s_per_user_partial", "tok_s_per_gpu_partial")}


def decode_point(per: dict) -> Optional[dict]:
    """One decode Pareto point from an estimate: its decode phase's SOL
    throughput as (tokens/s one request sees, tokens/s per GPU), or None
    when the phase published no time. Measured rates ride beside SOL when
    the step published a measured time. The cloud's producer tags the point
    with what it is a point of."""
    dec = (per.get("phases") or {}).get("decode") or {}
    rates = (dec.get("throughput") or {}).get("sol")
    if not rates or not rates.get("time_per_output_token_s"):
        return None
    tpot = float(rates["time_per_output_token_s"])
    out = {
        "tok_s_per_user": 1.0 / tpot,
        "tok_s_per_gpu": float(rates["per_gpu"]),
        "tpot_s": tpot,
        "batch": dec.get("batch"),
        "concurrent_requests": dec["throughput"]["concurrent_requests"],
    }
    meas = (dec.get("throughput") or {}).get("measured")
    if meas and meas.get("time_per_output_token_s"):
        mt = float(meas["time_per_output_token_s"])
        out["tok_s_per_user_measured"] = 1.0 / mt
        out["tok_s_per_gpu_measured"] = float(meas["per_gpu"])
        out["tpot_measured_s"] = mt
    # The partial track (`Folded.partial`): measured rows measured, the rest
    # at SOL, with how much of the step's time is measured.
    part = (dec.get("throughput") or {}).get("partial")
    if part and part.get("time_per_output_token_s"):
        pt = float(part["time_per_output_token_s"])
        out["tok_s_per_user_partial"] = 1.0 / pt
        out["tok_s_per_gpu_partial"] = float(part["per_gpu"])
        out["tpot_partial_s"] = pt
        out["partial_measured_share"] = dec.get("partial_measured_share")
    return out


def pareto_front(points: List[dict], *, track: str = "sol") -> List[dict]:
    """The points no other point beats on both of `track`'s axes, most
    interactive first. A point with no value on the track is not drawn."""
    user, gpu = AXES[track]
    ranked = sorted((p for p in points if p.get(user) is not None
                     and p.get(gpu) is not None),
                    key=lambda p: (-p[user], -p[gpu]))
    front: List[dict] = []
    for p in ranked:
        if not front or p[gpu] > front[-1][gpu]:
            front.append(p)
    return front


def frontier_view(cloud: Sequence[dict], *, dims: Sequence[str],
                  fix: Mapping[str, Sequence[Any]], group_by: Sequence[str],
                  track: str = "sol") -> dict:
    """The points of `cloud` whose `fix` dimensions each hold one of the given
    values, grouped by `group_by` (one line per group, in cloud order), each line
    its `pareto_front` on `track`. `dims` are the dimensions the cloud's producer
    declares; a view naming another is refused.

    `values`: each dimension's values among the points every OTHER held
    dimension admits — what a control for that dimension can offer without
    emptying the chart."""
    for d in (*fix, *group_by):
        if d not in dims:
            raise ValueError(f"unknown cloud dimension {d!r}; one of {tuple(dims)}")

    def admits(p: dict, skip: Optional[str] = None) -> bool:
        return all(p[d] in vs for d, vs in fix.items() if d != skip)

    groups: Dict[Tuple[Any, ...], List[dict]] = {}
    for p in cloud:
        if admits(p):
            groups.setdefault(tuple(p[d] for d in group_by), []).append(p)
    return {
        "series": [{"group": dict(zip(group_by, k, strict=True)),
                    "frontier": pareto_front(pts, track=track)}
                   for k, pts in groups.items()],
        "values": {d: sorted({p[d] for p in cloud if admits(p, d)},
                             key=_control_order)
                   for d in dims},
    }
