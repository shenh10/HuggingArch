"""The one filling producer for throughput point clouds.

Every cloud states the same coordinates and keeps the same absence records.
Context×batch, parallel-layout and Gallery code enumerate their own questions
and yield `CloudRecord`s; this module is the one place that decides whether a
record becomes a served point or an absent one, and it alone calls the point
and absence constructors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping,
                    Optional, Sequence)

from .frontier import POINT_DIMS, PointRefusal, absent_point, cloud_point


def _held_coordinates(coordinates: Mapping[str, Any]) -> None:
    """Refuse a served point that is not fully identified."""
    missing = [name for name in POINT_DIMS if name not in coordinates]
    unknown = [name for name in coordinates if name not in POINT_DIMS]
    if missing or unknown:
        raise ValueError(
            f"a point states every cloud coordinate; missing={missing}, "
            f"unknown={unknown}")


@dataclass
class PointCloud:
    """A filled cloud, with the caller's token beside each point."""

    points: list[dict] = field(default_factory=list)
    tokens: dict[int, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CloudRecord:
    """One question a consumer asks of the cloud, before it is filled.

    A source enumerates records: it states the coordinates, the estimate to
    read and any source-side provenance. It may also state an existing refusal
    (layout, error, card) or the capacity verdict. The producer alone turns
    that record into a point.
    """
    coordinates: Mapping[str, Any]
    estimate: Any = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    token: Any = None
    refusal: Optional[PointRefusal] = None
    reason: str = ""
    served: bool = True
    beyond_max_context: bool = False
    capacity_reason: str = ""
    no_step_refusal: PointRefusal = "card"
    no_step_reason: str = "this card publishes no decode step"

    @classmethod
    def unanswered(cls, coordinates: Mapping[str, Any],
                   refusal: PointRefusal, reason: str, *,
                   token: Any = None) -> "CloudRecord":
        return cls(coordinates=coordinates, refusal=refusal, reason=reason,
                   token=token)


def point_coordinates(
    *, parallel: Any, gpu_sku: str, context: int,
    batch_key: Any, draft_tokens: int, acceptance: float,
    ep_load_factor: Any, num_gpus: Optional[int] = None,
) -> dict[str, Any]:
    """The one coordinate mapping, including a plan point's ρ. The
    deployment's three coordinates — its mode, its GPU count and its name
    (`frontier.POINT_FIXES`) — are read off the one `parallel`; `num_gpus`
    only for an estimate with no deployment of its own (`parallel` None)."""
    from .frontier import layout_label, parallel_mode
    if parallel is not None:
        num_gpus = parallel.cluster_size
    try:
        rho = float(ep_load_factor)
    except (TypeError, ValueError):
        rho = str(ep_load_factor)
    return {
        "mode": parallel_mode(parallel),
        "num_gpus": int(num_gpus or 1),
        "layout": layout_label(parallel) if parallel is not None else "TP1",
        "gpu_sku": gpu_sku,
        "context": context,
        "batch_key": batch_key,
        "draft_tokens": max(0, int(draft_tokens)),
        "acceptance": float(acceptance),
        "ep_load_factor": rho,
    }


def produce_cloud(records: Sequence[CloudRecord]) -> PointCloud:
    """Fill one cloud from records, served or absent.

    This is the only point-cloud filling path. A record that already knows it
    is refused stays absent. A readable record is read here (`cloud_point`:
    both phases of the one estimate); a point whose requests do not fit is
    capacity-refused, a point past the model's published context keeps its
    values and is flagged, and no decode time is the source's stated card or
    capacity refusal.
    """
    cloud = PointCloud()
    for record in records:
        coordinates = dict(record.coordinates)
        _held_coordinates(coordinates)
        if record.refusal is not None:
            point = absent_point(coordinates, record.refusal, record.reason)
        else:
            drawn = cloud_point(record.estimate)
            provenance = dict(record.provenance)
            if not record.served:
                point = absent_point(coordinates, "capacity",
                                     record.capacity_reason)
            elif record.beyond_max_context and drawn is not None:
                point = {**drawn, **coordinates, **provenance,
                         "beyond_max_context": True}
            elif drawn is None:
                point = absent_point(coordinates, record.no_step_refusal,
                                     record.no_step_reason)
            else:
                point = {**drawn, **coordinates, **provenance}
        cloud.points.append(point)
        cloud.tokens[id(point)] = record.token
    return cloud


def estimate_records(
    *,
    seq_lens: Sequence[int],
    verify_widths: Sequence[int],
    batch_points: Sequence[tuple[Optional[str], Optional[int], Optional[int]]],
    estimates: Sequence[Sequence[Any]],
    gpu_sku: str,
    parallel: Any,
    num_gpus: int,
    acceptance_rate: float,
    ep_load_factor: Any,
    fits: Callable[..., bool],
) -> Iterable[CloudRecord]:
    """The context×batch cloud from a `CapacitySweepPlan`'s finished points.

    `fits` is capacity's `fits` (the KV capacity fact), passed by the caller
    to keep this module below capacity's preparation layer: a point whose
    requests do not fit is capacity-refused; one that fits past the model's
    published context keeps its values, flagged — the parallel cloud's rule
    (`parallel_sweep.decode_rungs`). A `max` point whose rung is an explicit
    `bN` point is that point once, named by the display alias (`max`), not
    two coordinates for one step.
    """

    def displayed(points: Sequence[Any]) -> Iterator[int]:
        """One index per resolved (prefill, decode) batch; `max` names it.

        The request's grid is display vocabulary. What exists is the batch
        the estimate ran, and a `max` whose rung equals a `bN` rung is that
        rung — one cloud point, with the alias a reader filters on.
        """
        groups: Dict[Any, List[int]] = {}
        for index, est in enumerate(points):
            phases = (est or {}).get("phases") or {}
            def resolved(name: str, requested: Any) -> Any:
                if requested is not None and int(requested) > 0:
                    return int(requested)
                value = (phases.get(name) or {}).get("batch")
                return value if value is not None else (name, index)

            batch = (("error", index) if not est else
                     (resolved("prefill", batch_points[index][1]),
                      resolved("decode", batch_points[index][2])))
            groups.setdefault(batch, []).append(index)
        for indices in groups.values():
            yield next((index for index in indices
                        if batch_points[index][0] in (None, "max")),
                       indices[0])

    for seq, verify, points in zip(seq_lens, verify_widths, estimates,
                                   strict=True):
        for index in displayed(points):
            key, _prefill, decode = batch_points[index]
            est = points[index]
            coordinates = point_coordinates(
                parallel=parallel, num_gpus=num_gpus, gpu_sku=gpu_sku,
                context=seq,
                batch_key=key if key is not None else "max",
                draft_tokens=int(verify) - 1,
                acceptance=acceptance_rate, ep_load_factor=ep_load_factor)
            if not est:
                yield CloudRecord.unanswered(
                    coordinates, "error", "the sweep returned no point")
                continue
            beyond = bool(est.get("beyond_max_context"))
            if not fits(max_batch_per_gpu=est.get("max_batch_per_gpu"),
                        requests=decode or 1):
                ceiling = int(est.get("max_batch_per_gpu") or 0)
                yield CloudRecord(
                    coordinates=coordinates, estimate=est, served=False,
                    capacity_reason=(
                        f"this GPU serves {ceiling} request"
                        f"{'s' if ceiling != 1 else ''} at this context"))
                continue
            yield CloudRecord(coordinates=coordinates, estimate=est,
                              beyond_max_context=beyond)


def estimate_cloud(**kwargs: Any) -> PointCloud:
    """The context×batch cloud, filled by the one producer."""
    return produce_cloud(estimate_records(**kwargs))


def parallel_records(plan: Any, results: Sequence[Any]
                     ) -> Iterable[CloudRecord]:
    """The Parallelism sweep's cloud from every layout/card/context/batch
    answer — each point a decode rung that also carries its deployment's
    prefill.

    `plan` is the caller-side `ParallelSweepPlan`; only its published
    coordinate and absence interface is read here (`coordinates`,
    `contexts`, `gpu_skus`, `requested_batch_key`), so this module owns the
    cloud without owning layout enumeration. A point's mode is its
    deployment's (`frontier.parallel_mode`), one point per deployment: that
    the one-GPU deployment is every ladder's first rung is the views' rule
    (`frontier.mode_ladders`).
    """
    for cell, result in zip(plan.cells, results, strict=True):

        def at(gpu: str, context: int, batch_key: Any = None) -> dict:
            return plan.coordinates(
                cell.parallel, gpu, context=context,
                batch_key=(plan.requested_batch_key if batch_key is None
                           else batch_key))

        if "refused" in result or result.get("error"):
            refusal, reason = (("layout", str(result["refused"]))
                               if "refused" in result
                               else ("error", str(result["error"])))
            for gpu in plan.gpu_skus:
                for context in plan.contexts:
                    yield CloudRecord.unanswered(at(gpu, context), refusal, reason)
            continue
        for gpu, entries in (result.get("cards") or {}).items():
            for context, card in zip(plan.contexts, entries, strict=True):
                # A card or a context that could not be priced is its own
                # absence, in the cloud's refusal vocabulary
                # (`parallel_sweep._absent`); the rest stand.
                if "refused" in card:
                    yield CloudRecord.unanswered(at(gpu, context),
                                                 card["refused"],
                                                 str(card["reason"]))
                    continue
                # The card's prefill, on every rung of its ladder. The sweep
                # prices prefill once per (card, layout, context) — one
                # request, whatever the decode batch beside it
                # (`parallel_sweep._decode_ladder` moves the decode rows
                # alone) — so it is the same fact at every rung rather than
                # a fact of the rung. It is read here, after the sweep's one
                # measured fold has written into `card["sol"]`, so prefill
                # carries its measured track where the request asked for
                # one. That a prefill line along the decode batch would be
                # flat is then the cloud's own answer (`frontier.view_forms`),
                # not a rule a consumer remembers.
                prefill = ((card.get("sol") or {}).get("phases")
                           or {}).get("prefill")
                for point in card.get("ladder") or ():
                    phases = dict(point.get("phases") or {})
                    if prefill:
                        phases["prefill"] = prefill
                    yield CloudRecord(
                        coordinates=at(gpu, context,
                                       point.get("batch_key", "max")),
                        estimate={**point, "phases": phases},
                        provenance={
                            "x": cell.x,
                            "parallel": cell.parallel.to_dict()},
                        beyond_max_context=bool(
                            point.get("beyond_max_context")),
                        no_step_refusal="capacity",
                        no_step_reason="this card serves no decode step")


def parallel_cloud(plan: Any, results: Sequence[Any]) -> PointCloud:
    """The Parallelism cloud, filled by the one producer."""
    return produce_cloud(parallel_records(plan, results))
