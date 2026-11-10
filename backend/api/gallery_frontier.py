"""Gallery decode Pareto frontier — tokens/s per user against tokens/s per GPU.

The InferenceMax-style view of decode: for each model on one GPU type, every
layout up to `FRONTIER_MAX_GPUS` (`parallel_sweep.frontier_layouts`: one
GPU, then 2 … 64 GPUs under every split of them it prices), each at every
per-GPU decode batch of `FRONTIER_BATCH_GRID`, at every Gallery context. A
point is one (layout, batch): how fast one request's tokens come out
(interactivity) and how many tokens each GPU outputs per second. The frontier at a context is the
points no other point beats on both.

Every number is the estimate's own: a point reads its decode phase's
`throughput.sol` (`phases.phase_throughput` — `time_per_output_token_s`,
`per_gpu`), and whether it exists is `capacity.point_served`
(`capacity.served_points`). A layout is one walk, every (context, batch)
of it by substitution (`capacity.sweep_points`), each its own compute job.

The walk does not depend on the GPU, only on its walk class
(`framework_bindings.card_class`: what a framework's dispatch reads of the
card and the dtypes it lists a peak for — what changes a walk's rows). A
model has one CSE SOL law per layout and card class
(`_layout_law`, the shared SOL pool); each card's entry substitutes that law
at its peaks, bandwidth, fabric, floors and capacity facts (`_restamp`). A
view at
another HBM fraction walks its own card alone and keeps no class walk.

The SOL half is cached under scope `none`. Kernel Bench's measured plan is an
explicit later layer, never a reason to expand the canonical law into rows.
One request reads the lake's version once (`frontier_track`), is
identified by that answer (`frontier_request_key`) and draws every model
under it (`build_frontier`): a whole response drawn while the lake was
unavailable is never served once it answers. A track the lake cannot give —
its version unanswered, or its rows' lookup failing mid-request — draws the
SOL drawing, labelled "unavailable": one drawing per view, kept under scope
`none` (`_Stored.view_key`), so a lake that flaps reads it back rather than
drawing again off the entry, and a lake that answers again folds its rates.

The lake's answer reaches the projection, never the law. A class law carries
no measurement scope (`_Stored.walk_identity` under `none`), so a lake outage
neither re-walks a class nor leaves a law under an outage's key.

A model's frontier entry on one GPU is kept on disk and in the remote tier
(`gallery_overview._FRONTIER`), keyed on its `result_version` and the
render config: its points — each only what the read path reads
(`_theory_item`), their decode phase, step forms and canonical
measure-record name. One GPU view spans every
layout up to `FRONTIER_MAX_GPUS`. A page reads drawings: each view's, and
per Kernel Bench snapshot the entry's measured rates with each folded
point's coverage, in `gallery_overview._FRONTIER_VIEWS` (`_Stored`; for
DeepSeek-V3 on H100 a drawing is ~21 KB and the measured rates ~300 KB, most
of it the 2,545 distinct unmeasured keys the coverage panel names). A view already drawn reads
no entry; another view reads the points and their named records
(`_frontier_read`).

Each point's step is cached as a function of the two floors (`FloorForm`,
the projection's own form of the law: `sol_law._FloorTerms`), and for a
model with an MTP draft at every K of `FRONTIER_DRAFT_TOKENS` too —
projected once when the entry is built (the warm builds it), one
speculative law per layout with every K read off it by substitution. So
the entry is keyed on the walk alone (`entry_key`): a view's floors, K and
acceptance are substituted when it is drawn (`_at_view`); none of them
walks or projects. A K whose step failed to price is published with the
views at that K (`draft_errors`).
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import statistics
import functools
import time
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple)

from backend.api._models import BatchPoint, InferenceEstimateSweepRequest
from backend.api.estimate_env import resolve_estimate_environment
from backend.api.gallery_overview import (
    _FRONTIER,
    _FRONTIER_VIEWS,
    _model_concurrency,
    _resolve_one,
)
from backend.api.offload import (
    PrioritySlots,
    ensure_spec_generation,
    map_off_loop,
    report_degraded,
    report_partial,
    report_progress,
    run_off_loop,
    run_ranked,
    spec_generation,
    urgency,
)
from backend.arch.eval.precision import AS_SHIPPED
from backend.arch.snapshots import iter_snapshot_ids
from backend.inference.batch_sweep import measure_points
from backend.inference.capacity import point_served
from backend.inference.card import Card
from backend.measure import lookup as _measure_lookup
from backend.measure.auto_sweep import DECLINED, STATES, blend
from backend.measure.lookup import MeasurementsUnavailable
from backend.inference.defaults import (COMM_LATENCY_US_DEFAULT,
                                        GPU_MEMORY_FRACTION_DEFAULT,
                                        LAUNCH_FLOOR_US_DEFAULT)
from backend.inference.gpu_specs import NoPeakForDtype
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import Speculation, phase_throughput, step_tokens
from backend.inference.row_roofline import FloorForm
from backend.inference.speculative_draft import SpeculativeDraft
from backend.arch.eval.program import Program
from backend.arch.eval.refusal import LayoutRefusal
from backend.inference.parallel_sweep import (
    ALL_MODES,
    class_cards,
    class_frontier_layouts,
    frontier_layouts,
    layout_label,
    layout_verdict,
    parallel_configs_for_mode,
    sweep_caps,
)
from backend.inference.frontier import (POINT_DIMS, frontier_view, parallel_mode,
                                       pareto_front as _pareto_front)
from backend.inference.sweep_grid import (
    FRONTIER_ACCEPTANCE_DEFAULT,
    FRONTIER_BATCH_GRID,
    FRONTIER_DRAFT_TOKENS,
    FRONTIER_MAX_GPUS,
    GALLERY_OUTPUT_TOKENS,
    GALLERY_SEQ_LENS,
    ROUTINE_SEQ_LENS,
    SWEEP_MAX_GPUS,
)
from backend.inference.sol_data import (
    StepLawKey,
    ServingWalkFacts,
    ServingWalkKey,
    walkManifestIsValid,
    walkRecordSchema,
)
from backend.inference.sol_data import sharedSolPool
from backend.code_trace.provider import recording, run_recorded
from backend.inference.sol_law import (
    draftForwardsLaw,
    servingLawCapacity,
    servingMeasureMatrix,
    servingLawPoints,
    servingLawRecord,
)
from backend.inference.point_cloud import (CloudRecord, point_coordinates,
                                           produce_cloud)
from backend.inference.serving_projection import (
    PublicServingLawRequest,
    projectServingLawPoints,
)
from backend.measure.scope import NONE as NO_MEASUREMENTS, of_render

logger = logging.getLogger(__name__)

# Compatibility for callers that learned the front function from this module;
# the implementation and every new caller read `frontier.pareto_front`.
pareto_front = _pareto_front

# Payload-schema stamps folded into the frontier's keys — bump on edit of
# the fields of what each keys (the grid constants sit under the engine
# version): the entry, and what views read off it (the measured rates and a
# drawn view). A drawing's fields change without the entry's, and an entry is
# priced again only when its own do.
_FRONTIER_SCHEMA = "gallery-frontier-v14"
_DRAWING_SCHEMA = "gallery-frontier-drawing-v6"
_GALLERY_SELECTION = "envelope"
#: The precision profile every frontier is priced under: as shipped.
_GALLERY_PRECISION = AS_SHIPPED
#: How many of a coverage state's keys a frontier's coverage lists, most
#: needed first (`frontier_coverage`; the count is of all of them).
COVERAGE_KEYS_LISTED = 10


class FrontierView(NamedTuple):
    """What one frontier is priced under besides the models: the GPU, the
    HBM fraction a deployment budgets, and the two time floors (one link
    traversal's latency and the shortest one kernel launch takes, µs;
    `defaults`). One GPU view at the defaults is what the page
    opens on and the warm fills."""
    gpu_sku: str
    gpu_memory_fraction: float = GPU_MEMORY_FRACTION_DEFAULT
    comm_latency_us: float = COMM_LATENCY_US_DEFAULT
    launch_floor_us: float = LAUNCH_FLOOR_US_DEFAULT
    #: Speculative decoding with the model's MTP draft: K drafted tokens
    #: per step (0 is plain decode; one of `FRONTIER_DRAFT_TOKENS`) kept at
    #: `acceptance_rate`. Read off the entry like the floors.
    draft_tokens: int = 0
    acceptance_rate: float = FRONTIER_ACCEPTANCE_DEFAULT

    def walk_parts(self) -> List[Any]:
        """What a model's cached entry is priced under: not the floors —
        each point carries its step as a function of them (`FloorForm`)."""
        return [self.gpu_sku, _GALLERY_PRECISION.key,
                float(self.gpu_memory_fraction)]

    def keeps_walk(self) -> bool:
        """Whether the view reads the model's kept class walk
        (`_Stored.walk_identity`): at the default HBM fraction. Each card's
        ceiling batch is walked on that card at that fraction, so a view at
        another walks its own card alone (`walk_cards`), kept nowhere."""
        return (float(self.gpu_memory_fraction)
                == ServingWalkFacts.producer().gpu_memory_fraction)

    def _servingLawEnvironment(self) -> dict[str, Any]:
        return {
            **ServingWalkFacts.producer().to_environment(),
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "comm_latency_us": self.comm_latency_us,
            "launch_floor_us": self.launch_floor_us,
        }

    def _servingLawAxes(self, *, draft_tokens: int = 0) -> dict[str, Any]:
        batches = tuple(
            point["decode_batch_size"] for point in FRONTIER_BATCH_GRID)
        return {
            "contexts": tuple(GALLERY_SEQ_LENS),
            "decode_batches": batches,
            "prefill_batches": (1,) * len(batches),
            "prefix_tokens": (0,) * len(GALLERY_SEQ_LENS),
            "verify_widths": (1 + int(draft_tokens),)
            * len(GALLERY_SEQ_LENS),
        }

    def servingLawRequest(self, walk: ServingWalkKey, layout: Any, *,
                          draft_tokens: int = 0) -> PublicServingLawRequest:
        return PublicServingLawRequest(
            walk=walk, card=self.gpu_sku, layout=layout,
            environment=self._servingLawEnvironment(),
            draft_tokens=draft_tokens,
            **self._servingLawAxes(draft_tokens=draft_tokens))

    def entry_view(self) -> "FrontierView":
        """The view a model's entry is projected under: this one at the
        default floors under plain decode. Every point carries its step as
        a function of the floors and each K's beside it, so neither the
        floors nor the draft enter the entry (`walk_parts` keys it); a view
        substitutes its own when it is drawn (`_at_view`)."""
        return FrontierView(self.gpu_sku, self.gpu_memory_fraction)

    def walk_cards(self) -> List[str]:
        """The cards one walk serves this view from, the walk card first:
        the GPU's walk class (`parallel_sweep.class_cards`) when the view
        keeps the walk, else the GPU alone."""
        return class_cards(self.gpu_sku) if self.keeps_walk() else [self.gpu_sku]

    def key_parts(self) -> List[Any]:
        return [*self.walk_parts(), float(self.comm_latency_us),
                float(self.launch_floor_us), int(self.draft_tokens),
                float(self.acceptance_rate) if self.draft_tokens else None]

    def floors_s(self) -> Tuple[float, float]:
        """The two floors in seconds: link latency, launch floor."""
        return self.comm_latency_us * 1e-6, self.launch_floor_us * 1e-6


def _batch_key(batch: Any) -> str:
    """A decode batch's frontier-grid rung (`max` for a card's ceiling)."""
    for point in FRONTIER_BATCH_GRID:
        if point["decode_batch_size"] == batch:
            return str(point["key"])
    return "max"


def _theory_item(seq: str, label: str, mode: str, num_gpus: int,
                 batch_key: str, per: dict, measure_record: Optional[str],
                 floors: Optional[dict]) -> dict:
    """One cached point: what the read path reads of it and nothing else.
    The frontier draws decode, so the point keeps its decode phase alone
    (priced on the entry's card), its step as a function of the floors, and
    — when the canonical record carries concrete measure keys — the name of
    that record. The key table itself belongs to the record and is never
    copied into the entry."""
    return {"seq": seq, "label": label, "mode": mode, "num_gpus": num_gpus,
            "batch_key": batch_key,
            "sol": {"phases": {"decode": per["phases"]["decode"]}},
            "floors": floors,
            **({} if measure_record is None else
                {"measureRecord": measure_record})}


def _layout_request(model_id: str, gpu_sku: str, fraction: float,
                    precision: dict, parallel: dict
                    ) -> InferenceEstimateSweepRequest:
    """The sweep one layout of a model is priced by on `gpu_sku` under the
    precision profile `precision` (canonical): every Gallery context at
    every frontier batch, at the default floors (the view's are applied
    when the entry is read, `_at_view`)."""
    defaults = FrontierView(gpu_sku)
    return InferenceEstimateSweepRequest(
        model_id=model_id,
        gpu_sku=gpu_sku,
        parallel=parallel,
        precision={"canonical": precision},
        gpu_memory_fraction=fraction,
        comm_latency_us=defaults.comm_latency_us,
        launch_floor_us=defaults.launch_floor_us,
        avg_output_tokens=GALLERY_OUTPUT_TOKENS,
        seq_lens=list(GALLERY_SEQ_LENS),
        batch_grid=[BatchPoint(**bp) for bp in FRONTIER_BATCH_GRID],
        # Keys on each point so Kernel Bench rows can fold later; the walk
        # itself reads no measurements (`measurement_scope` stays none).
        selection_mode=_GALLERY_SELECTION,
    )


def _layout_law(item: Tuple[str, List[str], float, dict, dict]) -> dict:
    """One layout's SOL laws, walked once and stored without point payloads."""
    model_id, cards, fraction, precision, parallel = item
    cfg = (parallel if isinstance(parallel, ParallelConfig)
           else ParallelConfig.from_dict(parallel))
    request = _layout_request(model_id, cards[0], fraction, precision,
                              cfg.to_dict())
    try:
        environment = resolve_estimate_environment(request)
        if draft_refusal(model_id) is None:
            capacity = servingLawCapacity(
                environment, gpu_sku=cards[0], parallel=cfg,
                capacityCards=cards, capacityContexts=ROUTINE_SEQ_LENS)
            if all(int(fact["max_decode_batch"] or 0) <= 0
                   and int(fact["max_prefill_batch"] or 0) <= 0
                   for fact in capacity):
                return {"refused": next(
                    fact.get("infeasibility_reason")
                    or "layout admits no request" for fact in capacity)}
            spec = request.model_copy(update={
                "speculative_draft_tokens": 1,
                "speculative_draft": "recursive",
                "acceptance_rate": FRONTIER_ACCEPTANCE_DEFAULT,
                "selection_mode": "theoretical_only",
            })
            record = servingLawRecord(
                resolve_estimate_environment(spec),
                gpu_sku=cards[0], anchorContext=GALLERY_SEQ_LENS[0],
                outputTokens=GALLERY_OUTPUT_TOKENS, parallel=cfg,
                capacityCards=cards, capacityContexts=ROUTINE_SEQ_LENS,
                draft_forwards=draftForwardsLaw,
                publicCapacity=capacity)
            if "refused" in record:
                return record
        else:
            record = servingLawRecord(
                environment,
                gpu_sku=cards[0], anchorContext=GALLERY_SEQ_LENS[0],
                outputTokens=GALLERY_OUTPUT_TOKENS, parallel=cfg,
                capacityCards=cards, capacityContexts=ROUTINE_SEQ_LENS)
        return record
    except Exception as e:
        try:
            layout_verdict(_resolve_one(model_id)[1], cfg)
        except LayoutRefusal as reason:
            return {"refused": str(reason)}
        except Exception as diagnostic:
            logger.warning(
                "gallery frontier: law for %s under %s failed (%s); layout "
                "verdict also failed (%s)", model_id, layout_label(cfg), e,
                diagnostic)
            return {"error": str(e), "diagnostic": str(diagnostic)}
        if isinstance(e, NoPeakForDtype):
            return {"no_peak": [e.dtype, e.kind]}
        logger.warning("gallery frontier: law for %s under %s failed: %s",
                       model_id, layout_label(cfg), e)
        return {"error": str(e)}


def draft_refusal(model_id: str) -> Optional[str]:
    """Why the model's frontier has no speculative view (it drafts with
    no MTP stack, `SpeculativeDraft.refusal`), or None when it has one."""
    resolved = _resolve_one(model_id)[1]
    if resolved is None:
        return "no spec resolves for this model"
    return SpeculativeDraft.refusal(Program.of(resolved))


def _restamp_layout(walked: dict, gpu_sku: str, cfg: ParallelConfig,
                    walk_key: Optional[ServingWalkKey] = None,
                    view: Optional[FrontierView] = None) -> dict:
    """One layout's serving law on `gpu_sku`, one of the class's
    cards: `{"theory": [point…], "draft_errors": […]}` — each point served
    on the card, its decode step as a function of the floors there, and for
    a model with an MTP draft its step at every K (`_draft_points`) — or
    the walk's refusal or error, said on this card."""
    if walked.get("schema") == walkRecordSchema:
        card = Card.of(gpu_sku)
        view = (view if view is not None else FrontierView(gpu_sku)).entry_view()
        layoutStarted = time.perf_counter()

        theory: List[dict] = []
        draftErrors: List[dict] = []
        byPoint: Dict[Tuple[str, int], dict] = {}
        label, num_gpus = layout_label(cfg), cfg.cluster_size
        measureRecord = (
            StepLawKey.of_layout(walk_key, cfg).record_name
            if walk_key is not None
            and "measurePoints" in walked["phases"]["decode"] else None)

        def rows(record: dict, *, draft_tokens: int = 0) -> list[list[dict | None]]:
            if walk_key is None:
                return servingLawPoints(
                    record, card=card, contexts=list(GALLERY_SEQ_LENS),
                    decodeBatches=[point["decode_batch_size"]
                                   for point in FRONTIER_BATCH_GRID],
                    outputTokens=GALLERY_OUTPUT_TOKENS, prefillBatches=1,
                    prefixCached=0, verifyWidth=draft_tokens + 1,
                    draftTokens=draft_tokens,
                    acceptanceRate=FRONTIER_ACCEPTANCE_DEFAULT,
                    comm_latency_us=view.comm_latency_us,
                    launch_floor_us=view.launch_floor_us,
                    servedOnly=True, floorForms=True)
            request = view.servingLawRequest(
                walk_key, record["parallel"], draft_tokens=draft_tokens)
            return projectServingLawPoints(record, request, floorForms=True)

        def form(served: dict) -> dict:
            """The point's decode step as a function of the floors, moved
            off its decode phase onto the point (`_theory_item`)."""
            return served["phases"]["decode"].pop("floors")

        try:
            points = rows(walked)
        except Exception as e:
            logger.warning("gallery frontier law for %s on %s failed: %s",
                           label, gpu_sku, e)
            return {"error": str(e)}
        for seq, row in zip(GALLERY_SEQ_LENS, points, strict=True):
            for batchPoint, served in zip(FRONTIER_BATCH_GRID, row, strict=True):
                if served is None:
                    continue
                if not point_served(
                        max_batch_per_gpu=served["max_batch_per_gpu"],
                        beyond_max_context=served["beyond_max_context"],
                        requests=batchPoint["decode_batch_size"] or 1):
                    continue
                item = _theory_item(
                    str(seq), label, parallel_mode(cfg), num_gpus,
                    _batch_key(served["phases"]["decode"]["batch"]), served,
                    measureRecord, form(served))
                theory.append(item)
                byPoint[(str(seq), served["phases"]["decode"]["batch"])] = item
        # A layout no card serves has no point a draft could extend; its K
        # projections are absent, not errors, and are not priced at all.
        if walked.get("speculative") and theory:
            draftRecords = {str(k): walked for k in FRONTIER_DRAFT_TOKENS}
        else:
            draftRecords = {}
        for textK, draftRecord in draftRecords.items():
            k = int(textK)
            try:
                draftPoints = rows(draftRecord, draft_tokens=k)
            except Exception as exc:
                draftErrors.append({
                    "layout": label, "draft_tokens": k, "error": str(exc),
                })
                continue
            for seq, row in zip(GALLERY_SEQ_LENS, draftPoints, strict=True):
                for batchPoint, served in zip(FRONTIER_BATCH_GRID, row,
                                              strict=True):
                    if served is None:
                        continue
                    if not point_served(
                            max_batch_per_gpu=served["max_decode_batch_per_gpu"],
                            beyond_max_context=served["beyond_max_context"],
                            requests=batchPoint["decode_batch_size"] or 1):
                        continue
                    batch = served["phases"]["decode"]["batch"]
                    stepForm = form(served)
                    item = byPoint.get((str(seq), batch))
                    if item is None:
                        # A batch only the draft's capacity admits: a
                        # K-step point, with no plain step (`_at_view`).
                        item = _theory_item(
                            str(seq), label, parallel_mode(cfg), num_gpus,
                            _batch_key(batch), served, None, None)
                        theory.append(item)
                        byPoint[(str(seq), batch)] = item
                    item.setdefault("draft", {})[textK] = stepForm
        return {
            "theory": theory,
            "draft_errors": draftErrors,
            "timings": {"layout_s": time.perf_counter() - layoutStarted},
        }
    # The walk's own answer for a layout it does not deploy: a refusal, an
    # error, or a dtype the class lists no peak for — said on this card.
    if "refused" in walked or "error" in walked:
        return {k: walked[k] for k in ("refused", "error") if k in walked}
    if "no_peak" in walked:
        return {"error": str(NoPeakForDtype(gpu_sku, *walked["no_peak"]))}
    raise ValueError(
        f"{layout_label(cfg)}: not a {walkRecordSchema} record "
        f"(schema {walked.get('schema')!r})")


def _restamp(model_id: str, walk: dict, gpu_sku: str,
             view: Optional[FrontierView] = None) -> dict:
    """A model's frontier entry on `gpu_sku` off its class walk
    (one serving-law record per layout any card of the class prices): the
    layouts a frontier on this card prices (`frontier_layouts`), each
    priced on it (`_restamp_layout`) and folded (`_fold`). No walk."""
    walked = {json.dumps(w["parallel"], sort_keys=True): w["walk"]
              for w in walk["layouts"]}
    layouts = [ParallelConfig.from_dict(p)
               for p in walk["layoutsByCard"][gpu_sku]]
    walk_key = (
        _Stored(model_id, _resolve_one(model_id)[2]).serving_key(view)
        if view is not None and view.keeps_walk() else None)
    results = [_restamp_layout(
                   walked[json.dumps(cfg.to_dict(), sort_keys=True)],
                   gpu_sku, cfg, walk_key, view) for cfg in layouts]
    entry = _fold(model_id, layouts, results)
    if walk_key is not None:
        entry["_measurePool"] = walk_key.identity.key
    # Why a speculative view draws nothing for this model (no MTP draft).
    entry["draft_unavailable"] = draft_refusal(model_id)
    return entry


def _fold(model_id: str, layouts: List[ParallelConfig], results: List[dict]) -> dict:
    """A model's frontier entry from its layouts' points: the frontier at
    each context, what each layout answered, and every layout's failed
    speculative K (`draft_errors`)."""
    summary: List[dict] = []
    theory: List[dict] = []
    draft_errors: List[dict] = []
    for cfg, res in zip(layouts, results, strict=True):
        row: Dict[str, Any] = {"layout": layout_label(cfg), "parallel": cfg.to_dict()}
        if "theory" in res:
            row["points"] = len(res["theory"])
            theory.extend(res["theory"])
            draft_errors.extend(res.get("draft_errors") or [])
        else:
            row.update({k: res[k] for k in ("refused", "error") if k in res})
        summary.append(row)
    failed = [r for r in summary if "error" in r]
    layoutTimings: Dict[str, float] = {}
    for res in results:
        for key, value in (res.get("timings") or {}).items():
            layoutTimings[key] = layoutTimings.get(key, 0.0) + value
    return {
        "model_id": model_id,
        "layouts": summary,
        "_theory": theory,
        "_draft_errors": draft_errors,
        "_measure_timings": layoutTimings,
        # Every layout failing (not refused) is the model's failure.
        "error": (failed[0]["error"] if failed and len(failed) == len(summary)
                  else None),
    }


def _frontier_scope() -> Tuple[Any, str]:
    """The shared rows' scope, or `none` / `unavailable` when there is
    nothing to fold."""
    try:
        scope = _measure_lookup.resolve_scope(reads=True, owner=None)
    except MeasurementsUnavailable:
        return NO_MEASUREMENTS, "unavailable"
    return scope, ("reads" if scope.reads else "none")


def _public_entry(entry: dict) -> dict:
    """The entry without what each view reads its own part of (`_theory`,
    `_draft_errors`)."""
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def _at_view(item: dict, view: FrontierView,
             fold: Optional[Mapping[str, Any]] = None) -> Optional[dict]:
    """A theory point with its decode step under the view: the step as a
    function of the two floors (`FloorForm`) — the plain step's, or with
    K drafted tokens the speculative step's at that K — at the view's
    floors, the one place a view's floors enter a frontier; the tokens a
    step keeps at the view's acceptance (`phases.step_tokens`), and its
    throughput off those (`phases.phase_throughput`, the estimate's own
    reading). The decode binding is the one owner of which step a stored
    point is (`speculative_draft_tokens`): None when the point has no step
    at the view's K (its projection failed, `draft_errors`), or under
    plain decode for a point stored from a K-step — a batch the draft's
    smaller capacity ceiling admits but plain decode does not. A plain
    point stored without its form is not an entry of this schema.

    `fold` is what the measured fold said of the point's step at the view's
    K (`_measured_rates`): its measured total, which no floor moves, and the
    partial track (`auto_sweep.blend`) — the measured terms' SOL read off the
    form at the view's floors (`FloorForm.terms_at`)."""
    k = int(view.draft_tokens)
    sol = item["sol"]
    dec = dict(sol["phases"]["decode"])
    if not k and _stepDraftTokens(dec):
        return None
    if k:
        form = (item.get("draft") or {}).get(str(k))
        if form is None:
            return None
    else:
        form = item.get("floors")
        if form is None:
            raise ValueError(
                f"frontier point {item.get('label')} at {item.get('seq')} "
                "carries no step form: not a "
                f"{_FRONTIER_SCHEMA} entry")
    stepForm = FloorForm.from_json(form)
    dec["t_sol_s"] = stepForm.at(*view.floors_s())
    if fold is not None:
        dec["t_measured_s"] = fold["t_measured_s"]
        partial = blend(dec["t_sol_s"],
                        stepForm.terms_at(*view.floors_s(),
                                          _terms(fold["measured_terms"])),
                        fold["measured_part_s"])
        dec["t_partial_s"] = partial["t_s"]
        dec["partial_measured_share"] = partial["measured_share"]
    if k:
        binding = dict(dec["binding"])
        tokens, formula = step_tokens(
            "decode", int(dec["batch"]), binding["tokens_per_step"],
            Speculation(k, float(view.acceptance_rate)))
        binding.update(effective_tokens_per_step=tokens,
                       effective_tokens_formula=formula,
                       speculative_draft_tokens=k,
                       acceptance_rate=float(view.acceptance_rate))
        dec["binding"] = binding
    # The groups and GPUs the step's own rates were taken over, as the
    # measured fold re-reads them (`batch_sweep.apply_measured`).
    dec["throughput"] = phase_throughput(
        dec, unique_tokens_multiplier=dec["throughput"]["groups"],
        num_gpus=dec["throughput"]["num_gpus"])
    return {**item, "sol": {**sol, "phases": {**sol["phases"], "decode": dec}}}


def _termMask(terms: Sequence[int]) -> str:
    """Step-term indices as a hex bit mask (bit k set: term k), the compact
    form a fold keeps per point."""
    mask = 0
    for k in terms:
        mask |= 1 << int(k)
    return format(mask, "x")


def _terms(mask: str) -> List[int]:
    """`_termMask` read back."""
    value = int(mask, 16)
    return [k for k in range(value.bit_length()) if value >> k & 1]


def _stepDraftTokens(dec: Mapping[str, Any]) -> int:
    """The decode binding's K: the one statement of whether a step drafts."""
    return int((dec.get("binding") or {}).get("speculative_draft_tokens") or 0)


def _view_scope(view: FrontierView) -> Tuple[Any, str]:
    """The measurement scope a view's measured track reads, and its state:
    the shared rows' (`_frontier_scope`); a speculative view reads the K-aware
    measure table its record carries. A request reads it once and draws every
    model under it (`build_frontier`)."""
    return _frontier_scope()


#: `(scope, state)` of `_view_scope`, as one request read it.
Track = Tuple[Any, str]


def _labelled(drawing: dict, state: str) -> dict:
    """A drawing of no measured rates under the track state it is served
    with: "unavailable" draws exactly what "none" does (`_draw` reads no
    rates for either), so one drawing serves both."""
    if drawing["measured_track"]["state"] == state:
        return drawing
    return {**drawing, "measured_track": {"state": state}}


def _measureReader(core: dict, gpu_sku: str, view: FrontierView,
                   timings: Optional[Dict[str, Any]] = None) -> Callable[
                       [int, dict], Optional[dict]]:
    """Read points' measures from the canonical records they name.

    The entry carries only a record name per point and the pool bundle's key;
    the records' deduplicated key tables remain their one copy. Each record is
    read and projected once per fold. No `PointMeasure` is materialized per
    Gallery point: the fold reads the record's point × term matrix.
    """
    timings = {} if timings is None else timings
    bundle = core.get("_measurePool")
    wanted: Dict[str, List[Tuple[int, int, int]]] = {}
    for point in core.get("_theory") or []:
        name = point.get("measureRecord")
        at = _at_view(point, view)
        if at is None:
            continue
        dec = at["sol"]["phases"]["decode"]
        if _stepDraftTokens(dec) != int(view.draft_tokens):
            raise MeasurementFoldMismatch(
                f"frontier point {point.get('label')} binds K="
                f"{_stepDraftTokens(dec)}, not the requested K={view.draft_tokens}")
        if name is not None:
            wanted.setdefault(str(name), []).append(
                (int(point["seq"]), int(dec["batch"]), int(view.draft_tokens)))
    if not wanted:
        raise MeasurementFoldMismatch(
            f"the entry has no K={view.draft_tokens} point with a measure record")
    matrices: Dict[str, Any] = {}
    points: Dict[Tuple[str, str, int, int, int], tuple[Any, int]] = {}

    def read(_i: int, point: dict) -> Optional[dict]:
        name = point.get("measureRecord")
        if bundle is None or name is None:
            raise MeasurementFoldMismatch(
                "a frontier point that can be drawn carries no measure record")
        name = str(name)
        dec = point["sol"]["phases"]["decode"]
        if _stepDraftTokens(dec) != int(view.draft_tokens):
            raise MeasurementFoldMismatch(
                f"frontier point {point.get('label')} binds K="
                f"{_stepDraftTokens(dec)}, not the requested K={view.draft_tokens}")
        at = (name, gpu_sku, int(point["seq"]), int(dec["batch"]),
              int(view.draft_tokens))
        if at not in points:
            if name not in matrices:
                started = time.perf_counter()
                stored = sharedSolPool.record(str(bundle), name)
                timings["measure_record_read_s"] = (
                    timings.get("measure_record_read_s", 0.0)
                    + time.perf_counter() - started)
                timings["measure_records"] = len(matrices) + 1
                if stored is None:
                    raise MeasurementFoldMismatch(
                        f"the serving-law record {name} is not readable")
                # A pool record is `{parallel, walk}`; the law is its walk.
                record = stored.get("walk", stored)
            started = time.perf_counter()
            try:
                matrix = servingMeasureMatrix(
                    record, card=Card.of(gpu_sku),
                    points=wanted.get(name) or [])
            except (KeyError, ValueError) as e:
                raise MeasurementFoldMismatch(
                    f"record {name} has no measure at {gpu_sku}: {e}"
                ) from e
            timings["measure_point_read_s"] = (
                timings.get("measure_point_read_s", 0.0)
                + time.perf_counter() - started)
            matrices[name] = matrix
            for coordinates, row in matrix.point_rows.items():
                points[(name, gpu_sku, *coordinates)] = (matrix, row)
        matrix = matrices[name]
        matrix, row = points[at]
        return {"matrix": matrix, "row": row, "terms": [],
                "phase": "decode"}

    read.timings = timings
    return read


def _measured_rates(core: dict,
                    measureAt: Callable[[int, dict], Optional[dict]],
                    view: FrontierView,
                    scope: Any) -> dict:
    """What the measured fold (`batch_sweep.measure_points`) says of each
    point's step at the view's K under `scope`, by the point's index in the
    entry (`rates`): its measured total (`t_measured_s`, None unless every
    term is measured), which step terms are measured (`measured_terms`, a
    `_termMask`) and the formula's share of them (`measured_part_s`) — and
    its decode coverage (`coverage`, `_PointCoverage`). K selects this
    view's measure table; no floor and no acceptance moves any of these, so
    every view at that K reads them, and the view's floors and acceptance
    enter where it draws (`_at_view`)."""
    index: List[int] = []
    items: List[dict] = []
    timings: Dict[str, Any] = {}
    for i, t in enumerate(core.get("_theory") or []):
        at = _at_view(t, view)
        if at is None:
            continue
        m = measureAt(i, at)
        if m is None:
            raise MeasurementFoldMismatch(
                f"frontier point {t.get('label')} has no K={view.draft_tokens} measure")
        index.append(i)
        items.append({"sol": copy.deepcopy(at["sol"]),
                      "measures": {"decode": m}})
    timings["measure_points"] = len(items)
    total = time.monotonic()
    if items:
        measure_points(items, selection_mode=_GALLERY_SELECTION,
                       measurement_scope=scope, timings=timings,
                       blend_terms=True)
    timings["total_s"] = time.monotonic() - total
    rates: Dict[str, dict] = {}
    coverage = _PointCoverage()
    for i, item in zip(index, items, strict=True):
        dec = item["sol"]["phases"]["decode"]
        if "measured_terms" not in dec:
            raise MeasurementFoldMismatch(
                f"the fold stated no measured terms for point {i}")
        rates[str(i)] = {"t_measured_s": dec.get("t_measured_s"),
                         "measured_terms": _termMask(dec["measured_terms"]),
                         "measured_part_s": dec["measured_part_s"]}
        coverage.add(i, dec["coverage"])
    timings.update(getattr(measureAt, "timings", {}))
    return {"rates": rates, "coverage": coverage.to_json(),
            "timings": timings}


class _PointCoverage:
    """Each folded point's decode coverage (`MeasuredTrack.coverage`) as
    the measured rates keep it: `[measured, total, key indices]` per point,
    each non-measured key named once for the entry (`keys`: `[state, key]`,
    or `[state, key, reason]` when declined). A model's points share most of
    their keys, so each is kept once rather than once per point: DeepSeek-V3
    on H100 has 2,545 of them (~231 KB) and the per-point index lists
    ~69 KB. Kept exact rather than grouped by op signature (~30 KB), because
    what the panel lists is the shape to measure next."""

    def __init__(self) -> None:
        self.keys: List[list] = []
        self.index: Dict[tuple, int] = {}
        self.points: Dict[str, list] = {}

    def add(self, i: int, cov: dict) -> None:
        held: List[int] = []
        for name, items in (cov.get("states") or {}).items():
            for it in items:
                k = ((name, it["key"], it["reason"]) if name == DECLINED
                     else (name, it))
                if k not in self.index:
                    self.index[k] = len(self.keys)
                    self.keys.append(list(k))
                held.append(self.index[k])
        self.points[str(i)] = [cov["measured"], cov["total"], held]

    def to_json(self) -> dict:
        return {"keys": self.keys, "points": self.points}


def frontier_coverage(points: List[list], keys: List[list]) -> dict:
    """How much of one context's SOL frontier Kernel Bench has measured,
    from each frontier point's folded coverage (`_PointCoverage`:
    `[measured, total, key indices]` into `keys`): how many points, how
    many are fully measured (those draw a measured point), the best point's
    `{measured, total}` and the median share of rows measured, and per
    state of a row not measured (`auto_sweep.STATES`) how many distinct
    keys are in it and the `COVERAGE_KEYS_LISTED` most needed, each with
    how many frontier points need it — what to measure next. A key names
    one physical shape, so a model's frontier misses hundreds (DeepSeek-V3
    on H100: 142–279 per context): the list is cut, the count is not."""
    shares = [m / t for m, t, _ in points if t]
    best = max(points, key=lambda p: p[0] / p[1] if p[1] else 0.0, default=None)
    needed: Dict[str, Dict[tuple, int]] = {}
    for _m, _t, held in points:
        for k in held:
            state, *rest = keys[k]
            by_key = needed.setdefault(state, {})
            by_key[tuple(rest)] = by_key.get(tuple(rest), 0) + 1
    states: Dict[str, dict] = {}
    for name in STATES:
        if name not in needed:
            continue
        rows = sorted(needed[name].items(), key=lambda kv: (-kv[1], kv[0]))
        states[name] = {"keys": len(rows), "top": [
            {"key": k[0], **({"reason": k[1]} if name == DECLINED else {}), "points": n}
            for k, n in rows[:COVERAGE_KEYS_LISTED]]}
    return {
        "points": len(points),
        "complete": sum(1 for m, t, _ in points if t and m == t),
        "best": None if best is None else {"measured": best[0], "total": best[1]},
        "median": statistics.median(shares) if shares else None,
        **({"states": states} if states else {}),
    }


def _draw(core: dict, view: FrontierView, state: str,
          rates: Optional[dict]) -> dict:
    """A model's entry drawn under `view`: every theory point at the view's
    floors and draft (`_at_view`), the SOL frontier through them, and the
    measured one through the points' measured rates (`_measured_rates`,
    None when the track reads none), and how much of each context's SOL
    frontier Kernel Bench covers (`measured_track.coverage`,
    `frontier_coverage`; only when the track reads). Under a draft, the
    layouts whose step at that K failed, with why (`draft_errors`; none
    under plain decode). The entry is not mutated."""
    out = _public_entry(core)
    out["draft_errors"] = [e for e in core.get("_draft_errors") or []
                           if view.draft_tokens
                           and e["draft_tokens"] == int(view.draft_tokens)]
    out["measured_track"] = {"state": state}
    records: list[CloudRecord] = []
    # The entry index each drawn point was read off, for its coverage.
    for i, point in enumerate(core.get("_theory") or []):
        coordinates = point_coordinates(
            mode=point.get("mode", "custom"), num_gpus=point["num_gpus"],
            gpu_sku=view.gpu_sku, context=int(point["seq"]),
            batch_key=point.get("batch_key", "max"),
            draft_tokens=int(view.draft_tokens),
            acceptance=float(view.acceptance_rate), ep_load_factor=1.0)
        item = _at_view(point, view,
                        (rates or {}).get("rates", {}).get(str(i)))
        if item is None:
            records.append(CloudRecord.unanswered(
                coordinates, "layout",
                "this layout has no step at the view's draft width",
                token=i))
            continue
        dec = item["sol"]["phases"]["decode"]
        records.append(CloudRecord(
            coordinates=coordinates,
            estimate={"phases": {"decode": dec}},
            provenance={"layout": item["label"]}, token=i))

    cloud = produce_cloud(records)

    def fronts(track: str) -> Dict[str, List[dict]]:
        """The one `frontier_view`, one line per context, as the Gallery
        always drew them (the front across every layout and batch)."""
        drawn = frontier_view(cloud.points, dims=POINT_DIMS, fix={},
                              group_by=["context"], track=track)
        answer = {str(seq): [] for seq in GALLERY_SEQ_LENS}
        for series in drawn["series"]:
            answer[str(series["group"]["context"])] = series["frontier"]
        return answer

    out["frontier"] = fronts("sol")
    out["frontier_measured"] = fronts("measured")
    out["frontier_partial"] = fronts("partial")
    if state == "reads":
        cov = rates["coverage"]
        out["measured_track"]["timings"] = rates.get("timings") or {}
        out["measured_track"]["timings"].update(
            core.get("_measure_timings") or {})
        out["measured_track"]["coverage"] = {
            sk: frontier_coverage([_point_coverage(cov, cloud.tokens[id(p)],
                                                   core)
                                   for p in pts], cov["keys"])
            for sk, pts in out["frontier"].items()}
    return out


class MeasurementFoldMismatch(ValueError):
    """A kept SOL entry and the current measurement fold disagree on points.

    This is an engine bug, not a lake state: the SOL drawing remains valid,
    but the degraded measured drawing is neither cached nor labelled as if
    Kernel Bench had been unreadable.
    """


def _point_coverage(cov: dict, i: int, core: dict) -> list:
    """Frontier point `i`'s folded coverage. A point the fold had no
    measure for would carry the walk's unread coverage — no row in any
    state — which is not a coverage of zero: refused rather than counted
    as one."""
    got = cov["points"].get(str(i))
    if got is None:
        t = core["_theory"][i]
        decode = t["sol"]["phases"]["decode"]
        raise MeasurementFoldMismatch(
            f"frontier point {t['label']} at {t['seq']} "
            f"batch {decode.get('batch')} (K={_stepDraftTokens(decode)}) "
            "has no decode measure to fold")
    return got


def _rates_cover(core: dict, rates: dict, view: FrontierView) -> bool:
    """Whether a rates entry folds every point of this view.

    A draft-only capacity point is a K-step stored for K views; its binding
    is the one statement of which step it is, and the requested view is the
    only obligation of its measured rates.
    """
    try:
        coverage = rates.get("coverage") or {}
        for i, point in enumerate(core.get("_theory") or []):
            at = _at_view(point, view)
            if at is None:
                continue
            decode = at["sol"]["phases"]["decode"]
            if _stepDraftTokens(decode) == int(view.draft_tokens):
                _point_coverage(coverage, i, core)
    except (MeasurementFoldMismatch, KeyError, TypeError):
        return False
    return True


class _Stored(NamedTuple):
    """Where one model's frontier is kept under one walk (`FrontierView.
    walk_parts`) — its entry in `_FRONTIER`, and what views read off it in
    `_FRONTIER_VIEWS` — or nowhere, for a
    model with no `result_version`."""
    model_id: str
    version: Optional[str]

    def _key(self, kind: str, parts: List[Any], scope: Any,
             schema: str = _FRONTIER_SCHEMA) -> str:
        return of_render(kind, self.version, [schema, self.model_id,
                                              *parts], scope).key

    def serving_key(self, view: FrontierView) -> ServingWalkKey:
        """The key of this model's serving-walk bundle for the view's card's
        walk class (`ServingWalkKey.of_card` — the one place a card enters
        a projection key)."""
        return ServingWalkKey.of_card(
            self.version, model=self.model_id,
            card=Card.of(view.gpu_sku),
            facts=ServingWalkFacts.producer())

    def walk_identity(self, view: FrontierView):
        """The shared identity of this model's serving-walk bundle for the
        walk class of the view's card (`ServingWalkKey`): one bundle per
        class, never another class's."""
        return self.serving_key(view).identity

    def keep_walk(self, view: FrontierView, walk: dict,
                  traces: Optional[List[Any]] = None) -> None:
        """Keep a model's class walk, when the view's is the kept one; each
        layout's record with the trace of the job that walked it (`traces`,
        in layout order)."""
        if not (self.version and view.keeps_walk()):
            return False
        walk_key = self.serving_key(view)
        names = [StepLawKey.of_layout(walk_key, item["parallel"]).record_name
                 for item in walk["layouts"]]
        return sharedSolPool.write(
            walk_key.identity,
            zip(names, walk["layouts"], strict=True),
            traces=(dict(zip(names, traces, strict=True))
                    if traces is not None else None),
            metadata={
                "cards": list(walk["cards"]),
                "layoutsByCard": walk["layoutsByCard"],
                "contexts": list(GALLERY_SEQ_LENS),
                "facts": ServingWalkFacts.producer().to_json(),
                "recordSchema": walkRecordSchema,
            })

    def shared_walk(self, view: FrontierView) -> Optional[dict]:
        """The class walk from the local SOL pool, or None on any miss.

        The pool's manifest carries the non-record walk plan. Records are
        read one layout at a time; only the model's walk being drawn enters
        objects, not every model in the pool.
        """
        walk_key = self.serving_key(view)
        bundle_key = walk_key.identity.key
        manifest = sharedSolPool.manifest(bundle_key)
        if not walkManifestIsValid(manifest):
            return None
        metadata = manifest.get("metadata") or {}
        # A canonical bundle holds every public layout (Gallery's frontier ∪
        # Scaling's ladder). The Gallery draws only its own subset of them.
        galleryLayouts = metadata.get("galleryLayoutsByCard",
                                      metadata.get("layoutsByCard", {}))
        if view.gpu_sku not in galleryLayouts:
            return None
        wantedNames = {StepLawKey.of_layout(walk_key, layout).record_name
                       for layout in galleryLayouts[view.gpu_sku]}
        recordIds = sharedSolPool.recordIds(bundle_key)
        if recordIds is None:
            return None
        layouts = []
        for record_name in recordIds:
            if record_name not in wantedNames:
                continue
            item = sharedSolPool.record(bundle_key, record_name)
            if item is None:
                return None
            layouts.append(item)
        return {**metadata, "layouts": layouts}

    def entry_key(self, view: FrontierView) -> str:
        """The entry's: under the constant scope `none` (SOL: no
        measurement moves it), under the walk alone (`FrontierView.
        walk_parts`) — not the floors, K or acceptance: each point carries
        its step as a function of the floors, at every K (`entry_view`)."""
        return self._key("gallery-frontier", view.walk_parts(),
                         NO_MEASUREMENTS)

    def rates_key(self, view: FrontierView, scope: Any) -> str:
        """The measured rates': the selection that picks each key's row
        (`_GALLERY_SELECTION`) is the fold's, and K selects a different
        `PointMeasure` formula. Neither floors nor acceptance move what the
        rates keep (step times and measured terms); a view's enter where it
        draws (`_at_view`)."""
        return self._key("gallery-frontier-measured",
                         [*view.walk_parts(), int(view.draft_tokens),
                          _GALLERY_SELECTION], scope,
                         _DRAWING_SCHEMA)

    def view_key(self, view: FrontierView, scope: Any, state: str) -> str:
        """The drawing's. A track the lake could not give ("unavailable")
        is the SOL drawing (`_labelled`), kept once under scope `none`
        whatever the outage: a lake that flaps between answering and not
        reads both drawings back, and draws neither again off the entry."""
        if state == "unavailable":
            scope, state = NO_MEASUREMENTS, "none"
        return self._key("gallery-frontier-view",
                         [*view.key_parts(), state, _GALLERY_SELECTION],
                         scope, _DRAWING_SCHEMA)

    def kept(self, view: FrontierView, track: Track) -> Optional[dict]:
        """The drawing kept for `view` under `track`, or None."""
        hit = _FRONTIER_VIEWS.get(self.view_key(view, *track))
        return None if hit is None else _labelled(hit, track[1])

    def drawn(self, view: FrontierView, core: dict,
              track: Optional[Track] = None) -> Optional[dict]:
        """`core` drawn under `view` (`_draw`) and `track` (the request's
        lake answer; read here when not given), its measured rates read off
        the cache or folded from the records the entry names. The drawing is
        kept under the view.

        A fold the lake fails mid-request — its version answered (the view
        reads a measured track) but the rows' lookup did not — draws the
        SOL drawing (`measured_track.state` "unavailable") rather than
        failing the whole frontier. It is kept where every unavailable
        track's is (`view_key`), never under the measured track's: a retry
        the lake answers folds the rates."""
        scope, state = track if track is not None else _view_scope(view)
        rates = None
        if state == "reads":
            key = self.rates_key(view, scope) if self.version else None
            rates = _FRONTIER_VIEWS.get(key) if key else None
            if rates is not None and not _rates_cover(core, rates, view):
                rates = None
            if rates is None:
                try:
                    timings: Dict[str, Any] = {}
                    rates = _measured_rates(
                        core, _measureReader(core, view.gpu_sku, view, timings),
                        view, scope)
                    if rates is None or not _rates_cover(core, rates, view):
                        raise MeasurementFoldMismatch(
                            "the fold did not cover every frontier point")
                except MeasurementsUnavailable as e:
                    logger.warning("gallery frontier: %s: measured track "
                                   "dropped: %s", self.model_id, e)
                    scope, state = NO_MEASUREMENTS, "unavailable"
                except MeasurementFoldMismatch as e:
                    logger.error(
                        "gallery frontier engine bug: %s: measured fold "
                        "did not cover the plain frontier: %s", self.model_id,
                        e, exc_info=e)
                    out = _draw(core, view, "engine_error", None)
                    out["measured_track"]["error"] = str(e)
                    return _labelled(out, "engine_error")
                else:
                    if key:
                        _FRONTIER_VIEWS.put(key, rates)
        out = _draw(core, view, "none" if state == "unavailable" else state,
                    rates)
        if self.version:
            _FRONTIER_VIEWS.put(self.view_key(view, scope, state), out)
        return _labelled(out, state)

    def store(self, view: FrontierView, entry: dict) -> dict:
        """Keep a built entry, and return what every view reads."""
        core = entry
        if self.version:
            # Shareable (no user calibration rows) unless the entry is an
            # error: one box's transient failure must not fan out globally.
            shared = not entry.get("error")
            _FRONTIER.put(self.entry_key(view), core, remote_ok=True,
                          remote_if=lambda _part: shared)
        return core


def _read_entry(entry: dict, view: FrontierView) -> dict:
    """A built entry, in hand, drawn under `view` (`_draw`), its measured
    rates folded off its own points' measures. Nothing is kept."""
    return _Stored(entry.get("model_id", ""), None).drawn(view, entry)


def frontier_request_key(view: FrontierView, track: Track) -> str:
    """The identity of one frontier response: the view, the spec generation
    and the measured track it is drawn under (`frontier_track`: the lake's
    scope and its state). Concurrent requests for one view under one lake
    answer join one build, and a response held for a retry
    (`offload.within_deadline`) is only ever served under the track it was
    drawn under: one drawn while the lake was unavailable is not the answer
    once the lake answers, nor one of an older snapshot after a new one."""
    scope, state = track
    return of_render("gallery-frontier-request", spec_generation(),
                     [*view.key_parts(), state], scope).key


async def frontier_track(view: FrontierView) -> Track:
    """The measured track a request for `view` is answered under
    (`_view_scope`), read once per request, before its identity
    (`frontier_request_key`) — one `/internal/bench/version`, off the
    event loop and outside the compute pool (it is a request, not
    compute)."""
    return await asyncio.to_thread(_view_scope, view)


def _frontier_read(model_id: str, view: FrontierView,
                   track: Optional[Track] = None, *,
                   cache_only: bool = False) -> dict:
    """One model under `view` and `track` (the request's lake answer; read
    here when not given), as far as the caches answer it: `{"drawn":
    …}` — the view's drawing kept, else drawn off the kept entry (its named
    measure records read only when the view's measured rates are not kept) —
    else priced off the model's class walk kept for the view's card
    (`_restamp`) — or `{"walk": plan}`: the cards and layouts a class walk
    walks (`FrontierView.walk_cards`, `class_frontier_layouts`, and per
    card the layouts its frontier prices). A whole entry is never read for
    a view already drawn, and its measure records never for floors, K or
    acceptance.
    Runs where the compute runs: what crosses back is the drawing."""
    _config, resolved, version = _resolve_one(model_id)
    stored = _Stored(model_id, version)
    if resolved is None:
        return {"drawn": _draw({"model_id": model_id, "layouts": [],
                                "error": "no spec resolves for this model"},
                               view, "none", None)}
    if version is not None:
        track = track if track is not None else _view_scope(view)
        hit = stored.kept(view, track)
        if hit is not None:
            return {"drawn": hit}
        if cache_only:
            return {}
        core = _FRONTIER.get(stored.entry_key(view), remote_ok=True)
        if core is not None:
            drawn = stored.drawn(view, core, track)
            return {"drawn": drawn}
        if view.keeps_walk():
            walk = stored.shared_walk(view)
            if walk is not None and view.gpu_sku in walk["layoutsByCard"]:
                return {"drawn": _keep_and_draw(
                    stored, view, _restamp(model_id, walk, view.gpu_sku, view),
                    track)}
    try:
        plan = _walk_plan(resolved, view)
    except Exception as e:
        return {"drawn": _draw({"model_id": model_id, "layouts": [],
                                "error": str(e)}, view, "none", None)}
    return {"walk": plan}


def _scalingLayouts(caps: Any, cards: List[str]) -> List[ParallelConfig]:
    """Every layout Scaling's public ladder prices, each once.

    The Gallery's frontier ends at 64 GPUs; the Scaling ladder continues to
    the engine bound in its four canonical modes. Those higher rungs belong to
    a Scaling warm's bundle. They are not produced for every Gallery model:
    only the panel's featured cohort reaches past the frontier's 64-GPU bound.
    """
    out = list(class_frontier_layouts(caps, cards))
    for mode in ALL_MODES:
        out += [cfg for _x, cfg in parallel_configs_for_mode(
            mode, caps, x_max=SWEEP_MAX_GPUS) if cfg not in out]
    return out


def _walk_plan(resolved: Any, view: FrontierView, *,
               scaling: bool = False) -> dict:
    """What a model's canonical walk for `view` covers: its cards — the view
    card's walk class (`FrontierView.walk_cards`, the walk card first): a
    law's rows are one class's, so a bundle never spans two —, per card the
    layouts a Gallery frontier on it renders (`frontier_layouts`), and every
    layout any producer reads — with `scaling`, the public ladder's extra
    layouts (`_scalingLayouts`)."""
    cards = view.walk_cards()
    caps = sweep_caps({}, resolved=resolved)
    frontierByCard = {
        sku: [cfg.to_dict() for cfg in frontier_layouts(caps, sku)]
        for sku in cards
    }
    layouts = (_scalingLayouts(caps, cards) if scaling
               else class_frontier_layouts(caps, cards))
    return {"cards": cards,
            "layoutsByCard": {sku: [cfg.to_dict() for cfg in layouts]
                              for sku in cards},
            "galleryLayoutsByCard": frontierByCard,
            "layouts": [cfg.to_dict() for cfg in layouts]}


def _keep_and_draw(stored: _Stored, view: FrontierView, entry: dict,
                   track: Optional[Track]) -> dict:
    """A model's entry on the view's card kept (`_Stored.store`) and drawn
    under `view` and `track` (the request's one lake answer)."""
    core = stored.store(view, entry)
    return stored.drawn(view, core, track)


def _finish_frontier(model_id: str, view: FrontierView, plan: dict,
                     results: List[dict],
                     track: Optional[Track] = None,
                     traces: Optional[List[Any]] = None) -> dict:
    """A model's layout laws (`_layout_law`, one per layout of `plan`) as its
    class bundle, kept (`_Stored.keep_walk`), projected on the view's card
    (`_restamp`), kept and drawn under `view` and `track`. One job, so the
    entry is built, kept and drawn in one process and only the drawing
    crosses back.

    The walk is kept before the card's entry is priced, and per model: a
    build no request waits on any more is cancelled (`offload.UNWATCHED_S`)
    and what it had kept stays kept, so the models it got through are a
    walk every card of the class reads — including the card the user
    switched to, which is the request that cancelled this one."""
    walk = {**plan, "layouts": [{"parallel": p, "walk": r} for p, r in
                                zip(plan["layouts"], results, strict=True)]}
    stored = _Stored(model_id, _resolve_one(model_id)[2])
    stored.keep_walk(view, walk, traces)
    return _keep_and_draw(
        stored, view, _restamp(model_id, walk, view.gpu_sku, view), track)


async def keepSharedWalk(model_id: str, gpu_sku: str) -> bool:
    """`_keptSharedWalk` as one recorded computation
    (`code_trace.provider.recording`)."""
    with recording():
        return await _keptSharedWalk(model_id, gpu_sku)


async def _keptSharedWalk(model_id: str, gpu_sku: str) -> bool:
    """Ensure one card class's serving-walk records exist in the shared pool.

    A page need not be rendered for a class to contribute canonical facts:
    the Scaling series includes RTX-4090 while the Gallery picker does not.
    Existing complete bundles are left alone; immutable records are never
    rewritten or migrated.
    """
    _config, resolved, version = _resolve_one(model_id)
    if resolved is None or version is None:
        return False
    view = FrontierView(gpu_sku)
    stored = _Stored(model_id, version)
    walk_key = stored.serving_key(view)
    bundle_key = walk_key.identity.key
    recordIds = sharedSolPool.recordIds(bundle_key)
    plan = _walk_plan(resolved, view, scaling=True)
    stepKeys = [StepLawKey.of_layout(walk_key, layout)
                for layout in plan["layouts"]]
    wanted = {key.record_name for key in stepKeys}
    existing = {}
    traces: Dict[str, Any] = {}
    if recordIds is not None:
        if walkManifestIsValid(sharedSolPool.manifest(bundle_key)):
            for name in wanted:
                # A record whose code changed reads as a miss
                # (`SolPool.record`): only it is walked again.
                record = sharedSolPool.record(bundle_key, name)
                if (record is not None
                    and record.get("walk", {}).get("schema") == walkRecordSchema):
                    existing[name] = record
                    traces[name] = sharedSolPool.recordTrace(bundle_key, name)
        published = {
            StepLawKey.of_layout(walk_key, layout).record_name
            for layouts in ((sharedSolPool.manifest(bundle_key).get(
                "metadata") or {}).get("layoutsByCard") or {}).values()
            for layout in layouts
        }
        if len(existing) == len(wanted) and wanted <= published:
            return True
    layouts = [ParallelConfig.from_dict(layout) for layout in plan["layouts"]
               if StepLawKey.of_layout(walk_key, layout).record_name
               not in existing]
    results = await map_off_loop(
        functools.partial(run_recorded, None, _layout_law),
        [(model_id, plan["cards"], float(view.gpu_memory_fraction),
          _GALLERY_PRECISION.canonical(), layout)
         for layout in layouts])
    computed = {}
    for layout, (result, trace) in zip(layouts, results, strict=True):
        name = StepLawKey.of_layout(walk_key, layout).record_name
        computed[name] = {"parallel": layout.to_dict(), "walk": result}
        traces[name] = trace
    return sharedSolPool.write(
        walk_key.identity,
        ((name, existing.get(name) or computed[name])
         for name in (key.record_name for key in stepKeys)),
        traces=traces,
        metadata={
            "cards": list(plan["cards"]),
            "layoutsByCard": plan["layoutsByCard"],
            "contexts": list(GALLERY_SEQ_LENS),
            "facts": ServingWalkFacts.producer().to_json(),
            "recordSchema": walkRecordSchema,
        })


async def frontier_model(model_id: str, view: FrontierView,
                         track: Optional[Track] = None, *,
                         compute: Optional[PrioritySlots] = None,
                         cache_only: bool = False) -> dict:
    """`_frontier_model` as one recorded computation
    (`code_trace.provider.recording`): the read, the layout jobs and the fold
    (each a worker job whose trace comes back to it), so every part it
    keeps is stored with the code that made it."""
    with recording():
        return await _frontier_model(
            model_id, view, track, compute=compute, cache_only=cache_only)


async def _frontier_model(model_id: str, view: FrontierView,
                          track: Optional[Track] = None, *,
                          compute: Optional[PrioritySlots] = None,
                          cache_only: bool = False) -> dict:
    """One model's frontier under `view` and `track`: read off the caches —
    the view's drawing, its entry, or its class walk priced on its card —
    or each layout of the class walk its own compute job and the fold
    (`_finish_frontier`). Kernel Bench rows fold on after the SOL entry is
    in hand, so the lake's answer reaches the pricing and never the walk.

    The read is one quick job and queues for nothing, so a model already
    kept lands at once whatever is computing. Only a model that computes
    holds one of `compute` (the frontier's model slots), and its steps queue
    at its urgency (`offload.urgency`): the model a client picked, or has on
    screen, before the rest."""
    got = await run_off_loop(
        _frontier_read, model_id, view, track, cache_only=cache_only)
    if "drawn" in got:
        return got["drawn"]
    if cache_only:
        return {}
    if compute is None:
        return await _compute_frontier(model_id, view, track, got)
    async with compute.hold(urgency(model_id)):
        return await _compute_frontier(model_id, view, track, got)


async def _compute_frontier(model_id: str, view: FrontierView,
                            track: Optional[Track], got: dict) -> dict:
    """The frontier of a model its read did not answer: each layout a job,
    then the fold."""
    plan = got["walk"]
    # Each layout is its own recorded job, so the record it becomes names
    # only the code that walked it (`SolPool.write`).
    recorded = await map_off_loop(
        functools.partial(run_recorded, None, _layout_law),
        [(model_id, plan["cards"], float(view.gpu_memory_fraction),
          _GALLERY_PRECISION.canonical(), p)
                      for p in plan["layouts"]], name=model_id)
    results = [value for value, _trace in recorded]
    traces = [trace for _value, trace in recorded]
    return await run_ranked(_finish_frontier, model_id, view, plan, results,
                            track, traces, name=model_id)


async def build_frontier(view: FrontierView,
                         track: Optional[Track] = None) -> dict:
    """Every model's decode frontier under `view`, `_model_concurrency`
    models computing at a time, reporting how many have finished and
    publishing each as it lands (`report_partial`), so a page draws it
    without waiting for the slowest. Every model is read first, at once: a
    kept one lands without waiting behind one that computes. Which model
    computes next is the client's order (`offload.urgency`: picked, then on
    screen). Every model is drawn under one lake answer, `track` — the one
    the request's identity names (`frontier_request_key`), read here
    (`frontier_track`) when the caller has none (the warm): one response
    reads one lake, and an unanswering lake costs one timeout rather than
    one per model. A model whose fold the lake fails mid-request is drawn
    "unavailable" under a "reads" track; such a response is not that
    track's answer, so it is handed to the requests waiting on it and held
    for none (`offload.report_degraded`). A spec refined meanwhile serves
    nothing (`ensure_spec_generation`)."""
    generation = spec_generation()
    model_ids = sorted(iter_snapshot_ids())
    head = _frontier_head(view)
    report_partial(head=head)
    if track is None:
        track = await frontier_track(view)
    slots = PrioritySlots(_model_concurrency())
    done = 0
    report_progress(done, len(model_ids))

    async def one(mid: str) -> dict:
        nonlocal done
        if track is not None and track[1] == "reads":
            entry = await frontier_model(
                mid, view, track, cache_only=True)
            if entry:
                done += 1
                print(f"frontier {view.gpu_sku}: {done}/{len(model_ids)} · {mid}",
                      flush=True)
                report_progress(done, len(model_ids))
                report_partial(entry)
                return entry
            # SOL is a first-class partial answer: publish and keep it before
            # any measurement lookup or fold is allowed to delay the page.
            sol = await frontier_model(
                mid, view, (NO_MEASUREMENTS, "none"), compute=slots)
            report_partial(sol)
            entry = await frontier_model(mid, view, track, compute=slots)
            report_partial(entry)
        else:
            entry = await frontier_model(mid, view, track, compute=slots)
        done += 1
        print(f"frontier {view.gpu_sku}: {done}/{len(model_ids)} · {mid}",
              flush=True)
        report_progress(done, len(model_ids))
        report_partial(entry)
        return entry

    models = list(await asyncio.gather(*[one(m) for m in model_ids]))
    ensure_spec_generation(generation)
    if track[1] != "unavailable" and any(
            (m.get("measured_track") or {}).get("state") == "unavailable"
            for m in models):
        report_degraded()
    return {
        **head,
        "models": models,
        "errors_count": sum(1 for m in models if m.get("error")),
    }


@dataclass(frozen=True)
class _WarmLayoutWork:
    """One offline layout job, with the cost that ordered it in the queue."""

    model_id: str
    layout_index: int
    cost: int
    item: tuple[Any, ...]


def _warm_layout_works(
        walks: list[tuple[str, dict]], view: FrontierView,
        model_params: Mapping[str, Any]) -> list[_WarmLayoutWork]:
    """Every missing model × layout job, most expensive first.

    The cost is the existing main-model parameter fact times the layout's GPU
    count. It is a scheduling estimate, not a persisted fact and not a second
    identity: its only owner is this offline queue.
    """
    out: list[_WarmLayoutWork] = []
    for model_id, plan in walks:
        params = int(model_params.get(model_id) or 0)
        for index, layout in enumerate(plan["layouts"]):
            gpus = ParallelConfig.from_dict(layout).cluster_size
            out.append(_WarmLayoutWork(
                model_id=model_id, layout_index=index,
                cost=max(1, params) * int(gpus),
                item=(model_id, plan["cards"],
                      float(view.gpu_memory_fraction),
                      _GALLERY_PRECISION.canonical(), layout)))
    return sorted(out, key=lambda work: (-work.cost, work.model_id,
                                          work.layout_index))


def _layout_job(work: _WarmLayoutWork) -> tuple[dict, Any]:
    """One offline layout walk, isolated to that layout's own answer."""
    try:
        return run_recorded(None, _layout_law, work.item)
    except Exception as exc:
        return ({"error": str(exc)}, None)


async def warm_frontier(view: FrontierView,
                        model_params: Mapping[str, Any]) -> dict:
    """Warm one card by putting every model's layouts in one ranked queue.

    This is the offline shape of `build_frontier`, not its request path. All
    missing layouts are submitted together through the existing `run_ranked`
    cell slots, largest first; a model's entry is finished only after every
    one of its layouts has landed, then `_finish_frontier` stores the same
    walk and Gallery entry the interactive path stores.
    """
    generation = spec_generation()
    model_ids = sorted(iter_snapshot_ids())
    head = _frontier_head(view)
    report_partial(head=head)
    track = await run_off_loop(_view_scope, view)

    read_slots = asyncio.Semaphore(_model_concurrency())

    async def read_one(model_id: str) -> dict:
        async with read_slots:
            try:
                return await run_off_loop(
                    _frontier_read, model_id, view, track)
            except Exception as exc:
                return {"drawn": _draw(
                    {"model_id": model_id, "layouts": [], "error": str(exc)},
                    view, "none", None)}

    reads = list(await asyncio.gather(*[read_one(mid) for mid in model_ids]))
    drawings: dict[str, dict] = {}
    walks: list[tuple[str, dict]] = []
    for model_id, got in zip(model_ids, reads, strict=True):
        if "drawn" in got:
            drawings[model_id] = got["drawn"]
        else:
            walks.append((model_id, got["walk"]))

    works = _warm_layout_works(walks, view, model_params)
    print(f"frontier {view.gpu_sku}: {len(walks)} models to walk · "
          f"{len(works)} layouts · largest first", flush=True)
    outcomes: dict[str, list[tuple[dict, Any] | None]] = {
        model_id: [None] * len(plan["layouts"])
        for model_id, plan in walks
    }
    remaining = {model_id: len(plan["layouts"]) for model_id, plan in walks}
    finish_tasks: dict[str, Any] = {}
    layouts_done = 0
    models_done = 0
    report_progress(0, len(works))

    async def finish_one(model_id: str, plan: dict) -> dict:
        if outcomes[model_id]:
            answers, traces = zip(*(outcomes[model_id]), strict=True)
            answers, traces = list(answers), list(traces)
        else:
            answers, traces = [], []
        try:
            out = await run_ranked(
                _finish_frontier, model_id, view, plan, answers,
                track, traces, name=model_id)
            nonlocal models_done
            models_done += 1
            print(f"frontier {view.gpu_sku}: {models_done}/{len(walks)} · "
                  f"{model_id}", flush=True)
            return out
        except Exception as exc:
            return _draw({"model_id": model_id, "layouts": [],
                          "error": str(exc)}, view, "none", None)

    def maybe_finish(model_id: str, plan: dict) -> None:
        if remaining[model_id] == 0:
            finish_tasks[model_id] = asyncio.create_task(
                finish_one(model_id, plan))

    for model_id, plan in walks:
        maybe_finish(model_id, plan)

    async def layout(work: _WarmLayoutWork) -> None:
        nonlocal layouts_done
        try:
            outcomes[work.model_id][work.layout_index] = await run_ranked(
                _layout_job, work, name=work.model_id)
        except Exception as exc:
            outcomes[work.model_id][work.layout_index] = (
                {"error": str(exc)}, None)
        remaining[work.model_id] -= 1
        layouts_done += 1
        report_progress(layouts_done, len(works))
        for model_id, plan in walks:
            if model_id == work.model_id:
                maybe_finish(model_id, plan)
                break

    await asyncio.gather(*[asyncio.create_task(layout(work))
                           for work in works])
    finished = (await asyncio.gather(
        *(finish_tasks[model_id] for model_id in sorted(finish_tasks)))
        if finish_tasks else [])
    by_model = dict(zip(sorted(finish_tasks), finished, strict=True))
    models = [drawings.get(mid) or by_model[mid] for mid in model_ids]
    ensure_spec_generation(generation)
    return {
        **head,
        "models": models,
        "errors_count": sum(1 for model in models if model.get("error")),
    }


def _frontier_head(view: FrontierView) -> dict:
    """What every model of a frontier response is drawn under: the view's
    config and the contexts. The same head frames a partial answer
    (`report_partial`) and the whole one."""
    return {
        "config": {
            "gpu_sku": view.gpu_sku,
            "comm_latency_us": view.comm_latency_us,
            "launch_floor_us": view.launch_floor_us,
            "draft_tokens": view.draft_tokens,
            "acceptance_rate": (view.acceptance_rate if view.draft_tokens
                                else None),
            "max_gpus": FRONTIER_MAX_GPUS,
            "output_tokens": GALLERY_OUTPUT_TOKENS,
            "precision": _GALLERY_PRECISION.to_json(),
        },
        "seq_lens": list(GALLERY_SEQ_LENS),
    }
