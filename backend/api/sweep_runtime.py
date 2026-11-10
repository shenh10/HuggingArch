"""Cached sweep compute, below the HTTP route module.

The Scaling-panel warm job and the `/inference/estimate-sweep` /
`/inference/estimate-parallel-sweep` handlers must write and read the SAME
`KeyedCache`. The route module imports FastAPI `app`, so putting this
compute there forced `warm` to lazy-import the route (and hide the
`_app → warm → routes.inference → _app` cycle). This module is the
composition root for keying + compute + store; it does not import `app`.
"""
from __future__ import annotations

import asyncio
import logging
import os
import dataclasses
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, FrozenSet, Iterator, List, Mapping,
                    Optional, Tuple, get_args)

from fastapi.responses import Response

from backend.arch.loader import get_default_loader
from backend.code_trace.provider import recording
from backend.fingerprint import engine_version, result_version
from backend.hf.metadata import fetch_config
from backend.sweep_cache_artifact import SWEEP_CACHE
from backend.measure.cache import (DiskBackend, KeyedCache, MemoryBackend,
                                   TieredBackend, env_bytes, private_copy)
from backend.api._models import (
    CompareFrontierRequest,
    InferenceEstimateSweepRequest,
    InferenceParallelSweepRequest,
    ParallelFrontierRequest,
)
from backend.api.estimate_env import resolve_estimate_environment
from backend.measure.scope import (NONE as NO_MEASUREMENTS, ResultIdentity,
                                   of_request, of_sweep_step,
                                   theory_of_request)
from backend.api.offload import (
    call_key,
    ensure_spec_generation,
    map_off_loop,
    render_json,
    report_partial,
    report_progress,
    spec_generation,
    within_deadline,
)
from backend.inference.batch_sweep import measure_points
from backend.inference.calibration_overlay import looks_up
from backend.inference.capacity import (CapacitySweepPlan, SweepCloudDim,
                                        iter_sweep_cell, point_served)
from backend.inference.card import Card
from backend.inference.frontier import frontier_view
from backend.inference.sol_data import (
    ServingWalkKey,
    walkManifestIsValid,
)
from backend.inference.sol_data import sharedSolPool
from backend.inference.serving_projection import (
    PublicServingLawRequest,
    projectServingLawPoints,
)
from backend.inference.serving_view import servingLawPointView
from backend.measure import lookup as _measure_lookup
from backend.inference.parallel_sweep import (
    PER_POINT_FIELDS,
    ParallelSweepPlan,
    SCALING_DECODE_LADDER,
    config_for_degree,
    decode_frontier,
    run_layout_cell,
)


_log = logging.getLogger(__name__)


def resolved_spec_for(model_id: str) -> Any:
    """The resolved spec for a model, or None when it cannot be resolved
    right now (unfetchable config, no matching spec). Cheap: ``fetch_config``
    is cached and a warm spec match is ~1 ms."""
    if not model_id:
        raise ValueError("resolved_spec_for: empty model_id")
    try:
        config = fetch_config(model_id)
    except (OSError, ValueError) as exc:
        _log.warning("resolved_spec_for(%s): config unavailable (%s)", model_id, exc)
        return None
    if config is None:
        return None
    resolved = get_default_loader().match(config, model_id=model_id)
    return resolved if resolved is not None and resolved.spec is not None else None


# Declared once with the artifact that bootstraps it at container start.
_SWEEP_CACHE_DIR = SWEEP_CACHE.directory()
# Both sweep tiers share the directory's one budget (the directory is the
# disk the operator sizes); finished answers keep the larger share, the
# unfinished steps of one running sweep the smaller.
_SWEEP_DISK_BYTES = env_bytes("HA_SWEEP_CACHE_BYTES", 512 << 20)
_SWEEP = KeyedCache(
    "sweep",
    TieredBackend(
        MemoryBackend(max_entries=256),
        # Featured models × DEFAULT_SWEEP_GPUS × 4 modes × 8 degrees is ~800
        # files; --all-gpus × the same grid is a few thousand.
        SWEEP_CACHE.disk(max_files=8000,
                         max_bytes=_SWEEP_DISK_BYTES * 3 // 4),
    ),
    single_flight=True,
)


#: The steps of sweeps that have not finished, kept as each one does
#: (`_Partials`). Its own directory and cap, and never the shared remote
#: tier: a step is one machine's unfinished work, read back only by the next
#: request for the same sweep on the same box, where the finished answer is
#: what gets published. Separate from `_SWEEP` so a long sweep's steps
#: cannot evict finished answers from it.
_SWEEP_STEP = KeyedCache(
    "sweep-step",
    TieredBackend(
        MemoryBackend(max_entries=512),
        DiskBackend(_SWEEP_CACHE_DIR / "steps", max_files=16000,
                    max_bytes=_SWEEP_DISK_BYTES // 4),
    ),
)

# `sharedSolPool` is imported from the inference data layer: a Scaling theory
# cell and a Gallery layout walk are two consumers of the same pool, not two
# pools configured to point at one directory.
poolOnly = os.environ.get("HA_SOL_POOL_ONLY", "").lower() in (
    "1", "true", "yes")


def set_sweep_remote(remote: Any) -> None:
    """Inject the durable shared remote tier for the sweep cache. Called by the
    service layer at import (see backend/service/__init__.py) so this core
    module never imports `backend.service` — keeping `backend.api`
    service-clean. Unset → local-only (OSS self-host without Supabase)."""
    _SWEEP.set_remote(remote)


# The parallel sweep's cache key is the request minus exactly the fields every
# point sets for itself (`PER_POINT_FIELDS`, declared beside the code that
# overrides them). Every other field reaches some point's result —
# `avg_output_tokens` among them: the request length `in + out` sizes the KV
# per request, so it moves every decode batch ceiling.


def _sweep_version(kind: str, req: Any) -> Optional[str]:
    """The engine part of a sweep's identity: the model's own
    `result_version` (an edit to data this model cannot read leaves its
    sweeps cached), with the estimate-sweep payload's schema — or None when
    the model has no resolved spec right now (no spec, or the config could
    not be fetched): the request is computed uncached rather than cached
    under a key every such request would share."""
    if not req.model_id:
        return None
    resolved = resolved_spec_for(req.model_id)
    if resolved is None:
        return None
    version = result_version(resolved)
    schema = (_ESTIMATE_SWEEP_PAYLOAD if kind == _ESTIMATE.kind
              else _PARALLEL_SWEEP_PAYLOAD)
    return f"{version}|{schema}"


def sweep_theory_identity(kind: str, req: Any, *, answer: bool = False
                          ) -> Optional[ResultIdentity]:
    """The identity of a sweep's points, or of its assembled SOL answer.
    No scope and no measurement-only field reaches it, so one SOL answer
    serves every selection mode, every Kernel Bench state and every user.
    The parallel sweep drops the fields its points set for themselves
    (`PER_POINT_FIELDS`). This identity keys request-cache entries and
    resumed steps; it never names a record in the canonical law pool."""
    version = _sweep_version(kind, req)
    if version is None:
        return None
    return theory_of_request(f"{kind}-sol" if answer else kind, req, version,
                             drop=PER_POINT_FIELDS if kind == "parallel" else frozenset())


def sweep_cache_key(kind: str, req: Any) -> Optional[str]:
    """The key the sweep's answer is cached under, or None.

    Only an answer that reads no measurement is cached — the SOL answer
    (`theoretical_only`, or `sol_only`), assembled and keyed on its own
    content: no Kernel Bench state, no selection mode, no user reaches it.
    An answer that reads measurements is never stored: it computes its
    measures live, makes the lookup and folds, so it is never older than
    what Kernel Bench holds and a Kernel Bench write invalidates nothing."""
    if req.reads_measurements():
        return None
    identity = sweep_theory_identity(kind, req, answer=True)
    return None if identity is None else identity.key


def _sharedWalkResults(sweep: _CachedSweep, plan: Any) -> dict[int, Any]:
    """Canonical law records a sweep can project without running a cell.

    Only the public anchor is reused: the request's context is one the record
    walked, its batch is each card's own capacity, and every workload fact
    matches the producer anchor. A layout outside the canonical bundle simply
    has no record and runs as its own cell.
    """
    if sweep.kind not in ("parallel", "sweep") or not getattr(plan, "cells", None):
        return {}

    requestAxes: dict[str, Any]

    if sweep.kind == "sweep":
        knobs = getattr(plan, "knobs", None)
        points = getattr(plan, "points", None)
        if (not getattr(plan, "gridded", False) or knobs is None
                or points is None):
            return {}
        environment = {
            "avg_output_tokens": plan.avg_output_tokens,
            "gpu_memory_fraction": knobs.gpu_memory_fraction,
            "precision": knobs.precision,
            "s_prefix_cached": knobs.s_prefix_cached,
            "s_verify": knobs.s_verify,
            "acceptance_rate": knobs.acceptance_rate,
            "speculative_draft": knobs.speculative_draft,
            "include_aux_weights": knobs.include_aux_weights,
            "host_offload": knobs.host_offload,
            "framework": knobs.framework,
            "framework_options": knobs.framework_options,
            "kv_block_size": knobs.kv_block_size,
            "prefill_chunk_size": plan.prefill_chunk_size,
            "ep_load_factor": knobs.ep_load_factor,
            "comm_latency_us": knobs.comm_latency_us,
            "launch_floor_us": knobs.launch_floor_us,
        }
        contexts = tuple(int(context) for context in plan.seq_lens)
        decode_batches = tuple(
            decode for _key, _prefill, decode in points)
        requestAxes = {
            "environment": environment,
            "contexts": contexts,
            "decode_batches": decode_batches,
            "prefill_batches": tuple(
                int(prefill) for _key, prefill, _decode in points),
            "prefix_tokens": tuple(
                int(prefix) for prefix in plan.s_prefixes),
            "verify_widths": tuple(
                int(width) for width in plan.s_verifies),
            "measurement_scope": knobs.measurement_scope,
        }
        if not PublicServingLawRequest.covers(
                environment=environment, contexts=contexts,
                measurement_scope=knobs.measurement_scope):
            return {}

        def gridRequest(walk: ServingWalkKey, card: str,
                        layout: Any) -> PublicServingLawRequest:
            return PublicServingLawRequest(
                walk=walk, card=card, layout=layout, **requestAxes)

        modelId = knobs.model_id
    else:
        if plan.batch != "max":
            return {}
        payload = plan.cells[0].payload
        environment = payload
        requestAxes = {
            "environment": environment,
            "contexts": (int(payload["avg_input_tokens"]),),
            "decode_batches": (None,),
            "prefill_batches": (1,),
            "prefix_tokens": (int(payload.get("s_prefix_cached") or 0),),
            "verify_widths": (int(payload.get("s_verify") or 1),),
            "measurement_scope": payload.get("measurement_scope"),
        }
        if not PublicServingLawRequest.covers(
                environment=environment,
                contexts=requestAxes["contexts"],
                measurement_scope=payload.get("measurement_scope")):
            return {}

        def gridRequest(walk: ServingWalkKey, card: str,
                        layout: Any) -> PublicServingLawRequest:
            return PublicServingLawRequest(
                walk=walk, card=card, layout=layout, **requestAxes)

        modelId = payload["model_id"]

    resolved = resolved_spec_for(modelId)
    if resolved is None:
        return {}
    version = result_version(resolved)
    # One bundle per walk class (`ServingWalkKey.of_card`): a card is projected
    # only off its own class's records, each named by that card's class.
    classWalks: dict[str, Any] = {}

    def classWalk(sku: str) -> Any:
        card = Card.of(sku)
        classKey = card.klass.key()
        if classKey not in classWalks:
            walk_key = ServingWalkKey.of_card(
                version, model=modelId,
                card=card)
            manifest = sharedSolPool.manifest(walk_key.identity.key)
            classWalks[classKey] = (
                (walk_key, manifest.get("metadata") or {})
                if walkManifestIsValid(manifest) else None)
        return classWalks[classKey]

    out: dict[int, Any] = {}
    cells = plan.cells() if callable(plan.cells) else plan.cells
    for index, cell in enumerate(cells):
        if sweep.kind == "sweep":
            modelWalk = classWalk(plan.gpu_sku)
            if modelWalk is None:
                break
            walk_key, metadata = modelWalk
            if plan.gpu_sku not in metadata.get("layoutsByCard", {}):
                break
            projectionRequest = gridRequest(
                walk_key, plan.gpu_sku, plan.knobs.parallel)
            stepKey = projectionRequest.law_key
            item = sharedSolPool.record(stepKey.bundle_key, stepKey.record_name)
            if item is None:
                break
            law = item["walk"]
            if "refused" in law or "error" in law:
                break
            points = [decode for _key, _prefill, decode in plan.points]
            try:
                projectedRows = projectServingLawPoints(law, projectionRequest)
            except Exception:
                break
            contexts = []
            for row in projectedRows:
                # Drawing surfaces filter with `point_served`; the projection
                # itself prices an over-batch or beyond-context point, exactly
                # as the direct sweep does.
                pointsOut = [{
                    "sol": served_sweep_point(point),
                    "measures": {},
                } for point in row]
                contexts.append(pointsOut)
            out[index] = contexts
            continue

        projected: Any = None
        byClass: dict[str, list[str]] = {}
        for sku in plan.gpu_skus:
            byClass.setdefault(Card.of(sku).klass.key(), []).append(sku)
        walks = [(classWalk(skus[0]), skus) for skus in byClass.values()]
        if not walks or any(walk is None for walk, _skus in walks):
            break
        for (walk_key, metadata), skus in walks:
            if any(sku not in metadata.get("layoutsByCard", {})
                   for sku in skus):
                projected = None
                break
            firstRequest = gridRequest(walk_key, skus[0], cell.parallel)
            stepKey = firstRequest.law_key
            item = sharedSolPool.record(stepKey.bundle_key, stepKey.record_name)
            if item is None:
                projected = None
                break
            law = item["walk"]
            if "refused" in law:
                projected = {"refused": law["refused"]}
                continue
            if "error" in law:
                projected = {"error": law["error"]}
                continue
            cards = {}
            failed = False
            for sku in skus:
                try:
                    context = int(cell.payload["avg_input_tokens"])
                    request = gridRequest(walk_key, sku, cell.parallel)
                    ladderRequest = PublicServingLawRequest(
                        walk=walk_key, card=sku, layout=cell.parallel,
                        environment=request.environment,
                        contexts=(context,),
                        decode_batches=(None, *SCALING_DECODE_LADDER),
                        prefill_batches=(
                            1, *(1 for _ in SCALING_DECODE_LADDER)),
                        prefix_tokens=request.prefix_tokens[:1],
                        verify_widths=request.verify_widths[:1],
                        measurement_scope=request.measurement_scope,
                        draft_tokens=request.draft_tokens)
                    projectedRow = projectServingLawPoints(
                        law, ladderRequest)[0]
                    served = projectedRow[0]
                    byBatch = {
                        batch: point for batch, point in zip(
                            ladderRequest.decode_batches, projectedRow,
                            strict=True)
                    }
                    ladder = []
                    ceiling = int(served["max_decode_batch_per_gpu"] or 0)
                    beyond = bool(served["beyond_max_context"])
                    for batch in SCALING_DECODE_LADDER:
                        point = byBatch[batch]
                        if not point_served(max_batch_per_gpu=ceiling,
                                            beyond_max_context=beyond,
                                            requests=int(batch)):
                            continue
                        ladder.append({
                            "phases": {"decode": point["phases"]["decode"]},
                            "batch_key": f"b{int(batch)}",
                            **({"beyond_max_context": True} if beyond else {}),
                        })
                    if point_served(max_batch_per_gpu=ceiling,
                                    beyond_max_context=beyond,
                                    requests=1):
                        ladder.append({
                            "phases": {"decode": served["phases"]["decode"]},
                            "batch_key": "max",
                        })
                    cards[sku] = {"sol": served, "measures": {},
                                  "ladder": ladder}
                except Exception as exc:
                    _log.warning("serving-law projection failed for %s: %s",
                                 sku, exc)
                    failed = True
                    break
            if failed:
                # One class's card the grid cannot answer: the whole cell is
                # estimated, never drawn with that class's cards missing.
                projected = None
                break
            group = {"cards": cards}
            if projected is None:
                projected = group
            elif "cards" in projected and "cards" in group:
                projected = {**projected, "cards": {
                    **projected["cards"], **group["cards"]}}
            elif projected != group:
                projected = None
                break
        if projected is not None:
            out[index] = projected
    return out


#: The estimate-sweep payload's schema, folded into its cache keys
#: (`_sweep_version`): the payload is shaped here, outside the code the
#: engine version hashes, so a change of its shape bumps this — bump on
#: edit. v2: a point carries no per-row table (no `flops_by_module`,
#: `op_table_total` or phase `formulas`) and an encoder stack no per-op
#: rows or module table. v3: the batch's routing skew is a workload input
#: (`ep_load_factor` beside the deployment, not inside its `parallel`), so
#: a request that stated it in its parallel preset keys differently. v4:
#: that cloud states the one point-cloud vocabulary and keeps absent points.
#: v6: a point is the serving projection's public vocabulary, and the direct
#: estimate is normalized through the same owner as a StepLaw projection.
_ESTIMATE_SWEEP_PAYLOAD = "estimate-sweep-v6"

#: The parallel answer's server-side point cloud changed to the one coordinate
#: vocabulary and began keeping absent points, so its cached answers move once.
_PARALLEL_SWEEP_PAYLOAD = "parallel-sweep-v2"


def served_sweep_point(result: Any) -> Any:
    """The serving projection owner, kept under the sweep response's name."""
    return servingLawPointView(result)


def _estimate_sweep_env(req: InferenceEstimateSweepRequest) -> dict:
    env = resolve_estimate_environment(req)
    if req.parallel_preset and not req.parallel:
        mode = str(req.parallel_preset.get("mode") or "")
        degree = int(req.parallel_preset.get("degree") or 0)
        rho = req.parallel_preset.get("ep_load_factor")
        parallel = config_for_degree(mode, degree)
        errs = parallel.validate()
        if errs:
            raise ValueError("; ".join(errs))
        env["parallel"] = parallel
        if rho is not None:
            env["ep_load_factor"] = float(rho)
    return env


def _parallel_plan(req: InferenceParallelSweepRequest) -> ParallelSweepPlan:
    return ParallelSweepPlan.of(
        modes=req.modes,
        context_len=req.context_len,
        batch=req.batch if req.batch is not None else "max",
        gpu_skus=req.gpu_skus,
        base_env=resolve_estimate_environment(req),
        ep_load_factor=req.ep_load_factor,
    )


def _parallel_cells(plan: ParallelSweepPlan) -> list:
    return list(plan.cells)


def _parallel_assemble(plan: ParallelSweepPlan, results: list) -> dict:
    return plan.assemble(results)


def _estimate_plan(req: InferenceEstimateSweepRequest) -> CapacitySweepPlan:
    grid = [bp.model_dump() for bp in req.batch_grid] if req.batch_grid else None
    return CapacitySweepPlan.of(
        seq_lens=list(req.seq_lens), batch_grid=grid, **_estimate_sweep_env(req))


def _estimate_cells(plan: CapacitySweepPlan) -> list:
    return plan.cells()


def _estimate_cell_step_count(cell: Any) -> int:
    """The cell's steps: the contexts it prices."""
    return len(cell.walk["seq_lens"])


def _estimate_cell_steps(cell: Any, prices: Optional[FrozenSet[int]]
                         ) -> Iterator[Tuple[int, Any]]:
    """The (context × batch) sweep's cell, context by context — one walk,
    every point of a context read off it by substitution
    (`capacity.iter_sweep_cell`), each point projected to what the sweep
    serves, with its rows before measurement as plain data: the per-op
    drill-down stays in the process that computed it.

    This cell is the whole sweep (one walk serves every point), so its
    contexts are the only movement a waiting client can be told about and
    the only work a cancelled sweep can leave behind."""
    for index, seq in iter_sweep_cell(cell, prices):
        yield index, None if seq is None else [
            {"sol": served_sweep_point(pt["sol"]), "measures": pt["measures"]}
            for pt in seq]


def _estimate_cell_of_steps(steps: List[Any]) -> Any:
    """The cell's result: its contexts, in `seq_lens` order."""
    return steps


def _layout_cell_step_count(cell: Any) -> int:
    return 1


def _layout_cell_steps(cell: Any, prices: Optional[FrozenSet[int]]
                       ) -> Iterator[Tuple[int, Any]]:
    """The parallel sweep's cell: one step, the layout walked
    (`parallel_sweep.run_layout_cell`). The sweep's movement is its cells,
    counted by the server that schedules them."""
    if prices is not None and 0 not in prices:
        yield 0, None
        return
    yield 0, run_layout_cell(cell)


def _layout_cell_of_steps(steps: List[Any]) -> Any:
    return steps[0]


def _estimate_assemble(plan: CapacitySweepPlan, results: list) -> dict:
    """Every point's measured track from the sweep's ONE lookup
    (`batch_sweep.measure_points`), then the points by batch key."""
    per_seq = [seq for cell in results for seq in cell]
    measure_points([pt for seq in per_seq for pt in seq], **plan.measurement())
    points = [[pt["sol"] for pt in seq] for seq in per_seq]
    shaped = plan.shape(points)
    return {"results_by_batch" if plan.gridded else "results": shaped,
            "decode_cloud": plan.decode_cloud(points)}


@dataclass(frozen=True)
class _CachedSweep:
    """One kind of Scaling-panel sweep, as the steps that compute it: the
    plan from the request, its independent cells, each cell run step by
    step, the cell's result from its steps, and the answer assembled from
    the cells' results in order. Every one is a top-level function, so a
    compute worker can run it."""

    kind: str
    plan: Callable[[Any], Any]
    cells: Callable[[Any], list]
    #: How many steps one cell has, before it runs.
    step_count: Callable[[Any], int]
    #: One cell, step by step: `(index in the cell, the step's result)`,
    #: pricing only the steps in `prices` (None: every one; a step not
    #: priced yields None). The step is what a sweep reports and keeps as
    #: it goes, so it is the smallest piece of the sweep's work that stands
    #: on its own.
    steps: Callable[[Any, Optional[FrozenSet[int]]],
                    Iterator[Tuple[int, Any]]]
    #: The cell's result from its steps, in index order.
    cell_of_steps: Callable[[List[Any]], Any]
    assemble: Callable[[Any, list], dict]
    #: The answer's point cloud, if it has one: kept in the cache for the
    #: views read off it on the server (`parallel_frontier_response`), not
    #: shipped by the endpoint.
    cloud: Optional[str] = None
    #: Whether a cell's steps are this sweep's progress. A plan whose whole
    #: sweep is ONE cell (`CapacitySweepPlan`: one walk serves every point)
    #: has no movement the server can see — the cells it schedules go 0 of 1
    #: to 1 of 1 — so that cell reports its own steps from the worker
    #: (`offload.report_progress` travels there). A plan of many one-step
    #: cells lets the server count the cells instead
    #: (`map_off_loop(progress=True)`), one writer either way.
    steps_are_progress: bool = False
    #: Whether the answer of some of the cells is an answer: a plan of many
    #: independent cells (`ParallelSweepPlan`) assembles the cells it has —
    #: the points not there yet are simply absent — so a page can draw a
    #: sweep as its cells land (`_publish_partials`). A one-cell plan has
    #: nothing in between.
    assembles_partially: bool = False


_PARALLEL = _CachedSweep("parallel", _parallel_plan, _parallel_cells,
                         _layout_cell_step_count, _layout_cell_steps,
                         _layout_cell_of_steps, _parallel_assemble,
                         cloud="decode_cloud", assembles_partially=True)
_ESTIMATE = _CachedSweep("sweep", _estimate_plan, _estimate_cells,
                         _estimate_cell_step_count, _estimate_cell_steps,
                         _estimate_cell_of_steps, _estimate_assemble,
                         cloud="decode_cloud", steps_are_progress=True)


@dataclass(frozen=True)
class _SweepStart:
    """A sweep's first step: its answer's cache key (None: not cached — it
    reads measurements, or the model has no spec), the cached answer if there
    is one, else the plan to run and the cells canonical laws already
    answer."""

    key: Optional[str]
    hit: Any
    plan: Any
    scope_digest: str
    #: Whether the answer reads measurements (`reads_measurements`): it folds.
    reads: bool = False
    theory_identity: Optional[ResultIdentity] = None
    #: Cell results already projected from canonical walk records.
    reused: Mapping[int, Any] = field(default_factory=dict)

    @property
    def keeps_steps(self) -> Optional[str]:
        """The key this sweep's cells keep their steps under (`_Partials`),
        or None when they keep none: a sweep with no spec fingerprint, or
        one whose mode looks no measurement up — such a run asks no point
        for its rows' keys, so its steps answer no measured request. One
        gate, so a step is never read back by a request that could not use
        it."""
        if self.theory_identity is None or self.plan is None:
            return None
        return (self.theory_identity.key
                if looks_up(self.plan.measurement()["selection_mode"]) else None)


def _answer(sweep: _CachedSweep, out: dict, render: bool) -> Any:
    if not render:
        return out
    return render_json({k: v for k, v in out.items() if k != sweep.cloud})


@dataclass(frozen=True)
class _Partials:
    """Where one cell's steps are kept as each finishes, so a sweep no
    request waits on any more — cancelled after `offload.UNWATCHED_S` —
    leaves what it got through and the next request for it resumes rather
    than starts over. The shape the Gallery already has per model
    (`gallery_frontier._finish_frontier` keeps a model's walk before the
    card is priced); a sweep had only the one store at the end.

    Keyed on the sweep's point identity (`sweep_theory_identity`), the cell and
    the step: a step kept under that key was computed for this very sweep,
    off the same walk and the same anchors, so resuming reads back the
    numbers a run through would have produced. `key` is None when the sweep
    is not cached at all (no spec fingerprint) or when its mode asks no point
    for its rows' keys: then nothing is kept and nothing is read.

    Local only (`_SWEEP_STEP`): an unfinished step is this box's business.
    """

    key: Optional[str]
    cell: int

    def _step_key(self, index: int) -> str:
        return of_sweep_step(self.key, self.cell, index).key

    def kept(self, total: int) -> Tuple[List[Any], Optional[FrozenSet[int]]]:
        """What is already kept of the cell's `total` steps, and which steps
        are left to price (None: every one — nothing is kept, or nothing is
        keepable)."""
        if self.key is None:
            return [None] * total, None
        have = [_SWEEP_STEP.get(self._step_key(i)) for i in range(total)]
        left = frozenset(i for i, step in enumerate(have) if step is None)
        return have, left

    def keep(self, index: int, value: Any) -> None:
        if self.key is not None and value is not None:
            _SWEEP_STEP.put(self._step_key(index), value)


@dataclass(frozen=True)
class _Cell:
    """One cell of a sweep as a compute job runs it: which sweep, the cell
    itself, and where its steps are kept. Plain data and top-level
    functions, so it pickles to a worker."""

    sweep: _CachedSweep
    cell: Any
    partials: _Partials


def _run_cell(item: _Cell) -> Any:
    """One cell in a compute worker: the steps an earlier run kept are read
    back, the rest priced (`_CachedSweep.steps`) — each kept as it finishes
    and, when this sweep's steps are its progress, reported as it finishes
    (`offload.report_progress`, which travels from a worker to the work the
    request waits on). The cell's result from its steps
    (`cell_of_steps`)."""
    sweep, keep = item.sweep, item.partials
    total = sweep.step_count(item.cell)
    out, left = keep.kept(total)
    done = total - (total if left is None else len(left))
    if sweep.steps_are_progress:
        report_progress(done, total)
    for index, value in sweep.steps(item.cell, left):
        if value is None:  # a step this run was not asked to price
            continue
        out[index] = value
        keep.keep(index, value)
        done += 1
        if sweep.steps_are_progress:
            report_progress(done, total)
    return sweep.cell_of_steps(out)


def _sweep_start(sweep: _CachedSweep, req: Any, render: bool) -> _SweepStart:
    key = sweep_cache_key(sweep.kind, req)
    hit = None if key is None else _SWEEP.get(key, remote_ok=True)
    digest = of_request(sweep.kind, req, None).scope_digest
    if hit is not None:
        return _SweepStart(key, _answer(sweep, hit, render), None, digest)
    theory_identity = sweep_theory_identity(sweep.kind, req)
    plan = sweep.plan(req)
    reused = _sharedWalkResults(sweep, plan)
    return _SweepStart(
        key, None, plan, digest, req.reads_measurements(),
        theory_identity, reused)


def _sweep_finish(sweep: _CachedSweep, start: _SweepStart, results: list,
                  render: bool) -> Any:
    """The sweep's answer from its cells' results: make the lookup and the
    fold (`assemble`; none for the SOL answer), then store the SOL answer.
    Cell point clouds never enter the canonical law pool: a measured request
    computes its measures live, and only the request cache may hold an
    already-rendered SOL answer."""
    # The fold writes the measured track into the points on a copy. The SOL
    # answer folds nothing and reads the points as they are.
    out = sweep.assemble(
        start.plan, private_copy(results) if start.reads else results)
    out["code_version"] = engine_version()
    # What the measured track was read under, reported: the scope's digest
    # (the constant scope's for the SOL answer).
    out["measurement_scope"] = start.scope_digest
    if start.key is not None and not poolOnly:
        _SWEEP.put(start.key, out, remote_ok=True)
    return _answer(sweep, out, render)


#: How often a sweep that assembles partially republishes what it has.
_PARTIAL_EVERY_S = 2.0


def _partial_answer(sweep: _CachedSweep, start: _SweepStart,
                    have: Dict[int, Any]) -> dict:
    """The answer of the cells finished so far (`have`, by cell index, the
    reused ones among them): the plan narrowed to those cells, assembled as
    the whole one is (`sweep.assemble`), never stored."""
    cells = sweep.cells(start.plan)
    done = sorted(have)
    part = dataclasses.replace(start.plan, cells=[cells[i] for i in done])
    return sweep.assemble(part, [have[i] for i in done])


async def _publish_partials(sweep: _CachedSweep, start: _SweepStart,
                            have: Dict[int, Any], moved: asyncio.Event) -> None:
    """While the cells run, republish the answer of those finished
    (`offload.report_partial`) when one has landed since, at most every
    `_PARTIAL_EVERY_S`; assembled on a thread, off the loop. Runs until
    cancelled."""
    while True:
        await moved.wait()
        moved.clear()
        snapshot = dict(have)
        report_partial(head=await asyncio.to_thread(
            _partial_answer, sweep, start, snapshot))
        await asyncio.sleep(_PARTIAL_EVERY_S)


async def _run_sweep(sweep: _CachedSweep, req: Any, render: bool) -> Any:
    """`_sweep_computed` as one recorded computation
    (`code_trace.provider.recording`): its start, its cells (their worker
    jobs' traces come back to it, `measure.pool.submit`) and its finish, so
    the answer and every step it keeps are stored with the code that made
    them."""
    with recording():
        return await _sweep_computed(sweep, req, render)


async def _sweep_computed(sweep: _CachedSweep, req: Any, render: bool) -> Any:
    """The one place a Scaling-panel sweep is keyed, computed, and stored —
    so the warm job fills the SAME key the endpoint reads and a warmed value
    is byte-identical to a live one. `key is None` (no spec fingerprint)
    computes uncached, matching the endpoint's contract.

    Three steps off the event loop: start (key, cache read, plan — the key
    reads the resolved spec), the cells (in the server each its own compute
    job, so one sweep spreads across the workers; canonical laws project
    their cells), finish (the sweep's one lookup and fold in `assemble`,
    store the answer). Start and finish stay on a thread: after the law split
    they move small dictionaries, and sending each through process IPC would
    cost more than the work itself.

    Each cell keeps its steps as they finish (`_Partials`), so a sweep
    cancelled because no request waits on it any more
    (`offload.UNWATCHED_S`) leaves them for the next request to resume
    from. The progress a request that misses the deadline reads in its 503
    has one writer: the cells as they finish, or — for a sweep that is one
    cell — that cell's own steps, reported from the worker
    (`_CachedSweep.steps_are_progress`).

    `render`: the answer comes back as the endpoint's JSON response, encoded
    where it was computed. A spec refined while the cells ran stores and
    serves nothing (`ensure_spec_generation`): the cells may have read two
    versions of it."""
    generation = spec_generation()
    start = await asyncio.to_thread(_sweep_start, sweep, req, render)
    if start.hit is not None:
        return start.hit
    cells = sweep.cells(start.plan)
    items = [_Cell(sweep, cell, _Partials(start.keeps_steps, i))
             for i, cell in enumerate(cells) if i not in start.reused]
    if not sweep.steps_are_progress:
        report_progress(0, len(cells))
    # A sweep a page can draw cell by cell publishes what it has as it goes;
    # a measured one does not — its points read the lake once, at the end.
    publisher = None
    on_result = None
    if sweep.assembles_partially and not start.reads and items:
        have: Dict[int, Any] = dict(start.reused)
        moved = asyncio.Event()
        computedAt = [i for i in range(len(cells)) if i not in start.reused]

        def on_result(position: int, result: Any) -> None:
            have[computedAt[position]] = result
            moved.set()

        if have:
            moved.set()
        publisher = asyncio.ensure_future(
            _publish_partials(sweep, start, have, moved))
    try:
        computed = (await map_off_loop(_run_cell, items,
                                       progress=not sweep.steps_are_progress,
                                       on_result=on_result)
                    if items else [])
    finally:
        if publisher is not None:
            publisher.cancel()
    results = []
    for index, _cell in enumerate(cells):
        if index in start.reused:
            results.append(start.reused[index])
        else:
            results.append(computed.pop(0))
    ensure_spec_generation(generation)
    return await asyncio.to_thread(_sweep_finish, sweep, start, results, render)


async def _scoped(req: Any) -> Any:
    """`req` with a measurement scope: the route set it
    (`_access.with_measurement_scope`, the caller's); the warm job sends
    none and gets the shared rows' (owner None). A request that reads no
    measurement — the SOL answer — reads under `NONE` and with no rows,
    whatever it carried: its fold then reads nothing whatever its mode."""
    if not req.reads_measurements():
        return req.model_copy(update={"measurement_scope": NO_MEASUREMENTS,
                                      "calibration_rows": None})
    if req.measurement_scope is not None:
        return req
    scope = await asyncio.to_thread(_measure_lookup.resolve_scope, reads=True, owner=None)
    return req.model_copy(update={"measurement_scope": scope})


async def _sweep_within_deadline(sweep: _CachedSweep, req: Any, render: bool) -> Any:
    req = await _scoped(req)
    return await within_deadline(
        f"{sweep.kind}:{render}:" + call_key(_run_sweep, (req,), {}),
        lambda: _run_sweep(sweep, req, render))


async def compute_parallel_sweep_cached(req: InferenceParallelSweepRequest) -> dict:
    """The parallel sweep's payload (the warm job's entry)."""
    return await _sweep_within_deadline(_PARALLEL, req, False)


async def parallel_sweep_response(req: InferenceParallelSweepRequest) -> Response:
    """`/inference/estimate-parallel-sweep`'s response."""
    return await _sweep_within_deadline(_PARALLEL, req, True)


async def parallel_frontier_response(req: ParallelFrontierRequest) -> dict:
    """`/inference/parallel-sweep/frontier`'s response: a view of the
    sweep's decode point cloud (`decode_frontier`), read off the sweep's own
    cache entry — the view keys nothing and computes no point."""
    sweep = await _sweep_within_deadline(_PARALLEL, req.sweep, False)
    return decode_frontier(sweep[_PARALLEL.cloud], fix=req.fix, group_by=req.group_by)


#: The Compare page's views of its rows' cloud, each a grouping at every
#: context: the front across the rows at each batch point, each row's own
#: line over its batch points, and the envelope over every row and batch.
_COMPARE_VIEWS = {"by_batch": ("context", "batch_key"), "by_row": ("context", "row"),
                  "envelope": ("context",)}


async def compare_frontier_response(req: CompareFrontierRequest) -> dict:
    """`/inference/estimate-sweep/frontier`'s response: the compared rows'
    decode points (each row's sweep answer's cloud, read as the page reads
    it — `CapacitySweepPlan.decode_cloud`, tagged with its `row`) and, per
    track, each of `_COMPARE_VIEWS`: its lines, each line its Pareto front."""
    answers = await asyncio.gather(*(_sweep_within_deadline(_ESTIMATE, s, False)
                                     for s in req.sweeps))
    cloud = [{**pt, "row": i} for i, answer in enumerate(answers)
             for pt in answer[_ESTIMATE.cloud]]
    # A row is the request's own provenance dimension; the rest are each
    # point's coordinates.
    dims = get_args(SweepCloudDim)
    return {"points": cloud, "fronts": {
        track: {view: frontier_view(cloud, dims=dims, fix={}, group_by=by,
                                    track=track)["series"]
                for view, by in _COMPARE_VIEWS.items()}
        for track in ("sol", "measured")}}


async def compute_estimate_sweep_cached(req: InferenceEstimateSweepRequest) -> dict:
    """The seq/batch sweep's payload (the warm job's entry): the Context ×
    Batch and KV-storage POSTs the Scaling panel sends, each cell projected
    where it ran (`served_sweep_point`)."""
    return await _sweep_within_deadline(_ESTIMATE, req, False)


async def estimate_sweep_response(req: InferenceEstimateSweepRequest) -> Response:
    """`/inference/estimate-sweep`'s response."""
    return await _sweep_within_deadline(_ESTIMATE, req, True)
