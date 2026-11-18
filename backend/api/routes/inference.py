"""Inference-tab endpoints — GPU picker data, deployment-dtype defaults,
and KV-cache capacity estimation.

  • GET  /inference/gpus               — full GPU spec table.
  • GET  /inference/deployment-dtype   — per-model default dtype/granularity.
  • POST /inference/estimate           — KV cache budget + max tokens.
  • GET  /inference/compute-graph      — final post-passes IR (comm + fusion).
  • POST /inference/bench/plan-jobs    — one step as Kernel Bench jobs.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

from backend.arch.eval.precision import ResolvedPrecision
from backend.arch.eval.dim import value_of
from backend.arch.eval.program import Program
from backend.arch.media_processor import ProcessorRefusal, processed_media
from backend.arch.snapshots import load_processor
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.stages import StagePlacement
from backend.inference.gpu_specs import load_gpu_specs
from backend.inference.module_rows import served
from backend.inference.capacity import (
    estimate_pending,
    finish_estimates,
)
from backend.inference.analysable import layersplit_refusal
from backend.inference.speculative_draft import DraftStyle, SpeculativeDraft
from backend.inference.parallel_config import ParallelConfig
from backend.inference.disagg import disagg_handoff
from backend.inference.sweep_grid import (
    AUDIT_PARALLEL, ROUTINE_OUTPUT_TOKENS, ROUTINE_TRACKS, SCALING_CONTEXTS,
    SCALING_DEGREES, SPEC_PRESETS,
)
from backend.inference.defaults import (
    COMM_LATENCY_US_DEFAULT,
    GPU_MEMORY_FRACTION_DEFAULT,
    LAUNCH_FLOOR_US_DEFAULT,
)
from backend.rendering._compute_graph import (
    serialize_post_tree,
    slice_pipeline_fragment,
)
from backend.rendering.kv_marks import KVMarks
from backend.views import declarations
from backend.views.graph_view import stamp_l1


from backend.api.estimate_env import resolve_estimate_environment as _resolve_estimate_environment
from backend.api.estimate_env import split_load_factor
from backend.analyzer import explain_config_fetch_failure
from backend.hf.metadata import fetch_config
from backend.arch.loader import get_default_loader


from backend.inference.framework_bindings import (
    DEFAULT_FRAMEWORK, ServingDeployment, get_binding)
from backend.inference.phases import Speculation
from backend.inference.pipeline import analyze
from backend.inference.step_eval import evaluate_encode
from backend.arch.layout.layout import IndivisibleSequence
from backend.arch.layout.attention_plan import (
    IndivisibleHeadRepartition,
    legal_cp_degrees,
    query_heads_per_tp_shard,
)

from .._access import with_measurement_scope
from backend.measure.scope import of_request
from .._app import app
from ..offload import offload, offload_response
from backend.api.sweep_runtime import _SWEEP_CACHE_DIR
from backend.fingerprint import result_version
from backend.inference.default_deployment import default_deployment, fit_identity
from backend.measure.cache import DiskBackend, KeyedCache, MemoryBackend, TieredBackend
from backend.measure.scope import make_key

from backend.fingerprint import engine_version as _code_version
from .._models import (
    BenchPlanJobsRequest,
    CompareFrontierRequest,
    InferenceEstimateRequest,
    InferenceEstimateSweepRequest,
    InferenceParallelSweepRequest,
    ParallelFrontierRequest,
    ParallelShapeRequest,
    ProcessedMediaRequest,
    SPECULATIVE_WINDOW_RETIRED,
    SweepShapesRequest,
    precision_from_query,
)

from backend.api.sweep_runtime import (
    _SWEEP,  # noqa: F401  # tests patch the route-module name
    compare_frontier_response,
    estimate_sweep_response,
    parallel_frontier_response,
    parallel_sweep_response,
    sweep_cache_key as _sweep_cache_key,  # noqa: F401
)


@app.get("/inference/version")
async def inference_version():
    """Estimation-code version stamp (see `_cache.engine_version`). Stamped into
    every estimate/sweep response so the frontend can key its sweep cache on it:
    a redeploy that changes any estimation logic flips the hash → cached curves
    auto-invalidate. The hash reads the engine's source (seconds, once per
    process): a compute worker answers, which has it from startup."""
    return {"code_version": await offload(_code_version)}


# Sweep cache (keying + store + compute) lives in sweep_runtime so the
# warm job shares it without importing this route module / `app`.


def _head_repartition_refusal(e: IndivisibleHeadRepartition) -> JSONResponse:
    return JSONResponse(status_code=400, content={
        "detail": str(e), "reason": e.reason,
        "heads_per_tp_shard": e.heads_per_tp_shard,
        "degree": e.degree,
        "legal_cp_degrees": list(e.legal_cp_degrees),
    })


@app.post("/inference/parallel-shape")
async def inference_parallel_shape(req: ParallelShapeRequest):
    """How many GPUs a deployment occupies, per pipeline stage and in all,
    and what makes it invalid — `ParallelConfig`'s own answers, so the page
    that edits a deployment does not restate the mesh arithmetic. With a
    `model_id`, the mesh is the one that model runs
    (`ParallelConfig.for_program`): a dense model has no expert region."""
    fields, _rho = split_load_factor(req.parallel)
    try:
        ParallelConfig.from_dict(fields)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not req.model_id:
        return _parallel_shape(fields, None)
    return await offload(_parallel_shape, fields, req.model_id)


#: Each model's opening deployment on a card (`default_deployment`), by the
#: model's result version and everything else the fit reads
#: (`fit_identity`): seconds of preparation walks for a large MoE, asked
#: whenever the page picks a model or a card.
_DEFAULT_DEPLOYMENTS = KeyedCache(
    "default-deployment",
    TieredBackend(MemoryBackend(max_entries=256),
                  DiskBackend(_SWEEP_CACHE_DIR / "default-deployment",
                              max_files=4000)),
    single_flight=True,
)


@app.post("/inference/default-deployment")
async def inference_default_deployment(req: InferenceEstimateRequest):
    """The deployment the Inference page opens a model at on a card, before
    any estimate (`backend.inference.default_deployment`): one GPU for a
    dense model, DP = EP from the card's fabric unit up to the first degree
    whose weights fit for a MoE. The request is an estimate's; its
    `parallel` is what this answers, so it is not read."""
    return await offload(_default_deployment, req)


def _default_deployment(req: InferenceEstimateRequest) -> Dict[str, Any]:
    env = _resolve_estimate_environment(req)
    resolved = get_default_loader().match(env["config"], model_id=req.model_id)
    if resolved is None or resolved.spec is None:
        raise HTTPException(status_code=404,
                            detail=f"No spec matched config for {req.model_id!r}.")
    key = make_key("default-deployment", result_version(resolved),
                   fit_identity(env))
    try:
        return _DEFAULT_DEPLOYMENTS.get_or_compute(
            key, lambda: default_deployment(env, resolved))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _parallel_shape(parallel_fields: Dict[str, Any],
                    model_id: Optional[str]) -> Dict[str, Any]:
    parallel = ParallelConfig.from_dict(parallel_fields)
    if model_id:
        parallel = parallel.for_program(
            Program.of(_resolved_or_404(model_id)))
    return {
        "cluster_size": parallel.cluster_size,
        "per_stage_size": parallel.per_stage_size,
        "errors": parallel.validate(),
    }


@app.get("/inference/gpus")
async def inference_gpus():
    """Full GPU spec table used to populate the Inference tab picker,
    plus the canonical reference deployment the pickers default to."""
    return {
        "items": [g.to_json() for g in load_gpu_specs()],
        "audit_parallel": AUDIT_PARALLEL,
    }


@app.get("/inference/defaults")
async def inference_defaults():
    """Product defaults the inference pickers seed from — one declaration
    in `backend/inference/defaults.py` (+ `sweep_grid.AUDIT_PARALLEL`, the
    speculative presets and the Scaling panel's context / GPU-count grid), so no frontend copy can
    drift."""
    return {
        "audit_parallel": AUDIT_PARALLEL,
        "scaling_contexts": SCALING_CONTEXTS,
        "scaling_degrees": SCALING_DEGREES,
        "gpu_memory_fraction": GPU_MEMORY_FRACTION_DEFAULT,
        "comm_latency_us": COMM_LATENCY_US_DEFAULT,
        "launch_floor_us": LAUNCH_FLOOR_US_DEFAULT,
        "spec_decode_presets": SPEC_PRESETS,
    }


def _decode_deployment_env(
    req: InferenceEstimateRequest, base_env: Dict[str, Any]
) -> Dict[str, Any]:
    """Env for the SECOND (decode) deployment in a disaggregated request.

    Starts from the top-level (prefill/shared) env and overrides the
    deployment-shape knobs the `decode` block supplies — `gpu_sku`,
    `num_gpus`, `parallel`. Everything else (model, dtypes, workload) is
    shared: the same model produces the same KV the decode side holds.

    `parallel` describes the decode mesh WHOLE. It used to merge field by
    field into the prefill mesh, so `{"parallel": {"cp": 2}}` against a
    prefill `tp_attention=2, tp_dense_ffn=2` silently produced a four-rank
    decode mesh — priced as four GPUs, reported as the two the request
    asked for. A decode block is a second deployment, and the axes it does
    not name are one, not the other deployment's.
    """
    d = req.decode or {}
    env = dict(base_env)
    if d.get("gpu_sku"):
        env["gpu_sku"] = str(d["gpu_sku"])
    # The GPU count belongs to a mesh: the prefill's stated count says
    # nothing about the decode mesh, so only the decode block's own carries.
    env["num_gpus"] = (int(d["num_gpus"]) if d.get("num_gpus") is not None
                       else None)
    if d.get("parallel") is not None:
        fields, rho = split_load_factor(d["parallel"])
        env["parallel"] = ParallelConfig.from_dict(fields)
        if "ep_load_factor" in d["parallel"]:
            env["ep_load_factor"] = rho
    return env


# Fields owned by the DECODE deployment in a disaggregated result: the decode
# latency phase + everything KV-capacity (the decode deployment holds the KV
# cache). Prefill-owned fields (prefill phase, `max_prefill_batch_*`, the
# prefill weight breakdown) stay from the base (prefill) result.
#: The decode cluster's own row lists. Every field named after one — the
#: list and what is derived from it (`<list>_measured_by_declaration`, …) —
#: is the decode cluster's too (`_decode_owned`).
_DISAGG_DECODE_ROWS = ("per_op_timing_decode", "per_op_timing_draft")
_DISAGG_DECODE_FIELDS = (
    "max_tokens_per_gpu", "max_tokens_total",
    "max_batch_per_gpu", "max_batch_total",
    "max_decode_batch_per_gpu", "max_decode_batch_total",
    "kv_cache", "kv_replication_factor", "memory",
    "unique_tokens_multiplier", "estimated_max_fraction",
)


def _refuse_count_beside_mesh(stated: Optional[int],
                              result: Dict[str, Any]) -> None:
    """A deployment occupies the ranks its mesh spans (the result's
    `cluster_size`, after the model's own regions are applied). A GPU count
    stated beside it that disagrees describes a deployment nobody asked for:
    refused, never reported next to the mesh."""
    cluster = result.get("cluster_size")
    if stated is not None and cluster is not None and int(stated) != cluster:
        raise ValueError(
            f"deployment asks for {stated} GPUs but its parallelism spans "
            f"{cluster} ranks; state a parallel config that fits the "
            "cluster, or omit the GPU count")


def _dep_summary(r: Dict[str, Any]) -> Dict[str, Any]:
    """Per-deployment header shown by the disaggregated UI. Its GPU count
    is the mesh's rank count (`_refuse_count_beside_mesh` refuses a stated
    count that disagrees), so the two fields are one fact. It carries one
    cluster's whole memory budget (mesh, weights, memory, the formulas behind
    them): the stitched top level mixes the prefill cluster's mesh and weights
    with the decode cluster's memory, so a budget is drawn from here."""
    return {
        "num_gpus": r.get("cluster_size"),
        "cluster_size": r.get("cluster_size"),
        "gpu": r.get("gpu"),
        "parallel": r.get("parallel"),
        "parallel_warnings": r.get("parallel_warnings"),
        "weight_breakdown": r.get("weight_breakdown"),
        "memory": r.get("memory"),
        "kv_cache": r.get("kv_cache"),
        "gpu_memory_fraction": r.get("gpu_memory_fraction"),
        "formulas": r.get("formulas"),
        "infeasibility_reason": r.get("infeasibility_reason"),
    }


def _stitch_disagg(
    prefill_result: Dict[str, Any], decode_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Combine two single-deployment `estimate_capacity` results into one
    disaggregated response: prefill/TTFT + prefill throughput from the prefill
    run, decode/TPOT + KV capacity from the decode run. The response keeps the
    single-deployment SHAPE (so existing renderers work) with values stitched,
    plus `disagg: true` and per-deployment summaries for the two clusters."""
    out = dict(prefill_result)  # prefill is the base
    out["phases"] = {
        "prefill": prefill_result.get("phases", {}).get("prefill"),
        "decode": decode_result.get("phases", {}).get("decode"),
    }
    # The decode cluster's fields are its own, present or absent: a decode
    # cluster that holds no step publishes no decode rows (`without_step`),
    # and keeping the prefill cluster's in their place would show the other
    # cluster's step under the decode tab.
    owned = {k for k in (*prefill_result, *decode_result)
             if any(k == rows or k.startswith(rows + "_")
                    for rows in _DISAGG_DECODE_ROWS)} | set(_DISAGG_DECODE_FIELDS)
    for k in owned:
        if k in decode_result:
            out[k] = decode_result[k]
        else:
            out.pop(k, None)
    # The fusion delta is per phase, each cluster's own.
    delta = {phase: side["vs_naive_delta"][phase]
             for phase, side in (("prefill", prefill_result),
                                 ("decode", decode_result))
             if phase in (side.get("vs_naive_delta") or {})}
    out.pop("vs_naive_delta", None)
    if delta:
        out["vs_naive_delta"] = delta
    # Either cluster may admit no request; the top level says which. Each
    # reason is a whole sentence (the page shows it as one), so the two are
    # two sentences, not one joined by ";" after a full stop.
    reasons = [f"{name} cluster: {r['infeasibility_reason']}"
               for name, r in (("Prefill", prefill_result),
                               ("Decode", decode_result))
               if r.get("infeasibility_reason")]
    out["infeasibility_reason"] = " ".join(reasons) or None
    out["disagg"] = True
    out["prefill_deployment"] = _dep_summary(prefill_result)
    out["decode_deployment"] = _dep_summary(decode_result)
    # Warnings surface from BOTH deployments.
    out["parallel_warnings"] = (
        list(prefill_result.get("parallel_warnings") or [])
        + list(decode_result.get("parallel_warnings") or [])
    )
    # So do assumptions: each deployment states what it assumed (the decode
    # cluster may spread experts where the prefill one does not). An entry
    # both deployments make identically is said once.
    assumptions = list(prefill_result.get("assumptions") or [])
    for a in decode_result.get("assumptions") or []:
        if a not in assumptions:
            assumptions.append(a)
    out["assumptions"] = assumptions
    return out




@app.post("/inference/estimate")
async def inference_estimate(req: InferenceEstimateRequest, request: Request,
                             authorization: Optional[str] = Header(None)):
    """Compute KV cache budget and max tokens for a (model, GPU) pair.

    `model_id` is required. If `weight_bytes_total` is omitted, fetches
    it via hf_metadata. The measurement scope is resolved first
    (`with_measurement_scope`: the caller as its owner with
    `calibrate_with_mine`, the lake's snapshot the lookup is pinned at).
    """
    req = await with_measurement_scope(req, request, authorization)
    return await offload_response(_estimate_payload, req)


def _say_measurements(result: Dict[str, Any], req: InferenceEstimateRequest) -> None:
    """What the estimate's measured track was read from: `measurements` is
    `none` (no service configured, or a mode that reads none — the estimate
    is SOL and says so) or `kernel_bench`; `measurement_scope` the scope's
    digest (`ResultIdentity.scope_digest`)."""
    identity = of_request("estimate", req, None)
    result["measurements"] = identity.measurements
    result["measurement_scope"] = identity.scope_digest


def _estimate_payload(req: InferenceEstimateRequest) -> Any:
    """The `/inference/estimate` answer: one deployment's estimate, or the
    disaggregated pair stitched. Runs in a compute worker — the env
    resolve fetches config and metadata, and `estimate_capacity` is
    seconds of pure-Python analysis."""
    try:
        env = _resolve_estimate_environment(req)
        # Disaggregated: the speculative draft runs where decode runs, so it is
        # priced (or refused) on the decode deployment only.
        prefill_env = (env if req.decode is None
                       else {**env, "speculative_draft": None})
        pending = [estimate_pending(
            avg_input_tokens=req.avg_input_tokens,
            # The encoders run where prefill runs, before it reads them.
            media=req.media,
            encode_batch_size=req.encode_batch_size,
            **prefill_env,
        )]
        # Disaggregated serving: a second (decode) deployment, estimated
        # with the decode env and stitched — prefill metrics from the first
        # run, decode + KV capacity from the second. Both deployments' keys
        # go to one measurement lookup (`finish_estimates`). The
        # single-deployment path (`req.decode is None`) is untouched.
        if req.decode is not None:
            decode_env = _decode_deployment_env(req, env)
            pending.append(estimate_pending(
                avg_input_tokens=req.avg_input_tokens, **decode_env))
        result, *rest = finish_estimates(pending)
        _refuse_count_beside_mesh(req.num_gpus, result)
        if req.decode is not None:
            (decode_result,) = rest
            _refuse_count_beside_mesh(decode_env.get("num_gpus"),
                                      decode_result)
            handoff = disagg_handoff(
                result, decode_result,
                kv_transfer_bw_gbps=req.kv_transfer_bw_gbps,
                avg_output_tokens=req.avg_output_tokens,
            )
            result = _stitch_disagg(result, decode_result)
            result["disagg_handoff"] = handoff
    except IndivisibleHeadRepartition as e:
        return _head_repartition_refusal(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    result["timings"] = pending[0].timings if pending[0] is not None else {}
    if len(pending) > 1:
        result["timings"]["decode"] = (
            pending[1].timings if pending[1] is not None else {})
    if pending[0] is not None:
        logger.info("estimate timings %s", json.dumps(
            pending[0].timings, sort_keys=True, default=str))
        if len(pending) > 1 and pending[1] is not None:
            logger.info("decode estimate timings %s", json.dumps(
                pending[1].timings, sort_keys=True, default=str))
    result = served(result)
    result["code_version"] = _code_version()
    _say_measurements(result, req)
    # The per-op table's rows per declaration: a parent for each op of a
    # declaration split into derived groups, its children indices into the
    # rows beside it. The rows themselves stay per group.
    for rows_key in ("per_op_timing", "per_op_timing_decode"):
        # Each parent's measured time is the measured fold's share of its own
        # terms, stated where the fold is (`batch_sweep.apply_measured`); the
        # view groups the rows and reads it.
        parents = declarations.per_op_rows(
            result.get(rows_key) or [],
            result.pop(f"{rows_key}_measured_by_declaration", None))
        if parents:
            result[f"{rows_key}_by_declaration"] = parents
    for stack in (result.get("encode") or {}).get("stacks") or ():
        stack["per_op_timing_by_declaration"] = declarations.per_op_rows(
            stack["per_op_timing"],
            stack.pop("per_op_timing_measured_by_declaration", None))
    return result


@app.post("/inference/estimate-sweep")
async def inference_estimate_sweep(req: InferenceEstimateSweepRequest, request: Request,
                                   authorization: Optional[str] = Header(None)):
    """Run `/inference/estimate` at multiple `seq_lens` in one call.

    The seq-invariant preparation (spec match, materialize quant defaults,
    KV per-token, weight breakdown, GPU peaks, calibration registry +
    framework binding) is computed ONCE; `_capacity_finish` then runs
    per seq inside a single threadpool submission. Replaces the
    9-fanout that the inference + compare pages used to do — same
    result, ~one preparation's worth of work instead of N.

    Response: `{"results": [<chart view>, ...]}` ordered by `seq_lens`.
    Per-seq failures are NOT swallowed (matches the single-seq
    endpoint contract). Each result is the estimate with per-op rows /
    weight tree / formulas stripped — those belong on `/inference/estimate`,
    not on a 9-point landscape payload.
    """
    if not req.model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    req = await with_measurement_scope(req, request, authorization)
    try:
        return await estimate_sweep_response(req)
    except IndivisibleHeadRepartition as e:
        return _head_repartition_refusal(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/inference/bench/sweep-shapes")
async def inference_bench_sweep_shapes(req: SweepShapesRequest):
    """Export the Scaling panel's FULL sweep as measurable per-op shapes for
    (model, gpu): the routine context×batch grid × parallel modes/degrees,
    VRAM-gated on that GPU, expanded to the deduped job list. The single source
    the panel's export button AND kernel_bench both consume — so the measured
    corpus covers exactly what the panel plots. No auth: pure analyze over public
    config. Returns `{"jobs": [...], "stats": {...}, "filename": <jsonl name>}`.
    """
    from kernel_bench.sweep import (
        build_sweep_jobs, resolve_sweep_model, routine_parallel_configs,
    )

    def _routine_jobs():
        # The Scaling panel's surface: the routine tracks over the model's
        # routine deployments, at the routine output length.
        config, resolved = resolve_sweep_model(req.model_id)
        return build_sweep_jobs(
            model_id=req.model_id, gpu=req.gpu, seq_lens=req.seq_lens,
            tracks=ROUTINE_TRACKS,
            ep_load_factors=req.ep_load_factors,
            parallel_configs=routine_parallel_configs(config, resolved=resolved),
            avg_output_tokens=ROUTINE_OUTPUT_TOKENS, resolved=resolved,
            framework=req.framework, precision=req.precision_profile(),
        )

    try:
        jobs, stats = await offload(_routine_jobs)

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    slug = req.model_id.replace("/", "_")
    return {"jobs": jobs, "stats": stats,
            "filename": f"{slug}_{req.gpu}_sweep_jobs.jsonl"}


@app.post("/inference/bench/plan-jobs")
async def inference_bench_plan_jobs(req: BenchPlanJobsRequest):
    """Export ONE step as Kernel Bench jobs — the `jobs.jsonl` `hbench plan`
    writes, for the step the Inference page is showing.

    Same builder, second entry point: `kernel_bench.plan.build_jobs_over_points`
    is called with the request's own fields and the file text is produced by
    the harness's own writer (`plan.jobs_jsonl`), so the bytes a browser
    downloads here and the bytes the CLI writes for the same inputs are the
    same bytes. No job field is restated in this layer.

    Returns `{"jsonl", "job_count", "dropped", "filename"}`. `dropped` names
    the ops of this step that carry no kernel signature (or no per-op shape
    law); they are part of the answer, not a silence — a measured total is
    never shown while an op of the step is unmeasured.
    """
    try:
        return await offload_response(_plan_jobs, req)
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _plan_jobs(req: BenchPlanJobsRequest) -> Dict[str, Any]:
    """The `/inference/bench/plan-jobs` answer; runs in a compute worker.

    Reads the request, calls the builder once, hands back its two answers.
    """
    from kernel_bench.plan import (PlanPoint, build_jobs_over_points,
                                   jobs_jsonl)

    parallel_fields, ep_load_factor = split_load_factor(req.parallel)
    point = PlanPoint(
        seq_len=req.avg_input_tokens,
        s_prefix_cached=req.prefix_cached_tokens,
        # K drafted tokens are verified in one forward over K + 1 positions
        # — the estimate's own reading of the knob (`estimate_env`).
        s_verify=Speculation(req.speculative_draft_tokens).s_verify,
        batch=req.batch_size,
        ep_load_factor=ep_load_factor,
    )
    by_point = build_jobs_over_points(
        model_id=req.model_id,
        points=[point],
        phase=req.phase,
        avg_output_tokens=req.avg_output_tokens,
        parallel=ParallelConfig.from_dict(parallel_fields),
        framework=req.framework,
        precision=req.precision_profile(),
        media=req.media or None,
        gpu=req.gpu_sku,
    )
    jobs, dropped = by_point[point]
    slug = req.model_id.replace("/", "_")
    return {
        "jsonl": jobs_jsonl(jobs),
        "job_count": len(jobs),
        "dropped": dropped,
        "filename": f"{slug}_{req.phase}_jobs.jsonl",
    }


@app.post("/inference/estimate-parallel-sweep")
async def inference_estimate_parallel_sweep(req: InferenceParallelSweepRequest,
                                            request: Request,
                                            authorization: Optional[str] = Header(None)):
    """Deployment-scale sweep for the Scaling panel.

    Holds model + batch fixed and sweeps the parallelism DEGREE (EP for
    dp_ep/cp_ep/tp_ep, TP for tp_tp, each from one GPU), one series per GPU
    type, at every context of `contexts`, reporting PER-GPU prefill
    throughput. `modes` batches several axes into one process-pool pass (env
    resolved once). Each deployment is one cell (parallel is part of the
    seq-invariant preparation); inside it one walk per card class prices every
    card and context. Decode is the sweep's decode point cloud, kept on the
    server and read through `/inference/parallel-sweep/frontier`.

    One mode: `{"mode", "moe_enabled", "series": [...], "by_mode": {...}}`.
    Several modes: `{"moe_enabled", "modes", "by_mode": {mode: {series…}}}`.
    A series' points name their `context`. EP modes on a dense model return
    an empty series.
    """
    if not req.model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    req = await with_measurement_scope(req, request, authorization)
    try:
        return await parallel_sweep_response(req)
    except IndivisibleHeadRepartition as e:
        return _head_repartition_refusal(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/inference/estimate-sweep/frontier")
async def inference_estimate_sweep_frontier(req: CompareFrontierRequest,
                                            request: Request,
                                            authorization: Optional[str] = Header(None)):
    """The Compare page's decode Pareto fronts across its rows: each row's
    estimate sweep read as the page posted it (its cache entry for a SOL
    answer; measured answers are read live), every point the rows' sweeps
    serve, and per track the front at each (context, batch key)."""
    if any(not s.model_id for s in req.sweeps):
        raise HTTPException(status_code=400, detail="model_id required")
    scoped = await asyncio.gather(*(with_measurement_scope(s, request, authorization)
                                    for s in req.sweeps))
    try:
        return await compare_frontier_response(req.model_copy(update={"sweeps": scoped}))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/inference/parallel-sweep/frontier")
async def inference_parallel_sweep_frontier(req: ParallelFrontierRequest,
                                            request: Request,
                                            authorization: Optional[str] = Header(None)):
    """A view of the Scaling panel's parallel sweep's throughput point
    cloud: one `phase` of it (prefill or decode), some dimensions held to a
    set of values (`fix`: cards, modes, GPU counts, contexts, batches) and
    the rest grouped into lines (`group_by`), each line its Pareto front
    (`form="frontier"`) or its points along `x` — GPU count, per-GPU decode
    batch or context (`form="line"`).

    The view reads the cache entry of the sweep BODY the caller posts and
    computes no point of its own: a `sol_only` body is the SOL answer (what
    the panel draws by default, and all a decode rung ever has), and the
    panel's measured body is the measured answer, whose prefill carries the
    measured track its fold wrote. Computed on a miss, as the sweep
    endpoint would."""
    if not req.sweep.model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    req = req.model_copy(update={"sweep": await with_measurement_scope(
        req.sweep, request, authorization)})
    try:
        return await parallel_frontier_response(req)
    except IndivisibleHeadRepartition as e:
        return _head_repartition_refusal(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/inference/legal-cp")
async def inference_legal_cp(
    model_id: str,
    tp_attention: int = 1,
    cp_scheme: str = "pcp",
):
    """Prefill CP degrees the current (model, TP, scheme) may take.

    Layersplit stripes the cache by layer, which needs one cache growth
    law: a model that mixes cache types gets an empty list and the refusal
    as `unavailable_reason` (`analysable.layersplit_refusal`, the predicate
    validate's replay deploys by).

    Ulysses interior requires the sequence degree to divide the query
    heads on one TP shard; the list is that constraint solved across
    every head-repartition attention node. Schemes that gather context
    rather than repartition heads return `legal_cp_degrees: null`
    (unconstrained by this cut).
    """
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    try:
        return await offload_response(_legal_cp, model_id, tp_attention, cp_scheme)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _resolved_or_404(model_id: str) -> Any:
    config = fetch_config(model_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=explain_config_fetch_failure(model_id),
        )
    resolved = get_default_loader().match(config, model_id=model_id)
    if resolved is None:
        raise HTTPException(
            status_code=404,
            detail=f"No spec matched config for {model_id!r}.",
        )
    return resolved


def _legal_cp(model_id: str, tp_attention: int, cp_scheme: str) -> dict:
    resolved = _resolved_or_404(model_id)
    parallel = ParallelConfig.from_dict({
        "tp_attention": max(1, int(tp_attention)),
        "cp_scheme": cp_scheme,
    })
    analysis = Deployment(Program.of(resolved)).analyze()
    legal = legal_cp_degrees(analysis, parallel, scheme=cp_scheme)
    unavailable = (layersplit_refusal(analysis)
                   if cp_scheme == "layersplit" else None)
    if unavailable is not None:
        # One stripe per layer needs one cache growth law; a model that
        # mixes cache types has no legal layersplit degree.
        legal = ()
    return {
        "cp_scheme": cp_scheme,
        "tp_attention": max(1, int(tp_attention)),
        "heads_per_tp_shard": query_heads_per_tp_shard(analysis, parallel),
        "legal_cp_degrees": None if legal is None else list(legal),
        "unavailable_reason": unavailable,
    }


def _refuse_retired_window(value: Optional[str]) -> None:
    """A query still naming the old verify window is refused (422) as the
    estimate body refuses it, not ignored into autoregressive decode."""
    if value is not None:
        raise HTTPException(status_code=422, detail=SPECULATIVE_WINDOW_RETIRED)


@app.get("/inference/speculative-draft")
async def inference_speculative_draft(
    model_id: str,
    speculative_draft_tokens: Optional[int] = Query(None, ge=0),
    speculative_window: Optional[str] = Query(None, include_in_schema=False),
):
    """Whether the decode step can count the speculative draft's cost here.

    The draft graph is the model's MTP stack (`aux_kind: mtp`); a model
    without one has nothing to price. `unavailable_reason` is the same refusal
    `/inference/estimate` raises for `speculative_draft` on this deployment
    (`SpeculativeDraft.refusal`), so the page shows the reason the estimate
    would give.
    """
    _refuse_retired_window(speculative_window)
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    try:
        return await offload_response(_speculative_draft, model_id,
                             speculative_draft_tokens)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _speculative_draft(model_id: str,
                       speculative_draft_tokens: Optional[int]) -> dict:
    program = Program.of(_resolved_or_404(model_id))
    reason = SpeculativeDraft.refusal(
        program, draft_tokens=speculative_draft_tokens)
    return {
        "available": reason is None,
        "unavailable_reason": reason,
        "draft_stacks": list(SpeculativeDraft.stacks_of(program)),
        "styles": [s.value for s in DraftStyle],
    }


@app.get("/inference/media-inputs")
async def inference_media_inputs(model_id: str):
    """The media a request to this model may carry, as the spec declares it
    (`model_inputs:`), and the encoder stack that reads each: what the
    estimate's `media` takes. Per input: its kind, its axes in order (the
    items per request first), each axis's fixed value when the config fixes
    it, and the least extent one item's axis takes — a legal side is a whole
    number of it (an image's resize factor, `patchify`'s `align`). An input
    the engine does not walk carries its reason (`unmodeled`): a request
    states its items as 0. `encoders` is empty for a model with none."""
    return await offload(_media_inputs, model_id)


def _media_inputs(model_id: str) -> Dict[str, Any]:
    resolved = _resolved_or_404(model_id)
    program = Program.of(resolved)
    inputs = []
    for name, entry in program.model_inputs.items():
        if entry.kind == "tokens":
            continue
        axes = []
        for position, axis in enumerate(entry.axes):
            fixed = entry.fixed.get(axis)
            least = entry.least.get(axis)
            axes.append({
                "name": axis,
                "items": position == 0,
                "fixed": (None if fixed is None
                          else int(value_of(fixed))),
                "least": None if least is None else int(value_of(least)),
            })
        inputs.append({"name": name, "kind": entry.kind,
                       "stack": entry.stack, "axes": axes,
                       "per_item_of": entry.per_item_of,
                       "unmodeled": entry.unmodeled})
    encoders = [{"stack": stack.name, "aux_kind": stack.aux_kind,
                 "inputs": [i["name"] for i in inputs
                            if i["stack"] == stack.name]}
                for stack in program.stacks if stack.is_encoder]
    return {"model": model_id, "inputs": inputs, "encoders": encoders}


@app.post("/inference/processed-media")
async def inference_processed_media(req: ProcessedMediaRequest):
    """A raw image's processed size, as the model's processor resizes it
    (`arch.media_processor`, read off the processor configuration the
    model's HF repo ships, kept in its snapshot): `raw` gives, per image
    input, its items and the raw `height` / `width` in pixels; the answer is
    the estimate's `media` for them, and per input the processor rule and
    parameters that resized it. A model whose repo ships no processor
    configuration, or one naming a processor with no rule, is refused with
    the reason (state the processed size then)."""
    return await offload(_processed_media, req.model_id, req.raw)


def _processed_media(model_id: str, raw: Dict[str, Dict[str, int]]
                     ) -> Dict[str, Any]:
    resolved = _resolved_or_404(model_id)
    try:
        media, rules = processed_media(
            Program.of(resolved), load_processor(model_id), raw,
            repo=model_id)
    except ProcessorRefusal as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"model": model_id, "media": media, "rules": rules}


@app.get("/inference/compute-graph")
async def inference_compute_graph(
    model_id: str,
    framework: str = DEFAULT_FRAMEWORK,
    pp: int = 1,
    tp_attention: int = 1,
    tp_dense_ffn: int = 1,
    ep_moe: int = 1,
    tp_moe_expert: int = 1,
    cp: int = 1,
    dp_attention: int = 1,
    dp_replicas: int = 1,
    sequence_parallel: bool = False,
    cp_scheme: str = "pcp",
    phase: str = "prefill",
    pp_rank: Optional[int] = None,
    avg_input_tokens: int = 1,
    avg_output_tokens: int = 0,
    prefix_cached_tokens: int = 0,
    speculative_draft_tokens: int = Query(0, ge=0),
    batch_size: int = 1,
    gpu_sku: Optional[str] = None,
    framework_flags: str = "",
    precision: Optional[str] = None,
    stack: Optional[str] = None,
    media: str = "",
    speculative_window: Optional[str] = Query(None, include_in_schema=False),
):
    """Return the FINAL compute graph after every inference-layer pass
    (parallelism comm injection + kernel fusion) is applied.

    This is the graph that actually runs at deploy time: comm
    primitives (all_reduce / reduce_scatter / all_gather / all_to_all)
    grafted in where the parallelism policy requires them, and matched
    subgraphs collapsed into fused-op leaves. Flipping the TP / EP
    knobs changes the graph — comm appears, and norm fusions switch
    between fused_add_rmsnorm (TP=1) and allreduce_fused_rmsnorm (TP>1).
    `sequence_parallel=true` selects Megatron-SP: the 2 all_reduce per
    block become 2 all_gather + 2 reduce_scatter (norm/residual region
    runs Sharded(S)).

    `phase` (prefill / decode) picks the topology the serving stack
    actually runs at that phase. The only structural difference today
    is MLA: prefill runs the decompressed form (kv_b_proj as an
    explicit linear); decode swaps to the absorbed form (kv_b_proj is
    folded into q_absorbed_nope + v_absorbed per-head matmuls and the
    QKᵀ contracts the latent R+P dim). Non-MLA models render the same
    topology in either phase.

    The workload is the estimate's own: `avg_input_tokens`,
    `avg_output_tokens`, `prefix_cached_tokens` and `speculative_draft_tokens`
    go through the phase binding `/inference/estimate` uses (prefill runs the
    uncached input; decode verifies `speculative_draft_tokens` + 1 tokens
    against the average cache), at `batch_size` requests. The shapes and the KV amounts
    on every node are that step's. The defaults are one token at batch one.

    `phase=encode` draws encoder `stack` alone, as the estimate's encode
    phase runs it: `batch_size` requests each carrying `media` (JSON, the
    estimate's `media`), no language-model step bound.

    A length the sequence world cannot split (a CP prefill of 1 token at
    CP=2) is a 400 whose body carries `reason: "sequence_not_divisible"`
    beside `detail`, with the `extent`, `world` and `degree` involved.

    `precision` is the estimate's precision profile as JSON text
    (`{"preset": "kv_fp8"}`); absent is as shipped.
    """
    _refuse_retired_window(speculative_window)
    try:
        profile = precision_from_query(precision)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    if phase not in ("prefill", "decode", "encode"):
        raise HTTPException(
            status_code=400,
            detail=f"phase must be 'prefill', 'decode' or 'encode', got "
                   f"{phase!r}",
        )
    if (phase == "encode") != (stack is not None):
        raise HTTPException(
            status_code=400,
            detail="an encode graph names its encoder `stack` (and only an "
                   "encode graph names one)")
    try:
        media_given = json.loads(media) if media else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400,
                            detail=f"media is not JSON: {e}") from e
    if phase == "encode" and pp_rank is not None:
        raise HTTPException(
            status_code=400,
            detail="an encoder runs whole on one pipeline stage; pp_rank "
                   "names a language-model stage")

    return await offload_response(
        _compute_graph,
        model_id=model_id,
        framework=framework,
        pp=pp,
        tp_attention=tp_attention,
        tp_dense_ffn=tp_dense_ffn,
        ep_moe=ep_moe,
        tp_moe_expert=tp_moe_expert,
        cp=cp,
        dp_attention=dp_attention,
        dp_replicas=dp_replicas,
        sequence_parallel=sequence_parallel,
        cp_scheme=cp_scheme,
        phase=phase,
        pp_rank=pp_rank,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        prefix_cached_tokens=prefix_cached_tokens,
        speculative_draft_tokens=speculative_draft_tokens,
        batch_size=batch_size,
        gpu_sku=gpu_sku,
        framework_flags=framework_flags,
        precision=profile,
        stack=stack,
        media=media_given,
    )


def _compute_graph(
    *,
    model_id: str,
    framework: str,
    pp: int,
    tp_attention: int,
    tp_dense_ffn: int,
    ep_moe: int,
    tp_moe_expert: int,
    cp: int,
    dp_attention: int,
    dp_replicas: int,
    sequence_parallel: bool,
    cp_scheme: str,
    phase: str,
    pp_rank: Optional[int],
    avg_input_tokens: int,
    avg_output_tokens: int,
    prefix_cached_tokens: int,
    speculative_draft_tokens: int,
    batch_size: int,
    gpu_sku: Optional[str],
    framework_flags: str,
    precision: Any,
    stack: Optional[str] = None,
    media: Optional[Dict[str, Any]] = None,
) -> Any:
    """The `/inference/compute-graph` answer; runs in a compute worker."""
    resolved = _resolved_or_404(model_id)
    binding = get_binding(framework)
    if binding is None:
        raise HTTPException(
            status_code=404,
            detail=f"No framework binding for {framework!r}.",
        )
    # The kernels the framework runs only under conditions (`when:`) are
    # answered for this deployment: the GPU's device, the model's
    # activation dtype under the profile, and the server flags turned on.
    binding = binding.deployed(ServingDeployment.at(
        gpu_sku=gpu_sku,
        activation_dtype=ResolvedPrecision(precision, Program.of(resolved)).model_dtype,
        server_flags={f.strip(): True
                      for f in framework_flags.split(",") if f.strip()}))

    parallel = ParallelConfig.from_dict({
        "pp": pp,
        "tp_attention": tp_attention,
        "tp_dense_ffn": tp_dense_ffn,
        "ep_moe": ep_moe,
        "tp_moe_expert": tp_moe_expert,
        "cp": cp,
        "dp_attention": dp_attention,
        "dp_replicas": dp_replicas,
        "sequence_parallel": sequence_parallel,
        "cp_scheme": cp_scheme,
    }).for_program(Program.of(resolved))
    parallel_errs = parallel.validate()
    if parallel_errs:
        raise HTTPException(status_code=400, detail="; ".join(parallel_errs))

    # Decode-side topology swap: real serving stacks (vLLM / SGLang /
    # TRT-LLM) bind the absorbed MLA at decode time (kv_b_proj folded
    # into Q/O-side per-head matmuls, KV cache held in latent R+P).
    # Without the swap the decode graph would draw the prefill chain.
    # The step's axes come from the estimate's phase binding, so the
    # graph and the numbers beside it describe one step.
    s_verify = Speculation(speculative_draft_tokens).s_verify
    try:
        if phase == "encode":
            result = analyze(
                resolved,
                parallel=parallel,
                framework_binding=binding,
                batch_size=max(1, int(batch_size)),
                phase=phase,
                step_eval=evaluate_encode(
                    resolved, parallel, stack=stack,
                    batch=max(1, int(batch_size)), media=media or {},
                    precision=precision),
                timed_stacks=frozenset({stack}),
                precision=precision,
                breakdown=True,
            )
        else:
            result = analyze(
                resolved,
                parallel=parallel,
                framework_binding=binding,
                batch_size=max(1, int(batch_size)),
                avg_input_tokens=avg_input_tokens,
                avg_output_tokens=avg_output_tokens,
                s_prefix_cached=prefix_cached_tokens,
                phase=phase,
                s_verify=s_verify,
                precision=precision,
                breakdown=True,
            )
    except IndivisibleSequence as e:
        return JSONResponse(status_code=400, content={
            "detail": str(e), "reason": e.reason,
            "extent": e.extent, "world": e.world, "degree": e.degree,
        })
    except IndivisibleHeadRepartition as e:
        return _head_repartition_refusal(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    graph = serialize_post_tree(
        result.post_tree, resolved,
        # Resolve absorbed fused subgraphs against comm_tree (the tree
        # fusion operated on): it carries the injected all_reduce / RS /
        # … nodes that a TP>1 fusion absorbs, which pre_tree lacks.
        pre_tree=result.comm_tree,
        fusion_matches=result.fusion_matches,
        program=result.program,
        shapes=result.shapes,
        full_shapes=result.full_shapes,
        pre_program=result.pre_program,
        pre_shapes=result.pre_shapes,
        pre_full_shapes=result.pre_full_shapes,
        kv=KVMarks.of_serving(result),
    )
    stage = None
    if pp_rank is not None:
        if pp <= 1 or not (0 <= pp_rank < pp):
            raise HTTPException(
                status_code=400,
                detail=f"pp_rank requires pp > 1 and 0 <= pp_rank < pp "
                       f"(pp={pp}, pp_rank={pp_rank})",
            )
        graph = slice_pipeline_fragment(
            graph, StagePlacement.of(result.analysis).view(pp_rank))
        stage = graph.pop("stage")
    # Shown per declaration: the derived groups of one fold into one box,
    # themselves unchanged inside it. Last, so the stage slice saw the
    # groups the engine evaluated.
    graph = declarations.fold_graph(graph)
    stamp_l1(graph.get("nodes") or [])
    return {
        "model": model_id,
        "framework": framework,
        "framework_audited_at": binding.audited_at.tag,
        "parallel": {
            "pp": pp,
            "tp_attention": tp_attention,
            "tp_dense_ffn": tp_dense_ffn,
            "ep_moe": ep_moe,
            "tp_moe_expert": tp_moe_expert,
            "cp": cp,
            "dp_attention": dp_attention,
            "dp_replicas": dp_replicas,
        },
        "phase": phase,
        "stack": stack,
        "stage": stage,
        "graph": graph,
    }
