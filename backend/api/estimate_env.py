"""Seq-invariant kwargs for `estimate_capacity` / `estimate_capacity_sweep`.

The HTTP routes and the gallery render both need the same preparation:
validate the model id, fetch config and tensor metadata, build the dict
the estimator consumes. That is not a property of a FastAPI route, so
it does not live next to `app`.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from fastapi import HTTPException

from backend.analyzer import explain_config_fetch_failure
from backend.arch.loader import get_default_loader
from backend.api._models import InferenceEstimateRequest
from backend.hf.metadata import fetch_config, fetch_metadata
from backend.inference.parallel_config import ParallelConfig
from backend.inference.phases import Speculation


def split_load_factor(fields: Optional[Mapping[str, Any]]
                      ) -> Tuple[Dict[str, Any], float]:
    """A request's `parallel` block as the deployment it states and the
    batch's routing skew beside it. The page sends `ep_load_factor` (max/mean
    expert load) with the deployment's fields; it is a workload input
    (`Workload.ep_load_factor`), so it is taken out here and the rest is the
    `ParallelConfig`. A skew below 1 is refused: the hottest expert carries
    at least the mean."""
    rest = dict(fields or {})
    rho = rest.pop("ep_load_factor", None)
    rho = 1.0 if rho is None else float(rho)
    if not rho >= 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"ep_load_factor must be a number ≥ 1.0 (max/mean expert "
                   f"load), got {rho!r}")
    return rest, rho


def resolve_estimate_environment(
    req: InferenceEstimateRequest,
) -> Dict[str, Any]:
    """Validate `model_id`, fetch config + tensor metadata, and pre-build
    the kwargs dict that `estimate_capacity` / `estimate_capacity_sweep`
    both consume (everything seq-invariant — excludes `avg_input_tokens`
    and `seq_lens`).

    Raises `HTTPException` on missing model_id or unfetchable config.
    """
    if not req.model_id:
        raise HTTPException(status_code=400, detail="model_id required")
    # Config and checkpoint come from the record the spec is served from.
    overlay_dir = get_default_loader().overlay_for(req.model_id)
    config = fetch_config(req.model_id, overlay_dir)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=explain_config_fetch_failure(req.model_id),
        )

    # Fetch tensor-level metadata once and reuse for:
    #   1. published_weight_bytes (informational, UI-only)
    #   2. shape validation (`tensor_patterns_meta` → spec_vs_tensors check)
    # Both HF-hub and custom models (`custom:name` registered with
    # --hf-model-path) flow through fetch_metadata and receive the same
    # `header` + totals structure.
    meta = fetch_metadata(req.model_id, overlay_dir)
    published_weight_bytes = req.weight_bytes_total
    parallel_fields, ep_load_factor = split_load_factor(req.parallel)
    if published_weight_bytes is None:
        published_weight_bytes = meta.get("checkpoint_storage_bytes")

    return dict(
        config=config,
        gpu_sku=req.gpu_sku,
        gpu_memory_fraction=req.gpu_memory_fraction,
        num_gpus=req.num_gpus,
        avg_output_tokens=req.avg_output_tokens,
        parallel=ParallelConfig.from_dict(parallel_fields),
        ep_load_factor=ep_load_factor,
        dcp_selected_load_factor=req.dcp_selected_load_factor,
        dcp_failed_owners=req.dcp_failed_owners,
        # The request's precision profile, canonical (`PrecisionProfile`):
        # every weight format, edge dtype and cache format is its answer.
        precision=req.precision_profile(),
        published_weight_bytes=(

            float(published_weight_bytes) if published_weight_bytes else None
        ),
        tensor_patterns_meta=meta or None,
        s_prefix_cached=req.prefix_cached_tokens,
        s_verify=Speculation(req.speculative_draft_tokens).s_verify,
        acceptance_rate=req.acceptance_rate,
        speculative_draft=req.speculative_draft,
        include_aux_weights=req.include_aux_weights,
        host_offload=req.host_offload,
        comm_latency_us=req.comm_latency_us,
        launch_floor_us=req.launch_floor_us,
        model_id=req.model_id,
        selection_mode=req.resolved_selection_mode(),
        framework=req.framework,
        kernel_mix=req.kernel_mix,
        framework_options=req.framework_options,
        kv_block_size=req.kv_block_size,
        prefill_batch_size=req.prefill_batch_size,
        decode_batch_size=req.decode_batch_size,
        prefill_chunk_size=req.prefill_chunk_size,
        calibration_rows=req.calibration_rows,
        measurement_scope=req.scope(),
    )
