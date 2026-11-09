"""plan — enumerate the measurement jobs for a (model, config).

CPU-only. Runs the analyze pipeline (same path as
`backend.cli_bench.emit_suggest_shapes`), then for every per-op row that
maps to a measurable op-signature emits a job carrying:
  - identity (model, spec_hash, op, node_spec_id, parallel)
  - dims (shared model shape_json: B,S,N,Nk,D,H,I,…) + dtype_env
  - the per-op roofline numerators (flops_per_rank, access_bytes_per_rank)
    so `run` can compute achieved_* from measured latency.

Both numerators are the WALK's, carried over the launches the step makes of
the node (`PointRow.launches × cadence`). Plan states no cost model of
its own: the row already says what this node costs per rank per step, and it
is the same row the estimate, the module breakdown and the SOL gate read.

Rows the queue cannot ask a GPU for are recorded beside the plan with a closed
reason: a zero-cost layout row is measured@0 (`zero_launch`), an op with no
signature is `missing_signature`, and a signature whose shape/dtype projection
cannot be stated is `missing_projection` (or `declined`, when the leaf's own
key cannot be stated exactly). A drop from the old two-value planner API can
only be migrated as `legacy_drop`; its original text remains the detail. The
old textual drop notes remain for callers that already read them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from backend.inference.framework_bindings import DEFAULT_FRAMEWORK
from backend.inference.benchmark_identity import MatrixShard
from backend.inference.op_signatures import get_signature
from backend.measure.auto_sweep import (DECLINED_REASON, MISSING_PROJECTION,
                                        MISSING_SIGNATURE, ZERO_LAUNCH)

from .providers import served_kv_block_sizes

#: A migration-only reason for callers that still return the pre-structural
#: `(jobs, dropped)` planner form. The text is preserved verbatim as detail;
#: this name says only what is known there — the producer said the row was
#: dropped, not which placement law applied. It never enters `PointMeasure`.
LEGACY_DROP = "legacy_drop"



def _scale_block_refusal(row: Dict[str, Any]) -> Optional[str]:
    """Why a deployment with this per-rank shape cannot load, when a weight
    it multiplies carries a BLOCK scale — or None when it can.

    A block scale ties a SHARD to the block: the frameworks refuse at load
    time when a parallel world cuts a weight into rank slices that are not
    a whole number of blocks — along K for a row-parallel cut, along N for
    a column-parallel one (vLLM `fp8_utils.validate_fp8_block_shape`, and
    `…_moe` for the expert GEMMs, which raise rather than pad). Past some TP
    degree every block-scaled model hits this — GLM-5.3-Flash at TP 64,
    where d_model/64 is under the 128-wide block — and nothing could
    measure such a cell: no kernel reads a block split across two ranks.

    Every fact this reads is the walk's (`weight_shards`): each partition of
    each matrix the kernel multiplies, as the grid its scale tiles,
    unsharded and on one rank, read off the placement — a cut on a heads
    axis, one segment of a packed `gate|up`, one partition of a merged
    launch — with its own scale layout and the weight it is packed into.

    For a per-TILE (block FP8) scale, a width nobody cut is not a shard. Its
    last block is partial in the checkpoint itself, and the frameworks load
    it (DeepSeek-V3's `kv_a_proj_with_mqa`, 576 wide, replicated on every
    rank: vLLM 0.32's column layers skip the block check outright when a
    partition is ragged, `linear._maybe_allow_fp8_block_shape_mismatch`, and
    sglang serves the same projection as one replicated 2112-wide linear).
    The policy here sits between the two frameworks, on the conservative
    side of vLLM: a cut partition must be whole blocks per rank, and of the
    uncut partitions one weight packs at most ONE may end mid-block (the
    order they are packed in is the framework's; sglang wants every merged
    partition whole and avoids DeepSeek-V3's case by not merging it).

    An explicit `block_scale_layout` can instead state that checkpoint TP
    chunks own the grids. The selected loader either merges whole chunks
    (SGLang) or gathers heads (vLLM), then requantizes a fresh rank-local
    fused grid. Logical Q/K/V boundaries need no block alignment on those
    paths. See docs/fp8_tp_interleaved.md for the audited source contracts.

    A per-GROUP scale (GPTQ / AWQ, Marlin) is refused on a ragged reduction
    width whether or not it was cut: Marlin and AWQ raise on a K that is
    not a whole number of groups at any degree (vLLM `marlin_utils`
    `check_marlin_supports_shape`, `auto_awq`).

    Not modelled: an expert bank's intermediate width uncut under expert
    parallelism, which vLLM's `validate_fp8_block_shape_moe` still wants
    whole blocks — no block-FP8 MoE in the corpus has a ragged one.

    The block widths come from the granularity the walk states per tensor
    (`TensorFormat.weight_granularity`), not from a second table of model
    names: `per-tile:<n>x<k>` carries both, `per-group:<k>` the reduction
    one, and `per-channel` / `per-tensor` tie nothing to a shard.
    """
    ragged_uncut: Dict[Tuple[int, int], List[int]] = {}
    for shard in (MatrixShard.of(s) for s in row.get("weight_shards") or ()):
        gran = shard.granularity or ""
        if gran.startswith("per-tile:"):
            block_n, _, block_k = gran[len("per-tile:"):].partition("x")
        elif gran.startswith("per-group:"):
            block_n, block_k = "", gran[len("per-group:"):]
        else:
            continue
        width_n = int(block_n) if block_n.isdigit() else 0
        width_k = int(block_k) if block_k.isdigit() else 0

        def refusal(axis: str, extent: int, width: int) -> str:
            return (f"this rank holds {axis}={extent}, not a whole number of "
                    f"the weight's {width}-wide scale blocks ({gran}) — the "
                    f"frameworks refuse such a shard at load, so no "
                    f"deployment runs this cell")

        grouped = gran.startswith("per-group:")
        if width_k and shard.k_local % width_k and (shard.cuts_k or grouped):
            return refusal("d_reduce", shard.k_local, width_k)
        layout = shard.block_scale_layout
        loader = layout.get("loader") if layout is not None else None
        if layout is not None and layout.get("load_refusal"):
            return str(layout["load_refusal"])
        if loader:
            # SGLang loads whole checkpoint chunks at matching TP. At lower
            # TP it dequantizes those chunks, deinterleaves Q/K/V and
            # requantizes the entire fused rank matrix, padding its final
            # block. vLLM can also gather individual heads from those
            # chunks and requantize at higher TP, replicating KV as needed.
            # Neither loader cuts scale blocks at logical Q/K/V boundaries.
            # The reduction-axis check above still applies.
            continue
        if not width_n or not shard.n_local % width_n:
            continue
        if shard.cuts_n:
            return refusal("d_out", shard.n_local, width_n)
        ragged_uncut.setdefault(shard.pack, []).append(shard.n_local)
    for widths in ragged_uncut.values():
        if len(widths) > 1:
            return (f"this launch packs {len(widths)} partitions "
                    f"({', '.join(map(str, widths))} wide) that are not a "
                    f"whole number of their scale blocks into one weight — "
                    f"only the last may end mid-block, so the frameworks "
                    f"refuse it at load and no deployment runs this cell")
    return None


def _starved_axes(row: Dict[str, Any]) -> List[str]:
    """`expr=value` for every per-rank axis of `row` that is not a whole row.

    The named forms (`shape_in_dims` / `shape_out_dims`, `as_wire`) carry the
    EXACT extent — a rational where the walk averaged one — beside the int
    the display column floors it to. An extent under 1 is an average of a
    ragged distribution, not an extent anything allocates, so it is reported
    by name rather than rounded into a shape.
    """
    return [text for text, extent in _unpadded_axes(row) if extent < 1]


def _rounded_axes(row: Dict[str, Any]) -> List[str]:
    """`expr=value` for every per-rank axis of `row` that is a fraction OVER
    one row.

    `_starved_axes` catches the mean that falls under one row; this catches
    the same mean one step up, where it does not look empty and is no more
    a geometry: a cell's extents are the law's rounded UP
    (`RowIdentity.one_event`), so a row whose law states 3/2 rows is keyed
    at 2 and a measurement of it is a third wider than the law priced.

    A routed rows axis is exempt (`_unpadded_axes`): its averaged
    extent is the mean rows one expert receives, every expert the rows reach
    is padded up alike by the owner's decision, and the job carries both the padded and
    the exact count with numerators to match. Every other averaged extent
    is refused by name — a compressor that pools once every few tokens has
    a CADENCE (`RowIdentity.cadence`: one launch every m calls), and
    widening its buffer would invent a launch per call.
    """
    return [text for text, extent in _unpadded_axes(row)
            if extent > 1 and extent.denominator != 1]


def _unpadded_axes(row: Dict[str, Any]) -> List[Tuple[str, Fraction]]:
    """`(expr=value, extent)` for every per-rank axis of `row` the launch may
    NOT pad, once each: every axis but the rows of a routed buffer — the
    tensors the walk says are one (`row["routed"]`: `inputs`, `output`),
    `[n_expert_copies, M, ...]` as the dispatch lays it out. Its `M` is the
    mean rows an expert receives, which the launch pads
    (`RoutedLaunch.rows_per_expert`); any other averaged extent is refused
    by name, inside a MoE dispatch zone or out of it."""
    routed = row.get("routed") or {}
    inputs = frozenset(routed.get("inputs") or ())
    output = bool(routed.get("output"))
    out: List[Tuple[str, Fraction]] = []
    for key in ("shape_in_dims", "shape_out_dims"):
        wire = row.get(key)
        if isinstance(wire, (list, tuple)) and wire and isinstance(wire[0], dict):
            wire = [wire]
        for index, axes in enumerate(wire if isinstance(wire, (list, tuple)) else ()):
            for position, axis in enumerate(axes if isinstance(axes, (list, tuple)) else ()):
                padded = (output if key == "shape_out_dims"
                          else index in inputs)
                if not isinstance(axis, dict) or (position == 1 and padded):
                    continue
                expr, value = axis.get("expr"), axis.get("value")
                try:
                    extent = Fraction(str(value))
                except (ValueError, ZeroDivisionError, TypeError):
                    continue
                text = f"{expr}={value}"
                if text not in (seen for seen, _extent in out):
                    out.append((text, extent))
    return out


def build_jobs(*, batch_size: int, **kwargs: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (jobs, dropped) for one (model, config, phase) at one batch —
    `build_jobs_over_batches` at a single batch."""
    return build_jobs_over_batches(batch_sizes=[batch_size], **kwargs)[batch_size]


def build_jobs_with_unplaced(**kwargs: Any) -> Tuple[
        List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    """`build_jobs`, with the structured sidecar records beside the old notes.

    The first two elements are exactly `build_jobs`' tuple. The third is a
    list of `{op, aggregation_id, reason, detail?}` records; it includes
    zero-launch rows, which are not gaps and are never sent to a GPU.
    """
    if "batch_size" in kwargs:
        batch_size = int(kwargs.pop("batch_size"))
        return build_jobs_over_batches(
            batch_sizes=[batch_size], with_unplaced=True,
            **kwargs)[batch_size]  # type: ignore[return-value]
    by_point = _build_jobs_over_points(**kwargs)
    jobs = [job for _jobs, _notes, _records in by_point.values() for job in _jobs]
    notes: List[str] = []
    records: List[Dict[str, Any]] = []
    for _jobs, point_notes, point_records in by_point.values():
        for note in point_notes:
            if note not in notes:
                notes.append(note)
        for record in point_records:
            if record not in records:
                records.append(record)
    return jobs, notes, records


@dataclass(frozen=True)
class PlanPoint:
    """One point of a phase cell the plan keys kernels at: the prompt, the
    prefix of it already cached, the verify width and the batch — every one
    a workload count the cell's walk reads (`counts`), so every point of a
    cell is that one walk read there. An encode step's point is its batch
    alone (the media are the cell's)."""

    seq_len: int
    s_prefix_cached: int
    s_verify: int
    batch: int
    #: The batch's routing skew, ρ (`Workload.ep_load_factor`). A point-time
    #: workload coordinate, not a deployment field and not a walk key: it changes
    #: the busiest routed rank's launch geometry.
    ep_load_factor: Any = 1

    def counts(self, phase: str) -> Dict[str, int]:
        """The workload counts this point sets, by the names the walk's laws
        read (`transforms.Workload.symbols`)."""
        if phase == "encode":
            return {"B": int(self.batch)}
        return {"B": int(self.batch), "s_input": int(self.seq_len),
                "s_prefix": int(self.s_prefix_cached),
                "s_verify": int(self.s_verify),
                "rho": self.ep_load_factor}


def _prefill_chunk_provenance(endpoint_index: int, chunk: Any) -> Dict[str, Any]:
    endpoint = chunk.endpoint
    return {
        "endpoint_index": endpoint_index,
        "endpoint": {
            "seq_len": endpoint.seq_len,
            "s_prefix_cached": endpoint.s_prefix_cached,
            "s_verify": endpoint.s_verify,
            "batch": endpoint.batch,
        },
        "index": chunk.index,
        "start": chunk.start,
        "end": chunk.end,
        "width": chunk.width,
        "launch_batch": chunk.launch_batch,
    }


def attach_prefill_chunk_provenance(
        job: Dict[str, Any], endpoint_index: int,
        chunks: Sequence[Any]) -> Dict[str, Any]:
    """Carry endpoint/chunk ownership beside a physical launch job.

    The caller adds this only on a decomposed endpoint.  Every job from a
    launch point carries every endpoint provenance that reached that point
    (262144's first chunks are also 1048576's).  The list is fold provenance
    only: it is not in ``physical_key``, and ``Invocation.from_job`` reads
    none of it.
    """
    job.setdefault("prefill_chunks", []).extend(
        _prefill_chunk_provenance(endpoint_index, chunk) for chunk in chunks
    )
    return job


def build_jobs_over_batches(
    *,
    model_id: str,
    batch_sizes: Sequence[int],
    seq_len: int,
    phase: str,
    avg_output_tokens: int,
    s_prefix_cached: int = 0,
    s_verify: int = 1,
    with_unplaced: bool = False,
    **kwargs: Any,
) -> Dict[int, Tuple[List[Dict[str, Any]], List[str]]]:
    """`{batch: (jobs, dropped)}` at one context and each of `batch_sizes`:
    `build_jobs_over_points` at those points."""
    by_point = build_jobs_over_points(
        model_id=model_id, phase=phase, avg_output_tokens=avg_output_tokens,
        points=[PlanPoint(seq_len, s_prefix_cached, s_verify, b)
        for b in batch_sizes], **kwargs, with_unplaced=with_unplaced)
    return {point.batch: out for point, out in by_point.items()}


def _build_jobs_over_points(
    *,
    model_id: str,
    points: Sequence[PlanPoint],
    phase: str,
    avg_output_tokens: int,
    tp: int = 1,
    parallel: Optional[Any] = None,
    framework: str = DEFAULT_FRAMEWORK,
    resolved: Optional[Any] = None,
    binding: Optional[Any] = None,
    precision: Optional[Any] = None,
    media: Optional[Dict[str, Dict[str, int]]] = None,
    kv_block_sizes: Optional[Sequence[int]] = None,
    gpu: Optional[str] = None,
) -> Dict[PlanPoint, Tuple[List[Dict[str, Any]], List[str],
                           List[Dict[str, Any]]]]:
    """`{point: (jobs, notes, unplaced)}` for one (model, config, phase) at
    each of `points` (`PlanPoint`). `notes` keeps the old textual drops;
    `unplaced` carries closed `{op, aggregation_id, reason, detail?}` records,
    including zero-launch rows, for the plan's sidecar.

    The engine walks ONCE — one phase cell, at the canonical anchor's workload
    and the largest batch any point asks there — and every other point (another
    batch, prompt, cached prefix or verify width) is that walk's rows
    re-read there (`batch_sweep.rows_at`) — the same
    per-row laws the Scaling panel re-prices with, shapes included. A point
    the deployment refuses for a cached-prefix geometry (`prefix_refusal`)
    comes back as that point's explicit dropped note rather than discarding
    the other points; a point the walk does not admit is refused as a walk
    there would refuse it (`Point.admit`); a measurable row without a shape
    law refuses the point (`NoShapeLaw`).

    `phase="encode"` plans the model's encoders, each in its encode phase:
    a point's batch of requests each carrying `media` (`Workload.media`,
    sizes the caller states — the plan binds none of its own), every
    encoder whose inputs the media names. A point's prompt, prefix and
    verify width say nothing there.

    `avg_output_tokens` is the served request's output length, and it has no
    default: decode runs over the cache at the prompt plus half the output
    (`phases.phase_binding`), so a decode plan built without it keys a cache
    no served point reads. A point's `s_prefix_cached` is the cached prefix
    of a prefill: 0 is the plain prefill (no prefix cache), not a stand-in.

    `parallel` (a ParallelConfig) is the deployment sharding — it decides the
    per-RANK op shapes analyze() emits (TP shards heads/FFN, EP shards experts,
    CP shards the sequence). Pass it to sweep the parallelism axis; the legacy
    `tp` int is a convenience shorthand for pure tensor-parallel when `parallel`
    is not given.

    `parallel.pp` splits the layers across ranks rather than the tensors, so
    there every stage is walked and the jobs are folded back into one per
    launch (`_step_results`, `_one_job_per_launch`): a job's
    `count_per_step` is then the layers one rank holds.

    Each `PlanPoint.ep_load_factor` is the batch's routing skew (ρ = max/mean
    expert load, `Workload.ep_load_factor`) — a workload input, not a
    deployment field, and the one the estimate also takes beside the mesh. It
    decides how many rows the busiest rank's expert cell runs, so a plan built
    at ρ=1 for a step costed at ρ>1 would key that cell's kernel at a geometry
    the step never ran."""
    from backend.analyzer import fetch_config
    from backend.arch.loader import SpecLoader
    from backend.inference.benchmark_identity import (
        declares_routed_token_total,
    )
    from backend.inference.calibration_overlay import (
        WorkloadShape,
        build_deployment_context,
        point_keys,
    )
    from backend.arch.precision_profile import AS_SHIPPED, PrecisionProfile
    from backend.arch.eval.precision import ResolvedPrecision
    from backend.arch.eval.program import Program
    from backend.inference.framework_bindings import (ServingDeployment,
                                                      get_binding)
    from backend.inference.gpu_specs import get_gpu
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.parallel_sweep import config_for_degree
    from backend.inference.phases import phase_binding
    from backend.inference.pipeline import analyze
    from backend.inference.walk_registry import shared_walk_cells

    # `resolved` / `binding` are (model, framework)-invariant — the same across
    # every cell of a sweep. Resolving the spec is ~900ms (SpecLoader.match, no
    # cross-instance cache), so a caller sweeping many shapes passes them in ONCE
    # (build_sweep_jobs) instead of paying it per cell. Fall back to resolving
    # here for the single-shape `plan` path.
    if resolved is None:
        config = fetch_config(model_id)
        if config is None:
            raise RuntimeError(f"could not fetch config for {model_id!r}.")
        resolved = SpecLoader().match(config, model_id=model_id)
    if resolved is None:
        raise RuntimeError(f"no spec matches {model_id!r}; run the spec generator first.")
    if binding is None:
        binding = get_binding(framework)
    if binding is None:
        raise RuntimeError(
            f"no framework binding for {framework!r}."
        )
    card = get_gpu(gpu) if gpu else None
    described = binding.deployment
    if (described.device is not None
            or described.compute_capability is not None) and not gpu:
        raise ValueError(
            "a deployed binding is card-aware, so its plan names that card "
            "(`gpu=`); an undeployed, card-free plan is shape inspection "
            "only and every job it emits says `plan_scope=cardless-shapes`")
    if (card is not None and described.device is not None
            and (described.device != card.device
                 or described.compute_capability != card.compute_capability)):
        raise ValueError(
            f"the deployed binding describes {described.device or 'its card'} "
            f"at capability {described.compute_capability or '?'}, not "
            f"{card.sku}; pass the card that binding was deployed for")
    if gpu and described.device is None:
        # The same deployment the sweep plans with: card facts answer the
        # binding's dispatches, including the mixed-precision launch table.
        binding = binding.deployed(ServingDeployment.at(
            gpu_sku=gpu,
            activation_dtype=ResolvedPrecision(
                precision if precision is not None else AS_SHIPPED,
                Program.of(resolved)).model_dtype))
    # The card the plan is for. A box plan's dtype environment is not card-free:
    # a quantised matmul whose operands need an MMA family this card has no
    # column for runs weight-only there, which is a different ACTIVATION and
    # so a different cell (`pipeline.leaf_peak`, `phases.compute_per_op_timing`).
    # Without it the plan emitted the key of a launch the box cannot perform
    # and the estimate on that box looked up the weight-only one, so the two
    # sides of the corpus asked for different rows. It is the same card the
    # deployed binding already carries, which is why a sweep has it in hand.
    # Only launch selection and its dtype consequences are read from it: the
    # plan still asks for no times, so a card keeps every other number out of
    # the key (`PhysicalKey` stamps the card a row was measured on). With no
    # card there is no box plan — only an explicitly marked shape-inspection
    # artifact whose mixed-precision keys must not be queued.
    if parallel is None:
        # The `tp` shorthand is pure tensor parallelism at that degree, as
        # the scaling modes already declare it (`config_for_degree("tp_tp")`:
        # the three TP axes together, ep=1) — not a `tp` key, which
        # `ParallelConfig.from_dict` does not read, so `--tp 4` used to walk
        # an all-ones mesh and stamp `tp: 4` on whole-model shapes.
        parallel = (config_for_degree("tp_tp", tp) if tp > 1
                    else ParallelConfig())
    # Run analyze under the SAME phase binding the serving path uses so the
    # per-op shape_in is phase-correct (decode → Q seq collapses to Sq=1,
    # Lq=cache_len). Without this, decode jobs would carry prefill shapes.
    #
    # `s_verify` is the speculative-decoding verify width: one step submits the
    # draft's K tokens together, so the query block is K wide over a cache of
    # cache_len + K − 1. Every op sees it — a GEMM's M grows K×, attention keeps
    # the full cache but widens its query block, which is exactly the regime
    # where a decode kernel starts behaving like a small prefill.
    #
    # A cell is walked once per RANK it has: the rows the plan keys kernels
    # by are what one rank runs, and under pipeline parallelism no rank runs
    # the whole model. `_analyze_cell` therefore answers with a list of
    # ranks, each a list of analyses (an encode step walks one analysis per
    # encoder), and `_one_job_per_launch` folds the ranks back into one job
    # per launch at the end.
    # The cell's one walk, and the point it is read at first: the first
    # point's context at the largest batch any point asks there.
    if not points:
        raise ValueError("a plan names at least one point")
    from backend.inference.step_eval import evaluate_step
    from backend.inference.step_eval import prefix_refusal
    profile = (precision if isinstance(precision, PrecisionProfile)
               else PrecisionProfile.from_request(precision))
    shared = shared_walk_cells(resolved, parallel, profile, model_id=model_id)
    resolved = shared.resolved
    cells = shared.cells

    def _analyze_cell(point: PlanPoint) -> List[List[Any]]:
        if phase == "encode":
            return [_encode_results(point.batch)]
        return _step_results(point)

    def _encode_results(batch_size: int) -> List[Any]:
        # Each encoder over the stated media, in its own encode phase: the
        # shapes its kernels run at grow from the media's axes.
        if not media:
            raise ValueError(
                "an encode plan states the media each request carries "
                "(`media`, e.g. {'pixel_values': {'n_img': 1, 'img_h': ..., "
                "'img_w': ...}}); /inference/media-inputs lists the axes")
        from backend.arch.eval.program import Program
        from backend.inference.step_eval import evaluate_encode
        program = Program.of(resolved)
        stacks = sorted({entry.stack for name, entry
                         in program.model_inputs.items()
                         if name in media and entry.stack is not None})
        return [analyze(
            resolved, parallel=parallel, framework_binding=binding,
            batch_size=batch_size, phase=phase,
            step_eval=evaluate_encode(resolved, parallel, stack=stack,
                                      batch=batch_size, media=media,
                                      precision=precision),
            timed_stacks=frozenset({stack}),
            precision=precision,
            gpu=card,
            allow_load_refusals=True,
            breakdown=False,
        ) for stack in stacks]

    def _step_results(point: PlanPoint) -> List[List[Any]]:
        refusal = prefix_refusal(parallel, phase=phase,
                                 input_tokens=point.seq_len,
                                 s_prefix_cached=point.s_prefix_cached)
        if refusal is not None:
            raise ValueError(refusal)

        def at_stage(stage: Optional[int]) -> Any:
            return analyze(
                resolved, parallel=parallel, framework_binding=binding,
                batch_size=point.batch,
                # Every stage reads the one walk of the cell (`StepCells`).
                # The verify width binds the walk's own `n_verify` too, as
                # the estimate's walk does (`overlay_walk`): a verify step
                # takes logits at every verified position, not only the last.
                step_eval=evaluate_step(
                    resolved, parallel, phase=phase, batch=point.batch,
                    input_tokens=point.seq_len,
                    output_tokens=avg_output_tokens,
                    s_prefix_cached=point.s_prefix_cached,
                        s_verify=point.s_verify, precision=precision,
                        ep_load_factor=point.ep_load_factor,
                    cells=cells),
                phase=phase,
                # The pipeline stage whose rank these rows describe: what it
                # runs, and the hops across its own exit cut
                # (`layout.stages.StageView`). None is the whole model, which
                # is one rank only when there is no pipeline.
                stage=stage,
                # MTP runs real kernels. Estimate timing stays backbone-only;
                # the harness measures every stack the step walks. An encoder
                # is no part of the step: it is planned in its encode phase.
                include_aux_timing=True,
                precision=precision,
                gpu=card,
                allow_load_refusals=True,
                breakdown=False,
            )

        # One rank per pipeline stage. A per-layer kernel's `count_per_step`
        # is then the layers THIS rank holds — 15 or 16 of DeepSeek-V3's 61
        # at pp=4, not all 61 — and so are its FLOPs and bytes, because the
        # stage is what the walk counted them over. Reading the whole-model
        # ledger instead billed every rank for the whole model, silently (the
        # parallel signature carries `pp`, so the rows got their own key and
        # were found, with numbers no rank runs) and by a factor that grows
        # with pp.
        if parallel.pp > 1:
            return [[at_stage(s)] for s in range(parallel.pp)]
        return [[at_stage(None)]]


    # Per-rank shapes come from analyze's own rows: its walk applies the
    # deployment's sharding (TP shards heads/FFN, EP the expert count, CP the
    # sequence) and emits post-fusion spans already at per-rank geometry —
    # verified to agree with the forward-derived per-rank walk on every op
    # they share (tp=2, ep=2, both phases, three models). The second engine
    # walk that used to supply them (per_rank_shapes, and with it the
    # pre-fusion↔post-fusion sid join) is gone; the row's own
    # shape_in / shape_in_exact / shape_out are the source.

    # Paged-KV page size is a DEPLOYMENT parameter, and which value a backend
    # can run is the backend's own constraint (FlashInfer [16,32,64], FlashMLA
    # [64], CUTLASS MLA [128]). Rather than pick one — which is what put 2867
    # rows on page_size=1, a layout no deployment can run — a kernel that
    # reads a paged cache is measured at each page the request asks for, and
    # each backend declines the ones it cannot serve.
    #
    # WHICH kernels those are is not a list here: a signature that reads the
    # cache declares a `role="kv"` operand (`OpSignature.reads_paged_cache`),
    # which is the same declaration the `kv` dtype slot comes from. The list
    # this replaced named three attention signatures and had already drifted
    # — it left out DSA's lightning indexer, which reads the same paged K.
    #
    # The page is an axis of such a kernel's cell (`op_signatures.PAGE_AXIS`,
    # derived from that same operand), so the rows are keyed once per page
    # (`point_keys` under a deployment at that page) and each page is its own
    # cell. It used to be folded in after the projection and kept out of the
    # physical key, so the jobs of three pages shared one key, and the row a
    # measurement wrote was refused at persist for an axis no signature
    # declared.
    _kv_block_sizes = tuple(kv_block_sizes) if kv_block_sizes else (
        binding.kv_block_size,)
    spec_hash = resolved.identity_hash()
    # A value that changes how much WORK a kernel does is no kernel parameter:
    # it is an axis of that one cell and travels in its shape. The Mamba-2
    # scan chunk is the example — the launch runs `ceil(s_query / chunk)`
    # tiles of it — so `mamba2_scan` declares it (`structural_axes`), the
    # walked program reads it off the checkpoint config at the fields the
    # framework binding names (`Program.scan_chunk_size`) and
    # `point_keys` hands it to the row's identity. The plan carries no copy.
    # Canonical parallel signature — MUST equal the query side's
    # `parallel_signature`, or the measured row's hardware_hash won't
    # match the estimate's lookup. Single source of truth, not f"tp={tp}"
    # (which only coincidentally matches tp=1).
    from backend.inference.calibration_overlay import parallel_signature
    parallel_str = parallel_signature(parallel)
    # The deployment the rows are keyed under. The physical point keys still
    # name no GPU; the run's card is stamped later
    # (`driver.physical_key_for`). The dtype environment nevertheless follows
    # this plan's card, because framework launch dispatch can change it.
    deployment = build_deployment_context(
        framework=framework, framework_binding=binding, parallel=parallel,
        gpu_sku="", phase=phase)
    def _step_seq(point: PlanPoint) -> int:
        """The query block a point's workload is stated at, as the estimate
        states it (`overlay_walk`): one request's media for an encoder."""
        return 1 if phase == "encode" else max(1, int(phase_binding(
            phase=phase, avg_input_tokens=point.seq_len,
            avg_output_tokens=avg_output_tokens,
            s_prefix_cached=point.s_prefix_cached,
            s_verify=point.s_verify)["s_query"]))

    def _jobs_from(results: List[Tuple[Any, List[Dict[str, Any]]]],
                   step_seq: int
                   ) -> Tuple[List[Tuple[Tuple[Any, ...], Dict[str, Any]]],
                              List[str], List[Dict[str, Any]]]:
        """Jobs for one rank at one batch, from each analysis and the per-op
        rows read at that batch (the analysis's own rows at the batch it
        walked).

        Each job comes out beside `_launch_identity(...)`, the identity the
        stage fold folds on."""
        jobs: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        dropped: List[str] = []
        seen_dropped = set()
        unplaced: List[Dict[str, Any]] = []
        seen_unplaced = set()

        def record(point: Any, reason: str, detail: str = "",
                   *, note: Optional[str] = None) -> None:
            """Remember one row the queue cannot ask for.

            `reason` is closed vocabulary. The optional note is the legacy
            display string; absent means keep the old no-signature spelling.
            """
            op = point.op or point.aggregation_id or "?"
            item = {"op": op, "aggregation_id": point.aggregation_id or op,
                    "reason": reason}
            if detail:
                item["detail"] = detail
            key = tuple(sorted(item.items()))
            if key not in seen_unplaced:
                seen_unplaced.add(key)
                unplaced.append(item)
            text = note if note is not None else op
            if text and text not in seen_dropped:
                seen_dropped.add(text)
                dropped.append(text)

        # One derivation for every row's identity and launch, shared with the
        # estimate's measured overlay — see `calibration_overlay.point_keys`.
        # It refuses (`NoShapeLaw`) a measurable row re-read at a batch its
        # walk stated no shape law for.
        for ar, rows in results:
            workload = WorkloadShape.of_walk(ar, seq_len=step_seq)
            # The rows keyed at each page the plan measures at. A row whose
            # kernel reads no paged cache keys alike under every page and is
            # read off the first.
            by_page = [point_keys(ar, rows, workload,
                                  replace(deployment, kv_block_size=page),
                                  model_id=model_id, spec_hash=spec_hash)
                       for page in _kv_block_sizes]
            for row, *at_pages in zip(rows, *by_page, strict=True):
                point = at_pages[0]
                op, row_shape_json, sig = point.op, point.shape_json, point.signature
                # `point_keys` owns the ordered leaf classification. The plan
                # consumes that closed result; its own launch-geometry refusals
                # below only explain a row that already reached an exact key.
                reason = point.placement_reason
                if reason == ZERO_LAUNCH:
                    record(point, reason)
                    continue
                if reason == DECLINED_REASON:
                    note = (f"{op} [{point.aggregation_id or op}]: "
                            f"{point.declined}")
                    record(point, reason, point.declined, note=note)
                    continue
                if reason == MISSING_SIGNATURE:
                    record(point, reason)
                    continue
                # TRUE per-rank geometry from analyze's own rows (see above). The
                # param's shape rides the row too — pipeline projected the slot
                # dims before export, so a TP job carries the shard a kernel allocates.
                # TRUE per-rank geometry from analyze's own rows (see above). The
                # param's shape rides the row too — pipeline projected the slot
                # dims before export, so a TP job carries the shard a kernel allocates.
                # The point's launch geometry (`RowIdentity.one_event`) is the
                # row's own shape for every row that gets here: the axes it
                # would lift are exactly the averaged ones `_starved_axes`
                # drops below, so nothing is rounded into a shape here.
                shp_in, shp_out = point.launch_shape_in, point.launch_shape_out
                shp_w = point.shape_weight
                s_window = point.leaf.get("s_window")
                # No key, no job — one check, and the point says why there is
                # none. A measurable op REQUIRES a real per-op shape
                # (`RowIdentity.of` refuses rather than fabricating one).
                # Drop and report, which names the missing projection instead
                # of hiding it behind a plausible number — fixing the
                # projection turns the drop back into a row.
                _signature = get_signature(sig)
                _paged = (_signature is not None
                          and _signature.reads_paged_cache)
                if reason == MISSING_PROJECTION:
                    why = (
                           f"{sig!r} reads a paged cache and the deployment "
                           f"states no page (the framework declares "
                           f"`kv_block_size: null` and the request named none)"
                           if _paged and _kv_block_sizes == (None,) else
                           f"{sig!r} tiles by the Mamba-2 scan chunk and the "
                           f"checkpoint config states none of the fields "
                           f"{binding.framework} reads it from "
                           f"{list(binding.ssm_chunk_size_from)}"
                           if (_signature is not None
                               and "ssm_chunk_size" in _signature.cell_dims
                               and ar.analysis.program.scan_chunk_size(
                                   binding.ssm_chunk_size_from) is None) else
                           f"no shape projection for signature {sig!r} "
                           f"(shape_in={shp_in}) — would fabricate model-wide dims")
                    note = f"{op} [{point.aggregation_id or op}]: {why}"
                    record(point, reason, why, note=note)
                    continue
                # A geometry no kernel can allocate is not a cell. The
                # per-rank shape of a node inside a MoE dispatch zone is an
                # AVERAGE — `B·n_experts_active·s / n_route_targets` rows per
                # expert — and the row renders it with `as_extent(...,
                # "floor")`, so an average below one row reads 0. Nothing can
                # build a 0-row tensor, so the providers built from the
                # identity scalars instead (`Invocation.dims`, the lossy
                # corpus key): the launch that got measured and the numerator
                # that got billed were two different geometries. That is how a
                # pointwise `mul` in DeepSeek-V4's compressor came to report
                # 3,660 GB/s on a card whose HBM tops out at 3,350 — 109% of
                # peak, with the cross-check green, because both sides of the
                # check saw the same substituted shape.
                #
                # Drop it and name the axis, exactly as a missing shape
                # projection is dropped: the hole is then visible and the fix
                # is a projection, not a rounding convention chosen here.
                #
                # A MoE dispatch zone is NOT that question. Its averaged
                # extent is `M`, the mean rows one expert receives, and
                # tokens shared out over experts land on a whole number only
                # by coincidence — so the owner's decision is that every
                # expert that receives rows is padded up to a whole number of
                # them and the launch runs `[touched, ceil(R/touched), H]`
                # (`RoutedLaunch`), at every value of `M` including under
                # one row. The padding is not hidden: the job carries
                # the law's exact row count beside the geometry, and the
                # numerators it carries are the same law AT that geometry
                # (`PointRow.geometry_flops`), so the latency and the
                # numerator it divides describe one launch.
                #
                # Which TENSORS are the routed buffer is the walk's
                # (`LeafAccess.routed_inputs` / `routed_output`, on the row
                # as `routed`), and its rows are the axis after its experts,
                # as the dispatch lays it out. Every other axis keeps both
                # refusals, inside a zone or out of it: a compressor's pool
                # that runs once every few tokens is a cadence, not a padded
                # buffer, and widening it would invent a launch per call.
                # The padding and the geometry numerators read the same
                # record (`RoutedLaunch`, via `_launch_rows` and
                # `calibration_overlay._at_geometry`), so no axis the plan
                # refuses could be quietly re-priced.
                starved = _starved_axes(row)
                if starved:
                    agg = str(row.get("aggregation_id", "") or op)
                    note = (f"{op} [{agg}]: per-rank shape has an axis below "
                            f"one row ({', '.join(starved)}) — an average, not "
                            f"a geometry; no kernel can be built at it")
                    record(point, MISSING_PROJECTION, note[note.index("]: ") + 3:],
                           note=note)
                    continue
                rounded = _rounded_axes(row)
                if rounded:
                    agg = str(row.get("aggregation_id", "") or op)
                    note = (f"{op} [{agg}]: per-rank shape has a fractional "
                            f"axis ({', '.join(rounded)}) — the launch "
                            f"geometry rounds it up, so the launch would run "
                            f"more rows than the walk billed")
                    record(point, MISSING_PROJECTION, note[note.index("]: ") + 3:],
                           note=note)
                    continue
                dims = json.loads(row_shape_json)
                moe_distribution = None
                if declares_routed_token_total(sig) and "n_tokens" in dims:
                    # The routed rows the launch runs: the cell's experts
                    # are the ones that receive rows, each at the same padded
                    # rows (`RoutedLaunch.launch_rows`), so the counts the
                    # provider is handed are `s_expert_rows` each — every
                    # launched expert alike, no ragged remainder.
                    moe_distribution = "balanced"
                # The dtype half of the measurement identity, as the one
                # key derivation stated it (`RowIdentity.dtype_env`): the
                # leaf's own weight format, the dtype its matmul runs at, its
                # own scales, the step's cache slot. The plan used to restate
                # it from the spec's quant scheme — a second derivation that
                # agreed on bf16 models and disagreed on every row of a
                # quantized one (gpt-oss and DSv4-Flash: the scale slot read
                # `fp32` where the walk says `ue8m0`, an fp4 expert's
                # activation `bf16` where the walk says `fp8_e4m3`). Those
                # rows are measured under a key the estimate never asks for,
                # which is GPU time spent on a row nobody reads.
                #
                # Kept as the bytes the owner produced, not just as the dict:
                # they are half of the measurement identity, and the stage
                # fold below folds on the owner's key rather than on a second
                # serialization of it.
                row_dtype_env_json = point.dtype_env_json
                row_dtype_env = json.loads(row_dtype_env_json)
                unshardable = _scale_block_refusal(row)
                if unshardable:
                    agg = str(row.get("aggregation_id", "") or op)
                    note = f"{op} [{agg}]: {unshardable}"
                    record(point, MISSING_PROJECTION, unshardable, note=note)
                    continue
                # How many times THIS RANK runs this op in a step, and the
                # count its numerators below are aggregated over — one
                # number, because the row is one rank's (the pipeline stage
                # `_step_results` walked). A row the rank holds no layer of
                # counts 0: a stage's row set still covers the whole model,
                # so the embedding is stage 0's row and the head the last
                # stage's, each 0 everywhere else. Zero launches is not a
                # job; the stage that does run it plans it. The count a job
                # carries is the LAUNCHES (`PointRow.launches`), the number
                # its per-launch numerators below were divided by — not the
                # row's instance count, which for an unrolled expert bank is
                # its layers times its experts while one launch runs the bank.
                if point.count < 1:
                    continue
                cnt = point.launches
                # Roofline numerators for `achieved`, from the WALK and only
                # from the walk. The row states this node's whole per-step,
                # per-rank cost — per rank being the pipeline stage walked
                # above — and a job times one launch, so the job carries that
                # cost over the launches the step makes of it.
                #
                # Nothing is recomputed from `dims`. `dims` is the LOSSY scalar
                # projection of the row's shapes, and a cost model read off it
                # is a second answer to a question the walk already answered —
                # the answer the estimate, the module breakdown and the SOL
                # gate all read (CLAUDE.md §2.7). The formula that used to live
                # here drifted from the walk on 6 of the 11 signatures it
                # covered, because the shapes do not carry what the walk knows:
                # that an FP8 GEMM writes BF16 and also writes and re-reads the
                # narrow copy of its own operand (`convert_bytes_per_rank`),
                # that a lookup table is read `min(table, table·lookups/rows)`,
                # that a norm reads its gain vector. An FP8 projection came out
                # at 0.28× its real traffic, and `achieved` MBU is the number
                # the write gate filters on (`emit.above_sol`).
                #
                # Dividing is what attention has always done here
                # (`flops_per_rank ÷ count_per_step`), and attention was the
                # one class whose billing invariant came out at exactly 1.0:
                # numerator and launch count are read off one row. Under
                # context parallelism it is also the only correct answer — the
                # stored shape is the KV STORAGE shard (Lq = S/cp) while the
                # compute attends the full context, so `4·B·N·Sq·Lq·D` from the
                # sharded shape under-counts by cp× where the walk's
                # `flops_per_rank` is cp-correct. It is also what makes one
                # stage's job stand for its neighbour's: both read their own
                # row, and `_one_job_per_launch` keeps the busiest.
                #
                # A row whose launch happens once every m calls (`cadence` =
                # 1/m, DSv4's compressor pool) is keyed as ONE launch; the
                # walk's numerators are averaged over the m calls, so one
                # launch did m times a call's share. The division happens once,
                # where the launch count is stated (`PointRow.flops_per_launch`
                # / `access_bytes_per_launch`), not again here.
                #
                # At the geometry the launch RUNS, which for a routed cell is
                # wider than the rows its law states: `M` is a mean and every
                # expert that receives rows is padded up to whole rows, so a
                # measurement of this cell times `[touched, ceil(R/touched),
                # H]` and its latency has to divide that geometry's work.
                # The same law at the padded
                # rows, computed where the law's own decomposition is stated
                # (`PointRow.geometry_flops`) and not re-derived here; equal
                # to the walk's own numerators, bit for bit, on every row
                # nothing is padded on.
                b_flops = point.geometry_flops
                b_bytes = point.geometry_access_bytes
                # One job per page the deployment may run, for the kernels
                # that read a paged cache — each page is its own cell, with
                # the page in its dims and its key; one job for everything
                # else, whose cell has no page. The numerators do not change
                # with the page: a page is a layout of the same bytes.
                # A page the caller asked to sweep is kept for the kernels
                # whose own backends serve it; the deployment's own page
                # (no list asked) is the estimate's key and always kept.
                if _paged and kv_block_sizes:
                    served = served_kv_block_sizes(sig)
                    paged_points = [ap for ap, pg in zip(at_pages, _kv_block_sizes, strict=True)
                                    if pg in served]
                else:
                    paged_points = at_pages
                for at_page in (paged_points if _paged else [point]):
                    key = at_page.key
                    jobs.append((_launch_identity(at_page), {
                        "signature": sig,
                        "op": op,
                        # The leaf's variant provenance (`PointRow.leaf`), kept
                        # beside (not inside) the physical key for campaign
                        # facets such as H4's model/page/dtype/variant slices.
                        "variant": dict(point.leaf),
                        "dims": json.loads(at_page.shape_json),
                        # The shapes the identity above was projected FROM, carried
                        # whole. The projection is lossy on purpose — a corpus key must
                        # be scalars — but input construction needs the layout, and
                        # rebuilding it from the scalars is how the harness came to
                        # hold a second, unreconciled statement of it.
                        "shape_in": shp_in,
                        "shape_out": shp_out,
                        "shape_weight": shp_w,
                        "dtype_env": row_dtype_env,
                        # The measurement key, as the one derivation stated
                        # it (`PointRow.key`); the card is the run's, so the
                        # driver stamps that and nothing else.
                        "physical_key": [key.signature, key.shape_json,
                                         key.dtype_env_json, key.phase],
                        **({} if card is not None
                           else {"plan_scope": "cardless-shapes"}),
                        "phase": phase,
                        # The deployment's ATTENTION-TP degree, read off the
                        # config the walk sharded by — the legacy `tp` int
                        # only seeds that config when `parallel` is not
                        # given (:297-304), so stamping the argument would
                        # leave every `parallel=`-planned sharded cell at a
                        # stale 1 (`sweep.py:392-396` plans exactly that
                        # way). DCP's kv_tp = attn_tp / dcp split is not
                        # this scalar; no routine mode plans dcp today.
                        "tp": parallel.tp_attention,
                        "kernel_params": dict(
                            # The scalars this kernel takes beside its
                            # operands — the bound a clamped SwiGLU clamps
                            # to, SiTU's betas, a router's expert grouping
                            # — read off the ROW, where eval bound them at
                            # the node (`OpSignature.params_from_row`). The
                            # signature names which of them its kernels
                            # take; a job carries those and no others, and
                            # nothing here reads a config or the model's
                            # dims table. Absent means absent: the
                            # provider declines and `report` shows the gap.
                            point.kernel_params,
                            **({"moe_token_distribution": moe_distribution}
                               if moe_distribution is not None else {}),
                            **({"s_window": int(s_window)} if s_window else {}),
                            **({"sinks": True} if point.leaf.get("sinks") else {})),
                        "flops_per_rank": b_flops,
                        "access_bytes_per_rank": b_bytes,
                        # How many rows the launch above runs, and how many
                        # the walk's law states for it. They differ on a
                        # routed cell, where `M` is a MEAN and every expert
                        # that receives rows is padded up: the pair is what
                        # says how much of the measured launch was padding, so a
                        # reader never has to take the geometry for the
                        # routing. Equal on every other row.
                        "launch_rows": point.launch_rows,
                        "launch_rows_exact": str(point.launch_rows_exact),
                        # The routed boundary the launch is read off
                        # (`RoutedLaunch.wire`), None on no routed buffer:
                        # its expert counts and token side
                        # (`Invocation.routed`).
                        "routed": point.routed,
                        # The leaf carries a recurrent state whose read and
                        # write-back the numerators above include
                        # (`RowIdentity.carries_state`): only a kernel that
                        # takes the state is this cell
                        # (`OpSignature.carries_state`).
                        **({"carries_state": True} if point.carries_state
                           else {}),
                        "model": model_id,
                        "spec_hash": spec_hash,
                        "node_spec_id": point.aggregation_id,
                        "parallel": parallel_str,
                        # How many times ONE RANK runs this op per step — the
                        # rank whose rows this is, so under pp it is the
                        # stage's layer count and not the model's.
                        # Provenance: the measurement key carries no count,
                        # and `launches x flops_per_rank` is what it has to
                        # multiply back to.
                        "count_per_step": cnt,
                        "moe_block_member": bool(row.get("moe_block_member")),
                    }))
        return jobs, dropped, unplaced

    def _jobs_over_ranks(ranks: List[List[Any]], rows_of: Any,
                         step_seq: int
                 ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """`_jobs_from` for each rank of the deployment, folded back into one
        job per launch."""
        per_rank = [_jobs_from([(ar, rows_of(ar)) for ar in analyses],
                               step_seq)
                    for analyses in ranks]
        dropped: List[str] = []
        unplaced: List[Dict[str, Any]] = []
        said: Set[str] = set()
        seen_records = set()
        for _jobs, notes, records in per_rank:
            for note in notes:
                if note not in said:
                    said.add(note)
                    dropped.append(note)
            for item in records:
                key = tuple(sorted(item.items()))
                if key not in seen_records:
                    seen_records.add(key)
                    unplaced.append(item)
        return (_one_job_per_launch([jobs for jobs, _notes, _records in per_rank]),
                dropped, unplaced)

    # A step that does not speculate is walked with its verify width as the
    # number 1 (`Workload.walked_as_number`), so its walk is no walk of a
    # wider verify: the points split into one group per answer, each
    # walked at its own anchor and read only at its own points.
    from backend.arch.eval.transforms import Workload
    groups: Dict[bool, List[PlanPoint]] = {}
    for point in points:
        groups.setdefault(Workload.walked_as_number("s_verify", point.s_verify),
                          []).append(point)
    out: Dict[PlanPoint, Tuple[List[Dict[str, Any]], List[str],
                               List[Dict[str, Any]]]] = {}
    for group in groups.values():
        out.update(_read_group(group, _analyze_cell, _jobs_over_ranks,
                               _step_seq, parallel, phase))
    return out


def build_jobs_over_points(
    *,
    model_id: str,
    points: Sequence[PlanPoint],
    phase: str,
    avg_output_tokens: int,
    tp: int = 1,
    parallel: Optional[Any] = None,
    framework: str = DEFAULT_FRAMEWORK,
    resolved: Optional[Any] = None,
    binding: Optional[Any] = None,
    precision: Optional[Any] = None,
    media: Optional[Dict[str, Dict[str, int]]] = None,
    kv_block_sizes: Optional[Sequence[int]] = None,
    gpu: Optional[str] = None,
    with_unplaced: bool = False,
) -> Dict[PlanPoint, Tuple[List[Dict[str, Any]], List[str],
                           List[Dict[str, Any]]]]:
    """`{point: (jobs, notes, unplaced)}` for one (model, config, phase) at each of
    `points` (`PlanPoint`); the compatible two-value form of
    `_build_jobs_over_points`.

    `with_unplaced=True` returns each point's structured sidecar record as a
    third element. Existing two-value callers (and the queue) are unchanged.
    """
    by_point = _build_jobs_over_points(
        model_id=model_id, points=points, phase=phase,
        avg_output_tokens=avg_output_tokens, tp=tp, parallel=parallel,
        framework=framework, resolved=resolved, binding=binding,
        precision=precision, media=media, kv_block_sizes=kv_block_sizes,
        gpu=gpu)
    if with_unplaced:
        return by_point  # type: ignore[return-value]
    return {point: (jobs, notes)
            for point, (jobs, notes, _records) in by_point.items()}


def _read_group(points, _analyze_cell, _jobs_over_ranks, _step_seq, parallel,
                phase):
    """One walk's points, read from one canonical anchor.

    The anchor is the smallest admitted workload tuple and the largest batch
    that asks it — not the first point a caller happened to list. The cell's
    walk is symbolic, so caller order was never allowed to change its meaning;
    making the anchor canonical also keeps two callers from splitting the same
    walk family by their input order. A point the deployment refuses before the
    walk contributes its own dropped note and does not become the anchor."""
    from backend.arch.eval.analysis import Point
    from backend.inference.batch_sweep import rows_at
    from backend.inference.comm_cost import Fabric
    from backend.inference.step_eval import prefix_refusal
    def refused(point: PlanPoint) -> Optional[str]:
        if phase == "encode":
            return None
        return prefix_refusal(parallel, phase=phase,
                              input_tokens=point.seq_len,
                              s_prefix_cached=point.s_prefix_cached)

    def refusal_note(point: PlanPoint, refusal: str) -> str:
        return (f"prefill point seq_len={point.seq_len} "
                f"s_prefix_cached={point.s_prefix_cached} "
                f"batch={point.batch}: {refusal}")

    point_refusals = [(point, refused(point)) for point in points]
    admitted = [point for point, refusal in point_refusals
                if refusal is None]
    if admitted:
        anchor_shape = min((pt.seq_len, pt.s_prefix_cached, pt.s_verify)
                           for pt in admitted)
        anchor = max(
            (pt for pt in admitted
             if (pt.seq_len, pt.s_prefix_cached, pt.s_verify) == anchor_shape),
            key=lambda pt: pt.batch)
        walked = _analyze_cell(anchor)
    else:
        anchor = None
        walked = None
    out: Dict[PlanPoint, Tuple[List[Dict[str, Any]], List[str],
                               List[Dict[str, Any]]]] = {}
    for point, refusal in point_refusals:
        if refusal is not None:
            out[point] = ([], [refusal_note(point, refusal)], [])
            continue
        if walked is None:
            raise RuntimeError(
                "a point was admitted after the walk found no anchor")
        if point == anchor:
            out[point] = _jobs_over_ranks(walked, lambda ar: ar.per_op_timing,
                                          _step_seq(point))
            continue
        if phase != "encode":
            refusal = prefix_refusal(parallel, phase=phase,
                                     input_tokens=point.seq_len,
                                     s_prefix_cached=point.s_prefix_cached)
            if refusal is not None:
                raise ValueError(refusal)
        at = Point(point.counts(phase))
        for analyses in walked:
            for ar in analyses:
                at.admit(ar)
        # The rows at the point, read off the walk's laws: the plan keys
        # kernels by the rows' shapes and prices nothing, so on no card and
        # no fabric.
        out[point] = _jobs_over_ranks(
            walked,
            lambda ar, at=at: rows_at(
                ar.per_op_timing, at, cell=ar,
                peak_flops=None, peak_bw=None,
                fabric=Fabric(None, None, None), breakdown=False),
            _step_seq(point))
    return out


def _launch_identity(point: Any) -> Tuple[Any, ...]:
    """What makes two jobs the same launch.

    The measurement key itself (`PointRow.key`, the one derivation —
    signature, projected shape, dtype environment, phase; the card is
    stamped later by `driver.physical_key_for`), beside the node it belongs
    to. No second reading of `dims` or `dtype_env` that could drift from the
    key the query side asks with. A paged kernel's page is an axis of its
    cell, so it is in the key already.
    """
    return (point.key, point.aggregation_id)


def _one_job_per_launch(
    per_rank: Sequence[List[Tuple[Tuple[Any, ...], Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """One job per launch over the ranks a deployment has.

    Under pipeline parallelism every stage is walked, so a per-layer kernel
    is a job on every stage holding a layer of it. They are one measurement:
    a measurement key carries no stage (`BenchmarkKey` — model, spec, phase,
    parallel signature, gpu, op, node, shape, dtype env), and the launch one
    stage runs is the launch its neighbour runs, same dims and same dtypes.
    What differs is how often a rank runs it, and of the stages that run it
    the busiest is the one that has to fit — the reason `stage_layer_ranges`
    hands its remainder out the way it does — so that stage's job stands for
    all of them. One job per stage instead would put pp rows under one key
    with pp different counts, which is the collision this folds away.

    The two numerators are the busiest stage's for the same reason they are
    not averaged: a row's FLOPs are rounded once, after scaling by its count
    (`phases.compute_per_op_timing`), so `flops_per_rank / count` differs
    between stages by up to half a FLOP and the longest stage states it most
    precisely. Everything else a job carries must be equal across the stages
    it folds, and the fold says which field is not rather than picking one.

    A deployment with one rank (no pipeline, or an encode step) comes back as
    walked: nothing is folded that was not walked twice.
    """
    if len(per_rank) == 1:
        return [job for _launch, job in per_rank[0]]
    # The three fields the fold is allowed to choose between: how often the
    # rank runs the launch, and the two numerators which are the walk's
    # total over exactly that count. Every OTHER field — the geometry a
    # provider builds its inputs from included — has to agree.
    chosen = ("count_per_step", "flops_per_rank", "access_bytes_per_rank")
    busiest: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for launches in per_rank:
        for launch, job in launches:
            found = busiest.get(launch)
            if found is None:
                busiest[launch] = job
                continue
            for field, value in job.items():
                if field not in chosen and found[field] != value:
                    raise ValueError(
                        f"{job['op']} [{job['node_spec_id']}]: pipeline "
                        f"stages run this launch at different {field} "
                        f"({found[field]!r} vs {value!r}) under one "
                        f"measurement key; the key names no stage, so one "
                        f"of them would be recorded as the other")
            if job["count_per_step"] > found["count_per_step"]:
                busiest[launch] = job
    return list(busiest.values())


def jobs_jsonl(jobs: List[Dict[str, Any]]) -> str:
    """`jobs` as the jobs.jsonl text — one job per line.

    The one serialization of a job list, so the file `hbench plan` writes and
    the body an HTTP export hands a browser are the same bytes. A second
    writer (a client re-serializing the job dicts it was sent) would differ
    on key order and on how a float prints, and the harness reads this file
    back by line (`read_jobs_jsonl`).
    """
    return "".join(json.dumps(j) + "\n" for j in jobs)


def write_jobs_jsonl(jobs: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(jobs_jsonl(jobs))


def read_jobs_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def split_by_candidates(jobs: Sequence[Dict[str, Any]], box: Any
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """`(queued, gaps)`: every job `box` (a `box.BoxEnvs`) can measure,
    annotated with its `candidates` (registered kernel -> the library whose
    env runs it, as `providers.candidates` picks them from the box's envs),
    and every job it cannot, annotated with `no_provider` (kernel -> why it
    declines; "" when no kernel targets the signature at all). Both carry
    `judged_on`, the premises the judgement rests on (`BoxEnvs.premises`),
    so a run on other premises asks again (`run-all`).

    A gap is not dropped. It is a cell the corpus still lacks, written beside
    the plan (`gaps_path`) so `report` counts it as `no provider` — only the
    GPU stops spending a pass on it. The env and the canary still judge each
    queued job again at run time; this is the declarations, asked early."""
    from .invocation import Invocation
    from .providers import candidates, dependency_manifest, reads_the_key

    envs = box.envs()
    judged_on = box.premises(dependency_manifest())
    queued: List[Dict[str, Any]] = []
    gaps: List[Dict[str, Any]] = []
    for job in jobs:
        # A job judged before keeps nothing of that judgement.
        job = {k: v for k, v in job.items()
               if k not in ("candidates", "no_provider", "judged_on")}
        try:
            inv = Invocation.from_job(job)
        except ValueError as e:
            gaps.append({**job, "judged_on": judged_on,
                         "no_provider": {"": f"invalid job: {e}"}})
            continue
        chosen, why = candidates(inv, envs)
        if chosen:
            queued.append({**job, "judged_on": judged_on, "candidates": {
                p.kernel_impl: p.library
                for p in sorted(chosen, key=lambda p: p.kernel_impl)}})
        else:
            # Why this cell has no kernel decides whether it is the box's
            # gap or the key's error (`providers.reads_the_key`). Both stay
            # gaps and both stay in the coverage denominator; the second is
            # named so it cannot sit there looking like a card nobody owns.
            unreadable = reads_the_key(inv)
            gaps.append({**job, "judged_on": judged_on, "no_provider": why or {
                "": f"no kernel registered for {inv.signature}"},
                **({"unreadable_key": unreadable}
                   if unreadable is not None else {})})
    return queued, gaps


def write_plan(jobs: Sequence[Dict[str, Any]], path: str, box: Any = None,
               unplaced: Sequence[Dict[str, Any]] = ()
               ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """THE writer of a plan a person or the queue reads: the jobs `box` can
    measure to `path`, the rest to `gaps_path(path)`. Without a box every
    job goes to `path` unjudged. Structural rows (`unplaced`, including
    zero-launch records) go beside the plan under `unplaced_path(path)`; the
    queue does not read that file. Either way a sidecar the new plan does not
    have is removed, so `report` never counts an earlier plan's rows. (The
    run's own `remaining-*.jsonl` are not plans and keep `write_jobs_jsonl`.)
    Returns `(queued, gaps)`."""
    import os

    if box is None:
        queued, gaps = list(jobs), []
    else:
        queued, gaps = split_by_candidates(jobs, box)
    write_jobs_jsonl(queued, str(path))
    if gaps:
        write_jobs_jsonl(gaps, gaps_path(path))
    elif os.path.exists(gaps_path(path)):
        os.unlink(gaps_path(path))
    records = [dict(record) for record in unplaced]
    if records:
        write_jobs_jsonl(records, unplaced_path(path))
    elif os.path.exists(unplaced_path(path)):
        os.unlink(unplaced_path(path))
    return queued, gaps


def gaps_by_signature(gaps: Sequence[Dict[str, Any]]) -> List[str]:
    """One line per signature: how many cells no kernel serves on this box,
    and the most common reason each targeting kernel gave.

    A cell no kernel ANYWHERE reads (`unreadable_key`, set by
    `split_by_candidates`) is counted in the same total and called out on
    its own line: it is not waiting for a card, it is keyed to a launch
    nothing in the corpus performs, and the fix is the key."""
    from collections import Counter, defaultdict

    cells: Counter = Counter()
    reasons: Dict[str, Counter] = defaultdict(Counter)
    unreadable: Dict[str, Counter] = defaultdict(Counter)
    for job in gaps:
        sig = job.get("signature") or job.get("op", "?")
        cells[sig] += 1
        for kernel, why in job["no_provider"].items():
            reasons[sig][f"{kernel}: {why}" if kernel else why] += 1
        if job.get("unreadable_key"):
            unreadable[sig][job["unreadable_key"]] += 1
    out = []
    for sig, n in cells.most_common():
        out.append(f"{sig}: {n} cell(s) — " + "; ".join(
            f"{why} ×{k}" for why, k in reasons[sig].most_common(3)))
        for env, k in unreadable[sig].most_common(3):
            out.append(f"    KEYING ERROR: {k} cell(s) keyed {env}, which no "
                       f"registered kernel reads on any card — the key names "
                       f"a launch nothing performs, so fix the key, not the box")
    return out


def unplaced_by_reason(records: Sequence[Dict[str, Any]]) -> List[str]:
    """One line per closed structural reason, in report order.

    A projection note can name the axis or deployment fact it lacked. The
    closed `reason` stays machine-readable; the detail is display-only.
    """
    from collections import Counter

    order = (ZERO_LAUNCH, DECLINED_REASON, MISSING_SIGNATURE, MISSING_PROJECTION,
             LEGACY_DROP)
    counts: Counter = Counter()
    examples: Dict[str, List[str]] = {}
    allowed = set(order)
    for record in records:
        reason = record.get("reason")
        if reason not in allowed:
            raise ValueError(
                f"unknown structural reason {reason!r}; one of {tuple(sorted(allowed))}")
        counts[reason] += 1
        if len(examples.setdefault(reason, [])) < 3:
            example = f"{record.get('op', '?')} [{record.get('aggregation_id', '?')}]"
            if record.get("detail"):
                example += f": {record['detail']}"
            examples[reason].append(example)
    meanings = {ZERO_LAUNCH: "measured@0, no GPU launch",
                DECLINED_REASON: "the exact key was declined",
                MISSING_SIGNATURE: "no measurement signature binds the op",
                MISSING_PROJECTION: "the signature has no stated projection",
                LEGACY_DROP: "an old two-value planner dropped this row"}
    return [f"{reason}: {counts[reason]} row(s) ({meanings[reason]}) — "
            + "; ".join(examples[reason])
            for reason in order if counts[reason]]


def gaps_path(plan_path: str) -> str:
    """Where a plan's gaps live: beside it, under a name no queue reads as a
    plan (the queue takes `*.jsonl`)."""
    plan_path = str(plan_path)
    stem = plan_path.removesuffix(".jsonl")
    return stem + ".no_provider"


def unplaced_path(plan_path: str) -> str:
    """Where a plan's structural records live: beside it, under a name no
    queue reads as a plan (the queue takes `*.jsonl`)."""
    plan_path = str(plan_path)
    stem = plan_path.removesuffix(".jsonl")
    return stem + ".unplaced"
