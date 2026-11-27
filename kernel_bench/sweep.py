"""sweep — enumerate the measurement grid that the inference tool ACTUALLY
queries, gated by each GPU's VRAM.

The single-point `plan` (one batch, one seq, tp=1) covers almost none of the
cells the charts plot. The tool sweeps, per model:
  - context (seq)  : a 9-point log grid, 1K … 1M
  - batch          : {1, 16, 64, max} per phase (max = KV-bound capacity ceiling)
  - parallelism    : tp=1 baseline + deployment modes × power-of-two degrees
  - GPU            : every SKU is a peer; its VRAM decides which cells fit

This module expands (model, gpu) into the deduped per-op job list for the cells
that are *feasible on that GPU's memory* — using the tool's own
`estimate_capacity` as the feasibility oracle (a cell the tool marks infeasible
shows no data, so measuring it is wasted). Parallelism is realized as per-RANK
sharded shapes (via `build_jobs(parallel=...)`), so the measured rows key to the
same sharded identity the estimate looks up.

Modes are included only when EVERY axis they shard is reflected per-rank in the
identity (else a row mis-keys / over-counts):
  - tp_tp : pure TP — heads/FFN/expert-GEMM shard by K/N (via analyze). ✓
  - tp_ep : TP + expert parallel — heads shard via TP, expert COUNT shards via
            ep_moe (identity_shape_from_shape_in now divides E by ep on BOTH the
            write and query sides). batch/seq correctly unsharded. ✓
  - dp_ep : data-parallel attention + expert parallel — attention weights are
            REPLICATED (dp doesn't shard heads), experts shard by ep_moe. dp does
            NOT touch the attention batch axis: the batch chain is per-rank end to
            end — capacity's `max_*_batch_per_gpu` is the per-dp-rank ceiling
            (cluster batch = that × dp_attention, capacity.py:504/1009), sweep
            feeds that per-GPU `b` unchanged, and the engine leaves attention
            `B = b` while only scaling the MoE gather `M = ⌈B·S·K·dp/E⌉`
            (engine.py:_derive_M_dim). So the identity's attention B already
            equals the batch one rank runs — no ×dp over-count. This mode is
            what lets a 500G+ MoE fit at all: ep_moe=E (E≫heads) shards the
            expert weights past the tp head-cap. ✓
  - cp_ep : context parallel + expert parallel. The attention identity DOES
            carry the cp-sharded geometry (analyze is run under the phase
            binding, so prefill gives pcp's `Sq/cp` queries against the full
            context, and decode gives DCP's `Lq/cp` KV slice against Sq=1) —
            verified against the engine's own flops in
            TestAttentionCostFromEngine and re-checked per phase on a real
            plan. ✓
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Set, Tuple)

# The routine shape grid — single source of truth (backend.inference.sweep_grid),
# shared with every chart surface. "max" batch resolves to the
# capacity ceiling per cell.
from backend.arch.eval.precision import AS_SHIPPED
from backend.inference.framework_bindings import DEFAULT_FRAMEWORK
from backend.inference.capacity import point_served
from backend.inference.parallel_sweep import ALL_MODES
from backend.inference.sweep_grid import (
    ROUTINE_SEQ_LENS as SWEEP_SEQ_LENS,
    TrackSpec,
    batch_point,
    requests_needed,
)
from .prefill_chunks import (
    DEFAULT_PREFILL_CHUNK_SIZE,
    decompose_prefill,
    normalize_prefill_chunk_size,
    prefill_chunk_map,
)

# The parallelism modes a sweep measures: every mode the Scaling panel prices
# (`parallel_sweep.ALL_MODES` — the vocabulary's one definition), not a second
# list of them. tp_tp = pure TP; tp_ep = TP + expert parallel (E shards by
# ep_moe); dp_ep = dp-attention (replicated heads) + expert parallel — the batch
# chain is per-rank end to end (dp only scales MoE M, never the attention B), so
# its identity is correct and its ep_moe=E degree breaks the tp head-cap that
# left 500G+ MoE models with 0 feasible cells; cp_ep = context parallel + expert
# parallel, whose attention identity already carries the cp-sharded geometry.
# Each one's per-rank identity is physically correct (see the module
# docstring), which is not a kernel_bench judgement: a mode whose per-rank
# shapes were wrong would be priced wrong on the panel too, so it would not
# be in that vocabulary either.
SAFE_MODES = ALL_MODES


def resolve_sweep_model(model_id: str) -> Tuple[dict, Any]:
    """`(config, resolved)` for the model a sweep plans, the spec matched
    once (`SpecLoader.match` is ~900ms with no cross-instance cache): the
    caller hands the spec to `routine_parallel_configs` and
    `build_sweep_jobs` alike."""
    from backend.analyzer import fetch_config
    from backend.arch.loader import SpecLoader
    config = fetch_config(model_id)
    if config is None:
        raise RuntimeError(f"no config for {model_id!r}")
    resolved = SpecLoader().match(config, model_id=model_id)
    if resolved is None:
        raise RuntimeError(f"no spec matches {model_id!r}")
    return config, resolved


def preset_parallel_configs(presets: Sequence[Dict[str, int]]
                            ) -> List[Tuple[str, Any]]:
    """(label, ParallelConfig) for each deployment preset a surface states
    (`sweep_grid`'s preset lists), built the way an estimate request builds
    its `parallel` (`ParallelConfig.from_dict`)."""
    from backend.inference.calibration_overlay import parallel_signature
    from backend.inference.parallel_config import ParallelConfig
    out = []
    for preset in presets:
        pc = ParallelConfig.from_dict(dict(preset))
        out.append((parallel_signature(pc), pc))
    return out


def routine_parallel_configs(config: dict, *, resolved: Any = None
                             ) -> List[Tuple[str, Any]]:
    """(label, ParallelConfig) for the baseline + every safe-mode degree that
    validates for this model. Baseline is single-rank (tp=1).

    `resolved` is the spec the sweep already matched once — passed through so
    the caps read from it rather than re-matching (matching no longer scores a
    config without a model_id)."""
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.parallel_sweep import (
        config_for_degree, legal_parallel_degrees, sweep_caps,
    )
    out: List[Tuple[str, Any]] = [("tp=1", ParallelConfig())]
    caps = sweep_caps(config, resolved=resolved)
    # The degrees the model deploys at are the layout's answer — the same
    # `legal_parallel_degrees` the Scaling panel asks — not a cap on heads:
    # the head cut offered degrees the layout refuses and withheld ones it
    # accepts by replicating (f904dbe63). A refused degree has no per-rank
    # shapes to measure.
    for mode in SAFE_MODES:
        out.extend((f"{mode}:{d['x']}", config_for_degree(mode, d["x"]))
                   for d in legal_parallel_degrees(
                       resolved, mode, caps)["accepted"])
    return out


def frontier_parallel_configs(resolved: Any, gpu: str) -> List[Tuple[str, Any]]:
    """(label, ParallelConfig) for every deployment the gallery's decode
    frontier prices on `gpu` (`parallel_sweep.frontier_layouts`) that the
    layout walk accepts — a refused one has no per-rank shapes to measure."""
    from backend.inference.parallel_sweep import (
        frontier_layouts, layout_label, layout_verdict, sweep_caps,
    )
    out: List[Tuple[str, Any]] = []
    for cfg in frontier_layouts(sweep_caps({}, resolved=resolved), gpu):
        try:
            layout_verdict(resolved, cfg)
        except (ValueError, NotImplementedError):
            continue
        out.append((layout_label(cfg), cfg))
    return out


def _ceilings_for_deployment(
    *, config: dict, gpu: str, parallel: Any, seq_lens: List[int],
    avg_output_tokens: int, precision: Any, resolved: Any = None,
) -> Dict[int, Any]:
    """L1 for one deployment across every seq — `{seq: CapacityCeilings}`.

    One query per (model, gpu, parallel), not per grid coordinate: the model
    only has to be built once for all of them, and building it is ~80% of a
    single-point query. The gate asks "does this fit, and up to what batch?" and
    reads nothing else, so it calls the capacity layer directly rather than the
    full L1..L4 composition (phases, per-op rows, formula strings).

    `avg_output_tokens` is the served output length: a request's cache holds
    its prompt and its output, so the KV-bound `max` batch is read at the
    output length the surface serves.
    """
    from backend.inference.capacity import estimate_capacity_ceilings_at
    # A crash here is NOT "infeasible" — masking it as [] silently drops a model
    # (e.g. a transient shared-tree read race in gpu_specs). Retry once; if it
    # still fails, propagate so the caller logs a GATE ERROR distinct from a
    # genuine no-fit. Only a real max_batch<=0 marks a cell infeasible.
    for attempt in range(2):
        try:
            return estimate_capacity_ceilings_at(
                config=config, gpu_sku=gpu, parallel=parallel,
                num_gpus=getattr(parallel, "cluster_size", 1) or 1,
                seq_lens=seq_lens, avg_output_tokens=avg_output_tokens,
                # The spec the sweep matched once — capacity uses it directly
                # instead of re-matching (its single-entry-point path).
                resolved=resolved,
                # The gate's precision, stated: the profile the sweep
                # measures under (as shipped unless the caller names one).
                precision=precision,
            )
        except Exception:
            if attempt == 1:
                raise
    return {}


# New query tokens one *legacy unchunked* prefill launch may carry while the
# sweep selects endpoint batch points.  This 32768 cap predates explicit chunk
# geometry and is NOT the chunk width: for `q > 32768` it floors the endpoint
# batch at 1 while H1/H2 still represent that launch through the separate
# policy below.  The name is kept because exported tooling and tests use it as
# the launch-query cap.
PREFILL_LAUNCH_TOKEN_CAP = 32768
PREFILL_TOKEN_BUDGET = PREFILL_LAUNCH_TOKEN_CAP

# The scheduler chunk width itself is a request/deployment/campaign knob.  0
# and None explicitly mean unchunked; a framework binding may document where a
# deployment default came from, but it must never infer this geometry silently
# from framework identity.  H2 feeds the pure decomposition's launch points
# to job planning: a one-chunk query keeps today's bytes, and a longer query
# is re-keyed at `Sq<=chunk,Lq=end`.
PREFILL_CHUNK_SIZE = DEFAULT_PREFILL_CHUNK_SIZE

# Sequences one DECODE forward pass carries. `max_batch` answers "how many
# streams fit in VRAM" — a capacity question, and at small models it reaches
# absurd widths: 521,391 for Qwen3-Coder-Next, half a million concurrent
# sequences in a single launch, which no scheduler admits and no deployment
# runs. Prefill already had its own ceiling for exactly this reason; decode
# was left with none because `B·1` is trivially inside any token budget.
#
# It is also where kernels stop working. sglang's causal_conv1d_update indexes
# its `[B, C, W]` state with int32 and faults once the element count crosses
# 2**31 — an illegal memory access, which poisons the CUDA context and takes
# the rest of the model's run with it. At the widest conv width in the corpus
# (C=8192, W=4) that boundary sits at B=65,536 exactly.
DECODE_BATCH_CEILING = 65536


def _per_gpu_ceiling(cap: Any, phase: str) -> int:
    """The batch the `max` sentinel resolves to on one card — the one place
    the plan reads it. A batch that goes into a shape is what one GPU runs,
    so it is the per-GPU field (`CapacityCeilings.max_batch`, wire name
    `max_batch_per_gpu`, for decode; `max_prefill_batch` for prefill) — the
    same fields an estimate's `max` point runs at (`capacity._direct_batches`).
    `max_batch_total` (the cluster's count) never resolves a point."""
    # The `max` sentinel resolves through the one rule
    # (`max_batch_rung`): the shape a measurement is built at is the
    # rung, the exact ceiling a capacity fact.
    from backend.inference.capacity import max_batch_rung
    return max_batch_rung(cap.max_prefill_batch if phase == "prefill"
                          else cap.max_batch)


def _feasible_batches(
        cap: Any, phase: str, seq: int, batch_keys: Sequence[str], *,
        prefill_query_tokens: Optional[int] = None) -> List[int]:
    """The batch points to measure at one already-gated cell — the grid
    entries named by `batch_keys` (`sweep_grid.batch_point`), each explicit
    batch kept where it is ≤ the phase ceiling and the `max` sentinel
    resolved to that ceiling. Empty when the deployment doesn't fit at all
    (weights over VRAM). Pure selection over an L1 result; it issues no
    query of its own.

    Prefill takes a SECOND ceiling that decode does not. `max_prefill_batch`
    answers "how many prefill streams coexist in the activation budget" — a
    capacity question. The benched shape is one FORWARD PASS, whose width is
    the query the scheduler admits: `B·q ≤ PREFILL_LAUNCH_TOKEN_CAP`. Without it
    the sweep asked for shapes no deployment runs — B=14681 at seq=1024 is 15M
    tokens in a single launch, a 37 GiB output tensor. Those jobs OOMed, and
    the ones that fit had their fp32 reference OOM instead, so they were
    recorded WITHOUT the numeric cross-check that is the harness's whole
    guarantee. Decode is unaffected: its B streams each contribute one token,
    so `B·1` is already within any chunk."""
    ceiling = _per_gpu_ceiling(cap, phase)
    if phase == "prefill" and seq > 0 and ceiling > 0:
        # `q` is the fresh prompt work, `L-P`, not total context: a prefix-
        # cached endpoint of 1 MiB with 4 KiB new tokens is admitted by its
        # 4 KiB query.  Floor at 1 past the cap: a single sequence is itself
        # chunked.  This selection remains endpoint batch policy; the chunk
        # policy below decides each launch's `B` and geometry.
        query = int(seq if prefill_query_tokens is None else prefill_query_tokens)
        if query <= 0:
            return []
        ceiling = max(1, min(ceiling, PREFILL_LAUNCH_TOKEN_CAP // query))
    else:
        # Decode's own ceiling — see DECODE_BATCH_CEILING. Applied here rather
        # than as a per-kernel gate because the shape is not one any deployment
        # asks for: declining it per provider would still plan the cell, build
        # its inputs, and leave the torch baseline measuring a launch nobody
        # runs.
        ceiling = min(ceiling, DECODE_BATCH_CEILING)
    if ceiling <= 0:
        return []                       # infeasible on this VRAM → skip cell
    # The grid's own encoding of the sentinel, per phase: prefill -1, decode
    # None (`sweep_grid.batch_point`).
    field, sentinel = (("prefill_batch_size", -1) if phase == "prefill"
                       else ("decode_batch_size", None))
    out = set()
    for key in batch_keys:
        bp = batch_point(key)
        # A point the deployment does not serve — past the model's context,
        # or more requests than fit — is no point on any page, so nothing
        # reads its measurement (`capacity.point_served`, the question a
        # page that draws the point asks).
        if not point_served(max_batch_per_gpu=_per_gpu_ceiling(cap, "decode"),
                            beyond_max_context=cap.beyond_max_context,
                            requests=requests_needed(key, phase)):
            continue
        b = bp[field]
        if b == sentinel:
            out.add(ceiling)
        elif isinstance(b, int) and b <= ceiling:
            out.add(int(b))
    return sorted(out)



def _deployed_binding(framework: str, gpu: str, precision: Any,
                      resolved: Any) -> Any:
    """The framework binding a sweep plans under, answered for the card."""
    # The binding, answered for the deployment being measured. A `when:`
    # condition the binding cannot answer is a kernel the framework will not
    # run — and `device: cuda` is on nine of them, so an unanswered binding
    # plans a graph with nine fewer fusions than the estimate it calibrates:
    # the cells measured would be the un-fused ops, and the fused spans the
    # estimate prices would have no measurement at all. The estimate side
    # does this (`routes/inference.py`, `playground/module_bench.py`); the
    # measurement side has to say the same thing.
    from backend.arch.eval.precision import ResolvedPrecision
    from backend.arch.eval.program import Program
    from backend.inference.framework_bindings import (ServingDeployment,
                                                      get_binding)
    return get_binding(framework).deployed(ServingDeployment.at(
        gpu_sku=gpu,
        activation_dtype=ResolvedPrecision(precision,
                                           Program.of(resolved)).model_dtype))


@dataclass
class _DeploymentPlan:
    """One deployment's share of a sweep, before the jobs every deployment
    plans are deduped against each other (`build_sweep_jobs`)."""

    jobs: List[Dict[str, Any]] = field(default_factory=list)
    cells: int = 0
    gate_error: bool = False
    reused: List[str] = field(default_factory=list)
    points: List[List[Any]] = field(default_factory=list)
    chunk_map: List[Dict[str, Any]] = field(default_factory=list)
    prefill_context_key_provenance: List[Dict[str, Any]] = field(
        default_factory=list)
    chunk_terms: int = 0
    chunk_endpoints: int = 0
    dropped: set = field(default_factory=set)
    unplaced: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


def _job_physical_key(job: Dict[str, Any]) -> Tuple[Any, ...]:
    """The exact launch key, from the job's own identity fields."""
    if job.get("physical_key"):
        return tuple(job["physical_key"])
    return (job["signature"], json.dumps(job["dims"], sort_keys=True),
            json.dumps(job["dtype_env"], sort_keys=True), job["phase"],
            json.dumps(job.get("kernel_params") or {}, sort_keys=True))


def _prefill_context_key_record(
        job: Dict[str, Any], model_id: str) -> Optional[Dict[str, Any]]:
    """Named H4 facets for a pre-dedup context-key request.

    The values come from the planner's structured job fields, never by parsing
    ``shape_json`` or ``dtype_env_json`` back out of the physical key.  The
    exact physical key remains a field so H4 can union the raw set first and
    group its provenance only when needed.
    """
    dims = job.get("dims") or {}
    if not isinstance(dims, Mapping) or not any(
            axis in dims for axis in ("s_kv", "s_ctx")):
        return None
    key = _job_physical_key(job)
    return {
        "physical_key": list(key),
        "signature": job.get("signature") or key[0],
        "model": model_id,
        "page": dims.get("kv_block_size"),
        "dtype": dict(job.get("dtype_env") or {}),
        "variant": dict(job.get("variant") or {}),
    }


def _facet_order(record: Dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, default=str)


def _plan_deployment(label: str, par: Any, *, model_id: str, gpu: str,
                     tracks: Sequence[TrackSpec], seq_lens: List[int],
                     avg_output_tokens: int, framework: str, precision: Any,
                     config: dict, resolved: Any, binding: Any,
                     prefill_chunk_size: Optional[int]
                     ) -> _DeploymentPlan:
    """Plan every cell one deployment serves (`build_sweep_jobs`)."""
    from backend.inference.phases import Speculation
    from backend.inference.step_eval import prefix_refusal
    from .plan import (LEGACY_DROP, PlanPoint, attach_prefill_chunk_provenance,
                       build_jobs_over_points)
    # Importing the attention providers registers them, which is
    # what makes their declared pages visible here.
    # `served_kv_block_sizes` discovers every provider module itself, so the
    # pages are the union of what is REGISTERED rather than of what this
    # caller remembered to import.
    from .providers import served_kv_block_sizes

    legacy_dropped: Set[str] = set()
    out = _DeploymentPlan()
    log = out.logs.append
    reused, points = out.reused, out.points
    # `build_jobs` names every op it had to drop — an op family with no shape
    # projection, or one no signature covers. Discarding that (the call used to
    # take `cell_jobs, _`) made whole kernel families vanish from the corpus
    # with nothing reported: `rope` and `moe_gate` are absent from every
    # sweep-planned run, which is why sglang's rope/gate providers had nothing
    # to measure even when they loaded. The plan says which projection to add;
    # it just has to reach the operator.
    dropped = out.dropped
    # One L1 query per deployment covers all of its (seq × phase) cells —
    # the model is built once here instead of once per coordinate.
    try:
        ceilings = _ceilings_for_deployment(
            config=config, gpu=gpu, parallel=par, seq_lens=list(seq_lens),
            avg_output_tokens=avg_output_tokens,
            precision=precision, resolved=resolved)
    except Exception as e:
        # A gate crash is NOT a fit verdict — surface it, don't let the
        # model look silently "infeasible".
        out.gate_error = True
        log(f"GATE ERROR {label}: {type(e).__name__}: {e}")
        return out
    # Each phase is one cell: its walk read at every (context, cached
    # prefix, verify width, batch) point the tracks ask of it
    # (`build_jobs_over_points`).
    phase_points: Dict[str, List[PlanPoint]] = {}
    phase_chunk_points: Dict[str, Dict[PlanPoint, List[Any]]] = {}
    phase_endpoint_indices: Dict[str, Dict[PlanPoint, int]] = {}
    for seq in seq_lens:
        for track in tracks:
            n = track.prefix_new_tokens
            if n is None:
                s_prefix = 0
            elif seq > n:
                s_prefix = seq - n
            else:
                s_prefix = 0
                reused.append(f"{label} {track.name} @ {seq}: context ≤ "
                              f"{n} new tokens, reads the {track.phase} "
                              f"cell with no prefix cache")
            refusal = prefix_refusal(par, phase=track.phase,
                                     input_tokens=seq,
                                     s_prefix_cached=s_prefix)
            if refusal is not None:
                log(f"skip {label} seq={seq} {track.phase} "
                    f"prefix={s_prefix}: {refusal}")
                continue
            for key in track.batch_keys:
                got = _feasible_batches(ceilings[seq], track.phase, seq,
                                        (key,),
                                        prefill_query_tokens=(
                                            seq - s_prefix
                                            if track.phase == "prefill"
                                            else None))
                if not got:
                    continue
                for drafts in track.draft_tokens:
                    verify = Speculation(drafts).s_verify
                    for rho in track.ep_load_factors:
                        wanted = phase_points.setdefault(track.phase, [])
                        chunk_points = phase_chunk_points.setdefault(
                            track.phase, {})
                        endpoint_indices = phase_endpoint_indices.setdefault(
                            track.phase, {})
                        for b in sorted(got):
                            endpoint = PlanPoint(seq, s_prefix, verify, b, rho)
                            endpoint_index = endpoint_indices.setdefault(
                                endpoint, len(endpoint_indices))
                            if track.phase == "prefill":
                                chunks = decompose_prefill(
                                    endpoint,
                                    prefill_chunk_size=prefill_chunk_size)
                            else:
                                chunks = ()
                            for chunk in chunks:
                                launch = chunk.point
                                point = replace(
                                    endpoint,
                                    seq_len=launch.seq_len,
                                    s_prefix_cached=launch.s_prefix_cached,
                                    s_verify=launch.s_verify,
                                    batch=launch.batch)
                                if point not in wanted:
                                    wanted.append(point)
                                if len(chunks) > 1:
                                    chunk_points.setdefault(point, []).append(
                                        (endpoint_index, chunk))
                            if not chunks and endpoint not in wanted:
                                wanted.append(endpoint)
                            if len(chunks) > 1:
                                out.chunk_endpoints += 1
                                out.chunk_terms += len(chunks)
                                mapped = prefill_chunk_map(
                                    endpoint,
                                    prefill_chunk_size=prefill_chunk_size,
                                    deployment=label,
                                    endpoint_index=endpoint_index)
                                assert mapped is not None
                                out.chunk_map.append(mapped)
                        points.append([label, track.name, key, seq, drafts,
                                       rho])
    for phase, wanted in phase_points.items():
        out.cells += len(wanted)
        try:
            by_point = build_jobs_over_points(
                model_id=model_id, points=wanted, phase=phase,
                avg_output_tokens=avg_output_tokens,
                parallel=par, framework=framework, precision=precision,
                resolved=resolved, binding=binding, gpu=gpu,
                # The pages to measure at are DERIVED from what
                # the backends declare they can serve
                # (`KernelSignature.kv_block_sizes`), not listed
                # here: a page no backend can run is a row
                # nothing will look up.
                #
                # BOTH PHASES. This was decode only, called a
                # budget decision, and the cost of that was not
                # budget: prefill fell back to the binding's own
                # global page, which is vLLM's CacheConfig default
                # of 16, and most paged kernels cannot run there.
                # Five of the seven signatures that read a paged
                # cache serve no 16 at all — the sliding-window
                # MLA cache declares `MultipleOf(32)` and prefers
                # 256 (vLLM `v1/attention/backends/mla/
                # sparse_swa.py:154`, `:158`), FlashMLA serves 64 —
                # so every prefill cell of theirs was keyed at a
                # page no deployment allocates, and the only thing
                # that caught it was each provider declining to
                # measure a paging layout that never happened. The
                # page is an axis of these cells in both phases
                # (`op_signatures.PAGE_AXIS`), so this changes
                # which cells are planned and no identity.
                kv_block_sizes=served_kv_block_sizes(), with_unplaced=True)
        except Exception as e:
            log(f"skip {label} {phase} ({len(wanted)} points): {e}")
            continue
        for point in wanted:
            planned = by_point[point]
            # The public API also has a two-value compatibility form. Real
            # plans pass `with_unplaced=True`. An older caller returning that
            # form still plans jobs and textual drops; those drops are
            # migrated, never treated as an empty structural denominator.
            if len(planned) == 3:
                cell_jobs, cell_dropped, cell_unplaced = planned
            else:
                cell_jobs, cell_dropped = planned
                legacy_dropped.update(str(note) for note in cell_dropped if note)
                cell_unplaced = []
            # `plan` names the offending shape in each note, which is
            # what you want for one cell and unreadable across
            # thousands — the same op reappears once per shape. Collapse
            # to the part that identifies the gap: op + signature.
            dropped.update(n.split(" (shape_in=")[0] for n in cell_dropped)
            if phase == "prefill":
                for cell_job in cell_jobs:
                    record = _prefill_context_key_record(
                        cell_job, model_id)
                    if record is not None:
                        out.prefill_context_key_provenance.append(record)
                    for endpoint_index, chunk in phase_chunk_points[phase].get(
                            point, ()):
                        attach_prefill_chunk_provenance(
                            cell_job, endpoint_index, (chunk,))
            out.unplaced.extend(cell_unplaced)
            out.jobs.extend(cell_jobs)
    out.unplaced.extend(_legacy_unplaced(sorted(legacy_dropped), LEGACY_DROP))
    return out


def _legacy_unplaced(notes: Iterable[str], reason: str) -> List[Dict[str, Any]]:
    """Migrate old textual drops to one structural record per reported drop.

    The two-value planner API could not distinguish a declined key from a
    missing signature or projection, so inventing one of those reasons here
    would be a second classifier. `legacy_drop` is therefore explicit: the
    original note is the detail, and the row remains in the sidecar/report
    denominator.
    """
    records: List[Dict[str, Any]] = []
    seen: Set[Tuple[Tuple[str, Any], ...]] = set()
    for note in notes:
        text = str(note)
        label, has_detail, detail = text.partition(": ")
        head = label if has_detail else text
        if head.endswith("]") and " [" in head:
            op, _, aggregation_id = head[:-1].partition(" [")
        else:
            op, aggregation_id = head, head
        op = op or "unknown"
        aggregation_id = aggregation_id or op
        record = {"op": op, "aggregation_id": aggregation_id,
                  "reason": reason,
                  "detail": detail if has_detail else text}
        key = tuple(sorted(record.items()))
        if key not in seen:
            seen.add(key)
            records.append(record)
    return records


#: What every deployment of the sweep being planned shares, set before the
#: workers fork so they inherit it: a spec does not pickle.
_SHARED: Dict[str, Any] = {}


def _plan_in_worker(task: Tuple[str, Any]) -> _DeploymentPlan:
    label, par = task
    return _plan_deployment(label, par, **_SHARED)


def plan_points(model_id: str, gpu: str, parallel: Any, phase: str,
                points: Sequence[Any], *, avg_output_tokens: int,
                framework: str = DEFAULT_FRAMEWORK, resolved: Any = None,
                ) -> Dict[Any, Tuple[List[Dict[str, Any]], List[str]]]:
    """`{point: (jobs, dropped)}` of one deployment and phase at `points`
    (`plan.PlanPoint`), as a sweep plans them — the deployed binding, the
    served pages, every model as shipped — but at points the caller names
    rather than the ones a surface's tracks and capacity choose. For a
    measured row whose point today's grid no longer reaches (a batch the
    capacity moved, a deployment the surface dropped)."""
    from .plan import build_jobs_over_points
    from .providers import served_kv_block_sizes
    if resolved is None:
        _config, resolved = resolve_sweep_model(model_id)
    return build_jobs_over_points(
        model_id=model_id, points=list(points), phase=phase,
        avg_output_tokens=avg_output_tokens, parallel=parallel,
        framework=framework, precision=AS_SHIPPED, resolved=resolved,
        binding=_deployed_binding(framework, gpu, AS_SHIPPED, resolved),
        gpu=gpu, kv_block_sizes=served_kv_block_sizes())


def sweep_surfaces() -> Dict[str, Dict[str, Any]]:
    """The surfaces a sweep plans (`sweep --tracks`): each registered track
    table with the contexts, deployments and output length it is asked over,
    all read from `sweep_grid`. `deployments` is `routine` (every safe mode
    at each degree the layout accepts) or `frontier` (every deployment the
    gallery's decode frontier prices on the GPU, `frontier_parallel_configs`)."""
    from backend.inference import sweep_grid as grid
    return {
        "routine": dict(tracks=grid.ROUTINE_TRACKS,
                        seq_lens=grid.ROUTINE_SEQ_LENS, deployments="routine",
                        output_tokens=grid.ROUTINE_OUTPUT_TOKENS),
        "frontier": dict(tracks=grid.FRONTIER_TRACKS,
                         seq_lens=grid.GALLERY_SEQ_LENS, deployments="frontier",
                         output_tokens=grid.GALLERY_OUTPUT_TOKENS),
    }


def plan_surface(
    model_id: str,
    gpu: str,
    surface: str,
    *,
    framework: str = DEFAULT_FRAMEWORK,
    max_seq: int = 0,
    output_tokens: Optional[int] = None,
    workers: int = 1,
    log=lambda m: None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """`(jobs, stats)` of one surface (`sweep_surfaces`) for one model on
    `gpu`, every model as shipped — the corpus the shipped estimates read.
    `max_seq` caps the context grid (0 = the surface's whole grid);
    `output_tokens` overrides the surface's served output length, and
    `stats["output_tokens"]` is the one planned with."""
    spec = sweep_surfaces()[surface]
    seq_lens = [s for s in spec["seq_lens"] if max_seq <= 0 or s <= max_seq]
    if output_tokens is None:
        output_tokens = spec["output_tokens"]
    config, resolved = resolve_sweep_model(model_id)
    parallel_configs = (
        frontier_parallel_configs(resolved, gpu)
        if spec["deployments"] == "frontier"
        else routine_parallel_configs(config, resolved=resolved))
    # The SAME grid the Scaling panel exports (build_sweep_jobs is what the
    # /inference/bench/sweep-shapes endpoint wraps) — one source, VRAM-gated.
    # Speculative verify widths are the tracks' own (`TrackSpec.draft_tokens`).
    jobs, stats = build_sweep_jobs(
        model_id=model_id, gpu=gpu, seq_lens=seq_lens,
        tracks=spec["tracks"], parallel_configs=parallel_configs,
        avg_output_tokens=output_tokens, resolved=resolved,
        framework=framework, precision=AS_SHIPPED, log=log, workers=workers)
    return jobs, {**stats, "output_tokens": output_tokens}


def build_sweep_jobs(
    *,
    model_id: str,
    gpu: str,
    tracks: Sequence[TrackSpec],
    parallel_configs: Sequence[Tuple[str, Any]],
    avg_output_tokens: int,
    seq_lens: Optional[List[int]] = None,
    framework: str = DEFAULT_FRAMEWORK,
    precision: Any = None,
    ep_load_factors: Optional[Sequence[Any]] = None,
    resolved: Any = None,
    log=lambda m: None,
    workers: int = 1,
    prefill_chunk_size: Optional[int] = DEFAULT_PREFILL_CHUNK_SIZE,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Expand one (model, gpu) into the deduped per-op job list over the
    feasible (parallel × seq × track × batch × verify-width) grid. Returns
    (jobs, stats), `stats["points"]` the served points planned, as
    `[parallel label, track, batch key, context, draft tokens]`.

    The surface's questions are the caller's, as data: `tracks` (one of the
    track tables `sweep_grid` registers),
    `parallel_configs` ((label, ParallelConfig) pairs — `routine_parallel_
    configs` or `preset_parallel_configs`), and `avg_output_tokens`, the
    served output length, which has no default (a decode cell's cache is the
    context plus half the output). Each (parallel, context, track) asks
    points of its phase's one cell (`plan.PlanPoint`):

      - no prefix cache: the context with nothing cached
        (`s_prefix_cached = 0`), at the track's batch points;
      - a prefix of N new tokens at a context L > N: the context with
        `s_prefix_cached = L − N` cached (`s_query = N`, `s_kv = L`);
      - a prefix of N at a context L ≤ N: nothing is cached, so the track
        reads the plain context's points — recorded in `stats["reused"]`.

    A decode track's speculative draft lengths (`TrackSpec.draft_tokens`)
    are verify widths, K + 1 each (`phases.Speculation`); a prefill has no
    draft to verify. Every (context, cached prefix, verify width, batch) of
    one (deployment, phase) is a point of one walk
    (`plan.build_jobs_over_points`).  A prefill endpoint is first decomposed
    into launch chunk points, and all of that deployment/phase's chunk points
    remain one walk.  A point a context with a cached prefix is not modelled
    on (`step_eval.prefix_refusal`) is an explicit dropped geometry, not a
    reason to fall back to an endpoint-sized cell.  A point the deployment
    does not serve (`capacity.point_served`) is not planned.

    `prefill_chunk_size` is explicit campaign policy: 8192 by default, and
    0/None explicitly unchunked.  It is a deployment/campaign knob, never a
    framework-binding fact.  The decomposition in
    `kernel_bench.prefill_chunks` is the only geometry code: its launch points
    are handed to `plan.build_jobs_over_points`.  A one-chunk query emits the
    endpoint point unchanged; a longer query changes its physical cells to
    `Sq<=c,Lq=end`.  Stats carry the endpoint-to-chunk map and exact-key
    counts, while H4 owns corpus-level campaign staging — this call assumes
    no full-corpus measurement rollout.

    `precision` is the precision profile every cell is gated and shaped
    under (`PrecisionProfile`; None is as shipped), so the shapes a
    contributor measures carry the identity the estimate asks for.
    `resolved` is the spec the caller already matched
    (`resolve_sweep_model`), or None to match it here.

    Deployments are independent (~7s each, 28 for a routine MoE), so
    `workers > 1` plans them in that many forked processes, each reading
    the spec this call holds. The jobs are merged in deployment order
    either way, so the list is the one a single process plans."""
    # Validate before spec matching or worker forks: a malformed campaign knob
    # must not masquerade as a model or capacity failure.
    chunk_size = normalize_prefill_chunk_size(prefill_chunk_size)
    if precision is None:
        precision = AS_SHIPPED
    if ep_load_factors is not None:
        tracks = tuple(replace(track, ep_load_factors=tuple(ep_load_factors))
                       for track in tracks)
    # Resolve the spec + binding ONCE — they're the same for every cell, and
    # SpecLoader.match is ~900ms with no cross-instance cache. Threading these
    # into build_jobs turns per-cell cost from ~1s (re-resolve) to ~0.14s
    # (analyze only). See project_sweep_perf_truth.
    if resolved is None:
        config, resolved = resolve_sweep_model(model_id)
    else:
        from backend.analyzer import fetch_config
        config = fetch_config(model_id)
    seq_lens = list(seq_lens or SWEEP_SEQ_LENS)
    parallels = list(parallel_configs)
    shared = dict(model_id=model_id, gpu=gpu, tracks=tracks,
                  seq_lens=seq_lens, avg_output_tokens=avg_output_tokens,
                  framework=framework, precision=precision, config=config,
                  resolved=resolved,
                  binding=_deployed_binding(framework, gpu, precision,
                                            resolved),
                  prefill_chunk_size=chunk_size)

    jobs: List[Dict[str, Any]] = []
    seen = set()
    cells = 0
    gate_errors = 0
    reused: List[str] = []
    points: List[List[Any]] = []
    chunk_map: List[Dict[str, Any]] = []
    prefill_context_key_provenance: List[Dict[str, Any]] = []
    chunk_terms = 0
    chunk_endpoints = 0
    dropped: set = set()
    unplaced: List[Dict[str, Any]] = []
    seen_unplaced = set()

    def merge(plans) -> None:
        nonlocal cells, gate_errors, chunk_map, chunk_terms, chunk_endpoints
        for plan in plans:
            for line in plan.logs:
                log(line)
            cells += plan.cells
            gate_errors += plan.gate_error
            reused.extend(plan.reused)
            points.extend(plan.points)
            chunk_map.extend(plan.chunk_map)
            prefill_context_key_provenance.extend(
                plan.prefill_context_key_provenance)
            chunk_terms += plan.chunk_terms
            chunk_endpoints += plan.chunk_endpoints
            dropped.update(plan.dropped)
            for record in plan.unplaced:
                key = tuple(sorted(record.items()))
                if key not in seen_unplaced:
                    seen_unplaced.add(key)
                    unplaced.append(record)
            for j in plan.jobs:
                # Kernel parameters are part of what makes a job
                # distinct (a clamp bound, a token distribution). The
                # paged signatures are expanded over the block sizes
                # production runs, and the page is an axis of their
                # cell, so it is in `dims`; reading it anywhere else
                # once made every block collapse to one key, so the
                # expansion was silently undone and only the first
                # block survived (16 — which FlashInfer's MLA backend
                # declines, so MLA decode measured nothing at all).
                # The deployment is not part of it: one card
                # runs this kernel at this shape the same way
                # whichever parallel config cut the shape
                # (`PhysicalKey`). The first deployment that
                # reaches a cell keeps it as provenance.
                k = (j["signature"],
                     json.dumps(j["dims"], sort_keys=True),
                     json.dumps(j["dtype_env"], sort_keys=True),
                     j["phase"],
                     json.dumps(j.get("kernel_params") or {},
                                sort_keys=True))
                if k in seen:
                    continue
                seen.add(k)
                jobs.append(j)

    if workers > 1 and len(parallels) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        _SHARED.update(shared)
        try:
            with ProcessPoolExecutor(
                    max_workers=min(workers, len(parallels)),
                    mp_context=multiprocessing.get_context("fork")) as pool:
                # `map` yields in submission order: deployment order.
                merge(pool.map(_plan_in_worker, parallels))
        finally:
            _SHARED.clear()
    else:
        merge(_plan_deployment(label, par, **shared)
              for label, par in parallels)
    prefill_keys = {_job_physical_key(j) for j in jobs
                    if j["phase"] == "prefill"}
    # Facets are collected in each deployment plan before the global physical
    # key dedup above.  Identical provenance requests are deduplicated only as
    # records; distinct models/pages/dtypes/variants survive beside their raw
    # physical keys for a corpus-level H4 union.
    context_key_records: List[Dict[str, Any]] = []
    seen_records: Set[str] = set()
    for record in prefill_context_key_provenance:
        token = _facet_order(record)
        if token in seen_records:
            continue
        seen_records.add(token)
        context_key_records.append(record)
    context_key_records.sort(key=_facet_order)
    context_keys = {tuple(record["physical_key"])
                    for record in context_key_records}
    context_keys_by_signature = {
        signature: sum(key[0] == signature for key in context_keys)
        for signature in sorted({key[0] for key in context_keys})
    }
    # H4 first unions ``physical_key`` as an exact set; the named facets are
    # pre-dedup provenance for campaign staging.  Scalar counts are reporting
    # only and must never become the rollout denominator.
    context_key_manifest = context_key_records
    context_key_manifest_by_signature = {
        signature: [record for record in context_key_records
                    if record["signature"] == signature]
        for signature in sorted({record["signature"]
                                 for record in context_key_records})
    }
    mode = "chunked" if chunk_size else "unchunked"
    stats = {"cells": cells, "unique_jobs": len(jobs),
             "parallel_configs": len(parallels), "seqs": len(seq_lens),
             "tracks": [t.name for t in tracks],
             "gate_errors": gate_errors, "dropped": sorted(dropped),
             "unplaced": unplaced,
             "reused": reused, "points": points,
             "prefill_chunk_size": chunk_size or 0,
             "prefill_chunk_jobs": f"{mode}-batch-h2",
             "prefill_chunk_campaign": f"h2-{mode}-jobs/c={chunk_size or 0}",
             "prefill_chunk_endpoints": chunk_endpoints,
             "prefill_chunk_terms": chunk_terms,
             "prefill_chunk_map": chunk_map,
             "distinct_prefill_chunk_keys": len(prefill_keys),
             "distinct_prefill_context_keys": len(context_keys),
             "distinct_prefill_context_keys_by_signature":
                 context_keys_by_signature,
             "distinct_prefill_context_key_manifest": context_key_manifest,
             "distinct_prefill_context_key_manifest_by_signature":
                 context_key_manifest_by_signature,
             "prefill_chunk_rollout": {
                 "stage": "h2-plan-rekey",
                 "mode": mode,
                 "measurement_rollout": "staged-corpus-campaign-pending-h4",
                 "full_corpus_assumed": False,
                 "exact_key_only": True,
                 "missing_context_key_is_incomplete": True,
             }}
    return jobs, stats
