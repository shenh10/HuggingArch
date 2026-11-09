"""Per-step compute + memory traffic, split by prefill / decode phase.

A phase is a view of one evaluated step and that step's per-op rows.
There is no closed-form substitute: callers without a graph raise.

Phase bindings (used in the spec-eval path):

    Prefill (no cache, no prefix):
        s_query = avg_input_tokens
        s_kv = s_query
        s_pairs = s_query · (s_query + 1) / 2   (the causal triangle)

    Decode (autoregressive, single new token):
        s_query = 1
        s_kv = avg_input_tokens + avg_output_tokens / 2   (avg cache len)
        s_pairs = s_kv   (single query, no triangle to skip)

    Decode (speculative, K draft tokens — `Speculation`):
        s_query = s_verify = K + 1   (the last accepted token + the K drafts)

The memory side is the same fold. A step's HBM traffic is Σ over the
kernels it runs of what each kernel reads: the leaves ARE the answer —
post-fusion, which is how a flash kernel's saved scratch shows up without
anyone modelling scratch. Resident-only `weight_bytes + kv_bytes_read`
never sees an activation written by one kernel and read back by the next.
"""
from __future__ import annotations


from backend.arch.eval.analysis import Point, is_count
from backend.arch.eval.dim import as_number, exact
from backend.arch.eval.launch import FusedNode
from backend.arch.eval.refusal import node_context
from backend.arch.layout.stages import StagePlacement
from backend.arch.layout.materialize_comm import payload_substituted
import sympy as sp
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from backend.arch.ops import OpSpec
from backend.arch.ops import weight_shape_of
from backend.arch.dtypes import dtype_bytes, peak_kind

from .comm_cost import Fabric, comm_display_shape, topology
from .gpu_specs import NoPeakForDtype
from .intensity import arithmetic_intensity
from .module_rows import instance_count
from .row_roofline import (LawBook, RowCard, RowRoofline, RowTimes, StepSum,
                           WeightDemand, build_row_exprs, comm_factor,
                           bare_row, law_columns, point_columns, weight_demand)
from .phase_formulas import (
    FormulaDefs,
    _access_bytes_formula_for_row,
    _flops_formula_for_row,
    access_display_facts,
    phase_roofline_formulas,
)
from .step_eval import evaluate_step
from backend.measure.auto_sweep import MeasuredTrack


@dataclass
class PhaseMetrics:
    phase: str
    batch: int
    seq_effective: float       # tokens touched per forward (prefill: S_in; decode: 1)
    kv_cache_length: float     # average cache length used in this phase
    active_params_per_rank: float
    #: Executed FLOPs: Σ the served graph's per-op rows — activation casts,
    #: fused launches and collectives in it — per rank. Not the model's
    #: FLOPs (`flops_fold`, the logical walk): a phase-level MFU must take
    #: its numerator from there, materialised as its own field, never
    #: divide this one.
    flops_per_step: float
    mem_read_bytes: float
    mem_write_bytes: float
    total_bytes: float
    #: FLOPs / step bytes (`intensity.arithmetic_intensity`); None when the
    #: step moves no bytes.
    arithmetic_intensity: Optional[float]
    note: str
    formulas: dict
    # Per-module breakdown of the served graph (`module_rows`, walked off
    # the grafted + fused tree; the orchestrator sets it — a phase does not
    # walk a table of its own). The frontend renders it as a collapsible
    # per-module table under the prefill / decode dashboard.
    flops_by_module: List[Dict[str, Any]] = field(default_factory=list)
    # Phase-specific dim binding actually used (s_query, s_kv, s_pairs).
    # Surfaced for tooltip display + downstream debug.
    binding: Dict[str, Any] = field(default_factory=dict)
    # Per-step rooflines under the chosen GPU's peak compute / HBM bw /
    # interconnect bw, and the request's link latency and launch floor. When
    # the graph ran (`graph_rows=`), these are *sums of per-leaf rooflines*:
    # each leaf contributes `max(t_compute, t_memory, t_comm, t_launch)`
    # (`row_roofline.BOUNDS`), and the totals here are the component sums —
    # not addends of `t_sol_s` — and their max-sum (`t_sol_s`; each leaf's
    # launch floor is on its row). This matches what the per-module breakdown table sums in its
    # Total row — the two numbers MUST agree, and they are the same fold.
    # `None` where the phase declines to publish a time — see
    # `time_unpublished_reason`. A time is either the step's or it is absent;
    # there is no partial one, because every consumer of these fields divides
    # tokens by them and calls the quotient a rate.
    t_compute_s: Optional[float] = 0.0
    t_memory_s: Optional[float] = 0.0
    t_comm_s: Optional[float] = 0.0
    t_sol_s: Optional[float] = 0.0
    #: `t_sol_s` split by the term that set each leaf's time: every leaf's
    #: `t_op` counted once, under compute, memory, comm or launch, so the four add
    #: up to `t_sol_s` (`RowRoofline.by_bound`). `None` with the times.
    t_sol_by_bound: Optional[Dict[str, float]] = None
    #: Each part of `t_sol_by_bound` as a fraction of `t_sol_s`
    #: (`RowRoofline.shares`). `None` with the parts.
    t_sol_by_bound_share: Optional[Dict[str, float]] = None
    #: The measured track beside the theoretical one
    #: (`auto_sweep.MeasuredTrack`): the one measured formula's total, only
    #: when every row has a measured time — never the roofline in for a row
    #: Kernel Bench has not measured. None otherwise; `coverage` says what is
    #: there. Stated by the measured fold (`batch_sweep.apply_measured`).
    t_measured_s: Optional[float] = None
    #: How many of the step's per-op rows carry a measured time, of how many.
    measured_rows: int = 0
    total_rows: int = 0
    #: The rows' coverage (`MeasuredTrack.coverage`): `{measured, total,
    #: states}`, the keys or ops in each state that is not `measured`.
    coverage: Dict[str, Any] = field(default_factory=dict)
    #: Why there is no step time, or "" when there is one. Set at `pp > 1`
    #: when the pipeline-step time could not be summed (no `pipeline_time`
    #: from the overlay): a stage's step is not the model's, and publishing
    #: it under a name every consumer reads as TTFT / TPOT / throughput
    #: would divide the true latency by `pp`.
    time_unpublished_reason: str = ""
    #: pp>1: the sums the times above were published from — Σ per-stage
    #: roofline + inter-stage hops, bubble unmodelled — and every stage's
    #: row laws (`stage_laws`), which the batch sweep re-prices at another
    #: batch or card and sums the same way. None at pp=1.
    pipeline_time: Optional[Dict[str, Any]] = None
    #: The part of `note` the batch does not change (`phase_note`): a phase
    #: re-read at another batch writes its note from it.
    note_detail: str = ""
    # NOTE: `per_op_timing` is intentionally NOT a field of this
    # dataclass. Adding it with a default `[]` would make
    # `vars(estimate_phase(...))` always serialize empty
    # `phases.{prefill,decode}.per_op_timing: []` lists — even in
    # `vars(estimate_phase(...))` from growing an empty per-op list.
    # The real per_op_timing flows through `compute_per_op_timing` and
    # is attached at the TOP level by `estimate_phases`.


@dataclass(frozen=True)
class Speculation:
    """Speculative decoding as a request states it: K draft tokens per step,
    accepted at rate α.

    The target verifies the drafts in one forward over `s_verify = K + 1`
    positions — the last accepted token plus the K drafts — and keeps
    `1 + α·K` tokens per sequence per step on average (α is the mean
    fraction of the drafts accepted; the verify's own token is always kept).
    K = 0 is plain autoregressive decode: one position, one token.
    """

    draft_tokens: int = 0
    acceptance_rate: float = 1.0

    def __post_init__(self) -> None:
        if int(self.draft_tokens) < 0:
            raise ValueError(
                f"speculative draft tokens must be >= 0, got "
                f"{self.draft_tokens}")
        object.__setattr__(self, "draft_tokens", int(self.draft_tokens))

    @classmethod
    def verifying(cls, s_verify: int,
                  acceptance_rate: float = 1.0) -> "Speculation":
        """The speculation whose verify forward is `s_verify` wide."""
        return cls(max(1, int(s_verify or 1)) - 1, acceptance_rate)

    @property
    def s_verify(self) -> int:
        """The verify forward's width: K + 1 positions."""
        return self.draft_tokens + 1

    @property
    def accepted_tokens(self) -> float:
        """Tokens one sequence keeps per step: 1 + α·K."""
        return 1.0 + self.acceptance_rate * self.draft_tokens


def phase_note(phase: str, s_query: int, batch: int, detail: str) -> str:
    """The phase's one-line note: its fresh tokens, its batch and `detail`
    (`PhaseMetrics.note_detail`, the part the batch does not change), so a
    phase re-read at another batch (`batch_sweep`) states it there."""
    if phase == "prefill":
        return (f"Compute-bound: forward processes {s_query} fresh tokens × "
                f"{batch} req{detail}")
    return f"Memory-bound: {s_query} fresh tok/step × {batch} req{detail}"


def step_tokens(phase: str, batch: int, tokens_per_step: float,
                speculation: Speculation) -> Tuple[float, Dict[str, str]]:
    """The tokens one step of `batch` sequences produces, and how: `batch ×`
    the tokens one sequence produces — its uncached input (`s_query`) in
    prefill, one in decode, `1 + α·K` kept when decode verifies K drafts.
    Every throughput reads this value and shows this derivation."""
    if phase == "decode" and speculation.draft_tokens > 0:
        alpha, k = speculation.acceptance_rate, speculation.draft_tokens
        return speculation.accepted_tokens * batch, {
            "symbolic": "B × (1 + α·K)",
            "substituted": f"{batch} × (1 + {alpha:g}·{k})",
        }
    per_seq = "s_query" if phase == "prefill" else "1"
    return float(tokens_per_step) * batch, {
        "symbolic": f"B × {per_seq}",
        "substituted": f"{batch} × {tokens_per_step:g}",
    }


def phase_binding(
    *,
    phase: str,
    avg_input_tokens: int,
    avg_output_tokens: int,
    s_prefix_cached: int = 0,
    s_verify: int = 1,
) -> Dict[str, Any]:
    """Resolve the (s_query, s_kv, s_pairs, s_cache, tokens_per_step)
    binding for one phase.

    The three axes come from `eval.transforms.Phase.resolve` — the same
    arithmetic the evaluator walk binds. Extra keys (`cache_len`,
    `tokens_per_step`, the phase flag) are this module's. Four workload shapes:

    Prefill (no cache, no prefix):
        s_query = input_tokens, s_kv = input_tokens
        s_pairs = s_query·(s_query+1)/2
        tokens_per_step = input_tokens
    Prefill (prefix-cached):
        s_query = input_tokens − prefix_tokens
        s_kv = input_tokens
        s_pairs = s_query·prefix_tokens + s_query·(s_query+1)/2
            (every uncached query attends to all prefix_tokens
             unconditionally, plus the lower triangle of the s_query×s_query
             tail block)
        tokens_per_step = s_query
    Decode AR (n_verify=1):
        s_query = 1, s_kv = s_cache, s_pairs = s_kv
        tokens_per_step = 1
    Decode speculative (n_verify>1):
        s_query = n_verify
        s_kv = s_cache + n_verify − 1
        s_pairs = n_verify·(s_cache − 1) + n_verify·(n_verify+1)/2
            (query i sees s_cache + i keys: the intra-window block has a
             causal saving, cross-cache attention is unrestricted)
        tokens_per_step = α-amortized in the caller (alpha is applied
            outside this helper since it scales tok/s, not per-step
            FLOPs).
    """
    from backend.arch.eval.dim import value_of
    from backend.arch.eval.transforms import Phase, Workload

    if phase not in ("prefill", "decode"):
        raise ValueError(f"Unknown phase: {phase}")
    resolved = Phase(kind=phase).resolve(Workload(
        input_tokens=avg_input_tokens,
        output_tokens=avg_output_tokens,
        s_prefix_cached=s_prefix_cached,
        s_verify=s_verify,
    ))
    # The walk binds Dim identities (`n_verify` on decode). This dict is
    # the numeric binding the serving layer and the JSON wire use.
    sq = value_of(resolved["s_query"])
    return {
        "phase": phase,
        "s_query": sq,
        "s_kv": value_of(resolved["s_kv"]),
        "s_pairs": resolved["s_pairs"],
        "cache_len": (
            0 if phase == "prefill"
            else (avg_input_tokens + avg_output_tokens / 2.0)
        ),
        "tokens_per_step": sq,
        "s_prefix_cached": s_prefix_cached if phase == "prefill" else 0,
        "s_verify": s_verify if phase == "decode" else 1,
    }


def _rows_total(rows: List[Dict[str, Any]]) -> float:
    """The logical walk's FLOPs (the model's) as the sum of its depth-0 rows.
    Not a phase's FLOPs since the casts became nodes: those are the served
    rows' sum (`estimate_phase`). What follows is the history of the two
    walks this replaced.

    Every depth-0 row is either a block template charged its layers or the
    model-level tail, and nothing else is at depth 0, so this is the whole
    tree counted once. Reading it off the rows ties this phase's headline to
    the breakdown underneath it. Pipeline-stage multiplicity has already been
    folded into those same rows, so no second aggregate is re-counted.

    The per-op rows from `pipeline.analyze` fold to the same total, and the
    breakdown table under this headline displays them, so this number and
    that one have to be the same number. They are, and since both are folds
    of the step's one
    `StepEvaluation` the only way they can differ at all is arithmetic: an
    expert leaf's per-expert FLOPs is a rational (Kimi-K3's `act` and
    `gate_mul` carry .429 and .857), which this fold keeps exact in the block
    total while the per-op row truncates to an integer before scaling by the
    active-expert count. Bounded by `n_experts_active × the fractional
    leaves`, ~1.6e3 per layer there; measured worst case across the corpus at
    8192 prefill tokens and at decode with `s_verify` 1 and 4 is 4.8e-7
    relative, on the MoE decodes (Kimi-K2/K3, DeepSeek-V2, MiMo-V2.5).

    That used to be a claim about two walks agreeing, and four differences of
    CONTENTS had to be found and closed by hand before it was true — each
    invisible in the totals because only one of the two walks moved:

      * the LM head runs where the sampler reads. This walk used to bill it at
        every prefill position (8191 × its per-position FLOPs on a 8192-token
        step) while the per-op rows billed the sampled one;
      * aux stacks (MTP, vision) are excluded by Program stack semantics;
      * a template instantiated zero times costs zero. `max(count, 1)` here
        billed one layer of each of GLM-5.1's two empty stacks;
      * the tail is every model-level node, not just the head — the final
        norm was in the per-op sum and not in this one.

    None of those can recur in that form now: there is one evaluation and both
    numbers fold it, so a fifth difference of contents would have to be one
    fold dropping rows the other keeps, not two walks pricing the same rows
    differently. `invariants.flops_closure` still asserts the agreement at
    1e-5 on every estimate — cheap, and it is the thing that would notice.
    """
    return float(sum(r["flops_full"] for r in rows if r["depth"] == 0))


def _flops_rows(
    step_eval: Any,
    layer_multiplicity: Any = None,
) -> List[Dict[str, Any]]:
    """Fold one Analysis directly into the logical walk's FLOPs rows.

    Semantic and physical values are read by UID in the same traversal.  The
    external pipeline partition is resolved once to ``Layer[int]`` by
    ``Analysis.layer_multiplicity``; no model DTO is recounted and no row is
    joined back by display identity.

    The model's FLOPs, not a phase's: the phase sums its served rows
    (`estimate_phase`), which carry the grafted casts this walk has not.
    """
    from backend.arch.eval.dim import as_extent, value_of
    from backend.arch.eval.program import BlockStage, OpStage
    analysis = step_eval.analysis
    multiplicity = layer_multiplicity or analysis.layer_multiplicity()
    if multiplicity.program is not analysis.program:
        raise ValueError("layer multiplicity belongs to another Program")
    physical = step_eval.physical_flops() if step_eval.has_deployment else None
    rows: List[Dict[str, Any]] = []

    def shape_in(shape: Any) -> Any:
        if shape is None or not shape.ins:
            return None
        values = [[as_extent(axis, rounding="floor") for axis in axes]
                  for axes in shape.ins]
        return values[0] if len(values) == 1 else values

    def shape_out(shape: Any) -> Any:
        if shape is None or shape.out is None:
            return None
        return [as_extent(axis, rounding="floor") for axis in shape.out]

    def visit(node: Any, depth: int, count: int, template: str) -> None:
        with node_context(node.display_path):
            cost = analysis.cost.values.get(node.uid)
            full = round(float(value_of(cost.flops if cost is not None else 0)) * count)
            local = 0
            if physical is not None:
                local_v = physical.values.get(node.uid)
                if local_v is None:
                    raise ValueError(
                        f"physical FLOPs have no value for {node.display_path}"
                    )
                local = round(float(value_of(local_v)) * count)
            shape = analysis.shapes.values.get(node.uid)
            row = {
                "name": node.name,
                "op": node.op,
                "role": node.role,
                "aggregation_id": node.aggregation_id,
                "template_name": template,
                "layer_count": count,
                "depth": depth,
                "shape_in": shape_in(shape),
                "shape_out": shape_out(shape),
                "flops_full": full,
            }
            if physical is not None:
                row["flops_per_rank"] = local
            rows.append(row)
        for child in node.children or ():
            visit(child, depth + 1, count, template)

    included = analysis.program.included_stack_names()
    for stack in analysis.program.stacks:
        if stack.name not in included:
            continue
        for stage in stack.stages:
            if isinstance(stage, BlockStage):
                roots = tuple(stage.template.roots)
                count = (
                    multiplicity.values[roots[0].uid] if roots else 0
                )
                full = sum(
                    value_of(analysis.cost.values[root.uid].flops)
                    for root in roots
                )
                full_i = round(float(full) * count)
                local_i = 0
                if physical is not None:
                    local = sum(value_of(physical.values[root.uid]) for root in roots)
                    local_i = round(float(local) * count)
                row = {
                    "name": stage.template.name,
                    "op": "<template>",
                    "role": "block_template",
                    "aggregation_id": "",
                    "template_name": stage.template.name,
                    "template_label": analysis.program.block_label(stage.template),
                    "layer_count": count,
                    "depth": 0,
                    "flops_full": full_i,
                }
                if physical is not None:
                    row["flops_per_rank"] = local_i
                rows.append(row)
                for root in roots:
                    visit(root, 1, count, stage.template.name)
            elif isinstance(stage, OpStage):
                node = stage.node
                cost = analysis.cost.values.get(node.uid)
                shape = analysis.shapes.values.get(node.uid)
                if (
                    (cost is None or (not cost.flops and not cost.params))
                    and (shape is None or not shape.ins)
                ):
                    continue
                visit(node, 0, multiplicity.values[node.uid], "")
    return rows


def evaluate_phase_step(
    resolved_spec: Any,
    parallel: Optional[Any] = None,
    *,
    phase: str,
    batch: int = 1,
    avg_input_tokens: int,
    avg_output_tokens: int,
    s_prefix_cached: int = 0,
    s_verify: int = 1,
) -> tuple:
    """Evaluate the step a phase describes, and return it with its binding.

    The pairing is the point: `phase_binding` resolves the workload to
    (s_query, s_kv, s_pairs) and the evaluator is given those same
    three, so the step and the note printed beside it cannot describe
    different dims.
    """
    bind = phase_binding(
        phase=phase, avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        s_prefix_cached=s_prefix_cached, s_verify=s_verify,
    )
    step_eval = evaluate_step(
        resolved_spec, parallel, phase=phase, batch=max(1, int(batch)),
        input_tokens=avg_input_tokens, output_tokens=avg_output_tokens,
        s_prefix_cached=s_prefix_cached, s_verify=s_verify,
    )
    return step_eval, bind


@dataclass(frozen=True)
class FlopsFold:
    """One evaluated step's compute as the logical walk states it: the
    model's FLOPs table and the active params.

    Not a phase's FLOPs. A phase's are the sum of its served per-op rows
    (`estimate_phase`): the graph serving runs, with the collectives, the
    activation casts and the fused launches grafted in. The logical walk
    has none of those, so its depth-0 sum is the model's FLOPs; the two
    differ by exactly the grafted casts' FLOPs, a law the closure tests
    hold rather than an equality either side may assume."""

    rows: List[Dict[str, Any]]
    active_params: float


#: What a phase's FLOPs are (`PhaseMetrics.flops_per_step`), as its FLOPs
#: formula's substituted line names it.
PER_RANK_DERIVATION = (
    "per-rank = Σ serving-graph per-op rows "
    "(incl. activation casts, fused launches)")


def _step_multiplicity(step_eval: Any, stage: Optional[int]) -> Any:
    """How many times each node of the step runs on this rank: the whole
    step, or what one pipeline stage runs (`stages.StagePlacement.view`)."""
    analysis = step_eval.analysis
    if stage is None:
        return analysis.layer_multiplicity()
    return StagePlacement.of(analysis).view(stage).rows(
        analysis.program.included_stack_names(include_aux=True))


def active_params(step_eval: Any, *, stage: Optional[int] = None) -> float:
    """The parameters one token touches on this rank (or this stage's)."""
    multiplicity = _step_multiplicity(step_eval, stage)
    if not step_eval.has_deployment:
        return float(step_eval.analysis.total_for(multiplicity).active_params)

    from backend.arch.layout.persistent import (
        active_parameter_elements,
        artifacts_for_layers,
        physical_artifacts,
    )

    artifacts = physical_artifacts(step_eval.analysis)
    if stage is not None:
        artifacts = artifacts_for_layers(
            step_eval.analysis, artifacts, multiplicity
        )
    return float(active_parameter_elements(step_eval.analysis, artifacts))


def flops_fold(step_eval: Any, *, stage: Optional[int] = None) -> FlopsFold:
    """Fold a step into the logical walk's FLOPs table and its active
    params: the model's FLOPs, per module, for a caller that wants them —
    the closure tests' independent route to a phase's FLOPs, the per-rank
    column. `stage` restricts it to what one pipeline stage runs."""
    rows = _flops_rows(step_eval, _step_multiplicity(step_eval, stage))
    return FlopsFold(rows=rows, active_params=active_params(step_eval, stage=stage))


@dataclass(frozen=True)
class PhaseInputs:
    """Everything a phase is priced against that does NOT differ between the
    two phases: the workload, the per-rank memory account, the deployment,
    the machine's peaks and the KV read law.

    `estimate_phases` used to declare these fields, take them apart into a
    `kw` dict and hand the same dict to `estimate_phase` twice, which is what
    an object looks like before it is given a name. What genuinely differs
    per phase stays an argument — the batch, the step evaluation and the
    per-op rows — because prefill and decode ARE different steps.

    Four fields the old signature carried are not here. `weight_dtype_bytes`
    and `kv_dtype_bytes` reached `estimate_phase` from every caller and were
    read by none of it — the widths that matter are already inside the rows
    and the per-rank byte totals. `peak_flops_per_sec` and
    `peak_hbm_bw_bytes_per_sec` likewise: the peaks are spent upstream, where
    `compute_per_op_timing` turns them into each row's t_compute and
    t_memory, and a phase folds those rows rather than re-dividing by a peak.
    Nothing read them here, so they are not part of what a phase is priced
    against.
    """

    #: The workload.
    avg_input_tokens: int
    avg_output_tokens: int
    #: The per-rank memory account this deployment resolved.
    weight_bytes_per_rank: float
    per_token_kv_bytes_per_rank: float
    static_kv_bytes_per_rank: float
    #: The model and the mesh. Optional: a caller that passes neither gets the
    #: closed-form path.
    resolved_spec: Optional[Any] = None
    parallel: Optional[Any] = None
    #: Workload knobs.
    s_prefix_cached: int = 0
    s_verify: int = 1
    acceptance_rate: float = 1.0
    #: Sparse-attention read cardinality (a sympy Expr in `s_kv`, None for
    #: dense) and the per-query KV read law it came from.
    kv_read_expr: Optional[Any] = None
    kv_reader: Optional[Any] = None
    #: Capacity's explanation of why no batch fit, shown on a phase priced at
    #: batch 0.
    infeasibility_reason: Optional[str] = None
    #: The pipeline stage this rank runs (`stages.StagePlacement.view` of
    #: the step's program); None at pp=1.
    stage: Optional[int] = None



def phase_frame(inputs: PhaseInputs, *, phase: str, cache_len: float,
                draft_spec: Optional[Any], draft_forward_flops: Optional[float],
                ) -> Tuple[str, Dict[str, str]]:
    """What a phase says about the request beside its numbers: the note's
    detail (a cached prefix and the part of it a sparse read touches; the
    cache a decode step attends to, by attention class) and the FLOPs
    formula's lines. Facts of the request at its point — its prompt, prefix,
    cache and verify width — and never of the batch, so a sweep's point
    states them as a walk there does (`batch_sweep.phase_at`). A counted
    speculative draft (`draft_spec`, `speculative_draft.SpeculativeDraft`)
    joins the decode note, and one forward's FLOPs its formula."""
    s_prefix_cached = inputs.s_prefix_cached
    s_verify = inputs.s_verify
    acceptance_rate = inputs.acceptance_rate
    speculation = Speculation.verifying(s_verify, acceptance_rate)
    kv_read_expr = inputs.kv_read_expr
    kv_reader = inputs.kv_reader

    def _kv_read_positions_legacy(at_cache_len: float) -> int:
        from backend.arch.eval.cache import positions_at
        return positions_at(kv_read_expr, max(0, round(at_cache_len)))

    if phase == "prefill":
        # Sparse-attention (SWA/DSA) shrinks the effective per-query
        # touched footprint of a cached prefix to
        # `kv_read_positions(s_prefix_cached)`, so a 64k prefix under
        # top-K=2048 only loads 2048 positions per query instead of all
        # 64k.
        prefix_kv_positions = _kv_read_positions_legacy(s_prefix_cached)
        sparse_note = (
            f" (sparse: {prefix_kv_positions}/{s_prefix_cached} pos/query)"
            if (s_prefix_cached > 0 and prefix_kv_positions < s_prefix_cached)
            else ""
        )
        prefix_note = (
            f", with {s_prefix_cached} cached prefix tok/req{sparse_note}"
            if s_prefix_cached > 0 else ""
        )
        note_detail = f"{prefix_note}."
        flops_formula = {
            "symbolic": ("engine: Σ serving-graph per-op rows at "
                         "(s_query, s_kv, s_pairs)"),
            "substituted": (PER_RANK_DERIVATION
                            + (f" · prefix={s_prefix_cached}"
                               if s_prefix_cached > 0 else "")),
        }
        return note_detail, flops_formula
    # Decode: the hybrid class terms or the `[sparse: x/y pos/query]`
    # summary of the cache the step reads.
    decode_breakdown = (kv_reader.read_bytes_at(float(cache_len))[1]
                        if kv_reader is not None else [])
    kv_read_positions = _kv_read_positions_legacy(cache_len)
    spec_note = (
        f" (K={speculation.draft_tokens} draft tokens, "
        f"n_verify={s_verify}, α={acceptance_rate:.2f})"
        if speculation.draft_tokens > 0 else ""
    )
    if decode_breakdown:
        class_terms = " + ".join(
            f"{c.get('label') or c.get('component')}×{c['layer_count']}@{c['read_positions']}"
            for c in decode_breakdown
        )
        sparse_note = f" [hybrid: {class_terms} pos/query]"
    else:
        sparse_note = (
            f" [sparse: {kv_read_positions}/{int(cache_len)} pos/query]"
            if kv_read_positions < round(cache_len) else ""
        )
    note_detail = (f", attending to ~{cache_len:.0f} cached tok/req"
                   f"{spec_note}{sparse_note}.")
    flops_formula = {
        "symbolic": "engine: Σ serving-graph per-op rows at "
                    "(s_query=n_verify, s_kv=s_cache+n_verify-1, s_pairs)",
        "substituted": PER_RANK_DERIVATION,
    }
    if draft_spec is not None:
        d = draft_spec
        note_detail += (f" Draft counted: {d.style.value}, {d.forwards} MTP "
                        f"forward(s) × {d.tokens_per_forward} tok/seq.")
        flops_formula = {
            "symbolic": flops_formula["symbolic"]
                        + " + draft forwards × Σ MTP-stack ops",
            "substituted": (
                f"{PER_RANK_DERIVATION} + {d.forwards} × "
                f"{draft_forward_flops:.3e}"),
        }
    return note_detail, flops_formula


def estimate_phase(
    inputs: PhaseInputs,
    *,
    phase: str,
    batch: int,
    # The step this phase is a view of, evaluated once by the orchestrator and
    # shared with the per-op path (`step_eval.StepEvaluation`).
    step_eval: Optional[Any] = None,
    # The per-op rows for that step, from `pipeline.analyze` through the
    # calibration overlay. When given they carry the phase's traffic and time.
    graph_rows: Optional[List[Dict[str, Any]]] = None,
    # The speculative draft's forwards, priced off the MTP stack
    # (`speculative_draft.DraftCost`). Decode only; None leaves the step the
    # verify forward alone.
    draft: Optional[Any] = None,
    # pp>1: the pipeline-step time (Σ per-stage roofline + inter-stage
    # hops, bubble unmodelled). When given, published instead of withheld.
    pipeline_time: Optional[Dict[str, Any]] = None,
) -> PhaseMetrics:
    """One phase of one step, priced. `inputs` is the deployment half of the
    question; `phase`, `batch`, `step_eval`, `graph_rows` and `draft` are
    this phase's."""
    if draft is not None and phase != "decode":
        raise ValueError("a speculative draft is part of a decode step only")
    # One local name per field, so the body below reads as it did when
    # these were parameters.
    avg_input_tokens = inputs.avg_input_tokens
    avg_output_tokens = inputs.avg_output_tokens
    resolved_spec = inputs.resolved_spec
    parallel = inputs.parallel
    s_prefix_cached = inputs.s_prefix_cached
    s_verify = inputs.s_verify
    acceptance_rate = inputs.acceptance_rate
    speculation = Speculation.verifying(s_verify, acceptance_rate)
    infeasibility_reason = inputs.infeasibility_reason
    stage = inputs.stage

    if not graph_rows:
        raise ValueError(
            "estimate_phase needs the step's per-op rows: pass `graph_rows=` "
            "from `pipeline.analyze` through the calibration overlay. Without "
            "them the only traffic available is what is RESIDENT — weights and "
            "cached KV, nothing a kernel produces — which is 5x low on a "
            "prefill step, and it would be published under the same field name "
            "as the fold."
        )
    # No capacity for a single request. What is withheld is the TIME, and only
    # the time: a DURATION of zero says the step is instantaneous, and every
    # consumer divides tokens by it — the disaggregation route read it as
    # `decode_time_per_request_s: 0.0` and a decode:prefill ratio of 0.0 for
    # DeepSeek-V3 on 8×H200, a config whose weights (626.92 GB/rank) do not fit
    # its 126.90 GB of usable HBM, with `unavailable_reason` empty beside it.
    #
    # The step's STRUCTURE survives, described at one request. Returning zeros
    # here instead was the wrong reading of "there is no step": `batch` is the
    # anchor these numbers were computed at, not the count the deployment
    # admits — that is `max_batch_per_gpu`, which is already 0 and says so.
    # Zeroing left the phase disagreeing with the per-op rows it is the fold of
    # (0 against 90 GB, in 38 of the 78 cells of the SOL surface) and made
    # DeepSeek-V3 untestable at every deployment in the corpus, since none of
    # them hold it. The shape walk already clamps to `max(1, batch)` for
    # exactly this reason, and prefill has always shown the B=1 step by
    # default; this is that same view, with the verdict attached to the time.
    #
    # The verdict is READ from the field that declares it, not recovered from
    # `batch <= 0`. Routing it through the number made `batch` carry two
    # meanings — the anchor the counts were computed at, and whether the
    # deployment admits a request — so capacity had to shrink the count to
    # signal the verdict and this function had to grow it back to describe the
    # step. `infeasibility_reason` is `max_batch == 0` already disambiguated at
    # its source (weights over budget vs. KV budget under one request), and it
    # catches the second case, which `max_prefill_batch` does not: a rank whose
    # weights fit but whose KV budget cannot hold one request has no prefill
    # step either, because the cache that step ends by writing has no room.
    # `batch` is now unconditionally the anchor, which is what it has always
    # meant on the prefill side.
    #
    # This is the WALK's phase. What a card that admits no request PUBLISHES
    # is not it: `capacity.without_step` drops the step entirely (no batch,
    # no rows, no table) at the publishing boundary. The walk keeps the
    # structure because every other card and batch is re-priced from it.
    no_capacity = bool((infeasibility_reason or "").strip())
    batch = max(1, int(batch))

    if step_eval is None:
        if resolved_spec is None:
            raise ValueError(
                "estimate_phase needs the step it describes: pass "
                "`step_eval=` (already evaluated) or `resolved_spec=` "
                "(evaluate here). Neither is a request for an estimate "
                "without a graph — there is no such estimate."
            )
        step_eval, bind = evaluate_phase_step(
            resolved_spec, parallel, phase=phase, batch=batch,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            s_prefix_cached=s_prefix_cached, s_verify=s_verify,
        )
    else:
        bind = phase_binding(
            phase=phase, avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            s_prefix_cached=s_prefix_cached, s_verify=s_verify,
        )
    Sq, Lq, pairs, cache_len = bind["s_query"], bind["s_kv"], bind["s_pairs"], bind["cache_len"]
    tokens_per_step = bind["tokens_per_step"]

    used_absorbed_mla = step_eval.mla_absorbed
    p_active = active_params(step_eval, stage=stage)

    note_detail, flops_formula = phase_frame(
        inputs, phase=phase, cache_len=cache_len,
        draft_spec=draft.draft if draft is not None else None,
        draft_forward_flops=(draft.per_forward["flops_per_step"]
                             if draft is not None else None))

    # The step's rows added once (`StepSum`): the verify forward's rows and,
    # under a counted draft, its forwards' rows — their traffic, and below
    # their time, join the verify forward's.
    step = StepSum.of(graph_rows,
                      draft_rows=draft.rows if draft is not None else (),
                      draft_forwards=draft.draft.forwards if draft is not None else 0)
    # The FLOPs come off the same rows as the bytes: the executed FLOPs of
    # the graph serving runs. The logical walk (`flops_fold`) has no grafted
    # activation cast, so a total folded off it fell short of the rows by
    # every cast's FLOPs, and the phase described a different step than its
    # own per-op table. The batch sweep re-reads its points this way too.
    flops = step.total(lambda t: int(t.row.get("flops_per_rank") or 0))
    total = step.total(lambda t: int(t.row.get("access_bytes_per_rank") or 0))
    mem_write = step.total(lambda t: int(t.row.get("act_out_bytes_per_rank") or 0))
    mem_read = total - mem_write
    ai = arithmetic_intensity(flops, total)

    # Tokens leaving the system per step (all batch sequences in parallel).
    effective_tokens, effective_tokens_formula = step_tokens(
        phase, batch, tokens_per_step, speculation)

    # Time, from the same place the bytes came from.
    #
    # With a graph, each leaf already carries its own `max(t_compute, t_memory,
    # t_comm)` and the step is their sum: no kernel overlaps the next one's
    # memory with its own math, and the collectives are in there too, which the
    # cluster form drops entirely because collective traffic never touches HBM.
    # Without a graph there are no leaves, and the whole-step `max(F/Fp, B/Bp)`
    # is the only ceiling available — it assumes perfect overlap across every
    # kernel in the model, so it is a bound and not an estimate.
    #
    # The component sums are surfaced beside `t_sol` for the summary card, and
    # they are NOT addends: each leaf contributes only its winning bound to
    # `t_sol`, so `t_compute + t_memory + t_comm` exceeds it.
    t_compute = step.total(lambda t: float(t.row.get("t_compute_s") or 0.0))
    t_memory = step.total(lambda t: float(t.row.get("t_memory_s") or 0.0))
    t_comm = step.total(lambda t: float(t.row.get("t_comm_s") or 0.0))
    t_sol = step.total(lambda t: float(t.row.get("t_op_s") or 0.0))
    t_sol_by_bound: Optional[Dict[str, float]] = step.by_bound(lambda t: RowTimes.of_row(t.row))
    # Nothing is measured here: every term of the step — a pipeline's other
    # stages too — waits for the estimate's measured fold
    # (`batch_sweep.apply_measured`), which states the track.
    measured = MeasuredTrack.unread(len(step.terms) + (
        sum(len(laws) for s, laws in enumerate(pipeline_time["stage_laws"]) if s != stage)
        if stage is not None and pipeline_time is not None
        and "stage_laws" in pipeline_time else 0))

    # A pipelined rank's step is not the model's step. Everything above this
    # line is now one STAGE's — its layers' FLOPs, its layers' bytes — which is
    # the right per-rank account and the wrong thing to call a latency. The
    # model's step is the stages in sequence plus the inter-stage handoff,
    # which `pipeline_time` carries when the overlay summed every stage
    # (each stage's rows price its own exit cut, so hops land once). What is
    # still not modelled is the fill/drain bubble — a steady-state throughput
    # artifact, not a roofline term; the note says so rather than calling the
    # sum one. Without `pipeline_time` the old rule holds: no number at all
    # is better than 1/pp of the truth, because `t_sol_s` is read directly as
    # TTFT (gallery), as TPOT, and as the denominator of every tokens/s. The
    # capacity account beside them — fits, max_tokens, max_batch, weights and
    # KV per rank — is unaffected either way: those are per-rank questions
    # with per-rank answers.
    t_measured = measured.t_s
    time_unpublished_reason = ""
    formulas_note_pp = ""
    if no_capacity:
        # The counts describe a step, the time would describe a run, and
        # this deployment has no run. Checked before the pp branch: an
        # infeasible pp>1 config has no admissible batch to time either.
        time_unpublished_reason = (
            infeasibility_reason or "No capacity for a batch at this seq.")
        # The step's structure stays — every stage's row laws, so another
        # card that does admit a batch can price it — and its sums go: they
        # would be the time of a run this deployment does not have.
        if pipeline_time is not None:
            pipeline_time = {key: value for key, value in pipeline_time.items()
                             if key in ("stages", "stage", "stage_laws")}
        t_compute = t_memory = t_comm = t_sol = t_measured = None
        t_sol_by_bound = None
    elif stage is not None and pipeline_time is not None:
        draft_t = draft.step if draft is not None else lambda _k: 0.0
        t_compute = pipeline_time["t_compute"] + draft_t("t_compute_s")
        t_memory = pipeline_time["t_memory"] + draft_t("t_memory_s")
        t_comm = pipeline_time["t_comm"] + draft_t("t_comm_s")
        t_sol = pipeline_time["t_sol"] + draft_t("t_sol_s")
        t_sol_by_bound = {
            bound: t + draft_t(f"t_sol_{bound}_s")
            for bound, t in pipeline_time["t_sol_by_bound"].items()}
        t_measured = None
        # The phase carries the sums the times were actually published
        # from — draft included — so the sweep and the invariants read the
        # same basis.
        pipeline_time = {
            **pipeline_time,
            "t_compute": t_compute, "t_memory": t_memory,
            "t_comm": t_comm, "t_sol": t_sol,
            "t_sol_by_bound": dict(t_sol_by_bound),
        }
        formulas_note_pp = (
            f"pp={pipeline_time.get('stages')}: one microbatch's latency — "
            f"Σ every stage's roofline + inter-stage hops; the steady-state "
            f"pipeline bubble is not modelled"
        )
    elif stage is not None:
        view = StagePlacement.of(step_eval.analysis).view(stage)
        time_unpublished_reason = (
            f"pipeline parallelism: the FLOPs, bytes and ops here are this "
            f"rank's stage ({view.describe()}), and the step the model takes is "
            f"every stage in sequence plus the pipeline bubble and the "
            f"inter-stage handoff. Those are not modelled, so no step time, "
            f"latency or throughput is published at pp>1."
        )
        t_compute = t_memory = t_comm = t_sol = t_measured = None
        t_sol_by_bound = None

    formulas = phase_roofline_formulas(
        flops=flops_formula,
        flops_per_step=flops,
        mem_read=mem_read,
        mem_write=mem_write,
        t_sol=t_sol,
        t_compute=t_compute,
        t_memory=t_memory,
        t_comm=t_comm,
        unpublished_reason=time_unpublished_reason,
        t_sol_context=formulas_note_pp,
    )

    return PhaseMetrics(
        phase=phase,
        batch=batch,
        seq_effective=Sq,
        kv_cache_length=cache_len,
        active_params_per_rank=p_active,
        flops_per_step=flops,
        mem_read_bytes=mem_read,
        mem_write_bytes=mem_write,
        total_bytes=total,
        arithmetic_intensity=ai,
        note=phase_note(phase, Sq, batch, note_detail),
        note_detail=note_detail,
        formulas=formulas,
        binding={
            "s_query": Sq, "s_kv": Lq,
            "s_pairs": int(pairs),
            "cache_len": cache_len,
            "tokens_per_step": tokens_per_step,
            "effective_tokens_per_step": effective_tokens,
            "effective_tokens_formula": effective_tokens_formula,
            "s_prefix_cached": s_prefix_cached if phase == "prefill" else 0,
            "s_verify": s_verify if phase == "decode" else 1,
            "speculative_draft_tokens": (
                speculation.draft_tokens if phase == "decode" else 0),
            "acceptance_rate": acceptance_rate if phase == "decode" else 1.0,
            "mla_absorbed": used_absorbed_mla,
        },
        t_compute_s=t_compute,
        t_memory_s=t_memory,
        t_comm_s=t_comm,
        t_sol_s=t_sol,
        t_sol_by_bound=t_sol_by_bound,
        t_sol_by_bound_share=RowRoofline.shares(t_sol_by_bound),
        t_measured_s=t_measured,
        measured_rows=measured.measured_rows,
        total_rows=measured.total_rows,
        coverage=measured.coverage(),
        pipeline_time=(
            pipeline_time if (stage is not None and pipeline_time is not None)
            else None
        ),
        time_unpublished_reason=time_unpublished_reason,
    )


def estimate_phases(
    inputs: PhaseInputs,
    *,
    # One batch per phase: an activation-bound prefill and a KV-bound decode
    # do not admit the same batch.
    prefill_batch: Optional[int] = None,
    decode_batch: Optional[int] = None,
    # One step evaluation and one row list per phase, for the same reason:
    # prefill and decode are different steps, with a different binding and a
    # different graph after the decode rebind.
    prefill_step_eval: Optional[Any] = None,
    decode_step_eval: Optional[Any] = None,
    prefill_rows: Optional[List[Dict[str, Any]]] = None,
    decode_rows: Optional[List[Dict[str, Any]]] = None,
    decode_draft: Optional[Any] = None,
    # pp>1: per-phase pipeline-step time sums (Σ stages + inter-stage hops,
    # bubble unmodelled), from the calibration overlay's per-stage passes.
    # None at pp=1.
    pipeline_time: Optional[Dict[str, Dict[str, float]]] = None,
) -> dict:
    """Phase math (prefill / decode roofline) only. No calibration.

    Calibration enrichment is the responsibility of
    `backend.inference.calibration_overlay.CalibrationOverlay`, called by the
    capacity orchestrator after this function returns. Splitting the stages
    this way means this output is independent of whether a calibration
    registry is on disk — the phase rooflines do not change just because
    measured CSV rows exist.
    """
    # An unset phase batch is 1, the safe minimum.
    pbatch = prefill_batch if prefill_batch is not None else 1
    dbatch = decode_batch if decode_batch is not None else 1
    # Field by field, not `asdict`: that deep-copies the module table's
    # laws, and a SymPy expression copied is built again.
    decode = vars(estimate_phase(
        inputs, phase="decode", batch=dbatch,
        step_eval=decode_step_eval, graph_rows=decode_rows,
        draft=decode_draft,
        pipeline_time=(pipeline_time or {}).get("decode")))
    if decode_draft is not None:
        # Its own term beside the totals it is already folded into, so a page
        # can show the draft's share without subtracting anything.
        decode["speculative_draft"] = decode_draft.wire(
            step_t_sol=decode["t_sol_s"],
            times_published=decode["t_sol_s"] is not None)
    return {
        "prefill": vars(estimate_phase(
            inputs, phase="prefill", batch=pbatch,
            step_eval=prefill_step_eval, graph_rows=prefill_rows,
            pipeline_time=(pipeline_time or {}).get("prefill"))),
        "decode": decode,
    }


def phase_throughput(phase: Dict[str, Any], *, unique_tokens_multiplier: int,
                     num_gpus: int) -> Dict[str, Any]:
    """The three throughputs a phase dict supports, named, from its own
    fields — so no page divides tokens by time itself.

    `t_sol_s` is one model-parallel replica group's step time and
    `effective_tokens_per_step` the tokens that group processes per step,
    so tokens over time is PER GROUP. Independent groups (dp_attention ×
    dp_replicas, the `unique_tokens_multiplier`) run in parallel, so the
    CLUSTER total multiplies by that count, and PER GPU divides the cluster
    total by the GPU count. Each is computed for the theoretical step time
    and, where a measured step time exists, for that too.
    """
    # A phase that never ran (OOM, infeasible) has no step time and so no
    # rate; one that ran states its tokens and how they were counted.
    timed = any(t is not None and t > 0
                for t in (phase.get("t_sol_s"), phase.get("t_measured_s")))
    binding = phase.get("binding") or {}
    tokens = float(binding["effective_tokens_per_step"]) if timed else None
    # The tokens' own derivation (`B × (1 + α·K)`), written out in place.
    tok_f = binding["effective_tokens_formula"] if timed else None
    groups = max(1, int(unique_tokens_multiplier or 1))
    gpus = max(1, int(num_gpus or 1))

    # A decode request's own tokens per step: 1, or 1 + α·K under
    # speculative decoding, where one step verifies the drafts and keeps
    # what is accepted. Its time per output token is the step over that.
    batch = phase.get("batch") or 0
    per_request = (tokens / batch
                   if phase.get("phase") == "decode" and tokens and batch
                   else None)

    def rates(t: Optional[float]) -> Optional[Dict[str, Any]]:
        if t is None or t <= 0 or not tokens or tokens <= 0:
            return None
        per_group = tokens / t
        return {
            # Decode only: one request's time per token it outputs
            # (`t / (1 + α·K)`) — the step itself under plain decode.
            **({"time_per_output_token_s": t / per_request}
               if per_request else {}),
            "per_group": per_group,
            "cluster": per_group * groups,
            "per_gpu": per_group * groups / gpus,
            "formula": {
                "symbolic": (f"{tok_f['symbolic']} × dp_attention × "
                             "dp_replicas / (t × n_gpus)"),
                "substituted": (f"{tok_f['substituted']} × {groups} / "
                                f"({t:.6g} s × {gpus}) = "
                                f"{per_group * groups / gpus:,.0f} tok/s per GPU"),
                "note": (f"tokens per step = {tok_f['symbolic']} = {tokens:g}: "
                         "B sequences in parallel, each producing its "
                         "per-step tokens; × dp_attention × dp_replicas "
                         "for the independent groups"),
            },
        }

    return {
        "tokens_per_step": tokens,
        # Decode: the tokens one request outputs per step (1 + α·K), and
        # the requests the cluster decodes at once (B per group × groups).
        **({"output_tokens_per_request_step": per_request,
            "concurrent_requests": int(batch) * groups} if per_request else {}),
        "groups": groups,
        "num_gpus": gpus,
        "sol": rates(phase.get("t_sol_s")),
        "measured": rates(phase.get("t_measured_s")),
    }


def _routed_columns(node: Any, launch: Any, law_routed: Any, law_ab: Any,
                    book: LawBook, demand: Any) -> Tuple[Optional[Dict[str, Any]],
                                            Optional[Dict[str, Any]]]:
    """A row's routed boundary (`RoutedLaunch.wire`, at the point) and the
    laws of its rows, or `(None, None)` for a row on no routed buffer.

    Beside the record: which of the row's tensors are the routed buffer
    (`LeafAccess.routed_inputs` / `routed_output`), and the bytes the row
    moves per routed row (`routed_extent_per_rank` over the routed rows).
    That ratio is the law's own, the same at every point — every routed
    tensor is `rows x width` — and it is refused where it is not, since a
    padded geometry is read off it.

    And, for a leaf whose weight read saturates per routed row (an expert
    bank, `row_roofline.weight_demand`), what that law asks per routed row
    in whole-bank reads (`weight_rows_per_row`, `rows x read_per_row /
    total`): what the launch's weight read is,
    the same law re-read at the launched rows (`RowRoofline.stored_read`,
    `calibration_overlay._at_geometry`). None where the read does not
    move with the routed rows."""
    if launch is None:
        return None, None
    law_launch = law_routed.values[node.uid]
    per_row = sp.cancel(
        exact(law_ab.routed_extent_per_rank) / exact(law_launch.routed_rows))
    if per_row.free_symbols:
        raise ValueError(
            f"{node.display_path}: its routed bytes are not a fixed number per "
            f"routed row ({per_row}), so no padded launch can be read off them")
    weight_rows = None
    if demand is not None and demand.saturates:
        # In whole-bank reads: `rows x read_per_row / total`, so the law
        # re-read at another row count is `min(weight_rows x rows', 1)` of
        # the bank whatever one row asks of it (an expert's slice, a
        # table's lookup).
        weight_rows = sp.cancel(
            exact(demand.rows) * exact(demand.read_per_row)
            / (exact(demand.total) * exact(law_launch.routed_rows)))
        if weight_rows.free_symbols:
            raise ValueError(
                f"{node.display_path}: its weight law's rows are not a fixed "
                f"number per routed row ({weight_rows})")
    row = {**launch.wire(),
           "inputs": list(law_ab.routed_inputs),
           "output": bool(law_ab.routed_output),
           "bytes_per_row": as_number(per_row),
           "weight_rows_per_row": (None if weight_rows is None
                                   else as_number(weight_rows))}
    law = {"source_rows": book.write(law_launch.source_rows),
           "routed_rows": book.write(law_launch.routed_rows)}
    return row, law


def _row_laws(book: LawBook, *, flops_one: Any, flops_identity: Any,
                   law_ins: Any, law_out: Any,
                   law_shape: Any, law_full: Any, is_comm: bool,
                   law_m: Mapping[str, Any], record: Any,
                   frame: Any, law_ab: Any,
                   law_access_shape: Any,
                   definitions: Any = None) -> Dict[str, Any]:
    """A row's laws in the workload's counts, as the row carries them, off
    the layers before they are read at a point — the same at every point of
    the cell (`ServingLayers.written`):

      * `shape`: each axis of the row's shape columns (`shape_law`), a
        collective's its typed flow (`materializer._int_shape`); None for a
        collective with no flow out;
      * `wire`: the walked shape's axes — every input, the output, the
        ports — for what is printed off it (the wire dims, the access
        popover's outputs);
      * `access_in`: each input's axes as the access record priced them
        (`ServingLayers.access_shapes`), for the access popover's inputs;
      * `dims`: the node's dims (None with no dims); a dim that is neither
        a number nor a law (a parameter's shape) is its text;
      * `definitions`: each name the node's frame binds by a declared
        definition (`ServingLayers.definitions`) as
        `{law, bindings}`: the law's text in names, and each name it reads
        as its law here;
      * `flops`: the FLOPs formula's per-instance law, its substitution
        record (by the identity each value is bound to), and its identity
        where that reads the workload's counts (None where it is every
        point's);
      * `comm`: a collective's payload and buffer axes;
      * `in_terms`: each activation-read term's input (its index in
        `shape_in`), fan-out, elements, whether they are still the
        input's span (`LeafAccess.in_terms_per_rank`) and whether the
        fan-out expanded a store with no head axis — where the cache law
        takes a span over is a fact of the point (None with no access
        record);
      * `out_terms`: each activation-write term's elements and whether they
        are still the whole of what the port publishes
        (`LeafAccess.out_terms_per_rank`), in the same order — a region a
        fused boundary rewrote moves with the workload like any other.
    """
    from backend.arch.eval.dim import Dim, as_number, value_of

    def value_law(value: Any) -> Any:
        v = value_of(value)
        if isinstance(v, (int, sp.Basic)) and not isinstance(v, bool):
            return book.write(v)
        return {"value": as_number(value)}

    def truth_law(value: Any) -> Any:
        v = value_of(value)
        if isinstance(v, sp.Basic) and v.free_symbols:
            return v
        return bool(v)

    ports = (law_shape.ports if law_shape is not None
             and isinstance(law_shape.ports, dict) and law_shape.ports else {})
    shape = None
    if law_ins is not None and not (is_comm and law_out is None):
        shape = {
            "in": [book.axes(axes) for axes in law_ins],
            "out": book.axes(law_out) if law_out is not None else None,
            "ports": [book.axes(p) for p in ports.values() if p] or None,
            # The unsharded shape, for the `*_full` columns.
            "in_full": ([book.axes(axes) for axes in law_full.ins]
                        if law_full is not None and law_full.ins else None),
            "out_full": (book.axes(law_full.out)
                         if law_full is not None and law_full.out is not None
                         else None),
        }
    return {
        "shape": shape,
        "wire": None if law_shape is None else {
            "in": [book.axes(axes) for axes in law_shape.ins or ()],
            "out": (book.axes(law_shape.out)
                    if law_shape.out is not None else None),
            "ports": [book.axes(p) for p in ports.values() if p]},
        "access_in": (None if law_access_shape is None else
                      [book.axes(axes) for axes in law_access_shape.ins or ()]),
        "dims": ({str(k): value_law(v) for k, v in frame.items()}
                 if frame else None),
        "definitions": {
            name: {"law": str(definition.law),
                   "bindings": {key: value_law(value) for key, value
                                in definition.bindings.items()}}
            for name, definition in (definitions or {}).items()},
        # NOT an axis: `as_wire` spells an axis {expr, value}, and a reader
        # that duck-types that pair must not match this. The law and its
        # per-instance value, named as what they are.
        "flops": {
            "one": book.write(flops_one),
            "identity": (flops_identity
                         if isinstance(flops_identity, sp.Basic) and any(
                             is_count(symbol)
                             for symbol in flops_identity.free_symbols)
                         else None),
            "record": {key: value_law(Dim(key, value))
                       for key, value in (record or {}).items()},
        },
        "comm": ({key: book.axes(law_m[f"{key}_extents"])
                  for key in ("payload", "collective_in", "collective_out")}
                 if is_comm and law_m else {}),
        "in_terms": (None if law_ab is None else
                     [{"index": int(term.index),
                       "fanout": value_law(term.fanout),
                       "elements": value_law(term.elements),
                       "spanned": truth_law(term.spanned),
                       "headless_store": bool(term.headless_store)}
                      for term in law_ab.in_terms_per_rank]),
        "out_terms": (None if law_ab is None else
                      [{"elements": value_law(term.elements),
                        "spanned": truth_law(term.spanned)}
                       for term in law_ab.out_terms_per_rank]),
    }


def compute_per_op_timing(
    program: Any,
    costs: Any,
    shapes: Any,
    access_bytes: Any,
    *,
    laws: Any,
    pre: bool,
    at: Mapping[str, int],
    instances: Any,
    breakdown: bool,
    peak_flops_per_sec: Optional[float] = None,
    peak_hbm_bw_bytes_per_sec: Optional[float] = None,
    peak_flops_for_op_fn: Optional[Any] = None,
    overlay_batch: int = 1,
    active_ratios: Any = None,
    tokens_per_step: int = 1,
    comm_meta: Any = None,
    intra_bw_bytes_per_sec: Optional[float] = None,
    inter_bw_bytes_per_sec: Optional[float] = None,
    gpus_per_node: Optional[int],
    intra_bw_label: str = "NVLink",
    inter_bw_label: str = "RDMA",
    #: One link traversal's latency and the shortest a kernel
    #: launch takes, in seconds (the request's, `defaults` unless it says).
    comm_latency_s: float = 0.0,
    launch_floor_s: float = 0.0,
    full_shapes: Any = None,
    node_dims: Any = None,
    consts: Any = None,
    flops_bindings: Any = None,
    #: The nodes of a MoE block on this program (`Program.dispatch_members`,
    #: fused kernels included where what they absorbed was).
    moe_block_members: FrozenSet[int] = frozenset(),
    #: Each leaf on a MoE dispatch's routed buffer, with that dispatch's
    #: boundary on the rank, at this point (`ServingLayers.routed` read
    #: there; `access.routed_launches`).
    routed: Any = None,
    #: The matrices a kernel multiplies, partition by partition, beside one
    #: rank's extents of them (`Analysis.weight_shards`, asked of the row's
    #: `LeafAccess.weight_packs`) — what each row publishes as
    #: `weight_shards`.
    weight_shards: Callable[[Tuple[Tuple[int, ...], ...]], Tuple[Any, ...]],
    #: The card the peaks came from, named so a refusal can say which one
    #: lists no column for a width the deployment chose.
    gpu_sku: str = "",
    #: The card itself, when there is one: its topology decides which of the
    #: framework's all-reduce algorithms run on it (`comm_cost.topology`).
    gpu: Any = None,
) -> List[Dict[str, Any]]:
    """Build per-leaf rooftop timing rows scaled to full per-step model
    totals. Calibration enrichment is NOT done here — it lives in
    `backend.inference.calibration_overlay.CalibrationOverlay.apply`.

    Each leaf in ``program`` becomes one row:
        `{spec_id, op, role, count_per_step, flops_per_rank,
          access_bytes_per_rank, t_compute_s, t_memory_s, t_sol_s,
          t_op_s (= t_sol_s rooftop), source, source_badge}`.

    `source` defaults to `"theoretical_naive"` on every row here;
    CalibrationOverlay overrides to `measured` / `theoretical_fused`
    as appropriate.

    With `breakdown` each row also shows what an estimate's breakdown
    draws: its formula popovers, a collective's per-phase rows, link and
    buffer shapes, and the access facts its popover prints (`point_law`'s
    `access`); without it — a walk a sweep reads its points off — it carries
    none of them, as a row read with no breakdown does (`batch_sweep.
    rows_at`). The numbers and the laws are the same either way.

    Every input is a UID-keyed layer over the same program. String ids are
    rendered only in the output rows; no result tree is joined by display id.

    `laws` are the serving layers before they are read at a point
    (`serving.ServingLayers`), and `pre` says which of its programs the
    rows are over: before the framework's rewrite or after it. A row's laws
    (`roofline_expr`, `shape_law`, `point_law`) are read off those layers,
    written into the cell's book (`laws.book`) once per row
    (`laws.written`); its numbers at `at` off the layers above, the same
    layers read there.
    """
    if pre:
        (law_costs, law_shapes, law_full_shapes, law_access,
         law_access_shapes, law_comm_meta, law_node_dims,
         law_flops_bindings, law_written, law_definitions, law_routed) = (
            laws.pre_cost, laws.pre_shapes, laws.pre_full_shapes,
            laws.pre_access, laws.pre_access_shapes,
            laws.materialized.comm_meta, laws.pre_node_dims,
            laws.pre_flops_bindings, laws.pre_written, laws.pre_definitions,
            laws.pre_routed)
    else:
        (law_costs, law_shapes, law_full_shapes, law_access,
         law_access_shapes, law_comm_meta, law_node_dims,
         law_flops_bindings, law_written, law_definitions, law_routed) = (
            laws.cost, laws.shapes, laws.full_shapes, laws.access,
            laws.access_shapes, laws.comm_meta, laws.node_dims,
            laws.flops_bindings, laws.written, laws.definitions,
            laws.routed)
    ob = max(int(overlay_batch or 1), 1)
    tps = max(int(tokens_per_step or 1), 1)
    from backend.arch.eval.dim import (as_extent, as_number, as_wire, expr_of,
                                       value_of)
    # The card's topology rides with its bandwidths: what it admits of the
    # framework's own all-reduces (`comm_cost.collective_ladder`).
    fabric = Fabric(gpus_per_node, intra_bw_bytes_per_sec,
                    inter_bw_bytes_per_sec, intra_bw_label, inter_bw_label,
                    comm_latency_s)._replace(**topology(gpu))
    out: List[Dict[str, Any]] = []
    for uid, cost in costs.values.items():
        node = program.by_uid(uid)
        if node is None or node.children:
            continue
        with node_context(node.display_path):
            sid = node.aggregation_id
            cnt = instance_count(instances, uid)
            # Round ONCE, after scaling. `cost.flops` is rational wherever the
            # step's work per instance is: a routed expert sees
            # `tokens x k / n_experts` tokens, which for Kimi-K2 (4096 tokens, top
            # 8 of 384) is 85.33 — an expectation, not a count. Truncating that
            # first and multiplying by 23,040 instances threw away 15,360 FLOPs
            # per step, and the error grows with the expert count while the
            # table's block total keeps the fraction exact.
            flops_per_rank = round(float(value_of(cost.flops or 0)) * cnt)

            # The row's law is the walk's own: each component its law in
            # the workload's symbols, off the same layers before they were
            # read at this point (`laws`).
            law_cost = law_costs.values.get(uid)
            flops_one = (value_of(law_cost.flops)
                         if law_cost.flops is not None else 0)
            law_ab = law_access.values.get(uid)
            law_shape = law_shapes.values.get(uid)
            ab = access_bytes.values.get(uid)
            # Component-split fields for frontend batch-dropdown rescaling.
            # Engine ran at `overlay_batch`; weight + metadata are
            # *capacity-bounded* (a leaf can be read at most `weight_total`
            # bytes per step no matter how many tokens hit it) while
            # activation R/W scales linearly with tokens. The saturating read
            # `min(rows × read_per_row, weight_total)` lives here; its
            # operands are stated a few lines below, each off the leaf's own
            # walk. A reader at another batch substitutes the row's own law
            # (`roofline_expr`) rather than recomposing the min.
            if ab is not None:
                # Use per-rank component fields directly — they already
                # incorporate both the structural shard divisor AND the seq
                # CP factor (via `ss.shape_in` substitution in access_bytes
                # annotate). Do NOT divide `*_bytes / divisor` here: that only
                # covers the structural axis and misses CP, inflating numeric
                # `access_per_rank` by the CP factor while the substituted
                # formula string already shows CP-divided.
                weight_total_pr = int(ab.weight_bytes_per_rank) * cnt
                meta_total_pr = int(ab.metadata_bytes_per_rank) * cnt
                # The input conversion a quantised matmul runs is activation
                # traffic like any other: the narrow copy of its own operand,
                # written once and read back (`LeafAccess.convert_bytes`).
                act_in_one = value_of(law_ab.act_in_extent_per_rank)
                act_out_one = value_of(law_ab.act_out_extent_per_rank)
                convert_one = value_of(law_ab.convert_extent_per_rank)
                kv_read_one = value_of(law_ab.kv_read_extent_per_rank)
            else:
                weight_total_pr = 0
                meta_total_pr = 0
                act_in_one = act_out_one = convert_one = kv_read_one = 0
            # How much of its own resident weights this step asks of this
            # leaf, as a fraction of them (`RowRoofline.stored_read`). One
            # source: the leaf's OWN extents.
            #
            # A leaf whose active copies are its stored ones reads them once
            # per launch, whatever the batch: the fraction is 1 and the
            # saturating min collapses. A leaf whose routing reaches only
            # some of its copies (`active:` under `repeat:` — an expert bank)
            # reads one expert per routed ROW, so the fraction is the rows
            # its kernel runs on, read off the same per-rank shape this row
            # reports. It must not be the step's token count times the active
            # ratio: the step's count is a model-level fact and the rows are
            # the leaf's, and under DP attention an expert bank receives
            # every rank's tokens while the attention beside it receives only
            # its own — so the two readings disagreed by the DP degree and
            # the bank's weight read came out that much too small.
            active_ratio = float(
                active_ratios.values.get(uid, 1.0)
                if active_ratios is not None else 1.0
            )
            shape = shapes.values.get(uid)
            # One token's active share of the leaf's weights, as the row
            # states it; not the law's operand.
            weight_per_token_active_pr = int(weight_total_pr * active_ratio)
            # The rows the saturation counts and the bytes one of them reads
            # arrive as one pair (`row_roofline.weight_demand`), so the two
            # halves cannot be taken from different readings: their product
            # is what the row's law saturates. The pair's own docstring
            # states which leaf gets which reading.
            demand = weight_demand(
                program, node, law_shape, active_ratio=active_ratio,
                step_rows=laws.step_rows, total=weight_total_pr)
            # A lookup table reads the rows its lookups gather,
            # `min(table, table x lookups / rows)` — the same saturating min
            # as an expert bank, with the whole table as its total and the
            # rows one token gathers as its per-token read.
            table = ab.table_read_per_rank if ab is not None else None
            if table is not None:
                # One request's lookups: the step's over its batch.
                lookups_per_request = float(
                    sp.Rational(value_of(table.lookups)) / ob)
                weight_total_pr = int(value_of(table.weight)) * cnt
                meta_total_pr = int(value_of(table.metadata)) * cnt
                # A gather's read is stated per TOKEN — `table x lookups /
                # rows` below is one token's share — so its row count is the
                # step's tokens, whatever its copies do. (A vocab-parallel
                # embedding holds a share of the table, so its active copies
                # are under its stored ones and the branch above would
                # otherwise hand it an expert bank's reading.)
                weight_per_token_active_pr = (
                    weight_total_pr * lookups_per_request
                    / float(value_of(table.rows)) / (tps / ob))
                # As a law: the table's bytes in the share its lookups
                # gather, `table x lookups / rows`.
                demand = WeightDemand(
                    value_of(law_ab.table_read_per_rank.lookups),
                    sp.Rational(weight_total_pr)
                    / value_of(law_ab.table_read_per_rank.rows),
                    weight_total_pr, saturates=True)
            asked_law = demand.asked
            # Whether this row's stored read is the saturating law's rather
            # than the leaf's whole weights once: the access popover then
            # prints the law's read, not the leaf's weights.
            saturating = demand.saturates
            # Routing imbalance is in the walk: a routed tensor's rows are
            # the busiest rank's (`ExpectedRoutedLoad.busiest_rank_load_factor`
            # times the mean, `layout.physical`), so this leaf's FLOPs, its
            # activation bytes and the rows its weight read saturates on are
            # that rank's already, and the route exchange's payload is too.
            # The dtype this leaf's matmul runs at, and what a card without
            # that MMA means. A card with no column for a quantised matmul's
            # dtype runs it the way the framework does: weight-only — the
            # weights still streamed at their stored format, the activation
            # NOT quantised, the matmul at the activation's own dtype (vLLM's
            # FP8 Marlin path on pre-Hopper, `pipeline.leaf_peak`). First,
            # however, an explicitly declared promotion ladder promotes within
            # its own audited families: missing FP4 beside present FP8 prices
            # at FP8 (Humming's software-dequantised Hopper path), retaining
            # the activation conversion it really performs. Only when no
            # declared ladder family remains feasible does the row lose its
            # conversion and take the edge's column. A width the deployment
            # CHOSE has no such fallback and is refused below.
            #
            # Per-op peak FLOPs/s — mixed-precision specs pick the GPU CSV
            # column matching each leaf's compute dtype **name**, not a byte
            # width (BF16 and FP16 are both 2 bytes). The closure returns
            # None when the SKU has no column or the row has no name; fall
            # through to the global peak. A user pin ignores the leaf and
            # always returns the pinned column. Weight-only quant (W4A16)
            # dequantizes to the activation dtype, so the name is the
            # activation's. The column is asked once per dtype the row runs
            # at: the declared one, then the edge's if the card lacks it.
            #
            # "The card lacks it" is asked of every family the SELECTED
            # launch declares (`LeafAccess.mma_dtypes`), not inferred from
            # operand width. DeepGEMM's Blackwell FP8×MXFP4 launch needs both
            # families, so an H100 with no FP4 column cannot select it.
            # Humming's Hopper WGMMA launch explicitly declares the
            # alternative-family ladder; a quantised activation alone proves
            # nothing about the instruction.
            adt_name = ab.act_dtype if ab is not None else None
            converts = True
            op_peak: Optional[float] = None
            if peak_flops_for_op_fn is not None and adt_name is not None:
                # Asked once per DISTINCT family the launch declares; the
                # compute dtype is one of them.
                columns = {dt: peak_flops_for_op_fn(dt) for dt in
                           dict.fromkeys((adt_name, *(ab.mma_dtypes or ())))}
                op_peak = (columns[adt_name]
                           if all(c is not None for c in columns.values())
                           else None)
                can_promote = ab.promotion_ladder
                if (op_peak is None and can_promote
                        and not ab.act_dtype_chosen):
                    supported = tuple(dt for dt, peak in columns.items()
                                      if peak is not None)
                    promoted = (max(supported, key=dtype_bytes)
                                if supported else None)
                    if promoted is not None:
                        # Monotone promotion, never a second precision owner:
                        # every candidate family was declared by the selected
                        # launch, and the widest card-supported one is the
                        # highest feasible rung.
                        adt_name = promoted
                        op_peak = columns[promoted]
                if op_peak is None and not ab.act_dtype_chosen:
                    adt_name = ab.edge_dtype
                    convert_one = 0
                    converts = False
                    if adt_name is not None:
                        op_peak = peak_flops_for_op_fn(adt_name)
            # The activation dtype is the access layer's declaration for the
            # leaf; a leaf that does FLOPs and has none is a hole in that layer,
            # not a reason to price it at the dominant column.
            if flops_per_rank > 0 and ab is not None and adt_name is None:
                raise ValueError(
                    f"{node.display_path}: {flops_per_rank} FLOPs on a leaf whose "
                    "activation dtype the access layer did not declare; the peak "
                    "column cannot be chosen")
            if (op_peak is None and flops_per_rank > 0 and ab is not None
                    and ab.act_dtype_chosen
                    and peak_flops_for_op_fn is not None):
                # §3.8: a width the DEPLOYMENT chose and this card cannot
                # run. A hardware fact, so the step is refused with the
                # reason rather than priced on another column — and only
                # with a card in hand: a caller that passed no roofline
                # asked for no times, not for a verdict. The refusal is the
                # one a caller showing FLOPs and bytes without a time
                # already handles (`NoPeakForDtype`).
                raise NoPeakForDtype(gpu_sku or "This GPU", adt_name,
                                     peak_kind(adt_name))
            if op_peak is None:
                op_peak = peak_flops_per_sec
            # Comm rooftop: a parallelism collective's per-rank traffic
            # crosses the INTERCONNECT, not HBM — so it's a third term, not
            # folded into t_memory. For cross-node AR, NCCL/Megatron use a
            # hierarchical 3-phase schedule (intra-RS + inter-AR on the
            # shard + intra-AG), so the cost sums `bytes / fabric_bw` over
            # the phases rather than `flat_ring_bytes / inter_bw`. The
            # `is_comm_op` flag on the op spec is the dispatch — never
            # op-name string match, so adding a future collective primitive
            # (broadcast, gather, scatter, …) is one-line in ops.py.
            m = (
                comm_meta.values.get(uid) or {}
                if comm_meta is not None else {}
            )
            # The row facts read below (`is_comm_op`, `shape_in`, SOURCE) are a
            # primitive's registration fields. A custom op's declaration has
            # none of them and a fused kernel has no declaration at all.
            decl = None if isinstance(node, FusedNode) else node.declaration
            op_spec = decl if isinstance(decl, OpSpec) else None
            # The declaration: a primitive's registration, or — for a fused
            # kernel, which has no registration — the collective it absorbed,
            # carried on the node by the rewrite.
            is_comm = (node.is_comm_op if isinstance(node, FusedNode)
                       else bool(op_spec and op_spec.is_comm_op))
            comm_world = int(m.get("comm_world", 1) or 1)
            comm_op = str(m.get("op") or node.op or "") if is_comm else ""
            comm_base_stride = int(m.get("comm_base_stride", 1) or 1) if is_comm else 1
            # The framework's dispatch of this collective
            # (`ServingLayers.collectives`): the rungs it is priced over.
            comm_rungs = ([r.as_dict() for r in laws.collectives.get(comm_op, ())]
                          if is_comm and comm_world > 1 else [])
            # The payload as the emitter built it, one instance's, as the
            # walk's law.
            law_m = (law_comm_meta.values.get(uid) or {}
                     if law_comm_meta is not None else {})
            comm_payload_one = (value_of(law_m["tensor_bytes_extent"])
                                if is_comm and m else 0)
            # The row's law, over the workload and the card, built from the
            # walk's laws: the same at every point of the cell, so written
            # once (`ServingLayers.written`).
            written = law_written.setdefault((uid, converts), {})
            law = written.get("law") or written.setdefault("law", law_columns(build_row_exprs(
                count=cnt, flops=flops_one, act_in=act_in_one,
                act_out=act_out_one, convert=convert_one, kv_read=kv_read_one,
                # The rows the saturation counts times what one of them reads
                # of this leaf's weights — the leaf's own, not the step's
                # token count times an active ratio (`weight_demand`).
                demand=asked_law,
                weight_total=weight_total_pr, metadata_total=meta_total_pr,
                comm_payload=comm_payload_one,
                comm_factor=comm_factor(comm_op, comm_world) if is_comm else 0.0,
                comm_priced=is_comm and comm_world > 1,
                comm_rungs=len(comm_rungs)), laws.book))
            # This row at the walk's point: its law read there on the card
            # (`LawBook.row_at`), the one reading every re-evaluation of it
            # makes, and published through the one rendering
            # (`row_roofline.point_columns`).
            facts = {"count_per_step": cnt, "act_dtype": adt_name, "is_comm": is_comm,
                     "comm_op": comm_op, "comm_world": comm_world,
                     "comm_base_stride": comm_base_stride,
                     "comm_rungs": comm_rungs}
            card = RowCard(op_peak, peak_hbm_bw_bytes_per_sec, fabric,
                           launch_floor_s)
            row_at = laws.book.row_at(
                {**facts, "roofline_expr": law["roofline_expr"]}, at, card)
            point = point_columns(
                row_at, facts, card, breakdown=breakdown,
                t_sym=(m.get("tensor_symbolic")
                       if breakdown and is_comm and m else None),
                t_sub=(payload_substituted(m["payload_extents"],
                                           m["payload_dtype_bytes"])
                       if breakdown and is_comm and m else None))
            # Shape columns are projections of the physical and semantic layers.
            full_shape = (
                full_shapes.values.get(uid) if full_shapes is not None else None
            )

            # The three readings of one axis, each taken from `dim` rather than
            # restated here: whole geometry for the shape column, the exact value
            # for the column beside it, identity plus value for the wire.
            def _dims(axes: Any) -> Optional[List[int]]:
                if axes is None:
                    return None
                return [as_extent(axis, rounding="floor") for axis in axes]

            def _exact_dims(axes: Any) -> Optional[List[Any]]:
                if axes is None:
                    return None
                return [as_number(axis) for axis in axes]

            def _named_dims(axes: Any) -> Optional[List[Any]]:
                if axes is None:
                    return None
                return [as_wire(axis) for axis in axes]

            def _inputs(row: Any, *, exact: bool = False, named: bool = False) -> Any:
                if row is None or not row.ins:
                    return None
                if named:
                    convert = _named_dims
                elif exact:
                    convert = _exact_dims
                else:
                    convert = _dims
                values = [convert(axes) for axes in row.ins]
                return values[0] if len(values) == 1 else values

            def _outputs(row: Any, convert: Any) -> Any:
                """`convert` of the row's output: its one `out`, else one
                entry per port (the `shape_outs` the row publishes)."""
                if row is None:
                    return None
                if row.out is not None:
                    return convert(row.out)
                ports = row.ports if isinstance(row.ports, dict) else {}
                values = [convert(p) for p in ports.values() if p]
                return values or None

            shape_out_val: Optional[List[int]] = None
            if shape is not None and shape.out is not None:
                shape_out_val = _dims(shape.out)
            elif m.get("tensor_shape"):
                shape_out_val = [int(v) for v in m["tensor_shape"]]
            typed_flow_out = (None if m.get("flow_out_shape") is None
                              else [int(v) for v in m["flow_out_shape"]])
            if is_comm and typed_flow_out is not None:
                shape_out_val = list(typed_flow_out)
            elif is_comm and shape_out_val is not None:
                shape_out_val = comm_display_shape(
                    str(node.op or ""), shape_out_val,
                    ([str(expr_of(axis)) for axis in shape.out]
                     if shape is not None and shape.out is not None else None),
                    comm_world,
                ) or shape_out_val
            shape_out_full_val: Optional[List[int]] = None
            semantic_out = _dims(full_shape.out) if full_shape is not None else None
            if semantic_out and shape_out_val and semantic_out != shape_out_val:
                shape_out_full_val = semantic_out
            # Input tensor shape — per-rank `nr.shape_in` (flat list for a
            # single-input op, list-of-lists for multi-input; preserved so the
            # renderer can show `[a×b]+[c×d]`). Comm input keeps the producer
            # shape (only the collective's OUTPUT axis shrinks/grows).
            shape_in_val: Optional[List[Any]] = None
            typed_flow_in = (None if m.get("flow_in_shape") is None
                             else [int(v) for v in m["flow_in_shape"]])
            if is_comm and typed_flow_in is not None:
                shape_in_val = list(typed_flow_in)
            elif shape is not None:
                shape_in_val = _inputs(shape)
            elif m.get("tensor_shape"):
                shape_in_val = [int(v) for v in m["tensor_shape"]]
            shape_in_full_val: Optional[List[Any]] = None
            semantic_in = _inputs(full_shape)
            if semantic_in and shape_in_val and semantic_in != shape_in_val:
                shape_in_full_val = semantic_in
            # Per-PORT output shapes for a fused node whose absorbed siblings emit
            # separate tensors (e.g. qkv_parallel_linear → q/k/v, gate_up → gate/up).
            # `shape_out` is None for these (no single primary output), but the
            # real per-port widths ARE propagated here; the concatenated GEMM width
            # is their sum. Surface them so consumers (calibration identity, DAG)
            # can recover the true output instead of reverse-engineering from flops.
            _shape_outs_src = shape.ports if shape is not None else None
            shape_outs_val: Optional[List[Any]] = (
                [_dims(p) for p in _shape_outs_src.values() if p]
                if isinstance(_shape_outs_src, dict) and _shape_outs_src else None
            )
            # Each axis as its law in the workload's symbols, off the same
            # layers before they were read at this point, so a reader at
            # another point reads the shape there instead of re-walking
            # (`batch_sweep.rows_at`). A collective's columns are its typed
            # flow (`materializer._int_shape`), laws as well.
            law_full = (law_full_shapes.values.get(uid)
                        if law_full_shapes is not None else None)
            if is_comm:
                law_ins = ([law_m["flow_in_shape"]]
                           if law_m.get("flow_in_shape") is not None else None)
                law_out = law_m.get("flow_out_shape")
            else:
                # A source op (a parameter vector, an embedding table read)
                # has no inputs; its law is its own output's, and an empty
                # input list is still a law.
                law_ins = list(law_shape.ins) if law_shape is not None else None
                law_out = law_shape.out if law_shape is not None else None
            ports = (shape.ports if shape is not None
                     and isinstance(shape.ports, dict) and shape.ports else {})
            if "shape" not in written:
                written.update(_row_laws(
                    laws.book, flops_one=flops_one,
                    flops_identity=(expr_of(law_cost.flops)
                                    if law_cost.flops is not None else 0),
                    law_ins=law_ins, law_out=law_out, law_shape=law_shape,
                    law_full=law_full, is_comm=is_comm, law_m=law_m,
                    record=(law_flops_bindings.values.get(uid)
                            if law_flops_bindings is not None else None),
                    frame=(law_node_dims.values.get(uid)
                           if law_node_dims is not None else None),
                    law_ab=law_ab,
                    law_access_shape=law_access_shapes.values.get(uid),
                    definitions=law_definitions.values.get(uid)))
            _weight_axes = weight_shape_of(
                node.op, node.declaration,
                node_dims.values.get(uid) if node_dims is not None else None)
            record = (flops_bindings.values.get(uid)
                      if flops_bindings is not None else None) or {}
            routed_row, routed_law = _routed_columns(
                node, None if routed is None else routed.values.get(uid),
                law_routed, law_ab, laws.book, demand)
            point_law: Dict[str, Any] = {
                "point": dict(at),
                # The output ports by name, in `shape_outs` order: the table
                # row of a kernel that emits several (`module_rows`).
                "ports": [name for name, p in ports.items() if p],
                "wire": written["wire"],
                "access_in": written["access_in"],
                "dims": written["dims"],
                "definitions": written["definitions"],
                # NOT an axis: the FLOPs formula's expression at this
                # point, beside its laws.
                "flops": {"law": expr_of(cost.flops or 0),
                          **written["flops"]},
                "access": (access_display_facts(ab)
                           if breakdown and ab is not None else None),
                # The access record's read and write terms as laws, for a
                # reader at another point (`batch_sweep._access_at`).
                "in_terms": written["in_terms"],
                "out_terms": written["out_terms"],
                # Whether the row's stored read is its saturating law's (an
                # expert bank, a gathered table): the popover then prints the
                # law's read at every point, not the leaf's weights.
                "saturating": saturating,
                "comm": ({
                    "symbolic": m.get("tensor_symbolic"),
                    "dtype_bytes": float(m["payload_dtype_bytes"]),
                    **written["comm"],
                } if is_comm and m else None),
                # The routed boundary's rows as laws (`routed` below), for a
                # reader at another point (`batch_sweep.rows_at`).
                "routed": routed_law,
            }
            shape_law = written["shape"]
            # The popovers' intermediate quantities at this point
            # (`FormulaDefs`), off the row's own laws.
            defs = FormulaDefs(
                dims=written["dims"], definitions=written["definitions"],
                point=at,
                value=lambda law: Point(at).laws(laws.book, (law,))[law],
                kv_law=law["roofline_expr"]["kv_read"],
                kv_total=row_at.kv_read, count=cnt)
            # `sid` is the aggregation_id (`f"{template}|{display_path}"` for
            # block-internal leaves, bare display_path at model root). The
            # frontend joins per_op_timing rows to graph nodes by `aggregation_id`
            # directly — that's the canonical id carried from `SpecIRNode` all
            # the way to ELK input, no translation at any layer boundary.
            # `template_name` is the node's own field, so consumers can filter
            # by block template without re-parsing aggregation_id.
            row_template = node.template or ""
            # The constants this launch's math takes beside its operands, as
            # the walk bound them at this node (a fused kernel: its
            # origins', carried across the rewrite).
            _row_consts = (dict(consts.values.get(node.uid) or {})
                           if consts is not None else {})
            _nd = node_dims.values.get(node.uid) if node_dims is not None else None
            _row_node_dims = ({k: as_number(v) for k, v in _nd.items()}
                              if _nd else None)
            # A derived group's row names its declaration and history length as
            # fields, so a view that shows the model per declaration groups the
            # rows without reading them out of `template_name`: the block
            # invocation the node runs in answers.
            block = program.block_of(uid)
            derived = (program.blocks[block].meta
                       if block is not None else None)
            out.append({
                "aggregation_id": sid,
                # The one exact join key. aggregation_id is a reporting fact
                # (not unique across template invocations); the module-tree
                # walk joins timing onto its rows by uid instead.
                "uid": uid,
                "template_name": row_template,
                **({"derived_group": derived.derived_group()}
                   if derived is not None and derived.visible else {}),
                # The node's path in its block — with `template_name`, what
                # `aggregation_id` composes; a reader joins on the fields.
                "display_path": node.display_path,
                "op": node.op,
                "role": node.role,
                # This node's Invocation dims, carried on the row so the identity
                # reads the constants a tensor shape can't (a rope rotation width
                # hidden inside heads x width) off the row itself — not a global
                # model-wide dim table. Fused nodes get their origins' dims from
                # the rebase (`_first`) above. Resolved to plain numbers here:
                # a `Dim` carries a sympy symbol the wire encoder cannot serialise.
                "node_dims": _row_node_dims,
                # The function constants this launch's math takes beside its
                # operands, `{symbol: value}` (`Analysis.consts`: the ones
                # its op declares, `OpSpec.consts`, as this node's frame
                # bound them). On the row because a consumer that has to
                # COMPUTE the function needs the model's value and must not
                # pick one: a clamped SwiGLU measured at a bound no model
                # publishes, with an oracle that picked the same bound,
                # agrees with itself about nothing. Absent where the op
                # takes none or the model binds none.
                **({"consts": _row_consts} if _row_consts else {}),
                # Whether this row belongs to a MoE expert block — a node inside
                # the dispatch zone (asked of the program so an expert leaf whose
                # own flag is unset still answers via its bank ancestor) or a
                # `permute` / `combine` boundary of it. The block aggregation reads
                # this off the row instead of matching `expert` / `moe` substrings.
                "moe_block_member": uid in moe_block_members,
                # The leaf's activation dtype (the access layer's declaration)
                # rides the row so a consumer that re-evaluates the roofline at
                # another batch or on another card asks the row, not a global.
                "act_dtype": adt_name,
                # The ordered framework selector's closed outcome. Empty when
                # a launch was selected; a non-empty `launch_gap` says every
                # candidate was skipped and must not be queued as if its
                # operand semantics named a kernel.
                "launch_gap": (ab.launch_gap if ab is not None else None),
                "launch_gaps": list(
                    ab.launch_gaps if ab is not None else ()),
                # The input format's granularity when the selected launch
                # quantises that operand: an activation-group-32 and a
                # group-128 launch are different kernels at one dtype.
                "activation_granularity": (
                    ab.activation_granularity if ab is not None else None),
                "activation_scale_dtype": (
                    ab.activation_scale_dtype if ab is not None else None),
                # The quantised copy of its own operand this kernel needs
                # because the edge does not carry it, and what that
                # conversion is: part of the row's bytes and of its memory
                # time, named so a reader can see which term it is.
                "convert_bytes_per_rank": point["convert_bytes_per_rank"],
                "conversion": (ab.conversion if ab is not None else ""),
                # The dtype environment THIS leaf ran in, which is what a
                # measured row for it is keyed by (`calibration_overlay`):
                # its own resident weight dtype and its own scale dtype, not
                # the step's dominant ones.
                "weight_dtype": (ab.weight_dtype if ab is not None else None),
                "metadata_dtype": (ab.scale_dtype if ab is not None else None),
                "weight_granularity": (ab.weight_granularity
                                       if ab is not None else None),
                # The dtype the leaf writes its primary output at, as the
                # walk priced it (`LeafAccess.output_dtype`) — a cached
                # tensor's format dtype on the leaf that writes it — and
                # which cached tensor that output is, `(node, name)`, so
                # the measurement key states that format once, as the
                # output, and not again as a cache the row touches.
                "output_dtype": (ab.output_dtype if ab is not None else None),
                "output_cache": (
                    (ab.output_cached.node, ab.output_cached.name)
                    if ab is not None and ab.output_cached is not None
                    else None),
                # And how that cached tensor is stored, by its registry
                # name (`LeafAccess.output_format`): the key's
                # `output_format`, which the dtype alone cannot state.
                "output_format": (ab.output_format if ab is not None
                                  else None),
                # Where that output is a quantised activation format
                # (`LeafAccess.output_activation_format`: a grafted cast), the format by
                # name — one value whose granularity and scale dtype are the
                # declared format's (`ACTIVATION_FORMATS`), not columns to
                # keep in step. A cast's cell is keyed by it; without it every
                # FP8 cast was a cell no kernel serves.
                **({"output_activation_format": ab.output_activation_format.name}
                   if ab is not None and ab.output_activation_format is not None
                   else {}),
                # The declaration, on the row: a collective's timing is t_comm
                # and the breakdown splices it under its module. The frontend
                # used to keep its own list of collective op names.
                "is_comm": is_comm,
                "count_per_step": cnt,
                "shape_in": shape_in_val,
                "shape_in_exact": _inputs(shape, exact=True),
                "shape_law": shape_law,
                # The shape columns are this walk's, at its own batch
                # (`batch_sweep` says False for a row it could not re-read).
                "shape_at_batch": True,
                "point_law": point_law,
                "shape_in_dims": _inputs(shape, named=True),
                "shape_in_full": shape_in_full_val,
                "shape_out": shape_out_val,
                # Exact, like `shape_in_exact`: a routed buffer's per-expert
                # rows are a MEAN and may be a fraction. Of whichever output
                # the row states — its one `shape.out`, or every port of a
                # multi-output node, in `shape_outs`' order — so a reader
                # pairing `shape_out or shape_outs` with these columns
                # (`benchmark_identity._sides`) finds the exact rows and the
                # routed axis on every port, not floored means.
                "shape_out_exact": _outputs(shape, _exact_dims),
                "shape_out_dims": _outputs(shape, _named_dims),
                "shape_out_full": shape_out_full_val,
                "shape_outs": shape_outs_val,
                # The MoE dispatch boundary this launch's routed buffer is
                # on (`eval.routed.RoutedLaunch`, its `wire`), with which of
                # its tensors are that buffer and the bytes it moves per
                # routed row: what a padded launch geometry is read off.
                # None for a row on no routed buffer.
                "routed": routed_row,
                # The PARAM's shape, derived at this export boundary from what
                # the node already holds: its declaration's `weight_shape`
                # template (a primitive's registration or the spec's own custom
                # op — gpt-oss's packed expert banks answer the same way) and
                # its resolved slot dims. Pipeline projects those dims through
                # `local_node_dims` before they arrive, so under TP the row
                # carries the shard a kernel allocates. None for a node with no
                # single weight (a fused span).
                "shape_weight": _dims(_weight_axes),
                # And each of its axes by identity (`as_wire`, as
                # `shape_in_dims`): which of them is an expert axis is a
                # declaration (`ops_base.weight_leading_axes`), and the
                # launch geometry asks it by identity rather than by where
                # the matrix is (`benchmark_identity._launch_weight`).
                "shape_weight_dims": _named_dims(_weight_axes),
                # The matrices the kernel multiplies, each by its axes'
                # identities beside one rank's extents of them: which axes a
                # parallel world cut is read off these, not off an
                # activation's width (`kernel_bench.plan.
                # _scale_block_refusal`).
                "weight_shards": [
                    {"pack": list(shard.pack),
                     "rows": as_number(shard.rows),
                     "rows_local": as_number(shard.rows_local),
                     "width": as_number(shard.width),
                     "width_local": as_number(shard.width_local),
                     "granularity": shard.granularity,
                     **({"block_scale_layout": {
                         **shard.block_scale_layout,
                         "rows_local": as_number(shard.block_scale_layout["rows_local"]),
                     }} if shard.block_scale_layout is not None else {})}
                    # A row with no access record streams no stored tensor.
                    for shard in (weight_shards(ab.weight_packs)
                                  if ab is not None else ())],
                "flops_per_rank": point["flops_per_rank"],
                "access_bytes_per_rank": point["access_bytes_per_rank"],
                "t_compute_s": point["t_compute_s"],
                "t_memory_s": point["t_memory_s"],
                "t_comm_s": point["t_comm_s"],
                "t_launch_s": point["t_launch_s"],
                # Per-op peak compute throughput — driven by this leaf's actual
                # dtype rather than the dashboard's dominant-weight column.
                # Frontend batch-dropdown rescaling reads this per row so that
                # an FP8 expert leaf and a BF16 norm leaf rescale on their own
                # rooflines instead of a shared global peak.
                "peak_flops_per_sec_for_op": float(op_peak) if op_peak else 0.0,
                "comm_bytes_per_rank": point["comm_bytes_per_rank"],
                # Comm-row enrichments (empty / 0 for non-comm rows so the
                # frontend can branch on `comm_bw_label` truthiness without
                # an extra `is_comm` flag). World, bandwidth fabric, and the
                # readable bytes derivation are all surfaced so the t_op
                # popover can render the four-term roofline equation.
                "comm_world": comm_world if is_comm else 0,
                # The collective's payload and its node-boundary stride, so another
                # card prices the same collective with `price_collective` instead
                # of scaling this card's total by a bandwidth ratio.
                "comm_op": comm_op,
                "comm_tensor_bytes": point["comm_tensor_bytes"],
                "comm_base_stride": comm_base_stride,
                # The framework's dispatch rungs the collective is priced
                # over, so another card reads the same ladder.
                "comm_rungs": comm_rungs,
                "comm_collective_in_shape": (
                    [int(v) for v in m.get("collective_in_shape") or []]
                    if is_comm else []
                ),
                "comm_collective_out_shape": (
                    [int(v) for v in m.get("collective_out_shape") or []]
                    if is_comm else []
                ),
                "comm_payload_shape": (
                    [int(v) for v in m.get("tensor_shape") or []]
                    if is_comm else []
                ),
                "comm_payload_role": str(m.get("payload_role") or ""),
                "comm_payload_dtype_bytes": (
                    float(m.get("payload_dtype_bytes") or 0.0)
                    if is_comm else 0.0
                ),
                "comm_bw_bytes_per_sec": point.get("comm_bw_bytes_per_sec"),
                "comm_bw_label": point.get("comm_bw_label"),
                "comm_bytes_formula": point.get("comm_bytes_formula"),
                "t_comm_formula": point.get("t_comm_formula"),
                # Per-phase breakdown for the popover — empty for non-comm
                # and intra-only collectives (which collapse to one phase),
                # populated for cross-node hierarchical AR (three phases:
                # intra-RS + inter-AR + intra-AG). Frontend renders one
                # subterm per phase under the row-total formula.
                "comm_phases": point.get("comm_phases"),
                # A hop across a pipeline cut: the cut (sent by stage `cut`),
                # the occurrence that produced the value and the ones that
                # read it past the cut. None / [] on every other row.
                "cut": m.get("cut"),
                "source_occurrence": (
                    None if m.get("source_occurrence") is None
                    else {"node": m["source_occurrence"][0],
                          "iteration": m["source_occurrence"][1]}),
                "sink_occurrences": [
                    {"node": node_name, "iteration": iteration}
                    for node_name, iteration in m.get("sink_occurrences", ())],
                "t_sol_s": point["t_sol_s"],
                # The term that set `t_sol_s` and FLOPs per access byte —
                # the row's own labels, off the same times and components.
                "bound": point["bound"],
                "arithmetic_intensity": point["arithmetic_intensity"],
                "arithmetic_intensity_formula": point.get("arithmetic_intensity_formula"),
                # The row's law over the batch and the card (sstr'd sympy,
                # `row_roofline.build_row_exprs`): a re-evaluation substitutes
                # into it rather than re-deriving the law.
                **law,
                # Component split for frontend batch-dropdown rescaling.
                # All in `per_rank` units (already cnt × / cs scaled).
                # `weight_bytes_per_rank` aliases the *total* (capacity
                # ceiling for MoE banks) — kept for back-compat with any
                # consumer that read it before MoE saturation was added.
                # `weight_bytes_per_token_active_per_rank` is the *per-token
                # active* weight (= total × n_experts_active/n_route_targets
                # for MoE expert leaves, = total for non-MoE) and nothing
                # more: it is what ONE TOKEN activates, a fact about the
                # leaf's weights.
                #
                # It is NOT the token side of the saturating read, and a
                # consumer must not rebuild that read from it. The rows a
                # step reads for are the LEAF's
                # (`row_roofline.kernel_rows`), and a
                # routed bank receives every DP rank's tokens while the
                # attention beside it receives only its own — so
                # `min(B × s_ctx × per_token_active, total)` is the dp-blind
                # derivation this field used to invite. The read at THIS step
                # is `weight_read_bytes_per_rank` below; the read at ANOTHER
                # batch is `roofline_expr`, substituted
                # (`batch_sweep.rows_at`).
                "weight_bytes_per_rank": weight_total_pr,
                "weight_bytes_total_per_rank": weight_total_pr,
                "weight_bytes_per_token_active_per_rank": weight_per_token_active_pr,
                # What this leaf ACTUALLY read of its weights on this step —
                # `min(rows x read_per_row, total)`, the same number that
                # went into `access_bytes_per_rank` two lines up. It is
                # published because it cannot be recovered from the two columns
                # above: the leaf's own row count is not on the row. So every consumer that
                # wanted the read half re-derived a min without them and then
                # took activation as the remainder, which reports the
                # re-derivation's error as activation traffic. Under a balanced
                # routing trace and no DP attention the re-derivation agreed,
                # which is why it stayed invisible.
                "weight_read_bytes_per_rank": point["weight_read_bytes_per_rank"],
                # The scales this step read (same coverage as the weight read
                # above) and the leaf's whole set, for a re-evaluation at
                # another batch — the metadata counterpart of the two weight
                # columns.
                "metadata_bytes_per_rank": point["metadata_bytes_per_rank"],
                "metadata_bytes_total_per_rank": meta_total_pr,
                # The read half of the activation term, named — the counterpart of
                # `act_out_bytes_per_rank` below. With both on the row,
                # `access_bytes_per_rank` is the sum of the five published
                # columns (weight, metadata, act_in, act_out, convert) and
                # nothing has to be recovered by subtraction.
                "act_in_bytes_per_rank": point["act_in_bytes_per_rank"],
                # Of the read half, the bytes read from the cache: the
                # step's KV read is the sum of this column.
                "kv_read_bytes_per_rank": point["kv_read_bytes_per_rank"],
                # The write half of this leaf's traffic, split out of
                # `access_bytes_per_rank` (which is weight + metadata + act_in +
                # act_out + convert — a leaf's whole HBM traffic, both
                # directions). The
                # phase folds it into `mem_write_bytes`; without it on the row,
                # the write could only be published under a field named "read".
                "act_out_bytes_per_rank": point["act_out_bytes_per_rank"],
                # Per-token-active fraction used for the saturation formula
                # (n_experts_active/n_route_targets for MoE expert leaves, 1.0
                # elsewhere). Surfaced so
                # frontend / debug consumers can verify the MoE detection.
                "active_ratio_per_token": active_ratio,
                # Rooftop-only — CalibrationOverlay enriches these. `t_op_s` stays
                # the theoretical rooftop (== t_sol_s, the authoritative default);
                # `t_measured_s` is the SEPARATE measured track (None until a
                # measurement is found — measured sits BESIDE theory, never
                # overwrites it).
                "t_op_s": point["t_op_s"],
                "t_measured_s": None,
                "source": "theoretical_naive",
                "source_badge": {},
                # Placeholders — CalibrationOverlay refreshes with workload identity:
                "shape_json": "",
                "dtype_env_json": "",
                # Per-leaf formula tooltips for the Per-Module Breakdown.
                # `flops_formula` is free — the numeric engine already
                # populates `nr.flops_symbolic` / `nr.flops_values` from
                # the sympy Expr during compute_model. `t_op_formula` is
                # synthesized from t_compute / t_memory so users can read
                # which side is the roofline-binding term.
                "flops_formula": defs.attach(_flops_formula_for_row(
                    point_law["flops"]["law"], value_of(cost.flops or 0),
                    record, cnt=cnt, result_value=row_at.flops))
                if breakdown else None,
                "access_bytes_formula": defs.attach(_access_bytes_formula_for_row(
                    point_law["access"],
                    shape_in_vals=ab.shape_in_vals if ab is not None else None,
                    shape_out_vals=ab.shape_out_vals if ab is not None else None,
                    stored_read=((row_at.weight_read + row_at.metadata)
                                 if saturating else None),
                    cnt=cnt, result_value=row_at.access,
                )) if breakdown else None,
                "t_op_formula": point.get("t_op_formula"),
                "t_compute_formula": point.get("t_compute_formula"),
                "t_memory_formula": point.get("t_memory_formula"),
            })
            # Breakdown-anchor metadata for grafted comm ops — points the
            # frontend at the producing module (AR/RS) or consumer (AG/A2A)
            # row so the comm renders inline under it. `parent_aggregation_id`
            # is the canonical join key (matches `flops_by_module[*].aggregation_id`
            # directly — no template/spec_id split needed at the frontend).
            # `depth` is the comm node's natural tree depth (block-roots
            # at 1) so the frontend can render it at the level it actually
            # lives at — beside the wrapper-level consumer (combine / add)
            # rather than as a child of the row-parallel exit leaf.
            if not breakdown:
                out[-1] = bare_row(out[-1])
            if m:
                out[-1]["parent_aggregation_id"] = m.get("parent_aggregation_id", "")
                out[-1]["template_name"] = m.get("template_name", "")
                if "depth" in m:
                    out[-1]["depth"] = int(m["depth"])
    return out
