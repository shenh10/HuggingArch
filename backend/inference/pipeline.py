"""Serving analysis folded from one ``Analysis``.

The evaluator produces structure-independent facts as UID-keyed layers. This
module adds physical geometry, communication nodes, framework fusion and the
roofline projection in that order. No tree-shaped evaluation DTO is created:
graph rewrites return a ``Rewrite`` and every affected layer follows it with
its own combine law.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Dict, FrozenSet, List, Mapping, Optional, Tuple

from backend.arch.eval import cache as cache_mod
from backend.arch.eval.access import LeafAccess
from backend.arch.eval.analysis import Point
from backend.arch.eval.dim import value_of
from backend.arch.eval.layer import Layer, Rewrite
from backend.arch.eval.refusal import DeploymentRefusal
from backend.arch.eval.values import Cost, Shape
from backend.arch.dtypes import peak_kind
from backend.arch.spec_tree import SpecIRTree

from .framework_bindings import FrameworkBinding
from . import gpu_specs as _gpu_specs
from .gpu_specs import NoPeakForDtype, peak_flops_for_dtype
from .parallel_config import ParallelConfig
from .phases import compute_per_op_timing
from .serving import ServingLayers
from .step_eval import evaluate_step


@dataclass
class AnalysisResult:
    """Physical and fused views of one evaluator result.

    ``analysis`` is the only evaluated model. ``pre_*`` layers add runtime
    communication to its physical layers; the unprefixed layers are those
    same facts rebased across ``rewrite``. String-keyed dictionaries exist
    only inside ``per_op_timing``, the API wire projection.
    """

    analysis: Any
    pre_program: Any
    program: Any
    pre_tree: SpecIRTree
    comm_tree: SpecIRTree
    post_tree: SpecIRTree
    pre_cost: Layer[Cost]
    cost: Layer[Cost]
    pre_shapes: Layer[Shape]
    shapes: Layer[Shape]
    pre_full_shapes: Layer[Shape]
    full_shapes: Layer[Shape]
    pre_access: Layer[LeafAccess]
    access: Layer[LeafAccess]
    #: Which nodes write and read which cached tensor (`cache.kv_access`),
    #: carried like cost: the fused node's entry is its members' union, and
    #: a member keeps its own in `pre_kv_access` for the fused node's
    #: expanded children to read.
    pre_kv_access: Layer[cache_mod.KVAccess]
    kv_access: Layer[cache_mod.KVAccess]
    pre_instances: Layer[int]
    instances: Layer[int]
    rewrite: Rewrite
    fusion_matches: Tuple[Mapping[str, Any], ...] = ()
    per_op_timing: List[Dict[str, Any]] = field(default_factory=list)
    #: The same rows before fusion, priced by the same exit on the grafted
    #: tree the rewrite started from (collectives included).
    pre_per_op_timing: List[Dict[str, Any]] = field(default_factory=list)
    dispatch_zone_uids: frozenset[int] = frozenset()
    #: The Ulysses K/V decompression region, grafted casts included
    #: (`ServingLayers.kv_region`).
    kv_region: frozenset[int] = frozenset()
    #: The unsharded cost as one data-parallel replica's step, same uid
    #: space as `cost` (`Analysis.replica_cost`, extended onto the grafted
    #: tree and rebased across the rewrite): what the module table's
    #: `flops_full` column reads. Optional only so stub-constructed results
    #: (overlay contract tests) stay valid.
    replica_cost: Optional[Layer[Cost]] = None
    #: The layers above before they were read at `point`: the walk's laws
    #: (`ServingLayers`), for a reader that reads another point off them.
    laws: Optional["ServingLayers"] = None
    point: Mapping[str, int] = field(default_factory=dict)
    #: What its rows are measured as, per the (page, scan chunk) a key reads them under
    #: (`benchmark_identity.RowIdentity`): resolved once per walk, and read
    #: for every point read off it (`calibration_overlay.point_keys`).
    identities: Dict[Any, Any] = field(default_factory=dict)

    def fusion_delta(self) -> Dict[str, float]:
        """What the framework's fusions change in this step's time, in
        theory on both sides: the rows before fusion and after, each
        summed by the one row exit (collectives and weight saturation
        included). `delta_s` is `pre - post` as it stands —
        negative when the fused graph is the slower one."""
        pre = sum(float(r["t_sol_s"]) for r in self.pre_per_op_timing)
        post = sum(float(r["t_sol_s"]) for r in self.per_op_timing)
        return {"pre_total_s": pre, "post_total_s": post,
                "delta_s": pre - post}


def peak_or_raise(gpu: Any, dtype: str) -> float:
    """The SKU's peak for a dtype's MMA family. With a SKU in hand, a
    missing column is not "use the dominant column": an FP8 leaf on an
    A100 has no fp8 peak, and pricing it at BF16 reported a number nobody
    could source. The prefill / UI path shows `peak_source: unknown` and no
    number (A10's "not raise" side); the estimate cannot proceed without
    one, so it says so."""
    peak = peak_flops_for_dtype(gpu, dtype)
    if peak is None:
        raise NoPeakForDtype(gpu.sku, dtype, peak_kind(dtype))
    return peak


def leaf_peak(gpu: Any, dtype: str) -> Optional[float]:
    """The SKU's peak for ONE LEAF's matmul dtype, `None` when the card has
    no MMA for it.

    `peak_or_raise` answers for a step: a caller holding one dtype for the
    whole forward has nothing to fall back to. A leaf is different, because
    what a missing column means depends on whose choice the dtype was, and
    only the row knows (`phases.compute_per_op_timing`, which holds the
    leaf's `act_dtype_chosen`):

      * a dtype the CHECKPOINT ships runs weight-only on such a card, and
        that is the framework's own fallback rather than an estimate's
        guess: vLLM serves an FP8 checkpoint on pre-Hopper through
        `MarlinFP8ScaledMMLinearKernel` ("FP8 Marlin kernel for GPUs that
        lack FP8 hardware support … fast weight-only FP8 quantization",
        model_executor/kernels/linear/scaled_mm/marlin.py:29-50), which
        deletes the activation scale (`layer.input_scale = None`, then
        `del layer.input_scale`, :66-81) — the weights stay FP8, the
        activation is not quantised, and the matmul runs at the
        activation's own dtype;
      * a dtype the DEPLOYMENT chose (a compute pin, or a weight rule of
        its own) has no such fallback, and that step is refused with
        `NoPeakForDtype` — §3.8 of the precision design, a hardware fact.
    """
    return None if not dtype else peak_flops_for_dtype(gpu, dtype)


def serving_layers(step_eval: Any, *,
                   framework_binding: Optional[FrameworkBinding],
                   stage: Optional[int],
                   timed: FrozenSet[str]) -> ServingLayers:
    """The layers a step's rows are read from (`ServingLayers`), off its
    walk: its cell's for this reader configuration (`step_eval.Cell.
    serving`), so the points of a cell share them."""
    step_eval.walk  # noqa: B018 — the cell's walk, built before it is read
    return step_eval.cell.serving(framework_binding, stage, timed)


def analyze(
    resolved_spec: Any,
    *,
    parallel: ParallelConfig,
    framework_binding: Optional[FrameworkBinding] = None,
    batch_size: int = 1,
    precision: Any = None,
    peak_flops_per_sec: Optional[float] = None,
    peak_hbm_bw_bytes_per_sec: Optional[float] = None,
    gpu: Optional[Any] = None,
    intra_bw_bytes_per_sec: Optional[float] = None,
    inter_bw_bytes_per_sec: Optional[float] = None,
    # The node size is the SKU's declaration (`gpu_specs.csv`), read off
    # `gpu` when one is given. A caller pricing fabrics without a SKU
    # (bandwidths only) names the node size itself; None with no SKU means
    # there is no fabric to split a collective over.
    gpus_per_node: Optional[int] = None,
    intra_bw_label: str = "NVLink",
    inter_bw_label: str = "RDMA",
    interconnect_bw_bytes_per_sec: Optional[float] = None,
    # One link traversal's latency and the shortest a kernel
    # launch takes, in seconds: a caller that prices on a card states them
    # (the request's, `defaults`); 0 prices neither floor.
    comm_latency_s: float = 0.0,
    launch_floor_s: float = 0.0,
    avg_input_tokens: Optional[int] = None,
    avg_output_tokens: Optional[int] = None,
    s_prefix_cached: int = 0,
    phase: str = "prefill",
    s_verify: int = 1,
    ep_load_factor: Any = 1,
    stage: Optional[int] = None,
    step_eval: Optional[Any] = None,
    include_aux_timing: bool = False,
    timed_stacks: Optional[Collection[str]] = None,
    allow_load_refusals: bool = False,
    breakdown: bool,
) -> AnalysisResult:
    """Run physical serving analysis and return layer views plus wire rows.

    `breakdown` as `step_rows`'.

    `stage` restricts the rows to one pipeline stage
    (`stages.StagePlacement.view` of this step's program): what runs there,
    and the hops across its exit cut, which are the ones it pays.

    `include_aux_timing` keeps the step's aux leaves (MTP) in
    `per_op_timing`. Estimate stays backbone-only (default False);
    kernel_bench turns it on so those stacks are measurable. An encoder is
    no part of a language-model step: its encode step names it in
    `timed_stacks`.

    `timed_stacks` names the stacks whose leaves are timed, and replaces the
    choice above when given: a speculative draft prices its MTP stack alone
    (`speculative_draft.SpeculativeDraft`), on the same step the backbone
    rows describe; an encode step (`step_eval.evaluate_encode`) its encoder.

    A selected loader's refusal rejects the analysis. Only the kernel planner
    passes `allow_load_refusals`, so it can turn the published refusal into an
    unplaced cell instead of losing every other measurable cell.

    Without `step_eval`, the step is evaluated here at the workload the
    request states, and it must state it: `avg_input_tokens` and
    `avg_output_tokens` (a decode step's cache is the prompt and half the
    output), with `s_prefix_cached` 0 for no cached prefix and `s_verify`
    1 for no speculative verify. The rows count the step's own tokens
    (`step_rows`).
    """
    if step_eval is None:
        if avg_input_tokens is None or avg_output_tokens is None:
            raise TypeError(
                "pipeline.analyze walks the step itself only at a stated "
                "workload: pass avg_input_tokens and avg_output_tokens, or "
                "the step_eval to read")
        step_eval = evaluate_step(
            resolved_spec,
            parallel,
            phase=phase,
            batch=batch_size,
            input_tokens=avg_input_tokens,
            output_tokens=avg_output_tokens,
            s_prefix_cached=s_prefix_cached,
            s_verify=s_verify,
            ep_load_factor=ep_load_factor,
            precision=precision,
        )
    resolved_spec = step_eval.spec
    timed = (
        frozenset(timed_stacks) if timed_stacks is not None
        else step_eval.walk.program.included_stack_names(
            include_aux=include_aux_timing)
    )
    layers = serving_layers(step_eval, framework_binding=framework_binding,
                            stage=stage, timed=timed)
    if layers.analysis.load_refusals and not allow_load_refusals:
        raise DeploymentRefusal("; ".join(layers.analysis.load_refusals),
                                knobs=("tp_attention",), phase=phase)
    # The rows are read at this step's point: every layer above is the
    # walk's laws, and this is where they become this point's numbers.
    return step_rows(
        layers, step_eval.reading,
        peak_flops_per_sec=peak_flops_per_sec,
        peak_hbm_bw_bytes_per_sec=peak_hbm_bw_bytes_per_sec, gpu=gpu,
        intra_bw_bytes_per_sec=intra_bw_bytes_per_sec,
        inter_bw_bytes_per_sec=inter_bw_bytes_per_sec,
        gpus_per_node=gpus_per_node, intra_bw_label=intra_bw_label,
        inter_bw_label=inter_bw_label,
        interconnect_bw_bytes_per_sec=interconnect_bw_bytes_per_sec,
        comm_latency_s=comm_latency_s, launch_floor_s=launch_floor_s,
        breakdown=breakdown)


def step_rows(
    layers: ServingLayers,
    point: Point,
    *,
    peak_flops_per_sec: Optional[float] = None,
    peak_hbm_bw_bytes_per_sec: Optional[float] = None,
    gpu: Optional[Any] = None,
    intra_bw_bytes_per_sec: Optional[float] = None,
    inter_bw_bytes_per_sec: Optional[float] = None,
    gpus_per_node: Optional[int] = None,
    intra_bw_label: str = "NVLink",
    inter_bw_label: str = "RDMA",
    interconnect_bw_bytes_per_sec: Optional[float] = None,
    comm_latency_s: float = 0.0,
    launch_floor_s: float = 0.0,
    breakdown: bool,
) -> AnalysisResult:
    """One point's step: the serving layers read at it (`ServingLayers.at`)
    and the per-op rows priced off them on a card, each row carrying its
    laws off the same layers before the reading (`laws`). With `breakdown`
    the rows also carry what an estimate's breakdown draws
    (`phases.compute_per_op_timing`).
    The step's tokens are the walk's batch times its query block; an
    encode step has no query block, and its tokens are its requests."""
    at = layers.at(point)
    batch_size = int(point.read(value_of(layers.analysis.binding["B"])))
    tokens_per_step = int(point.read(layers.step_rows))
    analysis = at.analysis
    pre_tree = at.pre_tree
    materialized = at.materialized
    pre_program, program, rewrite = at.pre_program, at.program, at.rewrite
    matches = at.fusion_matches
    pre_cost, cost = at.pre_cost, at.cost
    replica_cost = at.replica_cost
    pre_shapes, shapes = at.pre_shapes, at.shapes
    pre_full_shapes, full_shapes = at.pre_full_shapes, at.full_shapes
    pre_access, access = at.pre_access, at.access
    pre_kv_access, kv_access = at.pre_kv_access, at.kv_access
    pre_instances, instances = at.pre_instances, at.instances
    pre_ratios, ratios = at.pre_ratios, at.ratios
    comm_meta = at.comm_meta
    node_dims, pre_node_dims = at.node_dims, at.pre_node_dims
    consts, pre_consts = at.consts, at.pre_consts
    flops_bindings = at.flops_bindings
    pre_flops_bindings = at.pre_flops_bindings
    untimed, dispatch = set(at.untimed), set(at.dispatch)
    members, pre_members = at.members, at.pre_members
    timing_cost = Layer(
        program,
        {uid: value for uid, value in cost.values.items()
         if uid not in untimed},
    )

    intra = (
        intra_bw_bytes_per_sec
        if intra_bw_bytes_per_sec is not None
        else interconnect_bw_bytes_per_sec
    )
    inter = (
        inter_bw_bytes_per_sec
        if inter_bw_bytes_per_sec is not None
        else interconnect_bw_bytes_per_sec
    )
    peak_flops_for_op_fn = None
    if gpu is not None:
        # Each leaf's matmul dtype is the precision profile's answer
        # (`ResolvedPrecision.compute_dtype`, on the access row); what a card
        # without that column means is `leaf_peak`'s docstring, and the row
        # that knows whose choice the dtype was decides it.
        peak_flops_for_op_fn = (lambda dt: leaf_peak(gpu, dt))

    timing_kwargs = dict(
        peak_flops_per_sec=peak_flops_per_sec,
        peak_hbm_bw_bytes_per_sec=peak_hbm_bw_bytes_per_sec,
        peak_flops_for_op_fn=peak_flops_for_op_fn,
        gpu_sku=(gpu.sku if gpu is not None else ""),
        gpu=gpu,
        overlay_batch=batch_size,
        tokens_per_step=tokens_per_step,
        intra_bw_bytes_per_sec=intra,
        inter_bw_bytes_per_sec=inter,
        gpus_per_node=(gpus_per_node if gpus_per_node is not None
                       else _gpu_specs.gpus_per_node(gpu) if gpu is not None
                       else None),
        intra_bw_label=intra_bw_label,
        inter_bw_label=inter_bw_label,
        comm_latency_s=comm_latency_s,
        launch_floor_s=launch_floor_s,
    )
    per_op_timing = compute_per_op_timing(
        program,
        timing_cost,
        shapes,
        access,
        moe_block_members=frozenset(members),
        routed=at.routed,
        weight_shards=analysis.weight_shards,
        instances=instances,
        active_ratios=ratios,
        comm_meta=comm_meta,
        full_shapes=full_shapes,
        node_dims=node_dims,
        consts=consts,
        flops_bindings=flops_bindings,
        laws=layers, pre=False, at=point.counts, breakdown=breakdown,
        **timing_kwargs,
    )
    # The same rows before fusion: the same exit, over the grafted tree the
    # rewrite started from — every layer as it stood before the rebase.
    pre_per_op_timing = compute_per_op_timing(
        pre_program,
        Layer(pre_program, {uid: value for uid, value in pre_cost.values.items()
                            if uid not in untimed}),
        pre_shapes,
        pre_access,
        moe_block_members=pre_members,
        routed=at.pre_routed,
        weight_shards=analysis.weight_shards,
        instances=pre_instances,
        active_ratios=pre_ratios,
        comm_meta=materialized.comm_meta,
        full_shapes=pre_full_shapes,
        node_dims=pre_node_dims,
        consts=pre_consts,
        flops_bindings=pre_flops_bindings,
        laws=layers, pre=True, at=point.counts, breakdown=breakdown,
        **timing_kwargs,
    )

    post_tree = at.post_tree
    return AnalysisResult(
        analysis=analysis,
        pre_program=pre_program,
        program=program,
        pre_tree=pre_tree,
        comm_tree=materialized.tree,
        post_tree=post_tree,
        pre_cost=pre_cost,
        cost=cost,
        replica_cost=replica_cost,
        pre_shapes=pre_shapes,
        shapes=shapes,
        pre_full_shapes=pre_full_shapes,
        full_shapes=full_shapes,
        pre_access=pre_access,
        access=access,
        pre_kv_access=pre_kv_access,
        kv_access=kv_access,
        kv_region=at.kv_region,
        pre_instances=pre_instances,
        instances=instances,
        rewrite=rewrite,
        fusion_matches=matches,
        per_op_timing=per_op_timing,
        pre_per_op_timing=pre_per_op_timing,
        dispatch_zone_uids=(
            frozenset(dispatch) & frozenset(timing_cost.values)
        ),
        laws=layers,
        point=dict(point.counts),
    )
