"""KV cache capacity — per-rank memory + max tokens/batch under a ParallelConfig.

Resident weight bytes come from the layout join (`resident_weights` on the
walk's artifacts). A pipeline stage is the same join after
`artifacts_for_layers`; serving-cast and per-role dtype still price through
the display tree, which is the module breakdown beside those numbers, not
the budget. NOT divided by the data-parallel axes: `dp_replicas` replicates
the whole model and `dp_attention` replicates weights (it shards the batch),
so neither reduces per-rank weight — they only raise concurrency
(`unique_tokens_multiplier`). `cluster_size` is the GPU COUNT — a throughput
denominator and nothing else. It divides no quantity on this page: not weight,
not FLOPs (`phases`), not the activation peak (`activation_peak`). Each of
those three reads its per-rank number off a walk at the deployment's degrees,
because a rank group's size cannot tell you which axis it splits.

KV per rank is the deployment's physical cache view, not a divisor applied to
the whole model: TP splits KV heads until they run out (the MLA latent has no
head axis and stays replicated), and `dcp` / layersplit CP split what is left.
Under ulysses/pcp, `cp` divides no KV, because prefill context parallelism
all_gathers K/V before the cache write. The displayed shard and replication
factors are the ratio of the whole-model view to this rank's view.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, fields, replace
from functools import cached_property
from typing import (Callable, FrozenSet, Any, Dict, Iterable, Iterator, List,
                    Literal, Mapping, NamedTuple, Optional, Tuple, Union)

import sympy as sp

from backend.arch.eval.dim import value_of
from backend.arch.eval.formats import quant_granularity
from backend.arch.eval.precision import PrecisionProfile
from backend.arch.eval.program import Program
from backend.arch.eval.refusal import DeploymentRefusal, stage_context
from backend.arch.eval.transforms import Workload
from backend.arch.layout.stages import StagePlacement
from backend.arch.loader import get_default_loader

from .activation_peak import estimated_max_fraction
from .assumptions import deployment_assumptions
from .gpu_specs import (
    get_gpu, peak_hbm_bw,
)
from .defaults import COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT
from .card import Card, CardPrice
from .framework_bindings import ServingDeployment
from .frontier import POINT_DIMS
from .point_cloud import estimate_cloud
from .kv_topology import _counted
from .parallel_config import ParallelConfig
from .chunk_terms import ChunkLaunchTerm, merge_chunk_launch_terms
from .intensity import arithmetic_intensity
from .prefill_geometry import PrefillChunkSchedule, PrefillEndpointGeometry
from .batch_sweep import (measure_points, measures_on_card, phase_at, point_measures,
                          fold_chunk_point_measures, rows_at, step_measure)
from .row_roofline import (PHASE, ROW, CardForm, FloorForm, LawBook,
                           RowRoofline, StepSum, column_names)
from backend.measure.auto_sweep import Cell
from .speculative_draft import ForwardFull, SpeculativeDraft
from .step_eval import StepCells, estimate_peak_activation, prefix_refusal
from .comm_cost import Fabric
from .phases import (PhaseInputs, Speculation,
                     estimate_phases, phase_binding, phase_frame,
                     phase_throughput)
from .pipeline import leaf_peak
from .weight_breakdown import per_rank_weight_breakdown
from .calibration_overlay import (  # overlay stage, not a budget scalar
    identity_columns,
    key_columns,
    looks_up,
    point_keys,
    _resolve_overlay_context,
    deployment_kv_block_size,
    overlay_walk,
)
from .capacity_formulas import capacity_formulas  # popover strings, not a budget scalar
from .phase_formulas import enrich_module_breakdown
from .encode_phase import plan_encode
from backend.measure.scope import NONE as NO_MEASUREMENTS
from .walk_registry import shared_walk_cells
from backend.arch.eval.analysis import Point

_GB = 1024 ** 3


@dataclass(frozen=True)
class _CacheBudgetTerms:
    reader: Any
    static: float
    constant_state: float
    per_token: float
    shard_divisor: float
    replication_factor: float


def _cache_budget_terms(
    reader: Any,
    whole: Any,
    parallel: ParallelConfig,
) -> _CacheBudgetTerms:
    """Capacity scalars from two views of one cache geometry.

    The local view prices memory. The whole view supplies only observed ratios;
    it never divides a tensor itself. The fold reads the two views and nothing
    about how they were built.

    A request's cache is `growth × L + floor` past every window. Either term
    may be zero: a cache that is all floor (every layer windowed, or a pure
    recurrent state) holds a fixed amount per request however long it runs.
    Only a cache with neither term is no cache to budget.

    `shard_divisor` is the whole-to-local ratio of that law as `L` grows —
    of the growth when there is growth, of the floor when the cache is all
    floor. It is one ratio, the long-context one, read off the leading term.
    """
    per_token, constant_state = reader.growth_and_floor()
    static = reader.static_scale_bytes()
    if per_token <= 0 and constant_state <= 0:
        raise ValueError(
            "The model's cache neither grows with length nor holds a "
            "per-request floor: there is no KV cache to budget")
    whole_per_token, whole_floor = whole.growth_and_floor()
    shard_divisor = (whole_per_token / per_token if per_token > 0
                     else whole_floor / constant_state)
    attention_ranks = (
        max(1, parallel.tp_attention) * max(1, parallel.cp)
    )
    replication_factor = attention_ranks / shard_divisor
    # layersplit folds occupancy by ~cp, so divisor grows with cp and
    # this ratio stays ≥ 1 at dcp=1. The guard is still only for dcp
    # cutting copies that are not there.
    if replication_factor < 1 - 1e-9:
        uncut_replication = attention_ranks / (
            shard_divisor / max(1, parallel.dcp)
        )
        raise ValueError(
            f"decode context parallelism (dcp={parallel.dcp}) splits the KV "
            f"cache {shard_divisor:.6g}× across {attention_ranks} attention "
            "ranks, which is more ways than there are copies to cut: this "
            f"model's cache is replicated only {uncut_replication:.6g}× "
            "across them at dcp=1. dcp splits what TP left replicated, so it "
            "needs a head-free (MLA) or TP-replicated (MQA) cache — raise "
            "tp_attention or lower dcp"
        )
    return _CacheBudgetTerms(
        reader=reader,
        static=static,
        constant_state=constant_state,
        per_token=per_token,
        shard_divisor=shard_divisor,
        replication_factor=replication_factor,
    )


def _binding_key(free: float, *costs: float) -> Tuple[float, ...]:
    """A stage's place in the smallest-binds order, as a comparable key.

    Two regimes, because a stage that fits and a stage that does not are not
    measured in the same unit. A stage that FITS is ranked by how much of each
    `cost` its free bytes still buy — a count, in whatever the cost is priced
    in, and the counts are compared in the order given. A stage that does NOT
    fit buys none of anything, so there is nothing to count: it is ranked by
    its deficit in BYTES, the memory that would have to be found before the
    question means anything at all, and below every stage that fits.

    The deficit is deliberately NOT divided by a cost. That division is not the
    count continued below zero — it ranks a stage with a small cache and a
    modest overflow below one that overflows by far more, purely because the
    divisor is smaller. Kimi-K3 at `pp=4` overflows by 226, 239, 255 and 239
    GiB; dividing picked the 226 GiB stage, naming the rank CLOSEST to fitting
    as the reason nothing fitted.

    A zero cost never runs out on its term, so it sorts to the top of it.
    """
    if free <= 0:
        return (0.0, free) + (0.0,) * len(costs)
    return (1.0, *(free / c if c > 0 else float("inf") for c in costs))


@dataclass(frozen=True)
class _StageProfile:
    """One pipeline stage's per-rank memory, before any workload.

    A stage is not a fraction of the model — it is a set of layers — so it has
    its own weight AND its own cache, and the two are not the same fraction.
    DeepSeek-V3 at `pp=2` is the case that makes the distinction pay: stage 0
    holds 31 layers (the three dense ones, plus the embedding table) and stage
    1 holds 30 MoE layers, so stage 1 is 20 GiB the heavier while stage 0 is
    the one that caches more. Whichever runs out first is the one that decides
    what the deployment can serve.
    """

    stage: int
    label: str
    weight_bytes: float
    reader: Any
    #: Cache bytes this stage's layers add per token of length: the cached
    #: tensors' data and the dynamic scales stored with it
    #: (`CacheView.growth_and_floor`).
    per_token: float
    static: float
    constant_state: float
    #: Per-request stored bytes at the model's published context length
    #: (`s_max`) — the longest request this rank can be asked to hold. None
    #: when the model publishes no context length: the preparation then
    #: orders stages by tokens alone, and the finish pass — which has the
    #: request — orders them by the request itself.
    request_bytes_at_max_context: Optional[float]

    def free_bytes(self, usable_bytes: float) -> float:
        """What is left for growing cache after this stage's weights and its
        length-independent floors."""
        return (usable_bytes - self.weight_bytes - self.static
                - self.constant_state)

    def headroom(self, usable_bytes: float) -> Tuple[float, ...]:
        """How much this stage can serve, smallest-first, as a sort key.

        Two cost terms because one does not separate the stages a hybrid
        backbone produces. The first is cache TOKENS, which is `max_tokens`'
        own quantity. The second is whole REQUESTS at full context, and it is
        what catches gpt-oss: its layers alternate SWA and GQA with identical
        per-token slopes, so every three-layer stage looks the same by tokens
        while the stage holding two GQA layers caches twice what the stage
        holding one does. Ordering by the slope alone picked the cheap stage
        and doubled the batch the deployment could really serve.

        A stage whose weights alone overflow the GPU serves nothing, and is
        ordered by how badly instead — see `_binding_key`.
        """
        costs = (self.per_token,) if self.request_bytes_at_max_context is None \
            else (self.per_token, self.request_bytes_at_max_context)
        return _binding_key(self.free_bytes(usable_bytes), *costs)


def _pipeline_stage_profiles(placement, stage_bytes, reader,
                             max_context: Optional[float]
                             ) -> Tuple[_StageProfile, ...]:
    """Every stage priced, from ONE cache walk and ONE placement.

    Neither is repeated per stage: `CacheView.on` re-counts laws the
    partition does not change, and `stage_bytes` are the stages' own weight
    projections (`StageView.weights`). Repeating either would be asking the
    same question `pp` times and inviting the two answers to differ.
    """
    out: List[_StageProfile] = []
    for s in range(placement.pp):
        view = placement.view(s)
        r = reader.on(view)
        per_token, constant = r.growth_and_floor()
        out.append(_StageProfile(
            stage=s, label=view.describe(), weight_bytes=stage_bytes[s],
            reader=r, per_token=per_token,
            static=r.static_scale_bytes(),
            constant_state=constant,
            request_bytes_at_max_context=(
                None if max_context is None
                else r.storage_bytes_at(max_context))))
    return tuple(out)


@dataclass
class _CapacityPreparation:
    """Seq-invariant context computed once per (model, GPU, parallel,
    dtypes) tuple. Shared between the single-seq `estimate_capacity`
    path and the multi-seq `estimate_capacity_sweep` path so the heavy
    work (spec match, materialize, KV per-token, weight breakdown, GPU
    peaks, framework binding, measurement scope) is amortized
    across N sweep points instead of repeated per seq.
    """
    # GPU + cluster
    #: The card this preparation is priced on (`card.Card`): the one way it
    #: names a card, and what every road that moves this walk to another
    #: card asks (`Card.refuse_unless_one_class`). `gpu` and `gpu_sku` are
    #: its sheet row and its name, kept as fields because the response
    #: publishes the row and ~160 leaf accessors still read one column.
    card: Card
    gpu: Any
    gpu_sku: str
    cluster_size: int
    num_gpus: int
    parallel: ParallelConfig
    gpu_memory_fraction: float
    model_id: str
    # Spec + materialize defaults
    resolved: Any
    #: The deployment's precision profile (`precision.PrecisionProfile`),
    #: and the budget walk's reading of it against this model — every
    #: dtype, cache format and matmul column below is its answer.
    precision: Any
    resolved_precision: Any
    #: `Analysis.precision_effective()` of the budget walk.
    precision_report: Dict[str, Any]
    # KV per-token (seq-invariant; scalars and displays from the eval cache
    # layer via `kv_reader`)
    kv_div: float
    kv_reader: Any
    kv_static_per_rank: float
    #: Length-independent state that is not the KV cache and does not grow with
    #: it (a linear-attention layer's recurrent state). Subtracted from the
    #: budget beside the static floor, and published because `usable_for_tokens`
    #: cannot be reconciled against `kv_budget_bytes` without it.
    kv_constant_state_per_rank: float
    per_token_per_rank: float
    # Weight breakdown
    wb: Dict[str, Any]
    #: Rebuild `wb` at a given pipeline stage. Carried so the layer that
    #: re-picks the stage at a workload can move the tree with the scalars,
    #: without re-assembling the arguments that decided what a weight costs.
    weight_tree_at: Any
    weight_bytes_per_rank: float
    weight_bytes_total: float
    #: The host-offloadable weights' bytes on a rank of each pipeline stage
    #: (one entry at pp=1), wherever they are kept, and whether that is
    #: host memory (then they are in no per-rank weight figure).
    host_offload_bytes: Tuple[float, ...]
    host_offload: bool
    # Every pipeline stage priced — empty when `pp == 1`. The scalars above are
    # ONE of these, the stage that holds the fewest cache tokens. The rest are
    # kept because that choice is only exact for the workload-free question:
    # WHICH stage runs out first depends on how long the requests are, so
    # `capacity_ceilings` re-picks at the length it is asked about. See
    # `_StageProfile.headroom`.
    pp_stages: Tuple[_StageProfile, ...]
    # WHICH of those stages every per-rank scalar in this preparation is
    # describing — the weight tree, the cache reader and the budget. None
    # when `pp == 1`, where the question does not arise. It is the
    # workload-free answer only: an estimate publishes, and bills compute
    # on, the stage `capacity_ceilings` binds at its length
    # (`CapacityCeilings.stage`), which starts from this one.
    pp_bound: Optional[_StageProfile]
    # Budget (all seq-invariant: depend only on weights + KV per-token)
    mem_total_bytes: float
    usable_bytes: float
    fits: bool
    usable_for_tokens: float
    unique_tokens_multiplier: int
    kv_replication_factor: float
    # GPU peaks (rooflines used by phases + overlay)
    peak_flops: Optional[float]
    peak_bw: Optional[float]
    peak_intra_bw: Optional[float]
    peak_inter_bw: Optional[float]
    gpus_per_node_n: int
    intra_bw_label_s: str
    inter_bw_label_s: str
    peak_source_s: str
    #: The request's link latency and launch floor, in
    #: seconds (`CapacityKnobs.comm_latency_us` / `launch_floor_us`).
    comm_latency_s: float
    launch_floor_s: float
    # Other invariant outputs surfaced in the response
    roles_present: List[str]
    shape_warnings: List[Dict[str, Any]]
    #: The model's published context length (dim `s_max`), in positions;
    #: None when the model publishes none.
    max_context: Optional[int]
    # Overlay context — resolved once. `overlay_walk` consumes it verbatim
    # instead of re-running `get_binding` per seq. The measurements are
    # looked up per estimate, for the keys its overlays ask
    # (`op_calibration.registry_for` under `measurement_scope`).
    framework: str
    framework_binding: Any
    measurement_scope: Any
    calibration_rows: Optional[list]
    strategy: str
    enable_measured_lookup: bool
    selection_mode: str
    kernel_mix: Optional[dict]
    #: The paged-KV page this deployment allocates the cache in, resolved
    #: once (the request's, else the framework binding's default). Part of
    #: the key of every kernel that reads a paged cache.
    kv_block_size: int
    # Workload knobs (seq-invariant; threaded into finish/overlay)
    s_prefix_cached: int
    s_verify: int
    acceptance_rate: float
    #: The batch's routing skew, max/mean expert load (`Workload.
    #: ep_load_factor`).
    ep_load_factor: Any
    # The speculative draft this decode step counts, or None for the verify
    # forward alone (`speculative_draft.SpeculativeDraft`).
    draft: Optional[SpeculativeDraft] = None
    #: Why this preparation's weights cannot load, or None when they can. The
    #: preparation owns the refusal, so every estimate consumer refuses before
    #: a phase walk instead of remembering a separate preflight.
    refusal_reason: Optional[str] = None
    #: The walks every step of this preparation shares, across all the
    #: points of a sweep (`step_eval.StepCells`).
    cells: StepCells = field(default_factory=StepCells)

    def fabric(self) -> Fabric:
        """This card's interconnect at this deployment's link latency
        (`Card.fabric`, which states why it has one home): the other half
        of what a card contributes to a preparation (`CardPrice`, which
        states the first)."""
        return self.card.fabric(self.comm_latency_s)

    def verify_widths(self) -> Dict[str, int]:
        """The verify width each per-op row list's step runs at: the
        backbone's prefill and decode steps the request's (`s_verify`), a
        counted draft's forward its own (`SpeculativeDraft.
        tokens_per_forward`) — the widths `_capacity_pending` walks them
        at, and the ones a point of this preparation reads them at
        (`_point_rows`)."""
        widths = {"per_op_timing": self.s_verify,
                  "per_op_timing_decode": self.s_verify}
        if self.draft is not None:
            widths["per_op_timing_draft"] = self.draft.tokens_per_forward
        return widths

    def walks_alike(self, other: "_CapacityPreparation") -> bool:
        """Whether this preparation's steps are the same walks as `other`'s
        — so its points read `other`'s rows at its own verify widths by
        substitution (`batch_sweep.rows_at`): every row list's width walked
        the same way (`Workload.walked_as_number`: a width of 1 is the
        number, every wider one the symbol), and a width walked as the
        number the same number. A speculative step at K draft tokens and at
        any other K ≥ 1 are one walk."""
        def walks(prep: "_CapacityPreparation") -> Dict[str, Optional[int]]:
            # A width walked as a number is its own walk; None is the symbol.
            return {name: (width if Workload.walked_as_number("s_verify", width)
                           else None)
                    for name, width in prep.verify_widths().items()}
        return walks(self) == walks(other)


@dataclass(frozen=True)
class CapacityKnobs:
    """The seq-invariant request every capacity entry point shares.

    `estimate_capacity` / `estimate_capacity_sweep` take this object.
    The flat kwargs those functions still accept are a compatibility
    shell that packs into one of these; new callers pass `knobs=`.
    Sequence length and per-phase batch stay off this type — they are
    the per-point axes, not the deployment.
    """
    #: The precision profile (`precision.PrecisionProfile`), or its request
    #: form (`{"preset": id}` / `{"canonical": {...}}`). None is as shipped:
    #: every weight, edge and cache slot at the precision the model ships.
    precision: Any = None
    gpu_memory_fraction: float = 0.9
    #: The GPU count a caller states, or None. A deployment occupies the
    #: ranks its parallelism spans (`ParallelConfig.cluster_size`); the
    #: response reports that, and nothing is priced off this field.
    num_gpus: Optional[int] = None
    parallel: Optional[ParallelConfig] = None
    published_weight_bytes: Optional[float] = None
    tensor_patterns_meta: Optional[dict] = None
    s_prefix_cached: int = 0
    #: The verify forward's width, K + 1 for K draft tokens
    #: (`phases.Speculation`); 1 is plain autoregressive decode.
    s_verify: int = 1
    acceptance_rate: float = 1.0
    #: The batch's routing skew: the hottest expert's rows over the mean
    #: expert's (≥ 1, from a routing trace). A workload input the walk reads
    #: as the symbol `rho`; 1 is balanced routing, and the response says
    #: it is an assumption (`assumptions.deployment_assumptions`).
    ep_load_factor: float = 1.0
    #: Count the speculative draft's forwards in the decode step, in this
    #: style (`speculative_draft.DraftStyle`); None prices the verify alone.
    speculative_draft: Optional[str] = None
    include_aux_weights: bool = False
    #: Keep the weights the spec declares a framework can hold in host
    #: memory (`Program.host_offloadable`: vLLM's Engram `cpu_offload` of
    #: the n-gram tables, on by default, `VLLM_PLE_CPU_OFFLOAD=1`) there
    #: rather than on the GPU. Residency only: the rows a step gathers are
    #: still the step's reads.
    host_offload: bool = True
    #: One link traversal's latency and the shortest one kernel
    #: launch takes, in µs (`defaults`): the floors a small step pays.
    comm_latency_us: float = COMM_LATENCY_US_DEFAULT
    launch_floor_us: float = LAUNCH_FLOOR_US_DEFAULT
    model_id: str = ""
    #: A pre-resolved spec. When set, capacity uses it directly — matching
    #: happens once, at the caller's single entry point, never re-run here.
    #: When None, capacity falls back to resolving `model_id` itself.
    resolved: Any = None
    selection_mode: str = "theoretical_only"
    framework: Optional[str] = None
    kernel_mix: Optional[dict] = None
    #: The paged-KV page the request deploys with. None takes the framework
    #: binding's own default — the framework owns the cache manager, so it
    #: owns the default, and no call site repeats a number.
    kv_block_size: Optional[int] = None
    #: Rows a headless caller posts with the request (validated, bounded at
    #: the API), over the looked-up ones for this estimate only.
    calibration_rows: Optional[list] = None
    #: What the measured numbers are read under
    #: (`measure.scope.MeasurementScope`, resolved by the API: the owner, the
    #: lake and its snapshot, the annotation rules). None reads none.
    measurement_scope: Any = None
    #: Server options the deployment sets on the framework
    #: (`{"enable_fused_qk_norm_rope": true}`); an unset flag keeps the
    #: framework's default. String choices such as vLLM's
    #: `linear_backend=auto` are first-class values.
    framework_options: Optional[Dict[str, Union[bool, str, int]]] = None
    #: The media each request carries (`Workload.media`), asking for the
    #: encode phase: the response then carries `encode`, each encoder over
    #: `encode_batch_size` requests (the prefill batch when None), None for
    #: a model with no encoder (`encode_phase.plan_encode`). None asks
    #: for no encode phase (a sweep's points). No language-model number
    #: reads it.
    media: Optional[Dict[str, Any]] = None
    encode_batch_size: Optional[int] = None

    @classmethod
    def from_kwargs(cls, ns: Mapping[str, Any], *,
                    envelope: FrozenSet[str] = frozenset()) -> "CapacityKnobs":
        """Pack a forwarding dict into knobs. `envelope` names the non-knob
        keys the caller knows it carries alongside (the request's own axes:
        `config`, `avg_input_tokens`, a sweep's `gpu_skus`). Anything else
        is refused: a misspelled knob that silently fell back to its
        default was an estimate nobody asked for."""
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(ns) - names - set(envelope))
        if unknown:
            raise TypeError(
                "unexpected capacity argument(s): " + ", ".join(unknown)
                + f" (knobs are {sorted(names)}; envelope keys {sorted(envelope)})")
        return cls(**{k: v for k, v in ns.items() if k in names})

    def as_flat(self) -> Dict[str, Any]:
        """Knob fields as a dict, without recursing into ParallelConfig."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _require_knobs(
    knobs: Optional[CapacityKnobs],
    kwargs: Mapping[str, Any],
) -> CapacityKnobs:
    """One CapacityKnobs from either the object or the flat kwargs, not both."""
    names = {f.name for f in fields(CapacityKnobs)}
    if knobs is not None:
        if kwargs:
            raise TypeError("pass CapacityKnobs or the flat fields, not both")
        return knobs
    extra = [k for k in kwargs if k not in names]
    if extra:
        raise TypeError(
            "unexpected capacity argument(s): " + ", ".join(sorted(extra)))
    return CapacityKnobs.from_kwargs(kwargs)


def _capacity_walk(resolved: Any, parallel: ParallelConfig,
                   precision: Any = None, cells: Any = None) -> Any:
    """The one walk every budget scalar descends from.

    Prefill geometry at a token length (the workload-free deployment
    question); per-rank by construction, so nothing below divides by a
    world afterwards. The dummy length is 128, rounded up to a multiple
    of `cp` — embedding requires the token extent to divide the sequence
    world, and weights do not care which multiple we picked.
    """
    from .step_eval import evaluate_step

    tokens = 128
    world = max(1, int(parallel.cp or 1))
    if tokens % world:
        tokens = ((tokens + world - 1) // world) * world
    return evaluate_step(
        resolved,
        parallel,
        phase="prefill",
        batch=1,
        input_tokens=tokens, output_tokens=0,
        precision=precision, cells=cells,
    ).analysis


def _cache_room(usable_bytes: float, weight_bytes: float, static: float,
                constant_state: float) -> Tuple[bool, float]:
    """Whether a rank's length-independent cache floors fit beside its
    weights, and the bytes then left for cache that grows with length."""
    kv_budget_bytes = usable_bytes - weight_bytes
    fits = kv_budget_bytes > static + constant_state
    return fits, max(0.0, kv_budget_bytes - static - constant_state)


def _preparation_fields(price: CardPrice, *, model_dtype: str,
                        weight_bytes_per_rank: float, static: float,
                        constant_state: float) -> Dict[str, Any]:
    """The preparation fields a card's price decides: the card, its memory,
    the room left for growing cache beside the weights and floors of the
    stage the preparation describes (`_cache_room`), and its peak rates at
    the model's own dtype (`CardPrice.rooflines`; every per-op row still
    reads its own leaf's column).

    The join of a card's price with the deployment's weights, so it belongs
    to neither: `CardPrice` is a fact of the card at an HBM fraction, and
    how those bytes divide between weights and cache is a fact of this
    deployment. Stated once because it is asked twice — when a preparation
    is built on a card (`_capacity_preparation`) and when a prepared walk
    is moved to another card of its class (`_restamp_gpu`). The second
    statement of this arithmetic is how one card comes to be priced two
    ways.
    """
    fits, usable_for_tokens = _cache_room(
        price.usable_bytes, weight_bytes_per_rank, static, constant_state)
    gr = price.rooflines(model_dtype)
    return dict(
        card=price.card,
        gpu=price.gpu,
        gpu_sku=price.gpu_sku,
        mem_total_bytes=price.mem_total_bytes,
        usable_bytes=price.usable_bytes,
        fits=fits,
        usable_for_tokens=usable_for_tokens,
        peak_flops=gr.peak_flops,
        peak_bw=gr.peak_bw,
        peak_intra_bw=gr.peak_intra_bw,
        peak_inter_bw=gr.peak_inter_bw,
        gpus_per_node_n=gr.gpus_per_node,
        intra_bw_label_s=gr.intra_bw_label,
        inter_bw_label_s=gr.inter_bw_label,
        peak_source_s=gr.peak_source,
    )


def _restamp_gpu(p: _CapacityPreparation, card: Card) -> _CapacityPreparation:
    """Same walk and weights, a different card's price
    (`_preparation_fields`), on a card of this walk's own class
    (`Card.refuse_unless_one_class`, which states why).
    """
    p.card.refuse_unless_one_class(card)
    return replace(p, **_preparation_fields(
        card.price(gpu_memory_fraction=p.gpu_memory_fraction),
        model_dtype=p.resolved_precision.model_dtype,
        weight_bytes_per_rank=p.weight_bytes_per_rank,
        static=p.kv_static_per_rank,
        constant_state=p.kv_constant_state_per_rank))


def _point_batches(ceilings: CapacityCeilings, *,
                   prefill_batch_override: Optional[int] = None,
                   decode_batch_override: Optional[int] = None) -> Dict[str, int]:
    """The batch each phase of a restamped point runs at, by row list: a
    caller-pinned per-GPU batch clamped to this card's capacity, or the
    card's own ceiling (None / "max"). Without the override every point used
    to rescale to the max, so `estimate-parallel-sweep` returned identical
    throughput for a fixed batch and "max"."""
    pre_cap = max(1, int(ceilings.max_prefill_batch or 1))
    dec_cap = max(1, int(ceilings.max_batch or 1))
    pre_b = (min(pre_cap, max(1, int(prefill_batch_override)))
             if prefill_batch_override else pre_cap)
    dec_b = (min(dec_cap, max(1, int(decode_batch_override)))
             if decode_batch_override else dec_cap)
    return {"per_op_timing": pre_b, "per_op_timing_decode": dec_b,
            "per_op_timing_draft": dec_b}


def _direct_batches(ceilings: CapacityCeilings, *,
                    prefill_batch_override: Optional[int] = None,
                    decode_batch_override: Optional[int] = None) -> Dict[str, int]:
    """The batch each phase runs at under a direct estimate's rule
    (`_capacity_pending`), by row list: prefill the override (`-1` its
    activation-bound ceiling) or 1; decode the override or the KV-bound
    ceiling; a phase is never walked below one request."""
    pre = (ceilings.max_prefill_batch if prefill_batch_override == -1
           else int(prefill_batch_override) if prefill_batch_override is not None
           else 1)
    dec = (int(decode_batch_override) if decode_batch_override is not None
           else ceilings.max_batch)
    pre, dec = max(1, int(pre or 1)), max(1, int(dec or 1))
    return {"per_op_timing": pre, "per_op_timing_decode": dec,
            "per_op_timing_draft": dec}


#: What a result states of its STEPS beside its phases and row lists
#: (`PendingEstimate.finish`): the fusion delta, a step time per phase, and
#: the encode step — its stacks' rows, module tables and times. A card that
#: admits no request runs none of them (`without_step`).
_STEP_ONLY = ("vs_naive_delta", "encode")


def without_step(result: dict, row_lists: Iterable[str]) -> dict:
    """`result` as a card that admits no request publishes it.

    A deployment whose weights do not fit (`infeasibility_reason`) runs no
    step at any batch, so there is no batch to state and nothing measured
    at one: each phase is only its name and why it has no step, and the
    per-op rows (`row_lists`), the module tables and the other steps'
    results (`_STEP_ONLY`: the fusion delta, the encode step) are not
    published. The walk still ran — at one request, the point every
    other card and batch is re-priced from (`_point_rows`) — and that walk
    is kept by whoever holds it; it is not this card's step, and publishing
    its rows quoted a B=1 step the card cannot run."""
    reason = result.get("infeasibility_reason")
    if not reason:
        return result
    out = {k: v for k, v in result.items()
           if k not in set(row_lists) and k not in _STEP_ONLY}
    out["phases"] = {name: {"phase": phase.get("phase", name), "batch": None,
                            "time_unpublished_reason": reason}
                     for name, phase in (result.get("phases") or {}).items()
                     if phase}
    return out


def _point_rows(base: dict, p_dst: _CapacityPreparation, batches: Mapping[str, int],
                pending: "PendingEstimate", point: Point, *,
                breakdown: bool) -> Dict[str, list]:
    """`base`'s per-op rows on `p_dst`'s card at the point `point` moves
    them to (`{}` for the walk's own; a sweep's context, `{"s_input": L}`),
    at each row list's batch (`_point_batches`) and verify width
    (`_CapacityPreparation.verify_widths`): each leaf's peak re-read
    for its own `act_dtype`,
    then every row's law and shape law substituted there
    (`batch_sweep.rows_at`) — the collective re-priced on
    this card's fabric. A card with no column for a leaf's dtype runs that
    matmul the way the framework does — weight-only, at the card's own
    widest column (`pipeline.leaf_peak`) — which is what a zero peak means
    here: no row peak, so `RowRoofline` takes the card's. A width the
    deployment CHOSE was already refused where the estimate was built.

    `p_dst`'s card must be one the walk may be re-priced on
    (`PendingEstimate.priced_on`, the one road from a walk to a card's
    numbers). This is the pairing `_restamp_gpu` cannot see: it checks the
    card a PREPARATION moves to, and the rows moved here come from
    `pending`'s walk, which is a different fact. Handed a walk's rows and
    another class's card, this function would substitute happily and
    publish the walk card's row LIST — which ops ran, and in which
    format — as the destination card's.

    The rows carry no measurement key: a reader that looks measurements up
    asks for them (`PendingEstimate.keyed`)."""
    pending.priced_on(p_dst)
    fabric = p_dst.fabric()
    launch_s = p_dst.launch_floor_s
    widths = p_dst.verify_widths()
    out: Dict[str, list] = {}
    for key, batch in batches.items():
        if not base.get(key):
            # A row list the estimate has none of (no draft is counted).
            out[key] = []
            continue
        rows = []
        for r in base.get(key) or []:
            adt = r.get("act_dtype")
            peak = (leaf_peak(p_dst.gpu, adt)
                    if adt and p_dst.gpu is not None else None)
            # A point's rows are keyed at the point (another card, another
            # batch): they carry no measurement until the sweep's lookup.
            rows.append({**{k: v for k, v in r.items()
                            if k not in ("measured_state", "measured_key",
                                         "measured_reason")},
                         "t_measured_s": None,
                         "peak_flops_per_sec_for_op": float(peak) if peak else 0.0})
        out[key] = rows_at(
            rows, Point({**point.counts, "B": batch, "s_verify": widths[key]}),
            cell=pending.walk_of_rows[key].analysis, peak_flops=p_dst.peak_flops,
            peak_bw=p_dst.peak_bw, fabric=fabric, breakdown=breakdown,
            launch_s=launch_s)
    return out

#: The row and phase columns that add across repeated launches, derived from
#: the producer's one declaration (`row_roofline.COLUMNS`).
_ROW_ADDITIVE_COLUMNS = column_names(published_by=ROW, additive=True)
_PHASE_LAUNCH_TOTALS = column_names(published_by=PHASE, additive=True)


def _freeze_launch_shape(value: Any) -> Any:
    """Make a published row shape hashable without reinterpreting it."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_launch_shape(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(
            (key, _freeze_launch_shape(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    return value


def _row_launch_key(row: Mapping[str, Any], row_index: int) -> Tuple[Any, ...]:
    """A SOL row's exact launch identity at a substituted point.

    This is not a second measurement identity: it uses the graph row plus the
    shape and dtype fields that row already publishes.  Its sole purpose is to
    let the chunk-term merger decide which launches are physically identical.
    """
    identity = row.get("uid")
    if identity is None:
        identity = ("unkeyed", row_index,
                    row.get("aggregation_id"), row.get("display_path"))
    return (
        identity, row.get("op"), row.get("shape_json"),
        _freeze_launch_shape(row.get("shape_in")),
        _freeze_launch_shape(row.get("shape_out")), row.get("act_dtype"),
        row.get("dtype_env_json"), bool(row.get("is_comm")),
    )


def _fold_chunk_sol_rows(
        endpoint_rows: List[dict], chunk_rows: List[List[dict]],
        schedule: PrefillChunkSchedule, *, gpu_sku: str, framework: str,
) -> Tuple[List[dict], int]:
    """Fold endpoint SOL rows over the workload's materialized launches."""
    endpoint_batch = schedule.endpoint_batch
    records: List[Tuple[ChunkLaunchTerm, int, Mapping[str, Any]]] = []
    records_by_key: Dict[Tuple[Any, ...], List[Tuple[int, Mapping[str, Any]]]] = {}
    for chunk, rows in zip(schedule.chunks, chunk_rows, strict=True):
        launch_batch = int(chunk.launch_batch)
        if endpoint_batch < 1 or launch_batch < 1 or endpoint_batch % launch_batch:
            raise ValueError(
                "endpoint batch must be a whole number of chunk launch batches "
                f"(endpoint={endpoint_batch}, launch={launch_batch})")
        repeat = endpoint_batch // launch_batch
        for row_index, row in enumerate(rows):
            inputs = {column: row.get(column)
                      for column in _ROW_ADDITIVE_COLUMNS}
            records.append((
                ChunkLaunchTerm(
                    key=_row_launch_key(row, row_index),
                    launches=float(repeat),
                    formula_weight=1,
                    zero_cost=bool(row.get("zero_cost"))
                    or all(not value for value in inputs.values()),
                    inputs=inputs, context=(gpu_sku, framework), cadence=1),
                row_index, row))
        for term, row_index, row in records[-len(rows):]:
            records_by_key.setdefault(term.key, []).append((row_index, row))

    merged = merge_chunk_launch_terms(term for term, *_ in records)
    totals = [{**{column: 0 for column in _ROW_ADDITIVE_COLUMNS},
               "count_per_step": 0} for _ in endpoint_rows]
    for out in merged:
        group = records_by_key[out.key]
        if not group:
            raise ValueError("a merged SOL chunk term lost its source provenance")
        row_index, row = group.pop(0)
        if row_index >= len(totals):
            raise ValueError("a chunk SOL term moved to another graph row")
        total = totals[row_index]
        for column in _ROW_ADDITIVE_COLUMNS:
            total[column] += float(row.get(column) or 0) * float(out.launches)
        total["count_per_step"] += (
            int(row.get("count_per_step") or 0) * int(out.launches))

    out_rows: List[dict] = []
    for row, total in zip(endpoint_rows, totals, strict=True):
        folded = {
            **row, **total, "t_measured_s": None,
            "arithmetic_intensity": arithmetic_intensity(
                total["flops_per_rank"], total["access_bytes_per_rank"]),
        }
        folded["t_sol_s"] = folded["t_op_s"]
        out_rows.append(folded)
    return out_rows, len(merged)


def _sum_phase_launches(phases: List[Mapping[str, Any]],
                        *, repeat: int) -> Dict[str, Any]:
    """The per-launch phase totals, added — only those. The endpoint's own
    workload (its batch, `binding`, note) is the caller's phase and stays
    put: a chunk's copy names one launch (`launch_batch`, often 1), and
    folded over it the endpoint's throughput counted one request's tokens
    against every request's time."""
    out: Dict[str, Any] = {}
    for column in _PHASE_LAUNCH_TOTALS:
        out[column] = sum(float(phase[column] or 0) for phase in phases) * repeat
    bounds: Dict[str, float] = {}
    for phase in phases:
        for bound, value in (phase.get("t_sol_by_bound") or {}).items():
            bounds[bound] = bounds.get(bound, 0.0) + float(value or 0)
    for bound in bounds:
        bounds[bound] *= repeat
    out["t_sol_by_bound"] = bounds
    out["t_sol_by_bound_share"] = RowRoofline.shares(bounds)
    out["arithmetic_intensity"] = arithmetic_intensity(
        out["flops_per_step"], out["total_bytes"])

    pipeline = phases[0].get("pipeline_time")
    if pipeline is not None:
        per_stage = [
            sum(float(phase["pipeline_time"]["per_stage"][index] or 0)
                for phase in phases) * repeat
            for index in range(len(pipeline["per_stage"]))]
        out["pipeline_time"] = {
            **pipeline, "per_stage": per_stage,
            "t_compute": out["t_compute_s"], "t_memory": out["t_memory_s"],
            "t_comm": out["t_comm_s"], "t_sol": out["t_sol_s"],
            "t_sol_by_bound": dict(bounds)}
    return out


def _chunk_point_rows_and_phase(
        base: Mapping[str, Any], phase: Mapping[str, Any],
        p: _CapacityPreparation, pending: "PendingEstimate", chunk: Any, *,
        verify_width: int) -> Tuple[List[dict], Dict[str, Any]]:
    """The one structural walk read at one scheduler launch, then phase-summed."""
    rows = _point_rows(
        dict(base), p, {"per_op_timing": int(chunk.launch_batch)}, pending,
        Point({"s_input": int(chunk.end), "s_prefix": int(chunk.start)}),
        breakdown=False)["per_op_timing"]
    point = Point({
        "B": int(chunk.launch_batch), "s_input": int(chunk.end),
        "s_prefix": int(chunk.start), "s_verify": int(verify_width)})
    at_phase = phase_at(
        dict(phase), rows, point, p.peak_flops, p.peak_bw, p.fabric(),
        cell=pending.walk_of_rows["per_op_timing"].analysis,
        breakdown=False, launch_s=p.launch_floor_s)
    return rows, at_phase


def _rollup_module_flops(modules: List[dict]) -> None:
    """Recompute module-tree FLOPs after each leaf became a launch sum."""
    parents = {
        index for index, _row in enumerate(modules)
        if index + 1 < len(modules)
        and int(modules[index + 1].get("depth") or 0)
        > int(_row.get("depth") or 0)}
    parents.update(index for index, row in enumerate(modules)
                   if row.get("uid") is None)
    for index in parents:
        modules[index]["flops_per_rank"] = 0
        modules[index]["flops_full"] = 0
    stack: List[Tuple[int, int]] = []
    for index, row in enumerate(modules):
        depth = int(row.get("depth") or 0)
        while stack and stack[-1][1] >= depth:
            stack.pop()
        rank_flops = float(row.get("flops_per_rank") or 0)
        full_flops = float(row.get("flops_full") or 0)
        if rank_flops or full_flops:
            for parent, _depth in stack:
                modules[parent]["flops_per_rank"] = (
                    modules[parent].get("flops_per_rank") or 0) + rank_flops
                modules[parent]["flops_full"] = (
                    modules[parent].get("flops_full") or 0) + full_flops
        stack.append((index, depth))


def _fold_chunk_module_rows(
        modules: List[dict], endpoint_rows: List[dict],
        folded_rows: List[dict]) -> List[dict]:
    """Make the module table describe the same launch sums as its leaves."""
    by_uid = {
        row.get("uid"): folded
        for row, folded in zip(endpoint_rows, folded_rows, strict=True)
        if row.get("uid") is not None}
    endpoint_by_uid = {row.get("uid"): row for row in endpoint_rows
                       if row.get("uid") is not None}
    out: List[dict] = []
    for module in modules:
        row = dict(module)
        folded = by_uid.get(row.get("uid"))
        if folded is not None:
            endpoint_leaf = endpoint_by_uid[row.get("uid")]
            for column in (*_ROW_ADDITIVE_COLUMNS, "count_per_step"):
                if column in row:
                    row[column] = folded[column]
            endpoint_flops = float(endpoint_leaf.get("flops_per_rank") or 0)
            if endpoint_flops:
                row["flops_full"] = round(
                    float(row.get("flops_full") or 0)
                    * float(folded["flops_per_rank"]) / endpoint_flops)
        else:
            for column in _ROW_ADDITIVE_COLUMNS:
                if column in row:
                    row[column] = 0
            if row.get("uid") is None:
                row["flops_per_rank"] = 0
                row["flops_full"] = 0
        out.append(row)
    _rollup_module_flops(out)
    return out


def _endpoint_memo_key(kind: str, p: _CapacityPreparation,
                       schedule: PrefillChunkSchedule) -> Tuple[Any, ...]:
    """What one point's chunked prefill fold is a function of, as a key.

    The schedule states the endpoint — its length, cached prefix and batch —
    and the launches it decomposes into, but NOT the verify width those
    launches are priced at: a schedule's endpoint is materialized at
    `s_verify=1` whatever width the request asks for
    (`CapacitySweepPlan.of`), while the fold reads the rows and sums the
    phase at the preparation's width (`_CapacityPreparation.verify_widths`,
    `_chunk_point_rows_and_phase`). So the widths and the card are named
    here beside the schedule.

    Within one sweep cell the widths are also what names the preparation
    (`_sweep_cell.preparations` is that dict), and the rest of what the fold
    reads — the walk, the ceilings and the anchor — is a function of the
    preparation, the length and the prefix. A memo keyed this way is
    therefore sound for every point of one cell, not only for the points of
    one context: two contexts of the same length and prefix at different
    draft K (`gallery_frontier._draft_walk` builds exactly those) key apart.
    """
    return (kind, p.gpu_sku, tuple(sorted(p.verify_widths().items())), schedule)


def _fold_chunked_prefill_sol(
        result: dict, p: _CapacityPreparation, pending: "PendingEstimate",
        schedule: PrefillChunkSchedule,
        endpoints: Optional[Dict[Any, Any]] = None) -> dict:
    """Publish the SOL endpoint as the sum of its materialized chunk walks.

    The fold reads `result`'s prefill rows and prefill phase, `p`, `pending`
    and `schedule`. `endpoints` is a caller's memo of it, keyed by the
    endpoint and the widths it is priced at (`_endpoint_memo_key`), for
    points whose prefill rows and phase are a function of those
    (`_sweep_cell`: the points of one cell, on one walk and card), so points
    that differ only in their decode batch fold once."""
    if len(schedule.chunks) <= 1 or not result.get("per_op_timing"):
        return result
    phase = (result.get("phases") or {}).get("prefill")
    if not phase:
        return result
    if phase.get("time_unpublished_reason"):
        return result
    memo_key = (None if endpoints is None
                else _endpoint_memo_key("sol", p, schedule))
    if memo_key is not None and memo_key in endpoints:
        # Each point its own list and phase: a measured track is written
        # into a point's phase (`batch_sweep.apply_measured`). The copies
        # are shallow, so the row dicts and the phase's nested values ARE
        # SHARED by every point that reads this endpoint: a writer that
        # edited a row or a nested phase value in place would write into
        # its sibling points. Every writer today rebinds instead
        # (`batch_sweep._measured_row` returns `{**row}`, `apply_measured`
        # rebinds the keys it sets); keep it that way, or copy deeper here.
        folded_rows, folded_phase = endpoints[memo_key]
        return {
            **result, "per_op_timing": list(folded_rows),
            "phases": {**result.get("phases", {}), "prefill": dict(folded_phase)},
        }
    endpoint_rows = list(result["per_op_timing"])
    chunk_rows: List[List[dict]] = []
    chunk_phases: List[Dict[str, Any]] = []
    repeats: List[int] = []
    verify_width = p.verify_widths()["per_op_timing"]
    for chunk in schedule.chunks:
        rows, at_phase = _chunk_point_rows_and_phase(
            result, phase, p, pending, chunk, verify_width=verify_width)
        chunk_rows.append(rows)
        chunk_phases.append(at_phase)
        repeats.append(schedule.endpoint_batch // int(chunk.launch_batch))
    if len(set(repeats)) != 1:
        raise ValueError("a schedule changed endpoint repetition between chunks")

    walk = pending.walk_of_rows["per_op_timing"]
    folded_rows, term_count = _fold_chunk_sol_rows(
        endpoint_rows, chunk_rows, schedule,
        gpu_sku=p.gpu_sku, framework=walk.deployment.framework)
    folded_phase = {
        **phase,
        **_sum_phase_launches(chunk_phases, repeat=repeats[0]),
    }
    folded_phase["batch"] = schedule.endpoint_batch
    folded_phase["measured_rows"] = 0
    folded_phase["total_rows"] = term_count
    folded_phase["t_measured_s"] = None
    folded_phase["coverage"] = {"measured": 0, "total": term_count}
    folded_phase["prefill_chunk_map"] = dict(schedule.chunk_map() or {})
    if "flops_by_module" in folded_phase:
        folded_phase["flops_by_module"] = _fold_chunk_module_rows(
            list(phase.get("flops_by_module") or []), endpoint_rows, folded_rows)
        enrich_module_breakdown(folded_phase)
    if "throughput" in folded_phase:
        folded_phase["throughput"] = phase_throughput(
            folded_phase,
            unique_tokens_multiplier=folded_phase["throughput"]["groups"],
            num_gpus=folded_phase["throughput"]["num_gpus"])

    row_total = sum(float(row.get("t_op_s") or 0) for row in folded_rows)
    pipeline = folded_phase.get("pipeline_time")
    phase_total = (float(folded_phase["t_sol_s"]) if pipeline is None
                   else float(pipeline["per_stage"][pipeline["stage"]]))
    if abs(row_total - phase_total) > max(1e-18, abs(row_total) * 1e-12):
        raise ValueError(
            "chunked prefill SOL rows and phase disagree: "
            f"rows={row_total}, phase={phase_total}")
    if memo_key is not None:
        endpoints[memo_key] = (list(folded_rows), dict(folded_phase))
    return {
        **result, "per_op_timing": folded_rows,
        "phases": {**result.get("phases", {}), "prefill": folded_phase},
    }


def _point_measures_with_chunks(
        rows: Mapping[str, List[dict]], keyed: Optional[Mapping[str, list]],
        p: _CapacityPreparation,
        pending: "PendingEstimate", phases: Mapping[str, Any], *,
        schedule: PrefillChunkSchedule,
        endpoints: Optional[Dict[Any, Dict[str, Any]]] = None
        ) -> Dict[str, Dict[str, Any]]:
    """A point's measures, with a long prefill read at its explicit chunks.

    Every chunk is substituted from the same structural walk and keyed at its
    own launch point.  The endpoint fold reads only those keys; the endpoint's
    old whole-sequence measure, computed for compatibility with decode/draft
    fields, is replaced before it can enter a selection union.

    The card is the POINT's, read off `p` alone: `p` is the preparation the
    point is priced on, so the bandwidth that prices its chunk rows and the
    card their keys are stamped with are one card's. A walk is shared across a
    card series, so a SKU passed beside the preparation could disagree with it.
    `endpoints` is a caller's memo of the folded prefill, keyed as
    `_fold_chunked_prefill_sol`'s is (`_endpoint_memo_key`).
    """
    if keyed is None:
        return {}
    gpu_sku = p.gpu_sku
    out = point_measures(pending, rows, keyed, phases, gpu_sku=gpu_sku)
    chunk_map = schedule.chunk_map()
    if (not chunk_map or len(chunk_map.get("chunks") or ()) < 2
            or "prefill" not in out):
        return out
    memo_key = (None if endpoints is None
                else _endpoint_memo_key("measures", p, schedule))
    if memo_key is not None and memo_key in endpoints:
        out["prefill"] = dict(endpoints[memo_key])
        return out
    chunk_measures = []
    for chunk in chunk_map["chunks"]:
        # Every chunk is read off this point's own rows (the laws are the
        # point's; a chunk is where they are substituted), not off the
        # previous chunk's.
        chunk_rows = _point_rows(
            rows, p,
            {"per_op_timing": int(chunk["launch_batch"])}, pending,
            Point({"s_input": int(chunk["end"]),
                   "s_prefix": int(chunk["start"])}),
            breakdown=False)
        chunk_rows, chunk_keys = pending.keyed_at(chunk_rows, gpu_sku=gpu_sku)
        chunk_point = point_measures(
            pending, chunk_rows, chunk_keys, phases, gpu_sku=gpu_sku,
            include_cadence=True)
        if "prefill" not in chunk_point:
            raise ValueError("a prefill chunk lost its walked prefill step")
        chunk_measures.append(chunk_point["prefill"])
    folded = fold_chunk_point_measures(chunk_map, chunk_measures)
    out["prefill"] = {**folded, "phase": "prefill",
                      "prefill_chunk_map": dict(chunk_map)}
    if memo_key is not None:
        endpoints[memo_key] = dict(out["prefill"])
    return out


def _stamp_gpu_on_result(
    base: dict,
    p_dst: _CapacityPreparation,
    ceilings: CapacityCeilings,
    walked_on: Card,
    *,
    batches: Mapping[str, int],
    rows: Mapping[str, list],
    walks: Mapping[str, Any],
    point: Point,
    breakdown: bool,
) -> dict:
    """Same layout's per-op components, a different card's HBM and peaks.

    The graph walk and the two phase evaluations do not mention the SKU.
    Max-batch and t_sol are arithmetic over those components plus this
    card's rooflines — no second evaluate_step.

    `ceilings` are this card's (`capacity_ceilings(p_dst, ...)`). Under pp
    they bind a stage, and the rows in `base` must be that stage's: a card
    with less HBM can run out on a different stage than the card `base` was
    estimated on, and the rows of one stage under another stage's batch are
    no rank's answer. The caller picks a `base` per bound stage; a mismatch
    here is refused rather than re-priced.

    `walked_on` is the walk's IDENTITY — whose class its rows are — and is
    deliberately not the laws source below. It asks the class here
    (`Card.refuse_unless_one_class`); the destination ceilings ask the stage
    (`CapacityCeilings.refuse_unless_one_stage`). A caller that supplies rows
    directly therefore cannot bypass the class road.

    `batches` is each phase's batch at this point (`_point_batches`,
    `_direct_batches`), `rows` the per-op rows there (`_point_rows`),
    `walks` the walk each row list's LAWS are read from
    (`PendingEstimate.walk_of_rows`) and `point` the workload counts the
    point moves beside the batch (`_point_rows`).

    `walks` is a parameter and not read off a `PendingEstimate`, which it
    briefly was: a caller may legitimately supply a laws source that is
    not a walk object. A published response's rows carry their laws as
    TEXT, and re-reading them needs a book of their own and no walk
    (`row_roofline.LawBook`, `tests/sweep_parity.response_cell`) — which is
    the property that lets a response be re-priced in a process that never
    walked anything. The walk's IDENTITY (whose class, whose stage) and the
    laws' SOURCE are two facts, and merging them loses that one. The phases are summed from them, unmeasured — the
    measured track is applied over the sweep's one lookup
    (`batch_sweep.apply_measured`) — and the result carries them.
    `breakdown` as `batch_sweep.phase_at`'s.
    """
    _c = ceilings
    walked_on.refuse_unless_one_class(p_dst.card)
    _c.refuse_unless_one_stage(base["memory"]["pp_stage"], p_dst.gpu_sku)

    fabric = p_dst.fabric()
    launch_s = p_dst.launch_floor_s

    def _on_card(rows: list) -> list:
        """Each leaf's peak re-read on this card for its own `act_dtype`
        (`_point_rows`)."""
        out = []
        for r in rows:
            adt = r.get("act_dtype")
            peak = (leaf_peak(p_dst.gpu, adt)
                    if adt and p_dst.gpu is not None else None)
            out.append({**r, "peak_flops_per_sec_for_op":
                        float(peak) if peak else 0.0})
        return out

    def _phase_on_card(phase: dict) -> dict:
        """The phase as this card states it: whether a step runs at all is
        this card's capacity verdict (`_c.infeasibility_reason`), not the
        card the walk ran on; under pp, every stage's row laws read this
        card's peaks too."""
        if not phase:
            return phase
        phase = {**phase,
                 "time_unpublished_reason": _c.infeasibility_reason or ""}
        pipeline = phase.get("pipeline_time")
        if pipeline is None:
            return phase
        return {**phase, "pipeline_time": {
            **pipeline,
            "stage_laws": [_on_card(laws) for laws in pipeline["stage_laws"]]}}

    phases_in = base.get("phases") or {}
    pre_b = batches["per_op_timing"]
    dec_b = batches["per_op_timing_decode"]
    pre_in = _phase_on_card(phases_in.get("prefill") or {})
    dec_in = _phase_on_card(phases_in.get("decode") or {})
    pre_rows = rows["per_op_timing"]
    dec_rows = rows["per_op_timing_decode"]
    draft_rows = rows["per_op_timing_draft"]
    # Each phase at its rows' verify width (`_point_rows`).
    widths = p_dst.verify_widths()
    pre = phase_at(
        pre_in, pre_rows,
        Point({**point.counts, "B": pre_b, "s_verify": widths["per_op_timing"]}),
        p_dst.peak_flops,
        p_dst.peak_bw, fabric,
        cell=walks["per_op_timing"].analysis, breakdown=breakdown,
        launch_s=launch_s)
    dec = phase_at(
        dec_in, dec_rows,
        Point({**point.counts, "B": dec_b,
               "s_verify": widths["per_op_timing_decode"]}),
        p_dst.peak_flops,
        p_dst.peak_bw, fabric,
        cell=walks["per_op_timing_decode"].analysis, breakdown=breakdown,
        draft_rows=draft_rows, launch_s=launch_s)
    utm = p_dst.unique_tokens_multiplier or 1
    ng = max(1, int(p_dst.cluster_size or 1))
    if pre:
        pre = {**pre, "throughput": phase_throughput(
            pre, unique_tokens_multiplier=utm, num_gpus=ng)}
    if dec:
        dec = {**dec, "throughput": phase_throughput(
            dec, unique_tokens_multiplier=utm, num_gpus=ng)}
    return {
        "max_batch_per_gpu": _c.max_batch,
        "max_decode_batch_per_gpu": _c.max_batch,
        "max_prefill_batch_per_gpu": _c.max_prefill_batch,
        # Why `max_batch_per_gpu` is 0 on this card; None when it serves.
        "infeasibility_reason": _c.infeasibility_reason,
        "unique_tokens_multiplier": utm,
        "num_gpus": ng,
        "cluster_size": ng,
        "phases": {"prefill": pre, "decode": dec},
        # The layout's, not the card's: an axis split fewer ways than asked.
        "parallel_warnings": base["parallel_warnings"],
        # The model's and the request's, not the card's.
        "max_context_tokens": base["max_context_tokens"],
        "beyond_max_context": base["beyond_max_context"],
        # The rows at this point — each phase's batch on this card: what the
        # phases above were summed from, and what a direct estimate there
        # must equal row for row.
        "per_op_timing": pre_rows,
        "per_op_timing_decode": dec_rows,
        **({"per_op_timing_draft": draft_rows} if draft_rows else {}),
    }


@dataclass(eq=False)
class CellBasis:
    """What a capacity preparation reads off its cells alone — the model,
    its version, deployment and precision (`_model_cells`' key), and no
    card or request: the budget walk (`_capacity_walk`), its physical
    artifacts, the placement of its pipeline stages, and its cache views
    per cache scope. Built once, as the first preparation of the entry asks
    for each, and read by every later one."""

    _resolved: Any
    _parallel: ParallelConfig
    _precision: PrecisionProfile
    _cells: StepCells
    _lock: Any = field(default_factory=threading.RLock)
    _views: Dict[FrozenSet[str], Tuple[Any, Any]] = field(default_factory=dict)
    _loaded: Dict[Tuple[Tuple[str, str], ...], Tuple[Any, Tuple[Any, ...]]] = (
        field(default_factory=dict))

    @cached_property
    def analysis(self) -> Any:
        """The one walk every budget scalar descends from (`_capacity_walk`)."""
        with self._lock:
            return _capacity_walk(self._resolved, self._parallel,
                                  self._precision, self._cells)

    @cached_property
    def artifacts(self) -> Tuple[Any, ...]:
        """The walk's physical artifacts, every weight leaf's
        (`layout.persistent.physical_artifacts`)."""
        from backend.arch.layout.persistent import physical_artifacts
        with self._lock:
            return physical_artifacts(self.analysis)

    def loaded(self, framework_binding: Any) -> Tuple[Any, Tuple[Any, ...]]:
        """The resident walk and artifacts for the binding's selected loaders.

        The base walk remains framework-free and shared with every reader. A
        framework's resolved loader mapping keys only the derived format view,
        so two frameworks cannot publish each other's resident scale grids.
        """
        from backend.arch.layout.persistent import physical_artifacts
        base = self.analysis
        loaders = (framework_binding.weight_loaders_for(base.program, base)
                   if framework_binding is not None else {})
        key = tuple(sorted(loaders.items()))
        if not key:
            return base, self.artifacts
        with self._lock:
            entry = self._loaded.get(key)
            if entry is None:
                analysis = base.with_load_formats(loaders)
                if analysis.load_refusals:
                    raise DeploymentRefusal(
                        "; ".join(analysis.load_refusals),
                        knobs=("tp_attention",))
                entry = self._loaded[key] = (
                    analysis, physical_artifacts(analysis))
            return entry

    @cached_property
    def placement(self) -> Optional[StagePlacement]:
        """The pipeline stages' placement, None without pipeline parallelism."""
        with self._lock:
            return (StagePlacement.of(self.analysis, self._parallel.pp)
                    if self._parallel.pp > 1 else None)

    def cache_views(self, scope: FrozenSet[str]) -> Tuple[Any, Any]:
        """The rank's and the whole model's cache views over the stacks of
        `scope` (`physical_cache_view`, `whole_model_cache_view`)."""
        from backend.arch.layout.physical_cache import (
            physical_cache_view,
            whole_model_cache_view,
        )
        with self._lock:
            views = self._views.get(scope)
            if views is None:
                views = self._views[scope] = (
                    physical_cache_view(self.analysis, cache_scope=scope),
                    whole_model_cache_view(self.analysis, cache_scope=scope))
            return views


class ModelCells(NamedTuple):
    """One `_model_cells` entry: the resolution its cells walk, the cells,
    and what a capacity preparation reads off them alone (`CellBasis`)."""
    resolved: Any
    cells: StepCells
    basis: CellBasis


def _model_cells(resolved: Any, parallel: ParallelConfig,
                 precision: PrecisionProfile, *, model_id: Optional[str]
                 ) -> ModelCells:
    """The cells of one model's spec, deployment and precision under this
    code (`fingerprint.result_version`: the code, the shared data, the
    model's config and spec footprint): its walks, the serving layers and
    projections built off them, and their laws compiled. A request at
    another workload — batch, prompt, cached prefix, output, verify width —
    reads these cells at its own point; another model, deployment or
    precision, or another code version, is another key and so other cells.
    Returned with the resolution the cells walk, which a request then
    reads: an equal resolution of the same spec is not the same object, and
    the cells are found by their spec object (`StepCells.cell`).

    Bounded at every level: these entries (`_CELLS`), the cells of one
    (`step_eval.CELLS_KEPT`) and what is built off each cell's walk
    (`step_eval.READINGS_KEPT`); a cell builds what it holds under its own
    lock, so requests on several threads share one cell. With them, what a
    capacity preparation reads off the cells alone (`CellBasis`)."""
    shared = shared_walk_cells(resolved, parallel, precision,
                               model_id=model_id)
    return shared.reader(
        "capacity", lambda: ModelCells(
            shared.resolved, shared.cells,
            CellBasis(shared.resolved, parallel, precision, shared.cells)))


@dataclass(frozen=True)
class _CapacityInput:
    """The deployment facts every capacity question starts from."""

    card: Card
    resolved: Any
    parallel: ParallelConfig
    precision: PrecisionProfile


def _capacity_input(*, config: dict, gpu_sku: str,
                    knobs: CapacityKnobs) -> _CapacityInput:
    """Resolve the one card/spec/mesh/precision context a caller names."""
    card = Card.of(gpu_sku)
    if not 0.1 <= knobs.gpu_memory_fraction <= 1.0:
        raise ValueError("gpu_memory_fraction must be in [0.1, 1.0]")

    resolved = knobs.resolved
    if resolved is None:
        resolved = get_default_loader().match(config, model_id=knobs.model_id)
    if resolved is None:
        raise ValueError(
            f"No spec matches {knobs.model_id!r}. "
            "Add a spec in backend/arch/models/ or run /generate-spec first.")

    parallel = (knobs.parallel or ParallelConfig()).for_program(
        Program.of(resolved))
    errs = parallel.validate()
    if errs:
        raise ValueError("; ".join(errs))
    precision = (knobs.precision
                 if isinstance(knobs.precision, PrecisionProfile)
                 else PrecisionProfile.from_request(knobs.precision))
    return _CapacityInput(card, resolved, parallel, precision)


def _weight_overflow_reason(*, gpu: Any, weight_bytes: float,
                            usable_bytes: float, static_bytes: float,
                            constant_state_bytes: float) -> str:
    """The one sentence a rank whose weights cannot load is refused with."""
    static = (f" + KV static ({static_bytes / _GB:.2f} GiB)"
              if static_bytes > 0 else "")
    constant = (f" + recurrent state ({constant_state_bytes / _GB:.2f} GiB)"
                if constant_state_bytes > 0 else "")
    return (
        f"weights ({weight_bytes / _GB:.2f} GiB){static}{constant} exceed "
        f"usable HBM "
        f"({usable_bytes / _GB:.2f} GiB) on {gpu.sku} at this parallel config. "
        "Raise TP/EP or use a smaller-weight model."
    )


@stage_context("preparation")
def _capacity_preparation(
    *,
    config: dict,
    gpu_sku: str,
    knobs: CapacityKnobs,
) -> _CapacityPreparation:
    """Seq-invariant first half of `estimate_capacity`.

    Computes everything that does NOT depend on `avg_input_tokens` /
    `avg_output_tokens`: GPU validation, spec match, materialize quant
    defaults, KV per-token bytes, per-rank weight breakdown, GPU
    rooflines, framework binding and measurement scope. The result feeds
    `_capacity_finish` which adds the per-seq math.
    """
    gpu_memory_fraction = knobs.gpu_memory_fraction
    num_gpus = knobs.num_gpus
    # A counted speculative draft runs the model's MTP stack, so that stack
    # is resident: its weights here, as its cache is in the draft's cache
    # scope below (`SpeculativeDraft.cache_scope`). Asked once, so the
    # budget never prices a draft whose weights it did not load. The aux
    # stacks a language-model step walks are its MTP stacks (an encoder is
    # its encode phase's), so the flag admits exactly the draft's stacks.
    include_aux_weights = (knobs.include_aux_weights
                           or knobs.speculative_draft is not None)
    tensor_patterns_meta = knobs.tensor_patterns_meta
    published_weight_bytes = knobs.published_weight_bytes

    # One card/spec/mesh/precision context, shared with the weights preflight:
    # neither path can admit a deployment the other refuses. The card's price
    # reaches this preparation through `_preparation_fields` below.
    capacity_input = _capacity_input(config=config, gpu_sku=gpu_sku,
                                     knobs=knobs)
    card, resolved, parallel, precision = (
        capacity_input.card, capacity_input.resolved, capacity_input.parallel,
        capacity_input.precision)

    # The GPU count is the mesh's rank count; nothing is priced off a count
    # stated beside it. The request boundary refuses a stated count that
    # disagrees with this mesh (`api/routes/inference.py`), so the response
    # never shows two GPU counts.
    cluster_size = parallel.cluster_size
    num_gpus = cluster_size

    # The walks this preparation's steps share — every point of a sweep
    # reads the same ones, and so does every later request of the same
    # model, deployment and precision (`_model_cells`).
    resolved, cells, basis = _model_cells(resolved, parallel, precision,
                                          model_id=knobs.model_id)
    analysis = basis.analysis
    resolved_precision = analysis.resolved_precision
    # Resolve overlay context once. Per-seq loops reuse this tuple.
    (fw, framework_binding, strategy,
     enable_measured_lookup) = _resolve_overlay_context(
        framework=knobs.framework,
        selection_mode=knobs.selection_mode,
        measurement_scope=knobs.measurement_scope,
        calibration_rows=knobs.calibration_rows,
        deployment=ServingDeployment.at(
            gpu_sku=gpu_sku,
            activation_dtype=resolved_precision.model_dtype,
            server_flags=knobs.framework_options),
    )

    analysis, artifacts = basis.loaded(framework_binding)
    all_artifacts = artifacts
    # Weights kept in host memory are not GPU weights: every per-rank weight
    # sum below — the stage profiles, the cache room — reads the device
    # artifacts alone. A weight's scale metadata goes with it.
    offloadable = analysis.program.host_offloadable
    host_artifacts = tuple(
        row for row in artifacts if row.node_uid in offloadable)
    if knobs.host_offload:
        artifacts = tuple(
            row for row in artifacts if row.node_uid not in offloadable)
    # Model-level budget scalars from the eval cache layer: the per-token
    # GROWTH slope (every tensor whose storage law has the length free in
    # it — a window grows until its window), the constant recurrent-state
    # FLOOR billed once per request, and the quant-scale overhead split
    # into its dynamic (per stored position) and static halves. Per rank
    # by construction — TP is in the walk's tensor elements, CP in the
    # position spread — so there is no `kv_div` division here; `kv_div`
    # survives only as the legacy display divisor until the popovers move.
    # The speculative draft, when counted, keeps its own stack's cache
    # beside the backbone's: the request's cache scope.
    draft = (
        SpeculativeDraft.of(
            analysis.program, style=knobs.speculative_draft,
            draft_tokens=Speculation.verifying(knobs.s_verify).draft_tokens)
        if knobs.speculative_draft else None)
    cache_scope = (analysis.program.included_stack_names() if draft is None
                   else draft.cache_scope(analysis.program))
    cache_budget = _cache_budget_terms(*basis.cache_views(cache_scope),
                                       parallel)
    kv_reader = cache_budget.reader
    kv_static_per_rank = cache_budget.static
    kv_constant_state_per_rank = cache_budget.constant_state
    per_token_per_rank = cache_budget.per_token

    # The two display scalars, as the RATIO OF THE TWO VIEWS rather than as a
    # claim about them. `tp_attention × cp` was the claim, and it was the same
    # uniform-divisor mistake the rest of this page has retired: eight-way TP
    # over two KV heads divides the cache twice, not eight times, so the panel
    # read "KV divided by 8×" beside a per-rank budget that was an eighth of no
    # such thing — and `replication_factor`, being `1` whenever ANY axis
    # sharded, reported those four redundant copies as none. Both fall out of
    # dividing the whole model's per-token bytes by this rank's, which cannot
    # disagree with the budget below because it IS the budget's numerator.
    kv_div = cache_budget.shard_divisor
    # What the ranks that did NOT get a slice hold: an identical copy. This is
    # vLLM's `num_kv_head_replicas` read from the other end — it does not need
    # to know that MLA has no head axis to shard or that a GQA model ran out of
    # KV heads partway, because a copy is whatever the split did not divide.
    kv_replication_factor = cache_budget.replication_factor
    # And below one is not a small replication factor — it is a split of bytes
    # that no rank holds a second copy of. Only a subgroup world can reach it.
    # An axis that BUYS ranks cannot: `cp` is in the numerator and in no
    # denominator under ulysses/pcp, because prefill context parallelism
    # all_gathers the K/V before the cache write and leaves each of its
    # ranks a whole copy — it raises this ratio, never lowers it.
    # layersplit is the exception: it folds per-rank occupancy by
    # ~cp, so the local view puts cp in the denominator and the extra
    # copies disappear. `dcp` partitions
    # ranks already counted, so it is the one knob that can ask for a cut
    # nobody has the copies to give. That is why decode context parallelism is an
    # MLA / MQA feature and vLLM caps it at `tp // total_num_kv_heads`: the
    # cache it splits is the cache TP left replicated. Read from this end the
    # cap needs no head arithmetic and no per-architecture case — the same
    # ratio that prices the budget answers it.
    #
    # The one place the two answers part: vLLM's GQA guard
    # (`config/model.py`, `max_dcp_size = tensor_parallel_size //
    # total_num_kv_heads`) counts only the copies TP left, so it refuses a
    # `dcp` that spans a PCP column of ranks holding the SAME head shard —
    # copies this ratio can see and that guard does not look for. Narrower
    # rule, not different physics; a deployment in that gap prices correctly
    # here and will not launch on vLLM today.
    def _weight_tree(
        pp_stage: Optional[int] = None,
    ) -> Dict[str, Any]:
        """The per-rank weight tree, optionally restricted to one pp stage.

        A closure rather than three call sites carrying the same nine
        arguments: WHICH stage a rank's weights belong to is decided twice —
        here by cache headroom, and again in `capacity_ceilings` by batch at
        the requested length — and both need the tree rebuilt when their
        answer is not the heaviest stage's. Passing the ingredients forward so
        a later layer can re-issue the call is how two builds come to differ.
        """
        common = dict(
            precision=precision,
            quant_granularity=(quant_granularity(resolved.spec.quant_context)
                               or "per-tensor"),
            published_weight_bytes=published_weight_bytes,
            include_aux_weights=include_aux_weights,
            **({} if pp_stage is None else {"pp_stage": pp_stage}),
        )
        return per_rank_weight_breakdown(
            resolved,
            parallel,
            analysis=analysis,
            # Every weight leaf is in the tree, a table kept in host memory
            # too; the per-rank sums below are the device artifacts'.
            artifacts=all_artifacts,
            **common,
        )

    price = card.price(gpu_memory_fraction=gpu_memory_fraction)
    usable_bytes = price.usable_bytes

    # --- Which pipeline stage IS this rank? ---
    #
    # `pp` selects a stage, and a stage holds its own layers' weights and its
    # own layers' cache. Weight already knew that; the cache did not, so a rank
    # at `pp=8` was told it held an eighth of the weights and all twenty-eight
    # layers of KV — which understated Llama-3.2-3B's serving batch sevenfold.
    # Both now read the same partition.
    #
    # WHICH stage is a memory question, so it is asked here rather than in the
    # weight tree: price every stage, take the one with the least room left for
    # cache — by tokens first, and by whole requests at full context where the
    # token slope cannot tell two stages apart. `_StageProfile.headroom` says
    # why the second term is not optional.
    #
    # The choice is workload-free by construction, so a sweep keeps reusing one
    # preparation across every sequence length rather than re-pricing per cell.
    #
    # `kv_div` and `kv_replication_factor` above are deliberately taken BEFORE
    # this: they are ratios of whole/local views over the WHOLE model, and a stage's
    # share of the layers is not a form of replication.
    # One ledger: parameters + metadata + buffers, already at serving
    # width when the walk's precision profile converts weights. Pipeline stages filter the
    # same artifacts (`artifacts_for_layers`) rather than a second byte
    # derivation. The display tree is looked up from this, never recast.
    from backend.arch.layout.persistent import (
        artifacts_for_layers, resident_weights, whole_resident,
    )

    def _resident(view: Any = None, art: Any = None) -> float:
        art = artifacts if art is None else art
        if view is not None:
            art = artifacts_for_layers(
                analysis, art,
                view.weights(analysis.program.included_stack_names(
                    include_aux=include_aux_weights)),
            )
        ledger = resident_weights(art)
        return float(
            ledger.resident if include_aux_weights
            else ledger.without_aux(analysis.program)
        )

    # The model's whole weight, wherever each table is kept.
    weight_bytes_total = whole_resident(
        all_artifacts, analysis.program, include_aux=include_aux_weights,
    )

    pp_stages: Tuple[_StageProfile, ...] = ()
    pp_stage: Optional[int] = None
    # The offloadable weights on a rank of each stage (one entry at pp=1).
    host_offload_bytes: Tuple[float, ...] = (
        _resident(art=host_artifacts),)
    pp_bound: Optional[_StageProfile] = None
    # The published context length is the dim `s_max`, bound only by specs
    # whose models publish one (dims.yaml). Under pp, without it the stage
    # order here rests on tokens alone; `_capacity_finish` re-orders by the
    # request it is given either way. Every estimate publishes it, and says
    # when the request it was asked about is longer.
    _s_max = (value_of(analysis.binding.get("s_max"))
              if analysis.binding is not None else None)
    max_context = None if _s_max is None else int(_s_max)
    if parallel.pp > 1:
        _placement = basis.placement
        _max_ctx = None if max_context is None else float(max_context)
        stage_bytes = [
            _resident(_placement.view(s)) for s in range(parallel.pp)
        ]
        host_offload_bytes = tuple(
            _resident(_placement.view(s), art=host_artifacts)
            for s in range(parallel.pp))
        pp_stages = _pipeline_stage_profiles(
            _placement, stage_bytes, kv_reader, _max_ctx)
        pp_stage = min(range(len(pp_stages)),
                       key=lambda s: pp_stages[s].headroom(usable_bytes))
        chosen = pp_bound = pp_stages[pp_stage]
        kv_reader = chosen.reader
        per_token_per_rank = chosen.per_token
        kv_static_per_rank = chosen.static
        kv_constant_state_per_rank = chosen.constant_state

    weight_bytes_per_rank = (
        pp_bound.weight_bytes if pp_bound is not None
        else _resident()
    )
    # What this card decides: its memory, the cache room left beside the
    # bound stage's weights and floors, and its peak rates at the model's
    # own dtype — the one statement of it (`_preparation_fields`).
    card_fields = _preparation_fields(
        price,
        model_dtype=resolved_precision.model_dtype,
        weight_bytes_per_rank=weight_bytes_per_rank,
        static=kv_static_per_rank,
        constant_state=kv_constant_state_per_rank)
    refusal_reason = (
        None if card_fields["fits"] else _weight_overflow_reason(
            gpu=card, weight_bytes=weight_bytes_per_rank,
            usable_bytes=price.usable_bytes,
            static_bytes=kv_static_per_rank,
            constant_state_bytes=kv_constant_state_per_rank))

    if refusal_reason is not None:
        # A rank that cannot load its weights has no display tree to build and
        # no phase rows to walk. Keep the budget facts already read — enough
        # for every ceiling and the no-step response — and return here.
        wb = {
            "total_bytes_per_rank": weight_bytes_per_rank,
            "deployment_total_bytes": weight_bytes_total,
            "deployment_data_bytes": weight_bytes_total,
            "deployment_scale_bytes": 0.0,
            "unattributed_scale_bytes": 0.0,
            "pipeline": {"stage": pp_stage},
            "published_checkpoint_bytes": published_weight_bytes,
            "left_out_stacks": [],
            "groups": [],
            "model_level": [],
        }
        return _CapacityPreparation(
            **card_fields,
            cluster_size=cluster_size, num_gpus=num_gpus,
            parallel=parallel,
            gpu_memory_fraction=gpu_memory_fraction,
            model_id=knobs.model_id,
            resolved=resolved,
            precision=precision,
            resolved_precision=resolved_precision,
            precision_report=analysis.precision_effective(),
            kv_div=kv_div,
            kv_reader=kv_reader,
            kv_static_per_rank=kv_static_per_rank,
            kv_constant_state_per_rank=kv_constant_state_per_rank,
            per_token_per_rank=per_token_per_rank,
            wb=wb,
            weight_tree_at=lambda _stage=None: wb,
            weight_bytes_per_rank=weight_bytes_per_rank,
            weight_bytes_total=weight_bytes_total,
            host_offload_bytes=host_offload_bytes,
            host_offload=knobs.host_offload,
            pp_stages=pp_stages,
            pp_bound=pp_bound,
            unique_tokens_multiplier=(
                parallel.dp_attention * parallel.dp_replicas),
            kv_replication_factor=kv_replication_factor,
            comm_latency_s=knobs.comm_latency_us * 1e-6,
            launch_floor_s=knobs.launch_floor_us * 1e-6,
            roles_present=[],
            shape_warnings=[],
            max_context=max_context,
            framework=fw,
            framework_binding=framework_binding,
            measurement_scope=knobs.measurement_scope or NO_MEASUREMENTS,
            calibration_rows=knobs.calibration_rows,
            strategy=strategy,
            enable_measured_lookup=enable_measured_lookup,
            selection_mode=knobs.selection_mode,
            kernel_mix=knobs.kernel_mix,
            kv_block_size=deployment_kv_block_size(
                framework_binding, knobs.kv_block_size),
            s_prefix_cached=knobs.s_prefix_cached,
            s_verify=knobs.s_verify,
            acceptance_rate=knobs.acceptance_rate,
            ep_load_factor=knobs.ep_load_factor,
            draft=draft,
            refusal_reason=refusal_reason,
            cells=cells,
        )

    # The tree is built for the stage the numbers beside it describe.
    wb = _weight_tree(pp_stage)

    # Unique-token multiplier for cluster-level totals (see estimate_capacity
    # docstring).
    unique_tokens_multiplier = parallel.dp_attention * parallel.dp_replicas

    # Shape check: validate spec leaves against per-tensor safetensors
    # metadata when the caller supplied it. The matcher lives next to
    # `bind_leaves_to_hf`; this is not a budget scalar.
    from backend.arch.validate import spec_vs_tensors_check
    shape_warnings = (
        spec_vs_tensors_check(resolved, tensor_patterns_meta)
        if tensor_patterns_meta else []
    )

    # Roles actually present on this spec's leaves — surfaced so the
    # frontend Activation Dtype UI knows which per-role override rows
    # to render.
    from backend.arch.spec_tree import iter_leaves
    _roles_seen: Dict[str, None] = {}
    for lt in iter_leaves(resolved):
        if lt.role:
            _roles_seen.setdefault(lt.role, None)
    roles_present = list(_roles_seen.keys())

    return _CapacityPreparation(
        **card_fields,
        cluster_size=cluster_size, num_gpus=num_gpus,
        parallel=parallel,
        gpu_memory_fraction=gpu_memory_fraction,
        model_id=knobs.model_id,
        resolved=resolved,
        precision=precision,
        resolved_precision=resolved_precision,
        precision_report=analysis.precision_effective(),
        kv_div=kv_div,
        kv_reader=kv_reader,
        kv_static_per_rank=kv_static_per_rank,
        kv_constant_state_per_rank=kv_constant_state_per_rank,
        per_token_per_rank=per_token_per_rank,
        wb=wb,
        weight_tree_at=_weight_tree,
        weight_bytes_per_rank=weight_bytes_per_rank,
        weight_bytes_total=weight_bytes_total,
        host_offload_bytes=host_offload_bytes,
        host_offload=knobs.host_offload,
        pp_stages=pp_stages,
        pp_bound=pp_bound,
        unique_tokens_multiplier=unique_tokens_multiplier,
        kv_replication_factor=kv_replication_factor,
        comm_latency_s=knobs.comm_latency_us * 1e-6,
        launch_floor_s=knobs.launch_floor_us * 1e-6,
        roles_present=roles_present,
        shape_warnings=shape_warnings,
        max_context=max_context,
        framework=fw,
        framework_binding=framework_binding,
        measurement_scope=knobs.measurement_scope or NO_MEASUREMENTS,
        calibration_rows=knobs.calibration_rows,
        strategy=strategy,
        enable_measured_lookup=enable_measured_lookup,
        selection_mode=knobs.selection_mode,
        kernel_mix=knobs.kernel_mix,
        kv_block_size=deployment_kv_block_size(
            framework_binding, knobs.kv_block_size),
        s_prefix_cached=knobs.s_prefix_cached,
        s_verify=knobs.s_verify,
        acceptance_rate=knobs.acceptance_rate,
        ep_load_factor=knobs.ep_load_factor,
        draft=draft,
        refusal_reason=refusal_reason,
        cells=cells,
    )


def estimate_capacity_ceilings_at(
    *,
    config: dict,
    gpu_sku: str,
    seq_lens: List[int],
    gpu_memory_fraction: float = 0.9,
    num_gpus: Optional[int] = None,
    avg_output_tokens: int = 1024,
    parallel: Optional[ParallelConfig] = None,
    prefill_chunk_size: Optional[int] = None,
    **prep_kwargs: Any,
) -> Dict[int, CapacityCeilings]:
    """L1 over MANY seq points that share one deployment — `{seq: ceilings}`.

    The preparation (spec match, per-rank weight breakdown, KV per-token bytes,
    GPU rooflines) depends on the deployment, NOT on the sequence length, and it
    dominates a single-point query (~80% of it). A caller sweeping a grid asks
    about many points of the SAME deployment, so it should build the model once
    and evaluate the cheap per-seq arithmetic against it — rather than
    reconstructing the model for every coordinate.

    `estimate_capacity_ceilings` is the single-point case of this.
    """
    p = _capacity_preparation(
        config=config, gpu_sku=gpu_sku,
        knobs=CapacityKnobs.from_kwargs({
            **prep_kwargs,
            "gpu_memory_fraction": gpu_memory_fraction,
            "num_gpus": num_gpus, "parallel": parallel,
        }),
    )
    schedules = {
        seq: PrefillChunkSchedule.materialize(
            PrefillEndpointGeometry(seq, p.s_prefix_cached, 1, 1),
            prefill_chunk_size=prefill_chunk_size)
        for seq in seq_lens
    }
    return {
        seq: capacity_ceilings(
            p,
            avg_input_tokens=seq,
            avg_output_tokens=avg_output_tokens,
            prefill_chunk_schedule=schedules[seq],
        )
        for seq in seq_lens
    }


def estimate_capacity_ceilings(
    *,
    config: dict,
    gpu_sku: str,
    gpu_memory_fraction: float = 0.9,
    num_gpus: Optional[int] = None,
    avg_input_tokens: int = 1024,
    avg_output_tokens: int = 1024,
    parallel: Optional[ParallelConfig] = None,
    prefill_chunk_size: Optional[int] = None,
    **prep_kwargs: Any,
) -> CapacityCeilings:
    """L1 only — "does this deployment fit, and up to what batch?".

    The entry point for a feasibility gate (kernel_bench's shape sweep asks this
    once per parallel × seq × phase point). It runs the seq-invariant
    preparation and the capacity arithmetic, and stops there: no phase
    estimates, no per-op rows, no formula strings. Callers that need those ask
    `estimate_capacity`, which composes L1..L4 into the full response.

    `prep_kwargs` forwards the remaining dtype / quant / workload knobs to
    `_capacity_preparation` unchanged.
    """
    p = _capacity_preparation(
        config=config, gpu_sku=gpu_sku,
        knobs=CapacityKnobs.from_kwargs({
            **prep_kwargs,
            "gpu_memory_fraction": gpu_memory_fraction,
            "num_gpus": num_gpus, "parallel": parallel,
        }),
    )
    schedule = PrefillChunkSchedule.materialize(
        PrefillEndpointGeometry(avg_input_tokens, p.s_prefix_cached, 1, 1),
        prefill_chunk_size=prefill_chunk_size)
    return capacity_ceilings(
        p,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        prefill_chunk_schedule=schedule,
    )


def estimate_capacity(
    *,
    config: dict,
    gpu_sku: str,
    knobs: Optional[CapacityKnobs] = None,
    avg_input_tokens: int = 1024,
    avg_output_tokens: int = 1024,
    prefill_batch_size: Optional[int] = None,
    decode_batch_size: Optional[int] = None,
    prefill_chunk_size: Optional[int] = None,
    **kwargs: Any,
) -> dict:
    """Estimate KV cache budget, max tokens, and max batch under a parallel
    deployment config.

    Seq-invariant fields live on `CapacityKnobs`. Callers may pass
    `knobs=` or the same names they used to pass flat; not both.
    Sequence length and per-phase batch stay arguments because they
    vary per point.

    `precision` (a `PrecisionProfile` or its request form; as shipped when
    None) decides every weight format, edge dtype, matmul column and cache
    format; the response's `precision` block says what it changed.

    The published checkpoint size is surfaced for reference only — not
    used in the memory budget math.
    """
    return _estimate(
        config=config, gpu_sku=gpu_sku, knobs=knobs,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        prefill_batch_size=prefill_batch_size,
        decode_batch_size=decode_batch_size,
        prefill_chunk_size=prefill_chunk_size,
        **kwargs,
    )


def _estimate(
    *,
    config: dict,
    gpu_sku: str,
    knobs: Optional[CapacityKnobs] = None,
    avg_input_tokens: int = 1024,
    avg_output_tokens: int = 1024,
    prefill_batch_size: Optional[int] = None,
    decode_batch_size: Optional[int] = None,
    prefill_chunk_size: Optional[int] = None,
    **kwargs: Any,
) -> dict:
    knobs = _require_knobs(knobs, kwargs)
    if avg_input_tokens < 0 or avg_output_tokens < 0:
        raise ValueError("avg_input_tokens and avg_output_tokens must be ≥ 0")
    preparation = _capacity_preparation(
        config=config, gpu_sku=gpu_sku, knobs=knobs,
    )
    return _capacity_finish(
        preparation,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        prefill_batch_override=prefill_batch_size,
        decode_batch_override=decode_batch_size,
        prefill_chunk_size=prefill_chunk_size,
        media=knobs.media,
        encode_batch_override=knobs.encode_batch_size,
        breakdown=True,
    )


def estimate_pending(
    *,
    config: dict,
    gpu_sku: str,
    knobs: Optional[CapacityKnobs] = None,
    avg_input_tokens: int = 1024,
    avg_output_tokens: int = 1024,
    prefill_batch_size: Optional[int] = None,
    decode_batch_size: Optional[int] = None,
    prefill_chunk_size: Optional[int] = None,
    **kwargs: Any,
) -> "PendingEstimate":
    """`estimate_capacity` up to its overlays' walks, for a caller that
    finishes several estimates after one lookup (`finish_estimates`): a
    disaggregated request's two deployments."""
    knobs = _require_knobs(knobs, kwargs)
    if avg_input_tokens < 0 or avg_output_tokens < 0:
        raise ValueError("avg_input_tokens and avg_output_tokens must be ≥ 0")
    preparation = _capacity_preparation(config=config, gpu_sku=gpu_sku, knobs=knobs)
    return _capacity_pending(
        preparation,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        prefill_batch_override=prefill_batch_size,
        decode_batch_override=decode_batch_size,
        prefill_chunk_size=prefill_chunk_size,
        media=knobs.media,
        encode_batch_override=knobs.encode_batch_size,
        breakdown=True,
    )


#: What a point of a (context × batch) sweep's decode cloud is held or grouped
#: by: the one point-cloud vocabulary, plus the Compare row a producer may tag
#: the point with (`frontier.POINT_DIMS` names the point's own coordinates).
SweepCloudDim = Literal["row", *POINT_DIMS]


@dataclass(frozen=True)
class CapacitySweepPlan:
    """A (context × batch) sweep before its cells run: the contexts — each a
    prompt length, the cached prefix of it (`s_prefixes`, beside
    `seq_lens`) and the verify width its decode step runs at (`s_verifies`)
    — the batch points (`(key, prefill_batch, decode_batch)`;
    one keyless point when the sweep has no grid) and the single-estimate
    arguments every cell shares.

    The whole sweep is one cell (`cells`): one (model, deployment, precision)
    walks once over the workload's counts, and every (context, batch) point
    is read off that walk by substitution (`walk_sweep_cell`) — a verify
    width too, wherever the walk takes it as a symbol (every speculative
    width is one walk, `_CapacityPreparation.walks_alike`); the cell runs
    anywhere — another process included — and `shape` folds its points back
    by batch key.

    Those are the axes a point substitutes into one walk, and each is
    already named once — there is no separate vocabulary of them:

    * context — `seq_lens` and `s_prefixes` here, reaching a row as the
      `Point` counts `s_input` / `s_prefix`;
    * batch — `points` here (`gridded`: a batch grid or the single pair),
      resolved per card by `_point_batches` / `_direct_batches`;
    * the card within a walk class — `parallel_sweep.class_cards` names
      them and `ClassSweep.cards` keeps them;
    * K, the speculative width — `s_verifies` here
      (`CapacityKnobs.s_verify`), walked as the symbol wherever it is wider
      than one;
    * the acceptance rate and the two time floors — substituted after the
      walk, into a step's already-priced form (`row_roofline.FloorForm`),
      which is why `api.gallery_frontier.FrontierView.walk_parts` keeps
      them out of what a walk is keyed by."""

    seq_lens: Tuple[int, ...]
    #: Each context's cached prefix, in `seq_lens` order.
    s_prefixes: Tuple[int, ...]
    #: Each context's verify width (`CapacityKnobs.s_verify`: K + 1 for K
    #: draft tokens), in `seq_lens` order.
    s_verifies: Tuple[int, ...]
    points: Tuple[Tuple[Optional[str], Optional[int], Optional[int]], ...]
    gridded: bool
    config: dict
    gpu_sku: str
    knobs: CapacityKnobs
    avg_output_tokens: int
    prefill_chunk_size: Optional[int]
    #: Each context's endpoint launches, materialized when the workload cells
    #: are built.  A point's batch changes only endpoint repetition.
    prefill_chunk_schedules: Tuple[PrefillChunkSchedule, ...]

    @classmethod
    def of(
        cls,
        *,
        seq_lens: List[int],
        config: dict,
        gpu_sku: str,
        knobs: Optional[CapacityKnobs] = None,
        avg_output_tokens: int = 1024,
        prefill_batch_size: Optional[int] = None,
        decode_batch_size: Optional[int] = None,
        prefill_chunk_size: Optional[int] = None,
        batch_grid: Optional[List[dict]] = None,
        s_prefixes: Optional[List[int]] = None,
        s_verifies: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> "CapacitySweepPlan":
        """`s_prefixes` are the contexts' cached prefixes and `s_verifies`
        their verify widths, one per entry of `seq_lens`; without them
        every context caches the request's own prefix
        (`knobs.s_prefix_cached`) and verifies at its width
        (`knobs.s_verify`)."""
        knobs = _require_knobs(knobs, kwargs)
        if not seq_lens:
            raise ValueError("seq_lens must be non-empty")
        if any(s < 0 for s in seq_lens):
            raise ValueError("seq_lens entries must be ≥ 0")
        if s_prefixes is None:
            s_prefixes = [knobs.s_prefix_cached] * len(seq_lens)
        if len(s_prefixes) != len(seq_lens):
            raise ValueError("s_prefixes must name one prefix per seq_lens entry")
        if any(pre < 0 for pre in s_prefixes):
            raise ValueError("s_prefixes entries must be ≥ 0")
        if s_verifies is None:
            s_verifies = [knobs.s_verify] * len(seq_lens)
        if len(s_verifies) != len(seq_lens):
            raise ValueError("s_verifies must name one width per seq_lens entry")
        if any(int(w) < 1 for w in s_verifies):
            raise ValueError("s_verifies entries must be ≥ 1")
        if avg_output_tokens < 0:
            raise ValueError("avg_output_tokens must be ≥ 0")
        # Batch axis: the grid (each {key, prefill, decode}), or the single pair.
        points = (tuple((bp["key"], bp.get("prefill_batch_size"),
                         bp.get("decode_batch_size")) for bp in batch_grid)
                  if batch_grid
                  else ((None, prefill_batch_size, decode_batch_size),))
        schedules = tuple(
            PrefillChunkSchedule.materialize(
                PrefillEndpointGeometry(seq, prefix, 1, 1),
                prefill_chunk_size=prefill_chunk_size)
            for seq, prefix in zip(seq_lens, s_prefixes, strict=True))
        return cls(
            seq_lens=tuple(seq_lens), s_prefixes=tuple(s_prefixes),
            s_verifies=tuple(int(w) for w in s_verifies),
            points=points, gridded=bool(batch_grid),
            config=config, gpu_sku=gpu_sku, knobs=knobs,
            avg_output_tokens=avg_output_tokens,
            prefill_chunk_size=prefill_chunk_size,
            prefill_chunk_schedules=schedules)

    def cells(self) -> List[Cell]:
        """The sweep's one cell: its walk's arguments — every context it
        prices — and the batch points at each."""
        return [Cell(walk={"config": self.config, "gpu_sku": self.gpu_sku,
                           "knobs": self.knobs, "seq_lens": self.seq_lens,
                           "s_prefixes": self.s_prefixes,
                           "s_verifies": self.s_verifies,
                           "avg_output_tokens": self.avg_output_tokens,
                           "prefill_chunk_size": self.prefill_chunk_size,
                           "prefill_chunk_schedules": self.prefill_chunk_schedules},
                     points=self.points)]

    def measurement(self) -> Dict[str, Any]:
        """What the sweep's measured track is read under — the request's
        measurement fields (`batch_sweep.measure_points`)."""
        k = self.knobs
        return {"selection_mode": k.selection_mode, "measurement_scope": k.measurement_scope,
                "calibration_rows": k.calibration_rows, "kernel_mix": k.kernel_mix}

    def shape(self, per_seq: List[List[Any]]) -> Any:
        """Each context's points (in `seq_lens` order, each in `points`
        order) → `{key: [<per-context>…]}` for a grid, else
        `[<per-context>…]`."""
        grid = {k: [seq[j] for seq in per_seq]
                for j, (k, _pb, _db) in enumerate(self.points)}
        return grid if self.gridded else grid[None]

    def decode_cloud(self, per_seq: List[List[Any]]) -> List[dict]:
        """Every grid point as a point of the one decode cloud
        (`point_cloud.estimate_cloud`)."""
        parallel = self.knobs.parallel
        num_gpus = (parallel.cluster_size if parallel is not None
                    else self.knobs.num_gpus) or 1
        return estimate_cloud(
            seq_lens=self.seq_lens, verify_widths=self.s_verifies,
            batch_points=self.points, estimates=per_seq,
            gpu_sku=self.gpu_sku, parallel=parallel, num_gpus=num_gpus,
            acceptance_rate=self.knobs.acceptance_rate,
            ep_load_factor=self.knobs.ep_load_factor,
            served=point_served).points


def _sweep_cell(p: _CapacityPreparation, cell: Cell,
                prices: Optional[FrozenSet[int]] = None
                ) -> Iterator[Tuple[int, Optional[List[dict]]]]:
    """A `CapacitySweepPlan`'s cell on its preparation `p`: ONE estimate
    walks (`_capacity_pending` at the first seq, batch 1), and every point —
    each seq length, each batch a direct estimate there runs
    (`_direct_batches`) — is read off its laws by substitution
    (`_point_rows`, `_stamp_gpu_on_result`), its capacity at that length
    (`capacity_ceilings`, `_capacity_fields`) and its phases' frame there
    (`_phase_at_length`) beside them. A length whose ceilings bind another
    pipeline stage reads that stage's walk, which is another cell's rows.
    A context at another verify width (`s_verifies`) is `p` at that width
    (`_preparation_at`): its own capacity and draft, its rows read off the
    walk it shares (`_CapacityPreparation.walks_alike`: every speculative
    width is one walk) at its width — a width walked otherwise (a step
    that does not speculate beside one that does) walks its own.
    Each point is `{"sol": <the estimate there, no measurement read>,
    "measures": <its rows before measurement, `batch_sweep.point_measures`>}`
    — plain data; the sweep's one lookup and the fold come after
    (`batch_sweep.measure_points`). A request that looks no measurement up
    (`calibration_overlay.looks_up`) asks no point for its rows' keys, and
    its points measure nothing (`{}`).

    Yields `(index, points)` per context as that context finishes, in
    `seq_lens` order — the cell's own schedule, so a caller can say how far
    the cell has got and keep each context as it arrives rather than only
    the whole cell at the end.

    `prices` names the contexts to price; the others yield None. The walk
    is still read for every context, priced or not, because which walk a
    context reads off is decided by walking the ones before it (a stage's
    first context walks it and the rest substitute off that anchor): a
    context's points are then the same whichever of its siblings the caller
    also asked for. Only the reading-off is skipped, never the walking.

    Facts of the walk's own point that no law carries are not passed on to
    a point: `vs_naive_delta` (the fusion delta, a step time there)."""
    w = cell.walk
    out_tokens = w["avg_output_tokens"]
    preparations = {(p.s_verify, p.s_prefix_cached): p}
    # Per bound stage, the walks read so far: each with the preparation it
    # walked at and its finished anchor.
    walked: Dict[Optional[int],
                 List[Tuple[_CapacityPreparation, PendingEstimate, dict]]] = {}
    for index, (length, prefix, width, base_schedule) in enumerate(zip(
            w["seq_lens"], w["s_prefixes"], w["s_verifies"],
            w["prefill_chunk_schedules"], strict=True)):
        preparation = (width, prefix)
        if preparation not in preparations:
            preparations[preparation] = _preparation_at(
                w, s_verify=width, s_prefix_cached=prefix)
        pw = preparations[preparation]
        # A context the deployment cannot serve is refused as an estimate
        # there refuses it.
        refusal = prefix_refusal(pw.parallel, phase="prefill",
                                 input_tokens=length, s_prefix_cached=prefix)
        if refusal is not None:
            raise ValueError(refusal)
        ceilings = capacity_ceilings(pw, avg_input_tokens=length,
                                     avg_output_tokens=out_tokens,
                                     prefill_chunk_schedule=base_schedule)
        stage_walks = walked.setdefault(ceilings.pp_stage, [])
        shared_walk = next((entry for entry in stage_walks
                            if pw.walks_alike(entry[0])), None)
        if shared_walk is None:
            pending = _capacity_pending(
                pw, avg_input_tokens=length, avg_output_tokens=out_tokens,
                prefill_batch_override=1, decode_batch_override=1,
                prefill_chunk_schedule=base_schedule,
                fold_prefill_chunks=False, breakdown=False)
            shared_walk = (pw, pending, pending.finish())
            stage_walks.append(shared_walk)
        if prices is not None and index not in prices:
            # Walked (the anchors above are this cell's, whatever the caller
            # asked for), not read off.
            yield index, None
            continue
        p_walk, pending, anchor = shared_walk
        if (pw is p_walk and (length, prefix)
                == (anchor["avg_input_tokens"], p_walk.s_prefix_cached)):
            counts: Dict[str, int] = {}
            base = anchor
        else:
            counts = {"s_input": int(length), "s_prefix": int(prefix)}
            # The point must be one the walk admits, refused as a walk there
            # refuses it (a sequence cut that does not divide it).
            for walk in pending.walk_of_rows.values():
                Point(counts).admit(walk.analysis)
            base = {**anchor,
                    **_capacity_fields(pw, ceilings, anchor["weight_breakdown"],
                                       avg_input_tokens=length),
                    "phases": {name: _phase_at_length(
                        phase, pw, ceilings, avg_input_tokens=length,
                        avg_output_tokens=out_tokens, s_prefix_cached=prefix)
                        for name, phase in anchor["phases"].items()}}
        shared = {k: v for k, v in base.items() if k != "vs_naive_delta"}
        points = []
        # This context's chunked prefills, folded once per endpoint
        # (`_fold_chunked_prefill_sol`, `_point_measures_with_chunks`): on
        # one walk, card and context, a point's prefill rows and phase are
        # those at its prefill batch (`_point_rows`, `_stamp_gpu_on_result`),
        # which its endpoint's schedule states. The key also names what the
        # fold reads beside the schedule — the card and the verify widths
        # (`_endpoint_memo_key`, which says what sharing it wider than this
        # loop would need) — so no point can be served another width's or
        # another card's fold.
        endpoints: Dict[Any, Dict[str, Any]] = {}
        for (_key, pb, db) in cell.points:
            batches = _direct_batches(ceilings, prefill_batch_override=pb,
                                      decode_batch_override=db)
            rows = _point_rows(base, pw, batches, pending, Point(counts),
                               breakdown=False)
            rows, keys = pending.keyed_at(rows, gpu_sku=pw.gpu_sku)
            point = without_step({**shared, **_stamp_gpu_on_result(
                base, pw, ceilings, pending.walked_on, batches=batches,
                rows=rows,
                walks=pending.walk_of_rows, point=Point(counts),
                breakdown=False)}, pending.walk_of_rows)
            endpoint_schedule = base_schedule.with_endpoint_batch(
                batches["per_op_timing"])
            if not point.get("infeasibility_reason"):
                point = _fold_chunked_prefill_sol(
                    point, pw, pending, endpoint_schedule, endpoints)
            points.append({"sol": point, "measures": {}
                           if keys is None or point.get("infeasibility_reason")
                           else _point_measures_with_chunks(
                               rows, keys, pw, pending, point["phases"],
                               schedule=endpoint_schedule, endpoints=endpoints)})
        yield index, points


def _preparation_at(walk: Mapping[str, Any], *, s_verify: int,
                    s_prefix_cached: Optional[int] = None
                    ) -> _CapacityPreparation:
    """A sweep cell's preparation (`walk`, `CapacitySweepPlan.cells`) at
    verify width `s_verify` and, for a sweep whose contexts name cached
    prefixes, at that context's prefix: the request's knobs there — the
    draft's K with the width — on the same cells (`_model_cells`). A
    context's point still reads the one walk; the preparation states the
    endpoint its own capacity and phase frame are asked about."""
    knobs = replace(walk["knobs"], s_verify=int(s_verify))
    if s_prefix_cached is not None:
        knobs = replace(knobs, s_prefix_cached=int(s_prefix_cached))
    return _capacity_preparation(
        config=walk["config"], gpu_sku=walk["gpu_sku"],
        knobs=knobs)


def iter_sweep_cell(cell: Cell, prices: Optional[FrozenSet[int]] = None
                    ) -> Iterator[Tuple[int, Optional[List[dict]]]]:
    """`_sweep_cell` on a preparation of its own — what a compute worker
    drives, one context at a time, so it can report and keep each as it
    finishes. Top-level, so it pickles."""
    return _sweep_cell(_preparation_at(cell.walk,
                                       s_verify=cell.walk["s_verifies"][0]),
                       cell, prices)


def walk_sweep_cell(cell: Cell) -> List[List[dict]]:
    """Every context of `cell`, priced — `iter_sweep_cell` run to its end."""
    return [points for _index, points in iter_sweep_cell(cell)]


def sweep_points(**kwargs: Any) -> Tuple[CapacitySweepPlan, List[List[dict]]]:
    """A (seq × batch) sweep's points before any measurement: its plan, and
    per seq (in `seq_lens` order) each batch point's `{"sol", "measures"}` —
    theory, plain data (`_sweep_cell`). Takes `estimate_capacity_sweep`'s
    arguments."""
    plan = CapacitySweepPlan.of(**kwargs)
    (cell,) = plan.cells()
    return plan, walk_sweep_cell(cell)


def estimate_capacity_sweep(
    *,
    seq_lens: List[int],
    config: dict,
    gpu_sku: str,
    knobs: Optional[CapacityKnobs] = None,
    avg_output_tokens: int = 1024,
    prefill_batch_size: Optional[int] = None,
    decode_batch_size: Optional[int] = None,
    prefill_chunk_size: Optional[int] = None,
    batch_grid: Optional[List[dict]] = None,
    **kwargs: Any,
):
    """Multi-context, multi-batch variant of `estimate_capacity`: one walk,
    every (context, batch) point of it by substitution (`sweep_points`), every
    point's measured times from ONE lookup (`batch_sweep.measure_points`;
    docs/auto_sweep.md). Returns:
      - `batch_grid` given → `{key: [<per-seq>...]}` (one list per batch point),
      - else                → `[<per-seq>...]` ordered by `seq_lens`.
    Each point is the estimate a direct call there returns; the per-op rows
    it carries are the point's (read through their laws)."""
    plan, walked = sweep_points(
        seq_lens=seq_lens, config=config, gpu_sku=gpu_sku, knobs=knobs,
        avg_output_tokens=avg_output_tokens,
        prefill_batch_size=prefill_batch_size,
        decode_batch_size=decode_batch_size,
        prefill_chunk_size=prefill_chunk_size, batch_grid=batch_grid, **kwargs)
    measure_points([pt for cell in walked for pt in cell], **plan.measurement())
    return plan.shape([[pt["sol"] for pt in cell] for cell in walked])


@dataclass
class CapacityCeilings:
    """L1 — the capacity layer: how much fits, and at what batch.

    Pure budget arithmetic over one (model, GPU, parallel, seq) point: the KV
    pool divides into per-request footprints (`max_batch`); the activation
    budget divides into per-stream footprints (`max_prefill_batch`). This is
    the ONE question a deployment gate asks — does it fit, and up to what
    batch — and it depends on nothing above it.

    The layers above consume this and add meaning, never the reverse:
      L2 `estimate_phases`          — TTFT / TPOT / throughput at these batches
      L3 `_run_calibration_overlay` — per-op rows under those phases
      L4 formula rendering          — display strings over those rows
    """
    request_tokens: Any
    effective_storage_tokens: Any
    per_request_kv_bytes: Any
    kv_per_class_breakdown: Any
    max_batch: Any
    max_batch_total: Any
    infeasibility_reason: Any
    kv_per_rank: Any
    peak_act: Any
    est_frac: Any
    peak_act_per_b1: Any
    overhead_bytes: Any
    max_prefill_batch: Any
    max_prefill_batch_total: Any

    # --- The per-rank memory picture, and the stage it describes ---
    #
    # These are not new arithmetic. They are the numbers this layer ALREADY
    # computed for its own use and used to keep as locals, so a caller wanting
    # "the weights on this rank" had to go back to the preparation and got a
    # DIFFERENT stage's answer. Qwen3-Coder-Next at pp=8 is the cell that
    # showed it: the stage with the smallest batch at this length is not the
    # stage with the least room for cache, and the published weight figure came
    # from the second while `max_batch` beside it came from the first — 19.559
    # GiB reported against a heaviest stage holding 19.571, a 0.06% shortfall
    # in the direction that invents capacity.
    #
    # So the picture is published whole, with `stage` naming whose picture
    # it is. Every field below is that one stage's; mixing them is the bug.
    # Without pp there is one stage and these equal the preparation's.
    #
    # `stage` is the stage itself (`_StageProfile`): its index, its
    # cache reader and its per-token split. The compute the layers above
    # bill — the per-op rows, the phase FLOPs, the KV read law — is
    # restricted by the same object, so memory and compute describe one
    # rank. None when `pp == 1`.
    stage: Any
    #: The cache reader of that stage's layers (`CacheView.on`): the
    #: per-request footprint, the KV read law and the growth formula.
    kv_reader: Any
    weight_bytes_per_rank: Any
    kv_static_per_rank: Any
    kv_constant_state_per_rank: Any
    per_token_per_rank: Any
    kv_budget_bytes: Any
    #: Bytes the rank's weights exceed usable HBM by. 0 when the weights
    #: fit; published so a page can say how far short the card is without
    #: subtracting the two fields itself.
    weight_deficit_bytes: Any
    usable_for_tokens: Any
    fits: Any
    #: None when the cache does not grow with length: nothing bounds it.
    max_tokens: Any
    max_tokens_total: Any
    #: Whether this request's prompt runs past the model's published context
    #: (`s_max`). The prompt, not prompt + output: every page puts a request
    #: at its context length (a chart's x, the Plan's input tokens), and a
    #: 128K model serves a 128K context — a serving stack stops or truncates
    #: the generation at the limit. The numbers are still computed (the
    #: formulas do not stop at the limit), so a page that draws them must
    #: mark or drop such a point (`point_served`), and say why.
    beyond_max_context: bool

    @property
    def pp_stage(self) -> Optional[int]:
        """The index of `stage`, as the response names it."""
        return None if self.stage is None else self.stage.stage

    def refuse_unless_one_stage(self, rows_stage: Optional[int],
                                gpu_sku: str) -> None:
        """Refuse rows of pipeline stage `rows_stage` unless this card's
        ceilings bind that stage.

        The one statement of that rule, the sibling of
        `card.Card.refuse_unless_one_class` on the other axis a walk
        depends on a card by. Under `pp` the rows are one stage's, and a
        card with less HBM can run out on a different stage than the card
        the rows were walked on (MiMo-V2-Flash at pp=4: stage 0 on an
        H200, stage 1 on an H100) — one stage's rows under another stage's
        batch are no rank's answer. Asked by the walk
        (`PendingEstimate.priced_on`) and where a result is stamped
        (`_stamp_gpu_on_result`), which is two askers of one rule rather
        than two spellings of it.
        """
        if rows_stage == self.pp_stage:
            return
        raise ValueError(
            f"{gpu_sku}: the rows to re-price are pipeline stage "
            f"{rows_stage}'s, and this card's ceilings bind stage "
            f"{self.pp_stage}; re-price an estimate of stage {self.pp_stage}")


@stage_context("capacity")
def capacity_ceilings(
    p: _CapacityPreparation,
    *,
    avg_input_tokens: int,
    avg_output_tokens: int,
    prefill_chunk_size: Optional[int] = None,
    prefill_chunk_schedule: Optional[PrefillChunkSchedule] = None,
    prefill_capacity: bool = True,
) -> CapacityCeilings:
    """Compute L1 (see `CapacityCeilings`) from the seq-invariant preparation.

    Produces numbers only, so it runs with engine display strings suppressed:
    a formula popover is an L4 concern and nothing in this layer reads one. The
    layer is cheap by construction, not by a caller-side opt-out.

    `prefill_capacity=False` answers the serving anchor alone: decode capacity
    and a one-request prefill. It skips activation liveness, whose physical
    symbolic shapes are orders larger on high context-parallel layouts.
    """
    gpu = p.gpu
    parallel = p.parallel
    resolved = p.resolved
    kv_static_per_rank = p.kv_static_per_rank
    kv_div = p.kv_div
    weight_bytes_per_rank = p.weight_bytes_per_rank
    mem_total_bytes = p.mem_total_bytes
    usable_bytes = p.usable_bytes
    fits = p.fits
    usable_for_tokens = p.usable_for_tokens
    unique_tokens_multiplier = p.unique_tokens_multiplier
    kv_replication_factor = p.kv_replication_factor
    if prefill_chunk_schedule is not None and prefill_chunk_size is not None:
        raise TypeError(
            "pass a materialized prefill schedule or a chunk size, not both")
    if (prefill_chunk_schedule is not None
            and prefill_chunk_schedule.endpoint.seq_len != avg_input_tokens):
        raise ValueError(
            "the prefill schedule and capacity endpoint name different lengths "
            f"({prefill_chunk_schedule.endpoint.seq_len} vs {avg_input_tokens})")
    if (prefill_chunk_schedule is not None
            and prefill_chunk_schedule.endpoint.s_prefix_cached
            != p.s_prefix_cached):
        raise ValueError(
            "the prefill schedule and capacity endpoint name different prefixes")
    # Effective storage tokens per request — SWA with eviction caps at s_window
    # independent of cache_len (positions outside the window are dropped
    # by the runtime), so the per-request KV footprint is `min(s_window,
    # request_tokens)`. Dense and DSA keep the full `request_tokens` (DSA
    # still cache every position so the indexer can score them for
    # top-K; only read traffic shrinks, not storage).
    #
    # Hybrid backbones (MiMo V2.5 GQA+SWA, Qwen3-Next full+linear) need
    # per-class storage_positions and per-class per_token_per_rank — one
    # window-capped class can't represent the GQA layers, and one
    # unbounded class overcounts SWA layers. `per_request_kv_bytes_hybrid`
    # returns the per-class breakdown when classes>1, and falls back to
    # the single-class formula otherwise.
    request_tokens = avg_input_tokens + avg_output_tokens
    _reader = p.kv_reader

    # --- Under pp, which stage runs out first at THIS length? ---
    #
    # The preparation could not know. It ranks stages by cache tokens, which is
    # the right question for `max_tokens` and the wrong one for a batch: a
    # hybrid backbone's stages differ in how many of their layers CAP, and a
    # cap only shows up once requests are long enough to reach it. MiMo-V2.5 at
    # pp=8 reported 3770 concurrent requests where its worst stage could hold
    # 2182 — a 1.7x overstatement that no workload-free key can see.
    #
    # So the whole per-rank memory account is re-bound to the stage with the
    # smallest batch at `request_tokens` — or, when no stage fits at all, the
    # one furthest from fitting. Re-bound TOGETHER: weights, floors, budget and
    # per-request bytes all move to one stage, because a budget from one rank
    # divided by a footprint from another is not a rank's answer.
    #
    # The re-bound picture is PUBLISHED, not kept as locals here. It used to be
    # local, and the caller that wanted "the weights on this rank" read the
    # preparation's — a different stage — while reading `max_batch` from this
    # one. See the field block on `CapacityCeilings`.
    stage = p.pp_bound
    per_token_per_rank = p.per_token_per_rank
    kv_constant_state_per_rank = p.kv_constant_state_per_rank
    if p.pp_stages:
        def _batch_on(prof) -> Tuple[float, ...]:
            return _binding_key(prof.free_bytes(usable_bytes),
                                prof.reader.storage_bytes_at(request_tokens))
        stage = _bind = min(p.pp_stages, key=_batch_on)
        _reader = _bind.reader
        kv_static_per_rank = _bind.static
        kv_constant_state_per_rank = _bind.constant_state
        per_token_per_rank = _bind.per_token
        weight_bytes_per_rank = _bind.weight_bytes
        usable_for_tokens = max(0.0, _bind.free_bytes(usable_bytes))
        fits = _bind.free_bytes(usable_bytes) > 0

    # Cache tokens at the SAME stage the batch was bound on — the only place
    # they are counted, so the response carries one answer, not one per stage
    # choice.
    #
    # The signed remainder can be negative when weights overflow usable HBM.
    # `_cache_room` already clamps the *internal* leftover; the published
    # budget follows `fits`: a rank that does not fit has no KV pool, and
    # `weight_deficit_bytes` is the shortfall a page can show.
    signed_kv_budget = usable_bytes - weight_bytes_per_rank
    weight_deficit_bytes = max(0.0, weight_bytes_per_rank - usable_bytes)
    kv_budget_bytes = 0.0 if not fits else signed_kv_budget
    # A cache that does not grow with length (every layer windowed, or a
    # recurrent state) never runs out of room for tokens: past its floor a
    # request may run as long as it likes. The count is unbounded — None —
    # not zero; what the budget does bound is the batch, below.
    if not fits:
        max_tokens: Optional[int] = 0
    elif per_token_per_rank > 0:
        max_tokens = int(usable_for_tokens // per_token_per_rank)
    else:
        max_tokens = None
    max_tokens_total = (None if max_tokens is None
                        else max_tokens * unique_tokens_multiplier)

    effective_storage_tokens = _reader.storage_positions_at(
        float(request_tokens))
    if effective_storage_tokens <= 0:
        effective_storage_tokens = max(1, request_tokens)
    # Every tensor's data AND its dynamic scales under that tensor's own
    # storage law (a windowed layer's scales stop at its window with its
    # data). Static scales are not here — the budget subtracted them once
    # already, before dividing.
    per_request_kv_bytes = _reader.storage_bytes_at(request_tokens)
    kv_per_class_breakdown = _reader.storage_breakdown(request_tokens)
    max_batch = (
        int(usable_for_tokens // per_request_kv_bytes)
        if (fits and per_request_kv_bytes > 0) else 0
    )

    # Honest infeasibility note. `batch == 0` reaches phases.estimate_phase
    # for two structurally distinct reasons; the canned "weights exceed
    # budget" string used to fire on both, misleading users in the second
    # case. Disambiguate here, where fits / max_tokens / per-request are
    # still in scope, and let the phase function consume the precomposed
    # string.
    if max_batch > 0:
        infeasibility_reason: Optional[str] = None
    elif not fits:
        infeasibility_reason = _weight_overflow_reason(
            gpu=gpu, weight_bytes=weight_bytes_per_rank,
            usable_bytes=usable_bytes, static_bytes=kv_static_per_rank,
            constant_state_bytes=kv_constant_state_per_rank)
    else:
        infeasibility_reason = (
            f"KV budget ({usable_for_tokens / _GB:.2f} GiB) is too small for "
            f"one request at seq={request_tokens:,} "
            f"(needs {per_request_kv_bytes / _GB:.2f} GiB). "
            f"Use a shorter seq or raise TP_attention."
        )

    max_batch_total = max_batch * unique_tokens_multiplier

    # Scaled KV block surfaced for the UI (reflects per-rank values).
    # Built from the reader's entries — the labels come from the same
    # component declarations the attention badge resolver reads, so the
    # panel and the badge cannot disagree.
    _labels: dict = {}
    _classes: list = []
    for _entry, _count, *_ in _reader.entries:
        _lbl = _entry.label or _entry.variant or "?"
        _labels[_lbl] = _labels.get(_lbl, 0) + _count
        _classes.append({
            "label": _entry.label,
            "variant": _entry.variant or _lbl,
            "layer_count": _count,
            "compute_dims": {
                t.name: _counted(t.elements, t.name, "elements")
                for t in _entry.tensors
            },
        })
    # The type line is the family (MLA), not the assignment split.
    # Identical leaf-Σ across dense+MoE MLA still factors in
    # growth_formula as `(3 + 58) × …`; putting `MLA × 3 + MLA × 58` on
    # the badge hid the original law. The variant line is the distinct
    # labels.
    _variant_order: list = []
    for _c in _classes:
        _v = _c["variant"] or _c["label"] or "?"
        if _v not in _variant_order:
            _variant_order.append(_v)
    # Hybrid is the backbone running more than one attention / sequence
    # mixer (`Program.is_hybrid_attention`) — the badge's question, asked once.
    _program = Program.of(p.resolved)
    kv_per_rank = {
        "attention_type": next(iter(_labels), None),
        "attention_variant": " + ".join(_variant_order) or None,
        "attention_mix": list(_program.attention_mix),
        "is_hybrid_attention": _program.is_hybrid_attention,
        # Whether the cache grows by more than one law (layersplit's refusal).
        "mixed_growth_laws": _reader.mixed_growth_laws,
        "num_layers": sum(c["layer_count"] for c in _classes),
        "classes": _classes,
        "growth_formula": _reader.growth_formula,
    }
    # The law the budget divides by, published as it is used. A request's
    # cache is `per_token_bytes × L + constant_bytes` past every window
    # (`CacheView.growth_and_floor`): the slope is a Σ over cached tensors at
    # each tensor's own rate (a window adds nothing per token once full, a
    # compressed pool 1/m of a row), so it is well defined for a hybrid
    # backbone too — it is not "the" per-token footprint of one attention
    # class, and it is not a request's bytes at a length (that is
    # `per_request_kv_bytes`). Both terms are the bound stage's, per rank —
    # the same two `max_tokens` subtracts and divides by. `per_token_bytes`
    # is 0 for a cache that never grows (then `max_tokens` is None).
    kv_per_rank["per_token_bytes"] = float(per_token_per_rank)
    kv_per_rank["constant_bytes"] = float(kv_constant_state_per_rank)
    kv_per_rank["static_scale_bytes"] = kv_static_per_rank
    kv_per_rank["shard_divisor"] = kv_div
    kv_per_rank["replication_factor"] = kv_replication_factor
    kv_per_rank["unique_tokens_multiplier"] = unique_tokens_multiplier

    # KV read cardinality — replace the raw SymPy Expr with a string for
    # JSON serialization, and surface the resolved per-query positions
    # at decode's average cache length (≈ avg_in + avg_out/2).
    _kv_read_expr_obj = _reader.read_law
    decode_cache_len = max(1, int(avg_input_tokens + avg_output_tokens / 2.0))
    if _kv_read_expr_obj is not None:
        from backend.arch.eval.cache import positions_at
        kv_read_pos_decode = positions_at(_kv_read_expr_obj, decode_cache_len)
        kv_per_rank["kv_read_expr"] = sp.sstr(_kv_read_expr_obj)
    else:
        kv_read_pos_decode = decode_cache_len
        kv_per_rank["kv_read_expr"] = None
    kv_per_rank["kv_read_positions_at_decode_cache"] = kv_read_pos_decode
    kv_per_rank["decode_cache_len_for_read"] = decode_cache_len

    # Storage-side asymmetry. `kv_storage_expr` is the parallel of
    # `kv_read_expr` — `s_kv` for dense/DSA,
    # `Min(s_kv, s_window)` for SWA.
    _kv_storage_expr_obj = _reader.storage_law
    if _kv_storage_expr_obj is not None:
        kv_per_rank["kv_storage_expr"] = sp.sstr(_kv_storage_expr_obj)
    else:
        kv_per_rank["kv_storage_expr"] = None
    kv_per_rank["effective_storage_tokens_per_request"] = int(effective_storage_tokens)
    kv_per_rank["per_request_kv_bytes"] = per_request_kv_bytes
    # Hybrid backbones: surface the per-class breakdown (component label,
    # layer count, per-class storage_positions, per-class request bytes)
    # so the UI can show "GQA × 9 stores s_kv positions; SWA × 39 caps at
    # s_window"
    # — single-class models keep the legacy single-window display.
    if kv_per_class_breakdown:
        kv_per_rank["per_class_request"] = kv_per_class_breakdown
    if _reader.tp_shards_cache is None:
        raise ValueError(
            "the per-rank cache reader carries no physical geometry, so it "
            "cannot say whether tensor parallelism divides the cache")
    kv_per_rank["shard_note"] = (
        f"KV divided by {kv_div:.3g}× "
        f"({'CP only — no cached tensor carries a TP-divided axis' if not _reader.tp_shards_cache else 'the cache axes this deployment could split'})"
    )
    if kv_replication_factor > 1:
        kv_per_rank["shard_note"] += (
            f" · replicated ×{kv_replication_factor:.3g} "
            f"(identical copies this split did not divide)"
        )

    # Peak activation memory (liveness analysis on the block graph).
    # Uses prefill-scale inputs (batch=1, seq=peak_seq) where attention's
    # QK^T matrix is typically the dominant activation.
    #
    # Chunked prefill: real stacks (vLLM `max_num_batched_tokens`, SGLang
    # `chunked_prefill_size`) split a long prefill into fixed-size chunks,
    # so the peak activation per concurrent prefill stream is bounded by
    # the CHUNK, not the full sequence. Cap the activation-estimate seq at
    # the chunk; for seq ≤ chunk (or chunking off) it's the full-seq
    # footprint as before. This is what makes `max_prefill_batch` PLATEAU
    # past the chunk size instead of decaying ∝ seq.
    if prefill_capacity:
        peak_seq = max(1, int(avg_input_tokens or 1))
        if prefill_chunk_schedule is not None:
            peak_seq = min(peak_seq, prefill_chunk_schedule.launch_width)
        elif prefill_chunk_size and prefill_chunk_size > 0:
            peak_seq = min(peak_seq, int(prefill_chunk_size))
        peak_batch = 1
        from .step_eval import evaluate_step as _peak_eval
        activation_step = _peak_eval(
            resolved, parallel, phase="prefill", batch=peak_batch,
            input_tokens=peak_seq, output_tokens=0, precision=p.precision,
            cells=p.cells,
        )
        peak_act = estimate_peak_activation(
            resolved,
            batch=peak_batch,
            seq_len=peak_seq,
            # The whole deployment, not a scalar picked out of it: the walk
            # divides each axis by the world that owns it, so a model sharding
            # attention and the FFN at different degrees no longer has to pick
            # one.
            parallel=parallel,
            step=activation_step,
            framework_binding=p.framework_binding,
        )
        est_frac = estimated_max_fraction(
            peak_activation_bytes=peak_act["peak_activation_bytes_per_rank"],
            overhead_bytes=peak_act["overhead_bytes"],
            mem_total_bytes=mem_total_bytes,
        )
    else:
        peak_act = {}
        est_frac = None

    # Prefill is bounded by HBM activation memory, NOT KV cache: at prefill
    # time the KV pool is still empty (every request's KV is the run's
    # output, not the run's input). And prefill does NOT coexist with
    # decode — when you size the prefill forward you don't reserve for a
    # decode KV pool that isn't there. So the activation budget is the
    # normal `(1 - gpu_memory_fraction) × HBM` residual PLUS the entire KV
    # pool (`usable_for_tokens`) that decode would otherwise occupy — i.e.
    # the whole card minus weights and fixed overhead. Charging prefill
    # only the residual under-counted its batch by ~the KV-pool factor
    # (it pinned `max_prefill_batch` at 1 well before the model actually
    # OOMs). `peak_act["peak_activation_bytes_per_rank"]` is at batch=1,
    # seq=min(avg_input_tokens, prefill_chunk_size) — the per-forward
    # chunk, not the full prompt — divide the budget by that per-B footprint.
    peak_act_per_b1 = float(peak_act.get("peak_activation_bytes_per_rank") or 0)
    overhead_bytes = float(peak_act.get("overhead_bytes") or 0)
    act_budget = max(
        0.0,
        (1.0 - p.gpu_memory_fraction) * mem_total_bytes - overhead_bytes
        + max(0.0, usable_for_tokens),
    )
    if peak_act_per_b1 > 0 and fits:
        max_prefill_batch = max(1, int(act_budget // peak_act_per_b1))
    else:
        max_prefill_batch = 0 if not fits else 1
    max_prefill_batch_total = max_prefill_batch * unique_tokens_multiplier

    return CapacityCeilings(
        request_tokens=request_tokens,
        effective_storage_tokens=effective_storage_tokens,
        per_request_kv_bytes=per_request_kv_bytes,
        kv_per_class_breakdown=kv_per_class_breakdown,
        max_batch=max_batch,
        max_batch_total=max_batch_total,
        infeasibility_reason=infeasibility_reason,
        kv_per_rank=kv_per_rank,
        peak_act=peak_act,
        est_frac=est_frac,
        peak_act_per_b1=peak_act_per_b1,
        overhead_bytes=overhead_bytes,
        max_prefill_batch=max_prefill_batch,
        max_prefill_batch_total=max_prefill_batch_total,
        stage=stage,
        kv_reader=_reader,
        weight_bytes_per_rank=weight_bytes_per_rank,
        kv_static_per_rank=kv_static_per_rank,
        kv_constant_state_per_rank=kv_constant_state_per_rank,
        per_token_per_rank=per_token_per_rank,
        kv_budget_bytes=kv_budget_bytes,
        weight_deficit_bytes=weight_deficit_bytes,
        usable_for_tokens=usable_for_tokens,
        fits=fits,
        max_tokens=max_tokens,
        max_tokens_total=max_tokens_total,
        beyond_max_context=(p.max_context is not None
                            and avg_input_tokens > p.max_context),
    )


def point_served(*, max_batch_per_gpu: Any, beyond_max_context: bool,
                 requests: int) -> bool:
    """Whether a point of `requests` concurrent requests exists on this
    deployment at this context: the context is within the model's published
    length (`CapacityCeilings.beyond_max_context`), and that many requests'
    cache fits on one GPU (`max_batch_per_gpu`, the KV-bound per-GPU
    ceiling). A page that draws points and a plan that measures them ask
    this one question."""
    return (not beyond_max_context
            and int(max_batch_per_gpu or 0) >= int(requests))


def floor_form_on(form: CardForm, gpu: Any) -> FloorForm:
    """A step's `CardForm` bound to `gpu` as a point on it is priced
    (`_point_rows`): each dtype at the card's own column for it
    (`pipeline.leaf_peak`), the card's HBM bandwidth and its fabric."""
    return form.bind({dt: leaf_peak(gpu, dt) for dt in form.dtypes},
                     peak_bw=peak_hbm_bw(gpu), fabric=Fabric.of(gpu),
                     gpu_sku=gpu.sku)


def served_points(swept: Mapping[str, List[Any]], seq_lens: List[int],
                  batch_grid: List[Mapping[str, Any]]
                  ) -> Iterator[Tuple[str, dict, dict]]:
    """`(context, estimate, sweep point)` of every point of a gridded sweep
    over `seq_lens` (`{batch key: [point per context]}`) that serves its
    batch (`point_served`) with a decode time, one per (context, batch), in
    grid order: the ceiling batch (`max`) lands on a ladder batch when the
    ceiling is one, and that batch is drawn once."""
    seen: set = set()
    for bp in batch_grid:
        requests = bp["decode_batch_size"] or 1
        for seq, pt in zip(seq_lens, swept.get(bp["key"]) or [], strict=False):
            per = (pt or {}).get("sol") or {}
            if not point_served(max_batch_per_gpu=per.get("max_batch_per_gpu"),
                                beyond_max_context=bool(per.get("beyond_max_context")),
                                requests=requests):
                continue
            dec = (per.get("phases") or {}).get("decode") or {}
            rates = (dec.get("throughput") or {}).get("sol") or {}
            if not rates.get("time_per_output_token_s") or (
                    str(seq), dec.get("batch")) in seen:
                continue
            seen.add((str(seq), dec.get("batch")))
            yield str(seq), per, pt or {}


def card_form_of(per: Mapping[str, Any], gpu_sku: str, *, phase: str,
                 book: LawBook, floors_s: Tuple[float, float]
                 ) -> Optional[CardForm]:
    """A sweep point's phase step as a function of the card and the two
    floors (`CardForm`), read off its rows, or None when the step publishes
    no time. Bound to the card the point was priced on (`floor_form_on`) at
    the floors it was priced at (`floors_s`, link latency and launch floor
    in seconds) it must reproduce the step's own time; a step it does not
    is refused rather than carried wrong. A counted draft's forwards are in
    the form; a pipelined step is refused."""
    dec = (per.get("phases") or {}).get(phase) or {}
    t_sol = dec.get("t_sol_s")
    if not t_sol:
        return None
    if dec.get("prefill_chunk_map"):
        # A chunked endpoint is a sum of separately scheduled walks. Its
        # endpoint row laws do not state that sum as one launch rule, so the
        # card's own priced phase is carried instead of a misleading form.
        return None
    if dec.get("pipeline_time"):
        raise ValueError("a step's card form is one stage's step: this one "
                         "is pipelined")
    draft = dec.get("speculative_draft") or {}
    gpu = get_gpu(gpu_sku)
    rows = (per["per_op_timing"] if phase == "prefill"
            else per["per_op_timing_decode"])
    form = CardForm.of(rows, book=book,
                       fabric=Fabric.of(gpu),
                       draft_rows=(per.get("per_op_timing_draft") or ()
                                   if phase == "decode" else ()),
                       draft_forwards=(int(draft.get("forwards") or 0)
                                       if phase == "decode" else 0))
    at = floor_form_on(form, gpu).at(*floors_s)
    if abs(at - float(t_sol)) > 1e-9 * float(t_sol):
        raise ValueError(f"card form gives {at!r} s on {gpu_sku} for a step "
                         f"of {t_sol!r} s")
    return form


@dataclass(frozen=True)
class ClassSweep:
    """One gridded serving walk of a deployment (`sweep_points`), walked once
    for every card of a walk class (`framework_bindings.card_class`: the cards
    that walk a model alike) and read on any of them without walking again.

    What a card changes in such a sweep is its price and its capacity, not
    its rows. So the sweep keeps, per context: for each batch the grid fixes,
    the walk card's phases as `project` keeps them, each step as a function of
    the card and the floors (`CardForm`), and its measures; and per card its
    capacity there (`max_batch_per_gpu`, `beyond_max_context`) and its own
    point at each batch the card's ceiling sets, each read off a sweep ON that
    card — the
    capacity and the ceiling batch are the engine's answer for the card,
    not a second derivation of it. The walk card is the class's card with
    the most HBM: a deployment that fits no card of the class does not fit
    it either, so a fixed batch it cannot price no card can serve. A fixed
    point no card of the class serves is not kept.

    `on_card` reads the sweep on one of its cards: the same
    `{batch key: [point per context]}` a sweep on that card gives, its
    points served as that card serves them (`served_points`), each with its
    step's time and rates on that card at the sweep's floors, its step as a
    functions of the floors (`CardForm`, `forms`) and its measures keyed there
    (`batch_sweep.measures_on_card`)."""

    walk_card: str
    #: Per batch key the grid fixes: per context, the walk card's point or
    #: None.
    fixed: Dict[str, List[Optional[dict]]]
    #: Per card: `{"ceilings": [[max_batch_per_gpu, beyond_max_context]
    #: per context], "own": {ceiling batch key: [point or None per
    #: context]}}`, or `{"error": reason}` when the card's sweep failed.
    cards: Dict[str, dict]

    @classmethod
    def of(cls, swept: Mapping[str, Mapping[str, Any]], *, walk_card: str,
           batch_grid: List[Mapping[str, Any]],
           project: Callable[[str, Dict[str, Any]], Dict[str, Any]],
           book: LawBook, floors_s: Tuple[float, float]) -> "ClassSweep":
        """The class walk of `swept` — per card, its sweep (`{batch key:
        [point per context]}`, the walk card's over the whole grid, every
        other card's over the ceiling batches alone), or `{"error":
        reason}` — at the floors the sweeps were priced at.

        A card of another class than `walk_card` is refused here, by the
        one statement of that rule (`Card.refuse_unless_one_class`): the fixed
        batches below are the WALK CARD's points, read on each card
        through their forms (`on_card`), so a card that walks the model
        differently would be published with the walk card's step."""
        for sku in swept:
            Card.of(walk_card).refuse_unless_one_class(Card.of(sku))
        def point(pt: Any, gpu_sku: str) -> Optional[dict]:
            per = (pt or {}).get("sol") or {}
            if not per:
                return None
            phases: Dict[str, Dict[str, Any]] = {}
            forms: Dict[str, Dict[str, Any]] = {}
            groups: Dict[str, int] = {}
            gpuCounts: Dict[str, int] = {}
            for name, phase in (per.get("phases") or {}).items():
                form = card_form_of(per, gpu_sku, phase=name, book=book,
                                    floors_s=floors_s)
                phases[name] = project(name, phase)
                if form is not None:
                    forms[name] = form.to_json()
                elif phase.get("prefill_chunk_map"):
                    phases[name] = {**phases[name],
                                    "t_sol_s": phase.get("t_sol_s")}
                throughput = phase.get("throughput") or {}
                groups[name] = throughput.get("groups", 1)
                gpuCounts[name] = throughput.get("num_gpus", 1)
            return {
                "phases": phases,
                "forms": forms,
                "groups": groups,
                "num_gpus": gpuCounts,
                "facts": {name: per.get(name) for name in (
                    "parallel_warnings", "max_context_tokens",
                    "infeasibility_reason", "max_decode_batch_per_gpu",
                    "max_prefill_batch_per_gpu")},
                "measures": (pt or {}).get("measures") or {},
            }

        def ceilings(card: Mapping[str, Any]) -> List[List[Any]]:
            out: List[List[Any]] = []
            for at in zip(*card.values(), strict=True):
                per = next(((pt or {}).get("sol") or {} for pt in at
                            if (pt or {}).get("sol")), {})
                out.append([per.get("max_batch_per_gpu"),
                            bool(per.get("beyond_max_context"))])
            return out

        cards: Dict[str, dict] = {}
        for sku, card in swept.items():
            if "error" in card:
                cards[sku] = {"error": str(card["error"])}
                continue
            own = {
                bp["key"]: [point(pt, sku) for pt in card[bp["key"]]]
                for bp in batch_grid
                if not bp["decode_batch_size"]
                and (sku == walk_card or bp["key"] != "max")}
            if sku != walk_card and "capacity" in own:
                own["max"] = own["capacity"]
            cards[sku] = {"ceilings": ceilings(card), "own": own}
        if "error" in cards[walk_card]:
            raise ValueError(cards[walk_card]["error"])
        fixed: Dict[str, List[Optional[dict]]] = {}
        for bp in batch_grid:
            if not bp["decode_batch_size"]:
                continue
            fixed[bp["key"]] = [
                point(pt, walk_card)
                if any(point_served(max_batch_per_gpu=card["ceilings"][i][0],
                                    beyond_max_context=card["ceilings"][i][1],
                                    requests=bp["decode_batch_size"])
                       for card in cards.values() if "error" not in card)
                else None
                for i, pt in enumerate(swept[walk_card][bp["key"]])]
        return cls(walk_card, fixed, cards)

    def on_card(self, gpu_sku: str, *, batch_grid: List[Mapping[str, Any]],
                floors_s: Tuple[float, float]) -> Dict[str, List[Optional[dict]]]:
        """The sweep on `gpu_sku`, one of its cards, at the floors
        `floors_s`. A card the sweep was not walked for is refused, and a
        card whose own sweep failed raises its reason."""
        card = self.cards.get(gpu_sku)
        if card is None:
            raise KeyError(f"{gpu_sku} is not a card this sweep was walked for "
                           f"({', '.join(self.cards)})")
        if "error" in card:
            raise ValueError(card["error"])
        gpu = get_gpu(gpu_sku)

        def priced(point: Optional[dict], ceiling: List[Any]) -> Optional[dict]:
            if point is None:
                return None
            phases: Dict[str, Dict[str, Any]] = {}
            forms: Dict[str, Dict[str, Any]] = {}
            for name, projection in point["phases"].items():
                phase = dict(projection)
                if name in point["forms"]:
                    form = floor_form_on(
                        CardForm.from_json(point["forms"][name]), gpu)
                    phase["t_sol_s"] = form.at(*floors_s)
                    forms[name] = form.to_json()
                if "binding" in phase:
                    phase["throughput"] = phase_throughput(
                        phase, unique_tokens_multiplier=point["groups"][name],
                        num_gpus=point["num_gpus"][name])
                phases[name] = phase
            return {"sol": {"max_batch_per_gpu": ceiling[0],
                            "beyond_max_context": ceiling[1],
                            "phases": phases, **point["facts"]},
                    "measures": measures_on_card(point["measures"], gpu_sku),
                    "forms": forms}

        return {bp["key"]: [priced(pt, card["ceilings"][i]) for i, pt in enumerate(
                    self.fixed[bp["key"]] if bp["decode_batch_size"]
                    else card["own"][bp["key"]])]
                for bp in batch_grid}

    def to_json(self) -> Dict[str, Any]:
        return {"walk_card": self.walk_card, "fixed": self.fixed,
                "cards": self.cards}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "ClassSweep":
        return cls(str(d["walk_card"]), dict(d["fixed"]), dict(d["cards"]))


def _phase_inputs(p: _CapacityPreparation, _c: "CapacityCeilings", *,
                  avg_input_tokens: int,
                  avg_output_tokens: int,
                  s_prefix_cached: int) -> PhaseInputs:
    """What both phases at one context are priced against
    (`phases.PhaseInputs`), off the preparation and that length's ceilings;
    `s_prefix_cached` of the prompt is cached.
    The verify forward reads the backbone's cache; a draft's cache, kept in
    the same budget, is read by the draft's own rows. The compute is billed
    on the same stage the memory account describes (`_c.stage`), so
    `active_params_per_rank` (which comes out of that stage's weight tree),
    the KV read and `flops_per_step` are one rank's."""
    verify_reader = _c.kv_reader
    if p.draft is not None:
        verify_reader = verify_reader.within(
            verify_reader.source_analysis.program.included_stack_names())
    return PhaseInputs(
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        weight_bytes_per_rank=_c.weight_bytes_per_rank,
        per_token_kv_bytes_per_rank=_c.per_token_per_rank,
        static_kv_bytes_per_rank=_c.kv_static_per_rank,
        resolved_spec=p.resolved,
        parallel=p.parallel,
        s_prefix_cached=s_prefix_cached,
        s_verify=p.s_verify,
        acceptance_rate=p.acceptance_rate,
        kv_read_expr=verify_reader.read_law,
        kv_reader=verify_reader,
        infeasibility_reason=_c.infeasibility_reason,
        stage=_c.pp_stage,
    )


def _phase_at_length(phase_dict: Mapping[str, Any], p: _CapacityPreparation,
                     _c: "CapacityCeilings", *, avg_input_tokens: int,
                     avg_output_tokens: int,
                     s_prefix_cached: int) -> Dict[str, Any]:
    """`phase_dict` (an estimate's phase at another context) restated at
    `avg_input_tokens` with `s_prefix_cached` of it cached, at `p`'s verify
    width and draft (a walk shared across verify widths,
    `_CapacityPreparation.walks_alike`): the step's
    binding there (`phase_binding`) and what the phase says about the
    request (`phase_frame`). Its numbers are the point's rows'
    (`batch_sweep.phase_at`). A counted draft's FLOPs line keeps the walk's
    forward, as a point at another batch does."""
    phase = phase_dict["phase"]
    bind = phase_binding(phase=phase, avg_input_tokens=avg_input_tokens,
                         avg_output_tokens=avg_output_tokens,
                         s_prefix_cached=s_prefix_cached,
                         s_verify=p.s_verify)
    drafted = phase == "decode" and bool(phase_dict.get("speculative_draft"))
    note_detail, flops_formula = phase_frame(
        _phase_inputs(p, _c, avg_input_tokens=avg_input_tokens,
                      avg_output_tokens=avg_output_tokens,
                      s_prefix_cached=s_prefix_cached),
        phase=phase, cache_len=bind["cache_len"],
        draft_spec=p.draft if drafted else None,
        draft_forward_flops=0.0)
    if drafted:
        flops_formula = phase_dict["formulas"]["flops"]
    return {
        **phase_dict,
        "seq_effective": bind["s_query"],
        "kv_cache_length": bind["cache_len"],
        "note_detail": note_detail,
        "binding": {**phase_dict["binding"],
                    "s_query": bind["s_query"], "s_kv": bind["s_kv"],
                    "s_pairs": int(bind["s_pairs"]),
                    "cache_len": bind["cache_len"],
                    "tokens_per_step": bind["tokens_per_step"],
                    "s_prefix_cached": bind["s_prefix_cached"],
                    "s_verify": bind["s_verify"]},
        "formulas": {**phase_dict["formulas"], "flops": flops_formula},
        # The draft `p` counts — its K, and so its forwards, are the
        # verify width's; the point prices it (`batch_sweep.phase_at`).
        **({"speculative_draft": {**phase_dict["speculative_draft"],
                                  **p.draft.to_wire()}} if drafted else {}),
    }


def _capacity_fields(p: _CapacityPreparation, _c: "CapacityCeilings",
                     wb: Mapping[str, Any], *,
                     avg_input_tokens: int) -> Dict[str, Any]:
    """The capacity half of an estimate at one prompt length: the per-rank
    memory picture, the cache and the batch ceilings at that length
    (`capacity_ceilings`), and their formulas — what an estimate states
    beside its phases, and what a sweep's point at another length states
    there (`_sweep_cell`)."""
    weight_bytes_per_rank = _c.weight_bytes_per_rank
    return {
        "avg_input_tokens": avg_input_tokens,
        "memory": {
            "total_bytes": p.mem_total_bytes,
            "usable_bytes": p.usable_bytes,
            "weight_bytes_total": p.weight_bytes_total,
            "weight_bytes_per_gpu": weight_bytes_per_rank,
            # The host-offloadable weights on this rank
            # (`Program.host_offloadable`), and whether they are in host
            # memory — then not in the weights above, and the GPU's room
            # for cache is larger by them.
            "host_offload_bytes_per_gpu":
                p.host_offload_bytes[_c.pp_stage or 0],
            "host_offload": p.host_offload,
            "kv_budget_bytes": _c.kv_budget_bytes,
            "weight_deficit_bytes": _c.weight_deficit_bytes,
            "static_scale_bytes": _c.kv_static_per_rank,
            "fits": _c.fits,
            # WHOSE per-rank picture the five numbers above are. Every one of
            # them describes this pipeline stage and no other — the stage that
            # binds the batch at this length, which is also the stage that
            # fails to fit first when one does, because a stage in deficit
            # sorts ahead of every stage in surplus. None when `pp == 1`.
            #
            # Named because it is not the only defensible stage to show. A
            # deployment asking "can a rank hold this" wants the HEAVIEST
            # stage, and it is not always this one: DeepSeek-V3 at pp=2 binds
            # on stage 0's 38.63 GiB while stage 1 holds 41.10. Both are
            # published, and neither has to be inferred from the other.
            "pp_stage": _c.pp_stage,
            "weight_bytes_per_gpu_heaviest_stage": (
                max((s.weight_bytes for s in p.pp_stages),
                    default=weight_bytes_per_rank)),
        },
        "kv_cache": _c.kv_per_rank,
        "max_tokens_per_gpu": _c.max_tokens,
        "max_tokens_total": _c.max_tokens_total,
        # `max_batch_*` are decode aliases, read by the web pages
        # (web/src/pages: Inference, ResultSummary, ScalingPanel, Compare*)
        # and the sweep payload; the explicit
        # `max_{prefill,decode}_batch_*` siblings name the phase.
        "max_batch_per_gpu": _c.max_batch,
        "max_batch_total": _c.max_batch_total,
        # Why `max_batch_per_gpu` is 0 here; None when it serves. The same
        # field a GPU-restamped result carries (`_stamp_gpu_on_result`).
        "infeasibility_reason": _c.infeasibility_reason,
        "max_decode_batch_per_gpu": _c.max_batch,
        "max_decode_batch_total": _c.max_batch_total,
        "max_prefill_batch_per_gpu": _c.max_prefill_batch,
        "max_prefill_batch_total": _c.max_prefill_batch_total,
        "activation_peak": _c.peak_act,
        "estimated_max_fraction": _c.est_frac,
        # Whether this request's prompt runs past the model's published
        # context length (`CapacityCeilings.beyond_max_context`).
        "beyond_max_context": _c.beyond_max_context,
        "formulas": capacity_formulas(
            preparation=p, ceilings=_c, weight_breakdown=wb),
    }


def _capacity_pending(
    p: _CapacityPreparation,
    *,
    avg_input_tokens: int,
    avg_output_tokens: int,
    # Per-phase batch overrides. None → defaults: prefill=1 (one-shot
    # serving anchor), decode=max_batch (KV-bound ceiling). When set, the
    # value also drives the phase's calibration overlay anchor so per-op
    # rows reflect the chosen B.
    prefill_batch_override: Optional[int] = None,
    decode_batch_override: Optional[int] = None,
    # Chunked-prefill token budget (vLLM `max_num_batched_tokens`,
    # SGLang `chunked_prefill_size`). When set, the prefill activation
    # footprint is bounded by min(seq, chunk) — the per-forward chunk —
    # not the full sequence. None / 0 → un-chunked (peak ∝ full seq).
    prefill_chunk_size: Optional[int] = None,
    # The media each request carries, for the encode phase; None asks for
    # no encode phase (a sweep's points). The encode batch defaults to the
    # prefill batch.
    media: Optional[Mapping[str, Mapping[str, int]]] = None,
    encode_batch_override: Optional[int] = None,
    prefill_chunk_schedule: Optional[PrefillChunkSchedule] = None,
    fold_prefill_chunks: bool = True,
    prefill_capacity: bool = True,
    breakdown: bool,
) -> "PendingEstimate":
    """Per-seq second half of `estimate_capacity` — applies the
    seq-dependent math (max_batch, per-request KV, peak activation,
    phases, calibration overlay, formulas, response dict) on top of
    the invariant context produced by `_capacity_preparation`.

    `breakdown`: whether the walks' rows carry what an estimate's breakdown
    draws, and its phases their module tables (`overlay_walk`) — an
    estimate's do; the walk a sweep reads its points off does not
    (`_sweep_cell`), whose points carry none of them.

    In two steps, so an estimate looks its measurements up once: this one
    runs everything up to the overlays' structural walks and returns them;
    `finish()` runs the rest with no measurement read, and the estimate's
    measured totals (`PendingEstimate.measures`) are folded over one lookup
    as a one-point sweep's are (`finish_estimates`, `_capacity_finish`)."""
    if avg_input_tokens < 0 or avg_output_tokens < 0:
        raise ValueError("avg_input_tokens and avg_output_tokens must be ≥ 0")
    if prefill_chunk_schedule is None:
        prefill_chunk_schedule = PrefillChunkSchedule.materialize(
            PrefillEndpointGeometry(
                avg_input_tokens, p.s_prefix_cached, 1, 1),
            prefill_chunk_size=prefill_chunk_size)
    elif prefill_chunk_schedule.endpoint.seq_len != avg_input_tokens:
        raise ValueError(
            "the prefill schedule and estimate endpoint name different lengths "
            f"({prefill_chunk_schedule.endpoint.seq_len} vs "
            f"{avg_input_tokens})")
    if prefill_chunk_schedule.endpoint.s_prefix_cached != p.s_prefix_cached:
        raise ValueError(
            "the prefill schedule and estimate endpoint name different "
            "workload facts")

    # Re-bind frequently-used invariants as locals for readability —
    # the rest of this function reads almost identically to the pre-
    # refactor estimate_capacity body, just sourcing from `p.*`.
    gpu = p.gpu
    parallel = p.parallel
    resolved = p.resolved
    weight_bytes_total = p.weight_bytes_total
    unique_tokens_multiplier = p.unique_tokens_multiplier
    kv_replication_factor = p.kv_replication_factor
    # NOTE: the per-rank memory picture — weights, floors, budget, cache
    # tokens — is NOT taken from `p` here. Under pp the preparation bound it to
    # the stage with the least room for cache, which is the right stage for a
    # workload-free question and not necessarily the one this length's batch is
    # bound on. Those fields arrive below, off `_c`, from the single stage that
    # layer picked. See the field block on `CapacityCeilings`.
    peak_flops = p.peak_flops
    peak_bw = p.peak_bw

    # L1 — capacity ceilings (see `capacity_ceilings`). Everything below this
    # line is L2+: it interprets these numbers, it never recomputes them.
    _c = capacity_ceilings(
        p,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        prefill_chunk_schedule=prefill_chunk_schedule,
        prefill_capacity=prefill_capacity,
    )
    max_batch = _c.max_batch
    max_prefill_batch = _c.max_prefill_batch

    wb = p.wb
    if _c.pp_stage is not None and _c.pp_stage != wb["pipeline"]["stage"]:
        # The tree the scalars are quoted beside has to describe the same
        # stage they came from, or the response shows a module breakdown that
        # does not add up to the total printed above it. The preparation does
        # exactly this when ITS argmin disagrees with the heaviest stage; this
        # is the same rebuild at the same cost, for the one length in a sweep
        # where the batch-bound stage differs from the token-bound one.
        wb = p.weight_tree_at(_c.pp_stage)

    # Per-call batch overrides win; otherwise default to the per-request
    # prefill (batch=1). The sentinel `-1` requests the capacity-admitted
    # prefill batch (`max_prefill_batch`) — used by the compare page's
    # throughput/TPOT charts as their "max" anchor so the four B curves
    # (1, max/16, max/4, max) actually span a range instead of all
    # collapsing onto B=1. (Decode already auto-maxes by default; prefill
    # stays B=1 by default so the single-request page shows per-request
    # prefill latency, not the capacity-batch number.)
    #
    # The bare 1 is the anchor, and it stays the anchor whether or not the
    # deployment admits a request: prefill's default view is the single-request
    # step. Whether that step can run is `infeasibility_reason`'s answer, and
    # it is passed to `estimate_phase` beside this — GLM-5.2 on eight H100s
    # (175.65 GB of weights against 72.00 GB usable) withholds both phases'
    # times against that reason, rather than having this line shrink the batch
    # to 0 so the phase can infer the same verdict back out of it.
    eff_prefill_batch = (
        max_prefill_batch
        if prefill_batch_override == -1
        else int(prefill_batch_override)
        if prefill_batch_override is not None
        else 1
    )
    eff_decode_batch = (
        int(decode_batch_override)
        if decode_batch_override is not None
        else max_batch
    )
    prefill_chunk_schedule = prefill_chunk_schedule.with_endpoint_batch(
        eff_prefill_batch)

    if p.refusal_reason is not None:
        reason = p.refusal_reason

        def refused_estimate() -> dict:
            """The complete no-step answer; no phase walk can load its weights."""
            phases = {
                name: {"phase": name, "batch": None,
                       "time_unpublished_reason": reason}
                for name in ("prefill", "decode")
            }
            return without_step({
                "gpu": gpu.to_json(),
                "num_gpus": p.num_gpus,
                "cluster_size": p.cluster_size,
                "parallel": {
                    "pp": parallel.pp, "tp_attention": parallel.tp_attention,
                    "tp_dense_ffn": parallel.tp_dense_ffn,
                    "tp_moe_expert": parallel.tp_moe_expert,
                    "ep_moe": parallel.ep_moe, "cp": parallel.cp,
                    "dp_attention": parallel.dp_attention,
                    "dp_replicas": parallel.dp_replicas,
                },
                "gpu_memory_fraction": p.gpu_memory_fraction,
                "weight_bytes_total": weight_bytes_total,
                "avg_output_tokens": avg_output_tokens,
                **_capacity_fields(
                    p, _c, wb, avg_input_tokens=avg_input_tokens),
                "peak_flops_per_sec": p.peak_flops,
                "peak_source": p.peak_source_s,
                "peak_hbm_bw_bytes_per_sec": p.peak_bw,
                "unique_tokens_multiplier": p.unique_tokens_multiplier,
                "kv_replication_factor": p.kv_replication_factor,
                "phases": phases,
                "weight_breakdown": wb,
                "precision": p.precision_report,
                "parallel_warnings": wb.get("parallel_warnings", []),
                "assumptions": [],
                "roles_present": p.roles_present,
                "moe_enabled": Program.of(p.resolved).has_dispatch_zone,
                "shape_warnings": p.shape_warnings,
                "max_context_tokens": p.max_context,
            }, ())

        return PendingEstimate(
            scope=p.measurement_scope,
            calibration_rows=p.calibration_rows,
            reads=p.enable_measured_lookup,
            walks=(),
            finish=refused_estimate,
            ceilings=_c,
            walked_on=p.card,
            selection_mode=p.selection_mode,
            kernel_mix=p.kernel_mix,
            preparation=p,
            prefill_chunk_schedule=prefill_chunk_schedule)

    # The two steps this estimate describes, each evaluated ONCE. Everything
    # below is a view: the phase table folds them by module, the per-op rows
    # flatten them to leaves, the cost tree reads the physical rows as-is. They
    # are built here — the only place that holds both the workload and the
    # deployment — because a view that built its own would be free to bind it
    # differently, and for a long time three of them did.
    #
    # Lazy inside, so a caller with no resolved spec pays for nothing.
    # A spec that cannot evaluate raises; estimate_phase no longer
    # swallows that into the no-graph closed form.
    prefill_step_eval = decode_step_eval = None
    if resolved is not None:
        from .step_eval import evaluate_step
        for _phase, _batch in (("prefill", eff_prefill_batch),
                               ("decode", eff_decode_batch)):
            _ev = evaluate_step(
                resolved, parallel, phase=_phase,
                batch=max(1, int(_batch or 1)),
                input_tokens=avg_input_tokens,
                output_tokens=avg_output_tokens,
                s_prefix_cached=p.s_prefix_cached, s_verify=p.s_verify,
                ep_load_factor=p.ep_load_factor,
                precision=p.precision, cells=p.cells)
            if _phase == "prefill":
                prefill_step_eval = _ev
            else:
                decode_step_eval = _ev

    # The per-op rows come FIRST, because the phase table is a view of them.
    # They used to come second and be substituted into a table that had already
    # published its own weights-and-KV closed form under the same field names —
    # so a phase's bytes, its arithmetic intensity and its step time all
    # depended on whether the overlay below had raised, at a fifth of the bytes
    # when it had. Now the graph is computed once and folded, and there is no
    # substitution to be conditional on.
    #
    # Always built (rooftop even with no calibration data) so the merged
    # Per-Module Breakdown's `Pre t_op` column always has data.
    overlay_kwargs = dict(
        resolved=resolved,
        parallel=parallel,
        framework=p.framework,
        framework_binding=p.framework_binding,
        gpu_sku=p.gpu_sku,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        s_prefix_cached=p.s_prefix_cached,
        s_verify=p.s_verify,
        # `max_batch` is the per-phase overlay anchor — per_op rows are
        # computed at this B and the frontend's `_inferBatchCache`
        # rescales linearly to whatever the dropdown shows. Anchor each
        # phase at its own batch (user override > physical max) so the
        # cache covers the full per-tab range.
        max_batch=eff_decode_batch,
        precision=p.precision,
        peak_flops=peak_flops,
        peak_bw=peak_bw,
        gpu=p.gpu,
        peak_intra_bw=p.peak_intra_bw,
        peak_inter_bw=p.peak_inter_bw,
        intra_bw_label=p.intra_bw_label_s,
        inter_bw_label=p.inter_bw_label_s,
        comm_latency_s=p.comm_latency_s,
        launch_floor_s=p.launch_floor_s,
        model_id=p.model_id,
        kv_block_size=p.kv_block_size,
        # The stage the memory picture above describes (`_c.stage`), so the
        # per-op rows are that rank's.
        stage=_c.pp_stage,
        breakdown=breakdown,
    )
    # Per-phase overlay anchors: decode at its KV-bound max (or user
    # override), prefill at its activation-bound max (or user override).
    # Distinct anchors → each phase's per_op rows are computed at its own
    # ceiling, and the scaling-cache rescale below that B stays linear.
    prefill_overlay_kwargs = {**overlay_kwargs, "max_batch": eff_prefill_batch,
                              "step_eval": prefill_step_eval}
    decode_overlay_kwargs = {**overlay_kwargs, "max_batch": eff_decode_batch,
                             "step_eval": decode_step_eval}
    # Structural layout analysis is part of the estimate, not a
    # best-effort calibration lookup. Missing benchmark rows already fall back
    # inside CalibrationOverlay; any exception here rejects the estimate
    # analysis and must abort the request rather than publish a communication-
    # free closed form.
    prefill_walk = overlay_walk(
        phase="prefill", pipeline_stages=parallel.pp, **prefill_overlay_kwargs)
    decode_walk = overlay_walk(
        phase="decode", pipeline_stages=parallel.pp, **decode_overlay_kwargs)

    # The speculative draft, when counted: one forward through the MTP stack,
    # priced on the step it runs — a K-token decode step for a parallel
    # draft, a 1-token one for a recursive draft — with only the draft
    # stack's leaves timed.
    draft_walk = _draft_eval = None
    if p.draft is not None:
        _draft_eval = p.draft.step(
            resolved, parallel,
            batch=eff_decode_batch,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            ep_load_factor=p.ep_load_factor,
            precision=p.precision, cells=p.cells)
        draft_walk = overlay_walk(
            phase="decode", want_module_rows=False, **{
                **decode_overlay_kwargs,
                # The MTP stack lives on the LAST stage (placed after its
                # latest computed input), which is not necessarily the stage
                # the memory account binds. Price it where it runs — on the
                # bound stage its rows can come out empty and the draft
                # silently drops out of the pipeline step.
                "stage": (parallel.pp - 1) if parallel.pp > 1 else None,
                "s_verify": p.verify_widths()["per_op_timing_draft"],
                "step_eval": _draft_eval,
                "timed_stacks": p.draft.stacks,
            })

    # The encoders, each its own step over the request's media: never part
    # of the prefill / decode figures below.
    encode_plan = None
    if media is not None and resolved is not None:
        encode_plan = plan_encode(
            resolved=resolved, parallel=parallel,
            batch=(eff_prefill_batch if encode_batch_override is None
                   else int(encode_batch_override)),
            media=media, precision=p.precision,
            overlay_kwargs=overlay_kwargs,
            # vLLM's `max_num_batched_tokens`: the encoder cache's budget.
            scheduler_tokens=prefill_chunk_schedule.prefill_chunk_size)
    walks = [w for w in (prefill_walk, decode_walk, draft_walk) if w is not None]
    if encode_plan is not None:
        walks += encode_plan.walks

    # Each row list's `PointRow`s at the walk's point, filled by `finish`:
    # what the estimate's measured fold keys its rows by.
    keyed: Dict[str, list] = {}
    measurement_rows: Dict[str, list] = {}

    def head_repartition_kv_producers() -> list[str]:
        """Aggregation ids of the Ulysses K/V decompression region.

        This is layout provenance, not a shape-derived guess: the reverse
        dataflow walk marked the row-wise producers between whole-cache views
        and attention.  The invariant battery reads it to state what PCP owes
        those rows, then still checks the actual walked shapes and FLOPs.
        """
        if prefill_step_eval is None:
            return []
        # The region as the served graph has it (`ServingLayers.kv_region`):
        # the rows the layout walk named, and the casts grafted onto their
        # operands, which run on the same rows.
        result = prefill_walk.analysis
        return sorted({
            node.aggregation_id for uid in result.kv_region
            if (node := result.pre_program.by_uid(uid)) is not None})

    def finish() -> dict:
        """The rest of the estimate — no measurement read: the measured track
        is the measured fold's (`finish_estimates`, `batch_sweep.
        measure_points`)."""
        # The per-op rows, and which leaves route tokens to experts as the
        # per-op arithmetic decided it — the prefill rows'.
        (
            per_op_timing, fusion_prefill, has_dispatch_zone,
            module_rows_prefill, pipeline_time_prefill, keyed["per_op_timing"],
        ) = prefill_walk.finish()
        (
            per_op_timing_decode, fusion_decode, _,
            module_rows_decode, pipeline_time_decode, keyed["per_op_timing_decode"],
        ) = decode_walk.finish()
        # What fusion changes, per phase: each side's theoretical step time.
        vs_naive_delta = {"prefill": fusion_prefill, "decode": fusion_decode}

        per_op_timing_draft: list = []
        decode_draft = None
        if draft_walk is not None:
            per_op_timing_draft, *_rest, keyed["per_op_timing_draft"] = draft_walk.finish()
            draft_book = draft_walk.analysis.laws.book
            full_law = p.draft.forward_flops_law(_draft_eval, draft_book)
            decode_draft = p.draft.price(
                per_op_timing_draft, forward_full=ForwardFull(
                    full_law, dict(_draft_eval.point),
                    float(Point(_draft_eval.point).laws(
                        draft_book, (full_law,))[full_law])))

        encode = encode_plan.finish() if encode_plan is not None else None
        kv_producers = head_repartition_kv_producers()

        # Prefill / decode per-step metrics (P2b — memory + compute, no comm/latency),
        # folding the rows above. A phase whose rows carry no bytes gets none: an
        # empty list is a missing graph: estimate_phase raises rather than
        # publishing resident-only traffic under the same field names.
        phases = estimate_phases(
            _phase_inputs(p, _c, avg_input_tokens=avg_input_tokens,
                          avg_output_tokens=avg_output_tokens,
                          s_prefix_cached=p.s_prefix_cached),
            prefill_batch=eff_prefill_batch,
            decode_batch=eff_decode_batch,
            prefill_step_eval=prefill_step_eval,
            decode_step_eval=decode_step_eval,
            prefill_rows=per_op_timing,
            decode_rows=per_op_timing_decode,
            decode_draft=decode_draft,
            # pp>1 publishes the pipeline-step time (Σ stages + hops); None at
            # pp=1, where the phase's own row sums are the step time already.
            pipeline_time=(
                None if pipeline_time_prefill is None
                else {"prefill": pipeline_time_prefill,
                      "decode": pipeline_time_decode}
            ),
        )

        # The per-module table comes off the grafted + fused tree, where
        # collective, cast and fused rows are native: the same tree whose
        # per-op rows the phase summed for its FLOPs, bytes and time, so the
        # table and the headline describe one step by construction. A phase
        # walks no table of its own. The speculative draft's depth-0 row is
        # appended here.
        if "prefill" in phases:
            phases["prefill"]["flops_by_module"] = module_rows_prefill
        if "decode" in phases:
            phases["decode"]["flops_by_module"] = module_rows_decode
            if decode_draft is not None:
                phases["decode"]["flops_by_module"].append(
                    decode_draft.module_row())
        if "prefill" in phases:
            enrich_module_breakdown(phases["prefill"])
        if "decode" in phases:
            enrich_module_breakdown(phases["decode"])
        for phase_dict in phases.values():
            phase_dict["throughput"] = phase_throughput(
                phase_dict, unique_tokens_multiplier=p.unique_tokens_multiplier,
                num_gpus=p.cluster_size)

        result = {
            "gpu": gpu.to_json(),
            "num_gpus": p.num_gpus,
            "cluster_size": p.cluster_size,
            "parallel": {
                "pp": parallel.pp, "tp_attention": parallel.tp_attention,
                "tp_dense_ffn": parallel.tp_dense_ffn,
                "tp_moe_expert": parallel.tp_moe_expert, "ep_moe": parallel.ep_moe,
                "cp": parallel.cp, "dp_attention": parallel.dp_attention,
                "dp_replicas": parallel.dp_replicas,
            },
            "gpu_memory_fraction": p.gpu_memory_fraction,
            "weight_bytes_total": weight_bytes_total,
            "avg_output_tokens": avg_output_tokens,
            **_capacity_fields(p, _c, wb, avg_input_tokens=avg_input_tokens),
            "peak_flops_per_sec": peak_flops,
            "peak_source": p.peak_source_s,
            "peak_hbm_bw_bytes_per_sec": peak_bw,
            "unique_tokens_multiplier": unique_tokens_multiplier,
            "kv_replication_factor": kv_replication_factor,
            "phases": phases,
            "weight_breakdown": wb,
            # What the precision profile is and what it changed on this model
            # (`Analysis.precision_effective`): the one precision block.
            "precision": p.precision_report,
            "parallel_warnings": wb.get("parallel_warnings", []),
            # What this estimate assumed that its inputs did not determine — see
            # `assumptions.deployment_assumptions`. Distinct from
            # `parallel_warnings`, which says a deployment is degenerate; an
            # assumption says the numbers are answering a slightly easier question
            # than the one asked, and in which direction.
            "assumptions": deployment_assumptions(
                parallel, ep_load_factor=p.ep_load_factor,
                has_dispatch_zone=has_dispatch_zone,
                joined_streams=(
                    sorted(prefill_step_eval.analysis.program.joined_streams)
                    if prefill_step_eval is not None else ())),
            "roles_present": p.roles_present,
            # Whether the model routes tokens to experts: its graph declares a
            # dispatch zone. The one MoE answer every page reads.
            "moe_enabled": Program.of(p.resolved).has_dispatch_zone,
            "shape_warnings": p.shape_warnings,
            # The model's published context length.
            "max_context_tokens": p.max_context,
            **({"head_repartition_kv_producers": kv_producers}
               if kv_producers else {}),
            **({"per_op_timing": per_op_timing} if per_op_timing else {}),
            **({"per_op_timing_decode": per_op_timing_decode}
               if per_op_timing_decode else {}),
            # One draft forward's rows; the decode phase's `speculative_draft`
            # says how many forwards the step runs.
            **({"per_op_timing_draft": per_op_timing_draft}
               if per_op_timing_draft else {}),
            "vs_naive_delta": vs_naive_delta,
            **({"encode": encode} if media is not None else {}),
        }
        measurement_rows.update({
            "per_op_timing": per_op_timing,
            "per_op_timing_decode": per_op_timing_decode,
            **({"per_op_timing_draft": per_op_timing_draft}
               if per_op_timing_draft else {}),
        })
        return result

    estimate = PendingEstimate(
        scope=p.measurement_scope, calibration_rows=p.calibration_rows,
        reads=p.enable_measured_lookup, walks=tuple(walks), finish=finish,
        keyed=keyed, encode=encode_plan, walked_on=p.card,
        selection_mode=p.selection_mode, kernel_mix=p.kernel_mix,
        walk_of_rows={"per_op_timing": prefill_walk,
                      "per_op_timing_decode": decode_walk,
                      **({"per_op_timing_draft": draft_walk}
                         if draft_walk is not None else {})},
        ceilings=_c, preparation=p,
        prefill_chunk_schedule=prefill_chunk_schedule,
        measurement_rows=measurement_rows)
    if fold_prefill_chunks and len(prefill_chunk_schedule.chunks) > 1:
        def finish_chunked() -> dict:
            return _fold_chunked_prefill_sol(
                finish(), p, estimate, prefill_chunk_schedule)
        estimate = replace(estimate, finish=finish_chunked)
    return estimate


@dataclass(frozen=True)
class PendingEstimate:
    """An estimate up to its overlays' structural walks (`_capacity_pending`):
    `finish() -> dict` runs the rest, no measurement read; `measures(result)`
    states its measured totals before measurement (`batch_sweep.
    point_measures`), which `finish_estimates` folds over one lookup."""
    scope: Any
    calibration_rows: Optional[list]
    #: Whether the estimate reads measurements (the mode reads them and
    #: there are some to read: a Kernel Bench scope or posted rows).
    reads: bool
    walks: Tuple[Any, ...]
    finish: Callable[[], dict]
    #: The walk each per-op row list of the finished estimate comes from
    #: (`per_op_timing` the prefill walk, `per_op_timing_decode` the decode
    #: walk, `per_op_timing_draft` the draft's): what reads those rows'
    #: keys at another point (`calibration_overlay.point_keys`).
    walk_of_rows: Mapping[str, Any] = field(default_factory=dict)
    #: The capacity ceilings the walk was bound by (`capacity_ceilings`).
    ceilings: Any = None
    #: Each row list's `PointRow`s at the walk's own point, once finished.
    keyed: Mapping[str, list] = field(default_factory=dict)
    #: The structural endpoint rows the measured fold substitutes from.  A
    # multi-chunk SOL fold publishes summed rows, but a chunk measurement is
    # still keyed off one endpoint graph row, not that launch total.
    measurement_rows: Mapping[str, list] = field(default_factory=dict)
    #: The encode phase's plan (its stacks' walks and `PointRow`s), or None.
    encode: Any = None
    #: The card this walk was TAKEN on — one card per walk class, the
    #: class's representative (`parallel_sweep.class_cards` picks the
    #: roomiest). A fact about the walk, not a leak: the walk's laws keep
    #: the card's symbols unsubstituted, and naming the card it was taken
    #: on is what makes the one road checkable: `priced_on` can only ask
    #: "is your class and your bound stage mine?" because the walk says
    #: whose class and whose stage it is.
    walked_on: Card = None  # type: ignore[assignment]
    selection_mode: str = "theoretical_only"
    kernel_mix: Optional[dict] = None
    #: The preparation a long prefill's chunk points are substituted from;
    #: absent for a short or unchunked estimate.
    preparation: Any = None
    #: The endpoint's one materialized chunk schedule; one-chunk endpoints
    #: keep the existing endpoint measure byte-for-byte.
    prefill_chunk_schedule: Optional[PrefillChunkSchedule] = None
    #: The measured formulas its points' steps are written as, by the
    #: step's shape (`batch_sweep.step_measure`): one walk's points share
    #: them.
    measured_formulas: Dict[Any, Any] = field(default_factory=dict)

    def priced_on(self, p_dst: Any, ceilings: Any = None, *,
                  rows_stage: Optional[int] = None) -> None:
        """The one road from this walk to a card's numbers: whether
        `p_dst`'s card may carry it.

        Two facts, asked in one place because a walk is a function of both
        and a caller that checked one published the other's answer:

        * the card's walk CLASS. This walk was taken on one card — its
          class's representative (`walked_on`) — and reads on another card
          of that class and no further
          (`Card.refuse_unless_one_class`, which states why and names the
          two times a crossing shipped);
        * the pipeline stage the destination card's HBM BINDS, by the one
          statement of that rule (`CapacityCeilings.refuse_unless_one_
          stage`). `ceilings` are the destination card's; `rows_stage` is
          the stage the rows being moved describe, which is this walk's own
          except where a caller moved an anchor to another context first. A
          caller with no `ceilings` in hand is asking the class question
          alone — which is the rows road, where the destination's stage is
          not yet bound.

        No answer but yes or a refusal. There is no sound recovery from a
        mismatch: both sides are plausible numbers, and the one thing that
        must not happen is this walk's rows published as the other card's.
        """
        self.walked_on.refuse_unless_one_class(p_dst.card)
        if ceilings is not None:
            ceilings.refuse_unless_one_stage(
                self.ceilings.pp_stage if rows_stage is None else rows_stage,
                p_dst.gpu_sku)

    def keyed_at(self, rows: Mapping[str, list], *, gpu_sku: str
                 ) -> Tuple[Dict[str, list], Optional[Dict[str, list]]]:
        """Each row list of a point (`_point_rows`, on `gpu_sku`) keyed
        there by the one derivation (`calibration_overlay.point_keys`, off
        the walk the list came from), its rows stating what their keys are
        read off (`key_columns`) and their identity there
        (`identity_columns`): the rows, and per list their `PointRow`s —
        what the point's measure is built from (`batch_sweep.
        point_measures`). Only a request that looks measurements up
        (`calibration_overlay.looks_up`) asks: for another, the rows as
        they are and no keys."""
        out_rows: Dict[str, list] = dict(rows)
        if not looks_up(self.selection_mode):
            return out_rows, None
        points: Dict[str, list] = {}
        for name, walk in self.walk_of_rows.items():
            listed = key_columns(rows.get(name) or [], walk.analysis)
            points[name] = point_keys(
                walk.analysis, listed, walk.workload,
                replace(walk.deployment, gpu_sku=gpu_sku),
                model_id=walk.model_id, spec_hash=walk.spec_hash)
            out_rows[name] = [{**r, **identity_columns(pt)}
                              for r, pt in zip(listed, points[name], strict=True)]
        return out_rows, points

    def measures(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        """The finished estimate's measured totals before measurement — one
        per phase, one per encoder that ran items — the same plain data a
        sweep point's are (`batch_sweep.point_measures`, `step_measure`)."""
        source_rows = self.measurement_rows or result
        rows = {name: source_rows.get(name) or [] for name in self.keyed}
        if (self.prefill_chunk_schedule is not None
                and self.preparation is not None):
            out = _point_measures_with_chunks(
                rows, self.keyed, self.preparation, self,
                result.get("phases") or {},
                schedule=self.prefill_chunk_schedule)
        else:
            out = point_measures(
                self, rows, self.keyed, result.get("phases") or {},
                gpu_sku=self.walked_on.sku)
        if self.encode is not None and result.get("encode") is not None:
            for index, walk in self.encode.timed():
                entry = result["encode"]["stacks"][index]
                rows = entry["per_op_timing"]
                keyed = dict(zip(map(id, rows), self.encode.points[index], strict=True))
                out[f"encode:{index}"] = {**step_measure(
                    StepSum.of(rows), keyed, dict.fromkeys(map(id, rows), "per_op_timing"),
                    gpu_sku=self.walked_on.sku, framework=walk.deployment.framework,
                    binding=walk.deployment.framework_binding,
                    written=self.measured_formulas), "encode": index}
        return out


def finish_estimates(pending: List[PendingEstimate]) -> List[dict]:
    """Finish estimates that read one measurement scope — one estimate, a
    disaggregated pair — with ONE lookup: each finished without a
    measurement, then the union of their keys looked up once and each
    estimate's measured totals folded over it (`batch_sweep.measure_points`)
    — an estimate is a one-point sweep."""
    results = [without_step(pe.finish(), pe.walk_of_rows) for pe in pending]
    reading = [(pe, r) for pe, r in zip(pending, results, strict=True)
               if pe.reads and not r.get("infeasibility_reason")]
    if reading:
        first = reading[0][0]
        if any((pe.scope, pe.calibration_rows, pe.selection_mode, pe.kernel_mix)
               != (first.scope, first.calibration_rows, first.selection_mode,
                   first.kernel_mix) for pe, _r in reading):
            raise ValueError("estimates finished together read one measurement scope")
        measure_points([{"sol": r, "measures": pe.measures(r)} for pe, r in reading],
                       selection_mode=first.selection_mode,
                       measurement_scope=first.scope,
                       calibration_rows=first.calibration_rows,
                       kernel_mix=first.kernel_mix)
    return results


@stage_context("phases")
def _capacity_finish(p: _CapacityPreparation, **kwargs: Any) -> dict:
    """One estimate's second half: `_capacity_pending`, its one lookup,
    the rest (`finish_estimates`)."""
    return finish_estimates([_capacity_pending(p, **kwargs)])[0]
