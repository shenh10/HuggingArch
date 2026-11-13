"""One evaluator result shared by every view of an inference step.

A single forward step at a single deployment used to be evaluated four times.
`estimate_phase` walked the graph for the FLOPs table and walked it again,
sharded, for that table's per-rank column; `pipeline.analyze` walked it for the
per-op rows and `compute_per_rank_analysis` walked it again, sharded, for their
bytes. Measured on `meta-llama/Llama-3.2-3B` at 8×H200, one `/inference/estimate`
entered `project_model` nine times across seven argument sets — and two of those
sets were byte-identical, the same question asked twice.

The cost of that was not the arithmetic. It was that the answers could differ.
Every walk re-derived its own binding, its own decode rebind, its own
`logit_positions`, and each place that did so was a place one of them could be
wrong alone: prefill billed the LM head at all 8192 positions in the table while
the rows beside it billed the sampled one, and the table and the rows disagreed
by 6.5e12 FLOPs with nothing in the system positioned to notice. The fix each
time was to teach the new walk what the old one knew, which left the same shape
standing with one more thing to keep in step.

So this is the shape instead: a `StepEvaluation` IS the (workload, deployment,
phase) triple and caches one `Analysis`. Consumers fold its UID-keyed layers:
the FLOPs table groups them by module and per-op timing emits leaf rows.
Agreement between those views is no longer a property of independent walks,
because there is one evaluation underneath them.

`analysis` is the sole evaluator walk. It binds the whole model and carries
distributed layouts beside the existing shape, cost, storage and cache layers.
Physical compute, storage, cache and communication values are derived from that
`Analysis`; rank-local geometry is never obtained by replaying the evaluator.
"""

from __future__ import annotations
from backend.arch.runtime_profile import RuntimeProfile

import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, List, Mapping, Optional, Tuple

from backend.arch.eval.analysis import Point
from backend.arch.layout.physical_cost import physical_flops

from .activation_peak import (FRAMEWORK_OVERHEAD_BYTES, ActivationLiveness,
                              activation_peak_at)
from .parallel_config import ParallelConfig
from .serving import build_serving_layers


@dataclass
class StepEvaluation:
    """The graph, the workload and the deployment, evaluated once.

    Construct through `evaluate_step` rather than directly: the decode variant
    rebind and the sampled-head width are part of what the triple MEANS, and a
    consumer that set them itself would be back to a walk with its own opinion.
    """

    # The spec the walks actually see — decode-rebound where that applies, so
    # `mla_absorbed` says whether the swap fired rather than each consumer
    # re-deciding. One clone, shared: `iter_tree` memoizes on the ResolvedSpec,
    # so two clones of the same rebind are two node identity spaces, and a
    # consumer holding nodes from one cannot be joined against the other.
    spec: Any
    # Always concrete: even an undescribed deployment evaluates as one rank so
    # the global Analysis uses the same propagation path as every other step.
    parallel: Any
    # Presentation fact kept separate from the evaluator input.  A caller that
    # supplied no deployment gets global rows only, even though the analysis
    # itself ran through the concrete one-rank layout.
    deployment_described: bool
    phase: str
    batch: int
    # The workload, as the request states it: the phase resolves the step's
    # query block, attended context and causal fraction from these
    # (`transforms.Phase.resolve`), in the walk and nowhere else.
    input_tokens: int
    output_tokens: int
    s_prefix_cached: int
    s_verify: int
    #: The batch's routing skew, max/mean expert load (`Workload.
    #: ep_load_factor`): a workload input the walk reads as the symbol
    #: `rho`, so steps of one deployment at any skew share a cell.
    ep_load_factor: Any
    dp_attention: int
    # Where the sampler reads. Not a walk-local choice: a walk that sampled
    # while its neighbour did not reported the head shrinking by `s_query` on
    # top of the world that actually splits it.
    logit_positions: int
    mla_absorbed: bool
    # Built once, off the SPLICED spec (the same sampled-head splice the walk
    # applies), so it matches the program the walk evaluates node-for-node.
    # `iter_tree(step_eval.spec)` would give the UNSPLICED tree with a shifted
    # uid space — use this field, not that.
    pre_tree: Any
    #: The deployment's precision profile (`precision.PrecisionProfile`);
    #: None is as shipped.
    precision: Any = None
    dcp_selected_load_factor: Any = 1
    dcp_failed_owners: Any = 0
    #: An encode step's encoder stack and the media each request carries
    #: (`Workload.media`); a language-model step names neither.
    stack: Optional[str] = None
    media: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    #: The walk this step shares with every other point of it (`Cell`).
    cell: Optional["Cell"] = None

    _analysis: Optional[Any] = field(default=None, init=False, repr=False)
    _reading: Optional[Point] = field(default=None, init=False, repr=False)

    def as_deployment(self):
        """This step's deployment, at its point: the workload as numbers.
        Its walk is `analysis`'s — one walk of the workload's symbols read
        here — and a caller that walks it itself (a checkpoint question) asks
        that numeric walk directly."""
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import ENCODE, Deployment

        runtime_profile = self.cell.runtime_profile if self.cell is not None else RuntimeProfile.DECLARED
        if self.phase == ENCODE:
            # The encoder's own step: the batch and its media, nothing of
            # the language model's (`Deployment.binding`).
            deployment = Deployment(
                Program.of(self.spec), config=self.parallel, runtime_profile=runtime_profile,
            ).at(batch=self.batch, media=self.media).during(
                ENCODE, stack=self.stack)
            if self.precision is not None:
                deployment = deployment.with_precision(self.precision)
            return deployment
        deployment = Deployment(
            Program.of(self.spec),
            config=self.parallel, runtime_profile=runtime_profile,
        ).at(
            batch=self.batch,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            s_prefix_cached=self.s_prefix_cached,
            s_verify=self.s_verify,
            ep_load_factor=self.ep_load_factor,
            dcp_selected_load_factor=self.dcp_selected_load_factor,
            dcp_failed_owners=self.dcp_failed_owners,
            logit_positions=self.logit_positions,
        )
        if self.phase != "prefill":
            deployment = deployment.during(self.phase)
        if self.precision is not None:
            deployment = deployment.with_precision(self.precision)
        return deployment.with_dims(dp_attention=self.dp_attention)

    @property
    def point(self) -> Mapping[str, int]:
        """This step's workload as the point its walk is read at
        (`Workload.symbolic`)."""
        return self.as_deployment().workload.stated_point()

    @property
    def walk(self):
        """The walk of this step's workload as symbols (`Workload.symbolic`):
        every point's, its cell's (`Cell.walk`)."""
        return self.cell.walk(self)

    @property
    def analysis(self):
        """The one evaluation, with global extents and per-edge layouts:
        the walk read at this step's point (`Analysis.at`). The numbers are
        a walk's at the point; the walk is every point's."""
        if self._analysis is None:
            self._analysis = self.walk.at(self.reading)
        return self._analysis

    @property
    def reading(self) -> Point:
        """This step's point as its readings share it (`analysis.Point`):
        the walk, read for the step, is the walk the serving layers built
        off it hold, and is read once."""
        if self._reading is None:
            self._reading = Point(self.point)
        return self._reading

    def physical_flops(self) -> Any:
        """Each node's per-rank FLOPs at this step's point: the cell's
        (`physical_cost.physical_flops` of its walk, written once) read
        here."""
        self.walk  # noqa: B018 — the cell's walk, built before it is read
        return self.reading.read(self.cell.physical_flops())

    @property
    def has_deployment(self) -> bool:
        """Whether a deployment was described at all.

        Without one there is no per-rank tree to ask for, and consumers say so
        by declining the column rather than by filling it with the global
        number under a per-rank name.
        """
        return self.deployment_described


def sampled_logit_positions(phase: str, s_verify: int = 1) -> int:
    """How many positions of the query block the LM head actually runs at.

    Serving slices the hidden states down to the positions it is going to
    sample from before the head sees them — vLLM does it in
    `logits_processor`, ahead of the vocab projection. In prefill that is one
    position per sequence no matter how long the prompt was; in decode it is
    the whole verified window, which is `s_verify` (1 for autoregressive
    decode, K under a K-token speculative step, and the same number the query
    block itself is `s_verify` wide for).

    Asked once per step, here, and every view of that step inherits the
    answer. It used to live in `phases` and be asked by each of four walks,
    and when they disagreed the FLOPs table and the per-op rows described
    different work — prefill's head at 8192 positions in one and 1 in the
    other, a 6.5e12 FLOPs gap on a 3B model at 8k, roughly a ninth of the
    step — and the per-rank column read across that gap reported the head
    sharded by a factor of `s_query` on top of the world that actually
    splits it.
    """
    from backend.arch.eval.dim import maximum

    return 1 if phase == "prefill" else maximum(
        1, 1 if s_verify is None else s_verify)


#: How many readings of one cell's walk it keeps, per kind of reader
#: (`Cell.serving`, `Cell.liveness`): the least recently read one goes
#: first, and is built again if asked again.
READINGS_KEPT = 32

#: How many cells one registry keeps (`StepCells.cell`), the same way.
CELLS_KEPT = 8


def _kept(table: "OrderedDict[Any, Any]", key: Any, build: Any) -> Any:
    """`table[key]`, built by `build()` when absent; the `READINGS_KEPT`
    most recently read are kept."""
    if key in table:
        table.move_to_end(key)
        return table[key]
    table[key] = value = build()
    while len(table) > READINGS_KEPT:
        table.popitem(last=False)
    return value


@dataclass
class Cell:
    """The walk every point of one step shares: one (model, deployment,
    phase, precision) — for an encoder, one (stack, media) — whose workload
    changes only numbers, walked once over the workload's symbols — and
    everything built off it: its per-rank projections (`physical`), the
    serving layers each framework binding, pipeline stage and timed stacks
    read it through (`serving`), its activation liveness (`liveness`), and
    the one book every law any of them writes is held and compiled in
    (`laws`). Each point reads them at its own numbers (`Analysis.at`,
    `serving.ServingLayers.at`).

    What a cell holds is built the first time a reader asks, under the
    cell's lock: two readers of one cell on two threads build it once and
    read the same objects."""

    spec: Any
    parallel: Any
    phase: str
    precision: Any = None
    stack: Optional[str] = None
    media: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    #: Whether the cell's steps speculate (verify width above 1). A step
    #: that does not walks its verify width as the number 1
    #: (`Workload.symbolic`), so the two never share a walk.
    speculative: bool = False
    # Sparse DCP verification reads sum causal rows at this fixed width.
    verify_width: Optional[int] = None
    runtime_profile: RuntimeProfile = RuntimeProfile.DECLARED
    _walk: Optional[Any] = field(default=None, init=False, repr=False)
    #: The spec rebound to the forms the phase runs
    #: (`rebind_decode_variants`), as a one-tuple once asked (its content
    #: may be None: no variant to swap in).
    phase_spec: Optional[Tuple[Any]] = field(default=None, init=False,
                                             repr=False)
    #: The tree its steps' walks bind (`pre_tree`), once built.
    _pre_tree: Optional[Any] = field(default=None, init=False, repr=False)
    #: Every law the readings below write, compiled once for every point.
    laws: Any = field(default=None, init=False, repr=False)
    _physical: Optional[Any] = field(default=None, init=False, repr=False)
    _physical_flops: Optional[Any] = field(default=None, init=False,
                                           repr=False)
    _serving: "OrderedDict[Any, Any]" = field(default_factory=OrderedDict,
                                              init=False, repr=False)
    _liveness: "OrderedDict[Any, Any]" = field(default_factory=OrderedDict,
                                               init=False, repr=False)
    lock: Any = field(default_factory=threading.RLock, init=False,
                      repr=False, compare=False)

    def __post_init__(self) -> None:
        from .row_roofline import LawBook
        self.laws = LawBook()

    def pre_tree(self, spec: Any) -> Any:
        """The pre-binding tree of `spec` — the cell's: the resolved spec,
        or its phase rebind — with the sampled-head splice the walk
        applies (`transforms._walk` splices whenever `logit_positions` is
        set, which it always is here), built once: every step of the cell
        binds the same tree, node for node. Building it from the raw spec
        left it one `sample` view short of `pre_program`, so
        `framework_binding=None` — a no-op — looked like it had changed the
        tree; None from the splice means no output head."""
        with self.lock:
            if self._pre_tree is None:
                from backend.arch.spec_tree import iter_tree
                from backend.arch.eval.transforms import splice_sampled_head

                self._pre_tree = iter_tree(splice_sampled_head(spec) or spec)
            return self._pre_tree

    def phase_rebind(self, spec: Any) -> Optional[Any]:
        """`spec` rebound to the forms the cell's phase runs
        (`rebind_decode_variants`), once: None when no component has
        another one."""
        with self.lock:
            if self.phase_spec is None:
                from backend.arch.eval.transforms import rebind_decode_variants
                self.phase_spec = (rebind_decode_variants(spec, self.phase),)
            return self.phase_spec[0]

    def walk(self, step: "StepEvaluation") -> Any:
        """The walk, built off the first step that asks: its deployment with
        the workload's numbers turned into symbols — the same walk whichever
        point asks first."""
        with self.lock:
            if self._walk is None:
                from backend.arch.eval.transforms import ENCODE, Workload

                if (self.phase != ENCODE and self.speculative
                        == Workload.walked_as_number("s_verify",
                                                     step.s_verify)):
                    raise ValueError(
                        f"a step at verify width {step.s_verify} cannot walk "
                        f"the cell of steps that "
                        f"{'do' if self.speculative else 'do not'} speculate")
                deployment = step.as_deployment()
                workload, _point = deployment.workload.symbolic()
                if self.verify_width is not None:
                    workload = replace(workload, s_verify=self.verify_width)
                if self.phase != ENCODE:
                    workload = replace(
                        workload, logit_positions=sampled_logit_positions(
                            self.phase, workload.s_verify))
                compiled = replace(deployment, workload=workload)
                if self.verify_width is not None:
                    # This is already a symbolic cell, specialized only in
                    # verify width. Reading it at that width here would erase
                    # its admission guard before other points can check it.
                    self._walk = compiled._walk(compiled.binding())
                else:
                    self._walk = compiled.analyze()
            return self._walk

    def physical(self) -> Any:
        """The walk's per-rank projections (`physical_cost.
        PhysicalProjection`): its shapes, kernel and access shapes, cache
        and traffic geometry and cost plan, each derived once for every
        reader of the cell."""
        with self.lock:
            if self._physical is None:
                from backend.arch.layout.physical_cost import PhysicalProjection
                self._physical = PhysicalProjection(self.built_walk())
            return self._physical

    def physical_flops(self) -> Any:
        """Each node's per-rank FLOPs, as laws (`physical_cost.
        physical_flops` of the walk, off its kernel shapes)."""
        with self.lock:
            if self._physical_flops is None:
                walk = self.built_walk()
                with walk.assumed():
                    self._physical_flops = physical_flops(
                        walk, None if walk.rule.is_trivial
                        else self.physical().kernel_shapes)
            return self._physical_flops

    def serving(self, framework_binding: Any, stage: Optional[int],
                timed: Any) -> Any:
        """The serving layers the walk is read through under
        `framework_binding`, for pipeline stage `stage` (None: the whole
        model), timing the stacks `timed` (`serving.ServingLayers`), built
        once, by the one builder of them (`serving.build_serving_layers`):
        what is kept under a key is what that key builds."""
        with self.lock:
            return _kept(self._serving, (framework_binding, stage, timed),
                         lambda: build_serving_layers(
                             self, framework_binding=framework_binding,
                             stage=stage, timed=timed))

    def liveness(self, framework_binding: Any, timed: Any) -> Any:
        """The walk's activation liveness under `framework_binding`, timing
        the stacks `timed` (`activation_peak.ActivationLiveness`): folded
        once off the whole model's serving layers under the same key
        (`serving`), its laws in the cell's book."""
        with self.lock:
            return _kept(self._liveness, (framework_binding, timed),
                         lambda: ActivationLiveness(
                             self.serving(framework_binding, None, timed),
                             self.laws))

    def built_walk(self) -> Any:
        """The walk, which a step built (`walk`)."""
        if self._walk is None:
            raise ValueError("the cell is read before any step walked it")
        return self._walk


class StepCells:
    """The cells one caller's steps share: an estimate's — its activation
    step and its prefill step are one walk — or a sweep's, across all its
    points."""

    def __init__(self, runtime_profile: RuntimeProfile = RuntimeProfile.DECLARED) -> None:
        self.runtime_profile = RuntimeProfile(runtime_profile)
        self._cells: List[Cell] = []
        self._lock = threading.Lock()

    def cell(self, spec: Any, parallel: Any, *, phase: str,
             precision: Any = None, stack: Optional[str] = None,
             media: Optional[Mapping[str, Mapping[str, int]]] = None,
             speculative: bool = False, verify_width: Optional[int] = None) -> Cell:
        """The cell of these, one registry-wide: the `CELLS_KEPT` most
        recently asked are kept."""
        media = dict(media or {})
        with self._lock:
            for i, cell in enumerate(self._cells):
                if (cell.spec is spec and cell.parallel == parallel
                        and cell.phase == phase and cell.precision == precision
                        and cell.stack == stack and cell.media == media
                        and cell.speculative == speculative
                        and cell.verify_width == verify_width):
                    self._cells.append(self._cells.pop(i))
                    return cell
            cell = Cell(spec=spec, parallel=parallel, phase=phase,
                        precision=precision, stack=stack, media=media,
                        speculative=speculative, verify_width=verify_width,
                        runtime_profile=self.runtime_profile)
            self._cells.append(cell)
            del self._cells[:-CELLS_KEPT]
            return cell


def evaluate_step(resolved_spec: Any, parallel: Any, *, phase: str, batch: int,
                  input_tokens: int, output_tokens: int,
                  s_prefix_cached: int = 0, s_verify: int = 1,
                  ep_load_factor: Any = 1,
                  dcp_selected_load_factor: Any = 1,
                  dcp_failed_owners: Any = 0,
                  precision: Any = None,
                  cells: Optional[StepCells] = None,
                  ) -> StepEvaluation:
    """Bind one (workload, deployment, phase) to the graph it describes.

    The workload is the request's: the prompt and the output, both stated
    (a decode step's cache is the prompt and half the output, so an output
    nobody stated would price the cache at a length nobody asked for); the
    cached prefix, 0 for none; the verify width, 1 for no speculative
    verify. The step's query block, attended context and causal fraction
    are the phase's reading of them (`transforms.Phase.resolve`), made in
    the walk — a caller does not resolve them first and hand them in.

    `cells` are the walks the caller's steps share (`StepCells`); a step
    with none walks its own.
    """
    spec = resolved_spec
    deployment_described = parallel is not None
    if parallel is None:

        parallel = ParallelConfig()
    refusal = prefix_refusal(parallel, phase=phase, input_tokens=input_tokens,
                             s_prefix_cached=s_prefix_cached)
    if refusal is not None:
        raise ValueError(refusal)
    from backend.arch.eval.transforms import Workload
    s_verify = int(s_verify or 1)
    fixed_verify = None
    cells = cells if cells is not None else StepCells()
    if cells.runtime_profile == RuntimeProfile.DECLARED and parallel.dcp > 1 and phase == "decode":
        from backend.arch.eval.program import Program
        if Program.of(resolved_spec).sparse_dcp_components:
            fixed_verify = s_verify
    cell = cells.cell(
        resolved_spec, parallel, phase=phase, precision=precision,
        speculative=not Workload.walked_as_number("s_verify", s_verify),
        verify_width=fixed_verify)
    # vLLM / SGLang / TRT-LLM all decode through the absorbed MLA form
    # (kv_b_proj folded into the Q- and O-side per-head matmuls, cache held
    # in latent d_kv_lora+d_qk_rope), and prefill a sparse (top-k) MLA the
    # same way, so that is the graph the roofline should price
    # (`transforms.phase_form` says which phase runs which form). Clone,
    # never mutate — the engine caches on the spec. Once per cell: its
    # points share the clone, and with it the walk.
    cloned = cell.phase_rebind(resolved_spec)
    mla_absorbed = cloned is not None
    if cloned is not None:
        spec = cloned

    return StepEvaluation(
        spec=spec,
        parallel=parallel,
        deployment_described=deployment_described,
        phase=phase,
        batch=int(batch),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        s_prefix_cached=int(s_prefix_cached),
        s_verify=int(s_verify or 1),
        ep_load_factor=ep_load_factor,
        dcp_selected_load_factor=dcp_selected_load_factor,
        dcp_failed_owners=dcp_failed_owners,
        dp_attention=max(1, int(getattr(parallel, "dp_attention", 1) or 1)),
        logit_positions=sampled_logit_positions(phase, s_verify),
        mla_absorbed=mla_absorbed,
        pre_tree=cell.pre_tree(spec),
        precision=precision,
        cell=cell,
    )


def prefix_refusal(parallel: Any, *, phase: str, input_tokens: int,
                   s_prefix_cached: int) -> Optional[str]:
    """Why a step over a cached prefix is not modelled on `parallel`, or
    None: a prefill that reads its prefix (`_reads_a_cached_prefix`) under
    a position-sharded cache (dcp>1) — the prefix lives 1/dcp per rank and
    the kernel needs it whole. The step there (`evaluate_step`) and a
    sweep's context there (`capacity._sweep_cell`) refuse it alike."""
    if phase != "prefill" or not _reads_a_cached_prefix(input_tokens,
                                                        s_prefix_cached):
        return None
    if parallel is None or parallel.degree_of_world("dcp", phase="prefill") <= 1:
        return None
    return ("prefill with a cached prefix under a position-sharded "
            "cache (dcp>1) is not modelled: the prefix lives 1/dcp per "
            "rank and the kernel needs it whole. Set dcp=1 or pass no "
            "prefix.")


def _reads_a_cached_prefix(input_tokens: int, s_prefix_cached: int) -> bool:
    """Whether a prefill step's queries are fewer than the keys it attends —
    a cached prefix the step reads rather than computes (`Phase.resolve`)."""
    from backend.arch.eval.dim import value_of
    from backend.arch.eval.transforms import PREFILL, Phase, Workload

    resolved = Phase(kind=PREFILL).resolve(Workload(
        input_tokens=input_tokens, s_prefix_cached=s_prefix_cached))
    return value_of(resolved["s_query"]) < value_of(resolved["s_kv"])


def evaluate_encode(resolved_spec: Any, parallel: Any, *, stack: str,
                    batch: int, media: Mapping[str, Mapping[str, int]],
                    precision: Any = None) -> StepEvaluation:
    """One encode step: encoder `stack` over `batch` requests, each carrying
    `media` (`Workload.media`). The encoder's step has no query block, cache
    or sampled head; its token fields hold nothing (zero), and nothing
    of the step reads them — `as_deployment` binds the media alone."""
    from backend.arch.spec_tree import iter_tree

    if parallel is None:

        parallel = ParallelConfig()
    return StepEvaluation(
        spec=resolved_spec,
        parallel=parallel,
        deployment_described=True,
        phase="encode",
        batch=int(batch),
        input_tokens=0,
        output_tokens=0,
        s_prefix_cached=0,
        s_verify=1,
        ep_load_factor=1,
        dp_attention=1,
        logit_positions=0,
        mla_absorbed=False,
        pre_tree=iter_tree(resolved_spec),
        precision=precision,
        stack=stack,
        media=dict(media),
        cell=StepCells().cell(resolved_spec, parallel, phase="encode",
                              precision=precision, stack=stack, media=media),
    )


def estimate_peak_activation(
    resolved,
    *,
    batch: int,
    seq_len: int,
    parallel=None,
    framework_binding=None,
    overhead_bytes: int = FRAMEWORK_OVERHEAD_BYTES,
    step=None,
) -> dict:
    """Compute per-rank peak activation bytes for the model's block.

    Strategy:
      - The prefill step's walk, folded once per cell
        (`ActivationLiveness`): its derived physical shapes plus the
        program's edges carry everything liveness needs, so every tensor
        the liveness sums is already one rank's and no shard-on-read
        evaluator is replayed. `step` is that prefill step
        (`step_eval.evaluate_step`), whose cell the caller's other steps
        share; without one the step at `batch` and `seq_len` is evaluated
        here.
      - The same comm-then-fusion stretch `pipeline.analyze` uses
        (`serving.rewrite_serving_graph`). No second matcher. No binding means
        the unfused graph, including collectives if the walk injected
        them.
      - Run liveness on each block template at the step's point; take the
        max across blocks (heterogeneous models may have different peaks;
        we want the worst case for capacity planning).
      - Each produced tensor is priced at the producing leaf's edge width
        under the walk's precision profile. A block's external residual
        has no producer in the template and takes the model's own width.

    No divisor. The `peak / tp` this replaced was the last uniform-shard
    scalar in the inference path, and it was wrong in both directions: it
    divided the residual and every layernorm output, which plain tensor
    parallelism replicates on every rank, and it took the MIN of
    `tp_attention` / `tp_dense_ffn` so a model sharding the two at different
    degrees had one of them mis-stated. Under-reporting an activation peak
    is the expensive direction — it is what the OOM headroom is computed
    from.
    """
    if step is None:
        step = evaluate_step(
            resolved,
            parallel or ParallelConfig(),
            phase="prefill",
            batch=batch,
            input_tokens=seq_len, output_tokens=0,
        )
    return activation_peak_at(step, framework_binding=framework_binding,
                              overhead_bytes=overhead_bytes)
