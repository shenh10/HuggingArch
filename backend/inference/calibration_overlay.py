"""Calibration overlay — typed stage that enriches a structural
AnalysisResult with measured per-op timings from a registry of measured
rows (looked up in Kernel Bench for the keys the overlay names —
`OverlayWalk.keys`, `op_calibration.registry_for` — and any rows the request
posts).

Before this module existed, calibration logic was woven through
phases.estimate_phases / pipeline.analyze / compute_per_op_timing via
a Dict[str, Any] called `calibration_context` that threaded
batch_size / seq_len / dtypes / framework / parallel / gpu / kernel_mix
across three layers. Each round of Codex review caught a different
field that wasn't being threaded correctly. The root cause was the
sidecar shape: calibration was an `if selection_mode != "theoretical":`
side-effect tucked inside the phase pipeline rather than a first-class
stage with explicit typed inputs.

This module is the rewrite. Two input dataclasses (`WorkloadShape` /
`DeploymentContext`) make the contract explicit;
`CalibrationOverlay.apply(analysis, workload, deployment, ...)` is the only
entry point: each row's identity and theory classification, and its
`PointRow`. What is measured is the measured fold's (`batch_sweep.
measured_selector`, `inference.measure_matrix.MeasureMatrix.fold`).

Stage placement in the pipeline:

    spec → engine → fusion → CalibrationOverlay → render
                              ↑
                         this module
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.arch.loader import ResolvedSpec


from backend.arch.eval.dim import as_extent, as_number
from backend.arch.sharding import ENCODE
from backend.arch.eval.analysis import Point
from .benchmark_identity import (
    RowIdentity,
    routed_launch,
)
from .framework_bindings import (DEFAULT_FRAMEWORK, THEORY,
                                 DispatchDecision, ResolvedLaunch,
                                 ServingDeployment, get_binding)
# Top-level is safe here: module_rows pulls only arch.eval layers, not
# pipeline/phases (which is why the pipeline import below stays lazy).
from .module_rows import physical_module_rows
from .row_roofline import RowRoofline, RowTimes, StepSum, StepTerm, row_law
from .op_signatures import get_signature, op_bindings
from .op_calibration import (
    BenchmarkKey,
    PhysicalKey,
)
import sympy as sp
from backend.measure.auto_sweep import (DECLINED_REASON, MISSING_PROJECTION,
                                         MISSING_SIGNATURE, ZERO_LAUNCH)


# ---------------------------------------------------------------------------
# Typed inputs — the new contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkloadShape:
    """The "what's being asked" — request-time workload dimensions.

    It used to carry the five dtype slots of the measurement key too, and
    that was the second statement of them: every one was read off the walk
    a line before the dataclass was built, and the kernel_bench plan — which
    has no `WorkloadShape` — stated its own. A dtype slot now has exactly one
    producer (`RowIdentity.dtype_env`, which holds the analysis and so can
    answer the step's dominant dtypes itself), and what is left here is the
    workload: how many requests, and how long. It carried the model's dim
    table too, for a row with no dims of its own to be keyed by; every row
    the walk publishes carries its node's (`RowIdentity._node_dims`).
    """
    batch_size: int
    seq_len: int

    @classmethod
    def from_request(cls, *, batch_size: int, seq_len: int) -> "WorkloadShape":
        return cls(batch_size=int(batch_size), seq_len=int(seq_len))

    @classmethod
    def of_walk(cls, analysis: Any, *, seq_len: int) -> "WorkloadShape":
        """The workload of one walk (`pipeline.analyze`) as a lookup states
        it. The batch is 1: a lookup key is canonical at B=1, whatever batch
        the walk ran at. A walk with no resident weight dtype is refused —
        the key's dtype half (`RowIdentity.dtype_env`) has no step frame to
        stand in."""
        if not analysis.analysis.dominant_weight_dtype():
            raise ValueError(
                "calibration overlay needs a weight dtype from the analysis"
            )
        return cls.from_request(batch_size=1, seq_len=seq_len)


@dataclass(frozen=True)
class DeploymentContext:
    """The "how it's served" — hardware + parallelism + framework."""
    framework: str
    framework_binding: Optional[Any]   # FrameworkBinding or None
    parallel_signature: str            # canonical "tp=4,ep=2,pp=1" string
    gpu_sku: str
    phase: str                          # "prefill" / "decode"
    #: The paged-KV page this deployment allocates the cache in. Part of the
    #: key of every kernel that reads a paged cache, so the estimate asks
    #: for the page it is deploying with rather than whichever of the three
    #: the plan happened to write last. It is an axis of those kernels'
    #: cells (`op_signatures.PAGE_AXIS`), handed to the projection with the
    #: row's dims (`RowIdentity`). `None` when neither the request nor the
    #: framework declared one, and then those cells have no key at all — a
    #: paged kernel has no cell without its page, and the alternative is a
    #: guessed 16, which would key a lookup on a paging layout nobody chose.
    kv_block_size: Optional[int] = None
    #: The degree of the vocabulary/attention tensor-parallel world. A
    #: dispatch audited only on the unsharded embedding path reads it; this
    #: is a request/deployment fact, not a graph-match condition.
    tensor_parallel: int = 1


@dataclass(frozen=True)
class CalibrationResult:
    """Output of running the overlay over an AnalysisResult: the per-leaf
    rows with their identity and theory classification (`per_op_timing`,
    `AnalysisResult.per_op_timing`'s shape), and each row's `PointRow` at the
    walk's point — what the estimate's measured fold keys them by."""
    per_op_timing: List[Dict[str, Any]]
    points: Tuple["PointRow", ...] = ()


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


class CalibrationOverlay:
    """The overlay stage: each per-op row's measurement identity at its
    point (`point_keys`, stated by `identity_columns`) and its theory
    classification — fused or naive, and for a fused row the kernel it
    stands for and what it absorbed (`source`, `source_badge`).

    It reads no measurement. A row's measured time, coverage state and
    measurement badge come from the one measured fold over the estimate's
    one lookup (`batch_sweep.measure_points`), for an estimate exactly as
    for every sweep point."""

    def apply(
        self,
        analysis: Any,                  # AnalysisResult (forward ref to avoid cycle)
        workload: WorkloadShape,
        deployment: DeploymentContext,
        *,
        model_id: str,
        spec_hash: str,
    ) -> CalibrationResult:
        """`analysis.per_op_timing` with each row's identity and theory
        classification, and each row's `PointRow` at the walk's point."""
        matches = tuple(analysis.fusion_matches)
        fused_agg_ids = {
            match["fused_aggregation_id"] for match in matches
        }
        # Map fused anchor's aggregation_id → fused_op name so the UI
        # can label rows with the abstract fused op they got folded into
        # (e.g. "flash_attention" instead of a generic "fused" pill).
        # Note `fused_op` is the registry name (framework-agnostic); the
        # concrete `kernel_impl` (e.g. flash_attention_v3) is added
        # separately from the binding. `MatchInfo.fused_aggregation_id`
        # is the synthesized leaf's aggregation_id (`template|spec_id`)
        # recorded by the rewrite; that is the same key
        # the row carries when source = theoretical_fused.
        #
        # ALSO map anchor aggregation_id → list of all matched sibling
        # LOCAL names so the breakdown UI can spell out "flash_attention
        # ← qk_score + softmax + attn_apply". Without this, a fused row
        # reads as "one op got renamed", hiding the many-to-one nature of the
        # fusion (and that the sibling rows the user expected to see
        # were actually absorbed, not missing). Local name = the last
        # path segment of the absorbed aggregation_id (the per-instance
        # leaf name).
        fused_op_by_sid: Dict[str, str] = {}
        absorbed_by_sid: Dict[str, List[Dict[str, Any]]] = {}
        pre_program = analysis.pre_program
        pre_shapes = analysis.pre_shapes
        # The absorbed parts' output laws in the workload: the same pre-fusion
        # shape layer `shape_out` reads, before it was read at this point, so
        # the badge reads at another point too (`batch_sweep`). Every part
        # has one, a view the timing exit drew no row for included.
        laws = analysis.laws
        pre_law_shapes = laws.pre_shapes if laws is not None else None
        for match in matches:
            fused_sid = match["fused_aggregation_id"]
            fused_op_by_sid[fused_sid] = match["fused_op"]
            items: List[Dict[str, Any]] = []
            for uid in match["matched_uids"]:
                node = pre_program.by_uid(uid) if pre_program is not None else None
                shape = pre_shapes.values.get(uid) if pre_shapes is not None else None
                items.append({
                    "name": node.name if node is not None else "",
                    "op": node.op if node is not None else "",
                    # `shape_out` is the post-op tensor shape; the hover
                    # popup on the inference page renders the chain of
                    # shape evolutions (input → intermediate → output)
                    # so the reader can see what the kernel internally
                    # avoids materializing.
                    "shape_out": (
                        [as_extent(axis, rounding="floor")
                         for axis in shape.out]
                        if shape is not None and shape.out is not None
                        else None
                    ),
                    "shape_out_law": (
                        laws.book.axes(law.out)
                        if pre_law_shapes is not None
                        and (law := pre_law_shapes.values.get(uid)) is not None
                        and law.out is not None
                        else None),
                })
            absorbed_by_sid[fused_sid] = items

        # The read side's key comes from the same owner the write side used
        # (kernel_bench plan, suggest-shapes template), so a stored row and
        # this lookup key match by construction — the refusal included. No
        # key means no lookup and the row shows as unmeasured, which is the
        # honest answer; the whole-step workload shape used to stand in here,
        # and it fabricates a key the corpus cannot contain — or worse, one
        # it wrongly does (the DSA rows measured at d_head = seq_len were keyed
        # that way).
        points = point_keys(analysis, analysis.per_op_timing, workload,
                            deployment, model_id=model_id, spec_hash=spec_hash)
        enriched: List[Dict[str, Any]] = []
        for row, point in zip(analysis.per_op_timing, points, strict=True):
            enriched_row = {**row, **identity_columns(point)}
            op = point.op

            # Fusion-identity tables (fused_op_by_sid, absorbed_by_sid,
            # fused_agg_ids) are keyed by aggregation_id — the canonical
            # global id recorded in the fusion match and post-program.
            agg_id = point.aggregation_id

            # `t_op_s` ALWAYS stays the theoretical rooftop (t_sol_s); a
            # measurement lands in the SEPARATE `t_measured_s` track beside
            # it, from the measured fold (`batch_sweep.apply_measured`).
            enriched_row["t_op_s"] = float(row.get("t_sol_s", 0.0))
            enriched_row["t_measured_s"] = None
            enriched_row["source"] = _classify_fused_source(
                agg_id, op, fused_agg_ids, deployment.framework_binding,
            )
            # A miss reason is the measured fold's to add (`missed_badge`),
            # where a lookup was made.
            badge: Dict[str, Any] = {}
            # Annotate every theoretical_fused row with the fused_op
            # name (and the framework binding's kernel_impl /
            # source_anchor) so the UI badge can show
            # "flash_attention" instead of a generic "fused" pill.
            if enriched_row["source"] == "theoretical_fused":
                fused_op = fused_op_by_sid.get(agg_id) or (
                    op if op in {
                        *(set(deployment.framework_binding.fused_ops())
                          if deployment.framework_binding is not None
                          else set())
                    } else None
                )
                if fused_op:
                    badge["fused_op"] = fused_op
                    # `absorbed` is the list of every matched sibling
                    # (including the anchor) carrying its local name,
                    # original op, and post-op shape. Lets the
                    # breakdown render "flash_attention ← qk_score +
                    # softmax + attn_apply" AND drives the hover
                    # popup that shows the naive subgraph + shape
                    # evolution. Only populated for rows we built
                    # fused_op_by_sid for — naive rows whose `op`
                    # happens to be a bound fused_op (the `or op in
                    # bound` fallback) don't have absorbed sibling
                    # info, so leave it off.
                    absorbed = absorbed_by_sid.get(agg_id)
                    if absorbed:
                        badge["absorbed"] = absorbed
                    binding = deployment.framework_binding
                    if binding is not None:
                        impls = binding.implementations_for(fused_op)
                        # More than one audited span is possible for this
                        # fused-op name; without rewrite provenance the badge
                        # must not present file order as the selected anchor.
                        if len(impls) == 1 and impls[0].source_anchor:
                            badge.setdefault("source_anchor", binding.audited_at.cite(
                                impls[0].source_anchor))
                        badge.setdefault("framework_audited_at",
                                         binding.audited_at.tag)
                    if deployment.framework:
                        badge.setdefault("framework", deployment.framework)
            enriched_row["source_badge"] = badge
            enriched.append(enriched_row)

        return CalibrationResult(per_op_timing=enriched, points=tuple(points))

def identity_columns(point: "PointRow") -> Dict[str, Any]:
    """What a per-op row states of its measurement identity at its point:
    the canonical identity bytes (`shape_json`, `dtype_env_json`; a walk's
    own rows carry placeholders until here) and the corpus bridge key — op →
    signature, library-agnostic, which the UI deep-links a row to its
    measurements by (an op no signature binds is its own name). The
    estimate's rows and every sweep point's rows state them alike."""
    return {"shape_json": point.shape_json, "dtype_env_json": point.dtype_env_json,
            "signature": point.signature or point.op}


def source_anchor(binding: Any, op: str) -> Optional[Dict[str, Any]]:
    """Where the framework binding says its kernel for `op` comes from, cited
    with links to the audited commit (`AuditedAt.cite`), for a measurement's
    badge; None without a binding or an entry."""
    impls = (binding.implementations_for(op) if binding is not None else ())
    if len(impls) != 1 or impls[0].source_anchor is None:
        return None
    return binding.audited_at.cite(impls[0].source_anchor)


def dispatch_source_anchor(binding: Any,
                           dispatch: Optional[DispatchDecision]
                           ) -> Optional[Dict[str, Any]]:
    """The audited serving-framework citation for a declared dispatch."""
    if (binding is None or dispatch is None
            or dispatch.state != DispatchDecision.DECLARED
            or dispatch.source_anchor is None):
        return None
    return binding.audited_at.cite(dispatch.source_anchor)


def missed_badge() -> Dict[str, str]:
    """What a row's badge says when its measurement was asked for and none
    was selected (missing, ambiguous, unplaced — or zero-cost, which no
    kernel measures): the estimate's overlay and every sweep point alike."""
    return {"reason": "no exact-match measurement for this key"}


def zero_cost(row: Mapping[str, Any]) -> bool:
    """Whether a row costs nothing — its law states no FLOPs, no access bytes
    and no collective payload (`row_roofline.StepTerm.zero_cost`, the one
    reading): a view op, a split. Such a row is measured at 0 — no kernel
    runs — and is never missing or unplaced, on the estimate and on every
    sweep point alike."""
    return StepTerm(row).zero_cost()


class NoShapeLaw(ValueError):
    """A measurable row's shape cannot be read at the point: the row states
    no law for it (`shape_law` is None — shape columns that are not the
    walked shape), so the rows re-read there
    (`batch_sweep.rows_at`) carry none. That point is
    walked, never guessed at."""


@dataclass(frozen=True)
class PointRow:
    """What one per-op row looks up, and launches, at one point: a cell's
    walk (`pipeline.analyze`) read at a batch and card (`point_keys`).

    Plain data. The estimate's measured overlay reads the identity and the
    key; the Kernel Bench plan builds its jobs from the identity and the
    launch (`launch_shape_*`, `*_per_launch`, `kernel_params`, `leaf`).
    """
    uid: Any
    aggregation_id: str
    #: The kernel behind the engine's label (`RowIdentity.op_of`).
    op: str
    #: The signature it is measured under, None when no signature binds the
    #: op (`op_signatures.op_bindings`): such a row is not measurable.
    signature: Optional[str]
    #: The identity shape, None when the row has no projection — a refusal,
    #: not a gap (`RowIdentity.of`).
    shape_json: Optional[str]
    dtype_env_json: str
    #: The physical op it looks up; None when there is no shape.
    key: Optional[PhysicalKey]
    #: Calls per step (`count_per_step`, provenance).
    count: int
    #: The step's launches of `launch_shape_*` (`RowIdentity.launches`).
    launches: int
    #: The share of a launch one call is (`RowIdentity.cadence`): 1, or
    #: `1/m` for a row that launches once every m steps. The step bills one
    #: launch's time `billed = launches x cadence` times — the one place a
    #: launch count enters the measured track.
    cadence: Fraction
    #: The row prices to nothing (`zero_cost`: its law states no FLOPs, no
    #: access bytes, no collective payload).
    zero_cost: bool
    #: Why the key is not this leaf's exactly ("" when it is): the row is
    #: never looked up and its coverage state is `declined`
    #: (`RowIdentity.leaf_dtype_env`).
    declined: str
    #: The closed placement reason, in the required order: a zero-cost row is
    #: measured@0 before declined, signature, and projection are considered.
    #: Empty when the row has an exact key and needs no measurement exemption.
    placement_reason: str
    #: The row's roofline time at this point.
    t_sol_s: float
    #: ONE launch's geometry (`RowIdentity.one_event`) and its weight.
    launch_shape_in: Any
    launch_shape_out: Any
    shape_weight: Any
    #: One launch's work: the walk's per-step numerators over
    #: `max(launches, 1) x cadence`. None when the row carries none.
    flops_per_launch: Optional[float]
    access_bytes_per_launch: Optional[float]
    #: Rows one launch runs, and the rows the law states for it
    #: (`RowIdentity.launch_rows`). They differ only where the law averaged:
    #: a MoE dispatch zone's `M` is the MEAN rows one expert receives and
    #: the launch pads every expert that receives any alike
    #: (`RoutedLaunch.touched_experts`).
    launch_rows: int
    launch_rows_exact: Fraction
    #: One launch's work AT THE GEOMETRY IT RUNS — the same law at the
    #: padded row count, which is the geometry a measurement of this cell is
    #: built at and therefore the numerator its latency divides. Equal to
    #: `*_per_launch` wherever nothing is padded (`padding` is 1).
    geometry_flops: Optional[float]
    geometry_access_bytes: Optional[float]
    #: The scalars the signature's kernels take beside their operands, read
    #: off the row (`OpSignature.params_from_row`).
    kernel_params: Mapping[str, float]
    #: The leaf's kernel variants (`s_window`, `sinks`), or empty
    #: (`RowIdentity.leaf_of`).
    leaf: Mapping[str, Any]
    #: The MoE dispatch boundary the launch's routed buffer is on, as the row
    #: carries it (`RoutedLaunch.wire`; None on no routed buffer): what a
    #: builder of the launch reads its expert counts and token side off.
    routed: Optional[Mapping[str, Any]] = None
    #: The leaf carries a recurrent state (`RowIdentity.carries_state`):
    #: its numerators include that state's read and write-back, so only a
    #: kernel that takes the state is this cell.
    carries_state: bool = False
    #: The framework's audited provider dispatch for this row, after the walk
    #: has resolved its physical key. No declaration is an explicit gap, not a
    #: request to borrow the fastest or only corpus row.
    dispatch: Optional[DispatchDecision] = None

    @property
    def billed(self) -> float:
        """How many of one launch's time the step bills."""
        return float(self.launches * self.cadence)

    @property
    def padding(self) -> Fraction:
        """How much wider the launch is than the law's own rows: 1 for every
        row the law did not average and for a routed one whose rows are
        whole, more where an expert's mean rows were padded up."""
        if self.launch_rows_exact <= 0:
            return Fraction(1)
        return Fraction(self.launch_rows) / self.launch_rows_exact


def _law_to_json(value: Any) -> Any:
    """A row's law value as plain JSON, expressions under one marker.

    The spelling is `sp.sstr`, read back through `LawBook.read` — the same
    closed law-text vocabulary `roofline_expr` and `shape_law` already use.
    """
    if isinstance(value, sp.Basic):
        return {"__law__": sp.sstr(value)}
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = (f"__lawkey__:{sp.sstr(k)}"
                   if isinstance(k, sp.Basic) else _law_to_json(k))
            out[key] = _law_to_json(v)
        return out
    if isinstance(value, (list, tuple)):
        return [_law_to_json(v) for v in value]
    return value


def _law_from_json(value: Any) -> Any:
    if (isinstance(value, dict) and set(value) == {"__law__"}
            and isinstance(value["__law__"], str)):
        from .row_roofline import LawBook
        return LawBook.read(value["__law__"])
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = (LawBook.read(k[len("__lawkey__:"):])
                   if isinstance(k, str)
                   and k.startswith("__lawkey__:") else _law_from_json(k))
            out[key] = _law_from_json(v)
        return out
    if isinstance(value, list):
        return [_law_from_json(v) for v in value]
    return value


def _walk_laws(value: Any) -> Any:
    if isinstance(value, sp.Basic):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk_laws(k)
            yield from _walk_laws(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_laws(v)


def _materialize_laws(value: Any, binding: Mapping[str, Any],
                      book: Any) -> Any:
    if isinstance(value, sp.Basic):
        return book.evaluate(value, {
            name: binding[name] for name in
            (symbol.name for symbol in value.free_symbols)})
    if isinstance(value, dict):
        return {_materialize_laws(k, binding, book):
                _materialize_laws(v, binding, book)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_materialize_laws(v, binding, book) for v in value]
    return value


class _StoredRowIdentity(RowIdentity):
    """A serialized row's resolved identity, without its graph.

    The key derivation is still `RowIdentity._keyed`: only the facts that
    derivation asked the graph for at walk time (`op`, leaf variants, state
    carrier) have been resolved and stored. A consumer cannot rederive them.
    """

    def __init__(self, *, op: str, leaf: Mapping[str, Any],
                 served: Mapping[str, int], dtype_env_json: str,
                 declined: str):
        self._served = dict(served)
        self._leaves = {}
        self._result = None
        self._program = None
        self._node_by_sid = {}
        self._op = op
        self._leaf = dict(leaf)
        self._dtype_env_json = dtype_env_json
        self._declined = declined

    def leaf_of(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        return self._leaf

    def leaf_dtype_env(self, row: Mapping[str, Any]
                      ) -> Tuple[str, str]:
        return self._dtype_env_json, self._declined

    def op_of(self, row: Mapping[str, Any], node: Optional[Any]) -> str:
        return self._op

    def _node_of(self, row: Mapping[str, Any]) -> Optional[Any]:
        return None

    def carries_state(self, row: Mapping[str, Any]) -> bool:
        return bool(row.get("__carries_state__"))


def key_columns_at(read: Mapping[Any, Any], row: Mapping[str, Any]
                    ) -> Dict[str, Any]:
    """What the row's measurement key is read off at a point (`read`)
    beside its shapes (`_shape_columns_at`): its node's dims and a
    collective's payload shape (`point_keys`)."""
    point = row["point_law"]
    dims = point["dims"]
    out: Dict[str, Any] = {
        "node_dims": (None if dims is None else
                      {name: (law["value"] if isinstance(law, dict)
                              else as_number(read[law]))
                       for name, law in dims.items()})}
    if point["comm"] is not None:
        out["comm_payload_shape"] = [int(read[law])
                                     for law in point["comm"]["payload"]]
    return out


def key_columns(rows: Sequence[Mapping[str, Any]], cell: Any
                ) -> List[Mapping[str, Any]]:
    """Each row with its key columns (`key_columns_at`): a row that states
    them is itself — the walk's own, or one read at another point with its
    breakdown — and a row read at another point without it (`batch_sweep.
    rows_at`: it carries its laws, `point_law`, and not what its key is
    read off) has them read there, at its own point, off its laws in the
    book of `cell` (the walk the rows come from). A row with no laws at all
    keys off what it states, its own `node_dims` included (`RowIdentity`)."""
    unkeyed = {id(r): r for r in rows
               if "point_law" in r and "node_dims" not in r}
    if not unkeyed:
        return list(rows)
    read = Point.laws_by_point(cell.laws.book, (
        (r["point_law"]["point"], law) for r in unkeyed.values()
        for law in (*(v for v in (r["point_law"]["dims"] or {}).values()
                      if not isinstance(v, dict)),
                    *(r["point_law"]["comm"] or {}).get("payload", ()))))
    return [{**r, **key_columns_at(read(r["point_law"]["point"]), r)}
            if id(r) in unkeyed else r
            for r in rows]


def _at_geometry(row: Mapping[str, Any], *, billed: float,
                 ) -> Tuple[Optional[float], Optional[float]]:
    """One launch's `(flops, access_bytes)` at the geometry it RUNS.

    The walk's numerators are the law at the rows the law states, and a
    launch on a routed buffer runs more than that: the experts that receive
    rows, each padded up to a whole number of rows (`RoutedLaunch`:
    `launch_rows` against `routed_rows`), which is the geometry a
    measurement of the cell is built at and therefore the geometry its
    latency has to be divided by.

    The same law at that geometry, term by term, each term the walk's own:

      * the bytes on the routed buffer (`routed["bytes_per_row"]`, the
        walk's per routed row: `LeafAccess.routed_extent_per_rank`) at the
        launched rows instead of the routed ones — a dispatch boundary's
        token side and its routing are not on the buffer and do not move;
      * the resident weights and their scales: the walk's saturating read
        (`row_roofline.weight_demand`) re-read by its owner
        (`RowRoofline.stored_read`) at the launched rows — the experts the
        launch touches, where the walk's mean reads `Min(R, E)` of them;
      * the FLOPs, which run on the routed rows, at the launched rows.

    Every other row pads nothing, and this is the walk's numerators, bit for
    bit, whatever its extents do: a compressor that pools once every few
    tokens has a CADENCE, which the plan refuses by name rather than
    widening it (`plan._rounded_axes`).
    """
    flops, access = row.get("flops_per_rank"), row.get("access_bytes_per_rank")
    f = None if flops is None else float(flops) / billed
    a = None if access is None else float(access) / billed
    launch = routed_launch(row)
    if launch is None or not launch.routed_rows:
        return f, a
    routed, launched = launch.routed_rows, Fraction(launch.launch_rows)
    if f is not None:
        f *= float(launched / routed)
    if a is None:
        return f, a
    per_row = Fraction(str(row["routed"]["bytes_per_row"]))
    a += float(per_row * (launched - routed))
    weight_rows = row["routed"].get("weight_rows_per_row")
    total = row.get("weight_bytes_total_per_rank") or 0
    if weight_rows is not None and total:
        # The walk's own saturating read (`row_roofline.weight_demand`:
        # its rows times one row's demand, capped at the bank), re-read by
        # its owner at the launched rows; the walk states it in whole-bank
        # reads per routed row, so one row asks `total` here.
        at_launch = RowRoofline.stored_read(
            rows=Fraction(str(weight_rows)) * launched, read_per_row=total,
            total=total,
            metadata=row.get("metadata_bytes_total_per_rank") or 0)
        walked = ((row.get("weight_read_bytes_per_rank") or 0)
                  + (row.get("metadata_bytes_per_rank") or 0))
        a += float(at_launch.weight + at_launch.metadata - walked) / billed
    return f, a


def point_row_of(row: Mapping[str, Any], identity: RowIdentity,
                 deployment: DeploymentContext, *, model_id: str,
                 spec_hash: str,
                 dispatch_override: Optional[DispatchDecision] = None
                 ) -> PointRow:
    """One row's PointRow — the single key derivation.

    """
    binding = deployment.framework_binding
    dispatched_binding = (
        None if binding is None
        else binding.deployed(replace(
            binding.deployment,
            kv_block_size=deployment.kv_block_size,
            tensor_parallel=deployment.tensor_parallel)))
    bindings = op_bindings()
    op, shape_json, (shape_in, shape_out) = identity.of_event(row)
    signature = bindings.get(op)
    if signature is not None and row.get("shape_at_batch") is False:
        raise NoShapeLaw(
            f"{row.get('aggregation_id')}: no shape law to read at this batch")
    sid = str(row.get("aggregation_id", "") or "")
    dtype_env_json, declined = identity.leaf_dtype_env(row)
    placement_reason = (
        ZERO_LAUNCH if zero_cost(row) else
        DECLINED_REASON if declined else
        MISSING_SIGNATURE if signature is None else
        MISSING_PROJECTION if shape_json is None else "")
    launches = identity.launches(row)
    cadence = identity.cadence(row)
    billed = float(max(launches, 1) * cadence)
    flops, access = row.get("flops_per_rank"), row.get("access_bytes_per_rank")
    declared = get_signature(signature) if signature is not None else None
    resolved_launch = ResolvedLaunch.from_dtype_env_json(dtype_env_json)
    analysis = (None if identity._result is None
                else identity._result.analysis)
    dispatch = (
        dispatch_override if dispatch_override is not None else None
        if binding is None or analysis is None else
        dispatched_binding.dispatch_for(
            op, analysis.program, analysis, resolved_launch))
    padded_rows, exact_rows = identity.launch_rows(row)
    # Padding is a routed buffer's and no other tensor's
    # (`RoutedLaunch`): a grouped launch wants whole rows per expert,
    # and every other averaged extent is a cadence the plan refuses by
    # name (`plan._starved_axes` / `_rounded_axes`). So a row on no
    # routed buffer keeps the walk's numerators whatever its extents do.
    at_geometry = _at_geometry(row, billed=billed)
    return PointRow(
        uid=row.get("uid"),
        aggregation_id=sid,
        op=op,
        signature=signature,
        shape_json=shape_json,
        dtype_env_json=dtype_env_json,
        key=None if shape_json is None else BenchmarkKey(
            # Model, spec and node are provenance, not physics, and the
            # version is the measurement's, not the request's.
            model=model_id, spec_hash=spec_hash,
            phase=deployment.phase, parallel=deployment.parallel_signature,
            framework=deployment.framework, framework_version="",
            gpu=deployment.gpu_sku, op=op, node_spec_id=sid,
            shape_json=shape_json, dtype_env_json=dtype_env_json,
            kernel_impl="",
        ).to_physical(),
        count=int(row.get("count_per_step", 1)),
        launches=launches,
        cadence=cadence,
        zero_cost=zero_cost(row),
        declined=declined,
        placement_reason=placement_reason,
        t_sol_s=float(row.get("t_sol_s") or 0.0),
        launch_shape_in=shape_in,
        launch_shape_out=shape_out or row.get("shape_outs"),
        shape_weight=identity.launch_weight(row),
        flops_per_launch=None if flops is None else float(flops) / billed,
        access_bytes_per_launch=None if access is None else float(access) / billed,
        launch_rows=padded_rows,
        launch_rows_exact=exact_rows,
        routed=row.get("routed"),
        geometry_flops=at_geometry[0],
        geometry_access_bytes=at_geometry[1],
        kernel_params=declared.params_from_row(row) if declared is not None else {},
        leaf=identity.leaf_of(row),
        carries_state=identity.carries_state(row),
        dispatch=dispatch,
    )



def point_keys(cell: Any, rows: Sequence[Mapping[str, Any]],
               workload: WorkloadShape, deployment: DeploymentContext, *,
               model_id: str, spec_hash: str) -> List[PointRow]:
    """Every row's identity, key and launch at one point — the one place a
    per-op row becomes "what this looks up / measures".

    `cell` is the walk the rows come from (`pipeline.analyze`'s result), and
    `rows` its per-op rows at the point: the walk's own (`per_op_timing`),
    or those rows re-read at another batch and card
    (`batch_sweep.rows_at`). The estimate's overlay, a Kernel Bench plan, a
    serving-law producer and any rescaled sweep point ask here, so a stored
    measurement and the lookup for it are keyed by one derivation.

    Raises `NoShapeLaw` for a measurable row re-read at a batch its walk
    stated no shape law for.
    """
    # What each key is read off, at each row's own point (`key_columns`).
    rows = key_columns(rows, cell)
    # The serving choices a key reads: the page and the Mamba-2 scan chunk,
    # which the walk's checkpoint config states at the fields the framework
    # reads it from (`Program.scan_chunk_size`).
    binding = deployment.framework_binding
    served = (deployment.kv_block_size,
              None if binding is None else cell.analysis.program
              .scan_chunk_size(binding.ssm_chunk_size_from))
    identity = cell.identities.get(served)
    if identity is None:
        identity = cell.identities[served] = RowIdentity(
            cell, kv_block_size=served[0], ssm_chunk_size=served[1])
    return [point_row_of(row, identity, deployment, model_id=model_id,
                         spec_hash=spec_hash) for row in rows]


# ---------------------------------------------------------------------------
# Classification helpers (formerly in phases.py)
# ---------------------------------------------------------------------------


def _classify_fused_source(
    sid: str,
    op: str,
    fused_aggregation_ids: Optional[set],
    framework_binding: Optional[Any],
) -> str:
    """Return `"theoretical_fused"` when this row is a fused leaf,
    else `"theoretical_naive"`.

    Two signals, tried in order:
      1. `fused_aggregation_ids` — structural set of synthesized leaves'
         aggregation_ids (`template|spec_id`) from the graph rewrite. Works
         regardless of whether the framework binding is available.
      2. Framework binding's `fused_ops()` set. When `framework_binding
         is None`, returns `"theoretical_naive"` — the string heuristic
         (`flash_*` / `*fused*`) is intentionally NOT a fallback
         (mislabels binding-given names like
         `merged_column_parallel_linear`).
    """
    if fused_aggregation_ids and sid in fused_aggregation_ids:
        return "theoretical_fused"
    if framework_binding is None:
        return "theoretical_naive"
    # No try/except: `fused_ops()` is a tuple comprehension over the
    # binding's own declarations — it only raises on a malformed binding,
    # and swallowing that classified every op as naive with no symptom.
    return ("theoretical_fused" if op in set(framework_binding.fused_ops())
            else "theoretical_naive")


# ---------------------------------------------------------------------------
# Convenience: build the typed inputs from raw HTTP-style params
# ---------------------------------------------------------------------------


def build_deployment_context(
    *,
    framework: str,
    framework_binding: Optional[Any],
    parallel: Any,                       # ParallelConfig
    gpu_sku: str,
    phase: str = "prefill",
    kv_block_size: Optional[int] = None,
) -> DeploymentContext:
    """Build a DeploymentContext from the call-site primitives. The
    `parallel_signature` is the canonical sorted string form used as a
    benchmark identity component.

    `kv_block_size` is the request's paged-KV page, and omitting it takes
    the framework binding's own default — one owner for the default, the
    framework that owns the cache manager, rather than a number repeated at
    every call site.
    """
    sig = parallel_signature(parallel)
    tp = 1 if parallel is None else parallel.tp_attention
    return DeploymentContext(
        framework=framework,
        framework_binding=framework_binding,
        parallel_signature=sig,
        gpu_sku=gpu_sku,
        phase=phase,
        kv_block_size=deployment_kv_block_size(framework_binding, kv_block_size),
        tensor_parallel=tp)


def parallel_signature(parallel: Any) -> str:
    """Canonical sorted `k=v,k=v,...` form for ParallelConfig — the CSV
    `parallel` column."""
    if parallel is None:
        return "tp=1"
    pairs: List[tuple] = []
    for knob in (
        "pp", "tp_attention", "tp_dense_ffn", "tp_moe_expert",
        "ep_moe", "cp", "dp_attention", "dp_replicas",
    ):
        v = getattr(parallel, knob, None)
        if v is None:
            continue
        # ParallelConfig declares every knob `int`, so int(v) is a no-op
        # coercion here. Don't swallow a coercion failure: this string is
        # the CSV benchmark *identity*, and a silently-dropped knob would
        # corrupt it into a wrong-but-plausible match (§1).
        iv = int(v)
        if iv > 1:
            pairs.append((knob, iv))
    if not pairs:
        return "tp=1"
    pairs.sort()
    return ",".join(f"{k}={v}" for k, v in pairs)


# ---------------------------------------------------------------------------
# Capacity orchestration — resolve the registry once, then per seq
# ---------------------------------------------------------------------------


def resolve_framework_binding(
    *, framework: Optional[str],
    deployment: Optional[ServingDeployment] = None,
) -> tuple:
    """`(framework, binding)` an estimate runs under: the named framework's
    binding, or the default framework's (`DEFAULT_FRAMEWORK`) when none is
    named, deployed on `deployment`; no binding for `THEORY`, the engine's own
    analysis. The one place a request's framework is resolved."""
    fw = framework or DEFAULT_FRAMEWORK
    if fw == THEORY:
        return THEORY, None
    framework_binding = get_binding(fw)
    if framework_binding is not None and deployment is not None:
        framework_binding = framework_binding.deployed(deployment)
    return fw, framework_binding


def deployment_kv_block_size(framework_binding: Optional[Any],
                             requested: Optional[int] = None) -> Optional[int]:
    """The paged-KV page this deployment allocates the cache in, or None.

    The request's value when it states one; otherwise the framework
    binding's own declared page, because the framework owns the cache
    manager. `None` when neither says — sglang declares no page (it
    resolves one per attention backend) and `THEORY` has no binding and
    runs no kernels. A kernel that reads a paged cache then has no cell
    (the page is an axis of it, `op_signatures.PAGE_AXIS`) instead of one
    keyed on a guessed page: a lookup keyed on 16 for a deployment that
    never said 16 is a wrong answer dressed as a match.

    One function, because two callers need the answer (the deployment
    context the estimate looks rows up with, and the capacity params that
    carry it across a multi-seq sweep) and a number repeated at two call
    sites is the next thing to drift.
    """
    if requested is not None:
        return int(requested)
    if framework_binding is not None and framework_binding.kv_block_size:
        return int(framework_binding.kv_block_size)
    return None


def looks_up(selection_mode: str) -> bool:
    """Whether a request in `selection_mode` looks measurements up at all:
    every mode but `theoretical_only` does, under whatever scope it comes
    with. Only for such a request is a point's measured total — and so its
    rows' keys (`point_keys`) — asked for."""
    return selection_mode != "theoretical_only"


def _resolve_overlay_context(
    *, framework: Optional[str],
    selection_mode: str,
    measurement_scope: Any = None,
    calibration_rows: Optional[list] = None,
    deployment: Optional[ServingDeployment] = None,
) -> tuple:
    """Resolve the (framework, framework_binding, strategy,
    enable_measured_lookup) tuple once. Pulled out of the overlay walk so
    the multi-seq sweep can hoist this work above the per-seq loop.

    Measured lookup runs when the mode reads measurements (not
    `theoretical_only`) and there are some to read: a Kernel Bench scope
    (`measure.scope.MeasurementScope.reads`) or rows a headless caller posts
    with the request (`calibration_rows`). The rows themselves are fetched
    once per estimate, for the keys its overlays ask (`OverlayWalk.keys`,
    `op_calibration.registry_for`).

    The binding comes back `deployed` on `deployment` (GPU device,
    activation dtype, server flags), so an entry whose framework runs its
    kernel only under conditions is answered for this request."""
    reads = bool(measurement_scope is not None and measurement_scope.reads)
    enable_measured_lookup = (
        looks_up(selection_mode) and (reads or bool(calibration_rows)))

    fw, framework_binding = resolve_framework_binding(
        framework=framework, deployment=deployment)

    strategy = "envelope" if selection_mode == "envelope" else "latest"
    return (fw, framework_binding, strategy, enable_measured_lookup)


@dataclass
class OverlayWalk:
    """One phase's overlay: the structural walk (`pipeline.analyze`) and the
    typed inputs the overlay reads. `finish()` applies it — identity and
    theory classification; the measured track is the estimate's measured
    fold's (`batch_sweep.measure_points`)."""
    analysis: Any
    workload: WorkloadShape
    deployment: DeploymentContext
    model_id: str
    spec_hash: str
    pipeline_time: Optional[dict]
    want_module_rows: bool
    #: The step whose rows `analysis` walked. The phase facts the rows do
    # not carry (notably whether the decode MLA rebind fired) are read from
    # here, not re-derived by a consumer.
    step_eval: Any = None

    def finish(self) -> tuple:
        """``(per_op_timing, fusion_delta, has_dispatch_zone, module_rows,
        pipeline_time, points)`` — see `overlay_walk`; `points` each row's
        `PointRow` at the walk's point."""
        result = CalibrationOverlay().apply(
            self.analysis, self.workload, self.deployment,
            model_id=self.model_id, spec_hash=self.spec_hash)
        # The phase's per-op table, walked off the grafted + fused tree: comm
        # and fused rows are native here, so the breakdown needs no splice, no
        # fused-away filter and no aggregation_id-keyed shape override.
        module_rows = (
            physical_module_rows(self.analysis, result.per_op_timing)
            if self.want_module_rows else []
        )
        return (
            result.per_op_timing,
            self.analysis.fusion_delta(),
            bool(self.analysis.dispatch_zone_uids),
            module_rows,
            self.pipeline_time,
            result.points,
        )


def overlay_walk(
    *,
    resolved: ResolvedSpec,
    parallel: Any,  # ParallelConfig
    framework: str,
    gpu_sku: str,
    phase: str,
    avg_input_tokens: int,
    avg_output_tokens: int,
    s_prefix_cached: int,
    s_verify: int,
    max_batch: int,
    precision: Any,
    peak_flops: Optional[float],
    peak_bw: Optional[float],
    gpu: Any,
    peak_intra_bw: Optional[float],
    peak_inter_bw: Optional[float],
    intra_bw_label: str,
    inter_bw_label: str,
    comm_latency_s: float,
    launch_floor_s: float,
    model_id: str,
    # The binding `_resolve_overlay_context` returns, resolved once by the
    # caller with the request's deployment (GPU device, activation dtype,
    # server flags), so it answers its `when:` conditions for this request.
    framework_binding: Any,
    # The pipeline stage this rank runs — the bound stage
    # (`_StageProfile.stage`): its rows, and the hops across its exit cut.
    # None at pp=1.
    stage: Optional[int] = None,
    # The step these rows are a view of, evaluated once by the caller and
    # shared with the phase table. See `step_eval.StepEvaluation`.
    step_eval: Optional[Any] = None,
    # Which stacks' leaves become rows; None is the backbone. The speculative
    # draft names its MTP stack here (`pipeline.analyze`'s `timed_stacks`).
    timed_stacks: Optional[Any] = None,
    # The request's paged-KV page, or None for the framework binding's own
    # default. Part of the key of every kernel that reads a paged cache.
    kv_block_size: Optional[int] = None,
    # The draft pass only needs the per-op rows; it has no module table of
    # its own, so it skips the post-tree walk.
    want_module_rows: bool = True,
    # pp: how many stages the pipeline-step time sums over. 1 = no
    # pipeline; >1 runs one extra analyze per additional stage.
    pipeline_stages: int = 1,
    # Whether the rows carry what an estimate's breakdown draws, and the
    # phase its module table (`pipeline.step_rows`): an estimate's walk
    # does, the walk a sweep reads its points off does not.
    breakdown: bool,
) -> OverlayWalk:
    """Run pipeline.analyze for structural per-op rows, ready to be enriched
    via CalibrationOverlay (`OverlayWalk.finish`), which returns
    ``(per_op_timing, fusion_delta, has_dispatch_zone, module_rows,
    pipeline_time)`` — `fusion_delta` is the theoretical step time before
    and after fusion (`AnalysisResult.fusion_delta`).

    `module_rows` is the phase's per-op table, walked off the grafted +
    fused tree (`module_rows.physical_module_rows`): comm and fused rows
    are native there, so no splice, no fused-away filter and no
    aggregation_id-keyed shape override runs afterwards.

    `pipeline_time` is None at pp=1; at pp>1 it is the pipeline-step time
    model — Σ per-stage roofline + inter-stage hops, bubble unmodelled —
    with every stage's row laws (`stage_laws`) to re-price it by.

    Structural analysis failures propagate. Missing measurements are handled
    by ``CalibrationOverlay`` as theoretical rows, without changing geometry.
    """
    # Structural walk stays a function-body import so this module can
    # load without pulling pipeline/phases (and whatever they pull) at
    # import time. The overlay stage runs after that walk.
    from .pipeline import analyze as _pipeline_analyze

    # The phase resolves the step's query block from the workload
    # (`phases.phase_binding`, which is `Phase.resolve` read at the point);
    # the walk is handed the workload itself.
    from .phases import phase_binding
    if phase == ENCODE:
        # An encoder's step binds its media, not a query block: the walk is
        # the caller's (`step_eval.evaluate_encode`), and a row's "token" is
        # one request.
        if step_eval is None:
            raise ValueError("an encode step is evaluated by the caller "
                             "(`step_eval.evaluate_encode`)")
        Sq = 1
    else:
        binding = phase_binding(
            phase=phase,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            s_prefix_cached=s_prefix_cached,
            s_verify=s_verify,
        )
        Sq = max(1, int(binding["s_query"]))

    # Run structural analysis at the capacity-admitted batch so per-op
    # rows describe one full-step at max_batch — Σ per_op flops/access
    # then equals the per-step top-table aggregates. Workload (below)
    # stays pinned to B=1 because CSV identity is canonical at B=1.
    overlay_batch = max(1, int(max_batch or 1))
    analyze_kwargs = dict(
        parallel=parallel,
        framework_binding=framework_binding,
        batch_size=overlay_batch,
        precision=precision,
        peak_flops_per_sec=peak_flops,
        peak_hbm_bw_bytes_per_sec=peak_bw,
        gpu=gpu,
        intra_bw_bytes_per_sec=peak_intra_bw,
        inter_bw_bytes_per_sec=peak_inter_bw,
        intra_bw_label=intra_bw_label,
        inter_bw_label=inter_bw_label,
        comm_latency_s=comm_latency_s,
        launch_floor_s=launch_floor_s,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        s_prefix_cached=s_prefix_cached,
        phase=phase,
        s_verify=s_verify,
        step_eval=step_eval,
        timed_stacks=timed_stacks,
        breakdown=breakdown,
    )
    analysis = _pipeline_analyze(resolved, stage=stage, **analyze_kwargs)

    # The pipeline-step time at pp>1: Σ over every stage's roofline plus
    # the inter-stage hops (each stage's rows price its own exit cut, so
    # each hop lands exactly once). The pipeline bubble is a steady-state
    # throughput artifact, not a roofline term — deliberately unmodelled.
    # Every stage sums the same theoretical rows (no per-stage overlay, so
    # no stage reads a different calibration than another).
    pipeline_time = None
    if pipeline_stages > 1 and stage is not None:
        all_stage_rows = [
            analysis.per_op_timing if s == stage
            else _pipeline_analyze(resolved, stage=s, **analyze_kwargs).per_op_timing
            for s in range(pipeline_stages)]
        # The step's sum (`StepSum`) over every stage's rows; each stage's own
        # time beside it.
        step = StepSum.of(stages=all_stage_rows, rank_stage=stage)
        sums = {key: step.total(lambda t, col=col: float(t.row[col]))
                for key, col in (("t_compute", "t_compute_s"),
                                 ("t_memory", "t_memory_s"),
                                 ("t_comm", "t_comm_s"), ("t_sol", "t_op_s"))}
        sums.update(stages=pipeline_stages, stage=stage)
        sums["per_stage"] = [
            StepSum.of(stage_rows).total(lambda t: float(t.row["t_op_s"]))
            for stage_rows in all_stage_rows]
        # Each stage's rows under the term that set them, summed over the
        # stages: the pipeline step's `t_sol_by_bound`.
        sums["t_sol_by_bound"] = step.by_bound(
            lambda t: RowTimes.of_row(t.row))
        # Every stage's row laws — its exit-cut hop's included — so another
        # batch or card re-prices each stage by substitution and sums them
        # the same way (`batch_sweep.phase_at`).
        sums["stage_laws"] = [[row_law(r) for r in stage_rows]
                              for stage_rows in all_stage_rows]
        pipeline_time = sums

    # Batch 1 for the lookup identity, independent of overlay_batch.
    workload = WorkloadShape.of_walk(analysis, seq_len=Sq)
    deployment = build_deployment_context(
        framework=framework,
        framework_binding=framework_binding,
        parallel=parallel,
        gpu_sku=gpu_sku,
        phase=phase,
        kv_block_size=kv_block_size,
    )
    return OverlayWalk(
        analysis=analysis, workload=workload, deployment=deployment,
        model_id=model_id, spec_hash=resolved.identity_hash(),
        pipeline_time=pipeline_time,
        want_module_rows=want_module_rows and breakdown,
        step_eval=step_eval)
