"""Emit a measured CSV row conforming to the shared schema.

One measurement (Invocation × Provider × LatencyStats in an Env) → one
row dict keyed by `backend.inference.benchmarks.ALL_COLUMNS`. The library
+ version land in the identity's `framework` / `framework_version` /
`kernel_impl` (provenance-as-identity, so cross-library rows coexist and
`best_of` picks the envelope). `achieved_*` are computed from the engine's
per-rank flops / access_bytes ÷ the measured latency.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, Optional, Tuple

from backend.inference import benchmarks as _benchmarks
from backend.inference.benchmark_identity import (
    InvalidIdentityJSON,
    canonical_identity_json,
)
from backend.inference.op_signatures import op_signature

from .env import Env
from .invocation import Invocation
from .providers import Launch, LatencyStats, Provider


# A row may name the card's own peak, never beat it: a larger claimed rate is
# the signature of a numerator that does not describe the timed launch.
UTILIZATION_CEILING = 1.0

PhysicalCell = Tuple[str, str, str, str]
NumeratorPair = Tuple[Optional[float], Optional[float]]


def utilization(row: Dict[str, str]) -> Optional[tuple[float, float]]:
    """MFU and MBU as fractions of this row's recorded peaks.

    None when the numbers cannot be read. 0 for a side whose peak was not
    recorded — the SOL gate does not judge an unstated peak.
    """
    try:
        peak_t = float(row.get("peak_tflops_at_test") or 0)
        peak_b = float(row.get("peak_bw_at_test") or 0)
        mfu = float(row.get("achieved_tflops") or 0) / peak_t if peak_t else 0.0
        mbu = float(row.get("achieved_bw_gbps") or 0) / peak_b if peak_b else 0.0
    except ValueError:
        return None
    return mfu, mbu


def achieved_from_latency(
    flops_per_rank: Optional[float],
    access_bytes_per_rank: Optional[float],
    p50_us: float,
) -> tuple[str, str]:
    """`achieved_tflops`, `achieved_bw_gbps` from plan numerators and p50.

    Empty string when a numerator is missing. `0.0000` when the numerator is
    legitimately zero (a gather, a permute) so the schema still sees a number.
    """
    latency_s = float(p50_us) * 1e-6
    tflops = bw = ""
    if flops_per_rank is not None and latency_s > 0:
        tflops = f"{float(flops_per_rank) / latency_s / 1e12:.4f}"
    if access_bytes_per_rank is not None and latency_s > 0:
        bw = f"{float(access_bytes_per_rank) / latency_s / 1e9:.4f}"
    return tflops, bw


def restamp_row(row: Dict[str, str], *, flops_per_rank: Optional[float],
                access_bytes_per_rank: Optional[float]) -> Dict[str, str]:
    """Rewrite achieved_* from new plan numerators and the row's own p50."""
    out = dict(row)
    tflops, bw = achieved_from_latency(
        flops_per_rank, access_bytes_per_rank, float(row.get("p50_us") or 0))
    if tflops:
        out["achieved_tflops"] = tflops
    if bw:
        out["achieved_bw_gbps"] = bw
    return out


@dataclass(frozen=True)
class RestampUnmatched:
    """Why one row could not be priced by the supplied plan."""

    row_index: int
    reason: str
    op: str = ""
    shape_json: str = ""
    dtype_env_json: str = ""
    phase: str = ""


@dataclass
class RestampAmbiguity:
    """A complete physical cell that the supplied plan prices two ways."""

    cell: PhysicalCell
    pairs: tuple[NumeratorPair, ...]
    row_indexes: list[int] = field(default_factory=list)


@dataclass
class RestampRowsResult:
    """The auditable result of a numerator restamp.

    Unpacking remains ``(rows, changed, missed)`` for the command-line caller.
    ``unmatched`` and ``ambiguous`` carry the facts a count alone hides.  An
    ambiguous row is deliberately absent from ``missed``: its cell was
    planned, but no numerator may be selected from the collision.
    """

    rows: list
    changed: int
    unmatched: list[RestampUnmatched]
    ambiguous: list[RestampAmbiguity]

    @property
    def missed(self) -> int:
        return len(self.unmatched)

    def __iter__(self) -> Iterator[Any]:
        return iter((self.rows, self.changed, self.missed))


def _physical_key_cell(job: Dict[str, Any]) -> PhysicalCell:
    """Read the measurement key a plan carries; never rebuild it here.

    ``physical_key`` is the ownership boundary established by
    ``calibration_overlay.point_keys`` and ``driver.physical_key_for``.  Its
    JSON fields are only checked against the canonical producer so malformed
    plan text is refused rather than silently joined under a second spelling.
    """
    carried = job.get("physical_key")
    if not isinstance(carried, (list, tuple)) or len(carried) != 4:
        raise ValueError(
            f"job {job.get('op') or '<unknown>'} "
            f"[{job.get('node_spec_id', '')}] carries no physical_key; "
            "re-run `bench plan` (restamp must not reconstruct the key)")
    op, shape_json, dtype_env_json, phase = carried
    try:
        if (canonical_identity_json(shape_json) != shape_json
                or canonical_identity_json(dtype_env_json) != dtype_env_json):
            raise InvalidIdentityJSON("physical_key JSON is not canonical")
    except InvalidIdentityJSON as exc:
        raise ValueError(
            f"job {op!r} [{job.get('node_spec_id', '')}] has an invalid "
            f"physical_key: {exc}") from exc
    return op, shape_json, dtype_env_json, phase


def _row_cell(
        row: Dict[str, Any], row_index: int,
) -> Tuple[Optional[PhysicalCell], RestampUnmatched]:
    """Project and canonicalize a row's CSV identity, or say why it cannot join.

    A row carries the engine's fused op name; a plan carries the signature
    that op is measured under.  Project through the signature catalog's one
    ``binds_ops`` map (``op_signature``), exactly as plan-key generation does.
    An op no signature binds remains its own name and therefore misses as
    ``no_plan`` rather than being silently re-labelled.
    """
    op = row.get("op", "")
    phase = row.get("phase", "")
    prefix = {"row_index": row_index, "op": op, "phase": phase}
    if "dtype_env_json" not in row:
        return None, RestampUnmatched(reason="missing_dtype_column", **prefix)
    if row.get("dtype_env_json") in (None, ""):
        return None, RestampUnmatched(reason="invalid_dtype_identity", **prefix)
    try:
        shape_json = canonical_identity_json(row.get("shape_json"))
    except InvalidIdentityJSON:
        return None, RestampUnmatched(
            reason="invalid_shape_identity",
            shape_json=str(row.get("shape_json", "")),
            dtype_env_json=str(row.get("dtype_env_json", "")), **prefix)
    try:
        dtype_env_json = canonical_identity_json(row.get("dtype_env_json"))
    except InvalidIdentityJSON:
        return None, RestampUnmatched(
            reason="invalid_dtype_identity", shape_json=shape_json,
            dtype_env_json=str(row.get("dtype_env_json", "")), **prefix)
    cell = op_signature(op), shape_json, dtype_env_json, phase
    return cell, RestampUnmatched(
        shape_json=cell[1], dtype_env_json=cell[2], reason="", **prefix)


def _pair(job: Dict[str, Any]) -> NumeratorPair:
    out: list[Optional[float]] = []
    for field_name in ("flops_per_rank", "access_bytes_per_rank"):
        value = job.get(field_name)
        if value is None:
            out.append(None)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"job {job.get('op') or '<unknown>'} has invalid "
                f"{field_name}: {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(
                f"job {job.get('op') or '<unknown>'} has non-finite "
                f"{field_name}")
        out.append(number)
    return out[0], out[1]


def _miss_reason(cell: PhysicalCell, cells: dict[PhysicalCell, Any],
                 op_shapes_phases: set, op_dtypes_phases: set) -> str:
    if not cells:
        return "no_plan"
    op, shape, dtype, phase = cell
    if (op, shape, phase) in op_shapes_phases:
        return "dtype_not_in_plan"
    if (op, dtype, phase) in op_dtypes_phases:
        return "geometry_not_in_plan"
    return "no_plan"


def restamp_rows(rows: list, jobs: list) -> RestampRowsResult:
    """Apply plan numerators to measured rows' physical cells.

    Current plans carry ``physical_key``; those keys are read verbatim.  A CSV
    row has no carried key, so its four identity fields go through the one
    canonical physical-cell helper also used by lake reconciliation.  Dtype is
    part of the complete cell: a BF16 and FP32 launch at one shape move
    different byte counts.  Distinct numerator pairs on one complete cell are
    reported, and every such row is left unchanged.
    """
    pairs_by_cell: dict[PhysicalCell, set[NumeratorPair]] = {}
    for job in jobs:
        pairs_by_cell.setdefault(_physical_key_cell(job), set()).add(_pair(job))

    ambiguous = [
        RestampAmbiguity(cell, tuple(sorted(pairs, key=repr)))
        for cell, pairs in sorted(pairs_by_cell.items(), key=repr)
        if len(pairs) > 1
    ]
    ambiguity_by_cell = {item.cell: item for item in ambiguous}
    op_shapes_phases = {(op, shape, phase)
                        for op, shape, _dtype, phase in pairs_by_cell}
    op_dtypes_phases = {(op, dtype, phase)
                        for op, _shape, dtype, phase in pairs_by_cell}

    out: list = []
    changed = 0
    unmatched: list[RestampUnmatched] = []
    for row_index, row in enumerate(rows):
        cell, detail = _row_cell(row, row_index)
        if cell is None:
            unmatched.append(detail)
            out.append(row)
            continue

        collision = ambiguity_by_cell.get(cell)
        if collision is not None:
            collision.row_indexes.append(row_index)
            out.append(row)
            continue

        pair = pairs_by_cell.get(cell)
        if pair is None or not pair:
            unmatched.append(replace(detail, reason=_miss_reason(
                cell, pairs_by_cell, op_shapes_phases, op_dtypes_phases)))
            out.append(row)
            continue

        flops, nbytes = next(iter(pair))
        new = restamp_row(
            row, flops_per_rank=flops, access_bytes_per_rank=nbytes)
        if (new.get("achieved_tflops") != row.get("achieved_tflops")
                or new.get("achieved_bw_gbps") != row.get("achieved_bw_gbps")):
            changed += 1
        out.append(new)

    return RestampRowsResult(out, changed, unmatched, ambiguous)


def above_sol(row: Dict[str, str]) -> Optional[str]:
    """Why this row could not have happened, or None.

    `t_measured < t_sol(同形状)` is physically impossible: the work the row
    claims (`achieved_*`) exceeds what the card it was measured on can do
    (`peak_*_at_test`). Rows with no recorded peaks pass — the gate judges
    only what the row itself states.
    """
    got = utilization(row)
    if got is None:
        # Unparseable numbers are the schema validator's call, not this gate's.
        return None
    mfu, mbu = got
    if mfu > UTILIZATION_CEILING:
        return (f"claimed compute is {mfu * 100:.0f}% of the card's peak "
                f"(>{UTILIZATION_CEILING:.0%})")
    if mbu > UTILIZATION_CEILING:
        return (f"claimed bandwidth is {mbu * 100:.0f}% of the card's peak "
                f"(>{UTILIZATION_CEILING:.0%})")
    return None


def _stamp_driver(row: Dict[str, str], env: Env) -> None:
    """Which driver ran this row: the box's kernel-mode driver and the
    user-mode driver the measuring process actually loaded.

    One definition for both row builders, because the two columns are only
    meaningful as a PAIR — `benchmarks.is_driver_substituted` reads a driver
    substitution off their disagreement — and two hand-written copies could
    disagree by a swap that nothing else would catch.

    Empty when `detect_env` could not observe one. Never defaulted to the
    other: that would assert a native configuration nobody saw.
    """
    row["kmd_version"] = env.kmd_version or ""
    row["umd_version"] = env.umd_version or ""


def build_refusal_row(
    inv: Invocation,
    provider: Provider,
    env: Env,
    *,
    outcome: str,
    date: str,
    contributor: str,
    impl_id: str,
    framework_version: str,
    run_id: str = "",
    box: str = "",
    detail: str = "",
    rel_err: Optional[float] = None,
    cosine: Optional[float] = None,
    xcheck_typical: Optional[float] = None,
    xcheck_worst: Optional[float] = None,
) -> Dict[str, str]:
    """A row for a cell this implementation did not produce a measurement
    of: the card refused it (`oom`), or the attempt failed (`mismatch`,
    `error`, `crash`, `timeout` — `benchmarks.OUTCOMES`), with `detail` as
    the row's `outcome_detail`. A `mismatch` carries the comparison's
    numbers it was refused on (the finite ones; an output of the wrong shape
    has none).

    Same identity and the same provenance as a measurement — the cell, the
    implementation, the GPU, the box, the day — and NO measurement: the
    schema allows the numbers empty exactly when `outcome` says there are
    none. That is what makes "this does not fit" a fact a later pass can
    read instead of a failure it rediscovers; the cell OOMs again at the
    same shape on the same card however many times it is tried.

    Not a latency of 0, and not an absent row: zero would price the cell as
    free in every consumer that reads latencies, and absence is
    indistinguishable from "nobody has measured this yet" — which is the
    distinction the sweep keeps paying for.
    """
    row = {c: "" for c in _benchmarks.ALL_COLUMNS}
    row["model"] = inv.model
    row["spec_hash"] = inv.spec_hash
    row["phase"] = inv.phase
    row["parallel"] = inv.parallel
    row["framework"] = provider.library
    # The caller's, not recomputed: the driver already resolved both when it
    # decided to run this (provider, cell), and asking the provider again
    # inside an `except` is how a row that merely records a refusal came to
    # be able to raise and take the rest of the sweep with it.
    row["framework_version"] = framework_version or "unknown"
    row["gpu"] = env.gpu_sku or (env.device_name or "unknown")
    row["op"] = inv.op
    row["node_spec_id"] = inv.node_spec_id
    row["moe_block_member"] = "1" if inv.moe_block_member else ""
    row["shape_json"] = _cell_text(inv)
    row["dtype_env_json"] = json.dumps(inv.dtype_env, separators=(",", ":"),
                                       sort_keys=True)
    row["kernel_impl"] = impl_id
    row["outcome"] = outcome
    row["outcome_detail"] = one_line(detail)
    row["schema_version"] = str(_benchmarks.KNOWN_SCHEMA_VERSION)
    row["contributor"] = contributor
    row["date"] = date
    row["sm"] = env.sm or ""
    # NOT "disabled" — that value means the operator passed
    # `--no-cross-check`, and claiming it here would describe a decision
    # nobody made. `cross_check` says how a MEASUREMENT was vetted and this
    # row has none: for an OOM / error / crash / timeout nothing was
    # checked, and a mismatch's verdict is its outcome, its basis the
    # numbers below.
    row["cross_check"] = "unknown"
    for column, value, fmt in (("cross_check_rel_err", rel_err, "{:.6g}"),
                               ("cross_check_cosine", cosine, "{:.6f}"),
                               ("xcheck_typical", xcheck_typical, "{:.6g}"),
                               ("xcheck_worst", xcheck_worst, "{:.6g}")):
        if value is not None and math.isfinite(value):
            row[column] = fmt.format(value)
    # What the card had free when it refused is NOT recorded: there is no
    # column for it, and borrowing one that means something else (the timed
    # call, the routing's load factor) would make the row lie in a second
    # place. It goes to the run's log instead, and the column documentation
    # says plainly that the row is a dated observation rather than a law —
    # a free-memory provenance column is the follow-up that would let a
    # reader tell "cannot fit" from "did not fit that time".
    row["run_id"] = run_id
    row["box"] = box
    row["input_seed"] = str(inv.input_seed)
    _stamp_driver(row, env)
    return row


def one_line(text: str) -> str:
    """`text` as an `outcome_detail`: one line, cut to the schema's bound."""
    flat = " ".join(str(text or "").split())
    limit = _benchmarks.OUTCOME_DETAIL_MAX
    return flat if len(flat) <= limit else flat[:limit - 1] + "…"



def _cell_text(inv: "Invocation") -> str:
    """The cell this launch measured, as the row states it.

    The dims, and -- for a collective group that reached past one chassis
    -- where it reached (`comm_domain`). The reach is not a dim, so it
    does not live in `inv.dims`; it joins here, at the one place the row's
    cell is written, so the shape on disk and the key coverage asks about
    cannot say different things about one measurement.

    Intra-host writes nothing, which is what keeps every row measured
    before the axis existed byte-identical.
    """
    dims = dict(inv.dims)
    reach = getattr(inv, "comm_domain", 0) or 0
    if int(reach) > 1 and "n_comm_ranks" in dims:
        from backend.inference.op_signatures import COMM_DOMAIN_AXIS
        dims[COMM_DOMAIN_AXIS] = int(reach)
    return json.dumps(dims, separators=(",", ":"), sort_keys=True)


def build_row(
    inv: Invocation,
    provider: Provider,
    stats: LatencyStats,
    env: Env,
    *,
    date: str,
    contributor: str,
    case: Optional[Any] = None,
    cross_check: str = "unknown",
    rel_err: Optional[float] = None,
    cosine: Optional[float] = None,
    xcheck_typical: Optional[float] = None,
    xcheck_worst: Optional[float] = None,
    kernel_symbol: str = "",
    launch_trace: Optional[Any] = None,
    run_id: str = "",
    box: str = "",
) -> Dict[str, str]:
    achieved_tflops, achieved_bw = achieved_from_latency(
        inv.flops_per_rank, inv.access_bytes_per_rank, stats.p50_us)

    row = {c: "" for c in _benchmarks.ALL_COLUMNS}
    # identity
    row["model"] = inv.model
    row["spec_hash"] = inv.spec_hash
    row["phase"] = inv.phase
    row["parallel"] = inv.parallel
    row["framework"] = provider.library
    framework_version = provider.version(env)
    if not framework_version:
        raise ValueError(
            f"{provider.library}: no version could be determined for this "
            "measurement; a corpus row without its provider version has half "
            "an identity and is not written")
    row["framework_version"] = framework_version
    row["gpu"] = env.gpu_sku or (env.device_name or "unknown")
    row["op"] = inv.op
    row["node_spec_id"] = inv.node_spec_id
    # A boundary/expert-block flag rides the row as "1"/"" so the block
    # aggregation reads it without re-deriving from op names.
    row["moe_block_member"] = "1" if inv.moe_block_member else ""
    row["shape_json"] = _cell_text(inv)
    row["dtype_env_json"] = json.dumps(inv.dtype_env, separators=(",", ":"), sort_keys=True)
    # What the kernel COMPUTED in, when that is not what the cell asks for
    # (`Provider.computes_at`): a load-time dequantising path runs bf16 on an
    # fp8 checkpoint, and the latency is a bf16 latency. Provenance beside the
    # cell's own precision, never in place of it — the cell is the identity.
    computes_at = provider.actual_computes_at(inv)
    if computes_at:
        row["measured_dtype_env_json"] = json.dumps(
            dict(computes_at), separators=(",", ":"), sort_keys=True)
    # The call that was timed. Two observers, and the row prefers the one
    # that can name the kernel: a provider returning a `Launch` says which
    # library function it called (including one behind a pybind entry point,
    # where the dispatcher sees nothing), while the dispatch trace says what
    # aten actually performed and with which operands — for every other
    # provider, with nothing declared (`launch_trace`).
    if case is not None and isinstance(getattr(case, "run", None), Launch):
        try:
            row["launch_signature"] = case.run.describe()
        except Exception:
            # The same policy `kernel_symbol` already states: this column is
            # evidence ABOUT the measurement, not the measurement, so a
            # renderer that trips leaves it empty rather than discarding a
            # row that was measured and checked — and blaming the provider
            # for it, which is what the driver's failure path would read as.
            row["launch_signature"] = ""
    elif launch_trace is not None:
        try:
            row["launch_signature"] = launch_trace.describe()
        except Exception:
            row["launch_signature"] = ""
    # A resolved kernel parameter is part of WHICH implementation ran, so the
    # recorded name is `impl_id` — byte-identical to the value coverage was
    # asked about before this row was measured.
    row["kernel_impl"] = provider.impl_id(inv)
    # measurement
    row["latency_us"] = f"{stats.p50_us:.4f}"
    row["p50_us"] = f"{stats.p50_us:.4f}"
    row["p95_us"] = f"{stats.p95_us:.4f}"
    row["samples"] = str(stats.samples)
    row["timing_method"] = stats.timing_method
    row["count_per_step"] = str(inv.count_per_step)
    row["achieved_tflops"] = achieved_tflops
    row["achieved_bw_gbps"] = achieved_bw
    peak_dtype = computes_at.get("activation") or inv.dtype
    peak_tflops = env.peak_tflops(peak_dtype)
    row["peak_tflops_at_test"] = f"{peak_tflops:.4f}" if peak_tflops else ""
    row["peak_bw_at_test"] = f"{env.peak_bw_gbps:.4f}" if env.peak_bw_gbps else ""
    # provenance
    row["schema_version"] = str(_benchmarks.KNOWN_SCHEMA_VERSION)
    row["contributor"] = contributor
    row["date"] = date
    row["sm"] = env.sm or ""      # compute-capability tag, e.g. "sm_89" (Ada)
    # Whether an independent reference actually vetted this number. The caller
    # knows (it is the one that ran, skipped, or could not build the oracle);
    # defaulting here would let an unchecked row claim to be checked.
    row["cross_check"] = cross_check
    # The verdict's evidence, not just the verdict. A pass/fail alone cannot
    # tell "lossy but correct" from "computing the wrong thing", and cannot say
    # whether a tolerance is set anywhere near the dtype's own error — which is
    # what let the fp8 band sit below its floor, rejecting 45,190 correct
    # measurements while looking like a provider that simply had few rows.
    # `rel_err` is max|delta| / mean|ref| (large on a reduction by
    # construction); `cosine` is scale-free and is the one that stays near 1
    # when a kernel is right.
    #
    # NEITHER OF THESE TWO IS THE NUMBER THE VERDICT WAS MADE ON, and the
    # name `cross_check_rel_err` invites the opposite reading. The gate
    # (`driver._cross_check`) decides on `error = |delta| / (|ref| + mean|ref|)`
    # and asks two questions of it — `typical` (its RMS) and `worst` (its
    # max) — against the dtype's band, bf16 `Band(8e-3, 1e-1)`, both of which
    # must hold. That self-normalising denominator is the whole point: a pure
    # relative criterion has |err|/|ref| → ∞ as ref → 0 and would fail a
    # correct low-precision kernel on its near-zero entries.
    #
    # `rel_err` has NO `|ref_i|` in its denominator, so it is a strictly
    # harsher statistic on any output with wide dynamic range: a worst element
    # sitting where `|ref|` is large gives a small GATED value and a large
    # RECORDED one. A `rel_err` of 0.215 beside `cross_check='passed'` is
    # therefore ordinary, not a 21.5% disagreement that slipped through — and
    # reading it as the latter has now cost two people an hour each in one
    # night, both with this file open.
    #
    # THE DISTRIBUTION IS THE PROOF of that, and the reason this column must
    # not be read against the bands. Measured over the lake's parquet,
    # 8,132,269 rows with `cross_check='passed'`: 48% record `rel_err` above
    # 2e-2 and 20% above 1e-1, max 230.8. Those are not 3.9M disagreements —
    # they land exactly where `max|delta| / mean|ref|` must land, on wide
    # dynamic range, and nowhere else:
    #
    #     per_head_linear                 150,544 rows   89.5% over 0.1
    #     flash_dsa                        33,156         84.3%
    #     merged_column_parallel_linear   513,093         44.8%
    #     grouped_gemm                    267,970         33.8%
    #     linear                        1,391,379         33.3%
    #     gather / add / moe_topk_softmax 214,061          0.0%
    #
    # A statistic that fires on every reduction and never on a pointwise op
    # is behaving as constructed. The gate is sound; this column is simply
    # not the gate.
    #
    # Kept under this name because it is persisted in published rows, golden
    # files and purge records; a rename is a schema migration, not a comment.
    # Compare it against the bands at your peril — the bands are not measured
    # against it.
    row["cross_check_rel_err"] = "" if rel_err is None else f"{rel_err:.6g}"
    row["cross_check_cosine"] = "" if cosine is None else f"{cosine:.6f}"
    # THE TWO THE VERDICT WAS MADE ON (schema 13). Recorded so a row
    # carries the basis of its own agreement claim and its distance from
    # both bands, which the two above cannot give: before these, a row
    # held two numbers the gate never read and neither that it did, so
    # "passed" could not be audited from the row at all.
    row["xcheck_typical"] = ("" if xcheck_typical is None
                             else f"{xcheck_typical:.6g}")
    row["xcheck_worst"] = ("" if xcheck_worst is None
                           else f"{xcheck_worst:.6g}")
    # What the DEVICE ran — the only evidence separating "declared A, ran A"
    # from "declared A, ran B" when both give plausible numbers. Passed in by
    # the caller, which is the one that observed it; defaulting to anything
    # here would put a value in the column that nothing watched a GPU to get.
    row["kernel_symbol"] = kernel_symbol
    # Which sweep produced the row — contamination is found per batch, so
    # revocation works per run_id. Passed in by the caller, the one place the
    # batch boundary is known; `measured_load_factor` stays empty until a
    # measurement carries a real routing trace (declared, not yet produced).
    row["run_id"] = run_id
    row["box"] = box
    # The data this latency was measured on, as the seed it was drawn at
    # (`Invocation.input_seed`, a function of the cell's key). A rerun of the
    # same cell draws the same inputs, so a verdict that moves is the kernel
    # or the box, never the luck of the draw — and a reader can recompute the
    # seed from the key and tell whether a row predates the data it sees.
    row["input_seed"] = str(inv.input_seed)
    _stamp_driver(row, env)
    return row


def rows_to_csv(rows: list) -> str:
    return _benchmarks.emit_csv(list(_benchmarks.ALL_COLUMNS), rows)


def existing_run_keys(path: str) -> set:
    """Provider identities already persisted in one resumable output file.

    The question is "what has this file measured", not "is this file
    publishable", and the difference is not academic: this read used to go
    through `validate_text`, whose gate includes the CROSS-ROW rule that no
    two rows share an identity — so one duplicated row anywhere in a box's
    results stopped every env of every model from starting, because the
    queue hands each run every other model's results as `--measured-in`.
    Duplicates are nothing to this question (the keys are a set, which
    collapses them) and a row that fails its own schema check still says
    truthfully that its cell was measured. So the header is checked — a file
    in another schema cannot be read at all — and the rows are parsed, not
    gated. Publishability is asked where a file is published
    (`validate_file`, the upload path, CI).
    """
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return set()
    from backend.inference.op_calibration import BenchmarkKey

    with open(path, encoding="utf-8") as handle:
        header, rows = _benchmarks.parse_rows(handle.read())
    try:
        _benchmarks.validate_header(header)
    except _benchmarks.BenchmarkValidationError as error:
        raise _benchmarks.BenchmarkValidationError(f"{path}:1: {error}") from None
    keys = set()
    for line, row in enumerate(rows, start=2):
        key = BenchmarkKey.from_csv_row(row)
        try:
            physical = key.to_physical()
        except Exception as error:
            # A cell this build cannot name (an op or shape symbol the
            # registry has since dropped) — said with the line, because the
            # answer is to migrate the file, not to re-measure it.
            raise _benchmarks.BenchmarkValidationError(
                f"{path}:{line}: cannot read this row's cell: "
                f"{type(error).__name__}: {error}") from error
        keys.add((physical, key.framework,
                  key.framework_version, key.kernel_impl))
    return keys


def resumed_verdict(row: Dict[str, str], measured_keys) -> str:
    """Whether a run that is refusing a repeat had resumed KNOWING the cell.

    The one fact that separates the two things able to produce a repeat:
    either the resumed key set lacked the cell — so "measure once" never had
    a chance to skip it — or it held the cell and the measurement happened
    anyway, which puts the fault downstream of the check. Nothing else
    distinguishes them, and the duplicate rows that cost a box three outages
    have survived the elimination of every mechanism two sessions could
    propose, so the next occurrence should answer the question by itself
    rather than start another investigation.

    Keyed by the same function `existing_run_keys` keys with, so a mismatch
    here is a mismatch there.
    """
    from backend.inference.op_calibration import BenchmarkKey
    try:
        key = BenchmarkKey.from_csv_row(row)
        mine = (key.to_physical(), key.framework,
                key.framework_version, key.kernel_impl)
    except Exception as error:               # never the reason a row is lost
        return (f"could not key this row to ask whether the run resumed with "
                f"it ({type(error).__name__}: {error})")
    held = "HAS" if mine in set(measured_keys) else "does NOT have"
    return (f"the {len(set(measured_keys))} keys this run resumed with "
            f"{held} this cell")


def resume_from_output(path: str, *, skip_existing: bool,
                       append: bool) -> tuple[bool, set]:
    """Whether to append to `path`, and which identities it already holds.

    `--skip-existing` means "don't re-measure what we already have". The
    corpus check in `run_jobs` covers promoted rows; a private results CSV
    from a previous `run`/`run-all` is the other half of that contract.
    Opening that file in write mode (the default when `--append-output` was
    only set after a stall) truncated it — a 4090 resume wiped 22k fa2 rows
    in seconds. If we intend to skip what is already measured, we must
    both keep the file and load its keys.
    """
    if append or (skip_existing and os.path.isfile(path)
                  and os.path.getsize(path) > 0):
        return True, existing_run_keys(path)
    return False, set()


class IncrementalCsvWriter:
    """Append rows to the output CSV as they are measured, not at the end.

    A sweep is hours of GPU time and the rows were being buffered in memory
    until the run finished, so anything that ended the process early — a kill,
    an OOM, a dropped ssh session — discarded every measurement taken up to
    that point. One observed run held 20 hours of work and had written nothing.

    Each row is written and flushed as it arrives, so the file on disk is
    always the set of measurements completed so far. `csv.writer` shares the
    dialect `emit_csv` uses (comma, LF, minimal quoting) and the column order
    is `ALL_COLUMNS`, so the incremental file is byte-identical to the
    all-at-once text for the same rows.

    Rows are validated by the CALLER before they get here (per-row validation
    that drops a bad row instead of failing a whole sweep) — this class only
    writes.

    What it does own is the one rule that is about the FILE rather than about
    a row: no two rows the corpus cannot tell apart (`benchmarks.row_identity`
    — the 12 identity columns and the outcome). It has to be owned here,
    because the driver's "measure once" key is a different key by design (the
    library-agnostic physical cell, so another model's measurement counts as
    this model's), and two keys cannot both decide what one file holds: a
    pair of jobs the driver reads as different cells wrote two refusal rows
    with one identity, and since every box hands every other model's results
    to the next run as `--measured-in`, that single file then refused to be
    read and the H100 measured NOTHING for 35 minutes. This used to be
    checked after the run, on the file it had already written.
    """

    def __init__(self, path: str, *, append: bool = False):
        import csv
        self._path = path
        self._n = 0
        existing = append and os.path.isfile(path) and os.path.getsize(path) > 0
        self._has_existing_rows = False
        #: identity → the file line that holds it, so a repeat can say where.
        self._identities: Dict[tuple, int] = {}
        #: the last line number in the file — the header's, until rows land.
        self._line = 1
        if existing:
            self._bring_forward(path)
            with open(path, encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                for line, values in enumerate(reader, start=2):
                    self._has_existing_rows = True
                    self._line = line
                    self._identities.setdefault(
                        _benchmarks.row_identity(dict(zip(header, values, strict=True))), line)
        # Held open across append() calls; close() releases it.
        self._fh = open(path, "a" if existing else "w",  # noqa: SIM115
                        encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh, lineterminator="\n")
        if not existing:
            self._writer.writerow(list(_benchmarks.ALL_COLUMNS))
            self._fh.flush()

    @staticmethod
    def _bring_forward(path: str) -> None:
        """Rewrite an older-schema results file at the current schema before
        appending to it, by the one migration (`benchmark_migration`, the
        same fills `tools/migrate_schema.py` writes), so a schema bump does
        not stop every sweep whose results files predate it.

        Appending writes rows in `ALL_COLUMNS` order, so the file must end up
        with exactly that header. A current header in another order (legal
        to read) is rewritten in canonical order, not appended beneath. The
        source is checked structurally FIRST — no repeated column, no column
        the schema lacks (a newer harness's file, or not a results file),
        every row as wide as its header — and refused untouched otherwise:
        the migration maps rows by name, so a repeated column or a ragged row
        would lose a value silently. The bytes about to replace the file are
        checked the same way, then installed by renaming a temp file over it:
        a kill mid-rewrite leaves the old file whole (process-kill atomicity;
        nothing is fsync'd). Assumes one writer per results file, which
        `run-all` gives each library's file.
        """
        import csv
        import io
        from backend.inference import benchmark_migration as _migration
        expected = list(_benchmarks.ALL_COLUMNS)

        def refuse(why: str) -> ValueError:
            return ValueError(
                f"cannot append to {path}: header does not match schema ({why})")

        def parse(text: str):
            rows = [r for r in csv.reader(io.StringIO(text, newline="")) if r]
            return (rows[0] if rows else []), rows[1:]

        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
        header, body = parse(text)
        if header == expected:
            return
        repeated = sorted({c for c in header if header.count(c) > 1})
        if repeated:
            raise refuse(f"repeated columns: {repeated}")
        foreign = [c for c in header if c not in expected]
        if foreign:
            raise refuse(f"columns this schema does not have: {foreign}")
        ragged = [i for i, r in enumerate(body, start=2) if len(r) != len(header)]
        if ragged:
            raise refuse(f"rows not as wide as the header, lines {ragged[:5]}")
        if set(header) == set(expected):
            # Every column, another order: the same rows in canonical order.
            migrated = _benchmarks.emit_csv(
                expected, [dict(zip(header, r, strict=True)) for r in body])
        else:
            try:
                migrated = _migration.migrate_text(text)
            except ValueError as exc:
                raise refuse(f"cannot be migrated: {exc}") from exc
        out_header, out_body = parse(migrated)
        if out_header != expected or len(out_body) != len(body) or any(
                len(r) != len(expected) for r in out_body):
            raise refuse("the migration did not produce the current schema")
        tmp = f"{path}.migrating"
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as handle:
                handle.write(migrated)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @property
    def count(self) -> int:
        return self._n

    @property
    def has_rows(self) -> bool:
        return self._has_existing_rows or self._n > 0

    def write(self, row: Dict[str, str]) -> Optional[str]:
        """Persist `row`; returns why it was NOT written, or None.

        The one refusal is a row this file already holds under the same
        identity — reported to the caller the way the schema and SOL gates
        are, at the point of loss, because it means two cells the planner
        and the driver read as distinct arrive as one row: either the
        distinction is real and the schema is missing the column that
        carries it, or the dedup key missed a collapse. Writing both would
        make the file unreadable; dropping it silently would hide which.
        """
        ident = _benchmarks.row_identity(row)
        first = self._identities.get(ident)
        if first is not None:
            return (f"this file already holds this identity (line {first}); "
                    f"two rows whose 12 identity columns and `outcome` are "
                    f"equal are one row")
        self._writer.writerow([row[c] for c in _benchmarks.ALL_COLUMNS])
        self._fh.flush()
        self._n += 1
        self._line += 1
        self._identities[ident] = self._line
        return None

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "IncrementalCsvWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
