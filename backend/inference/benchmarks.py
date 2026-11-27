"""Benchmark CSV strict parser and validator.

Single executor of the schema defined at `backend/inference/BENCHMARKS_CSV.md`.
Validates measured-row CSVs. The CLI subcommand `huggingarch bench
validate <csv>`, the Kernel Bench upload and the lookup's gate all call into
here.

Pattern mirrors `backend/inference/fusion_plans/__init__.py`: strict
whitelisted column set, per-column typing, symbol / vocabulary restriction
on JSON columns, forward-incompatible `schema_version`, byte-identical
round-trip on resave.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import math
import re
from typing import Dict, FrozenSet, List, Optional, Tuple

from backend.arch.activation_formats import ACTIVATION_GRANULARITIES
from backend.arch.dtypes import dtype_bytes as _dtype_bytes, CORPUS_TOKENS
from backend.arch.eval.formats import WEIGHT_GRANULARITY
from backend.inference.benchmark_identity import (
    FORMAT_SLOTS, current_shape, declares_routed_token_total, format_refusal)
from backend.inference.op_signatures import all_signatures, op_bindings, op_signature
from backend.code_trace.memo import memoize


# 3 adds the `cross_check` provenance column; 4 adds the two numbers behind that
# verdict; 5 adds `kernel_symbol`; 6 adds `n_tokens` to grouped-GEMM identity.
# 9 adds `input_seed`, the seed the measured cell's inputs were drawn at.
# 10 adds `measured_dtype_env_json`: the precision the kernel computed at
# when it is not the cell's (load-time dequantisation). 11 adds
# `launch_signature`: the host-side call that was timed, with its operands.
# 12 adds `outcome`: what happened when the cell met the implementation,
# which is how an OOM becomes a recorded fact instead of a retried failure.
# 13 adds `xcheck_typical` and `xcheck_worst`, THE TWO NUMBERS THE
# CROSS-CHECK VERDICT IS MADE ON. Before them a row carried two numbers the
# gate never reads (`cross_check_rel_err`, `cross_check_cosine`) and neither
# that it does, so a `passed` could not be audited from the row and the
# rel-err column was twice mistaken for the margin.
# 14 adds `kmd_version` and `umd_version`, WHICH DRIVER RAN THIS ROW. Before
# them a box could measure one env through NVIDIA's forward-compatibility
# user-mode driver and another through its own, and no row said which — see
# the two columns for why that is not recoverable after the fact.
# 15 widens `outcome` from the two facts (`measured`, `oom`) to the four
# failures a sweep hits (`mismatch`, `error`, `crash`, `timeout`) and adds
# `outcome_detail`, the one line saying what went wrong. Before it a failure
# left no row at all, so "nobody has tried this" and "this was tried and
# broke" read the same, and the sweep paid to rediscover every failure.
# 16 adds `timing_method`, WHICH CLOCK TIMED THIS ROW (`TIMING_METHODS`).
# The clock changed on 2026-10-26 to bill a launch's own L2 write-back; the
# rows timed before it were kept, by the user's decision, and read empty.
# `schema_version` says WHICH SCHEMA THIS FILE NOW CONFORMS TO, never when
# the row was measured: `tools/migrate_schema.py` relabels an old row to the
# current version and fills the new columns empty (a finished CSV cannot be
# asked what an oracle vetted or what device ran it), so after a migration a
# row written months ago is indistinguishable by this column from one written
# today. Anything asking "was this row measured under the code that records
# X" must read `date` — reading the version instead reports every row as
# current, which is a confirmation-shaped answer and was briefly believed
# (`tools/launch_blind_spots.py`, which now filters on the date and says so).
# The header check is exact except for the columns a later version added
# (`OPTIONAL_COLUMNS`), which an older CSV may omit and which read as empty,
# so an older CSV still validates. What does NOT load before
# `tools/migrate_schema.py` rewrites it is a row whose IDENTITY changed
# meaning — the grouped GEMM's routed `n_tokens` (v6), gated in
# `validate_row`.
KNOWN_SCHEMA_VERSION = 16


IDENTITY_COLUMNS: Tuple[str, ...] = (
    "model",
    "spec_hash",
    "phase",
    "parallel",
    "framework",
    "framework_version",
    "gpu",
    "op",
    "node_spec_id",
    "shape_json",
    "dtype_env_json",
    "kernel_impl",
)

MEASUREMENT_COLUMNS: Tuple[str, ...] = (
    "latency_us",
    "p50_us",
    "p95_us",
    "samples",
    "count_per_step",
    "achieved_tflops",
    "achieved_bw_gbps",
    "peak_tflops_at_test",
    "peak_bw_at_test",
)

#: The two driver columns, as ONE declaration. Everything that needs the
#: pair spreads or iterates this: `PROVENANCE_COLUMNS` (where they sit in
#: the schema), `OPTIONAL_COLUMNS` (an older CSV omits them),
#: `validate_row`'s shape check, and `op_calibration.ROW_COLUMNS` (the wire
#: the lookup sends). They are only ever meaningful together — the whole
#: fact is their agreement — so the pair, not the two names, is the unit.
DRIVER_COLUMNS: Tuple[str, str] = ("kmd_version", "umd_version")


PROVENANCE_COLUMNS: Tuple[str, ...] = (
    "schema_version",
    "contributor",
    "date",
    # GPU compute capability tag (e.g. "sm_89" Ada / "sm_90" Hopper /
    # "sm_100" Blackwell). A per-card-architecture label recorded alongside
    # the `gpu` SKU — kernel availability / performance is SM-specific, so
    # rows are stamped with the arch they were measured on. Provenance (not
    # identity): `gpu` already keys the physical corpus; `sm` is the explicit
    # arch annotation, so adding it does not change the calibration match key.
    "sm",
    # Whether this latency was checked against the independent fp32 reference,
    # and if not, why not. A row that FAILED the check is never written as a
    # measurement — the driver records it as `outcome="mismatch"`, with no
    # latency — so on a measurement this does not separate right from wrong. It
    # separates CHECKED from UNCHECKED, which a consumer otherwise cannot tell:
    # the fp32 oracle costs ~2× the kernel's memory and is O(s_ctx²) for attention,
    # so at the biggest cells it OOMs and the kernel is measured with no oracle
    # at all. Those rows were indistinguishable from verified ones. Provenance,
    # not identity — one cell measured with and without an available reference
    # is one measurement, not two.
    "cross_check",
    # The cross-check's evidence. A verdict alone cannot distinguish a lossy
    # but correct kernel from one computing the wrong thing, nor show whether a
    # tolerance sits anywhere near the dtype's own error. Empty when no oracle
    # ran. `cross_check_rel_err` is max|delta| / mean|ref|, which runs large on
    # a reduction by construction; `cross_check_cosine` is scale-free and is
    # the number that stays near 1 when a kernel is right.
    #
    # BOTH ARE DIAGNOSTIC. Neither is what the verdict was made on, despite
    # the name: the gate decides on |delta| / (|ref| + mean|ref|), as its RMS
    # and its max against the dtype's band (`driver._cross_check`,
    # `driver._XCHECK_BANDS`). `rel_err` lacks the per-element `|ref_i|` in
    # its denominator, so it is harsher wherever the output has wide dynamic
    # range — a large `rel_err` beside `passed` is ordinary and is not a
    # disagreement that slipped through. Do not compare this column to the
    # bands; they are not measured against it.
    "cross_check_rel_err",
    "cross_check_cosine",
    # The verdict's BASIS: `error = |delta| / (|ref| + mean|ref|)` as its
    # RMS and its max, which `driver._cross_check` compares to the dtype's
    # band (both must hold). These are the numbers a `passed` was decided
    # by; the two above are diagnostics measured differently and are not.
    "xcheck_typical",
    "xcheck_worst",
    # What the DEVICE ran, as `<digest>:<name>`. The three checks before it
    # prove the declaration is well formed, uniquely named, and numerically
    # right; none can tell `declared A, ran A` from `declared A, ran B` when B
    # computes the same thing — FlashKDA takes over `chunk_kda` on SM90+ unless
    # `FLA_FLASH_KDA=0`, and vLLM's CustomOp runs its native torch path unless
    # `custom_ops=all`. The digest covers the whole launch sequence in order,
    # so a substitution anywhere moves it; the name is the kernel that spent
    # the most device time, so the row says whose kernel it was.
    #
    # Provenance, not identity: one cell measured on a box whose profiler could
    # not attach is the same measurement, and `PhysicalKey` must not split on
    # it. Empty means the observation was not taken, which is why it is empty
    # rather than a placeholder that would read as "nothing ran".
    "kernel_symbol",
    # Which sweep produced this row. Contamination is discovered after the
    # fact and arrives in batches (16,045 rows from one concurrent run), so
    # revocation works per batch: one run_id, one tombstone, history is not
    # rewritten. Empty on rows measured before the column existed.
    "run_id",
    # Which machine profile (`kernel_bench/envs/boxes/<name>.toml`) measured
    # the row. The shared envelope's meaning is "the best this SKU can do",
    # so box state (clocks / power cap / driver) is provenance for after-the-
    # fact attribution, never part of the key — min over boxes is exactly the
    # envelope. Empty when no profile was resolved.
    "box",
    # max/mean expert load of the routing distribution this row was measured
    # under, when the measurement carried a real routing trace. Declared ahead
    # of any producer: rows without a trace stay empty — no default is
    # invented, because a fabricated 1.0 would read as "provably balanced".
    "measured_load_factor",
    # Whether this row belongs to a MoE expert block — a dispatch-zone node or
    # its permute/combine boundary — so the block aggregation reads the fact
    # off the row instead of matching expert/moe name substrings. Optional for
    # backward compat: a schema-7 CSV predates the column and reads as empty
    # (that CSV simply yields no MoE-block rollup until re-emitted).
    "moe_block_member",
    # The seed this row's input tensors were drawn at
    # (`kernel_bench.invocation.Invocation.input_seed`): a decimal unsigned
    # 64-bit integer, stored as a string so no reader turns it signed or
    # floating. Derived from the four fields of the measurement key, so it is
    # recomputable from the row — which is what makes "was this measured on
    # the data today's harness builds?" a query rather than an excavation.
    # Provenance, not identity: the key already says which cell this is, and
    # the seed is a function of it. Empty on rows measured before the column
    # existed (their data was whatever the global RNG happened to hold).
    "input_seed",
    # The dtypes the KERNEL actually computed in, when they are not the ones
    # the cell asks for — a `dtype_env` of the same shape as `dtype_env_json`,
    # holding only the slots that differ. Empty means the kernel computed at
    # the cell's precision, which is the normal case.
    #
    # The motivating case is load-time dequantisation, which is not an
    # exception but the default for several real paths: vLLM and sglang
    # dequantise an absorbed MLA projection (and DeepSeek-V4.1's grouped
    # output LoRA) in `process_weights_after_loading` and then run bf16
    # `torch.bmm`. Such a row is a true measurement of a real deployment of
    # an fp8 checkpoint — it is what a server pays — but its arithmetic is
    # bf16, and the latency is not an fp8 kernel's. Without this column the
    # only hint was a naming convention in `kernel_impl`, which no consumer
    # can read reliably.
    #
    # Identity stays with the CELL (`dtype_env_json`): the cell is what the
    # checkpoint ships and what an estimate asks for, and two configurations
    # of one cell belong on one key so they can be compared. This column says
    # which configuration a row is.
    "measured_dtype_env_json",
    # What HAPPENED when this cell met this implementation. Empty or
    # `"measured"` is a measurement, which is every row written before this
    # column and the overwhelming majority after it. The rest of the closed
    # vocabulary (`OUTCOMES`) is what took the measurement's place:
    #
    #   * `"oom"` — the card could not hold this shape for this
    #     implementation;
    #   * `"mismatch"` — it ran, and the cross-check (or the launch gate)
    #     said it computed something other than the cell;
    #   * `"error"` — the provider's build or launch raised, including a
    #     runtime "unsupported", which means its declaration missed a case;
    #   * `"crash"` — the process died, or the CUDA context was lost, while
    #     this implementation ran;
    #   * `"timeout"` — the watchdog saw no progress, or one launch cost more
    #     than the timing budget.
    #
    # Every one of them, the measurement included, says what happened ON
    # THIS IMPLEMENTATION: `framework`, `framework_version`, `kernel_impl`
    # are identity, so a new library version is a new implementation and
    # retries the cell. What the row cannot see is the provider's own bench
    # code changing at the same library version — and that holds for a
    # measurement exactly as for a failure, so it is handled once for both:
    # the change that fixes a provider's build or input conversion purges
    # that `kernel_impl`'s old rows, all outcomes, through the existing
    # revocation channel (`kernel_bench/ARCHITECTURE.md`).
    #
    # Every value other than `"measured"` carries NO latency: the measurement
    # columns are empty, and `validate_row` allows them empty only there.
    # Read by the harness's coverage question (a cell × implementation with
    # any row at that version has been tried, so a sweep does not rediscover
    # an OOM or a failure every pass) and NOT by the estimate, which indexes
    # rows for their latency and skips any row that has none
    # (`op_calibration.registry_from_lookup`).
    "outcome",
    # What went wrong, in one line, on a row that is not a measurement:
    # the exception (`RuntimeError: …`), the cross-check's verdict, the
    # watchdog's reason. Required there, optional on `"oom"`, and empty on a
    # measurement — a measurement has nothing to explain. Its own column
    # because no other can carry it without lying: `launch_signature` is the
    # call that was timed, `kernel_symbol` what the device ran, `cross_check`
    # a closed vocabulary about vetting, `run_id` / `box` provenance others
    # join on. Free text and never parsed: the outcome is the class, this is
    # for the human who reads the row.
    "outcome_detail",
    # The host-side call that was timed, rendered from the arguments it was
    # made with: `module.fn(<dtype>[<shape>], …, kw=…)`. Empty when the
    # provider still returns a bare closure.
    #
    # The row already says which cell (`shape_json` / `dtype_env_json`) and
    # which kernel sequence the DEVICE ran (`kernel_symbol`); neither shows
    # what the host handed the library, and that is where a measurement
    # silently stops being the cell's. The case that motivated it:
    # `moe_permute`'s cell stated a dispatch reading token-ordered rows and
    # writing an expanded expert-ordered buffer, while the timed call was
    # `x[idx]` with both operands the same length — 1.5x less traffic than
    # the row's numerator counted, which is why those rows read at 110% of
    # HBM peak (fixed 10-24: the cell names both sides of the dispatch). A
    # rendered call makes such a gap visible on the row instead of only in
    # provider source.
    #
    # Provenance, not identity: one cell called through two libraries' own
    # conventions is still one cell.
    "launch_signature",
    # WHICH DRIVER RAN THIS ROW, as the two version strings whose agreement is
    # the question: the box's KERNEL-mode driver, and the USER-mode driver
    # (`libcuda.so`) the measuring process actually had mapped.
    #
    # They are normally the same string, because a process normally loads the
    # driver its kernel module came with. They differ whenever something else
    # is ahead of it on `LD_LIBRARY_PATH`. What the pair then establishes is
    # exactly one thing: the driver that ran this row was NOT the box's own.
    #
    # It does not establish the DIRECTION (a newer overlay is NVIDIA forward
    # compatibility; an older or stale one is a substitution too, and is not
    # "forward" anything), and it does not establish a SUPPORT verdict, which
    # is a property of the specific pairing rather than of inequality. The
    # motivating instance, where both happen to be known:
    # `kernel_bench/envs/boxes/h100-sxm-cu128.toml` puts a NEWER user-mode
    # driver on the path for three of its envs and deliberately not the rest,
    # so a CUDA-13 wheel runs on a CUDA-12 kernel driver with no reboot — and
    # that profile states that THAT configuration is outside NVIDIA's support
    # matrix, because the `cuda-compat-13-x` family's documented base install
    # is r580 and that box's kernel driver is r575. Another box's
    # substitution can sit inside the matrix; the row cannot tell, carries no
    # verdict, and nothing reading it should invent one.
    #
    # So one box measures some envs under a substituted user-mode driver and
    # others natively, and before these columns nothing on a row said which.
    # That is not recoverable afterwards: `box` names a PROFILE, which two
    # machines share and which is edited over time, and `date` is not a
    # driver. A reader comparing a new number against an old one could not see
    # that the two had different drivers under them.
    #
    # TWO facts and not three. "Was a compat driver in use" is not a third
    # column, because it is `umd_version != kmd_version` — read it through
    # `is_driver_substituted`, the one place that comparison is made. Nor is there
    # a list of which libraries get the overlay: the box profile's
    # `[library_env]` already declares that, and a module-level restatement
    # here would be a second table that drifts (CLAUDE.md §1.4). The row
    # records what was OBSERVED, so an `LD_LIBRARY_PATH` that did not take
    # effect reports the system driver, correctly.
    #
    # Provenance, not identity. The `box` column beside this one already
    # settled the class — "box state (clocks / power cap / driver) is
    # provenance for after-the-fact attribution, never part of the key" — and
    # the driver is named in it. Keyed, these would also split the corpus
    # permanently rather than supersede: every row written before v14 has them
    # EMPTY, and empty is unknown, so an unknown-driver key would never again
    # meet a known-driver one. Surfaced instead on the measured badge, so the
    # number on screen says which driver produced it.
    #
    # Empty means unknown and is never filled in by inference: rows measured
    # before v14 carry no driver, and writing a presumed one into them would
    # invent the fact the pair exists to check.
    #
    # Spread from `DRIVER_COLUMNS` above rather than spelled again: the pair
    # is one declaration, and three executable spellings of it (here, the
    # optional set, the validator's loop) is the §1.4 name-list pattern this
    # very change exists to argue against.
    *DRIVER_COLUMNS,
    # WHICH CLOCK TIMED THIS ROW, one of `TIMING_METHODS`, written by the
    # clock itself (`kernel_bench.timing.time_callable` returns it with the
    # latency). Empty on a measurement means it was timed BEFORE the clock
    # recorded itself — the old clock, whose L2 flush was a memset: a launch
    # paid a flush line's write-back for every line it touched and nothing
    # for its own writes, so a small kernel's row can be off, either way, by
    # up to one L2's write-back (L2 / DRAM bandwidth: ~17 us H100, ~82 us
    # RTX 4090). Those rows were kept rather than re-measured, by the user's
    # decision (`kernel_bench/ARCHITECTURE.md`, "How a launch is timed"); the
    # Kernel Bench page marks them, and nothing filters on it. Empty on a row
    # that is not a measurement: nothing was timed.
    #
    # Provenance, not identity: keyed, the kept rows would never again meet
    # a new row of the same cell.
    "timing_method",
)

ALL_COLUMNS: Tuple[str, ...] = (
    IDENTITY_COLUMNS + MEASUREMENT_COLUMNS + PROVENANCE_COLUMNS
)

#: Columns an OLDER CSV may omit, each added by a schema bump after the file
#: was written; a current emit always writes all of them. Read as empty when
#: absent, which is what each one's documentation says absence means.
#:
#: One declaration, at module level, because two things need it and used to
#: have no way to agree: `validate_header` exempts these from the strict
#: header, and the migration (`tools/migrate_schema.py`) must declare a fill for exactly
#: these — a column added here with no fill there leaves the migration tool
#: refusing every older CSV, which is how v13 shipped (`xcheck_typical` /
#: `xcheck_worst` had none, and the set was a local inside `validate_header`
#: so nothing could compare the two). The fill lives in
#: `benchmark_migration.FILL`, which both the CLI and the harness's append
#: path (`kernel_bench.emit.IncrementalCsvWriter`) run — so a results file a
#: box wrote before the bump is brought forward, not refused (v16 stopped
#: every sweep on exactly that).
OPTIONAL_COLUMNS: FrozenSet[str] = frozenset({
    "moe_block_member", "input_seed", "measured_dtype_env_json",
    "launch_signature", "outcome", "xcheck_typical", "xcheck_worst",
    *DRIVER_COLUMNS, "outcome_detail", "timing_method",
})

#: The clocks a measurement can carry in `timing_method`, each a branch of
#: `kernel_bench.timing.time_callable`; the one definition, which
#: `validate_row` holds every row to.
TIMING_METHODS: Dict[str, str] = {
    # GPU, the default: a read-only sweep of 2x L2 before the window, and a
    # second one inside it, after the launch, that drains the launch's own
    # dirty lines; the sweep's own cost is timed alone and subtracted.
    "l2_drain": "L2 swept clean before the launch; its own write-back drained inside the window",
    # The same drain on a device with no spin primitive (Ascend: the Sleep
    # operator is not built for the ascend910_93 SoC), where the sweep's own
    # device time covers the host launch path instead — MEASURED per run,
    # not assumed. A separate name because the guard differs: a reader
    # comparing an Ascend row with an NVIDIA one must be able to see which
    # kept the queue full.
    "l2_drain_sweep_hold": "L2 drained; the sweep itself held the queue (no spin on this device)",
    # The drain ran but nothing covered the host launch path: the sweep was
    # shorter than the time the CPU needs to queue the call, so the window
    # may bill launch overhead as kernel time. Published rather than
    # silently dropped, because the number is still an upper bound.
    "l2_drain_unheld": "L2 drained, queue NOT held: the window may include host launch overhead",
    # GPU with `flush_l2=False`: inputs may still be in L2 from the last call.
    "warm_l2": "no L2 flush; the launch may read what the previous one left in cache",
    # No GPU: `perf_counter` around the call.
    "wall_clock": "host wall clock, no GPU",
}

PHASE_VOCABULARY: FrozenSet[str] = frozenset({"prefill", "decode", "unified"})

# Each value names one path through the driver, so the vocabulary is a list of
# what to FIX rather than a shrug: `no_reference` wants a refs entry or a
# memory-lean oracle, `not_comparable` wants a `to_canonical` on the provider,
# `disabled` is the operator's own choice, and `unknown` marks a row measured
# before this column existed — which is not the same claim as "unchecked", and
# must not be written as one.
CROSS_CHECK_VOCABULARY: FrozenSet[str] = frozenset({
    "passed", "no_reference", "not_comparable", "disabled", "unknown",
})

DTYPE_SLOTS: FrozenSet[str] = frozenset({
    "weight", "activation", "kv", "output", "metadata",
})


# Every token the corpus accepts is a width the estimator can price: the
# vocabulary is the schema's, the bytes are `arch.dtypes`', and a token added
# here without a row there fails at import instead of in a calibration join.
for _token in CORPUS_TOKENS:
    _dtype_bytes(_token)
del _token

# Phase-binding symbols and the engine-supplied dims now live in the glossary's
# own sections, so `_load_glossary_symbols` already collects them — there is one
# namespace, not three. The names are kept as aliases because the validator's
# error message names each group separately (a contributor who wrote `s_cache`
# into a shape wants to be told it is a capacity-model quantity, not simply
# "unknown").
PHASE_BINDING_EXTRAS: FrozenSet[str] = frozenset({
    "s_query", "s_kv", "s_cache", "n_verify", "s_past", "s_pairs",
})

ENGINE_DIMS: FrozenSet[str] = frozenset({"B", "s_ctx"})

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_PARALLEL_RE = re.compile(r"^[a-z][a-z0-9_]*=\d+(,[a-z][a-z0-9_]*=\d+)*$")

def _registry_symbols() -> FrozenSet[str]:
    """Every symbol the dim registry names.

    A `shape_json` key is a dim of the tensor a kernel was measured on, so
    the names it may use are the names a spec may use — `backend/arch/
    dims.yaml`, read directly. This used to re-extract them by regex from
    the prose glossary's first column, which made the CSV schema depend on
    a markdown table's shape: `Sar` was documented on a shared row and the
    parser's earlier version took only the first token, so it stayed
    unregistered while `Kar` beside it went through.
    """
    from backend.arch import dims
    return frozenset(dims.base_symbols())


REGISTERED_SYMBOLS: FrozenSet[str] = _registry_symbols()


def _signature_axes() -> FrozenSet[str]:
    """Every axis a measurement signature declares for its cell
    (`OpSignature.cell_dims`, `variant_dims`, `output_axes`).

    A `shape_json` is a signature's cell, so the axes the signatures declare
    are names it may use whether or not a model spec may bind them: the
    Mamba-2 scan chunk (`ssm_chunk_size`) is the serving framework's
    (`Program.scan_chunk_size`) and is not in `dims.yaml`, which is
    what keeps a spec from binding it.
    """
    return frozenset(axis for sig in all_signatures()
                     for axis in (*sig.cell_dims, *sig.variant_dims,
                                  *sig.output_axes))


SIGNATURE_AXES: FrozenSet[str] = _signature_axes()


def _allowed_shape_keys() -> FrozenSet[str]:
    return REGISTERED_SYMBOLS | SIGNATURE_AXES | PHASE_BINDING_EXTRAS | ENGINE_DIMS


class BenchmarkValidationError(ValueError):
    """Raised when a CSV row or header fails strict schema validation."""


def _check_int(value: str, *, column: str, positive: bool = True) -> int:
    if value is None or value == "":
        raise BenchmarkValidationError(
            f"column `{column}` is required (got empty string)."
        )
    try:
        n = int(value)
    except ValueError as e:
        raise BenchmarkValidationError(
            f"column `{column}` must be int, got {value!r}."
        ) from e
    if positive and n <= 0:
        raise BenchmarkValidationError(
            f"column `{column}` must be positive int, got {n}."
        )
    return n


def _check_float(value: str, *, column: str, positive: bool = True) -> float:
    if value is None or value == "":
        raise BenchmarkValidationError(
            f"column `{column}` is required (got empty string)."
        )
    try:
        x = float(value)
    except ValueError as e:
        raise BenchmarkValidationError(
            f"column `{column}` must be float, got {value!r}."
        ) from e
    if not math.isfinite(x):
        raise BenchmarkValidationError(
            f"column `{column}` must be a finite number, got {value!r}."
        )
    if positive and not (x > 0):
        raise BenchmarkValidationError(
            f"column `{column}` must be > 0, got {x}."
        )
    return x


def _check_nonempty_str(value: str, *, column: str) -> str:
    if value is None or value == "":
        raise BenchmarkValidationError(
            f"column `{column}` is required (got empty string)."
        )
    return value


def _check_op(value: str) -> str:
    """The `op` column against the only vocabulary that can produce it.

    One owner decides a row's `op`: `RowIdentity.op_of`, which the three
    producers (the kernel_bench plan, the `suggest-shapes` template, the
    serving overlay's own lookup) all go through. Everything it can return
    is a name the kernel catalog binds — a passthrough of the engine's op,
    or a `resolve_op` rewrite, and `_rewrite` already refuses any target
    outside `op_bindings()`.

    So a stored row whose `op` the catalog does not bind came from no
    producer this repo has, and no lookup can ever reach it: `op_signature`
    hands the name back unchanged, `get_signature` answers None, the shape
    never projects, and the physical key names a kernel no provider serves.
    It validates, it loads, and it sits in the lake forever — and rows only
    accumulate, so one such spelling poisons the corpus permanently.

    This gate is why there is no list of accepted names here. The check that
    used to stand in for it was `op in {"grouped_gemm", "grouped_gemm_moe"}`
    below, and that list had already drifted into exactly this failure:
    `grouped_gemm_moe` is a name in the FUSION-PLAN vocabulary
    (`fusion_plans/grouped_gemm_moe.yaml`, the `fused_op` of four
    `providers/gemm.py` entries) and not a measurement op at all, so the
    validator admitted a second spelling of the grouped expert GEMM that
    `RowIdentity` can never produce and no estimate can ever look up.
    """
    op = _check_nonempty_str(value, column="op")
    if op not in op_bindings():
        raise BenchmarkValidationError(
            f"column `op` has {op!r}, which no signature in the kernel "
            f"catalog binds. The measurement key's op vocabulary is "
            f"`binds_ops` in backend/inference/op_signatures_catalog.py, "
            f"and it is the whole of what `RowIdentity.op_of` can produce; "
            f"a row spelled any other way is unreachable by every lookup. "
            f"If this is a kernel the corpus should hold, bind its op on a "
            f"signature there — do not widen this gate. Note a fused_op "
            f"from a fusion plan is not a measurement op: those spans "
            f"resolve through the same table."
        )
    return op


def _check_phase(value: str) -> str:
    v = _check_nonempty_str(value, column="phase")
    if v not in PHASE_VOCABULARY:
        raise BenchmarkValidationError(
            f"column `phase` must be one of "
            f"{sorted(PHASE_VOCABULARY)!r}, got {v!r}."
        )
    return v


#: What a row's `outcome` may say, measurement first (the column's
#: documentation says what each means). Empty reads as `"measured"`: every
#: row written before the column existed is one, and the vocabulary is closed
#: so a typo is a refusal rather than a row that quietly stops being read as
#: a measurement. The one declaration of the vocabulary; everything else asks
#: it (`outcome_of`, `row_status`, the service's outcome filter).
OUTCOMES: Tuple[str, ...] = ("measured", "oom", "mismatch", "error", "crash",
                             "timeout")
OUTCOME_VOCABULARY: FrozenSet[str] = frozenset(OUTCOMES)

#: The longest `outcome_detail` a row may carry. One line for a human, not a
#: traceback: the run's log keeps the rest.
OUTCOME_DETAIL_MAX = 240


def outcome_of(row: Dict[str, str]) -> str:
    """The row's outcome, empty read as `"measured"` (not validated: a
    reader of a row that passed `validate_row`)."""
    return (row.get("outcome") or "").strip() or "measured"


def row_status(row: Dict[str, str]) -> str:
    """What one row says about its cell × implementation, in the words a
    coverage report counts: `passed` / `unverified` for a measurement (its
    `cross_check` did / did not vet it), else the outcome itself. The two
    statuses no row can carry — `not_supported` (the provider's own
    declaration declines) and `pending` (no row) — are the report's, read
    off the declarations and the absence of rows (`hbench report`)."""
    outcome = outcome_of(row)
    if outcome != "measured":
        return outcome
    return ("passed" if (row.get("cross_check") or "").strip() == "passed"
            else "unverified")


#: Every status `row_status` can answer, measurements first: the row half of
#: a coverage report's classes, spelled from the two vocabularies it reads.
ROW_STATUSES: Tuple[str, ...] = (
    "passed", "unverified", *(o for o in OUTCOMES if o != "measured"))


def is_measurement(row: Dict[str, str]) -> bool:
    """Whether this row carries a measurement at all.

    ONE place, asked by every consumer: the estimate's registry, the MoE
    block rollup, the service's envelopes and counts, `hbench report`. A row
    recording an outcome (`outcome="oom"`) has no latency, and a reader that
    does not ask ends up either crashing on `float("")` or — worse —
    charging a cell's FLOPs against zero time.

    Empty reads as a measurement: every row written before the column has
    numbers in it, because the validator required them.
    """
    return outcome_of(row) == "measured"


#: The shape of an NVIDIA driver version, as a pattern FRAGMENT so a producer
#: can compose it into a larger match (`kernel_bench/env.py` reads both
#: drivers out of `/proc` with it). The schema owns this grammar and is the
#: only place that states it: a second copy in the producer would be free to
#: drift from the column it has to satisfy.
#:
#: A leading component of THREE or more digits is the whole point. NVIDIA
#: driver branches are three-digit and have been for the entire CUDA era
#: (470, 535, 550, 575, 580, 595), while a CUDA version is two (`12.9`,
#: `13.0`) — and those are precisely the two things that get confused here,
#: because a CUDA runtime version is the wrong-but-plausible value to write
#: into a driver column. `^\d+(\.\d+)+$` accepted `13.0`, so the contract
#: these comments stated was false: a row with `umd_version="13.0"` beside a
#: real `kmd_version` validated and then read as a substituted driver.
DRIVER_VERSION_PATTERN = r"\d{3,}(?:\.\d+)+"

#: The same grammar, anchored, for validating a whole cell.
DRIVER_VERSION_RE = re.compile(rf"^{DRIVER_VERSION_PATTERN}$")

def is_driver_substituted(row: Dict[str, str]) -> Optional[bool]:
    """Whether the user-mode driver that ran this row was NOT the box's own
    kernel-mode driver.

    That is the whole claim, and it is exactly what the two observations
    entail. It deliberately does NOT say:

      * which DIRECTION the substitution went. A newer user-mode driver ahead
        of the system one is NVIDIA forward compatibility; an older or stale
        overlay is also a substitution and is not "forward" anything. The
        pair cannot tell them apart without ordering two vendor version
        strings, and nothing here needs the distinction — what a reader needs
        to know is that the number did not run on the box's own driver.
      * whether the configuration is SUPPORTED. That is a fact about a
        specific pairing, not about inequality: NVIDIA documents a base-
        install floor of r580 for the `cuda-compat-13-x` family, and the H100
        box in `kernel_bench/envs/boxes/h100-sxm-cu128.toml` is r575, so THAT
        configuration is outside the matrix. Another box's substitution can
        sit comfortably inside it. A row carries no support verdict and no
        `box` (`op_calibration.ROW_COLUMNS` drops it on purpose), so no
        consumer of this function is in a position to assert one.

    THREE-VALUED, and the third value is the point:

      * `False` — both drivers known and the same string. The ordinary case.
      * `True`  — both known and different.
      * `None`  — either side unknown, which is every row written before
        schema 14. NOT `False`: a missing driver is not evidence of a native
        one, and answering `False` there would state about 295k existing rows
        exactly the thing nobody observed.

    ONE place, so no consumer re-derives it — the badge, a report and a future
    gate all ask here. A caller that wants to branch must handle `None`;
    `if is_driver_substituted(row):` alone reads unknown as fine, which for a
    warning is the safe direction and for a refusal is not.
    """
    kmd = (row.get("kmd_version") or "").strip()
    umd = (row.get("umd_version") or "").strip()
    if not kmd or not umd:
        return None
    return kmd != umd


def _check_driver_version(value: str, *, column: str) -> None:
    """Empty, or a driver-version-shaped string.

    Empty is legitimate and permanent for rows measured before v14, and is
    also the honest answer for an env with no NVIDIA driver mapped at all —
    so presence is not required, only shape.

    What the shape rule buys and what it does not: it rejects a CUDA version
    (`13.0`) and an `sm_90`-style tag, which are the values that get written
    into these columns by mistake and which would then be compared against a
    real driver by `is_driver_substituted`. It does NOT verify that the value
    is a driver that exists — nothing here can — so a well-formed wrong
    number still gets through, and the producer reading `/proc` is what makes
    the value true.
    """
    text = (value or "").strip()
    if not text:
        return
    if not DRIVER_VERSION_RE.match(text):
        raise BenchmarkValidationError(
            f"column `{column}` must be an NVIDIA driver version as "
            f"`nvidia-smi` prints it — a three-or-more-digit branch then one "
            f"or more dotted components (`575.57.08`, `595.84`) — got "
            f"{text!r}. A CUDA version (`13.0`) is two digits and is "
            f"rejected here on purpose: the two driver columns are compared "
            f"to each other, so a runtime version in one of them would read "
            f"as a substituted driver."
        )


def _check_outcome(value: str) -> str:
    """The row's outcome, defaulted to `"measured"` for a row that predates
    the column — not guessed: a row with numbers in it IS a measurement, and
    every row before v12 has them (the validator required them)."""
    text = (value or "").strip() or "measured"
    if text not in OUTCOME_VOCABULARY:
        raise BenchmarkValidationError(
            f"column `outcome` must be one of {sorted(OUTCOME_VOCABULARY)}, "
            f"got {value!r}.")
    return text


#: The columns a comparison against the oracle fills.
_COMPARISON_COLUMNS: Tuple[str, ...] = (
    "cross_check_rel_err", "cross_check_cosine", "xcheck_typical", "xcheck_worst")


def _check_outcome_evidence(row: Dict[str, str], outcome: str) -> None:
    """What a row with this outcome must and may say beside it.

    * `outcome_detail`: empty on a measurement (nothing to explain), and
      required on every other row (the class without the reason is a row
      nobody can act on) EXCEPT an `oom`: OOM rows were written for three
      versions before the column existed, a finished row cannot be asked
      what the card had free, and requiring it would refuse every one of
      them — in the lookup, as an invalid shared row. The driver writes one
      on every new OOM. One line, at most `OUTCOME_DETAIL_MAX` characters.
    * The comparison's numbers on a row that is not a measurement: only a
      `mismatch` had a comparison, so only it may carry them — the two the
      verdict was made on (`xcheck_typical`, `xcheck_worst`) and the two
      diagnostics. They are absent on a mismatch only where the comparison
      produced no finite number (an output of the wrong SHAPE, which the
      detail names). Any other non-measured row ran no comparison, so a
      number there came from somewhere else.
    """
    detail = row.get("outcome_detail") or ""
    if outcome == "measured":
        if detail.strip():
            raise BenchmarkValidationError(
                f"`outcome_detail`={detail!r} is set on a measurement; it says "
                f"what went wrong on a row whose outcome is not `measured`")
        return
    if outcome != "oom" and not detail.strip():
        raise BenchmarkValidationError(
            f"a row with outcome={outcome!r} must say why in `outcome_detail`")
    if len(detail) > OUTCOME_DETAIL_MAX or "\n" in detail or "\r" in detail:
        raise BenchmarkValidationError(
            f"`outcome_detail` must be one line of at most "
            f"{OUTCOME_DETAIL_MAX} characters, got {len(detail)}")
    if outcome == "mismatch":
        return
    for column in _COMPARISON_COLUMNS:
        if (row.get(column) or "").strip():
            raise BenchmarkValidationError(
                f"`{column}` is set on a row with outcome={outcome!r}, which "
                f"ran no comparison")


def _check_timing_method(value: str, measured: bool) -> None:
    """`timing_method` is one of `TIMING_METHODS`, or empty: a measurement
    timed before the column (the old clock), or a row nothing timed. Set on a
    row that is not a measurement it would name a clock nothing ran."""
    if not value:
        return
    if not measured:
        raise BenchmarkValidationError(
            f"`timing_method`={value!r} is set on a row that is not a "
            f"measurement; nothing was timed")
    if value not in TIMING_METHODS:
        raise BenchmarkValidationError(
            f"column `timing_method` must be one of {sorted(TIMING_METHODS)} "
            f"or empty, got {value!r}.")


def _check_cross_check(value: str) -> str:
    v = _check_nonempty_str(value, column="cross_check")
    if v not in CROSS_CHECK_VOCABULARY:
        raise BenchmarkValidationError(
            f"column `cross_check` must be one of "
            f"{sorted(CROSS_CHECK_VOCABULARY)!r}, got {v!r}."
        )
    return v


def _check_cross_check_number(value: str, *, column: str,
                              lo: float, hi: float) -> Optional[float]:
    """Empty, or a finite number in range.

    Empty is legitimate and common: rows written before v4, and rows whose
    cross-check never produced a number (`disabled`, `no_reference`,
    `not_comparable`). So this validates the format and the range, and leaves
    presence to `_check_cross_check_evidence` — which is the half that can say
    something true about both old and new rows.
    """
    v = (value or "").strip()
    if not v:
        return None
    f = _check_float(v, column=column, positive=False)
    if f != f or f in (float("inf"), float("-inf")):
        raise BenchmarkValidationError(
            f"column `{column}` must be finite, got {v!r}."
        )
    if not (lo <= f <= hi):
        raise BenchmarkValidationError(
            f"column `{column}` must lie in [{lo}, {hi}], got {f!r}."
        )
    return f


def _check_cross_check_evidence(row: Dict[str, str]) -> None:
    """A number may only accompany a label that could have produced one.

    `disabled` / `no_reference` / `not_comparable` mean no comparison was ever
    made, so there is nothing to have measured. A row carrying both is either a
    mislabelled comparison or a number carried over from a different one, and
    both would read as evidence that a kernel was vetted when it was not.
    """
    label = (row.get("cross_check") or "").strip()
    if label in ("passed", "unknown"):
        return
    for column in ("cross_check_rel_err", "cross_check_cosine"):
        if (row.get(column) or "").strip():
            raise BenchmarkValidationError(
                f"column `{column}` is set but `cross_check` is {label!r}, "
                f"which means no comparison ran — a number here cannot have "
                f"come from one."
            )


def _check_parallel(value: str) -> str:
    v = _check_nonempty_str(value, column="parallel")
    if not _PARALLEL_RE.match(v):
        raise BenchmarkValidationError(
            f"column `parallel` must match `key=int(,key=int)*` with "
            f"snake_case keys (e.g. `tp=4,ep=2`), got {v!r}."
        )
    keys = [pair.split("=")[0] for pair in v.split(",")]
    if keys != sorted(keys):
        raise BenchmarkValidationError(
            f"column `parallel` keys must be in lexicographic order, "
            f"got {keys!r}."
        )
    return v


def _check_date(value: str) -> str:
    v = _check_nonempty_str(value, column="date")
    if not _DATE_RE.match(v):
        raise BenchmarkValidationError(
            f"column `date` must be ISO 8601 YYYY-MM-DD, got {v!r}."
        )
    try:
        datetime.date.fromisoformat(v)
    except ValueError as e:
        raise BenchmarkValidationError(
            f"column `date` {v!r} is not a real calendar date: {e}."
        ) from e
    return v


def _check_schema_version(value: str) -> int:
    if value is None or value == "":
        raise BenchmarkValidationError(
            f"column `schema_version` is required."
        )
    try:
        n = int(value)
    except ValueError as e:
        raise BenchmarkValidationError(
            f"column `schema_version` must be an integer string, "
            f"got {value!r}."
        ) from e
    if n < 1:
        raise BenchmarkValidationError(
            f"column `schema_version` must be >= 1, got {n}."
        )
    if n > KNOWN_SCHEMA_VERSION:
        raise BenchmarkValidationError(
            f"column `schema_version`={n} is newer than "
            f"KNOWN_SCHEMA_VERSION={KNOWN_SCHEMA_VERSION}; this CSV "
            f"is forward-incompatible with the current validator."
        )
    return n


def _check_shape_json(value: str) -> Dict[str, int]:
    v = _check_nonempty_str(value, column="shape_json")
    try:
        obj = json.loads(v)
    except json.JSONDecodeError as e:
        raise BenchmarkValidationError(
            f"column `shape_json` must be a JSON object string, "
            f"got {v!r}: {e}"
        ) from e
    if not isinstance(obj, dict):
        raise BenchmarkValidationError(
            f"column `shape_json` must decode to an object, got "
            f"{type(obj).__name__}."
        )
    allowed = _allowed_shape_keys()
    out: Dict[str, int] = {}
    for k, vv in obj.items():
        if not isinstance(k, str):
            raise BenchmarkValidationError(
                f"column `shape_json` key {k!r} must be a string."
            )
        if k not in allowed:
            raise BenchmarkValidationError(
                # Derived from the constants, never spelled out: the
                # hardcoded list went stale the moment the symbols were
                # renamed, and told contributors to use names the
                # validator had already stopped accepting.
                f"column `shape_json` has unknown symbol {k!r}; not in "
                f"arch/dims.yaml, phase binding "
                f"({', '.join(sorted(PHASE_BINDING_EXTRAS))}), or "
                f"engine-supplied ({', '.join(sorted(ENGINE_DIMS))})."
            )
        if not isinstance(vv, int) or isinstance(vv, bool) or vv <= 0:
            raise BenchmarkValidationError(
                f"column `shape_json` value for {k!r} must be a positive "
                f"integer, got {vv!r}."
            )
        out[k] = vv
    return out


def _check_dtype_env_json(value: str, column: str = "dtype_env_json",
                          allow_empty: bool = False) -> Dict[str, str]:
    """One validator for both dtype_env columns: the cell's requested
    precision and (`measured_dtype_env_json`) the precision the kernel
    computed at. Same slots, same tokens — a second copy of that vocabulary
    is how the two would drift."""
    if allow_empty and not (value or "").strip():
        return {}
    v = _check_nonempty_str(value, column=column)
    try:
        obj = json.loads(v)
    except json.JSONDecodeError as e:
        raise BenchmarkValidationError(
            f"column `" + column + "` must be a JSON object string, "
            f"got {v!r}: {e}"
        ) from e
    if not isinstance(obj, dict):
        raise BenchmarkValidationError(
            f"column `" + column + "` must decode to an object, got "
            f"{type(obj).__name__}."
        )
    out: Dict[str, str] = {}
    for k, vv in obj.items():
        if k == "weight_granularity":
            # Not a dtype: which weight elements share a scale
            # (`TensorFormat.weight_granularity`).
            if not isinstance(vv, str) or not WEIGHT_GRANULARITY.fullmatch(vv):
                raise BenchmarkValidationError(
                    f"column `{column}` slot `weight_granularity` must "
                    f"match {WEIGHT_GRANULARITY.pattern!r}, got {vv!r}.")
            out[k] = vv
            continue
        if k == "activation_granularity":
            # The same fact on the other operand: which activation elements
            # share one dynamic scale (`ActivationFormat.granularity`). Also
            # not a dtype, and also part of the identity — per-token,
            # per-128-group and per-32-group are three different kernels on
            # the same two dtypes, and a cast measured under one and billed
            # under another is a row nothing can tell apart.
            #
            # Checked against what the declared formats STATE rather than a
            # pattern: an activation's granularity comes from the format its
            # checkpoint runs under (`arch/activation_formats/*.yaml`), so a
            # string outside that set is a cell no declaration produced.
            if not isinstance(vv, str) or vv not in ACTIVATION_GRANULARITIES:
                raise BenchmarkValidationError(
                    f"column `{column}` slot `activation_granularity` must "
                    f"be one of {sorted(ACTIVATION_GRANULARITIES)!r} (what "
                    f"`arch/activation_formats/*.yaml` declares), got "
                    f"{vv!r}.")
            out[k] = vv
            continue
        if k in FORMAT_SLOTS:
            # Not a dtype either: the registry name of the cache format a
            # cached tensor the row touches is stored in — its primary
            # output (`output_format`) or the cache its `kv` slot names
            # (`kv_format`) — when that format has scales, a transform or an
            # unquantised tail (`CachedTensor.format_name`). Checked against
            # the registry and against its dtype slot's token below, once
            # the whole cell is read.
            if not isinstance(vv, str):
                raise BenchmarkValidationError(
                    f"column `{column}` slot `{k}` must be a "
                    f"cache format name, got {vv!r}.")
            out[k] = vv
            continue
        if k not in DTYPE_SLOTS:
            raise BenchmarkValidationError(
                f"column `{column}` has unknown slot {k!r}; "
                f"allowed slots are {sorted(DTYPE_SLOTS)!r}."
            )
        if not isinstance(vv, str) or vv not in CORPUS_TOKENS:
            raise BenchmarkValidationError(
                f"column `{column}` value for slot {k!r} must be "
                f"one of {sorted(CORPUS_TOKENS)!r}, got {vv!r}."
            )
        out[k] = vv
    for k, dtype_slot in FORMAT_SLOTS.items():
        refusal = format_refusal(k, out.get(k), out.get(dtype_slot))
        if refusal:
            raise BenchmarkValidationError(f"column `{column}`: {refusal}.")
    if not allow_empty:
        # The COMPLETENESS rule (every row states activation and output) is
        # the cell's: `measured_dtype_env_json` states only the slots that
        # DIFFER from it, so a bf16 BMM on an fp8 cell names weight and
        # activation and nothing else. Requiring the full set there would
        # force a row to restate slots it did not change.
        check_dtype_slots(obj, column=f"column `{column}`")
    return out


def check_dtype_slots(env: Dict[str, str], *, column: str) -> None:
    """Which slots a dtype environment must state (rule 3). Every slot is
    part of the row's identity (`workload_hash` hashes what is there), and
    a slot that does not apply to the row is left out rather than filled:

      * `activation` and `output` — every kernel has them: required;
      * `weight` — only a row that streams a weight;
      * `kv` — only a row that reads or writes a cache;
      * `metadata` — only a row that writes or reads scales, which says so
        with a granularity: a weight's `weight_granularity`, or a quantised
        activation's `activation_granularity`. Scales need something they
        scale, and until the activation cast became a row of its own that
        could only be a weight — an fp8 cast streams no weight at all and
        its scales are the whole of what it writes beside the elements."""
    missing = sorted({"activation", "output"} - set(env))
    if ("weight_granularity" in env or "activation_granularity" in env) \
            and "metadata" not in env:
        missing.append("metadata")
    if ("metadata" in env and "weight" not in env
            and "activation_granularity" not in env):
        missing.append("weight")
    if missing:
        raise BenchmarkValidationError(
            f"{column} must state `activation` and `output`, `metadata` "
            f"where a weight or a quantised activation stores scales, and a "
            f"`weight` where the scales are a weight's; missing {missing!r}."
        )


def validate_row(row: Dict[str, str]) -> None:
    # Identity columns
    _check_nonempty_str(row.get("model", ""), column="model")
    _check_nonempty_str(row.get("spec_hash", ""), column="spec_hash")
    _check_phase(row.get("phase", ""))
    _check_parallel(row.get("parallel", ""))
    _check_nonempty_str(row.get("framework", ""), column="framework")
    _check_nonempty_str(row.get("framework_version", ""), column="framework_version")
    _check_nonempty_str(row.get("gpu", ""), column="gpu")
    op = _check_op(row.get("op", ""))
    _check_nonempty_str(row.get("node_spec_id", ""), column="node_spec_id")
    shape = _check_shape_json(row.get("shape_json", ""))
    _check_dtype_env_json(row.get("dtype_env_json", ""))
    _check_dtype_env_json(row.get("measured_dtype_env_json", ""),
                          column="measured_dtype_env_json", allow_empty=True)
    _check_nonempty_str(row.get("kernel_impl", ""), column="kernel_impl")

    schema_version = _check_schema_version(row.get("schema_version", ""))
    # Which rows owe a routed token total is the catalog's own statement
    # (the signature declares the axis), not a pair of op names kept here.
    # Owed in the CURRENT identity: a dispatch-boundary row measured before
    # the boundaries carried the total states both of its factors, and the
    # measured-row migration derives it (`current_shape`) exactly as the
    # read side does.
    if declares_routed_token_total(op_signature(op)) and (
            schema_version < 6
            or ("n_tokens" not in shape
                and "n_tokens" not in json.loads(
                    current_shape(op, row["shape_json"])))):
        raise BenchmarkValidationError(
            "routed rows (expert GEMM, dispatch boundaries) require "
            "schema_version >= 6 and a shape_json.n_tokens; migrate the CSV "
            "with tools/migrate_schema.py before loading it.")

    # What happened, before anything is required of the numbers: an outcome
    # other than a measurement HAS no numbers, and demanding them would make
    # "this does not fit on this card" unrecordable — which is how it stayed
    # a retried failure rather than a fact.
    outcome = _check_outcome(row.get("outcome", ""))
    measured = outcome == "measured"
    if not measured:
        # The numbers, and ONLY the numbers, are waived — a row with an
        # outcome still has a contributor, a date, a well-formed seed and a
        # `cross_check` from the vocabulary. Returning here instead skipped
        # all of those, so a third party could have uploaded a refusal with
        # no contributor and `date="yesterday"`.
        for column in MEASUREMENT_COLUMNS:
            if (row.get(column) or "").strip():
                raise BenchmarkValidationError(
                    f"a row with outcome={outcome!r} carries no measurement, "
                    f"but `{column}`={row[column]!r} is set")
    _check_outcome_evidence(row, outcome)
    _check_timing_method(row.get("timing_method") or "", measured)

    # Measurement columns
    if measured:
        _check_float(row.get("latency_us", ""), column="latency_us")
        _check_float(row.get("p50_us", ""), column="p50_us")
        _check_float(row.get("p95_us", ""), column="p95_us")
        _check_int(row.get("samples", ""), column="samples")
        _check_int(row.get("count_per_step", ""), column="count_per_step")
    # A measured op does real work in EITHER compute or memory. Require at
    # least one of achieved_tflops / achieved_bw_gbps to be positive, but allow
    # the other to be 0 — a memory-bound dispatch op (MoE permute/combine) has
    # 0 matmul FLOPs, so its achieved_tflops is legitimately 0 with a positive
    # bandwidth. Both must still be present (non-empty).
        _at = _check_float(row.get("achieved_tflops", ""), column="achieved_tflops",
                           positive=False)
        _bw = _check_float(row.get("achieved_bw_gbps", ""), column="achieved_bw_gbps",
                           positive=False)
        if not (_at > 0 or _bw > 0):
            raise BenchmarkValidationError(
                "at least one of `achieved_tflops` / `achieved_bw_gbps` must be > 0 "
                f"(got {_at} / {_bw})."
            )
        _check_float(row.get("peak_tflops_at_test", ""), column="peak_tflops_at_test")
        _check_float(row.get("peak_bw_at_test", ""), column="peak_bw_at_test")

    # Provenance columns
    _check_cross_check(row.get("cross_check", ""))
    # rel_err is max|delta| / mean|ref|: unbounded above, never negative.
    # cosine is a cosine; anything outside [-1, 1] (modulo fp rounding at the
    # endpoint) did not come from one.
    _check_cross_check_number(row.get("cross_check_rel_err", ""),
                              column="cross_check_rel_err", lo=0.0, hi=float(1e12))
    # The verdict's basis. Both are means/maxima of a non-negative
    # per-element ratio, so never negative; unbounded above because a
    # layout mismatch records inf for both.
    for _col in ("xcheck_typical", "xcheck_worst"):
        _check_cross_check_number(row.get(_col, ""), column=_col,
                                  lo=0.0, hi=float(1e12))
    _check_cross_check_number(row.get("cross_check_cosine", ""),
                              column="cross_check_cosine", lo=-1.000001, hi=1.000001)
    _check_cross_check_evidence(row)
    # Which driver ran this row. Shape only, and only when stated: a row from
    # before v14 carries neither, and an env with no NVIDIA driver mapped
    # carries neither honestly. The shape is enforced because the two are
    # compared to EACH OTHER as strings (`is_driver_substituted`), so a CUDA
    # runtime version in one of them would read as a driver substitution.
    for _col in DRIVER_COLUMNS:
        _check_driver_version(row.get(_col, ""), column=_col)
    _check_nonempty_str(row.get("contributor", ""), column="contributor")
    _check_date(row.get("date", ""))
    # run_id / box are free-form provenance (any string, empty allowed):
    # constraining their spelling would constrain the operators' naming, and
    # nothing joins on them. measured_load_factor is max/mean of a routing
    # distribution — mathematically >= 1, empty when no trace was carried.
    _check_cross_check_number(row.get("measured_load_factor", ""),
                              column="measured_load_factor",
                              lo=1.0 - 1e-6, hi=float(1e12))
    _check_input_seed(row.get("input_seed", ""))


def _check_input_seed(value: str) -> None:
    """The seed column: a decimal unsigned 64-bit integer, or empty on a row
    measured before the column existed. A sign, a decimal point or an
    exponent means a reader (or a spreadsheet) has already changed the
    number, so the row no longer says which data was measured."""
    if value == "":
        return
    if not value.isdigit() or not (0 <= int(value) < 2 ** 64):
        raise BenchmarkValidationError(
            f"column `input_seed` must be a decimal integer in "
            f"[0, 2**64), got {value!r}.")


def parse_rows(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse CSV text. Returns `(header, rows)`.

    Header is whatever the input declares (column order preserved for
    round-trip). Caller is responsible for invoking `validate_header`
    and `validate_row` separately.
    """
    reader = csv.reader(io.StringIO(text))
    rows_iter = iter(reader)
    try:
        header = next(rows_iter)
    except StopIteration:
        raise BenchmarkValidationError("CSV is empty (no header row).") from None
    rows: List[Dict[str, str]] = []
    for raw in rows_iter:
        if not raw:
            continue
        if len(raw) != len(header):
            raise BenchmarkValidationError(
                f"row has {len(raw)} fields but header has {len(header)}."
            )
        rows.append(dict(zip(header, raw, strict=True)))
    return header, rows


def validate_header(header: List[str]) -> None:
    seen = set(header)
    if len(seen) != len(header):
        dupes = [c for c in header if header.count(c) > 1]
        raise BenchmarkValidationError(
            f"CSV header has duplicate columns: {sorted(set(dupes))!r}."
        )
    expected = set(ALL_COLUMNS)
    missing = (expected - seen) - OPTIONAL_COLUMNS
    extra = seen - expected
    if missing or extra:
        msgs = []
        if missing:
            msgs.append(f"missing column(s) {sorted(missing)!r}")
        if extra:
            msgs.append(f"unknown column(s) {sorted(extra)!r}")
        raise BenchmarkValidationError(
            f"CSV header mismatch: {'; '.join(msgs)}. "
            f"Expected exactly the {len(ALL_COLUMNS)} columns documented at "
            f"backend/inference/BENCHMARKS_CSV.md."
        )


def emit_csv(header: List[str], rows: List[Dict[str, str]]) -> str:
    """Re-emit CSV text from parsed (header, rows). Uses the same column
    order as `header` so a well-formed input round-trips byte-identically
    (assuming the input used the same csv dialect: comma delimiter, LF
    line endings, no quoting except where csv.writer would itself quote).
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([row[c] for c in header])
    return buf.getvalue()


def row_identity(row: Dict[str, str]) -> Tuple[str, ...]:
    """What makes this row one row: the 12 identity columns and its outcome.

    The key the corpus can tell rows apart BY — so it is also the key
    anything writing a corpus file must not write twice, and there is one
    statement of it rather than one per writer (`RowCheck` below enforces
    it over a file; `kernel_bench.emit.IncrementalCsvWriter` holds a file to
    it as the rows are produced).

    The outcome rides in the key: one cell may hold a measurement AND the
    record of having once been refused on this card, and those are different
    rows by design (same identity, no latency on the second). Without it the
    harness's own append-only results file became unreadable the moment a
    cell that OOMed on one pass was measured on the next.
    """
    return (*(row[c] for c in IDENTITY_COLUMNS),
            "measured" if is_measurement(row) else (row.get("outcome") or ""))


class RowCheck:
    """The per-row half of `validate_text` over a stream of rows: each row
    passes `validate_row`, and its 12-column identity is new to the stream.
    `reason` answers why a row fails (None when it passes) instead of
    raising, so a caller that keeps the passing rows of a stream — an owner's
    calibration corpus, where an obsolete row must not reject the others —
    applies the same gate the CLI / CI applies to a whole file."""

    def __init__(self) -> None:
        self._seen: Dict[Tuple[str, ...], int] = {}

    def reason(self, row: Dict[str, str], line: int) -> Optional[str]:
        """Why `row` (at `line` of the stream) fails, or None; a passing
        row's identity is recorded, so a later duplicate of it fails."""
        try:
            validate_row(row)
        except BenchmarkValidationError as e:
            return str(e)
        ident = row_identity(row)
        first = self._seen.get(ident)
        if first is not None:
            return (f"duplicate identity tuple (first seen "
                    f"at row {first}); rows whose 12 identity "
                    f"columns and `outcome` are all equal must be "
                    f"deduplicated by the contributor.")
        self._seen[ident] = line
        return None


def validate_text(text: str, *, source: str = "<csv>") -> Tuple[List[str], List[Dict[str, str]]]:
    """Validate raw CSV text. Returns `(header, rows)` on success.
    Raises `BenchmarkValidationError` with a `source:lineno:` prefix
    on first failure.
    """
    header, rows = parse_rows(text)
    try:
        validate_header(header)
    except BenchmarkValidationError as e:
        raise BenchmarkValidationError(f"{source}:1: {e}") from None
    check = RowCheck()
    for idx, row in enumerate(rows, start=2):
        why = check.reason(row, idx)
        if why is not None:
            raise BenchmarkValidationError(f"{source}:{idx}: {why}")
    return header, rows


@memoize(maxsize=1)
def validator_revision() -> str:
    """The revision rows are valid under: this module's code and the
    vocabularies it reads (the dim registry's symbols, the corpus dtype
    tokens, the weight-granularity pattern). A file validated under another
    revision may hold rows the current validator refuses — the live case was
    a `shape_json` symbol dropped from `dims.yaml` — so whatever keeps
    validated rows on disk rebuilds them when this moves."""
    with open(__file__, "rb") as f:
        code = f.read()
    h = hashlib.sha256(code)
    for part in (sorted(_allowed_shape_keys()), sorted(CORPUS_TOKENS),
                 [WEIGHT_GRANULARITY.pattern],
                 sorted(ACTIVATION_GRANULARITIES)):
        h.update("\x1f".join(part).encode())
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def validate_file(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Validate `path`. Convenience wrapper around `validate_text`."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return validate_text(text, source=path)


def assert_round_trip(path: str) -> None:
    """Round-trip rule: parse + emit must be byte-identical on a
    well-formed CSV (per backend/inference/BENCHMARKS_CSV.md rule 6).
    """
    with open(path, encoding="utf-8") as f:
        original = f.read()
    header, rows = parse_rows(original)
    emitted = emit_csv(header, rows)
    if emitted != original:
        raise BenchmarkValidationError(
            f"{path}: round-trip changed the file bytes. Either the "
            f"input uses a non-canonical CSV dialect (need comma "
            f"delimiter, LF line endings, no surrounding whitespace) "
            f"or the validator has a bug. Diff at first divergence: "
            f"orig[:120]={original[:120]!r} vs emit[:120]={emitted[:120]!r}."
        )
