"""Driver — run a job list through every applicable provider.

For each invocation, measure every provider that (a) targets its
signature, (b) is importable in this env, (c) passes its capability
predicate. Optionally numeric-cross-check the kernel against the
signature reference (what makes cross-library latencies comparable).
Emit one CSV row per (invocation × provider) that ran: a measurement, or
the outcome that took its place (`benchmarks.OUTCOMES`: the card refused it,
the cross-check refused its numbers, the provider raised, one launch was too
slow to time).

`run_jobs` never raises on a single provider failure — it records the
failure and moves on, so one bad kernel doesn't sink the whole sweep. The
one exception is a lost CUDA context (`CudaContextLost`): every later launch
in this process would fail for a reason that is not its own, so the process
ends and `run-all` records the implementation in flight as a `crash`.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping,
                    NamedTuple, Optional, Tuple, Union)

from backend.arch.dtypes import CORPUS_TOKENS

from . import launch_trace
from . import quant as _quant
from .env import Env, detect_env
from .invocation import Invocation
from .kernel_symbol import describe as describe_kernel
from .providers import applicable
from .timing import TooSlowToMeasure, time_callable


# A record the oracle writes itself: a cache-format name, or the activation
# grid an output cast shares between oracle and kernel.
ComparedRecord = Union[str, "_quant.ActivationCast"]


def physical_key_for(inv: Invocation, env: Env):
    """The calibration `PhysicalKey` this invocation measures — the same
    library-agnostic key an estimate looks a measurement up under — read off
    the invocation rather than assembled again.

    The plan carries it (`Invocation.physical_key`, from
    `calibration_overlay.point_keys`), so the write side and the query side
    cannot disagree: the only field the plan cannot know is the card, which
    is this run's. A job planned before the key was carried has none, and
    rebuilding it from `dims` and `dtype_env` would be the second derivation
    the carry removes — so it is refused, and `bench plan` restates it.
    """
    from backend.inference.op_calibration import PhysicalKey

    if inv.physical_key is None:
        raise ValueError(
            f"{inv.op} [{inv.node_spec_id}]: this job carries no measurement "
            f"key; re-run `bench plan` to state it (the key is the plan's, "
            f"not the driver's)")
    signature, shape_json, dtype_env_json, phase = inv.physical_key
    return PhysicalKey(
        signature=signature,
        shape_json=shape_json,
        dtype_env_json=dtype_env_json,
        gpu=env.gpu_sku or (env.device_name or "unknown"),
        phase=phase,
    )


# Leave this share of free VRAM unclaimed by the oracle. The kernel still has to
# run afterwards, and the allocator needs headroom to avoid fragmenting.
_ORACLE_VRAM_SHARE = 0.5


def _reference_footprint_bytes(inv, inputs) -> int:
    """Bytes the fp32 reference is expected to need, from the shapes alone.

    Not the kernel's footprint — the oracle's, which is the whole point: it runs
    in fp32 (2× a bf16 kernel) and, for attention, materializes B·N·Sq·Lq scores
    that the kernel never forms.
    """
    import torch

    from .refs import get_ref

    live = sum(t.numel() * t.element_size()
               for t in inputs.values() if isinstance(t, torch.Tensor))
    ref = get_ref(inv.signature)
    # Whether the oracle forms the score matrix is the ORACLE's own
    # declaration (`RefSpec.materializes_scores`), not a list of signature
    # names kept here: the reference that allocates it is the thing that
    # knows, and a new attention oracle states it where it is written.
    if ref is not None and ref.materializes_scores:
        return _score_elements(inv) * 4 + 2 * live
    return 3 * live                      # fp32 copies of the inputs and output


def _score_elements(inv) -> int:
    """B·heads·Sq·Lq: the score matrix a quadratic oracle forms. The head
    axis is the one the signature's query operand (its activation input)
    declares `head_count` — `n_heads` for attention, `n_indexer_heads` for
    the indexer — read from the declaration, not from a name list. A cell
    whose dims or declaration cannot size it raises: an estimate that
    silently fell to zero let every indexer cell past the budget."""
    from backend.inference.op_signatures import get_signature
    sig = get_signature(inv.signature)
    # The query is the one activation input. Asked rather than taken by
    # position: a signature with two of them has no "the" query, and which
    # one scores is then an assumption nothing here could check.
    queries = [t for t in (sig.inputs if sig else ()) if t.role == "activation"]
    heads = queries[0].head_axes() if len(queries) == 1 else ()
    if len(queries) != 1 or len(heads) != 1:
        raise ValueError(
            f"{inv.signature}: its {len(queries)} activation operand(s) "
            f"declare head axes {heads}, not one query carrying one, so the "
            f"oracle's score matrix cannot be sized")
    d = inv.dims
    try:
        return (int(d["B"]) * int(d[heads[0]]) * int(d["s_query"])
                * int(d["s_kv"]))
    except KeyError as e:
        raise ValueError(f"{inv.signature}: the cell binds no {e} to size "
                         f"the oracle's score matrix by") from None


def _reference_will_not_fit(inv, inputs, env) -> Optional[str]:
    """Reason the oracle should not be attempted for this cell, or None.

    Predicted rather than attempted, because attempting it does not reliably
    fail: on a DeepSeek-V3 MLA decode cell the build neither returned nor raised
    — the card sat at 100% for over half an hour while the allocator retried,
    and every remaining job for that model died with the run timeout. A cell the
    kernel can serve must not be lost to the cost of checking it.
    """
    if not env.cuda_available:
        return None
    try:
        import torch
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return None
    try:
        need = _reference_footprint_bytes(inv, inputs)
    except ValueError as e:
        # Not sized is not "fits": the oracle is declined, with the reason.
        return f"fp32 reference cannot be sized: {e}"
    budget = int(free * _ORACLE_VRAM_SHARE)
    if need <= budget:
        return None
    return (f"fp32 reference needs ~{need / 2**30:.1f} GiB, over the "
            f"{budget / 2**30:.1f} GiB oracle budget "
            f"({free / 2**30:.1f} GiB free)")


class _WithoutReference:
    """A RefSpec view whose `reference` is gone — inputs still build, the oracle
    does not. Used instead of mutating the registered spec, which is shared by
    every job of that signature."""

    def __init__(self, spec):
        self._spec = spec
        self.reference = None

    def __getattr__(self, name):
        return getattr(self._spec, name)


#: Keys per coverage question. A whole model's plan is ~18-22k keys; asked at
#: once, the lookup took 3s on the service's own host and timed out across a
#: tunnel. A 2000-key batch takes 1-3s on the service's host (2026-10-24,
#: DeepSeek-V3's plan: ~1s with the service's row cache warm, ~2.5s cold).
COVERAGE_BATCH = 2000
#: Times one batch is asked before its keys are given up as not asked, and
#: the pause before the n-th retry (`COVERAGE_BACKOFF_S * n`). A busy service
#: (the dev box at load 15-27) times a batch out now and then; one batch
#: failing used to turn the skip off for the whole run.
COVERAGE_ATTEMPTS = 3
COVERAGE_BACKOFF_S = 5.0


def _kernel_bench_coverage(jobs: List[Dict], env: Env, result: "RunResult",
                           log: Callable[[str], None]) -> Dict[Any, set]:
    """Per physical key, the implementations Kernel Bench's shared rows
    already measure (`backend.measure.lookup.covered`), for `--skip-existing`,
    asked `COVERAGE_BATCH` keys at a time.

    Each batch is asked up to `COVERAGE_ATTEMPTS` times. A batch that never
    answers leaves its keys out of the answer — not covered, so they are
    measured again (visible waste), never skipped on a guess — and the run
    says how many keys went unasked; the other batches' answers still count.
    Two batches in a row that never answer mean the service is down, not
    busy: the rest are not asked, so a hung service costs two batches'
    retries rather than every batch's."""
    from backend.measure import lookup as _lookup

    if not _lookup.service_url():
        log("--skip-existing: no Kernel Bench service configured "
            "(HUGGINGARCH_BENCH_URL): only this output and --measured-in count")
        return {}
    keys = []
    for job in jobs:
        try:
            keys.append(physical_key_for(Invocation.from_job(job), env))
        except ValueError:
            continue                    # reported where the job is run
    batches = [keys[i:i + COVERAGE_BATCH] for i in range(0, len(keys), COVERAGE_BATCH)]
    covered: Dict[Any, set] = {}
    failed = not_asked = 0          # batches that never answered / were not asked
    unasked = 0                     # their keys
    in_a_row = 0
    last_error = ""
    for n, batch in enumerate(batches):
        if in_a_row >= 2:
            not_asked += 1
            unasked += len(batch)
            continue
        for attempt in range(1, COVERAGE_ATTEMPTS + 1):
            try:
                covered.update(_lookup.covered(batch))
                in_a_row = 0
                break
            except _lookup.MeasurementsUnavailable as e:
                last_error = str(e)
                log(f"--skip-existing: coverage batch {n + 1}/{len(batches)} "
                    f"({len(batch)} keys), attempt {attempt}/{COVERAGE_ATTEMPTS}: {e}")
                if attempt < COVERAGE_ATTEMPTS:
                    time.sleep(COVERAGE_BACKOFF_S * attempt)
        else:
            failed += 1
            in_a_row += 1
            unasked += len(batch)
    if unasked:
        state = "INACTIVE" if unasked == len(keys) else "PARTIAL"
        rest = f", {not_asked} more not asked after two in a row" if not_asked else ""
        msg = (f"--skip-existing is {state} against Kernel Bench: {unasked} of "
               f"{len(keys)} keys unanswered ({failed} of {len(batches)} batches "
               f"failed {COVERAGE_ATTEMPTS} attempts{rest}; last: {last_error}) — "
               "those keys count as not measured, so cells among them it already "
               "measures will be measured again")
        result.failures.append(msg)
        log(msg)
    return covered


class CudaContextLost(RuntimeError):
    """The CUDA context latched an error while one implementation ran: no
    later launch in this process can succeed. Raised out of `run_jobs`
    rather than recorded as that implementation's `error` and carried on —
    carrying on would write an `error` row against every cell after it, each
    blaming a kernel for a context it never broke. The process exits; the
    supervisor (`run-all`) reads the implementation in flight off the
    progress sidecar and records it as a `crash`."""


def _context_lost(env: Env) -> bool:
    """Whether the CUDA context has latched an error (a synchronize raises).
    Asked after a provider raised, before anything else touches the device."""
    if not env.cuda_available:
        return False
    try:
        import torch
        torch.cuda.synchronize()
    except Exception:
        return True
    return False


def _stop_if_context_lost(env: Env, error: BaseException, stage: str) -> None:
    """In a job's SHARED stage (the oracle, a weight grid): a lost context is
    raised at once, while no implementation is named in flight, so the
    supervisor records nothing. Swallowed instead, it surfaced at the next
    provider's launch, and that provider was recorded as the crash."""
    if _context_lost(env):
        raise CudaContextLost(
            f"{stage}: the CUDA context latched {type(error).__name__}: {error}"
        ) from error


@dataclass
class RunResult:
    rows: List[Dict[str, str]] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)     # "no applicable provider" notes
    failures: List[str] = field(default_factory=list)    # provider errors
    # Cells the card refused — an OOM is a recorded outcome, and its row is
    # in `rows` like any other. Listed separately because a sweep that OOMs
    # a tenth of its cells has not FAILED a tenth of its cells, and the
    # summary saying so is what kept sending people to look for a bug.
    refused: List[str] = field(default_factory=list)
    # Rows whose LAUNCH the harness could not see: a kernel behind a pybind
    # entry point dispatches no observable ops, so the launch gate had
    # nothing to compare against the cell. Recorded rather than silent,
    # because "the door agreed" and "the door never looked" used to take the
    # same branch — and a span where every framework row lands here is a
    # span the gate has never guarded, which is exactly where a framework
    # moving work across a launch boundary inside compiled code would hide
    # (the live case: `gated_delta_rule`, where vLLM moved the q/k l2norm
    # into an earlier launch and nothing in the rows could disagree).
    unobserved: List[str] = field(default_factory=list)
    # Launch paths switched off for the rest of this run because their
    # canary — the path's smallest pending cell, attempted first — ended in
    # no measurement: the path, and why. Its other cells were not attempted
    # (`not_attempted` counts them) and carry no row, so the next run asks
    # again.
    disabled: Dict[str, str] = field(default_factory=dict)
    not_attempted: int = 0


def run_jobs(
    jobs: Iterable[Dict],
    *,
    env: Optional[Env] = None,
    date: str,
    contributor: str,
    run_id: str = "",
    box: str = "",
    cross_check: bool = True,
    skip_existing: bool = False,
    log: Callable[[str], None] = lambda m: None,
    on_row: Optional[Callable[[Dict], None]] = None,
    on_job_start: Optional[Callable[[int, Dict], None]] = None,
    measured_keys: Optional[Iterable[Any]] = None,
    on_attempt: Optional[Callable[[List[Dict[str, str]]], None]] = None,
    on_disable: Optional[Callable[[str, str], None]] = None,
) -> RunResult:
    """`on_row` is called with each row as soon as it exists — a measurement
    or a recorded outcome — so a caller can persist it immediately. Rows are
    also accumulated in `RunResult.rows` as before — a sweep is hours long,
    and buffering everything until the end meant a kill discarded all of it.

    `on_attempt` is told which implementation is in flight: before each
    provider is built, the row that would record it if the process died
    there (a `build_refusal_row` whose outcome the supervisor decides —
    `crash` or `timeout`), and `[]` once the provider is done. The shared
    stage of a job (inputs, oracle) names no implementation: a death there is
    the harness's, not any kernel's.

    `on_job_start(index, job)` says where a restarted run resumes (`index`)
    and which job is on the GPU (`job`). They are the same job except while
    a launch path's canary runs ahead of its own job (see the canaries
    below); `on_disable(path, why)` is told each path a canary disables."""
    from backend.inference import benchmarks as _benchmarks

    from .emit import build_refusal_row, build_row

    env = env or detect_env()
    result = RunResult()

    def record(row: Dict[str, str], run_key, label: str, note: List[str],
               message: str) -> None:
        """Keep and persist one row that is not a measurement, and say so in
        `note` (the run's refused or failures list)."""
        result.rows.append(row)
        measured_here.add(run_key)
        verdicts[run_key] = row
        note.append(message)
        if on_row is not None:
            try:
                on_row(row)
            except Exception as write_error:
                result.failures.append(
                    f"{label}: {row['outcome']} recorded but not persisted "
                    f"({type(write_error).__name__}: {write_error})")
        settled()

    def settled() -> None:
        """This implementation's row is on disk: nothing is in flight any
        more. Said AS SOON as the row lands, not only when the provider's
        loop body ends, so a kill between the two cannot add a `crash` row
        beside the row it already has."""
        if on_attempt is not None:
            on_attempt([])

    # Measure once: an identity already measured — in this run's output file
    # or another results CSV (`--skip-existing`, `--measured-in`: the CLI
    # passes their keys as `measured_keys`) — is skipped. The same WITHIN
    # this run: the plan sweeps deployment parameters (paged-KV
    # block size, …) as separate jobs, but a provider only takes on a parameter
    # it actually serves: `impl_params` folds the ones it does into `impl_id`,
    # so a paged backend's identity varies with the block size while torch SDPA
    # — which has no paged KV — resolves to the SAME identity for every block in
    # the sweep. Measuring it once per block produced N byte-identical rows and
    # a `duplicate identity tuple` warning at write time. Keyed on the
    # physical op and the implementation, so ANY swept parameter a provider
    # doesn't consume collapses, not just this one.
    measured_here: set = set(measured_keys or ())
    # Which (implementation, phase, dtype, size band) have had their launch
    # sequence observed in this run — see the profiler call below.
    observed_here: set = set()

    # What each attempt came to, by its run key: the row that recorded it,
    # measurement or outcome. Read to judge a launch path's canary.
    verdicts: Dict[Any, Dict[str, str]] = {}

    def pending(inv, provs, *, say: bool = True) -> list:
        """The providers of this cell still to attempt, as `(provider,
        impl_id, version, run_key, label)`; `say` reports the ones dropped.

        Decided BEFORE allocating anything. Both the physical key and every
        provider's `impl_id` are functions of the invocation alone, so a job
        whose providers are all already covered can be dropped without ever
        touching the GPU.

        This used to run after the inputs and the fp32 oracle were built, so a
        cell the corpus fully covered still paid for its oracle — and the
        oracle is the expensive part: ~2× the kernel's memory, O(S²) for
        attention. One such cell (MLA decode at B=75,588) sat at 100% GPU for
        36 minutes building a reference for a row that would then be skipped,
        and the whole model's remaining jobs were lost to the timeout."""
        def skip(note: str, line: str) -> None:
            if say:
                result.skipped.append(note)
                log(line)

        pk = physical_key_for(inv, env)
        todo = []
        for p in provs:
            # Resolve the deployment kernel parameters first: it decides the
            # implementation's recorded identity (so coverage is asked about the
            # right thing) and it declines unsupported configurations before any
            # GPU input is allocated for a shape this backend won't measure.
            try:
                impl_id = p.impl_id(inv)
            except NotImplementedError as e:
                skip(f"{p.library}:{p.kernel_impl} {inv.op}: {e}",
                     f"skip {p.library}:{p.kernel_impl} ({e})")
                continue
            label = f"{p.library}:{impl_id} {inv.op} {inv.dims}"
            version = p.version(env) or "unknown"
            run_key = (pk, p.library, version, impl_id)
            if run_key in canary_keys and run_key in measured_here:
                if say:
                    log(f"skip {label} (attempted first, as its launch "
                        f"path's canary)")
                continue
            if run_key in measured_here:
                skip(f"{label}: already measured in this run — this provider's "
                     f"identity does not vary with the swept kernel parameters",
                     f"skip {label} (duplicate identity in this run)")
                continue
            if (p.library, version, impl_id) in covered.get(pk, ()):
                skip(f"{label}: already measured (Kernel Bench covers it)",
                     f"skip {label} (already measured)")
                continue
            todo.append((p, impl_id, version, run_key, label))
        return todo

    def attempt(inv, todo) -> None:
        """Attempt one cell with the given providers (`pending`'s entries):
        its inputs and oracle built once, then each provider built, checked
        and timed, every attempt ending in a row or a note."""

        # Canonical inputs + reference for this op — built ONCE, shared by
        # every provider so their outputs (and latencies) are comparable.
        # Guard per-job: a malformed job (e.g. dims missing a signature's
        # required axis) records a failure and is skipped, never sinking the
        # whole sweep — as the module docstring promises.
        from .cell_check import complaints as cell_complaints
        from .refs import get_ref, input_generator
        device = "cuda" if env.cuda_available else "cpu"
        def refuse_inputs(detail: str) -> None:
            # The card cannot hold this cell's INPUTS, so no provider of it
            # can run — one recorded refusal per implementation that was
            # going to, because the row's identity names one (`kernel_impl`)
            # and "this cell does not fit" is true of each. The commonest
            # OOM by far is here rather than in a provider: the biggest
            # cells are refused while the harness is still allocating.
            for p, impl_id_of, version_of, run_key, label in todo:
                row = build_refusal_row(
                    inv, p, env, outcome="oom", date=date,
                    contributor=contributor, impl_id=impl_id_of,
                    framework_version=version_of, run_id=run_id, box=box,
                    detail=f"building the cell's inputs: {detail}")
                record(row, run_key, label, result.refused,
                       f"{label}: out of memory (inputs)")
            log(f"oom  inputs {inv.op} {inv.dims}")

        # A first out-of-memory on a card that has room once released is the
        # state this process left it in, not the cell: the inputs are built
        # once more, here, and a second out-of-memory is the cell's `oom` —
        # the provider's rule. Deferred to the next run instead, with no
        # row, a cell larger than the card whose every single allocation
        # fits (two 40 GiB operands on an 80 GiB H100) failed again on every
        # pass and never counted as tried.
        crowded = None
        for retried in (False, True):
            try:
                spec = get_ref(inv.signature)
                # One generator per cell, seeded off the cell's own key
                # (`Invocation.input_seed`): the data is this cell's, the
                # same on every box and every rerun, and it is built ONCE
                # here so every provider of the cell is timed and checked on
                # one batch.
                inputs = (spec.make_inputs(inv, device,
                                           input_generator(inv, device))
                          if spec else None)
                break
            except Exception as e:
                _stop_if_context_lost(env, e,
                                      f"{inv.op} {inv.dims}: the inputs")
                detail = f"{type(e).__name__}: {e}"
                if _is_out_of_memory(e, env) and not retried:
                    # Judged below, once this clause has dropped the
                    # traceback — it holds the builder's frames, and with
                    # them whatever it had allocated.
                    crowded = RuntimeError(detail)
                elif _is_out_of_memory(e, env):
                    refuse_inputs(detail)
                else:
                    result.failures.append(
                        f"{inv.op} {inv.dims}: input build failed "
                        f"({detail}); job skipped")
                    log(f"FAIL inputs {inv.op}: {e}")
            if crowded is not None and _fits_after_release(env, crowded):
                crowded = None
                log(f"retry inputs {inv.op} {inv.dims}: out of memory on a "
                    f"crowded card; released, building once more")
                continue
            if crowded is not None:
                refuse_inputs(str(crowded))
            # Whatever was allocated before the raise is still on the device,
            # and an input builder raises PART WAY through. Left there it
            # accumulates across every failing job and eventually takes the
            # context down — the failure then surfaces at some unrelated later
            # launch, pointing the stack at a kernel that never ran.
            if env.cuda_available:
                import torch
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return
        # Is this the cell the row will be billed for? A shape or a dtype
        # the operands do not have is a measurement of something else, and
        # it is cheaper to refuse than to find it later by multiplying out
        # a shape by hand (`cell_check`).
        if inputs is not None:
            wrong = cell_complaints(inv, inputs)
            if wrong:
                result.failures.append(
                    f"{inv.op} {inv.dims}: built cell disagrees with the row "
                    f"({'; '.join(wrong)}); job skipped")
                log(f"FAIL cell {inv.op}: {'; '.join(wrong)}")
                if env.cuda_available:
                    import torch
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                return
        # Reference is built SEPARATELY from inputs: the fp32 eager oracle is
        # O(S²) for attention (materializes the full score matrix) and ~2× the
        # bf16 kernel for GEMM, so it OOMs at high seq×batch where the real
        # O(S) flash / bf16 kernel fits. On a reference OOM, measure WITHOUT
        # cross-check rather than drop a cell the card can actually run — the
        # kernel was already validated against the reference at smaller shapes.
        # Which of those happened is recorded per row (`cross_check`), because
        # "measured with no oracle" and "measured and verified" are different
        # claims and the CSV used to make them look identical.
        ref_out = None
        if spec and spec.reference and inputs is not None:
            too_big = _reference_will_not_fit(inv, inputs, env)
            if too_big:
                log(f"no-xcheck {inv.op} {inv.dims}: {too_big} — measuring "
                    f"without cross-check")
                spec = _WithoutReference(spec)
        if spec and spec.reference and inputs is not None:
            try:
                ref_out = spec.reference(inputs)
            except Exception as e:
                _stop_if_context_lost(env, e, f"{inv.op} {inv.dims}: the reference")
                log(f"no-xcheck {inv.op} {inv.dims}: reference build failed "
                    f"({type(e).__name__}) — measuring without cross-check")
                if env.cuda_available:
                    import torch
                    torch.cuda.empty_cache()

        # A kernel that quantizes the weight itself gets the canonical weight
        # placed on ITS grid first — see kernel_bench/quant.py. Both the kernel
        # and the reference then see the same representable values, so what the
        # cross-check measures is the kernel's arithmetic rather than the
        # format's lossiness, and the activation-dtype band is the right band
        # again. Cached per (dtype, group): providers sharing a grid share the
        # work, and marlin's NVFP4 blocks of 16 do NOT share with humming's 128.
        eff_cache: Dict[Any, Any] = {}

        def _effective(prov):
            """(inputs, reference) this provider should be measured against."""
            if inputs is None or prov.weight_group is None:
                return inputs, ref_out
            key = (inv.dtype_env.get("weight", ""), int(prov.weight_group),
                   prov.weight_scale)
            if key not in eff_cache:
                try:
                    eff = _quant.effective_weights(
                        inputs, key[0], key[1], scale_encoding=key[2],
                        act_dtype=_act_torch_dtype(inv))
                except Exception as e:
                    _stop_if_context_lost(env, e, f"{inv.op} {key}: the weight grid")
                    # Never fall back to the un-rounded weight silently: that is
                    # the lossy comparison this replaces, and it would look
                    # identical to a pass.
                    log(f"grid {inv.op} {key}: {type(e).__name__}: {e} — "
                        f"measuring without cross-check")
                    eff_cache[key] = (inputs, None)
                    return eff_cache[key]
                if eff is None:
                    eff_cache[key] = (inputs, ref_out)
                else:
                    r = None
                    if spec and spec.reference:
                        try:
                            r = spec.reference(eff)
                        except Exception as e:
                            _stop_if_context_lost(
                                env, e, f"{inv.op} {key}: the grid's reference")
                            log(f"grid {inv.op} {key}: reference failed "
                                f"({type(e).__name__}) — no cross-check")
                    eff_cache[key] = (eff, r)
            return eff_cache[key]

        # Each implementation's attempt, with whether it is the one retry a
        # crowded card earns (below, where an out-of-memory is classified).
        queue = [(entry, False) for entry in todo]
        while queue:
            (p, impl_id, version, run_key, label), retried = queue.pop(0)
            # The loop names its own provider's `impl_id` and `version`: they
            # used to be the leftovers of the selection loop above, i.e. the
            # LAST provider's, so a job with two providers recorded the
            # first one's OOM under the second one's identity.
            if inputs is None:
                result.skipped.append(f"{label}: no canonical inputs for signature "
                                      f"{inv.signature!r} (add a refs entry)")
                continue

            def failed(outcome: str, detail: str, *, p=p, impl_id=impl_id,
                       version=version, run_key=run_key, label=label,
                       **numbers) -> None:
                """Record this provider's attempt as `outcome` (an attempt,
                `benchmarks.OUTCOMES`), with the line that says why."""
                record(build_refusal_row(
                    inv, p, env, outcome=outcome, date=date,
                    contributor=contributor, impl_id=impl_id,
                    framework_version=version, run_id=run_id, box=box,
                    detail=detail, **numbers),
                    run_key, label, result.failures,
                    f"{label}: {outcome} — {detail}")

            # The weight grid and its reference are SHARED work (one per
            # grid, cached for every provider on it), so they run before this
            # provider is named in flight: a death or a lost context there is
            # the harness's, and must not be recorded against this kernel.
            try:
                p_inputs, p_ref = _effective(p)
            except CudaContextLost:
                raise
            except Exception as e:
                result.failures.append(
                    f"{label}: weight grid failed ({type(e).__name__}: {e}); "
                    f"not recorded — the grid is shared, not this kernel's")
                continue
            context_lost = False
            if on_attempt is not None:
                on_attempt([build_refusal_row(
                    inv, p, env, outcome="crash", date=date,
                    contributor=contributor, impl_id=impl_id,
                    framework_version=version, run_id=run_id, box=box,
                    detail="in flight")])
            try:
                p_inputs = p.convert_inputs(inv, p_inputs)
                if p.reference is not None:
                    try:
                        p_ref = p.reference(p_inputs)
                    except Exception as e:
                        p_ref = None
                        result.failures.append(
                            f"{label}: provider oracle failed "
                            f"({type(e).__name__}: {e}); job skipped")
                        log(f"FAIL provider-oracle {label}: {e}")
                        continue
                case = p.build(inv, env, p_inputs)
                case.to_canonical = p.wrap_output(inv, case.to_canonical)
                # One label per reason, resolved where the reason is known.
                xrel = xcos = xtyp = xwor = None
                if not cross_check:
                    xchk = "disabled"
                elif p_ref is None:
                    xchk = "no_reference"
                elif case.to_canonical is None:
                    xchk = "not_comparable"
                else:
                    xchk = "passed"          # downgraded below only by a reject
                trace = None
                if cross_check and p_ref is not None and case.to_canonical is not None:
                    try:
                        # Recorded around the call the check already makes, so
                        # the timed loop is untouched: what aten performed,
                        # with the dtype and shape of every operand.
                        recorded = launch_trace.record()
                        x_dtype, x_records = _xcheck_basis(inv, p_inputs)
                        try:
                            err, xrel, xcos, xtyp, xwor = _cross_check(
                                case, p_ref, x_dtype, trace=recorded,
                                records=x_records)
                            trace = recorded.trace
                        except launch_trace.Unobservable as e:
                            # The trace could not hand the launch's result
                            # back, which is not the kernel failing. What the
                            # launch dispatched is on the trace, so the gate
                            # still judges it; the result is made again
                            # untraced, on a case built afresh (the traced
                            # call may have written its buffers).
                            log(f"untraced {label}: {e}")
                            trace = recorded.trace
                            case = p.build(inv, env, p_inputs)
                            case.to_canonical = p.wrap_output(
                                inv, case.to_canonical)
                            err, xrel, xcos, xtyp, xwor = _cross_check(
                                case, p_ref, x_dtype, records=x_records)
                    except Exception as e:
                        if not _is_out_of_memory(e, env):
                            raise
                        # The CHECK ran out of memory, not the kernel: the
                        # comparison materialises fp32 copies and a mask on
                        # top of the oracle. The policy for that is already
                        # stated above (`_reference_will_not_fit`) — measure
                        # without a cross-check rather than lose a cell the
                        # card can run — and recording a refusal here would
                        # instead write "this card cannot hold this cell" and
                        # stop anyone measuring it again.
                        err = xrel = xcos = xtyp = xwor = None
                        xchk = "no_reference"
                        if env.cuda_available:
                            import torch
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        log(f"no-xcheck {label}: the check itself ran out of "
                            f"memory — measuring without it")
                    if trace is None or not trace.observed:
                        # Not assent: nothing was observable. Said here so a
                        # run reports how much of itself the launch gate
                        # could not judge.
                        says = None
                        result.unobserved.append(label)
                    else:
                        says = launch_trace.disagreement(
                            trace, declared_dtypes=sorted(set(
                                (inv.dtype_env or {}).values())),
                            declared_elements=_declared_elements(inv),
                            records=_xcheck_basis(inv, p_inputs)[1],
                            packed_byte_shapes=tuple(
                                launch_trace.paged_byte_cache_shape(
                                    p_inputs[name].shape, inv.kv_block_size,
                                    inv.dtype_env[slot])
                                for name, slot in p.paged_byte_cache_inputs))
                    if says is not None:
                        # The launch is not this cell's, so its latency is
                        # not either: a `mismatch` like a cross-check reject,
                        # because both say the row would price something
                        # else than it measured — and both depend only on the
                        # provider and the cell, so a pass that did not record
                        # it would pay for the build and the check again.
                        failed("mismatch", f"launch gate: {says} "
                                           f"[{trace.describe()}]")
                        log(f"LAUNCH {label}: {says}")
                        continue
                    if err is not None:
                        # Recorded, not dropped: "this implementation computes
                        # something else here" is what a coverage report most
                        # needs to show, and a row-less rejection read as
                        # "nobody has tried". Never as a measurement — the
                        # latency is not timed, the numbers are the verdict's.
                        x_dtype, x_records = _xcheck_basis(inv, p_inputs)
                        band = _xcheck_band(x_dtype, x_records)
                        failed("mismatch",
                               f"cross-check FAILED on the "
                               f"{x_dtype} band (typical "
                               f"{xtyp:.3g} vs {band.typical:g}, worst "
                               f"{xwor:.3g} vs {band.worst:g}; cosine "
                               f"{xcos:.5f})" if math.isfinite(xtyp) else
                               "cross-check FAILED: the output's shape is "
                               "not the reference's",
                               rel_err=xrel, cosine=xcos,
                               xcheck_typical=xtyp, xcheck_worst=xwor)
                        log(f"XCHK {label}: worst={err:.3g} "
                            f"rel_err={xrel:.3g} cos={xcos:.5f}")
                        continue
                try:
                    stats = time_callable(case.run)
                except TooSlowToMeasure as e:
                    # A `timeout`: one launch costs more than the whole timing
                    # budget, so this implementation cannot be sampled here.
                    # Recorded, because it is as slow on the next pass — and
                    # left running it would stall the sweep with no output.
                    failed("timeout", f"one launch over the timing budget: {e}")
                    log(f"slow {label}: {e}")
                    continue
                # AFTER timing, never inside it: the profiler serializes the
                # stream, so observing within the timed region would inflate
                # every latency it touched. A box whose profiler cannot attach
                # still measures — the column stays empty and says so, rather
                # than sinking a sweep over evidence that is not the number.
                #
                # And once per GROUP, not once per row. The observation costs
                # a profiler start/stop with its own launch: measured on an
                # RTX 4090 it is 23% of everything a measured row spends,
                # second only to the timing loop itself. What it buys is the
                # answer to "did this implementation substitute another
                # kernel", and that answer does not vary row by row — it
                # varies by implementation, by phase, by dtype and by SIZE
                # (a dispatch heuristic picks differently at 1k and at 1M),
                # which is exactly `_symbol_group`. Rows outside the sample
                # carry the empty column the schema already defines as "not
                # observed", rather than a copy of a neighbour's evidence.
                ksym = ""
                group = _symbol_group(inv, p.library, impl_id)
                if group not in observed_here:
                    try:
                        ksym = describe_kernel(case.run)
                        observed_here.add(group)
                    except Exception as e:
                        result.skipped.append(
                            f"{label}: no kernel symbol ({type(e).__name__}: {e})")
                row = build_row(inv, p, stats, env, case=case,
                                date=date, contributor=contributor,
                                cross_check=xchk, rel_err=xrel, cosine=xcos,
                                xcheck_typical=xtyp, xcheck_worst=xwor,
                                kernel_symbol=ksym, launch_trace=trace,
                                run_id=run_id, box=box)
                result.rows.append(row)
                measured_here.add(run_key)
                verdicts[run_key] = row
                if on_row is not None:
                    # Persisting must not sink the sweep: a write error is
                    # reported like any other per-row failure and measuring
                    # continues.
                    try:
                        on_row(row)
                    except Exception as e:
                        result.failures.append(
                            f"{label}: measured but not persisted "
                            f"({type(e).__name__}: {e})")
                        log(f"WRITE {label}: {e}")
                settled()
                log(f"ok   {label}  p50={stats.p50_us:.1f}us")
            except Exception as e:
                context_lost = _context_lost(env)
                if context_lost:
                    # Before anything else touches the device: an OOM test or
                    # an `error` row now would describe the latched context,
                    # not this cell. `on_attempt` still names this provider.
                    raise CudaContextLost(
                        f"{label}: the CUDA context latched "
                        f"{type(e).__name__}: {e}") from e
                out_of_memory = _is_out_of_memory(e, env)
                if (out_of_memory and not retried
                        and _fits_after_release(env, e)):
                    # The card had room for the request once this process
                    # let go of what it held: crowded, not too small. Neither
                    # result is known yet, so the attempt is made once more,
                    # here; a second out-of-memory is the cell's `oom`. An
                    # `error` row would have kept the cell tried for good
                    # (and, on a canary, stopped its path) over the state
                    # this process left the card in.
                    log(f"retry {label}: out of memory on a crowded card; "
                        f"released, attempting once more")
                    queue.insert(0, ((p, impl_id, version, run_key, label),
                                     True))
                elif out_of_memory:
                    # A RESULT, not a failure: this shape does not fit on this
                    # card for this implementation, and the row says so.
                    # Written through the same `on_row` as a measurement so it
                    # lands wherever measurements land, and counted separately
                    # in the run's summary — a sweep that OOMs half its cells
                    # has not failed half its cells.
                    record(build_refusal_row(
                        inv, p, env, outcome="oom", date=date,
                        contributor=contributor, impl_id=impl_id,
                        framework_version=version, run_id=run_id, box=box,
                        detail=f"{type(e).__name__}: {e}"),
                        run_key, label, result.refused, f"{label}: out of memory")
                    log(f"oom  {label}")
                else:
                    # An `error`: the provider's build or launch raised. A
                    # `NotImplementedError` here is one too — the provider
                    # accepted the cell (`accepts`, `impl_id`) and then could
                    # not serve it, which is its declaration missing a case,
                    # not a decline.
                    failed("error", f"{type(e).__name__}: {e}")
                    log(f"FAIL {label}: {e}")
                # A build that raises PART WAY through has already allocated —
                # and the next provider for this same job runs before the
                # end-of-job release below. Synchronize so an async failure
                # surfaces here rather than detonating at an unrelated call
                # later (its traceback then names a kernel that never ran), and
                # drop the partial allocation instead of carrying it forward.
                if env.cuda_available:
                    import torch
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass                              # already latched
                    torch.cuda.empty_cache()
            finally:
                # Except when the context is lost: the run stops HERE, and the
                # supervisor reads this provider off the sidecar as the crash.
                if not context_lost:
                    settled()

        # Release this shape's GPU inputs/reference before the next job so the
        # caching allocator doesn't accumulate reserved-but-free blocks across a
        # long sweep — fragmentation there turns later big-MoE shapes into false
        # OOMs (real on a 24GB part; avoidable when it's just leftover cache).
        if env.cuda_available:
            import torch
            # `eff_cache` holds a per-grid weight copy AND its reference, so it
            # is part of this shape's footprint, not bookkeeping — dropping only
            # `inputs`/`ref_out` would leave the largest tensors alive.
            inputs = ref_out = None
            eff_cache.clear()
            torch.cuda.empty_cache()
    # Across runs: what Kernel Bench's shared rows already measure
    # (`--skip-existing`), one lookup for every job's physical op through the
    # estimates' own client (`backend.measure.lookup.covered`). With no
    # service configured nothing is known there — said, not silent; keys
    # whose batch the service never answers count as not measured, loudly.
    jobs = list(jobs)
    covered: dict = {}
    if skip_existing:
        covered = _kernel_bench_coverage(jobs, env, result, log)

    # Each job's invocation and the providers that would measure it, read
    # once: the canaries are chosen from them before anything runs.
    # A job that cannot even state its invocation keeps its exception and
    # raises at its own turn, after `on_job_start` has named it: the sidecar
    # resumes at the job that died, not at one it never reached.
    staged: List[Tuple[Optional[Invocation], Any, Optional[BaseException]]] = []
    for job in jobs:
        try:
            inv = Invocation.from_job(job)
        except ValueError as e:
            staged.append((None, str(e), None))
            continue
        except (KeyError, TypeError) as e:
            staged.append((None, None, e))
            continue
        staged.append((inv, applicable(inv, env), None))

    # A CANARY per launch path (`_launch_path`: one implementation at one
    # phase and precision). When an installed release moves, renames or
    # drops what a provider calls, every cell of that path fails the same
    # way — `cannot import name 'fused_grouped_topk'` ten thousand times a
    # pass, each after its inputs and oracle were built. So the path's
    # SMALLEST pending cell (the model's own, so the exact signature, dtype
    # and phase the rest would run) is attempted first, when the path first
    # comes up, as an ordinary attempt: a pass is that cell's measurement
    # and costs nothing extra; an attempt that ends in no measurement is
    # that cell's row, as always, and — unless it ran out of memory, which
    # is the cell's size and not the path (`judge`) — the path's other
    # cells are not attempted this run. They are left without rows, so the next run asks
    # again — one cell's worth — and a fixed env is picked up by itself.
    # A path that passes runs exactly as before, so a failure particular to
    # one cell is still that cell's row.
    canaries: Dict[tuple, Tuple[float, int]] = {}   # path → (size, job)
    cells: Dict[tuple, int] = {}                     # path → pending cells
    canary_keys: set = set()      # run keys attempted ahead of their job
    for j, (inv, provs, _deferred) in enumerate(staged):
        if inv is None or not provs:
            continue
        try:
            entries = pending(inv, provs, say=False)
        except ValueError:
            continue                    # no key: reported where the job is run
        size = _cell_elements(inv)
        for p, impl_id, *_rest in entries:
            path = _launch_path(inv, p.library, impl_id)
            cells[path] = cells.get(path, 0) + 1
            if path not in canaries or (size, j) < canaries[path]:
                canaries[path] = (size, j)
    tried: set = set()
    disabled: set = set()

    def judge(inv, entry) -> None:
        """Disable the canary's path if its attempt left a row that is not a
        measurement. No row, no verdict: an attempt that left none (an input
        build that raised, a cell check, the oracle, the weight grid) says
        nothing about the provider, which runs on.

        Nor does an `oom` row, whether the inputs or the provider ran out:
        it is a verdict on the cell's SIZE, and the path is about what fails
        whatever the size. Disabling on it left the larger cells pending
        with no row of their own, so every pass ran the model again to
        record one more of them (the H100's 2.6M-row grouped-GEMM prefill
        cells, one a pass); attempted, each records its own `oom` in this
        run, at the cost of an allocation."""
        p, impl_id, _version, run_key, _label = entry
        row = verdicts.get(run_key)
        if (row is None or _benchmarks.is_measurement(row)
                or _benchmarks.outcome_of(row) == "oom"):
            return
        path = _launch_path(inv, p.library, impl_id)
        name = _describe_path(path)
        why = (f"canary {inv.op} {inv.dims}: {_benchmarks.outcome_of(row)} — "
               f"{row.get('outcome_detail', '')}")
        disabled.add(path)
        result.disabled[name] = why
        log(f"DISABLE {name}: {why}; its other {cells.get(path, 1) - 1} "
            f"pending cell(s) are not attempted this run")
        if on_disable is not None:
            on_disable(name, why)

    for job_index, job in enumerate(jobs):
        if on_job_start is not None:
            on_job_start(job_index, job)
        inv, provs, deferred = staged[job_index]
        if deferred is not None:
            raise deferred
        if inv is None:
            result.skipped.append(provs)
            continue
        if not provs:
            result.skipped.append(
                f"{inv.op} {inv.dims} {inv.dtype}: no applicable provider "
                f"in this env (libs={list(env.libs)}, cuda={env.cuda_available})"
            )
            continue

        todo = pending(inv, provs)
        # The paths that come up here for the first time: their canary is
        # this cell, or a smaller one later in the plan, attempted now.
        here, ahead = [], {}
        for p, impl_id, *_rest in todo:
            path = _launch_path(inv, p.library, impl_id)
            if path in tried or path in disabled:
                continue
            tried.add(path)
            j = canaries[path][1]
            if j == job_index:
                here.append(path)
            else:
                ahead.setdefault(j, set()).add(path)
        for j, paths in sorted(ahead.items()):
            inv_j, provs_j, _deferred_j = staged[j]
            entries = [e for e in pending(inv_j, provs_j, say=False)
                       if _launch_path(inv_j, e[0].library, e[1]) in paths]
            canary_keys.update(e[3] for e in entries)
            # The sidecar keeps THIS job as where to resume and names the
            # canary's job as the one on the GPU: a death here is recorded
            # against the canary's cell, and nothing before it is lost.
            if on_job_start is not None:
                on_job_start(job_index, jobs[j])
            attempt(inv_j, entries)
            for e in entries:
                judge(inv_j, e)
        if ahead and on_job_start is not None:
            on_job_start(job_index, job)

        kept = []
        for entry in todo:
            if _launch_path(inv, entry[0].library, entry[1]) in disabled:
                result.not_attempted += 1
                continue
            kept.append(entry)
        if not kept:
            continue
        attempt(inv, kept)
        for entry in kept:
            if _launch_path(inv, entry[0].library, entry[1]) in here:
                judge(inv, entry)

    return result


class Band(NamedTuple):
    """How far a correct kernel of a given dtype may sit from the oracle.

    Two numbers because the gate asks two questions (`_cross_check`): how
    big the error is, as its RMS, and whether any single element is
    something other than that noise, as its maximum. Both are MEASURED the
    same way and carry the same margin, which is the only way they can be
    read against each other.
    """
    typical: float
    worst: float


# Per-dtype tolerance: the kernel runs in low precision, so its output differs
# from the fp32 reference by the dtype's numerical error — the gate must allow
# that band. Outside it → the kernel computes the WRONG thing (adapter bug),
# not just precision noise.
#
# Every number here is MEASURED, by `tools.xcheck_floors`, as the floor of
# the dtype itself: the cell's inputs built as the harness builds them, the
# signature's own oracle run on exactly those, and then run again doing what
# the kernel does that the oracle does not — narrowing every intermediate
# where the inputs are already in the dtype, quantising the operands where
# the provider casts them (fp8 and below). Nothing can do better than that,
# so a band under its floor cannot be passed arithmetically, which is what
# the old 0.15 fp8 band was: 45,190 rejections against 1,848 rows admitted,
# a 4% pass rate, the failures piled exactly where the floor is. Widening a
# tolerance to make failures go away is how a harness stops measuring; this
# is the other thing, and the difference is that the number comes from a
# measurement anyone can repeat.
#
# Floors over the signatures the tool walks — gemm, rope, rmsnorm,
# silu_and_mul, GQA, absorbed MLA, sparse MLA, fused_moe, causal_conv1d —
# and the band is the largest of them x2.5:
#
#     dtype       RMS floor   worst floor   band
#     fp16        0.0004      0.0050        (0.001, 0.013)
#     bf16        0.0031      0.0402        (0.008, 0.10)
#     fp8_e4m3    0.0446      1.2353        (0.11,  3.1)
#     fp8_e5m2    0.0646      2.7954        (0.16,  7.0)
#
# The RMS floor is a property of the DTYPE (bf16 lands between 0.0010 and
# 0.0031 whatever the op) while the maximum's distance from it is a property
# of the OP — 6x the RMS for a GEMM, 13x for a MoE, 38x for GQA attention.
# That is why one band per dtype covers every signature and the eight
# per-signature multipliers that used to stand here are gone rather than
# retuned: they were correcting a maximum read against a constant, which is
# also what made four of six identical reruns of one sparse-MLA cell flip
# verdict.
#
# fp32's pair is not a round-trip floor — there is none, the narrowing is an
# identity — but an allowance for a kernel summing in a different order than
# the oracle.
#
# Two things these bands do NOT separate, stated so they are not discovered
# as surprises. A chunked recurrent scan's floor is not measurable this way
# at bf16 (the model narrows a state every implementation keeps in fp32, and
# reads 12 where the kernels sit near 0.002), so those four signatures ride
# on the bands the other nine set; their recorded position under the old
# gate (0.022-0.060 on a metric that runs about 2x the maximum here) is well
# inside. And a kernel running NVFP4 where the cell declares fp8 has a floor
# of 0.074 at a pointwise op, under the fp8 band, so that one is caught on
# the reduction-heavy ops and nowhere else — a mis-declared precision rather
# than a wrong function, whose door is the symbol the row records.
_XCHECK_BANDS = {
    "fp32": Band(1e-4, 1e-3),
    "fp16": Band(1e-3, 1.3e-2),
    "bf16": Band(8e-3, 1e-1),
    "fp8_e4m3": Band(1.1e-1, 3.1),
    "fp8_e5m2": Band(1.6e-1, 7.0),
}

# Each band is keyed by the dtype token a cell is measured at, and that is
# checked here at import against the dtype declaration: a band keyed at a
# spelling no cell can carry (`fp8`, `e4m3`, `float16`) would never be read
# and every cell of that dtype would be judged on bf16's band instead
# (`_xcheck_band`'s default), which is precisely the silent gate-wider-than-
# the-kernel this file exists to avoid.
for _token in _XCHECK_BANDS:
    if _token not in CORPUS_TOKENS:
        raise ValueError(
            f"cross-check band keyed at {_token!r}, which is not a dtype a "
            f"cell can be measured at — use a corpus token "
            f"(`backend.arch.dtypes.CORPUS_TOKENS`)")
del _token


def _band_for(dtype: str, *, role: str = "dtype") -> Band:
    """The measured band for `dtype`, refusing a dtype the table does not carry.

    bf16 is not a default: a new activation dtype without a measured band must
    stop the cross-check, not silently be judged as though it were bf16.
    """
    try:
        return _XCHECK_BANDS[dtype]
    except KeyError:
        raise ValueError(
            f"no cross-check band for {role} {dtype!r}; add a measured "
            f"entry to kernel_bench.driver._XCHECK_BANDS — bf16 is not "
            f"a fallback") from None


def _fits_after_release(env: Env, error: BaseException) -> bool:
    """Release what this process is holding and say whether the card looks
    able to take the allocation that failed after all.

    A refusal is only worth recording if it is about the CELL rather than
    about the state this process left the card in. The comment a few lines
    up says what that state can be: an input builder that raises part way
    through has already allocated, and left there it accumulates across
    jobs. So the allocator is emptied and the free memory re-read; a card
    that now has room for what was asked had been fragmented or crowded,
    and the cell is tried again rather than written down as impossible —
    once more in this run, whether a provider's build or the cell's inputs
    ran out, and the second out-of-memory is the `oom`.
    False — still no room — is the case that gets recorded.

    Room for what was ASKED (`_requested_bytes`), not room in general: the
    release cannot return what the cell itself still holds — its inputs
    stay alive through every provider — so free memory after it says
    nothing on its own. The H100 had 38 GiB free after a provider asked for
    40 GiB beside the cell's own 40 GiB of inputs; read as "a tenth of the
    card is free", that OOM was recorded as an `error` and its canary
    disabled the path. Only an allocation failure that does not say its
    size falls back to that line.
    """
    if not env.cuda_available:
        return False
    import torch
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return False
    asked = _requested_bytes(error)
    if asked is not None:
        return free >= asked
    # A tenth of the card is the line: below it this process was not what
    # stood in the cell's way, above it there is room worth another attempt.
    return free > total * 0.10 and free > (1 << 30)


def _requested_bytes(error: BaseException) -> Optional[int]:
    """The size of the allocation that failed, as torch's allocator states
    it ("Tried to allocate 40.00 GiB"), through the exception chain; None
    when no error in the chain says it.

    The units are the ones torch's caching allocator prints (`format_size`,
    c10/cuda/CUDACachingAllocator.cpp); the pattern is built from them, so
    the words this reads and the words it can convert cannot drift apart.
    """
    import re
    units = {"bytes": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
    pattern = r"Tried to allocate ([0-9.]+) (%s)" % "|".join(units)
    seen = set()
    cursor: Optional[BaseException] = error
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        found = re.search(pattern, str(cursor))
        if found:
            return int(float(found.group(1)) * units[found.group(2)])
        cursor = cursor.__cause__ or cursor.__context__
    return None


def _is_out_of_memory(error: BaseException, env: Optional[Env] = None) -> bool:
    """Whether this exception is the CARD refusing the shape.

    Only on a GPU: on a CPU-only box the same exception is the host running
    out of RAM, and recording it would publish a row claiming a card that
    was never involved could not hold the cell.

    Type first, and through the exception chain — a provider that wraps the
    failure still raised an allocation failure — then the library's own
    wording. The type test is `isinstance`, not a name comparison, so a
    subclass counts and another library's unrelated `OutOfMemoryError` does
    not.
    """
    if env is not None and not env.cuda_available:
        return False
    chain = []
    seen = set()
    cursor: Optional[BaseException] = error
    while cursor is not None and id(cursor) not in seen:
        seen.add(id(cursor))
        chain.append(cursor)
        cursor = cursor.__cause__ or cursor.__context__
    try:
        import torch
        oom_types: tuple = (torch.cuda.OutOfMemoryError,)
    except Exception:
        oom_types = ()
    # Allocation failures a library reports in its own words rather than as
    # torch's `OutOfMemoryError`: cuBLAS and cuDNN name their status codes,
    # the driver API its error, and Triton its shared-memory budget. All of
    # them mean the same thing — this launch does not fit on this card — and
    # all of them were being rediscovered every pass.
    wordings = ("out of memory", "cublas_status_alloc_failed",
                "cudnn_status_alloc_failed", "cudaerrormemoryallocation",
                "cuda_error_out_of_memory", "out of resource")
    for item in chain:
        if oom_types and isinstance(item, oom_types):
            return True
        text = str(item).lower()
        if any(token in text for token in wordings):
            return True
    return False


def _launch_path(inv, library: str, impl_id: str) -> tuple:
    """What one implementation runs as on a class of cells, whatever their
    size: the implementation, the signature, the phase and the precision.
    A provider's build and launch differ along these (a prefill and a
    decode build are different calls; an fp8 cell takes another path than
    a bf16 one), and a run's canary asks once per path.

    Treating the path as failing together is an ASSUMPTION, true of what
    the canary exists for (a moved import, a renamed symbol, a changed
    signature) and not of everything: `_symbol_group` keeps the size band
    because a dispatch picks differently at 1k and at 1M, and a failure only
    the smallest cell hits (a tile it does not divide, an edge branch) stops
    the whole path for that run. The cost is bounded — the path's other
    cells stay pending, not recorded, and the run exits not-done
    (`cli.EXIT_PATHS_DISABLED`) so the next pass asks again."""
    return (library, impl_id, inv.signature, inv.phase,
            tuple(sorted(inv.dtype_env.items())))


def _describe_path(path: tuple) -> str:
    library, impl_id, signature, phase, dtype_env = path
    dtypes = ",".join(f"{k}={v}" for k, v in dtype_env)
    return f"{library}:{impl_id} {signature} {phase} [{dtypes}]"


def _cell_elements(inv) -> float:
    """How big a cell is, for choosing a path's smallest: the elements of
    the tensors the walk published for it (`_declared_elements`). A cell
    that published none is not known to be small, so it sorts last."""
    return sum(_declared_elements(inv)) or math.inf


def _symbol_group(inv, library: str, impl_id: str) -> tuple:
    """What one profiler observation stands for.

    A substitution — FlashKDA taking over `chunk_kda`, a vLLM CustomOp
    running its native torch path — is a property of the implementation on a
    class of cells, not of one cell. The class is the implementation, the
    phase, the precision and the SIZE BAND: a library's dispatch heuristic
    picks a different kernel at 1k and at 1M, so the band is the power of two
    of the cell's largest extent, which keeps that signal while collapsing
    the dozens of cells inside each band to one observation.
    """
    largest = max((int(v) for v in inv.dims.values()
                   if isinstance(v, (int, float)) and v > 0), default=1)
    return (*_launch_path(inv, library, impl_id), largest.bit_length())


def _declared_elements(inv) -> Tuple[int, ...]:
    """How many elements each tensor the WALK published for this cell holds
    — its operands and its output, as the plan carried them. The launch's
    own largest operand is read against the largest of these
    (`launch_trace.disagreement`), so a build that slices, or one that
    forgets to expand, is a row that never gets written.

    The engine spells one operand as a bare shape (`[B, s_query, d_model]`)
    and several as a list of shapes, which is the convention `_operand_list`
    already reads — so it is asked rather than re-read here. Iterating the
    raw field instead saw a single operand's EXTENTS as its tensors, found
    no list among them, and returned nothing: with no declared element
    count the gate asks nothing, and the half of it that catches a sliced
    or unexpanded launch was inert for every signature with one input
    (rmsnorm, every activation, the cast this comment was written for)."""
    from .refs import _operand_list
    out = []
    for side in (_operand_list(inv.shape_in), _operand_list(inv.shape_out),
                 (inv.shape_weight,)):
        for tensor in (side or ()):
            if not isinstance(tensor, (list, tuple)):
                continue
            if tensor and isinstance(tensor[0], (list, tuple)):
                continue                       # a nested list of tensors
            count = 1
            for extent in tensor:
                try:
                    count *= int(extent)
                except (TypeError, ValueError):
                    count = 0
                    break
            if count > 0:
                out.append(count)
    return tuple(out)


def compared_records(inputs: Any) -> Dict[str, ComparedRecord]:
    """`{dtype slot: record}` for every record the oracle writes itself, so
    that kernel and oracle are compared record to record on it: the format
    slots (`benchmark_identity.FORMAT_SLOTS` — `output_format`, `kv_format`)
    its inputs were handed, the way each record oracle reads which record to
    write (`refs.rows_on_record`), and a quantised activation output as the
    `ActivationCast` the reference and provider both read. A cell that names
    a record its oracle does not write — an attention reading an FP8 cache
    against the fp32 function — is not in it, and stays banded by that
    slot's dtype (`_xcheck_dtype`)."""
    from backend.inference.benchmark_identity import FORMAT_SLOTS
    if not isinstance(inputs, Mapping):
        return {}
    records = {dtype_slot: inputs[slot]
               for slot, dtype_slot in FORMAT_SLOTS.items()
               if isinstance(inputs.get(slot), str)}
    casts = [value for value in inputs.values()
             if isinstance(value, _quant.ActivationCast)]
    if len(casts) > 1:
        raise ValueError(
            f"the inputs carry {len(casts)} activation casts; one output "
            "has one grid")
    if casts:
        records["activation"] = casts[0]
    return records


def _xcheck_basis(inv, inputs: Any) -> Tuple[str, Tuple[ComparedRecord, ...]]:
    """`(dtype, records)` a cell's cross-check is judged on: the band's
    dtype (`_xcheck_dtype`) less the slots its oracle writes a record for,
    and those records (`_xcheck_band`). One answer for the driver and
    smoke."""
    compared = compared_records(inputs)
    return (_xcheck_dtype(inv, compared=tuple(compared)),
            tuple(compared.values()))


def _xcheck_dtype(inv, compared: Tuple[str, ...] = ()) -> str:
    """The dtype whose band a cell's cross-check reads: the least precise of
    the operands the kernel reads (its dtype env's activation, weight and kv
    slots). The activation alone undersold an attention over an fp8 cache —
    FlashMLA's fp8 sparse decode and its torch floor both sat at cosine
    0.9997 against the reference and were rejected on the bf16 band.

    A slot in `compared` (`compared_records`) is a record the oracle writes
    too, so the two agree value for value on it and its dtype is no slack
    either owes the other: DeepSeek-V4's q/KV insert writes the FlashMLA
    record on its `kv` port, and its oracle writes the same one."""
    _band_for(inv.dtype, role="activation dtype")
    slots = [inv.dtype_env.get(s) for s in ("activation", "weight", "kv")
             if s not in compared]
    banded = [d for d in slots if d in _XCHECK_BANDS] or [inv.dtype]
    return max(banded, key=lambda d: _XCHECK_BANDS[d].typical)


def _act_torch_dtype(inv):
    """The activation dtype as torch sees it — a kernel that stores its scale
    in the activation dtype grids differently on a bf16 cell than an fp16 one."""
    from .refs import torch_dtype
    return torch_dtype(inv.dtype)


def _record_step(record: ComparedRecord) -> float:
    """One step of a compared record's element grid, whichever registry
    states it: a cache format through `refs.record_step`, an activation
    through the `ActivationCast` the reference and provider both read."""
    if isinstance(record, _quant.ActivationCast):
        return record.format.step
    from .refs import record_step
    return record_step(record)


def _xcheck_band(dtype: str,
                 records: Tuple[ComparedRecord, ...] = ()) -> Band:
    """The band a cell's cross-check reads: its dtype's (`_xcheck_dtype`),
    and, for a cell whose launch writes cached records its oracle writes
    too (`compared_records`: the `output_format` it outputs, the
    `kv_format` it writes beside), a worst element no tighter than one step
    of the coarsest of those records' grids (`_record_step`).

    The oracle of such a cell writes the same record, so a correct kernel
    sits on it value for value — except where the two computed the value
    before the write a hair apart and it sat on a rounding boundary: one
    step there, never more (the decoded FP8 value is within `eps · |ref|`
    of the oracle's). The typical error stays the dtype's, and that is what
    separates a wrong record: an fp32 row compared to the record, a scale
    rounded to nearest instead of up, each sit 1.4-2.7x over it (measured
    on sglang's and vLLM's V4 stores, `TestFlashMLARecordOracle`)."""
    band = _band_for(dtype)
    if not records:
        return band
    return Band(band.typical,
                max(band.worst, *(_record_step(r) for r in records)))


def _cross_check(case, ref_out, dtype: str, *, trace=None,
                 records: Tuple[ComparedRecord, ...] = ()):
    """Return `(verdict, rel, cos, typical, worst)`: the worst normalized
    error if the kernel DISAGREES with the reference (outside the dtype's
    band) else None, then the two diagnostic numbers and THE TWO THE VERDICT
    WAS MADE ON. Comparison is in fp32 and layout-normalized via the
    provider's `to_canonical`.

    `typical` and `worst` come back on both paths because they are the
    verdict's basis: a row that records only `rel` and `cos` cannot say what
    it was judged on or how close it came, and both of those are measured
    differently from the band. Two readers independently mistook `rel` for
    the margin before the row carried these.

    `trace` records what the ONE call it makes dispatched — the launch, not
    the comparison's own arithmetic, which is why the recorder goes here
    and not around this function."""
    import torch
    produced = trace.run(case.run) if trace is not None else case.run()
    out = case.to_canonical(produced)
    out = (out[0] if isinstance(out, (tuple, list)) else out).float()
    ref = ref_out.float()
    if out.shape != ref.shape:
        # layout mismatch = definite disagreement
        return float("inf"), float("inf"), 0.0, float("inf"), float("inf")
    # Per element, the error over what the dtype may cost it: `|ref_i|` for
    # its own rounding, plus the output's mean-abs magnitude for the
    # cancellation a reduction centred near zero brings (a fixed atol cannot
    # stop the near-zero entries from failing a numerically-correct
    # low-precision kernel, |err|/|ref| → ∞ as ref → 0; outputs range O(1)
    # (rmsnorm) … O(100) (large-K GEMM)).
    band = _xcheck_band(dtype, records)
    scale = ref.abs().mean().clamp_min(1e-6)
    delta = (out - ref).abs()
    error = delta.div(ref.abs() + scale)
    # TWO questions, because one number cannot answer both. How big the
    # kernel's error is — a question about all of it, so read against the
    # dtype's own noise — and whether any single element is something other
    # than that noise, which only a maximum sees. Averaging alone would hide
    # one wrong row among a million right ones; a maximum alone made the
    # verdict a function of the cell's SIZE, because the largest of more
    # samples is larger: a correct bf16 rope failed by being handed more
    # tokens (0.0174 at 1024, 0.0288 at 32768, cosine 1.00000 throughout,
    # 97 cells per LLaMA-class model refused and the run marked rc=1), and
    # four of six identical reruns of one sparse-MLA cell used to flip
    # verdict. The RMS has no such drift: measured across 14 signatures it
    # moves by 1.2x where the maximum moves by 2.4x.
    typical = error.pow(2).mean().sqrt().item()
    worst = error.max().item()
    rel = delta.max().item() / scale.item()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0).item()
    if typical <= band.typical and worst <= band.worst:
        return None, rel, cos, typical, worst
    # Failed. Both reported numbers come back either way, because a verdict
    # alone throws away what made it. `rel` is max|delta| / mean|ref|, which
    # runs large on a reduction; `cos` does not. Together they separate
    # "lossy but correct" from "computing the wrong thing": fp8 sits at rel
    # 0.24 with cos 0.9993, and reading only the first is what kept the fp8
    # tolerance below the dtype's own floor for as long as it was.
    return worst, rel, cos, typical, worst
