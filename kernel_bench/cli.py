"""kernel_bench CLI — `python -m kernel_bench {up,plan,sweep,run,run-all,report,check,…}`.

up     (any) : clone → working bench — resolve the box profile, build the
               pip envs, self-check with smoke.
plan   (CPU) : enumerate measurement jobs for a (model, config) sweep → jobs.jsonl
run    (GPU) : in a provider env, measure every applicable provider per job,
               validate + emit measured CSV rows.
report (any) : coverage + envelope + gap list for a jobs file against the corpus.
check  (any) : MFU / MBU / allclose on measured CSVs — same physics as the
               write gate, applied to a file after the fact.
restamp (any): rewrite achieved_* from a new jobs.jsonl and each row's p50.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import sys
from typing import Callable, Dict, List


def _new_run_id() -> str:
    """One id per sweep batch. Contamination is revoked per batch, so the id
    only has to be unique and human-sortable — not meaningful."""
    import os
    import socket
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{socket.gethostname()}-{os.getpid()}"


def _cmd_plan(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="kernel_bench plan")
    ap.add_argument("model_id")
    ap.add_argument("--batch", default="1", help="comma list, e.g. 1,8,64")
    ap.add_argument("--seq", default="2048", help="comma list, e.g. 2048,8192")
    ap.add_argument("--phase", default="prefill",
                    choices=["prefill", "decode", "encode"])
    ap.add_argument("--media", default="",
                    help="encode phase: the media each request carries, JSON "
                         "by model input, e.g. '{\"pixel_values\": "
                         "{\"n_img\": 1, \"img_h\": 672, \"img_w\": 672}}' "
                         "(processed sizes; --seq is not read)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--framework", default="vllm")
    ap.add_argument("--gpu", default=None,
                    help="the SKU a box plan is for; omitted produces a "
                         "cardless shape-inspection plan, never a queue plan")
    from backend.inference.sweep_grid import ROUTINE_OUTPUT_TOKENS
    ap.add_argument("--output-tokens", type=int, default=ROUTINE_OUTPUT_TOKENS,
                    help="the served request's output length; decode runs over "
                         "the cache at seq + half of it (default: the routine "
                         f"grid's {ROUTINE_OUTPUT_TOKENS})")
    ap.add_argument("-o", "--output", default="jobs.jsonl")
    args = ap.parse_args(argv)
    media = json.loads(args.media) if args.media else None

    from .plan import (build_jobs_with_unplaced, unplaced_by_reason,
                       unplaced_path, write_plan)

    batches = [int(x) for x in args.batch.split(",") if x.strip()]
    seqs = [int(x) for x in args.seq.split(",") if x.strip()]
    all_jobs = []
    unplaced = []
    seen_unplaced = set()
    for b in batches:
        for s in seqs:
            # NO `binding=`, so `build_jobs` makes its own undeployed one —
            # and a FUSION WHOSE CONDITION NAMES THE CARD THEN DOES NOT
            # MATCH. The plan this prints is therefore not the plan a sweep
            # on a GPU box prints: it re-ranks to the naive members and says
            # nothing about having done so.
            #
            # Worked example, DeepSeek-V4-Flash prefill, framework vllm: the
            # hyper-connection seam's four spans (`mhc_post`,
            # `mhc_pre_gemm`, `mhc_pre_big_fuse`, `mhc_fused_post_pre`) are
            # absent here and `hc_mix` + `hc_split_sinkhorn` stand in their
            # place; under `get_binding("vllm").deployed(ServingDeployment
            # .at(gpu_sku="H100-SXM", ...))` the four appear and the two
            # naive ops do not. Same model, same phase, same framework.
            #
            # So: this command is for eyeballing shapes, not for deciding
            # what a box will measure. Anything that must agree with serving
            # passes a DEPLOYED binding — `sweep` does (it is the path the
            # queue and `enqueue` take), and so does anything comparing
            # against a deployment's own measurement keys.
            jobs, _drp, records = build_jobs_with_unplaced(
                model_id=args.model_id, batch_size=b, seq_len=s,
                phase=args.phase, avg_output_tokens=args.output_tokens,
                tp=args.tp, framework=args.framework, media=media,
                gpu=args.gpu,
            )
            all_jobs.extend(jobs)
            for record in records:
                key = tuple(sorted(record.items()))
                if key not in seen_unplaced:
                    seen_unplaced.add(key)
                    unplaced.append(record)
    write_plan(all_jobs, args.output, unplaced=unplaced)  # unjudged
    print(f"wrote {len(all_jobs)} jobs → {args.output} "
          f"({len(batches)}×{len(seqs)} shape cells, phase={args.phase}, tp={args.tp})")
    print(f"plan scope: {args.gpu or 'cardless shapes'}")
    if unplaced:
        print(f"structural records → {unplaced_path(args.output)}:")
        for line in unplaced_by_reason(unplaced):
            print(f"  {line}")
    return 0


def _add_workers(ap: argparse.ArgumentParser) -> None:
    """`--workers`: the processes a sweep plans its deployments in
    (`build_sweep_jobs(workers=…)`). A routine MoE is ~28 deployments of
    ~7s each, so one process takes minutes per model."""
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="processes to plan deployments in (default: "
                         "min(8, cores); 1 = in this process)")


def _cmd_sweep(argv: List[str]) -> int:
    from .sweep import sweep_surfaces
    surfaces = sweep_surfaces()
    ap = argparse.ArgumentParser(prog="kernel_bench sweep")
    ap.add_argument("model_id")
    ap.add_argument("--gpu", default=None,
                    help="GPU SKU (gates cells by its VRAM). Default: the box "
                         "profile's card; RTX-4090 with no profile. A --gpu "
                         "that is not the box's card is refused")
    ap.add_argument("-o", "--output", default="jobs.jsonl")
    ap.add_argument("--framework", default="vllm")
    ap.add_argument("--max-seq", type=int, default=0,
                    help="cap the routine context grid at this seq len (0 = full "
                         "1K…1M). Use for a fast breadth pass — the biggest seq "
                         "cells (256K/1M) are O(S²) and mostly OOM on small cards.")
    ap.add_argument("--tracks", choices=sorted(surfaces), default="routine",
                    help="the surface whose points to plan: its track table, "
                         "contexts and deployments (routine = Inference "
                         "Scaling; frontier = gallery decode Pareto)")
    ap.add_argument("--output-tokens", type=int, default=None,
                    help="the served output length (default: the surface's own)")
    ap.add_argument("--box", default=None, metavar="PROFILE",
                    help="box profile the plan is for (default: "
                         "$KERNEL_BENCH_BOX, then envs/boxes/<hostname>.toml). "
                         "With one, jobs no env of that box would measure go "
                         "to <output>.no_provider instead of the plan")
    ap.add_argument("--envs-dir", default="kernel_bench/envs")
    _add_workers(ap)
    args = ap.parse_args(argv)

    from .box import BoxEnvs, BoxProfileError, find_box_profile
    from .plan import (gaps_by_signature, gaps_path, unplaced_by_reason,
                       unplaced_path, write_plan)
    from .sweep import plan_surface

    # The box first: its card is the one the cells are gated and the
    # kernels judged on.
    try:
        profile = find_box_profile(args.box, args.envs_dir)
        box = (BoxEnvs.of(profile, args.envs_dir, gpu=args.gpu)
               if profile else None)
        if box is not None:
            if not box.gpu:
                # A box whose card nobody can name: neither the grid nor the
                # kernels may be judged on a default card.
                raise BoxProfileError(
                    f"box {profile.name} names no GPU SKU; pass --gpu")
            box.envs()          # its env files exist, before hours of planning
    except BoxProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.gpu = box.gpu if box else (args.gpu or "RTX-4090")
    jobs, stats = plan_surface(
        args.model_id, args.gpu, args.tracks, framework=args.framework,
        max_seq=args.max_seq, output_tokens=args.output_tokens,
        log=lambda m: print(" ", m), workers=args.workers)
    output_tokens = stats["output_tokens"]

    jobs, gaps = write_plan(jobs, args.output, box,
                            unplaced=stats.get("unplaced") or [])
    if box is None:
        print("no box profile: every job written, none checked against a "
              "box's envs (pass --box)")
    print(f"wrote {len(jobs)} jobs → {args.output} "
          f"({stats['cells']} cells, {stats['parallel_configs']} parallel configs, "
          f"{stats['seqs']} seqs, tracks={args.tracks}, "
          f"output={output_tokens}, gpu={args.gpu})")
    if gaps:
        print(f"no provider on {profile.name} — {len(gaps)} job(s) → "
              f"{gaps_path(args.output)}:")
        for line in gaps_by_signature(gaps):
            print(f"  {line}")
    # What this sweep will not send to a GPU, and why. Zero-launch rows are
    # accounted as measured@0; the other reasons name the model/signature or
    # projection to repair rather than a provider corpus gap.
    if stats.get("unplaced"):
        print(f"structural records → {unplaced_path(args.output)}:")
        for line in unplaced_by_reason(stats["unplaced"]):
            print(f"  {line}")
    return 0


def _cmd_run(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="kernel_bench run")
    ap.add_argument("--jobs", required=True, help="jobs.jsonl from `plan`")
    ap.add_argument("-o", "--output", default="results.csv")
    ap.add_argument("--contributor", default="@kernel-bench")
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    ap.add_argument("--run-id", default=None,
                    help="batch id stamped on every row (revocation unit). "
                         "Default: generated per invocation")
    ap.add_argument("--box", default=None,
                    help="box profile name stamped on every row. Default: "
                         "$KERNEL_BENCH_BOX when set, else empty")
    ap.add_argument("--no-cross-check", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip jobs already measured in this output file, a "
                         "--measured-in CSV or Kernel Bench's shared rows "
                         "(HUGGINGARCH_BENCH_URL; measure-once; also keeps the file)")
    ap.add_argument("--wait-for-gpu", action="store_true",
                    help="queue behind another run holding this GPU instead of "
                         "refusing (default: refuse — a long sweep rarely wants "
                         "to block for hours)")
    ap.add_argument("--no-gpu-lock", action="store_true",
                    help="do not take the per-GPU exclusive lock. Only for a "
                         "device you know is otherwise idle: concurrent runs "
                         "make every latency wrong and the rows do not record it")
    ap.add_argument("--append-output", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--measured-in", action="append", default=[],
                    metavar="CSV",
                    help="another results CSV whose rows count as measured "
                         "(repeatable). run-all passes the envs before this "
                         "one and the queue every other model's results")
    ap.add_argument("--library", default=None,
                    help="the library this env was built for: measure only "
                         "the providers registered under it (run-all passes "
                         "it, so a provider several envs can import — every "
                         "torch floor, flashinfer in the sglang env — is "
                         "measured in its own env only). Default: every "
                         "provider importable here")
    ap.add_argument("--progress-file", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    import os

    from backend.inference import benchmarks as _benchmarks
    from backend.inference.benchmarks import BenchmarkValidationError, validate_row
    from .driver import CudaContextLost, run_jobs
    from .emit import (IncrementalCsvWriter, above_sol, existing_run_keys,
                       resume_from_output, resumed_verdict)
    from .env import detect_env
    from .plan import read_jobs_jsonl
    from .supervisor import write_progress

    if args.library is not None:
        # A name the registry does not know owns no provider, so the run
        # would measure nothing, write a header-only file and exit 0 — which
        # the queue reads as this env done. Refuse it before the lock and the
        # output file instead.
        from .providers import get_library
        try:
            get_library(args.library)
        except KeyError as e:
            print(f"error: --library: {e.args[0]}", file=sys.stderr)
            return 2
    env = dataclasses.replace(detect_env(), library=args.library)
    print(f"env: library={env.library or 'any'} torch={env.torch_version} "
          f"cuda={env.cuda_available} gpu={env.device_name} libs={env.libs}")
    jobs = read_jobs_jsonl(args.jobs)

    # Persist each row as it is measured. A sweep is hours of GPU time, and
    # holding every row until the run ended meant a kill / OOM / dropped
    # session threw all of it away — one observed run had 20 hours in memory
    # and nothing on disk. Validation stays PER ROW (a malformed row is
    # dropped and reported, never failing the sweep); only valid rows reach
    # the file, so what is on disk is always usable as-is.
    # One sweep per physical card. Taken BEFORE the output file is opened so a
    # refused run doesn't truncate a file it will never write.
    #
    # EVERY card this run can see, not only the first: a collective cell
    # runs a rank on each of them (`kernel_bench.comm`), and a run on any of
    # them would be in its number. A one-card queue sees one card.
    #
    # All or none, in UUID order, so two runs that see the same cards in a
    # different order cannot each hold one and wait for the other; waiting
    # (`--wait-for-gpu`) lets go of everything between tries rather than
    # holding one card idle while the other is busy.
    import time

    from .gpulock import GpuBusy, GpuLock
    locks = sorted((GpuLock(device=i, enabled=not args.no_gpu_lock)
                    for i in range(max(1, env.devices or 0))),
                   key=lambda lock: lock.path or "")
    while True:
        try:
            for lock in locks:
                lock.acquire()
            break
        except GpuBusy as e:
            for lock in locks:
                lock.release()
            if not args.wait_for_gpu:
                print(f"\nREFUSED: {e}")
                return 2
            time.sleep(10)
    for lock in locks:
        if lock.active:
            print(f"gpu lock: held ({lock.path})")
        if lock.preflight_warning:
            print(f"WARNING: {lock.preflight_warning}")

    bad: list = []
    append, measured_keys = resume_from_output(
        args.output, skip_existing=args.skip_existing,
        append=args.append_output)
    if append and measured_keys:
        print(f"resuming {len(measured_keys)} identities from {args.output}")
    for other in args.measured_in:
        if os.path.isfile(other) and os.path.getsize(other) > 0:
            keys = existing_run_keys(other)
            measured_keys = set(measured_keys) | keys
            print(f"{len(keys)} identities already measured in {other}")
    writer = IncrementalCsvWriter(args.output, append=append)

    def _persist(row):
        try:
            validate_row(row)
        except BenchmarkValidationError as e:
            # Report at the point of loss, not only in the end-of-run summary.
            # A row that measured fine and then failed validation is GPU time
            # already spent, and the summary is exactly what a crashing run
            # never reaches: `causal_conv1d` lost every row it ever measured
            # this way — 798 planned cells, four kernels, zero rows — because
            # `d_conv_channels` was missing from the shape-symbol glossary and
            # the only notice was printed after a run that kept dying first.
            bad.append((row, str(e)))
            print(f"  DROP {row.get('kernel_impl')} {row.get('op')} "
                  f"{row.get('shape_json')}: {e}")
            return
        # The SOL gate: a row whose claimed work exceeds the card's own peaks
        # timed one thing and counted another. Rejected at the same point of
        # loss as a schema violation, for the same reason.
        why = above_sol(row)
        if why is not None:
            bad.append((row, why))
            print(f"  DROP {row.get('kernel_impl')} {row.get('op')} "
                  f"{row.get('shape_json')}: above SOL — {why}")
            return
        # The third gate, and the file's own: a row it already holds under
        # this identity. Reported here with the others so the run says which
        # cells collapsed onto one row rather than leaving a file no later
        # run can read.
        repeated = writer.write(row)
        if repeated is not None:
            repeated = f"{repeated}; {resumed_verdict(row, measured_keys)}"
            bad.append((row, repeated))
            print(f"  DROP {row.get('kernel_impl')} {row.get('op')} "
                  f"{row.get('shape_json')}: {repeated}")

    current: dict = {}
    disabled: dict = {}

    def _job_start(index, job):
        current.update(index=index, job=job)
        if args.progress_file:
            write_progress(args.progress_file, index, job, disabled=disabled)

    def _attempt(rows):
        # Which implementation of the job is in flight, so a supervisor that
        # finds this process dead or stalled can record THAT one — not the
        # whole job, and not one that already finished.
        if args.progress_file and current:
            write_progress(args.progress_file, current["index"], current["job"],
                           in_flight=rows, disabled=disabled)

    def _disable(path, why):
        # Into the sidecar at once, so `run-all` lists it even if this
        # process dies before its summary.
        disabled[path] = why
        if args.progress_file and current:
            write_progress(args.progress_file, current["index"], current["job"],
                           disabled=disabled)

    try:
        res = run_jobs(jobs, env=env, date=args.date, contributor=args.contributor,
                       run_id=args.run_id or _new_run_id(),
                       box=args.box if args.box is not None
                       else os.environ.get("KERNEL_BENCH_BOX", ""),
                       cross_check=not args.no_cross_check,
                       skip_existing=args.skip_existing,
                       log=lambda m: print(" ", m), on_row=_persist,
                       on_job_start=_job_start, measured_keys=measured_keys,
                       on_attempt=_attempt, on_disable=_disable)
    except CudaContextLost as e:
        # Not a row here: the progress sidecar names the implementation, and
        # `run-all` records it as a `crash` and restarts the rest.
        print(f"\nSTOPPED: {e}", file=sys.stderr)
        return 1
    finally:
        writer.close()
        # The group's ranks leave their cards before the cards are unlocked:
        # the next run's lock would otherwise find them there.
        from .comm import Group
        Group.release()
        for lock in locks:
            lock.release()

    from collections import Counter
    outcomes = Counter(_benchmarks.outcome_of(r) for r in res.rows)
    print(f"\nmeasured {outcomes.pop('measured', 0)} rows | "
          + "".join(f"{o} {n} | " for o, n in sorted(outcomes.items()))
          + f"skipped {len(res.skipped)} | "
          f"failed {len(res.failures)} | "
          f"launch unobserved {len(res.unobserved)} | "
          f"disabled {len(res.disabled)} launch paths "
          f"({res.not_attempted} cells not attempted)")
    for s in res.skipped[:10]:
        print("  skip:", s)
    for f in res.failures[:10]:
        print("  FAIL:", f)
    for path, why in res.disabled.items():
        print(f"  DISABLED: {path} — {why}")

    if bad:
        print(f"\ndropped {len(bad)} rows that failed schema validation:")
        for r, e in bad[:10]:
            print(f"  drop {r.get('model')} {r.get('op')} {r.get('kernel_impl')}: {e}")
    if not writer.has_rows:
        print("\nno valid rows measured (this env runs no applicable provider, "
              "or every row failed validation). Output holds a header only.")
        return 0 if not res.rows else EXIT_NO_VALID_ROWS

    # Structural guard (header + duplicate identity) read back from what was
    # actually persisted — the rows are already on disk, so this reports a
    # structural problem rather than deciding whether to keep them.
    try:
        with open(args.output, encoding="utf-8") as f:
            _benchmarks.validate_text(f.read(), source=args.output)
    except BenchmarkValidationError as e:
        print(f"\nWARNING: {e}\n  (rows are individually valid and already written)")
    print(f"\nwrote {writer.count} validated rows → {args.output}"
          + (f" ({len(bad)} dropped)" if bad else ""))
    return EXIT_PATHS_DISABLED if res.disabled else 0


# uv's own default is 30s per download, which is not enough for these envs: a
# provider env pulls torch and its CUDA runtime wheels, several of them hundreds
# of MB (nvidia-cudnn alone is ~350 MB). On a slow link every env failed with
# "Failed to download distribution due to network timeout" before measuring
# anything. Raised here rather than left to the operator because the size of
# these wheels is a property of the envs, not of one machine; `UV_HTTP_TIMEOUT`
# still wins if it is already set.
UV_HTTP_TIMEOUT_S = "600"

#: `run`'s exit when it reached the end of its jobs and every row it measured
#: failed validation. Not a crash: `run-all` reads any other non-zero exit as
#: one and quarantines the job in flight — which, at the end of a run, is an
#: innocent last job — and then fails the env, so the queue never marks the
#: model done and measures it again every pass.
EXIT_NO_VALID_ROWS = 3

#: `run`'s exit when it reached the end of its jobs and a launch path's
#: canary disabled that path: its other cells were not attempted and carry
#: no row. Reaching the end, so `run-all` quarantines nothing; but not done,
#: so `run-all` exits with it too and the queue does not mark the model
#: `.done` — the next pass runs it again, which is one canary per path while
#: the env stays broken, and the whole path once it is fixed.
EXIT_PATHS_DISABLED = 4


def uv_run_cmd(env_file: str, run_args: List[str], *,
               python_version: str | None = None,
               constraints: str | None = None,
               no_build_isolation_for: str | None = None) -> List[str]:
    """The `uv run` argv that measures one provider env: uv resolves + caches
    that env's deps (torch + the one kernel lib, from `env_file`) — no manual
    venv — and runs `python -m kernel_bench run …` inside it. The harness
    itself comes from source via PYTHONPATH (set by the caller), so the env
    stays minimal. `--no-project` keeps uv from installing huggingarch's own
    deps on top.

    `constraints` is a pip constraints file the CALLER supplies — and which
    one, for this env, is `box.library_requirements`'s answer rather than this
    function's business. The env files ask only for `torch>=2.5`, so uv
    resolves the newest build, which on a box whose driver/CUDA stack is pinned
    (an NGC image, a source-built flashinfer) is the wrong one, and silently
    so: the wheels install and then mismatch at runtime. Which torch a box can
    run is a fact about that box, so it is supplied per-run instead of pinned
    in the repo."""
    cmd = ["uv", "run", "--no-project"]
    if python_version:
        cmd += ["--python", python_version]
    if constraints:
        cmd += ["--with-requirements", constraints]
    # flash-attn's setup.py imports torch at build time; uv's default
    # build isolation hides the env's torch from the build, so it fails
    # before compiling. Declared on the Library record.
    if no_build_isolation_for:
        cmd += ["--no-build-isolation-package", no_build_isolation_for]
    cmd += ["--with-requirements", env_file, "--",
            "python", "-m", "kernel_bench", "run", *run_args]
    return cmd


def _cmd_run_all(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="kernel_bench run-all")
    ap.add_argument("--jobs", required=True, help="jobs.jsonl from `plan`")
    ap.add_argument("-o", "--output-dir", default="results",
                    help="write results/<lib>.csv here")
    ap.add_argument("--envs-dir", default="kernel_bench/envs")
    ap.add_argument("--box", default=None, metavar="PROFILE",
                    help="box profile path/name. Defaults to $KERNEL_BENCH_BOX, "
                         "then envs/boxes/<hostname>.toml when present")
    ap.add_argument("--python", default=None, help="pin the uv python (e.g. 3.11)")
    ap.add_argument("--env", action="append", default=[], metavar="LIB=PYTHON",
                    help="interpreter for a source-build env, e.g. "
                         "fa3=/opt/venvs/fa3/bin/python (repeatable). Also read "
                         "from KERNEL_BENCH_ENV_<LIB>.")
    ap.add_argument("--contributor", default="@kernel-bench")
    ap.add_argument("--no-cross-check", action="store_true")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="re-measure rows already in the corpus")
    ap.add_argument("--measured-in", action="append", default=[],
                    metavar="CSV",
                    help="results CSV whose rows count as measured, for every "
                         "env (repeatable) — e.g. the rows the database already "
                         "holds for this GPU, exported, so a box without "
                         "database access does not re-measure them")
    ap.add_argument("--constraints", default=None, metavar="FILE",
                    help="pip constraints applied to every uv-resolved env, "
                         "INCLUDING the ones the box profile overrides in "
                         "[env_requirements] — use it to pin the torch build "
                         "this box can actually run (e.g. torch==2.11.*, a "
                         "cuXXX index). The env files ask only for torch>=2.5, "
                         "so without this uv takes the newest build, which "
                         "mismatches a pinned driver/CUDA stack silently. An "
                         "env that cannot satisfy what you pass fails to "
                         "resolve, which is the point: this is the flag for "
                         "when a driver is misbehaving, and a pin it quietly "
                         "ignored would be worse than no flag. The PROFILE's "
                         "own `constraints` is a default instead, and an "
                         "override replaces it.")
    ap.add_argument("--budget", default="0", metavar="DURATION",
                    help="wall-clock budget per provider env, e.g. 2h (0 disables)")
    ap.add_argument("--stall-timeout", default="15m", metavar="DURATION",
                    help="terminate an env when its output CSV mtime does not "
                         "advance for this long (0 disables)")
    ap.add_argument("--max-job-failures", type=int, default=20,
                    help="maximum stalled/crashed jobs to quarantine per env")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-env commands without running them")
    args = ap.parse_args(argv)
    # A stop (the queue passing its SIGTERM on, an operator's kill) must stop
    # the env this run is supervising too: each runs in its own session, so
    # nothing reaches it unless the supervisor is unwound and stops it
    # (`supervisor.supervise_process`). Python's default for SIGTERM exits
    # without unwinding; this makes it an exit that does.
    import signal

    def _stop(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _stop)

    import os
    import pathlib
    import tempfile
    import time

    from .box import (BoxEnvs, BoxProfileError, apply_library_env,
                      apply_profile_env, find_box_profile, library_requirements,
                      prefix_command)
    from .plan import read_jobs_jsonl, write_jobs_jsonl
    from .providers import dependency_manifest
    from .supervisor import (NoProgress, parse_duration, read_disabled,
                             read_in_flight, read_progress, run_command)

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = pathlib.Path(args.output_dir)
    overrides = dict(kv.split("=", 1) for kv in args.env if "=" in kv)

    try:
        profile = find_box_profile(args.box, args.envs_dir)
    except BoxProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if profile:
        print(f"box: {profile.name} ({profile.path})")
    # One run_id for the whole batch: every env's rows carry it, so a
    # contaminated sweep is revoked by a single tombstone.
    run_id = _new_run_id()
    try:
        budget_s = parse_duration(args.budget)
        stall_timeout_s = parse_duration(args.stall_timeout)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    base_env = apply_profile_env(os.environ, profile)
    base_env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root)]
        + ([base_env["PYTHONPATH"]] if base_env.get("PYTHONPATH") else []))
    # These envs pull hundreds of MB of CUDA wheels; uv's 30s default aborts
    # the whole resolve on a slow link. An operator-set value still wins.
    base_env.setdefault("UV_HTTP_TIMEOUT", UV_HTTP_TIMEOUT_S)

    # The envs this run can start — the same object a plan judged its jobs
    # against, so the two cannot disagree about which envs this box has.
    box = BoxEnvs.of(profile, args.envs_dir, overrides=overrides,
                     asked_constraints=args.constraints, environ=base_env)
    python_version = args.python or (profile.python if profile else None)

    manifest = dependency_manifest()
    # The envs this plan needs: those owning a candidate of some job. A plan
    # records the premises it judged on (`judged_on`); where this run's
    # differ — another card, a source build given here with --env, an
    # operator's --constraints, a plan written before plans were judged —
    # every job, the plan's gaps included, is judged again against the envs
    # THIS run can start, and those are the jobs it runs. Never fewer envs
    # than the run needs.
    from .plan import gaps_path, split_by_candidates
    planned = read_jobs_jsonl(args.jobs)
    gaps_file = gaps_path(args.jobs)
    # The plan's gaps are part of the judgement too: an all-gap plan whose
    # premises changed has an empty plan file and must still be asked again.
    gap_jobs = read_jobs_jsonl(gaps_file) if os.path.exists(gaps_file) else []
    premises = box.premises(manifest)
    if any(job.get("judged_on") != premises for job in planned + gap_jobs):
        planned, _gaps = split_by_candidates(planned + gap_jobs, box)
        out_dir.mkdir(parents=True, exist_ok=True)
        args.jobs = str(out_dir / ".rejudged-jobs.jsonl")
        write_jobs_jsonl(planned, args.jobs)
        print(f"plan judged on other premises: re-judged against this run's "
              f"envs — {len(planned)} jobs to run, {len(_gaps)} with no "
              f"provider")
    needed = {lib for job in planned for lib in job["candidates"].values()}
    ran, failed, not_needed = [], [], []
    quarantined_by_lib = {}
    disabled_by_lib: dict = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for lib in sorted(manifest):
        info = manifest[lib]
        if lib not in needed:
            # Not started, not failed: nothing it owns serves this plan.
            not_needed.append(lib)
            print(f"[{lib}] skip: no provider for this plan's jobs")
            continue
        output_path = str(out_dir / f"{lib}.csv")
        # What THIS env runs in: `base_env` plus whatever the box states for
        # this library alone (`[library_env.<lib>]`) — the forward-compat
        # user-mode driver a CUDA-13 env needs and a CUDA-12 env must not see.
        lib_env = apply_library_env(base_env, profile, lib)
        py = box.python_for(lib) if info["source_build"] else None
        if info["source_build"] and not py:
            # Needed and not startable: its jobs were not attempted, so the
            # run is not done — unlike an env this plan does not need.
            hint = info["build_hint"] or "build it, then pass --env"
            failed.append(lib)
            print(f"[{lib}] SKIP (source-build) — NOT DONE: this plan needs it; {hint}; "
                  f"then: --env {lib}=/path/to/venv/bin/python "
                  f"(or KERNEL_BENCH_ENV_{lib.upper()}).")
            continue

        def _command(jobs_path, *, append_output=False, progress_path=None):
            run_args = ["--jobs", jobs_path, "-o", output_path,
                        "--library", lib,
                        "--contributor", args.contributor,
                        "--run-id", run_id]
            if profile:
                run_args.extend(["--box", profile.name])
            if not args.no_skip_existing:
                run_args.append("--skip-existing")
            if args.no_cross_check:
                run_args.append("--no-cross-check")
            if append_output:
                run_args.append("--append-output")
            # The envs measured before this one, in this output directory —
            # a row already there (an earlier pass over the same jobs) is not
            # measured again. Not what keeps a provider to one env: that is
            # `--library` above, since another env's torch is another
            # `framework_version` and so another identity to this check.
            for other in sorted(out_dir.glob("*.csv")):
                if str(other) != output_path:
                    run_args.extend(["--measured-in", str(other)])
            for given in args.measured_in:
                run_args.extend(["--measured-in", given])
            if progress_path:
                run_args.extend(["--progress-file", progress_path])
            if info["source_build"]:
                command = [py, "-m", "kernel_bench", "run", *run_args]
            else:
                env_file, env_constraints = library_requirements(
                    profile, lib, args.envs_dir,
                    asked_constraints=args.constraints)
                command = uv_run_cmd(
                    env_file, run_args, python_version=python_version,
                    constraints=env_constraints,
                    no_build_isolation_for=info.get("no_build_isolation"))
            return prefix_command(command, profile)

        if args.dry_run:
            cmd = _command(args.jobs)
            print(f"[{lib}] PYTHONPATH={repo_root} " + " ".join(cmd))
            continue
        print(f"\n=== {lib} ===")
        if not info["source_build"]:
            # Resolve the env BEFORE the watchdog runs: uv resolving and
            # downloading a provider env can take tens of minutes on a cold
            # cache, and the stall watchdog watches the output CSV — which
            # cannot exist until the env does. vllm's env was killed at
            # 907s mid-resolve exactly this way. `python -c pass` doubles
            # as the resolve; the supervised run then starts with a warm
            # cache and the stall timer measures the measurement.
            env_file, env_constraints = library_requirements(
                profile, lib, args.envs_dir,
                asked_constraints=args.constraints)
            resolve_cmd = prefix_command(
                ["uv", "run", "--no-project"]
                + (["--python", python_version] if python_version else [])
                + (["--with-requirements", env_constraints]
                   if env_constraints else [])
                + (["--no-build-isolation-package",
                    info["no_build_isolation"]]
                   if info.get("no_build_isolation") else [])
                + ["--with-requirements", env_file,
                   "--", "python", "-c", "pass"],
                profile)
            print(f"[{lib}] resolving env …", flush=True)
            # Through the same launcher as the measured run, with both limits
            # set to zero. "The resolve is not on the stall timer" is the
            # whole point of doing it here, and it is said by passing 0 —
            # not by reaching past the supervisor to `subprocess.run`. A
            # second way to start a child process is a second thing to
            # configure, a second thing to log, and a second thing every
            # test has to know to intercept; this one was none of those, so
            # `run-all` on a box where the resolve fails could not be
            # exercised at all.
            r = run_command(
                resolve_cmd, output_path="", env=lib_env,
                cwd=str(repo_root), budget_s=0, stall_timeout_s=0)
            if r.returncode != 0:
                failed.append(lib)
                print(f"[{lib}] env failed to resolve", file=sys.stderr)
                continue
        current_jobs = args.jobs
        append_output = False
        quarantined = []
        env_started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f"hbench-{lib}-") as temp_dir:
            attempt = 0
            while True:
                progress_path = str(pathlib.Path(temp_dir) / "progress.json")
                pathlib.Path(progress_path).unlink(missing_ok=True)
                cmd = _command(
                    current_jobs, append_output=append_output,
                    progress_path=progress_path)
                remaining_budget = budget_s
                if budget_s > 0:
                    remaining_budget = max(
                        0.0, budget_s - (time.monotonic() - env_started))
                    if remaining_budget == 0:
                        rc = 124
                        print(f"[{lib}] STOPPED: wall-clock budget exhausted",
                              file=sys.stderr)
                        break
                try:
                    supervised = run_command(
                        cmd, output_path=output_path, env=lib_env,
                        cwd=str(repo_root), budget_s=remaining_budget,
                        stall_timeout_s=stall_timeout_s)
                except FileNotFoundError:
                    print(f"[{lib}] interpreter not found: {cmd[0]!r} "
                          "(install uv, or fix --env path).", file=sys.stderr)
                    rc = 127
                    break
                rc = supervised.returncode
                # What this attempt's canaries disabled. Every attempt asks
                # again (a restart re-runs the canary), so merged, not kept.
                disabled_by_lib.setdefault(lib, {}).update(
                    read_disabled(progress_path))
                if supervised.stopped_by:
                    print(f"[{lib}] STOPPED after {supervised.elapsed_s:.1f}s: "
                          f"{supervised.stopped_by}", file=sys.stderr)
                if rc == EXIT_NO_VALID_ROWS:
                    print(f"[{lib}] ran to the end; every row it measured "
                          "failed validation (see the drops above)")
                    rc = 0
                if rc == EXIT_PATHS_DISABLED:
                    # Reached the end. The paths it disabled are read off
                    # the sidecar above, the one place that also holds
                    # them when the run died after disabling.
                    rc = 0
                if rc == 0:
                    break
                if (supervised.stopped_by
                        and supervised.stopped_by.startswith("wall-clock")):
                    break
                if len(quarantined) >= args.max_job_failures:
                    print(f"[{lib}] quarantine limit reached "
                          f"({args.max_job_failures})", file=sys.stderr)
                    break
                try:
                    failed_index, failed_job = read_progress(progress_path)
                    remaining_jobs = read_jobs_jsonl(current_jobs)
                except NoProgress:
                    # Not an isolation failure: there was no job to isolate.
                    # The child exited before starting one, so whatever it
                    # printed above is the cause — the sidecar is innocent.
                    print(f"[{lib}] exited {rc} before its first job; "
                          f"nothing to quarantine (the cause is above, not "
                          f"the sidecar)", file=sys.stderr)
                    break
                except (ValueError, OSError) as exc:
                    print(f"[{lib}] cannot isolate failed job: {exc}",
                          file=sys.stderr)
                    break
                if failed_index >= len(remaining_jobs):
                    print(f"[{lib}] progress index {failed_index} is outside "
                          f"{len(remaining_jobs)} jobs", file=sys.stderr)
                    break
                quarantined.append(failed_job)
                # The implementation in flight gets the row: `timeout` when
                # the watchdog killed it for making no progress, `crash` when
                # the process died by itself (including a CUDA context it
                # lost, `driver.CudaContextLost`).
                recorded = _record_in_flight(
                    output_path, read_in_flight(progress_path),
                    outcome="timeout" if supervised.stopped_by else "crash",
                    detail=(supervised.stopped_by
                            or f"process exited {rc} while this ran"))
                for line in recorded.notes:
                    print(f"[{lib}] {line}", file=sys.stderr)
                # Every in-flight implementation now has its row, so the job
                # itself goes back in: its other implementations still get
                # measured, and the recorded one is skipped by its row. A job
                # whose death named no implementation (its shared stage), or
                # whose row could not be written, is dropped as before.
                # Resume where the sidecar says (`index`); the job it names
                # as on the GPU is the one dropped when no row could take
                # the blame — the same job, except for a launch path's
                # canary, which runs ahead of its own job.
                rest = remaining_jobs[failed_index:]
                if not recorded.all:
                    if failed_job not in rest:
                        print(f"[{lib}] the job in flight is not among the "
                              f"{len(rest)} still to run", file=sys.stderr)
                        break
                    rest.remove(failed_job)
                print(f"[{lib}] QUARANTINE job {failed_job.get('op')} "
                      f"{failed_job.get('dims')} after process exit {rc}; "
                      f"continuing with {len(rest)} jobs", file=sys.stderr)
                if not rest:
                    break
                attempt += 1
                current_jobs = str(
                    pathlib.Path(temp_dir) / f"remaining-{attempt}.jsonl")
                write_jobs_jsonl(rest, current_jobs)
                append_output = True
        if quarantined:
            quarantined_by_lib[lib] = quarantined
        (ran if rc == 0 and not quarantined else failed).append(lib)
    disabled_by_lib = {lib: paths for lib, paths in disabled_by_lib.items()
                       if paths}

    if args.dry_run:
        return 0
    print(f"\nran {len(ran)} ok: {ran} | failed {len(failed)}: {failed} | "
          f"not needed by this plan {len(not_needed)}: {not_needed}")
    for lib, jobs in quarantined_by_lib.items():
        print(f"  {lib}: quarantined {len(jobs)} job(s)")
    for lib, paths in disabled_by_lib.items():
        for path, why in paths.items():
            print(f"  {lib}: disabled {path} — {why}")
    if failed:
        return 1
    if disabled_by_lib:
        # Every env ran to its end, but cells of the disabled paths were not
        # attempted: not done (`EXIT_PATHS_DISABLED`).
        print(f"not done: {sum(len(p) for p in disabled_by_lib.values())} "
              f"launch path(s) disabled — their cells stay pending for the "
              f"next pass")
        return EXIT_PATHS_DISABLED
    return 0


class _InFlightRecord:
    """What `_record_in_flight` did: whether every in-flight implementation
    now has its row (`all`; False when there was none to record), and the
    lines to report."""

    def __init__(self) -> None:
        self.all = False
        self.notes: List[str] = []


def _record_in_flight(output_path: str, in_flight: List[dict], *,
                      outcome: str, detail: str) -> _InFlightRecord:
    """Append the rows naming the implementations a dead or stalled `run`
    had in flight (`supervisor.read_in_flight`) to its output, as `outcome`
    with `detail` — through the same gate `run` applies to its own rows
    (schema, then the file's one-row-per-identity rule)."""
    from backend.inference import benchmarks as _benchmarks

    from .emit import IncrementalCsvWriter, one_line

    done = _InFlightRecord()
    if not in_flight:
        return done
    done.all = True
    existing = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    with IncrementalCsvWriter(output_path, append=existing) as writer:
        for template in in_flight:
            row = {c: str(template.get(c) or "") for c in _benchmarks.ALL_COLUMNS}
            row["outcome"] = outcome
            row["outcome_detail"] = one_line(detail)
            name = f"{row['framework']}:{row['kernel_impl']} {row['op']} {row['shape_json']}"
            try:
                _benchmarks.validate_row(row)
            except _benchmarks.BenchmarkValidationError as e:
                done.all = False
                done.notes.append(f"{outcome} of {name} NOT recorded: {e}")
                continue
            why = writer.write(row)
            if why is not None:
                done.all = False
                done.notes.append(f"{outcome} of {name} NOT recorded: {why}")
                continue
            done.notes.append(f"recorded {outcome}: {name}")
    return done


def _cmd_envs(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="kernel_bench envs")
    ap.add_argument("-o", "--output-dir", default="kernel_bench/envs",
                    help="write the generated envs/*.txt here")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if on-disk files differ from generated (CI)")
    args = ap.parse_args(argv)

    import pathlib

    from .providers import render_env_files

    files = render_env_files()
    out_dir = pathlib.Path(args.output_dir)
    if args.check:
        stale = [name for name, content in files.items()
                 if not (out_dir / name).is_file()
                 or (out_dir / name).read_text() != content]
        if stale:
            print(f"stale env files (run `kernel_bench envs`): {stale}", file=sys.stderr)
            return 1
        print("env files up to date")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out_dir / name).write_text(content)
    print(f"wrote {len(files)} env files → {out_dir}: {sorted(files)}")
    return 0


def _cmd_upload(argv: List[str]) -> int:
    """Upload measured CSV(s) to a HuggingArch deployment as private benchmark
    sets. Auth via a personal access token (web account page → Access tokens).
    Each CSV file becomes one named upload set.

    `check` runs first (MFU / MBU / recorded allclose). A file that claims
    work the card cannot do does not leave the box.
    """
    import glob
    import json as _json
    import os
    import urllib.error
    import urllib.request

    from .check import check_paths, format_report

    ap = argparse.ArgumentParser(prog="hbench upload")
    ap.add_argument("path", help="a results CSV, or a dir of *.csv")
    ap.add_argument("--host", default=os.environ.get("HUGGINGARCH_HOST", "https://huggingarch.com"),
                    help="deployment base URL (or $HUGGINGARCH_HOST)")
    ap.add_argument("--token", default=os.environ.get("HUGGINGARCH_TOKEN"),
                    help="personal access token (or $HUGGINGARCH_TOKEN)")
    ap.add_argument("--label", default=None, help="label for the set (default: filename)")
    ap.add_argument("--public", action="store_true",
                    help="publish straight to the SHARED corpus (admins only; "
                         "ignored server-side for non-admins)")
    args = ap.parse_args(argv)

    args.token = (args.token or "").strip()
    if not args.token:
        print("error: no token — pass --token or set $HUGGINGARCH_TOKEN "
              "(mint one in the web account page → Access tokens)", file=sys.stderr)
        return 2
    files = ([args.path] if args.path.endswith(".csv")
             else sorted(glob.glob(os.path.join(args.path, "*.csv"))))
    if not files:
        print(f"no CSV files at {args.path}", file=sys.stderr)
        return 1

    report = check_paths(args.path)
    print(format_report(report))
    if not report.ok():
        print("upload refused — check failed", file=sys.stderr)
        return 1

    url = args.host.rstrip("/") + "/api/bench/upload"
    rc = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            csv_text = fh.read()
        if not csv_text.strip():
            print(f"skip (empty): {f}")
            continue
        label = args.label or os.path.splitext(os.path.basename(f))[0]
        body = _json.dumps({"csv_text": csv_text, "label": label,
                            "make_public": bool(args.public)}).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.token}",
            # urllib's default UA is refused at a Cloudflare edge (403, 1010).
            "User-Agent": "huggingarch-kernel-bench/1"})
        tag = os.path.basename(f)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = _json.loads(resp.read().decode())
            anom = out.get("anomalous_count", 0)
            tier = "SHARED corpus" if out.get("shared") else "private"
            if args.public and not out.get("shared"):
                tier = "private (--public ignored: not an admin)"
            print(f"✓ {tag}: {out.get('row_count', '?')} rows → {tier} "
                  f"(id={out.get('upload_id', '?')[:8]}"
                  + (f", {anom} anomalous" if anom else "")
                  + f") label={out.get('label', label)}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            print(f"✗ {tag}: {e.code} {detail}", file=sys.stderr)
            rc = 1
        except (urllib.error.URLError, OSError) as e:
            print(f"✗ {tag}: {e}", file=sys.stderr)
            rc = 1
    return rc


def _cmd_up(argv: List[str]) -> int:
    """Clone → a working bench in one command: resolve the box profile,
    materialize every pip env (uv builds and caches it), then self-check
    with smoke. Anything this command cannot set up is said out loud —
    "you had to know" is exactly what it exists to end."""
    ap = argparse.ArgumentParser(prog="hbench up")
    ap.add_argument("--envs-dir", default="kernel_bench/envs")
    ap.add_argument("--box", default=None, metavar="PROFILE",
                    help="box profile path/name. Defaults to $KERNEL_BENCH_BOX, "
                         "then envs/boxes/<hostname>.toml when present")
    ap.add_argument("--python", default=None, help="pin the uv python (e.g. 3.11)")
    ap.add_argument("--constraints", default=None, metavar="FILE",
                    help="pip constraints applied to every env, including the "
                         "profile's [env_requirements] overrides (the torch "
                         "build this box runs — same meaning as in run-all)")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-env commands without building them")
    args = ap.parse_args(argv)

    import os
    import pathlib
    import subprocess

    from .box import (BoxProfileError, apply_library_env, apply_profile_env,
                      find_box_profile, library_requirements, prefix_command)
    from .providers import dependency_manifest

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    try:
        profile = find_box_profile(args.box, args.envs_dir)
    except BoxProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if profile:
        print(f"box: {profile.name} ({profile.path})")
    else:
        print("box: no profile — defaults (pass --box, set $KERNEL_BENCH_BOX, "
              "or drop envs/boxes/<hostname>.toml in place)")

    base_env = apply_profile_env(os.environ, profile)
    base_env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root)]
        + ([base_env["PYTHONPATH"]] if base_env.get("PYTHONPATH") else []))
    base_env.setdefault("UV_HTTP_TIMEOUT", UV_HTTP_TIMEOUT_S)
    python_version = args.python or (profile.python if profile else None)

    failed = []
    for lib, info in sorted(dependency_manifest().items()):
        if info["source_build"]:
            hint = info["build_hint"] or "build it, then point run-all at it"
            print(f"[{lib}] SKIP (source-build) — {hint}")
            continue
        imports = ",".join(sorted(info["import_names"]))
        env_file, env_constraints = library_requirements(
            profile, lib, args.envs_dir, asked_constraints=args.constraints)
        cmd = ["uv", "run", "--no-project"]
        if python_version:
            cmd += ["--python", python_version]
        if env_constraints:
            cmd += ["--with-requirements", env_constraints]
        cmd += ["--with-requirements", env_file, "--",
                "python", "-c", f"import {imports}"]
        cmd = prefix_command(cmd, profile)
        if args.dry_run:
            print(f"[{lib}] " + " ".join(cmd))
            continue
        print(f"[{lib}] building env ({len(info['providers'])} providers) …",
              flush=True)
        r = subprocess.run(
            cmd, env=apply_library_env(base_env, profile, lib), check=False)
        if r.returncode != 0:
            failed.append(lib)
            print(f"[{lib}] FAILED — its providers cannot measure on this box",
                  file=sys.stderr)
        else:
            print(f"[{lib}] ok")
    if args.dry_run:
        return 0
    if not args.skip_smoke:
        # The self-check: meet the base (torch) providers' declarations in
        # this env — harness machinery, canonical inputs, cross-check, the
        # whole path, before any GPU hour is spent.
        print("\nsmoke (torch providers, this env):")
        if _cmd_smoke(["--library", "torch"]) != 0:
            return 1
    if failed:
        print(f"\n{len(failed)} env(s) failed to materialize: {failed}")
        return 1
    print("\nup complete — next: hbench sweep MODEL -o jobs.jsonl && "
          "hbench run-all --jobs jobs.jsonl")
    return 0


def report_statuses() -> tuple:
    """The classes `hbench report` counts each cell × implementation in:
    what a row can say (`benchmarks.ROW_STATUSES`), then the two read off
    the declarations and the absence of rows."""
    from backend.inference.benchmarks import ROW_STATUSES
    return (*ROW_STATUSES, "not_supported", "pending")


def _slot_status(rows) -> str:
    """One cell × implementation's status from its rows'
    `(is_measurement, date, status)`: a measurement wins (vetted over not,
    `row_status`), otherwise the latest row says what happened last; no row
    is `pending`. The rows are whichever results the report was given."""
    if not rows:
        return "pending"
    measured = {status for is_measured, _date, status in rows if is_measured}
    if measured:
        return "passed" if "passed" in measured else "unverified"
    return max(rows)[2]


def _cmd_report(argv: List[str]) -> int:
    """Coverage and envelope against a jobs file: what is measured, who wins
    each measured cell, and the gap list (no provider / declined / pending).
    Reads the results CSVs you point it at (`--results`)."""
    ap = argparse.ArgumentParser(prog="hbench report")
    ap.add_argument("--jobs", default="jobs.jsonl")
    ap.add_argument("--results", default=None,
                    help="measured CSV or dir of CSVs (the runs to report on)")
    ap.add_argument("--gpu", default="RTX-4090",
                    help="GPU SKU the jobs target (the envelope is per-SKU)")
    args = ap.parse_args(argv)

    import glob
    import os
    from collections import Counter, defaultdict

    from backend.inference import benchmarks as _benchmarks
    from backend.inference.op_calibration import BenchmarkKey

    from .driver import physical_key_for
    from .env import Env
    from .invocation import Invocation
    from .plan import (gaps_by_signature, gaps_path, read_jobs_jsonl,
                       unplaced_by_reason, unplaced_path)
    from .providers import all_providers

    env = Env(None, False, None, args.gpu, None)
    jobs = read_jobs_jsonl(args.jobs)
    # The plan's cells no env of its box would measure (`plan.write_plan`):
    # still cells, counted below as `no provider` with the reason they gave.
    gaps = (read_jobs_jsonl(gaps_path(args.jobs))
            if os.path.exists(gaps_path(args.jobs)) else [])
    unplaced = (read_jobs_jsonl(unplaced_path(args.jobs))
                if os.path.exists(unplaced_path(args.jobs)) else [])

    # measured[(physical_key, impl_id)] = (latency_us, library) — the
    # results, latest source winning the slot.
    measured = {}
    # recorded[(physical_key, impl_id)] = the rows' statuses there
    # (`benchmarks.row_status`), whatever they are.
    recorded: defaultdict = defaultdict(list)
    if args.results:
        paths = ([args.results] if args.results.endswith(".csv")
                 else sorted(glob.glob(os.path.join(args.results, "*.csv"))))
        for path in paths:
            with open(path, encoding="utf-8") as f:
                _h, rows = _benchmarks.validate_text(f.read(), source=path)
            for r in rows:
                key = BenchmarkKey.from_csv_row(r)
                slot = (key.to_physical(), key.kernel_impl)
                recorded[slot].append((_benchmarks.is_measurement(r),
                                       r.get("date") or "", _benchmarks.row_status(r)))
                # A recorded outcome has no latency; the envelope is about
                # what the corpus MEASURES, and reading one here crashed on
                # `float("")`.
                if _benchmarks.is_measurement(r):
                    measured[slot] = (float(r["latency_us"]), key.framework)

    # One status per cell × implementation (`_slot_status`): the rows' own,
    # plus the two no row carries — `not_supported` (declined by the
    # provider's declaration, or no provider at all) and `pending`.
    statuses: dict = {}

    covered = partial = pending = 0
    no_provider: Counter = Counter()
    declined: Counter = Counter()
    pending_examples: defaultdict = defaultdict(list)
    envelope: defaultdict = defaultdict(Counter)  # signature -> impl -> cells
    for job in jobs:
        try:
            inv = Invocation.from_job(job)
        except ValueError as e:
            declined[f"invalid job: {e}"] += 1
            continue
        pk = physical_key_for(inv, env)
        # `Provider.accepts` is the one gate the run takes: it also refuses a
        # disabled kernel and a scale layout this one does not read. Asking
        # `serves` directly reported those cells as provider-backed and
        # merely pending, which reads as coverage — the opposite of what a
        # gap list is for.
        targeting = [p for p in all_providers() if p.signature == inv.signature]
        provs = [p for p in targeting if p.accepts(inv, env)]
        for p in targeting:
            if p not in provs:
                statuses[(pk, p.kernel_impl)] = "not_supported"
        if not provs:
            if not targeting:
                statuses[(pk, "")] = "not_supported"
            no_provider[inv.signature] += 1
            continue
        got, missing = [], []
        for p in provs:
            try:
                impl_id = p.impl_id(inv)
            except NotImplementedError as e:
                declined[f"{p.library}:{p.kernel_impl}: {e}"] += 1
                statuses[(pk, p.kernel_impl)] = "not_supported"
                continue
            statuses[(pk, impl_id)] = _slot_status(recorded.get((pk, impl_id), ()))
            hit = measured.get((pk, impl_id))
            (got.append((impl_id, hit[0], hit[1])) if hit
             else missing.append(impl_id))
        if got and not missing:
            covered += 1
        elif got:
            partial += 1
        else:
            pending += 1
        if got:
            impl_id, _lat, lib = min(got, key=lambda g: g[1])
            envelope[inv.signature][f"{lib}:{impl_id}"] += 1
        if missing:
            pending_examples[inv.signature].append((inv.dims, sorted(missing)))

    for job in gaps:
        no_provider[job.get("signature") or job.get("op", "?")] += 1
    total = (covered + partial + pending + sum(no_provider.values())
             + len(unplaced))
    print(f"report: {total} cells on {args.gpu} "
          f"({args.jobs} vs corpus{f' + {args.results}' if args.results else ''})")
    print(f"  covered {covered} | partial {partial} | pending {pending} | "
          f"no provider {sum(no_provider.values())} | structural {len(unplaced)}")
    by_status = Counter(statuses.values())
    print(f"  {len(statuses)} cell × implementation: " + " | ".join(
        f"{name} {by_status.get(name, 0)}" for name in report_statuses()))
    if envelope:
        print("\nenvelope (winning impl per measured cell):")
        for sig in sorted(envelope):
            wins = ", ".join(f"{impl}×{n}"
                             for impl, n in envelope[sig].most_common())
            print(f"  {sig}: {wins}")
    if no_provider:
        print("\nno serving provider:")
        for sig, n in no_provider.most_common():
            print(f"  {sig}: {n} cell(s)")
    if gaps:
        print(f"\nno provider on the plan's box ({gaps_path(args.jobs)}):")
        for line in gaps_by_signature(gaps):
            print(f"  {line}")
    if unplaced:
        print(f"\nstructural rows ({unplaced_path(args.jobs)}):")
        for line in unplaced_by_reason(unplaced):
            print(f"  {line}")
    if declined:
        print("\ndeclined by the provider's own declaration:")
        for why, n in declined.most_common(10):
            print(f"  {why} ×{n}")
    if pending_examples:
        print("\npending (up to 10 per signature):")
        for sig in sorted(pending_examples):
            for dims, impls in pending_examples[sig][:10]:
                print(f"  {sig} {dims} → {', '.join(impls)}")
    return 0


def _cmd_check(argv: List[str]) -> int:
    """MFU, MBU, and recorded allclose on measured CSVs.

    Does not touch the GPU. Disagreements with the reference were never
    written; this counts which rows were actually compared, and refuses a
    file that still claims work above the card's peak.
    """
    ap = argparse.ArgumentParser(prog="hbench check")
    ap.add_argument("results", help="measured CSV or directory of CSVs")
    args = ap.parse_args(argv)

    from .check import check_paths, format_report

    report = check_paths(args.results)
    print(format_report(report))
    if report.errors and report.n_rows == 0:
        return 2
    return 0 if report.ok() else 1


def _cmd_restamp(argv: List[str]) -> int:
    """Rewrite achieved_tflops / achieved_bw from a new plan and the row's p50.

    Latency stays. Numerators come from jobs.jsonl (engine export after
    count_per_step). A row with no matching job is left alone and counted.
    """
    import glob
    import os

    from backend.inference import benchmarks as _benchmarks
    from .emit import restamp_rows
    from .plan import read_jobs_jsonl

    ap = argparse.ArgumentParser(prog="hbench restamp")
    ap.add_argument("--jobs", required=True, help="jobs.jsonl from the new plan")
    ap.add_argument("results", help="measured CSV or directory of CSVs")
    args = ap.parse_args(argv)

    jobs = read_jobs_jsonl(args.jobs)
    paths = ([args.results] if args.results.endswith(".csv")
             else sorted(glob.glob(os.path.join(args.results, "*.csv"))))
    if not paths:
        print(f"no CSV files at {args.results}", file=sys.stderr)
        return 2
    rc = 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        header, rows = _benchmarks.validate_text(text, source=path)
        result = restamp_rows(rows, jobs)
        if result.ambiguous:
            rc = 1
            print(f"{os.path.basename(path)}: REFUSED "
                  f"{len(result.ambiguous)} ambiguous plan cell(s) "
                  "({len(sum((a.row_indexes for a in result.ambiguous), []))} "
                  "rows); file unchanged")
            for item in result.ambiguous:
                print(f"  {item.cell[0]} {item.cell[3]} "
                      f"{item.cell[2]}: {len(item.pairs)} numerator pairs")
            continue
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_benchmarks.emit_csv(header, result.rows))
        print(f"{os.path.basename(path)}: {result.changed} restamped, "
              f"{result.missed} unmatched, {len(rows)} rows")
        if result.unmatched:
            rc = 1
            reasons: Dict[str, int] = {}
            for item in result.unmatched:
                reasons[item.reason] = reasons.get(item.reason, 0) + 1
            print("  unmatched by reason: " +
                  ", ".join(f"{reason}={count}"
                            for reason, count in sorted(reasons.items())))
    return rc



def _workspace(box_arg, envs_dir, work_dir_arg):
    """The box's workspace, from the profile unless one is named."""
    import pathlib

    from .box import find_box_profile, require_work_dir
    from .queue import Workspace

    if work_dir_arg:
        return Workspace(pathlib.Path(work_dir_arg)), None
    profile = find_box_profile(box_arg, envs_dir)
    return Workspace(require_work_dir(profile)), profile


def _cmd_queue(argv: List[str]) -> int:
    """Serve this box's measuring queue: the oldest plan with no `.done`,
    measured with every other model's results as `--measured-in`, then the
    next one. Stop it with SIGTERM — the run stops with it."""
    ap = argparse.ArgumentParser(prog="hbench queue")
    ap.add_argument("--box", default=None, metavar="PROFILE",
                    help="box profile path/name (default: $KERNEL_BENCH_BOX, "
                         "then envs/boxes/<hostname>.toml)")
    ap.add_argument("--envs-dir", default="kernel_bench/envs")
    ap.add_argument("--work-dir", default=None,
                    help="override the profile's `work_dir`")
    ap.add_argument("--measured-in", action="append", default=[], metavar="CSV",
                    help="results that already exist elsewhere (a corpus "
                         "export); repeatable, merged with this box's own")
    ap.add_argument("--budget", default="6h", help="wall-clock budget per provider env")
    ap.add_argument("--stall-timeout", default="15m")
    ap.add_argument("--contributor", default=None)
    ap.add_argument("--python", default=None, help="pin the uv python")
    ap.add_argument("--poll", type=float, default=120.0,
                    help="seconds to wait when the queue is empty")
    ap.add_argument("--once", action="store_true",
                    help="measure one model and return")
    ap.add_argument("--list", action="store_true", help="print the queue and return")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the run-all command for the next model")
    args = ap.parse_args(argv)

    from .box import BoxProfileError
    from .queue import pending, serve

    try:
        space, profile = _workspace(args.box, args.envs_dir, args.work_dir)
    except BoxProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    space.prepare()
    if args.list:
        todo = pending(space)
        print(f"{space.root}: {len(todo)} to measure")
        for model in todo:
            jobs = sum(1 for _ in space.plan_of(model).open(encoding="utf-8"))
            print(f"  {model} ({jobs} jobs)")
        return 0
    return serve(space, poll=args.poll, once=args.once or args.dry_run,
                 dry_run=args.dry_run,
                 # The profile's PATH, not its name: a box's name is its
                 # identity in the corpus (`h100-sxm-cu128`) and its file is
                 # whatever it is called (`h100-sxm-cu128.toml`). Handing the name
                 # to `run-all` made it look for a profile that is not there
                 # and refuse every model in the queue.
                 box=args.box or (str(profile.path) if profile else None),
                 budget=args.budget, stall_timeout=args.stall_timeout,
                 contributor=args.contributor, python=args.python,
                 measured_in=args.measured_in, envs_dir=args.envs_dir)


def _cmd_enqueue(argv: List[str]) -> int:
    """Plan models and hand each plan to this box's queue as it lands, so
    the card measures the first while the rest are still being planned."""
    ap = argparse.ArgumentParser(prog="hbench enqueue")
    ap.add_argument("models", nargs="*", help="model ids; omit with --all/--missing")
    ap.add_argument("--all", action="store_true",
                    help="every model the spec corpus has a snapshot for")
    ap.add_argument("--missing", action="store_true",
                    help="of those, the ones this box has not measured")
    ap.add_argument("--gpu", default=None,
                    help="GPU SKU to plan for (default: the box profile's name)")
    ap.add_argument("--framework", default="vllm")
    ap.add_argument("--box", default=None, metavar="PROFILE")
    ap.add_argument("--envs-dir", default="kernel_bench/envs")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be planned and where it would land")
    _add_workers(ap)
    args = ap.parse_args(argv)

    import pathlib
    import tempfile

    from .box import BoxEnvs, BoxProfileError, find_box_profile
    from .plan import gaps_by_signature, write_plan
    from .queue import corpus_models, hand_over, unmeasured
    from .sweep import (build_sweep_jobs, resolve_sweep_model,
                        routine_parallel_configs)
    from backend.arch.eval.precision import AS_SHIPPED
    from backend.inference.sweep_grid import ROUTINE_OUTPUT_TOKENS, ROUTINE_TRACKS

    try:
        space, profile = _workspace(args.box, args.envs_dir, args.work_dir)
        profile = profile or find_box_profile(args.box, args.envs_dir)
    except BoxProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    models = list(args.models)
    if args.all or args.missing:
        models += corpus_models(repo_root)
    if args.missing:
        models = unmeasured(space, models)
    if not models:
        print("nothing to enqueue (name models, or --all / --missing)",
              file=sys.stderr)
        return 2
    try:
        # The envs this box runs: a job none of them would measure is a gap
        # beside the plan, not a pass of the GPU.
        box = BoxEnvs.of(profile, args.envs_dir, gpu=args.gpu)
    except BoxProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    gpu = box.gpu
    if not gpu:
        print("error: no --gpu and the box profile names no SKU", file=sys.stderr)
        return 2

    space.prepare()
    failed = 0
    for model_id in models:
        if args.dry_run:
            print(f"plan {model_id} (gpu={gpu}) → {space.plan_of(model_id)}")
            continue
        try:
            config, resolved = resolve_sweep_model(model_id)
            jobs, stats = build_sweep_jobs(
                model_id=model_id, gpu=gpu, seq_lens=None,
                tracks=ROUTINE_TRACKS,
                parallel_configs=routine_parallel_configs(
                    config, resolved=resolved),
                avg_output_tokens=ROUTINE_OUTPUT_TOKENS, resolved=resolved,
                framework=args.framework, precision=AS_SHIPPED,
                log=lambda m: None, workers=args.workers)
        except Exception as exc:                      # reported, not raised
            print(f"FAILED {model_id}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            failed += 1
            continue
        # Written beside the queue and moved in, so the loop never reads a
        # plan that is still being written.
        with tempfile.NamedTemporaryFile(
                "w", suffix=".jsonl", dir=str(space.root), delete=False) as tmp:
            staged = pathlib.Path(tmp.name)
        queued, gaps = write_plan(jobs, str(staged), box,
                                  unplaced=stats.get("unplaced") or [])
        hand_over(space, model_id, staged)
        print(f"queued {model_id}: {len(queued)} jobs "
              f"({stats['cells']} cells); {len(gaps)} with no provider on "
              f"this box", flush=True)
        for line in gaps_by_signature(gaps):
            print(f"  no provider — {line}", flush=True)
    return 1 if failed else 0


def _cmd_keepalive(argv: List[str]) -> int:
    """Hold the card while the queue has nothing to measure: the cluster
    reclaims a GPU idle for ten minutes, and a box between models (or
    one whose queue has died) would lose it."""
    ap = argparse.ArgumentParser(prog="hbench keepalive")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--idle-limit", type=float, default=None,
                    help="seconds of idleness before a burst (default 300)")
    ap.add_argument("--burst", type=float, default=None,
                    help="seconds per burst (default 90)")
    ap.add_argument("--duty", type=float, default=None,
                    help="fraction of the burst spent computing (default 0.6)")
    args = ap.parse_args(argv)

    from . import keepalive

    return keepalive.serve(
        poll=args.poll,
        idle_limit=args.idle_limit if args.idle_limit is not None else keepalive.IDLE_LIMIT,
        burst_seconds=args.burst if args.burst is not None else keepalive.BURST,
        duty=args.duty if args.duty is not None else keepalive.DUTY)


def _commands() -> Dict[str, Callable[[List[str]], int]]:
    """Every subcommand, read off this module's `_cmd_<name>` definitions:
    the function IS the command, and `run-all` is `_cmd_run_all`.

    One declaration rather than a name list beside a dispatch chain that
    repeated it — adding a subcommand meant editing three places, and a
    command missing from the list was simply unreachable.
    """
    return {name[len("_cmd_"):].replace("_", "-"): fn
            for name, fn in globals().items()
            if name.startswith("_cmd_") and callable(fn)}


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = _commands()
    if not argv or argv[0] not in commands:
        print(f"usage: hbench {{{','.join(commands)}}} ...", file=sys.stderr)
        return 2
    return commands[argv[0]](argv[1:])


def _cmd_smoke(argv: List[str]) -> int:
    """Meet every kernel's declaration once — layer 3 of the guarantee.

    Exits non-zero on a FALSE declaration (a kernel whose numbers disagree with
    the reference) or on one that could not be tried at all. An exception on
    this particular box is printed, not judged: it may equally mean the
    declaration is wrong or that the kernel wants a newer architecture.
    """
    import argparse
    from .env import detect_env
    from .smoke import format_report, smoke_check

    ap = argparse.ArgumentParser(prog="hbench smoke")
    ap.add_argument("--library", default="", help="only this library")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    env = detect_env()
    results = smoke_check(env, only_library=args.library, device=args.device)
    print(format_report(results))

    false_ = [r for r in results if r.false_declaration]
    untried = [r for r in results if r.untried]
    if false_:
        print(f"\n{len(false_)} FALSE declaration(s) — a kernel produced wrong "
              f"numbers for a combination it claims to serve")
    if untried:
        print(f"\n{len(untried)} declaration(s) could not be tried — a claim "
              f"nothing met is not a claim that held")
    return 1 if (false_ or untried) else 0


def _cmd_aggregate(argv: List[str]) -> int:
    import argparse
    import csv as _csv
    ap = argparse.ArgumentParser(prog="kernel_bench aggregate")
    ap.add_argument("csv", help="measured CSV from `run` / `run-all`")
    args = ap.parse_args(argv)
    from .aggregate import moe_block_mfu
    with open(args.csv, encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    blocks = moe_block_mfu(rows)
    if not blocks:
        print("no MoE-block rows found in", args.csv)
        return 0
    for b in blocks:
        mfu = "—" if b["block_mfu"] is None else f"{b['block_mfu'] * 100:.1f}%"
        print(f"{b['gpu']} {b['phase']:7s} MoE block: time={b['block_time_us']:.1f}us "
              f"useful={b['gemm_tflops_useful']:.2f}TF  block_MFU={mfu}  "
              f"({b['n_ops']} ops: {','.join(b['ops'])})")
    return 0
