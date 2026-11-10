"""Fill the gallery and Scaling-panel caches from a separate process.

Same stores the `/gallery/models`, `/gallery/frontier`, `/inference/estimate-sweep`,
and `/inference/estimate-parallel-sweep` endpoints read (`KeyedCache` in
`backend.measure.cache`, memory → disk → remote) — this is not a second cache. The server does not warm itself: a warm inside the
web process competes with user requests for the GIL and the sweep pool.
Run this next to the server instead (the local `ha` supervisor starts it
once the server reports healthy), or once per engine/spec at image build:

    python -m backend.warm_caches              # Gallery + Scaling, full public grids
    python -m backend.warm_caches --gallery    # every Gallery GPU only
    python -m backend.warm_caches --gpu H200   # facts + that Gallery GPU
    python -m backend.warm_caches --sweeps     # Scaling only (parallel + batch)
    python -m backend.warm_caches --landing    # homepage view only
    python -m backend.warm_caches --landing --sweeps

Process knobs (flag, else env, else default):

    --nice N      HA_WARM_NICE           10   lowest CPU priority it runs at
    --workers N   HA_WARM_WORKERS        min(32, cpu-2) its own sweep pool
    --pidfile P   —                      —    written at start, removed at exit

Cache locations are the server's: `HA_GALLERY_CACHE_DIR`,
`HA_SOL_POOL_DIR`, and `HA_SWEEP_CACHE_DIR`.
Running as `python -m backend.warm_caches` imports `backend.api`, which binds
the Supabase remote tier when `SUPABASE_*` is configured; remote reads and
writes follow the same `remote_ok` / `remote_if` gates as the endpoints, and
queued remote writes are drained before the process exits.

The gallery has no precision axis: every model is shown as shipped.
Its offline frontier warm hands one card class every missing model × layout
walk in one ranked queue (parameter count × layout GPUs, largest first); a
model's entry is assembled only after all of its layouts have landed.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import sys

from backend.fingerprint import engine_version
from backend.measure.cache import WRITE_BEHIND
from backend.api._models import (
    BatchPoint,
    InferenceEstimateSweepRequest,
    InferenceParallelSweepRequest,
)
from backend.api.gallery_overview import (
    _GALLERY_CACHE_DIR,
    build_model_facts,
    featured_examples,
)
from backend.api.gallery_frontier import (
    FrontierView, keepSharedWalk, warm_frontier,
)
from backend.api.sweep_runtime import (
    _SWEEP_CACHE_DIR,
    compute_estimate_sweep_cached,
    compute_parallel_sweep_cached,
    poolOnly,
    sharedSolPool,
)
from backend.api import offload
from backend.measure import pool
from backend.inference import gpu_specs
from backend.inference.parallel_sweep import (
    ALL_MODES,
    DEFAULT_SWEEP_GPUS,
    class_cards,
)
from backend.inference.sweep_grid import (
    DEFAULT_INFER_MODEL,
    GALLERY_GPUS,
    GALLERY_LANDING_GPU,
    ROUTINE_OUTPUT_TOKENS,
    ROUTINE_SEQ_LENS,
    SCALING_BATCH_GRID,
    SCALING_CONTEXTS,
    SCALING_DEGREES,
)

logger = logging.getLogger(__name__)




def _gallery_gpus() -> list[str]:
    """The gallery picker's GPUs (`GALLERY_GPUS` — not the sweep series,
    which also carries the Kernel Bench card), the landing GPU first so the
    homepage default view is the first one filled. Their three walk classes
    are three class walks per model (`parallel_sweep.class_cards`)."""
    return [GALLERY_LANDING_GPU,
            *(g for g in GALLERY_GPUS if g != GALLERY_LANDING_GPU)]


def _gpu_list(
    *, gpus: list[str] | None, all_gpus: bool, default: list[str] | None = None,
) -> list[str]:
    if all_gpus:
        # Module import (not `from ... import load_gpu_specs`) so test_warm's
        # `monkeypatch.setattr("backend.inference.gpu_specs.load_gpu_specs", ...)`
        # is seen: the name is resolved on the module at call time.
        return [s.sku for s in gpu_specs.load_gpu_specs()]
    if gpus:
        # Preserve order, drop dupes from repeated --gpu.
        out, seen = [], set()
        for sku in gpus:
            if sku not in seen:
                out.append(sku)
                seen.add(sku)
        return out
    if default:
        return list(default)
    return [GALLERY_LANDING_GPU]


# ---------------------------------------------------------------------------
# Scaling-panel sweeps (the durable-DB-tier target).
# ---------------------------------------------------------------------------


# The Scaling panel's POSTs, mirrored field-for-field so a warmed entry lands
# on the EXACT cache key production traffic reads. Sources:
#   web/src/pages/ScalingPanel.tsx  — fetch bodies (parallel / batch)
#   web/src/pages/Inference.tsx     — seeded state defaults (0.9 / "envelope")
# Precision is not mirrored: the panel's default is as shipped, and the
# sweep key holds a precision profile by its key (`sweep_cache_key`), so a
# request naming as shipped and one naming nothing share one entry.
_SCALING_FRAC = 0.9                  # Inference.tsx useState(0.9)
_SCALING_SELECTION_MODE = "envelope"  # Inference.tsx useState("envelope")
_SCALING_OUTPUT_TOKENS = ROUTINE_OUTPUT_TOKENS  # compare.ts SWEEP_OUTPUT_TOKENS
# The parallel POST sends Plan's output length, not the batch grid's: the
# request length `in + out` sizes the KV, so it is a keyed field there.
_PLAN_OUTPUT_TOKENS = ROUTINE_OUTPUT_TOKENS  # Inference.tsx useState(SWEEP_OUTPUT_TOKENS) (outTok)


def _scaling_models(models: list[str] | None) -> list[str]:
    if models:
        return list(models)
    # The default landing model (DeepSeek-V3) is the single most common
    # first hit; the featured set may not include it. Union it in, default
    # first, de-duped, so the default page load is always warmed.
    featured = featured_examples()
    return [DEFAULT_INFER_MODEL] + [m for m in featured if m != DEFAULT_INFER_MODEL]


def default_scaling_sweep_request(model_id: str, context_len: int):
    """The exact `InferenceParallelSweepRequest` a first visitor to the Scaling
    panel produces for (model, context) — so warm writes the key the endpoint
    reads. See the field-source comment above; `gpu_sku` is inherited-but-ignored
    in the parallel key, and `gpu_skus=None` matches the panel (which does not
    send it → the server's curated default series)."""
    return InferenceParallelSweepRequest(
        model_id=model_id,
        gpu_sku="",
        context_len=context_len,
        modes=list(ALL_MODES),
        gpu_memory_fraction=_SCALING_FRAC,
        avg_output_tokens=_PLAN_OUTPUT_TOKENS,
        selection_mode=_SCALING_SELECTION_MODE,
        gpu_skus=None,
        batch=None,  # each config's max
        sol_only=True,  # the panel's first POST; its measured one is live
    )


def default_scaling_batch_request(
    model_id: str, gpu: str, mode: str, degree: int,
):
    """Context × Batch POST: seq × batch_grid at one (GPU, parallel, degree)."""
    return InferenceEstimateSweepRequest(
        model_id=model_id,
        gpu_sku=gpu,
        seq_lens=list(ROUTINE_SEQ_LENS),
        batch_grid=[BatchPoint(**bp) for bp in SCALING_BATCH_GRID],
        avg_output_tokens=_SCALING_OUTPUT_TOKENS,
        parallel_preset={"mode": mode, "degree": degree},
        gpu_memory_fraction=_SCALING_FRAC,
        selection_mode=_SCALING_SELECTION_MODE,
        sol_only=True,  # the panel's first POST; its measured one is live
    )


def sweep_jobs(
    models: list[str] | None = None, contexts: list[int] | None = None
) -> list[tuple[str, int]]:
    """(model_id, context_len) pairs for the parallel-sweep half. Default: the
    featured model set × the panel's context grid (`SCALING_CONTEXTS`)."""
    ctxs = contexts or list(SCALING_CONTEXTS)
    return [(mid, ctx) for mid in _scaling_models(models) for ctx in ctxs]


def scaling_jobs(
    models: list[str] | None = None,
    contexts: list[int] | None = None,
    gpus: list[str] | None = None,
    modes: list[str] | None = None,
    degrees: list[int] | None = None,
) -> list[tuple[str, str, object]]:
    """Every Scaling-panel POST as shipped: parallel (model × ctx) plus
    Context × Batch (model × gpu × mode × degree). KV storage reads the batch
    payload, so it is not a separate job.

    `kind` is `parallel` / `batch`. GPU default is the parallel-chart series
    (`DEFAULT_SWEEP_GPUS`); pass `--all-gpus` / `--gpu` to expand.
    """
    mids = _scaling_models(models)
    ctxs = contexts or list(SCALING_CONTEXTS)
    sku = gpus or list(DEFAULT_SWEEP_GPUS)
    par = modes or list(ALL_MODES)
    deg = degrees or list(SCALING_DEGREES)
    jobs: list[tuple[str, str, object]] = []
    for mid in mids:
        jobs.extend(
            ("parallel", f"{mid} · ctx={ctx}",
             default_scaling_sweep_request(mid, ctx))
            for ctx in ctxs
        )
        jobs.extend(
            ("batch", f"{mid} · {gpu} · {mode}@{n}",
             default_scaling_batch_request(mid, gpu, mode, n))
            for gpu in sku for mode in par for n in deg
        )
    return jobs


async def ensureSharedWalks(models: list[str] | None,
                            gpus: list[str] | None) -> int:
    """Produce the canonical walk classes a Scaling warm will consume.

    The Gallery warm fills its public card classes. A Scaling-only card (the
    measured RTX-4090) still needs one producer-side bundle before its cells
    can be projections; this walks only classes whose complete record bundle
    is absent. The anchor spans every Scaling context, so this is once per
    model and card class, not once per request matrix cell.
    """
    series = list(gpus or DEFAULT_SWEEP_GPUS)
    representatives: dict[str, str] = {}
    for gpu in series:
        cards = class_cards(gpu, series)
        representatives[cards[0]] = cards[0]
    wanted = list(dict.fromkeys(_scaling_models(models)))
    ok = 0
    total = len(wanted) * len(representatives)
    done = 0
    for model in wanted:
        for walkCard, askedGpu in representatives.items():
            done += 1
            label = f"{model} · {walkCard}"
            print(f"[walk {done}/{total}] {label}", flush=True)
            try:
                ok += int(await keepSharedWalk(model, askedGpu))
            except Exception as e:
                print(f"  fail walk {label}: {e}", flush=True)
    print(f"canonical walks ensured {ok}/{total}", flush=True)
    return ok


async def warm_sweeps(
    models: list[str] | None = None,
    contexts: list[int] | None = None,
    gpus: list[str] | None = None,
    modes: list[str] | None = None,
    degrees: list[int] | None = None,
) -> int:
    """Fill the sweep cache (`_SWEEP`: memory→disk→Supabase) for every Scaling
    panel POST (parallel + Context × Batch + KV) through the SAME
    key/compute/store path the endpoints read: each sweep's SOL answer, plus
    the canonical layout laws that let every default cell project without a
    walk. A measured POST computes its breakdown and measures live, then makes
    the lookup and fold.
    Per-job failures are isolated;
    one bad model never aborts the run. Returns the count of jobs that stored
    successfully."""
    walks = await ensureSharedWalks(models, gpus)
    if poolOnly:
        print("SOL answer cache disabled: canonical walks kept only", flush=True)
        return int(walks > 0)
    jobs = scaling_jobs(models, contexts, gpus, modes, degrees)
    n_par = sum(1 for k, _, _ in jobs if k == "parallel")
    n_batch = sum(1 for k, _, _ in jobs if k == "batch")
    print(f"sweep engine version: {engine_version()}", flush=True)
    print(
        f"warming {len(jobs)} sweeps "
        f"({n_par} parallel + {n_batch} batch)",
        flush=True,
    )
    ok = 0
    for i, (kind, label, req) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {kind} {label}", flush=True)
        try:
            if kind == "parallel":
                await compute_parallel_sweep_cached(req)
            else:
                await compute_estimate_sweep_cached(req)
            ok += 1
        except Exception as e:  # isolate: one model's failure never aborts warm
            print(f"  fail {kind} {label}: {e}", flush=True)
    print(f"warmed {ok}/{len(jobs)} sweeps", flush=True)
    return ok


class WarmProcess:
    """The OS-level footing of a warm run: CPU priority, sweep-pool size,
    pidfile, and flushing queued remote writes before exit. The warm runs
    next to a live server and must yield to it."""

    _DRAIN_TIMEOUT_S = 120.0

    def __init__(self, *, nice: int, workers: int, pidfile: str | None):
        self.nice = nice
        self.workers = workers
        self.pidfile = pathlib.Path(pidfile) if pidfile else None

    @staticmethod
    def default_nice() -> int:
        return int(os.environ.get("HA_WARM_NICE", "10"))

    @staticmethod
    def default_workers() -> int:
        """`HA_WARM_WORKERS`, its own knob — not the API's pool size
        (`HA_SWEEP_POOL_WORKERS`): the warm runs beside the API's pool in one
        container, so it is budgeted apart (fly.toml) — else min(32, cpu-2):
        run offline, a cold warm is thousands of independent layout walks,
        and a 4-worker pool left a 112-core box idle for hours."""
        raw = os.environ.get("HA_WARM_WORKERS", "").strip()
        if raw:
            return int(raw)
        return max(0, min(32, (os.cpu_count() or 1) - 2))

    def __enter__(self) -> "WarmProcess":
        current = os.nice(0)
        if self.nice > current:
            os.nice(self.nice - current)  # pool workers inherit it
        # Read by `pool._max_workers` when the pool is first built,
        # which happens after this point (first sweep cell).
        os.environ["HA_SWEEP_POOL_WORKERS"] = str(self.workers)
        # Every heavy job — each gallery model, each sweep cell — runs in
        # the pool, as in the server. The pool is budgeted beside the API's
        # in one container (fly.toml).
        if self.workers >= 1:
            offload.serve_requests_off_process(deadline_s=None)
        if self.pidfile is not None:
            self.pidfile.parent.mkdir(parents=True, exist_ok=True)
            self.pidfile.write_text(f"{os.getpid()}\n")
        print(f"warm pid {os.getpid()} · nice {os.nice(0)} · "
              f"pool workers {self.workers}", flush=True)
        return self

    def __exit__(self, *exc) -> None:
        offload.stop_serving()
        if not WRITE_BEHIND.drain(self._DRAIN_TIMEOUT_S):
            print("remote cache writes still pending at exit; dropped",
                  file=sys.stderr, flush=True)
        pool.shutdown()
        if (self.pidfile is not None and self.pidfile.exists()
                and self.pidfile.read_text().strip() == str(os.getpid())):
            self.pidfile.unlink(missing_ok=True)


def _warm_frontier(gpu: str,
                   model_params: dict[str, int | None]) -> bool:
    """Every model's decode Pareto frontier on `gpu` (`/gallery/frontier`):
    one GPU view spans every layout the frontier prices, so it is one job
    per GPU. The first GPU of a walk class walks each model for every card
    of the class (`gallery_frontier._layout_law`); the class's other GPUs
    price that walk."""
    print(f"warming the decode frontier on {gpu}", flush=True)
    return _report_payload(asyncio.run(warm_frontier(FrontierView(gpu),
                                                     model_params)))


def _warm_model_facts() -> dict[str, int | None]:
    """Every model's own facts (`/gallery/models`) — the half of the
    gallery that shows before any GPU view has computed. Cheap; run before
    the frontier so a first visitor sees the page at once."""
    print(f"gallery cache dir: {_GALLERY_CACHE_DIR}", flush=True)
    payload = asyncio.run(build_model_facts())
    models = payload.get("models") or []
    print(f"model facts: {len(models) - payload.get('errors_count', 0)}"
          f"/{len(models)} models", flush=True)
    return {model.get("model_id", ""): model.get("total_params")
            for model in models}


def _report_payload(payload: dict) -> bool:
    models = payload.get("models") or []
    ok = sum(1 for m in models if not m.get("error"))
    print(f"  {ok}/{len(models)} models", flush=True)
    if payload.get("errors_count"):
        for m in models:
            if m.get("error"):
                print(f"  fail {m.get('model_id')}: {m['error']}", flush=True)
    return ok > 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-compute gallery / Scaling-panel results into the "
                    "server's caches, from a separate low-priority process.",
    )
    parser.add_argument(
        "--gpu", action="append", dest="gpus", metavar="SKU",
        help="SKU to warm (repeatable). Default: the gallery's GPUs, landing "
             "GPU first (the Scaling sweeps: the parallel-chart series).",
    )
    parser.add_argument(
        "--all-gpus", action="store_true",
        help="Warm every SKU in gpu_specs.csv. Only the Scaling sweeps take "
             "it: the gallery frontier serves the gallery's GPUs alone.",
    )
    parser.add_argument(
        "--landing", action="store_true",
        help="Warm only the homepage's default view (model facts plus the "
             "decode frontier on the landing GPU).",
    )
    parser.add_argument(
        "--gallery", action="store_true",
        help="Warm the Gallery without the Scaling sweeps. With no surface "
             "flag at all, both surfaces are warmed.",
    )
    parser.add_argument(
        "--sweeps", action="store_true",
        help="Warm every Scaling-panel POST (parallel ctx dropdown and "
             "Context × Batch GPU/parallel/degree; KV storage shares the batch "
             "payload) into the sweep cache (memory→disk→Supabase). GPU list "
             "defaults to the parallel-chart series; --gpu / --all-gpus expand "
             "it. Alone it replaces the gallery warm; with --landing it runs "
             "after it.",
    )
    parser.add_argument(
        "--model", action="append", dest="models", metavar="MODEL_ID",
        help="With --sweeps: model id to warm (repeatable). Default: featured set.",
    )
    parser.add_argument(
        "--nice", type=int, default=None,
        help="Run at least this niceness (env HA_WARM_NICE, default 10).",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Sweep pool size for this process (env HA_WARM_WORKERS, "
             "default min(32, cpu-2)).",
    )
    parser.add_argument(
        "--show-config", action="store_true",
        help="Print the cache directories this process writes, and exit.",
    )
    parser.add_argument(
        "--pidfile", default=None,
        help="Write this process's pid here; removed on a clean exit.",
    )
    args = parser.parse_args(argv)
    if args.show_config:
        print(f"gallery cache dir: {_GALLERY_CACHE_DIR}")
        print(f"SOL pool dir:      {sharedSolPool.root}")
        print(f"sweep cache dir:   {_SWEEP_CACHE_DIR}")
        return 0

    process = WarmProcess(
        nice=args.nice if args.nice is not None else WarmProcess.default_nice(),
        workers=(args.workers if args.workers is not None
                 else WarmProcess.default_workers()),
        pidfile=args.pidfile,
    )
    with process:
        rc = 0
        gallerySelected = bool(
            args.landing or args.gallery or args.gpus or args.all_gpus)
        warmScaling = bool(args.sweeps or not gallerySelected)
        warmGallery = bool(
            not args.sweeps or args.landing or args.gallery or args.gpus)
        if warmGallery:
            model_params = _warm_model_facts()
        if args.landing and warmGallery:
            print("warming the homepage default view", flush=True)
            if not _warm_frontier(GALLERY_LANDING_GPU, model_params):
                print("landing warm produced no successful entries",
                      file=sys.stderr)
                rc = 1
        elif warmGallery:
            gpus = _gpu_list(gpus=args.gpus, all_gpus=False,
                             default=_gallery_gpus())
            any_ok = False
            for gpu in gpus:
                if _warm_frontier(gpu, model_params):
                    any_ok = True
            if gpus and not any_ok:
                print("warm produced no successful entries", file=sys.stderr)
                rc = 1
        if warmScaling:
            sweep_gpus = _gpu_list(
                gpus=args.gpus, all_gpus=args.all_gpus,
                default=list(DEFAULT_SWEEP_GPUS),
            )
            if not asyncio.run(warm_sweeps(models=args.models, gpus=sweep_gpus)):
                rc = 1
    return rc


if __name__ == "__main__":
    # Reads only the process env. `python -m backend.warm_caches` is the
    # entry point that loads `.env` first.
    raise SystemExit(main())
