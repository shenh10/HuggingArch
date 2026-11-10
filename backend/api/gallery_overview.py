"""Gallery facts, KV leaderboard, and the gallery caches.

`/gallery/models` and the Inspector quick-pick read this. The decode Pareto
frontier (`backend.api.gallery_frontier`) keeps its entries and views in
caches of their own here (`_FRONTIER`, `_FRONTIER_VIEWS`) and shares the
resolve helper and `_grid_points`. It does not import the FastAPI app.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.hf.metadata import fetch_config
from backend.arch.eval.precision import PRESETS
from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.loader import get_default_loader
from backend.arch.snapshots import iter_snapshot_ids, load_release_date, snapshot_path
from backend.measure import pool
from backend.inference.capacity import sweep_points
from backend.inference.parallel_config import ParallelConfig
from backend.fingerprint import result_version
from backend.gallery_cache_artifact import GALLERY_CACHE
from backend.measure.cache import (
    KeyedCache, MemoryBackend, TieredBackend, env_bytes,
)
from backend.measure.scope import make_key
from backend.api.offload import (
    ensure_spec_generation,
    map_off_loop,
    report_progress,
    run_off_loop,
    spec_generation,
)

logger = logging.getLogger(__name__)

# The per-request KV is config-independent and cheap (a storage walk, not a
# sweep): the two contexts the leaderboard bar chart draws, each as its
# formula, without widening any sweep grid.
KV_SEQ_LENS = [262144, 1048576]
# The same stamp for a model's deployment-independent facts entry
# (`model_facts_cached`) — bump when `_model_facts_into` changes its fields.
_FACTS_SCHEMA = "gallery-facts-v6"
KV_TITLE = "Per-request KV cache"
#: The KV leaderboard's precisions: every preset that sets nothing but the
#: cache (as shipped sets nothing). The leaderboard draws the cache alone,
#: so a preset that also sets weights would draw as one of these.
KV_PRECISIONS = {pid: p for pid, p in PRESETS.items()
                 if not p.layouts and not any(
                     rules for table, rules in p.tables().items()
                     if table != "cache")}


def _model_concurrency() -> int:
    """How many models build at once: half the compute pool. A model's facts
    or frontier entry is one job, so half the workers stay free for every
    other request while a cold gallery builds."""
    return max(1, pool.worker_count() // 2)

# --- Persistent per-model render cache -----------------------------------
# A model's gallery facts are a function of its spec and the engine. We key
# a disk cache on exactly those, so:
#   • a spec that didn't change → its entry is reused verbatim, even across
#     a process restart;
#   • a spec edit → only that model's key changes → only it recomputes;
#   • an engine change → the engine fingerprint changes → every key misses.
# The cache is content-addressed (filename = sha of the key), so a stale entry
# is never served — a changed input simply lands on a different file.
# Declared once with the artifact that bootstraps it at container start.
_GALLERY_CACHE_DIR = GALLERY_CACHE.directory()
# The gallery's disk tiers all live under that one directory, so they share
# one operator budget — the directory is the disk the operator sizes — and
# the split among the tiers is a code fact, not three more knobs: the
# frontier entries are the mass (up to ~5 MB each, compressed) and get the
# largest share, the model facts one entry a model, the views kilobytes
# beside either.
# The shares sum to the whole; the remote tier, not this disk, is the
# durable one, so a bigger box raises the one knob.
_GALLERY_DISK_BYTES = env_bytes("HA_GALLERY_CACHE_BYTES", 1 << 30)
_FACTS_DISK_BYTES = _GALLERY_DISK_BYTES // 4          # 25%
_FRONTIER_DISK_BYTES = _GALLERY_DISK_BYTES * 13 // 20  # 65%
_VIEWS_DISK_BYTES = _GALLERY_DISK_BYTES // 10          # 10%
# Gallery render cache — disk-backed (persists across restarts) with
# single-flight so two concurrent cold hits for one model render once. Shares
# the cache machinery (`backend.measure.cache`) with the sweep +
# supported-models caches; its keys fold `backend.fingerprint`'s stamps.
#
# Remote tier (Supabase, shared with the sweep cache table): gallery facts
# and frontier entries are shareable — they never take request-supplied
# calibration rows. Two gates keep the global table clean: `remote_if` keeps
# error ENTRIES local (one box's transient failure must not become every
# deployment's homepage).
_GALLERY = KeyedCache(
    "gallery",
    TieredBackend(
        MemoryBackend(max_entries=256),
        GALLERY_CACHE.disk(max_files=10000, max_bytes=_FACTS_DISK_BYTES),
    ),
    single_flight=True,
)

# A model's decode-frontier entry (`gallery_frontier`) is megabytes of JSON,
# most of it each point's step as a function of the floors at every K
# (`FloorForm`) — DeepSeek-V3 on H100 with every layout: 9.0 MB, Kimi-K3:
# 0.85 MB, DeepSeek-V4-Flash: 30.7 MB; all 63 models on H100: 357 MB
# (measured 2026-11-10) — one per model per GPU view (6 GPUs × every
# model, ~2.3 GB of JSON). It is kept in two parts on disk and in the
# remote tier only (`_FRONTIER`): no process holds an entry in memory. On
# disk an entry is gzip level 1 (`DiskBackend(compress=True)`, declared on
# the artifact's tier): DeepSeek-V3 1.6 MB, V4-Flash 5.3 MB, the six GPUs'
# set ~0.4 GB inside this tier's share; reading V4-Flash's costs 0.69 s
# against 0.61 s plain, writing it 0.19 s more. The remote tier keeps the
# JSON (jsonb), so an entry above its 8 MiB limit stays local.
# What a page reads is a view's drawing — the frontier points under its
# floors, K and acceptance, and the Kernel Bench coverage of those points,
# ~21 KB for DeepSeek-V3 — and, per Kernel Bench snapshot, the entry's
# measured rates with each point's
# coverage, ~300 KB for DeepSeek-V3 on H100 (2,545 distinct unmeasured keys)
# (`_FRONTIER_VIEWS`): small next to an entry, so they get a memory tier,
# which holds a few dozen of the larger ones; the rest read from disk.
#
# The memory budget is per process that reads them — every API compute
# worker (`HA_SWEEP_POOL_WORKERS`, spawned, each with its own tiers) and the
# warm process — so a container holds (workers + 1) × the budget × ~4 (a
# value's Python objects over its JSON, measured 4.0 on a frontier entry):
# on the 6 GiB VM's 2 workers, 3 × 8 MiB × 4 ≈ 0.1 GiB beside the cells
# cache (fly.toml). Drawing a view not yet kept parses the entry's first
# part (DeepSeek-V3: 9.7 MB of JSON, 0.2 s; Kimi-K3: 24 MB, ~0.13 GiB of
# objects), and a Kernel Bench snapshot not yet folded parses the measures
# once per model and GPU view (Kimi-K3: 126 MB, ~0.5 GiB of objects). Each
# is one compute job's transient, inside the ~0.85 GiB fly.toml already
# allows a worker's job while it runs; a worker runs one job at a time.
FRONTIER_VIEW_MEMORY_BYTES = 8 << 20
_FRONTIER = KeyedCache(
    "gallery-frontier",
    GALLERY_CACHE.disk("frontier", max_files=4000,
                       max_bytes=_FRONTIER_DISK_BYTES),
    single_flight=True,
)
_FRONTIER_VIEWS = KeyedCache(
    "gallery-frontier-views",
    TieredBackend(
        MemoryBackend(max_entries=4096, max_bytes=FRONTIER_VIEW_MEMORY_BYTES),
        GALLERY_CACHE.disk("frontier-views", max_files=20000,
                           max_bytes=_VIEWS_DISK_BYTES),
    ),
    single_flight=True,
)

def set_gallery_remote(remote) -> None:
    """Inject the durable shared remote tier for the gallery caches (facts
    and frontier entries). Called by the service layer at import so this
    core module never imports `backend.service`. Unset → local-only (OSS
    self-host without Supabase)."""
    _GALLERY.set_remote(remote)
    _FRONTIER.set_remote(remote)


def _resolve_one(model_id: str) -> Tuple[Any, Any, str]:
    """(config, resolved_spec, result_version) for one model — the cheap part
    (~7 ms), run before the expensive work to form the cache key. Never
    raises: an unresolved/erroring model gets a sentinel hash and still
    computes (its failure is captured downstream in the entry's ``error``
    field)."""
    try:
        config = fetch_config(model_id)
        resolved = (
            get_default_loader().match(config, model_id=model_id)
            if config is not None
            else None
        )
    except Exception:
        config, resolved = None, None
    # No spec (or no config right now) → no version → no cache key: the
    # entry is computed and shown, not persisted under a sentinel where a
    # transient failure would outlive its cause.
    version = result_version(resolved) if resolved is not None else None
    return config, resolved, version


def _family_of(model_id: str) -> str:
    """Owner slug — drives the gallery's family colour grouping."""
    return (model_id.split("/", 1)[0] or model_id).lower()


def _attention_families(resolved: Any) -> List[str]:
    """The attention / sequence mixers the backbone runs
    (`Program.attention_mix`) — the badge's and the KV card's answer. A
    hybrid model has several (Kimi-K3: MLA beside KDA)."""
    if resolved is None:
        return []
    return list(Program.of(resolved).attention_mix)


def _parallel_label(pc: ParallelConfig) -> str:
    return f"TP{pc.tp_attention}/EP{pc.ep_moe}/CP{pc.cp}/DP{pc.dp_attention}"


def _model_facts_into(
    entry: Dict[str, Any], model_id: str, config: Any, resolved: Any,
) -> None:
    """Fill a model's deployment-independent facts into `entry`: release
    date, attention mixers, params, MoE, per-request KV. They read the
    model alone — no GPU, no parallelism — so the gallery serves them on
    their own (`model_facts_cached`) before any sweep has run. Raises on a
    model without a config; the caller records it."""
    # Per-checkpoint release date — snapshot metadata, independent of whether
    # the config resolves to a spec, so an undated/unmatched model still ships
    # the field (as None) and the timeline chart just omits it.
    entry["release_date"] = load_release_date(model_id)
    if config is None:
        raise ValueError(f"config unavailable for {model_id}")
    if resolved is None:
        return
    entry["attention_families"] = _attention_families(resolved)
    # --- main params (aux groups like MTP excluded, matching the
    # convention analyzer._analyze_from_spec uses for the dashboard) ---
    # The first production consumer served by `eval` for COMPUTE
    # (storage went first): four scalars, each verified bit-equal to
    # the engine across every shipped snapshot before the switch. No
    # fallback to the engine — a spec eval cannot walk is a failure
    # to surface, not a reason to serve the incumbent's numbers.
    deployment = Deployment(Program.of(resolved)).at(batch=1, input_tokens=1)
    analysis = deployment.analyze()
    main = analysis.main_total()
    entry["total_params"] = int(main.params)
    entry["active_params"] = int(main.active_params)
    # MoE = the graph declares a dispatch zone (tokens routed to
    # experts) — the fact, not the active/total ratio it implies.
    entry["is_moe"] = analysis.program.has_dispatch_zone

    # Per-request KV cache (total) at each context length, under each
    # leaderboard precision — architecture × context × cache format only,
    # independent of the deployment: the backbone's cache as `CacheView`
    # reads it for one request (every tensor under its own storage law,
    # its walk-counted copies), the same reader capacity's budget divides by.
    for pid, profile in KV_PRECISIONS.items():
        cache = cache_view_from_analysis(
            analysis if profile.is_as_shipped
            else deployment.with_precision(profile).analyze())
        entry["kv_per_request"][pid] = {
            str(seq): cache.request_formula(seq) for seq in KV_SEQ_LENS}
        entry["kv_cache_identity"][pid] = cache.storage_identity
        entry["kv_dtypes"][pid] = cache.storage_dtypes(max(KV_SEQ_LENS))


def _kv_precisions() -> List[Dict[str, str]]:
    """The KV leaderboard's choices, as the presets name them."""
    return [{"id": pid, "label": p.label} for pid, p in KV_PRECISIONS.items()]


def _empty_facts(model_id: str) -> Dict[str, Any]:
    return {
        "model_id": model_id,
        "family": _family_of(model_id),
        # Attention display labels (MLA / DSA / KDA / GQA / ...). A hybrid
        # model can have several; empty until a spec resolves.
        "attention_families": [],
        "total_params": None,
        "active_params": None,
        # By leaderboard precision (`KV_PRECISIONS`), then context: the
        # cache total/unsharded — config-independent — as its formula
        # (`CacheView.request_formula`; the bytes are `result_bytes`).
        "kv_per_request": {},
        # By leaderboard precision: `CacheView.storage_identity` — models
        # whose caches are the same draw as one leaderboard row.
        "kv_cache_identity": {},
        # By leaderboard precision: the element dtypes the cache is stored
        # in at the longest context, most bytes first.
        "kv_dtypes": {},
        # Sparse-activation (MoE) vs dense — drives the model-type filter.
        "is_moe": None,
        # Per-checkpoint release date ("YYYY-MM-DD" from the repo's initial
        # public commit), stored in the snapshot JSON. None when undated —
        # timeline omits those.
        "release_date": None,
        "error": None,
    }


def _model_facts_sync(model_id: str, config: Any, resolved: Any) -> dict:
    """One model's deployment-independent facts. Never raises: a failure is
    the entry's `error`."""
    entry = _empty_facts(model_id)
    try:
        _model_facts_into(entry, model_id, config, resolved)
    except Exception as e:
        entry["error"] = str(e)
        logger.warning("gallery: model %s facts failed: %s", model_id, e)
    return entry


def _facts_key(model_id: str, version: str) -> str:
    return make_key(_FACTS_SCHEMA, model_id, version,
                    load_release_date(model_id))


def model_facts_cached(model_id: str) -> dict:
    """`_model_facts_sync` behind the gallery cache, keyed on the model's
    `result_version` and release date — computed once per spec and engine,
    whatever GPU the page later asks for."""
    config, resolved, version = _resolve_one(model_id)
    if version is None:
        return _model_facts_sync(model_id, config, resolved)
    key = _facts_key(model_id, version)
    return _GALLERY.get_or_compute(
        key, lambda: _model_facts_sync(model_id, config, resolved),
        remote_ok=True,
        remote_if=lambda e: not (isinstance(e, dict) and e.get("error")),
    )


def _grid_points(env: dict, *, seq_lens: List[int], batch_grid: List[dict],
                 s_prefixes: Optional[List[int]] = None,
                 s_verifies: Optional[List[int]] = None) -> dict:
    """`{batch key: [point per context]}`: one walk, every (context, batch)
    point by substitution (`capacity.sweep_points`) — a context a prompt
    length, the prefix of it cached (`s_prefixes`, none by default) and
    the verify width it decodes at (`s_verifies`, the request's by
    default) — each `{"sol", "measures"}` before any measurement."""
    plan, walked = sweep_points(seq_lens=seq_lens, batch_grid=batch_grid,
                                s_prefixes=s_prefixes, s_verifies=s_verifies,
                                **env)
    return plan.shape(walked)


def _snapshot_stamp() -> list:
    """Cheap library identity: id + mtime + size per snapshot file.

    Spec edits change the file; the facts-list and featured-examples memos
    then miss. Avoids paying ``loader.match`` just to form the key.
    """
    def _row(mid: str):
        try:
            st = snapshot_path(mid).stat()
            return (mid, st.st_mtime_ns, st.st_size)
        except OSError:
            return (mid, None, None)

    return [_row(mid) for mid in sorted(iter_snapshot_ids())]


async def _per_model(fn: Any, model_ids: List[str], **kwargs: Any) -> List[dict]:
    """`fn(model_id, **kwargs)` for every model, each its own compute job,
    `_model_concurrency` at a time, in `model_ids` order. Reports how many
    have finished (`report_progress`), so a client waiting on a cold build
    sees it advance."""
    sem = asyncio.Semaphore(_model_concurrency())
    done = 0
    report_progress(done, len(model_ids))

    async def one(mid: str) -> dict:
        nonlocal done
        async with sem:
            entry = await run_off_loop(fn, mid, **kwargs)
        done += 1
        report_progress(done, len(model_ids))
        return entry

    return list(await asyncio.gather(*[one(m) for m in model_ids]))


def _facts_probe() -> Tuple[List[str], Dict[str, dict]]:
    """(model ids, the facts entries already in the cache by model id) —
    one job, so a warm list is not one job per model."""
    model_ids = sorted(iter_snapshot_ids())
    hits: Dict[str, dict] = {}
    for mid in model_ids:
        _config, _resolved, version = _resolve_one(mid)
        if version is None:
            continue
        # Local tiers only: a cold list must not wait on one remote read
        # per model here; a miss reads the remote tier in its own job.
        hit = _GALLERY.get(_facts_key(mid, version))
        if hit is not None:
            hits[mid] = hit
    return model_ids, hits


async def build_model_facts() -> dict:
    """Every model's deployment-independent facts (`model_facts_cached`):
    what the gallery's timeline, filters, model list and KV leaderboard
    read. It needs no sweep, so the page shows them while a GPU view is
    still computing. A spec refined meanwhile serves nothing
    (`ensure_spec_generation`)."""
    generation = spec_generation()
    model_ids, hits = await run_off_loop(_facts_probe)
    missing = [mid for mid in model_ids if mid not in hits]
    computed = dict(zip(missing, await _per_model(model_facts_cached, missing),
                        strict=True))
    ensure_spec_generation(generation)
    models = [hits.get(mid) or computed[mid] for mid in model_ids]
    return {
        "kv_title": KV_TITLE,
        "kv_precisions": _kv_precisions(),
        "models": models,
        "errors_count": sum(1 for m in models if m.get("error")),
    }


# --- Featured examples (Inspector "Try:" quick-pick) ----------------------
# The latest N models above a params floor, derived from the corpus so the
# quick-pick list stays current as bigger/newer models land — never a
# hand-maintained list. The floor reads the main model's parameters
# (`Analysis.main_total`: no MTP head, no encoder) — the headline the gallery
# and `/analyze`'s `main_total_params` state. That takes one analysis per
# model, so each model's figure is cached beside its gallery entries, keyed on
# its `result_version`: computed once per spec and engine, not per process.
_EXAMPLES_MEMO: Dict[tuple, list] = {}
_MAIN_PARAMS_SCHEMA = "main-params-v1"


def _main_model_params(resolved: Any) -> int:
    """Parameters of the main model: no MTP head, no encoder."""
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=1).analyze()
    return int(analysis.main_total().params)


def _model_size_and_date(model_id: str) -> Tuple[Optional[int], Optional[str]]:
    """(main-model parameter count, release_date) for one snapshot; None
    for a fact there is no answer to — no spec resolves, or the analysis
    fails, which is logged and leaves the model off the list rather than
    failing the list."""
    release = load_release_date(model_id)
    _config, resolved, version = _resolve_one(model_id)
    if resolved is None:
        return None, release

    def _compute() -> dict:
        try:
            return {"params": _main_model_params(resolved)}
        except Exception as exc:  # one model's failure must not sink the list
            logger.warning("featured examples: %s has no LM size: %s",
                           model_id, exc)
            return {"params": None, "error": str(exc)}

    entry = _GALLERY.get_or_compute(
        make_key(_MAIN_PARAMS_SCHEMA, model_id, version), _compute,
        store_if=lambda e: e.get("params") is not None)
    return entry["params"], release


def _pick_featured(rows: List[Tuple[str, Optional[int], Optional[str]]],
                   min_params: int, limit: int) -> list:
    """The latest `limit` of `(model, params, release)` at or above the
    floor, newest first."""
    dated = [(mid, release) for mid, params, release in rows
             if params is not None and params >= min_params and release]
    dated.sort(key=lambda r: r[1], reverse=True)
    return [mid for mid, _ in dated[:limit]]


def _size_row(model_id: str) -> Tuple[str, Optional[int], Optional[str]]:
    return (model_id, *_model_size_and_date(model_id))


def featured_examples(min_params: int = 100_000_000_000, limit: int = 4) -> list:
    """Model ids of the latest `limit` snapshots whose language model has
    >= `min_params` params, newest first by release date. Memoized on
    library identity."""
    memo_key = (tuple(map(tuple, _snapshot_stamp())), int(min_params), int(limit))
    hit = _EXAMPLES_MEMO.get(memo_key)
    if hit is not None:
        return hit
    out = _pick_featured([_size_row(mid) for mid in sorted(iter_snapshot_ids())],
                         int(min_params), int(limit))
    _EXAMPLES_MEMO[memo_key] = out
    return out


async def featured_examples_off_loop(min_params: int, limit: int) -> list:
    """`featured_examples` for a request: each model's size is its own
    compute job (cached per model), so a cold list spreads across the
    workers. The list is memoized here on library identity and spec
    generation."""
    memo_key = ("off-loop", spec_generation(),
                tuple(map(tuple, _snapshot_stamp())), int(min_params), int(limit))
    hit = _EXAMPLES_MEMO.get(memo_key)
    if hit is not None:
        return hit
    model_ids = sorted(iter_snapshot_ids())
    rows = await map_off_loop(_size_row, model_ids)
    out = _pick_featured(rows, int(min_params), int(limit))
    _EXAMPLES_MEMO[memo_key] = out
    return out
