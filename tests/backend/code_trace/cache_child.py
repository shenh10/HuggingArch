"""One fresh process of a cache-layer oracle (`test_cache_oracles.py`):
does one thing under the process's provider and prints what it recorded as
one JSON line. Caches and the SOL pool live where the environment says
(`HA_SWEEP_CACHE_DIR`, `HA_SOL_POOL_DIR`), so a later process reads what an
earlier one stored.

    python -m tests.backend.code_trace.cache_child point MODEL warm|route
    python -m tests.backend.code_trace.cache_child pool here|pooled
    python -m tests.backend.code_trace.cache_child keep-walk MODEL GPU
    python -m tests.backend.code_trace.cache_child layout-law ITEM_JSON
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

#: The one card the oracles price on.
GPU = "H100-SXM"


def _point_request(model: str):
    """A Context × Batch sweep narrowed to one point: one context, one
    batch, TP1 — the smallest request the sweep cache keeps."""
    from backend.api import warm

    req = warm.default_scaling_batch_request(model, GPU, "tp_tp", 1)
    return req.model_copy(update={"seq_lens": [1024],
                                  "batch_grid": req.batch_grid[:1]})


def _stored_traces() -> list:
    """The trace id of every answer the sweep cache keeps on disk."""
    from backend.measure.cache import TRACE_FIELD

    root = Path(os.environ["HA_SWEEP_CACHE_DIR"])
    return sorted(json.loads(p.read_text())[TRACE_FIELD]
                  for p in root.glob("*.json"))


def _plumbing() -> list:
    """What the sweep cache's own read and write path runs: a throwaway value
    put and read back through the sweep cache itself (`_SWEEP`, with the
    tiers this process wired into it), under a key no sweep has."""
    from backend.api.sweep_runtime import _SWEEP
    from backend.code_trace.provider import recording

    from backend.measure.cache import WRITE_BEHIND

    key = hashlib.sha256(tempfile.mkdtemp().encode()).hexdigest()
    with recording() as rec:
        _SWEEP.get_or_compute(key, lambda: {"v": 1}, remote_ok=True)
        _SWEEP.get(key, remote_ok=True)
        # The remote tier writes on a thread of its own: its path is part of
        # the plumbing only once it has run.
        WRITE_BEHIND.drain(timeout=30)
    return rec["trace"].to_json()


def _point(model: str, entry: str) -> dict:
    """The one-point sweep through the warm job's entry or the HTTP route,
    in a recording of its own: its trace, its answer, what the cache kept,
    and the cache's own plumbing."""
    from backend.api.sweep_runtime import compute_estimate_sweep_cached
    from backend.code_trace.provider import recording

    req = _point_request(model)
    with recording() as rec:
        if entry == "warm":
            out = asyncio.run(compute_estimate_sweep_cached(req))
        else:
            from fastapi.testclient import TestClient
            from backend.api._app import app
            response = TestClient(app).post("/inference/estimate-sweep",
                                            json=req.model_dump(mode="json"))
            assert response.status_code == 200, response.text
            out = response.json()
    blob = json.dumps({k: v for k, v in out.items()
                       if k not in ("code_version", "throughput_cloud")},
                      sort_keys=True, default=str)
    return {"trace": rec["trace"].to_json(),
            "result": hashlib.sha256(blob.encode()).hexdigest(),
            "stored": _stored_traces(), "plumbing": _plumbing()}


def _pool(where: str) -> dict:
    """An engine function over a few items, here or in spawned workers
    (`measure.pool.run_parallel`), in one recording."""
    from backend.code_trace.provider import recording
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.parallel_sweep import layout_label
    from backend.measure import pool

    items = [ParallelConfig(tp_attention=2), ParallelConfig(tp_attention=4)]
    with recording() as rec:
        out = (pool.run_parallel(layout_label, items) if where == "pooled"
               else [layout_label(item) for item in items])
    pool.shutdown()
    return {"trace": rec["trace"].to_json(), "result": out}


def _plan(model: str, gpu: str):
    from backend.api import gallery_frontier as f

    _config, resolved, version = f._resolve_one(model)
    view = f.FrontierView(gpu)
    stored = f._Stored(model, version)
    return f, view, stored, f._walk_plan(resolved, view, scaling=True)


def _keep_walk(model: str, gpu: str) -> dict:
    """The class's records kept, then each one's stored trace and the job
    that walked it (`_layout_law`'s item), by layout."""
    f, view, stored, plan = _plan(model, gpu)
    asyncio.run(f.keepSharedWalk(model, gpu))
    walk_key = stored.serving_key(view)
    out, items = {}, {}
    for i, layout in enumerate(plan["layouts"]):
        name = f.StepLawKey.of_layout(walk_key, layout).record_name
        trace = f.sharedSolPool.recordTrace(walk_key.identity.key, name)
        out[str(i)] = None if trace is None else trace.to_json()
        items[str(i)] = [model, plan["cards"], float(view.gpu_memory_fraction),
                         f._GALLERY_PRECISION.canonical(), layout]
    return {"records": out, "items": items}


def _layout_law(item: str) -> dict:
    """One layout's law walked alone from its item, in a process that did
    nothing before it: no plan, no other layout — every memo it reaches
    misses."""
    from backend.api import gallery_frontier as f
    from backend.code_trace.provider import run_recorded

    _value, trace = run_recorded(None, f._layout_law, tuple(json.loads(item)))
    return {"trace": trace.to_json()}


def main(what: str, *args: str) -> None:
    out = {"point": _point, "pool": _pool, "keep-walk": _keep_walk,
           "layout-law": _layout_law}[what](*args)
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
