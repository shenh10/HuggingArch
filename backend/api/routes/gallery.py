"""Gallery HTTP routes.

Facts (`/gallery/models`) and the Inspector quick-pick are built by
`backend.api.gallery_overview`. The decode Pareto frontier is
`backend.api.gallery_frontier`. This module only parses query args and
registers the endpoints on `app`.
"""
from __future__ import annotations

from backend.api.gallery_overview import (
    build_model_facts,
    featured_examples_off_loop,
)
from fastapi import HTTPException, Query

from backend.api.gallery_frontier import (FrontierView, build_frontier,
                                          frontier_request_key, frontier_track)
from backend.inference.defaults import (COMM_LATENCY_US_DEFAULT,
                                        GPU_MEMORY_FRACTION_DEFAULT,
                                        LAUNCH_FLOOR_US_DEFAULT)
from backend.inference.sweep_grid import (FRONTIER_ACCEPTANCE_DEFAULT,
                                          FRONTIER_DRAFT_TOKENS, GALLERY_GPUS,
                                          GALLERY_LANDING_GPU, GALLERY_SEQ_LENS)
from .._app import app
from backend.measure.scope import make_key
from ..offload import call_key, spec_generation, within_deadline

# Re-export so `test_sweep_grid` can pin that the route module still
# names the canonical seq axis, not a private copy.
__all__ = ["GALLERY_SEQ_LENS"]


@app.get("/inspector/examples")
async def inspector_examples(min_params: float = 100_000_000_000, limit: int = 4):
    """The Inspector "Try:" quick-pick — the latest `limit` models with at least
    `min_params` params, newest first. Derived from the corpus (never a
    hardcoded list) so it stays current as bigger/newer models are added."""
    args = (int(min_params), int(limit))
    ids = await within_deadline(
        call_key(featured_examples_off_loop, args, {}),
        lambda: featured_examples_off_loop(*args))
    return {"examples": ids}


@app.get("/gallery/models")
async def gallery_models():
    """Every model's deployment-independent facts — params, MoE, attention
    mixers, release date, per-request KV — for the gallery's timeline,
    filters, model list and KV leaderboard. No sweep: the page shows these
    while `/gallery/frontier` computes a GPU view."""
    return await within_deadline(
        make_key("gallery-models", spec_generation()), build_model_facts)


@app.get("/gallery/frontier")
async def gallery_frontier(
        gpu_sku: str = GALLERY_LANDING_GPU,
        gpu_memory_fraction: float = GPU_MEMORY_FRACTION_DEFAULT,
        comm_latency_us: float = Query(COMM_LATENCY_US_DEFAULT, ge=0.0,
                                       le=1000.0),
        launch_floor_us: float = Query(LAUNCH_FLOOR_US_DEFAULT, ge=0.0,
                                       le=1000.0),
        draft_tokens: int = 0,
        acceptance_rate: float = Query(FRONTIER_ACCEPTANCE_DEFAULT, gt=0.0,
                                       le=1.0),
        priority: str = "",
        have_tag: int = 0,
        have: int = Query(0, ge=0)):
    """Every model's decode Pareto frontier on one GPU type — tokens/s per
    user against tokens/s per GPU, over every layout up to
    `FRONTIER_MAX_GPUS` and every per-GPU batch (`backend.api.gallery_frontier`).
    The GPU is one of the Gallery's series (`GALLERY_GPUS`, the picker's
    list and what the warm fills); another is refused rather than swept cold.
    `comm_latency_us` / `launch_floor_us` are the two time floors
    (`defaults`), as the estimate takes them. Each model's frontier is
    cached on its spec and this view (`FrontierView`); the warm fills the
    defaults' (``python -m backend.warm_caches``). `draft_tokens` K > 0 (one
    of `FRONTIER_DRAFT_TOKENS`) reads the speculative view of the same
    entry: the MTP draft's K tokens kept at `acceptance_rate`; a model with
    no MTP draft draws nothing there and says why.

    While the view computes, each 503 carries the models finished since the
    client's last one (`have_tag` / `have`: the work and how many of its
    models the client holds), so the page draws each as it lands.
    `priority` is the comma-separated model ids the client wants first —
    the one it picked, then those on screen; it orders what is not started
    yet and is not part of the view's identity."""
    if gpu_sku not in GALLERY_GPUS:
        raise HTTPException(
            400, f"the gallery frontier covers {', '.join(GALLERY_GPUS)}; "
                 f"got {gpu_sku!r}")
    if draft_tokens and draft_tokens not in FRONTIER_DRAFT_TOKENS:
        raise HTTPException(
            400, f"draft_tokens is 0 or one of {list(FRONTIER_DRAFT_TOKENS)}, "
                 f"got {draft_tokens}")
    view = FrontierView(gpu_sku, float(gpu_memory_fraction),
                        float(comm_latency_us), float(launch_floor_us),
                        int(draft_tokens), float(acceptance_rate))
    # The lake's answer is read per request and is part of the response's
    # identity: a response drawn under one track is never served under
    # another (`frontier_request_key`).
    track = await frontier_track(view)
    return await within_deadline(
        frontier_request_key(view, track), lambda: build_frontier(view, track),
        hint=[m for m in priority.split(",") if m] if priority else None,
        cursor=(have_tag, have))
