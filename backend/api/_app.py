"""FastAPI app instance + static-file mounts + CORS middleware.

The single `app` object is created here once and re-imported by every
`routes/<domain>.py` module. Centralizing construction here lets the
per-domain routers register against the same app without circular
imports.
"""
from __future__ import annotations

import inspect
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.gzip import GZipMiddleware

logger = logging.getLogger(__name__)

# Autonomous propose → review → auto-merge pipeline. Watches
# logs/op_proposals/ for gap statements the spec_worker drops and drives
# them through the same SSE generator that `POST /propose-op?chain_review=true`
# uses. On by default; OP_PIPELINE_ENABLED=0 turns it off.
from backend.api import offload as _offload
from backend.code_trace.provider import seal as code_trace_seal
from backend.measure import pool
from backend.measure.lookup import MeasurementsUnavailable
from backend.op_workers.pipeline import coordinator as _op_pipeline
from backend.spec_worker import runtime as _spec_rt

# Stall forensics: with HA_FAULTHANDLER=1 (the `ha` supervisor sets it),
# SIGUSR1 dumps every thread's traceback to stderr (→ /tmp/ha_server.log).
# The event loop has wedged for ~2min at a time with the process alive
# (cloudflared TLS-handshake timeouts → CF 502s); this is how we see where.
# Off by default so pytest/CI stays quiet.
if os.environ.get("HA_FAULTHANDLER"):
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)


#: Background loops a route module owns (the Kernel Bench maintainer),
#: started by the lifespan once the server is up. FastAPI runs no
#: `@app.on_event("startup")` hook when a lifespan is set.
_background_loops: list = []


def run_in_background(start):
    """Register `start()` to be called when the server starts (not under
    pytest); it starts its own daemon thread and returns."""
    _background_loops.append(start)
    return start


def _memory_report() -> dict:
    """The machine's memory as `/health`'s `detail.memory` — what an
    approaching OOM looks like before it kills the process. `used_pct` is
    read off `MemAvailable`, the kernel's own estimate of what is left once
    reclaimable cache is reclaimed, not a naive total-minus-free."""
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                key, _, value = line.partition(":")
                info[key] = int(value.strip().split()[0]) // 1024  # kB → MB
    except OSError:
        return {"error": "no /proc/meminfo"}
    total, available = info.get("MemTotal"), info.get("MemAvailable")
    return {
        "total_mb": total,
        "available_mb": available,
        "used_pct": (round(100.0 * (total - available) / total, 1)
                     if total and available is not None else None),
    }


#: Named reports the service layer adds to `/health`'s `detail` (the Kernel
#: Bench service's health). Each is called per request and must be cheap and
#: bounded (a coroutine is awaited); one that raises is reported as its
#: error, never failing /health.
_health_details: dict = {}


def health_detail(name: str, report) -> None:
    """Register `report()` (a JSON-able dict, or a coroutine of one) as
    `/health`'s `detail[name]`."""
    _health_details[name] = report


async def health_details() -> dict:
    out = {}
    for name, report in _health_details.items():
        try:
            value = report()
            out[name] = (await value) if inspect.isawaitable(value) else value
        except Exception as e:      # /health itself must never fail
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


health_detail("memory", _memory_report)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    await _op_pipeline.startup()
    # The cache warm runs out of process (`python -m backend.warm_caches`, low
    # priority, its own sweep pool): inside this process it would compete
    # with user requests for the GIL and the sweep pool. The server reads
    # the caches the warm fills and computes on a miss.
    #
    # Do NOT spawn the compute process pool here. Lifespan runs before bind;
    # a failed bind (EADDRINUSE) used to leak those workers as PID-1 orphans.
    # The pool starts a few seconds later, or on the first heavy request
    # if that comes first. Heavy request work runs in it, within the
    # request deadline (`backend.api.offload`): this process only routes.
    # Under pytest the app stays in-process (tests patch in-process seams).
    if not pool.in_test_run():
        _offload.serve_requests_off_process()
        _offload.prestart_workers_later()
    _spec_rt.recover_on_boot()
    if not pool.in_test_run():
        for start in _background_loops:
            start()
    # The code this server runs is fixed before it answers anything
    # (`code_trace.provider.seal`; nothing under `GlobalHash`) — last, so the
    # workers' prestart timer is not held back by it.
    code_trace_seal()
    try:
        yield
    finally:
        _spec_rt.interrupt_on_shutdown()
        await _op_pipeline.shutdown()
        _offload.stop_serving()
        pool.shutdown(freeze=True)


app = FastAPI(title="HuggingArch API", lifespan=_lifespan)


@app.exception_handler(_spec_rt.ServerDraining)
async def _server_draining(_request, exc: _spec_rt.ServerDraining):
    """Every spec-job admission goes through `spec_worker.acquire`; during a
    restart drain it raises, and the caller gets one clear 503."""
    return JSONResponse(status_code=503, content={
        "success": False, "status": "restarting", "message": str(exc)})


@app.exception_handler(_offload.StillComputing)
@app.exception_handler(_offload.ComputeUnavailable)
@app.exception_handler(MeasurementsUnavailable)
async def _compute_not_ready(_request, exc: Exception):
    """Heavy work that missed the request deadline (it goes on, and the same
    request joins it), spanned a spec refine, found no compute pool to run
    on, or could not read its measurements (the Kernel Bench service did not
    answer — nothing was computed or cached): one 503 naming which, that the
    client may send again."""
    after = _offload.RETRY_AFTER_S
    if isinstance(exc, _offload.StillComputing) and exc.retry_after_s is not None:
        after = exc.retry_after_s
    content = {"detail": str(exc), "reason": exc.reason, "retry_after_s": after}
    if isinstance(exc, _offload.StillComputing):
        if exc.progress is not None:
            done, total = exc.progress
            content["progress"] = {"done": done, "total": total}
        if exc.jobs_done is not None:
            content["jobs_done"] = exc.jobs_done
        if exc.partial is not None:
            content["partial"] = jsonable_encoder(exc.partial)
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": str(int(after))},
        content=content)


# React SPA static files. Logo: web/public/static → /static/logo.png.
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_web_dist_path = os.path.join(_repo_root, "web", "dist")
_web_public_static = os.path.join(_repo_root, "web", "public", "static")


def require_web_dist_if_configured(dist_path: str | None = None) -> None:
    """Production (`HA_REQUIRE_WEB_DIST=1`) refuses to boot without a Vite build.

    Pytest and local `uv run` leave the variable unset so `import app` stays
    cheap. Docker sets it. Top-tab URLs (`/`, `/inference.html`, …) serve
    the React SPA; missing dist in production is a boot failure.
    """
    if os.environ.get("HA_REQUIRE_WEB_DIST", "").lower() not in ("1", "true", "yes"):
        return
    path = dist_path if dist_path is not None else _web_dist_path
    if not os.path.isfile(os.path.join(path, "index.html")):
        raise RuntimeError(
            "HA_REQUIRE_WEB_DIST=1 but web/dist/index.html is missing. "
            "The image build must copy the Vite output."
        )


require_web_dist_if_configured()
_static_dir = os.path.join(_web_dist_path, "static")
if not os.path.isdir(_static_dir):
    _static_dir = _web_public_static
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Mounted whether or not the build is there yet: the directory is read per
# request, so a `web/dist` built after this import (a dev server started
# first, the test that builds it in the same process) is served without a
# restart, and a missing one answers 404 per request.
class _BuiltLaterStaticFiles(StaticFiles):
    """StaticFiles whose directory may appear after the app is built: a file
    asked for before then is a 404, not a 500 for a missing directory."""

    async def check_config(self) -> None:
        if self.directory is not None and os.path.isdir(self.directory):
            await super().check_config()


_web_assets_path = os.path.join(_web_dist_path, "web-assets")
app.mount("/web-assets", _BuiltLaterStaticFiles(directory=_web_assets_path, check_dir=False),
          name="web-assets")
if not os.path.isdir(_web_assets_path):
    logger.info("/web-assets: no %s yet; served once it is built", _web_assets_path)
logger.info("homepage: React SPA at / when web/dist is snapshotted")


# CORS. The SPA is same-origin, so the default is no cross-origin caller
# at all beyond the production host. `CORS_ORIGINS` (comma-separated) widens
# it. A wildcard together with credentials is refused rather than shipped:
# browsers reject that combination for credentialed requests, so it only
# ever meant "nobody looked at this".
_cors_origins = [
    o.strip() for o in (os.environ.get("CORS_ORIGINS")
                        or "https://www.huggingarch.com").split(",")
    if o.strip()
]
_cors_credentials = "*" not in _cors_origins
if "*" in _cors_origins and len(_cors_origins) > 1:
    raise RuntimeError("CORS_ORIGINS: '*' cannot be combined with named origins")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Last-added = outermost, so this compresses after cache-control is set.
# Bundle JS goes over the dev tunnel; gzip cuts it to roughly a third.
# compresslevel 5 is plenty for on-the-fly.
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)


# Never let NON-HASHED frontend assets be cached. When the server sits behind a
# CDN / tunnel, StaticFiles sends only ETag /
# Last-Modified with NO Cache-Control — so Cloudflare edge-caches the .js/.css
# by default and serves a STALE bundle that a browser hard-refresh can't bust
# (the edge is in front of the browser). That's the "I deployed a fix but the
# page still shows the old bug / OOMs" failure. `no-store` on every frontend
# response (HTML + /web-assets + /static) tells CF and the browser to always
# fetch fresh from origin. API (JSON) responses are untouched.
#
# Exception: /web-assets/<name>-<hash>.{js,css}. Vite puts the content hash in
# the filename and rewrites web/dist/index.html to point at it on every build,
# so a stale copy is impossible — these get immutable caching, and a page
# switch does not re-download the bundle through the tunnel.
_DIST_HASHED = re.compile(
    r"^/web-assets/[^/]+-[A-Za-z0-9_-]{8}\.(js|css)$"
)


@app.middleware("http")
async def _no_store_frontend(request, call_next):
    resp = await call_next(request)
    path = request.url.path
    if _DIST_HASHED.match(path):
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif (path == "/" or path.startswith(("/static", "/web-assets")) or path.endswith((".html", ".js", ".css", ".map"))):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


def get_web_dist_path() -> str:
    """Vite production output (`web/dist`). Empty / missing until `npm run build`."""
    return _web_dist_path


_spa_index = os.path.join(_web_dist_path, "index.html")
_SPA_INDEX = _spa_index if os.path.isfile(_spa_index) else None
if _SPA_INDEX:
    logger.info("SPA index: %s (top-tab URLs)", _SPA_INDEX)
else:
    logger.info("SPA index: missing; top-tab URLs return 503")


def spa_index_path() -> str | None:
    """Import-time snapshot of web/dist/index.html, or None."""
    return _SPA_INDEX


def spa_page_response():
    """Production entry for top-tab URLs: React dist only."""
    if _SPA_INDEX:
        return FileResponse(_SPA_INDEX)
    return JSONResponse(
        {"message": "React SPA missing — run npm run build --prefix web"},
        status_code=503,
    )


def public_file_response(name: str):
    """A root-level file from `web/public` (`favicon.ico`, `robots.txt`):
    Vite copies it to the root of `web/dist`; before a build it is read from
    the source directory."""
    for root in (_web_dist_path, os.path.join(_repo_root, "web", "public")):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})
    return JSONResponse({"detail": "Not Found"}, status_code=404)


@app.exception_handler(StarletteHTTPException)
async def _page_not_found(request, exc: StarletteHTTPException):
    """A browser opening a URL no route serves gets the site's not-found
    page (the SPA renders it for any path it does not know), still with
    status 404. Everything else — API calls, which do not ask for HTML, a
    404 an endpoint raised itself, any other status — keeps FastAPI's JSON
    answer."""
    no_route = "endpoint" not in request.scope
    wants_page = "text/html" in request.headers.get("accept", "")
    if (exc.status_code == 404 and no_route and wants_page
            and request.method in ("GET", "HEAD") and _SPA_INDEX):
        return FileResponse(_SPA_INDEX, status_code=404)
    return await http_exception_handler(request, exc)
