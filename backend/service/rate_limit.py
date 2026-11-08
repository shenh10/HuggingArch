"""In-process sliding-window rate limiter for the CPU-heavy routes.

Why service-layer: rate limiting is an operational concern, not analyzer
behavior. The core endpoints don't know about it: the middleware gives a
limited request a meter, and the core's one start-or-join decision
(`offload.within_deadline`) charges it through a hook `ha_api` injects.

Which routes: every route whose handler hands work off the event loop
(`backend.api.offload`, or a thread via `asyncio.to_thread` — the
avatar upload proxies to Supabase Storage in one) is in `_LIMITS`, and
`tests/backend/service/test_rate_limit.py` derives that set from the route
modules so a new heavy endpoint cannot ship unlimited — and that every
limited handler reaches a start (`starts_work`), so none is limited in name
only. Agent flows are deliberately not here.

Key resolution (most-specific wins):
  1. Verified JWT subject — `user:<sub>` (signed-in users)
  2. `Fly-Client-IP`        — `ip:<addr>`, only when proxy headers are
                              trusted (`supabase_env.trust_proxy_headers`;
                              the Fly edge overwrites the header, elsewhere
                              a client could supply it)
  3. request.client.host   — `ip:<addr>` (direct)

What is counted: the work a client starts, not its requests. A request on
a limited route costs one unit of that route's budget the first time it
starts work (`starts_work`), and nothing otherwise. Whether a request starts
work or joins work already running is decided where work is keyed —
`backend.api.offload.within_deadline`, which tells `starts_work` of each
start (wired in `ha_api`). A page polling its in-flight work (the Gallery
asks every second while a cold GPU computes) joins it on every poll and
spends nothing; a client asking for many different computations spends
one unit each. A route whose every request is its own work (a relay to
GoTrue, an upload) calls `starts_work` itself before doing it.

Limits read from env at import time, units per minute; 0 disables a
route. A start that exceeds is refused — the work is never begun — with
HTTP 429 and `Retry-After` set to the seconds until the oldest start in
the window expires.

Limits are intentionally **per-process**. With multi-worker uvicorn the
effective limit is N×configured. Swap the deque store for Redis if a hard
global cap is ever needed.
"""
from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request

from backend import supabase_env
from backend.service import auth as _auth


logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# Per-route limits in work starts per minute (module docstring). Override via env.
_LIMITS: Dict[str, int] = {
    "/analyze":                          _int_env("ANALYZE_RPM",   60),
    # /compare runs one analysis per model, up to five; without its own row
    # it would be a 5x multiplier on the /analyze budget.
    "/compare":                          _int_env("COMPARE_RPM",   20),
    # Same analysis as /analyze, entered through the spec editor; without
    # its own row it was an unmetered door to the same work.
    "/api/spec/analyze":                 _int_env("ANALYZE_RPM",   60),
    "/inference/estimate":               _int_env("INFERENCE_RPM", 120),
    "/inference/version":                _int_env("INFERENCE_RPM", 120),
    # Walks the model once under a precision profile.
    "/precision/effective":              _int_env("INFERENCE_RPM", 120),
    # Builds the model's program to solve the Ulysses head cut — one walk,
    # like the precision lookup.
    "/inference/legal-cp":               _int_env("INFERENCE_RPM", 120),
    # Builds the model's program to find its MTP draft stack — the same
    # spec read as the legal-CP lookup.
    "/inference/speculative-draft":      _int_env("INFERENCE_RPM", 120),
    # The opening deployment's fit walks the model's resident weights; it is
    # the page's first estimate-shaped job, not a cheap form lookup.
    "/inference/default-deployment":     _int_env("INFERENCE_RPM", 120),
    # The same spec read again: the deployment's GPU count on the model's
    # mesh, the media the spec declares, and a raw image through the
    # model's processor each build the model's program in a worker.
    "/inference/parallel-shape":         _int_env("INFERENCE_RPM", 120),
    "/inference/media-inputs":           _int_env("INFERENCE_RPM", 120),
    "/inference/processed-media":        _int_env("INFERENCE_RPM", 120),
    "/inference/compute-graph":          _int_env("INFERENCE_RPM", 120),
    "/inference/estimate-sweep":         _int_env("SWEEP_RPM",     30),
    "/inference/estimate-parallel-sweep": _int_env("SWEEP_RPM",    30),
    # A view of that sweep's cache entry, which it computes on a miss.
    "/inference/parallel-sweep/frontier": _int_env("SWEEP_RPM",    30),
    # The Compare page's rows' sweeps, which it reads (computes on a miss).
    "/inference/estimate-sweep/frontier": _int_env("SWEEP_RPM",    30),
    "/inference/bench/sweep-shapes":     _int_env("SWEEP_RPM",     30),
    # One step as Kernel Bench jobs: a whole-model analysis per call.
    "/inference/bench/plan-jobs":        _int_env("ANALYZE_RPM",   60),
    # A directory listing in a thread — cheap, but the rule the coverage
    # test enforces is total: threaded means metered.
    "/supported-models":                 _int_env("INFERENCE_RPM", 120),
    "/inspector/examples":               _int_env("INFERENCE_RPM", 120),
    # The decode Pareto frontier: a warm GPU is one cache read, a cold one
    # a sweep of the whole library under every layout up to 64 GPUs. A
    # caller cycling the gallery's GPUs can fill the compute pool. A client
    # polling a cold build (every second while models land) joins it and
    # spends nothing.
    "/gallery/frontier":                 _int_env("GALLERY_RPM",   30),
    # The models' own facts (no sweep): one analysis per model when cold,
    # a cache read per model when warm.
    "/gallery/models":                   _int_env("GALLERY_RPM",   30),
    # Playground: compose-model is one whole-model analysis and module-bench
    # one analysis per compared component — the /analyze budget; the
    # skeleton and dims lookups are spec reads.
    "/api/playground/compose-model":     _int_env("ANALYZE_RPM",   60),
    "/api/playground/module-bench":      _int_env("ANALYZE_RPM",   60),
    "/api/playground/model-skeleton":    _int_env("INFERENCE_RPM", 120),
    "/api/playground/components":        _int_env("INFERENCE_RPM", 120),
    # The fired-fusion report loads the spec library and walks the model
    # in a compute worker.
    "/fusion/fired":                     _int_env("INFERENCE_RPM", 120),
    "/api/playground/component-dims":    _int_env("INFERENCE_RPM", 120),
    # Avatar upload proxies the image to Supabase Storage in a thread
    # (upload_avatar stays async for `await file.read()`); same rule.
    "/api/account/avatar":             _int_env("AVATAR_RPM",    30),
    # Login + the implicit-flow exchange proxy credentials/tokens to
    # GoTrue from our IP — without a cap the server is an anonymizing
    # relay for credential stuffing. Anonymous callers are IP-keyed.
    "/api/auth/login":                 _int_env("AUTH_LOGIN_RPM", 10),
    "/api/auth/exchange":              _int_env("AUTH_LOGIN_RPM", 10),
}
_WINDOW_SECONDS = 60.0


# {(route, key): deque[timestamp]} — append on hit, popleft when older
# than _WINDOW_SECONDS. Lock guards the whole map; per-key locking
# isn't worth the complexity at this throughput.
_buckets: Dict[Tuple[str, str], Deque[float]] = {}
_lock = threading.Lock()


def _resolve_key(request: Request) -> str:
    """Pick a stable key from the request. JWT sub > Fly-Client-IP > client.host.
    Fly-Client-IP counts only when proxy headers are trusted (Fly, or an
    operator opt-in) — on a directly-exposed deployment it is client-
    supplied and would let a caller rotate bucket keys at will."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and supabase_env.jwt_verification_available():
        token = auth[len("Bearer "):].strip()
        try:
            claims = _auth.verify_supabase_jwt(token)
            sub = claims.get("sub")
            if sub:
                return f"user:{sub}"
        except Exception:
            pass  # fall through to IP
    if supabase_env.trust_proxy_headers():
        edge_ip = request.headers.get("Fly-Client-IP", "").strip()
        if edge_ip:
            return f"ip:{edge_ip}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def _check_and_record(route: str, key: str, limit: int) -> Tuple[bool, int]:
    """Append `now` to the (route, key) bucket; drop expired entries.
    Returns (allowed, retry_after_sec). When `allowed` is False, the
    int is how many seconds until the oldest call ages out.
    """
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS
    with _lock:
        bucket = _buckets.setdefault((route, key), deque())
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry = max(1, int(bucket[0] + _WINDOW_SECONDS - now) + 1)
            return False, retry
        bucket.append(now)
        return True, 0


class RateLimited(HTTPException):
    """A start refused (429): the client's budget on the route is spent."""

    def __init__(self, route: str, limit: int, retry_after: int):
        super().__init__(
            status_code=429,
            detail=(f"Rate limit exceeded for {route}: {limit} new "
                    f"computations a minute. Retry in {retry_after}s."),
            headers={"Retry-After": str(retry_after),
                     "X-RateLimit-Limit": str(limit)})


class _Meter:
    """One limited request's charge on its route's budget: taken at its
    first start, once however many works the request starts (a Compare of
    five models is one unit)."""

    __slots__ = ("charged", "key", "limit", "route")

    def __init__(self, route: str, key: str, limit: int) -> None:
        self.route, self.key, self.limit = route, key, limit
        self.charged = False

    def start(self) -> None:
        if self.charged:
            return
        allowed, retry_after = _check_and_record(self.route, self.key, self.limit)
        if not allowed:
            logger.info(f"rate-limit hit on {self.route} for {self.key}")
            raise RateLimited(self.route, self.limit, retry_after)
        self.charged = True


#: The meter of the limited request this task serves; None outside one.
_meter: "contextvars.ContextVar[Optional[_Meter]]" = contextvars.ContextVar(
    "rate_limit_meter", default=None)


def starts_work() -> None:
    """The request this task serves starts work: charge its route's budget
    (once per request), or raise `RateLimited` and the work is not begun.
    A no-op outside a limited request. `offload.within_deadline` calls it
    for each work it begins; a route whose requests cannot join one
    another's work calls it before doing its own."""
    meter = _meter.get()
    if meter is not None:
        meter.start()


async def rate_limit_middleware(request: Request, call_next):
    """ASGI middleware. On the routes in `_LIMITS` (any method — the
    compute-graph and corpus routes are GETs and just as heavy) the request
    gets a meter its starts charge (`starts_work`); all others fall
    through. Anonymous users are keyed by IP, so one host cannot exhaust
    the box."""
    path = request.url.path
    limit = _LIMITS.get(path)
    if limit is None or limit <= 0:
        return await call_next(request)
    token = _meter.set(_Meter(path, _resolve_key(request), limit))
    try:
        return await call_next(request)
    finally:
        _meter.reset(token)


def install(app) -> None:
    """Attach the middleware. Called from backend/service/__init__.py
    so it only runs when the service layer is loaded. Skipped when
    every limit is 0 (lets self-hosters disable cleanly via env)."""
    if all(v <= 0 for v in _LIMITS.values()):
        logger.info("rate limiter disabled — all limits 0")
        return
    app.middleware("http")(rate_limit_middleware)
    logger.info(
        "rate limiter installed: %s",
        ", ".join(f"{p}={v}/min" for p, v in _LIMITS.items()),
    )
