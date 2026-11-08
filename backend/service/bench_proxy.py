"""The HA API's side of Kernel Bench (docs/measured_calibration.md §5.1):
`/api/bench/*` forwarded to the Kernel Bench service.

The service (`backend.service.bench.server`, 127.0.0.1) owns the lake and
authenticates no one. This proxy does, as the bench routes did before they
moved: a session (bearer JWT or the scheme's access cookie), and a personal
access token on the measurement pipeline's upload, nominate and promote routes.
Credentials that do not verify are 401 here and never reach the service. Then it forwards:

  * headers built fresh — the body's `Content-Type`, and the three it alone
    sets: `X-HA-User` (the user id; empty when anonymous), `X-HA-Admin` (`1`
    for an admin, `public.admins`) and `X-HA-Internal` (the shared secret).
    Nothing else of the browser's — no cookie, no bearer, no `X-HA-*` it
    spelled — is sent;
  * the method, the raw path and query; the body streamed up, the answer
    streamed down (CSV downloads, uploads up to the 200k-row bound);
  * within a timeout per route: reads (GET) 30 s, writes 120 s;
  * cancelled upstream when the client disconnects.

One rule for the headers that cross, both ways: a body's `Content-Type`
travels with it, nothing else does. The service reads JSON bodies and
answers JSON only (every route returns a `JSONResponse` or a model; no
download, no conditional request, no `Retry-After` of its own) — the rest of
an exchange's headers are the caller's credentials, framing, or this proxy's
own (its 503 sets `Retry-After`).

Answers: the service's own status and body pass through unchanged (400,
401, 403, 404, 413, 422, its 503 naming why the lake does not serve…); the
service unreachable, timing out or failing (5xx other than 503) is one 503
"Kernel Bench is unavailable" with `Retry-After`. Redirects are not
followed; `/internal/bench/*` is not routable from here.

`/health` carries the service's health (`bench_health`, bounded at 2 s),
registered by `install`'s caller.
"""
from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any, AsyncIterator, Callable, Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from backend.measure import lookup as measurements
from backend.service import auth as _auth

_LOG = logging.getLogger(__name__)

READ_TIMEOUT_S = 30.0
WRITE_TIMEOUT_S = 120.0
CONNECT_TIMEOUT_S = 2.0
HEALTH_TIMEOUT_S = 2.0
RETRY_AFTER_S = 5
_UNAVAILABLE = "Kernel Bench is unavailable"



def _body_type(headers: Any) -> dict:
    """The one header that crosses the proxy, either way (module docstring):
    the body's `Content-Type`, when it has one."""
    return {"content-type": headers["content-type"]} if "content-type" in headers else {}

#: Tests route the proxy to the service in process (`httpx.ASGITransport`).
transport: Optional[httpx.AsyncBaseTransport] = None

_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary())


def _client() -> httpx.AsyncClient:
    """One keep-alive client per event loop (a client belongs to its loop)."""
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None:
        # Never through an environment proxy (http_proxy / HTTPS_PROXY): the
        # service is on loopback and every request carries the secret.
        client = _clients[loop] = httpx.AsyncClient(
            transport=transport, follow_redirects=False, timeout=READ_TIMEOUT_S,
            trust_env=False)
    return client


def _unavailable(why: str) -> JSONResponse:
    _LOG.warning("Kernel Bench proxy: %s", why)
    return JSONResponse(status_code=503, headers={"Retry-After": str(RETRY_AFTER_S)},
                        content={"detail": _UNAVAILABLE})


def _caller(request: Request) -> Optional[dict]:
    """The verified caller (401 for credentials that do not verify), None
    when anonymous: a session — or a personal access token on the measurement
    pipeline's upload, nominate and promote routes.  A one-hour session JWT is
    not a credential a GPU box can use."""
    authorization = request.headers.get("authorization")
    parts = request.url.path.strip("/").split("/")
    pipeline = (
        request.method == "POST" and (
            request.url.path in ("/api/bench/upload", "/api/bench/promote")
            or (len(parts) in (4, 5) and parts[:2] == ["api", "bench"]
                and parts[-1] == "contribute")))
    if pipeline:
        return _auth.user_or_pat(authorization, _auth.session_cookie(request))
    return _auth.get_current_user(request, authorization)


def identity_headers(user: Optional[dict]) -> dict:
    """The three headers only the API sets."""
    uid = _auth.uid(user) if user is not None else ""
    return {measurements.USER_HEADER: uid,
            measurements.ADMIN_HEADER: "1" if uid and _auth.is_admin(uid) else "0",
            measurements.INTERNAL_HEADER: measurements.secret()}


class _Body:
    """The request body streamed up; `done` once it is all read (only then
    may the disconnect watcher read the connection)."""

    def __init__(self, request: Request):
        self.request = request
        self.done = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.request.stream():
                if chunk:
                    yield chunk
        finally:
            self.done.set()


async def _until_disconnected(request: Request, body: _Body) -> None:
    await body.done.wait()
    while not await request.is_disconnected():
        await asyncio.sleep(0.25)


async def forward(request: Request) -> Response:
    """`/api/bench/*`: authenticate, forward, stream the answer back."""
    base = measurements.service_url()
    if not base:
        return _unavailable("HUGGINGARCH_BENCH_URL is not set")
    user = await asyncio.to_thread(_caller, request)
    headers = {**await asyncio.to_thread(identity_headers, user),
               **_body_type(request.headers)}
    raw_path = request.scope.get("raw_path") or request.url.path.encode()
    url = base + raw_path.decode("latin-1")
    if request.url.query:
        url += "?" + request.url.query
    limit = READ_TIMEOUT_S if request.method in ("GET", "HEAD") else WRITE_TIMEOUT_S
    body = _Body(request)
    upstream = _client().build_request(
        request.method, url, headers=headers,
        content=None if request.method in ("GET", "HEAD") else body,
        timeout=httpx.Timeout(limit, connect=CONNECT_TIMEOUT_S))
    if request.method in ("GET", "HEAD"):
        body.done.set()
    send = asyncio.ensure_future(_client().send(upstream, stream=True))
    gone = asyncio.ensure_future(_until_disconnected(request, body))
    try:
        await asyncio.wait({send, gone}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        gone.cancel()
    if not send.done():
        send.cancel()           # the client went away: so does the request
        return Response(status_code=499)
    try:
        resp = send.result()
    except ClientDisconnect:
        return Response(status_code=499)
    except httpx.HTTPError as e:
        return _unavailable(f"{type(e).__name__}: {e}")
    if resp.status_code >= 500 and resp.status_code != 503:
        await resp.aclose()
        return _unavailable(f"the service answered {resp.status_code}")
    out_headers = _body_type(resp.headers)

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        except httpx.HTTPError as e:
            _LOG.warning("Kernel Bench proxy: the answer broke off: %s", e)
        finally:
            await resp.aclose()

    return StreamingResponse(stream(), status_code=resp.status_code, headers=out_headers)


async def bench_health() -> dict:
    """`/health`'s Kernel Bench detail: the service's own health (its lake's
    state, never a lake read there), or why it cannot be had.  A lake that is
    starting a required data migration, or failed one, is reported at this
    level as `degraded`; `/health` must not call an unmigrated lake `up`."""
    base = measurements.service_url()
    if not base:
        return {"state": "not_configured"}
    try:
        r = await _client().get(
            base + "/health", headers={measurements.INTERNAL_HEADER: measurements.secret()},
            timeout=HEALTH_TIMEOUT_S)
    except httpx.HTTPError as e:
        return {"state": "unreachable", "reason": f"{type(e).__name__}: {e}"}
    if r.status_code != 200:
        return {"state": "unreachable", "reason": f"the service answered {r.status_code}"}
    body = r.json()
    out = dict(body)
    out.setdefault("state", "up")
    lake = out.get("lake")
    if isinstance(lake, dict) and lake.get("state") in ("degraded", "unavailable"):
        out["state"] = "degraded"
    return out


def install(app: Any, health_detail: Callable[[str, Any], None]) -> None:
    """Route `/api/bench/*` through the proxy, and register `bench_health` as
    `/health`'s `kernel_bench` detail. Both are handed in by the composition
    root (`backend.service.ha_api`): this module imports nothing of the HA
    API's, so it is importable on its own (a test of the proxy needs no app)
    and no import of it can re-enter `backend.api`'s package."""
    app.add_api_route("/api/bench/{path:path}", forward,
                      methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
                      include_in_schema=False)
    health_detail("kernel_bench", bench_health)
