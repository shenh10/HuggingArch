"""Cookie-based browser session endpoints.

The browser never sees a token: sign-in, refresh, sign-out and the
OAuth/magic-link exchange all happen server-to-server against GoTrue,
and the session lives in two httpOnly cookies:

  ha_at — the Supabase access JWT. HttpOnly, SameSite=Lax, Path=/.
          Named `__Host-ha_at` under https, and https requests honor ONLY
          that name (a sibling subdomain can plant the bare name but never
          the prefixed one). Read by `backend.service.auth` as a fallback
          when a request carries no Authorization header.
  ha_rt — the refresh token. HttpOnly, SameSite=Strict, Path=/api/auth
          so it is only ever sent to the refresh/logout endpoints here.

Endpoints:
  POST /api/auth/login      — email+password → GoTrue, sets both cookies.
  POST /api/auth/refresh    — ha_rt → new token pair, rotates cookies.
  POST /api/auth/logout     — best-effort GoTrue logout, clears cookies.
  POST /api/auth/exchange   — implicit-flow hash (OAuth / magic link /
                              recovery) hands its tokens over once; the
                              access token is verified before Set-Cookie.
  GET  /api/auth/me         — session probe: user + profile + is_admin.
                              200 with user=null when signed out; 401 when
                              the access cookie expired (client refreshes).
  PUT  /api/auth/password   — change password (cookie or bearer auth).
  POST /api/auth/admin-rpc  — allowlisted admin RPCs, forwards the
                              caller's JWT so SQL-side auth.uid() gates
                              keep working (same semantics as admin_token).

The Authorization: Bearer path stays fully functional — hbench PATs,
scripts, and tests never touch cookies.

CSRF: SameSite=Lax already keeps cross-site POSTs from sending ha_at.
Defense in depth, the middleware below rejects any cookie-authenticated
mutation whose Origin/Referer names a different host. Bearer requests
are unaffected (they are not ambient).
"""
from __future__ import annotations

import logging
import time
from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import Cookie, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend import supabase_env
from backend.api._app import app
from backend.service import auth as _auth
from backend.service import rate_limit as _rate_limit

logger = logging.getLogger(__name__)

ACCESS_COOKIE = "ha_at"
#: Under https the access cookie takes the `__Host-` prefix (host-only,
#: Secure, Path=/ — immune to sibling-subdomain tossing). The prefix
#: requires Secure, so plain-http dev keeps the bare name; readers accept
#: both.
ACCESS_COOKIE_SECURE = "__Host-ha_at"
REFRESH_COOKIE = "ha_rt"
REFRESH_COOKIE_PATH = "/api/auth"
# Display hint only — GoTrue owns the real refresh-token lifetime.
_REFRESH_MAX_AGE = 60 * 60 * 24 * 30

_ADMIN_RPC_ALLOWLIST = frozenset({
    # Admin page mutations
    "add_admin", "remove_admin", "set_super_admin",
    # Admin dashboard read-only stats
    "get_users_count", "get_active_users_today",
    "get_total_login_logs_count", "get_total_analyze_logs_count",
    "get_users_with_admin_status", "get_login_logs_with_email",
    "get_active_users_stats", "get_daily_call_trend",
    "get_latency_summary", "get_top_models", "get_inference_query_dist",
})

_GOTURE_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------

def _secure(request: Request) -> bool:
    """Secure flag when the client-facing scheme is https. Proxy headers
    count only when the deployment trusts them (Fly auto-detected, else
    HA_TRUST_PROXY_HEADERS=1) — a directly-exposed server must not let a
    client talk the Secure flag off its own session cookie."""
    return supabase_env.secure_request(request)


def _set_session_cookies(
    response: Response,
    request: Request,
    session: dict,
) -> None:
    access = session.get("access_token")
    if not access:
        raise HTTPException(status_code=502, detail="Auth provider returned no access token")
    secure = _secure(request)
    expires_in = int(session.get("expires_in") or 3600)
    response.set_cookie(
        ACCESS_COOKIE_SECURE if secure else ACCESS_COOKIE, access,
        max_age=expires_in,
        httponly=True, secure=secure, samesite="lax", path="/",
    )
    refresh = session.get("refresh_token")
    if refresh:
        response.set_cookie(
            REFRESH_COOKIE, refresh, max_age=_REFRESH_MAX_AGE,
            httponly=True, secure=secure, samesite="strict",
            path=REFRESH_COOKIE_PATH,
        )


def _clear_session_cookies(response: Response, request: Request) -> None:
    secure = _secure(request)
    # Delete under both names — the caller may hold whichever variant the
    # scheme that issued it used.
    response.delete_cookie(ACCESS_COOKIE, path="/", secure=False, samesite="lax")
    response.delete_cookie(ACCESS_COOKIE_SECURE, path="/", secure=True, samesite="lax")
    response.delete_cookie(
        REFRESH_COOKIE, path=REFRESH_COOKIE_PATH, secure=secure, samesite="strict",
    )


def _access_cookie(request: Request) -> Optional[str]:
    return _auth.session_cookie(request)


# ---------------------------------------------------------------------------
# GoTrue helpers
# ---------------------------------------------------------------------------

def _gotrue_config() -> tuple[str, str]:
    url, anon = supabase_env.url(), supabase_env.anon_key()
    if not (url and anon):
        raise HTTPException(status_code=503, detail="Auth is not configured on this server")
    return url, anon


def _gotrue_error(r: httpx.Response, fallback: str) -> str:
    try:
        body = r.json()
        msg = body.get("error_description") or body.get("msg") or body.get("message")
        if isinstance(msg, str) and msg:
            return msg
    except ValueError:
        logger.warning("GoTrue answered HTTP %s with a non-JSON body: %.200s",
                       r.status_code, r.text)
    return fallback


def _gotrue_token(grant: dict) -> dict:
    """POST /auth/v1/token with the given grant body. Session JSON on
    success; HTTPException(401) carrying GoTrue's own message otherwise."""
    url, anon = _gotrue_config()
    grant_type = grant.pop("grant_type")
    try:
        r = httpx.post(
            f"{url}/auth/v1/token?grant_type={grant_type}",
            json=grant,
            headers={"apikey": anon, "Content-Type": "application/json"},
            timeout=_GOTURE_TIMEOUT,
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Auth provider unreachable: {e}") from e
    if r.status_code != 200:
        # 4xx: GoTrue rejected the credentials/token itself — except 429,
        # which is the provider's own rate limit ("retry later"), not a
        # verdict on the credential. 5xx: the provider is having a moment.
        # Both surface as 502 so refresh keeps the local session.
        rejected = 400 <= r.status_code < 500 and r.status_code != 429
        raise HTTPException(
            status_code=401 if rejected else 502,
            detail=_gotrue_error(r, f"Auth request failed (HTTP {r.status_code})"),
        )
    return r.json()


# ---------------------------------------------------------------------------
# CSRF guard for cookie-authenticated mutations
# ---------------------------------------------------------------------------

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
# Custom header the SPA sends on /api/auth/* mutations. Cross-site forms
# can't set custom headers, and a cross-site fetch carrying one dies in
# the CORS preflight — so its presence pins the caller to our own origin.
_SESSION_AUTH_HEADER = "x-ha-auth"


@app.middleware("http")
async def _cookie_csrf_guard(request: Request, call_next):
    is_session_route = request.url.path.startswith("/api/auth/")
    if request.method not in _SAFE_METHODS and "authorization" not in request.headers:
        source = request.headers.get("origin") or request.headers.get("referer")
        foreign = bool(source) and (
            urlparse(source).netloc != request.headers.get("host", "")
        )
        if is_session_route:
            # These endpoints SET cookies, so the guard cannot key on cookie
            # presence: a cross-site form POSTs no cookie, yet its Set-Cookie
            # still lands — classic login CSRF / session fixation. The custom
            # header requirement closes that, and covers the no-Origin case.
            if foreign or request.headers.get(_SESSION_AUTH_HEADER) != "1":
                return JSONResponse(
                    {"detail": "Cross-origin request rejected"},
                    status_code=403,
                )
        elif foreign and (
            ACCESS_COOKIE in request.cookies
            or ACCESS_COOKIE_SECURE in request.cookies
            or REFRESH_COOKIE in request.cookies
        ):
            return JSONResponse(
                {"detail": "Cross-origin request rejected"},
                status_code=403,
            )
    response = await call_next(request)
    if is_session_route:
        # Session payloads (and Set-Cookie itself) must never be cached by
        # an intermediary.
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class _LoginBody(BaseModel):
    email: str
    password: str


class _ExchangeBody(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None


class _PasswordBody(BaseModel):
    password: str


class _AdminRpcBody(BaseModel):
    name: str
    args: dict = {}


@app.post("/api/auth/login")
def login(body: _LoginBody, request: Request, response: Response):
    _rate_limit.starts_work()       # each login is its own GoTrue relay
    session = _gotrue_token({
        "grant_type": "password",
        "email": body.email,
        "password": body.password,
    })
    _set_session_cookies(response, request, session)
    return {"user": session.get("user")}


@app.post("/api/auth/refresh")
def refresh(
    request: Request,
    response: Response,
    ha_rt: Optional[str] = Cookie(None),
):
    if not ha_rt:
        raise HTTPException(status_code=401, detail="No session to refresh")
    try:
        session = _gotrue_token({"grant_type": "refresh_token", "refresh_token": ha_rt})
    except HTTPException as e:
        if e.status_code != 401:
            # Transient provider failure — the locally-held session may
            # still be valid, so keep both cookies and let the next page
            # load retry.
            raise
        # The injected Response's headers are dropped when an HTTPException
        # propagates, so build the error reply by hand to clear the cookies.
        out = JSONResponse({"detail": e.detail}, status_code=e.status_code)
        _clear_session_cookies(out, request)
        return out
    _set_session_cookies(response, request, session)
    return {"user": session.get("user")}


@app.post("/api/auth/logout")
def logout(
    request: Request,
    response: Response,
    ha_rt: Optional[str] = Cookie(None),
):
    access = _access_cookie(request)
    if not access and ha_rt and supabase_env.url() and supabase_env.anon_key():
        # The access cookie already expired locally; redeem the refresh
        # token once so the session is revoked server-side too, not just
        # forgotten by this browser.
        try:
            access = _gotrue_token({
                "grant_type": "refresh_token", "refresh_token": ha_rt,
            }).get("access_token")
        except HTTPException:
            access = None
    if access and supabase_env.url() and supabase_env.anon_key():
        try:
            httpx.post(
                f"{supabase_env.url()}/auth/v1/logout",
                headers={
                    "apikey": supabase_env.anon_key() or "",
                    "Authorization": f"Bearer {access}",
                },
                timeout=_GOTURE_TIMEOUT,
            )
        except httpx.RequestError as exc:
            # A GoTrue outage must not trap the local session; the tokens
            # stay valid server-side until they expire, so say so.
            logger.warning("GoTrue logout failed, session not revoked "
                           "server-side: %s", exc)
    _clear_session_cookies(response, request)
    return {"ok": True}


@app.post("/api/auth/exchange")
def exchange(body: _ExchangeBody, request: Request, response: Response):
    """Implicit-flow callback: the URL hash delivers tokens to JS (the one
    step Supabase cannot do server-side). Verify before trusting, set the
    cookies, and the JS discards its copy."""
    _rate_limit.starts_work()       # each exchange is its own GoTrue relay
    claims = _auth.verify_supabase_jwt(body.access_token)
    # The client-supplied expires_in is a hint; never outlive the JWT.
    expires_in = body.expires_in or 3600
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        expires_in = max(0, min(expires_in, int(exp - time.time())))
    _set_session_cookies(response, request, {
        "access_token": body.access_token,
        "refresh_token": body.refresh_token,
        "expires_in": expires_in,
    })
    return {"user": {"id": claims.get("sub"), "email": claims.get("email")}}


@app.get("/api/auth/me")
def me(
    request: Request,
    authorization: Optional[str] = Header(None),
    ha_rt: Optional[str] = Cookie(None),
    user: Optional[dict] = Depends(_auth.get_current_user),
):
    if not supabase_env.browser_auth_available():
        return {"authEnabled": False, "user": None, "profile": None, "isAdmin": False}
    if user is None:
        # The access cookie's Max-Age matches the JWT lifetime, so an
        # expired session arrives with ha_rt alone — answer 401 to send
        # the client through /api/auth/refresh instead of signing it out.
        if ha_rt is not None:
            raise HTTPException(status_code=401, detail="Session expired")
        return {"authEnabled": True, "user": None, "profile": None, "isAdmin": False}
    user_id = user.get("sub")
    if not user_id:
        return {"authEnabled": True, "user": None, "profile": None, "isAdmin": False}

    url, anon = _gotrue_config()
    # Identity fetches must use the same token that was verified — a bearer
    # and a cookie can name different users.
    verified = _auth._extract_bearer(authorization) or _access_cookie(request)
    as_user = {"apikey": anon, "Authorization": f"Bearer {verified}"} if verified else None

    # GoTrue /auth/v1/user fills avatar/full_name metadata; the verified
    # JWT only carries identity claims.
    full_user: dict = {"id": user_id, "email": user.get("email")}
    if as_user:
        try:
            r = httpx.get(f"{url}/auth/v1/user", headers=as_user, timeout=_GOTURE_TIMEOUT)
            if r.status_code == 200 and r.json().get("id"):
                full_user = r.json()
        except (httpx.RequestError, ValueError) as exc:
            # The verified JWT's identity claims stand in; only the
            # avatar / full-name metadata is missing.
            logger.warning("GoTrue /auth/v1/user unavailable for %s: %s",
                           user_id, exc)

    profile = None
    if as_user:
        meta = full_user.get("user_metadata") or {}
        try:
            httpx.post(
                f"{url}/rest/v1/rpc/create_user_profile_if_not_exists",
                json={
                    "p_user_id": user_id,
                    "p_username": meta.get("username") or None,
                    "p_avatar_url": meta.get("avatar_url") or None,
                    "p_full_name": meta.get("full_name") or None,
                },
                headers={**as_user, "Content-Type": "application/json"},
                timeout=_GOTURE_TIMEOUT,
            )
            r = httpx.get(
                f"{url}/rest/v1/profiles?id=eq.{user_id}&select=*",
                headers=as_user,
                timeout=_GOTURE_TIMEOUT,
            )
            if r.status_code == 200:
                rows = r.json()
                profile = rows[0] if rows else None
        except (httpx.RequestError, ValueError):
            profile = None  # signed-in chrome still shows email

    try:
        admin = _auth.is_admin(user_id)
    except (HTTPException, httpx.RequestError, ValueError):
        # Deployments without a service-role key (503) or a Supabase blip
        # can't answer this — the session probe must still succeed (admin
        # routes will refuse on their own when actually called).
        admin = False

    return {
        "authEnabled": True,
        "user": full_user,
        "profile": profile,
        "isAdmin": admin,
    }


@app.put("/api/auth/password")
def update_password(
    body: _PasswordBody,
    token: str = Depends(_auth.current_token),
):
    url, anon = _gotrue_config()
    try:
        r = httpx.put(
            f"{url}/auth/v1/user",
            json={"password": body.password},
            headers={
                "apikey": anon,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=_GOTURE_TIMEOUT,
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Auth provider unreachable: {e}") from e
    if r.status_code != 200:
        raise HTTPException(
            status_code=r.status_code,
            detail=_gotrue_error(r, f"Password update failed (HTTP {r.status_code})"),
        )
    return {"ok": True}


@app.post("/api/auth/admin-rpc")
def admin_rpc(
    body: _AdminRpcBody,
    _admin: dict = Depends(_auth.require_admin),
    token: str = Depends(_auth.admin_token),
):
    """Browser replacement for the admin page's direct PostgREST RPCs.
    Allowlisted names only, and the caller's own JWT is forwarded so the
    SQL-side is_admin(auth.uid()) defense-in-depth still applies."""
    if body.name not in _ADMIN_RPC_ALLOWLIST:
        raise HTTPException(status_code=403, detail=f"RPC {body.name!r} is not allowlisted")
    url, anon = _gotrue_config()
    try:
        r = httpx.post(
            f"{url}/rest/v1/rpc/{body.name}",
            json=body.args,
            headers={
                "apikey": anon,
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=_GOTURE_TIMEOUT,
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Auth provider unreachable: {e}") from e
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"RPC {body.name} failed (HTTP {r.status_code})",
        )
    return r.json() if r.content else {"ok": True}
