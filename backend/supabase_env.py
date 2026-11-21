"""Where the Supabase credentials are read, and what "configured" means.

Nine modules read `SUPABASE_URL` and seven read the service-role key, each
answering "is this deployment wired to Supabase?" for itself. They agreed —
but each spelled the answer its own way (`get(...)` vs `get(..., "")`, some
stripping the trailing slash, some not), so a URL written with a trailing
slash produced a double slash from half of them, and a deployment that set
the URL without a key looked configured to the modules that only checked the
URL.

It sits at the top of `backend` rather than inside `backend.service`
because importing anything from that package runs its `__init__`, which
registers HTTP routes against the shared app — so a spec worker asking
"is Supabase configured?" would drag the whole service layer into its
import graph, and `backend.api._app` into a cycle with itself. Reading two
environment variables needs none of that.

The environment is still read on every call, not captured at import: tests
set these with `monkeypatch.setenv` around individual cases, and a value
frozen at import would ignore them.
"""
from __future__ import annotations

import os
from typing import Optional


def url() -> Optional[str]:
    """The project URL with any trailing slash removed, or None. Every caller
    appends its own `/rest/v1/...`, so the slash belongs to neither side and
    is dropped here once."""
    raw = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    return raw or None


def service_role_key() -> Optional[str]:
    """The key that bypasses row-level security. Server-side only: it has no
    `auth.uid()` context, so a policy written as `is_admin(auth.uid())`
    evaluates against NULL under it."""
    return (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip() or None


def anon_key() -> Optional[str]:
    """The public key a request carries as `apikey` when it should run AS a
    user, with that user's JWT in `Authorization`."""
    return (os.environ.get("SUPABASE_ANON_KEY") or "").strip() or None


def jwt_secret() -> Optional[str]:
    """The secret the project signs user JWTs with."""
    return (os.environ.get("SUPABASE_JWT_SECRET") or "").strip() or None


def configured() -> bool:
    """Whether a server-side Supabase call can be made at all. Both halves:
    a URL without a key is not a configuration, it is half of one."""
    return url() is not None and service_role_key() is not None


def jwt_verification_available() -> bool:
    """Whether the backend can verify a user's Supabase JWT: the project URL
    is set (the JWKS is fetched from it; a legacy HS256 project adds
    `jwt_secret`). This is the multi-user switch — anonymous spec generation
    is off, a globally cached spec is not served, and privileged core routes
    want an admin JWT, exactly when it is true. A different question from
    `configured()`: a deployment may verify users without holding the
    service-role key, and may hold the key without any users to verify."""
    return url() is not None


def browser_auth_available() -> bool:
    """Whether the browser can sign in: the URL plus the public anon key the
    client sends as `apikey`. `/site-config.js` hides the login UI when it
    is false."""
    return url() is not None and anon_key() is not None


def trust_proxy_headers() -> bool:
    """Whether edge-injected headers (`client_ip_header()`, `X-Forwarded-Proto`)
    are trustworthy. On Fly the edge overwrites them, so a client cannot
    spoof them; anywhere else they are client-controlled unless the operator
    opts in (their own proxy strips/sets them). Auto-detects Fly via its
    injected FLY_APP_NAME; HA_TRUST_PROXY_HEADERS=1/0 overrides."""
    v = (os.environ.get("HA_TRUST_PROXY_HEADERS") or "").strip().lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return bool(os.environ.get("FLY_APP_NAME"))


def client_ip_header() -> Optional[str]:
    """The one request header the edge proxy writes the client's address
    into, or None when proxy headers are not trusted. Exactly one header
    counts: behind Cloudflare (`CF-Connecting-IP`) a `Fly-Client-IP` is
    client-supplied, and on Fly the reverse. HA_CLIENT_IP_HEADER names it;
    `Fly-Client-IP` when unset."""
    if not trust_proxy_headers():
        return None
    return (os.environ.get("HA_CLIENT_IP_HEADER") or "").strip() or "Fly-Client-IP"


def secure_request(request) -> bool:
    """Whether the client-facing scheme for this request is https. Proxy
    headers only count when the deployment trusts them. `request` is any
    object with `.url.scheme` and `.headers` (a fastapi.Request) — typed
    loosely so this module stays import-light for workers."""
    if request.url.scheme == "https":
        return True
    if not trust_proxy_headers():
        return False
    return request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"


def service_role() -> tuple[str, str]:
    """`(url, key)` for a caller that has already checked `configured()`.
    Raises rather than handing back an empty string, which is how a request
    reaches Supabase with `apikey: ` and comes back 401."""
    project, key = url(), service_role_key()
    if project is None or key is None:
        raise RuntimeError(
            "Supabase is not configured; call configured() before "
            "service_role() — SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
            "must both be set")
    return project, key
