"""The service layer wired into the HA API (the package docstring): imported
once, by `backend/api/__init__.py`, after the core's routes — never by the
Kernel Bench service, which must not build the HA API.

Importing it:
  • registers the service's HTTP routes against the shared `app`
    (`routes`, a side-effect import);
  • installs the rate-limit middleware (disabled cleanly when every
    configured limit is 0, so self-hosters can opt out via env), and tells
    it of every work a request starts (`offload.set_start_hook`: the core
    decides start vs join, the limiter charges starts only);
  • forwards `/api/bench/*` to the Kernel Bench service, the process that
    owns the lake (`bench_proxy`, docs/measured_calibration.md §5.1);
  • injects the admin gate for the privileged core routes (op workers,
    debug, raw job-key access) — core cannot import service auth; it is
    evaluated per request, so self-host mode stays unchanged (the anonymous
    spec-gen gate reads `supabase_env` directly);
  • injects the owner resolver: on a `calibrate_with_mine` estimate or
    sweep, the caller's user id, resolved from their session in the server
    process before the work is offloaded or keyed — the owner of its
    measurement scope (docs/measured_calibration.md §7.1); without this layer such a request is
    refused (501);
  • binds the durable shared remote tier of the two core KeyedCaches (sweep
    and gallery), write-behind: the upsert runs on a background thread, so a
    request that computed a shareable value returns without waiting on
    Supabase (a batch job, `python -m backend.warm_caches`, drains
    `_cache.WRITE_BEHIND` before exiting). Unbound when this layer is not
    installed (OSS self-host): those caches run local-only.

The shared spec-miss hook is registered by `backend.api` before this layer;
service code re-exports the core cache client for compatibility, but must
not overwrite the hook registration.
"""
from __future__ import annotations

from typing import Optional

from backend.api._access import set_admin_access_gate, set_owner_resolver
from backend.api._app import app, health_detail
from backend.api.offload import set_start_hook
from backend.measure.cache import WriteBehindBackend
from backend.api.gallery_overview import set_gallery_remote
from backend.api.sweep_runtime import set_sweep_remote
from backend.service import auth as _svc_auth
from backend.service import bench_proxy, rate_limit
from backend.service import routes  # noqa: F401  (registers the service's routes)
from backend.service.sweep_cache import SupabaseBackend


def owner_of(authorization: Optional[str], cookie: Optional[str]) -> Optional[str]:
    """The user id of a session or personal access token, None without one."""
    user = _svc_auth.user_or_pat(authorization, cookie)
    return _svc_auth.uid(user) if user is not None else None


rate_limit.install(app)
set_start_hook(rate_limit.starts_work)
bench_proxy.install(app, health_detail)
set_admin_access_gate(_svc_auth.require_admin_if_enabled)
set_owner_resolver(owner_of)
set_sweep_remote(WriteBehindBackend(SupabaseBackend()))
set_gallery_remote(WriteBehindBackend(SupabaseBackend()))
