"""Shared fixtures for backend.service tests.

Every test runs with:
  • SUPABASE_* env vars set to deterministic dummy values
  • httpx_mock intercepting *all* outbound HTTP (so no real Supabase
    call is ever attempted; failure to set up an expected mock surfaces
    as a clear pytest failure rather than a network timeout)
  • backend.service.auth caches reset (admin lookup + JWKS client) so
    cross-test state doesn't leak

JWT helpers `make_user_jwt` / `make_expired_jwt` mint signed tokens
the verifier code path then accepts. Public/private keypair fixture
mirrors what Supabase exposes via /auth/v1/.well-known/jwks.json.
"""
from __future__ import annotations

import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Deterministic env values (no real secrets here)
# ---------------------------------------------------------------------------
TEST_SUPABASE_URL = "https://test-project.supabase.co"
TEST_ANON_KEY = "sb_publishable_test_anon"
TEST_SERVICE_KEY = "sb_secret_test_service_role"
TEST_JWT_SECRET = "test-jwt-secret-deterministic"


@pytest.fixture(autouse=True)
def _service_env(monkeypatch):
    """Pin Supabase env for every service test."""
    monkeypatch.setenv("SUPABASE_URL", TEST_SUPABASE_URL)
    monkeypatch.setenv("SUPABASE_ANON_KEY", TEST_ANON_KEY)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", TEST_SERVICE_KEY)
    monkeypatch.setenv("SUPABASE_JWT_SECRET", TEST_JWT_SECRET)
    # Behave like the Fly deployment: edge-injected proxy headers
    # (Fly-Client-IP, X-Forwarded-Proto) are trusted. Tests for the
    # distrust path delete this.
    monkeypatch.setenv("FLY_APP_NAME", "huggingarch-test")
    # Force HF offline so /analyze + /inference/estimate handlers fail
    # fast on unknown model_ids instead of pausing 30s on retries.
    #
    # The env var alone only works if nothing has imported huggingface_hub
    # yet: `constants.HF_HUB_OFFLINE` is read from the environment once, at
    # import, and `is_offline_mode()` returns that module global. So in a
    # process where some earlier test already imported the library, setting
    # the variable here silently does nothing and the handler really does
    # reach out to huggingface.co — which `httpx_mock` then reports as an
    # unexpected request, at teardown, in whichever test happened to be
    # running. Pin the flag the library actually consults.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    import huggingface_hub.constants as _hf_constants
    monkeypatch.setattr(_hf_constants, "HF_HUB_OFFLINE", True)
    # Rate limiter env: leave at module defaults; specific tests override
    # per-route limits via monkeypatch.setattr against rate_limit._LIMITS.


@pytest.fixture(autouse=True)
def _reset_service_caches():
    """Wipe in-memory caches between tests so admin lookups + JWKS
    fetches don't carry results across cases."""
    from backend.service import auth as _auth
    from backend.service import spec_generation_logs as _logs
    _auth._ADMIN_CACHE.clear()
    _auth._jwks_client = None
    _logs._reset_schema_compat_for_tests()
    yield
    _auth._ADMIN_CACHE.clear()
    _auth._jwks_client = None
    _logs._reset_schema_compat_for_tests()


BENCH_SECRET = "test-bench-secret"


@pytest.fixture(autouse=True)
def _bench_secret(monkeypatch):
    """The secret the HA API and the Kernel Bench service share."""
    monkeypatch.setenv("HUGGINGARCH_BENCH_SECRET", BENCH_SECRET)
    monkeypatch.delenv("HUGGINGARCH_BENCH_SECRET_FILE", raising=False)


@pytest.fixture
def bench_gate(monkeypatch):
    """A fresh `serving.Gate` for the Kernel Bench service's routes (the
    module's one keeps its state across tests otherwise), a fresh shared
    revision and verdict memo; the maintainer not started."""
    from backend.service.bench import lake as lake_mod
    from backend.service.bench import lookup
    from backend.service.bench import serving
    from backend.service.bench import routes as R
    monkeypatch.setenv("HUGGINGARCH_BENCH_MAINTENANCE", "off")
    monkeypatch.delenv("BENCH_LAKE_BACKUP_DATA", raising=False)
    monkeypatch.setattr(lake_mod, "_LAKE", None)
    monkeypatch.setattr(R, "shared_revision", lookup.SharedRevision())
    monkeypatch.setattr(R, "verdicts", lookup.Verdicts())
    monkeypatch.setattr(R, "lookup_stats", lookup.LookupStats())
    cells = lookup.Cells()
    monkeypatch.setattr(R, "cells", cells)
    gate = serving.Gate(annotate=R.row_annotations, ready=cells.refresh,
                        on_serving=R._serving_started)
    monkeypatch.setattr(R, "gate", gate)
    yield gate
    if lake_mod._LAKE is not None:
        lake_mod._LAKE.close()
        lake_mod._LAKE = None


@pytest.fixture
def bench_lake(tmp_path, monkeypatch, bench_gate):
    """The service's lake: a fresh local one under tmp_path/lake, opened
    from the environment as in production (the lease and the backups off —
    tests reach no Supabase and no bucket)."""
    monkeypatch.setenv("BENCH_LAKE_CATALOG", str(tmp_path / "lake" / "catalog.ducklake"))
    monkeypatch.setenv("BENCH_LAKE_DATA", str(tmp_path / "lake" / "data"))
    monkeypatch.delenv("BENCH_LAKE_CACHE", raising=False)
    monkeypatch.setenv("BENCH_LAKE_LEASE", "off")
    monkeypatch.setenv("BENCH_LAKE_BACKUP", "off")
    yield tmp_path


@pytest.fixture
def bench_app():
    """The Kernel Bench service's app, its lake started by the first request."""
    from backend.service.bench import server
    return server.create_app(start_lake=False)


@pytest.fixture
def bench_client(bench_app):
    """A loopback caller of the service carrying the shared secret (what the
    HA API is); identity headers per request."""
    return TestClient(bench_app, client=("127.0.0.1", 50000),
                      headers={"X-HA-Internal": BENCH_SECRET})


@pytest.fixture
def bench_proxied(monkeypatch, bench_app):
    """The HA API's `/api/bench/*` proxy routed to the service in process."""
    import httpx
    from backend.service import bench_proxy
    monkeypatch.setenv("HUGGINGARCH_BENCH_URL", "http://bench.internal")
    monkeypatch.setattr(bench_proxy, "_clients", type(bench_proxy._clients)())
    monkeypatch.setattr(bench_proxy, "transport", httpx.ASGITransport(app=bench_app))


def as_user(user: str, admin: bool = False) -> dict:
    """The identity headers the HA API forwards for `user`."""
    return {"X-HA-User": user, "X-HA-Admin": "1" if admin else "0"}


@pytest.fixture
def client():
    """FastAPI TestClient. Built lazily after env is patched so the app's
    /site-config.js + middleware see the test env."""
    from backend.api import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# JWT minting helpers
# ---------------------------------------------------------------------------

def make_user_jwt(
    sub: str = "user-uuid-1",
    email: str = "user@example.com",
    *,
    secret: str = TEST_JWT_SECRET,
    audience: str = "authenticated",
    expires_in: int = 3600,
    extra: dict | None = None,
) -> str:
    """Mint a HS256 token whose claims pass `verify_supabase_jwt`."""
    now = int(time.time())
    claims = {
        "sub": sub,
        "email": email,
        "aud": audience,
        "iat": now,
        "exp": now + expires_in,
    }
    if extra:
        claims.update(extra)
    return jwt.encode(claims, secret, algorithm="HS256")


def make_expired_jwt(sub: str = "user-uuid-1") -> str:
    """Pre-aged token (exp 60s ago) for testing the expiry path."""
    return make_user_jwt(sub=sub, expires_in=-60)


@pytest.fixture
def user_token():
    return make_user_jwt(sub="user-uuid-1", email="alice@example.com")


@pytest.fixture
def admin_token():
    return make_user_jwt(sub="admin-uuid-1", email="root@example.com")


# ---------------------------------------------------------------------------
# Asymmetric (ES256) keypair fixture — for tests covering the JWKS path
# ---------------------------------------------------------------------------

@pytest.fixture
def ec_keypair():
    """Generate a P-256 EC keypair. Returns (private_pem, public_jwk).
    The private PEM is used to sign test tokens; the public JWK is what
    Supabase would expose via /.well-known/jwks.json (we plant it via
    httpx_mock in tests that exercise the asymmetric path).
    """
    priv = ec.generate_private_key(ec.SECP256R1())
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub = priv.public_key()
    pub_numbers = pub.public_numbers()

    def _b64u(n: int) -> str:
        import base64
        b = n.to_bytes(32, "big")
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    jwk = {
        "kty": "EC",
        "crv": "P-256",
        "alg": "ES256",
        "use": "sig",
        "kid": "test-kid-1",
        "x": _b64u(pub_numbers.x),
        "y": _b64u(pub_numbers.y),
    }
    return priv_pem, jwk


def sign_es256(claims: dict, priv_pem: bytes, kid: str = "test-kid-1") -> str:
    """Mint an ES256-signed token whose `kid` matches the JWK fixture."""
    return jwt.encode(claims, priv_pem, algorithm="ES256", headers={"kid": kid})


# ---------------------------------------------------------------------------
# httpx_mock helpers — register a few standard mocks
# ---------------------------------------------------------------------------

def mock_admins_yes(httpx_mock, user_id: str):
    """Plant a mocked /rest/v1/admins?user_id=eq.<id> response saying the
    user IS an admin (one row returned)."""
    httpx_mock.add_response(
        method="GET",
        url=f"{TEST_SUPABASE_URL}/rest/v1/admins?user_id=eq.{user_id}&select=user_id",
        json=[{"user_id": user_id}],
        is_reusable=True,
    )


def mock_admins_no(httpx_mock, user_id: str):
    """Plant a mocked admin lookup that says the user is NOT admin (empty)."""
    httpx_mock.add_response(
        method="GET",
        url=f"{TEST_SUPABASE_URL}/rest/v1/admins?user_id=eq.{user_id}&select=user_id",
        json=[],
        is_reusable=True,
    )
