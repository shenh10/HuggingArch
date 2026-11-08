"""The HA API's `/api/bench/*` proxy to the Kernel Bench service
(docs/measured_calibration.md §5.1): the API authenticates, strips every
inbound `X-HA-*` header and sets the three it alone sets; the service's
statuses pass through; the service unreachable, timing out or failing is one
503 while every other route stays up; the upstream request is cancelled when
the client goes away; `/health` carries the service's health.

The service runs in process (`httpx.ASGITransport`), or a recording stand-in
transport where the test needs to see what was sent."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from starlette.requests import Request

import tests.corpus_row as _corpus
from backend.service import bench_proxy

from .conftest import BENCH_SECRET, make_user_jwt, mock_admins_no, mock_admins_yes

ALICE = "11111111-1111-4111-8111-111111111111"
BOB = "22222222-2222-4222-8222-222222222222"
ADMIN = "33333333-3333-4333-8333-333333333333"


def _jwt(sub: str) -> dict:
    return {"Authorization": f"Bearer {make_user_jwt(sub=sub, email=f'{sub[:4]}@example.com')}"}


def _csv(n: int = 1) -> str:
    rows = []
    for i in range(n):
        shape = json.loads(_corpus.row()["shape_json"])
        shape["s_ctx"] = 2048 * (i + 1)
        rows.append(_corpus.row(shape_json=json.dumps(shape, sort_keys=True, separators=(",", ":"))))
    return _corpus.to_csv(rows)


@pytest.fixture
def route_to(monkeypatch):
    """Point the proxy at `transport` (the service in process, or a stand-in)."""
    monkeypatch.setenv("HUGGINGARCH_BENCH_URL", "http://bench.internal")
    monkeypatch.setattr(bench_proxy, "_clients", type(bench_proxy._clients)())

    def to(transport):
        monkeypatch.setattr(bench_proxy, "transport", transport)
    return to


@pytest.fixture
def service(route_to, bench_app, bench_lake):
    route_to(httpx.ASGITransport(app=bench_app))


@pytest.fixture
def recorded(route_to):
    """A stand-in service that records each request and answers 200 `{}`."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})
    route_to(httpx.MockTransport(handler))
    return seen


def test_the_caller_is_the_session_end_to_end(client, service, httpx_mock):
    mock_admins_no(httpx_mock, ALICE)
    mock_admins_no(httpx_mock, BOB)
    r = client.post("/api/bench/upload", headers=_jwt(ALICE), json={"csv_text": _csv(2)})
    assert r.status_code == 200, r.text
    assert r.json()["row_count"] == 2
    (s,) = client.get("/api/bench/list", headers=_jwt(ALICE)).json()["uploads"]
    assert s["row_count"] == 2
    assert client.get("/api/bench/list", headers=_jwt(BOB)).json() == {"uploads": []}
    # Signed out: the shared corpus only, and 401 where a caller is needed.
    assert client.get("/api/bench/list").status_code == 401
    assert client.get("/api/bench/browse?view=rows").status_code == 200


def test_forged_internal_headers_are_stripped(client, recorded, httpx_mock):
    forged = {"X-HA-User": ADMIN, "X-HA-Admin": "1", "X-HA-Internal": "guess",
              "X-HA-Anything": "x", "Cookie": "a=b"}
    client.get("/api/bench/list?x=1", headers=forged)
    mock_admins_no(httpx_mock, ALICE)
    client.get("/api/bench/list", headers={**forged, **_jwt(ALICE)})
    anon, alice = recorded
    for sent in (anon, alice):
        names = {k.lower() for k in sent.headers}
        assert not {"x-ha-anything", "cookie", "authorization"} & names
        assert sent.headers["x-ha-internal"] == BENCH_SECRET
        assert sent.headers.get_list("x-ha-user") == [sent.headers["x-ha-user"]]
    assert (anon.headers["x-ha-user"], anon.headers["x-ha-admin"]) == ("", "0")
    assert (alice.headers["x-ha-user"], alice.headers["x-ha-admin"]) == (ALICE, "0")
    assert anon.url.path == "/api/bench/list" and anon.url.query == b"x=1"


def test_the_admin_bit_is_the_apis_lookup(client, recorded, httpx_mock):
    mock_admins_yes(httpx_mock, ADMIN)
    client.get("/api/bench/nominations", headers=_jwt(ADMIN))
    assert recorded[0].headers["x-ha-admin"] == "1"


def test_bad_credentials_never_reach_the_service(client, recorded):
    r = client.get("/api/bench/list", headers={"Authorization": "Bearer not-a-jwt"})
    assert r.status_code == 401 and recorded == []


def test_a_token_is_accepted_for_the_measurement_pipeline_only(
        client, recorded, monkeypatch):
    from backend.service import personal_tokens
    monkeypatch.setattr(personal_tokens, "looks_like_pat", lambda t: t.startswith("hapat_"))
    monkeypatch.setattr(personal_tokens, "verify", lambda t: ALICE if t == "hapat_ok" else None)
    monkeypatch.setattr("backend.service.auth.is_admin", lambda uid: False)
    pat = {"Authorization": "Bearer hapat_ok"}
    assert client.post("/api/bench/upload", headers=pat, json={"csv_text": "x"}).status_code == 200
    assert recorded[-1].headers["x-ha-user"] == ALICE
    assert client.post("/api/bench/run/run-1/contribute", headers=pat).status_code == 200
    assert client.post("/api/bench/promote", headers=pat, json={"sets": []}).status_code == 200
    assert client.post("/api/bench/upload", headers={"Authorization": "Bearer hapat_no"},
                       json={"csv_text": "x"}).status_code == 401
    # Other Kernel Bench reads and everywhere else: a long-lived pipeline
    # token is not a session.
    assert client.get("/api/bench/list", headers=pat).status_code == 401
    assert client.get("/api/account/tokens", headers=pat).status_code == 401


def test_the_services_statuses_pass_through(client, service, httpx_mock):
    mock_admins_no(httpx_mock, ALICE)
    h = _jwt(ALICE)
    r = client.post("/api/bench/upload", headers=h, json={"csv_text": "not,a,csv"})
    assert r.status_code == 422 and "detail" in r.json()
    r = client.post("/api/bench/upload", headers=h, json={"csv_text": _csv(), "make_public": True})
    assert r.status_code == 403 and "requires admin" in r.json()["detail"]
    assert client.post("/api/bench/run/nope/contribute", headers=h).status_code == 404
    assert client.get("/api/bench/rows", headers=h).status_code == 400


def test_the_lakes_503_passes_through_with_its_reason(client, service, bench_gate, httpx_mock):
    mock_admins_no(httpx_mock, ALICE)
    client.get("/api/bench/list", headers=_jwt(ALICE))
    bench_gate.phase, bench_gate.reason = "unavailable", "not the authoritative lake"
    r = client.get("/api/bench/list", headers=_jwt(ALICE))
    assert r.status_code == 503 and "not the authoritative lake" in r.json()["detail"]


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), 500, 502,
])
def test_the_service_down_is_503_and_the_rest_stays_up(client, route_to, failure):
    def handler(request):
        if isinstance(failure, Exception):
            raise failure
        return httpx.Response(failure, text="Traceback … secret")
    route_to(httpx.MockTransport(handler))
    r = client.get("/api/bench/browse")
    assert r.status_code == 503
    assert r.json() == {"detail": "Kernel Bench is unavailable"}
    assert r.headers["retry-after"]
    assert client.get("/inference/gpus").status_code == 200
    detail = client.get("/health").json()["detail"]["kernel_bench"]
    assert detail["state"] == "unreachable"


def test_no_service_configured_is_503(client, monkeypatch):
    monkeypatch.delenv("HUGGINGARCH_BENCH_URL", raising=False)
    assert client.get("/api/bench/browse").status_code == 503
    assert client.get("/health").json()["detail"]["kernel_bench"] == {"state": "not_configured"}


def test_the_internal_endpoints_are_not_routable(client, recorded):
    for method, path in (("POST", "/internal/bench/lookup"), ("GET", "/internal/bench/version"),
                         ("GET", "/api/bench/../internal/bench/version")):
        assert client.request(method, path, json={}).status_code in (404, 405)
    assert recorded == []


def test_health_carries_the_services(client, service):
    detail = client.get("/health").json()["detail"]["kernel_bench"]
    assert detail["state"] == "up" and detail["status"] == "ok" and "lake" in detail


def test_an_unmigrated_lake_is_degraded_at_the_api_level(client, route_to):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={
            "status": "ok",
            "lake": {"state": "degraded",
                     "reason": "the lake's data rules are legacy; migrating",
                     "data_rules": {"state": "behind"}},
        })

    route_to(httpx.MockTransport(handler))
    detail = client.get("/health").json()["detail"]["kernel_bench"]
    assert detail["state"] == "degraded"
    assert detail["lake"]["data_rules"]["state"] == "behind"
def test_the_response_headers_are_the_services_own(client, route_to):
    def handler(request):
        return httpx.Response(200, content=b"a,b\n1,2\n", headers={
            "content-type": "text/csv", "content-disposition": "attachment; filename=x.csv",
            "x-ha-user": "leak", "set-cookie": "s=1"})
    route_to(httpx.MockTransport(handler))
    r = client.get("/api/bench/rows.csv")
    assert r.content == b"a,b\n1,2\n"
    assert r.headers["content-type"] == "text/csv"     # the body's type crosses; nothing else
    assert not {"content-disposition", "x-ha-user", "set-cookie"} & set(r.headers)


def test_a_client_that_goes_away_cancels_the_upstream_request(route_to, monkeypatch):
    """A slow upstream (an upload being stored) and a client that
    disconnects: the proxy cancels its request instead of waiting on it."""
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def handler(request):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return httpx.Response(200)
    route_to(httpx.MockTransport(handler))
    monkeypatch.setattr(bench_proxy, "identity_headers", lambda user: {})
    monkeypatch.setattr(bench_proxy, "_caller", lambda request: None)

    async def run():
        messages = [{"type": "http.request", "body": b"{}", "more_body": False}]

        async def receive():
            if messages:
                return messages.pop(0)
            await started.wait()
            return {"type": "http.disconnect"}
        scope = {"type": "http", "method": "POST", "path": "/api/bench/upload",
                 "raw_path": b"/api/bench/upload", "query_string": b"", "headers": [],
                 "client": ("1.2.3.4", 1), "server": ("x", 80), "scheme": "http"}
        resp = await asyncio.wait_for(bench_proxy.forward(Request(scope, receive)), 10)
        await asyncio.wait_for(cancelled.wait(), 5)
        return resp
    assert asyncio.run(run()).status_code == 499
