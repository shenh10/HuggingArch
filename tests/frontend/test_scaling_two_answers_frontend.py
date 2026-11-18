"""The Scaling panel's two answers per sweep (docs/auto_sweep.md §7.5).

The panel asks for a sweep's SOL answer first (`sol_only`) and draws it; it
asks for the measured answer of the same body only while Measured is ticked,
and a measured answer that fails leaves the SOL curves on screen with a note
beside the checkbox — never the whole chart's "sweep load failed".

Same harness as `test_pages_smoke.py` (built `web/dist`, canned payloads).
"""
from __future__ import annotations

import json

from tests.frontend.test_pages_smoke import Session, browser_base  # noqa: F401

_PATH = "/inference/estimate-parallel-sweep"


def _blob() -> dict:
    point = {"x": 2, "context": 4096, "feasible": True}
    blob = {"moe_enabled": True, "series": [{"gpu_sku": "B200", "points": [point]}]}
    return {"by_mode": {m: blob for m in ("dp_ep", "cp_ep", "tp_ep", "tp_tp")}}


def test_sol_first_and_a_failed_measured_answer_keeps_the_curves(browser_base):  # noqa: F811
    browser, base = browser_base
    s = Session(browser, base, overrides={_PATH: _blob()})
    bodies: list = []

    def answer(route):
        body = json.loads(route.request.post_data or "{}")
        bodies.append(body)
        if body.get("sol_only"):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(_blob()))
        else:
            route.fulfill(status=500, content_type="application/json",
                          body=json.dumps({"detail": "lookup broke"}))

    # Registered after the session's catch-all, so it answers this path.
    s.page.route(f"**{_PATH}", answer)
    s.goto("/inference.html")
    s.page.wait_for_selector("#infer-throughput canvas", timeout=15000)
    assert bodies and all(b.get("sol_only") is True for b in bodies), bodies
    s.page.wait_for_timeout(500)
    assert all(b.get("sol_only") is True for b in bodies), "measured asked while unticked"
    batch = [r["body"] for r in s.requests.get("/inference/estimate-sweep", [])]
    assert all(b.get("sol_only") is True for b in batch), batch

    s.page.check("#infer-throughput-measured")
    assert s.wait_for(lambda: any(not b.get("sol_only") for b in bodies))
    note = s.page.wait_for_selector("#infer-scaling [data-measured-note]", timeout=5000)
    assert s.wait_for(lambda: "lookup broke" in (note.inner_text() or ""))
    assert "measured unavailable" in note.inner_text()
    # The SOL curves stay; the panel does not fail.
    assert s.page.query_selector("#infer-throughput canvas") is not None
    assert "sweep load failed" not in s.page.inner_text("#infer-scaling")
