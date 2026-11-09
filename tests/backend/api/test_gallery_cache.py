"""Persistent per-model render cache for gallery facts.

The cache key is the model's `result_version`. These tests pin that an
unchanged model is served without re-running the facts walk. The walk is
stubbed — we're testing the caching layer, not the analyzer; which edits
move which model's `result_version` is `tests/backend/test_fingerprint.py`.

The cache machinery itself (LRU/disk backends, single-flight) lives in
`backend.measure.cache` and is unit-tested in
`tests/backend/measure/test_cache.py`; here we test how the gallery facts
route wires it together.
"""
from __future__ import annotations

import asyncio
import importlib

import backend.api.gallery_overview as g
from backend.measure.cache import DiskBackend, KeyedCache

_MID = "test/Model-A"


def test_the_entry_is_keyed_on_the_models_result_version(monkeypatch):
    """`_resolve_one` hands the key the resolved spec's `result_version`."""
    resolved = object()
    monkeypatch.setattr(g, "fetch_config", lambda m: {"model_type": "x"})
    monkeypatch.setattr(g, "get_default_loader", type(
        "L", (), {"match": staticmethod(lambda c, model_id: resolved)}))
    monkeypatch.setattr(g, "result_version",
                        lambda r: "rv" if r is resolved else None)
    assert g._resolve_one(_MID)[2] == "rv"


def test_the_gallery_disk_budget_is_one_knob_split_by_declared_shares(monkeypatch):
    """The operator sizes the directory, not each tier: the knob lands on
    the tiers as the declared shares and they sum to the whole."""
    monkeypatch.setenv("HA_GALLERY_CACHE_BYTES", "2000")
    reloaded = importlib.reload(g)
    try:
        facts, frontier, views = (reloaded._FACTS_DISK_BYTES,
                                  reloaded._FRONTIER_DISK_BYTES,
                                  reloaded._VIEWS_DISK_BYTES)
        assert facts + frontier + views == 2000
        assert (facts, frontier, views) == (2000 // 4, 2000 * 13 // 20, 2000 // 10)
        assert reloaded._GALLERY._backend._cold._max_bytes == facts
    finally:
        monkeypatch.delenv("HA_GALLERY_CACHE_BYTES")
        importlib.reload(g)


def _stub_facts(monkeypatch, tmp_path, ids):
    """Hermetic facts list: disk cache, a tiny snapshot list, and a fake
    `_model_facts_into` that counts its calls."""
    monkeypatch.setattr(
        g, "_GALLERY", KeyedCache("gallery", DiskBackend(tmp_path), single_flight=True)
    )
    monkeypatch.setattr(g, "iter_snapshot_ids", lambda: list(ids))
    monkeypatch.setattr(g, "_resolve_one", lambda m: (object(), object(), "versionA"))
    monkeypatch.setattr(g, "load_release_date", lambda m: "2026-01-01")
    n = {"c": 0}

    def fake_facts(entry, model_id, config, resolved):
        n["c"] += 1
        entry["total_params"] = 1000 + len(model_id)
        entry["release_date"] = "2026-01-01"

    monkeypatch.setattr(g, "_model_facts_into", fake_facts)
    return n


def test_model_facts_are_served_without_a_sweep_and_cached(monkeypatch, tmp_path):
    """The page's GPU-independent half (timeline, filters, model list, KV)
    comes from `build_model_facts`: one facts walk per model, no sweep, and
    a second build reads every model from the cache."""
    n = _stub_facts(monkeypatch, tmp_path, ["m1", "m22"])
    monkeypatch.setattr(g, "sweep_points",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("swept")))
    first = asyncio.run(g.build_model_facts())
    assert n["c"] == 2
    assert [m["total_params"] for m in first["models"]] == [1002, 1003]
    assert first["errors_count"] == 0 and first["kv_title"] == g.KV_TITLE
    second = asyncio.run(g.build_model_facts())
    assert n["c"] == 2
    assert second == first


def test_the_models_route_serves_the_facts_list(monkeypatch):
    """`GET /gallery/models` answers with `build_model_facts`' payload."""
    from fastapi.testclient import TestClient

    import backend.api  # noqa: F401  # registers routes
    from backend.api._app import app
    from backend.api.routes import gallery as route

    async def fake():
        return {"kv_title": g.KV_TITLE, "models": [{"model_id": "m"}],
                "errors_count": 0}

    monkeypatch.setattr(route, "build_model_facts", fake)
    resp = TestClient(app).get("/gallery/models")
    assert resp.status_code == 200
    assert resp.json()["models"] == [{"model_id": "m"}]


def test_the_overview_route_is_gone():
    """The per-preset SOL+measured page is retired; the homepage reads
    `/gallery/frontier`."""
    from fastapi.testclient import TestClient

    import backend.api  # noqa: F401
    from backend.api._app import app

    resp = TestClient(app).get("/gallery/overview")
    assert resp.status_code == 404
