"""`/precision/*`: the vocabulary, the canonical form, and what a profile
changes on one model."""
from __future__ import annotations

from fastapi.testclient import TestClient


def _client():
    from backend.api import app
    return TestClient(app)


def test_the_presets_are_listed():
    ids = [p["id"] for p in _client().get("/precision/presets").json()["presets"]]
    assert ids == ["as_shipped", "kv_fp8", "kv_nvfp4", "w_fp8_kv_fp8",
                   "w_nvfp4_w4a4_kv_fp8", "w_int8_w8a8_ffn"]


def test_canonicalize_returns_repairable_errors_not_a_failure():
    body = {"weights": [
        {"select": {"roles": ["ffn_moe"]},
         "format": {"scheme": "nvfp4", "activation": "w4a4"}},
        {"select": {"stacks": ["aux:mtp"]}, "format": {"dtype": "BF16"}}]}
    r = _client().post("/precision/canonicalize", json=body)
    assert r.status_code == 200
    assert r.json()["canonical"] is None
    assert r.json()["errors"][0]["intersection"] == {
        "roles": ["ffn_moe", "ffn_moe_latent", "ffn_moe_shared"],
        "stacks": ["aux:mtp"]}


def test_effective_marks_only_what_changed():
    r = _client().post("/precision/effective", json={
        "model_id": "meta-llama/Llama-3.2-1B",
        "precision": {"preset": "kv_fp8"}})
    assert r.status_code == 200, r.text
    changed = {(row["category"], row["selector"])
               for row in r.json()["effective"] if row["changed"]}
    assert changed == {("cache", "kv")}


