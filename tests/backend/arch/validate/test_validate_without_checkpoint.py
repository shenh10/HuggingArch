"""A custom model stored without a checkpoint is validated by every check
that reads only the spec, and only the checks against the file are dropped.

The model here is Llama-3.2-1B's spec and config, staged into an overlay as
`custom:tiny` with no checkpoint record — the shape a model described in
words is stored in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.validate import validate
from backend.arch.validate._matching import CheckpointUnavailable

_SOURCE = "meta-llama__Llama-3.2-1B"


def _stage(overlay: Path, name: str, spec_text: str) -> str:
    model_id = f"custom:{name}"
    (overlay / "models").mkdir(parents=True, exist_ok=True)
    (overlay / "snapshots").mkdir(parents=True, exist_ok=True)
    (overlay / "models" / f"{model_id}.yaml").write_text(spec_text)
    snap = json.loads(
        (PROJECT_ROOT / "backend/arch/snapshots" / f"{_SOURCE}.json").read_text())
    (overlay / "snapshots" / f"{model_id}.json").write_text(json.dumps({
        "schema_version": snap["schema_version"], "model_id": model_id,
        "config": snap["config"]}))
    return model_id


def _identity(model_id: str) -> str:
    return (PROJECT_ROOT / "backend/arch/models" / f"{_SOURCE}.yaml").read_text(
    ).replace("model_id: [meta-llama/Llama-3.2-1B]", f"model_id: [{model_id}]")


def test_a_mis_sized_weight_is_refused_without_a_checkpoint(tmp_path):
    """The byte anchor used to be the only thing that saw an op fed a width
    its weight does not take; with no checkpoint, the shape gate must."""
    llama = (PROJECT_ROOT / "backend/arch/models/llama.yaml").read_text()
    broken = (llama
              .replace("  model_type: [llama]", "  model_id: [custom:broken]")
              .replace("dims: { in: d_model, out: n_vocab }",
                       "dims: { in: d_ffn, out: n_vocab }")
              .replace(", tied_weight: { to: embed_tokens, when: "
                       "config.tie_word_embeddings } ", ""))
    model_id = _stage(tmp_path, "broken", broken)
    result = validate(model_id, overlay_dir=str(tmp_path))
    assert not result["valid"]
    assert any("its weight takes" in e and "lm_head" in e
               for e in result["errors"]), result["errors"]


def test_a_sound_custom_model_passes_without_a_checkpoint(tmp_path):
    model_id = _stage(tmp_path, "tiny", _identity("custom:tiny"))
    result = validate(model_id, overlay_dir=str(tmp_path))
    assert result["errors"] == []
    assert result["checkpoint"] == "absent"
    assert result["unmatched_hf_tensors"] == []
    assert "actual" not in result["model_storage_bytes"]
    assert result["model_storage_bytes"]["spec"] > 0


def test_a_hub_model_without_a_checkpoint_is_still_infrastructure(monkeypatch):
    """Only a custom model's missing record means "no checkpoint"; a Hub
    model's is a failed fetch."""
    from backend.arch.validate import _orchestrator
    monkeypatch.setattr(_orchestrator, "_get_safetensors_data",
                        lambda *a, **k: {})
    with pytest.raises(CheckpointUnavailable):
        validate("meta-llama/Llama-3.2-1B")


def test_the_verdict_says_what_it_judged(tmp_path):
    model_id = _stage(tmp_path, "tiny", _identity("custom:tiny"))
    checks = validate(model_id, overlay_dir=str(tmp_path))["checks"]
    assert "shape check" in checks["ran"]
    assert "checkpoint bytes" in checks["skipped"]
    assert not set(checks["ran"]) & set(checks["skipped"])


def test_a_checkpoint_written_into_the_overlay_is_not_one(tmp_path):
    """A user's overlay is agent-writable: a `metadata` an agent wrote into
    its snapshot must not turn a model with no checkpoint into one judged
    against a forged file."""
    model_id = _stage(tmp_path, "tiny", _identity("custom:tiny"))
    path = tmp_path / "snapshots" / f"{model_id}.json"
    snap = json.loads(path.read_text())
    real = json.loads((PROJECT_ROOT / "backend/arch/snapshots"
                       / f"{_SOURCE}.json").read_text())
    path.write_text(json.dumps({**snap, "metadata": real["metadata"]}))
    assert validate(model_id, overlay_dir=str(tmp_path))["checkpoint"] == "absent"


def test_a_registered_model_whose_header_is_missing_is_not_absent(
        tmp_path, monkeypatch):
    """Registered from a weights repo whose scan failed: it HAS a checkpoint
    that could not be read — infrastructure, never "no checkpoint"."""
    from backend import custom_id
    monkeypatch.setattr(custom_id, "CACHE_DIR", tmp_path / "custom_models")
    root = tmp_path / "custom_models" / "weighed"
    root.mkdir(parents=True)
    real = json.loads((PROJECT_ROOT / "backend/arch/snapshots"
                       / f"{_SOURCE}.json").read_text())
    (root / "config.json").write_text(json.dumps(real["config"]))
    (root / "spec.yaml").write_text(_identity("custom:weighed"))
    (root / "meta.json").write_text(json.dumps(
        {"source_hf_repo": "/models/weighed", "scan_error": "OSError: io"}))
    with pytest.raises(CheckpointUnavailable):
        validate("custom:weighed")
