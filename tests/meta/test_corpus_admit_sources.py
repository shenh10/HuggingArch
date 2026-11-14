"""Admission honors a snapshot's explicitly recorded serving-source identity."""
import json
from types import SimpleNamespace

import pytest

from tools import corpus_admit as admit
from tools.corpus_pins.corpus import CorpusModel
from tools.corpus_pins.decisions.framework_split import MANIFEST, FrameworkSplit, Oracle


def _forbid_other_frameworks(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("explicit SGLang source must not probe vLLM or stage reference code")
    monkeypatch.setattr(admit, "stage_vllm", forbidden)
    monkeypatch.setattr(admit, "stage_reference", forbidden)


def test_recorded_sglang_stages_offline_and_settles_three_root_weights(tmp_path, monkeypatch):
    _forbid_other_frameworks(monkeypatch)
    model = CorpusModel("moonshotai__Kimi-K3.1-Pro")
    recorded = model.payload["forward_source"]
    (root,) = admit.stage_sources([model.model_id, "--cache-dir", str(tmp_path)])
    manifest = json.loads((root / MANIFEST).read_text())
    assert manifest["kind"] == "sglang"
    assert manifest["revision"] == recorded["revision"]
    expected = {f["path"]: f["text"] for f in recorded["files"]
                if f["path"].endswith(".py")}
    actual = {p.relative_to(root).as_posix(): p.read_text()
              for p in root.rglob("*.py")}
    assert actual == expected
    assert set(manifest["read_from"]) == set(expected)
    assert not list(root.rglob("*.cuh"))
    assert any(f["path"].endswith(".cuh") for f in recorded["files"])
    oracle = Oracle([FrameworkSplit.read(root)])
    for node, placement in {
        "language_model.model.embed_tokens": "split",
        "language_model.lm_head": "split",
        "language_model.model.norm": "replicated",
    }.items():
        answer = oracle.answer(node)
        assert answer.placement == placement
        assert answer.decided_by == "sglang"
        assert any(answer.citation.startswith(path + ":") for path in expected)


def _fake_model(tmp_path, source):
    path = tmp_path / "snapshot.json"
    payload = {"forward_source": source}
    path.write_text(json.dumps(payload))
    return SimpleNamespace(path=path, payload=payload, stem="other__Architecture",
                           model_id="other/Architecture", config={"architectures": ["Other"]})


def test_source_kind_not_model_name_selects_offline_sglang(tmp_path, monkeypatch):
    _forbid_other_frameworks(monkeypatch)
    path = "sglang/python/sglang/srt/models/other.py"
    model = _fake_model(tmp_path, {
        "source": "sglang", "revision": "abc123", "imported": [path],
        "files": [{"path": path, "text": "class Other: pass\n"}],
    })
    monkeypatch.setattr(admit, "CorpusModel", lambda _stem: model)
    (root,) = admit.stage_sources([model.model_id, "--cache-dir", str(tmp_path / "cache")])
    manifest = json.loads((root / MANIFEST).read_text())
    assert manifest["revision"] == "abc123"
    assert manifest["kind"] == "sglang"
    assert manifest["imported"] == [path]
    assert (root / path).read_text() == "class Other: pass\n"


def test_missing_sglang_revision_refuses_without_network_fallback(tmp_path, monkeypatch):
    _forbid_other_frameworks(monkeypatch)
    model = _fake_model(tmp_path, {"source": "sglang", "files": [
        {"path": "src/model.py", "text": "class Other: pass\n"}]})
    monkeypatch.setattr(admit, "CorpusModel", lambda _stem: model)
    with pytest.raises(ValueError, match="needs its revision"):
        admit.stage_sources([model.model_id, "--cache-dir", str(tmp_path / "cache")])


def test_other_sources_keep_the_existing_vllm_reference_fallback(tmp_path, monkeypatch):
    model = _fake_model(tmp_path, {"source": "hf_modeling_files", "files": []})
    monkeypatch.setattr(admit, "CorpusModel", lambda _stem: model)
    calls = []
    def stage(kind):
        def run(candidate, cache):
            assert candidate is model
            calls.append(kind)
            return cache / kind
        return run
    monkeypatch.setattr(admit, "stage_vllm", stage("vllm"))
    monkeypatch.setattr(admit, "stage_reference", stage("reference"))
    roots = admit.stage_sources([model.model_id, "--cache-dir", str(tmp_path / "cache")])
    assert calls == ["vllm", "reference"]
    assert roots == [tmp_path / "cache" / kind for kind in calls]
