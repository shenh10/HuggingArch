"""A checkpoint header keeps every tensor name the file holds, and a
spec-composed name selects exactly the tensors it stands for."""
from __future__ import annotations

import json

from backend.hf.header import SafetensorsHeader

TENSORS = {
    "model.embed_tokens.weight": ("BF16", [16, 8]),
    **{f"model.layers.{i}.mlp.gate_proj.weight": ("BF16", [32, 8])
       for i in range(3)},
    **{f"model.layers.{i}.mlp.experts.{e}.w1.weight": ("F8_E4M3", [4, 8])
       for i in range(3, 6) for e in range(4)},
    "model.layers.5.mlp.experts.9.w1.weight": ("F8_E4M3", [4, 8]),
    **{f"model.layers.{i}.mlps.{j}.down_proj.weight": ("BF16", [8, 8])
       for i in range(2) for j in range(2)},
    "model.layers.0.self_attn.q_proj.weight": ("BF16", [8, 8]),
    "model.layers.1.self_attn.q_proj.weight": ("BF16", [12, 8]),
}


def _header():
    return SafetensorsHeader.from_tensors(TENSORS, revision="abc", total_size=1)


def test_every_name_comes_back_verbatim_through_the_stored_record():
    stored = json.loads(json.dumps(_header().to_json()))
    back = SafetensorsHeader.from_json(stored)
    assert {name: (g.dtype, list(g.shape)) for name, g in back.names()} == {
        name: (dtype, shape) for name, (dtype, shape) in TENSORS.items()}
    assert back.revision == "abc"


def test_an_index_placeholder_takes_every_index_the_file_holds_there():
    sel = _header().select("model.layers.X.mlp.experts.X.w1.weight")
    assert sel.count == 13


def test_a_literal_index_takes_only_its_own():
    header = _header()
    assert header.select("model.layers.X.mlps.0.down_proj.weight").count == 2
    assert header.select("model.layers.1.mlps.1.down_proj.weight").count == 1
    assert not header.select("model.layers.7.mlps.0.down_proj.weight")


def test_an_index_set_as_display_writes_it_takes_each_index_listed():
    header = _header()
    group = header.select("model.layers.X.mlp.gate_proj.weight").parts[0][0]
    sel = header.select(group.display())
    assert group.display() == "model.layers.{0-2}.mlp.gate_proj.weight"
    assert sel.count == 3 and sel.complete
    sel = header.select("model.layers.{3-5}.mlp.experts.{0,9}.w1.weight")
    assert sel.count == 4 and not sel.complete  # only layer 5 holds expert 9
    assert not header.select("model.layers.{0-2,...}.mlp.gate_proj.weight")


def test_a_name_missing_an_index_the_file_writes_selects_nothing():
    header = _header()
    assert not header.select("model.layers.X.mlps.down_proj.weight")
    assert not header.select("model.layers.mlp.gate_proj.weight")


def test_one_name_stored_at_two_shapes_selects_both():
    sel = _header().select("model.layers.X.self_attn.q_proj.weight")
    assert sel.shapes == ((8, 8), (12, 8))
    assert sel.elements == 64 + 96


def test_unclaimed_lists_what_no_name_selects():
    left = _header().unclaimed([
        "model.embed_tokens.weight",
        "model.layers.X.mlp.gate_proj.weight",
        "model.layers.X.mlp.experts.X.w1.weight",
        "model.layers.X.self_attn.q_proj.weight",
        "model.layers.X.mlps.0.down_proj.weight",
    ])
    assert [(g.display(indices), len(indices)) for g, indices in left] == [
        ("model.layers.{0-1}.mlps.{1}.down_proj.weight", 2)]


def test_a_record_with_folded_names_is_refused():
    """A record written before the verbatim header (`tensor_patterns`, names
    folded on write) cannot be read back into the file's names: it is
    refused with the refetch it needs."""
    import pytest
    from backend.hf.metadata import metadata_from_record
    old = {"tensor_patterns": {"model.layers.X.mlp.gate_proj.weight":
                               {"shape": [32, 8], "dtype": "BF16"}},
           "pattern_counts": {"model.layers.X.mlp.gate_proj.weight": 3}}
    with pytest.raises(ValueError, match="no verbatim safetensors header"):
        metadata_from_record(old, source="snapshot org/m")


def test_legacy_wrapper_record_recovers_only_its_declared_delegate():
    import pytest
    from backend.arch.quant_schemes import PackedStorageMismatch
    from backend.arch.snapshots import load_snapshot
    from backend.hf.metadata import metadata_from_record

    snapshot = load_snapshot("moonshotai/Kimi-K3")
    current = snapshot["metadata"]
    legacy = {**current, "quant_scheme": "compressed-tensors"}
    restored = metadata_from_record(
        legacy, source="legacy snapshot", legacy_config=lambda: snapshot["config"])
    expected = metadata_from_record(current, source="current snapshot")
    assert restored["quant_scheme"] == expected["quant_scheme"]
    assert restored["total_params"] == expected["total_params"]
    with pytest.raises(PackedStorageMismatch):
        metadata_from_record(legacy, source="unrelated config", legacy_config=lambda: {})
