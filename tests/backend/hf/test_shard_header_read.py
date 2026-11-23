"""A sharded checkpoint's headers are read side by side.

Every shard's tensors are in the header whichever read came back first, and
the read reports how many shards are done, so the work waiting on it shows
movement. No network: the Hub client is monkeypatched.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from backend.hf import metadata

_SHARDS = [f"model-{i:05d}-of-00004.safetensors" for i in range(4)]


@pytest.fixture(autouse=True)
def _hub(monkeypatch, tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {
        f"layers.{i}.w": shard for i, shard in enumerate(_SHARDS)}}))

    class Hub:
        def __call__(self):  # stands in for the `HfApi` class
            return self

        def model_info(self, model_id, **_kw):
            names = ["model.safetensors.index.json", *_SHARDS]
            return SimpleNamespace(
                siblings=[SimpleNamespace(rfilename=n) for n in names],
                sha="abc123")

        def parse_safetensors_file_metadata(self, repo, shard, **_kw):
            i = _SHARDS.index(shard)
            # The first shard comes back last.
            time.sleep(0.05 * (len(_SHARDS) - i))
            t = SimpleNamespace(dtype="BF16", shape=[i + 1, 4],
                                data_offsets=[0, 8 * (i + 1)])
            return SimpleNamespace(tensors={f"layers.{i}.w": t})

    monkeypatch.setattr("huggingface_hub.HfApi", Hub())
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda **_kw: str(index))


def test_every_shard_is_read_whatever_order_reads_finish():
    header = metadata.read_header_hf("org/m")
    assert header.total_size == sum(8 * (i + 1) for i in range(len(_SHARDS)))


def test_a_header_read_reports_each_shard_done(monkeypatch):
    ticks = []
    monkeypatch.setattr(metadata, "_progress_hook",
                        lambda done, total: ticks.append((done, total)))
    metadata.read_header_hf("org/m")
    assert ticks == [(i, len(_SHARDS)) for i in range(len(_SHARDS) + 1)]
