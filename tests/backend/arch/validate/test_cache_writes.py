"""A cache write the forward quantises first wants a cache slot stored at
that format.

The cache has no checkpoint bytes, so a spec that reads a buffer's
`register_buffer` dtype instead of its write passes every byte gate:
DeepSeek-V4.1-Flash billed its FP4 compressed KV and FP8 window at BF16
until its paper's per-token figure (890 B) disagreed. The scan reads the
writes and hints, never gates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.cache_formats import CacheFormat
from backend.arch.validate import _cache_writes
from backend.arch.validate._cache_writes import (quantised_writes,
                                                 run_cache_write_check)
from tests.snapshots import shared_loader

SNAPSHOTS = PROJECT_ROOT / "backend" / "arch" / "snapshots"

SOURCE = '''
class Attention(nn.Module):
    def __init__(self):
        self.register_buffer("kv_cache", torch.zeros(8, 128, 512))
        self.register_buffer("k_cache", torch.zeros(8, 128, 128))
        self.register_buffer("v_cache", torch.zeros(8, 128, 128))
        self.register_buffer("o_cache", torch.zeros(8, 128, 128))

    def forward(self, x, kv, k, v, o):
        act_quant(kv[..., :-64], 64, None, None, True)
        self.kv_cache[:, 0] = kv.squeeze(1)
        k_fp4, s = fp4_quant(k, 32)
        k = (k_fp4.float() * s).to(x.dtype)
        self.k_cache[:, 0] = k
        v = dequant(v)
        self.v_cache[:, 0] = v
        self.o_cache[:, 0] = o
        act_quant(o, 64, None, None, True)
'''


def _resolved(stem: str):
    config = json.loads((SNAPSHOTS / f"{stem}.json").read_text())["config"]
    return shared_loader().match(config, model_id=stem.replace("__", "/", 1))


def test_a_write_is_quantised_by_what_came_before_it_in_place_or_assigned():
    writes = quantised_writes([{"path": "m/model.py", "text": SOURCE}])
    assert [(w.buffer, w.width, w.partial) for w in writes] == [
        ("kv_cache", None, True), ("k_cache", 0.5, False)]
    # Followed through the hand-written round trip back to the model dtype;
    # a dequantise is not a quantise; a quantise after the write is not
    # what the write stored.


def test_native_kernel_evidence_does_not_disable_python_cache_write_check():
    python = {"path": "model.py", "text": SOURCE}
    native = {"path": "cache.cuh", "text": "template<typename T> __global__ void store(T* p) {}"}
    assert quantised_writes([native, python]) == quantised_writes([python])


def test_the_spec_is_asked_for_each_format_the_writes_need(monkeypatch):
    stem = "deepseek-ai__DeepSeek-V4.1-Flash"
    resolved = _resolved(stem)
    # Shipped as declared: every write is met.
    assert run_cache_write_check(stem.replace("__", "/"), resolved) == []

    # The buffer-dtype reading: every slot at BF16.
    bf16 = CacheFormat.plain("BF16")
    real = list(_cache_writes._stored_formats(resolved))
    monkeypatch.setattr(_cache_writes, "_stored_formats",
                        lambda _r: iter([(bf16, False, n) for _f, _q, n in real]))
    findings = run_cache_write_check(stem.replace("__", "/"), resolved)
    assert {f.kind for f in findings} == {"cache_write_quantised_unmodelled"}
    assert {(f.extra["buffer"], f.extra["requires"]) for f in findings} == {
        ("k_cache", "width:0.5"), ("compress_kv_cache", "width:0.5"),
        ("k_cache", "quantised"), ("compress_kv_cache", "quantised"),
        ("window_kv_cache", "quantised")}


def test_one_quantised_slot_does_not_answer_for_two_quantised_buffers(
        monkeypatch):
    """DeepSeek-V3.2 fills its indexer key and its MLA latent quantised
    (the latent through a hand-written dequantise back to BF16); a spec
    that stores only the key quantised is one buffer short."""
    stem = "deepseek-ai__DeepSeek-V3.2"
    resolved = _resolved(stem)
    assert run_cache_write_check(stem.replace("__", "/"), resolved) == []
    real = list(_cache_writes._stored_formats(resolved))
    bf16 = CacheFormat.plain("BF16")
    only_key = [(f, q, n) if "indexer" in n else (bf16, False, n)
                for f, q, n in real]
    monkeypatch.setattr(_cache_writes, "_stored_formats",
                        lambda _r: iter(only_key))
    findings = run_cache_write_check(stem.replace("__", "/"), resolved)
    assert {(f.extra["buffer"], f.extra["requires"]) for f in findings} == {
        ("k_cache", "quantised"), ("kv_cache", "quantised")}


def test_a_partially_quantised_row_wants_an_unquantised_tail(monkeypatch):
    stem = "deepseek-ai__DeepSeek-V4-Flash"
    resolved = _resolved(stem)
    assert run_cache_write_check(stem.replace("__", "/"), resolved) == []
    untailed = [(CacheFormat(dtype=f.dtype, scale=f.scale), q, n)
                for f, q, n in _cache_writes._stored_formats(resolved)]
    monkeypatch.setattr(_cache_writes, "_stored_formats",
                        lambda _r: iter(untailed))
    findings = run_cache_write_check(stem.replace("__", "/"), resolved)
    assert {f.extra["requires"] for f in findings} == {"tail"}
