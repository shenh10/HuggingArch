"""The result caches (`backend.measure.cache`).

Pins the shared machinery every cached result leans on: the LRU / disk
backends, write-behind, single-flight, and the key hash beside it
(`scope.make_key`). Route-level wiring is tested per route (e.g.
`tests/backend/api/test_gallery_cache.py`); the version stamps keys fold are
`backend.fingerprint`'s (`tests/backend/test_fingerprint.py`).
"""
from __future__ import annotations

import json
import threading

import pytest

from backend.measure.cache import (
    DiskBackend,
    KeyedCache,
    MemoryBackend,
    TieredBackend,
    WriteBehindBackend,
    WriteBehindQueue,
    deep_size_of,
    env_bytes,
)
from backend.measure.scope import make_key


def test_memory_backend_lru_evicts_oldest():
    be = MemoryBackend(max_entries=3)
    for k in ("a", "b", "c"):
        be.put(k, k.upper())
    assert be.get("a") == "A"        # touch → most-recently used
    be.put("d", "d_head")                 # over cap → evict LRU, which is now "b"
    assert be.get("b") is None
    assert be.get("a") == "A" and be.get("c") == "C" and be.get("d") == "d_head"


def _blob(n: int) -> dict:
    """A value whose JSON is `n` bytes."""
    value = {"x": ""}
    value["x"] = "a" * (n - len(json.dumps(value)))
    assert len(json.dumps(value)) == n
    return value


def test_memory_backend_byte_budget_evicts_lru_until_the_rest_fits():
    be = MemoryBackend(max_bytes=300)
    for k in ("a", "b", "c"):
        be.put(k, _blob(100))
    assert be.get("a") is not None       # touch → most-recently used
    be.put("d", _blob(150))              # 450 bytes → evict "b", then "c"
    assert be.get("b") is None and be.get("c") is None
    assert be.get("a") is not None and be.get("d") is not None
    # Putting a key again counts its new size, not both.
    be.put("d", _blob(200))
    assert be.get("a") is not None and be.get("d") == _blob(200)
    be.put("e", _blob(100))              # 400 → evict "a" only
    assert be.get("a") is None and be.get("e") is not None


def test_memory_backend_keeps_no_value_larger_than_its_budget(tmp_path):
    be = MemoryBackend(max_bytes=300)
    be.put("small", _blob(100))
    be.put("huge", _blob(301))
    assert be.get("huge") is None
    assert be.get("small") is not None   # nothing else was evicted for it
    # A tier behind memory still holds it, and a hit there is not promoted.
    tiered = TieredBackend(MemoryBackend(max_bytes=300), DiskBackend(tmp_path))
    tiered.put("huge", _blob(301))
    assert tiered.get("huge") == _blob(301)
    # The same key shrunk to fit is kept, and its old size is gone.
    be.put("huge", _blob(150))
    assert be.get("huge") is not None and be.get("small") is not None


def test_deep_size_of_counts_working_objects_and_memory_backend_uses_it():
    """Byte-bounded memory works for non-JSON working values too: size is
    attributed once per object graph, shared code is not charged, and the
    same LRU budget evicts by resident bytes."""
    shared = {"payload": bytearray(512)}
    twice = [shared, shared]
    once = deep_size_of(shared)
    assert deep_size_of(twice) < once + deep_size_of({"payload": bytearray(512)})
    assert deep_size_of(threading.Lock()) > 0

    one = {"payload": bytearray(500)}
    other = {"payload": bytearray(700)}
    be = MemoryBackend(
        max_entries=16, max_bytes=deep_size_of(one) + deep_size_of(other) - 1,
        size_of=deep_size_of)
    be.put("one", one)
    be.put("other", other)
    assert be.get("one") is None and be.get("other") is not None
    huge = {"payload": bytearray(4_000_000)}
    be.put("huge", huge)
    assert be.get("huge") is None and be.get("other") is not None


def test_memory_backend_reaccounts_a_value_that_grew_after_it_was_stored():
    value = {"payload": bytearray(100)}
    be = MemoryBackend(max_bytes=deep_size_of(value) * 2,
                       size_of=deep_size_of)
    be.put("k", value)
    value["payload"] = bytearray(100_000)
    assert not be.resize("k", value)
    assert be.get("k") is None
    small = {"payload": bytearray(100)}
    be.put("small", small)
    small["other"] = bytearray(50)
    assert be.resize("small", small)
    assert be.get("small") is small
    assert be._bytes == deep_size_of(small)


def test_a_promoted_value_is_sized_by_its_file_not_serialised_again(tmp_path, monkeypatch):
    """A durable hit or a put reaches a byte-bounded hot tier with its
    file's length: the hot tier never serialises the value to size it, so
    an oversized value is turned away at the cost of a stat."""
    from backend.measure import cache as cache_mod
    tiered = TieredBackend(MemoryBackend(max_bytes=300), DiskBackend(tmp_path))
    DiskBackend(tmp_path).put("huge", _blob(1000))
    DiskBackend(tmp_path).put("small", _blob(100))
    assert DiskBackend(tmp_path).nbytes("huge") == 1000
    assert DiskBackend(tmp_path).nbytes("missing") is None
    huge, small, other = _blob(1000), _blob(100), _blob(120)
    dumps = []
    real = cache_mod.json.dumps
    monkeypatch.setattr(cache_mod.json, "dumps",
                        lambda *a, **k: dumps.append(1) or real(*a, **k))
    assert tiered.get("huge") == huge
    assert tiered.get("small") == small
    tiered.put("other", other)
    assert dumps == []
    # The small ones are held hot, the huge one is not.
    hot = tiered._hot
    assert hot.get("small") is not None and hot.get("other") is not None
    assert hot.get("huge") is None


def test_disk_backend_roundtrip_and_prune(tmp_path):
    be = DiskBackend(tmp_path, max_files=10)
    be.put("k", {"v": 1})
    assert be.get("k") == {"v": 1}
    assert be.get("missing") is None
    for i in range(25):
        be.put(f"k{i:03d}", {"i": i})
    assert len(list(tmp_path.glob("*.json"))) == 10  # capped, oldest evicted


def test_disk_backend_survives_bad_dir(tmp_path):
    # A read/write failure must never raise — the endpoint just recomputes.
    be = DiskBackend(tmp_path / "nonexistent" / "nested")
    assert be.get("k") is None
    be.put("k", {"v": 1})            # creates the dir
    assert be.get("k") == {"v": 1}


def _aged(disk, key, value, age_s):
    """Put an entry and give its file an explicit mtime, so a prune's
    oldest-first order is the test's, not the filesystem clock's."""
    disk.put(key, value)
    import os as _os
    p = disk._dir / f"{key}.json"
    st = p.stat()
    _os.utime(p, (st.st_atime - age_s, st.st_mtime - age_s))


def test_disk_backend_byte_budget_evicts_oldest_until_the_rest_fits(tmp_path):
    be = DiskBackend(tmp_path, max_files=100, max_bytes=350)
    for i, k in enumerate(("a", "b", "c")):
        _aged(be, k, _blob(100), age_s=100 - i)   # "a" is the oldest
    be.put("d", _blob(100))                       # 400 bytes → evict "a"
    assert be.get("a") is None
    assert be.nbytes("b") == be.nbytes("c") == be.nbytes("d") == 100


def test_disk_backend_keeps_an_oversized_entry_alone(tmp_path):
    be = DiskBackend(tmp_path, max_files=100, max_bytes=100)
    _aged(be, "small", _blob(50), age_s=10)
    be.put("huge", _blob(1000))
    # The bound caps growth at one oversized entry; it never deletes the
    # entry it just wrote.
    assert be.get("small") is None
    assert be.nbytes("huge") == 1000


def _noise(n: int, seed: int) -> dict:
    """A value whose JSON is `n` bytes of hex — gzip halves it, no more."""
    import random
    rng = random.Random(seed)
    value = {"x": ""}
    value["x"] = "".join(rng.choice("0123456789abcdef")
                         for _ in range(n - len(json.dumps(value))))
    return value


def test_compressed_disk_backend_roundtrip_is_gzip_of_the_json(tmp_path):
    import gzip
    be = DiskBackend(tmp_path, compress=True)
    value = {"points": [[i, i * 0.5, "x"] for i in range(500)]}
    be.put("k", value)
    assert be.get("k") == value
    assert be.get("missing") is None
    (path,) = be.files()
    assert path.name == "k.json.gz" and be.suffix == ".json.gz"
    # The file is the value's JSON, gzip'd — any gzip reader opens it — and
    # holds no name or time: one value, one file.
    assert json.loads(gzip.decompress(path.read_bytes())) == value
    again = DiskBackend(tmp_path / "again", compress=True)
    again.put("k", value)
    assert (tmp_path / "again" / "k.json.gz").read_bytes() == path.read_bytes()
    # A plain backend over the same directory sees none of it.
    assert DiskBackend(tmp_path).get("k") is None
    assert DiskBackend(tmp_path).files() == []


def test_compressed_nbytes_is_the_json_length_not_the_file_length(tmp_path):
    be = DiskBackend(tmp_path, compress=True)
    be.put("big", _blob(100_000))
    assert be.nbytes("big") == 100_000
    assert (tmp_path / "big.json.gz").stat().st_size < 1_000
    assert be.nbytes("missing") is None


def test_compressed_byte_budget_counts_the_bytes_on_disk(tmp_path):
    be = DiskBackend(tmp_path, max_files=100, max_bytes=40_000, compress=True)
    for i in range(12):
        be.put(f"k{i:02d}", _noise(10_000, i))
    on_disk = [p.stat().st_size for p in be.files()]
    # 12 × 10 KB of JSON is three times the budget; at about half that each
    # on disk, more than the 4 entries a JSON count would allow are kept,
    # and the files fit the budget.
    assert sum(on_disk) <= 40_000
    assert 4 < len(on_disk) < 12
    assert be.get("k11") == _noise(10_000, 11)   # newest kept
    assert be.get("k00") is None                 # oldest evicted


def test_a_corrupt_compressed_entry_reads_as_a_miss(tmp_path):
    be = DiskBackend(tmp_path, compress=True)
    be.put("k", {"v": 1})
    path = tmp_path / "k.json.gz"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])     # truncated
    assert be.get("k") is None
    path.write_bytes(b"not gzip at all")
    assert be.get("k") is None


def test_a_file_of_the_other_format_is_never_read_and_is_pruned(tmp_path):
    """No backend reads both formats: an entry written before its tier
    changed format is a miss, and the next write's prune removes it."""
    DiskBackend(tmp_path).put("old", {"v": 1})
    be = DiskBackend(tmp_path, compress=True)
    assert be.get("old") is None
    be.put("new", {"v": 2})
    assert not (tmp_path / "old.json").exists()
    assert [p.name for p in be.files()] == ["new.json.gz"]
    # And the other way round.
    plain = DiskBackend(tmp_path)
    plain.put("plain", {"v": 3})
    assert not (tmp_path / "new.json.gz").exists()


def test_env_bytes_parses_operator_sizes_and_refuses_a_mistyped_budget(monkeypatch):
    monkeypatch.delenv("HA_TEST_BYTES", raising=False)
    assert env_bytes("HA_TEST_BYTES", 5) == 5
    for raw, want in (("300", 300), ("512KB", 512 << 10), ("1.5MiB", int(1.5 * (1 << 20))),
                      ("2G", 2 << 30)):
        monkeypatch.setenv("HA_TEST_BYTES", raw)
        assert env_bytes("HA_TEST_BYTES", 5) == want, raw
    monkeypatch.setenv("HA_TEST_BYTES", "512 MB")
    assert env_bytes("HA_TEST_BYTES", 5) == 512 << 20   # a space is fine
    monkeypatch.setenv("HA_TEST_BYTES", "five")
    with pytest.raises(ValueError, match="HA_TEST_BYTES"):
        env_bytes("HA_TEST_BYTES", 5)


def test_tiered_memory_miss_reads_disk_written_by_another_process(tmp_path):
    """The server's memory tier must not hide an entry the separate warm
    process wrote to the shared directory after the server first looked."""
    server = TieredBackend(MemoryBackend(), DiskBackend(tmp_path))
    assert server.get("k") is None
    DiskBackend(tmp_path).put("k", {"v": "warmed"})   # the warm process
    assert server.get("k") == {"v": "warmed"}


class _SlowStore:
    def __init__(self):
        self.release = threading.Event()
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, value):
        self.release.wait(5)
        self.store[key] = value

    def clear(self):
        self.store.clear()


def test_write_behind_put_does_not_wait_on_the_store():
    slow = _SlowStore()
    writes = WriteBehindQueue()
    cache = KeyedCache("t", MemoryBackend(), remote=WriteBehindBackend(slow, writes))
    got = cache.get_or_compute("k", lambda: {"v": 1}, remote_ok=True)
    assert got == {"v": 1}
    assert "k" not in slow.store                 # returned before the store wrote
    assert not writes.drain(timeout=0.05)        # still in flight
    slow.release.set()
    assert writes.drain(timeout=5)
    assert slow.store["k"]["value"] == {"v": 1}


def test_write_behind_drops_when_full():
    slow = _SlowStore()
    writes = WriteBehindQueue(max_pending=1)
    be = WriteBehindBackend(slow, writes)
    for i in range(5):                           # worker holds 1, queue holds 1
        be.put(f"k{i}", i)
    slow.release.set()
    assert writes.drain(timeout=5)
    assert 1 <= len(slow.store) <= 2             # the rest were dropped, not queued


def test_get_or_compute_caches_and_gates_with_store_if():
    cache = KeyedCache("t", MemoryBackend())
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"error": "boom"}

    # store_if rejects the error entry → recomputed every time, never cached.
    cache.get_or_compute("k", compute, store_if=lambda v: not v.get("error"))
    cache.get_or_compute("k", compute, store_if=lambda v: not v.get("error"))
    assert calls["n"] == 2

    good = {"ok": 1}
    cache.get_or_compute("k2", lambda: good)
    hit = cache.get_or_compute("k2", lambda: {"ok": 999})  # cached → not recomputed
    assert hit == good


def test_single_flight_computes_once_under_concurrency():
    cache = KeyedCache("t", MemoryBackend(), single_flight=True)
    calls = {"n": 0}
    started = threading.Event()

    def compute():
        calls["n"] += 1
        started.set()
        # Hold the per-key lock long enough that the other threads pile up on it.
        threading.Event().wait(0.05)
        return "value"

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(cache.get_or_compute("k", compute)))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1                 # only one thread actually computed
    assert results == ["value"] * 8        # all got the same value


def test_make_key_is_order_stable_and_distinct():
    assert make_key("a", 1, {"x": 1, "y": 2}) == make_key("a", 1, {"y": 2, "x": 1})
    assert make_key("a", 1) != make_key("a", 2)


def test_single_flight_keeps_no_lock_for_a_key_not_in_flight():
    """A single-flight miss holds a lock for its key while it computes; once
    no caller holds or waits on it the lock goes, so the table does not grow
    with the keys ever asked."""
    from backend.measure.cache import KeyedCache, MemoryBackend

    cache = KeyedCache("t", MemoryBackend(max_entries=4), single_flight=True)
    for i in range(1000):
        assert cache.get_or_compute(f"k{i}", lambda i=i: i) == i
    assert cache._key_locks == {}
