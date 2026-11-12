"""A result cache's local tiers + gated durable remote tier.

Pins that a shareable request promotes/writes the durable remote tier, that a
NON-shareable request (`remote_ok=False`) is confined to the local tiers (the
remote is neither read nor written — a measured/user-calibrated result must not
land in a global table), and that an unconfigured Supabase degrades to
local-only with no error.
"""
from __future__ import annotations

from backend.code_trace.provider import TRACES, Trace, provider
from backend.measure.cache import (REMOTE_TRACE_PREFIX, TRACE_FIELD, DiskBackend,
                                   KeyedCache, MemoryBackend, TieredBackend)


def _trace() -> Trace:
    """What the running code's provider records for a computation."""
    p = provider()
    return p.end(p.begin(nested=True))


def _stored(value):
    """An entry as a cache stores it: the value beside its trace's id."""
    return {TRACE_FIELD: TRACES.put(_trace()), "value": value}


def _value(stored):
    return stored["value"]


class FakeRemote:
    """A durable remote backend stand-in (dict) with get/put/clear."""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, value):
        self.store[key] = value

    def clear(self):
        self.store.clear()


def _cache(tmp_path):
    local = TieredBackend(MemoryBackend(max_entries=8), DiskBackend(tmp_path, max_files=100))
    remote = FakeRemote()
    return KeyedCache("sweep", local, single_flight=True, remote=remote), remote


def test_shareable_remote_hit_is_promoted_to_local(tmp_path):
    cache, remote = _cache(tmp_path)
    remote.store["k"] = _stored({"v": 1})  # only in the durable remote tier
    got = cache.get_or_compute("k", lambda: {"v": "COMPUTED"}, remote_ok=True)
    assert got == {"v": 1}                      # served from remote, not recomputed
    assert cache.get("k") == {"v": 1}           # promoted into local


def test_shareable_put_writes_local_and_remote(tmp_path):
    cache, remote = _cache(tmp_path)
    cache.get_or_compute("k", lambda: {"v": 2}, remote_ok=True)
    assert cache.get("k") == {"v": 2}           # local
    assert _value(remote.store["k"]) == {"v": 2}  # remote


def test_non_shareable_does_not_read_or_write_remote(tmp_path):
    """A measured result (`remote_ok=False`) stays local-only: a value sitting
    in the remote tier is NOT served, and a freshly computed one is NOT written
    there — so one user's calibrated numbers never reach the global table."""
    cache, remote = _cache(tmp_path)
    remote.store["k"] = _stored({"v": "REMOTE"})  # someone else's shareable entry
    # remote_ok=False: ignore the remote hit, compute locally.
    got = cache.get_or_compute("k2", lambda: {"v": "LOCAL"}, remote_ok=False)
    assert got == {"v": "LOCAL"}
    assert "k2" not in remote.store             # not written to the global table
    # and even for the same key, a non-shareable read skips the remote tier
    miss = cache.get_or_compute("k", lambda: {"v": "FRESH"}, remote_ok=False)
    assert miss == {"v": "FRESH"}               # remote "REMOTE" ignored, recomputed local


def test_unconfigured_supabase_degrades_to_local(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    from backend.service.sweep_cache import SupabaseBackend

    local = TieredBackend(MemoryBackend(max_entries=8), DiskBackend(tmp_path, max_files=100))
    cache = KeyedCache("sweep", local, single_flight=True, remote=SupabaseBackend())
    cache.get_or_compute("k", lambda: {"v": 3}, remote_ok=True)  # remote put is a no-op
    assert cache.get("k") == {"v": 3}           # served from local


def test_remote_if_gates_only_the_remote_write(tmp_path):
    """`remote_if` confines a value to the local tiers while still caching it
    locally — the gallery uses this so one box's transient error entry never
    reaches the global table but still caches (and serves) locally."""
    cache, remote = _cache(tmp_path)
    err = {"model_id": "m", "error": "boom"}
    got = cache.get_or_compute(
        "k", lambda: err, remote_ok=True, remote_if=lambda v: not v.get("error"))
    assert got == err
    assert cache.get("k") == err                # local store happened
    assert "k" not in remote.store              # remote write gated off
    # A clean value under another key writes both tiers.
    cache.get_or_compute(
        "ok", lambda: {"v": 1}, remote_ok=True, remote_if=lambda v: not v.get("error"))
    assert _value(remote.store["ok"]) == {"v": 1}


def test_get_put_remote_ok_roundtrip(tmp_path):
    """The manual get/put pair honours the same gates (gallery facts and
    frontier entries cache through them, outside get_or_compute)."""
    cache, remote = _cache(tmp_path)
    trace = _trace()
    cache.put("k", {"v": 1}, trace=trace)       # default: local-only
    assert "k" not in remote.store
    cache.put("k2", {"v": 2}, trace=trace, remote_ok=True)
    assert _value(remote.store["k2"]) == {"v": 2}
    cache.put("k3", {"v": 3}, trace=trace, remote_ok=True,
              remote_if=lambda v: False)
    assert "k3" not in remote.store
    assert cache.get("k3") == {"v": 3}
    # A remote-only value is visible through get(remote_ok=True) and
    # promoted; a plain get() stays local-only.
    remote.store["k4"] = _stored({"v": 4})
    assert cache.get("k4") is None
    assert cache.get("k4", remote_ok=True) == {"v": 4}
    assert cache.get("k4") == {"v": 4}          # promoted into local


def test_service_import_leaves_the_core_caches_local_only():
    """The service layer binds NO remote tier to the core sweep + gallery
    caches: a cold rebuild is seconds (float projection + matrix fold), so
    each machine warms its own local disk cache (5841a9f81). The caches are
    built remote-less so `backend.api` stays service-clean, and the service
    import is the one place a remote could be bound — so pin that it binds
    none: a re-added Supabase tier would bring back the timing-out
    `sweep_cache` RPCs on every request."""
    import backend.api  # noqa: F401  — imports backend.service, running its wiring
    from backend.api.sweep_runtime import _SWEEP
    from backend.api.gallery_overview import _FRONTIER, _GALLERY
    assert _SWEEP._remote is None
    assert _GALLERY._remote is None
    assert _FRONTIER._remote is None


def test_an_entry_from_before_traces_is_a_miss(tmp_path):
    """A stored value that names no trace says nothing about the code that
    computed it: recomputed, and overwritten with one that does."""
    cache, remote = _cache(tmp_path)
    remote.store["k"] = {"v": "OLD"}
    assert cache.get_or_compute("k", lambda: {"v": "NEW"},
                                remote_ok=True) == {"v": "NEW"}
    assert _value(remote.store["k"]) == {"v": "NEW"}


def test_an_entry_whose_trace_is_stale_is_a_miss_and_is_overwritten(tmp_path):
    cache, remote = _cache(tmp_path)
    stale = Trace((("<engine>", "an-older-engine"),))
    cache.put("k", {"v": "OLD"}, trace=stale, remote_ok=True)
    assert cache.get("k", remote_ok=True) is None
    assert cache.get_or_compute("k", lambda: {"v": "NEW"},
                                remote_ok=True) == {"v": "NEW"}
    assert cache.get("k", remote_ok=True) == {"v": "NEW"}


def test_a_remote_entry_brings_its_trace_from_the_remote_tier(tmp_path):
    """Another machine's entry names a trace this one never recorded: its
    body is read from the remote tier, where the writer put it."""
    cache, remote = _cache(tmp_path)
    trace = Trace(_trace().items + (("x.py::f", "0123456789abcdef"),))
    other = KeyedCache("sweep", MemoryBackend(), remote=remote)
    other.put("k", {"v": 1}, trace=trace, remote_ok=True)
    assert REMOTE_TRACE_PREFIX + trace.id in remote.store
    assert remote.store["k"][TRACE_FIELD] == trace.id
