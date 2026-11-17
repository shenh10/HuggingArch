"""The estimate and Kernel Bench planner read one walk-family registry."""
from __future__ import annotations

import inspect

import pytest

from backend.inference import walk_registry
from backend.measure.cache import TRACE_FIELD, deep_size_of


def test_capacity_and_plan_have_no_private_step_cells():
    """No entry point creates its own `StepCells` family."""
    from backend.inference import capacity
    from kernel_bench import plan

    for module, name in ((capacity, "_model_cells"), (plan, "_build_jobs_over_points")):
        source = inspect.getsource(getattr(module, name))
        assert "StepCells()" not in source, name
    assert "shared_walk_cells(" in inspect.getsource(capacity._model_cells)
    assert "shared_walk_cells(" in inspect.getsource(
        plan._build_jobs_over_points)


def test_a_walk_family_key_names_no_point_card_price_or_measurement():
    fields = tuple(walk_registry.WalkCellKey.__dataclass_fields__)
    assert fields == ("model_id", "version", "config", "parallel", "precision",
                      "runtime_profile")


def test_walk_families_are_bound_by_resident_bytes_not_only_by_count():
    backend = walk_registry.WALK_CELLS._backend
    assert backend._max == 16
    assert backend._max_bytes == walk_registry.WALK_CELLS_MEMORY_BYTES
    assert backend._size_of is deep_size_of


def test_a_reader_that_grows_a_walk_family_reaccounts_its_byte_budget():
    from backend.measure.cache import KeyedCache, MemoryBackend

    backend = MemoryBackend(max_bytes=1024 * 1024, size_of=deep_size_of)
    cache = KeyedCache("test-walk-cells", backend)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(walk_registry, "WALK_CELLS", cache)
    try:
        shared = walk_registry.SharedWalkCells(
            resolved={}, cells=walk_registry.StepCells(),
            key=walk_registry.WalkCellKey("m", "v", "{}", "parallel", "p"),
            cache_key="cache-key")
        cache.put("cache-key", shared)
        assert backend._bytes < deep_size_of(
            shared.reader("grown", lambda: bytearray(256 * 1024)))
        stored = backend.get("cache-key")
        assert TRACE_FIELD in stored and stored["value"] is shared
        assert backend._bytes == deep_size_of(stored)
    finally:
        monkey.undo()
