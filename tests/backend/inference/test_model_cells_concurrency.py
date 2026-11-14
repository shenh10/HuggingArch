"""Two requests that ask for one model's cells at once: the first builds
them, the second waits for that build and reads the same cells
(`capacity._model_cells`, single flight)."""
from __future__ import annotations

import threading
import time

from backend.inference import capacity
from backend.inference.parallel_config import ParallelConfig
from backend.inference import walk_registry
from tests.snapshots import resolve_snapshot

LLAMA = "meta-llama/Llama-3.2-3B"


def test_a_cold_key_is_built_once_while_the_second_caller_waits(monkeypatch):
    resolved = resolve_snapshot(LLAMA)
    precision = capacity.PrecisionProfile.from_request(None)
    parallel = ParallelConfig(tp_attention=2, tp_dense_ffn=2)
    walk_registry.WALK_CELLS.clear()

    entered, release = threading.Event(), threading.Event()
    builds = []
    real = walk_registry.StepCells

    def blocking_cells(runtime_profile):
        builds.append(threading.get_ident())
        entered.set()
        assert release.wait(30)
        return real(runtime_profile)

    monkeypatch.setattr(walk_registry, "StepCells", blocking_cells)
    got = {}

    def ask(name):
        got[name] = capacity._model_cells(resolved, parallel, precision,
                                          model_id=LLAMA)

    first = threading.Thread(target=ask, args=("first",))
    first.start()
    assert entered.wait(30)
    second = threading.Thread(target=ask, args=("second",))
    second.start()
    # The second caller is waiting on the first one's build, not building.
    deadline = time.monotonic() + 30
    while not any(n == 2 for _lock, n in walk_registry.WALK_CELLS._key_locks.values()):
        assert time.monotonic() < deadline, "the second caller never waited"
        time.sleep(0.01)
    assert len(builds) == 1 and "second" not in got
    release.set()
    first.join(30)
    second.join(30)
    assert len(builds) == 1
    assert got["first"] is got["second"]
    assert walk_registry.WALK_CELLS._key_locks == {}


def test_one_cell_builds_a_serving_reading_once_for_two_threads(monkeypatch):
    """Two readers of one cell asking for the same serving layers at once:
    the first builds them under the cell's lock, the second waits and reads
    the same object (`step_eval.Cell.serving`)."""
    from backend.inference import step_eval
    from backend.inference.step_eval import Cell

    cell = Cell(spec=object(), parallel=ParallelConfig(), phase="prefill")
    entered, release = threading.Event(), threading.Event()
    builds = []

    def build(cell, *, framework_binding, stage, timed):
        builds.append(threading.get_ident())
        entered.set()
        assert release.wait(30)
        return object()

    monkeypatch.setattr(step_eval, "build_serving_layers", build)
    got = {}

    def ask(name):
        got[name] = cell.serving(None, None, frozenset({"lm"}))

    first = threading.Thread(target=ask, args=("first",))
    first.start()
    assert entered.wait(30)
    second = threading.Thread(target=ask, args=("second",))
    second.start()
    time.sleep(0.2)
    assert "second" not in got and len(builds) == 1
    release.set()
    first.join(30)
    second.join(30)
    assert len(builds) == 1
    assert got["first"] is got["second"]


def test_a_cell_reading_is_the_one_its_key_builds(monkeypatch):
    """A cell's serving layers and liveness are built by the cell's own one
    builder, keyed by what they are read under: a caller cannot hand in
    another builder, so two builders can never be kept under one key; one
    key reads one object, another key another, and the liveness of a key is
    folded off the serving layers of the same key."""
    import inspect

    import pytest

    from backend.inference import step_eval
    from backend.inference.step_eval import Cell

    built = []

    def build(cell, *, framework_binding, stage, timed):
        built.append((framework_binding, stage, timed))
        return object()

    folded = []

    class Liveness:
        def __init__(self, layers, laws):
            folded.append(layers)

    monkeypatch.setattr(step_eval, "build_serving_layers", build)
    monkeypatch.setattr(step_eval, "ActivationLiveness", Liveness)
    cell = Cell(spec=object(), parallel=ParallelConfig(), phase="prefill")
    for method in (Cell.serving, Cell.liveness):
        assert not any(name.startswith("build")
                       for name in inspect.signature(method).parameters)
    with pytest.raises(TypeError):
        cell.serving(None, None, frozenset({"lm"}), lambda *a, **k: object())
    lm, mtp = frozenset({"lm"}), frozenset({"lm", "mtp"})
    assert cell.serving(None, None, lm) is cell.serving(None, None, lm)
    assert cell.serving(None, None, lm) is not cell.serving(None, None, mtp)
    assert built == [(None, None, lm), (None, None, mtp)]
    cell.liveness(None, mtp)
    assert folded == [cell.serving(None, None, mtp)]
