"""What a compute-pool cell hands back to the server is plain data.

A cell runs in a worker; its result crosses back through the pool's one
result thread, which unpickles it. A SymPy expression in it is rebuilt there
— `Mul.flatten`, a `Max` that compares its terms — on the thread every other
request's result waits behind. The decode frontier once returned each
point's whole estimate, per-op shape and point laws included: tens of
thousands of expressions per layout, some of which do not rebuild at all, and
after a restart the light endpoints answered 503 for as long as it ran. The
laws live in the process that walked them (`module_rows.served`,
`sweep_runtime.served_sweep_point`); this checks each kind of cell keeps them
there.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.api import gallery_frontier, sweep_runtime, warm
from backend.api.gallery_frontier import _resolve_one
from backend.inference.defaults import GPU_MEMORY_FRACTION_DEFAULT
from backend.inference.parallel_sweep import class_cards, frontier_layouts, sweep_caps

MODEL = "Qwen/Qwen3-8B"


def _expressions(value, path="result"):
    """Where SymPy objects sit in `value`, as paths."""
    if isinstance(value, sympy.Basic):
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _expressions(v, f"{path}.{k}")]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in _expressions(v, f"{path}[{i}]")]
    return []


def _assert_plain(result):
    found = _expressions(result)
    assert not found, f"{len(found)} SymPy objects, e.g. {found[:3]}"
    pickle.loads(pickle.dumps(result))


def test_a_frontier_layout_returns_plain_laws():
    layout = frontier_layouts(sweep_caps({}, resolved=_resolve_one(MODEL)[1]), "B300")[0]
    result = gallery_frontier._layout_law(
        (MODEL, class_cards("B300"), GPU_MEMORY_FRACTION_DEFAULT,
         gallery_frontier._GALLERY_PRECISION.canonical(), layout.to_dict()))
    assert result.get("schema") == gallery_frontier.walkRecordSchema, result
    _assert_plain(result)


def test_a_scaling_sweep_cell_returns_plain_points():
    requests = (
        (sweep_runtime._PARALLEL, warm.default_scaling_sweep_request(MODEL)),
        (sweep_runtime._ESTIMATE,
         warm.default_scaling_batch_request(MODEL, "B200", "dp_ep", 8)),
    )
    for sweep, request in requests:
        plan = sweep.plan(request)
        # Through the step machinery a worker runs, keeping nothing.
        _assert_plain(sweep_runtime._run_cell(sweep_runtime._Cell(
            sweep, sweep.cells(plan)[0], sweep_runtime._Partials(None, 0))))
