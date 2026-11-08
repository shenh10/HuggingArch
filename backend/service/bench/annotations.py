"""The annotation rules that turn a lake input into a lookup key.

This module owns data derivation, not revision mechanics.  ``rules_revision``
names the functions and modules declared here as the data-rules boundary, so
an HTTP route, startup gate, or lookup execution edit is not mistaken for a
change to row validity or a lookup key.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Mapping

from backend.inference.benchmark_identity import (
    current_shape,
    workload_coordinates,
)
from backend.inference.op_calibration import physical_identity
from backend.service.bench import store as _store


def _catalog():
    # kernel_bench.providers imports torch (~2 s): loaded on first use.
    from kernel_bench import providers
    return providers


@lru_cache(maxsize=1)
def _shape_identity(op: str, shape_json: str) -> tuple[dict, str, str]:
    """One op/shape's catalog shape and physical identity.

    Startup streams inputs ordered by ``(op, shape_json)``, so this one-entry
    memo computes the expensive physical half once for every duplicate input
    in that run without keeping a per-process shape table. It is the same
    annotation road ``row_annotations`` always takes; callers never derive it.
    """
    shape = current_shape(op, _store.canonical_json(shape_json))
    identity = physical_identity(op, shape)
    return shape, identity["signature"], identity["physical_shape"]


def row_annotations(row: Mapping[str, Any]) -> dict:
    """The annotations of one row's input (``op``, ``shape_json``, ``phase``,
    and ``kernel_impl`` read)."""
    shape, signature, physical_shape = _shape_identity(
        row["op"], row["shape_json"])
    batch, context = workload_coordinates(row["phase"], shape)
    return {
        "signature": signature,
        "physical_shape": physical_shape,
        "workload_batch": batch,
        "workload_context": context,
        "catalog_kernel": _catalog().catalog_kernel(row["kernel_impl"]),
    }


def registered_kernels() -> Dict[str, Dict[str, str]]:
    """Every registered kernel, by library: ``{library: {kernel: signature}}``."""
    out: Dict[str, Dict[str, str]] = {}
    for p in _catalog().all_providers():
        out.setdefault(p.library, {})[p.kernel_impl] = p.signature
    return out
