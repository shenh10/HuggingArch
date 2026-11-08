"""Fixtures for the Kernel Bench lake tests (backend/service/bench).

  lake          a fresh local lake per test (a .ducklake catalog file and a
                data directory under tmp_path), schema applied, its in-memory
                annotations built (`store.annotate_all`)
  make_rows     valid CSV-row dicts (strings, as `validate_text` yields them),
                distinct identities, any field overridable
  annotate      a deterministic stand-in for `bench.annotations.row_annotations`
                (which imports kernel_bench): same keys, no torch
  make_annotate the same with another signature prefix
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Mapping

import pytest

from backend.inference.benchmark_identity import workload_coordinates
from backend.inference.benchmarks import validate_text
from backend.service.bench import store
from backend.service.bench.lake import Lake, LakeConfig
from tests.corpus_row import row as corpus_row
from tests.corpus_row import to_csv


@pytest.fixture
def lake(tmp_path):
    lk = Lake.open(LakeConfig(catalog=str(tmp_path / "catalog.ducklake"),
                              data=str(tmp_path / "data"),
                              cache_dir=str(tmp_path / "cache")))
    store.ensure_schema(lk)
    store.annotate_all(lk, make_annotator())
    yield lk
    lk.close()


def _rows(n: int = 1, **over: Any) -> list[dict]:
    """`n` valid rows, as `validate_text` yields them (checked). Row i's
    shape is the corpus row's with s_ctx 2048·(i+1), so identities differ;
    an override is a string, or a callable of the row index returning one."""
    out = []
    for i in range(n):
        shape = json.loads(corpus_row()["shape_json"])
        shape["s_ctx"] = 2048 * (i + 1)
        fields = {"shape_json": json.dumps(shape, sort_keys=True, separators=(",", ":"))}
        for k, v in over.items():
            fields[k] = v(i) if callable(v) else v
        out.append(corpus_row(**fields))
    return validate_text(to_csv(out), source="make_rows")[1]


@pytest.fixture
def make_rows() -> Callable[..., list[dict]]:
    return _rows


def make_annotator(prefix: str = "sig") -> Callable[[Mapping[str, Any]], dict]:
    def annotate(inp: Mapping[str, Any]) -> dict:
        shape = store.canonical_json(inp["shape_json"])
        batch, context = workload_coordinates(inp["phase"], shape)
        return {"signature": f"{prefix}:{inp['op']}", "physical_shape": shape,
                "workload_batch": batch, "workload_context": context,
                "catalog_kernel": inp["kernel_impl"]}
    return annotate


@pytest.fixture
def annotate():
    return make_annotator()


@pytest.fixture
def make_annotate():
    return make_annotator


@pytest.fixture
def new_owner() -> Callable[[], str]:
    """A fresh owner uuid per call."""
    return lambda: str(uuid.uuid4())
