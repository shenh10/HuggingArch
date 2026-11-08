"""The declared boundary of code and data that decides lake row rules.

The boundary is not an import closure.  Import closure made unrelated
infrastructure — a DuckDB connection setting, lookup table budget, or route
module — look like a row-format change and forced a full lake migration.  The
entries below are the semantic owners instead; each is derived from the
imported function/module object, not from a hand-maintained file list.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable

from backend.arch import activation_formats as _activation_formats
from backend.arch import dtypes as _dtypes
from backend.arch.eval import formats as _eval_formats
from backend.arch.spec_migrate import measured_row_migration
from backend.fingerprint import shared_data_version
from backend.inference.benchmark_identity import (
    current_shape,
    workload_coordinates,
)
from backend.inference.benchmarks import row_identity, validate_row
from backend.inference.op_calibration import physical_identity
from backend.inference.op_signatures import all_signatures
from backend.service.bench import annotations as _annotations
from backend.service.bench import data_rules as _data_rules
from backend.service.bench import store as _store


@dataclass(frozen=True)
class RuleInput:
    name: str
    owner: ModuleType | Callable[..., object]
    reason: str


_RULE_INPUTS: tuple[RuleInput, ...] = (
    RuleInput("row_annotations", _annotations.row_annotations,
              "turns a stored input into signature, physical shape and workload"),
    RuleInput("registered_kernels", _annotations.registered_kernels,
              "the loaded provider catalog that binds implementations to signatures"),
    RuleInput("catalog_kernel", _annotations._catalog().catalog_kernel,
              "names the implementation a stored kernel_impl selects"),
    RuleInput("current_shape", current_shape,
              "retires and derives shape axes in the current row identity"),
    RuleInput("measured_row_migration", measured_row_migration,
              "the shared migration table current_shape reads"),
    RuleInput("workload_coordinates", workload_coordinates,
              "derives workload batch and context from a shape"),
    RuleInput("physical_identity", physical_identity,
              "projects an op and shape to the physical lookup key"),
    RuleInput("canonical_json", _store.canonical_json,
              "canonicalizes identity JSON before it is stored or projected"),
    RuleInput("validate_row", validate_row,
              "owns whether a row is valid under the current schema"),
    RuleInput("row_identity", row_identity,
              "owns the identity-plus-outcome tuple duplicates are judged by"),
    RuleInput("dtype_rules", _dtypes,
              "dtype names, byte widths and corpus tokens used by validation"),
    RuleInput("activation_format_rules", _activation_formats,
              "activation granularity vocabulary used by validation"),
    RuleInput("format_rules", _eval_formats,
              "weight and cache format vocabulary used by validation"),
    RuleInput("signatures", all_signatures,
              "the loaded signature contracts and their identity axes"),
    RuleInput("migration_owner", _data_rules.migrate,
              "the registered migration that brings an older lake to the rules"),
    RuleInput("dtype_rekey", _data_rules.rekey_apply,
              "the precomputed dtype-env rewrite's implementation"),
)


def rule_inputs() -> tuple[RuleInput, ...]:
    """The declared semantic inputs, for tests and operational reporting."""
    return _RULE_INPUTS


def _owner_module(owner: ModuleType | Callable[..., object]) -> ModuleType:
    return owner if isinstance(owner, ModuleType) else sys.modules[owner.__module__]


@lru_cache(maxsize=1)
def rules_rev() -> str:
    """The revision of the data rules named by ``rule_inputs``.

    Hashes the source module of each declared semantic owner (not that
    module's unrelated import closure), the engine shared-data revision, every
    loaded signature, and every registered kernel. Infrastructure modules are
    absent unless they are explicitly added here as a rule owner.
    """
    repo = Path(__file__).resolve().parents[3]
    modules = {_owner_module(entry.owner) for entry in _RULE_INPUTS}
    files = {Path(module.__file__ or "") for module in modules}
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.relative_to(repo).as_posix().encode())
        h.update(path.read_bytes())
    h.update(shared_data_version().encode())
    h.update(json.dumps(sorted(repr(signature) for signature in all_signatures())).encode())
    h.update(json.dumps(_annotations.registered_kernels(), sort_keys=True).encode())
    return h.hexdigest()[:16]
