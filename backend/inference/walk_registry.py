"""The process-wide registry of walks shared by estimate and plan.

A `StepCells` value is the in-process owner of one model/deployment/precision
family's symbolic walks. Capacity uses one for every point of an estimate; the
Kernel Bench planner needs the same family when it turns those walks into
physical measurement jobs. Before this module each entry point constructed its
own `StepCells`, so a plan and an estimate in the same process could hold two
walks for the same facts even though both were careful to walk only once
*within* their own call.

This is deliberately not the durable `WalkKey` tier (`measure.scope.of_walk`):
it is a bounded, in-process derivation registry. It names the model's result
version, the checkpoint config a binding may read, the deployment and the
precision profile — the facts `StepCells` is a function of — and never a
point, card price, framework view or measurement.
Framework bindings, pipeline stages and timed-stack projections are readers
attached to a cell (`Cell.serving`, keyed there by exactly those facts), so a
plan's measurable-launch projection and an estimate's served view share the
symbolic walk without being forced to expose the same row subset.
"""
from __future__ import annotations
from backend.arch.runtime_profile import RuntimeProfile

import json
from dataclasses import dataclass, field
import threading
from typing import Any, Callable, Optional

from backend.code_trace.provider import Growing, provider
from backend.fingerprint import data_version
from backend.measure.cache import KeyedCache, MemoryBackend, deep_size_of
from backend.measure.scope import make_key

from .step_eval import StepCells


@dataclass(frozen=True)
class WalkCellKey:
    """What a family of shared walks is a function of."""
    model_id: Optional[str]
    version: str
    config: str
    parallel: str
    precision: str
    runtime_profile: RuntimeProfile = RuntimeProfile.DECLARED

    def key(self) -> str:
        return make_key("walk-cells", self.model_id, self.version, self.config,
                        self.parallel, self.precision, self.runtime_profile.value)


@dataclass
class SharedWalkCells:
    """One registry entry: the resolved spec, its cells, and derived readers.

    `readers` holds the expensive whole-family projections that sit above the
    symbolic walks but below a consumer (capacity's placement/budget basis).
    A reader is named by its owner and built only by the function that owns the
    concept; the registry provides identity and memoisation, not a second
    derivation.
    """
    resolved: Any
    cells: StepCells
    key: WalkCellKey
    cache_key: str
    #: The code every computation that read this entry ran.
    trace: Growing = field(default_factory=Growing, repr=False)
    _readers: dict[str, Any] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reader(self, name: str, build: Callable[[], Any]) -> Any:
        """A named projection of this one walk family, built once."""
        with self._lock:
            if name not in self._readers:
                self._readers[name] = build()
            reader = self._readers[name]
        # The reader may have walked and attached state to `cells`; the cache
        # sized this entry before that lazy work existed. Re-account it once
        # per completed reader build through the same byte budget.
        self.account()
        return reader

    def account(self) -> bool:
        """Re-measure this family under `WALK_CELLS`' byte budget."""
        return WALK_CELLS.resize(self.cache_key, self)


# A process-local memo, not an estimate-result cache: at most sixteen
# model/deployment families and, more importantly, at most this much resident
# walk state. Entry count alone cannot bound memory: one Kimi family is many
# times a 3B family. `StepCells` and the readers attached to each cell provide
# the finer-grained bounds below this entry; a family larger than the whole
# budget is returned to its caller but not retained.
WALK_CELLS_MEMORY_BYTES = 768 * 1024 * 1024
WALK_CELLS = KeyedCache(
    "walk_cells",
    MemoryBackend(max_entries=16, max_bytes=WALK_CELLS_MEMORY_BYTES,
                  size_of=deep_size_of),
    single_flight=True)


def shared_walk_cells(resolved: Any, parallel: Any, precision: Any, *,
                      model_id: Optional[str] = None,
                      runtime_profile: RuntimeProfile = RuntimeProfile.DECLARED) -> SharedWalkCells:
    """The `StepCells` for one model, deployment and precision.

    `parallel` and `precision` are their canonical domain objects
    (`ParallelConfig`, `PrecisionProfile`), not request dictionaries: this is
    below the HTTP boundary, where spelling has already been resolved once.
    The raw config is named beside the engine version because a framework
    binding may read checkpoint fields (the Mamba scan chunk, for example)
    that the model footprint digest does not change.
    """
    p = provider()
    # The key is derivation, not a dependency of what the entry holds.
    with p.untraced():
        key = WalkCellKey(model_id=model_id, version=data_version(resolved),
                          config=json.dumps(resolved.config, sort_keys=True,
                                            default=str, separators=(",", ":")),
                          parallel=repr(parallel), precision=precision.key,
                          runtime_profile=RuntimeProfile(runtime_profile))

    def build() -> SharedWalkCells:
        return SharedWalkCells(resolved, StepCells(key.runtime_profile), key, key.key())

    entry = WALK_CELLS.get_or_compute(key.key(), build)
    p.attach(entry.trace)
    return entry
