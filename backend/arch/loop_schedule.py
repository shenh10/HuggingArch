"""A stack's executions in order, and the stored layer and pass of each.

A stack's iterations are its executions. Without a loop each stored layer
runs once and the iteration IS the stored layer. With one (`LoopSpec`), a
range of stored layers runs `passes` times, each pass followed by the
loop's gate when it declares one, so iterations and stored layers are two
coordinates: the walk, the occurrences, the carried chain, the stream tables
and the pipeline cut number executions, and what a weight path needs (which
checkpoint index, which per-layer format) is the stored layer an execution
runs. This table is the one statement of that projection
(`ResolvedSpec.execution_schedule`); nothing else restates where a loop
sits.
"""
from __future__ import annotations

from dataclasses import dataclass
from backend.code_trace.memo import memo_property
from typing import Dict, Literal, Optional, Tuple


@dataclass(frozen=True)
class Execution:
    """One iteration of a stack: the stored layer it runs (for a gate, the
    pass index its stored copy is kept under), its pass (1 outside a loop),
    and whether it runs a layer or the loop's gate."""

    stored: int
    pass_: int
    part: Literal["layer", "gate"] = "layer"


@dataclass(frozen=True)
class ExecutionSchedule:
    """The executions of one stack, ordinal → `Execution`."""

    stored_layers: int
    executions: Tuple[Execution, ...]
    #: The loop's stored range and pass count; None without a loop.
    loop: Optional[Tuple[int, int, int]] = None

    @staticmethod
    def unlooped(stored_layers: int) -> "ExecutionSchedule":
        return ExecutionSchedule(
            stored_layers,
            tuple(Execution(i, 1) for i in range(stored_layers)))

    @staticmethod
    def looped(stored_layers: int, first: int, last: int, passes: int,
               gated: bool, where: str) -> "ExecutionSchedule":
        """Stored layers `0..first-1`, then `first..last` once per pass,
        then `last+1..`; a gate after each pass when `gated`."""
        if not 0 <= first <= last < stored_layers:
            raise ValueError(
                f"{where}: `loop.layers: [{first}, {last}]` is not a "
                f"non-empty range of the stack's {stored_layers} stored "
                f"layers (0..{stored_layers - 1})")
        if passes < 2:
            raise ValueError(
                f"{where}: `loop.passes: {passes}` runs the range once; a "
                f"loop runs it at least twice")
        out = [Execution(i, 1) for i in range(first)]
        for pass_ in range(1, passes + 1):
            out.extend(Execution(i, pass_) for i in range(first, last + 1))
            if gated:
                out.append(Execution(pass_ - 1, pass_, "gate"))
        out.extend(Execution(i, 1) for i in range(last + 1, stored_layers))
        return ExecutionSchedule(stored_layers, tuple(out),
                                 (first, last, passes))

    def __len__(self) -> int:
        return len(self.executions)

    @property
    def is_looped(self) -> bool:
        return self.loop is not None

    @memo_property
    def _by_layer(self) -> Dict[Tuple[int, int], int]:
        return {(e.stored, e.pass_): ordinal
                for ordinal, e in enumerate(self.executions)
                if e.part == "layer"}

    def layer_ordinals(self) -> Tuple[int, ...]:
        """Every ordinal that runs a layer (not a gate), ascending."""
        return tuple(i for i, e in enumerate(self.executions)
                     if e.part == "layer")

    def gate_ordinals(self) -> Tuple[int, ...]:
        """The gate's ordinals, one per pass, ascending."""
        return tuple(i for i, e in enumerate(self.executions)
                     if e.part == "gate")

    def ordinal(self, stored: int, pass_: int) -> Optional[int]:
        """The execution of stored layer `stored` in pass `pass_`, or None
        when that layer has no such pass."""
        return self._by_layer.get((stored, pass_))

    def ordinals_of(self, stored: int) -> Tuple[int, ...]:
        """Every execution of stored layer `stored`, ascending."""
        return tuple(sorted(o for (s, _p), o in self._by_layer.items()
                            if s == stored))

    def latest_before(self, stored: int, before: int) -> Optional[int]:
        """The last execution of stored layer `stored` before ordinal
        `before`, or None."""
        found = [o for o in self.ordinals_of(stored) if o < before]
        return found[-1] if found else None

    def in_loop(self, stored: int) -> bool:
        return (self.loop is not None
                and self.loop[0] <= stored <= self.loop[1])

    def passes(self) -> int:
        return self.loop[2] if self.loop is not None else 1

    def pass_entry(self, pass_: int) -> int:
        """The ordinal whose exit is the carried value entering pass
        `pass_` — the iteration before the pass's first layer; -1 when that
        is the stack's own input."""
        if self.loop is None:
            raise ValueError("an unlooped stack has no passes")
        return self._by_layer[(self.loop[0], pass_)] - 1
