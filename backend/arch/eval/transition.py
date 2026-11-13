"""The walk's transition protocol — what a rule requires and produces.

`Evaluator` threads distributed state along the same edges as the shapes
and asks its rule what each node consumes and emits. The public occupant
is `LayoutPropagation`. This module holds only the records both sides
share: a site, an input/output bundle, and one redistribute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

from .occurrence import StageCrossing


class RedistributeSite(str, Enum):
    """Structural site at which a state transition occurs.

    ``input_index`` remains the positional edge coordinate used by existing
    analysis consumers.  It cannot identify the site on its own: ``-1`` has
    historically represented both a stage close and an op-internal output.
    Materializers need that distinction to decide whether to rewrite a graph
    edge or append an endpoint event, so the walk records it explicitly.
    """

    EDGE = "edge"
    INTERNAL_IN = "internal_in"
    INTERNAL_OUT = "internal_out"
    STAGE_CLOSE = "stage_close"
    STAGE_CROSSING = "stage_crossing"
    CACHE_ENDPOINT = "cache_endpoint"
    MODEL_OUTPUT = "model_output"


@dataclass(frozen=True)
class OutputStates:
    """Distributed states for a node's primary and named output ports.

    Most rules return one state directly.  A true multi-output primitive can
    instead return this wrapper, and the evaluator publishes every port during
    the same walk that propagated its shape.  Secondary names exclude the
    conventional primary ``out`` port by construction.
    """

    out: Any
    secondary: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _state in self.secondary]
        if "out" in names:
            raise ValueError("OutputStates.secondary cannot redefine 'out'")
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                f"secondary output names must be non-empty and unique: {names!r}")


@dataclass(frozen=True)
class KernelGeometry:
    """Concrete physical input/output axes for one kernel invocation."""

    inputs: Tuple[Optional[Tuple[Any, ...]], ...]
    output: Optional[Tuple[Any, ...]]
    # Actual work can be ragged across query rows even when allocation
    # axes are rectangular (striped sparse verification).
    pair_share: Any = None


@dataclass(frozen=True)
class InputRequirements:
    """Separate an SSA edge repair from the buffer a kernel consumes.

    Usually both views are identical, so transition rules keep returning a
    plain sequence of states.  An op backed by persistent state can instead
    use the edge only to identify the logical value while reading a differently
    placed runtime buffer.  ``edges`` drives redistribution; ``kernels`` is
    recorded in ``Analysis.input_states`` and drives physical kernel geometry.
    """

    edges: Tuple[Any, ...]
    kernels: Tuple[Any, ...]
    geometry: Optional[KernelGeometry] = None

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.kernels):
            raise ValueError(
                "InputRequirements edge and kernel arity must match: "
                f"{len(self.edges)} != {len(self.kernels)}"
            )
        if (
            self.geometry is not None
            and len(self.geometry.inputs) != len(self.kernels)
        ):
            raise ValueError(
                "InputRequirements kernel state and geometry arity must "
                f"match: {len(self.kernels)} != "
                f"{len(self.geometry.inputs)}"
            )


@dataclass(frozen=True)
class Redistribute:
    """One edge where the state that arrived is not the state the op needs.

    Carries both endpoints rather than a classification of the difference. What
    KIND of change it is — a gather, a reduction, an exchange — is what
    `collectives` already says, decomposed per mesh world; a second derivation
    of it would be free to disagree, and a classification enum is three
    readings of one fact.
    """

    node: int
    input_index: int
    producer: int
    arrived: Any
    required: Any
    #: `(world, collective)` per differing mesh axis, in world order
    collectives: Tuple[Tuple[Optional[str], str], ...] = ()
    #: The producer's SSA output identity. ``out`` is the primary output;
    #: named ports are distinct tensors even when their states compare equal.
    producer_port: str = "out"
    #: The graph site owns topology; sentinel input indices are retained only
    #: as the established analysis coordinate and are not re-interpreted.
    site: RedistributeSite = RedistributeSite.EDGE
    #: For a hop across a pipeline cut (`site` `STAGE_CROSSING`): which
    #: value crosses which cut, for which sinks (`occurrence.StageCrossing`).
    #: None for every in-stage transition.
    crossing: Optional[StageCrossing] = None
    #: The earlier redistribute object whose required state is this record's
    #: arrived state. An internal collective that consumes an edge already
    #: settled by another collective is an ordered event chain, not another
    #: reader of the producer. Object identity survives the analysis's own
    #: reorderings and pickling; a list index would not.
    dependency: Optional[Any] = field(
        default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.producer_port, str) or not self.producer_port:
            raise ValueError("Redistribute.producer_port must be non-empty")
        if not isinstance(self.site, RedistributeSite):
            raise TypeError(
                f"Redistribute.site must be RedistributeSite, got {self.site!r}"
            )

    @property
    def cut(self) -> Optional[int]:
        """The cut a hop crosses: it is sent by stage `cut` to stage
        `cut + 1`. None for an in-stage transition."""
        return None if self.crossing is None else self.crossing.cut

    @property
    def resolves_a_partial(self) -> bool:
        """Whether the producer held a partial that has to be reduced.

        The STATE answers, not this record — so a state type with a
        different value algebra answers for itself.
        """
        return self.arrived.carries_partial()
