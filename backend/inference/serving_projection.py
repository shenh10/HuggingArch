"""Direct projections of canonical serving-law records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from backend.inference.card import Card
from backend.inference.sol_data import (
    ServingWalkFacts,
    ServingWalkKey,
    StepLawKey,
)
from backend.inference.sol_law import servingLawPoints
from backend.inference.sweep_grid import ROUTINE_SEQ_LENS
from backend.measure.scope import NONE


# Capacity facts are precomputed for these contexts in the canonical record.
# Floors, batches, prefixes and verify widths are law inputs, not cache axes.
servingProjectionContexts = tuple(ROUTINE_SEQ_LENS)


@dataclass(frozen=True)
class PublicServingLawRequest:
    """One request a canonical serving law can answer by substitution.

    The pool owns model/version, card class, precision and producer workload
    facts. A request may choose any positive law point — including non-default
    floors and batch — as long as the record carries capacity facts for its
    contexts. Measurement folding is not a SOL projection and remains the
    complete estimate's question.
    """
    walk: ServingWalkKey
    card: str
    layout: Any
    environment: Mapping[str, Any]
    contexts: tuple[int, ...]
    decode_batches: tuple[int | None, ...]
    prefill_batches: tuple[int, ...]
    prefix_tokens: tuple[int, ...]
    verify_widths: tuple[int, ...]
    measurement_scope: Any = NONE
    draft_tokens: int = 0

    def __post_init__(self) -> None:
        layout = (
            self.layout.to_dict() if hasattr(self.layout, "to_dict")
            else self.layout)
        if layout.get("pp", 1) != 1:
            raise ValueError(
                "a serving request does not price pipeline stages")
        if Card.of(self.card).klass.key() != self.walk.card_class:
            raise ValueError(
                f"card {self.card} is not in serving law's walk class "
                f"{self.walk.card_class}")
        if len(self.prefill_batches) != len(self.decode_batches):
            raise ValueError(
                "a serving request needs one prefill batch per decode batch")
        if (len(self.prefix_tokens) != len(self.contexts)
                or len(self.verify_widths) != len(self.contexts)):
            raise ValueError(
                "a serving request needs one prefix and verify width per context")

    @property
    def law_key(self) -> StepLawKey:
        return StepLawKey.of_layout(self.walk, self.layout)

    @classmethod
    def covers(
            cls, *, environment: Mapping[str, Any],
            contexts: Sequence[int],
            measurement_scope: Any = NONE) -> bool:
        """Whether this request is the producer walk's question to answer."""
        return (
            ServingWalkFacts.from_environment(environment)
            == ServingWalkFacts.producer()
            and (measurement_scope is NONE or measurement_scope is None)
            and bool(contexts)
            and set(contexts) <= set(servingProjectionContexts))


def projectServingLawPoints(
        record: Mapping[str, Any],
        request: PublicServingLawRequest, *,
        floorForms: bool = False) -> list[list[dict | None]]:
    """Project a serving record's contexts and batches in one call; with
    ``floorForms`` each decode phase also carries its step as a function of
    the two floors (`sol_law.servingLawPoints`)."""
    return servingLawPoints(
        record,
        card=Card.of(request.card),
        contexts=request.contexts,
        decodeBatches=request.decode_batches,
        outputTokens=int(request.environment["avg_output_tokens"]),
        prefillBatches=request.prefill_batches,
        prefixCached=request.prefix_tokens,
        verifyWidth=request.verify_widths,
        draftTokens=request.draft_tokens,
        acceptanceRate=float(
            request.environment.get("acceptance_rate") or 1.0),
        comm_latency_us=float(
            request.environment.get("comm_latency_us") or 0.0),
        launch_floor_us=float(
            request.environment.get("launch_floor_us") or 0.0),
        servedOnly=False,
        floorForms=floorForms)

