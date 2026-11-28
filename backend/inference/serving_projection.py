"""Direct projections of canonical serving-law records."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from backend.measure.cache import (
    KeyedCache,
    MemoryBackend,
    TieredBackend,
    private_copy,
)
from backend.inference.card import Card
from backend.inference.sol_data import (
    ServingWalkFacts,
    ServingWalkKey,
    StepLawKey,
)
from backend.inference.sol_law import servingLawPoints
from backend.inference.sweep_grid import (
    FRONTIER_BATCH_GRID,
    GALLERY_SEQ_LENS,
    SCALING_BATCH_GRID,
)
from backend.measure.scope import NONE, ResultIdentity
from backend.measure.timing import log_timing
from backend.sweep_cache_artifact import SWEEP_CACHE, sweep_disk_bytes


# Capacity facts are precomputed for these contexts in the canonical record.
# Floors, batches, prefixes and verify widths are law inputs, not cache axes.
servingProjectionContexts = tuple(GALLERY_SEQ_LENS)


# The public envelope Gallery and Scaling project once: every shared context
# and every public decode rung. A public request's own axes are a slice of
# this; a custom request never expands into it.
_PUBLIC_DECODE_BATCHES = tuple(
    point["decode_batch_size"] for point in FRONTIER_BATCH_GRID)
_PUBLIC_PREFILL_BATCHES = tuple(
    point["prefill_batch_size"] for point in FRONTIER_BATCH_GRID)


# One machine-local tier for projected rows. The canonical record is the
# expensive, immutable fact; this holds only the substitution of that record at
# a request's card and point axes. It is deliberately beside the sweep cache
# (and never published): Gallery and Scaling read the same rows without either
# response cache becoming the other's owner.
_PROJECTION_DISK_BYTES = sweep_disk_bytes() * 3 // 8
SERVING_PROJECTION = KeyedCache(
    "serving-projection",
    TieredBackend(
        MemoryBackend(max_entries=64, max_bytes=64 << 20),
        SWEEP_CACHE.disk("projection", max_files=50_000,
                         max_bytes=_PROJECTION_DISK_BYTES, prune_every=128),
    ),
    single_flight=True,
)


def _canonicalAxes(request: PublicServingLawRequest) -> tuple[
        PublicServingLawRequest, tuple[tuple[int, int, int], ...],
        tuple[tuple[int | None, int], ...]]:
    """A request with its axes sorted, beside those axes.

    Sorting makes a point set's order a property of the caller, not of the
    projected rows: the same set is one cache entry, read back in each order.
    """
    contexts = tuple(sorted(zip(
        request.contexts, request.prefix_tokens, request.verify_widths,
        strict=True)))
    batches = tuple(sorted(
        zip(request.decode_batches, request.prefill_batches, strict=True),
        key=lambda pair: (
            pair[0] is None, pair[0] if pair[0] is not None else 0,
            pair[1] if pair[1] is not None else 0)))
    canonical = replace(
        request,
        contexts=tuple(axis[0] for axis in contexts),
        prefix_tokens=tuple(axis[1] for axis in contexts),
        verify_widths=tuple(axis[2] for axis in contexts),
        decode_batches=tuple(axis[0] for axis in batches),
        prefill_batches=tuple(axis[1] for axis in batches),
    )
    return canonical, contexts, batches


def _recordCovers(record: Mapping[str, Any], request: PublicServingLawRequest,
                  contexts: Sequence[int]) -> bool:
    """Whether the record states the capacity facts an envelope needs.

    A canonical public record carries every shared context. A narrower test
    or hand-built record may not; widening it would turn a Gallery request the
    record can answer into an error, so that record keeps its exact axes. A
    record with no capacity block is a projection test double — the law owner
    decides what it covers.
    """
    capacity = (
        record.get("draft_capacity")
        if request.draft_tokens and "draft_capacity" in record
        else record.get("capacity"))
    if capacity is None:
        return True
    available = {
        int(item["context"]) for item in capacity
        if item.get("gpu_sku") == request.card
    }
    return set(contexts) <= available


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

    def publicEnvelope(self) -> "PublicServingLawRequest | None":
        """This request widened to the public projection envelope, or None.

        Only a whole public context grid and a whole public batch grid widen:
        a visitor's one-context or custom-batch request stays exact, so the
        envelope never turns a small request into the full grid. Prefix and
        verify width must be one value each (as every public surface asks);
        prefill must remain the interactive batch the envelope owns.
        """
        contextSet = set(self.contexts)
        if (len(contextSet) != len(self.contexts)
                or contextSet != set(servingProjectionContexts)):
            return None
        decodeSet = set(self.decode_batches)
        if (len(decodeSet) != len(self.decode_batches)
                or decodeSet not in (set(_PUBLIC_DECODE_BATCHES),
                                     {point["decode_batch_size"]
                                      for point in SCALING_BATCH_GRID})
                or set(self.prefill_batches) != {1}):
            return None
        if (len(set(self.prefix_tokens)) != 1
                or len(set(self.verify_widths)) != 1):
            return None
        return replace(
            self,
            contexts=tuple(servingProjectionContexts),
            prefix_tokens=(self.prefix_tokens[0],)
                         * len(servingProjectionContexts),
            verify_widths=(self.verify_widths[0],)
                          * len(servingProjectionContexts),
            decode_batches=_PUBLIC_DECODE_BATCHES,
            prefill_batches=_PUBLIC_PREFILL_BATCHES,
        )

    def projection_identity(self, *, recordDigest: str) -> ResultIdentity:
        """The identity of this request's projected rows.

        The record half comes from ``law_key`` (model, version, card class,
        layout and producer facts) and the pool slot digest the caller read.
        The content half is exactly what
        ``projectServingLawPoints`` substitutes: the card, contexts, paired
        batches, prefixes, verify widths, draft tokens and the law inputs the
        record does not own. Response-only fields are absent by construction.
        Axes are canonicalized by their own values, and the richer floor-form
        row is the stored shape, so order and the caller's floor-form view
        never split one projection.
        """
        _canonical, contexts, batches = _canonicalAxes(self)
        return ResultIdentity(
            kind="serving-projection",
            version={"record": self.law_key.identity.key,
                     "recordDigest": recordDigest,
                     "schema": "serving-projection-v2"},
            content={
                "card": self.card,
                "contexts": contexts,
                "batches": batches,
                "draft_tokens": int(self.draft_tokens),
                "output_tokens": int(self.environment["avg_output_tokens"]),
                "acceptance_rate": float(
                    self.environment.get("acceptance_rate") or 1.0),
                "comm_latency_us": float(
                    self.environment.get("comm_latency_us") or 0.0),
                "launch_floor_us": float(
                    self.environment.get("launch_floor_us") or 0.0),
            },
            scope=NONE,
            reads=False,
            local_only=True,
        )

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
        floorForms: bool = False, publicEnvelope: bool = False,
        recordDigest: str) -> list[list[dict | None]]:
    """Project a serving record's contexts and batches in one call; with
    ``floorForms`` each decode phase also carries its step as a function of
    the two floors (`sol_law.servingLawPoints`).

    The cache stores the richer floor-form row and strips that view only from
    the caller's copy, so a Gallery request and a Scaling request share one
    projection. With ``publicEnvelope`` a whole public request is widened to
    the shared-context × public-batch envelope and narrowed back to the
    caller's axes; a custom request stays exact.

    ``recordDigest`` is the pool slot digest the caller read the record under.
    It is an explicit input, never re-derived here: a projection cache hit
    must not serialize and hash a whole law record again.
    """
    started = time.perf_counter()
    target = request.publicEnvelope() if publicEnvelope else None
    if (target is not None
            and not _recordCovers(record, target, target.contexts)):
        target = None
    target = target if target is not None else request
    canonical, contexts, batches = _canonicalAxes(target)

    def project() -> list[list[dict | None]]:
        return servingLawPoints(
            record,
            card=Card.of(canonical.card),
            contexts=canonical.contexts,
            decodeBatches=canonical.decode_batches,
            outputTokens=int(canonical.environment["avg_output_tokens"]),
            prefillBatches=canonical.prefill_batches,
            prefixCached=canonical.prefix_tokens,
            verifyWidth=canonical.verify_widths,
            draftTokens=canonical.draft_tokens,
            acceptanceRate=float(
                canonical.environment.get("acceptance_rate") or 1.0),
            comm_latency_us=float(
                canonical.environment.get("comm_latency_us") or 0.0),
            launch_floor_us=float(
                canonical.environment.get("launch_floor_us") or 0.0),
            servedOnly=False,
            floorForms=True)

    rows = SERVING_PROJECTION.get_or_compute(
        target.projection_identity(recordDigest=recordDigest).key, project)

    contextAt = {axis: index for index, axis in enumerate(contexts)}
    batchAt = {axis: index for index, axis in enumerate(batches)}
    # `private_copy` keeps a caller that folds or narrows a point (the Gallery
    # pops a floor form) from writing into the row every other surface reads.
    out = private_copy([[
        rows[contextAt[axis]][batchAt[batch]]
        for batch in zip(request.decode_batches, request.prefill_batches,
                         strict=True)
    ] for axis in zip(
        request.contexts, request.prefix_tokens, request.verify_widths,
        strict=True)])
    if floorForms:
        elapsed = time.perf_counter() - started
        log_timing(
            "projection", model=request.walk.model, gpu=request.card,
            layout=request.layout, contexts=len(request.contexts),
            batches=len(request.decode_batches), floor_forms=True,
            public_envelope=publicEnvelope, elapsed_s=elapsed)
        return out
    for row in out:
        for point in row:
            phases = point.get("phases") if point is not None else None
            decode = phases.get("decode") if isinstance(phases, dict) else None
            if isinstance(decode, dict):
                decode.pop("floors", None)
    elapsed = time.perf_counter() - started
    log_timing(
        "projection", model=request.walk.model, gpu=request.card,
        layout=request.layout, contexts=len(request.contexts),
        batches=len(request.decode_batches), floor_forms=False,
        public_envelope=publicEnvelope, elapsed_s=elapsed)
    return out
