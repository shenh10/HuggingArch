"""The neutral identity shared by every SOL walk producer and consumer.

The pool stores walk facts, not a page payload. This module is the one place
that names those facts: a walk belongs to a model/engine version, a layout, a
card class, a precision profile, and the workload anchor its rows were taken
at. Gallery, Scaling, and Compare may project such a record; none of them owns
its spelling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from backend.arch.eval.precision import AS_SHIPPED
from backend.inference.card import Card
from backend.inference.defaults import GPU_MEMORY_FRACTION_DEFAULT
from backend.inference.sol_pool import defaultPool, recordId
from backend.inference.sweep_grid import (
    GALLERY_OUTPUT_TOKENS,
    ROUTINE_SEQ_LENS,
)
from backend.measure.scope import ResultIdentity, of_walk


#: The schema of a canonical law record and of the bundle manifest that lists
#: it — one name for both. v2 named the card it was walked on (`walked_on`)
#: and made a bundle one walk class's (`ServingWalkKey`). v3 made a record a
#: real CSE graph. v4 carries the serving capacity view and the effective
#: parallel config beside that graph. v7 names each step term in the graph
#: (`term_<k>`): the decode floor form is kept per term, so a reader can
#: swap a measured term out of it (`auto_sweep.blend`).
walkRecordSchema = "serving-law-v7"
servingContexts = tuple(ROUTINE_SEQ_LENS)
defaultPrefillChunkSize = 8192

# The capacity row is not part of the public decode ladder. It binds both
# phases at each card's own ceiling, which is what a deployment-scale sweep's
# default point reads.
capacityBatchPoint = {
    "key": "capacity",
    "prefill_batch_size": -1,
    "decode_batch_size": None,
}


@dataclass(frozen=True)
class ServingWalkFacts:
    """The workload facts a canonical serving walk is taken at.

    Floors, selection mode, measurement scope, and a card's own HBM are
    deliberately absent: they are read when a record is projected, not when
    it is walked. The two JSON views are the existing artifact spellings:
    a manifest's ``facts`` block, and the walk identity's ``anchor``.
    """
    output_tokens: int
    gpu_memory_fraction: float
    precision: str
    prefix_cached_tokens: int
    verify_width: int
    acceptance_rate: float
    speculative_draft: str | None
    include_aux_weights: bool
    host_offload: bool
    framework: str
    framework_options: tuple[tuple[str, Any], ...]
    kv_block_size: int | None
    prefill_chunk_size: int | None
    expert_load_factor: float

    @classmethod
    def producer(cls) -> "ServingWalkFacts":
        return cls(
            output_tokens=GALLERY_OUTPUT_TOKENS,
            gpu_memory_fraction=GPU_MEMORY_FRACTION_DEFAULT,
            precision=AS_SHIPPED.key,
            prefix_cached_tokens=0,
            verify_width=1,
            acceptance_rate=1.0,
            speculative_draft=None,
            include_aux_weights=False,
            host_offload=True,
            framework="",
            framework_options=(),
            kv_block_size=None,
            prefill_chunk_size=defaultPrefillChunkSize,
            expert_load_factor=1.0,
        )

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "ServingWalkFacts":
        """Facts read back from an artifact, refusing a partial spelling."""
        required = {
            "outputTokens", "gpuMemoryFraction", "precision",
            "prefixCachedTokens", "verifyWidth", "acceptanceRate",
            "speculativeDraft", "includeAuxWeights", "hostOffload",
            "framework", "frameworkOptions", "kvBlockSize",
            "prefillChunkSize", "expertLoadFactor",
        }
        if set(value) != required:
            raise ValueError("serving walk facts are not the record vocabulary")
        block_size = value["kvBlockSize"]
        chunk_size = value["prefillChunkSize"]
        return cls(
            output_tokens=int(value["outputTokens"]),
            gpu_memory_fraction=float(value["gpuMemoryFraction"]),
            precision=str(value["precision"]),
            prefix_cached_tokens=int(value["prefixCachedTokens"]),
            verify_width=int(value["verifyWidth"]),
            acceptance_rate=float(value["acceptanceRate"]),
            speculative_draft=value["speculativeDraft"],
            include_aux_weights=bool(value["includeAuxWeights"]),
            host_offload=bool(value["hostOffload"]),
            framework=str(value["framework"]),
            framework_options=tuple(sorted(
                (str(name), option)
                for name, option in dict(value["frameworkOptions"]).items()
            )),
            kv_block_size=None if block_size is None else int(block_size),
            prefill_chunk_size=(None if chunk_size is None
                                else int(chunk_size)),
            expert_load_factor=float(value["expertLoadFactor"]),
        )

    @classmethod
    def from_environment(cls, environment: Mapping[str, Any]) -> "ServingWalkFacts":
        precision = environment.get("precision")
        block_size = environment.get("kv_block_size")
        chunk_size = environment.get("prefill_chunk_size")
        return cls(
            output_tokens=int(environment["avg_output_tokens"]),
            gpu_memory_fraction=float(environment["gpu_memory_fraction"]),
            precision=str(getattr(precision, "key", precision)),
            prefix_cached_tokens=int(environment.get("s_prefix_cached") or 0),
            verify_width=int(environment.get("s_verify") or 1),
            acceptance_rate=float(environment.get("acceptance_rate") or 1.0),
            speculative_draft=environment.get("speculative_draft"),
            include_aux_weights=bool(environment.get("include_aux_weights", False)),
            host_offload=bool(environment.get("host_offload", True)),
            framework=str(environment.get("framework") or ""),
            framework_options=tuple(sorted(
                (str(name), value)
                for name, value in (environment.get("framework_options") or {}).items()
            )),
            kv_block_size=None if block_size is None else int(block_size),
            prefill_chunk_size=(None if chunk_size is None
                                else int(chunk_size)),
            expert_load_factor=float(environment.get("ep_load_factor") or 1.0),
        )

    def to_environment(self) -> dict[str, Any]:
        """The request-side spelling of these facts.

        Floors are deliberately absent: they bind a projection, not a walk.
        `from_environment(to_environment(facts))` is the identity, so a
        consumer can ask one owner both what the producer walked and what a
        request asked for.
        """
        return {
            "avg_output_tokens": self.output_tokens,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "precision": self.precision,
            "s_prefix_cached": self.prefix_cached_tokens,
            "s_verify": self.verify_width,
            "acceptance_rate": self.acceptance_rate,
            "speculative_draft": self.speculative_draft,
            "include_aux_weights": self.include_aux_weights,
            "host_offload": self.host_offload,
            "framework": self.framework,
            "framework_options": dict(self.framework_options),
            "kv_block_size": self.kv_block_size,
            "prefill_chunk_size": self.prefill_chunk_size,
            "ep_load_factor": self.expert_load_factor,
        }

    def to_json(self) -> dict[str, Any]:
        """The manifest's facts block (its schema is the manifest's own)."""
        return {
            "outputTokens": self.output_tokens,
            "gpuMemoryFraction": self.gpu_memory_fraction,
            "precision": self.precision,
            "prefixCachedTokens": self.prefix_cached_tokens,
            "verifyWidth": self.verify_width,
            "acceptanceRate": self.acceptance_rate,
            "speculativeDraft": self.speculative_draft,
            "includeAuxWeights": self.include_aux_weights,
            "hostOffload": self.host_offload,
            "framework": self.framework,
            "frameworkOptions": dict(self.framework_options),
            "kvBlockSize": self.kv_block_size,
            "prefillChunkSize": self.prefill_chunk_size,
            "expertLoadFactor": self.expert_load_factor,
        }

    def to_anchor(self) -> dict[str, Any]:
        """The walk identity's anchor: a record kind and schema beside them."""
        return {"kind": "serving-law", "schema": walkRecordSchema,
                **self.to_json()}


@dataclass(frozen=True)
class ServingWalkKey:
    """The identity owner of one card class's serving-law bundle.

    A card enters here as its walk class and nowhere else in a projection
    key. The facts and precision are checked against each other because the
    existing artifact carries precision in both places; a caller cannot
    assemble two spellings that disagree.
    """
    version: str
    model: str
    card_class: str
    facts: ServingWalkFacts

    @property
    def precision(self) -> str:
        """The walk's precision, owned by its facts and never restated."""
        return self.facts.precision

    @classmethod
    def of_card(cls, version: str, *, model: str, card: Card,
                facts: ServingWalkFacts | None = None) -> "ServingWalkKey":
        return cls(
            version=version,
            model=model,
            card_class=card.klass.key(),
            facts=(facts if facts is not None
                   else ServingWalkFacts.producer()),
        )

    @property
    def identity(self) -> ResultIdentity:
        return of_walk(
            self.version,
            model=self.model,
            layout=None,
            card_class=self.card_class,
            precision=self.precision,
            pp_stage=None,
            anchor=self.facts.to_anchor())


@dataclass(frozen=True)
class StepLawKey:
    """One layout's law record: its bundle's walk key plus its layout.

    The layout is held in its canonical serialized form — a deployment and
    its dict round-trip to the same key, and a mapping field's order is not
    a fact of the deployment. Its hashes are derived, never stored: the
    record name at the pool's edge, the identity when a cache wants one.
    """
    walk: ServingWalkKey
    layout_json: str
    schema: str = walkRecordSchema

    @classmethod
    def of_layout(cls, walk: ServingWalkKey, layout: Any) -> "StepLawKey":
        encoded = layout.to_dict() if hasattr(layout, "to_dict") else layout
        return cls(
            walk=walk,
            layout_json=json.dumps(
                encoded, sort_keys=True, separators=(",", ":")))

    @property
    def layout(self) -> dict[str, Any]:
        return json.loads(self.layout_json)

    @property
    def record_name(self) -> str:
        """The record's stable name inside its bundle (`SolPool`)."""
        return recordId("layout", self.layout)

    @property
    def bundle_key(self) -> str:
        """The bundle's cache key — the walk identity's, read once here so
        callers do not re-spell the derivation."""
        return self.walk.identity.key

    @property
    def identity(self) -> ResultIdentity:
        """One law record's own identity: its walk's facts with its layout
        bound, not a re-implementation of either half."""
        return of_walk(
            self.walk.version,
            model=self.walk.model,
            layout=self.layout,
            card_class=self.walk.card_class,
            precision=self.walk.precision,
            pp_stage=None,
            anchor={"kind": "serving-law-step", "schema": self.schema,
                    **self.walk.facts.to_json()})


def walkManifestIsValid(manifest: Mapping[str, Any] | None) -> bool:
    """Whether a manifest really is the current canonical walk record."""
    if manifest is None:
        return False
    metadata = manifest.get("metadata") or {}
    try:
        facts = ServingWalkFacts.from_record(metadata.get("facts") or {})
    except (KeyError, TypeError, ValueError):
        return False
    return (
        metadata.get("recordSchema") == walkRecordSchema
        and facts == ServingWalkFacts.producer()
        and bool(metadata.get("layoutsByCard"))
    )


# One process reads and writes one pool. Distinct wrappers around the same
# directory would still be correct on disk, but they would split the hot record
# cache and make the data layer harder to reason about.
sharedSolPool = defaultPool()
