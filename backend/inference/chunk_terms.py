"""Exact-key launch terms used to fold chunked prefill measurements.

This module is deliberately owned by the inference backend: the fold reads
estimator rows, while Kernel Bench is only one producer of the measurements it
selects.  It states no model or campaign policy and never imports Kernel Bench.

TODO(H3 deferred measurement ledgers): exact-key folding alone does not prove
framework state semantics.  Later batches must test recurrent/SSM pending tail
state, compressor global cache-group cadence, and indexer raw/pooled ``s_kv``.
Projected SWA/top-k ``s_kv`` is covered by framework-bound chunk-fold tests;
the still-missing state/indexer provenance is named here so it cannot silently
hide behind an exact-key sum.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ChunkLaunchTerm:
    """One endpoint-chunk term considered for an exact-key launch fold.

    The fields beside ``launches`` are precisely the ones that make
    ``Σ w_i n_i T_i`` valid: a term may be added only when every one of them
    agrees.  The key is deliberately opaque (physical-key fields or an
    equivalent exact key); this module does not invent a second shape identity.
    """

    key: Tuple[Any, ...]
    launches: float
    formula_weight: Any
    zero_cost: bool = False
    inputs: Any = None
    context: Any = None
    cadence: Any = 1
    #: The term's SOL over its `launches`, one occurrence before the
    #: formula's weight (`PointMeasure.sol` times the chunk's repetition),
    #: or None when the chunk's measure states none. Summed with the
    #: launches it rides: like them, a count of the same law.
    sol: Optional[float] = None


def _same_chunk_term_semantics(left: ChunkLaunchTerm,
                               right: ChunkLaunchTerm) -> bool:
    """Whether two exact-key terms describe the same launch law.

    ``launches`` is intentionally absent: it is the count being folded.  The
    formula weight, zero-cost state, selection inputs, selection context and
    cadence/context semantics must all be equal.  In particular a zero-cost
    term must never be absorbed into a looked-up term (or vice versa), and a
    different pipeline/draft coefficient is a different mathematical term even
    when its physical key is the same.
    """
    return (left.formula_weight == right.formula_weight
            and left.zero_cost == right.zero_cost
            and left.inputs == right.inputs
            and left.context == right.context
            and left.cadence == right.cadence)


def merge_chunk_launch_terms(
        terms: Iterable[ChunkLaunchTerm]) -> List[ChunkLaunchTerm]:
    """Merge same-key chunk launches only when their fold law is invariant.

    Safe groups become one term whose ``launches`` is the sum.  An unsafe group
    is returned term-for-term: lookup can still deduplicate its key, but the
    fold keeps every distinct weight/input/cadence contribution.  No effective
    launch weight is invented here; constructing one is an explicit fold
    operation and must divide by a common stated weight.
    """
    out: List[ChunkLaunchTerm] = []
    groups: Dict[Tuple[Any, ...], List[ChunkLaunchTerm]] = {}
    for term in terms:
        if term.launches < 0:
            raise ValueError(
                "a chunk launch count cannot be negative "
                f"(got {term.launches!r})")
        groups.setdefault(term.key, []).append(term)
    for key in groups:
        group = groups[key]
        first = group[0]
        if any(not _same_chunk_term_semantics(first, other)
               for other in group[1:]):
            out.extend(group)
            continue
        out.append(ChunkLaunchTerm(
            key=key,
            launches=sum(term.launches for term in group),
            sol=(None if any(term.sol is None for term in group)
                 else sum(term.sol for term in group)),
            formula_weight=first.formula_weight,
            zero_cost=first.zero_cost,
            inputs=first.inputs,
            context=first.context,
            cadence=first.cadence,
        ))
    return out


def chunk_term_count(terms: Iterable[ChunkLaunchTerm]) -> Dict[str, int]:
    """The small H4-facing census of a merge-ready chunk term list.

    Distinct keys are exact physical keys, not endpoint repetitions.  This pure
    count is intentionally separate from sweep stats (which describe a planned
    campaign) so a dry run can aggregate both without planning a full corpus by
    accident.
    """
    materialized = list(terms)
    return {
        "terms": len(materialized),
        "distinct_keys": len({term.key for term in materialized}),
        "zero_cost_terms": sum(term.zero_cost for term in materialized),
    }
