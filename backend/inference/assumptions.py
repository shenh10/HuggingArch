"""Modelling assumptions a reader cannot see from the numbers.

Most of what this estimate asserts is derivable: a shape came from the spec, a
byte count came from a dtype, a bandwidth came from the SKU table. This module
is for the rest — places where the model had to assume something the inputs do
not determine, and where the assumption is not visibly an assumption because it
has a plausible default that produces a plausible number.

The distinction that matters is between a KNOB and an ASSUMPTION. A knob left
at its default is fine when the default is the neutral element and the reader
can see it in the parallel block. It becomes an assumption the moment the
default is a claim about the world: `ep_load_factor = 1.0` does not mean "no
imbalance modelling"; it means "every expert receives exactly the mean number
of tokens", which is false of every routing trace anyone has published. The
number that comes out is not neutral, it is optimistic, and nothing on the
response said so.

`feedback_mfu_no_defaults` is why the fix is not to invent a number. Picking
1.2 because a paper reported 1.2 would put a scenario constant in the model
and make it look like a measurement. The honest step is the one taken here:
the deployment says out loud what it assumed, names the knob that would
replace the assumption with a measurement, and leaves the value alone.

An entry is emitted only when the assumption BITES — when the model routes,
when the deployment actually spreads experts across ranks, and when nobody
supplied a trace. A dense model at `ep_moe=1` is not assuming anything about
routing balance, and telling it that it is would train readers to skip the
list.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from .parallel_config import ParallelConfig



def deployment_assumptions(
    parallel: ParallelConfig,
    *,
    ep_load_factor: float = 1.0,
    has_dispatch_zone: bool = False,
    joined_streams: Sequence[str] = (),
    sparse_dcp: bool = False,
    dcp_selected_load_factor: float = 1,
    dcp_failed_owners: int = 0,
) -> List[Dict[str, Any]]:
    """What this deployment assumed that its inputs did not determine.

    ``has_dispatch_zone`` comes from the same UID set that the timing fold
    scales by the straggler factor. It is passed in rather than re-derived from
    the spec: a note whose condition is "this model looks MoE" and arithmetic
    whose condition is "this leaf is in a dispatch zone" are two predicates
    that agree until they do not.

    Returns a list of entries, each with:

      ``id``      stable key, for a consumer that wants to render one specially
      ``knob``    the deployment field that turns the assumption into an input
      ``value``   what that field is currently set to
      ``effect``  which direction the estimate errs while the assumption holds
      ``text``    the sentence to show

    ``joined_streams`` are the streams some reader joins whole
    (`Program.joined_streams`): each element is held from the layer that
    appends it to the join, and the activation peak does not count it
    across the blocks in between.

    Empty is the common case and the honest one: a dense model on one GPU is
    not assuming anything on this list.
    """
    out: List[Dict[str, Any]] = []

    ep = max(1, int(parallel.ep_moe or 1))
    rho = float(ep_load_factor)
    if ep > 1 and rho <= 1.0 and has_dispatch_zone:
        out.append({
            "id": "moe_routing_balance",
            "knob": "ep_load_factor",
            "value": rho,
            "effect": "optimistic",
            "text": (
                f"Expert routing is assumed perfectly balanced: every one of "
                f"the {ep} expert-parallel ranks receives exactly the mean "
                f"token load. Real routing is skewed, and under synchronous "
                f"EP the layer's wall clock is the busiest rank's, so the "
                f"MoE step times and the all-to-all here are a lower bound "
                f"rather than an estimate. Set `ep_load_factor` to the "
                f"max/mean expert load from a routing trace to price it (the "
                f"busiest rank's load follows from it and the experts each "
                f"rank holds); the model carries no default for it, because "
                f"a number nobody measured would read like one somebody did."
            ),
        })

    if sparse_dcp:
        out.extend([
            {
                "id": "sparse_dcp_selected_ownership",
                "knob": "dcp_selected_load_factor",
                "value": float(dcp_selected_load_factor),
                "effect": "optimistic" if dcp_selected_load_factor == 1 else "scenario",
                "text": "Selected-position max/mean ratio is an explicit scenario, not a measured distribution. "
                        "The busiest rank is capped by its striped resident context; padded top-k slots remain 2048.",
            },
            {
                "id": "sparse_dcp_candidate_fallback",
                "knob": "dcp_failed_owners",
                "value": int(dcp_failed_owners),
                "effect": "optimistic" if dcp_failed_owners == 0 else "scenario",
                "text": "Failed candidate-certificate owners per query are assumed, not observed. "
                        "Fallback prices the sender hotspot and a second dependent exchange stage; "
                        "polling and transport framing are excluded.",
            },
            {
                "id": "sparse_dcp_runtime_profile",
                "knob": None,
                "value": "sglang_fi_a2a_push1536",
                "effect": "scope",
                "text": "TP8/DCP8, replicated Q, BF16 projected partials, prewarmed V projection, "
                        "no LoRA, default 384-candidate merge and registered push capacity 1536 query rows. "
                        "This is target-step timing, not DFLASH acceptance or server scheduling. "
                        "Process-wide communication workspaces and CUDA graph pools are outside resident weight accounting.",
            },
        ])

    if joined_streams:
        names = ", ".join(f"`{name}`" for name in joined_streams)
        out.append({
            "id": "joined_stream_residency",
            "knob": None,
            "value": None,
            "effect": "optimistic",
            "text": (
                f"The activation peak leaves out the elements of {names} "
                f"between the layer that appends each and the join that "
                f"reads them: they stay resident across the blocks in "
                f"between, and the peak counts them only inside the block "
                f"that joins them. The join itself is priced like any "
                f"concat."
            ),
        })
    return out
