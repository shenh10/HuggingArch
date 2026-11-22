"""The sparse DCP candidate fallback is a second collective row, priced by the
envelope — not a global `direct_exchange` algorithm or a latency multiplier
patched into the roofline."""
import sympy as sp

from backend.arch.layout.comm_volume import NCCL, ALGORITHMS, algorithms_for
from backend.inference.row_roofline import (
    COMM_LATENCY_S, COMM_S_PER_BYTE, build_row_exprs,
)


def test_all_gather_and_all_to_all_price_only_nccl():
    """main dropped `direct_exchange` from `ALGORITHMS`; the candidate
    multicast, its fallback and the projected combine are therefore priced by
    NCCL's envelope alone, so no unrelated model's NVLink collective moves."""
    assert algorithms_for("all_gather") == (NCCL,)
    assert algorithms_for("all_to_all") == (NCCL,)
    assert "direct_exchange" not in ALGORITHMS


def test_fallback_round_adds_a_second_collective_latency():
    """Two all_gather rows — the candidate multicast and the dependent
    fallback — are two sequential rounds: their latencies add, their payloads
    are priced apart. This is the roofline reading of `sparse_dcp_comm`
    emitting the fallback as its own event chained off the multicast."""
    base = build_row_exprs(count=1, flops=0, act_in=0, act_out=0, demand=0,
                           weight_total=0, metadata_total=0,
                           comm_payload=21504 * 8 // 7, comm_priced=True)
    fallback = build_row_exprs(count=1, flops=0, act_in=0, act_out=0, demand=0,
                               weight_total=0, metadata_total=0,
                               comm_payload=114688 * 8 // 7, comm_priced=True)
    both = sp.expand(base["t_comm"] + fallback["t_comm"])
    assert both == (24576 + 131072) * COMM_S_PER_BYTE + 2 * COMM_LATENCY_S
    # With no failed owner the fallback payload is zero: one round remains.
    inactive = build_row_exprs(count=1, flops=0, act_in=0, act_out=0, demand=0,
                               weight_total=0, metadata_total=0, comm_payload=0,
                               comm_active=0, comm_priced=True)
    none = sp.expand(base["t_comm"] + inactive["t_comm"])
    assert none == 24576 * COMM_S_PER_BYTE + COMM_LATENCY_S


def test_scheduler_admission_uses_the_requested_batch_and_symbolic_context():
    from backend.arch.runtime_laws import SparseWindow
    from backend.inference.sol_law import LawGraph, _windowed_batch_admission

    batch, context = sp.symbols("B s_input", positive=True, integer=True)
    parts = sp.Piecewise((74, batch < 74), (148, True))
    graph = LawGraph.of({"term_0": SparseWindow(0, batch, parts, context)})
    admitted = _windowed_batch_admission(graph)
    assert [admitted(n) for n in (1, 73, 74, 2220, 2221)] == [
        True, True, True, True, False,
    ]
