"""A fallback peer fetch depends on the candidate certificate, within one launch."""
import sympy as sp

from backend.inference.row_roofline import (
    COMM_LATENCY_S, COMM_S_PER_BYTE, build_row_exprs, rung_symbols,
)


def test_fallback_adds_transfer_dependency_not_a_second_launch():
    f = sp.Symbol('failed', integer=True, nonnegative=True)
    rounds = sp.Piecewise((2, f > 0), (1, True))
    laws = build_row_exprs(count=3, flops=0, act_in=0, act_out=0,
                          demand=0, weight_total=0, metadata_total=0,
                          comm_payload=24576, comm_factor=7/8,
                          comm_priced=True, comm_dependency_rounds=rounds)
    fast = laws['t_comm'].subs(f, 0)
    slow = laws['t_comm'].subs(f, 2)
    assert sp.simplify(slow - fast) == COMM_LATENCY_S
    assert fast == 24576 * COMM_S_PER_BYTE + COMM_LATENCY_S
    assert laws['t_launch'] == 0


def test_dependency_rounds_also_scale_direct_exchange_rung_latency():
    laws = build_row_exprs(count=1, flops=0, act_in=0, act_out=0,
                          demand=0, weight_total=0, metadata_total=0,
                          comm_payload=24576, comm_factor=7/8,
                          comm_priced=True, comm_rungs=1, comm_dependency_rounds=2)
    bound, rate, fixed = rung_symbols(0)
    value = laws['t_comm'].subs({bound: 100000, rate: 0, fixed: 3,
                                COMM_LATENCY_S: 20, COMM_S_PER_BYTE: 0})
    assert value == 6


def test_fallback_phase_breakdown_matches_total_and_legacy_wire_defaults():
    import pytest
    from backend.inference.comm_cost import CollectiveRung, Fabric
    from backend.inference.row_roofline import LawBook, RowCard, RowRoofline, law_columns
    book = LawBook()
    fabric = Fabric(8, 1e12, None, step_latency=1e-6, nvlink=True)
    row = {'is_comm': True, 'comm_op': 'all_gather', 'comm_world': 8,
           'count_per_step': 3, 'comm_base_stride': 1,
           'comm_rungs': [{'algorithm': 'direct_exchange', 'worlds': [8]}],
           **law_columns(build_row_exprs(count=3, flops=0, act_in=0, act_out=0,
                                        demand=0, weight_total=0, metadata_total=0,
                                        comm_payload=155648, comm_factor=7/8,
                                        comm_priced=True, comm_rungs=1,
                                        comm_dependency_rounds=2), book)}
    point = {'B': 1, 's_input': 1, 's_output': 0, 's_prefix': 0, 's_verify': 1, 'rho': 1}
    at = book.row_at(row, point, RowCard(None, None, fabric))
    total, phases = RowRoofline.comm(op='all_gather', world=8, tensor_bytes=155648,
                                     count=3, base_stride=1, fabric=fabric,
                                     rungs=(CollectiveRung('direct_exchange', worlds=(8,)),),
                                     dependency_rounds=at.comm_dependency_rounds)
    assert total == pytest.approx(at.times.t_comm)
    assert sum(p['t_phase_s'] for p in phases) == pytest.approx(total)
    assert phases[0]['hops'] == 2
    assert phases[0]['latency_s'] == pytest.approx(6e-6)
    legacy = {**row, 'roofline_expr': dict(row['roofline_expr'])}
    del legacy['roofline_expr']['comm_dependency_rounds']
    assert book.row_at(legacy, point, RowCard(None, None, fabric)).comm_dependency_rounds == 1
