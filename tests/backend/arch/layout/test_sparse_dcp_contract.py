import pytest
import sympy as sp

from backend.arch.eval.dim import Dim, value_of
from backend.arch.layout.sparse_dcp_contract import (
    CandidateExchange, OutputExchange, SparseDcpWorkload,
)


def test_selected_valid_is_not_padded_and_skew_is_explicit():
    mean = SparseDcpWorkload(2, 16384, 2048, 8, 'balanced_mean')
    peak = SparseDcpWorkload(2, 16384, 2048, 8, 'busiest_rank', 8)
    trace = SparseDcpWorkload(2, 16384, 2048, 8, 'trace',
                              trace_counts=(2048, 0, 0, 0, 0, 0, 0, 0))
    assert mean.selected_positions == 512
    assert peak.selected_positions == trace.selected_positions == 4096
    assert mean.padded_positions == peak.padded_positions == 4096
    assert mean.local_index_scan_positions == 4096
    with pytest.raises(ValueError, match='conserve'):
        SparseDcpWorkload(1, 16, 16, 8, 'trace', trace_counts=(1,) * 8)
    with pytest.raises(ValueError, match='trace count'):
        SparseDcpWorkload(1, 16, 16, 8, 'trace', trace_counts=(16,) + (0,) * 7)


def test_candidate_recipient_injection_fallback_and_residency_are_distinct():
    fast = CandidateExchange(1, 2048, 8, failed_owners=0)
    fallback = CandidateExchange(1, 2048, 8, failed_owners=2)
    assert fast.multicast_injected_bytes == 3072
    assert fast.multicast_recipient_bytes == 21504
    assert fast.fallback_max_send_bytes == 0
    assert fallback.fallback_mean_receive_bytes == 28672
    assert fallback.fallback_max_receive_bytes == 32768
    assert fallback.fallback_max_send_bytes == 114688
    assert fast.resident_bytes(1) == fallback.resident_bytes(1) == 213068


def test_output_path_lse_padding_and_projection():
    assert OutputExchange(1, 96, 8, 128, 'push').remote_bytes == 22848
    assert OutputExchange(1, 96, 8, 128, 'direct').remote_bytes == 22848
    assert OutputExchange(1, 96, 8, 128, 'flashinfer').remote_bytes == 22176
    assert OutputExchange(1, 96, 8, 512, 'packed').remote_bytes == 86352
    with pytest.raises(ValueError, match='128-wide'):
        OutputExchange(1, 96, 8, 512, 'push')


def test_symbolic_extents_preserve_dim_identity_and_values():
    b = sp.Symbol('batch', integer=True, positive=True)
    s = sp.Symbol('context', integer=True, positive=True)
    work = SparseDcpWorkload(Dim(b, b), Dim(s, s), 2048, 8, 'balanced_mean')
    assert value_of(work.selected_positions) == b * sp.Min(2048, s) / 8
    assert value_of(work.local_index_scan_positions) == b * sp.ceiling(s / 8)
    out = OutputExchange(Dim(b, b), Dim(sp.Symbol('n_heads'), 96), 8, 128, 'push')
    assert value_of(out.remote_bytes) == 22848 * b
    assert sp.Symbol('n_heads') in out.remote_bytes.expr.free_symbols


def test_local_combine_counts_arithmetic_and_successful_slot_traffic():
    from backend.arch.layout.sparse_dcp_contract import LocalAttentionCombine
    push = LocalAttentionCombine(1, 96, 8, 128, 16)
    assert push.flops == 12 * (17 * 128 + 31)
    assert push.read_bytes == 96 * 260 + 22848
    assert push.write_bytes == 2 * 22848 + 12 * 256
    fi = LocalAttentionCombine(1, 96, 8, 128, 8, clears_slots=0)
    assert fi.write_bytes == 22176 + 12 * 256
