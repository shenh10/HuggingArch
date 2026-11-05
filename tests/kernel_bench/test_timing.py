"""A timed launch pays for what it writes, not only for what it reads.

L2 is write-back: a launch's stores sit dirty in L2 when it finishes and
reach DRAM later, when something evicts them. A window that closes at the
end of the launch therefore leaves out the write-back of any output that
fits in L2. The harness used to flush with a memset, which leaves L2 full of
dirty lines; a launch then paid to write back one flush line for every line
it touched — reads too — while its own writes went unpaid. The result depended on
the L2 state at the start event: one Kernel Bench
cell (vLLM's DeepSeek-V4 un-rotation, 208 tokens, RTX 4090) read 0.60 of
DRAM peak under it on an idle card and 1.10 in the corpus; this method
reads 0.78 either way.

A read-only pass and a write-only pass over the same bytes move the same
DRAM traffic in opposite directions, so a timing method that charges each
launch for its own traffic times them alike. The memset flush timed the
16 MiB read 1.88x the 16 MiB write on an RTX 4090 (reads charged the
flush's write-back, writes charged nothing); with the write-back inside the
window the ratio is 0.83 (2026-10-25).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")


def test_a_write_costs_what_a_read_of_the_same_bytes_costs():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from kernel_bench.timing import time_callable
    # 16 MiB: inside L2 on every card the bench runs on (50 MiB H100, 72 MiB
    # RTX 4090), large enough that launch overhead does not dominate.
    n = 16 * 2**20 // 4
    src = torch.randn(n, device="cuda")
    dst = torch.empty(n, device="cuda")
    sink = torch.empty((), device="cuda")
    read = time_callable(lambda: torch.amax(src, dim=0, out=sink)).p50_us
    write = time_callable(lambda: dst.fill_(1.0)).p50_us
    assert 0.6 <= read / write <= 1.4, (
        f"read {read:.1f} us vs write {write:.1f} us of the same 16 MiB: the "
        f"window charges one direction for traffic it does not move")


def test_the_clock_names_itself():
    # The row's `timing_method` is whatever the clock returns, so each branch
    # must return its own name — the default the drain, `flush_l2=False` not.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from kernel_bench.timing import time_callable
    x = torch.randn(1024, device="cuda")
    assert time_callable(lambda: x.mul_(1.0), iters=5).timing_method == "l2_drain"
    assert time_callable(lambda: x.mul_(1.0), iters=5,
                         flush_l2=False).timing_method == "warm_l2"


def test_unflushed_timing_still_reads_a_warm_cache():
    # `flush_l2=False` keeps its meaning: no sweep, inputs stay in L2.
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from kernel_bench.timing import time_callable
    # Reduce independent rows, not the entire tensor to one scalar. A
    # global reduction adds cross-block coordination that can hide the
    # memory-access difference (16 MiB on B300: cold/hot ~1.03). Row-wise
    # reduction reads the same bytes with only a small output, keeping this
    # a cache-read check rather than a reduction-latency check.
    x = torch.randn(16 * 2**20 // 4 // 1024, 1024, device="cuda")
    sink = torch.empty(x.shape[0], device="cuda")
    hot = time_callable(lambda: torch.amax(x, dim=1, out=sink), flush_l2=False)
    cold = time_callable(lambda: torch.amax(x, dim=1, out=sink))
    assert cold.p50_us >= hot.p50_us * 1.1, (hot.p50_us, cold.p50_us)
