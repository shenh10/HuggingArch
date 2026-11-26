"""Kernel timing. torch-guarded so the module imports on a CPU box (for
`plan` + tests). Uses CUDA events on GPU; falls back to a wall-clock loop on
CPU so CPU-runnable ops (e.g. the torch GEMM baseline) can be measured too."""
from __future__ import annotations

from typing import Callable, List, Optional, Protocol

from .providers import LatencyStats


class RankGroup(Protocol):
    """What the clock needs from a group of processes timing one collective
    together (`kernel_bench.comm`): every rank runs this clock at the same
    moment, and these three calls are where they meet."""

    def agree(self, value: float) -> float:
        """The max of `value` over the ranks — the sizing decisions are made
        from it, so every rank makes the same number of calls."""

    def align(self) -> None:
        """Queue a meeting point on the GPU (a 1-element all-reduce): what
        follows starts on every rank at once."""

    def slowest(self, samples: List[float]) -> List[float]:
        """Each sample's max over the ranks: a collective has finished when
        its last rank has."""


# Wall-clock a single (provider, shape) measurement may take. `iters` samples
# of a microsecond kernel cost nothing, but the grid also contains cells that
# run for seconds apiece, and 75 of those is an hour on one cell with no output
# — a sweep looks hung while it is merely being thorough. Past this, the sample
# count is cut to whatever fits, and a cell too slow for even a few samples is
# declined rather than left running.
TIME_BUDGET_S: float = 30.0


class TooSlowToMeasure(RuntimeError):
    """A single launch costs more than the whole budget for this cell."""


def time_callable(
    call: Callable[[], object],
    *,
    warmup: int = 15,
    iters: int = 60,
    flush_l2: bool = True,
    budget_s: float = TIME_BUDGET_S,
    group: Optional[RankGroup] = None,
) -> LatencyStats:
    """Time `call`, reducing per-launch samples to p50/p95/mean microseconds.
    One sample per iteration so p95 surfaces tail / clock-throttle, not just a
    batched average. CUDA-event timed on GPU; `perf_counter` on CPU.

    `flush_l2` (GPU only, default on): each sample sees cold L2 and pays for
    every byte it moves through DRAM, reads AND writes. Without it, the same
    op runs back-to-back with its inputs/weights resident in L2, so
    memory-bound kernels (norms, MoE permute/gather/combine) read from cache
    instead of HBM and report optimistically low latency.

    Reads are the easy half: sweep a 2x-L2 scratch buffer before the start
    event. Writes are the other half. L2 is write-back, so a launch's stores
    sit dirty in L2 when it finishes and reach DRAM only when something later
    evicts them; an end event right after the launch closes the window before
    that, and a launch whose output fits in L2 has its writes timed as free.
    So the sweep is READ-ONLY (it leaves L2 clean: the launch never pays to
    write back someone else's lines), a second sweep inside the window, after
    the launch, evicts the launch's own dirty lines, and that sweep's own cost
    is timed alone on the same clean L2 and subtracted, sample by sample.

    The old flush zeroed the buffer, leaving L2 full of DIRTY lines, so its
    answer depended on the L2 state at the start event. Measured 2026-10-25
    on an RTX 4090 (tools/timing_window_probe.py): vLLM's DeepSeek-V4
    un-rotation at 208 tokens (27.3 MB moved) read 45.1 us, 0.60 of DRAM
    peak, on an idle card under the old flush (each line it touched paid a
    flush line's write-back); from a clean L2 with the window closing at the
    launch it read 18.4 us, 1.47 (its own writes unpaid); the corpus row for
    the same cell, old flush on the bench card, reads 24.6 us, 1.10 — so
    there L2 was at least partly clean at the start event, for a reason not
    identified. Under this method it reads 34.8 us, 0.78, from either
    state, since the sweep fixes the state and the drain bills the writes.

    p95 now also carries the spread of the two sweeps (a stalled host thread
    inside either window moves the difference, both ways), not only the
    launch's own tail.

    `group` (GPU only): this call is one rank of a collective, and every rank
    of it is in this same function now. The sizing reads the slowest rank's
    probe, each window opens after an on-GPU meeting point, and each sample
    is the slowest rank's (`RankGroup`). Without it, nothing changes."""
    import torch
    if not torch.cuda.is_available():
        import time
        for _ in range(warmup):
            call()
        us = []
        for _ in range(iters):
            t0 = time.perf_counter()
            call()
            us.append((time.perf_counter() - t0) * 1e6)
        us.sort()
        n = len(us)
        return LatencyStats(
            p50_us=us[n // 2], p95_us=us[min(n - 1, round(0.95 * (n - 1)))],
            mean_us=sum(us) / n, samples=n, timing_method="wall_clock",
        )

    # One timed warmup call sizes the rest: `iters` samples of a kernel that
    # runs for seconds is an hour on a single cell, and the run shows nothing
    # while it happens. Measured on the H100 sweep, which sat at 100% GPU with
    # flat memory and no log line for 30+ minutes.
    import time as _time
    call()
    torch.cuda.synchronize()
    _t0 = _time.perf_counter()
    call()
    torch.cuda.synchronize()
    per_call = max(_time.perf_counter() - _t0, 1e-9)
    if group is not None:
        per_call = group.agree(per_call)

    remaining = max(1, warmup - 2)
    if per_call * (remaining + iters) > budget_s:
        affordable = int(budget_s / per_call)
        if affordable < 3:
            raise TooSlowToMeasure(
                f"one launch takes {per_call:.1f}s; {budget_s:.0f}s buys "
                f"{affordable} samples, too few for a p50/p95")
        # Spend a fifth of what is left on warmup, the rest on samples.
        remaining = max(1, affordable // 5)
        iters = max(3, affordable - remaining)

    # Host time to issue one call: how long the CPU takes to queue its
    # kernels, read off the warmup launches (no sync inside the measurement).
    host_s = 0.0
    for _ in range(remaining):
        _h0 = _time.perf_counter()
        call()
        host_s = max(host_s, _time.perf_counter() - _h0)
    torch.cuda.synchronize()
    if group is not None:
        host_s = group.agree(host_s)

    sweep = None
    if flush_l2:
        # 2× L2 so one sweep evicts prior residents (fallback 128 MiB when the
        # property is unavailable). torch spells it `L2_cache_size`; keep the
        # lowercase fallback for older builds.
        props = torch.cuda.get_device_properties(0)
        l2_bytes = (getattr(props, "L2_cache_size", 0)
                    or getattr(props, "l2_cache_size", 0) or (128 << 20))
        failed = None
        try:
            scratch = torch.zeros(2 * int(l2_bytes) // 4, dtype=torch.float32,
                                  device="cuda")
            sink = torch.empty((), dtype=torch.float32, device="cuda")
        except Exception as e:
            failed = e
        # In a group, one rank that cannot allocate must not leave the others
        # waiting in a collective it will never join: they all stop here.
        if group is not None and group.agree(1.0 if failed else 0.0) > 0:
            raise failed or RuntimeError(
                "another rank of the group could not allocate its L2 sweep")
        if failed is not None:
            raise failed

        def sweep():
            # A read of every line, at ~0.9 of DRAM peak (a reduction over
            # fp32, not int8: torch's int8 reduction runs at ~5%). Read-only,
            # so the lines it leaves are clean and it writes back whatever
            # was dirty — which is the point of the second one.
            torch.amax(scratch, dim=0, out=sink)

        # The window also queues the drain sweep after the call, so the hold
        # must cover its host time too.
        _h0 = _time.perf_counter()
        sweep()
        host_s += _time.perf_counter() - _h0
        torch.cuda.synchronize()

    # The start event must not fire before the call's kernels are queued. If
    # the GPU drains the flush before the CPU has issued the call, it idles
    # between the two events and the launch path is timed as kernel time:
    # FA3/FA4's Python launch (~90 us) outlasts an H100's ~35 us flush, so a
    # 5 us prefill read 79 us, slower than torch SDPA beside it. A GPU-side
    # spin queued ahead of the start event, longer than the host time, keeps
    # the queue full; the events then bracket GPU work only.
    hold_cycles = _hold_cycles(torch, 2.0 * host_s + 20e-6)

    def window(body, cold=True):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if cold and sweep is not None:
            sweep()
        torch.cuda._sleep(hold_cycles)
        if group is not None:
            group.align()
        start.record()
        body()
        end.record()
        return start, end

    def launch_then_sweep():
        call()
        sweep()

    # A sample is `launch + sweep − sweep alone`: the launch with its own
    # write-back. Each window also carries the events' own gap (~2 us on a
    # 4090), which the old single window kept in every row; the difference
    # cancels it, so an empty window adds it back and a row means what rows
    # before it meant. The three are taken back to back, so a clock step
    # between samples lands on all of them rather than on the difference.
    samples = []
    for _ in range(iters):
        if sweep is None:
            samples.append((window(call),))
        else:
            samples.append((window(launch_then_sweep), window(sweep),
                            window(lambda: None, cold=False)))
    torch.cuda.synchronize()

    def elapsed_us(w):
        return w[0].elapsed_time(w[1]) * 1000.0

    def latency_us(windows):
        if len(windows) == 1:
            return elapsed_us(windows[0])
        launch, alone, empty = (elapsed_us(w) for w in windows)
        return launch - alone + empty

    us = [latency_us(w) for w in samples]
    if group is not None:
        us = group.slowest(us)
    us = sorted(us)
    n = len(us)
    p50 = us[n // 2]
    p95 = us[min(n - 1, round(0.95 * (n - 1)))]
    mean = sum(us) / n
    return LatencyStats(p50_us=p50, p95_us=p95, mean_us=mean, samples=n,
                        timing_method="l2_drain" if sweep is not None else "warm_l2")


_CYCLES_PER_S: list = []


def _hold_cycles(torch, seconds: float) -> int:
    """`torch.cuda._sleep` cycles that keep the GPU busy for `seconds`
    (capped at 50 ms: a call that synchronizes inside itself has a host time
    that includes its own GPU time, and no hold helps it). The spin rate is
    measured once per process rather than read from a nominal clock."""
    if not _CYCLES_PER_S:
        probe = 1_000_000
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        torch.cuda._sleep(probe)          # first launch pays its own setup
        a.record()
        torch.cuda._sleep(probe)
        b.record()
        torch.cuda.synchronize()
        _CYCLES_PER_S.append(probe / max(a.elapsed_time(b) * 1e-3, 1e-9))
    return int(min(seconds, 0.05) * _CYCLES_PER_S[0])
