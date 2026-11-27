"""Kernel timing. torch-guarded so the module imports on a CPU box (for
`plan` + tests). Device-event timed on an accelerator (`device.DeviceBackend`
names which); falls back to a wall-clock loop only when the box has no
accelerator at all, so CPU-runnable ops (e.g. the torch GEMM baseline) can
still be measured.

The fallback is the dangerous edge, so it is a decision and not a default:
a box WITH an accelerator never reaches it. Before the device backend
existed this module asked `torch.cuda.is_available()`, so an Ascend box
answered "no GPU" and every row it produced was a CPU wall-clock number
wearing an accelerator's SKU."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

from .device import DeviceBackend, detect_backend
from .providers import LatencyStats


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
    device: Optional[DeviceBackend] = None,
    align: Optional[Callable[[], None]] = None,
    agree: Optional[Callable[[int, int], Tuple[int, int]]] = None,
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

    `align`, when given, runs once at the start of every sample — before
    the three windows, not inside them. A collective group uses it for
    `dist.barrier(); device.synchronize()` so ranks do not drift from one
    sample into the next; a single process leaves it None and the clock
    is unchanged."""
    import torch
    dev = device if device is not None else detect_backend()
    if dev is None:
        import time
        for _ in range(warmup):
            call()
        us = []
        for _ in range(iters):
            if align is not None:
                align()
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
    dev.synchronize()
    _t0 = _time.perf_counter()
    call()
    dev.synchronize()
    per_call = max(_time.perf_counter() - _t0, 1e-9)

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

    # In a group, the counts above must be the SAME on every rank. Each
    # decided them from its own `per_call`, and the test is a threshold --
    # so two ranks either side of it run different numbers of samples,
    # while `align` makes every sample a collective. Unequal counts mean
    # the faster rank leaves the barrier the slower one is still waiting
    # on, and the group hangs until the reaper kills it: a timeout, not an
    # error, which is the expensive way to find out.
    #
    # One node hid this because sibling ranks on one card measure nearly
    # the same `per_call`. Across a fabric the spread is an order of
    # magnitude wider and the threshold sits inside it.
    #
    # So rank 0's numbers win. Not a fixed constant: a cell whose launch
    # is genuinely slow still gets its budget cut, and the cut is simply
    # agreed rather than computed twice.
    if agree is not None:
        remaining, iters = agree(remaining, iters)

    # Host time to issue one call: how long the CPU takes to queue its
    # kernels, read off the warmup launches (no sync inside the measurement).
    #
    # The MEDIAN, not the max. Without a sync inside the loop the queue fills,
    # and once it is full the enqueue itself blocks until the device drains a
    # slot — so the slowest sample is back-pressure, not launch cost. On CUDA
    # the two are close enough that the max was a safe over-estimate; on an
    # Ascend 910C they differ by two orders of magnitude (a 4096^3 bf16 matmul
    # measured 62 us typical against a 9,595 us worst sample). The max then
    # claimed the host needed ~9.6 ms to issue a call, so nothing could be
    # judged long enough to cover it and every row was marked unheld.
    # The median is what "how long does issuing one call take" asks for, and
    # the hold is still sized at twice it.
    host_samples = []
    for _ in range(remaining):
        _h0 = _time.perf_counter()
        call()
        host_samples.append(_time.perf_counter() - _h0)
    dev.synchronize()
    host_samples.sort()
    host_s = host_samples[len(host_samples) // 2] if host_samples else 0.0

    sweep = None
    if flush_l2:
        # 2× L2 so one sweep evicts prior residents (fallback 128 MiB when the
        # property is unavailable). torch spells it `L2_cache_size`; keep the
        # lowercase fallback for older builds.
        # Read off the device, never a table (`DeviceBackend.l2_bytes`);
        # the 128 MiB fallback is for a device that will not say.
        l2_bytes = dev.l2_bytes() or (128 << 20)
        scratch = torch.zeros(2 * int(l2_bytes) // 4, dtype=torch.float32,
                              device=dev.tensor_device)
        sink = torch.empty((), dtype=torch.float32, device=dev.tensor_device)

        def sweep():
            # A read of every line, at ~0.9 of DRAM peak (a reduction over
            # fp32, not int8: torch's int8 reduction runs at ~5%). Read-only,
            # so the lines it leaves are clean and it writes back whatever
            # was dirty — which is the point of the second one.
            torch.amax(scratch, dim=0, out=sink)

        # The window also queues the drain sweep after the call, so the hold
        # must cover its host time too. Timed twice and the second taken:
        # the first call allocates the 2x-L2 scratch and compiles the kernel
        # (9.6 ms on a 910C against 24 us steady), and a hold sized at a
        # one-off would stand in for nothing.
        sweep()
        dev.synchronize()
        _h0 = _time.perf_counter()
        sweep()
        host_s += _time.perf_counter() - _h0
        dev.synchronize()

    # The start event must not fire before the call's kernels are queued. If
    # the device drains the flush before the CPU has issued the call, it idles
    # between the two events and the launch path is timed as kernel time:
    # FA3/FA4's Python launch (~90 us) outlasts an H100's ~35 us flush, so a
    # 5 us prefill read 79 us, slower than torch SDPA beside it. A device-side
    # spin queued ahead of the start event, longer than the host time, keeps
    # the queue full; the events then bracket device work only.
    #
    # A device with no spin (Ascend: `DeviceBackend.spin`) keeps the queue
    # full another way — the sweep itself is queued work, and where it
    # outlasts the host path it serves. That is a MEASURED property of the
    # box, not an assumption: `sweep_holds` records whether it held, and a
    # sweep too short to cover the host path leaves the window unguarded,
    # which the method name then reports rather than hides.
    hold_cycles = _hold_cycles(dev, 2.0 * host_s + 20e-6)
    sweep_holds: Optional[bool] = None
    if sweep is not None and not hold_cycles:
        # What has to outlast the host path is whatever the device is still
        # working on when the CPU queues the call. Two things are: the sweep
        # just queued, and — because the samples run back to back with no
        # sync between them — the previous sample's own launch. So the guard
        # is the pair, and a kernel that runs longer than it takes to issue
        # guards its own window.
        #
        # Measured on a 910C, 4096^3 bf16 matmul (per_call 666 us, host 62
        # us typical): holding the queue 8x longer moved the result 0.1%.
        sweep_holds = (_sweep_gpu_s(dev, sweep) + per_call) > host_s

    def window(body, cold=True):
        start = dev.event()
        end = dev.event()
        if cold and sweep is not None:
            sweep()
        if hold_cycles:
            dev.spin(hold_cycles)
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
        if align is not None:
            align()
        if sweep is None:
            samples.append((window(call),))
        else:
            samples.append((window(launch_then_sweep), window(sweep),
                            window(lambda: None, cold=False)))
    dev.synchronize()

    def elapsed_us(w):
        return w[0].elapsed_time(w[1]) * 1000.0

    def latency_us(windows):
        if len(windows) == 1:
            return elapsed_us(windows[0])
        launch, alone, empty = (elapsed_us(w) for w in windows)
        return launch - alone + empty

    us = sorted(latency_us(w) for w in samples)
    n = len(us)
    p50 = us[n // 2]
    p95 = us[min(n - 1, round(0.95 * (n - 1)))]
    mean = sum(us) / n
    return LatencyStats(p50_us=p50, p95_us=p95, mean_us=mean, samples=n,
                        timing_method=_method(dev, sweep, hold_cycles,
                                              sweep_holds))


_CYCLES_PER_S: list = []


def _method(dev: DeviceBackend, sweep, hold_cycles: int,
            sweep_holds: Optional[bool]) -> str:
    """Which clock timed this row, in `benchmarks.TIMING_METHODS` terms.

    Two devices that reach a number by different routes must not record the
    same method: a reader comparing an Ascend row against an NVIDIA one has
    to be able to see that one was guarded by a spin and the other by its
    own sweep. So the name carries the device's hold, not just whether L2
    was drained.
    """
    if sweep is None:
        return "warm_l2"
    if hold_cycles:
        return "l2_drain"
    return "l2_drain_sweep_hold" if sweep_holds else "l2_drain_unheld"


def _sweep_gpu_s(dev: DeviceBackend, sweep) -> float:
    """How long one sweep occupies the device, in seconds — what decides
    whether it can stand in for a spin hold."""
    start, end = dev.event(), dev.event()
    start.record()
    sweep()
    end.record()
    dev.synchronize()
    return start.elapsed_time(end) / 1e3


def _hold_cycles(dev: DeviceBackend, seconds: float) -> int:
    """Spin cycles that keep the device busy for `seconds` (capped at 50 ms:
    a call that synchronizes inside itself has a host time that includes its
    own device time, and no hold helps it). The spin rate is measured once
    per process rather than read from a nominal clock.

    Zero when the device has no spin primitive (`DeviceBackend.spin`), which
    is not a failure: the caller then asks whether the sweep covers the host
    path instead."""
    if not _CYCLES_PER_S:
        probe = 1_000_000
        if not dev.spin(probe):           # first launch pays its own setup
            _CYCLES_PER_S.append(0.0)     # no spin on this device
        else:
            a, b = dev.event(), dev.event()
            a.record()
            dev.spin(probe)
            b.record()
            dev.synchronize()
            _CYCLES_PER_S.append(
                probe / max(a.elapsed_time(b) * 1e-3, 1e-9))
    if not _CYCLES_PER_S[0]:
        return 0
    return int(min(seconds, 0.05) * _CYCLES_PER_S[0])
