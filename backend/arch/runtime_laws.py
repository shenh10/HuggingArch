"""Closed, source-audited runtime laws with bounded numeric evaluation.

No architecture imports: the generated-law reader and compilers share this
single registry. SparseWindow is the windowed sparse-attention scheduler's
aggregate counts, not a callback supplied by model text or a second walk.
"""
from bisect import bisect_right
from backend.code_trace.memo import memoize
from itertools import pairwise

import sympy as sp


class WindowedSchedulerOverrun(ValueError):
    """A batch too large for the audited 48 KiB windowed-schedule precompute.

    The SGLang wrapper falls back to FlashMLA's native serial scheduler past
    this bound, and that distinct schedule is not modelled. Carried as a
    typed `ValueError` subclass so a sweep's per-point isolation refuses the
    one overrun point while its siblings stand; the point-cloud producers
    read the same fact as `windowed_admits` BEFORE evaluating a law."""


@memoize(maxsize=512)
def windowed_admits(batch: int, parts: int, lengths: tuple[int, ...]) -> bool:
    """Whether a periodic-owner batch fits the 48 KiB windowed-schedule
    precompute admission (SGLang 1b2870d3 flashmla_sched_meta.cuh, block64 /
    overhead5). Pure — the point producers call it before law evaluation so
    the overrun is a refusal, not an exception inside a compiled law."""
    return batch <= windowed_admission_ceiling(parts, lengths)


@memoize(maxsize=512)
def windowed_admission_ceiling(parts: int, lengths: tuple[int, ...]) -> int:
    """The largest request batch whose periodic-owner rows fit the windowed
    schedule's 48 KiB admission (`windowed_admits`). One statement of the
    bound, so a producer filters a batch axis without restating the constant."""
    period = len(lengths)
    return (48 * 1024 // 4 - 1 - parts * 8) // (5 * period)


@memoize(maxsize=2048)
def windowed_counts(batch: int, parts: int, lengths: tuple[int, ...]) -> tuple[int, int, int]:
    """Total splits, combining rows, combining splits for a periodic owner.

    SGLang 1b2870d3 flashmla_sched_meta.cuh:246, block64/overhead5.
    Iterate hardware partitions, not workload rows. Only the declared repeated
    owner pattern is modeled; it is not evidence of measured top-k ownership.
    """
    if batch < 1 or parts < 1 or not lengths or min(lengths) < 0:
        raise ValueError('positive batch/parts and nonnegative local lengths required')
    # The SGLang wrapper falls back to FlashMLA's native scheduler outside
    # this precompute admission bound. That distinct schedule is not modeled,
    # so the overrun is a refusal (`windowed_admits`) the point producers
    # consult before evaluating a law — an unadmitted batch reaching this
    # function is a bug in the producer's admission check.
    if not windowed_admits(batch, parts, lengths):
        raise WindowedSchedulerOverrun(
            'sparse DCP windowed scheduler exceeds 48KiB admission; '
            'native FlashMLA scheduler fallback is not modeled')
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + (max(length, 1) + 63) // 64 + 5)
    period, cycle = len(lengths), prefix[-1]
    rows, total = batch * period, batch * cycle
    window = (total + parts - 1) // parts

    def prefix_at(row):
        quotient, remainder = divmod(row, period)
        return quotient * cycle + prefix[remainder]

    def details(snap):
        cuts = []
        for p in range(parts + 1):
            pos = p * window
            row, block = rows, 0
            if pos < total:
                quotient, remainder = divmod(pos, cycle)
                inner = bisect_right(prefix, remainder) - 1
                row = quotient * period + inner
                if snap and prefix[inner + 1] - remainder <= 5:
                    row += 1
                else:
                    block = max(remainder - prefix[inner] - 5, 0)
            cuts.append((row, block))
        positions = [prefix_at(row) + (5 + block if block else 0) for row, block in cuts]
        cost = max(right - left + (5 if cuts[i][1] else 0)
                   for i, (left, right) in enumerate(pairwise(positions)))
        extra = active = 0
        for i in range(1, parts):
            row, block = cuts[i]
            if row < rows and block > 0:
                extra += 1
                active += row != cuts[i - 1][0] or cuts[i - 1][1] <= 0
        return cost, extra, active

    plain, snap = details(False), details(True)
    _, extra, active = snap if snap[0] <= plain[0] else plain
    return rows + extra, active, active + extra


def sparse_window(mode, batch, parts, *lengths):
    values = (mode, batch, parts, *lengths)
    integers = tuple(int(x) for x in values)
    if any(x != n for x, n in zip(values, integers, strict=True)):
        raise ValueError('SparseWindow arguments must be integers')
    index, b, p, *pattern = integers
    if index not in (0, 1, 2):
        raise ValueError('SparseWindow mode must be 0, 1 or 2')
    return windowed_counts(b, p, tuple(pattern))[index]


class SparseWindow(sp.Function):
    """Compact typed law: mode0 splits, mode1 active rows, mode2 active splits."""
    evaluate_numeric = staticmethod(sparse_window)
    is_integer = True
    is_nonnegative = True

    @classmethod
    def eval(cls, *args):
        if len(args) < 4:
            raise ValueError('SparseWindow requires mode, batch, parts and local lengths')
        if all(arg.is_number for arg in args):
            return sp.Integer(cls.evaluate_numeric(*args))
        return None


RUNTIME_LAW_OPERATORS = {operator.__name__: operator for operator in (SparseWindow,)}
RUNTIME_LAW_FUNCTIONS = {name: operator.evaluate_numeric
                         for name, operator in RUNTIME_LAW_OPERATORS.items()}
