"""CPU oracle for the audited FlashMLA windowed sparse schedule.

SGLang 1b2870d3, kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh:246.
This schedules supplied local valid counts, not globally padded top-k slots.
Native SM partitions: FlashMLA 85b5264b csrc_legacy/api/sparse_decode_sched.h.
"""
from dataclasses import dataclass
from bisect import bisect_right
from backend.code_trace.memo import memoize
from itertools import pairwise


@dataclass(frozen=True)
class WindowedSchedule:
    num_sm_parts: int
    num_splits: tuple[int, ...]
    snapped: bool
    max_partition_cost: int


def native_head_partitions(sm_count: int, rows: int, heads: int,
                           query_rows_per_request: int = 1) -> int:
    if min(sm_count, rows, heads, query_rows_per_request) < 1:
        raise ValueError('positive launch extents required')
    parts = max(sm_count // query_rows_per_request, 1)
    head_blocks = (heads + 63) // 64
    divided = max(parts // head_blocks, 1)
    return divided if head_blocks > 1 and rows < divided else parts


def windowed_schedule(local_lengths: tuple[int, ...], parts: int, *,
                      block_size: int = 64, overhead: int = 5) -> WindowedSchedule:
    if parts < 1 or block_size < 1 or overhead < 0 or not local_lengths:
        raise ValueError('nonempty rows and positive scheduler geometry required')
    if any(length < 0 for length in local_lengths):
        raise ValueError('local valid lengths cannot be negative')
    blocks = [(max(length, 1) + block_size - 1) // block_size for length in local_lengths]
    prefix = [0]
    for count in blocks:
        prefix.append(prefix[-1] + count + overhead)
    rows, total = len(blocks), prefix[-1]
    window = (total + parts - 1) // parts

    def cuts(snap):
        out = []
        for part in range(parts + 1):
            pos = part * window
            row, block = rows, 0
            if pos < total:
                row = bisect_right(prefix, pos) - 1
                offset = pos - prefix[row]
                if snap and prefix[row + 1] - pos <= overhead:
                    row += 1
                elif offset > overhead:
                    block = offset - overhead
            out.append((row, block))
        return out

    def cost(cut):
        def position(row, block):
            return prefix[row] + (overhead + block if block > 0 else 0)
        return max(position(*right) - position(*left) + (overhead if left[1] else 0)
                   for left, right in pairwise(cut))

    plain, snap = cuts(False), cuts(True)
    snapping = cost(snap) <= cost(plain)
    counts = tuple(1 + max((prefix[r + 1] - 1 - (overhead if snapping else 0)) // window
                           - (prefix[r] + overhead) // window, 0) for r in range(rows))
    return WindowedSchedule(parts, counts, snapping, cost(snap if snapping else plain))


@dataclass(frozen=True)
class WindowedWork:
    parts: object
    total_splits: object
    combining_rows: object
    combining_splits: object


@memoize(maxsize=128)
def windowed_work(batch, local_lengths: tuple, parts, *, block_size=64, overhead=5):
    """Compact numeric/symbolic law over a declared periodic owner scenario.

    The closed runtime-law registry evaluates the same bounded CPU algorithm
    after substitution and in compiled scalar/vector readers. This avoids a
    270 KB expression per metric for just 74 hardware partitions.
    """
    from backend.arch.runtime_laws import SparseWindow
    if block_size != 64 or overhead != 5:
        raise ValueError('registered window scheduler requires source block64/overhead5')
    return WindowedWork(parts, *(SparseWindow(mode, batch, parts, *local_lengths)
                                 for mode in range(3)))
