"""Typed law adapter for the audited FlashMLA windowed sparse schedule.

SGLang 1b2870d3, kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh:246.
This schedules supplied local valid counts, not globally padded top-k slots.
Native SM partitions: FlashMLA 85b5264b csrc_legacy/api/sparse_decode_sched.h.
"""
from dataclasses import dataclass
from backend.code_trace.memo import memoize


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
