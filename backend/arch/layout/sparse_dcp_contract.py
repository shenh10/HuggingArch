"""Typed source contracts for striped sparse DCP, with no timing model.

Pinned SGLang 1b2870d33de7b5f31844fd74e97eea6d7896c53a:
* python/sglang/srt/layers/attention/dsa/dsa_dcp_sharded.py:178,384,410
* python/sglang/kernels/jit/csrc/dsa/dcp_topk_merge.cuh:301-424
* python/sglang/srt/layers/dcp/comm.py:552-600,683
* python/sglang/kernels/ops/attention/dcp_kernels.py:807

Inputs are walk-produced extents, not model config. Logical recipient bytes
are not NVLink hardware counters: multicast injection, peer-read traffic and
resident allocation have deliberately separate names. Synchronization polling
has no finite byte count derivable from these extents and is excluded.
"""
from dataclasses import dataclass
from typing import Any, Literal

import sympy as sp

from backend.arch.eval.dim import (
    ceil_div, exact_div, maximum, minimum, require, require_equal,
    require_multiple, value_of, ratio,
)


def _bounds(value: Any, lower: Any, upper: Any, label: str) -> None:
    require(sp.And(sp.Ge(value_of(value), value_of(lower)),
                   sp.Le(value_of(value), value_of(upper))),
            lambda at: ValueError(f"{label} outside [{at(lower)}, {at(upper)}]"))


@dataclass(frozen=True)
class SparseDcpWorkload:
    """One query's selected ownership, separately from fixed K-slot padding.

    `balanced_mean` reports an expectation, not a bound on the slowest rank.
    `busiest_rank` explicitly supplies max/mean valid-count ratio. `trace`
    supplies one query's rank counts (or homogeneous rows with these counts).
    Heterogeneous traces must be aggregated by the caller, not collapsed to
    one busiest count and presented as a measured distribution.
    """
    rows: Any
    context: Any
    topk: Any
    world: int
    selection_assumption: Literal["balanced_mean", "busiest_rank", "trace"]
    busiest_rank_factor: Any = 1
    trace_counts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.world < 1:
            raise ValueError("world must be positive")
        if self.selection_assumption not in ("balanced_mean", "busiest_rank", "trace"):
            raise ValueError("selected ownership requires an explicit assumption")
        _bounds(self.busiest_rank_factor, 1, self.world, "busiest_rank_factor")
        if self.selection_assumption != "busiest_rank":
            require_equal(self.busiest_rank_factor, 1,
                          lambda at: ValueError("factor requires busiest_rank assumption"))
        if self.selection_assumption == "trace":
            if self.trace_counts is None or len(self.trace_counts) != self.world:
                raise ValueError("trace requires one valid count per rank")
            for rank, count in enumerate(self.trace_counts):
                _bounds(count, 0, maximum(ceil_div(self.context - rank, self.world), 0),
                        "trace count")
            require_equal(sum(self.trace_counts), self.global_valid,
                          lambda at: ValueError("trace must conserve global selected count"))
        elif self.trace_counts is not None:
            raise ValueError("trace_counts requires trace assumption")

    @property
    def global_valid(self) -> Any:
        return minimum(self.context, self.topk)

    @property
    def mean_valid_per_query(self) -> Any:
        return ratio(self.global_valid, self.world, empty=0)

    @property
    def valid_per_query(self) -> Any:
        if self.trace_counts is not None:
            return max(self.trace_counts)
        if self.selection_assumption == "balanced_mean":
            return self.mean_valid_per_query
        # The maximum striped resident shard is an absolute ownership cap.
        chosen = self.mean_valid_per_query * self.busiest_rank_factor
        if self.selection_assumption == "busiest_rank":
            chosen = ceil_div(chosen, 1)
        return minimum(chosen, ceil_div(self.context, self.world))

    @property
    def selected_positions(self) -> Any:
        return self.rows * self.valid_per_query

    @property
    def padded_positions(self) -> Any:
        return self.rows * self.topk

    @property
    def local_index_scan_positions(self) -> Any:
        """Busiest striped shard; scan all local keys, before top-k selection."""
        return self.rows * ceil_div(self.context, self.world)


@dataclass(frozen=True)
class CandidateExchange:
    """Default multicast merge, not opt-in DCP fused top-k.

    `failed_owners` is the certificate's failed owner count per query, NOT the
    boolean fallback buffer or selected-count skew. Supply 0 only as an
    explicit no-fallback scenario. For mixed rows the mean yields mean traffic;
    peak traffic needs the largest observed failed-owner count separately.
    """
    rows: Any
    topk: Any
    world: int
    failed_owners: Any
    candidate_count: int = 384
    pair_bytes: int = 8

    def __post_init__(self) -> None:
        if self.world < 1 or self.candidate_count < 1 or self.pair_bytes < 1:
            raise ValueError("positive world, candidate_count and pair_bytes required")
        _bounds(self.failed_owners, 0, self.world, "failed_owners")
        _bounds(self.candidate_count, 1, self.topk, "candidate_count")

    @property
    def multicast_injected_bytes(self) -> Any:
        return self.rows * self.candidate_count * self.pair_bytes

    @property
    def multicast_recipient_bytes(self) -> Any:
        return self.multicast_injected_bytes * (self.world - 1)

    @property
    def fallback_mean_receive_bytes(self) -> Any:
        return (self.rows * self.failed_owners * (self.world - 1)
                * exact_div(self.topk * self.pair_bytes, self.world))

    @property
    def fallback_max_receive_bytes(self) -> Any:
        return (self.rows * minimum(self.failed_owners, self.world - 1)
                * self.topk * self.pair_bytes)

    @property
    def fallback_max_send_bytes(self) -> Any:
        # A failed owner serves a whole K-vector to every other rank.
        return (self.rows * minimum(self.failed_owners, 1)
                * (self.world - 1) * self.topk * self.pair_bytes)

    def resident_bytes(self, max_rows: Any) -> Any:
        """Double buffer + full union scratch + row counters + peer pointers.

        Full union scratch is resident even when every query avoids fallback.
        Excludes symmetric allocator overhead and separately allocated outputs.
        """
        return (2 * max_rows * (self.world * self.candidate_count + self.topk)
                * self.pair_bytes + max_rows * self.world * self.topk
                * self.pair_bytes + max_rows * 12 + self.world * 8)


@dataclass(frozen=True)
class OutputExchange:
    """Already-selected output merge path; eligibility belongs to dispatch.

    push/direct use 16-byte LSE slots, FlashInfer uses two F32 statistics,
    packed generic A2A uses one F32. A 128-value projection must be represented
    by value_dim=128, not a 512-latent tensor assigned a cheaper wire width.
    """
    rows: Any
    heads: Any
    world: int
    value_dim: Any
    kind: Literal["push", "direct", "flashinfer", "packed"]
    element_bytes: int = 2

    def __post_init__(self) -> None:
        if self.world < 1:
            raise ValueError("world must be positive")
        if self.kind not in ("push", "direct", "flashinfer", "packed"):
            raise ValueError("unknown output exchange path")
        require_multiple(self.heads, self.world,
                         lambda value: ValueError("heads must divide over DCP ranks"))
        if self.kind == "push":
            require_equal(self.value_dim, 128,
                          lambda at: ValueError("push kernel requires 128-wide partials"))
        if self.kind in ("push", "direct") and self.element_bytes != 2:
            raise ValueError("push/direct require BF16 partials")

    @property
    def lse_slot_bytes(self) -> int:
        return {"push": 16, "direct": 16, "flashinfer": 8, "packed": 4}[self.kind]

    @property
    def local_slot_bytes(self) -> Any:
        return (self.rows * exact_div(self.heads, self.world)
                * (self.value_dim * self.element_bytes + self.lse_slot_bytes))

    @property
    def remote_bytes(self) -> Any:
        """Logical outgoing recipient bytes per rank (excluding self)."""
        return self.local_slot_bytes * (self.world - 1)


@dataclass(frozen=True)
class LocalAttentionCombine:
    """Determinate arithmetic and memory floor for LSE-weighted merge.

    Arithmetic counts mathematical scalar work (FMA=2), including one exp per
    rank/head as one operation, not a claim about SFU instruction throughput.
    Slot reads/clears count one successful read; unbounded arrival polling and
    cache-line/transaction overhead are excluded. Received DMA writes are
    counted in HBM writes, not confused with additional interconnect traffic.
    """
    rows: Any
    heads: Any
    world: int
    value_dim: Any
    lse_slot_bytes: Any
    clears_slots: Any = 1

    @property
    def local_heads(self) -> Any:
        return exact_div(self.heads, self.world)

    @property
    def flops(self) -> Any:
        # max N-1, subtract N, exp N, sum N-1, reciprocal 1,
        # N weighted FMAs per value plus final normalization multiply.
        return self.rows * self.local_heads * (
            (2 * self.world + 1) * self.value_dim + 4 * self.world - 1)

    @property
    def remote_slots(self) -> Any:
        return self.rows * self.local_heads * (self.world - 1) * (
            self.value_dim * 2 + self.lse_slot_bytes)

    @property
    def read_bytes(self) -> Any:
        return self.rows * self.heads * (self.value_dim * 2 + 4) + self.remote_slots

    @property
    def write_bytes(self) -> Any:
        return (self.remote_slots * (1 + self.clears_slots)
                + self.rows * self.local_heads * self.value_dim * 2)


def sparse_dcp_query_pairs(context: Any, width: int, topk: Any, world: int,
                           factor: Any = 1) -> tuple[Any, Any]:
    """Valid selected/index pairs over a fixed-width causal verify block.

    ``context`` is the last query's context. Index scheduling receives a
    separate causal local length for each query (SGLang dsa_backend.py
    _build_paged_mqa_schedule_2d_ctx_lens); these are useful pairs, excluding
    tile padding. Index K HBM may be reused across those query rows.
    """
    if not isinstance(width, int) or width < 1:
        raise ValueError('sparse DCP query pairs require a positive fixed verification width')
    selected, indexed = 0, 0
    for offset in range(width):
        length = context - width + 1 + offset
        query = SparseDcpWorkload(1, length, topk, world, 'busiest_rank', factor)
        selected += query.valid_per_query
        indexed += ceil_div(length, world)
    return selected, indexed


@dataclass(frozen=True)
class SparseDcpLocalStages:
    """Local stages of default DCP decode, excluding repeated arrival polls.

    The localize stage has exact tensor traffic. Candidate merge includes
    mandatory candidate/slot traffic and conservative fallback scratch passes;
    radix iterations depend on scores and are reported as bounds, not FP32 FLOPs.
    """
    rows: Any
    topk: Any
    world: int
    local_candidates: Any
    failed_owners: Any = 0
    candidates: int = 384

    def localize(self):
        from .runtime_stage import RuntimeStageWork
        return RuntimeStageWork(
            'dcp_selected_owner_localize', self.rows * self.topk * 4,
            self.rows * (self.topk * 4 + 4),
            'https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/'
            'python/sglang/srt/layers/attention/dsa/dsa_dcp_sharded.py#L334',
            integer_ops=self.rows * (self.topk - 1) * 3,
            work_basis='exact_tensor_traffic; integer_scan_additions_floor',
        )

    def candidate_merge(self):
        from backend.arch.eval.dim import where
        from .runtime_stage import RuntimeStageWork
        scratch = where(lambda f: sp.Gt(f, 2), (self.failed_owners,),
                        self.rows * self.world * self.topk * 8, 0)
        read = self.rows * (self.topk * 8 + self.world * self.candidates * 8
                            + self.local_candidates * 4 + 12)
        read += self.rows * self.topk * 8 * self.failed_owners + 3 * scratch
        write = self.rows * (self.topk * 8 + self.world * self.candidates * 8
                             + self.topk * 4 + 12) + scratch
        return RuntimeStageWork(
            'dcp_topk_merge_exchange_certificate', read, write,
            'https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/'
            'python/sglang/kernels/jit/csrc/dsa/dcp_topk_merge.cuh#L301',
            integer_ops=self.rows * 2 * (self.topk + self.world * self.candidates),
            work_basis='mandatory_buffers_and_minimum_radix_key_visits; polling_excluded',
        )


@dataclass(frozen=True)
class SparseMlaLocalStages:
    """Legacy native-head FlashMLA runtime for a declared owner scenario.

    No attention sink, no extra-topk, BF16 output, flattened query rows and
    FLASHMLA_DIVIDE_PARTS unset. SM count is an explicit hardware fact. Unknown
    hardware retains only a labelled known-traffic floor, never SM148 by fiat.
    """
    batch: Any
    local_lengths: tuple
    heads: int
    value_dim: int
    sm_count: int | None

    @property
    def rows(self):
        return self.batch * len(self.local_lengths)

    @property
    def parts(self):
        from backend.arch.eval.dim import where
        if self.sm_count is None:
            return 0
        head_blocks = (self.heads + 63) // 64
        divided = max(self.sm_count // head_blocks, 1)
        return (where(lambda rows: sp.Lt(rows, divided), (self.rows,), divided, self.sm_count)
                if head_blocks > 1 else self.sm_count)

    @property
    def work(self):
        from .flashmla_schedule import WindowedWork, windowed_work
        if self.sm_count is None:
            return WindowedWork(0, self.rows, 0, 0)
        return windowed_work(self.batch, self.local_lengths, self.parts)

    @property
    def basis(self):
        return ('source_windowed_scheduler_declared_owner_pattern; scalar_math_floor; SFU_excluded'
                if self.sm_count is not None else 'unresolved_sm_count; known_traffic_floor_only')

    def scheduler(self):
        from .runtime_stage import RuntimeStageWork
        return RuntimeStageWork(
            'flashmla_windowed_scheduler', self.rows * 4,
            self.parts * 32 + (self.rows + 1) * 4,
            'https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/'
            'python/sglang/kernels/jit/csrc/deepseek_v4/flashmla_sched_meta.cuh#L246',
            integer_ops=2 * self.rows, work_basis=self.basis)

    def combine(self, producer_uid):
        from .runtime_stage import RuntimeStageWork
        work = self.work
        active, splits = work.combining_rows, work.combining_splits
        partials = splits * self.heads * (self.value_dim + 1) * 4
        final = active * self.heads * (self.value_dim * 2 + 4)
        # Mathematical scalar work: max(S-1), two subtract+exp passes(4S),
        # sum(S-1), log+add+convert(3), and S weighted value FMAs.
        flops = self.heads * ((2 * self.value_dim + 6) * splits + active)
        producer = partials + (self.rows - active) * self.heads * (self.value_dim * 2 + 4)
        return RuntimeStageWork(
            'flashmla_split_combine', partials + (self.rows + 1) * 4, final,
            'https://github.com/deepseek-ai/FlashMLA/blob/85b5264bbe2fe66327afc8a94f1eed80629937ca/'
            'csrc_legacy/kernels/smxx/decode/combine/combine.cu#L20',
            flops=flops, compute_dtype='fp32', work_basis=self.basis,
            producer_uid=producer_uid, producer_write_bytes=producer)
