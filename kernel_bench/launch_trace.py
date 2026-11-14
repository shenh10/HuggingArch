"""What the timed call actually handed the GPU, read off the dispatcher.

A provider's `build` is an adapter, and an adapter can quietly measure
something else than its row prices. Three of them did: the MoE dispatch
built equal-length operands for a launch that expands rows, the conv1d
decode build sliced one token out of a cell billed for several, the MoE
combine built fp32 operands for a row billed in bf16. Each was found by
hand, months apart, after its rows were already in the corpus; each was
invisible to every existing check, because `cell_check` compares the
tensors handed TO the provider and the parity report compares the
signature's declaration against the walk — and the slicing, packing and
casting all happen in between.

This reads the other end: torch's dispatcher, during the one call the
cross-check already makes, records every aten op the launch performs with
the dtype and shape of each operand. Nothing is declared and no provider
is edited — a kernel reached through `torch.ops` (vLLM, sglang, flashinfer,
every torch baseline) appears as itself. A kernel reached through a raw
pybind entry point (DeepGEMM, flash-attn's extension) or launched by
Triton does not, and its absence is reported as such rather than as an
empty launch: `Launch` is how those name themselves. Absent includes the
ops aten does see AROUND such a kernel — the output's allocation, a little
index arithmetic — which is why the trace asks whether the call's result
came out of an op it saw before it judges anything (`Trace.saw_result`).

Two questions are asked of the trace, both tolerant of layout, because a
provider is free to permute, pack and page its operands:

  * what its largest operand is MADE OF — a row priced in bf16 whose
    launch moves fp32 is measuring another precision;
  * how much of it there is — that same buffer against the largest
    tensor the walk published for the cell, which is what a sliced or an
    unexpanded operand gets wrong. Both sides count inputs AND outputs,
    since the walk publishes both.

Neither compares shapes, and neither names an op.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple



class Unobservable(RuntimeError):
    """The traced call ran, and the trace cannot hand its result back.

    A dispatch mode returns each op's result through torch's Python ->
    C++ conversion, and that conversion has no spelling for an UNDEFINED
    tensor inside a `Tensor[]` return: C++ hands Python `None` there, and
    `None` does not convert back. vLLM's FA3 forward
    (`torch.ops._vllm_fa3_C.fwd -> Tensor[]`, vllm_flash_attn/
    flash_attn_interface.py:379 at v0.32.0) returns two such entries, so
    every traced call of it raised pybind's "Unable to cast Python instance
    of type <class 'list'>" — a failure of the trace, recorded as the
    kernel's `error`, and its canary disabled the path (gpt-oss,
    2026-10-26). Raised in place of that, after the op is on the trace, so
    the caller can tell the two apart: the trace of the launch is whole,
    only the call's result is lost, and the call is made again untraced."""


@dataclass
class Operand:
    """One tensor an op was handed or produced.

    `buffer`, `start` and `stop` place it in the allocation it views (bytes,
    from its storage offset to the last element its strides reach), so that
    a strided view is read as the buffer region it spans and several views
    of one buffer as one buffer. An Operand built by hand has no buffer and
    stands for itself."""
    dtype: str
    shape: Tuple[int, ...]
    buffer: Optional[int] = None
    start: int = 0
    stop: int = 0
    itemsize: int = 0

    @property
    def elements(self) -> int:
        count = 1
        for extent in self.shape:
            count *= int(extent)
        return count

    def __str__(self) -> str:
        return f"{self.dtype}[{','.join(str(int(e)) for e in self.shape)}]"


@dataclass
class Extent:
    """How much of one buffer a launch touched, and the view that shows it."""
    elements: int
    operand: Operand

    @property
    def dtype(self) -> str:
        return self.operand.dtype

    def __str__(self) -> str:
        return str(self.operand)


@dataclass
class Call:
    """One dispatched op, with the operands it was handed and the tensors it
    produced. `describe` renders the first only: that is the call."""
    op: str
    operands: Tuple[Operand, ...]
    scalars: Tuple[str, ...] = ()
    produced: Tuple[Operand, ...] = ()
    #: False for the harness's adapter around a launch — an allocation, a
    #: view, or a copy (`.contiguous()`, `.to(dtype)`, a write into a
    #: preallocated buffer): recorded, so the trace is the whole call, but
    #: not the kernel's work.
    moves: bool = True

    def __str__(self) -> str:
        parts = [str(o) for o in self.operands] + list(self.scalars)
        return f"{self.op}({', '.join(parts)})"


@dataclass
class Trace:
    """Every op one call dispatched, in order.

    `saw_result` says whether the call's result was written by an op the
    trace saw (directly, or through a view or copy of what one wrote): None
    when nobody handed the trace the result (`record.run` does), False when
    the result came out of a kernel the dispatcher never saw — a Triton or
    pybind launch, around which aten saw only an allocation or some index
    arithmetic. Those ops are not the launch, and judging them as if they
    were refused every vLLM Triton GEMM for its output's template."""
    calls: List[Call] = field(default_factory=list)
    saw_result: Optional[bool] = None

    @property
    def work(self) -> List[Call]:
        """The calls that moved something of their own."""
        return [c for c in self.calls
                if c.moves
                and (c.operands or c.produced)]

    @property
    def observed(self) -> bool:
        """Whether the trace saw the launch: some work, and — when the result
        is known — the result among what that work wrote."""
        return bool(self.work) and self.saw_result is not False

    def describe(self, limit: int = 3) -> str:
        """The signature a row records. Several calls are several launches,
        and the row says so rather than naming the first."""
        work = self.work
        if not work:
            return ""
        shown = "; ".join(str(c) for c in work[:limit])
        if len(work) > limit:
            shown += f"; +{len(work) - limit} more"
        return (shown if len(work) == 1
                else f"{len(work)} launches: {shown}")

    @property
    def widest(self) -> Optional[Extent]:
        """The first of `widest_buffers`: one buffer to name in a message."""
        tied = self.widest_buffers
        return tied[0] if tied else None

    @property
    def widest_buffers(self) -> Tuple[Extent, ...]:
        """The largest buffers the call moved — every one of them when
        several tie — by how much of it the launch's operands and products
        span.

        All the tied ones, because which of them came first is the order a
        kernel happens to take its arguments in, not a fact about the launch:
        a full-width rope over one head is handed q, k and a cos/sin table of
        exactly as many elements, and the table may be any of the three.

        Both sides, because the cell declares both: the walk publishes the
        output beside the inputs (`driver._declared_elements`), and an op's
        output is not among its arguments unless it was handed an `out=`.
        Read without them, a GEMM whose output outgrows its inputs (any
        `d_out > d_reduce` at prefill) and every embedding lookup "measured a
        part of the cell" — 5k cuBLAS and 1.2k embedding cells on the 4090.

        By buffer, because a launch is handed buffers: a kernel that strides
        over the rope lanes of q is handed q (the walk publishes q whole for
        exactly that reason, `test_eval_traffic`), and a floor that loops over
        experts touches the stacked weight one `w[e]` at a time. A view counts
        as the region of its buffer it spans; views of one buffer count as
        their union. A compact copy is a buffer of its own, so a build that
        copies one token out of five is still a fifth of the cell."""
        groups: dict = {}
        for call in self.work:
            for operand in (*call.operands, *call.produced):
                key = (operand.buffer if operand.buffer is not None
                       else ("own", id(operand)))
                groups.setdefault(key, []).append(operand)
        extents = [_coverage(operands) for operands in groups.values()]
        if not extents:
            return ()
        most = max(e.elements for e in extents)
        return tuple(e for e in extents if e.elements == most)


def _coverage(operands: Sequence[Operand]) -> Extent:
    """The union of the regions `operands` span in their common buffer."""
    if operands[0].buffer is None:
        return Extent(operands[0].elements, operands[0])
    intervals = sorted((o.start, o.stop) for o in operands if o.stop > o.start)
    covered, reach = 0, None
    for lo, hi in intervals:
        if reach is None or lo > reach:
            covered += hi - lo
            reach = hi
        elif hi > reach:
            covered += hi - reach
            reach = hi
    shown = max(operands, key=lambda o: (o.stop - o.start, o.elements))
    return Extent(covered // max(shown.itemsize, 1), shown)


def _moves_nothing(func, inputs, outputs, written) -> bool:
    """Whether an op moved no data of its own, by torch's declarations: an
    allocation (torch's own tensor-constructor classification, which is
    what FakeTensor dispatches on), a view (every tensor it returned shares
    storage with one it was handed, and its schema writes nothing), or an op
    that returned no tensor and wrote none (`is_pinned`, `.item()`)."""
    if _is_constructor(func):
        return True
    if written:
        return False
    if not outputs:
        return True
    held = {_storage_key(t) for t in inputs}
    return all(_storage_key(t) in held for t in outputs)


def _is_constructor(func) -> bool:
    try:
        from torch._subclasses import fake_tensor
    except Exception:                                      # pragma: no cover
        return False
    try:
        return bool(fake_tensor._is_tensor_constructor(func)
                    or func in fake_tensor._like_tensor_constructors)
    except Exception:
        return False


def _storage_key(tensor) -> int:
    try:
        return int(tensor.untyped_storage()._cdata)
    except Exception:
        return id(tensor)


class _Buffers:
    """Names the allocations one call touches. A storage's address can be
    reused once it is freed, so each name is held through a weak reference
    and a dead one is never matched: two buffers that happened to share an
    address are two buffers."""

    def __init__(self) -> None:
        self._refs: dict = {}
        self._next = 0
        self.written: set = set()

    def name(self, tensor) -> Optional[int]:
        try:
            from torch.multiprocessing.reductions import StorageWeakRef
            storage = tensor.untyped_storage()
            key = int(storage._cdata)
        except Exception:
            return None
        entry = self._refs.get(key)
        if entry is not None and not entry[0].expired():
            return entry[1]
        self._next += 1
        self._refs[key] = (StorageWeakRef(storage), self._next)
        return self._next

    def operand(self, tensor) -> Operand:
        shape = tuple(int(e) for e in tensor.shape)
        dtype = str(tensor.dtype).replace("torch.", "")
        buffer = self.name(tensor)
        try:
            itemsize = int(tensor.element_size())
            strides = [abs(int(s)) for s in tensor.stride()]
            first = int(tensor.storage_offset())
            last = first
            for size, stride in zip(shape, strides, strict=True):
                if size > 0:
                    last += (size - 1) * stride
            empty = any(size == 0 for size in shape)
            # Rows of a wider row: a view whose innermost axis is a
            # contiguous prefix of rows laid `pitch` apart (q[..., :rotary]
            # of [tokens, heads, head_size], as sglang hands its rope kernel,
            # srt/layers/rotary_embedding/base.py:397-401 at v0.5.22) is
            # handed those rows, so its region runs to the end of its last
            # row. Read as its span alone, a single such row was "64
            # elements where the cell's holds 256" and refused, while the
            # same view over two rows already spanned nearly the whole head.
            if len(shape) > 1 and not empty and (
                    shape[-1] == 1 or strides[-1] == 1):
                pitch = min((s for s in strides[:-1] if s > 0), default=0)
                if pitch > shape[-1]:
                    held = tensor.untyped_storage().nbytes() // itemsize
                    last = min(last - (shape[-1] - 1) + pitch - 1,
                               max(held - 1, last))
            start, stop = first * itemsize, (
                first * itemsize if empty else (last + 1) * itemsize)
        except Exception:
            buffer, itemsize, start, stop = None, 0, 0, 0
        return Operand(dtype, shape, buffer, start, stop, itemsize)


class record:
    """Context manager: collect the ops a block dispatches.

    A no-op without torch, and silent about its own failure to install —
    a box whose torch has no dispatch mode records nothing, which reads as
    "not observed" everywhere below. `run(fn)` makes the call inside it and
    tells the trace whether what came back was written by an op it saw.
    """

    def __init__(self) -> None:
        self.trace = Trace()
        self._mode: Any = None
        self._buffers = _Buffers()

    def run(self, fn):
        with self:
            result = fn()
            self.trace.saw_result = self._saw(result)
        return result

    def _saw(self, result) -> Optional[bool]:
        try:
            import torch
            from torch.utils._pytree import tree_leaves
        except Exception:                                  # pragma: no cover
            return None
        tensors = [t for t in tree_leaves(result) if torch.is_tensor(t)]
        if not tensors:
            return None
        return any(self._buffers.name(t) in self._buffers.written
                   for t in tensors)

    def __enter__(self) -> "record":
        try:
            import torch
            from torch.utils._pytree import tree_leaves
            from torch.utils._python_dispatch import TorchDispatchMode
        except Exception:                                  # pragma: no cover
            return self
        trace, buffers = self.trace, self._buffers

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                scalars = [str(value)[:24]
                           for value in (*args, *kwargs.values())
                           if isinstance(value, (int, float, bool, str))]
                inputs = [t for t in tree_leaves((args, kwargs))
                          if torch.is_tensor(t)]
                written = _written(func, args, kwargs)
                result = func(*args, **kwargs)
                outputs = [t for t in tree_leaves(result)
                           if torch.is_tensor(t)]
                op = str(func).split(".default")[0]
                try:
                    idle = _moves_nothing(func, inputs, outputs, written)
                except Exception:
                    idle = False
                moves = not idle and not _is_copy(func)
                # Who wrote what: a working op writes its products and its
                # mutated arguments; a view or a copy of something written
                # carries that on; an allocation writes nothing of the
                # launch's (vLLM's Triton GEMM allocates its output through
                # aten and fills it where aten cannot see).
                read = {buffers.name(t) for t in inputs}
                carries = (not _is_constructor(func)
                           and (moves
                                or bool(read & buffers.written)))
                if carries:
                    for t in (*outputs, *written):
                        name = buffers.name(t)
                        if name is not None:
                            buffers.written.add(name)
                trace.calls.append(Call(
                    op, tuple(buffers.operand(t) for t in inputs),
                    tuple(scalars),
                    tuple(buffers.operand(t) for t in outputs
                          if all(t is not w for w in written)),
                    moves=moves))
                if _unreturnable(func, result):
                    # Recorded first: the launch happened, and the trace of
                    # it is whole.
                    raise Unobservable(
                        f"{func} returned an undefined tensor inside a "
                        f"Tensor[]; a dispatch mode cannot hand that back")
                return result

        self._mode = _Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exception: Any) -> None:
        if self._mode is not None:
            self._mode.__exit__(*exception)
            self._mode = None


@lru_cache(maxsize=1)
def _copy_declarations() -> Tuple[Any, ...]:
    """torch's three copy-primitive dispatch declarations.

    They are identified by the dispatched `OpOverload` object itself, not
    by the display string the trace records: copy is a fact about the call,
    and `Call.moves` carries it once where the call is made. A schema cannot
    tell a copy from any other unary op, which is why these declarations —
    not `_moves_nothing` — answer this one question.
    """
    import torch

    return (torch.ops.aten.clone.default,
            torch.ops.aten._to_copy.default,
            torch.ops.aten.copy_.default)


def _is_copy(func: Any) -> bool:
    return any(func is declaration for declaration in _copy_declarations())


def _unreturnable(func, result) -> bool:
    """Whether `result` holds `None` where `func`'s schema returns a list of
    (non-optional) tensors — what `Unobservable` is raised for."""
    try:
        import torch
        returns = func._schema.returns
        tensor = torch.TensorType.get()
    except Exception:
        return False
    values = (result,) if len(returns) == 1 else tuple(result or ())
    for ret, value in zip(returns, values, strict=False):
        kind = ret.type
        if (isinstance(kind, torch.ListType)
                and kind.getElementType().isSubtypeOf(tensor)
                and isinstance(value, (list, tuple))
                and any(item is None for item in value)):
            return True
    return False


def _written(func, args, kwargs) -> List[Any]:
    """The tensors an op's schema says it writes (`Tensor(a!)`: in place, or
    an `out=`)."""
    try:
        import torch
        from torch.utils._pytree import tree_leaves
        arguments = func._schema.arguments
    except Exception:
        return []
    out = []
    for index, argument in enumerate(arguments):
        info = argument.alias_info
        if info is None or not info.is_write:
            continue
        value = (args[index] if index < len(args)
                 else kwargs.get(argument.name))
        out.extend(t for t in tree_leaves(value) if torch.is_tensor(t))
    return out


def paged_byte_cache_shape(shape, page: int, record: str) -> Tuple[int, ...]:
    """Paged [B,L,D] cache stored as bytes, including its declared scales.

    The format supplies payload and scale widths; the adapter supplies the
    page layout. This is a storage contract, not permission for any byte
    operand to stand for a floating-point tensor.
    """
    from backend.arch.cache_formats import CacheFormat
    fmt = CacheFormat.parse(record, "paged byte cache")
    batch, length, width = map(int, shape)
    if page <= 0 or fmt.tail is not None or fmt.transform != "none":
        raise ValueError("paged byte cache requires positive pages and no tail/transform")
    dynamic, static = fmt.scale_counts(width, width, 1)
    size = width * fmt.bytes_per_element(width) + dynamic * fmt.scale_bytes
    if static or int(size) != size or fmt.scale is None:
        raise ValueError("paged byte cache requires integral dynamic records")
    return (batch * ((length + page - 1) // page), page, 1, int(size))


def disagreement(trace: Trace, *, declared_dtypes: Sequence[str],
                 declared_elements: Sequence[int],
                 records: Sequence[str] = (),
                 packed_byte_shapes: Sequence[Tuple[int, ...]] = ()) -> Optional[str]:
    """Why this launch is not the cell's, or None.

    Both questions are asked of the WIDEST buffer the launch moved, which
    is the one whose bytes the row is mostly pricing: what it is made of,
    and how much of it there is. A trace that did not see the launch — every
    kernel behind a pybind entry point or Triton, whatever aten saw around
    it (`Trace.saw_result`) — is not a disagreement.

    Neither question is asked of the smaller operands, deliberately: a
    narrow kernel carries fp32 scales, int32 positions and a page table
    beside its payload, and those are not a precision claim.
    """
    if not trace.observed:
        return None
    widest = trace.widest
    if widest is None or not declared_elements:
        return None
    wanted = max(int(e) for e in declared_elements)
    # An order of magnitude, not equality: a launch is free to pad its rows
    # to a tile, pack two operands into one buffer, or carry a page table
    # beside them. Off by the tokens, the experts or the heads is what the
    # three bugs this exists for all were (0.2x, 1.5x, 5x).
    if widest.elements * 2 < wanted:
        return (f"the launch's largest operand holds {widest.elements} "
                f"elements where the cell's holds {wanted} ({widest}) — it "
                f"measures a part of the cell")
    spellings = _spellings(declared_dtypes)
    if records:
        # A writer that names its cached record may compute (and decode for
        # the cross-check) in fp32 before it lands on that record's grid.
        # The record's own step — not this intermediate precision — is what
        # `_xcheck_band` judges those pre-rounding differences against.
        spellings = tuple(sorted({*spellings, "float32"}))
    # Buffers tied for widest share the row's bytes, and none of them alone
    # stands for it: the launch is the cell's precision if any of them is.
    if spellings and all(e.dtype not in spellings and not (
                         e.dtype == "uint8" and
                         e.operand.shape in packed_byte_shapes)
                         for e in trace.widest_buffers):
        return (f"the launch's largest operand is {widest.dtype} where the "
                f"cell declares {'/'.join(sorted(spellings))} ({widest}) — "
                f"its latency is another precision's")
    return None


def _spellings(declared: Sequence[str]) -> Tuple[str, ...]:
    """`declared` as torch spells them, through the harness's own mapping
    (`refs.torch_dtype`) rather than a second table of dtype names. A name
    that mapping does not know yields nothing, which asks nothing."""
    try:
        from .refs import torch_dtype
    except Exception:                                      # pragma: no cover
        return ()
    out = set()
    for name in declared:
        if not name:
            continue
        try:
            out.add(str(torch_dtype(str(name))).replace("torch.", ""))
        except Exception:
            continue
    return tuple(sorted(out))
