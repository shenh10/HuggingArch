"""The value types the layers hold.

Derived facts are separated by concern rather than stored on one mutable node:

    Cost      flops / params / active_params / quant metadata
    Shape     the tensors in and out, including secondary ports
    Formulas  Layer[Expr] — no wrapper needed, the value IS the expression
    Display   derived from the two above, never stored

Each is a frozen dataclass with a `combine` that says what happens when a graph
transform fuses several nodes into one. That is required rather than defaulted
because there is no safe default: taking the first origin's cost would make a
fused QKV report one projection's FLOPs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import sympy as sp

from ..dims import query_axes
from .dim import Dim, names_of

# A resolved axis extent. The walk produces `Dim` (identity + value).
# Untracked ints still mix through `Dim`'s operators at module edges.
Extent = Dim
Dims = Tuple[Extent, ...]

#: The cache length a KV footprint is a function OF.
#
# A walk runs at one cache length and every other layer is that length's answer.
# The cache is the exception: capacity asks how many requests fit, which is the
# same footprint evaluated at a length the walk never saw. So the positions an op
# declares are recorded in this symbol and substituted by the consumer, and it is
# spelled `s_kv` because that is the name the op's own lambda reads — a second
# spelling would need a translation step, and a translation step is somewhere for
# the two to disagree.
CACHE_LEN = sp.Symbol("s_kv")


def _add_extents(left: Extent, right: Extent) -> Extent:
    if isinstance(left, Dim):
        return left + right
    if isinstance(right, Dim):
        return right.__radd__(left)
    return left + right


def _multiply_extents(left: Extent, right: Extent) -> Extent:
    if isinstance(left, Dim):
        return left * right
    if isinstance(right, Dim):
        return right.__rmul__(left)
    return left * right


def product(extents: Optional[Iterable[Extent]]) -> Extent:
    """Π of a sequence of extents, and 0 for nothing.

    One implementation. There were four — `Shape.elements`,
    `StoredTensor.elements`, `traffic._elements`, `formats._shape_elements` —
    each three lines of `acc = acc * d`, each with its own answer for an empty
    or missing sequence. Zero is the right one: a shape that could not be
    inferred contributes nothing, where one would make it a scalar that is
    silently counted.
    """
    if not extents:
        return Dim.of(0)
    acc: Extent = Dim.of(1)
    for d in extents:
        acc = acc * d
    return acc


def _names(extent: Extent) -> frozenset:
    """The dim symbols an extent is built from — empty for one nobody named."""
    return names_of(extent)


@dataclass(frozen=True)
class Cost:
    """What one node costs. `params` excludes quantisation metadata (scale
    tensors and friends) because the HF checkpoint's parameter total does, and
    an accounting that mixes them cannot be checked against the checkpoint."""

    # `Dim` is a frozen dataclass over immutable sympy/int parts, so these
    # defaults are safe to share across instances.
    flops: Extent = Dim.of(0)  # noqa: RUF009 — immutable default
    params: Extent = Dim.of(0)  # noqa: RUF009 — immutable default
    # Per-token ACTIVE params: equal to `params` outside a MoE, smaller inside
    # one (a token reaches `n_experts_active` of `n_route_targets`
    # destinations, and only the ones that hold weights count here).
    active_params: Extent = Dim.of(0)  # noqa: RUF009 — immutable default
    quant_metadata_params: Extent = Dim.of(0)  # noqa: RUF009 — immutable default

    def __post_init__(self) -> None:
        if not isinstance(self.flops, Dim):
            object.__setattr__(self, "flops", Dim.of(self.flops))
        if not isinstance(self.params, Dim):
            object.__setattr__(self, "params", Dim.of(self.params))
        if not isinstance(self.active_params, Dim):
            object.__setattr__(self, "active_params", Dim.of(self.active_params))
        if not isinstance(self.quant_metadata_params, Dim):
            object.__setattr__(self, "quant_metadata_params",
                               Dim.of(self.quant_metadata_params))

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            flops=_add_extents(self.flops, other.flops),
            params=_add_extents(self.params, other.params),
            active_params=_add_extents(
                self.active_params, other.active_params),
            quant_metadata_params=_add_extents(
                self.quant_metadata_params, other.quant_metadata_params),
        )

    def scaled(self, storage: Extent, active: Extent) -> "Cost":
        """Apply a node's storage and active copy multipliers: `storage`
        copies are held, `active` of them run per token."""
        return Cost(
            flops=_multiply_extents(self.flops, active),
            params=_multiply_extents(self.params, storage),
            active_params=_multiply_extents(self.active_params, active),
            quant_metadata_params=_multiply_extents(
                self.quant_metadata_params, storage),
        )

    @staticmethod
    def combine(parts: Sequence["Cost"]) -> "Cost":
        out = Cost()
        for p in parts:
            out = out + p
        return out


    @property
    def is_zero(self) -> bool:
        return not (self.flops or self.params or self.active_params
                    or self.quant_metadata_params)


@dataclass(frozen=True)
class Moved:
    """How much of a boundary tensor a launch actually moves.

    A tensor's shape and the bytes a launch spends on it are the same question
    for every node but one: a FUSED node publishes the tensors its launch is
    handed, and a plan may absorb the `slice`/`concat` pair around a partial
    rotation to say the re-join is no copy in serving. Absorbing a node says it
    does not happen, so the whole of `q` crosses `fused_qk_rope_partial_q` on
    both sides while the rotary kernel touches only each head's last `d_rope`
    lanes, and the nope lanes are bytes nobody moved. Shape propagation and the
    harness that builds the real tensor need the published tensor; the byte
    price needs the region.

    Keyed by input position for the read side and by output port (`""` for the
    primary) for the write side. An entry is a side the launch MAY move less
    of than it publishes; it equals the published tensor wherever the region
    could not be established (`fusion._passed_through`'s clamp), and a
    consumer that needs to know which it got compares the two
    (`traffic._PortWrite.spanned`).

    Derived from the match's own topology (`fusion.rebase_shapes`), never
    declared by a plan — the plan registry carries no cost arithmetic and an
    author-filled region would be one more thing to drift. What a plan does
    declare is its output contract, and a launch that writes a buffer of its
    own is asked for no region at all.
    """

    reads: Mapping[int, Extent] = field(default_factory=dict)
    writes: Mapping[str, Extent] = field(default_factory=dict)


@dataclass(frozen=True)
class Shape:
    """The tensors crossing one node.

    `ins` is a tuple per input port, in the node's declared input order, so a
    two-input contraction keeps them distinguishable — the engine's flat-list-or-
    list-of-lists convention on `shape_in` forces every consumer to re-derive
    which case it is.

    `ports` holds secondary outputs for multi-output primitives; the main output
    is `out` and is never duplicated in here.

    `moved` is the one place where bytes are not the tensor: a fused boundary's
    launch may move less than it publishes (`Moved`). Ask `read_elements` /
    `written_elements` for a byte price and `out` / `ins` / `ports` for a shape.
    """

    out: Optional[Dims] = None
    ins: Tuple[Dims, ...] = ()
    ports: Mapping[str, Dims] = field(default_factory=dict)
    moved: Optional[Moved] = None
    pair_share: Any = None

    @property
    def rank(self) -> Optional[int]:
        return None if self.out is None else len(self.out)

    def elements(self) -> Extent:
        """Product of the output extents — the element count a bytes derivation
        multiplies by a dtype width. 0 for an unknown shape, so a missing shape
        contributes nothing rather than silently counting as one element."""
        return product(self.out)

    def axes_for(self, port: str = "out") -> Optional[Dims]:
        """Output axes for one port. Empty port is the primary."""
        key = port or "out"
        if key == "out":
            return self.out
        return self.ports.get(key)

    def read_elements(self, index: int) -> Extent:
        """Elements a launch reads at input `index`: what that input carries,
        less any region the launch only passed through (`Moved`)."""
        dims = self.ins[index] if index < len(self.ins) else None
        whole = product(dims)
        return (whole if self.moved is None
                else self.moved.reads.get(index, whole))

    def written_elements(self, port: str = "") -> Extent:
        """Elements a launch writes on one output port (`""` the primary):
        what that port carries, less any region it only passed through."""
        whole = product(self.axes_for(port))
        return (whole if self.moved is None
                else self.moved.writes.get(port, whole))

    # -- asking by identity ----------------------------------------------
    #
    # Every question below used to be asked by position or by arithmetic, in the
    # consumer, differently each time: `cache` divided by `s_ctx` and then
    # guessed from a producer's law, `traffic` walked the tuple, `flops` rebuilt
    # labels from the op's templates. They are the same question — WHICH axis is
    # this — and a tracked extent answers it, so it is answered here once.

    def names(self) -> "Tuple[frozenset, ...]":
        """The dim symbols each output axis is made of, per axis."""
        return tuple(_names(d) for d in (self.out or ()))

    def index_of(self, name: str) -> Optional[int]:
        """Which output axis is built from `name`, or None.

        The FIRST such axis: a shape carrying two axes over one symbol is a
        shape whose identity is ambiguous, and the only ones that did — an
        attention score's `(B, n_heads, 1, 1)` in decode, where both the query and the
        key axis ground to 1 — stopped being ambiguous once every dim was
        tracked rather than only the ones bound above 1.
        """
        for i, d in enumerate(self.out or ()):
            if name in _names(d):
                return i
        return None

    def _score_rows(self, kernel: Optional[str]) -> Optional[Dims]:
        """The score matrix this shape's node works on, `(…, s_query, keys)`:
        its output for a score (`QKᵀ`) or any op that is no attention
        kernel, its probabilities (its first input) for an apply (`PV`) —
        `kernel` is the op's `attention_kernel`. Read off the matrix, not
        the node's boundary: Ulysses enters and leaves an attention split
        by sequence while the matrix holds every query for its heads."""
        return (self.ins[0] if kernel == "apply" and self.ins else self.out)

    def queries(self, kernel: Optional[str]) -> Optional[Extent]:
        """The score matrix's query rows (`dims.query_axes`); None when it
        carries no query axis."""
        rows = self._score_rows(kernel)
        wanted = query_axes()
        return next((extent for extent in rows or ()
                     if _names(extent) & wanted), None)

    def pairs(self, kernel: Optional[str]) -> Optional[Extent]:
        """The score matrix's (query, key) cells per batch row and head: its
        query rows times its keys, the last axis of every matrix an
        attention scores. A rank's share of them is what context
        parallelism cuts — PCP the rows, DCP the keys, Ulysses neither.
        None for a node whose matrix has no query axis, or none past it."""
        rows = self._score_rows(kernel)
        queries = self.queries(kernel)
        if queries is None or rows[-1] is queries:
            return None
        return queries * rows[-1]

    def index_of_any(self, names: Iterable[str]) -> Optional[int]:
        """The first output axis built from any of `names`."""
        wanted = frozenset(names)
        return next((i for i, d in enumerate(self.out or ())
                     if _names(d) & wanted), None)

    def without(self, *names: str) -> Extent:
        """Product of every output axis NOT built from any of `names`.

        What "one row" means, stated as a question about identity rather than as
        a division. A cache row is `shape.without("B", "s_ctx")`, and that is
        correct for a compressing producer whose sequence axis is
        `floor(s_ctx/m)` without knowing that it compresses — the axis is
        still the one made of `s_ctx`, so it is still the one to drop. Dividing
        by the batch and the context instead needs to know the axis's extent,
        which is why it needed a rule about producers and consumers.
        """
        excluded = frozenset(names)
        return product(tuple(d for d in (self.out or ())
                             if not (_names(d) & excluded)))

    @staticmethod
    def combine(parts: Sequence["Shape"]) -> "Shape":
        """A fused node spans its origins: it reads what the first one read and
        produces what the last one produced. Secondary ports of interior nodes
        are internal to the fusion and do not survive it.

        No `moved`, and it would have nothing to put there: a region is a
        fact about the whole match's topology, not about the first and last
        of it. `fusion.rebase_shapes` calls this and then REPLACES every
        fused row with the boundary it builds itself (`moved` included), so
        a row this returns reaches a price only where that failed — and
        then it prices the published tensor, which over-bills loudly rather
        than under-billing silently."""
        if not parts:
            return Shape()
        return Shape(out=parts[-1].out, ins=parts[0].ins,
                     ports=dict(parts[-1].ports))

    def with_port(self, name: str, dims: Dims) -> "Shape":
        merged = dict(self.ports)
        merged[name] = dims
        return replace(self, ports=merged)
