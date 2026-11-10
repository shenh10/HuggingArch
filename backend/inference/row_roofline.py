"""The per-row roofline, written once — as sympy expressions in the
workload and the card.

A per-op row carries batch-invariant components (resident weight bytes,
metadata bytes) and the parts that move with the workload (FLOPs, activation
bytes, the payload of its collective), each the walk's own law in the
workload's symbols (`transforms.Workload.symbols`: the batch `B`, the prompt,
the output, the cached prefix and the verify width) — the walk is every
point's, so a component needs no reading of its own. `n` is the row's count
of instances in the step, and a byte column is whole per instance, as the
walk counts it:

    flops           = round(n x flops_one)
    act_in          = n x floor(act_in_one)
    act_out, convert, kv_read                         the same, each its own
    act             = act_in + act_out + convert
    asked           = rows x read_per_row                      (`weight_demand`)
    weight_read     = floor(Min(asked, weight_total))          <- MoE saturation
    metadata        = floor(Min(asked, weight_total) x metadata / weight)
    access          = weight_read + metadata + act
    comm_payload    = comm_one                                 (one instance)
    comm_volume     = ring_factor x comm_payload x n           (per-rank bytes)
    comm_bytes      = comm_payload                 (what the collective prices)
    t_compute       = flops / PEAK_FLOPS
    t_memory        = (weight_read + metadata + act) / PEAK_BW
    t_comm          = comm_bytes x COMM_S_PER_BYTE + COMM_LATENCY_S
                      (by payload range, below)
    t_launch        = n x LAUNCH_S                  (a row that launches)
    t_op            = Max(t_compute, t_memory, t_comm, t_launch)

`COMM_LATENCY_S` is the collective's fixed time over the step (each phase's
fabric latency per step it takes, `comm_cost.collective_latency`), present
only where the collective is priced. Which algorithm runs the collective —
and so what a byte costs and how many traversals it waits on — is the
framework's dispatch by payload (`comm_cost.collective_ladder`): a
collective whose framework declares rungs (`row["comm_rungs"]`) is priced
as cases over its payload,

    t_comm = comm_bytes x COMM_S_PER_BYTE_r<k> + COMM_LATENCY_S_r<k>
               for the first k with comm_bytes <= COMM_MAX_BYTES_r<k>,
             comm_bytes x COMM_S_PER_BYTE + COMM_LATENCY_S  past them all,

each range's bound, rate and fixed time the card's (a range the card does
not admit has bound 0): another point lands in the range its own payload
does. At one point the range is fixed, so the time stays linear in the
link latency (`FloorForm`). `LAUNCH_S` is the shortest one kernel
launch takes (the request's `launch_floor_us`): a row that runs a
kernel — it moves bytes or FLOPs and is not a collective, whose own floor
is its latency — takes at least that per instance, however little it does.

`round` is `floor(x + 1/2)`; the walk rounds with Python's `round`, which
sends an exact half to the even neighbour, so the two can differ by one FLOP
on a row whose count times its per-instance FLOPs lands on an exact half.

`PEAK_FLOPS` (the row's own peak, by its `act_dtype`), `PEAK_BW`,
`COMM_S_PER_BYTE` (what `price_collective` charges per payload byte for the
row's collective on the card's fabric, its latency apart), `COMM_LATENCY_S`
and `LAUNCH_S` are the card: another point substitutes the workload,
another card the five.

`rows` is what the LEAF's own kernel runs on, read off the leaf's own extents
and nothing else — the step's tokens for a leaf that runs on them, and the
routed rows ONE EXPERT receives for an expert bank. `read_per_row` is what one
of those rows demands of the leaf's resident weights: the whole resident set
for a launch that streams it, one token's gathered share for a lookup, and —
since a bank's rows are counted per expert — the whole local bank for a
routed row, so that the product is the rows the rank received times one
expert's bytes. It is deliberately not the step's token count
for every leaf: the step's count is a model-level fact and the rows are the
leaf's, and under DP attention an expert bank receives every rank's tokens
while the attention beside it receives only its own. Reading the step's count
here while the leaf's own extents said otherwise under-read a bank by the DP
degree.

Routing imbalance is the walk's: a routed tensor's rows are the busiest
expert-parallel rank's (`ExpectedRoutedLoad.busiest_rank_load_factor` times
the mean, `layout.physical`), so a row's FLOPs, activation bytes, the rows its
weight read saturates on and a route exchange's payload are that rank's
already, and nothing here scales them.

A quantization scale belongs to the weight it scales, so a leaf reads its
scales exactly as far as it reads its weights — the Min covers both.

The anchor row in `phases` reads the same law numerically (`RowRoofline`)
from the same components; the expressions on the row (`roofline_expr`) are
what every re-evaluation substitutes into — there is no second statement of
the law. A re-evaluation's numbers are exact rationals until a card's rate
divides them (`_compiled` prints a law's rationals as `Fraction`s), so a
`floor` at another batch is the floor of the exact value, not of its float.

What a row publishes at a point beside those numbers — the term that set
its time (`bound`, an argmax), its arithmetic intensity and the popover
formulas — is rendered from the point's numbers by `point_columns`, which the
walk calls on its own numbers too: one rendering, at any point.

A step adds its rows (`StepSum`): Σ over terms of a structural weight times a
leaf, the leaf the row's `t_op` law for the SOL total and `n_i x T_i` (count
at the point times a measured launch time) for the measured one.
"""
from __future__ import annotations

import json
import math
import operator
import re
from fractions import Fraction
from dataclasses import dataclass
from enum import Enum
from typing import (Any, Callable, Dict, Iterable, List, Mapping, NamedTuple,
                    Optional, Sequence, Tuple)

import sympy as sp
from sympy.core.parameters import evaluate

from backend.arch.dtypes import peak_kind
from backend.arch.eval.dim import point_value, value_of
from backend.arch.eval.storage import is_expert_axis
from backend.arch.eval.transforms import Workload
from backend.arch.law_text import as_written, read_walk_law
from backend.measure.auto_sweep import compile_law
from backend.arch.layout.comm_volume import collective_per_rank_bytes
from .comm_cost import (CollectiveRung, Fabric, Segment, collective_ladder,
                        ladder_segment, price_collective)
from .gpu_specs import NoPeakBandwidth, NoPeakForDtype
from .intensity import arithmetic_intensity, intensity_formula
from .phase_formulas import (_collective_bytes_formula, _t_comm_formula_for_row,
                             _t_compute_formula_for_row,
                             _t_memory_formula_for_row, _t_op_formula_for_row)
from backend.code_trace.memo import memoize

#: The workload a point sets: the batch, the prompt, the output, the cached
#: prefix and the verify width (`transforms.Workload.symbols`), the symbols
#: the walk's laws are written in.
WORKLOAD = tuple(Workload.symbols().values())
#: The batch.
B = Workload.symbols()["batch"]
#: The card a row is priced on.
PEAK_FLOPS = sp.Symbol("PEAK_FLOPS", positive=True)
PEAK_BW = sp.Symbol("PEAK_BW", positive=True)
COMM_S_PER_BYTE = sp.Symbol("COMM_S_PER_BYTE", nonnegative=True)
COMM_LATENCY_S = sp.Symbol("COMM_LATENCY_S", nonnegative=True)
LAUNCH_S = sp.Symbol("LAUNCH_S", nonnegative=True)
#: The arguments every row law is compiled over; a collective's symbols
#: follow them (`comm_names`), as many as its rungs ask.
_ARGS = (*WORKLOAD, PEAK_FLOPS, PEAK_BW, LAUNCH_S)
#: A point's names, in the order the laws are compiled over.
POINT_NAMES = tuple(symbol.name for symbol in WORKLOAD)

#: The names an `sstr`'d expression may read, as the symbols the law was
#: built on. Deliberately closed — rows cross the API boundary, so a
#: re-read trusts no other name.
_ARG_BY_NAME: Dict[str, sp.Symbol] = {arg.name: arg for arg in _ARGS}
#: A row law's argument names, in `_ARGS` order (`LawBook.row_at`).
_ARG_NAMES: Tuple[str, ...] = tuple(arg.name for arg in _ARGS)



def rung_symbols(k: int) -> Tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    """The card symbols of a collective's payload range `k` (the
    framework's `k`-th rung): its bound, its seconds per byte and its fixed
    time."""
    return (sp.Symbol(f"COMM_MAX_BYTES_r{k}", nonnegative=True),
            sp.Symbol(f"COMM_S_PER_BYTE_r{k}", nonnegative=True),
            sp.Symbol(f"COMM_LATENCY_S_r{k}", nonnegative=True))


def row_rungs(row: Mapping[str, Any]) -> Tuple[CollectiveRung, ...]:
    """The framework's dispatch rungs the row's collective is priced over
    (`comm_cost.CollectiveRung`, carried on the row)."""
    return tuple(CollectiveRung.from_dict(r)
                 for r in row.get("comm_rungs") or ())


#: The expressions a row carries, by name (the module docstring states each).
EXPR_NAMES = ("flops", "act_in", "act_out", "convert", "kv_read",
              "weight_read", "metadata", "access", "comm_payload",
              "comm_volume", "comm_bytes", "t_compute", "t_memory", "t_comm",
              "t_launch", "t_op")


class RowTimes(NamedTuple):
    t_compute: float
    t_memory: float
    t_comm: float
    t_op: float
    t_launch: float = 0.0

    @classmethod
    def of_row(cls, row: Mapping[str, Any]) -> "RowTimes":
        """The times a per-op row publishes (`point_columns`)."""
        return cls(float(row["t_compute_s"]), float(row["t_memory_s"]),
                   float(row["t_comm_s"]), float(row["t_op_s"]),
                   float(row["t_launch_s"]))

    @property
    def bound(self) -> str:
        """The term that sets `t_op` — compute, memory, comm or launch (the
        first of them on a tie)."""
        terms = (self.t_compute, self.t_memory, self.t_comm, self.t_launch)
        return BOUNDS[terms.index(self.t_op)]


#: The terms of the row roofline, in tie-break order: `launch` is the
#: floor a launch takes however little it does (`LAUNCH_S`).
BOUNDS = ("compute", "memory", "comm", "launch")


class StoredRead(NamedTuple):
    """The stored bytes one step streams: weights and the scales beside them."""
    weight: int
    metadata: int


def kernel_rows(program: Any, node: Any, shape: Any) -> Any:
    """The rows one of this leaf's kernels runs on, per rank, as an extent.

    Read off the leaf's own per-rank shape — the same `Shape` a row reports
    as `shape_in` — so that nothing re-derives "how many tokens reached this
    leaf" from a model-level count. Which of that operand's axes are ROWS is
    the op's own declaration: every axis whose label its contraction does not
    name (`couples_on`), except the expert axis of a dispatch zone's
    `[n_expert_copies, M, d]`: the rows one expert receives are `M`, so the
    product is the share of its bank one step reaches.

    Both declaration kinds answer one attribute: a registered primitive's
    `OpSpec.shape_in` and a YAML op's `CustomOpSpec.shape_in` (its first
    `signature:` entry). A fused kernel declares nothing itself and answers
    through the member whose operands its shape took — its first origin, the
    way `fusion.FusedProgram.block_of` answers for the block.

    This is only asked of a leaf whose weights are read per row. Anything
    about it that is not declared is refused here rather than guessed: a
    positional convention ("the last axis contracts") is right for today's
    expert GEMMs and silently wrong for any op that contracts elsewhere.
    """
    decl, through = node.declaration, ""
    if decl is None:
        # A leaf with no declaration of its own is a fused kernel
        # (`fusion.FusedNode.declaration`), and its operands are the member's
        # its shape took.
        member = program.by_uid(node.origins[0])
        decl, through = member.declaration, f" (through {member.op})"
    labels = tuple(decl.shape_in or ())
    couples = tuple(decl.couples_on or ())
    axes = tuple(shape.ins[0]) if shape is not None and shape.ins else ()
    contracted = [i for i, label in enumerate(labels) if label in couples]
    if not axes or not labels or len(labels) != len(axes) or not contracted:
        raise ValueError(
            f"{node.display_path} ({node.op}){through}: its weights are read "
            f"per row, so the rows must come from its own operand, and its "
            f"contraction is not declared over that operand — declared axes "
            f"{labels}, contracting {couples or '()'}, walked axes {len(axes)}")
    rows: Any = 1
    for i, axis in enumerate(axes):
        # An expert axis (`n_expert_copies`, asked of the axis's identity) is
        # a batch axis the weight's own experts pair with, not rows: each
        # expert's matrix runs on its own `M` rows.
        if i not in contracted and not is_expert_axis(axis):
            rows = rows * axis
    return rows


class WeightDemand(NamedTuple):
    """One leaf's weight demand, as the PAIR the saturation law multiplies:
    the rows its kernel runs on and what one of those rows asks of its
    resident weights. Produced together by `weight_demand` and never apart —
    the two halves are one statement, and a consumer that assembles them
    itself can pair a per-rank row count with a per-model byte count and get
    a number that is wrong by whatever the two readings differ by.

    `total` is the leaf's whole resident set, the ceiling the demand
    saturates at. `rows` and `read_per_row` carry the walk's own value
    domain — an int, a `Fraction` (a routed bank's mean rows are an
    expectation) or a sympy extent — and the law multiplies them as they are.
    """

    rows: Any
    read_per_row: Any
    total: Any
    #: Whether the `min` can bind: rows reach only part of the resident set
    #: per row (an expert bank's routed rows, a table's lookups), so the
    #: read is the law's and not the leaf's whole weights once.
    saturates: bool = False

    @property
    def asked(self) -> Any:
        """What the step's rows ask of the resident weights:
        `rows x read_per_row`, the operand of the law's saturating `min`
        (`build_row_exprs`)."""
        return self.rows * self.read_per_row

    def read(self, *, metadata: float = 0.0) -> StoredRead:
        """The bytes one step streams, `min(rows x read_per_row, total)`,
        with the scales in the same proportion
        (`RowRoofline.stored_read`)."""
        return RowRoofline.stored_read(rows=self.rows,
                                       read_per_row=self.read_per_row,
                                       total=self.total, metadata=metadata)

    def coverage(self) -> float:
        """The share of the resident set one step reaches, in [0, 1] — the
        same `min` read as a fraction, for a consumer that applies it to
        byte pools the walk prices separately (a borrowed tensor read in its
        reader's share). A leaf with nothing resident is covered whole:
        there is no read to saturate."""
        if not self.total:
            return 1.0
        return float(self.read().weight) / float(self.total)


def weight_demand(program: Any, node: Any, shape: Any, *,
                  active_ratio: float, step_rows: Any,
                  total: Any) -> WeightDemand:
    """The saturation law's two halves for one leaf, stated once.

    A leaf whose active copies are its stored ones streams them once per
    launch, whatever the batch: it runs on the step's rows and reads its
    resident set whole, so the `min` collapses. A leaf whose routing reaches
    only some of its copies (`active:` under `repeat:` — an expert bank) runs
    on the routed rows ONE EXPERT receives, read off the leaf's own per-rank
    shape (`kernel_rows`), and each of those rows demands the whole local
    bank: every expert reads its own share for that row, and the rows are
    counted per expert, so the product is the rows the rank received times
    one expert's bytes.

    `step_rows` is what a leaf that runs on the step's rows runs on (the
    step's tokens); it is deliberately NOT what a routed bank is asked
    about. The step's count is a model-level fact and the rows are the
    leaf's: under DP attention a bank receives every rank's tokens while the
    attention beside it receives only its own, and an encode step's rows are
    its media's, not its requests'.

    Only a leaf with resident weights is asked for its rows — they exist
    here to saturate a weight read, and a weightless leaf in the same bank
    (the activation between its two GEMMs) has none to saturate; its
    `active:` says what its copies do, not what any read of its does.

    `shape` and `step_rows` are numbers at a point or the walk's laws; the
    pair is the same reading either way.
    """
    if active_ratio >= 1.0 or not total:
        return WeightDemand(step_rows, total * active_ratio, total)
    return WeightDemand(kernel_rows(program, node, shape), total, total,
                        saturates=True)


class RowAt(NamedTuple):
    """A row evaluated at a point on a card: its components and times there.
    The walk and a re-evaluation both read it off the row's law at the point
    (`LawBook.row_at`); `point_columns` publishes it."""
    flops: int
    access: int
    act_out: int
    times: RowTimes
    act_in: int = 0
    convert: int = 0
    kv_read: int = 0
    weight_read: int = 0
    metadata: int = 0
    #: One instance's payload.
    comm_payload: float = 0.0
    #: Per-rank interconnect bytes over the row's instances (ring factor in).
    comm_volume: float = 0.0
    #: One instance's priced payload: `comm_payload` where the collective
    #: is priced, else 0.
    comm_bytes: float = 0.0


class RowRoofline:
    @staticmethod
    def stored_read(*, rows: float, read_per_row: float,
                    total: float, metadata: float) -> StoredRead:
        """Stored bytes a leaf actually streams in one step.

        `rows` is what the LEAF's own kernel runs on, never the step's token
        count, and `read_per_row` what one of them demands of its resident
        weights (module docstring).

        The read is `min(rows x read_per_row, total)`. A leaf whose
        weights are streamed once has `read_per_row == total` and `rows >=
        1`, so the min collapses and its figure is exactly its total.

        Metadata: the same covered fraction of the leaf's scales,
        `metadata x weight_covered / total` — the scales of an expert no
        token reaches are no more read than its weights are. At full
        coverage both are returned whole, not re-multiplied."""
        demand = rows * read_per_row
        if demand >= total:
            return StoredRead(int(total), int(metadata))
        return StoredRead(int(demand), int(metadata * demand / total))

    @staticmethod
    def by_bound(rows: Any) -> Dict[str, float]:
        """Σ `t_op` over rows, each row's under the term that set it: parts
        that add up to the rows' `t_sol` (the per-term sums do not — a row
        counts in each of them). The step's sum (`StepSum.by_bound`) over
        the rows' own times."""
        return StepSum.of(list(rows)).by_bound(
            lambda t: RowTimes.of_row(t.row))

    @staticmethod
    def shares(by_bound: Optional[Mapping[str, float]]
               ) -> Optional[Dict[str, float]]:
        """Each part of `by_bound` as a fraction of `t_sol` — their sum, since
        the parts add up to it. None with the parts, or when no time was
        spent."""
        if by_bound is None:
            return None
        total = sum(by_bound.values())
        if not total > 0:
            return None
        return {bound: t / total for bound, t in by_bound.items()}

    @staticmethod
    def labels(times: RowTimes, *, flops: float, access: float
               ) -> Dict[str, Any]:
        """What a per-op row publishes beside its times: the term that set
        `t_op` (None for a row that takes no time — there is no winner) and
        its arithmetic intensity (its popover is `point_display`'s)."""
        return {
            "bound": times.bound if times.t_op > 0 else None,
            "arithmetic_intensity": arithmetic_intensity(flops, access),
        }

    @staticmethod
    def comm(*, op: str, world: int, tensor_bytes: float,
             count: int, base_stride: int, fabric: Fabric,
             rungs: Tuple[CollectiveRung, ...] = (),
             t_sym: Optional[str] = None, t_sub: Optional[str] = None
             ) -> Tuple[float, List[Dict[str, Any]]]:
        """`(t_comm, phase rows)` of a collective whose payload is
        `tensor_bytes`, priced on `fabric` by the algorithm the framework's
        `rungs` give that payload there (`comm_cost.collective_ladder`).
        Zero for a group of one."""
        if world <= 1:
            return 0.0, []
        seg = ladder_segment(collective_ladder(
            op=op, world=world, count_per_step=count, base_stride=base_stride,
            fabric=fabric, rungs=rungs), float(tensor_bytes))
        return price_collective(
            op=op, world=world, tensor_bytes=float(tensor_bytes),
            count_per_step=count, fabric=fabric, base_stride=base_stride,
            algorithm=seg.algorithm, t_sym=t_sym, t_sub=t_sub)


class RowCard(NamedTuple):
    """The card one row is priced on: its own peak (by the row's
    `act_dtype`), the HBM bandwidth, the interconnect and the shortest a
    launch takes (the request's `launch_floor_us`)."""
    peak_flops: Optional[float]
    peak_bw: Optional[float]
    fabric: Fabric
    launch_s: float = 0.0

    @classmethod
    def of(cls, row: Mapping[str, Any], *, peak_flops: Optional[float],
           peak_bw: Optional[float], fabric: Fabric,
           launch_s: float = 0.0) -> "RowCard":
        return cls(row.get("peak_flops_per_sec_for_op") or peak_flops,
                   peak_bw, fabric, launch_s)

    def comm_ladder(self, row: Mapping[str, Any]) -> Tuple[Segment, ...]:
        """The row's collective on this fabric, range by payload range
        (`comm_cost.collective_ladder`): one per rung the row carries, then
        NCCL's past them — each with the seconds `price_collective` charges
        per payload byte and the fixed time over the step. One free range
        off a collective and for a group of one."""
        if not row["is_comm"] or int(row["comm_world"]) <= 1:
            return (Segment("", math.inf, 0.0, 0.0),)
        return collective_ladder(
            op=row["comm_op"], world=int(row["comm_world"]),
            count_per_step=int(row["count_per_step"]),
            base_stride=int(row["comm_base_stride"]), fabric=self.fabric,
            rungs=row_rungs(row))

    def comm_values(self, row: Mapping[str, Any]) -> Tuple[float, ...]:
        """The card's values of the row law's collective symbols, in the
        order `comm_names` names them: the last range's rate and fixed time,
        then each rung's bound, rate and fixed time."""
        ladder = self.comm_ladder(row)
        out = [ladder[-1].s_per_byte, ladder[-1].latency_s]
        for seg in ladder[:-1]:
            out += [seg.max_bytes, seg.s_per_byte, seg.latency_s]
        return tuple(out)


def comm_names(rungs: int) -> Tuple[sp.Symbol, ...]:
    """A row law's collective symbols for a collective of `rungs` rungs, in
    `RowCard.comm_values`' order."""
    return (COMM_S_PER_BYTE, COMM_LATENCY_S,
            *(sym for k in range(rungs) for sym in rung_symbols(k)))


def _exact(v: Any) -> sp.Expr:
    """A component as an exact number or law: an integer stays an Integer, a
    fraction is its exact value (a float's own binary value), a law is
    itself — so the expression carries no rounding of its own."""
    v = value_of(v)
    if type(v) is int:
        return sp.Integer(v)
    # A law reads a symbol (`is_number` is False exactly where it does:
    # laws apply no undefined function), and the test stops at the first.
    if isinstance(v, sp.Expr) and not v.is_number:
        return v
    return sp.Rational(v)


def build_row_exprs(*, count: int, flops: Any, act_in: Any, act_out: Any,
                    convert: Any = 0, kv_read: Any = 0, demand: Any,
                    weight_total: float, metadata_total: float,
                    comm_payload: Any = 0, comm_factor: float = 0.0,
                    comm_priced: bool = False,
                    comm_rungs: int = 0) -> Dict[str, sp.Expr]:
    """One row's law as expressions over the workload and the card symbols
    (the module docstring). `flops`, `act_in`, `act_out`, `convert`,
    `kv_read` and `comm_payload` are one instance's, each the walk's own law
    in the workload's symbols; `count` is the row's instances in the step.
    `kv_read` is the part of `act_in` read from the cache, published beside
    it and not added again. `demand` is what the step's rows ask of the
    leaf's resident weights (`weight_demand`): the leaf's own
    rows times what one of them reads. `comm_priced` says the collective is
    priced (a group of more than one rank); its payload is the row's
    either way. `comm_rungs` is how many rungs the framework's dispatch of
    the collective has: its time is cases over the payload, one per rung,
    then the last range (the module docstring)."""
    n = sp.Integer(int(count))

    def whole(law: Any) -> sp.Expr:
        return n * sp.floor(_exact(law))

    flops_e = sp.floor(n * _exact(flops) + sp.Rational(1, 2))
    act_in_e, act_out_e, convert_e = whole(act_in), whole(act_out), whole(convert)
    act = act_in_e + act_out_e + convert_e
    asked = _exact(demand)
    if weight_total > 0:
        covered = as_written(sp.Min, asked, _exact(weight_total))
        weight_read = sp.floor(covered)
        metadata = sp.floor(covered * _exact(metadata_total)
                            / _exact(weight_total))
    else:
        # No resident weights: nothing saturates; metadata (if any) rides
        # whole.
        weight_read = sp.Integer(0)
        metadata = _exact(metadata_total)
    payload = _exact(comm_payload)
    comm_bytes = payload if comm_priced else sp.Integer(0)
    t_compute = flops_e / PEAK_FLOPS
    t_memory = (weight_read + metadata + act) / PEAK_BW
    t_comm: sp.Expr = sp.Integer(0)
    if comm_priced:
        t_comm = comm_bytes * COMM_S_PER_BYTE + COMM_LATENCY_S
        if comm_rungs:
            t_comm = sp.Piecewise(*(
                (comm_bytes * rate + fixed, comm_bytes <= bound)
                for bound, rate, fixed in map(rung_symbols, range(comm_rungs))),
                (t_comm, True))
    # A row launches a kernel when it moves anything and is not a
    # collective (whose floor is its latency, priced or not).
    launches = payload == 0 and not (flops_e == 0 and act == 0
                                     and weight_read == 0 and metadata == 0)
    t_launch = n * LAUNCH_S if launches else sp.Integer(0)
    return {
        "flops": flops_e,
        "act_in": act_in_e,
        "act_out": act_out_e,
        "convert": convert_e,
        "kv_read": whole(kv_read),
        "weight_read": weight_read,
        "metadata": metadata,
        "access": weight_read + metadata + act,
        "comm_payload": payload,
        "comm_volume": _exact(comm_factor) * payload * n,
        "comm_bytes": comm_bytes,
        "t_compute": t_compute,
        "t_memory": t_memory,
        "t_comm": t_comm,
        "t_launch": t_launch,
        "t_op": as_written(sp.Max, *(_exact(t) for t in (
            t_compute, t_memory, t_comm, t_launch))),
    }


def _exact_number(value: Any) -> Any:
    """A compiled law's exact value (a `Fraction` or an int) as the SymPy
    number it is."""
    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    return sp.Integer(value) if isinstance(value, int) else sp.Float(value)


def comm_factor(op: str, world: int) -> float:
    """Per-rank interconnect bytes per payload byte of one instance of a
    collective (`collective_per_rank_bytes` is linear in the payload); 0
    for a group of one."""
    return collective_per_rank_bytes(op, world, 1.0)


def law_columns(exprs: Mapping[str, sp.Expr], book: "LawBook"
                ) -> Dict[str, Any]:
    """A row's law as the row carries it (`roofline_expr`,
    written into `book`) and whether the row costs nothing — its law states
    no FLOPs, no access bytes and no collective payload, each identically 0
    (a view op, a sampler that reads what is already resident). Stated
    once, where the law is built; every reader of a step asks the row
    (`StepTerm`)."""
    return {"roofline_expr": {name: book.write(exprs[name])
                              for name in EXPR_NAMES},
            "zero_cost": all(exprs[name] == 0
                             for name in ("flops", "access", "comm_payload"))}


#: Every name a law may read, by family: the workload's counts and its
#: routing skew `rho` (`transforms.Workload.symbols`), the card's HBM
#: bandwidth, a peak (`PEAK_FLOPS` on a row; `PEAK_FLOPS_<dtype>` once a
#: step sums rows of several dtypes), a collective's seconds per byte
#: (likewise `COMM_S_PER_BYTE_<collective>`) and fixed time
#: (`COMM_LATENCY_S[_<collective>]`), the card's launch floor (`LAUNCH_S`),
#: and the measured side's
#: launches `n_<k>` and launch time `T_<k>` of a step's term `k`. Closed:
#: laws cross processes and the disk, and a re-read trusts no other name.
_LAW_NAME = re.compile(
    r"(?P<workload>B|s_input|s_output|s_prefix|s_verify|rho)"
    r"|(?P<bw>PEAK_BW)|(?P<peak>PEAK_FLOPS(_\w+)?)"
    r"|(?P<comm>COMM_S_PER_BYTE(_\w+)?)|(?P<latency>COMM_LATENCY_S(_\w+)?)"
    r"|(?P<bound>COMM_MAX_BYTES(_\w+)?)"
    r"|(?P<launch_floor>LAUNCH_S)|(?P<launches>n_\d+)"
    r"|(?P<launch_time>T_\d+)")


def law_symbol(name: str) -> sp.Symbol:
    """The symbol a law reads under `name`, with the assumptions the law
    was built on — or a refusal for a name outside the vocabulary."""
    m = _LAW_NAME.fullmatch(name)
    if m is None:
        raise ValueError(
            f"a law may not read {name!r}: its names are {', '.join(POINT_NAMES)}, "
            "PEAK_BW, PEAK_FLOPS[_<dtype>], COMM_S_PER_BYTE[_<collective>], "
            "COMM_LATENCY_S[_<collective>], COMM_MAX_BYTES[_<collective>], "
            "LAUNCH_S, n_<k>, T_<k>")
    if m.group("workload"):
        return _WORKLOAD_BY_NAME[name]
    if (m.group("comm") or m.group("latency") or m.group("bound")
            or m.group("launch_floor")
            or m.group("launches") or m.group("launch_time")):
        return sp.Symbol(name, nonnegative=True)
    return sp.Symbol(name, positive=True)


_WORKLOAD_BY_NAME = {symbol.name: symbol for symbol in WORKLOAD}


def floor_of(name: str) -> Optional[str]:
    """Which of the two floors a law's name is a time of, by its family in
    the vocabulary (`_LAW_NAME`): "latency" for a collective's fixed time
    (`COMM_LATENCY_S[_<collective>]`, link traversals times the link
    latency), "launch_floor" for `LAUNCH_S`, None for every other name."""
    m = _LAW_NAME.fullmatch(name)
    if m is None:
        return None
    for family in ("latency", "launch_floor"):
        if m.group(family):
            return family
    return None


def read_row_law(text: str) -> sp.Expr:
    """Law text (`sp.sstr` of a row's or a step's law) back as the law: the
    one reader, over the closed vocabulary (`law_symbol`). The operators are
    the law language's plus `Piecewise`, which a per-bound sum is written
    with."""
    return read_walk_law(text, symbol=law_symbol)


class LawBook:
    """A cell's laws, each compiled once: the laws its rows carry — the
    walk's expressions, kept as they are — every point of the cell
    substituting into the one compiled form (`auto_sweep.compile_law`).
    Owned by the cell whose rows it holds (`step_eval.Cell.laws`), or by a
    reader of laws from elsewhere, for as long as it reads them; such a
    reader turns law text back into the law through `read`, the one reader
    of law text."""

    def __init__(self) -> None:
        self._compiled: Dict[Tuple[Tuple[sp.Expr, ...], Tuple[str, ...]],
                             Callable[..., Any]] = {}
        #: Each row law's compiled form (`row_at`), by the law's own object
        #: — held beside it, so the object stays the one the entry names: a
        #: cell's rows carry one law object at every point, and asking for
        #: it by identity spares hashing its twelve expressions per point.
        self._row_laws: Dict[int, Tuple[Mapping[str, Any], Callable[..., Any]]] = {}

    @staticmethod
    def write(extent: Any) -> sp.Expr:
        """`extent`'s law as a row carries it: its value half, exact
        (`_exact`)."""
        return _exact(extent)

    def axes(self, axes: Iterable[Any]) -> List[sp.Expr]:
        """Each extent of `axes` as its law (`write`)."""
        return [self.write(axis) for axis in axes]

    @staticmethod
    def read(text: str) -> sp.Expr:
        """Law text (`sp.sstr` of a law) back as the law, over the closed
        vocabulary (`read_row_law`): the one way law text becomes a law."""
        return read_row_law(text)

    def compiled(self, laws: Tuple[sp.Expr, ...],
                 names: Tuple[str, ...]) -> Callable[..., Any]:
        """`laws` as one function of `names` (in that order), returning
        their values in order, exactly. The names are checked against the
        closed vocabulary (`law_symbol`); a law that reads a name outside
        `names` is refused."""
        key = (laws, names)
        fn = self._compiled.get(key)
        if fn is None:
            for name in names:
                law_symbol(name)
            stray = sorted({sym.name for expr in laws
                            for sym in expr.free_symbols} - set(names))
            if stray:
                raise ValueError(f"laws {list(laws)!r} read {stray}; they "
                                 f"may read only {sorted(names)}")
            fn = self._compiled[key] = compile_law(laws, names)
        return fn

    def evaluate(self, law: sp.Expr, values: Mapping[str, Any]) -> Any:
        """A law at `values` (name → number): a step's law at a point, its
        card binding and batch (and, on the measured side, its launches
        and launch times) substituted. Exact where no float enters."""
        names = tuple(sorted(values))
        return self.compiled((law,), names)(
            *(values[name] for name in names))[0]

    def row_at(self, row: Mapping[str, Any], point: Mapping[str, Any],
               card: RowCard) -> RowAt:
        """The row's law (`roofline_expr`) at a point (its workload counts
        by name) on `card` — the whole of a re-evaluation.

        An absent rate is infinite, so its term is free — which is the
        reading of a caller that passed NO CARD and therefore asked for
        bytes and FLOPs, not for times (`phases`, §3.8). It is not a card
        with a blank column: a card always carries an HBM bandwidth
        (`gpu_specs.check_memory_bandwidth` refuses a sheet row without
        one at load) and a dtype column it lacks is refused where a card is
        in hand (`NoPeakForDtype`, `NoPeakBandwidth`), never priced free."""
        wire = row["roofline_expr"]
        comm = card.comm_values(row)
        args = (*(point_value(point[name]) for name in POINT_NAMES),
                card.peak_flops if card.peak_flops and card.peak_flops > 0
                else math.inf,
                card.peak_bw if card.peak_bw and card.peak_bw > 0
                else math.inf,
                card.launch_s, *comm)
        entry = self._row_laws.get(id(wire))
        if entry is None or entry[0] is not wire:
            names = (*_ARG_NAMES, *(sym.name for sym in comm_names(
                (len(comm) - 2) // 3)))
            entry = self._row_laws[id(wire)] = (wire, self.compiled(
                tuple(wire[name] for name in EXPR_NAMES), names))
        at = dict(zip(EXPR_NAMES, entry[1](*args), strict=True))
        times = RowTimes(float(at["t_compute"]), float(at["t_memory"]),
                         float(at["t_comm"]), float(at["t_op"]),
                         float(at["t_launch"]))
        return RowAt(flops=int(at["flops"]), access=int(at["access"]),
                     act_out=int(at["act_out"]), times=times,
                     act_in=int(at["act_in"]),
                     convert=int(at["convert"]), kv_read=int(at["kv_read"]),
                     weight_read=int(at["weight_read"]),
                     metadata=int(at["metadata"]),
                     comm_payload=float(at["comm_payload"]),
                     comm_volume=float(at["comm_volume"]),
                     comm_bytes=float(at["comm_bytes"]))


def evaluate_law(text: str, values: Mapping[str, Any]) -> Any:
    """Law text read once at `values` (`LawBook.evaluate`): a one-off
    reading. A reader of many points holds a book across them."""
    book = LawBook()
    return book.evaluate(book.read(text), values)


def row_law(row: Mapping[str, Any]) -> Dict[str, Any]:
    """The part of a per-op row a re-evaluation reads: its law
    (`roofline_expr`), the collective it prices (`LawBook.row_at`), its own
    peak and the `act_dtype` another card re-reads that peak by
    (`RowCard.of`). What a pipeline step carries for the stages whose full
    rows it does not ship."""
    return {
        "uid": row.get("uid"),
        "roofline_expr": row["roofline_expr"],
        "zero_cost": row["zero_cost"],
        # The point the row's numbers are at: another point is this one
        # with its counts moved.
        "point": row["point_law"]["point"],
        "is_comm": row["is_comm"],
        "comm_op": row["comm_op"],
        "comm_world": row["comm_world"],
        "count_per_step": row["count_per_step"],
        "comm_base_stride": row["comm_base_stride"],
        "comm_rungs": row.get("comm_rungs") or [],
        "peak_flops_per_sec_for_op": row.get("peak_flops_per_sec_for_op"),
        "act_dtype": row.get("act_dtype"),
    }


def bare_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """A per-op row as a reading with no breakdown carries it: without its
    formula popovers, a collective's per-phase rows, link and buffer
    shapes, and a fused kernel's absorbed parts' output shapes — what an
    estimate's breakdown draws and nothing else reads. The walk's rows
    (`phases.compute_per_op_timing`) and a row read at another point
    (`batch_sweep.rows_at`) drop the same columns."""
    out = {key: value for key, value in row.items()
           if not key.endswith("_formula")
           and key not in ("comm_phases", "comm_bw_bytes_per_sec",
                           "comm_bw_label", "comm_collective_in_shape",
                           "comm_collective_out_shape")}
    badge = out.get("source_badge") or {}
    if badge.get("absorbed"):
        out["source_badge"] = {**badge, "absorbed": [
            {k: v for k, v in part.items() if k != "shape_out"}
            for part in badge["absorbed"]]}
    return out


#: What publishes a declared column: a per-op row, or a phase of the step.
ROW, PHASE = "row", "phase"


class Share(Enum):
    """Whose share of a deployment a published number is.

    `RANK` is one GPU's, and the deployment's is the sum over ranks. The
    payload spells the notion two ways (`_per_rank` on rows and `_per_gpu` on
    result fields); the declaration says RANK for both and no published name
    changes. `GROUP` is one data-parallel replica group's — neither a rank nor
    the whole mesh (`phase_throughput.per_group`). `DEPLOYMENT` is the whole
    mesh's already. `NONE` is a number that is nobody's share: a duration is
    the critical path every rank shares; a ratio is dimensionless.

    A consumer may sum across GPUs exactly where `share is RANK`. Whether a
    number adds across repeated launches is a different question, declared by
    `Column.additive`; the two are deliberately separate axes.
    """

    RANK = ("rank", ("_per_rank", "_per_gpu"))
    GROUP = ("group", ("_per_group",))
    DEPLOYMENT = ("deployment", ("_total",))
    NONE = ("none", ())

    def __init__(self, word: str, suffixes: Tuple[str, ...]) -> None:
        self.word = word
        self.suffixes = suffixes


def _share_implied_by(name: str) -> Optional[Share]:
    """The share a column's NAME states, or None where it states none.

    Nested keys are read by their leaf (`throughput.per_gpu` is `_per_gpu`).
    The name is an oracle on a hand-declared share, not the declaration's
    source: a name saying `_per_rank` while its producer declares the whole
    deployment fails at import.
    """
    flat = name.replace(".", "_")
    for share in Share:
        if any(flat.endswith(suffix) for suffix in share.suffixes):
            return share
    return None


@dataclass(frozen=True)
class Column:
    """One published number, with the properties a consumer reads.

    `published_by` says whether a per-op row or a step phase publishes the
    column; a name may be both (the four times are), but each publisher has one
    record. `source` names the `RowAt` field a row column reads, replacing the
    old positional zip. `share` says whose slice of the deployment the number
    is. `additive` says whether repeated launches of the same row add. Neither
    field tries to say a unit, a token count or a schedule granularity: those
    are laws and dimensions elsewhere, not properties of a published column.
    """
    name: str
    published_by: str = ROW
    source: str = ""
    share: Share = Share.NONE
    additive: bool = False
    on_module_table: bool = True

    def __post_init__(self) -> None:
        if self.published_by not in (ROW, PHASE):
            raise ValueError(f"column {self.name!r}: {self.published_by!r} "
                             "publishes no column; a column is a row's or a "
                             "phase's")
        if not isinstance(self.share, Share):
            raise ValueError(f"column {self.name!r}: share is a Share, not "
                             f"{self.share!r}")
        if self.source and self.published_by != ROW:
            raise ValueError(f"column {self.name!r}: only a row column reads "
                             f"a point attribute; this is a {self.published_by}'s")
        self._hold_source()
        self._hold_name()

    def _hold_source(self) -> None:
        """A row column reads a field `RowAt` really has.

        The old tuple said this only by position, so a moved name moved silently
        to another value. The declaration refuses that at import.
        """
        if not self.source:
            return
        head, _, rest = self.source.partition(".")
        if head not in RowAt._fields:
            raise ValueError(f"column {self.name!r} reads {self.source!r}, "
                             "which is no field of the row at the point; it "
                             f"has {list(RowAt._fields)}")
        if rest and rest not in RowTimes._fields:
            raise ValueError(f"column {self.name!r} reads {self.source!r}, "
                             "which is no field of the row's times; they have "
                             f"{list(RowTimes._fields)}")

    def _hold_name(self) -> None:
        stated = _share_implied_by(self.name)
        if stated is not None and stated is not self.share:
            raise ValueError(f"column {self.name!r} is declared {self.share.word!r} "
                             f"but its name states {stated.word!r}; declare the "
                             "share the name states, or rename the column")


#: Every column whose properties a consumer reads, at its producer. A column
#: absent here is still published; it is additive on neither axis and renders on
#: the module table. The row's own components and times come first because that
#: is `point_columns`' published order.
COLUMNS: Tuple[Column, ...] = (
    Column("flops_per_rank", source="flops", additive=True, share=Share.RANK),
    Column("access_bytes_per_rank", source="access", additive=True,
           share=Share.RANK),
    Column("act_in_bytes_per_rank", source="act_in", additive=True,
           share=Share.RANK, on_module_table=False),
    Column("kv_read_bytes_per_rank", source="kv_read", additive=True,
           share=Share.RANK, on_module_table=False),
    Column("act_out_bytes_per_rank", source="act_out", additive=True,
           share=Share.RANK, on_module_table=False),
    Column("convert_bytes_per_rank", source="convert", additive=True,
           share=Share.RANK),
    Column("weight_read_bytes_per_rank", source="weight_read", additive=True,
           share=Share.RANK, on_module_table=False),
    Column("metadata_bytes_per_rank", source="metadata", additive=True,
           share=Share.RANK, on_module_table=False),
    Column("comm_bytes_per_rank", source="comm_volume", additive=True,
           share=Share.RANK),
    Column("comm_tensor_bytes", source="comm_payload", additive=True,
           on_module_table=False),
    Column("t_compute_s", source="times.t_compute", additive=True),
    Column("t_memory_s", source="times.t_memory", additive=True),
    Column("t_comm_s", source="times.t_comm", additive=True),
    Column("t_launch_s", source="times.t_launch", additive=True),
    Column("t_sol_s", source="times.t_op", additive=True),
    Column("t_op_s", source="times.t_op", additive=True),

    # The phase account is one rank's (`_phase_inputs` says so), even though
    # its names carry no suffix; the declaration is the hand-read producer side
    # of the suffix oracle.
    Column("flops_per_step", published_by=PHASE, additive=True,
           share=Share.RANK),
    Column("mem_read_bytes", published_by=PHASE, additive=True,
           share=Share.RANK),
    Column("mem_write_bytes", published_by=PHASE, additive=True,
           share=Share.RANK),
    Column("total_bytes", published_by=PHASE, additive=True, share=Share.RANK),
    Column("t_compute_s", published_by=PHASE, additive=True),
    Column("t_memory_s", published_by=PHASE, additive=True),
    Column("t_comm_s", published_by=PHASE, additive=True),
    Column("t_sol_s", published_by=PHASE, additive=True),
    Column("t_launch_s", published_by=PHASE),
    Column("throughput.per_group", published_by=PHASE, share=Share.GROUP),
    Column("throughput.cluster", published_by=PHASE,
           share=Share.DEPLOYMENT),
    Column("throughput.per_gpu", published_by=PHASE, share=Share.RANK),

    # Row facts that stay on `per_op_timing`: the module table would send the
    # same leaf twice in one response.
    *(Column(name, on_module_table=False,
             share=(Share.RANK if name.endswith(("_per_rank", "_per_gpu"))
                    else Share.NONE)) for name in (
        "node_dims", "shape_in_exact", "shape_in_dims", "shape_out_dims",
        "weight_shards", "shape_json", "dtype_env_json", "shape_weight",
        "shape_weight_dims", "weight_bytes_per_rank",
        "weight_bytes_total_per_rank",
        "weight_bytes_per_token_active_per_rank", "metadata_bytes_total_per_rank",
        "peak_flops_per_sec_for_op", "act_dtype", "moe_block_member",
        "active_ratio_per_token", "derived_group", "count_per_step",
        "parent_aggregation_id", "comm_op", "comm_base_stride",
        "comm_collective_in_shape", "comm_collective_out_shape",
        "comm_payload_shape", "comm_payload_role",
        "comm_payload_dtype_bytes", "comm_payload_scale", "point_law",
        "zero_cost",
    )),
)


def columns(*, published_by: str = ROW, additive: Optional[bool] = None,
            on_module_table: Optional[bool] = None,
            source: Optional[bool] = None) -> Tuple[Column, ...]:
    """The declared columns with the properties asked, in published order."""
    return tuple(
        column for column in COLUMNS
        if column.published_by == published_by
        and additive in (None, column.additive)
        and on_module_table in (None, column.on_module_table)
        and source in (None, bool(column.source)))


def column_names(**asked: Any) -> Tuple[str, ...]:
    """The names of `columns(**asked)` — the only list of column names."""
    return tuple(column.name for column in columns(**asked))


def _hold_columns() -> None:
    seen: set[tuple[str, str]] = set()
    for column in COLUMNS:
        owner = (column.published_by, column.name)
        if owner in seen:
            raise ValueError(f"column {column.name!r} is declared twice for "
                             f"one {column.published_by}")
        seen.add(owner)


_hold_columns()

#: The row columns that add across repeated launches, derived once from the
#: declaration for callers that still ask by the old name.
POINT_ADDITIVE_COLUMNS = column_names(published_by=ROW, additive=True)

#: Each row column's reader, compiled from the declaration rather than zipped
#: from a positional tuple.
_POINT_READERS: Tuple[Tuple[str, Callable[[RowAt], Any]], ...] = tuple(
    (column.name, operator.attrgetter(column.source))
    for column in columns(source=True))


def point_columns(at: RowAt, row: Mapping[str, Any], card: RowCard, *,
                  breakdown: bool, t_sym: Optional[str] = None,
                  t_sub: Optional[str] = None) -> Dict[str, Any]:
    """Every column a per-op row publishes that moves with the point or the
    card, from the row at that point (`at`) and the facts that do not move
    (`row`: `count_per_step`, `act_dtype`, `is_comm`,
    `comm_op`, `comm_world`, `comm_base_stride`): its components, its times,
    the term that set the time and its arithmetic intensity — and, with
    `breakdown`, what the row shows beside them: its popovers, printed from
    the numbers, and a collective's per-phase rows on the card's fabric (how
    many phases, and on which link, is the fabric's topology;
    `RowRoofline.comm`), whose slowest phase names the bottleneck fabric.
    `t_sym` / `t_sub` are the collective payload's symbolic and substituted
    lines. The walk publishes its row through this, and a re-evaluation its
    re-read row: one rendering, the numbers the same with or without the
    breakdown."""
    t = at.times
    out = {
        **{name: read(at) for name, read in _POINT_READERS},
        **RowRoofline.labels(t, flops=at.flops, access=at.access),
    }
    if not breakdown:
        return out
    cnt = int(row["count_per_step"])
    is_comm = bool(row["is_comm"])
    world = int(row["comm_world"] or 0)
    phases: List[Dict[str, Any]] = []
    if is_comm and world > 1:
        phases = RowRoofline.comm(
            op=row["comm_op"], world=world, tensor_bytes=at.comm_bytes,
            count=cnt, base_stride=int(row["comm_base_stride"]),
            fabric=card.fabric, rungs=row_rungs(row),
            t_sym=t_sym, t_sub=t_sub)[1]
    bw, label = 0.0, ""
    if phases:
        bottleneck = max(phases, key=lambda ph: ph["t_phase_s"])
        bw = (float(bottleneck["bw_bytes_per_sec"])
              if bottleneck["bw_bytes_per_sec"] else 0.0)
        label = bottleneck["fabric_label"]
    comm_formula: Dict[str, Any] = (
        _collective_bytes_formula(op=row["comm_op"], world=world,
                                  tensor_bytes=at.comm_payload, cnt=cnt,
                                  phases=phases)
        if is_comm and world > 1 else {})
    return {
        **out,
        "arithmetic_intensity_formula": intensity_formula(at.flops, at.access),
        "comm_bw_bytes_per_sec": bw,
        "comm_bw_label": label,
        # One phase is the row itself; the breakdown lists two or more.
        "comm_phases": phases if len(phases) > 1 else [],
        "comm_bytes_formula": comm_formula,
        "t_comm_formula": (_t_comm_formula_for_row(
            t_comm_s=t.t_comm, comm_bytes_per_rank=at.comm_volume,
            comm_bw_bytes_per_sec=bw, comm_bw_label=label, comm_world=world,
            comm_bytes_formula=comm_formula, comm_phases=phases, cnt=cnt,
            link_latency_s=card.fabric.step_latency)
            if is_comm else None),
        "t_op_formula": _t_op_formula_for_row(
            t_compute_s=t.t_compute, t_memory_s=t.t_memory,
            flops_per_rank=at.flops, access_per_rank=at.access,
            # The peak the row's t_compute was priced at (this leaf's
            # dtype), so the substituted line reproduces the t_compute cell.
            peak_flops_per_sec=card.peak_flops,
            peak_hbm_bw_bytes_per_sec=card.peak_bw,
            # Per-step component bytes (post-MoE-saturation, per rank): the
            # note's `weights + act_in + act_out + metadata` sums to the
            # access the substituted line divides by peak_bw.
            weight_bytes_step_pr=at.weight_read,
            act_in_bytes_step_pr=at.act_in,
            act_out_bytes_step_pr=at.act_out,
            meta_bytes_step_pr=at.metadata,
            t_comm_s=t.t_comm, comm_bytes_per_rank=at.comm_volume,
            comm_bw_bytes_per_sec=bw, comm_bw_label=label,
            comm_bytes_formula=comm_formula, t_launch_s=t.t_launch,
            launch_floor_s=card.launch_s, cnt=cnt),
        "t_compute_formula": _t_compute_formula_for_row(
            t_compute_s=t.t_compute, flops_per_rank=at.flops,
            peak_flops_per_sec=card.peak_flops,
            act_dtype=row["act_dtype"] or ""),
        "t_memory_formula": _t_memory_formula_for_row(
            t_memory_s=t.t_memory, access_per_rank=at.access,
            peak_hbm_bw_bytes_per_sec=card.peak_bw,
            weight_bytes_step_pr=at.weight_read,
            act_in_bytes_step_pr=at.act_in,
            act_out_bytes_step_pr=at.act_out,
            meta_bytes_step_pr=at.metadata),
    }


# ---------------------------------------------------------------------------
# A step: its rows added, once, with the leaf swappable.
# ---------------------------------------------------------------------------


def peak_symbol(act_dtype: Optional[str]) -> sp.Symbol:
    """The peak a row of `act_dtype` is priced at, as a step's law names it;
    a row with no dtype runs no FLOPs it is priced for and reads the card's
    own (`PEAK_FLOPS`)."""
    return law_symbol(f"PEAK_FLOPS_{act_dtype}" if act_dtype else "PEAK_FLOPS")


def collective_symbol(row: Mapping[str, Any],
                      rate: sp.Symbol = COMM_S_PER_BYTE) -> sp.Symbol:
    """One of the row law's collective symbols (`comm_names`: a rate, a
    fixed time or a range's bound), as a step's law names it: one symbol per
    (op, group size, instances, node stride) — everything
    `RowCard.comm_ladder` prices besides the fabric and the framework's
    rungs, which one step's rows share."""
    if not row["is_comm"] or int(row["comm_world"]) <= 1:
        return rate
    base, _r, k = rate.name.rpartition("_r")
    stem, suffix = (base, f"_r{k}") if _r and k.isdigit() else (rate.name, "")
    return law_symbol(
        f"{stem}_{row['comm_op']}_{int(row['comm_world'])}_"
        f"{int(row['count_per_step'])}_{int(row['comm_base_stride'])}{suffix}")


def _row_comm_names(row: Mapping[str, Any]) -> Tuple[sp.Symbol, ...]:
    """The collective symbols the row's law reads (`comm_names`)."""
    priced = row["is_comm"] and int(row["comm_world"]) > 1
    return comm_names(len(row_rungs(row)) if priced else 0)


def lifted_law(row: Mapping[str, Any], name: str) -> sp.Expr:
    """The row's `name` law with its card symbols named for the step: its
    peak by its dtype, its collective by its signature (`peak_symbol`,
    `collective_symbol`), so rows of several dtypes and collectives add."""
    with evaluate(False):
        return row["roofline_expr"][name].xreplace({
            PEAK_FLOPS: peak_symbol(row.get("act_dtype")),
            **{sym: collective_symbol(row, sym)
               for sym in _row_comm_names(row)},
        })


def card_binding(rows: Iterable[Mapping[str, Any]], *,
                 peak_flops: Optional[float], peak_bw: Optional[float],
                 fabric: Fabric, launch_s: float = 0.0) -> Dict[str, float]:
    """The card as a step's law reads it — `{symbol name: value}` for every
    peak and collective `rows` name (`lifted_law`), and the HBM bandwidth.
    `rows` carry their own peak on this card (`peak_flops_per_sec_for_op`,
    restamped per card) and `peak_flops` is the card's own for a row with
    none, as `LawBook.row_at` reads them. A missing rate is infinite: its
    term is free, which is what a caller holding no card asked for
    (`LawBook.row_at`) — a card's own missing column is refused where the
    card is in hand, not priced free."""
    def rate(v: Optional[float]) -> float:
        return float(v) if v and v > 0 else math.inf
    out: Dict[str, float] = {PEAK_BW.name: rate(peak_bw),
                             PEAK_FLOPS.name: rate(peak_flops),
                             COMM_S_PER_BYTE.name: 0.0,
                             COMM_LATENCY_S.name: 0.0,
                             LAUNCH_S.name: float(launch_s)}
    for r in rows:
        card = RowCard.of(r, peak_flops=peak_flops, peak_bw=peak_bw,
                          fabric=fabric, launch_s=launch_s)
        comm = zip((collective_symbol(r, sym) for sym in _row_comm_names(r)),
                   card.comm_values(r), strict=True)
        for sym, value in ((peak_symbol(r.get("act_dtype")), rate(card.peak_flops)),
                           *comm):
            if out.setdefault(sym.name, value) != value:
                raise ValueError(
                    f"{sym.name} is {out[sym.name]} on one row and {value} on "
                    "another; a step's law names one value for it")
    return out


@dataclass(frozen=True)
class StepTerm:
    """One row of a step's sum: the row, the structural weight the step
    runs it with beyond its own count (a speculative draft's forwards), and
    where it sits — its pipeline stage (None without pp) and whether it is
    a draft forward's."""
    row: Mapping[str, Any]
    weight: int = 1
    stage: Optional[int] = None
    draft: bool = False

    def where(self) -> Dict[str, Any]:
        return {"uid": self.row.get("uid"), "stage": self.stage,
                "draft": self.draft}

    def zero_cost(self) -> bool:
        """The row moves nothing itself (`law_columns`). Such a row costs 0
        on both tracks; it is never missing a measurement."""
        return bool(self.row["zero_cost"])


@dataclass(frozen=True)
class FloorForm:
    """A step's SOL time as a function of its two floors alone — one link
    traversal's latency `α` and the shortest a launch takes `L` — with
    every other part of each row's law read at the step's point on its card:

        t(α, L) = K + Σ_k max(a_k, b_k + c_k α, n_k L)

    `a` is the row's compute or memory time, whichever binds; `b` its
    collective's bandwidth time and `c` the link traversals it waits on
    over the step (`COMM_LATENCY_S` is `c α`); `n` its launches
    (`t_launch = n L`). A row whose time moves with neither floor adds
    into `K`. Each coefficient is the row's law read twice
    (`LawBook.row_at`): on the card with both floors 0 and with both 1, so
    the form is the law's and not a second statement of it. Another `α`
    or `L` is then a sum, not a walk.

    The step's rows once and a counted draft's one-forward rows `forwards`
    times, as the step adds them (`StepSum.of(rows, draft_rows=…,
    draft_forwards=…)`): a draft forward's term is the same form, weighted.
    A pipelined step is the caller's to refuse."""
    constant: float
    terms: Tuple[Tuple[float, float, float, float], ...]
    #: Each term's index among the step's terms (`StepSum.terms`, the
    #: measured formula's), when the form keeps them apart (`termwise`);
    #: empty for a `collapsed` form, whose terms are rays, not step terms.
    index: Tuple[int, ...] = ()
    #: How many step terms a `termwise` form is over (its all-zero ones
    #: left out of `terms`); 0 for a `collapsed` form.
    width: int = 0

    @classmethod
    def of(cls, rows: Iterable[Mapping[str, Any]], *, book: "LawBook",
           peak_flops: Optional[float], peak_bw: Optional[float],
           fabric: Fabric, draft_rows: Iterable[Mapping[str, Any]] = (),
           draft_forwards: int = 0, gpu_sku: str = "") -> "FloorForm":
        """The form of the step `rows` (and `draft_forwards` of a draft's
        one-forward `draft_rows`) add up to, each row read at its own point
        (`point_law.point`) on the card `peak_*` and `fabric` describe: the
        step's `CardForm` bound to that card, each row at the peak it
        carries (`peak_flops_per_sec_for_op`, restamped per card) and
        `peak_flops` for a row with none, as `RowCard.of` reads them.
        `gpu_sku` names that card in a refusal (`CardForm.bind`)."""
        rows, draft_rows = list(rows), list(draft_rows)
        peaks: Dict[str, float] = {}
        for r in (*rows, *draft_rows):
            own = r.get("peak_flops_per_sec_for_op")
            dtype = r.get("act_dtype") or ""
            if own and dtype and peaks.setdefault(dtype, float(own)) != float(own):
                raise ValueError(
                    f"{dtype} peaks at {peaks[dtype]} on one row and {own} on "
                    "another; a step's card names one peak per dtype")
        default = float(peak_flops) if peak_flops and peak_flops > 0 else None
        form = CardForm.of(rows, book=book, fabric=fabric,
                           draft_rows=draft_rows, draft_forwards=draft_forwards)
        return form.bind({dt: peaks.get(dt, default) for dt in form.dtypes},
                         peak_bw=peak_bw, fabric=fabric, gpu_sku=gpu_sku)

    def at(self, link_latency_s: float, launch_floor_s: float) -> float:
        """The step's SOL time at these floors (seconds)."""
        return self.constant + sum(
            max(a, b + c * link_latency_s, n * launch_floor_s)
            for a, b, c, n in self.terms)

    def terms_at(self, link_latency_s: float, launch_floor_s: float,
                 steps: Iterable[int]) -> float:
        """The SOL of the step terms `steps` names (indices into the
        step's terms, `StepSum.terms`) at these floors: what the partial
        track swaps out for their measured share (`auto_sweep.blend`).
        Only a `termwise` form can say which of its terms a step term is,
        and an index past its width is a step this form is not of."""
        if not self.width:
            raise ValueError("a collapsed form cannot tell its step terms apart")
        wanted = set(steps)
        stray = sorted(k for k in wanted if not 0 <= k < self.width)
        if stray:
            raise ValueError(f"step terms {stray[:5]} are not among this "
                             f"form's {self.width}")
        return sum(max(a, b + c * link_latency_s, n * launch_floor_s)
                   for k, (a, b, c, n) in zip(self.index, self.terms, strict=True)
                   if k in wanted)

    @classmethod
    def termwise(cls, terms: Iterable[Tuple[int, Tuple[float, float, float, float]]],
                 width: int) -> "FloorForm":
        """The form of a step's `width` terms, `(step-term index,
        coefficients)` pairs each kept apart under its index so a reader
        can price any of them alone (`terms_at`); terms with every
        coefficient 0 are left out."""
        kept = [(k, t) for k, t in terms if any(t)]
        if any(not 0 <= k < width for k, _t in kept):
            raise ValueError(f"a step term past the step's {width}")
        return cls(0.0, tuple(t for _k, t in kept), tuple(k for k, _t in kept),
                   int(width))

    @classmethod
    def collapsed(cls, constant: float,
                  terms: Iterable[Tuple[float, float, float, float]]
                  ) -> "FloorForm":
        """The form of `constant` and `terms`, every set of terms equal up
        to a positive weight collapsed into one: `w x max(a, b + cα, nL)`
        is `max(wa, wb + wcα, wnL)`, so terms on one ray add. Exact: the
        ray of a term is its coefficients over its first nonzero one, as
        rationals (`Fraction` of each float), so only terms that are
        exactly proportional share one. The one constructor a builder
        calls when its terms need not be told apart (`CardForm.bind`); a
        form whose terms a reader prices one by one is `termwise`."""
        merged: Dict[Tuple[Fraction, ...], List[float]] = {}
        for term in terms:
            pivot = Fraction(next((x for x in term if x), 0.0))
            if not pivot:
                continue
            ray = tuple(Fraction(x) / pivot for x in term)
            if ray in merged:
                merged[ray] = [s + x for s, x in zip(merged[ray], term,
                                                    strict=True)]
            else:
                merged[ray] = list(term)
        return cls(float(constant),
                   tuple((a, b, c, n) for a, b, c, n in merged.values()))

    def to_json(self) -> Dict[str, Any]:
        """Columnar: the constant and one array per coefficient."""
        return {"constant": self.constant,
                **{name: [term[i] for term in self.terms]
                   for i, name in enumerate("abcn")},
                **({"index": list(self.index), "width": self.width}
                   if self.width else {})}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "FloorForm":
        return cls(float(d["constant"]), tuple(
            (float(a), float(b), float(c), float(n))
            for a, b, c, n in zip(d["a"], d["b"], d["c"], d["n"],
                                  strict=True)),
            tuple(int(k) for k in d.get("index") or ()),
            int(d.get("width") or 0))


@dataclass(frozen=True)
class CardForm:
    """A step's SOL time as a function of the CARD and the two floors — one
    link traversal's latency `α` and the shortest a launch takes `L`:

        t(card, α, L) = Σ_r w_r x max(flops_r / peak(card, dtype_r),
                                      access_r / bw(card),
                                      payload_r x s_per_byte(card) + hops_r(card) x α,
                                      n_r x L)

    `FloorForm` is this form bound to one card (`bind`): it fixes the card
    and leaves the floors. A CardForm leaves both. Each term keeps one
    row's card-independent quantities at the step's point — its FLOPs, its
    access bytes, its launches and its collective's payload, all for one
    forward — and the weight `w` the step runs the row with (a counted
    draft's forwards), with the row's dtype and its collective's signature
    in the side tables (`dtypes`, `sigs`). The card enters only at `bind`:
    the peak is the card's column of the row's dtype, and the collective's
    payload range is decided on that card (`collective_ladder`) on ONE
    forward's payload, as the row's law decides it — an inadmissible rung
    collapses there exactly as on a fresh walk.

    Built in the walk's worker from the rows' laws (`LawBook.row_at`: the
    components are card-independent; the launch count is the floors-0/1
    read's difference), the rows once and a counted draft's one-forward
    rows `draft_forwards` times, as the step adds them (`StepSum`). A form
    is as valid on another card as the walk's rows are: within one
    `card_class` (`framework_bindings.card_class`). A card whose dispatch or dtype
    columns differ walks differently, and no form carries that.

    This form is the row laws with the card's symbols not yet substituted,
    written as plain data — nothing more. Keeping the laws themselves and
    substituting the card later says the same thing and is what `row_at`
    does (the two agree to float noise, and the form is ~50x the cheaper
    read), but a law is SymPy: a step's rows carry thousands of
    expressions, which no cache can hold and which must not cross the
    compute pool's boundary (`api/gallery_frontier._theory_item`,
    `tests/backend/api/test_offload_plain_results.py`). So what is cached
    between requests is the law's card-independent numbers, and a card's
    own numbers enter where they always did."""

    #: Per row: (flops, access, dtype index, launches, signature index, one
    #: forward's comm payload, weight), the indices into `dtypes` / `sigs`
    #: (-1: no priced collective).
    terms: Tuple[Tuple[float, float, int, float, int, float, int], ...]
    dtypes: Tuple[str, ...] = ()
    #: Per priced collective signature: the facts `collective_ladder` reads
    #: (op, world, count_per_step, base_stride, the framework's rungs), as
    #: canonical JSON.
    sigs: Tuple[str, ...] = ()

    @classmethod
    def of(cls, rows: Iterable[Mapping[str, Any]], *, book: "LawBook",
           fabric: Fabric, draft_rows: Iterable[Mapping[str, Any]] = (),
           draft_forwards: int = 0) -> "CardForm":
        """The form of the step `rows` (and `draft_forwards` of a draft's
        one-forward `draft_rows`) add up to, each row read at its own point
        (`point_law.point`). `fabric` evaluates the laws; the card symbols'
        values do not reach the form."""
        dtypes: List[str] = []
        dtype_idx: Dict[str, int] = {}
        sigs: List[str] = []
        sig_idx: Dict[str, int] = {}
        terms: List[Tuple[float, float, int, float, int, float, int]] = []
        weighted = [(row, 1) for row in rows] + [
            (row, int(draft_forwards)) for row in draft_rows]
        for row, w in weighted:
            if row["zero_cost"] or not w:
                continue
            point = row["point_law"]["point"]
            reads = []
            for floor in (0.0, 1.0):
                card = RowCard.of(row, peak_flops=1.0, peak_bw=1.0,
                                  fabric=fabric._replace(step_latency=floor),
                                  launch_s=floor)
                reads.append(book.row_at(row, point, card))
            zero, one = reads
            dtype = row.get("act_dtype") or ""
            if dtype not in dtype_idx:
                dtype_idx[dtype] = len(dtypes)
                dtypes.append(dtype)
            si = -1
            if row["is_comm"] and int(row["comm_world"]) > 1:
                sig = json.dumps({
                    "op": row["comm_op"], "world": int(row["comm_world"]),
                    "count": int(row["count_per_step"]),
                    "stride": int(row["comm_base_stride"]),
                    "rungs": [dict(r) for r in row.get("comm_rungs") or ()],
                }, sort_keys=True)
                if sig not in sig_idx:
                    sig_idx[sig] = len(sigs)
                    sigs.append(sig)
                si = sig_idx[sig]
            terms.append((float(zero.flops), float(zero.access),
                          dtype_idx[dtype], float(one.times.t_launch), si,
                          float(zero.comm_bytes), w))
        return cls(tuple(terms), tuple(dtypes), tuple(sigs))

    def bind(self, peaks: Mapping[str, Optional[float]], *,
             peak_bw: Optional[float], fabric: Fabric,
             gpu_sku: str = "") -> FloorForm:
        """The step's `FloorForm` on one card — the floors stay free:
        `peaks` the card's peak per dtype of `dtypes`, `peak_bw` its HBM
        bandwidth, `fabric` its interconnect. A row that computes in a
        dtype the card has no peak for is refused (`NoPeakForDtype`), not
        priced at zero, and a row that READS BYTES on a card with no
        bandwidth is refused the same way (`NoPeakBandwidth`): `bytes / bw`
        with no `bw` prices every byte at zero seconds, which says the card
        is infinitely fast. `w` forwards of `max(a, b + cα, nL)` is
        `max(wa, wb + wcα, wnL)`."""
        ladders = [_sig_ladder(sig, fabric) for sig in self.sigs]
        constant = 0.0
        terms: List[Tuple[float, float, float, float]] = []
        for flops, access, di, n, si, payload, w in self.terms:
            t_compute = 0.0
            if flops:
                peak = peaks.get(self.dtypes[di])
                if not peak or peak <= 0:
                    dtype = self.dtypes[di]
                    if not dtype:
                        raise ValueError(
                            f"{flops:g} FLOPs on a row with no dtype: the "
                            "peak column cannot be chosen")
                    raise NoPeakForDtype(gpu_sku or "This GPU", dtype,
                                         peak_kind(dtype))
                t_compute = flops / peak
            t_memory = 0.0
            if access:
                if (peak_bw is None or not math.isfinite(peak_bw)
                        or peak_bw <= 0):
                    raise NoPeakBandwidth(gpu_sku or "This GPU")
                t_memory = access / peak_bw
            b = c = 0.0
            if si >= 0 and payload > 0:
                seg = ladder_segment(ladders[si], payload)
                b, c = payload * seg.s_per_byte, seg.latency_s
            a, b, c, n = w * max(t_compute, t_memory), w * b, w * c, w * n
            if c == 0 and n == 0:
                constant += max(a, b)
            else:
                terms.append((a, b, c, n))
        return FloorForm.collapsed(constant, terms)

    def to_json(self) -> Dict[str, Any]:
        return {"terms": [list(t) for t in self.terms],
                "dtypes": list(self.dtypes), "sigs": list(self.sigs)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "CardForm":
        return cls(
            tuple((float(t[0]), float(t[1]), int(t[2]), float(t[3]),
                   int(t[4]), float(t[5]), int(t[6])) for t in d["terms"]),
            tuple(str(x) for x in d.get("dtypes") or ()),
            tuple(str(s) for s in d.get("sigs") or ()))


@memoize(maxsize=4096)
def _sig_ladder(sig: str, fabric: Fabric) -> Tuple[Segment, ...]:
    """A collective signature's ladder on `fabric` (`CardForm.sigs`) —
    built at a unit step latency, so a segment's `latency_s` is the hops
    coefficient one link traversal multiplies. One per (signature, card):
    every point of a walk shares its collectives' signatures."""
    d = json.loads(sig)
    return collective_ladder(
        op=d["op"], world=int(d["world"]), count_per_step=int(d["count"]),
        base_stride=int(d["stride"]),
        fabric=fabric._replace(step_latency=1.0),
        rungs=tuple(CollectiveRung.from_dict(r) for r in d.get("rungs") or ()))


@dataclass(frozen=True)
class StepSum:
    """A step's rows added up, written once: `Σ_k w_k x leaf(k)` over the
    terms (`StepTerm`), the multiplicity the walk gives — every row of the
    step, every stage's rows and its exit hop under pp, a draft forward's
    rows times the forwards. The leaf is the caller's: a number per row (the
    anchor's `t_op_s`, a re-evaluated row's), the row's law (`law`: the SOL
    total over the batch and the card) or `n_k x T_k` (`measured_law`).

    Under pp the step's time is every stage's terms and the rank's account
    (FLOPs, bytes) its own stage's: `account=True` keeps the terms of
    `rank_stage` (and the draft's)."""
    terms: Tuple[StepTerm, ...]
    rank_stage: Optional[int] = None

    @classmethod
    def of(cls, rows: Sequence[Mapping[str, Any]] = (), *,
           stages: Optional[Sequence[Sequence[Mapping[str, Any]]]] = None,
           rank_stage: Optional[int] = None,
           draft_rows: Sequence[Mapping[str, Any]] = (),
           draft_forwards: Any = 0) -> "StepSum":
        """`rows` at pp=1; `stages` (every stage's rows, its exit hop in
        them) and `rank_stage` under pp; a draft's one-forward rows and its
        forwards per step."""
        if stages is None:
            terms = [StepTerm(r) for r in rows]
        else:
            terms = [StepTerm(r, stage=s) for s, stage_rows in enumerate(stages)
                     for r in stage_rows]
        terms += [StepTerm(r, weight=draft_forwards, draft=True)
                  for r in draft_rows]
        return cls(tuple(terms), rank_stage if stages is not None else None)

    def _groups(self, account: bool) -> List[Tuple[Any, List[StepTerm]]]:
        """The terms in runs of one weight and one place, in order: a
        stage's rows, the draft's."""
        groups: List[Tuple[Any, List[StepTerm]]] = []
        place = None
        for t in self.terms:
            if account and not t.draft and t.stage != self.rank_stage:
                continue
            if (t.weight, t.stage, t.draft) != place:
                groups.append((t.weight, []))
                place = (t.weight, t.stage, t.draft)
            groups[-1][1].append(t)
        return groups

    def total(self, leaf: Callable[[StepTerm], Any], *,
              account: bool = False) -> Any:
        """`Σ_k w_k x leaf(k)`: each run of terms summed, then weighted —
        the draft's forwards multiply its one forward's sum, as the step
        runs it. Numbers in, a number out; expressions in, an expression."""
        total: Any = 0
        for weight, run in self._groups(account):
            part = sum((leaf(t) for t in run), 0)
            total = total + (part if weight == 1 else weight * part)
        return total

    def by_bound(self, leaf: Callable[[StepTerm], RowTimes]) -> Dict[str, Any]:
        """`total` of each term's time under the term that set it, the
        parts adding to the total's time (`RowRoofline.by_bound`), from a
        per-term `RowTimes`."""
        times = {id(t): leaf(t) for t in self.terms}
        return {bound: self.total(
            lambda t, bound=bound: (times[id(t)].t_op
                                    if times[id(t)].bound == bound else 0.0))
            for bound in BOUNDS}

    def term_laws(self, name: str = "t_op") -> List[sp.Expr]:
        """Each term's own share of `law`, in term order: its run's weight
        (`_groups`, the weighting `total` applies) times its row's `name`
        law — the summands `law` adds, one per term of the measured formula
        (`measured_law`), so a reader can price any of them alone."""
        out: List[sp.Expr] = []
        for weight, run in self._groups(False):
            out.extend(lifted_law(t.row, name) if weight == 1
                       else weight * lifted_law(t.row, name) for t in run)
        if len(out) != len(self.terms):
            raise ValueError("a step's runs do not hold each term once")
        return out

    def law(self, name: str = "t_op", *, account: bool = False) -> sp.Expr:
        """The SOL total of the rows' `name` law over the batch and the card
        (`lifted_law`)."""
        return self.total(lambda t: lifted_law(t.row, name), account=account)

    def bound_law(self, bound: str) -> sp.Expr:
        """The SOL time under `bound`: each row's `t_op` where that term sets
        it (compute, then memory, then comm, then launch on a tie), 0
        elsewhere."""
        def leaf(t: StepTerm) -> sp.Expr:
            c, m, k, ln, op = (lifted_law(t.row, name) for name in
                               ("t_compute", "t_memory", "t_comm", "t_launch",
                                "t_op"))
            wins = {"compute": (c >= m) & (c >= k) & (c >= ln),
                    "memory": (m > c) & (m >= k) & (m >= ln),
                    "comm": (k > c) & (k > m) & (k >= ln),
                    "launch": (ln > c) & (ln > m) & (ln > k)}[bound]
            return sp.Piecewise((op, wins), (0, True))
        return self.total(leaf)

    def laws(self) -> Dict[str, str]:
        """The phase's SOL quantities as law text over `B` and the card:
        its account (FLOPs, access, writes) and its time, per term and per
        bound."""
        account = {"flops_per_step": "flops", "total_bytes": "access",
                   "mem_write_bytes": "act_out"}
        time = {"t_compute_s": "t_compute", "t_memory_s": "t_memory",
                "t_comm_s": "t_comm", "t_sol_s": "t_op"}
        out = {wire: sp.sstr(self.law(name, account=True))
               for wire, name in account.items()}
        out.update({wire: sp.sstr(self.law(name)) for wire, name in time.items()})
        out.update({f"t_sol_{bound}_s": sp.sstr(self.bound_law(bound))
                    for bound in BOUNDS})
        return out

    def measured_law(self) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """The measured total as law text, `Σ_k w_k x n_k x T_k`: `n_k` the
        term's launches at the point (its count times its events per call,
        which the point's key states) and `T_k` the launch time measured
        under that key. A zero-cost term is 0 and names neither. With the
        symbol → term map (`StepTerm.where`, and which of the two it is)."""
        names: Dict[str, Dict[str, Any]] = {}

        def leaf(k: int, t: StepTerm) -> sp.Expr:
            if t.zero_cost():
                return sp.Integer(0)
            n, T = law_symbol(f"n_{k}"), law_symbol(f"T_{k}")
            names[n.name] = {**t.where(), "symbol": "launches"}
            names[T.name] = {**t.where(), "symbol": "launch_time"}
            return n * T

        index = {id(t): k for k, t in enumerate(self.terms)}
        expr = self.total(lambda t: leaf(index[id(t)], t))
        return sp.sstr(expr), names
