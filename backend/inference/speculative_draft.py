"""The draft half of a speculative-decode step.

Speculative decoding runs two forwards per step: a draft proposes K tokens and
the target verifies them. The decode phase prices the target's verify forward
over K + 1 positions — the last accepted token plus the K drafts
(`phases.Speculation`, `phases.phase_binding`) — and amortises it by the
acceptance rate: `(1 + α·K) × B` accepted tokens per step.

This module prices the draft forwards, when the request asks for them. The
draft graph is the model's own MTP stack (`aux_kind: mtp`), priced by the same
per-op machinery as the verify rows (`pipeline.analyze` with `timed_stacks`),
so its weight reads, KV/state reads and activations come from the same
leaves the rest of the estimate uses. Two draft styles, which differ only in
how the drafted tokens are produced:

  recursive (EAGLE; vLLM's MTP proposer): one draft forward per draft token,
      each over 1 token per sequence — K forwards.
  parallel (DFlash): one draft forward over the K positions it predicts.

The draft's time is added to the verify step's. The accepted tokens per step
do not change, so every rate the decode phase publishes is the same tokens
over a step that now includes the draft: the draft is amortised per accepted
token exactly as the verify forward is.

A draft request also keeps the draft stack's cache beside the backbone's
(`cache_scope`): the MTP layers attend over the same positions, so their cache
grows with the request's, on the pipeline stage their stack is placed on.

Not modelled: a recursive draft's cache grows by one position per forward
(each forward is priced at the verify step's cache length), and a stack of
several MTP layers is one draft forward through all of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from .phase_formulas import table_total
from .row_roofline import RowRoofline, StepSum
from .step_eval import evaluate_step
from backend.arch.eval.dim import value_of


class DraftStyle(str, Enum):
    RECURSIVE = "recursive"
    PARALLEL = "parallel"


#: The time and traffic columns a draft forward folds, as `(wire key, row key)`.
#: The wire keys are `PhaseMetrics`' own names, so the decode phase adds the
#: draft into each field it already publishes.
_FOLDED = (
    ("flops_per_step", "flops_per_rank"),
    ("total_bytes", "access_bytes_per_rank"),
    ("mem_write_bytes", "act_out_bytes_per_rank"),
    ("t_compute_s", "t_compute_s"),
    ("t_memory_s", "t_memory_s"),
    ("t_comm_s", "t_comm_s"),
    ("t_launch_s", "t_launch_s"),
    ("t_sol_s", "t_op_s"),
)


@dataclass(frozen=True)
class SpeculativeDraft:
    """Which draft a speculative-decode step runs: its style, the number of
    tokens it drafts (K), and the MTP stacks that are its graph."""

    style: DraftStyle
    draft_tokens: int
    stacks: Tuple[str, ...]

    def __post_init__(self) -> None:
        # Every constructor (`of`, `from_wire`, a direct call) passes
        # through here, so the declaration `refusal` states holds for each
        # draft that exists, not only for the ones `of` built.
        reason = self._refusal(self.stacks, self.draft_tokens)
        if reason is not None:
            raise ValueError(f"cannot count the speculative draft: {reason}")

    @staticmethod
    def stacks_of(program: Any) -> Tuple[str, ...]:
        """The model's draft graph: its `aux_kind: mtp` stacks."""
        return tuple(s.name for s in program.stacks if s.aux_kind == "mtp")

    @staticmethod
    def _refusal(stacks: Tuple[str, ...], draft_tokens: Optional[int]
                 ) -> Optional[str]:
        """Why a draft over `stacks` drafting `draft_tokens` tokens cannot
        be counted, or None when it can — the one statement of the rule."""
        if not stacks:
            return ("this model declares no MTP stack (`aux_kind: mtp`), so "
                    "there is no draft graph to price")
        if draft_tokens is not None and int(draft_tokens) < 1:
            return ("0 draft tokens is plain autoregressive decode: "
                    "nothing is drafted")
        return None

    @classmethod
    def refusal(cls, program: Any, *, draft_tokens: Optional[int] = None
                ) -> Optional[str]:
        """Why this deployment cannot count a draft, or None when it can.
        `draft_tokens` None asks about the model alone (the page asks before
        K is chosen)."""
        return cls._refusal(cls.stacks_of(program), draft_tokens)

    @classmethod
    def of(cls, program: Any, *, style: str, draft_tokens: int
           ) -> "SpeculativeDraft":
        try:
            draft_style = DraftStyle(style)
        except ValueError:
            raise ValueError(
                f"unknown draft style {style!r}; expected one of "
                f"{[s.value for s in DraftStyle]}") from None
        return cls(draft_style, int(draft_tokens), cls.stacks_of(program))

    @classmethod
    def drafted_by(cls, program: Any, *, style: str, draft_tokens: int
                   ) -> Optional["SpeculativeDraft"]:
        """The draft a step of this model drafting `draft_tokens` tokens
        runs — or None when it runs none, because this model declares no
        draft graph or because nothing is drafted (`refusal`).

        The two callers ask different questions of the same statement. A
        REQUEST that names a draft asks `of`, and a deployment that cannot
        run one is refused with the reason. A caller that only states a
        verify width — a step may verify K + 1 positions against an
        external drafter or an n-gram proposer, neither of them this
        model's graph — asks here, and gets the forwards there are to
        price: the verify forward alone when the model has no draft graph.
        Asking `of` for that was the plan's bug: it read a verify width as
        a declaration that this model drafts, and crashed on every model
        without an MTP stack."""
        if cls.refusal(program, draft_tokens=draft_tokens) is not None:
            return None
        return cls.of(program, style=style, draft_tokens=draft_tokens)

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> "SpeculativeDraft":
        return cls(DraftStyle(wire["style"]), int(wire["draft_tokens"]),
                   tuple(wire["stacks"]))

    def to_wire(self) -> Dict[str, Any]:
        """Which draft this is, as a decode phase states it (`from_wire`
        reads it back), with the forwards it runs."""
        return {"style": self.style.value, "stacks": list(self.stacks),
                "draft_tokens": self.draft_tokens, "forwards": self.forwards,
                "tokens_per_forward": self.tokens_per_forward}

    def cache_scope(self, program: Any) -> FrozenSet[str]:
        """The stacks whose cache a request running this draft keeps: the
        backbone's and the draft's own."""
        return program.included_stack_names() | frozenset(self.stacks)

    @property
    def forwards(self) -> int:
        """Draft forwards per step: one per draft token (recursive), or one
        for all of them (parallel)."""
        if self.style is DraftStyle.RECURSIVE:
            return self.draft_tokens
        return 1

    @property
    def tokens_per_forward(self) -> int:
        """Tokens per sequence each draft forward runs over."""
        return self.forward_width(self.draft_tokens)

    def forward_width(self, draft_tokens: int) -> int:
        """Tokens per sequence one forward of THIS style runs over when the
        step drafts `draft_tokens` of them — the verify width of the draft
        forward's own walk, which is not the step's.

        A reader that moves a draft forward's rows to another point of the
        step's cell reads it here: a recursive draft forward is one token
        per sequence whatever K the step verifies, so its walk is the same
        walk at every K, while a parallel draft's widens with K."""
        return 1 if self.style is DraftStyle.RECURSIVE else int(draft_tokens)

    def step(self, resolved: Any, parallel: Any, *,
             batch: int, avg_input_tokens: int, avg_output_tokens: int,
             ep_load_factor: Any = 1,
             precision: Any = None, cells: Any = None) -> Any:
        """The step one draft forward runs on: a `tokens_per_forward`-token
        decode step at the verify's cache length and batch. The verify step
        itself is K + 1 positions wide, which no draft forward is."""
        return evaluate_step(
            resolved, parallel, phase="decode", batch=max(1, int(batch or 1)),
            input_tokens=avg_input_tokens, output_tokens=avg_output_tokens,
            s_verify=self.tokens_per_forward, ep_load_factor=ep_load_factor,
            precision=precision, cells=cells)

    def forward_flops_law(self, step_eval: Any, book: Any) -> Any:
        """One draft forward's unsharded FLOPs — one data-parallel
        replica's, as the module table's column is (`Analysis.replica_cost`)
        — as a law in the workload's counts, off the walk of the step it was
        priced on, written into `book` (`row_roofline.LawBook`)."""
        by_stack = step_eval.walk.by_stack(replica=True)
        return book.write(sum((value_of(by_stack[s].flops)
                               for s in self.stacks if s in by_stack), 0))

    def price(self, rows: List[Dict[str, Any]], *,
              forward_full: Optional["ForwardFull"] = None) -> "DraftCost":
        """The draft's cost per step from ONE forward's per-op rows: their
        sum (`StepSum`), which the step runs `forwards` times;
        `forward_full` is one forward's unsharded FLOPs, its law at a
        point."""
        forward = StepSum.of(rows)
        per_forward: Dict[str, Optional[float]] = {
            wire: float(forward.total(lambda t, key=key: t.row[key]))
            for wire, key in _FOLDED}
        # The draft's measured share is the step's measured formula's
        # (`batch_sweep.apply_measured`); nothing is measured here.
        per_forward["t_measured_s"] = None
        for bound, t in RowRoofline.by_bound(rows).items():
            per_forward[f"t_sol_{bound}_s"] = t
        forward_table = table_total(rows)
        # The forward's collective bytes as its table states them: None when
        # a row does not publish its own, never a sum that skips it.
        per_forward["comm_bytes_per_rank"] = forward_table["comm_bytes_per_rank"]
        return DraftCost(self, per_forward, forward_full, tuple(rows),
                         forward_table=forward_table)


@dataclass(frozen=True)
class ForwardFull:
    """One draft forward's unsharded FLOPs: its law (`law`, in the workload's
    counts), the point it is read at and its value there."""
    law: Any
    point: Mapping[str, int]
    value: float


@dataclass(frozen=True)
class DraftCost:
    """A draft priced: one forward's folded rows, and the step's `forwards`
    of them."""

    draft: SpeculativeDraft
    per_forward: Mapping[str, Optional[float]]
    forward_full: Optional[ForwardFull] = None
    #: The one forward's per-op rows: the step's sum adds them `forwards`
    #: times (`StepSum.of(draft_rows=...)`).
    rows: Tuple[Mapping[str, Any], ...] = ()
    #: The Total row of one forward's per-op table (`phase_formulas.table_total`
    #: over the rows priced above) — what the draft's own table sums to.
    forward_table: Mapping[str, Any] = field(default_factory=dict)

    def step(self, key: str) -> Optional[float]:
        """The draft's share of the step for one folded column. Its measured
        time is the step's measured fold's to state
        (`batch_sweep.apply_measured`): None here."""
        if key == "t_measured_s":
            return None
        value = self.per_forward[key]
        # A column the forward does not publish stays unpublished per step.
        return None if value is None else self.draft.forwards * value

    def module_row(self) -> Dict[str, Any]:
        """The draft as one depth-0 row of the phase's FLOPs table, so the
        table still sums to the headline it sits under. Carries the draft's
        step timing too: without it the table's Total row would count the
        draft's FLOPs but not its time."""
        d = self.draft
        return {
            "name": f"speculative draft ({', '.join(d.stacks)}) × {d.forwards}",
            "op": "<speculative_draft>",
            "role": "speculative_draft",
            "aggregation_id": "speculative_draft",
            "template_name": "",
            "layer_count": d.forwards,
            "depth": 0,
            **({"flops_full": round(self.forward_full.value * d.forwards),
                "flops_law": {"full": self.forward_full.law,
                              "calls": d.forwards,
                              "point": dict(self.forward_full.point)}}
               if self.forward_full is not None else {"flops_full": 0}),
            "flops_per_rank": round(self.step("flops_per_step")),
            "access_bytes_per_rank": self.step("total_bytes"),
            "comm_bytes_per_rank": self.step("comm_bytes_per_rank"),
            "t_compute_s": self.step("t_compute_s"),
            "t_memory_s": self.step("t_memory_s"),
            "t_comm_s": self.step("t_comm_s"),
            "t_launch_s": self.step("t_launch_s"),
            "t_op_s": self.step("t_sol_s") or 0.0,
            "t_sol_s": self.step("t_sol_s"),
            # A whole forward's rows: no single term set its time.
            "bound": None,
            "t_measured_s": self.step("t_measured_s"),
        }

    def wire(self, *, step_t_sol: Optional[float],
             times_published: bool = True) -> Dict[str, Any]:
        """The draft as its own term of the decode phase. `step_t_sol` is the
        whole step's (verify + draft), for the draft's share of it."""
        d = self.draft
        step = {wire: self.step(wire) for wire, _ in _FOLDED}
        step["t_measured_s"] = self.step("t_measured_s")
        step["mem_read_bytes"] = step["total_bytes"] - step["mem_write_bytes"]
        per_forward = dict(self.per_forward)
        per_forward["mem_read_bytes"] = (
            per_forward["total_bytes"] - per_forward["mem_write_bytes"])
        if not times_published:
            for k in ("t_compute_s", "t_memory_s", "t_comm_s", "t_launch_s",
                      "t_sol_s", "t_measured_s"):
                step[k] = per_forward[k] = None
        share = (step["t_sol_s"] / step_t_sol
                 if step["t_sol_s"] is not None and step_t_sol else None)
        if d.style is DraftStyle.RECURSIVE:
            symbolic = ("K draft forwards × 1 token/seq, "
                        "each through the MTP stack")
        else:
            symbolic = "1 draft forward × K tokens/seq through the MTP stack"
        t_fwd = per_forward["t_sol_s"]
        substituted = (
            f"{d.forwards} × {t_fwd:.3e} s = {step['t_sol_s']:.3e} s"
            if t_fwd is not None else f"{d.forwards} forward(s); no step time")
        return {
            **d.to_wire(),
            "per_forward": per_forward,
            # The Total row of the draft's own per-op table (one forward's
            # rows, `per_op_timing_draft`).
            "op_table_total": dict(self.forward_table),
            **step,
            "t_sol_share": share,
            "formula": {"symbolic": symbolic, "substituted": substituted},
        }
