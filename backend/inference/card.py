"""One GPU, as an object, with its two views.

A card was four field bags and a handful of free functions: the sheet row
(`gpu_specs.GPUSpec`), its walk class (`framework_bindings.CardClass`), its
memory and peaks (`capacity._CardStamp`, `capacity._GpuRooflines`), its
interconnect (`comm_cost.Fabric`), and `card_class` / `_gpu_rooflines` /
`topology` / `peak_flops_for_dtype` converting between them. Callers passed
PIECES — a sku here, a `GPUSpec` there, six bandwidth fields and a topology
splat somewhere else — which is how a card came to travel separately from
the walk it was pricing, and how one card's rows came to be published as
another's (`capacity.Card.refuse_unless_one_class`).

`Card` is the one way to name a card. It owns its two views and they stay
two distinct TYPES, because the whole point of the split is that one of
them may enter a key and the other may not:

  * `Card.klass` -> `CardClass` — the STRUCTURAL facts: what a framework's
    dispatch reads of the card and which peak columns the sheet lists.
    These change a walk's rows, so they are what a walk is keyed by.
  * `Card.price(fraction)` -> `CardPrice` — the HBM a rank budgets, the
    peaks it computes and reads at, and its fabric. These change only
    numbers, so they are substituted into an already-walked row and never
    keyed.

Which sheet column belongs to which view is checked at load, column by
column (`framework_bindings._check_card_fact_classification`): a column in
neither, and in no third descriptive-only bucket, fails the load.

`CardPrice` takes the deployment's HBM fraction because "the bytes a rank
budgets" is not a fact of the card alone. Everything past that — how those
bytes divide between weights and cache — is a fact of the deployment, and
stays in `capacity`.

This module deliberately does NOT absorb the sheet's leaf accessors
(`gpu_specs.peak_hbm_bw`, `gpus_per_node`, …). They read one column and
answer one question, they have ~160 call sites across the inference engine,
Kernel Bench and the tools, and converting them is a mechanical change with
its own re-pin. `Card` delegates to them; the places that pass a card AS A
CARD ask `Card`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .comm_cost import Fabric
from .framework_bindings import CardClass, card_class
from .gpu_specs import (GPUSpec, get_gpu, gpus_per_node, hbm_bytes,
                        inter_bw_label, intra_bw_label, peak_flops_for_dtype,
                        peak_hbm_bw, peak_inter_bw_bytes_per_sec,
                        peak_intra_bw_bytes_per_sec, peak_source)


@dataclass(frozen=True)
class CardPrice:
    """What a card contributes to an estimate: its PRICE, and nothing
    structural.

    How much memory a rank has and how fast it computes, reads and talks.
    Stated once because it is asked twice: when a walk is prepared on a
    card and when a prepared walk is moved to another card of its class
    (`capacity._restamp_gpu`). The second statement of this arithmetic is
    how one card comes to be priced two ways.

    Exhaustive over the sheet's price columns, which is why there is no
    third price type: `mem_gib` is `mem_total_bytes`, `mem_bw_gbps` and the
    peak columns are `rooflines`, and `scaleup_gbps` / `scaleup_fabric` /
    `pcie_gbps` /
    `rdma_gbps` / `nvls` / `gpus_per_node` are `fabric`. `nvls` reaches
    only the fabric, which is why the memory-and-peaks half alone was never
    the whole price and every pricing site used to assemble the other half
    itself.
    """

    card: "Card"
    #: HBM capacity, bytes.
    mem_total_bytes: float
    #: The bytes a rank budgets: its HBM times the deployment's fraction.
    usable_bytes: float

    @property
    def gpu(self) -> GPUSpec:
        """The sheet row, for a caller that still reads one column."""
        return self.card.spec

    @property
    def gpu_sku(self) -> str:
        return self.card.sku

    def fabric(self, step_latency: float = 0.0) -> Fabric:
        """This card's interconnect at a link latency of `step_latency`
        seconds (`comm_cost.Fabric`) — the other half of this price."""
        return self.card.fabric(step_latency)

    def rooflines(self, model_dtype: str) -> "_Rooflines":
        """The peak rates and link labels an estimate publishes once per
        card, at the deployment's own dtype. Every per-op row still reads
        its own leaf's column (`Card.peak_for`); this is the figure a row
        with no dtype of its own falls back to, and the one the response
        prints beside `peak_source`."""
        gpu = self.card.spec
        return _Rooflines(
            peak_flops=peak_flops_for_dtype(gpu, model_dtype),
            peak_bw=peak_hbm_bw(gpu),
            peak_intra_bw=peak_intra_bw_bytes_per_sec(gpu),
            peak_inter_bw=peak_inter_bw_bytes_per_sec(gpu),
            gpus_per_node=gpus_per_node(gpu),
            intra_bw_label=intra_bw_label(gpu),
            inter_bw_label=inter_bw_label(gpu),
            peak_source=peak_source(gpu, model_dtype),
        )


@dataclass(frozen=True)
class _Rooflines:
    """A card's peak rates and link labels at one dtype
    (`CardPrice.rooflines`)."""
    peak_flops: Optional[float]
    peak_bw: Optional[float]
    peak_intra_bw: Optional[float]
    peak_inter_bw: Optional[float]
    gpus_per_node: int
    intra_bw_label: str
    inter_bw_label: str
    #: `"<sku>.<column>"` the dtype's peak was read from, `"unknown"` when
    #: the sheet has no column for it (A10: shown, not guessed; an estimate
    #: raises where it must price such a row).
    peak_source: str


@dataclass(frozen=True)
class Card:
    """One GPU of `gpu_specs.csv` (module docstring).

    Built through `of` so an unknown SKU, or one the sheet lists no HBM
    for, is refused where the card is named rather than where a number
    comes out short.
    """

    spec: GPUSpec

    @classmethod
    def of(cls, gpu_sku: str) -> "Card":
        """The card named `gpu_sku`, refused here if the sheet does not
        know it or lists no memory capacity for it — before a caller
        resolves a spec on its behalf."""
        gpu = get_gpu(gpu_sku)
        if gpu is None:
            raise ValueError(f"Unknown GPU SKU: {gpu_sku}")
        if hbm_bytes(gpu) is None:
            raise ValueError(f"GPU {gpu_sku} has no memory capacity on file")
        return cls(gpu)

    @classmethod
    def at(cls, gpu: Any) -> "Card":
        """The card a sheet row already in hand describes."""
        return cls(gpu)

    @property
    def sku(self) -> str:
        return self.spec.sku

    @property
    def klass(self) -> CardClass:
        """This card's WALK class (`framework_bindings.CardClass`): the
        facts that change a walk's rows, and the only view of a card a walk
        may be keyed by."""
        return card_class(self.spec)

    def price(self, *, gpu_memory_fraction: float) -> CardPrice:
        """This card's PRICE when a rank budgets `gpu_memory_fraction` of
        its HBM (`CardPrice`)."""
        total = float(hbm_bytes(self.spec))
        return CardPrice(self, total, total * gpu_memory_fraction)

    def peak_for(self, dtype: str) -> Optional[float]:
        """The card's peak throughput (FLOPs/s) for the MMA family `dtype`
        computes on, or None when the sheet lists no column for it — which
        is what makes a leaf of that dtype walk weight-only."""
        return peak_flops_for_dtype(self.spec, dtype)

    def fabric(self, step_latency: float = 0.0) -> Fabric:
        """This card's interconnect, as a collective is priced on it: node
        size, each fabric's per-direction bandwidth and label, one link
        traversal's latency, and what its topology admits of a framework's
        own all-reduces.

        One statement of it. Every road that prices a walk on a card needs
        it, and each used to build its own from six loose fields plus a
        `comm_cost.topology` splat — a constructor written four times, in
        which a card whose `nvls` the sheet changed, or a fifth road that
        forgot the splat, would price its all-reduce on a fabric the others
        do not have.
        """
        return Fabric.of(self.spec, step_latency)

    def walks_alike(self, other: "Card") -> bool:
        """Whether a walk made on this card may be read on `other`: when
        the two are one walk class (`klass`), and never otherwise.

        `other` is a `Card`, not a SKU: a caller that still has a name
        asks `Card.of` for the card, so an unknown SKU is refused where
        the card is named rather than read here as "a different class".
        """
        if not isinstance(other, Card):
            raise TypeError(
                f"a card's walk class is compared against another Card, not "
                f"{other!r}; `Card.of(sku)` names one (and refuses a SKU the "
                f"sheet does not know)")
        return self.klass == other.klass

    def refuse_unless_one_class(self, asked: "Card") -> None:
        """Refuse `asked` unless this card's walk may be re-priced on it.

        The one statement of that rule, because a walk reaches another card
        by more than one road: its PREPARATION moves
        (`capacity._restamp_gpu`: the card's memory and peaks), its ROWS
        move (`capacity._point_rows`: every row's law read at the other
        card's peaks and fabric), and a kept class sweep moves a step's
        form (`the serving-law pool`). Each asks here; none says it again.

        The graph does not mention the SKU — but it does mention the card's
        CLASS (`framework_bindings.CardClass`: what a framework's dispatch
        reads of the card, and the peak columns the sheet lists), because
        two cards a dispatch answers differently fuse the same graph
        differently, and the launches a step runs in a quantised format are
        chosen per card too (`serving.build_serving_layers`). So a walk
        reads on another card of ITS OWN class and no further: a
        (layout × GPU) sweep pays the walk once per (ParallelConfig, card
        class) and re-prices within each class
        (`parallel_sweep.class_cards` names the cards of one class).

        A card of another class is REFUSED, not re-priced. It is the one
        way this reuse can be wrong and the number still look reasonable:
        anchoring a whole series on one card priced Kimi-K3's Hopper and
        Blackwell decode steps off an RTX-4090 walk — 1092 rows where that
        card's own walk has 1012, because vLLM gates `fused_kda_decode` at
        compute capability 9.0 and Ada is 8.9 — and published it as those
        cards' answer. On the row road the same crossing priced
        DeepSeek-V4-Flash's B200 prefill off an H100-SXM walk: 270 rows
        where B200's own walk has 274, with the routed activation at BF16
        where that card runs it in FP8 (8192 B/row against 4224). Walk the
        other class instead.
        """
        if self.walks_alike(asked):
            return
        raise ValueError(
            f"this walk is {self.sku}'s, whose card class is "
            f"{self.klass.key()}; {asked.sku} is class {asked.klass.key()}, "
            "which walks the graph differently (its framework dispatch or "
            "its peak columns differ). A prepared walk restamps within one "
            f"class only — walk {asked.sku}'s class to price it")
