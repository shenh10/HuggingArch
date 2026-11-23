"""Layer-layout pass for the architecture-view's `assignments:` cards.

Folds a heterogeneous `assignments:` stack into UI cards that reflect the
actual layer-index order: contiguous runs of one block kind collapse into
single cards, periodic interleavings (hybrid attention) collapse into a
single "cycle container" with per-position inner cards.

A card is one block KIND (`BlockMeta.family_name`, `CSA + MoE FFN`), not
one template: the templates of a kind that differ only in a component they
bind (DeepSeek-V4.1's KV-owning and KV-borrowing CSA layers) are its
`variants`, each with its own layers and its own expandable block. Cutting
a card at every template change would split the stack wherever a layer
owns some state the next ones reuse.

Pure layout — the per-index map is the stack's block invocations
(`BlockMeta.layer_indices`), plus the `block_label_fn` callback the
architecture view supplies. No spec / engine modification, no rendering /
styling.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from backend.arch.layer_span import layer_span

from ._periodic import find_periodic_span


def _layers_label(indices: Sequence[int]) -> str:
    return f"{'layer' if len(indices) == 1 else 'layers'} {layer_span(indices)}"


def layout_assignments_in_order(
    metas,
    block_label_fn: Callable[[str, int, int], tuple],
) -> List[dict]:
    """Render an `assignments:` stack as cards in actual layer-index
    order, collapsing periodic interleavings into a single cycle card.

    `metas` are the stack's block invocations (`BlockMeta`s whose
    `layer_path` is the stack's).

    Pipeline:
    1. Build a per-layer-index map of each layer's block kind
       (`family_name`) and template from those invocations.
    2. Search the kind sequence for the longest contiguous range with a
       periodic pattern (period ≥ 2) — covers e.g. miaomiao's
       `(deltanet×3 + mla_moe)` cycle, Qwen3-Next's `(linear×3 + full)`
       cycle, GPT-OSS's `(SWA + GQA)` 2-period alternation. A kind
       carries attention and FFN together, so a dense prefix before an
       MoE steady state (Step-3.7-Flash) stays a prefix run.
    3. Emit cards: edge runs (single-kind stretches outside the cycle),
       one collapsed cycle card, edge runs after.
    4. Fallback: when no periodic span exists, emit one card per
       contiguous single-kind run.

    Returns a list of card dicts ready to extend `layers`.
    """
    cards: List[dict] = []

    occupied = [index for meta in metas for index in meta.layer_indices or ()]
    if not occupied:
        return cards
    L = max(occupied) + 1

    # Enumeration already applied declaration-order precedence and filled
    # the remainder from `rest:`, so the invocations partition [0, L). The
    # cards follow the iterations in the order they run; a looped stack's
    # are its executions, and a card names them by stored layer and pass
    # (`stored`).
    idx_to_tmpl: List[Optional[str]] = [None] * L
    idx_to_kind: List[Optional[str]] = [None] * L
    stored: Dict[int, str] = {}
    meta_of: Dict[str, Any] = {}
    for meta in metas:
        meta_of.setdefault(meta.template_name, meta)
        numbers = (dict(zip(meta.layer_indices, meta.layer_numbers,
                            strict=True))
                   if meta.layer_indices is not None else {})
        for idx in meta.layer_indices or ():
            if idx >= 0:
                idx_to_tmpl[idx] = meta.template_name
                idx_to_kind[idx] = meta.family_name or meta.template_name
                if meta.stored_indices is not None:
                    stored[idx] = (f"loop gate {numbers[idx] + 1}"
                                   if meta.loop_gate
                                   else f"{numbers[idx]}"
                                   + (f" · pass {meta.pass_}"
                                      if meta.pass_ > 1 else ""))

    def _named(indices: Sequence[int]) -> str:
        """`indices` in words: as layer numbers outside a loop, else each
        by its stored layer and pass."""
        if not any(i in stored for i in indices):
            return _layers_label(indices)
        shown = [stored.get(i, str(i)) for i in indices]
        return (shown[0] if len(shown) == 1
                else f"{shown[0]} … {shown[-1]}")

    def _card(indices: List[int], where: str, count: int) -> dict:
        """One card for the layers `indices`, all of one kind, standing
        for `count` layers: the block itself when one template runs them,
        else the kind with each template as a variant."""
        by_tmpl: Dict[str, List[int]] = {}
        for idx in indices:
            by_tmpl.setdefault(idx_to_tmpl[idx], []).append(idx)
        first = idx_to_tmpl[indices[0]]
        label, is_moe, detail = block_label_fn(first, count, 1)
        card = {
            "type": "moe_block" if is_moe else "dense_block",
            "detail": f"{where} · {detail}",
            "expandable": True,
            "layers": list(indices),
        }
        if len(by_tmpl) == 1:
            return {"name": label, **card, "template": first}
        # The slots bound differently among THIS card's variants.
        bound = [{slot: name for slot, name, _text in meta_of[t].distinct_slots}
                 for t in by_tmpl]
        differ = {slot for found in bound for slot in found
                  if len({b.get(slot) for b in bound}) > 1}
        variants = []
        for tmpl, at in by_tmpl.items():
            v_label, v_moe, _detail = block_label_fn(tmpl, len(at), 1)
            variants.append({
                "name": v_label,
                "type": "moe_block" if v_moe else "dense_block",
                "detail": f"{_named(at)} · ×{len(at)}",
                "expandable": True,
                "template": tmpl,
                "layers": at,
                # What tells it from the card's other variants, each slot
                # with the component bound there and its description.
                "slots": [{"slot": slot, "component": name,
                           "description": text}
                          for slot, name, text in meta_of[tmpl].distinct_slots
                          if slot in differ],
            })
        return {"name": meta_of[first].family_name or label, **card,
                "variants": variants}

    def _emit_runs_in_range(lo: int, hi: int) -> None:
        """Emit per-kind-run cards covering layers [lo, hi)."""
        i = lo
        while i < hi:
            kind = idx_to_kind[i]
            if kind is None:
                i += 1
                continue
            j = i
            while j + 1 < hi and idx_to_kind[j + 1] == kind:
                j += 1
            where = _named(list(range(i, j + 1)))
            cards.append(_card(list(range(i, j + 1)), where, j - i + 1))
            i = j + 1

    def _emit_cycle(start: int, end: int, period: int) -> None:
        """Emit a cycle CONTAINER card for layers [start, end].

        The container holds `period` inner cards — one per cycle
        position, each a kind card over that position's layers in every
        cycle — so the frontend can render the actual interleaved
        structure inside a wrapper that says "×N cycles, layers X–Y".

        Inner cards carry the absolute layer index of the FIRST cycle
        instance (start, start+1, …) plus a `cycle_offset` so the
        frontend can label them as "+0", "+1", … relative to the
        cycle's start.
        """
        cycles = (end - start + 1) // period
        is_moe = False
        inner_cards: List[dict] = []
        for offset in range(period):
            abs_idx = start + offset
            # One position of one cycle is one layer.
            inner = _card([abs_idx + k * period for k in range(cycles)],
                          f"+{offset} (e.g. {_named([abs_idx])})", 1)
            is_moe = is_moe or inner["type"] == "moe_block"
            inner_cards.append({**inner, "cycle_offset": offset})

        cards.append({
            "name": f"× {cycles} cycles",
            "type": "cycle_container",
            "detail": f"{_named(list(range(start, end + 1)))} · "
                      f"period {period}",
            "expandable": False,
            "cycle_count": cycles,
            "cycle_period": period,
            "cycle_layer_range": [start, end],
            "layers": list(range(start, end + 1)),
            "cycle_blocks": inner_cards,
            "is_moe": is_moe,
        })

    span = find_periodic_span(idx_to_kind)
    if span is not None:
        start, end, period = span
        # The detected span may NOT be an exact multiple of `period` —
        # `find_periodic_span` extends `end` as long as `seq[i+P] ==
        # seq[i]`, so a periodic run of 73 indices with period 8 still
        # qualifies (9 full cycles + 1 leftover that happens to match
        # the start of a 10th cycle but doesn't form one). The cycle
        # card only renders complete cycles; trim end to the last full
        # cycle and emit the remainder as edge runs so no layer is
        # silently dropped (Ling-2.6-1T's idx-79 full_moe was lost
        # before this trim — span [7,79] period 8 → 9 cycles cover
        # [7,78], idx 79 belongs in the trailing edge run).
        cycles = (end - start + 1) // period
        cycle_end = start + cycles * period - 1
        _emit_runs_in_range(0, start)
        _emit_cycle(start, cycle_end, period)
        _emit_runs_in_range(cycle_end + 1, L)
    else:
        _emit_runs_in_range(0, L)
    return cards
