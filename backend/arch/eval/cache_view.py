"""`CacheView` — the cache layer, bound to one deployment.

`eval.cache` states each cached tensor's laws per model; a deployment asks
questions that also need the widths it runs at, the ranks it spreads
positions over, and the per-rank tensor elements its sharding produces. This
view is that binding — derived from one existing Analysis and optional
physical geometry, with the laws symbolic in the length so a capacity sweep
re-asks lengths without re-walking.

Like `placement.Placement`, it is a post-walk view: every method is a pure
derivation over the walk's cache entries. The methods that return display
dicts (`growth_formula`, `storage_breakdown`, the read
breakdown) are the presentation rows the dashboard renders
— kept beside the math they present so the two cannot drift."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import sympy as sp

from .cache import law_grows
from ..dtypes import display_name
from .dim import algebra, expr_of, substitute, value_of
from .values import CACHE_LEN
from backend.code_trace.memo import memo_property

if TYPE_CHECKING:
    from .cache import CacheGeometry


class Rate(NamedTuple):
    """How a cached tensor's stored positions split, past any window, into
    positions per token of length and positions held whatever the length.

    `ceil(s_kv / 4)` is `(1/4, 0)`; `Min(s_kv, 128)` is `(0, 128)`; a
    recurrent state `1` is `(0, 1)`; `s_kv` is `(1, 0)`. A per-token figure
    is the first times the tensor's bytes per position — a compressed pool
    adds a row every `m` tokens, a window adds nothing once full — and the
    second is the per-request constant. `storage_bytes_at` at any length
    past the windows follows this asymptotic slope; integer position
    padding is retained by `storage_bytes_at`, not flattened into the slope."""

    per_token: Any
    fixed: int

    @staticmethod
    def of(tensor: Any) -> "Rate":
        law = algebra(tensor.storage_law)
        if CACHE_LEN not in law.free_symbols:
            return Rate(sp.Integer(0), int(value_of(law)))
        if not law_grows(law):
            return Rate(sp.Integer(0), int(tensor.positions(10 ** 15)))
        slope = sp.limit(law / CACHE_LEN, CACHE_LEN, sp.oo)
        if not slope.is_number or not slope.is_finite or slope <= 0:
            raise ValueError(
                f"cached tensor {tensor.name!r}: storage law {law} grows, "
                f"but its positions per token as s_kv→∞ is {slope}")
        return Rate(slope, 0)


def _scaled(text: str, factor: Any) -> str:
    """`text` times a rational display factor: `x / m` for `1/m`.

    `factor` is exact already — `Rate.of` refuses a rate that is not a
    number, and `positions_over` is a whole count — so `fraction`, which
    sympifies what it is handed, reads its two halves straight off. Asking
    `nsimplify` first, as this did, is the opposite question: it searches for
    a closed form that might match a measurement, and handed `1/2` it runs
    mpmath's integer-relation algorithm to rediscover the 2."""
    if factor == 1:
        return text
    num, den = sp.fraction(factor)
    return f"{text} / {den}" if num == 1 else f"{text} × {factor}"


class Occupancy(NamedTuple):
    """How many layers this rank stores, versus how many it still computes.

    `scale` is `held / total`. 1 when every computed layer is stored.
    """

    held: int
    total: int
    scale: float

    @property
    def prefix(self) -> str:
        if self.held == self.total:
            return ""
        return f"{self.held} / {self.total} × "

    def fold(self, n: float) -> float:
        return n * self.scale


class CacheClass(NamedTuple):
    """One cache-declaring component, and how many copies of it a rank holds.

    `count` is the walk's: `Analysis.instances` of the component — its block's
    layers times every `repeat:` it sits under. `block_index` is the position
    in `Program.blocks` the component belongs to — None for a top-level
    entry — and is only the key the per-block breakdowns group by; it takes
    no part in the count.

    `stack` is the stack the cache belongs to (`""` for the backbone), so a
    view held for one cache scope can be narrowed to another (`within`).
    """

    cache: Any
    count: int
    block_index: Optional[int] = None
    stack: str = ""


@dataclass
class CacheView:
    """The per-query KV laws, per deployment.

    The numbers come from `eval.cache`: each cached tensor's own read law at the
    asked length (windowed stops at its window, top-K at `n_topk_keys`, a
    recurrent state at its constant), TP already in the per-rank tensor
    elements, and DCP ownership on each tensor's `positions_over`. There is no
    `kv_div`: which axes shard is the walk's answer, not a class name's.

    Its entries are one cache scope's: the stacks whose cache a request
    keeps. The default is the backbone; a speculative-draft request adds its
    draft stack (`cache_view_from_analysis`'s `cache_scope`).
    """

    #: `(Cache, layer_count, block_index)` per backbone component that declares
    #: one — see `CacheClass`.
    entries: Tuple[CacheClass, ...]
    #: Whether tensor parallelism divides ANY cached tensor in this
    #: deployment — the physical geometry's `CacheGeometry.tp_divided`, so
    #: "the MLA latent replicates under TP" is a width the projection left
    #: unchanged. None for a view with no physical geometry: a whole-model
    #: view has no deployment to answer for.
    tp_shards_cache: Optional[bool] = None
    #: The dominant growing tensor's laws, as display strings — what the
    #: single-law summary fields show. None when nothing grows.
    read_law: Optional[str] = None
    storage_law: Optional[str] = None
    #: Identity provenance for downstream projections. Excluded from value
    #: equality because the same laws derived by independent oracle walks are
    #: still numerically comparable; composition boundaries check identity.
    source_analysis: Any = field(default=None, repr=False, compare=False)
    #: How many CP ranks stripe stored layers. 1 = every rank stores
    #: every layer. >1 is layersplit occupancy: stored layers are
    #: min(L, ceil(L/group)+1). Read traffic still uses the full count.
    layer_split_over: int = 1
    #: The stacks whose cache these entries are (`cache_view_from_analysis`'s
    #: `cache_scope`) — what a pipeline stage re-counts within (`on`).
    cache_scope: FrozenSet[str] = frozenset({""})

    def on(self, stage: Any) -> "CacheView":
        """The same laws, re-counted for one pipeline stage
        (`layout.stages.StageView`): each component's copies counted for the
        layers of it the stage keeps cache for (`StageView.cache`).

        A pipeline stage holds cache for ITS layers and no others, so the
        restriction is a re-count and not a divisor: the laws, the widths and
        the per-rank element counts are the stage's already — only how many
        copies of each class it holds changes. The re-count is the one every
        stage count goes through (`Analysis.recount`), over the same
        `instances` the whole view counts, in this view's cache scope.
        """
        counts = self.source_analysis.recount(
            self.source_analysis.instances, stage.cache(self.cache_scope))
        return replace(self, entries=tuple(
            c._replace(count=counts.values[c.cache.component])
            for c in self.entries))

    def within(self, scope: FrozenSet[str]) -> "CacheView":
        """The same laws, keeping only the entries of `scope`'s stacks: what
        a forward timed over `scope` reads of a cache held for a wider
        one (a verify step reads the backbone's cache, not its draft's)."""
        return replace(self, entries=tuple(
            c for c in self.entries if c.stack in scope),
            cache_scope=self.cache_scope & frozenset(scope))

    def on_layer_split(self, group_size: int) -> "CacheView":
        """Mark stored KV as striped across `group_size` CP ranks.

        Counts stay the model's layer counts — those are how many layers
        this rank still computes. Occupancy methods fold through
        ``occupancy()``; read traffic does not. Mixed leaf-Σ classes
        (GQA+SWA) are refused here; identical leaf-Σ assignments
        (DSv3 dense+MoE MLA) are one stripe.
        """
        group = max(1, int(group_size))
        refusal = self.layer_split_refusal()
        if group > 1 and refusal is not None:
            raise ValueError(refusal)
        return replace(self, layer_split_over=group)

    def layer_split_refusal(self) -> Optional[str]:
        """Why layersplit cannot stripe this cache, or None when it can.

        One stripe per layer needs one growth law; a hybrid cache has more.
        """
        if not self.mixed_growth_laws:
            return None
        return (
            "This model mixes more than one KV cache type, so "
            "layersplit cannot say how much cache each GPU holds. "
            "Use ulysses or pcp, or a model with a single cache type."
        )

    @staticmethod
    def _tensor_sig(t: Any) -> tuple:
        """One cached tensor's storage per layer, by its declared fields:
        the law, the leaf identity, the position owners, the bytes of a
        stored position (data and dynamic scales) and the format they are
        in. Never a display label, never a figure at some length."""
        return (sp.srepr(algebra(t.storage_law)), str(expr_of(t.elements)),
                t.positions_over, value_of(t.position_bytes),
                t.resolved_format, t.recurrent)

    def _storage_sig(self, cache: Any) -> tuple:
        """A cache class's storage per layer: two classes with the same one
        store the same bytes per position, under the same laws, at every
        length (static scales are `static_scale_bytes`), whatever their
        labels — so a formula counts them as one class over their summed
        layers."""
        return tuple(sorted(((t.name, *self._tensor_sig(t))
                             for t in cache.tensors), key=repr))

    @memo_property
    def storage_identity(self) -> str:
        """What a request's cache IS, as one digest: every class's
        `_storage_sig` with the layers it has. Two views with the same
        identity store the same tensors, under the same laws, in the same
        formats, at every length — whatever their models, labels or
        checkpoints are called; one that differs anywhere differs here."""
        layers: Dict[tuple, int] = {}
        for entry, count, *_ in self.entries:
            sig = self._storage_sig(entry)
            layers[sig] = layers.get(sig, 0) + count
        held = sorted(((repr(sig), n) for sig, n in layers.items()))
        return hashlib.sha256(
            repr((held, self.occupancy())).encode()).hexdigest()[:16]

    def _growth_sig(self, cache: Any) -> Optional[tuple]:
        """`_storage_sig` of the growing tensors, names aside: the leaf Σ
        `growth_formula` factors `n_layers` over."""
        grow = sorted((self._tensor_sig(t) for t in cache.tensors
                       if t.is_growing), key=repr)
        return tuple(grow) or None

    def _growth_signatures(self) -> frozenset:
        return frozenset(
            sig for entry, _count, *_ in self.entries
            if (sig := self._growth_sig(entry)) is not None
        )

    @property
    def mixed_growth_laws(self) -> bool:
        """More than one distinct leaf-Σ among the growing caches.

        GQA + SWA (gpt-oss) has two growth laws. Dense + MoE MLA (DSv3) has
        one Σ, factored over `n_layers` in `growth_formula`. Two components
        whose laws coincide have one, whatever their family names; a
        constant-state mixer (GDN / KDA / Mamba) has none. Whether the
        backbone mixes attention types is `Program.attention_mix`.
        """
        return len(self._growth_signatures()) > 1

    def occupancy(self) -> Occupancy:
        """Stored vs computed layers, striped by global layer index.

        The extra +1 is a scratch buffer for one remote layer brought
        in for a step. Mixed leaf-Σ classes are refused in
        ``on_layer_split``.
        """
        group = self.layer_split_over
        total = sum(c.count for c in self.entries)
        if group <= 1 or total <= 0:
            held = total
        else:
            held = min(total, (total + group - 1) // group + 1)
        scale = 1.0 if total <= 0 else held / total
        return Occupancy(held, total, scale)

    def read_bytes_at(self, cache_len: float) -> Tuple[float, List[dict]]:
        """Per-query read bytes, with the per-class display breakdown when
        the backbone mixes classes (the incumbent's contract: single-class
        callers render the legacy summary line instead)."""
        total = 0.0
        breakdown: List[dict] = []
        for entry, count, *_ in self.entries:
            bytes_read = float(entry.bytes_at(cache_len, read=True)) * count
            positions = max(
                (int(t.positions(round(cache_len), read=True))
                 for t in entry.tensors), default=0)
            breakdown.append({
                "component": entry.label or entry.variant or "attention",
                "label": entry.label,
                "variant": entry.variant,
                "layer_count": count,
                "read_positions": positions,
                "per_token_per_rank": (bytes_read / positions)
                if positions else 0.0,
                "read_bytes": bytes_read,
            })
            total += bytes_read
        return total, (breakdown if len(self.entries) > 1 else [])

    def storage_bytes_at(self, cache_len: float) -> float:
        """Per-request stored bytes at `cache_len` — the other law of the
        same tensors (SWA storage caps at its window under eviction; DSA
        keeps every position it will score). Each tensor's
        `CachedTensor.bytes_at`: its data and the dynamic scales riding each
        position, so a windowed layer's scales stop at its window with its
        data. Static scales are not here: the budget subtracts them once
        (`static_scale_bytes`), before dividing."""
        return self.occupancy().fold(sum(
            float(entry.bytes_at(cache_len)) * count
            for entry, count, *_ in self.entries))

    def storage_bytes_by_block(self, cache_len: float) -> Dict[Optional[int], float]:
        """`storage_bytes_at`, split by the block invocation that holds it —
        keyed by `block_index` (the ordinal `Program.blocks` and
        `iter_blocks` share; None for a top-level cache). The values sum to
        `storage_bytes_at`: the same entries and the same fold, grouped."""
        out: Dict[Optional[int], float] = {}
        occupancy = self.occupancy()
        for entry, count, block_index, _stack in self.entries:
            out[block_index] = out.get(block_index, 0.0) + occupancy.fold(
                float(entry.bytes_at(cache_len)) * count)
        return out

    def storage_positions_at(self, cache_len: float) -> int:
        """The dominant per-request occupancy at `cache_len` — the display
        summary ("effective storage tokens"), not a budget input: budgets
        use `storage_bytes_at`, where every tensor answers its own law."""
        best = 0
        for entry, _count, *_ in self.entries:
            for tensor in entry.tensors:
                best = max(best, int(tensor.positions(round(cache_len))))
        return best

    def growth_and_floor(self) -> Tuple[float, float]:
        """`(per_token_growth, constant_floor)` — per rank, stored bytes:
        each tensor's `position_bytes`, its data and the dynamic scales
        riding each position (a quantised recurrent state's arrive once with
        the state; a window's stop at its window). Static scales are
        `static_scale_bytes`.

        The split is each tensor's `Rate`: bytes per token of length (a
        compressed pool adds a row every `m` tokens, so `1/m` of a row)
        versus the positions held whatever the length (a full window, a
        recurrent state), billed once per request. Past every window,
        `growth × L + floor` is the asymptotic storage law; integer owner
        padding remains in `storage_bytes_at(L)`. Decided per TENSOR:
        the law and the format are the tensor's, and a class label has no
        say."""
        growth = 0.0
        floor = 0.0
        for g, f in self._growth_and_floor_by_block().values():
            growth += g
            floor += f
        scale = self.occupancy().scale
        return growth * scale, floor * scale

    def growth_and_floor_by_block(
        self,
    ) -> Dict[Optional[int], Tuple[float, float]]:
        """`growth_and_floor`, split by the block invocation that holds each
        cache — keyed by `block_index` (None for a top-level cache). The
        values sum to `growth_and_floor`: the same tensors, grouped."""
        scale = self.occupancy().scale
        return {key: (g * scale, f * scale) for key, (g, f) in
                self._growth_and_floor_by_block().items()}

    def _growth_and_floor_by_block(
        self,
    ) -> Dict[Optional[int], Tuple[float, float]]:
        out: Dict[Optional[int], Tuple[float, float]] = {}
        for entry, count, block_index, _stack in self.entries:
            growth = 0.0
            floor = 0.0
            for tensor in entry.tensors:
                per_position = (float(value_of(tensor.position_bytes))
                                * count / tensor.positions_over)
                rate = Rate.of(tensor)
                growth += per_position * float(rate.per_token)
                floor += per_position * rate.fixed
            g0, f0 = out.get(block_index, (0.0, 0.0))
            out[block_index] = (g0 + growth, f0 + floor)
        return out

    def static_scale_bytes(self) -> float:
        """Per rank, the quant-scale bytes held once whatever the length —
        per-tensor and per-channel scales (`CachedTensor.static_scale_bytes`),
        a recurrent state's included. Not divided by position ownership: a
        per-tensor or per-channel scale has no position axis. The dynamic
        scales ride each position and are in `position_bytes`."""
        static = sum(count * tensor.static_scale_bytes
                     for entry, count, *_ in self.entries
                     for tensor in entry.tensors)
        return static * self.occupancy().scale

    @memo_property
    def growth_formula(self) -> dict:
        """The per-token slope formula, per rank: `growth_and_floor`'s
        growth, written out — the number the KV formula badge explains, and
        the one the budget divides by. A fact of the view, written once:
        every length an estimate or a sweep reads the view at shows it.

        The data term is built from the LEAVES: each cached tensor that adds
        positions per token contributes `expr_of(elements)` (the dim identity
        the walk kept, e.g. `d_kv_lora`, `d_head*n_kv_heads`), not the node
        name or the grounded integer, at its `Rate` — a pool compressing `m`
        tokens into a row is `d_head / m`; a window, full past its length,
        adds nothing per token and is the floor's. The law is
        `n_layers × Σ_leaf expr × bpe`. Several invocations of the same Σ
        (DSv3 dense+MoE MLA) factor `n_layers`; different Σ (GQA+SWA) stay
        a sum of that law.

        A format whose scales ride each stored position (per-token-head,
        per-group, per-tile) stores them beside the data, so they are part
        of the slope: the formula is then `data + dynamic scales`, those two
        its subterms. Without such a format it is the data law alone.
        """
        data = self._data_formula()
        scales = self._scale_formula()
        growth = self.growth_and_floor()[0]
        if scales is None:
            return {**data, "label": f"per-token KV = {data['label']}",
                    "result_bytes": growth}
        return {
            "label": "per-token KV = data + dynamic scales",
            "symbolic": f"{data['symbolic']} + {scales['symbolic']}",
            "substituted": f"{data['substituted']} + {scales['substituted']}",
            "result_bytes": growth,
            "subterms": [{**data, "label": f"data = {data['label']}"},
                         scales],
        }

    def _data_formula(self) -> dict:
        """`growth_formula`'s data term: the leaf law, per rank, with one
        subterm per class when there is more than one."""
        def _factor(t) -> Any:
            return Rate.of(t).per_token / t.positions_over

        def _leaf_sym(t) -> str:
            # The walk kept both halves of the Dim: print the identity, not
            # the grounded integer. `*` → `×` so a product of leaf dims
            # reads the same as the n_layers × bpe wrap around it.
            s = str(expr_of(t.elements)).replace("*", " × ")
            factor = _factor(t)
            return _scaled(s if factor == 1 or s.isidentifier()
                           else f"({s})", factor)

        def _grows(t) -> bool:
            return Rate.of(t).per_token > 0

        # One width for every growing tensor is the `× bpe` factor; tensors
        # stored at different formats carry their own width in the sum.
        widths = {t.element_bytes for entry, _c, *_ in self.entries
                  for t in entry.tensors if _grows(t)}
        uniform = len(widths) <= 1

        def _leaf_val(t) -> str:
            v = int(value_of(t.elements) or 0)
            text = _scaled(str(v), _factor(t))
            return text if uniform else f"{text} × {t.element_bytes:g}"

        terms: List[dict] = []
        for entry, count, *_ in self.entries:
            grow = [t for t in entry.tensors if _grows(t)]
            if not grow:
                continue
            sigma_sym = " + ".join(_leaf_sym(t) for t in grow)
            sigma_val = " + ".join(_leaf_val(t) for t in grow)
            sig = self._growth_sig(entry)
            bytes_ = sum(
                float(value_of(t.data_position_bytes)) * float(_factor(t))
                for t in grow
            ) * count
            terms.append({
                "tag": (entry.variant or entry.label or "?").lower(),
                "count": count,
                "sigma_sym": sigma_sym,
                "sigma_val": sigma_val,
                "law": f"n_layers × ({sigma_sym}) × bpe",
                "sig": sig,
                "bytes": bytes_,
            })
        if not terms:
            return {"label": "0", "symbolic": "0",
                    "substituted": "0", "result_bytes": 0.0}
        bpe_tail = f" × {next(iter(widths)):g}" if uniform else ""
        occ = self.occupancy()
        scale = occ.scale
        prefix = occ.prefix
        total = sum(t["bytes"] for t in terms) * scale

        def _sub(text: str) -> str:
            return f"{prefix}({text})" if prefix else text

        subterms = [
            {
                "label": f"{t['tag']} × {t['count']}",
                "symbolic": t["law"],
                "substituted": _sub(
                    f"{t['count']} × ({t['sigma_val']}){bpe_tail}"),
                "result_bytes": t["bytes"] * scale,
            }
            for t in terms
        ]
        # Factor n_layers inside each identical leaf-Σ; then sum the
        # distinct Σs. No model name in the grouping — the signature is
        # the leaf exprs + the position law.
        groups: Dict[tuple, List[dict]] = {}
        order: List[tuple] = []
        for t in terms:
            if t["sig"] not in groups:
                groups[t["sig"]] = []
                order.append(t["sig"])
            groups[t["sig"]].append(t)
        parts_sym: List[str] = []
        parts_val: List[str] = []
        for sig in order:
            ts = groups[sig]
            n_parts = " + ".join(str(x["count"]) for x in ts)
            n_expr = n_parts if len(ts) == 1 else f"({n_parts})"
            parts_sym.append(f"n_layers × ({ts[0]['sigma_sym']}) × bpe")
            parts_val.append(f"{n_expr} × ({ts[0]['sigma_val']}){bpe_tail}")
        inner_val = " + ".join(parts_val)
        substituted = _sub(inner_val) if prefix else inner_val
        return {
            "label": "n_layers × (Σ_leaf expr) × bpe",
            "symbolic": " + ".join(parts_sym),
            "substituted": substituted,
            "result_bytes": total,
            "subterms": subterms if len(terms) > 1 else [],
        }

    def _scale_formula(self) -> Optional[dict]:
        """`growth_formula`'s dynamic-scale term, written like its data
        term: `n_layers × scales × scale_bytes` at each growing tensor's
        rate (`/ m` for a pool of `m` tokens a row), the layers of tensors
        that store the same scales at the same rate counted together —
        `(3 + 58) × 4 × 4`. Values are arithmetic, names stay out of them.
        None when no growing tensor's format writes scales beside each
        position."""
        groups: Dict[tuple, List[int]] = {}
        total = 0.0
        for entry, count, *_ in self.entries:
            for tensor in entry.tensors:
                rate = Rate.of(tensor).per_token
                width = tensor.scale_position_bytes
                if not rate or not width:
                    continue
                factor = rate / tensor.positions_over
                total += count * width * float(factor)
                dynamic, _static = tensor.scale_overhead()
                key = (dynamic, tensor.resolved_format.scale_bytes, factor)
                groups.setdefault(key, []).append(count)
        if not groups:
            return None
        sym_parts: List[str] = []
        val_parts: List[str] = []
        for (dynamic, scale_bytes, factor), counts in groups.items():
            n = (str(counts[0]) if len(counts) == 1
                 else f"({' + '.join(map(str, counts))})")
            sym_parts.append(_scaled("n_layers × scales × scale_bytes", factor))
            val_parts.append(_scaled(f"{n} × {dynamic:g} × {scale_bytes:g}",
                                     factor))
        inner = " + ".join(val_parts)
        occ = self.occupancy()
        if occ.prefix:
            inner = f"{occ.prefix}({inner})"
        return {
            "label": "dynamic scales",
            "symbolic": " + ".join(sym_parts),
            "substituted": inner,
            "result_bytes": total * occ.scale,
        }

    @property
    def stores_every_token(self) -> bool:
        """Every cached tensor keeps one position per token of length (its
        `Rate` is `(1, 0)`): no window, no compression, no fixed state. Then,
        and only then, the per-token growth times the length IS the stored
        bytes of a request at any length."""
        return all(Rate.of(t) == (1, 0)
                   for entry, *_ in self.entries for t in entry.tensors)

    def storage_breakdown(self, cache_len: float) -> List[dict]:
        """Per-class storage rows for the popover, when the backbone mixes
        classes — node groups merged where the law and the width agree
        (`Cache.groups`), the incumbent's own contract."""
        if len(self.entries) <= 1:
            return []
        return self.storage_rows(cache_len)

    def storage_rows(self, cache_len: float) -> List[dict]:
        """`storage_breakdown`'s rows for every class, one class included:
        each cached tensor's stored bytes per position (`position_bytes`,
        data and dynamic scales) and its positions at `cache_len`, under its
        own law. Folded by the same occupancy as `storage_bytes_at`, so the
        rows' `request_bytes` sum to it."""
        out: List[dict] = []
        occupancy = self.occupancy()
        for entry, count, *_ in self.entries:
            rows = []
            positions_summary = 0
            for tensor in entry.tensors:
                positions = int(tensor.positions(round(cache_len)))
                per_token = occupancy.fold(
                    float(value_of(tensor.position_bytes))
                    * count / tensor.positions_over)
                rows.append({
                    "node": tensor.name,
                    "expr": str(tensor.storage_law),
                    "positions": positions,
                    "per_token_per_rank": per_token,
                    "bytes": per_token * positions,
                })
                positions_summary = max(positions_summary, positions)
            request_bytes = occupancy.fold(
                float(entry.bytes_at(cache_len)) * count)
            out.append({
                "component": entry.label or entry.variant or "attention",
                "label": entry.label,
                "variant": entry.variant,
                "layer_count": count,
                "storage_positions": positions_summary,
                "per_token_per_rank": (request_bytes / positions_summary)
                if positions_summary else 0.0,
                "request_bytes": request_bytes,
                "nodes": rows,
            })
        return out

    def storage_dtypes(self, cache_len: float) -> List[str]:
        """The element dtypes a request's cache is stored in at
        `cache_len`, most bytes first — each tensor's format's own dtype
        (`FP8 (E4M3)`, not its scales or its unquantised tail; those are
        `CacheFormat.label`'s). The cache's precision in a word or three."""
        by_dtype: Dict[str, float] = {}
        for entry, count, *_ in self.entries:
            for tensor in entry.tensors:
                name = display_name(tensor.resolved_format.dtype)
                by_dtype[name] = by_dtype.get(name, 0.0) + count * float(
                    value_of(tensor.bytes_at(round(cache_len))))
        return sorted(by_dtype, key=lambda name: -by_dtype[name])

    def _stored_term(self, tensors: List[Any], count: int,
                     length: int) -> dict:
        """`count` layers of `tensors` — one law, one set of position
        owners — at `length`, written from the leaves: `n_layers × (Σ leaf)
        × bpe × law`, with `+ scales` when the format stores dynamic scales
        beside each position. Values are the walk's: leaf sizes, the
        format's widths, the law's positions."""
        law = tensors[0].storage_law
        owners = tensors[0].positions_over
        positions = int(value_of(substitute(law, CACHE_LEN, length)))
        widths = {t.element_bytes for t in tensors}
        leaves_sym = " + ".join(
            str(expr_of(t.elements)).replace("*", " × ") for t in tensors)
        leaves_val = [str(int(value_of(t.elements))) for t in tensors]
        many = len(tensors) > 1
        data_sym = f"({leaves_sym}) × bpe" if many else f"{leaves_sym} × bpe"
        if len(widths) == 1:
            width = f"{next(iter(widths)):g}"
            data_val = (f"({' + '.join(leaves_val)}) × {width}" if many
                        else f"{leaves_val[0]} × {width}")
        else:
            data_val = " + ".join(
                f"{v} × {tensor.element_bytes:g}"
                for v, tensor in zip(leaves_val, tensors, strict=True))
        scaled = [t for t in tensors if t.scale_position_bytes]
        if scaled:
            scale_val = " + ".join(
                f"{t.scale_overhead()[0]:g} × {t.resolved_format.scale_bytes:g}"
                for t in scaled)
            inner_sym = f"({data_sym} + scales)"
            inner_val = f"({data_val} + {scale_val})"
        elif many or len(widths) != 1:
            inner_sym, inner_val = data_sym, (
                data_val if len(widths) == 1 else f"({data_val})")
        else:
            inner_sym, inner_val = data_sym, data_val
        symbolic = f"n_layers × {inner_sym} × {law}"
        substituted = f"{count} × {inner_val} × {positions}"
        if owners != 1:
            symbolic += " / position_owners"
            substituted += f" / {owners}"
        occupancy = self.occupancy()
        per_layer = sum(float(value_of(t.position_bytes)) for t in tensors)
        return {
            "symbolic": symbolic,
            "substituted": occupancy.prefix + substituted,
            "result_bytes": occupancy.fold(
                per_layer * count * positions / owners),
        }

    def request_formula(self, cache_len: float) -> dict:
        """`storage_bytes_at(cache_len)`, written out: per-request terms that
        add up to it, never a per-token figure hung beneath it. The formula
        every page that shows one request's cache explains it with.

        One term per class, classes with the same storage per layer
        (`_storage_sig` — DSv3's dense and MoE MLA, DSv4's CSA at every
        invocation) counted once over their summed layers. A term is written
        from the leaves, as `growth_formula` is: `n_layers × (Σ leaf) × bpe
        × law`, the dynamic scales a format stores beside each position
        added to the per-position bytes — the tensors that share a law one
        Σ, a class holding several laws (a window, a compressed pool) one
        line per law beneath it. When every tensor keeps
        one position per token the node's own line is the per-token law ×
        the length (`growth_formula`); otherwise it is the sum of the
        class terms."""
        total = self.storage_bytes_at(cache_len)
        occupancy = self.occupancy()
        length = round(cache_len)
        merged: Dict[tuple, List[Any]] = {}
        for entry, count, *_ in self.entries:
            slot = merged.setdefault(self._storage_sig(entry), [entry, 0, []])
            slot[1] += count
            tag = entry.variant or entry.label or "attention"
            if tag not in slot[2]:
                slot[2].append(tag)
        class_terms: List[dict] = []
        for entry, count, tags in merged.values():
            tag = " / ".join(tags)
            key = "_".join(tags)
            # The tensors that share a law (and position owners) are one
            # Σ over their leaves — the K and V of one attention, DSA's
            # latent, rope and index key — as `growth_formula` writes them.
            by_law: Dict[tuple, List[Any]] = {}
            for t in entry.tensors:
                by_law.setdefault(
                    (sp.srepr(algebra(t.storage_law)), t.positions_over), []).append(t)
            groups = [{
                "symbol": f"kv_{key}_{'_'.join(t.name for t in ts)}",
                "label": f"{tag}.{', '.join(t.name for t in ts)}",
                **self._stored_term(ts, count, length),
            } for ts in by_law.values()]
            term = {"symbol": f"kv_{key}_request",
                    "label": f"{tag} × {count} layers",
                    "note": "; ".join(sorted({
                        t.resolved_format.label() for t in entry.tensors})),
                    "result_bytes": occupancy.fold(
                        float(entry.bytes_at(cache_len)) * count)}
            if len(groups) == 1:
                names = ", ".join(t.name for t in entry.tensors)
                class_terms.append({**groups[0], **term,
                                    "label": f"{tag} ({names}) × {count} layers"})
            else:
                class_terms.append({
                    **term,
                    "symbolic": " + ".join(g["symbolic"] for g in groups),
                    "substituted": " + ".join(g["substituted"] for g in groups),
                    "subterms": groups,
                })
        growth = self.growth_formula
        growth_sym = growth.get("symbolic") or ""
        if growth_sym and growth_sym != "0" and self.stores_every_token:
            return {
                "label": "Per-request KV",
                "symbolic": f"({growth_sym}) × {self.storage_law}",
                "substituted": (f"({growth.get('substituted')}) × "
                                f"{self.storage_positions_at(cache_len)}"),
                "result_bytes": total,
                "subterms": class_terms,
            }
        return {
            "label": ("Per-request KV (hybrid)" if self.mixed_growth_laws
                      else "Per-request KV"),
            "symbolic": "Σ_class request_KV_class",
            "substituted": " + ".join(
                f"{t['result_bytes']:.0f}" for t in class_terms) or "0",
            "result_bytes": total,
            "subterms": class_terms,
        }

def cache_view_from_analysis(
    analysis: Any,
    *,
    geometry: Optional["CacheGeometry"] = None,
    cache_scope: Optional[FrozenSet[str]] = None,
) -> CacheView:
    """Bind cache laws to an existing Analysis and explicit geometry.

    ``geometry`` is a typed ``cache.CacheGeometry`` when the physical tensor
    geometry differs from ``analysis.shapes``. Laws, component ownership and
    stack attribution still come from that same Analysis; only geometry is
    projected.  This is the no-replay entry point used by layout. Each
    component's copies are `analysis.instances` — the walk's count, every
    enclosing `repeat:` included.

    ``cache_scope`` names the stacks whose cache is kept; None is the
    backbone (`Program.included_stack_names`).
    """
    if cache_scope is None:
        cache_scope = analysis.program.included_stack_names()
    from . import cache as cache_mod

    program = analysis.program
    layer = cache_mod.derive(analysis, geometry=geometry)
    entries: List[CacheClass] = []
    dominant: Tuple[float, Optional[str], Optional[str]] = (0.0, None, None)
    for uid, entry in layer.values.items():
        if not entry.persists:
            continue
        node = program.by_uid(uid)
        stack = program.stack_of(node)
        if stack not in cache_scope:
            continue
        copies = analysis.instances.values.get(uid)
        if copies is None:
            raise ValueError(
                f"{node.display_path}: holds a cache and the walk counted "
                f"no copies of it")
        count = int(value_of(copies))
        entries.append(CacheClass(entry, count, program.block_of(uid), stack))
        for tensor in entry.tensors:
            if cache_mod.CACHE_LEN in algebra(tensor.storage_law).free_symbols:
                weight = float(tensor.elements) * count
                if weight > dominant[0]:
                    dominant = (weight, str(tensor.read), str(tensor.storage_law))
    return CacheView(
        entries=tuple(entries),
        tp_shards_cache=(
            None if geometry is None else bool(geometry.tp_divided)),
        read_law=dominant[1],
        storage_law=dominant[2],
        source_analysis=analysis,
        cache_scope=frozenset(cache_scope),
    )

