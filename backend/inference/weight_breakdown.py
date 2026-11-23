"""Per-node weight breakdown — hierarchical tree sourced from the
resolved spec BlockGraph (same truth source as Inspector).

Returns a tree matching the BlockGraph structure:

  groups: [                           ← top-level: one per stack/block
    { name: 'dense_block × 3',
      nodes: [                        ← block's direct children
        { name: 'self_attn', class: 'attention', children: [...] },
        { name: 'mlp', class: 'dense_ffn', ... },
        ...
      ]
    },
    ...
  ]
  model_level: [                      ← embedding / lm_head / other non-block nodes
    { name: 'embed_tokens', class: 'embedding', ... },
    { name: 'lm_head', class: 'lm_head', ... },
  ]

Each node reports bytes_full (full-cluster) and bytes_per_rank, and the two
run in that order: `bytes_per_rank` comes from the layout local SHAPE,
and the displayed `divisor` is read
back OFF the pair as their observed ratio. Not the reverse — a class cannot
produce a ratio (a fused QKV's axes enter additively, an extent below the
degree does not split at all), so a node whose ratio is not a whole number
reports `divisor = 0` and shows its shape instead of a fraction nothing
would justify.

`class` says which part of the model a weight belongs to, and nothing more;
it is deployment-independent by contract (see `sharding.leaf_class`).

MTP / other aux group stacks are walked normally — their internals get their
own per-rank shapes, rather than lumped as a replicated aux blob.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import FrozenSet, Any, Dict, List, Optional


from backend.arch import dims
from backend.arch.dtypes import display_label, display_name
from backend.arch.eval.cache import derive as derive_cache
from backend.arch.eval.access import params_bytes_formula
from backend.arch.eval.refusal import node_context
from backend.arch.layout.stages import StagePlacement
from backend.arch.ops import OpSpec
from backend.arch.spec_tree import (
    LeafTensor,
    iter_blocks,
    iter_leaves,
)

from .parallel_config import ParallelConfig
from .sharding import (
    dim_to_tp_field,
    is_attention_role,
    leaf_class,
    moe_child_class,
    step_class,
    vocab_kind_of_leaf,
)
from backend.arch.sharding import PREFILL


def _knob_label(field: str) -> str:
    """The `ParallelConfig` knob's label in the storage tooltip.

    A knob names itself, with one exception the declaration states rather
    than this label: a world on the `ep` mesh divides a single axis — the
    expert bank's leading copy axis — and the knob's own name does not say
    which, so it is spelled with it. Which axis that is (and that there is
    exactly one) is `dims.yaml`'s to say.
    """
    axes = sorted(sym for sym, knob in dims.degree_fields("ep").items()
                  if knob == field)
    if len(axes) != 1:
        return field
    return f"{axes[0]}→{axes[0]}/{dims.world_of(axes[0]).mesh}"


def _knob_order(field: str) -> int:
    """Position in the tooltip's `a × b × c`: the order the worlds that
    divide a weight's own axes are declared in (`dims.yaml`, through the
    same `dim_to_tp_field` the caller reads). A stable reading order beats
    whichever symbol a leaf happened to carry first, and reading it off the
    declaration keeps it one order rather than a second list of the knobs."""
    return list(dict.fromkeys(dim_to_tp_field().values())).index(field)


def _reject_ambiguous_template_names(block_metas) -> None:
    """Refuse a stack where one template NAME has two invocations with layers.

    A leaf records the template it came from (`BlockMeta.label`: the template
    name, which a derived group of a `read: all` stack extends with what its
    layers see) and not WHICH invocation of it, so an index keyed by that
    name can only answer for a name invoked once.
    The weight tree depends on that everywhere: the block-group loop charges
    every meta the whole name's leaves. Two live invocations would report the
    template twice at full size and price its layers double.

    No model in the corpus does. GLM-5.1 and GLM-5.2 each declare one MoE
    template twice — which is exactly why the CACHE side is keyed by block
    ordinal rather than by name — but in both, one of the two invocations has
    zero layers, so the name still identifies the single live one here. Saying
    so out loud is the point: the cache side already paid for this ambiguity
    with a 1.96x over-report, and the next spec to reach it on the weight side
    should get a sentence instead of a silent doubling.
    """
    live: Dict[str, list] = {}
    for meta in block_metas:
        # A call tied to another's weights holds none of its own.
        if int(meta.count) > 0 and meta.tied_to is None:
            live.setdefault(meta.label, []).append(meta)
    ambiguous = sorted((n, ms) for n, ms in live.items() if len(ms) > 1)
    if ambiguous:
        name, metas = ambiguous[0]
        raise ValueError(
            f"block template {name!r} is invoked {len(metas)} times with "
            f"layers ({', '.join(str(int(m.count)) for m in metas)}), and a "
            f"leaf records only the template it came from. The weight tree "
            f"would charge every invocation the whole template's tensors. "
            f"Give the invocations distinct template names, or carry the "
            f"invocation ordinal on the leaf as `CacheClass.block_index` does."
        )


def _copy_factors(lt: LeafTensor, factors: Any) -> list:
    """The leaf's named counts for the tooltip, from the walk's own factor
    table (`Analysis.copy_factors`).

    Read, not derived. Each factor arrives as `(symbol, count, kind, unit)`
    outermost-first, with the block template's layer count among them, and
    their product IS the leaf's multiplicity — asserted here rather than
    reconstructed, because a consumer that factors a multiplicity itself has
    to decide which factor is which, and the only thing it has to decide
    with is the symbol's spelling. (It used to:
    `"experts" if "expert" in raw else "copies"` labelled every layer loop
    in six vision towers `copies`.)

    The display noun is the symbol's declaration (`dims.yaml`'s `unit`). A
    symbol that declares none shows its own NAME — true and incomplete —
    rather than a default noun that would be false for most of them.
    """
    if factors is None:
        raise ValueError(
            f"weight leaf '{lt.display_path or '?'}' is repeated "
            f"{int(lt.multiplicity)} times and the walk published no factor "
            f"table for it; the named counts are the walk's to state "
            f"(`Analysis.copy_factors`), and a consumer that factors the "
            f"multiplicity itself can only label the factors by their "
            f"spelling")
    product = 1
    for factor in factors:
        product *= int(factor.count)
    total = int(lt.multiplicity) or 1
    if product != total:
        raise ValueError(
            f"weight leaf '{lt.display_path or '?'}' is repeated {total} "
            f"times but its published factors multiply to {product} "
            f"({[(f.symbol, int(f.count)) for f in factors]})")
    shown = [f for f in factors if int(f.count) > 1]
    unnamed = [f for f in shown if not (f.unit or f.symbol)]
    if unnamed:
        raise ValueError(
            f"weight leaf '{lt.display_path or '?'}' is repeated by "
            f"{[int(f.count) for f in unnamed]} with neither a declared unit "
            f"nor a symbol to name it; the noun is the declaration's to state "
            f"(`dims.yaml` `unit`)")
    return [{"n": int(f.count), "unit": f.unit or f.symbol} for f in shown]


def _copies_dropped(pp_copies_div: float = 1.0) -> list:
    """The knobs that drop whole COPIES of a leaf on a rank, as labels.

    A count, never a shape. A rank ends up with less of a tensor either
    because each copy is thinner — a SHAPE division, the forward transform's,
    the expert axis of a bank among them — or because it holds fewer copies.
    Only pipeline parallelism does the second: it partitions the layer index
    space, so it arrives as `pp_copies_div`, the described stage's share of
    its layers (`StagePlacement.block_counts`), already computed.
    """
    return ["pp"] if pp_copies_div > 1 else []


# Deployment-dtype design:
# --------------------------------------------------------------------------
# Per-leaf storage bytes are sourced from materialize_for_resolved_spec
# (the single quant fusion entry, per CLAUDE.md §2.3). `weight_dtype_bytes`
# survives as a UI hint for the dashboard label + roofline; mixed-precision
# models (FP8 attention + BF16 norms + INT4 experts) flow through the
# per-leaf MaterializedTensor.storage_bytes naturally — no uniform-dtype
# proxy on the numeric path.




def _attention_kind_from_cache(caches, leaf) -> Optional[str]:
    """Breakdown-header family from `eval.cache.Cache.label` on the
    enclosing attention call."""
    node = leaf.ir_node
    while node is not None:
        entry = caches.get(node)
        if entry is not None and entry.label:
            return entry.label
        node = node.parent
    return None


# Per-leaf class logic lives in `sharding.py`. Use the imported helpers
# `step_class` / `leaf_class` directly — no in-file duplicates.


@dataclass
class TreeNode:
    name: str
    op: str
    class_: Optional[str]
    params: int
    bytes_full: float
    divisor: int
    bytes_per_rank: float
    divisor_source_label: str = ""       # e.g. "tp_attention" — which knobs shard this leaf
    # The same derivation as two renderable halves. Authored here because the
    # derivation is here: a consumer that composes `bytes_full / divisor`
    # itself is writing a formula, and it disagrees with `bytes_per_rank`
    # wherever the ratio is not an integer (any fused projection under TP).
    shard_symbolic: str = ""     # "bytes_full × (per-rank shape / full shape)"
    shard_substituted: str = ""  # "18874368 × (768×4096 / 4608×4096)"
    # Leaf identity for the Full / Per-rank hover: params * dt_w, products
    # kept. Empty on containers (those stay a Σ of children).
    leaf_symbolic: str = ""
    leaf_substituted: str = ""
    leaf_cnt: int = 1
    leaf_note: str = ""
    # Named copy counts whose product is `leaf_cnt` (`layers`, `experts`).
    leaf_copy_factors: list = field(default_factory=list)
    # The analysis node this leaf is. Layout, checkpoint rows and cost
    # all key on uid; a display name is not unique across stacks.
    uid: int = 0
    weight_shape: str = ""       # readable weight shape, e.g. "[1536, 7168]"
    weight_dtype: str = ""       # display name (e.g. "FP8 E4M3", "BF16")
    weight_dtype_bytes: float = 0.0  # effective bytes/elem (data + scale overhead)
    # Split of bytes_full into pure data vs quant-scale metadata. Matches:
    #   bytes_full = data_bytes_full + scale_bytes_full
    data_bytes_full: float = 0.0
    scale_bytes_full: float = 0.0
    # Derivation chain for `params`: same length, both emitted so the UI can
    # render `layers × per-unit × repeat = 58 × 14,680,064 × 256`.
    param_factor_labels: list = field(default_factory=list)
    param_factor_values: list = field(default_factory=list)
    is_component: bool = False
    children: list = field(default_factory=list)
    #: How the rank's share reads beside the number: `÷N` for a whole
    #: divisor, the per-rank fraction when the shard is not one, `(mixed)`
    #: for a container of differently sharded leaves.
    shard_label: str = ""
    #: A block group's invocation (`BlockMeta.label`), the identity its
    #: display name (the block's short name) does not carry. None elsewhere.
    template: Optional[str] = None

    def __post_init__(self) -> None:
        if self.divisor > 0:
            self.shard_label = f"÷{self.divisor}"
        elif self.shard_symbolic and self.bytes_full > 0:
            self.shard_label = f"×{self.bytes_per_rank / self.bytes_full:.3f}"
        else:
            self.shard_label = "(mixed)"


def _leaf_weight_shape_str(lt: LeafTensor) -> str:
    """UI-facing weight shape label."""
    if not lt.shape:
        return ""
    # Every axis as its extent: one shape reads in one unit (the embedding
    # printed its vocab axis as `n_vocab` beside numbers, and the LM head of
    # the same vocab as a number).
    return "[" + ", ".join(str(d) for d in lt.shape) + "]"


def _leaf_display_name(lt: LeafTensor) -> str:
    """Drop the parent component_path prefix from display_path so the
    UI shows just the leaf-local name. For primary-folded leaves whose
    display_path collapses entirely onto the wrapper's id (e.g. DSv3
    router's `gate` linear absorbs to `mlp.gate`), fall back to the
    wrapper's name from the path tail."""
    dp = lt.display_path
    if not lt.component_path:
        return dp
    prefix = ".".join(name for name, _op in lt.component_path)
    if dp == prefix:
        # Primary-fold: leaf id IS the wrapper id. Use the wrapper's name.
        return lt.component_path[-1][0]
    if dp.startswith(prefix + "."):
        return dp[len(prefix) + 1:]
    return dp


def _bias_elements(lt: LeafTensor) -> int:
    """The bias's element count: the product of the weight axes the op's
    `.bias` declaration follows (`TensorDecl.axis_refs`). A leaf that says it
    has a bias while its op declares none is a contradiction, not a default."""
    decl = lt.ir_node.declaration
    bias = decl.bias_decl() if isinstance(decl, OpSpec) else None
    if bias is None or bias.axis_refs is None:
        raise ValueError(
            f"weight leaf {lt.tensor_name!r} has a bias but its op "
            f"{lt.op!r} declares no bias tensor with axis references"
        )
    n = 1
    for axis in bias.axis_refs:
        n *= int(lt.shape[axis])
    return n


def _resolve_leaf_bytes(
    lt: LeafTensor,
    rows_by_node: dict,
    fallback_bpe: float,
    fallback_label: str,
) -> tuple:
    """Look up per-leaf storage bytes from the analysis join.

    Returns (data_bytes_full, scale_bytes_full, dtype_label, effective_bpe,
    params_total). The leaf's rows are its node's resident rows
    (`Analysis.iter_resident()`), joined on the node rather than on a name:
    the weight, its bias, the parts a `stored_split` file stores it as — and,
    apart, the format metadata. The parameter count is the rows' elements times their copies,
    the same join the bytes come from, not a second hand-written bias rule.

    A precision profile's weight table lives on the walk, so the rows
    already carry resident bytes. This function does not recast.
    """
    rows = rows_by_node.get(lt.ir_node.uid, ()) if lt.ir_node is not None else ()
    primary = [row for row in rows if not row.is_metadata]
    if not primary:
        if getattr(lt, "from_custom_op", False):
            # A custom-op leaf the join has no row for: no bytes, and the
            # parameter count is the declared shape, which is all there is.
            return (0.0, 0.0, fallback_label, fallback_bpe,
                    _volume(lt.shape) * int(lt.multiplicity))
        raise ValueError(
            f"weight leaf {lt.tensor_name!r} (×{int(lt.multiplicity)}) has no "
            f"checkpoint row; a missing join used to count as 0 bytes"
        )
    data_bf = sum(float(row.storage_bytes) * int(row.count) for row in primary)
    params_total = sum(int(row.elements) * int(row.count) for row in primary)
    scale_bf = sum(float(row.storage_bytes) * int(row.count)
                   for row in rows if row.is_metadata)
    label = primary[0].logical_dtype or primary[0].storage_dtype
    # Named as every other dtype on the page is (`display_name`: FP8 (E4M3),
    # not the header's F8_E4M3); a format the table does not spell keeps
    # its own name, as `Analysis`'s precision labels do.
    label = display_label(label) if label else fallback_label
    eff = (data_bf / params_total) if params_total > 0 else fallback_bpe
    return (data_bf, scale_bf, label, eff, params_total)


def _volume(shape) -> int:
    out = 1
    for d in shape or ():
        out *= int(d)
    return out


def _shape_str(shape) -> str:
    return "×".join(str(int(d)) for d in shape or ()) or "1"




def _shape_knobs(lt: LeafTensor, parallel: ParallelConfig) -> list:
    """The knobs that thin this leaf's copies — a LABEL, not a number.

    Which knob shards an axis is the declaration's to say, and it says it the
    same way whatever the extents work out to. How MUCH each rank ends up with
    is the shape's to say. Reading the first here and the second from the shape
    is what lets the tooltip name `tp_attention` beside a fused projection whose
    volume no `tp_attention`-sized divisor explains.
    """
    tp_fields = dim_to_tp_field()
    fields = {tp_fields[s] for s in set(lt.dim_symbols) & tp_fields.keys()}
    return [_knob_label(f) for f in sorted(fields, key=_knob_order)
            if int(getattr(parallel, f, 1) or 1) > 1]


def _shard_display(bytes_full: float, bytes_per_rank: float,
                   pr_shape, gl_shape, copies_div: float,
                   copies_labels: list, shape_labels: list) -> tuple:
    """The `÷N` badge, the knob label, the per-rank derivation — all DERIVED
    from the numbers above rather than a third source for them — and whether
    the derivation reproduces `bytes_per_rank` (it does not when a bias or
    format metadata is placed by axes of its own; the caller then states the
    sum instead of the ratio).

    `divisor` is the observed ratio when it is a whole number and 0 — the
    `(mixed)` sentinel container nodes already use — when it is not. A fused
    projection genuinely has no integer divisor at some degrees, and printing
    the nearest one is how a display becomes a claim.

    The derivation strings are authored here because this is where the
    derivation is. The front end used to compose `bytes_full / divisor`
    itself, which is a formula rather than a rendering, and it stopped matching
    the number beside it the moment the ratio stopped being one.
    """
    ratio = bytes_full / bytes_per_rank if bytes_per_rank else 0.0
    divisor = (round(ratio)
               if ratio and abs(ratio - round(ratio)) < 1e-9 else 0)
    thinned = _volume(pr_shape) != _volume(gl_shape)

    sym, sub = "bytes_full", f"{bytes_full:.0f}"
    if thinned:
        # A share MULTIPLIES — it is the fraction of the tensor this rank was
        # given, and it is stated as the two shapes so a reader can see WHICH
        # axes moved rather than only by how much.
        sym += " × (per-rank shape / full shape)"
        sub += f" × ({_shape_str(pr_shape)} / {_shape_str(gl_shape)})"
    if copies_div > 1:
        # A count DIVIDES — fewer whole copies of the same tensor. The only
        # knob that does this is pipeline parallelism, which partitions the
        # layer index space; every other degree divides a shape, above.
        sym += f" / ({' × '.join(copies_labels)})"
        # `:.6g` so a whole count still reads `8` while a pipeline stage's
        # genuine 28/10 reads `2.8` rather than seventeen digits of float. An
        # infinite divisor is a template with no layer on the stage being
        # sized; `/ inf` is arithmetic rather than an explanation.
        sub += (" × 0 (no layer of this template on the stage sized)"
                if copies_div == float("inf") else f" / {copies_div:.6g}")

    # Whether the derivation just written reproduces the number it will be
    # shown beside: a display's own consistency, not a second source of
    # the bytes (which stay the placement's).
    claimed = (0.0 if copies_div == float("inf") else
               bytes_full * _volume(pr_shape) / _volume(gl_shape) / copies_div)
    reproduces = abs(claimed - bytes_per_rank) <= 1e-6

    label = " × ".join([*shape_labels, *copies_labels])
    if not (thinned or copies_div > 1):
        return divisor, "1 (replicated)", sym, sub, reproduces
    return divisor, label or "shape", sym, sub, reproduces


def _build_leaf_node(lt: LeafTensor, *args: Any, **kwargs: Any):
    """`_leaf_node` with the leaf named on any refusal it raises."""
    with node_context(lt.display_path or ""):
        return _leaf_node(lt, *args, **kwargs)


def _leaf_node(lt: LeafTensor, prev_class: Optional[str],
                     parallel: ParallelConfig, weight_dtype_bytes: float,
                     weight_dtype_label: str,
                     rows_by_node: dict,
                     v3_weight_leaves: Optional[dict] = None,
                     force_class: Optional[str] = None,
                     pp_copies_div: float = 1.0,
                     routed_uids: FrozenSet[int] = frozenset(),
                     analysis: Any = None) -> TreeNode:
    """Construct a leaf TreeNode from a LeafTensor.

    Per-leaf bytes come from `materialize_for_resolved_spec` (the single
    quant fusion entry). The mat index is built once in
    `per_rank_weight_breakdown` and threaded through.

    Class assignment is `sharding.leaf_class`, from the class of the
    component the leaf sits in.
    """
    walked_class = leaf_class(
        lt, prev_class, routed=lt.ir_node.uid in routed_uids)
    # `force_class` is the model-level caller's answer: embedding / lm_head /
    # aux are named by WHERE the tensor sits in the model, not by what the
    # block-internal class walk would infer from a parent it doesn't have.
    own_class = force_class or walked_class
    # Per-rank storage in its two irreducible halves: how thin each copy is,
    # read off the forward-propagated SHAPE, and how many copies this rank
    # holds, read off the COUNT the leaf's `repeat:` declares. Neither is a
    # restatement of the other and only the second is a divisor.
    # The per-rank shape and the copy count are the layout walk's placement
    # for this leaf — the one source. There is no second derivation to fall
    # back on when it is missing: a leaf the walk did not place is an error
    # in the walk, and answering from a shape ratio instead hid exactly that.
    uid = lt.ir_node.uid
    if v3_weight_leaves is None:
        raise ValueError(
            "weight breakdown needs the layout walk's placements "
            "(`physical_weight_leaves`); none were supplied")
    placement = v3_weight_leaves.get(uid)
    if placement is None:
        raise ValueError(
            f"weight leaf '{lt.display_path or '?'}' "
            f"(uid {uid}) has no layout resident artifact record"
        )
    pr_shape = tuple(placement.runtime_shape)
    gl_shape = tuple(placement.global_shape)
    if gl_shape != tuple(int(d) for d in lt.shape):
        raise ValueError(
            f"weight leaf {lt.display_path!r} has global shape {tuple(lt.shape)!r} "
            f"but its layout primary artifact has {gl_shape!r}"
        )
    # `_copies_dropped` still names the knobs that drop whole copies (the
    # display's labels); the divisor itself is the placement's.
    copies_labels = _copies_dropped(pp_copies_div)
    if placement.runtime_instances == 0:
        copies_div = float("inf")
    else:
        copies_div = (
            float(placement.global_instances) / placement.runtime_instances
        )

    data_bf, scale_bf, leaf_dtype_label, effective_bpe, params_total = (
        _resolve_leaf_bytes(
            lt, rows_by_node,
            fallback_bpe=float(weight_dtype_bytes),
            fallback_label=weight_dtype_label,
        ))
    bytes_full = data_bf + scale_bf
    local_data = float(placement.resident_data_bytes_per_rank)
    local_metadata = float(placement.resident_metadata_bytes_per_rank)
    bytes_per_rank = local_data + local_metadata
    divisor, divisor_source, shard_symbolic, shard_substituted, reproduces = (
        _shard_display(
            bytes_full, bytes_per_rank, pr_shape, gl_shape,
            copies_div, copies_labels, _shape_knobs(lt, parallel)))
    if not reproduces:
        # The derivation string did not reproduce the placement's number:
        # a bias or format metadata owns endpoints of its own, placed by
        # their own axes, and the primary matrix's shape ratio is not a
        # derivation of their sum. The number beside the text is the
        # placement's either way; only the text changes.
        shard_symbolic = "Σ layout resident artifacts"
        shard_substituted = (
            f"{local_data:.0f} data + {local_metadata:.0f} metadata"
        )

    cnt = int(lt.multiplicity) or 1
    leaf_symbolic = ""
    leaf_substituted = ""
    leaf_note = ""
    cost = analysis.cost.values.get(uid) if analysis is not None else None
    record = (
        analysis.params_bindings.values.get(uid)
        if analysis is not None else None)
    if cost is not None:
        meta_per = float(scale_bf) / cnt if cnt else float(scale_bf)
        # Each scale tensor the format stores, as its grid and width, when
        # every layer stores it alike; a total otherwise.
        meta_rows = [row for row in rows_by_node.get(uid, ())
                     if row.is_metadata]
        meta_grids = (
            [(tuple(int(d) for d in row.shape),
              float(row.storage_bytes) / int(row.elements))
             for row in meta_rows]
            if meta_rows and all(int(row.count) == cnt for row in meta_rows)
            else [])
        leaf = params_bytes_formula(
            cost, record, dt_w=float(effective_bpe), meta_bytes=meta_per,
            meta_grids=meta_grids)
        leaf_symbolic = leaf["symbolic"]
        leaf_substituted = leaf["substituted"]
        leaf_note = leaf.get("note") or ""
        if not (gl_shape != pr_shape or copies_div > 1):
            shard_symbolic = leaf_symbolic
            shard_substituted = leaf_substituted

    # Param derivation factors: layers × per-unit × (repeat).
    # multiplicity already folds layer count × expert repeat into one
    # number; we split it back into named factors using the highest-level
    # multiplier (block layer count) as `layers` and the rest as `repeat`
    # so the UI can show `58 × 14,680,064 × 256` for an MoE expert leaf.
    p_labels: list = []
    p_values: list = []
    per_unit = 1
    for d in lt.shape:
        per_unit *= int(d)
    if lt.has_bias:
        # Bias is part of per-unit so the ratio matches engine totals. Which
        # axes of the weight the bias follows is the op's declaration
        # (`TensorDecl.axis_refs`: a linear's output axis, a per-head
        # linear's head and output axes), not a rule per op name.
        per_unit += _bias_elements(lt)
    p_labels.append("per-unit")
    p_values.append(per_unit)
    if lt.multiplicity > 1:
        # Heuristic split: factor multiplicity into one or two factors
        # based on whether a `layers` count is reasonable. For top-level
        # leaves multiplicity is 1; for block-internal it's at least the
        # layer count, possibly times an expert repeat.
        # We don't know the exact split here without the block group's
        # layer count; the caller can refine. For now emit as one
        # `multiplicity` factor.
        p_labels.insert(0, "multiplicity")
        p_values.insert(0, int(lt.multiplicity))

    return TreeNode(
        name=_leaf_display_name(lt),
        op=lt.op,
        class_=own_class,
        params=params_total,
        bytes_full=bytes_full,
        divisor=divisor,
        bytes_per_rank=bytes_per_rank,
        divisor_source_label=divisor_source,
        shard_symbolic=shard_symbolic,
        shard_substituted=shard_substituted,
        uid=uid,
        weight_shape=_leaf_weight_shape_str(lt),
        weight_dtype=leaf_dtype_label,
        weight_dtype_bytes=float(effective_bpe),
        data_bytes_full=data_bf,
        scale_bytes_full=scale_bf,
        param_factor_labels=p_labels,
        param_factor_values=p_values,
        is_component=False,
        leaf_symbolic=leaf_symbolic,
        leaf_substituted=leaf_substituted,
        leaf_cnt=cnt,
        leaf_note=leaf_note,
        leaf_copy_factors=_copy_factors(
            lt, analysis.copy_factors.values.get(uid)
            if analysis is not None else None),
        children=[],
    )


def _container_divisor(children: List[TreeNode]) -> tuple:
    """The `÷N` a container may print, or 0 for mixed — plus what named it.

    Read off the children's RATIOS, not their `divisor` fields: a child whose
    ratio is not whole reports 0, and skipping those would ask the question of
    a subset. An attention block holding one fused QKV at 6/36 beside three
    projections at 1/8 would inherit the 8 and print a `bytes_full / 8` that is
    not its own `bytes_per_rank` — a divisor is a claim about THIS node's two
    numbers, and it may only be printed when it describes them.
    """
    ratios = {round(c.bytes_full / c.bytes_per_rank, 9)
              for c in children if c.bytes_full > 0 and c.bytes_per_rank > 0}
    one = next(iter(ratios)) if len(ratios) == 1 else 0.0
    divisor = round(one) if one and abs(one - round(one)) < 1e-9 else 0
    if divisor <= 0:
        return 0, ""
    return divisor, next((c.divisor_source_label for c in children
                          if c.divisor == divisor and c.divisor_source_label), "")


def _build_component_node(name: str, op: str, class_: Optional[str],
                          children: List[TreeNode],
                          parallel: ParallelConfig,
                          weight_dtype_label: str,
                          effective_bpe: float) -> TreeNode:
    """Wrap a list of child TreeNodes in a component-invocation TreeNode.
    Aggregates bytes/params from children; divisor is inherited only when
    every child shares the same divisor, else marked as mixed (0)."""
    bytes_full = sum(c.bytes_full for c in children)
    data_bf = sum(c.data_bytes_full for c in children)
    scale_bf = sum(c.scale_bytes_full for c in children)
    bytes_per_rank = sum(c.bytes_per_rank for c in children)
    params = sum(c.params for c in children)
    # Component display divisor: inherited only when every weight-bearing child
    # sheds the same FRACTION of itself, and that fraction is a whole divisor.
    # Read off the ratios, not off the children's `divisor` fields: a child
    # whose ratio is not whole reports 0, and skipping those asked the question
    # of a subset — an attention block holding one fused QKV at 6/36 beside
    # three projections at 1/8 would inherit the 8 and print a `bytes_full / 8`
    # that is not its own `bytes_per_rank`. Numeric bytes_per_rank is the child
    # sum regardless; this drives only the `÷N` label and the derivation line.
    effective_divisor, divisor_source = _container_divisor(children)
    if effective_divisor > 0 and class_:
        shard_symbolic = f"bytes_full / ({divisor_source})"
        shard_substituted = f"{bytes_full:.0f} / {effective_divisor}"
    else:
        shard_symbolic = "Σ children bytes_per_rank"
        shard_substituted = f"{bytes_per_rank:.0f}"
    return TreeNode(
        name=name,
        op=op,
        class_=class_,
        params=params,
        bytes_full=bytes_full,
        divisor=effective_divisor,
        bytes_per_rank=bytes_per_rank,
        divisor_source_label=divisor_source,
        shard_symbolic=shard_symbolic,
        shard_substituted=shard_substituted,
        weight_dtype=weight_dtype_label,
        weight_dtype_bytes=float(effective_bpe),
        data_bytes_full=data_bf,
        scale_bytes_full=scale_bf,
        is_component=True,
        children=children,
    )


def _component_at(lt: LeafTensor, depth: int) -> Any:
    """The component invocation at step `depth` of `lt.component_path`:
    its `depth`-th component ancestor, from the template root down."""
    chain = []
    node = lt.ir_node.parent
    while node is not None:
        if node.is_component:
            chain.append(node)
        node = node.parent
    chain.reverse()
    found = chain[depth]
    if (found.name, found.op) != tuple(lt.component_path[depth]):
        raise ValueError(
            f"{lt.ir_node.display_path}: component path step {depth} "
            f"{lt.component_path[depth]!r} is not its ancestor "
            f"{(found.name, found.op)!r}")
    return found


def _build_subtree(leaves: List[LeafTensor], depth: int,
                   prev_class: Optional[str],
                   parallel: ParallelConfig, weight_dtype_bytes: float,
                   weight_dtype_label: str,
                   rows_by_node: dict,
                   v3_weight_leaves: Optional[dict] = None,
                   pp_copies_div: float = 1.0,
                   routed_uids: FrozenSet[int] = frozenset(),
                   analysis: Any = None) -> List[TreeNode]:
    """Build TreeNode children for a single level of the spec hierarchy.

    Walks `leaves` (already in spec order) and partitions them by
    `component_path[depth]`:
      - leaves whose path length == depth are direct children at this
        level (emit as leaf TreeNode);
      - consecutive leaves sharing a path[depth] tuple are wrapped in a
        component TreeNode and recursed into.

    iter_leaves preserves spec graph emission order, so leaves under the
    same component invocation are guaranteed contiguous — a single
    forward scan recovers the structural grouping without re-walking
    the spec.
    """
    children: List[TreeNode] = []
    i = 0
    n = len(leaves)
    while i < n:
        lt = leaves[i]
        if depth >= len(lt.component_path):
            children.append(_build_leaf_node(
                lt, prev_class, parallel, weight_dtype_bytes,
                weight_dtype_label, rows_by_node,
                v3_weight_leaves=v3_weight_leaves,
                pp_copies_div=pp_copies_div,
                routed_uids=routed_uids,
                analysis=analysis,
            ))
            i += 1
            continue
        step = lt.component_path[depth]
        # Find the run of leaves that share this exact step at this depth.
        j = i + 1
        while j < n:
            lt2 = leaves[j]
            if depth >= len(lt2.component_path) or lt2.component_path[depth] != step:
                break
            j += 1
        cname, cop = step
        new_class = step_class(prev_class, _component_at(lt, depth).role)
        sub_children = _build_subtree(
            leaves[i:j], depth + 1, new_class,
            parallel, weight_dtype_bytes, weight_dtype_label,
            rows_by_node,
            v3_weight_leaves=v3_weight_leaves,
            pp_copies_div=pp_copies_div,
            routed_uids=routed_uids,
            analysis=analysis,
        )
        # A call inside an MoE layer shows the part of it that its weights
        # are, when they are one part (the gate, the routed bank, the
        # shared expert); one holding several is the layer itself.
        shown = new_class
        if prev_class == "ffn_moe":
            parts = {moe_child_class(x.role, x.ir_node.uid in routed_uids)
                     for x in leaves[i:j]}
            if len(parts) == 1:
                (shown,) = parts
        children.append(_build_component_node(
            name=cname, op=cop, class_=shown,
            children=sub_children, parallel=parallel,
            weight_dtype_label=weight_dtype_label,
            effective_bpe=float(weight_dtype_bytes),
        ))
        i = j
    return children


def shard_degrades(analysis: Any) -> list[str]:
    """One warning per DEGREE whose axes could not honour it.

    One sentence per (degree asked for, pieces the layout settled on) —
    the clause a reader is actually told — naming every axis that landed
    there with its own extent. Keying the sentence on the extent as well
    split one fact into a paragraph: a TP16 GQA card said "`n_heads` (24
    entries) … cut into 8 pieces" and then the same sentence again for
    `n_kv_heads` (8 entries), because the two axes differ in extent while
    saying the same thing about the degree.

    Per degree rather than per axis, and that is the whole of the change
    from the clamp this replaces. That clamp capped `tp_attention` at
    `n_kv_heads` and then
    applied the cap to every attention axis at once, which is a claim about
    the query projections that no framework makes: vLLM's `QKVParallelLinear`
    splits `total_num_heads` by the FULL `tp_size` and, once `tp_size` reaches
    `total_num_kv_heads`, gives each rank ONE kv head and replicates it
    `tp_size / total_num_kv_heads` ways (`num_kv_head_replicas`, in
    `model_executor/layers/linear.py`). Two axes, two answers, from one
    degree. Serving both from the smaller of them put 1.8 GiB of q_proj and
    o_proj on every rank of a TP8 glm-4-9b that the rank does not hold — 80%
    over on a 2.2 GiB model, in the direction that reports a model as not
    fitting.

    Nothing is clamped and no deployment is rewritten. The warning is read
    from the exact layouts carried by the analysis, so it cannot
    disagree with the shapes and storage being reported.
    """
    from backend.arch.layout.physical import physical_axis_partition_audit

    # Axes one degree fails on the same way (MLA's `n_heads` and
    # `n_kv_heads`; GQA's 24 heads and 8 kv heads) are one fact: one
    # sentence names them all, each with its own extent.
    groups: dict[tuple, list[tuple[str, int]]] = {}
    for record in physical_axis_partition_audit(analysis):
        if record.applied == record.requested:
            continue
        key = (record.degree, record.requested, record.applied)
        axes = groups.setdefault(key, [])
        if record.symbol not in [symbol for symbol, _ in axes]:
            axes.append((record.symbol, record.extent))
    warnings: list[str] = []
    for (degree, requested, applied), axes in groups.items():
        # Said in the deployment's words: the degree asked for, the axes it
        # cannot divide that finely, and what the GPUs hold instead.
        many = len(axes) > 1
        it = "each" if many else "it"
        held = (f"every GPU keeps all of {'them' if many else 'it'}"
                if applied == 1
                else f"{it} is cut into {applied} pieces and the "
                     f"other GPUs hold copies of them")
        named = [f"`{symbol}` ({extent} {'entry' if extent == 1 else 'entries'})"
                 for symbol, extent in axes]
        names = " and ".join([", ".join(named[:-1]), named[-1]]
                             if many else named)
        warnings.append(
            f"{degree or 'A parallel degree'} = {requested} "
            f"would split {names} "
            f"{requested} ways, which {'they' if many else 'it'} cannot be: "
            f"{held}."
        )
    return warnings


def per_rank_weight_breakdown(
    resolved,
    parallel: ParallelConfig,
    *,
    precision: Any = None,
    quant_granularity: str = "per-tensor",
    published_weight_bytes: Optional[float] = None,
    include_aux_weights: bool = False,
    pp_stage: Optional[int] = None,
    analysis: Any = None,
    artifacts: Any = None,
) -> dict:
    """Compute a hierarchical weight breakdown.

    `pp_stage` names which pipeline stage the tree describes, required at
    `pp > 1`: which stage binds is capacity's memory question (it weighs the
    stages' weights against their cache), so the tree is built for the stage
    it settled on, from that stage's weight projection
    (`stages.StageView.weights`), and the tree and the number beside it
    describe one rank.

    Per-leaf storage bytes come from `materialize_for_resolved_spec`
    (CLAUDE.md §2.3: single quant fusion entry). Mixed precision
    (FP8 attention + BF16 norms + INT4 experts) flows through naturally
    — no uniform-dtype proxy on the numeric path.

    The tree's "Weight Dtype" label is the walk's dominant resident dtype
    (`Analysis.dominant_weight_dtype`); every byte comes from the rows.

    `precision` (a `PrecisionProfile`, as shipped when None) is the walk's,
    so the rows already carry resident bytes. The tree itself does not
    recast. A caller passing `analysis` passes the walk it priced.

    `quant_granularity` is deprecated; it was an input to the old
    synthetic per-element overhead and is now ignored by storage math.
    Kept on the signature for caller compatibility. (The companion
    `scale_dtype_bytes` knob was removed once every caller stopped
    depending on it.)

    `published_weight_bytes` is surfaced informationally only.
    """
    # Per-rank weight GEOMETRY, from the forward transform — the same walk the
    # cost path runs, at this deployment. What a rank stores is the shape it
    # was left with; the alternative is a scalar divisor recovered from the
    # leaf's axis names, and that cannot express a fused projection's volume
    # (its axes ADD) nor tell a thinner tensor from one a pipeline stage
    # holds fewer layers of.
    from .step_eval import evaluate_step

    sequence_extent = max(
        1,
        int(parallel.cp or 1),
        (
            int(parallel.tp_attention or 1)
            if parallel.sequence_parallel else 1
        ),
    )
    def _walk():
        return evaluate_step(
            resolved,
            parallel,
            phase=PREFILL,
            batch=1,
            input_tokens=sequence_extent, output_tokens=0,
            precision=precision,
        ).analysis

    if analysis is None:
        analysis = _walk()

    if artifacts is None:
        from backend.arch.layout.persistent import physical_artifacts

        artifacts = physical_artifacts(analysis)

    from backend.arch.layout.persistent import physical_weight_leaves

    # The walk folds one shared CLONE of the spec, so the analysis's program is
    # never built from the caller's `resolved` OBJECT. `Program.is_for` answers
    # "same program?" (config identity) from the eval side, so this layer asks
    # rather than reaching into `resolved.config` itself.
    if not analysis.program.is_for(resolved):
        raise ValueError(
            "per_rank_weight_breakdown received an Analysis from a "
            "different resolved program"
        )
    if analysis.rule.parallel != parallel:
        raise ValueError(
            "per_rank_weight_breakdown received an Analysis from a "
            "different deployment"
        )
    program = analysis.program
    # Leaves of THIS program's IR, so `lt.ir_node.uid` is the analysis uid.
    # `iter_leaves(resolved)` is a different object (the walk clones) and
    # its uids do not join.
    leaves = iter_leaves(program.resolved)
    block_metas = iter_blocks(program.resolved)
    v3_weight_leaves = dict(physical_weight_leaves(
        analysis, artifacts=artifacts
    ))
    parallel_warnings = shard_degrades(analysis)
    included_stacks = program.included_stack_names(
        include_aux=include_aux_weights,
    )
    backbone_stacks = program.included_stack_names()

    caches = derive_cache(analysis)

    # --- 0. Index the RESIDENT rows from the same Analysis the rest of this
    # function already holds, by the node that stores them: a leaf is one
    # node, and its weight, bias, split parts, literal-index paths and
    # format metadata are that node's rows. Keyed by uid, same as
    # `physical_weight_leaves`.
    #
    # `iter_resident()`, not `iter_checkpoint()`: this page is what a rank
    # holds, and the file's list answers a different question. The two
    # coincided for every model until one did not — a weight rule's F32
    # attention moved the deployment total by nothing, and gemma-2's F32
    # checkpoint would have been billed four bytes a weight on a rank that
    # holds two.
    rows_by_node: dict = {}
    for row in analysis.iter_resident():
        rows_by_node.setdefault(row.uid, []).append(row)

    # Scale metadata of a node that stores no weight row has no leaf to
    # land on; it is summed as a replicated residual below.
    unmatched_scale_bytes_total = 0.0
    unmatched_scale_examples: list = []
    for rows in rows_by_node.values():
        if any(not row.is_metadata for row in rows):
            continue
        for row in rows:
            unmatched_scale_bytes_total += float(row.storage_bytes) * int(row.count)
            if len(unmatched_scale_examples) < 3:
                unmatched_scale_examples.append(row.name)

    _dominant = analysis.dominant_weight_dtype()
    weight_dtype_bytes = float(analysis.dominant_weight_dtype_bytes() or 0.0)
    effective_bpe = weight_dtype_bytes
    weight_dtype_label = display_name(_dominant) if _dominant else ""

    # Index leaves by template for the block-group pass. Assignment-mode
    # specs may invoke the same synthesized template more than once: one
    # selector can resolve to zero layers while a later ``rest`` selector owns
    # the live layers (GLM-5.1 is the concrete case). Both invocations
    # share the template IR (same uid). Feeding both leaf lists to one
    # group maps each duplicate to the one live PhysicalArtifact and
    # bills it twice. Keep the unique live declaration per uid; two live
    # declarations remain an ambiguity and are rejected by
    # ``local_weight_leaves`` above.
    leaves_by_template: dict[str, list[LeafTensor]] = {}
    leaf_index: dict[tuple[str, int], int] = {}
    for lt in leaves:
        bucket = leaves_by_template.setdefault(lt.template, [])
        key = (lt.template, lt.ir_node.uid)
        previous_index = leaf_index.get(key)
        if previous_index is None:
            leaf_index[key] = len(bucket)
            bucket.append(lt)
            continue
        previous = bucket[previous_index]
        previous_live = int(previous.multiplicity) > 0
        candidate_live = int(lt.multiplicity) > 0
        if previous_live and candidate_live:
            raise ValueError(
                f"duplicate live weight leaf {lt.display_path!r} in template "
                f"{lt.template!r}"
            )
        if candidate_live:
            bucket[previous_index] = lt

    _reject_ambiguous_template_names(block_metas)

    # --- 1. Block-group tree: one TreeNode per BlockMeta ---
    # Everything the leaf builder reads that is the same for every template.
    subtree_env = dict(
        parallel=parallel, weight_dtype_bytes=weight_dtype_bytes,
        weight_dtype_label=weight_dtype_label, rows_by_node=rows_by_node,
        v3_weight_leaves=v3_weight_leaves,
        routed_uids=program.routed_uids,
        analysis=analysis,
    )

    # What the described stage holds (`StageView.weights`): its layers of
    # each block, the units placed on it, and the tables it reads. Empty at
    # pp=1, which is every deployment that does not pipeline, so a
    # single-stage model reaches the same builder it always did. The
    # per-block divisor is only the leaves' label (`_copies_dropped`); the
    # bytes are the re-counted artifacts'.
    _pp = max(1, int(parallel.pp or 1))
    pp_divisors: Dict[int, float] = {}
    pipeline: Optional[dict] = None
    if _pp > 1:
        if pp_stage is None:
            raise ValueError(
                f"pp={_pp}: name the pipeline stage the weight tree "
                f"describes (`pp_stage`)")
        placement = StagePlacement.of(analysis, _pp)
        view = placement.view(pp_stage)
        whole = placement.block_counts(None)
        kept = placement.block_counts(pp_stage)
        pp_divisors = {i: (whole[i] / n) if n else float("inf")
                       for i, n in kept.items() if whole[i]}
        subtree_env["v3_weight_leaves"] = dict(
            physical_weight_leaves(
                analysis,
                artifacts=artifacts,
                layer_multiplicity=view.weights(included_stacks),
            )
        )
        parallel_warnings.append(
            f"pp={_pp} is reported on the BINDING one of the {_pp} stages "
            f"— {view.describe()} — since that is the one a rank has to "
            f"survive. A block template with no layer on it reports 0 B "
            f"here, and the embedding, the head and any aux stack count "
            f"on the stage placement puts them on (a tied table also where "
            f"it is read).")
        pipeline = {
            "n_stages": _pp,
            "stage": pp_stage,
            "stage_labels": [placement.view(s).describe()
                             for s in range(_pp)],
        }
    groups: list[TreeNode] = []
    attention_kind: Optional[str] = None
    num_layers = sum(
        template.stored_count
        for block_index, template in enumerate(program.blocks)
        if (template.meta.indexed
            and program.stack_of_block(block_index) in backbone_stacks)
    )
    for meta_index, meta in enumerate(block_metas):
        # The invocation's name, as its leaves carry it: a derived group's
        # names what its layers see.
        tmpl = meta.label
        count = meta.count
        stack_name = meta.stack_name
        # A zero-layer group contributes no weight — skip it. This happens
        # when a `config_list` assignment matches no layers (e.g. GLM-5.1 has
        # no `indexer_types`, so the full/shared selectors are empty and a
        # sibling catches those layers) or an empty `rest:`. The template's
        # leaves carry the FULL multiplicity, so emitting a group per meta
        # regardless of count would re-add the whole template's bytes and
        # double-count (a count-0 group otherwise reported a full-size spurious).
        if count == 0:
            continue
        # A call (`ModelLayer.call`) holds no tensors of its own: they are
        # the called layer's.
        if meta.tied_to is not None:
            continue
        block_stack = program.stack_of_block(meta_index)
        tmpl_leaves = leaves_by_template.get(tmpl, [])
        if not tmpl_leaves:
            continue
        # The group is the invocation (`stack_of_block`). Per-leaf
        # `stack_of` can disagree: a cloned/shared inner node (MTP
        # embed, vision patch embed) is walked under another stack
        # while `iter_leaves` emits it once under this template.
        # Raising on that dropped every MTP/vision model from the
        # gallery. included_stacks still drops a whole aux group when
        # the caller asked for backbone only.
        if block_stack not in included_stacks:
            continue
        node_trees = _build_subtree(
            tmpl_leaves, depth=0, prev_class=None,
            pp_copies_div=pp_divisors.get(meta_index, 1.0), **subtree_env,
        )
        # Detect attention kind from Cache.label on the enclosing
        # attention call. Roles include the slot name (`attention`,
        # `attention_swa`); both normalize to the attention class.
        if attention_kind is None:
            for lt in tmpl_leaves:
                if is_attention_role(lt.role):
                    attention_kind = _attention_kind_from_cache(caches, lt)
                    if attention_kind:
                        break

        # Shown by the block's short name (`BlockMeta.short_name`), as on
        # every other page; a derived group says what its layers read.
        group_label = f"{meta.short_name or tmpl} × {count}" + "".join(
            f" (reads {m} {stream})" for stream, m in meta.visible) + (
            f"  [{stack_name}]" if stack_name and stack_name != "layers" else ""
        )
        # A block group aggregates a whole template — attention beside FFN
        # beside norms — so a divisor is printable only in the degenerate case
        # where every one of them shed the same whole fraction. It used to be
        # hardcoded to 1, which read as "this block is replicated" on every
        # sharded model in the corpus.
        group_divisor, _group_src = _container_divisor(node_trees)
        groups.append(TreeNode(
            name=group_label, op="block", class_=None, template=tmpl,
            params=sum(n.params for n in node_trees),
            bytes_full=sum(n.bytes_full for n in node_trees),
            divisor=group_divisor,
            bytes_per_rank=sum(n.bytes_per_rank for n in node_trees),
            data_bytes_full=sum(n.data_bytes_full for n in node_trees),
            scale_bytes_full=sum(n.scale_bytes_full for n in node_trees),
            is_component=False, children=node_trees,
        ))

    # --- 2. Model-level nodes: embedding + lm_head (top-level leaves) ---
    # These go through the SAME leaf builder as everything inside a block.
    # A top-level leaf has no component path, so `_leaf_display_name` returns
    # its display path unchanged and the class walk has no parent to inherit
    # from — the only thing the caller still has to say is the class, which it
    # passes explicitly. What the shared builder buys is that a vocab-sharded
    # embedding gets its per-rank bytes from the same forward-propagated shape
    # as every other tensor, instead of from a second, private derivation.
    #
    # Which top-level leaf is a vocab endpoint is `vocab_kind_of_leaf`'s question,
    # asked here in the leaf keyspace off the leaf's declared `boundary_role`
    # (input_embedding / output_head). This is only the DISPLAY class — which
    # card the parameter shows under; placement is independent and reads
    # `weight_axes`. Naming the two by display path instead — `dp ==
    # "embed_tokens"` — answered `aux` for every model that spells them
    # differently, so gemma-4's `language_model.embed_tokens` and GLM-4's
    # `transformer.output_layer` were shown replicated while the shape layer
    # vocab-sharded them.
    model_level: list[TreeNode] = []
    vocab_class = {"embed": "embedding", "lm_head": "lm_head"}
    for lt in leaves_by_template.get("", []):
        leaf_stack = program.stack_of(lt.ir_node)
        if leaf_stack not in included_stacks:
            continue
        # Final norm and other top-level params (Llama's `model.norm`,
        # vision-tower-final norms, MiMo's speech embedding bank whose table
        # is its own smaller vocab) — aux-class, replicated.
        cls = vocab_class.get(vocab_kind_of_leaf(lt), "aux")
        model_level.append(_build_leaf_node(
            lt, prev_class=None,
            parallel=parallel, weight_dtype_bytes=weight_dtype_bytes,
            weight_dtype_label=weight_dtype_label,
            rows_by_node=rows_by_node,
            v3_weight_leaves=subtree_env.get("v3_weight_leaves"),
            force_class=cls,
            analysis=subtree_env.get("analysis"),
        ))

    # --- 3. Residual: unattributed scale metadata (defensive) ---
    # Every weight leaf — including fused custom-op weights (gpt-oss mxfp4
    # `_blocks`) — now resolves its bytes through the role-sharded tree, and
    # its scale siblings attribute to it above. This residual only catches a
    # genuinely orphaned scale tensor (a malformed spec whose metadata names a
    # parent that no leaf claims); divisor=1 is the conservative default.
    if unmatched_scale_bytes_total > 0:
        parallel_warnings.append(
            f"{unmatched_scale_bytes_total:.0f} B of scale metadata could "
            f"not be attributed to a parent leaf (examples: "
            f"{', '.join(unmatched_scale_examples)}); summed as replicated residual."
        )

    by_stack = analysis.stored_bytes_by_stack()

    # --- 4. Deployment totals ---
    deployment_total = (
        sum(g.bytes_full for g in groups)
        + sum(m.bytes_full for m in model_level)
        + unmatched_scale_bytes_total
    )

    total_bytes_per_rank = (
        sum(g.bytes_per_rank for g in groups)
        + sum(m.bytes_per_rank for m in model_level)
        + unmatched_scale_bytes_total      # divisor=1 (replicated)
    )

    # The two addends of the leaves' share of `deployment_total_bytes`: the
    # weight data and its scale metadata, each folded up the same tree (every
    # leaf's split comes from `Analysis.tensors()`). Published once here so
    # the Memory card and the deployment formula read them, not re-sum.
    deployment_data = (sum(g.data_bytes_full for g in groups)
                       + sum(m.data_bytes_full for m in model_level))
    deployment_scale = (sum(g.scale_bytes_full for g in groups)
                        + sum(m.scale_bytes_full for m in model_level))

    return {
        "total_bytes_per_rank": total_bytes_per_rank,
        "deployment_total_bytes": deployment_total,
        "deployment_data_bytes": deployment_data,
        "deployment_scale_bytes": deployment_scale,
        # The scale metadata no weight leaf claims — the third term of
        # `deployment_total_bytes` beside the groups' and model-level bytes.
        "unattributed_scale_bytes": unmatched_scale_bytes_total,
        "pipeline": pipeline,
        "published_checkpoint_bytes": published_weight_bytes,
        # The stacks the model stores that this deployment does not load —
        # an MTP head with aux weights off, an encoder whose weights are its
        # own encode phase's — with their bytes at the walk's precision: why
        # the deployment total is short of the published checkpoint.
        "left_out_stacks": [
            {"stack": stack.display_name(stack.name),
             "aux_kind": stack.aux_kind,
             "encoder": stack.is_encoder,
             "bytes": by_stack[stack.name]}
            for stack in program.stacks
            if stack.name not in included_stacks
            and by_stack.get(stack.name, 0) > 0
        ],
        "groups": [_tree_to_dict(g) for g in groups],
        "model_level": [_tree_to_dict(m) for m in model_level],
        "attention_class": attention_kind or "unknown",
        "num_layers": num_layers,
        "spec_name": resolved.spec.name,
        "weight_dtype": weight_dtype_label,
        "weight_dtype_bytes": weight_dtype_bytes,
        "effective_bytes_per_elem": effective_bpe,
        # Legacy sentinel — scales are now sourced per-leaf from materialize.
        "quant_overhead_per_elem": 0.0,
        "quant_granularity": quant_granularity,
        "parallel_warnings": parallel_warnings,
    }


def _scale_tree(trees: list[TreeNode], scale: float,
                exclude_class: Optional[set] = None) -> None:
    """Scale each node's own bytes in place by `scale`.

    Because parent and child both already carry their own authoritative
    bytes (component params includes any `repeat`), we just scale each
    level uniformly. The `bytes_full = sum(children)` invariant held at
    construction time continues to hold after a uniform scale.
    """
    for t in trees:
        if not (exclude_class and t.class_ in exclude_class):
            t.bytes_full *= scale
            # Uniform scale preserves the "sum-of-children" invariant for
            # container nodes; don't recompute from bytes_full/divisor
            # since container divisors may be mixed (sentinel 0).
            t.bytes_per_rank *= scale
        if t.children:
            _scale_tree(t.children, scale, exclude_class)


def _tree_to_dict(t: TreeNode) -> dict:
    d = asdict(t)
    # asdict doesn't recurse dataclass field 'children' names cleanly —
    # it does actually, but we also want rounded floats for JSON size.
    return d
