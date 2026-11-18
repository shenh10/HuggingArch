"""The model's composition: what each kind of block is made of, read off the
spec and the checkpoint alone.

One row per block declaration (the derived groups a `read: all` history split
it into are joined back under it, as `declarations` does for the per-op
table), one per model-level node that stores weights (embedding, final norm,
head). Every figure is a quantity of the model, not of a run: parameters by
role, the active share, the stored bytes by width, and the KV cache a layer
adds per token — or, for a recurrent state, holds per sequence. None of it
depends on a batch, a sequence length or a device.

Each number is read from the ledger that owns it: params from the projected
block reports, bytes from `capacity.weights` (`Weights.by_root`), KV from
`CacheView.growth_and_floor_by_block`. The rows are those ledgers grouped, so
they add up to the dashboard's totals, which are the same ledgers summed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.arch.eval import capacity as capacity_mod
from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.arch.eval.dim import expr_of
from backend.arch.dtypes import display_name
from backend.arch import roles
from backend.arch.ops import owns_lookup_table
from backend.inference.declaration_groups import declaration_key
from backend.views.declarations import histogram


def _call_kinds() -> Tuple[str, ...]:
    """The role families a call is lined up under, in the order the
    composition table shows them.

    WHICH families is this view's decision — a block's norms are not calls,
    and a lookup memory is counted apart from both (`roles.yaml`,
    `memory`) — but the words are the registry's, so a kind this view no
    longer shares with `roles.yaml` is refused here, at import. Left
    unchecked it reads like a live rule and lines up nothing:
    `roles.kind_of` answers `""` for a family it does not know, and every
    call would simply stop being found.
    """
    kinds = ("attention", "ffn")
    unknown = sorted(set(kinds) - roles.KINDS)
    if unknown:
        raise ValueError(
            f"the composition view lines calls up under role kinds "
            f"{unknown}, which roles.yaml does not classify; its kinds are "
            f"{sorted(roles.KINDS)}")
    return kinds


#: The role families a call is lined up under (`roles.yaml` `kind`).
CALL_KINDS: Tuple[str, ...] = _call_kinds()


def call_signature(node: Any) -> Dict[str, Any]:
    """What a component call IS: the component it runs (its slots already
    bound — `op` is the resolved name) and, by node name, the sub-calls its
    body makes, each the same way. Two calls of one component whose slots
    bind different components differ here, which a component name alone
    does not show."""
    return {"component": node.op,
            "calls": {child.name: call_signature(child)
                      for child in node.children or () if child.is_component}}


def calls_of(analysis: Any, template: Any) -> List[Dict[str, Any]]:
    """The outermost attention and FFN calls of one block invocation:
    each call's role family, signature, layer count, and its parameters
    and active parameters over those layers (the walk's `Cost`)."""
    out: List[Dict[str, Any]] = []
    pending = list(template.roots)
    while pending:
        node = pending.pop(0)
        kind = roles.kind_of(node.role or "")
        if node.is_component and kind in CALL_KINDS:
            cost = analysis.cost.values.get(node.uid)
            count = int(template.count)
            out.append({
                "role": kind,
                "signature": call_signature(node),
                "layers": count,
                "params": int(cost.params) * count if cost else 0,
                "active_params": int(cost.active_params) * count if cost else 0,
            })
            continue
        pending.extend(node.children or ())
    return out


def root_kind(name: str, node: Any) -> Tuple[str, str]:
    """`(layer_type, label)` of a model-level node: the main lookup table is
    "Token Embedding" (which one is main is the spec's `boundary_role`), the
    model's head is "LM Head", and every other node keeps its spec name."""
    if owns_lookup_table(node.declaration):
        return ("embedding", "Token Embedding"
                if node.boundary_role == "input_embedding" else name)
    if node.is_lm_head:
        return "lm_head", "LM Head"
    return "model_root", name


_DIM_MEANINGS = {
    "d_model": "hidden-state width",
    "d_head": "attention head width",
    "d_v_head": "attention value-head width",
    "d_qk_nope": "non-rotary query/key width",
    "d_qk_rope": "rotary query/key width",
    "d_q_lora": "query low-rank width",
    "d_kv_lora": "KV low-rank width",
    "d_ffn": "dense FFN intermediate width",
    "d_ffn_expert": "one expert's intermediate width",
    "n_heads": "number of attention heads",
    "n_kv_heads": "number of KV heads",
    "n_route_targets": "number of routed experts",
    "n_expert_copies": "number of stored expert copies",
    "n_experts_active": "experts selected for one token",
}


def _role_roots(nodes: List[Any], kind: str) -> List[Any]:
    """Outermost report nodes counted by projection's role classifier."""
    out: List[Any] = []

    def visit(node: Any) -> None:
        node_kind = roles.kind_of(node.role or "")
        if node_kind == kind:
            out.append(node)
            return
        if node.role is not None and node_kind != "wrapper":
            return
        for child in node.children or ():
            visit(child)

    for node in nodes:
        visit(node)
    return out


def _routed_roots(nodes: List[Any], program: Any) -> List[Any]:
    """Outermost routed banks, the same boundary as `routed_params`."""
    out: List[Any] = []

    def visit(node: Any) -> None:
        if node.ir_node is not None and program.is_routed(node.ir_node.uid):
            out.append(node)
            return
        for child in node.children or ():
            visit(child)

    for node in nodes:
        visit(node)
    return out


def _other_roots(nodes: List[Any]) -> List[Any]:
    """Parameter-bearing roots outside attention and FFN classifiers."""
    out: List[Any] = []

    def visit(node: Any) -> None:
        node_kind = roles.kind_of(node.role or "")
        if node_kind in CALL_KINDS:
            return
        if node.role is not None and node_kind != "wrapper":
            if node.params:
                out.append(node)
            return
        if not node.children:
            if node.params:
                out.append(node)
            return
        for child in node.children:
            visit(child)

    for node in nodes:
        visit(node)
    return out


def _sum_expr(nodes: List[Any], symbolic_attr: str = "params_symbolic",
              values_attr: str = "params_values") -> Tuple[str, str, int]:
    held = [n for n in nodes if n.params]
    symbolic = " + ".join(
        f"({getattr(n, symbolic_attr) or n.params})" for n in held) or "0"
    values = " + ".join(
        f"({getattr(n, values_attr) or n.params})" for n in held) or "0"
    return symbolic, values, sum(int(n.params) for n in held)


def _bound_expr(symbolic: str, dims: Dict[str, int]) -> str:
    """Keep the expression visible while replacing each named dimension."""
    return re.sub(
        r"[A-Za-z_][A-Za-z_0-9]*",
        lambda match: str(dims.get(match.group(0), match.group(0))),
        symbolic,
    )


def _active_expr(nodes: List[Any], analysis: Any,
                 dims: Dict[str, int]) -> Tuple[str, str, int]:
    parts: List[Tuple[str, str, int]] = []
    for node in nodes:
        cost = analysis.cost.values.get(node.ir_node.uid)
        value = int(node.active_params)
        if cost is None or not value:
            continue
        symbolic = str(expr_of(cost.active_params))
        parts.append((symbolic, _bound_expr(symbolic, dims), value))
    return (
        " + ".join(f"({p[0]})" for p in parts) or "0",
        " + ".join(f"({p[1]})" for p in parts) or "0",
        sum(p[2] for p in parts),
    )


def _parameter_term(label: str, layers: int, symbolic: str, substituted: str,
                    result: int, dims: Dict[str, int]) -> Dict[str, Any]:
    return {
        "label": label,
        "symbolic": f"n_layers × ({symbolic})",
        "substituted": f"{layers} × ({substituted})",
        "result": result,
        "_dims": dims,
    }


def _formula_defs(symbolic: str, terms: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    values: Dict[str, set[str]] = {}
    for term in terms:
        for name, value in term.get("_dims", {}).items():
            values.setdefault(name, set()).add(str(value))
    names = list(dict.fromkeys(re.findall(
        r"[A-Za-z_][A-Za-z_0-9]*", symbolic)))
    out: List[Dict[str, str]] = []
    for name in names:
        if name == "n_layers":
            meaning = "number of layers represented by this formula term"
            value = " / ".join(dict.fromkeys(
                str(t["substituted"]).split(" × ", 1)[0] for t in terms))
        elif name in values:
            meaning = _DIM_MEANINGS.get(name, "resolved model dimension")
            value = " / ".join(sorted(values[name]))
        else:
            meaning = "formula quantity defined by the architecture"
            value = "see substituted expression"
        out.append({"name": name, "formula": meaning, "value": value})
    return out


def _parameter_formula(label: str, terms: List[Dict[str, Any]],
                       result: int) -> Dict[str, Any]:
    symbolic = " + ".join(t["symbolic"] for t in terms) or "0"
    substituted = " + ".join(t["substituted"] for t in terms) or "0"
    clean = [{k: v for k, v in term.items() if k != "_dims"}
             for term in terms]
    formula: Dict[str, Any] = {
        "label": label,
        "symbolic": symbolic,
        "substituted": substituted,
        "result": result,
    }
    if len(clean) > 1:
        formula["subterms"] = clean
    defs = _formula_defs(symbolic, terms)
    if defs:
        formula["defs"] = defs
    return formula


def _storage_formula(label: str, weights: Any) -> Dict[str, Any]:
    parameters = int(weights.parameters)
    metadata = int(weights.metadata)
    buffers = int(weights.buffers)
    terms = [
        {"label": dtype, "symbolic": "stored_dtype_bytes",
         "substituted": str(int(amount)), "result_bytes": int(amount),
         "defs": [{"name": "stored_dtype_bytes",
                   "formula": "bytes stored at this dtype",
                   "value": str(int(amount))}]}
        for dtype, amount in sorted(weights.by_dtype.items(),
                                    key=lambda item: -item[1]) if amount
    ]
    return {
        "label": label,
        "symbolic": "parameter_bytes + quantization_metadata_bytes + buffer_bytes",
        "substituted": f"{parameters} + {metadata} + {buffers}",
        "result_bytes": int(weights.resident),
        "defs": [
            {"name": "parameter_bytes", "formula": "learnable tensor storage",
             "value": str(parameters)},
            {"name": "quantization_metadata_bytes",
             "formula": "scale and zero-point storage", "value": str(metadata)},
            {"name": "buffer_bytes", "formula": "non-learnable tensor storage",
             "value": str(buffers)},
        ],
        "subterms": terms,
    }


def _append_block_formula_terms(row: "_Row", block: Any, layers: int,
                                analysis: Any, program: Any) -> None:
    dims = {
        name: int(value) for name, value in
        analysis.program.resolved.dim_values.items()
        if isinstance(value, (int, float)) and int(value) == value
    }
    dims.update(block.dims)
    total = _sum_expr(block.nodes)
    attention = _sum_expr(_role_roots(block.nodes, "attention"))
    ffn_total = _sum_expr(_role_roots(block.nodes, "ffn"))
    experts = _sum_expr(_routed_roots(block.nodes, program))
    other = _sum_expr(_other_roots(block.nodes))
    active = _active_expr(block.nodes, analysis, dims)

    expected = {
        "total": int(block.total_params),
        "attention": int(block.attention_params),
        "experts": int(block.routed_params),
        "other": int(block.total_params - block.attention_params - block.ffn_params),
        "active": int(block.total_active_params),
    }
    measured = {
        "total": total[2], "attention": attention[2],
        "experts": experts[2], "other": other[2], "active": active[2],
    }
    if measured != expected:
        raise ValueError(
            f"{block.template_name}: formula terms {measured} do not match "
            f"the parameter ledger {expected}")

    if experts[2]:
        ffn = (
            f"({ffn_total[0]}) - ({experts[0]})",
            f"({ffn_total[1]}) - ({experts[1]})",
            ffn_total[2] - experts[2],
        )
    else:
        ffn = ffn_total
    expected_ffn = int(block.ffn_params - block.routed_params)
    if ffn[2] != expected_ffn:
        raise ValueError(
            f"{block.template_name}: FFN formula is {ffn[2]}, expected "
            f"{expected_ffn}")

    per_kind = {
        "attention": attention,
        "ffn": ffn,
        "experts": experts,
        "other": other,
        "total": total,
        "active": active,
    }
    for kind, (symbolic, substituted, per_layer) in per_kind.items():
        if not per_layer:
            continue
        row.formula_terms.setdefault(kind, []).append(_parameter_term(
            f"{block.template_name} × {layers} layers", layers,
            symbolic, substituted, layers * per_layer, dims))


def _set_model_node_formula_terms(row: "_Row", node: Any,
                                  analysis: Any) -> None:
    dims = {
        name: int(value) for name, value in
        analysis.program.resolved.dim_values.items()
        if isinstance(value, (int, float)) and int(value) == value
    }
    dims.update({name: int(value) for name, value in
                 (node.op_dims or {}).items()
                 if isinstance(value, (int, float)) and int(value) == value})
    symbolic = node.params_symbolic or str(node.params)
    substituted = node.params_values or str(node.params)
    total = _parameter_term(node.name, 1, symbolic, substituted,
                            int(node.params), dims)
    row.formula_terms["total"] = [total]
    row.formula_terms["other"] = [total]
    active = _active_expr([node], analysis, dims)
    if active[2]:
        row.formula_terms["active"] = [_parameter_term(
            node.name, 1, active[0], active[1], active[2], dims)]


def _component_total_formula(label: str, symbol: str, rows: List["_Row"],
                             key: str, result: int, *,
                             result_bytes: bool = False) -> Dict[str, Any]:
    subterms = []
    for row in rows:
        formula = row.formulas().get(key)
        if formula and (formula.get("result") or formula.get("result_bytes")):
            subterms.append({**formula, "label": row.label})
    out: Dict[str, Any] = {
        "label": label,
        "symbolic": f"Σ {symbol}",
        "substituted": " + ".join(
            str(term.get("result_bytes") if result_bytes else term.get("result"))
            for term in subterms),
        "defs": [{
            "name": symbol,
            "formula": "value contributed by one model-composition row",
            "value": "see row subterms",
        }],
        "subterms": subterms,
    }
    out["result_bytes" if result_bytes else "result"] = result
    return out


@dataclass
class _Row:
    """One row's sums, folded over the invocations (or the node) it holds."""

    key: str
    kind: str
    label: str
    template: Optional[str] = None
    templates: List[str] = field(default_factory=list)
    stack: Optional[str] = None
    aux_kind: Optional[str] = None
    #: Whether the row is an encoder's (`Stack.is_encoder`): its weights
    #: are the encoder's own accounting, beside the language model's.
    encoder: bool = False
    layer_indices: List[int] = field(default_factory=list)
    layers: int = 0
    attention: int = 0
    ffn: int = 0
    experts: int = 0
    total: int = 0
    active: int = 0
    weights: Any = field(default_factory=capacity_mod.Weights)
    kv_per_token: Optional[float] = None
    kv_constant: Optional[float] = None
    #: `{dtype: source}` of the cached tensors behind the KV figures:
    #: "declared" by the spec's `cache.entries`, else the model's stated
    #: default ("native").
    kv_dtypes: Dict[str, str] = field(default_factory=dict)
    kv_growth_formula: Optional[Dict[str, Any]] = None
    kv_constant_formula: Optional[Dict[str, Any]] = None
    cache_blocks: List[int] = field(default_factory=list, repr=False)
    formula_terms: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict, repr=False)
    derived: List[Dict[str, Any]] = field(default_factory=list)
    #: The attention and FFN calls its layers make (`calls_of`): what a
    #: comparison lines up, since one component bound with different slots
    #: is a different call.
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def add_kv(self, growth: float, floor: float) -> None:
        self.kv_per_token = (self.kv_per_token or 0.0) + growth
        self.kv_constant = (self.kv_constant or 0.0) + floor

    def formulas(self) -> Dict[str, Dict[str, Any]]:
        results = {
            "attention": self.attention,
            "ffn": self.ffn - self.experts,
            "experts": self.experts,
            "other": self.total - self.attention - self.ffn,
            "total": self.total,
            "active": self.active,
        }
        labels = {
            "attention": "Attention parameters",
            "ffn": "FFN parameters excluding routed experts",
            "experts": "Routed-expert parameters",
            "other": "Other parameters",
            "total": "Total parameters",
            "active": "Active parameters per token",
        }
        formulas = {
            key: _parameter_formula(labels[key], self.formula_terms.get(key, []), value)
            for key, value in results.items() if value
        }
        formulas["storage"] = _storage_formula("Stored model state", self.weights)
        return formulas

    def payload(self) -> Dict[str, Any]:
        by_dtype = sorted(self.weights.by_dtype.items(), key=lambda kv: -kv[1])
        out: Dict[str, Any] = {
            "key": self.key,
            "kind": self.kind,
            "label": self.label,
            "template": self.template,
            # Every invocation's template: a declaration split into derived
            # groups drew each under its own name.
            "templates": list(dict.fromkeys(self.templates)),
            "stack": self.stack,
            "aux_kind": self.aux_kind,
            "layer_indices": sorted(self.layer_indices),
            "layers": self.layers,
            "params": {
                "attention": self.attention,
                # The feed-forward path without its routed experts: dense
                # FFN, router, always-on shared experts.
                "ffn": self.ffn - self.experts,
                "experts": self.experts,
                # Norms, residual mixers and model-level tables.
                "other": self.total - self.attention - self.ffn,
                "total": self.total,
            },
            "active_params": self.active,
            "storage_bytes": int(self.weights.resident),
            "storage_by_dtype": [{"dtype": name, "bytes": int(amount)}
                                 for name, amount in by_dtype if amount],
            "formulas": self.formulas(),
            # Backbone caches only (`CacheView`'s scope): None for an aux
            # stack, whose cache is not part of serving the model.
            "kv": (None if self.kv_per_token is None else {
                "per_token_bytes": self.kv_per_token,
                "constant_bytes": self.kv_constant,
                "growth_formula": self.kv_growth_formula,
                "constant_formula": self.kv_constant_formula,
                "dtypes": [{"dtype": d, "source": src}
                           for d, src in self.kv_dtypes.items()],
            }),
            "calls": list(self.calls),
        }
        if self.derived:
            out["histogram"] = histogram(self.derived)
            out["axis"] = self.derived[0]["axis"]
            # The `read: all` stream whose history length splits the layers.
            out["stream"] = self.derived[0]["stream"]
        return out


class ModelComposition:
    """The rows of one projected model (`ModelReport`), in reading order:
    embeddings, the backbone's blocks by first layer, the other model-level
    nodes, then each auxiliary stack's blocks."""

    def __init__(self, report: Any, weights: Any):
        """`weights` is `capacity.weights(report.analysis)` — the ledger the
        caller already priced the model's weight bytes from."""
        self.report = report
        analysis = report.analysis
        program = analysis.program
        self.weights = weights
        # The KV columns price each cached tensor at the dtype the model
        # runs it at: its `cache.entries` declaration, else the model's own
        # dtype. Read off the walk's resolved precision
        # (`Analysis.resolved_precision`), which is the one place that says
        # what the model runs at — the spec's `quant_context`, then the
        # checkpoint header, then what the frameworks serve that width at
        # (`dtypes.served_dtype`). Asking `ResolvedSpec.shipped_dtype`
        # again here is the second derivation path, and it answered F32 for
        # a gemma-2 cache the walk had already priced at BF16. A model that
        # states no precision never reaches an analysis at all
        # (`FormatPolicy.of` refuses), so the None branch below is history.
        precision = analysis.resolved_precision
        self.native = precision.model_dtype
        # Named as `precision_effective` names it, because it is the same
        # fact: `model_dtype_source` says where the CHECKPOINT's width came
        # from, and once a framework serves that width at another one the
        # checkpoint is no longer who decided it. Two signatures on one fact
        # is the bug this view already had once.
        stated = precision.model_dtype_source or "config"
        self.native_source = (stated if precision.served_as_stored
                              else "framework")
        self.cache = (cache_view_from_analysis(analysis)
                      if self.native is not None else None)
        block_of_root = {c.node.uid: c.block_index
                         for c in program.contributions}
        held = self.weights.by_root(program)
        block_weights: Dict[int, Any] = {}
        root_weights: Dict[int, Any] = {}
        for root, weights in held.items():
            index = block_of_root[root]
            if index is None:
                root_weights[root] = weights
            else:
                block_weights[index] = block_weights.get(
                    index, capacity_mod.Weights()).merged(weights)
        kv = (self.cache.growth_and_floor_by_block()
              if self.cache is not None else {})
        self._kv_dtypes: Dict[Optional[int], Dict[str, str]] = {}
        for entry, _count, index, _stack in (self.cache.entries if self.cache else ()):
            seen = self._kv_dtypes.setdefault(index, {})
            for tensor in entry.tensors:
                for dtype in tensor.resolved_format.dtypes:
                    seen[display_name(dtype)] = (
                        "declared" if tensor.declared is not None
                        else "native")
        self._encoders = frozenset(
            stack.name for stack in program.stacks if stack.is_encoder)
        # A stack that runs another's module again (Qwen3-VL's videos
        # through the image tower, `Stack.reruns`) holds no weights and no
        # cache: its blocks are the other stack's rows, not rows of their own.
        self._reruns = frozenset(
            stack.name for stack in program.stacks if stack.reruns is not None)
        self._head: List[_Row] = []
        self._tail: List[_Row] = []
        aux_kinds = {stack.name: stack.aux_kind for stack in program.stacks}
        aux_roots: List[_Row] = []
        for name, node in report.model_level_nodes.items():
            weights = root_weights.pop(node.ir_node.uid, None)
            if not node.params and weights is None:
                continue
            kind, label = root_kind(name, node)
            stack = program.stack_of(node.ir_node)
            row = _Row(key=f"root:{name}", kind=kind, label=label,
                       total=int(node.params),
                       active=int(node.active_params),
                       weights=weights or capacity_mod.Weights(),
                       # A model-level node of an aux stack (an MTP head's
                       # fc, a vision merger) is that stack's row.
                       stack=stack or None,
                       aux_kind=aux_kinds.get(stack),
                       encoder=stack in self._encoders)
            _set_model_node_formula_terms(row, node, analysis)
            if row.aux_kind is not None:
                aux_roots.append(row)
            else:
                (self._head if kind == "embedding" else self._tail).append(row)
        if root_weights:
            raise ValueError(
                f"model-level roots {sorted(root_weights)} store weights but "
                f"no model-level node reports them")
        if None in kv:
            raise ValueError(
                "a model-level node declares a KV cache; no row shows one")
        self._main, self._aux = self._block_rows(block_weights, kv)
        # Each aux stack's model-level rows after its blocks.
        stacks = list(dict.fromkeys(
            [r.stack for r in self._aux] + [r.stack for r in aux_roots]))
        self._aux = sorted(self._aux + aux_roots,
                           key=lambda r: stacks.index(r.stack))

    def _all_kv_dtypes(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for seen in self._kv_dtypes.values():
            for dtype, src in seen.items():
                out.setdefault(dtype, src)
        return out

    def _block_rows(self, block_weights: Dict[int, Any],
                    kv: Dict[Optional[int], Tuple[float, float]]
                    ) -> Tuple[List[_Row], List[_Row]]:
        report = self.report
        positions = report.layer_positions()
        rows: Dict[Tuple[Any, ...], _Row] = {}
        program = report.analysis.program
        for i, (template, count, block) in enumerate(report.block_results):
            stack = report.block_stack_names[i]
            if stack in self._reruns:
                if i in block_weights or kv.get(i, (0.0, 0.0)) != (0.0, 0.0):
                    raise ValueError(
                        f"stack {stack!r} runs another's module but block "
                        f"{template!r} holds weights or cache of its own")
                continue
            group = report.block_derived_groups[i]
            key = ((stack, "declaration", declaration_key(group))
                   if group is not None else (stack, i))
            row = rows.get(key)
            if row is None:
                row = rows[key] = _Row(
                    key=f"block:{stack or ''}:{len(rows)}", kind="block",
                    # The short name every page shows the block by.
                    label=program.block_label(program.blocks[i]),
                    template=template, stack=stack,
                    aux_kind=report.block_aux_kinds[i],
                    encoder=stack in self._encoders)
            row.templates.append(template)
            row.layer_indices.extend(positions[i])
            row.layers += int(count)
            row.attention += int(count) * int(block.attention_params)
            row.ffn += int(count) * int(block.ffn_params)
            row.experts += int(count) * int(block.routed_params)
            row.total += int(count) * int(block.total_params)
            row.active += int(count) * int(block.total_active_params)
            _append_block_formula_terms(
                row, block, int(count), report.analysis, program)
            if i in block_weights:
                row.weights = row.weights.merged(block_weights[i])
            if row.aux_kind is None and self.cache is not None:
                growth, floor = kv.get(i, (0.0, 0.0))
                row.add_kv(growth, floor)
                row.kv_dtypes.update(self._kv_dtypes.get(i, {}))
                row.cache_blocks.append(i)
            if group is not None:
                row.derived.append(group)
            row.calls.extend(calls_of(report.analysis,
                                      report.analysis.program.blocks[i]))
        if self.cache is not None:
            for row in rows.values():
                scoped = self.cache.for_blocks(frozenset(row.cache_blocks))
                if scoped.entries:
                    row.kv_growth_formula = scoped.growth_formula_explained
                    row.kv_constant_formula = scoped.constant_formula
        main = [r for r in rows.values() if r.aux_kind is None]
        aux = [r for r in rows.values() if r.aux_kind is not None]
        first = (lambda r: min(r.layer_indices) if r.layer_indices else 0)
        main.sort(key=first)
        stacks = list(dict.fromkeys(r.stack for r in aux))
        aux.sort(key=lambda r: (stacks.index(r.stack), first(r)))
        return main, aux

    @property
    def rows(self) -> List[_Row]:
        return self._head + self._main + self._tail + self._aux

    def payload(self) -> Dict[str, Any]:
        """The rows and the whole-model figures they sum to. The params and
        bytes totals are the dashboard's own (`total_params`,
        `total_active_params`, `weight_bytes`) and the page shows those; the
        KV totals are `CacheView.growth_and_floor`, which the dashboard does
        not carry."""
        rows = self.rows
        report = self.report
        sums = {
            "params": sum(r.total for r in rows),
            "active_params": sum(r.active for r in rows),
            "storage_bytes": sum(int(r.weights.resident) for r in rows),
        }
        totals = {
            "params": int(report.total_params),
            "active_params": int(report.total_active_params),
            "storage_bytes": int(self.weights.resident),
        }
        if sums != totals:
            raise ValueError(
                f"composition rows sum to {sums}, the model to {totals}")
        def by_dtype_of(held: List[_Row]) -> List[Dict[str, Any]]:
            by_dtype: Dict[str, int] = {}
            for row in held:
                for name, amount in row.weights.by_dtype.items():
                    if amount:
                        by_dtype[name] = by_dtype.get(name, 0) + int(amount)
            return [{"dtype": name, "bytes": amount}
                    for name, amount in sorted(
                        by_dtype.items(), key=lambda kv: -kv[1])]

        language = [r for r in rows if not r.encoder]
        growth, floor = (self.cache.growth_and_floor()
                         if self.cache is not None else (None, None))
        total_formulas = {
            "params": _component_total_formula(
                "Model parameters", "component_params", rows, "total",
                totals["params"]),
            "active": _component_total_formula(
                "Active parameters per token", "component_active_params",
                rows, "active", totals["active_params"]),
            "storage": _component_total_formula(
                "Stored model state", "component_storage_bytes", rows,
                "storage", totals["storage_bytes"], result_bytes=True),
        }
        return {
            "rows": [r.payload() for r in rows],
            "totals": {
                **totals,
                "formulas": total_formulas,
                # The rows' stored bytes by dtype, summed: the whole model's
                # weight ledger split by width.
                "storage_by_dtype": by_dtype_of(rows),
                # The same figures over the language model's rows alone —
                # every row but an encoder's (`Stack.is_encoder`), whose
                # weights are the encoder's own accounting.
                "language_model": {
                    "params": sum(r.total for r in language),
                    "active_params": sum(r.active for r in language),
                    "storage_bytes": sum(int(r.weights.resident)
                                         for r in language),
                    "storage_by_dtype": by_dtype_of(language),
                },
                "kv_per_token_bytes": growth,
                "kv_constant_bytes": floor,
                "kv_growth_formula": (self.cache.growth_formula_explained
                                      if self.cache is not None else None),
                "kv_constant_formula": (self.cache.constant_formula
                                        if self.cache is not None else None),
                # Every dtype the cached tensors behind those figures are
                # priced at, with where it is stated: a cache that declares
                # FP8 entries beside native ones is both.
                "kv_dtypes": [{"dtype": d, "source": src}
                              for d, src in self._all_kv_dtypes().items()],
            },
            # The dtype an undeclared cached tensor is priced at, and where
            # the model states it ("quant_context" / "config" /
            # "config.text_config"); None when it
            # states none and the KV columns are empty.
            "kv_native_dtype": self.native,
            "kv_native_source": self.native_source,
        }
