"""The dashboard formula payload, off one walk over the workload's counts.

Each printed formula is the walk's identity halves (`expr_of`: the model's
names), its `with_values` the same terms each bound by its node's frame, and
its result the walk's law halves read at the request's point. See the
package docstring."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

import sympy as sp
from sympy.printing.str import StrPrinter

from .result import _ATTN_ROLES, _FFN_ROLES, _WRAPPER_ROLES
from .. import dim as dim_mod
from ..cache_view import cache_view_from_analysis
from ..analysis import Point
from ..program import Program
from ..fusion import FusedProgram
from ..transforms import Deployment


def _weights_formula(walk: Any, numeric_analysis: Any,
                     bound: Any) -> Tuple[Any, Any, int, str]:
    """Sum each stored tensor's `storage_bytes × copies` in dim names (the
    walk's identity halves), and the same terms each bound by its node's
    frame (`bound`).

    The rows are the checkpoint file's (`checkpoint_tensors()`: what the
    model stores plus the redundant copies of tied weights), the rows
    `checkpoint_param_bytes()` evaluates, so the product is that figure
    before it collapses to a byte count. Like terms still collect (`16`
    layers of `2·d_model` become `32·d_model`); the dim names stay.
    """
    terms = [(row.uid, dim_mod.expr_of(row.storage_bytes)
              * dim_mod.expr_of(row.count))
             for row in walk.checkpoint_tensors()
             if not (row.is_metadata or row.is_buffer)]
    return (
        sum((term for _uid, term in terms), sp.Integer(0)),
        bound(terms, numeric_analysis.params_bindings),
        numeric_analysis.checkpoint_param_bytes(),
        "On-disk weight bytes (Σ storage × copies)",
    )


def _renderable(extent: Any) -> Any:
    """The value half of a `Dim` — at a point, its number — which the
    numbers beside the formulas sum. Arithmetic on a `Dim` through SymPy
    takes the identity half (`Dim._sympy_`), so a sum of values reads each
    one's value first."""
    return dim_mod.value_of(extent)


def _display_expr(expr: Any) -> str:
    """SymPy → display text, `^` for powers; `*` stays so a user can paste
    the formula back into Python unchanged."""
    return str(expr).replace("**", "^")


class _ReadablePrinter(StrPrinter):
    """SymPy → text for reading rather than pasting: ⌊x⌋ / ⌈x⌉ for floor
    and ceiling, grouped digits, `·` for products and `^` for powers."""

    def _print_floor(self, expr: Any) -> str:
        return f"⌊{self._print(expr.args[0])}⌋"

    def _print_ceiling(self, expr: Any) -> str:
        return f"⌈{self._print(expr.args[0])}⌉"

    def _print_Integer(self, expr: Any) -> str:
        return f"{int(expr):,}"


def _readable_expr(expr: Any) -> str:
    return (_ReadablePrinter().doprint(expr)
            .replace("**", "^").replace("*", "·"))


def _render_formula(expr: Any, with_values: Any,
                    description: str, *, result: Any) -> Dict[str, Any]:
    """The dashboard tooltip dict — five keys, JSON-clean by construction
    (the payload contract test's whole subject). `result` is the figure
    the walk read at the request's point gives, the one the rest of the
    dashboard shows; `with_values` the expression's terms each bound where
    it was walked.
    """
    if not isinstance(expr, sp.Basic):
        text = str(int(expr or 0))
        return {"symbolic": text, "with_values": text,
                "result": f"{int(result):,}", "latex": text,
                "description": description}
    return {
        "symbolic": _display_expr(expr),
        "with_values": _display_expr(with_values),
        "result": f"{int(result):,}",
        "latex": sp.latex(expr),
        "description": description,
    }


def _leaves_of(node: Any) -> List[Any]:
    """`node`'s leaves, itself for a leaf."""
    if not node.children:
        return [node]
    return [leaf for child in node.children for leaf in _leaves_of(child)]


def display_formulas(resolved: Any, *, batch_size: int = 1,
                            seq_len: int = 1) -> Dict[str, Any]:
    """The dashboard's symbolic compute + memory payload, off one walk.

    One walk over the workload's counts: each node's cost has an identity
    half — its formula, in the model's and the step's names — and the same
    walk read at the request's point (`Analysis.at`) is its number. The
    formulas sum the identity halves and the figures the numbers, both by
    the rule an estimate's breakdown is summed by (the outermost node of a
    role holds its subtree), so a figure is the breakdown's
    (`test_dashboard_is_the_breakdown`) and a tooltip its own formula's.

    The backbone's alone: no aux stack (an MTP draft) is counted, as an
    estimate's breakdown counts none (`Program.included_stack_names`).
    """
    program = Program.of(resolved)
    deployment = Deployment(program).at(
        batch=batch_size, input_tokens=seq_len, logit_positions=1)
    # The one walk, over the workload's counts, and the same walk read at
    # the request's point: each node's cost is its identity half (the
    # formula, in the model's and the step's names) beside its value there
    # (the number).
    workload, point = deployment.workload.symbolic()
    walk = replace(deployment, workload=workload).analyze()
    numeric_analysis = walk.at(Point(point), structure=(FusedProgram,))
    substitutions: Dict[Any, Any] = {
        sp.Symbol(name): dim_mod.exact(value)
        for name, value in deployment.binding().flat().items()
        if not dim_mod.varies(dim_mod.value_of(value))
    }
    backbone = program.included_stack_names()

    def leaves(roles: Any) -> List[Any]:
        """Every backbone leaf counted under `roles` (None: every role), by
        the rule an estimate's breakdown is summed by: the outermost node of
        one of `roles` holds its subtree, a node of another role is skipped
        with its own, a wrapper is looked through. A leaf's term is its
        instances (`Analysis.instances`: every layer and `repeat:` that
        copies it) times one instance's cost — the breakdown's row."""
        out: List[Any] = []
        seen: set = set()

        def visit(node: Any, held: bool) -> None:
            if node.uid in seen:
                return
            seen.add(node.uid)
            if roles is None or node.role in roles:
                held = True
            elif (not held and node.role is not None
                  and node.role not in _WRAPPER_ROLES):
                return
            if not node.children:
                if held and walk.cost.values.get(node.uid) is not None:
                    out.append(node)
                return
            for child in node.children:
                visit(child, held)
        for p in walk.program.contributions:
            if p.stack in backbone:
                visit(p.node, False)
        return out

    def bound(terms: List[Tuple[int, Any]], records: Any) -> Any:
        """The terms, each bound where it was walked: its node's record of
        the names its evaluation read (`records`, identity → value: a
        router's `bias`, a linear's `out`) and the names its frame binds (a
        pool's `compress_ratio`), beside the model's and the step's,
        products kept (`dim.bind_keep_products`)."""
        out = []
        for uid, term in terms:
            record = records.values.get(uid) or {}
            frame = numeric_analysis.node_dims.values.get(uid) or {}
            out.append(dim_mod.bind_keep_products(
                term,
                {**substitutions,
                 **{sp.Symbol(name): dim_mod.exact(value)
                    for name, value in frame.items()
                    if not dim_mod.varies(dim_mod.value_of(value))},
                 **{key: value for key, value in record.items()
                    if isinstance(key, sp.Symbol)}},
                {key: value for key, value in record.items()
                 if not isinstance(key, sp.Symbol)}))
        # Terms that print the same once bound are one term `n*(term)`, in
        # the order they first came: a merge of text, no algebra.
        merged: Dict[str, List[Any]] = {}
        for term in out:
            merged.setdefault(_display_expr(term), []).append(term)
        terms_out = [group[0] if len(group) == 1
                     else sp.Mul(sp.Integer(len(group)), group[0], evaluate=False)
                     for group in merged.values()]
        return (sp.Add(*terms_out, evaluate=False) if terms_out
                else sp.Integer(0))

    def flops(nodes: List[Any]) -> Tuple[Any, Any, Any]:
        """The identity halves' sum, the same terms bound, and the number."""
        # A leaf that computes nothing (a view, a split) adds no term.
        terms = [(node.uid, term) for node, term in (
            (node, dim_mod.expr_of(walk.instances.values[node.uid])
             * dim_mod.expr_of(walk.cost.values[node.uid].flops))
            for node in nodes) if not dim_mod.same_law(term, 0)]
        number = sum((_renderable(numeric_analysis.instances.values[node.uid])
                      * _renderable(numeric_analysis.cost.values[node.uid].flops)
                      for node in nodes), sp.Integer(0))
        return (sum((term for _uid, term in terms), sp.Integer(0)),
                bound(terms, numeric_analysis.flops_bindings), number)

    attention = flops(leaves(_ATTN_ROLES))
    ffn = flops(leaves(_FFN_ROLES))
    total = flops(leaves(None))
    lm_head = flops([leaf for node in walk.program.nodes()
                     if walk.program.is_lm_head(node)
                     for leaf in _leaves_of(node)
                     if walk.cost.values.get(leaf.uid) is not None])

    output: Dict[str, Any] = {
        "attention": _render_formula(
            attention[0], attention[1],
            "Forward attention FLOPs across all layers", result=attention[2]),
        "ffn": _render_formula(
            ffn[0], ffn[1], "Forward FFN FLOPs across all layers",
            result=ffn[2]),
        "lm_head": _render_formula(
            lm_head[0], lm_head[1], "LM head FLOPs", result=lm_head[2]),
        "total": _render_formula(
            total[0], total[1], "Total forward FLOPs", result=total[2]),
    }

    # Memory: weights as params × dtype when the checkpoint is one
    # width, a sum of those terms when mixed; the cache under each
    # tensor's storage law at this context.
    reader = cache_view_from_analysis(numeric_analysis)
    weights_expr, weights_bound, weights_num, weights_desc = _weights_formula(
        walk, numeric_analysis, bound)
    kv_num = reader.storage_bytes_at(seq_len) * batch_size
    kv_text = f"B × Σ_layers Σ_kv_cached law({seq_len}) × bpe"
    output["memory"] = {
        "weights": _render_formula(
            weights_expr, weights_bound, weights_desc, result=weights_num),
        "kv_cache": {
            "symbolic": kv_text,
            "with_values": f"{int(kv_num):,}",
            "result": f"{int(kv_num):,}",
            "latex": kv_text,
            "description": ("Total KV cache bytes for B requests at this "
                            "context — each cached tensor billed under its "
                            "own storage law"),
        },
    }

    groups = []
    for template in program.stack_blocks():
        call = template.attention_call()
        groups.append({
            "count": template.count,
            # Stored layers it holds: a looped stack's later pass and its
            # gate hold none (`BlockMeta.stored_count`).
            "stored_count": template.stored_count,
            "attn_type": (
                (call.attn_variant or call.kv_label) if call is not None else None),
            "ffn_type": template.ffn_type(),
        })
    output["hetero_config"] = {
        "total_layers": sum(item["stored_count"] for item in groups),
        "layer_groups": groups,
    }
    return output


def encoder_item_formulas(resolved: Any) -> Dict[str, Dict[str, Any]]:
    """Per encoder stack, the FLOPs one media item costs in its encode
    phase, as an expression in the item's own axes (an image's processed
    height and width, an audio clip's frames).

    The encode walk at one request carrying one item of each of the
    encoder's inputs, the item's axes declared counts of their own:
    `symbolic` is its FLOPs' identity half (every dim a name); `per_item`
    its value half, the model's dims their values and the item's axes
    free. No size is bound to those axes — the number for a given
    item is the encode phase's, at the media a request states. An input
    the engine does not walk (`unmodeled`) is stated with no item; the
    items of each item of another input are a variable."""
    program = Program.of(resolved)
    out: Dict[str, Dict[str, Any]] = {}
    for stack in program.stacks:
        if not stack.is_encoder:
            continue
        own = {name: entry for name, entry in program.model_inputs.items()
               if entry.stack == stack.name
               or (entry.unmodeled is not None and entry.per_item_of
                   and program.model_inputs[entry.per_item_of].stack
                   == stack.name)}
        media: Dict[str, Dict[str, int]] = {}
        variables = []
        for name, entry in own.items():
            if entry.unmodeled is not None:
                media[name] = {entry.axes[0]: 0}
                continue
            # One item; each free axis at a legal size, declared a count of
            # its own below (`with_dims`), so the walk's laws read it. The
            # items of each item of another input (Step-3.7's crops) are a
            # count of their own, a variable too.
            media[name] = {entry.axes[0]: 1}
            if entry.per_item_of is not None:
                media[name][entry.axes[0]] = 2
                variables.append(entry.axes[0])
            for axis in entry.axes[1:]:
                if axis in entry.fixed:
                    continue
                media[name][axis] = 2 * int(dim_mod.value_of(entry.least[axis]))
                variables.append(axis)
        # The encode walk over the item's axes as counts of their own: its
        # FLOPs' value half is the law in them, every model dim a number
        # (`per_item`); its identity half names the dims (`symbolic`).
        encode = Deployment(program).at(batch=1, media=media).during(
            "encode", stack=stack.name).with_dims(**{
                axis: sp.Symbol(axis, integer=True, positive=True)
                for axis in variables})
        flops = encode.analyze().by_stack().get(stack.name).flops
        expr = dim_mod.expr_of(flops)
        per_item = sp.expand(dim_mod.value_of(flops))
        out[stack.name] = {
            "aux_kind": stack.aux_kind,
            "inputs": sorted(name for name, entry in own.items()
                             if entry.unmodeled is None),
            "variables": variables,
            "symbolic": _display_expr(expr),
            "per_item": _display_expr(per_item),
            # The same expression to read (`_ReadablePrinter`).
            "per_item_readable": _readable_expr(per_item),
            "latex": sp.latex(per_item),
            "description": "Encoder FLOPs per media item (encode phase)",
        }
    return out
