"""Read-only tree reports projected from ``Analysis`` for presentation."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

from ... import roles
from ..access import _int
from ..binding import Definition
from ..dim import algebra, as_extent, as_number, bound_or_folded, expr_of, value_of
from ..flops import merge_bindings
from ..cache import prefill_reach
from ..program import BlockStage, BlockTemplate, OpStage, Program
from ..transforms import CallSite, Deployment, isolate_resolved
from ..values import Cost, Shape

#: Which roles count as attention / feed-forward work, and which are mere
#: containers to recurse through. Projections of `roles.yaml` rather than
#: literals: a new attention variant is classified where it is declared, and
#: the classification cannot disagree with the one `inference/sharding.py`
#: reads, because there is only one.
_ATTN_ROLES = roles.of_kind("attention")
_FFN_ROLES = roles.of_kind("ffn")
_MEMORY_ROLES = roles.of_kind("memory")
_RESIDUAL_ROLES = roles.of_kind("residual")
_WRAPPER_ROLES = roles.of_kind("wrapper")


def _exact_num(value: Any):
    """The EXACT value of a cost, as a number that can still be added.

    A dispatch-zone leaf's per-copy flops is a rational (the exact `M`
    average), and truncating it would make the projected tree inconsistent
    with itself: the bank parent's integral total no longer equals the
    multiplier times the truncated children, which the pipeline's coverage
    check rightly refuses. `Fraction` is the stdlib's exact carrier and the
    reports sum these — `total_flops` over a tree of them — so the value has
    to stay arithmetic. It is NOT JSON-native: `jsonable_encoder` refuses a
    `Fraction` exactly as it refuses a `Rational`, so whoever puts one on a
    response passes it through `as_number` first.
    """
    from fractions import Fraction

    v = value_of(value)
    if not v:
        return 0
    if v == int(v):
        return int(v)
    import sympy as sp
    r = sp.Rational(v)
    return Fraction(int(r.p), int(r.q))


def _dims(value: Optional[Tuple[Any, ...]]) -> Optional[List[int]]:
    if value is None:
        return None
    return [as_extent(axis, rounding="floor") for axis in value]


def _dims_exact(value: Optional[Tuple[Any, ...]]) -> Optional[List[Any]]:
    if value is None:
        return None
    return [as_number(axis) for axis in value]


@dataclass
class NodeReport:
    name: str
    op: str
    flops: int
    params: int
    active_params: int = 0
    shape_in: Any = None
    shape_in_exact: Any = None
    shape_out: Optional[List[int]] = None
    shape_outs: Dict[str, List[int]] = field(default_factory=dict)
    is_component: bool = False
    children: Optional[List["NodeReport"]] = None
    #: The leaf's op declaration (`SpecIRNode.declaration`), carried so a
    #: consumer of the report asks it, not the registry.
    declaration: Any = None
    flops_expr: Any = None
    params_expr: Any = None
    flops_symbolic: Optional[str] = None
    #: `flops_expr` in the names of the block it is drawn in: each name a
    #: leaf's op rebinds by a definition (`Analysis.node_rebinds`: a window's
    #: `s_pairs`) read as that definition's law, and a component's the sum of
    #: its children's, as its cost is. `flops_symbolic` keeps the leaf's own
    #: name, beside the values it binds there (`flops_values`); a law summed
    #: over several nodes (an L1 box) reads this one, where one name means
    #: one thing.
    flops_law: Any = None
    params_symbolic: Optional[str] = None
    flops_values: Optional[str] = None
    params_values: Optional[str] = None
    shape_in_symbolic: Optional[str] = None
    shape_out_symbolic: Optional[str] = None
    shape_in_values: Optional[str] = None
    shape_out_values: Optional[str] = None
    shape_in_axes: Optional[List[List[str]]] = None
    shape_out_axes: Optional[List[str]] = None
    shape_outs_axes: Dict[str, List[str]] = field(default_factory=dict)
    op_dims: Optional[Dict[str, Any]] = None
    repeat_display: Optional[Tuple[str, str]] = None
    active_display: Optional[Tuple[str, str]] = None
    kv_read_positions_expr: Any = None
    kv_storage_positions_expr: Any = None
    kv_block_anchor: bool = True
    axis: Optional[int] = None
    axes: Optional[List[int]] = None
    quant_metadata_params: int = 0
    role: Optional[str] = None
    ir_node: Any = None
    #: `Program.is_lm_head`, materialized: the model's LM head, not an
    #: auxiliary stack's own output head.
    is_lm_head: bool = False
    #: Checkpoint bytes of the stored tensors under this node, for one copy
    #: of the block it sits in — every tensor at its own storage width, the
    #: same rows `Analysis.checkpoint_param_bytes` sums for the whole model.
    storage_bytes: int = 0

    @property
    def boundary_role(self) -> Optional[str]:
        return (self.ir_node.boundary_role
                if self.ir_node is not None else None)

    @property
    def spec_id(self) -> str:
        return self.ir_node.display_path if self.ir_node is not None else ""

    @property
    def aggregation_id(self) -> str:
        return self.ir_node.aggregation_id if self.ir_node is not None else ""


@dataclass
class BlockReport:
    template_name: str
    nodes: List[NodeReport]
    total_flops: int
    total_params: int
    total_active_params: int = 0
    attention_flops: int = 0
    attention_params: int = 0
    ffn_flops: int = 0
    ffn_params: int = 0
    #: A lookup memory's share (`roles.yaml` kind `memory`: Engram, PLE).
    memory_flops: int = 0
    memory_params: int = 0
    #: The residual stream's own mixing (`roles.yaml` kind `residual`:
    #: Hyper-Connections).
    residual_flops: int = 0
    residual_params: int = 0
    #: The part of `ffn_params` that runs on dispatched tokens — the routed
    #: expert banks (`Program.routed_uids`); the router and any always-on
    #: shared expert stay in the rest of `ffn_params`.
    routed_params: int = 0
    quant_metadata_params: int = 0
    #: Per-copy checkpoint bytes of the block (sum of its roots').
    total_storage_bytes: int = 0
    #: The whole-number dims of the frame the block's shown nodes ran in
    #: (`Analysis.block_frame`): its assignment's `dims:` applied, not the
    #: model's table. A dim the walk left symbolic or fractional has none.
    dims: Dict[str, int] = field(default_factory=dict)


@dataclass
class ModelReport:
    block_results: List[Tuple[str, int, BlockReport]]
    total_params: int = 0
    #: Parameters the model's routing activates, over every stack it holds.
    total_active_params: int = 0
    #: Parameters one decoded token touches: every layer of the language
    #: model it passes, an encoder's none (it runs in its encode phase).
    total_decode_active_params: int = 0
    #: Parameters each prompt token touches in prefill, for a long prompt:
    #: only the layers that must run on every position
    #: (`cache.prefill_reach`).
    total_prefill_active_params: int = 0
    #: The prompt length past which `total_prefill_active_params` is what
    #: each further prompt token touches (`cache.PrefillReach.full_from`).
    prefill_full_from: int = 0
    #: The main model alone (`Analysis.main_total`): no MTP head, no
    #: encoder. The headline a model card states and every page leads with.
    main_params: int = 0
    #: Parameters one decoded token touches in the main model.
    main_active_params: int = 0
    #: `total_prefill_active_params` over the main model alone.
    main_prefill_active_params: int = 0
    total_flops: int = 0
    quant_metadata_params: int = 0
    #: Parameters held in lookup tables other than the model's input
    #: embedding (`Program.lookups`): stored whole, read a few rows a token.
    lookup_table_params: int = 0
    #: `lookup_table_params` on the main stack alone, the figure that sits
    #: beside `main_params` (an MTP head's own table is aux, not main).
    main_lookup_table_params: int = 0
    aux_params: Dict[str, int] = field(default_factory=dict)
    aux_active_params: Dict[str, int] = field(default_factory=dict)
    aux_prefill_active_params: Dict[str, int] = field(default_factory=dict)
    aux_flops: Dict[str, int] = field(default_factory=dict)
    block_stack_names: List[Optional[str]] = field(default_factory=list)
    block_aux_kinds: List[Optional[str]] = field(default_factory=list)
    #: Per block, its `BlockMeta.derived_group` record (None for a block that
    #: is its whole declaration), so a report can show derived groups under
    #: their declaration.
    block_derived_groups: List[Optional[Dict[str, Any]]] = field(
        default_factory=list)
    top_level_node_params: Dict[str, int] = field(default_factory=dict)
    model_level_nodes: Dict[str, NodeReport] = field(default_factory=dict)
    #: Source of every displayed value. Reports never become an alternate
    #: evaluation tree and production inference does not consume them.
    analysis: Any = None

    def layer_positions(self) -> List[List[int]]:
        """Per block invocation, the layers it covers in its own stack, as
        the checkpoint numbers them (`BlockMeta.layer_numbers`): the
        resolver's selection where the spec made one, else every layer of
        its stack; in a looped stack, the stored layers (`layer_passes`
        says which pass)."""
        return [list(template.meta.layer_numbers)
                for template in self.analysis.program.blocks]

    def layer_passes(self) -> List[int]:
        """Per block invocation, the pass of a looped stack it runs in (1
        outside a loop), and 0 for a loop's gate, which is no layer."""
        return [0 if template.meta.loop_gate else template.meta.pass_
                for template in self.analysis.program.blocks]

    def _lm_head_node(self) -> Optional[NodeReport]:
        for node in self.model_level_nodes.values():
            if node.is_lm_head:
                return node
        return None

    @property
    def embedding_params(self) -> int:
        """The model's input embedding: the model-level node whose
        `boundary_role` is `input_embedding` — the same node
        `top_level_node_params` and `_lookup_table_params` set aside."""
        return sum(
            node.params for node in self.model_level_nodes.values()
            if node.boundary_role == "input_embedding"
        )

    @property
    def lm_head_params(self) -> int:
        node = self._lm_head_node()
        return node.params if node else 0

    @property
    def lm_head_flops(self) -> int:
        node = self._lm_head_node()
        return node.flops if node else 0


def _shape_in(shape: Optional[Shape]) -> Any:
    if shape is None or not shape.ins:
        return None
    inputs = [_dims(dims) for dims in shape.ins]
    return inputs[0] if len(inputs) == 1 else inputs


def _shape_in_exact(shape: Optional[Shape]) -> Any:
    if shape is None or not shape.ins:
        return None
    inputs = [_dims_exact(dims) for dims in shape.ins]
    return inputs[0] if len(inputs) == 1 else inputs


def _axis_exprs(dims: Optional[Tuple[Any, ...]]) -> Optional[List[str]]:
    if dims is None:
        return None
    return [str(expr_of(axis)) for axis in dims]


def _product(dims: Optional[Tuple[Any, ...]], *, symbolic: bool) -> Any:
    if not dims:
        return None
    result: Any = 1
    for axis in dims:
        result *= expr_of(axis) if symbolic else value_of(axis)
    return result


def _shape_product(shape: Optional[Shape], *, inputs: bool,
                   symbolic: bool) -> Optional[str]:
    if shape is None:
        return None
    tensors = shape.ins if inputs else tuple(
        dims for dims in (shape.out, *tuple(shape.ports.values())) if dims
    )
    values = [_product(dims, symbolic=symbolic) for dims in tensors]
    values = [value for value in values if value is not None]
    if not values:
        return None
    total = values[0]
    for value in values[1:]:
        total += value
    return str(total)


def _stored_bytes_by_uid(analysis: Any) -> Dict[int, int]:
    """`{uid → checkpoint bytes × declared count}` over the rows
    `checkpoint_param_bytes` counts: metadata siblings and buffers out."""
    out: Dict[int, int] = {}
    for row in analysis.tensors():
        if row.is_metadata or row.is_buffer:
            continue
        out[row.uid] = out.get(row.uid, 0) + (
            int(row.storage_bytes) * int(value_of(row.count)))
    return out


def _subtree_record(layer: Any, node: Any) -> Any:
    """The merged substitution record of a subtree's leaves.

    A component's cost is the sum of its children's — it never went through
    `flops_from_shape` itself, so it has no record of its own. The merged
    leaf records share the walk's vocabulary, so the sum's expression binds
    with them (and `bind_record`'s exact-reproduction gate rejects the cases
    where it doesn't).
    """
    records = []

    def visit(n: Any) -> None:
        record = layer.values.get(n.uid)
        if record:
            records.append(record)
        for child in n.children or ():
            visit(child)

    visit(node)
    return merge_bindings(records) if records else None


def _project_node(
    analysis: Any,
    node: Any,
    *,
    stored_bytes: Dict[int, int],
    per: int,
) -> NodeReport:
    """Project one program subtree into a presentation report. `per` is how
    many copies of the enclosing block the tree holds; a node's rows carry
    that count and the report is per copy, like its `params`."""
    cost = analysis.cost.values.get(node.uid, Cost())
    shape = analysis.shapes.values.get(node.uid)
    children = [
        _project_node(analysis, child, stored_bytes=stored_bytes, per=per)
        for child in node.children
    ]
    params = _int(cost.params)
    flops_record = (analysis.flops_bindings.values.get(node.uid)
                    or _subtree_record(analysis.flops_bindings, node))
    params_record = (analysis.params_bindings.values.get(node.uid)
                     or _subtree_record(analysis.params_bindings, node))
    own_bytes = stored_bytes.get(node.uid, 0)
    if own_bytes % per:
        raise ValueError(
            f"{node.display_path}: {own_bytes} stored bytes are not "
            f"{per} copies of anything")
    storage_bytes = own_bytes // per + sum(c.storage_bytes for c in children)
    kv_positions = analysis.cache_positions.values.get(node.uid)
    return NodeReport(
        name=node.name,
        op=node.op,
        declaration=node.declaration,
        is_lm_head=analysis.program.is_lm_head(node),
        flops=_exact_num(cost.flops),
        params=params,
        active_params=_int(cost.active_params),
        shape_in=_shape_in(shape),
        shape_in_exact=_shape_in_exact(shape),
        shape_out=_dims(shape.out) if shape is not None else None,
        shape_outs={name: _dims(dims) or []
                    for name, dims in (shape.ports.items() if shape else ())},
        is_component=bool(node.children),
        children=children or None,
        flops_expr=expr_of(cost.flops),
        flops_law=_flops_law(analysis, node, cost, children),
        params_expr=expr_of(cost.params),
        params_symbolic=str(expr_of(cost.params)),
        params_values=bound_or_folded(cost.params, params_record, params),
        flops_symbolic=str(expr_of(cost.flops)),
        flops_values=bound_or_folded(
            cost.flops, flops_record, _int(cost.flops)),
        shape_in_symbolic=_shape_product(shape, inputs=True, symbolic=True),
        shape_out_symbolic=_shape_product(shape, inputs=False, symbolic=True),
        shape_in_values=_shape_product(shape, inputs=True, symbolic=False),
        shape_out_values=_shape_product(shape, inputs=False, symbolic=False),
        shape_in_axes=(
            [_axis_exprs(dims) or [] for dims in shape.ins]
            if shape is not None and shape.ins else None
        ),
        shape_out_axes=_axis_exprs(shape.out) if shape is not None else None,
        shape_outs_axes={
            name: _axis_exprs(dims) or []
            for name, dims in (shape.ports.items() if shape else ())
        },
        op_dims={
            name: value_of(value)
            for name, value in (analysis.node_dims.values.get(node.uid, {}) or {}).items()
        } or None,
        kv_read_positions_expr=(kv_positions.read
                                if kv_positions is not None else None),
        kv_storage_positions_expr=(kv_positions.stored
                                   if kv_positions is not None else None),
        kv_block_anchor=(kv_positions.is_block_anchor
                         if kv_positions is not None else True),
        # Every node a report lays out answers for its own axes: a tree node
        # carries the authored `axis:` / `axes:`, and a fused launch declares
        # them None (`launch.FusedNode`) because the axes are its origins'.
        axis=node.axis,
        axes=list(node.axes) if node.axes is not None else None,
        quant_metadata_params=_int(cost.quant_metadata_params),
        role=node.role,
        ir_node=node,
        storage_bytes=storage_bytes,
    )


def _flops_law(analysis: Any, node: Any, cost: Cost,
               children: List[NodeReport]) -> Any:
    """`NodeReport.flops_law`: a component's cost is its children's
    (`Cost.combine`), so its law is the sum of theirs; a leaf's is its own
    FLOPs with each name its op rebinds by a definition read as the law that
    defines it."""
    if children:
        return sum((child.flops_law for child in children), sp.Integer(0))
    expr = algebra(cost.flops)
    defined = {name: value.law for name, value in
               (analysis.node_rebinds.values.get(node.uid) or {}).items()
               if isinstance(value, Definition)}
    return expr.xreplace({symbol: defined[symbol.name]
                          for symbol in expr.free_symbols
                          if symbol.name in defined})


def _classified(nodes: List[NodeReport], roles: frozenset[str],
                field_name: str) -> int:
    def visit(node: NodeReport) -> int:
        if node.role in roles:
            return int(getattr(node, field_name))
        if node.role is not None and node.role not in _WRAPPER_ROLES:
            return 0
        return sum(visit(child) for child in (node.children or ()))
    return sum(visit(node) for node in nodes)


def _routed_params(nodes: List[NodeReport], program: Any) -> int:
    """Params of the outermost nodes that run on dispatched tokens. Taken at
    the outermost one because a node's params carry its `repeat:` and its
    children's do not — an expert bank holds `n_experts` copies of the one
    expert its children describe."""
    def visit(node: NodeReport) -> int:
        if node.ir_node is not None and program.is_routed(node.ir_node.uid):
            return int(node.params)
        return sum(visit(child) for child in (node.children or ()))
    return sum(visit(node) for node in nodes)


def _block(analysis: Any, template: BlockTemplate,
           stored_bytes: Dict[int, int]) -> BlockReport:
    nodes = [_project_node(analysis, node, stored_bytes=stored_bytes,
                           per=template.count)
             for node in template.shown]
    return BlockReport(
        template_name=template.name,
        nodes=nodes,
        total_flops=sum(node.flops for node in nodes),
        total_params=sum(node.params for node in nodes),
        total_storage_bytes=sum(node.storage_bytes for node in nodes),
        total_active_params=sum(node.active_params for node in nodes),
        attention_flops=_classified(nodes, _ATTN_ROLES, "flops"),
        attention_params=_classified(nodes, _ATTN_ROLES, "params"),
        ffn_flops=_classified(nodes, _FFN_ROLES, "flops"),
        ffn_params=_classified(nodes, _FFN_ROLES, "params"),
        memory_flops=_classified(nodes, _MEMORY_ROLES, "flops"),
        memory_params=_classified(nodes, _MEMORY_ROLES, "params"),
        residual_flops=_classified(nodes, _RESIDUAL_ROLES, "flops"),
        residual_params=_classified(nodes, _RESIDUAL_ROLES, "params"),
        routed_params=_routed_params(nodes, analysis.program),
        quant_metadata_params=sum(node.quant_metadata_params for node in nodes),
        dims={name: number for name, value in
              analysis.block_frame(template).items()
              if type(number := as_number(value)) is int},
    )


def _lookup_table_params(analysis: Any, stack: Optional[str] = None) -> int:
    """Parameters in tables a leaf gathers rows from (`Program.lookups`),
    the model's input embedding aside — the one table every report counts
    as the model's embedding. Each table once per copy the model holds;
    `stack` keeps the tables of one stack ("" the main one)."""
    program = analysis.program
    tables = {site.table for site in program.lookups.values()}
    total: Any = 0
    for part in program.contributions:
        pending = [part.node]
        while pending:
            node = pending.pop()
            pending.extend(node.children or ())
            if (node.uid in tables and node.boundary_role != "input_embedding"
                    and (stack is None or program.stack_of(node) == stack)):
                cost = analysis.cost.values.get(node.uid)
                if cost is not None:
                    total = total + cost.params * part.count
    return _int(total)


def _whole_active(analysis: Any,
                  counts: Optional[Dict[int, int]] = None) -> int:
    """Active parameters over a selection of the model's accounting roots
    (`Program.multiplicity`), as whole counts the report's rows add up to.

    The walk's active count is exact and need not be whole: a lookup touches
    the rows a token is expected to read (DeepSeek-V4.1's Engram, ~6143.4
    parameters of one table per token). A report shows whole counts, so it
    rounds — once per accounting root, the figure that root's row shows
    (`NodeReport.active_params`) — and every active total here is those
    figures times the copies the selection keeps, so a total is the sum of
    the rows it reports (rounding the exact grand total once would not be).
    """
    program = analysis.program
    kept = program.multiplicity(counts).values
    total = 0
    for part in program.contributions:
        cost = analysis.cost.values.get(part.node.uid)
        if cost is not None:
            total += kept[part.node.uid] * _int(cost.active_params)
    return total


def project_analysis(analysis: Any) -> ModelReport:
    """Project one analysis into the tree report used by presentation tools."""
    blocks: List[Tuple[str, int, BlockReport]] = []
    stack_names: List[Optional[str]] = []
    aux_kinds: List[Optional[str]] = []
    derived_groups: List[Optional[Dict[str, Any]]] = []
    model_nodes: Dict[str, NodeReport] = {}
    top_level_params: Dict[str, int] = {}

    stage_by_meta = {
        id(stage.template.meta): stage
        for stack in analysis.program.stacks
        for stage in stack.stages
        if isinstance(stage, BlockStage)
    }
    stored_bytes = _stored_bytes_by_uid(analysis)
    for template in analysis.program.blocks:
        stage = stage_by_meta[id(template.meta)]
        meta = template.meta
        blocks.append((template.name, template.count,
                       _block(analysis, template, stored_bytes)))
        stack_names.append(stage.stack or meta.stack_name)
        aux_kinds.append(stage.aux_kind)
        derived_groups.append(meta.derived_group())

    for stack in analysis.program.stacks:
        for stage in stack.stages:
            if isinstance(stage, OpStage):
                projected = _project_node(
                    analysis, stage.node, stored_bytes=stored_bytes, per=1)
                model_nodes[stage.name] = projected
                repeated = stage.node.repeat is not None
                is_main_embedding = projected.boundary_role == "input_embedding"
                is_lm_head = projected.is_lm_head and not repeated
                if projected.params and not is_main_embedding and not is_lm_head:
                    top_level_params[stage.name] = projected.params

    total = analysis.total()
    main = analysis.main_total()
    by_stack = analysis.by_stack()
    aux = {name: cost for name, cost in by_stack.items() if name}
    tree_flops = sum(count * block.total_flops
                       for _name, count, block in blocks)
    # Every model-level node, not the head by name. `model_nodes` is exactly
    # the program's top-level roots — the ones `Program.contributions` hands
    # out at count 1 — and a model's tail is all of them: the embedding, the
    # final norm, the head. Adding back only the node whose `boundary_role` is
    # `output_head` dropped the final norm from `total_flops`, which is the
    # same defect `test_flops_closure` already names on the table side ("only
    # the LM head was emitted as the model-level tail, so the final norm was
    # in the per-op sum and not in the table") living in a second habitat. It
    # cost one op's `2 x d_model x s_ctx` — 25,165,824 on Llama-3.2-3B,
    # 16,777,216 on Qwen3-30B-A3B — under `display_formulas`' "Total forward
    # FLOPs", against a phase that counted it.
    #
    # The `Sigma count x block + tail` FORM is load-bearing and stays: it is
    # what lets `CacheView.on` re-count a pipeline stage by changing counts alone.
    # What changes is that the tail is now a fold over roots rather than a
    # name match on one of them.
    tree_flops += sum(node.flops for node in model_nodes.values())
    program = analysis.program
    reach = prefill_reach(analysis)
    prefill = reach.counts

    def on_stack(counts: Dict[int, int], stack: Optional[str]
                 ) -> Dict[int, int]:
        """`counts` with every root of another stack than `stack` at zero
        (`None` keeps every stack)."""
        if stack is None:
            return counts
        return {part.node.uid: (counts.get(part.node.uid, part.count)
                                if part.stack == stack else 0)
                for part in program.contributions}

    def prefill_active(stack: Optional[str] = None) -> int:
        return _whole_active(analysis, on_stack(prefill, stack))

    return ModelReport(
        block_results=blocks,
        total_params=_int(total.params),
        total_active_params=_whole_active(analysis),
        total_decode_active_params=_whole_active(
            analysis, {uid: 0 for uid in program.encoder_uids
                       if uid in prefill}),
        total_prefill_active_params=prefill_active(),
        prefill_full_from=reach.full_from,
        main_params=_int(main.params),
        main_active_params=_whole_active(analysis, on_stack({}, "")),
        main_prefill_active_params=prefill_active(""),
        lookup_table_params=_lookup_table_params(analysis),
        main_lookup_table_params=_lookup_table_params(analysis, ""),
        aux_prefill_active_params={name: prefill_active(name) for name in aux},
        total_flops=tree_flops,
        quant_metadata_params=_int(total.quant_metadata_params),
        aux_params={name: _int(cost.params) for name, cost in aux.items()},
        aux_active_params={name: _whole_active(analysis, on_stack({}, name))
                           for name in aux},
        aux_flops={name: _int(cost.flops) for name, cost in aux.items()},
        block_stack_names=stack_names,
        block_aux_kinds=aux_kinds,
        block_derived_groups=derived_groups,
        top_level_node_params=top_level_params,
        model_level_nodes=model_nodes,
        analysis=analysis,
    )


# -- report entries -------------------------------------------------------
#
# These assemble `Deployment(Program.of(...))` once so consumers do not each
# grow their own pipeline ceremony, and project the analysis into a report.
# The composition root (`transforms`) owns `Deployment`; this module composes
# on top of it, so the dependency runs one way and needs no deferred import.

def project_model(resolved: Any, *, batch: int = 1,
                         input_tokens: int = 1,
                         dim_overrides: Optional[Dict[str, Any]] = None,
                         logit_positions: Optional[int] = None,
                         dp_attention: int = 1,
                         config: Any = None,
                         phase: str = "prefill",
                         checkpoint: bool = False,
                         precision: Any = None) -> ModelReport:
    """Analyze a resolved spec and project a model report.

    Keep construction of the eval pipeline at this migration boundary so
    consumers do not each grow their own ``Deployment(Program.of(...))``
    ceremony while they are moved to native ``Analysis`` layers.
    `checkpoint` walks every encoder beside the language model
    (`Deployment.checkpoint`): the report a question about the whole
    checkpoint reads — its tensors, parameters and weights. `precision` is
    the deployment's precision profile, as shipped when None.
    """
    deployment = Deployment(Program.of(resolved), precision=precision).at(
        batch=batch, input_tokens=input_tokens,
        logit_positions=logit_positions,
    ).with_dims(dp_attention=max(1, int(dp_attention or 1)))
    if phase != "prefill":
        deployment = deployment.during(phase)
    if config is not None:
        deployment = replace(deployment, config=config)
    if dim_overrides:
        deployment = deployment.with_dims(**dim_overrides)
    analysis = (deployment.checkpoint() if checkpoint
                else deployment.analyze())
    return project_analysis(analysis)


def project_blocks(resolved: Any, *, batch: int = 1,
                          input_tokens: int = 1,
                          phase: str = "prefill",
                          precision: Any = None,
                          ) -> Dict[str, BlockReport]:
    """Analyze model-referenced blocks and index their presentation reports.

    A resolved program may contain unreferenced library templates; those are not
    executable model stages and intentionally do not appear here. `phase` is
    `project_model`'s: at decode each block is one new token per request against
    a cache of `input_tokens`, evaluated in its declared decode form under the
    block's own template key.
    """
    result = project_model(
        resolved, batch=batch, input_tokens=input_tokens, phase=phase,
        precision=precision,
    )
    return {
        template: block
        for template, _count, block in result.block_results
    }


def project_block(resolved: Any, template: str, *, batch: int = 1,
                         input_tokens: int = 1,
                         dim_overrides: Optional[Dict[str, Any]] = None,
                         phase: str = "prefill",
                         site: Optional[CallSite] = None,
                         precision: Any = None,
                         ) -> BlockReport:
    """Evaluate one block or component graph outside a model stack.

    The isolated graph is wrapped as a one-stage temporary model while keeping
    the original component library, config, and dimensions. This is the eval
    single-block entry point; consumers use it
    for validation and topology questions, not model accounting. The report
    is the component's: its nodes are the body the stage's call runs
    (`BlockTemplate.shown`), under the component's name.
    """
    isolated = isolate_resolved(resolved, template, dim_overrides, site)
    blocks = project_blocks(
        isolated, batch=batch, input_tokens=input_tokens, phase=phase,
        precision=precision,
    )

    try:
        report = blocks[isolated.site.block]
    except KeyError as exc:
        raise KeyError(f"isolated eval omitted block/component {template!r}") from exc
    return replace(report, template_name=template)
