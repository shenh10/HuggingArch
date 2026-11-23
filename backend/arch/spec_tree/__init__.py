"""Layer-0 spec walker — single source of truth.

Two primary APIs, both memoized per ResolvedSpec:

  • `iter_blocks(resolved) -> List[BlockMeta]`
        Block-template invocation metadata: which template each layer
        slot uses, how many copies, which stack it lives in, whether
        it's an aux group (mtp / vision / audio / speculative), and
        any assignment-level dim overrides.

  • `iter_leaves(resolved) -> List[LeafTensor]`
        One `LeafTensor` per weight-bearing leaf, with HF-folded
        `spec_id` (matches renderer ids), HF on-disk `tensor_name`,
        `shape`, `multiplicity`, `role`, `component_path`, `dim_symbols`.

Nothing here evaluates cost. It walks `spec.model` and the spec's
BlockGraphs directly. Rendering, materialize's test path, and the
inference layer (weight_breakdown / capacity) consume structure from
here.

Layout: `nodes` holds the IR data types, `build` constructs them
(`SpecTreeWalker` + `iter_tree`), `projections` answers questions off
the tree. Internal dim resolution goes through `eval.dim.eval_dim_expr`
— the same funnel the evaluator uses — so IR stays mode-agnostic even
though the current public APIs return numeric data only.
"""
from .build import (
    SpecTreeWalker,
    StreamReadRefusal,
    compute_dispatch_zone,
    enumerate_blocks,
    model_spine,
    spine_of_spec,
    finalize_ir_tree,
    is_pure_delegation,
    iter_blocks,
    layer_at,
    iter_tree,
    module_name_of,
    populate_weight_names,
    wire_parents_and_uids,
)
from .nodes import (
    BlockMeta,
    InputRef,
    LeafTensor,
    SpecIRNode,
    SpecIRTree,
    STACK_ENTRY,
    SpineEntry,
    StackReads,
    TreeStream,
    iter_block_scopes,
    validate_overrides_against_spec,
)
from .projections import (
    as_matrix,
    checkpoint_prefix,
    collect_leaves,
    compose_tensor_name,
    ir_count_symbols,
    ir_dim_symbols,
    ir_to_leaf,
    iter_leaves,
    require_module,
    resolve_symbols,
    scope_prefix,
    symbols_in,
    to_int,
    world_of_dim,
)

__all__ = [
    "STACK_ENTRY",
    "BlockMeta",
    "InputRef",
    "LeafTensor",
    "SpecIRNode",
    "SpecIRTree",
    "SpecTreeWalker",
    "SpineEntry",
    "StackReads",
    "StreamReadRefusal",
    "TreeStream",
    "as_matrix",
    "checkpoint_prefix",
    "collect_leaves",
    "compose_tensor_name",
    "compute_dispatch_zone",
    "enumerate_blocks",
    "finalize_ir_tree",
    "ir_count_symbols",
    "ir_dim_symbols",
    "ir_to_leaf",
    "is_pure_delegation",
    "iter_block_scopes",
    "iter_blocks",
    "iter_leaves",
    "iter_tree",
    "layer_at",
    "model_spine",
    "module_name_of",
    "populate_weight_names",
    "require_module",
    "resolve_symbols",
    "scope_prefix",
    "spine_of_spec",
    "symbols_in",
    "to_int",
    "validate_overrides_against_spec",
    "wire_parents_and_uids",
    "world_of_dim",
]
