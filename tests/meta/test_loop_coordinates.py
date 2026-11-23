"""Every reader of a stack's iterations says which coordinate it means.

A looped stack (`backend/arch/DESIGN.md`, "Looped stacks") numbers its iterations by
EXECUTION; its stored layers are a projection of them
(`BlockMeta.stored_indices` / `layer_numbers`, `Program.stored_layers`,
`BlockMeta.stored_count`). Outside a loop the two coincide, so a reader that
means "stored layer" but reads the execution coordinate is right on every
other model and wrong on a looped one — and nothing else turns red. This scan
finds every read of the execution coordinate under `backend/` — the
attributes `layer_indices`, `occurrence_layers`, `stage_iterations` and
`count` (every receiver: a name says nothing about a type, so `.count` is
read wherever it is not a method call such as `str.count(...)`), and the
same names passed to `getattr` / `attrgetter` — keyed by file, enclosing
function, attribute and receiver expression, and holds them to the table
below. A new reader, or an old one under a new receiver, is red until it is
classified here.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

#: The attributes that carry the execution coordinate.
_ATTRIBUTES = ("layer_indices", "occurrence_layers", "stage_iterations",
               "count")
#: Functions that read an attribute named by a string.
_DYNAMIC = ("getattr", "attrgetter")

E, H, S, N = "execution", "holder", "stored", "other"
#: (file, enclosing function, attribute, receiver) → what the reader means:
#: `execution` — iterations as the walk runs them, or how many times an
#: invocation runs, which is what they mean in a loop too; `holder` — the
#: copies of something that stores weights (a stored tensor's row, a copy
#: factor, an invocation that holds weights: a later pass holds none, so a
#: holder's count IS its stored copies); `stored` — the stored projection
#: itself, or a field of that name already holding stored layers; `other` —
#: a `count` of something that is no block invocation (a header group, a
#: collective, a calibration tally). A reader that names or counts LAYERS
#: reads `layer_numbers` / `stored_count` / `stored_layers` and is not
#: listed.
READERS: Dict[Tuple[str, str, str, str], str] = {
    # The walk, the occurrences, the chain, the streams, PP, fusion, cache.
    ("backend/arch/eval/activation_edges.py", "iteration_classes", "occurrence_layers", "program"): E,
    ("backend/arch/eval/cache.py", "Edges._runs", "occurrence_layers", "self.program"): E,
    ("backend/arch/eval/cache_view.py", "CacheView.occupancy", "count", "c"): E,
    ("backend/arch/eval/evaluator.py", "Evaluator._element", "layer_indices", "stage.template"): E,
    ("backend/arch/eval/evaluator.py", "Evaluator._eval_link", "count", "template"): E,
    ("backend/arch/eval/evaluator.py", "Evaluator._eval_link", "layer_indices", "template"): E,
    ("backend/arch/eval/evaluator.py", "Evaluator._eval_link", "layer_indices", "other.template"): E,
    ("backend/arch/eval/evaluator.py", "Evaluator._eval_stage", "count", "template"): E,
    ("backend/arch/eval/fusion.py", "FusedProgram.occurrence_layers", "occurrence_layers", "self.base"): E,
    ("backend/arch/eval/fusion.py", "Seams.of", "occurrence_layers", "program"): E,
    ("backend/arch/eval/fusion.py", "_Runs.has", "occurrence_layers", "self.program"): E,
    ("backend/arch/eval/fusion.py", "_Runs.take", "occurrence_layers", "self.program"): E,
    ("backend/arch/eval/layer.py", "Layer.rebase_counts", "occurrence_layers", "source"): E,
    ("backend/arch/eval/layer.py", "Layer.rebase_counts.per_run", "occurrence_layers", "source"): E,
    ("backend/arch/eval/layer.py", "Rewrite.iteration_classes", "occurrence_layers", "program"): E,
    ("backend/arch/eval/program.py", "BlockTemplate.count", "count", "self.meta"): E,
    ("backend/arch/eval/program.py", "BlockTemplate.layer_indices", "count", "self"): E,
    ("backend/arch/eval/program.py", "BlockTemplate.layer_indices", "layer_indices", "self.meta"): E,
    ("backend/arch/eval/program.py", "Program.block_stream_reads", "layer_indices", "template"): E,
    ("backend/arch/eval/program.py", "Program.contributions", "count", "template"): E,
    ("backend/arch/eval/program.py", "Program.declared_stream_reads", "layer_indices", "template"): E,
    ("backend/arch/eval/program.py", "Program.inner_stream_reference", "layer_indices", "template"): E,
    ("backend/arch/eval/program.py", "Program.iteration_space", "layer_indices", "stage.template"): E,
    ("backend/arch/eval/program.py", "Program.multiplicity", "count", "contribution"): E,
    ("backend/arch/eval/program.py", "Program.occurrence_domains", "occurrence_layers", "self"): E,
    ("backend/arch/eval/program.py", "Program.occurrence_layers", "stage_iterations", "self"): E,
    ("backend/arch/eval/program.py", "Program.stack_iterations", "layer_indices", "stage.template"): E,
    ("backend/arch/eval/program.py", "Program.stage_iterations", "layer_indices", "stage.template"): E,
    ("backend/arch/eval/program.py", "Program.stream_writers", "layer_indices", "template"): E,
    ("backend/arch/eval/program.py", "Stack.spanning", "layer_indices", "other.template"): E,
    ("backend/arch/eval/program.py", "Stack.spanning", "layer_indices", "stage.template"): E,
    ("backend/arch/eval/program.py", "_first_iteration", "layer_indices", "stage.template"): E,
    ("backend/arch/layout/stages.py", "StagePlacement._occurrence_edges", "stage_iterations", "self.program"): E,
    ("backend/arch/layout/stages.py", "StagePlacement.block_counts", "count", "template"): E,
    ("backend/arch/layout/stages.py", "StagePlacement.block_counts", "layer_indices", "template"): E,
    ("backend/arch/layout/stages.py", "StagePlacement.contribution_counts", "count", "contribution"): E,
    ("backend/arch/spec_tree/build.py", "SpecTreeWalker.build_ir_tree", "count", "meta"): E,
    ("backend/arch/spec_tree/build.py", "_BlockEnumerator.run", "count", "meta"): E,
    ("backend/arch/spec_tree/build.py", "_BlockEnumerator.run", "layer_indices", "meta"): E,
    ("backend/arch/spec_tree/build.py", "_BlockEnumerator.split_by_pass", "layer_indices", "meta"): E,
    ("backend/arch/spec_tree/build.py", "_BlockEnumerator.visibility_groups", "count", "meta"): E,
    ("backend/arch/spec_tree/build.py", "_BlockEnumerator.visibility_groups", "layer_indices", "meta"): E,
    ("backend/arch/spec_tree/build.py", "_StreamTable.build", "count", "meta"): E,
    ("backend/arch/spec_tree/build.py", "_StreamTable.build", "layer_indices", "meta"): E,
    ("backend/arch/spec_tree/build.py", "declared_blocks", "count", "meta"): E,
    ("backend/arch/spec_tree/build.py", "declared_blocks", "layer_indices", "meta"): E,
    ("backend/inference/activation_peak.py", "_eval_leaf_records", "count", "contribution"): E,
    ("backend/playground/module_bench.py", "whole_model_kv", "count", "entry"): E,
    ("backend/arch/eval/cache.py", "prefill_reach", "layer_indices", "st.template"): E,
    ("backend/arch/eval/evaluator.py", "Evaluator._elements", "layer_indices", "st.template"): E,
    ("backend/arch/eval/evaluator.py", "Evaluator._name_final", "layer_indices", "st.template"): E,
    ("backend/arch/eval/layer.py", "Rewrite.iteration_classes", "layer_indices", "program.blocks[block]"): E,
    ("backend/arch/eval/program.py", "Program._element_reads", "layer_indices", "template"): E,
    ("backend/arch/eval/program.py", "Program.whole_stream_pairings", "layer_indices", "stage.template"): E,
    # A derived group (`read: all`), which a looped stack cannot declare.
    ("backend/arch/spec_tree/nodes.py", "BlockMeta.derived_group", "count", "self"): E,
    ("backend/arch/spec_tree/nodes.py", "BlockMeta.derived_group", "layer_indices", "self"): E,
    # Per-invocation multipliers and execution counts, each row named by its
    # invocation (`label` / `short_name` say the pass).
    ("backend/analyzer.py", "_analyze_from_spec", "count", "template"): E,
    ("backend/arch/eval/analysis.py", "Analysis._contributions", "count", "part"): E,
    ("backend/arch/eval/projection/formulas.py", "display_formulas", "count", "template"): E,
    ("backend/arch/eval/projection/result.py", "_block", "count", "template"): E,
    ("backend/arch/eval/projection/result.py", "_lookup_table_params", "count", "part"): E,
    ("backend/arch/eval/projection/result.py", "project_analysis", "count", "template"): E,
    ("backend/arch/eval/projection/result.py", "project_analysis.on_stack", "count", "part"): E,
    ("backend/arch/eval/transforms.py", "CallSite.calls", "count", "program.blocks[index]"): E,
    ("backend/arch/eval/transforms.py", "CallSite.calls", "count", "template"): E,
    ("backend/arch/structure.py", "layer_configs", "count", "template"): E,
    ("backend/arch/structure.py", "render_structure", "count", "meta"): E,
    ("backend/inference/module_rows.py", "physical_module_rows", "count", "stage.template"): E,
    ("backend/playground/compose.py", "Composition.check_skeleton", "count", "meta"): E,
    ("backend/playground/compose.py", "_group_dto", "count", "g.meta"): E,
    ("backend/playground/compose.py", "_group_dto", "count", "meta"): E,
    ("backend/playground/module_bench.py", "_encoder_only_components", "count", "template"): E,
    ("backend/playground/module_bench.py", "_runs_of_row", "occurrence_layers", "pre_program"): E,
    ("backend/rendering/_compute_graph.py", "serialize_post_tree._emit_block", "count", "meta"): E,
    ("backend/rendering/_compute_graph.py", "slice_pipeline_fragment", "count", "template"): E,
    ("backend/rendering/_layout.py", "layout_assignments_in_order", "layer_indices", "meta"): E,
    ("backend/rendering/builders.py", "build_model_config", "count", "template"): E,
    ("backend/views/composition.py", "calls_of", "count", "template"): E,
    # Copies of what stores weights.
    ("backend/analyzer.py", "_analyze_from_spec", "count", "row"): H,
    ("backend/arch/eval/analysis.py", "Analysis._bytes_by", "count", "row"): H,
    ("backend/arch/eval/analysis.py", "Analysis._stored_parts", "count", "row"): H,
    ("backend/arch/eval/analysis.py", "Analysis._with_metadata", "count", "row"): H,
    ("backend/arch/eval/analysis.py", "Analysis.checkpoint_buffer_bytes", "count", "row"): H,
    ("backend/arch/eval/analysis.py", "Analysis.checkpoint_param_bytes", "count", "row"): H,
    ("backend/arch/eval/analysis.py", "Analysis.dominant_weight_scale_dtype", "count", "row"): H,
    ("backend/arch/eval/analysis.py", "Analysis.precision_effective", "count", "row"): H,
    ("backend/arch/eval/analysis.py", "Analysis.stored_bytes_by_stack", "count", "row"): H,
    ("backend/arch/eval/checkpoint.py", "Reconciliation.of", "count", "row"): H,
    ("backend/arch/eval/copies.py", "_block_factors", "count", "meta"): H,
    ("backend/arch/eval/copies.py", "copy_factors_of", "count", "factor"): H,
    ("backend/arch/eval/projection/formulas.py", "_weights_formula", "count", "row"): H,
    ("backend/arch/eval/projection/result.py", "_stored_bytes_by_uid", "count", "row"): H,
    ("backend/arch/eval/rule.py", "Rule.resident_weights", "count", "row"): H,
    ("backend/arch/eval/traffic.py", "_weight_traffic._bytes", "count", "row"): H,
    ("backend/arch/layout/propagation.py", "LayoutPropagation.local_tensors", "count", "row"): H,
    ("backend/arch/validate/_matching.py", "bind_leaves_to_hf", "count", "row"): H,
    ("backend/arch/validate/_matching.py", "match_checkpoint_to_hf", "count", "row"): H,
    ("backend/arch/validate/_matching.py", "spec_vs_tensors_check", "count", "row"): H,
    ("backend/arch/validate/_orchestrator.py", "_headline_blocks", "count", "row"): H,
    ("backend/arch/validate/_packing_consistency.py", "run_packing_consistency_check", "count", "row"): H,
    ("backend/inference/weight_breakdown.py", "_copy_factors", "count", "f"): H,
    ("backend/inference/weight_breakdown.py", "_copy_factors", "count", "factor"): H,
    ("backend/inference/weight_breakdown.py", "_leaf_node", "count", "row"): H,
    ("backend/inference/weight_breakdown.py", "_reject_ambiguous_template_names", "count", "m"): H,
    ("backend/inference/weight_breakdown.py", "_reject_ambiguous_template_names", "count", "meta"): H,
    ("backend/inference/weight_breakdown.py", "_resolve_leaf_bytes", "count", "row"): H,
    ("backend/inference/weight_breakdown.py", "per_rank_weight_breakdown", "count", "meta"): H,
    ("backend/inference/weight_breakdown.py", "per_rank_weight_breakdown", "count", "row"): H,
    ("backend/playground/module_bench.py", "walk_dim_names", "count", "row"): H,
    # The stored projection itself, and fields of that name already holding
    # stored layers (`ModelReport.layer_positions`).
    ("backend/arch/eval/program.py", "Program.iteration_label", "layer_indices", "meta"): S,
    ("backend/arch/eval/program.py", "Program.stored_layers", "layer_indices", "meta"): S,
    ("backend/arch/eval/program.py", "Program.stored_layers", "occurrence_layers", "self"): S,
    ("backend/arch/eval/program.py", "Program.where", "layer_indices", "meta"): S,
    ("backend/arch/layout/stages.py", "StagePlacement.stored_counts", "layer_indices", "block.template"): S,
    ("backend/arch/spec_tree/nodes.py", "BlockMeta.layer_numbers", "count", "self"): S,
    ("backend/arch/spec_tree/nodes.py", "BlockMeta.layer_numbers", "layer_indices", "self"): S,
    ("backend/arch/spec_tree/nodes.py", "BlockMeta.stored_count", "count", "self"): S,
    ("backend/views/composition.py", "ModelComposition._block_rows", "layer_indices", "row"): S,
    ("backend/views/composition.py", "ModelComposition._block_rows", "layer_indices", "r"): S,
    ("backend/views/composition.py", "_Row.payload", "layer_indices", "self"): S,
    # Counts of things that are no block invocation.
    ("backend/arch/eval/checkpoint.py", "_fill_header_side", "count", "group"): N,
    ("backend/arch/layout/materialize_comm.py", "PhysicalCollective.__post_init__", "count", "self"): N,
    ("backend/arch/layout/materialize_comm.py", "PhysicalCollective.at_width", "count", "self"): N,
    ("backend/arch/layout/materialize_comm.py", "_comm_layers", "count", "event"): N,
    ("backend/arch/validate/_model_card.py", "_near", "count", "stated"): N,
    ("backend/arch/validate/_model_card.py", "run_model_card_check", "count", "s"): N,
    ("backend/cli_bench.py", "emit_suggest_shapes", "count", "point"): N,
    ("backend/hf/header.py", "SafetensorsHeader.tensor_count", "count", "g"): N,
    ("backend/hf/header.py", "Selection.complete", "count", "self"): N,
    ("backend/hf/metadata.py", "format_header", "count", "g"): N,
    ("backend/hf/metadata.py", "summarize", "count", "group"): N,
    ("backend/inference/comm_cost.py", "CollectiveSignature.key", "count", "self"): N,
    ("backend/inference/comm_cost.py", "CollectiveSignature.symbol_suffix", "count", "self"): N,
    ("backend/inference/comm_cost.py", "Envelope.choice", "count", "sig"): N,
    ("backend/inference/comm_cost.py", "collective_envelope", "count", "sig"): N,
    ("backend/inference/op_calibration.py", "_Divergences.__init__", "count", "self"): N,
    ("backend/inference/op_calibration.py", "_Divergences.add", "count", "self"): N,
    ("backend/inference/op_calibration.py", "_Divergences.report", "count", "self"): N,
    ("backend/inference/phase_formulas.py", "FormulaDefs.attach", "count", "self"): N,
    ("backend/inference/row_roofline.py", "RowRoofline.comm", "count", "sig"): N,
}


def _hits() -> Iterator[Tuple[str, str, str, str]]:
    for path in sorted((ROOT / "backend").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # `x.count(...)` is a method call (`str.count`, `list.count`), not
        # an attribute read.
        called = {id(node.func) for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "count"}
        stack: list = []
        hits: list = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ClassDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_Attribute(self, node):
                if node.attr in _ATTRIBUTES and id(node) not in called:
                    hits.append((rel, ".".join(stack) or "<module>",
                                 node.attr, ast.unparse(node.value)))
                self.generic_visit(node)

            def visit_Call(self, node):
                func = node.func
                name = (func.id if isinstance(func, ast.Name)
                        else func.attr if isinstance(func, ast.Attribute)
                        else "")
                if name in _DYNAMIC:
                    hits.extend(
                        (rel, ".".join(stack) or "<module>", arg.value,
                         f"{name}(...)")
                        for arg in node.args
                        if isinstance(arg, ast.Constant)
                        and arg.value in _ATTRIBUTES)
                self.generic_visit(node)

        Visitor().visit(tree)
        yield from hits


def test_every_reader_of_the_execution_coordinate_is_classified():
    found = set(_hits())
    unclassified = sorted(found - set(READERS))
    stale = sorted(set(READERS) - found)
    assert not unclassified, (
        "these read a stack's iterations or a `count` and are not "
        "classified; a reader that names or counts a layer (a page, a "
        "checkpoint index, a per-layer format) reads the stored coordinate "
        "(`layer_numbers` / `stored_count` / `stored_layers`); one that "
        "counts or orders executions, or counts something else, is listed "
        "here with what it means:\n"
        + "\n".join(f"  {hit}" for hit in unclassified))
    assert not stale, ("listed readers no longer read the coordinate; drop "
                       "them:\n" + "\n".join(f"  {s}" for s in stale))
