"""The Inspector/dashboard node popovers: values lines are substituted.

Every node whose symbolic flops/params line names dims must render the
values line as the numeric substitution (`2*1*12288*4096*1024`), bound from
the walk's own substitution record — never the folded product, and never
with a symbol left over. The bound form must reproduce the node's figure
exactly (the record is what the evaluation substituted, so it must).
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import resolve_snapshot

IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _report(model_name):
    resolved = resolve_snapshot(model_name)
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    from backend.arch.eval.projection import project_analysis
    analysis = (Deployment(Program.of(resolved))
                .at(batch=1, input_tokens=1024, logit_positions=1)
                .analyze())
    return project_analysis(analysis)


def _nodes(report):
    def visit(node):
        yield node
        for child in node.children or ():
            yield from visit(child)
    for _template, _count, block in report.block_results:
        for node in block.nodes:
            yield from visit(node)
    for node in report.model_level_nodes.values():
        yield from visit(node)


class TestNodeValuesAreSubstituted(unittest.TestCase):
    MODEL = "Qwen__Qwen3-8B"

    def test_every_node_binds_its_record(self):
        report = _report(self.MODEL)
        checked_flops = checked_params = 0
        for node in _nodes(report):
            if node.flops_symbolic and IDENT.search(node.flops_symbolic):
                checked_flops += 1
                self.assertNotRegex(
                    node.flops_values, IDENT,
                    f"{node.spec_id}: flops values line not substituted: "
                    f"{node.flops_values!r}")
                bound = sp.sympify(node.flops_values)
                self.assertEqual(
                    bound, sp.sympify(node.flops),
                    f"{node.spec_id}: values line {node.flops_values!r} != "
                    f"flops {node.flops}")
            if (node.params_symbolic and IDENT.search(node.params_symbolic)
                    and node.params):
                checked_params += 1
                self.assertNotRegex(
                    node.params_values, IDENT,
                    f"{node.spec_id}: params values line not substituted: "
                    f"{node.params_values!r}")
                self.assertEqual(
                    sp.sympify(node.params_values), sp.sympify(node.params),
                    f"{node.spec_id}: values line {node.params_values!r} != "
                    f"params {node.params}")
        self.assertGreater(checked_flops, 20, "too few nodes checked")
        self.assertGreater(checked_params, 10, "too few nodes checked")

    def test_component_nodes_bind_too(self):
        # A component's cost is the sum of its children's — its values line
        # binds the merged leaf records, not a folded sum.
        report = _report(self.MODEL)
        for node in _nodes(report):
            if not node.children:
                continue
            if node.flops_symbolic and IDENT.search(node.flops_symbolic):
                self.assertNotRegex(
                    node.flops_values, IDENT,
                    f"component {node.spec_id}: values line folded or "
                    f"symbolic: {node.flops_values!r}")
                return
        self.fail("no component with a symbolic flops line found")


if __name__ == "__main__":
    unittest.main()
