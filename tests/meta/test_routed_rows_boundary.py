"""The rows a MoE dispatch routes onto a rank are eval's fact, stated once.

`eval.routed.RoutedLaunch` (`Analysis.routed`) is the one statement of a
rank's dispatch boundary — the tokens it holds, the rows routed onto its
experts, its experts, the padded launch — and `layout` produces the
per-rank evidence it is read from (`ExpectedRoutedLoad`). Everything
downstream (`inference`, `kernel_bench`, `rendering`, `api`, `service`,
`tools`) reads the record; none of it may derive the routed rows again.
The 2026-10-25 audit found six such derivations, each disagreeing with the
others somewhere.

What this gate refuses, by AST, outside `backend/arch/eval` and
`backend/arch/layout`:

  * arithmetic that combines the routing counts `n_experts_active` and
    `n_route_targets` — the per-expert mean `M = B·s·k·g/e` restated
    (`Evaluator._tokens_per_expert` is its one statement);
  * reading the routed-load evidence's row readings
    (`expected_rows_per_expert`, `expected_rows_on_rank`,
    `routed_row_capacity_per_rank`, `busiest_rank_load_factor`) — the
    per-rank rows `RoutedLaunch` already carries.

It is deliberately not a ban on the names: `n_experts_active` is a cell
axis of the router's own kernels (`moe_topk_softmax`, `moe_gate`), which
read it as the top-k width and derive no rows from it. A name ban would
refuse those and be argued down; this refuses only the derivation.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

#: Where the routed rows are produced: the walk and its per-rank rule.
_PRODUCERS = ("backend/arch/eval/", "backend/arch/layout/")
_SCANNED = ("backend", "kernel_bench", "tools")
_COUNTS = frozenset({"n_experts_active", "n_route_targets"})
#: The routed-load evidence's members that read ROWS — refused downstream.
_READINGS = frozenset({"routed_rows_in_cell", "routed_rows_per_expert",
                       "expected_rows_per_expert", "expected_rows_on_rank",
                       "routed_row_capacity_per_rank",
                       "busiest_rank_load_factor"})
#: Every other member of `ExpectedRoutedLoad`, and why reading it is no row
#: derivation. With `_READINGS` it covers the class exactly
#: (`test_the_readings_cover_the_evidence`), so a row reading added to the
#: evidence has to be classified here before anything can read it unseen.
_NOT_ROWS = {
    "group": "the route group, a placement",
    "expert_shard_axis": "the expert axis's cut, a placement",
    "num_experts": "the bank's expert count, a declaration",
    "local_experts": "the experts a rank holds, a placement",
    "load_assumption_id": "a label naming the routing assumption",
    "expert_load_factor": "rho, the measured expert skew, an input",
    "fanout": ("the dispatch's top-k fan-out (`RouteFanout`): the "
               "destinations a token's picks land on, which prices the "
               "exchange and derives no rows"),
    "row_replication": "how many copies of a row a gather makes, a layout",
    "dispatched_rows": "the rows axis as dispatched, a layout",
    "is_measure_preserving": "a conservation check, no number",
    "replicate_rows": "builds a new record, reads nothing",
    "as_dispatched": "builds a new record (a row gather undone), reads nothing",
}


def _count_named(node: ast.AST) -> Optional[str]:
    """The routing count an operand names: a variable, an attribute, or a
    string key (`dims["n_route_targets"]`, `d.get("n_experts_active")`)."""
    if isinstance(node, ast.Name) and node.id in _COUNTS:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in _COUNTS:
        return node.attr
    if isinstance(node, ast.Constant) and node.value in _COUNTS:
        return node.value
    return None


def _arithmetic_counts(expr: ast.BinOp) -> set:
    """Every routing count an arithmetic expression reads, down through its
    operands (subscripts and calls included: `d["k"]`, `d.get("k")`)."""
    found = set()
    for node in ast.walk(expr):
        named = _count_named(node)
        if named is not None:
            found.add(named)
    return found


def _violations(path: Path) -> Iterator[Tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and _arithmetic_counts(node) == _COUNTS:
            yield node.lineno, "derives rows from n_experts_active / n_route_targets"
        if isinstance(node, ast.Attribute) and node.attr in _READINGS:
            yield node.lineno, f"reads the routed-load evidence's `{node.attr}`"


def _scanned_files() -> Iterator[Path]:
    for top in _SCANNED:
        for path in sorted((PROJECT_ROOT / top).rglob("*.py")):
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            if not rel.startswith(_PRODUCERS):
                yield path


def test_nothing_downstream_derives_the_routed_rows():
    found: List[str] = []
    for path in _scanned_files():
        for line, what in _violations(path):
            found.append(f"{path.relative_to(PROJECT_ROOT)}:{line}: {what}")
    assert not found, (
        "the rows a dispatch routes onto a rank are `Analysis.routed`'s "
        "(`RoutedLaunch`); read the record instead:\n" + "\n".join(found))


def test_the_gate_sees_a_derivation(tmp_path):
    """The gate's own check: each form it refuses, written the way a
    consumer would, is caught; the router's own top-k read is not."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        "m = dims['B'] * dims['n_experts_active'] / dims['n_route_targets']\n"
        "r = load.expected_rows_per_expert * 4\n"
        "k = int(dims['n_experts_active'])\n")
    lines = sorted(line for line, _what in _violations(probe))
    assert lines == [1, 2], lines
    assert 3 not in lines


def test_the_readings_cover_the_evidence():
    """The refused readings are `ExpectedRoutedLoad`'s own members, and its
    members are each either refused or classified as no row reading, so
    this list cannot fall behind the class it restates."""
    import dataclasses
    from backend.arch.layout.workload import ExpectedRoutedLoad

    members = {field.name for field in dataclasses.fields(ExpectedRoutedLoad)}
    members |= {name for name in vars(ExpectedRoutedLoad)
                if not name.startswith("_")}
    assert members >= _READINGS, sorted(_READINGS - members)
    assert set(_NOT_ROWS) <= members, sorted(set(_NOT_ROWS) - members)
    assert not _READINGS & set(_NOT_ROWS)
    assert members == _READINGS | set(_NOT_ROWS), (
        "classify each new member of ExpectedRoutedLoad: a row reading in "
        "_READINGS, anything else in _NOT_ROWS with its reason: "
        f"{sorted(members - _READINGS - set(_NOT_ROWS))}")
