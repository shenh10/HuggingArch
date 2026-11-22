"""The intermediate quantities a per-op popover names (`phase_formulas.
FormulaDefs`): each is its law at the point, its arithmetic folded and its
branch kept — numbers only, never a workload variable — and the value the
row's numbers were priced at."""
from __future__ import annotations

import re

import pytest

from tests.sweep_parity import Deployment

LLAMA = "meta-llama/Llama-3.2-3B"
#: The workload's counts by name: none may be left in a definition's formula.
_COUNTS = re.compile(r"\b(B|s_input|s_output|s_prefix|s_verify)\b")


@pytest.mark.parametrize("context", [0, 4096])
def test_physical_pair_definition_reads_guarded_workload_share(context):
    from backend.inference.phase_formulas import FormulaDefs

    definitions = {"s_pairs": {
        "law": "s_pairs * Quotient(Eq(s_input, 0), 0, ceiling(s_input / 8), s_input)",
        "bindings": {"s_pairs": {"value": context}, "s_input": {"value": context}},
    }}
    defs = FormulaDefs(dims=None, point={}, value=None, definitions=definitions)
    rendered = defs._defined("s_pairs", context // 8)
    assert rendered["definition"]
    assert not _COUNTS.search(rendered["formula"])


@pytest.fixture(scope="module")
def decode_rows():
    est = Deployment(LLAMA).direct(avg_input_tokens=4096, prefill_batch=1,
                                   decode_batch=1)
    return est["per_op_timing_decode"]


def _attention(rows):
    """The first layer's attention row."""
    return next(r for r in rows if r["op"].endswith("attention"))


def test_a_decode_steps_pairs_are_defined_and_read_off_the_point(decode_rows):
    """A decode step of a 4096-token prompt with 256 output tokens reads a
    cache of the prompt and half the output, 4224 keys, all live for its
    one query: `s_pairs` is shown as the phase declares it, with the
    point's numbers in it, and its value."""
    defs = {d["name"]: d for d in _attention(decode_rows)["flops_formula"]["defs"]}
    assert defs["s_pairs"] == {
        "name": "s_pairs",
        "definition": "s_past·s_query + s_query·(s_query + 1)/2",
        "formula": "4223·1 + 1·(1 + 1)/2", "value": "4,224"}


def test_the_cache_read_is_the_term_the_access_line_prints(decode_rows):
    """`kv` is one instance's cache read: the number the access line adds."""
    formula = _attention(decode_rows)["access_bytes_formula"]
    (kv,) = [d for d in formula["defs"] if d["name"] == "kv"]
    assert kv["value"] in formula["substituted"].split(" + ")
    assert not _COUNTS.search(kv["formula"])


def test_no_definition_names_a_count_an_alias_or_a_config_dim(decode_rows):
    """A count is its own symbol, an alias (`n_verify`, whose law is
    `s_verify`) a single symbol and a config dim a number: none is defined,
    and no formula keeps a workload variable."""
    for row in decode_rows:
        for key in ("flops_formula", "access_bytes_formula"):
            for d in (row.get(key) or {}).get("defs") or ():
                assert d["name"] not in ("B", "s_input", "s_output", "n_verify",
                                         "d_model", "d_head")
                assert not _COUNTS.search(d["formula"]), (row["aggregation_id"], d)
