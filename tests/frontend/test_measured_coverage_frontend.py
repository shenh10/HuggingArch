"""The measured track on the page, all or nothing (`web/src/measured.ts`,
docs/auto_sweep.md §6).

An end-to-end measured total — the Inference page's step time and its
per-op table's Total row, a Scaling chart point, a Gallery point — is shown
only when every op of the step is measured; otherwise the page shows the
coverage ("N / M ops measured" and the unmeasured keys by state), never the
partial or mixed number the backend may still send. A per-op row shows its
own measured time under any coverage, and 0 is a value.

Same harness as `test_pages_smoke.py` (built `web/dist`, canned payloads):
the responses are the DeepSeek-V3 estimate fixture with its measured fields
rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.frontend.test_pages_smoke import Session, browser_base  # noqa: F401
from backend.inference.phase_formulas import table_total

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_LINK = ("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
         "&framework=vllm&selection=latest")
_STATES = {"missing": ["rmsnorm|d_model=7168", "linear|n=1536"],
           "unplaced": ["all_reduce"], "ambiguous": [], "declined": []}


def _estimate() -> dict:
    return json.loads((_FIXTURES / "estimate_deepseek_v3_b200.json").read_text())


def _leaf(rows: list, i: int) -> bool:
    return not (i + 1 < len(rows) and (rows[i + 1].get("depth") or 0) > (rows[i].get("depth") or 0))


def _rollup(rows: list) -> None:
    """`t_measured_agg` as the backend states it (`batch_sweep._measured_table`:
    the formula's share of the leaves under a row, None unless every one of
    them is measured). Here a plain sum stands in for the formula's share."""
    for i, r in enumerate(rows):
        d = r.get("depth") or 0
        under = [i] if _leaf(rows, i) else []
        for j in range(i + 1, len(rows)):
            if (rows[j].get("depth") or 0) <= d:
                break
            if _leaf(rows, j):
                under.append(j)
        times = [rows[k].get("t_measured_s") for k in under]
        r["t_measured_agg"] = (None if not times or any(t is None for t in times)
                               else sum(times))


def _with_measured(per_leaf, *, t_step, measured_rows, total_rows, coverage=True) -> dict:
    """The fixture with decode's leaves measured by `per_leaf(row)` (None:
    unmeasured) and the step's measured fields set as given."""
    est = _estimate()
    dec = est["phases"]["decode"]
    rows = dec["flops_by_module"]
    for i, r in enumerate(rows):
        r["t_measured_s"] = per_leaf(r) if _leaf(rows, i) else None
    _rollup(rows)
    # The table's Total row as the backend publishes it beside the fold
    # (`enrich_module_breakdown`); the fixture predates it.
    dec["op_table_total"] = table_total(rows)
    dec["t_measured_s"] = t_step
    dec["measured_rows"] = measured_rows
    dec["total_rows"] = total_rows
    if coverage:
        complete = measured_rows == total_rows
        dec["coverage"] = {"measured": measured_rows, "total": total_rows,
                           "states": {} if complete else _STATES}
    return est


def _page(browser, base, est):
    s = Session(browser, base, overrides={"/inference/estimate": est})
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    s.page.wait_for_selector("#infer-phase-breakdown", state="attached", timeout=15000)
    return s


def _metric(s) -> str:
    return s.page.inner_text("#infer-phase-table [data-measured-total]")


def _total_cell(s) -> str:
    return s.page.inner_text("#infer-phase-breakdown [data-total-row] [data-measured-total]")


def _leaf_cell(s, name: str) -> str:
    return s.page.eval_on_selector_all(
        "#infer-phase-breakdown tr[data-leaf-row]",
        """(els, name) => els.filter(e => e.innerText.includes(name))
             .map(e => e.querySelector('[data-measured-cell]').innerText.trim())[0]""",
        name)


def _template_cell(s) -> str:
    return s.page.eval_on_selector_all(
        "#infer-phase-breakdown tr[data-fold-row][data-depth='0']",
        "els => els.map(e => e.querySelector('[data-measured-cell]').innerText.trim())[0]")


def test_a_complete_measured_total_renders(browser_base):  # noqa: F811
    browser, base = browser_base
    est = _with_measured(lambda r: r.get("t_op_s") or 0.0, t_step=1.3e-3,
                         measured_rows=73, total_rows=73)
    s = _page(browser, base, est)
    assert _metric(s).strip() == "1.3 ms"
    # The Total row is the backend's total (the one measured formula), not
    # the page's sum of the rows — which here would read 1.4 ms.
    assert _total_cell(s).splitlines()[0].strip() == "1.3 ms"
    # Every leaf under the block is measured, so the block row shows the
    # backend's share for it.
    assert _template_cell(s).splitlines()[0] == "1.4 ms"
    s.assert_clean()


def test_a_partial_measured_total_renders_only_coverage(browser_base):  # noqa: F811
    """The backend still sends today's mixed total (12.3 ms); below full
    coverage the page shows the coverage and the unmeasured keys' counts."""
    browser, base = browser_base
    est = _with_measured(lambda r: r.get("t_op_s") if r["name"] == "embed_tokens" else None,
                         t_step=12.3e-3, measured_rows=1, total_rows=73)
    s = _page(browser, base, est)
    metric = _metric(s)
    assert "12.3 ms" not in metric, metric
    assert metric.strip() == "— (1 / 73 ops measured; missing 2 · unplaced 1)", metric
    assert _total_cell(s).strip() == "—"
    assert _template_cell(s) == "—"
    s.assert_clean()


def test_a_partial_total_without_coverage_states_is_still_not_shown(browser_base):  # noqa: F811
    """Today's response: `measured_rows` / `total_rows`, no `coverage`."""
    browser, base = browser_base
    est = _with_measured(lambda r: r.get("t_op_s") if r["name"] == "embed_tokens" else None,
                         t_step=12.3e-3, measured_rows=1, total_rows=73, coverage=False)
    s = _page(browser, base, est)
    assert _metric(s).strip() == "— (1 / 73 ops measured)"
    s.assert_clean()


def test_a_breakdown_row_shows_its_measurement_under_partial_coverage(browser_base):  # noqa: F811
    browser, base = browser_base
    est = _with_measured(lambda r: 1.876e-6 if r["name"] == "embed_tokens" else None,
                         t_step=None, measured_rows=1, total_rows=73)
    s = _page(browser, base, est)
    assert _leaf_cell(s, "embed_tokens").startswith("1.9 µs")
    # An unmeasured row is blank.
    assert _leaf_cell(s, "input_layernorm") == "—"
    s.assert_clean()


def test_a_measured_zero_renders_as_zero(browser_base):  # noqa: F811
    browser, base = browser_base
    est = _with_measured(lambda r: 0.0 if r["name"] == "embed_tokens" else None,
                         t_step=None, measured_rows=1, total_rows=73)
    s = _page(browser, base, est)
    assert _leaf_cell(s, "embed_tokens").startswith("0")
    s.assert_clean()


def test_scaling_draws_a_measured_point_only_at_full_coverage(browser_base):  # noqa: F811
    """2 GPUs is fully measured; 4 carries a mixed rate under partial
    coverage; 8 a rate with no coverage at all. Only 2 is drawn; the others'
    hover gives the coverage instead of a number. Prefill is a value on the
    throughput cloud's points, so this is the one place a measured rate of
    the Scaling panel is drawn."""
    from tests.frontend.test_pages_smoke import (
        prefill_measured_session, prefill_point, prefill_tooltip)

    full = {"measured": 73, "total": 73}
    part = {"measured": 40, "total": 73, "states": _STATES}
    browser, base = browser_base
    s = prefill_measured_session(browser, base, [
        prefill_point(2, 80, full), prefill_point(4, 70, part),
        prefill_point(8, 60, None)])
    at2 = prefill_tooltip(s, 0, 3)
    assert "· measured: 80 tok/s" in at2, at2
    assert "ops measured" not in at2, at2
    at4 = prefill_tooltip(s, 1, 3)
    assert "· measured:" not in at4, at4
    assert "measured: — (40 / 73 ops measured; missing 2 · unplaced 1)" in at4, at4
    at8 = prefill_tooltip(s, 2, 3)
    assert "· measured:" not in at8, at8
    assert "coverage not reported" in at8, at8
    s.assert_clean()
