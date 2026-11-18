"""The Inference page shows what the backend computed, and nothing else.

Same harness as `test_pages_smoke.py` (built `web/dist` in headless
chromium, canned payloads). Each test feeds a value the page used to derive
itself and checks the page shows the backend's: the GPU count of the
deployment being edited, the per-GPU max batch (or a dash, never a total),
the fusion delta of the phase on screen with its sign, the EP modes of a
model the estimate says routes no tokens to experts, and the degrees of a
parallel ladder the layout refused apart from the ones never enumerated.
"""
from __future__ import annotations

import copy

from tests.frontend.test_pages_smoke import Session, _fixture, browser_base  # noqa: F401

_LINK = ("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
         "&framework=vllm&selection=latest")


def _estimate(**fields) -> dict:
    est = copy.deepcopy(_fixture("estimate_deepseek_v3_b200.json"))
    est.update(fields)
    return est


def _results_open(s):
    s.page.wait_for_selector("#infer-max-batch", state="attached", timeout=15000)
    s.page.evaluate("document.querySelectorAll('#infer-results details').forEach(d => d.open = true)")


def test_the_cluster_is_the_backends_and_the_estimate_sends_no_gpu_count(browser_base):  # noqa: F811
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/inference/parallel-shape": {"cluster_size": 64, "per_stage_size": 64,
                                      "errors": ["dp_attention must divide the cluster"]},
    })
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    # Asked again for the deployment the link sets, after each edit.
    assert s.wait_for(lambda: any(
        (r["body"] or {}).get("parallel", {}).get("ep_moe") == 8
        for r in s.requests.get("/inference/parallel-shape", [])))
    assert s.wait_for(lambda: "64" in s.page.inner_text("#infer-cluster-size"))
    assert s.page.input_value("#infer-ngpu") == "64"
    assert "dp_attention must divide" in s.page.inner_text("#infer-par-errors")
    body = s.requests["/inference/estimate"][0]["body"]
    assert "num_gpus" not in body, body
    s.assert_clean()


def test_a_missing_per_gpu_max_batch_shows_a_dash_not_a_total(browser_base):  # noqa: F811
    est = _estimate(max_batch_total=4096)
    est.pop("max_decode_batch_per_gpu", None)
    est.pop("max_batch_per_gpu", None)
    browser, base = browser_base
    s = Session(browser, base, overrides={"/inference/estimate": est})
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    _results_open(s)
    assert s.page.inner_text("#infer-max-batch").strip() == "—"
    s.assert_clean()


def test_fusion_delta_follows_the_phase_tab_and_its_sign(browser_base):  # noqa: F811
    est = _estimate(vs_naive_delta={
        "prefill": {"pre_total_s": 0.003, "post_total_s": 0.002, "delta_s": 0.001},
        "decode": {"pre_total_s": 0.0010, "post_total_s": 0.0015, "delta_s": -0.0005},
    })
    browser, base = browser_base
    s = Session(browser, base, overrides={"/inference/estimate": est})
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    _results_open(s)
    s.page.click("#infer-phase-tab-prefill")
    assert s.wait_for(lambda: "fusion saves" in s.page.inner_text("#infer-fusion-delta"))
    assert "1.0 ms" in s.page.inner_text("#infer-fusion-delta")
    s.page.click("#infer-phase-tab-decode")
    assert s.wait_for(lambda: "fusion costs" in s.page.inner_text("#infer-fusion-delta"))
    assert "500" in s.page.inner_text("#infer-fusion-delta")
    s.assert_clean()


def test_ep_modes_follow_the_estimates_moe_answer(browser_base):  # noqa: F811
    """A model the estimate says routes no tokens to experts has no EP axis:
    the parallel modes KV storage offers are TP alone, and the notes of the
    EP ladders — about an axis this model does not have — are not printed."""
    browser, base = browser_base
    s = Session(browser, base, overrides={"/inference/estimate": _estimate(moe_enabled=False)})
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    s.page.wait_for_selector("#infer-scaling-par", timeout=15000)
    assert s.wait_for(lambda: s.page.eval_on_selector_all(
        "#infer-scaling-par option", "els => els.map(e => e.textContent)") == ["TP"])
    s.assert_clean()


def test_refused_degrees_and_the_enumeration_bound_are_different_answers(browser_base):  # noqa: F811
    points = [{"x": x, "feasible": True, "parallel": {"tp_attention": x, "tp_dense_ffn": x,
                                                      "tp_moe_expert": x},
               "warnings": ["n_heads=24 splits 8 ways, not the 16 requested; the "
                            "remaining ranks replicate that tensor rather than divide "
                            "it further."] if x == 16 else [],
               "context": 4096}
              for x in (2, 4, 8, 16)]
    tp = {"moe_enabled": True, "series": [{"gpu_sku": "B200", "points": points}],
          "ladder": {"refused": [{"x": 32, "reason": "mlp.linear_fc1: exact endpoint "
                                  "requires 4304 parts divisible by world degree 32"}],
                     "beyond_cap": {"cap": 512}}}
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/inference/estimate-parallel-sweep": {"by_mode": {"tp_tp": tp}},
    })
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate-parallel-sweep") >= 1)
    # Collapsed by default: one line counts the degrees and names them, the
    # reasons are a click away. The enumeration bound is NOT in there — it
    # is a different answer and stays on its own line.
    note = s.page.wait_for_selector("[data-ladder-notes]", timeout=15000)
    head = s.page.inner_text("[data-ladder-notes] summary")
    assert "1 degree did not hold (TP32)" in head
    assert "1 degree replicates an axis (TP16)" in head
    assert "divisible by world degree 32" not in head
    note.evaluate("e => e.open = true")
    refused = s.page.inner_text('[data-ladder="refused"][data-x="32"]')
    assert "does not hold" in refused and "divisible by world degree 32" in refused
    assert "replicate" in s.page.inner_text('[data-ladder="replicated"][data-x="16"]')
    capped = s.page.inner_text('[data-ladder="beyond-cap"]')
    assert "cap 512" in capped and "does not hold" not in capped
    s.assert_clean()


def test_bound_and_intensity_columns_are_the_backends(browser_base):  # noqa: F811
    """The per-op table's `bound` / `AI` cells and the phase's t_SOL-by-bound
    row show the backend's labels: a leaf its row's winning term and
    intensity, a parent a dash for the bound and its own intensity."""
    from backend.inference.phase_formulas import enrich_module_breakdown
    from backend.inference.row_roofline import RowRoofline, RowTimes
    est = _estimate()
    dec = est["phases"]["decode"]
    timed = []
    for m in dec["flops_by_module"]:
        if m.get("t_op_s") is None:
            continue
        times = RowTimes(m["t_compute_s"] or 0.0, m["t_memory_s"] or 0.0,
                         m["t_comm_s"] or 0.0, m["t_op_s"],
                         m.get("t_launch_s") or 0.0)
        m.update(RowRoofline.labels(times, flops=m["flops_per_rank"] or 0,
                                    access=m["access_bytes_per_rank"] or 0))
        timed.append({"t_compute_s": times.t_compute, "t_memory_s": times.t_memory,
                      "t_comm_s": times.t_comm, "t_launch_s": times.t_launch,
                      "t_op_s": times.t_op})
    enrich_module_breakdown(dec)
    dec["t_sol_by_bound"] = RowRoofline.by_bound(timed)
    dec["t_sol_by_bound_share"] = RowRoofline.shares(dec["t_sol_by_bound"])
    rows = {m["aggregation_id"]: m for m in dec["flops_by_module"]}
    assert {m["bound"] for m in rows.values() if m.get("bound")} >= {"memory"}

    browser, base = browser_base
    s = Session(browser, base, overrides={"/inference/estimate": est})
    s.goto(_LINK)
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    _results_open(s)
    s.page.click("#infer-phase-tab-decode")
    split = s.page.eval_on_selector_all(
        "#infer-phase-table tr[data-metric]",
        "els => els.map(e => e.innerText).find(t => t.includes('t_SOL by bound'))")
    for bound, share in dec["t_sol_by_bound_share"].items():
        pct = "<1%" if 0 < share < 0.01 else f"{share * 100:.0f}%"
        assert f"{bound} " in split and f"({pct})" in split, (bound, split)

    s.page.click("#infer-phase-breakdown [data-module-expand-all]")
    cells = s.page.eval_on_selector_all(
        "#infer-phase-breakdown tr[data-leaf-row], #infer-phase-breakdown tr[data-fold-row]",
        """els => els.map(e => ({
            leaf: e.hasAttribute('data-leaf-row'),
            agg: e.getAttribute('data-aggregation-id'),
            name: e.getAttribute('data-name'),
            bound: e.querySelector('[data-bound]').getAttribute('data-bound'),
            ai: e.querySelector('[data-ai]').getAttribute('data-ai'),
        }))""")
    by_name = {m["name"]: m for m in dec["flops_by_module"]}
    leaves = parents = 0
    for c in cells:
        m = rows[c["agg"]] if c["leaf"] else by_name[c["name"]]
        if c["leaf"]:
            leaves += 1
            assert c["bound"] == (m["bound"] or ""), c
        else:
            parents += 1
            assert c["bound"] == "", c
        want_ai = m.get("arithmetic_intensity")
        if want_ai is None:
            assert c["ai"] == "", c
        else:
            assert float(c["ai"]) == want_ai, c
    assert leaves and parents
    s.assert_clean()


def test_a_parallel_sweep_draws_the_cells_that_landed_while_the_rest_compute(browser_base):  # noqa: F811
    """A cold parallel sweep's 503 carries the server's assembly of the
    cells finished so far: the panel draws it under a small corner note,
    then the whole answer replaces it and the note goes."""
    import json

    def tp(xs):
        return {"moe_enabled": True, "ladder": {"refused": [], "beyond_cap": {"cap": 512}},
                "series": [{"gpu_sku": "B200", "points": [
                    {"x": x, "feasible": True,
                     "parallel": {"tp_attention": x, "tp_dense_ffn": x, "tp_moe_expert": x},
                     "warnings": [], "context": 4096}
                    for x in xs]}]}

    browser, base = browser_base
    s = Session(browser, base)
    calls = []
    release = []

    def sweep(route):
        calls.append(1)
        if not release:
            route.fulfill(status=503, content_type="application/json", body=json.dumps({
                "reason": "still_computing", "retry_after_s": 1,
                "progress": {"done": 2, "total": 4},
                "partial": {"tag": 3, "count": 0, "items": [],
                            "head": {"by_mode": {"tp_tp": tp((2, 4))}}}}))
        else:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps({"by_mode": {"tp_tp": tp((2, 4, 8, 16))}}))

    s.page.route("**/inference/estimate-parallel-sweep", sweep)
    s.goto(_LINK)
    # The panel's one status line carries the server's own count while the
    # cells land; it goes when the whole answer has. What those cells say —
    # here the bound past which nothing was enumerated — is drawn meanwhile.
    s.page.wait_for_selector("[data-panel-status]", timeout=15000)
    assert s.wait_for(lambda: "2 / 4" in s.page.inner_text("[data-panel-status]")), (
        s.page.inner_text("[data-panel-status]"))
    s.page.wait_for_selector('[data-ladder="beyond-cap"]', timeout=15000)
    assert "cap 512" in s.page.inner_text('[data-ladder="beyond-cap"]')
    release.append(1)
    assert s.wait_for(lambda: "2 / 4" not in s.page.inner_text("#infer-scaling"), 15000)
    s.errors = [e for e in s.errors if "503" not in e]
    s.assert_clean()
