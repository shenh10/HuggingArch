"""The model composition (`backend.views.composition`): one row per block
declaration and per weight-storing model-level node, every figure fixed by
the spec and the checkpoint.

The rows are the model's ledgers grouped, so they sum to the model's totals:
params and active params to the projected report's, stored bytes to
`capacity.weights`, KV to `CacheView.growth_and_floor`. Kimi-K3 splits three
declarations into derived groups by history length; they come back as one row
each. DeepSeek-V3's routed experts are 256 copies of a 7168 x 2048 SwiGLU per
MoE layer. Qwen3-Coder-Next's Gated DeltaNet layers hold a recurrent state and
no per-token cache; its full-attention layers grow one. KV is priced at what
the model states: a `cache.entries` dtype where the spec declares one (the
Gated DeltaNet state is kept in fp32), else its stated default precision.
"""
from __future__ import annotations

import re

import pytest

from backend.arch.eval import capacity
from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.arch.eval.projection import project_model
from backend.views.composition import ModelComposition
from tests.snapshots import resolve_snapshot


def _composition(model: str):
    report = project_model(resolve_snapshot(model), batch=1, input_tokens=1)
    return report, ModelComposition(report, capacity.weights(report.analysis)).payload()


@pytest.fixture(scope="module", params=[
    "Qwen/Qwen3-8B", "deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-Coder-Next",
    "moonshotai/Kimi-K3", "openai/gpt-oss-20b",
    # FP8 / FP4 cache rows with per-group scales stored beside each one.
    "deepseek-ai/DeepSeek-V4.1-Flash",
])
def built(request):
    return _composition(request.param)


def test_rows_sum_to_the_model_totals(built):
    report, comp = built
    rows = comp["rows"]
    totals = comp["totals"]
    assert totals["params"] == report.total_params
    assert totals["active_params"] == report.total_active_params
    assert totals["storage_bytes"] == int(capacity.weights(report.analysis).resident)
    assert sum(r["params"]["total"] for r in rows) == totals["params"]
    assert sum(r["active_params"] for r in rows) == totals["active_params"]
    assert sum(r["storage_bytes"] for r in rows) == totals["storage_bytes"]
    for r in rows:
        p = r["params"]
        assert p["attention"] + p["ffn"] + p["experts"] + p["other"] == p["total"]
        assert sum(d["bytes"] for d in r["storage_by_dtype"]) == r["storage_bytes"]
        expected = {
            "attention": p["attention"], "ffn": p["ffn"],
            "experts": p["experts"], "other": p["other"],
            "total": p["total"], "active": r["active_params"],
        }
        for name, result in expected.items():
            formula = r["formulas"].get(name)
            if result:
                assert formula["result"] == result
                symbols = set(re.findall(
                    r"[A-Za-z_][A-Za-z_0-9]*", formula["symbolic"]))
                assert symbols <= {d["name"] for d in formula["defs"]}
            else:
                assert formula is None
        assert r["formulas"]["storage"]["result_bytes"] == r["storage_bytes"]
    # The walk's resolved precision, not `ResolvedSpec.shipped_dtype` — the
    # view used to ask that a second time and answered F32 for a gemma-2
    # cache the walk had already priced at BF16. Asserting against the
    # second route would pin the bug: these models all store 16-bit, so the
    # two agree here and the test could not tell them apart.
    precision = report.analysis.resolved_precision
    assert comp["kv_native_dtype"] == precision.model_dtype
    assert precision.served_as_stored          # every model here stores 16-bit
    assert comp["kv_native_source"] == (precision.model_dtype_source
                                        or "config")
    view = cache_view_from_analysis(report.analysis)
    growth, floor = view.growth_and_floor()
    kv = [r["kv"] for r in rows if r["kv"] is not None]
    assert sum(k["per_token_bytes"] for k in kv) == pytest.approx(growth, rel=1e-12)
    assert sum(k["constant_bytes"] for k in kv) == pytest.approx(floor, rel=1e-12)
    # The per-token column is the slope the budget divides by and the KV
    # formula states — stored bytes, the dynamic scales riding each position
    # included — not the data alone.
    assert totals["kv_per_token_bytes"] == growth == view.growth_formula["result_bytes"]
    assert totals["kv_constant_bytes"] == floor
    assert totals["kv_growth_formula"]["result_bytes"] == growth
    assert totals["kv_constant_formula"]["result_bytes"] == floor
    assert totals["formulas"]["params"]["result"] == totals["params"]
    assert totals["formulas"]["active"]["result"] == totals["active_params"]
    assert totals["formulas"]["storage"]["result_bytes"] == totals["storage_bytes"]
    for row in (r for r in rows if r["kv"] is not None):
        assert row["kv"]["growth_formula"]["result_bytes"] == pytest.approx(
            row["kv"]["per_token_bytes"], rel=1e-12)
        assert row["kv"]["constant_formula"]["result_bytes"] == pytest.approx(
            row["kv"]["constant_bytes"], rel=1e-12)
    length = 1 << 17
    assert growth * length + floor == pytest.approx(
        view.storage_bytes_at(length), rel=1e-12)


def test_kv_per_token_is_the_estimates():
    """The Inspector's per-token KV is the estimate's `per_token_bytes` for a
    cache whose format stores scales beside every position."""
    from backend.inference.capacity import estimate_capacity
    from backend.inference.parallel_config import ParallelConfig
    from tests.snapshots import shared_loader, snapshot_config

    model = "deepseek-ai/DeepSeek-V4.1-Flash"
    _report, comp = _composition(model)
    stem = model.replace("/", "__")
    est = estimate_capacity(
        config=snapshot_config(stem), gpu_sku="H100-SXM",
        resolved=resolve_snapshot(stem, shared_loader()),
        selection_mode="theoretical_only",
        avg_input_tokens=4096, avg_output_tokens=1,
        parallel=ParallelConfig(tp_attention=1))
    assert comp["totals"]["kv_per_token_bytes"] == pytest.approx(
        est["kv_cache"]["per_token_bytes"], rel=1e-12)
    assert comp["totals"]["kv_constant_bytes"] == pytest.approx(
        est["kv_cache"]["constant_bytes"], rel=1e-12)


def test_derived_groups_are_one_row_per_declaration():
    _report, comp = _composition("moonshotai/Kimi-K3")
    backbone = [r for r in comp["rows"] if r["kind"] == "block" and r["aux_kind"] is None]
    declared = [r for r in backbone if "histogram" in r]
    # block_kda_moe, block_mla_moe, block_kda_moe_start, by first layer.
    assert [r["layers"] for r in declared] == [61, 24, 7]
    for r in declared:
        assert sum(h["layers"] for h in r["histogram"]) == r["layers"]
    layers = sorted(i for r in backbone for i in r["layer_indices"])
    assert layers == list(range(93))


def test_routed_experts_are_the_expert_banks():
    _report, comp = _composition("deepseek-ai/DeepSeek-V3")
    moe = [r for r in comp["rows"] if r["kind"] == "block"
           and r["aux_kind"] is None and r["params"]["experts"]]
    assert [r["layers"] for r in moe] == [58]
    # 256 routed experts, each a gate / up / down projection of 7168 x 2048.
    assert moe[0]["params"]["experts"] == 58 * 256 * 3 * 7168 * 2048
    # The FFN that stays: one shared expert of the same shape, and the router.
    assert moe[0]["params"]["ffn"] == 58 * (3 * 7168 * 2048 + 256 * 7168 + 256)
    assert {d["dtype"] for d in moe[0]["storage_by_dtype"]} >= {"F8_E4M3", "scales"}


def test_a_recurrent_state_is_constant_per_sequence():
    _report, comp = _composition("Qwen/Qwen3-Coder-Next")
    blocks = [r for r in comp["rows"] if r["kind"] == "block"]
    linear = [r for r in blocks if r["kv"]["constant_bytes"]]
    full = [r for r in blocks if r["kv"]["per_token_bytes"]]
    assert [r["layers"] for r in linear] == [36]
    assert [r["layers"] for r in full] == [12]
    assert not linear[0]["kv"]["per_token_bytes"]
    assert not full[0]["kv"]["constant_bytes"]


def test_kv_formulas_explain_their_symbols():
    """The Dashboard does not publish an unexplained KV byte total."""
    _report, comp = _composition("Qwen/Qwen3-Coder-Next")
    blocks = [r for r in comp["rows"] if r["kind"] == "block"]
    linear = next(r for r in blocks if r["kv"]["constant_bytes"])
    full = next(r for r in blocks if r["kv"]["per_token_bytes"])

    growth = full["kv"]["growth_formula"]
    assert growth["result_bytes"] == full["kv"]["per_token_bytes"]
    assert {d["name"] for d in growth["defs"]} >= {
        "n_layers", "bpe"}

    constant = linear["kv"]["constant_formula"]
    assert constant["result_bytes"] == linear["kv"]["constant_bytes"]
    assert constant["subterms"]
    assert {d["name"] for d in constant["defs"]} >= {
        "n_layers", "bytes_per_position", "state_positions"}


def test_kv_is_priced_at_the_dtype_the_model_states():
    _report, comp = _composition("Qwen/Qwen3-Coder-Next")
    assert comp["kv_native_dtype"] == "BF16"
    blocks = [r for r in comp["rows"] if r["kind"] == "block"]
    linear, full = blocks
    assert {"dtype": "FP32", "source": "declared"} in linear["kv"]["dtypes"]
    assert full["kv"]["dtypes"] == [{"dtype": "BF16", "source": "native"}]


def test_an_aux_stacks_model_level_nodes_are_its_rows():
    """Qwen3.5's MTP head has model-level nodes of its own (`mtp.fc`, its
    norms): they are the MTP stack's rows, tagged and listed after its
    blocks, not untagged rows beside the main model's head."""
    report, comp = _composition("Qwen/Qwen3.5-35B-A3B")
    program = report.analysis.program
    rows = comp["rows"]
    for r in rows:
        if r["kind"] == "block":
            continue
        name = r["key"].removeprefix("root:")
        stack = program.stack_of(report.model_level_nodes[name].ir_node)
        assert r["stack"] == (stack or None), name
    mtp = [i for i, r in enumerate(rows) if r["stack"] == "mtp"]
    assert any(rows[i]["key"] == "root:mtp.fc" for i in mtp)
    assert all(rows[i]["aux_kind"] == "mtp" for i in mtp)
    assert mtp == list(range(mtp[0], mtp[0] + len(mtp)))


def test_the_kv_dtypes_are_every_dtype_the_cache_is_priced_at():
    """DeepSeek-V4 declares FP8 cache entries beside native BF16 ones: the
    totals name both, not only the native default."""
    _report, comp = _composition("deepseek-ai/DeepSeek-V4-Pro")
    dtypes = {d["dtype"] for d in comp["totals"]["kv_dtypes"]}
    rows = {d["dtype"] for r in comp["rows"] if r["kv"]
            for d in r["kv"]["dtypes"]}
    assert dtypes == rows
    assert {"BF16", "FP8 (E4M3)"} <= dtypes


def test_a_narrowed_checkpoint_credits_the_framework_not_the_file():
    """gemma-2 stores F32 and ships running BF16 (`dtypes.served_dtype`).

    The panel prices its cache at the served width, so the signature beside
    that width has to be the one `precision_effective` gives the same fact —
    `framework`. The tooltip used to read "BF16 · checkpoint": the value the
    deployment runs, credited to the file that does not store it.
    """
    _report, comp = _composition("google/gemma-2-9b")
    assert comp["kv_native_dtype"] == "BF16"
    assert comp["kv_native_source"] == "framework"
