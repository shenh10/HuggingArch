"""Playground module-bench (semantics B) — canonical, same-source, per-L.

Numbers come from the existing single-component eval path, so a module row must
equal a direct `project_block` / cache call (same-source), and different
attention families must produce different numbers under one shared contract.
Snapshots gate: a model whose snapshot is not checked out skips, never falsely
passes (repo convention, see tests/snapshots.py).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from backend.arch.loader import SpecLoader
from backend.arch.eval.projection import project_block
from backend.arch.eval.transforms import DECODE, PREFILL
from backend.playground import module_bench as mb
from backend.arch.eval.precision import AS_SHIPPED
from tests.snapshots import bench_call, resolve_snapshot


@pytest.fixture(scope="module")
def loader() -> SpecLoader:
    return SpecLoader()


def _site(base, name, loader=None):
    """The call a test benches `name` at in `base` (`bench_call`). Naming it
    is the test's choice; the bench never picks."""
    return bench_call(base, name, loader)


def _call(model_id, name, loader):
    """`_site`'s call, by id, for a module_bench entry."""
    return _site(resolve_snapshot(model_id, loader), name, loader).call


# -- the bench report, once per distinct bench ---------------------------
#
# Benching one component resolves the model, walks the isolated component
# for both phases, reads its cache view and prices its rows. Nineteen of
# this file's thirty-eight benches ask for a report another test already
# asked for -- seven alone for DeepSeek-V3.2's DSA attention at L=4096 --
# and the report is a pure function of what `_bench` is given.
#
# So the module keeps each distinct report and the file's tests share it.
# They only read it: nothing here writes to a report it was handed, which
# is what makes one object serve every reader. The memo is this module's
# and dies with it, and the loader is part of the key, so a test holding a
# loader of its own (`test_a_buffer_is_drawn_on_the_node_that_declares_it`
# builds one to inject a buffer) shares nothing with the fixture's.
#
# A test about the BUILDING rather than the report takes `_bench_now`: a
# memo hit would run no build at all, and a test that patches what the
# build may not call would then assert nothing. There is one
# (`test_fused_graph_is_the_priced_runtime_rewrite`, which makes
# `serving_analyze` raise), and it must stay out of the memo whatever other
# tests come to ask for.

_REPORTS: dict = {}


def _bench_now(base_id, name, loader, *, L=2048, ov=None):
    """Build the report, memo aside."""
    base = resolve_snapshot(base_id, loader)  # SkipTest if snapshot absent
    return mb.bench_component(
        base, name, site=_site(base, name), batch=1, input_tokens=L,
        dim_overrides=ov or {}, loader=loader, precision=AS_SHIPPED)


def _bench(base_id, name, loader, *, L=2048, ov=None):
    """The report, built once per distinct bench for this module."""
    key = (base_id, name, loader, L, tuple(sorted((ov or {}).items())))
    if key not in _REPORTS:
        _REPORTS[key] = _bench_now(base_id, name, loader, L=L, ov=ov)
    return _REPORTS[key]


@pytest.fixture(scope="module", autouse=True)
def _forget_reports():
    """Bound the memo to this module: a module-level dict would
    otherwise hold the reports it kept for the rest of the session,
    and one file's worth resident at a time is the point."""
    yield
    _REPORTS.clear()


def _grafted_cast_flops(base_id, name, loader, phase, *, L=2048):
    """The FLOPs of the activation casts serving grafts into the isolated
    component's step, off its rows before fusion: what a step adds to the
    declared block's FLOPs."""
    from backend.inference.materialize_cast import CAST_OP
    base = resolve_snapshot(base_id, loader)
    host = mb.inject_component(base, name, loader)
    spec = mb.isolate_resolved(host, name, {}, _site(base, name))
    step = mb.serving_step(spec, batch=1, input_tokens=L, phase=phase,
                           include_aux_timing=True, precision=AS_SHIPPED)
    return sum(int(row["flops_per_rank"]) for row in step.pre_per_op_timing
               if row["op"] == CAST_OP)


def test_component_provenance_maps_to_binding_models(loader):
    """A swapped-in library component carries the shipped models that bind it
    (provenance), so the comparison is not an anonymous abstraction. The DTO
    surfaces it; a library-only variant no model binds maps to []."""
    inv = mb._component_to_models()
    # DSA attention is bound by DeepSeek-V3.2 (and GLM 5.x).
    assert any("DeepSeek-V3.2" in m for m in inv.get("mla_dsa_attention", []))
    # GQA is bound by many models (Qwen / Llama families).
    assert len(inv.get("gqa_attention", [])) >= 2
    # bench_component carries the same list.
    r = _bench("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader, L=4096)
    assert any("DeepSeek-V3.2" in m for m in r["model_ids"])
    # list_library_components carries provenance for the picker too.
    libs = {c["name"]: c for c in mb.list_library_components()}
    assert "model_ids" in libs.get("gqa_attention", {})


def test_module_bench_can_report_sol_and_measured_effective_time(loader, monkeypatch):
    """The resource summary says how much work; a GPU adds how long.

    Calibration is monkeypatched empty so this test pins the deterministic
    contract, not whichever measured corpus happens to be checked out: SOL is
    positive, while empty measured coverage leaves the measured total null
    rather than silently falling back to SOL. Without a GPU the resource-only
    DTO stays stable.
    """
    base = resolve_snapshot("deepseek-ai/DeepSeek-V3.2", loader)
    monkeypatch.setattr(mb, "_measurement_scope", lambda: None)

    resource = mb.bench_component(
        base, "mla_dsa_attention", site=_site(base, "mla_dsa_attention"), batch=1, input_tokens=512,
        dim_overrides={}, loader=loader, precision=AS_SHIPPED,
    )
    assert "timings" not in resource

    timed = mb.bench_component(
        base, "mla_dsa_attention", site=_site(base, "mla_dsa_attention"), batch=1, input_tokens=512,
        dim_overrides={}, loader=loader, precision=AS_SHIPPED,
        gpu_sku="B200",
    )
    for phase in (PREFILL, DECODE):
        timing = timed["timings"][phase]
        assert timing["gpu_sku"] == "B200"
        assert timing["t_sol_s"] > 0
        assert timing["t_measured_s"] is None
        assert timing["measured_rows"] == 0
        assert timing["total_rows"] > 0
        assert timing["strategy"] == "envelope"
        # The Details table's module tree is Inference's: every priced leaf
        # sits under it once, and its depth-0 rollups are the phase's t_SOL.
        tree = timing["module_rows"]
        assert any(int(r["depth"]) > 0 for r in tree)
        leaf_uids = {r["uid"] for r in timing["op_rows"]}
        drawn = [r["uid"] for r in tree if r["uid"] in leaf_uids]
        assert sorted(drawn) == sorted(leaf_uids)
        top = sum(r["t_op_agg"] for r in tree if int(r["depth"]) == 0)
        assert top == pytest.approx(timing["t_sol_s"])

    with pytest.raises(ValueError, match="unknown gpu_sku"):
        mb.bench_component(
            base, "mla_dsa_attention", site=_site(base, "mla_dsa_attention"), batch=1, input_tokens=512,
            dim_overrides={}, loader=loader, precision=AS_SHIPPED,
            gpu_sku="not-a-gpu",
        )


def _cell_in_substituted(formula, n, suffix=""):
    compact = formula["substituted"].replace(",", "")
    assert f"{int(n)}{suffix}" in compact, (n, formula["substituted"])


def _adds_up(formula, cell, parts=None):
    """The popover's values line ends at the cell and lists the cell's
    parts: largest first, the ones past 20 as one number. `parts` are the
    parts as the payload holds them (a table's rows); without them the
    listed numbers must still add up to the cell."""
    left, _, right = (" " + formula["substituted"]).rpartition(" = ")
    assert int(right.split()[0].replace(",", "")) == int(cell), formula
    if parts is not None:
        parts = [int(p) for p in parts if int(p) > 0]
        assert sum(parts) == int(cell)
        if len(parts) <= 1:
            return
        top = sorted(parts, reverse=True)[:20]
        expected = top + ([sum(parts) - sum(top)] if len(parts) > 20 else [])
    else:
        assert " + " in left, formula
        expected = None
    listed = [int(p.replace(",", "")) for p in left.strip().split(" + ")]
    assert sum(listed) == int(cell), formula
    if expected is not None:
        assert listed == expected, formula


@pytest.mark.parametrize("model_id,name", [
    # an expert bank: one expert's leaves run n_experts times
    ("deepseek-ai/DeepSeek-V3", "moe_ffn_gate_routing_bias"),
])
def test_summary_popovers_add_up_to_their_cells(loader, model_id, name):
    """Every summary popover's parts are the parts of the cell's own number:
    the cell is the runtime table's total, the popover lists its rows — each
    leaf at the copies the walk runs it at, so an expert bank and a repeated
    block count every copy."""
    r = _bench(model_id, name, loader, L=4096)
    rf = r["formulas"]
    prefill = r["op_tables"][PREFILL]["rows"]
    _adds_up(rf["params"], r["params"], [row["params"] for row in prefill])
    _adds_up(rf["active_params"], r["active_params"],
             [row["active_params"] for row in prefill])
    _adds_up(rf["weight_bytes"], r["weight_bytes"])
    for phase in (PREFILL, DECODE):
        pr, rows = r["phases"][phase], r["op_tables"][phase]["rows"]
        _adds_up(pr["formulas"]["flops"], pr["flops"], [x["flops"] for x in rows])
        _adds_up(pr["formulas"]["weight_read"], pr["weight_read_bytes"],
                 [x["weight_read_bytes"] for x in rows])
        _adds_up(pr["formulas"]["act_bytes"], pr["act_bytes_fused"],
                 [x["act_read_bytes"] + x["act_write_bytes"] for x in rows])
    _adds_up(r["phases"][DECODE]["formulas"]["kv_step"],
             r["phases"][DECODE]["kv_read_bytes"],
             [x["kv_read_bytes"] for x in r["op_tables"][DECODE]["rows"]])


def test_a_popover_whose_parts_miss_its_cell_is_withheld():
    """Parts that do not add up to the cell explain nothing: no popover."""
    terms = [{"label": "a", "result": 2}, {"label": "b", "result": 3}]
    assert mb._addends_formula(terms, total=6, unit="B") is None
    assert mb._addends_formula(terms[:1], total=3, unit="B") is None


def test_bench_row_formulas_substitute_the_displayed_numbers(loader):
    """Every summary cell's popover substitutes the exact number the cell
    shows — leaf addends, not a hand-written Σ that could drift. `*` for
    multiplication, never `·`."""
    r = _bench("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader, L=4096)
    rf = r["formulas"]
    assert set(rf) == {"params", "active_params", "weight_bytes", "kv_storage"}
    assert "Σ leaf" not in rf["params"]["symbolic"]
    _cell_in_substituted(rf["params"], r["params"])
    assert "Σ leaf" not in rf["active_params"]["symbolic"]
    _cell_in_substituted(rf["active_params"], r["active_params"])
    # Weight storage names the checkpoint leaves; the ledger split stays
    # in the note so the three-way identity is still readable.
    ws = r["weight_storage"]
    assert "Σ leaf" not in rf["weight_bytes"]["symbolic"]
    _cell_in_substituted(rf["weight_bytes"], r["weight_bytes"], " B")
    assert f"{ws['parameter_bytes']:,d}" in rf["weight_bytes"]["note"]
    assert f"{ws['metadata_bytes']:,d}" in rf["weight_bytes"]["note"]
    assert f"{ws['buffer_bytes']:,d}" in rf["weight_bytes"]["note"]
    # KV storage is the cache walk's per-tensor identities. Several
    # tensors expand as subterms, so the parent formula is their IDs.
    _cell_in_substituted(
        rf["kv_storage"], r["kv"]["storage_bytes_at_L"], " B")
    assert "*" in rf["kv_storage"]["symbolic"] or rf["kv_storage"].get("subterms")

    for phase in (PREFILL, DECODE):
        pr = r["phases"][phase]
        pf = pr["formulas"]
        assert set(pf) >= {"flops", "weight_read", "kv_step", "act_bytes"}
        assert "Σ leaf" not in pf["flops"]["symbolic"]
        assert "*" in pf["flops"]["symbolic"] or pf["flops"].get("subterms")
        _cell_in_substituted(pf["flops"], pr["flops"], " FLOPs")
        assert "Σ leaf" not in pf["weight_read"]["symbolic"]
        _cell_in_substituted(pf["weight_read"], pr["weight_read_bytes"], " B")
        assert "Σ leaf" not in pf["act_bytes"]["symbolic"]
        _cell_in_substituted(pf["act_bytes"], pr["act_bytes_fused"], " B")
    assert r["phases"][PREFILL]["formulas"]["kv_step"]["substituted"] == (
        f"{r['kv']['storage_bytes_at_L']:,d} * 1 = "
        f"{r['phases'][PREFILL]['kv_step_bytes']:,d} B")
    _cell_in_substituted(
        r["phases"][DECODE]["formulas"]["kv_step"],
        r["phases"][DECODE]["kv_read_bytes"], " B")

    blobs = (rf, r["phases"][PREFILL]["formulas"], r["phases"][DECODE]["formulas"])
    for blob in blobs:
        for formula in blob.values():
            for text in formula.values():
                if isinstance(text, str):
                    assert "·" not in text
            subs = formula.get("subterms")
            if subs:
                assert formula["symbolic"] == " + ".join(
                    st.get("label") or st.get("symbolic") for st in subs)


def test_addends_formula_is_the_one_sum_shape():
    """Every named sum — KV tensors, runtime kernels, ledger nodes — is the
    same popover shape: one part keeps its identity; several name the
    parts and expand identities as subterms."""
    terms = [
        {"label": "KV_A_LAYERNORM", "symbolic": "d_kv_lora * s_kv * 2",
         "substituted": "512 * 1024 * 2", "result": 1_048_576},
        {"label": "K_PE_ROPE", "symbolic": "d_qk_rope * s_kv * 2",
         "substituted": "64 * 1024 * 2", "result": 131_072},
        {"label": "INDEXER.KEY_CONCAT", "symbolic": "d_indexer_head * s_kv * 1",
         "substituted": "128 * 1024 * 1", "result": 131_072},
    ]
    f = mb._addends_formula(terms, total=1_310_720, unit="B")
    assert f["symbolic"] == "KV_A_LAYERNORM + K_PE_ROPE + INDEXER.KEY_CONCAT"
    assert f["subterms"][0]["symbolic"] == "d_kv_lora * s_kv * 2"
    assert f["substituted"] == "1,048,576 + 131,072 + 131,072 = 1,310,720 B"
    one = mb._addends_formula(terms[:1], total=1_048_576, unit="B")
    assert one == {"symbolic": "d_kv_lora * s_kv * 2",
                   "substituted": "512 * 1024 * 2 = 1,048,576 B"}
    # A kernel row with no identity of its own still names the addend;
    # nothing expands because there is nothing to expand.
    named = mb._addends_formula(
        [{"label": "q_proj", "result": 10}, {"label": "o_proj", "result": 20}],
        total=30, unit="B",
    )
    assert named["symbolic"] == "o_proj + q_proj"
    assert "subterms" not in named


def test_addends_formula_caps_at_20_and_still_adds_up():
    """Summary addends read large→small and stop at 20; the parts the
    ceiling drops are one number on the values line, so it still adds up."""
    terms = [{"label": f"m{i}", "symbolic": f"2*{i}", "result": i}
             for i in range(1, 26)]
    f = mb._addends_formula(terms, total=325, unit="FLOPs")
    assert f["symbolic"].startswith("m25 + m24 + m23")
    assert "m6" in f["symbolic"]
    assert "m5 +" not in f["symbolic"] and not f["symbolic"].endswith("m5")
    assert f["note"] == "+ 5 smaller terms"
    assert f["substituted"].endswith(" + 15 = 325 FLOPs")
    assert len(f["subterms"]) == 20
    assert f["subterms"][0]["label"] == "m25"
    _adds_up(f, 325, list(range(1, 26)))


def test_timed_phase_rows_carry_the_t_sol_formula(loader, monkeypatch):
    """With a GPU, the t SOL popover mirrors the per-op table's identity —
    Σ_leaf max(t_compute, t_memory, t_comm, t_launch) — substituting t_sol_s itself."""
    base = resolve_snapshot("deepseek-ai/DeepSeek-V3.2", loader)
    monkeypatch.setattr(mb, "_measurement_scope", lambda: None)
    timed = mb.bench_component(
        base, "mla_dsa_attention", site=_site(base, "mla_dsa_attention"), batch=1, input_tokens=512,
        dim_overrides={}, loader=loader, precision=AS_SHIPPED, gpu_sku="B200",
    )
    for phase in (PREFILL, DECODE):
        timing = timed["timings"][phase]
        assert "t_sol_formula" not in timing
        f = timed["phases"][phase]["formulas"]["t_sol"]
        assert f["symbolic"] == "Σ_leaf max(t_compute, t_memory, t_comm, t_launch)"
        from backend.inference.phase_formulas import _fmt_time
        assert f["substituted"].endswith(f"= {_fmt_time(timing['t_sol_s'])}")
        assert "·" not in f["note"]


def test_graph_diff_is_per_entry_and_refers_to_the_rows(loader):
    """Two bindings of one component are two entries: they get a graph diff
    whose classes name nodes of each row's own block_graph rather than
    copying them."""
    resolve_snapshot("deepseek-ai/DeepSeek-V3.2", loader)
    resolve_snapshot("zai-org/GLM-5", loader)
    r = mb.module_bench(base_model_id="deepseek-ai/DeepSeek-V3.2",
                        entries=[{"name": "mla_dsa_attention", "model_id": "deepseek-ai/DeepSeek-V3.2",
                                  "call": _call("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader)},
                                 {"name": "mla_dsa_attention", "model_id": "zai-org/GLM-5",
                                  "call": _call("zai-org/GLM-5", "mla_dsa_attention", loader)}],
                        batch=1, input_tokens=512, precision=AS_SHIPPED)
    gd = r["graph_diff"]
    assert gd["available"] and gd["entries"] == [row["entry"] for row in r["results"]]
    for c in gd["classes"]:
        for row, nid in zip(r["results"], c["members"], strict=True):
            assert nid in {n["id"] for n in row["block_graph"]["nodes"]}


def test_graph_diff_needs_every_entry(loader):
    """One entry is not a diff; an entry that fails refuses it, naming the
    entry — a diff over part of the selection would read as the whole."""
    mla = {"name": "mla_attention", "model_id": None,
           "call": _call("deepseek-ai/DeepSeek-V3", "mla_attention", loader)}
    one = mb.module_bench(base_model_id="deepseek-ai/DeepSeek-V3",
                          entries=[mla],
                          batch=1, input_tokens=512, precision=AS_SHIPPED)
    assert "graph_diff" not in one
    gla = "gla_qknorm_partial_rope_sigmoid_gated_attention"
    out = mb.module_bench(base_model_id="deepseek-ai/DeepSeek-V3",
                          entries=[mla, {"name": gla, "model_id": None}],
                          batch=1, input_tokens=512, precision=AS_SHIPPED)
    assert out["errors"] and len(out["results"]) == 1
    assert out["graph_diff"]["available"] is False
    assert gla in out["graph_diff"]["reason"]


def test_editable_dims_are_the_components_own_knobs(loader):
    """The editable dims table is the component's OWN referenced knobs (allowlist)
    — not the shared-contract dims (d_model/seq/batch) nor the host's ambient dims
    (n_experts / n_vocab / n_layers / dp_attention), which sit in dims_used but are
    not this module's knobs."""
    r = _bench("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader, L=4096)
    ed = r["editable_dims"]
    assert "d_head" in ed or "d_kv_lora" in ed          # real attention knobs
    assert "d_model" not in ed and "s_kv" not in ed      # shared contract, not here
    for ambient in ("n_experts", "n_vocab", "n_layers", "dp_attention"):
        assert ambient not in ed, f"{ambient} is ambient host, not a module knob"


def test_gqa_vs_mha_on_llama_differ(loader):
    """Drop-in swap on one base: MHA carries more KV heads than GQA, so more
    weight and a larger cache; both produce positive canonical numbers."""
    gqa = _bench("meta-llama/Llama-3.2-3B", "gqa_attention", loader)
    mha = _bench("meta-llama/Llama-3.2-3B", "mha_attention", loader)
    for r in (gqa, mha):
        assert r["phases"]["prefill"]["flops"] > 0 and r["weight_bytes"] > 0
        assert r["kv"]["storage_bytes_at_L"] > 0
    assert mha["weight_bytes"] > gqa["weight_bytes"]
    assert mha["kv"]["storage_bytes_at_L"] > gqa["kv"]["storage_bytes_at_L"]
    assert gqa["kv"]["family"] == "GQA"
    assert mha["kv"]["family"] == "MHA"


def test_mla_vs_dsa_on_deepseek(loader):
    """MLA vs DSA on one base. DSA stores all positions but reads only top-K,
    so its read bytes are strictly below its storage bytes; dense MLA reads
    everything it stores."""
    mla = _bench("deepseek-ai/DeepSeek-V3.2", "mla_attention", loader, L=4096)
    dsa = _bench("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader, L=4096)
    assert mla["phases"]["decode"]["kv_read_bytes"] == mla["kv"]["storage_bytes_at_L"]
    assert dsa["phases"]["decode"]["kv_read_bytes"] < dsa["kv"]["storage_bytes_at_L"]
    assert (dsa["kv"]["display"], dsa["kv"]["family"]) == ("DSA", "MLA")
    assert mla["phases"]["prefill"]["flops"] != dsa["phases"]["prefill"]["flops"]


def test_kv_names_the_mechanism_apart_from_the_storage_family(loader):
    """The row names the mechanism by `attn_display` and keeps the storage
    family beside it: KDA's recurrent state is GatedDeltaNet-shaped, but the
    mechanism is KDA. Where the two agree (GDN, MLA) they read the same."""
    kda = _bench("moonshotai/Kimi-K3", "kimi_k3_kda_attention", loader)
    gdn = _bench("Qwen/Qwen3-Coder-Next", "gated_delta_net_fused_attention",
                 loader)
    mla = _bench("deepseek-ai/DeepSeek-V3.2", "mla_attention", loader)
    assert (kda["kv"]["display"], kda["kv"]["family"]) == (
        "KDA", "GatedDeltaNet")
    assert (gdn["kv"]["display"], gdn["kv"]["family"]) == (
        "GatedDeltaNet", "GatedDeltaNet")
    assert (mla["kv"]["display"], mla["kv"]["family"]) == ("MLA", "MLA")


def test_flops_and_weight_are_same_source(loader):
    """bench numbers must equal a direct project_block on the same host — the
    endpoint adds a DTO, never a second cost path."""
    base = resolve_snapshot("deepseek-ai/DeepSeek-V3", loader)
    host = mb.inject_component(base, "mla_attention", loader)
    site = _site(base, "mla_attention")
    direct = project_block(host, "mla_attention", batch=1, input_tokens=2048,
                           site=site)
    r = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader)
    # The step's FLOPs are the block's plus the passes of the activation
    # casts serving grafts before its FP8 GEMMs.
    casts = _grafted_cast_flops("deepseek-ai/DeepSeek-V3", "mla_attention",
                                loader, PREFILL)
    assert casts > 0
    assert r["phases"]["prefill"]["flops"] == direct.total_flops + casts
    # The stored parameters are the block report's; storage adds their scales.
    storage = r["weight_storage"]
    assert storage["parameter_bytes"] == direct.total_storage_bytes
    assert r["weight_bytes"] == (storage["parameter_bytes"]
                                 + storage["metadata_bytes"]
                                 + storage["buffer_bytes"])
    assert r["params"] == direct.total_params
    # The structure view adds up to the same stored copy, scales included.
    assert r["l1"]["available"], r["l1"]["reason"]
    assert r["l1"]["totals"]["weight_bytes"] == r["weight_bytes"]


def test_block_graph_reconciles_with_totals(loader):
    """The block_graph the breakdown/diagram render from is the same cost source
    as the totals: summing its per-op params/flops reconciles with row totals
    (MLA has no absorbed weights). Guards the builder from drifting silently.
    V3 stores MLA in FP8, so the nodes' weight bytes add up to the stored copy
    only when each carries the scales of the weight it holds."""
    r = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader)
    nodes = r["block_graph"]["nodes"]
    assert nodes
    assert sum(int(n.get("params") or 0) for n in nodes) == r["params"]
    assert sum(int(n.get("flops") or 0) for n in nodes) == (
        r["phases"]["prefill"]["flops"]
        - _grafted_cast_flops("deepseek-ai/DeepSeek-V3", "mla_attention",
                              loader, PREFILL))
    assert r["weight_storage"]["metadata_bytes"] > 0
    assert sum(int(n.get("weight_bytes") or 0) for n in nodes) == r["weight_bytes"]


def test_a_buffer_is_drawn_on_the_node_that_declares_it():
    """A buffer is stored and resident, so the entry's `weight_bytes` holds
    it; the node whose op declares it draws those bytes although it has no
    parameters, and the graph and the structure view still add up to the
    entry. No library component declares one today (the corpus's RoPE
    tables are model-level), so the entry is GQA with Llama-2's per-layer
    `inv_freq` table moved inside it, under the module the checkpoint stores
    it in (`self_attn.rotary_emb.inv_freq`)."""
    import yaml
    from pathlib import Path

    from backend.arch.loader import _parse_component_document

    own = SpecLoader()
    raw = yaml.safe_load((Path(mb.__file__).parents[1] / "arch" / "components"
                          / "library" / "gqa_attention.yaml").read_text())
    raw["graph"]["rotary_emb"] = {
        "op": "rotary_inv_freq_buffer", "dim": "d_head // 2",
        "weight_axes": [{"replicated": True}]}
    _name, graph = _parse_component_document(raw, "gqa_attention_inv_freq.yaml")
    own._component_lib["gqa_attention_inv_freq"] = graph
    base = resolve_snapshot("meta-llama/Llama-2-7b-hf", own)
    r = mb.bench_component(base, "gqa_attention_inv_freq", site=_site(base, "gqa_attention_inv_freq", own), batch=1,
                           input_tokens=512, dim_overrides={}, loader=own,
                           precision=AS_SHIPPED)
    storage = r["weight_storage"]
    # 64 entries at the F32 Llama-2 stores this table in.
    assert storage["buffer_bytes"] == 64 * 4
    assert r["weight_bytes"] == storage["parameter_bytes"] + storage["buffer_bytes"]
    nodes = r["block_graph"]["nodes"]
    (table,) = [n for n in nodes if n["op"] == "rotary_inv_freq_buffer"]
    assert (table["params"], table["weight_bytes"]) == (0, storage["buffer_bytes"])
    assert sum(int(n.get("weight_bytes") or 0) for n in nodes) == r["weight_bytes"]
    assert r["l1"]["available"], r["l1"]["reason"]
    assert r["l1"]["totals"]["weight_bytes"] == r["weight_bytes"]


def test_runtime_op_table_folds_single_consumer_parameter_sources(loader):
    """GDN's dt_bias / A_log remain parameter attribution without becoming
    pseudo-kernel rows: each is folded into its lone consumer and named in
    that row's origins, while params / active params still close."""
    r = _bench("Qwen/Qwen3-Coder-Next", "gated_delta_net_fused_attention",
               loader)
    table = r["op_tables"]["prefill"]
    rows = table["rows"]
    assert sum(row["params"] for row in rows) == r["params"]
    assert sum(row["active_params"] for row in rows) == r["active_params"]
    assert not any(row["kernel"] == "parameter_vector" for row in rows)
    folded = {origin for row in rows for origin in row["origins"]}
    assert {"dt_bias", "A_log"} <= folded
    assert abs(sum(row["params_share"] for row in rows) - 1) < 1e-12
    assert abs(sum(row["flops_share"] for row in rows) - 1) < 1e-12
    assert abs(sum(row["bytes_share"] for row in rows) - 1) < 1e-12


def test_runtime_op_table_uses_post_fusion_live_kernels(loader):
    """DSA's score/softmax/apply chain is one flash_dsa runtime row, not three
    unfused rows; its origins explain where the aggregated FLOPs came from."""
    r = _bench("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader, L=4096)
    for phase in (PREFILL, DECODE):
        table = r["op_tables"][phase]
        rows = table["rows"]
        contract = table["contract"]
        assert contract["parallel"] == "single_rank_unsharded"
        assert contract["fusion"] == "shipped"
        assert contract["precision"]["preset"] == "as_shipped"
        assert sum(row["flops"] for row in rows) == r["phases"][phase]["flops"]
        assert sum(row["act_read_bytes"] + row["act_write_bytes"]
                   for row in rows) == r["phases"][phase]["act_bytes_fused"]
        assert sum(row["kv_read_bytes"] for row in rows) == (
            r["phases"][phase]["kv_read_bytes"])
        assert sum(row["weight_read_bytes"] for row in rows) == (
            r["phases"][phase]["weight_read_bytes"])
        assert sum(row["total_bytes"] for row in rows) == table["totals"]["total_bytes"]

        # Both phases run the absorbed form (a sparse MLA prefills
        # absorbed, `transforms.phase_form`): one sparse MLA kernel over the
        # latent.
        flash = next(row for row in rows if row["kernel"] == "flash_dsa")
        assert set(flash["origins"]) == {
            "qk_score", "softmax", "attn_apply_latent"}
        if phase == "prefill":
            assert not {"qk_score", "softmax", "attn_apply_latent"} & {
                row["kernel"] for row in rows}
            assert any(row["kernel"] == "indexer_logits" for row in rows)
            assert any(row["kernel"] == "dsa_topk_mask" for row in rows)


@pytest.mark.parametrize("model_id,name,fused_op", [
    ("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", "flash_dsa"),
    ("meta-llama/Llama-3.2-3B", "gqa_attention", "flash_attention"),
])
def test_fused_graph_is_the_priced_runtime_rewrite(
        loader, monkeypatch, model_id, name, fused_op):
    """The runtime graph family is not a second opinion: its fused kernels are
    the runtime table's rows, drawn off the one serving step that priced
    them (one per phase, no second serving walk), and both graph
    granularities close to the same phase totals the summary shows."""
    served = []
    serving_analyze = mb.serving_analyze

    def counted(*args, **kwargs):
        served.append(kwargs.get("phase"))
        return serving_analyze(*args, **kwargs)

    monkeypatch.setattr(mb, "serving_analyze", counted)
    # Its own build: this test is about what the build calls, and a
    # memoised report would be handed over without building anything.
    r = _bench_now(model_id, name, loader, L=512)
    assert sorted(served) == sorted([PREFILL, DECODE])
    graph = r["fused_graph"]
    fused_l1 = r["fused_l1"]
    assert graph["nodes"] and graph["edges"]
    assert "fused_graph" not in r["op_tables"]["prefill"]
    assert "fused_l1" not in r["op_tables"]["prefill"]

    nodes = []

    def walk(items):
        nodes.extend(items)
        for item in items:
            walk(item.get("children") or [])

    walk(graph["nodes"])
    ids = {node.get("id") for node in nodes}
    assert len(ids) == len(nodes), "fused graph node ids must be unique"
    priced = [node for node in nodes if "access_bytes" in node]
    assert len(priced) == len(r["op_tables"]["prefill"]["rows"])
    assert {int(row["uid"]) for row in r["op_tables"]["prefill"]["rows"]} == {
        int(node["uid"]) for node in priced
    }
    assert any(node.get("style") == "fused" and node.get("fused") == fused_op
               for node in nodes)
    assert sum(int(node["params"]) for node in priced) == r["params"]
    assert sum(int(node["flops"]) for node in priced) == (
        r["phases"]["prefill"]["flops"])
    assert sum(int(node["access_bytes"]) for node in priced) == (
        r["op_tables"]["prefill"]["totals"]["total_bytes"])
    # The serving serializer's whole-model namespace is gone: every root edge
    # names a node in this component graph, including synthesized input edges.
    assert all(edge.get("src") in ids and edge.get("dst") in ids
               for edge in graph["edges"])
    edge_ids = [
        *(edge.get("id") for edge in graph["edges"]),
        *(edge.get("id")
          for node in nodes for edge in node.get("child_edges") or []),
    ]
    assert len(edge_ids) == len(set(edge_ids)), "fused graph edge ids must be unique"

    def walk_edges(items):
        for node in items:
            child_ids = {child.get("id") for child in node.get("children") or []}
            assert all(
                edge.get("src") in child_ids and edge.get("dst") in child_ids
                for edge in node.get("child_edges") or []
            ), f"unclosed child edges at {node.get('id')}"
            walk_edges(node.get("children") or [])

    walk_edges(graph["nodes"])

    # Fused has the same two abstractions as the declared graph: its structure
    # boxes partition the runtime roots and close to the runtime table.
    assert fused_l1["available"], fused_l1["reason"]
    assert fused_l1["weight_semantics"] == "weight_read_step"
    assert set(fused_l1["node_box"]) == {
        node["id"] for node in graph["nodes"]
    }
    totals = r["op_tables"]["prefill"]["totals"]
    assert sum(int(box["flops"] or 0) for box in fused_l1["boxes"]) == (
        totals["flops"])
    assert sum(int(box["params"] or 0) for box in fused_l1["boxes"]) == (
        totals["params"])
    assert sum(int(box["weight_bytes"] or 0) for box in fused_l1["boxes"]) == (
        totals["weight_read_bytes"])


def test_fused_structure_aggregates_runtime_scopes(loader):
    """MoE keeps router / experts / shared experts as serving scopes; Fused
    Structure aggregates each scope's priced kernels without flattening the
    detailed graph, then names the same routing/expert boxes as declared L1."""
    r = _bench("deepseek-ai/DeepSeek-V3", "moe_ffn_gate_routing_bias",
               loader, L=512)
    l1 = r["fused_l1"]
    assert l1["available"], l1["reason"]
    assert l1["weight_semantics"] == "weight_read_step"
    labels = {box["label"] for box in l1["boxes"]}
    assert {"Routing path", "Routed experts", "Parallel branch"} <= labels
    members = {box["label"]: set(box["members"]) for box in l1["boxes"]}
    assert members["Routing path"] == {
        "moe_ffn_gate_routing_bias_site|gate"}
    # The expert-side FP8 cast is placed by its own test below
    # (`test_fused_structure_puts_the_expert_cast_in_the_routed_core`).
    cast = "moe_ffn_gate_routing_bias_site|gate_proj__cast"
    assert members["Routed experts"] - {cast} == {
        "moe_ffn_gate_routing_bias_site|experts",
        "moe_ffn_gate_routing_bias_site|combine__moe_ffn_gate_routing_bias",
        "moe_ffn_gate_routing_bias_site|permute__moe_ffn_gate_routing_bias",
    }
    assert members["Parallel branch"] - {cast} == {
        "moe_ffn_gate_routing_bias_site|shared_experts"}
    assert set().union(*members.values()) == set(l1["node_box"])
    totals = r["op_tables"]["prefill"]["totals"]
    assert sum(int(box["flops"] or 0) for box in l1["boxes"]) == totals["flops"]
    assert sum(int(box["params"] or 0) for box in l1["boxes"]) == totals["params"]
    assert sum(int(box["weight_bytes"] or 0) for box in l1["boxes"]) == (
        totals["weight_read_bytes"])
    # The detailed graph still opens the serializer's semantic scopes rather
    # than replacing them with the local L1 projection.
    roots = {node["id"]: node for node in r["fused_graph"]["nodes"]}
    experts = next(node for node in roots.values() if node.get("op") == "gated_experts")
    assert experts.get("children")
    assert any(child.get("priced_runtime") for child in experts["children"])


def test_fused_structure_puts_the_expert_cast_on_the_tokens_path(loader):
    """The expert GEMM's FP8 copy is a launch of its own where serving
    grafts it (`materialize_casts`), and the structure view boxes it where
    it stands. In an isolated MoE module the tokens are the module's input,
    which no node writes: the copy moves above the dispatch onto it all the
    same and stands in the module's own scope, token-side, once per token —
    as the whole model makes it on the norm's output — read by the dispatch
    and the shared expert alike: the tokens path holds it, not the routed
    experts."""
    r = _bench("deepseek-ai/DeepSeek-V3", "moe_ffn_gate_routing_bias",
               loader, L=512)
    members = {box["label"]: set(box["members"])
               for box in r["fused_l1"]["boxes"]}
    cast = "moe_ffn_gate_routing_bias_site|gate_proj__cast"
    assert cast in members["Tokens path"]
    assert cast not in members["Routed experts"]


def test_fused_graph_refuses_a_runtime_row_with_no_node():
    """A missing priced kernel is a graph/table split, not a cheaper graph."""
    row = {"uid": 7, "params": 1, "active_params": 1, "flops": 1,
           "act_read_bytes": 1, "act_write_bytes": 1,
           "weight_read_bytes": 1, "kv_read_bytes": 1, "total_bytes": 3,
           "arithmetic_intensity": None, "origins": [], "kernel": "linear"}
    with pytest.raises(ValueError, match="rows without a node"):
        mb._stamp_fused_resources(
            [{"uid": 8}], {"rows": [row], "totals": {**row, "total_bytes": 3}})


def test_fused_graph_refuses_duplicate_and_unclosed_pricing():
    """The refusal covers both sides of the join, not only a missing row."""
    row = {"uid": 7, "params": 1, "active_params": 1, "flops": 1,
           "act_read_bytes": 1, "act_write_bytes": 1,
           "weight_read_bytes": 1, "kv_read_bytes": 1, "total_bytes": 3,
           "arithmetic_intensity": None, "origins": [], "kernel": "linear"}
    totals = {**row, "total_bytes": 3}

    with pytest.raises(ValueError, match="runtime table repeats priced uid 7"):
        mb._stamp_fused_resources(
            [{"uid": 7}], {"rows": [row, dict(row)], "totals": totals})

    with pytest.raises(ValueError, match="fused graph has two priced nodes"):
        mb._stamp_fused_resources(
            [{"uid": 7}, {"uid": 7}], {"rows": [row], "totals": totals})

    unclosed = {**totals, "total_bytes": 4}
    with pytest.raises(ValueError, match="does not close on total_bytes"):
        mb._stamp_fused_resources(
            [{"uid": 7}], {"rows": [row], "totals": unclosed})

    with pytest.raises(ValueError, match="live kernels without a node"):
        mb._stamp_fused_resources(
            [{"uid": 7}], {"rows": [row], "totals": totals}, live_uids={7, 8})


def test_fused_graph_refuses_a_bad_serving_scope():
    """A malformed serializer shell or edge is reported, never quietly drawn."""
    with pytest.raises(ValueError, match="exactly one block stack"):
        mb._unwrap_isolated_serving_graph({"nodes": [], "edges": []})

    with pytest.raises(ValueError, match="exactly one component body"):
        mb._unwrap_isolated_serving_graph({
            "nodes": [{"op": "block_stack", "id": "block", "children": []}],
            "edges": [],
        })

    with pytest.raises(ValueError, match="leaves its component scope"):
        mb._validate_scoped_graph({
            "nodes": [{"id": "a"}],
            "edges": [{"id": "a->missing", "src": "a", "dst": "missing"}],
        })


def test_runtime_params_values_bind_the_walk_like_inspector(loader):
    """A linear's params values line is bind_record of the walk — the
    product tree Inspector's node popover shows — not the folded count
    on both sides of `=`."""
    import sympy as sp

    r = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader, L=4096)
    q_b = next(row for row in r["op_tables"]["prefill"]["rows"]
               if row["label"] == "q_b_proj")
    f = q_b["formulas"]["params"]
    assert "d_q_lora" in f["symbolic"]
    left = f["substituted"].split("=")[0].strip().replace(",", "")
    assert "*" in left
    assert left != str(q_b["params"])
    assert sp.sympify(left) == q_b["params"]
    # RMSNorm is one gain vector: a single bound dim, equal to the count.
    # The playground prices the shipped fusions' union, where the q and
    # latent-KV a-norms are one launch (`fused_q_kv_rmsnorm`, 2e9b07932); a
    # fused row's params identity carries each member's own law.
    q_n = next(row for row in r["op_tables"]["prefill"]["rows"]
               if "q_a_layernorm" in (row["origins"] or [row["label"]]))
    nf = q_n["formulas"]["params"]
    if q_n["origins"]:
        nf = next(st for st in nf["subterms"] if st["label"] == "q_a_layernorm")
    assert nf["symbolic"] == "d_q_lora"
    assert str(r["dims_used"]["d_q_lora"]) == nf["substituted"].replace(",", "")


def test_runtime_byte_values_bind_the_traffic_identity(loader):
    """HBM values lines are the traffic Dim with dims bound, products
    kept — Inference's access popover, not the folded byte count."""
    import sympy as sp

    r = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader, L=4096)
    q_b = next(row for row in r["op_tables"]["prefill"]["rows"]
               if row["label"] == "q_b_proj")
    act = next(st for st in q_b["formulas"]["total_bytes"]["subterms"]
               if st.get("label") == "act_read")
    assert "d_q_lora" in (act.get("symbolic") or "")
    left = (act.get("substituted") or "").split("=")[0].strip().replace(",", "")
    assert "*" in left
    assert left != str(q_b["act_read_bytes"])
    # The FP8 operand's width carries its scales (1 + 4/128 bytes), a
    # fraction: compare the value, not sympy's Float against an Integer.
    assert float(sp.sympify(left)) == q_b["act_read_bytes"]
    weight = q_b["formulas"]["weight"]
    wleft = weight["substituted"].split("=")[0].strip().replace(",", "")
    assert "*" in wleft
    assert sp.sympify(wleft) == q_b["weight_read_bytes"]


def test_kv_values_bind_elements_and_the_positions_law(loader):
    """A KV tensor's values line binds elements × s_kv × dtype, not a
    folded element count on the left of `=`."""
    mla = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader, L=4096)
    latent = next(t for t in mla["kv"]["formulas"]["storage"]["subterms"]
                  if t["label"] == "kv_a_layernorm")
    assert "d_kv_lora" in latent["symbolic"]
    assert "s_kv" in latent["symbolic"]
    left = latent["substituted"].split("=")[0].strip()
    assert "*" in left
    # 512 * 4096 * 2 (BF16), not `4194304 * 4096 * 2`
    assert "512" in left.replace(",", "")


def test_runtime_op_table_rows_carry_identities_for_hover(loader):
    """Each kernel's params / FLOPs / HBM cell pops over the walk's identity,
    the same FormulaHover the Inference per-op table uses. A fused row names
    the members it absorbed; HBM is the four traffic addends."""
    r = _bench("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader, L=4096)
    flash = next(row for row in r["op_tables"]["prefill"]["rows"]
                 if row["kernel"] == "flash_dsa")
    flops = flash["formulas"]["flops"]
    named = " ".join([flops["symbolic"], *(
        st.get("label") or "" for st in flops.get("subterms") or [])])
    assert "qk_score" in named and "softmax" in named and "attn_apply" in named
    assert str(flash["flops"]) in flops["substituted"].replace(",", "") or (
        f"{flash['flops']:,d}" in flops["substituted"])
    bytes_f = flash["formulas"]["total_bytes"]
    assert "act_read" in bytes_f["symbolic"] or any(
        st.get("label") == "act_read" for st in bytes_f.get("subterms") or [])
    assert bytes_f["substituted"].endswith(" B") or " B" in bytes_f["substituted"]
    # The HBM addends drill past their labels to the traffic Dim's raw
    # identity — the dims and widths the bytes are made of, not one aggregate
    # number. `int()` reads were what used to throw these away.
    act_read = next(st for st in bytes_f["subterms"]
                    if st.get("label") == "act_read")
    assert act_read["symbolic"] != "act_read"
    assert "*" in act_read["symbolic"] and "B" in act_read["symbolic"]
    linear = next(row for row in r["op_tables"]["prefill"]["rows"]
                  if row["label"] == "o_proj")
    assert linear["formulas"]["params"]["symbolic"]
    assert linear["formulas"]["flops"]["symbolic"]
    weight_f = linear["formulas"]["weight"]
    assert "*" in weight_f["symbolic"]
    # The phase cells sum these terms across kernels and drill through to the
    # same identities (the row's own act / weight split, by `identity=`).
    phase_act = r["phases"]["prefill"]["formulas"]["act_bytes"]
    kernel_sub = next(st for st in phase_act["subterms"]
                      if st.get("label") == flash["label"])
    assert kernel_sub["symbolic"] == "act_read + act_write"
    assert any("*" in (st.get("symbolic") or "")
               for st in kernel_sub.get("subterms") or [])
    phase_weight = r["phases"]["prefill"]["formulas"]["weight_read"]
    weight_sub = next(st for st in phase_weight["subterms"]
                      if st.get("label") == "o_proj")
    assert "*" in weight_sub["symbolic"]


def test_runtime_op_table_saturates_moe_weight_reads(loader):
    """Prefill reaches the whole expert bank, but a one-token decode step reads
    only the active experts' weights — the same canonical serving rule as the
    phase total, without adding a parallel-layout dimension."""
    r = _bench("deepseek-ai/DeepSeek-V3", "moe_ffn_gate_routing_bias",
               loader, L=2048)
    for phase in (PREFILL, DECODE):
        table = r["op_tables"][phase]
        assert sum(row["weight_read_bytes"] for row in table["rows"]) == (
            r["phases"][phase]["weight_read_bytes"])
    assert (r["op_tables"][DECODE]["totals"]["total_bytes"]
            < r["op_tables"][PREFILL]["totals"]["total_bytes"])


def test_runtime_op_table_keeps_packed_banks_unmultiplied(loader):
    """GPT-OSS's packed expert op already carries its own 32-expert bank.

    The runtime row must not multiply that already-banked cost a second time,
    while its borrowed packed biases follow the reader's routing coverage.
    Both phases close against the component report and serving weight read.
    """
    r = _bench("openai/gpt-oss-20b", "moe_ffn_fused_packed_bias", loader,
               L=1024)
    for phase in (PREFILL, DECODE):
        table = r["op_tables"][phase]
        rows = table["rows"]
        assert sum(row["params"] for row in rows) == r["params"]
        assert sum(row["active_params"] for row in rows) == r["active_params"]
        assert sum(row["flops"] for row in rows) == r["phases"][phase]["flops"]
        assert sum(row["act_read_bytes"] + row["act_write_bytes"]
                   for row in rows) == r["phases"][phase]["act_bytes_fused"]
        assert sum(row["kv_read_bytes"] for row in rows) == (
            r["phases"][phase]["kv_read_bytes"])
        assert sum(row["weight_read_bytes"] for row in rows) == (
            r["phases"][phase]["weight_read_bytes"])


@pytest.mark.parametrize("phase", [PREFILL, DECODE])
def test_an_fp8_module_runs_the_launches_its_serving_graph_runs(loader, phase):
    """An FP8 module's runtime table is the serving graph's: the same
    launches — its activation casts among them, standing alone or taken into
    the launch that writes them (`fused_q_kv_rmsnorm_quant`) — at the same
    FLOPs and bytes per layer.

    Checked by a second route: the WHOLE model's serving step, the rows of
    the block the bench isolates read off it (that block's template, under
    the call's path), each divided by the layers that run it. The module's
    input edge is the one difference the isolation makes — a whole model
    copies the normed hidden state outside the call (the norm's launch
    writes it), an isolated call reads it as its input and copies it in a
    launch of its own, once for every projection that reads it — so the
    activation read is left out, and the bench holds exactly one more cast:
    the one whose read is the module's input — `[1, L, d_model]` in BF16 at
    prefill, one token's row at decode.
    Every other launch, and its FLOPs, writes, weight and cache reads, must
    agree."""
    from collections import Counter, defaultdict

    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    from backend.inference.step_eval import evaluate_step

    model_id, name, L = "deepseek-ai/DeepSeek-V3", "mla_attention", 2048
    base = resolve_snapshot(model_id, loader)
    site = _site(base, name)
    bench = _bench(model_id, name, loader, L=L)["op_tables"][phase]["rows"]

    parallel = ParallelConfig()
    whole = analyze(
        base, parallel=parallel, framework_binding=mb._UNION_BINDING,
        batch_size=1, phase=phase, breakdown=False,
        step_eval=evaluate_step(base, parallel, phase=phase, batch=1,
                                input_tokens=L, output_tokens=0,
                                precision=AS_SHIPPED))
    template = whole.analysis.program.blocks[site.invocation].meta.template_name
    served = [
        row for row in whole.per_op_timing
        if row["template_name"] == template
        and whole.program.by_uid(row["uid"]).display_path.startswith(
            f"{site.path}.")
        and (row["flops_per_rank"] or row["access_bytes_per_rank"])]
    assert served

    def per_op(rows, op, facts):
        out = defaultdict(lambda: [0, 0, 0, 0])
        for row in rows:
            for i, value in enumerate(facts(row)):
                out[op(row)][i] += value
        return dict(out)

    def per_layer(row):
        count = int(row["count_per_step"])
        values = (int(row["flops_per_rank"]), int(row["act_out_bytes_per_rank"]),
                  int(row["weight_read_bytes_per_rank"])
                  + int(row["metadata_bytes_per_rank"]),
                  int(row["kv_read_bytes_per_rank"]))
        assert all(v % count == 0 for v in values)
        return [v // count for v in values]

    live = [row for row in bench if row["flops"] or row["total_bytes"]]
    d_model = base.config["hidden_size"]
    entry = [row for row in live if row["kernel"] == "quantise_activation"
             and row["act_read_bytes"]
             == (L if phase == PREFILL else 1) * d_model * 2]
    assert len(entry) == 1, [row["label"] for row in entry]
    live = [row for row in live if row is not entry[0]]
    assert Counter(row["kernel"] for row in live) == Counter(
        row["op"] for row in served)
    if phase == PREFILL:
        # The casts are launches here as they are in the serving graph.
        assert any(row["kernel"] == "quantise_activation" for row in live)
        assert any(row["kernel"] == "fused_q_kv_rmsnorm_quant" for row in live)
    assert per_op(live, lambda r: r["kernel"], lambda r: (
        r["flops"], r["act_write_bytes"], r["weight_read_bytes"],
        r["kv_read_bytes"])) == per_op(served, lambda r: r["op"], per_layer)


def test_runtime_op_table_follows_the_requested_precision(loader):
    """Runtime HBM follows the precision profile: FP8 KV changes both the
    phase total and every runtime row that reads it."""
    from backend.arch.eval.precision import PRESETS
    base = resolve_snapshot("meta-llama/Llama-3.2-3B", loader)
    kwargs = {"batch": 1, "input_tokens": 1024, "dim_overrides": {},
              "loader": loader}
    bf16 = mb.bench_component(base, "gqa_attention", site=_site(base, "gqa_attention"), precision=AS_SHIPPED,
                              **kwargs)
    fp8 = mb.bench_component(base, "gqa_attention", site=_site(base, "gqa_attention"),
                             precision=PRESETS["kv_fp8"], **kwargs)
    assert (bf16["op_tables"][DECODE]["totals"]["total_bytes"]
            > fp8["op_tables"][DECODE]["totals"]["total_bytes"])
    assert fp8["op_tables"][DECODE]["contract"]["precision"]["preset"] == "kv_fp8"


def test_fused_activation_drops_score_matrix_for_dense_attention(loader):
    """The fused view applies the registry kernel fusions (flash_attention),
    so dense attention's materialized score matrix disappears from activation
    bytes — fused << naive. FLOPs are a single view-independent number."""
    r = _bench("meta-llama/Llama-3.2-3B", "gqa_attention", loader, L=4096)
    naive = r["phases"]["prefill"]["act_bytes_naive"]
    fused = r["phases"]["prefill"]["act_bytes_fused"]
    assert 0 < fused < naive
    assert fused < naive * 0.5  # score elision is a large saving, not a rounding


def test_recurrent_state_elides_no_score_matrix(loader):
    """A linear-attention layer has no score matrix to elide: what its fused
    view saves is per-token intermediates (the short conv's SiLU, the gated
    norm's gate), so the saving grows with L, not L² as a score matrix's
    does — the fused view shows no fake score elision. Affine, not
    proportional: a fused gate prep also elides the per-step `-exp(A_log)`
    over the heads, whatever L is (f20cad462)."""
    def saving(L):
        prefill = _bench("Qwen/Qwen3-Coder-Next", "gated_delta_net_fused_attention",
                         loader, L=L)["phases"]["prefill"]
        assert 0 < prefill["act_bytes_fused"] <= prefill["act_bytes_naive"]
        return prefill["act_bytes_naive"] - prefill["act_bytes_fused"]
    assert saving(8192) - saving(4096) == 2 * (saving(4096) - saving(2048)) > 0


def test_kv_grows_with_l_for_dense(loader):
    """Dense (MLA) cache storage scales with the sequence length in the
    contract — the number is per-L, not a fixed model total."""
    kv = {}
    for L in (1024, 2048, 4096):
        r = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader, L=L)
        kv[L] = r["kv"]["storage_bytes_at_L"]
    assert kv[1024] < kv[2048] < kv[4096]


def test_missing_dim_surfaces_as_gap_not_crash(loader):
    """A foreign component whose dims the base lacks (MLA on a GQA-only Llama)
    raises a contract error at bench_component, which module_bench catches into
    a per-component error entry rather than failing the whole request."""
    base = resolve_snapshot("meta-llama/Llama-3.2-3B", loader)
    with pytest.raises(mb.CONTRACT_ERRORS):
        mb.bench_component(base, "mla_attention", site=_site(base, "mla_attention"), batch=1, input_tokens=2048,
                           dim_overrides={}, loader=loader,
                           precision=AS_SHIPPED)


def test_recurrent_family_is_not_silently_zero(loader):
    """A linear-attention (recurrent-state) component has no cached node, so the
    isolated cache walk emits nothing — but its declared `state:` entries are
    constant per-request states that serving DOES bill. It must show up as a
    non-zero constant segment, cache_len-independent, or an iso-hidden row would
    mislead by putting 0 next to dense KV."""
    def gdn(L):
        return mb.bench_component(
            resolve_snapshot("Qwen/Qwen3-Coder-Next", loader),
            "gated_delta_net_fused_attention", batch=1, input_tokens=L,
            dim_overrides={}, loader=loader, precision=AS_SHIPPED)
    small = gdn(1024)
    big = gdn(1 << 16)
    assert small["kv"]["storage_bytes_at_L"] > 0
    # recurrent state does not grow with the sequence length
    assert small["kv"]["storage_bytes_at_L"] == big["kv"]["storage_bytes_at_L"]
    laws = {n["law"] for n in small["kv"]["nodes"]}
    assert "constant" in laws
    rec = [n for n in small["kv"]["nodes"] if n["name"] == "recurrent_state"]
    assert rec and rec[0]["elements_per_request"] > 0
    # per-token is None (not 0) so the UI shows "<n>/req", not a misleading "0"
    assert rec[0]["elements_per_token"] is None


def test_storage_law_distinguishes_dense_from_recurrent(loader):
    """`law` separates growing dense KV from a constant recurrent state — a
    bare `growing` bool could not, so cross-family rows stay comparable."""
    mla = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader)
    assert all(n["law"] == "growing" for n in mla["kv"]["nodes"])


def test_a_window_of_any_size_reads_as_windowed():
    """No probe length decides it: a window wider than any length a probe
    would pick is still a window, and a pool that grows by a fraction of a
    position per token still grows."""
    import sympy as sp
    from types import SimpleNamespace
    from backend.arch.eval.values import CACHE_LEN

    def tensor(law):
        return SimpleNamespace(
            name="t", stored=law, storage_law=law, is_growing=CACHE_LEN in law.free_symbols,
            positions=lambda n: int(law.subs(CACHE_LEN, n)))

    wide = 1 << 21
    assert mb._storage_law(tensor(sp.Min(CACHE_LEN, wide))) == {
        "law": "windowed", "window": wide}
    assert mb._storage_law(tensor(sp.ceiling(CACHE_LEN / 4))) == {
        "law": "growing"}
    assert mb._storage_law(tensor(sp.Integer(1))) == {"law": "constant"}


def test_swa_law_is_windowed_with_window(loader):
    """SWA saturates at a fixed window: the law is read off the storage
    expression (`cache_view.Rate.of`), so a window of 4096 reads as
    `windowed` (window=4096), not `growing` (the r2 bug). Per-node
    elements_per_token is populated for the growing/windowed nodes."""
    swa = _bench("google/gemma-2-9b", "swa_attention", loader, L=8192)
    cached = [n for n in swa["kv"]["nodes"] if n["name"] != "recurrent_state"]
    assert cached
    assert all(n["law"] == "windowed" for n in cached)
    assert all(n.get("window") == 4096 for n in cached)
    assert all(n["elements_per_token"] for n in cached)


def test_dims_used_surfaces_internal_config(loader):
    """The response carries the resolved internal dims the component used, so
    the UI can show internal config beside params (drop-in, not a ranking)."""
    mla = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader)
    assert mla["dims_used"]  # non-empty
    assert all(isinstance(v, (int, float)) for v in mla["dims_used"].values())
    # The phase's live pairs are the dense triangle the bench's step scores.
    assert mla["dims_used"]["s_pairs"] > 0
    assert mla["kv"]["unit"].startswith("bytes/")


def test_dims_used_is_what_the_component_depends_on(loader):
    """`dims_used` is the component's own evaluation-dependency set (plus
    what its L1 box formulas bind), not the host model's whole environment:
    the MoE card on DeepSeek-V3 does not carry the host's layer count, vocab
    or the attention's latent width."""
    moe = _bench("deepseek-ai/DeepSeek-V3", "moe_ffn_gate_routing_bias", loader)
    used = moe["dims_used"]
    for host_dim in ("n_layers", "n_vocab", "n_mtp_layers", "d_kv_lora", "d_q_lora"):
        assert host_dim not in used, host_dim
    assert {"d_model", "n_experts_active", "n_route_targets"} <= set(used)
    # "editable" stays the narrower set: a dependency, but not this module's knob
    assert "d_model" not in moe["editable_dims"]


def test_list_library_components_groups_by_role():
    """The picker's source of truth: every listed component has a role (only
    top-level swappable slots), and the attention family is present."""
    items = mb.list_library_components()
    assert items and all(i["role"] for i in items)
    names = {i["name"] for i in items}
    assert {"mla_attention", "gqa_attention", "mla_dsa_attention"} <= names
    roles = {i["role"] for i in items}
    assert "attention" in roles


def test_the_playground_leaves_encoder_calls_out():
    """An encoder call is sized by its media, which a module compare does
    not give, so the playground compares the language model's calls only.
    A component the models call only in a vision / audio tower is not in
    the picker (Kimi-K2.5's vision block, Step-3.7's perception encoder,
    MiMo-V2.5's audio encoder); one called in the backbone too stays, and
    its encoder calls are no bindings rows (MiMo-V2.5 calls `swiglu_ffn` in
    its audio encoder as well as in the backbone and MTP)."""
    names = {i["name"] for i in mb.list_library_components()}
    assert not names & {"kimi_k25_vision_block", "perception_encoder_attention",
                        "mimo_v2_5_audio_encoder", "qwen35_vision_tower"}
    assert "swiglu_ffn" in names
    resolved = resolve_snapshot("XiaomiMiMo/MiMo-V2.5", mb.get_default_loader())
    called = [site for site in mb.CallSite.calls(resolved)
              if mb.binding_of(resolved.spec, site.component, "swiglu_ffn")]
    assert any(site.encoder == "audio" for site in called)
    compared = [site for site in mb._compared_calls(resolved)
                if mb.binding_of(resolved.spec, site.component, "swiglu_ffn")]
    assert compared and all(site.encoder is None for site in compared)


def test_list_library_components_carries_declared_family():
    """Each component reports its declared family (attn_display: attn_variant
    else kv_label) so the picker groups by category, not a name prefix. DSA is
    split out from MLA by variant; SWA/GQA come from kv_label."""
    fam = {i["name"]: i.get("attn_display") for i in mb.list_library_components()}
    assert fam["mla_attention"] == "MLA"
    assert fam["mla_dsa_attention"] == "DSA"      # attn_variant wins over kv_label
    assert fam["gqa_attention"] == "GQA"
    assert fam["swa_attention"] == "SWA"
    assert fam["dsv4_csa_attention"] == "CSA"


def test_module_bench_reports_partial_results(loader, monkeypatch):
    """End-to-end module_bench with a snapshot base (fetch_config/metadata/
    loader monkeypatched to stay offline): good components return rows, the
    foreign one lands in errors[]."""
    base_id = "meta-llama/Llama-3.2-3B"
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config",
                        lambda mid, overlay_dir=None: snapshot_config(base_id))
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    out = mb.module_bench(base_model_id=base_id,
                          components=["gqa_attention", "mla_attention"],
                          batch=1, input_tokens=2048, precision=AS_SHIPPED)
    ok = {r["component"] for r in out["results"]}
    bad = {e["component"] for e in out["errors"]}
    assert "gqa_attention" in ok
    assert "mla_attention" in bad


def _offline_bench(loader, monkeypatch, base_id, components, ov=None):
    """module_bench wired to git snapshots only: fetch_config dispatches per
    model id so binding models also resolve offline."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    base = resolve_snapshot(base_id, loader)
    entries = [{"name": name, "model_id": None,
                "call": _site(base, name, loader).call} for name in components]
    return mb.module_bench(base_model_id=base_id, entries=entries,
                           batch=1, input_tokens=2048, dim_overrides=ov,
                           precision=AS_SHIPPED)


def test_missing_dims_named_for_symbolic_failure(loader, monkeypatch):
    """MHA on an MLA base (no `d_head` binding) used to leak sympy's bare
    'Cannot convert symbols to int'. The diagnosis names the missing dim —
    facts only, NO suggested values: repair is binding to a model that
    parameterizes the component, never patching in a guess."""
    out = _offline_bench(loader, monkeypatch, "deepseek-ai/DeepSeek-V3.2",
                         ["mha_attention"])
    (err,) = out["errors"]
    assert err["component"] == "mha_attention"
    assert err["missing_dims"] == ["d_head"]
    assert "d_head" in err["error"]
    assert "Cannot convert symbols" not in err["error"]
    assert "suggestions" not in err and "suggestion_sources" not in err


def test_missing_dims_all_reported_in_one_round(loader, monkeypatch):
    """KDA on a DeepSeek base needs the whole `linear_attn_config` dim set;
    every missing dim is reported in one round (not one KeyError per rerun)."""
    out = _offline_bench(loader, monkeypatch, "deepseek-ai/DeepSeek-V3.2",
                         ["kimi_k3_kda_attention"])
    (err,) = out["errors"]
    assert err["component"] == "kimi_k3_kda_attention"
    assert {"d_lin_k_head", "d_lin_v_head", "n_lin_k_heads",
            "n_lin_v_heads", "s_conv_kernel",
            "n_gate_lower_bound"} <= set(err["missing_dims"])


def test_explicit_dims_repair_the_gap(loader, monkeypatch):
    """The repair path is the caller's OWN explicit dims (a customization
    derived from a real model), never a backend-suggested guess."""
    out = _offline_bench(
        loader, monkeypatch, "deepseek-ai/DeepSeek-V3.2",
        ["mha_attention", "kimi_k3_kda_attention"],
        ov={"d_head": 56, "d_gate_lowrank": 128, "d_lin_k_head": 128,
            "d_lin_v_head": 128, "n_lin_k_heads": 96, "n_lin_v_heads": 96,
            "s_conv_kernel": 4, "s_conv_history": 4,
            "n_gate_lower_bound": 1})
    assert out["errors"] == []
    assert {r["component"] for r in out["results"]} == {
        "mha_attention", "kimi_k3_kda_attention"}


def test_error_entries_carry_structured_fields(loader, monkeypatch):
    """Contract: every error entry exposes missing_dims (empty when nothing is
    diagnosable) and NO guesses; successful runs keep errors empty."""
    out = _offline_bench(loader, monkeypatch, "meta-llama/Llama-3.2-3B",
                         ["gqa_attention", "mla_attention"])
    assert {r["component"] for r in out["results"]} == {"gqa_attention"}
    (err,) = out["errors"]
    assert err["component"] == "mla_attention"
    assert "d_kv_lora" in err["missing_dims"]
    assert "suggestions" not in err and "suggestion_sources" not in err


def _offline_bench_entries(loader, monkeypatch, base_id, entries, **contract):
    """Entries-form module_bench, offline via git snapshots (bindings resolve
    per model id; unsnapshotted models fail soft)."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    monkeypatch.setattr(mb, "_bindings_cache", {})
    return mb.module_bench(base_model_id=base_id, entries=entries,
                           batch=1, input_tokens=2048, precision=AS_SHIPPED,
                           **contract)


def test_entries_same_component_two_bindings(loader, monkeypatch):
    """The component × bound-model comparison: mla_dsa_attention as DeepSeek-V3.2
    and as GLM-5.1 are TWO entries of one component — distinct entry keys and
    provenance on each row, dims seeded from each binding."""
    calls = {mid: _call(mid, "mla_dsa_attention", loader)
             for mid in ("deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-5.1")}
    out = _offline_bench_entries(loader, monkeypatch, "deepseek-ai/DeepSeek-V3.2", [
        {"name": "mla_dsa_attention", "model_id": mid, "call": call}
        for mid, call in calls.items()
    ])
    assert out["errors"] == []
    assert len(out["results"]) == 2
    entries = [r["entry"] for r in out["results"]]
    assert entries == [f"mla_dsa_attention@{mid}#{call}"
                       for mid, call in calls.items()]
    assert [r["model_id"] for r in out["results"]] == [
        "deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-5.1"]
    # inline seed: each entry carries its binding's dims in the applied echo
    applied = out["contract"]["per_component_dims"]
    assert set(applied) == set(entries)
    assert applied[entries[0]] and applied[entries[1]]


def test_bound_moe_entry_runs_its_models_routing(loader, monkeypatch):
    """`moe_ffn @ DeepSeek-V2-Lite` on a DeepSeek-V3.2 base routes as V2-Lite
    does: 64 experts, top-6 — not the base's 256 / top-8. The router and the
    permute read the routing dims, not the component's own nodes, so the seed
    has to follow what the evaluation depends on. With the shared d_model set
    to V2-Lite's 2048 the row is V2-Lite's own per-layer MoE: the same FLOPs
    and params as the moe_ffn node of the whole-model analysis."""
    from backend.arch.eval.projection import project_model
    v2_lite = "deepseek-ai/DeepSeek-V2-Lite"
    out = _offline_bench_entries(loader, monkeypatch, "deepseek-ai/DeepSeek-V3.2",
                                 [{"name": "moe_ffn", "model_id": v2_lite}],
                                 d_model=2048)
    assert out["errors"] == []
    (row,) = out["results"]
    seeded = out["contract"]["per_component_dims"][row["entry"]]
    assert seeded["n_experts_active"] == 6
    assert seeded["n_expert_copies"] == 64
    assert seeded["n_route_targets"] == 64
    assert "n_router_classes" not in seeded

    def moe_nodes(nodes):
        for node in nodes:
            if node.op == "moe_ffn":
                yield node
            yield from moe_nodes(node.children or [])

    model = project_model(resolve_snapshot(v2_lite, loader),
                          batch=1, input_tokens=2048)
    own = [n for _t, _count, block in model.block_results
           for n in moe_nodes(block.nodes)]
    assert own
    for node in own:
        flops, params, active = node.flops, node.params, node.active_params
        assert row["phases"]["prefill"]["flops"] == flops
        assert row["params"] == params
        assert row["active_params"] == active


def test_dense_entry_seeds_only_its_own_dims(loader, monkeypatch):
    """A dense component's seed is its knobs as the bound model sets them and
    nothing of the host's: Llama-3.2-3B's attention and FFN sizes."""
    out = _offline_bench_entries(loader, monkeypatch, "deepseek-ai/DeepSeek-V3.2", [
        {"name": "gqa_attention", "model_id": "meta-llama/Llama-3.2-3B"},
        {"name": "swiglu_ffn", "model_id": "meta-llama/Llama-3.2-3B"},
    ])
    assert out["errors"] == []
    applied = out["contract"]["per_component_dims"]
    assert applied["gqa_attention@meta-llama/Llama-3.2-3B"] == {
        "d_head": 128, "n_heads": 24, "n_kv_heads": 8}
    assert applied["swiglu_ffn@meta-llama/Llama-3.2-3B"] == {"d_ffn": 8192}


def test_entries_explicit_dims_win_and_seeded_sticks(loader, monkeypatch):
    """An explicit entry dim beats the binding seed; once the client marks the
    entry seeded, un-set dims are NOT re-seeded (a user-deleted dim stays
    deleted instead of silently resurrecting from the model binding)."""
    base = "deepseek-ai/DeepSeek-V3.2"
    call = _call("deepseek-ai/DeepSeek-V3", "mla_attention", loader)
    fresh = _offline_bench_entries(loader, monkeypatch, base, [
        {"name": "mla_attention", "model_id": "deepseek-ai/DeepSeek-V3",
         "call": call, "dims": {"d_kv_lora": 256}}])
    key = f"mla_attention@deepseek-ai/DeepSeek-V3#{call}"
    applied = fresh["contract"]["per_component_dims"][key]
    assert applied["d_kv_lora"] == 256          # explicit beats the V3 seed (512)
    assert len(applied) > 1                     # the rest was seeded from V3
    seed = fresh["contract"]["seed_dims"][key]
    assert seed["d_kv_lora"] == 512             # the baseline the edit is against
    assert {k: v for k, v in applied.items() if seed.get(k) != v} == {"d_kv_lora": 256}

    sticky = _offline_bench_entries(loader, monkeypatch, base, [
        {"name": "mla_attention", "model_id": "deepseek-ai/DeepSeek-V3",
         "call": call, "dims": {"d_kv_lora": 256}, "seeded": True}])
    applied2 = sticky["contract"]["per_component_dims"][key]
    assert applied2 == {"d_kv_lora": 256}       # seeded: no silent re-seed
    assert key not in sticky["contract"]["seed_dims"]


def test_component_bindings_table(loader, monkeypatch):
    """The single-pick bindings table: every shipped model binding the component
    with the dims each resolves to; unsnapshotted bindings (GLM-5.2 offline)
    fail soft and are skipped, never failing the table."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    monkeypatch.setattr(mb, "_bindings_cache", {})
    table = mb.component_bindings("mla_dsa_attention")
    by_model = {b["model_id"]: b["dims"] for b in table["bindings"]}
    assert "deepseek-ai/DeepSeek-V3.2" in by_model
    assert "zai-org/GLM-5.1" in by_model
    assert by_model["deepseek-ai/DeepSeek-V3.2"].get("d_indexer_head")
    # a binding that cannot resolve is REPORTED with its reason, never silently
    # dropped — an all-failing table must not read as "library-only"
    monkeypatch.setattr(mb, "_component_to_models",
                        lambda: {"ghost_attention": ["not-a-real/model"]})
    monkeypatch.setattr(mb, "_bindings_cache", {})
    ghost = mb.component_bindings("ghost_attention")
    assert ghost["bindings"] == []
    assert "not-a-real/model" in ghost["failed"] and ghost["failed"]["not-a-real/model"]


def test_every_call_is_a_bindings_row_the_expert_call_included(loader, monkeypatch):
    """DeepSeek-V3 calls `swiglu_ffn` three times: as the dense layers' MLP
    and as the shared expert of the MoE layers and of the MTP layer. Every
    call is a row, each saying the model makes three, in two
    parameterizations — the shared expert reads its own width. Nemotron-3-
    Nano's routed bank is the `relu2_experts` template, whose one call is the
    expert bank: it evaluates alone, its tokens the dispatch's `[E, M, d]`
    rows, and it reads the routing counts its rows are sized by."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    monkeypatch.setattr(mb, "_bindings_cache", {})
    v3 = "deepseek-ai/DeepSeek-V3"
    nano = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    resolve_snapshot(nano, loader)
    monkeypatch.setattr(mb, "_component_to_models", lambda: {
        "swiglu_ffn": [v3], "relu2_experts": [nano]})
    rows = mb.component_bindings("swiglu_ffn")["bindings"]
    by_path = {r["call"].split("/", 1)[1]: r for r in rows}
    assert set(by_path) == {"mlp", "mlp.shared_experts",
                            "transformer.mlp.shared_experts"}
    assert {r["model_calls"] for r in rows} == {3}
    assert {r["model_parameterizations"] for r in rows} == {2}
    assert sorted(r["parameterization"] for r in rows) == [0, 1, 1]
    (expert,) = mb.component_bindings("relu2_experts")["bindings"]
    assert expert["call"].split("/", 1)[1] == "mixer.experts"
    assert "error" not in expert, expert.get("error")
    assert {"n_experts_active", "n_route_targets", "n_expert_copies"} <= set(
        expert["dims"])


def test_calls_that_evaluate_alike_are_one_parameterization(loader, monkeypatch):
    """Kimi-K3 calls its KDA attention at many layer groups, each rebinding
    the AttnRes history `Sar` — which attention never reads. The calls hand
    over the same tensor at the same address, so they evaluate alike: one
    parameterization, which a client may take without asking for a call."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    monkeypatch.setattr(mb, "_bindings_cache", {})
    k3 = "moonshotai/Kimi-K3"
    resolve_snapshot(k3, loader)
    monkeypatch.setattr(mb, "_component_to_models",
                        lambda: {"kimi_k3_kda_attention": [k3]})
    rows = mb.component_bindings("kimi_k3_kda_attention")["bindings"]
    assert len(rows) > 1
    assert {r["model_calls"] for r in rows} == {len(rows)}
    assert {r["model_parameterizations"] for r in rows} == {1}
    assert {r["parameterization"] for r in rows} == {0}
    assert len({tuple(sorted(r["dims"].items())) for r in rows}) == 1


def test_calls_at_different_addresses_stored_alike_are_one_parameterization(
        loader, monkeypatch):
    """LongCat-Flash calls its MLA twice in every backbone layer
    (`self_attn.0`, `self_attn.1`) and once in the MTP layer: three
    checkpoint addresses, one form at one set of dims, its weights stored
    in one format at all three. The address names the copy and prices
    nothing, so the calls are one parameterization — a client takes one
    without asking which call (Regression: the address was in the key,
    which split them and left the playground's column at no call)."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    monkeypatch.setattr(mb, "_bindings_cache", {})
    longcat = "meituan-longcat/LongCat-Flash-Thinking-2601"
    monkeypatch.setattr(mb, "_component_to_models",
                        lambda: {"mla_attention": [longcat]})
    rows = mb.component_bindings("mla_attention")["bindings"]
    assert {r["call"].split("/", 1)[1] for r in rows} == {
        "self_attn.0", "self_attn.1", "self_attn"}
    assert {r["model_parameterizations"] for r in rows} == {1}


def test_calls_alike_by_value_and_split_by_stored_format(loader, monkeypatch):
    """The key compares what the walk sees, by value. MiMo-7B's MTP layer
    calls its MLP with `d_ffn: d_ffn, d_model: d_model` — rebinding nothing
    — so it is the backbone's parameterization. Hy4's MTP layer stores its
    attention's `learnable_sink_param` in BF16 where the backbone stores F32
    (the checkpoint's own dtypes): the same form at the same dims, priced
    differently, so two parameterizations."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    monkeypatch.setattr(mb, "_bindings_cache", {})
    mimo, hy4 = "XiaomiMiMo/MiMo-7B-RL", "tencent/Hy4-preview"
    monkeypatch.setattr(mb, "_component_to_models", lambda: {
        "swiglu_ffn": [mimo], "mla_dsa_gated_sink_attention": [hy4]})
    rows = mb.component_bindings("swiglu_ffn")["bindings"]
    assert {r["model_calls"] for r in rows} == {2}
    assert {r["model_parameterizations"] for r in rows} == {1}
    rows = mb.component_bindings("mla_dsa_gated_sink_attention")["bindings"]
    by_stack = {r["label"].split(" ")[0]: r["parameterization"] for r in rows}
    assert {r["model_parameterizations"] for r in rows} == {2}
    assert by_stack["backbone"] != by_stack["mtp"]


def test_a_call_in_another_stack_that_evaluates_alike_needs_no_name(loader, monkeypatch):
    """DeepSeek-V3 calls its MLA in the dense layers, the MoE layers and the
    MTP layer. The MTP call sits in another stack, but the stack declares no
    stream attention reads, so all three evaluate alike: one
    parameterization, and an entry naming no call benches at the first —
    from the bench itself, whichever way the column was added (a rebind, a
    link). Calls that differ (`swiglu_ffn` as DeepSeek-V3's dense MLP and
    as its shared expert) are still refused until one is named."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    monkeypatch.setattr(mb, "_bindings_cache", {})
    v3 = "deepseek-ai/DeepSeek-V3"
    monkeypatch.setattr(mb, "_component_to_models", lambda: {
        "mla_attention": [v3], "swiglu_ffn": [v3]})
    rows = mb.component_bindings("mla_attention")["bindings"]
    assert {r["model_calls"] for r in rows} == {3}
    assert any(r["call"].startswith("2/") for r in rows)   # the MTP call
    assert {r["model_parameterizations"] for r in rows} == {1}

    out = mb.module_bench(entries=[
        {"name": "mla_attention", "model_id": v3},
        {"name": "swiglu_ffn", "model_id": v3}],
        batch=1, input_tokens=2048, precision=AS_SHIPPED)
    assert [r["entry"] for r in out["results"]] == [f"mla_attention@{v3}"]
    first = mb.module_bench(entries=[
        {"name": "mla_attention", "model_id": v3, "call": rows[0]["call"]}],
        batch=1, input_tokens=2048, precision=AS_SHIPPED)["results"][0]
    assert out["results"][0]["phases"] == first["phases"]
    (refused,) = out["errors"]
    assert refused["entry"] == f"swiglu_ffn@{v3}"
    assert "Pick the one this column compares" in refused["error"]


def test_variant_bound_component_fully_repairable(loader, monkeypatch):
    """dsv4_csa_attention's compress_ratio lives ONLY in deepseek_v4's per-layer
    recipes (the yaml matches by model_type, dims sit in the variant map). The
    playground must still: (a) credit V4-Pro as a binding/provenance, (b) seed
    a bound entry inline with the REAL per-layer dims (bench succeeds), (c)
    report only REAL gaps for an unbound pick — call-site-bound subcomponent
    params (Cape/Cdim_io/…) and unused siblings' dims (d_ffn) are phantom
    noise — facts only, no guesses; repair is explicit dims or a binding."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    monkeypatch.setattr(mb, "_bindings_cache", {})

    # (a) provenance via the resolved snapshot spec
    assert "deepseek-ai/DeepSeek-V4-Pro" in mb._component_to_models()["dsv4_csa_attention"]

    # (b) bound entry benches via the inline variant-dims seed from the
    # RESOLVED spec's assignments (compress_ratio=4 included, honestly)
    v4 = _call("deepseek-ai/DeepSeek-V4-Pro", "dsv4_csa_attention", loader)
    bound = mb.module_bench(
        base_model_id="deepseek-ai/DeepSeek-V4-Pro", batch=1, input_tokens=2048,
        entries=[{"name": "dsv4_csa_attention", "call": v4,
                  "model_id": "deepseek-ai/DeepSeek-V4-Pro"}], precision=AS_SHIPPED)
    assert bound["errors"] == []
    assert bound["results"][0]["phases"]["prefill"]["flops"] > 0
    key = f"dsv4_csa_attention@deepseek-ai/DeepSeek-V4-Pro#{v4}"
    assert bound["results"][0]["entry"] == key
    applied = bound["contract"]["per_component_dims"][key]
    assert applied["compress_ratio"] == 4  # the CSA layers' own literal

    # (c) unbound pick: real gaps only, no phantom dims, NO suggestions — the
    # caller's explicit dims (customization) repair it
    slot = _call("deepseek-ai/DeepSeek-V3.2", "dsv4_csa_attention", loader)
    out = mb.module_bench(base_model_id="deepseek-ai/DeepSeek-V3.2", batch=1,
                          input_tokens=2048,
                          entries=[{"name": "dsv4_csa_attention", "call": slot}],
                          precision=AS_SHIPPED)
    (err,) = out["errors"]
    assert "compress_ratio" in err["missing_dims"]
    assert "d_ffn" not in err["missing_dims"]      # unused sibling, phantom
    for phantom in ("Cape", "Cdim_io", "Cdim_norm"):  # bound at every call site
        assert phantom not in err["missing_dims"]
    assert "suggestions" not in err
    healed = mb.module_bench(base_model_id="deepseek-ai/DeepSeek-V3.2", batch=1,
                             input_tokens=2048,
                             entries=[{"name": "dsv4_csa_attention", "call": slot,
                                       "dims": applied, "seeded": True}], precision=AS_SHIPPED)
    assert healed["errors"] == [] and healed["results"][0]["phases"]["prefill"]["flops"] > 0


def test_provenance_comes_from_resolved_specs_not_match_candidates():
    """model_type is a match CANDIDATE, not a binding fact: several yamls claim
    deepseek_v2 / deepseek_v3, but only loader.match knows the winner. The
    no-qcomp MLA variant must credit only V2-Lite (V2/V3/V3.1 actually use
    mla_attention), and mla_attention must not absorb V2-Lite."""
    inv = mb._component_to_models()
    no_qcomp = inv.get("mla_no_qcomp_attention", [])
    assert "deepseek-ai/DeepSeek-V2-Lite" in no_qcomp
    for foreign in ("deepseek-ai/DeepSeek-V2", "deepseek-ai/DeepSeek-V3",
                    "deepseek-ai/DeepSeek-V3.1"):
        assert foreign not in no_qcomp, f"{foreign} does not bind mla_no_qcomp_attention"
    assert "deepseek-ai/DeepSeek-V2-Lite" not in inv.get("mla_attention", [])


def test_failed_seed_is_an_error_not_a_base_dims_fallback(loader, monkeypatch):
    """A bound entry whose binding can't resolve must NOT bench on with base
    dims under an `@ model` label — that's a disguised guess. It lands in
    errors with the seed failure as the reason, and never in results."""
    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)

    def boom(resolved, component, loader, site):
        raise ValueError("binding does not resolve offline")

    monkeypatch.setattr(mb, "_dims_at", boom)
    out = mb.module_bench(base_model_id="deepseek-ai/DeepSeek-V3.2", batch=1,
                          input_tokens=2048,
                          entries=[{"name": "mla_dsa_attention",
                                    "model_id": "zai-org/GLM-5.2"}], precision=AS_SHIPPED)
    assert out["results"] == []
    (err,) = out["errors"]
    assert err["model_id"] == "zai-org/GLM-5.2"
    assert "GLM-5.2" in err["error"] and err["missing_dims"] == []
    assert "does not resolve offline" in err["error"]


def test_inject_keeps_the_role_the_host_already_gave(loader):
    """Injecting a component the host already binds keeps the host's role, as
    the loader's registration does. Role is the quant key: gpt-oss binds
    `swa_bias_sink_attention` under `attention_swa`, and re-injecting it must
    not turn it into the library's `attention` for every SWA layer."""
    base = resolve_snapshot("openai/gpt-oss-20b", loader)
    name = "swa_bias_sink_attention"
    assert base.spec.component_roles[name] == "attention_swa"
    host = mb.inject_component(base, name, loader)
    assert host.spec.component_roles[name] == "attention_swa"
    assert base.spec.component_roles[name] == "attention_swa"
    # a component the host does not bind yet takes the library's role
    fresh = mb.inject_component(base, "mla_attention", loader)
    assert fresh.spec.component_roles["mla_attention"] == "attention"


def test_unbound_dims_callsite_subtraction_is_intersection():
    """A subcomponent parameter is 'satisfied at the call site' only when EVERY
    call of it binds the key. Two calls with divergent key sets must not mute
    each other's gaps (a union would), and a parameter no call binds surfaces."""
    from types import SimpleNamespace as NS
    from backend.arch.loader import ResolvedSpec
    sub = NS(kv_states=[], graph={
        "n": NS(op="linear", dims={"x": "a", "y": "b"}, dim=None,
                shape=["c"], copies=None, active=None, when=None, params={}),
    })
    root = NS(kv_states=[], graph={
        "call1": NS(op="sub", dims={"a": "d_model", "b": "d_model"}, dim=None,
                    shape=None, copies=None, active=None, when=None, params={}),
        "call2": NS(op="sub", dims={"a": "d_model"}, dim=None,
                    shape=None, copies=None, active=None, when=None, params={}),
    })
    components = {"root": root, "sub": sub}
    host = ResolvedSpec(NS(components=components, blocks={},
                           get_block=components.__getitem__),
                        {"d_model": 4096}, {}, {})
    # a: every call site binds it → satisfied; b: only one call site → gap;
    # c: never bound → gap. (d_model comes from the host.)
    assert mb.unbound_dims(host, {}, component="root") == ["b", "c"]


# --- prefill vs decode ------------------------------------------------------
#
# Decode is one new token per request against a cache of input_tokens — the
# phase the Inference page's decode column evaluates. Weight bytes are one copy
# of the block in either phase, so they stay a single field.

def test_phase_contract_echoes_the_phase_binding():
    """The contract says what each phase evaluated, read off the walk's own
    phase binding: prefill runs L tokens against L; decode runs 1 against L."""
    c = mb.phase_contract(2, 4096)
    assert c["prefill"] == {"batch": 2, "s_query": 4096, "s_kv": 4096}
    assert c["decode"] == {"batch": 2, "s_query": 1, "s_kv": 4096}


def test_decode_ffn_is_one_token_of_prefill(loader):
    """A token-wise FFN has no cache and no cross-token term: one decode token
    costs exactly prefill / L in FLOPs and activation bytes."""
    L = 2048
    r = _bench("meta-llama/Llama-3.2-3B", "swiglu_ffn", loader, L=L)
    pre, dec = r["phases"]["prefill"], r["phases"]["decode"]
    assert dec["flops"] * L == pre["flops"]
    for k in ("act_read_bytes", "act_write_bytes"):
        assert dec[k] * L == pre[k]
    assert dec["kv_read_bytes"] == 0 and dec["runs_as"] == "swiglu_ffn"


def test_decode_flops_are_same_source(loader):
    """Decode FLOPs are the decode-phase `project_block` — no second path."""
    base = resolve_snapshot("deepseek-ai/DeepSeek-V3", loader)
    host = mb.inject_component(base, "mla_attention", loader)
    direct = project_block(host, "mla_attention", batch=1, input_tokens=2048,
                           phase="decode", site=_site(base, "mla_attention"))
    r = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader)
    assert r["phases"]["decode"]["flops"] == direct.total_flops + (
        _grafted_cast_flops("deepseek-ai/DeepSeek-V3", "mla_attention",
                            loader, DECODE))


def test_decode_attention_reads_the_cache(loader):
    """Dense GQA at decode: the linear projections cost one token's worth, but
    the score/value work spans the whole cache, so a decode token costs MORE
    than an average prefill token (prefill is causal, decode attends all L).
    The KV column is the cache read — the whole cache for dense attention —
    and the step's traffic reads exactly that from the cache, fused or not,
    while its activation reads carry one token's worth, not the cache."""
    L = 4096
    r = _bench("meta-llama/Llama-3.2-3B", "gqa_attention", loader, L=L)
    pre, dec = r["phases"]["prefill"], r["phases"]["decode"]
    assert pre["flops"] < dec["flops"] * L < 2 * pre["flops"]
    assert dec["kv_read_bytes"] == r["kv"]["storage_bytes_at_L"]
    assert pre["kv_step_bytes"] == r["kv"]["storage_bytes_at_L"]
    assert dec["act_read_bytes_fused"] < dec["kv_read_bytes"]
    assert pre["kv_read_bytes"] == 0


def test_decode_kv_read_by_family(loader):
    """MLA reads its whole latent cache per decode step; DSA stores every
    position but reads only the top-k, so on one base its decode read is below
    both its own storage and dense MLA's read. DSA at decode runs as its
    declared absorbed form — identical to benching that form directly."""
    L = 8192
    mla = _bench("deepseek-ai/DeepSeek-V3.2", "mla_attention", loader, L=L)
    dsa = _bench("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader, L=L)
    absorbed = _bench("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention_absorbed",
                      loader, L=L)
    mla_d, dsa_d = mla["phases"]["decode"], dsa["phases"]["decode"]
    assert mla_d["kv_read_bytes"] == mla["kv"]["storage_bytes_at_L"] > 0
    assert 0 < dsa_d["kv_read_bytes"] < dsa["kv"]["storage_bytes_at_L"]
    assert dsa_d["kv_read_bytes"] < mla_d["kv_read_bytes"]
    assert dsa_d["runs_as"] == "mla_dsa_attention_absorbed"
    assert dsa_d["flops"] == absorbed["phases"]["decode"]["flops"]
    # A sparse attention prefills absorbed too (each query reads its selected
    # latent rows), so prefill runs the same form; dense MLA prefills
    # expanded, MHA.
    assert dsa["phases"]["prefill"]["runs_as"] == "mla_dsa_attention_absorbed"
    assert dsa["phases"]["prefill"]["flops"] == absorbed["phases"]["prefill"]["flops"]
    assert mla["phases"]["prefill"]["runs_as"] == "mla_attention"
    assert "one decode step" in dsa_d["units"]["traffic"]
    assert "cache_len = 8,192 tokens" in dsa_d["units"]["traffic"]
    assert "one forward" in dsa["phases"]["prefill"]["units"]["traffic"]


def _unit_strings(row):
    yield row["kv"]["unit"]
    yield from row["units"].values()
    for phase in row["phases"].values():
        yield from phase["units"].values()


def test_unit_strings_carry_the_values_they_were_computed_at(loader):
    """Every unit string names the numbers the walk ran at — batch, tokens,
    cache_len — and the per-layer scope; none leaves a bare symbol for the
    reader to guess."""
    base = resolve_snapshot("meta-llama/Llama-3.2-3B", loader)
    r = mb.bench_component(base, "gqa_attention", site=_site(base, "gqa_attention"), batch=3, input_tokens=2048,
                           dim_overrides={}, loader=loader, precision=AS_SHIPPED)
    for text in _unit_strings(r):
        for bare in ("@cache_len", "(batch", "input_tokens", "@(", "cache_len)"):
            assert bare not in text, (bare, text)
    assert r["kv"]["unit"] == "bytes/layer/request @ cache_len = 2,048 tokens"
    pre, dec = r["phases"]["prefill"]["units"], r["phases"]["decode"]["units"]
    assert pre["flops"] == "per layer · one forward @ batch 3 × 2,048 tokens"
    assert dec["flops"] == ("per layer · one decode step @ batch 3 × 1 new token, "
                            "cache_len = 2,048 tokens")
    assert all("per layer" in t for t in (*pre.values(), *dec.values()))
    assert "per layer" in r["units"]["weight_bytes"]


def test_kv_headline_dense_vs_recurrent_state(loader):
    """Dense KV grows one per-token slice per position; a recurrent state is
    constant per request and is reported as state, never as per-token KV."""
    gqa = _bench("meta-llama/Llama-3.2-3B", "gqa_attention", loader, L=2048)["kv"]
    assert gqa["bytes_per_token"] > 0 and gqa["state_bytes"] == 0
    assert gqa["bytes_per_token"] * 2048 == gqa["storage_bytes_at_L"]
    short, long = (_bench("Qwen/Qwen3-Coder-Next", "gated_delta_net_fused_attention",
                          loader, L=L)["kv"] for L in (1024, 8192))
    for kv in (short, long):
        assert kv["bytes_per_token"] == 0
        assert kv["state_bytes"] == kv["storage_bytes_at_L"] > 0
    assert short["state_bytes"] == long["state_bytes"]


def test_decode_memory_streams_kv_read_vs_weight_read(loader):
    """A decode step's two memory streams: the weights (one copy per step,
    whatever the batch) and the KV read (one cache per request). The weight
    read is the traffic engine's and equals the block's weight bytes; which
    stream dominates flips as the batch grows."""
    base = resolve_snapshot("meta-llama/Llama-3.2-3B", loader)

    def dec(batch):
        r = mb.bench_component(base, "gqa_attention", site=_site(base, "gqa_attention"), batch=batch,
                               input_tokens=4096, dim_overrides={},
                               loader=loader, precision=AS_SHIPPED)
        return r, r["phases"]["decode"]

    r1, d1 = dec(1)
    _r64, d64 = dec(64)
    assert d1["weight_read_bytes"] == r1["weight_bytes"] == d64["weight_read_bytes"]
    assert d64["kv_read_bytes"] == 64 * d1["kv_read_bytes"]
    assert d1["memory_dominant"] == "weights"
    assert d64["memory_dominant"] == "kv"
    # The unit says what the read covers: routed weights capped by the
    # step's tokens, and their scales with them.
    unit = d1["units"]["weight_read"]
    assert "cache_len = 4,096 tokens" in unit
    assert "weights + scales" in unit
    assert "routed experts: as far as tokens reach" in unit


# --- KV storage across context lengths --------------------------------------

_SWEEP = (1024, 8192, 131072)


def _kv_sweep(base_id, name, loader, *, L=4096):
    base = resolve_snapshot(base_id, loader)
    r = mb.bench_component(base, name, site=_site(base, name), batch=1, input_tokens=L, dim_overrides={},
                           loader=loader, precision=AS_SHIPPED,
                           kv_cache_lens=(*_SWEEP, L))
    kv = r["kv"]
    by_len = {p["cache_len"]: p["storage_bytes"] for p in kv["storage_by_cache_len"]}
    # the sweep point at the contract length is the headline storage number
    assert by_len[L] == kv["storage_bytes_at_L"]
    return kv, by_len


def test_kv_sweep_formulas_are_the_tensor_laws(loader):
    """The KV cells are the cached tensors' own laws: what a token adds and
    what each sweep length stores equal the sum, over the component's cached
    tensors (`eval.cache`), of each tensor's bytes under its law, and each
    cell's popover breaks into one subterm per tensor whose value is that
    tensor's bytes and whose expression is its law."""
    from backend.arch.eval.cache import of_component
    from backend.arch.eval.dim import expr_of
    from backend.arch.eval.transforms import analyze_isolated

    model, name = "deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention"
    kv, by_len = _kv_sweep(model, name, loader)
    base = resolve_snapshot(model, loader)
    host = mb.inject_component(base, name, loader)
    tensors = of_component(
        analyze_isolated(host, name, {}, _site(base, name)), name)
    assert len(tensors) > 1
    def bytes_at(t, n):
        return int(t.bytes_at(n, read=False))

    assert kv["bytes_per_token"] == sum(
        bytes_at(t, 1) for t in tensors if t.is_growing)
    for n, stored in by_len.items():
        assert stored == sum(bytes_at(t, n) for t in tensors), n
    cell = next(p for p in kv["storage_by_cache_len"] if p["cache_len"] == 4096)
    terms = {t["label"]: t for t in cell["formula"]["subterms"]}
    assert set(terms) == {t.name for t in tensors}
    for t in tensors:
        term = terms[t.name]
        assert term["result"] == bytes_at(t, 4096), t.name
        assert mb._star(str(expr_of(t.stored))) in term["symbolic"], t.name


def test_kv_tensor_nodes_carry_the_elements_formula(loader):
    """The aligned tensors fold pops over the identity of the number it shows,
    same contract as the KV panel's per-token / length cells."""
    mla = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader, L=4096)
    latent = next(n for n in mla["kv"]["nodes"] if n["name"] == "kv_a_layernorm")
    f = latent["formula"]
    assert "d_kv_lora" in f["symbolic"]
    _cell_in_substituted(f, latent["elements_per_token"], " el/tok")
    assert "·" not in f["symbolic"] and "·" not in f["substituted"]

    gdn = _bench("Qwen/Qwen3-Coder-Next", "gated_delta_net_fused_attention",
                 loader, L=4096)
    rec = next(n for n in gdn["kv"]["nodes"] if n["law"] == "constant")
    assert rec["elements_per_token"] is None
    _cell_in_substituted(rec["formula"], rec["elements_per_request"], " el/req")

    swa = _bench("google/gemma-2-9b", "swa_attention", loader, L=8192)
    windowed = next(n for n in swa["kv"]["nodes"] if n["law"] == "windowed")
    _cell_in_substituted(
        windowed["formula"], windowed["elements_per_token"], " el/tok")
    for row in (mla, gdn, swa):
        for node in row["kv"]["nodes"]:
            assert "·" not in node["formula"]["symbolic"]
            assert "·" not in node["formula"]["substituted"]


@pytest.mark.parametrize("base_id,name", [
    ("meta-llama/Llama-3.2-3B", "gqa_attention"),
    ("deepseek-ai/DeepSeek-V3", "mla_attention"),
    ("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention"),
    ("zai-org/GLM-5.2", "mla_dsa_attention_shared"),
])
def test_kv_sweep_dense_families_grow_linearly(loader, base_id, name):
    """GQA, MLA and DSA store every position (DSA reads only the top-k but
    keeps them all): storage is bytes-per-token × cache_len at every length."""
    kv, by_len = _kv_sweep(base_id, name, loader)
    assert kv["state_bytes"] == 0
    for n in _SWEEP:
        assert by_len[n] == kv["bytes_per_token"] * n, (n, by_len[n])


def test_kv_sweep_dsa_stores_the_indexer_key_and_shared_does_not(loader):
    """On one base, DSA's per-token cache is the MLA latent plus the indexer
    key, stored in full at every length; the shared variant reuses another
    layer's top-k and keeps no indexer key, so it stores the latent only."""
    full, full_by = _kv_sweep("zai-org/GLM-5.2", "mla_dsa_attention", loader)
    shared, shared_by = _kv_sweep("zai-org/GLM-5.2", "mla_dsa_attention_shared", loader)
    indexer = [n for n in full["nodes"] if "indexer" in n["name"]]
    assert indexer and not any("indexer" in n["name"] for n in shared["nodes"])
    key_per_token = sum(n["storage_bytes_at_L"] for n in indexer) // full["cache_len"]
    assert full["bytes_per_token"] - shared["bytes_per_token"] == key_per_token > 0
    for n in _SWEEP:
        assert full_by[n] - shared_by[n] == key_per_token * n


def test_kv_sweep_sliding_window_caps_at_the_window(loader):
    """A sliding-window cache grows until its window, then stays there."""
    kv, by_len = _kv_sweep("mistralai/Mistral-7B-v0.1", "swa_attention", loader)
    window = next(n["window"] for n in kv["nodes"] if n["law"] == "windowed")
    assert by_len[1024] == kv["bytes_per_token"] * 1024
    assert by_len[8192] == by_len[131072] == kv["bytes_per_token"] * window
    assert window < 8192


def test_kv_sweep_recurrent_state_is_flat(loader):
    """A recurrent state is the same bytes at every context length."""
    kv, by_len = _kv_sweep("Qwen/Qwen3-Coder-Next", "gated_delta_net_fused_attention", loader)
    assert kv["bytes_per_token"] == 0 and kv["state_bytes"] > 0
    assert set(by_len.values()) == {kv["state_bytes"]}


def test_model_caches_count_the_models_cache_copies(loader):
    """The whole-model KV is the entry's model's own cache of the component:
    its copies (the MTP head's included); a model that never instantiates
    the component has none. GLM-5.2 splits its layers between the full and
    the shared indexer variants. Every DSv3.2 copy keeps the same cache, so
    there the whole model is one layer's storage times the copies."""
    from backend.playground.compose import model_cache_at
    for model_id, shared_held in (("zai-org/GLM-5.2", True),
                                  ("zai-org/GLM-5.1", False)):
        resolved = resolve_snapshot(model_id, loader)
        site = mb.CallSite.of(
            resolved, "mla_dsa_attention",
            call=_call(model_id, "mla_dsa_attention", loader), model_id=model_id)
        shared = model_cache_at(resolved, model_id=model_id,
                                component="mla_dsa_attention_shared", site=site,
                                dims={}, loader=loader)
        assert bool(shared.entries) is shared_held, model_id
    call = _call("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader)
    r = mb.module_bench(base_model_id="deepseek-ai/DeepSeek-V3.2",
                        entries=[{"name": "mla_dsa_attention", "call": call,
                                  "model_id": "deepseek-ai/DeepSeek-V3.2"},
                                 {"name": "mla_dsa_attention_absorbed",
                                  "call": call}],
                        batch=1, input_tokens=2048, precision=AS_SHIPPED,
                        kv_cache_lens=[1024, 2048], model_cache=model_cache_at)
    bound, absorbed = r["results"]
    cfg = resolve_snapshot("deepseek-ai/DeepSeek-V3.2", loader).config
    assert bound["kv"]["model_layers"] == {
        "model_id": "deepseek-ai/DeepSeek-V3.2",
        "layers": cfg["num_hidden_layers"] + cfg.get("num_nextn_predict_layers", 0),
        "refused": None}
    # dropped into DSv3.2's call in place of what it calls there: DSv3.2 has
    # no whole model of it, so there is no whole-model number
    assert absorbed["kv"]["model_layers"]["layers"] is None
    assert "takes the place" in absorbed["kv"]["model_layers"]["refused"]
    assert [p["cache_len"] for p in bound["kv"]["storage_by_cache_len"]] == [1024, 2048]
    layers = bound["kv"]["model_layers"]["layers"]
    assert all(p["model_storage_bytes"] == p["storage_bytes"] * layers
               for p in bound["kv"]["storage_by_cache_len"])
    assert not any("model_storage_bytes" in p
                   for p in absorbed["kv"]["storage_by_cache_len"])


def test_whole_model_kv_follows_the_entrys_edits(loader):
    """An edit to the entry's dims is an edit to the model the whole-model
    column describes: halving Qwen3-8B's KV heads halves both its per-layer
    cache and its whole-model cache (36 layers, all calling this attention)."""
    from backend.playground.compose import model_cache_at
    mid, name = "Qwen/Qwen3-8B", "gqa_qknorm_attention"
    resolve_snapshot(mid, loader)

    def kv(dims):
        r = mb.module_bench(entries=[{"name": name, "model_id": mid,
                                      "dims": dims}],
                            batch=1, input_tokens=4096, precision=AS_SHIPPED,
                            kv_cache_lens=[4096], model_cache=model_cache_at)
        assert not r["errors"], r["errors"]
        return r["results"][0]["kv"]

    shipped, halved = kv({}), kv({"n_kv_heads": 4})
    assert halved["model_layers"]["layers"] == shipped["model_layers"]["layers"] == 36
    (s,), (h,) = shipped["storage_by_cache_len"], halved["storage_by_cache_len"]
    assert 2 * h["storage_bytes"] == s["storage_bytes"]
    assert 2 * h["model_storage_bytes"] == s["model_storage_bytes"]
    assert h["model_storage_bytes"] == 36 * h["storage_bytes"]


def test_whole_model_kv_keeps_each_calls_own_cache(loader):
    """MiMo-V2-Flash calls its sliding-window attention on 39 backbone layers
    and 3 MTP layers, both at 8 KV heads (the checkpoint's k_proj is
    [1536, 4096] in either). The whole model is each call's own cache times
    its copies — the same number whichever call the entry is benched at."""
    from backend.playground.compose import model_cache_at
    mid = "XiaomiMiMo/MiMo-V2-Flash"
    name = "swa_partial_rope_sink_vscale_attention"
    resolved = resolve_snapshot(mid, loader)
    calls = mb.component_calls(resolved, name, loader)
    assert len(calls) == 2
    rows = []
    for call in calls:
        r = mb.module_bench(entries=[{"name": name, "model_id": mid,
                                      "call": call["call"]}],
                            batch=1, input_tokens=4096, precision=AS_SHIPPED,
                            kv_cache_lens=[4096], model_cache=model_cache_at)
        assert not r["errors"], r["errors"]
        rows.append(r["results"][0]["kv"])
    site = mb.CallSite.of(resolved, name, call=calls[0]["call"], model_id=mid)
    view = model_cache_at(resolved, model_id=mid, component=name, site=site,
                          dims={}, loader=loader)
    copies = {view_entry.block_index: view_entry.count
              for view_entry in view.entries}
    per_call = {int(call["call"].split("/")[0]): kv["storage_by_cache_len"][0]["storage_bytes"]
                for call, kv in zip(calls, rows, strict=True)}
    assert len(per_call) == 2 and len(set(per_call.values())) == 1
    whole = sum(per_call[block] * n for block, n in copies.items())
    for kv in rows:
        point = kv["storage_by_cache_len"][0]
        assert point["model_storage_bytes"] == whole
        _adds_up(point["model_formula"], whole)


def test_fused_above_naive_sentinel_never_fires(loader):
    """Fusion only removes intermediates, so fused activation bytes above naive
    cannot be right; `fused_exceeds_naive` is a sentinel for exactly that. It is
    computed from the numbers the row reports, and it stays down for every
    family and phase — absorbed MLA and DSA at decode included, where the
    kernel reads the head-shared latent once per position rather than once per
    head."""
    rows = {
        "dsa": _bench("deepseek-ai/DeepSeek-V3.2", "mla_dsa_attention", loader, L=4096),
        "dsa_glm": _bench("zai-org/GLM-5.1", "mla_dsa_attention", loader, L=4096),
        "mla": _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader, L=4096),
        "gqa": _bench("meta-llama/Llama-3.2-3B", "gqa_attention", loader, L=4096),
    }
    for family, row in rows.items():
        assert set(row["phases"]) == {"prefill", "decode"}, family
        for phase, ph in row["phases"].items():
            naive = ph["act_read_bytes"] + ph["act_write_bytes"]
            fused = ph["act_read_bytes_fused"] + ph["act_write_bytes_fused"]
            assert ph["act_bytes_naive"] == naive, (family, phase)
            assert ph["act_bytes_fused"] == fused, (family, phase)
            assert ph["fused_exceeds_naive"] == (fused > naive), (family, phase)
            assert ph["fused_exceeds_naive"] is False, (family, phase, fused, naive)


# --- Weights at the checkpoint's precision ----------------------------------

_V4 = "deepseek-ai/DeepSeek-V4-Pro"


def _nodes_named(nodes, op):
    for node in nodes:
        if node.op == op:
            yield node
        yield from _nodes_named(node.children or [], op)


def _subtree_uids(node):
    out = {node.uid}
    for child in node.children:
        out |= _subtree_uids(child)
    return out


def test_bound_entry_weights_are_the_models_checkpoint(loader, monkeypatch):
    """`dsv4_csa_attention @ DeepSeek-V4-Pro` stores what one CSA layer of
    V4-Pro stores: the FP8 projections, the BF16 compressor and indexer
    weights the checkpoint keeps wide, and their scales. The parameters equal
    the Inspector's per-layer attention node, and with the scales the
    full-model walk's tensors under that node — on a base whose own weights
    are BF16 too, since the bound model's checkpoint is the one priced."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.projection import project_model
    from backend.arch.eval.transforms import Deployment
    from backend.arch.eval.dim import value_of

    v4 = resolve_snapshot(_V4, loader)
    model = project_model(v4, batch=1, input_tokens=4096)
    inspector = {n.storage_bytes for _t, _c, block in model.block_results
                 for n in _nodes_named(block.nodes, "dsv4_csa_attention")}
    assert len(inspector) == 1
    (inspector_bytes,) = inspector
    analysis = Deployment(Program.of(v4)).at(batch=1, input_tokens=1).analyze()
    call = next(n for n in analysis.program.nodes()
                if n.op == "dsv4_csa_attention")
    under = _subtree_uids(call)
    per_layer = sum(float(value_of(row.storage_bytes))
                    / float(value_of(row.count))
                    for row in analysis.iter_checkpoint()
                    if row.uid in under and not row.is_buffer)

    from tests.snapshots import snapshot_config
    monkeypatch.setattr(mb, "fetch_config", snapshot_config)
    monkeypatch.setattr(mb, "fetch_metadata", lambda mid, overlay_dir=None: {})
    monkeypatch.setattr(mb, "get_default_loader", lambda: loader)
    for base_id in (_V4, "Qwen/Qwen3-8B"):
        out = mb.module_bench(
            base_model_id=base_id, batch=1, input_tokens=4096,
            precision=AS_SHIPPED,
            entries=[{"name": "dsv4_csa_attention", "model_id": _V4,
                      "call": _call(_V4, "dsv4_csa_attention", loader)}],
            d_model=v4.dim_values["d_model"])
        assert out["errors"] == []
        (row,) = out["results"]
        storage = row["weight_storage"]
        assert storage["parameter_bytes"] == inspector_bytes, base_id
        assert row["weight_bytes"] == int(per_layer)
        assert storage["metadata_bytes"] > 0
        dtypes = [d["dtype"] for d in storage["by_dtype"]]
        assert dtypes[0] == "F8_E4M3" and "BF16" in dtypes
        assert storage["precision"].startswith("checkpoint precision: FP8")
        assert storage["source"] == {
            "model_id": _V4, "bound": True, "invoked": True,
            "role": "attention", "address": "layers.X.attn.",
            "note": f"as stored in {_V4}'s checkpoint"}
        # one stored copy: the unit names no step
        assert "stored once" in row["units"]["weight_bytes"]
        assert "prefill" not in row["units"]["weight_bytes"]
        assert "decode" not in row["units"]["weight_bytes"]


def test_unbound_entry_keeps_the_base_models_quant_context(loader):
    """An unbound entry is priced at the base model's checkpoint, and the
    payload says so: MLA on DeepSeek-V3 is FP8 like V3's attention."""
    r = _bench("deepseek-ai/DeepSeek-V3", "mla_attention", loader)
    source = r["weight_storage"]["source"]
    assert source["bound"] is False and source["invoked"] is True
    assert source["note"].startswith("unbound: priced at")
    assert r["weight_storage"]["by_dtype"][0]["dtype"] == "F8_E4M3"


def test_an_entry_in_the_place_of_a_call_says_which(loader):
    """A component the base does not call is evaluated at the call it takes
    the place of, and the note names that call."""
    base = resolve_snapshot("deepseek-ai/DeepSeek-V3.2", loader)
    site = _site(base, "mla_attention")
    assert site.invoked is False
    source = mb.bench_component(base, "mla_attention", batch=1,
                                input_tokens=2048, dim_overrides={},
                                loader=loader, precision=AS_SHIPPED,
                                site=site)["weight_storage"]["source"]
    assert source["invoked"] is False
    assert source["note"].endswith(
        f"in the place of the component called at {site.label}")


def test_fp8_storage_is_half_of_bf16_plus_scales(loader):
    """The same MLA block stored as DeepSeek-V3's FP8 checkpoint and as BF16:
    FP8 parameters are about half, and the FP8 block adds its scales."""
    from backend.arch.eval.formats import FormatPolicy
    base = resolve_snapshot("deepseek-ai/DeepSeek-V3", loader)
    fp8_site = _site(base, "mla_attention")
    bf16_site = replace(fp8_site, formats=FormatPolicy(default_dtype="BF16"))

    def storage(site):
        return mb.bench_component(base, "mla_attention", batch=1,
                                  input_tokens=2048, dim_overrides={},
                                  loader=loader, precision=AS_SHIPPED,
                                  site=site)["weight_storage"]

    fp8, bf16 = storage(fp8_site), storage(bf16_site)
    assert bf16["metadata_bytes"] == 0
    assert bf16["bytes"] == bf16["parameter_bytes"]
    assert fp8["metadata_bytes"] > 0
    assert fp8["bytes"] == fp8["parameter_bytes"] + fp8["metadata_bytes"]
    ratio = fp8["parameter_bytes"] / bf16["parameter_bytes"]
    assert 0.5 <= ratio < 0.51, ratio


def test_moe_decode_reads_only_the_routed_experts(loader):
    """A batch-1 decode step of DeepSeek-V3's MoE reads the experts its one
    token routes to, with those experts' scales (plus the shared expert and
    the router); a 2,048-token prefill reads the whole bank. Both
    equal the serving per-op rows of one MoE layer of the whole model."""
    from backend.arch.eval.projection import project_model
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze

    base = resolve_snapshot("deepseek-ai/DeepSeek-V3", loader)
    name = "moe_ffn_gate_routing_bias"
    r = _bench("deepseek-ai/DeepSeek-V3", name, loader)
    prefill = r["phases"]["prefill"]["weight_read_bytes"]
    decode = r["phases"]["decode"]["weight_read_bytes"]
    assert decode < prefill <= r["weight_bytes"]

    layers = next(count for template, count, _b in
                  project_model(base).block_results if name in template)
    for phase, got in (("prefill", prefill), ("decode", decode)):
        rows = analyze(
            base, parallel=ParallelConfig(), batch_size=1,
            avg_input_tokens=2048, avg_output_tokens=0,
            phase=phase, breakdown=True).per_op_timing
        serving = sum(int(x["weight_read_bytes_per_rank"])
                      + int(x["metadata_bytes_per_rank"])
                      for x in rows
                      if f"{name}_norm" in (x["aggregation_id"] or "")
                      and "|mlp." in x["aggregation_id"])
        assert serving == got * layers, phase


def test_dense_weight_read_is_the_whole_block_in_both_phases(loader):
    """A dense block reads every weight once per forward, prefill or decode."""
    r = _bench(_V4, "dsv4_csa_attention", loader, L=4096,
               ov={"compress_ratio": 4})
    phases = r["phases"]
    assert (phases["prefill"]["weight_read_bytes"]
            == phases["decode"]["weight_read_bytes"] == r["weight_bytes"])


@pytest.mark.parametrize("model_id", [
    "Qwen/Qwen3.5-35B-A3B",      # hybrid: a library block (linear attention) has most layers
    "deepseek-ai/DeepSeek-V3",   # a template the resolution specialized
    "Qwen/Qwen3-8B",             # a block the model declares itself
])
def test_compare_representative_block_benches_at_its_call(model_id, loader):
    """Compare's "compare in Playground" opens module comparison on each
    model's representative block at the call the analysis publishes. That
    call runs the block, the column benches it as the model parameterizes
    it — one layer's parameters — and its listed calls name every other
    layer block of the model, so the column can move to any of them."""
    from backend.analyzer import _representative_template
    from backend.arch.eval.projection import project_model
    resolved = resolve_snapshot(model_id, loader)
    report = project_model(resolved, batch=1, input_tokens=1)
    rep = _representative_template(report)
    assert rep["basis"] == "most_layers"
    site = mb.CallSite.of(resolved, rep["name"], call=rep["call"])
    assert site.invoked and site.is_stage

    r = mb.module_bench(entries=[{"name": rep["name"], "model_id": model_id,
                                  "call": rep["call"]}],
                        batch=1, input_tokens=2048, precision=AS_SHIPPED)
    assert not r["errors"], r["errors"]
    (row,) = r["results"]
    _name, _count, block = report.block_results[rep["bucket_id"]]
    assert row["params"] == block.total_params
    stages = {(b, n) for b, (n, c, _) in enumerate(report.block_results)
              if c and report.block_aux_kinds[b] is None}
    listed = {(int(c["call"]), c["component"]) for c in row["calls"]
              if "/" not in c["call"]}
    assert stages <= listed


def test_tied_representative_blocks_are_distinct_columns(loader):
    """DeepSeek-V4-Flash has no single representative block: its HCA and
    CSA blocks tie at 20 of 43 layers. Compare hands Playground every tied
    block at its own call; the request accepts one column per block, and
    each call runs its block. (Kimi-K3 used to be the tie here; since
    1f63a7e2d a block's invocations count together, and its one block holds
    most layers.)"""
    from backend.analyzer import _representative_template
    from backend.api._models import ModuleBenchRequest
    from backend.arch.eval.projection import project_model
    mid = "deepseek-ai/DeepSeek-V4-Flash"
    resolved = resolve_snapshot(mid, loader)
    rep = _representative_template(project_model(resolved, batch=1, input_tokens=1))
    assert rep["basis"] == "tie" and rep["name"] is None
    tied = rep["tied"]
    assert len(tied) > 1 and len({b["call"] for b in tied}) == len(tied)
    req = ModuleBenchRequest(entries=[
        {"name": b["name"], "model_id": mid, "call": b["call"]} for b in tied])
    assert len(req.entries) == len(tied)
    for b in tied:
        site = mb.CallSite.of(resolved, b["name"], call=b["call"])
        assert site.invoked and site.is_stage


def test_representative_blocks_draw_one_graph_diff(loader):
    """The columns Compare opens are blocks the models declare themselves —
    DeepSeek-V3's specialized MoE layer, Qwen3-8B's own `block`. Two of
    them bench together and their compute graphs diff like any two
    library components'."""
    from backend.analyzer import _representative_template
    from backend.arch.eval.projection import project_model
    entries = []
    for mid in ("deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-8B"):
        resolved = resolve_snapshot(mid, loader)
        rep = _representative_template(project_model(resolved, batch=1, input_tokens=1))
        assert rep["name"] not in loader.component_names()
        entries.append({"name": rep["name"], "model_id": mid, "call": rep["call"]})
    r = mb.module_bench(entries=entries, batch=1, input_tokens=2048,
                        precision=AS_SHIPPED)
    assert not r["errors"], r["errors"]
    diff = r["graph_diff"]
    assert diff["available"] is True, diff.get("reason")
    assert len(diff["entries"]) == 2


def test_every_binding_of_a_slot_parameterized_component_evaluates(loader):
    """DeepSeek-V4.1-Flash binds csa2_attention at six calls, all through
    the per-layer map (synthesized specializations whose names carry the
    slot bindings). Every one is a bindings row with the dims its own
    binding resolves to.

    Regression: the six specializations were never listed at all, because
    the row filter matched the bare component name only."""
    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash", loader)
    sites = [site for site in mb.CallSite.calls(resolved)
             if mb.binding_of(resolved.spec, site.component, "csa2_attention")]
    assert len(sites) == 6
    assert all(site.component != "csa2_attention" for site in sites)
    rows = mb._model_bindings(resolved, "deepseek-ai/DeepSeek-V4.1-Flash",
                              "csa2_attention", loader, sites)
    assert all("error" not in row and row["dims"] for row in rows)
    # Layer 14's call and layers 2/8's bind the same form at the same dims
    # (`binding_key`: one parameterization); every other call is its own —
    # layers 3-19 and 21-39 bind one form at two compress ratios.
    assert {row["model_parameterizations"] for row in rows} == {5}
    by_call = {row["call"]: row["parameterization"] for row in rows}
    assert by_call["3/attn"] == by_call["4/attn"]
    assert by_call["7/attn"] != by_call["8/attn"]
    # The candidate-source call sizes the candidate blocks, and the
    # shared-keys call scores only those candidates (sglang
    # v41_indexer/sparse_table.py consume_decode): the two calls that read
    # the block count.
    for call in ("5/attn", "6/attn"):
        row = next(row for row in rows if row["call"] == call)
        assert row["dims"]["n_topk_blocks"] == 2048
    assert all("n_topk_blocks" not in row["dims"] for row in rows
               if row["call"] not in ("5/attn", "6/attn"))
    # Seeding a compare entry at a call resolves too.
    dims = mb.component_dims_from_model(
        "csa2_attention", "deepseek-ai/DeepSeek-V4.1-Flash", "8/attn")
    assert dims == next(row["dims"] for row in rows if row["call"] == "8/attn")


def _attention_rows(analysis, block=None):
    """`(s_kv, live pairs, FLOPs)` of each attention score, softmax
    and apply of `block` (every block when None): the rows a query reads
    and what reading them costs."""
    from backend.arch.eval.dim import value_of
    out = []
    for node in analysis.program.nodes():
        name = node.name.split("__")[0]
        if (node.children or name not in ("qk_score", "softmax", "attn_apply")
                or (block is not None
                    and analysis.program.block_of(node.uid) != block)):
            continue
        dims = analysis.node_dims.values[node.uid]
        out.append((name, value_of(dims["s_kv"]),
                    value_of(dims["s_pairs"]),
                    value_of(analysis.cost.get(node.uid).flops)))
    return out


@pytest.mark.parametrize("phase", [PREFILL, DECODE])
def test_a_borrowing_call_reads_the_publishers_law_alone_as_in_the_model(
        loader, phase):
    """A DeepSeek-V4.1-Flash layer that reads another layer's compressed KV
    (`csa2_compress_borrowed`) reads `min(index_topk, end_pos // ratio)`
    of its rows per query (source/model.py:578): the law the kv-source
    layer's cached `compress_pooled` states, which the model hands the call
    with the `compress_kv` stream. Benched alone, the call is handed the
    same law, so its attention reads the rows, the live pairs and the FLOPs
    its layers read in the whole model — 128 window + min(512, 4096 // r)
    compressed rows at 4096 tokens, and fewer at 1000 where 1000 // 2 < 512."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import (CallSite, Deployment,
                                              isolate_resolved)
    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash", loader)
    borrowing = [site for site in CallSite.calls(resolved)
                 if "csa2_compress_borrowed" in str(site.component)
                 and site.component.startswith("csa2_attention")]
    assert len(borrowing) == 3

    def at(spec, tokens):
        step = Deployment(Program.of(spec)).at(batch=2, input_tokens=tokens)
        return (step.during(DECODE, cache_len=tokens) if phase == DECODE
                else step).analyze()
    seen = set()
    for tokens in (4096, 1000):
        whole = at(resolved, tokens)
        for site in borrowing:
            alone = at(isolate_resolved(resolved, "csa2_attention", None,
                                        site), tokens)
            rows = _attention_rows(alone)
            assert rows == _attention_rows(whole, site.invocation), site.call
            seen.add((tokens, rows[0][1]))
    assert (4096, 640) in seen and (1000, 628) in seen


def test_csa2_slot_caches_explain_the_graph_without_joining_the_layer_total(loader):
    """CSA2's graph draws the caches its bound slots own. The parent KV formula
    still counts only its window — the shared compressed/index caches are not
    billed to every consumer layer — so those graph-visible tensors are listed
    separately with their owning slot."""
    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash", loader)
    source = next(site for site in mb.CallSite.calls(resolved)
                  if "csa2_compress_store" in site.component
                  and site.component.startswith("csa2_attention"))
    consumer = next(site for site in mb.CallSite.calls(resolved)
                    if "csa2_shared_selection" in site.component
                    and site.component.startswith("csa2_attention"))

    def bench(site):
        return mb.bench_component(
            resolved, site.component, site=site, batch=1, input_tokens=4096,
            dim_overrides={}, loader=loader, precision=AS_SHIPPED)

    owned = bench(source)
    assert [node["name"] for node in owned["kv"]["nodes"]] == ["kv_concat"]
    slots = {node["name"]: node for node in owned["kv"]["slot_nodes"]}
    assert {"kv_state", "score_state", "k_cache", "compress_pooled"} <= set(slots)
    assert {slots["k_cache"]["owner"], slots["compress_pooled"]["owner"]} == {
        "indexer", "compressed"}
    assert owned["kv"]["storage_bytes_at_L"] == 67_584
    assert "slot-owned shared caches" in (
        owned["kv"]["formulas"]["storage"].get("note") or "")

    def graph_writes(nodes):
        names = set()
        for node in nodes:
            names.update(
                entry["tensor"] for entry in (node.get("kv") or {}).get("writes") or [])
            names.update(graph_writes(node.get("children") or []))
        return names

    assert graph_writes(owned["block_graph"]["nodes"]) == (
        {node["name"] for node in owned["kv"]["nodes"]}
        | set(slots)
    )

    borrowed = bench(consumer)
    assert [node["name"] for node in borrowed["kv"]["nodes"]] == ["kv_concat"]
    assert borrowed["kv"]["slot_nodes"] == []
    assert graph_writes(borrowed["block_graph"]["nodes"]) == {"kv_concat"}


def test_csa2_whole_model_kv_follows_the_drawn_binding(loader):
    """The whole-model KV sweep answers the binding the graph drew: source
    layers carry their slot-owned shared caches, while the two shared-consumer
    ratios are separate parameterizations rather than one 30-layer mixture."""
    from backend.playground.compose import model_cache_at

    mid = "deepseek-ai/DeepSeek-V4.1-Flash"
    resolve_snapshot(mid, loader)
    expected = {
        "3/attn": (3, 2_414_592),
        "7/attn": (15, 1_013_760),
        "8/attn": (15, 1_013_760),
    }
    out = mb.module_bench(
        base_model_id=mid,
        entries=[{"name": "csa2_attention", "model_id": mid, "call": call}
                 for call in expected],
        batch=1, input_tokens=4096, precision=AS_SHIPPED,
        kv_cache_lens=[4096], model_cache=model_cache_at,
    )
    assert not out["errors"], out["errors"]
    rows = {row["call"]["call"]: row for row in out["results"]}
    for call, (layers, storage) in expected.items():
        row = rows[call]
        assert row["kv"]["model_layers"] == {
            "model_id": mid, "layers": layers, "refused": None}
        point = row["kv"]["storage_by_cache_len"][0]
        assert point["model_storage_bytes"] == storage
        assert point["model_formula"] is not None


def test_sub_caches_absorbed_by_a_parent_are_not_relabeled_as_slot_owned(loader):
    """DSv4's parent cache declaration already owns its compressor/indexer
    tensors. They belong in the parent formula once; explaining them again as
    “slot-owned outside the total” would contradict the number beside them."""
    r = _bench("deepseek-ai/DeepSeek-V4-Flash", "dsv4_csa_attention",
               loader, L=4096)
    names = {node["name"] for node in r["kv"]["nodes"]}
    assert any(name.startswith("compressor.") for name in names)
    assert any(name.startswith("indexer.compressor.") for name in names)
    assert r["kv"]["slot_nodes"] == []
    assert "slot-owned shared caches" not in (
        r["kv"]["formulas"]["storage"].get("note") or "")


def test_a_call_the_model_hands_no_compressed_kv_names_the_stream(loader):
    """The same consumer dropped in at a window-only layer, which no
    kv-source publishes to: nothing hands it the compressed KV's law, and
    the union it scores is refused naming the streams that came without
    one."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import (CallSite, Deployment,
                                              isolate_resolved)
    resolved = resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash", loader)
    consumer = next(site.component for site in CallSite.calls(resolved)
                    if "csa2_shared_selection" in str(site.component)
                    and site.component.startswith("csa2_attention"))
    site = CallSite.of(resolved, consumer, call="1/attn")
    with pytest.raises(ValueError, match=r"compress_kv.*no cache law"):
        Deployment(Program.of(isolate_resolved(resolved, consumer, None, site))
                   ).at(batch=1, input_tokens=4096).analyze()


def test_an_inline_call_keeps_its_literal_slot_bindings(loader):
    """Kimi-K3's decoder blocks call gated_ffn inline in `blocks:` with
    literal `params: { act: situ }`. Each such call is a bindings row with
    dims.

    Regression: the inline call's `params:` were dropped when the isolated
    walk rebuilt the call, leaving `<act>` unbound — the Bound-models row
    showed no dims."""
    resolved = resolve_snapshot("moonshotai/Kimi-K3", loader)
    inline = [site for site in mb.CallSite.calls(resolved)
              if site.component == "gated_ffn" and site.params]
    assert inline
    assert all(dict(site.params) == {"act": "situ"} for site in inline)
    rows = mb._model_bindings(resolved, "moonshotai/Kimi-K3", "gated_ffn",
                              loader, inline)
    assert rows and all("error" not in row and row["dims"] for row in rows)


def test_a_component_with_open_slots_takes_no_other_components_call(loader):
    """A component whose slots only its caller fills (`BlockGraph.open_slots`)
    cannot take another component's place: that call binds none of them. So
    a model that does not call it lists no call for it — neither another
    block's stage nor a call of a component in its role — and an entry named
    at such a call is refused in words, not with an unbound `<slot>` op."""
    resolved = resolve_snapshot("Qwen/Qwen2.5-7B", loader)
    for name in ("csa2_attention", "dsv41_hc_block"):
        assert loader.component(name).open_slots
        assert mb.component_calls(resolved, name, loader) == []
    # a component without open slots still takes the place of one in its role
    assert mb.component_calls(resolved, "mla_attention", loader)
    v41 = resolve_snapshot("deepseek-ai/DeepSeek-V4.1-Flash", loader)
    own = mb.component_calls(v41, "dsv41_hc_block", loader)
    assert own and all(c["component"].startswith("dsv41_hc_block") for c in own)
    # Call 2 (layer 1) runs dsv41_hc_block_engram, another component.
    assert "2" not in {c["call"] for c in own}
    site = mb.CallSite.of(v41, "dsv41_hc_block", call="2",
                          model_id="deepseek-ai/DeepSeek-V4.1-Flash")
    with pytest.raises(ValueError, match="leaves its slots"):
        mb.isolate_resolved(mb.inject_component(v41, "dsv41_hc_block", loader),
                            "dsv41_hc_block", {}, site)


def test_a_walk_without_a_binding_has_no_env():
    """`Analysis.binding` is None for a walk bound to no workload: that is
    an empty env, read off the field, not a crash and not a probe."""
    from types import SimpleNamespace
    assert mb._walk_env(SimpleNamespace(binding=None)) == {}
