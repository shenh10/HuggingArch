"""The React app (`web/`) must typecheck and build.

A failure here means `web/` does not produce a production bundle — run
`npm ci && npm run build` in `web/`. CI installs Node so this is a hard
gate there; locally a missing toolchain is a skip (same pattern as
`test_frontend_build.py`).
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB = PROJECT_ROOT / "web"


def _missing(reason: str) -> None:
    if os.environ.get("CI"):
        pytest.fail(f"React web build gate cannot run in CI: {reason}")
    pytest.skip(reason)


def test_web_app_builds():
    if shutil.which("node") is None:
        _missing("node not on PATH; React web build requires Node")
    if shutil.which("npm") is None:
        _missing("npm not on PATH; React web build requires npm")
    if not (WEB / "node_modules").exists():
        _missing("web/node_modules missing; run `npm ci` in web/")
    # Own process group: npm spawns tsc/vite grandchildren, and killing npm
    # alone on timeout orphans them mid-build — they keep rewriting web/dist
    # after the test has moved on.
    proc = subprocess.Popen(
        ["npm", "run", "build"],
        cwd=WEB, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        pytest.fail("web/ build timed out after 180s")
    assert proc.returncode == 0, (
        f"web/ build failed:\n{stdout}\n{stderr}"
    )
    html = (WEB / "dist" / "index.html").read_text()
    assert (WEB / "dist" / "index.html").is_file(), "web/dist/index.html missing after build"
    assert 'id="root"' in html
    assert "/site-config.js" in html


def test_fusion_pattern_graph_draws_sources_and_sinks(tmp_path):
    """The fusion DAG shows the I/O boundary, not only its interior nodes."""
    import json

    if shutil.which("node") is None:
        _missing("node not on PATH")
    esbuild = WEB / "node_modules" / ".bin" / "esbuild"
    if not esbuild.exists():
        _missing("web/node_modules missing; run `npm ci` in web/")
    pattern = {
        "kind": "nodes",
        "external_inputs": ["q", "kv", "k_pe", "index_key"],
        "nodes": {
            "q_norm": {"op": "rmsnorm", "style": "norm", "inputs": ["external.q"]},
            "kv_norm": {"op": "rmsnorm", "style": "norm", "inputs": ["external.kv"]},
            "k_rope": {"op": "rope_emb", "style": "op", "inputs": ["external.k_pe"]},
            "index_key_norm": {"op": "layernorm", "style": "norm", "inputs": ["external.index_key"]},
            "index_key_rope_slice": {"op": "slice", "style": "view", "inputs": ["index_key_norm.out"]},
            "index_key_norope_slice": {"op": "slice", "style": "view", "inputs": ["index_key_norm.out"]},
            "index_key_rope": {"op": "rope_emb", "style": "op", "inputs": ["index_key_rope_slice.out"]},
            "index_key_concat": {"op": "concat", "style": "view", "inputs": ["index_key_rope.out", "index_key_norope_slice.out"]},
        },
        "edges": [
            {"src": "external.q", "dst": "q_norm"},
            {"src": "external.kv", "dst": "kv_norm"},
            {"src": "external.k_pe", "dst": "k_rope"},
            {"src": "external.index_key", "dst": "index_key_norm"},
            {"src": "index_key_norm.out", "dst": "index_key_rope_slice"},
            {"src": "index_key_norm.out", "dst": "index_key_norope_slice"},
            {"src": "index_key_rope_slice.out", "dst": "index_key_rope"},
            {"src": "index_key_rope.out", "dst": "index_key_concat"},
            {"src": "index_key_norope_slice.out", "dst": "index_key_concat"},
        ],
        "io_mapping": {
            "q": "external.q",
            "kv": "external.kv",
            "k_pe": "external.k_pe",
            "index_key": "external.index_key",
            "output": "q_norm.out",
            "kv_out": "kv_norm.out",
            "k_out": "k_rope.out",
            "index_key_out": "index_key_concat.out",
        },
    }
    bundle = tmp_path / "fusion.mjs"
    proc = subprocess.run(
        [str(esbuild), str(WEB / "src" / "pages" / "Fusion.tsx"),
         "--bundle", "--format=esm", "--platform=neutral", f"--outfile={bundle}"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, f"esbuild Fusion.tsx failed:\n{proc.stdout}\n{proc.stderr}"
    runner = tmp_path / "run.mjs"
    runner.write_text(
        "import { patternToDag } from './fusion.mjs';\n"
        f"const dag = patternToDag({json.dumps(pattern)});\n"
        "const sources = dag.nodes.filter((n) => n.subtitle === 'source').map((n) => n.label);\n"
        "const sinks = dag.nodes.filter((n) => n.subtitle === 'sink').map((n) => n.label);\n"
        "const edge = (src, dst) => dag.edges.some((e) => e.src === src && e.dst === dst);\n"
        "if (JSON.stringify(sources) !== JSON.stringify(['q', 'kv', 'k_pe', 'index_key'])) throw new Error('sources');\n"
        "if (JSON.stringify(sinks) !== JSON.stringify(['output', 'kv_out', 'k_out', 'index_key_out'])) throw new Error('sinks');\n"
        "if (!edge('q_norm', 'sink_output')) throw new Error('primary output edge');\n"
        "if (!edge('index_key_concat', 'sink_index_key_out')) throw new Error('indexer output edge');\n"
    )
    run = subprocess.run(
        ["node", str(runner)], cwd=tmp_path,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert run.returncode == 0, f"patternToDag failed:\n{run.stdout}\n{run.stderr}"


def test_inference_defaults_model_and_gpu():
    """Empty model POSTs sweeps that 400; the page must ship DeepSeek-V3 + H100."""
    src = (WEB / "src" / "pages" / "Inference.tsx").read_text()
    assert 'DEFAULT_INFER_MODEL = "deepseek-ai/DeepSeek-V3"' in src
    assert 'DEFAULT_INFER_GPU = "H100-SXM"' in src
    assert "useState(DEFAULT_INFER_MODEL)" in src
    assert "useState(DEFAULT_INFER_GPU)" in src


def test_web_scaffold_has_no_gallery_presets():
    """React Gallery reads generated axes.json — never a handwritten presets.ts."""
    assert not (WEB / "src" / "gallery" / "presets.ts").exists()
    assert (WEB / "src" / "pages" / "Gallery.tsx").is_file()
    assert (WEB / "src" / "gallery" / "axes.json").is_file()
    src = (WEB / "src" / "pages" / "Gallery.tsx").read_text()
    assert "from \"../gallery/axes.json\"" in src or "from '../gallery/axes.json'" in src
    assert '<option value="B200">B200</option>' not in src
    assert "Failed to load GPUs" in src


def test_playground_staged_selection_waits_for_one_benchmark():
    """Picking several modules must not launch (and cancel) a bench per chip."""
    page = (WEB / "src" / "pages" / "Playground.tsx").read_text()
    picks = (WEB / "src" / "pages" / "pg" / "PicksBar.tsx").read_text()
    assert "selectionDirty" in page and "setSelectionDirty(true)" in page
    assert "if (selectionDirty)" in page
    # The run button lives under the catalog panel, not in the sticky picks bar.
    assert 'id="pg-run-picks"' in page and 'id="pg-run-picks"' not in picks
    assert "Benchmark ${entries.length}" in page
    assert "onClick={applySelection}" in page


def test_gallery_scatter_tooltips_show_attention_badges():
    """A point identifies its attention mechanism without a legend lookup."""
    backend = (PROJECT_ROOT / "backend" / "api" / "gallery_overview.py").read_text()
    assert "attention_families" in backend
    assert "_attention_families" in backend
    charts = (WEB / "src" / "pages" / "GalleryCharts.tsx").read_text()
    assert "attentionBadges" in charts
    assert 'title="Attention family"' in charts
    gallery = (WEB / "src" / "pages" / "Gallery.tsx").read_text()
    assert "attention: m.attention_families" in gallery


def test_gallery_merges_timeline_and_activated_params():
    """One bubble chart carries date, total, activated params, and family."""
    gallery = (WEB / "src" / "pages" / "Gallery.tsx").read_text()
    assert "y: total" in gallery
    assert "activeBubble" in gallery
    charts = (WEB / "src" / "pages" / "GalleryCharts.tsx").read_text()
    assert "size?: number" in charts
    assert "symbolSize: p.size" in charts


def test_formula_popover_is_structured_and_wraps_long_equations():
    """The per-op formula popover renders the shared formulaHTML rows (formula /
    values / per layer / = / note) inside its anchored shell, keeping the
    FORMULA header and the "{x}-bound" verdict badge."""
    src = (WEB / "src" / "FormulaHover.tsx").read_text()
    assert "formulaHTML(fields)" in src
    assert "dangerouslySetInnerHTML" in src
    assert "{bound}-bound" in src
    phase = (WEB / "src" / "pages" / "PhaseTable.tsx").read_text()
    playground = (WEB / "src" / "pages" / "Playground.tsx").read_text()
    assert "FormulaHover" in phase
    assert "PerOpTable" in playground
    # The old Eq/Vals grid and its helpers are gone — one renderer, in tooltip.ts.
    for dead in ("formatFormulaText", "splitTopLevelCommas", "splitTopLevelPluses",
                 "FormulaNote", "prettyResult"):
        assert dead not in src
    tip = (WEB / "src" / "tooltip.ts").read_text()
    # Component Σs use the same generic rows — no bespoke list renderer.
    assert "components" not in tip
    # The parent formula line is the node's symbolic identity; children's
    # IDs fill in only when the parent has none.
    assert "function formulaLine" in tip
    assert "st.label || st.symbol" in tip
    assert "if (symbolic) return symbolic" in tip


def test_per_op_table_paints_the_program_fold():
    """Cost table is flops_by_module (walk order + depth), not path-prefix groups."""
    src = (WEB / "src" / "pages" / "PhaseTable.tsx").read_text()
    assert "groupByModule" not in src
    assert "moduleGroupOf" not in src
    assert "visibleFold" in src
    assert "flops_by_module" in src
    assert 'data-fold-row=""' in src
    assert 'data-module-toggle=""' in src
    assert "data-leaf-row" in src
    assert "DataTable" in src and "renderBody" in src


def test_react_write_hash_preserves_unknown_keys():
    """writeHash must not drop vanilla keys this cut does not own (`sub`/`m`/`dec`)."""
    url_ts = (WEB / "src" / "url.ts").read_text()
    assert "export function mergeHash" in url_ts
    assert "new URLSearchParams(raw)" in url_ts
    write = url_ts.split("export function writeHash", 1)[1]
    assert "mergeHash(window.location.hash" in write


def test_react_inference_has_no_silent_fallbacks():
    """Fetch failure must not plant H100-SXM / EP=8. No island hooks.

    H100-SXM is the page's declared initial GPU (DEFAULT_INFER_GPU); it may
    appear there and nowhere else, so no failure path can substitute it."""
    src = (WEB / "src" / "pages" / "Inference.tsx").read_text()
    compare = (WEB / "src" / "compare.ts").read_text()
    assert "ep_moe: 8" not in src
    assert "dp_attention: 8" not in src
    assert src.count("H100-SXM") == 1 and 'const DEFAULT_INFER_GPU = "H100-SXM";' in src
    assert "H100-SXM" not in compare
    assert "registerSyntheticHook" not in src
    assert "registerSyntheticHook" not in compare


def test_react_sweep_hydrate_keeps_per_row_ep_and_gpu(tmp_path):
    """Later rows keep their own ep_moe / gpu_sku (the vanilla clobber). A
    link saved before precision profiles loses its precision fields on
    hydrate — Compare prices every model as shipped — and says so."""
    import json
    import base64

    if shutil.which("node") is None:
        _missing("node not on PATH")
    esbuild = WEB / "node_modules" / ".bin" / "esbuild"
    if not esbuild.exists():
        _missing("web/node_modules missing; run `npm ci` in web/")

    models = [
        "deepseek-ai/DeepSeek-V3",
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V4-Pro",
    ]
    cfgs = [
        {"gpu_sku": "GPU-A", "weight_dtype_bytes": 2, "parallel": {"ep_moe": 8, "dp_attention": 4}},
        {"gpu_sku": "GPU-B", "kv_dtype": "fp8", "parallel": {"ep_moe": 4, "dp_attention": 2}},
        {"gpu_sku": "GPU-C", "act_dtype_per_role": {"attention": 1}, "parallel": {"ep_moe": 2, "dp_attention": 1}},
    ]
    raw = json.dumps({"template": cfgs[0], "cfgs": cfgs}, separators=(",", ":"))
    s = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
    m = "|".join(models)
    outfile = tmp_path / "compare.mjs"
    proc = subprocess.run(
        [str(esbuild), str(WEB / "src" / "compare.ts"),
         "--bundle", "--format=esm", "--platform=neutral", f"--outfile={outfile}"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, f"esbuild compare.ts failed:\n{proc.stdout}\n{proc.stderr}"
    runner = tmp_path / "run.mjs"
    runner.write_text(
        "import { hydrateSweep, encodeSweep, mergeSweep } from './compare.mjs';\n"
        f"const m = {json.dumps(m)};\n"
        f"const s = {json.dumps(s)};\n"
        "const { rows, template, droppedPrecision } = hydrateSweep(m, s);\n"
        "if (!droppedPrecision) throw new Error('old precision fields not reported');\n"
        "const want = [[8,'GPU-A'],[4,'GPU-B'],[2,'GPU-C']];\n"
        "const retired = ['weight_dtype_bytes', 'kv_dtype', 'act_dtype_per_role'];\n"
        "for (let i = 0; i < 3; i++) {\n"
        "  const ep = rows[i]?.cfg?.parallel?.ep_moe;\n"
        "  const gpu = rows[i]?.cfg?.gpu_sku;\n"
        "  if (ep !== want[i][0] || gpu !== want[i][1]) {\n"
        "    throw new Error(`row ${i} clobbered: ep=${ep} gpu=${gpu}`);\n"
        "  }\n"
        "  for (const k of retired) if (k in rows[i].cfg) throw new Error(`row ${i} kept ${k}`);\n"
        "  if (rows[i].model_id !== " + json.dumps(models) + "[i]) throw new Error('model');\n"
        "}\n"
        "if ('weight_dtype_bytes' in template) throw new Error('template kept weight_dtype_bytes');\n"
        "const enc = encodeSweep(rows, template);\n"
        "const again = hydrateSweep(enc.m, enc.s);\n"
        "if (again.droppedPrecision) throw new Error('re-encoded link still carries precision');\n"
        "if (again.rows[1].cfg.parallel.ep_moe !== 4) throw new Error('roundtrip ep');\n"
        "if (again.rows[2].cfg.gpu_sku !== 'GPU-C') throw new Error('roundtrip gpu');\n"
        "const merged = mergeSweep('#dec=1&p=prefill&foo=bar', enc);\n"
        "if (!merged.includes('dec=1')) throw new Error('dec dropped');\n"
        "if (!merged.includes('p=prefill')) throw new Error('p dropped');\n"
        "console.log('ok');\n"
    )
    run = subprocess.run(
        ["node", str(runner)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert run.returncode == 0, f"hydrateSweep failed:\n{run.stdout}\n{run.stderr}"


def test_react_sweep_posts_estimate_sweep_from_row_cfg():
    """Sweep POSTs /inference/estimate-sweep; bodies come from row cfg, not a shared form."""
    src = (WEB / "src" / "pages" / "Inference.tsx").read_text()
    compare = (WEB / "src" / "compare.ts").read_text()
    assert "/inference/estimate-sweep" in src
    assert "export function sweepRequestBodies" in compare
    assert "row.cfg?.gpu_sku" in compare
    assert "body.parallel = { ...cfg.parallel }" in compare


def test_react_sweep_request_bodies_keep_per_row_cfg(tmp_path):
    """Later rows keep their own ep_moe / gpu_sku in the estimate-sweep POST body."""
    import json

    if shutil.which("node") is None:
        _missing("node not on PATH")
    esbuild = WEB / "node_modules" / ".bin" / "esbuild"
    if not esbuild.exists():
        _missing("web/node_modules missing; run `npm ci` in web/")

    rows = [
        {"model_id": "deepseek-ai/DeepSeek-V3",
         "cfg": {"gpu_sku": "GPU-A", "parallel": {"ep_moe": 8}}},
        {"model_id": "deepseek-ai/DeepSeek-V3.2",
         "cfg": {"gpu_sku": "GPU-B", "parallel": {"ep_moe": 4}}},
        {"model_id": "deepseek-ai/DeepSeek-V4-Pro",
         "cfg": {"gpu_sku": "GPU-C", "parallel": {"ep_moe": 2}}},
    ]
    outfile = tmp_path / "compare.mjs"
    proc = subprocess.run(
        [str(esbuild), str(WEB / "src" / "compare.ts"),
         "--bundle", "--format=esm", "--platform=neutral", f"--outfile={outfile}"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, f"esbuild compare.ts failed:\n{proc.stdout}\n{proc.stderr}"
    runner = tmp_path / "run.mjs"
    runner.write_text(
        "import { sweepRequestBodies } from './compare.mjs';\n"
        f"const rows = {json.dumps(rows)};\n"
        "const built = sweepRequestBodies(rows);\n"
        "if (built.error) throw new Error(built.error);\n"
        "const bodies = built.bodies;\n"
        "if (bodies.length !== 3) throw new Error('len');\n"
        "if (bodies[0].seq_lens.length !== 9) throw new Error('seq_lens');\n"
        "const g0 = bodies[0].batch_grid;\n"
        "if (!g0 || g0.length !== 10) throw new Error('batch_grid len');\n"
        "if (g0[0].key !== 'b1' || g0[0].prefill_batch_size !== 1 || g0[0].decode_batch_size !== 1) throw new Error('b1');\n"
        "if (g0[1].key !== 'b8' || g0[1].prefill_batch_size !== 1 || g0[1].decode_batch_size !== 8) throw new Error('b8');\n"
        "if (g0[8].key !== 'b1024' || g0[8].prefill_batch_size !== 1 || g0[8].decode_batch_size !== 1024) throw new Error('b1024');\n"
        "if (g0[9].key !== 'max' || g0[9].prefill_batch_size !== 1 || g0[9].decode_batch_size !== null) throw new Error('max');\n"
        "if (JSON.stringify(g0) !== JSON.stringify(bodies[1].batch_grid)) throw new Error('grid drifted');\n"
        "if (bodies[0].avg_output_tokens !== 256) throw new Error('avg_output_tokens');\n"
        "if (bodies[1].avg_output_tokens !== 256 || bodies[2].avg_output_tokens !== 256) throw new Error('out drifted');\n"
        "const want = [[8,'GPU-A'],[4,'GPU-B'],[2,'GPU-C']];\n"
        "for (let i = 0; i < 3; i++) {\n"
        "  const ep = bodies[i].parallel?.ep_moe;\n"
        "  const gpu = bodies[i].gpu_sku;\n"
        "  if (ep !== want[i][0] || gpu !== want[i][1]) {\n"
        "    throw new Error(`body ${i} clobbered: ep=${ep} gpu=${gpu}`);\n"
        "  }\n"
        "  if ('precision' in bodies[i]) throw new Error('Compare row carries a precision');\n"
        "}\n"
        "const missing = sweepRequestBodies([\n"
        "  {model_id:'a/A', cfg:{gpu_sku:'GPU-A'}},\n"
        "  {model_id:'b/B', cfg:null},\n"
        "]);\n"
        "if (!missing.error) throw new Error('missing gpu should error, not default a SKU');\n"
        "const pin = sweepRequestBodies([\n"
        "  {model_id:'a/A', cfg:{gpu_sku:'GPU-A', avg_output_tokens:999}},\n"
        "  {model_id:'b/B', cfg:{gpu_sku:'GPU-B', avg_output_tokens:1}},\n"
        "]);\n"
        "if (pin.error) throw new Error(pin.error);\n"
        "if (pin.bodies[0].avg_output_tokens !== 256 || pin.bodies[1].avg_output_tokens !== 256) {\n"
        "  throw new Error('cfg must not override IC_SWEEP_OUTPUT pin');\n"
        "}\n"
        "console.log('ok');\n"
    )
    run = subprocess.run(
        ["node", str(runner)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert run.returncode == 0, f"sweepRequestBodies failed:\n{run.stdout}\n{run.stderr}"


def test_react_sweep_pushed_row_posts_the_single_model_body(tmp_path):
    """A compare row that received the template by "push" posts the same
    deployment fields as the single-model estimate of that config: both go
    through `estimateCfgFields`, and every pushed key reaches the body
    (speculative preset → speculative_draft_tokens/acceptance_rate,
    cp_scheme). Neither carries a precision field: the backend refuses the
    retired ones, and Compare prices every model as shipped. A shared config saved when K was the verify window
    (`speculative_K`) hydrates as `speculative_draft_tokens` = window − 1."""
    import json

    if shutil.which("node") is None:
        _missing("node not on PATH")
    esbuild = WEB / "node_modules" / ".bin" / "esbuild"
    if not esbuild.exists():
        _missing("web/node_modules missing; run `npm ci` in web/")

    src = (WEB / "src" / "pages" / "Inference.tsx").read_text()
    # The single-model run and the compare template come from one snapshot
    # of the deployment the run uses (the form's, or the one the model opens
    # at), and the run posts it through the shared builder.
    assert "...estimateCfgFields(deploymentCfg(usePar))" in src
    assert "const snap = deploymentCfg(usePar);" in src

    template = {
        "gpu_sku": "GPU-A", "gpu_memory_fraction": 0.85,
        "parallel": {"tp_attention": 2, "tp_dense_ffn": 2, "tp_moe_expert": 1,
                     "ep_moe": 8, "cp": 1, "pp": 1, "dp_attention": 4, "dp_replicas": 1},
        "cp_scheme": "ring", "tie_tp": True,
        "prefix_cached_tokens": 256,
        "prefill_chunk_size": 4096,
        "speculative_mode": "eagle", "speculative_draft_tokens": 4, "acceptance_rate": 0.65,
        "include_aux_weights": True,
    }
    outfile = tmp_path / "compare.mjs"
    proc = subprocess.run(
        [str(esbuild), str(WEB / "src" / "compare.ts"),
         "--bundle", "--format=esm", "--platform=neutral", f"--outfile={outfile}"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, f"esbuild compare.ts failed:\n{proc.stdout}\n{proc.stderr}"
    runner = tmp_path / "run.mjs"
    runner.write_text(
        "import { applyTemplateSection, encodeSweep, estimateCfgFields, hydrateSweep,\n"
        "         sweepRequestBodies, SPEC_PRESETS, SWEEP_SECTION_KEYS } from './compare.mjs';\n"
        f"const template = {json.dumps(template)};\n"
        "let rows = [{model_id:'a/A', cfg:null}, {model_id:'b/B', cfg:{gpu_sku:'GPU-Z'}}];\n"
        "for (const section of Object.keys(SWEEP_SECTION_KEYS)) {\n"
        "  rows = applyTemplateSection(rows, template, section).rows;\n"
        "}\n"
        "const built = sweepRequestBodies(rows);\n"
        "if (built.error) throw new Error(built.error);\n"
        "const single = estimateCfgFields(template);\n"
        "const eagle = SPEC_PRESETS.eagle;\n"
        "if (single.speculative_draft_tokens !== eagle.K || single.acceptance_rate !== eagle.alpha) {\n"
        "  throw new Error('speculative preset not materialised: ' + JSON.stringify(single));\n"
        "}\n"
        "const legacy = encodeSweep([{model_id:'a/A', cfg:{speculative_mode:'custom', speculative_K:4}},\n"
        "                            {model_id:'b/B', cfg:null}], {speculative_mode:'custom', speculative_K:1});\n"
        "const hydrated = hydrateSweep(legacy.m, legacy.s);\n"
        "const oldRow = hydrated.rows[0].cfg;\n"
        "if (oldRow.speculative_draft_tokens !== 3 || 'speculative_K' in oldRow\n"
        "    || hydrated.template.speculative_draft_tokens !== 0) {\n"
        "  throw new Error('legacy window not migrated: ' + JSON.stringify(hydrated));\n"
        "}\n"
        "if (estimateCfgFields(oldRow).speculative_draft_tokens !== 3) throw new Error('migrated K not posted');\n"
        "if (single.parallel.cp_scheme !== 'ring') throw new Error('cp_scheme dropped');\n"
        "if (single.prefill_chunk_size !== 4096) throw new Error('prefill_chunk_size dropped');\n"
        "if ('quant' in SWEEP_SECTION_KEYS) throw new Error('Compare still pushes precision');\n"
        "const legacyBody = estimateCfgFields({...template, kv_dtype: 'fp8', weight_dtype_bytes: 1});\n"
        "if ('kv_dtype' in legacyBody || 'weight_dtype_bytes' in legacyBody) throw new Error('retired field posted');\n"
        "if ('tie_tp' in single || 'speculative_mode' in single) throw new Error('UI-only key posted');\n"
        "const sweepOnly = new Set(['model_id', 'seq_lens', 'batch_grid', 'avg_output_tokens']);\n"
        "for (const body of built.bodies) {\n"
        "  const rest = Object.fromEntries(Object.entries(body).filter(([k]) => !sweepOnly.has(k)));\n"
        "  const sort = (o) => JSON.stringify(o, Object.keys(o).sort());\n"
        "  const a = Object.keys(rest).sort(), b = Object.keys(single).sort();\n"
        "  if (JSON.stringify(a) !== JSON.stringify(b)) {\n"
        "    throw new Error(`key sets differ: row ${JSON.stringify(a)} vs single ${JSON.stringify(b)}`);\n"
        "  }\n"
        "  for (const k of b) {\n"
        "    const x = rest[k], y = single[k];\n"
        "    const same = typeof x === 'object' ? sort(x) === sort(y) : x === y;\n"
        "    if (!same) throw new Error(`${body.model_id}.${k}: ${JSON.stringify(x)} vs ${JSON.stringify(y)}`);\n"
        "  }\n"
        "}\n"
        "console.log('ok');\n"
    )
    run = subprocess.run(
        ["node", str(runner)],
        cwd=tmp_path, capture_output=True, text=True, timeout=30, check=False,
    )
    assert run.returncode == 0, f"pushed-row body differs from single run:\n{run.stdout}\n{run.stderr}"


def test_react_gallery_line_chart_zoom():
    """React charts use ECharts toolbox dataZoom (box-zoom / restore /
    save) plus inside wheel-zoom — not homemade SVG gestures."""
    charts = (WEB / "src" / "pages" / "GalleryCharts.tsx").read_text()
    ha = (WEB / "src" / "charts" / "HaChart.tsx").read_text()
    opts = (WEB / "src" / "charts" / "options.ts").read_text()
    pkg = (WEB / "package.json").read_text()
    assert "echarts" in pkg and "echarts-for-react" in pkg
    assert "from \"../charts/HaChart\"" in charts
    # Axes run a little past the data so edge bubbles are whole, and the
    # padded ends carry no label of their own.
    assert "showMinLabel: false" in charts
    assert "makeScatterLabelLayout" in charts
    assert "makeLineEndLabelLayout" in charts
    layout = (WEB / "src" / "charts" / "labelLayout.ts").read_text()
    assert "placed.length = 0" in layout
    # New pass = a (series, point) key repeats. A decreasing composite
    # key is not a new pass — ECharts does not visit series in index order.
    assert "seen.has(key)" in layout
    assert "key <= prev" not in layout
    assert "function wrapChartLabel" in charts
    assert 'type: "plain"' in charts
    assert 'type: "scroll"' not in charts
    assert "labelLinePoints" in (WEB / "src" / "charts" / "labelLayout.ts").read_text()
    assert "lazy(" in ha
    assert "interactiveBits" in opts and "dataZoom" in opts
    assert "toolbox" in opts
    assert "restore" in opts
    assert "saveAsImage" in opts
    assert "frontend/service" not in charts


def test_react_gallery_signin_reads_site_config():
    nav = (WEB / "src" / "TopNav.tsx").read_text()
    auth = (WEB / "src" / "auth.ts").read_text()
    assert 'href="/auth/login.html"' in nav
    assert "SITE_CONFIG" in auth
    assert "authEnabled" in auth
    assert "Sign in" in nav
    assert "Profile" in nav
    assert 'href="/account/profile.html"' in nav
    assert "Workspace" in nav
    assert 'href="/account/workspace.html"' in nav
    assert "Spec generations" not in nav
    assert "My modules" not in nav
    assert "Kernel benchmarks" not in nav
    assert 'id="auth-admin-link"' in nav
    assert 'href="/admin/"' in nav
    assert "Admin" in nav
    assert 'id="auth-signout"' in nav
    assert "Sign out" in nav
    assert "function AuthMenuIcon" in nav
    assert 'aria-hidden="true"' in nav
    assert "PROFILE_ICON" in nav and "WORKSPACE_ICON" in nav
    assert "ADMIN_ICON" in nav and "SIGNOUT_ICON" in nav
    assert 'id="auth-avatar"' in nav
    assert "/analytics/event" in auth
    # Cookie-based sessions: the bundle never builds an Authorization header.
    assert "Authorization" not in auth and "Bearer" not in auth
    assert "/api/auth/refresh" in auth and "/api/auth/exchange" in auth
    assert "huggingarch.auth.token" in auth


def test_account_pages_share_account_nav():
    """The three account pages share AccountNav — no hand-written sidebar, and
    the panel lists live once in AccountNav (ACCOUNT_PANELS / WORKSPACE_PANELS)."""
    nav = (WEB / "src" / "AccountNav.tsx").read_text()
    assert 'aria-label="Account sections"' in nav
    assert "WORKSPACE_PANELS" in nav
    assert '"modules"' in nav
    for page in ("AccountProfile", "AccountWorkspace", "Admin"):
        src = (WEB / "src" / "pages" / f"{page}.tsx").read_text()
        assert 'from "../AccountNav"' in src, page
        assert 'aria-label="Account sections"' not in src, page
    app = (WEB / "src" / "App.tsx").read_text()
    assert 'path="/account/workspace.html"' in app
    assert 'path="/account/bench-review.html"' in app
    assert "AccountWorkspace" in app
    assert "BenchReview" in app
    assert 'to="/account/workspace.html#benchmarks"' in app  # retired page → redirect
    assert "AccountBenchmarks" not in app


def test_per_task_model_editor_collapses_to_default_model():
    """The credential card stays one summary line; Edit expands the per-task grid."""
    src = (WEB / "src" / "pages" / "AccountProfile.tsx").read_text()
    assert "function TaskModelSummary" in src
    assert "function TaskModelGrid" in src
    summary = src.split("function TaskModelSummary", 1)[1].split(
        "function TaskModelGrid", 1,
    )[0]
    assert "Task models" in summary
    assert "all tasks use this default" in summary
    assert '"Configure"' not in summary
    editor = src.split("function AgentCredForm", 1)[1].split(
        "\nexport function AccountProfile", 1,
    )[0]
    assert "<TaskModelGrid" in editor
    assert "task_models: taskModels" in editor
    assert "load endpoint's model list" not in src


def test_structure_l1_flops_follow_symbolic_shapes():
    """L1 headers use the walk's flops_symbolic when the backend says it is
    still a law over the workload (`flops_is_law`), not a bound integer."""
    src = (WEB / "src" / "pages" / "Structure.tsx").read_text()
    assert "flops_is_law" in src
    assert "flops_symbolic" in src
    assert "boxSubtitle" in src


def test_credential_edit_expands_in_place():
    """Edit and Add both expand a card in the list; there is no bottom form."""
    src = (WEB / "src" / "pages" / "AccountProfile.tsx").read_text()
    assert "function AgentCredForm" in src
    assert "setEditingId" in src
    assert "setAdding" in src
    assert "function credLabel" in src
    assert "function credId" in src
    assert 'mode="edit"' in src
    assert 'mode="add"' in src
    assert "setCredEditing(true)" not in src
    assert "function TaskModelEditor" not in src
    editor = src.split("function AgentCredForm", 1)[1].split(
        "\nexport function AccountProfile", 1,
    )[0]
    assert "leave blank to keep the current key" in editor
    assert "<TaskModelGrid" in editor
    assert "onDefaultChange" in editor
    assert "Name (optional)" in editor
    profile = src.split("export function AccountProfile", 1)[1]
    assert "editingId === id" in profile
    assert "Add a key" in profile
    assert "New key" in profile
    assert "<TaskModelSummary" in profile


def test_playground_summary_has_sol_and_measured_time_tracks():
    """Resource columns say work; timing columns need a GPU and measured coverage."""
    table = (WEB / "src" / "pages" / "pg" / "SummaryTable.tsx").read_text()
    assert '"time"' in table
    assert "siPrefix" in table
    page = (WEB / "src" / "pages" / "Playground.tsx").read_text()
    assert "gpu_sku: gpuSku" in page
    assert "gpuSku={gpuSku} gpus={gpus} onGpuSku={onGpuSku}" in page
    assert '"prefill_t_sol"' in page
    assert '"prefill_t_measured"' in page
    assert '"decode_t_sol"' in page
    assert '"decode_t_measured"' in page
    assert "measured" in page
    assert 'label: "t measured"' in page
    assert "groupEdge" in table


def test_open_pr_shows_in_flight_like_review():
    """Open/Update PR must spin like Review — the GitHub POST is silent for
    seconds otherwise, and a second click would fire another create."""
    src = (WEB / "src" / "pages" / "AccountWorkspace.tsx").read_text()
    assert "Stop review" in src
    assert "Stop fix" in src
    assert "Loading your specs" in src
    assert "groupRowsByModel" in src
    assert "earlier spec" in src
    assert "function HfButton" in src
    assert "function runAgentLine" in src
    assert "data-model-name" in src
    assert 'm.kind === "gen"' in src
    assert "publishedDate" in src
    assert "release_date" in src
    assert 'kind: "pr"' in src
    assert "Opening PR…" in src
    assert "fix_logs" in src
    assert "review_logs" in src
    assert "Fix log" in src
    assert "Review log" in src
    assert "Updating PR…" in src
    assert "Rebase onto main" in src
    assert "rebase-pr" in src
    assert "prInflight" in src
    card = (WEB / "src" / "AgentJob.tsx").read_text()
    assert 'pr: "Pull request in progress"' in card


def test_react_topnav_is_the_only_live_nav():
    nav = (WEB / "src" / "TopNav.tsx").read_text()
    assert 'from "react-router-dom"' in nav
    assert "<Link" in nav
    assert "to={tabHref(t)}" in nav
    assert "readLastInspectorHref" in nav
    assert 'href: "/inference.html"' in nav
    assert 'key: "inspector", href: "/"' in nav
    assert 'href: "/gallery.html"' in nav
    assert 'href: "/fusion.html"' in nav
    assert 'href: "/kernel-corpus.html"' in nav
    assert 'label: "Kernel Bench"' in nav


def test_graph_shell_is_the_shared_compute_graph():
    """Inspector / Inference / Playground draw graphs through GraphShell."""
    src = (WEB / "src" / "pages" / "Structure.tsx").read_text()
    assert "export function GraphShell" in src
    assert "export function StructureGraph" in src
    assert "function applyInnerL1" in src
    assert "scopeId" in src
    assert "<GraphDiffView gd={diff} rows={rows} />" in src
    assert "data-graph-empty" in src
    # A refused overlay stacks each entry's graph under its label.
    assert "data-graph-label" in src
    assert "function stackedGraphs" in src
    playground = (WEB / "src" / "pages" / "Playground.tsx").read_text()
    assert "GraphShell" in playground
    inspector = (WEB / "src" / "pages" / "Inspector.tsx").read_text()
    assert "GraphShell" in inspector
    assert 'aria-label={open ? "Collapse computation" : "Expand computation"}' in inspector
    assert 'arch-expand-btn${open ? " is-open" : ""}' in inspector
    infer = (WEB / "src" / "pages" / "ComputeGraph.tsx").read_text()
    assert "GraphShell" in infer


def test_spec_generate_asks_over_kept_draft_before_starting():
    """A kept draft is chosen over in a dialog BEFORE the job starts — not by
    letting the plain POST fail 409 `draft_kept` and asking in the banner."""
    src = (WEB / "src" / "pages" / "Inspector.tsx").read_text()
    assert "checkKeptDraft" in src
    assert "setResumePrompt(mid)" in src
    assert 'id="resume-draft-dialog"' in src
    # The gate precedes runSpecJob's POST, and only a plain "new" run asks…
    gate = src.index('(await checkKeptDraft(mid)) === "prompt"')
    post = src.index("await runSpecJob(")
    assert gate < post
    assert 'start === "new" && !overlayOpen.current' in src
    # …while the dialog's buttons resume / discard explicitly.
    assert 'generateSpec(mid, "discard")' in src
    assert 'generateSpec(mid, "resume")' in src
    # The status pre-check reads the same `resumable` flag the server states
    # when no run is live (the 409 banner stays only as the race fallback).
    assert "s?.resumable" in src


def test_scaling_chart_click_applies_nearest_point():
    """Click-to-Plan is the whole plot area (nearest point), not a 5px symbol;
    the tooltip carries no click hint and the panel notes what was applied."""
    panel = (WEB / "src" / "pages" / "ScalingPanel.tsx").read_text()
    assert "containPixel" in panel and "convertToPixel" in panel
    assert "getZr()" in panel
    assert "Applied:" in panel
    assert 'id="infer-scaling-export"' not in panel
    assert 'id="infer-scaling-metric"' not in panel
    assert "GPU: 同 Plan" not in panel
    assert "MeasuredCheck" in panel
    # One measured checkbox, in the one section that draws a step time.
    assert 'id="infer-throughput-measured"' in panel
    assert 'id="infer-scaling-measured"' not in panel
    assert 'id="infer-par-measured"' not in panel
    assert 'id="infer-scaling-ngpu"' in panel
    assert "BATCH_PAR_DEGREE = 8" in panel
    assert "function feasibleDegrees" in panel
    assert "function degreeOptions" in panel
    assert "p.feasible" in panel
    assert "batchDegrees.map" in panel
    fn = panel.index("export function ScalingPanel")
    assert panel.find("<ParallelSection", fn) < panel.find("<ThroughputSection", fn)
    assert panel.find("<ThroughputSection", fn) < panel.find(">KV storage<", fn)
    assert panel.find(">KV storage<", fn) < panel.find("<KvStorageSection", fn)
    # One chart area: no section draws a throughput of its own beside it.
    assert "prefill_tps_per_gpu" not in panel
    assert 'title="Prefill (B=1)"' not in panel
    # And the hover rows carry the facts: B/GPU and the step time that made it.
    assert "B/GPU=${p.b}" in panel and "fmtT(p.t)" in panel
    # A line's hover reads the drawn phase's own batch and step time.
    assert 'const tField = phase === "decode" ? "tpot_s" : "prefill_t_s"' in panel
    # SOL and measured share a legend name (same colour, dashed vs solid).
    # A legend click must not setOption inside ECharts' own dispatch, and a
    # hidden series is selected === false (empty {} means all visible).
    assert "function legendNames" in panel
    assert "function mirrorMeasuredSel" not in panel
    assert "function legendSelEqual" in panel
    assert "const names = legendNames(series)" in panel and "data: names," in panel
    assert "skipPlotClick" in panel
    assert "hidden?.[s.label] === false" in panel
    assert 'label: `${label} · measured`' not in panel
    assert 'label: `${gpu} · measured`' not in panel
    chart = (WEB / "src" / "charts" / "HaChart.tsx").read_text()
    assert "onInstance" in chart


def test_scaling_panel_uses_frontend_sweep_cache():
    """Dropdown switches read an in-tab cache keyed by POST body, so going
    back to a combination already drawn does not wait on HTTP."""
    panel = (WEB / "src" / "pages" / "ScalingPanel.tsx").read_text()
    cache = (WEB / "src" / "sweepCache.ts").read_text()
    assert "readSweepCache" in panel
    assert "fetchSweepCached" in panel
    assert 'from "../sweepCache"' in panel
    assert "sweepCacheKey" in cache
    assert "MAX_ENTRIES" in cache
    assert "fetchJson" in cache
    # Theory only: a measured sweep is always asked of the server (docs/measured_calibration.md §7.5).
    assert "isTheory" in cache and "built_at" not in panel


def test_chart_numbers_read_one_way(tmp_path):
    """Every chart reads counts, context lengths and durations through one
    set of formatters: at most three significant digits, the unit stepped up
    (10T, not 10000.00B; 1M, not 1048576; 700 s, not 700.00 s)."""
    if shutil.which("node") is None:
        _missing("node not on PATH")
    esbuild = WEB / "node_modules" / ".bin" / "esbuild"
    if not esbuild.exists():
        _missing("web/node_modules missing; run `npm ci` in web/")
    outfile = tmp_path / "format.mjs"
    proc = subprocess.run(
        [str(esbuild), str(WEB / "src" / "format.ts"),
         "--bundle", "--format=esm", "--platform=neutral", f"--outfile={outfile}"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    runner = tmp_path / "run.mjs"
    runner.write_text(
        "import { fmtCompact, fmtCtx, fmtDuration } from './format.mjs';\n"
        "console.log(JSON.stringify([\n"
        "  [1e13, 1e12, 1e11, 30.53e9, 539.01e6, 1500, 999].map(fmtCompact),\n"
        "  [1024, 131072, 1048576, 10000, 32406].map(fmtCtx),\n"
        "  [700, 0.05, 0.014, 1.234, 3.2e-6].map(fmtDuration),\n"
        "]));\n"
    )
    run = subprocess.run(["node", str(runner)], cwd=tmp_path, capture_output=True,
                         text=True, timeout=30, check=False)
    assert run.returncode == 0, run.stderr
    import json
    assert json.loads(run.stdout) == [
        ["10T", "1T", "100B", "30.5B", "539M", "1.5K", "999"],
        ["1K", "128K", "1M", "10K", "32.4K"],
        ["700 s", "50 ms", "14 ms", "1.23 s", "3.2 µs"],
    ]
    charts = (WEB / "src" / "pages" / "GalleryCharts.tsx").read_text()
    assert "formatNumber" not in charts and "function fmtSeq(" not in charts


def test_scaling_kv_storage_section():
    """KV storage shares Context × Batch GPU / nGPU / parallel and Plan's
    gpu_memory_fraction. Max-requests is the estimate's max_batch_per_gpu
    (the Summary's figure); hover names weights, the KV budget and what one
    request needs."""
    panel = (WEB / "src" / "pages" / "ScalingPanel.tsx").read_text()
    assert 'id="infer-scaling-kv"' in panel
    assert 'id="infer-scaling-kv-gpu"' not in panel
    assert 'id="infer-scaling-kv-ngpu"' not in panel
    assert 'id="infer-scaling-kv-par"' not in panel
    assert 'id="infer-scaling-frac"' not in panel
    assert "gpu_memory_fraction: props.frac" in panel
    assert "per_request_kv_bytes" in panel
    # One request's cache on ONE GPU of the deployment, said as such — the
    # Gallery's per-request figure is the whole request's.
    assert 'title="KV per request, on one GPU"' in panel
    assert 'yTitle="KV / request / GPU"' in panel
    assert 'title="Max requests / GPU"' in panel
    assert "function kvMaxRequests" in panel
    assert "kv_budget_bytes" in panel
    assert "weight_bytes_per_gpu" in panel
    assert "weights over HBM" in panel
    assert "fmtGB" not in panel
    assert "tipFormat={formatGiB}" in panel
    assert "function KvStorageSection" in panel
    assert "sweepSeqResults({ results_by_batch: batchResults })" in panel
    kv_fn = panel.index("function KvStorageSection")
    assert '"/inference/estimate-sweep"' not in panel[kv_fn:panel.index("export function ScalingPanel")]
    plan = (WEB / "src" / "pages" / "PlanCard.tsx").read_text()
    assert "GPU_MEMORY_FRACS = [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5]" in plan
    assert 'id="infer-frac"' in plan
    assert 'type="range"' not in plan.split("GPU memory fraction", 1)[1][:800]


def test_batch_curve_legend_wraps_below_the_axis_name():
    """Group names and the x-axis title must not share one cramped row."""
    src = (WEB / "src" / "pages" / "Corpus.tsx").read_text()
    charts = src[src.index("function CurveFacet"):src.index("export function Corpus")]
    assert 'type: "plain"' in charts
    assert 'type: "scroll"' not in charts
    assert "bottom: 46 + 18 * rows" in charts
    # The reserve is measured, not a fixed chart width (a phone draws one
    # chart per row, a desktop two).
    assert "legendRows(kernels, width" in charts
    assert "/ 600" not in charts


def test_kernel_bench_providers_tab_reads_the_registry():
    """The Providers tab is the live /api/bench/providers catalog, not a list."""
    src = (WEB / "src" / "pages" / "Corpus.tsx").read_text()
    assert 'id="tab-providers"' in src
    assert '"/api/bench/providers"' in src
    assert 'view === "providers"' in src
    assert "unmeasured" in src
    assert "undeclared" in src


def test_kernel_bench_ops_dropdown_lists_measured_ops():
    """All ops on envelope/rows is the signatures the server measured under
    the other filters (its facets), not the provider catalog (which still
    seeds the Providers tab)."""
    src = (WEB / "src" / "pages" / "Corpus.tsx").read_text()
    assert 'view === "providers" ? libraries.flatMap((l) => l.kernels.map((k) => k.signature))' in src
    assert "...(facets.signature || [])" in src
