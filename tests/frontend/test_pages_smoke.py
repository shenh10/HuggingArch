"""Headless smoke + regression tests for the main React pages.

Same harness as `test_playground_frontend.py`: serve the built `web/dist` from
a throwaway HTTP server with SPA fallback and drive it in headless chromium.
Every backend call is answered from canned payloads (trimmed real responses
under `fixtures/`, or small literals); a request the stubs do not know fails
the test, so the page's backend surface is explicit here.

Each page must load without a page error or console error. The targeted tests
pin fixed bugs: a share link's inputs reaching the estimate, the drill-downs
reading the estimate's own deployment, the Scaling panel not re-posting on a
metric switch or stalling on a trailing space, the Gallery not reporting a
superseded request as a failure, the Inspector re-reading the hash on each
visit, the Kernel Bench hash carrying its workload filters, the
Providers tab joining the live catalog to measured coverage, and a
still-computing frontier or sweep drawing a progress bar while the page
retries.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from backend.arch.eval.precision import presets as precision_presets
from backend.inference.defaults import COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT
from backend.inference.sweep_grid import (FRONTIER_ACCEPTANCE_DEFAULT,
                                          FRONTIER_ACCEPTANCE_RATES,
                                          FRONTIER_DRAFT_TOKENS, SCALING_CONTEXTS,
                                          SCALING_DEGREES)
from tests.frontend.test_playground_frontend import (
    _DIST, _ensure_built, _serve_dist, AGENTS)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"

_SITE_CONFIG = "window.SITE_CONFIG = { authEnabled: false, supabase: { url: '', anon_key: '' } };"
_GPUS = {"items": [{"sku": "B200", "gpus_per_node": 8}, {"sku": "H100-SXM", "gpus_per_node": 8},
                   {"sku": "A100-40G-PCIe", "gpus_per_node": 4}]}
_MODELS = {"model_ids": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3.2"]}
_PRESETS = {"presets": precision_presets()}
_RETIRED_PRECISION_FIELDS = {
    "act_dtype_bytes", "act_dtype_per_role", "kv_dtype", "kv_quant_granularity",
    "kv_scale_dtype", "weight_dtype_bytes", "weight_quant_granularity",
    "weight_scale_dtype", "weight_dtype_per_role", "compute_dtype_bytes",
}
_SEQ_LENS = 9  # compare.ts SWEEP_SEQ_LENS
_SWEEP_EST = {
    "max_batch_per_gpu": 128,
    "kv_cache": {"per_request_kv_bytes": 7e7},
    "memory": {
        "total_bytes": 180e9, "weight_bytes_per_gpu": 80e9,
        "kv_budget_bytes": 82e9, "fits": True,
    },
    "phases": {
        "prefill": {"t_sol_s": 0.1, "flops_per_step": 1e12, "throughput": {"sol": {"per_gpu": 9000}}},
        "decode": {"t_sol_s": 0.01, "flops_per_step": 1e10, "throughput": {"sol": {"per_gpu": 900}}},
    },
}
_SWEEP = {
    "results_by_batch": {k: [_SWEEP_EST] * _SEQ_LENS for k in ("b1", "b16", "b64", "max")},
    "results": [_SWEEP_EST] * _SEQ_LENS,
}
_PAR_SWEEP = {"by_mode": {}}
# `/inference/estimate-sweep/frontier`: two compared rows at the trade-off
# chart's default context (16K), each a line over two batch points; row 1's
# batch-1 point is on the envelope and on the max-batch front.
_COMPARE_PT = {"context": 16384, "batch_key": "max", "batch": 64, "tpot_s": 0.02,
               "tok_s_per_gpu": 1000.0, "tok_s_per_user": 50.0}
_COMPARE_POINTS = [
    {**_COMPARE_PT, "row": 0},
    {**_COMPARE_PT, "row": 0, "batch_key": "b1", "batch": 1, "tpot_s": 0.012,
     "tok_s_per_gpu": 80.0},
    {**_COMPARE_PT, "row": 1, "tpot_s": 0.01, "tok_s_per_user": 100.0},
    {**_COMPARE_PT, "row": 1, "batch_key": "b1", "batch": 1, "tpot_s": 0.005,
     "tok_s_per_gpu": 200.0},
]
_COMPARE_FRONTS = {
    "points": _COMPARE_POINTS,
    "fronts": {
        "sol": {
            "by_batch": [{"group": {"context": 16384, "batch_key": "max"},
                          "frontier": [_COMPARE_POINTS[2]]}],
            "by_row": [{"group": {"context": 16384, "row": r},
                        "frontier": [p for p in _COMPARE_POINTS if p["row"] == r]}
                       for r in (0, 1)],
            "envelope": [{"group": {"context": 16384},
                          "frontier": [_COMPARE_POINTS[3], _COMPARE_POINTS[2]]}],
        },
        "measured": {"by_batch": [], "by_row": [], "envelope": []},
    },
}
# `/inference/parallel-sweep/frontier`: one card's line through two points
# of its decode cloud, and the values each hold control offers.
_FRONTIER_VIEW_PT = {"tok_s_per_user": 50.0, "tok_s_per_gpu": 4000.0, "tpot_s": 0.02,
                     "batch": 64, "layout": "EP8·DP8", "num_gpus": 8,
                     "gpu_sku": "B200", "mode": "dp_ep", "x": 8, "concurrent_requests": 512,
                     "batch_key": "b64",
                     "parallel": {"dp_attention": 8, "ep_moe": 8}}
_FRONTIER_VIEW = {
    "series": [{"group": {"gpu_sku": "B200"}, "frontier": [
        _FRONTIER_VIEW_PT,
        {**_FRONTIER_VIEW_PT, "tok_s_per_user": 90.0, "tok_s_per_gpu": 900.0,
         "batch": 8, "concurrent_requests": 64, "batch_key": "b8"}]}],
    "values": {"gpu_sku": ["B200"], "mode": ["dp_ep", "tp_tp"],
               "num_gpus": [8], "batch_key": ["b8", "b64"]},
}
_GALLERY_MODEL = {
    "family": "deepseek", "total_params": 671e9, "active_params": 37e9, "is_moe": True,
    "release_date": "2024-12-26",
    "kv_per_request": {
        "as_shipped": {"262144": {"result_bytes": 7e7}, "1048576": {"result_bytes": 2.8e8}},
        "kv_fp8": {"262144": {"result_bytes": 3.5e7}, "1048576": {"result_bytes": 1.4e8}}},
}
# `/gallery/models`: each model's own facts, and the KV leaderboard's title
# and precisions.
_GALLERY = {
    "models": [
        {**_GALLERY_MODEL, "model_id": "deepseek-ai/DeepSeek-V3"},
        {**_GALLERY_MODEL, "model_id": "deepseek-ai/DeepSeek-V3.2", "release_date": "2025-09-29"},
    ],
    "kv_title": "Per-request KV cache",
    "kv_precisions": [{"id": "as_shipped", "label": "As shipped"},
                      {"id": "kv_fp8", "label": "KV FP8"}],
}
_FRONTIER_PT = {"tok_s_per_user": 50.0, "tok_s_per_gpu": 4000.0, "tpot_s": 0.02,
                "batch": 64, "layout": "EP8·DP8", "num_gpus": 8}
_FRONTIER = {
    "seq_lens": [1024],
    "config": {"gpu_sku": "B200", "max_gpus": 8, "output_tokens": 256},
    "models": [{"model_id": "deepseek-ai/DeepSeek-V3", "error": None, "layouts": [],
                "frontier": {"1024": [
                    {**_FRONTIER_PT, "tok_s_per_user": 300.0, "tok_s_per_gpu": 300.0,
                     "batch": 1, "layout": "TP8", "num_gpus": 8},
                    _FRONTIER_PT]}},
               {"model_id": "deepseek-ai/DeepSeek-V3.2", "error": None, "layouts": [],
                "frontier": {"1024": []}}],
}
_CORPUS_ROW = {
    "op": "rmsnorm", "signature": "rmsnorm", "framework": "flashinfer", "framework_version": "0.6.16",
    "kernel_impl": "flashinfer_rmsnorm", "gpu": "B200", "phase": "prefill", "parallel": "tp=1",
    "dtype_env_json": "{\"activation\":\"bf16\"}", "model": "deepseek-ai/DeepSeek-V3",
    "latency_us": 40.0, "achieved_tflops": 0.5, "achieved_bw_gbps": 800.0,
    "peak_tflops_at_test": 2250.0, "peak_bw_at_test": 8000.0, "samples": 60, "date": "2026-08-01",
}
_CORPUS = {"rows": [
    {**_CORPUS_ROW, "shape_json": json.dumps({"B": b, "s_query": s, "d_model": 7168})}
    for b in (1, 16) for s in (1024, 4096)
] + [
    {**_CORPUS_ROW, "op": "gqa_attention", "signature": "gqa_attention",
     "framework": "fa2", "kernel_impl": "flash_attention_v2_kvcache",
     "phase": "decode", "shape_json": json.dumps({"B": 1, "s_query": 1, "s_kv": 4096, "d_head": 128})},
    # A second model + GPU, only under gqa_attention: lets the dropdown test
    # prove options follow the active filters instead of listing the corpus.
    {**_CORPUS_ROW, "op": "gqa_attention", "signature": "gqa_attention",
     "framework": "fa3", "kernel_impl": "flash_attention_v3",
     "gpu": "H100-SXM", "model": "deepseek-ai/DeepSeek-V3.2",
     "phase": "decode", "shape_json": json.dumps({"B": 1, "s_query": 1, "s_kv": 4096, "d_head": 128})},
]}
_PROVIDERS = {
    "kernel_count": 6,
    "declared_count": 5,
    "libraries": [
        {
            "library": "fa2",
            "import_names": ["flash_attn"],
            "provisioning": "pip",
            "pip": "flash-attn>=2.6,<3",
            "build_hint": "",
            "kernels": [
                {
                    "kernel_impl": "flash_attention_v2",
                    "signature": "gqa_attention",
                    "fused_op": "",
                    "min_sm": None,
                    "capabilities": {
                        "signature": "gqa_attention",
                        "weight_dtypes": ["bf16"],
                        "activation_dtypes": ["bf16"],
                        "phases": ["prefill"],
                        "shape_constraints": ["d_head <= 256"],
                    },
                },
                {
                    "kernel_impl": "flash_attention_v2_kvcache",
                    "signature": "gqa_attention",
                    "fused_op": "",
                    "min_sm": None,
                    "capabilities": {
                        "signature": "gqa_attention",
                        "weight_dtypes": ["bf16"],
                        "activation_dtypes": ["bf16"],
                        "phases": ["decode"],
                        "shape_constraints": ["d_head <= 256"],
                    },
                },
            ],
        },
        {
            "library": "flashinfer",
            "import_names": ["flashinfer"],
            "provisioning": "pip",
            "pip": "flashinfer-python>=0.2",
            "build_hint": "",
            "kernels": [
                {
                    "kernel_impl": "flashinfer_rmsnorm",
                    "signature": "rmsnorm",
                    "fused_op": "",
                    "min_sm": None,
                    "capabilities": {
                        "signature": "rmsnorm",
                        "weight_dtypes": ["bf16"],
                        "activation_dtypes": ["bf16"],
                        "phases": [],
                        "shape_constraints": [],
                    },
                },
                {
                    "kernel_impl": "flashinfer_ragged_prefill",
                    "signature": "gqa_attention",
                    "fused_op": "",
                    "min_sm": None,
                    "capabilities": {
                        "signature": "gqa_attention",
                        "weight_dtypes": ["bf16", "fp16"],
                        "activation_dtypes": ["bf16", "fp16"],
                        "phases": ["prefill"],
                        "shape_constraints": ["d_head % 8 == 0"],
                    },
                },
            ],
        },
        {
            "library": "cutlass_mla",
            "import_names": ["sgl_kernel"],
            "provisioning": "source_build",
            "pip": "",
            "build_hint": "Blackwell-only; build sgl-kernel from source",
            "kernels": [
                {
                    "kernel_impl": "cutlass_mla_decode",
                    "signature": "mla_attention",
                    "fused_op": "",
                    "min_sm": 100,
                    "capabilities": {
                        "signature": "mla_attention",
                        "weight_dtypes": ["fp8_e4m3"],
                        "activation_dtypes": ["fp8_e4m3"],
                        "phases": ["decode"],
                        "shape_constraints": [],
                    },
                },
            ],
        },
        {
            "library": "torch",
            "import_names": ["torch"],
            "provisioning": "pip",
            "pip": "torch>=2.5",
            "build_hint": "",
            "kernels": [
                {
                    "kernel_impl": "torch_cublas_linear",
                    "signature": "gemm",
                    "fused_op": "",
                    "min_sm": None,
                    "capabilities": None,
                },
            ],
        },
    ],
}
_GRAPH = {
    "model": "deepseek-ai/DeepSeek-V3", "framework": "vllm", "framework_audited_at": "v0.32.0",
    "phase": "prefill", "parallel": {"tp_attention": 1, "tp_dense_ffn": 1, "ep_moe": 8},
    "graph": {"nodes": [{"id": "x", "op": "input", "label": "x"},
                        {"id": "lm_head", "op": "linear", "label": "lm_head", "inputs": ["x"]}],
              "edges": [{"src": "x", "dst": "lm_head"}]},
}


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def _fake_browse(q: dict) -> dict:
    """/api/bench/browse over the canned corpus, by its contract: filters,
    each dropdown's values under the other filters, one page of the view.
    (The SQL itself is tested in tests/backend/service/test_bench_browse.py.)"""
    from backend.inference.benchmark_identity import workload_coordinates
    names = sorted({k["kernel_impl"] for lib in _PROVIDERS["libraries"] for k in lib["kernels"]},
                   key=lambda n: (-len(n), n))
    library = {k["kernel_impl"]: lib["library"] for lib in _PROVIDERS["libraries"] for k in lib["kernels"]}

    def catalog_kernel(kernel_impl: str) -> str | None:
        return next((n for n in names if kernel_impl.startswith(n)), None)

    rows = []
    for i, r in enumerate(_CORPUS["rows"]):
        shape = json.loads(r["shape_json"])
        b, c = workload_coordinates(r["phase"], shape)
        rows.append({**r, "id": f"r{i}", "origin": "shared", "roof_status": "ok", "shape_json": shape,
                     "dtype_env_json": json.loads(r["dtype_env_json"]),
                     "catalog_kernel": catalog_kernel(r["kernel_impl"]),
                     "workload_batch": b, "workload_context": c})
    value = {"model": lambda r: r["model"], "signature": lambda r: r["signature"],
             "gpu": lambda r: r["gpu"], "framework": lambda r: r["framework"],
             "dtype": lambda r: r["dtype_env_json"].get("activation"),
             "phase": lambda r: r["phase"], "kernel": lambda r: r["catalog_kernel"]}

    def match(r: dict, skip: str = "") -> bool:
        return not any(q.get(d) and d != skip and value[d](r) != q[d] for d in value)

    view = q.get("view", "envelope")
    out: dict = {"facets": {d: sorted({value[d](r) for r in rows if match(r, d)} - {None}) for d in value}}
    hit = sorted((r for r in rows if match(r)), key=lambda r: r["latency_us"])
    if view == "rows":
        return {**out, "total": len(hit), "items": hit[:50]}
    if view == "envelope":
        groups: dict = {}
        for r in hit:
            groups.setdefault((r["signature"], json.dumps(r["shape_json"], sort_keys=True), r["gpu"], r["phase"]), []).append(r)
        items = [{"winner": g[0], "n_candidates": len(g), "gain": None, "models": [g[0]["model"]],
                  "kernels": sorted({c["catalog_kernel"] for c in g if c["catalog_kernel"]}),
                  "key": {"signature": g[0]["signature"], "physical_shape": json.dumps(g[0]["shape_json"]),
                          "dtype_env_json": g[0]["dtype_env_json"], "gpu": g[0]["gpu"],
                          "phase": g[0]["phase"], "parallel": g[0]["parallel"]},
                  "candidates": [{k: c[k] for k in ("id", "framework", "kernel_impl", "catalog_kernel",
                                                    "latency_us", "model")} for c in g]}
                 for g in groups.values()]
        return {**out, "total": len(items), "items": items}
    if view == "workload":
        base = [r for r in rows if match(r)]
        out = {**out, "batches": sorted({r["workload_batch"] for r in base if r["workload_batch"] is not None}),
               "unplaced": sum(r["workload_batch"] is None for r in base)}
        if not q.get("batch"):
            return {**out, "contexts": []}
        at = [r for r in base if r["workload_batch"] == int(q["batch"])]
        out["contexts"] = sorted({r["workload_context"] for r in at}, key=lambda c: (c == 0, c))
        if not q.get("context"):
            return out
        cand = [r for r in at if r["workload_context"] in (int(q["context"]), 0)]
        pick: dict = {}
        for r in sorted(cand, key=lambda r: r["latency_us"]):
            pick.setdefault((r.get("node_spec_id"), r["op"]), r)
        items = sorted(pick.values(), key=lambda r: -r["latency_us"])
        return {**out, "candidates": len(cand), "total": len(items),
                "total_latency_us": sum(r["latency_us"] for r in items), "items": items[:50]}
    if view == "coverage":
        libraries: dict = {lib["library"]: {k["kernel_impl"]: 0 for k in lib["kernels"]}
                           for lib in _PROVIDERS["libraries"]}
        unclassified: dict = {}
        for r in rows:
            if any(q.get(d) and value[d](r) != q[d] for d in ("gpu", "dtype", "phase")):
                continue
            k = r["catalog_kernel"]
            if k:
                libraries[library[k]][k] += 1
            else:
                unclassified[r["framework"]] = unclassified.get(r["framework"], 0) + 1
        return {**out, "libraries": libraries, "unclassified": unclassified}
    # curves: per op, the envelopes with the curve axis taken out of the shape.
    from backend.inference.op_calibration import measured_utilization
    from backend.service.bench.browse import _lines, _Placement, _saturation, call_tokens
    tokens = call_tokens()
    place = _Placement(tokens["runs"], tokens["total"], tokens["factor"])
    best: dict = {}
    for r in hit:
        best.setdefault((r["signature"], json.dumps(r["shape_json"], sort_keys=True), r["gpu"], r["phase"]), r)
    ops: dict = {}
    for r in best.values():
        op = ops.setdefault(r["signature"], {"signature": r["signature"], "points": 0, "unplaced": 0, "curves": {}})
        at = place.place(r["signature"], json.dumps(r["shape_json"]))
        if at is None:
            op["unplaced"] += 1
            continue
        axis, x, rest, *_ = at
        c = op["curves"].setdefault((r["gpu"], axis, json.dumps(rest, sort_keys=True)), {
            "gpu": r["gpu"], "axis": axis,
            "dtype_env_json": r["dtype_env_json"], "shape": rest, "points": []})
        c["points"].append({"x": x, "shape": json.dumps(r["shape_json"]),
                            "phase": r["phase"], "framework": r["framework"], "kernel": r["catalog_kernel"], "model": r["model"],
                            "latency_us": r["latency_us"], "tflops": r["achieved_tflops"],
                            "gbps": r["achieved_bw_gbps"],
                            **{k: measured_utilization(r)[k] for k in ("mfu", "mbu")}})
        op["points"] += 1
    for op in ops.values():
        op["curves"] = sorted(op["curves"].values(), key=lambda c: -len(c["points"]))
        op["curves_total"] = len(op["curves"])
        for c in op["curves"]:
            c["points"].sort(key=lambda pt: pt["x"])
            c["saturation"] = _saturation(c["points"])
            c["lines"] = _lines([(pt["phase"], tuple(json.loads(pt["shape"])[a] for a in c["axis"].split("×")))
                                 for pt in c["points"]])
    # The list names the ops; an op's curves are asked by its signature.
    if not q.get("signature"):
        return {"ops": [{"signature": k} for k in sorted(ops)]}
    return {"ops": [ops[k] for k in sorted(ops) if k == q["signature"]]}


class Refusal:
    """A stub the backend answers with an error status and a JSON body."""

    def __init__(self, status: int, body: dict):
        self.status = status
        self.body = body


class Session:
    """One headless page with every backend path stubbed. Records the request
    bodies / queries per path, console errors, page errors and any path the
    stubs do not answer."""

    def __init__(self, browser, base: str, overrides: dict | None = None):
        self.base = base
        self.requests: dict[str, list] = {}
        self.errors: list[str] = []
        self.unstubbed: list[str] = []
        self.held: dict[str, list] = {}
        self.hold: set[str] = set()
        estimate = _fixture("estimate_deepseek_v3_b200.json")
        self.stubs = {
            "/site-config.js": ("application/javascript", _SITE_CONFIG),
            "/inference/gpus": _GPUS,
            "/supported-models": _MODELS,
            "/inference/defaults": {"scaling_contexts": SCALING_CONTEXTS,
                                    "scaling_degrees": SCALING_DEGREES,
                                    "comm_latency_us": COMM_LATENCY_US_DEFAULT,
                                    "launch_floor_us": LAUNCH_FLOOR_US_DEFAULT},
            "/api/bench/browse": _fake_browse,
            "/precision/presets": _PRESETS,
            "/inference/legal-cp": {"legal_cp_degrees": [1, 2, 4, 8, 16, 32, 64, 128]},
            "/inference/speculative-draft": {
                "available": True, "unavailable_reason": None,
                "draft_stacks": ["mtp"], "styles": ["recursive", "parallel"]},
            "/inference/estimate": estimate,
            "/inference/parallel-shape": {"cluster_size": 8, "per_stage_size": 8, "errors": []},
            # The deployment a link that states none opens at: one GPU, as
            # the form starts — a test that wants a seed stubs its own.
            "/inference/default-deployment": {"parallel": {}, "moe": False,
                                              "start_degree": 1, "fits": True},
            # A text-only model: no media inputs, no encoder.
            "/inference/media-inputs": {"model": "deepseek-ai/DeepSeek-V3",
                                        "inputs": [], "encoders": []},
            "/inference/estimate-sweep": _SWEEP,
            "/inference/estimate-parallel-sweep": _PAR_SWEEP,
            "/inference/parallel-sweep/frontier": _FRONTIER_VIEW,
            "/inference/estimate-sweep/frontier": _COMPARE_FRONTS,
            "/inference/compute-graph": _GRAPH,
            "/gallery/models": _GALLERY,
            "/gallery/frontier": _FRONTIER,
            "/fusion/registry": _fixture("fusion_registry.json"),
            "/dim-glossary": {"items": {}},
            "/inspector/examples": {"examples": ["deepseek-ai/DeepSeek-V3"]},
            "/analyze": {"success": False, "message": "stub: analysis unavailable"},
            "/analytics/event": {},
            # Cookie-auth probes the chrome fires on every page (TopNav's
            # loadAuth → /api/auth/me; Inference, signed in → its calibration
            # corpus). Anonymous with auth off, matching _SITE_CONFIG;
            # signed-in tests override /api/auth/me. Kernel Bench Providers
            # reads /api/bench/providers.
            "/api/auth/me": {"authEnabled": False, "user": None, "profile": None, "isAdmin": False},
            "/api/bench/calibration": {"rows_used": 0, "rows_rejected": 0, "first_rejected": [],
                                       "rejected_reasons": [], "snapshot": 1,
                                       "measurement_scope": "0" * 16},
            "/api/bench/providers": _PROVIDERS,
            # The coding agents' names (Inspector, account pages).
            "/api/agents": AGENTS,
            # The nav's API status asks this on load and every so often.
            "/health": {"status": "ok", "spec_jobs_running": 0},
            **(overrides or {}),
        }
        self.page = browser.new_page()
        self.page.on("console", lambda m: m.type == "error" and self.errors.append(f"console: {m.text}"))
        self.page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        self.page.route("**/*", self._route)

    def _route(self, route):
        url = urlparse(route.request.url)
        path = url.path
        if path in self.stubs:
            req = route.request
            body = None
            if req.method == "POST" and req.post_data:
                try:
                    body = json.loads(req.post_data)
                except ValueError:
                    body = req.post_data
            self.requests.setdefault(path, []).append(
                {"query": {k: v[0] for k, v in parse_qs(url.query).items()}, "body": body})
            if path in self.hold:
                self.held.setdefault(path, []).append(route)
                return
            self.fulfill(route, path)
            return
        rel = path.lstrip("/")
        if path == "/" or path.endswith(".html") or (_DIST / rel).is_file():
            route.continue_()  # an SPA page or a built asset
            return
        self.unstubbed.append(path)
        route.fulfill(status=404, content_type="application/json", body='{"detail":"unstubbed"}')

    def fulfill(self, route, path):
        stub = self.stubs[path]
        if callable(stub):  # answered from the query, like the real endpoint
            query = {k: v[0] for k, v in parse_qs(urlparse(route.request.url).query).items()}
            route.fulfill(status=200, content_type="application/json", body=json.dumps(stub(query)))
        elif isinstance(stub, Refusal):
            route.fulfill(status=stub.status, content_type="application/json",
                          body=json.dumps(stub.body))
        elif isinstance(stub, tuple):
            route.fulfill(status=200, content_type=stub[0], body=stub[1])
        else:
            route.fulfill(status=200, content_type="application/json", body=json.dumps(stub))

    def goto(self, path: str):
        self.page.goto(f"{self.base}{path}")

    def count(self, path: str) -> int:
        return len(self.requests.get(path, []))

    def wait_for(self, predicate, timeout_ms: int = 10000):
        step = 100
        for _ in range(timeout_ms // step):
            if predicate():
                return True
            self.page.wait_for_timeout(step)
        return predicate()

    def assert_clean(self):
        assert not self.unstubbed, f"requests with no stub: {self.unstubbed}"
        assert not self.errors, "page/console errors:\n" + "\n".join(self.errors)


@pytest.fixture(scope="module")
def browser_base():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    _ensure_built()
    if not (_DIST / "index.html").exists():
        pytest.skip("web/dist not built (run `npm run build` in web/)")
    httpd = _serve_dist()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # chromium not installed
                pytest.skip(f"chromium unavailable: {exc}")
            yield browser, base
            browser.close()
    finally:
        httpd.shutdown()


@pytest.mark.parametrize("path, ready", [
    ("/inference.html", "#infer-scaling"),
    ("/", "#model-id"),
    ("/gallery.html", "#g-status"),
    ("/kernel-corpus.html", "#f-source"),
    ("/fusion.html", "main"),
])
def test_page_loads_without_errors(browser_base, path, ready):
    browser, base = browser_base
    s = Session(browser, base)
    s.goto(path)
    s.page.wait_for_selector(ready, timeout=15000)
    s.page.wait_for_timeout(1500)  # let the load-time fetches settle
    s.assert_clean()


def test_inference_cp_picker_lists_backend_legal_degrees(browser_base):
    """The CP control is a select of the legal degrees the backend sent."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html")
    assert s.wait_for(lambda: s.count("/inference/legal-cp") >= 1)
    s.page.wait_for_function(
        "() => document.querySelector('#infer-par-cp')?.tagName === 'SELECT'",
        timeout=15000,
    )
    tag = s.page.eval_on_selector("#infer-par-cp", "el => el.tagName")
    assert tag == "SELECT"
    opts = s.page.eval_on_selector_all(
        "#infer-par-cp option", "els => els.map(e => e.value)")
    assert opts == ["1", "2", "4", "8", "16", "32", "64", "128"]
    s.assert_clean()


def test_inference_cp_with_no_legal_degree_is_unavailable_with_the_reason(browser_base):
    """When the backend offers no legal CP degree (layersplit over a cache
    that mixes growth laws), the control is disabled and says why in the
    backend's words; it is not a free field."""
    reason = ("This model mixes more than one KV cache type, so layersplit "
              "cannot say how much cache each GPU holds. Use ulysses or pcp, "
              "or a model with a single cache type.")
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/inference/legal-cp": {"legal_cp_degrees": [], "unavailable_reason": reason}})
    s.goto("/inference.html")
    s.page.wait_for_selector("#infer-par-cp-scheme", timeout=15000)
    s.page.select_option("#infer-par-cp-scheme", "layersplit")
    assert s.wait_for(lambda: any(
        r["query"].get("cp_scheme") == "layersplit"
        for r in s.requests.get("/inference/legal-cp", [])))
    s.page.wait_for_function(
        "() => document.querySelector('#infer-par-cp')?.disabled === true",
        timeout=15000,
    )
    assert s.page.eval_on_selector("#infer-par-cp", "el => el.tagName") == "SELECT"
    assert s.page.eval_on_selector("#infer-par-cp", "el => el.title") == reason
    shown = s.page.eval_on_selector(
        "#infer-par-cp ~ [data-unavailable-reason]", "el => el.textContent")
    assert shown == reason
    # CP was already 1: nothing was reset, so nothing is announced.
    assert s.page.locator("#infer-par-cp-reset").count() == 0
    s.assert_clean()


def test_inference_cp_reset_to_one_is_announced_with_the_reason(browser_base):
    """A CP > 1 the scheme cannot run on this model is reset to 1, and the
    page says so in the backend's words."""
    reason = ("This model mixes more than one KV cache type, so layersplit "
              "cannot say how much cache each GPU holds. Use ulysses or pcp, "
              "or a model with a single cache type.")
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/inference/legal-cp": {"legal_cp_degrees": [], "unavailable_reason": reason}})
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3.2&gpu=B200&cp=4&cp_scheme=layersplit")
    s.page.wait_for_selector("#infer-par-cp-reset", timeout=15000)
    notice = s.page.text_content("#infer-par-cp-reset")
    assert notice == (
        f"CP reset to 1: layersplit doesn't support CP>1 for this model — {reason}"), notice
    assert s.wait_for(lambda: "cp=1&" in s.page.evaluate("location.hash"))
    s.assert_clean()


def test_inference_draft_cost_is_unavailable_with_the_reason(browser_base):
    """A model with no draft graph: "Count draft cost" is disabled and says
    why in the backend's words."""
    reason = ("this model declares no MTP stack (`aux_kind: mtp`), so there "
              "is no draft graph to price")
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/inference/speculative-draft": {
            "available": False, "unavailable_reason": reason,
            "draft_stacks": [], "styles": ["recursive", "parallel"]}})
    s.goto("/inference.html")
    assert s.wait_for(lambda: s.count("/inference/speculative-draft") >= 1)
    s.page.wait_for_selector("#infer-spec-draft-reason", state="attached", timeout=15000)
    assert s.page.eval_on_selector("#infer-spec-draft-on", "el => el.disabled") is True
    shown = s.page.eval_on_selector("#infer-spec-draft-reason", "el => el.textContent")
    assert reason in shown
    s.assert_clean()


def test_inference_draft_cost_reaches_estimate_and_shows_its_term(browser_base):
    """Ticking "Count draft cost" and picking DFlash sends the style with the
    estimate; the decode tab shows the draft's own term from the response."""
    estimate = _fixture("estimate_deepseek_v3_b200.json")
    decode = estimate["phases"]["decode"]
    draft_rows = estimate["per_op_timing_decode"][:3]
    draft = {
        "style": "parallel", "stacks": ["mtp"], "draft_tokens": 3, "forwards": 1,
        "tokens_per_forward": 3, "flops_per_step": 1e9, "total_bytes": 1e8,
        "t_sol_s": 0.001, "t_measured_s": 0.001, "t_sol_share": 0.042,
        "formula": {"symbolic": "1 draft forward × K tokens/seq through the MTP stack",
                    "substituted": "1 × 1.000e-03 s = 1.000e-03 s"},
    }
    estimate = {**estimate, "per_op_timing_draft": draft_rows,
                "phases": {**estimate["phases"],
                           "decode": {**decode, "speculative_draft": draft}}}
    browser, base = browser_base
    s = Session(browser, base, overrides={"/inference/estimate": estimate})
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&spec_mode=custom&spec_drafts=3")
    assert s.wait_for(lambda: s.count("/inference/speculative-draft") >= 1)
    assert s.requests["/inference/speculative-draft"][-1]["query"]["speculative_draft_tokens"] == "3"
    s.page.click("summary:has-text('Workload')")
    s.page.wait_for_selector("#infer-spec-draft-on:not([disabled])", timeout=15000)
    s.page.check("#infer-spec-draft-on")
    s.page.select_option("#infer-spec-draft-style", "parallel")
    before = s.count("/inference/estimate")
    s.page.click("[data-action=run-estimate]")
    assert s.wait_for(lambda: s.count("/inference/estimate") > before)
    assert s.requests["/inference/estimate"][-1]["body"]["speculative_draft"] == "parallel"
    assert s.requests["/inference/estimate"][-1]["body"]["speculative_draft_tokens"] == 3
    s.page.wait_for_selector("#infer-phase-draft", timeout=15000)
    row = s.page.text_content("#infer-phase-draft")
    assert "parallel draft" in row and "1 × 3 tok/seq" in row and "4.2% of step" in row
    assert s.page.locator("#infer-phase-draft-breakdown").count() == 1
    s.assert_clean()


def test_inference_old_window_links_load_in_their_saved_meaning(browser_base):
    """A link saved when K was the verify window (`spec_k`, draft tokens + 1)
    prices the step it did: `spec_k=4` loads as 3 draft tokens, and the next
    write replaces the retired key with `spec_drafts`. A preset is named, not
    numbered: the MTP preset drafts one token."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&spec_mode=custom&spec_k=4")
    assert s.wait_for(lambda: s.count("/inference/speculative-draft") >= 1)
    assert s.requests["/inference/speculative-draft"][-1]["query"]["speculative_draft_tokens"] == "3"
    s.page.click("summary:has-text('Workload')")
    assert s.page.input_value("#infer-spec-K") == "3"
    assert s.wait_for(lambda: "spec_drafts=3" in s.page.evaluate("location.hash"))
    assert "spec_k=" not in s.page.evaluate("location.hash")
    s.page.select_option("#infer-spec-mode", "mtp")
    before = s.count("/inference/estimate")
    s.page.click("[data-action=run-estimate]")
    assert s.wait_for(lambda: s.count("/inference/estimate") > before)
    body = s.requests["/inference/estimate"][-1]["body"]
    assert body["speculative_draft_tokens"] == 1 and body["acceptance_rate"] == 0.85, body
    assert "speculative_window" not in body
    s.assert_clean()


def test_inference_chunked_prefill_size_reaches_estimate(browser_base):
    """Workload's chunked-prefill size is posted as `prefill_chunk_size`
    so the activation-bound max prefill batch uses min(input, chunk)."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    assert s.requests["/inference/estimate"][0]["body"]["prefill_chunk_size"] == 8192
    s.page.click("summary:has-text('Workload')")
    s.page.fill("#infer-prefill-chunk", "2048")
    assert s.wait_for(lambda: "prefill_chunk=2048" in s.page.evaluate("location.hash"))
    before = s.count("/inference/estimate")
    s.page.click("[data-action=run-estimate]")
    assert s.wait_for(lambda: s.count("/inference/estimate") > before)
    assert s.requests["/inference/estimate"][-1]["body"]["prefill_chunk_size"] == 2048
    s.assert_clean()


def test_inference_routing_imbalance_input_reaches_estimate_and_sweep(browser_base):
    """ρ = `ep_load_factor` is a Parallelism input shown only when this
    model's estimate says it routes; a link's value reaches the estimate, an
    edit reaches the next estimate and the Scaling sweep, and the backend's
    balanced-routing assumption is a chip beside it whose hover shows the
    sentence as the backend wrote it.
    The Scaling sweep waits for the estimate instead of posting a balanced
    sweep first: every parallel sweep it posts carries the link's ρ."""
    browser, base = browser_base
    estimate = {**_fixture("estimate_deepseek_v3_b200.json"), "moe_enabled": True}
    note = estimate["assumptions"][0]["text"]
    s = Session(browser, base, overrides={"/inference/estimate": estimate})
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8&ep_load=1.3")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    assert s.requests["/inference/estimate"][0]["body"]["parallel"]["ep_load_factor"] == 1.3
    s.page.wait_for_selector("#infer-par-ep-load-factor", timeout=15000)
    assert "max/mean expert tokens" in s.page.inner_text("#infer-par-ep-load")
    assert s.page.input_value("#infer-par-ep-load-factor") == "1.3"
    chip = s.page.locator(".infer-par-ep-load-assumption")
    assert chip.get_attribute("aria-label") == note
    chip.hover()
    s.page.wait_for_selector("#ha-tip:not(.hidden)", timeout=5000)
    assert s.page.inner_text("#ha-tip") == note
    assert s.wait_for(lambda: s.count("/inference/estimate-parallel-sweep") >= 1)
    s.page.wait_for_timeout(500)
    assert all((r["body"] or {}).get("ep_load_factor") == 1.3
               for r in s.requests["/inference/estimate-parallel-sweep"])

    s.page.fill("#infer-par-ep-load-factor", "1.5")
    assert s.wait_for(lambda: "ep_load=1.5" in s.page.evaluate("location.hash"))
    before = s.count("/inference/estimate")
    s.page.click("[data-action=run-estimate]")
    assert s.wait_for(lambda: s.count("/inference/estimate") > before)
    assert s.requests["/inference/estimate"][-1]["body"]["parallel"]["ep_load_factor"] == 1.5
    assert s.wait_for(lambda: any(
        (r["body"] or {}).get("ep_load_factor") == 1.5
        for r in s.requests.get("/inference/estimate-parallel-sweep", [])))
    s.assert_clean()

    dense = {**estimate, "moe_enabled": False, "assumptions": []}
    d = Session(browser, base, overrides={"/inference/estimate": dense})
    d.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200")
    assert d.wait_for(lambda: d.count("/inference/estimate") >= 1)
    d.page.wait_for_timeout(500)
    assert d.page.query_selector("#infer-par-ep-load-factor") is None
    d.assert_clean()


def test_inference_routing_imbalance_is_this_models_answer(browser_base):
    """The MoE answer belongs to the model it was estimated for. Switching
    from a routed model to another hides ρ and its note until that model's
    own estimate answers, and no Scaling sweep goes out with ρ meanwhile; a
    dense answer keeps ρ out of every request the panel sends."""
    browser, base = browser_base
    estimate = {**_fixture("estimate_deepseek_v3_b200.json"), "moe_enabled": True}
    s = Session(browser, base, overrides={"/inference/estimate": estimate})
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8&ep_load=1.5")
    s.page.wait_for_selector("#infer-par-ep-load-factor", timeout=15000)

    s.stubs["/inference/estimate"] = {**estimate, "moe_enabled": False, "assumptions": []}
    s.page.fill("#infer-model", "deepseek-ai/DeepSeek-V3.2")
    s.page.wait_for_selector("#infer-par-ep-load-factor", state="detached", timeout=5000)
    assert s.page.query_selector(".infer-par-ep-load-assumption") is None
    swept = s.count("/inference/estimate-parallel-sweep")
    s.page.wait_for_timeout(1500)
    assert s.count("/inference/estimate-parallel-sweep") == swept, \
        "Scaling posted for the new model before its estimate answered"

    s.page.click("[data-action=run-estimate]")
    assert s.wait_for(lambda: s.count("/inference/estimate-parallel-sweep") > swept)
    s.page.wait_for_timeout(500)
    assert s.page.query_selector("#infer-par-ep-load-factor") is None
    for r in s.requests["/inference/estimate-parallel-sweep"][swept:]:
        assert "ep_load_factor" not in (r["body"] or {})
    s.assert_clean()


@pytest.mark.parametrize("raw", ["0.4", "abc"])
def test_inference_invalid_routing_imbalance_is_an_error_not_one(browser_base, raw):
    """A ρ that is not a number ≥ 1 is never read as 1: the link's value stays
    in the URL, the input and a visible error say what is wrong, and neither
    the estimate nor a Scaling sweep is sent until it is fixed."""
    browser, base = browser_base
    estimate = {**_fixture("estimate_deepseek_v3_b200.json"), "moe_enabled": True}
    s = Session(browser, base, overrides={"/inference/estimate": estimate})
    s.goto(f"/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8&ep_load={raw}")
    s.page.wait_for_selector("#infer-par-ep-load-error", timeout=15000)
    assert raw in s.page.inner_text("#infer-par-ep-load-error")
    assert raw in s.page.inner_text("#infer-status")
    s.page.wait_for_timeout(1500)
    assert s.count("/inference/estimate") == 0
    assert s.count("/inference/estimate-parallel-sweep") == 0
    assert f"ep_load={raw}" in s.page.evaluate("location.hash")

    s.page.click("[data-action=run-estimate]")
    s.page.wait_for_timeout(500)
    assert s.count("/inference/estimate") == 0

    s.page.fill("#infer-par-ep-load-factor", "1.2")
    s.page.wait_for_selector("#infer-par-ep-load-error", state="detached", timeout=5000)
    s.page.click("[data-action=run-estimate]")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    assert s.requests["/inference/estimate"][-1]["body"]["parallel"]["ep_load_factor"] == 1.2
    s.assert_clean()


def test_inference_model_seed_waits_for_complete_debounced_id(browser_base):
    """Typing does not fan legal-cp requests out per keystroke, and a
    truncated owner-only id never reaches the backend."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html")
    assert s.wait_for(lambda: s.count("/inference/legal-cp") >= 1)
    initial_cp = s.count("/inference/legal-cp")

    model = s.page.locator("#infer-model")
    model.fill("")
    model.press_sequentially("glm-", delay=40)
    s.page.wait_for_timeout(600)
    assert s.count("/inference/legal-cp") == initial_cp

    model.fill("")
    model.press_sequentially("deepseek-ai/DeepSeek-V3.2", delay=20)
    s.page.wait_for_timeout(100)
    assert s.count("/inference/legal-cp") == initial_cp
    assert s.wait_for(lambda: s.count("/inference/legal-cp") == initial_cp + 1)
    assert s.requests["/inference/legal-cp"][-1]["query"]["model_id"] == "deepseek-ai/DeepSeek-V3.2"
    s.assert_clean()


def test_inference_signed_in_asks_the_server_to_calibrate_with_ones_rows(browser_base):
    """Signed in, the page never downloads the caller's rows: the estimate
    asks the server to calibrate with them (`calibrate_with_mine`), and the
    page says how many of them calibrate it and how many were rejected, why."""
    browser, base = browser_base
    s = Session(browser, base, {
        "/api/auth/me": {"authEnabled": True, "user": {"id": "u1", "email": "a@example.com",
                                                       "user_metadata": {}},
                         "profile": {"username": "a"}, "isAdmin": False},
        "/api/bench/calibration": {
            "rows_used": 71174, "rows_rejected": 9882,
            "first_rejected": [{"id": "r1", "reason": "unknown symbol 'H'"}],
            "rejected_reasons": [{"reason": "column `shape_json` has unknown symbol 'H'",
                                  "rows": 1584}],
            "snapshot": 7, "measurement_scope": "a" * 16},
    })
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200")
    note = s.page.locator("#infer-calibration")
    assert s.wait_for(lambda: "71,174 of your rows" in note.inner_text()), note.inner_text()
    assert "9,882 rejected" in note.inner_text()
    s.page.click("#infer-calibration summary")
    assert "unknown symbol 'H'" in note.inner_text() and "1,584" in note.inner_text()
    # The link's estimate ran once the calibration was known.
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    body = s.requests["/inference/estimate"][0]["body"]
    assert body["calibrate_with_mine"] is True and "calibration_rows" not in body, body
    assert s.count("/api/bench/rows") == 0
    s.assert_clean()


_SIGNED_IN = {
    "/api/auth/me": {"authEnabled": True, "user": {"id": "u1", "email": "a@example.com",
                                                   "user_metadata": {}},
                     "profile": {"username": "a"}, "isAdmin": False},
    "/api/bench/calibration": {
        "rows_used": 12, "rows_rejected": 0, "first_rejected": [], "rejected_reasons": [],
        "snapshot": 7, "measurement_scope": "b" * 16},
}


def _sweep_link() -> str:
    import base64
    state = {"template": {"gpu_sku": "B200"}, "cfgs": [{"gpu_sku": "B200"}, {"gpu_sku": "B200"}]}
    enc = base64.urlsafe_b64encode(json.dumps(state).encode()).decode().rstrip("=")
    return ("/inference.html#sub=sweep&m=deepseek-ai%2FDeepSeek-V3%7Cdeepseek-ai%2FDeepSeek-V3.2"
            f"&s={enc}")


def test_decode_frontier_is_a_view_the_backend_draws_of_the_sweep(browser_base):
    """The Scaling panel's decode Pareto frontier asks the backend for a view
    of the parallel sweep it shows (the same body, so the same cache entry):
    a hold control adds each ticked value to `fix` (several of one dimension
    at once), the lines control picks `group_by`; the page draws what comes
    back and computes no front."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8")
    view = "/inference/parallel-sweep/frontier"
    assert s.wait_for(lambda: s.count(view) >= 1)
    first = s.requests[view][0]["body"]
    # The panel's sweep body; the server reads it as the SOL answer.
    posted = s.requests["/inference/estimate-parallel-sweep"][-1]["body"]
    assert {**first["sweep"], "sol_only": True} == {**posted, "sol_only": True}
    assert first["fix"] == {} and first["group_by"] == ["gpu_sku"]

    def tick(control: str, value: str) -> None:
        box = f"#{control} input[value='{value}']"
        if not s.page.is_visible(box):
            s.page.click(f"#{control} summary")
        s.page.check(box)

    s.page.wait_for_function(
        "document.querySelectorAll('#infer-pareto-fix-mode input').length >= 2",
        timeout=10000)
    modes = s.page.eval_on_selector_all("#infer-pareto-fix-mode input",
                                        "els => els.map(e => e.value)")
    tick("infer-pareto-fix-mode", modes[0])
    assert s.wait_for(lambda: s.requests[view][-1]["body"]["fix"] == {"mode": modes[:1]})
    # A second value of the same dimension joins the first.
    tick("infer-pareto-fix-mode", modes[1])
    assert s.wait_for(lambda: s.requests[view][-1]["body"]["fix"] == {"mode": modes[:2]})
    gpus = s.page.eval_on_selector_all("#infer-pareto-fix-num_gpus input",
                                       "els => els.map(e => e.value)")
    tick("infer-pareto-fix-num_gpus", gpus[0])
    assert s.wait_for(lambda: s.requests[view][-1]["body"]["fix"]
                      == {"mode": modes[:2], "num_gpus": [int(gpus[0])]})
    tick("infer-pareto-lines", "batch_key")
    assert s.wait_for(lambda: s.requests[view][-1]["body"]["group_by"]
                      == ["gpu_sku", "batch_key"])
    tick("infer-pareto-lines", "mode")
    assert s.wait_for(lambda: s.requests[view][-1]["body"]["group_by"]
                      == ["gpu_sku", "mode", "batch_key"])
    s.assert_clean()


@pytest.mark.parametrize("signed_in", [True, False])
def test_scaling_and_compare_sweeps_calibrate_like_the_estimate(browser_base, signed_in):
    """The Scaling panels' sweeps and the Compare sweep ask the server to
    calibrate with the caller's rows exactly when the estimate does: signed
    in, every body carries `calibrate_with_mine`; signed out, none does."""
    browser, base = browser_base
    s = Session(browser, base, _SIGNED_IN if signed_in else None)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8")
    assert s.wait_for(lambda: s.count("/inference/estimate-sweep") >= 1)
    assert s.wait_for(lambda: s.count("/inference/estimate-parallel-sweep") >= 1)
    for path in ("/inference/estimate", "/inference/estimate-sweep",
                 "/inference/estimate-parallel-sweep"):
        for r in s.requests[path]:
            assert r["body"].get("calibrate_with_mine") is (True if signed_in else None), \
                (path, r["body"])
            assert "calibration_rows" not in r["body"], (path, r["body"])
    s.assert_clean()
    s.page.close()

    s = Session(browser, base, _SIGNED_IN if signed_in else None)
    s.goto(_sweep_link())
    if signed_in:
        assert s.wait_for(lambda: s.count("/api/bench/calibration") >= 1)
        s.page.wait_for_timeout(300)
    s.page.click("#ic-run", timeout=15000)
    assert s.wait_for(lambda: s.count("/inference/estimate-sweep") >= 2)
    for r in s.requests["/inference/estimate-sweep"]:
        assert r["body"].get("calibrate_with_mine") is (True if signed_in else None), r["body"]
    # The trade-off's fronts read the rows' sweeps as they were posted.
    assert s.wait_for(lambda: s.count("/inference/estimate-sweep/frontier") >= 1)
    assert s.requests["/inference/estimate-sweep/frontier"][-1]["body"]["sweeps"] == [
        r["body"] for r in s.requests["/inference/estimate-sweep"]]
    # A chart draws once it is near the viewport (`useNearViewport`).
    s.page.locator("#ic-chart-scatter").scroll_into_view_if_needed()
    s.page.wait_for_selector("#ic-chart-scatter canvas", timeout=10000)
    # One line per row over its batch points by default; a batch point
    # draws the rows there, with the baseline control.
    assert s.page.input_value("#ic-scatter-batch") == "all"
    assert s.page.locator("#ic-scatter-baseline").count() == 0
    s.page.select_option("#ic-scatter-batch", "max")
    s.page.wait_for_selector("#ic-scatter-baseline", timeout=5000)
    s.page.wait_for_selector("#ic-chart-scatter canvas", timeout=5000)
    s.assert_clean()


def test_inference_signed_out_sends_no_calibration(browser_base):
    """Signed out: no corpus to read, nothing asked of the server; the page
    says signing in would calibrate with one's own rows."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    body = s.requests["/inference/estimate"][-1]["body"]
    assert "calibrate_with_mine" not in body and "calibration_rows" not in body, body
    assert s.count("/api/bench/calibration") == 0 and s.count("/api/bench/rows") == 0
    assert "Sign in to calibrate" in s.page.locator("#infer-calibration").inner_text()
    s.assert_clean()


def test_inference_share_link_reaches_estimate_and_drilldowns(browser_base):
    """The link's calibration inputs reach the estimate; the compute graph is
    drawn for the estimate's deployment and framework, not the live form. The
    link is an old one: its `framework_version` (bindings are not versioned
    any more) is read and ignored, its `selection=framework_version` is the
    framework's measured kernels, `latest`, and its precision keys predate
    precision profiles — they leave the URL unapplied, the page says so, and
    the estimate is as shipped."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
           "&w_dtype=1&w_gran=per-channel&compute_dtype=1&kv_dtype=fp8&selection=framework_version"
           "&framework=vllm&framework_version=0.21.0")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1), "link did not auto-run the estimate"
    body = s.requests["/inference/estimate"][0]["body"]
    assert not _RETIRED_PRECISION_FIELDS & set(body), body
    assert "precision" not in body, body
    assert body["selection_mode"] == "latest"
    assert body["framework"] == "vllm" and "framework_version" not in body
    assert body["parallel"]["ep_moe"] == 8

    notice = s.page.locator("#infer-precision [data-precision-notice]")
    notice.wait_for(timeout=10000)
    assert "predate precision profiles" in notice.inner_text()
    h = parse_qs(urlparse(s.page.url).fragment)
    for k in ("w_dtype", "w_gran", "compute_dtype", "kv_dtype", "prec"):
        assert k not in h, h
    assert h.get("framework") == ["vllm"] and h.get("selection") == ["latest"], h
    assert "framework_version" not in h, h

    # Change EP in the live form without re-estimating: the graph still draws
    # the deployment the numbers on screen were computed for.
    s.page.wait_for_selector("#infer-graph-refresh", state="attached", timeout=15000)
    s.page.evaluate("document.querySelectorAll('#infer-results details').forEach(d => d.open = true)")
    s.page.click("#infer-graph-refresh")
    assert s.wait_for(lambda: s.count("/inference/compute-graph") >= 1)
    q = s.requests["/inference/compute-graph"][-1]["query"]
    assert q["framework"] == "vllm" and "framework_version" not in q, q
    assert q["ep_moe"] == "8", q
    s.assert_clean()


def test_inference_precision_preset_reaches_every_request(browser_base):
    """A `prec=` link's preset is the page's profile: the estimate, both
    Scaling sweeps and the compute graph carry it, and the response's label
    is shown. Picking As shipped drops it from the URL and the requests."""
    browser, base = browser_base
    estimate = {**_fixture("estimate_deepseek_v3_b200.json"), "precision": {
        "profile_key": "k", "label": "KV FP8", "preset": "kv_fp8", "canonical": {},
        "overridden": True, "inapplicable": [],
        "effective": [
            {"category": "cache", "selector": "latent", "as_shipped": "BF16",
             "effective": "FP8 E4M3 · per-tensor F32", "source": "profile", "changed": True},
            {"category": "weights", "selector": "weight_class", "as_shipped": "FP8 block128",
             "effective": "FP8 block128", "source": "as shipped", "changed": False},
        ]}}
    s = Session(browser, base, overrides={"/inference/estimate": estimate})
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8&prec=kv_fp8")
    want = {"preset": "kv_fp8"}
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    assert s.requests["/inference/estimate"][0]["body"]["precision"] == want
    assert s.wait_for(lambda: s.count("/inference/estimate-sweep") >= 1)
    assert s.wait_for(lambda: s.count("/inference/estimate-parallel-sweep") >= 1)
    for path in ("/inference/estimate-sweep", "/inference/estimate-parallel-sweep"):
        for r in s.requests[path]:
            assert r["body"]["precision"] == want, (path, r["body"])
            assert not _RETIRED_PRECISION_FIELDS & set(r["body"]), (path, r["body"])
    applied = s.page.locator("#infer-precision-applied")
    applied.wait_for(timeout=10000)
    text = applied.inner_text()
    assert "KV FP8" in text and "latent: BF16 → FP8 E4M3 · per-tensor F32" in text, text
    assert "weight_class" not in text, text
    assert s.page.get_attribute("#infer-precision [data-preset=kv_fp8]", "aria-pressed") == "true"

    s.page.evaluate("document.querySelectorAll('#infer-results details').forEach(d => d.open = true)")
    s.page.click("#infer-graph-refresh")
    assert s.wait_for(lambda: s.count("/inference/compute-graph") >= 1)
    assert json.loads(s.requests["/inference/compute-graph"][-1]["query"]["precision"]) == want

    s.page.click("#infer-precision [data-preset=as_shipped]")
    assert s.wait_for(lambda: "prec=" not in s.page.evaluate("location.hash"))
    s.page.click("[data-action=run-estimate]")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 2)
    assert "precision" not in s.requests["/inference/estimate"][-1]["body"]
    s.assert_clean()


def test_inference_refused_preset_shows_the_reason_instead_of_numbers(browser_base):
    """A profile the GPU cannot run is refused by the backend; its reason
    stands where the estimate's numbers would be."""
    browser, base = browser_base
    why = "H100-SXM has no peak-FLOPs column for FP4_E2M1"
    s = Session(browser, base, overrides={
        "/inference/estimate": Refusal(422, {"detail": why}),
    })
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=H100-SXM&prec=w_nvfp4_w4a4_kv_fp8")
    refused = s.page.locator("#infer-estimate-refused")
    refused.wait_for(timeout=15000)
    assert why in refused.inner_text()
    assert s.page.locator("#infer-results").is_hidden()
    s.errors = [e for e in s.errors if "422" not in e]  # the refusal itself
    s.assert_clean()


def test_inference_graph_is_analysed_at_the_estimates_step(browser_base):
    """The compute graph query carries the step of the estimate on screen —
    its input / output length and each phase's batch as the estimate
    reports them — not the live form's length."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
           "&framework=vllm&selection=latest")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    est = _fixture("estimate_deepseek_v3_b200.json")
    s.page.wait_for_selector("#infer-graph-refresh", state="attached", timeout=15000)
    s.page.evaluate("document.querySelectorAll('#infer-results details').forEach(d => d.open = true)")
    s.page.fill("#infer-avg-in", "4096")  # live form only; the estimate stays
    s.page.click("#infer-graph-refresh")
    assert s.wait_for(lambda: s.count("/inference/compute-graph") >= 1)
    q = s.requests["/inference/compute-graph"][-1]["query"]
    assert q["phase"] == "prefill", q
    assert q["avg_input_tokens"] == str(est["avg_input_tokens"]), q
    assert q["avg_output_tokens"] == str(est["avg_output_tokens"]), q
    assert q["batch_size"] == str(est["phases"]["prefill"]["batch"]), q
    assert q["prefix_cached_tokens"] == "0" and q["speculative_draft_tokens"] == "0", q
    assert q["cp_scheme"] == "pcp", q

    s.page.click("#infer-graph-phase-decode")
    s.page.click("#infer-graph-refresh")
    assert s.wait_for(lambda: s.requests["/inference/compute-graph"][-1]["query"]["phase"] == "decode")
    q = s.requests["/inference/compute-graph"][-1]["query"]
    assert q["batch_size"] == str(est["phases"]["decode"]["batch"]), q
    assert q["avg_input_tokens"] == str(est["avg_input_tokens"]), q
    s.assert_clean()


def test_inference_scaling_ngpu_dropdown_reposts_sweep(browser_base):
    """Context × Batch GPU-count is a dropdown; changing it re-posts the sweep
    with that degree. Default stays 8 (AUDIT_PARALLEL)."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html")
    assert s.wait_for(lambda: s.count("/inference/estimate-sweep") >= 1), "Scaling sweep never fired"
    s.page.wait_for_selector("#infer-scaling-ngpu", timeout=15000)
    opts = s.page.eval_on_selector_all(
        "#infer-scaling-ngpu option", "els => els.map(e => e.value)")
    assert opts == ["1", "2", "4", "8", "16", "32", "64", "128"]
    assert s.page.input_value("#infer-scaling-ngpu") == "8"
    first = s.requests["/inference/estimate-sweep"][-1]["body"] or {}
    assert (first.get("parallel_preset") or {}).get("degree") == 8, first
    s.page.select_option("#infer-scaling-ngpu", "16")
    assert s.wait_for(lambda: any(
        (r["body"] or {}).get("parallel_preset", {}).get("degree") == 16
        for r in s.requests["/inference/estimate-sweep"]
    )), "changing nGPU did not re-post the sweep at degree 16"
    s.assert_clean()


def test_inference_scaling_ngpu_hides_infeasible_degrees(browser_base):
    """Once the parallel sweep lands, GPU-count options are the feasible
    degrees on that SKU — infeasible rungs drop off the dropdown."""
    feasible = {4, 8, 16}
    points = [
        {"x": x, "feasible": x in feasible,
         "prefill_tps_per_gpu": 1000 if x in feasible else None,
         "decode_tps_per_gpu": 100 if x in feasible else None}
        for x in (2, 4, 8, 16, 32, 64, 128)
    ]
    blob = {"moe_enabled": True, "series": [{"gpu_sku": "B200", "points": points}]}
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/inference/estimate-parallel-sweep": {
            "by_mode": {m: blob for m in ("dp_ep", "cp_ep", "tp_ep", "tp_tp")},
        },
    })
    s.goto("/inference.html")
    s.page.wait_for_selector("#infer-scaling-ngpu", timeout=15000)
    assert s.wait_for(lambda: s.count("/inference/estimate-parallel-sweep") >= 1)
    assert s.wait_for(lambda: s.page.eval_on_selector_all(
        "#infer-scaling-ngpu option", "els => els.map(e => e.value)") == ["4", "8", "16"]), (
        s.page.eval_on_selector_all("#infer-scaling-ngpu option", "els => els.map(e => e.value)"))
    assert s.page.input_value("#infer-scaling-ngpu") == "8"
    s.assert_clean()


def test_inference_scaling_ngpu_says_why_no_degree_fits(browser_base):
    """A swept SKU with no feasible degree offers no GPU count and says the
    backend's reason, instead of offering a default rung that does not fit."""
    reason = "weights (900.00 GiB) exceed usable HBM."
    points = [{"x": x, "feasible": False, "reason": reason,
               "prefill_tps_per_gpu": None, "decode_tps_per_gpu": None}
              for x in (2, 4, 8, 16)]
    blob = {"moe_enabled": True, "series": [{"gpu_sku": "B200", "points": points}]}
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/inference/estimate-parallel-sweep": {
            "by_mode": {m: blob for m in ("dp_ep", "cp_ep", "tp_ep", "tp_tp")},
        },
    })
    s.goto("/inference.html")
    s.page.wait_for_selector("#infer-scaling-ngpu", timeout=15000)
    assert s.wait_for(lambda: s.count("/inference/estimate-parallel-sweep") >= 1)
    assert s.wait_for(lambda: s.page.eval_on_selector_all(
        "#infer-scaling-ngpu option", "els => els.map(e => e.value)") == [])
    assert s.wait_for(lambda: reason in s.page.inner_text("#infer-scaling"))
    s.assert_clean()


def test_inference_scaling_ctx_options_are_the_backend_grid(browser_base):
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html")
    s.page.wait_for_selector("#infer-scaling-ctx", timeout=15000)
    assert s.wait_for(lambda: s.page.eval_on_selector_all(
        "#infer-scaling-ctx option", "els => els.map(e => Number(e.value))") == SCALING_CONTEXTS)
    s.assert_clean()


def test_inference_scaling_measured_line_breaks_where_unmeasured(browser_base):
    """The measured track never borrows the SOL value: at a degree with no
    measured rate the hover lists the SOL line only."""
    def pt(x, meas):
        # A measured rate is drawn only at complete coverage (`measured.ts`).
        cov = {"measured": 10, "total": 10} if meas is not None else {"measured": 0, "total": 10}
        return {"x": x, "feasible": True, "prefill_tps_per_gpu": 1000, "decode_tps_per_gpu": 100,
                "prefill_tps_per_gpu_measured": meas, "decode_tps_per_gpu_measured": meas,
                "prefill_measured_coverage": cov, "decode_measured_coverage": cov}
    blob = {"moe_enabled": True, "series": [
        {"gpu_sku": "B200", "points": [pt(2, 80), pt(4, None), pt(8, 60)]}]}
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/inference/estimate-parallel-sweep": {
            "by_mode": {m: blob for m in ("dp_ep", "cp_ep", "tp_ep", "tp_tp")},
        },
    })
    s.goto("/inference.html")
    s.page.wait_for_selector("#infer-par-measured", timeout=15000)
    s.page.check("#infer-par-measured")
    s.page.wait_for_selector("#infer-scaling canvas", timeout=15000)
    s.page.wait_for_timeout(500)
    canvas = s.page.query_selector_all("#infer-scaling canvas")[0]
    box = canvas.bounding_box()
    # Grid is left 56 / right 20; three categories, boundaryGap false → x=4
    # sits mid-grid.
    x_of = lambda i: box["x"] + 56 + (box["width"] - 76) * i / 2
    y = box["y"] + box["height"] / 2

    def tooltip_at(i):
        s.page.mouse.move(x_of(i), y)
        expected = f"EP = {[2, 4, 8][i]}"
        assert s.wait_for(lambda: s.page.evaluate(
            "() => [...document.querySelectorAll('#infer-scaling div')].some(d => "
            "d.style && d.style.position === 'absolute' "
            f"&& d.innerText.includes('{expected}'))"))
        return s.page.evaluate(
            "() => [...document.querySelectorAll('#infer-scaling div')]"
            ".filter(d => d.style && d.style.position === 'absolute' && /EP = /.test(d.innerText))"
            ".map(d => d.innerText).join('\\n')")

    measured_at_2 = tooltip_at(0)
    assert "B200 · measured" in measured_at_2, measured_at_2
    unmeasured_at_4 = tooltip_at(1)
    assert "EP = 4" in unmeasured_at_4, unmeasured_at_4
    assert "B200 · measured" not in unmeasured_at_4, unmeasured_at_4
    s.assert_clean()


def test_inference_scaling_dropdown_roundtrip_hits_frontend_cache(browser_base):
    """Switching Context × Batch nGPU away and back must not re-POST: the
    panel caches a theory sweep's JSON by request body (a measured one is
    asked again, `test_scaling_tab_cache_holds_theory_only`)."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#selection=theoretical_only")
    assert s.wait_for(lambda: s.count("/inference/estimate-sweep") >= 1), "Scaling sweep never fired"
    s.page.wait_for_selector("#infer-scaling-ngpu", timeout=15000)
    s.page.select_option("#infer-scaling-ngpu", "16")
    assert s.wait_for(lambda: any(
        (r["body"] or {}).get("parallel_preset", {}).get("degree") == 16
        for r in s.requests["/inference/estimate-sweep"]
    )), "changing nGPU did not re-post the sweep at degree 16"
    before = s.count("/inference/estimate-sweep")
    s.page.select_option("#infer-scaling-ngpu", "8")
    s.page.wait_for_timeout(800)
    assert s.count("/inference/estimate-sweep") == before, (
        "switching nGPU back to 8 re-posted; frontend cache missed")
    s.assert_clean()


def test_inference_scaling_measured_toggle_asks_only_the_measured_answer(browser_base):
    """A sweep's SOL answer is drawn first (`sol_only`); ticking Measured asks
    each sweep's measured answer once — the same body without `sol_only` —
    and re-posts no SOL body (the tab holds it). Unticking asks nothing."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html")
    assert s.wait_for(lambda: s.count("/inference/estimate-sweep") >= 1), "Scaling sweep never fired"
    s.page.wait_for_selector("#infer-scaling-measured", timeout=15000)
    s.page.wait_for_timeout(800)
    paths = ("/inference/estimate-sweep", "/inference/estimate-parallel-sweep")
    bodies = lambda p: [r["body"] for r in s.requests.get(p, [])]
    for p in paths:
        assert bodies(p) and all(b.get("sol_only") is True for b in bodies(p)), p
    before = {p: s.count(p) for p in paths}
    s.page.check("#infer-scaling-measured")
    s.page.check("#infer-par-measured")
    s.page.wait_for_timeout(1000)
    for p in paths:
        new = bodies(p)[before[p]:]
        assert len(new) == 1 and "sol_only" not in new[0], (p, new)
        assert {**new[0], "sol_only": True} == bodies(p)[before[p] - 1], p
    after = {p: s.count(p) for p in paths}
    s.page.uncheck("#infer-scaling-measured")
    s.page.uncheck("#infer-par-measured")
    s.page.wait_for_timeout(1000)
    assert {p: s.count(p) for p in paths} == after, "unticking asked again"
    s.assert_clean()


def test_inference_scaling_kv_storage_shares_batch_sweep(browser_base):
    """KV storage has no own dropdowns or POST; it reads the Context × Batch
    estimate-sweep (batch_grid) under the same GPU / parallel / nGPU."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html")
    s.page.wait_for_selector("#infer-scaling-kv", timeout=15000)
    s.page.wait_for_selector("#infer-frac", timeout=15000)
    assert s.page.eval_on_selector_all(
        "#infer-frac option", "els => els.map(e => e.value)") == [
        "0.95", "0.9", "0.85", "0.8", "0.7", "0.6", "0.5"]
    assert s.page.query_selector("#infer-scaling-frac") is None
    assert s.page.query_selector("#infer-scaling-kv-gpu") is None
    assert s.page.query_selector("#infer-scaling-kv-ngpu") is None
    assert s.page.query_selector("#infer-scaling-kv-par") is None
    assert s.wait_for(lambda: s.count("/inference/estimate-sweep") >= 1)
    for r in s.requests["/inference/estimate-sweep"]:
        body = r.get("body") or {}
        if isinstance(body, dict):
            assert "batch_grid" in body, body
    s.assert_clean()


def test_inference_scaling_fires_for_model_with_trailing_space(browser_base):
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html")
    assert s.wait_for(lambda: s.count("/inference/estimate-sweep") >= 1)
    s.page.fill("#infer-model", "deepseek-ai/DeepSeek-V3.2 ")
    assert s.wait_for(lambda: any(
        (r["body"] or {}).get("model_id") == "deepseek-ai/DeepSeek-V3.2"
        for r in s.requests["/inference/estimate-sweep"])), "Scaling stayed gated on the untrimmed id"
    s.assert_clean()


def test_gallery_superseded_request_is_not_a_failure(browser_base):
    browser, base = browser_base
    s = Session(browser, base)
    s.hold = {"/gallery/frontier"}
    s.goto("/gallery.html")
    assert s.wait_for(lambda: s.count("/gallery/frontier") >= 1)
    # Another GPU than the landing one supersedes the held request (the
    # default is the gallery's, so ask the page rather than naming one).
    other = s.page.eval_on_selector(
        "#g-gpu", "el => [...el.options].map(o => o.value).find(v => v !== el.value)")
    s.page.select_option("#g-gpu", other)
    assert s.wait_for(lambda: s.count("/gallery/frontier") >= 2)
    s.page.wait_for_timeout(300)
    status = s.page.inner_text("#g-status")
    assert "Failed" not in status, status
    s.hold = set()
    for route in s.held.pop("/gallery/frontier"):
        try:
            s.fulfill(route, "/gallery/frontier")
        except Exception:  # the superseded request was aborted by the page
            pass
    assert s.wait_for(lambda: "models" in s.page.inner_text("#g-status"))
    assert "Per-request KV cache" in s.page.inner_text("main")
    # The KV leaderboard switches between the cache-only presets.
    assert "bg-gray-900" in s.page.get_attribute("#g-kv-prec-as_shipped", "class")
    s.page.click("#g-kv-prec-kv_fp8")
    assert "bg-gray-900" in s.page.get_attribute("#g-kv-prec-kv_fp8", "class")
    s.assert_clean()


def test_a_gallery_link_from_before_the_inspector_took_root_lands_on_the_gallery(browser_base):
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/#gpu=H100-SXM&ctx=4096")
    s.page.wait_for_selector("#g-status", timeout=15000)
    assert s.page.evaluate("location.pathname") == "/gallery.html"
    assert "gpu=H100-SXM" in s.page.evaluate("location.hash")


def test_gallery_gpu_picker_lists_the_gallery_series_only(browser_base):
    """The picker offers `axes.json`'s GPUs, not the whole spec table: the
    frontier route refuses another SKU."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/gallery.html")
    def skus():
        return s.page.eval_on_selector_all("#g-gpu option", "els => els.map(e => e.value)")
    assert s.wait_for(lambda: skus() == ["B200", "H100-SXM"]), skus()
    s.assert_clean()


def test_inference_floors_seed_from_defaults_and_a_link_sets_its_own(browser_base):
    """The two time floors start at the server's defaults and reach the
    estimate; a link that states its own sends those."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    body = s.requests["/inference/estimate"][-1]["body"]
    assert body["comm_latency_us"] == COMM_LATENCY_US_DEFAULT
    assert body["launch_floor_us"] == LAUNCH_FLOOR_US_DEFAULT
    s.assert_clean()
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&comm_lat=5&launch_floor=0.5")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    body = s.requests["/inference/estimate"][-1]["body"]
    assert (body["comm_latency_us"], body["launch_floor_us"]) == (5, 0.5)
    s.assert_clean()


def test_gallery_frontier_is_asked_under_the_floors_and_an_edit_reasks(browser_base):
    """The frontier is asked under the defaults' floors; an edit applies on
    Enter and asks again under the new one."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/gallery.html")
    assert s.wait_for(lambda: s.count("/gallery/frontier") >= 1)
    q = s.requests["/gallery/frontier"][-1]["query"]
    assert (float(q["comm_latency_us"]), float(q["launch_floor_us"])) == (
        COMM_LATENCY_US_DEFAULT, LAUNCH_FLOOR_US_DEFAULT)
    before = s.count("/gallery/frontier")
    s.page.fill("#g-comm-latency", "5")
    s.page.press("#g-comm-latency", "Enter")
    assert s.wait_for(lambda: s.count("/gallery/frontier") > before)
    assert float(s.requests["/gallery/frontier"][-1]["query"]["comm_latency_us"]) == 5
    s.assert_clean()


def test_gallery_speculative_switch_reasks_the_frontier_under_its_draft(browser_base):
    """The MTP switch asks the frontier again with the draft length and
    acceptance; another K asks again; switched off, neither is sent. A
    model with no MTP draft is named under the chart, not drawn; a layout
    whose speculative step failed is named with why."""
    browser, base = browser_base
    no_draft = {**_FRONTIER, "models": [
        {**_FRONTIER["models"][0], "draft_errors": [
            {"layout": "TP8", "draft_tokens": 1, "error": "verify refused"}]},
        {**_FRONTIER["models"][1], "draft_unavailable": "no MTP stack"}]}
    s = Session(browser, base, overrides={"/gallery/frontier": no_draft})
    s.goto("/gallery.html")
    assert s.wait_for(lambda: s.count("/gallery/frontier") >= 1)
    s.page.wait_for_selector("#g-dec-tab-1024", timeout=15000)
    assert s.page.query_selector("#g-spec-k") is None
    before = s.count("/gallery/frontier")
    s.page.check("#g-spec-on")
    assert s.wait_for(lambda: s.count("/gallery/frontier") > before)
    q = s.requests["/gallery/frontier"][-1]["query"]
    assert int(q["draft_tokens"]) == FRONTIER_DRAFT_TOKENS[0]
    assert float(q["acceptance_rate"]) == FRONTIER_ACCEPTANCE_DEFAULT
    s.page.wait_for_selector("#g-front-no-draft", timeout=15000)
    assert "DeepSeek-V3.2" in s.page.inner_text("#g-front-no-draft")
    errors = s.page.inner_text("#g-front-draft-errors")
    assert "TP8" in errors and "verify refused" in errors
    assert "one link latency" in s.page.inner_text("#g-front-footnote")
    before = s.count("/gallery/frontier")
    s.page.select_option("#g-spec-k", str(FRONTIER_DRAFT_TOKENS[-1]))
    assert s.wait_for(lambda: s.count("/gallery/frontier") > before)
    assert int(s.requests["/gallery/frontier"][-1]["query"]["draft_tokens"]) == (
        FRONTIER_DRAFT_TOKENS[-1])
    before = s.count("/gallery/frontier")
    s.page.select_option("#g-spec-alpha", str(FRONTIER_ACCEPTANCE_RATES[0]))
    assert s.wait_for(lambda: s.count("/gallery/frontier") > before)
    assert float(s.requests["/gallery/frontier"][-1]["query"]["acceptance_rate"]) == (
        FRONTIER_ACCEPTANCE_RATES[0])
    before = s.count("/gallery/frontier")
    s.page.uncheck("#g-spec-on")
    assert s.wait_for(lambda: s.count("/gallery/frontier") > before)
    # The plain view again: no draft in its identity (what the page holds and
    # wants first ride along and are not part of it).
    assert set(s.requests["/gallery/frontier"][-1]["query"]) - {
        "priority", "have_tag", "have"} == {
        "gpu_sku", "comm_latency_us", "launch_floor_us"}
    # Once the plain frontier is in, every model draws again.
    assert s.wait_for(lambda: s.page.query_selector("#g-front-no-draft") is None)
    assert s.page.query_selector("#g-front-draft-errors") is None
    s.assert_clean()


def test_inspector_reads_the_hash_of_each_visit(browser_base):
    """Arriving with #model=… auto-analyzes it; coming back through the nav
    (client-side route, same JS module) with no hash must not re-analyze the
    first visit's model."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inspector.html#model=deepseek-ai%2FDeepSeek-V3.2")
    assert s.wait_for(lambda: s.count("/analyze") >= 1)
    assert s.requests["/analyze"][0]["body"]["model_id"] == "deepseek-ai/DeepSeek-V3.2"
    s.page.click("a[data-tab='fusion']")
    s.page.wait_for_selector("main", timeout=10000)
    s.page.click("a[data-tab='inspector']")
    s.page.wait_for_selector("#model-id", timeout=10000)
    s.page.wait_for_timeout(1000)
    assert s.count("/analyze") == 1, [r["body"] for r in s.requests["/analyze"]]
    s.assert_clean()


_PRIVATE_ENGINE_FAILURE = {
    "success": False, "served_from": "user_generation", "record_status": "generated",
    "error_kind": "blocked_by_engine", "message": "cannot close the gather",
    "engine_gap": {"capability": "gather extent", "node": "layers.0.lm_head"},
}


@pytest.mark.parametrize("analyze, generates", [
    # the caller's own spec hit an engine gap: shown, never regenerated
    (_PRIVATE_ENGINE_FAILURE, False),
    # the caller's own spec failed on a spec refusal: shown, never regenerated
    ({**_PRIVATE_ENGINE_FAILURE, "error_kind": "spec_invalid",
      "errors": ["Fix `dims`: declare d_model."]}, False),
    # no spec at all: the one case the page generates one by itself
    ({"success": False, "no_spec": True, "message": "No spec matches 'acme/Foo'."}, True),
])
def test_inspector_generates_only_when_the_model_has_no_spec(browser_base, analyze, generates):
    """An engine regression must not turn every open of a private record
    into a paid spec-generation run."""
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/analyze": analyze,
        "/spec-status": {"state": "idle"},
        "/generate-spec": {"status": "started", "success": True},
        "/stop-spec": {"success": True},
    })
    s.goto("/inspector.html#model=acme%2FFoo")
    assert s.wait_for(lambda: s.count("/analyze") >= 1)
    s.page.wait_for_timeout(1500)
    assert (s.count("/generate-spec") > 0) is generates, s.requests
    if not generates:
        banner = ("[data-engine-blocked-banner]" if analyze["error_kind"] == "blocked_by_engine"
                  else "[data-record-failed]")
        assert s.page.locator(banner).count() == 1
    s.assert_clean()


def test_kernel_bench_label_on_nav_page_and_fusion_link(browser_base):
    """The product name is Kernel Bench: top-nav tab, page heading, and the
    Fusion deep-link into envelope timings. The /kernel-corpus.html URL stays."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/kernel-corpus.html")
    s.page.wait_for_selector("#f-source", timeout=15000)
    tab = s.page.locator("nav a[data-tab='corpus']").first
    assert tab.inner_text().strip() == "Kernel Bench"
    assert s.page.locator("h1").inner_text().strip() == "Kernel Bench"
    s.goto("/fusion.html")
    s.page.wait_for_selector("#fusion-plan-detail", timeout=15000)
    link = s.page.get_by_role("link", name="Kernel Bench ↗")
    assert s.wait_for(lambda: link.count() > 0)
    assert s.page.get_by_text("在 Corpus 查看测速").count() == 0
    link.first.click()
    s.page.wait_for_selector("#f-source", timeout=15000)
    assert "/kernel-corpus.html" in s.page.url
    h = parse_qs(urlparse(s.page.url).fragment)
    assert h.get("view") == ["envelope"]
    assert s.page.locator("h1").inner_text().strip() == "Kernel Bench"
    s.assert_clean()


def test_fusion_provider_link_carries_kernel_context(browser_base):
    """Clicking a provider's kernel_impl lands on Kernel Bench with the row's
    own filters filled: signature + framework + kernel ride in the hash."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/fusion.html")
    s.page.wait_for_selector("#fusion-plan-detail", timeout=15000)
    link = s.page.get_by_role("link", name="torch_allreduce_fused_rmsnorm")
    assert s.wait_for(lambda: link.count() > 0)
    link.first.click()
    s.page.wait_for_selector("#f-source", timeout=15000)
    h = parse_qs(urlparse(s.page.url).fragment)
    assert h.get("view") == ["envelope"]
    assert h.get("signature") == ["allreduce_fused_rmsnorm"]
    assert h.get("framework") == ["torch"]
    assert h.get("kernel") == ["torch_allreduce_fused_rmsnorm"]
    assert s.page.locator("#f-framework").input_value() == "torch"
    assert s.page.locator("#f-kernel-clear").inner_text().startswith("torch_allreduce_fused_rmsnorm")
    s.assert_clean()


def test_kernel_bench_dropdowns_follow_active_filters(browser_base):
    """Each filter dropdown lists the values that actually occur under the
    other active filters, not the whole corpus."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/kernel-corpus.html#view=envelope&signature=rmsnorm")
    s.page.wait_for_selector("#f-source", timeout=15000)
    # the options are the server's answer (its facets), so they arrive with it
    assert s.wait_for(lambda: s.page.locator("#f-model option").count() > 1)
    models = s.page.locator("#f-model option").all_inner_texts()
    assert models == ["All models", "deepseek-ai/DeepSeek-V3"]
    gpus = s.page.locator("#f-gpu option").all_inner_texts()
    assert gpus == ["All GPUs", "B200"]
    s.goto("/kernel-corpus.html#view=envelope")
    s.page.wait_for_selector("#f-signature", timeout=15000)
    assert s.wait_for(lambda: s.page.locator("#f-signature option").count() > 2)
    ops = s.page.locator("#f-signature option").all_inner_texts()
    assert ops[0] == "All ops"
    assert set(ops[1:]) == {"rmsnorm", "gqa_attention"}
    s.assert_clean()


def test_kernel_bench_providers_catalog_and_coverage(browser_base):
    """Providers tab is the live catalog joined to measured rows.

    flashinfer_rmsnorm is in the canned corpus (4 cells); ragged prefill is
    declared and unmeasured; torch_cublas_linear is still on a hand-written
    gate; cutlass_mla is source-build. Clicking a kernel lands on envelope
    with signature/framework/kernel filters. The cover hash is restored."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/kernel-corpus.html#view=providers")
    s.page.wait_for_selector("#providers-section", timeout=15000)
    assert s.wait_for(lambda: s.count("/api/bench/providers") >= 1)
    assert s.page.locator("#tab-providers").inner_text().strip() == "Providers"
    rms = s.page.locator("#kernel-flashinfer-flashinfer_rmsnorm")
    assert s.wait_for(lambda: rms.count() > 0)
    assert s.wait_for(lambda: "4 rows" in rms.inner_text())   # coverage arrives with its own answer
    ragged = s.page.locator("#kernel-flashinfer-flashinfer_ragged_prefill")
    assert "unmeasured" in ragged.inner_text()
    torch_row = s.page.locator("#kernel-torch-torch_cublas_linear")
    assert "undeclared" in torch_row.inner_text()
    cutlass = s.page.locator("#kernel-cutlass_mla-cutlass_mla_decode")
    assert "100" in cutlass.inner_text()
    assert s.page.get_by_text("source-build").count() > 0
    v2 = s.page.locator("#kernel-fa2-flash_attention_v2")
    kv = s.page.locator("#kernel-fa2-flash_attention_v2_kvcache")
    assert "unmeasured" in v2.inner_text()
    assert "measured" in kv.inner_text()
    assert "1 rows" in kv.inner_text()
    s.page.select_option("#f-cover", "unmeasured")
    assert s.wait_for(lambda: s.page.locator("#kernel-flashinfer-flashinfer_rmsnorm").count() == 0)
    assert s.page.locator("#kernel-flashinfer-flashinfer_ragged_prefill").count() == 1
    h = parse_qs(urlparse(s.page.url).fragment)
    assert h.get("view") == ["providers"] and h.get("cover") == ["unmeasured"], h
    s.page.select_option("#f-cover", label="All coverage")
    assert s.wait_for(lambda: s.page.locator("#kernel-flashinfer-flashinfer_rmsnorm").count() == 1)
    s.page.locator("#kernel-flashinfer-flashinfer_rmsnorm button").click()
    s.page.wait_for_selector("#table-section", timeout=15000)
    h = parse_qs(urlparse(s.page.url).fragment)
    assert h.get("view") == ["envelope"]
    assert h.get("signature") == ["rmsnorm"]
    assert h.get("framework") == ["flashinfer"]
    assert h.get("kernel") == ["flashinfer_rmsnorm"]
    assert s.page.locator("#f-kernel-clear").inner_text().strip().startswith("flashinfer_rmsnorm")
    s.assert_clean()


def test_kernel_bench_providers_hash_restores_cover(browser_base):
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/kernel-corpus.html#view=providers&cover=undeclared")
    s.page.wait_for_selector("#f-cover", timeout=15000)
    assert s.wait_for(lambda: s.page.input_value("#f-cover") == "undeclared")
    # The select hydrates from the hash before the catalog/coverage joins
    # finish; both assertions are about that join, not about the local hash.
    assert s.wait_for(
        lambda: s.page.locator("#kernel-torch-torch_cublas_linear").count() == 1)
    assert s.wait_for(
        lambda: s.page.locator("#kernel-flashinfer-flashinfer_rmsnorm").count() == 0)
    s.assert_clean()


def test_corpus_hash_keeps_workload_filters(browser_base):
    """A workload restored from the hash stays chosen; a new batch starts at
    its smallest context; a model + phase without a batch starts at the
    smallest batch."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/kernel-corpus.html#view=workload&model=deepseek-ai%2FDeepSeek-V3&phase=prefill&batch=16&context=4096")
    s.page.wait_for_selector("#f-batch", timeout=15000)
    assert s.wait_for(lambda: s.page.input_value("#f-batch") == "16")
    assert s.page.input_value("#f-context") == "4096"
    assert s.wait_for(lambda: "op shapes measured at this workload" in s.page.inner_text("#count"))
    asked = [r["query"] for r in s.requests["/api/bench/browse"] if r["query"].get("view") == "workload"]
    assert asked[-1].get("batch") == "16" and asked[-1].get("context") == "4096"
    s.page.select_option("#f-batch", "1")
    assert s.wait_for(lambda: s.page.input_value("#f-context") == "1024")
    h = parse_qs(urlparse(s.page.url).fragment)
    assert h.get("batch") == ["1"] and h.get("context") == ["1024"], h
    s.goto("/kernel-corpus.html#view=workload&model=deepseek-ai%2FDeepSeek-V3&phase=decode")
    s.page.reload()   # only the hash changed: the page reads it on load
    assert s.wait_for(lambda: s.page.input_value("#f-batch") == "1")
    assert s.wait_for(lambda: s.page.input_value("#f-context") == "4096")
    s.assert_clean()


_OPENING = {"parallel": {"ep_moe": 8, "dp_attention": 8}, "moe": True,
            "start_degree": 8, "fits": True}


def test_inference_opens_at_the_backend_s_deployment(browser_base):
    """A link that states no deployment asks the backend which one the model
    opens at on the card before it estimates, and the first estimate is
    priced there (no estimate at one GPU first); a link that states one GPU
    (every axis 1) is an answer, estimated as given, and asks nothing."""
    browser, base = browser_base
    s = Session(browser, base, overrides={"/inference/default-deployment": _OPENING})
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=H100-SXM")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    assert s.count("/inference/default-deployment") >= 1
    assert s.requests["/inference/default-deployment"][-1]["body"]["model_id"] == "deepseek-ai/DeepSeek-V3"
    s.page.wait_for_timeout(1500)
    bodies = [r["body"]["parallel"] for r in s.requests["/inference/estimate"]]
    assert all((b["ep_moe"], b["dp_attention"]) == (8, 8) for b in bodies), bodies
    s.assert_clean()

    s = Session(browser, base, overrides={"/inference/default-deployment": _OPENING})
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=H100-SXM&ep=1&dp_attn=1")
    assert s.wait_for(lambda: s.count("/inference/estimate") >= 1)
    s.page.wait_for_timeout(1500)
    bodies = [r["body"]["parallel"] for r in s.requests["/inference/estimate"]]
    assert all((b["ep_moe"], b["dp_attention"]) == (1, 1) for b in bodies), bodies
    assert s.count("/inference/default-deployment") == 0
    s.assert_clean()


def test_inference_unfit_result_is_not_a_success(browser_base):
    """An estimate whose weights do not fit says so in the status line
    instead of a green "Done."; zero output tokens show no decode TPOT."""
    browser, base = browser_base
    est = _fixture("estimate_deepseek_v3_b200.json")
    est = {**est, "avg_output_tokens": 0, "max_batch_per_gpu": 0,
           "infeasibility_reason": "weights need 650 GiB per GPU, 160 GiB usable",
           "memory": {**est["memory"], "fits": False, "weight_deficit_bytes": 1e9}}
    s = Session(browser, base, overrides={"/inference/estimate": est})
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8&out_tok=0")
    s.page.wait_for_function(
        "() => /Does not fit/.test(document.querySelector('#infer-status')?.textContent || '')",
        timeout=15000)
    assert "650 GiB" in s.page.inner_text("#infer-status")
    assert "green" not in (s.page.get_attribute("#infer-status", "class") or "")
    assert "no output tokens" in s.page.inner_text("#infer-tpot-none")
    s.assert_clean()


def test_inference_results_are_marked_stale_after_an_edit(browser_base):
    """Editing an input the estimate reads marks the numbers on screen as
    the previous inputs' until Estimate runs again."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8")
    s.page.wait_for_function(
        "() => /Done/.test(document.querySelector('#infer-status')?.textContent || '')",
        timeout=15000)
    assert s.page.query_selector("#infer-results-stale") is None
    s.page.fill("#infer-avg-in", "4096")
    s.page.wait_for_selector("#infer-results-stale", timeout=5000)
    s.page.click("[data-action=run-estimate]")
    s.page.wait_for_selector("#infer-results-stale", state="detached", timeout=15000)
    s.assert_clean()


def test_fusion_follows_the_hash_and_history(browser_base):
    """`#op=` names the plan shown: an edited hash selects its plan, a click
    is a history entry, and Back returns to the plan before it."""
    browser, base = browser_base
    s = Session(browser, base)
    plans = [p["fused_op"] for p in s.stubs["/fusion/registry"]["plans"]]
    s.goto("/fusion.html")
    s.page.wait_for_selector("#fusion-plan-detail", timeout=15000)
    detail = s.page.locator("#fusion-plan-detail")
    assert s.wait_for(lambda: f"op={plans[0]}" in s.page.url)
    s.page.get_by_role("button", name=plans[1], exact=False).first.click()
    assert s.wait_for(lambda: f"op={plans[1]}" in s.page.url)
    assert s.wait_for(lambda: plans[1] in detail.inner_text())
    s.page.go_back()
    assert s.wait_for(lambda: f"op={plans[0]}" in s.page.url)
    assert s.wait_for(lambda: plans[0] in detail.inner_text()
                      and plans[1] not in detail.inner_text())
    s.page.evaluate(f"location.hash = 'op={plans[1]}'")
    assert s.wait_for(lambda: plans[1] in detail.inner_text())
    s.assert_clean()


def test_verify_without_a_token_says_so(browser_base):
    """The sign-in completion page opened with no token in its address
    explains that at once and links back to sign in — no endless spinner."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/auth/verify.html")
    s.page.get_by_text("carries no sign-in token").wait_for(timeout=4000)
    assert s.page.get_by_role("link", name="Back to sign in").count() == 1
    s.errors = [e for e in s.errors if "401" not in e]
    s.assert_clean()


def test_inspector_without_a_spec_shows_only_its_banner(browser_base):
    """A model with no spec has no dashboard: the no-spec banner is the
    answer, and no empty metric cards (NaN %, 0 B) render under it."""
    browser, base = browser_base
    s = Session(browser, base, overrides={
        "/analyze": {"success": False, "no_spec": True,
                     "message": "No spec matches 'acme/Foo'."},
        "/spec-status": {"state": "idle"},
        "/generate-spec": {"status": "started", "success": True},
        "/stop-spec": {"success": True},
    })
    s.goto("/inspector.html#model=acme%2FFoo")
    assert s.wait_for(lambda: s.count("/analyze") >= 1)
    s.page.wait_for_selector("#no-spec-banner", timeout=15000)
    assert s.page.locator("#results").is_hidden()
    assert "NaN" not in s.page.inner_text("main")
    s.assert_clean()


def test_api_status_follows_health(browser_base):
    """The nav says "API Ready" only while /health answers, and says so
    when it does not."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/fusion.html")
    s.page.wait_for_selector("#api-status[data-api-up='true']", timeout=15000)
    assert "API Ready" in s.page.inner_text("#api-status")
    s.assert_clean()

    down = Session(browser, base, {"/health": Refusal(503, {"detail": "down"})})
    down.goto("/fusion.html")
    down.page.wait_for_selector("#api-status[data-api-up='false']", timeout=15000)
    assert "API unavailable" in down.page.inner_text("#api-status")


def test_unknown_page_says_not_found(browser_base):
    """A URL no page answers shows "Page not found" in the site chrome with
    a way back, instead of silently landing on the Gallery."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/no-such-page.html")
    s.page.wait_for_selector("[data-not-found]", timeout=15000)
    assert "Page not found" in s.page.inner_text("[data-not-found]")
    assert s.page.url.endswith("/no-such-page.html")
    assert s.page.title().startswith("Page not found")
    s.assert_clean()


def test_fusion_notes_read_as_text_and_counts_agree(browser_base):
    """A plan's notes render as paragraphs (inline `code` drawn as code, not
    backticks), the provider count agrees in number, and a long plan name
    in the list carries its full name as a tooltip."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/fusion.html#op=allreduce_fused_rmsnorm")
    s.page.wait_for_selector("#fusion-plan-detail summary", timeout=15000)
    s.page.click("#fusion-plan-detail summary")
    notes = s.page.locator(".plan-notes")
    assert notes.locator("p").count() > 1
    assert notes.locator("p code").count() > 0
    assert "`" not in " ".join(notes.locator("p").all_inner_texts())
    detail = s.page.inner_text("#fusion-plan-detail")
    assert "1 library," in detail and "1 libraries" not in detail
    assert s.page.locator("#fusion-plan-list span[title='allreduce_fused_rmsnorm']").count() == 1
    assert s.page.title().startswith("Fusion")
    s.assert_clean()


def test_kernel_bench_draws_one_utilization_chart_per_op(browser_base):
    """Below the envelope table: one chart per op in view, asked as the
    curves view under the page's filters."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/kernel-corpus.html#view=envelope")
    s.page.wait_for_selector("#curve-charts", timeout=15000)
    assert s.wait_for(lambda: s.page.locator("#curve-charts canvas").count() >= 1)
    asked = [r["query"] for r in s.requests["/api/bench/browse"] if r["query"].get("view") == "curves"]
    assert asked
    ops = s.page.locator("#curve-charts > [data-op]")
    assert {ops.nth(i).get_attribute("data-op") for i in range(ops.count())} == {"rmsnorm", "gqa_attention"}
    # Each op draws a row per GPU · dtype; the first is open, with MFU and
    # MBU on charts of their own; the others open on a click.
    def metrics_of(facet):
        m = facet.locator("[data-metric]")
        return [m.nth(j).get_attribute("data-metric") for j in range(m.count())]
    for i in range(ops.count()):
        facets = ops.nth(i).locator("[data-facet]")
        assert facets.count() >= 1
        assert facets.first.get_attribute("data-open") == "1" and metrics_of(facets.first) == ["mfu", "mbu"]
        for k in range(1, facets.count()):
            closed = facets.nth(k)
            assert closed.get_attribute("data-open") == "0" and metrics_of(closed) == []
            closed.locator(".curve-facet-toggle").click()
            assert s.wait_for(lambda closed=closed: metrics_of(closed) == ["mfu", "mbu"])
    s.page.select_option("#f-signature", "rmsnorm")
    assert s.wait_for(lambda: s.page.locator("#curve-charts > [data-op]").count() == 1)
    assert s.page.locator("#curve-charts > [data-op]").get_attribute("data-op") == "rmsnorm"
    s.assert_clean()


@pytest.mark.parametrize("selection, ticked", [("latest", True), ("theoretical_only", True),
                                               ("latest", False)])
def test_scaling_tab_cache_holds_sol_answers_only(browser_base, selection, ticked):
    """Switching the Scaling context back to one already drawn: its SOL
    answer is served from the tab whatever the mode (it reads no
    measurements); its measured answer, while Measured is ticked and the
    mode reads measurements, is asked of the server again — what Kernel
    Bench holds may have changed since (docs/auto_sweep.md §7.5)."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200&ep=8&dp_attn=8"
           f"&selection={selection}")
    path = "/inference/estimate-parallel-sweep"
    assert s.wait_for(lambda: s.count(path) >= 1)
    if ticked:
        s.page.check("#infer-par-measured")
    s.page.wait_for_timeout(800)
    first = s.requests[path][-1]["body"]
    assert first["selection_mode"] == selection
    bodies = lambda: [r["body"] for r in s.requests[path]]
    measured = selection != "theoretical_only" and ticked
    a = s.page.eval_on_selector("#infer-scaling-ctx", "e => e.value")
    b = next(str(c) for c in SCALING_CONTEXTS if str(c) != a)
    n = s.count(path)
    s.page.select_option("#infer-scaling-ctx", b)
    assert s.wait_for(lambda: s.count(path) == n + (2 if measured else 1))
    n = s.count(path)
    s.page.select_option("#infer-scaling-ctx", a)
    s.page.wait_for_timeout(1000)
    back = bodies()[n:]
    assert not any(x.get("sol_only") for x in back), "the SOL answer was asked again"
    assert len(back) == (1 if measured else 0), back
    if measured:
        assert back[0]["context_len"] == int(a)
    s.assert_clean()


def test_kernel_bench_pins_a_curve_and_steps_its_points(browser_base):
    """A click on a curve pins it and lists its points; ←/→ move the
    selected point; Esc unpins."""
    browser, base = browser_base
    s = Session(browser, base)
    s.page.set_viewport_size({"width": 1400, "height": 1000})
    s.goto("/kernel-corpus.html#view=envelope&signature=rmsnorm")
    s.page.wait_for_selector("#curve-charts canvas", timeout=15000)
    s.page.wait_for_timeout(1500)
    canvas = s.page.locator("#curve-charts [data-metric=mbu] canvas").first
    box = canvas.bounding_box()
    pinned = False
    for dx in range(50, int(box["width"]) - 16, 4):
        for dy in range(30, int(box["height"]) - 64, 4):
            s.page.mouse.click(box["x"] + dx, box["y"] + dy)
            if s.page.locator(".curve-pin").count():
                pinned = True
                break
        if pinned:
            break
    assert pinned, "no click on the MBU chart pinned a curve"
    rows = s.page.locator(".curve-pin tbody tr")
    assert rows.count() >= 2
    selected = lambda: [i for i in range(rows.count()) if "bg-blue-100" in (rows.nth(i).get_attribute("class") or "")]
    first = selected()
    assert len(first) == 1
    s.page.keyboard.press("ArrowRight" if first[0] < rows.count() - 1 else "ArrowLeft")
    assert s.wait_for(lambda: selected() != first and len(selected()) == 1)
    rows.nth(0).click()
    assert s.wait_for(lambda: selected() == [0])
    s.page.locator(".curve-pin").focus()
    s.page.keyboard.press("Escape")
    assert s.wait_for(lambda: s.page.locator(".curve-pin").count() == 0)
    s.assert_clean()


def test_kernel_bench_curve_charts_grow_with_the_legend_they_fit(browser_base):
    """The room under the axis follows the width the chart is drawn at: one
    op whose points eight kernels won wraps its legend into more rows when
    narrow, and the chart is taller for them, so the axis name is never
    under the legend."""
    browser, base = browser_base
    kernels = ["flashinfer_paged_decode", "flash_attention_v3_kvcache", "flash_attention_v2_kvcache",
               "flashinfer_ragged_prefill", "flash_attention_v3", "flash_attention_v2",
               "torch_sdpa", "flash_attention_v4"]
    curves = {"ops": [{"signature": "gqa_attention", "points": len(kernels), "curves_total": 1,
                       "unplaced": 0, "curves": [{
                           "gpu": "H100-SXM", "dtype_env_json": {"activation": "bf16"},
                           "axis": "B×s_kv", "shape": {"d_head": 128},
                           "saturation": {"state": "memory", "tokens": 2 ** (len(kernels) + 1)},
                           "points": [{"x": 2 ** (i + 1), "shape": '{"B":1}', "phase": "decode",
                                       "framework": "x", "kernel": k, "model": "m/m",
                                       "latency_us": 10.0, "tflops": 1.0, "gbps": 100.0,
                                       "mfu": 0.1, "mbu": 0.5} for i, k in enumerate(kernels)],
                           "lines": [list(range(len(kernels)))]}]}]}
    s = Session(browser, base, overrides={
        "/api/bench/browse": lambda q: curves if q.get("view") == "curves" else _fake_browse(q)})
    s.goto("/kernel-corpus.html#view=envelope")
    s.page.wait_for_selector("#curve-charts canvas", timeout=15000)

    def chart_height(w):
        s.page.set_viewport_size({"width": w, "height": 900})
        s.page.wait_for_timeout(1200)
        return s.page.locator("#curve-charts [data-metric=mbu] canvas").first.bounding_box()["height"]
    wide = chart_height(1400)
    narrow = chart_height(390)
    assert narrow > wide, f"narrow {narrow} should reserve more legend rows than wide {wide}"
    s.assert_clean()


def test_gallery_decode_frontier_follows_the_gpu_and_names_its_misses(browser_base):
    """The decode chart is the Pareto frontier: fetched per GPU, titled by
    its context tab, and a checked model with no frontier at that context is
    named under it. The page asks for no per-preset overview."""
    browser, base = browser_base
    s = Session(browser, base)
    s.goto("/gallery.html")
    assert s.wait_for(lambda: s.count("/gallery/frontier") >= 1)
    s.page.wait_for_selector("#g-dec-tab-1024", timeout=15000)
    assert "Decode Pareto frontier — 1K context" in s.page.inner_text("main")
    s.page.wait_for_selector("#g-front-missing", timeout=15000)
    assert "DeepSeek-V3.2" in s.page.inner_text("#g-front-missing")
    # The view, plus what a cold view's retries carry: the models wanted
    # first and what the page already holds (not part of the view).
    query = set(s.requests["/gallery/frontier"][0]["query"])
    assert {"gpu_sku", "comm_latency_us", "launch_floor_us"} <= query
    assert query <= {"gpu_sku", "comm_latency_us", "launch_floor_us",
                     "priority", "have_tag", "have"}
    assert s.count("/gallery/overview") == 0
    s.assert_clean()


def test_gallery_coverage_panel_draws_the_servers_coverage(browser_base):
    """Under the frontier, each drawn model's Kernel Bench coverage at the
    context tab, as the server states it: the median share, the best
    point's ops, the fully measured points, and — expanded — the unmeasured
    keys by state with a link to the model's decode rows on that GPU. An
    unreadable lake says so instead of showing zeros."""
    browser, base = browser_base
    track = {"state": "reads", "coverage": {"1024": {
        "points": 2, "complete": 1, "best": {"measured": 75, "total": 75},
        "median": 0.66,
        "states": {"missing": {"keys": 12, "top": [
            {"key": 'grouped_gemm {"n_experts":256}', "points": 1}]},
            "unplaced": {"keys": 1, "top": [{"key": "concat", "points": 2}]}}}}}
    read = {**_FRONTIER, "models": [
        {**_FRONTIER["models"][0], "measured_track": track}, _FRONTIER["models"][1]]}
    s = Session(browser, base, overrides={"/gallery/frontier": read})
    s.goto("/gallery.html")
    s.page.wait_for_selector("#g-coverage .g-cov-row", timeout=15000)
    row = s.page.locator('#g-coverage .g-cov-row[data-model="deepseek-ai/DeepSeek-V3"]')
    text = row.inner_text()
    assert "66%" in text and "best 75 / 75 ops" in text
    assert "1 / 2 points fully measured" in text
    row.locator("button").click()
    detail = row.locator(".g-cov-detail")
    assert "missing — 12 keys" in detail.inner_text()
    assert 'grouped_gemm {"n_experts":256}' in detail.inner_text()
    assert "unplaced — 1 key" in detail.inner_text()
    href = detail.locator("a").get_attribute("href")
    assert href.startswith("/kernel-corpus.html#")
    assert "model=deepseek-ai%2FDeepSeek-V3" in href and "phase=decode" in href
    s.assert_clean()

    lake_down = {**_FRONTIER, "models": [
        {**m, "measured_track": {"state": "unavailable"}} for m in _FRONTIER["models"]]}
    s2 = Session(browser, base, overrides={"/gallery/frontier": lake_down})
    s2.goto("/gallery.html")
    s2.page.wait_for_selector("#g-coverage-none", timeout=15000)
    assert "could not be read" in s2.page.inner_text("#g-coverage-none")
    s2.assert_clean()


def test_gallery_frontier_shows_a_progress_bar_while_computing(browser_base):
    """A cold decode frontier's 503s say how far the pricing got; while the
    page retries it draws a determinate progress bar (the done count as
    `aria-valuenow`) of models, says how many compute jobs have finished
    (they move between two models), and the chart renders when the payload
    lands."""
    browser, base = browser_base
    s = Session(browser, base)
    calls = []

    def frontier(route):
        calls.append(1)
        if len(calls) <= 2:
            route.fulfill(status=503, content_type="application/json",
                          body=json.dumps({"reason": "still_computing",
                                           "retry_after_s": 1,
                                           "progress": {"done": len(calls), "total": 4},
                                           "jobs_done": 1234}))
        else:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(_FRONTIER))

    # Registered after the session's catch-all, so it answers first.
    s.page.route("**/gallery/frontier*", frontier)
    s.goto("/gallery.html")
    s.page.wait_for_selector('[role="progressbar"]', timeout=15000)

    def valuemaxes():
        return s.page.eval_on_selector_all(
            '[role="progressbar"]', "els => els.map(e => e.getAttribute('aria-valuemax') || '')")

    assert s.wait_for(lambda: "4" in valuemaxes()), "no determinate progress bar while computing"
    # A small corner note, not a banner; the long explanation is its tooltip.
    text = s.page.inner_text("#g-computing")
    assert "/ 4 models" in text, text
    assert "1,234 compute jobs finished" in (s.page.get_attribute("#g-computing", "title") or "")
    box = s.page.eval_on_selector("#g-computing", "e => getComputedStyle(e).position")
    assert box == "fixed", box
    assert s.wait_for(lambda: len(calls) >= 3)
    s.page.wait_for_selector("#g-dec-tab-1024", timeout=15000)
    # The stubbed 503s the page retried through log as console resource errors.
    s.errors = [e for e in s.errors if "503" not in e]
    s.assert_clean()


def test_gallery_draws_each_model_as_it_lands_and_asks_for_the_shown_ones_first(browser_base):
    """While a cold frontier computes, each 503 carries the models finished
    since the page's cursor: the page draws them at once, sends the cursor
    back so they are not sent again, and names the models on its chart
    first (`priority`). A model not arrived yet is pending, not missing."""
    browser, base = browser_base
    s = Session(browser, base)
    calls = []
    head = {k: v for k, v in _FRONTIER.items() if k != "models"}
    first = _FRONTIER["models"][0]

    def frontier(route):
        calls.append(route.request.url)
        if len(calls) == 1:
            route.fulfill(status=503, content_type="application/json",
                          body=json.dumps({"reason": "still_computing", "retry_after_s": 1,
                                           "progress": {"done": 1, "total": 2},
                                           "partial": {"tag": 7, "count": 1, "head": head,
                                                       "items": [first]}}))
        elif len(calls) == 2:
            route.fulfill(status=503, content_type="application/json",
                          body=json.dumps({"reason": "still_computing", "retry_after_s": 1,
                                           "progress": {"done": 1, "total": 2},
                                           "partial": {"tag": 7, "count": 1, "head": head,
                                                       "items": []}}))
        else:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(_FRONTIER))

    s.page.route("**/gallery/frontier*", frontier)
    s.goto("/gallery.html")
    # The first model draws while the second computes ...
    s.page.wait_for_selector("#g-dec-tab-1024", timeout=15000)
    assert s.wait_for(lambda: len(calls) >= 2)
    assert s.page.query_selector("#g-front-missing") is None, "a pending model read as missing"
    # ... and the retry says what the page holds and what it wants first.
    from urllib.parse import parse_qs, urlparse
    retry = parse_qs(urlparse(calls[1]).query)
    assert retry.get("have_tag") == ["7"] and retry.get("have") == ["1"], retry
    assert "deepseek-ai/DeepSeek-V3" in (retry.get("priority") or [""])[0].split(","), retry
    # The whole answer lands: now a model with no frontier is named.
    s.page.wait_for_selector("#g-front-missing", timeout=15000)
    s.errors = [e for e in s.errors if "503" not in e]
    s.assert_clean()


def test_a_frontier_with_no_model_landed_yet_says_it_is_computing(browser_base):
    """A cold frontier's first answers carry the view's head and no model:
    the chart is waiting, not empty — "no model serves this context" is a
    claim about the whole answer for this view, said only once it landed."""
    browser, base = browser_base
    s = Session(browser, base)
    calls = []
    head = {k: v for k, v in _FRONTIER.items() if k != "models"}

    def frontier(route):
        calls.append(1)
        if len(calls) <= 3:
            route.fulfill(status=503, content_type="application/json",
                          body=json.dumps({"reason": "still_computing", "retry_after_s": 1,
                                           "progress": {"done": 0, "total": 2},
                                           "partial": {"tag": 3, "count": 0, "head": head,
                                                       "items": []}}))
        else:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(_FRONTIER))

    s.page.route("**/gallery/frontier*", frontier)
    s.goto("/gallery.html")
    assert s.wait_for(lambda: len(calls) >= 2)
    s.page.wait_for_selector("#g-front-waiting", timeout=15000)
    assert "no model serves this context" not in s.page.inner_text("body")
    s.page.wait_for_selector("#g-dec-tab-1024", timeout=15000)
    s.errors = [e for e in s.errors if "503" not in e]
    s.assert_clean()


def test_a_request_rides_out_a_kernel_bench_restart(browser_base):
    """While the Kernel Bench service restarts, an estimate answers 503
    `measurements_unavailable` and asks to be sent again; the page sends it
    again and draws the answer, instead of failing on the first 503 and
    leaving everything below it empty."""
    browser, base = browser_base
    s = Session(browser, base)
    calls = []

    def frontier(route):
        calls.append(1)
        if len(calls) <= 2:
            route.fulfill(status=503, content_type="application/json",
                          body=json.dumps({"reason": "measurements_unavailable",
                                           "retry_after_s": 1,
                                           "detail": "measurements unavailable: the "
                                                     "Kernel Bench store is starting"}))
        else:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(_FRONTIER))

    s.page.route("**/gallery/frontier*", frontier)
    s.goto("/gallery.html")
    s.page.wait_for_selector("#g-dec-tab-1024", timeout=15000)
    assert len(calls) >= 3
    s.errors = [e for e in s.errors if "503" not in e]
    s.assert_clean()


def test_scaling_sweep_shows_a_progress_bar_while_computing(browser_base):
    """Same on the Inference page's Context × Batch sweep: a cold sweep's
    503s carry the cells' count and the waiting slot draws the bar; the
    curves render when the payload lands."""
    browser, base = browser_base
    s = Session(browser, base)
    calls = []

    def sweep(route):
        calls.append(1)
        if len(calls) <= 2:
            route.fulfill(status=503, content_type="application/json",
                          body=json.dumps({"reason": "still_computing",
                                           "retry_after_s": 1,
                                           "progress": {"done": len(calls), "total": 9}}))
        else:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(_SWEEP))

    s.page.route("**/inference/estimate-sweep*", sweep)
    s.goto("/inference.html#model=deepseek-ai%2FDeepSeek-V3&gpu=B200")

    def valuemaxes():
        return s.page.eval_on_selector_all(
            '#infer-scaling [role="progressbar"]',
            "els => els.map(e => e.getAttribute('aria-valuemax') || '')")

    assert s.wait_for(lambda: "9" in valuemaxes(), timeout_ms=20000), \
        "no determinate progress bar while the sweep computes"
    assert s.wait_for(lambda: len(calls) >= 3)
    s.page.wait_for_selector("#infer-scaling canvas", timeout=20000)
    # The stubbed 503s the page retried through log as console resource errors.
    s.errors = [e for e in s.errors if "503" not in e]
    s.assert_clean()
