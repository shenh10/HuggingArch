"""Tests for the Fusion Patterns page's backend endpoints.

Covers:
  * GET /fusion/registry  — catalog shape (plans + bindings + serialized
                            naive_pattern with nodes form).
  * GET /fusion/fired     — framework rewrite roundtrip; structural diagnostics
                            map cleanly into the per-fused_op response.

The fired endpoint resolves a spec via SpecLoader so the test points at
a small model with both attention + FFN in its spec library
(`meta-llama/Llama-2-7b-hf`) — that exercises all four shipped fusion plans
(flash_attention, silu_and_mul, merged_column_parallel_linear,
fused_add_rmsnorm) without depending on DeepSeek-V3's much heavier
spec / config.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.api import app


class TestFusionRegistryEndpoint(unittest.TestCase):
    """Catalog endpoint: every shipped plan + binding shows up with the
    expected serialized fields."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_registry_returns_plans_and_bindings(self):
        r = self.client.get("/fusion/registry")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("plans", data)
        self.assertIn("bindings", data)
        plan_names = [p["fused_op"] for p in data["plans"]]
        # Every shipped plan must appear, including the sglang-only
        # fused_qk_norm_rope.
        for expected in [
            "flash_attention",
            "silu_and_mul",
            "fused_add_rmsnorm",
            "merged_column_parallel_linear",
            "fused_qk_norm_rope",
            "grouped_gemm_moe",
            "fused_moe",
        ]:
            self.assertIn(expected, plan_names, f"missing plan: {expected}")
        # Every shipped binding appears in the catalog, one per framework,
        # each labelled with the release it was audited at.
        by_fw = {b["framework"]: b for b in data["bindings"]}
        self.assertEqual(sorted(by_fw),
                         ["deepinfer", "sglang", "vllm", "vllm_ascend"])
        for fw, b in by_fw.items():
            self.assertEqual(b["label"], f"{fw} (audited at {b['audited_at']['tag']})")
            self.assertTrue(b["binding_file"].endswith(f"{fw}.yaml"))
            self.assertNotIn("version", b)

    def test_plan_backends_enumerate_kernel_bench_providers(self):
        # Each fused op surfaces the measured backend kernels that realize it
        # (from the kernel_bench provider registry) — the "supported backends".
        r = self.client.get("/fusion/registry")
        data = r.json()
        fa = next(p for p in data["plans"] if p["fused_op"] == "flash_attention")
        self.assertIsNotNone(fa["backends"])
        self.assertEqual(fa["backends"]["signature"], "gqa_attention")
        fa_libs = {p["library"] for p in fa["backends"]["providers"]}
        self.assertTrue({"fa2", "fa3", "flashinfer", "fa4"} <= fa_libs, fa_libs)
        for p in fa["backends"]["providers"]:
            self.assertIn(p["provisioning"], ("pip", "source_build"))
        # MoE expert GEMM fused op → grouped_gemm backends (torch + DeepGEMM).
        gg = next(p for p in data["plans"] if p["fused_op"] == "grouped_gemm_moe")
        self.assertEqual(gg["backends"]["signature"], "grouped_gemm")
        self.assertTrue(
            {"torch", "deepgemm"} <= {p["library"] for p in gg["backends"]["providers"]})
        # The WHOLE MoE expert block (coarse fusion) → fused_moe backends:
        # torch floor, FlashInfer cutlass (single-GPU), DeepGEMM MegaMoE
        # (EP + symmetric memory, source-build).
        fm = next(p for p in data["plans"] if p["fused_op"] == "fused_moe")
        self.assertEqual(fm["backends"]["signature"], "fused_moe")
        fm_libs = {p["library"] for p in fm["backends"]["providers"]}
        self.assertTrue({"torch", "flashinfer", "deepgemm"} <= fm_libs, fm_libs)
        self.assertIn("deepinfer", fm_libs, fm_libs)
        mega = next(
            p for p in fm["backends"]["providers"]
            if p["kernel_impl"] == "deepgemm_fp8_fp4_mega_moe")
        self.assertEqual(mega["library"], "deepgemm")
        self.assertEqual(mega["provisioning"], "source_build")
        # each provider declares the fusion boundary (span) it realizes —
        # MegaMoE absorbs the whole expert block → fused_op == "fused_moe".
        self.assertEqual(mega["fused_op"], "fused_moe")
        # The fused ops wired to the corpus surface their providers now.
        by_op = {p["fused_op"]: p for p in data["plans"]}
        expected_backends = {
            "fused_qk_rope": ("rope", {"torch", "flashinfer", "vllm", "sglang"}),
            "moe_fused_gate": ("moe_gate", {"torch", "vllm", "sglang"}),
            "fused_qk_norm_rope": ("fused_qk_norm_rope", {"torch", "sglang", "vllm"}),
            "indexer_logits": ("indexer_logits", {"torch", "deepgemm"}),
            "mhc_mega_post_pre": (
                "mhc_mega_post_pre", {"torch", "deepinfer"}),
            "fused_engram_gate": (
                "fused_engram_gate", {"torch", "deepinfer"}),
            "fused_compress_pool_state": (
                "fused_compress_pool_state", {"torch", "deepinfer"}),
            "moe_mega_gate_sqrtsoftplus": (
                "moe_mega_gate_sqrtsoftplus", {"torch", "deepinfer"}),
        }
        for op, (sig, libs) in expected_backends.items():
            b = by_op[op]["backends"]
            self.assertIsNotNone(b, f"{op} should have backends now")
            self.assertEqual(b["signature"], sig)
            self.assertTrue(libs <= {p["library"] for p in b["providers"]},
                            f"{op}: {[p['library'] for p in b['providers']]}")

    def test_structural_empty_backends_carry_a_reason_note(self):
        # gated_act_ffn is a composite span vLLM does not ship as one kernel.
        r = self.client.get("/fusion/registry")
        by_op = {p["fused_op"]: p for p in r.json()["plans"]}
        self.assertIsNone(by_op["gated_act_ffn"]["backends"])
        self.assertTrue(by_op["gated_act_ffn"]["backends_note"])
        # all-reduce+RMSNorm has a 1-rank torch floor (all-reduce is identity).
        ar = by_op["allreduce_fused_rmsnorm"]["backends"]
        self.assertIsNotNone(ar)
        self.assertIn("torch", {p["library"] for p in ar["providers"]})

    def test_plan_payload_has_serialized_pattern(self):
        r = self.client.get("/fusion/registry")
        data = r.json()
        fa = next(p for p in data["plans"] if p["fused_op"] == "flash_attention")
        self.assertIn("naive_pattern", fa)
        self.assertEqual(fa["naive_pattern"]["kind"], "nodes")
        self.assertEqual(
            sorted(fa["naive_pattern"]["nodes"].keys()),
            ["apply", "score", "softmax"],
        )
        # external_inputs are the plan's input ports, in io_mapping order
        # (`NaivePattern.input_ports`) — the optional mask and sinks too —
        # which the page lifts directly into ELK input pills.
        self.assertEqual(
            fa["naive_pattern"]["external_inputs"],
            ["q", "k", "v", "mask", "sinks"],
        )
        self.assertEqual(
            fa["naive_pattern"]["io_mapping"]["output"], "apply.out",
        )
        # plan_file is a repo-relative path so the UI can deep-link to
        # the yaml without revealing the absolute filesystem layout.
        self.assertTrue(
            fa["plan_file"].endswith("flash_attention.yaml"),
            fa["plan_file"],
        )

    def test_pattern_nodes_carry_the_compute_graph_style(self):
        # The page colours a node by the class the compute graph draws the
        # same op in, never by matching op names itself.
        from backend.arch.ops import is_comm_op
        from backend.rendering._consts import STYLE_BY_OP

        data = self.client.get("/fusion/registry").json()
        fa = next(p for p in data["plans"] if p["fused_op"] == "flash_attention")
        self.assertEqual(
            {n["style"] for n in fa["naive_pattern"]["nodes"].values()},
            {"attn"})
        for plan in data["plans"]:
            for node in (plan["naive_pattern"].get("nodes") or {}).values():
                if node["op"]:
                    want = ("comm" if is_comm_op(node["op"])
                            else STYLE_BY_OP.get(node["op"], "op"))
                    self.assertEqual(node["style"], want, node)

    def test_registry_states_the_paging_the_page_size_control_offers(self):
        # The default framework, each binding's own page, and the served
        # union are the declarations' answers, not a list the route keeps.
        from backend.inference.framework_bindings import (
            DEFAULT_FRAMEWORK, FRAMEWORK_BINDINGS)
        from kernel_bench.providers import served_kv_block_sizes

        data = self.client.get("/fusion/registry").json()
        self.assertEqual(data["default_framework"], DEFAULT_FRAMEWORK)
        self.assertIn(DEFAULT_FRAMEWORK, {b["framework"] for b in data["bindings"]})
        self.assertEqual(data["served_kv_block_sizes"],
                         list(served_kv_block_sizes()))
        self.assertTrue(data["served_kv_block_sizes"])
        for b in data["bindings"]:
            self.assertEqual(b["kv_block_size"],
                             FRAMEWORK_BINDINGS[b["framework"]].kv_block_size,
                             b["framework"])

    def test_binding_payload_lists_ops_with_their_source_anchor(self):
        r = self.client.get("/fusion/registry")
        data = r.json()
        vllm = next(b for b in data["bindings"] if b["framework"] == "vllm")
        ops_by_name = {op["fused_op"]: op for op in vllm["ops"]}
        # An entry says where the framework runs the fused op, not which
        # kernel: that is a measured row's identity.
        for op in vllm["ops"]:
            self.assertTrue(op["source_anchor"], op["fused_op"])
            self.assertNotIn("kernel_impl", op)
        self.assertIn(
            "MergedColumnParallelLinear",
            ops_by_name["merged_column_parallel_linear"]["source_anchor"]["text"],
        )
        # Every place the anchor cites is a permalink at the audited commit.
        commit = vllm["audited_at"]["commit"]
        links = [s["url"] for op in vllm["ops"]
                 for s in op["source_anchor"]["segments"] if "url" in s]
        self.assertTrue(links)
        for url in links:
            self.assertTrue(url.startswith(
                f"https://github.com/vllm-project/vllm/blob/{commit}/"), url)


class TestFusionFiredEndpoint(unittest.TestCase):
    """Fired endpoint: runs the framework rewrite against a real spec and reports
    per-fused_op stats."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_fired_endpoint_returns_plan_match_counts(self):
        # llama-7b's spec exercises GQA attention + SwiGLU FFN + RMSNorm,
        # so the vLLM binding's silu_and_mul / parallel_linears /
        # fused_add_rmsnorm / flash_attention should all fire.
        r = self.client.get(
            "/fusion/fired",
            params={
                "model": "meta-llama/Llama-2-7b-hf",
                "framework": "vllm",
                # A retired `version` parameter on an old link is ignored.
                "version": "0.21.0",
            },
        )
        self.assertEqual(r.status_code, 200, r.text[:500])
        data = r.json()
        self.assertEqual(data["framework"], "vllm")
        self.assertTrue(data["audited_at"].startswith("v"))
        self.assertNotIn("version", data)
        # Sanity: flash_attention is a 3-node chain that ships in every
        # MHA/GQA spec — must fire on llama-7b.
        self.assertIn("flash_attention", data["fired"])
        self.assertGreaterEqual(
            data["fired"]["flash_attention"]["match_count"], 1,
        )
        # Each match exposes anchor + folded sibling spec_ids the UI
        # uses for the "Fire status" table.
        sample = data["fired"]["flash_attention"]["matches"][0]
        self.assertIn("anchor", sample)
        self.assertIn("matched", sample)
        self.assertGreaterEqual(len(sample["matched"]), 2)

    def _sglang_fired(self, model, **params):
        r = self.client.get("/fusion/fired", params={
            "model": model, "framework": "sglang", **params})
        self.assertEqual(r.status_code, 200, r.text[:500])
        return r.json()["fired"]

    def test_sglang_fuses_qknorm_rope_only_where_sglang_dispatches_it(self):
        """sglang calls `fused_qk_norm_rope` only from `Qwen3MoeAttention`,
        only with `enable_fused_qk_norm_rope` on (default off;
        srt/server_args.py), BF16 on CUDA (qwen3_moe.py:518-533, 600). By
        default it fires nowhere; with the flag on an H100 it fires on
        Qwen3-30B-A3B and not on dense Qwen3-8B, which has no fused path."""
        on = {"gpu_sku": "H100-SXM",
              "framework_flags": "enable_fused_qk_norm_rope"}
        self.assertNotIn("fused_qk_norm_rope",
                         self._sglang_fired("Qwen/Qwen3-30B-A3B"))
        self.assertIn("fused_qk_norm_rope",
                      self._sglang_fired("Qwen/Qwen3-30B-A3B", **on))
        dense = self._sglang_fired("Qwen/Qwen3-8B", **on)
        self.assertNotIn("fused_qk_norm_rope", dense)
        self.assertIn("fused_qk_rope", dense)
        # Without a GPU the device is unknown, and the kernel is not claimed.
        self.assertNotIn("fused_qk_norm_rope", self._sglang_fired(
            "Qwen/Qwen3-30B-A3B",
            framework_flags="enable_fused_qk_norm_rope"))

    def test_a_decode_only_kernel_is_reported_in_its_phase(self):
        """The gated delta net's decode recurrence is bound with `when:
        phase: [decode]`. The endpoint walks both phases: it fires in
        decode, is `not_in_phase` for prefill, and is not reported as a
        bound kernel the model fails to match."""
        r = self.client.get("/fusion/fired", params={
            "model": "Qwen/Qwen3.8-27B", "framework": "vllm"})
        self.assertEqual(r.status_code, 200, r.text[:500])
        data = r.json()
        op = "fused_recurrent_gated_delta_rule"
        self.assertIn(op, data["fired"])
        self.assertEqual(set(data["fired"][op]["phases"]), {"decode"})
        self.assertNotIn(op, data["unfired_bound"])
        self.assertIn(op, data["phases"]["prefill"]["not_in_phase"])
        self.assertNotIn(op, data["phases"]["prefill"]["unfired_bound"])
        self.assertIn(op, data["phases"]["decode"]["fired"])
        # A phase-independent kernel fires in both walks.
        self.assertEqual(set(data["fired"]["causal_conv1d"]["phases"]),
                         {"prefill", "decode"})

    def test_the_answer_is_the_single_rank_projection(self):
        """The endpoint walks one rank. sglang's grouped Mamba-2 norm fuses
        only while TP divides n_groups; the response says it was answered at
        degree 1 rather than presenting it as an unconditional match."""
        r = self.client.get("/fusion/fired", params={
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "framework": "sglang"})
        self.assertEqual(r.status_code, 200, r.text[:500])
        data = r.json()
        self.assertEqual(data["projection"], "single_rank")
        self.assertIn("mamba2_rmsnorm_gated", data["world_conditioned"])
        self.assertIn("mamba2_rmsnorm_gated", data["fired"])
        self.assertNotIn("causal_conv1d", data["world_conditioned"])

    def test_unknown_framework_returns_404(self):
        r = self.client.get(
            "/fusion/fired",
            params={"model": "meta-llama/Llama-2-7b-hf", "framework": "noframework"},
        )
        self.assertEqual(r.status_code, 404)


class TestFusionPageServed(unittest.TestCase):
    """The new HTML page must be reachable via the same route pattern
    as inference.html / compare.html."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_fusion_html_returns_200(self):
        from backend.api._app import spa_index_path
        r = self.client.get("/fusion.html")
        if spa_index_path():
            self.assertEqual(r.status_code, 200)
            self.assertIn('id="root"', r.text)
            self.assertNotIn('data-page="fusion"', r.text)
            return
        self.assertEqual(r.status_code, 503)
        self.assertNotIn('data-page="fusion"', r.text)

    def test_fusion_page_is_the_react_catalog(self):
        # Catalog wiring lives in the React page. Pin the registry fetch,
        # the DAG helper, and the absence of per-model fire-status (that
        # moved onto Inference hover). Vanilla `frontend/fusion.js` is gone.
        src = (PROJECT_ROOT / "web" / "src" / "pages" / "Fusion.tsx").read_text()
        self.assertIn("/fusion/registry", src)
        self.assertIn("export function patternToDag", src)
        self.assertIn("fusion-framework-chips", src)
        self.assertNotIn("loadFusionFireStatus", src)
        self.assertNotIn("/fusion/fired", src)
        for stale in (
            "fusion-model",
            "fusion-refresh",
            "Check fire status",
            "查询命中情况",
        ):
            self.assertNotIn(stale, src, f"Fusion.tsx: stale {stale!r}")

    def test_nav_links_to_fusion_from_other_pages(self):
        nav = (PROJECT_ROOT / "web" / "src" / "TopNav.tsx").read_text()
        self.assertIn("/fusion.html", nav, "TopNav is missing the Fusion tab")

    def test_fusion_page_is_model_agnostic(self):
        # Per-model status moved off this page (it's now a hover popover
        # on the Inference page). The fusion page is a pure catalog —
        # framework filter chips + plan list + plan detail. Pin that
        # contract so a regression doesn't reintroduce a model input
        # box / "查询命中" button / fire-status legend here.
        html = (PROJECT_ROOT / "web" / "src" / "pages" / "Fusion.tsx").read_text()
        # "unbound" stays on the page as a catalog fact about a plan (no
        # framework binds it), not as a per-model fire state.
        for stale in (
            "fusion-model",       # the old model-id input id
            "fusion-refresh",     # the old "Check fire status" button id
            "Check fire status",  # the old English button label
            "查询命中情况",       # the old Chinese button label
            "bound, unfired",     # legacy state label
            ">fired<",
        ):
            self.assertNotIn(stale, html, f"Fusion.tsx: stale element {stale!r}")
        # Framework filter chips ARE expected.
        self.assertIn("fusion-framework-chips", html)


if __name__ == "__main__":
    unittest.main()


class TestProviderCatalogIsPublished(unittest.TestCase):
    """`/api/bench/providers` publishes the registry, not a maintained list.

    The declaration was moved out of per-kernel lambdas so consumers could SEE
    what each kernel can serve. That only pays off if the catalog comes from the
    registry itself — a hand-maintained copy would drift from what runs, which
    is the drift the whole refactor exists to remove.
    """

    @classmethod
    def setUpClass(cls):
        # Served by the Kernel Bench service (the HA API forwards it), which
        # answers a loopback caller carrying the shared secret.
        from fastapi.testclient import TestClient
        from backend.service.bench import server
        app = server.create_app(start_lake=False)
        app.user_middleware.clear()
        cls.client = TestClient(app)
        from backend.api._app import app as ha_app
        cls.ha = TestClient(ha_app)

    def test_every_registered_kernel_appears_exactly_once(self):
        from kernel_bench.providers import all_providers
        data = self.client.get("/api/bench/providers").json()
        published = [k["kernel_impl"]
                     for lib in data["libraries"] for k in lib["kernels"]]
        registered = [p.kernel_impl for p in all_providers()]
        self.assertEqual(sorted(published), sorted(registered))
        self.assertEqual(data["kernel_count"], len(registered))

    def test_grouped_by_library_with_its_install_facts(self):
        data = self.client.get("/api/bench/providers").json()
        by_name = {lib["library"]: lib for lib in data["libraries"]}
        # Installation is per library, so the catalog groups the way the
        # harness installs. The import gate is per KERNEL and is published as a
        # list: sglang's kernels gate on `sglang` and `sgl_kernel` both, and
        # naming only one of them is how fa4 was declared as `sglang` and 1,436
        # cells failed one at a time.
        self.assertEqual(by_name["cutlass_mla"]["import_names"], ["sgl_kernel"])
        self.assertEqual(by_name["cutlass_mla"]["provisioning"], "source_build")
        self.assertEqual(by_name["cutlass_mla"]["pip"], "")
        self.assertIn("Blackwell", by_name["cutlass_mla"]["build_hint"])
        # PRESENT, not the whole set. The regression this guards is a MISSING
        # name (fa4 above), and a superset still catches that. An extra name
        # can only make a provider stricter — harder to be available, failing
        # loudly — so equality here would break every time a kernel
        # legitimately gates on a submodule, which `Provider.import_name` is
        # defined to allow: "the module that has to import for THIS kernel to
        # run, which is not one per library". `mhc` gates on
        # `sglang.kernels.ops.layernorm.mhc`, a package layout that exists
        # only from 0.5.11, which is the fact that gate is carrying.
        self.assertLessEqual({"sgl_kernel", "sglang"},
                             set(by_name["sglang"]["import_names"]))
        # The framework libraries install the release their binding was
        # audited at; a row still records the version actually installed.
        from backend.inference.framework_bindings import get_binding
        self.assertEqual(
            by_name["sglang"]["pip"],
            f"sglang=={get_binding('sglang').audited_at.release} sglang-kernel")
        self.assertEqual(
            by_name["vllm"]["pip"],
            f"vllm=={get_binding('vllm').audited_at.release}")

    def test_an_undeclared_scope_is_null_not_empty(self):
        # A kernel still on a hand-written gate has promised nothing. Publishing
        # an empty constraint list would read as "serves everything".
        from kernel_bench.providers import all_providers
        data = self.client.get("/api/bench/providers").json()
        caps = {k["kernel_impl"]: k["capabilities"]
                for lib in data["libraries"] for k in lib["kernels"]}
        for p in all_providers():
            if p.serves is None:
                self.assertIsNone(caps[p.kernel_impl], p.kernel_impl)
            else:
                self.assertIsNotNone(caps[p.kernel_impl], p.kernel_impl)
        self.assertEqual(
            data["declared_count"],
            sum(1 for p in all_providers() if p.serves is not None))

    def test_a_cache_format_restriction_is_published(self):
        """The permanent cache-format fact of a launch is part of the
        published capability record, not a comment beside the build: a
        consumer can tell the index-K store serves only an FP8 cache, and
        the packed-UE8M0 rope/insert overload likewise."""
        data = self.client.get("/api/bench/providers").json()
        caps = {k["kernel_impl"]: k["capabilities"]
                for lib in data["libraries"] for k in lib["kernels"]}
        for impl in ("vllm_c_fused_qnorm_rope_kv_insert",
                     "flashmla_sparse_fp8"):
            self.assertEqual(caps[impl]["kv_dtypes"], ["fp8_e4m3"], impl)
        self.assertEqual(caps["flashmla_sparse"]["kv_dtypes"], ["bf16"])
        # The index-K store's cached key is the tensor it outputs: its
        # format is published as the launch's output, not as a cache it
        # touches beside it.
        # The signature makes `kv` material, so the declaration states it
        # (closed-world); what it states is the slot a cell without a cache
        # of its own reads as its activation, never the FP8 record.
        store = caps["sglang_fused_indexer_k_norm_rope_store"]
        self.assertEqual(store["output_dtypes"], ["fp8_e4m3"])
        self.assertEqual(store["kv_dtypes"], ["bf16"])

    def test_a_secondary_write_is_published(self):
        """The indexer-Q launch writes the rescaled head weights fp32 beside
        the FP8 query; the published record states both, not just the
        primary output."""
        data = self.client.get("/api/bench/providers").json()
        caps = {k["kernel_impl"]: k["capabilities"]
                for lib in data["libraries"] for k in lib["kernels"]}
        q = caps["sglang_fused_indexer_q_rope_first_quant"]
        self.assertEqual(q["output_dtypes"], ["fp8_e4m3"])
        self.assertEqual(q["secondary_outputs"],
                         [{"tensor": "weights", "dtype": "fp32"}])

    def test_a_hardware_floor_is_visible(self):
        data = self.client.get("/api/bench/providers").json()
        k = next(k for lib in data["libraries"] for k in lib["kernels"]
                 if k["kernel_impl"] == "cutlass_mla_decode")
        self.assertEqual(k["min_sm"], 100)

    def test_capabilities_reach_the_fusion_backends_view(self):
        # Same facts, in the place the UI already reads them.
        data = self.ha.get("/fusion/registry").json()
        fa = next(p for p in data["plans"] if p["fused_op"] == "flash_attention")
        prefill = next(p for p in fa["backends"]["providers"]
                       if p["kernel_impl"] == "flashinfer_ragged_prefill")
        self.assertEqual(prefill["capabilities"]["phases"], ["prefill"])
