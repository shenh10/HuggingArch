"""Tests for the fusion-scheme + framework-binding registries.

Covers loader positive cases (the shipped schemes load, one binding per
framework loads with its `audited_at`), negative cases (missing required fields, unknown
top-level / op-spec fields rejected, unknown fused_op in binding), and
registry semantics (`get_binding`, etc.).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.fusion_plans import (
    FUSION_SCHEMES,
    reload_fusion_schemes,
)
from backend.inference.framework_bindings import (
    FRAMEWORK_BINDINGS,
    AuditedAt,
    DispatchCondition,
    FrameworkBinding,
    ServingDeployment,
    SourceAnchor,
    SourceLocation,
    _check_dispatch_vocabulary,
    card_class,
    card_dispatch_facts,
    get_binding,
    reload_framework_bindings,
)
from backend.inference.gpu_specs import get_gpu, load_gpu_specs


SHIPPED_SCHEMES_DIR = PROJECT_ROOT / "backend" / "inference" / "fusion_plans"
SHIPPED_BINDINGS_DIR = PROJECT_ROOT / "backend" / "inference" / "framework_bindings"


class TestShippedSchemesLoad(unittest.TestCase):
    """The four shipped schemes load cleanly under the strict validator."""

    def test_shipped_schemes_present(self):
        names = sorted(FUSION_SCHEMES.keys())
        self.assertEqual(
            names,
            [
                "add_attn_res_combine_norm",
                "allreduce_fused_rmsnorm",
                "causal_conv1d",
                "causal_conv1d_packed",
                "flash_attention",
                "flash_dsa",
                "flash_dsa_gated",
                "flash_dsa_sink",
                "fused_add_rmsnorm",
                "fused_compress_norm_rope_store",
                "fused_compress_pool_state",
                "fused_compress_rope_store",
                "fused_dsa_qkv_norm_rope",
                "fused_dsa_qkv_norm_rope_indexer",
                "fused_eh_norm",
                "fused_engram_gate",
                "fused_fwht128_quant",
                "fused_gdn_gating",
                "fused_gdn_post_conv_prep",
                "fused_hc_head",
                "fused_indexer_k_norm_rope_store",
                "fused_indexer_q_rope_first_quant",
                "fused_indexer_q_rope_quant",
                "fused_indexer_q_rope_quant_split",
                "fused_inv_rope",
                "fused_kda_decode",
                "fused_kda_gate",
                "fused_kpool_compress_store",
                "fused_mla_norm_rope_attn_rope_cast",
                "fused_moe",
                "fused_norm_rope",
                "fused_overlap_compress_norm_rope_store",
                "fused_q_kv_rmsnorm",
                "fused_q_kv_rmsnorm_quant",
                "fused_qk_mrope",
                "fused_qk_norm_partial_rope",
                "fused_qk_norm_rope",
                "fused_qk_rmsnorm_rope_gate",
                "fused_qk_rmsnorm_rope_gate_split_gain",
                "fused_qk_rope",
                "fused_qk_rope_partial",
                "fused_qk_rope_partial_q",
                "fused_qnorm_rope",
                "fused_qnorm_rope_kv_insert",
                "fused_qnorm_rope_kv_insert_no_q_norm",
                "fused_recurrent_gated_delta_rule",
                "fused_recurrent_kda",
                "fused_recurrent_kda_per_channel",
                "fused_save_compress_norm",
                "fused_save_compress_norm_decode",
                "gated_act_ffn",
                "gelu_and_mul",
                "gemma_dual_rmsnorm_residual_scalar",
                "gemma_qkv_rmsnorm",
                "gemma_rmsnorm_residual_scalar",
                "grouped_gemm_moe",
                "hc_combine_norm",
                "hc_gate_mix",
                "hc_silu",
                "ihc_post",
                "ihc_pre_gate_reduce",
                "ihc_pre_mix",
                "indexer_block_score",
                "indexer_logits",
                "mamba2_rmsnorm_gated",
                "merged_column_parallel_linear",
                "merged_column_parallel_linear_wide",
                "mhc_fused_post_pre",
                "mhc_mega_post_pre",
                "mhc_post",
                "mhc_pre_big_fuse",
                "mhc_pre_gemm",
                "mhc_pre_gemm_stream",
                "mla_q_concat_kv_insert",
                "moe_fused_gate",
                "moe_mega_gate_sqrtsoftplus",
                "moe_topk_hash_sqrtsoftplus",
                "moe_topk_sigmoid",
                "moe_topk_softmax",
                "moe_topk_softmax_bias",
                "moe_topk_sqrtsoftplus",
                "ple_gate",
                "qkv_parallel_linear",
                "qsa_index_k_compress_store",
                "qsa_index_q_norm_rope",
                "rmsnorm_gated",
                "sigmoid_gate_mul",
                "silu_and_mul",
                "silu_and_mul_with_clamp",
                "situ_and_mul",
                "softcap_logits",
            ],
        )

    def test_each_scheme_carries_naive_pattern(self):
        for name, scheme in FUSION_SCHEMES.items():
            # Either explicit nodes / chain OR parallel form; both are
            # valid pattern declarations.
            has_nodes_or_chain = bool(scheme.naive_pattern.expanded_nodes())
            is_parallel = scheme.naive_pattern.is_parallel()
            self.assertTrue(
                has_nodes_or_chain or is_parallel,
                f"{name}: naive_pattern must declare nodes / chain / parallel",
            )
            self.assertTrue(
                scheme.naive_pattern.io_mapping,
                f"{name}: naive_pattern.io_mapping must be non-empty",
            )

    def test_pattern_vocabulary_stays_readable(self):
        """Pattern labels are the catalog UI's vocabulary, so the shipped
        registry keeps one naming style instead of re-growing abbreviations."""
        snake = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
        legacy = {
            "a_dt", "act_gate", "ape", "cat", "enorm", "f_bias", "f_heads",
            "f_last", "f_split", "fn", "g_bound", "g_first", "g_rate",
            "g_squash", "hnorm", "hw", "hwk", "ik", "ik_cat", "ik_norm",
            "ik_nope", "ik_pe", "ik_rope", "inj_exp", "inj_gate",
            "inj_scale", "inj_sig", "k_nope", "kw", "nope", "o_cat",
            "o_nope", "o_pe", "og_act", "og_heads", "og_split", "out",
            "output_ik", "output_k", "output_kv", "output_v", "output_w",
            "pe", "pooled", "pooled_out", "q_cat", "q_nope",
            "row", "smax", "x1", "x2", "xn",
        }
        for name, scheme in FUSION_SCHEMES.items():
            with self.subTest(scheme=name):
                pattern = scheme.naive_pattern
                labels = [
                    *pattern.expanded_nodes(),
                    *pattern.external_input_names(),
                    *pattern.io_mapping,
                ]
                for label in labels:
                    self.assertRegex(label, snake, f"{name}: {label}")
                    self.assertNotIn(label, legacy, f"{name}: {label}")
                for port, target in pattern.io_mapping.items():
                    if str(target).startswith("external.") or port == "output":
                        continue
                    self.assertTrue(
                        port.endswith("_out"),
                        f"{name}: secondary output {port!r} must use *_out",
                    )

    def test_flash_attention_naive_pattern_matches_known_chain(self):
        scheme = FUSION_SCHEMES["flash_attention"]
        nodes = scheme.naive_pattern.expanded_nodes()
        self.assertEqual(
            nodes["score"].op_choices,
            ["attention_score", "swa_attention_score"],
        )
        self.assertEqual(
            nodes["softmax"].op_choices,
            ["softmax", "softmax_with_sink", "swa_softmax",
             "swa_softmax_with_sink", "swa_softmax_key0_bias"],
        )
        self.assertEqual(
            nodes["apply"].op_choices,
            ["attention_apply", "swa_attention_apply"],
        )
        # Required externals use the lowercase q/k/v display names. The
        # per-head sink is optional (softmax-with-sink only) and is not a
        # required matcher bind.
        self.assertEqual(
            sorted(scheme.naive_pattern.external_input_names()),
            ["k", "q", "v"],
        )
        self.assertEqual(
            nodes["softmax"].optional_inputs, ["external.sinks"],
        )

    def test_indexer_logits_pattern_covers_flat_and_grouped_chains(self):
        """One plan, two spellings of the one weighted-ReLU logits kernel:
        the flat head sum (DeepSeek DSA, QSA) and Step-5's per-provider-group
        sum, whose grouping the naive graph spells as views around the sum.
        The three view nodes are optional pass-throughs, so the flat chain
        binds exactly the nodes it always did, and the output port falls
        through to the sum when the closing transpose is absent."""
        pattern = FUSION_SCHEMES["indexer_logits"].naive_pattern
        nodes = pattern.expanded_nodes()
        self.assertEqual(
            {name: (node.op, node.optional)
             for name, node in nodes.items()},
            {
                "score": ("attention_score", False),
                "relu": ("relu", False),
                "score_weighted": ("mul", True),
                "heads_last": ("transpose", True),
                "group_split": ("view_split", True),
                "head_sum": ("sum_reduce", False),
                "logits": ("transpose", True),
            })
        self.assertEqual(pattern.io_mapping["output"], "logits.out")
        flat = {name: i for i, name in
                enumerate(("score", "relu", "score_weighted", "head_sum"))}
        self.assertEqual(pattern.port_node("logits", flat), "head_sum")

class TestShippedBindingsLoad(unittest.TestCase):
    """One binding per framework loads, named by the framework alone and
    carrying the release its facts were read at."""

    def test_one_binding_per_framework(self):
        self.assertEqual(sorted(FRAMEWORK_BINDINGS),
                         ["deepinfer", "sglang", "vllm"])
        for name, binding in FRAMEWORK_BINDINGS.items():
            self.assertEqual(binding.framework, name)
            self.assertTrue(binding.audited_at.tag.startswith("v"), name)
            self.assertEqual(len(binding.audited_at.commit), 40, name)
        self.assertEqual(get_binding("vllm").audited_at.label("vllm"),
                         f"vllm (audited at {get_binding('vllm').audited_at.tag})")

    def test_deepinfer_binds_only_boundaries_it_really_runs(self):
        """DeepInfer is a one-model runtime: its binding names the V4.1
        launch boundaries that have exact plans, fixes the cache at 128
        tokens, and leaves only launch families the matcher cannot yet
        express (notably repeated-expert MegaMoE) unbound."""
        b = get_binding("deepinfer")
        self.assertIsNotNone(b)
        self.assertEqual(b.kv_block_size, 128)
        self.assertEqual(set(b.fused_ops()), {
            "fused_mla_norm_rope_attn_rope_cast",
            "fused_compress_pool_state", "fused_compress_norm_rope_store",
            "fused_norm_rope", "mhc_mega_post_pre",
            "moe_mega_gate_sqrtsoftplus",
            "fused_engram_gate", "indexer_logits"})
        self.assertNotIn("flash_attention", b.fused_ops())
        self.assertEqual(
            b.get("mhc_mega_post_pre").crosses, "component")

    def test_vllm_binding_covers_vllm_aligned_schemes(self):
        # grouped_gemm_moe is registered as a placeholder but NOT bound in
        # the shipped vLLM binding — binding it would let the single-linear
        # placeholder overmatch dense linears in non-MoE models. The set
        # covers attention + activation + the two merged-linear wrappers +
        # the residual+norm fusion.
        b = get_binding("vllm")
        self.assertIsNotNone(b)
        self.assertEqual(
            set(b.fused_ops()),
            {
                "flash_attention",
                "silu_and_mul",
                "gelu_and_mul",
                "situ_and_mul",
                # DeepSeek-V4's clamped SwiGLU (SiluAndMulWithClamp).
                "silu_and_mul_with_clamp",
                "merged_column_parallel_linear",
                "qkv_parallel_linear",
                "fused_add_rmsnorm",
                "allreduce_fused_rmsnorm",
                "fused_qk_rope",
                # The same rotary launch handed whole heads: partial RoPE in
                # place (GQA, rope first) and MLA's query tail + rope key.
                "fused_qk_rope_partial",
                "fused_qk_rope_partial_q",
                # Multimodal RoPE: `MRotaryEmbedding` -> `triton_mrope`.
                "fused_qk_mrope",
                # Output-gated Qwen attention prologue (q|gate split + QK
                # RMSNorm + partial rope + gate copy), both norm spellings.
                "fused_qk_rmsnorm_rope_gate",
                "fused_qk_rmsnorm_rope_gate_split_gain",
                # MiniMax-M3 per-head QK norm + partial rope.
                "fused_qk_norm_partial_rope",
                "moe_fused_gate",
                # Ungrouped top-K routers: sigmoid + bias, softmax (with and
                # without a bias), DeepSeek-V4's sqrt-softplus and hash.
                "moe_topk_sigmoid",
                "moe_topk_softmax",
                "moe_topk_softmax_bias",
                "moe_topk_sqrtsoftplus",
                "moe_topk_hash_sqrtsoftplus",
                # DSA-specific (V3.2-Exp day-0 / V4 reuse).
                # `flash_dsa` is the stage-3 sparse-attention chain;
                # `indexer_logits` is the stage-1b dense lightning-indexer
                # logits (fp8 paged-MQA-logits kernel — one launch produces
                # the [B,Sq,Lq] logits the top-K selector reads). The
                # indexer K-PROJ+norm path is still NOT fused into one
                # kernel in either backend, so `indexer_k_proj_norm` stays
                # absent — only the logits MQA matmul is a single kernel.
                "flash_dsa",
                # Hy4: the same sparse kernel handed the learnable sink.
                "flash_dsa_sink",
                "indexer_logits",
                # MiniMax-M3 MSA: per-head q·k max-pooled over key blocks.
                "indexer_block_score",
                # DeepSeek-V4 attention: q head norm + q/kv RoPE + SWA cache
                # insert, the overlap compressor's pool + norm + RoPE + store
                # and the no-overlap one's norm + RoPE + store (also the
                # DeepSeek-V4.1 index key's norm + RoPE + store), the
                # indexer's query rotation + weight scaling, and the output's
                # inverse RoPE + FP8 quant.
                "fused_qnorm_rope_kv_insert",
                "fused_qnorm_rope_kv_insert_no_q_norm",
                "fused_overlap_compress_norm_rope_store",
                "fused_compress_norm_rope_store",
                # The q-LoRA and latent-KV a-norms in one launch.
                "fused_q_kv_rmsnorm",
                "fused_q_kv_rmsnorm_quant",
                # DeepSeek-V4.1 Engram: the gate after `wkv`.
                "fused_engram_gate",
                # DeepSeek-V4's hyper-connection head reduce.
                "fused_hc_head",
                # DeepSeek-V4.1's CSA2 compressor: pool + norm, then
                # rope + store.
                "fused_save_compress_norm",
                "fused_save_compress_norm_decode",
                "fused_compress_rope_store",
                # DeepSeek-V3.2 / GLM-5.x DSA: both a-norms, the k_pe rope and
                # the indexer key in one launch.
                "fused_dsa_qkv_norm_rope",
                "fused_dsa_qkv_norm_rope_indexer",
                # The MTP entry: enorm + hnorm + concat.
                "fused_eh_norm",
                # GLM-5.3-Flash's k-pool indexer compressor, and its query
                # rotated into the cached key's basis + FP8 in one launch.
                "fused_kpool_compress_store",
                "fused_fwht128_quant",
                "fused_indexer_q_rope_quant",
                "fused_indexer_q_rope_quant_split",
                "fused_inv_rope",
                # The hyper-connection sublayer boundary, one plan per launch,
                # plus DeepSeek-V4.1's spelling of the projection over the
                # already-flattened stream.
                "mhc_post",
                "mhc_pre_gemm",
                "mhc_pre_gemm_stream",
                "mhc_pre_big_fuse",
                "mhc_fused_post_pre",
                # SSM / linear-attention mixers: the short conv with its SiLU,
                # and the gated output RMSNorm.
                "causal_conv1d",
                "rmsnorm_gated",
                # Gated DeltaNet: prefill's post-conv prep and decode's one
                # recurrent launch (`when: phase`).
                "fused_gdn_post_conv_prep",
                "fused_recurrent_gated_delta_rule",
                # KDA decode (GLM-5.3-Flash's class only).
                "fused_recurrent_kda",
                # Kimi-K3 KDA decode: the whole mixer, and the per-channel
                # gate + recurrence where the whole mixer does not run.
                "fused_kda_decode",
                "fused_recurrent_kda_per_channel",
                # Kimi-K3: the packed q|k|v short conv, the six-way merged
                # KDA input projection, the Attention Residuals readout with
                # its norm (and the residual add before it), and the MLA
                # decode's q concat + latent cache insert.
                "causal_conv1d_packed",
                "merged_column_parallel_linear_wide",
                "add_attn_res_combine_norm",
                "mla_q_concat_kv_insert",
                # HY V4 iHC: the pre's two launches (mix accumulation, then
                # gates + pre-weighted reduce) and the post
                # (out = post * y + residual).
                "ihc_pre_mix",
                "ihc_pre_gate_reduce",
                "ihc_post",
                # Qwen3.8-Flash-Next hyper-connection and PLE kernels.
                "hc_silu",
                "hc_gate_mix",
                "hc_combine_norm",
                "ple_gate",
                # Qwen3.8-Flash-Next QSA: indexer q prep and key compression
                # (one vLLM launch, two spec scopes), and the sparse attention
                # with its output gate as the epilogue.
                "qsa_index_q_norm_rope",
                "qsa_index_k_compress_store",
                "flash_dsa_gated",
                # ... whose split-K decode merge applies the same gate.
                "sigmoid_gate_mul",
            },
        )
        self.assertNotIn("grouped_gemm_moe", b.fused_ops())
        # The old mega-fusion is deliberately UNBOUND now — vLLM has no
        # single kernel for the entire SwiGLU FFN.
        self.assertNotIn("gated_act_ffn", b.fused_ops())

    def test_an_entry_is_where_and_when_not_which_kernel(self):
        """An entry says where the framework's source runs the fused op as
        one launch and when it does. Which kernel launched is the measured
        row's identity, so the binding carries no kernel name."""
        impl = get_binding("vllm").get("flash_attention")
        self.assertIsNotNone(impl)
        self.assertIn("vllm.v1.attention.backends.flash_attn.FlashAttentionImpl",
                      impl.source_anchor.text)
        self.assertFalse(hasattr(impl, "kernel_impl"))
        for binding in FRAMEWORK_BINDINGS.values():
            for op, entry in binding.op_implementations:
                self.assertTrue(entry.source_anchor, f"{binding.framework}:{op}")

    def test_get_binding_returns_none_on_miss(self):
        self.assertIsNone(get_binding("not-a-framework"))

    def test_sglang_binding_present_with_qknorm_rope(self):
        # sglang ships fused_qk_norm_rope (Qwen3-specific CUDA kernel)
        # plus the common-with-vLLM trio (flash_attention, silu_and_mul,
        # fused_add_rmsnorm). The binding loader must accept it and the
        # binding's op set must contain the sglang-unique fusion.
        b = get_binding("sglang")
        self.assertIsNotNone(b)
        self.assertEqual(b.framework, "sglang")
        self.assertIn("fused_qk_norm_rope", b.fused_ops())
        self.assertIn("situ_and_mul", b.fused_ops())

        impl = b.get("fused_qk_norm_rope")
        self.assertIn("Qwen3MoeAttention", impl.source_anchor.text)

    def test_sglang_binds_the_qwen4exp_write_at_its_audited_condition(self):
        """sglang's one Qwen4Exp hyper-connection launch at a vLLM boundary:
        the write fused with the next norm, which the model reaches only on
        a one-token bf16 decode on a capability-major-10 device (the side
        stream's own gate is `== 10`, hence both bounds). The read gates
        stay unbound — sglang folds them into its mix GEMMs."""
        b = get_binding("sglang")
        self.assertIsNone(b.get("hc_silu"))
        self.assertIsNone(b.get("hc_gate_mix"))
        entry = b.get("hc_combine_norm")
        self.assertIsNotNone(entry)
        when = entry.when
        self.assertEqual(when.graph.phase, ("decode",))
        self.assertEqual(when.graph.architectures,
                         ("Qwen4ExpForConditionalGeneration",))
        self.assertEqual(when.device, ("cuda",))
        self.assertEqual(when.activation_dtype, ("bf16",))
        self.assertEqual(when.min_compute_capability, (10, 0))
        self.assertEqual(when.below_compute_capability, (11, 0))
        self.assertEqual(dict(when.graph.dims),
                         {"B": (1,), "n_verify": (1,),
                          "n_hidden_copies": (4,), "d_model": (2560,)})

        def runs(deployment):
            return "hc_combine_norm" in {
                op for op, _c, _x in b.deployed(deployment).dispatched()}

        self.assertTrue(runs(ServingDeployment.at(
            gpu_sku="B200", activation_dtype="BF16")))
        self.assertFalse(runs(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16")))
        self.assertFalse(runs(ServingDeployment.at(
            gpu_sku="B200", activation_dtype="FP16")))

    def test_a_capability_threshold_is_asked_as_the_framework_asks_it(self):
        """`when.min_compute_capability` is a threshold over the SKU sheet's
        `compute_capability` (vLLM `has_device_capability(100)`, sglang
        `is_sm100_or_newer`): H100 fails 10.0, every listed Blackwell and a
        later capability the sheet does not name yet pass, an unknown one
        fails, and a malformed bound is refused at load."""
        from backend.inference.framework_bindings import DispatchCondition
        when = DispatchCondition.parse(
            {"device": ["cuda"], "min_compute_capability": "10.0"}, "test")
        self.assertEqual(when.min_compute_capability, (10, 0))
        self.assertEqual(when.as_dict()["min_compute_capability"], "10.0")

        def at(sku):
            return ServingDeployment.at(gpu_sku=sku, activation_dtype="BF16")

        self.assertFalse(when.holds_for(at("H100-SXM")))
        self.assertTrue(when.holds_for(at("B200")))
        self.assertTrue(when.holds_for(at("GB300")))
        self.assertTrue(when.holds_for(ServingDeployment.of(
            device="cuda", compute_capability="12.0")))
        self.assertFalse(when.holds_for(ServingDeployment.of(device="cuda")))
        with self.assertRaises(ValueError):
            DispatchCondition.parse({"min_compute_capability": "10"}, "test")
        with self.assertRaises(ValueError):
            DispatchCondition.parse({"min_compute_capability": 10.0}, "test")

    def test_the_q_quantising_a_norm_states_vllms_whole_predicate(self):
        """vLLM picks V4.1's q-quantising a-norm launch only where its q
        readers eat MXFP8 through a FlashInfer kernel: CUDA, capability
        >= 10.0 and the readers' format. FlashInfer's shape test and the
        default linear backend are stated assumptions, not asked."""
        b = get_binding("vllm")
        when = b.get("fused_q_kv_rmsnorm_quant").when
        self.assertEqual(when.min_compute_capability, (10, 0))
        self.assertEqual(when.device, ("cuda",))
        self.assertEqual(when.graph.readers_eat, "fp8_token_group32_ue8m0")
        self.assertEqual(when.graph.dims, ())

        def runs(deployment):
            return "fused_q_kv_rmsnorm_quant" in {
                op for op, _c, _x in b.deployed(deployment).dispatched()}

        self.assertFalse(runs(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16")))
        self.assertTrue(runs(ServingDeployment.at(
            gpu_sku="B200", activation_dtype="BF16")))

    def test_capability_bounds_and_roles_are_load_checked(self):
        """`below_compute_capability` is the exclusive upper bound beside
        `min_compute_capability`; `roles` asks the role of ONE named pattern
        node, and names roles of roles.yaml."""
        from backend.inference.framework_bindings import DispatchCondition
        when = DispatchCondition.parse(
            {"below_compute_capability": "10.0",
             "roles": {"norm": ["attention_indexer"]}}, "t")
        self.assertTrue(when.holds_for(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16")))
        self.assertFalse(when.holds_for(ServingDeployment.at(
            gpu_sku="B200", activation_dtype="BF16")))
        self.assertFalse(when.holds_for(ServingDeployment.of(device="cuda")))
        self.assertEqual(when.graph.roles,
                         (("norm", ("attention_indexer",)),))
        self.assertEqual(when.as_dict()["roles"],
                         {"norm": ["attention_indexer"]})
        for bad in ({"roles": {"norm": ["not_a_role"]}},
                    {"roles": ["attention_indexer"]},
                    {"roles": {"norm": []}},
                    {"roles": {}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DispatchCondition.parse(bad, "t")
        with self.assertRaises(ValueError):
            DispatchCondition.parse({"below_compute_capability": 9}, "t")

    def test_routed_is_a_load_checked_bool(self):
        """`routed` asks whether the match sits in the dispatch zone; it is
        a bool and states itself back."""
        from backend.inference.framework_bindings import DispatchCondition
        when = DispatchCondition.parse({"routed": False}, "t")
        self.assertIs(when.graph.routed, False)
        self.assertEqual(when.as_dict(), {"routed": False})
        self.assertIsNone(DispatchCondition.parse({}, "t").graph.routed)
        for bad in ({"routed": "no"}, {"routed": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DispatchCondition.parse(bad, "t")

    def test_a_quantising_launch_is_its_plain_pattern_plus_the_cast(self):
        """The q-quantising a-norm plan is the plain one's pattern plus the
        `quantise_activation` cast as a required member, so it is the larger
        pattern and is tried first: its `when:` decides, and without a cast
        on q the plain plan matches. Two bound plans over one pattern are
        refused at load — which one matched would be yaml order."""
        from backend.inference.framework_bindings import _reject_shared_patterns
        order = [op for op, _impl in get_binding("vllm").op_implementations
                 if op.startswith("fused_q_kv_rmsnorm")]
        self.assertEqual(order, ["fused_q_kv_rmsnorm_quant",
                                 "fused_q_kv_rmsnorm"])
        quant = FUSION_SCHEMES["fused_q_kv_rmsnorm_quant"].naive_pattern
        plain = FUSION_SCHEMES["fused_q_kv_rmsnorm"].naive_pattern
        self.assertEqual(quant.ops() - plain.ops(), {"quantise_activation"})
        _reject_shared_patterns("x", ["fused_q_kv_rmsnorm",
                                      "fused_q_kv_rmsnorm_quant",
                                      "silu_and_mul"])
        with self.assertRaisesRegex(ValueError, "one pattern"):
            _reject_shared_patterns("x", ["fused_q_kv_rmsnorm",
                                          "fused_q_kv_rmsnorm"])

    def test_sglang_asks_the_layer_compress_ratio_for_the_c4_indexer(self):
        """sglang builds `C4Indexer` (the fused query prep) for a layer whose
        `compress_ratio` is 4 or 128 and `DeepseekV41Indexer` for 1 or 2,
        whatever the checkpoint's class string: the entry asks the matched
        layer's walked dim, not the architecture."""
        graph = {op: c for op, c, _x in get_binding("sglang").deployed(
            ServingDeployment.at(gpu_sku="H100-SXM", activation_dtype="BF16")
        ).dispatched()}["fused_indexer_q_rope_quant"]
        self.assertEqual(graph.dims, (("compress_ratio", (4, 128)),))
        self.assertEqual(graph.architectures, ())

    def test_a_dilated_conv_is_not_the_causal_conv_kernel(self):
        """sglang's causal-conv kernel takes no dilation: its entry asks the
        conv's declared `dilation`, and vLLM's (whose dilated Qwen PLE conv
        folds the SiLU in too) asks nothing."""
        graph = {f: {op: c for op, c, _x in get_binding(f).dispatched()}["causal_conv1d"]
                 for f in ("vllm", "sglang")}
        self.assertIsNone(graph["vllm"])
        self.assertEqual(graph["sglang"].dims, (("dilation", (1,)),))

    def test_sglang_runs_the_softmax_router_where_its_model_asks_for_it(self):
        """sglang's softmax router kernel is reached from the model classes
        that build an ungrouped `TopK` (the Qwen MoE family); DeepSeek-V2
        asks for grouped top-k even with one group and runs a torch chain.
        Its softmax kernel also takes no selection bias, so the biased
        softmax router (LongCat) is vLLM's alone."""
        h100 = ServingDeployment.at(gpu_sku="H100-SXM", activation_dtype="BF16")
        sglang = {op: c for op, c, _x in get_binding("sglang").deployed(h100).dispatched()}
        self.assertIn("Qwen3MoeForCausalLM",
                      sglang["moe_topk_softmax"].architectures)
        self.assertNotIn("DeepseekV2ForCausalLM",
                         sglang["moe_topk_softmax"].architectures)
        self.assertNotIn("moe_topk_softmax_bias", sglang)
        vllm = {op: c for op, c, _x in get_binding("vllm").deployed(h100).dispatched()}
        self.assertIsNone(vllm["moe_topk_softmax"])
        self.assertIsNone(vllm["moe_topk_softmax_bias"])

    def test_sglang_rotates_deepseek_v4_q_and_kv_apart(self):
        """sglang's DeepSeek-V4 class (which also serves V4.1 checkpoints)
        never rotates q and the latent KV in one shared-rotary call, so the
        paired-rope entries exclude both checkpoint classes."""
        sglang = {op: c for op, c, _x in get_binding("sglang").dispatched()}
        for op in ("fused_qk_rope", "fused_qk_rope_partial_q"):
            self.assertEqual(
                sglang[op].excluded_architectures,
                ("DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"))
        # vLLM likewise never rotates DeepSeek-V4.1's q and latent KV in one
        # rotary call (below SM100 the q/KV insert does both; on SM10x the
        # mega attention takes the q rotation).
        vllm = {op: c for op, c, _x in get_binding("vllm").dispatched()}
        for op in ("fused_qk_rope", "fused_qk_rope_partial_q"):
            self.assertEqual(vllm[op].excluded_architectures,
                             ("DeepseekV41ForCausalLM",))

    def test_sglang_qknorm_rope_runs_only_where_sglang_runs_it(self):
        """The `when:` is sglang's own dispatch condition: the server flag
        (default off; srt/arg_groups/fields/exec_.py:318-320), the one call
        site `Qwen3MoeAttention` with head_dim in {64, 128, 256} and the
        whole head rotated (qwen3_moe.py:563-588), BF16 (fused_qknorm_rope.py,
        qwen3_moe.py:662) and CUDA (`_is_cuda`). The deployment answers the
        flag, the dtype and the device."""
        b = get_binding("sglang")
        when = b.get("fused_qk_norm_rope").when
        self.assertEqual(when.server_flag, ("enable_fused_qk_norm_rope", False))
        self.assertEqual(when.graph.architectures, ("Qwen3MoeForCausalLM",))
        self.assertEqual(when.graph.head_dim, (64, 128, 256))
        self.assertIs(when.graph.rotation_covers_head, True)
        self.assertEqual((when.activation_dtype, when.device), (("bf16",), ("cuda",)))

        def runs(deployment) -> bool:
            return "fused_qk_norm_rope" in {
                op for op, _c, _x in b.deployed(deployment).dispatched()}

        flag = {"enable_fused_qk_norm_rope": True}
        self.assertFalse(runs(ServingDeployment()))
        self.assertFalse(runs(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16")))
        self.assertTrue(runs(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16", server_flags=flag)))
        self.assertFalse(runs(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="F32", server_flags=flag)))
        self.assertFalse(runs(ServingDeployment.of(
            server_flags=flag, activation_dtype="BF16", device="rocm")))
        # Without a described GPU the device is unknown: not claimed.
        self.assertFalse(runs(ServingDeployment.at(
            gpu_sku=None, activation_dtype="BF16", server_flags=flag)))
        # An unconditional entry is dispatched for every deployment.
        self.assertIn("fused_qk_rope",
                      {op for op, _c, _x in b.dispatched()})

    def test_a_binding_chooses_an_algorithm_and_prices_nothing(self):
        """A binding holds what the FRAMEWORK decides and nothing else: its
        fusion patterns, the paged-KV page its cache manager allocates in,
        and which all-reduce algorithm it runs on a payload (`collectives`).
        No pipeline wire, no bytes or link traversals of its own — what an
        algorithm costs is `comm_volume.ALGORITHMS`', the same under every
        framework.

        `kv_block_size` is here rather than in a request default because the
        framework owns the cache manager; it is a page size, not a wire.
        `ssm_chunk_size` likewise: where the framework reads the chunk
        length it hands the Mamba-2 scan kernel, a tile size, not a wire.
        `edge_dtypes` likewise: the width the kernel this framework
        dispatches to REQUIRES a declared role's output at, which is a fact
        of that kernel's interface and not a price of its own.
        """
        from backend.arch.layout.comm_volume import ALGORITHMS
        from backend.inference.comm_cost import CollectiveRung
        self.assertEqual(
            FrameworkBinding.KEYS,
            {"framework", "audited_at", "op_implementations", "kv_block_size",
             "ssm_chunk_size", "collectives", "precision_launches",
             "edge_dtypes", "weight_loaders", "kernel_dispatches"})
        self.assertEqual(set(CollectiveRung._fields), {
            "algorithm", "max_bytes", "max_bytes_times_world",
            "min_compute_capability", "below_compute_capability",
            "worlds"})
        for binding in FRAMEWORK_BINDINGS.values():
            self.assertFalse(hasattr(binding, "pipeline_wire"))
            self.assertFalse(hasattr(binding, "comm_cost"))
            for _op, rungs in binding.collectives:
                for rung, _graph in rungs:
                    self.assertIn(rung.algorithm, ALGORITHMS)

    def test_weight_loader_is_selected_by_model_architecture(self):
        from types import SimpleNamespace
        from backend.inference.framework_bindings import get_binding
        mimo = SimpleNamespace(architectures=("MiMoV2ForCausalLM",))
        other = SimpleNamespace(architectures=("LlamaForCausalLM",))
        for framework, loader in (("sglang", "whole_chunks"),
                                  ("vllm", "head_gather")):
            binding = get_binding(framework)
            self.assertEqual(binding.weight_loaders_for(mimo, None),
                             {"tp_interleaved": loader})
            self.assertEqual(binding.weight_loaders_for(other, None), {})
        self.assertEqual(
            get_binding("deepinfer").weight_loaders_for(mimo, None), {})

    def test_fused_qk_norm_rope_plan_has_four_node_chain(self):
        # The plan is the (q_norm, k_norm, q_rope, k_rope) chain the
        # matcher looks for in gqa_qknorm_attention components. RoPE is
        # modeled as two independent 1-input rotations (one per tensor)
        # — the runtime kernel still fuses them into a single launch.
        # Matching itself uses op identity and connectivity; this pin keeps
        # the display labels and io_mapping contract stable.
        scheme = FUSION_SCHEMES["fused_qk_norm_rope"]
        nodes = scheme.naive_pattern.expanded_nodes()
        self.assertEqual(
            sorted(nodes.keys()),
            ["k_norm", "k_rope", "q_norm", "q_rope"],
        )
        self.assertEqual(nodes["q_norm"].op, "rmsnorm")
        self.assertEqual(nodes["k_norm"].op, "rmsnorm")
        self.assertEqual(nodes["q_rope"].op, "rope")
        self.assertEqual(nodes["k_rope"].op, "rope")
        self.assertEqual(scheme.naive_pattern.io_mapping["output"], "q_rope.out")
        self.assertEqual(scheme.naive_pattern.io_mapping["k_out"], "k_rope.out")


class TestTheCardsWalkClassIsTheDispatchsCardKeys(unittest.TestCase):
    """A card's walk class (`framework_bindings.card_class`) is the card
    facts a framework's dispatch reads (`DispatchCondition.CARD_KEYS`, read
    off the sheet by `card_dispatch_facts`) plus the peak columns the sheet
    lists — not a second list of them. So:

      - every `when:` key is classified at load as a fact of the card, of
        the deployment or of the walk, and one classified nowhere is a load
        error rather than a key only `holds_for` reads;
      - two cards of one class are dispatched alike by every shipped
        binding. A card fact a condition reads and the class does not would
        break that, which is what the third test shows by dropping one card
        key from the classification.
    """

    def _conditions(self):
        """Every `when:` alternative of every shipped binding."""
        return [(name, op, when) for name, binding in FRAMEWORK_BINDINGS.items()
                for op, impl in binding.op_implementations
                for when in impl.whens]

    def _dispatch(self, when, sku, dtype):
        return when.holds_for(ServingDeployment.at(
            gpu_sku=sku, activation_dtype=dtype))

    def _by_class(self):
        out = {}
        for gpu in load_gpu_specs():
            out.setdefault(card_class(gpu).key(), []).append(gpu.sku)
        return out

    def test_the_shipped_conditions_read_the_card_through_the_card_keys(self):
        """The card keys are not dead: some shipped condition tests each of
        them, and the facts they name are the ones a card answers with."""
        read = {key for _name, _op, when in self._conditions()
                for key, field in DispatchCondition.CARD_KEYS.items()
                if getattr(when, key)}
        self.assertEqual(read, set(DispatchCondition.CARD_KEYS))
        self.assertEqual(
            [name for name, _value in card_dispatch_facts(get_gpu("H100-SXM"))],
            sorted(set(DispatchCondition.CARD_KEYS.values())))

    def test_a_when_key_classified_nowhere_is_refused_at_load(self):
        """A new card condition — `when: {nvlink_domain: ...}` — is a load
        error until it is classified, because the class reads a card
        through the classified keys alone."""
        with mock.patch.object(DispatchCondition, "KEYS",
                               DispatchCondition.KEYS | {"nvlink_domain"}), \
                self.assertRaises(ValueError) as caught:
            _check_dispatch_vocabulary()
        self.assertIn("nvlink_domain", str(caught.exception))
        self.assertIn("CARD_KEYS", str(caught.exception))
        # A card key naming a fact no deployment carries is refused too: the
        # dispatch reads a card through a `ServingDeployment` field.
        with mock.patch.object(
                DispatchCondition, "CARD_KEYS",
                {**DispatchCondition.CARD_KEYS, "device": "nvlink_domain"}), \
                self.assertRaises(ValueError) as caught:
            _check_dispatch_vocabulary()
        self.assertIn("nvlink_domain", str(caught.exception))
        # The shipped vocabulary passes.
        self.assertIsNone(_check_dispatch_vocabulary())

    def test_cards_of_one_class_are_dispatched_alike_by_every_binding(self):
        """Every sheet card, every shipped `when:`: cards of one walk class
        answer it the same way, so one walk of them serves all of them."""
        groups = self._by_class()
        self.assertTrue(any(len(skus) > 1 for skus in groups.values()), groups)
        for key, skus in groups.items():
            for name, op, when in self._conditions():
                for dtype in ("BF16", "F8_E4M3"):
                    answers = {self._dispatch(when, sku, dtype) for sku in skus}
                    self.assertEqual(len(answers), 1,
                                     (key, skus, name, op, dtype))

    def test_a_card_key_the_class_drops_splits_the_dispatch(self):
        """The check above bites: with the capability dropped from the card
        keys — a class that ignores a fact the dispatch reads — cards of one
        class are dispatched differently, which is the silent wrong walk."""
        with mock.patch.object(DispatchCondition, "CARD_KEYS",
                               {"device": "device"}):
            groups = self._by_class()
            split = [(skus, op) for skus in groups.values()
                     for _name, op, when in self._conditions()
                     if len({self._dispatch(when, sku, "BF16")
                             for sku in skus}) > 1]
        self.assertTrue(split, "no shipped condition tests the capability")


class TestLoaderNegativeCases(unittest.TestCase):
    """Loader rejects malformed schemes with file+field-named error messages."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fusion-test-"))
        # Restore the shipped registry after each test.
        self.addCleanup(self._restore_shipped)

    def _restore_shipped(self):
        reload_fusion_schemes(str(SHIPPED_SCHEMES_DIR))
        reload_framework_bindings(str(SHIPPED_BINDINGS_DIR))
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, fname: str, content: str):
        # Every fixture declares the paged-KV page and the scan chunk,
        # because the loader requires both (each is part of a measurement
        # key, so it is declared and never defaulted). These cases are about
        # the OTHER rejections, so both are present and valid in each.
        if "kv_block_size" not in content:
            content = content.replace("op_implementations:",
                                      "kv_block_size: 16\nop_implementations:")
        if "ssm_chunk_size" not in content:
            content = content.replace("op_implementations:",
                                      "ssm_chunk_size: {from_config: [chunk_size]}\n"
                                      "op_implementations:")
        (self.tmp / fname).write_text(content)

    def test_missing_naive_pattern_rejected(self):
        self._write("bad.yaml", """\
fused_op: bad
""")
        with self.assertRaises(ValueError) as ctx:
            reload_fusion_schemes(str(self.tmp))
        self.assertIn("bad.yaml", str(ctx.exception))
        self.assertIn("naive_pattern", str(ctx.exception))

    def test_output_dtype_is_a_port_map_of_known_dtypes(self):
        """`output_dtype` maps an io_mapping OUTPUT port to a dtype the width
        table knows. The scalar form, a non-string value, an unknown dtype,
        an external operand and a port the pattern lacks are all refused."""
        pattern = """\
naive_pattern:
  nodes:
    act:   { op: sigmoid, inputs: [external.gate] }
    gated: { op: mul,     inputs: [external.x, act.out] }
  io_mapping:
    x:      external.x
    gate:   external.gate
    output: gated.out
"""
        for value, needle in (("float32", "mapping"),
                              ("{ output: 4 }", "output"),
                              ("{ output: float77 }", "float77"),
                              ("{ x: float32 }", "not an output port"),
                              ("{ nowhere: float32 }", "not an output port")):
            with self.subTest(value=value):
                self._write("bad.yaml",
                            f"fused_op: bad\noutput_dtype: {value}\n"
                            + pattern)
                with self.assertRaises(ValueError) as ctx:
                    reload_fusion_schemes(str(self.tmp))
                self.assertIn("bad.yaml", str(ctx.exception))
                self.assertIn(needle, str(ctx.exception))
        self._write("bad.yaml",
                    "fused_op: bad\noutput_dtype: { output: float32 }\n"
                    + pattern)
        scheme = reload_fusion_schemes(str(self.tmp))["bad"]
        self.assertEqual(dict(scheme.output_dtype), {"output": "float32"})

    def test_unknown_top_level_field_rejected(self):
        # The loader accepts only `fused_op` and `naive_pattern` at the
        # YAML top level. Every other field that used to live there
        # (op_spec wrapper, category, dtype_env, description, etc.) is
        # rejected so authors get a loud failure rather than a silently
        # ignored field.
        for field, value in (
            ("op_spec", "{ signature: { x: [B, S, H] } }"),
            ("category", "vertical"),
            ("dtype_env", "{ x: activation_dtype }"),
            ("canonical_reference", '"some paper"'),
            ("description", '"something"'),
        ):
            with self.subTest(field=field):
                self._write("bad.yaml", f"""\
fused_op: bad
{field}: {value}
naive_pattern:
  ops: [linear]
  io_mapping: {{ x: linear.x, output: linear.out }}
""")
                with self.assertRaises(ValueError) as ctx:
                    reload_fusion_schemes(str(self.tmp))
                self.assertIn(field, str(ctx.exception))

    _MIRRORED = """\
fused_op: mirrored
naive_pattern:
  nodes:
    a_norm: {{ op: rmsnorm, inputs: [external.a] }}
    b_norm: {{ op: rmsnorm, inputs: [external.b] }}
    a_rope: {{ op: rope, inputs: [a_norm.out]{a_feeds} }}
    b_rope: {{ op: rope, inputs: [b_norm.out]{b_feeds} }}
  io_mapping:
    a:        external.a
    b:        external.b
    output:   a_rope.out
    output_b: b_rope.out
"""

    def test_ports_on_interchangeable_nodes_rejected(self):
        # Two mirror-image tracks: nothing in op or edges says which rope is
        # `output`, so the port name would follow match order.
        self._write("mirrored.yaml", self._MIRRORED.format(a_feeds="", b_feeds=""))
        with self.assertRaises(ValueError) as ctx:
            reload_fusion_schemes(str(self.tmp))
        message = str(ctx.exception)
        self.assertIn("mirrored.yaml", message)
        self.assertIn("a_rope <-> b_rope", message)
        self.assertIn("feeds", message)

    def test_one_sided_feeds_still_breaks_the_symmetry(self):
        # A signature differing on one side is enough: no relabelling can
        # carry a node declaring `feeds: q` onto one declaring nothing.
        self._write("mirrored.yaml", self._MIRRORED.format(
            a_feeds=", feeds: q", b_feeds=""))
        self.assertIn("mirrored", reload_fusion_schemes(str(self.tmp)))

    def test_feeds_breaks_the_symmetry(self):
        self._write("mirrored.yaml", self._MIRRORED.format(
            a_feeds=", feeds: q", b_feeds=", feeds: k"))
        scheme = reload_fusion_schemes(str(self.tmp))["mirrored"]
        nodes = scheme.naive_pattern.nodes
        self.assertEqual((nodes["a_rope"].feeds, nodes["b_rope"].feeds),
                         ("q", "k"))

    def test_equal_feeds_do_not_break_the_symmetry(self):
        self._write("mirrored.yaml", self._MIRRORED.format(
            a_feeds=", feeds: q", b_feeds=", feeds: q"))
        with self.assertRaises(ValueError):
            reload_fusion_schemes(str(self.tmp))

    def test_feeds_outside_the_role_vocabulary_rejected(self):
        self._write("mirrored.yaml", self._MIRRORED.format(
            a_feeds=", feeds: query", b_feeds=", feeds: k"))
        with self.assertRaises(ValueError) as ctx:
            reload_fusion_schemes(str(self.tmp))
        self.assertIn("'query'", str(ctx.exception))

    _PASS_THROUGH = """\
fused_op: through
naive_pattern:
  nodes:
    act:   {{ op: relu, inputs: [external.x] }}
    scale: {{ op: mul, inputs: [{scale_inputs}], optional: true }}
    total: {{ op: sum_reduce, inputs: [scale.out] }}
  io_mapping:
    x:      external.x
    output: {output}
"""

    def test_an_optional_pass_through_loads(self):
        self._write("through.yaml", self._PASS_THROUGH.format(
            scale_inputs="act.out, external.w", output="total.out"))
        nodes = reload_fusion_schemes(str(self.tmp))["through"].naive_pattern.nodes
        self.assertTrue(nodes["scale"].optional)
        self.assertFalse(nodes["total"].optional)

    def test_an_optional_node_off_one_internal_edge_is_rejected(self):
        # With no internal input there is no edge for its reader to read
        # through when the node is absent.
        self._write("through.yaml", self._PASS_THROUGH.format(
            scale_inputs="external.w", output="total.out"))
        with self.assertRaises(ValueError) as ctx:
            reload_fusion_schemes(str(self.tmp))
        self.assertIn("one internal input", str(ctx.exception))

    def test_an_optional_node_naming_a_port_is_rejected(self):
        self._write("through.yaml", self._PASS_THROUGH.format(
            scale_inputs="act.out, external.w", output="scale.out"))
        with self.assertRaises(ValueError) as ctx:
            reload_fusion_schemes(str(self.tmp))
        self.assertIn("names no `io_mapping` port", str(ctx.exception))

    _OPTIONAL_SINK = """\
fused_op: tail
naive_pattern:
  nodes:
    act:   { op: relu,       inputs: [external.x] }
    total: { op: sum_reduce, inputs: [act.out] }
    scale: { op: scale_const, inputs: [total.out], optional: true }
  io_mapping:
    x:      external.x
    output: scale.out
"""

    def test_an_optional_sink_names_its_input_when_absent(self):
        """A chain whose last step some models do not have (a router's
        routed-scaling multiply): the port names the sink when it is there
        and the node before it when it is not."""
        self._write("tail.yaml", self._OPTIONAL_SINK)
        pattern = reload_fusion_schemes(str(self.tmp))["tail"].naive_pattern
        self.assertTrue(pattern.nodes["scale"].optional)
        self.assertEqual(pattern.port_node("scale", {"act": 1, "total": 2,
                                                     "scale": 3}), "scale")
        self.assertEqual(pattern.port_node("scale", {"act": 1, "total": 2}),
                         "total")

    def test_asymmetric_ports_need_no_feeds(self):
        # `fused_add_rmsnorm`'s two outputs differ by edge (the add feeds
        # the rmsnorm), which already fixes each port's node: it loads with
        # no `feeds` at all.
        pattern = FUSION_SCHEMES["fused_add_rmsnorm"].naive_pattern
        self.assertTrue(all(n.feeds is None for n in pattern.nodes.values()))


class TestEstimateCarriesTheServerFlag(unittest.TestCase):
    """`framework_options` on an estimate is the deployment's server flags:
    sglang's Qwen3-MoE runs the fused qk-norm+RoPE kernel only when the
    estimate turns `enable_fused_qk_norm_rope` on."""

    def test_the_flag_switches_the_kernel(self):
        import json
        from backend.inference.capacity import estimate_capacity
        snapshot = (Path(__file__).resolve().parents[3] / "backend" / "arch"
                    / "snapshots" / "Qwen__Qwen3-30B-A3B.json")
        config = json.loads(snapshot.read_text())["config"]

        def fused_ops(options):
            out = estimate_capacity(
                config=config, gpu_sku="H100-SXM", model_id="Qwen/Qwen3-30B-A3B",
                framework="sglang",
                framework_options=options,
                avg_input_tokens=256, avg_output_tokens=16)
            return {row.get("op") for row in out["per_op_timing"]}

        self.assertNotIn("fused_qk_norm_rope", fused_ops(None))
        self.assertIn("fused_qk_norm_rope",
                      fused_ops({"enable_fused_qk_norm_rope": True}))


class TestBindingLoaderNegativeCases(unittest.TestCase):
    """Framework-binding loader rejects bindings that reference unknown fused_ops."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fb-test-"))
        self.addCleanup(self._restore_shipped)

    def _restore_shipped(self):
        reload_framework_bindings(str(SHIPPED_BINDINGS_DIR))
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, fname: str, content: str):
        # Every fixture declares the paged-KV page and the scan chunk,
        # because the loader requires both (each is part of a measurement
        # key, so it is declared and never defaulted). These cases are about
        # the OTHER rejections, so both are present and valid in each.
        if "kv_block_size" not in content:
            content = content.replace("op_implementations:",
                                      "kv_block_size: 16\nop_implementations:")
        if "ssm_chunk_size" not in content:
            content = content.replace("op_implementations:",
                                      "ssm_chunk_size: {from_config: [chunk_size]}\n"
                                      "op_implementations:")
        (self.tmp / fname).write_text(content)

    def test_kernel_dispatches_are_explicit_and_do_not_enter_rewrite(self):
        """A primitive dispatch loads with its provider identity and audited
        anchor, is answered by the deployment, and leaves graph rewriting to
        `op_implementations` alone."""
        self._write("good.yaml", """\
framework: good
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations: {}
kernel_dispatches:
  - op: linear
    library: torch
    impl_id: torch_cublas_linear
    when: {device: [cuda]}
    source_anchor: "the framework's auto linear path calls F.linear (src/linear.py:10-12)"
""")
        binding = reload_framework_bindings(str(self.tmp))["good"]
        self.assertEqual(binding.dispatch_for("linear").state, "no_active")
        deployed = binding.deployed(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16"))
        decision = deployed.dispatch_for("linear")
        self.assertEqual((decision.state, decision.library, decision.impl_id),
                         ("declared", "torch", "torch_cublas_linear"))
        self.assertEqual(deployed.dispatched(), ())

    def test_overlapping_kernel_dispatches_are_ambiguous(self):
        self._write("good.yaml", """\
framework: good
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations: {}
kernel_dispatches:
  - {op: linear, library: torch, impl_id: torch_a, when: {device: [cuda]}, source_anchor: "src/a.py:1"}
  - {op: linear, library: torch, impl_id: torch_b, when: {device: [cuda]}, source_anchor: "src/b.py:2"}
""")
        binding = reload_framework_bindings(str(self.tmp))["good"].deployed(
            ServingDeployment.at(gpu_sku="H100-SXM", activation_dtype="BF16"))
        decision = binding.dispatch_for("linear")
        self.assertEqual((decision.state, decision.reason),
                         ("ambiguous", "ambiguous_kernel_dispatch"))

    def test_one_fused_op_can_have_disjoint_implementations(self):
        """Two span entries for one fused op load only when their conditions
        are provably disjoint. Rewrite still sees fused-op names and graph
        conditions — never a provider kernel name or a file-order priority."""
        from types import SimpleNamespace
        from backend.arch.eval.fusion import Framework
        from backend.inference.fusion_plans import FUSION_SCHEMES

        self._write("good.yaml", """\
framework: good
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  causal_conv1d:
    - source_anchor: "src/prefill.py:1"
      when: {architectures: [A], phase: [prefill]}
    - source_anchor: "src/decode.py:1"
      when: {architectures: [A], phase: [decode]}
""")
        binding = reload_framework_bindings(str(self.tmp))["good"].deployed(
            ServingDeployment.at(gpu_sku="H100-SXM", activation_dtype="BF16"))
        self.assertEqual(len(binding.implementations_for("causal_conv1d")), 2)
        with self.assertRaises(ValueError):
            binding.get("causal_conv1d")

        graph = Framework.of(binding, FUSION_SCHEMES)
        self.assertEqual([fusion.scheme_name for fusion in graph.fusions],
                         ["causal_conv1d", "causal_conv1d"])
        program = SimpleNamespace(architectures=("A",), resolved=None)
        for phase in ("prefill", "decode"):
            selected = []
            for fusion in graph.fusions:
                analysis = SimpleNamespace(rule=SimpleNamespace(phase=phase))
                if fusion.condition is None or fusion.condition.admits_program(
                        program, analysis):
                    selected.append(
                        (fusion.scheme_name, fusion.condition.phase))
            self.assertEqual(
                selected, [("causal_conv1d", (phase,))])

        overlap = """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  causal_conv1d:
    - source_anchor: "src/a.py:1"
      when: {architectures: [A], phase: [prefill]}
    - source_anchor: "src/b.py:1"
      when: {architectures: [A], phase: [prefill]}
"""
        self._write("bad.yaml", overlap)
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("can be active together", str(ctx.exception))

    def test_bad_kernel_dispatches_are_refused(self):
        base = """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations: {}
kernel_dispatches:
"""
        cases = {
            "unknown op": base + "  - {op: no_such_op, library: torch, impl_id: x, when: {device: [cuda]}, source_anchor: 'src/a.py:1'}\n",
            "implicit condition": base + "  - {op: linear, library: torch, impl_id: x, source_anchor: 'src/a.py:1'}\n",
            "match condition": base + "  - {op: linear, library: torch, impl_id: x, when: {dims: {n_verify: [1]}}, source_anchor: 'src/a.py:1'}\n",
            "bad anchor": base + "  - {op: linear, library: torch, impl_id: x, when: {device: [cuda]}, source_anchor: 'a.py:1'}\n",
            "unknown field": base + "  - {op: linear, library: torch, impl_id: x, when: {device: [cuda]}, source_anchor: 'src/a.py:1', weight_dtype: bf16}\n",
        }
        self._write("bad.yaml", base + "  - {op: linear, library: torch, impl_id: x, when: {device: [cuda]}, source_anchor: 'src/a.py:1'}\n"
                    "  - {op: linear, library: torch, impl_id: x, when: {device: [cuda]}, source_anchor: 'src/a.py:1'}\n")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("repeats", str(ctx.exception))
        for name, content in cases.items():
            with self.subTest(case=name):
                self._write("bad.yaml", content)
                with self.assertRaises(ValueError) as ctx:
                    reload_framework_bindings(str(self.tmp))
                self.assertTrue(str(ctx.exception))

    #: An anchor in the shape an edge width must cite: the place the
    #: framework writes the width, and the place the kernel requires it.
    _BOTH = "writes at src/gate.py:10, required at src/topk.py:20"

    def test_bad_edge_dtypes_are_refused(self):
        """A required edge width is a DECLARATION, so every part of it is
        checked at load: a registered role, a known dtype, an anchor citing
        both the write and the requirement that makes it one, and no second
        entry that could answer the same role under a condition that also
        holds. The two `when:` keys refused by name here state facts
        resolved AFTER this width (the launch, and the deployment's
        activation dtype), so asking them would make the answer depend on
        itself; every other refused key is refused by
        `test_an_edge_dtype_when_may_only_ask_what_resolving_it_answers`.
        """
        base = """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations: {}
edge_dtypes:
"""
        both = self._BOTH
        good = (base + "  r: {roles: [router], dtype: F32, "
                f"source_anchor: '{both}'}}\n")
        self._write("bad.yaml", good)
        reload_framework_bindings(str(self.tmp))
        # Each case names what the message must say, so a fixture cannot
        # pass by failing for somebody else's reason.
        cases = {
            "unregistered role": (base + f"  r: {{roles: [not_a_role], dtype: F32, source_anchor: '{both}'}}\n", "roles.yaml"),
            "empty roles": (base + f"  r: {{roles: [], dtype: F32, source_anchor: '{both}'}}\n", "`roles`"),
            "unknown dtype": (base + f"  r: {{roles: [router], dtype: F33, source_anchor: '{both}'}}\n", "`dtype`"),
            "no dtype": (base + f"  r: {{roles: [router], source_anchor: '{both}'}}\n", "`dtype`"),
            "no anchor": (base + "  r: {roles: [router], dtype: F32}\n", "source_anchor"),
            "bare-line anchor": (base + "  r: {roles: [router], dtype: F32, source_anchor: 'gate.py:10'}\n", "source_anchor"),
            "anchor cites one place": (base + "  r: {roles: [router], dtype: F32, source_anchor: 'src/gate.py:10'}\n", "KERNEL requires it"),
            "unknown field": (base + f"  r: {{roles: [router], dtype: F32, source_anchor: '{both}', tensor: output}}\n", "unknown fields"),
            "launch condition": (base + f"  r: {{roles: [router], dtype: F32, when: {{unquantized_bf16_launch: true}}, source_anchor: '{both}'}}\n", "unquantized_bf16_launch"),
            "activation condition": (base + f"  r: {{roles: [router], dtype: F32, when: {{activation_dtype: [BF16]}}, source_anchor: '{both}'}}\n", "activation_dtype"),
            "one role twice": (base + (
                f"  a: {{roles: [router], dtype: F32, when: {{device: [cuda]}}, source_anchor: '{both}'}}\n"
                f"  b: {{roles: [router], dtype: BF16, source_anchor: '{both}'}}\n"), "one required width"),
        }
        for name, (content, says) in cases.items():
            with self.subTest(case=name):
                self._write("bad.yaml", content)
                with self.assertRaises(ValueError) as ctx:
                    reload_framework_bindings(str(self.tmp))
                self.assertIn(says, str(ctx.exception))

    def test_an_edge_dtype_when_may_only_ask_what_resolving_it_answers(self):
        """A condition nobody evaluates must not load.

        `edge_dtypes_for` puts a `when:` to two matchers and no others: the
        deployment's (`DispatchCondition.holds_for`) and the walk's
        (`GraphCondition.admits_program`, which answers the model class and
        the phase). A key bound at a FUSION MATCH — `routed`, `dims`,
        `roles` — is read by neither, so it would hold for every program
        while reading as a restriction. The damage is not hypothetical: two
        entries for one role that differ only on such a key are provably
        disjoint to the overlap check, yet both apply at lookup and FILE
        ORDER picks the width — exactly what "one role, one width" exists to
        prevent. Refusing the key at load makes the declared condition and
        the answered condition the same condition, and the overlap proof
        sound because it runs on the keys the evaluator reads.
        """
        base = """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations: {}
edge_dtypes:
"""
        both = self._BOTH
        cases = {
            # The pair first: it is the one that silently decided a width.
            "two entries differing only on an unanswered key": base + (
                f"  inside:  {{roles: [router], dtype: F32,  when: {{routed: true}},  source_anchor: '{both}'}}\n"
                f"  outside: {{roles: [router], dtype: BF16, when: {{routed: false}}, source_anchor: '{both}'}}\n"),
            "one entry asking an unanswered key": base + (
                "  r: {roles: [router], dtype: F32, "
                "when: {dims: {compress_ratio: [4]}, routed: true}, "
                f"source_anchor: '{both}'}}\n"),
        }
        for name, content in cases.items():
            with self.subTest(case=name):
                self._write("bad.yaml", content)
                with self.assertRaises(ValueError) as ctx:
                    reload_framework_bindings(str(self.tmp))
                self.assertIn("routed", str(ctx.exception))

    def test_disjoint_edge_dtype_conditions_on_one_role_load(self):
        """Two entries for one role are fine while no deployment reaches both
        — a framework whose kernel requires one width on one card generation
        and another elsewhere states two entries, as a fused op states two
        implementations. The proof runs on keys the lookup really reads."""
        both = self._BOTH
        self._write("bad.yaml", f"""\
framework: bad
audited_at: {{ repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }}
op_implementations: {{}}
edge_dtypes:
  on_blackwell:
    roles: [router]
    dtype: F32
    when: {{device: [cuda], min_compute_capability: "10.0"}}
    source_anchor: "{both}"
  before_blackwell:
    roles: [router]
    dtype: BF16
    when: {{device: [cuda], below_compute_capability: "10.0"}}
    source_anchor: "{both}"
""")
        bindings = reload_framework_bindings(str(self.tmp))
        self.assertEqual(len(bindings["bad"].edge_dtypes), 2)

    def test_unknown_fused_op_rejected(self):
        self._write("bad.yaml", """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  not_a_real_fused_op:
    source_anchor: "src/stub.py:1"
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        msg = str(ctx.exception)
        self.assertIn("bad.yaml", msg)
        self.assertIn("not_a_real_fused_op", msg)

    def test_a_when_key_outside_the_vocabulary_is_rejected(self):
        self._write("bad.yaml", """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  fused_qk_norm_rope:
    source_anchor: "src/stub.py:1"
    when:
      yarn: false
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        msg = str(ctx.exception)
        self.assertIn("bad.yaml", msg)
        self.assertIn("yarn", msg)

    def test_a_phase_outside_the_walk_vocabulary_is_rejected(self):
        """`when.phase` names the walk's phases (`dims.PHASES`); a name the
        walk never runs under (`generate`) would leave the entry unread."""
        self._write("bad.yaml", """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  causal_conv1d:
    source_anchor: "src/stub.py:1"
    when:
      phase: [generate]
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        msg = str(ctx.exception)
        self.assertIn("bad.yaml", msg)
        self.assertIn("generate", msg)

    def test_a_phase_is_the_walks_part_of_the_condition(self):
        self._write("good.yaml", """\
framework: good
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  causal_conv1d:
    source_anchor: "src/stub.py:1"
    when:
      phase: [decode]
""")
        when = reload_framework_bindings(str(self.tmp))["good"].get(
            "causal_conv1d").when
        self.assertEqual(when.graph.phase, ("decode",))
        self.assertEqual(when.as_dict(), {"phase": ["decode"]})

    def test_world_local_names_a_world_a_dim_it_splits_and_one_form(self):
        for when, needle in (
                ("{ world: tp_nowhere, dim: n_ssm_groups, divides: true }",
                 "tp_nowhere"),
                ("{ world: tp_attention, dim: d_model, divides: true }",
                 "d_model"),
                ("{ world: tp_attention, dim: n_ssm_groups }", "world_local"),
                ("{ world: tp_attention, dim: n_ssm_groups, divides: true, "
                 "local_in: [8] }", "world_local"),
                ("{ world: tp_attention, dim: n_ssm_groups, local_in: [] }",
                 "world_local"),
                ("{ world: tp_attention, dim: n_ssm_groups, divides: false }",
                 "world_local")):
            with self.subTest(when=when):
                self._write("bad.yaml", f"""\
framework: bad
audited_at: {{ repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }}
op_implementations:
  mamba2_rmsnorm_gated:
    source_anchor: "src/stub.py:1"
    when:
      world_local: {when}
""")
                with self.assertRaises(ValueError) as ctx:
                    reload_framework_bindings(str(self.tmp))
                self.assertIn(needle, str(ctx.exception))

    def test_a_when_list_states_alternatives(self):
        """`when:` as a list holds where any alternative holds: each one the
        deployment satisfies is dispatched with its own graph condition, and
        the entry has no single `when` to read. A one-item list is refused —
        one condition is written as a mapping."""
        self._write("good.yaml", """\
framework: good
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  causal_conv1d:
    source_anchor: "src/stub.py:1"
    when:
      - { architectures: [A], phase: [decode] }
      - { architectures: [B], device: [rocm] }
""")
        binding = reload_framework_bindings(str(self.tmp))["good"]
        entry = binding.get("causal_conv1d")
        self.assertEqual(len(entry.whens), 2)
        with self.assertRaises(ValueError):
            _ = entry.when
        cuda = binding.deployed(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16")).dispatched()
        self.assertEqual([(op, c.architectures, c.phase) for op, c, _x in cuda],
                         [("causal_conv1d", ("A",), ("decode",))])
        undescribed = binding.dispatched()
        self.assertEqual(len(undescribed), 1)
        self._write("good.yaml", """\
framework: good
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  causal_conv1d:
    source_anchor: "src/stub.py:1"
    when:
      - { phase: [decode] }
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("alternatives", str(ctx.exception))

    def test_a_verify_width_is_an_ordinary_dim(self):
        """The step's verify width is asked with `dims`, the existing key
        for a walked dim: `n_verify: [1]` is a step that does not
        speculate. There is no second key for it."""
        self._write("good.yaml", """\
framework: good
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  causal_conv1d:
    source_anchor: "src/stub.py:1"
    when:
      dims: { n_verify: [1] }
""")
        when = reload_framework_bindings(str(self.tmp))["good"].get(
            "causal_conv1d").when
        self.assertEqual(when.graph.dims, (("n_verify", (1,)),))
        self.assertEqual(when.as_dict(), {"dims": {"n_verify": [1]}})
        self._write("bad.yaml", """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  causal_conv1d:
    source_anchor: "src/stub.py:1"
    when:
      spec_decode: false
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("spec_decode", str(ctx.exception))

    def test_a_server_flag_without_its_default_is_rejected(self):
        self._write("bad.yaml", """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  fused_qk_norm_rope:
    source_anchor: "src/stub.py:1"
    when:
      server_flag: { name: enable_x }
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("server_flag", str(ctx.exception))

    def test_a_pipeline_wire_is_rejected(self):
        """How a tensor crosses a pipeline cut is the graph's fact, priced the
        same under every framework: a binding cannot declare it."""
        self._write("bad.yaml", """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations: {}
pipeline_wire:
  shard_over: tp_attention
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("pipeline_wire", str(ctx.exception))

    def test_missing_framework_field_rejected(self):
        self._write("bad.yaml", """\
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations: {}
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("framework", str(ctx.exception))

    def test_a_version_or_a_kernel_name_is_rejected(self):
        """A binding is not versioned (`audited_at` names the release its
        facts were read at) and names no kernel (that is a measured row's
        identity)."""
        self._write("bad.yaml", """\
framework: bad
version: "1.0"
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations: {}
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("version", str(ctx.exception))
        self._write("bad.yaml", """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations:
  flash_attention:
    kernel_impl: flash_attention_v3
    source_anchor: "src/stub.py:1"
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("kernel_impl", str(ctx.exception))

    def test_audited_at_is_required(self):
        self._write("bad.yaml", """\
framework: bad
op_implementations: {}
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("audited_at", str(ctx.exception))

    def test_the_file_is_named_for_its_framework(self):
        self._write("bad__2.0.yaml", """\
framework: bad
audited_at: { repo: acme/fw, tag: v1.0, commit: "0000000000000000000000000000000000000000" }
op_implementations: {}
""")
        with self.assertRaises(ValueError) as ctx:
            reload_framework_bindings(str(self.tmp))
        self.assertIn("bad.yaml", str(ctx.exception))


class TestSourceAnchor(unittest.TestCase):
    """An anchor cites every place as a path from the repository root, so
    each links to the audited commit; a place it cannot link is a load
    error."""

    def test_a_bare_line_is_refused(self):
        # The case that recurs: a `:N` that leaves its file to the reader.
        with self.assertRaises(ValueError) as ctx:
            SourceAnchor.parse("f (pkg/a.py:10, called at :20)", "x")
        self.assertIn(":20", str(ctx.exception))

    def test_a_file_name_without_its_path_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            SourceAnchor.parse("f (a.py:10)", "x")
        self.assertIn("a.py", str(ctx.exception))

    def test_an_anchor_cites_at_least_one_place(self):
        with self.assertRaises(ValueError):
            SourceAnchor.parse("pkg.module.function", "x")

    def test_places_split_the_text_and_link_to_the_commit(self):
        anchor = SourceAnchor.parse(
            "pkg.mod.f (pkg/mod.py:10-12, from csrc/k.cu)", "x")
        self.assertEqual(anchor.locations, (
            SourceLocation("pkg/mod.py", 10, 12), SourceLocation("csrc/k.cu")))
        audited = AuditedAt(tag="v1", commit="c" * 40, repo="acme/fw")
        cited = audited.cite(anchor)
        self.assertEqual("".join(seg["text"] for seg in cited["segments"]),
                         anchor.text)
        self.assertEqual(
            [seg["url"] for seg in cited["segments"] if "url" in seg],
            [f"https://github.com/acme/fw/blob/{'c' * 40}/pkg/mod.py#L10-L12",
             f"https://github.com/acme/fw/blob/{'c' * 40}/csrc/k.cu"])


def _clone_at(commit: str, framework: str):
    """A local clone holding `commit`: the directories named in
    `HA_FRAMEWORK_SRC` (os.pathsep-separated), else this checkout's sibling
    directories whose name mentions the framework."""
    named = os.environ.get("HA_FRAMEWORK_SRC")
    if named:
        dirs = [Path(d) for d in named.split(os.pathsep) if d]
    else:
        dirs = sorted(d for d in PROJECT_ROOT.parent.iterdir()
                      if framework in d.name and d.is_dir())
    for d in dirs:
        if subprocess.run(["git", "-C", str(d), "cat-file", "-e",
                           f"{commit}^{{commit}}"],
                          capture_output=True, check=False).returncode == 0:
            return d
    return None


class TestAnchorsResolveAtTheAuditedCommit(unittest.TestCase):
    """Every place a shipped anchor cites is a file at the audited commit and
    its lines are inside it, so every link opens where it says. Needs a
    local clone of the framework (`_clone_at`); skipped without one."""

    def test_every_cited_place_exists(self):
        checked = 0
        for name, binding in sorted(FRAMEWORK_BINDINGS.items()):
            clone = _clone_at(binding.audited_at.commit, name)
            if clone is None:
                continue
            lengths: dict = {}
            anchors = [(op, impl.source_anchor)
                       for op, impl in binding.op_implementations]
            anchors += [("collectives", a) for a in binding.collective_anchors]
            anchors += [("precision_launches", launch.source_anchor)
                        for launch in binding.precision_launches]
            anchors += [("edge_dtypes", entry.source_anchor)
                        for entry in binding.edge_dtypes]
            anchors += [("weight_loaders", entry.source_anchor)
                        for entry in binding.weight_loaders]
            anchors += [("kernel_dispatches", dispatch.source_anchor)
                        for dispatch in binding.kernel_dispatches]
            for op, anchor in anchors:
                for loc in anchor.locations:
                    if loc.path not in lengths:
                        shown = subprocess.run(
                            ["git", "-C", str(clone), "show",
                             f"{binding.audited_at.commit}:{loc.path}"],
                            capture_output=True, text=True, check=False)
                        lengths[loc.path] = (None if shown.returncode
                                             else len(shown.stdout.splitlines()))
                    n = lengths[loc.path]
                    with self.subTest(framework=name, op=op, place=str(loc)):
                        self.assertIsNotNone(n, "no such file at the commit")
                        if loc.last is not None:
                            self.assertLessEqual(loc.last, n)
            checked += 1
        if not checked:
            self.skipTest("no local clone of a bound framework at its "
                          "audited commit (set HA_FRAMEWORK_SRC)")


if __name__ == "__main__":
    unittest.main()


def test_a_server_flag_the_binding_does_not_read_is_refused():
    """`framework_options` names the framework's server flags its kernels
    depend on; a name none of its conditions reads is refused, not kept as
    a deployment that differs in name only."""
    import pytest

    from backend.inference.framework_bindings import ServingDeployment, get_binding

    sglang = get_binding("sglang")
    sglang.deployed(ServingDeployment.of(
        server_flags={"enable_fused_qk_norm_rope": True}))
    with pytest.raises(ValueError, match="reads no server flag"):
        sglang.deployed(ServingDeployment.of(server_flags={"enable_warp_drive": True}))


class TestAnEstimateConsumesItsFusions(unittest.TestCase):
    """A fused kernel over model-level ops (Qwen3.8-Flash-Next's final
    hyper-connection mixer under vLLM's `hc_silu` / `hc_gate_mix`) must not
    leave the per-op table's spine walk a stage with no node."""

    def test_a_flash_next_vllm_estimate_completes(self):
        from backend.api._models import InferenceEstimateRequest
        from backend.api.estimate_env import resolve_estimate_environment
        from backend.inference.capacity import (
            estimate_pending, finish_estimates)

        request = InferenceEstimateRequest(
            model_id="Qwen/Qwen3.8-Flash-Next", gpu_sku="H100-SXM",
            avg_input_tokens=1024, avg_output_tokens=1,
            parallel={"tp_attention": 1}, framework="vllm")
        env = resolve_estimate_environment(request)
        (result,) = finish_estimates(
            [estimate_pending(avg_input_tokens=1024, **env)])
        self.assertTrue(result)
