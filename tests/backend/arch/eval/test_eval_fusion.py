"""Fusion as a graph transform that hands out a correspondence, not a result.

Falsification targets:
  - cost must survive the rewrite exactly. Every plan asserts cost
    preservation, and here it is `Cost.combine` carrying the layer across rather than a
    synthesised value, so a discrepancy means the correspondence is wrong;
  - the absorbed nodes must STOP EXISTING. `origins` says where a new node's
    value comes from; without `dropped` the originals survive beside the node
    that replaced them and anything summing leaves counts them twice;
  - a node with a consumer outside the match may only be absorbed if the
    scheme exposes it. `fused_add_rmsnorm`'s `add` feeds both the rmsnorm
    beside it AND the next block's add, which is why the plan declares
    `residual_out`; absorbing it silently would cut the residual stream;
  - a one-node pattern is not a fusion. It matches every node of that op in the
    model, which is how `fused_moe` — declared as a single `linear` — matched
    seven plain projections in a dense Llama.
"""
from __future__ import annotations

import sys
import unittest

import pytest
from pathlib import Path
from typing import Any, Optional, Tuple, ClassVar

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.fusion import (
    BoundaryView, Framework, FusedNode, Fusion, materialize_tree,
    rebase_analysis, union_consts)
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import DECODE, Deployment
from backend.arch.eval import traffic
from backend.arch.eval import dim as dim_mod
from tests.snapshots import shared_program, shared_walk
from backend.inference.framework_bindings import ServingDeployment, get_binding
from backend.inference.fusion_plans import FUSION_SCHEMES

MODELS = ["meta-llama__Llama-3.2-1B", "deepseek-ai__DeepSeek-V3",
          "Qwen__Qwen3-30B-A3B", "moonshotai__Kimi-K3"]


def _framework(name: str) -> Framework:
    binding = get_binding(name)
    if binding is None:
        raise KeyError(f"no framework binding for {name}")
    return Framework.of(binding, FUSION_SCHEMES)


def _only(framework: Framework, *schemes: str) -> Framework:
    """`framework` with only the named fusions, each under its own dispatch
    condition. A cache test that asks one launch to keep a read uses it: an
    attention fusion elsewhere in the binding saves the second read of an
    unfused score/apply pair, which is that fusion's effect, not this one's."""
    kept = tuple(f for f in framework.fusions if f.scheme_name in schemes)
    if {f.scheme_name for f in kept} != set(schemes):
        raise KeyError(f"{framework.name} binds none of {schemes}")
    return Framework(framework.name, kept)


def _program(name: str) -> Program:
    """This model's shared program (`tests.snapshots.shared_program`): one
    per model per process, so two deployments of it are recognised as the
    same walk and the graph is expanded once rather than 50 times."""
    return shared_program(name)


def _walked(name: str) -> Any:
    """The analysis a rewrite runs against: the program as the walk ran it.

    Shared (`shared_walk`), which is what this file needed: the walk is
    ~900 ms on a large model and these tests ask for the same one of the
    same model over and over. Every consumer here only READS the analysis --
    `Framework.rewrite` reads it and `rebase_analysis` returns a `replace`d
    copy -- so one object answers them all."""
    return shared_walk(Deployment(_program(name)).at(input_tokens=128))


def _rewrite(transform: Any, name: str) -> Tuple[Any, Any]:
    analysis = _walked(name)
    return transform.rewrite(analysis.program, analysis)


def _leaf_totals(analysis: Any, runs: Any = None) -> Tuple[Any, Any]:
    """Summed over LEAVES, because `total()` sums component roots whose cost
    already includes their children — it would report preservation even if the
    leaf layer were double-counted.

    Summed as the exact numbers the walk produced, not per-node `int()`s: a
    routed expert's per-token share is a rational at small batch (Kimi-K3 at
    128 tokens bills its expert activation 540672/7 FLOPs), and a fused node
    that carries the exact sum of two such leaves truncates once where the two
    leaves truncated twice — 2 FLOPs apart on a total that the rewrite in fact
    preserved to the last unit.

    Weighted by how many times each leaf runs (`runs`, a count layer over
    the analysis's program): a launch across a layer seam runs once per
    seam, and a member it took only some runs of keeps the rest as a leaf
    of its own (`Rewrite.shares`), so a per-node sum would count the member
    in both. Unweighted when `runs` is None."""
    flops = params = 0
    for uid, cost in analysis.cost.values.items():
        if getattr(analysis.program.by_uid(uid), "children", None):
            continue
        n = 1 if runs is None else runs.values[uid]
        flops += n * dim_mod.value_of(cost.flops)
        params += n * dim_mod.value_of(cost.params)
    return flops, params


class TestExpertPoolsSurviveTheRewrite(unittest.TestCase):
    def test_a_fused_analysis_answers_the_walks_pools(self):
        """A kernel fusion merges nodes, not expert pools: the fused
        deployment's analysis states the pools the walk read off its banks."""
        framework = _framework("vllm")
        for name in ("Qwen__Qwen3-30B-A3B", "moonshotai__Kimi-K3",
                     "deepseek-ai__DeepSeek-V3", "openai__gpt-oss-20b"):
            with self.subTest(model=name):
                d = Deployment(_program(name)).at(batch=1, input_tokens=128)
                fused = shared_walk(d.fused(framework))
                self.assertGreater(len(fused.rewrite.origins), 0)
                self.assertTrue(fused.moe_pools)
                self.assertEqual(fused.moe_pools, shared_walk(d).moe_pools)
                self.assertEqual(fused.moe, shared_walk(d).moe)


class TestCostSurvivesTheRewrite(unittest.TestCase):
    def test_leaf_cost_is_preserved_exactly(self):
        framework = _framework("vllm")
        for name in MODELS:
            with self.subTest(model=name):
                d = Deployment(_program(name)).at(batch=1, input_tokens=128)
                walked = shared_walk(d)
                rewritten = shared_walk(d.fused(framework))
                naive = _leaf_totals(walked, walked.instances)
                fused = _leaf_totals(rewritten, walked.instances.rebase_counts(
                    rewritten.rewrite, rewritten.program))
                self.assertEqual(naive, fused)

    def test_own_copies_follow_the_rewrite(self):
        """A survivor keeps its own multipliers; a fused node carries its
        members' when they agree and none when they do not — asking for it
        is then refused rather than answered with one member's count."""
        d = Deployment(_program("Qwen__Qwen3-30B-A3B")).at(
            batch=1, input_tokens=128)
        naive = shared_walk(d)
        fused = shared_walk(d.fused(_framework("vllm")))
        self.assertGreater(len(fused.rewrite.origins), 0)
        for new_uid, sources in fused.rewrite.origins.items():
            node = fused.program.by_uid(new_uid)
            members = [naive.own_copies.values[uid] for uid in sources]
            with self.subTest(node=node.display_path):
                if all(m == members[0] for m in members):
                    self.assertEqual(fused.own_copies_of(node), members[0])
                else:
                    with self.assertRaises(ValueError):
                        fused.own_copies_of(node)
        for uid, copies in naive.own_copies.values.items():
            if uid not in fused.rewrite.dropped:
                self.assertEqual(fused.own_copies.values[uid], copies)

    def test_members_at_different_multipliers_have_no_one_count(self):
        """A two-copy shared-expert bank fused with a single node runs two
        copies beside one; the fused node has no own count to report."""
        from backend.arch.eval.analysis import OwnCopies
        one = OwnCopies(1, 1)
        bank = OwnCopies(2, 2)
        self.assertEqual(OwnCopies.fused((one, one)), one)
        self.assertIsNone(OwnCopies.fused((one, bank)))

    def test_the_graph_actually_shrinks(self):
        framework = _framework("vllm")
        d = Deployment(_program("deepseek-ai__DeepSeek-V3")).at(input_tokens=128)
        naive, fused = shared_walk(d), shared_walk(d.fused(framework))
        self.assertLess(len(fused.cost.values), len(naive.cost.values))
        self.assertGreater(len(fused.rewrite.origins), 0)

    def test_absorbed_nodes_are_dropped_not_kept(self):
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(input_tokens=128)
        fused = shared_walk(d.fused(Fusion.named("fused_add_rmsnorm", FUSION_SCHEMES)))
        for new_uid, sources in fused.rewrite.origins.items():
            self.assertIn(new_uid, fused.cost.values)
            for source in sources:
                # A node a launch across a seam took only some runs of
                # keeps the rest (layer 0's input norm reads the embedding,
                # not a previous layer's add) and so stays; every other
                # absorbed node is gone.
                if fused.rewrite.partly(source):
                    self.assertIn(source, fused.cost.values)
                    continue
                self.assertNotIn(source, fused.cost.values,
                                 "an absorbed node survived its replacement")

    def test_nothing_matched_is_the_identity(self):
        d = Deployment(_program("meta-llama__Llama-3.2-1B")).at(input_tokens=128)
        before = shared_walk(d)
        after = shared_walk(d.fused(Fusion.named("flash_dsa", FUSION_SCHEMES)))
        self.assertTrue(after.rewrite.is_identity(), "Llama has no DSA")
        self.assertEqual(len(before.cost.values), len(after.cost.values))

    def test_swa_chain_fuses_as_flash_attention(self):
        """Serving runs SWA as FA/FlashInfer with window_size, not as the
        naive score/softmax/apply primitives. The matcher has to collapse
        `swa_*` the same way it collapses dense attention, or kernel_bench
        drops every sliding-window layer as unmeasurable."""
        fusion = Fusion.named("flash_attention", FUSION_SCHEMES)
        _, rewrite = _rewrite(fusion, "mistralai__Mistral-7B-v0.1")
        self.assertFalse(rewrite.is_identity(),
                         "Mistral's SWA chain must fuse into flash_attention")

    def test_swa_sink_chain_fuses_as_flash_attention(self):
        """GPT-OSS sliding layers add a per-head sink on softmax. That is
        still one FA kernel; the extra operand is an optional trailing
        external, not a different fused op."""
        fusion = Fusion.named("flash_attention", FUSION_SCHEMES)
        _, rewrite = _rewrite(fusion, "openai__gpt-oss-20b")
        self.assertFalse(rewrite.is_identity(),
                         "GPT-OSS sink-SWA chain must fuse into flash_attention")

    def test_dense_sink_chain_fuses_as_flash_attention(self):
        """GPT-OSS full-attention layers also pass sinks into FA. Dense
        `softmax_with_sink` has to collapse the same way or those layers
        stay unmeasurable primitives."""
        fusion = Fusion.named("flash_attention", FUSION_SCHEMES)
        analysis = _walked("openai__gpt-oss-20b")
        program = analysis.program
        _, rewrite = fusion.rewrite(program, analysis)
        self.assertTrue(any(
            getattr(program.by_uid(uid), "op", None) == "softmax_with_sink"
            for origins in rewrite.origins.values()
            for uid in origins
        ), "GPT-OSS full-attention softmax_with_sink must be absorbed")


class TestWhatMayBeAbsorbed(unittest.TestCase):
    def test_a_one_node_pattern_matches_nothing(self):
        """`ParallelPatternForm` calls single-sibling fusion degenerate and
        rejects it; the chain form had no such check."""
        fusion = Fusion.named("fused_moe", FUSION_SCHEMES)
        self.assertEqual(len(fusion.scheme.naive_pattern.nodes), 1)
        _, rewrite = _rewrite(fusion, "meta-llama__Llama-3.2-1B")
        self.assertTrue(rewrite.is_identity(),
                        "a dense model has no MoE to fuse")

    def test_the_residual_is_exposed_so_the_add_may_be_absorbed(self):
        """`add.out` has two consumers — the rmsnorm and the NEXT block's add —
        so the scheme re-exports it as `residual_out`. Without that the match
        would be refused, and with it absorbed silently the residual is cut."""
        fusion = Fusion.named("fused_add_rmsnorm", FUSION_SCHEMES)
        exposed = fusion._exposed_names(fusion.scheme.naive_pattern)
        self.assertIn("add", exposed)
        _, rewrite = _rewrite(fusion, "meta-llama__Llama-3.2-1B")
        self.assertFalse(rewrite.is_identity())

    def test_an_unexposed_node_with_an_outside_consumer_is_refused(self):
        class Interior:
            io_mapping: ClassVar[dict] = {"output": "rmsnorm.out"}   # `add` NOT re-exported
            parallel = None

            def __init__(self, nodes):
                self.nodes = nodes

            def port_node(self, name, bound):
                return name             # no optional nodes here

        real = Fusion.named("fused_add_rmsnorm", FUSION_SCHEMES)
        pattern = real.scheme.naive_pattern
        guarded = Fusion("fused_add_rmsnorm", type("S", (), {
            "naive_pattern": Interior(pattern.nodes)})())
        _, rewrite = _rewrite(guarded, "meta-llama__Llama-3.2-1B")
        self.assertTrue(rewrite.is_identity(),
                        "the add's residual consumer sits outside the match")


_QK_NORM_ROPE = """
match: {{ model_type: [_qk_norm_rope] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, n_heads: "2", d_head: "8" }}
blocks:
  blk:
    graph:
      q:       {{ op: linear, in: x, dims: {{ in: d_model, out: "n_heads * d_head" }},
                 weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      q_heads: {{ op: head_split, in: q, dims: {{ n_heads: n_heads, d_head: d_head }} }}
      q_norm:  {{ op: rmsnorm, in: q_heads, {norm_width},
                 weight_axes: [{{ replicated: true }}] }}
      q_rope:  {{ op: rope, in: q_norm, dim: {rope_width} }}
      q_flat:  {{ op: head_merge, in: q_rope, dims: {{ n_heads: n_heads, d_head: d_head }} }}
      o:       {{ op: linear, in: q_flat, dims: {{ in: "n_heads * d_head", out: d_model }},
                 weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
    outputs: [o]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }} }}
    per_layer:
      assign:
        - {{ type: b, rest: true }}
      map:
        b: {{ template: blk }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: n_vocab }}, inputs: [layers] }}
"""


_GEMMA_TAILS = """
match: {{ model_type: [_gemma_tails] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size }}
blocks:
  blk:
    graph:
      mlp:      {{ op: linear, in: x, dims: {{ in: d_model, out: d_model }},
                  weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
{branches}
      post_ff:  {{ op: rmsnorm, in: {tail_in}, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      residual: {{ op: add, in: [x, post_ff], dim: d_model }}
      scalar:   {{ op: parameter_vector, dim: "1", weight_axes: [{{ replicated: true }}] }}
      out:      {{ op: mul, in: [residual, scalar], dim: d_model }}
    outputs: [out]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }} }}
    per_layer:
      assign:
        - {{ type: b, rest: true }}
      map:
        b: {{ template: blk }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: n_vocab }}, inputs: [layers] }}
"""

_MOE_BRANCHES = """\
      other:    { op: linear, in: x, dims: { in: d_model, out: d_model },
                  weight_axes: [{ replicated: true }, { replicated: true }] }
      norm_1:   { op: rmsnorm, in: mlp, dim: d_model, weight_axes: [{ replicated: true }] }
      norm_2:   { op: rmsnorm, in: other, dim: d_model, weight_axes: [{ replicated: true }] }
      branches: { op: add, in: [norm_1, norm_2], dim: d_model }"""


def _gemma_tail_ops(moe: bool) -> dict:
    """The sglang gemma-4 tail kernels a one-block synthetic model fuses."""
    import os
    import tempfile

    from backend.arch.loader import SpecLoader

    text = _GEMMA_TAILS.format(branches=_MOE_BRANCHES if moe else "",
                               tail_in="branches" if moe else "mlp")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(text)
        path = f.name
    loader = SpecLoader()
    try:
        spec = loader.load_model_spec(path)
    finally:
        os.unlink(path)
    resolved = loader.resolve(spec, {
        "torch_dtype": "bfloat16", "hidden_size": 16, "num_hidden_layers": 1,
        "vocab_size": 32, "model_type": "_gemma_tails",
        "architectures": ["Gemma4ForCausalLM"]})
    analysis = shared_walk(Deployment(Program.of(resolved)).at(input_tokens=4))
    fused, _ = Framework.of(
        get_binding("sglang").deployed(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16")),
        FUSION_SCHEMES).rewrite(analysis.program, analysis)
    return {n.op: sorted(analysis.program.by_uid(u).name for u in n.origins)
            for n in fused.nodes()
            if isinstance(n, FusedNode) and n.op.startswith("gemma_")}


class TestDispatchConditionReadsTheWalk(unittest.TestCase):
    """`rotation_covers_head` asks whether a rotation spans the whole head
    the norm before it normalised. The widths are the ones the walk bound,
    so the answer does not depend on how the spec spelled them."""

    def _admitted(self, norm_width: str, rope_width: str) -> bool:
        import os
        import tempfile

        from backend.arch.eval.fusion import GraphCondition
        from backend.arch.loader import SpecLoader

        with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                         delete=False) as f:
            f.write(_QK_NORM_ROPE.format(norm_width=norm_width,
                                         rope_width=rope_width))
            path = f.name
        loader = SpecLoader()
        try:
            spec = loader.load_model_spec(path)
        finally:
            os.unlink(path)
        resolved = loader.resolve(spec, {
            "torch_dtype": "bfloat16", "hidden_size": 16, "num_hidden_layers": 1, "vocab_size": 32,
            "model_type": "_qk_norm_rope"})
        analysis = shared_walk(Deployment(Program.of(resolved)).at(input_tokens=4))
        by_name = {n.name: n for n in analysis.program.walked_nodes()}
        bound = {"norm": by_name["q_norm"], "rope": by_name["q_rope"]}
        return GraphCondition(head_dim=(8,), rotation_covers_head=True
                              ).admits_match(bound, analysis)

    def test_a_norm_written_as_a_dims_mapping_is_covered(self):
        self.assertTrue(self._admitted("dim: d_head", "d_head"))
        self.assertTrue(self._admitted("dims: { dim: d_head }", "d_head"))

    def test_a_partial_rotation_does_not_cover_the_head(self):
        self.assertFalse(self._admitted("dims: { dim: d_head }", '"d_head // 2"'))


class TestPhaseIsTheWalks(unittest.TestCase):
    """A `phase` condition is answered by the walk being rewritten: the same
    deployed plan fires on the decode walk and not on the prefill walk, and a
    walk whose rule names no phase admits it in neither."""

    def test_a_decode_only_plan_fires_only_in_decode(self):
        from backend.arch.eval.fusion import GraphCondition

        base = Deployment(_program("Qwen__Qwen3.8-27B")).at(input_tokens=128)
        fusion = Fusion.named("causal_conv1d", FUSION_SCHEMES,
                              GraphCondition(phase=(DECODE,)))

        def fired(analysis):
            fused, _ = fusion.rewrite(analysis.program, analysis)
            return sum(n.op == "causal_conv1d" for n in fused.nodes())

        prefill = shared_walk(base)
        decode = shared_walk(base.during(DECODE, cache_len=4096))
        self.assertEqual(prefill.rule.phase, "prefill")
        self.assertEqual(decode.rule.phase, DECODE)
        self.assertEqual(fired(prefill), 0)
        self.assertGreater(fired(decode), 0)
        # Unconditioned, the same plan fires in both: the phase is the
        # only thing that told them apart.
        plain = Fusion.named("causal_conv1d", FUSION_SCHEMES)
        self.assertGreater(
            sum(n.op == "causal_conv1d"
                for n in plain.rewrite(prefill.program, prefill)[0].nodes()),
            0)

    def test_a_walk_that_names_no_phase_admits_no_phase(self):
        from backend.arch.eval.fusion import GraphCondition
        from backend.arch.eval.rule import Rule

        self.assertIsNone(Rule().phase)
        walk = type("Walk", (), {"rule": Rule()})()
        program = type("P", (), {"architectures": ()})()
        self.assertFalse(GraphCondition(phase=(DECODE,)).admits_program(
            program, walk))
        self.assertTrue(GraphCondition().admits_program(program, walk))


class TestFrameworkDecidesWhichSchemesExist(unittest.TestCase):
    """A `Fusion` is never applied on its own: which kernels a framework ships
    is declared in `framework_bindings/*.yaml`, each entry anchored to real
    source. The vLLM binding records `gated_act_ffn` being REMOVED because vLLM
    has no such kernel and the mega-fusion hid HBM traffic it really performs —
    applying the plan registry directly would reintroduce exactly that."""

    def test_only_declared_schemes_are_applied(self):
        vllm = _framework("vllm")
        names = {f.scheme_name for f in vllm.fusions}
        self.assertIn("flash_attention", names)
        self.assertNotIn("gated_act_ffn", names)
        self.assertNotIn("fused_moe", names)

    def test_frameworks_differ(self):
        """sglang's fused qk-norm+RoPE kernel runs only with its server flag
        on, BF16 on CUDA (the binding's `when:`): absent from the default
        deployment, present on one that turns it on. vLLM has none."""
        vllm = {f.scheme_name for f in _framework("vllm").fusions}
        sglang = {f.scheme_name
                  for f in _framework("sglang").fusions}
        self.assertNotEqual(vllm, sglang)
        self.assertNotIn("fused_qk_norm_rope", sglang)
        flagged = Framework.of(get_binding("sglang").deployed(
            ServingDeployment.at(
                gpu_sku="H100-SXM", activation_dtype="BF16",
                server_flags={"enable_fused_qk_norm_rope": True})),
            FUSION_SCHEMES)
        self.assertIn("fused_qk_norm_rope",
                      {f.scheme_name for f in flagged.fusions} - vllm)

    def test_deepseek_v4_attention_prep_launches_follow_the_framework(self):
        """DeepSeek-V4 on CUDA: vLLM prepares q and the latent KV row in one
        launch and the compressor pools, norms and rotates in another; sglang
        norms + rotates q, the latent KV row and each pooled compressor entry
        in a launch apiece, and never rotates q and k together."""
        analysis = _walked("deepseek-ai__DeepSeek-V4-Flash")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def fused_ops(framework):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return {n.op for n in fused.nodes() if not n.children}

        vllm, sglang = fused_ops("vllm"), fused_ops("sglang")
        self.assertTrue({"fused_qnorm_rope_kv_insert",
                         "fused_overlap_compress_norm_rope_store",
                         "fused_compress_norm_rope_store"} <= vllm)
        self.assertFalse({"fused_overlap_compress_norm_rope_store",
                          "fused_compress_norm_rope_store"} & sglang)
        self.assertFalse({"fused_qnorm_rope", "fused_norm_rope"} & vllm)
        self.assertTrue({"fused_qnorm_rope", "fused_norm_rope"} <= sglang)
        self.assertFalse({"fused_qk_rope", "fused_qk_rope_partial_q",
                          "fused_qnorm_rope_kv_insert"} & sglang)

    def test_a_partial_rotation_is_rotated_in_place(self):
        """Both frameworks hand the rotary kernel whole heads and rotate the
        first `rotary_dim` of each in place, so GLM-4.5-Air's re-join of the
        rotated half with the pass-through half is no copy: one launch takes
        both slices, both ropes and both concats. MLA rotates the query's
        tail through a view and the rope key whole (DeepSeek-V3 prefill); its
        key concat joins a different tensor and stays a copy, and in the
        absorbed decode form the query's concat does too. A whole-head
        rotation (Llama) is untouched."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def launches(name, framework, phase=None):
            d = Deployment(_program(name)).at(input_tokens=128)
            analysis = shared_walk(d.during(DECODE, cache_len=4096) if phase else d)
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return {node.op: sorted(analysis.program.by_uid(u).op
                                    for u in node.origins)
                    for node in fused.nodes()
                    if node.op.startswith("fused_qk_rope")}

        for framework in ("vllm", "sglang"):
            glm = launches("zai-org__GLM-4.5-Air", framework)
            self.assertEqual(glm, {"fused_qk_rope_partial": [
                "concat", "concat", "rope_emb", "rope_emb",
                "slice", "slice", "slice", "slice"]})
            dsv3 = launches("deepseek-ai__DeepSeek-V3", framework)
            self.assertEqual(dsv3, {"fused_qk_rope_partial_q": [
                "concat", "rope", "rope_emb", "slice", "slice"]})
            self.assertEqual(
                set(launches("deepseek-ai__DeepSeek-V3", framework, DECODE)),
                {"fused_qk_rope"})
            self.assertEqual(
                set(launches("meta-llama__Llama-3.2-1B", framework)),
                {"fused_qk_rope"})

    def test_sglang_never_pairs_deepseek_v41s_q_and_kv_rotations(self):
        """sglang rotates DeepSeek-V4.1's query and latent KV in two
        launches; no fused node may hold both rotations there. vLLM's wider
        q + KV rope launch may."""
        analysis = _walked("deepseek-ai__DeepSeek-V4.1-Flash")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def pairs(framework):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return [n for n in fused.nodes() if isinstance(n, FusedNode)
                    and sum(analysis.program.by_uid(u).op == "rope_emb"
                            for u in n.origins) >= 2]

        self.assertEqual(pairs("sglang"), [])
        self.assertTrue(pairs("vllm"))

    def test_an_absorbed_cache_entry_keeps_its_cache_bytes(self):
        """GLM-4.5-Air's key cache is written from `key_concat`, which the
        partial-rope launch absorbs: the step reads the same cache bytes fused
        as unfused."""
        analysis = (shared_walk(Deployment(_program("zai-org__GLM-4.5-Air"))
                    .at(batch=2, input_tokens=4096)
                    .during(DECODE, cache_len=4096)))
        fused, rewrite = Framework.of(
            get_binding("vllm").deployed(ServingDeployment.at(
                gpu_sku="H100-SXM", activation_dtype="BF16")),
            FUSION_SCHEMES).rewrite(analysis.program, analysis)
        self.assertIn("fused_qk_rope_partial", {n.op for n in fused.nodes()})

        def cache_reads(walk):
            return sum(int(t.kv_reads) for t in
                       traffic.derive(walk).leaves().values.values())

        self.assertEqual(cache_reads(rebase_analysis(analysis, fused, rewrite)),
                         cache_reads(analysis))

    def test_dspark_branch_coverage_survives_fused_walks(self):
        """A rewritten external K/V edge still resolves to the branch that
        made it; fusion may move the row, not lose the uncovered span.

        Unfused, the score and the apply are two kernels and each reads the
        selected window pairs; both frameworks run them as one attention
        launch, which reads them once — half the unfused bill."""
        from backend.arch.eval import cache

        analysis = (shared_walk(Deployment(
            _program("deepseek-ai__DeepSeek-V4.1-Flash"))
            .at(batch=1, input_tokens=64, s_prefix_cached=32)
            .during("prefill")))

        def dspark_cache_reads(walk):
            return sum(int(value.kv_reads) for uid, value in
                       traffic.derive(walk).values.items()
                       if (node := walk.program.by_uid(uid)) is not None
                       and "dsv41_dspark_attention" in node.display_path)

        unfused = dspark_cache_reads(analysis)
        self.assertEqual(unfused, 2 * 2458368)
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        for name in ("vllm", "sglang"):
            with self.subTest(framework=name):
                fused, rewrite = Framework.of(
                    get_binding(name).deployed(deployment),
                    FUSION_SCHEMES).rewrite(analysis.program, analysis)
                rebased = rebase_analysis(analysis, fused, rewrite)
                self.assertEqual(cache.context_span_gaps(rebased), ())
                self.assertEqual(dspark_cache_reads(rebased), 2458368)

    def test_the_indexer_query_prep_fuses_where_its_framework_fuses_it(self):
        """The indexer's query preparation: rope + V4's Hadamard +
        head-weight scaling is one launch in vLLM and sglang for DeepSeek-V4;
        V4.1 has no Hadamard and its split sglang kernel runs only in SM100+
        decode."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def indexer_origins(model, framework, op):
            analysis = _walked(model)
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return sorted(analysis.program.by_uid(u).name
                          for n in fused.nodes()
                          if n.op == op
                          for u in n.origins)

        for framework in ("vllm", "sglang"):
            with self.subTest(framework=framework):
                self.assertIn("q_rotate", indexer_origins(
                    "deepseek-ai__DeepSeek-V4-Flash", framework,
                    "fused_indexer_q_rope_quant"))
        self.assertIn("q_pe_rope", indexer_origins(
            "deepseek-ai__DeepSeek-V4.1-Flash", "vllm",
            "fused_indexer_q_rope_quant_split"))
        self.assertEqual(indexer_origins(
            "deepseek-ai__DeepSeek-V4.1-Flash", "sglang",
            "fused_indexer_q_rope_quant_split"), [])

    def test_step5s_grouped_indexer_logits_is_one_launch(self):
        """Step-5's indexer sums the weighted-ReLU scores per PROVIDER GROUP:
        the publisher's Triton kernel (`_indexer_logits_kernel`, the
        stepfun-ai__Step-5-Preview snapshot's kernel.py:427-499) runs score +
        ReLU + head-weighted group sum in one launch per (query tile, key
        tile, group), so the per-head score never lands in HBM. The spec
        spells the grouping as views around the head sum (transpose →
        view_split → sum_reduce → transpose), and the plan absorbs all seven
        nodes into one `indexer_logits` launch, leaving the top-k its own
        launch, as the kernel boundary has it (`region_topk_*`,
        kernel.py:605-713, wrappers :814/:852). DeepSeek's flat chain — the same sum over every
        indexer head at once, no grouping views — matches the same plan with
        exactly the four nodes it always did."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def indexer(model, framework):
            analysis = _walked(model)
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            nodes = [n for n in fused.nodes()
                     if isinstance(n, FusedNode) and n.op == "indexer_logits"]
            origins = [sorted(analysis.program.by_uid(u).name
                              for u in n.origins) for n in nodes]
            survivors = {n.name for n in fused.walked_nodes()
                         if not n.children and not isinstance(n, FusedNode)}
            return origins, survivors

        grouped = sorted(("index_score", "index_relu", "index_weighted",
                          "index_heads_last", "index_group_split",
                          "index_group_sum", "index_logits"))
        for framework in ("vllm", "sglang"):
            with self.subTest(framework=framework):
                origins, survivors = indexer(
                    "stepfun-ai__Step-5-Preview", framework)
                self.assertTrue(origins)
                self.assertTrue(all(o == grouped for o in origins))
                self.assertFalse(set(grouped) & survivors)
                self.assertIn("index_topk", survivors)
                flat, _ = indexer("deepseek-ai__DeepSeek-V4.1-Flash",
                                  framework)
                self.assertTrue(flat)
                self.assertTrue(all(
                    o == ["reduced", "score", "score_relu", "weighted"]
                    for o in flat))

    def test_the_a_norm_pair_is_one_launch_only_where_vllm_fuses_it(self):
        """vLLM norms the q-LoRA and the latent KV of DeepSeek-V4 in one
        launch (`fused_q_kv_rmsnorm`); DeepSeek-V3's identical pair runs as
        two (the MLA wrapper's `fuse_qkv_rmsnorm` defaults off), and sglang
        never fuses an a-norm pair. The MTP block's `enorm` / `hnorm` each feed the
        whole layer after them — q and k alike — so neither is the query's
        side, and they are not taken for a q/kv pair."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def fused(model, framework):
            analysis = _walked(model)
            program, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return [sorted(analysis.program.by_uid(u).name for u in n.origins)
                    for n in program.nodes() if n.op == "fused_q_kv_rmsnorm"]

        v4 = fused("deepseek-ai__DeepSeek-V4-Flash", "vllm")
        self.assertTrue(v4)
        self.assertTrue(all(pair == ["kv_norm", "q_norm"] for pair in v4))
        self.assertFalse(fused("deepseek-ai__DeepSeek-V4-Flash", "sglang"))
        self.assertFalse(fused("deepseek-ai__DeepSeek-V3", "vllm"))

    def test_sglang_runs_the_per_head_qk_norms_as_one_launch_where_its_class_does(self):
        """sglang's shared `apply_qk_norm` runs the per-head Q and K RMSNorms
        as one in-place launch (`fused_inplace_qknorm`) for the classes that
        call it — Qwen3 among them — on prompt and decode steps alike. HY V3
        has the same per-head pair in front of the same rotation but norms
        q and k with two plain RMSNorm calls, so the class, not the
        pattern, decides."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def pairs(model, phase):
            base = Deployment(_program(model)).at(input_tokens=128)
            walk = shared_walk(base.during(DECODE, cache_len=4096)
                               if phase else base)
            program, _ = Framework.of(
                get_binding("sglang").deployed(deployment),
                FUSION_SCHEMES).rewrite(walk.program, walk)
            return [sorted(walk.program.by_uid(u).name for u in n.origins)
                    for n in program.nodes() if n.op == "fused_q_kv_rmsnorm"]

        for phase in (None, DECODE):
            qwen3 = pairs("Qwen__Qwen3-8B", phase)
            self.assertTrue(qwen3)
            self.assertTrue(all(p == ["k_norm", "q_norm"] for p in qwen3))
            self.assertFalse(pairs("tencent__Hy3", phase))

    def test_v41_attention_prep_is_the_v4_kernel_without_its_q_norm(self):
        """DeepSeek-V4.1 has no query head norm; vLLM runs its q/kv rope and
        SWA insert through the V4 kernel with `apply_q_norm=False`, so every
        attention form's slices, ropes and re-joins are one launch, and
        q and k are never rotated as a pair on their own. The indexer key
        is `indexer_k_norm_rope_store`, one launch too."""
        analysis = _walked("deepseek-ai__DeepSeek-V4.1-Flash")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        fused, _ = Framework.of(get_binding("vllm").deployed(deployment),
                                FUSION_SCHEMES).rewrite(analysis.program,
                                                        analysis)
        ops = [n.op for n in fused.nodes() if not n.children]
        self.assertIn("fused_qnorm_rope_kv_insert_no_q_norm", ops)
        # The indexer key, on the layers that own one: norm + rope + store.
        self.assertIn("fused_compress_norm_rope_store", ops)
        self.assertNotIn("fused_norm_rope", ops)
        self.assertNotIn("fused_qk_rope", ops)
        self.assertNotIn("fused_qnorm_rope_kv_insert", ops)

    def test_the_engram_gate_is_one_launch_in_both_frameworks(self):
        """DeepSeek-V4.1's Engram after `wkv` (split, per-copy norms, the
        signed-sqrt sigmoid gate, the gated add) is one Triton launch in
        vLLM and sglang alike; the lookup and `wkv` stay their own."""
        analysis = _walked("deepseek-ai__DeepSeek-V4.1-Flash")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        for framework in ("vllm", "sglang"):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            gates = [n for n in fused.nodes() if n.op == "fused_engram_gate"]
            self.assertTrue(gates, framework)
            for node in gates:
                absorbed = {analysis.program.by_uid(u).op for u in node.origins}
                self.assertNotIn("linear", absorbed)
                self.assertNotIn("hash_embedding", absorbed)
                self.assertEqual(len(node.origins), 17)

    def test_the_hc_head_is_one_launch_in_vllm(self):
        """DeepSeek-V4's hyper-connection head — norm, the fn GEMM, the
        sigmoid gate and the weighted reduce over the copies — is one launch
        at the model exit in both frameworks (vLLM also runs it fused in the
        MTP head; sglang's NextN head is plain torch, a draft-only row this
        binding over-fuses until a per-stack condition exists). Binding it in
        both also keeps the 3-node pre-mix GEMM plan off the exit's
        same-shaped first three leaves."""
        analysis = _walked("deepseek-ai__DeepSeek-V4-Flash")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def heads(framework):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return [n for n in fused.nodes() if n.op == "fused_hc_head"]


        for framework in ("vllm", "sglang"):
            nodes = heads(framework)
            self.assertEqual(len(nodes), 2, framework)
            for node in nodes:
                self.assertEqual(
                    sorted(analysis.program.by_uid(u).op
                           for u in node.origins),
                    sorted(["view_merge", "rmsnorm_unscaled",
                            "linear_external_weight", "mul", "add", "sigmoid",
                            "broadcast", "mul", "sum_reduce"]), framework)

    def test_v41_compressor_is_two_vllm_launches_in_each_phase(self):
        """vLLM runs DeepSeek-V4.1's CSA2 compressor as the pool + norm
        (`fused_save_compress_norm`) and then the rope + store
        (`rope_quant_insert`); the prefill and decode spellings of the pool
        reach the same kernel. sglang binds the same pair on SM90+ (its
        `fused_low_ratio_compress_supported` is capability major >= 9)."""
        program = _program("deepseek-ai__DeepSeek-V4.1-Flash")
        prefill = Deployment(program).at(input_tokens=128)
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        for analysis, pool in (
                (shared_walk(prefill), "fused_save_compress_norm"),
                (shared_walk(prefill.during(DECODE, cache_len=4096)),
                 "fused_save_compress_norm_decode")):
            for framework, expected in (("vllm", True), ("sglang", True)):
                fused, _ = Framework.of(
                    get_binding(framework).deployed(deployment),
                    FUSION_SCHEMES).rewrite(analysis.program, analysis)
                ops = {n.op for n in fused.nodes() if not n.children}
                self.assertEqual({pool, "fused_compress_rope_store"} <= ops,
                                 expected, (framework, pool))

    def test_dsa_input_prep_is_one_vllm_launch_per_layer(self):
        """vLLM's DeepSeek-V3.2 / GLM-5 attention norms q-LoRA and the
        latent, rotates k_pe and prepares the indexer key in one launch
        (`fused_norm_rope`); q_pe rotates in the next launch, so q and k are
        not a rope pair there. sglang rotates them together and norms
        separately. GLM-5.2's shared-indexer layers take the indexer-free
        form."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        analysis = _walked("zai-org__GLM-5.2")

        def ops(framework):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return {n.op for n in fused.nodes() if not n.children}

        vllm, sglang = ops("vllm"), ops("sglang")
        self.assertTrue({"fused_dsa_qkv_norm_rope",
                         "fused_dsa_qkv_norm_rope_indexer"} <= vllm)
        self.assertNotIn("fused_qk_rope", vllm)
        self.assertFalse({"fused_dsa_qkv_norm_rope",
                          "fused_dsa_qkv_norm_rope_indexer"} & sglang)
        self.assertIn("fused_qk_rope", sglang)

    def test_the_mtp_entry_norms_fuse_where_each_framework_fuses_them(self):
        """enorm + hnorm + concat is one launch in sglang's DeepSeek NextN
        draft (DeepSeek-V3 included) and in vLLM's V3.2 / GLM-5 MTP; vLLM's
        DeepSeek-V3 MTP runs the two norms and the cat apart."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def fused(model, framework):
            analysis = _walked(model)
            program, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return any(n.op == "fused_eh_norm" for n in program.nodes())

        self.assertTrue(fused("zai-org__GLM-5", "vllm"))
        self.assertTrue(fused("zai-org__GLM-5", "sglang"))
        self.assertTrue(fused("deepseek-ai__DeepSeek-V3", "sglang"))
        self.assertFalse(fused("deepseek-ai__DeepSeek-V3", "vllm"))

    def test_v41_a_norms_write_mxfp8_q_where_vllm_selects_it(self):
        """vLLM's V4.1 a-norm launch writes q already MXFP8 where every q
        reader eats the MXFP8 activation through a FlashInfer linear
        (compute capability >= 10.0) and q_lora_rank passes FlashInfer's
        shape test, and is the plain launch everywhere else — on Hopper, and
        on Blackwell when the projections take another format (block-128 FP8
        weights eat a per-128 activation). DeepSeek-V4 has no quantising
        form. Counted on the serving graph: the quantising launch takes the
        q cast in as a member, and only the serving graph has the cast."""
        from backend.arch.precision_profile import PrecisionProfile
        from backend.inference.serving import rewrite_serving_graph
        program = _program("deepseek-ai__DeepSeek-V4.1-Flash")
        shipped = Deployment(program).at(input_tokens=128)
        fp8 = shipped.with_precision(PrecisionProfile.from_request(
            {"preset": "w_fp8_kv_fp8"}))
        v4 = _walked("deepseek-ai__DeepSeek-V4-Flash")

        def count(walk, sku):
            fused = rewrite_serving_graph(
                walk, framework_binding=get_binding("vllm").deployed(
                    ServingDeployment.at(gpu_sku=sku,
                                         activation_dtype="BF16"))).program
            ops = [n.op for n in fused.nodes() if not n.children]
            return (ops.count("fused_q_kv_rmsnorm"),
                    ops.count("fused_q_kv_rmsnorm_quant"))

        for phase in (None, DECODE):
            walk = shared_walk(shipped.during(DECODE, cache_len=4096) if phase
                    else shipped)
            plain, quant = count(walk, "H100-SXM")          # SM90: plain
            self.assertTrue(plain and not quant)
            self.assertEqual(count(walk, "B200"), (0, plain))  # SM100: quant
        self.assertEqual(count(shared_walk(fp8), "B200"), (11, 0))  # other format
        self.assertEqual(count(v4, "B200")[1], 0)

    def test_v41_attention_prep_follows_vllms_sm100_default(self):
        """vLLM v0.32.0's V4.1 attention on SM10x is FlashMLA mega attention:
        the q/KV insert no longer rotates q, so the q/KV insert plan holds
        below 10.0 only, and no plan pairs the q and latent-KV rotations on
        any GPU. The indexer key's norm → rope → store is its own launch on
        both, the store included; the latent KV's same-shaped chain is not
        (its norm is in the a-norm launch, and no store marker ends it)."""
        analysis = _walked("deepseek-ai__DeepSeek-V4.1-Flash")

        def fused(sku):
            out, _ = Framework.of(
                get_binding("vllm").deployed(ServingDeployment.at(
                    gpu_sku=sku, activation_dtype="BF16")),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return [(n.op, sorted(analysis.program.by_uid(u).name
                                  for u in n.origins))
                    for n in out.nodes() if isinstance(n, FusedNode)]

        h100, b200 = fused("H100-SXM"), fused("B200")
        self.assertEqual(sum(op == "fused_qnorm_rope_kv_insert_no_q_norm"
                             for op, _ in h100), 11)
        self.assertFalse(any(op == "fused_qnorm_rope_kv_insert_no_q_norm"
                             for op, _ in b200))
        for ops in (h100, b200):
            self.assertFalse(any(op.startswith("fused_qk_rope")
                                 for op, _ in ops))
            self.assertFalse(any(op == "fused_norm_rope" for op, _ in ops))
            store = [names for op, names in ops
                     if op == "fused_compress_norm_rope_store"]
            self.assertEqual(len(store), 3)
            self.assertTrue(all({"k_norm", "k_cache"} <= set(names)
                                for names in store))

    def test_sglang_v41_indexer_query_is_fused_in_sm100_decode(self):
        """sglang v0.5.22 runs V4.1's indexer query rope + weight scale as one
        `index_q_rope_pack_weights` launch in SM100+ decode; prefill in its
        default breakable prefill graph, and anything below SM100, keep them
        apart."""
        program = _program("deepseek-ai__DeepSeek-V4.1-Flash")
        prefill = Deployment(program).at(input_tokens=128)
        decode = prefill.during(DECODE, cache_len=4096)

        def fused(walk, sku):
            out, _ = Framework.of(
                get_binding("sglang").deployed(ServingDeployment.at(
                    gpu_sku=sku, activation_dtype="BF16")),
                FUSION_SCHEMES).rewrite(walk.program, walk)
            return any(n.op == "fused_indexer_q_rope_quant_split"
                       for n in out.nodes())

        decode_walk, prefill_walk = shared_walk(decode), shared_walk(prefill)
        self.assertTrue(fused(decode_walk, "B200"))
        self.assertFalse(fused(prefill_walk, "B200"))
        self.assertFalse(fused(decode_walk, "H100-SXM"))

    def test_the_norms_role_says_which_chain_the_launch_runs(self):
        """DeepSeek-V4.1 writes its latent KV and its indexer key with the
        same norm -> rope -> re-join, and only the two norms differ:
        `attention` / `attention_swa` against `attention_indexer`. sglang's
        latent store asks the role of the norm its kernel normalises (every
        SM, both checkpoint classes, which it serves through one class), so
        it takes the eight latent chains and none of the three indexer ones.
        sglang's own indexer-K launch needs SM100 and one plan id carries one
        condition, so its three chains stay unfused — more launches than
        sglang runs there, never a grouping it does not run. vLLM norm-ropes
        no chain on its own: its indexer-K launch also stores the key
        (`test_vllms_index_key_launch_takes_the_store`)."""
        walk = shared_walk(Deployment(_program("deepseek-ai__DeepSeek-V4.1-Flash")).at(
            input_tokens=128))

        def norms(framework, sku):
            out, _ = Framework.of(
                get_binding(framework).deployed(ServingDeployment.at(
                    gpu_sku=sku, activation_dtype="BF16")),
                FUSION_SCHEMES).rewrite(walk.program, walk)
            return sorted(walk.program.by_uid(u).name
                          for n in out.nodes() if n.op == "fused_norm_rope"
                          for u in n.origins
                          if walk.program.by_uid(u).op == "rmsnorm")

        for sku in ("H100-SXM", "B200"):
            self.assertEqual(norms("sglang", sku), ["kv_norm"] * 8, sku)
            self.assertEqual(norms("vllm", sku), [], sku)

    def _index_key_launches(self, walk, framework, sku="H100-SXM"):
        """The `fused_compress_norm_rope_store` nodes of `walk` under
        `framework`, each as the sorted names of the nodes it absorbed."""
        out, _ = Framework.of(
            get_binding(framework).deployed(ServingDeployment.at(
                gpu_sku=sku, activation_dtype="BF16")),
            FUSION_SCHEMES).rewrite(walk.program, walk)
        return sorted(tuple(sorted(walk.program.by_uid(u).name
                                   for u in n.origins))
                      for n in out.nodes()
                      if n.op == "fused_compress_norm_rope_store")

    def test_vllms_index_key_launch_takes_the_store(self):
        """vLLM's `indexer_k_norm_rope_store` normalises, rotates, quantises
        and stores DeepSeek-V4.1's index key in one launch, so the fused node
        ends at the indexer's `k_cache` store, one component above the norm
        and the rotation — on each of the three key-owning layer kinds (the
        two gated sources and the candidate source). sglang runs the same
        store on SM90+ and takes the same three."""
        walk = shared_walk(Deployment(_program("deepseek-ai__DeepSeek-V4.1-Flash")).at(
            input_tokens=128))
        chain = ("k_cache", "k_nope_slice", "k_norm", "k_pe_rope",
                 "k_pe_slice", "k_roped")
        self.assertEqual(self._index_key_launches(walk, "vllm"), [chain] * 3)
        self.assertEqual(self._index_key_launches(walk, "sglang"), [chain] * 3)

    @pytest.mark.regression          # a second V4.1 walk, at decode
    def test_vllms_index_key_launch_takes_the_store_in_decode(self):
        """No phase branch at the call site (`_produce_k`, both branches of
        the indexer's `forward`): decode is the same launch."""
        walk = shared_walk(Deployment(_program("deepseek-ai__DeepSeek-V4.1-Flash")).at(
            input_tokens=128).during(DECODE, cache_len=4096))
        chain = ("k_cache", "k_nope_slice", "k_norm", "k_pe_rope",
                 "k_pe_slice", "k_roped")
        for sku in ("H100-SXM", "B200"):
            self.assertEqual(self._index_key_launches(walk, "vllm", sku),
                             [chain] * 3, sku)

    def test_sglang_dsa_indexer_prep_follows_the_rope_layout(self):
        """sglang fuses the DSA indexer's query rope + quant and its key
        norm + rope + quant + store only when the indexer's RoPE is
        interleaved (`n_indexer_rope_interleave`, bound from
        `indexer_rope_interleave`): GLM-5 yes in both phases, DeepSeek-V3.2
        (NeoX) no. vLLM's binding does not ask it."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        plans = {"fused_indexer_q_rope_first_quant",
                 "fused_indexer_k_norm_rope_store"}
        for stem, fused in (("zai-org__GLM-5", True),
                            ("deepseek-ai__DeepSeek-V3.2", False)):
            d = Deployment(_program(stem)).at(input_tokens=128)
            for walk in (shared_walk(d),
                         shared_walk(d.during(DECODE, cache_len=4096))):
                def ops(framework):
                    out, _ = Framework.of(
                        get_binding(framework).deployed(deployment),
                        FUSION_SCHEMES).rewrite(walk.program, walk)
                    return {n.op: sorted(walk.program.by_uid(u).name
                                         for u in n.origins)
                            for n in out.nodes() if n.op in plans}
                sglang = ops("sglang")
                self.assertEqual(set(sglang), plans if fused else set(), stem)
                if fused:
                    self.assertIn("indexer.q_pe_rope",
                                  sglang["fused_indexer_q_rope_first_quant"])
                    self.assertIn("indexer.k_norm",
                                  sglang["fused_indexer_k_norm_rope_store"])
                self.assertEqual(ops("vllm"), {})

    def test_readers_eat_is_the_format_traffic_prices(self):
        """`readers_eat` reads each reader's format from
        `traffic.reader_format` — the representations `traffic.derive`
        prices that reader's input conversion from — so the fusion condition
        and the conversion bill answer one question once: V4.1's q readers
        eat group-32 MXFP8 as shipped, and the per-128 FP8 activation once
        the weights are block-128 FP8; a reader that eats a format pays a
        conversion on its input unfused."""
        from backend.arch.activation_formats import ACTIVATION_FORMATS
        from backend.arch.precision_profile import PrecisionProfile
        program = _program("deepseek-ai__DeepSeek-V4.1-Flash")
        shipped = Deployment(program).at(input_tokens=128)
        fp8 = shipped.with_precision(PrecisionProfile.from_request(
            {"preset": "w_fp8_kv_fp8"}))
        for walk, name in ((shared_walk(shipped), "fp8_token_group32_ue8m0"),
                           (shared_walk(fp8), None)):
            readers = [n for n in walk.program.walked_nodes()
                       if n.name == "wq_b" and not n.children]
            self.assertTrue(readers)
            formats = {traffic.reader_format(walk, n) for n in readers}
            self.assertEqual(len(formats), 1)
            fmt = formats.pop()
            if name is not None:
                self.assertEqual(fmt, ACTIVATION_FORMATS[name])
            else:
                self.assertNotEqual(
                    fmt, ACTIVATION_FORMATS["fp8_token_group32_ue8m0"])
            billed = traffic.derive(walk).leaves().values
            self.assertTrue(all(int(billed[n.uid].convert_bytes) > 0
                                for n in readers))

    def test_a_dims_condition_does_not_ask_a_grafted_cast(self):
        """DeepInfer binds Mega-MHC under `dims: {n_hidden_copies: [4]}`
        and its plan takes the activation cast as an optional member. A
        cast is grafted after the walk and ran at no spec dim, so the
        condition reads the members that did — it neither raises on the
        cast nor refuses the match."""
        program = _program("deepseek-ai__DeepSeek-V4.1-Flash")
        analysis = shared_walk(Deployment(program).at(input_tokens=128)
                               .during(DECODE, cache_len=4096))
        deployment = ServingDeployment.at(gpu_sku="B200",
                                          activation_dtype="BF16")
        from backend.inference.serving import rewrite_serving_graph
        # The serving graph: only it has the casts grafted.
        fused = rewrite_serving_graph(
            analysis, framework_binding=get_binding("deepinfer").deployed(
                deployment)).program
        self.assertTrue([n for n in fused.nodes()
                         if n.op == "mhc_mega_post_pre"])

    def test_the_kpool_compressor_is_one_launch_in_both_phases(self):
        """GLM-5.3-Flash's k-pool indexer compressor — position bias, softmax
        pool, weighted sum, quantised cache write — is one launch in vLLM
        and sglang, prefill and decode, the index-K format's Hadamard
        rotation included; the pool windows and the k norm stay outside.
        Counted on the serving graph: only it has the rotation node the
        cache format grafts (`materialize_casts`)."""
        from backend.inference.serving import rewrite_serving_graph
        program = _program("zai-org__GLM-5.3-Flash")
        prefill = Deployment(program).at(input_tokens=128)
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        for analysis in (shared_walk(prefill),
                         shared_walk(prefill.during(DECODE, cache_len=4096))):
            for framework in ("vllm", "sglang"):
                serving = rewrite_serving_graph(
                    analysis, framework_binding=get_binding(
                        framework).deployed(deployment))
                pools = [n for n in serving.program.nodes()
                         if n.op == "fused_kpool_compress_store"]
                self.assertTrue(pools, framework)
                for node in pools:
                    self.assertEqual(
                        sorted(serving.pre_program.by_uid(u).op
                               for u in node.origins),
                        sorted(["position_table_add", "softmax_act", "mul",
                                "sum_reduce", "hadamard_transform",
                                "token_pool_compressor"]))

    def test_the_dilated_short_conv_fuses_where_its_framework_fuses_it(self):
        """Qwen3.8-Flash-Next has two short convs: the gated-delta-net mixer's
        (undilated) and the PLE n-gram conv (dilated by the n-gram size).
        vLLM folds the SiLU into both; sglang's causal-conv kernel takes no
        dilation, so its PLE conv and SiLU stay two launches."""
        analysis = _walked("Qwen__Qwen3.8-Flash-Next")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def fused_convs(framework):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return sorted(analysis.node_dims[analysis.program.by_uid(uid)]
                          ["dilation"]
                          for node in fused.nodes()
                          if node.op == "causal_conv1d"
                          for uid in node.origins
                          if analysis.program.by_uid(uid).op == "conv1d")

        vllm, sglang = fused_convs("vllm"), fused_convs("sglang")
        self.assertEqual(set(sglang), {1})
        self.assertTrue(any(d != 1 for d in vllm))
        self.assertEqual([d for d in vllm if d == 1], sglang)

    def test_a_fused_recurrent_state_carrier_keeps_its_read(self):
        """The overlap compressor's kv/score state is carried by `weighted` and
        `smax`, which vLLM's compress kernel absorbs. The state's per-step
        read moves onto that launch: the step reads the same cache bytes
        fused as unfused, and the absorbed carriers bill nothing."""
        analysis = (shared_walk(Deployment(_program("deepseek-ai__DeepSeek-V4-Flash"))
                    .at(batch=2, input_tokens=4096)
                    .during(DECODE, cache_len=4096)))
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        fused, rewrite = _only(Framework.of(
            get_binding("vllm").deployed(deployment),
            FUSION_SCHEMES), "fused_overlap_compress_norm_rope_store").rewrite(
                analysis.program, analysis)
        self.assertIn("fused_overlap_compress_norm_rope_store",
                      {n.op for n in fused.nodes()})

        def cache_reads(walk):
            return sum(int(t.kv_reads) for t in
                       traffic.derive(walk).leaves().values.values())

        self.assertEqual(
            cache_reads(rebase_analysis(analysis, fused, rewrite)),
            cache_reads(analysis))

    def test_the_gated_delta_net_launches_follow_the_phase(self):
        """Qwen3.8-27B's gated delta net: on a decode step both frameworks
        run l2norm, gates and the recurrence as one launch; on a prompt
        sglang runs only the gate chain as one launch (`fused_gdn_gating`),
        and vLLM runs the split, both l2norms and the gates as one
        (`fused_gdn_post_conv_prep`)."""
        base = Deployment(_program("Qwen__Qwen3.8-27B")).at(input_tokens=128)
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        new = {"fused_recurrent_gated_delta_rule", "fused_gdn_gating",
               "fused_gdn_post_conv_prep"}

        def fired(framework, analysis):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return {n.op for n in fused.nodes()} & new

        prefill = shared_walk(base)
        decode = shared_walk(base.during(DECODE, cache_len=4096))
        self.assertEqual(fired("vllm", prefill), {"fused_gdn_post_conv_prep"})
        self.assertEqual(fired("sglang", prefill), {"fused_gdn_gating"})
        self.assertEqual(fired("vllm", decode),
                         {"fused_recurrent_gated_delta_rule"})
        self.assertEqual(fired("sglang", decode),
                         {"fused_recurrent_gated_delta_rule"})

    def test_sglang_gdn_gates_are_written_fp32_on_their_ports(self):
        """sglang's gate launch writes g and beta as float32, each on a named
        port (`output_dtype: {g_out, beta_out}`): traffic bills both writes
        at 4 bytes and the recurrence reads both edges at 4 bytes. One fused
        node stands for the template's 48 layers; nothing in another stack."""
        from backend.arch.eval.layer import Layer

        walk = shared_walk(Deployment(_program("Qwen__Qwen3.8-27B")).at(
            batch=1, input_tokens=4096))
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        fused, rewrite = Framework.of(
            get_binding("sglang").deployed(deployment),
            FUSION_SCHEMES).rewrite(walk.program, walk)
        rebased = rebase_analysis(walk, fused, rewrite)
        leaves = traffic.derive(rebased).leaves().values
        gates = [n for n in fused.nodes() if n.op == "fused_gdn_gating"]
        self.assertEqual(len(gates), 1)
        (gate,) = gates
        # `out_terms` carries `_PortWrite` records, not tuples: each says
        # which port it landed on, how many elements, at what width, and
        # whether that is still the whole of what the port publishes.
        terms = {term.port: term.width for term in leaves[gate.uid].out_terms
                 if term.port}
        self.assertEqual(terms, {"g_out": 4.0, "beta_out": 4.0})
        published = set(gate.output_ports)
        (rule,) = [n for n in fused.nodes() if n.op == "gated_delta_rule"]
        read = {e.producer: e.width for e in leaves[rule.uid].in_edges
                if e.producer in published}
        self.assertEqual(set(read.values()), {4.0})
        self.assertEqual(len(read), 2)
        counts = walk.recount(walk.instances, walk.layer_multiplicity())
        rebased_counts = Layer(walk.program, dict(counts.values)).rebase(
            rewrite, fused, lambda parts: parts[0])
        self.assertEqual(rebased_counts.values[gate.uid], 48)
        self.assertEqual(walk.program.stack_of(
            walk.program.by_uid(gate.origins[0])),
            walk.program.stack_of(rule))

    def test_vllm_gdn_prefill_prep_writes_contiguous_q_k_v_and_fp32_gates(self):
        """vLLM's post-conv prep reads the packed conv output once and writes
        q and k normalised, v copied to its own contiguous buffer (the
        kernel allocates all five outputs) and g / beta in fp32: traffic
        bills exactly those five writes, and the recurrence reads q, k, v
        and both gates from the launch's ports."""
        walk = shared_walk(Deployment(_program("Qwen__Qwen3.8-27B")).at(
            batch=1, input_tokens=4096))
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        fused, rewrite = Framework.of(
            get_binding("vllm").deployed(deployment),
            FUSION_SCHEMES).rewrite(walk.program, walk)
        leaves = traffic.derive(rebase_analysis(walk, fused, rewrite)
                                ).leaves().values
        (prep,) = [n for n in fused.nodes()
                   if n.op == "fused_gdn_post_conv_prep"]
        widths = {t.port: t.width for t in leaves[prep.uid].out_terms}
        self.assertEqual(widths, {"": 2.0, "k_out": 2.0, "v_out": 2.0,
                                  "g_out": 4.0, "beta_out": 4.0})
        # One activation read: the conv output, whole (q|k|v), plus a and b.
        (conv,) = [n for n in fused.nodes() if n.op == "causal_conv1d"]
        self.assertEqual(
            sum(1 for e in leaves[prep.uid].in_edges
                if e.producer in conv.origins), 1)
        (rule,) = [n for n in fused.nodes() if n.op == "gated_delta_rule"]
        # The recurrence reads all five operands off the launch's ports. An
        # edge names the producer the walk declared — for v the `head_split`
        # wrapper — so each is resolved to the leaf that writes it.
        def writer(uid):
            node = walk.program.by_uid(uid)
            return (walk.program.output_leaf(node, "out")[0].uid
                    if node.children else uid)
        reads = {writer(e.producer): e.width
                 for e in leaves[rule.uid].in_edges}
        self.assertEqual(set(reads), set(prep.output_ports))
        self.assertEqual(sorted(reads.values()), [2.0, 2.0, 2.0, 4.0, 4.0])

    def test_sglang_gdn_gates_do_not_move_decode(self):
        """The gate plan is prefill-only: a decode step's bytes and cache
        reads are those of the binding without it."""
        from dataclasses import replace

        walk = (shared_walk(Deployment(_program("Qwen__Qwen3.8-27B"))
                .at(batch=2, input_tokens=4096)
                .during(DECODE, cache_len=4096)))
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        binding = get_binding("sglang").deployed(deployment)
        without = replace(binding, op_implementations=tuple(
            (op, impl) for op, impl in binding.op_implementations
            if op != "fused_gdn_gating"))

        def totals(b):
            fused, rewrite = Framework.of(b, FUSION_SCHEMES).rewrite(
                walk.program, walk)
            leaves = traffic.derive(rebase_analysis(walk, fused, rewrite)
                                    ).leaves().values.values()
            return (sum(int(t.act_reads + t.act_writes) for t in leaves),
                    sum(int(t.kv_reads) for t in leaves))

        self.assertEqual(totals(binding), totals(without))

    def test_an_absorbed_recurrence_keeps_its_state_read(self):
        """The gated delta rule carries the recurrent state; the decode
        launch that absorbs it reads the same cache bytes per step as the
        unfused graph did, and the absorbed carrier bills none of its own."""
        analysis = (shared_walk(Deployment(_program("Qwen__Qwen3.8-27B"))
                    .at(batch=2, input_tokens=4096)
                    .during(DECODE, cache_len=4096)))
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        fused, rewrite = Framework.of(
            get_binding("vllm").deployed(deployment),
            FUSION_SCHEMES).rewrite(analysis.program, analysis)
        self.assertIn("fused_recurrent_gated_delta_rule",
                      {n.op for n in fused.nodes()})

        def cache_reads(walk):
            return sum(int(t.kv_reads) for t in
                       traffic.derive(walk).leaves().values.values())

        rebased = rebase_analysis(analysis, fused, rewrite)
        self.assertGreater(cache_reads(analysis), 0)
        self.assertEqual(cache_reads(rebased), cache_reads(analysis))
        recurrent = [uid for uid, t in
                     traffic.derive(rebased).leaves().values.items()
                     if rebased.program.by_uid(uid).op
                     == "fused_recurrent_gated_delta_rule"]
        self.assertTrue(recurrent)
        self.assertTrue(all(
            int(traffic.derive(rebased).leaves().values[uid].kv_reads) > 0
            for uid in recurrent))

    def test_the_grouped_mamba_norm_fuses_only_in_sglang(self):
        """Nemotron-3's Mamba-2 norm gates before it normalises, per group
        (8 groups). sglang runs it as one FLA launch; vLLM runs that kernel
        only for one group and goes native here. Neither framework's
        `rmsnorm_gated` (norm first) matches it."""
        analysis = _walked("nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def fired(framework):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return {n.op for n in fused.nodes()} & {
                "mamba2_rmsnorm_gated", "rmsnorm_gated"}

        self.assertEqual(fired("sglang"), {"mamba2_rmsnorm_gated"})
        self.assertEqual(fired("vllm"), set())

    def test_the_grouped_mamba_norm_needs_whole_groups_per_rank(self):
        """sglang fuses the grouped norm only while its TP group divides
        n_groups (8 here); the binding asks the walk's `tp_attention`
        degree. A degree that leaves a rank part of a group admits no match.
        (The walk itself refuses tp_attention=16 for this model at the
        group split, so the non-dividing degree is asked of the rule.)"""
        from backend.arch.eval.fusion import GraphCondition
        from backend.arch.layout.parallel_config import ParallelConfig

        analysis = (shared_walk(Deployment(
            _program("nvidia__NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
            config=ParallelConfig(tp_attention=8)).at(input_tokens=16)))
        self.assertEqual(analysis.rule.world_degree("tp_attention"), 8)
        condition = {op: c for op, c, _x in get_binding("sglang").dispatched()}[
            "mamba2_rmsnorm_gated"]
        self.assertEqual(condition.world_local,
                         ("tp_attention", "n_ssm_groups", ()))
        fusion = Fusion.named("mamba2_rmsnorm_gated", FUSION_SCHEMES,
                              condition)
        fused, _ = fusion.rewrite(analysis.program, analysis)
        self.assertTrue(any(n.op == "mamba2_rmsnorm_gated"
                            for n in fused.nodes()))

        class Degree:
            def __init__(self, degree):
                self.degree = degree

            def world_degree(self, world):
                return self.degree

        by_name = {n.name: n for n in analysis.program.walked_nodes()}
        bound = {"groups": by_name["groups"], "norm": by_name["group_rms"]}
        for degree, admitted in ((1, True), (8, True), (3, False),
                                 (16, False)):
            walk = type("Walk", (), {"node_dims": analysis.node_dims,
                                     "rule": Degree(degree)})()
            self.assertEqual(condition.admits_match(bound, walk), admitted,
                             degree)
        # A match none of whose nodes binds the dim answers no.
        walk = type("Walk", (), {"node_dims": analysis.node_dims,
                                 "rule": Degree(1)})()
        self.assertFalse(GraphCondition(
            world_local=("tp_attention", "n_lin_v_heads", ())
        ).admits_match(bound, walk))
    def test_an_ungrouped_router_is_one_launch(self):
        """The ungrouped top-K routers end where each kernel ends: Hy3's
        sigmoid router with its routed scale, MiniMax-M2.7's without one
        (the optional last step absent), Qwen3's softmax router in both
        frameworks, LongCat's biased softmax router in vLLM only and with its
        routed scale left a launch of its own, and DeepSeek-V4's sqrt-softplus
        and hash routers. Hy4 keeps its degenerate group nodes and the
        grouped kernel."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def routers(name, framework):
            analysis = _walked(name)
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return {(node.op, tuple(sorted(analysis.program.by_uid(u).op
                                           for u in node.origins)))
                    for node in fused.nodes()
                    if isinstance(node, FusedNode)
                    and node.op.startswith("moe_")}

        sigmoid = ("add", "gather", "normalize_sum_one", "sigmoid", "topk")
        softmax = ("gather", "normalize_sum_one", "softmax_act", "topk")
        for framework in ("vllm", "sglang"):
            self.assertEqual(routers("tencent__Hy3", framework), {
                ("moe_topk_sigmoid",
                 tuple(sorted((*sigmoid, "scale_const"))))})
            self.assertEqual(routers("MiniMaxAI__MiniMax-M2.7", framework),
                             {("moe_topk_sigmoid", sigmoid)})
            self.assertEqual(routers("Qwen__Qwen3-30B-A3B", framework),
                             {("moe_topk_softmax", softmax)})
            self.assertEqual(
                {op for op, _ in routers("deepseek-ai__DeepSeek-V4-Flash",
                                         framework)},
                {"moe_topk_sqrtsoftplus", "moe_topk_hash_sqrtsoftplus"})
            self.assertEqual({op for op, _ in routers("tencent__Hy4-preview",
                                                      framework)},
                             {"moe_fused_gate"})
        self.assertEqual(
            routers("meituan-longcat__LongCat-Flash-Omni", "vllm"),
            {("moe_topk_softmax_bias",
              ("add", "gather", "softmax_act", "topk"))})
        self.assertEqual(
            routers("meituan-longcat__LongCat-Flash-Omni", "sglang"), set())
        self.assertEqual(routers("deepseek-ai__DeepSeek-V2-Lite", "sglang"),
                         set())

    def test_a_learnable_sink_rides_into_the_sparse_kernel(self):
        """Hy4's sparse attention softmaxes against a per-head learnable
        sink. vLLM hands the sink to its FlashMLA sparse kernels in both
        phases, so score, softmax and apply are one `flash_dsa_sink` launch.
        sglang (v0.5.22) does the same: Hy4's attention forces the sparse
        flashmla path with the sink in both phases, never dense MHA."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        attention = {"dsa_attention_score", "dsa_softmax",
                     "dsa_attention_apply"}
        for phase in (None, DECODE):
            d = Deployment(_program("tencent__Hy4-preview")).at(
                input_tokens=128)
            analysis = shared_walk(d.during(DECODE, cache_len=4096) if phase
                        else d)

            def ops(framework):
                fused, _ = Framework.of(
                    get_binding(framework).deployed(deployment),
                    FUSION_SCHEMES).rewrite(analysis.program, analysis)
                return {n.op for n in fused.nodes()}

            for framework in ("vllm", "sglang"):
                fused = ops(framework)
                self.assertIn("flash_dsa_sink", fused, (framework, phase))
                self.assertFalse(attention & fused, (framework, phase))

    def test_gemma4_runs_sglangs_own_kernels(self):
        """sglang's gemma-4 path norms q, k and v in one launch, folds the
        post-FFN norm, the residual add and the layer scalar into another,
        and soft-caps the logits in place; vLLM runs each step on its own.
        Gemma-2 shares only the soft-cap."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        mine = {"gemma_qkv_rmsnorm", "gemma_rmsnorm_residual_scalar",
                "softcap_logits"}

        def fused(name, framework):
            analysis = _walked(name)
            out, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return {n.op: sorted(analysis.program.by_uid(u).op
                                 for u in n.origins)
                    for n in out.nodes()
                    if isinstance(n, FusedNode) and n.op in mine}

        gemma4 = fused("google__gemma-4-12B-it", "sglang")
        self.assertEqual(gemma4["gemma_qkv_rmsnorm"],
                         ["rmsnorm", "rmsnorm", "rmsnorm_unscaled"])
        self.assertEqual(gemma4["gemma_rmsnorm_residual_scalar"],
                         ["add", "mul", "rmsnorm"])
        self.assertEqual(gemma4["softcap_logits"],
                         ["scale_const", "scale_const", "tanh"])
        self.assertEqual(fused("google__gemma-4-12B-it", "vllm"), {})
        self.assertEqual(set(fused("google__gemma-2-9b", "sglang")),
                         {"softcap_logits"})

    def test_gemma4_moe_tail_is_the_dual_kernel(self):
        """sglang picks its gemma-4 tail kernel by whether the layer has the
        MoE block; the graph says the same thing — a MoE layer's post-FFN
        norm reads the sum of two normed branches. The dense tail takes the
        3-node kernel, the MoE tail the 6-node one, never the dense one."""
        self.assertEqual(_gemma_tail_ops(moe=False), {
            "gemma_rmsnorm_residual_scalar": ["out", "post_ff", "residual"]})
        self.assertEqual(_gemma_tail_ops(moe=True), {
            "gemma_dual_rmsnorm_residual_scalar": [
                "branches", "norm_1", "norm_2", "out", "post_ff",
                "residual"]})

    def test_the_kda_decode_step_follows_its_framework(self):
        """KDA decode: GLM-5.3-Flash runs gate + beta + l2norm + recurrence
        as one launch in both frameworks. Kimi-K3 is not claimed by either:
        vLLM runs its whole mixer as one kernel, and sglang does at
        attention TP 8. In PREFILL vLLM runs the same span again — FlashKDA
        takes the gate, beta and both l2 norms as kernel arguments — while
        sglang's chunk path launches the gate apart, so only vLLM claims
        GLM's prefill. The recurrent state read stays on the launch."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def rewrite(framework, analysis):
            return Framework.of(get_binding(framework).deployed(deployment),
                                FUSION_SCHEMES).rewrite(analysis.program,
                                                        analysis)

        def fired(framework, analysis):
            fused, _ = rewrite(framework, analysis)
            return any(n.op == "fused_recurrent_kda" for n in fused.nodes())

        for model, claimed in (("moonshotai__Kimi-K3", False),
                               ("zai-org__GLM-5.3-Flash", True)):
            # Kimi-K3's per-channel gate is other plans
            # (test_kimi_k3_whole_mixer_decode_follows_its_dispatch).
            base = Deployment(_program(model)).at(batch=2, input_tokens=128)
            prefill = shared_walk(base)
            decode = shared_walk(base.during(DECODE, cache_len=4096))
            for framework in ("vllm", "sglang"):
                # vLLM's FlashKDA prefill is this same span, for the class
                # that has it; sglang's prefill is the chunk path.
                self.assertEqual(fired(framework, prefill),
                                 claimed and framework == "vllm",
                                 (model, framework, "prefill"))
                self.assertEqual(fired(framework, decode), claimed,
                                 (model, framework))

        def cache_reads(walk):
            return sum(int(t.kv_reads) for t in
                       traffic.derive(walk).leaves().values.values())

        glm = (shared_walk(Deployment(_program("zai-org__GLM-5.3-Flash"))
               .at(batch=2, input_tokens=128)
               .during(DECODE, cache_len=4096)))
        fused, rw = _only(Framework.of(
            get_binding("sglang").deployed(ServingDeployment.at(
                gpu_sku="H100-SXM", activation_dtype="BF16")),
            FUSION_SCHEMES), "fused_recurrent_kda").rewrite(glm.program, glm)
        self.assertIn("fused_recurrent_kda", {n.op for n in fused.nodes()})
        self.assertEqual(cache_reads(rebase_analysis(glm, fused, rw)),
                         cache_reads(glm))

    def test_the_kda_plan_is_one_exact_gate_form(self):
        """The plan spells GLM's per-head-rate bounded gate and nothing else:
        unconditioned, it binds all thirteen of GLM's nodes and does not match
        Kimi-K3's per-channel-rate gate (a different formula's spelling)."""
        plan = Fusion.named("fused_recurrent_kda", FUSION_SCHEMES)

        def member_sets(model):
            walk = (shared_walk(Deployment(_program(model)).at(batch=1, input_tokens=16)
                    .during(DECODE, cache_len=64)))
            fused, _ = plan.rewrite(walk.program, walk)
            return [sorted(walk.program.by_uid(u).name for u in n.origins)
                    for n in fused.nodes() if n.op == "fused_recurrent_kda"]

        glm = member_sets("zai-org__GLM-5.3-Flash")
        self.assertTrue(glm)
        self.assertTrue(all(m == sorted([
            "q_l2norm", "k_l2norm", "f_plus_dt", "f_unflatten", "f_heads",
            "f_heads_t", "exp_A", "g_rate", "g_sigmoid", "g_decay_t",
            "g_decay", "beta", "delta_rule"]) for m in glm), glm)
        self.assertEqual(member_sets("moonshotai__Kimi-K3"), [])

    def test_sglang_kda_prefill_gates_follow_the_framework(self):
        """KDA prefill: sglang forms the bounded gate and beta as one launch
        (`fused_kda_gate`) for both KDA classes — GLM-5.3-Flash's head-last
        spelling and Kimi-K3's head-first one, the plan's two `optional`
        transposes. vLLM's prefill is FlashKDA, which takes the gate inside
        the recurrence launch, so this gate-only plan binds nothing there
        (Kimi-K3's prefill is the per-channel span instead); decode is the
        decode plans' in both frameworks."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def fired(framework, analysis):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return any(n.op == "fused_kda_gate" for n in fused.nodes())

        for model in ("moonshotai__Kimi-K3", "zai-org__GLM-5.3-Flash"):
            base = Deployment(_program(model)).at(batch=2, input_tokens=128)
            prefill = shared_walk(base)
            decode = shared_walk(base.during(DECODE, cache_len=4096))
            self.assertTrue(fired("sglang", prefill), model)
            self.assertFalse(fired("vllm", prefill), model)
            for framework in ("vllm", "sglang"):
                self.assertFalse(fired(framework, decode), (model, framework))

    def test_sglang_kda_prefill_gate_matches_both_spellings(self):
        """Unconditioned, the plan binds GLM's ten nodes (both transposes
        present) and Kimi-K3's eight (neither), and nothing else — the same
        gate formula in two frames, not two formulas."""
        plan = Fusion.named("fused_kda_gate", FUSION_SCHEMES)

        def member_sets(model):
            walk = shared_walk(Deployment(_program(model)).at(
                batch=1, input_tokens=16))
            fused, _ = plan.rewrite(walk.program, walk)
            return [sorted(walk.program.by_uid(u).name for u in n.origins)
                    for n in fused.nodes() if n.op == "fused_kda_gate"]

        glm = member_sets("zai-org__GLM-5.3-Flash")
        self.assertTrue(glm)
        self.assertTrue(all(m == sorted([
            "f_plus_dt", "f_unflatten", "f_heads", "f_heads_t", "exp_A",
            "g_rate", "g_sigmoid", "g_decay", "g_decay_t", "beta"])
            for m in glm), glm)
        kimi = member_sets("moonshotai__Kimi-K3")
        self.assertTrue(kimi)
        self.assertTrue(all(m == sorted([
            "f_plus_dt", "f_unflatten", "f_heads", "exp_A", "g_rate",
            "g_sigmoid", "g_decay", "beta"]) for m in kimi), kimi)

    def test_sglang_kda_prefill_gates_are_written_fp32_on_their_ports(self):
        """The launch writes the (chunk-cumsummed) gate and beta as float32,
        each on a named port (`output_dtype: {output, beta_out}`): traffic
        bills both writes at 4 bytes and the recurrence reads both at 4."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        for model in ("moonshotai__Kimi-K3", "zai-org__GLM-5.3-Flash"):
            walk = shared_walk(Deployment(_program(model)).at(
                batch=1, input_tokens=4096))
            fused, rewrite = Framework.of(
                get_binding("sglang").deployed(deployment),
                FUSION_SCHEMES).rewrite(walk.program, walk)
            leaves = traffic.derive(rebase_analysis(walk, fused, rewrite)
                                    ).leaves().values
            gates = [n for n in fused.nodes() if n.op == "fused_kda_gate"]
            self.assertTrue(gates, model)
            for gate in gates:
                terms = {t.port: t.width
                         for t in leaves[gate.uid].out_terms}
                self.assertEqual(terms, {"": 4.0, "beta_out": 4.0}, model)
                published = set(gate.output_ports)
                read = [e.width for t in leaves.values()
                        for e in t.in_edges if e.producer in published]
                self.assertEqual(sorted(read), [4.0, 4.0], model)


    def test_kernels_that_need_cuda_and_half_precision_ask_for_them(self):
        """vLLM's Hy4 iHC launches (`triton_ihc_supported`, triton_ihc.py:
        19-27) run only on CUDA with fp16/bf16 activations -- the post and
        both launches of the pre, which the one wrapper gates together; its
        Qwen4Exp PLE gate needs fp16/bf16 on any platform (v0.32.0 runs the
        same kernel on ROCm). An FP32 deployment bills the unfused ops, and
        so does an undescribed GPU where the platform matters."""
        cases = (("tencent__Hy4-preview", "ihc_pre_mix", False),
                 ("tencent__Hy4-preview", "ihc_pre_gate_reduce", False),
                 ("tencent__Hy4-preview", "ihc_post", False),
                 ("Qwen__Qwen3.8-Flash-Next", "ple_gate", True))
        for model, op, any_platform in cases:
            walk = _walked(model)
            for sku, dtype, expected in (("H100-SXM", "BF16", True),
                                         ("H100-SXM", "FP32", False),
                                         (None, "BF16", any_platform)):
                deployment = ServingDeployment.at(gpu_sku=sku,
                                                  activation_dtype=dtype)
                fused, _ = Framework.of(
                    get_binding("vllm").deployed(deployment),
                    FUSION_SCHEMES).rewrite(walk.program, walk)
                self.assertEqual(any(n.op == op for n in fused.nodes()),
                                 expected, (model, sku, dtype))

    def test_only_vllms_ihc_entries_ask_the_activation_dtype(self):
        """The two bindings differ on purpose, so neither may be tidied into
        the other. vLLM gates every iHC launch behind `triton_ihc_supported`,
        which checks the activation dtype (triton_ihc.py:19-27), so its
        entries say `activation_dtype: [bf16, fp16]` and an FP32 deployment
        bills the pre unfused. sglang's wrapper asserts only the shape
        (`fused_hy4_ihc_pre`, hy4_ihc.py:204-215 -- the one dtype assert in
        the file is the post GATE's fp32, :309), so its entries state no
        dtype and it fuses at FP32 too."""
        walk = _walked("tencent__Hy4-preview")
        for framework, expected in (("vllm", False), ("sglang", True)):
            fused, _ = Framework.of(
                get_binding(framework).deployed(ServingDeployment.at(
                    gpu_sku="H100-SXM", activation_dtype="FP32")),
                FUSION_SCHEMES).rewrite(walk.program, walk)
            ops = {n.op for n in fused.nodes()}
            for launch in ("ihc_pre_mix", "ihc_pre_gate_reduce"):
                self.assertEqual(launch in ops, expected,
                                 (framework, launch, "FP32"))
            # Both fuse at BF16: the asymmetry is the dtype key, not the bind.
            fused_bf16, _ = Framework.of(
                get_binding(framework).deployed(ServingDeployment.at(
                    gpu_sku="H100-SXM", activation_dtype="BF16")),
                FUSION_SCHEMES).rewrite(walk.program, walk)
            ops_bf16 = {n.op for n in fused_bf16.nodes()}
            for launch in ("ihc_pre_mix", "ihc_pre_gate_reduce"):
                self.assertIn(launch, ops_bf16, (framework, launch, "BF16"))

    def test_hy4_ihc_post_and_attention_gate(self):
        """Hy4: both frameworks run each iHC post (gate the sub-block output
        onto the residual copies, add the residual) as one Triton launch;
        sglang also runs the attention output gate x * sigmoid(g) as one,
        while vLLM runs it as two eager ops."""
        analysis = _walked("tencent__Hy4-preview")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")

        def fired(framework):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return sorted(analysis.program.by_uid(n.sink).name
                          for n in fused.nodes()
                          if n.op in ("ihc_post", "sigmoid_gate_mul"))

        vllm, sglang = fired("vllm"), fired("sglang")
        self.assertEqual(set(vllm), {"attn_hc_out", "out"})
        self.assertEqual(set(sglang) - set(vllm), {"o_gated"})
        self.assertEqual([n for n in sglang if n != "o_gated"], vllm)

    def test_flash_next_hc_and_ple_kernels_are_vllms(self):
        """Qwen3.8-Flash-Next on vLLM, prefill: the hyper-connection mix
        activation and read run as one launch each, the in-layer write runs
        with the MLP read's norm, and the PLE gate is one launch. sglang
        binds none of them in prefill — its only Qwen4Exp hc launch at a
        vLLM boundary is the decode write (the next test)."""
        analysis = _walked("Qwen__Qwen3.8-Flash-Next")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        new = {"hc_silu", "hc_gate_mix", "hc_combine_norm", "ple_gate"}

        def fired(framework):
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return {n.op for n in fused.nodes()} & new

        self.assertEqual(fired("vllm"), new)
        self.assertEqual(fired("sglang"), set())

    def test_flash_next_hc_write_is_sglangs_on_sm100_single_token_decode(
            self):
        """Decode: vLLM runs all three gates fused; sglang fuses only the
        write (`hc_combine_apply_norm`), and only on a one-token SM100
        decode — B200 fires `hc_combine_norm`, H100 fires nothing, and
        neither fires the read gates, which sglang folds into its mix
        GEMMs (the audit in providers/hc.py)."""
        base = Deployment(_program("Qwen__Qwen3.8-Flash-Next")).at(
            input_tokens=128)
        walk = shared_walk(base.during(DECODE, cache_len=4096))
        new = {"hc_silu", "hc_gate_mix", "hc_combine_norm"}

        def fired(framework, sku):
            deployment = ServingDeployment.at(gpu_sku=sku,
                                              activation_dtype="BF16")
            fused, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(walk.program, walk)
            return {n.op for n in fused.nodes()} & new

        self.assertEqual(fired("vllm", "H100-SXM"), new)
        self.assertEqual(fired("sglang", "B200"), {"hc_combine_norm"})
        self.assertEqual(fired("sglang", "H100-SXM"), set())

    def test_a_kernel_over_model_level_ops_keeps_the_spine_whole(self):
        """Qwen3.8-Flash-Next's final hyper-connection mixer is written on the
        model spine; vLLM's `hc_silu` absorbs two of its ops. The fused tree
        stages that kernel once, where the last of them stood, and the
        absorbed op is no missing stage - the per-op table and the estimate
        walk the post-fusion program's stacks."""
        from backend.arch.eval.fusion import materialize_tree

        analysis = _walked("Qwen__Qwen3.8-Flash-Next")
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        fused, rewrite = Framework.of(
            get_binding("vllm").deployed(deployment),
            FUSION_SCHEMES).rewrite(analysis.program, analysis)
        tree = materialize_tree(analysis.program, fused, rewrite)
        self.assertTrue(tree.fused_layers)
        post = Program(resolved=analysis.program.resolved, ir=tree,
                       schedule=analysis.program.schedule)
        staged = [stage.node for stack in post.stacks
                  for stage in stack.stages if hasattr(stage, "node")]
        uids = [node.uid for node in staged]
        self.assertEqual(len(uids), len(set(uids)))
        spine_kernels = [node.op for node in staged
                         if node.provenance == "fusion"]
        self.assertIn("hc_silu", spine_kernels)
        self.assertIn("hc_gate_mix", spine_kernels)

    def test_sglang_v41_indexer_query_is_fused_in_sm100_decode_only(self):
        """sglang fuses DeepSeek-V4.1's indexer query rope + pack + head
        weights only in DeepGEMM decode, which it takes on SM100+; prefill
        and Hopper decode run it apart."""
        base = Deployment(_program("deepseek-ai__DeepSeek-V4.1-Flash")).at(
            input_tokens=128)
        walks = {"prefill": shared_walk(base),
                 "decode": shared_walk(base.during(DECODE, cache_len=4096))}
        for sku, phase, expected in (("B200", "decode", True),
                                     ("B200", "prefill", False),
                                     ("H100-SXM", "decode", False)):
            walk = walks[phase]
            deployment = ServingDeployment.at(gpu_sku=sku,
                                              activation_dtype="BF16")
            fused, _ = Framework.of(
                get_binding("sglang").deployed(deployment),
                FUSION_SCHEMES).rewrite(walk.program, walk)
            self.assertEqual(
                any(n.op == "fused_indexer_q_rope_quant_split"
                    for n in fused.nodes()), expected, (sku, phase))
    def test_qwen_qsa_indexer_prep_and_gated_attention(self):
        """Qwen3.8-Flash-Next's QSA layers, both phases. The indexer's query
        norm + rope and its key block pool + norm + rope + store are a launch
        each in both frameworks (vLLM runs them as one, which the spec's two
        scopes cannot say); the main attention's q/k stay with the wider
        qk-norm/rope/gate plan. vLLM's sparse attention gates its own output
        when it runs unsplit (a long prefill) and in its split-K merge kernel
        otherwise (decode): one launch in prefill, attention + gate in decode.
        sglang gates after the attention in both phases, and its key launch
        is prefill-only (see the key-pool test). The key pool carries the
        ring's state: a decode step reads the same cache bytes fused or
        not."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        for phase in (None, DECODE):
            d = Deployment(_program("Qwen__Qwen3.8-Flash-Next")).at(
                batch=2, input_tokens=4096)
            analysis = shared_walk(d.during(DECODE, cache_len=4096) if phase
                        else d)

            def fused(framework):
                out, rewrite = Framework.of(
                    get_binding(framework).deployed(deployment),
                    FUSION_SCHEMES).rewrite(analysis.program, analysis)
                return out, rewrite, {
                    n.op: sorted(analysis.program.by_uid(u).name
                                 for u in n.origins)
                    for n in out.nodes() if isinstance(n, FusedNode)}

            for framework in ("vllm", "sglang"):
                _, _, ops = fused(framework)
                self.assertEqual(ops["qsa_index_q_norm_rope"], [
                    "index_q_nope", "index_q_pe", "index_q_pe_rope",
                    "index_q_rope", "indexer.q_layernorm",
                    "indexer.q_layernorm.rms"])
                if framework == "vllm" or not phase:
                    self.assertIn("index_k_sum",
                                  ops["qsa_index_k_compress_store"])
                else:
                    self.assertNotIn("qsa_index_k_compress_store", ops)
                self.assertIn("q_norm.rms",
                              ops["fused_qk_rmsnorm_rope_gate_split_gain"])
            out, rewrite, ops = fused("vllm")
            if phase:
                self.assertNotIn("flash_dsa_gated", ops)
                self.assertIn("attn_apply", ops["flash_dsa"])
                self.assertEqual(ops["sigmoid_gate_mul"],
                                 ["gate_act", "gated_attn"])
            else:
                self.assertIn("gated_attn", ops["flash_dsa_gated"])
                self.assertNotIn("flash_dsa", ops)
            sglang = fused("sglang")[2]
            self.assertNotIn("flash_dsa_gated", sglang)
            self.assertEqual(sglang["sigmoid_gate_mul"],
                             ["gate_act", "gated_attn"])

            def cache_reads(walk):
                return sum(int(t.kv_reads) for t in
                           traffic.derive(walk).leaves().values.values())

            self.assertEqual(
                cache_reads(rebase_analysis(analysis, out, rewrite)),
                cache_reads(analysis))

    def test_sglang_decode_does_not_claim_the_key_pool_it_replays(self):
        """sglang's CUDA-graph decode compresses every decoded row every step
        and dumps the rows that complete no block, while the decode key-pool
        form prices one completed block per compress_ratio steps — vLLM's law
        and the reference's. So the key launch is bound in prefill only, and
        the indexer QUERY prep, which runs every forward, is bound in both.
        `feeds: q` is what keeps the query plan off the key chain's own
        norm -> rope -> re-join once the wider key plan is not there to take
        it first."""
        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        d = Deployment(_program("Qwen__Qwen3.8-Flash-Next")).at(
            batch=2, input_tokens=4096)
        for phase in (None, DECODE):
            analysis = shared_walk(d.during(DECODE, cache_len=4096) if phase
                        else d)

            def spans(framework):
                out, _ = Framework.of(
                    get_binding(framework).deployed(deployment),
                    FUSION_SCHEMES).rewrite(analysis.program, analysis)
                return {n.op: sorted(analysis.program.by_uid(u).name
                                     for u in n.origins)
                        for n in out.nodes()
                        if n.op.startswith("qsa_index_")}

            vllm, sglang = spans("vllm"), spans("sglang")
            self.assertIn("qsa_index_k_compress_store", vllm, phase)
            self.assertIn("index_q_rope", sglang["qsa_index_q_norm_rope"])
            if phase:
                self.assertNotIn("qsa_index_k_compress_store", sglang)
                # and the query plan did not take the key chain instead
                self.assertEqual(set(sglang), {"qsa_index_q_norm_rope"})
            else:
                self.assertIn("index_k_sum",
                              sglang["qsa_index_k_compress_store"])

    def test_kimi_k3_whole_mixer_decode_follows_its_dispatch(self):
        """Kimi-K3 decode: vLLM runs the whole KDA mixer as one launch
        (`fused_kda_decode`) at 12 / 24 / 48 / 96 local heads on SM90+
        without speculation, else the per-channel gate + recurrence launch;
        sglang runs the per-channel launch at 96 local heads. In PREFILL
        vLLM runs FlashKDA, which
        takes the gate, beta and both l2 norms as kernel arguments — the
        per-channel span again, on SM90 and newer only. The conv and
        recurrent state reads stay on the launch."""
        from backend.arch.layout.parallel_config import ParallelConfig

        def walk(tp=1, s_verify=1, phase=DECODE):
            base = Deployment(_program("moonshotai__Kimi-K3"),
                              config=ParallelConfig(tp_attention=tp)).at(
                batch=2, input_tokens=128, s_verify=s_verify)
            return (shared_walk(base) if phase != DECODE else
                    shared_walk(base.during(DECODE, cache_len=4096)))

        def fired(framework, analysis, sku="H100-SXM"):
            deployment = ServingDeployment.at(gpu_sku=sku,
                                              activation_dtype="BF16")
            fused, rewrite = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            return ({n.op for n in fused.nodes()}
                    & {"fused_kda_decode", "fused_recurrent_kda_per_channel"},
                    fused, rewrite)

        whole, per_channel = ({"fused_kda_decode"},
                              {"fused_recurrent_kda_per_channel"})
        tp1 = walk()
        self.assertEqual(fired("vllm", tp1)[0], whole)
        self.assertEqual(fired("sglang", tp1)[0], per_channel)
        # A speculative verify: vLLM still runs the per-channel span
        # (kda.py:1136); sglang's SM10x verify kernel also writes per-token
        # roll-back state the graph has no node for, so neither sglang entry
        # claims a speculative step.
        speculating = walk(s_verify=3)
        self.assertEqual(fired("vllm", speculating)[0], per_channel)
        self.assertEqual(fired("sglang", speculating)[0], set())
        self.assertEqual(fired("vllm", tp1, sku="A100-80G-SXM")[0],
                         per_channel)
        # Prefill: FlashKDA's one launch on SM90+, and below it the Triton
        # chunk path, whose gate is its own launch — no plan for that pair.
        self.assertEqual(fired("vllm", walk(phase="prefill"))[0], per_channel)

        def cache_reads(analysis):
            return sum(int(t.kv_reads) for t in
                       traffic.derive(analysis).leaves().values.values())

        fused, rewrite = _only(Framework.of(
            get_binding("vllm").deployed(ServingDeployment.at(
                gpu_sku="H100-SXM", activation_dtype="BF16")),
            FUSION_SCHEMES), "fused_kda_decode").rewrite(tp1.program, tp1)
        self.assertIn("fused_kda_decode", {n.op for n in fused.nodes()})
        rebased = rebase_analysis(tp1, fused, rewrite)
        self.assertGreater(cache_reads(tp1), 0)
        self.assertEqual(cache_reads(rebased), cache_reads(tp1))

    def test_kda_small_launches_follow_the_frameworks(self):
        """The KDA mixers' small launches, at the deployments where each
        framework's source runs them as one.

        vLLM, Kimi-K3 at attention TP 32 on H100: the packed q|k|v conv at
        decode only (prefill convolves three times), the six-way merged input
        projection, the pre-FFN AttnRes add + readout + norm, the MLA q
        concat + latent cache insert, and the compiled output gate in both
        phases. GLM-5.3's decode keeps the same packed-conv and six-way
        projection families on another mixer.
        sglang packs the conv for both models and none of vLLM's K3-only
        launches; the MLA q concat + latent cache insert is no K3-only
        launch — both frameworks' kernels ask only the 512/64 widths — so
        sglang fuses it too. The K3 MLA is NoPE: no rotary op is left to
        bill."""
        from backend.arch.layout.parallel_config import ParallelConfig

        def walk(model, tp=32, phase=DECODE):
            base = Deployment(_program(model),
                              config=ParallelConfig(tp_attention=tp)).at(
                batch=1, input_tokens=128)
            return (shared_walk(base) if phase != DECODE else
                    shared_walk(base.during(DECODE, cache_len=4096)))

        def fused(framework, analysis):
            deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                              activation_dtype="BF16")
            program, _ = Framework.of(
                get_binding(framework).deployed(deployment),
                FUSION_SCHEMES).rewrite(analysis.program, analysis)
            out: dict = {}
            for n in program.nodes():
                if isinstance(n, FusedNode):
                    out.setdefault(n.op, []).append(sorted(
                        analysis.program.by_uid(u).name for u in n.origins))
            return out

        k3 = "moonshotai__Kimi-K3"
        k3_only = {"merged_column_parallel_linear_wide",
                   "add_attn_res_combine_norm"}
        decode = walk(k3)
        vllm = fused("vllm", decode)
        self.assertLessEqual(k3_only | {"causal_conv1d_packed",
                                        "sigmoid_gate_mul",
                                        "mla_q_concat_kv_insert"}, set(vllm))
        self.assertNotIn("causal_conv1d", vllm)
        self.assertNotIn("fused_qk_rope", vllm)
        self.assertTrue(all(m == sorted(
            ["q_conv1d", "q_act", "k_conv1d", "k_act", "v_conv1d", "v_act"])
            for m in vllm["causal_conv1d_packed"]))
        self.assertTrue(all(m == sorted(
            ["q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj"])
            for m in vllm["merged_column_parallel_linear_wide"]))
        self.assertTrue(all(m == sorted(["q_concat", "kv_concat"])
                            for m in vllm["mla_q_concat_kv_insert"]))
        # After attention, the attention add with the FFN readout and its
        # norm (model.py:1050-1068); before it, the previous layer's FFN add
        # with the attention readout and the input norm (model.py:1024-1035)
        # — a launch across the layer seam (`fusion.Seams`). At the exit the
        # pattern also meets the final norm, which vLLM runs later in
        # `compute_logits` (model.py:1413-1414): one launch a step this graph
        # bills fused that vLLM runs apart, a known gap of the graph (the
        # norm's call site is not a fact it holds), not a binding case.
        self.assertTrue(all(
            m in (sorted(["attn_residual", "mlp_res",
                          "post_attention_layernorm"]),
                  sorted(["mlp_residual", "attn_res", "input_layernorm"]),
                  sorted(["mlp_residual", "language_model.model.norm",
                          "language_model.model.output_attn_res"]))
            for m in vllm["add_attn_res_combine_norm"]))

        sglang = fused("sglang", decode)
        self.assertIn("causal_conv1d_packed", sglang)
        self.assertIn("sigmoid_gate_mul", sglang)
        self.assertFalse(k3_only & set(sglang))
        self.assertTrue(all(m == sorted(["q_concat", "kv_concat"])
                            for m in sglang["mla_q_concat_kv_insert"]))

        prefill = fused("vllm", walk(k3, phase="prefill"))
        self.assertNotIn("causal_conv1d_packed", prefill)
        self.assertIn("causal_conv1d", prefill)
        self.assertIn("merged_column_parallel_linear_wide", prefill)
        self.assertIn("sigmoid_gate_mul", prefill)

        glm = "zai-org__GLM-5.3-Flash"
        glm_decode = walk(glm, tp=1)
        for framework in ("vllm", "sglang"):
            got = fused(framework, glm_decode)
            with self.subTest(framework=framework):
                self.assertIn("causal_conv1d_packed", got)
                self.assertNotIn("causal_conv1d", got)
        self.assertIn("merged_column_parallel_linear_wide",
                      fused("vllm", glm_decode))

        ops = {n.op for n in decode.program.walked_nodes()}
        self.assertFalse(ops & {"rope", "rope_emb"}, ops & {"rope", "rope_emb"})

    def test_kimi_k3_merged_projection_covers_the_tp_boundaries(self):
        """Kimi-K3's six-way KDA input projection is one merged GEMM at the
        boundary attention-TP degrees. vLLM splits it into two stream GEMMs
        only at TP 8 (`_KDA_TP_SIZE`, low_latency_gemm.py:65) on a CUDA-graph
        decode step of at most 14 tokens
        (`KDA_PROJECTION_OVERLAP_MAX_TOKENS`, :59) — a token threshold,
        which is billed on the phase's default path rather than conditioned.
        TP1, the stream-GEMM TP, and TP96 span the whole divisor range;
        intermediate divisors repeat the same graph vocabulary."""
        from backend.arch.layout.parallel_config import ParallelConfig

        deployment = ServingDeployment.at(gpu_sku="H100-SXM",
                                          activation_dtype="BF16")
        framework = _only(Framework.of(
            get_binding("vllm").deployed(deployment), FUSION_SCHEMES),
            "merged_column_parallel_linear_wide")
        unfused = []
        for tp in (1, 8, 96):
            walk = shared_walk(Deployment(
                _program("moonshotai__Kimi-K3"),
                config=ParallelConfig(tp_attention=tp)).at(
                batch=1, input_tokens=16).during(DECODE, cache_len=64))
            fused, _ = framework.rewrite(walk.program, walk)
            if not any(n.op == "merged_column_parallel_linear_wide"
                       for n in fused.nodes()):
                unfused.append(tp)
        self.assertEqual(unfused, [])

    def test_a_launch_that_exposes_a_cache_join_keeps_its_readers_on_the_cache(
            self):
        """Kimi-K3's MLA decode launch joins the cached latent and the cached
        `k_pe` into the row it stores and exposes that row; the attention
        reads it through a head broadcast. The read must stay a cache read
        when the join is absorbed: the step's cache reads with the launch
        equal those without it (attention fused in both), and the
        attention's K input carries no activation bytes."""
        from backend.arch.layout.parallel_config import ParallelConfig

        walk = shared_walk(Deployment(
            _program("moonshotai__Kimi-K3"),
            config=ParallelConfig(tp_attention=32)).at(
            batch=1, input_tokens=128).during(DECODE, cache_len=4096))
        vllm = Framework.of(get_binding("vllm").deployed(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="BF16")), FUSION_SCHEMES)

        def leaves(*schemes):
            fused, rewrite = _only(vllm, *schemes).rewrite(walk.program, walk)
            return fused, traffic.derive(rebase_analysis(
                walk, fused, rewrite)).leaves().values

        def kv(values):
            return sum(int(t.kv_reads) for t in values.values())

        _, without = leaves("flash_attention")
        fused, after = leaves("mla_q_concat_kv_insert", "flash_attention")
        self.assertIn("mla_q_concat_kv_insert", {n.op for n in fused.nodes()})
        self.assertEqual(kv(after), kv(without))
        for node in fused.nodes():
            if node.op != "flash_attention":
                continue
            k_edge = [e for e in after[node.uid].in_edges if e.index == 1]
            self.assertEqual([int(e.elements) for e in k_edge], [0])

    def test_world_local_and_the_verify_width_read_the_walk(self):
        """`world_local` asks the per-rank count (value / world degree) in a
        set; the verify width is an ordinary `dims` fact — 1 is a step that
        does not speculate, and a width the walk left symbolic is in no
        list, so the entry does not bind."""
        import sympy as sp
        from backend.arch.eval.fusion import GraphCondition

        walk = (shared_walk(Deployment(_program("moonshotai__Kimi-K3"))
                .at(batch=1, input_tokens=16)
                .during(DECODE, cache_len=64)))
        by_name = {n.name: n for n in walk.program.walked_nodes()}
        bound = {"rule": by_name["delta_rule"]}

        class Degree:
            def __init__(self, degree):
                self.degree = degree

            def world_degree(self, world):
                return self.degree

        local = GraphCondition(world_local=(
            "tp_attention", "n_lin_v_heads", (12, 24, 48, 96)))
        for degree, admitted in ((1, True), (8, True), (16, False),
                                 (5, False)):
            stub = type("Walk", (), {"node_dims": walk.node_dims,
                                     "rule": Degree(degree)})()
            self.assertEqual(local.admits_match(bound, stub), admitted,
                             degree)

        plain = GraphCondition(dims=(("n_verify", (1,)),))
        verify = GraphCondition(dims=(("n_verify", (2, 3, 4)),))
        self.assertTrue(plain.admits_match(bound, walk))
        self.assertFalse(verify.admits_match(bound, walk))

        speculating = (shared_walk(Deployment(_program("moonshotai__Kimi-K3"))
                       .at(batch=1, input_tokens=16, s_verify=3)
                       .during(DECODE, cache_len=64)))
        spec_bound = {"rule": {n.name: n for n
                               in speculating.program.walked_nodes()
                               }["delta_rule"]}
        self.assertFalse(plain.admits_match(spec_bound, speculating))
        self.assertTrue(verify.admits_match(spec_bound, speculating))

        symbolic = (shared_walk(Deployment(_program("moonshotai__Kimi-K3"))
                    .at(batch=1, input_tokens=16,
                        s_verify=sp.Symbol("s_verify", integer=True,
                                           positive=True))
                    .during(DECODE, cache_len=64)))
        sym_bound = {"rule": {n.name: n for n
                              in symbolic.program.walked_nodes()
                              }["delta_rule"]}
        self.assertFalse(plain.admits_match(sym_bound, symbolic))
        self.assertFalse(verify.admits_match(sym_bound, symbolic))

    def test_unknown_framework_is_an_error(self):
        with self.assertRaises(KeyError):
            _framework("nonesuch")

    def test_composed_rewrites_flatten(self):
        """Several fusions in sequence must leave ONE correspondence a layer
        can cross in a single rebase."""
        d = Deployment(_program("deepseek-ai__DeepSeek-V3")).at(input_tokens=128)
        fused = shared_walk(d.fused(_framework("vllm")))
        for sources in fused.rewrite.origins.values():
            for source in sources:
                self.assertNotIn(source, fused.rewrite.origins,
                                 "an intermediate uid leaked into the result")


class TestFusedKernelCarriesItsConstants(unittest.TestCase):
    """A fused kernel stands in for the ops it absorbed, and its origins
    stop existing in the rewritten program — so the function constants
    their math took (`OpSpec.consts`, bound per node by the walk into
    `Analysis.consts`) follow the rewrite onto the fused node
    (`union_consts`). Kimi-K3's SiTU gate is the case: the launch takes the
    two betas the `situ` node's frame bound, and at the harness's own beta
    of 1 a measured kernel and its oracle agreed with each other about a
    gate no model has."""

    def _fused_consts(self, name: str):
        analysis = _walked(name)
        fused, rewrite = _framework("vllm").rewrite(analysis.program, analysis)
        rebased = rebase_analysis(analysis, fused, rewrite)
        return fused, rebased.consts

    def test_the_situ_launch_carries_the_models_betas(self):
        fused, consts = self._fused_consts("moonshotai__Kimi-K3")
        gates = [n for n in fused.nodes()
                 if isinstance(n, FusedNode) and n.op == "situ_and_mul"]
        self.assertTrue(gates, "no SiTU gate fused in Kimi-K3")
        for node in gates:
            self.assertEqual(consts.values.get(node.uid),
                             {"situ_beta": 4.0, "situ_linear_beta": 25.0})

    def test_a_launch_whose_ops_take_none_carries_none(self):
        _fused, consts = self._fused_consts("meta-llama__Llama-3.2-1B")
        self.assertEqual(consts.values, {})

    def test_origins_that_disagree_leave_the_constant_absent(self):
        # One launch takes one bound. Two absorbed clamps bound differently
        # give it none, so a consumer that has to compute it declines
        # rather than run one origin's bound for both.
        self.assertEqual(union_consts(({"clamp_limit": 10.0},
                                       {"clamp_limit": 10.0})),
                         {"clamp_limit": 10.0})
        self.assertEqual(union_consts(({"clamp_limit": 10.0, "situ_beta": 4.0},
                                       {"clamp_limit": 7.5})),
                         {"situ_beta": 4.0})


class TestTheDispatchZoneSplitsOneActivation(unittest.TestCase):
    """`routed` reads the topology: one act-and-mul pattern inside the
    dispatch zone (the routed expert bank) and outside it (dense MLP, shared
    expert) is two sites a framework may run differently. sglang's Kimi-K3
    runs `SituAndMul` only outside the zone; its MoE runner applies the
    routed SiTU itself, so the routed pair stays unfused there, while vLLM
    runs one launch at both sites."""

    def _sites(self, transform):
        analysis = _walked("moonshotai__Kimi-K3")
        fused, rewrite = transform.rewrite(analysis.program, analysis)
        sites = [analysis.program.is_routed(rewrite.origins[n.uid][0])
                 for n in fused.nodes()
                 if isinstance(n, FusedNode) and n.op == "situ_and_mul"]
        return sites.count(True), sites.count(False)

    def test_sglang_fuses_the_situ_outside_the_dispatch_zone_only(self):
        routed, outside = self._sites(_framework("sglang"))
        self.assertEqual(routed, 0)
        self.assertGreater(outside, 0)

    def test_the_condition_reads_both_ways(self):
        from backend.arch.eval.fusion import GraphCondition
        plain = self._sites(Fusion.named("situ_and_mul", FUSION_SCHEMES))
        self.assertTrue(all(plain))
        self.assertEqual(self._sites(Fusion.named(
            "situ_and_mul", FUSION_SCHEMES, GraphCondition(routed=True))),
            (plain[0], 0))
        self.assertEqual(self._sites(Fusion.named(
            "situ_and_mul", FUSION_SCHEMES, GraphCondition(routed=False))),
            (0, plain[1]))
        self.assertEqual(self._sites(_framework("vllm")), plain)


if __name__ == "__main__":
    unittest.main()


class TestASeamCrossesComponentsInsideOneLayer(unittest.TestCase):
    """A hyper-connection sublayer boundary: several of our components, one
    or two or three of the framework's launches (SEAM stage 1).

    A binding entry that declares `crosses: component` matches over the
    boundary view (`eval.fusion.BoundaryView`): a leaf's producer resolved to
    the leaf that writes the port it reads, and the block occurrence as the
    scope. The plans decompose the frameworks' wrapper into one plan per
    physical launch.
    """

    DEPLOYMENT: ClassVar[Any] = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.DEPLOYMENT = ServingDeployment.at(gpu_sku="H100-SXM",
                                              activation_dtype="BF16")

    def _ops(self, model: str, framework: str, phase: Optional[str] = None,
             ) -> Tuple[Any, Any, Any]:
        program = _program(model)
        walk = Deployment(program).at(input_tokens=128)
        if phase == "decode":
            walk = walk.during(DECODE, cache_len=4096)
        analysis = shared_walk(walk)
        fused, rewrite = Framework.of(
            get_binding(framework).deployed(self.DEPLOYMENT),
            FUSION_SCHEMES).rewrite(analysis.program, analysis)
        return analysis, fused, rewrite

    def _counts(self, model: str, framework: str, phase: str) -> dict:
        """Launches a step runs per mHC op: a launch across a layer seam
        runs once per seam, and a member it took only some runs of keeps
        the rest (`Rewrite.shares`)."""
        analysis, fused, rewrite = self._ops(model, framework, phase)
        runs = analysis.instances.rebase_counts(rewrite, fused)
        counts: dict = {}
        for node in fused.nodes():
            if node.op and node.op.startswith("mhc_"):
                counts[node.op] = (counts.get(node.op, 0)
                                   + runs.values.get(node.uid, 0))
        return counts

    def test_each_phase_takes_the_launches_its_framework_makes(self):
        """Both frameworks run the boundary as three launches on a large step
        and two on a small one, by the step's flattened token count. That
        threshold only moves the launch count at edge sizes, so the binding
        bills the default path of each phase (the cost-model doc's "How
        precisely to model a dispatch condition"): prefill reads as the
        large-token path, decode as the small-token one.

        DeepSeek-V4-Flash runs 44 layers a step (43 and the MTP layer), each
        with an attention→FFN boundary and, from the second on, the boundary
        from the previous layer's FFN into its attention: vLLM calls the
        fused post+pre on both (models/deepseek_v4/nvidia/model.py:1264 and
        :1236), a launch across the layer seam here (`fusion.Seams`). In
        prefill every boundary is post + GEMM + big fuse. In decode both
        are one fused post+pre launch plus the big fuse, except where a
        layer has no boundary before it or after it: layer 0's attention
        pre (and the MTP layer's) is a GEMM alone, and the stack's exit (and
        the MTP layer's) a post alone."""
        for framework in ("vllm", "sglang"):
            with self.subTest(framework=framework):
                self.assertEqual(
                    self._counts("deepseek-ai__DeepSeek-V4-Flash",
                                 framework, "prefill"),
                    {"mhc_post": 88, "mhc_pre_gemm": 88,
                     "mhc_pre_big_fuse": 88})
                self.assertEqual(
                    self._counts("deepseek-ai__DeepSeek-V4-Flash",
                                 framework, "decode"),
                    {"mhc_fused_post_pre": 86, "mhc_post": 2,
                     "mhc_pre_gemm": 2, "mhc_pre_big_fuse": 88})

    def test_the_big_fuse_reaches_the_sublayer_norm_across_components(self):
        """The launch our `mhc_pre_big_fuse` stands for ends in the sublayer
        RMSNorm. The reduce's leaves are in the `hc_weighted_reduce` the
        pre-mix calls, and the norm is the block's own leaf, which reads the
        pre-mix COMPONENT: three scopes, and the norm's producer is a wrapper
        the view resolves to the reduce's `sum_reduce`."""
        analysis, fused, _rewrite = self._ops(
            "deepseek-ai__DeepSeek-V4-Flash", "vllm", "prefill")
        for node in fused.nodes():
            if node.op != "mhc_pre_big_fuse":
                continue
            members = [analysis.program.by_uid(uid) for uid in node.origins]
            self.assertEqual(
                sorted(m.op for m in members),
                ["broadcast", "mul", "rmsnorm", "sum_reduce"])
            parents = {m.parent.uid if m.parent else None for m in members}
            self.assertGreater(len(parents), 1)
            blocks = {analysis.program.block_of(m.uid) for m in members}
            self.assertEqual(len(blocks), 1)
            self.assertNotIn(None, blocks)

    def test_a_crossing_match_never_leaves_its_block_occurrence(self):
        """The scope the view confines a match to is the block occurrence
        `Program.block_of` names — which is why the two false positives
        `_scopes` was written for stay rejected: gemma-4's vision and audio
        embedders and Kimi-K3's once-per-model `add_pe` are in different
        occurrences, or in none."""
        for model in ("deepseek-ai__DeepSeek-V4-Flash",
                      "zai-org__GLM-5.3-Flash"):
            for framework in ("vllm", "sglang"):
                analysis, fused, _rw = self._ops(model, framework, "decode")
                for node in fused.nodes():
                    if not (node.op or "").startswith("mhc_"):
                        continue
                    # A launch across a layer seam runs in the reading
                    # layer; its members of the layer before are that
                    # layer's, one occurrence each side.
                    for side in (
                            set(node.origins) - node.previous_iteration,
                            node.previous_iteration):
                        if not side:
                            continue
                        blocks = {analysis.program.block_of(uid)
                                  for uid in side}
                        self.assertEqual(len(blocks), 1, (model, framework,
                                                          node.op))
                        self.assertNotIn(None, blocks)

    def test_the_seam_preserves_flops_and_weights(self):
        """A fusion moves bytes, never arithmetic."""
        for framework in ("vllm", "sglang"):
            analysis, fused, rewrite = self._ops(
                "deepseek-ai__DeepSeek-V4-Flash", framework, "prefill")
            before = _leaf_totals(analysis)
            after = _leaf_totals(rebase_analysis(analysis, fused, rewrite))
            self.assertEqual(before, after, framework)

    TRIPLE: ClassVar[Tuple[str, ...]] = (
        "view_merge", "rmsnorm_unscaled", "linear_external_weight")

    def test_the_model_level_exit_is_not_in_the_view(self):
        """The exit reduce opens with a flatten, an unscaled RMS and an
        external-weight GEMM — which is `mhc_pre_gemm`'s whole pattern, op
        for op. What keeps that plan off the model's exit is the view and
        not the match: the exit hangs off the model rather than off a block
        invocation, so none of its leaves is in `scopes` to be matched.

        Asserted on the view directly, because the rewrite cannot tell us:
        the 9-node `fused_hc_head` is more specific and claims those leaves
        first, so a result-only test stays green even with the guard gone.
        The positive control is the same triple one layer down — the
        in-layer launch that DID fuse — plus the head firing in both
        frameworks.
        """
        for framework in ("vllm", "sglang"):
            with self.subTest(framework=framework):
                analysis, fused, _rw = self._ops(
                    "deepseek-ai__DeepSeek-V4-Flash", framework, "prefill")
                program = analysis.program
                view = BoundaryView.of(program, list(program.walked_nodes()))

                exit_node, = [node for node in program.walked_nodes()
                              if node.op == "hc_head"
                              and program.block_of(node.uid) is None]
                leaves = tuple(exit_node.children)
                self.assertEqual(tuple(leaf.op for leaf in leaves)[:3],
                                 self.TRIPLE)
                # Named, not `assertNotIn`: the scopes table is one entry
                # per walked node, and a failure should print the leaves.
                self.assertEqual([leaf.name for leaf in leaves
                                  if leaf.uid in view.scopes], [])

                in_layer = [node for node in fused.nodes()
                            if node.op == "mhc_pre_gemm"]
                self.assertTrue(in_layer, framework)
                origins = in_layer[0].origins
                self.assertEqual(
                    sorted(program.by_uid(uid).op for uid in origins),
                    sorted(self.TRIPLE))
                self.assertEqual([program.by_uid(uid).name for uid in origins
                                  if uid not in view.scopes], [])

                self.assertTrue([node for node in fused.nodes()
                                 if node.op == "fused_hc_head"], framework)

    def test_the_tree_resolves_a_consumer_that_names_an_emptied_component(self):
        """`materialize_tree` on both phases. A launch takes the leaves out
        of the components they were in, so a surviving consumer that names
        one of those components — the next post-mix reads the previous
        post's `hc_post`, whose `add` the fused form absorbed and re-exports
        as its residual port — has to be rewired to the launch. Resolving
        every component reference in the materialized tree is the check: a
        component that no longer writes the tensor is not a producer."""
        program = _program("deepseek-ai__DeepSeek-V4-Flash")
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                walk = Deployment(program).at(input_tokens=128)
                if phase == "decode":
                    walk = walk.during(DECODE, cache_len=4096)
                analysis = shared_walk(walk)
                fused, rewrite = Framework.of(
                    get_binding("vllm").deployed(self.DEPLOYMENT),
                    FUSION_SCHEMES).rewrite(analysis.program, analysis)
                tree = materialize_tree(analysis.program, fused, rewrite)
                post = Program.of(analysis.program.resolved, tree)
                seams = [node for node in post.nodes()
                         if (node.op or "").startswith("mhc_")]
                self.assertTrue(seams, phase)
                for node in post.nodes():
                    for ref in node.inputs_obj or ():
                        if ref.src is None or not ref.src.children:
                            continue
                        # Raises where a reference still names a component
                        # that no longer holds the leaf writing that port.
                        post.output_leaf(ref.src, ref.port or "out")

    def test_the_other_hyper_connection_families_are_not_this_boundary(self):
        """Hy4's iHC has no Sinkhorn split and Qwen3.8-Flash-Next combines
        and norms instead — neither is bound by the stage-1 plans.
        (DeepSeek-V4.1 IS this boundary on a shifted schedule; the next
        class pins it.)"""
        for model in ("tencent__Hy4-preview",
                      "Qwen__Qwen3.8-Flash-Next"):
            for framework in ("vllm", "sglang"):
                with self.subTest(model=model, framework=framework):
                    self.assertEqual(
                        self._counts(model, framework, "prefill"), {})


class TestTheShiftedSeamIsTheSameLaunchesOverACarriedPre(unittest.TestCase):
    """DeepSeek-V4.1's hyper-connection chain (SEAM stage 1, the delayed
    schedule): each sublayer's mix coefficients are consumed by the NEXT
    sublayer, and both frameworks run that boundary with the same launches
    the V4 seam prices — the post-mix, the prenorm projection and the
    reduce + norm epilogue — plus V4.1's own spellings audited in the
    bindings:

      * the projection reads the already-flattened stream, so the two-node
        `mhc_pre_gemm_stream` plan bills it (the `mhc_pre_gemm` signature);
      * vLLM's epilogue is the delayed `MHC_PRE_NORM_KERNEL` and sglang's is
        `hc_combine_norm`, both billed by the `mhc_pre_big_fuse` plan (the
        carried pre is a 32-byte-per-token delta on a seam moving the whole
        residual, and the Sinkhorn split stays a billed node in both).

    The model has eleven distinct block templates, each with an in-layer
    attention→FFN seam and a cross-layer FFN→attention one, so 22 of each
    per-seam launch; the stage-level exit collapse + norm is the 23rd
    epilogue. vLLM's decode takes the fused small-token form for the
    in-layer seam, as V4's does; sglang turns its fused path off for V4.1
    (`hc_pre_from_prev_sublayer`), so both of its phases bill the same
    three launches.
    """

    DEPLOYMENT: ClassVar[Any] = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.DEPLOYMENT = ServingDeployment.at(gpu_sku="H100-SXM",
                                              activation_dtype="BF16")

    def _ops(self, framework: str, phase: Optional[str] = None
             ) -> Tuple[Any, Any, Any]:
        program = _program("deepseek-ai__DeepSeek-V4.1-Flash")
        walk = Deployment(program).at(input_tokens=128)
        if phase == "decode":
            walk = walk.during(DECODE, cache_len=4096)
        analysis = shared_walk(walk)
        fused, rewrite = Framework.of(
            get_binding(framework).deployed(self.DEPLOYMENT),
            FUSION_SCHEMES).rewrite(analysis.program, analysis)
        return analysis, fused, rewrite

    def _counts(self, framework: str, phase: str) -> dict:
        _analysis, fused, _rewrite = self._ops(framework, phase)
        counts: dict = {}
        for node in fused.nodes():
            if node.op and node.op.startswith("mhc_"):
                counts[node.op] = counts.get(node.op, 0) + 1
        return counts

    def test_each_phase_takes_the_launches_its_framework_makes(self):
        self.assertEqual(
            self._counts("vllm", "prefill"),
            {"mhc_post": 22, "mhc_pre_gemm_stream": 22,
             "mhc_pre_big_fuse": 23})
        self.assertEqual(
            self._counts("vllm", "decode"),
            {"mhc_fused_post_pre": 11, "mhc_post": 11,
             "mhc_pre_gemm_stream": 11, "mhc_pre_big_fuse": 23})
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                self.assertEqual(
                    self._counts("sglang", phase),
                    {"mhc_post": 22, "mhc_pre_gemm_stream": 22,
                     "mhc_pre_big_fuse": 23})

    def test_the_shifted_seam_preserves_flops_and_weights(self):
        """A fusion moves bytes, never arithmetic."""
        for framework in ("vllm", "sglang"):
            analysis, fused, rewrite = self._ops(framework, "prefill")
            before = _leaf_totals(analysis)
            after = _leaf_totals(rebase_analysis(analysis, fused, rewrite))
            self.assertEqual(before, after, framework)

    def test_a_crossing_match_never_leaves_its_block_occurrence(self):
        for framework in ("vllm", "sglang"):
            for phase in ("prefill", "decode"):
                with self.subTest(framework=framework, phase=phase):
                    analysis, fused, _rw = self._ops(framework, phase)
                    for node in fused.nodes():
                        if not (node.op or "").startswith("mhc_"):
                            continue
                        blocks = {analysis.program.block_of(uid)
                                  for uid in node.origins}
                        self.assertEqual(len(blocks), 1,
                                         (framework, phase, node.op))
                        self.assertNotIn(None, blocks)

    def test_the_projection_bills_every_mix_stats_including_the_entry(self):
        """The two-node plan matches where V4's three-node one cannot: the
        attention-side stats read the block's flat stream input with no
        `view_merge` leaf in between — and the members of every projection
        match are exactly the unscaled RMS and the external-weight GEMM."""
        analysis, fused, _rw = self._ops("vllm", "prefill")
        nodes = [n for n in fused.nodes() if n.op == "mhc_pre_gemm_stream"]
        self.assertTrue(nodes)
        for node in nodes:
            self.assertEqual(
                sorted(analysis.program.by_uid(uid).op
                       for uid in node.origins),
                ["linear_external_weight", "rmsnorm_unscaled"])

    def test_the_deepseek_v4_seam_is_unchanged(self):
        """Binding V4.1 moved nothing for the model the plans were written
        for: the same launches a step as the stage-1 class pins
        (`test_each_phase_takes_the_launches_its_framework_makes`), both
        frameworks."""
        for framework in ("vllm", "sglang"):
            with self.subTest(framework=framework):
                program = _program("deepseek-ai__DeepSeek-V4-Flash")
                for phase, expect in (
                        ("prefill", {"mhc_post": 88, "mhc_pre_gemm": 88,
                                     "mhc_pre_big_fuse": 88}),
                        ("decode", {"mhc_fused_post_pre": 86, "mhc_post": 2,
                                    "mhc_pre_gemm": 2,
                                    "mhc_pre_big_fuse": 88})):
                    walk = Deployment(program).at(input_tokens=128)
                    if phase == "decode":
                        walk = walk.during(DECODE, cache_len=4096)
                    analysis = shared_walk(walk)
                    fused, rewrite = Framework.of(
                        get_binding(framework).deployed(self.DEPLOYMENT),
                        FUSION_SCHEMES).rewrite(analysis.program, analysis)
                    runs = analysis.instances.rebase_counts(rewrite, fused)
                    counts: dict = {}
                    for node in fused.nodes():
                        if node.op and node.op.startswith("mhc_"):
                            counts[node.op] = (counts.get(node.op, 0)
                                               + runs.values.get(node.uid, 0))
                    self.assertEqual(counts, expect, (framework, phase))


class TestTheBoundaryViewKeepsTheScopeGuard(unittest.TestCase):
    """A launch is one kernel over one set of rows, so a `crosses: component`
    match may leave the component but not the thing that says how many times
    its members run.

    Two graphs the ordinary immediate-parent scope rejected for free, and a
    view over the whole block occurrence has to reject on purpose. Both are
    walked as the CHECKPOINT (`Deployment.checkpoint`) — these components are
    in the encoder stacks, which the language-model walk does not run, and a
    guard tested on a walk that cannot reach the nodes asserts nothing. Each
    test therefore names the candidate nodes, asserts the view SAW them
    (`scopes`, `edges`), and asserts a legitimate match of the same scheme
    that must survive, so neither a matcher that stops matching nor a guard
    that stops guarding can pass.

    The shipped crossing entries are architecture-gated, so neither model
    reaches one through a binding; the fusion is built directly, because the
    matcher is where the guard lives.
    """

    def _checkpoint(self, model: str):
        return Deployment(_program(model)).at(input_tokens=128).checkpoint()

    def _crossing(self, scheme: str, analysis):
        program = analysis.program
        view = BoundaryView.of(program, list(program.walked_nodes()))
        fusion = Fusion.named(scheme, FUSION_SCHEMES, crosses="component")
        fused, _rewrite = fusion.rewrite(program, analysis)
        return view, [node for node in fused.nodes()
                      if isinstance(node, FusedNode)]

    def _named(self, program, name: str):
        found = [node for node in program.walked_nodes()
                 if node.name.rsplit(".", 1)[-1] == name]
        self.assertTrue(found, f"the walk holds no node named {name!r}")
        return found

    def test_gemma_4_keeps_its_two_vision_embedders_apart(self):
        """gemma-4 runs `gemma4_vision_embedder` twice — once for the image
        stack, once for the video stack — and both feed the same model-level
        merge. The two occurrences' leaves are in the view, and the guard is
        that their scopes differ; the language model's own QKV triple fusing
        is what says the matcher is still matching."""
        analysis = self._checkpoint("google__gemma-4-12B-it")
        program = analysis.program
        view, nodes = self._crossing("qkv_parallel_linear", analysis)

        embedders = self._named(program, "patch_dense")
        self.assertEqual(len(embedders), 2, [n.name for n in embedders])
        scopes = []
        for node in embedders:                      # the view saw both
            self.assertIn(node.uid, view.scopes, node.name)
            scopes.append(view.scopes[node.uid])
        self.assertNotEqual(*scopes)                # and keeps them apart
        self.assertEqual({program.block_of(n.uid) for n in embedders},
                         {0, 1})

        for node in nodes:                          # no match spans the two
            blocks = {program.block_of(uid) for uid in node.origins}
            self.assertEqual(len(blocks), 1, (node.op, node.name))
            self.assertNotIn(None, blocks, (node.op, node.name))
        self.assertEqual(
            [sorted(program.by_uid(uid).name for uid in node.origins)
             for node in nodes],
            [["k_proj", "q_proj", "v_proj"]])

    def test_kimi_k3_does_not_fuse_27_vision_blocks_into_one_norm(self):
        """Kimi-K3's vision tower ends `blocks(×27) → final_layernorm`. The
        view resolves the norm's producer — the repeated component — to the
        `out` add inside it, which is the whole point of the view and also
        the trap: that add runs 27 times and the norm once. The scope carries
        the repeat, so they are two launches; the `attn_residual → norm1`
        pair inside the block, which does run together, still fuses."""
        analysis = self._checkpoint("moonshotai__Kimi-K3")
        program = analysis.program
        view, nodes = self._crossing("fused_add_rmsnorm", analysis)

        final_norm, = self._named(program, "final_layernorm")
        component, = [ref.src for ref in final_norm.inputs_obj]
        self.assertTrue(component.children, component.name)
        self.assertEqual(int(component.repeat), 27)
        written, _port = program.output_leaf(component, "out")
        self.assertEqual((written.name, written.op), ("out", "add"))

        # The view resolved the edge, so only the scope stands between them.
        self.assertIn((written.uid, "out"), view.edges[final_norm.uid])
        self.assertNotEqual(view.scopes[written.uid],
                            view.scopes[final_norm.uid])

        members = [sorted(program.by_uid(uid).name.rsplit(".", 1)[-1]
                          for uid in node.origins) for node in nodes]
        self.assertNotIn(sorted(["out", "final_layernorm"]), members)
        self.assertIn(sorted(["attn_residual", "norm1"]), members)
        for node in nodes:
            blocks = {program.block_of(uid) for uid in node.origins}
            self.assertEqual(len(blocks), 1, (node.op, node.name))
            self.assertNotIn(None, blocks, (node.op, node.name))


class TestTheFusedTreeIsAValidGraph(unittest.TestCase):
    """`materialize_tree` after a boundary-crossing rewrite, in both phases.

    The crossing launch takes leaves out of the components they were in, so
    the tree it produces has to answer two questions the earlier tests
    stopped short of: can it be built at all for a decode walk, and does a
    surviving consumer that named one of those components now read the
    launch?
    """

    MODEL: ClassVar[str] = "deepseek-ai__DeepSeek-V4-Flash"

    def _tree(self, phase: str):
        program = _program(self.MODEL)
        walk = Deployment(program).at(input_tokens=128)
        if phase == "decode":
            walk = walk.during(DECODE, cache_len=4096)
        analysis = shared_walk(walk)
        fused, rewrite = Framework.of(
            get_binding("vllm").deployed(ServingDeployment.at(
                gpu_sku="H100-SXM", activation_dtype="BF16")),
            FUSION_SCHEMES).rewrite(analysis.program, analysis)
        tree = materialize_tree(analysis.program, fused, rewrite)
        return analysis, fused, Program.of(analysis.program.resolved, tree)

    def test_both_phases_materialize(self):
        """A decode walk splices its decode variants, so the tree holds uids
        the walked program never shows. The rewrite allocates its fused uids
        above the tree's own counter, not above the program's, or the tree
        hands out a uid the rewrite did not expect ("fusion tree allocated
        uid N, expected rewrite uid M")."""
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                _analysis, fused, post = self._tree(phase)
                launches = {node.op for node in post.nodes()}
                self.assertIn("mhc_pre_big_fuse", launches)
                self.assertEqual(
                    {node.uid for node in fused.nodes()
                     if isinstance(node, FusedNode)}
                    - {node.uid for node in post.nodes()},
                    set())

    def test_a_consumer_of_an_emptied_component_reads_the_launch(self):
        """The post-mix's `add` is absorbed, and the component that held it
        keeps no child that writes its output. Every reference that named
        that component — the next post-mix's residual, the next pre-mix's
        flatten — must name the launch and the port it publishes the tensor
        on; an emptied component is not a producer."""
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                _analysis, _fused, post = self._tree(phase)
                unresolved = []
                for node in post.nodes():
                    for ref in node.inputs_obj or ():
                        if ref.src is None or not ref.src.children:
                            continue
                        try:
                            post.output_leaf(ref.src, ref.port or "out")
                        except ValueError as refusal:
                            unresolved.append(
                                (node.name, ref.src.name, ref.port,
                                 str(refusal)))
                self.assertEqual(unresolved, [])

    def test_the_residual_consumer_names_the_fused_port(self):
        """Concretely, in the phase where the post-mix is fused into the
        next pre-mix (`mhc_fused_post_pre`, decode), the block's own closing
        post-mix still reads the residual that launch wrote: its reference
        resolves to the launch, on the port the plan exposes."""
        _analysis, _fused, post = self._tree("decode")
        seen = 0
        for node in post.nodes():
            for ref in node.inputs_obj or ():
                src = ref.src
                if src is None or src.op != "mhc_fused_post_pre":
                    continue
                seen += 1
                self.assertIn(ref.port, ("", "residual_out", "output"))
        self.assertGreater(seen, 0)


class TestARegionIsNotSubtractedTwiceOrBackwards(unittest.TestCase):
    """`_passed_through` on hand-built matches.

    The region a fused boundary passes through is a SUBTRACTION from the
    tensor it publishes, and two topologies make that subtraction lie. Two
    regions of one input that overlap are each subtracted in full, and their
    sum can exceed the input. And a view that publishes MORE than it reads
    (`pad_rows`, a view by registration whose output is the larger tensor)
    is subtracted at its own, larger size. Either way the launch would be
    billed below zero.

    Neither reaches a price today — `fused_overlap_compress_norm_rope_store`
    has two overlapping windows over one `external.kv` and escapes only
    because both of them are live, and no plan yet absorbs a growing view —
    so they are asserted here, on a match built by hand, where they cost
    milliseconds and cannot stop being visible. Both are a topology; no
    model is needed to have one.
    """

    def _passed(self, members, pre, sources, output_ports=None,
                allocates=False):
        from backend.arch.eval.fusion import _passed_through
        from backend.arch.eval.values import Shape

        class _Fused:
            allocates_output = allocates
            output_dtype: ClassVar[dict] = {}

        return _passed_through(
            _Fused(), members, {uid: Shape(out=out, ins=tuple(ins))
                                for uid, (out, ins) in pre.items()},
            sources, output_ports or {})

    @staticmethod
    def _node(uid, op, *inputs):
        """A member reading `inputs`, each a producing member or, for `None`,
        the one tensor arriving from outside this match."""
        class _Ref:
            def __init__(self, src):
                self.src, self.port = src, ""
                self.external_sid = None if src is not None else "outside"

        class _Node:
            def __init__(self):
                self.uid, self.op = uid, op
                self.inputs_obj = tuple(_Ref(src) for src in inputs)

        return _Node()

    def test_one_dead_slice_is_the_region_it_exposes(self):
        """The control: a single dead view publishes a region OF its input,
        and the launch is billed the rest of it."""
        dead = self._node(1, "slice", None)
        live = self._node(2, "rope_emb", None)
        moved = self._passed(
            [dead, live], {1: ((4,), ((10,),)), 2: ((6,), ((10,),))},
            [(1, 0)])
        self.assertIsNone(moved, "a reader of the tensor as it arrives")

        moved = self._passed([dead], {1: ((4,), ((10,),))}, [(1, 0)])
        self.assertEqual(int(moved.reads[0]), 6)

    def test_two_overlapping_dead_slices_bill_the_whole_input(self):
        """Two windows over one input, 6 and 6 of 10. Subtracted in full the
        read is -2; nothing here knows they are disjoint, so the launch is
        billed the tensor it was handed."""
        a, b = self._node(1, "slice", None), self._node(2, "slice", None)
        moved = self._passed(
            [a, b], {1: ((6,), ((10,),)), 2: ((6,), ((10,),))},
            [(1, 0)])
        self.assertEqual(int(moved.reads[0]), 10)

    def test_a_dead_view_that_grows_bills_the_whole_input(self):
        """`pad_rows` publishes 16 rows of the 10 it reads — a view by
        registration, at the same rank, repeating nothing, so the rank and
        repetition guards let it through. Its output is no region of its
        input to subtract."""
        from backend.arch.ops import OP_REGISTRY

        spec = OP_REGISTRY.get("pad_rows")
        self.assertTrue(spec.is_view_op, "the premise: a registered view")
        pad = self._node(1, "pad_rows", None)
        moved = self._passed([pad], {1: ((16,), ((10,),))}, [(1, 0)])
        self.assertEqual(int(moved.reads[0]), 10)

    def test_a_view_exposed_as_a_port_still_costs_nothing(self):
        """And the case the clamp must not catch: a view over the outside
        exposed as an output port passes the whole of it through, which is
        the naive graph's price for a view — zero. Exactly covering the
        tensor is a partition; only overshooting it is not."""
        view = self._node(1, "slice", None)
        moved = self._passed([view], {1: ((10,), ((10,),))}, [],
                             output_ports={1: "gate_out"})
        self.assertEqual(int(moved.writes["gate_out"]), 0)
