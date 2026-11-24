"""Layer-0 integration: spec_tree.iter_leaves + Analysis.iter_checkpoint.

Tests the full chain end-to-end on real SpecLoader-resolved specs.
Covers:
  - iter_leaves emits one LeafTensor per weight-bearing leaf
    (linear / parameter_vector / norm / embedding)
  - role propagation from component-invocation node down to leaves
  - Analysis.iter_checkpoint produces on-disk rows for every committed spec
  - validate_overrides_against_spec catches typo'd tensor names
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any, ClassVar, Dict

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from backend.arch.loader import SpecLoader
from backend.arch.spec_tree import (
    iter_leaves, iter_tree, iter_block_scopes,
    validate_overrides_against_spec,
)
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.snapshots import STRUCTURAL_SNAPSHOT_IDS

pytestmark = pytest.mark.regression  # walks every snapshot

# Tests still call the local name `walk_spec` for the leaf list.
walk_spec = iter_leaves
from backend.arch.schema import QuantContext


def _checkpoint(resolved):
    """Every tensor the checkpoint holds: the language model's and each
    encoder's (`Deployment.checkpoint`)."""
    return list(
        Deployment(Program.of(resolved)).at(batch=1, input_tokens=8)
        .checkpoint().iter_checkpoint()
    )


SNAPSHOT_DIR = PROJECT_ROOT / "backend" / "arch" / "snapshots"


def _resolve(model_name: str):
    """Load a snapshot's config + match it against the spec library."""
    path = SNAPSHOT_DIR / f"{model_name}.json"
    config = json.loads(path.read_text())["config"]
    return SpecLoader().match(config, model_id=model_name.replace("__", "/", 1))


class TestWalkSpecBasics(unittest.TestCase):
    """Sanity on a simple non-quantized model."""

    def test_llama_3_2_1b_walks(self):
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        self.assertIsNotNone(resolved)
        logical = walk_spec(resolved)
        # At least: embed + lm_head + per-block (q/k/v/o + gate/up/down)
        self.assertGreaterEqual(len(logical), 8)

        # Embed should be present
        names = [lt.name for lt in logical]
        self.assertIn("model.embed_tokens.weight", names)

        # All linear ops should have in_dim/out_dim populated
        for lt in logical:
            if lt.op == "linear":
                self.assertIsNotNone(lt.in_dim)
                self.assertIsNotNone(lt.out_dim)
                self.assertGreater(lt.in_dim, 0)
                self.assertGreater(lt.out_dim, 0)

    def test_block_tensor_names_use_the_stack_address(self):
        """Llama's layers sit at `model.layers.<i>` in its checkpoint."""
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        logical = walk_spec(resolved)
        block_tensors = [lt for lt in logical if "self_attn" in lt.name
                         or "mlp" in lt.name]
        self.assertGreater(len(block_tensors), 0)
        for lt in block_tensors:
            self.assertTrue(
                lt.name.startswith("model.layers.X."),
                f"Block tensor {lt.name!r} should start with 'model.layers.X.'",
            )


class TestRolePropagation(unittest.TestCase):
    """Engine attaches role to component-invocation nodes; walker
    must propagate it down to descendant leaves so per_role matching
    works."""

    def test_dsv3_attention_role_propagates_to_leaves(self):
        resolved = _resolve("deepseek-ai__DeepSeek-V3")
        logical = walk_spec(resolved)
        # MLA attention has multiple linears: q_a_proj, q_b_proj, etc.
        attn_linears = [
            lt for lt in logical
            if lt.role == "attention" and lt.op == "linear"
        ]
        # At least q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, o_proj.
        self.assertGreaterEqual(len(attn_linears), 5)

    def test_dsv3_ffn_role_propagates(self):
        resolved = _resolve("deepseek-ai__DeepSeek-V3")
        logical = walk_spec(resolved)
        ffn_linears = [
            lt for lt in logical
            if lt.role and "ffn" in lt.role and lt.op == "linear"
        ]
        self.assertGreater(len(ffn_linears), 0)


class TestCheckpointSurface(unittest.TestCase):
    """End-to-end: walker + Analysis.iter_checkpoint."""

    def test_no_quant_context_emits_only_default_bf16(self):
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        mat = _checkpoint(resolved)
        self.assertGreater(len(mat), 0)
        for m in mat:
            self.assertEqual(m.storage_dtype, "BF16")
            self.assertIsNone(m.method)
            self.assertEqual(m.selected_by, "default")
            self.assertFalse(m.is_metadata)

    def test_dsv3_with_fp8_per_role_emits_siblings(self):
        resolved = _resolve("deepseek-ai__DeepSeek-V3")
        resolved.spec.quant_context = QuantContext(
            # The unquantised remainder is BF16 (the header's F32 outliers
            # aside); the per-role "none" leaves two float dtypes in the
            # checkpoint, so the bulk one is stated.
            default_dtype="BF16",
            per_role={
                "attention": {"method": "fp8", "block_h": 128, "block_w": 128},
                "ffn_dense": {"method": "fp8", "block_h": 128, "block_w": 128},
                "default":   {"method": "none"},
            },
        )
        mat = _checkpoint(resolved)

        # Every role-tagged linear gets a weight + at least one
        # metadata sibling.
        weights_quant = [m for m in mat
                         if m.method == "fp8" and not m.is_metadata]
        siblings = [m for m in mat if m.is_metadata]
        self.assertGreater(len(weights_quant), 0)
        self.assertEqual(len(weights_quant), len(siblings),
                         "FP8 emits exactly one sibling per linear")

        # Storage dtype check
        for m in weights_quant:
            self.assertEqual(m.storage_dtype, "F8_E4M3")
        for m in siblings:
            self.assertEqual(m.storage_dtype, "F32")
            self.assertTrue(m.name.endswith(".weight_scale_inv"))

    def test_explicit_overrides_pin_role_tensors_to_default(self):
        """quant_context.overrides expressed inline in the spec
        winning over per_role for matching tensor names."""
        resolved = _resolve("deepseek-ai__DeepSeek-V3")
        # Find an actual attention leaf name to target.
        logical = walk_spec(resolved)
        attn_names = [lt.name for lt in logical if lt.role == "attention"]
        self.assertGreater(len(attn_names), 0)
        sample = attn_names[0]
        resolved.spec.quant_context = QuantContext(
            per_role={
                "attention": {"method": "fp8", "block_h": 128, "block_w": 128},
                "ffn_dense": {"method": "fp8", "block_h": 128, "block_w": 128},
            },
            overrides=[{"tensors": [sample], "dtype": "BF16"}],
            default_dtype="BF16",
        )
        mat = _checkpoint(resolved)
        pinned = [m for m in mat if m.name == sample]
        self.assertGreaterEqual(len(pinned), 1)
        for p in pinned:
            self.assertEqual(p.storage_dtype, "BF16")
            # The intent: an OVERRIDE matched, not per_role and not default.
            # The label's text is the producer's: the incumbent wrote
            # `override:manual`, `eval` writes `override:<pattern>` — which
            # names the rule that fired, and is the better audit string.
            self.assertTrue(p.selected_by.startswith("override:"), p.selected_by)


class TestComponentScopedOverride(unittest.TestCase):
    """gemma-2's sliding-window and global attention are two components that
    both store `model.layers.X.self_attn.o_proj.weight`. A `component:` scope
    reaches one of them and leaves the other on its role's format."""

    SCOPED: ClassVar[Dict[str, Any]] = {
        "select": {"component": "swa_attention",
                   "tensors_re": r"self_attn\.o_proj\.weight$"},
        "dtype": "F8_E4M3", "method": "fp8", "block_h": 128, "block_w": 128}

    def _analysis(self, overrides):
        resolved = _resolve("google__gemma-2-9b")
        resolved.spec.quant_context = QuantContext(
            per_role={"default": {"method": "none"}}, overrides=overrides)
        return Deployment(Program.of(resolved)).at(
            batch=1, input_tokens=8).analyze()

    def test_the_scope_reaches_only_the_named_component(self):
        analysis = self._analysis([self.SCOPED])
        seen = {}
        for row in analysis.iter_checkpoint():
            if not row.name.startswith("model.layers.X.self_attn.o_proj."):
                continue
            node = analysis.program.by_uid(row.uid)
            seen.setdefault(node.enclosing_components[-1], set()).add(
                (row.name, row.storage_dtype))
        self.assertEqual(seen, {
            "swa_attention": {
                ("model.layers.X.self_attn.o_proj.weight", "F8_E4M3"),
                ("model.layers.X.self_attn.o_proj.weight_scale_inv", "F32")},
            # The snapshot's checkpoint (axolotl-ai-co/gemma-2-9b) stores
            # its unquantised tensors at F32 — the header's dtype is the
            # shipped one, the config's only cross-checks it.
            "gqa_attention": {
                ("model.layers.X.self_attn.o_proj.weight", "F32")},
        })
        # The role stays the attention role: cost books where it belongs.
        roles = {analysis.program.by_uid(row.uid).role
                 for row in analysis.tensors()
                 if row.name == "model.layers.X.self_attn.o_proj.weight"}
        self.assertEqual(roles, {"attention"})

    def test_an_overlapping_row_is_a_load_error(self):
        broad = {"tensors_re": r"o_proj\.weight$", "dtype": "BF16"}
        with self.assertRaisesRegex(
                ValueError, r"overrides\[0\].*overrides\[1\].*both select"):
            self._analysis([broad, self.SCOPED])

    def test_an_unknown_component_is_refused(self):
        from backend.arch.schema import ArchSpec

        with self.assertRaisesRegex(ValueError, "no component of that name"):
            ArchSpec.model_validate({
                "name": "t",
                "dims": {"d_model": "config.hidden_size"},
                "blocks": {"blk": {"graph": {
                    "norm": {"op": "rmsnorm", "inputs": ["x"],
                             "dim": "d_model"},
                }, "outputs": ["norm"]}},
                "model": [{"name": "layers", "template": "blk",
                           "repeat": "1"}],
                "quant_context": {"overrides": [{
                    "select": {"component": "no_such_attention",
                               "tensors_re": r"\.weight$"},
                    "dtype": "BF16"}]},
            })

    def test_literal_names_are_checked_inside_the_scope(self):
        resolved = _resolve("google__gemma-2-9b")
        logical = walk_spec(resolved)
        name = "model.layers.X.self_attn.o_proj.weight"
        inside = QuantContext(overrides=[{
            "select": {"component": "swa_attention", "tensors": [name]},
            "dtype": "BF16"}])
        self.assertEqual(validate_overrides_against_spec(inside, logical), [])
        outside = QuantContext(overrides=[{
            "select": {"component": "swa_attention",
                       "tensors": ["model.embed_tokens.weight"]},
            "dtype": "BF16"}])
        self.assertEqual(
            len(validate_overrides_against_spec(outside, logical)), 1)


class TestValidateOverridesAgainstSpec(unittest.TestCase):

    def test_existing_tensor_passes(self):
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        logical = walk_spec(resolved)
        # Pick a real name and assert it validates.
        real_name = next(
            lt.name for lt in logical if lt.name == "model.embed_tokens.weight"
        )
        qc = QuantContext(overrides=[
            {"tensors": [real_name], "dtype": "BF16"},
        ])
        errs = validate_overrides_against_spec(qc, logical)
        self.assertEqual(errs, [])

    def test_missing_tensor_reported(self):
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        logical = walk_spec(resolved)
        qc = QuantContext(overrides=[
            {"tensors": ["totally.not.a.real.tensor"], "dtype": "BF16"},
        ])
        errs = validate_overrides_against_spec(qc, logical)
        self.assertEqual(len(errs), 1)
        self.assertIn("totally.not.a.real.tensor", errs[0])

    def test_regex_overrides_not_checked(self):
        resolved = _resolve("meta-llama__Llama-3.2-1B")
        logical = walk_spec(resolved)
        qc = QuantContext(overrides=[
            {"tensors_re": ".*never_matches.*", "dtype": "BF16"},
        ])
        # Regex overrides may legitimately match nothing (proposal R6).
        errs = validate_overrides_against_spec(qc, logical)
        self.assertEqual(errs, [])


class TestCustomOpAsLinear(unittest.TestCase):
    """Walker recognizes custom_ops with `out_key` declared as
    linear-like and emits a LogicalTensor with op='linear'. Used by
    gpt-oss `mxfp4_fused_linear` (params: "0", weight materialized
    via the mxfp4 quant scheme).
    """

    def test_gpt_oss_mxfp4_fused_linear_emitted_as_linear(self):
        resolved = _resolve("openai__gpt-oss-20b")
        logical = walk_spec(resolved)
        # mxfp4_fused_linear should be emitted as op='linear', NOT skipped.
        expert_linears = [
            lt for lt in logical
            if "experts.gate_up_proj.weight" in lt.name
            or "experts.down_proj.weight" in lt.name
        ]
        self.assertGreater(len(expert_linears), 0)
        for lt in expert_linears:
            self.assertEqual(lt.op, "linear")
            # One tensor per layer: gpt-oss-20b has two block templates of
            # 12 layers each.
            self.assertEqual(lt.multiplicity, 12)

    def test_gpt_oss_expert_axis_is_the_weights_leading_axis(self):
        """gpt-oss's packed experts declare `shape: [n_expert_copies, out,
        in]`: the expert count is the leaf's leading axis, not a copy count
        of the emitted LogicalTensor."""
        resolved = _resolve("openai__gpt-oss-20b")
        logical = walk_spec(resolved)
        E = resolved.dim_values["n_experts"]
        expert_linears = [
            lt for lt in logical
            if "experts.gate_up_proj.weight" in lt.name
        ]
        self.assertGreater(len(expert_linears), 0)
        for lt in expert_linears:
            self.assertEqual(int(lt.shape[0]), E)
            self.assertEqual(lt.multiplicity, 12)


class TestStructuralSpecsMaterialize(unittest.TestCase):
    """Smoke test: one structural representative per family must walk +
    materialize without crashing. Catches regressions in the Program/Analysis
    walk across the template families the corpus currently has.
    """

    def test_structural_snapshots(self):
        snapshots = [
            SNAPSHOT_DIR / f"{model_id.replace('/', '__')}.json"
            for model_id in STRUCTURAL_SNAPSHOT_IDS
        ]
        self.assertGreater(len(snapshots), 0)

        loader = SpecLoader()
        skipped = []
        walked = 0
        for snap_path in snapshots:
            try:
                config = json.loads(snap_path.read_text())["config"]
                resolved = loader.match(
                    config, model_id=snap_path.stem.replace("__", "/", 1))
            except Exception as e:
                skipped.append((snap_path.name, str(e)[:80]))
                continue
            if resolved is None:
                skipped.append((snap_path.name, "no spec match"))
                continue
            with self.subTest(model=snap_path.stem):
                logical = walk_spec(resolved)
                self.assertGreater(
                    len(logical), 0,
                    f"{snap_path.name}: walker emitted zero tensors",
                )
                # Materialize must produce at least one MaterializedTensor
                # per LogicalTensor (the weight itself); when the spec
                # has a quant_context that triggers a per_role match,
                # additional sibling tensors push the count higher. Both
                # cases are valid — assert the lower bound only.
                mat = _checkpoint(resolved)
                self.assertGreaterEqual(len(mat), len(logical))
            walked += 1
        self.assertGreaterEqual(
            walked, len(STRUCTURAL_SNAPSHOT_IDS),
            f"A structural representative failed to materialize. Skipped: "
            f"{skipped[:5]}",
        )


class TestIterBlockScopes(unittest.TestCase):
    """`iter_block_scopes` is the shared per-block scoping every
    tree-rewriting pass (comm injection / fusion / future SP / dequant)
    must use, since template-local spec_ids collide across templates."""

    @staticmethod
    def _walk(ns):
        for n in ns:
            yield n
            yield from TestIterBlockScopes._walk(n.children or ())

    def test_scopes_separate_cross_template_collisions(self):
        # DSv3 is hybrid (dense + MoE + MTP): the same template-local
        # spec_id (`post_attention_layernorm`, `self_attn.attn_apply`)
        # occurs in multiple block subtrees. The helper's whole job is to
        # put each block in its own scope so a per-scope index sees every
        # occurrence — a SINGLE global index would collapse them.
        from backend.analyzer import fetch_config
        resolved = SpecLoader().match(fetch_config("deepseek-ai/DeepSeek-V3"),
                                      model_id="deepseek-ai/DeepSeek-V3")
        tree = iter_tree(resolved)
        scopes = list(iter_block_scopes(tree))
        # One model-root scope + one per block.
        self.assertEqual(len(scopes), 1 + len(tree.blocks))

        # `post_attention_layernorm` lives in the dense AND MoE blocks —
        # so it must surface in >= 2 separate scopes, never collapsed.
        scopes_with_norm = sum(
            1 for scope in scopes
            if any(n.display_path == "post_attention_layernorm"
                   for n in self._walk(scope.top_level))
        )
        self.assertGreaterEqual(
            scopes_with_norm, 2,
            "the shared spec_id must appear in each block's scope "
            "separately; a global index would have collapsed it to 1",
        )
        # Within ONE scope, a spec_id repeats only via primary folding
        # (a component + its primary leaf share the id) — never via the
        # cross-template collision the helper exists to prevent. So each
        # scope holds AT MOST one block's worth of that norm.
        for scope in scopes:
            n_norm = sum(
                1 for n in self._walk(scope.top_level)
                if n.display_path == "post_attention_layernorm"
            )
            self.assertLessEqual(n_norm, 1)

    def test_scope_top_level_is_the_real_roots_object(self):
        # In-place mutation by a pass must propagate to the parent tree,
        # so each block scope's top_level must BE the real roots list.
        from backend.analyzer import fetch_config
        resolved = SpecLoader().match(fetch_config("meta-llama/Llama-2-7b-hf"),
                                      model_id="meta-llama/Llama-2-7b-hf")
        tree = iter_tree(resolved)
        scopes = list(iter_block_scopes(tree))
        block_scopes = scopes[1:]
        self.assertEqual(len(block_scopes), len(tree.blocks))
        for scope, (_meta, roots) in zip(block_scopes, tree.blocks, strict=True):
            self.assertIs(scope.top_level, roots)

    def test_scope_nodes_allocate_from_the_complete_tree(self):
        from backend.analyzer import fetch_config
        resolved = SpecLoader().match(fetch_config("meta-llama/Llama-2-7b-hf"),
                                      model_id="meta-llama/Llama-2-7b-hf")
        tree = iter_tree(resolved)
        existing = {node.uid for node in self._walk(tree.top_level)}
        for _meta, roots in tree.blocks:
            existing.update(node.uid for node in self._walk(roots))

        allocated = []
        for index, scope in enumerate(iter_block_scopes(tree)):
            node = scope.make_node(
                name=f"scope_synthetic_{index}",
                op="add",
                is_component=False,
            )
            allocated.append(node.uid)
        self.assertEqual(len(set(allocated)), len(allocated))
        self.assertTrue(existing.isdisjoint(allocated))
        self.assertEqual(tree._next_uid, max(allocated) + 1)


class TestModuleDeclaration(unittest.TestCase):
    """`module:` — the nn.Module the checkpoint has at a node's position.

    `inner_modules` is reachable only through specs the snapshot corpus does
    not cover (Qwen3.5's MTP adapter is its only user and there is no Qwen3.5
    snapshot), so the corpus cannot exercise it and it is asserted directly.
    """

    def test_the_three_states_of_module(self):
        """Absent is not the same as written-as-null, and the difference is
        the whole point: one says nothing, the other says nothing-is-here."""
        from backend.arch.schema import GraphNode
        from backend.arch.spec_tree import module_name_of

        self.assertEqual(
            module_name_of(GraphNode(op="linear", inputs=["x"]), "q_proj"),
            "q_proj", "an omitted `module` must fall back to the node's own name")
        self.assertIsNone(
            module_name_of(GraphNode(op="linear", inputs=["x"], module=None), "qkv"),
            "`module: null` must contribute no segment")
        self.assertEqual(
            module_name_of(GraphNode(op="linear", inputs=["x"],
                                   module="query_key_value"), "qkv"),
            "query_key_value")

    @staticmethod
    def _graph(**nodes):
        from backend.arch.schema import BlockGraph
        return BlockGraph(inputs=["x"], graph=nodes, outputs=[list(nodes)[-1]])

    def _builder(self):
        from backend.arch.spec_tree import SpecTreeWalker
        return SpecTreeWalker.__new__(SpecTreeWalker)

    def test_a_call_site_can_name_a_segment_the_library_declares_absent(self):
        """The library says "no module here" for the checkpoints it was
        written against; a consumer whose checkpoint DOES hold one names it.
        Qwen3.5's MTP head really has `self.layers = nn.ModuleList(...)` where
        DeepSeek's and GLM's store the layer flat."""
        from backend.arch.schema import GraphNode
        from backend.arch.spec_tree import module_name_of
        graph = self._graph(
            inner=GraphNode(op="pre_norm", inputs=["x"], module=None))
        self.assertIsNone(module_name_of(graph.graph["inner"], "inner"))
        out = self._builder()._apply_inner_modules(graph, {"inner": "layers"})
        self.assertEqual(module_name_of(out.graph["inner"], "inner"), "layers")
        # and the source graph is untouched — the override is a copy, so two
        # consumers of one library component cannot see each other's answer.
        self.assertIsNone(module_name_of(graph.graph["inner"], "inner"))

    def test_a_call_site_can_remove_a_segment_the_library_names(self):
        """The other direction. This is why `module` has a real third value
        rather than leaning on `model_fields_set`: `model_copy` does not
        update that set, so an override to `null` used to read as "omitted"
        and fall back to the node's own name."""
        from backend.arch.schema import GraphNode
        from backend.arch.spec_tree import module_name_of
        graph = self._graph(inner=GraphNode(op="pre_norm", inputs=["x"]))
        out = self._builder()._apply_inner_modules(graph, {"inner": None})
        self.assertIsNone(module_name_of(out.graph["inner"], "inner"))

    def test_an_override_naming_a_node_that_does_not_exist_is_an_error(self):
        """A silently-ignored override is a spec that says something about a
        checkpoint and has no effect — the failure mode this whole file is
        about. Renaming the library's node must break the call site loudly."""
        from backend.arch.schema import GraphNode
        graph = self._graph(inner=GraphNode(op="pre_norm", inputs=["x"]))
        with self.assertRaises(ValueError) as caught:
            self._builder()._apply_inner_modules(graph, {"typo": None})
        self.assertIn("typo", str(caught.exception))


class TestAliasAdapterInlining(unittest.TestCase):
    def test_a_one_node_delegating_template_contributes_no_segment(self):
        """An adapter's body is one node delegating to a component, so the
        walk descends into the sub-graph and the wrapper's own name never
        reaches a path. Read off the graph, not declared: the `primary: true`
        that used to declare it also meant "fold this node's name away" one
        branch over, and marking the eight-node `mtp_block_with_head` with it
        took the wrong reading and dropped 1,956,140,032 parameters in
        silence."""
        from backend.arch.spec_tree import iter_leaves

        # `chatglm_block` is a one-node adapter over `pre_norm`.
        names = [lt.tensor_name for lt in iter_leaves(_resolve("THUDM__glm-4-9b"))]
        self.assertTrue(names)
        self.assertFalse(
            [n for n in names if ".layer." in n],
            "the adapter's own node name leaked into a checkpoint path",
        )
        self.assertIn(
            "transformer.encoder.layers.X.self_attention.query_key_value.weight",
            names)

    def test_a_one_node_template_that_says_something_is_not_an_adapter(self):
        """Inlining drops the node, so it may only happen when the node
        carries nothing but the delegation. A `repeat:` on it would be a
        multiplier that never applies and a `role:` an attribution that never
        lands — both silently, since the leaves underneath still appear."""
        from backend.arch.schema import GraphNode
        from backend.arch.spec_tree import is_pure_delegation

        delegating = GraphNode(op="pre_norm", inputs=["x"],
                               params={"attn": "gqa_attention"})
        self.assertTrue(is_pure_delegation(delegating))
        self.assertTrue(is_pure_delegation(
            delegating.model_copy(update={"inner_modules": {"attn": "x"},
                                          "module": None})))
        for says_something in ({"repeat": "n_experts"}, {"role": "ffn_moe"},
                               {"dims": {"in": "H"}},
                               # naming a segment is also saying something:
                               # inlining would erase a module that exists
                               {"module": "layers"}):
            with self.subTest(**says_something):
                self.assertFalse(is_pure_delegation(
                    delegating.model_copy(update=says_something)))

    def test_a_multi_node_template_keeps_its_nodes(self):
        """The counterpart: `mtp_block_with_head` is not an adapter, so
        nothing about it is inlined and every node it declares survives.
        DSv3 declares a checkpoint address after the backbone, so the MTP
        leaves land in the backbone's layer index space (the checkpoint
        ships them as `model.layers.61.*`), not under `mtp.*`."""
        from backend.arch.spec_tree import iter_leaves

        names = {lt.tensor_name
                 for lt in iter_leaves(_resolve("deepseek-ai__DeepSeek-V3"))}
        for expected in ("model.layers.X.embed_tokens.weight",
                         "model.layers.X.eh_proj.weight",
                         "model.layers.X.shared_head.head.weight",
                         "model.layers.X.enorm.weight",
                         "model.layers.X.hnorm.weight"):
            self.assertIn(expected, names)
        self.assertFalse({n for n in names if "mtp." in n})


class TestCheckpointAddress(unittest.TestCase):
    """Checkpoint path and index ownership are independent declarations."""

    def test_the_three_declared_contracts_compose_three_prefixes(self):
        from backend.arch.schema import CheckpointAddress
        from backend.arch.spec_tree import BlockMeta, checkpoint_prefix

        def aux(**kw):
            return BlockMeta(template_name="t", count=1, stack_name="mtp",
                             aux_kind="mtp", group_name="mtp", **kw)

        def address(prefix, index="local"):
            return CheckpointAddress(prefix=prefix, index=index)

        with self.assertRaisesRegex(ValueError, "no resolved checkpoint"):
            checkpoint_prefix(aux())
        self.assertEqual(
            checkpoint_prefix(aux(checkpoint=address("mtp"))),
            "mtp.X.")
        self.assertEqual(
            checkpoint_prefix(aux(checkpoint=address("model.mtp.layers"))),
            "model.mtp.layers.X.")
        # A stack with no index hangs off its scope's root.
        self.assertEqual(
            checkpoint_prefix(aux(checkpoint=CheckpointAddress(root="model"))),
            "model.")
        # A prefix matching the backbone plus an after-backbone index composes
        # the same normalized key, folding the MTP artifact into that pattern.
        backbone = BlockMeta(template_name="t", count=61, stack_name="layers",
                             checkpoint=address("model.layers"))
        self.assertEqual(
            checkpoint_prefix(aux(
                checkpoint=address("model.layers", {"after": "backbone"}),
                backbone_layers=61)),
            checkpoint_prefix(backbone))

    def test_after_backbone_without_a_bound_backbone_count_refuses(self):
        """No bound backbone layer count → no tail index for the tensors
        to ship under. Composing a name anyway would silently invent
        one, so the composition raises instead."""
        from backend.arch.schema import CheckpointAddress
        from backend.arch.spec_tree import BlockMeta, checkpoint_prefix

        meta = BlockMeta(template_name="t", count=1, stack_name="mtp",
                         aux_kind="mtp", group_name="mtp",
                         checkpoint=CheckpointAddress(
                             prefix="model.layers",
                             index={"after": "backbone"}),
                         backbone_layers=None)
        with self.assertRaises(ValueError) as caught:
            checkpoint_prefix(meta)
        self.assertIn("after: backbone", str(caught.exception))

    def test_the_vocabulary_is_closed_at_the_schema(self):
        from pydantic import ValidationError

        from backend.arch.schema import ModelLayer

        with self.assertRaises(ValidationError):
            ModelLayer(name="mtp", group=True, aux_kind="mtp",
                       checkpoint={"prefix": "mtp.X", "index": "local"},
                       layers=[ModelLayer(name="mtp", template="t")])

    def test_each_entry_declares_its_own_kind_of_address(self):
        """A stack declares the prefix its index follows, an op the path of
        its module, a group the root its ops hang off — each field is a fact
        about one kind of entry, and on another it would be a second voice."""
        from pydantic import ValidationError

        from backend.arch.schema import ModelLayer

        ModelLayer(name="layers", template="t", repeat="n_layers",
                   checkpoint={"prefix": "model.layers"})
        ModelLayer(name="lm_head", op="linear",
                   checkpoint={"path": "lm_head"})
        with self.assertRaises(ValidationError):
            ModelLayer(name="lm_head", op="linear",
                       checkpoint={"prefix": "lm_head"})
        with self.assertRaises(ValidationError):
            ModelLayer(name="layers", template="t", repeat="n_layers",
                       checkpoint={"root": "model"})
        with self.assertRaises(ValidationError):
            ModelLayer(name="norm", op="rmsnorm",
                       checkpoint={"index": "local"})


    def test_a_call_nothing_repeats_takes_no_index_from_its_group(self):
        """An indexed group's index is its repeated stack's: a component call
        beside it that nothing repeats hangs off the root it declares (Hy4's
        DFly drafter: `model.dfly_layers.X.*` layers beside
        `model.dfly_layers.mtp_fusion.*`), else the group's root — not under
        the stack's index, where the checkpoint has no such module."""
        import tempfile

        from backend.arch.spec_tree import iter_leaves

        spec = """\
match: { model_type: [_draft_group], model_id: [_draft_group] }
dims: { d_model: config.hidden_size, n_layers: "1", n_mtp_layers: "2" }
blocks:
  blk:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [n]
  fuse:
    graph:
      enorm: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [enorm]
model_inputs:
  input_ids: { kind: tokens }
checkpoint: { root: model }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: d_model, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: layers, template: blk, repeat: n_layers, inputs: [embed],
      checkpoint: { prefix: model.layers }, streams: { hidden: { carries: true } } }
  - name: draft
    group: true
    aux_kind: speculative
    checkpoint: { prefix: model.draft, index: local }
    inputs: [layers]
    layers:
      - { name: fusion, template: fuse, inputs: [layers],
          checkpoint: { root: model.draft.fusion } }
      - { name: layers, template: blk, repeat: n_mtp_layers, inputs: [fusion],
          streams: { hidden: { carries: true } } }
      - { name: tail, template: fuse, inputs: [layers] }
      - { name: head, op: linear, boundary_role: output_head, inputs: [tail],
          dims: { in: d_model, out: d_model },
          weight_axes: [{ replicated: true }, { replicated: true }] }
  - { name: lm_head, op: linear, boundary_role: output_head, inputs: [layers],
      dims: { in: d_model, out: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for sub in ("components/library", "components/adapters",
                        "components/drafts", "models"):
                (base / sub).mkdir(parents=True)
            (base / "models" / "_draft_group.yaml").write_text(spec)
            resolved = SpecLoader(base_dir=str(base)).match(
                {"model_type": "_draft_group", "hidden_size": 8,
                 "torch_dtype": "bfloat16"}, model_id="_draft_group")
            names = {lt.tensor_name for lt in iter_leaves(resolved)}
        self.assertIn("model.draft.fusion.enorm.weight", names)
        self.assertIn("model.draft.X.n.weight", names)
        self.assertIn("model.enorm.weight", names)
        self.assertFalse({n for n in names if n.startswith("model.draft.X.enorm")})

    def test_a_call_nothing_repeats_declares_only_its_root(self):
        from pydantic import ValidationError

        from backend.arch.schema import ModelLayer

        ModelLayer(name="fusion", template="t",
                   checkpoint={"root": "model.draft.fusion"})
        with self.assertRaises(ValidationError):
            ModelLayer(name="fusion", template="t",
                       checkpoint={"prefix": "model.draft"})


class TestQuantOverrideSelector(unittest.TestCase):
    def test_stack_scope_lives_under_select(self):
        from pydantic import ValidationError

        from backend.arch.schema import OverrideEntry

        entry = OverrideEntry.model_validate({
            "select": {"stack": "mtp", "tensors_re": r"\.weight$"},
            "dtype": "BF16",
        })
        self.assertEqual(entry.select.stack, "mtp")
        self.assertEqual(entry.select.tensors_re, r"\.weight$")

        with self.assertRaisesRegex(ValidationError, "move it under"):
            OverrideEntry.model_validate({
                "stack": "mtp", "tensors_re": r"\.weight$", "dtype": "BF16",
            })
        with self.assertRaisesRegex(ValidationError, "mutually exclusive"):
            OverrideEntry.model_validate({
                "select": {"stack": "mtp", "tensors_re": r"\.weight$"},
                "tensors_re": r"\.bias$",
                "dtype": "BF16",
            })

    def test_component_scope_lives_under_select(self):
        from pydantic import ValidationError

        from backend.arch.schema import OverrideEntry

        entry = OverrideEntry.model_validate({
            "select": {"component": "mla_attention",
                       "tensors_re": r"o_proj\.weight$"},
            "dtype": "BF16",
        })
        self.assertEqual(entry.select.component, "mla_attention")
        with self.assertRaisesRegex(ValidationError, "move it under"):
            OverrideEntry.model_validate({
                "component": "mla_attention",
                "tensors_re": r"o_proj\.weight$", "dtype": "BF16",
            })
        with self.assertRaisesRegex(ValidationError, "must be non-empty"):
            OverrideEntry.model_validate({
                "select": {"component": " ", "tensors_re": r"\.weight$"},
                "dtype": "BF16",
            })

    def test_the_component_is_part_of_the_duplicate_signature(self):
        from pydantic import ValidationError

        row = {"select": {"component": "a", "tensors_re": r"o_proj\.weight$"},
               "dtype": "BF16"}
        other = {"select": {"component": "b", "tensors_re": r"o_proj\.weight$"},
                 "dtype": "BF16"}
        QuantContext(overrides=[row, other])
        with self.assertRaisesRegex(ValidationError, "duplicates selector"):
            QuantContext(overrides=[row, dict(row)])


class TestModelLayerGuards(unittest.TestCase):
    def test_geometry_guard_still_checks_every_non_quantized_model_layer(self):
        """Aux/override validation must not capture the pre-existing per-layer
        guards in its indentation scope."""
        from backend.arch.schema import ArchSpec

        with self.assertRaisesRegex(ValueError, "literal integer divisor"):
            ArchSpec.model_validate({
                "name": "t",
                "dims": {"d_model": "config.hidden_size"},
                "blocks": {"blk": {"graph": {
                    "norm": {"op": "rmsnorm", "inputs": ["x"],
                             "dim": "d_model"},
                }, "outputs": ["norm"]}},
                "model": [
                    {"name": "packed.weight", "op": "parameter_vector",
                     "dim": "d_model / 2"},
                    {"name": "layers", "template": "blk", "repeat": "1"},
                ],
            })


if __name__ == "__main__":
    unittest.main()
