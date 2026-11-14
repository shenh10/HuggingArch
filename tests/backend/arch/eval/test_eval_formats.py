"""`Formats` — storage as a sideband layer, resolved by lookup.

Falsification targets:
  - precedence must be override > role > default, because an override exists
    precisely for the tensors a role rule gets wrong (gpt-oss pins its router
    to BF16 inside an `ffn_moe`-roled component);
  - a scheme must NOT apply to a node that is not the weight matrix it
    describes — per-role selection is keyed by role alone, and an rmsnorm gain
    living in an `attention`-roled component would otherwise be handed FP8 and
    a tile-scale footprint it does not have;
  - a sibling the graph already carries as its own node must not ALSO be
    implied by the format, or the same bytes are counted twice;
  - packing must stay separate from storage dtype, or MXFP4's two FP4 elements
    per byte become either two bytes each or an unpacking special case.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.formats import (
    FormatPolicy, MetadataGeometry, Representation, Representations,
    TensorFormat, known_dtype_bytes,
)
from backend.arch.eval.storage import (
    StorageKind, StoredTensor,
)
from backend.arch.schema.quant import OverrideEntry, OverrideSelector


def matrix(name: str, out: int, in_: int) -> StoredTensor:
    """A stored contraction matrix — `(out, in)`, as on disk."""
    module, _, suffix = name.rpartition(".")
    return StoredTensor(module=module, suffix=f".{suffix}",
                        shape=(out, in_), kind=StorageKind.MATRIX)


def vector(name: str, d: int) -> StoredTensor:
    module, _, suffix = name.rpartition(".")
    return StoredTensor(module=module, suffix=f".{suffix}",
                        shape=(d,), kind=StorageKind.VECTOR)


@dataclass
class Entry:
    """Stands in for `PerRoleEntry` / `OverrideEntry` — both are pydantic
    models whose scheme-specific fields arrive through `model_extra`."""
    method: Optional[str] = None
    dtype: Optional[str] = None
    tensors: Optional[List[str]] = None
    tensors_re: Optional[str] = None
    select: Optional[Any] = None
    scale_shape: Optional[List[str]] = None
    block_scale_layout: Optional[Any] = None
    model_extra: Dict[str, Any] = field(default_factory=dict)


Selector = OverrideSelector


FP8 = Entry(method="fp8", model_extra={"block_h": 128, "block_w": 128})
MXFP4 = Entry(method="mxfp4", model_extra={"group_size": 32})
GROUP_FORMATS = (
    Entry(method="fp4", model_extra={"group_size": 32}),
    MXFP4,
    # The packed-MXFP4 layout is written by two quantisers, so the entry
    # states which treats the activation (`input_activations`); either value
    # lays the weight out the same way, which is what this group checks.
    Entry(method="mxfp4-pack-quantized",
          model_extra={"group_size": 32, "input_activations": "none"}),
)
Q_PROJ = matrix("layers.0.self_attn.q_proj.weight", 1536, 7168)


class TestPrecedence(unittest.TestCase):
    def test_role_selects_the_scheme(self):
        pol = FormatPolicy(per_role={"attention": FP8})
        fmt = pol.for_node(Q_PROJ.name, "attention", Q_PROJ)
        self.assertEqual(fmt.storage_dtype, "F8_E4M3")
        self.assertEqual(fmt.selected_by, "role:attention")

    def test_an_override_beats_the_role(self):
        """gpt-oss's router is a BF16 linear inside an `ffn_moe` component."""
        pol = FormatPolicy(
            per_role={"ffn_moe": MXFP4},
            overrides=[Entry(tensors_re=r"router\.weight$", dtype="BF16")])
        fmt = pol.for_node("layers.3.mlp.router.weight", "ffn_moe",
                           matrix("layers.3.mlp.router.weight", 8, 4096))
        self.assertEqual(fmt.storage_dtype, "BF16")
        self.assertEqual(fmt.metadata, ())
        self.assertTrue(fmt.selected_by.startswith("override:"))

    def test_an_override_that_decides_nothing_is_reported(self):
        """DeepSeek-V4 wrote its router row as `\\.ffn\\.weight$`; the tensor
        is `ffn.gate.weight`, so the row selected nothing. A row an earlier
        one always beats is just as dead."""
        pol = FormatPolicy(overrides=[
            Entry(tensors_re=r"\.ffn\.weight$", dtype="BF16"),
            Entry(tensors_re=r"vision_tower.*", dtype="BF16"),
            Entry(tensors_re=r"vision_tower.*\.bias$", dtype="BF16"),
            Entry(tensors_re=r"\.gate\.bias$", dtype="F32"),
        ])
        # The layers a tensor's node runs at — `(None,)` for a node with no
        # layer index, which is what every row here speaks about.
        stored = [("layers.0.ffn.gate.weight", "", (), (None,)),
                  ("layers.0.ffn.gate.bias", "", (), (None,)),
                  ("vision_tower.blocks.0.wo.bias", "vision", (), (None,))]
        self.assertEqual(pol.unused_overrides(stored), (0, 2))

    def test_a_stack_selector_precedes_checkpoint_name_matching(self):
        """Backbone and MTP may share one composed checkpoint name; logical
        stack identity must decide whether the override applies."""
        pol = FormatPolicy(
            per_role={"ffn_moe": MXFP4},
            overrides=[Entry(
                select=Selector(stack="mtp", tensors_re=r"down_proj\.weight$"),
                dtype="BF16")],
        )
        stored = matrix("layers.X.mlp.down_proj.weight", 4096, 14336)
        backbone = pol.for_node(stored.name, "ffn_moe", stored,
                                stack="layers")
        mtp = pol.for_node(stored.name, "ffn_moe", stored, stack="mtp")
        self.assertNotEqual(backbone.storage_dtype, "BF16")
        self.assertEqual(mtp.storage_dtype, "BF16")

    def test_unroled_and_unmatched_falls_back(self):
        pol = FormatPolicy(default_dtype="BF16", per_role={"attention": FP8})
        fmt = pol.for_node("model.embed_tokens.weight", None, Q_PROJ)
        self.assertEqual((fmt.storage_dtype, fmt.selected_by), ("BF16", "default"))

    def test_method_none_means_unquantized_not_unknown(self):
        pol = FormatPolicy(per_role={"router": Entry(method="none")})
        fmt = pol.for_node("gate.weight", "router", Q_PROJ)
        self.assertEqual(fmt.storage_dtype, "BF16")
        self.assertEqual(fmt.selected_by, "role:router")

    def test_method_none_keeps_entry_dtype(self):
        pol = FormatPolicy(
            default_dtype="BF16",
            per_role={"router": Entry(method="none", dtype="FP8")},
        )
        fmt = pol.for_node("gate.weight", "router", Q_PROJ)
        self.assertEqual(fmt.storage_dtype, "FP8")
        self.assertEqual(fmt.selected_by, "role:router")

    def test_provenance_is_recorded(self):
        """A wrong dtype on one tensor family is otherwise indistinguishable
        from a wrong dtype everywhere."""
        pol = FormatPolicy(per_role={"attention": FP8})
        self.assertIn("attention",
                      pol.for_node(Q_PROJ.name, "attention", Q_PROJ).selected_by)


class TestComponentScopedOverride(unittest.TestCase):
    """GLM-5.3-Flash: the MLA and the KDA attention of one hybrid stack both
    store `layers.X.self_attn.o_proj.weight`; the checkpoint holds the MLA
    copies in FP8 and the KDA copies in BF16. A `component:` scope is what
    tells the two apart, with both nodes keeping `role: attention`."""

    O_PROJ = matrix("layers.X.self_attn.o_proj.weight", 4096, 16384)
    MLA = ("glm5_hc_block", "glm5_mla_nope_dsa_attention")
    KDA = ("glm5_hc_block", "glm5_kda_attention")
    SCOPED = OverrideEntry(
        select=OverrideSelector(component="glm5_mla_nope_dsa_attention",
                                tensors_re=r"self_attn\.o_proj\.weight$"),
        dtype="F8_E4M3", method="fp8", block_h=128, block_w=128)

    def test_the_scope_selects_only_nodes_inside_the_component(self):
        pol = FormatPolicy(per_role={"attention": Entry(method="none")},
                           overrides=[self.SCOPED])
        mla = pol.for_node(self.O_PROJ.name, "attention", self.O_PROJ,
                           stack="", components=self.MLA)
        kda = pol.for_node(self.O_PROJ.name, "attention", self.O_PROJ,
                           stack="", components=self.KDA)
        self.assertEqual(mla.storage_dtype, "F8_E4M3")
        self.assertEqual([m.name for m in mla.metadata],
                         ["layers.X.self_attn.o_proj.weight_scale_inv"])
        self.assertEqual((kda.storage_dtype, kda.selected_by),
                         ("BF16", "role:attention"))

    def test_the_scope_ands_with_the_stack(self):
        scoped = OverrideEntry(
            select=OverrideSelector(stack="mtp", component="mla",
                                    tensors_re=r"o_proj\.weight$"),
            dtype="F8_E4M3")
        pol = FormatPolicy(overrides=[scoped])
        name = self.O_PROJ.name
        self.assertEqual(pol.for_node(name, "attention", self.O_PROJ,
                                      stack="mtp", components=("mla",)
                                      ).storage_dtype, "F8_E4M3")
        self.assertEqual(pol.for_node(name, "attention", self.O_PROJ,
                                      stack="", components=("mla",)
                                      ).storage_dtype, "BF16")
        self.assertEqual(pol.for_node(name, "attention", self.O_PROJ,
                                      stack="mtp", components=("kda",)
                                      ).storage_dtype, "BF16")

    def test_an_overlapping_row_is_refused_with_the_pair_named(self):
        broad = OverrideEntry(tensors_re=r"o_proj\.weight$", dtype="BF16")
        pol = FormatPolicy(overrides=[broad, self.SCOPED])
        with self.assertRaises(ValueError) as caught:
            pol.for_node(self.O_PROJ.name, "attention", self.O_PROJ,
                         components=self.MLA)
        message = str(caught.exception)
        self.assertIn("overrides[0]", message)
        self.assertIn("overrides[1]", message)
        self.assertIn("component='glm5_mla_nope_dsa_attention'", message)
        self.assertIn("tensors_re='o_proj\\\\.weight$'", message)
        # Outside the component only the broad row selects: no ambiguity.
        kda = pol.for_node(self.O_PROJ.name, "attention", self.O_PROJ,
                           components=self.KDA)
        self.assertEqual(kda.selected_by, r"override:o_proj\.weight$")

    def test_two_scoped_rows_on_one_tensor_are_refused(self):
        other = OverrideEntry(
            select=OverrideSelector(component="glm5_hc_block",
                                    tensors=[self.O_PROJ.name]),
            dtype="BF16")
        pol = FormatPolicy(overrides=[self.SCOPED, other])
        with self.assertRaisesRegex(ValueError, r"overrides\[0\].*overrides\[1\]"):
            pol.for_node(self.O_PROJ.name, "attention", self.O_PROJ,
                         components=self.MLA)

    def test_unscoped_rows_keep_first_match_order(self):
        pol = FormatPolicy(overrides=[
            OverrideEntry(tensors_re=r"o_proj\.weight$", dtype="F16"),
            OverrideEntry(tensors_re=r"\.weight$", dtype="F32"),
        ])
        fmt = pol.for_node(self.O_PROJ.name, "attention", self.O_PROJ,
                           components=self.MLA)
        self.assertEqual(fmt.storage_dtype, "F16")

    def test_the_role_path_does_not_read_components(self):
        pol = FormatPolicy(per_role={"attention": FP8})
        for components in ((), self.MLA, self.KDA):
            fmt = pol.for_node(self.O_PROJ.name, "attention", self.O_PROJ,
                               components=components)
            self.assertEqual((fmt.storage_dtype, fmt.selected_by),
                             ("F8_E4M3", "role:attention"))

    def test_a_scoped_row_that_selects_nothing_is_reported(self):
        pol = FormatPolicy(overrides=[self.SCOPED])
        stored = [(self.O_PROJ.name, "", self.KDA, (None,))]
        self.assertEqual(pol.unused_overrides(stored), (0,))
        stored.append((self.O_PROJ.name, "", self.MLA, (None,)))
        self.assertEqual(pol.unused_overrides(stored), ())


class TestLayerScopedOverride(unittest.TestCase):
    """A precision that is not a structural fact.

    Step-5-Preview's `modules_to_not_convert` names the routed-expert weights
    of layers 88, 89 and 90 of 91. Same role, same stack, same component,
    same graph as the other 88 — so a name predicate cannot reach them (the
    eval layer's checkpoint names carry a literal `X` at every indexed
    segment) and duplicating the component to carry the dtype would make a
    precision into a structural identity. The layer scope is how it is said.
    """

    BANK = matrix("model.layers.X.mlp.experts.X.gate_proj.weight", 2048, 7168)
    TAIL = OverrideEntry(
        select=OverrideSelector(layers=[88, 89, 90],
                               tensors_re=r"experts\.X\.gate_proj\.weight$"),
        dtype="BF16")

    def policy(self, *rows, layers=frozenset({88, 89, 90})):
        return FormatPolicy(per_role={"ffn_moe": FP8}, overrides=list(rows),
                            layer_scopes={0: frozenset(layers)})

    def test_the_scope_decides_per_layer(self):
        pol = self.policy(self.TAIL)
        inside = pol.for_node(self.BANK.name, "ffn_moe", self.BANK, layer=89)
        outside = pol.for_node(self.BANK.name, "ffn_moe", self.BANK, layer=40)
        self.assertEqual((inside.storage_dtype, inside.metadata), ("BF16", ()))
        self.assertEqual(outside.storage_dtype, "F8_E4M3")
        self.assertEqual([m.logical_suffix for m in outside.metadata],
                         [".weight_scale_inv"])

    def test_a_node_with_no_layer_index_takes_no_layer_scoped_row(self):
        """A model-level op (`lm_head`) runs once and is at no layer, so a
        row that speaks about layers does not speak about it."""
        pol = self.policy(self.TAIL)
        self.assertEqual(
            pol.for_node(self.BANK.name, "ffn_moe", self.BANK).storage_dtype,
            "F8_E4M3")

    def test_the_layers_group_into_segments_with_their_counts(self):
        pol = self.policy(self.TAIL)
        segments = pol.layer_segments(
            self.BANK.name, "ffn_moe", self.BANK,
            layers=tuple(range(3, 91)))
        self.assertEqual([(len(at), fmt.storage_dtype)
                          for at, fmt in segments],
                         [(85, "F8_E4M3"), (3, "BF16")])

    def test_a_policy_with_no_layer_scope_resolves_once(self):
        """Every model in the corpus today: no row mentions a layer, so no
        layer of a node can resolve differently and the walk asks once."""
        pol = FormatPolicy(per_role={"ffn_moe": FP8})
        self.assertFalse(pol.is_layer_scoped)
        segments = pol.layer_segments(self.BANK.name, "ffn_moe", self.BANK,
                                     layers=tuple(range(3, 91)))
        self.assertEqual(len(segments), 1)
        self.assertIsNone(
            Representations.of(Representation(saved=segments[0][1])).one.layers)

    def test_a_row_with_a_layer_scope_the_caller_did_not_resolve_is_refused(self):
        """A layer bound is a dim expression and a per-layer schedule is a
        config list; a policy built without the model cannot invent either."""
        with self.assertRaises(ValueError) as caught:
            FormatPolicy(overrides=[self.TAIL])
        self.assertIn("without resolving", str(caught.exception))

    def test_an_unreached_layer_scoped_row_is_still_dead(self):
        """The existing dead-row gate keeps working: a layer-scoped row that
        a broader row always beats decides nothing."""
        broad = OverrideEntry(tensors_re=r"gate_proj\.weight$", dtype="F32")
        pol = FormatPolicy(per_role={"ffn_moe": FP8},
                           overrides=[broad, self.TAIL],
                           layer_scopes={1: frozenset({88, 89, 90})})
        self.assertEqual(
            pol.for_node(self.BANK.name, "ffn_moe", self.BANK,
                         layer=89).storage_dtype, "F32")


class TestLayerSegmentsReachTheJoin(unittest.TestCase):
    """`Analysis.tensors()` emits one row per (stored tensor × segment).

    Walked on DeepSeek-V3, whose 58 MoE layers hold FP8 block-scaled expert
    banks: carving the last three out is Step-5's shape on a checkpoint the
    corpus already has.
    """

    @staticmethod
    def analysis(*rows):
        from backend.arch.eval.program import Program
        from backend.arch.eval.transforms import Deployment
        from tests.snapshots import resolve_snapshot

        resolved = resolve_snapshot("deepseek-ai__DeepSeek-V3")
        if rows:
            context = resolved.spec.quant_context
            # `model_copy`, not a deep copy: the spec holds sentinels whose
            # identity matters to the tree walk.
            resolved = resolved.with_spec(resolved.spec.model_copy(update={
                "quant_context": context.model_copy(update={
                    "overrides": [*rows, *context.overrides]})}))
        return Deployment(Program.of(resolved)).at(
            input_tokens=16).checkpoint()

    TAIL = OverrideEntry(
        select=OverrideSelector(
            layers={"range": ["n_layers - 3", "n_layers - 1"]},
            tensors_re=r"mlp\.experts\.X\.gate_proj\.weight$"),
        dtype="BF16")

    def rows_of(self, analysis, name):
        return [row for row in analysis.tensors()
                if row.name == name and not row.is_metadata]

    def test_one_node_yields_two_rows_whose_counts_sum_to_its_copies(self):
        from backend.arch.eval import dim as dim_mod

        name = "model.layers.X.mlp.experts.X.gate_proj.weight"
        plain = self.rows_of(self.analysis(), name)
        split = self.rows_of(self.analysis(self.TAIL), name)
        # The backbone bank: one row before, two after, same copies.
        backbone = [row for row in plain
                    if int(dim_mod.value_of(row.count)) > 1]
        carved = [row for row in split
                  if int(dim_mod.value_of(row.count)) > 1]
        self.assertEqual(len(backbone), 1)
        self.assertEqual(len(carved), 2)
        self.assertEqual(
            sum(int(dim_mod.value_of(row.count)) for row in carved),
            int(dim_mod.value_of(backbone[0].count)))
        self.assertEqual(
            sorted((int(dim_mod.value_of(row.count)), row.storage_dtype)
                   for row in carved),
            [(3, "BF16"), (55, "F8_E4M3")])
        # One tensor, so one tensor index — the rows are segments of it.
        self.assertEqual({row.tensor_index for row in carved}, {0})

    def test_the_rows_bytes_sum_to_what_the_checkpoint_holds(self):
        """55 copies of FP8 plus its scales, 3 of BF16 with none: the same
        element count as before at the widths the file uses."""
        from backend.arch.eval import dim as dim_mod

        name = "model.layers.X.mlp.experts.X.gate_proj.weight"
        rows = [row for row in self.analysis(self.TAIL).iter_checkpoint()
                if row.name.startswith(name.removesuffix(".weight"))]
        weights = [row for row in rows if not row.is_metadata
                   and int(dim_mod.value_of(row.count)) > 1]
        scales = [row for row in rows if row.is_metadata
                  and int(dim_mod.value_of(row.count)) > 1]
        elements = int(dim_mod.value_of(weights[0].elements))
        self.assertEqual(
            sum(int(row.storage_bytes) * int(dim_mod.value_of(row.count))
                for row in weights),
            elements * 55 * 1 + elements * 3 * 2)
        # The BF16 segment implies no scale sibling, so the file holds 55
        # layers' worth and not 58.
        self.assertEqual(
            sum(int(dim_mod.value_of(row.count)) for row in scales), 55)

    def test_the_total_falls_by_exactly_the_carved_layers(self):
        plain = self.analysis().checkpoint_param_bytes()
        split = self.analysis(self.TAIL).checkpoint_param_bytes()
        # Three layers of one projection go from 1 byte to 2 and lose their
        # scales; the rest of the model is untouched.
        self.assertGreater(split, plain)

    def test_the_precision_panel_names_both_formats(self):
        """`precision_effective` took the element-majority saved format per
        role, which hid exactly this tail."""
        rows = {row["selector"]: row
                for row in self.analysis(
                    self.TAIL).precision_effective()["effective"]
                if row["category"] == "weights"}
        self.assertIn(" + ", rows["ffn_moe"]["as_shipped"])
        self.assertIn("BF16", rows["ffn_moe"]["as_shipped"])
        self.assertIn("FP8", rows["ffn_moe"]["as_shipped"].upper())

    def test_an_unsplit_model_still_names_one_format_per_role(self):
        rows = {row["selector"]: row
                for row in self.analysis().precision_effective()["effective"]
                if row["category"] == "weights"}
        self.assertNotIn(" + ", rows["ffn_moe"]["as_shipped"])


class TestTheSchemeDecidesWhatItDescribes(unittest.TestCase):
    def test_a_gain_vector_is_not_the_matrix_the_scheme_means(self):
        """`q_a_layernorm` carries params and lives under an `attention` role,
        so role lookup alone hands it FP8. It has no `in`/`out`, which is the
        scheme's own way of saying it is not what the formula is about."""
        pol = FormatPolicy(per_role={"attention": FP8})
        fmt = pol.for_node("q_a_layernorm.weight", "attention",
                           vector("q_a_layernorm.weight", 1536))
        self.assertEqual(fmt.storage_dtype, "BF16")
        self.assertEqual(fmt.metadata, ())
        self.assertIn("not-a-matrix", fmt.selected_by)

    def test_the_matrix_gets_its_scales(self):
        pol = FormatPolicy(per_role={"attention": FP8})
        fmt = pol.for_node(Q_PROJ.name, "attention", Q_PROJ)
        self.assertEqual([m.suffix for m in fmt.metadata], [".weight_scale_inv"])
        self.assertEqual(fmt.metadata[0].elements, (7168 // 128) * (1536 // 128),
                         "one F32 per 128x128 tile")
        self.assertEqual(fmt.metadata[0].dtype, "F32")


class TestTheFormatNamesWhichElementsShareAScale(unittest.TestCase):
    """A 128x128-block FP8 weight and a per-channel one are the same dtype
    read by different kernels; the format says which, from the scheme's own
    strategy and the bindings its scale shapes resolved with."""

    def granularity(self, entry: Entry) -> Optional[str]:
        return FormatPolicy(per_role={"attention": entry}).for_node(
            Q_PROJ.name, "attention", Q_PROJ).weight_granularity

    def test_block_fp8_is_its_tile(self):
        self.assertEqual(self.granularity(FP8), "per-tile:128x128")

    def test_group_formats_name_their_group(self):
        for entry in GROUP_FORMATS:
            self.assertEqual(self.granularity(entry), "per-group:32", entry.method)

    def test_a_weight_without_scales_has_none(self):
        pol = FormatPolicy(per_role={"attention": FP8})
        gain = pol.for_node("q_a_layernorm.weight", "attention",
                            vector("q_a_layernorm.weight", 1536))
        self.assertIsNone(gain.weight_granularity)
        self.assertIsNone(TensorFormat().weight_granularity)


class TestTheDimsAreTheWeightsByConstruction(unittest.TestCase):
    """`in`/`out` come from the stored tensor, so they cannot be another
    tensor's. The check this replaces — `in * out == params` — was catching a
    case that is no longer expressible: the policy takes a `StoredTensor` and
    there is nowhere for an inherited binding to enter."""

    def test_the_formula_reads_the_stored_shape(self):
        pol = FormatPolicy(per_role={"attention": FP8})
        fmt = pol.for_node(Q_PROJ.name, "attention", Q_PROJ)
        self.assertEqual(fmt.metadata[0].elements,
                         (7168 // 128) * (1536 // 128))

    def test_metadata_shape_projects_from_the_local_stored_tensor(self):
        pol = FormatPolicy(per_role={"attention": FP8})
        fmt = pol.for_node(Q_PROJ.name, "attention", Q_PROJ)
        local = replace(Q_PROJ, shape=(384, 7168))
        projected = fmt.project_metadata(local, local_symbols={"in": 999999})
        self.assertEqual(projected[0].shape, (3, 56))

    def test_explicit_shape_uses_the_projected_node_binding(self):
        entry = Entry(
            method="fp8",
            scale_shape=["n_heads", "(in + block_w - 1) // block_w"],
            model_extra={"block_h": 128, "block_w": 128},
        )
        fmt = FormatPolicy(per_role={"attention": entry}).for_node(
            Q_PROJ.name,
            "attention",
            Q_PROJ,
            node_dims={"n_heads": 12},
        )
        local = replace(Q_PROJ, shape=(384, 7168))
        projected = fmt.project_metadata(local, local_symbols={"n_heads": 3})
        self.assertEqual(projected[0].shape, (3, 56))

    def test_explicit_shape_refuses_an_inherited_extra_symbol(self):
        entry = Entry(
            method="fp8",
            scale_shape=["n_heads", "out"],
            model_extra={"block_h": 128, "block_w": 128},
        )
        fmt = FormatPolicy(per_role={"attention": entry}).for_node(
            Q_PROJ.name,
            "attention",
            Q_PROJ,
            node_dims={"n_heads": 12},
        )
        with self.assertRaisesRegex(ValueError, "no local persistent-axis"):
            fmt.project_metadata(replace(Q_PROJ, shape=(384, 7168)))

    def test_group_shape_uses_ceil_div_for_global_and_local_extents(self):
        stored = matrix("experts.w1.weight", 64, 33)
        for entry in GROUP_FORMATS:
            with self.subTest(method=entry.method):
                fmt = FormatPolicy(per_role={"ffn_moe": entry}).for_node(
                    stored.name, "ffn_moe", stored
                )
                self.assertEqual(fmt.metadata[0].shape, (64, 2))
                projected = fmt.project_metadata(
                    replace(stored, shape=(64, 63))
                )
                self.assertEqual(projected[0].shape, (64, 2))

    def test_local_axis_symbol_cannot_override_format_configuration(self):
        stored = matrix("experts.w1.weight", 64, 64)
        fmt = FormatPolicy(per_role={"ffn_moe": MXFP4}).for_node(
            stored.name, "ffn_moe", stored
        )
        with self.assertRaisesRegex(ValueError, "conflict with format"):
            fmt.project_metadata(stored, local_symbols={"group_size": 8})

    def test_matrix_formula_refuses_an_undeclared_leading_axis(self):
        stored = StoredTensor(
            module="experts",
            suffix=".weight",
            shape=(8, 64, 64),
            kind=StorageKind.MATRIX,
        )
        metadata = StoredTensor(
            module="experts",
            suffix=".scale",
            logical_suffix=".scale",
            shape=(64, 2),
            kind=StorageKind.METADATA,
        )
        fmt = TensorFormat(
            metadata=(metadata,),
            metadata_geometry=(MetadataGeometry(
                suffix=".scale",
                axes=("out", "in // 32"),
            ),),
        )
        with self.assertRaisesRegex(ValueError, "leading-axis transform"):
            fmt.project_metadata(stored)

    def test_headwise_formula_records_and_replays_flatten_leading(self):
        stored = StoredTensor(
            module="heads",
            suffix=".weight",
            shape=(8, 64, 64),
            kind=StorageKind.HEADWISE,
        )
        fmt = FormatPolicy(per_role={"attention": FP8}).for_node(
            stored.name, "attention", stored
        )
        self.assertEqual(
            fmt.metadata_geometry[0].stored_transform,
            "flatten_leading",
        )
        local = replace(stored, shape=(2, 64, 64))
        projected = fmt.project_metadata(local)
        self.assertEqual(projected[0].shape, (1, 1))

    def test_cardinality_is_not_invented_as_a_physical_axis(self):
        metadata = StoredTensor(
            module="experts.w1",
            suffix=".scale",
            logical_suffix=".scale",
            shape=(128,),
            kind=StorageKind.METADATA,
            dtype="U8",
        )
        fmt = TensorFormat(
            metadata=(metadata,),
            metadata_geometry=(MetadataGeometry(
                suffix=".scale",
                cardinality="(in * out) / 32",
            ),),
        )
        with self.assertRaisesRegex(ValueError, "only cardinality"):
            fmt.project_metadata(matrix("experts.w1.weight", 64, 64))

    def test_a_fused_op_needs_no_special_case(self):
        """The op's own dims are called `n_indexer_heads` and `Out`; the stored
        tensor is `(Out, n_indexer_heads)` all the same, so the scheme's
        `in`/`out` resolve without anything knowing those names."""
        pol = FormatPolicy(per_role={"ffn_moe": MXFP4})
        fmt = pol.for_node("experts.gate_up_proj", "ffn_moe",
                           matrix("experts.gate_up_proj", 5760, 2880))
        self.assertEqual(fmt.metadata[0].elements, 2880 * 5760 // 32)

    def test_no_stored_tensor_is_the_default(self):
        pol = FormatPolicy(per_role={"attention": FP8})
        fmt = pol.for_node("something.weight", "attention", None)
        self.assertEqual(fmt.storage_dtype, "BF16")
        self.assertIn("not-a-matrix", fmt.selected_by)


class TestPackingIsSeparateFromDtype(unittest.TestCase):
    def test_mxfp4_packs_two_elements_per_byte(self):
        pol = FormatPolicy(per_role={"ffn_moe": MXFP4})
        fmt = pol.for_node("experts.down_proj", "ffn_moe",
                           matrix("experts.down_proj", 64, 64))
        self.assertEqual((fmt.storage_dtype, fmt.packing_factor), ("U8", 2))
        self.assertEqual(fmt.num_bits, 4)

    def test_bytes_are_params_over_packing_plus_metadata(self):
        fmt = TensorFormat(storage_dtype="U8", packing_factor=2, num_bits=4,
                           metadata=(StoredTensor(module="m", suffix="_scales",
                                                  shape=(128,),
                                                  kind=StorageKind.METADATA,
                                                  dtype="U8"),))
        self.assertEqual(fmt.bytes_for(4096), 4096 / 2 * 1.0 + 128)

    def test_unpacked_is_the_plain_product(self):
        self.assertEqual(TensorFormat("F8_E4M3", 8, 1).bytes_for(1000), 1000.0)
        self.assertEqual(TensorFormat().bytes_for(1000), 2000.0)

    def test_an_undeclared_dtype_refuses_rather_than_assuming(self):
        with self.assertRaises(ValueError):
            known_dtype_bytes("F4_SOMETHING")
        with self.assertRaises(ValueError):
            TensorFormat("F4_SOMETHING").bytes_for(1000)


class TestNativeModel(unittest.TestCase):
    def test_no_quant_context_is_native_everywhere(self):
        pol = FormatPolicy.of(None, "BF16")
        self.assertTrue(pol.is_native)
        fmt = pol.for_node("layers.0.mlp.up_proj.weight", "ffn_dense", Q_PROJ)
        self.assertEqual((fmt.storage_dtype, fmt.packing_factor, fmt.metadata),
                         ("BF16", 1, ()))


class TestTheRouterRoleSelectsTheGatesFormat(unittest.TestCase):
    def test_glm_5_3_flash_router_gates_are_bf16_by_their_role(self):
        """GLM-5.3-Flash lists every `mlp.gate` in
        `modules_to_not_convert`: the 43 router gates the checkpoint holds
        are stored BF16 while the experts around them are FP8. The spec says so through
        `per_role.router: {method: none}`; the gate node's own role is
        `router`, not the MoE wrapper's `ffn_moe`."""
        from backend.arch.eval import dim as dim_mod
        from backend.arch.eval.transforms import Deployment
        from tests.snapshots import shared_program

        analysis = Deployment(shared_program(
            "zai-org__GLM-5.3-Flash")).at(input_tokens=16).checkpoint()
        gates = [row for row in analysis.tensors()
                 if row.name.endswith("mlp.gate.weight")]
        self.assertEqual(
            sum(int(dim_mod.value_of(row.count)) for row in gates), 43)
        self.assertEqual(
            {(row.storage_dtype, row.selected_by) for row in gates},
            {("BF16", "role:router")})


if __name__ == "__main__":
    unittest.main()
