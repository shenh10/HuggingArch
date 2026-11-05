"""Author-facing spec fields reject ambiguity and silent typos."""

import pytest
from pydantic import ValidationError

from backend.arch.document import (MatchSpec, PerLayerAssign,
                                        PerLayerBlock, SpecDocument)
from backend.arch.spec_parsing import parse_document
from backend.arch.schema import (ArchSpec, BlockGraph, CustomOpSpec,
                                      GraphNode, ModelLayer, QuantContext,
                                      WeightAxisSpec)


def _block() -> BlockGraph:
    return BlockGraph(
        graph={"norm": GraphNode(op="rmsnorm", inputs=["x"], dim="d_model")},
        outputs=["norm"],
    )


def test_match_is_closed_and_nonempty_when_present():
    # An empty model_type list is still a mistake (omit it / use None).
    with pytest.raises(ValidationError, match="model_type"):
        MatchSpec(model_type=[])
    with pytest.raises(ValidationError, match="extra_forbidden"):
        MatchSpec(model_type=["x"], architecture=["X"])


def test_match_is_optional_for_base_specs():
    # id-keyed matching (contract B): a base spec exists only to be inherited
    # via `base:` and is never matched directly, so it carries no `match:`;
    # an identity file discriminates on `model_id` alone, so `model_type` is
    # optional too.
    assert SpecDocument.model_fields["match"].is_required() is False
    assert MatchSpec.model_fields["model_type"].is_required() is False
    assert MatchSpec(model_type=None).model_type is None


def test_literal_dimension_bindings_are_normalized_to_expressions():
    doc = SpecDocument(
        match={"model_type": ["x"]},
        dims={"n_layers": 12},
        model=[
            ModelLayer(
                name="embed", op="embedding",
                boundary_role="input_embedding",
                dims={"n_vocab": "n_vocab", "d_model": "d_model"},
            ),
            ModelLayer(
                name="head", op="linear", boundary_role="output_head",
                dims={"in": "d_model", "out": "n_vocab"},
            ),
        ],
    )
    assert doc.dims == {"n_layers": "12"}


@pytest.mark.parametrize("selector", [
    {},
    {"range": ["0", "1"], "rest": True},
    {"range": ["0"]},
    {"layers": []},
    {"offset": "1"},
    {"is": "full"},
    {"from": "config.layer_types", "is": "full", "base": 1},
    {"from": "layer_types", "is": "full"},
    {"rest": False},
])
def test_per_layer_assign_has_one_complete_selector(selector):
    with pytest.raises(ValidationError):
        PerLayerAssign(type="full", **selector)


def test_per_layer_map_has_no_dead_rows_and_rest_is_final():
    with pytest.raises(ValidationError, match="unused type"):
        PerLayerBlock(
            assign=[{"type": "full", "rest": True}],
            map={
                "full": {"template": "full_block"},
                "dead": {"template": "dead_block"},
            },
        )
    with pytest.raises(ValidationError, match="final generator"):
        PerLayerBlock(
            assign=[
                {"type": "fallback", "rest": True},
                {"type": "full", "layers": ["0"]},
            ],
            map={
                "fallback": {"template": "fallback_block"},
                "full": {"template": "full_block"},
            },
        )


def test_model_layer_rejects_unknown_and_conflicting_forms():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelLayer(name="x", op="linear", typo_field=1)
    with pytest.raises(ValidationError, match="requires exactly one"):
        ModelLayer(name="x", op="linear", template="block")
    with pytest.raises(ValidationError, match="requires exactly one"):
        ModelLayer(name="x")
    with pytest.raises(ValidationError, match="aux_kind"):
        ModelLayer(
            name="aux", group=True,
            layers=[ModelLayer(name="inner", template="block")],
        )
    with pytest.raises(ValidationError, match="another layer form"):
        ModelLayer(name="x", op="linear", params={"ignored": "value"})
    with pytest.raises(ValidationError, match="another layer form"):
        ModelLayer(
            name="x", template="block", boundary_role="output_head")
    with pytest.raises(ValidationError, match="checkpoint"):
        ModelLayer(
            name="aux", group=True, aux_kind="mtp", repeat="2",
            inputs=["norm"],
            layers=[ModelLayer(name="inner", template="block")],
        )
    with pytest.raises(ValidationError, match="no block index"):
        ModelLayer(
            name="aux", group=True, aux_kind="vision",
            checkpoint={"prefix": "vision", "index": "local"},
            layers=[ModelLayer(name="inner", template="block")],
        )


@pytest.mark.parametrize("field, replacement", [
    ("primary", "nothing"),
    ("rename", "inner_modules"),
    ("hf_inner", "inner_modules"),
    ("hf", "module"),
])
def test_graph_node_retired_naming_names_the_replacement(field, replacement):
    with pytest.raises(ValidationError, match=replacement):
        GraphNode(op="linear", **{field: True})


def test_graph_node_rejects_unknown_fields_and_roles():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GraphNode(op="linear", typo_field=1)
    with pytest.raises(ValidationError, match="role"):
        GraphNode(op="linear", role="ffn")
    with pytest.raises(ValidationError, match="only valid with `repeat`"):
        GraphNode(op="linear", active="n_active")


def test_explicit_default_block_input_is_accepted():
    doc = parse_document({
        "match": {"model_type": ["x"]},
        "blocks": {
            "block": {
                "inputs": ["x"],
                "graph": {"norm": {"op": "rmsnorm", "in": "x", "dim": "d_model"}},
                "outputs": ["norm"],
            },
        },
        "model": [
            {"name": "embed", "op": "embedding",
             "boundary_role": "input_embedding",
             "dims": {"n_vocab": "n_vocab", "d_model": "d_model"}},
            {"name": "layers", "template": "block"},
            {"name": "head", "op": "linear",
             "boundary_role": "output_head",
             "dims": {"in": "d_model", "out": "n_vocab"}},
        ],
    }, "inline.yaml")
    assert doc.blocks["block"].inputs == ["x"]


def test_tied_weight_target_must_be_an_earlier_layer():
    with pytest.raises(ValidationError, match="not an earlier"):
        ArchSpec(
            name="x", dims={"d_model": "1"}, blocks={"block": _block()},
            model=[
                ModelLayer(
                    name="lm_head", op="linear",
                    boundary_role="output_head",
                    dims={"in": "d_model", "out": "n_vocab"},
                    tied_weight={"to": "embed_tokens"},
                ),
                ModelLayer(
                    name="embed_tokens", op="embedding",
                    boundary_role="input_embedding",
                    dims={"n_vocab": "n_vocab", "d_model": "d_model"},
                ),
            ],
        )


def test_tied_weight_target_must_have_the_same_weight_geometry():
    with pytest.raises(ValidationError, match="cannot tie"):
        ArchSpec(
            name="x", dims={"d_model": "1"}, blocks={"block": _block()},
            model=[
                ModelLayer(
                    name="embed_tokens", op="embedding",
                    boundary_role="input_embedding",
                    dims={"n_vocab": "n_vocab", "d_model": "d_model"},
                ),
                ModelLayer(
                    name="lm_head", op="linear", boundary_role="output_head",
                    dims={"in": "d_model", "out": "other_vocab"},
                    tied_weight={"to": "embed_tokens"},
                ),
            ],
        )


def test_weight_axis_segment_fields_cannot_leak_onto_parent():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        WeightAxisSpec(
            world="attention", parts="d_model", fit="exact",
            segment="q", extent="d_model",
        )


def test_quant_context_validates_roles_and_scheme_fields_at_load_time():
    with pytest.raises(ValidationError, match="unknown role"):
        QuantContext(per_role={"ffn": {"method": "none"}})
    with pytest.raises(ValidationError, match="block_wd"):
        QuantContext(per_role={
            "attention": {"method": "fp8", "block_h": 128, "block_wd": 128},
        })
    with pytest.raises(ValidationError, match="no quantization method"):
        QuantContext(per_role={
            "attention": {"method": "none", "scale_shape": ["1"]},
        })
    QuantContext(per_role={
        "ffn_dense": {
            "method": "compressed-tensors", "num_bits": 4,
            "strategy": "group", "type": "int", "symmetric": True,
            "format": "pack-quantized", "group_size": 32,
        },
    })
    with pytest.raises(ValidationError, match="valid regular expression"):
        QuantContext(overrides=[{"tensors_re": "[", "dtype": "BF16"}])
    with pytest.raises(ValidationError, match="duplicates selector"):
        QuantContext(overrides=[
            {"tensors": ["x.weight"], "dtype": "BF16"},
            {"tensors": ["x.weight"], "dtype": "F32"},
        ])


def test_block_scale_layout_contains_only_checkpoint_facts():
    with pytest.raises(ValidationError, match="loaders"):
        QuantContext(per_role={"attention": {
            "method": "fp8", "block_h": 128, "block_w": 128,
            "block_scale_layout": {
                "kind": "tp_interleaved", "checkpoint_tp": 4,
                "world": "tp_attention",
                "loaders": {"sglang": "whole_chunks"},
            },
        }})
    with pytest.raises(ValidationError, match="without block_scale_layout"):
        QuantContext(per_role={"attention": {
            "method": "fp8", "block_h": 128, "block_w": 128,
            "scale_shape": ["chunks_per_rank", "1"],
        }})


def test_custom_op_cannot_shadow_a_registered_primitive():
    custom = CustomOpSpec(
        signature={"x": ["B", "s_ctx", "d_model"]},
        couples_on=[], shape_kind="preserve", description="test primitive",
    )
    with pytest.raises(ValidationError, match="shadows registered"):
        ArchSpec(
            name="x", dims={"d_model": "1"}, blocks={"block": _block()},
            model=[], custom_ops={"linear": custom},
        )
