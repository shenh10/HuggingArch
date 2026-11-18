"""Kimi-K3's attention residuals, as the source runs them.

`modeling_kimi_linear.py` (snapshot 9f62e4e9): each layer l enters with the
prefix sum `x` and the history `block_residual` (empty at layer 0,
`:1188-1192`). Before attention, when the history is non-empty, the input is
`_apply_attn_res(x, H)` (`:986-993`); at l % K == 0 the entry `x` joins the
history and the prefix sum restarts (`:995-998`); before the MLP the input is
`_apply_attn_res(p1, H)` over the history as it is then (`:1028-1033`); a
restarted layer's output is the MLP output alone (`:1041-1044`). After the
last layer one more combine reads the whole history (`:1215-1232`). Each
combine scores with its own `*_res_norm` / `*_res_proj` weights
(`:1075-1088`): scoring weights, not a gate.

K = 12 and 93 layers: entries of layers 0, 12, …, 84 are appended, so layer
l's first combine reads ceil(l / 12) of them.
"""
from __future__ import annotations

import math
import os

import pytest

from backend.arch.eval import traffic as traffic_mod
from backend.arch.eval.dim import value_of
from backend.arch.eval.evaluator import Evaluator
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from tests.snapshots import (resolve_snapshot, shared_loader, snapshot_config,
                             snapshot_header)

MODEL = "moonshotai/Kimi-K3"
RES_TENSORS = ("self_attention_res_norm", "self_attention_res_proj",
               "mlp_res_norm", "mlp_res_proj")


@pytest.fixture(scope="module")
def config():
    os.environ["HF_HUB_OFFLINE"] = "1"
    resolve_snapshot(MODEL)
    return snapshot_config(MODEL)


@pytest.fixture(scope="module")
def program(config):
    return Program.of(shared_loader().match(config, model_id=MODEL))


@pytest.fixture(scope="module")
def analysis(program):
    return Deployment(program).at(batch=1, input_tokens=4096).analyze()


def _text(config):
    return config["text_config"]


def _layer_kinds(config):
    """Each layer's kind from the config alone: MLA where
    `full_attn_layers` (1-indexed) says so, a restart at l % K == 0, and
    layer 0, whose history is empty."""
    text = _text(config)
    n, k = text["num_hidden_layers"], text["attn_res_block_size"]
    mla = {i - 1 for i in text["linear_attn_config"]["full_attn_layers"]}
    kinds = {}
    for layer in range(n):
        if layer == 0:
            kinds[layer] = "block_kda_dense_first"
        elif layer % k == 0:
            assert layer not in mla
            kinds[layer] = "block_kda_moe_start"
        else:
            kinds[layer] = "block_mla_moe" if layer in mla else "block_kda_moe"
    return kinds, n, k


def _group(program, layer):
    (block,) = [t for t in program.blocks if layer in t.layer_indices
                and program.stack_of_block(program.blocks.index(t)) == ""]
    return block


def _node(block, name):
    (node,) = [n for n in block.nodes() if n.name == name]
    return node


def test_layers_are_grouped_by_the_history_they_see(config, program):
    kinds, _n, k = _layer_kinds(config)
    expected = {}
    for layer, kind in kinds.items():
        m = math.ceil(layer / k)
        key = (kind, (("block_residual", m),) if layer else ())
        expected.setdefault(key, []).append(layer)
    got = {(t.name, tuple(t.visible.items())): list(t.layer_indices)
           for t in program.blocks if t.name != "kimi_k3_vision_tower"}
    assert got == expected
    assert len(got) == 24
    counts = {}
    for (kind, seen), layers in got.items():
        counts.setdefault(dict(seen).get("block_residual", 0), {})[kind] = (
            len(layers))
    assert counts[0] == {"block_kda_dense_first": 1}
    for m in range(1, 8):
        assert counts[m] == {"block_kda_moe": 8, "block_mla_moe": 3,
                             "block_kda_moe_start": 1}
    assert counts[8] == {"block_kda_moe": 5, "block_mla_moe": 3}


@pytest.mark.parametrize("layer,attn,mlp", [
    (0, None, 1), (1, 1, 1), (11, 1, 1), (12, 1, 2), (13, 2, 2),
    (24, 2, 3), (84, 7, 8), (92, 8, 8)])
def test_each_combine_reads_the_history_of_its_layer(program, analysis,
                                                     layer, attn, mlp):
    block = _group(program, layer)
    shapes = analysis.shapes.values
    names = {n.name for n in block.nodes()}
    if attn is None:
        assert "attn_res" not in names
    else:
        history = shapes[_node(block, "attn_res").uid].ins[1]
        assert int(value_of(history[2])) == attn
    history = shapes[_node(block, "mlp_res").uid].ins[1]
    assert int(value_of(history[2])) == mlp


def test_a_restarted_layer_hands_on_the_mlp_output(config, program):
    kinds, n, k = _layer_kinds(config)
    starts = [layer for layer, kind in kinds.items()
              if kind in ("block_kda_dense_first", "block_kda_moe_start")]
    assert starts == list(range(0, n, k))
    for layer in starts:
        block = _group(program, layer)
        (out,) = block.output_roots()
        # A segment-start block now publishes attention + MLP, not the bare
        # MLP branch; the combine is the block's only output root.
        assert out.name == "mlp_residual"
        assert [x for x in block.nodes() if x.op == "add"
                and x.parent is None] == [out]
        # The MLP combine reads the history with this layer's entry in it.
        combine = _node(block, "mlp_res")
        grown = combine.inputs_obj[1].src
        assert grown.op == ("stack_init" if layer == 0 else "stack_append")
        assert [ref.external_sid for ref in grown.inputs_obj
                if ref.src is None] == (["x"] if layer == 0
                                        else ["block_residual", "x"])


def test_layer_zero_starts_the_history_with_one_copy(program, analysis):
    block = _group(program, 0)
    init = _node(block, "block_residual_init")
    entry = 1 * 4096 * 7168 * 2
    layer = traffic_mod.derive(analysis)
    assert (layer.values[init.uid].act_reads,
            layer.values[init.uid].act_writes) == (entry, entry)
    assert [int(value_of(x)) for x in analysis.shapes.values[init.uid].out] == [
        1, 4096, 1, 7168]


def test_the_output_combine_reads_the_whole_history(config, program,
                                                    analysis):
    text = _text(config)
    out = next(n for n in program.ir.top_level
               if n.name == "language_model.model.output_attn_res")
    history = analysis.shapes.values[out.uid].ins[1]
    assert int(value_of(history[2])) == math.ceil(
        text["num_hidden_layers"] / text["attn_res_block_size"]) == 8
    norm = next(n for n in program.ir.top_level
                if n.name == "language_model.model.norm")
    assert [ref.src.name for ref in norm.inputs_obj] == [
        "language_model.model.output_attn_res"]


def test_every_attn_res_tensor_is_bound_to_its_parameter_node(config,
                                                              program):
    """The 374 AttnRes tensors (93 layers x 4 + 2 at the model level), one
    by one: the layer's group holds exactly one parameter node of that name
    and shape, and the matcher's key for it is the checkpoint's."""
    header = snapshot_header(MODEL)
    n = _text(config)["num_hidden_layers"]
    names = [f"language_model.model.layers.{layer}.{t}.weight"
             for layer in range(n) for t in RES_TENSORS]
    names += [f"language_model.model.output_attn_res_{t}.weight"
              for t in ("norm", "proj")]
    assert len(names) == 374
    top = {node.name: node for node in program.ir.top_level}
    for name in names:
        (stored_at,) = header.select(name).parts
        shape = list(stored_at[0].shape)
        parts = name.split(".")
        if parts[2] == "layers":
            node = _node(_group(program, int(parts[3])), ".".join(parts[4:]))
            stored = program.checkpoint_prefix(node) + node.name
        else:
            node = top[name]
            stored = node.name
        assert node.op == ("parameter_vector" if len(shape) == 1
                           else "parameter_tensor"), name
        extent = (list(node.shape) if node.shape
                  else [node.dim])
        assert extent == shape, name
        assert stored_at[0] in [g for g, _n in header.select(stored).parts], (
            stored, name)


def test_the_weights_are_bound_and_the_gate_passes():
    from backend.arch.validate import validate
    report = validate(MODEL)
    assert report["valid"] is True
    assert report["model_storage_bytes"]["diff"] == 0
    assert (report.get("unmatched_hf_tensors") or []) == []
    assert (report.get("param_mismatches") or []) == []


def test_every_group_hands_on_the_state_it_entered_with(program, monkeypatch):
    """Each of the 24 groups is a link of the carried chain; under tensor
    parallelism every one exits with the layout it entered with."""
    links = []
    original = Evaluator._eval_link

    def recording(self, stack, stage, carry, inbounds, scope, walk, **kw):
        entry = walk.carried.get(carry.key, inbounds[0])
        out = original(self, stack, stage, carry, inbounds, scope, walk,
                       **kw)
        if stack.name == "":
            links.append((stage.template.layer_indices, entry.state,
                          out.state))
        return out

    monkeypatch.setattr(Evaluator, "_eval_link", recording)
    Deployment(program).at(batch=1, input_tokens=4096).on(tp=8).analyze()
    assert len(links) == 24
    assert sorted(i for indices, _, _ in links for i in indices) == list(
        range(93))
    for indices, entry, exit_ in links:
        assert entry is not None
        assert entry == exit_, indices


def test_the_deepest_history_is_live_through_attention(config, monkeypatch):
    """At the deepest groups the history holds 8 entries; it is read before
    the MLP, so it is live across attention — the attention half's peak of
    the deepest block holds all 8. The whole block's peak is the routed
    bank's, where the history is already dead: `permute` writes every
    token-expert row at once (`[n_expert_copies, M, d]`, transformers'
    `grouped_mm` experts forward, integrations/moe.py:394), so every MoE
    block peaks at the same bytes."""
    from backend.inference import activation_peak
    from backend.inference.step_eval import estimate_peak_activation
    from backend.inference.framework_bindings import get_binding
    seen = []
    original = activation_peak._peak_for_leaves

    def recording(records, external_bytes):
        seen.append((records, dict(external_bytes)))
        return original(records, external_bytes)

    monkeypatch.setattr(activation_peak, "_peak_for_leaves", recording)
    resolved = shared_loader().match(config, model_id=MODEL)
    peak = estimate_peak_activation(
        resolved, batch=1, seq_len=4096, framework_binding=get_binding("vllm"))
    assert peak["worst_block"].startswith("block_mla_moe")
    assert "|block_sparse_moe." in peak["peak_node_id"]
    history = 8 * 1 * 4096 * 7168 * 2
    deepest = [(records, external) for records, external in seen
               if external.get("block_residual") == history]
    assert deepest
    for records, _external in deepest:
        ids = [record["id"] for record in records]
        last_read = max(i for i, record in enumerate(records)
                        if "block_residual" in record["inputs"])
        attention = next(i for i, name in enumerate(ids)
                         if name.endswith("|self_attn.o_proj"))
        moe = min(i for i, name in enumerate(ids)
                  if "|block_sparse_moe." in name)
        # Live from the block's entry to a read after attention's output
        # projection, dead before the bank's first step.
        assert attention < last_read < moe


@pytest.mark.parametrize("model_id", [
    "deepseek-ai/DeepSeek-V3",
    "Qwen/Qwen3-30B-A3B",
    "meta-llama/Llama-2-7b-hf",
])
def test_models_without_attn_res_hold_no_history(model_id):
    from backend.inference.step_eval import estimate_peak_activation
    from backend.inference.framework_bindings import get_binding
    os.environ["HF_HUB_OFFLINE"] = "1"
    resolved = shared_loader().match(snapshot_config(model_id),
                                     model_id=model_id)
    peak = estimate_peak_activation(
        resolved, batch=1, seq_len=4096, framework_binding=get_binding("vllm"))
    assert not any("attn_res" in row["id"] or "block_residual" in row["id"]
                   for row in peak["top_live_at_peak"])
