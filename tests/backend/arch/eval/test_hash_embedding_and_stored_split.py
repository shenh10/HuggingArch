"""A hashed multi-row lookup, and one logical table stored as several tensors.

Qwen3.8-Flash-Next's per-layer n-gram embedding hashes each token's context
into 16 ids of ONE `nn.Embedding(320001536, 160)` and gathers the 16 rows
(modeling_qwen4_exp.py:1080-1180); the checkpoint stores that table as 128
row blocks `ngram_embedding.shard_{i}.weight` that transformers concatenates
along dim 0 at load (conversion_mapping.py:1823-1832). `hash_embedding` is
the lookup; `stored_split` is the file layout.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment
from backend.arch.layout.materializer import materialize_comm_events
from backend.arch.layout.physical_cost import physical_traffic
from backend.arch.loader import SpecLoader
from backend.arch.schema.graph import GraphNode

BF16 = 2
HEADS, HEAD_DIM, ROWS, TABLE, PARTS = 4, 8, 4, 4096, 8
WIDTH = HEADS * HEAD_DIM


def _spec(weight_axes: str, split: str = "{ axis: 0, parts: \"8\", segment: \"shard_{i}\" }",
          op: str = "hash_embedding", dims: str = 'n_vocab: n_ple_vocab, d_model: "n_heads * d_head", rows: n_ngram_heads',
          after: str = ""):
    return f"""\
match: {{ model_type: [hashed], model_id: [hashed] }}
dims:
  n_vocab: config.vocab_size
  n_heads: config.num_attention_heads
  d_head: config.head_dim
  d_model: config.hidden_size
  n_ple_vocab: "4096"
  n_ngram_heads: "4"
blocks:
  blk:
    graph:
      n: {{ op: rmsnorm, in: x, dim: d_model, weight_axes: [{{ replicated: true }}] }}
    outputs: [n]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
  - {{ name: ngram, op: {op}, inputs: [input_ids], dims: {{ {dims} }},
      weight_axes: {weight_axes}, stored_split: {split} }}
{after}  - {{ name: layers, template: blk, repeat: "1", inputs: [embed_tokens],
      streams: {{ hidden: {{ carries: true }} }} }}
  - {{ name: norm, op: rmsnorm, dim: d_model, inputs: [layers], weight_axes: [{{ replicated: true }}] }}
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [norm],
      dims: {{ in: d_model, out: n_vocab }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


REPLICATED = "[{ replicated: true }, { replicated: true }]"
WIDTH_SPLIT = "[{ replicated: true }, { world: attention, parts: n_heads, fit: exact }]"


def _program(spec: str) -> Program:
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / "hashed.yaml").write_text(spec)
        resolved = SpecLoader(base_dir=str(base)).match(
            {"model_type": "hashed", "vocab_size": 1000,
             "num_attention_heads": HEADS, "head_dim": HEAD_DIM,
             "hidden_size": WIDTH, "torch_dtype": "bfloat16"},
            model_id="hashed")
    return Program.of(resolved)


def _ngram(program: Program):
    (node,) = [n for n in program.nodes() if n.name == "ngram"]
    return node


def test_hash_embedding_gathers_rows_per_token_from_one_table():
    """[B, S, rows, d] out of one [R, d] table; a step reads B·S·rows rows."""
    program = _program(_spec(REPLICATED))
    batch, tokens = 2, 16
    analysis = Deployment(program).at(batch=batch, input_tokens=tokens).analyze()
    node = _ngram(program)
    assert [int(x) for x in analysis.shapes.values[node.uid].out] == [
        batch, tokens, ROWS, WIDTH]
    assert int(analysis.cost.values[node.uid].params) == TABLE * WIDTH
    assert int(analysis.cost.values[node.uid].flops) == 0
    read = float(physical_traffic(analysis).values[node.uid].weight_reads)
    assert read == batch * tokens * ROWS * WIDTH * BF16


def test_width_split_hash_table_reads_rows_at_the_rank_width_exactly():
    """Cut along the row width (Qwen4Exp's `colwise_gather_output`), every
    rank gathers every row at its own width: B·S·rows·(d/tp)·w."""
    program = _program(_spec(WIDTH_SPLIT))
    batch, tokens = 2, 16
    for tp in (1, 2, 4):
        analysis = Deployment(program).at(
            batch=batch, input_tokens=tokens).on(tp=tp).analyze()
        read = float(physical_traffic(analysis).values[_ngram(program).uid].weight_reads)
        assert read == batch * tokens * ROWS * (WIDTH // tp) * BF16, tp


FLATTEN = ('  - { name: ngram_flat, op: view_merge, inputs: [ngram], '
           'dims: { n_heads: n_ngram_heads, d_head: d_model } }\n')


def test_a_width_split_table_is_gathered_before_its_rows_are_flattened():
    """Flattening rows × width with the width cut would leave each rank
    strided intervals of the merged axis, so the width is gathered first —
    the all-gather Qwen4Exp's `colwise_gather_output` runs — and priced:
    each rank receives (tp-1)/tp of the B·S·rows·d table rows. A replicated
    table flattens with no collective."""
    batch, tokens = 2, 16
    for axes, tps in ((WIDTH_SPLIT, (2, 4)), (REPLICATED, (2,))):
        program = _program(_spec(axes, after=FLATTEN))
        (flat,) = [n for n in program.nodes() if n.name == "ngram_flat"]
        for tp in tps:
            analysis = Deployment(program).at(
                batch=batch, input_tokens=tokens).on(tp=tp).analyze()
            assert [int(x) for x in analysis.shapes.values[flat.uid].out] == [
                batch, tokens, ROWS * WIDTH]
            gathers = [
                (e.op, e.group_size, e.bytes_per_rank)
                for e in materialize_comm_events(analysis)
                if flat.uid in (e.placement_uid, e.attribution_uid)
            ]
            whole = batch * tokens * ROWS * WIDTH * BF16
            expected = ([("all_gather", tp, whole * (tp - 1) / tp)]
                        if axes == WIDTH_SPLIT else [])
            assert gathers == expected, (axes, tp)


def test_a_split_table_is_one_tensor_at_runtime_and_parts_in_the_file():
    """`tensors()` — what a deployment holds — keeps the one table;
    `checkpoint_tensors()` — what the file holds — has `parts` row blocks
    named `<module>.shard_{i}`, which together are the table."""
    program = _program(_spec(REPLICATED))
    analysis = Deployment(program).at(batch=1, input_tokens=8).analyze()
    held = [r for r in analysis.tensors() if r.name.startswith("ngram")]
    filed = [r for r in analysis.checkpoint_tensors() if r.name.startswith("ngram")]
    assert [(r.name, tuple(int(x) for x in r.shape)) for r in held] == [
        ("ngram.weight", (TABLE, WIDTH))]
    assert [r.name for r in filed] == [f"ngram.shard_{i}.weight" for i in range(PARTS)]
    assert {tuple(int(x) for x in r.shape) for r in filed} == {(TABLE // PARTS, WIDTH)}
    assert sum(int(r.storage_bytes) for r in filed) == int(held[0].storage_bytes)


@pytest.mark.parametrize("split, message", [
    ('{ axis: 0, parts: "1", segment: "shard_{i}" }', "declares no `stored_split`"),
    ('{ axis: 0, parts: "8", segment: "shard" }', "exactly once"),
])
def test_stored_split_grammar_names_the_fix(split, message):
    import yaml
    with pytest.raises(ValueError, match=message):
        GraphNode.model_validate({"op": "embedding", "stored_split": yaml.safe_load(split)})


def test_parts_that_do_not_divide_the_axis_are_refused_with_the_fix():
    with pytest.raises(ValueError, match="does not divide weight axis 0"):
        _program(_spec(REPLICATED, split='{ axis: 0, parts: "6", segment: "shard_{i}" }'))


def test_a_weight_with_a_sibling_cannot_be_split():
    """How a bias would be cut is not declared, so a split weight stands
    alone."""
    spec = _spec(REPLICATED, op="linear",
                 dims='in: d_model, out: "4096", bias: "1"').replace(
        "inputs: [input_ids], dims:", "inputs: [embed_tokens], dims:")
    program = _program(spec)
    analysis = Deployment(program).at(batch=1, input_tokens=8).analyze()
    with pytest.raises(ValueError, match="stores sibling tensors"):
        list(analysis.checkpoint_tensors())


def test_stored_split_does_not_combine_with_an_expert_weight():
    with pytest.raises(ValueError, match="declare one or the other"):
        GraphNode.model_validate({
            "op": "linear", "shape": ["n_expert_copies", "out", "in"],
            "stored_split": {"axis": 0, "parts": "2", "segment": "shard_{i}"},
            "expert_weight": "unrolled"})


def test_a_split_reordered_weight_is_cut_on_its_logical_axis_then_laid_out_in_file_order():
    """`stored_split.axis` is an axis of the op's logical `weight_shape`;
    `stored_axes` then orders each block's payload as the file stores it.
    A logical (out=8, in=d_model) linear stored transposed (`stored_axes:
    [1, 0]`) and cut in two along `out` is two file tensors of shape
    (in, out/2) = (d_model, 4), not (4, d_model)."""
    spec = _spec("[{ replicated: true }, { replicated: true }]", op="linear",
                 dims='in: d_model, out: "8"',
                 split='{ axis: 0, parts: "2", segment: "part_{i}" }').replace(
        "inputs: [input_ids], dims:",
        "inputs: [embed_tokens], stored_axes: [1, 0], dims:")
    analysis = Deployment(_program(spec)).at(batch=1, input_tokens=8).analyze()
    held = [r for r in analysis.tensors() if r.name.startswith("ngram")]
    filed = [r for r in analysis.checkpoint_tensors() if r.name.startswith("ngram")]
    assert [(r.name, tuple(int(x) for x in r.shape)) for r in held] == [
        ("ngram.weight", (8, WIDTH))]
    assert [(r.name, tuple(int(x) for x in r.shape)) for r in filed] == [
        ("ngram.part_0.weight", (WIDTH, 4)), ("ngram.part_1.weight", (WIDTH, 4))]
    assert sum(int(r.storage_bytes) for r in filed) == int(held[0].storage_bytes) == 8 * WIDTH * BF16


def test_a_token_activates_the_rows_it_gathers_not_the_table():
    """Activated parameters are the ones a token touches: a lookup reads
    `rows` rows of its table per token (`Evaluator._active_params`), the law
    traffic bills the read by. The model's input embedding is counted whole,
    as activated counts are reported."""
    program = _program(_spec(REPLICATED))
    analysis = Deployment(program).at(batch=1, input_tokens=8).analyze()
    node = _ngram(program)
    cost = analysis.cost.values[node.uid]
    assert int(cost.params) == TABLE * WIDTH
    assert int(cost.active_params) == ROWS * WIDTH
    (embed,) = [n for n in program.nodes() if n.name == "embed_tokens"]
    embed_cost = analysis.cost.values[embed.uid]
    assert int(embed_cost.active_params) == int(embed_cost.params)
