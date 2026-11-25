"""Inner streams as `in:` — CPU tests.

A stack-declared name is a real input: `in: s` resolves like any other
unresolved ref. The producer tensor is the edge. A reader whose producer
has not walked raises; the walk does not seed a yaml shape.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from backend.arch.loader import SpecLoader
from backend.arch.eval.projection import project_model


def _load(spec: str, config: dict):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(spec)
        path = f.name
    loader = SpecLoader()
    try:
        sp = loader.load_model_spec(path)
    finally:
        os.unlink(path)
    return loader.resolve(sp, config)


def _node_shape(block_report, name):
    for nr in block_report.nodes:
        if nr.name == name or str(nr.name).endswith(f".{name}"):
            return getattr(nr, "shape_out", None)
    return None


_PROBE = """
match: { model_type: [_stream_flow] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size }
blocks:
  prod: { stream_out: { s: mk }, graph: { mk: { op: rmsnorm, in: x, dim: d_model,
                                                weight_axes: [{ replicated: true }] } }, outputs: [mk] }
  cons: { reads: [s], graph:
      { u: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] },
        g: { op: mul, in: [u, s], dim: d_model } }, outputs: [g] }
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, range: ["0", "0"] }
        - { type: c, rest: true, reads_from: { s: "0" } }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab },
      inputs: [layers],
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""

_CONFIG = {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 2, "vocab_size": 16,
           "model_type": "_stream_flow"}


def test_stream_reaches_consumer_input():
    """cons's `g: mul(u, s)` consumes the stream: the producer tensor flows
    in, so `g` evaluates to [B, s_query, d_model] rather than falling off."""
    resolved = _load(_PROBE, _CONFIG)
    result = project_model(resolved, batch=1, input_tokens=4)
    blocks = {t: br for t, _c, br in result.block_results}
    g_shape = _node_shape(blocks["cons"], "g")
    assert g_shape is not None, "consumer node missing from cons block"
    assert [int(d) for d in g_shape] == [1, 4, 8]


def test_stream_name_only_still_follows_the_producer():
    """The stack catalog is the name. Eval does not need a yaml `shape`."""
    spec = _PROBE.replace(
        "    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }\n",
        "    streams: { hidden: { carries: true }, s: {} }\n")
    resolved = _load(spec, _CONFIG)
    result = project_model(resolved, batch=1, input_tokens=4)
    blocks = {t: br for t, _c, br in result.block_results}
    g_shape = _node_shape(blocks["cons"], "g")
    assert [int(d) for d in g_shape] == [1, 4, 8]


def test_live_stream_overrides_consumer_variant_dims():
    """A live stream is the producer tensor. The consumer variant rebinds
    `d_stream` to `n_vocab` on the yaml shape; that does not resize the
    tensor the producer already published (`d_model`)."""
    spec = """
match: { model_type: [_stream_flow] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size }
blocks:
  prod: { stream_out: { s: mk }, graph: { mk: { op: rmsnorm, in: x, dim: d_model } }, outputs: [mk] }
  cons: { reads: [s], graph:
      { u: { op: mul, in: [s, s], dim: n_vocab },
        g: { op: mul, in: [u, s], dim: n_vocab } }, outputs: [g] }
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_stream] } }
    per_layer:
      assign:
        - { type: p, range: ["0", "0"], dims: { d_stream: d_model } }
        - { type: c, rest: true, dims: { d_stream: n_vocab }, reads_from: { s: "0" } }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""
    resolved = _load(spec, _CONFIG)
    result = project_model(resolved, batch=1, input_tokens=4)
    blocks = {t: br for t, _c, br in result.block_results}
    g_shape = _node_shape(blocks["cons"], "g")
    assert g_shape is not None
    assert [int(d) for d in g_shape] == [1, 4, 8]


def test_a_first_layer_reader_has_no_layer_to_read():
    """Layer 0 reads `s` and nothing before it writes it: its `reads_from`
    names no earlier layer, and the refusal names the declaration and the
    layers that do write the stream."""
    from backend.arch.spec_tree import StreamReadRefusal
    spec = """
match: { model_type: [_stream_flow] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size }
blocks:
  prod: { stream_out: { s: mk }, graph: { mk: { op: rmsnorm, in: x, dim: d_model } }, outputs: [mk] }
  cons: { reads: [s], graph:
      { u: { op: mul, in: [s, s], dim: n_vocab },
        g: { op: mul, in: [u, s], dim: n_vocab } }, outputs: [g] }
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_stream] } }
    per_layer:
      assign:
        - { type: c, range: ["0", "0"], dims: { d_stream: n_vocab },
            reads_from: { s: "i - 1" } }
        - { type: p, rest: true, dims: { d_stream: d_model } }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""
    resolved = _load(spec, _CONFIG)
    with pytest.raises(StreamReadRefusal,
                       match="from layer -1, which is not a layer before it"
                       ) as refused:
        project_model(resolved, batch=1, input_tokens=4)
    assert refused.value.declaration == "reads_from"
    assert "layers 1 write it" in refused.value.fix


def test_undeclared_stream_is_not_an_edge():
    """A name no stack listed under `streams:` is not a stream. The
    reader does not see the producer's tensor — the catalog is the gate —
    and a read of a name nothing binds is refused, not dropped."""
    spec = """
match: { model_type: [_stream_flow] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size }
blocks:
  prod: { stream_out: { s: wide }, graph:
      { wide: { op: linear, in: x, dims: { in: d_model, out: n_vocab } },
        y: { op: rmsnorm, in: x, dim: d_model } },
    outputs: [y] }
  cons: { reads: [s], graph:
      { u: { op: rmsnorm, in: x, dim: d_model },
        g: { op: mul, in: [u, s], dim: d_model } }, outputs: [g] }
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: p, range: ["0", "0"] }
        - { type: c, rest: true }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""
    resolved = _load(spec, _CONFIG)
    with pytest.raises(ValueError, match="'s', which nothing binds"):
        project_model(resolved, batch=1, input_tokens=4)


def test_stream_survives_later_readers():
    """One producer, several later readers: the stream is not consumed."""
    cfg = dict(_CONFIG)
    cfg["num_hidden_layers"] = 4
    resolved = _load(_PROBE, cfg)
    result = project_model(resolved, batch=1, input_tokens=4)
    blocks = {t: br for t, _c, br in result.block_results}
    g_shape = _node_shape(blocks["cons"], "g")
    assert [int(d) for d in g_shape] == [1, 4, 8]
    counts = {t: c for t, c, _br in result.block_results}
    assert counts["prod"] == 1
    assert counts["cons"] == 3


def test_cached_stream_sets_reader_s_kv():
    """A stream_out cache entry carries Min(s_kv, s_window) to readers.

    The reader does not declare cache.entries — storage stays on the writer.
    """
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_kv] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  prod:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: s, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, range: ["0", "0"] }
        - { type: c, rest: true, reads_from: { s: "0" } }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 3,
                            "vocab_size": 16, "model_type": "_stream_kv"})
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    score_s_kv = None
    for uid, dims in analysis.node_dims.values.items():
        node = analysis.program.by_uid(uid)
        if node is not None and (
                node.name == "score" or str(node.name).endswith(".score")):
            score_s_kv = dims.get("s_kv")
    assert score_s_kv == 4, f"reader s_kv should be the window, got {score_s_kv}"
    counts = {t.meta.template_name: t.count for t in analysis.program.blocks}
    assert counts.get("prod") == 1
    assert counts.get("cons") == 2


def test_nested_component_stream_out_sets_reader_s_kv():
    """stream_out on a nested component (not the block template) still binds."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_kv_nested] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  inner:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  prod:
    graph:
      body: { op: inner, in: x }
    outputs: [body]
  cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: s, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, range: ["0", "0"] }
        - { type: c, rest: true, reads_from: { s: "0" } }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 3,
                            "vocab_size": 16, "model_type": "_stream_kv_nested"})
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    score_s_kv = None
    for uid, dims in analysis.node_dims.values.items():
        node = analysis.program.by_uid(uid)
        if node is not None and (
                node.name == "score" or str(node.name).endswith(".score")):
            score_s_kv = dims.get("s_kv")
    assert score_s_kv == 4, f"nested producer should bind s_kv, got {score_s_kv}"


def _score_s_kv(analysis):
    for uid, dims in analysis.node_dims.values.items():
        node = analysis.program.by_uid(uid)
        if node is not None and (
                node.name == "score" or str(node.name).endswith(".score")):
            return dims.get("s_kv")
    return None


def test_later_group_reads_without_redeclaring_streams():
    """A following stack that only `reads` still sees the earlier group's cache law."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_kv_groups] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_mtp_layers: config.num_nextn_predict_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  prod:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: s, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: enc
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, rest: true }
      map:
        p: { template: prod }
  - name: dec
    inputs: [enc]
    repeat: n_mtp_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: c, rest: true,
            reads_from: { s: { stack: enc, layer: "n_layers - 1" } } }
      map:
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [dec]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16,
                            "num_hidden_layers": 1,
                            "num_nextn_predict_layers": 2,
                            "model_type": "_stream_kv_groups"})
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    assert _score_s_kv(analysis) == 4


def test_declared_stream_bad_shape_does_not_take_live():
    """Eval does not seed from the declaration: a broken yaml shape still
    leaves the reader on the producer tensor."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_kv_bad] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  prod:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: s, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_window +] } }
    per_layer:
      assign:
        - { type: p, range: ["0", "0"] }
        - { type: c, rest: true, reads_from: { s: "0" } }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 3,
                            "vocab_size": 16, "model_type": "_stream_kv_bad"})
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    assert _score_s_kv(analysis) == 4


def test_undeclared_cached_stream_does_not_bind():
    """stream_out without any stack `streams:` must not feed readers."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment
    import pytest

    spec = """
match: { model_type: [_stream_kv_free] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  prod:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: s, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: p, range: ["0", "0"] }
        - { type: c, rest: true }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 3,
                            "vocab_size": 16, "model_type": "_stream_kv_free"})
    with pytest.raises(ValueError, match="'s', which nothing binds"):
        Deployment(Program.of(resolved)).at(batch=1, input_tokens=16).analyze()


def test_wrapper_component_carries_stream_kv_length():
    """A view wrapper around the stream still dictates s_kv to the score: its
    leaf reads the stream as a formal that carries the length, as the same
    view inline would."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_kv_wrap] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  prod:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  wrap_kv:
    graph:
      out: { op: identity, in: x }
    outputs: [out]
  cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      raw: { op: wrap_kv, in: s }
      kh: { op: head_split, in: raw, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, range: ["0", "0"] }
        - { type: c, rest: true, reads_from: { s: "0" } }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 3,
                            "vocab_size": 16, "model_type": "_stream_kv_wrap"})
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    assert _score_s_kv(analysis) == 4


def test_nested_component_reads_live_stream():
    """A later stack that omitted `streams:` still binds what a nested
    component `reads`, even when the wrapper template itself does not."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_kv_nested_reads] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_mtp_layers: config.num_nextn_predict_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  prod:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  inner_cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: s, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
  cons:
    graph:
      body: { op: inner_cons, in: x }
    outputs: [body]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: enc
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, rest: true }
      map:
        p: { template: prod }
  - name: dec
    inputs: [enc]
    repeat: n_mtp_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: c, rest: true,
            reads_from: { s: { stack: enc, layer: "n_layers - 1" } } }
      map:
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [dec]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16,
                            "num_hidden_layers": 1,
                            "num_nextn_predict_layers": 2,
                            "model_type": "_stream_kv_nested_reads"})
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    assert _score_s_kv(analysis) == 4
    prog = Program.of(resolved)
    cons = next(t for t in prog.blocks if t.name == "cons")
    _, reads = prog.component_stream_roles(cons)
    assert reads == ("s",)


def test_param_bound_nested_reads_live_stream():
    """`op: <body>` bound at the call site still seeds the inner `reads`."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_kv_param] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_mtp_layers: config.num_nextn_predict_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  prod:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  inner_cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: s, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
  wrap_cons:
    params: { body: { kind: component } }
    graph:
      b: { op: "<body>", in: x }
    outputs: [b]
  cons:
    graph:
      w: { op: wrap_cons, in: x, params: { body: inner_cons } }
    outputs: [w]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: enc
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, rest: true }
      map:
        p: { template: prod }
  - name: dec
    inputs: [enc]
    repeat: n_mtp_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: c, rest: true,
            reads_from: { s: { stack: enc, layer: "n_layers - 1" } } }
      map:
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [dec]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16,
                            "num_hidden_layers": 1,
                            "num_nextn_predict_layers": 2,
                            "model_type": "_stream_kv_param"})
    prog = Program.of(resolved)
    cons = next(t for t in prog.blocks if t.name == "cons")
    _, reads = prog.component_stream_roles(cons)
    assert reads == ("s",)
    analysis = Deployment(prog).at(batch=1, input_tokens=16).analyze()
    assert _score_s_kv(analysis) == 4


def test_param_default_nested_reads_live_stream():
    """A slot `default:` binds the inner component when the caller omits params."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_kv_param_default] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_mtp_layers: config.num_nextn_predict_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  prod:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  inner_cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: s, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
  wrap_cons:
    params: { body: { kind: component, default: inner_cons } }
    graph:
      b: { op: "<body>", in: x }
    outputs: [b]
  cons:
    graph:
      w: { op: wrap_cons, in: x }
    outputs: [w]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: enc
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, rest: true }
      map:
        p: { template: prod }
  - name: dec
    inputs: [enc]
    repeat: n_mtp_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: c, rest: true,
            reads_from: { s: { stack: enc, layer: "n_layers - 1" } } }
      map:
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [dec]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16,
                            "num_hidden_layers": 1,
                            "num_nextn_predict_layers": 2,
                            "model_type": "_stream_kv_param_default"})
    prog = Program.of(resolved)
    cons = next(t for t in prog.blocks if t.name == "cons")
    _, reads = prog.component_stream_roles(cons)
    assert reads == ("s",)
    analysis = Deployment(prog).at(batch=1, input_tokens=16).analyze()
    assert _score_s_kv(analysis) == 4


def test_wrapper_card_lists_nested_stream_reads():
    """The inspector card for a wrapper with no `reads:` still lists the
    inner component's stream — same closure the evaluator seeds from."""
    from backend.arch.eval.program import Program
    from backend.rendering.builders import _block_stream_roles

    spec = """
match: { model_type: [_stream_kv_nested_reads] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_mtp_layers: config.num_nextn_predict_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  prod:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { s: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  inner_cons:
    reads: [s]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: s, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
  cons:
    graph:
      body: { op: inner_cons, in: x }
    outputs: [body]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: enc
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, rest: true }
      map:
        p: { template: prod }
  - name: dec
    inputs: [enc]
    repeat: n_mtp_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: c, rest: true,
            reads_from: { s: { stack: enc, layer: "n_layers - 1" } } }
      map:
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [dec]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16,
                            "num_hidden_layers": 1,
                            "num_nextn_predict_layers": 2,
                            "model_type": "_stream_kv_nested_reads"})
    prog = Program.of(resolved)
    cons = next(t.meta for t in prog.blocks if t.name == "cons")
    prod = next(t.meta for t in prog.blocks if t.name == "prod")
    _prod, reads = _block_stream_roles(cons, prog)
    assert reads == ["s"]
    _prod, reads = _block_stream_roles(prod, prog)
    assert _prod == ["s"]


def test_renderer_draws_stream_pseudo_node():
    """The consumer block gets a `stream: s` pseudo-node and the cross-layer
    edge into the consuming node, with the seeded shape on it."""
    from backend.rendering.builders import build_block_graph_data
    resolved = _load(_PROBE, _CONFIG)
    data, _unmatched = build_block_graph_data(resolved, batch_size=1,
                                               seq_len=4)
    cons = data["cons"]
    pseudo = [n for n in cons["nodes"] if n["id"] == "s"]
    assert pseudo and pseudo[0]["op"] == "stream_in"
    assert pseudo[0].get("tensor_out") == "[1,4,8]"
    edges = cons["edges"]
    assert any(e.get("src") == "s" for e in edges), \
        f"no edge out of stream pseudo-node: {edges}"


def test_inspector_stream_in_matches_architecture_card_stream_reads():
    """Inspector virtual `stream_in` nodes are the same set the architecture
    card lists as `stream_reads` — both from `_block_stream_roles`."""
    from backend.arch.eval.program import Program
    from backend.rendering.builders import (
        _block_stream_roles, build_block_graph_data,
    )
    resolved = _load(_PROBE, _CONFIG)
    data, _unmatched = build_block_graph_data(resolved, batch_size=1,
                                               seq_len=4)
    prog = Program.of(resolved)
    for tmpl, payload in data.items():
        meta = next(t.meta for t in prog.blocks if t.name == tmpl)
        _, card_reads = _block_stream_roles(meta, prog)
        stream_in = [n["id"] for n in payload["nodes"]
                     if n.get("op") == "stream_in"]
        assert set(stream_in) == set(card_reads), tmpl


def _seq_extent(analysis, stack):
    from backend.arch.eval.dim import value_of
    for uid, shape in analysis.shapes.values.items():
        node = analysis.program.by_uid(uid)
        if node is None or getattr(node, "stack_name", None) != stack:
            continue
        if shape is None or not shape.out or len(shape.out) < 2:
            continue
        return value_of(shape.out[-2])
    return None


def test_view_op_is_just_an_edge():
    """A slice between stacks is an op. Downstream `in:` sees that tensor."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import DECODE, Deployment

    spec = """
match: { model_type: [_stream_act] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_mtp_layers: config.num_nextn_predict_layers,
        n_vocab: config.vocab_size, s_window: "4" }
blocks:
  enc:
    graph:
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  dec:
    graph:
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: enc
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: e, rest: true }
      map:
        e: { template: enc }
  - { name: suffix, op: slice, axis: -2, dims: { out: s_window }, inputs: [enc] }
  - name: dec
    inputs: [suffix]
    repeat: n_mtp_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: d, rest: true }
      map:
        d: { template: dec }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [dec]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16,
                            "num_hidden_layers": 2,
                            "num_nextn_predict_layers": 2,
                            "model_type": "_stream_act"})
    prefill = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    assert _seq_extent(prefill, "enc") == 16
    assert _seq_extent(prefill, "dec") == 4
    decode = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).during(DECODE, cache_len=16).analyze()
    assert _seq_extent(decode, "enc") == 1
    assert _seq_extent(decode, "dec") == 4


def _named_seq_extent(analysis, name):
    from backend.arch.eval.dim import value_of
    for uid, shape in analysis.shapes.values.items():
        node = analysis.program.by_uid(uid)
        if node is None or not (
                node.name == name or str(node.name).endswith(f".{name}")):
            continue
        if shape is None or not shape.out or len(shape.out) < 2:
            continue
        return value_of(shape.out[-2])
    return None


def test_live_stream_shape_is_the_edge():
    """After a producer walks, the reader sees that tensor, not the yaml shape.

    Declaration is `[B, s_query, d_model]` (16). Producer `stream_out`s a
    slice to `s_window` (4). Seeding from the declaration would leave the
    consumer at 16.
    """
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_live_shape] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, s_window: "4" }
blocks:
  prod:
    stream_out: { s: cut }
    graph:
      cut: { op: slice, in: x, axis: -2, dims: { out: s_window } }
    outputs: [cut]
  cons:
    reads: [s]
    graph:
      got: { op: rmsnorm, in: s, dim: d_model }
    outputs: [got]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, s: { shape: [B, s_query, d_model] } }
    per_layer:
      assign:
        - { type: p, range: ["0", "0"] }
        - { type: c, rest: true, reads_from: { s: "0" } }
      map:
        p: { template: prod }
        c: { template: cons }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16,
                            "num_hidden_layers": 2,
                            "model_type": "_stream_live_shape"})
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    assert _named_seq_extent(analysis, "got") == 4


def test_moe_after_slice_follows_the_window():
    """A view shortens the residual; combine's sequence and FLOPs follow it,
    not Binding `s_ctx`."""
    from backend.arch.eval.dim import value_of
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_moe_window] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_mtp_layers: config.num_nextn_predict_layers,
        n_vocab: config.vocab_size, s_window: "4",
        n_experts: "4", n_experts_active: "1" }
blocks:
  enc:
    graph:
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  dec:
    graph:
      gate: { op: linear, in: x, dims: { in: d_model, out: n_experts } }
      permute: { op: permute, in: [x, gate] }
      experts: { op: linear, in: permute, dims: { in: d_model, out: d_model },
                 shape: [n_expert_copies, d_model, d_model] }
      combine: { op: combine, in: [experts, gate] }
    outputs: [combine]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: enc
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: e, rest: true }
      map:
        e: { template: enc }
  - { name: suffix, op: slice, axis: -2, dims: { out: s_window }, inputs: [enc] }
  - name: dec
    inputs: [suffix]
    repeat: n_mtp_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: d, rest: true }
      map:
        d: { template: dec }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [dec]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16,
                            "num_hidden_layers": 1,
                            "num_nextn_predict_layers": 1,
                            "model_type": "_moe_window"})
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    seq = None
    flops = None
    permute_rows = None
    permute_mean = None
    for uid, shape in analysis.shapes.values.items():
        node = analysis.program.by_uid(uid)
        if node is None:
            continue
        if node.op == "permute" and shape is not None and shape.out:
            permute_rows = value_of(shape.out[0])
            permute_mean = value_of(
                analysis.node_dims.values[node.uid]["M"])
        if node.op != "combine":
            continue
        if shape is not None and shape.out and len(shape.out) >= 2:
            seq = value_of(shape.out[-2])
        cost = analysis.cost.values.get(uid)
        if cost is not None:
            flops = value_of(cost.flops)
    # The dispatch declares the rows it WRITES — `R = M x n_experts`, the
    # whole routed buffer — and the per-expert mean it hands its zone is on
    # the node's own binding. Both follow the window: `M = 1·4·1/4 = 1` and
    # `R = 1 x 4 experts = 4`, where the un-sliced sequence would make them
    # 4 and 16.
    assert permute_mean == 1, (
        f"the zone's mean should follow the window, got {permute_mean}")
    assert permute_rows == 4, (
        f"the buffer should be 4 experts of that mean, got {permute_rows}")
    assert seq == 4, f"combine sequence should follow the slice, got {seq}"
    # B=1, S=4, n_experts_active=1, d_model=8
    assert flops == 32, f"combine FLOPs should follow the window, got {flops}"


def _moe_toy(dims: str, token: str) -> str:
    """One MoE layer whose permute reads `token`'s output, with the dims
    table `dims` — the two things the routed-rows source depends on."""
    return f"""
match: {{ model_type: [_moe_toy] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, {dims} }}
blocks:
  moe:
    graph:
      gate: {{ op: linear, in: x, dims: {{ in: d_model, out: n_experts }} }}
      token: {token}
      permute: {{ op: permute, in: [token, gate] }}
      experts: {{ op: linear, in: permute, dims: {{ in: d_model, out: d_model }},
                 shape: [n_expert_copies, d_model, d_model] }}
      combine: {{ op: combine, in: [experts, gate] }}
    outputs: [combine]
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: {{ n_vocab: n_vocab, d_model: d_model }} }}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }} }}
    per_layer:
      assign:
        - {{ type: m, rest: true }}
      map:
        m: {{ template: moe }}
  - {{ name: head, op: linear, boundary_role: output_head,
      dims: {{ in: d_model, out: n_vocab }} , inputs: [layers]}}
"""


_ROUTING = 'n_experts: "4", n_experts_active: "1"'
_TOY_CONFIG = {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16, "num_hidden_layers": 1,
               "model_type": "_moe_toy"}


def _moe_toy_analysis(dims: str, token: str):
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    resolved = _load(_moe_toy(dims, token), _TOY_CONFIG)
    return Deployment(Program.of(resolved)).at(
        batch=2, input_tokens=4).analyze()


def test_permute_m_is_the_arriving_tokens_per_expert():
    """`M = B·s_ctx·n_experts_active/n_route_targets` = 2·4·1/4 = 2, the
    mean one expert gets, read off the token tensor permute receives; the
    buffer permute writes is `[n_expert_copies, M, d_model]`, every expert's
    M rows, its leading axis the copies axis by identity."""
    from backend.arch.eval.dim import value_of

    analysis = _moe_toy_analysis(
        _ROUTING, "{ op: rmsnorm, in: x, dim: d_model }")
    permute = next(n for n in analysis.program.nodes() if n.op == "permute")
    dims = analysis.node_dims.values[permute.uid]
    assert value_of(dims["M"]) == 2
    out = analysis.shapes.values[permute.uid].out
    assert [value_of(extent) for extent in out[:2]] == [4, 2]
    assert out[0].is_("n_expert_copies")
    assert out[1].expr == dims["M"].expr


def test_permute_refuses_a_token_tensor_that_is_not_b_s_d():
    """A head-split residual `(B, s_ctx, h, d)` has no `(B, s_ctx)` prefix
    the routed rows could be counted from; the walk refuses it rather than
    counting the workload's tokens instead."""
    with pytest.raises(ValueError, match="permute"):
        _moe_toy_analysis(
            _ROUTING + ', n_heads: "2", d_head: "4"',
            "{ op: view_split, in: x, axis: -1, "
            "dims: { n_heads: n_heads, d_head: d_head } }")


def test_permute_refuses_routing_counts_that_do_not_resolve():
    """Without `n_experts_active` there is no tokens-per-expert to bind,
    and a permute that bound no `M` is not walked on."""
    with pytest.raises(ValueError, match="n_experts_active"):
        _moe_toy_analysis(
            'n_experts: "4"',
            "{ op: rmsnorm, in: x, dim: d_model }")


def test_window_view_lives_in_decode_variant():
    """Prefill shortens the residual with a slice; decode swaps that
    component for identity, so the decoder residual is 1."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import DECODE, Deployment

    spec = """
match: { model_type: [_window_var] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_mtp_layers: config.num_nextn_predict_layers,
        n_vocab: config.vocab_size, s_window: "4" }
blocks:
  enc:
    graph:
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
  window_prefill:
    decode_variant: window_decode
    graph:
      cut: { op: slice, in: x, axis: -2, dims: { out: s_window } }
    outputs: [cut]
  window_decode:
    graph:
      y: { op: identity, in: x }
    outputs: [y]
  dec:
    graph:
      y: { op: rmsnorm, in: x, dim: d_model }
    outputs: [y]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model } }
  - name: enc
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: e, rest: true }
      map:
        e: { template: enc }
  - { name: window, template: window_prefill, inputs: [enc] }
  - name: dec
    inputs: [window]
    repeat: n_mtp_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: d, rest: true }
      map:
        d: { template: dec }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [dec]}
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "vocab_size": 16,
                            "num_hidden_layers": 2,
                            "num_nextn_predict_layers": 2,
                            "model_type": "_window_var"})
    prefill = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    assert _seq_extent(prefill, "enc") == 16
    assert _seq_extent(prefill, "dec") == 4
    decode = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).during(DECODE, cache_len=16).analyze()
    assert _seq_extent(decode, "enc") == 1
    assert _seq_extent(decode, "dec") == 1


def test_shared_kv_is_kept_by_its_writer_alone():
    """YOCO / CLA: later layers attend the KV an earlier layer cached,
    through the inner stream the writer `stream_out`s and the readers name
    the writing layer of (`reads_from`). The cache and its bytes stay with
    the writer; the readers declare none, keep none, and read at the
    writer's KV length. No join is involved."""
    from backend.arch.eval import cache as cache_mod
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = """
match: { model_type: [_stream_kv_shared] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  inner:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { shared_kv: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [y]
  self_decoder:
    graph:
      body: { op: inner, in: x }
    outputs: [body]
  cross_decoder:
    reads: [shared_kv]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: shared_kv, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      out: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [score, out]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, shared_kv: {} }
    per_layer:
      assign:
        - { type: s, range: ["0", "0"] }
        - { type: c, rest: true, reads_from: { shared_kv: "0" } }
      map:
        s: { template: self_decoder }
        c: { template: cross_decoder }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers],
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 4,
                            "vocab_size": 16,
                            "model_type": "_stream_kv_shared"})
    assert resolved.spec.get_block("inner").kv_cached == ["kv"]
    assert not resolved.spec.get_block("cross_decoder").kv_cached
    analysis = Deployment(Program.of(resolved)).at(
        batch=1, input_tokens=16).analyze()
    assert not [n for n in analysis.program.nodes()
                if n.provenance == "lowering"]
    assert _score_s_kv(analysis) == 4
    caches = cache_mod.derive(analysis).values
    holders = [(analysis.program.by_uid(uid).template, entry)
               for uid, entry in caches.items()]
    assert [(template, [t.name for t in entry.tensors])
            for template, entry in holders] == [("self_decoder", ["kv"])]
    counts = {t.meta.template_name: t.count for t in analysis.program.blocks}
    assert (counts["self_decoder"], counts["cross_decoder"]) == (1, 3)
    # One copy of the cache: the writer's, however many layers read it.
    (holder,) = caches
    assert analysis.instances.values[holder] == 1

    # The serving path marks who writes and who reads it: the writer's `kv`
    # writes the cache, the readers' head reshape of the stream publishes
    # it, and their attention score reads it — traced through the layer the
    # readers' `reads_from` names, since the reader's ref names the stream,
    # not a node.
    from backend.inference.parallel_config import ParallelConfig
    from backend.inference.pipeline import analyze
    result = analyze(resolved, parallel=ParallelConfig(), batch_size=1,
                     avg_input_tokens=16, avg_output_tokens=0, phase="prefill", breakdown=False)
    program = result.pre_program
    access = result.pre_kv_access.values
    by_name = {}
    for uid, entry in access.items():
        by_name.setdefault(program.by_uid(uid).name, []).append(entry)
    (write,) = by_name["kv"]
    (ref,) = write.writes
    assert write.publishes and not write.reads
    (kh,) = by_name["kh"]
    assert kh.publishes and not kh.reads and not kh.writes
    (score,) = by_name["score"]
    assert score.reads == frozenset({ref}) and not score.writes
    # Everything else marked is the reshape's own steps, moving the cache
    # along: no second writer, no other reader.
    others = [e for name, entries in by_name.items()
              if name not in ("kv", "score") for e in entries]
    assert all(e.publishes and not (e.reads or e.writes) for e in others)
    assert len(cache_mod.resident(result.analysis)) == 1


def _two_writer_spec() -> str:
    """YOCO with a second writing variant: layer 0 (`self_decoder`) and layer
    2 (`self_decoder_b`) each cache their own KV and publish it as
    `shared_kv`; layers 1 and 3 — one template — read the layer before. The
    readers hand on a `[B, S, d_model]` hidden state, so both writers cache
    one shape."""
    return _yoco_spec().replace("_stream_kv_shared", "_stream_kv_two").replace(
        '''  self_decoder:
    graph:''', '''  self_decoder_b:
    graph:
      body: { op: inner, in: x }
    outputs: [body]
  self_decoder:
    graph:''').replace(
        '''        - { type: c, rest: true, reads_from: { shared_kv: "0" } }''',
        '''        - { type: b, range: ["2", "2"] }
        - { type: c, rest: true, reads_from: { shared_kv: "i - 1" } }''').replace(
        '''        c: { template: cross_decoder }''', '''        b: { template: self_decoder_b }
        c: { template: cross_decoder }''')


def test_each_reader_of_a_two_writer_cache_reads_the_layer_it_names():
    """Two layers cache and publish `shared_kv` (0 and 2); the readers' `"i -
    1"` sends layer 1 to layer 0's cache and layer 3 to layer 2's. Each
    cache is kept once by its writer; the one reading template marks both
    as what it reads, and each reading layer's decode step reads one
    cache's law — the law of the layer it reads, not the two summed."""
    from backend.arch.eval import cache as cache_mod
    from backend.arch.eval import traffic
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    spec = _two_writer_spec()
    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 4,
                            "vocab_size": 16, "model_type": "_stream_kv_two"})
    program = Program.of(resolved)
    (reader,) = [t for t in program.blocks if t.name == "cross_decoder"]
    owner = reader.meta.iteration_owner
    writer_of = {}
    for template in program.blocks:
        for node in template.nodes():
            if node.name == "kv":
                writer_of[node.uid] = template.name
    origins = {layer: program.stream_origin(owner, layer, "shared_kv")
               for layer in reader.layer_indices}
    assert {layer: (writer_of[o.uid], o.iteration)
            for layer, o in origins.items()} == {
        1: ("self_decoder", 0), 3: ("self_decoder_b", 2)}

    analysis = Deployment(program).at(batch=1, input_tokens=16).analyze()
    access = cache_mod.kv_access(analysis).values
    caches = {ref for entry in access.values() for ref in entry.writes}
    assert len(caches) == 2
    (score,) = [uid for uid in access
                if analysis.program.by_uid(uid).name == "score"]
    assert access[score].reads == frozenset(caches)
    resident = cache_mod.resident(analysis)
    assert sorted(analysis.instances.values[holder.uid]
                  for holder, _entry, _tensor in resident.values()) == [1, 1]

    analysis, layer = _decode_traffic(spec, "_stream_kv_two", 4)
    law = traffic._kv_reads(analysis)
    assert sorted(_num(v) for v in law.values()) == [64, 64]
    ((_uid, row),) = _leaf_rows(analysis, layer, "score")
    assert _num(row.kv_reads) == 64
    # Two writing layers and two reading layers, each streaming one cache.
    assert _kv_total(analysis, layer) == pytest.approx(2 * 64)


_NEAREST = """
match: { model_type: [_stream_nearest] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  kv_writer:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { slot: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      y: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [y]
  wa: { graph: { body: { op: kv_writer, in: x } }, outputs: [body] }
  wb: { graph: { body: { op: kv_writer, in: x } }, outputs: [body] }
  wc: { graph: { body: { op: kv_writer, in: x } }, outputs: [body] }
  reuse:
    reads: [slot]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: slot, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      out: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [score, out]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, slot: {} }
    per_layer:
      assign:
        - { type: a, layers: ["0"] }
        - { type: b, layers: ["3"] }
        - { type: c, layers: ["6"] }
        - { type: r, rest: true,
            reads_from: { slot: { nearest: { from: config.kv_source_layers } } } }
      map:
        a: { template: wa }
        b: { template: wb }
        c: { template: wc }
        r: { template: reuse }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers],
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def test_a_reader_template_over_three_sources_reads_one_law_per_layer():
    """One reading template whose layers read three writers (layers 0, 3, 6
    write; every other layer reads the nearest of the config's
    `kv_source_layers`): each reading layer's value is the nearest writer
    before it, the template reads all three caches and keeps none, and its
    decode step reads ONE cache's law, in prefill and in decode alike."""
    from backend.arch.eval import cache as cache_mod
    from backend.arch.eval import traffic
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import DECODE, Deployment

    config = {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 9, "vocab_size": 16,
              "kv_source_layers": [0, 3, 6], "model_type": "_stream_nearest"}
    resolved = _load(_NEAREST, config)
    program = Program.of(resolved)
    (reader,) = [t for t in program.blocks if t.name == "reuse"]
    owner = reader.meta.iteration_owner
    writer_layer = {}
    for template in program.blocks:
        for node in template.nodes():
            if node.name == "kv":
                (writer_layer[node.uid],) = template.layer_indices
    produced = {layer: program.stream_origin(owner, layer, "slot")
                for layer in reader.layer_indices}
    assert {layer: (writer_layer[o.uid], o.iteration)
            for layer, o in produced.items()} == {
        1: (0, 0), 2: (0, 0), 4: (3, 3), 5: (3, 3), 7: (6, 6), 8: (6, 6)}
    assert program.component_stream_roles(reader)[0] == ()
    assert len(program.block_stream_reads(
        program.block_index(reader))["slot"]) == 3

    for phase in ("prefill", "decode"):
        deployment = Deployment(program).at(batch=1, input_tokens=16)
        if phase == "decode":
            deployment = deployment.during(DECODE, cache_len=16)
        analysis = deployment.analyze()
        access = cache_mod.kv_access(analysis).values
        caches = {ref for entry in access.values() for ref in entry.writes}
        assert len(caches) == 3, phase
        (score,) = [uid for uid in access
                    if analysis.program.by_uid(uid).name == "score"]
        assert access[score].reads == frozenset(caches), phase
        # The readers keep no cache: three are kept, one per writer.
        assert len(cache_mod.resident(analysis)) == 3, phase
        layer = traffic.derive(analysis)
        law = traffic._kv_reads(analysis)
        if phase == "prefill":
            continue
        assert sorted(_num(v) for v in law.values()) == [64, 64, 64]
        ((_uid, row),) = _leaf_rows(analysis, layer, "score")
        assert _num(row.kv_reads) == 64
        # Six reading layers, each streaming one cache once.
        assert _kv_total(analysis, layer) == pytest.approx(6 * 64)


def _yoco_spec() -> str:
    import inspect

    source = inspect.getsource(test_shared_kv_is_kept_by_its_writer_alone)
    return source.split('spec = """', 1)[1].split('"""', 1)[0]


def _decode_traffic(spec: str, model_type: str, n_layers: int, precision=None):
    """The decode step's analysis and traffic, `cache_len` 16."""
    from backend.arch.eval import traffic
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import DECODE, Deployment

    resolved = _load(spec, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": n_layers,
                            "vocab_size": 16, "model_type": model_type})
    analysis = Deployment(Program.of(resolved), precision=precision).at(
        batch=1, input_tokens=16).during(DECODE, cache_len=16).analyze()
    layer = traffic.derive(analysis)
    return analysis, layer


def _num(extent) -> float:
    from backend.arch.eval.dim import value_of
    return float(value_of(extent))


def _leaf_rows(analysis, layer, name):
    return [(uid, row) for uid, row in layer.values.items()
            if not analysis.program.by_uid(uid).children
            and analysis.program.by_uid(uid).name == name]


def _kv_total(analysis, layer):
    """Cache bytes a step reads over the whole model: each leaf's share
    times the layers it stands for."""
    return sum(_num(row.kv_reads) * analysis.instances.values.get(uid, 1)
               for uid, row in layer.values.items()
               if not analysis.program.by_uid(uid).children)


def test_yoco_reader_reads_the_shared_cache_by_its_law():
    """A later layer attending the KV an earlier one `stream_out`ed reads
    the CACHE, not an activation edge: at decode its read is the writer's
    own law — `2*B*d_model*Min(s_window, s_kv)` at the cache dtype — billed
    as `kv_reads`, exactly as a reader in the writer's layer would be.
    Each reading layer reads it once; the cache is still stored once."""
    import sympy as sp

    from backend.arch.eval import cache as cache_mod
    from backend.arch.eval import traffic

    spec = _yoco_spec()
    analysis, layer = _decode_traffic(spec, "_stream_kv_shared", 4)
    (writer,) = [(uid, row) for uid, row in _leaf_rows(analysis, layer, "kv")]
    (_reader_uid, reader), = _leaf_rows(analysis, layer, "score")
    law = traffic._kv_reads(analysis)
    assert set(law) == {writer[0]}
    B, d_model, s_kv = sp.symbols("B d_model s_kv")
    assert sp.simplify(sp.sympify(str(law[writer[0]].expr))
                       - 2 * B * d_model * sp.Min(4, s_kv)) == 0
    assert reader.kv_reads == law[writer[0]]
    assert _num(reader.kv_reads) == 64
    # The span the law took over is off the edge: what remains is the query.
    assert _num(reader.act_reads) == 2 * 8
    # The reader writes its own output, `[B, n_heads, 1, s_window]`, and
    # nothing of the cache.
    assert _num(reader.act_writes) == 2 * 4
    assert _num(writer[1].kv_reads) == 0

    # The cache's own dtype prices it, not the activation dtype.
    from backend.arch.eval.precision import PRESETS, PrecisionProfile
    _a, fp8 = _decode_traffic(spec, "_stream_kv_shared", 4,
                              PRESETS["kv_fp8"])
    ((_uid, row),) = _leaf_rows(_a, fp8, "score")
    assert _num(row.kv_reads) == 32
    assert _num(row.act_reads) == _num(reader.act_reads)
    _a, f32_act = _decode_traffic(
        spec, "_stream_kv_shared", 4, PrecisionProfile.canonicalize(
            {"activations": [{"format": {"dtype": "F32"}}]}))
    ((_uid, row),) = _leaf_rows(_a, f32_act, "score")
    assert _num(row.kv_reads) == 64

    # One reading layer, then two: reads double, the cache does not.
    totals, stored = {}, {}
    for readers in (1, 2):
        analysis, layer = _decode_traffic(spec, "_stream_kv_shared",
                                          readers + 1)
        totals[readers] = _kv_total(analysis, layer)
        resident = cache_mod.resident(analysis)
        stored[readers] = [analysis.instances.values[holder.uid]
                           for holder, _entry, _tensor in resident.values()]
    assert totals == {1: 64, 2: 128}
    assert stored == {1: [1], 2: [1]}


_CLA = """
match: { model_type: [_stream_kv_cla] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, s_window: "4", n_heads: "1",
        d_head: config.hidden_size, n_kv_heads: "1" }
blocks:
  attn_w:
    cache: { family: MLA, entries: [{ node: kv, kind: latent }] }
    stream_out: { shared_kv: kv }
    graph:
      kv: { op: sliding_window_kv, in: x, dims: { s_window: s_window } }
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: kv, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
             dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
  writer:
    graph:
      body: { op: attn_w, in: x }
    outputs: [body]
  reader:
    reads: [shared_kv]
    graph:
      qh: { op: head_split, in: x, dims: { n_heads: n_heads, d_head: d_head } }
      kh: { op: head_split, in: shared_kv, dims: { n_heads: n_heads, d_head: d_head } }
      score: { op: attention_score, in: [qh, kh],
               dims: { n_heads: n_heads, d_head: d_head } }
      prob: { op: softmax, in: score, dims: { n_heads: n_heads } }
      ctx: { op: attention_apply, in: [prob, kh],
             dims: { n_heads: n_heads, d_head: d_head } }
      o: { op: head_merge, in: ctx, dims: { n_heads: n_heads, d_head: d_head } }
    outputs: [o]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true }, shared_kv: {} }
    per_layer:
      assign:
        - { type: w, layers: ["0", "2"] }
        - { type: r, rest: true, reads_from: { shared_kv: "i - 1" } }
      map:
        w: { template: writer }
        r: { template: reader }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab } , inputs: [layers],
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def test_cla_adjacent_layer_reads_the_cache_as_its_writer_does():
    """CLA: each even layer caches its KV and attends it; the odd layer
    after it attends that same KV through the stream. Both layers read the
    whole law, and inside one layer the score and the apply are two unfused
    kernels that each read it; the cache is kept once per writing layer."""
    from backend.arch.eval import cache as cache_mod
    from backend.arch.eval import traffic

    analysis, layer = _decode_traffic(_CLA, "_stream_kv_cla", 4)
    law = traffic._kv_reads(analysis)
    ((_seed, per_layer),) = law.items()
    assert _num(per_layer) == 64
    by_block = {}
    for uid, row in layer.values.items():
        node = analysis.program.by_uid(uid)
        if node.children or not row.kv_reads:
            continue
        assert node.name in ("score", "ctx")
        by_block.setdefault(node.template, []).append(
            (_num(row.kv_reads), analysis.instances.values[uid]))
    assert sorted(by_block) == ["reader", "writer"]
    for rows in by_block.values():
        assert len(rows) == 2
        assert [r for r, _n in rows] == pytest.approx([64, 64])
        assert {n for _r, n in rows} == {2}
    # Four layers, each with two kernels that stream the cache.
    assert _kv_total(analysis, layer) == pytest.approx(4 * 2 * 64)
    resident = cache_mod.resident(analysis)
    assert [analysis.instances.values[holder.uid]
            for holder, _entry, _tensor in resident.values()] == [2]


_POOLED_INDEX = """
match: { model_type: [_pooled_index] }
dims: { d_model: config.hidden_size, n_layers: config.num_hidden_layers,
        n_vocab: config.vocab_size, compress_ratio: "4", n_indexer_heads: "2",
        d_indexer_head: "8" }
blocks:
  blk:
    graph:
      k:        { op: linear, in: x, dims: { in: d_model, out: d_indexer_head },
                  weight_axes: [{ replicated: true }, { replicated: true }] }
      k_usable: { op: slice, in: k, axis: 1,
                  dims: { out: "floor(s_ctx / compress_ratio) * compress_ratio" } }
      k_win:    { op: view_split, in: k_usable, axis: 1,
                  dims: { s_slots: "floor(s_ctx / compress_ratio)", window: compress_ratio } }
      k_pool:   { op: sum_reduce, in: k_win, axis: 2, dims: { in: compress_ratio } }
      k_stream: { op: token_pool_compressor, in: k_pool, dim: d_indexer_head,
                  dims: { compress_ratio: compress_ratio } }
      w:        { op: linear, in: x, dims: { in: d_model, out: n_indexer_heads },
                  weight_axes: [{ replicated: true }, { replicated: true }] }
      q:        { op: linear, in: x, dims: { in: d_model, out: "n_indexer_heads * d_indexer_head" },
                  weight_axes: [{ replicated: true }, { replicated: true }] }
      q_heads:  { op: head_split, in: q, dims: { n_heads: n_indexer_heads, d_head: d_indexer_head } }
      logits:   { op: indexer_logits, in: [q_heads, k_stream, w],
                  dims: { n_indexer_heads: n_indexer_heads, d_indexer_head: d_indexer_head } }
      y:        { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
    outputs: [y]
model_inputs:
  input_ids: { kind: tokens }
model:
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: n_vocab, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: b, rest: true }
      map:
        b: { template: blk }
  - { name: head, op: linear, boundary_role: output_head,
      dims: { in: d_model, out: n_vocab }, inputs: [layers] }
"""


def test_a_reader_of_a_pooled_stream_scores_the_complete_pools():
    """An indexer fed kpool-pooled keys (GLM-5.3 `get_pooled_states`) scores
    ⌊s_kv / m⌋ keys, not `s_kv`: the pooled stream's own law sizes the key
    port, and the FLOPs follow that shape. At decode the pools are the
    cache's, and a cache shorter than one pool has none to score — the
    source keeps only pools whose every token is present."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.transforms import Deployment

    resolved = _load(_POOLED_INDEX, {"torch_dtype": "bfloat16", "hidden_size": 16, "num_hidden_layers": 1,
                                     "vocab_size": 32, "model_type": "_pooled_index"})
    base = Deployment(Program.of(resolved)).at(batch=1, input_tokens=16)

    def logits(deployment):
        analysis = deployment.analyze()
        (node,) = [n for n in analysis.program.nodes() if n.name == "logits"]
        dims = analysis.node_dims.values[node.uid]
        return int(dims["s_kv"]), analysis.cost.values[node.uid].flops

    per_key = 2 * (2 * 8 + 3)      # n_indexer_heads · (2 · d_indexer_head + 3)
    s_kv, flops = logits(base)
    # In prefill the query at position c (1-based) scores the ⌊c / 4⌋ pools
    # closed by then: Σ_{c=1..16} ⌊c / 4⌋ = 28 (query, pool) pairs, the
    # stream's own pairs law — not 16 · 4 halved by the dense causal factor.
    pairs = sum(c // 4 for c in range(1, 17))
    assert s_kv == 4 and pairs == 28 and flops == pairs * per_key
    for cache, pools in ((17, 4), (3, 0)):
        s_kv, flops = logits(base.during("decode", cache_len=cache))
        assert s_kv == pools and flops == pools * per_key, cache
