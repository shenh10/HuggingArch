"""Cross-layer stream anchor — CPU tests.

Covers the checks in _cross_layer_channels.py:
  • truthfulness (gates): a declared stream `src` that isn't in the model's
    recorded forward source is an error, and every declared inner stream
    needs a producer and a consumer;
  • with no recorded source, the anchors are noted as unchecked (does not
    gate).
"""
from __future__ import annotations

import tempfile
import os
from unittest import mock

import pytest

from tests.snapshots import resolve_snapshot

from backend.arch.loader import SpecLoader
from backend.arch.validate._cross_layer_channels import (
    cross_layer_anchor_notes,
    run_cross_layer_checks,
    _resolves,
)


# --- unit: anchor resolution ------------------------------------------------

def test_resolves_real_anchor():
    src = "def forward(self, hidden_states, prev_topk_indices=None):\n" \
          "    hidden, prev = layer(hidden, prev_topk_indices=prev)"
    assert _resolves("`hidden, prev = layer(hidden, prev_topk_indices=prev)`", src)


def test_resolves_rejects_phantom():
    src = "def forward(self, hidden_states): return hidden_states"
    assert not _resolves("`self.made_up_function(xyz)`", src)


def test_empty_anchor_is_vacuously_true():
    assert _resolves("", "anything")


# --- integration: real snapshots -------------------------------------------

def test_glm_dsa_stream_wellformed():
    """GLM-DSA declares its DSA top-k sharing as the `dsa_topk` stream
    (mla_dsa_attention stream_out, mla_dsa_attention_shared reads, src anchored)
    — the stream is declared, so NO cross-layer warning fires."""
    resolved = resolve_snapshot("zai-org/GLM-5.2")
    assert run_cross_layer_checks("zai-org/GLM-5.2", resolved) == []


def test_plain_model_silent():
    """A standard decoder (Llama) carries only `hidden` and declares no inner
    stream — no warning."""
    resolved = resolve_snapshot("meta-llama/Llama-3.2-1B")
    assert run_cross_layer_checks("meta-llama/Llama-3.2-1B", resolved) == []


# --- streams: producer / consumer well-formedness --------------------------

def _stream_kinds(*, streams: str, prod_so: str, cons_reads: str,
                  source=None, orphan: str = ""):
    """Build a synthetic spec whose stack binds a producer block (layer 0)
    and a consumer block (the rest, reading layer 0's write when the stream
    is declared), with the given stream declaration; run the check against
    `source` (None = no forward source recorded), return the set of warning
    kinds. The stack carries its hidden state (`head` reads it by name)
    beside the stream under test. `orphan` is the interface of one more
    block the spec declares and no layer binds."""
    carried = "hidden: { carries: true }"
    # A block that reads `s` consumes it: a read is an input, and a read no
    # node names is refused at load.
    cons_use = ", r: { op: identity, in: s }" if cons_reads else ""
    orphan_use = ", r: { op: identity, in: s }" if orphan else ""
    reads_from = (', reads_from: { s: "0" }'
                  if streams and cons_reads else "")
    streams = (streams.replace(", streams: { ", f"streams: {{ {carried}, ", 1)
               if streams else f"streams: {{ {carried} }}")
    spec = f"""
match: {{ model_type: [_stream_probe] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers,
         n_vocab: config.vocab_size }}
blocks:
  prod: {{ {prod_so} graph: {{ mk: {{ op: rmsnorm, in: x, dim: d_model }} }}, outputs: [mk] }}
  cons: {{ {cons_reads} graph: {{ u: {{ op: rmsnorm, in: x, dim: d_model }}{cons_use} }}, outputs: [u] }}
  orphan: {{ {orphan} graph: {{ o: {{ op: rmsnorm, in: x, dim: d_model }}{orphan_use} }}, outputs: [o] }}
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
       dims: {{ n_vocab: n_vocab, d_model: d_model }} }}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    {streams}
    per_layer:
      assign:
        - {{ type: first, range: ["0", "0"] }}
        - {{ type: rest, rest: true{reads_from} }}
      map:
        first: {{ template: prod }}
        rest: {{ template: cons }}
  - {{ name: head, op: linear, boundary_role: output_head,
       dims: {{ in: d_model, out: n_vocab }}, inputs: [layers] }}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(spec)
        path = f.name
    loader = SpecLoader()
    try:
        sp = loader.load_model_spec(path)
    finally:
        os.unlink(path)
    resolved = loader.resolve(
        sp, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 3, "vocab_size": 16,
             "model_type": "_stream_probe"})
    with mock.patch("backend.hf.forward_source.recorded_forward_source",
                    return_value=source):
        ws = run_cross_layer_checks("_stream_probe", resolved)
        hints = cross_layer_anchor_notes("_stream_probe", resolved)
    # The note is filed apart from the checks that gate.
    assert not {w.kind for w in ws} & {w.kind for w in hints}
    return {w.kind for w in ws} | {w.kind for w in hints}


def test_stream_wellformed_no_warning():
    kinds = _stream_kinds(
        streams=", streams: { s: { shape: [B, s_query, d_model] } }",
        prod_so="stream_out: { s: mk },",
        cons_reads="reads: [s],")
    assert kinds == set()


def test_stream_no_producer():
    # A reader of a stream no layer writes is refused where the IR is
    # built; a declaration no bound component writes is dangling here.
    kinds = _stream_kinds(
        streams=", streams: { s: { shape: [B, s_query, d_model] } }",
        prod_so="",                       # nobody stream_out's `s`
        cons_reads="")
    assert "cross_layer_stream_no_producer" in kinds


def test_a_component_the_stack_does_not_bind_neither_writes_nor_reads():
    """Producers and consumers are the components the declaring stack
    binds: a block the spec declares and no layer calls is not one."""
    kinds = _stream_kinds(
        streams=", streams: { s: { shape: [B, s_query, d_model] } }",
        prod_so="stream_out: { s: mk },",
        cons_reads="",
        orphan="reads: [s],")         # declared, read, bound by no layer
    assert "cross_layer_stream_no_consumer" in kinds


def test_stream_no_consumer():
    kinds = _stream_kinds(
        streams=", streams: { s: { shape: [B, s_query, d_model] } }",
        prod_so="stream_out: { s: mk },",
        cons_reads="")                    # nobody reads `s`
    assert "cross_layer_stream_no_consumer" in kinds


def test_stream_read_undeclared():
    kinds = _stream_kinds(
        streams="",                       # `s` never declared
        prod_so="stream_out: { s: mk },",
        cons_reads="reads: [s],")
    assert "cross_layer_stream_undeclared" in kinds


# --- anchors: checked against the recorded source, else noted -------------

_ANCHORED = ", streams: { s: { src: '`hidden, s = layer(hidden, s)`' } }"


def test_an_anchor_in_the_recorded_source_is_silent():
    kinds = _stream_kinds(streams=_ANCHORED, prod_so="stream_out: { s: mk },",
                          cons_reads="reads: [s],",
                          source="    hidden, s = layer(hidden, s)\n")
    assert kinds == set()


def test_an_anchor_missing_from_the_recorded_source_gates():
    kinds = _stream_kinds(streams=_ANCHORED, prod_so="stream_out: { s: mk },",
                          cons_reads="reads: [s],",
                          source="    return hidden\n")
    assert kinds == {"cross_layer_src_unresolved"}


def test_with_no_recorded_source_the_anchor_is_noted_unchecked():
    kinds = _stream_kinds(streams=_ANCHORED, prod_so="stream_out: { s: mk },",
                          cons_reads="reads: [s],")
    assert kinds == {"cross_layer_src_unchecked"}


# --- shape: declared vs producer node (IR vs IR) ----------------------------

def test_stream_shape_mismatch_flagged():
    """The producer node is [B, s_query, d_model]; declaring a different last
    axis must fire, and the detail names the producing variant."""
    ws = _stream_kinds(
        streams=", streams: { s: { shape: [B, s_query, n_vocab] } }",
        prod_so="stream_out: { s: mk },",
        cons_reads="reads: [s],")
    assert "cross_layer_stream_shape_mismatch" in ws


def _per_variant_kinds(narrow_override: str):
    """Two variants of one stream: `wide` layers run prod_a (d_model wide),
    `narrow` layers run prod_b (n_vocab wide); the declared shape tracks the
    per-variant `d_stream` dim. `narrow_override` is the dim expr the narrow
    variant binds d_stream to."""
    spec = f"""
match: {{ model_type: [_stream_probe] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers,
         n_vocab: config.vocab_size }}
blocks:
  prod_a: {{ stream_out: {{ s: mk }}, graph: {{ mk: {{ op: rmsnorm, in: x, dim: d_model }} }}, outputs: [mk] }}
  prod_b: {{ stream_out: {{ s: mk }}, graph: {{ mk: {{ op: linear, in: x, dims: {{ in: d_model, out: n_vocab }} }}, h: {{ op: rmsnorm, in: x, dim: d_model }} }}, outputs: [h] }}
  cons: {{ reads: [s], graph: {{ u: {{ op: rmsnorm, in: x, dim: d_model }}, r: {{ op: identity, in: s }} }}, outputs: [u] }}
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
       dims: {{ n_vocab: n_vocab, d_model: d_model }} }}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }}, s: {{ shape: [B, s_query, d_stream] }} }}
    per_layer:
      assign:
        - {{ type: wide, range: ["0", "0"], dims: {{ d_stream: d_model }} }}
        - {{ type: reader, range: ["2", "2"], reads_from: {{ s: "i - 1" }} }}
        - {{ type: narrow, rest: true, dims: {{ d_stream: {narrow_override} }} }}
      map:
        wide: {{ template: prod_a }}
        reader: {{ template: cons }}
        narrow: {{ template: prod_b }}
  - {{ name: head, op: linear, boundary_role: output_head,
       dims: {{ in: d_model, out: n_vocab }}, inputs: [layers] }}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(spec)
        path = f.name
    loader = SpecLoader()
    try:
        sp = loader.load_model_spec(path)
    finally:
        os.unlink(path)
    resolved = loader.resolve(
        sp, {"torch_dtype": "bfloat16", "hidden_size": 8, "num_hidden_layers": 3, "vocab_size": 16,
             "model_type": "_stream_probe"})
    with mock.patch("backend.hf.forward_source.recorded_forward_source",
                    return_value=None):
        return run_cross_layer_checks("_stream_probe", resolved)


def test_stream_shape_per_variant_dims_match():
    """One declared shape, two variant widths: each variant's `dims` override
    resolves d_stream to its own producer's width — no warning."""
    assert _per_variant_kinds("n_vocab") == []


def test_stream_shape_per_variant_dims_mismatch():
    """The narrow variant binds d_stream to the WIDE width while its producer
    is n_vocab wide — exactly one mismatch, tagged with the narrow variant."""
    ws = _per_variant_kinds("d_model")
    mismatches = [w for w in ws
                  if w.kind == "cross_layer_stream_shape_mismatch"]
    assert len(mismatches) == 1
    assert "prod_b" in mismatches[0].detail


def test_stream_spec_rejects_unknown_key():
    """Stream declarations are typed (extra=forbid) — a stray key fails at
    load instead of silently becoming dead annotation."""
    with pytest.raises(Exception, match="bogus"):
        _stream_kinds(
            streams=", streams: { s: { shape: [B, s_query, d_model], bogus: 1 } }",
            prod_so="stream_out: { s: mk },",
            cons_reads="reads: [s],")


def test_stream_name_only_skips_shape_gate():
    """The catalog is the name. Omitting `shape` is valid; the publish-gate
    that compares yaml shape to the producer node does not run."""
    kinds = _stream_kinds(
        streams=", streams: { s: { src: 'Model.forward threads s' } }",
        prod_so="stream_out: { s: mk },",
        cons_reads="reads: [s],")
    assert "cross_layer_stream_shape_mismatch" not in kinds
    assert "stream_shape_unresolved" not in kinds


@pytest.mark.parametrize("declared, flagged", [("d_model", False),
                                                ("n_vocab", True)])
def test_a_stream_seed_is_shape_checked_in_the_model_frame(declared, flagged):
    """A stream's `seed` is a model-level op that writes the stream at the
    stack's entry. Its shape is held to the declaration like every write,
    in the model's frame — it belongs to no block invocation (DeepSeek-V4.1's
    `hc_seed` crashed this check when it landed)."""
    spec = f"""
match: {{ model_type: [_seed_probe] }}
dims: {{ d_model: config.hidden_size, n_layers: config.num_hidden_layers,
         n_vocab: config.vocab_size }}
blocks:
  rw: {{ reads: [s], stream_out: {{ s: g }}, graph: {{ g: {{ op: mul, in: [x, s], dim: d_model }} }}, outputs: [g] }}
model_inputs:
  input_ids: {{ kind: tokens }}
model:
  - {{ name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
       dims: {{ n_vocab: n_vocab, d_model: d_model }} }}
  - {{ name: seed, op: one_hot, inputs: [embed], dims: {{ in: d_model, out: d_model }} }}
  - name: layers
    inputs: [embed]
    repeat: n_layers
    streams: {{ hidden: {{ carries: true }}, s: {{ seed: seed, shape: [B, s_query, {declared}] }} }}
    per_layer:
      assign:
        - {{ type: rw, rest: true, reads_from: {{ s: "i - 1" }} }}
      map:
        rw: {{ template: rw }}
  - {{ name: head, op: linear, boundary_role: output_head,
       dims: {{ in: d_model, out: n_vocab }}, inputs: [layers] }}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(spec)
        path = f.name
    loader = SpecLoader()
    try:
        sp = loader.load_model_spec(path)
    finally:
        os.unlink(path)
    resolved = loader.resolve(
        sp, {"torch_dtype": "bfloat16", "hidden_size": 8,
             "num_hidden_layers": 3, "vocab_size": 16,
             "model_type": "_seed_probe"})
    with mock.patch("backend.hf.forward_source.recorded_forward_source",
                    return_value=None):
        found = run_cross_layer_checks("_seed_probe", resolved)
    assert "stream_shape_unjudged" not in [w.kind for w in found]
    mismatched = sorted(w.template for w in found
                        if w.kind == "cross_layer_stream_shape_mismatch")
    assert mismatched == (["rw", "seed"] if flagged else [])
