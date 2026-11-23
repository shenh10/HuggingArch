"""Looped stacks: a range of stored layers runs more than once per forward
pass with the same weights (`loop:`, `backend/arch/DESIGN.md` "Looped stacks").

The stack's iterations are its executions; the stored layer and pass of each
is `ExecutionSchedule`. The oracles here compare a looped spec against the
same model written WITHOUT a loop — sibling stacks, each with its own
weights, the gates as model-level layers — which runs none of the loop
machinery: FLOPs agree, the looped model stores each layer once.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from backend.arch.eval.program import Program
from backend.arch.eval.transforms import DECODE, PREFILL, Deployment
from backend.arch.layout.stages import StagePlacement
from backend.arch.loader import SpecLoadError, SpecLoader
from backend.arch.loop_schedule import Execution, ExecutionSchedule
from backend.arch.spec_tree import StreamReadRefusal


def _load(name: str, text: str, **config):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        for sub in ("components/library", "components/adapters",
                    "components/drafts", "models"):
            (base / sub).mkdir(parents=True)
        (base / "models" / f"{name}.yaml").write_text(text)
        return SpecLoader(base_dir=str(base)).match(
            {"model_type": name, "torch_dtype": "bfloat16", "hidden_size": 8,
             **config}, model_id=name)


_HEADER = """\
match: {{ model_type: [{name}], model_id: [{name}] }}
dims: {{ d_model: config.hidden_size, n_layers: "6", loop_first: "2", loop_last: "3" }}
"""

_BLOCKS = """\
blocks:
  w:
    stream_out: { s: o }
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      o: { op: linear, in: n, dims: { in: d_model, out: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
    outputs: [o]
  r:
    reads: [s]
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      o: { op: linear, in: n, dims: { in: d_model, out: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
      g: { op: mul, in: [o, s], dim: d_model }
    outputs: [g]
  extra:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      o: { op: linear, in: n, dims: { in: d_model, out: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
      p: { op: linear, in: o, dims: { in: d_model, out: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
    outputs: [p]
  gate:
    inputs: [h, x0]
    graph:
      n: { op: rmsnorm, in: x0, dim: d_model, weight_axes: [{ replicated: true }] }
      a: { op: add, in: [h, n], dim: d_model }
    outputs: [a]
model_inputs:
  input_ids: { kind: tokens }
"""

_EMBED = """\
  - { name: embed, op: embedding, inputs: [input_ids], boundary_role: input_embedding,
      dims: { n_vocab: d_model, d_model: d_model },
      weight_axes: [{ replicated: true }, { replicated: true }] }
"""


def _head(after: str) -> str:
    return f"""\
  - {{ name: head, op: linear, boundary_role: output_head, inputs: [{after}],
      dims: {{ in: d_model, out: d_model }},
      weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
"""


_GATE = "gate: { template: gate, checkpoint: { prefix: model.loop_gates } }"


def _looped(name: str, assign: str, *, types: str = "w: { template: w }",
            gate: bool = True, loop: str = "", streams: str = "s: {}") -> str:
    """Six stored layers, 2..3 looped twice; `assign` rows over `types`."""
    loop = loop or ("{ layers: [loop_first, loop_last], passes: 2"
                    + (f", {_GATE}" if gate else "") + " }")
    return (_HEADER.format(name=name) + _BLOCKS + "model:\n" + _EMBED + f"""\
  - name: layers
    inputs: [embed]
    repeat: n_layers
    loop: {loop}
    streams: {{ hidden: {{ carries: true }}, {streams} }}
    per_layer:
      assign:
{assign}      map:
        {types}
""" + _head("layers"))


def _rows(*rows: str) -> str:
    return "".join(f"        - {row}\n" for row in rows)


# The model of `_looped(rows("{ type: w, rest: true }"))` written without a
# loop: each pass its own stack and weights, each gate a model-level layer
# reading the stack before the pass (formal 1) and the pass (formal 0).
_UNROLLED = (_HEADER.format(name="unrolled") + _BLOCKS + "model:\n" + _EMBED
             + """\
  - { name: prelude, inputs: [embed], repeat: "2", template: w, streams: { hidden: { carries: true } } }
  - { name: pass1, inputs: [prelude], repeat: "2", template: w, streams: { hidden: { carries: true } } }
  - { name: gate1, inputs: [pass1, prelude], template: gate }
  - { name: pass2, inputs: [gate1], repeat: "2", template: w, streams: { hidden: { carries: true } } }
  - { name: gate2, inputs: [pass2, gate1], template: gate }
  - { name: coda, inputs: [gate2], repeat: "2", template: w, streams: { hidden: { carries: true } } }
""" + _head("coda"))


def _program(name: str, text: str) -> Program:
    return Program.of(_load(name, text))


def _walk(program: Program, phase=PREFILL):
    return Deployment(program).during(phase).at(
        batch=1, input_tokens=16).analyze()


# --- the schedule -----------------------------------------------------------

def test_the_schedule_runs_the_range_once_per_pass_with_a_gate_after_each():
    schedule = ExecutionSchedule.looped(6, 2, 3, 2, True, "test")
    assert schedule.executions == (
        Execution(0, 1), Execution(1, 1), Execution(2, 1), Execution(3, 1),
        Execution(0, 1, "gate"),
        Execution(2, 2), Execution(3, 2),
        Execution(1, 2, "gate"),
        Execution(4, 1), Execution(5, 1))
    assert schedule.ordinals_of(3) == (3, 6)
    assert schedule.latest_before(3, 9) == 6
    assert [schedule.pass_entry(k) for k in (1, 2)] == [1, 4]
    assert ExecutionSchedule.unlooped(4).executions == tuple(
        Execution(i, 1) for i in range(4))


@pytest.mark.parametrize("first, last, passes", [(4, 2, 2), (2, 6, 2),
                                                 (2, 3, 1)])
def test_a_loop_outside_the_stack_or_of_one_pass_is_refused(first, last,
                                                            passes):
    with pytest.raises(ValueError, match="loop"):
        ExecutionSchedule.looped(6, first, last, passes, False, "test")


# --- oracle: looped vs the same model unrolled ------------------------------

@pytest.mark.parametrize("phase", [PREFILL, DECODE])
def test_a_looped_stack_costs_what_its_unrolled_twin_runs_and_stores_once(
        phase):
    """FLOPs are the executions' — equal, symbolically, to the unrolled
    twin's — while the looped model stores each layer once: the twin holds
    the two looped layers a second time."""
    looped = _walk(_program("looped", _looped(
        "looped", _rows("{ type: w, rest: true }"))), phase)
    twin = _walk(_program("unrolled", _UNROLLED), phase)
    assert looped.total().flops == twin.total().flops
    d = 8
    w_layer = d * d + d
    assert int(looped.total().params) + 2 * w_layer == int(twin.total().params)
    assert int(looped.total().active_params) == int(looped.total().params)
    # Every execution streams its layer's weights, the later pass through
    # its tie: the same bytes the twin reads from its own copies.
    assert _weight_reads(looped) == _weight_reads(twin)
    # And runs the same nodes as often: per op, the executions.
    assert _runs_by_op(looped) == _runs_by_op(twin)


def _weight_reads(analysis) -> int:
    """Weight bytes the step reads: every leaf's, times its copies."""
    from backend.arch.eval import traffic as traffic_mod
    from backend.arch.eval.dim import value_of
    leaves = traffic_mod.derive(analysis).leaves()
    return sum(int(value_of(t.weight_reads))
               * int(value_of(analysis.instances.values[uid]))
               for uid, t in leaves.values.items())


def _runs_by_op(analysis) -> dict:
    from backend.arch.eval.dim import value_of
    out: dict = {}
    for node in analysis.program.walked_nodes():
        if node.children or node.uid not in analysis.instances.values:
            continue
        out[node.op] = out.get(node.op, 0) + int(
            value_of(analysis.instances.values[node.uid]))
    return out


def test_the_gate_reads_the_value_leaving_the_pass_and_the_value_entering_it():
    """Written out: the gate after pass 1 (ordinal 4) reads the exit of
    ordinal 3 and the exit of ordinal 1 (the stack before the loop); the gate
    after pass 2 (ordinal 7) reads the exit of ordinal 6 and its own earlier
    output (ordinal 4), which is what pass 2 entered with."""
    analysis = _walk(_program("looped", _looped(
        "looped", _rows("{ type: w, rest: true }"))))
    program = analysis.program
    (layer,) = [t for t in program.blocks
                if t.meta.pass_ == 1 and not t.meta.loop_gate]
    (later,) = [t for t in program.blocks if t.meta.pass_ == 2]
    (gate,) = [t for t in program.blocks if t.meta.loop_gate]
    assert gate.layer_indices == (4, 7)
    norm = next(n for n in gate.nodes() if n.name == "n")
    add = next(n for n in gate.nodes() if n.name == "a")
    def out(template, name):
        return next(n for n in template.nodes() if n.name == name).uid

    w, w2 = out(layer, "o"), out(later, "o")

    def read(node, port):
        table = analysis.input_references.references[(node.uid, port)].table
        return {it: [(s.occurrence.uid, s.occurrence.iteration)
                     for s in found] for it, found in table}

    assert read(add, 0) == {4: [(w, 3)], 7: [(w2, 6)]}
    assert read(norm, 0) == {4: [(w, 1)], 7: [(out(gate, "a"), 4)]}


def test_a_later_pass_stores_nothing_and_reads_its_owner():
    program = _program("looped", _looped(
        "looped", _rows("{ type: w2, pass: 2 }", "{ type: w, rest: true }"),
        types="w: { template: w }\n        w2: { template: w }"))
    (owner,) = [t for t in program.blocks if t.meta.pass_ == 1
                and not t.meta.loop_gate]
    (later,) = [t for t in program.blocks if t.meta.pass_ == 2]
    assert later.layer_indices == (5, 6)
    assert later.meta.label == "w[pass=2 of w]"
    held = {n.weight_name: n.uid for n in owner.nodes() if not n.children}
    for node in later.nodes():
        if node.children or node.name == "g":
            continue
        assert not node.owns_weight
        assert node.tied_to_uid == held[node.weight_name]


def test_a_later_pass_that_reads_a_weight_its_stored_layer_lacks_is_refused():
    with pytest.raises(ValueError, match="adds none"):
        _program("looped", _looped(
            "looped", _rows("{ type: x2, pass: 2 }", "{ type: w, rest: true }"),
            types="w: { template: w }\n        x2: { template: extra }"))


# --- reads_from in a looped stack -------------------------------------------

def _reads(read: str) -> dict:
    program = _program("looped", _looped(
        "looped", _rows(f"{{ type: r2, pass: 2, reads_from: {{ s: {read} }} }}",
                      "{ type: w, rest: true }"),
        types="w: { template: w }\n        r2: { template: r }"))
    return {layer: at for (_stack, layer, _s), (_src, at)
            in program.ir.stream_reads.items()}


def test_pass_n_reads_the_same_stored_layer_in_pass_n():
    """Pass 2 of stored layers 2 and 3 (ordinals 5, 6) read pass 1's
    writes of the same layers (ordinals 2, 3); with `layer: "i - 1"`,
    pass 1 of the stored layer before."""
    assert _reads("{ pass: 1 }") == {5: 2, 6: 3}
    assert _reads('{ pass: 1, layer: "i - 1" }') == {5: 1, 6: 2}


def test_a_read_of_the_latest_execution_that_does_not_write_is_refused():
    """Without `pass:`, pass 2 of stored layer 3 reading `"i - 1"` reads the
    latest execution of stored layer 2 before it — its pass 2, which runs
    the reader and writes nothing. That is refused, not walked back to pass
    1."""
    with pytest.raises(StreamReadRefusal):
        _reads('"i - 1"')


# --- pipeline residency -------------------------------------------------------

def test_a_stage_holds_every_stored_layer_any_of_its_executions_runs():
    """pp = 3 cuts ordinals 0..2 | 3..6 | 7..9. Stage 1 runs pass 1 of
    stored layer 3, gate 1 and pass 2 of stored layers 2 and 3, so it holds
    stored layers 2 and 3; the later pass itself stores nothing."""
    analysis = _walk(_program("looped", _looped(
        "looped", _rows("{ type: w2, pass: 2 }", "{ type: w, rest: true }"),
        types="w: { template: w }\n        w2: { template: w }")))
    program = analysis.program
    placement = StagePlacement.of(analysis, 3)
    assert placement.ranges == (range(3), range(3, 7), range(7, 10))
    held = [placement.view(s).weights(placement.backbone).values
            for s in range(3)]
    by_label = {t.meta.label: t.roots[0].uid for t in program.blocks}
    assert [h[by_label["w"]] for h in held] == [3, 2, 2]
    assert [h[by_label["gate"]] for h in held] == [0, 1, 1]


# --- refusals -----------------------------------------------------------------

def test_pass_on_a_stack_without_a_loop_is_refused():
    text = _looped("looped", _rows("{ type: w2, pass: 2 }",
                                 "{ type: w, rest: true }"),
                   types="w: { template: w }\n        w2: { template: w }")
    text = text.replace(
        "    loop: { layers: [loop_first, loop_last], passes: 2, "
        + _GATE + " }\n", "")
    with pytest.raises(ValueError, match="declares no `loop:`"):
        _program("looped", text)


@pytest.mark.parametrize("row, match", [
    ("{ type: w2, pass: 3 }", "does not run"),
    ("{ type: w2, pass: 2, range: [\"0\", \"1\"] }", "none inside the loop"),
])
def test_a_pass_the_loop_does_not_run_is_refused(row, match):
    with pytest.raises(ValueError, match=match):
        _program("looped", _looped(
            "looped", _rows(row, "{ type: w, rest: true }"),
            types="w: { template: w }\n        w2: { template: w }"))


def test_appended_streams_on_a_looped_stack_are_refused():
    with pytest.raises(SpecLoadError, match="looped stack"):
        _program("looped", _looped(
            "looped", _rows("{ type: w, rest: true }"),
            streams="s: {}, taps: { from: hidden, before: [1], src: \"a forward line\" }"))


# --- oracle: per-layer formats follow the STORED layer ------------------------

_OVERRIDE = """\
quant_context:
  default_dtype: BF16
  per_role:
    default: { method: none }
  overrides:
    - select:
        layers: ["LAYER"]
        tensors_re: '\.o\.weight$'
      dtype: F32
      method: none
"""


@pytest.mark.parametrize("layer", ["4", "3"])
def test_a_per_layer_format_lands_on_the_stored_layer_it_names(layer):
    """A quant override names a stored layer. The looped model stores the
    same bytes as the same six layers without a loop: stored layer 4 runs at
    ordinal 8 and the gate at ordinal 4, so an override read at the ordinal
    would put F32 on the wrong tensor. The second case puts it on a loop
    layer, which a later pass reads and does not store again."""
    rows = _rows("{ type: w2, pass: 2 }", "{ type: w, rest: true }")
    types = "w: { template: w }\n        w2: { template: w }"
    looped = _looped("looped", rows, types=types) + _OVERRIDE.replace(
        "LAYER", layer)
    plain = (looped.replace(
        "    loop: { layers: [loop_first, loop_last], passes: 2, "
        + _GATE + " }\n", "")
        .replace("        - { type: w2, pass: 2 }\n", "")
        .replace("\n        w2: { template: w }", ""))
    assert "loop:" not in plain and "w2" not in plain

    def stored(text):
        analysis = _walk(_program("looped", text))
        return sorted((row.name, int(row.count), str(row.storage_dtype))
                      for row in analysis.tensors()
                      if not row.name.startswith("model.loop_gates"))

    assert stored(looped) == stored(plain)


# --- GLM-5.4-shaped: the loop on GLM-5.2's backbone ---------------------------
#
# The draft runs GLM-5.2's stored layers 26..53 twice (z.ai's GLM-5.4 blog);
# its twin writes the same model as sibling stacks with their own weights and
# every read as a literal `{map}`. Both resolve against GLM-5.2's snapshot
# config. The draft draws pass 2 with GLM-5.2's shared-indexer attention: the
# blog's second KV bank is not part of what these oracles check.

_GLM_DRAFT = """\
# GLM-5.4 loop draft (test fixture, not a served model): GLM-5.2 with stored
# layers 26..53 run twice (z.ai blog, 2026-11). Weights are not public; the
# range, the pass count and the gate are the blog's. Pass 2 runs no indexer
# and reads pass 1's top-k of the same layer. The blog's second KV bank (a
# 128-token window of pass 2's own latent) is not drawn here: this fixture
# exercises the loop, not the two-bank attention.
base: glm_moe_dsa
derived_from: glm_moe_dsa
match:
  model_id: [test/GLM-5.4-loop-draft]

blocks:
  # x_k = x_{k-1} + beta_k * (h_k - x_{k-1}),  beta_k = sigmoid(Linear_k(RMSNorm(x_{k-1})))
  damp_gate:
    inputs: [h, x0]
    graph:
      norm:  { op: rmsnorm, in: x0, dim: d_model, weight_axes: [{ replicated: true }] }
      proj:  { op: linear, in: norm, dims: { in: d_model, out: "1" }, weight_axes: [{ replicated: true }, { replicated: true }] }
      beta:  { op: sigmoid, in: proj, dim: "1" }
      neg:   { op: scale_const, in: x0, dims: { in: d_model } }
      delta: { op: add, in: [h, neg], dim: d_model }
      step:  { op: mul, in: [beta, delta], dim: d_model }
      out:   { op: add, in: [x0, step], dim: d_model }
    outputs: [out]

model:
  - { name: embed_tokens, inputs: [input_ids], op: embedding, boundary_role: input_embedding, dims: { n_vocab: n_vocab, d_model: d_model }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }] }
  - name: layers
    inputs: [embed_tokens]
    repeat: n_layers
    loop:
      layers: ["26", "53"]
      passes: "2"
      gate: { template: damp_gate, checkpoint: { prefix: model.loop_gates } }
    streams:
      hidden: { carries: true }
      dsa_topk:
        shape: [B, s_query, n_topk_keys]
        src: "GlmMoeDsaAttention.forward threads `prev_topk_indices` across layers"
    per_layer:
      assign:
        - { type: p2_full, from: config.indexer_types, is: full, pass: 2,
            reads_from: { dsa_topk: { pass: 1 } } }
        - { type: p2_shared, from: config.indexer_types, is: shared, pass: 2,
            reads_from: { dsa_topk: { pass: 1, nearest: { from: config.indexer_types, is: full } } } }
        - { type: dense,  range: ["0", "n_dense_layers - 1"] }
        - { type: full,   from: config.indexer_types, is: full }
        - { type: shared, from: config.indexer_types, is: shared,
            reads_from: { dsa_topk: { nearest: { from: config.indexer_types, is: full } } } }
      map:
        p2_full:   { attention: mla_dsa_attention_shared, ffn: moe_ffn_gate_routing_bias }
        p2_shared: { attention: mla_dsa_attention_shared, ffn: moe_ffn_gate_routing_bias }
        dense:  { attention: mla_dsa_attention,        ffn: swiglu_ffn }
        full:   { attention: mla_dsa_attention,        ffn: moe_ffn_gate_routing_bias }
        shared: { attention: mla_dsa_attention_shared, ffn: moe_ffn_gate_routing_bias }
  - { name: norm, op: rmsnorm, dim: d_model, weight_axes: [{ replicated: true }] , inputs: [layers]}
  - { name: lm_head, op: linear, checkpoint: { path: lm_head }, boundary_role: output_head, dims: { in: d_model, out: n_vocab }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }], tied_weight: { to: embed_tokens, when: config.tie_word_embeddings } , inputs: [norm]}
  - name: mtp
    group: true
    aux_kind: mtp
    checkpoint: { prefix: model.layers, index: { after: backbone } }
    inputs: [norm]
    repeat: n_mtp_layers
    streams:
      hidden: { carries: true }
    layers:
      - name: mtp
        template: mtp_block_shared_head
        inputs: [norm, embed_tokens]
        params:
          inner: { template: pre_norm, params: { attn: mla_dsa_attention, ffn: moe_ffn_gate_routing_bias, norm: rmsnorm } }
      - { name: mtp_logits, op: linear, boundary_role: output_head, dims: { in: d_model, out: n_vocab }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }], tied_weight: { to: lm_head } , inputs: [mtp]}
"""


def _glm_twin(kinds, first=26, last=53, n_dense=3):
    def layers(stack_range, pred):
        return [i for i in stack_range if pred(i)]

    def stack(name, inputs, lo, hi, pass2=False, owner_stack=None):
        """Stored layers lo..hi as a stack of local layers 0..hi-lo."""
        rows = []
        local = {s: s - lo for s in range(lo, hi + 1)}
        full = [s for s in range(lo, hi + 1) if kinds[s] == "full"]
        shared = [s for s in range(lo, hi + 1) if kinds[s] == "shared"]
        dense = [s for s in range(lo, hi + 1) if s < n_dense]
        if pass2:
            # Pass 2 of the loop: no indexer; each layer reads pass 1's
            # top-k of its own stored layer (full) or of the nearest full
            # before it (shared), in the pass-1 stack.
            def src(s):
                if kinds[s] == "full":
                    return s
                return max(j for j in range(s) if kinds[j] == "full")
            for sel, name_ in ((full, "p2f"), (shared, "p2s")):
                if not sel:
                    continue
                m = ", ".join(f'"{local[s]}": {{ stack: {owner_stack}, layer: {src(s) - first} }}'
                              for s in sel)
                rows.append(f'        - {{ type: {name_}, layers: [{", ".join(chr(34) + str(local[s]) + chr(34) for s in sel)}], '
                            f'reads_from: {{ dsa_topk: {{ map: {{ {m} }} }} }} }}')
            mapping = ("        p2f: { attention: mla_dsa_attention_shared, ffn: moe_ffn_gate_routing_bias }\n"
                       "        p2s: { attention: mla_dsa_attention_shared, ffn: moe_ffn_gate_routing_bias }\n")
        else:
            d = [s for s in dense]
            f = [s for s in full if s not in d]
            if d:
                rows.append(f'        - {{ type: dense, layers: [{", ".join(chr(34) + str(local[s]) + chr(34) for s in d)}] }}')
            if f:
                rows.append(f'        - {{ type: full, layers: [{", ".join(chr(34) + str(local[s]) + chr(34) for s in f)}] }}')
            if shared:
                m = ", ".join(f'"{local[s]}": {max(j for j in range(lo, s) if kinds[j] == "full") - lo}'
                              for s in shared)
                rows.append(f'        - {{ type: shared, layers: [{", ".join(chr(34) + str(local[s]) + chr(34) for s in shared)}], '
                            f'reads_from: {{ dsa_topk: {{ map: {{ {m} }} }} }} }}')
            used = {r.split("type: ")[1].split(",")[0].split(" ")[0] for r in rows}
            mapping = "".join(
                f"        {t}: {spec}\n" for t, spec in (
                    ("dense", "{ attention: mla_dsa_attention, ffn: swiglu_ffn }"),
                    ("full", "{ attention: mla_dsa_attention, ffn: moe_ffn_gate_routing_bias }"),
                    ("shared", "{ attention: mla_dsa_attention_shared, ffn: moe_ffn_gate_routing_bias }"))
                if t in used)
        return f"""\
  - name: {name}
    inputs: [{inputs}]
    repeat: "{hi - lo + 1}"
    streams:
      hidden: {{ carries: true }}
      dsa_topk: {{ shape: [B, s_query, n_topk_keys], src: "twin" }}
    per_layer:
      assign:
{chr(10).join(rows)}
      map:
{mapping}"""

    n = len(kinds)
    body = (stack("prelude", "embed_tokens", 0, first - 1)
            + stack("pass1", "prelude", first, last)
            + "  - { name: gate1, inputs: [pass1, prelude], template: damp_gate }\n"
            + stack("pass2", "gate1", first, last, pass2=True, owner_stack="pass1")
            + "  - { name: gate2, inputs: [pass2, gate1], template: damp_gate }\n"
            + stack("coda", "gate2", last + 1, n - 1))
    return f"""\
base: glm_moe_dsa
derived_from: glm_moe_dsa
match:
  model_id: [test/GLM-5.4-loop-twin]
blocks:
  damp_gate:
    inputs: [h, x0]
    graph:
      norm:  {{ op: rmsnorm, in: x0, dim: d_model, weight_axes: [{{ replicated: true }}] }}
      proj:  {{ op: linear, in: norm, dims: {{ in: d_model, out: "1" }}, weight_axes: [{{ replicated: true }}, {{ replicated: true }}] }}
      beta:  {{ op: sigmoid, in: proj, dim: "1" }}
      neg:   {{ op: scale_const, in: x0, dims: {{ in: d_model }} }}
      delta: {{ op: add, in: [h, neg], dim: d_model }}
      step:  {{ op: mul, in: [beta, delta], dim: d_model }}
      out:   {{ op: add, in: [x0, step], dim: d_model }}
    outputs: [out]
model:
  - {{ name: embed_tokens, inputs: [input_ids], op: embedding, boundary_role: input_embedding, dims: {{ n_vocab: n_vocab, d_model: d_model }}, weight_axes: [{{ world: attention, parts: n_vocab, fit: pad }}, {{ replicated: true }}] }}
{body}  - {{ name: norm, op: rmsnorm, dim: d_model, weight_axes: [{{ replicated: true }}] , inputs: [coda]}}
  - {{ name: lm_head, op: linear, checkpoint: {{ path: lm_head }}, boundary_role: output_head, dims: {{ in: d_model, out: n_vocab }}, weight_axes: [{{ world: attention, parts: n_vocab, fit: pad }}, {{ replicated: true }}], tied_weight: {{ to: embed_tokens, when: config.tie_word_embeddings }} , inputs: [norm]}}
"""


def _glm(text: str, model_id: str) -> Program:
    from tests.snapshots import snapshot_config
    config = snapshot_config("zai-org/GLM-5.2")
    with tempfile.TemporaryDirectory() as d:
        models = Path(d) / "models"
        models.mkdir()
        (models / (model_id.replace("/", "__") + ".yaml")).write_text(text)
        resolved = SpecLoader(overlay_model_dirs=[str(models)]).match(
            config, model_id=model_id)
    return Program.of(resolved)


@pytest.mark.regression
@pytest.mark.parametrize("phase", [DECODE, PREFILL])
def test_glm_shaped_loop_costs_its_unrolled_twin_and_stores_glm52(phase):
    """FLOPs and KV bytes are the twin's; the stored weights are GLM-5.2's
    plus the two gates (one RMSNorm and one d_model → 1 projection each)."""
    from backend.arch.eval.cache_view import cache_view_from_analysis
    from tests.snapshots import resolve_snapshot, shared_loader, snapshot_config
    kinds = snapshot_config("zai-org/GLM-5.2")["indexer_types"]
    loop = _glm(_GLM_DRAFT, "test/GLM-5.4-loop-draft")
    twin = _glm(_glm_twin(kinds), "test/GLM-5.4-loop-twin")
    base = Program.of(resolve_snapshot("zai-org/GLM-5.2", shared_loader()))
    walked = {name: Deployment(p).during(phase).at(
                  batch=1, input_tokens=4096).analyze()
              for name, p in (("loop", loop), ("twin", twin), ("base", base))}
    cost = {name: a.by_stack()[""] for name, a in walked.items()}
    assert int(cost["loop"].flops) == int(cost["twin"].flops)
    kv = {name: cache_view_from_analysis(a).storage_bytes_at(4096)
          for name, a in walked.items()}
    assert kv["loop"] == kv["twin"]
    d_model = int(loop.base_binding["d_model"])
    assert int(cost["loop"].params) - int(cost["base"].params) == 2 * (
        d_model + d_model)
    assert loop.headline.layers == 78
    assert loop.headline.loop == (26, 53, 2)
    # Attention families count the layers that run them in their own stack:
    # the twin's 106 (its sibling stacks number layers apart); the loop's 78
    # stored layers, the 7 full ones of the loop running shared attention
    # in pass 2 as well (57 + 7).
    families = {name: [f.layers for f in a.attention_families]
                for name, a in walked.items()}
    assert families["twin"] == [21, 85]
    assert families["loop"] == [21, 64]
    assert families["base"] == [21, 57]


@pytest.mark.regression
def test_glm_shaped_loop_ties_split_pieces_to_the_stored_holder():
    """Decode runs GLM's DSA attention absorbed: `kv_b_proj` is a stored
    holder and two pieces split from it at load. In pass 2 the pieces still
    partition their own graph's holder (`split_of`) and read the pass-1
    holder's tensor (`tied_to_uid`); storage keeps one `kv_b_proj` per
    stored layer."""
    analysis = Deployment(_glm(_GLM_DRAFT, "test/GLM-5.4-loop-draft")).during(
        DECODE).at(batch=1, input_tokens=64).analyze()
    program = analysis.program
    later = [t for t in program.blocks if t.meta.pass_ == 2]
    assert later
    owners = {t.meta.declaration: t for t in program.blocks
              if t.meta.pass_ == 1 and not t.meta.loop_gate}
    for template in later:
        owner = owners[template.meta.loop_owner]
        held = {n.weight_name: n.uid for n in owner.nodes()
                if not n.children and n.owns_weight}
        pieces = [n for n in template.nodes() if n.split_at_load]
        assert len(pieces) == 2
        (own,) = {n.split_of for n in pieces}
        assert own in {n.uid for n in template.nodes()}
        assert {n.tied_to_uid for n in pieces} == {held["self_attn.kv_b_proj"]}
        # Each graph's pieces partition their own holder: the shares sum to
        # one per `split_of`, read against the stored tensor's shape.
        shares = [next(iter(program.read_aliases[n.uid].values())).share
                  for n in pieces]
        assert sum(shares) == 1
    rows = [row for row in analysis.tensors() if "kv_b_proj" in row.name
            and "layers" in row.name]
    assert sum(int(row.count) for row in rows) == 78 + 1   # + the MTP layer


@pytest.mark.regression
def test_an_mtp_head_after_a_looped_backbone_indexes_after_its_stored_layers():
    """The MTP head's checkpoint index follows the backbone's 78 stored
    layers, and the final norm it reads is the backbone's last execution:
    stored layer 77 at ordinal 107 (78 layers + 28 again + 2 gates)."""
    analysis = Deployment(_glm(_GLM_DRAFT, "test/GLM-5.4-loop-draft")).during(
        DECODE).at(batch=1, input_tokens=64).analyze()
    program = analysis.program
    (mtp,) = [t for t in program.blocks if t.meta.aux_kind == "mtp"]
    assert mtp.meta.backbone_layers == 78
    norm = next(n for n in program.nodes() if n.name == "norm"
                and program.block_of(n.uid) is None)
    ((_it, (source,)),) = analysis.input_references.references[
        (norm.uid, 0)].table
    assert source.occurrence.iteration == 107
    assert program.iteration_label(
        program.blocks[program.block_of(source.occurrence.uid)], 107) == 77



# --- review round 1 (codex impl 1) --------------------------------------------

_SEVEN = """\
match: { model_type: [looped], model_id: [looped] }
dims: { d_model: config.hidden_size, n_layers: "7" }
blocks:
  w:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      o: { op: linear, in: n, dims: { in: d_model, out: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
    outputs: [o]
  v:
    graph:
      n: { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      o: { op: linear, in: n, dims: { in: d_model, out: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
      p: { op: linear, in: o, dims: { in: d_model, out: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
    outputs: [p]
model_inputs:
  input_ids: { kind: tokens }
model:
""" + _EMBED + """\
  - name: layers
    inputs: [embed]
    repeat: n_layers
    loop: { layers: ["1", "5"], passes: 2 }
    streams: { hidden: { carries: true } }
    per_layer:
      assign:
        - { type: w2, pass: 2, layers: ["1", "3", "5"] }
        - { type: v2, pass: 2, layers: ["2", "4"] }
        - { type: wa, layers: ["1", "3", "5"] }
        - { type: vb, layers: ["2", "4"] }
        - { type: wa, rest: true }
      map:
        w2: { template: w }
        v2: { template: v }
        wa: { template: w }
        vb: { template: v }
""" + _head("layers")


def test_a_stage_running_only_part_of_a_later_pass_holds_only_those_layers():
    """Seven stored layers, 1..5 looped twice, three owners with scattered
    layers (A: 1, 3, 5; B: 2, 4; C: 0, 6). pp = 3 cuts the 12 executions
    0..3 | 4..7 | 8..11 (stored 0-3 | 4, 5, then 1, 2 in pass 2 | 3, 4, 5
    in pass 2, then 6). Each stage holds exactly the stored layers it runs
    in any pass — not the whole owner, which the later pass's tied reads
    would have promoted onto a stage that runs part of it."""
    analysis = _walk(_program("looped", _SEVEN))
    program = analysis.program
    placement = StagePlacement.of(analysis, 3)
    assert placement.ranges == (range(4), range(4, 8), range(8, 12))
    owner = {t.meta.assignment_index: t.roots[0].uid for t in program.blocks
             if t.meta.pass_ == 1}
    held = [placement.view(s).weights(placement.backbone).values
            for s in range(3)]
    assert [(h[owner[2]], h[owner[3]], h[owner[4]]) for h in held] == [
        (2, 1, 1), (2, 2, 0), (2, 1, 1)]


def test_a_later_pass_binding_its_weight_at_another_frame_is_refused():
    """One address, one shape, one spelling `parts: n_heads` — but the
    later pass binds `n_heads` differently, so its tensor is cut otherwise:
    not the stored layer's."""
    text = _looped(
        "looped",
        _rows("{ type: w2, pass: 2, dims: { n_heads: \"4\" } }",
              "{ type: w, rest: true }"),
        types="w: { template: h }\n        w2: { template: h }")
    text = text.replace('loop_last: "3" }', 'loop_last: "3", n_heads: "2" }')
    text = text.replace("model_inputs:", """\
  h:
    graph:
      p: { op: parameter_tensor, shape: [d_model], tensor_suffix: { weight: .weight },
           weight_axes: [{ world: attention, parts: n_heads, fit: replica_cells }] }
      a: { op: add, in: [x, p], dim: d_model }
    outputs: [a]
model_inputs:""", 1)
    from backend.arch.eval.refusal import SpecRefusal
    # Refused before anything is priced: the walk asks it first, and so does
    # every reader of a tie (`read_aliases`).
    with pytest.raises(SpecRefusal, match="another shape or axis cut"):
        _walk(_program("looped", text))
    with pytest.raises(SpecRefusal, match="another shape or axis cut"):
        _program("looped", text).read_aliases  # noqa: B018
    # A dim the later pass binds otherwise and no stored field spells is no
    # other tensor: same address, shape and axis cut — tied.
    unused = (text.replace('dims: { n_heads: "4" }', 'dims: { n_kv_heads: "4" }')
              .replace('n_heads: "2" }', 'n_heads: "2", n_kv_heads: "2" }'))
    assert 'n_kv_heads: "4"' in unused and 'n_kv_heads: "2"' in unused
    assert _program("looped", unused).read_aliases



# --- review round 1 (kimi impl 1) ---------------------------------------------

def test_a_read_of_a_stored_layer_the_stack_does_not_have_is_refused():
    """Six stored layers run as ten executions; `"6"` names no stored layer
    and must not be read as ordinal 6 (stored 3, pass 2)."""
    with pytest.raises(StreamReadRefusal, match="stored layer 6"):
        _reads('{ pass: 1, layer: "6" }')


def test_another_stack_reads_a_looped_stacks_stored_layer_at_its_last_run():
    """An unlooped stack after the looped one reads `{stack: layers, layer:
    3}`: stored layer 3, whose last execution is its pass 2 (ordinal 6) —
    not ordinal 3."""
    text = _looped("looped", _rows("{ type: w2, pass: 2 }",
                                   "{ type: w, rest: true }"),
                   types="w: { template: w }\n        w2: { template: w }")
    text = text.replace(_head("layers"), """\
  - name: tail
    inputs: [layers]
    repeat: "1"
    template: r
    streams: { hidden: { carries: true } }
    reads_from: { s: { stack: layers, layer: "3" } }
""" + _head("tail"))
    program = _program("looped", text)
    (at,) = [at for (reader, _layer, _s), (source, at)
             in program.ir.stream_reads.items() if reader != source]
    assert at == 6
