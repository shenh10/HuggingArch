## Model Config YAML Schema

A model config is a thin YAML that references shared components from the library.

Its identity is its file stem, exactly as for a component: `models/glm_moe_dsa.yaml`
is the spec named `glm_moe_dsa`, and that is the name `validate --spec-name` asks
for. A top-level `name` is rejected — to name the spec something else, name the
file something else. Copying a neighbouring family's spec into a new file and
adding `name:` back to keep the old identity is the mistake this rules out.

### Top-level fields

```yaml
# Comment describing the model family
match:
  model_type: [type1, type2]     # config.model_type values this spec matches

components:
  attention: <component_name>     # default attention component
  attention_swa: <comp_name>      # (optional) alternate attention for SWA layers
  ffn_dense: <component_name>     # dense FFN slot (also the default in dense models)
  ffn_moe: <component_name>       # (optional) MoE FFN for MoE layers
  norm: rmsnorm | layernorm       # normalization type
  block: pre_norm | post_norm     # block pattern (residual structure)

dims:
  <SYMBOL>: <binding>             # dimension bindings (resolved top-to-bottom)

model:
  - <model_layer>                 # ordered list of model layers
```

### Dimension binding syntax
Dims are resolved top-to-bottom. Later dims can reference earlier ones.

```yaml
dims:
  d_model:    config.hidden_size
  n_heads:    config.num_attention_heads
  n_kv_heads: config.num_key_value_heads ?? n_heads
  d_head:     config.head_dim ?? "d_model // n_heads"
  d_ffn:      config.intermediate_size
  n_vocab:    config.vocab_size
  n_layers:   config.num_hidden_layers
  n_shared_experts: config.n_shared_experts ?? "0"
```

Syntax: `config.field` | `config.field ?? fallback_expr` | `"expr"` | `literal_int` | `checkpoint.<tensor>[<axis>]` | `present(config.field) ?? fallback`

A numeric step of a config path reads one entry of a list:
`config.text_config.engram_num_embeddings.1` is the second table height of a
config that publishes one per engram layer. Read a per-layer value this way
into a dim of its own rather than writing the number. A literal bound under a
model dim's name in a node's or a model layer's `dims:` is refused, at the
model's value and at any other; in an assignment's `dims:` (which exists to
give the selected layers another value) it is refused at the model's own
value.

`checkpoint.<tensor name>[<axis>]` (e.g.
`checkpoint.model.layers.X.self_attn.rotary_emb.inv_freq[0]`, `X` at an indexed
position as in every spec-composed name) reads that axis of the shape the
checkpoint header stores. Use it only for an extent that is a
fact about the file rather than the model — a buffer the exporter serialized
whose length no config field or modeling-code formula determines. It takes no
`??` fallback and no arithmetic, and resolution fails when the header does not
carry the tensor.

**Use canonical descriptive symbol names.** The current vocabulary uses names
such as `d_model`, `n_heads`, `n_kv_heads`, `d_head`, `d_ffn`, `n_experts`,
and `n_experts_active`; the old single-letter `H/N/Nk/D/I/E/K` vocabulary is
retired. `backend/arch/dims.yaml` is the shared table; `resolve_dimensions`
accepts a symbol it lists or one this spec declares under `dim_symbols:`
(below). Don't invent a new name for a dimension already listed there, and
don't reuse a canonical name for a different meaning. Use the Read tool on
that file if unsure.

**A symbol the architecture needs that dims.yaml does not have** is declared
in the spec, under a top-level `dim_symbols:` block. Each entry is a dims.yaml
row: `kind` (what the number counts: `head_count`, `split_width`, `vocab`,
`sequence`, `batch`, `expert_copies`, `layer_count`, `whole`, or `constant`),
`shardable_by` (the `worlds:` slugs whose ranks may split it, written for
every kind except `whole`, `layer_count` and `constant`), `group` (a `groups:` slug in
dims.yaml), `meaning` (what the number counts, citing the forward-source line
that fixes the classification), and optionally `config` (the config field it
comes from) and `derived` (its value when nothing binds it). Bind it in `dims:`
(top level, or a block's / assignment's `dims:`) like any other symbol. When
the model enters the library these rows move into dims.yaml:

```yaml
dim_symbols:
  n_engram_rows:
    kind: vocab
    shardable_by: [attention]
    group: per_arch_one_offs
    meaning: >-
      Rows of one n-gram hash table, split over the tensor-parallel group like
      the token vocabulary (source/model.py:296, ceil(rows / world_size) per rank).
    config: engram_num_embeddings
  d_engram_head:
    kind: whole
    group: per_arch_one_offs
    meaning: Width of one n-gram table row (source/model.py:112).
    config: engram_head_dim

dims:
  d_engram_head: config.engram_head_dim
```

This block is the spec's own: the `dim_symbols` set a layer reports in
`backend.arch.structure` output is a different thing (the symbols that
layer's tensors are shaped on).

**Function constants** are bound in `dims:` too. A few ops take a real
number beside their operands — `clamp` its bound `clamp_limit`, `situ` its
`situ_beta` and `situ_linear_beta` (the op's `consts`). These symbols are
dims.yaml kind `constant`: not extents, so the value may be fractional, and
a config that does not publish one leaves it unbound (never a default). The
op says only that it takes the constant; the spec says where the config
keeps it, in the scope it lives in, and the op reads it from its node's
frame like any symbol:

```yaml
dims:
  clamp_limit:     config.text_config.swiglu_limit     # the backbone's clamps
  vis_clamp_limit: config.vision_config.swiglu_limit   # the tower's own bound
```

A clamp that reads another scope rebinds the symbol on the node or at the
call site — `{ op: clamp, dims: { dim: d_vis_ffn, clamp_limit: vis_clamp_limit } }`
— so a vision tower's clamp never reads the backbone's bound. A spec whose
graph has a `clamp` / `situ` binds its constant; the formula text alone does
not.

### Model layer types

Every model row chooses exactly one structural form. Fields from another form
are rejected rather than ignored:

| form | required discriminator | form-specific fields |
|---|---|---|
| direct op | `op:` | `boundary_role`, `dims` or `dim`, `axis`, `shape`, `weight_axes`, `expert_weight`, `stored_axes`, `tied_weight`, `quant_metadata`, `module`, `inner_modules`, `checkpoint: {path, index: local}` |
| homogeneous block | `template:` | `params`, optional repeated-stack `streams`, `checkpoint: {prefix, index}` when repeated, `checkpoint: {root}` when not |
| heterogeneous block | `per_layer:` | repeated-stack `streams`, `checkpoint: {prefix, index}`; `repeat` is required |
| auxiliary group | `group: true` | non-empty `layers`, `aux_kind`, `checkpoint: {prefix, index}` when it contains an indexed block stack, `checkpoint: {root}` when its tensors hang off another path than the model's |

`name`, `in` and `repeat` are shared structural fields.

**Checkpoint addresses.** Every stored tensor's name is composed from the
spec and must equal a checkpoint header name. The spec's top-level
`checkpoint: { root: model }` names the path its entries hang off (absent: the
file root). A model-level op or stack lives at its `name` under the root of its
scope — the model's, or its group's `checkpoint.root`. An op elsewhere declares
its full `checkpoint: { path: lm_head }`; a stack elsewhere declares its full
`checkpoint: { prefix: model.mtp.layers }`. A repeated op or graph node whose
copies the checkpoint indexes declares `checkpoint: { index: local }`; a repeat
without it is stored as one module. A `template:` call nothing repeats has no
index, even inside an indexed group: its component's names hang off its scope's
root, or off the `checkpoint: { root: ... }` it declares — so a module that sits
beside a group's indexed layers (`model.draft.fusion.*` next to
`model.draft.<i>.*`) is a component called once with that root, its nodes named
relative to it, not a run of direct ops each spelling the full path. `streams` is valid only
on a repeated template/per-layer/aux stack. Do not put op geometry on a template
or group, and do not put template `params` on a direct op.

Every repeated stack that hands its block's output to its next iteration
declares that hidden state as its carried stream:
`streams: { hidden: { carries: true } }`. There is no default: a
stack without it hands nothing on. Declare it where the repeat is — on the
`template:` / `per_layer:` stack, on a repeated `group:` whose block does not
repeat itself (the MTP example below), or, under a repeated group around a
repeated stack, on the inner stack. The stack starts it from `inputs[0]` and
binds it to its block's FIRST formal every iteration.

What the model receives from outside its graph is declared once, at the top
level, in `model_inputs:` — one entry per tensor, by the name a layer's
`inputs:` reads it by. Token ids are `{ kind: tokens }`. An image is
`{ kind: image, axes: [n_img, img_h, img_w] }` — images per request, then the
height and width the processor hands the tower — a video
`{ kind: video, axes: [n_vid, vid_frames, vid_h, vid_w] }` — videos per
request, the frames the processor samples, the processed height and width —
and an audio clip `{ kind: audio, axes: [n_aud, s_audio], width: <frame
width> }`, with no `width` when each frame is one id per codebook. A side the
config fixes is written in `fixed: { img_h: <dim>, img_w: <dim> }`; the items
of each item of another input (an image's crops) name it in `per_item_of:`.

An encoder group reads its image through a `patchify` layer, which declares
how the processor cuts the image into rows, each value read off the
processor source and cited in `source_anchor:`:

```yaml
model_inputs:
  input_ids: { kind: tokens }
  pixel_values: { kind: image, axes: [n_img, img_h, img_w] }
model:
  - name: visual
    group: true
    aux_kind: vision
    layers:
      - name: visual.patchify
        op: patchify
        inputs: [pixel_values]
        # patch side, temporal patch, channels, and the resize factor every
        # processed side is a multiple of (smart_resize's patch * merge)
        dims: { patch: s_patch, temporal: s_patch_temporal, channels: n_vis_channels, align: "s_patch * n_spatial_merge" }
        # one attention segment per item, or per temporal patch (cu_seqlens)
        segment: temporal_patch
        source_anchor: "<processor file:lines for the factor and patch order>; <cu_seqlens file:lines>"
      - { name: visual.patch_embed.proj, op: linear, inputs: [visual.patchify], dims: { in: "n_vis_channels * s_patch_temporal * s_patch * s_patch", out: d_vis_model } }
```

Where the processor pads every image to a fixed row count, a `pad_rows`
layer (`dims: { rows: <count> }`) follows `patchify`, and an `unpad` layer
(`inputs: [<padded output>, <patchify>]`) sits where the framework strips the
padding, both with `source_anchor:`. Rows an encoder computes from its own
grid — one separator per merged row, a learned row per image — are counted
from the image's axes (`img_h // (s_patch * n_spatial_merge)`, `B * n_img`).

Where the forward hands a second input to the same module — a model's videos
through its image tower (`get_video_features` calling `get_image_features`),
an image's crops through the vision model — the spec writes the call, not a
copy: `- { name: <call>, call: <the earlier layer>, inputs: [<its input>] }`,
beside the called layer (a `group:` at the top level, or a `template:` call in
the same group). The call runs the called layer as it is, with its weights;
its layers take the call's name in place of the last segment of their shared
module path (`model.visual.merger` → `model.video.merger`), each stream the
called layer exports takes `_<call>` (`deepstack_feat` →
`deepstack_feat_video`), and `dims: { <symbol>: <symbol> }` rebinds a symbol
the other input sets on the activations (the soft tokens a video frame is
padded to); every layer holding a weight keeps the called one's dims and
placement, and every `repeat:` its count. The called layer is an encoder
`group:` or a `template:` call beside the call in its group, one that calls
nothing itself. An image
is a video of one frame, so the image tower's `patchify` cuts both: one
attention segment per temporal patch.
A `group_items`
layer (`dims: { items: <per item>, groups: <items> }`) lays each item's sub-items'
rows end to end.

Every edge from an encoder into the language model declares the prompt
positions it fills: `prompt_slots: { domain: <media input>, defines: true,
source_anchor: ... }` on the layer that writes the encoder's rows into the
token embeddings (`scatter_rows_replace`, the framework's
`masked_scatter` / `_merge_multimodal_embeddings`), and `defines: false` on an
edge that adds onto the same positions (a DeepStack stream's `read: each`
entry carries it). Each media input has exactly one `defines: true`. One
scatter writes every media input's rows (`inputs: [<embeddings>, <image
rows>, <video rows>]`), and then its `prompt_slots` names each encoder output
it reads: `prompt_slots: { <name it reads>: { domain, defines,
source_anchor }, ... }`.

The `model:` entries, one per kind of layer:

```yaml
# Simple op layer
- { name: embed_tokens, op: embedding, inputs: [input_ids], boundary_role: input_embedding, dims: { n_vocab: n_vocab, d_model: d_model }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }] }
- { name: layers, repeat: n_layers, template: block, inputs: [embed_tokens], streams: { hidden: { carries: true } } }
- { name: final_norm, op: rmsnorm, dim: d_model, weight_axes: [{ replicated: true }], inputs: [layers] }
- { name: lm_head, op: linear, boundary_role: output_head, dims: { in: d_model, out: n_vocab }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }], inputs: [final_norm] }
- { name: lm_head, op: linear, boundary_role: output_head, dims: { in: d_model, out: n_vocab }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }], tied_weight: { to: embed_tokens, when: config.tie_word_embeddings }, inputs: [final_norm] }

# Exactly one top-level direct embedding is the input boundary. Declare every
# externally returned logits projection as an output_head. A tied_weight row
# must be an output_head, must point to an EARLIER direct weight owner, and the
# two declared weight shapes must be symbolically identical. A conditional tie
# still declares its head placement for configurations where the condition is false.
# When the checkpoint also ships the head's own tensor (e.g. `lm_head.weight`
# beside `embed_tokens.weight`) and the serving loader skips it, declare that
# copy on the tie with `checkpoint_copy: { src }`, citing the loader line that
# skips it. The tie then sets parameters, compute and deployed weight bytes,
# and the checkpoint byte total and tensor-name match count the stored copy.
# Declare `checkpoint_copy` exactly when the checkpoint has that tensor.
- { name: lm_head, op: linear, boundary_role: output_head, dims: { in: d_model, out: n_vocab }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }], tied_weight: { to: embed_tokens, when: config.tie_word_embeddings, checkpoint_copy: { src: "modeling_hunyuan_vl.py:1146 _tied_weights_keys = {'lm_head.weight': 'model.language_model.embed_tokens.weight'}" } }, inputs: [final_norm] }

# Every layer names what it reads in `inputs:`/`in:`: an earlier layer, a
# stream, or a name in `model_inputs:`. A zero-input op (a parameter, a
# buffer) names nothing.
# A layer names its source explicitly (lm_head reads `norm`):
- { name: lm_head, op: linear, boundary_role: output_head, in: final_norm, dims: { in: d_model, out: n_vocab }, weight_axes: [{ world: attention, parts: n_vocab, fit: pad }, { replicated: true }] }

# An encoder group reads its own model input — an image through its
# `patchify` — and names no `inputs:` itself.
- name: vision
  group: true
  aux_kind: vision
  layers:
    - { name: vision.patchify, op: patchify, inputs: [pixel_values], dims: { patch: s_patch, temporal: "1", channels: n_vis_channels, align: "s_patch * n_spatial_merge" }, segment: item, source_anchor: "<processor file:lines>" }
    - { name: vision, template: vision_tower, inputs: [vision.patchify] }

# Homogeneous repeated block (all layers identical)
- { name: layers, repeat: n_layers, template: block, inputs: [embed_tokens], streams: { hidden: { carries: true } } }

# Heterogeneous layers — `per_layer`: generators + a type→template map.
# Each `assign` entry is a GENERATOR that stamps a `type` onto a set of
# layers (declaration order, first-wins); `map` binds each type to its
# template. Generators produce a per-layer type-list; the map turns types
# into templates.
- name: layers
  repeat: n_layers
  inputs: [embed_tokens]
  streams: { hidden: { carries: true } }
  per_layer:
    assign:
      - { type: dense, range: ["0", "n_dense_layers - 1"] }   # contiguous prefix
      - { type: moe,   rest: true }               # everything else
    map:
      dense: { attention: mla_attention, ffn: swiglu_ffn }
      moe:   { attention: mla_attention, ffn: moe_ffn }

# Config-list-driven — the model exposes a per-layer type list in its config
- name: layers
  repeat: n_layers
  inputs: [embed_tokens]
  streams: { hidden: { carries: true } }
  per_layer:
    assign:
      - { type: dense,  range: ["0", "n_dense_layers - 1"] }
      - { type: full,   from: config.indexer_types, is: full }    # read config list
      - { type: shared, from: config.indexer_types, is: shared }
      - { type: full,   rest: true }              # fallback if the field is absent
    map:
      dense:  { attention: mla_attention,        ffn: swiglu_ffn }
      full:   { attention: mla_attention,        ffn: moe_ffn }
      shared: { attention: mla_shared_attention, ffn: moe_ffn }

# Indexed auxiliary stack (e.g. MTP). The checkpoint address is mandatory:
# path spelling (`prefix`, in full from the file root) and index ownership
# (`index`) are separate facts.
- name: mtp
  group: true
  aux_kind: mtp
  in: final_norm
  repeat: n_mtp_layers
  streams: { hidden: { carries: true } }   # on the group: the block does not repeat itself
  checkpoint: { prefix: mtp.layers, index: local }
  layers:
    - { name: mtp, template: mtp_block, in: final_norm }
```

### Generators (entries in `assign`, first-wins by declaration order)
- `range: [start, end]` — inclusive contiguous range; expressions ok ("n_layers - 1").
- `layers: ["0", "5", "11"]` — explicit layer-index list.
- `stride: "2", offset: "0"` — every N-th layer (size-independent).
- `from: config.<field>, is: <value>` — read the model's per-layer type LIST
  from config and select layers whose entry == value. **Prefer this over a
  hardcoded index list whenever the config exposes a schedule list**
  (`indexer_types` / `layer_types` / `hybrid_layer_pattern` / …). Values match
  by string, so int-encoded lists (0/1) work too.
- `from: config.<field>` alone — the config lists the layer numbers themselves
  (`fullatt_block_indexes`); add `base: 1` when that list counts layers from 1
  (`ple_layer_ids`, read as `layer_idx + 1 in ...`).
- `within: { from: config.<field>, is: <value> }` or `within: { from:
  config.<field>, base: 1 }` beside another selector — the layers must also
  lie in this second config-listed set. Use it when a layer's kind is the
  product of two config lists: one generator per combination
  (`layer_types` × `ple_layer_ids`: linear+PLE, full+PLE, then plain linear
  and full).
- `rest: true` — all layers not claimed by earlier generators.
  It must be the final generator.

### `map` — type → template
Each non-empty `type` must have exactly one used `map` row. A row binds
`attention` + `ffn` (auto-generates a `{attention}__{ffn}` pre_norm block), a
`template:` (a named/inline block), or a parameterized template plus the
attention/FFN slot bindings it accepts. Optional `params:` / `dims:` override
the selected call. Unused map rows are rejected. `map` applies inside group
sub-layers too (author their heterogeneity the same way).

### Family authoring rules and the schedule gate
- **One spec serves a model FAMILY** (matched by `model_type` across sizes /
  checkpoints — e.g. one spec for gemma-2-9b AND -27b). Use size-independent
  generators (`range` / `stride` / `from: config.X`); do NOT hardcode a
  per-layer type list of one family member's length — it breaks the others.
- **Exhaustiveness**: every distinct value present in a config list you read
  via `from:` must be handled by a `map` type — a `rest:` does NOT count.
  An unhandled value fails validation.
- **Config coverage is an author review, not a validator gate**: inspect every
  architecturally significant config field (`*_types`, `index_*`,
  `sliding_window`, expert counts, `first_k_dense_replace`, ...). The mechanical
  schedule gate is narrower: `run_layer_schedule_check` rejects an unhandled
  value in a config list named by a `from:` generator. The loader does not
  otherwise reject an unused config field. There is one separate model-level
  gate: when the resolved graph contains a reachable component whose literal
  role is `attention`, an active `config.sliding_window` requires at least one
  such component to have a bounded KV stream. A graph with no `role: attention`
  component skips this check. If source proves the config field is irrelevant,
  `validation_exemptions: {sliding_window: "reason"}` exempts this one check;
  there is no general unused-config allowlist.

### Custom block graphs (`blocks:`) — for non-standard topologies
Models whose per-layer structure does NOT fit the standard pre_norm pattern
(attn → add → ffn → add) — e.g. dual attention branches + dual dense FFN +
MoE stacked as residuals — should declare an inline block graph:
```yaml
blocks:
  my_layer:
    graph:
      input_layernorm.0:  { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      self_attn.0:        { op: mla_attention, in: input_layernorm.0 }
      attn0_residual:     { op: add, in: [x, self_attn.0], dim: d_model }
      # ... more ops / component invocations ...
      out:                { op: add, in: [prev, mlp], dim: d_model }
    outputs: [out]
model:
  - { name: layers, template: my_layer, repeat: n_layers, inputs: [embed_tokens], streams: { hidden: { carries: true } } }
```
Component invocations (`op: mla_attention`, `op: moe_ffn`, etc.) inside the
block graph are recursively evaluated — DO NOT expand them into raw
linear/rmsnorm leaves. Express a non-standard main transformer stack with an
explicit block graph, either inline under `blocks:` as above or as an activated
draft block component. `group:` is reserved for auxiliary stacks like MTP.

### Node captions (`formula:`)

Every graph node accepts a display-only `formula:` — the graph card shows it
in place of the op registry's generic formula. Write it on a non-trivial node
whenever THIS call site's shape story is not fully readable from the generic
one: which axis a table broadcasts over, what gets unflattened / gathered /
summed first, which axis a softmax reduces. State the concrete semantics with
the component's real axis names, as the source code runs it:

```yaml
scores:
  op: position_table_add
  in: [wgate, ape]
  dim: Cdim_io
  formula: "y[.., p, :] = unflatten(wgate, m)[.., p, :] + ape[p, :]   ([B,S,D] → [B,S/m,m,D], p = position in window)"
```

A plain linear / rmsnorm / activation needs none — its registry formula
already says everything true of it. `formula:` changes no cost, no shape
validation, and no spec fingerprint; it only captions the card. Never encode
new semantics here: the engine never reads it, so a formula that disagrees
with the node's declared dims is a review finding, not a behavior.
Every symbol in the caption must carry its meaning at first use —
`m = compress_ratio = tokens pooled per slot`, `p = position within that
window` — a formula whose `m` or axis names are undefined has failed its
reader. State the reduction axis explicitly when it differs from what the
registry's generic formula implies (e.g. a softmax that normalizes over a
window axis, not the feature dim).

### Cross-iteration state (`streams:`)

Every value that flows across layers is a stream, declared on the stack.
How it attaches decides what else is written:

- **At the block boundary** — the hidden state the stack hands from one
  iteration to the next: `hidden: { carries: true }` (see
  "Model layer types"). A component never names it: it arrives on the
  block's first formal and leaves as the block's output. A component the
  stack binds naming it in `reads:` / `stream_out:` is refused, and so is
  a second `carries:` stream on one stack. Its layout settles at every
  iteration exit.
  The name is local to the stack — a backbone and an MTP head may both call
  theirs `hidden`.
- **At an inner node** — any other value threaded across layers. The
  component that writes it `stream_out`s it and the one that reads it
  `reads` it; each reading layer's declaration names the layer whose write
  it reads (`reads_from`, below). The stack whose layers write it lists it
  under `streams:`; the name is that stack's, and it may not also be a
  model layer's name or an appended stream's. A component's formal
  (`inputs:`) and a stream it reads may not share a name.
- **Appended** — the values a stack hands on after chosen iterations, read
  element by element by another stack (see "Appended streams" below). Its
  name is model-wide; its `from:` entry is its one declaration.

A repeated `model:` stack declares the catalog of inner streams beside its
carried one:

```yaml
streams:
  aux_state:
    src: "MyModel.forward threads `aux_state` across layers"
    shape: [B, s_query, d_stream]    # optional
```

| field | meaning |
|---|---|
| (key) | stream name; a component `reads` / `in:` this string |
| `src` | forward-source anchor (same truthfulness treatment as an op `source_anchor`) |
| `shape` | optional publish-gate: dim expressions compared to the `stream_out` node's evaluated shape (B=1, S=4096; per-variant via `per_layer.assign[].dims`) |

The producing component declares `stream_out: {<name>: <node>}`. The
consuming component declares `reads: [<name>]`, which makes the name
legal in any node's `in:`. Producer and consumer may be the same
component or different per-layer variants.
Eval binds a reader to the producer tensor — the output of the node that
produced what it reads. `shape` is not an eval seed of a whole model.

**Which layer a reader reads: `reads_from`.** Every declaration that selects
reading layers — the `per_layer.assign` entry, or a `template:` stack whose
every layer reads — names, per stream, the layer of the stack whose write
those layers read:

```yaml
per_layer:
  assign:
    - { type: dense,  range: ["0", "n_dense_layers - 1"] }
    - { type: full,   from: "config.indexer_types ?? absent", is: full }
    - { type: shared, from: "config.indexer_types ?? absent", is: shared,
        reads_from: { dsa_topk: { nearest: { from: "config.indexer_types ?? absent", is: full } } } }
```

| value | the layer read |
|---|---|
| `"i - k"` | k layers before the reading layer `i`, in its stack (`"i - 1"`: the layer before) |
| an integer or dim expression (`"20"`, `"n_dense_layers - 1"`) | that layer of the reading layer's stack |
| `{ nearest: { from: config.<list>, is: <value> } }` | the nearest earlier layer whose entry in a per-layer config list is `<value>` — the selector `per_layer.assign` uses |
| `{ nearest: { from: config.<list> } }` | the nearest earlier layer of a config list of layer numbers (one number is a set of one) |
| `{ stack: <path>, layer: <number> }` | that layer of another stack; `<path>` is its layer names from the top of `model:` joined by `/` (`layers`, `mtp/mtp`). When that stack loops, `<number>` is its stored layer and the read is that layer's last execution; this form takes no `pass:` |
| `{ map: { "<reading layer>": <source>, … } }` | each reading layer's source written out, one entry per layer the declaration selects; a source is a layer number or `{ stack, layer }` |
| `{ map: config.<list> }` | a config list of source layer numbers, one per selected layer in ascending order |
| `{ pass: n }`, alone or with `nearest:` / `layer:` | in a looped stack (`loop:`): pass n of the reading layer's own stored layer, or of the layer `nearest` / `layer` names |

The map is the general form: every other value is shorthand for the map it
gives over the selected layers, and all of them resolve the same way.

Layers are numbered as `per_layer.assign` selects them: the layer of the
stack as `repeat:` counts it. In a looped stack (below) that is the stored
layer — the checkpoint index — and a value that names a layer reads its
latest execution before the reader; `{ pass: n }` reads pass n instead. The named layer writes the stream:
a component on it `stream_out`s the stream from a node. In a `nearest` set
every layer writes it, and every reading layer has one before it.

A looped stack runs a range of its layers more than once per forward pass
with the same weights (a Loop / recurrent-depth transformer):
`loop: { layers: [<first>, <last>], passes: <n>, gate: { template: <block>,
checkpoint: { prefix: <path> } } }` on the stack, `repeat:` still counting
stored layers. A `per_layer.assign` entry with `pass: <k>` selects only the
loop layers' pass-k executions (a later pass binds its own components; its
weights are the stored layer's, and it may read no weight the stored layer
lacks); entries without `pass:` select every execution of their layers, so
write the later-pass entries first. The optional gate runs after every pass:
its block's first input is the value leaving the pass, its second the value
entering it, and it stores one copy per pass.

Write the source the forward source and the config state: a loop that hands
every layer what the layer before returned is `"i - 1"`; layers that reuse
the value of the most recent source layer read `{ nearest: ... }` of the
config field that marks those source layers (`indexer_types` entries
`full`, `kv_source_layer_ids`); a fixed source layer is its number. When no
shorthand states it, write the map. A layer reads one layer's write; a
layer that needs the values of several earlier layers at once reads an
appended stream (`read: concat` / `read: all`), not an inner one. When the
forward source seeds the value before the loop (DeepSeek-V4.1's
`pre_mix = make_identity_pre_mix(h, …)`), declare that model-level op as the
stream's `seed:` (`hc_pre_mix: { seed: hc_seed }`): a first layer's `"i - 1"`
then reads the seed, and every layer runs one template. Without a seed, a
first layer with nothing before it to read runs a variant that does not read
the stream. A model-level `op:` after the stack that names the stream in
`inputs:` reads its final value, the last layer's write (the exit reduce after
the loop), so no last-layer variant is needed for it. A
template is evaluated once for all its layers, so the layers one
declaration selects read sources of one shape and cache law.

If that `stream_out` node is also in `cache.entries`, readers inherit
its KV length as `s_kv`.

```yaml
blocks:
  my_first_layer:                 # layer 0: nothing before it to read
    graph:
      norm:     { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      attn:     { op: mla_attention, in: norm }
      attn_res: { op: add, in: [x, attn], dim: d_model }
      next_aux: { op: linear, in: attn_res, dims: { in: d_model, out: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
      out:      { op: add, in: [attn_res, next_aux], dim: d_model }
    outputs: [out]
    stream_out: { aux_state: next_aux }
  my_layer:
    reads: [aux_state]
    graph:
      norm:     { op: rmsnorm, in: x, dim: d_model, weight_axes: [{ replicated: true }] }
      attn:     { op: mla_attention, in: norm }
      gated:    { op: mul, in: [attn, aux_state], dim: d_model }
      attn_res: { op: add, in: [x, gated], dim: d_model }
      next_aux: { op: linear, in: attn_res, dims: { in: d_model, out: d_model }, weight_axes: [{ replicated: true }, { replicated: true }] }
      out:      { op: add, in: [attn_res, next_aux], dim: d_model }
    outputs: [out]
    stream_out: { aux_state: next_aux }
model:
  - name: layers
    repeat: n_layers
    inputs: [embed_tokens]
    streams:
      hidden: { carries: true }
      aux_state:
        src: "MyModel.forward threads `aux_state` across layers"
    per_layer:
      assign:
        - { type: first, layers: ["0"] }
        - { type: rest, rest: true, reads_from: { aux_state: "i - 1" } }
      map:
        first: { template: my_first_layer }
        rest:  { template: my_layer }
```

A node that is computed but neither in `outputs:` nor a `stream_out`
target is a `dead_end_node` warning. The cross-layer pass rejects an
inner stream that lacks a producer or consumer, a `reads` of a stream no
stack declares, and a `src` that does not occur in the model's recorded
forward source; those are errors.

`reads_from` belongs to the whole model: in every spec, every declaration
that selects reading layers carries it, and a model whose reading layer has
none is refused with the layers that write the stream. One evaluation reads
a stream without it — a single component evaluated alone, one call site run
as a model of its own (the Playground module bench, a per-component check).
There no other layer runs, so the component reads each stream at the
`shape:` the stack declaring it gives under `streams:`, evaluated with the
call site's dims; give that `shape:` on every inner stream a component
reads, so the component evaluates alone.

Two `model:` stacks sharing an inner stream: the writing stack lists the name
under `streams:`; the reading stack's component `reads` it.

```yaml
model:
  - name: encoder
    repeat: n_encoder_layers
    template: encoder_block      # stream_out + cache.entries
    inputs: [embed_tokens]
    streams:
      hidden: { carries: true }
      kv:
        src: "Model.forward: encoder writes KV; decoder reads it"
  - name: decoder
    repeat: n_decoder_layers
    template: decoder_block      # reads: [kv]
    inputs: [encoder]
    streams: { hidden: { carries: true } }
```

### Appended streams: values a stack hands on after chosen iterations

When the forward collects a value after chosen iterations of a repeated
stack and hands element j to another module's j-th call (Qwen3-VL's
DeepStack: vision blocks 8/16/24 each feed a merger, and merger j's output
is added to the visual rows after decoder layer j), the stack declares an
**appended stream**, and the reader pairs with it element by element.

```yaml
dims:
  n_deepstack: len(config.vision_config.deepstack_visual_indexes)   # counts a config list
model:
  - name: visual
    group: true
    aux_kind: vision
    checkpoint: { prefix: model.visual.blocks, index: local }
    layers:
      # ... patch embed ...
      - name: visual.blocks
        template: qwen35_vision_block
        repeat: n_vis_layers
        inputs: [visual.pos_add]
        streams:
          hidden: { carries: true }
          deepstack: { from: hidden, after: config.vision_config.deepstack_visual_indexes,
                       index: block, src: "if layer_num in self.deepstack_visual_indexes:" }
      # Reads `deepstack` at its block boundary: iteration j binds element j.
      - name: visual.deepstack_merger_list
        template: qwen3_vl_deepstack_merger
        repeat: n_deepstack             # must equal the stream's element count
        checkpoint: { prefix: model.visual.deepstack_merger_list, index: local }
        inputs: [deepstack]
        streams:
          deepstack_feat: { from: out, after: every, index: block,
                            src: "deepstack_feature_lists.append(deepstack_feature)" }
      # ... final merger ...
  - { name: language_model.embed_merge, op: scatter_rows_replace,
      inputs: [language_model.embed_tokens, visual.merger.linear_fc2] }
  # Reads `deepstack_feat` at an inner node: the j-th layer whose component
  # `reads: [deepstack_feat]` reads element j.
  - name: language_model.layers
    repeat: n_layers
    inputs: [language_model.embed_merge]
    streams:
      hidden: { carries: true }
      deepstack_feat: { read: each }
    per_layer:
      assign:
        - { type: ds, range: ["0", "n_deepstack - 1"] }
        - { type: moe, rest: true }
      map:
        ds: { template: qwen3_vl_deepstack_decoder_block, attention: gqa_qknorm_attention, ffn: moe_ffn_fused_packed }
        moe: { attention: gqa_qknorm_attention, ffn: moe_ffn_fused_packed }
```

| field | meaning |
|---|---|
| `from` | what is appended: the stack's carried stream, or `out` (the block's output) on a stack that carries none |
| `after` | list of integers or dim expressions (`[2, "n_layers // 2", "n_layers - 3"]`; `//` floors), `config.<path>` naming a list or one integer (a list of one) (`?? absent`: a config without it appends nothing), or `every` |
| `index` | required with `after`. `block`: i is "after block i" (Qwen `deepstack_visual_indexes`). `hidden_states`: HF `hidden_states[i]`, 0 is the stack's input (Llava `vision_feature_layer`); only with `from: <carried>`. Negatives count from the end. `after: every` counts blocks. |
| `before` | instead of `after` + `index`: append the ENTRY of the chosen iterations (iteration 0's entry is the stack's input). Integer list (iteration numbers, no negatives), `config.<path>`, `every`, or `{ every: <dim> }` (0, N, 2N, …). Only with `from: <carried>`. |
| `read: all` | on the appending declaration: the stack reads its own stream's prefix (below) |
| `order: declared` | with an explicit list only: the elements keep the list's order, which may descend or repeat an index (HF `vision_feature_layer: [-2, -5, -8, -11, 6]`). Without it the list must strictly ascend. Only a join reads such a stream. |
| `slice: { axis, start }` | publish each value from `start` on along `axis` — Llava's CLS crop `hs[:, 1:]` is `{ axis: 1, start: 1 }`. Only a join reads such a stream. |
| `src` | required: the forward line that appends |

- **Reading at the block boundary**: a stack names the stream in `inputs:`.
  It carries no stream (no `carries: true`) and its `repeat` equals the
  number of elements; its other `inputs:` are shared by every iteration.
  With zero elements its `repeat` must evaluate to 0 (`len(config.X) ?? 0`).
- **Reading at an inner node**: the stack declares `name: { read: each }`
  and the component `reads: [name]` and names it in a node's `in:`. The
  layers whose component reads it number exactly the elements and are
  contiguous.
- A stack that carries no stream cannot be read by its name — append what
  it produces (`from: out`) and read the stream.
- **Reading its own prefix** (`read: all`): every layer whose component
  `reads: [name]` sees, as one `[B, S, m, D]` tensor, the m elements
  appended strictly before that layer's entry (`before: j` is visible from
  layer j + 1, `after: j` from layer j + 1). A layer that would see none
  (m = 0) must use a variant that does not read it. To use the history
  with the layer's own entry in it, the component appends it itself:
  `stack_append [H, x]` (m ≥ 1), or `stack_init [x]` for a history of one.
  A stack has at most one `read: all` stream. Layers are grouped by m
  automatically; do not split them by hand.
- A model-level `op:` that names an appended stream in `inputs:` reads all
  of its elements, the stream's final value, as a `[B, S, k, D]` history.
- **Joining it** (`torch.cat` of the elements): a later stack declares
  `name: { read: concat, axis: -1 }` and its component `reads: [name]`
  and names it in a node's `in:`; that node reads the k elements joined
  along `axis`, in element order. A model-level `op:` joins the stream it
  names when the layer declares `axis:`. Do not write the `concat` node:
  k is known only at resolve, and the engine lays the join (and each
  element's `slice` crop) down itself. One element is read as it is.
  Elements must agree on every axis but the joined one, which may differ
  in length along any axis. A join is refused at `pp > 1`.

Llava with a list `vision_feature_layer` (`modeling_llava.py`: each
`hidden_states[i]` cropped `[:, 1:]`, then `torch.cat(dim=-1)`), and an
EAGLE-3 head reading the backbone's auxiliary hidden states (layers
`(2, n // 2, n - 3)` when the config names none; the carried `hidden`
is the whole layer boundary, residual added):

```yaml
      - name: vision_tower.blocks
        ...
        streams:
          hidden: { carries: true }
          # `config.<path>` names a list, or an int `vision_feature_layer: -2` (one element, read as is).
          feature: { from: hidden, after: config.vision_feature_layer, index: hidden_states,
                     order: declared, slice: { axis: 1, start: 1 },
                     src: "hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]" }
  - { name: multi_modal_projector.linear_1, op: linear, inputs: [feature], axis: -1,
      dims: { in: "n_feature_layers * d_vis_model", out: d_model } }

  - name: layers
    streams:
      hidden: { carries: true }
      aux: { from: hidden, after: [2, "n_layers // 2", "n_layers - 3"], index: hidden_states,
             src: "aux_hidden_states.append(hidden_states + residual)" }
  - name: eagle ...           # group, aux_kind: speculative
      streams: { hidden: { carries: true }, aux: { read: concat, axis: -1 } }
```

Kimi-K3's AttnRes (`modeling_kimi_linear.py`): the entry of layers 0, K,
2K, … joins a history every later layer combines with its input:

```yaml
  - name: language_model.model.layers
    repeat: n_layers
    inputs: [language_model.model.embed_merge]
    streams:
      hidden: { carries: true }
      block_residual: { from: hidden, before: { every: Kar }, read: all,
                        src: "block_residual = torch.cat(" }
    per_layer: ...   # layer 0 (empty history, `stack_init`), layers K, 2K, … (`stack_append`), the rest
  - { name: language_model.model.output_attn_res, op: attn_res_combine,
      inputs: [language_model.model.layers, block_residual,
               language_model.model.output_attn_res_norm, language_model.model.output_attn_res_proj] }
```
- A second indexed stack in the group declares its own `checkpoint:`
  prefix. A stack outside the group reads a vision/audio stage by its name;
  merge image rows into the text embeddings with a model-level
  `scatter_rows_replace` first, so the decoder reads one input.

### Stream glossary

| term | meaning |
|---|---|
| stream | a value that flows across the iterations of a repeated stack, declared under that stack's `streams:`; the only way a value crosses layers |
| carried stream | `carries: true` — the stack's `hidden`: enters on the block's first formal, leaves as the block's output, its layout settled at every iteration exit. One per stack, local to it |
| inner stream | none of `carries` / `from` / `read` — a component `stream_out`s it from a node, another `reads` it (`dsa_topk`; YOCO shared KV from a cached node). Model-wide name |
| appended stream | `from` + `after` + `index`, or `from` + `before` — the k values the stack appends at chosen iterations. Model-wide name |
| `read: each` / `all` / `concat` | reader modes of an appended stream: element j per reader; the appending stack's own `[B, S, m, D]` history; the k elements joined along `axis` |

Not a stream in this sense: a **KV stream** (a cached sequence of KV
positions; `kv_stream_concat` joins two, and the sliding-window check
wants a bounded one), the **token stream** a `dp` / `cp` world cuts, and
"streams" as a verb (a kernel streams bytes).

### Views and `decode_variant`

Prefill and decode are the two analyzes (different Binding; decode
`s_query=1`). Activation sequence length is the inbound tensor's
sequence axis. Cache read length is Binding `s_kv`.

`slice`, `identity`, and `sample` are view ops (zero FLOPs, zero
params). Downstream `in:` is the view's output tensor.

`decode_variant: <component>` on a block names the graph used at
decode when that graph is not the prefill graph. `rebind_decode_variants`
substitutes components that declared the field.

`slice` keeps a window of one axis and states it at the call site:
`dims: { out: <kept width> }`, with `start: <offset>` beside it when the window
does not begin at 0 (`axis` default last; sequence on `(B, s_ctx, d)` is
`axis: 1`; `out` is a dim such as `s_window`).

`view_merge` merges the last two axes of the tensor it reads into one, whose
width is their product: `{ op: view_merge, in: attn, dims: { n_heads:
n_heads, d_head: d_head } }` turns `(B, s_ctx, n_heads, d_head)` into
`(B, s_ctx, n_heads*d_head)`. Its `dims` name no width.

`view_split` has no default axis: every `view_split` declares `axis:`,
and a node without one is rejected. Its dims name the two factors in
order (outer, inner) and their product must equal the split axis:
`axis: -1, dims: { n_heads: n_heads, d_head: d_head }` splits
`(B, s_ctx, n_heads*d_head)` into `(B, s_ctx, n_heads, d_head)`;
`axis: 1, dims: { s_slots: "floor(s_ctx / m)", window: m }` folds the
sequence into windows after a `slice` to the complete-window prefix
`floor(s_ctx / m) * m`. Never round inside the split. `identity` preserves shape. A decode twin is a real graph
with at least one node — empty `outputs: [x]` is rejected.

Analyze inserts `sample` before every `boundary_role: output_head`. A
component's own logits projection (an MTP layer's private `shared_head.head`)
declares the same field on its graph node and is sampled the same way.
Binding `s_ctx` is the full context; the head's inbound sequence axis
is `s_sampled`.

```yaml
blocks:
  decoder_prefill:
    decode_variant: decoder_decode
    graph:
      window: { op: slice, in: x, axis: 1, dims: { out: s_window } }
      attn:   { op: mla_attention, in: window }
      out:    { op: add, in: [x, attn], dim: d_model }
    outputs: [out]
  decoder_decode:
    graph:
      y: { op: identity, in: x }
    outputs: [y]
```

MLA absorbed decode is the same field (`decode_variant:
mla_attention_absorbed`). A component whose prefill also runs that form says
so with `variant_phases: [prefill, decode]` (DSA: its query reads its top-k
rows of the cached latent, so serving prefills it absorbed); omitted, only
decode runs it. A decode twin that must not recompute still
declares the same `stream_out` / `cache.entries` so readers keep the
stream.

### Component composition
Components can reference other components. For example, `moe_ffn` calls the
`gated_experts` bank template on the routed rows and `swiglu_ffn` for its
shared experts:
```yaml
experts:        { op: gated_experts, in: permute, dims: { d_model: d_model, d_ffn: d_ffn_expert }, expert_weight: unrolled }
shared_experts: { op: swiglu_ffn, role: ffn_moe, in: x, dims: { d_model: d_model, d_ffn: d_ffn_expert }, repeat: n_shared_experts }
```

### Roles on architecture-bearing components
At the YAML top level alongside `graph:`, architecture-bearing components
declare a `role:`. The same vocabulary names a model's `components:` slots and
a node-level `role:` override; the loader rejects any name it does not list.
Every registered role (`backend/arch/roles.yaml`):

{roles_table}

A component declares the broad role of what it is: an SWA implementation
declares `role: attention` and the `attention_swa` slot binds it to the SWA
layers; the `norm` slot binds a primitive such as `rmsnorm` rather than a
roled component. A role with a `refines` entry is a part of the broader one,
written on nodes the checkpoint stores at another precision (see Quant
context below).

The engine uses `role:` to attribute params/FLOPs into the
`attention_params` / `ffn_params` columns. A weight-bearing component whose
leaves inherit no role is omitted from those tallies and receives a non-gating
`component_missing_role` warning. The schema permits roleless utility and wrapper
components whose leaves inherit a role from an ancestor; missing descriptions
also produce non-gating structural warnings rather than load errors.

```yaml
# the component is named by its file stem — my_attention_variant.yaml
role: attention      # needed when this component owns attention weights
graph: { ... }
```

When a graph has to inline an FFN as raw primitives (rare; only when
checkpoint tensor naming forces it — e.g. `block_sparse_moe.experts.w1`
in MiniMax-M2), tag each unrolled node with **call-site `role:`** on the
GraphNode itself:
```yaml
block_sparse_moe.experts.w1: { op: linear, role: ffn_moe, in: post_attn, ... }
```
Here `...` abbreviates the required `dims`, `shape`, `weight_axes`, and
`expert_weight` declarations; it is not literal authorable YAML.
Prefer wrapping in a sub-component so the role propagates automatically;
the call-site override is for cases where the standard component library
can't match the checkpoint's tensor namespace.

### Copies of one node: `chain:` or `repeat:`

A node that exists in several copies says which arrangement the forward puts
them in. The two cost the same and differ in what a reader of the node moves,
so the keyword is the only thing that says which:

- `chain: n_vis_layers` — the copies run one after another: copy *i* reads
  copy *i−1*'s output and a reader of this node sees only the LAST
  (`for blk in blocks: x = blk(x)` — a vision or audio tower's block stack;
  also a running sum folded over a repeated producer, `acc += x`).
- `repeat: n_shared_experts` — the copies sit side by side: every copy reads
  this node's own `in:` and a reader sees ALL of them (a shared-expert bank
  the residual `add` reads). Total copies stored → params multiplier.

Declaring both on one node is refused. A ROUTED expert bank is written with
its experts as the leading axis of its weights, by calling a bank template
(`gated_experts`, `fused_gated_experts`, … — see "Persistent distributed
endpoints"); which experts a token runs is the router's top-k.

### Config-switched steps (`when:`)

A step the forward runs only under a config flag (`if self.norm_topk_prob:
weights /= weights.sum(...)`) is a node with `when: <dim>`. The dim is a 0/1
switch the model spec binds from config, with the modeling code's default as
the fallback (`n_topk_renorm: config.norm_topk_prob ?? "0"`). At 0 the node is
not built and its readers read its first input instead:

```yaml
topk_norm:  { op: normalize_sum_one, in: topk_gather, dims: { in: n_experts_active }, when: n_topk_renorm }
topk_scale: { op: scale_const, in: topk_norm, dims: { in: n_experts_active } }
```

The switch must be the source's own condition. A step guarded by
`if config.x is not None` binds `present(config.x) ?? <class default: 1 or 0>`
(1 when the config sets a value, 0 when it sets null) — never `config.x`
itself, whose 0 would drop a step the source still runs and whose fractional
value is not a dim. A step the source always runs binds the literal `"1"`.

Every switch a component you call reads is bound in the model's `dims:`, and
so are the group counts beside it. A router whose forward multiplies its top-k
weights by `routed_scaling_factor` binds `n_routed_scale: "1"` (the factor's
value is not the switch), and one with no such step binds `"0"`. A
group-limited router binds `n_expert_groups: config.n_group` and
`n_groups_active: config.topk_group`; one with no group limit binds both to
`"1"`. Each binding cites the source line it comes from.

A weightless primitive carries `when:`. A shape-preserving one hands its first
input on at 0; one that changes the shape is simply absent at 0, together with
the steps it feeds, which carry the same `when:` — a group limit's group score
and group top-k beside the mask that reads them:

```yaml
group_max:  { op: group_max, in: scores, dims: { in: n_route_targets, out: n_expert_groups }, when: "n_expert_groups - n_groups_active" }
group_topk: { op: topk, in: group_max, dims: { in: n_expert_groups, k: n_groups_active }, when: "n_expert_groups - n_groups_active" }
group_mask: { op: group_mask_apply, in: [scores, group_topk], dims: { in: n_route_targets, n_groups_active: n_groups_active }, when: "n_expert_groups - n_groups_active" }
```

A component may name a shape-preserving switched step as its output (at 0 the
component returns the step's first input); a block template may not. `repeat: 0` is different: it removes
a branch (a zero-count shared expert) and its readers lose that operand.

### A stored parameter with no reader in the graph (`stored_without_graph_reader`)

Every stored parameter has a reader in the graph, the node the forward source
feeds it to. Two kinds of stored parameter have none, and each is written as
`parameter_vector` / `parameter_tensor` with a declaration that names its kind,
says why, and cites the source:

- `kind: forward_unread` — the forward never reads it; the framework loads it
  and ignores it.
- `kind: unmodelled_reader` — the forward reads it in a step the graph does
  not model (integer index math on the token ids, such as an n-gram id hash).

```yaml
self_attention_res_norm.weight:
  op: parameter_vector
  dim: d_model
  weight_axes: [{ replicated: true }]
  stored_without_graph_reader:
    kind: forward_unread
    reason: "layer 0's history is empty, so no combine runs before attention"
    source_anchor: "modeling_kimi_linear.py:986-993"
```

The tensor is stored and counted like any other weight; no graph node reads
it. Validate holds the declaration to the graph in both directions.

### A weight the framework keeps in host memory (`host_offload`)

Some frameworks keep a large lookup table in pinned host memory and gather its
rows over UVA instead of holding it on the GPU — vLLM's Engram `cpu_offload`
does this for the n-gram embedding tables, on by default. When the reference
framework can do this, declare it on the node that stores the weight, naming
the option and citing the source:

```yaml
embed:
  op: hash_embedding
  ...
  host_offload: >-
    vLLM EngramConfig.cpu_offload, on by default (VLLM_PLE_CPU_OFFLOAD=1)
    (vllm/config/engram.py:26-47)
```

The weight is still stored and counted; an estimate that offloads (the
default) leaves it out of the GPU's weights. Leave it off every other weight,
including the input embedding.

### Persistent distributed endpoints (`weight_axes` / `expert_weight`)

Every direct occurrence of `embedding`, `linear`, `linear_param`,
`per_head_linear`, `conv1d`, `rmsnorm`, `layernorm`, `parameter_vector`,
`parameter_tensor`, or `rotary_inv_freq_buffer` must declare one `weight_axes`
entry per STORED tensor axis, in checkpoint shape order. A reused component
supplies this contract for its own leaves; author it only for direct nodes in a
new draft/inline component or a top-level model weight.

This is the authoring rule for every new/edited spec. Legacy specs still load
while endpoint migration is in progress, but layout reports an omitted
contract as `DECLARATION_MISSING`; omission never means replicated.

Each axis chooses exactly one form:

```yaml
# Replicated axis.
- { replicated: true }

# Split axis. `world` must be permitted by the `parts` symbol in dims.yaml.
- { world: attention, parts: n_heads, fit: replica_cells }

# Reusable component: select by the model symbol bound at this occurrence.
- placement_by_symbol:
    d_ffn:        { world: dense_ffn, parts: d_ffn, fit: exact }
    d_ffn_expert: { world: moe_expert, parts: d_ffn_expert, fit: exact }

# Fused stored axis: ordered boundaries are declarations, not inferred from
# arithmetic such as `3 * d_model`.
- segments:
    - { segment: gate, extent: d_ffn, world: dense_ffn, parts: d_ffn, fit: exact }
    - { segment: up,   extent: d_ffn, world: dense_ffn, parts: d_ffn, fit: exact }
```

`fit: exact` requires divisibility; `replica_cells` uses the largest real head
partition and replicates inside cells; `pad` preserves the requested degree by
padding whole logical parts (normally vocabulary rows). `replicated: true` is
an explicit audited placement, not the absence of a decision.

A routed expert bank's experts are the LEADING axis of every one of its
weights: `shape: [n_expert_copies, out, in]` on the `linear` / `linear_param`
leaf, over the `[n_expert_copies, M, d_model]` activation `permute` writes. A
new MoE spec calls the bank template in `components/library` that matches the
HF `Experts` class — `gated_experts` (separate gate / up / down),
`fused_gated_experts` (one packed `gate_up_proj`), `clamped_gated_experts`,
`relu2_experts` — and declares how the checkpoint stores the expert axis with
`expert_weight`:

```yaml
experts:
  op: gated_experts
  in: permute
  dims: { d_model: d_model, d_ffn: d_ffn_expert }
  expert_weight: unrolled              # experts.{e}.gate_proj.weight, one entry per expert
  inner_modules: { gate_proj: w1, up_proj: w3, down_proj: w2 }   # when the checkpoint names differ
```

`expert_weight: leading_axis` is one checkpoint entry per weight whose leading
axis is the experts (`experts.gate_up_proj [E, 2*I, H]`); write
`module: null` beside it when the packed tensors sit directly under the
call's module. The expert axis's placement is the first `weight_axes` entry,
`{ world: moe_copies, parts: n_expert_copies, fit: exact }`, which the
templates already carry.

A checkpoint may store a weight in another axis order than the op computes
with. Say so with `stored_axes` on the same node: a permutation of the op's
weight-shape positions, where checkpoint payload axis `i` is
`shape[stored_axes[i]]` (the node's declared `shape`, leading expert axis
included). `weight_axes` then lists the axes in that checkpoint order.
Qwen3-VL-MoE's packed experts are stored `[in, out]` per expert
(`gate_up_proj [n_experts, d_model, 2*d_ffn_expert]`,
`down_proj [n_experts, d_ffn_expert, d_model]`) under the `(out, in)`
`linear_param`:

```yaml
experts.down_proj:
  op: linear_param
  dims: { in: d_ffn_expert, out: d_model }
  shape: [n_expert_copies, d_model, d_ffn_expert]
  stored_axes: [0, 2, 1]   # header [E, d_ffn_expert, d_model]
  weight_axes:
    - { world: moe_copies, parts: n_expert_copies, fit: exact }
    - { world: moe_expert, parts: d_ffn_expert, fit: exact }
    - { replicated: true }
  expert_weight: leading_axis
```

Write it only when the header shape proves the order differs (the same axis
lengths in another order); omit it when the checkpoint matches the op. Only a
single-tensor weight saved unquantized may declare it — a bias sibling or a
quantized format is refused.

Do NOT author process-group degrees, collectives, activation layouts, pending
reductions, routed load, or attention execution plans in a spec. If a required
dim symbol has no suitable `kind`/`shardable_by` entry in `dims.yaml`, or a new
custom op has no propagation semantics, report a maintainer-owned registry gap;
do not invent a YAML field.

### Tensor view ops (`slice` / `concat`)
Every tensor transformation is a node; `in:` is always a list of upstream
node names. `slice` and `concat` are zero-FLOP ops in the primitive table —
use them wherever the model does `torch.split`, `torch.chunk`, slicing, or
`torch.cat`.

Name view nodes by the *operation* (e.g. `q_nope_slice`, `query_concat`),
not by the resulting tensor (`q_nope`, `query_states`).

```yaml
q_b_proj:      { op: linear, in: q_a_layernorm,
                 dims: { in: d_q_lora, out: "n_heads * (d_qk_nope + d_qk_rope)" },
                 weight_axes: [{ world: attention, parts: n_heads, fit: replica_cells }, { replicated: true }] }
q_nope_slice:  { op: slice, in: q_b_proj,
                 dims: { in: "n_heads * (d_qk_nope + d_qk_rope)", out: "n_heads * d_qk_nope" } }
q_pe_slice:    { op: slice, in: q_b_proj,
                 dims: { in: "n_heads * (d_qk_nope + d_qk_rope)", out: "n_heads * d_qk_rope" } }
q_pe_rope:     { op: rope_emb, in: q_pe_slice, dims: { dim: "n_heads * d_qk_rope" } }
query_concat:  { op: concat, in: [q_nope_slice, q_pe_rope],
                 dims: { in_a: "n_heads * d_qk_nope", in_b: "n_heads * d_qk_rope",
                         out: "n_heads * (d_qk_nope + d_qk_rope)" } }
qk_score:      { op: attention_score, in: [q_split_heads, k_split_heads],
                 dims: { n_heads: n_heads, d_head: "d_qk_nope + d_qk_rope" } }
```

### Custom ops (`custom_ops:`) — for ATOMIC novel primitives only
**`custom_ops` is the LAST RESORT, not a convenience.** Use it only when
all four are true:
  1. The op is conceptually atomic — one elementwise kernel, one opaque
     numeric formula, one parameter bank.
  2. Its `params` formula is a SINGLE term (e.g. `d_model`, `0`). If
     your params formula is a sum of multiple d_model × something terms, you
     are bundling multiple Linear projections into one black box —
     decompose into a component instead.
  3. Its `flops` formula has no internal sub-projection structure (just
     scales with input dims, not with multiple separate projection sizes).
  4. No registered primitive in `ops.py` covers it (and adding a new
     primitive isn't a better solution).

For multi-projection submodules (DeltaNet, MoE variants, attention with
extra gates), build a NEW component YAML that decomposes into existing
primitives — see "Custom block graphs" above. **Do not** wrap them as
aggregate custom_ops; that loses the structure the UI / Compare
tools rely on.

```yaml
custom_ops:
  qwen35_spatial_merge:                              # snake_case, distinctive
    flops: "0"
    params: "0"
    signature: { x: [B, s_ctx, d_vis_model] }        # operand name -> its shape
    couples_on: [s_ctx, d_vis_model]           # shape labels this op joins/reduces across
    shape_kind: axis_replace
    out_key: Out
    axis_locked: -1
    formula: "y = concat_2x2_neighbors(x)"      # one-line math for UI
    description: "Zero-parameter spatial merge used by the visual merger to concatenate local 2x2 patches before the MLP."

  # Multi-input custom op: one signature entry per argument, in call order.
  sparse_score:
    flops: "B * n_heads * s_query * n_topk_keys * d_head"
    params: "0"
    signature:
      q: [B, n_heads, s_query, d_head]
      kt: [B, n_heads, d_head, s_kv]
    couples_on: [d_head]
    shape_kind: contract                       # (...,s_query,d_head) x (...,d_head,s_kv)
    input_roles: [q, k]                        # what each operand IS to the op, in signature order
    formula: "scores = contract(q, kt) with a top-k mask"

# Any node that uses the custom op by name:
merge: { op: qwen35_spatial_merge, in: x, dims: { d_vis_model: d_vis_model, n_spatial_merge: n_spatial_merge, Out: "n_spatial_merge * n_spatial_merge * d_vis_model" } }
```

Rules for custom_ops:
- Keep names distinct from built-in ops (`linear`, `rmsnorm`, `slice`, ...).
  The loader rejects a custom op that shadows a registered primitive.
- `flops` / `params` are SymPy expressions — use bound dims (`d_model`,
  `n_heads`, `d_head`, …) and the implicit `B` / `s_ctx`. A trailing `"0"`
  is fine for zero-param ops.
- `signature` declares arity and operand shapes; every call site's `in:` list
  is checked against it. `couples_on` is required: list the signature SHAPE
  labels the op joins/reduces/repartitions across, or write `[]` to assert the
  op is fully separable.
- `shape_kind` is how the output shape derives, one of {yaml_shape_kinds}.
  `preserve` keeps the first input's shape; `axis_replace` and `axis_select`
  also take `out_key`, the dim whose value becomes (or bounds) the new axis.
  There is no shape-template escape:
  an op whose output shape needs Python belongs in `ops.py::register_op`.
- `input_roles` (optional) says what each operand IS to the op, aligned with
  `signature` order, from the closed vocabulary {input_roles} (`null` for a
  position that plays no such part). Declare it when the op is an attention
  or recurrence core — a score reading a query and keys is `[q, k]`, a linear
  recurrence reading q/k/v is `[q, k, v]` — because readers of the graph
  (the fusion matcher telling Q's rope from K's, the structure view) find
  which operand a node computes from it. Omit it for ops that are no such
  core. No duplicates; at most one role per signature operand.
- Unknown keys are rejected. A behavior with no field here (sharding modes,
  KV positions per query, view-op / comm-op status, dtype overrides) is not
  authorable in a spec — register the op in `ops.py` instead, where it can
  carry the Python those behaviors need.
- `description` MUST explain why it can't decompose. Future humans decide
  whether to promote it to `ops.py::register_op`.

Validation: after declaring, run `python -m backend.inference.validate`. The
actual publish condition is its top-level `valid` field. Use
`model_storage_bytes.diff`, per-tensor mismatches, and the structural diagnostics
to locate a wrong `params:` formula; `total_params` is a secondary diagnostic
and there is no `params_diff` field.

### Component authoring fields — when you DO author a new component

You normally don't author components (Phase 2 said pick existing). When you
must, copy structure from `backend/arch/components/library/<closest>.yaml`
and ensure the following fields. Schema and loader errors reject malformed
graphs; role and description omissions are structural warnings.

The loader's complete authored block/component key set is `attn_variant`,
`cache`, `custom_ops`, `decode_variant`, `description`,
`graph`, `inputs`, `outputs`, `params`, `reads`, `role`, `secondary_outputs`,
`stream_out` and `variant_phases`.
`decode_variant` names the component used at decode when that graph is not
the prefill graph (MLA absorbed; a prefill-only window view);
`variant_phases: [prefill, decode]` beside it says prefill runs that form too
(a sparse MLA attention, which serving prefills absorbed). A component-level `name` is rejected because the file stem is
its identity. There is no hand-written `formula` on components either — the
UI derives the math from the graph itself (`derived_formula`), so anything
authored could only agree or drift. The retired `kv_cached`, `kv_cache_dtype`,
`kv_state_elements`,
`kv_label`, `carries`, `carry_outputs`, `carry_src`, and `carry_init` spellings
are also explicit load errors. On a graph node, the retired checkpoint-naming
spellings `rename`, `hf`, `hf_inner`, and `primary` are load errors too —
use `module` / `inner_modules`; a one-node adapter is detected from the graph.

```yaml
role: attention                      # a registered role (table under "Roles on architecture-bearing components")
description: "One-line structural description"
# `inputs: [x]` is the default; it may be written explicitly.
graph: { ... }                       # node DAG (op + dims + inputs)
outputs: [<node_name>]               # the node whose value it hands its caller
```

**A direct weight-bearing attention component MUST declare
`cache: {family: ...}`.** `family` is one of the families registered in
`backend/arch/cache_families.yaml`; the loader rejects any other. It is
used by inference KV-cache topology, weight-breakdown's class label, and
attention-type detection. Alias-adapter wrappers (1-node delegations) are
exempt from the load-time check because they inherit from the inner component.

{cache_families_table}

```yaml
role: attention
cache:
  family: GQA                        # KV-cache class — a STORAGE fact
  entries:                           # stored graph nodes appended per token
    - { node: k_rope, kind: kv }
    - { node: v_split_heads, kind: kv }
# attn_variant IS the attention label shown to users (the family alone is only
# KV-cache STORAGE accounting). Set it WHENEVER the compute flavor differs from
# the coarse cache family — omitting it MISLABELS the model. Use the mechanism's
# real name from the config / modeling source, not the generic family. Examples:
#   MLA-shaped cache + DeepSeek sparse compute         -> attn_variant: DSA
#   GatedDeltaNet-shaped state + Kimi Delta Attention  -> attn_variant: KDA
#     (config says `kda_layers` -> it is KDA, not the generic GatedDeltaNet)
# Repeating the family verbatim is an error (see the validator).
# attn_variant: DSA
graph: { ... }
outputs: [o_proj]
```

`cache:` is the ONE place a component says what it keeps between decode
steps. An entry is either a graph node (per-token; its shape is the node's
output) or a fixed recurrent state (linear attention / SSM; its `shape:` is
declared, because no node outputs it).

Every entry declares `kind:`, what the slot holds (`cache_families.yaml`
`slot_kinds:`; the loader rejects any other):

{cache_slot_kinds_table}

A `node:` entry that names a sub-component with its own `cache:` gives the
kind of that sub-cache's per-token tensors (DeepSeek-V4's indexer caches its
compressor's pooled rows as an `index_key`); the sub-cache's states keep
their own kinds.

`dtype:` is the precision the model stores the slot at, overriding the
deployment-wide KV dtype for that slot alone: a dtype spelling the forward
source shows (`fp8`, `F32`), or the config field the checkpoint states it
in, tried in order and ending on the dtype the HF forward stores the slot
at: `dtype: "config.mamba_ssm_cache_dtype ?? F32"`. A checkpoint whose
config field states another dtype than that last term is refused: the
shipped precision is the HF forward's. A chain may end on `model` (the
model's own precision) where the forward stores the slot at it.

```yaml
cache:
  family: MLA
  entries:
    - { node: kv_a_layernorm, kind: latent }
    - { node: k_pe_rope, kind: rope_key }
    - { node: indexer.k_norm, kind: index_key, dtype: fp8 }    # mixed-precision KV

cache:
  family: GatedDeltaNet
  entries:
    # The delta-rule matrix as the HF forward allocates it
    # (modeling_qwen3_5_moe.py torch.zeros(batch, num_heads, k_head_dim, v_head_dim)).
    - { state: recurrent_state, kind: recurrent_state, carrier: delta_rule, shape: [n_lin_v_heads, d_lin_k_head, d_lin_v_head], dtype: "config.text_config.mamba_ssm_dtype ?? config.mamba_ssm_dtype ?? F32" }
    # The causal conv's window over its channels, carried by the conv and
    # written in the conv's own dims (cache_utils.py conv_states
    # torch.zeros(batch, conv_dim, conv_kernel_size)).
    - { state: conv_state, kind: conv_state, carrier: conv1d, shape: [C, kernel_size] }

# Kimi Delta Attention: the matrix plus one window per short conv of q, k
# and v (fla ShortConvolution cache [B, D, W]). Kimi calls fla `chunk_kda`
# with `transpose_state_layout=True`, so fla holds the state V-first,
# [B, H, V, K]; a KDA whose forward keeps the default layout writes
# [n_lin_v_heads, d_lin_k_head, d_lin_v_head].
cache:
  family: GatedDeltaNet
  entries:
    - { state: recurrent_state, kind: recurrent_state, carrier: delta_rule, shape: [n_lin_v_heads, d_lin_v_head, d_lin_k_head], dtype: F32 }
    - { state: q_conv_state, kind: conv_state, carrier: q_conv1d, shape: [C, kernel_size] }
    - { state: k_conv_state, kind: conv_state, carrier: k_conv1d, shape: [C, kernel_size] }
    - { state: v_conv_state, kind: conv_state, carrier: v_conv1d, shape: [C, kernel_size] }

# Mamba2 (modeling_nemotron_h.py ssm_states [b, h, d, n]): d_state is the
# innermost axis; fp32 as the config declares (`mamba_ssm_cache_dtype: float32`).
cache:
  family: Mamba2
  entries:
    - { state: recurrent_state, kind: recurrent_state, carrier: scan, shape: [n_ssm_heads, d_ssm_head, d_ssm_state], dtype: "config.mamba_ssm_cache_dtype ?? F32" }
    - { state: conv_state, kind: conv_state, carrier: conv1d, shape: [C, kernel_size] }

# DeepSeek-V4's KV compressor: a per-token pooled slot plus two ring
# buffers the reference model holds in fp32 whatever the KV dtype, each
# `compress_ratio` rows of `Cdim_io` (inference/model.py:303 kv_state
# [coff * ratio, coff * head_dim], fp32).
cache:
  entries:
    - { node: pooled_out, kind: compressed_kv }
    - { state: kv_state,    kind: compressor_state, carrier: weighted, shape: [compress_ratio, Cdim_io], dtype: fp32 }
    - { state: score_state, kind: compressor_state, carrier: smax,     shape: [compress_ratio, Cdim_io], dtype: fp32 }
```

A state entry has four required fields:

- `state:` — its name, unique in the component (`recurrent_state`,
  `conv_state`, `q_conv_state`, …). A `dtype:` override keys on it.
- `kind:` — what it holds, from the state table above.
- `carrier:` — the graph node whose kernel reads and rewrites the state every
  decode step. It must be a primitive or custom-op node of this graph, and
  one node carries at most one state. The state's bytes are billed and drawn
  on that node, since no edge reads a state.
- `shape:` — the state tensor's axes, outermost first, as the HF forward
  allocates and updates it. Each axis is a dim expression resolved in the
  carrier's frame (the component's dims plus the carrier's own `dims:`).
  Write global extents; what each rank holds follows the carrier's own
  sharding, so write a head axis with the same name the carrier's output
  carries it under. The last axis is the one quant scales group over, and
  the state's size is the product of its axes. Its axes are the kernel's
  own widths — heads, head dims, the conv window — since a state is one
  fixed tensor per request.

and one optional field:

- `dtype:` — the state's dtype as the HF forward or the repo's config
  declares it (`mamba_ssm_dtype`, `mamba_ssm_cache_dtype`, an fp32
  `torch.zeros(...)`), cited; a config field is written as its expression
  with the source's default, `"config.mamba_ssm_cache_dtype ?? F32"`. Each
  state carries its own; omit it when the
  source keeps the state in the model dtype, and it follows the deployment's
  KV dtype.

Declare one entry per tensor the forward keeps between steps, each on the
kernel that owns it — read the forward's cache allocation and its update. A
Mamba2 / GatedDeltaNet layer keeps two: the scan's recurrent matrix and the
causal conv's window, `[C, kernel_size]`. Where each of q, k and v has
its own short conv (Kimi KDA), each conv carries its own window. Each tensor
is its own entry with its own `shape:`.

Every node whose op declares `carries_recurrent_state` (`gated_delta_rule`,
`mamba2_scan`, a custom recurrence with `carries_recurrent_state: true`) must
be named as some state's carrier; the validator rejects the component
otherwise, as it rejects a carrier that is not a node of the graph, two
states on one carrier, a name used by two cache entries (a state may not
share a node slot's name — the name keys the slot's `dtype:`), a `shape:`
axis reading a name that neither the carrier's `dims:`, the component's
graph, the spec's dims, nor the dim registry binds, and a `shape:` axis
reading a sequence or batch dim.

There is deliberately no component-level `kv_window` or
`parallelism.shard_symbols`. Cache read/storage cardinality comes from the
cached nodes' producer/consumer op laws, while per-rank sharding comes from
propagated axis identities under the eval binding. Declaring either removed
field is a load error.

Source-derived claims about library components live beside the library in
`backend/arch/claims/components/<component_name>.yaml` (`schema:
huggingarch.claims/v1`, matching `subject`). Validate judges them at every call
of the component in the model's analysis, against the cache that call grows.

Checkpoint naming is declared with `module` and `inner_modules`. There is no
`hf`, `hf_inner`, `rename`, or `primary` field. `module` has three states:
omit it when the checkpoint segment equals the graph node name, set a string
when the segment has another name, or set `null` when this graph position has
no checkpoint module. A compound call uses
`inner_modules: { child_node: checkpoint_name_or_null }` when the shared
component's internal module names differ at this call site.

```yaml
# router_noaux_tc — gate is the canonical Linear; e_score_correction_bias /
# softmax / topk_indices are siblings.
graph:
  linear:                  { op: linear, in: x, dims: { in: d_model, out: n_route_targets }, module: null, weight_axes: [{ replicated: true }, { replicated: true }] }
  e_score_correction_bias: { op: parameter_vector, dim: n_route_targets, weight_axes: [{ replicated: true }] }
  scores_biased:           { op: add, in: [linear, e_score_correction_bias], dim: n_route_targets }
  topk_indices:            { op: topk, in: scores_biased, dims: { in: n_route_targets, k: n_experts_active } }
outputs: [topk_indices]
```

A one-node pure delegation adapter is detected and inlined automatically when
it carries only binding fields. Use `module: null` and `inner_modules` on that
delegation when its checkpoint namespace differs; do not add marker fields.

### Quant context (`quant_context:`) — required for quantized models

Models that ship non-BF16 weights (FP8 / MXFP4 / INT4 / INT8 / GPTQ /
AWQ / compressed-tensors) MUST declare `quant_context`. Without it, the
analyzer assumes BF16 and miscounts deployment bytes / mis-renders
dtype labels / skips on-disk scale tensors.

```yaml
quant_context:
  per_role:
    attention: { method: fp8, block_h: 128, block_w: 128 }
    ffn_dense: { method: fp8, block_h: 128, block_w: 128 }
    ffn_moe:   { method: fp8, block_h: 128, block_w: 128 }
    router:    { method: none }            # router stays BF16
    default:   { method: none }            # any role not above → unquantized
    attention_output: { method: none }     # o_proj stored BF16 (a finer role)
  default_dtype: BF16                       # non-weight tensors (norms, biases)
  overrides:                                # (optional) per-tensor escape
    - tensors_re: 'mlp\.gate\.e_score_correction_bias$'
      dtype: F32                            # a vector stored at its own dtype
    # Composite identity selector: MTP may share the backbone's name.
    - select:
        stack: mtp
        tensors_re: '\.down_proj\.weight$'
      dtype: BF16
    # Layer-scoped: `modules_to_not_convert` named these layers by index,
    # and nothing structural tells them from the others.
    - select:
        layers: { range: ["n_layers - 3", "n_layers - 1"] }
        tensors_re: 'moe\.experts\.w(13|2)_weight$'
      dtype: BF16
```

Precision is chosen by role and stack, so within one role and one stack
every weight matrix is stored at one precision. Where the checkpoint stores
part of a component at another precision, put a finer role on those nodes
(`role:` on the node, a role from `roles.yaml` that refines the component's:
{refining_roles}) and state its
dtype as a `per_role` entry. A `per_role` entry for the broader role reaches
every role refining it unless that role has its own entry. Two attention
mixers of one model stored at different precisions take two slots
(`attention` and `attention_linear` / `attention_swa` in `components:`).

That holds for a split between two NODES, which is what a role addresses.
One node stored two ways across its OWN layers is the other case: a role is
a property of the node at every layer it runs, so no role expresses it.
Declare that with the layer scope on an override row
(`select: { layers: ..., tensors_re: ... }`, below). The validator asks
exactly this question and refuses only the first case.

When the model's own forward caches a slot at a format other than the
model's dtype, state it per slot kind under `quant_context.cache_formats`
(a named format from `arch/cache_formats/`, or `{dtype, scale, transform,
unquantized_tail}`), citing the forward line:
`cache_formats: { index_key: fp8_group128 }`.

The format is what the forward QUANTISES the write to, not the dtype of the
buffer it lands in. A reference forward often quantises in place
(`act_quant(kv, 64, ..., True)`, `fp4_act_quant(k, 32, True)`) and then
copies into a `torch.zeros(...)` BF16 buffer: the BF16 is a container, the
stored values are FP8 / FP4, so declare the FP8 / FP4 format. The same
holds when the forward quantises and dequantises back by hand before the
write (`kv_fp8, s = act_quant(kv); kv = kv_fp8 * s` — "we simulate the
precision"). Read every write into a cache buffer, not only its
`register_buffer`. A forward that
quantises only part of the row (`act_quant(kv[..., :-rd], ...)` leaves the
RoPE tail BF16) declares the rest as
`unquantized_tail: { channels: d_qk_rope, dtype: BF16 }`. A format only a
serving framework picks and the forward does not simulate (vLLM's
`--kv-cache-dtype fp8`) is a deployment precision choice, not a
`cache_formats` entry.

Available `method:` values, read from `backend/arch/quant_schemes/<method>.yaml`
(see the file for the full layout). A `per_role` / override entry must supply
every required field; an optional field may be given; a defaulted field takes
the value shown when omitted. The sidecars are the quantization metadata
tensors the method stores beside each weight: the scheme implies them, so the
spec never declares them.

{quant_schemes_table}

A lookup table (`embedding`, `hash_embedding`) takes a quant scheme the same
way a linear does: its stored `(rows, width)` is the scheme's `(out, in)`. An
FP8 table with one UE8M0 scale per 32 elements of a row is
`{ tensors_re: 'embed\.weight$', dtype: F8_E4M3, method: fp8_ue8m0, block_h: 1, block_w: 32 }`;
the scheme implies the `.scale` sibling `[rows, width / 32]` as quantization
metadata, and a lookup reads the gathered rows together with their scale rows.

Flat `overrides[].tensors` / `tensors_re` remain the simple name-only form.
`select:` is the composite identity form: it contains exactly one of `tensors`
or `tensors_re`, plus optional `stack`, `component` and `layers`. Every scope is
evaluated before the HF-on-disk name predicate, and all given fields must hold.
`stack` is the top-level aux group's logical name; use it when an aux stack
deliberately shares the backbone's checkpoint address. `component` is the name
of a component the spec declares (library or draft); it selects the tensors of
nodes inside an invocation of that component, at any nesting depth. Use it when
two components of a hybrid stack store tensors under the same template name and
the checkpoint header gives them different dtypes — for example an MLA and a
linear-attention layer that both own `self_attn.o_proj`. Every node keeps its
true `role:`; the component scope carries the storage difference, so params and
FLOPs stay booked under the role the forward pass gives them.

`layers` is the layer scope, and it is how you express a
`config.quantization_config.modules_to_not_convert` (or `ignore`) list that
names tensors by LAYER INDEX. Step-5-Preview names the routed-expert weights
of layers 88, 89 and 90 of its 91 — same role, same stack, same component,
same graph as the other 88 — so no name pattern reaches them (checkpoint
names in the spec carry `X` at every indexed position) and **no component may
be duplicated to carry the dtype**: a precision is not a structural identity,
which is the whole reason `quant_context` exists. Write the layer scope
instead. It takes the same selectors a `per_layer.assign` generator takes, so
there is one spelling to learn:

```yaml
layers: { range: ["n_layers - 3", "n_layers - 1"] }  # a span; bounds are dim expressions
layers: [88, 89, 90]                                  # an explicit set
layers: { config_list: layer_types, equals: full_attention }   # a config schedule
layers: { stride: 2, offset: 1 }                      # every other layer
```

Prefer a `range:` in dim expressions over literal indices when the config
gives the count: `n_layers - 3` follows a size variant, `[88, 89, 90]` does
not. The numbers are layers of the stack the row names (`stack:`), the
backbone when it names none.

A layer-scoped row MAY select tensors a broader row or a `per_role` rule also
selects — carving some layers out of a rule that covers all of them is the
point — so write it BEFORE the broader row: rows are evaluated in declaration
order, first match wins, asked once per layer, and a layer-scoped row placed
after the row it carves out of selects nothing and fails the dead-row gate.

A `component:` row, by contrast, is the only row that selects its tensors:
write the other rows so they select disjoint tensors (a load error names any
pair that both select one tensor). Every `tensors_re` is compiled at load
time. An exactly duplicated selector is rejected because the later row can
never run; two rows that differ only in their `layers` are two rules, not a
duplicate. `auto_imported` is regeneration metadata, not a field the spec
writer authors.

Reference real specs:
- `models/deepseek_v3.yaml` — FP8 per-128×128 tiles, all 3 weight roles
- `models/kimi_k25.yaml` — INT4 group + many BF16 overrides for non-MoE-expert tensors
- `models/gpt_oss.yaml` — MXFP4 + router BF16 override
- `models/ling_2_6_1t.yaml` — INT8 compressed-tensors with channel strategy
- `models/qwen3_coder_next.yaml` — GPTQ INT4

The validator's packing-consistency check and aggregate storage-byte gate catch
quantization declarations that disagree with checkpoint storage. Per-role and
per-override details are inputs to that accounting, not a separate per-role
gate.
