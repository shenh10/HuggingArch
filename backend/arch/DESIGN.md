# HuggingArch Spec System — Design

The spec system is the **single source of truth** for every architectural fact
about a model in HuggingArch. Once a model is specced, every downstream
analysis — FLOPs accounting, memory layout, parallelism planning, scheduler
simulation, codegen, accuracy/quantization estimation — consumes the same
declarative artifact. This document describes the system itself: what it is,
why it exists, and the layered representation that makes it composable.

## Why a spec layer exists

Without a spec layer, every analysis tool faces the same dilemma:

- Re-parse HF `modeling_*.py` source per model — brittle, requires Python
  execution, demands model-by-model special-casing every time the analysis
  needs a new field.
- Re-parse `safetensors.index.json` and guess at compute structure — lossy,
  no forward dependency information, no way to tell a Linear apart from an
  embedding from a reshape.

The spec system formalizes both ends into one declarative artifact. A spec
captures:

- **Forward computation graph** with per-node FLOPs/params formulas
- **Stored tensor names** that map onto the actual `safetensors.index.json`
- **Component roles** (attention / ffn_dense / ffn_moe / router / block /
  aux_block) for selectively transforming subgraphs
- **Dim variable bindings** (from `config.<field>` references) so analyses
  can vary `batch / seq / parallel-degree` analytically without re-running
  the model

The bet: **the cost of writing one spec per model amortizes across many
downstream consumers**. Today there are four (analyzer, response builders,
inference, custom-model loader); the count will grow.

YAML is the authoring grammar. At runtime it leaves once, through
`iter_tree`, as a `SpecIRTree`. `Program` organizes that tree into a walk;
`Evaluator` produces an `Analysis`; comm then fusion rewrite the Program
and rebase numbers. Consumers read those objects. They do not re-parse
YAML or re-read `dim_values` for a fact the tree or the Analysis already
holds. The walk-through is in [`docs/design.md`](../../docs/design.md) §2
and [`docs/evaluation_model.md`](../../docs/evaluation_model.md).

## The 4-tier representation

This is the core invention. Every model is expressed as a layered
composition:

```
            ┌─────────────────────────────────────────────┐
Layer 4:    │  Model spec  (models/*.yaml)                │  per-family wiring
            │  – dim bindings from config.<field>         │  + HF tensor paths
            │  – per-layer assignment of templates        │
            │  – top-level node names = HF prefixes       │
            └──────────────────┬──────────────────────────┘
                               │ binds & references
            ┌──────────────────┴──────────────────────────┐
Layer 3:    │  Adapter  (components/adapters/*.yaml)      │  binding + hf names
            │  – single-node body                         │  scope: 1 family
            │  – wraps a library compound + params        │
            │  – one node: no namespace of its own        │
            └──────────────────┬──────────────────────────┘
                               │ wraps & binds params
            ┌──────────────────┴──────────────────────────┐
Layer 2:    │  Library compound  (library/*.yaml)         │  reusable topology
            │  – multi-node compute graph                 │  cross-family
            │  – parameterized via `params:` slots        │
            │  – role-labeled (attention / ffn / …)       │
            └──────────────────┬──────────────────────────┘
                               │ composes
            ┌──────────────────┴──────────────────────────┐
Layer 1:    │  Primitive  (ops.py)                        │  irreducible kernel
            │  – flops_fn, params_fn, shape_out           │  ~70 of them
            │  – examples: linear, attention_score,       │
            │    rmsnorm, topk, gather, parameter_vector, │
            │    rotary_inv_freq_buffer, group_top2_sum   │
            └─────────────────────────────────────────────┘
```

What a primitive's `flops_fn` coefficient counts — scalar arithmetic of its
formula per output element, FMA = 2, one transcendental = 1, tables and
constants as bytes — is written once, in `eval/flops.py` ("What a FLOP is").

Each tier has a **specific job**, and new requirements land in exactly one
of them:

| Need | Tier touched | Cost |
|---|---|---|
| New atomic kernel (e.g. SSM recurrence) | Layer 1 — `ops.py` | one Python registration; all higher tiers can use it immediately |
| New topology variant (e.g. router with grouped-batch top-k) | Layer 2 — library | one YAML; multiple families can compose against it |
| New checkpoint naming (e.g. ChatGLM fused QKV) | Layer 3 — adapter | one YAML; library + model unchanged |
| New model family with existing structure | Layer 4 — model | one YAML; binds existing tiers |

This is what makes the system composable. The 4-tier hierarchy enforces
**separation of concerns**: a tensor-naming change (very common as new
training frameworks ship) doesn't touch the compute graph (rarely changes);
a new compute primitive (rare) doesn't force every model to re-spec.

## Schema is the contract

The Pydantic schema (`schema.py`) makes these tiers **first-class concepts**,
not string conventions. Key invariants the schema enforces:

- Every `GraphNode.op` is one of: a primitive name, a component name, or a
  `<param_slot>` placeholder
- Components declare `params:` slots so they're parameterizable; callers
  bind via `params: { slot: <component_name> }` or dict-typed nested form
- `module:` says what this node contributes to the CHECKPOINT path: absent
  means its own name, `null` means nothing (no `nn.Module` there), a string
  means that segment. Purely a tensor-naming statement; the node still
  evaluates and still owns its children
- `inner_modules:` says the same for the nodes of the component this node
  invokes, because what a checkpoint calls a module is a fact about that
  checkpoint and one library component is shared by checkpoints that disagree
- All dim expressions are SymPy-parseable strings (`"d_model // n_heads"`,
  `"2 * n_heads * d_head"`),
  evaluated against bound config values at load time
- Every BlockGraph validates: all `inputs:` references resolve, `outputs:`
  names at least one node and every one exists, a call site's `dims:` fit its
  op's DimMap (`axis_propagation.call_site_dims_error`: a `slice` states the
  window it keeps; a `view_merge` states no width — it is the product of the
  two axes it merges, `VIEW_MERGE_AXES`)

The schema turns specs from **documents** into **executable graphs**: the
engine traverses them directly, no Python reflection, no model-specific
loaders.

## Which phase runs which form

A component whose decode graph differs from its prefill graph names the other
form (`decode_variant`), and the phases that run it (`variant_phases`,
`[decode]` when omitted). `transforms.phase_form` reads the two and nothing
else. A dense MLA prefills expanded and decodes absorbed; a sparse MLA (DSA)
declares `variant_phases: [prefill, decode]`, because serving prefills it
absorbed through the sparse MLA kernel decode uses. Before the field existed
the engine inferred the second case from the graph holding a DSA op — the
graph's content standing in for a serving fact. The field is the existing
declaration extended by the one fact it lacked (four questions: no existing
field said it; it is observable — prefill runs a different graph; one
dimension, the phase set; the literature has no other name for it).

### A form runs the stored tensors; it does not restate them

What is stored does not change with the form a phase runs. Absorbed MLA runs
the checkpoint's `kv_b_proj` (`[n_heads·(d_qk_nope + d_v_head), d_kv_lora]`)
as two per-head GEMMs (`q_absorbed_nope`, `v_absorbed`), the way vLLM splits
`W_UK` / `W_UV` out of that one tensor at load. Nothing new says so: `module:`
already means "this node's weight lives at this checkpoint address", and
both GEMMs declare `module: kv_b_proj`. Weight-holding siblings at one
address are therefore one stored tensor. The absorbed component declares it
once, as the file stores it — `kv_b_proj: { op: parameter_tensor, shape:
[...], tensor_suffix: { weight: .weight } }`, the nn.Linear's weight a form
holds without running it as a linear — and the build ties the others at that
address to it (`spec_tree.build._tie_shared_modules`, onto the tie a tied LM
head uses: `SpecIRNode.tied_to_uid`, `Program.read_aliases`). An address
shared by weight holders with no stored parameter among them, or two, is
refused. Storage, formats, scales, the checkpoint list, residency and the
per-rank shards are then the stored tensor's in every phase, and every
consumer that walks a phase sees the file; the GEMMs keep their op, shapes,
FLOPs, fusion and Kernel Bench keys.

A tied read moves the share of the stored tensor its reader's declared weight
is (`Program.read_aliases`: the reader's declared weight over the holder's,
resolved with the tie): a tied LM head reads the whole table
(share 1), each absorbed half its half of `kv_b_proj`'s bytes and scales.
Before this, each half stored its own `[n_heads, out, in]` view at that
address, so the decode checkpoint listed two tensors the file does not hold
and tiled FP8 scales per half (GLM-5.3: 512 B per layer more than the file;
Ling-2.6-1T: per-channel scales over the view's rows), and the validator
walked the authored graph behind a flag (`checkpoint(as_stored=)`) to see the
file — one walk, two meanings.

A shared `module:` address is the declaration of a split at load, and the
only one (`SpecIRNode.split_at_load`, set by `_tie_shared_modules` and by
nothing else; it also records the holder it split, `SpecIRNode.split_of`, a
resolved IR fact and not an authored field — a looped stack's later pass
ties that holder on to the owner pass's, and its pieces still partition
their own graph's holder, which is what `Program.read_aliases` sums by): each sibling at the address runs a piece the framework builds
as a weight of its own when it loads the checkpoint, and the piece is what
the GEMM streams. `tied_weight:` is the other tie and means the other thing:
the reader names a layer whose tensor it streams whole and as stored (a tied
LM head) — one tensor, two readers, no copy. The address form is held to its
meaning at load: the readers' declared weights must partition the stored
tensor (`Program.read_aliases` refuses pieces that do not sum to it), so a
GEMM running the whole tensor cannot be written as a shared `module:`. No
component field states the split besides the address — a second statement
of the same fact is a second place for it to drift.

Both frameworks dequantise `kv_b_proj` into the model's dtype to build
W_UK / W_UV and run them through `torch.bmm` (vLLM `split_kv_b_proj`,
sglang `block_quant_dequant`), so an FP8 `kv_b_proj` is read as BF16 pieces
with no scales and no activation cast in front of either GEMM. What a piece
is held at is the deployment's answer (`ResolvedPrecision.load_split_format`,
today the served format), bound once into what a leaf streams
(`traffic.Streamed`, with the tie's own `TiedRead.split`) and applied in one
place (`Streamed.as_held`) that the bytes, the conversion, the row's dtypes,
the weights its row names and the promotion all read; the stored tensor
stays the file's format in `tensors()`. The row names a piece by the stored
tensor's name, the piece's dtype and the rank-local rows of the stored
layout it covers — its elements times its dtype are the bytes billed.

Not modelled, by decision, each to be its own item:
- The pieces' residency. Both frameworks keep the stored `kv_b_proj`
  resident beside W_UK / W_UV (the expanded MLA prefill runs it; vLLM frees
  it only on its CPU AMX path), about 2 GB on DeepSeek-V3. Capacity counts
  the stored tensor only, before and after: the walk of one form does not
  know the other form's pieces, and residency is the same in every phase.
- The framework × scheme launches the default path is not
  (`load_split_format` lists them): sglang's per-tensor `bmm_fp8` for a
  per-channel FP8 `kv_b_proj`, ROCm aiter, DeepGEMM BMM. They enter as the
  framework's input to the walk, not as a second answer outside it.

## Source of truth: compute vs naming

A spec describes two things that frequently diverge:

- **Forward computation graph** — ground truth is HF `modeling_*.py` source
- **Stored tensor names** — ground truth is `safetensors.index.json`

These diverge because a checkpoint is the output of *some* training
framework (Megatron-Core, internal vendor stack, DeepSpeed, …) whose
storage layout doesn't have to match HF transformers' modeling code.
Common patterns:

- Stacked-expert weights: `mlp.experts.gate_up_proj` shape
  `[n_expert_copies, 2*d_ffn_expert, d_model]`
  with no `.weight` suffix (grouped-GEMM perf path), while the HF source
  reads as `nn.ModuleList([SwiGLU(...) for _ in range(n_expert_copies)])`.
- Bare `nn.Parameter` instead of `nn.Linear`: DSv3's routing gate is
  `self.weight = nn.Parameter(...)`, so the state-dict key is
  `mlp.gate.weight` — no `.linear.` indirection.
- Asymmetric subgraphs in one model: Qwen3.5's main backbone uses
  stacked-Parameter experts (perf path) while its MTP head uses a per-expert
  ModuleList (standard path). Same `modeling_*.py` class, two checkpoint
  layouts.
- Quantization wrappers append suffixes (`*_qweight`, `*_scale_inv`,
  `*_packed`) that don't appear anywhere in the original `nn.Module`
  hierarchy.

This is precisely why the **library / adapters split is a physical
separation, not a tag**: library encodes the compute graph (read source);
adapters encode the checkpoint naming (read safetensors). The principle:

> **Source is ground truth for compute. Checkpoint is ground truth for
> naming.**

## Today's consumers

- `eval/` — the evaluation core: one walk over the resolved spec under
  a chosen binding (numeric, symbolic, or per-rank) producing an
  `Analysis` of layers — cost, shapes, storage, formats, cache laws —
  with HBM traffic, KV footprints and weight bytes **derived** from
  those layers rather than carried. See
  [`docs/evaluation_model.md`](../../docs/evaluation_model.md).
- `rendering package` — composes full spec node IDs (the only authority
  for path naming), matches against HF tensor patterns, renders the UI
  block_graph + dashboard + unmatched detection
- `analyzer.py` — orchestrates per-model analysis end-to-end
- `inference/` — the serving pipeline (per-op roofline, capacity,
  parallelism communication injection) consuming `Analysis` UID layers
  directly

## Forward-looking consumers

The same specs, no re-expression required, can drive:

- **Communication accounting** — all-reduce, all-to-all volumes per
  parallel config
- **Compiler / codegen** — lower spec to fused kernels, Triton, CUTLASS
- **Training-step cost** — fwd + bwd + optimizer state, gradient accumulation
- **Quantization planning** — per-op precision selection, mixed-precision
  policy validation
- **Accuracy-bound predictions** — per-layer sensitivity, quantization
  error propagation
- **Cross-version model diff** — schema-level diff = architecture diff,
  surface "what changed in DSv3.2 vs V3.1"
- **Hardware co-design** — op placement on heterogeneous accelerators,
  feasibility studies
- **Cost / capacity rollups** — TTFT / TPOT / throughput at any
  GPU SKU × parallel config combination

Each new consumer is **a new traversal of the existing spec library** — it
does not require re-speccing models. The spec system's value scales
super-linearly with consumer count; this is why it is the project's
foundation.

---

# Mechanism details

## `module`: what each node is called in the checkpoint

A spec graph has a node wherever the architecture has a step. A checkpoint
has a path segment wherever the model has an `nn.Module`. Those two are
mostly the same and sometimes not, and where they differ the spec says so:

```yaml
q_proj: { op: linear, in: x, dims: {...} }                         # absent → `q_proj`
linear: { op: linear, in: x, dims: {...}, module: null }           # no module here
qkv:    { op: linear, in: x, dims: {...}, module: query_key_value }
```

`module: null` means the checkpoint has no module at this node's position, so the
node contributes no segment to the leaves below it — the Linear inside a
`mlp.gate` wrapper stores as `mlp.gate.weight`, not `mlp.gate.gate.weight`.
A string means the checkpoint calls this position something else. Absent —
about 97% of nodes — means the two namespaces agree and nothing is written
down. **It is purely a tensor-naming statement**: the node still evaluates,
still costs, still owns its children.

Only `weight_name` is affected. `display_path` / `aggregation_id` stay pure
nested over EVERY node, so a wrapper and the leaf inside it remain
distinguishable — which is what keeps a fused subgraph from writing two
shape entries under one id.

UI side: the wrapper component box stays visible (`router_noaux_tc` /
`pre_norm` / …); the moduleless child drops out of the visible children
list and its grandchildren splice into the wrapper, so the tree has a node
exactly where the checkpoint has one. The wrapper inherits the child's
matched-tensor metadata so its `<id>.weight` lookup lands on the Linear's
weight at HF tensor matching time.

### Call-site override

Whether a module is there is a fact about a CHECKPOINT, and one library
component can be shared by two that disagree. `mtp_block_shared_head`
declares its inner-block node absent — DeepSeek and GLM store an MTP layer
flat in the main `layers` index space — while Qwen3.5's head really does
hold `self.layers = nn.ModuleList(...)` at that position (vLLM
`qwen3_5_mtp.py`). So its adapter says so:

```yaml
layer:
  op: mtp_block_shared_head
  inner_modules: { transformer: layers }
```

Keys are the invoked component's own node names and values use the same
grammar as `module`. An override naming a node the component does not declare is
an error, not a no-op: a silently-ignored override is a statement about a
checkpoint with no effect, which is the failure mode this whole mechanism
exists to end.

The legacy `rename:` / `hf_inner:` keys, which said the same thing by
REWRITING the node's structural name, are refused at load: a node called
`qkv` in the spec and `query_key_value` on disk is one node, and structural
ids carry no checkpoint spelling.

### Alias adapters

An adapter is a template whose whole body is ONE node delegating to a
component, carrying the `params` / `module` / `inner_modules` that specialise
it. It has no structure of its own, so it contributes no namespace: the
walk descends into the sub-graph directly and the ids land under
`model.language_model.layers.X.` with no intermediate `layer.` segment.

This is derived from the graph, not declared: being an adapter is a property
of the graph, so nothing has to be asserted and nothing can be asserted
wrongly. A declared flag would have to decide between two readings — "fold
this node's name away" and "replace this whole template" — and the second,
applied to a wrapper whose template has real siblings, silently discards
them: no error, visible only as a step in a byte total.

## `module` is NOT the same as frontend `absorption`

Both produce the visual "child folded into parent" effect, which makes it
tempting to merge them into one declarative field. **They cannot be
merged.** Conflating them breaks checkpoint matching.

| | `module: null` (this layer) | `absorption` (web `BlockDag.tsx`) |
|---|---|---|
| Layer | spec / IR — a claim about the checkpoint | frontend visual heuristic |
| Where defined | `GraphNode.module` on a yaml node | `computeAbsorptions()` in `web/src/pages/BlockDag.tsx`, structural trigger only |
| Trigger | author states it, checked against the header | automatic: `is_source` sibling with single consumer wrapper |
| Affects `weight_name` | YES — the node contributes no segment | NO — the param's name is unchanged |
| Affects `display_path` | NO — ids stay pure nested | NO |
| Affects HF tensor matching | YES — wrapper inherits the child's `matched_tensor` so `<W>.weight` lands on the Linear's weight | NO — checkpoint paths come from names, not visual nesting |
| Affects FLOPs / params totals | NO (purely naming) | NO (purely visual) |
| Removable? | NO — deleting it breaks dtype lookup for compound wrappers; mixed-precision models lose dtype on every fold | YES — ELK just lays params out as siblings; long thin lines, data still correct |

**The pivot test**: delete the mechanism and run materialize + validate on
a mixed-precision V4-Pro checkpoint.

- Delete `module`: the bytes-vs-bytes diff jumps, because every folded
  Linear (`mlp.gate.weight` and friends) loses its dtype lookup. Structural
  failure.
- Delete `absorption`: output is byte-identical. Only the SVG looks worse.

So `module` belongs in this doc and `absorption` belongs only in
`web/src/pages/BlockDag.tsx`. The right shared abstraction would be "what
does the rendered tree fold" — but one fold has data-faithful consequences
and the other does not, so they stay two mechanisms with different
invariants.

## `inner_modules` scope

`inner_modules:` on a `GraphNode` copies `module:` onto named nodes of the
invoked component. It does not rewrite structural ids. Two rules:

- Keys are the invoked component's own node names; an unknown key is an error
- Values use the same three-state grammar as `module` (string / `null`)

**The override is single-level.** It applies only to the immediate graph of
the called component. To name nodes inside a *nested* component invocation,
wrap that nested component in another adapter with its own `inner_modules`.
Example: `chatglm_block` sets `self_attn → self_attention` (one level into
`pre_norm`); the inner attention's `qkv → query_key_value` lives in
`chatglm_attention`, a separate adapter that wraps `gqa_fused_qkv_attention`.

`inner_modules` lives on `GraphNode` only — `BlockAssignment` and `ModelLayer`
do not carry it. If a model file needs naming-only divergence, write an
adapter (1-node wrapper carrying `module` / `inner_modules`).

## Loader's component BFS

After loading every YAML in `library/` and `adapters/` into
`_component_lib`, the loader walks transitive dependencies for each
spec. The walk visits **both `node.op` and `node.params` values** — the
latter ensures components bound via `params: { attn: <name> }` get loaded
even when the binding is the only reference path. Without walking
`node.params`, a parameterized component would silently fail with
"unbound op param" at engine-eval time.

## Alias-adapter transparency

A 1-node alias adapter (`chatglm_block`, `qwen35_full_moe_block`, etc.) has the
shape `{ <name>: { op: <library>, params, module, inner_modules } }`. Evaluated as a
top-level template, the walk **inlines the library's expansion in-place**
and the wrapper node never appears.

Nothing declares it — see "Alias adapters" above for why it is read off the
graph instead. The synthesized blocks the loader builds for **homogeneous
specs** (Llama / Qwen / Gemma2 / GLM-4 — anything with `components: {
block: pre_norm, attention: ..., ffn_dense: ..., norm: ... }`) are the same
shape, so they inline by the same rule with nothing set on them.

## Checkpoint names are the file's

A snapshot stores the checkpoint's safetensors header verbatim
(`hf/header.py::SafetensorsHeader`); no segment of a header name is stripped
or folded. A spec composes each stored tensor's name from its declarations,
with `X` at each position a declaration indexes, and that name selects the
header tensors agreeing with it at every other segment. A module the file
nests under an extra segment is named with that segment in the spec.

## Engine vs rendering responsibility split

- **Evaluation** (`eval/`) evaluates the spec graph: resolves dims, applies
  `node.params` bindings + `node.rename` to sub_graphs, recurses into
  components, computes per-node FLOPs/params/shape. Nodes carry their
  **local name only** (no full path).
- **Response builders** (`rendering package`) is the **single
  authority for path composition**. It walks the engine result tree and
  the spec block tree in parallel, composing the spec node ids that match
  against HF tensor patterns.

This split means: the engine is pure "what is the graph and what does it
compute"; rendering owns "what does this node call itself in the
checkpoint." Swapping path conventions (e.g. for a different downstream
consumer with different naming) requires touching rendering only.

## Cross-layer wiring

Values that flow from layer to layer are **streams**, and there is no other
mechanism for it. Each of these answers a different question:

- **The carried stream (`hidden`).** A repeated stack declares the value it hands from one iteration to the next as a named stream: `streams: { hidden: { carries: true } }`. It is an ordinary stream with no default and no reserved name — a stack that declares none hands nothing on. It attaches at the block boundary: its initial value is the stack's `inputs[0]`, each iteration binds the block's formal 0 to its current value (the block template's calling convention, by position, not by name), and the block's output becomes the new value. It is settled at each iteration exit (`_closed`), because it is the main activation a tensor-parallel exit leaves partial, and a stack has at most one carried stream. It is local to its stack: backbone, MTP head and vision tower may each call theirs `hidden`, eval keeps them apart (`Walk.carried`, keyed by declaring layer and name), and no component the stack binds `reads:` or `stream_out:`s that name — another stack may still declare an inner or appended stream of the same name. Outside readers name the stack itself in `inputs:`. The declaration goes on the layer that repeats: a `template:` / `per_layer:` stack, a repeated `group:` around a block that does not repeat (DeepSeek's `mtp`), or — under a repeated group around a repeated stack (Nemotron-3-Ultra's MTP) — the innermost repeated stack. `tools/chain_stacks.py` lists every such stack in the corpus.
  - **Chain.** The variants of a carrying stack are walked in the order of the first iteration each covers; each enters with the previous one's exit. A variant is still evaluated once and multiplied by its count, which equals walking every iteration only if every occurrence enters with the state it was evaluated at. So a variant that occurs more than once — repeated, interleaved with another, or inside a group that repeats the chain — must exit with the state it entered with (the `close_stage` fixed point), and a variant that occurs once and changes the state must not lie between two occurrences of another variant. Both are refused during the walk, before any result is published.
- **Looped stacks (`loop:`).** A repeated stack may run a range of its stored layers more than once per forward pass with the same weights (GLM-5.4; a Loop / recurrent-depth transformer). `repeat:` still counts stored layers; the stack's iterations are its EXECUTIONS, numbered in the order they run (`0..E-1`, the loop's gate taking one after each pass), and `ResolvedSpec.execution_schedule` is the one statement of which stored layer and pass each runs (`loop_schedule.ExecutionSchedule`). Everything that walks iterations — occurrences, the carried chain, stream tables, `iteration_space` (whose depth is `SpineEntry.iterations`), the pipeline cut, fusion seams, cache slots — numbers executions, so a loop changes nothing for it. What names a stored tensor or a layer reads the projection: `BlockMeta.stored_indices` / `layer_numbers` (copied from the schedule at enumerate), `Program.stored_layers` (per-layer formats, the quant-override check), `BlockMeta.stored_count` (pages that count layers), `StagePlacement.stored_counts` (a stage holds every stored layer any of its executions runs). `tests/meta/test_loop_coordinates.py` classifies every reader of the execution coordinate. Authored numbers name stored layers: a `per_layer.assign` entry selects the executions of the stored layers it names, `pass: k` narrows it to the loop layers' pass k, and first match wins over executions. A later pass's invocations are split by the pass-1 invocation (owner) of their stored layers and store nothing: each holder is tied to the owner's holder at the same checkpoint address with the same stored declaration (`_tie_loop_passes`, `SpecIRNode.stored_declaration`); a holder the owner lacks is refused. A `reads_from` value resolves to a stored layer as it would without a loop, then to that layer's latest execution before the reader, or with `{pass: n}` its pass-n one; the chosen execution must write the stream (never walked back to another pass), and a `nearest` set's members must each write in some pass. The gate (`loop.gate`) is one invocation over its ordinals with one stored copy per pass; its formal 0 is the carried value leaving the pass and formal 1 the one entering it (`PassEntry`, resolved against the walked chain like `CarryEntry`). Not supported, and refused: a loop on an aux stack or inside a group, appended streams on a looped stack, a later pass with weights of its own.
- **Inner streams (`streams:` without `carries`, `from` or `read`).** State threaded across layers at inner nodes (DSA top-k, YOCO shared KV). A component `stream_out`s it from a node and another `reads` it; the reader declares which layer's write it reads. Every declaration that selects reading layers — a `per_layer.assign` entry, or a `template:` stack whose every layer reads — writes `reads_from: { <stream>: <layer> }`: `"i - k"` (k layers before the reading layer, in its stack), a layer number (an integer or a dim expression), `{ nearest: <set> }` — the nearest layer before the reader of a set the config lists, `{ from: config.<list>, is: <value> }` (the layers whose per-layer entry is `<value>`, the selector `per_layer.assign` uses) or `{ from: config.<list> }` (a list of layer numbers; GLM-5.2's shared layers read the nearest `indexer_types` `full` layer, V4.1's CSA consumers the nearest of `kv_source_layer_ids` / `index_source_layer_ids`) — `{ stack: <path>, layer: <number> }` for another stack's layer (`<path>` is the layer names from the top of `model:` joined by `/`), or `{ map: … }`, each reading layer's source written out (`{ "<reader>": <source>, … }` over exactly the selected layers, or `config.<list>` aligned with them). The map is the general form; the others are shorthand for one and resolve through the same table builder (`_StreamTable.sources`). A layer that needs several earlier layers' values at once reads an appended stream, not an inner one. Layers are the stack's iterations as `per_layer.assign` numbers them, in the model layer whose iterations they are (`spec_tree.iteration_owner`: the layer itself when it repeats, else the repeated group around it). The named layer writes the stream: a component on it `stream_out`s it from a node. Every layer of a `nearest` set writes it, and every reader has a member before it; no layer outside the set is a candidate, and nothing looks for a writer by the stream's name. The name is its declaring stack's — keyed `(stack, name)` (`StreamNames.inner`) — so an MTP head may declare a stream of a name the backbone also declares, and it may not also name a model layer or an appended stream. A component's formal and a stream it reads may not share a name. The pairing is one table built with the IR (`SpecIRTree.stream_reads`: `(stack, layer, stream)` → the layer named, `stream_read_forms` → the `reads_from` value as declared; `stream_writes`: the node uid each layer writes), and every consumer asks it: `Program.stream_origin` reads the producing occurrence, the Inspector card shows the declared value and each reader's source (`Program.declared_stream_reads`), the walk binds a read's producing occurrence from it (`via` `stream`), seeds a reading template with the value its producers published (`Program.block_stream_reads`; a template evaluated once for layers that read several producers holds them to one shape, layout and cache law, and refuses otherwise), and a cached node streamed this way (YOCO / CLA: later layers attend the KV one layer cached) is traced through the producers its readers' layers name by `cache.kv_access` — a reader reads the union of their caches — and by `traffic` — the writer writes and publishes the cache, the reader's attention reads it at the cache's own law and dtype (`kv_reads`), each block invocation that attends it pays that law once (one law per kernel call, however many producers its layers read), readers inside one invocation share it, and the cache is kept once. Every refusal of a reader's declaration is a `SpecRefusal` naming `reads_from`, the layers that write the stream and the config field that lists exactly them when there is one: a reading layer whose declaration names none, a named layer that does not write the stream, a `nearest` set with a member that does not write it or with none before a reader, a map that leaves out or adds a layer, is not before the reader in its stack or is outside its stack, a value that names no single layer (a layer that combines several layers' values reads an appended stream), an entry for a stream no component of the declaration reads or that is not an inner stream, a layer that writes a stream twice. Stacks whose reads name each other have no walk order and are refused. Writing a stream nothing reads is no finding. `ArchSpec.stream_names()` holds every scope — stack → carried name, appended name → its one declaring stack, `(stack, name)` → inner declaration — and every clash check is asked of it.
- **Appended streams (`from` / `after` / `index`, or `from` / `before`).** A stack appends a value to a named stream after chosen iterations, or at their entry, and the stream is the k-element sequence of those values (Qwen3-VL's vision blocks append their `hidden` after blocks 8/16/24 to `deepstack`; Kimi-K3's layers 0, 12, …, 84 append their entry to `block_residual`). `from` is the stack's carried stream, or `out` — the block's output at each iteration of a stack that carries none; `after` is a list of integers or dim expressions (EAGLE-3's default `[2, "n_layers // 2", "n_layers - 3"]`; `//` floors), a `config.<path>` list (a single integer is a list of one, HF's int `vision_feature_layer`; `?? absent` allows none) or `every`; `index` declares what an index counts (`block`, or HF `hidden_states` whose 0 is the stack's input). `before` replaces `after` + `index`: the same list forms plus `{ every: <dim> }`, indices are iteration numbers (no negatives), iteration 0's entry is the stack's input, and only a carried stream has an entry value. k is known at resolve, and a stream with k = 0 is not published. An explicit list names strictly ascending iterations, unless the declaration says `order: declared`: then the elements keep the list's own order, which may descend and repeat an index (HF's `vision_feature_layer: [-2, -5, -8, -11, 6]` indexes `hidden_states` in that order). `slice: { axis, start }` publishes each value from `start` on along `axis` (Llava's CLS crop `hs[:, 1:]`). A stream that declares either is read only joined (below); every other reading follows the iterations and takes the values whole, so it is refused at load. The name is model-wide; the `from` declaration is its one producer, and no component `stream_out`s it. Two ways for another stack to read it, both element by element, and three ways to read it whole:
  - **At the block boundary.** A stack names the stream in `inputs:`. It carries no stream and repeats k times; iteration j binds element j to the formal at that position, and its other `inputs:` are shared by every iteration (the DeepStack mergers). A stack that carries no stream — a repeated `template:` / `per_layer:` layer, or a repeated `group:` around blocks that do not repeat — has no single value to read by name, so reading one by name is refused, at any depth of the model tree: what it produces leaves it as an appended stream (`from: out`).
  - **At an inner node.** The stack declares `name: { read: each }` — a reference, not a second declaration — and a component `reads: [name]`. The readers are the iterations whose component reads it, ascending after resolution; the j-th reads element j, they number exactly k, and they are contiguous (Qwen3-VL's decoder layers 0..2 add merger j's output onto their visual rows with `scatter_rows_add`).
  - **Pairing.** `Program.stream_pairings` is the one statement of which iteration reads which element (stream, element, writer and iteration, reader and iteration, the formal or node, the reading IR uid). Stack order, validate's report and the evaluator's binding read it; the evaluator checks each binding against it. A template is still evaluated once for all its iterations, so every element a stage reads must be one tensor, as it is when the appending stack is one template.
  - **Pipeline.** `layout.stages.StagePlacement` places every occurrence (`eval.occurrence.Occurrence`, one execution of a node at one iteration). The walk binds, beside every value an input reads, the occurrences that produced it at each iteration of the reading stage (`occurrence.Reference`, `Analysis.input_references`) — `src` edges, component aliases, stage formals and the carried stream, named stages, appended elements and histories, joins, inner streams (`reads_from`, and a component's `reads:` with no edge) — and `Analysis.occurrence_graph` holds those edges. The canonical partition (`stage_layer_ranges`) cuts the iteration space of the stacks that are not aux (`Stack.is_aux`, declared by the `group:` that opens a stack; no `aux_kind` value is read); every other unit — an aux stack whole, a stage root run once — is pulled to its earliest consumer when every input is free (a parameter, a request input) or pulled, and otherwise runs after its latest input; an unconsumed pulled unit runs on the last stage if it is a model output and the first otherwise. Every edge between two stages is a hop (`occurrence.StageCrossing`): one `send_recv` record on the `pp` world per value, source iteration and cut, in the state the value is handed on in (`Analysis.handed_states`), whole, under every framework (`close.stage_crossing_records`); the readers' own transitions are their ordinary input records. `StageView` projects a stage as rows, weights and cache, each in its own stack scope; a hop is priced by the stage before its cut when a stack it serves is timed.
  - **The appending stack's own prefix (`read: all`).** Written on the appending declaration. Every occurrence of a component that `reads` the name sees, as one `[B, S, m, D]` value, the elements appended strictly before its iteration's entry: `before: j` is appended at iteration j's entry and `after: j` at its exit, so both are visible from iteration j + 1 (`AppendedStream.visible`). One occurrence reading the name twice sees one value; a component that needs its own entry in the history appends it itself (`stack_append`, m ≥ 1; `stack_init` starts a history of one). An occurrence that sees nothing (m = 0) and reads is refused. A stack has at most one such stream; its history axis is the dim `Sar`.
  - **Derived groups.** The layers one template covers see different m, so their tensors differ in shape: block enumeration (`_BlockEnumerator.visibility_groups`) splits each reading invocation by m into one `BlockMeta` per value, with its layers, its count, `Sar` bound to m and `visible` recording the key. Each group is a variant of the carried chain like any other — walked in layer order, held to the fixed point when it occurs more than once — and adds no closing point of its own, so no redistribution. Its nodes carry the group in their template name (`BlockMeta.label`, `block_kda_moe[Sar=3]`), so every per-node table keeps groups apart. The evaluator hands each group the first m elements, already walked, as one tensor (`Rule.history_state` gives its layout: the entries' own, with a whole history axis). Reports keep the groups apart; each row, card and drawn box of one carries `BlockMeta.derived_group` (its declaration, m and layers) as fields, and the pages show the model per declaration by grouping on them (`backend/views/declarations.py`: a parent per op or box with the summed figures and the `m → layers` histogram, the groups as its children) — never by reading the `[Sar=m]` name.
  - **Joined (`read: concat`).** A later stack declares `name: { read: concat, axis: <axis> }` — a reference, like `read: each` — and every node of its templates that names the stream (the outermost, when a component call and the body it binds both carry the name) reads the k elements joined along `axis`, in element order: `torch.cat` of Llava's multi-layer vision features, the EAGLE-3 head reading the backbone's auxiliary hidden states (`fc`: `k·D → D`; the carried `hidden` is the whole layer boundary, residual added, which is vLLM's `hidden_states + residual`). Each reading node joins on its own, under its template's count. The join is what `concat` is: one rank, every axis but the joined one equal, the joined axis summed — along any axis, whatever each element holds on it (`axis: 1` joins S and S − 1 rows into 2S − 1: each operand names its joined axis for itself, `OpRef.input_templates`); elements that differ elsewhere are refused. With k = 1 the element is read as it is (HF's int branch), with no join and no copy. A stream with no elements that some node joins is refused; with no reading node it is accepted.
  - **Whole-stream read at the model root.** A model-level `op:` naming an appended stream in `inputs:` reads all k elements after the stack has run: as a history `[B, S, k, D]` (Kimi-K3's `output_attn_res`), or, when the layer declares `axis:` (`ModelLayer.axis`), joined along it as above (Llava's projector). An op that takes an `axis:` of its own cannot also join. The op always runs, so a join with k = 0 is refused.
  - **Whole-stream pairing.** `Program.whole_stream_pairings` lists every reader of a whole stream or its prefix — stream, writer, reader and its iterations, how many elements it takes and the iterations that appended them in the order it holds them, the reading uid, and the representation: `history` (the stack's own `read: all`, a model-level op without `axis:`) or `concat` with its `axis`. Validate reports it, the evaluator binds from it, the pipeline gate and the graph read it; element-wise reads stay in `stream_pairings`.
  - **Joins are nodes.** The spec cannot write the `concat` — k is known only at resolve — so the program writes it: `Program.lowered()` (called by `Deployment` before the walk) copies the IR and lays down, ahead of each joining reader, a `slice` view per element when the stream publishes a crop and one `concat` of the k inputs, with fresh uids (`SpecIRTree.make_node`) and provenance `lowering`; the reader reads the join (its edge keeps the stream's name). A model-level reader's nodes are top-level siblings its stage runs first (`OpStage.prelude`); a component reader's sit just before it in its parent. The walk binds each element input by name (`WholeStreamPairing.element_ref`), and prices, lays out and draws the nodes like any other: the join reads k elements — an element named twice is read twice — and writes their sum; a crop moves nothing, so the join reads the cropped rows (`hs[:, 1:]` then `torch.cat`). A `slice` that gives `start` and no `out` keeps the rest of its axis.
  - **Joins across pipeline stages.** A join is read outside the stack that appends it, and no pipeline stage is declared for such a reader (a projector at the model root, a speculative head); no transfer between stages is priced either. At `pp > 1` the gate refuses every join.
  - **The prefix across pipeline stages.** A `read: all` history crosses a pipeline cut when its stack does: the stage before the cut sends the entries appended so far that a later layer or the model-level reader still reads (sglang's Kimi-K3 hands on `attn_res.block_residual` beside the hidden state). These are the `send_recv` rows above, beside the carried stream's own handoff.
  - **Not yet.** `after: { every: <dim> }` is refused at the schema. A group that repeats around the appending or reading stack is refused at resolve. Activation peak holds a history only while its block reads it, and the elements of a join only inside the block that joins them, not across the blocks in between; an estimate with a join says so (`joined_stream_residency`).
- **`inputs:`.** Edges between `model:` entries, bound to the reader's formals by position, one to one (`embed_tokens` reads `input_ids`, `layers` reads `embed_tokens`, `norm` reads `layers`; a model-level `op:` and a stack are held to their op's or component's arity at load). Every layer that reads a tensor names it; only a zero-input op names nothing. A vision/audio group's stages are readable by name from outside it (the text model reads the merger that replaces its image-token embeddings). A vision/audio group names no `inputs:` of its own; its first stage reads its model input. MTP names the backbone tensor. A modality encoder is exactly such a group — one that reads a media model input and names no backbone tensor (`ArchSpec.encoder_groups`, `Stack.is_encoder`); a module of the language model's step is the language model's whatever it is called (DeepSeek-V4's compressors and indexers), and a group that names no `inputs:` and reads no media input is refused.
- **`model_inputs:` and the encode phase.** What the model receives from outside its graph (`ArchSpec.model_inputs`, `ModelInputSpec`). The token ids (`kind: tokens`) are `[B, s_ctx]`, the phase's step. An image (`kind: image`, `axes: [n_img, img_h, img_w]`) or an audio clip (`kind: audio`, `axes: [n_aud, s_audio]`, a frame `width` or none for ids) declares its own axes, which the workload gives (`Workload.media`) and only the encode phase binds; an axis the config fixes is in `fixed:`, and an input whose items sit inside another's names it in `per_item_of:`. An image is read by a `patchify` layer that cites the processor (`dims: {patch, temporal, channels, align}`, `segment:`): `[N, n, H, W]` → `[N·n, (H/patch)(W/patch), channels·temporal·patch²]`, one attention segment per item; a processor's padding is `pad_rows` and its strip `unpad`. Each encoder stack is its own walk (`Program.encoding`, `Deployment.during("encode", stack=...)`), whose binding holds `B` and the media axes and none of the language model's step — an encoder node reading `s_ctx` is refused naming it. The language model's walks (`Program.walk_stacks` with no schedule) skip the encoders and read their outputs as handoffs (`Program.handoffs`): every edge from an encoder into the language model declares `prompt_slots: {domain, defines}`, and every media input that is not the items of another has exactly one `defines: true` edge (checked at load). An input the processor hands the model that the engine cannot walk yet is declared with `unmodeled:` (its reason, cited; only the items of another input, `per_item_of`) and read by no layer: a request states its item count whenever it states the input it belongs to, and any item is refused (`EngineGap`) — Step-3.7's crops. The positions a request's media fill are each defining edge's rows, read off the encode walk at one request (`Deployment.handed_over`, `Deployment.media_tokens`) — an encoder-side fact: the language model's step is `input_tokens` and its handoffs carry no rows, so no prefill or decode number moves with the media, and the encoders' weights and activations are their own accounting, none of the language model's capacity. A question about the whole checkpoint — its tensors, its weights — walks every stack, each encoder in its own encode binding (`Deployment.checkpoint`). Readers name model inputs like any stage; the walk binds a model input or a handoff as free (nothing this walk produced it), a model input under its name (`Analysis.model_input_read`), and holds each reader's declared width to it as it binds the read (`Program.check_model_input_reader`).
- **`tied_weight`.** Two different ops share one stored table (embedding ↔ lm_head, or a second call site of `lm_head`). Storage counted once; FLOPs at each call site. Weight-holding siblings at one `module:` address are tied the same way to the parameter stored there (absorbed MLA's per-head halves of `kv_b_proj`); each reads its declared share.
- **`checkpoint.index`.** File path, not a dataflow edge.
- **`cache`.** Decode-time state, not layer-to-layer.

## Stream glossary

The terms of cross-layer wiring, each with one meaning. `streams:` is the
field on a stack (`ModelLayer.streams`); a stream is one entry of it
(`StreamSpec`). There is no other kind of cross-layer dataflow.

| term | meaning |
|---|---|
| **stream** | A value that flows across the iterations of a repeated stack, declared under that stack's `streams:`. Its kind follows from the fields it declares. |
| **carried stream** | `carries: true`. The value the stack hands from one iteration to the next, at the block boundary: its initial value is the stack's `inputs[0]`, each iteration binds it to its block's formal 0, and the block's output becomes its new value. Its pending layout is settled at every iteration exit. At most one per stack, local to that stack. The corpus names it `hidden`; the name is not reserved. |
| **inner stream** | None of `carries`, `from`, `read`. State threaded across layers at inner nodes: a component `stream_out:`s it from a node, another `reads:` it (`dsa_topk`, YOCO's `shared_kv` from a cached node). Each reading layer's declaration names the layer it reads, `reads_from`. Name local to its declaring stack. `seed:` names an earlier model-level `op:` whose output the stream holds at the stack's entry (a first layer's `"i - 1"` reads it); a model-level `op:` after the stack that names the stream in `inputs:` reads its final value, the last layer's write. |
| **reads_from** | On the declaration that selects an inner stream's reading layers: per stream, the layer of the stack whose write they read — `{map: …}` per reading layer, or the shorthands `"i - k"`, a layer number, `{nearest: <set>}` (the nearest earlier layer of a config-listed set), `{stack, layer}`. The one statement of which layer a reader reads (`SpecIRTree.stream_reads`). |
| **appended stream** | `from` with `after` + `index`, or `from` with `before`. The k-element sequence of values the stack appends at chosen iterations. Model-wide name; `order: declared` and `slice` shape what a join reads. |
| **reader modes** | How a stack reads an appended stream. `read: each`: iteration j reads element j at an inner node (a stack that names the stream in `inputs:` reads it the same way at its block boundary). `read: all`: on the appending declaration; each iteration reads the elements appended before its entry as one `[B, S, m, D]` history. `read: concat`: a later stack's node reads the k elements joined along `axis`. |

The same word also names things that are not streams in this sense:

- **KV stream** — one cached sequence of key/value positions inside an
  attention component. `kv_stream_concat` joins two of them along the
  sequence axis (DeepSeek-V4's window cache and compressed cache), and the
  sliding-window check asks for "a bounded KV stream", a cached tensor
  whose storage stops growing with `s_kv`. A marker op on one
  (`token_pool_compressor`, `sliding_window_kv`) states how many positions
  it holds, and a reader of it reads that many rather than `s_kv`: an
  indexer scoring pooled keys reads ⌊s_kv/m⌋. Neither is declared under
  `streams:`.
- **Token stream** — the sequence of tokens a `dp` or `cp` world cuts.
- **"streams" as a verb** — traffic: a kernel streams bytes through HBM, a
  tied LM head streams the embedding's stored tensor.

## SpecIRNode, Analysis and reports

There is one structural graph and one evaluation result:

- **`SpecIRNode` / `Program`** carry graph identity and topology:
  `uid`, `display_path`, `aggregation_id`, inputs, role and component scope.
- **`Analysis`** carries derived facts as `Layer[T]` keyed by those UIDs:
  cost, shape, instances, storage, cache, formats and distributed state.
- **`NodeReport` / `ModelReport`** are optional read-only presentation trees.
  Validation, the inspector and API serializers use them when a nested wire
  shape is convenient. They do not own facts and inference never consumes
  them as another evaluation.

The separation is lifecycle, not compatibility. A resolved program is shared
across bindings; each binding yields an `Analysis` without mutating the graph.
A number at a point and the formula over all points come from the same walk:
the workload's counts are symbols in the `Binding`, every fact remains
joinable through stable UID identity, and a point is read off the result.

Runtime graph changes follow the same rule. Communication materialization and
framework fusion create graph nodes and a `Rewrite`; cost, shape, instances and
traffic each cross that rewrite with their own combine law. No synthesized
object copies every field from an old tree into a new one. `provenance`
(`"spec"`, `"fusion"`, `"comm"`) records where a structural node originated;
it is not an alternate result type.

## noaux_tc grouped routing

`router_noaux_tc` encodes the two-stage grouped top-K routing used by
DeepSeek-V3 / Kimi-K2 / GLM-MoE-DSA / MiMo-V2. The pipeline:

1. `gate` → linear projection `d_model → n_route_targets` (sigmoid scoring)
2. `gate.sigmoid` → element-wise sigmoid
3. `gate.e_score_correction_bias` → load-balance bias parameter
4. `scores_biased` → scores + bias (used for routing decisions only;
   final weights gather from pre-bias scores)
5. `group_top2_sum` → reshape `[n_expert_groups, n_route_targets / n_expert_groups]`,
   take top-2 per group, sum
   (the "2" is a paper constant — top-2 sum is more stable than top-1
   max for ranking groups)
6. `group_topk` → pick top `n_groups_active` groups by group score
7. `group_mask` → expand selected groups into a `[n_route_targets]` mask, multiply
   with `scores_biased` to zero out unselected groups
8. `topk_indices` → top `n_experts_active` experts from masked scores
   (effectively n_experts_active from
   `n_groups_active × n_route_targets / n_expert_groups` candidates)
9. `topk_gather` → gather corresponding weights from pre-bias scores
10. `topk_norm` → normalize selected weights to sum=1
11. `topk_scale` → multiply by `routed_scaling_factor`

When `n_expert_groups=1` (`config.n_group`, which the spec binds), steps 5-7 degenerate — the group machinery stays
in the graph for visualization consistency but its FLOPs collapse to
small constants. This lets one `router_noaux_tc` component cover both
"grouped routing" (n_expert_groups>1) and "plain noaux_tc top-K"
(n_expert_groups=1) by
configuration alone. Models that don't use noaux_tc at all use a different
component (`moe_ffn_routing_bias` for MiniMax-M2, `moe_ffn_fused_scaled_shared`
for Qwen3.5-MoE).

## The symbolic system: one walk, two halves

The thing that makes specs **executable** rather than just well-typed
documents is the evaluation core in `eval/`. There is no symbolic engine
beside a numeric one: the same `Evaluator` walks the same `Program` once,
with the workload's counts (`Workload.symbolic()`) as symbols and the model's
dims as numbers. Every extent is a `Dim` with two halves: the identity half
(`expr_of`) is the quantity in the model's names, which the dashboard prints
as the formula, and the law half (`value_of`) is the law in the counts,
which a point reads (`Analysis.at(Point(...))`) for the figure beside it. The
formula and the figure are two readings of one walk, so a tooltip cannot
disagree with the number beside it. Per-rank geometry is the layout rule's,
not a divisor on the binding.

The full treatment — the program/binding/evaluation trio, the layer
taxonomy (propagated / walk-computed / sideband / derived), why bytes are
derived rather than carried — lives in
[`docs/evaluation_model.md`](../../docs/evaluation_model.md). Expression
mechanics (the `_DotPrinter`, dim parsing and caching, axis identity) live
with their code in `ops.py` and `eval/dim.py`.
