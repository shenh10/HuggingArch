"""Task-specific prompt bodies + the public `build_*_prompt` /
`prepare_*_prompt` entry points.

Two distinct tasks share the system prompt + ground truth, only the task
body differs:
  • generate — agent authors a NEW spec.yaml from scratch.
  • refine — a spec.yaml already exists but params don't match HF; agent
    enters Phase 4 of the workflow and iterates the architecture validator.
  • custom — local user-provided model: uses the user's config.json + forward
    source and, for a local HF-format repository, scanned tensor metadata.

`prepare_*_prompt` returns a dict containing the prompt PLUS
`prefetch_logs` (shown to the user before the agent starts) and
`snapshot_data` (so the post-success writer doesn't re-fetch).
"""
from __future__ import annotations

import logging
import sys

from backend.custom_id import is_custom_id, model_path as _custom_path, strip_prefix
from backend.hf.metadata import format_header
from backend.hf.vllm_source import VllmModelSource

from .system import build_system_prompt, forward_alignment_doc, source_files_section
from ..overlay import _safe, overlay_models_dir, stage_forward_source, stage_vllm_source
from .._ground_truth import (
    ForwardSourceUnavailable,
    full_module_sources,
    get_ground_truth,
    assemble_task_prompt,
)
from .._metadata import build_config_section
from .._stream import build_prefetch_logs

logger = logging.getLogger(__name__)

# Interpreter the agent should invoke for validate/structure commands.
# Resolves to whatever venv runs the backend (e.g. .venv locally,
# /opt/venv in the container) — never a hardcoded layout.
_PY = sys.executable


def shell_rules(model_id: str) -> str:
    """What the shell runs in an isolated agent workspace, as
    `ClaudeBackend._isolated_command` allows it: its `Bash(...)` rules
    match these commands only as spelled here, and its `dontAsk` mode also
    lets read-only commands inside the working directory run while denying
    everything else. Review and fix say it with this one text."""
    return (
        f"Run `{_PY} -m backend.inference.validate {model_id}`, "
        f"`{_PY} -m backend.arch.structure {model_id}` and "
        f"`{_PY} -m backend.arch.component_reuse <component.yaml>` exactly as "
        "written, each on its own, from the working directory you start in: "
        "no `cd`, no chaining (`&&`, `;`, `|`), no shell variables; spelled "
        "any other way, the call is denied — which is also why the reuse "
        "query reads a component FILE in this working directory, never piped "
        "input. Besides these three, only read-only commands "
        "(`ls`, `cat`, `grep`) on files inside this working directory run; a "
        "command that writes, or that reads outside this working directory, "
        "is denied."
    )


def _build_prompt_context(model_id: str):
    """Build (system, ground_truth, meta) — shared between generate / refine."""
    ground_truth, meta = get_ground_truth(model_id)
    return build_system_prompt(), ground_truth, meta


def _stage_source(target_root: str | None, model_id: str, meta: dict) -> list[str]:
    """Obtain the model's complete modeling / configuration files and stage
    them into the run overlay for `file:line` citations. Shared by generate,
    refine, review and fix, before any agent starts: raises
    `ForwardSourceUnavailable` when no real module source can be obtained.
    Without an overlay (direct calls; the agent is not sandboxed) the files
    are listed at their own paths."""
    snap = meta.get("_snapshot_data") or {}
    files = full_module_sources(model_id, snap.get("config") or {},
                                snap.get("forward_source"))
    if not target_root:
        return [f["path"] for f in files if f.get("path")]
    staged = stage_forward_source(target_root, files)
    # vLLM's implementation beside it: the reference for what the modeling
    # files do not run (an MTP head transformers drops on load). Optional:
    # a run without it still has its modeling source.
    vllm = VllmModelSource.for_config(snap.get("config") or {})
    if vllm is not None:
        logger.info("staged vLLM %s source for %s: %s", vllm.commit[:12], model_id,
                    ", ".join(vllm.architectures))
        staged += stage_vllm_source(target_root, vllm.files)
    return staged


def continuation_after_op_prompt(base_prompt: str, op_name: str) -> str:
    """Second-turn generate prompt after the op writer landed a primitive."""
    return (
        f"{base_prompt.rstrip()}\n\n---\n\n"
        f"# Continuation: primitive `{op_name}` is available this run\n\n"
        "The previous attempt stopped because this kernel could not be "
        "composed from the built-in table. The op-proposal worker has "
        f"registered `{op_name}` for THIS run's "
        "`python -m backend.inference.validate` (do not write another "
        f"`gap.json` for `{op_name}`). Compose the spec using it. Do not "
        "edit `ops.py` / `ops_registry.py`.\n"
    )


def build_generate_task_body(model_id: str, meta: dict, *,
                             target_root: str | None = None,
                             source_files: tuple | list = ()) -> str:
    """Build the task-specific body for new spec generation.

    When `target_root` (a per-run overlay root) is given, the agent's write /
    edit targets are redirected there instead of the shared repo tree, so
    concurrent generate/refine runs of the same model stay isolated. The
    in-loop validate/structure commands find the overlay via
    `HA_SPEC_OVERLAY_DIR` (injected into the agent env), so their command
    text is unchanged.
    """
    import os as _os
    from .. import runtime as _rt
    # Shadow the shared-tree path constants with overlay paths for this body
    # (function-local; the module-level constants other builders use are
    # untouched). Fall back to the shared tree when no overlay is given.
    _MODELS_DIR_REL = (_os.path.join(target_root, "models")
                       if target_root else _rt._MODELS_DIR_REL)
    _COMPONENTS_DIR_REL = (_os.path.join(target_root, "components")
                           if target_root else _rt._COMPONENTS_DIR_REL)
    unknown_guidance = build_unknown_structure_guidance(meta)
    return f"""# Task: Generate spec for {model_id}

## Instructions

Follow the four-phase workflow from the system prompt (Structure Inventory
→ Component Audit → Spec Authoring → Validate & Iterate). The Ground Truth
sections above are your Phase 1 input.

**Class → component mapping reference** (use during Phase 1 / Phase 2):
  - `*Attention(q_proj, k_proj, v_proj, o_proj)` → gqa_attention (or swa_attention if sliding window)
  - `*Attention(q_a_proj, q_b_proj, kv_a_proj, kv_b_proj)` → mla_attention (Q compression)
  - `*Attention(q_proj, kv_a_proj_with_mqa, kv_b_proj)` → mla_no_qcomp_attention
  - `*MLP(gate_proj, up_proj, down_proj, SiLU)` → swiglu_ffn
  - `*MLP(fc1/up_proj, fc2/down_proj, GELU)` → gelu_ffn
  - `*Moe(experts + gate + shared_experts)` → moe_ffn
  Check if different layers have heterogeneous structures — if so, use
  `per_layer:` (an `assign` list of generators — range/layers/stride/
  `from: config.X, is: v`/`from: config.X` (a list of layer numbers)/rest —
  tagged with a `type`, plus a `map` binding
  each type to its attention/ffn/template/dims). Read a per-layer config
  list via `from:` instead of hardcoding indices.

**Phase 3 target path**: `{_MODELS_DIR_REL}/{_safe(model_id)}.yaml`.
    This exact file is the only identity spec this run will harvest; base or
    intermediate model YAML must not use that filename. Dims come from Linear
    in_features/out_features in the architecture tree mapped to config.json
    field names.

**Phase 3 reuse priority**: first check the Formal Base Inventory and write a
thin identity spec with `base:` when a reviewed base already expresses the structure;
next reuse a formal component or merged draft from `library/`, `adapters/`, or `drafts/`
(keeping any draft in top-level `drafts:`); only otherwise author a new draft.

**Phase 3.5 — forward-alignment reflection (YOUR judgment; no complete checker).**
After authoring, run `{_PY} -m backend.arch.structure {model_id} --spec <name>`
(compiled nodes, op + dataflow `<- inputs`) and map it against the forward
source in BOTH directions, as the reviewer will:
  - every step of the real `forward` (cited as below) → the spec node that
    models it. A step with no node is under-modeling — add it;
  - every spec node → the forward step it comes from. A node with no source
    basis is invented — drop it.
  - dataflow must match too: same op types, same edges (which hidden feeds
    what), same switches and layer patterns.
{forward_alignment_doc()}
{source_files_section(source_files)}
The review that follows rejects any unmapped forward step, unsourced node or
unnamed reference choice. Fix every discrepancy before submitting.

**Phase 3.6 — the cache's growth is in the graph.** Validate grows every
call's cache from the graph you wrote. A sliding-window cache is a
`sliding_window_kv` over the stream `cache.entries` names, with the model's
window as `s_window`; a fixed-size recurrent state is a `state:` entry under
`cache.entries`; a full-attention cache is the cached node itself. Read the
source's cache allocation and eviction while it is in front of you and make the
graph say the same.

**Phase 4 command**: `{_PY} -m backend.inference.validate {model_id} --spec <name>`
Then re-run without `--spec` to confirm auto-match works. Phase 4 output
includes `axiom_warnings` (all gating: the library's source claims and the
cache contracts, judged at every call of a component in the model's analysis),
`type_contract_warnings` (gating), and `module_exec_warnings` (non-gating:
does the spec actually compute). Resolve them according to those semantics
(see the kinds below).

{unknown_guidance}

## General constraints (phase-independent)

These apply at all times. Phase-specific rules live inside the workflow
phases in the system prompt above.

- NEVER download model weights or safetensors files — all info is in this prompt.
- **NEVER write new components under `{_COMPONENTS_DIR_REL}/library/` or
  `{_COMPONENTS_DIR_REL}/adapters/`.** Those are review-gated, globally
  shared tiers — your additions there leak into every other model spec
  and pollute the manifest the next agent run sees. ALL new components
  MUST go to `{_COMPONENTS_DIR_REL}/drafts/<name>.yaml`, AND every name
  you create MUST appear under your spec yaml's top-level `drafts: [...]`
  list (without that entry the loader cannot see your file and reports
  the component as unknown).
- NEVER Read / Glob / Grep Python source files (engine.py, loader.py, ops.py,
  schema.py, FastAPI routes, validate/, promote/, ANY .py file). You MAY read
  YAML in `{_MODELS_DIR_REL}/`, `{_COMPONENTS_DIR_REL}/`, the shared
  `backend/arch/dims.yaml`, and `backend/arch/quant_schemes/`. Edit
  only your model YAML and new files under `{_COMPONENTS_DIR_REL}/drafts/`.
  Existing components and shared registries are read-only to this worker.
- Tool allowance: Write (YAML), Edit (YAML), Bash (validate + structure +
  reuse only — `python -m backend.inference.validate`,
  `python -m backend.arch.structure` and
  `python -m backend.arch.component_reuse <component.yaml>`).
- Component graph keys (`router`, `expert`, `gate_proj`, ...) are
  component-INTERNAL names; the model spec just references a component by
  name. Do not try to rename component internals from the model spec.
- The tensors the Weight Tensors section lists under "Quantization metadata"
  (the scheme's scale / zero-point / index sidecars) are 0 params.
  HF `total_params` is dtype-agnostic element count; do NOT divide linear
  weights by dtype byte size.
- When the validator reports `quant_method_unknown`, the checkpoint's
  `quant_method` names no registered scheme. Describe its tensors in
  `quant_context`: per_role / overrides rows whose `method:` is the registered
  scheme (`backend/arch/quant_schemes/`) those tensors and their metadata
  siblings follow, or `none` for tensors stored unquantized. The storage,
  coverage and packing gates check that description against the checkpoint
  header.
- For zero-FLOPs tensors HF counts toward `total_params` but that don't fit
  an existing op (`rotary_emb.inv_freq`, novel bias banks), declare a
  `custom_ops:` entry with `flops: "0"` and `params:` matching the element
  count. Do NOT repurpose `rmsnorm` / other semantic ops as accounting tricks.
- A MoE spec binds two counts: `n_experts`, how many experts HOLD WEIGHTS,
  and, when the router has destinations that hold no weights (LongCat's
  `zero_expert_num` reserve, dispatched to `nn.Identity()`), `n_zero_experts`.
  The walk derives the rest in every frame: the bank's expert axis
  `n_expert_copies` (= `n_experts`) and the destinations the top-k draws from,
  `n_route_targets` (= `n_experts + n_zero_experts`), which is also the width
  of the gate that scores them. A component reads `n_route_targets` for the
  gate's output; a gate that also scores other lanes writes that width in its
  own component (Inkling's `n_route_targets + n_shared_experts`).

## Missing-primitive escape hatch (HARD STOP)

The primitive ops table in the system prompt is the complete built-in inventory.
If — and ONLY if — none of the listed primitives composes into the
forward you're modeling AND the gap is a SINGLE opaque kernel (not a
multi-projection submodule that should be a component), first use a
`custom_ops:` declaration when its schema-limited shape, cost, port, and weight
contract is sufficient. You MUST NOT:
  - skip the kernel in your draft component ("dead-end" dt-like nodes
    that go nowhere — this is the Mamba-2 hack we explicitly forbid),
  - bundle the kernel into a `custom_ops:` entry whose `params:` formula
    sums multiple distinct `d_model * <dim>` terms (that's a hidden decomposition;
    see Anti-patterns above), or
  - rename an unrelated primitive (`rmsnorm` / `matmul`) to stand in
    for the missing kernel.

If the kernel needs behavior `custom_ops:` cannot author (for example a Python
shape function, sharding mode, KV-position law, or view/communication status),
emit a structured gap statement so a separate op-proposal worker can propose
the missing primitive cleanly:

  1. Write a JSON file to `logs/op_proposals/<model_safe_id>__<ts>.gap.json`
     where `<model_safe_id>` is the model_id with `/` → `__` and `<ts>`
     is `YYYYMMDD_HHMMSS`. JSON shape:

     {{
       "op_name": "<short snake_case name you propose>",
       "context_model_id": "<the model that surfaced the gap>",
       "kernel_source_hint": "<the function/class in the HF modeling
            source (or vLLM / SGLang) that implements this kernel,
            plus the file path if you know it>",
       "signature": "<inputs → output description, e.g.
            (hidden_states, dt, A_log, B, C, D, dt_bias) → y> (these are source argument names, not authored dim symbols)",
       "input_shapes": ["<shape tuple of each positional input>"],
       "output_shape": "<shape tuple of the output>",
       "why_no_existing_op_composes": "<one paragraph explaining why
            every relevant existing primitive in the system prompt's
            ops table either has a different signature or contracts on
            a different axis>",
       "candidate_new_dim_symbols": ["<symbol1>", "<symbol2>"]
     }}

  2. Immediately stop. Do NOT write any draft component or model yaml
     that depends on the missing primitive. The next agent run (an
     op-proposal worker triggered on this gap file) will land the
     primitive in `ops.py`, then the spec worker re-runs and composes
     cleanly.

This escape hatch is RESERVED for genuinely opaque kernels (Mamba-2
selective scan, novel recurrences, custom CUDA-only ops). It is NOT
a shortcut for "I couldn't be bothered to compose existing primitives"
— if you write a gap for something composable, the op-proposal worker
will reject your proposal and you'll have wasted everyone's time."""


def build_refine_task_body(
    model_id: str,
    spec_storage_bytes: int,
    hf_storage_bytes: int,
    storage_diff: int,
    meta: dict,
    *,
    target_root: str | None = None,
    source_files: tuple | list = (),
) -> str:
    """Build the task-specific body for spec refinement.

    Diff is in BYTES. The agent must close the storage-bytes gap, not
    the logical-params gap — bytes is the only unit that holds for
    mixed-precision checkpoints (FP4 routed experts + FP8 attention +
    BF16 norms). Per-tensor `param_mismatches[i]` (returned by
    the validator) compares physical elements per checkpoint pattern and
    gates publication; it is the surgical signal, and the headline diff
    in bytes tells the agent how far overall reconciliation has progressed.

    `target_root` (a per-run overlay root, service layer only) redirects the
    agent's read/edit targets there — the same isolation `build_generate_task_body`
    uses — so a signed-in user refines their OWN private spec (seeded into the
    overlay) without touching the shared repo tree. In-loop validate/structure
    resolve it via `HA_SPEC_OVERLAY_DIR`.
    """
    import os as _os
    from .. import runtime as _rt
    _MODELS_DIR_REL = (_os.path.join(target_root, "models")
                       if target_root else _rt._MODELS_DIR_REL)
    _COMPONENTS_DIR_REL = (_os.path.join(target_root, "components")
                           if target_root else _rt._COMPONENTS_DIR_REL)
    unknown_guidance = build_unknown_structure_guidance(meta)
    diff_pct = (storage_diff / hf_storage_bytes * 100) if hf_storage_bytes else 0
    return f"""# Task: Fix storage-bytes discrepancy for {model_id}

Current state (storage bytes — physical disk size, mixed-precision safe):
  Spec computed: {spec_storage_bytes:,} bytes  ({spec_storage_bytes / 1e9:,.2f} GB)
  HF actual:     {hf_storage_bytes:,} bytes  ({hf_storage_bytes / 1e9:,.2f} GB)
  Diff:          {storage_diff:+,} bytes  ({diff_pct:+.4f}%)

## Instructions

You're jumping into **Phase 4 (Validate & Iterate)** of the workflow — the
seeded identity is `{_MODELS_DIR_REL}/{_safe(model_id)}.yaml` and validation
does not pass. Keep that exact filename; do not replace it with a base or
intermediate model file.

1. Read the existing spec.
   Before changing graph structure, apply the system prompt's authoring
   priority: reuse a formal base, merged draft, or formal component before
   creating a new component. Keep a reused merged draft listed in `drafts:`.
2. Run `{_PY} -m backend.inference.validate {model_id}` —
   the stderr output tells you exactly what's wrong:
   - `param_mismatches[i].ratio_hint` → if multiple nodes share a
     ratio (e.g. "0.5×"), ONE dim binding is off by that factor.
   - a `checkpoint_name_absent` error → a spec node stores a tensor
     under a name the checkpoint does not hold; the header tensors it
     lists for that module are your rename target.
   - `unmatched_hf_tensors` → tensors the spec doesn't cover
     (often MTP / extra norms / shared embeddings).
   - `shape_mismatches[i]` → cross-input geometry violation:
     `concat` / `attention_score` / matmul got inputs whose
     ranks/dims don't compose. Common: forgot a `head_broadcast`
     before a 4-D concat, or transpose `axes:` is off-by-one.
     Each entry's `shapes` field shows the exact mismatched
     tensors so you know which upstream node is wrong.
   - `axiom_warnings[i]` → a gating fact about a call's cache failed.
     `axiom_kv_*` → a library component's source claim (`claims/components/`)
     disagrees with the cache your graph grows at that call; the claim quotes
     the source, so fix the graph. `swa_cache_unbounded` → a component with a
     sliding-window `cache.family` keeps a cached tensor that grows without a
     cap: put a `sliding_window_kv` (window `s_window`) over every stream
     `cache.entries` names — K and V alike. `windowing_declared_but_unmodelled` → active
     `config.sliding_window` but no attention call keeps a capped cache; model
     the window or, only when the field is genuinely unused, list
     `sliding_window` under top-level `validation_exemptions.sliding_window`
     (with a reason). `axiom_unjudged` → the model's analysis failed, so no
     cache was judged; fix the analysis error validate reports alongside it.
   - `type_contract_warnings[i]` → declared KV/attention semantics are
     inconsistent. The legacy-named `attention_missing_kv_label` /
     `..._kv_cached` diagnostics mean the component needs the current
     `cache: {{family, entries}}` declaration.
   - `module_exec_warnings[i]` → the engine actually RAN your block at
     B=1/2,S and it misbehaved — `[NEW module]` flags a freshly authored
     one: `module_exec_error` → a node's custom op crashed
     (wrong dims / arity / missing op); `module_nonfinite` → it produced
     inf/NaN/negative flops or params (a bad dim expression);
     `module_batch_nonlinear` / `module_batch_noninvariant` → forward
     FLOPs aren't ×2 under B→2B, or params changed with batch (a node
     using B where it shouldn't). These mean the spec doesn't compute
     correctly — fix the offending component, don't suppress.
3. Change ONE thing per iteration. Re-validate until top-level `valid` is true;
   that is the publish condition. Checkpoint-tensor coverage, per-tensor
   element agreement (`param_mismatches`), and storage-byte equality are
   gating checks. Then resolve or explain `unmatched_spec_nodes`
   and `shape_mismatches`, which are non-gating review diagnostics. Legacy
   `total_params.diff` is reported but unreliable for mixed-precision models.
4. **Forward-alignment reflection (YOUR judgment).** Once the numbers match,
   run `{_PY} -m backend.arch.structure {model_id}` to see your spec's
   compiled structure (components inlined, op + dataflow per node) and map
   it against the forward source above in both directions — every forward
   step → a spec node, every spec node → a forward step — by the standard
   below. The review that follows applies the same standard. The structural
   validator's warnings are best-effort; they do not prove equivalence.

{forward_alignment_doc()}
{source_files_section(source_files)}

Notes:
  - Quantization metadata tensors (the scheme's sidecars, listed apart in the
    Weight Tensors section) are 0 params (never "fix" them).
  - Extra repeated modules (e.g. MTP) belong as model-level `repeat:`
    layers, not in `group:`.
  - If the root cause is a missing component type not in the library, author it
    under `{_COMPONENTS_DIR_REL}/drafts/`, list its stem in top-level `drafts:`,
    and check Phase 2's reuse discipline first.

{unknown_guidance}

## Rules

- NEVER download model weights or safetensors files — all info is provided above
- NEVER set `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1` when running
  the validator. The first call for a large MoE checkpoint can take 10–30s
  while it reads safetensors shard headers; results are cached on disk, so
  the second call is fast. Forcing offline can make HF reads return empty
  metadata; the weight gate then fails with
  `Could not fetch model_storage_bytes from safetensors` and cannot provide a
  useful numeric diff. If validation
  appears stuck, just wait — do not "fix" it with env vars.
- NEVER Read/Glob/Grep Python source files — all system info is in this prompt.
  Read shared YAML registries as needed, but edit only the target model YAML and
  new scoped draft components.
- NEVER modify existing component files unless there's a genuine bug
- NEVER create `*_fp8*.yaml` component variants. FP8 weights count as plain
  element counts; the scheme's sidecar tensors are separate quant metadata and do
  NOT contribute to total_params. Reuse the standard non-FP8 components.
- Do NOT divide linear weights by dtype byte size — HF `total_params` is
  dtype-agnostic element count.
- Prefer adding model layers over creating new components
- Do NOT distort structure merely to make the param count match
- If a new repeated namespace appears inside attention or FFN, create a new component before creating a top-level parameter group
- For zero-FLOPs tensors HF counts toward total_params (`rotary_emb.inv_freq`
  etc.), use `custom_ops:` — don't repurpose semantic ops like rmsnorm.
  Reminder: `custom_ops:` is for ATOMIC primitives only (single param term);
  multi-projection submodules go in a NEW component yaml — see Decision Tree.
- If you find an existing spec uses an aggregate `custom_op` that bundles
  multiple Linear/Conv weights into one formula (sum of distinct
  `d_model * <dim>`
  terms), replace it with a proper component decomposition during refine.
- Your fix must make the validator return `valid: true`; storage-byte diff zero
  is one enforced part of that result.
- Only use Write/Edit (YAML files) and Bash (`backend.inference.validate` +
  `backend.arch.structure`) — nothing else"""


def build_unknown_structure_guidance(meta: dict) -> str:
    """Build extra discovery guidance only when transformers structure
    extraction failed."""
    if meta.get("has_transformers_structure"):
        return ""

    lines = [
        "## Unknown-Structure Guidance",
        "",
        "The current transformers environment could not build this model skeleton,",
        "so do NOT force this model into the closest known component. Treat config",
        "fields, tensor namespaces, and any official implementation repo as primary",
        "evidence of new structure.",
        "",
        "### Decision tree for new patterns",
        "",
        "**STEP 1 — Identify named submodules in forward()**",
        "Read `DecoderLayer.forward()` (and any sub-class forwards) from the",
        "Ground Truth section. List every `self.<NAME>(...)` call where",
        "`self.<NAME>` is a complex nn.Module (multi-projection class), not a",
        "single Linear/Norm. Examples: `self.self_attn`, `self.mlp`, `self.linear_attn`,",
        "`self.mlp.experts`, `self.mlp.shared_expert`. Each named submodule should",
        "be ONE node in the block graph — invoking either an existing component or",
        "a new component you create. NEVER inline a named submodule's internal",
        "projections directly into the block graph.",
        "",
        "**STEP 2 — For each named submodule, classify**",
        "",
        "1. **Exact match to existing component** (same op graph, same dim shape) →",
        "   Reuse it. `op: mla_attention`, `op: swiglu_ffn`, `op: moe_ffn`, etc.",
        "   This is >70% of cases.",
        "",
        "2. **Structural variation of existing component** (existing op graph plus",
        "   one twist: fused gate_up projection, scalar shared-expert gate, partial",
        "   RoPE, head-wise gate, KV-rope concat into o_proj, ...) →",
        "   Create a NEW component yaml under `components/drafts/` with a",
        "   STRUCTURAL name describing the variation: `moe_ffn_fused_scaled_shared`,",
        "   `gqa_qknorm_partial_rope_gated_attention`. Decompose internally into",
        "   existing primitives. Then invoke it from the block graph as ONE node.",
        "   NOT acceptable to inline 15+ raw `linear`/`slice`/`mul` nodes into the",
        "   block graph, and list its stem in the model spec's top-level `drafts:`.",
        "   If the only delta is checkpoint naming, a scoped draft wrapper around",
        "   the existing library component can carry bindings and",
        "   `module:` / `inner_modules:`.",
        "",
        "3. **Genuinely new architecture family** (linear-attn / DeltaNet / Mamba /",
        "   SSM, RWKV, Hyena, MoE-with-shortcut, novel patch embed) →",
        "   a) FIRST scan the primitive table in this prompt for the required",
        "      kernel (e.g. `conv1d`, `parameter_vector`, `gated_delta_rule`).",
        "      If an opaque primitive is absent, use the structured gap workflow",
        "      below and stop; this worker does not edit Python.",
        "   b) THEN create a component yaml decomposing the named submodule into",
        "      one node per Linear / Conv / parameter bank / recurrence kernel.",
        "      Reference `forward()` line-by-line — each line in `forward()` should",
        "      map to one or two nodes in your component graph.",
        "   c) Invoke from the block graph as ONE node.",
        "",
        "4. **Truly atomic novel primitive** that resists decomposition (a single",
        "   elementwise kernel or opaque numeric formula) → declare a small",
        "   spec-scoped `custom_ops:` entry when its authorable shape/cost contract",
        "   is sufficient. Otherwise emit the structured gap and stop.",
        "   `custom_ops` is RESERVED for atomic ops — its `params:` formula must be",
        "   ONE term, never a sum of multiple `d_model * <dim>` projections.",
        "",
        "**Anti-patterns (failure modes from past spec generations)**",
        "- ❌ Bundling a multi-projection submodule into a single `custom_op` whose",
        "  `params:` formula sums 4-5 distinct `d_model * <dim>` terms. That's a hidden",
        "  decomposition. Decompose it.",
        "- ❌ Inlining MoE / attention / FFN internals as 10-20 raw primitive nodes",
        "  in the block graph. Wrap them in a component.",
        "- ❌ Naming a structural component after the model (`step3p5_*_attention`,",
        "  `qwen35_linear_attention`). Use structural names so future variants reuse.",
        "- ❌ Authoring a component that itself inlines 10+ primitives with a",
        "  shared prefix (`expert.gate_up`, `expert.slice`, `expert.silu`, ...).",
        "  Each named sub-branch should be its own sub-component. Ratio of",
        "  inline primitives to component nodes is a code smell: if a new",
        "  component has >12 primitive ops with prefix clustering, you missed",
        "  an extraction.",
        "- ❌ Treating `group: true` as a way to express the main backbone. Group is",
        "  ONLY for auxiliary stacks (MTP, vision/audio towers, speculative heads).",
        "- ❌ Flattening an auxiliary module's non-trivial DAG into a sequential",
        "  `group.layers: [...]` list. If the forward has parallel branches, skip",
        "  connections, or a concat → projection join (e.g. DeepSeek MTP's enorm &",
        "  hnorm feeding a concat), declare an inline block in top-level `blocks:`",
        "  and reference it as the group's template. `group.layers` is strictly",
        "  sequential — the rendered DAG will lie about the structure otherwise.",
        "- ❌ Naming block-internal nodes with semantic aliases (pre_attn_norm,",
        "  pre_ffn_norm) instead of the exact transformers forward submodule names",
        "  (input_layernorm, post_attention_layernorm, self_attn, mlp). The per-node",
        "  dtype matcher looks up `<prefix><node.id>.weight` against checkpoint",
        "  keys. Do not rewrite a structural node id to match a checkpoint",
        "  spelling — use `module:` / `inner_modules:` for that.",
        "",
        "Default: prefer Step 2 → Step 3 → Step 4 in that order. The closer you",
        "stay to existing components, the easier the spec is to maintain.",
    ]
    return "\n".join(lines)


# --- Public entry points ---------------------------------------------------

def _custom_write_path(custom_name: str, target_root: str | None = None):
    """Where the custom-model agent writes this run's spec.

    A per-run overlay isolates generate from refine (they would otherwise
    share the cache `spec.yaml`). The cache file is the serving location,
    filled after a successful anonymous run lands.
    """
    if target_root:
        from pathlib import Path
        return overlay_models_dir(Path(target_root)) / f"custom:{custom_name}.yaml"
    return _custom_path(custom_name) / "spec.yaml"


def _custom_forward_source(custom_name: str, data: dict) -> str:
    """The custom model's registered `forward.py` — its only modeling source.
    Registration allows it to be empty; no agent may start on that."""
    text = data.get("forward_source") or ""
    if not text.strip():
        raise ForwardSourceUnavailable(
            f"Custom model 'custom:{custom_name}' has no modeling source "
            f"(its forward.py is empty or missing). Register it again with its "
            f"modeling files; retrying will not help.", retryable=False)
    return text


def build_custom_prompt(custom_name: str, *, target_root: str | None = None) -> str:
    """Prompt for generating a spec for a locally-registered custom model.

    Reuses the system context (components, ops, schema, examples) but swaps
    HF-backed ground truth for the user's local config.json + forward source
    + optional safetensors index from the custom-model cache dir.

    The prompt instructs the agent to Write the YAML to the run overlay
    when `target_root` is set, else the model's cache `spec.yaml`.
    """
    from backend.custom_model import load as _load_custom

    data = _load_custom(custom_name)
    config = data["config"]
    forward_src = _custom_forward_source(custom_name, data)
    target_spec = _custom_write_path(custom_name, target_root)

    sections = []
    sections.append(
        "## Ground Truth 1: Forward Logic (from user-provided modeling source)\n\n"
        "**THIS IS THE GOLD STANDARD FOR BLOCK WIRING.** Block graph node "
        "`in:` connections, norm placement, and residual adds MUST follow "
        "the compute order in this source.\n\n"
        f"```python\n{forward_src}\n```"
    )
    sections.append(build_config_section(config, len(sections) + 1))
    if data.get("header"):
        n = len(sections) + 1
        sections.append(
            f"## Ground Truth {n}: Weight Tensor Names (from local safetensors scan)\n\n"
            "Each line is one group of tensors, the index set at each numeric "
            "segment written in braces. Every tensor name your spec stores, "
            "with `X` at each position its declarations index, MUST name "
            "tensors of these groups; repeated or heterogeneous leaves may "
            "share one name. Follow the naming-convention rules.\n\n"
            f"```\n{format_header(data['header'])}\n```"
        )

    ground_truth = "\n\n".join(s for s in sections if s)
    system = build_system_prompt()

    custom_mt = config.get("model_type") or custom_name
    task_body = (
        f"# Task: generate spec YAML for custom model `custom:{custom_name}`\n\n"
        f"Config `model_type` is `{custom_mt}`. Your spec's `match.model_type` "
        f"MUST include this value so HuggingArch routes `custom:{custom_name}` "
        f"to your spec. The target file stem supplies the spec identity; do not "
        f"write a top-level `name:` field.\n\n"
        f"This custom-model flow persists only `{target_spec}`. Keep any new "
        f"component inline under that file's top-level `blocks:`; do not create "
        f"a draft file. Declare each new attention block using `cache.family` + "
        f"`cache.entries` (a sliding-window cache is a `sliding_window_kv` "
        f"over the cached stream). After "
        f"authoring, map the spec against the forward source above in both "
        f"directions (every forward step → a node, every node → a forward "
        f"step) by the standard below, and fix any discrepancy.\n\n"
        f"{forward_alignment_doc()}\n\n"
        f"**Write the final YAML to `{target_spec}` using your Write tool.** "
        f"Do not create any other files. After the Write succeeds, stop — "
        f"no summary text needed."
    )
    return assemble_task_prompt(system, ground_truth, task_body)


def prepare_custom_generate_prompt(
    custom_name: str, *, target_root: str | None = None,
) -> dict:
    """Prompt + prefetch logs for a custom-model spec generation. Mirrors
    `prepare_generate_prompt` so `/generate-spec` can stream custom-model
    runs through the same stream_agent path as HF models."""
    try:
        prompt = build_custom_prompt(custom_name, target_root=target_root)
    except FileNotFoundError as e:
        raise RuntimeError(str(e)) from e
    return {
        "prompt": prompt,
        "prefetch_logs": [f"Using cached custom model: custom:{custom_name}"],
        "target_root": target_root,
    }


def build_custom_refine_prompt(
    custom_name: str,
    spec_storage_bytes: int,
    hf_storage_bytes: int,
    storage_diff: int,
    *,
    target_root: str | None = None,
) -> str:
    """Prompt for refining an existing custom-model spec.

    Mirrors `build_custom_prompt` for ground truth (local config + forward
    source + safetensors index, no HF Hub fetch) but the task body is
    refinement-shaped: a spec already lives at the cache path, and the
    agent's job is to close the storage-bytes diff.

    Why a separate path: the HF refine prompt goes through
    `_build_prompt_context` which calls `hf_hub_download` — that fails
    with `Repo id must use alphanumeric chars` on `custom:<name>`.
    Custom models' ground truth lives entirely in
    `~/.huggingarch/custom_models/<name>/`; refinement reads from there.
    """
    from backend.custom_model import load as _load_custom

    data = _load_custom(custom_name)
    config = data["config"]
    forward_src = _custom_forward_source(custom_name, data)
    spec_path = _custom_write_path(custom_name, target_root)

    sections = []
    sections.append(
        "## Ground Truth 1: Forward Logic (from user-provided modeling source)\n\n"
        f"```python\n{forward_src}\n```"
    )
    sections.append(build_config_section(config, len(sections) + 1))
    if data.get("header"):
        n = len(sections) + 1
        sections.append(
            f"## Ground Truth {n}: Weight Tensor Names (from local safetensors scan)\n\n"
            "Each line is one group of tensors, the index set at each numeric "
            "segment written in braces. Spec leaf node ids MUST map onto "
            "these tensor names (minus the `.weight` suffix), with `X` at "
            "each position their declarations index.\n\n"
            f"```\n{format_header(data['header'])}\n```"
        )

    ground_truth = "\n\n".join(s for s in sections if s)
    system = build_system_prompt()

    diff_pct = (storage_diff / hf_storage_bytes * 100) if hf_storage_bytes else 0
    task_body = (
        f"# Task: Fix storage-bytes discrepancy for custom model "
        f"`custom:{custom_name}`\n\n"
        f"Current state (storage bytes — physical disk size, "
        f"mixed-precision safe):\n"
        f"  Spec computed:    {spec_storage_bytes:,} bytes  "
        f"({spec_storage_bytes / 1e9:,.2f} GB)\n"
        f"  Checkpoint actual: {hf_storage_bytes:,} bytes  "
        f"({hf_storage_bytes / 1e9:,.2f} GB)\n"
        f"  Diff:              {storage_diff:+,} bytes  "
        f"({diff_pct:+.4f}%)\n\n"
        f"## Instructions\n\n"
        f"A spec already exists at `{spec_path}`. Read it, compare against "
        f"the ground-truth sections above, and close the storage-bytes "
        f"gap. Change ONE thing per iteration. Finish only when the architecture "
        f"validator returns top-level `valid: true`; matching storage bytes is "
        f"one enforced part of that result.\n\n"
        f"Common refine targets:\n"
        f"- Off-by-one layer counts (`n_layers`, `n_mtp_layers`).\n"
        f"- Wrong intermediate size (`d_ffn` vs `d_ffn_expert`).\n"
        f"- Missing auxiliary heads / norms / projections that show up "
        f"in the tensor names list but not in the spec.\n"
        f"- Quant scheme mismatch — verify `quant_context` matches the "
        f"checkpoint's actual on-disk dtype layout.\n\n"
        f"**Write the updated YAML back to `{spec_path}` using your "
        f"Write tool.** Do not create any other files."
    )
    return assemble_task_prompt(system, ground_truth, task_body)


def prepare_custom_refine_prompt(
    custom_name: str,
    spec_storage_bytes: int,
    hf_storage_bytes: int,
    storage_diff: int,
    *,
    target_root: str | None = None,
) -> dict:
    """Prompt + prefetch logs for custom-model spec refinement.
    Counterpart to `prepare_custom_generate_prompt` for the refine path."""
    try:
        prompt = build_custom_refine_prompt(
            custom_name, spec_storage_bytes, hf_storage_bytes, storage_diff,
            target_root=target_root,
        )
    except FileNotFoundError as e:
        raise RuntimeError(str(e)) from e
    return {
        "prompt": prompt,
        "prefetch_logs": [f"Refining cached custom model: custom:{custom_name}"],
        "target_root": target_root,
        "seeded_spec": True,
    }


def _describe_base_section(base_model_id: str) -> str:
    """The starting point a description names: the library spec that model
    is served from, and its config, verbatim."""
    import json
    from pathlib import Path
    from backend.arch.loader import get_default_loader
    from backend.hf.metadata import fetch_config

    spec = get_default_loader().spec_for_model_id(base_model_id)
    config = fetch_config(base_model_id)
    if spec is None or not spec.source_path or config is None:
        raise RuntimeError(
            f"{base_model_id} is not a model this server serves; describe "
            f"the model without it, or pick one from the library.")
    spec_text = Path(spec.source_path).read_text(encoding="utf-8")
    return (
        f"## Starting point: {base_model_id}\n\n"
        f"The description builds on this model. Its spec (which may name a "
        f"`base:` from `backend/arch/models/` that you can read) and its "
        f"config follow. Start from them and change what the description "
        f"asks for.\n\n```yaml\n{spec_text}\n```\n\n"
        f"```json\n{json.dumps(config, indent=2)}\n```"
    )


def build_describe_prompt(custom_name: str, description: str, *,
                          target_root: str,
                          base_model_id: str | None = None) -> str:
    """Prompt for writing a model from a description in words.

    There is no checkpoint and no modeling source: the description is the
    only ground truth. The agent writes the structure (the spec) and the
    numbers (a config) as a custom model, and the validator judges it with
    every check that reads the spec alone (`checkpoint: absent`)."""
    from pathlib import Path
    model_id = f"custom:{custom_name}"
    spec_path = _custom_write_path(custom_name, target_root)
    snapshot = Path(target_root) / "snapshots" / f"{model_id}.json"

    sections = [
        "## Ground Truth: the user's description\n\n"
        "This is the only source. There is no checkpoint and no modeling "
        "code: build exactly the model it describes.\n\n"
        f"> {description.strip()}"
    ]
    if base_model_id:
        sections.append(_describe_base_section(base_model_id))
    task_body = (
        f"# Task: write the model `{model_id}` from its description\n\n"
        f"Write two files, and no others:\n\n"
        f"1. `{snapshot}` — JSON of the form "
        f"`{{\"model_id\": \"{model_id}\", \"config\": {{...}}}}`. "
        f"`config` is the model's config.json: every number the spec reads "
        f"(`config.<field>`), `model_type`, and `torch_dtype` (the dtype its "
        f"weights are stored in). Use the field names transformers models "
        f"use (`hidden_size`, `num_attention_heads`, …).\n"
        f"2. `{spec_path}` — the spec. Compose it from library components "
        f"wherever one fits (read `backend/arch/components/library/` and "
        f"`backend/arch/models/` for bases you can name with `base:`); keep "
        f"anything new inline under the file's top-level `blocks:`, with no "
        f"draft files. Set `match: {{ model_id: [{model_id}] }}` and write no "
        f"top-level `name:`. Weight precision other than `torch_dtype` (FP8 "
        f"experts, …) goes in `quant_context`.\n\n"
        f"The validator has no checkpoint to compare against, so it judges "
        f"structure alone: every edge's shape, every op's input against its "
        f"weight, caches, streams. "
        f"{shell_rules(model_id)} Iterate until it reports PASS.\n\n"
        f"When the description leaves a number open (vocab size, norm eps, "
        f"head count, dtype, …), choose the value a model of this kind most "
        f"commonly has. Finish with a short message in the user's language: "
        f"what you built and from which components, then a list headed "
        f"`Assumed:` with every value you chose that the description did not "
        f"give, one `name: value` per line."
    )
    return assemble_task_prompt(
        build_system_prompt(), "\n\n".join(sections), task_body)


def prepare_describe_prompt(custom_name: str, description: str, *,
                            target_root: str,
                            base_model_id: str | None = None) -> dict:
    """Prompt + prefetch logs for writing a model from a description."""
    return {
        "prompt": build_describe_prompt(
            custom_name, description, target_root=target_root,
            base_model_id=base_model_id),
        "prefetch_logs": [
            f"Writing custom:{custom_name} from its description"
            + (f", starting from {base_model_id}" if base_model_id else "")],
        "target_root": target_root,
    }


def prepare_generate_prompt(model_id: str, *, target_root: str | None = None) -> dict:
    """Build prompt plus user-visible prefetch logs for generation.

    `target_root` (a per-run overlay root) redirects the agent's writes there
    and is echoed into the returned dict so the post-success harvest reads
    artifacts from the overlay, not the shared tree.
    """
    system, ground_truth, meta = _build_prompt_context(model_id)
    task_body = build_generate_task_body(
        model_id, meta, target_root=target_root,
        source_files=_stage_source(target_root, model_id, meta))
    return {
        "prompt": assemble_task_prompt(system, ground_truth, task_body),
        "prefetch_logs": build_prefetch_logs(meta),
        "snapshot_data": meta.get("_snapshot_data"),
        "target_root": target_root,
    }


def prepare_refine_prompt(
    model_id: str,
    spec_storage_bytes: int,
    hf_storage_bytes: int,
    storage_diff: int,
    *,
    target_root: str | None = None,
    seed_bundle=None,
) -> dict:
    """Build prompt plus user-visible prefetch logs for refinement.
    Diff is in bytes — see `build_refine_task_body`.

    `target_root` (per-run overlay) is echoed back so the post-success harvest
    reads the refined artifacts from the overlay, mirroring
    `prepare_generate_prompt`.

    `custom:<name>` ids route to the custom-cache-backed variant; the
    default path calls `_build_prompt_context` which fetches HF metadata
    and would reject `custom:` as an invalid repo id.
    """
    if is_custom_id(model_id):
        return prepare_custom_refine_prompt(
            strip_prefix(model_id),
            spec_storage_bytes, hf_storage_bytes, storage_diff,
            target_root=target_root,
        )
    system, ground_truth, meta = _build_prompt_context(model_id)
    task_body = build_refine_task_body(
        model_id, spec_storage_bytes, hf_storage_bytes, storage_diff, meta,
        target_root=target_root, source_files=_stage_source(target_root, model_id, meta),
    )
    return {
        "prompt": assemble_task_prompt(system, ground_truth, task_body),
        "prefetch_logs": build_prefetch_logs(meta),
        "snapshot_data": meta.get("_snapshot_data"),
        "target_root": target_root,
        # Formal proposals live outside the serving view. Carrying the seed
        # bundle through publish lets a no-edit refine round-trip them.
        "seed_bundle": seed_bundle,
        # This run seeded the current spec into the overlay: a no-edit refine may
        # be judged on it (worker._verify_spec_output). A new generate never
        # sets this; a continued generation does (resume.prepare_resume).
        "seeded_spec": True,
    }
