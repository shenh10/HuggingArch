# Text to model

## What and why

The Playground gets a text box. A user describes a model in words, for
example "a 48-layer MoE, MLA attention with kv_lora 512, 128 routed experts
top-8, every 4th layer full attention and the rest sliding window 4096, FP8
weights". An agent writes that model as a **custom model**: a `config.json`
holding the numbers plus a `spec.yaml` holding the structure. The model then
opens in the Playground like any other. The user keeps refining it, either by
sending another chat turn to the same agent session or by editing the files
directly, and each edit is validated and re-analysed.

There may be no base model. When the description does name one ("Qwen3-8B but
with MLA"), the agent starts by copying that model's library spec. Otherwise
it composes the model from the component library, writing inline `blocks:`
only where the library has nothing that fits.

The difference from today's spec generation is what the model is checked
against. A generated HF spec is held to its checkpoint by the bytes gate. A
model built from a description has no checkpoint, so the user's description
is the only ground truth. The result is therefore a **hypothetical model**: it
passes every structural check, but nothing proves that its weights exist. It
is labelled as such everywhere it appears.

## What already exists and is reused

| Need | Existing piece |
|---|---|
| An agent loop that writes files, runs the validator, gets re-prompted with the errors, and can resume | `spec_worker.stream_agent` + `_repair_prompt_fn` + `resume.prepare_resume` |
| A sandboxed per-user, per-run working directory | overlay leaf `~/.huggingarch/spec_runs/<user>/<kind>__<id>/` |
| BYOK credentials, hourly quota, a per-run lock | `service/routes/spec.py::_preflight`, `preflight.claim` |
| Streaming progress to the browser | job events + `/api/spec/status` polling (`specGen.ts`) |
| Per-user storage of a spec and its config | Supabase `spec_generations` (`spec_yaml`, `snapshot_json`), `serving_generation`, `materialize_user_spec` |
| A model with no HF repo and no checkpoint header | `custom:<name>`: `fetch_config` reads the stored config, `fetch_metadata` returns `{}` |
| Viewing and editing the result | Playground `ModelEditor` (skeleton → compose-model) |

No new table, no new model format, and no new walker are needed.

## What has to change

### 1. A model stored without a checkpoint (backend/arch/validate, backend/hf)

Whether a model has a checkpoint is declared once, by
`hf.metadata.stored_without_checkpoint`: only a custom model can lack one —
one registered on this host without weights (its `meta.json` names no
`source_hf_repo` and it keeps no header), or one a user keeps (written from a
description; a user stores no weights). It is never inferred from a fetch
that came back empty: that is a failed fetch, and a model registered from a
weights repo whose scan failed still raises `CheckpointUnavailable`.

`validate()` reads that fact. With no checkpoint it skips only the checks
that compare against the file (coverage, bytes, names, packing against the
header, tied copies); **shape inference stays a hard gate**, and the result
says `checkpoint: absent` and lists every check it ran and skipped
(`checks`), which the UI shows as it is.

A user's overlay is agent-writable, so nothing in it is read as a
checkpoint: `fetch_metadata` answers `{}` for a custom id this host does not
register. A `custom:<name>` registered on this host is the host's model (its
dir wins for config and checkpoint), and the describe route refuses that name.

The walk also refuses an op whose input is not the width its own weight
declares (`OpRef.weight_operand_violation`, checked in the evaluator beside
the op's input contract): the checkpoint's bytes were the only thing that
used to see it.

### 2. A "describe" task prompt (spec_worker/_prompts)

`prepare_describe_prompt(name, description, base_model_id | None)`:

- Ground truth is the description text itself. When a base model is named,
  its library spec and config are staged into the leaf as the starting point,
  in the same way `refine` seeds its leaf.
- The prompt supplies the component library catalogue and `dims.yaml`, the
  same references the existing prompts copy in.
- The agent writes `models/custom:<name>.yaml` and `config.json`. Drafts
  follow the custom prompt's rule (inline `blocks:`, no draft files).
- Every number the description does not give (vocab size, norm eps,
  `torch_dtype`, …) gets a stated default, and the agent lists the defaults it
  chose in its final message. The user is then told what was assumed and does
  not have to discover it in the YAML.

### 3. A service route that runs it per user

`POST /api/spec/describe {name, description, base_model_id?}` reuses
`_preflight` (sign-in, BYOK, quota) and `claim`. The run key is
`(user, custom:<name>, describe)`. On success the bundle (spec + config) is
recorded in `spec_generations` under model id `custom:<name>`, with the
config in `snapshot_json` and no header. The custom name only has to be unique
per user. Nothing is written to the host-global `custom_models/` directory
from service mode.

Follow-up turns reuse the same leaf: `POST /api/spec/describe {name, message}`
calls `prepare_resume`, which re-enters the agent session with the current
files and verdict. Manual edits (below) change the leaf files first, so the
next agent turn sees them.

### 4. The Playground loads user-owned models

`model-skeleton` and `compose-model` are anonymous core routes today, so they
cannot see a user's specs. They get an optional signed-in user and resolve a
user-owned id the way `/api/spec/analyze` already does:
`serving_generation` → `materialize_user_spec` → `overlay_dir`. This uses the
same resolver, not a copy of it. Results for user-owned models are kept out of
the shared result cache, because the cache key does not identify the user.

### 5. Frontend: where it sits in the Playground

The Playground keeps its two tabs, and the precision presets and GPU picker
above them apply to described models unchanged.

- **Model editor.** "Describe in words" is a third source in the existing
  base-model card, next to "From the library" and "My models". A described
  model then *is* the loaded base: the layer-group table (slot swaps, group
  dims), Summary, KV, latency sweep and per-op table all work on it with no
  change. The agent is a collapsible drawer beside the editor. Swaps and dim
  edits made in the table are sent along with the next agent message, so
  the agent can write them into the spec; saving them directly as a new
  version needs the composition written back to YAML (Phase 2b) and is a
  later step.
- **Module comparison.** A described model's component calls appear in the
  bindings list next to shipped models, labelled "hypothetical · mine", so
  e.g. its MLA can be put in a column beside DeepSeek-V3's.

Drawer contents:

- A text box with a send button, and below it the agent's streamed log (the
  `AgentJob` / `specGen.ts` polling already used by the Inspector).
- After a successful turn: the defaults the agent chose, the validator
  verdict, and the model loaded into `ModelEditor`. The model carries a
  "hypothetical — not checked against a checkpoint" badge.
- A follow-up box that starts the next turn on the same session.
- A "Files" tab (the existing `SpecFiles` view, made editable) for direct
  edits to `spec.yaml` / `config.json`. Saving runs the checkpoint-free
  validation and re-analyses without an agent turn. A save that fails
  validation is kept as a draft and the errors are shown; the user can fix the
  files by hand or ask the agent to.


## Facts and who owns them

| Fact | Owner | Readers |
|---|---|---|
| Model structure | `spec.yaml` (agent or user) | loader → walk |
| Dim values, `torch_dtype` | `config.json` (agent or user) | `fetch_config` → walk |
| Weight precision | `quant_context` in the spec when the description asks for it, otherwise `torch_dtype` | `Analysis.tensors()` |
| "Hypothetical" | validation profile recorded with the run (no header) | badge, gallery/cache exclusion |
| Every cost number | the one walk | every view |

The LLM never produces a number that the views display directly. It writes
declarations, and the walk computes all costs from them.

## Out of scope for v1

- Anonymous use (it would spend the host's key).
- Promoting a hypothetical model into the public library.
- Writing `ModelEditor` swaps back into the spec.
