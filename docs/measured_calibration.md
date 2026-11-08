# Measured Calibration

The [inference cost model](inference_cost_model.md) prices every op at its
speed-of-light roofline: the larger of its compute time at the GPU's peak
FLOP rate and its memory time at the GPU's peak bandwidth. That is an upper
bound on performance. A real kernel reaches only a fraction of peak, and
which kernel is fastest depends on the shape, the dtype and the library.

Measured calibration closes that gap. A harness (`kernel_bench`) times real
kernels from many libraries at the shapes a serving run actually launches,
the rows are stored in a column store (the Kernel Bench lake) owned by one
service process, and an estimate looks up the rows for the ops it prices and
reports a measured time beside the roofline. This document describes that
system end to end, as built. The harness's command-line usage is in
[`kernel_bench/README.md`](../kernel_bench/README.md); its provider
architecture, timing method and failure records are in
[`kernel_bench/ARCHITECTURE.md`](../kernel_bench/ARCHITECTURE.md); the
frozen CSV schema of a row is in
[`backend/inference/BENCHMARKS_CSV.md`](../backend/inference/BENCHMARKS_CSV.md).

---

## 1. Overview

### 1.1 The measured envelope

The achievable cost of an op is not any single framework's dispatch choice:
it is the fastest correct kernel across the ecosystem. The reference for an
op is therefore the **measured envelope**: every library that implements the
op registers a provider, the harness measures every applicable provider at a
shape, and the consumer picks the fastest row. No library is privileged; a
new kernel adds a row to the envelope. An estimate may instead ask for the
exact kernel a serving framework dispatches to (§7.4).

### 1.2 End-to-end flow

```
 estimate walk ──► point_keys ──► PhysicalKey per op ─────────────┐
       │                                                          │
 kernel_bench plan (same point_keys) ──► jobs carrying the key    │
       │                                                          │
 kernel_bench run (GPU box) ──► results CSV ──► hbench upload     │
                                                    │             │
 browser ──► HA API ──/api/bench/*── proxy ──► Kernel Bench service (127.0.0.1:8100)
                                                    │      owns the lake
 estimate workers ──POST /internal/bench/lookup─────┘      ▲
       ▲                                                   │
       └──────────── candidate rows per key ───────────────┘
```

1. The estimate's structural walk produces per-op rows; one function,
   `calibration_overlay.point_keys`, states each row's measurement key (§2.3).
2. The harness plans from the same function, so every job carries the key
   its row will be stored under; a run on a GPU box measures the jobs and
   writes rows (§3).
3. A contributor uploads the rows; they land as a private batch in the lake
   (§4). An admin may promote reviewed batches to the shared corpus (§6).
4. An estimate resolves its measurement scope, asks the service for the
   candidate rows of its keys at one pinned snapshot, and selects a measured
   time per op (§7).

### 1.3 Terms

| Term | Meaning |
|---|---|
| row | One record of one kernel attempt at one shape on one GPU: a measurement, or a recorded failure (§2.1). |
| physical key | The five fields an estimate looks a row up by: signature, projected shape, dtype environment, GPU, phase (§2.2). |
| envelope | All rows sharing a physical key; its winner is the fastest. |
| batch | One upload: the unit of review and deletion (§4.1). |
| run | A grouping of batches by `run_id`; a run may hold several batches. |
| shared / private | A row is shared when its batch was promoted (§6.1); otherwise only its owner's estimates read it. |
| measurement scope | What an estimate's measured numbers are a function of: owner, lake, snapshot, annotation rules (§7.1). |

---

## 2. A measured row

### 2.1 Columns

A row follows the frozen CSV schema (`backend/inference/benchmarks.py`,
documented in [`BENCHMARKS_CSV.md`](../backend/inference/BENCHMARKS_CSV.md);
current `schema_version` 16). Its 43 columns form three groups:

- **Identity** (12 columns): `model`, `spec_hash`, `phase`, `parallel`,
  `framework`, `framework_version`, `gpu`, `op`, `node_spec_id`,
  `shape_json`, `dtype_env_json`, `kernel_impl`. Two rows with the same
  identity are the same measurement.
- **Measurement** (9 columns): latency statistics (`latency_us`, `p50_us`,
  `p95_us`, `samples`), calls per step, the achieved compute and bandwidth
  rates, and the GPU's peak compute and bandwidth at the time of the test.
- **Provenance** (22 columns): schema version, contributor and date, GPU
  architecture (`sm`), the numeric cross-check verdict and its evidence,
  the profiled kernel symbol that actually ran, `run_id`, `box`, driver
  versions, the outcome, the clock that timed the row (`timing_method`),
  and other columns described in the schema document.

The schema is one contract: the harness imports it and never forks it, and
the service validates every upload with the same validator
(`benchmarks.validate_text`).

**Outcome.** A row records what happened when a cell was attempted
(`benchmarks.OUTCOMES`): `measured`, `oom`, `mismatch`, `error`, `crash` or
`timeout`. Only `measured` rows carry a latency; the others record that the
attempt was made, so a re-planned sweep does not retry it at the same
library version. The table of who writes which outcome is in
[`ARCHITECTURE.md`](../kernel_bench/ARCHITECTURE.md#what-a-sweep-records-when-it-does-not-measure).

**Cross-check.** `cross_check` takes a value from
`benchmarks.CROSS_CHECK_VOCABULARY`: `passed`, `no_reference`,
`not_comparable`, `disabled`, or `unknown` (rows written before the column
existed). A kernel that disagrees with the reference is never written as a
measurement: it is recorded with `outcome = mismatch` and no latency. So
`cross_check = passed` is the correctness claim a row makes; a measured row
alone is not.

### 2.2 The physical key

An estimate does not look a row up by its 12-column identity. It looks it up
by its **physical key** (`op_calibration.PhysicalKey`), which keeps only what
determines a kernel's runtime:

| Field | Meaning |
|---|---|
| `signature` | The library-agnostic op signature (not the fused op name). |
| `shape_json` | The shape projected to the signature's identifying axes (`physical_shape` in the lake and on the wire). Model-only dims such as `d_model` or `n_layers` are dropped. |
| `dtype_env_json` | The leaf's own dtype environment (activation, weight, KV, scale metadata, output slots it states). |
| `gpu` | The GPU SKU (a row of `backend/inference/gpu_specs.csv`). |
| `phase` | `prefill`, `decode` or `unified`. |

The physical key deliberately drops provenance:

- `framework`, `framework_version`, `kernel_impl` name which implementation
  produced a number, so different libraries compete for one physical op;
- `model`, `spec_hash`, `node_spec_id` name where a shape came from, so a
  kernel measured once serves every model that launches the same shape;
- `parallel` names the deployment that cut a per-rank shape; one GPU runs a
  kernel at a shape the same way whichever deployment produced it. The row
  still records `parallel` as provenance.

Deployment choices that change the work a kernel does are axes of the shape,
not provenance: the paged-KV page size (`kv_block_size`) is an axis of every
signature that reads a paged cache, and the Mamba-2 scan chunk is an axis of
the scan signature.

### 2.3 One key, two sides

A measurement is useful only if the key it is stored under is the key an
estimate looks it up by. The writer (the harness plan) and the reader (the
estimate) are different programs, so the key must have exactly one
derivation, and both sides must read it rather than derive it.

**The rule.** `calibration_overlay.point_keys` is the only derivation. It
reads a walked per-op row's own facts and states that row's identity, key
and launch as a `PointRow`. Its readers derive nothing:

| Reader | Use of the key |
|---|---|
| The estimate's overlay (`CalibrationOverlay.apply`) | Looks the measurement up. |
| The sweeps (`capacity`, per swept point) | The same, per point. |
| The harness plan (`kernel_bench/plan.py`) | Stores under it and carries it on each job. |
| The contributor template (`cli_bench.emit_suggest_shapes`) | Writes it into the CSV a contributor fills. |

**The job carries the key.** A planned job holds `Invocation.physical_key`
(signature, projected shape, dtype environment, phase). The only field a
plan cannot know is the GPU, which the run stamps
(`driver.physical_key_for`). A job without a carried key is refused and must
be re-planned; the driver never rebuilds a key from the job's dims.

**Declined rows.** When a leaf's own dtype environment cannot be stated in
corpus tokens, `point_keys` marks the row `declined`. Neither side touches
it: the estimate does not look it up and the plan does not measure it,
because a row stored under a key nobody asks for is wasted GPU time.

**The gate compares values, not calls.** A static check that the plan
*calls* the key function cannot fail when the plan ignores the answer.
`tests/kernel_bench/test_plan_keys_are_the_estimates.py` therefore asserts
equality: for a BF16, an FP8 and an FP4 model, every job the plan exports
carries exactly the key bytes the estimate states for the same node. A BF16
model alone cannot exercise this rule, because every derivation agrees on
BF16; the divergences live in the quantized slots.

---

## 3. The harness

### 3.1 Signatures, providers and cross-checks

- **Op signature** (`backend/inference/op_signatures`, re-exported by
  `kernel_bench/signatures.py`) is the math contract of an op class: what
  must be computed. `binds_ops` maps engine ops onto it; providers register
  against it.
- **Provider** (`kernel_bench/providers/`) is one library's kernel for a
  signature. It declares what it serves (`serves`: dtypes, phase,
  divisibility and size limits), the minimum GPU architecture (`min_sm`), the
  kernel parameters that change what is measured (`impl_params`, which form
  part of the implementation id), the module that gates its availability
  (`import_name`) and an input adapter (`build`). A kernel known to be wrong
  or not yet measurable declares `disabled_reason` and is registered but
  never run. Registration is metadata only: `providers.discover()` imports no
  kernel library, torch included.
- **Numeric cross-check** (`kernel_bench/refs.py`, `kernel_bench/driver.py`)
  runs one canonical-input, torch-eager reference per signature, shared by
  all its providers. Each provider adapts the canonical inputs to its own
  layout; the driver compares the provider's result, mapped back to the
  canonical layout, against the reference within a per-dtype band
  (`driver._XCHECK_BANDS`: fp32, fp16, bf16, fp8_e4m3, fp8_e5m2, each with a
  typical and a worst-element bound). The band is that of the least precise
  operand the kernel reads (activation, weight or KV slot). A dtype with no
  band stops the cross-check; bf16 is not a fallback.

The three provider records (library, kernel, declaration), what guarantees
a declaration is true, and the vocabularies they share are specified in
[`ARCHITECTURE.md`](../kernel_bench/ARCHITECTURE.md).

### 3.2 Library environments

Kernel libraries are optional third-party extras, never core dependencies,
and their versions conflict. The unit of deployment is therefore **one
isolated environment per library** (`kernel_bench/envs/*.txt`). The thin
harness is installed into each; a run there measures only that library's
providers (`run --library <lib>`) and stamps every row with the versions
actually installed. Two provisioning modes exist:

- **pip-resolvable** (torch, FlashAttention 2, FlashInfer, vLLM, sglang,
  Humming, Marlin, FLA): each library is one `Library` record
  (`kernel_bench/providers/_libraries.py`) stating how to install it; a
  kernel keeps only its `import_name`. vLLM and sglang default to the release
  the framework bindings were audited at. The `envs/*.txt` files are
  generated from these records (`python -m kernel_bench envs`; `--check`
  fails when they are stale) and are never edited by hand.
- **source-build** (FlashAttention 3 and 4, FlashMLA, DeepGEMM, DeepInfer,
  CUTLASS MLA, FlashKDA, mamba-ssm): GPU-architecture-specific builds that
  are not reproducibly pip-installable. They are provisioned out of band and
  targeted with `--env <lib>=<python>` or `KERNEL_BENCH_ENV_<LIB>`.

A **box profile** (`kernel_bench/envs/boxes/*.toml`, `--box`) declares a
machine's GPU, its environments and the versions its constraints pin; plans
written for a box judge each job against that box before any GPU work, and
the `box` column records which profile a row came from.

### 3.3 Timing and run records

`timing.time_callable` is the only clock. On a GPU each sample is one launch
bracketed by CUDA events, with an L2 sweep before the window and a second
sweep inside it (so the launch's own write-back is timed), a GPU-side hold
so the events bracket queued work rather than host launch overhead, and the
sweep's own cost subtracted. The row records the p50 and the method
(`timing_method`; NULL on rows timed before the current method). The full
derivation is in
[`ARCHITECTURE.md`, "How a launch is timed"](../kernel_bench/ARCHITECTURE.md#how-a-launch-is-timed).

Every attempt leaves a row (§2.1, *Outcome*). Before a launch path's cells
run, its smallest cell runs as a canary; a canary that fails disables the
path for that run. A change to a provider's build or input conversion purges
that implementation's old rows in the same change, through the same deletion
channel the lake exposes (§4.3). The rules are in
[`ARCHITECTURE.md`](../kernel_bench/ARCHITECTURE.md#what-a-sweep-records-when-it-does-not-measure).

### 3.4 Versions and coverage

Version is an identity axis of a row (`framework` is the provider library,
`framework_version` its version, `kernel_impl` the complete implementation
id), because a kernel's performance changes across releases. Rows only
accumulate; a new release adds rows beside the old ones.

A cell counts as tried at an implementation version when any row exists for
its coverage key `(framework, framework_version, kernel_impl)`, whatever its
outcome. The harness asks the service this question through the same lookup
an estimate uses (`backend/measure/lookup.covered`): over the shared rows,
plus the contributing account's own rows when the harness names it in
`HUGGINGARCH_BENCH_OWNER`. A new library version is tried again.

### 3.5 Supported signatures and kernels

The registry holds 361 providers over 124 signatures. Every signature has a
torch-eager provider except `fused_mla_norm_rope_attn_rope_cast`. The table
groups the 89 signatures that also have providers from other libraries, with
the number of kernels each library registers for the group. It is a snapshot
of `kernel_bench.providers`; the authoritative list, with every kernel's
name and declared capabilities, is `GET /api/bench/providers` (§5.3).

| Family | Signatures | Libraries beyond torch (kernels) |
|---|---|---|
| Attention | `gqa_attention`, `mla_attention`, `mla_sparse_attention`, `mla_sparse_sink_attention`, `gqa_sparse_gated_attention`, `fused_mla_norm_rope_attn_rope_cast` | cutlass_mla (1), deepinfer (1, 1 disabled), fa2 (2), fa3 (2), fa4 (1), flashinfer (4), flashmla (3), sglang (3), vllm (7) |
| GEMM and fused MoE | `gemm`, `grouped_gemm`, `per_head_gemm`, `fused_moe` | deepgemm (6, 2 disabled), deepinfer (1, 1 disabled), flashinfer (1), humming (1), marlin (3, 1 disabled), vllm (9) |
| MoE routing | `moe_gate`, `moe_mega_gate_sqrtsoftplus`, `moe_topk_hash_sqrtsoftplus`, `moe_topk_sigmoid`, `moe_topk_softmax`, `moe_topk_softmax_bias`, `moe_topk_sqrtsoftplus` | deepinfer (1), sglang (5), vllm (6) |
| Normalization | `rmsnorm`, `fused_add_rmsnorm`, `rmsnorm_gated`, `mamba2_rmsnorm_gated`, `gemma_qkv_rmsnorm`, `gemma_rmsnorm_residual_scalar`, `gemma_dual_rmsnorm_residual_scalar`, `fused_q_kv_rmsnorm`, `fused_q_kv_rmsnorm_quant`, `fused_eh_norm` | flashinfer (2), sglang (12), vllm (12) |
| Gated activations | `silu_and_mul`, `gelu_and_mul`, `situ_and_mul`, `silu_and_mul_with_clamp`, `sigmoid_gate_mul`, `softcap_logits`, `ple_gate`, `fused_engram_gate` | deepinfer (1), flashinfer (1), sglang (9), vllm (8) |
| RoPE and fused norm + RoPE | `rope`, `rope_2d`, `rope_partial`, `rope_partial_q`, `xdrope`, `fused_qk_norm_rope`, `fused_qk_norm_partial_rope`, `fused_qk_rmsnorm_rope_gate`, `fused_qk_mrope`, `fused_qnorm_rope`, `fused_qnorm_rope_kv_insert`, `fused_qnorm_rope_kv_insert_no_q_norm`, `fused_inv_rope`, `fused_norm_rope` | deepinfer (1), flashinfer (1), sglang (10), vllm (14) |
| Linear attention and SSM | `gated_delta_rule`, `fused_recurrent_gated_delta_rule`, `fused_gdn_gating`, `fused_gdn_post_conv_prep`, `kda`, `fused_kda_decode`, `fused_kda_gate`, `fused_recurrent_kda`, `gla_simple_recurrence`, `mamba2_scan`, `causal_conv1d` | fla (6), flash_kda (1), flashinfer (1), mamba_ssm (1), sglang (14), vllm (14, 1 disabled) |
| Hyper-connections | `hc_combine_norm`, `hc_gate_mix`, `hc_silu`, `ihc_post`, `mhc_fused_post_pre`, `mhc_mega_post_pre`, `mhc_post`, `mhc_pre_big_fuse`, `mhc_pre_gemm`, `fused_hc_head` | deepinfer (2), sglang (8), vllm (11) |
| Sparse-attention indexers, compressed caches | `indexer_logits`, `indexer_block_score`, `fused_indexer_k_norm_rope_store`, `fused_indexer_q_rope_first_quant`, `fused_indexer_q_rope_quant`, `fused_indexer_q_rope_quant_split`, `fused_dsa_qkv_norm_rope`, `fused_dsa_qkv_norm_rope_indexer`, `fused_compress_norm_rope_store`, `fused_compress_pool_state`, `fused_compress_rope_store`, `fused_overlap_compress_norm_rope_store`, `fused_save_compress_norm`, `fused_kpool_compress_store`, `qsa_index_k_compress_store`, `qsa_index_q_norm_rope` | deepgemm (1, 1 disabled), deepinfer (3), sglang (7), vllm (17) |
| Activation quantization, transforms | `quantise_activation`, `fused_fwht128_quant`, `hadamard_transform` | sglang (5, 1 disabled), vllm (6, 1 disabled) |

The remaining 35 signatures are measured by torch only: `add`,
`allreduce_fused_rmsnorm`, `attn_res_combine`, `conv2d`, `dsa_topk_mask`,
`embedding`, `gather`, `gelu`, `gelu_tanh`, `hash_route`, `hc_mix`,
`hc_split_sinkhorn`, `layernorm`, `moe_combine`, `moe_permute`, `mul`,
`neg_exp`, `normalize_sum_one`, `position_table_add`, `relu_squared`,
`rmsnorm_unscaled`, `rope_emb`, `rope_emb_inverse`, `scale_const`,
`sigmoid`, `signed_sqrt`, `silu`, `situ`, `softmax_act`, `softplus`,
`sqrt`, `step3p5_headwise_gate`, `sum_reduce`, `tanh`, `topk`.

Ten providers are registered disabled, each with its reason:
`deepinfer_flashmla_mega_attention` and `deepinfer_mega_moe` (they need
state the serving engine creates at load time), `deepgemm_fp8_fp4_mega_moe`
and `deepgemm_fp8_paged_mqa_logits` (no runnable body yet),
`deepgemm_fp8_fp4_m_grouped_contiguous`, `marlin_gptq_int4_w4a16`,
`vllm_causal_conv1d_update`, `vllm_scaled_fp4_quant` and
`sglang_flashinfer_fp4_quantize` (packed FP4 outputs the cross-check cannot
yet map back), and `torch_allreduce_fused_rmsnorm`: collectives need a
multi-rank launcher that the single-kernel harness does not have.

### 3.6 Adding a kernel or an op class

- A kernel is a new `Provider` in the relevant `kernel_bench/providers/*.py`
  (its `serves` declaration and `build`). A new library also needs a
  `Library` record and regenerated `envs/*.txt`.
- An op class is a new signature and a torch-eager reference in
  `kernel_bench/refs.py`.

---

## 4. Storage: the Kernel Bench lake

### 4.1 Layout and tables

Rows live in a **DuckLake** (the DuckDB lakehouse format): table metadata
(snapshots, file lists, small inlined writes) in a catalog, and data as
immutable Parquet files. In production both are local to the machine's
volume:

| Path | Content |
|---|---|
| `/app/.cache/bench_lake/catalog.ducklake` | The catalog (`BENCH_LAKE_CATALOG`). |
| `/app/.cache/bench_lake/data/` | The Parquet files (`BENCH_LAKE_DATA`). |
| `/app/.cache/bench_lake/tmp/` | DuckDB spill. |
| `/app/.cache/bench_lake/staging/` | Backup staging. |

`lake.py` also supports a Postgres catalog (`postgres:<dsn>`, schema
`bench_lake`) and an `s3://` data path for a multi-machine deployment; the
production deployment does not use them.

The lake stores four tables and nothing derived (`store.py`):

| Table | Content |
|---|---|
| `batches` | One row per upload: `batch_id`, `owner`, `upload_id`, `piece`, `run_id`, `label`, `created_at`, `status` (§6.1), `within_peak`, `nominated_at`, `decided_at`, `decided_by`. Unique on `(owner, upload_id, piece)`. |
| `inputs` | An input's identity only: `input_id` and the key `(op, phase, kernel_impl, shape_json)`. |
| `measurements` | Every CSV column (`benchmarks.ALL_COLUMNS`, typed by `store.MEASUREMENT_TYPES`) plus `id` (UUID), `batch_id`, `input_id`, `env`, `created_at`. Written sorted by `(batch_id, input_id)`. |
| `lake_meta` | The lake's own facts: `lake_id` (a UUID written when the lake is created), `lake_role` (`authority` or `development`), `import_state`, `supersedes_lake_id` on a replacement, `data_rules`, and the last data-rules migration's report. |

Ids are hashes of the identities they name, not counters
(`md5_number_upper` over the identity's fields joined by `0x1F`), so two
writers for different owners need no coordination. `env` is the hash of the
row's `dtype_env_json` and `gpu` (`store.env_key`); the deployment
(`parallel`) is not part of it. An **envelope** is the group
`(phys, env, phase)`, where `phys` is the hash of the signature and physical
shape (§4.2).

A measurement is never updated after it is written, with two exceptions: the
roofline columns may be restamped (§4.3), and a registered data-rules migration
may rekey an older lake (below). The migration owner is not an ad-hoc script:
`backend/service/bench/data_rules.py` follows the chain registered in
`data_rules_version.json`, whose final target is the live `rules_rev`, and a
test refuses a key/validator edit whose target has not been registered. A
brand-new lake is stamped with the current revision; an older lake is migrated
after the lease is claimed and before annotations or serving. The heavy legacy
dtype-env rewrite ships as `migrations/prod_rekey_plan.json` (the same plan
`kernel_bench.rekey_lake` made offline), then today's validator is run over
the rows, refused rows are deleted through the owner deletion path, and
effective-shared duplicate identities are reduced to their earliest row.
`data_rules` is written only after every registered step commits. A read-only
restored copy that is behind this revision does not serve: it cannot migrate
itself, and its rows therefore do not yet carry the guarantee the read path
relies on.

### 4.2 Annotations

An input's **annotations** are the grouping labels the uploader does not
state: `signature`, `physical_shape`, `workload_batch`, `workload_context`,
`catalog_kernel` and `phys`. They are pure functions of the input under the
current code (`annotations.row_annotations`, composing `physical_identity`,
`workload_coordinates` and the provider catalog's `catalog_kernel`) and are
**not stored**. At startup the service computes them for every input into an
in-memory table `annot` (`store.annotate_all`); an upload adds its new
inputs' annotations before it commits them, so a committed row always has
them. An input the rules refuse has no `annot` entry: its rows are visible
in row views but belong to no envelope.

A change to the annotation rules or catalogs takes effect at the next service
start. `rules_revision.rules_rev` names the data format's revision from the
semantic owners declared by `rules_revision.rule_inputs`:

| Input | Why it can change a row or its key |
|---|---|
| `annotations.row_annotations`, `registered_kernels`, provider `catalog_kernel` | input annotations and the implementation catalog |
| `benchmark_identity.current_shape`, `measured_row_migration`, `workload_coordinates` | retired/derived shape axes and workload coordinates |
| `op_calibration.physical_identity` | signature and physical-shape projection |
| `store.canonical_json` and the storage schema in that module | canonical identity JSON and row storage format |
| `benchmarks.validate_row`, `row_identity` | row validity and duplicate identity |
| dtype, activation-format, eval-format, and measured-row migration modules | vocabulary and migrations the validator/key derivation reads |
| loaded `all_signatures()` and `registered_kernels()` | signature and kernel catalogs as data |
| `data_rules.migrate` and `kernel_bench.rekey_lake.apply` | the migration algorithm itself |
| `shared_data_version()` | shared migration tables read by `current_shape` |

The revision hashes the source module of each declared owner, not that owner's
transitive import closure. Connection, budget, startup, HTTP, and lookup
execution modules are deliberately outside the boundary. It is part of every
measurement scope (§7.1). A key, validator, shared-data, or migration change
that can make an older lake invalid must extend the registered chain and
regenerate `data_rules_version.json`; `test_data_rules.py` turns an
unregistered edit red, and `test_service.py` proves lake/lookup execution
edits do not move the revision while validator/physical-identity edits do.

### 4.3 Writes

All writes go through `store.py`, which knows no users; the routes pass
explicit owners (§5.3).

| Operation | Effect |
|---|---|
| Upload | Validates the CSV (`benchmarks.validate_text`) and the speed-of-light gate (a row whose achieved compute or bandwidth exceeds the peak it records is refused, 422; `kernel_bench.emit.above_sol`). At most 200,000 rows; one `run_id` per file. Under the owner's lock, rows whose 12-column identity the owner already has are skipped as duplicates; new inputs are registered under the inputs lock; then one transaction writes the inputs, the batch and its rows. |
| Restamp | Rewrites only the roofline columns (`peak_*_at_test`, `achieved_*`) of the owner's rows in place, keeping their batch and review status. Used when the GPU table's peaks or the plan's per-launch numerators are corrected; the latency is never touched. The result must still pass the speed-of-light gate. |
| Nominate, promote, reject | Change `batches.status` (and `within_peak`, `decided_*`); no row changes (§6.2). |
| Delete | Deletes a run, an upload, or selected rows of the owner's; a batch left without rows goes with them. |

Lock order is owner → inputs → commit; the backup and maintainer locks are
never taken while a write lock is held. A write transaction that loses to a
concurrent commit is attempted at most three times, then answered 503. Every
write, maintenance run and backup first checks the volume's free space and
refuses below 500 MiB (`lake.MIN_FREE_BYTES`). DuckDB runs with 4 threads, a
memory limit of `BENCH_LAKE_MEMORY_LIMIT` (default 384 MB) of which the held
selection table may take `BENCH_LAKE_SELECTION_LIMIT` (default 32 MiB), a 32 MB target
file size, and at most two concurrent analytical reads.

### 4.4 Snapshots and content identity

Every committed write creates a DuckLake snapshot. Reads that must be exact
pin a snapshot (`AT (VERSION => n)`), so an answer always matches the
identity it is cached under (§7.5).

A snapshot id names a commit only within one lake's history: a lake that is
restored or copied keeps its id and history, and once two copies diverge,
one snapshot id names two different contents. `Lake.content_id(n)` therefore
hashes the snapshot number together with the catalog rows live at it — the
data and delete files (DuckLake names every file uniquely and never rewrites
one), the inlined rows and inlined deletions by value, and the tables and
columns. Equal content ids mean equal row and table content at that
snapshot; divergent copies differ by construction. The snapshot number is
included because merging adjacent files produces one file spanning several
snapshots whose catalog rows are identical across them. Content ids are
memoized per snapshot.

### 4.5 Access control on disk

Spec-generation agents run as another user in the same container and must
not read or change measurements. The lake's local directories are
root-only: `Lake.open` creates a missing directory with mode 0700 and
refuses one not owned by the process user or granting any group or other
permission. The container entrypoint creates `bench_lake` root-owned with
mode 0700 and chowns only the agent's own directories (§8.1).

### 4.6 The lease: one authoritative, writable lake

A Supabase table `public.bench_lake_lease` (one row: `lake_id`, `machine`,
`claimed_at`) records which lake is authoritative, through the functions
`bench_lake_claim` and `bench_lake_transfer`. At startup the service claims
it with its `lake_id` and machine id (`FLY_MACHINE_ID`, or
`BENCH_LAKE_MACHINE` in development):

- no lake registered → this lake becomes the authority (`claimed`);
- the same lake on the same machine → serve (`ok`);
- another lake, or this lake on another machine (a cloned volume) → refused;
  the lake does not serve. Only an operator moves the lease
  (`bench_lake_transfer`).

An unreachable lease fails closed and is retried. A brand-new, empty lake is
refused when the lease already names a lake: an empty volume is restored
from backup, never started afresh. `BENCH_LAKE_LEASE=off` disables the lease
for development and tests only — and it never turns a copy of an identified
lake into a second writable history:

- a lake created or claimed with the lease has `lake_role=authority`;
- a lake created without the lease has `lake_role=development` and its own
  UUID, and can never later claim the production lease;
- an existing unleased lake that does not say `development` (including a
  backup restored before `lake_role` existed) serves **reads only**; `writer`
  refuses, startup runs no data migration, maintenance and backups do not
  write, and `/health` reports `read_only: true`.

Thus `lake_id` names one lineage. Caches may key on it without adding a
second content identity: two copies at one `lake_id` may exist for disaster
recovery, but only the lease holder can write one.

#### Replacing the authoritative lake

Never upload a copied lake under its old `lake_id`. Build a replacement in a
separate directory, give it a new lineage, upload that directory, then move
the lease in one operator action:

```bash
# 1. From a pristine copy (never the live authority), make a new candidate.
PYTHONPATH=. .venv/bin/python tools/reidentify_authoritative_lake.py \
  --source-catalog /scratch1/newlake/catalog.ducklake \
  --source-data /scratch1/newlake/data \
  --destination /scratch1/lake-v3-candidate \
  --migrate

# 2. Apply the operator-only lease migration once (migration 0027).
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 \
  -f supabase/migrations/0027_bench_lake_transfer_lineage.sql

# 3. Publish the candidate as its own backup generation (credentials elided).
BENCH_LAKE_CATALOG=/scratch1/lake-v3-candidate/catalog.ducklake \
BENCH_LAKE_DATA=/scratch1/lake-v3-candidate/data \
BENCH_LAKE_BACKUP_DATA=s3://bench-lake-backup/lake-v3 \
BENCH_LAKE_BACKUP_S3_ENDPOINT=... BENCH_LAKE_BACKUP_S3_REGION=... \
BENCH_LAKE_BACKUP_S3_KEY=... BENCH_LAKE_BACKUP_S3_SECRET=... \
  .venv/bin/python scripts/bench_lake_backup_now.py
```

Restore that **new generation/prefix** into an empty production volume
(`BENCH_LAKE_BACKUP_DATA=s3://bench-lake-backup/lake-v3` and
`HUGGINGARCH_BENCH_RESTORE=on`). Stop the old writer first. Then, with the
old lease's `lake_id`/machine and the candidate's new `lake_id`/machine, use
the four-argument `bench_lake_transfer`: it changes both the machine and the
lake lineage only when the old holder matches:

```sql
SELECT public.bench_lake_transfer(
  '<old-machine>', '<new-machine>',
  '<old-lake-id>'::uuid, '<new-lake-id>'::uuid);
```

Start the new service and confirm `/health` reports the new `lake_id`,
`read_only: false`, and current data rules. The old volume and old backup
generation are rollback artifacts, not writable peers. The two-argument
`bench_lake_transfer(old_machine, new_machine)` remains for moving the same
lineage between machines.

### 4.7 Backup and restore

Backups are mandatory and leave the machine, into their own bucket with
their own access key (`BENCH_LAKE_BACKUP_DATA`, `s3://…` or a local
directory, and `BENCH_LAKE_BACKUP_S3_*`). Layout under the backup root
(`backup.py`):

| Key | Content |
|---|---|
| `data/<sha256>` | Every data and delete file a snapshot references, content-addressed, never overwritten. |
| `snapshots/v<v>/catalog.ducklake` | The catalog copy at snapshot `v`. |
| `snapshots/v<v>/manifest.json` | The catalog object and every data object (path, size, sha256), and row counts at `v`. |
| `current` | `{"snapshot": v, "manifest": key}`. |

One backup, under the lake's backup lock: copy the catalog at a snapshot
`v` in one transaction (`COPY FROM DATABASE`; no `CHECKPOINT`, which would
compact and rewrite files), upload the referenced files, then the catalog
object, then the manifest, then `current`. A backup runs after every
committed write (coalesced, off the request path), on every maintainer
cycle, and at startup before writes are accepted.

**Fail-closed.** Writes are refused (503) while the backup is unconfigured,
failing, or more than 5 minutes or 20 commits behind; DuckLake expiry,
cleanup and backup retention wait while it is behind. `BENCH_LAKE_BACKUP=off`
opts out explicitly in development. A write is acknowledged after the local
commit and backed up within seconds; losing the volume inside that window
loses those writes, while a process crash loses nothing.

**Retention** keeps every manifest of the last 7 days, the last manifest of
each day for 30 days, and the one `current` names; data objects no kept
manifest references are deleted.

**Restore.** With `HUGGINGARCH_BENCH_RESTORE=on` and no catalog on the
volume, the service fills the lake directory from the backup's `current`
snapshot before anything opens it (`serving.restore_from_env`): every object
is checked against its sha256 and the row counts against the manifest. A
partly filled directory or a backup without `current` is refused, never
replaced by an empty lake. With a lake already present the switch does
nothing. A lake built elsewhere is uploaded as a backup snapshot with
`scripts/bench_lake_backup_now.py` and then restored this way.

### 4.8 Maintenance

A paced maintainer (`maintain.py`, behind `HUGGINGARCH_BENCH_MAINTENANCE`)
runs in the service. Each cycle backs up the lake if it moved past
`current`; at most hourly it runs DuckLake housekeeping — merge adjacent
small files, expire snapshots older than 7 days, delete files only those
snapshots referenced and orphaned files older than 7 days — under the backup
lock, then the backup retention. While the backup is behind, only the merge
runs. The first cycle starts 300 s after the lake serves; the loop rests
three times each cycle's duration (at least 30 s) and backs off up to 600 s
on errors. Snapshots older than 7 days cannot be read by a pinned lookup.

---

## 5. The Kernel Bench service

### 5.1 Process and trust boundary

The **Kernel Bench service** (`python -m backend.service.bench.server`) is a
FastAPI app bound to `127.0.0.1` (port `HUGGINGARCH_BENCH_PORT`, default
8100). It is the only process that opens the lake; DuckDB locks the catalog
file against a second. It always runs as one uvicorn process with no
workers and no reload, whatever concurrency the HA API runs with. Nothing
under `backend/` outside the service reaches it except over HTTP
(`tests/meta/test_service_layout.py`).

The service authenticates no one. Its trust boundary is:

- **Shared secret.** Every request must carry the shared secret in
  `X-HA-Internal` (compared in constant time) and come from a loopback peer
  (`server.InternalOnly`); anything else is 403 before it reaches a route,
  `/health` included. The secret is `HUGGINGARCH_BENCH_SECRET`, or the
  content of the 0600 file `HUGGINGARCH_BENCH_SECRET_FILE` names; a service
  started without one refuses to start.
- **Identity headers.** The caller is taken only from `X-HA-User` (user id;
  empty for an anonymous caller) and `X-HA-Admin` (`1` for an admin) on a
  request that passed the secret check (`identity.py`).
- **The HA API's proxy** (`backend/service/bench_proxy.py`) authenticates
  the browser — a session (bearer JWT or access cookie), with a personal access
  token accepted only on the measurement pipeline's upload, nominate and promote
  routes — and forwards `/api/bench/*` with headers built fresh: the body's
  `Content-Type` and the three `X-HA-*` headers it alone sets. No browser
  header, cookie or `X-HA-*` header crosses. Bodies stream both ways.
  Timeouts: reads 30 s, writes 120 s, connect 2 s; a client disconnect
  cancels the upstream request. The service's own statuses and bodies pass
  through; the service unreachable, timing out or failing with a 5xx other
  than 503 becomes one 503 "Kernel Bench is unavailable" with `Retry-After`.
  `/internal/bench/*` is not routable through the proxy.

Every internal client (the estimator's lookup client, the proxy, the
entrypoint's probe) ignores environment HTTP proxies (`trust_env=False`), so
the secret never leaves loopback.

### 5.2 Startup gate

`serving.Gate` walks the startup once, in a background thread after the
server is up:

1. restore from backup when asked and the volume is empty (§4.7);
2. open the lake (`lake.get_lake`; root-only directories);
3. a brand-new lake is decided before anything is written: refused when the
   lease already names a lake, otherwise its tables are created with a new
   `lake_id`;
4. a lake left in the `importing` state of a retired one-time import is
   refused (restore it from backup);
5. claim the lease (§4.6);
6. bring the schema current (`store.ensure_schema`) and rekey stale `env`
   values;
7. compute every input's annotations in memory (§4.2);
8. start backups: back up what `current` lacks, then follow every write;
9. serve, and start the shared-rows revision (§7.1) and the maintainer.

Until step 9, and whenever a step refuses, every lake endpoint answers 503
naming the reason. Transient failures (lake or lease unreachable) are
retried on a request after 30 s. Writes are additionally refused while the
backup refuses them or the volume is short of space.

### 5.3 HTTP API

All routes are in `backend/service/bench/routes.py`. The `/api/bench/*`
routes are reached through the HA API's proxy; the `/internal/bench/*`
routes are called by the HA API and its compute workers directly.

| Method and path | Caller | Purpose |
|---|---|---|
| `POST /api/bench/upload` | signed in | Validate a CSV and store it as one private batch; an admin may publish it directly as shared (`make_public`; 403 for a non-admin). Returns `upload_id`, `row_count`, `duplicates`, `label`, `shared`. |
| `GET /api/bench/list` | signed in | The caller's runs and uploads, newest first, each with its batches and their status. |
| `GET /api/bench/summary` | signed in | One set's counts and facets (an admin may name another `owner`). |
| `GET /api/bench/rows` | signed in | One page of one set (`run_id` or `upload_id` required; at most 1,000 rows; offset or keyset paging). |
| `GET /api/bench/calibration` | signed in | The caller's own rows as their estimates read them: used, not measurements, rejected by the current validator and why; the snapshot and scope digest. |
| `GET /api/bench/overlap` | signed in | One set against the shared corpus per op shape: shapes already shared, new ones, where the set is faster. |
| `POST /api/bench/rows/delete` | signed in | Delete selected rows of the caller's (at most 5,000 ids). |
| `POST /api/bench/rows/restamp` | signed in | Correct the roofline columns of the caller's rows (at most 5,000). |
| `DELETE /api/bench/run/{run_id}` | signed in | Delete every batch of one run. |
| `DELETE /api/bench/{upload_id}` | signed in | Delete one upload. |
| `POST /api/bench/run/{run_id}/contribute` | signed in | Nominate a run's private batches (`within_peak` optional). |
| `POST /api/bench/{upload_id}/contribute` | signed in | Nominate one upload's private batch. |
| `GET /api/bench/nominations` | admin | Nominated batches, newest first, with their submitter. |
| `POST /api/bench/promote` | admin | Nominated → shared, for the named sets. |
| `POST /api/bench/reject` | admin | Nominated → rejected. |
| `GET /api/bench/providers` | anyone | The measurement catalog: libraries, provisioning, kernels and their declared capabilities. |
| `GET /api/bench/browse` | anyone | One page of one Kernel Bench page view (§5.4). |
| `POST /internal/bench/lookup` | internal | The candidate rows of an estimate's keys at a snapshot (§7.2). |
| `GET /internal/bench/version` | internal | The lake id, the snapshot and the shared rows' snapshot with their content ids, and `rules_rev` (§7.1). |
| `GET /health` | internal | The lake's state, lease, backup status and lag, free space, and lookup statistics. |

A lake that does not serve, a lost commit race, or a refused write answers
503 with the reason (`routes.lake_unavailable`).

### 5.4 Kernel Bench page views

`GET /api/bench/browse` (`browse.py`) computes one view per request from the
stored tables joined with `annot`; nothing is aggregated in advance. Views:
`envelope`, `rows`, `workload`, `curves`, `coverage`; every view but
`curves` also returns facets (each filter's values under the other
filters). Scopes: the shared corpus, the caller's rows, both (a row counted
once), or one of the caller's sets; signed out, only the shared corpus is
visible.

An envelope's winner is its fastest row (ties broken by the smaller id), its
gain the runner-up's latency over the winner's, less one; it lists the 20
fastest candidates and every distinct kernel and model among its rows. The
latency views read only measurements; the rows view may include failures
(`outcome=all`). A request beyond a view's bounds is refused with 400 naming
the bound. Display fields computed per row (MFU, MBU, roofline status, node
label) come from `op_calibration.measured_utilization` and `node_label`.

### 5.5 Health

The service's `/health` never reads the lake and never fails: it reports the
gate's state and reason, the lease state, the backup's status (newest
recoverable snapshot, lag, last error), free space, and lookup statistics
(count, keys, rows, rows per key, the largest answer, refusals, the bound).
The gate also reports `data_rules`: the registered revision, the lake's
revision while a migration is pending, the completed migration's counts, or
the migration error. A lake behind or failed is `degraded`, and the HA API's
`kernel_bench` health carries that state instead of reporting the service up.
The HA API's `/health` includes it (`bench_proxy.bench_health`, bounded at
2 s).

---

## 6. Private and shared rows

### 6.1 Review state lives on batches

Review state is the batch's `status`, never a row's: `private`,
`nominated`, `shared` or `rejected`. A row is **shared** iff its batch is
`shared` and either the batch was not nominated within peak, or the row is
within both peaks it records (`store.SHARED_ROW`, the one SQL statement of
the rule, parity-tested against `measured_utilization`). Promoting or
rejecting a batch changes one `batches` row; no measurement is rewritten.

A batch is one upload. A run is a grouping attribute (`run_id`); re-uploading
into a run adds a new batch that starts private and is reviewed on its own.
The list and review pages derive a set's fields from its batches
(`sets.py`): a set is public when some batch is shared, and its status is
the first of `private` > `nominated` > `shared` > `rejected` that any of its
batches has.

### 6.2 Review flow

1. The owner uploads: the batch is private and calibrates only the owner's
   estimates (§6.3).
2. The owner nominates a run or upload (`.../contribute`), optionally
   offering only rows within their recorded peaks (`within_peak`).
3. An admin reads the pending queue (`/api/bench/nominations`) and the set's
   overlap with the shared corpus, then promotes or rejects. Promoted rows
   are in the public envelope and in every estimate's lookup; rejected
   batches stay their owner's.

An admin may also publish an upload directly as shared (`hbench upload
--public`). The owner may delete a run, an upload or selected rows at any
status; deleting a shared row removes it from the shared corpus.

The H100 and RTX-4090 harvesters use the same public review flow
(`tools/upload_to_authoritative_lake.py`): a personal access token of the
contributor/admin account, upload private, nominate `within_peak`, promote.
The loopback service on port 8100 is not an upload target; a local lake is a
restore of the authoritative backup. The queue may still ask that local copy
the read-only coverage question while it plans work.

### 6.3 How private and shared rows merge

An estimate reads:

- **the shared rows**, always, when measurements are read;
- **the caller's own rows**, on top, when the request sets
  `calibrate_with_mine`. The service selects both in one lookup; each row
  carries an **origin** (`shared` or `mine`). The client indexes shared rows
  first and the owner's over them, so the owner's row wins a collision on
  the 12-column identity (`op_calibration.registry_from_lookup`). A row that
  is both shared and the owner's is sent once per origin. Other users'
  estimates are untouched;
- **request rows**, for headless callers: `calibration_rows` on the request
  (at most 20,000; 422 beyond), re-validated and merged over the looked-up
  rows for that request only (`merge_validated_rows`).

The origin is kept on every merged row so the analyzer and the Kernel Bench
page can attribute a measurement to the shared corpus or to the user's own
upload. A shared row repeated within one answer is an integrity error.

---

## 7. How an estimate reads measurements

### 7.1 Measurement scope

Before an estimate or sweep is offloaded, joined with in-flight work, or
keyed for a cache, the HA API sets its **measurement scope**
(`backend/api/_access.with_measurement_scope`, `backend/measure/scope.py`):

```
MeasurementScope(owner, lake_id, snapshot, shared_snapshot, rules_rev,
                 snapshot_content, shared_snapshot_content)
```

- **owner**: the caller's user id, resolved from their session, when the
  request asks `calibrate_with_mine` and reads measurements. Without a valid
  session the request is 401; without the service layer installed, 501. The
  client never names an owner.
- **the rest** comes from one `GET /internal/bench/version`: the lake, the
  snapshot the lookup will be pinned at, the snapshot at which the shared
  rows last changed (`lookup.SharedRevision`), each with its content id
  (§4.4), and `rules_rev` (§4.2).
- **`NONE`**: a request whose mode reads no measurements
  (`theoretical_only`, or a sweep's SOL half) or a deployment with no service
  configured gets the constant scope and makes no request to the service.

The scope is an internal request field: excluded from every dump and
refused (422) when a body spells it.

### 7.2 The lookup

The estimate collects the physical keys of the ops it prices
(`calibration_overlay.OverlayWalk`) and asks the service for all of them in
one request (`backend/measure/lookup.py`):

```
POST /internal/bench/lookup
  {keys: [{signature, physical_shape, dtype_env_json, gpu, phase}, …],
   owner: <uuid|null>, snapshot: <int>}
→ {lake_id, snapshot, columns, values, rows: [[row cells…] per key], excluded_mine}
```

**Selection.** For each key the service returns every shared row, and every
row of the named owner, whose input's current annotations carry the key's
signature and physical shape and whose own dtype environment, GPU and phase
equal the key's, as text. A hash of the dtype environment and GPU may
narrow the search; text equality decides. Rows are read at the pinned
snapshot. One projection decides membership — the service's current
annotations — and the client indexes each row under the key that selected
it, never re-projecting the row's stored shape (`registry_from_lookup`).

**Validation.** The write path validates every upload, and a validator change
is applied to the lake by the registered data-rules migration before it serves
(§4.1). Lookup does not re-run Python's validator for every selected row in
every process; that defense cost about 80 µs per row and was a read-path
duplicate of the migration owner. `/api/bench/calibration` still validates and
explains one owner's rows on its own page. `excluded_mine` remains in the wire
contract and is zero for a validated, migrated lake.

**Wire form.** A row travels as the columns the estimator's registry reads
(`op_calibration.ROW_COLUMNS`: the 12-column identity, the measurement,
cross-check, outcome and driver columns) plus its origin; each cell is an
index into the answer's table of distinct texts (`values`). Provenance the
estimate never reads stays in the lake.

**Bounds and memory.**

| Bound | Value | Behaviour |
|---|---|---|
| Keys per lookup (`lookup.MAX_KEYS`) | 20,000 | Request refused. |
| Candidate rows per answer (`lookup.MAX_ROWS`) | 50,000 | 503 before any wide column is read; never truncated, since a partial candidate set would change what an estimate picks. |
| In-memory selection table (`BENCH_LAKE_SELECTION_LIMIT`, a share of `BENCH_LAKE_MEMORY_LIMIT`) | default 32 MiB (~460k rows at ~72 B a row) | Above it, or at an older pin, the service runs the exact pinned query over the lake's files instead. |
| Validator verdicts held by the service (`Verdicts`) | 120,000 row-content keys | Owner calibration only; lookup does not put cells or unvalidated “valid” marks here. |
| Lookup cells file (`lookup.Cells`, `<cache dir>/lookup_cells.duckdb`) | every row's `ROW_COLUMNS`, each version with the snapshots it is live at (4.28M rows: ~230 MB on disk, built in ~5–10 s) | The one place a lookup reads a row's cells; built for the current snapshot before the service serves. A pin past the file waits up to `TIMEOUT_S` (5 s) for it to catch up by DuckLake's change feed, else 503; a pin before its first snapshot is 503. |
| Rows held per client process (`measure.lookup.ROWS_KEPT`) | 50,000 rows | Evicted keys are asked again. |
| Client timeout (`lookup_timeout`) | 5 s + 2 ms per key sent | A slower answer is a failure (§7.3). |

**Cold reads.** A cold lookup's rows are a random point set over the lake
(row and input ids are hashes), so reading their wide cells from parquet
scans every data file; reordering files cannot help. The service therefore
reads a row's cells from one place, a native DuckDB table (`lookup.Cells`)
that keeps each row's versions with the snapshots they are live at, the way
DuckLake keeps the rows. The lake's wide columns are read only to build it
(at startup, before the service serves) and to catch it up by the lake's
change feed; a lookup never reads them. A lookup pinned past the file wakes
its follower and waits for it, at most the client's floor (`TIMEOUT_S`), then
fails as unavailable; a pin before the file's first snapshot (older than the
process's start, or than the lake's kept history) is refused. At 4.28M rows
a cold 5,000-key lookup went from 2.6–2.8 s to 1.0 s, the slowest of 8
concurrent 2,000-key lookups from 8.0 s to 2.9–3.3 s, and a warm lookup costs what
a cold one does (no per-process row cache). Narrowing candidates before the
wide read was measured and not built: per key, 3.1 rows arrive and 3.06
survive the client's one-row-per-implementation dedup, so it saves about 1%.

**Held rows.** Each process (the API and each compute worker) keeps the rows
of the keys it has looked up, keyed by the scope's cache identity and the
key, and asks the service only for keys it does not hold. The entries are
exact, never stale: a shared scope's identity names the shared rows'
snapshot, an owner's the pinned snapshot.

Sweeps look up the union of their points' keys; see
[`auto_sweep.md` §5](auto_sweep.md#5-the-measured-side).

### 7.3 Three outcomes, never confused

| Situation | Result |
|---|---|
| No service configured (`HUGGINGARCH_BENCH_URL` unset) | No measured rows; the estimate is SOL and says so (`measurements: "none"`). |
| A configured service answers | Measured where it has rows (`measurements: "kernel_bench"`); an empty answer is a normal answer and is cached like any other. |
| A configured service fails, times out or answers something malformed | `MeasurementsUnavailable`: the estimate fails with 503 "measurements unavailable" and nothing is cached. |

Responses also carry `measurement_scope`, a digest of the scope the
measured track was read under, for display.

### 7.4 Selection modes

The looked-up rows form a registry indexed by physical key
(`backend/inference/op_calibration.py`). The request's mode decides which
row prices each op (`InferenceEstimateRequest.resolved_selection_mode`):

| Mode | Selection |
|---|---|
| `envelope` (default: `prefer_measured=true`) | The fastest row of any library and version at the op's physical key (`physical_time_for`). |
| `latest` | The framework binding's exact declared provider implementation (`kernel_dispatches`: library and complete implementation id, compared by equality, no prefix or fastest-row fallback) at the latest release the corpus measured it on (`latest_time_for`). An older release fills a gap and is labelled as itself. |
| `kernel_mix` | A debug pin of a specific `kernel_impl` per op. |
| `theoretical_only` (`prefer_measured=false`) | No measurements; no lookup. |

Framework bindings are not versioned (`audited_at` names the release their
facts were read at), so an estimate never asks for a version; the badge
names the release a row answers with and distinguishes the serving
framework from the provider library.

The measured time is kept beside the roofline rather than replacing it.
Each row records the GPU's peaks at test time; when the current GPU table's
peaks differ, the latency is scaled by the ratio of the bound the row
saturated (compute or memory; a mixed row takes the slower result). A
measurement faster than the roofline at test time is still reported, with
an anomaly badge. Measured times are not published at `pp > 1`
([cost model](inference_cost_model.md)).

### 7.5 Cache identity

What identifies an estimate's result is answered in one place,
`backend/measure/scope.ResultIdentity`: the request content (the body
without the server-set scope, the precision profile by key), the engine
version, the measurement scope (or `NONE` where the result reads no
measurements), and whether the request posted its own rows. Every cache of
an estimate's result takes its key and its remote-tier eligibility from it:
the offload single-flight, the sweep caches, the Gallery's facts and decode
frontier, and the cache warm. `tests/meta/test_result_identity.py` fails
when one of them builds a key another way.

- A **shared** scope keys on the shared rows' snapshot and content id, so a
  private upload elsewhere does not invalidate shared results. An
  **owner's** scope keys on the pinned snapshot.
- Results with an owner or with posted rows never enter the remote tier
  other users read (`remote_ok`).
- **Theory results carry no scope element**: uploads, promotions and rule
  changes never move a theory key.
- A measured sweep is computed per request and never stored; its caches
  hold only the theory half and the SOL answer. The browser's in-tab sweep
  cache holds SOL answers only. See
  [`auto_sweep.md` §7](auto_sweep.md#7-caching-theory-and-measured-answers).

---

## 8. Deployment facts

### 8.1 Process layout

The production image runs one container on one machine with a persistent
volume at `/app/.cache` (`fly.toml`: one machine always running,
`HUGGINGARCH_BENCH_RESTORE=on`). `scripts/entrypoint.sh`:

1. As root, before anything serves: keeps the volume's top root-owned,
   chowns only the agent's directories to the agent user, and creates
   `bench_lake` root-owned with mode 0700.
2. Writes a new shared secret for this container start into a 0600 file
   (`HUGGINGARCH_BENCH_SECRET_FILE`) and exports `HUGGINGARCH_BENCH_URL`
   (`http://127.0.0.1:8100`) for the API, its compute workers and the cache
   warm.
3. Starts the Kernel Bench service, waits until it answers `/health`
   (bounded by `BENCH_START_WAIT_S`, default 60 s; past that the API starts
   anyway and `/api/bench/*` answers 503 until the service is up), then
   starts the HA API on `0.0.0.0:$PORT`.
4. When the lake serves (bounded by `BENCH_SERVING_WAIT_S`, default 1800 s),
   runs the cache warm once at low priority in its own process
   (`python -m backend.warm_caches $HA_WARM_ARGS`), folding measured tracks
   under the lake's real scope; `HA_RUNTIME_WARM=off` skips it. The warm is
   not a supervised process: its exit stops nothing.
5. When the service or the API exits, stops the other and exits with the
   first one's status; the platform restarts the machine and the service's
   gate restores its state. There are no in-container restarts.

The image's build-time warm runs without a service, so it bakes only theory
results, which are valid in production as baked.

### 8.2 Configuration

| Variable | Read by | Meaning |
|---|---|---|
| `BENCH_LAKE_CATALOG`, `BENCH_LAKE_DATA` | service | Catalog file and data directory (set in the Dockerfile). |
| `BENCH_LAKE_CACHE`, `BENCH_LAKE_MEMORY_LIMIT`, `BENCH_LAKE_SELECTION_LIMIT` | service | Spill directory; DuckDB memory limit (default 384 MB); the share of it the held selection table may take (default 32 MiB, must be below the memory limit). |
| `BENCH_LAKE_S3_*` | service | Credentials for an `s3://` data path (unused in production). |
| `BENCH_LAKE_BACKUP_DATA`, `BENCH_LAKE_BACKUP_S3_*` | service | Backup location and its own access key. |
| `BENCH_LAKE_BACKUP=off` | service | Development only: no backups. |
| `BENCH_LAKE_LEASE=off`, `BENCH_LAKE_MACHINE` | service | Development only: no lease; machine id outside the platform. |
| `HUGGINGARCH_BENCH_RESTORE=on` | service | Restore an empty volume from backup at start. |
| `HUGGINGARCH_BENCH_MAINTENANCE=off` | service | Disable the maintainer. |
| `HUGGINGARCH_BENCH_PORT` | service, entrypoint | Service port (default 8100). |
| `HUGGINGARCH_BENCH_URL` | API, workers, warm, harness | The service's URL; unset means no measurements (§7.3). |
| `HUGGINGARCH_BENCH_SECRET`, `HUGGINGARCH_BENCH_SECRET_FILE` | all | The shared secret. |
| `HUGGINGARCH_BENCH_OWNER` | harness | The contributing account whose own rows count as covered (§3.4). |

### 8.3 Memory and disk budget

The production machine has 6 GiB of memory. The budget `fly.toml` is sized
for: the Kernel Bench service about 220 MiB (provider discovery imports no
kernel library, so the service does not load torch); the API parent about
110 MiB; each compute worker about 145 MiB idle, keeping the cells of the 16
models it ran last (up to about 0.93 GiB when all are the largest models) and
adding up to about 0.85 GiB while computing one large model; the warm parent
about 110 MiB; each spec agent 250–400 MiB. With two API workers
(`HA_SWEEP_POOL_WORKERS=2`) and a one-worker warm (`HA_WARM_WORKERS=1`), the
worst case is about 4.5 GiB, leaving room for three agents
(`SPEC_AGENT_MAX_CONCURRENT=3`).

Inside the service, DuckDB's memory is bounded by `BENCH_LAKE_MEMORY_LIMIT`,
the selection table by `BENCH_LAKE_SELECTION_LIMIT` (within that limit), and a lookup's answer by 50,000 rows (§7.2).
The volume is 3 GB; writes, maintenance and backups stop when less than
500 MiB is free (§4.3).

### 8.4 Local development

`scripts/dev_bench_service.sh` gives a local supervisor the same layout:
`secret RUN_DIR` writes a new 0600 secret once per start, `env RUN_DIR`
prints the exports the API, workers, warm and service need, `run` execs the
service with the `BENCH_LAKE_*` settings from `.env`, and `pid` / `health`
report its state. A local lake uses a file catalog in a local directory,
usually with `BENCH_LAKE_LEASE=off` and `BENCH_LAKE_BACKUP=off` (or a local
backup directory). A fresh local lake is created as `lake_role=development`
with its own UUID and may write. Restoring an authoritative backup locally is
for reads and debugging: without the lease it stays read-only even when
`BENCH_LAKE_LEASE=off`. Never change that copy's `lake_id` by hand; build a
new development lake instead.

---

## 9. Code reference

| Concern | Location |
|---|---|
| Row schema and validator (the shared contract) | `backend/inference/benchmarks.py`, [`BENCHMARKS_CSV.md`](../backend/inference/BENCHMARKS_CSV.md) |
| The measurement key (one derivation) | `backend/inference/calibration_overlay.py` (`point_keys` → `PointRow`) |
| Physical key, registry, selection modes, peak-drift compensation | `backend/inference/op_calibration.py` |
| Op signatures | `backend/inference/op_signatures/`, `kernel_bench/signatures.py` |
| Providers and library records | `kernel_bench/providers/*.py`, `kernel_bench/providers/_libraries.py` |
| Reference and cross-check | `kernel_bench/refs.py`, `kernel_bench/driver.py` |
| Clock | `kernel_bench/timing.py` |
| Environments and box profiles | `kernel_bench/envs/*.txt` (generated), `kernel_bench/envs/boxes/` |
| Plan, run, check, upload CLI (`hbench`) | `kernel_bench/cli.py`, [`kernel_bench/README.md`](../kernel_bench/README.md) |
| Lake instance, locks, content identity | `backend/service/bench/lake.py` |
| Tables and writes | `backend/service/bench/store.py` |
| Registered lake data-format migrations | `backend/service/bench/data_rules.py`, `backend/service/bench/data_rules_version.json` |
| Kernel Bench page views | `backend/service/bench/browse.py` |
| Batch pages | `backend/service/bench/sets.py` |
| Estimate lookup (service side) | `backend/service/bench/lookup.py` |
| Startup gate, lease, restore | `backend/service/bench/serving.py` |
| Replacement lake lineage | `tools/reidentify_authoritative_lake.py`, `supabase/migrations/0027_bench_lake_transfer_lineage.sql` |
| Backup and restore | `backend/service/bench/backup.py`, `scripts/bench_lake_backup_now.py` |
| Maintainer | `backend/service/bench/maintain.py` |
| Annotation rules and kernel catalog | `backend/service/bench/annotations.py` |
| Data-rules revision boundary | `backend/service/bench/rules_revision.py` |
| Kernel Bench HTTP routes | `backend/service/bench/routes.py` |
| Service process, trust boundary | `backend/service/bench/server.py`, `backend/service/bench/identity.py` |
| API proxy | `backend/service/bench_proxy.py` |
| Measurement scope, result identity | `backend/measure/scope.py`, `backend/api/_access.py` |
| Lookup client, coverage question | `backend/measure/lookup.py` |
| Container layout | `scripts/entrypoint.sh`, `Dockerfile`, `fly.toml` |
| Lease table and functions | `supabase/init.sql` |
