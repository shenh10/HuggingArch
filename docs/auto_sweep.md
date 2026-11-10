# Sweeps: one walk per cell, substitution per point

This document describes how HuggingArch answers one estimate question at many
points — many batches, contexts, cards and parallel layouts — without walking
the model once per point, and how the measured (Kernel Bench) side of such a
sweep is read. It covers the Scaling panel's two sweeps (Context × Batch and
the parallel-degree sweep), the Compare page, the Gallery decode frontier, and
the single estimate, which is a sweep of one point.

Related documents: [`measured_calibration.md`](measured_calibration.md)
(where measured rows live, how a lookup is served, §4–§7, and how a measured
row is selected, §7.4), [`inference_cost_model.md`](inference_cost_model.md)
(what a row's roofline law is).

## 1. Purpose and terms

### 1.1 Cells and points

Two facts make a sweep cheap.

- **One walk per cell.** A *cell* is everything that changes the program the
  engine walks: the model, the parallel layout, the phase, the precision
  profile and, under pipeline parallelism, the pipeline stage whose capacity
  binds. The card enters only through its *walk class* (§4). Walking a cell
  is the expensive part: seconds per walk.
- **Substitution per point.** Inside a cell, a *point* differs only in the
  workload's counts — the batch `B`, the prompt `s_input`, the cached prefix
  `s_prefix`, the output length, the verify width `s_verify` — and the card
  within the walk class. Every per-op row the walk emits already carries its
  roofline as a sympy law over those counts and the card's peaks, and its
  shape as an exact law over the counts. A point is priced by substituting
  into those laws: milliseconds per point, and no new walk.

The acceptance rate of speculative decoding and the two time floors (the
launch floor and one link traversal's latency) are substituted later still,
into a step's already-priced form (`row_roofline.FloorForm`), and are
therefore not part of what a walk is keyed by.

### 1.2 Theory and measurement

A sweep has two halves with different invalidation needs.

- **Theory**: the SOL numbers, each row's laws, each point's lookup keys and
  the measured formula (§5.3). None of these depends on what Kernel Bench
  holds. Theory is cached without a measurement scope and can be produced at
  image build.
- **Measurement**: the measured time of each key, read from Kernel Bench at
  request time, and the fold of those times through the formula. Only this
  half depends on the lake.

The measured end-to-end time of a phase exists only when every row of the
phase has a measured time (§5.4). Otherwise the point reports its coverage.

## 2. Layering and ownership

### 2.1 The engine knows no models

`backend/measure/auto_sweep.py` is the sweep engine. It imports the standard
library, sympy, its siblings `backend.measure.{lookup, pool, scope}`, and the
law-text reader `backend.arch.law_text.read_walk_law` (the single owner of
the operator vocabulary a law may name; a copy in `measure` would be a second
table). It imports no spec, config, `Analysis` or card table.

The enforced rule is a ban, not an allowlist: all of `backend/measure` stays
free of `backend.inference`, `backend.api`, `backend.service`,
`backend.rendering` and `kernel_bench` (`tests/meta/test_eval_boundary.py`,
`DIRECTION_BANS`). `backend.inference` imports the engine, never the reverse.

The engine provides:

| Name | Role |
|---|---|
| `Cell(walk, points)` | One walk's picklable arguments and the coordinates substituted into it |
| `walk_cells` | Maps a walk over cells: across the process pool (`pool.map_cells`) or through a caller's mapper |
| `compile_law` / `evaluate` | The one compiler of law text: exact (`Fraction` until a float enters), CSE over whole laws and ordered graph assignments, `Piecewise`-aware |
| `PointMeasure` | One phase of one point before measurement: per row its key or why it has none, its launches, its label and reason; the phase's formula; the selection context |
| `sweep_keys` / `look_up` | The union of every point's keys, and the one lookup over it |
| `MeasuredFormula` | The measured total Σ w_k·n_k·T_k as law text, plus which terms cost nothing |
| `Folded` | A `PointMeasure` folded under a selection: per-term state and time, `track` (the whole) and `subtotal` (any part) |
| `MeasuredTrack` | Coverage over rows and the total at full coverage; `STATES` is the closed coverage vocabulary |

### 2.2 What the estimator supplies

Everything about models stays in `backend/inference`, passed through the
engine's interface as facts: what a cell's walk produces
(`capacity._capacity_pending`), the row laws and the card binding
(`row_roofline`, `batch_sweep.rows_at`), capacity fit and the bound pipeline
stage (`capacity.capacity_ceilings`), which layouts are valid
(`parallel_sweep.legal_parallel_degrees`), the point-key derivation
(`calibration_overlay.point_keys`, §5.1) and the measured selection
(`op_calibration.select_time`, modes `latest` / `envelope` / `kernel_mix`).

### 2.3 Plain data at the boundary

Everything that crosses the engine boundary is plain data: law text, numbers,
strings, key tuples. A cell can therefore be walked in a pool worker and its
points returned to the API process, which makes the one lookup and the fold.
`PointMeasure.to_json` stores each repeated string (a key's card, phase and
dtype environment) once in a table. Nothing is pickled into a cache.

## 3. Pricing a point by substitution

### 3.1 Row laws

Each per-op row carries three families of law (`batch_sweep` module
docstring):

- `roofline_expr` — its FLOPs, byte columns, collective payload and per-rank
  volume, and the four times, over the workload counts and the card's peaks
  (`row_roofline.build_row_exprs`);
- `shape_law` — its per-rank, exact and unsharded shapes, with source ops;
- `point_law` — everything else the row publishes that moves with the point:
  node dims, the FLOPs formula's substitution record, access-formula display
  facts, a collective's payload and buffer shapes.

`batch_sweep.rows_at` substitutes into these laws (each compiled once per
cell, `row_roofline.LawBook`) and renders the remaining tail — the binding
term, arithmetic intensity, popover text, the phase note, throughput — with
the same functions the walk uses for its own rows. A row re-read at a point
is therefore the row a walk there would emit. A phase's totals are the step's
one sum over its rows (`row_roofline.StepSum`).

A re-read row carries no measurement identity (`signature`, `shape_json`,
`dtype_env_json`); those fields are absent, not `None`, until the sweep keys
the row at its point (§5.1).

### 3.2 Context × Batch: one cell

`capacity.CapacitySweepPlan` describes a (context × batch) sweep. Its contexts
are prompt lengths with their cached prefixes (`s_prefixes`) and verify widths
(`s_verifies`); its batch points are `(key, prefill_batch, decode_batch)`
triples. The whole sweep is **one cell** (`CapacitySweepPlan.cells`): context
and batch are both point coordinates.

`capacity._sweep_cell` walks once — at the first context, batch 1 — and reads
every (context, batch) point off that walk:

1. per context, the capacity ceilings at that length (`capacity_ceilings`);
2. per batch point, the batch a direct estimate would run
   (`_direct_batches`: prefill the override, `-1` its activation-bound
   ceiling, else 1; decode the override, else the KV-bound ceiling);
3. the rows at the point (`_point_rows` → `batch_sweep.rows_at`), keyed there
   when the request reads measurements (`PendingEstimate.keyed_at`, §5.1);
4. the phases summed from those rows (`_stamp_gpu_on_result`), and a long
   prefill folded over its chunk schedule.

Two facts can force another walk inside the same cell. A context whose
ceilings bind a different pipeline stage reads that stage's own walk. A
context at a verify width the first walk does not take as a symbol (a step
that does not speculate beside one that does) walks its own; every
speculative width otherwise shares one walk
(`_CapacityPreparation.walks_alike`).

Each point is `{"sol": <estimate, unmeasured>, "measures": <PointMeasure per
phase>}`. Facts of the anchor walk that no law carries (`vs_naive_delta`) are
not passed on to points. `_sweep_cell` yields per context as each finishes,
which is the unit of progress reporting and resumption (§7.4).

### 3.3 The parallel sweep: one cell per layout

`parallel_sweep.ParallelSweepPlan` enumerates one `LayoutCell` per (mode,
degree) for the requested modes (`dp_ep`, `cp_ep`, `tp_ep`, `tp_tp`); the card
series (`DEFAULT_SWEEP_GPUS`) and the cards' batches are points.
`_estimate_layout` walks once per (card class, bound pipeline stage), restamps
each card of the class off that walk (§4), re-reads the rows at the card's
batch (`_point_batches`: the card's ceiling, or a pinned batch clamped to it)
and returns `{gpu_sku: {"sol", "measures", "ladder"}}`. A card that lists no
peak for a dtype the step computes in fails alone; a layout the model does not
deploy under is reported as refused, with the layout's reason
(`run_layout_cell`).

Each layout is also read at every per-GPU decode batch of
`sweep_grid.SCALING_DECODE_LADDER` (`_decode_ladder`, SOL only). Those points
form the sweep's decode point cloud (`point_cloud`, the one producer of cloud
records); a decode Pareto frontier is a view of that cloud
(`decode_frontier`), computed on the server from the cached answer.

`ParallelSweepPlan.assemble` makes the sweep's one lookup (§5.2) over every
card of every layout, then turns each card into a chart point
(`parallel_sweep._point`).

### 3.4 The single estimate is a one-point sweep

`/inference/estimate` finishes its walks without reading any measurement
(`PendingEstimate.finish`), states its measured totals as `PointMeasure`s
(`PendingEstimate.measures`), and calls `batch_sweep.measure_points` once —
the same call a sweep makes (`capacity.finish_estimates`). A disaggregated
prefill/decode pair is two estimates finished together over one lookup. There
is no second measured path for the single estimate.

### 3.5 Parity

The contract every sweep keeps is: **a swept point is the estimate a direct
call at that point returns.** `tests/sweep_parity.py` is the one statement of
"the same" — integers, booleans, strings and `None` compared exactly, floats
to a relative 1e-12 (substitution sums in another order than a walk) — with a
corpus-snapshot deployment, a direct estimate at a point, and an in-process
Kernel Bench stand-in (`FixtureBench`) that counts lookups.
`tests/backend/inference/test_sweep_parity.py` holds the sweeps to it: every
card of the parallel sweep, every (context, batch) point of Context × Batch,
every per-op row's law columns, the measured fields and coverage, and the
number of walks and lookups. For a model whose fused graph depends on the
card, the per-op row set is compared across the whole Scaling series, not
only the phase totals (§4.3).

## 4. Cross-card re-pricing

### 4.1 Two views of a card

`inference/card.Card` is the one way to name a card, and it has two views of
distinct types:

- `Card.klass` → `framework_bindings.CardClass`: the **structural** facts — what
  a framework's dispatch reads of the card (platform, compute capability) and
  which peak columns the sheet lists. These change a walk's rows (which fused
  kernels apply, which launches run in a quantized format), so they are what a
  walk is keyed by.
- `Card.price(gpu_memory_fraction)` → `CardPrice`: HBM budget, peaks, fabric.
  These change only numbers and are substituted into an already-walked row.

Every column of `gpu_specs.csv` is classified into one view at load
(`framework_bindings._check_card_fact_classification`); an unclassified
column fails the load.

### 4.2 The rule

**A walk may be re-priced on another card of its own walk class, and on no
other card.** A card of another class is refused, never re-priced: it is the
one way this reuse can be wrong while the numbers still look plausible. The
rule has one statement, `Card.refuse_unless_one_class`, and every road by
which a walk reaches another card asks it:

| Road | What moves | Where |
|---|---|---|
| Preparation | The card's HBM, peaks and fabric under the same walk and weights | `capacity._restamp_gpu` |
| Rows | Every row's law read at the other card's peaks and fabric | `capacity._point_rows`, `_stamp_gpu_on_result` |
| Kept class sweep | A step's priced form (`CardForm`) read on another card | `the serving-law pool.of` / `on_card` |

A walk also names the card it was taken on (`PendingEstimate.walked_on`), so
`PendingEstimate.priced_on` can check, in one call, both the class and the
pipeline stage the destination card's HBM binds
(`CapacityCeilings.refuse_unless_one_stage`): a card with less HBM can run out
on a different stage, and one stage's rows under another stage's batch are no
rank's answer.

### 4.3 The walk card of a class

`parallel_sweep.class_cards(gpu_sku, series)` names the cards of `series` that
share `gpu_sku`'s walk class, roomiest first. The roomiest is the class's walk
card: a deployment that fits no card of the class fits none of them, so its
walk can price the rest. The Gallery uses its own card series, the Scaling
sweep the sweep series; a card a series omits answers for itself alone.

This class of error does not crash, empty a response or change row counts
reliably; it biases numbers, usually optimistically. A test of cross-card
pricing must therefore compare the per-op row set and order, including each
row's dtype, and not only `t_sol` or the row count.

## 5. The measured side

### 5.1 Point keys

`calibration_overlay.point_keys` is the one derivation of what each per-op row
looks up and launches at a point. It answers one `PointRow` per row:

- **identity**: the kernel behind the engine's label, the measurement
  signature, the identity shape, the leaf's own dtype environment
  (`benchmark_identity.RowIdentity.leaf_dtype_env`) and the
  `op_calibration.PhysicalKey` they project to. A row with no signature or no
  shape projection has no key; that is a refusal, not a gap.
- **billing**: `launches` (the step's launches of the identity shape) and
  `cadence` (the share of one launch a call is — `1/m` for a row that launches
  once every m steps). The step bills one launch's time
  `launches × cadence` times; this is the only place a launch count enters
  the measured track.
- **refusal**: `declined`, with its reason, when the leaf's dtype environment
  has no corpus token and the step's environment would stand in for it. Such
  a row is never looked up.
- **zero cost**: read off the row's law (`calibration_overlay.zero_cost`): no
  FLOPs, no access bytes, no collective payload.
- **placement reason**: a closed reason (`zero_launch`, `missing_signature`,
  `missing_projection`, `pipeline_stage`) for a row with no exact key.

The paged-KV page size is part of the identity of every kernel that reads a
paged cache (`OpSignature.reads_paged_cache`); `point_keys` builds its
`RowIdentity` with the deployment's page, so every reader keys on it.

Every reader of a row's key asks `point_keys`: the estimate's overlay, every
sweep point (`PendingEstimate.keyed_at`), the Kernel Bench plan
(`kernel_bench/plan.py`), and the contributor template
(`cli_bench.emit_suggest_shapes`). The write side and the read side therefore
state one key: a job's key *is* the key the estimate asks for. The gate is a
byte comparison, not a scan:
`tests/kernel_bench/test_plan_keys_are_the_estimates.py` asserts this on a
bf16, an fp8 and an fp4 model. (Two derivations of the dtype half agreed on
every bf16 model and disagreed on every row of every quantized model, so a
bf16-only check could not have caught the divergence.)

### 5.2 One lookup per sweep

A sweep makes **one** Kernel Bench lookup for the union of every point's keys:

1. a canonical Gallery point names a row of a record-level
   `MeasureMatrix`; a live Inference or Scaling point becomes a one-row (or
   small dense) matrix at the fold boundary, so there is one path at both
   granularities;
2. `batch_sweep._measured_selector_for` resolves whether the request reads
   measurements, collects the matrix's distinct `(key, context, op, input)`
   questions, calls `auto_sweep.look_up` once, builds the registry
   (`op_calibration.registry_from_lookup`), and merges the request's own
   `calibration_rows` over it;
3. `MeasureMatrix.fold` selects each question once, binds the selected launch
   times through the point × term key matrix, evaluates the formula over
   whole rows, and vectorises coverage by state. `batch_sweep.apply_measured`
   unfolds the row a response asks for: each row's own time, each module
   group's and declaration parent's share, the phase total and coverage, the
   draft's share, the measured throughputs.
4. The same fold records stage timings — reading `PointMeasure`s, collecting
   and deduplicating keys, the lookup round trip (service and network split
   by the service's `service_s`), selector construction and selection, and
   `apply_measured`. Gallery publishes them under `measured_track.timings`;
   the estimate route records the same vocabulary in `timings`.

The selection strategy and any kernel pin come from the request at fold time,
not from the cell, so one set of measures serves every selection mode. The
Context × Batch sweep makes its lookup in the process that assembles it, the
parallel sweep in `ParallelSweepPlan.assemble`, the single estimate in
`finish_estimates`.

### 5.3 The measured total is a formula

The measured total of a phase at a point is the same aggregation as its SOL
total, with another leaf:

- the SOL leaf of row *i* is its roofline law `t_op_i(point, card)`;
- the measured leaf of row *i* is `n_i · T_i`, where `T_i` is one launch's
  time from the selection for `key_i(point)` and `n_i` is the row's billed
  launches at the point (§5.1);
- a zero-cost row is `0` in both, not a symbol.

`row_roofline.StepSum.measured_law` writes the formula with the same weights
as the SOL total (pipeline stages, a draft's forwards), and
`auto_sweep.MeasuredFormula` carries it as text with `n_k` / `T_k` bound per
point. `MeasuredFormula.at` reads one row; `MeasuredFormula.at_many` gives
the same compiled declaration two-dimensional `n` / `T` bindings. A matrix
fold's total and `MatrixFold.subtotal(terms)` are therefore the same formula
— the latter with every other term's time bound to zero — and a partial
subtotal is still refused. Every measured number in a response is one of the
two: the phase total, the draft's share, each module group's
`t_measured_agg`, each
declaration parent's share (grouped by `inference/declaration_groups.py`), an
encoder stack's time.

There is exactly one owner of "how long does one step of this deployment
take, measured": `MeasuredFormula` / `MatrixFold` for the value and
`MeasuredTrack` for the coverage. Code that prices a step states an unread
track (`MeasuredTrack.unread`) and waits for the fold; code that reads one
value aggregates nothing; and the frontend adds nothing up. Sums over
*measurements* — corpus rows with no launch weighting, no coverage gate and
no estimate reading them (`kernel_bench/aggregate.moe_block_mfu`, the corpus
page's latency footer) — answer a different question and are not measured
step times.

### 5.4 All or nothing, and coverage states

Each row is in exactly one state (`auto_sweep.STATES`):

| State | Meaning |
|---|---|
| `measured` | A measurement was selected for the row's key, or the row costs nothing (measured at 0) |
| `missing` | The row has a key and a declared exact implementation, but Kernel Bench has no candidate |
| `ambiguous` | `latest` has no unique framework dispatch (undeclared, inactive, or several active) |
| `unplaced` | The row has no key: no signature, no shape projection, or a pipeline stage this rank does not walk |
| `declined` | The key could not be stated exactly; carries its reason |

`MeasuredTrack.t_s` exists only when every row is `measured`. A sum that put
the roofline in for unmeasured rows would be SOL under the measured name, so
there is no such sum. Coverage counts rows, never launches; a draft's forwards
weight its terms inside the formula and leave the row counts alone.

Under pipeline parallelism only the bound stage is walked, so the other
stages' rows are `unplaced` with reason `pipeline_stage` and the step has no
measured total. A collective is a measurable kernel with its own signature; an
unmeasured collective is `missing`, and so every multi-GPU layout has no
measured total until its collectives are measured. This is the rule working:
a step whose collectives are unmeasured has no measured step time.

### 5.5 The partial track (opt-in)

Beside the all-or-nothing total there is one separately named blend,
`Folded.partial`: the step's SOL total with every fully measured row's SOL
share replaced by its measured share. Because the SOL total and the measured
formula are one aggregation (`StepSum.total` over `t_op` and over `n·T`), a
row's SOL share is its term's weight times its `t_op_s`, so

    t_partial = t_sol − Σ_{measured rows} w·t_op + Σ_{measured rows} share

A row with any term not measured keeps its SOL time whole. The fold states
`t_partial_s` and `partial_measured_share` (the measured part of it, by time)
on each phase, and `throughput.partial`; at no coverage it is `t_sol_s`, at
full coverage `t_measured_s`. It is never called measured: the page shows it
only behind a "Partial measured" checkbox, always beside its coverage.

`MeasuredTrack.coverage()` also reports `unchecked`: how many measured rows
rest on a corpus row whose cross-check is not `passed` (typically no fp32
reference could be built). Those rows are used, because refusing them would
fall back to the roofline, but the count lets a reader tell a total built on
checked numbers from one that is not.

### 5.5 Speculative steps in the measured track

The draft is a third row list of the same walked step, not a second measured
system. The canonical record's measure table is `measure-points-v7`: a point
selector carries `draftTokens`, while strings, shapes and physical keys remain
record-level facts shared by every K.

#### What a speculative step launches

A recursive MTP speculative decode step at K draft tokens runs:

1. **K draft forwards**, one per drafted token. Each is the model's declared
   MTP stack (`aux_kind: mtp`, named by `SpeculativeDraft.stacks_of`) at `B`
   requests, `s_query=1`, and the verify step's cache length. The walked rows
   are `per_op_timing_draft`; `StepSum` multiplies their formula by
   `SpeculativeDraft.forwards == K`.
2. **One verify forward** through the backbone at `s_query=K+1`. Its rows are
   `per_op_timing_decode`; the same `StepSum` adds them beside the draft.

Every launch becomes a `PointRow` through `calibration_overlay.point_keys`,
exactly as plain decode does. Its physical key is
`op_calibration.PhysicalKey(signature, physical_shape, dtype_env_json, gpu,
phase)`: `physical_identity` projects the op's shape, the leaf supplies the
dtype environment, the card stamps `gpu`, and the phase is `decode`. No
consumer derives a speculative key from K itself.

The natural overlap and the new shapes follow from those projections:

- Verify GEMMs whose identity retains `B` and `s_query` are new at K>0
  (`B×(K+1)` differs from plain decode's `B×1`).
- Grouped/routed GEMMs commonly collapse the token count into `n_tokens` (and
  `s_expert_rows`), so a K=1 verify at batch B can reuse a plain-decode key
  from another public batch when their `n_tokens` agree.
- Decode attention is new at `s_query>1` (and, where relevant, the longer
  cache length); the recursive draft's attention is `s_query=1`, so it can
  reuse a plain-decode attention key when the MTP stack's head geometry
  matches.
- MTP-stack-only head, router, embedding, quant and collective shapes are new
  when no backbone row projects to the same key.

At H100-SXM, as-shipped precision, memory fraction 0.9, the public context ×
batch grid, and each model's first Gallery layout that K=1 serves, the K=1
key tables are; counts are distinct non-null `PhysicalKey`s over that grid
zero-cost and unplaced rows are not measurements and are excluded:

| Model (layout) | Plain keys | Verify keys | Draft keys | K=1 union | New vs plain |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B (TP1) | 219 | 222 | 169 | 373 | 204 |
| DeepSeek-V4.1-Flash (EP8·DP8) | 716 | 721 | 429 | 1,127 | 830 |

A raw exact-key lookup in the read-only `/scratch1/lake-v3-candidate` (an
effective shared row exists; this is not yet the envelope selector's final
coverage) finds:

| Model | Plain hit | Verify hit | Draft hit | K=1 union hit | New-key hit |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-35B-A3B | 219/219 (100%) | 156/222 (70.3%) | 169/169 (100%) | 307/373 (82.3%) | 138/204 (67.6%) |
| DeepSeek-V4.1-Flash | 499/716 (69.7%) | 248/721 (34.4%) | 286/429 (66.7%) | 520/1,127 (46.1%) | 352/830 (42.4%) |

The two models show the two ends of the spectrum. Qwen's one-forward MTP
stack reuses plain backbone keys at K=1, while its verify width adds the 204
keys. DeepSeek reuses many backbone and routed-GEMM keys but has MTP-specific
head, router and grouped-GEMM shapes: at context 4096 and batch 8, 18 of its
39 distinct draft keys are new.

#### Ownership and the serving-law record

- `SpeculativeDraft` owns the step composition (stacks, style, forwards,
  tokens per forward). A parallel draft changes `tokens_per_forward` and
  therefore its row keys; it goes through the same `point_keys` boundary. The
  first implementation only publishes the recursive MTP style the Gallery
  currently requests.
- `point_keys` remains the only key producer for verify rows, draft rows and
  the Kernel Bench plan. There is no “speculative key” helper and no consumer
  that rewrites a plain shape for K.
- The decode binding remains the only statement of which step a point
  describes: `binding.speculative_draft_tokens == K`. A K=0 fold may not read
  a K>0 `PointMeasure`, and a K>0 fold may not substitute a K=0 measure even
  where their key sets intersect.
- The existing record-level measure table remains one table. Its schema moves
  from `measure-points-v6` to a K-aware v7 by adding `draftTokens` (default
  0) to each point selector: `(gpu, context, batch, K)`. Strings, shapes,
  dtype environments and physical keys stay globally deduplicated; a K only
  adds a `PointMeasure` variant when its formula or launch weights differ.
  Recursive draft keys are shared across K, so the table does not copy the key
  table once per K.
- The producer derives K=0 and every `FRONTIER_DRAFT_TOKENS` view from the
  same live walks: backbone rows are read at `s_verify=K+1`, recursive draft
  rows at `s_query=1`, and `point_measures` writes the K-weighted formula.
  Acceptance rate and the two floors do not change keys and do not create
  another table variant.
- Capacity continues to own serving, not the measured fold. K points are
  admitted through the record's draft capacity and the public batch grid; an
  unfit context is priced by the capacity owner's `without_step` answer, and
  Gallery/ladder filtering still uses `point_served`. The fold never broadens
  capacity to obtain a measurement.

The measured fold for a K view reads that view's `PointMeasure`, calls the
same one-lookup `measured_selector`, and applies it to a SOL point whose
binding names the same K. Complete coverage yields `t_measured_s`, the draft
share and measured throughput; partial coverage yields only per-term values
and the coverage state. No SOL value is presented as measured.

#### Measurement plan and queues

The plan does not start from the persistent key table: a stored table has
enough identity to fold, but not the full launch geometry a job needs. The
speculative planner walks the same model/layout/precision, asks `point_keys`
for the live verify and draft `PointRow`s at each public K and point, and
converts those `PointRow`s with the same launch identity the normal plan
already uses (`PointRow.key + aggregation_id`). The normal and speculative
planners share that conversion; neither writes a second shape derivation.

The queue is the gap owner: with `--skip-existing`, it asks the restored
authoritative lake under the real selection mode and runs only keys that fold
reports missing. This avoids re-measuring the large plain overlap above. A plan
may name the same physical launch under two structural nodes; after the first
measurement the coverage question answers the second, so the physical launch is
measured once. Recursive draft launches also repeat K times in a step's
formula, but the plan emits one forward's launch.

H100 and RTX-4090 receive ordinary Kernel Bench plan files from that union.
The existing queues run them, and the authoritative uploader uploads and
promotes them as shared rows. The card is stamped by the measurement driver;
the plan itself stays cardless where its class allows it.

#### Frontend behavior

The K selector no longer forces `measured_track.state = none`. It receives the
measured fold for the requested K:

- complete points draw the measured curve/rate using the step's effective
  accepted-token count;
- incomplete points draw SOL only and show `measured / total` plus the exact
  missing/ambiguous/unplaced/declined lists;
- a lake outage is `unavailable`, an engine bug is `engine_error`, and a
  model/view with no admissible K point says that — none of these is hidden
  behind the retired “Kernel Bench measures only plain decode” message.

Coverage remains a property of the K view. Partial coverage is useful
information: it names the exact launches the measurement plan will add.

#### Acceptance gates

Implementation is not complete just because a K curve appears. The gates are:

1. **Producer oracle:** for Qwen, DeepSeek and a non-MTP model, each stored K
   `PointMeasure` equals the direct speculative estimate at the same point,
   after removing only anomaly numerators as the existing pool oracle does.
2. **Binding oracle:** a fold asks for and receives exactly the requested K;
   K=0 cannot consume a K>0 point and vice versa, including points whose key
   sets intersect.
3. **Plan oracle:** the speculative plan's launch identity and `PointRow` are
   byte-equal to the direct `point_keys` result at the same model, layout,
   card, context, batch and K, including dtype and launch count.
4. **Fold oracle:** with the fixture lake, complete K coverage reproduces the
   direct measured total/draft share/throughput; missing any one key leaves
   `t_measured_s` absent and reports that key.
5. **Frontend oracle:** a K view with complete points draws measured points,
   partial coverage draws SOL plus coverage, and no code path synthesizes a
   measured value from SOL.

#### Engine and warm impact

The row key and validator schemas do not change, so the lake's data-rules
revision does not change and no lake migration runs. The serving-law measure
table schema does change, and its producer lives under `backend/inference`;
that flips `engine_version`, invalidates SOL-pool and deterministic Gallery
artifacts, and requires a new artifact generation.

This should ship in the same release as the matrix-fold and warm-scheduling
batches when they are ready. One engine version then produces one SOL-pool
archive, one deterministic Gallery archive, and one scheduled full warm.
Ideally the speculative gap plan runs on H100/4090 and its rows are promoted
into the authoritative lake before that deploy, so new K views can fold
measurements immediately; if not, their coverage panels progressively lose
missing keys as uploads land.


## 6. Response contract for the frontend

The TypeScript types are in `web/src/measured.ts` (`MeasuredCoverage`,
`MeasuredPhase`, `CoverageStates`, `UnplacedDetail`), and every page reads the
measured track through that module.

### 6.1 Rules the page applies

- **End-to-end totals** (the Inference step time, Scaling chart points, the
  Gallery's measured frontier) are drawn only when coverage is complete
  (`measured == total > 0`, `isComplete`). Otherwise the page shows
  "N / M ops measured" and the rows per state, with the keys in the hover.
  The page applies the rule itself (`completeTotal`), so a number is never
  drawn under incomplete coverage. Unknown coverage counts as incomplete.
- **The partial track** (`t_partial_s`, `throughput.partial`, the Gallery's
  `frontier_partial`) is the one blended number (§5.5). It is drawn only when
  the reader ticks "Partial measured", under its own name, with
  `partialCoverageText`: the share of the step's time that is measured and
  "N / M ops measured".
- **Per-op rows** show their own `t_measured_s` under any coverage. `0` is a
  value and renders as `0`; an absent or `null` value renders as `—`.
- **Group rows** (module groups, declaration parents) show the backend's
  share (`t_measured_agg`, the declaration map), which exists only when every
  term under them is measured (`Folded.subtotal`). The page sums nothing.

### 6.2 Coverage

```ts
type MeasuredCoverage = {
  measured: number;              // rows, not launches
  total: number;
  unchecked?: number;            // measured rows whose cross-check did not pass
  states?: {
    missing?: string[];          // keys: a key exists, the corpus has no row
    ambiguous?: string[];        // `latest` has no unique declared dispatch
    unplaced?: string[];         // op names: no signature or no shape
    declined?: { key: string; reason: string }[];
  };
  rows_by_state?: Partial<Record<"missing" | "ambiguous" | "unplaced" | "declined", number>>;
  unplaced_details?: {          // parallel to states.unplaced, one per row
    op: string;
    aggregation_id: string;
    reason: "missing_signature" | "missing_projection" | "pipeline_stage";
  }[];
};
```

The `measured` state is the count. The state lists name each key (or op)
once, however many rows share it — the thing to measure — so their lengths do
not add up to `total - measured`; `rows_by_state` does. Consumers must not
parse `states.unplaced` strings; `unplaced_details` carries the structured
reason. A zero-cost row is measured at 0 and never appears in an unplaced
list.

### 6.3 Where the fields are

| Response | Fields |
|---|---|
| `/inference/estimate` `phases.<phase>` | `t_measured_s` (null unless complete), `coverage`, and the older `measured_rows` / `total_rows` (the page falls back to these, with no states, when `coverage` is absent) |
| `/inference/estimate-sweep`, each point's `phases.<phase>` | The same after a measured fold; its SOL projection carries no breakdown row counts |
| `/inference/estimate-parallel-sweep`, each chart point | `<phase>_tps_per_gpu_measured`, `<phase>_t_measured_s` (null unless complete), `<phase>_measured_coverage` |
| `/inference/estimate` and sweep points, `phases.<phase>` | `t_partial_s`, `partial_measured_share`, `throughput.partial` (§5.5; absent until a measured fold reads) |
| `/gallery/frontier`, each model | `frontier_partial` (every point with a SOL time; each carries `partial_measured_share`), `frontier_measured` (through fully measured points only), `measured_track.state` (`reads` / `none` / `unavailable` / `engine_error`) and, when it reads, `measured_track.coverage` per context (points, complete points, best and median share, keys most needed per state) |
| Per-op rows (`per_op_timing*`, `phases.<phase>.flops_by_module`) | `t_measured_s` (`0` on zero-cost rows, null where not measured), `measured_state`, `measured_key` |

## 7. Caching: theory and measured answers

### 7.1 Three tiers

Each tier has the invalidation its data needs.

1. **Canonical law records** (`inference/sol_law.py`, schema
   `serving-law-v6`, `sol_data.walkRecordSchema`) are the only SOL artifact
   shared across surfaces. A record holds one layout's phase laws as a CSE
   formula graph over workload and card symbols (`sol_law.LawGraph`), its
   collective signatures, and compact capacity facts per card and context. It
   holds no expanded points, per-op breakdown, module table or response
   payload. The decode phase also carries the public measured grid's concrete
   `PointMeasure`s: at production time each card/context/batch enters the one
   `point_keys` owner (the same road a Kernel Bench plan uses). Physical keys,
   shape axes and term launches are record-level tables; a class's cards are
   stored once when their `PointMeasure`s differ only by the key's GPU field,
   and the reader binds the requested GPU. Per-point anomaly numerators are
   not stored: they do not select a kernel. Production uses the same
   `rows_at` road with `costs=False`, moving shapes, dtype, dispatch and
   launch geometry without re-pricing every row's SOL numerators, and passes
   `PointMeasure`s onward as objects. A pool reader looks the point up;
   it never replays a symbolic row key, and the SOL projection does not read
   the table. A Gallery entry stores only the record name and bundle key per
   point; a measured fold expands the points that view asks for and never
   keeps a second key table. The measured fold remains
   `batch_sweep.measure_points`. A
   reader compiles each CSE graph into one ordered assignment
   program in-process, keyed by the graph's law content, so a public point
   never re-parses SymPy per definition. `_capacity_preparation` itself reads
   the resident-weight ledger and returns a refused preparation when a rank
   cannot load its weights; estimate and canonical-law consumers therefore walk
   no phase rows, and no caller divides checkpoint bytes by a parallel world.
   A law's terms are the rows of one walk, and which rows exist
   depends on the card's walk class (§4), so a record belongs to one class:
   it names the card it was walked on (`walked_on`), and records live in the
   SOL pool (§8.2) under one identity per model **and walk class**
   (`sol_data.servingWalkIdentity`: result version, model, card class,
   as-shipped precision, and the producer's workload facts,
   `producerWalkFacts`), one record per layout (`sol_data.layoutRecordName`).
   A layout's record is walked on its class's walk card (§4.3); its capacity
   facts have one producer, `sol_law.servingLawCapacity`: every card of the
   class is restamped there, and a speculative record receives those facts
   once (`publicCapacity`). A card enters a law only through
   `sol_law.cardBinding`, which asks `Card.refuse_unless_one_class`, and a
   point is read off the record by `sol_law.servingLawPoint`. A card of
   another class names a different bundle and reads that one.
   The bundle's layout axis is the union every public consumer can ask for
   (`layoutsByCard`); `galleryLayoutsByCard` names the Gallery's frontier
   subset, so widening a bundle for Scaling never adds Gallery points.
2. **SOL answers** are projections of those laws, or of freshly walked cells,
   for one request. They are cached in the request cache
   (`api/sweep_runtime._SWEEP`: memory → disk → remote tier) under content
   without measurement scope, selection mode or posted rows, so every user and
   mode reads the same SOL answer.
3. **Measured answers of a Scaling sweep are never cached.** A measured
   request computes the measures it needs, makes the one lookup and folds. A
   cache keyed on the lake's snapshot would go stale with every Kernel Bench
   write, including writes no point reads, while measurements land
   continuously. (The Gallery frontier keeps its folded measured rates per
   lake snapshot beside its SOL drawing, because one frontier request reads
   the lake once for every model.)

### 7.2 Identity

Every cache key comes from `measure.scope.ResultIdentity` and `make_key`; no
cache builds a key another way (`tests/meta/test_result_identity.py`).

- Measurement-only request fields are marked on the request model with
  `_models.measurement_only` (`prefer_measured`, `selection_mode`,
  `kernel_mix`, `calibrate_with_mine`, `calibration_rows`, `sol_only`);
  `scope.measurement_fields` reads the marks.
- `scope.theory_of_request` is a request's theory identity: its content
  without those fields, under no scope. `sweep_runtime.sweep_theory_identity`
  applies it, and the SOL answer is keyed as kind `<kind>-sol`.
- The parallel sweep also drops the fields each point sets for itself
  (`parallel_sweep.PER_POINT_FIELDS`: `parallel`, `gpu_sku`, `num_gpus`,
  `avg_input_tokens`).
- Every axis that generates points stays in the key: the card series, the
  context grid, the batch grid, the modes.
- The version part is the model's `result_version` plus the payload schema
  (`_ESTIMATE_SWEEP_PAYLOAD`, `_PARALLEL_SWEEP_PAYLOAD`), so a code or data
  edit lands on a fresh key and no invalidation pass runs. A model with no
  resolved spec is computed uncached.

### 7.3 Serving a sweep

`sweep_runtime._run_sweep` is the one place a Scaling sweep is keyed,
computed and stored, so the warm job (§8.3) fills the same key the endpoint
reads:

1. **start** — key, cache read (a SOL answer only), plan, and the cells
   canonical law records already answer (`_sharedWalkResults`);
2. **cells** — every other cell as a compute job (`offload.map_off_loop`),
   each keeping its steps as it goes (§7.4);
3. **finish** — `assemble` (the one lookup and the fold, on a private copy,
   for a measured answer; none for a SOL answer), then store the SOL answer.

A cell is projected from a canonical record only when `PublicServingLawRequest`
says the record owns the question: the producer's `ServingWalkFacts`, the
request's card class and layout, and contexts on the serving grid
(`sol_data.servingContexts`), with no measurement scope. Floors, decode and
prefill batches, cached prefix and verify width are law inputs, not cache
axes. A different HBM budget changes `ServingWalkFacts` and walks its own
record; a non-canonical layout or measured request runs its own cell. The
direct estimate and the StepLaw projection leave through the one serving point
view (`serving_view.servingLawPointView`). A spec refined while the cells ran
stores and serves nothing (`ensure_spec_generation`).

### 7.4 Resumable steps

A cell's steps (a Context × Batch context, a layout's cards) are kept as each
finishes in a local step cache (`_Partials`, `_SWEEP_STEP`), keyed on the
sweep's theory identity, the cell and the step. A sweep cancelled because no
request waits for it any more leaves those steps, and the next request for the
same sweep resumes rather than starts over. Steps are kept only when the mode
looks measurements up (so the steps carry their rows' keys) and never enter
the shared remote tier.

### 7.5 Two answers per Scaling body

A Scaling sweep body has two answers, and they are two requests
(`SweepRequest.sol_only`):

- its **SOL answer** reads no measurement and is a function of the body
  alone. The panel asks for it first and draws it. It is cached on the server
  (§7.1) and in the tab (`web/src/sweepCache.ts`, memory only, keyed on path
  and body);
- its **measured answer**, asked only while Measured is ticked and the mode
  reads measurements, is what Kernel Bench holds now. It is asked of the
  server every time and never held in the tab, and it is drawn over the SOL
  curves. A slow or failed measured answer leaves the SOL curves in place.

Plan and Compare ask for measured answers directly. One cache implementation
owns every request entry and in-flight join (`measure/cache.KeyedCache`); no
endpoint hand-rolls a dict and a lock.

## 8. Bake and warm at deploy

### 8.1 Image build

The image build runs where no Kernel Bench service does, so it bakes only
theory that never reads the lake (`Dockerfile`):
`python -m backend.warm_caches --gallery --sweeps` with `HA_SOL_POOL_ONLY=1`.

- The Gallery warm walks every Gallery model's layouts once per public walk
  class into that class's canonical law bundle and stores the Gallery's SOL
  drawings.
- The Scaling warm first ensures a bundle for any card class the Gallery does
  not publish (`warm.ensureSharedWalks`), then reuses every complete bundle.
  Under `HA_SOL_POOL_ONLY=1` it writes no request-shaped Scaling answers: a
  Scaling answer is request-shaped and belongs to the runtime cache.

`--build-arg HA_SKIP_WARM=1` skips the build-time warm; the first request
then pays for the walk.

### 8.2 The SOL pool on disk

`inference/sol_pool.SolPool` stores each identity under `HA_SOL_POOL_DIR` as a
`manifest.json`, an `index.json` and `records-*.jsonl` shards. The index names
each record's shard, byte range and digest, so a reader seeks one record (one
layout) rather than parsing a bundle; the manifest carries the bundle's plan
(cards, layouts per card, contexts, producer facts, record schema), which
`sol_data.walkManifestIsValid` checks before any record is read. Law records
are plain JSON; the published archive is a gzip-compressed tar, so repeated
key-table indices compress together without a custom codec. Formulas serialize
as law text read back through `law_text`; concrete lookup keys serialize as
plain JSON. Neither is pickle nor generated program source.

In production the pool lives on the persistent volume. Each engine version has
a small Storage index (`serving-law-v6/<engine>/index.json`); it names one
immutable, digest-addressed archive. `backend.sol_pool_artifact` reads that
index at container start, downloads the archive it names and verifies its
SHA-256. An absent index is not an error (the runtime warm builds the pool),
but an index that names an archive which cannot be downloaded or verified is.
`scripts/publish_sol_pool.py` is the publisher: it writes the archive and
digest first, then moves the index; no deployment configuration names a
timestamped pool by hand.

The Gallery response cache is a different, mergeable layer. It lives under
`HA_GALLERY_CACHE_DIR` on the same persistent volume, and
`backend.gallery_cache_artifact` reads `gallery-cache-v2/<engine>/index.json`
(v2: frontier entries are stored gzip-compressed, `frontier/*.json.gz`).
Its archive contains deterministic model facts and frontier entries only:
measured views are folded live and are deliberately excluded.
`scripts.publish_gallery_cache.publish` is that layer's publisher. A new
deterministic artifact clears stale machine-local measured views before
merging its entries; runtime warm can then add more entries beside them.

The Scaling sweep's finished answers are the same kind of layer
(`HA_SWEEP_CACHE_DIR`, `backend.sweep_cache_artifact`,
`sweep-cache-v1/<engine>/index.json`); the unfinished `steps/` beside them
never ship. Both result-cache artifacts carry the code traces their entries
name, installed into `HA_CODE_TRACE_DIR` on the volume: an entry is served
only while its trace can be read. Publishing uses Supabase's resumable
upload for every object, through the environment's proxy only when one is
configured.

Gallery and Scaling project a serving-law record in-process. The compiled CSE
graph is memoized by its law content for the process's lifetime; there is no
durable projection cache or remote tier. Only the serving-law pool above is
bootstrapped as an immutable artifact.

### 8.3 Runtime warm

`scripts/entrypoint.sh` starts the Kernel Bench service and the API, and once
the lake serves runs the landing warm, then the full
`python -m backend.warm_caches`, at low priority in its own process
(`HA_RUNTIME_WARM=off` skips both). Fly bounds an HTTP health-check grace at
one minute, so the API opens first; the landing warm still runs before the
full grid and folds the current measurement overlay onto the shared H100
entries. The full warm materializes the public
Scaling request matrix (`warm.scaling_jobs`: featured models × contexts, the
parallel sweep over all modes, and Context × Batch over cards × modes ×
degrees) from the canonical laws, and folds the Gallery's measured tracks
under the lake's real scope. The server process never warms itself: a warm
inside it would compete with user requests for the GIL and the sweep pool.

## 9. Code reference

| Module | Owns |
|---|---|
| `backend/measure/auto_sweep.py` | `Cell`, `walk_cells`, the law compiler, `PointMeasure`, `MeasuredFormula`, `Folded`, `MeasuredTrack`, coverage states, `sweep_keys` / `look_up` |
| `backend/measure/lookup.py` | Key wire form and the Kernel Bench lookup client |
| `backend/measure/scope.py` | `MeasurementScope`, `ResultIdentity`, `theory_of_request`, `measurement_fields` |
| `backend/measure/cache.py` | `KeyedCache` and its backends, single-flight, remote tier |
| `backend/measure/pool.py` | The persistent process pool (`submit`, `map_cells`) |
| `backend/inference/capacity.py` | `CapacitySweepPlan`, `_sweep_cell`, `_point_rows`, `_restamp_gpu`, `_stamp_gpu_on_result`, `PendingEstimate` (`priced_on`, `keyed_at`, `measures`), `finish_estimates` |
| `backend/inference/card.py` | `Card`, its class and price views, `refuse_unless_one_class` |
| `backend/inference/parallel_sweep.py` | `ParallelSweepPlan`, `LayoutCell`, `_estimate_layout`, `class_cards`, `PER_POINT_FIELDS`, the decode ladder and frontier views |
| `backend/inference/batch_sweep.py` | `rows_at`, `phase_at`, `point_measures`, `step_measure`, `measured_selector`, `measure_points`, `apply_measured`, `measures_on_card` |
| `backend/inference/calibration_overlay.py` | `point_keys`, `PointRow`, `zero_cost`, `looks_up` |
| `backend/inference/row_roofline.py` | Row laws, `LawBook`, `StepSum` (SOL and measured laws), `FloorForm` / `CardForm` |
| `backend/inference/sol_law.py`, `sol_data.py`, `sol_pool.py` | Canonical law records, their identity, the seekable pool |
| `backend/inference/point_cloud.py` | The one producer of decode point-cloud records |
| `backend/inference/sweep_grid.py` | The product grids (contexts, batch grids, decode ladder, card series) |
| `backend/api/sweep_runtime.py` | Sweep keying, canonical-law projection, step resumption, compute and store |
| `backend/api/gallery_frontier.py` | The Gallery decode frontier: law bundles, class walks, measured rates |
| `backend/api/warm.py`, `backend/warm_caches.py` | The out-of-process cache warm |
| `backend/sol_pool_artifact.py`, `gallery_cache_artifact.py`, `release_artifacts.py` | Indexed, digest-verified pool and Gallery-cache bootstrap |
| `scripts/publish_sol_pool.py`, `scripts/publish_gallery_cache.py` | Immutable release-artifact publishers and their engine-version indexes |
| `web/src/measured.ts` | The frontend's reading of the measured track (§6) |
| `web/src/sweepCache.ts` | The tab's SOL-answer cache and the two-answer rule (§7.5) |
| `tests/sweep_parity.py` | The parity harness and the Kernel Bench stand-in (§3.5) |
