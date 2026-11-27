# Point rows: one derivation, evaluated at many points

**Status: implemented (Stage T).** The engine code is on this branch;
Stage V is not built and its numbers below are the measured bounds it
must justify itself against. Two implementation-time findings are folded
into §4: the traced skeleton of §4.2 was not needed — memoising the
projection by its own arguments achieves the same fold with less
machinery; the template engages only for rows the walk identifies (uid
and aggregation id), which a contributor stub without either is the case
that caught (§4.4); and the walked view of §4.1 was **not** built — the
structure half is a hand-listed `_RowTemplate` whose field set the
oracle holds, the boundary being the capture's own reads with no
allowlist object. §4 below is left as reviewed and marked where the
implementation parts from it; the engine code and
`tests/backend/inference/test_point_row_template.py` are what is true. This file
is the design to review; it is not a second truth holder, and by the
lesson of `docs/walk_identity.md` (branch `design/walk-identity`) it
comes to `main` only with the decision to implement, at which point the
rule it states is carried by the engine code and this file shrinks to a
pointer.

R1 was reviewed by two parties (archived under
`/scratch1/ha_reviews_archive/point_row_templates/`): the technical
review did not sign, with five blockers; the design review signed with
three required fixes. This revision folds both; the dispositions are
recorded with the archives. What the rounds changed: the exception row's
refusal mechanism was re-measured and re-attributed (it is a
value-conditional projection refusal, not `NoShapeLaw`; §4.3); the
Stage T effect was re-attributed from a guess to a measured bucket
composition (§5, §8); the walked view's boundary was restated as
guarding decision inputs rather than all reads, and its drift table
gained the fail-open row it was missing (§4.1); the template cache key
gained the deployment (§4.4); Stage V's exactness gate was hardened from
"oracle + ε" to admitted domains with bounds or exact shadow, and its
folding reference was corrected to a function that exists (§5).

The one sentence: **a per-op row's PointRow is one derivation's structure
evaluated at a point — today the structure is re-derived at every point,
and two independent measurements say the structure does not change.**

## 1. What is slow, measured

The warm's dominant loop is `_measurePointTables`
(`backend/inference/sol_law.py:814`): for a walk class it moves the
walked rows to every public point — context × draft-k × decode batch,
about 160 points on a model with MTP — and at each point derives every
row's `PointRow` through `capacity._point_rows` → `batch_sweep.rows_at` →
`calibration_overlay.point_keys` → `point_row_of`.

One layout job of Qwen3.8-Flash-Next (MTP, the kind of model the warm's
time is made of), 87.3 s total (`/scratch1/ha_perf/split3.log`):

| part | time | share | collapses per… |
|---|---|---|---|
| measure keys, per point | 37.7 s | 43.2% | nothing today |
| walk: rewrite + fusion + rows | 31.9 s | 36.5% | walk (Stage A/B's axis) |
| capacity, per layout × card | 8.9 s | 10.2% | layout |
| walk: law CSE | 0.9 s | 1.0% | walk |
| unattributed | 8.0 s | 9.1% | — |

The walk-identity line (branch `design/walk-identity`, withdrawn from
`main`) attacks the 38% that collapses per walk; what its archive
actually supports is a card-axis count ratio of 1.5× (216→144 walks on
its probe models), and its layout-axis estimate — after that line's own
unit correction — was ~1.5× in time. This design attacks the **43% that
no walk collapsing can reach**, because it is per point by nature. (The
"1.29×" figure once quoted for the card axis was a session-time
computation whose artifact was not archived; it is not relied on here.)

Inside that 43%, the redundancy is structural. The probe
(`/scratch1/ha_perf/tpl.log`, methodology `/tmp/template.py`) watched
`point_row_of` across the job: 399 rows, of which op, signature,
`dtype_env_json` and `launches` change across points in **0 / 399** each,
and the axis-name set of `shape_json` in **1 / 399**. The total call
count, 399 × ~160 = **63,836**, was independently reproduced by the
design review's own probe (38,714 calls returning a keyed row + 25,122
returning `shape_json=None`); the technical review was the party that
challenged the number for having no named artifact, which the
reproduction then answered. The probe's 15,500 is the number of *per-row*
distinct `shape_json` values summed over rows — a lower bound on
law-evaluation work, not a global dedup count; the global union is
unmeasured and is Stage V's to measure (§5).

The one exception is `qwen_hc_full_block | …index_k_pool.pooled_out__…`
(DSA's KV pool): at 133 of its 443 calls it keys with seven axes (`B,
compress_ratio, d_indexer_head, d_rope, kv_block_size, s_query,
s_slots`); at the other 310 it returns `shape_json=None`,
`placement_reason="missing_projection"`. One reviewer measured
`NoShapeLaw` raised **zero** times; the other's probe ran the same
job to completion without a raise. What flips is a value condition
inside the derivation (§4.3), not the walk-level `shape_at_batch`.

## 2. Why the obvious fixes do not work

**Memoising `point_row_of` by its inputs** was tried in the prior
session and measured at a ~21% hit rate with hashing overhead larger
than the savings (probe preserved as `/tmp/duprow.py`; its output was not
archived, so treat the number as session testimony, not an artifact):
the moved row's inputs (exact shapes, `node_dims`, point coordinates)
differ at every point even where the answer's structure does not. A memo
demands equal *inputs*; what repeats is the *computation's shape*.

**Collapsing points into walks** does not reach this cost: the per-point
work is the point's by definition — a Kernel Bench key names a concrete
shape, so each point must produce its own values. What it does not need
to produce, per point, is its own re-derivation of which axes there are.

**A sentinel for "no shape"** is rejected on semantics, not speed: `0`
is a real value for this very row (⌊s / compress_ratio⌋ pooled rows is
genuinely 0 before a pool closes — and it is exactly that zero which
refuses the projection today), `shape_json` is a join field a fabricated
key can phantom-match, and a refusal is a designed boundary, not a
missing value. Absence stays absence.

## 3. The rule

> `point_row_of` factors into a **structure half** that decides from
> what a point cannot change, and a **values half** that reads the
> columns a point moves. The structure half is derived once per row; the
> values half runs at every point. The composition of the two **is**
> `point_row_of` — there is no second derivation beside it.

This is a refactor of the one derivation into two phases, not an
accelerator bolted next to it. Consequences that follow from the wording:

* Every consumer of `point_keys` (the Kernel Bench plan, the estimate's
  overlay, the serving-law producer, the sweep points) keeps calling the
  same function and gets the same answer; the win comes from the
  structure half running once, not from anyone switching roads.
* A new `PointRow` field defaults to the **values half** (per-point). A
  field joins the structure half only when §4.1's boundary says it may.
* Refusals live in the values half — both kinds: the walk-level
  `shape_at_batch` / `NoShapeLaw`, and the value-conditional projection
  refusals that decide inside the structure half today (§4.3).

## 4. The mechanism

### 4.1 The boundary guards the structure half's decision inputs

The hazard this design must not create is a hand-maintained list saying
which field is point-invariant — that list is a second truth holder that
drifts the first time someone adds a field. The question is only where
such a list fails. A denylist at the producer ("these are the columns
`rows_at` moves") fails **open**: forget an entry and the structure half
captures a per-point column as a literal — a wrong key, in silence. So
the list lives at the consumer, as an **allowlist the reader cannot go
around**: the structure half's *decisions* are computed from a **walked
view** — an object exposing exactly the walked facts those decisions are
granted (`aggregation_id`, the dtype facts, the row's law objects,
`count_per_step`, …).

One correction R1 needed, from both reviewers: the view cannot be the
structure half's *only* input, because today's structure path also
**reads** moved columns without deciding from them — `_keyed` reads
`shape_in`, `_identity` reads `node_dims`, `CommLaunch.from_row` reads
the comm columns (`benchmark_identity.py:1827/:1932/:1945`). Those reads
are exactly the values the values half reproduces per point. So the
implementation shape is: run the full derivation once at an in-domain
anchor point, **instrumented so each output value traces back to the one
law that produced it**; what traces to a law is a slot, what traces only
to walked facts is a literal, and the view is the boundary the
**decision** computation (which operand form, which axes exist, which
signature) may consult. The view's entries are of two kinds — pure
walked columns, and walked *structural facts* the decisions read (an
operand's rank, the Dim identities that name the axes).

The drift table, now with the row R1 was missing:

| change | effect |
|---|---|
| a new walked column, not yet in the view | `KeyError` at template build — the row falls to today's path (silent in production, red in the oracle; §6 makes the oracle a usage gate, so it is not quietly swallowed) |
| a new moved column (`rows_at` rewrites it) | not in the view, so no *decision* may read it; if it feeds an output, it enters as a slot |
| a walked column **already in the view becomes moved** | the template keeps using the anchor value — **fail-open**, the one row the construction cannot close; the full-grid oracle is the net that must catch it |
| a moved column wrongly added to the view | review-visible, oracle-red |

The third row is why §6's oracle is a load-bearing wall, not a
formality. No producer is policed, and `batch_sweep` changes not at all.
The shape is the one `RowIdentity` itself set when the same hazard
appeared there: three producers passing arguments that could disagree
were not checked at their call sites — their inputs were bound into one
object the derivation reads.

### 4.2 Slots are the row's own laws; the skeleton is traced, not parsed

A slot's per-point value comes from the same law objects the row already
carries — `point_law` / `shape_law`, in the closed law-text vocabulary —
read through the same batched evaluation `rows_at` already uses. The
skeleton (axis-name sequence, operand-form choice, divisor structure) is
**captured by the anchor run's own trace**, at the one place the
derivation already computes it (`RowIdentity._keyed` and its
projectors): each output value is recorded with the law that produced
it, so the mapping slot → law is a fact of the trace, not something
recovered by parsing `shape_json`. Where a value cannot be traced to
exactly one law, the row keeps today's path. Fail closed, per row.

The boundary this draws in code: `calibration_overlay.py` owns the
composition and the view; the anchor instrumentation wraps the
projectors that already live in `benchmark_identity.py` — "one file
only" was R1's overstatement.

### 4.3 The exception is a value-conditional refusal, and it is a class

R1 attributed the pooled row's off-domain points to
`shape_at_batch: False` / `NoShapeLaw`. That was wrong, and both
reviewers measured it: `NoShapeLaw` is raised zero times in 63,836
calls; `shape_at_batch` is a walk-level constant (a row's `shape_law`
exists or does not, `phases.py:1883/:2019`), so it cannot flip across
points for one row. What actually flips is a **value condition inside
the structure derivation**: `s_slots = ⌊s_ctx / compress_ratio⌋` is 0 at
small contexts, and `_contract_identity` refuses a non-positive extent
to `None` (`benchmark_identity.py:560`) — 133 keyed calls against 310
`missing_projection` returns on the probe row.

This is not one row's curiosity; it is a class of refusals decided by
slot values: `_contract_identity`'s extent ≤ 0 and cross-operand
consistency (`:496`), `_rope_identity`'s divisibility (`:2053`),
`_conv2d_identity`'s square check (`:1989`). The rule, stated once:

> **A value-conditional refusal is upgraded to a per-point predicate
> slot. A refusal that cannot be so upgraded takes the whole row back to
> today's path. The oracle is the gate.**

Without the upgrade, a template would mint keys at points where today
refuses — the mirror image of a silent miss. With it, the pooled row
costs nothing: its axis names are constant on the in-domain points, and
off-domain points produce no key. Absence, not a sentinel (§2). The
walk-level `NoShapeLaw` class keeps its original treatment: its
condition reads a moved column, so it is a per-point slot by §4.1's
boundary. And the anchor of §4.4 must be an **in-domain** point — the
first call a row sees may be off-domain (310 of 443 for this row), so a
row that falls back at its first point is retried at its first in-domain
point rather than sticky.

### 4.4 Where the template lives, and what keys it

*As built:* the cache key is `(RowIdentity, served, deployment key)`,
where the deployment key is the six request fields **and the binding
itself** — a frozen dataclass, keyed by structure — because a binding's
`when:` reads its own deployment (server flags, activation dtype,
device) that no request field names; two requests differing only there
must not replay one another's dispatch. `key`/dispatch assembly lives in
the values half; the narrow `dispatch_for` belongs to the structure
half. The "first in-domain anchor" rule below is moot as built: with no
skeleton to capture, a template captured at an off-domain point is as
valid as any other (its literals read no moved value), and the
projector's per-point arguments carry the refusals.

Hanging off `RowIdentity` (`benchmark_identity.py:1447`), which is
already resolved once per analysis and cached at
`cell.identities[served]` — with one addition R1 missed: `point_row_of`'s
key and dispatch read the **deployment** (phase, parallel signature,
framework, `gpu_sku`), so the template cache key is
`(RowIdentity, served, deployment identity)`, and the key/dispatch
assembly itself lives in the values half — nothing deployment-dependent
is ever replayed from a template. `dispatch` in the narrow sense
(`dispatch_for`) decides from op, deployment, program, analysis and
`resolved_launch` — no moved column among them — so it belongs to the
structure half outright, not by measurement. The template is built
lazily at a row's first in-domain `point_row_of`; `point_keys`'s
signature does not change, and `_measurePointTables` needs no change at
all for Stage T.

## 5. Stages

**Stage T — factor the derivation (§3, §4).** Code: `calibration_overlay.py`
owns the composition and the view; the anchor instrumentation wraps the
`benchmark_identity.py` projectors (§4.2). First the walked view and the
oracle land (`tests/backend/inference/test_point_row_template.py`), then
`point_row_of` becomes the composition, so the refactor starts
green-on-red rather than red-on-faith.

Its effect, from the bucket composition the design review measured off
the same job's py-spy samples (`/scratch1/ha_perf/one_q38.txt`) — a
single reviewer's classification of those samples, and the structure
share is sensitive to how the ~25% unclassified frames are assigned (the
technical review's coarser independent split put it nearer ~37% than the
~51% below; the law-evaluation share agreed at ~22–23%, and the
conclusions drawn from the table hold under either split):

| inside the 43.2% bucket | share | Stage T |
|---|---|---|
| structure re-derivation (`of_event` / `_keyed` / `_identity`) | ~51% | **removed** |
| `key_columns` / `laws_by_point` (the values' own evaluation) | ~22% | stays |
| `rows_at` moving (incl. its law reads) | ~6% | stays |
| per-point assembly (`PointRow`/`BenchmarkKey` construction, `_at_geometry`, `params_from_row`) | ~6–8% | stays |
| `dispatch_for` / `leaf_dtype_env` / `zero_cost` / binding resolution | ~3% | removed (once per row/layout) |
| `point_measures` and the rest | ~4% | stays |

A fourth measurement — decomposing `of_event` itself before
implementing — found the flaw in all three earlier splits: `of_event`
is not all structure. Its single largest part is the projection
(`identity_shape_from_shape_in`, 4.66 s of 10.5 s), a pure function of
the point's values; what T can remove is the resolution around it. T as
built and measured takes the probe job from 77.6 s to **68.4 s
(1.13×)** — with the review fixes in, whose first cost (hashing the
binding now in the template key, once per lookup) was measured away
again by caching the binding's hash — the bucket from 35.1 s to ~26 s: the invariant decisions
(the op, signature, ports, dtype environment, dispatch, launch count)
replay from a template, the projection runs once per distinct identity —
6,124 of them over 38,714 keyed calls — by a memo **keyed on the
projection's own arguments** (a hashable-tuple key cost more than it
saved, ten times over; the repr of the fixed-order argument tuple does
not), and what is genuinely per point (shapes, routed rows, cadence,
launch rows, `key_columns`' evaluation at 27% of the residual bucket)
stays. No flag: an identity refactor ships green or not at all, and the
oracle (`tests/backend/inference/test_point_row_template.py`) holds the
two roads equal call by call.

**Stage V — evaluate the slots over the grid, vectorized.** The grid is
rectangular (context × k × batch; the served mask applied after). Two
things Stage V attacks that Stage T leaves: the ~22% of law evaluation
and the dedup across rows — the probe's 15,500 per-row-distinct values
are a lower bound; the global union is what V measures and folds, with
keys built once per distinct (template, value) — the folding
`MeasureMatrix.of_record` already does for a record's cards
(`backend/measure/measure_matrix.py:169`). Distinct slot laws are
compiled once each with `compile_law_vector` (the numpy printer the SOL
projection's whole-grid path already uses) and evaluated over the grid's
columns. **Assembly preserves row order**: every consumer pairs rows and
PointRows by strict positional zip (`calibration_overlay.py:238`,
`capacity.py:3792/:3817`, `batch_sweep.py:832`, `plan.py:765`), and
`StepSum` names terms by that order. Key objects themselves are safe to
share — every consumer treats them as values.

Stage V is where exactness needs its own gate, hardened from R1:

* `compile_law_vector` must refuse a stray free symbol the way the exact
  road does (`LawBook.compiled` raises; the vector compiler today does
  not check), and callers suppress the divide warnings `np.where`'s
  two-armed evaluation produces, with nan/inf handled in the comparison.
* A law joins the vector road only with an **admitted integer domain and
  a rounding bound** for that domain, or it runs **under exact shadow**
  at release; the flag (`HA_MEASURE_VECTOR`) retires only when the
  shadow has shown zero divergence across the corpus. A grid oracle
  alone covers only what it ran. The motivating counterexample is one
  line: `floor((1/49) * 49) == 0`.

Stage V changes `_measurePointTables` — the only caller that holds the
whole grid. Other `point_keys` callers keep Stage T's per-point road and
are correct by it.

**Out of scope, deliberately:** walk identity, `LayoutPropagation`, what
a walk is, and the `n_verify` / `walked_as_number` correctness hole —
that one is a wrong number, not a slow one, and is fixed on its own. The
Kimi-K3 probe `ValueError` is fixed as part of §6's coverage, not as
engine work.

## 6. The oracle

1. **Full-record equality, not a field subset.** (*As built* the oracle
   is the load-bearing wall — the structural boundary of §4.1 was
   replaced by the capture's hand-listed fields, so this diff is what a
   drifted field meets.) `PointRow` is a frozen
   dataclass: template output vs today's whole-derivation path, diffed
   as whole records across the full grid — `key`, `op`, `signature`,
   `count`, `cadence`, `declined`, `operand_ports`, `kernel_params`,
   `leaf`, `routed`, `carries_state`, `dispatch`, `launch_rows*`,
   `shape_weight`, `geometry_*`, the refusals — every field with a
   consumer (R1's short list was a drift channel). A mismatched row is
   pinned to today's path *and* the test is red.
2. **Order is asserted**: the paired rows↔PointRows sequences are
   compared element-wise, strict.
3. **Usage gate** (*as built*): every `point_row_of` call on the covered
   corpus either replayed a template or captured one —
   `captures + hits == calls`, exactly — and no row was captured twice
   (a recapture is a lookup that missed a key the store holds: impossible
   on the production road, held at zero as a tripwire). The §4.1
   `KeyError` fallback this gate once described was part of the walked
   view, which was not built; unindentified rows (the contributor-stub
   case) are the designed exclusion, and they take the whole road per
   point by construction, not by count.
4. **Mutation checks** (*as built*: an oracle that cannot be made to
   scream is a report): corrupt a template literal (`dtype_env`), corrupt
   a structure decision (`op`), and hold the memo key's composition —
   that an argument dropped from it is a collision by construction. The
   six-class list this item once promised belonged to the walked view,
   which was not built; the deployment class is held by
   `test_a_template_is_not_replayed_across_deployments`, and a refusal
   raised while keying names its node by test.
5. **Corpus coverage** (*as built*): Qwen3.8-Flash-Next (the exception
   row), DeepSeek-V4.1-Flash and V3, Llama-3.2-3B, Qwen3-30B-A3B.
   The looped-stack family is covered by DeepSeek-V4.1-Flash (no GLM
   looped model is checked out); Kimi-K3 (KDA) is capacity-refused on
   this walk class at every Gallery preset — 1.4 TB of weights — so the
   warm never asks its rows and neither does the oracle. Recorded in the
   test, with the refusals. One framework's binding per
   walk, as the Gallery warms — the binding is in the template key, and
   `test_a_template_is_not_replayed_across_deployments` holds that key
   apart where the flags differ. Non-first cases are `regression`-marked
   by the repo's rule; the trim is deliberate and recorded in the test.

The harness lives in `tests/backend/inference/`, mirroring the source
tree — not in `tools/`, so it cannot become a second implementation
someone runs instead of the engine's own check.

## 7. What the design must not do

* **No second derivation.** `point_row_of` stays the single place a row
  becomes "what this looks up / measures"; the template is its own
  traced output at one point, replayed. The boundary is one allowlist
  owned by the reader (§4.1) — not a producer-side denylist, and not a
  fact re-derived by instrumentation.
* **No new vocabulary.** Slots are the row's existing laws in the
  existing closed law text; the grid compiler is the existing
  `compile_law_vector` on the same expressions.
* **No sentinel semantics.** Off-domain is absence; both refusal classes
  survive as refusals (§4.3).
* **No dropped refusal for speed.** Value-conditional refusals upgrade
  to predicate slots or the row falls back; the oracle diffs them like
  values.
* **No caller switches roads.** Plan, estimate, serving-law and sweep
  callers keep `point_keys`; only `_measurePointTables` gains a grid in
  Stage V.

## 8. Expected effect, honestly

Stage T, built and reviewed: layout job 77.6 s → 68.4 s (**1.13×**
measured, same job, cache-warm, machine otherwise idle, every review
fix in), with the law-evaluation ~27% of the residual bucket explicitly
*not* claimed for T — it is Stage V's to attack. Stage V's measured marginal on top of T: folding
`key_columns`' evaluations 3× and key construction 6.3× targets roughly
another 10% of the job, putting **T+V at ~1.4×** on this job — below
the 1.5–1.7× R1 hoped, because the global union (6,124 / 5,174) is
larger than the sharing the per-row number suggested. The decision that
follows: build T first, re-measure the residual, and let V's grid
evaluator justify itself against what remains rather than against the
pre-T bucket. Both stack with the walk line if
it ever lands. The numbers with artifacts behind them: the 87.3 s split
(`split3.log`), the 0/399 invariances and 15,500-per-row-sum
(`tpl.log`), the 63,836 reproduction (both reviews), the bucket
composition (`one_q38.txt`). The numbers without: the 21% memo rate,
and any Stage V speedup — V measures before it claims.

The failure mode this design must survive is the silent one — a Kernel
Bench lookup that misses forever because a key came out differently, or
a key minted where today refuses. That is why every gate above diffs
whole records, treats a mismatch as a refusal to ship, and keeps the
one fail-open drift row under the oracle's eye.
