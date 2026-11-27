# The sweep's cells go through the card class

**Status: implemented, pending dual review.** The code and its oracle are
the design below. It rides the engine version this branch already broke:
one re-warm amortizes it with the all-model sweeps, the walk-family
retention and the point-rows line.

R1 was reviewed by two parties (archived under
`/scratch1/ha_reviews_archive/point_row_templates/`): the design review
signed with three required fixes, the technical review did not sign,
with three blockers — both parties' probes and this revision's changes
are folded in. What the rounds changed: the class count and the
consequent fold arithmetic (5 classes, not ~7 — the fold is 3×, not
~2×); the `max` sentinel's ownership (three declarations, one new
resolver to own the rung rule); A's exceptions (comm/fabric is priced
by contract, `pipeline_time` at pp>1 is a priced sum); the anchor
interlock (A's class-shared overlay anchors at the rung, so C lands
first); and D, redrawn: the parallel cloud is the whole decode ladder
to degree 256, not the max slice, so the projection expands what the
batch warm covers instead of assuming it.

The one sentence: **the sweep's estimate machinery runs once per card
CLASS per point, with each SKU's numbers substituted in — the Gallery's
class discipline, applied to the road that prices instead of the road
that keys; the `max` batch point resolves onto the grid's own
power-of-two rung through one named resolver, so a class's SKUs share
every point there is; and the parallel sweep, which was the pool's
parallelisation vehicle and never a second datum, retires as a computation:
its cells project off the one class walk.**

Implementation note: the resolver is `capacity.max_batch_rung`; A's
same-class re-pricing reads the cached walk through `batch_sweep.rows_at`
and `pipeline_step_at` (the same roads moved points and pipeline phases
take), and a `max` rung that coincides with a `bN` is one computed cell and
one cloud point, with `max` as its display alias.

The class cache's contract is the one the code now states: one walk per
`(card class, phase, model, context, config, request knobs)`. The context
names the walk (a context's step is the question the anchor was taken at);
neither the SKU nor the batch does. `rows_at` substitutes the SKU's peaks and
fabric and the point's batch, and the pipeline/module layers are read through
the same laws. A narrower sibling does not inherit the walking SKU's numbers,
and a later rung does not inherit the anchor's shapes.

## 1. What is expensive, measured

The sweep's GPU axis: fifteen SKUs are priced, and they fall on **five
card classes** (`CardClass` — dispatch facts and peak families:
A100-40G/80G × SXM/PCIe/A800; B100/B200/GB200; B300/GB300; RTX-4090;
H100/H100-PCIe/H200/H20). The walk already folds (the warm's H20, H200
and GB200 lines all say `0 models to walk`); the answer cannot fold
(numbers differ per SKU — HBM and bandwidth differ inside a class); the
machinery between them does not fold today, because the card is baked
into the estimate's overlay walk: `overlay_walk`'s analyze kwargs carry
the card's peaks and fabric (`capacity.py:3426-3458`), so the row
materialization — card-independent structure, laws and shapes — is
re-done per SKU.

The measured batch grids (one model, `dp_ep@8`, the routine contexts —
the resolved maxes are the non-power values; reproduced by the
technical review's probe):

```
H100-SXM        2   5  20  40  80 156 295 532
H20             3   6  25  50  99 192 363 655      <- same class
H200            4   9  38  76 151 294 555 1000    <- same class
```

Floored to `2^⌊log₂⌋`, H100 and H20 become **identical**; H200's rung
set shares 6 of its 7 distinct rungs with them (the seventh, 8, serves
its smallest context alone). The powers of two are the grid's own
rungs — and the Hopper table is also the honest warning: same-class
SKUs do NOT all agree on every rung, which §2.B's class-wide anchor
must state, not assume.

## 2. The rule

> An estimate's machinery — rows, laws, shapes, activation liveness —
> is a property of the walk class. A SKU enters its numbers into the
> pricing of each shared point. The `max` point lands on the rung the
> ceiling's floor names, through one resolver.

Four parts, in landing order **C → A → B → D** (the anchor interlock:
A's shared overlay anchors at the rung, which only exists once C
resolves onto it).

**C. The `max` resolver — first, because A anchors on it.** Today
three declarations carry a `max` shape (`ROUTINE_BATCH_GRID`'s prefill
`-1` and decode `None`, `SCALING_BATCH_GRID`'s decode `None`,
`FRONTIER_BATCH_GRID`'s decode `None` — `sweep_grid.py:76-93`, `:146`),
and the request's `batch=None` is a fourth spelling; they all end at
`ceilings.max_batch`, but no one owns the rule. One named resolver —
`max_decode_batch(ceilings)` — owns it from C on: the largest
power-of-two rung at or below the ceiling, **the same rule for the
prefill cap** (ROUTINE's `-1`): one semantics, no key that means two
things. The exact ceiling still ARITHMETICALLY decides feasibility
(`max_batch_per_gpu` keeps reporting it); what moves is the point the
panels draw. Where the rung coincides with an explicit `bN` key (H100's
156 → 128, which `FRONTIER_BATCH_GRID` already has), the `max` entry is
that key's display alias — one point, deduplicated as `served_points`
already dedups by `(seq, batch)` (`capacity.py:2980-3003`).

This is a product decision, taken with the measured cost in hand: the
claimed max understates the true ceiling by a median 20% and a worst
49% of batch. Three surfaces snap together — the sweep, the Gallery
frontier, and Compare (`web/src/compare.ts` mirrors the grids;
`test_sweep_grid.py` pins them) — and `test_stored_measure_points_are_
point_keys_after_json` goes red **by design**: its expected public
points include the exact ceilings (`test_serving_measure_law.py:313`),
which become the rungs — a documented diff, updated with the change,
not an accident discovered by it.

**A. The overlay walk becomes card-free.** Measured, not assumed — the
implementer's probe (Qwen3-30B, archived at
`/scratch1/ha_perf/card_invariance/`) and the technical review's
(Llama-3.2-3B, 22 rows, 79 fields byte-identical) agree: the overlay
rows' structural fields — op, shapes, laws, ports, fusion badges, byte
numerators — are identical across same-class cards, and the
card-dependent set is exactly the priced columns:
`peak_flops_per_sec_for_op`, `t_compute_s`, `t_memory_s`, `t_op_s`,
`t_sol_s`, `bound`. Two exceptions the probes forced into the open:

* **`t_comm_s` is priced by contract.** Both probes found it equal
  across H100/H20 — that is those configs' luck (the fabric values
  those runs read happened to coincide), not the rule: the collective
  phase, fabric label and bottleneck follow `Fabric`
  (`row_roofline.py:1033-1066`), and intra/inter bandwidths differ
  within a class. A treats comm like compute and memory: the numerator
  (payload, hops) is card-free, the time is substituted.
* **`pipeline_time` at pp>1 is a priced sum.** The overlay sums
  `t_compute/t_memory/t_comm/t_sol` over stages and keeps `stage_laws`
  (`calibration_overlay.py:1263-1295`); the laws share, the sums do
  not — they are re-derived per SKU off the shared laws.

The priced columns move from materialization time to read time — the
point layer already prices per card (`rows_at` takes `peak_flops`,
`peak_bw` and the fabric as values; `point_columns` re-reads them). The
first walk of a class at a context is the template; every SKU and every
batch rung there reads it by substitution. The card-invariance probe is a
**pre-landing
gate**, not a one-off: same-class pair, same config, same anchor,
diff the whole record — row set, order, dtypes and values (§1.5's
comparison discipline), structural fields equal, priced columns
unequal — run per corpus model before A ships.

**B. The batch grid resolves per class, on the rungs — and grows to
the cloud's own ladder.** A context's cell set for a class is the
power-of-two keys up to the class's widest SKU (the union: a rung a
narrower sibling cannot serve prices and refuses on its substitution —
a real refusal, not a missing point). `SCALING_PANEL_BATCH_GRID`
returns to the full declared grid — and beyond `{b1, b16, b64, max}`:
the batch warm covers `SCALING_DECODE_LADDER`'s rungs
(1, 8, 16, 32, 64, 128, 256, 512, 1024) and degrees to
`SWEEP_MAX_GPUS` (256), because D's cloud is drawn from exactly those
(`parallel_sweep.py:602-620`, `:826-850`). The trim to two points was
a cost decision — each rung ×15 SKUs — that A dissolves (a rung is one
class-shared machinery pass); the ladder costs what two points used
to.

**D. The parallel sweep retires as a computation, not as an endpoint.** R1
drew this as the max-rung slice; the
technical review's probe measured the truth: the parallel cloud
carries **the whole decode ladder per layout** (100 points where the
max slice has 1) and degrees to 256 where the batch warm stopped at
128 — so D is an expansion before it is a retirement:

* The batch warm (B) covers the ladder and the degrees the cloud
  draws; the class machinery is what makes that affordable.
* The endpoint stays because it is the panel's response shape; its cells
  are projected from the canonical walk records (`sharedSolPool`), and the
  request-keyed entry only stores the rendered SOL answer. It is a cache of
  a view, not a second datum.
* The cloud and `by_mode.ladder` metadata are assembled from those
  projected cells; an absent record or cell falls back to the direct walk,
  the same fallback every warm-cache reader has.
* The cells' projection is shared with the Gallery below the response
  boundary (`serving_projection`): one public envelope per record, card and
  substitution axes, sliced by each surface. The rendered answer caches stay
  separate views.
* Progress and partial answers keep the endpoint's existing contract:
  cells are independent, a finished cell can be assembled into a narrowed
  plan, and the full answer is stored only after every cell and the one
  measured fold.
* The batch jobs warm the panel's Context × Batch caches first. That is a
  warm-order optimisation, not a correctness dependency of the projection:
  both halves read the same class walks and canonical laws.

It was kept as an acceleration, not a datum: one POST that submits
the whole config cloud to the pool (a submitter, not a job — its
cells occupy the pool, `warm.py:306`), so a live panel load could
compute the cloud in parallel. With the cells warm, the reason is
gone; what remains of "parallel acceleration" is the pool the batch
warm already runs in.

## 3. What it buys, arithmetically

Fifteen SKUs on five classes: the fold is **3×**, cutting about
**two-thirds** of the sweep phase's machinery (the walk, analyze and
family layers are untouched — per (model, config), shared across cards
already). The batch warm's grid grows (the ladder, degrees to 256) inside
the same fold, and a parallel request's cells project rather than walk.

## 4. What it does not touch

* The answers at the rungs (byte-identical there; the max cells move
  as C states, the cloud assembles as D states).
* The walk, the cells, the families, `WALK_CELLS` — the retention line
  already landed.
* The batch and parallel request-keyed caches — each stores a rendered
  response; neither stores a second walk or a second point datum.
* kernel_bench's `FRONTIER_TRACKS` sweeps the same grids
  (`sweep_grid.py:285-290`); C's resolver reaches them as it reaches
  every other `max` shape — one rule, all consumers, and their
  rung cells stay byte-identical under A by the same oracle.
* Stage B (degree folding) — complementary: this collapses the GPU
  axis per config, Stage B the config axis per model.

## 5. The oracle

1. **The documented diff, mechanised**: before/after on identical
   inputs. At the rungs — including kernel_bench's tracks — the cells
   are byte-identical (row set, order, dtypes and values). At the
   `max` cells: `batch`/`concurrent_requests`/`tokens_per_step`/`
   time_per_output_token_s`/the `tok_s_*` family equal the same-batch
   rung cell's machinery with this SKU's pricing; `max_batch_per_gpu`
   still equals the exact ceiling; the `max`/`bN` coincidence is
   deduplicated, not duplicated. Prefill states its own side of the same
   diff: the public phase batch moves to the prefill rung, while
   `max_prefill_batch_per_gpu` stays the exact activation-bound ceiling;
   a chunked prefill's folded phase and endpoint measures equal the
   same-batch rung's chunk schedule.
2. **The class equality that cannot hold**: same-class SKUs' payloads
   differ in the priced columns (both reviewers' probes are the
   template) — proving the comparison is not vacuous.
3. **The dedup's shape, pinned**: `results_by_batch` and the cloud
   carry ONE entry where `max` coincides with a `bN` rung — the
   display alias, not a second point — asserted beside the
   `served_points` dedup it generalises.
4. **Machinery counting**: per (class, config, context) exactly one
   overlay materialization, however many SKUs ask — through the
   account hook, covering pp>1 and multi-stage.
5. **D's response contract**: a partial parallel answer assembles only
   the cells that finished, a late cell cannot overwrite an earlier
   partial head, and the stored full answer is byte-identical to a cold
   request's after the one measured fold.
6. **A's pre-landing gate**: the card-invariance probe, per corpus
   model, before A ships — structural fields equal, priced columns
   unequal, whole-record diff.

## 6. Order of landing

**C → A → B → D.** C first because the class walk's points resolve onto
one rung rule. A is the refactor (priced columns move to read time; the
probe is its gate). B grows the batch warm's grid to the cloud's own
ladder and degrees. D makes the parallel request a projection off the
canonical walks while preserving progress and partial answers. Each step's
oracle is the documented diff; the warm that already must re-run carries
all four.
