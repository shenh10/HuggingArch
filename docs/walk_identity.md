# What identifies a walk

**Status: proposal.** Nothing below is implemented. On implementation the
rule in §2 replaces `docs/symbolic_evaluation.md` §1's sentence "the
topology is fixed" and extends its §5 cell key; the card half replaces
`CardClass`'s docstring in `backend/inference/framework_bindings/__init__.py`
and the record identity at `ServingWalkKey`'s level in
`backend/inference/sol_data.py`. This file is the design to review, not a
second truth holder.

Reviewed in four rounds. **Both reviewers have now signed**: the
architecture review (Kimi) at R1 with three required fixes, the technical
review (Codex) at R4 after three rounds of not signing, with two required
hardenings of the oracle's own checks — an empty-container sentinel, and
`--refusal` asserting against a clean baseline rather than in isolation.
Both are in. Every round is archived under `reviews/`.

What the rounds changed: the question table gained three classes it
was missing (§2); "answerable before the walk" became "off the card-free
walk", because the `dims` questions read a walk product (§2.3);
unforgeability is no longer claimed for Stage A (§2.2); `priced_on` is one
of four callers of the single statement, not the single gate (§3); the owner
moved off `ServingWalkFacts` (§6); the word "theorem" was retired in favour
of what actually carries the claim (§5); and the saving is restated as what
the sound key delivers rather than its floor (§8).

## 1. The shape today: one rule, three axes, two of them declared

A walk is taken at a binding. Every input is on one of two sides:

* **static** — baked into the walk, so it is part of *which* walk this is.
  Two bindings that differ in a static input are two walks.
* **dynamic** — deferred to a symbol, so one walk answers every value by
  substitution. This is the `Point`.

Three axes feed a walk, and the split is declared on only two of them:

| axis | who declares the split | static side | dynamic side |
|---|---|---|---|
| workload counts | `Workload.walked_as_number` / `Workload.symbols()` (`arch/eval/transforms.py`) | `s_verify == 1` | `B`, `s_input`, `s_output`, `s_prefix`, `rho` |
| card facts | `CardClass` / `CardForm` (`inference/framework_bindings/__init__.py`, `inference/row_roofline.py`) | the dispatch facts (`device`, `compute_capability`) and the peak **columns the card lists** | HBM, bandwidths, `gpus_per_node`, and the peak **values** |
| deployment (parallel degree) | — nothing — | everything | nothing |

Both declared rows have the same form, and they were written independently.
The third row is empty: a degree is wholly static by default, so every rung
of every ladder is its own walk.

Two consequences, both measured:

* **Most layouts of a model are one walk.** Hopper class, every layout the
  Gallery and Scaling ladders reach: Llama-3.2-3B, 37 layouts → **3**
  distinct per-op row structures in prefill, **2** in decode (group sizes
  18/10/9 and 19/18); Qwen3-30B-A3B, 42 layouts → **5** structures (15/12/
  7/7/1). Compared field by field, EP2·DP2 and EP4·DP4 of Qwen3-30B-A3B
  produce the same 30 rows with the same `aggregation_id` sequence, the same
  ops, dtypes and fusion; what differs is 7 rows' shapes and the per-rank
  bytes, FLOPs and laws — numbers, not structure.
* **A card class is coarser than the answers.** Llama-3.2-3B and
  Qwen3-30B-A3B walk identically on A100-80G-SXM (Ampere), RTX-4090 (Ada)
  and H200 (Hopper) — op, order, dtype, `source`, `source_badge`,
  `signature`, `dtype_env_json` all equal, prefill and decode. DeepSeek-V3
  at EP64·DP64 walks the same 78 rows on A100 and H200 with **16 rows'
  activation dtype different**, because A100 lists no fp8 column. So the
  class split is right for DeepSeek-V3 and over-fine for Llama: three
  bundles where one would do.

## 2. The rule

> A walk is identified by **the answers it gives to the questions it cannot
> defer**. Everything else is the point.

A *question* is a declared predicate over the walk's inputs whose answer
changes the walk's node set, its laws' shape, or its refusal:

| question | where it is declared today |
|---|---|
| does a dispatch condition hold? | a binding's `when:` — `DispatchCondition` (card keys, server flags, `activation_dtype`, `kv_block_size`, `tensor_parallel`) and `GraphCondition` (`dims`, `world_local` incl. its `local_in` per-rank list, roles, `readers_eat`) |
| does the card list a column for a dtype this walk needs? | `gpu_specs.PEAK_COLUMNS`, read both by `peak_families` and by `serving.py`'s launch resolution |
| which activation launch does the precision profile resolve to? | `PrecisionLaunch` + `LAUNCH_KEYS`; it decides cast nodes (`materialize_casts`) and a leaf's required edge dtype, so it changes the node set |
| does this degree divide the axis it shards? | `dims.yaml` `kind` + `transforms.tp_divides`; `world_local.local_in` additionally asks what the per-rank value IS, not only that it divides |
| which collectives does this placement insert? | `layout/materialize_comm` |
| which stage does this rank walk? | `Rule.stage_of` (pipeline) |

This table is a **closed vocabulary**, not a list someone keeps adding to:
it must be registered where the declarations are loaded, with a load-time
refusal for a `when:` key or a card column that is classified into none of
these — the shape `_check_dispatch_vocabulary` and
`_check_card_fact_classification` already have in
`framework_bindings/__init__.py`. Without that gate the table is the second
table CLAUDE.md §1.4 warns about, and the first unclassified key drifts
silently.

Three classes above were missing from this proposal's first draft and were
named by review: the precision launch (it changes the cast node set — the
cache law and the required edge dtype follow it), the HBM budget and memory
fraction (they change a refusal, and under pipeline parallelism which stage
a card binds), and `world_local.local_in` (not pure divisibility). The HBM
pair is the one that must NOT be in the answer set but must be in the
theorem's premise — see §5.

The key is **the answers, not the inputs**. Two bindings whose answers agree
on every question this graph asks are claimed to share a walk, and the
claim is read off the declarations rather than off a comparison of two
walks' output. That is what separates it from a proxy condition (CLAUDE.md
§1.3): nothing here reads a model name, a count of classes, or "these two
happened to match". It is not a proof, and §5 says what carries it instead.

Three properties follow:

1. **Self-maintaining.** A binding that starts reading a new card fact or a
   new `world` divisibility adds a question; the models whose graphs ask it
   split into more walks on their own. No list to update, so nothing to
   drift (CLAUDE.md §1.4: a module-level name list is a second table).
2. **No fourth transport path — a property of the END STATE, not of Stage
   A.** The ambition is that a consumer re-pricing a walk takes a value
   that only asking the questions can mint, rather than a bare
   `gpu_sku: str`, so the §1.5 failure (three transports, the rule written
   on one) cannot be written. Stage A does **not** deliver that: it changes
   the predicate, not the type the callers take. What helps today is that
   the rule already has one statement — `Card.refuse_unless_one_class` —
   even though four places call it (`capacity.py:631`, `:1215`, `:3763`,
   `sol_law.py:1456`), so replacing the predicate reaches all four. Making
   the value unforgeable is separate work and is not claimed here.
3. **Answerable off the card-free walk.** Not "before any walk": the `dims`
   questions are answered against `Analysis.node_dims` (`admits_match`),
   which is a walk's product, so computing them from the inputs alone would
   be a second derivation path — the thing CLAUDE.md §1.2 forbids. What
   makes the key workable is narrower and was found by reading the code:
   the symbolic walk is already **card-free** (`step_eval.Cell` is keyed on
   spec, parallel, phase, precision, stack, media, speculative, and names no
   card). So the order is: one card-free walk, then its own `node_dims` and
   dtype needs answer every question for every card. One walk, then N cards
   grouped — never N walks to decide the grouping.

## 3. Two stages, deliberately split

The rule touches three axes, but the axes do not carry the same risk, so
they are two separate pieces of work and only the first is proposed for now.

**Stage A — change what counts as the same walk. The graph is untouched.**
§3.1 and §3.2. Nothing that is a number today becomes a symbol: every walk
is still taken at exactly the bindings it is taken at now, and walked
identically. What changes is only *which walks are recognised as one walk*,
and the rule behind it has one statement already —
`Card.refuse_unless_one_class`. Four places call it
(`capacity.py:631` in `_restamp_gpu`, `:1215`, `:3763` in
`PendingEstimate.priced_on`, and `sol_law.py:1456` when a record is read on
a card), so replacing the predicate reaches all four at once; this proposal
earlier called `priced_on` "the single gate", which was wrong — it is one
caller of the single statement. Stage A replaces the predicate and keeps
every refusal. The existing re-pricing road (`_restamp_gpu` →
`_point_rows` → `batch_sweep.rows_at`) carries it unchanged.

Stage A is not only a cache win. It gives an owner to an invariant that has
none: *a count a dispatch condition reads must be walked as a number.* Today
that holds for `n_verify` because someone hardcoded it and fails for `B`
with a comment explaining the consequence. A kernel added tomorrow with
`dims: {n_verify: [2,3,4,5,6,7,8]}` would silently never dispatch — a wrong
number, not a slow one, and no test would turn red.

**Stage B — make the pricing half of a degree a symbol.** §3.3. This one
does change what laws are written over: per-rank extents are bound during
the walk by the layout rule, so a symbolic divisor means that rule writes
laws in it, divisibility must be decided before the walk, and
`materialize_comm`'s payloads become laws. The reduction is the larger one
(§8), but the blast radius is `LayoutPropagation` and the meaning of a walk.
It needs its own design and its own proof, and it should not start until
Stage A's oracle (§4) is green over the corpus. It is listed here so the
rule is stated once for all three axes, not because it is next.

What Stage A does **not** reach, which bounds its blast radius: the
symbolic walk is already card-free. `step_eval.Cell` is keyed on (spec,
parallel, phase, precision, stack, media, speculative) and `step_eval.py`
names no card at all; the card enters one layer out, through
`_CapacityPreparation` (the framework binding bound at that card, and the
sheet's peak columns), and only the per-op rows, the fusion rewrite and the
dispatch are card-dependent. So "walk class" names the class a *row set* is
valid for, not the class of the symbolic walk. Stage A changes which cards
one row set covers; it does not touch `Cell`, its key, or the evaluator.

Three consequences for the live paths:

* **One `/inference/estimate` is unaffected.** It builds its own rows for
  its own card either way, so neither its answer nor its cost changes. It
  can only gain: a stored record that now covers more cards is a record a
  cold estimate can read instead of walking.
* **The Kernel Bench plan's job set does not move.** A measurement is of
  hardware, so it stays per card; what two cards with equal answers share
  is the *key derivation*, and `Key.gpu` still separates them (already the
  case since the measure table was made card-free). Merging answers must
  not merge measurements, and does not.
* **The oracle's recording must be cheap or opt-in.** It runs on the same
  code path a live estimate takes, and a cold DeepSeek-V4-Pro estimate is
  already ~16 s against a 20 s deadline (`offload.DEADLINE_S`).

### 3.1 Workload: derive `s_verify == 1` instead of hardcoding it (Stage A)

The fact being declared here is **one bit**: is this the walk that does not
speculate? It is written as a pair (name, value) instead, and that is the
whole problem.

Why the bit has to be static at all: a framework runs Kimi-K3's whole-mixer
fused KDA decode only on a batch with no speculative tokens
(`models/kimi_k3/nvidia/kda.py:985-990`, cited by vllm.yaml). To claim that
kernel the engine must be able to *prove* "no speculative token here", and
`GraphCondition.dims` can only prove it by reading a walked number. So the
non-speculative walk binds the literal 1 and the proof is available; a
speculative walk keeps `s_verify` symbolic on purpose, because the
K = 1/3/7/15 ladder is then read off one walk by substitution
(`symbolic()`: "Every other count, and a verify width above 1, is a
symbol"). Hence a speculative and a non-speculative step are two walks, and
`StepCells` keys its cells on exactly that.

`Workload.walked_as_number` hardcodes the pair (`s_verify`, `1`). The motive
is real: `GraphCondition.dims` tests the **walked** value of a dim
(`fusion.py` `admits_match`), and four binding entries declare
`dims: {n_verify: [1]}` (sglang `:391`, `:403`, `:530`; vllm `:466`) for the
kernels a non-speculative step runs — Kimi-K3's whole-mixer KDA decode among
them. A symbol is in no list, so with `s_verify` symbolic those entries never
bind and the fusion silently disappears.

But the fact "`n_verify` must be walked as a number, at 1" is **already
declared** — in those `when:` clauses. The Python pair is the second
spelling of it, and the two can drift with nothing turning red: add a kernel
that lists `dims: {n_verify: [2,3,4,5,6,7,8]}` (the `GraphCondition`
docstring describes exactly that case) and the hardcode does not make those
widths static, so the entry never binds and no test fails.

Under the rule, the static workload counts are **derived**: the counts any
declared `dims` condition reads, at the values it reads them at. `s_verify`
at 1 falls out; nothing is named in Python.

**That rule as stated is not closed, and the gap is the whole point of
Q3.** Taken literally it would also make `B` static, because sglang `:530`
declares `dims: {B: [1]}` — and `B` must stay symbolic, since one walk
prices every batch. So "derive it, list nothing" needs one more
declaration: *which discrete axes may be walked as a number at all*, as
against the sweep axes that are symbolic by construction. Without a home
for that distinction the derivation degenerates into an implicit denylist
of `B`, which is the second table again. The home has to be where the axis
is already classified once — `dims.yaml`'s `kind` for a model dim, and the
workload's own field declarations (`Workload.symbols()`'s metadata) for a
request count — with the default for a sweep axis being "symbolic, and a
condition that reads it is refused at load or declared unanswerable here".
Picking between those two is a decision this proposal owes before any code:
it is the difference between the design's central claim holding and not.

This also turns one documented asymmetry into an engine-visible fact. sglang
`:530` declares `dims: {B: [1], ...}`; `B` is always symbolic, so that
clause can never bind. Today that is a correct comment in the yaml
(`:516-520`: "a walk that leaves B symbolic … proves nothing about it and
this entry binds nothing there"). Under the rule the engine sees "a
condition reads a count this walk cannot answer", and whether to refuse it
or to keep the present modelling decision becomes a declaration rather than
a comment.

### 3.2 Card: the answers, not the fact set (Stage A)

`CardClass` carries the card's whole dispatch fact tuple and its whole peak
column set. Narrow both to what this graph asked:

* the subset of dispatch conditions the graph's ops actually have, with
  their answers — not `(device, compute_capability)` verbatim;
* per dtype this walk needs, whether the card lists a column — not the
  card's full column set. (The peak **values** are already dynamic:
  `_point_rows` re-reads `peak_flops_per_sec_for_op` through
  `leaf_peak(card, dtype)` on the card a point is priced on.)

`CardClass` then becomes a conservative upper bound of the answer set, kept
only as the cheap pre-filter it already is.

### 3.3 Deployment: split the degree into its structural and its pricing half (Stage B)

A degree is static only through the questions it answers:

* **structural** — the divisibility answers (`tp` over heads, `ep` over
  experts), the collective set the placement inserts, the `world_local`
  conditions a binding reads, and which pipeline stage this rank walks;
* **pricing** — the divisor itself, the per-rank shapes, the payload bytes,
  the HBM budget. These become point symbols, bound like `B` and `s_input`.

Within one structural answer set the whole degree ladder is one walk read at
several points. Evidence that this is the common case is in §1; the
boundaries are real and fall out of the same rule — the one refused layout
and the size-1 group in the Qwen3-30B-A3B measurement are degrees whose
divisibility answers differ.

## 4. The oracle

The answer set is computed before the walk (§2.3). The walk, while running,
already evaluates every one of those questions — `holds_for`,
`admits_match`, `tp_divides`, `materialize_comm`, `stage_of`. Have it
**record** what it actually asked and answered, and compare that with the
set computed up front. Two structurally different routes to the same fact,
which is what CLAUDE.md §1.3 asks of an oracle; a counter of bad things is
not one.

On top of it, the existing byte oracle: a record walked per layout and per
card must be byte-identical to the one projected from the folded walk. That
is the comparison already used for the `measure-points-v8` change — per-op
row set and order including dtype (CLAUDE.md §1.5 lesson 2), not `t_sol`.

## 5. What must be proven before the key changes

The rule rests on one theorem:

> If two bindings agree on every question this graph asks, the layout is
> feasible on both, and both cards bind the same pipeline stage, their
> walks have the same node set, the same law shapes and the same structural
> refusal.

All three qualifiers are load-bearing. The third is the sibling rule the
repo already states: `CapacityCeilings.refuse_unless_one_stage`
(`capacity.py:2538`) — "the one statement of that rule, the sibling of
`card.Card.refuse_unless_one_class` on the other axis a walk depends on a
card by" — with its own case on record, MiMo-V2-Flash at pp=4 binding stage
0 on an H200 and stage 1 on an H100. Two cards can therefore agree on every
answer, both be feasible, and still be two answers because the HBM they
have binds different stages. Stage A replaces the predicate of
`refuse_unless_one_class` ONLY; `refuse_unless_one_stage` stays exactly as
it is and carries this premise. The HBM budget is thereby kept off the
answer set (it is price) without letting the stage it decides slip through.

The second qualifier was found by running the oracle rather than by
thinking: a card the layout does not FIT on walks no
rows, and that is on the pricing side of the split by design (`CardClass`:
HBM is substituted when the walk is priced). Qwen3-32B at DP8 replicates
its weights per rank, so it is infeasible on a 24 GB RTX-4090 and feasible
on an 80 GB A100 — the two cards give the same answers and different output,
and reading that as a counterexample would be reading a capacity outcome as
a structural one. A *structural* refusal (a dtype the card lists no column
for, a degree that does not divide) does belong to the answers and must
agree.

**And it cannot be proved by running the corpus.** Calling it a theorem
overspent the word: a corpus sweep is a feasibility check over the models
that exist today, not a proof over the models that will exist tomorrow. What
actually has to carry the claim is three things, and the corpus run is only
the first:

1. **Feasibility.** The §4 oracle over the whole corpus — every model ×
   every card class × every layout the ladders reach — with no case of
   "answers agree, rows differ". A counterexample names a question missing
   from the table, which is the productive direction and the reason to run
   it first. Its coverage today is narrower than that sentence and §8 says
   exactly how; the pipeline premise above in particular is **not** exercised
   by it, because the layout it sweeps sets no `pp`.
2. **A closed vocabulary with a load-time gate.** The question table in §2
   registered where the declarations load, refusing a `when:` key or a
   sheet column that classifies into none of its classes — the shape
   `_check_dispatch_vocabulary` and `_check_card_fact_classification`
   already have. This is what makes tomorrow's binding safe rather than
   hoped-for: a new kind of question cannot enter unclassified.
3. **`recorded ⊆ computed` on every walk, as a hard refusal.** The walk
   records the questions it actually answered; a walk that answered a
   question the computed set did not contain is refused, not warned. This
   is the standing check after the one-off corpus run, and the second route
   that keeps the computed set honest.

And (1) itself is worth nothing unless the comparison it runs actually
compares. Two of this oracle's first three bugs were precisely a blind
comparison, so the tool carries its own check: `--mutate` perturbs each
compared field on one card of a group and requires the audit to redden.
All 24 fields redden today and the baseline is clean; a field that stayed
green would mean the conclusion does not cover it. `--refusal` is the other
half: it forces one card of a group to raise the structural refusal the
answers are meant to predict (`NoPeakForDtype`) and requires the audit to
report it. That check found that the path comparing a raised refusal had
never run — the record it returned omitted fields the audit reads, so the
first card to raise would have crashed it instead.

**Completeness, by construction rather than by a list.** The row
comparison is a curated projection, so on its own "it compares everything
that matters" rests on a field list. `--record` closes that the other way
round: it builds the whole record the pool stores
(`gallery_frontier._layout_law`) twice — walked on each of two cards,
**both records covering both cards** — and diffs every leaf. A differing
leaf is allowed only if it is a NUMBER, because the card's price is numeric
by construction (`cardBinding` binds peaks, bandwidth, fabric and the
launch floor into a law graph that is already fixed, and nothing else), or
if it is the one path that names the card on purpose (`walked_on`). Any
differing string, bool or structure is a counterexample nobody had to think
to look for.

Run inside one class, where today's gate permits it: Llama-3.2-3B on H200
vs H20, **4,616 leaves, one difference, and it is `walked_on`**. Zero
non-numeric differences, zero leaves on one side only. That is the shape
the same test must produce for a merged group after the predicate changes,
and it is the baseline to compare against.

Run ACROSS two groups it cannot run at all, and the refusal is the
proposal in one sentence. Asking for a record covering A100 and RTX-4090
is refused by `refuse_unless_one_class` with: *"RTX-4090 is class
…+fp8, **which walks the graph differently** (its framework dispatch or
its peak columns differ)"*. For Llama-3.2-3B the oracle has measured that
it does not walk the graph differently. The gate's sentence is true in
general and false for this (model, card pair), and the answer set is what
knows which — but it also means **the record-level check is not available
before the predicate changes**. Until then the row-level comparison, with
`--mutate` and `--refusal` bounding its blindness, is the whole of the
evidence, and this document should not be read as claiming more.

**The stage guard, per call site** (the other thing review asked for).
`Card.refuse_unless_one_class` has four callers and only two of them ask
the stage question — but the division is sound, not a gap:
`capacity.py:1215` and `priced_on` (`:3763`) MOVE rows between cards and
both ask `refuse_unless_one_stage`; `_restamp_gpu` (`:631`) moves a
preparation and defers the row pairing to those two, as its docstring
says; `sol_law.cardBinding` (`:1456`) returns only numbers — peaks,
bandwidth, fabric, launch floor — binding this card's prices into a law
graph whose stage is already fixed, so no stage question arises. Widening
the class predicate therefore does not reach the stage premise: the cards
that newly pass the class gate still meet `refuse_unless_one_stage`
wherever rows move.

Without (2) and (3), (1) alone is an observation that decays — exactly the
"报告长着门的形状" failure CLAUDE.md §1.3 names.

## 6. Risks

* **The owner is not `ServingWalkFacts`.** This proposal first put the
  whole static side there; review rejected it and is right.
  `ServingWalkFacts` (`inference/sol_data.py`) is the **stated workload
  anchor** as a strict field table — a missing or extra field is refused —
  and the card's class is deliberately not in it. Putting a derived answer
  set inside would make it the class that holds everything. The answer set
  belongs one level out, as its own derived value at `ServingWalkKey`'s
  level (`sol_data.py`, today's owner of `card_class`), with the deriving
  logic staying in `arch/eval` and `framework_bindings` where the questions
  are declared. Either way the record identity moves once, so every cached
  walk is invalidated once, and the schema is worth getting right in one
  go.
* **A degree as a symbol changes what laws are written over.** `Phase.resolve`
  and the layout rule bind per-rank extents at walk time today; making the
  divisor a symbol means the laws carry it, so every law reader must already
  accept it (`Point` binds it like `B`). CLAUDE.md §2.1 is the constraint:
  one walk, two bindings, no parallel symbolic implementation and no
  `isinstance(..., Symbol)` branch.
* **Memory.** One walk serving a whole ladder holds laws in more symbols;
  `WALK_CELLS`' 768 MiB bound is measured against today's per-walk size and
  must be re-measured.
* **Scope.** This is not a performance patch. It changes what a walk *is*,
  so it lands behind the §5 corpus proof or not at all.

## 7. Is this only for the cache?

For Stage B, close to it: the degree split buys cost, not correctness.

For Stage A, no. It buys three things, and only the first is cost:

1. the walks the engine does not have to repeat (§8);
2. an owner for "a count a dispatch condition reads must be walked as a
   number" — today unowned, held up by one hardcode and one comment, and a
   silent wrong number when the next gated kernel is added (§3);
3. one predicate where there are now two independent ones (`CardClass` and
   `walked_as_number`) for the same question, which is the shape CLAUDE.md
   §1.4 calls a second table.

The graph's semantics are the thing not being traded: Stage A changes no
binding, makes nothing symbolic, and leaves `priced_on`'s refusal in place.
If the §5 theorem does not hold, Stage A does not ship, and the oracle built
to test it is still worth having — it is the first check that can see a
dispatch condition the walk cannot answer.

## 8. What it is worth — measured, not projected

The oracle has now been run over the whole corpus at one layout
(EP8·DP8, `vllm`, one card per walk class of the Scaling series). The
numbers below are its output, not an estimate:

* **63 models, 0 counterexamples.** No case of "answers agree, rows
  differ" survives, once a capacity refusal is read as the pricing outcome
  it is (§5). So the claim is feasible on today's corpus — which is point
  (1) of three, not the proof.
* **What that run does NOT cover**, stated because the number is otherwise
  read as stronger than it is: one layout (EP8·DP8, so **no pipeline
  parallelism** and so the stage premise of §5 is never exercised), one
  framework (`vllm`), one request shape, one card per walk class rather
  than every card, and 99 of the 315 (model, card) pairs skipped as
  infeasible at that layout. Widening each of those is cheap — the tool
  takes `--parallel`, `--cards` and `--model` — and must happen before any
  key moves.
* **Two self-blind spots in the first oracle, both found by review and both
  fixed**, which is the reason the first run's "0 counterexamples" could not
  have been trusted: the peak half matched a row's `BF16` against a column
  named `bf16` with `in`, so it silently contributed nothing and the whole
  question left the key (now read through `dtypes.peak_kind`); and every
  empty row set was labelled infeasible and dropped, so a card that walked
  nothing without saying why could hide (now only a stated capacity refusal
  is excused, and a raised refusal is compared as an answer). A third,
  found by running it: comparing a row's whole displayed formula compares
  the card's SUBSTITUTED numbers, which made every card pair a
  counterexample; only the symbolic half is the walk's.
* **The sound key saves about a third of the card axis, not four fifths.**
  216 card walks today → **144** grouped by the answers: 1.5×. Per model
  the pattern is flat: 5 feasible cards → 3 groups (16 models), 4 → 3 (18),
  3 → 2 (15), 2 → 1 (7), 1 → 1 (5).
* **The gap to the floor is measured too: 91 over-fine pairs across 45
  models** — cards whose answers differ while their rows are identical.
  `CardClass` today gives 160 groups, the refined answer set 144, and the
  rows themselves would allow far fewer. The dominant cause is diagnosed:
  a condition like `below_compute_capability: 10.0` answers differently on
  A100 and B200 while another entry covers the other side, so the dispatch
  that comes out is the same. Asking each condition separately cannot see
  that; asking the **resolved dispatch per op** would, and it is equally
  computable off the card-free walk. That is the refinement to make before
  Stage A ships, and the oracle is what will measure it.

So the earlier figure in this proposal — 13,335 walks → 2,739, 4.87× — was
the **floor** of what a perfect key could reach, reported as if it were the
saving. It is not what Stage A delivers. What Stage A delivers, with the
key as specified here, is about 1.5× on the card axis; with the
resolved-dispatch refinement it moves toward the floor, by how much is
exactly what the next oracle run answers.

The rest of the prize is Stage B's. Cold-cache `warm_caches` on 12 workers
is 6–7 h, the Gallery alone is 5,289 layout walks, and extending the
Scaling panel to every model (today 5 are pre-computed; any other model's
panel is computed on request, which cannot finish inside a request
deadline) would take it to about 13,335. §1's measurements say most of
those are one walk at another point — Llama-3.2-3B's 37 layouts are 3 row
structures — but collecting that is the degree axis, which changes what
laws are written over and is not this proposal.
