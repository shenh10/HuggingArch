# kernel_bench provider architecture

## What this is for

A measurement is only worth as much as the claim attached to it. This corpus
says "kernel K, from library L at version V, ran this shape in this many
microseconds" — and every one of those words is a claim someone has to be able
to check. The refactor exists so that each claim has **exactly one place it is
stated**, and so that a claim which is false **fails loudly** instead of
producing a plausible number.

That is the whole of it. Everything below is one of the two halves:

* **Say it once.** A fact stated in two places drifts, and the drift is
  silent — the second copy keeps working while it slowly stops being true.
  Wherever a value can be derived, derive it; wherever it must be written,
  write it once and let the rest read from there.
* **Make it falsifiable.** A declaration nothing can contradict is decoration.
  Every claim needs something that would fail if it were wrong — a check, a
  differential, a run against a reference — and the failure has to be visible
  at the moment it happens, not in a summary at the end of a run that may never
  finish.

The measure of success is not that the registry is tidy. It is that a wrong
kernel, a wrong dtype, a wrong shape, or a wrong library version can no longer
pass as a measurement. Tidiness is what that discipline looks like from
outside.

## The failure this exists to prevent

Every defect this refactor has turned up has the same shape, and it is worth
stating once rather than re-deriving it from each example:

> Something is true in one place and assumed in another. Nothing crashes. The
> number comes out looking reasonable, and is wrong.

A gate wider than its build measures a cell under a quantization the kernel
never ran. A tolerance keyed on the wrong dtype slot rejects correct kernels
for a whole library. A dimension name registered in one vocabulary and not the
other throws away every row an entire signature ever measured. A hand-copied
mirror of another module's table falls behind it. In none of these does
anything fail — which is why each was found by measuring, not by reading.

The corollary is the working rule: **prefer the failure you can see**. Decline
loudly over serving quietly; report a gap over filling it with a plausible
default; raise on an unknown case over passing it through.

## The mistake this replaces

`Provider` was one `(library, kernel_impl)` pair — the library half and the
kernel half fused into one record. Every kernel of a library therefore repeated
that library's install facts, and they drifted:

| library | kernels | `pip` declared |
|---|---|---|
| torch | 12 | `torch>=2.5`, `torch>=2.4`, `""` |
| vllm | 14 | `vllm>=0.11`, `vllm>=0.6.0` |
| sglang | 9 | `sglang>=0.5`, `sgl-kernel` |

`envs/*.txt` is generated from that field, so which constraint an env got
depended on registration order. The same fusion made `applies_to` a per-kernel
lambda restating dtype knowledge the build already encoded — a second place to
be wrong, which is how four gates ended up wider than their builds in one day
(NVFP4 folded with MXFP4, `startswith("fp8")` swallowing e5m2, humming
accepting bf16, GPTQ's uniform int4 grid used for E2M1's non-uniform one).

## The three records

```
Library                       one per third-party library
  ├── name                    "marlin" | "vllm" | "flashinfer"
  ├── pip                     the specs to install — envs/*.txt derive from them
  ├── source_build            not pip-resolvable (FA3-hopper, FlashMLA, DeepGEMM, DeepInfer)
  └── build_hint              how to provision it when unconfigured

Kernel                        one per measurable implementation
  ├── library                 names a Library — registration rejects any other
  ├── import_name             the module THIS kernel needs to import
  ├── kernel_impl             goes into the CSV identity
  ├── serves: KernelSignature what this kernel claims it can do
  ├── build(inv, env, inputs) → BenchCase
  ├── impl_params             deployment parameters folded into impl_id
  └── fused_op                the span it realizes, "" = its own signature

KernelSignature               the capability claim, published to the API
  ├── signature               the op-signature it implements
  ├── weight / activation     dtypes, exactly the set `build` handles
  ├── output                  the dtype it writes; () = the cell's activation
  ├── dtype_slots             (slot, dtypes) for a slot with no field of its
  │                           own — the format of a cache a launch reads, or
  │                           writes beside its output (kv); a cache it
  │                           writes AS its output is `output`
  ├── output_format           the cached records it writes as its output,
  │                           by cache-format name (`arch/cache_formats/`):
  │                           the cell's `output_format`; () = plain only
  ├── kv_format               the record of the cache it reads, or writes
  │                           beside its output (the cell's `kv_format`,
  │                           same naming rule); () = plain only
  ├── activation_granularity  the activation scales it writes; () = none
  ├── variant_dims            the signature's optional axes it serves
  ├── requires_dims           ...and those a cell must carry
  ├── secondary_outputs       (tensor, dtype) writes beside the primary
  │                           output — published, not a gate
  ├── phases                  () = either phase
  ├── divisible               (dim, modulus) tiling requirements
  └── enum                    (dim, allowed values) — e.g. sgl rope's d_head
```

**The declaration is closed.** It lists what the kernel serves, and what it
does not list it does not serve; a kernel never states what it cannot do. An
exclusion list is open-ended — every dtype or variant axis the corpus grows
reaches each kernel that forgot to exclude it, which is how FA4 was handed
fp32 cells and why FA2, FA3, FlashInfer and torch SDPA each had to say "no
sinks". Which slots and axes a declaration must speak for is read off the
signature, not listed beside the kernels: every dtype slot the signature
makes material (`OpSignature.dtype_slots`) is declared — registration
rejects an empty one — and every variant axis (`OpSignature.variant_dims`) a
cell carries must be one the kernel names. The axes of the default geometry
need no mention. Undeclared formats and activation scales mean the plain
tensor; an undeclared output means the activation's dtype, so a plain norm
is no longer offered the FP8-output cell of a norm fused with its cast.
`phases` alone stays open: a launch cannot tell which phase its shapes came
from, so a phase list states the serving path a kernel stands for, not what
it can compute.

`Library` owns installation. `Kernel` owns measurement and its own import
gate. `KernelSignature` owns the claim. Nothing owns two of those.

`import_name` sits on the kernel and not the library because it is not one per
library: sglang's rope kernels gate on `sgl_kernel` while its conv kernels gate
on `sglang`, and fa4 gates on `flash_attn.cute` rather than on the `sglang` that
merely wraps it — declared as `sglang`, availability said yes and 1,436 cells
failed one at a time in a single pass. Installation is per library; which module
has to import is per kernel.

**Status, kept separate from the design above so the two are not confused.**
`KernelSignature` is real and carries 64 of 66 kernels. `Library` is wired: the
15 records live in `providers/_libraries.py`, `Provider.pip` / `.source_build` /
`.build_hint` read through to them, and `register()` raises for a library name
no record declares. The registry rows are still flat `Provider` objects rather
than `Library.kernels` lists — a shape, not a second source of truth, since
there is now exactly one place each install fact is written.

## Why the split lands where it does

**Installation is a library fact.** What to install, and whether pip can
install it at all, is true of the library and was copied onto each of its
kernels — which is how one library came to state three different constraints.
Declaring it once removes the drift; the version gate then reads that one spec,
so a kernel cannot claim to run on a version its own env would never install.

**Identity is a kernel fact.** A corpus row is keyed on
`(framework, framework_version, kernel_impl)`. `framework` is the library's
name, `kernel_impl` the kernel's — so uniqueness of `kernel_impl` within a
library is exactly what the identity needs, and in this shape it is a natural
constraint rather than a bolted-on check.

**Overlap between kernels is expected, not an error.** Two kernels serving one
cell is the comparison this corpus exists to make: chunked against recurrent
for a linear-attention scan, one paged block size against another, sglang's
GEMM against cuBLAS. The envelope picks from exactly those. What must
never overlap is `kernel_impl`, not the declared dtypes.

**`impl_params` belongs to the kernel, not beside it.** A paged block size is
one kernel run at a different setting, not a second kernel — so it expands
`impl_id` while `serves` stays put.

**A fact the invocation can state belongs to `serves`, not `impl_params`.**
`impl_params` is for the distinctions the invocation still cannot express — a
dilation, a wrapper branch picked on a scalar. A permanent capability fact the
cell CAN carry — the cache format a launch writes, an output dtype it produces
— is declared on `KernelSignature` (`dtype_slots`, `output`), where smoke, the
published capability record and the stale-row audit all read it. Declining it
only from `impl_id` leaves every surface but the bench driver able to validate
and publish a cell the launch would refuse.

## What guarantees the declaration is true

Four layers, each catching what the one before cannot.

1. **Well-formed** (registration, static). dtypes in `DTYPE_VOCABULARY`, phases
   in `PHASE_VOCABULARY`, the declared signature matching the kernel's, moduli
   positive, `enum` values non-empty. Catches typos and mismatched axes.

2. **Unique identity** (registration, static). No two kernels of one library
   share a `kernel_impl`; rows would be indistinguishable.

3. **The claim holds** (smoke, dynamic). For every declared
   `(weight, activation)` combination, one minimal cell must build, run, and
   pass the numeric cross-check. A declared-but-failing combination is a lie
   the CI refuses. This is the layer that matters: it catches humming on bf16
   (raises), marlin fp4 with a mis-named attribute (`Invalid thread config`),
   and e5m2 quantized with an e4m3 scale (error explodes against the fp32
   reference — "does not crash" would have missed it).

4. **What the device ran** (CUDA launch record, `kernel_symbol`). The only
   evidence separating "declared A, ran A" from "declared A, ran B" when both
   produce plausible numbers — FlashKDA takes over `chunk_kda` on SM90+ unless
   `FLA_FLASH_KDA=0`, and vLLM's CustomOp runs its native torch path unless
   `custom_ops=all`. Recorded per row; observed after the timed region, since
   profiling serializes the stream.

   This layer was specified as "compare against a declared symbol" and cannot
   be. One `kernel_impl` at three shapes gives three different kernel FAMILIES
   (`cutlass::Kernel2<…>`, `ampere_bf16_s1688gemm…`, `ampere_bf16_s16816gemm…`)
   because cuBLAS picks per shape, so even the provenance token moves. A
   declared pattern would be loose enough to catch nothing or wrong on every
   other cell. The observation is present for every runnable provider and
   stable across repeats.

   So it records rather than declares, and supports the two comparisons that
   need no expectation — two kernels of one signature launching the SAME
   sequence on the same cell are one kernel under two names (`hbench smoke`
   reports it), and the same `(kernel_impl, cell, gpu, version)` changing its
   symbol between runs is a substitution, visible by comparing rows.

Layers 1–2 are structural and always on. Layer 3 is `kernel_bench/smoke.py`
(`hbench smoke`), run in CI wherever a GPU exists. Layer 4 is
`kernel_bench/kernel_symbol.py`, written into every measured row.

Layer 3 asserts on exactly one outcome — WRONG NUMBERS — and reports the rest.
An exception during build is genuinely ambiguous on any single box: the same
`RuntimeError` covers "this declaration is wrong" and "this kernel wants
Hopper". Asserting on it would either fail CI on correct code or force an
allowlist, and an allowlist is where the next wrong measurement hides. What is
unambiguous is that nothing declared produced wrong numbers, and that nothing
declared went untried (`no_cell`), so those are the two assertions.

The minimal cell is synthesized from an explicit per-axis table. An axis with
no declared extent makes the cell unsynthesizable and is REPORTED, never
guessed: a plausible size for an axis nobody declared is how the DSA indexer
came to be benched at `d_head = seq_len` for 2,430 rows.

## Where the boundary runs

The compute graph decides WHAT is one measurable unit — which ops exist, which
spans collapse into one kernel, what shape each runs at. The harness owns which
LIBRARY provides it, which `kernel_impl`, how to build and run and time it. It
does not get a say in what the computation is.

Most of this already holds: `plan.py` takes its cells from the engine's own
per-op rows and DROPS an op whose shape does not project rather than inventing
one from model-wide dims, and the op→signature table and `OpSignature` live in
`backend` with `kernel_bench` re-exporting them.

**The map is a quotient, not a subset.** Graph → operator is many-to-one and
lossy on purpose: `linear`, `qkv_parallel_linear` and
`merged_column_parallel_linear` are one `gemm`; a three-node score/softmax/apply
span is one `gqa_attention`. Which operator a node is can also depend on context
its op name does not carry — an `…expert…` id makes a `linear` a `grouped_gemm`,
a 4-D decay operand makes a `gated_delta_rule` a `kda`. So an operator name need
not exist anywhere in the graph, and requiring one would forbid exactly the
collapsing the map exists to do.

What must hold is that the quotient is TOTAL: a context rewrite must land on
something measurable, or it has turned a node the harness could have measured
into a key nothing serves. Enforced in `_rewrite`, at the one place rewrites are
made — not as a table of "names `resolve_op` may produce" kept somewhere else,
which is a second declaration that drifts from the branches it describes.

Which fused attention spans run the MLA kernel is read off the cache the span
reads: the walk records the slot kinds its keys and values come from
(`KVAccess.key_slots`), and each slot kind names the kernel family reading it
(`kernel:` in `backend/arch/cache_families.yaml`: `latent` → `mla_attention`,
`kv` → `gqa_attention`). No component names its kernel. Slots naming no
family, or two, are refused the first time a key is built.

The reverse direction is reported, not asserted: `gated_act_ffn` is a span
the cost model can draw and no measurement can reach — vLLM/sglang run the
SwiGLU FFN as separate GEMMs plus `silu_and_mul`, not one kernel. Named so
the gap stays countable instead of reading as coverage. `allreduce_fused_rmsnorm`
is in the op→signature map (same per-rank layout as `fused_add_rmsnorm`);
cells plan, and stay unmeasured until a 2-rank harness exists.

**`refs.py` is off the wrong side.** The shape contract now lives on the
signature (`backend/inference/op_signatures.py`): each signature declares its
canonical input layout (`TensorLayout` — one entry per tensor, one axis per
dim, tuples for packed dims) plus the structural axes the math takes that are
no tensor's axis. `cell_dims` is DERIVED from that declaration, and `refs` is
checked against it in both directions: the recording test asserts the
reference reads exactly the declared axes, and
`TestRefsBuildWhatTheContractDeclares` asserts the tensors it builds match
the declared layouts in name, rank, and extent. The reference MATH stays in
the harness — it is the oracle, not the contract — and so do the earned
input-building details (fp8 store dtypes, log-space decays, strided
indices): those are verification knowledge, not layout truth.

`variant_dims` remains, honestly: MLA's absorbed/non-absorbed split is still
a runtime distinction (the contract declares the default geometry and names
the selecting axis). What changed is that the engine-side correspondence is
now declared too — `engine_inputs` names each operand's axes in the engine's
order, validated at registration to be a permutation of the contract, so
`identity_shape_from_shape_in` zips shapes with axis names instead of
hardcoding "tensor N, axis M" per family. The families whose identity is not
a layout question at all keep their own projectors: `grouped_gemm` (count
reconstruction), `rope` / `causal_conv1d` (spec constants), `moe_gate` /
`indexer_logits` (widths from `shape_out`), `flash_dsa` (selection mask).

## The three vocabularies

A dimension has three names on its way from a model to a corpus row, and they
are three because they answer three different questions. Collapsing them is a
recurring temptation and always wrong.

| layer | example | answers |
|---|---|---|
| model spec | `d_model`, `n_lin_v_heads · d_lin_k_head`, `n_ssm_heads·d_ssm_head + 2·n_ssm_groups·d_ssm_state` | how wide is this axis IN THIS MODEL |
| primitive op | `C_in` / `C_out`, `in` / `out` | which SLOT of this op the model plugged it into |
| measurement identity | `d_conv_channels`, `d_reduce`, `d_out` | are two measurements the SAME physical cell |

The three conv examples above are the same op in three models. The op cannot
name that axis in model terms, because there are no model terms it could use.
The corpus cannot name it in model terms either, for the opposite reason: three
models that land on the same channel count are ONE physical measurement and
must share one row, so the identity has to be shape-level and model-blind.

Primitive ops therefore keep generic slots and models map onto them. An op does
NOT declare corpus-facing names — that would put knowledge of a consumer into
the primitive, which has no business knowing it has one.

What the layering does NOT license is a copy. Both copies are now gone, and the
import direction — `kernel_bench` depends on `backend`, never the reverse —
decided where the single one lives: down in `backend`, next to the schema and
the glossary it has to agree with.

* the op→signature map (the catalog's `binds_ops`, projected by
  `op_signatures.op_signature`), once 23 entries in `kernel_bench` against 14
  here plus a third table in `fusion.py` that contradicted both on `flash_dsa`;
* the signature's axes (`OpSignature`, `backend/inference/op_signatures.py`),
  once `required_dims` keyed by signature and `_OP_RELEVANT_DIMS` keyed by OP.

Reaching into another module's private globals is the same violation wearing a
different hat. A test that reads `_OP_RELEVANT_DIMS` or scrapes a source file
with a regex is not checking the contract; it is a second mirror that happens
to be executable. `test_cell_dims_are_what_the_reference_reads` is the shape to
copy instead: it hands the reference a recording dims mapping and compares what
the code ACTUALLY consulted against what the signature declares.

The write/read asymmetry this section worried about is closed. Both sides
call `identity_shape_from_shape_in`, and both refuse identically when it
returns None: the write side (`kernel_bench/plan.py`) DROPS the cell and
reports it; the read side (`calibration_overlay`) shows the row as
unmeasured. The model-wide-shape fallback that fabricated keys the corpus
cannot contain is gone — an op with no projection is a visible gap, not a
quiet miss. `fused_moe` is the one mapped op still without a projection
(its cells DROP until MegaMoE lands), and that gap is now the same shape
on both sides.

(Working corpora predating schema v4 do carry the engine's symbols —
`K`/`N`/`Sq`/`E`/`H`/`I`, 25,189 rows in one scratch aggregate. Stale, not a
live producer: the committed corpus is schema-v4 and measurement-vocabulary
throughout, and today's whitelist refuses those keys outright.)

## How a launch is timed

`timing.time_callable` is the only clock. On a GPU each sample is one launch,
bracketed by CUDA events on the current stream, and the row records the p50
of the samples (p95 and the mean beside it). The order a sample is queued
in, and why each step is there:

```
sweep L2  →  hold  →  start  →  call()  →  sweep L2  →  end      (A)
sweep L2  →  hold  →  start  →             sweep L2  →  end      (B)
                      hold  →  start  →                    end      (C)
sample = A − B + C
```

* **The sweep** reads a buffer twice the L2, so a memory-bound kernel reads
  from HBM rather than from what the previous sample left in cache. It is
  read-only, so the L2 it leaves is clean: the launch never pays to write
  back someone else's lines.
* **The second sweep, inside the window,** is what puts the launch's writes
  in it. Rotating input buffers would not: they make the reads cold
  but leave the launch's own write-back after `end`, and they need a clone of
  every provider's inputs, which an in-place op, a captured graph or a pointer
  fixed in a plan cannot give. L2 is write-back: a launch's stores sit dirty in L2 when it ends and
  reach DRAM when something evicts them. An `end` right behind the launch
  closed the window first, and an output smaller than L2 was timed as free.
  The sweep evicts those lines; its own cost on a clean L2 (B) is
  subtracted, and the events' own gap (C, ~2 us on a 4090), which the
  difference cancels and every older row contains, is added back. The flush
  this replaced zeroed the buffer instead, leaving L2 full of dirty lines:
  each line a launch touched, read or written, paid a flush line's
  write-back, and the launch's own writes paid nothing, so the answer
  depended on the L2 state at `start`. One cell, RTX 4090, 2026-10-25 (vLLM's
  DeepSeek-V4 un-rotation, 208 tokens, 27.3 MB): 45.1 us (0.60 of peak)
  under the old flush on an idle card, 18.4 us (1.47) from a clean L2 with
  no drain, 24.6 us (1.10) in the corpus from the bench card under the old
  flush (its L2 partly clean, cause not identified), 34.8 us (0.78) under
  this method from either state. `tools/timing_window_probe.py` measures
  the window on a pure read, a pure write and a copy.
* **The rows timed before the drain were kept, not purged.** This changed
  the clock under every row while the coverage key (`framework`,
  `framework_version`, `kernel_impl`) stayed the same, so by the rule below
  for a provider change, every measured row would be purged and re-measured
  in the same merge. They were kept deliberately: purging means
  deleting ~2.9 million lake rows and re-measuring all of them, roughly five
  days of the 4090 queue plus an H100 campaign, and for most rows the old
  clock's error is small against their latency (it is at most one L2
  write-back: L2 / DRAM bandwidth, ~17 us on an H100, ~82 us on a 4090).
  Instead each row says which clock timed it: `time_callable` returns its
  method with the latency and the row records it (`timing_method`, schema
  v16, `benchmarks.TIMING_METHODS`). Old rows and files are not migrated or
  backfilled; the column is optional on ingest, so they read NULL, which
  means "timed before the drain". The Kernel Bench page marks a NULL-clock
  latency with that caveat and filters nothing; the estimate reads old and
  new rows alike. Consequence to keep in mind: the coverage key still counts
  an old row as tried, so a kept cell is re-measured only when its
  implementation's version changes, and an envelope can pick between an
  old-clock and a new-clock row of two libraries.
  Deploy order follows from the column: a v16 harness's rows are refused by
  a v15 server (the schema-version check in `backend/inference/benchmarks.py`
  and `validate_header`'s strict column list), so the server and its lake
  (`ensure_schema` adds the column) go first, the bench boxes' harness after.
* **An event's timestamp is taken when the GPU reaches it, not when the CPU
  queues it.** Stream order makes `start` precede the kernel — a kernel queued
  after it cannot begin first — but it does not make the kernel *present*.
  Without the hold, the GPU drains the sweep, reaches `start`, and then waits
  for the CPU to finish the provider's launch path before the kernel exists to
  run. That wait lands between the two events and reads as kernel time. An
  H100's sweep covers ~35 us of host time; FA3/FA4's Python launch takes
  ~90 us, so a 5 us prefill read 79 us, and every eager torch baseline was
  inflated the same way. The rows this produced were deleted rather than
  corrected (no stored number separates the wait from the kernel).
* **The hold** is a GPU-side spin (`torch.cuda._sleep`) queued after the sweep
  and before `start`, sized at twice the host time warmup measured for one
  call. The GPU is still spinning while the CPU queues `start`, the kernel and
  `end`; when it reaches `start` the kernel is already behind it, so the
  events bracket queued GPU work only. A 30-kernel eager call on a 4090 reads
  its CUDA-graph time (84 us) instead of its launch-bound one (180 us).
* **Same stream, assumed.** The events order only the stream they are recorded
  on. A provider that launches on a side stream escapes the bracket: `end` can
  fire before its kernel finishes, and the row reads fast. The providers here
  launch on `torch.cuda.current_stream()`; nothing checks it yet, and a kernel
  library that moves to its own stream would break this silently.

`perf_counter` appears twice and never in a row: to size the hold, and as the
clock on a box with no GPU.

## How a collective is measured

A collective cell (`all_reduce`, `all_gather`, `reduce_scatter`,
`all_to_all`, and the fused `allreduce_fused_rmsnorm`) is a launch over a
GROUP: its key carries `n_comm_ranks`, and on one process every one of these
ops is the identity. So the harness needs as many processes, on as many
cards, as the cell's group, and everything below follows from who owns
which fact.

* **What a cell needs is the signature's.** `n_comm_ranks` and
  `n_comm_stride` are declared structural axes of these signatures; a cell
  with `n_comm_ranks = N` needs N processes on N cards. Nothing lists the
  collective signatures again: the gate reads the axis.
* **How many cards a run can drive has one answer.** `Env.devices` is the
  measuring process's visible device count; a plan asks the same question
  of the run it will start (`BoxEnvs.devices`: the cards the run will be
  shown, `CUDA_VISIBLE_DEVICES`, never more than the profile's `devices`).
  A single-card queue's plan therefore puts a pair's cells in its gaps, and
  the queue that is shown both cards queues them.
* **Whether a group fits the box is the cost model's placement.** One gate
  in `Provider.refusals` declines a group larger than the cards, or one the
  card's own sheet places across nodes (`comm_cost.members_per_node`, the
  number the estimate prices the group's fabric by) — a box is one node.
  Inside a node the group's spacing does not change its fabric, so ranks
  sit on adjacent cards. A refused cell stays a gap in the coverage total,
  with its reason: missing, not exempt.
* **One path.** A collective is measured by the same `run` -> driver ->
  `build` -> cross-check -> clock -> row as every other cell. A provider's
  `build` is one rank's launch. The driver process is rank 0 on card 0;
  `comm.Group` spawns ranks 1..N-1 on the next cards the first time a cell
  of that group size comes up (process-death signal tied to the driver;
  NCCL for the data; numbered commands through the group's `TCPStore`, which
  no launch trace sees) and each peer builds the same provider, found by
  its `kernel_impl`. A one-card cell releases the group first: its NCCL
  buffers would otherwise sit on card 0 under a cell near the card's limit.
* **Every rank calls the same thing the same number of times**, or NCCL
  deadlocks. The case's `run` (the untimed calls: the cross-check, the
  launch trace, the kernel symbol) tells the peers to make the same call
  and waits for their answers. The clock is `time_callable` and nothing
  else: the driver times a group case as `with case.group.timed() as
  launch: time_callable(launch, group=case.group)`, where `timed` brings the
  peers into the same clock. Its sizing (warmup and sample counts) is made
  from the MAX of every rank's probe, and an allocation one rank cannot make
  stops every rank at the same point, so no rank waits in a collective the
  others never join. A peer that dies or fails marks the group broken, and
  the next collective cell starts a fresh one.
* **A sample is the slowest rank's.** Before each timed window a 1-element
  all-reduce aligns the ranks on the GPU, outside the window; after the
  loop each sample is the max over ranks. A collective has finished when
  its last rank has, and that is what a step waits for.
* **The oracle is a second route, and it is the signature's.** The inputs
  each rank holds its own copy of are declared on the signature's oracle
  (`RefSpec.per_rank`: the payload; for the fused seam `x`, not the
  replicated residual). Rank r's copy is rank 0's rotated by a
  rank-dependent offset (`comm.rank_tensor`), so the ranks hold different
  data and no two send the same shard, and rank 0 can rebuild all of them.
  The oracle computes the exchange from those copies in fp32 (the sum, the
  concatenation, the slice); the check compares it with what came over the
  fabric. A launch that skipped the exchange no longer agrees.
* **Both cards, exclusively.** A collective is timed with both cards' memory
  and copy engines; another run on either card would be in the number. A
  `run` locks every card it can see — all or none, in UUID order — and a
  box runs its collective cells as their own queue with every card visible,
  while the single-card queues stand aside.

What this does not measure: a group larger than the box (the 4090 box has
two cards; a plan's groups reach 512), the routed exchange
(`routed_all_to_all`, DeepEP), the pipeline hop (`send_recv`), and any
framework's custom all-reduce that needs peer-to-peer access the card does
not have (GeForce cards have none, so on them the serving path is NCCL's
own, which is what is measured).

## What a sweep records when it does not measure

Every cell × implementation a run attempts leaves a row in the same table as
the measurements, `outcome` saying what happened (`benchmarks.OUTCOMES`,
schema in `backend/inference/BENCHMARKS_CSV.md`):

| status | row | written by |
|---|---|---|
| `passed` / `unverified` | `measured`, `cross_check` = / ≠ `passed` | `driver.run_jobs` |
| `oom` | `oom` | `driver.run_jobs` (inputs or provider) |
| `mismatch` | `mismatch` + the comparison's numbers | `driver.run_jobs` (cross-check rejects, or the launch gate says the launch is not the cell's) |
| `error` | `error` | `driver.run_jobs` (build / launch raises, a runtime `NotImplementedError` included) |
| `timeout` | `timeout` | `driver.run_jobs` (one launch over the timing budget); `cli._cmd_run_all` (stall watchdog) |
| `crash` | `crash` | `cli._cmd_run_all` (process died, or `driver.CudaContextLost`) |
| `not_supported` | none — read off `accepts` / `impl_id` | `hbench report` |
| `pending` | none | `hbench report` |
| no provider on the box | none — the job is in `<plan>.no_provider` with each kernel's reason | `plan.write_plan`; counted by `hbench report` |

Before a run, a plan written for a box (`sweep --box`, `enqueue`) asks each
job the run's own gate — `Provider.declines`: the env's library, then
`unavailability`, then `refusals` — of the envs that box DECLARES
(`box.BoxEnvs`: the profile, its card's compute capability, the versions its
requirements and constraints pin, and the source builds it can start).
Nothing is imported; a version nothing pins reads as unknown, which the gate
allows. A job with a candidate goes to the plan tagged `candidates` (kernel →
library) and `judged_on` (the premises' digest); one with none goes to
`<plan>.no_provider` and is never handed to the GPU, but is still a cell:
`report` counts it, so coverage does not shrink. `run-all` starts only the
envs owning a candidate; when its own premises differ (another card, an
`--env`, `--constraints`, an unjudged plan) it judges the plan and its gaps
again against the envs it can start. A needed source build it cannot start
leaves the run not done (non-zero, no `.done`). The env's own gate and the
canary still judge every queued job at run time.

`crash` and `timeout` from the supervisor are precise to the implementation:
the run publishes the implementation in flight on the progress sidecar
(`supervisor.write_progress(…, in_flight=…)`) and clears it once the provider
is done, so a death in a job's shared stage (inputs, oracle) names nobody and
records nothing — that is the harness's failure, not a kernel's. After the
row lands the job goes back into the remaining list: its other
implementations are measured, the crashed one is skipped by its row.

A release that moves or drops what a provider calls fails every cell of it
the same way, so `run_jobs` attempts each launch path's canary first: the
smallest pending cell of one implementation at one phase and precision
(`driver._launch_path`), taken from the model's own jobs, attempted as an
ordinary attempt when the path first comes up. A canary that ends in a
measurement IS that cell's row and the path runs as before — a failure
particular to one cell is still that cell's row. A canary that ends in any
other outcome is that cell's row too, and the path's other cells are not
attempted this run: they stay without rows, and `run` and `run-all` exit
`EXIT_PATHS_DISABLED` (4) — reached the end, nothing to quarantine, but not
done — so the queue writes no `.done` and its next pass runs the model again:
one cell per still-broken path, the whole path once the env is fixed. Both
list the disabled paths with the canary's outcome (`RunResult.disabled`,
carried to `run-all` on the progress sidecar). The path failing together is
an assumption (`driver._launch_path`): a failure only the smallest cell hits
stops the path for that run. No row means no verdict — a canary whose
shared stage raised leaves its path running — and an `oom` row is no
verdict on the path either: running out of memory is the cell's size, not
the path failing, so the larger cells are attempted and each records its
own `oom` rather than staying pending for a pass apiece. A canary runs ahead of its own job,
so the sidecar's `index` (where a restart resumes) and its `job` (what is on
the GPU, dropped when no row can take the blame) are different jobs then.

Any row counts as tried — for `--skip-existing`, `lookup.covered` and
`--measured-in` alike — only at the implementation version on its identity:
the coverage key is `(framework, framework_version, kernel_impl)`, for a
measurement and a failure the same. A new library version re-tries it.

A change to the provider's OWN bench code at the same library version (its
`build`, `convert_inputs`, `wrap_output`, the inputs it is handed) is not on
any row, and it makes the old rows of that implementation doubtful whatever
their outcome — a measurement timed the old launch exactly as a failure
failed on it. So there is one rule for all of them, and no second mechanism
for failures: **the change that fixes a provider's build or input conversion
purges that `kernel_impl`'s old rows — measurements, OOMs and failures
together — in the same merge**, through the existing revocation channel:
locally the results CSVs, with the removed rows kept as a
`bench_purge_<date>_<reason>.csv` record (`tools/dedup_results_csv.py`'s
convention), and in the lake `store.delete_rows` by id. The rows to purge
are chosen by `kernel_impl` (narrowed by `framework_version` / `outcome`, or
by a `--cell` expression over the row's dims where the change only alters
the launch past a threshold), named on the command line, never kept as a
list in code — that is `tools/purge_impl_rows.py`, dry-run by default and
writing the `bench_purge_<date>_<reason>.csv` record before it deletes. A
purged cell is then pending again, and the next sweep re-measures it.
(`tools/unserved_rows.py` is the same channel for the other trigger —
a declaration that narrowed.)

Changing a provider, the checklist line: **if the edit changes what its
build launches or the inputs it converts, purge its `kernel_impl`'s rows in
the same merge**; an edit that cannot change a launch (a comment, a rename
inside the module) does not. A change to the clock itself
(`timing.time_callable`) changes every row's measurement at once; the one
so far (the L2 drain) kept the old rows deliberately and records
the clock per row instead — see "How a launch is timed".

## Consumers and how they change

| consumer | before | after |
|---|---|---|
| `applicable(inv, env)` | filter 66 flat providers | each kernel through `Provider.declines` (its env's library, availability, `serves`) |
| `candidates(inv, envs)` / `plan.write_plan` | — | the same gate per kernel against the box's declared envs (`box.BoxEnvs`), at plan time; `run-all` starts the envs the plan's candidates name |
| `dependency_manifest` / `envs/*.txt` | scan every provider's `pip` | one `pip` per library, no reconciliation *(live)* |
| coverage report (`kernel_bench report`, `cli._cmd_report`) | flat list | library → kernels, matching how the libraries are actually installed |
| `/api/bench/providers` | — | library → kernels → `capabilities()`, the published surface (live) |
| Kernel Bench Providers tab | — | the same catalog, joined to measured rows for coverage *(live)* |
| `/fusion/registry` backends | library, kernel_impl, pip | + `capabilities`, `min_sm` — same facts where the UI already reads them |
| CSV row | `framework`, `kernel_impl` | unchanged — the split matches what the identity already assumed |

## Migration

Mechanical first, semantic second.

1. Split the install facts out of `Provider` into `Library`, and reconcile the
   drifted specs — a real fix, not a rename. *(done — `providers/_libraries.py`;
   the 66 registrations name a library and keep their own `import_name`. The
   reconciliation moved five env files: torch to `>=2.5` everywhere, `vllm` to
   `>=0.11` with its stale `>=0.6.0` line dropped, and deepgemm/flashmla to a
   library-level build hint instead of whichever kernel happened to register
   last.)*
2. Declare `serves` per kernel, deleting each `applies_to` as it is replaced.
   Exactly one of the two, so there is never a second place to be wrong.
3. Add the layer-3 smoke test; it gates every further declaration.
   *(done — `kernel_bench/smoke.py`, `hbench smoke`)*
4. Publish `capabilities()` through the API. *(done — `/api/bench/providers`,
   and folded into the existing `/fusion/registry` backends view rather than
   given a second plumbing path)*

An undeclared scope publishes as `null`, never as an empty constraint list: a
kernel still on a hand-written gate has promised nothing, and empty constraints
would read as "serves everything".

`enum` exists because migration will meet gates the declaration cannot express
yet (sgl rope serves `d_head ∈ {64,128,256,512}`). The rule when that happens
is to extend the declaration language, never to fall back to a lambda: a gate
that opts out of the checks is where the next wrong measurement hides.
