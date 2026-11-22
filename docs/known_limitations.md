# Known limitations of the cost model

This document lists the places where the HuggingArch cost model knowingly
departs from what a real serving stack does. Each entry states what the model
does today, how that differs from serving, the direction and size of the
error where it is known, and the shape of the fix. The entries are not ranked.

Two kinds of departure appear below. Some are in the analytical model itself
(the walk that produces shapes, FLOPs and bytes). Others are in the measured
calibration (Kernel Bench cells and the kernel declarations that decide which
kernel may stand for which cell). A row with no measured cell is still priced
by the analytical roofline; it is reported as a coverage gap rather than
hidden.

Sections:

1. [Workload and deployment assumptions](#1-workload-and-deployment-assumptions)
2. [Mixture of experts](#2-mixture-of-experts)
3. [Activation quantisation casts](#3-activation-quantisation-casts)
4. [Measured calibration and kernel declarations](#4-measured-calibration-and-kernel-declarations)
5. [Spec representation](#5-spec-representation)

---

## 1. Workload and deployment assumptions

### 1.1 Expert-parallel load imbalance is an input, not a measured fact

**Today.** `ep_load_factor` (ρ, the ratio of the busiest expert's load to the
mean, workload symbol `rho`) sizes the busiest EP rank's routed rows and the
routed all-to-all (`Workload.ep_load_factor`, `ExpectedRoutedLoad`). Its
default is 1, so every EP number is computed at perfect balance unless the
caller supplies ρ.

**Deviation.** Real routing is skewed, and the slowest rank sets the step
time. At ρ = 1 the routed-expert time and the dispatch/combine exchange are
**optimistic** by roughly the factor by which the busiest rank exceeds the
mean.

**Fix.** The benchmark corpus already carries a nullable
`measured_load_factor` provenance column (`backend/inference/BENCHMARKS_CSV.md`),
but nothing fills it. Closing the gap needs a producer that measures under a
real routing trace, and then a decision: either the projection keeps declaring
perfect balance and says so, or it takes the measured factor per model and
load from the corpus. A tuned constant is not an acceptable substitute.

### 1.2 A step is priced as wholly one phase

**Today.** Every cost path takes one `phase` and prices the whole step under
it: `ParallelStep` carries `(deployment, phase)`, and `Phase.resolve` gives one
set of step extents (`s_query`, `s_kv`, `s_past`, `s_pairs`, `n_verify`).
Chunked prefill is modelled as a sequence of prefill chunk walks
(`capacity._fold_chunked_prefill_sol`), still prefill-only.

**Deviation.** Under chunked prefill a framework such as vLLM hands one
forward pass both prefill and decode tokens, and prefill context parallelism
(PCP) treats them differently within that batch: it splits the prefill tokens
across its group and replicates the decode tokens
(`attention/pcp.py::_gather_prefill_cache_inputs`). A mixed step priced as
prefill charges the decode tokens a PCP division they did not get; priced as
decode it denies the prefill tokens the division they did get. The error is the
full `cp` factor on whichever side is mispriced. Prefill and decode are
evaluated and reported separately today, so no current output asks for a mixed
step; the limitation binds only a consumer that wants one.

**Fix.** Make a step a token composition, `{prefill_tokens, decode_tokens}`,
rather than a label. Each world then states which component it divides (the
way `phases:` states which phase it is live in), `degree_in_phase` becomes a
per-component degree, the residual stream's sequence cut becomes
per-component, and the cost is the sum over components. This changes the
binding layer: `s_query` stops being one number.

### 1.3 A weight's placement is one per spec, though frameworks differ

**Today.** `weight_axes` is a spec-level fact, so a weight that two serving
stacks split differently is billed one way for both. DeepSeek-V4.1's Engram
`wkv` follows vLLM, the default framework, which column-splits it
(`ColumnParallelLinear(gather_output=True)`); sglang keeps it
`ReplicatedLinear`.

**Deviation.** An sglang deployment is billed vLLM's placement: 1/TP of the
157 MB weight per rank plus an all-gather of the output, where sglang reads the
whole weight on every rank and gathers nothing. For sglang the weight read is
therefore **under-billed** by (1 − 1/TP) of the weight per rank and the
communication is over-billed by one all-gather. The same shape of gap applies
wherever a framework's placement depends on its version or the GPU generation.

**Fix.** Let the framework binding choose among placements that the spec
declares. No such mechanism exists yet, and it must not be added per model.

---

## 2. Mixture of experts

A decode step routes `B·top_k` rows over the experts one rank holds, so the
mean rows per expert is usually fractional. The model pads every expert that
receives rows to a whole number of rows and charges nothing for an expert that
receives none (`eval.routed.RoutedLaunch`: `touched_experts`,
`rows_per_expert`, `launch_rows`). The measured overlay re-reads the FLOPs, the
routed-buffer bytes and the expert weight read at that padded geometry
(`calibration_overlay._at_geometry`). Two terms are not covered by that rule
(§2.1, §2.2); §2.3–§2.6 concern the expert-axis model itself
([MoE expert axis](moe_expert_axis.md)).

### 2.1 The gathered token read is not re-read at the padded geometry

**Today.** `permute` reads `min(tokens, E × M)` token rows
(`OpSpec.gather_source`; under EP a rank gathers only its own experts' rows).
`_at_geometry` rescales only the terms on the routed buffer, and the token
operand of a dispatch boundary is not on it, so the read side keeps the walk's
unpadded value.

**Deviation.** Where the touched experts' rows are fractional, the padded
launch gathers up to `touched_experts × ceil(R / touched_experts)` rows, and
the read is billed low by up to that padding. The error is small and
conservative: measured utilisation reads slightly low.

**Fix.** State the gather's read as a law of the launched rows, as the routed
buffer's bytes already are, and re-read it in `_at_geometry`.

### 2.2 Zero-compute experts have no identity term

**Today.** LongCat-Flash reserves zero-compute experts in its routing
(`n_route_targets = n_experts + n_zero_experts`, `moe_ffn.yaml`). The expert
bank's buffer holds only the rows routed to weighted experts, which is
self-consistent.

**Deviation.** A zero expert returns its input, weighted: `Σ_zero w·x` is model
arithmetic that no graph node represents. vLLM runs it as one elementwise launch
over the zero-routed slots (`compute_identity_kernel`, `fused_moe.py`), reading
`x` and performing a multiply-add per element. The step **under-bills** that
launch. It is small beside the expert GEMMs. vLLM's rewrite of zero slots into
weight-0 rows of expert 0 is a framework inefficiency and is deliberately not
modelled.

**Fix.** Add the identity contribution to the component as an explicit node
over the zero-routed slots.

### 2.3 The measured path through a dispatch zone has no golden coverage

A phase's measured total is all-or-nothing: if any row lacks a measurement,
the point has no measured total and no `calibration_error`. The committed
golden cells carry no complete measured MoE phase, so a defect in the
measurement key, launch count or cadence of a dispatch-zone row (`permute`,
`combine`, the routed GEMMs) moves no committed number. Closing this requires
a committed measured seed that covers every row of one MoE phase on one card.

### 2.4 A lookup table's read fraction is not part of `active_share`

`Analysis.active_share` is the share of a node that runs per token; for a
routed bank it is `n_experts_active / n_route_targets`. A leaf that gathers
rows from its own table (an MTP stack's second `embed_tokens`, hash
embeddings such as DeepSeek-V4.1's Engram) also reads only a fraction of
itself, `min(1, rows_per_token / table_rows)`. That fraction is applied to
`active_params` and to the table's traffic (`traffic._table_read_fractions`,
the table branch of `phases.py`), so bytes and time are correct; only the
published `active_ratio_per_token` column reads 1.0 on such leaves.

The fraction cannot simply be merged into `active_share`: the saturation law
(`row_roofline.weight_demand`) infers "this is a bank" from a ratio below one
and would apply the bank law to a gather. The fix is to have `weight_demand`
select its law from a declaration (the leaf gathers, or it is reached by
routed rows), after which the lookup fraction can join `active_share`.

### 2.5 The busiest-rank reading has no integer floor

Rows per expert are integers, so the busiest rank receives at least
`ceil(mean)` rows. The mean `M` must stay an exact rational, because every
total is built from it (`R = E · M`); a ceiling there overstates decode work
by up to the expert count. The bound belongs in `ExpectedRoutedLoad`, as
`routed_row_capacity_per_rank = max(ceil(per-rank mean), per-rank mean ·
busiest_rank_load_factor)`. At the default `ep_load_factor = 1` the per-rank
mean falls below one row whenever `ep > B · k`; stating the floor changes no
number the latency model reads today, because that model consumes the
factor, not the capacity.

### 2.6 Two boolean switches are registered as dims

`n_topk_renorm` and `n_routed_scale` are booleans read by `when:` predicates,
not extents; `n_routed_scale` is written by hand in model files rather than
derived from config. They are dims only because `when:` reads dims.

Combine under EP without DP-attention is priced at the rank's routed rows, the
speed-of-light form. vLLM's default Triton path sums all `T·k` slots after
zeroing the non-local ones; the model does not copy that inefficiency. This is
a modelling choice, not a limitation.

---

## 3. Activation quantisation casts

The serving graph grafts a `quantise_activation` node on every edge whose
consumer is a matmul that takes a quantised activation
(`inference.materialize_casts`, after communication and before fusion). The
node declares the format it writes (`SpecIRNode.output_format`), and its
write, every read of it, and the write of any fused launch that absorbs it are
priced at that format's record width, elements and scales
(`traffic._record_width`). There is one copy per tensor and format, moved above
element copies that keep the scale-group axis ([quantization](quantization.md#activation-casts-in-the-serving-graph)).
The MoE experts' copy is made once per token on the dispatch's token input, and
the dispatch and its exchange move it with its scales
(`traffic.payload_width`, `exchange_widths`). The entries below are where this
falls short.

### 3.1 The graft and fusion ask different questions of a view

**Today.** The graft moves a cast above a view only where the view keeps the
innermost axis whole (`traffic.ElementCopy.keeps_rows`); after a view that
regroups it (`head_merge`) the cast stays on the tensor the GEMM is handed.
Fusion reads through views on the rank's own shapes and asks whether each
scale block stays whole (`fusion.ViewChains._keeps_blocks`), so a plan that
names the cast as a member can absorb it across the view (DeepSeek-V4's
`fused_inv_rope` takes `wo_a`'s cast through `head_merge` / `head_split`).
Casts behind views that no plan absorbs, for example every FP8 model's `o_proj`
cast after `head_merge`, remain standalone launches keyed by the call site's
layout. An innermost-axis slice is refused outright, because the dimension map
does not carry its offset, even where the slice falls on a scale-group
boundary and would commute.

**Deviation.** Two rules answer one question, and the standalone casts are
keyed by a layout other than the one a framework quantises. Byte counts are
unaffected; the cast's measured cell can be.

**Fix.** One rule: the graft asks the scale-block question on per-rank shapes.
That moves the `head_merge` casts above the view in the graph itself and
leaves fusion nothing to read through. Carrying slice offsets in the dimension
map would let a slice at a group boundary commute as well.

### 3.2 DeepEP low latency quantises inside its dispatch kernel

**Today.** The graph makes the MoE experts' copy once per token, right after
the norm that writes it (shared with the shared expert's GEMMs where the layer
has one), as a standalone `quantise_activation` launch, and the dispatch moves
that copy. This is the default serving path in vLLM (`prepare_finalize/
no_dp_ep.py`, `naive_dp_ep.py`).

**Deviation.** DeepEP's low-latency dispatch takes the BF16 tokens and
quantises inside the dispatch kernel (vLLM `prepare_finalize/deepep_ll.py`,
`use_fp8`). Where the layer has no shared expert, no separate cast launch runs,
and the graph **over-bills** that launch (T·K reads at activation width, T·K
writes at record width). Where it has one, the cast still runs for the shared
expert while the dispatch reads the wide tokens, so the graph **under-bills**
the dispatch read by T·K × (activation width − record width).

Wire bytes are not correct in this mode either, and in the other direction:
low latency sends a token once per EXPERT it picked to keep the kernel
dispatch-free, while the graph prices the deduped dispatch — one send per
distinct destination — so the exchange is **under-billed** by up to 1.51x at
top-8 ([cost model](inference_cost_model.md#the-moe-exchange-sends-a-token-once-per-destination):
0.579 of the payload against the 0.875 a per-pick send moves at EP 8).
That is the modelled form's own assumption, not an oversight — the graph
prices the better implementation — and it matters here because sglang's
`--deepep-mode` defaults to `auto`, which selects low latency for DECODE.

**Fix.** A fusion plan that takes the cast, the permute and the exchange into
one DeepEP dispatch launch, with a binding conditioned on the all-to-all
backend. That backend is a deployment fact the graph does not state yet; it has
to be declared first.

### 3.3 The DP-attention dispatch exchange has no measured cell

**Today.** The exchange after a MoE dispatch moves the quantised tokens, so its
payload width (1.03125 bytes per element for FP8 with fp32 group-128 scales) is
not the activation width. A collective's measured cell is an element count
built at the activation dtype, so `benchmark_identity.CommLaunch.from_row`
refuses the row. The row stays in the measured-coverage denominator as
unplaced, and it states the format its payload is in
(`LeafAccess.output_activation_format`).

**Deviation.** This exchange is priced analytically only; no measurement
calibrates it.

**Fix.** A collective cell keyed by the payload's format, and a provider that
moves the elements together with their scales.

### 3.4 Fusion's closure test counts edge readers only

**Today.** `Fusion._closed` keeps a member out of a launch when something
outside the launch still reads its output, judged by the graph's edges.

**Deviation.** A member whose output is read past the edges (a cached tensor,
a stream a later layer reads, a published output) is not refused, and its
write could be dropped from the fused launch, which would **under-bill** that
write. This has not been observed on the model corpus, but nothing rules it
out.

**Fix.** The closure test should ask `traffic.TensorReaders`, which already
answers "who reads this tensor" for the graft.

### 3.5 A quantised write on a secondary port is not keyed

**Today.** A measurement key has one `output` slot, filled from the primary
port (`LeafAccess.output_activation_format`).

**Deviation.** A launch that writes a quantised tensor on a secondary port is
keyed as if that port wrote at the edge's dtype. The key states less than the
traffic bills, so a measured cell can stand for a launch it did not time.

**Fix.** Key every written port's format, not only the primary one.

### 3.6 DeepSeek-V4.1's layer seam is written into its spec, not matched across

**Today.** A launch that spans the seam between two layers is matched there:
the matcher follows the formal a layer reads into the iteration that wrote it
(`eval.fusion.Seams`, `docs/inference_cost_model.md`, "A launch across the
seam between two layers"). DeepSeek-V4.1's spec reaches the same launches
another way. Its blocks are written at an offset, so each template opens with
the previous layer's `hc_post` and carries the pre-mix across
(`mhc_shifted_post_pre`). That is a second mechanism for one phenomenon.

**Deviation.** None in the numbers today:
`TestTheShiftedSeamIsTheSameLaunchesOverACarriedPre` pins V4.1's counts. But
the offset spec puts layer 0's `hc_post` and the stack's last one where no
layer of the checkpoint has them, and a reader of the graph has to know
which of the two mechanisms a model uses.

**Fix (migration path).**
1. Write V4.1's blocks unshifted: `hc_post` back at each layer's tail, and
   the carried pre-mix as the stream the layer hands on.
2. Let the mHC plans meet `hc_pre`'s formal across the seam, as DeepSeek-V4's
   already do. The matcher needs nothing new for this.
3. Delete the offset, and check the pinned counts are unchanged.

---

## 4. Measured calibration and kernel declarations

Kernel providers declare what they serve in a closed-world `KernelSignature`
(`kernel_bench/providers/__init__.py`, `kernel_bench/ARCHITECTURE.md`):
whatever a declaration does not list, the kernel does not serve. The entries
below are cells no registered kernel can honestly serve, and facts the
declarations cannot yet express. In each case the fix is to declare the
missing fact or register the real kernel, not to widen an existing declaration.

### 4.1 The launch precision gate counts auxiliary tables

**Today.** `launch_trace.disagreement` reads the widest buffer a launch is
handed as the precision its row prices. A gathered constant table the cell does
not declare (a RoPE cos/sin table, a slot map, a page table) counts like the
payload. The RoPE oracles avoid this by cutting the table to the rows the
positions reach (`refs._rope_cos_sin_cache`), which changes no launch's work
but is not serving's `[max_position_embeddings, rotary_dim]` shape. When the
table ties the payload in size, any tied buffer at the declared precision
passes.

**Deviation.** The gate can either refuse a correct launch that is handed a
serving-sized fp32 table or pass a launch on the strength of a buffer that is
not its payload.

**Fix.** Let only the operands the walk declares for the cell
(`driver._declared_elements`) count, which needs the trace to match buffers to
declared tensors.

### 4.2 Cells whose output is a scaled FP8 record have no kernel

**Today.** A kernel that writes its activation's dtype does not serve a cell
whose output is an FP8 tensor with per-group scales, or a cache record it does
not read. These cells have no candidate on the H100 plans:

- `silu_and_mul_with_clamp` with an FP8 per-group output. vLLM's
  `silu_and_mul_per_block_quant` has no clamp operand, and the registered
  clamped kernels write bf16.
- DeepSeek-V3.2's indexer cells that read the cache's FP8 record rather than a
  plain tensor: `indexer_logits` at `kv_format=fp8_group128` (the torch floor
  reads plain keys; the DeepGEMM paged kernel is registered but disabled) and
  `fused_dsa_qkv_norm_rope_indexer` at `fp8_pow2_group128`.

**Deviation.** These rows are priced analytically only.

**Fix.** Register the real fused kernels and, for the floors, read the
record's scales.

### 4.3 Facts the closed declarations cannot yet state

- **Port and index dtypes are corpus labels, not walk facts.** A router writes
  fp32 weights and integer ids, `fused_gdn_gating` writes fp32, `topk` writes
  int64, and sglang's softcap writes fp32 logits, yet the cell keys these ports
  at the activation dtype. Until the walk materialises each port's dtype,
  `torch_topk` (`kernel_bench/providers/moe_gate.py`) declares any float output
  label, and the gating and router providers declare the label the cell
  carries. Two `applies_to` predicates in `moe_gate.py` are also not yet
  expressed as declarations.
- **FA4 takes q and kv at one dtype** (flash-attn-4 `interface.py`). Per-slot
  dtype lists cannot say "equal", so a cell whose query and cache dtypes differ
  passes the declaration and is refused by the library. The fix is a
  cross-slot constraint, if any such cell exists in the corpus.
- **Support that depends on the GPU generation.** sglang FA3's asymmetric V
  width (`d_v_head`) and FA4's attention sinks hold on Hopper only; the
  declarations state the H100 evidence and say nothing about other
  generations.
- **An undeclared `phases` means both phases.** The field is not checked
  against the phase vocabulary, so a misspelt phase silently widens a claim.

### 4.4 State-carrying cells that no registered kernel takes the state of

**Today.** A recurrent state's carrier (`cache.entries: state: … carrier:`) is
billed the state's read and write-back (`eval.traffic._settle_cache_reads`,
`_settle_state_writes`). A launch can therefore stand for that cell only if its
kernel is handed the state and writes it back. Each cell states whether it
carries state (`RowIdentity.carries_state`), and each kernel states whether it
takes it (`KernelSignature.takes_state`). The following cells have no provider
on the H100 plans:

- **The compressor pool as plain ops.** sglang pools DeepSeek-V4.1's
  low-ratio compressor in eager torch below SM90
  (`_low_ratio_compress_torch`), and DeepSeek-V4-Flash and Step-5-Preview plan
  the same `weighted` multiply unfused, because no plan groups vLLM's
  ratio-128 pool launch (`SparseAttnCompressC128Block8Kernel`). No `mul` kernel
  takes the state.
- **GLM-5.3's k-pool** (`fused_kpool_compress_store`). Both framework builds hand
  the kernel candidate windows of the cell's own keys, not the pool state.
- **Stateless floors.** `torch_depthwise_conv1d`, `torch_mamba2_scan` and the
  torch compressor floors start from a fresh window or zeros and keep nothing.
  The framework kernels serve those cells; a floor that read and wrote the
  state would only add another floor.

**Fix.** A plan and kernel for vLLM's ratio-128 pool launch; for sglang's eager
pool, a floor that gathers and rewrites the state beside the multiply; for the
k-pool, builds that hand the kernel the state rows as serving does. Longer
term, the operator contract could declare the state as an operand (a role
beside `kv`) so that `takes_state` is checked against what each build hands
the launch; today `mamba2_scan` and `causal_conv1d` name no state tensor at
all, and their builds allocate one.

### 4.5 A vector kernel's divisibility is declared at the bf16 constant

**Today.** `sgl_gelu_and_mul` serves bf16 and fp32 and declares
`divisible: d_ffn % 8`. Eight is `16 / sizeof(bf16)`: the kernel moves 16-byte
vectors and reads the `up` half at offset `d_ffn`, so the real requirement is
that `d_ffn · element_size` be a multiple of 16, which is a modulus of 4 for
fp32.

**Deviation.** fp32 cells whose `d_ffn` is a multiple of 4 but not of 8 are
declined although the kernel could serve them. This only arises when fp32
activations are chosen explicitly, since an fp32 checkpoint is served at 16 bits
by default (`dtypes.served_dtype`).

**Fix.** Declare the vector width in bytes and derive the modulus from the
cell's dtype, rather than adding a second constant per dtype. The gate guards
against a misaligned-address fault that poisons the CUDA context for the rest
of the run, so the relaxed form has to be confirmed on hardware before it
replaces the current one.

### 4.6 Primitive ops that a plan drops

**Today.** A measurement plan drops a per-op row whose op no signature
measures. That is correct for a view. For an op that really launches, it
leaves a hole. A signature, reference and floor are added only where an audited
serving path launches the op by itself; a missing binding for the surrounding
span is a coverage gap, not evidence of what a framework dispatches. `tanh` and
`signed_sqrt` meet that bar and are measured
(`tests/kernel_bench/test_activation_floors.py`). The others:

| op | launched by itself? | what is missing |
|---|---|---|
| `exp` | no: sglang fuses `fused_kda_gate`; vLLM's FlashKDA prepare takes the raw gate | an audited path that dispatches the primitive |
| `relu` | no: its only live use is the lightning indexer, and `indexer_logits` is bound unconditionally by every framework | an audited standalone launch, then a contract for its frame (the score matrix `[B, n_heads, s_query, s_kv]`), which the unary rows × feature contract does not describe |
| `clamp` | no: both bindings fuse `silu_and_mul_with_clamp` | a cell that carries the bound (`OpSpec.consts` names it, but no cell states it) |
| `add_const` | yes: hyper-connection gates add `hc_eps`, and no plan absorbs it | a name for its constant, so a cell can state it and an oracle can check it |
| `l2norm` | yes, audited: sglang's `chunk_kda` runs the q/k l2norms as two launches before the chunk kernel | a per-head contract; its frame is `[B, s, n_heads, d_head]`, normalised within each head, and the unary contract would normalise across `n_heads · d_head` |
| `scatter_rows_add` | probably: Qwen3-VL DeepStack's add of visual features into the decoder hidden states, absorbed by no plan | the same audit, plus a two-operand-with-index contract |
| `mrope` | no: `fused_qk_mrope` and the QSA rope plans are bound by both frameworks | an audited standalone launch |
| `group_max` | no: `indexer_block_score` is bound by both frameworks | an audited standalone launch |

Where an op is launched by itself but has no signature, its row is priced
analytically only.

### 4.7 Decode routed-expert cells not yet measured

**Today.** Under the padding rule of §2, a small decode batch keys `n_tokens`
below `n_experts`, and a mean above one row that is fractional keys a cell the
older measurements did not cover. Corpus rows measured under an earlier
convention, which padded every expert in the bank, key a bank size that
matches no current cell.

**Deviation.** Those cells have no measurement and fall back to the analytical
roofline. Cells with a large expert bank (Kimi-K3's 896 experts, the larger
DeepSeek-V4-Pro banks, some LongCat cells) cannot be built on a 24 GB card at
all, because the canonical input layout materialises every expert's weights.

**Fix.** Measure the missing cells, and for the large banks add a provider that
tiles or shards the weights, or measure on a larger card.

### 4.8 Collective communication is not measured

**Today.** The harness launches one kernel on one GPU. There is no
multi-rank launcher, so all-reduce, all-gather and all-to-all are always
priced by the analytical model (`inference/comm_cost.py`); the torch
`allreduce_fused_rmsnorm` provider is a disabled single-GPU stand-in.

**Deviation.** A deployment's measured total therefore cannot be complete
whenever it contains a collective, and communication time is never
calibrated against hardware.

**Fix.** A separate multi-rank sub-harness (`torchrun`) that writes rows
under the same physical key.

### 4.9 Some registered providers are disabled

**Today.** Several providers are registered but disabled — among them the
DeepInfer and DeepGEMM mega-MoE kernels, DeepGEMM paged MQA logits, the FP4
quantize kernels, Marlin GPTQ int4 and vLLM `causal_conv1d_update`
(`GET /api/bench/providers` lists them with their state).

**Deviation.** Cells those kernels would serve fall back to another provider
or stay unmeasured, so the envelope can be slower than the fastest kernel a
framework actually runs.

**Fix.** Enable each provider once its build and cross-check pass on a
supported card.

### 4.10 Rows from the previous timing method are kept

**Today.** Rows measured before the current L2-drain timing method
(`timing_method` empty) were kept rather than re-measured
([kernel_bench/ARCHITECTURE.md](../kernel_bench/ARCHITECTURE.md), "How a
launch is timed").

**Deviation.** One envelope can mix both clocks. The old clock's error is
bounded by one L2 write-back (about 17 µs on H100, 82 µs on RTX 4090), which
is small against most rows' latency but not against the shortest launches.

**Fix.** Re-measure the affected `kernel_impl`s and purge the old rows.

### 4.11 Promotion does not re-validate rows

**Today.** Promoting a batch to the shared corpus (`store.set_status`) does
not re-run the row validator, and the lookup refuses (fails closed on) rows
that fail today's validator.

**Deviation.** A validator change can leave invalid shared rows in the lake;
a lookup that selects one fails, and the estimate that asked for it returns
503 until the rows are deleted.

**Fix.** Validate at promotion, and re-validate shared rows when the
validator changes.

### 4.12 A lookup is bounded by its candidate count

**Today.** A lookup whose candidate set exceeds 50,000 rows is refused with
503 ([measured calibration §7.2](measured_calibration.md)).

**Deviation.** Very large sweeps over a dense corpus can lose their measured
answer although every row they need exists.

**Fix.** Reduce candidates in the store (per physical key) before the bound
is applied.

---

## 5. Spec representation

### 5.1 Selective state has no stream representation

**Today.** A cross-layer stream declaration takes `window:` (layer N reads at
most layer N − window), and the cycle badge shows it. The edge-kind vocabulary
includes `state` (`EDGE_KIND_VOCABULARY`).

**Deviation.** Mamba-style selective state, a state that mutates across
positions inside one layer, is not a cross-layer stream and has no graph-level
representation. Recurrent-state capacity is already computed correctly from
the cache declaration, so no number is affected today; the gap is in what the
graph can show.

**Fix.** Use the `state` edge kind for in-layer recurrent state once a
graph-level consumer needs it.

### 5.2 An edge leaving a `repeat:` bank is billed for one copy

**Today.** A copied node is evaluated once and multiplied, so every edge
leaving it is billed as one copy. `SpecIRNode.chained` records whether the
copies form a chain, but no cost consumer reads it yet.

**Deviation.** This is correct for `chain:` nodes and for edges inside a bank,
and it under-bills an edge that leaves a `repeat:` bank of two or more copies.
The affected edge today is the shared-expert bank read by `moe_out` in
DeepSeek-V2 and DeepSeek-V2-Lite (`n_shared_experts = 2`). On DeepSeek-V2-Lite
at prefill (B = 8, s_ctx = 2048, BF16, 26 MoE layers) the `moe_out` row is
billed 3,489,660,928 B against a correct 5,234,491,392 B (×1.5); the shared
operand alone is under-billed by `n_shared_experts`.

**Fix.** Rewrite the shared-expert banks as a leading expert axis, as the
routed banks already are; the edge bytes then follow from the shape and
`repeat:` can be removed. A rule that instead multiplies edges by the copy
count must also check that the consumer lies outside the bank, since
`repeat:` placed on every leaf of a bank (`moe_ffn_packed.yaml`,
`inkling_moe.yaml`) yields bank-internal edges whose factor is 1.
