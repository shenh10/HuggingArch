# Inference Cost Model: Prefill, Decode, and the Roofline

The inference analyzer answers the day-to-day serving question: *can I run this
model on this card, and how fast?* — time to first token (TTFT), time per output
token (TPOT), decode throughput, and how many requests fit. All of it is a
speed-of-light (SOL) roofline built on top of the evaluation core's per-op FLOPs
and derived byte counts (the [evaluation model](evaluation_model.md)),
evaluated per parallelism rank for a chosen GPU. This document explains the two
phases, how each op's time is derived and combined, why the SOL is a per-op *sum*
rather than a model-level max, and how the same machinery yields the capacity
numbers. The KV side has its own [deep dive](kv_cache_storage.md).

## Two phases: prefill and decode

Autoregressive inference has two regimes with very different cost profiles. The
estimator (`inference/phases.py::estimate_phase`) runs the same graph twice with
different sequence bindings:

| | Prefill | Decode |
|---|---|---|
| What happens | process the whole prompt at once | generate one token per step |
| Query length `s_query` | the prompt minus any cached prefix | `n_verify`: 1, or K + 1 with K speculative draft tokens |
| Cache length `s_kv` | grows within the prompt | `s_cache + n_verify − 1`, with `s_cache = input + output/2` (the average over the generation) |
| Compute | `O(B · s_ctx² · d_model)` — quadratic in sequence | `O(B · s_kv · d_model)` — one query against the cache |
| Bound by | **compute** (high arithmetic intensity) | **memory** (reads weights + KV for ~1 token of work) |
| Headline | **TTFT** = prefill `t_sol` | **TPOT** = decode `t_sol` |

Both phases share the same memory footprint — the KV cache is cumulative — but
differ in activation peak (prefill scales with the prompt, decode with
`n_verify`), compute, and per-step communication volume. (Op templates write the
sequence axis as `s_ctx`; it is an alias of the phase's `s_query`.) A component
that declares a `decode_variant` runs it in decode — MLA's **absorbed** form,
which keeps the KV in the `d_kv_lora + d_qk_rope` latent space, DSA's and
DeepSeek-V4's compressors likewise — because that is the configuration real
serving stacks run (`rebind_decode_variants`, applied in
`inference/step_eval.py`); see the [KV deep dive](kv_cache_storage.md). A
sparse (DSA) MLA runs its absorbed form in prefill as well (`phase_form`), and
every page that shows a prefill cost — the estimate, the architecture
dashboard, the Inspector, the playground — reads that walk. What is stored
does not change with the form: the absorbed form keeps the one stored
`kv_b_proj` and runs it as two per-head GEMMs stored at its address (`module:`), so
every walk lists the file's tensors.

## The roofline: four bounds per op

Every leaf op carries three cost quantities from the engine: its FLOPs, its
memory traffic (weights + activations + KV read/write bytes), and — for the
collectives the parallelism model injects — its communication bytes. Against a
chosen GPU's peak compute, HBM bandwidth, and interconnect bandwidth, and the
two time floors a request states, each op gets four candidate times
(`row_roofline.BOUNDS`: compute, memory, comm, launch):

```
t_compute = flops / peak_flops_per_sec      (the peak at the row's matmul precision)
t_memory  = bytes / peak_hbm_bw_bytes_per_sec
t_comm    = a collective, per phase of its schedule (inference/comm_cost.py):
              payload × ring_factor / link_bw            (β: the intra-node link
                                                          inside a node, the
                                                          inter-node link across)
            + count × hops × link_latency                (α: the algorithm's hops,
                                                          comm_volume.ALGORITHMS)
t_launch  = count × launch_floor   for a row that is not a collective and does
            any work (FLOPs, activations, weights or metadata); 0 otherwise
```

An op's speed-of-light time is the **max** of the four:

```
t_op = max(t_compute, t_memory, t_comm, t_launch)
```

Compute and memory transfer overlap on real hardware, so an op is limited by
whichever bound is tighter, not their sum. A GEMM with high arithmetic
intensity is compute-bound; an RMSNorm moving more bytes than it does math is
memory-bound; an all-reduce is comm-bound; a tiny decode kernel whose work
takes less than a launch is launch-bound. The launch floor is a floor, not an
addend: it changes the answer only where the work is below it, and a large
kernel pays nothing extra for it.

**The two floors.** One link traversal's latency (`COMM_LATENCY_US_DEFAULT`,
2 µs) and the shortest one kernel launch takes (`LAUNCH_FLOOR_US_DEFAULT`,
1 µs) are declared once in `inference/defaults.py`; a request may set its own
(`comm_latency_us`, `launch_floor_us`). Both are estimates until Kernel Bench
measures empty and small collectives and launches.

**What the collective latency assumes.**

- *Inside one NVLink domain the framework picks the all-reduce algorithm
  by payload.* vLLM and sglang do not run NCCL's schedule on a tensor-parallel
  all-reduce there: below a threshold they run a one-shot kernel (every rank
  writes its payload to every peer and reduces locally — 1 hop, a rank
  receives `(N−1)·T`), above it a two-shot (reduce-scatter + all-gather by
  direct writes — 2 hops, `2(N−1)/N·T`), and past that NVLS where the switch
  reduces in the network (2 hops, `(N−1)/N·T`). The algorithms and their
  coefficients are `comm_volume.ALGORITHMS`; which payload takes which is the
  framework's binding (`collectives:` in `framework_bindings/*.yaml`, one
  representative threshold per algorithm, source-anchored); whether the card
  admits one is `gpu_specs.csv` (`nvlink_gbps`: an NVLink domain of
  `gpus_per_node` GPUs, an HGX node or an NVL72 rack; `nvls`: the switch
  reduces). `comm_cost.collective_ladder` puts them together, and a row's law
  is cases over its payload, so a sweep's other batch lands in its own
  algorithm. Buffer caps, environment switches and CUDA-graph tables are
  billed on the default serving path and stated in the binding comments.
- *Elsewhere, hop counts are the latency-optimal small-message schedule*
  (Thakur, Rabenseifner & Gropp 2005; `comm_volume.latency_hops`) — NCCL's,
  on PCIe cards, across NVLink domains and for every collective but the
  all-reduce: a point-to-point send is 1 hop; reduce-scatter (recursive
  halving), all-gather (recursive doubling) and all-to-all (Bruck)
  `ceil(log2 N)`; all-reduce (reduce-scatter then all-gather)
  `2 ceil(log2 N)`; a hierarchical collective adds its phases' counts. For
  large messages NCCL may pick a bandwidth-optimal ring with more steps, but
  there the β term dominates.
- *One link latency for every fabric.* NVLink, PCIe, InfiniBand and RoCE pay
  the same α per traversal — a deliberate simplification, so a hierarchical
  64-GPU all-reduce pays 24 traversals at the same α whatever links they cross.
  The bandwidth term still prices each phase on its own link.

**No overlap.** Rows are serial: the step is the sum of its rows, and no
compute hides a collective or another kernel's launch. Within a row the four
terms are alternative bounds, never added.

## Why SOL is a per-op sum, not a model-level max

The tempting shortcut — `t_sol = max(Σ flops / peak, Σ bytes / bw)` — assumes the
whole model is uniformly one-bound. Real models are not: a transformer layer
interleaves compute-bound projections, memory-bound norms and activations,
comm-bound collectives and launch-bound small kernels. A model-level max hides
the memory- and comm-bound work behind the compute total (or vice versa) and
*under-counts* the true time.

HuggingArch instead sums the per-op rooflines:

```
t_sol = Σ_op  max(t_compute_op, t_memory_op, t_comm_op, t_launch_op)
```

Each op pays its own tighter bound. The phase totals `t_compute_s` / `t_memory_s`
/ `t_comm_s` are per-op sums of each term (a row's `t_launch_s` is on the row),
shown as components: they are **not addends** of `t_sol_s`, which takes the
max per op before summing — `t_sol_s` is at least the largest component sum
and at most the sum of all four. `t_sol_s`
is exactly what the per-module breakdown table's **Total** row adds up — the two
numbers must agree by construction. There is no closed-form fallback: a phase
without per-op rows or a graph raises, and a time that is not published is
`None` with a `time_unpublished_reason` beside it. The page also splits the step
by the term that bounded each op (`t_sol_by_bound`: compute, memory, comm,
launch).

## Placement: per-rank shapes, not divisors

The roofline is evaluated **per rank**, and per-rank cost follows from
per-rank *shapes*: extents stay global, and the layout propagated during the
same walk says which block of each tensor a rank holds (see the
[evaluation model](evaluation_model.md)), so an op's FLOPs and its HBM bytes
are computed from the geometry each rank actually holds — there is no per-op
divisor table maintained beside the shapes for the two to disagree. Data
parallelism never divides work, because it replicates rather than splits;
communication bytes come from the collectives the
[parallelism model](parallelism.md) injects at state-mismatch edges. Because
`t_sol` is a per-group latency, a deployment with data-parallel replicas has
the same per-request latency but proportionally higher aggregate throughput.

## Pipeline stages: what one stage's rows count

Under pipeline parallelism a rank runs one stage, and every number it
reports is that stage's ([parallelism model](parallelism.md)):

- **Rows** count what placement puts on the stage: each of its layers, and a
  model-level step (the embedding, the final norm, the LM head) or an aux
  stack only on the stage it is placed on. A timed stack is counted; an
  untimed one (an MTP head without a draft, an encoder) contributes no row.
- **Hops.** A value read on a later stage than it is produced on is sent
  across every cut between them, once per step, as each rank's local tensor:
  one `send_recv` row per value per cut, paid by the stage before the cut.
  The row names the cut (`cut`), the occurrence that produced the value
  (`source_occurrence`: node and iteration) and the ones that read it past
  the cut (`sink_occurrences`). A hop serving only an untimed stack is not
  priced. Each reader's own transition from the value as handed on (an
  `all_reduce` of a pending sum, a free slice) is its ordinary row, on its
  stage.
- **Weights and cache** are the stage's own: its layers, the model-level
  steps placed on it, and every table it reads (a tied embedding is resident
  on the stage of the head that reads it). Each is counted within its own
  scope — `include_aux_weights` adds aux weights to the stage that runs the
  aux stack and widens neither the rows nor the cache.

Which stage a rank describes is capacity's memory question: the stage that
runs out first binds, and every per-rank number — weights, cache, rows,
phase FLOPs — is that stage's. The step time at `pp > 1` is published as
one microbatch's latency: the sum of every stage's roofline plus the
inter-stage hops (each stage's rows price their own exit cut, so hops land
exactly once). The fill/drain bubble is a steady-state throughput artifact
rather than a roofline term and is deliberately not modelled; measured
times stay unpublished at `pp > 1` (coverage is per-stage and would mix
bases), and when the per-stage sums are unavailable the phase declines to
publish a time at all rather than show a stage's.

## Fusion: collapsing memory round-trips

Kernel fusion merges adjacent ops so their intermediate activations stay in
registers/SRAM instead of round-tripping HBM. That removes memory traffic without
changing FLOPs, so it lowers `t_memory` for the fused region — and the breakdown
reports "fusion saves *X* ms vs the pre-fusion roofline". The fused op owns the
combined FLOPs and the reduced byte count; the per-op sum above is taken over the
*post-fusion* leaves, which is why the breakdown table collapses a layer's
projections into one fused row plus whatever comm could not be fused into it.

### What a fused row stands for

A fused row is one kernel launch. Where a framework's wrapper launches two or
three kernels, that is two or three plans, each exposing what it writes to HBM —
one node standing for several launches would trade a launch overcount for a byte
undercount.

Which launches a framework actually makes is stated in its binding
(`backend/inference/framework_bindings/{vllm,sglang,deepinfer}.yaml`),
audited at the
release named by `audited_at`. When a kernel path is unreachable because of a bug
in that release, the binding moves to a release where the path works; the plan is
not dropped to accommodate a framework bug.

### How precisely to model a dispatch condition

A framework decides which kernel to run from many facts: the phase, the
parallelism degrees, the GPU, the token count of the step, the size of an
all-reduce payload, an epsilon value, server arguments, which optional packages
are installed. Modelling all of them would mean carrying the serving stack's
decision tree inside the cost model, and most of those branches are invisible at
this granularity — the SOL razor of `CLAUDE.md` §1.4: a distinction the output
cannot show is not a distinction worth a concept.

So a dispatch condition is modelled (as a `when:` key) only when it changes bytes
or the main kernel path **and** triggers under common deployments:

| | Examples | Treatment |
|---|---|---|
| Changes bytes or the main path | phase, TP / world divisibility, GPU generation (`min_compute_capability`), framework, speculative decode | a `when:` condition |
| Only shifts launch counts at edge sizes | token-count thresholds (two vs three launches below 32 tokens, a 2048-token cap), all-reduce size caps, `norm_eps` values, split-K by step shape | bill the default serving path — prefill reads as large-token, decode as small-token — and state the assumption in the binding comment |
| Deployment choices | `--linear-backend`, `VLLM_DISABLED_KERNELS`, ROCm/AITER environment flags, batch-invariant mode, optional packages | bill the default install with default arguments, assumption stated |

One condition's keys are conjunctive. Where a framework takes one launch on
two paths that no single conjunction states — vLLM convolves Kimi-K3's packed
q|k|v row in one launch at decode only and GLM-5.3's in both phases — `when:` is
a list of alternatives, and the entry holds where any of them holds.

A stated assumption is not a silent fallback: the comment says which path is
billed and what would change it. The cost of this choice is a small error in
non-default configurations, almost always in launch count rather than bytes.

### How a pattern is written

A plan (`backend/inference/fusion_plans/*.yaml`) names the fused op and declares
the naive subgraph the kernel replaces. The declaration has three layers, defined
in `backend/arch/schema/fusion.py`:

A **pattern node** says which primitive op to match (`op`, or `op_choices` for a
family such as silu / gelu), its ordered `inputs` — each either
`external.<name>` for a tensor arriving from outside the pattern or
`<node>.<port>` for an internal edge — and two ways of being flexible about
arity. `optional_inputs` are trailing external operands a candidate may also
carry, which is how a softmax with an attention sink and one without share a node
in `flash_attention`; internal edges are never optional, since a missing producer
is a different subgraph. An `optional` node may be absent entirely, and then its
one reader reads the node's own input instead: the DSA indexer weights its heads
before summing them and the QSA indexer does not, yet both are one MQA-logits
launch.

Edges alone cannot always say which node is which. `fused_qk_rope` has two rope
nodes of the same op with no edge between them, and only the operand each one
becomes tells them apart. A node can therefore declare `feeds` — an input role
from `OpSpec.input_roles` (q, k, v, …). The matcher walks every path downstream of
a candidate, collects the role held at the first role-declaring op on each, and
binds when the declared role is anywhere in that union, so a rope shared by K and
V still satisfies `feeds: k`. Two candidates that both satisfy one slot are an
ambiguity the matcher refuses rather than settling by enumeration order, and a
candidate whose union also holds a role another pattern node feeds binds neither
slot — a tensor that becomes both q and k is neither.

A **naive pattern** is either an explicit DAG of named nodes or, for a
variable-arity group, a `parallel` form: a count range of sibling nodes of one op
over shared inputs, which is how a merged QKV or gate/up projection is expressed
without naming each sibling. Either way an `io_mapping` ties the fused op's
operand and output names to places in the pattern, `output` being the primary
write.

The **plan** adds the fused op's own contracts: `allocates_output` (the launch
writes a buffer of its own, so a `slice` / `concat` it absorbed is not read as
lanes passed through in place) and `output_dtype`, a map from output port to width
(a kernel whose secondary outputs are wider than its primary one — fp32 gates
beside bf16 activations — states each port). A launch that writes its output
quantised states nothing: its pattern names the `quantise_activation` cast as a
member, as for any other op it absorbs.

The loader rejects a pattern naming an unregistered op, an `optional` node that is
not a pass-through on one internal edge, and two bound plans of one framework
whose pattern is identical, since nothing but file order would then decide
between them.

### When two plans want the same nodes

Plans overlap by construction: Kimi-K3's whole-mixer kernel contains the short
convolution, the gated norm and the recurrence, each of which is also a plan of
its own. Three rules settle it, in this order.

Bindings are sorted by pattern size at load (required nodes; an `optional` node
does not make a pattern larger), so the larger plan is tried first and
the kernel a framework really launches wins over its parts. During the rewrite a
candidate whose nodes include one already absorbed is skipped, which keeps every
node in exactly one launch. And where the larger plan's condition does not hold —
Kimi-K3 under sglang at an attention-TP other than 8 — its members are free and
the smaller plans match as before.

The weak spot is equal-size plans: specificity cannot order them, so file order
decides, and a pattern that happens to fit two different places in a graph is kept
apart only by a larger plan claiming one of them first. Where that matters, say it
with a condition or a `feeds` role rather than relying on the order.

### Activation epilogues follow their consumer

An elementwise activation epilogue (`silu_and_mul` and its siblings) writes its
output at the precision the consuming GEMM reads: its plan takes the
`quantise_activation` cast after it as an optional member. This is one rule
rather than a per-framework fact, and the graph decides when it applies.

The serving graph grafts a cast only where a reader needs a format the edge does
not carry (`inference.materialize_cast`, design in [quantization](quantization.md#activation-casts-in-the-serving-graph)):
one quantised copy per tensor and format, standing right after the tensor's
writer, moved upstream past element copies that keep the scale-group axis
(views, non-reducing collectives, whole-row gathers; `traffic.ElementCopy`)
when it is all that reads them. The writer's launch takes the cast in only when
it is the output's one reader (the match's closure), so a consumer that wants
the row wide keeps it apart, and a BF16 deployment has no cast to take. A MoE
dispatch is such a copy: the experts' copy is made once per token on the
dispatch's token input, and the dispatch and the exchange after it under expert
parallelism move it at the record's width (elements and scales,
`traffic.payload_width`). The shared expert and the routed experts read the one
copy; a framework that quantises the row twice (vLLM's shared-expert linear
beside its MoE prepare step) runs a launch the graph does not bill — the graph
models the better form. Where serving quantises after the dispatch (vLLM's
CUTLASS NVFP4 fallback per expert, DeepEP high-throughput for a non-block
format, experts that take unquantised input) the default serving path is
priced; DeepEP low latency quantises inside its dispatch kernel, a fusion
recorded in `docs/known_limitations.md` §3.2.

## The MoE exchange sends a token once per destination

Every other collective's per-rank bytes are a function of its group's size:
`(N-1)/N` of the payload, doubled for an all-reduce. A routed MoE exchange is
not, because its payload rows are one token's **top-k picks** and several
picks can land on the same destination. A dispatch sends such a token ONCE to
that destination and lets the receiver fan the copy out to the experts it
holds — DeepEP's dispatch, and the form every serving stack converged on.
Billing a copy per pick over-counts DeepSeek-V3's top-8 dispatch by 1.51x
(0.875 against 0.579 of the payload at EP 8).

So the exchange's **total** is the ring's coefficient generalised over
distinct destination RANKS (`comm_volume.RouteFanout`): every one of them has
to receive the token once, whatever the topology. With `k` picks drawn from `T`
route targets of which `E` hold weights, over `D` destinations each holding
`E_d` of the targets:

```
k_eff   = k·E/T                                 copies the payload holds per token
A(D,E_d) = D·(1 − C(T−E_d, k)/C(T, k))          distinct destinations reached
factor  = (D−1)/D · A(D,E_d)/k_eff              bytes sent per payload byte
```

`A` is hypergeometric: a destination is missed only when all `k` picks avoid
its `E_d` targets. At `k = 1` the two terms cancel and the coefficient is
`(D−1)/D` — the flat ring, unchanged, which is how every non-routed
all-to-all stays priced exactly as before.

**The levels partition that total; they do not add to it.** The deployment
places the route group at two levels (`comm_cost.placement_levels`): the ranks
inside one node, over NVLink, and the nodes inside the world, over RDMA. The
node level counts its own destinations — one RDMA send per distinct remote
node — and the rank level takes the remainder
(`comm_cost.routed_shares`): the destinations in the sender's own node, plus
the forwards inside each remote node past the rank the RDMA copy landed on,
which is what DeepEP's internode dispatch does. So the bytes a row publishes
and the bytes its phases add up to are one number, and a group spread over
more nodes never gets cheaper — copies move onto the slower fabric, they do
not disappear. A level with one destination is not a level, so a single-node
group has only the first and a group with one member per node only the second;
the degeneracies fall out of `D ≤ 1` rather than being tested for.

Latency follows the same split: `ceil(log2 D)` per level, so a single node
keeps the flat `ceil(log2 N)` (deduping inside a node forwards nothing
further) and a cross-node exchange pays `ceil(log2 nodes) +
ceil(log2 members)` — the NVLink forward hop counted. The combine mirrors the
dispatch in bytes and in hops, and the direction is not modelled. It mirrors
it only while it IS an exchange: where the expert ranks already hold every
token (`route_owner_inclusion`), the inverse dispatch is an all-reduce, which
carries no fan-out and is priced flat — the oracle counts the fan-out-carrying
legs per deployment so which it is, is on the record
(`test_route_dedup_oracle`, `Case.routed_legs`).

The levels are read off the CARD, not carried from the walk, because the same
group of eight ranks is one node on an SXM board and two nodes of four on PCIe
cards. A world that is not a whole number of nodes has no node level — its
nodes hold different numbers of the group's members, so no per-node
destination count applies to any of them — and takes one level instead,
carrying the whole total on the inter fabric. The bytes are the same; the
fabric is pessimistic, and this is the one placement the model does not level.

**No per-framework switch.** Every framework and both phases dedup, and the
flip side has to be said as plainly as the group-limit gap below: a stack that
runs the naive per-pick form is **under-counted** here, by up to the same
1.51x at top-8 — we price the better implementation, so a deployment that does
not run it looks faster than it is. DeepEP's low-latency mode is such a form
(it sends per expert to keep the kernel dispatch-free, and
`docs/known_limitations.md` §3.2 records its other deviation). It is not the
optional switch an earlier draft of this paragraph called it: sglang's
`--deepep-mode` defaults to `auto`, which is normal for prefill and
**low-latency for decode**, so on that stack the decode exchange's default
path is the non-dedup one. Modelling it as deduped is a deliberate choice —
the graph models the better form, the way it does for the doubly-quantised
shared expert above — not a claim about what every default does.

### Known gap: group-limited routing is not modelled

DeepSeek's and Kimi's routers do not draw their `k` picks from all `T` targets.
The targets are partitioned into `n_group` groups, the router keeps the
`topk_group` best of them, and all `k` picks come from those
(`config.n_group` / `config.topk_group`; `topk_method: noaux_tc`). Since the
groups are laid out contiguously and expert parallelism cuts the same axis
contiguously, a kept group lands on few destinations — at DeepSeek-V3's EP 64
over 8-GPU nodes, one group of 32 experts is exactly one node. So a
group-limited token reaches FEWER distinct destinations than a uniform one, at
both levels, and this model over-counts its bytes. The direction is always
pessimistic.

How much: DeepSeek-V3 at EP 64 over 8 nodes (`n_group = 8`, `topk_group = 4`)
is priced with a node-level share of 0.579 of the payload where the true
figure is about 0.397 — a token can reach at most 4 of the 8 nodes, never 8,
so the dispatch's RDMA bytes are over-counted by about 46%.

Closing it needs the same formula marginalised over which groups were kept.
With `g` groups of `T/g` targets each, `t` of them kept, and a destination
holding `E_d` targets spread over the groups as `(e_1 … e_g)`:

```
A(D,E_d) = D · (1 − Σ_S P(S) · C(Σ_{j∈S}(T/g − e_j), k) / C(t·T/g, k))
```

summed over the `C(g,t)` kept sets `S`, each equally likely under a balanced
router. Two facts have to be DECLARED before that can be written, and neither
exists today:

1. **the grouping itself** — `n_group` and `topk_group` reach no spec dim
   today (the router component binds `n_expert_groups` / `n_groups_active` for
   its own FLOPs, but the dispatch's fan-out never sees them), so the
   dispatch's `RouteFanout` would gain two counts and the spec would have to
   bind them at the `permute` node the way it binds `n_experts_active`; and
2. **how groups sit on destinations** — `e_j`, which is a placement fact, not
   a count: it needs the expert axis's group structure declared beside its
   shard (today the axis is a flat `n_expert_copies` that expert parallelism
   cuts, and nothing says a group of it is contiguous).

Until both are declared the dispatch is priced as a uniform draw, and the
hypergeometric form above is where to start. The oracle that checks the
current form (`tests/backend/inference/test_route_dedup_oracle.py`) simulates
a uniform draw for the same reason: it checks the model as stated, not the
router as shipped.

## From per-step time to the headline numbers

- **TTFT** = prefill `t_sol` (one prompt through the compute-bound phase).
- **TPOT** = decode `t_sol` per token: `t / (1 + α·K)` under speculative decoding,
  whose step already includes the draft's forward.
- **Decode throughput** = `effective_tokens_per_step / decode t_sol` — the honest
  cross-model comparison at each model's own max batch (TPOT at differing max
  batches is not apples-to-apples). Speculative decoding with K draft
  tokens per step (`speculative_draft_tokens`) verifies them in one forward
  over K + 1 positions and raises `effective_tokens_per_step` to
  `(1 + α·K)·B` (`phases.Speculation`); K = 0 is plain decode.
- **Throughput** is published per group, per cluster (× `dp_attention` ×
  `dp_replicas`) and per GPU. A model-level MFU is not published; utilization
  appears per measured kernel row (see [measured calibration](measured_calibration.md)).

## Capacity: how many requests fit

The roofline says how *fast* a step is; capacity
(`inference/capacity.py`) says how *many* concurrent requests
a rank holds, and the two ceilings differ by phase:

- **Decode is KV-bound.** After weights and static overhead, the residual HBM is
  the KV budget, and each request costs `per_request_kv(seq)` bytes — so
  `max_decode_batch ≈ (kv_budget − static) / per_request_kv(seq)`. This is where
  the per-node KV model of the [KV deep dive](kv_cache_storage.md) feeds in: a
  windowed class caps its contribution at the window, a compressor grows at
  `⌈s_kv/compression_ratio⌉`, a linear/SSM state is constant. The cache held is
  the backbone's; a request that counts the speculative draft
  (`speculative_draft`) also holds the MTP stack's cache, which attends over the
  same positions. Under pipeline parallelism that cache sits on the stage the
  MTP stack runs on (the last).
- **Prefill is activation-bound.** The KV of a prefilling request accumulates
  *after* the step, so the ceiling is the activation pool — the HBM outside
  `gpu_memory_fraction`, minus overhead, plus the KV pool not yet in use —
  divided by peak activation bytes per request, at `min(input,
  prefill_chunk_size)` tokens under chunked prefill.
- **Weights are the floor.** A deployment that cannot hold weights + one request's
  KV on a rank reports `max_batch = 0` with an `infeasibility_reason` (the
  weights do not fit, or the KV budget cannot hold one request), and publishes
  no step: each phase is only its name, no batch and the reason, with no per-op
  rows, module table, encode step or fusion delta (`capacity.without_step`). The
  walk behind it still runs — at one request — because every other card and
  batch is re-priced from it. The SOL surface golden (`scripts/dump_sol_surface.py`)
  pins that walk; the capacity, payload-field and Scaling-payload goldens pin
  what is published.
- **Published context.** A request whose prompt plus output runs past the
  model's published context (`s_max`) is computed but flagged
  (`beyond_max_context`, `max_context_tokens`).
- **MoE weight reads saturate.** A step reads `min(demand(B), weight_total)` of
  the expert weights over the leaf's rows — a routed leaf's are the busiest
  EP rank's, read off the batch's skew `ep_load_factor` (ρ).

## A matmul whose weight is wider than its activation

A matmul that multiplies a weight held wider than its activation **computes**
at the weight's dtype — the type promotion of a mixed BF16 × F32 product, one
rule for every op that declares itself a matmul (`OpSpec.gemm`, a custom op's
`is_linear_map`), never a list of op names (`traffic.weight_promotion`).
DeepSeek-V4 and Hy4 store their hyper-connection mixing matrices `hc_*_fn`
F32, and MiniMax-M2.7, MiMo-V2.x and LongCat-Flash their router gates: those
multiplies run on the F32 MMA, so the row's FLOPs are timed at the card's FP32
column and its measurement key says `activation: fp32`. The operand's own edge
keeps its width — the upcast happens as it is loaded.

**It does not widen what the matmul writes.** The width of a tensor is its
edge's declaration; the promotion is about the arithmetic. Both frameworks'
kernels narrow in registers as they store — vLLM `triton_ihc.py:297-299` and
sglang `hy4_ihc.py:325` allocate the iHC post's output at the sub-block
operand's dtype, and Hy4's iHC pre, post and `hc_head` each end in
`.to(hidden_states.dtype)` (sglang `models/hunyuan_v4.py:154`, `:222`,
`:321`) — so a promoted product's key says `output: bf16`, nothing downstream
turns F32, and the graph carries no cast node for a narrowing no kernel
launches. Pricing the product wide instead sent F32 down the whole residual
stream after it: Hy4's prefill then came out 5.2% over at TP1 and 14.0% over
under EP8 with DP attention, all of it tensors the hardware never writes.

So the two sides of a mixed matmul are two declared facts, and they are read
apart: *what the multiply runs on* is the promotion (and, through
`ResolvedPrecision.compute_dtype`, which of the card's columns the kernel
needs); *what the launch stores* is the edge, an op's own `output_dtype`
(DSA's int32 top-k indices), a cache entry's format, or a plan's port
declaration.

The weight is the one the leaf streams, at its resident format: the
checkpoint's declaration (`quant_context`, `config.torch_dtype`) unless a
precision profile converts it. A weight stored BF16 stays BF16 however a
framework loads it — GLM-5.3-Flash's `hc_*_fn` is BF16 on disk and vLLM makes
it an F32 parameter; DeepSeek-V4's and Hy4's router gates are BF16 on disk, so
both are priced at BF16 and neither gate GEMM is promoted. That V4's top-k
kernel REQUIRES those logits F32 — which is why the router GEMM is built
`out_dtype=torch.float32` — is a separate fact about the OUTPUT, declared by
the binding and billed on the edge ("Stored width vs loaded width" below); it
moves no weight and no peak column.
Reading the *resident* format is also what keeps this
rule off a checkpoint served narrower than it ships: gemma-2 stores F32 and
both frameworks load it BF16, so its resident weights are BF16 and nothing
promotes (`shipped_model_dtype` against `model_dtype`). Only weights that
store values as-is rank by width; a quantised weight is dequantised into the
activation's dtype.

## Stored width vs loaded width

A weight's bytes come from `Representation.resident` — `intent` where a precision
profile converts the tensor or the framework loads it narrower, the checkpoint's
`saved` format otherwise. One case of "the GPU holds something else" IS modelled:
the model's own width. Nothing ships running wider than 16 bits and nothing is
ever widened (`dtypes.served_dtype` — one monotone rule, with the two loaders
read at their pinned releases as corroboration), so every unquantised tensor of
gemma-2's float32 checkpoint carries a BF16 intent against an F32 saved format,
while a BF16 checkpoint stays BF16 whatever its loader materialises. What is left
is the PER-TENSOR case — a framework that runs one tensor family at a width of
its own while the model dtype stays put. There are two such objects, and they are
not the same object.

### The width a dispatched kernel REQUIRES of a role's output: modelled

An edge is priced at the width its producer implies — a BF16 gate weight times a
BF16 row writes BF16 logits. Sometimes the kernel that reads that output exists
only at a wider one, and the framework builds the producer to meet it. vLLM
builds DeepSeek-V4's router as `GateLinear(out_dtype=torch.float32)`
(`vllm/models/deepseek_v4/nvidia/model.py:724-729`) because `can_use_dsv4_topk`
requires fp32 logits (`fused_moe/router/dsv4_topk.py:22-24`) and
`fused_topk_bias` falls through to another kernel without them; sglang's hash
top-k kernel takes `const float* router_logits`, and `MoEGate.forward` writes
fp32 on every CUDA path it takes (`srt/models/deepseek_v2.py:573-587`) to
supply it.

The declared fact is the REQUIREMENT, not the habit. A framework that writes a
role's output wider while the kernel downstream would take either width
declares nothing: that is an inefficiency, and we model the better form rather
than mirror a framework (the standing rule elsewhere in this document). The
loader holds declarations to it — an entry's anchor must cite two places, where
the framework writes the width and where the kernel requires it, and a single
`out_dtype=` is not enough.

The framework declares it, because the framework's dispatch decides it:
`edge_dtypes:` in `framework_bindings/*.yaml` (`FrameworkBinding.edge_dtypes`),
selected by the DECLARED ROLE plus the facts a binding's `when:` already asks —
the framework's own dispatch key, its card, its flags. Never a tensor name,
never a model id. A `when:` here may ask only what resolving a width actually
answers (`_EDGE_DTYPE_WHEN`, derived from the kernel-dispatch vocabulary minus
the two keys resolved after a width): a condition the evaluator could not read
is refused at load, rather than silently holding, and because the same keys are
what the overlap proof runs on, "one role, one width per deployment" is a proof
and not a wish. The binding answers both halves of the `when:` and hands the
walk plain data (`edge_dtypes_for`, as `ssm_chunk_size_from` hands it a field
list), which `ResolvedPrecision.with_required_edge_dtypes` materializes onto the
serving analysis once, before anything there reads an edge; `edge_dtype` is then
the single answer behind that walk's edge bytes, the row's `act_dtype`, the
cell's key, the peak column and the oracle's inputs. It is materialized AFTER
the walk, which is also the limit of its reach — readers of a framework-free
analysis, the precision panel among them, see the checkpoint width; that is the
debt recorded in the next section. **A role no entry names carries no
requirement and keeps the width the checkpoint implies** — that is the standing
assumption, and it is stated in `edge_dtype` where it is read.

Two things this deliberately is not. It is not `traffic.weight_promotion`, which
covers a resident weight held WIDER than its edge and promotes the dtype the
matmul COMPUTES in: here the weight and the operand agree at BF16, the peak column
stays BF16, and only the output moves. And it is not a fusion plan's
`output_dtype`, which applies where that plan fired — a role's output has one
width under a deployment, whatever fuses around it.

Measured on DeepSeek-V4-Flash, stated beside the deployment it was measured in
(the rule "Compute width changes a row's identity and its time, not its
traffic" sets out — a width figure without its deployment is not a figure).
The logits edge goes from two bytes an element to four, which at the pinned
single-GPU case is 4096 tokens x 256 routed experts x 43 MoE layers x 2 =
90,177,536 extra bytes written and the same read: +180,355,072 of 422.5 GB per
prefill step (4.3e-4). That case publishes no time (its weights exceed one
H100), so the time was taken on the pinned eight-GPU case: prefill t_sol
+0.059%, decode +0.034%.

Unlike a compute-width change this moves TRAFFIC as well as the row's identity,
because the edge is what widened. It also moves one peak column: the top-k tail
that reads the logits (`ffn.topk_scale__*`) is now judged against F32 rather
than BF16, which is the honest column — it really does compute on fp32 logits —
and is immaterial because that row is memory-bound (5.3 MFLOP against 256 fp32
logits a token). The router GEMM itself keeps the BF16 column, since its
operands are still BF16.

The byte arithmetic is asserted from the checkpoint config in
`tests/kernel_bench/test_router_logits_width.py`, so the number is checked
against something other than the engine that produced it. **The pinned
`sol_surface` rows for the three V4 cases move by exactly this and need
regenerating.**

Scope: the DeepSeek-V4 classes, under both frameworks — and that is the whole
set, not a first instalment. sglang's `MoEGate` is the gate for its entire
DeepSeek-V2/V3/V4 and GLM MoE family and writes fp32 for all of them, but only
V4's top-k kernel REQUIRES fp32; the grouped V3/GLM top-k takes either width
under both frameworks. So V3 and GLM have nothing to declare here and their
router cells keep the width they build (`refs._moe_gate_inputs`) — the right
answer rather than an unfinished one, because following sglang into an fp32
write its own kernel does not need would be mirroring an inefficiency.

### The width a framework LOADS a resident tensor at: still unmodelled

DeepSeek-V4's hash router reads a token→expert table shipped as I64
(`layers.X.ffn.gate.tid2eid`, `[129280, 6]`). Neither framework launches on that
width: sglang allocates it `torch.int32` and its kernel verifies int32
(`srt/layers/moe/hash_topk.py`, `kernels/jit/csrc/deepseek_v4/hash_topk.cuh`), and
vLLM takes int64 only under the explicitly chosen native MegaMoE backend
(`models/deepseek_v4/nvidia/model.py`; `moe_backend` defaults to `auto`). So the
walk bills the gather and the resident table at 8 bytes an element where the
default kernel reads 4.

Measured, on DeepSeek-V4-Flash: 98,304 B over-billed per prefill step out of
553.5 GB (1.8e-7), 96 B per decode step out of 11.3 GB, and 9.3 MB of resident
weight out of ~150 GB. It is a correctness defect of a kind, not a material cost
error, and the blast radius is two models and one tensor family — V4-Flash and
V4-Pro's `tid2eid`. The packed expert banks that also differ between disk and
register (FP4 in a U8, INT4 in an I32) are a *different* pair, already modelled:
`TensorFormat.storage_dtype` against `logical_dtype`.

The same width is why the leaf cannot be keyed for measurement: its environment
names I64, which has no benchmark-corpus token, so the row is declined
(`RowIdentity.leaf_dtype_env` says why). Giving I64 a token would key a
measurement of a table no default deployment allocates, and a provider that
allocated one would be refused by sglang's own matcher. The decline is the
modelled fact.

**What is left, precisely — and it is one piece of work, not two.** A framework's
facts reach the analysis only AFTER the walk has run
(`serving.build_serving_layers` materializes them onto a copy). Two consequences
follow, and the same threading removes both.

*A resident width cannot be declared at all.* `edge_dtypes` works where it does
because `edge_dtype` is asked lazily, after the walk. `tid2eid` is a RESIDENT
tensor: its width lives in `Representation.intent`, and `Formats` is resolved
inside `Evaluator.__init__` from `precision.weight_intent`, before any framework
binding is in hand. Nothing is missing from the declaration's *shape* — "output"
and "stored" would be two values of one axis in one family, and a `tensor:` key
would be a cheap addition. The axis is absent only because it would have a single
value today (one value is not an axis), and the fact it would need could not be
answered where it is read.

*A declared edge width is invisible to the precision panel.* `edge_dtype` carries
it for the walk's own billing, the row's `act_dtype`, the cell's key and the
oracle, because those run on the serving analysis. `Analysis.precision_effective`
does not, because both published routes into it build a framework-free analysis:
capacity's budget walk (`_capacity_walk`, no framework argument) and
`/precision/effective` (which constructs its own `Deployment`). So an
`/inference/estimate` response states `activations: all → BF16` in its `precision`
block while the per-op rows of the SAME response show the V4 router at F32, and a
precision-profile pin on `router` is overridden by the framework with no row
saying so. The panel is the authority on effective values, so this is a real
defect and not a display nicety; it is recorded here rather than patched in the
panel, because a panel that re-derived the width itself would be the second
derivation path this layer exists to prevent.

The threading both want: the framework's facts resolved WITH the profile, inside
the walk — a field on `Deployment` carried through `step_eval`,
`pipeline.analyze`, the plan and the API routes, which also retires
`ResolvedPrecision.with_launches` and `with_required_edge_dtypes` and the
post-walk `replace`. That is the work, and it is not done.

Two things that are NO LONGER the blocker, so they should not be cited as one:
the corpus token (I32 has one — `dtypes.py` gives it deliberately, for the index
tensors a top-k writes), and the span's measurability (the hash router has a
signature, `moe_topk_hash_sqrtsoftplus`, and a reference builder). Declaring the
loaded I32 width would fix the billing *and* make the row keyable in one move;
giving I64 a token would still only hide the first half, exactly as before.

**Reopen when** that threading is wanted for its own sake (the panel above is a
second reason to want it), or when a second resident tensor family turns up whose
loader narrows it. The shape is unchanged
from what was written here before: `Representation.intent` is the slot, the
binding declares the width because the framework decides it (vLLM's MegaMoE
backend is the proof — the same op loads either width), the walk takes plain
data, selection is by the declared role of the table and never a tensor-name
pattern, and an undeclared tensor bills its stored width as a stated assumption.

## Inductor's generic fusion: measured, then left unmodelled

Every fusion this model prices is a NAMED kernel: a plan says which naive ops one
launch replaces, and a binding anchors that launch to a `file:line` in a framework
release. vLLM also fuses work for which there is no such kernel. The decision taken
was to leave it unmodelled and say so, because it cannot be anchored, and to record
what that costs.

Inductor is on by default — `optimization_level=O2` (`vllm/config/vllm.py:427`)
reaches `VLLM_COMPILE` (`:1752-1756`) with backend `"inductor"`
(`compilation.py:461-466`) — and, less obviously, vLLM's own CUDA CustomOps are
OFF by default: `custom_ops` appends `"none"` (`vllm.py:1765-1772`,
`custom_op.py:292-307`), so RMSNorm, SiluAndMul, GeluAndMul and RotaryEmbedding
take `forward_native` and are handed to Inductor. vLLM states this itself:
"Inductor generates (fused) Triton kernels for disabled custom ops"
(`compilation.py:489-492`).

Three mechanisms hide behind that, and only the third is this section's. Named
pattern passes are anchorable and ARE modelled. A single disabled CustomOp's
self-contained `torch.compile` does not fuse across ops — vLLM says so — so its
launch count equals eager's and nothing is unobservable at SOL granularity. The
third is the model-level piecewise graph, cut only at `_attention_ops`
(`compilation.py:768`): adjacent elementwise work written in a model file collapses
there.

### Why it cannot be anchored

The fusion happens on a DIFFERENT graph — Inductor's own post-lowering IR — chosen
by a greedy multi-round scheduler (`scheduler.py:3697-3729`) ranking candidates by
a score computed from the CONCRETE size hints present at trace time
(`sizevars.py:725`, `graph.py:644-662`). vLLM additionally enables `combo_kernels`
with `benchmark_combo_kernel` (`compilation.py:986-1002`), so some kernel counts
are settled by benchmarking on the card at compile time, register spills included
(`scheduler.py:6905-6960`). "Adjacent in the spec graph" therefore does not imply
"one launch", and there is no kernel to point at.

The case FOR modelling it was checked and is real as far as it goes: vertical
fusion is neither benchmarked (`benchmark_fusion=False`, `config.py:780`) nor
score-gated (`choices.py:583-591` returns True unconditionally), so a simple
adjacent pair almost certainly does fuse. It still fails on three independent
counts. The pattern language has no form for "a variable-length vertical chain
over an op class" — `schema/fusion.py:211-266` has fixed DAGs and horizontal
`parallel` — so enumeration would mean 21-plus plans restating one rule. All three
`source_anchor` candidates would be false: a model file's line proves two ops
exist, `Scheduler.fuse_nodes` is not in vLLM, and a config default is a deployment
fact rather than a launch. And the fact a `when:` would need is not a fact about
the MODEL, which is the line between this and conv1d's `dilation`: `dilation` was
declarable because a conv1d genuinely has one.

That last point generalises, and it is the unwritten premise behind "declare the
fact first rather than guessing": the fact has to be about the model. A fusion
score, an ATen buffer's read count, a Triton tiling, a compile-time benchmark —
these are facts about the compiler and the machine, and no declaration in
`dims.yaml` or a component can hold them.

### What the silence costs

Measured over 63 corpus models, H100 TP1, 4096-token prefill, BF16: the activation
bytes on maximal adjacent-elementwise chains are **1.44% of total prefill access
bytes** (42.99 GB of 2985.7 GB) and **0.00% of decode's**, whose bytes are weights
and KV. By launches it is 390 of 8328 in prefill and 300 of 7911 in decode. So the
walk OVERSTATES launches and activation round-trips by that much, in prefill only.

**21 of the 63 models are exactly zero** — the whole classic corpus, Llama,
Mistral, Qwen2.5, Qwen3, GPT-OSS and DeepSeek-V3, has no such chain. It
concentrates in a few frontier graphs: DeepSeek-V4.1-Flash 6.61%, Step-5-Preview
3.97%, Qwen3.8-Flash-Next 3.47%, Kimi-K3 2.40%. The shape to express would be 186
maximal chains over 21 distinct op multisets of length 2 to 5.

Decode's half is a launch-floor effect, worst case about 57 microseconds a step —
and `LAUNCH_FLOOR_US_DEFAULT = 1.0` carries its own comment saying it is an
estimate until Kernel Bench measures it. Modelling decode's share would be
refining a term multiplied by an unmeasured constant.

There is a precedent for the narrow case: `fusion_plans/fused_add_rmsnorm.yaml:45-63`
already names Inductor and quotes vLLM's "otherwise Inductor handles fusion", and
concludes that both paths launch the same number of kernels so no condition key is
needed. This section exists because here they do NOT launch the same number.

**Reopen when** any of three things changes. If the pattern language grows a form
for a variable-length chain over an op class, the 21 multisets become one rule
rather than 21 plans. If Inductor's scheduler gains a deterministic, readable
decision — a documented rule rather than a score over trace-time hints — there is
something to anchor. Or if a corpus model's share stops being a rounding error:
1.44% is below the uncertainty of the launch floor it would interact with, and a
model above, say, 10% would change that judgement on its own.

## Calibration: from peak to achievable

A peak-bandwidth + peak-FLOPs roofline is an upper bound — the *speed of light*,
not a promised latency; a real GEMM reaches some fraction of peak. That only
holds if the peak really is a bound, which is a constraint on what goes in
`gpu_specs.csv`: a compute cell is the card's dense peak at the clock it
actually runs tensor-core work at, not the clock a spec sheet is computed at.
The two agree on a datacenter part specified at a clock it holds, and part ways
on a GeForce part, whose rated boost clock is a floor GPU Boost clears — an
RTX 4090 runs bf16 GEMMs at 2.73 GHz against a rated 2.52 GHz, so the sheet's
165.2 TFLOPs sat 8% under the hardware and 1,439 corpus rows read up to 105% of
"peak". The
calibration overlay (`inference/calibration_overlay.py`) looks up each op's
measured time in the corpus by its physical identity (`inference/op_calibration.py`),
corrects for GPU peak drift and flags anomalies, and publishes the result as a
`t_measured` track beside the SOL track — never overwriting it. Measured times
are not published at `pp > 1`. The corpus is produced by the
[kernel_bench harness](measured_calibration.md).

Every per-op roofline is also kept as a sympy expression in batch and GPU
(`inference/row_roofline.py`), so the batch and parallel sweeps substitute new
values rather than re-walking the model.

## Where this lives in the code

| Concern | Location |
|---|---|
| Phase split, per-op roofline, PhaseMetrics | `inference/phases.py`, `inference/row_roofline.py` |
| Decode-variant swap per phase | `inference/step_eval.py` → `arch/eval/transforms.py` |
| Per-op FLOPs (the walk) + HBM bytes (derived traffic) | `arch/eval/` → `inference/pipeline.py` |
| Capacity / max-batch | `inference/capacity.py` |
| Fusion | `arch/eval/fusion.py`, `inference/fusion_plans/`, `inference/framework_bindings/` |
| Calibration overlay | `inference/calibration_overlay.py`, `inference/op_calibration.py` |
| Collective pricing | `inference/comm_cost.py` |
| Sweeps | `inference/batch_sweep.py`, `inference/parallel_sweep.py` |
| Per-request KV storage / read laws | `arch/eval/cache.py`, `arch/eval/cache_view.py` (see [KV deep dive](kv_cache_storage.md)) |
| Parallelism placement + collectives | `arch/layout/` (`parallel_config.py`, `materializer.py`) |
| GPU peak specs (compute / HBM / interconnect) | `inference/gpu_specs.csv` (loaded by `gpu_specs.py`) |
