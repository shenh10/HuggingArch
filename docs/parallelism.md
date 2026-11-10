# Parallelism and Sharding: One Axis Table, a Propagated State, Injected Communication

A model runs on many GPUs, and the same model can be split across them in many
ways at once — tensor-parallel attention heads, expert-parallel MoE, context-
parallel KV, pipeline stages, data-parallel replicas. HuggingArch models this
without enumerating the combinatorial space of configurations: one table
classifies which mesh worlds may shard each dimension symbol, each stored
weight declares which of them actually splits its axes, extents stay global,
per-rank shapes come from the layout on read, a distributed-tensor state
propagates forward along the dataflow, and the collectives that keep that
state consistent are injected as first-class graph nodes. This document
explains the knob surface,
the axis-kind table, the state-and-injection model, and how per-rank cost falls
out. It builds on the [evaluation model](evaluation_model.md) (sharding
propagates like shape) and feeds the
[inference cost model](inference_cost_model.md) (each rank's roofline).

## The knob surface: ParallelConfig

The user sets one `ParallelConfig`; every per-rank number derives from it.

| Knob | Splits |
|---|---|
| `tp_attention` | attention heads across ranks |
| `tp_dense_ffn` | a dense FFN's intermediate dim |
| `tp_moe_expert` | each MoE expert's intermediate dim |
| `ep_moe` | MoE experts across ranks |
| `cp` | prefill token stream (adds ranks). Does not shard KV storage; `layersplit` occupancy is `cp_scheme` |
| `dcp` | the KV cache along the sequence dim in decode; adds no ranks — its legal values are the subgroups the mesh can form, which with prefill CP on may span CP columns rather than divide `tp_attention` |
| `pp` | layers across pipeline stages |
| `dp_attention` | the batch across ranks that each hold full heads |
| `dp_replicas` | full-model data-parallel replicas |
| `sequence_parallel` | selects the sequence-parallel TP communication policy |
| `cp_scheme` | `pcp` (default; Q local, gather cache), `layersplit` (pcp geometry; per-rank KV occupancy `min(L, ⌈L/cp⌉+1)` where +1 is a scratch buffer for one remote layer; decode broadcast not priced), or `ulysses` (head all-to-all; modeled for comparison — vLLM and sglang serve pcp-style CP, so it is not the default) |
| `ep_load_factor` | not a deployment field: ρ = measured max/mean **expert** load, a workload input (`Workload.ep_load_factor`, the symbol `rho`) sent beside the deployment. The busiest EP rank holds the hot expert and `k − 1` others that share the remaining rows, so it carries `(ρ + (k−1)(E−ρ)/(E−1)) / k` × the mean rank's rows (`ExpectedRoutedLoad.busiest_rank_load_factor`; 1 without EP, ρ at one expert per rank); a routed tensor's per-rank rows are that rank's (`layout.physical`), and the route all-to-all moves them, whole |
| `world_degrees` | degrees for rank groups that have no knob of their own, keyed by world label; a world declared in `dims.yaml` runs at 1 until a deployment names it (e.g. the DSA indexer, which vLLM and sglang build whole on every rank) |

The degrees are not independent — the analyzer checks the consistency
constraints (`tp_attention × cp × dp_attention = attention_world_per_replica`,
`tp_dense_ffn × cp × dp_attention = dense_world_per_replica`, and
`ep_moe × tp_moe_expert = moe_world_per_replica`) and rejects any non-unity
group that differs from the per-stage size. `dcp` is not in those products:
it adds no ranks. There is no `dp_moe` knob; MoE ranks see every batch through
all-to-all routing. A model with no MoE dispatch region has no expert mesh:
`ParallelConfig.for_program` runs the axes only that region spans at 1.

## One axis table, and each stored weight's declared placement

Which parallelism *may* divide a dimension is a property of the *dimension*,
not of the component using it. `backend/arch/dims.yaml` is the authoritative
registry: it classifies every symbol — head axes and feature axes belong to
tensor parallelism, sequence axes to context parallelism, and
`n_expert_copies` to expert parallelism (`shardable_by:` on each symbol). The
same file declares the **worlds** themselves (`worlds:`): which axes live in
each, which mesh it is a subgroup of, and in which phases it is active (CP
splits the prefill token stream only). `resolve_dimensions` rejects an
authored model dim absent from the registry, and
`tests/backend/arch/test_eval_axis_kinds.py::TestAxisKindCoverage` checks the shipped corpus
and weight axes. `Parallelism.worlds()` reads the registry (`dims.world_labels()`)
to map symbol → mesh world; `Parallelism.policy()` is the degree table
the layout rule reads, not a divisor applied to the binding.

Activations need nothing more. A component that gives its projection
`dims: {out: "n_heads * d_head"}` has said everything the system needs about
the tensor that projection produces: `n_heads` is a head axis, so the per-rank
activation has `n_heads/tp_attention` heads. The negatives fall out the same
way — an MLA latent has *no* head axis in its cached tensor, so tensor
parallelism replicates it, and that conclusion is read off the dims rather
than off a per-component whitelist.

Stored weights are stricter: the table says which worlds *may* shard a
symbol, and each weight-bearing node declares, per stored axis, which one
*does* (`weight_axes`, `arch/layout/weight_endpoints.py`). An axis is either
`replicated: true`, or split on a `world` with a `parts` expression that
counts its pieces and a `fit` policy for a degree that does not divide them.
MLA's projections show all three shapes of the declaration:

    kv_a_proj_with_mqa  [{replicated: true}, {replicated: true}]                  # the latent has no head axis
    kv_b_proj           [{world: attention, parts: n_heads, fit: replica_cells},  # out axis split by head
                         {replicated: true}]
    o_proj              [{replicated: true},
                         {world: attention, parts: n_heads, fit: replica_cells}]  # in axis split by head

Nothing is inferred from a symbol being shardable: a stored tensor without a
declaration is rejected, and the declaration is checked against the table —
the `parts` expression must name exactly one symbol the chosen world may
shard. A fused checkpoint axis (a packed QKV) lists named `segments`, each
with its own extent and placement; a reusable component that binds one local
axis to different model symbols dispatches with `placement_by_symbol`; an
axis the checkpoint stores in a different order from the op's weight shape is
reordered by `stored_axes` in one place. Expert-copy tensors are the one
disjoint surface: their copy axis is declared by `weight_copies` and audited
by `copy_endpoints`.

Extents stay global. Per-rank geometry is the layout `Block`, and a degree
that does not divide a weight's part count is handled only as its `fit`
says: `exact` refuses the deployment at construction, `replica_cells` lets a
group of ranks co-own each shard (the head count is split `gcd(parts,
degree)` ways and replicated across the rest), and `pad` pads each shard to
whole parts (the vocab-parallel case frameworks actually run). There is no
silent "largest even divisor" on the binding, because that rewrite has no
replica-cell picture and the walk would be free to disagree with placement.

Expert parallelism is deliberately *not* a shape transform: `ep_moe`
distributes whole experts, so it folds the expert-copies **count**
(`n_expert_copies → n_expert_copies/ep_moe`)
while each expert keeps its full per-expert shape. Only `tp_moe_expert` shards
inside an expert. The two compose multiplicatively with no special case for
their combination. The count must divide the EP degree the same way a
weight axis must divide its TP degree.

## Sharding is a propagated state

Like shape, a tensor's distributed state is a first-class quantity propagated
forward along dataflow edges. A tensor's state per mesh axis is one of three —
**Sharded** (a 1/W slice), **Replicated** (whole on every rank), or **Partial**
(a per-rank partial reduction: full shape, incomplete values) — and the third
is why the state must be propagated rather than inferred from shape
differences: a partial has no shape delta to observe, yet everything
downstream of an unreduced row-parallel exit is wrong until a reduction
happens.

Which mode a weight-bearing op runs in — column-parallel, row-parallel,
replicated — is seeded from its declared `weight_axes`: a split on the
output axis is column-parallel, a split on the input axis row-parallel, no
split replicated:

    q_proj    out axis split on attention (parts: n_heads)  → column
    o_proj    in axis split on attention (parts: n_heads)   → row
    down_proj in axis split on dense_ffn (parts: d_ffn)     → row

The transition rule (`LayoutPropagation.transition` in
`arch/layout/propagation.py`; `eval/transition.py` holds only the records the
rule and the walk share) carries each node's state forward: a column-parallel matmul yields a head-sharded output; a row-parallel
one yields a partial sum that is not the true result until reduced. When the
state that arrives at an input is not the state the op's mode requires, that is
not an error to classify — it is a **`Redistribute`**, and its cost is the
collectives the state transition decomposes into, one per differing mesh world.
An op the layout rule cannot place is refused (the `analysis` validation gate),
not guessed at.

## Communication is injected, and first-class

The collectives are **not** in the spec YAML — parallelism is a runtime
choice, not a structural fact of the architecture. They are grafted after the
walk, at the state-mismatch edges, as first-class graph nodes: a partial sum
meeting a consumer that needs the full tensor injects an `all_reduce`; a
sharded tensor meeting a replicated consumer injects an `all_gather`; the MoE
token shuffle to and from expert ranks is an `all_to_all`; context-parallel
attention combines per-shard partials with a log-sum-exp reduction. Tensor
parallelism, expert routing, and context-parallel combine are the *same* state
algebra applied at different edges.

Because the injected collectives are real nodes they appear in the per-op
breakdown, carry their communication bytes, and contribute a `t_comm` term to
the roofline. The interconnect each one pays is chosen by mesh topology: a
collective inside one node uses the fast intra-node link; one whose group spans
nodes is priced hierarchically — a reduce-scatter inside each node, the
inter-node exchange over the slower link, then an all-gather inside each node
(`inference/comm_cost.py`) — so the same logical `all_reduce` costs differently
depending on where its group sits. Communication volumes live in
`arch/layout/comm_volume.py`.

Pipeline parallelism is placement. The canonical balanced partition cuts the
backbone's iteration space (the repeated layers of every stack that is not an
aux stack) into contiguous stages, ⌊n/pp⌋ layers each with the remainder from
the second-to-last stage backwards. Every other node's stage follows from the
dataflow, resolved per occurrence (one execution of a node at one iteration):
a unit whose every input is free (a parameter, a request input) or itself
pulled — the embedding, an encoder tower, what merges them — runs with its
earliest reader; any other unit — the final norm and head, an MTP head — runs
after its latest input. An aux stack is placed whole. Any edge whose two
occurrences sit on different stages is a hop: one `send_recv` on the `pp`
world per cut it crosses, once per step, sent once however many readers read
the value past that cut, as each rank's local tensor in the state it is handed
on in (a pending sum crosses as one). The hidden state crosses every cut; so
does an appended stream's element read past a cut (Kimi-K3's AttnRes bank, a
Qwen3-VL deepstack feature read by a decoder layer past the first cut) and a
inner stream's producer read past a cut (the DSA top-k a shared-indexer
layer reuses). Each reader's own transition from the handed state is its
ordinary input transition, on its stage, so a stage's in-stage communication
is what `pp = 1` makes. A stage pays the hops across its exit cut; its rows,
weights and cache are what placement puts on it, each within its own scope
(timing, weights, cache), and a table read through `tied_weight` is resident
where it is read.

## Per-rank cost: shapes divide, data parallelism does not

The [cost model](inference_cost_model.md) is evaluated per rank, and per-rank
work follows from per-rank *shapes* — the dims the shard policy divided —
never from a per-op divisor table maintained beside them. Data parallelism
(`dp_attention`, `dp_replicas`) divides nothing, because it replicates rather
than splits; the resulting `t_sol` is a per-group latency, so adding
data-parallel replicas leaves per-request latency unchanged and raises
aggregate throughput proportionally.

The unit is one replica everywhere a step is quoted, the unsharded column
included. The walk's `Analysis.cost` is each region's own whole work, and under
DP-attention the regions differ: an attention replica runs its own tokens, while
the expert bank the replicas share receives every replica's (`M` carries the
DP-attention degree), which is what the per-rank projection and the route's
measure preservation need. The module table's unsharded FLOPs read
`Analysis.replica_cost` instead — each leaf at one replica — so a tensor-parallel
and a DP-attention + expert-parallel deployment state the same per-layer figure.

Deriving bytes from per-rank shapes has a consequence worth stating as an
invariant. Ulysses and pcp context parallelism differ in *what is communicated*
and in what each rank holds of some tensors. Most producers owe the same
per-rank FLOPs; a K/V decompression producer on Ulysses's resident sequence cut
owes `1/cp` of PCP's work before the attention entry exchanges that cut for
heads. Per-rank activation bytes may differ only by the factors the two layouts
imply (query alike, KV ×cp under pcp, a per-token op with no head axis 1/cp).
`inference/invariants.py` (`_inv_cp_scheme`) reads the ratio from walked axis
identities rather than exempting rows by path, and
`tests/backend/inference/test_perturbation_invariants.py::TestCpSchemeInvariantActivation`
checks it on DeepSeek-V3, V3.2 and V4-Pro at `cp = 8`.

## The cases the model captures

- **MLA latent.** No head axis in the cached tensor, so TP replicates it;
  raising `tp_attention` does nothing for its KV; spreading it takes
  `dp_attention`, which gives each rank different requests. This is a fact read off the dims, not a
  per-component exception.
- **Context parallelism.** Whether `cp` shards the cache depends on the
  scheme. Under `dcp` (the decode deployment's "CP (DCP — KV shard)") and
  layersplit, it spreads cached *positions* across ranks (`positions_over`
  on the cache entry): at decode each rank attends its KV shard and the
  partials combine with an LSE reduction over `[B, n_heads, d_head_value]`.
  Under ulysses and pcp it divides no KV: prefill all_gathers K/V before
  the cache write, so every rank stores the whole sequence and per-request
  KV, max batch and TPOT are those of `cp = 1`. At prefill the split
  follows the Ulysses or pcp pattern (layersplit uses pcp geometry and
  folds stored layers).

  **Prefill CP is priced as the balanced split.** vLLM (DualChunkSwap,
  `vllm/v1/worker/gpu/pcp_manager.py:226,248-266`, v0.32.0) and sglang
  (`--cp-strategy zigzag`, `python/sglang/srt/layers/cp/zigzag.py:17-29`,
  v0.5.22) cut each prompt into `2·cp` chunks and give rank `r` chunks `r`
  and `2·cp − 1 − r`, so every rank owns the same share of the causal
  matrix. The engine prices exactly that: a rank's FLOPs are `1/cp` of the
  pairs (`physical.pairs_share`), and a law in the query count is
  re-evaluated over the two chunks each rank holds
  (`AttentionForm.query_runs`, `physical.rank_definitions`), then averaged
  over the balanced split:
  `(1/cp) · Σᵣ |reach(r) ∪ reach(2·cp − 1 − r)|`. For a sliding window
  that counts the keys each rank's queries reach rather than the whole
  gathered context. The `Block` layout of the query axis says how many
  queries the balanced average holds, not a busiest-rank wall-clock bound.
  sglang's
  `interleave` strategy (round-robin tokens) is not modelled either: there
  every rank's queries span the sequence and a window saves no K/V read.

  CP trades a combine
  collective for lower per-request latency, versus DP which trades none for
  higher throughput.
- **Expert parallelism.** `ep_moe` folds the expert-copies count; an
  `all_to_all` dispatches tokens to their experts and another gathers the
  results; the router weights are never sharded.

## Where this lives in the code

| Concern | Location |
|---|---|
| Knob surface + constraints | `arch/layout/parallel_config.py` (`ParallelConfig`; `inference/parallel_config.py` re-exports it) |
| Dimension and world registry | `arch/dims.yaml` + `arch/dims.py` |
| Public evaluation entry | `arch/eval/transforms.py` (`Deployment.analyze`) |
| Layout algebra + propagation | `arch/layout/propagation.py` (`LayoutPropagation`, `EdgeState`) |
| Stored-weight placement (`weight_axes`) | `arch/layout/weight_endpoints.py` |
| Pipeline stage placement | `arch/layout/stages.py` |
| Expected routed load (`ep_load_factor`) | `arch/layout/workload.py` (`ExpectedRoutedLoad`) |
| Collective materialization | `arch/layout/materializer.py` (`materialize_comm_tree`); events in `materialize_comm.py` |
| Communication volume | `arch/layout/comm_volume.py` |
| Collective cost formulas | `inference/comm_cost.py` |

## Terminology

| Code identifier | Literature term / source | Meaning here |
|---|---|---|
| `ProcessGroupFamily` | cf. MPI groups (no communication context attached); XLA replica_groups | a family of process groups over the ranks of one world |
| `refines` | partition refinement (order theory) | one partition of the ranks is finer than another |
| `orthogonal`, `joint_components` | — | two families jointly index ranks one-to-one (product-bijective per joint component) |
| `Block`, `pad_to` | HPF BLOCK distribution; owner computes | contiguous block distribution of one axis over a world |
| `replica_cells`, `replication_degree` | XLA replica groups; GSPMD | which ranks co-own a shard |
| `num_shards`, `shard_of_rank` | sharding (GSPMD/DTensor) | shard count / assignment |
| `Segmented` | analogy: HPF GEN_BLOCK (unequal contiguous blocks; ours adds per-segment layouts) | per-segment distributions, local concat |
| `Product` | analogy: multi-dim blocking (ScaLAPACK is 2-D block-cyclic; ours is single-axis ordered nesting) | ordered nesting = index map |
| `Replicated`, `Partial` | DTensor placements | whole copy on every rank / a pending sum across ranks |
| `owner_ranks`, `local_index`, `physical_local_extent` | denotation ⟦L⟧ | which ranks own an index and where it lands locally |
| `is_uniform_surjection` | fiber (preimage) of a map | every rank owns the same amount (uniform preimage) |
| `cross_replica_family` | XLA cross-replica reduction | the family a replicated value reduces across |
| `PAYLOAD_SCHEMAS`, `PENDING_MERGE_KINDS` | mergeable summaries; online softmax (Milakov & Gimelshein) + FlashAttention's weighted value | what a partial value carries and how two partials merge |
| `ReductionHomomorphism`, `may_defer_reduction` | homomorphism; reduction reordering | when a reduction may be moved past an op |
| `GROUP_INVARIANT`, `PARTITION_OF_UNITY`, `INDEX_EQUIVARIANT` | standard mathematical senses | coefficient classes |
| `REGISTERED_MERGE_STATISTICS`, `admits_deferred_reduction` | registration gate framed by Gray et al.'s classification (registration, not decomposability in principle) | runtime validation gate |
| `ExpectedRoutedLoad`, `routed_row_capacity_per_rank` | cf. GShard/Switch expert capacity (per-expert; ours is a per-rank total) | MoE routed rows as an expected per-rank load, not an index range |
| `is_measure_preserving` | measure preservation | conservation check |
| `validate_dispatch_region` | GShard dispatch/combine | well-formedness of a dispatch/combine region |
| axis_use `contract/normalize/window/recurrent/route` | tensor contraction; normalization; stencil/sliding window; scan/recurrence; MoE routing | per-(op, axis) dependence class |
| `shardable_by` | 自述词 | pure capability set: which worlds *may* shard a symbol; implies no division |
| `placement` | DTensor placements | which world actually shards one occurrence, at what degree |
| `Occurrence` | occurrence (one execution of a statement in a loop nest; polyhedral / dependence analysis) | one execution of a node: `(uid, iteration)` |
| `StagePlacement` | stage placement (pipeline-parallel partitioning) | the stage every occurrence runs on at `pp` stages, from the partition and the dataflow |
| `fit: exact\|replica_cells\|pad` | 自述词 | nondivisibility policy of a declared split |
| `carrier` | 自述词 | which mechanism carries the expert-copy count (`instances` or a stored axis) |
| `unrolled`, `leading_axis` | 自述词(社区口语 per-expert / stacked) | checkpoint encoding of the copy domain: one artifact per copy vs one stacked axis |
| `artifact` | 自述词(借建构词:a produced, stored thing) | one physically stored checkpoint tensor (primary or metadata sibling) |
| `provenance` | data lineage | where a header count came from: declared, recovered, absent |
| `oracle` | test oracle | an independent evidence source a declaration is checked against |
| `verdict` / `evidence` | 自述词 | checkpoint comparison outcome × what the header evidence showed (`copy_endpoints.py`) |
| cache slot `kind` (`slot_kinds:`) | 自述词(cf. vLLM `KVCacheSpec` subclasses, which classify per layer, not per tensor) | what one `cache.entries` item holds — `kv`, `latent`, `rope_key`, `index_key`, `compressed_kv`, `recurrent_state`, `conv_state`, `compressor_state`, `token_window`; the unit a deployment chooses cache precision by |
| `step_kv`, `s_live` | 自述词(cf. `kv_cache_view`, whose rows are cached; these are read and never cached) | a step's own keys as a KV stream: every query of the step reads its `s_live` rows, none are stored (DSv4.1 DSpark draft block) |
| dims kind `constant`, `OpSpec.consts` | 自述词(plain sense: a real-valued constant of a function, as a clamp bound or an activation β) | a symbol an op's math takes beside its operands that is no extent: may be fractional, no world divides it, unbound when the config does not publish it; read per node from the frame (`Analysis.consts`) |
| `RoutedLaunch` (`Analysis.routed`) | 自述词(cf. "padded / ragged grouped GEMM", "expert load") | one rank's MoE dispatch boundary as a grouped kernel launches it: tokens held, rows routed onto its experts (exact), its experts, and the padded launch — the experts that receive rows, each at `ceil(rows / touched)` (`eval/routed.py`) |
| `Trace`, `TracedFunctions` (`backend/code_trace/provider.py`) | verifying trace (Mokhov, Mitchell, Peyton Jones, *Build Systems à la Carte*, 2018; Shake, Salsa) | the code one computation executed — each outermost def named with the hash of its normalized source — against which a cached result is checked before it is served; `GlobalHash` is the degenerate trace of one global code hash |
| import-time trace (`TracedFunctions.global_trace`) | 自述词 | what importing the traced trees runs — module top levels, class bodies, functions called while importing — plus every method of a repo class with a base from outside the repo and the standard library (third-party caches can call it unseen), the recording machinery's own source and `uv.lock`; recorded once per source by a fresh interpreter that imports only the trees, so every process names the same one; one item, `<import-time>`, in every trace |
| `Growing` | 自述词(cf. Adapton's demanded computation graph, which tracks such a value per read) | a memo entry whose value is filled after it is built (`walk_registry.SharedWalkCells`): its trace is everything every computation that read it ran |
| `Folded.partial`, `t_partial_s` ("partial measured") | 自述词 | the step with each measured row at its measured time and every other row at its SOL time; never called measured, always shown with `partial_measured_share` |
