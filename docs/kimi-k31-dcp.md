# Kimi-K3.1 DCP scope and SGLang runtime audit

HuggingArch models the SGLang TP8/DCP8 sparse decode reference path with
explicit selected-position workload geometry, replicated runtime projections,
candidate exchange, and projection-before-combine. This is a target-step
analytical model, not a prediction of DFLASH acceptance or full server throughput.
The supported profile uses eight ranks, `fi_a2a`, BF16 dense projections,
FP8 cache, Q replication, V preprojection, and a prewarmed 1,536-query-row push
communicator. Other DCP degrees and non-SGLang serving profiles are rejected.

This audit uses SGLang main
[`1b2870d33de7b5f31844fd74e97eea6d7896c53a`](https://github.com/sgl-project/sglang/tree/1b2870d33de7b5f31844fd74e97eea6d7896c53a),
BF16 dense projections, `flashmla_kv`, TP8/DCP8, and the recipe's `fi_a2a`.
The recipe's original pin is a different commit. The facts below describe the
installed main and must be re-audited when that source changes.

## Cache ownership and execution contract

For each request, global key position `p` belongs to DCP rank `p % 8` at local
position `p // 8`. Storage is striped; there is no requirement that the query's
best 2,048 keys be evenly striped. The whole-model growing cache is
`24 × (656 + 132) = 18,912` bytes per token, or an average of 2,364 bytes per
rank at DCP8, before page rounding. Fixed KDA state follows attention TP, not
the DCP position cut. See the cache-writer probe and K3.1 tests for the physical
656-byte MLA / 132-byte index layouts.

In decode/target verification SGLang:

1. Scores all **local** index keys. It does not gather the full index-key cache.
2. Selects local top-k candidates, exchanges score/position candidates and
   determines the exact global top-k (with specialized fused/merge variants).
3. Compacts only this rank's selected slots and supplies their valid count to
   sparse MLA. Output index buffers remain padded to width 2,048 even when a
   rank has fewer valid slots.
4. Computes partial attention for the heads of the DCP group and merges
   partial outputs using their LSE statistics.

These are source contracts in
[`dsa_dcp_sharded.py`](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/layers/attention/dsa/dsa_dcp_sharded.py#L15),
[`dcp_local_topk`](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/layers/attention/dsa/dsa_dcp_sharded.py#L492)
and the indexer
[decode dispatch](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1636).
Prefill is different: cached index keys can be gathered in request order by
[`_dcp_gather_index_k`](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1984).

At context length 32,768, a query's 2,048 selected keys can all belong to one
rank. That rank then reads 2,048 selected MLA rows; the others read none. The
balanced mean is 256. The model exposes `dcp_selected_load_factor` (rho,
1 through 8) and uses `min(ceil(rho * min(S, 2048) / 8), ceil(S / 8))`
valid positions on the busiest rank. Rho=1 is an optimistic balanced scenario;
it is not an observed distribution. The selected-index capacity stays 2,048.
The indexer scans `ceil(S / 8)` local keys even for contexts below 2,048.
Cache capacity still grows with the whole context, independently of top-k.
Verification uses a fixed positive query width and sums each query's causal
selected/index pairs, including contexts crossing stripe and top-k boundaries.
Compiled serving cells include that width in their identity; changing it
requires another walk. The padded tensor remains rectangular. Index-key HBM
traffic is a shared-key lower bound, not a claim that every query reloads K.

The generic candidate fallback allocates `[rows, 2048, 2]` int32 values per
rank: 16,384 bytes per query, with an all-gather result of 131,072 bytes at
DCP8. These are buffer sizes, **not a claim about B300's wire traffic**: the
fused top-k, 384-candidate merge, prefix compaction, multicast and signaling
paths have different physical traffic.

## Replicated Q weights and resident bytes

With the recipe's `fi_a2a`, K3.1 main enables `dcp_replicate_q_proj=True`
[automatically](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py#L235).
[`ModelRunner._prepare_replicated_q_proj`](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/model_executor/model_runner.py#L1174)
creates full-head Q and absorbed K/V weight buffers. Decode and target verify
project full-head Q locally, skipping the per-layer query all-gather. CUDA
prefill continues to use the TP-local Q view; the resident replicas still
occupy memory in both phases. Merely making the checkpoint's Q weights
replicated in the model spec would therefore misstate both phases and the
checkpoint placement.

The following are raw persistent tensor bytes for one DSA layer on one rank.
They exclude the unchanged remainder of the model, allocator rounding/reserve,
page pools, graphs, draft, communication scratch and transient gather buffers.

| Allocation | Shape / relationship | Bytes |
|---|---|---:|
| Original TP-local `q_b_proj.weight` | `[2304,1536]`; aliases replica after setup | 7,077,888 logical |
| Full `q_b_proj_qrep_weight` | `[18432,1536]`, BF16 | 56,623,104 |
| Original `kv_b_proj.weight` | `[3072,512]`, BF16; retained | 3,145,728 |
| Local `w_kc` + `w_vc` | two separately contiguous absorbed buffers | 3,145,728 |
| `w_kc_qrep` | `[96,128,512]`, BF16 | 12,582,912 |
| `w_vc_qrep` | `[96,512,128]`, BF16 | 12,582,912 |
| CUDA `_dcp_v_weights` | `[96,512,128]`, BF16; a separate lazy allocation | 12,582,912 |

The original Q view and the full replica share storage: **count 56,623,104
bytes once, not that plus 7,077,888**. Conversely, at this commit CUDA's lazy
`_dcp_v_weights` does not reuse `w_vc_qrep`: they are distinct allocations.
The latter is consumed by the ROCm path; CUDA uses `_dcp_v_weights`.
Local absorbed weights are created by
[`post_load_weights`](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/models/kimi_k3.py#L3871),
while the lazy CUDA replica is created by
[`_prepare_dcp_v_weights`](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L172).

Across the **24 DSA layers**, per rank:

| Comparison | Extra bytes | GiB |
|---|---:|---:|
| Q replication vs the same TP8 model with local absorbed weights, before lazy V preprojection | 1,793,064,960 | 1.669922 |
| Lazy CUDA V preprojection, additionally | 301,989,888 | 0.281250 |
| Both above vs the local-absorbed baseline | 2,095,054,848 | 1.951172 |
| Both above plus local absorbed copies vs checkpoint-only TP8 accounting | 2,170,552,320 | 2.021484 |

The first row is the isolated Q-replication effect. V preprojection can also
occur when Q replication is off, so do not attribute the third row solely to
that flag. The last row identifies known runtime storage absent from a plain
checkpoint allocation count; it is not a complete server memory correction.

[The CPU storage oracle](../scripts/probe_kimi_k31_dcp_storage.py) executes the
pinned source's replica-allocation methods, replacing only the distributed
all-gather and type guards with CPU stand-ins. It verifies distinct storage
pointers and aliasing at the actual tensor shapes. It does **not** measure GPU
allocator memory. Its recorded output is
[`evidence/kimi-k31-dcp-storage.json`](evidence/kimi-k31-dcp-storage.json).
Run with the SGLang environment (which provides torch):

```bash
source /tmp/scratch-space/k3-performance/env.sh
cd /tmp/scratch-space/k3-performance/HuggingArch
/opt/sglang/bin/python scripts/probe_kimi_k31_dcp_storage.py \
  --sglang-source /tmp/scratch-space/k3-performance/sglang \
  --output /tmp/scratch-space/k3-performance/logs/k31-dcp-storage-oracle.json
```

## Projection and communication differences

K3.1 sets `dcp_v_preproject=True`. Under the CUDA path's eligibility conditions,
SGLang applies `W_UV` to **all 96 heads' local attention partials before the
merge**, reducing their width from the latent 512 to the value-head width 128.
It then combines to 12 local heads per rank. This shifts the projection's
compute before the collective and increases that per-rank GEMM's head count.
The existing dense DCP model merges latent-width partials first; applying it
unchanged would price the wrong compute and collective.
[Source and conditions](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py#L940).

Per query and DSA layer, with BF16 partials and FP32 LSE:

| Buffer / logical payload | Bytes |
|---|---:|
| Full-head Q `[96,576]` | 110,592 |
| Full-head unprojected partials + LSE, `96 × (512×2+4)` | 98,688 |
| Full-head projected partials + LSE, `96 × (128×2+4)` | 24,960 |
| Push-combine slots, `96 × (128×2+16)` | 26,112 |
| Remote push-combine slots per sender, `7 × 12 × (128×2+16)` | 22,848 |

The push kernel uses a **16-byte LSE word per head**, rather than the 4-byte
raw LSE value. These last two rows are the declared slot geometry, excluding
synchronization, transport framing and additional memory reads/writes; they
are not measured NVLink transfer counters or a collective latency estimate.
[Slot declaration](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/kernels/ops/attention/dcp_push_combine.py#L30).
The push path is conditional on registered communicator capacity; otherwise
SGLang uses packed A2A plus local LSE combination. CUDA-graph and verification
padding changes the number of physical query rows. With 5 speculative tokens
and 12 local heads, a 1,536-row setting allocates `1536 × 12 × 272`
bytes per peer slot. It therefore covers 1,536 physical query rows, or
`floor(1536 / 5) = 307` unpadded requests. The previous 25-request statement
divided by the local head count twice. The user-provided recipe is preserved
verbatim; this is an erratum to its explanation. Graph padding and forced
communicator size overrides can change the effective live-request limit.
CPU boundary evidence is preserved under `/tmp/scratch-space/k3-performance/logs/probe_dcp_push_bounds.py`.

## Execution model and reproducibility

The declared `sparse_dcp` component contract resolves concrete graph nodes.
The same binding drives physical shapes, weight aliases, cache access and
communication; it does not change checkpoint tensor placement. Decode executes
96-head Q and absorbed Q/V projections on every rank; prefill retains its
12-head TP-local projection geometry. The output is combined after projection
to 128 values/head, then returns to 12 heads/rank. There is no dense-DCP query
all-gather, latent LSE all-gather or latent reduce-scatter in this path.

The default SGLang path has fused DCP top-k disabled. It multicasts 384
score/index pairs (8 bytes/pair) per query and owner. Its exactness certificate
may require remote full top-k reads. `dcp_failed_owners` (0 through 8) is the
number of failed owners, not a boolean and not the selected-position skew.
Zero is an explicit no-fallback scenario, not a measurement of this run.
The timing model uses the failed sender's hotspot payload and a second dependent
transfer stage whenever any owner fails. It retains mean recipient and maximum
receiver bounds separately. One transfer stage is a direct peer exchange, not
an NCCL ring or tree round count. The projected-output event includes the local
FP32 LSE merge arithmetic and successful slot reads/writes; polling is excluded.
Communication bytes are logical peer payloads, not measured NVLink counters; synchronization
polling and transport framing are excluded.

Reproduce the balanced/no-fallback target-only scenario:

```bash
.venv313/bin/python scripts/estimate_kimi_k31_sglang.py \
  --dcp 8 --ep 2 --batch 1 --input-tokens 2048 --output-tokens 512 \
  --dcp-load-factor 1 --dcp-failed-owners 0 \
  --output /tmp/scratch-space/k3-performance/logs/k31-dcp8-2k.json
```

Repeat with 16,384 input tokens, and use `--dcp-load-factor 8` and/or
`--dcp-failed-owners 8` to inspect adverse scenarios. These are sensitivity
bounds; without selected-count/certificate instrumentation they must not be
presented as observed runtime conditions.

Resident weights include the 2,170,552,320 additional bytes/rank documented
above. This ledger does not include every server allocation: process-wide
candidate/push workspaces, graph pools, allocator reserve, HiCache, and KDA
checkpoint pools require runtime sizing. They must not be multiplied by the
24 DSA layers as if they were per-layer weights.

The capacity scope is raw tensor occupancy, before page allocation. With
physical page 64 and DCP8 the virtual allocator page is 512 tokens: a 17-token
request occupies 3 raw striped rows on its busiest rank but reserves 64 physical
rows per rank. The reported raw max_batch is not the runtime request-pool cap.
See the [cache-page boundary](kimi-k31-runtime-precision.md) for the source and
local-row formula.

The B300 target-only validation and comparison with the earlier DFLASH
benchmark are documented in the [DCP8 validation report](kimi-k31-dcp-validation.md). HTTP TPOT
includes scheduling and host overhead; GPU profile timings are reported
separately. Neither is obtained by multiplying DFLASH TPOT by acceptance.

Prefill gathers the complete packed index and MLA caches before computing
its local heads. The two collectives use equally padded rank slices, then
trim to the logical context. The model charges this communication and full
cache reads while preserving unpadded useful compute. Persistent striped
cache capacity rounds each request up to whole rank slots. Both phases
therefore support non-divisible context lengths in this profile.

### Source-declared local runtime stages

The DCP candidate event includes mandatory local merge-buffer traffic and
minimum integer radix visits. Three additional actual launches are represented
per DSA layer: selected-owner localization, FlashMLA windowed scheduler metadata,
and local FlashMLA split combine. The original apply producer writes FP32 split
partials only for rows that split; unsplit rows write their final BF16 output.
The combine stage reads the partials and writes final BF16 output and FP32 LSE.
One-split rows return before combine arithmetic and data traffic, while the
combine launch and split-prefix reads remain.

`SparseWindow` is a closed generated-law operator, with the same bounded CPU
scheduler for integer evaluation, symbolic substitution and compiled exact,
float and vector readers. It processes a declared repeated owner pattern and
hardware partitions; it does not infer actual top-k ownership. Independent
source-oracle tests cover mixed verify patterns and the native-head partition
boundary. Flattened query rows have `s_q=1`; native96 heads on SM148 use74
partitions for physical row counts below74 and148 otherwise. A single row with
256 valid local keys uses4 splits, not74. A row with2048 local keys uses32.

SM count comes from the selected GPU. Unknown hardware retains only explicitly
labelled `unresolved_sm_count; known_traffic_floor_only` rows. No B300 SM count
is substituted for another device. The profile assumes no extra-topk or
attention sink and an unset `FLASHMLA_DIVIDE_PARTS` override. SGLang is pinned
at1b2870d3; the separately audited FlashMLA source85b5264b matches native-head
API probes, but the installed1.0.5 wheel has no commit metadata, so exact build
provenance is not asserted.

Arithmetic is a mathematical scalar-work floor (FMA=2). Integer scans/radix
operations are reported separately, not priced as FP32. Score-dependent radix
passes, repeated synchronization polling, SFU throughput and transaction/cache
line amplification remain uncalibrated. The local MLA temporary allocation
upper bound is `(physical_rows + scheduler_parts) * heads * (value_dim+1) * 4`
bytes plus scheduler metadata. It is reusable scratch, not persistent KV cache;
the new runtime traffic accounting does not claim to measure allocator peaks.

The windowed scheduler is admitted only when SGLang's precompute guard
`4*(5*physical_rows + 1 + 8*parts) <= 48*1024` holds. With B300/head96 this
allows2220 flattened rows and rejects2221; outside that range SGLang selects
a different native scheduler. This analytical implementation explicitly rejects
that fallback at numeric or symbolic-point evaluation instead of applying the
windowed split counts. Verification widths count toward flattened rows.
