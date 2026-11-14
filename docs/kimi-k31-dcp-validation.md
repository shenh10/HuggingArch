# Kimi-K3.1 sparse DCP8: B300 validation

The implementation follows the [SGLang execution contract](kimi-k31-dcp.md).
It now produces target-step estimates through the public capacity API. Its
numerical result is an analytical lower bound, not a calibrated serving-time
prediction: the measured target-only server is materially slower.

## Matched target-only serving experiment

Eight B300 GPUs, TP8/DCP8, EP2/expert TP4, `fi_a2a`, FP8 cache, breakable prefill,
16K chunks and HiCache160 use the same installed SGLang commit as the source
model (`1b2870d33de7b5f31844fd74e97eea6d7896c53a`). DFLASH, its draft, and
linear ReplaySSM are removed to isolate target-only serving. Server and client
seeds are 20261113. Decode graphs remain enabled. Each input length has three
repeats, warmup, cache flush, concurrency one and 512 generated tokens.
The 60 measured requests all succeeded with exact lengths, zero prefix hits
and no compilation in measured intervals.

| Input | Mean decode context | Requests | Analytical target step | HTTP TPOT mean | Three-repeat TPOT range |
|---|---:|---:|---:|---:|---:|
| 2,048 | 2,304 | 36 | 7.096 ms | 9.945 ms | 9.9437–9.9464 ms |
| 16,384 | 16,640 | 24 | 7.096 ms | 10.081 ms | 10.0809–10.0812 ms |

The baseline assumes balanced selected ownership, zero candidate-certificate
fallback and balanced expert routing. These were **not observed distributions**.
The model is about 28.6% and 29.6% below HTTP TPOT respectively. HTTP TPOT
includes scheduler, sampling and streaming overhead; GPU profiles below keep
their own scope. No empirical multiplier is applied to make the numbers agree.

The longer context increases theoretical index work and cache bytes, but their
rows remain below the generic kernel launch floor. Thus the current baseline
predicts zero context-dependent latency increase, while HTTP TPOT increases
by 0.136 ms. This comparison validates neither the generic launch floor nor
peak-bandwidth efficiency for these kernels. It identifies the part requiring
kernel-specific calibration; source-correct geometry alone does not supply it.

## Sensitivity, not measured distributions

At input16K/output512/batch1:

| Selected max/mean ratio | Failed certificate owners/query | Analytical target step |
|---:|---:|---:|
| 1 | 0 | 7.096 ms |
| 8 | 0 | 7.096 ms |
| 1 | 8 | 7.147 ms |
| 8 | 8 | 7.147 ms |

Skew changes FLOPs and cache traffic even where the timing remains under the
launch floor. Fallback adds the busiest sender's full-top-k peer traffic and
another dependent transfer stage. A boolean fallback flag cannot supply the
failed-owner count. The compiled model keeps these two controls independent.

The earlier [DFLASH/EP/prefill-graph matrix](kimi-k31-b300-benchmark.md) remains
valid as a complete-serving comparison, but its TPOT is not a single target
forward. Multiplying it by reported acceptance length does not remove draft,
verification, scheduling or sampling costs and is not used here.

## Independent GPU profiles

A continuous capture recorded all eight ranks, two prefills and fourteen decode
steps per rank. Every decode contains 2,385 attributed kernels and 24 instances
of each explicit sparse-DCP stage. Kernel-to-step attribution uses CUPTI
correlation IDs rather than CPU annotation wall intervals.

| Input | Profile decode contexts | Mean GPU span | Mean union busy | Mean kernel-duration sum | Sum of 24 scan→push layer spans |
|---|---|---:|---:|---:|---:|
| 2,048 | 2,049–2,055 | 10.0920 ms | 9.8619 ms | 14.4532 ms | 1.6124 ms |
| 16,384 | 16,385–16,391 | 10.3252 ms | 10.0961 ms | 14.7799 ms | 1.7813 ms |

The separate theory evaluations at these profile midpoint contexts (2,052 and
16,388, output8) also give 7.096 ms. GPU spans already exceed that lower bound,
so HTTP overhead cannot explain the gap. Kernel sums exceed spans because
streams overlap. Profile timings include profiler perturbation and are not
substituted for the unprofiled serving measurements above.

Rank-local sums across the 24 DSA layers, per decode step:

| Observed kernel group | 2K | 16K |
|---|---:|---:|
| Local indexer logits | 0.1165 ms | 0.1190 ms |
| Local top-k candidates | 0.0625 ms | 0.1145 ms |
| Candidate exchange / merge / certificate | 0.3110 ms | 0.4025 ms |
| Selected-owner localization / compaction | 0.0669 ms | 0.0694 ms |
| Sparse MLA | 0.2545 ms | 0.2617 ms |
| Local MLA split combine | 0.1297 ms | 0.1338 ms |
| Projected DCP push combine | 0.2746 ms | 0.2808 ms |

The observed context growth is particularly visible in local top-k and
candidate merge, not simply the indexer matrix multiplication. The merge
symbol covers conditional fallback and cannot reveal its frequency or number
of failed owners. Similarly, a full kernel symbol does not establish selected
ownership counts. Those distributions remain unmeasured.

The model now includes source-declared localize, scheduler and split-combine
launches, FP32 partial-buffer traffic and mandatory candidate merge work.
The generic launch/transport floors still underestimate the observed sequence;
score-dependent scan work, polling and synchronization are not calibrated. The table is calibration evidence and a
limit on timing accuracy, not permission to fold overlapping kernel sums into
an end-to-end estimate. Q/V projections are not assigned arbitrary portions of
shared GEMM symbols without reliable scope/shape evidence.

The first staged profiler attempt saved prefill traces, then failed at the
second Kineto stop. Its truncated decode traces are excluded. One continuous
capture with Python stacks disabled succeeded; this is a workaround, not a
proven root cause. All artifacts and failure records are retained. Afterward,
all eight GPUs and benchmark processes were released.

## Reproduction and evidence

### Independent kernel calibration

The [B300 calibration evidence](evidence/kimi-k31-b300-calibration/calibration_rows.json)
contains nine validated physical-key rows: six paged indexer shapes, one
full-head Q projection, and the absorbed Q/V BMMs with SGLang's actual operand
strides. Separate runs with independent input seeds give mean absolute timing
differences of 8.52% for the indexer and 0.44% for the projections. These are
same-shape repeatability checks, not unseen-shape or end-to-end accuracy.
See the [indexer report](evidence/kimi-k31-b300-calibration/indexer/REPORT.md)
and [projection report](evidence/kimi-k31-b300-calibration/gemm/REPORT.md).

The estimate script accepts `--calibration-rows` pointing to that JSON file,
using the explicit `envelope` selector rather than claiming complete declared
framework-dispatch coverage for these providers.
Only matching physical keys receive measured timings; unmatched operations
retain theory. `t_sol_s` remains the analytical result, while the existing
partial-measurement fields report the mixed result and coverage. At the 16K
mean-context point, the public API gives 7.701 ms for this partial measured
envelope versus 7.096 ms for theory and 10.081 ms HTTP TPOT. The full measured
field remains null because coverage is incomplete; the reported measured time
share is 11.06%. [The public result](evidence/kimi-k31-dcp8-calibrated.json)
preserves the selector and coverage fields. The standard
L2-drain CUDA-event protocol differs from the warm-cache distributed serving
trace; neither result is substituted for the other.

The measurements do not calibrate FlashInfer MXFP8×MXFP4 MoE or the
cross-component Kimi fused-front GEMM. A numerically similar BF16 MoE provider
does not execute the same implementation. Fused-front combines projections
across components, emits FP32, and changes its sharding by token count; its
timing cannot be assigned to an individual unfused gate projection.

All raw requests, server logs, runtime configuration, profile attempts and
scripts remain under `/tmp/scratch-space/k3-performance/benchmark/dcp-validation`.
Full analytical rows remain under `/tmp/scratch-space/k3-performance/logs`.
The repository keeps compact summaries rather than the large raw trace files.
The [evidence index](evidence/README.md) records the persistent raw archive and
checksums, including detailed continuous-trace spans and per-job calibration
files. The [calibration reproduction guide](evidence/kimi-k31-b300-calibration/README.md)
retains the scripts and generates new raw outputs outside the checkout.

```bash
.venv313/bin/python scripts/compare_kimi_k31_dcp.py \
  --measurements /tmp/scratch-space/k3-performance/benchmark/dcp-validation/results.json \
  --theory-dir /tmp/scratch-space/k3-performance/logs \
  --output /tmp/scratch-space/k3-performance/HuggingArch/docs/evidence/kimi-k31-dcp-comparison.json
```

Memory accounting includes known runtime weight replicas and aliases. It is not
an exact SGLang allocator capacity forecast: graph pools, process-wide peer
workspaces, cache checkpoint multiplicity and HiCache are separate. The live
server's resolved cache pool and request cap are retained in the runtime audit.

Compact evidence: [comparison](evidence/kimi-k31-dcp-comparison.json),
[profile summary](evidence/kimi-k31-dcp-validation-summary.json),
[audit](evidence/kimi-k31-dcp-audit.json), and
[GPU release](evidence/kimi-k31-dcp-final-gpu-release.json).
