# Kimi-K3.1 colocated B300 measurements

These measurements exercise the [colocated recipe](recipes/kimi-k31-colocated.md)
on eight NVIDIA B300 SXM6 AC GPUs. They are serving measurements, separate from
HuggingArch's analytical target-step estimates. A separate
[target-only DCP8 validation](kimi-k31-dcp-validation.md) provides matched
measurements for the new sparse DCP model. All raw artifacts and
reproduction scripts are retained at
`/tmp/scratch-space/k3-performance/benchmark`.

## Configuration and method

SGLang main is pinned to `1b2870d33de7b5f31844fd74e97eea6d7896c53a`, rather
than the recipe's older `5792cf6e…` revision. The target is the local 96-shard
Kimi-K3.1-Pro checkpoint; DFLASH draft revision is
`ea257b38a8ee755b65287bb8ccd0938b519ce26b`.
The runtime uses Torch 2.14.1+cu130 and FlashInfer 0.7.4. Package manifests,
source revisions, launch commands, and startup logs are in the artifact directory.

The four configurations vary only EP1/EP2 and prefill graph `breakable`/`disabled`.
Attention remains TP8/DCP8; expert tensor parallelism is 8 for EP1 and 4 for EP2.
All use `fi_a2a`, FP8 target/draft KV, DFLASH five-token block verification,
linear ReplaySSM, static memory fraction 0.85, Mamba memory ratio 1.24,
16K prefill chunks, and HiCache size 160. Decode/verify graphs remain enabled.
The server listens only on localhost for this experiment.

The official SGLang serving benchmark sends fixed-length token IDs derived
from ShareGPT, with seed 20261113, greedy sampling, ignored EOS, and streaming.
These modified prompts test performance, not conversational quality or general
DFLASH acceptance on natural workloads.
The fixed seed belongs to the benchmark client. The recipe leaves the server
seed unset, so SGLang generates a different server seed per launch. Temperature
is zero, but numerical execution paths and selected kernels can still change
generated tokens and speculative acceptance; the runs do not isolate identical
target/draft token trajectories.

| Input/output tokens | Concurrency | Measured requests per repeat |
|---|---:|---:|
| 128/256 | 1 | 12 |
| 2048/256 | 1 | 12 |
| 16384/256 | 1 | 8 |
| 2048/512 | 8 | 32 |
| 16384/256 | 8 | 16 |

Each configuration runs two repeats per case, after warmup at twice the
concurrency and a cache flush. Validation checks all request counts, exact
input/output lengths, errors, and cache statistics. Startup, compilation,
autotuning, and cache allocation are excluded from measured throughput.

TPOT is `(request latency - TTFT)/(output tokens - 1)`. This SGLang client
normalizes each SSE gap by the number of newly emitted tokens, so its ITL
statistics are not raw DFLASH burst intervals. Server-reported acceptance
length uses completed logging intervals and can omit the final partial interval.
Only two repeats and a fixed configuration order are used; small differences
are not evidence of a repeatable speedup.

## Execution details and audit notes

- Breakable prefill captures 58 shapes up to 8192 tokens. Full 16K chunks run
  eager in both graph configurations, confirmed in per-batch logs.
- EP2 allocates 668,928 physical KV rows/rank and 204 Mamba slots. EP1 allocates
  664,192 rows and 202 slots. Both resolve to an actual scheduler cap of 40
  requests. `server_info` retains an initial cap of 48; scheduler logs are the
  source of the effective cap.
- Cargo is unavailable in this environment, so SGLang uses Python TreeCore.
  A misleading unconditional NPU fallback message does not describe the actual
  CUDA cache transfers: the resolved HiCache I/O backend remains `kernel`.
- Initial cold compilation was unusually expensive. An SGLang JIT cache path
  symlink caused generated staging-source dependencies to invalidate compiled
  entries. The persistent real-path cache fix, manifest repair audit, and
  unchanged kernel binaries are documented in `benchmark/RUNTIME.md`.
- All ten EP2 graph-on runs and five metrics captures completed. Editing the
  executing Bash script in place caused a later bookkeeping EOF; final
  `server_info` is missing for that configuration. Its ready snapshot and
  measured resource samples remain. Subsequent edits use atomic replacement.
- The first EP2 graph-off 128-token repeat was slower than the second without
  a measured-interval compiler event. Two extra confirmation repeats agree
  with the second. The original result is retained, and confirmations are
  reported separately instead of silently replacing it.
- An independent single-GPU MoE microbenchmark ran between the first two
  configurations while the serving model was stopped. SGLang autotune cache
  hashes were identical before and after. Its synthetic max-group kernel
  latencies exclude communication, routing, quantization and full-model
  effects, so they are not end-to-end speedups.

GPU capture logs label GiB as GB; HiCache host allocation logs use decimal GB.
The final resource comparison must use normalized units and total resident
memory, because allocator reuse makes individual capture increments non-additive.

## Results

All **40 canonical runs / 640 measured requests** passed exact count, length
and error checks; all reported zero cached tokens. Two additional 12-request
confirmation runs are separate. The [full results](evidence/kimi-k31-b300-results.md)
include repeat ranges and all relative comparisons. Machine-readable
[runs](evidence/kimi-k31-b300-runs.json), [audit](evidence/kimi-k31-b300-audit.json),
and [runtime evidence](evidence/kimi-k31-b300-runtime.json) are checked in.

These are mean output throughputs in tokens/second:

| Input/output | Concurrency | EP2 graph on | EP2 graph off | EP1 graph on | EP1 graph off |
|---|---:|---:|---:|---:|---:|
| 2048/256 | 1 | 296.86 | 241.65 | 292.10 | 230.81 |
| 2048/512 | 8 | 1128.93 | 1008.36 | 1062.72 | 977.35 |
| 16384/256 | 1 | 169.58 | 169.69 | 166.85 | 164.41 |
| 16384/256 | 8 | 270.17 | 269.99 | 261.15 | 260.51 |

**Prefill graphs help short prompts.** At 2K input / concurrency 1, turning
prefill graphs off raises TTFT from 158.26 to 371.10 ms for EP2 and from
173.00 to 365.42 ms for EP1. At 128 tokens, graph-on TTFT is about 69–71 ms,
versus 399–431 ms in the stable graph-off repeats and confirmations. The
original EP2-off first repeat was 645.65 ms and remains in the raw table.
Full 16K chunks have no consistent graph benefit, as neither configuration
captures that full prefill shape.

**EP2 gives a modest serving advantage in this matrix.** With graphs enabled,
changing EP2 to EP1 lowers throughput by 1.60–5.86% across the five cases.
With graphs disabled, the four 2K/16K cases fall by 3.07–4.48%. For 128 tokens,
EP1-off is only 0.93% below the additional EP2-off confirmation mean, with
nearby repeat ranges; the original slow EP2-off run must not be interpreted
as evidence of an EP1 advantage. These results include changes in speculative
acceptance and scheduling, not just expert-kernel execution.

Long uncached prompts can dominate colocated latency: EP2 graph-on 16K/C8
has mean TTFT 3.621 s and TPOT 15.04 ms, compared with 0.789 s and 2.81 ms at
C1. This workload does not test prefix-hit benefits of HiCache.

| Variant | Ready GPU0 GiB | Prefill graph retained increment GiB | Effective request cap |
|---|---:|---:|---:|
| EP2 graph on | 248.40 | 10.75 | 40 |
| EP2 graph off | 239.26 | 0 | 40 |
| EP1 graph on | 247.05 | 9.38 | 40 |
| EP1 graph off | 239.30 | 0 | 40 |

Ready memory is a device measurement, not a maximum serving footprint.
Allocator and workload behavior increase memory later; runtime artifacts
retain sampled device usage. GPU release is verified separately in
`benchmark/gpu-handoff.json` after stopping the serving processes.
