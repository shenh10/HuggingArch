# Kimi-K3.1 evidence

This directory keeps reviewable summaries, validated calibration rows, audits,
and calibration reproduction scripts. The original request logs, CUDA traces,
per-job inputs, environment dumps and raw calibration CSVs stay outside Git.

The files removed from this PR are preserved under the persistent local path
`/tmp/scratch-space/k3-performance/benchmark/pr-review-raw-evidence/`, using
paths relative to this directory. Its `manifest.json` records each original
repository path, archive path, byte count and SHA-256. All 16 archived files
were also verified byte-for-byte against their existing benchmark originals.

| Evidence | Checked-in entry | Persistent raw location |
|---|---|---|
| Four serving configurations | [runs](kimi-k31-b300-runs.json), [audit](kimi-k31-b300-audit.json) | `/tmp/scratch-space/k3-performance/benchmark/` |
| Target-only DCP profile | [summary](kimi-k31-dcp-validation-summary.json), [comparison](kimi-k31-dcp-comparison.json) | `/tmp/scratch-space/k3-performance/benchmark/dcp-validation/` |
| Detailed continuous-trace spans | Included in the profile summary above | `pr-review-raw-evidence/kimi-k31-dcp-continuous-trace-spans.json` beneath the benchmark directory |
| Independent kernel calibration | [nine training rows](kimi-k31-b300-calibration/calibration_rows.json), [reproduction](kimi-k31-b300-calibration/README.md) | `/tmp/scratch-space/k3-performance/benchmark/{indexer,gemm}-calibration/` |

The external archive includes calibration `jobs.json`, training/holdout CSVs,
per-run environment dumps, test logs and incidental GPU-process/release files.
These are preserved as evidence, not loaded by the estimate. The estimate uses
only the checked-in `calibration_rows.json`; independent-input holdout results
remain in the compact summaries and must not be treated as unseen-shape or
whole-model validation.
