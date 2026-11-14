# B300 paged indexer calibration

Six training rows are directly consumable through HuggingArch's existing calibration overlay: `../calibration_rows.json`. `summary.json` verifies all six keys via `merge_validated_rows` and `physical_time_for`. No global multiplier was fitted.

The provider calls the same DeepGEMM FP8 paged MQA logits entry as SGLang commit 1b2870d, with FP8 Q, float32 scaled head weights, and the byte-addressed per-page FP8 key payload followed by float32 key scales. Page size64, B1, query1, heads32, head width128. Independent GPU7 measurements cover local contexts256,257,288,2048,2049,2080, including both profile contexts and the mean served decode contexts. Quantization, cache packing, schedule creation, top-k, and distributed communication are excluded.

All12 train/holdout rows passed the standard numerical check and CSV validator. Train p50s:10.016,7.904,8.992,9.856,9.984,9.792us. Separate-process holdout input seeds use canonical seed XOR2026111302; CSV input_seed is corrected to the actual generator seed. Mean absolute heldout timing error8.52%, worst13.87%. This validates repeatability on independently generated inputs at measured shapes, not unseen-shape prediction or the whole model.

Timing is ordinary kernel_bench CUDA-event L2-drain (60samples), not CUPTI. The serving trace's roughly4.9us per logits call has a different cache/concurrency/timing regime. Importing these rows must not be represented as a prediction of identical trace latency. The GPU0/CPU parent test suite shared the host; GPU7 was exclusive.

Implementation adds an explicit `paged_byte_cache_inputs` operand declaration. The launch gate derives exact storage shape using the existing CacheFormat byte widths/scales and page size; only matching uint8 operands are accepted. CPU tests also decode the SoA page and verify values, partial pages, distinct requests and scale encodings. The gate still rejects undeclared or malformed byte storage. Normal FP8 numerical checks remain enabled.

An initial production DCP8 planning dependency error was repaired before timing. The initial valid numerical kernel was then refused by the old provenance gate solely because uint8 packed storage was not expressible; that rejected run remains under `rejected-byte-gate`. No refused timing enters calibration. `PLAN.md` records the predeclared split and intermediate planning issue; final jobs use the repaired DCP8 planner. The planner's display string omits dcp, so `summary.json` records the explicit deployment.

Reproduction commands and the required output-directory setup are in the
[calibration README](../README.md). Raw `jobs.json`, training/holdout CSVs,
per-run environment dumps and test logs are archived outside Git at
`/tmp/scratch-space/k3-performance/benchmark/pr-review-raw-evidence/kimi-k31-b300-calibration/indexer/`.
The original `rejected-byte-gate` attempt remains in
`/tmp/scratch-space/k3-performance/benchmark/indexer-calibration/rejected-byte-gate/`.

The retained [runtime record](runtime.json) and archived per-run environment files record packages, source hashes, driver and SM. The wheel reports DeepGEMM0.2.8 without an upstream commit; do not substitute a guessed commit. FlashInfer optional wheel mismatch warnings are retained but this provider calls DeepGEMM and does not compile FlashInfer kernels.
