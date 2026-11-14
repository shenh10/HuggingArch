# B300 exact BF16 projection calibration

Three ordinary L2-drain CUDA-event measurements on exclusive GPU7 produce consumable `../calibration_rows.json`; `summary.json` verifies lookup of each physical key. Train/heldout p50us:

| Actual SGLang operation | Train | Independent input/run holdout |
|---|---:|---:|
| Full-Q BF16 M1 K1536 N18432 |16.000|16.160|
| Absorb-Q BMM H96 M1 K128 N512 |9.760|9.760|
| Absorb-V BMM H96 M1 K512 N128 |9.728|9.760|

Mean absolute holdout error0.44%. All six numerical checks and CSV rows pass. The holdout uses canonical seed XOR2026111303, is separately generated and measured, and never enters training calibration. This is same-shape repeatability, not a heldout full-model prediction.

Full Q follows actual `deepseek_v2.q_b_proj_replicated_forward` BF16 `F.linear`, measured by the existing torch dense matmul provider. Absorbed Q/V follow actual `forward_mla.py` torch.bmm with contiguous [heads,reduce,out] weights, and activations transposed from token-major heads. Q is sliced from a192-wide full head: [96,1,128] strides(192,18432,1). V is [96,1,512] strides(512,49152,1). New `sglang_mla.py` providers preserve these exact strides. Four CPU tests cover numerical equivalence and strides across batch/query/head configurations.

Standard timing includes its prescribed output L2 drain, excludes load-time packing and input-layout setup. It cannot be equated with warm-cache serving CUPTI duration, distributed contention, or communication. No stream duration sums or global multipliers are used. The same host was running CPU/GPU0 tests. The DCP8 production planner supplies the keys; the archived `jobs.json` preserves the measured plan.

MoE MXFP8 x MXFP4 is not calibrated by these rows: existing FlashInfer fused-MoE provider only serves BF16/FP16, while DeepGEMM grouped providers dispatch a different implementation. Exact FlashInfer loader layout, routing semantics and a matching kernel boundary are required before measurements can be imported.

Reproduce using the [calibration README](../README.md), including the required
`plan.py` step before `run.py`. Source/runtime provenance is retained in the
[indexer runtime record](../indexer/runtime.json). Raw jobs, per-run environments,
complete training/holdout CSVs and test logs are archived outside Git at
`/tmp/scratch-space/k3-performance/benchmark/pr-review-raw-evidence/kimi-k31-b300-calibration/gemm/`.
