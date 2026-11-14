# Kimi-K3.1 on SGLang

Model: `moonshotai/Kimi-K3.1-Pro`. Runtime profile: SGLang
`1b2870d33de7b5f31844fd74e97eea6d7896c53a`, absorbed MLA with FP8 cache.
The model has 69 KDA layers, 24 DSA layers and 896 routed experts (top 16).

## Run an estimate

From the repository root:

```bash
uv run python scripts/estimate_kimi_k31_sglang.py \
  --dcp 8 --ep 2 --input-tokens 16384 --output-tokens 512 --batch 1 \
  --output /tmp/scratch-space/k31-estimate.json
```

The script uses eight B300 GPUs with TP8. It accepts DCP1 or DCP8 and EP1 or
EP2. `--dcp-load-factor` sets selected-position max/mean ownership (1–8);
`--dcp-failed-owners` sets candidate-certificate fallback owners (0–8).
Use `--calibration-rows <file.json>` to apply validated Kernel Bench rows.

## Runtime details

- Routed experts use MXFP4 weights and MXFP8 activations.
- Each DSA cache row stores 656 bytes of MLA state and 132 bytes of indexer
  state. Across 24 layers this is 18,912 bytes/token; DCP8 stripes the rows.
- KDA uses FP32 recurrent state and three BF16 convolution-history positions:
  56,171,520 bytes/request/rank at TP8 across all 69 layers.
- DCP8 models the local index scan, candidate exchange and fallback, selected
  attention, split combine, and projected output exchange. Prefill gathers
  the packed caches. Runtime projection copies are included in resident weights.

Results describe target-model steps and raw tensor capacity. DFLASH acceptance,
server scheduling and allocator pools require separate serving measurements.
For short DCP1 prefills, set
`SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0` on the server to select the
absorbed path modeled here.
