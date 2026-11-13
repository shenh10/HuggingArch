# Kimi-K3.1 Pro on SGLang

`moonshotai/Kimi-K3.1-Pro` resolves to the `kimi_k31` family. This implementation
is audited against SGLang main commit
`1b2870d33de7b5f31844fd74e97eea6d7896c53a`. It does not assert vLLM support.
The checkpoint snapshot comes from local safetensors headers, including all 96
shards, and freezes the SGLang sources used to define the graph.

## What is represented

- The multimodal checkpoint, including MoonViT-3D and its projector, AttnRes,
  SiTU, the dense first FFN, and 92 latent-MoE FFNs (896 experts, top 16).
- 69 KDA layers and 24 independent DSA layers. DSA occurs at one-based layers
  4, 8, …, 92, 93; its top-k is 2048. The analyzed profile uses absorbed MLA in both phases,
  with a sigmoid output gate and no rotary embedding or Hadamard transform.
- Each DSA layer has its own BF16 indexer projections and LayerNorm key.
  The indexer scans the entire history; only the subsequent MLA read is
  bounded by top-k. Indexer heads are 32 × 128, with one shared cached key.
- Only routed experts are checkpoint MXFP4. On the audited SGLang Blackwell
  `flashinfer_mxfp4` default, their activations are dynamically MXFP8 with
  group-32 E8M0 scales. The framework binding owns this choice; the checkpoint's
  null activation quantization field does not imply a W4A16 launch.

The [runtime precision audit](kimi-k31-runtime-precision.md) describes the
framework dispatch and explicit FP8 query conversion. This is a cost model,
not a bit-exact numerical reimplementation of each fused kernel.

## Cache geometry

The default CUDA `flashmla_kv` path uses `--kv-cache-dtype fp8_e4m3`.
These numbers were checked with the actual SGLang GPU writers on B300;
[the evidence](evidence/kimi-k31-b300-cache.json) includes the software revision.

| State | Bytes | Interpretation |
|---|---:|---|
| MLA latent | 528 | 512 FP8 values + four FP32 power-of-two scales |
| Unrotated key tail | 128 | 64 BF16 values; NoPE does not remove this tail |
| Index key | 132 | 128 FP8 values + one ordinary FP32 scale |
| All 24 DSA layers | 18,912 / token | 24 × (656 + 132), before DCP sharding |
| KDA state | 6,512,640 / layer / request | FP32 [96,128,128] recurrent state + three BF16 convolution histories of [3,96,128] |

The CUDA fused index-key writer uses an ordinary `amax/448` FP32 scale even
though the query quantization path requests UE8M0. The main MLA latent's
power-of-two scales are physically FP32, not one-byte E8M0 storage.
At TP8, a KDA layer's convolution state is `[3,4608]` and recurrent state is
`[12,128,128]`. All 69 KDA layers hold 56,171,520 bytes per request per TP8
rank, before runtime pool padding, replay slots, or speculative copies.
With DCP8, DSA cache growth is 2,364 bytes/token/rank, before page rounding.
The audited allocator uses 64 physical rows per rank and a 512-token virtual
page at DCP8 (`allocation_sizing.py:get_alloc_page_size`, `kv/alloc/band.py`).
For a 17-token request the raw tensor law counts three rows per rank, whereas
page allocation reserves 64. This estimator reports raw tensor capacity;
its maximum batch is an upper bound, not the server's resolved request limit.

The `trtllm`/`prims_ts` raw-FP8 layout and experimental FP4 index cache are
not represented by this default-layout spec. Selecting a different attention
backend requires a corresponding layout declaration.

## Analytical estimates and measured serving

Run a supported target-only TP8/DCP1 estimate from the repository root:

```bash
uv run --python 3.13 python scripts/estimate_kimi_k31_sglang.py \
  --ep 2 --input-tokens 16384 --output-tokens 512 --batch 1 \
  --output /tmp/scratch-space/k31-estimate.json
```

This is a theoretical estimate. It does not measure CUDA graph launch savings,
DFLASH acceptance, HiCache hit rate, or scheduler interference. The
[user-supplied colocated recipe](recipes/kimi-k31-colocated.md) uses DCP8 and
DFLASH and is a separate serving configuration; the recipe's pinned commit
`5792cf6e…` is also distinct from the main revision audited here.

At batch 1, 16,384 input tokens and 512 output tokens, the current B300 catalog
and SGLang binding give these **roofline lower bounds**, with balanced expert
routing and no measured-kernel calibration:

| TP8 / DCP1 | Prefill | One target decode step |
|---|---:|---:|
| EP1 | 509.868 ms | 6.753 ms |
| EP2 | 481.349 ms | 6.752 ms |

The text-step weight placement is 205,489,163,264 bytes per rank in both cases.
HuggingArch budgets the vision encoder in its separate encode phase; this
text-only estimate excludes its 894,717,952 resident bytes per server rank.
These are not a prediction of the DCP8/DFLASH benchmark's wall times. The
complete public output is summarized in [EP1](evidence/kimi-k31-tp8-dcp1-ep1.json)
and [EP2](evidence/kimi-k31-tp8-dcp1-ep2.json), including assumptions and cache law.
The catalog uses 269 GiB per B300; the tested card reports 275,040 MiB.

For DCP1 eager prefill, SGLang can choose dense `MHA_ONE_SHOT` when the maximum
KV length is at most 2048 and other batch conditions hold. That path is outside
this absorbed profile. Set `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD=0`
on a server to force the modeled path for short prompts; the default 16K example
above is already beyond that threshold. Within breakable/piecewise prefill
graphs and with DCP8, the audited backend does not select this dense shortcut.

**TP8/DCP8 target decode is modeled for the SGLang `fi_a2a` reference profile.**
Use `--dcp 8` with the estimate script. The model follows local index scans,
explicit selected-position imbalance, candidate exchange, full-head Q/absorbed
projections and 128-wide projected output combine. Its default balanced,
no-certificate-fallback assumptions are scenarios, not measured distributions.
See the [DCP execution model](kimi-k31-dcp.md) for controls, source evidence,
profile boundaries and runtime storage accounting. The
[target-only B300 validation](kimi-k31-dcp-validation.md) reports the remaining
gap between the analytical lower bound and measured serving time. DFLASH acceptance and
full-server scheduling remain outside this target-step analytical estimate.

Likewise, the fixed KDA state above is the state of one request, not SGLang's
complete allocator budget. `mamba_full_memory_ratio`, DFLASH replay state,
CUDA graph buffers, and HiCache host pools must be read from a live server
when sizing the full recipe.
The [B300 serving benchmark](kimi-k31-b300-benchmark.md) records the separate
EP1/EP2 and prefill graph comparisons, including the exact runtime and workload.
Runtime resident weights can also exceed checkpoint placement: SGLang makes
contiguous absorbed K/V copies, and casts indexer LayerNorm weight/bias to FP32
(12,288 extra bytes across 24 layers per rank). The estimator's weight figure
is not a measurement of allocator residency.

## Reproduction and validation

Freeze the snapshot from local files without loading tensors:

```bash
uv run --python 3.13 python scripts/snapshot_kimi_k31.py \
  --checkpoint /tmp/scratch-space/Kimi-K3.1-pro \
  --sglang-source /tmp/scratch-space/k3-performance/sglang
```

Run the cache oracle in an environment containing the audited SGLang and CUDA:

```bash
python scripts/probe_kimi_k31_sglang.py \
  --sglang-source /tmp/scratch-space/k3-performance/sglang \
  --output /tmp/scratch-space/k31-cache-probe.json
```

The checkpoint reconciliation compares tensor names, shapes, dtypes and bytes;
all 1,561,217,377,280 stored bytes are accounted for. Compared with K3, K3.1 adds
120 indexer tensors, 178,526,208 logical parameters and 357,052,416 stored bytes.
The old K3 spec retains its four-position reference convolution history; the
new SGLang spec uses three historical positions for a four-tap kernel.

The source AST validator cannot follow SGLang's `make_pp_layers` helper to locate
the decoder ModuleList. This is a documented topology-check limitation, not a
claim that checkpoint reconciliation independently proves forward execution.
It also reports the first layer's two intentionally unread AttnRes parameters,
which the spec retains for checkpoint coverage, and a 256-FLOP constant term
from KDA's parameter-only decay exponential in its batch-linearity check.

The [validation record](evidence/kimi-k31-validation.json) preserves the exact
byte/parameter reconciliation and test scope. K3.1's corpus key and value checks
pass. The older K3 SOL golden has pre-existing drift on the base commit: the
current H100 admission refuses its memory footprint and returns empty timing
rows, whereas its golden retains timing rows. A separate base checkout produced
the identical 4,605 field differences, so this change does not rewrite that
unrelated golden.
The all-model fingerprint test also finds an existing DeepSeek-V4.1-Flash
fingerprint mismatch; the unchanged base produces the same hash. Its golden
is preserved as well. All other model fingerprints, including K3 and K3.1,
match their registered rows.
