# Kimi-K3.1 runtime precision on SGLang

Audited against SGLang main `1b2870d33de7b5f31844fd74e97eea6d7896c53a`.
This is a runtime cost-model check, not a full-model numerical accuracy test.

## Routed experts

The checkpoint declares MXFP4 packed expert weights with no quantized input
activation. On SM100/103/107, SGLang's Kimi auto-configuration selects
`moe_runner_backend=flashinfer_mxfp4`. With the default
`flashinfer_mxfp4_moe_precision=default`, its activation preparation uses MXFP8:
E4M3 elements and one UE8M0 scale per 32 channels. HuggingArch now selects this
launch through its SGLang framework binding, without changing the checkpoint's
weight-only declaration. The existing cast materialization and cost path consume
that declaration. `flashinfer_mxfp4_moe_precision=bf16` selects the separate BF16
activation path. Other architectures, unsupported GPU families and other runner
backends do not inherit Kimi's default.

Source:

- [Kimi auto-configuration](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/arg_groups/model_overrides/kimi_k3.py#L244-L262)
- [Runtime default/BF16 branch](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py#L1554-L1595)
- [MXFP8 activation preparation](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/srt/layers/quantization/mxfp4.py#L79-L119)

## DSA indexer query

The indexer projections remain BF16. The existing `index_key.paired_operand`
contract automatically inserts an FP8 conversion on the query before its product
with cached FP8 keys. A CPU execution of the complete K3.1 serving graph rewrite
verifies that `indexer_logits` reads an `F8_E4M3` operand after cast materialization;
its pre-cast BF16 projection does not force BF16 scoring.

SGLang calls `act_quant(query, 128, "ue8m0")`. Despite that scale-format argument,
this function returns **FP32 scales**, with values rounded to powers of two.
K3.1's 128-channel query head therefore occupies 128 FP8 bytes plus one FP32
scale. HuggingArch's existing `fp8_per_token` query format has that byte layout
(one row per head), but does not encode power-of-two rounding. Its memory and
compute precision are modeled; bit-exact quantization semantics are not claimed.
The cached index key uses an ordinary FP32 scale, independently of this query
rounding rule.

Source: [TileLang activation quantization](https://github.com/sgl-project/sglang/blob/1b2870d33de7b5f31844fd74e97eea6d7896c53a/python/sglang/kernels/ops/quantization/act_quant_tilelang.py#L95-L125).

## Validation

`tests/backend/inference/test_sglang_precision_launches.py` covers auto and
explicit backends on B200/B300, BF16 override, non-Blackwell and unrelated-model
exclusion, query cast materialization, multi-flag dispatch and pinned source
citations. All 14 cases pass with `pytest -m ''`.

The pre-existing SGLang binding remains audited at `v0.5.22`; only the new source
citations pin the main commit above. The local shallow SGLang checkout lacks the
old `baa9befa381511cd4d56f83980951e5a45fffc6d` object. Source-anchor tests validate
available commits and therefore verify the new citations locally; they do not
revalidate the unavailable old revision. No older repository history was fetched.

## Header parameter counts

Packed-weight expansion now requires both the variant's tensor suffix and its
storage dtype to match the safetensors header. This matters when packed U8
`.weight` tensors and ordinary BF16 `.weight` tensors coexist. Real packed U8
plain-name weights still expand by two; BF16 weights count once. The generic
GPTQ/AWQ and other registered packed variants retain their declared expansion.

The K3.1 metadata count is **2,780,110,363,392** logical parameters. The previous
suffix-only summary overcounted BF16 weights by 57,368,590,976. Kimi-K3's older
snapshot also recorded the wrong scheme (`compressed-tensors`, I32 packing);
only its scheme field was corrected to `mxfp4-pack-quantized`, leaving its
four-bit width, header, configuration and source untouched. Its corrected count
is **2,779,931,837,184**.

A scan of all 64 current snapshots found no parameter-count change in the other
62 snapshots and no change in checkpoint storage bytes. Mixed-header integration
and all registered packing variants are covered by 33 passing tests with 63
subtests in `test_header_only_metadata.py` and `test_quant_schemes.py`.

Runtime memory and cache ownership
----------------------------------

The absorbed K/V matrices are separate resident buffers even at DCP1. At TP8
this adds 75,497,472 bytes per rank across the 24 DSA layers. DCP8 additionally
holds the gathered Q/K/V operands and the separate CUDA preproject-V buffer;
Q's original local weight aliases its gathered allocation and is counted once.
The runtime allocation ledger preserves the checkpoint's parameter count.

For striped DCP storage, the busiest owner reserves `ceil(context / 8)` rows
per request. This physical storage law is shared by the capacity total,
breakdown and formula; the asymptotic per-token slope remains divided by eight.
It does not reduce persistent capacity to the selected top-k. KDA state bytes
are unchanged. Prefill gathers complete index and MLA cache buffers, padding
rank slices for the collective and trimming back to the logical context.
Its attention reads therefore use the full gathered context; padding does not
increase the mathematical attention FLOPs.

Calibration entry points (SGLang `1b2870d33de7b5f31844fd74e97eea6d7896c53a`)
--------------------------------------------------------------------------

Ordinary unquantized dense layers call `quantization.unquant.bf16_gemm_dispatch`.
On SM103, auto selects CuTe DSL unless deterministic inference is enabled.
Graph execution may first take FlashInfer's low-M autotuner; eager execution
can select a different kernel. An exact calibration must preserve this dispatch
state, the model's registered TGV ranges, output dtype and operand strides.

Kimi's merged MoE front instead calls `models.kimi_k3._k3_bf16_gemm`, optionally
with an FP32 output. Its dense inputs are BF16, but output traffic is four bytes
per element when `_front_fp32` is active. Model construction registers
`_K3_TGV_M_RANGES` and, on SM103, `_K3_GB300_TGV_M_RANGES`.

The BF16 DCP replicated Q projection is a distinct path:
`DeepseekV2AttentionMLA.q_b_proj_replicated_forward` calls `F.linear` directly
with `[M,1536]` and `[18432,1536]`. The absorbed Q projection calls `torch.bmm`
with input `[96,M,128]`, obtained by slicing `[M,96,192]` then transposing;
its strides are `(192,18432,1)`. The preproject-V BMM input `[96,M,512]` is
transposed from `[M,96,512]`, with strides `(512,49152,1)`. Their contiguous
weights are `[96,128,512]` and `[96,512,128]`. A contiguous random batched input
is not the same layout. These torch calls are the actual audited execution
paths; they must not be labelled FlashInfer dense dispatch.

Timing fidelity: merged MoE front
--------------------------------

The current timing graph still exposes separate router, latent-down and shared
expert input projections. SGLang main merges those BF16 weights and runs one
GEMM whose output is FP32 on the audited CUDA path (`kimi_k3.py:795-924` and
`:1527-1547` at the commit above). At TP8, the full merged output width is 6144
for K3.1 (6016 for K3). A measurement of that GEMM must not calibrate the
standalone router projection: its matrix shape, output dtype and launch count
are different.

There is also a runtime dispatch boundary: with the default sharded-front
optimization eligible, 18 through 256 token rows use a TP-sharded router/latent
block and an FP32 gather. The full merged front is therefore not one fixed
kernel shape for all concurrency and verification widths. Existing fusion
conditions read a symbolic workload cell; a condition such as `B: [1]` cannot
safely specialize that cell to the caller's current batch. This launch and its
FP32 intermediate traffic remain an explicit timing-fidelity limitation,
including at C1, until workload-dependent fusion and output precision are
represented together. The current estimate is not a complete reconstruction of
SGLang's launch stream, and merged-front measurements are not consumed as
single-gate calibration.

Architecture-only analysis
--------------------------

The `theory` framework selects an architecture-only runtime profile. It does not
bind SGLang's absorbed-weight copies or sparse-DCP execution contracts. This
profile is part of the walk/cell cache identity, so an architectural publication
check cannot reuse SGLang allocations, or vice versa. Serving profiles retain
the declared runtime contracts and the SGLang-only admission check.

Page allocation is outside the raw tensor capacity bound
-------------------------------------------------------

For the recipe's page size 64 and DCP8, SGLang uses 64 physical cache rows per
rank and a 512-token virtual allocator page. The source distinguishes these
units in `mem_cache/allocation_sizing.py:get_alloc_page_size` and
`mem_cache/kv/alloc/band.py` (`pool_page_size` versus `page_size`). A request
with 17 logical tokens has `ceil(17/8) = 3` raw rows on its busiest rank, but
reserves a physical page of 64 rows per rank. More generally, this allocator
uses `64 * ceil(context/512)` local rows, before other reserves.

The estimator's `storage_law` rounds striped ownership only; `kv_block_size`
is a kernel measurement parameter and does not add page-pool rounding to
capacity. Its raw `max_batch` must therefore not be interpreted as SGLang's
runtime request limit. Page slack, recurrent-state slots, communication and
graph pools, and the KV/KDA memory split all require additional runtime sizing.
