# TP-interleaved FP8 checkpoint grids

A fused projection's logical Q/K/V segments do not necessarily own separate
FP8 scale grids. MiMo-V2.5 stores four interleaved `[Q|K|V]` chunks; V2.5-Pro
stores eight. Each complete chunk owns its own ceil-padded grid. For V2.5 full
attention, the weight is `[13568, 4096]` and its scales are
`[4 * ceil((13568 / 4) / 128), 32] = [108, 32]`. The K segment's 192 local
rows sit inside the fused chunk's grid.

The quant entry declares only checkpoint facts: `block_scale_layout.kind`,
`checkpoint_tp` and the parallel `world`. Its scale formula uses
`chunks_per_rank`, which is the checkpoint degree in the unsharded view and
the number of complete chunks a rank owns after projection. Framework behavior
lives in `framework_bindings/<framework>.yaml` under `weight_loaders`, selected
by the model architecture and backed by an audited `source_anchor`.

The resolved loader travels with the resident `TensorFormat`, then with the
walk's `WeightShard` and published `MatrixShard`. It does not change logical
head placement. The planner reads that resolved fact and retains ordinary
block-alignment checks for weights without a loader and for the reduction axis.

| Loader | Matching checkpoint TP | Other supported TP |
|---|---|---|
| SGLang v0.5.22, `whole_chunks` | Load one complete chunk and its scales | TP must divide checkpoint TP. Merge complete chunks, dequantize, deinterleave Q/K/V, and requantize one rank-local fused matrix. |
| vLLM v0.32.0, `head_gather` | Load one complete chunk and its scales | Gather each rank's Q heads and K/V heads from the checkpoint chunks, replicate K/V where necessary, then requantize one rank-local fused matrix. |

For V2.5 full attention, SGLang TP1/2/4 reads resident grids with 106/53/27
rows per rank; TP8 is refused. vLLM TP8 has local fused N=1856, including
replicated K/V, and a 15-row scale grid. V2.5-Pro's matching degree is TP8;
SGLang also supports TP1/2/4, while vLLM can gather heads at TP16.

Serving and capacity materialize the same resident format on the existing
analysis when a loader requantizes. Checkpoint formats retain the original
grids; `iter_resident`, whole-model residency, rank projection and the weight
breakdown share the resident format. An unsupported `whole_chunks` degree
publishes an explicit load refusal: ordinary analysis and capacity reject it,
while the kernel planner reports the affected cells as unplaced. At a
`head_gather` degree that cannot own whole checkpoint chunks, the physical
ledger omits the source grid's rank surface because gathering does not define
a contiguous shard of that grid. Global checkpoint tensors remain available
through `iter_checkpoint`. Load-time CPU buffers and traffic are not modelled.

`tests/kernel_bench/test_scale_block_refusal.py` compares the walk against an
independent oracle: it executes the audited vLLM loader function from
`tests/kernel_bench/fixtures/vllm_mimo_v2_0_32_0.py` with a shape-spying
requantizer, and checks both Q/K/V row identity and scale shapes.

Sources:

- [SGLang fused loader and deferred scale resolution](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/models/mimo_v2.py#L101-L289).
- [SGLang checkpoint TP selection](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/configs/model_config.py#L117-L145).
- [vLLM fused weight/scale sharding](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/mimo_v2.py#L495-L632).
