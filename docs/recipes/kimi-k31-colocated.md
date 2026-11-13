# Kimi-K3.1 on SGLang (colocated, 8x B300)

Serve Kimi-K3.1 with DFLASH speculative decoding on 8-GPU B300 nodes, prefill and decode in the same server, one server per node behind `sgl-router`. Prefixes are cached on the GPU and in host memory (HiCache).

## Requirements

| Item | Requirement |
|---|---|
| GPUs | 8x NVIDIA B300 (SM103), one node per server |
| Host memory | `--hicache-size` x 8 GB for the HiCache host pool, plus about 300 GB for the server. With `--hicache-size 160`: about 1.6 TB |
| Disk | About 1.5 TB for the target checkpoint, 5 GB for the draft |
| Image | `lmsysorg/sglang:nightly-dev-cu13-20261113-7e31007a` with SGLang switched to main at `5792cf6e5e11aebc62d41cf19828f615a8fa7e0a` (see below) |

Checkpoints:

- Target: `moonshotai/Kimi-K3.1-Pro`
- DFLASH draft: `RadixArk/Kimi-K3.1-Dflash2-Preview`

## Image

The nightly image ships SGLang at `7e31007a`. Pin it to main at `5792cf6e5e`, which includes the chunked-prefill admission fix ([#44823](https://github.com/sgl-project/sglang/pull/44823)); without it, a colocated server can exit with `Prefill out of memory` when the KV cache fills up under long prompts.

```dockerfile
FROM lmsysorg/sglang:nightly-dev-cu13-20261113-7e31007a
RUN cd /sgl-workspace/sglang \
 && (git config --unset-all http.https://github.com/.extraheader || true) \
 && git fetch --depth 1 https://github.com/sgl-project/sglang.git 5792cf6e5e11aebc62d41cf19828f615a8fa7e0a \
 && git checkout 5792cf6e5e11aebc62d41cf19828f615a8fa7e0a \
 && pip install -e "python"
```

```bash
docker build -t sglang-kimi-k31:5792cf6e .
```

The image's SGLang checkout carries a stale GitHub auth header from its CI build; the `git config --unset-all` line removes it so the fetch works. `pip install` may warn about `cosmos-guardrail` and `moviepy` version conflicts; those packages belong to the image/video generation stack and do not affect this server.

## Launch

```bash
docker run --gpus all --ipc=host --network host \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  -e SGLANG_DCP_PUSH_COMBINE_MAX_ROWS=1536 \
  -e SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.1 \
  sglang-kimi-k31:5792cf6e \
  sglang serve \
    --model-path moonshotai/Kimi-K3.1-Pro \
    --trust-remote-code \
    --host 0.0.0.0 --port 30000 \
    --tp-size 8 --dcp-size 8 --ep-size 2 \
    --dcp-comm-backend fi_a2a \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --mamba-full-memory-ratio 1.24 \
    --chunked-prefill-size 16384 \
    --cuda-graph-backend-prefill breakable \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path RadixArk/Kimi-K3.1-Dflash2-Preview \
    --speculative-num-draft-tokens 5 \
    --speculative-use-block-verification \
    --speculative-draft-kv-cache-dtype fp8_e4m3 \
    --enable-linear-replayssm-spec \
    --enable-hierarchical-cache \
    --hicache-size 160 \
    --weight-loader-prefetch-checkpoints \
    --weight-loader-prefetch-num-threads 16 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}' \
    --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
    --tokenizer-worker-num 8
```

Everything else (CUDA graph batch sizes, page size, draft attention backend, scheduling policy, mamba cache settings, HiCache host memory mode) is left to SGLang's defaults and Kimi-K3 auto-configuration.

## Fixed settings

Keep these as they are on 8x B300:

| Option | Why |
|---|---|
| `--tp-size 8 --dcp-size 8 --ep-size 2` | Attention over 8 GPUs with decode context parallelism; routed experts with expert parallelism 2 (FlashInfer MXFP4 kernels). Change only for a different GPU count. |
| `--kv-cache-dtype fp8_e4m3` | FP8 KV cache for the MLA/DSA layers; doubles KV capacity. |
| `--speculative-*`, `--enable-linear-replayssm-spec` | DFLASH with 5 draft tokens and block verification. The draft attention backend is set to `cutedsl_mla` automatically. |
| `--cuda-graph-backend-prefill breakable` | Enables CUDA graphs for small prefill batches. SGLang turns prefill graphs off by default for KDA models other than GLM-5.3 Flash; setting the backend explicitly keeps them on (tested with Kimi-K3.1). |
| `--weight-loader-prefetch-checkpoints`, `--model-loader-extra-config` | Each rank prefetches a share of the checkpoint shards, then loads with 16 threads. Without them, loading 1.5 TB takes over 30 minutes. |
| `--chunked-prefill-size 16384` | Prefill chunk size. A 16K chunk takes about 1 s and pauses decode for that time. |
| `SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION=0.1` | Memory budget for the DSA indexer's fp32 scratch (default 0.2); 0.1 chunks long prefills a little more and leaves more GPU memory headroom. |
| `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3` | Kimi-K3 reasoning and tool-call formats. |

## Settings to size for your machine and workload

| Option | Depends on | Guidance |
|---|---|---|
| `--mem-fraction-static` | GPU memory, activation headroom | 0.85 on B300 (275 GB). Lower it on GPUs with less memory or if large prefills run out of memory. |
| `--enable-hierarchical-cache --hicache-size` | Host memory | Host prefix cache, in GB per GPU; the node uses `--hicache-size` x 8 GB. Size it as (host RAM - 300 GB - 10-15% headroom) / 8; 160 fits a 2 TB host. Set it explicitly: the default (twice the GPU pool) gives only about 45 GB per GPU here. |
| `--mamba-full-memory-ratio` | Context length and concurrency | Splits GPU memory between the KV cache and the KDA state cache. 1.24 is a starting point. |
| `--max-running-requests` | Latency target | Unset by default (resolves to about 39 on B300, memory limited). Per-request decode speed falls as concurrency rises; lower it if you need faster decode per request. |
| `--weight-loader-prefetch-num-threads`, `num_threads` | Storage | 16 suits local NVMe; lower on network storage. |
| `--tokenizer-worker-num` | CPU cores, request rate | 8 for long prompts at high request rates; default is 1. |

Optional:

- `--dcp-comm-backend fi_a2a` with `SGLANG_DCP_PUSH_COMBINE_MAX_ROWS=1536`: one-kernel DCP reduce for decode batches up to 25 requests (default rows 1024 covers up to 17). Remove both to use the automatic backend.
- `--default-chat-template-kwargs '{"reasoning_effort":"max"}'` to default to maximum reasoning effort.
- `--served-model-name <name>` to expose a different model name.
- `--enable-metrics` for Prometheus metrics.

## Multiple nodes

Run the same server on each node and put the open-source `sgl-router` in front (`sglang-router` 0.3.2 ships in the image). Cache-aware routing keeps a session on the node that holds its prefix:

```bash
python -m sglang_router.launch_router \
  --host 0.0.0.0 --port 30000 \
  --policy cache_aware \
  --worker-urls http://node1:30000 http://node2:30000 http://node3:30000
```

Prefix caches are per node: a server restart drops its cached prefixes, and a session routed to another node starts cold there.

## Startup

After the weights load, the server captures CUDA graphs (prefill graphs about 45 s, decode and verify graphs about 1 minute). The server prints `The server is fired up and ready to roll!` when ready.

## Notes

- In colocated mode, decode pauses while a prefill chunk runs. Long uncached prompts (many 16K chunks in a row) cause multi-second decode stalls for the other requests on the node.
- Images are preprocessed on the GPU (nvJPEG); SGLang main has no switch to move Kimi-K3 image preprocessing to the CPU.
