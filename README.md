<div align="center">

<img src="docs/assets/logo.png" alt="HuggingArch" width="72">

# HuggingArch

**Architecture, cost and capacity of any HuggingFace model — from its model id alone. No weights, optional GPU.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/demo-huggingarch.com-e8891c.svg)](https://www.huggingarch.com)

[English](README.md) · [简体中文](README.zh-CN.md) · [Docs](docs/README.md) · [Live demo](https://www.huggingarch.com)

</div>

<p align="center">
  <img src="docs/assets/architecture.svg" alt="HuggingArch architecture: model truth from HuggingFace and hardware truth from the GPU, around three stages — an agent authors a checked spec, one walk evaluates it, every view reads that walk" width="100%">
</p>

HuggingArch turns a HuggingFace model id into a declarative spec, checks it against the real checkpoint, and evaluates it once. Everything you see — the architecture, per-layer parameters / FLOPs / memory traffic, KV-cache growth, per-GPU memory and latency under a parallel deployment — is read from that one evaluation.

- **Checked against the real model.** An LLM agent writes the spec; it ships only when every tensor's bytes match the checkpoint's safetensors headers and a review against the `forward()` source passes.
- **Numbers with their formulas.** The same walk is bound once to numbers and once to sympy symbols, so every figure can show how it was computed.
- **Deployment-aware.** Parallelism (TP / PP / EP / CP / DP), precision, GPU, batch × context and serving framework are inputs; collectives and kernel fusions are derived from the graph, not special-cased per model.
- **Bounded by hardware, calibrated by measurement.** Latency is a roofline bound; where a kernel has been measured at the per-GPU shape, the measurement sits beside the bound.

**[▶ Try it live](https://www.huggingarch.com)** — any model id, in the browser.

## What you can do

| | |
|---|---|
| **Gallery** | Compare the whole library: size, KV cache and prefill / decode speed against context, side by side. |
| **Inspector** | Open one model: its block graph, per-op cost, and the formula behind each number. |
| **Inference Planner** | Will it fit, and how fast — on which GPUs, under which parallel layout. |
| **Fusion** | Which vLLM / sglang / DeepInfer kernel fusions fire for this model. |
| **Playground** | Compose modules into a new model and see what it costs. |
| **Kernel Bench** | Measured kernel timings across FlashAttention, FlashInfer, FlashMLA, DeepGEMM and more. |

## Quickstart

Requires Python ≥ 3.13, [`uv`](https://docs.astral.sh/uv/) and Node.js.

```bash
git clone https://github.com/shenh10/HuggingArch.git && cd HuggingArch
uv sync
npm ci --prefix web && npm run build --prefix web
uv run python -m backend.api --port 8000          # open http://localhost:8000
```

Or from the command line:

```bash
uv run huggingarch deepseek-ai/DeepSeek-V3 --seq-len 1024
```

## Scope

HuggingArch is an analytical planner, not a profiler or a runtime.

- **Structure is ground truth.** Parameter counts, shapes and weight bytes are checked tensor by tensor against the checkpoint.
- **Latency is a bound.** Without a measured kernel it is a speed-of-light estimate — a planning envelope, not a benchmark.
- **It reads config, headers and source — not weights.** Behaviour that lives only in weights or custom runtime code is out of reach.

## Coverage

| | |
|---|---|
| **Model types** | Text-generation LLMs and image-to-text (vision-language) models. Other modalities — image or audio generation, speech, embedding-only models — are out of scope. |
| **Models** | 52 architecture families; **59 verified model ids** across 18 orgs — DeepSeek, Kimi, Qwen, Llama, Mistral, Gemma, GLM, MiMo, gpt-oss, Step, LongCat, Nemotron, … An unlisted id gets an agent-drafted spec that must pass validation before it is served. |
| **Attention** | MHA, GQA, MLA, sparse (DSA), sliding window, linear attention and SSM hybrids — 188 reusable component DAGs. |
| **Quantization** | FP8, FP4 / MXFP4 / NVFP4, AWQ, GPTQ, ModelOpt, compressed-tensors. |
| **GPUs** | 15 SKUs: A100 through GB300. |
| **Frameworks** | vLLM, sglang, and DeepInfer fusion bindings (91 fusion plans). |

## Learn more

- [Design](docs/design.md) — the spec language, the evaluation core, validation, parallelism. Documentation map: [`docs/README.md`](docs/README.md).
- [Blog](https://shenhan.cc/blog/huggingarch) — why the project exists, how it works, and how it got here ([中文](https://shenhan.cc/zh/blog/huggingarch)).
- [Custom models](docs/custom_models.md) · [Deployment](docs/deployment.md) (including the optional multi-user mode) · [Kernel Bench](kernel_bench/README.md)

## Contributing

The most useful contributions are a new model family (a YAML in `backend/arch/models/`) and a new reusable component (a YAML in `backend/arch/components/library/`) — no Python needed. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If HuggingArch helps your work, please cite:

```bibtex
@misc{shen2026huggingarch,
  author       = {Shen, Han},
  title        = {HuggingArch: Automating Model Architecture Analysis},
  year         = {2026},
  howpublished = {\url{https://shenhan.cc/blog/huggingarch}},
  note         = {Code: \url{https://github.com/shenh10/HuggingArch}}
}
```

## License

MIT — see [LICENSE](LICENSE).
