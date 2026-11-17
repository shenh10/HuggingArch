<div align="center">

<img src="docs/assets/logo.png" alt="HuggingArch" width="72">

# HuggingArch

**只凭一个 model id，看清任何 HuggingFace 模型的结构、成本和部署容量。不下权重，GPU 可选。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Live demo](https://img.shields.io/badge/demo-huggingarch.com-e8891c.svg)](https://www.huggingarch.com)

[English](README.md) · [简体中文](README.zh-CN.md) · [文档](docs/README.md) · [在线体验](https://www.huggingarch.com)

</div>

<p align="center">
  <img src="docs/assets/architecture.svg" alt="HuggingArch 架构：上层是来自 HuggingFace 的模型事实，下层是来自 GPU 的硬件事实，中间三步——agent 写出经校验的 spec，求值一次，所有页面读同一次结果" width="100%">
</p>

HuggingArch 把一个 HuggingFace model id 变成一份声明式 spec，与真实 checkpoint 对账之后只求值一次。你看到的一切——模型结构、逐层参数量 / FLOPs / 访存、KV cache 增长、并行部署下的单卡显存与延迟——都读自这一次求值。

- **对着真实模型校验。** spec 由 LLM agent 撰写；只有每个张量的字节数都和 checkpoint 的 safetensors 头对上，并且对照 `forward()` 源码的审查通过，才会上线。
- **数字带着公式。** 同一次走访，一次绑定数值，一次绑定 sympy 符号，所以每个数字都能展示它是怎么算出来的。
- **面向部署。** 并行方式（TP / PP / EP / CP / DP）、精度、GPU、batch × context 和推理框架都是输入；通信和 kernel 融合从计算图推出，不按模型逐个特判。
- **硬件给上界，实测来校准。** 延迟是 roofline 上界；某个 kernel 在单卡真实 shape 下测过的，实测值就和上界并排显示。

**[▶ 在线体验](https://www.huggingarch.com)**——输入任意 model id，浏览器里直接看。

## 能做什么

| | |
|---|---|
| **Gallery** | 横向比较整个模型库：规模、KV cache、prefill / decode 速度随上下文的变化。 |
| **Inspector** | 打开一个模型：block 图、逐算子成本，以及每个数字背后的公式。 |
| **Inference Planner** | 装不装得下、跑多快——用哪些卡、哪种并行方案。 |
| **Fusion** | 这个模型在 vLLM / sglang / DeepInfer 上会命中哪些 kernel 融合。 |
| **Playground** | 用模块拼一个新模型，看它的成本。 |
| **Kernel Bench** | FlashAttention、FlashInfer、FlashMLA、DeepGEMM 等算子库的实测耗时。 |

## 快速开始

需要 Python ≥ 3.13、[`uv`](https://docs.astral.sh/uv/) 和 Node.js。

```bash
git clone https://github.com/shenh10/HuggingArch.git && cd HuggingArch
uv sync
npm ci --prefix web && npm run build --prefix web
uv run python -m backend.api --port 8000          # 打开 http://localhost:8000
```

或者用命令行：

```bash
uv run huggingarch deepseek-ai/DeepSeek-V3 --seq-len 1024
```

## 边界

HuggingArch 是分析型的规划工具，不是 profiler，也不是推理 runtime。

- **结构是事实。** 参数量、形状和权重字节都与 checkpoint 逐张量对账。
- **延迟是上界。** 没有实测数据时，它是 speed-of-light 估计，用来做规划，不是 benchmark 结果。
- **读 config、权重头和源码，不读权重本身。** 只存在于权重数值或自定义 runtime 代码里的行为，它看不到。

## 覆盖范围

| | |
|---|---|
| **模型类型** | 文本生成 LLM，以及图像到文本（视觉语言）模型。图像 / 音频生成、语音、纯 embedding 模型等其他模态暂不支持。 |
| **模型** | 52 个架构族；**59 个已验证 model id**，覆盖 18 家机构——DeepSeek、Kimi、Qwen、Llama、Mistral、Gemma、GLM、MiMo、gpt-oss、Step、LongCat、Nemotron 等。不在列表里的 id 由 agent 起草 spec，校验通过后才会上线。 |
| **注意力** | MHA、GQA、MLA、稀疏注意力（DSA）、滑窗、线性注意力与 SSM 混合结构——188 个可复用 component DAG。 |
| **量化** | FP8、FP4 / MXFP4 / NVFP4、AWQ、GPTQ、ModelOpt、compressed-tensors。 |
| **GPU** | 15 种型号：从 A100 到 GB300。 |
| **推理框架** | vLLM、sglang 和 DeepInfer 的融合绑定（91 个融合 plan）。 |

## 延伸阅读

- [设计文档](docs/design.md)——spec 语言、求值核心、校验、并行。文档索引见 [`docs/README.md`](docs/README.md)。
- [博客](https://shenhan.cc/zh/blog/huggingarch)——项目为什么存在、怎么工作、怎么一步步走到现在（[English](https://shenhan.cc/blog/huggingarch)）。
- [自定义模型](docs/custom_models.md) · [部署](docs/deployment.md)（含可选的多用户模式） · [Kernel Bench](kernel_bench/README.md)

## 贡献

最有价值的贡献是新的模型家族（在 `backend/arch/models/` 加一份 YAML）和新的可复用组件（在 `backend/arch/components/library/` 加一份 YAML），都不需要写 Python。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 引用

如果 HuggingArch 对你的工作有帮助，请引用：

```bibtex
@misc{shen2026huggingarch,
  author       = {Shen, Han},
  title        = {HuggingArch: Automating Model Architecture Analysis},
  year         = {2026},
  howpublished = {\url{https://shenhan.cc/blog/huggingarch}},
  note         = {Code: \url{https://github.com/shenh10/HuggingArch}}
}
```

## 许可证

MIT，见 [LICENSE](LICENSE)。
