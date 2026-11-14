# HuggingArch documentation

Start with the [project README](../README.md) for what HuggingArch does and how
to run it, and the [blog post](https://shenhan.cc/blog/huggingarch)
([中文](https://shenhan.cc/zh/blog/huggingarch)) for the long-form motivation
and a walkthrough with screenshots. [`design.md`](design.md) is the technical
overview; every other document goes deeper into one subsystem.

## Overview

| Document | Covers |
|---|---|
| [`design.md`](design.md) | The complete technical walkthrough: the spec system (primitives, components, model specs), the evaluation core, information sources, validation, quantization, the agentic spec-generation pipeline, the user-facing pages, the spec authoring contract, and the compiler view of the system. |

## Spec system and evaluation

| Document | Covers |
|---|---|
| [`evaluation_model.md`](evaluation_model.md) | The core behind every number: `SpecIRTree` → `Program` → `Analysis`, the kinds of analysis layers, first-class shape propagation, why bytes are derived, and per-rank evaluation. |
| [`symbolic_evaluation.md`](symbolic_evaluation.md) | How one quantity travels from the walk to the page: one walk over symbolic workload counts, conditions and cases instead of decisions, reading a point, and cell reuse. |
| [`validation.md`](validation.md) | Why the numbers can be trusted. Part I: the spec publish gates (the checkpoint byte join, KV and window claims, ablation levers, diagnostics, the archived guard-ablation evidence). Part II: the A–E self-checks on inference results (axioms, conservation, perturbation invariants, type contracts, dual-path cross-checks). |
| [`quantization.md`](quantization.md) | Why structure is dtype-agnostic and storage is not: per-role quant context, schemes as data, per-tensor bytes, deployment precision profiles, and activation casts as a graph rewrite. |

## Model structure

| Document | Covers |
|---|---|
| [`kimi-k31-sglang.md`](kimi-k31-sglang.md) | Kimi-K3.1 SGLang estimates, parallel settings, and cache layout. |
| [`attention_families.md`](attention_families.md) | The attention variants (MHA/GQA, MLA, SWA, DSA, CSA/HCA, linear/SSM) and the three cost axes that distinguish them: compute, KV storage law, KV read cardinality. |
| [`kv_cache_storage.md`](kv_cache_storage.md) | How KV-cache storage is modelled per cached node (windowed / per-token / constant state) and how claims cross-check it at every component call. |
| [`moe_expert_axis.md`](moe_expert_axis.md) | How a MoE module is modelled: the experts are the outermost axis of every tensor of a bank, where each consumer reads the expert count, the mean rows per expert and the routing spread, and the activation peak. |

## Inference cost

| Document | Covers |
|---|---|
| [`inference_cost_model.md`](inference_cost_model.md) | Prefill and decode on the roofline: per-op compute, memory and communication time, why speed-of-light is a per-op sum, pipeline stages, fusion, capacity, and the headline TTFT / TPOT / throughput numbers. |
| [`parallelism.md`](parallelism.md) | How one model splits across GPUs: the parallel configuration, each weight's declared placement, sharding as a propagated state with injected communication nodes, and per-rank cost. Ends with the layout terminology table. |
| [`auto_sweep.md`](auto_sweep.md) | How a sweep prices many points off few walks: cells and points, substitution, cross-card re-pricing, one measurement lookup per sweep, the all-or-nothing measured total, caching, and baking at deploy. |
| [`measured_calibration.md`](measured_calibration.md) | Measured kernel calibration end to end: the kernel_bench harness, a measured row and its physical key, the Kernel Bench lake and service, private and shared rows, and how estimates look measurements up. |
| [`known_limitations.md`](known_limitations.md) | Where the cost model knowingly departs from real serving, with the direction of the error and the shape of the fix. |

## Guides

| Document | Covers |
|---|---|
| [`custom_models.md`](custom_models.md) | Plugging in a model that is not on the Hugging Face Hub via the `custom:<name>` protocol. |
| [`deployment.md`](deployment.md) | Building and deploying the service: the Docker image, Fly.io, the CI workflows, Supabase migrations, cache warm-up, custom domain. |

## Code-adjacent references

These live next to the code they describe.

| Document | Covers |
|---|---|
| [`../backend/arch/DESIGN.md`](../backend/arch/DESIGN.md) | Spec-system internals: Program / Analysis / report boundaries, `module` naming and alias adapters, loader BFS. |
| [`../backend/inference/DESIGN.md`](../backend/inference/DESIGN.md) | Inference internals: parallel configuration, the L1/L2/L3 layering, per-attention-type KV math, the phase model. |
| [`../backend/GLOSSARY.md`](../backend/GLOSSARY.md) · [`../backend/arch/dims.yaml`](../backend/arch/dims.yaml) | Term glossary and the dimension-symbol registry. |
| [`../backend/arch/components/README.md`](../backend/arch/components/README.md) | The component library (library / drafts / adapters tiers). |
| [`../kernel_bench/README.md`](../kernel_bench/README.md) · [`../kernel_bench/ARCHITECTURE.md`](../kernel_bench/ARCHITECTURE.md) | The kernel_bench CLI and its internals (how a launch is timed, run records). |
| [`../backend/inference/BENCHMARKS_CSV.md`](../backend/inference/BENCHMARKS_CSV.md) | The measured-row schema. |
| [`../supabase/README.md`](../supabase/README.md) | Supabase schema and migrations (multi-user mode). |

## Reading paths

- **Use HuggingArch** → the project README → the blog → `custom_models.md` if needed.
- **Understand the codebase** → `design.md` → `evaluation_model.md` → the topic documents.
- **Know why the numbers can be trusted** → `validation.md` → `kv_cache_storage.md` → `known_limitations.md`.
- **Understand TTFT / TPOT / max batch** → `inference_cost_model.md` → `kv_cache_storage.md`.
- **Understand TP / EP / CP / DP sharding** → `parallelism.md` → `inference_cost_model.md`.
- **Use measured numbers or add a kernel** → `measured_calibration.md` → `kernel_bench/README.md`.
- **Add a model family or a component** → `design.md` §1 and §8 → `backend/arch/components/README.md` → `validation.md`.
