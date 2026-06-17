# How Sluice compares

DeepSeek-V4-Pro (FP8) is ~**805 GiB** of weights, almost all experts. That does
not fit GPU-resident on a single 8×H100 node (**640 GiB**), so the engine either
needs **more GPUs** (≈2 nodes) or has to **offload the experts** off the GPU.

This page compares how the main inference engines handle that, as of June 2026.
The honest axis isn't just *where the experts live* but **where they compute**.

| Engine | Routed experts live in | Experts compute on | Runs V4-Pro on ≤8×H100 | Serving stack |
|---|---|---|---|---|
| **Sluice** (vLLM plugin) | host RAM, streamed into a small **GPU** cache (LRU, by routing) | **GPU** | ✅ — we ran it on **4×H100** + ~1 TB RAM | full vLLM (continuous batching, etc.) |
| KTransformers | host RAM (shared experts on GPU) | **CPU** (AMX kernels) | ✅ — even 1 GPU (48 GB+) + big RAM | its own runtime |
| llama.cpp | host RAM (`--n-cpu-moe` / `-ot exps=CPU`) | **CPU** | ✅ — flexible CPU/GPU split, GGUF | its own runtime |
| SGLang | GPU-resident (CPU offload is a WIP KTransformers integration) | GPU | ❌ not on one node — needs ~2 nodes (≥~11×H100) for the FP8 weights | full SGLang |
| vLLM (stock) | GPU-resident (`--cpu-offload-gb` is static, whole-tensor, routing-blind) | GPU | ❌ same — experts don't fit | full vLLM |

> Model-architecture support varies by project and moves fast. vLLM and SGLang
> support V4 day-0; KTransformers and llama.cpp add architectures on their own
> cadence. This table compares the **offloading approach and memory
> feasibility**, not who shipped a given model on a given day.

## The two families

**CPU-compute offload — KTransformers, llama.cpp.** Routed experts stay in host
RAM and are *computed on the CPU* (AMX / optimized GGUF kernels); attention,
embeddings and shared experts sit on the GPU. This needs the least GPU — a single
high-VRAM card plus a big-RAM box can serve a 671B-class model — but throughput
is CPU/memory-bandwidth bound and you leave vLLM/SGLang's GPU-serving stack.
KTransformers reports large speedups over prior CPU/GPU-hybrid systems and is the
reference design here (SOSP'25); llama.cpp exposes the same idea through
`--n-cpu-moe` and `-ot`. Notably, llama.cpp's own
[issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757) requests a
"two-tier GPU+RAM expert cache with pluggable eviction" — which is exactly what
Sluice does.

**GPU-compute, weights must be reachable — SGLang, vLLM, and Sluice.** These run
the experts on the GPU. SGLang and stock vLLM keep the weights GPU-resident, so
V4-Pro needs ≈2 nodes; SGLang's CPU offload is an
[in-progress KTransformers integration](https://github.com/sgl-project/sglang/issues/11425).
**Sluice** is the difference: it keeps the experts in host RAM and streams just
the router-selected ones into a small GPU cache each step, so V4-Pro runs on
4×H100 **without leaving vLLM** — you keep GPU compute, continuous batching, and
the rest of the serving stack, at the cost of PCIe bandwidth for the streamed
experts.

## So which should you use?

- **Fewest GPUs / GPU-poor, low concurrency** → KTransformers or llama.cpp
  (experts on CPU). Best when you have one big GPU and lots of RAM.
- **Maximum throughput and you have the VRAM** (or enough nodes) → SGLang or
  stock vLLM, GPU-resident.
- **Keep vLLM's serving stack but the experts don't fit** → Sluice. It's the
  narrow-but-real niche: routing-aware expert offload *inside* vLLM, trading some
  throughput (~14% when the cache covers the working set; ~17 tok/s on V4-Pro
  eager) to fit a model that otherwise can't load.

## Sources

- KTransformers — [repo](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md),
  [SOSP'25 paper](https://dl.acm.org/doi/10.1145/3731569.3764843)
- llama.cpp — [MoE offload guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide),
  [expert-cache feature request #20757](https://github.com/ggml-org/llama.cpp/issues/20757)
- SGLang — [KTransformers integration #11425](https://github.com/sgl-project/sglang/issues/11425),
  [DeepSeek-V4 day-0](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
