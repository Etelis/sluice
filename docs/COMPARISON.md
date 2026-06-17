# How Sluice compares

DeepSeek-V4-Pro (FP8) is ~**805 GiB** of weights, almost all experts. That does
not fit GPU-resident on a single 8×H100 node (**640 GiB**), so the engine either
needs **more GPUs** (≈2 nodes) or has to **offload the experts** off the GPU.

The honest axis isn't just *where the experts live* but **where they compute**.
Here's every engine, and what actually happens on our **4×H100 (320 GiB)** box
loading the **stock** DeepSeek-V4-Pro FP8 checkpoint:

| Engine | Routed experts | Expert compute | Tested on 4×H100, stock FP8 checkpoint |
|---|---|---|---|
| **Sluice** (vLLM) | host RAM → small **GPU** cache (LRU, by routing) | **GPU** | ✅ **runs · ~17 tok/s** |
| stock vLLM | GPU-resident (`--cpu-offload-gb` is static, routing-blind) | GPU | ❌ **OOM** — GPU fills to ~78 GiB |
| SGLang | GPU-resident (`kt_*` offload path needs KT-format weights) | GPU | ❌ **OOM** — GPU fills to ~76 GiB |
| KTransformers | host RAM (shared experts on GPU) | **CPU** (AMX) | ⚠️ not runnable here — needs KT-format weights + a V4 recipe |
| llama.cpp | host RAM (`--n-cpu-moe` / `-ot exps=CPU`) | **CPU** | ⚠️ not runnable here — needs GGUF + V4 arch support |

> **What "tested" means.** The three GPU-compute engines were run directly from
> the stock checkpoint on 4×H100: Sluice serves; stock vLLM and SGLang OOM
> (805 GiB of weights ≫ 320 GiB — and a single 8×H100 node, 640 GiB, is short
> too). KTransformers and llama.cpp use a different, **CPU-compute** approach
> that can serve 671B-class MoEs on far less GPU — but not from this FP8
> checkpoint without converting to their own weight formats and V4 architecture
> support landing in-tree, so they're compared by approach, not re-run here.
> Model support moves fast (vLLM and SGLang have V4 day-0); this is a June 2026
> snapshot.

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
V4-Pro needs ≈2 nodes. SGLang has gained a
[KTransformers offload path](https://github.com/sgl-project/sglang/issues/11425)
(exposed as `kt_weight_path` / `kt_method` server args), but it needs
KT-format weights — the stock FP8/FP4 checkpoint loads GPU-resident.
**Sluice** is the difference: it keeps the experts in host RAM and streams just
the router-selected ones into a small GPU cache each step, so V4-Pro runs on
4×H100 **without leaving vLLM** — you keep GPU compute, continuous batching, and
the rest of the serving stack, at the cost of PCIe bandwidth for the streamed
experts.

> **Verified on our cluster (June 2026), all on the same 4×H100 box.**
> - **SGLang** `0.5.13.post1` recognizes `DeepseekV4ForCausalLM` and loads V4-Pro
>   day-0, but with the stock checkpoint **OOMs** creating the FP8 expert weights
>   (`fp8.py:create_weights` → GPU 0 ~76 GiB, `exit 137`).
> - **stock vLLM** (no Sluice) **OOMs** the same way (GPU ~78 GiB, engine-init
>   failure).
> - **Sluice** loads and serves the same model at **~17 tok/s**.
>
> 805 GiB of weights ≫ 320 GiB (4×H100), and ≫ 640 GiB (8×H100), so the
> GPU-resident engines can't fit it on a single node either way.

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
