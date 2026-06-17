<p align="center">
  <img src="assets/sluice-icon.png" alt="Sluice" width="120">
</p>

<h1 align="center">Sluice</h1>

<p align="center">
  <b>Routing-aware MoE expert offloading for vLLM</b> — a plugin, not a fork.
</p>

<p align="center">
  Run Mixture-of-Experts models whose experts exceed GPU memory — like
  <b>DeepSeek-V4</b> — by keeping experts in host RAM and streaming only the
  router's picks into a small per-layer GPU cache each step.
</p>

<p align="center"><img src="assets/chart-comparison.png" alt="V2-Lite fits a GPU; V4-Pro's experts don't, until Sluice" width="680"></p>

## Quickstart

```bash
pip install -e .          # into an environment that already has vLLM
SLUICE_SLOTS=16 python examples/run_dsv4_ep4.py
```

`SLUICE_SLOTS` = resident experts per layer, per rank. Unset → Sluice is inert.

## Results

Stock vLLM, no fork. **V2-Lite** offloaded output is **bit-identical** to the
resident baseline (greedy, token ids). **V4-Pro** (FP8, EP=4) loads with just
**9.49 GiB** of GPU weights and answers correctly:

> *The capital of France is* → **Paris. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Japan is Tokyo.**

<p align="center"><img src="assets/chart-residency.png" alt="16 of 96 experts resident per layer/rank" width="560"></p>

Streaming experts on demand costs only **~14%** throughput vs keeping them all
resident (when the cache covers the per-step working set) — and lets V4-Pro
serve at all, where it otherwise OOMs.

<p align="center"><img src="assets/chart-throughput.png" alt="Decode throughput: ~14% overhead on V2-Lite; V4-Pro runs only with Sluice" width="700"></p>

## Deploy DeepSeek-V4

```bash
SLUICE_SLOTS=16 vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --moe-backend marlin --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.45 --enforce-eager --trust-remote-code
```

Needs an `expert_map`-honoring backend (`marlin` / `triton`, **not** FlashInfer);
the low `--gpu-memory-utilization` leaves VRAM for the slot caches. Design,
load/forward paths, and the gotchas: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Compared to other engines

V4-Pro (FP8) is ~805 GiB — it can't sit GPU-resident on one 8×H100 node (640 GiB).

| Engine | Routed experts | Expert compute | V4-Pro on ≤8×H100 |
|---|---|---|---|
| **Sluice** (vLLM) | host RAM → streamed to GPU cache | **GPU** | ✅ (ran on 4×H100) |
| KTransformers | host RAM | CPU | ✅ |
| llama.cpp | host RAM (`-ot` / `--n-cpu-moe`) | CPU | ✅ (GGUF) |
| SGLang · stock vLLM | GPU-resident | GPU | ❌ needs ~2 nodes |

Sluice is the only one that streams experts onto the **GPU** for compute **inside
vLLM's serving stack** — KTransformers and llama.cpp run experts on CPU (less GPU,
but CPU-bound); SGLang is fastest yet must fit the weights in VRAM.
[Full comparison + sources →](docs/COMPARISON.md)

---

<p align="center">
  Apache-2.0 · built on <a href="https://github.com/vllm-project/vllm">vLLM</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="docs/COMPARISON.md">Comparison</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>
