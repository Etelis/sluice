# Sluice

**Routing-aware MoE expert offloading for vLLM.** Run Mixture-of-Experts models
whose expert weights are far larger than GPU memory — including
**DeepSeek-V4** — by keeping the experts in host RAM and streaming only the
ones the router selects into a small GPU cache, per forward step.

Sluice ships as a **vLLM plugin**, not a fork. `pip install`, set one
environment variable, run stock vLLM.

---

## The problem

In a Mixture-of-Experts model, each token is routed to a small number of
experts (top-k), but the *full* set of experts must be available to the kernel.
For a model like DeepSeek-V4 the experts dominate the weight budget and do not
come close to fitting in a single GPU — even sharded with expert parallelism
across 8 GPUs, each rank's expert shard exceeds 80 GB.

vLLM can offload whole tensors to CPU (`--cpu-offload-gb`), but that offload is
**static and routing-blind**: it decides what lives where at load time and
moves fixed tensors across the PCIe bus every step regardless of which experts
a token actually needs. There is no mechanism that follows the router.

## What Sluice does

Sluice treats the GPU as a small **cache of experts** and the router as the
access pattern — essentially demand paging for experts:

- The full per-rank expert shard lives in **host RAM**.
- Each MoE layer keeps a fixed-size **GPU slot cache** (e.g. 16 experts) instead
  of all of them.
- Every forward step, right after the router picks `topk_ids`, Sluice streams
  exactly those experts CPU→GPU into free/LRU slots and rewrites the layer's
  `expert_map` so the **unmodified** fused-MoE kernel indexes the cache.

Because the kernel already supports `expert_map` (global expert id → resident
row), no kernel changes are needed — Sluice just keeps the map and the slot
contents in sync with routing.

## Quickstart

```bash
# 1. Install vLLM (pin a known-good version; see "Compatibility" below),
#    then install Sluice into the same environment.
pip install -e .

# 2. Run any MoE model with experts offloaded. SLUICE_SLOTS = GPU-resident
#    experts per layer, per rank.
SLUICE_SLOTS=16 python examples/run_dsv4_ep4.py
```

That's it — no `--offload-*` flags, no fork. When `SLUICE_SLOTS` is unset Sluice
is completely inert and vLLM behaves normally.

### Correctness demo

Streaming experts on demand must not change the output. On DeepSeek-V2-Lite
(small enough to fit either way), greedy decode is **bit-identical** with and
without Sluice:

```bash
python examples/bitcompare_v2lite.py > baseline.txt                # all experts resident
SLUICE_SLOTS=64 python examples/bitcompare_v2lite.py > sluice.txt   # streamed from CPU
diff baseline.txt sluice.txt && echo "BIT-IDENTICAL"
```

## Deploying DeepSeek-V4

**You do not need a vLLM fork.** Two separate things are involved, and both are
already available without forking:

- **V4 model support** is *upstream*: `DeepseekV4ForCausalLM` is in vLLM's model
  registry in recent builds. Sluice adds nothing here.
- **Expert offloading** is *this plugin*.

You need three things:

1. **A vLLM build that supports DeepSeek-V4.** Check:
   ```bash
   python -c "from vllm import ModelRegistry; print('DeepseekV4ForCausalLM' in ModelRegistry.get_supported_archs())"
   ```
   (Validated against vLLM `0.22.1rc1.dev26+g4721bb3aa`.)
2. **Sluice installed in the same environment** (`pip install -e .`).
3. **Enough host RAM for the experts.** The FP8 checkpoint is ~805 GiB; across
   expert-parallel ranks the offloaded experts occupy roughly that much host RAM
   in total. Validated on 4×H100-80GB + ~1.1 TiB RAM.

### Online (OpenAI-compatible server)

```bash
SLUICE_SLOTS=16 \
VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve deepseek-ai/DeepSeek-V4-Pro \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --moe-backend marlin \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.45 \
    --enforce-eager \
    --max-model-len 512 \
    --trust-remote-code
```

### Offline

```bash
SLUICE_SLOTS=16 python examples/run_dsv4_ep4.py
```

### Why each setting

| Setting | Why |
|---|---|
| `SLUICE_SLOTS=16` | Activates Sluice; 16 resident experts per layer per rank. |
| `--moe-backend marlin` | FP8 backend that honors `expert_map`. FlashInfer ignores it → wrong output. |
| `--kv-cache-dtype fp8` | Required by DeepSeek-V4. |
| `--enable-expert-parallel` | Shards experts across ranks; smaller per-rank host + slot footprint. |
| `--gpu-memory-utilization 0.45` | Leaves VRAM for the post-load slot caches (see the KV-cache caveat). |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Avoids fragmentation OOMs. |
| `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` | Optional; skip pinning hundreds of GB of host RAM. |

## Results

Validated on a single 8×H100-80GB node.

**DeepSeek-V2-Lite (BF16), single GPU — correctness:** offloaded output is
**bit-identical** to the resident baseline on every prompt (greedy decode, token
ids compared exactly).

**DeepSeek-V4-Pro (FP8), 4×H100, expert-parallel — scale:** a normal load
cannot fit the experts. With Sluice:

| | Value |
|---|---|
| Checkpoint size | 805 GiB (FP8) |
| GPU weight memory measured at load | **9.49 GiB** (experts offloaded to host) |
| Per-rank cache | **16 of 96** local experts resident per layer |
| KV cache after offload | 24.4 GiB / ~46.7k tokens |
| Output | coherent and correct |

```
'The capital of France is' -> ' Paris. The capital of Germany is Berlin.
                               The capital of Italy is Rome. The capital of
                               Japan is Tokyo.'
```

The coherent output is the real proof: if the per-step `expert_map` repoint or
the CPU→GPU stream were off by a single row, the logits would be garbage.

## How it works

### Load path

1. **`wrap_modules`** (model construction, before the checkpoint loads): move
   each MoE layer's expert parameters to CPU so the checkpoint fills host
   memory, and record the layer.
2. **`process_weights_after_loading`** runs inside vLLM's
   `device_loading_context`, which stages each layer's experts back to the GPU
   for processing (e.g. the CUDA-only FP4/FP8 marlin repack) and restores them
   to CPU afterward. Only one layer is on the GPU at a time, so a shard far
   larger than GPU memory can be processed.
3. **`post_init`** (after load completes): install the `[num_slots, …]` GPU slot
   cache as the live weights, keep the processed experts in host memory, and
   wrap the layer's `quant_method.apply` so the routing hook fires each forward.

> **Why `post_init`, not during `process_weights`?** `device_loading_context`
> records each parameter's device on entry (CPU, because `wrap_modules` parked
> it there) and **restores it on exit** — so installing the GPU slot cache
> inside `process_weights` is silently undone the moment the context exits. The
> cache must be installed *after* that restore. This was the central bug to get
> right.

### Forward path

After the router selects experts, the wrapped `apply` calls `on_experts_selected(layer, topk_ids)`:

- de-duplicate the step's selected experts;
- for each, ensure it occupies a GPU slot (stream it in from host if not,
  evicting the least-recently-used slot when full);
- write `expert_map[global_id] = slot` for the resident experts (and `-1` for
  the rest) so the fused-MoE kernel reads the right rows.

When the slot count ≥ the layer's local experts, every expert simply gets a
fixed slot (no eviction); below that, it's an LRU working set.

## Plugin, not a fork

Sluice changes **zero** vLLM source files. It hooks two things:

1. **Offloader selection.** A `vllm.general_plugins` entry point runs
   `sluice.plugin.register()` in every engine/worker process at startup. When
   `SLUICE_SLOTS` is set, it monkeypatches vLLM's `create_offloader` factory to
   return Sluice's `ExpertStreamOffloader`. vLLM then drives the standard
   `BaseOffloader` lifecycle (`wrap_modules` → `post_init`) on it.
2. **The routing hook.** Rather than editing the MoE runner to call the hook,
   Sluice **wraps each layer's `quant_method.apply`** in `post_init`. The
   modular MoE path already calls `apply(layer=…, topk_ids=…, …)`, so everything
   the hook needs is on hand — no source edit required.

That's the whole integration surface: one monkeypatched factory function plus a
method wrapper installed through vLLM's own offloader interface.

If you maintain a vLLM fork, the same offloader can instead be wired with a
one-line call in `moe_runner.py` after expert selection; the plugin path just
avoids needing the fork.

## Compatibility & caveats

- **Backend must honor `expert_map`.** Use a modular MoE backend: `marlin` for
  NVFP4/FP8 (`moe_backend="marlin"`), TRITON for unquantized. **FlashInfer-style
  / monolithic backends ignore `expert_map`** and will produce wrong output —
  they route inside the kernel and never expose `topk_ids`.
- **KV cache sizing.** vLLM sizes the KV cache from the weight memory it measures
  at load time. The offloaded experts are (correctly) excluded — but so are
  Sluice's slot caches, which are installed *after* that measurement. vLLM
  therefore over-budgets KV and can OOM. **Lower `gpu_memory_utilization`** to
  leave room for the slot caches (e.g. 0.45 for DeepSeek-V4 on H100), and
  consider `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Choose `SLUICE_SLOTS` ≥ the distinct experts a single step can select.** For
  decode that's small (≈ top-k per rank); a large prefill batch can touch many,
  so the startup profiling run may briefly overflow and log a warning — that is
  expected and harmless. If you see the warning during real decode, raise the
  slot count.
- **vLLM version.** Sluice monkeypatches an internal factory and relies on the
  modular `quant_method.apply(layer, …, topk_ids=…)` signature. Pin a
  known-good vLLM; expect to revisit on major vLLM refactors.
- **Throughput.** Sluice trades PCIe bandwidth for VRAM. With pinned host memory
  and a slot count that covers the working set, streaming overlaps compute well;
  the goal is *feasibility* for models that otherwise will not load at all.

## Tuning

| Knob | Effect |
|---|---|
| `SLUICE_SLOTS` | GPU-resident experts per layer per rank. Higher = more VRAM, fewer transfers. |
| `gpu_memory_utilization` | Lower to leave room for slot caches (see KV note). |
| `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` | Disable pinned host memory if host RAM is tight (slower transfers). |

## Roadmap

- Upstream a generic `on_experts_selected` offloader hook to vLLM so neither the
  monkeypatch nor the `apply` wrap is needed.
- Teach vLLM's KV-cache profiler about post-load offloader allocations so
  `gpu_memory_utilization` need not be hand-tuned.
- Prefetch next-layer experts to hide transfer latency behind compute.

## License

Apache-2.0. Built on and for [vLLM](https://github.com/vllm-project/vllm).
