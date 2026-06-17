# SPDX-License-Identifier: Apache-2.0
"""Flagship example: DeepSeek-V4-Pro on H100s with Sluice expert offloading.

The FP8 expert weights are far larger than one GPU's memory, so a normal load
OOMs. With Sluice active (``SLUICE_SLOTS`` set), the experts live in host RAM
and only the router-selected ones are streamed into a small GPU cache per step.

Run (4x H100, expert parallel):

    SLUICE_SLOTS=16 python examples/run_dsv4_ep4.py

Notes:
  * ``moe_backend="marlin"`` — Sluice needs a backend that honors ``expert_map``
    (marlin for NVFP4/FP8, triton for unquantized). FlashInfer-style backends
    ignore ``expert_map`` and will NOT work.
  * ``gpu_memory_utilization=0.45`` — vLLM sizes the KV cache from the weight
    memory it measures at load time, which (correctly) excludes the offloaded
    experts; it does not see Sluice's slot caches (installed post-load). Lower
    the utilization to leave room for them. See the README "KV cache" note.
  * ``kv_cache_dtype="fp8"`` is required by DeepSeek-V4.
"""

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="deepseek-ai/DeepSeek-V4-Pro",
        trust_remote_code=True,
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        moe_backend="marlin",
        kv_cache_dtype="fp8",
        enforce_eager=True,
        max_model_len=512,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.45,
    )
    prompts = ["Hello", "The capital of France is"]
    out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=24))
    for o in out:
        print(repr(o.prompt), "->", repr(o.outputs[0].text))


if __name__ == "__main__":
    main()
