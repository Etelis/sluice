# SPDX-License-Identifier: Apache-2.0
"""Correctness demo: streaming experts must not change the output.

Run the same greedy decode twice on DeepSeek-V2-Lite (small enough to fit
either way) and diff the token ids — once normally, once with Sluice active:

    # baseline (all experts GPU-resident)
    python examples/bitcompare_v2lite.py > baseline.txt

    # offloaded (experts streamed from CPU into 64 slots)
    SLUICE_SLOTS=64 python examples/bitcompare_v2lite.py > sluice.txt

    diff baseline.txt sluice.txt && echo "BIT-IDENTICAL"

``slots=64`` equals V2-Lite's local expert count, so this exercises the
CPU->GPU streaming and the ``expert_map`` repoint without LRU eviction; use a
smaller value to exercise eviction. Unquantized models need an ``expert_map``-
honoring backend — set ``moe_backend`` accordingly for your vLLM build.
"""

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite",
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=0.5,
    )
    prompts = ["The capital of France is", "Q: What is 2+2? A:", "Once upon a time"]
    out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
    for o in out:
        print(o.prompt, "->", list(o.outputs[0].token_ids))


if __name__ == "__main__":
    main()
