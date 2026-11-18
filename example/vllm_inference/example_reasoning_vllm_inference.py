"""

Example: Runtime reasoning compression in vLLM (budgets / ThinkBrake)

Registers OneComp's per-request logits processor in a vLLM engine and runs one
batch in which every request carries its own policy. Requests without
``onecomp_reasoning`` metadata are left exactly unchanged.

Supported in vLLM: none, fixed, prompt_budget and thinkbrake. Probe-based
policies (deer, stability) need the Transformers reference backend and are
rejected here. For a small accuracy/latency experiment over several questions,
see example_reasoning_compression.py.

Requirements:
  pip install vllm   (the logits processor was validated against vLLM 0.21.x)

Copyright 2025-2026 Fujitsu Ltd.

"""

from dataclasses import replace

from vllm import LLM, SamplingParams

from onecomp.reasoning import ReasoningConfig, ReasoningFormat, estimate_prompt_budget
from onecomp.reasoning.data import extract_answer
from onecomp.runtime.vllm import VLLMReasoningAdapter

MODEL_ID = "Qwen/Qwen3-8B"
QUESTION = (
    "Solve the following problem. Put only your final answer inside \\boxed{}.\n\n"
    "What is the smallest positive integer n such that 3n is a perfect square "
    "and 2n is a perfect cube?"
)


def main():
    llm = LLM(
        model=MODEL_ID,
        max_model_len=20000,
        logits_processors=[VLLMReasoningAdapter.processor_path],
    )
    tokenizer = llm.get_tokenizer()
    reasoning_format = ReasoningFormat.from_tokenizer(tokenizer)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=False,
    )

    policies = {
        "vanilla": None,
        "fixed_1024": ReasoningConfig(policy="fixed", budget=1024),
        # The caller substitutes the per-prompt budget into the config.
        "prompt_budget": replace(
            ReasoningConfig(policy="prompt_budget"), budget=estimate_prompt_budget(QUESTION)
        ),
        # Stops at a sentence boundary when </think> nearly wins; reads logits only.
        "thinkbrake": ReasoningConfig(policy="thinkbrake", threshold=0.1, temperature=0.6),
    }
    requests, params = [], []
    for name, config in policies.items():
        extra_args = (
            VLLMReasoningAdapter.extra_args(config, reasoning_format, request_id=name)
            if config is not None
            else None
        )
        requests.append({"prompt_token_ids": prompt_ids})
        params.append(SamplingParams(temperature=0, max_tokens=16000, extra_args=extra_args))

    # One batch: each request keeps its own controller state inside the processor.
    outputs = llm.generate(requests, params)
    for name, output in zip(policies, outputs):
        completion = output.outputs[0]
        print(
            f"{name:>13}: answer={extract_answer(completion.text)!r} "
            f"generated_tokens={len(completion.token_ids)}"
        )


if __name__ == "__main__":
    main()
