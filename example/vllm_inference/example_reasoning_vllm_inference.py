"""

Example: Runtime reasoning compression in vLLM

By default runs a vLLM 0.21.x batch with the per-request logits processor.
Pass --policy deer, stability or asag_reference to run a single request with
answer probes. ASAG's attention modes load a matching eager Transformers
model at --attention-device (CPU by default). Pass --asag-mode confidence to
run ASAG without loading that model.

Copyright 2025-2026 Fujitsu Ltd.

"""

from argparse import ArgumentParser
from dataclasses import replace

from vllm import LLM, SamplingParams

from onecomp.reasoning import ReasoningConfig, ReasoningFormat, estimate_prompt_budget
from onecomp.reasoning.data import extract_answer
from onecomp.runtime.vllm import VLLMProbeReasoningAdapter, VLLMReasoningAdapter

MODEL_ID = "Qwen/Qwen3-8B"
QUESTION = (
    "Solve the following problem. Put only your final answer inside \\boxed{}.\n\n"
    "What is the smallest positive integer n such that 3n is a perfect square "
    "and 2n is a perfect cube?"
)


def run_batch(llm, prompt_ids, reasoning_format, max_new_tokens):
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
        params.append(
            SamplingParams(temperature=0, max_tokens=max_new_tokens, extra_args=extra_args)
        )

    # One batch: each request keeps its own controller state inside the processor.
    outputs = llm.generate(requests, params)
    for name, output in zip(policies, outputs):
        completion = output.outputs[0]
        print(
            f"{name:>13}: answer={extract_answer(completion.text)!r} "
            f"generated_tokens={len(completion.token_ids)}"
        )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--policy", choices=sorted(VLLMProbeReasoningAdapter.supported_policies))
    parser.add_argument(
        "--asag-mode", choices=["full", "confidence", "attention", "both"], default="full"
    )
    parser.add_argument("--attention-device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    args = parser.parse_args()

    engine_options = {}
    if args.policy is None:
        engine_options["logits_processors"] = [VLLMReasoningAdapter.processor_path]
    elif args.policy == "asag_reference" and args.asag_mode == "full":
        engine_options["max_logprobs"] = -1
    llm = LLM(model=args.model, max_model_len=20000, **engine_options)
    tokenizer = llm.get_tokenizer()
    reasoning_format = ReasoningFormat.from_tokenizer(tokenizer)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=False,
    )
    if args.policy is None:
        run_batch(llm, prompt_ids, reasoning_format, args.max_new_tokens)
        return

    attention_adapter = None
    if args.policy == "asag_reference" and args.asag_mode != "confidence":
        from transformers import AutoModelForCausalLM

        from onecomp.runtime.transformers import TransformersReasoningAdapter

        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype="auto", attn_implementation="eager", device_map=args.attention_device
        )
        attention_adapter = TransformersReasoningAdapter(model, tokenizer)
    adapter = VLLMProbeReasoningAdapter(llm, attention_adapter=attention_adapter)
    result = adapter.generate(
        prompt_ids,
        reasoning_format=reasoning_format,
        config=ReasoningConfig(policy=args.policy, asag_mode=args.asag_mode),
        max_new_tokens=args.max_new_tokens,
    )
    print(
        f"{args.policy}: answer={extract_answer(result.final_answer)!r} "
        f"reasoning_tokens={result.state.reasoning_tokens} "
        f"probe_tokens={result.state.probe_tokens} "
        f"latency_s={result.metrics['latency_s']:.2f}"
    )


if __name__ == "__main__":
    main()
