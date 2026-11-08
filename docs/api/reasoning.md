# Reasoning Compression

Runtime reasoning-length control (experimental). Usage and policies are
described in the [Reasoning Compression user guide](../user-guide/reasoning-compression.md).

## Configuration and request state

::: onecomp.reasoning.ReasoningConfig

::: onecomp.reasoning.ReasoningFormat

::: onecomp.reasoning.RequestState

::: onecomp.reasoning.ReasoningCompressionController

::: onecomp.reasoning.estimate_prompt_budget

## Policies

::: onecomp.reasoning.policies
    options:
      members:
        - ReasoningPolicy
        - make_policy
        - NonePolicy
        - BudgetPolicy
        - ThinkBrakePolicy

## Signals

::: onecomp.reasoning.ReasoningSignal

::: onecomp.reasoning.signals.logit_observation

## Backends

::: onecomp.runtime.transformers.TransformersReasoningAdapter

::: onecomp.runtime.vllm.VLLMReasoningAdapter
