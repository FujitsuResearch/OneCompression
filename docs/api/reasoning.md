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
        - DeerPolicy
        - StabilityPolicy
        - AsagPolicy

## Signals

::: onecomp.reasoning.ReasoningSignal

::: onecomp.reasoning.signals.logit_observation

::: onecomp.reasoning.signals.deer_confidence

::: onecomp.reasoning.signals.asag_confidence

::: onecomp.reasoning.signals.attention_statistics

::: onecomp.reasoning.signals.relative_entropy_change

## Backends

::: onecomp.runtime.transformers.TransformersReasoningAdapter

::: onecomp.runtime.vllm.VLLMReasoningAdapter
