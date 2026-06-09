# Evaluation

Hydra-driven evaluation harness for models served through vLLM (`onecomp-eval` CLI).

For installation, MT-Bench data download, judge API keys, and CLI examples, see the [Evaluation user guide](../user-guide/evaluation.md).

!!! note "`ModelConfig` here is eval-specific"
    `onecomp.eval.ModelConfig` only holds the model path/name for evaluation.
    It is not the same as `onecomp.ModelConfig` used for quantization.

## Package exports

::: onecomp.eval
    options:
      show_source: false
      members:
        - EvalConfig
        - EvalsConfig
        - ModelConfig
        - MtBenchConfig
        - ThroughputConfig
        - InferenceConfig
        - VllmServerConfig
        - SummaryConfig
        - TaskResult
        - VllmServerManager
        - run_pipeline
        - aggregate_results
        - run_subprocess_eval

## Configuration schema

::: onecomp.eval.schema.EvalConfig
    options:
      show_source: false

::: onecomp.eval.schema.InferenceConfig
    options:
      show_source: false

::: onecomp.eval.schema.MtBenchConfig
    options:
      show_source: false

::: onecomp.eval.schema.ThroughputConfig
    options:
      show_source: false

::: onecomp.eval.schema.EvalsConfig
    options:
      show_source: false

::: onecomp.eval.schema.SummaryConfig
    options:
      show_source: false

## TaskResult

Per-evaluator subprocess output written to `<output_dir>/<eval_name>/result.json`.

::: onecomp.eval.schema.TaskResult
    options:
      show_source: false
      members:
        - create
        - save
        - load

## Orchestration

::: onecomp.eval.orchestrator.runner.run_pipeline
    options:
      show_source: false

::: onecomp.eval.orchestrator.server.VllmServerManager
    options:
      show_source: false

::: onecomp.eval.orchestrator.subprocess_runner.run_subprocess_eval
    options:
      show_source: false

::: onecomp.eval.orchestrator.aggregator.aggregate_results
    options:
      show_source: false
