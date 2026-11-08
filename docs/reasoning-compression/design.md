# Runtime reasoning compression design

## Existing architecture and extension point

`ModelConfig` loads Transformers models/tokenizers; `QuantizedModelLoader` reconstructs
OneComp packed GPTQ/DBF/OneBit checkpoints. `Runner` performs calibration and weight
compression. The `onecomp` argparse CLI is a quantization command. Existing
`onecomp-eval` uses Hydra, a vLLM HTTP server, and subprocess MT-Bench/throughput
evaluators. Neither evaluator exposes per-step model state. Quantization plugins
register through `vllm.general_plugins`. Logging uses standard Python logging.

Add independent `onecomp.reasoning` and `onecomp.runtime` packages. Preserve
existing quantization, evaluation, and default inference paths. A runtime
receives an already-loaded model, allowing the same controller to operate on
original and OneComp-compressed weights.

## Package layout

```text
onecomp/reasoning/
    config.py          ReasoningConfig (validated, frozen) and the prompt-budget heuristic
    state.py           ReasoningFormat (marker token IDs) and per-request RequestState
    controller.py      ReasoningCompressionController: boundary -> observation -> decision
    boundaries.py      sentence / transition / token-interval detectors
    actions.py         Decision and the stop action
    signals/           ReasoningSignal protocol, output-logit margin, probe-answer confidence
    policies/          one module per method; POLICIES registry and make_policy()
    data.py            boxed-answer extraction and conservative grading
onecomp/runtime/
    transformers/      serial greedy reference backend
    vllm/              vLLM 0.21 logits processor
```

Each policy class declares what a backend must observe (`uses_logits`,
`requires_logits`, `uses_probe`, `requires_attention`, `probe_options()`,
`needs_probe()`, `needs_attention_followup()`); backends query these instead of
matching policy names. The budget baselines need no observation
(`uses_logits=False`); ThinkBrake sets `requires_logits`, so backends pass it
the margin computed from logits they already produced. DEER and stability set
`uses_probe`, so a backend that can branch a request runs an answer probe at
each boundary; backends without that capability reject them. Further adaptive
policies are added by registering a new class in `POLICIES` without changing
this interface. Package `__init__.py` files only re-export.

## Contracts

Separate model-specific reasoning formats, boundary detection, observed signals,
decision policies, and applied actions. A controller owns immutable configuration;
each request owns its counters, marker parser, candidate boundaries, signals,
interventions, and timing. Backends translate decisions into forced marker
tokens. A disabled policy is an identity intervention. Fixed budgets do not wait
for linguistic boundaries.

Use explicit token sequences for start/end markers, not substring assumptions.
Count reasoning, answer, and control-marker tokens separately. A forced close
continues final-answer decoding. Every forced token is evaluated through the
normal KV-cache path and included in model-token accounting.

Baseline order: none, fixed, heuristic prompt budget, then adaptive policies.
Oracle correctness labels are offline measurements and are never supplied to a
deployed controller; the controller rejects them as observations.

## Runtime dependencies

The project allows vLLM >=0.10,<0.22; the Linux Python 3.13 lock resolution and
installed runtime are 0.21.0. Installed Transformers is 5.14.1, torch 2.11.0+cu130.
Inspect the installed source, not an assumed historic per-request logits hook.
vLLM v1 uses `LogitsProcessor.apply`, `update_state(BatchUpdate)`, and
`is_argmax_invariant`; request slots can be removed, added and moved. Adapt this
interface explicitly, including slot swaps. A declared sampling parameter alone
does not establish working native thinking-budget support.

The Transformers reference uses real model forwards and wall-clock timing. It is
an algorithmic reference, not a claim of production serving performance.
Unsupported policies in a backend must fail explicitly.

## Limitations

CPU synthetic/unit experiments validate mechanics and accounting but cannot answer
model-quality or GPU speedup questions; those come only from real-model runs. No
simulated score is to be labeled a Qwen3 or published-benchmark result.
