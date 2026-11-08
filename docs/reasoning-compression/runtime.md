# Runtime contract and vLLM integration

Inspected against Transformers 5.14.1, torch 2.11.0+cu130 and vLLM 0.21.0. CPU
integration tests exercise actual tiny Qwen3 forwards and the installed vLLM
batch-update types; they do not establish GPU serving performance.

## Transformers reference

`TransformersReasoningAdapter` accepts an already-loaded model and tokenizer,
including a model returned by OneComp's `QuantizedModelLoader`. It uses one
greedy trajectory, an actual KV cache, and batch size one. Requests own their
state and caches. Parallel forwards through the same model instance are not
promised; use separate inference workers. The callback emits each visible token
once, including forced reasoning-end markers. Its text is decoded incrementally:
a token that ends inside a multi-byte character (byte-level BPE) gets an empty
string, and the next token carries the completed character, so the streamed
texts concatenate to the decoded output.

The `None` controller and `policy="none"` paths produce identical tokens; a CPU
test also compares the adapter's output with `Qwen3ForCausalLM.generate` on the
same randomly initialized tiny model. No distribution or global random seed is
changed by the greedy adapter. Model-specific generation processors such as
repetition penalties and constrained decoding are outside this reference path.

At each boundary, a policy that uses logits receives `logit_observation` of the
logits the last forward already returned, so ThinkBrake adds no model forward.
Policies that require the margin reject multi-token end markers.

Forced end-marker sequences are emitted token by token through normal decoding,
with the cache updated for each token before the next forward. There is no cache
rewind. The maximum generation length can still interrupt a marker or final
answer; `finish_reason="length"` and the token counters retain that fact.

Timings use `perf_counter` and CUDA synchronization at measurement boundaries
when a CUDA device exists. Controller and intervention times have their own
fields. Synchronization overhead and KV bytes remain null when not measured; no
value is inferred from generated-token count. Peak GPU allocation is
process-local and only meaningful for this isolated serial measurement.

## vLLM 0.21 adapter

The adapter targets the inspected v1 interfaces in:

- `vllm/v1/sample/logits_processor/interface.py`: `LogitsProcessor`,
  `BatchUpdate`, and `MoveDirectionality`;
- `vllm/sampling_params.py`: per-request `extra_args`;
- `vllm/entrypoints/llm.py`: `logits_processors` engine argument;
- `vllm/tokenizers/registry.py`: `cached_tokenizer_from_config`.

Register `onecomp.runtime.vllm.logits_processor:ReasoningLogitsProcessor` in the
engine. `VLLMReasoningAdapter.extra_args(config, format, request_id)` creates the
request metadata for `SamplingParams`. Disabled requests allocate no policy
state and preserve the logits tensor exactly. Active state follows request
removals, additions, unidirectional moves and swaps; tests cover slot reuse and
different stopping times in one batch. Marker tokens are forced with the usual
sampler and remain in the normal output stream/KV history.

With async scheduling, which vLLM 0.21 enables by default, `update_state` can
see the newest output token as a `-1` placeholder that vLLM fills in just before
sampling. The processor consumes tokens only up to the first placeholder and
resynchronizes in `apply`; placeholders are never counted or decoded.

ThinkBrake reads the margin from the request's own logits row. For sentence
boundaries the processor decodes each sampled token with vLLM's cached
tokenizer; for transition boundaries it decodes the argmax candidate. Budget
policies skip decoding. Linguistic boundaries decode token fragments; they are
approximations, not a universal sentence parser.

Supported policies are listed in `VLLMReasoningAdapter.supported_policies`
(none, fixed, prompt budget and ThinkBrake); others fail explicitly. The processor is
restricted to vLLM 0.21.x, `n=1`, and no speculative decoding. An end marker may
not also be an EOS/stop token. These are capability guards, not evidence that
untested modes work. Worker-local trace state is not transported through the
stock serving API; serving runs must mark unavailable per-boundary diagnostics
as missing.

The installed `SamplingParams.thinking_token_budget` field has no execution-side
consumer in the inspected Python source. Its presence alone is insufficient to
claim a working native budget implementation, so this adapter forces explicit
model-specific marker tokens.
