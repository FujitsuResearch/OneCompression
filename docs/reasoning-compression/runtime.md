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

Forced end-marker sequences and jump text are emitted token by token through
normal decoding, with the cache updated for each token before the next forward. There is no cache
rewind. The maximum generation length can still interrupt a marker or final
answer; `finish_reason="length"` and the token counters retain that fact.

Probes branch from the live request cache. The adapter crops the `DynamicCache`
to the probe's starting point, forwards the policy's paper cue (plus, for
attention probes, the cue-sized run of preceding reasoning tokens, which ASAG's
query window covers), decodes the trial answer, then crops the cache back and
re-appends the saved key/value slices. The main trajectory's cache is
therefore restored bit-for-bit; a test checks tensor equality and identical
trajectories with and without reuse. Probe cost counts every forwarded cue,
re-forwarded reasoning and answer token (`probe_model_evaluated_tokens`); the cached prefix it reuses is
reported as `probe_reused_prefix_tokens`. Caches that cannot be restored from
key/value slices alone (including sliding-window and hybrid layers), models
with length-dependent RoPE, and `probe_prefix_reuse=False` use full-prefix
replay. For length-dependent RoPE, each probe uses frequencies for its own
sequence length and restores the model's rotary state on exit. Replay is the
conservative cost reference; branching matches prefix-cached serving such as
the official vLLM DEER implementation. Probability scoring excludes tokens containing only box
delimiters or whitespace; a token overlapping both a delimiter and answer
content remains included. A truncated or empty box cannot trigger a
confidence/stability stop. After a stop decision, the original reasoning stream
gets its end marker and produces a fresh final answer, following DEER/ASAG's
described action rather than returning the disposable trial answer.

ASAG uses eager attention on the probe forward and reduces only the last four
available layers and the last `2 * cue_length` query rows over all prefix keys.
With cache branching only those query rows are forwarded, so the attention
tensors are `[heads, 2 * cue_length, prefix]` per layer. In replay mode the full
`[heads, prefix, prefix]` matrices are materialized, which does not fit a 24 GB
GPU at Qwen3-8B reasoning lengths. The confidence
ablation skips attention; the attention ablation skips trial-answer decoding.
The probability mixture uses the probe's next-token distribution at the first
answer-content position (the first token inside `\boxed{...}`) for one live
decoding step. The distribution directly after the cue is not used: it predicts
the `{` delimiter almost deterministically and would make the mixture a no-op.
That interpretation is explicit because the paper does not fully specify vector
aggregation or injection duration. Attention probe forward and reduction times
are reported as `attention_latency_s`; this is not the marginal slowdown versus
the same forward with fused attention.

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

Logits-processor policies are listed in `VLLMReasoningAdapter.supported_policies`
(none, fixed, prompt budget and ThinkBrake). A logits hook sees one sampling
step, so use the offline adapter below for trial-answer decoding. The processor is
restricted to vLLM 0.21.x, `n=1`, and no speculative decoding. An end marker may
not also be an EOS/stop token. These are capability guards, not evidence that
untested modes work. Worker-local trace state is not transported through the
stock serving API; serving runs must mark unavailable per-boundary diagnostics
as missing.

The installed `SamplingParams.thinking_token_budget` field has no execution-side
consumer in the inspected Python source. Its presence alone is insufficient to
claim a working native budget implementation, so this adapter forces explicit
model-specific marker tokens.

## Offline vLLM probe adapter

`VLLMProbeReasoningAdapter` generates short greedy chunks with `LLM.generate`
and checks candidate boundaries against the accepted prefix. Sentence and token
boundary checks run before requesting another chunk. Transition checks use the
next candidate token. A trial-answer
request uses that prefix plus the policy cue. It supplies the sampled tokens'
raw log probabilities for DEER's geometric mean or ASAG's arithmetic mean.
vLLM captures these probabilities before applying logits processors. The
offline adapter rejects custom processors registered on the engine or through
the `vllm.logits_processors` entry-point group. OneComp's own processor is
allowed because these requests carry no metadata that activates it.
Only content from a complete, balanced box enters the confidence and stability
checks. The main request continues from its original prefix plus any accepted
intervention. vLLM prefix caching can reuse shared blocks across requests.

ASAG attention modes use an eager Transformers model with matching weights
and tokenizer. For compatible full-attention caches, it keeps a request-local
KV cache of the accepted prefix, extending it in chunks of at most 128 tokens
without retaining attention weights. Each attention probe temporarily branches
that cache and materializes weights only for the cue and the preceding
cue-length window. The adapter releases the eager KV cache when the request
enters the answer phase. Models without a compatible full-attention cache, models
using length-dependent RoPE such as Dynamic RoPE or LongRoPE, and adapters
configured with `probe_prefix_reuse=False` use full-prefix replay and its
higher memory cost. ASAG without attention uses vLLM alone. On a bias decision,
two one-step vLLM requests read
the full distributions at the live reasoning position and the first trial
answer-content position. Stop and jump insert tokens into the resumed prefix.
Sentence checks ignore the inserted jump's last token and resume as soon as
the model emits a new sentence boundary.

`vllm_main_work` and `vllm_probe_work` record request counts, submitted prompt
tokens, reported cached tokens and sampled tokens. Those counters include
lookahead discarded on intervention and samples from distribution queries.
vLLM stops a trial at a closing brace. If it closes an inner group, the adapter
resumes the trial from its existing token prefix within the same probe budget.
It scores only the contents of a complete box. Evaluated-token estimates sum uncached prompt
tokens and all but the last sampled token per request, plus eager attention
forwards. They do not measure vLLM scheduling or hardware work. If the cue will
not fit in the model context, the boundary has no probe evidence.

## What optimized ASAG would require

`vllm/v1/attention/backend.py:AttentionImpl.forward` and the FlashAttention
implementation return attention outputs, not attention probability matrices.
Some split/context-parallel paths request softmax log-normalizers via
`return_softmax_lse=True`; those alone do not determine entropy.

For attention scores `s_i` and probabilities `p_i = exp(s_i) / Z`, exact entropy
is `log(Z) - sum_i p_i s_i`. A fused implementation could accumulate the weighted
score sum alongside online softmax normalization, plus probability sums over
the previous/current thought key spans for the trap signal. It must preserve
causal masking, grouped-query heads, context partition reductions and each
request's query/key span mapping. Selective observations would need the cue query
rows and preceding reasoning queries used by the reference, rather than only
the current decode query.

No validated kernel or probe-scheduling hook is implemented here. Logits hooks
cannot safely fork/replay a request, supply earlier query states, or transport
those internal reductions. An optimized ASAG path therefore remains a separate
GPU engineering and validation task. Its required checks are scalar equivalence
against the eager reference, batching correctness, peak memory and measured
wall-clock overhead; an entropy approximation must retain a separate method name.
