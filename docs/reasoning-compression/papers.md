# Reasoning-compression sources and reproduction contract

A tested equation or CPU implementation is not a reproduction of a published
model-quality or GPU-performance result. See [design](design.md) for the contracts.

## ThinkBrake

Source: [Song et al., arXiv:2510.00546v3](https://arxiv.org/html/2510.00546v3),
§3.1–3.2, §4.1, Appendix C.2; also the
[ACL 2026 publication](https://aclanthology.org/2026.findings-acl.1095.pdf).
At sentence boundaries, compare the best **non-end** continuation with the
reasoning-end token. Stop when their log-probability difference is at most `tau`:

```text
margin = (max(logits excluding end) - logits[end]) / T
stop = margin <= tau
```

Qwen3 default `tau=0.1`; reported sweeps include `0.1, 0.25, 1.0, 2.5`.
Paper Qwen3 decoding uses temperature `0.6`, top-p `0.95`, top-k `20`,
reasoning limit `16384`, answer limit `4096`, seed `42`. The boundary test needs
no additional model forward or attention output.

The paper's `T` is the decoding temperature (Eq. 1 uses the temperature-scaled
distribution). OneComp decodes greedily, so `ReasoningConfig.temperature` is an
explicit *signal* temperature: the default `1.0` uses raw logits, while the
examples set `0.6` to keep the paper's `tau` scale. The two are different method
configurations. A punctuation-based sentence detector is an operational
approximation because the inspected paper does not specify a complete
tokenizer-aware segmenter. Every-N-token checks must be identified as a boundary
ablation. Published sampling results and the greedy reference are separate
experiment settings.

## DEER

Source: [Yang et al., arXiv:2504.15895v3](https://arxiv.org/html/2504.15895v3),
§3.2, §4.1, §4.3, Algorithm 1. The primary transition marker is `Wait`;
`Alternatively` is an ablation. The entropy-monitor variant checks the next-token
entropy after double-newline steps, using threshold `0.672`.
At a transition, append an answer-inducing cue to a temporary branch, greedily
produce an answer, and compute:

```text
confidence = exp(mean(log(max_token_probability_at_each_answer_position)))
stop = confidence > 0.95
```

Equation 4 specifies a **geometric** mean. Rejecting a probe resumes the original
trajectory. Algorithm 1 accepts by inserting the reasoning-end delimiter into
the main prefix and generating the final response. Reusing the trial answer is
a different intervention. Greedy single-sample evaluation is described in
Appendix C; AMC/AIME additionally use repeated runs in the paper. The
branch-parallel acceleration and DEER-Pro are separate extensions.

The [official Qwen3 implementation](https://raw.githubusercontent.com/iie-ycx/DEER/main/vllm-deer-qwen3.py)
uses the cue `\n**Final Answer**\n\\boxed` and provides several confidence
aggregations. Its default geometric calculation skips an initial generated
token and checks a terminal marker. The
[official Transformers implementation](https://raw.githubusercontent.com/iie-ycx/DEER/main/vanilla_deer.py)
instead defaults to arithmetic confidence. These implementation versions do not
define a single interchangeable confidence convention. OneComp follows paper
Equation 4, explicitly records answer-content masking, and treats truncated or
unparseable probes as missing confidence. OneComp's transition detector checks the
greedy next token *before* it is emitted, like the official stop string, but
matches it case-insensitively (`Wait`, `wait`, `Alternatively`), so mid-sentence
"But wait" also triggers a probe; the official vLLM stop string `Wait` is
case-sensitive. Probes branch from the live KV cache (equivalent to the official
implementation's prefix caching). These are reported as a reference
interpretation, not an exact reproduction of the authors' branch-parallel runtime.

## ASAG

Source: [Li et al., arXiv:2606.15070v1](https://arxiv.org/html/2606.15070v1),
Equations 2, 5–8, §4, §5.1 and Algorithm 1.

| Component | Paper definition |
|---|---|
| Confidence | Arithmetic mean of selected trial-answer token probabilities |
| Attention | Last four layers, all heads; normalized Shannon entropy summed across query rows/heads/layers |
| Normalization | Divide entropy by `log(k)`, where `k` is the full key-prefix width |
| Query window | Probe cue plus an equal number of immediately preceding reasoning tokens (§4.1) |
| Convergence | `(H - H_first) / H_first < -0.1` |
| First transition | Confidence `>0.95` permits exit |
| Later transitions | Both confidence `>0.95` and convergence permit exit |
| Low-confidence convergence | Mix current next-token probabilities with probe probabilities at weights `0.95/0.05` |
| Trap | Previous thought's attention per token exceeds current thought's, with entropy change above threshold |
| Escape | Append a jump prompt; eventually force final-answer transition |

Defaults are BF16, greedy decoding, maximum `16000` new tokens, confidence `0.95`,
entropy change `-0.1`, jump threshold `1`. Appendix A's 32-row analysis window
must not be confused with §4.1's cue-sized method window. Attention is observed;
Equation 7 intervenes in output probabilities, not attention weights.

Exact short cue and jump strings from Appendix D.2:

```python
probe_cue = "\n\n Final Answer\n\n \\boxed"
jump_prompt = (
    "Wait, my previous reasoning is not correct. I should adopt a more concise "
    "and different approach to reexamine this problem.\n\n"
)
```

Implementation choices the paper leaves open, and the choice made by
`asag_reference`:

- Which probe-position probability vector is retained and how long injection
  lasts. The paper does not unambiguously define either. OneComp uses the
  next-token distribution at the first answer-content position inside
  `\boxed{}` (Eq. 7's `a_r`), for one live decoding step. The position right
  after the cue is not used because it predicts the `{` delimiter.
- Whether the jump limit follows the stated maximum or Algorithm 1's inclusive
  comparison; these admit different numbers of jumps when the counter starts at
  zero. OneComp allows exactly `max_jumps` (default 1) jumps, then stops.
- Whether a boundary is detected before or after the marker token is committed.
  OneComp probes before the transition token is emitted.
- Whether attention is replayed from a prefix or collected during ordinary
  decoding, and the complete cost of that replay. OneComp branches the live
  cache, re-forwards the cue-sized run of preceding reasoning tokens plus the
  cue with eager attention, and charges all forwarded tokens.
- A zero initial entropy gives an undefined relative change, represented as
  missing data; it is not evidence of convergence.

The implementation is an **ASAG reference interpretation** until its conventions
are cross-validated on a real model and a reported qualitative trend is
reproduced. Last-four-layer reductions on smaller synthetic models report the
actual available layer count. An optimized vLLM approximation needs its own
method identity and evaluation. No fused attention instrumentation is implied by
the existence of this reference.

## Certaindex / Dynasor

Source: [Fu et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/d037fd021c9aace128b8ce25001cdb6c-Paper-Conference.pdf),
§2.1 and Appendix G.1. The CoT variant periodically probes the current answer on a
temporary branch, discards branch tokens before continuing, and measures
agreement with the latest answer over a sliding window. Answers containing
hesitation indicators such as `wait` or `hmm` are excluded. The paper sweeps
checkpoint intervals `32, 64, 128, 256, 320` and the required consecutive-answer
count under a `16K` generation cap. Stability concerns whether an answer will
change, not whether it is correct.

OneComp's `stability` baseline is only a single-trajectory answer-consistency
signal. A three-answer window is an explicit initial configuration, not a
universal paper optimum. It does not reproduce Dynasor's scheduling, multi-path
semantic-entropy metric, or serving-throughput claims. It probes with DEER's cue
rather than Dynasor's own prompt, which is a deviation from the paper. Invalid
probes must not create agreement.
