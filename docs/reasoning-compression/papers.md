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
