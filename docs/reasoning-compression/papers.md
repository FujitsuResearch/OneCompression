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
