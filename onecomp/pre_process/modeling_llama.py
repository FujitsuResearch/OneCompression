"""Custom Llama model for rotation training.

Extends the standard transformers ``LlamaForCausalLM`` to pass rotation
matrix ``R1`` through the forward pass so that ``QuantEmbedding``,
``QuantLinear`` (lm_head) and ``QuantLlamaDecoderLayer`` can apply rotation
during training.

Only ``LlamaForCausalLM`` and ``LlamaModel`` are overridden; all other
classes (``LlamaAttention``, ``LlamaMLP``, ``LlamaDecoderLayer``, etc.)
are re-exported from the installed transformers package.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami
"""

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
    apply_rotary_pos_emb,
)
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as _BaseLlamaForCausalLM,
    LlamaModel as _BaseLlamaModel,
)
from transformers.utils import auto_docstring, can_return_tuple


class LlamaModel(_BaseLlamaModel):
    """LlamaModel that passes ``R1`` to ``embed_tokens`` and decoder layers."""

    def forward(self, input_ids=None, R1=None, **kwargs):
        """Embed with ``R1`` rotation when given, then delegate to the base forward.

        When ``R1`` is provided, embeddings are computed via
        ``QuantEmbedding(input_ids, R1=R1)`` and passed as ``inputs_embeds``.
        ``R1`` is also forwarded through ``**kwargs`` so that decoder layers
        (``QuantLlamaDecoderLayer``) can receive it.
        """
        if input_ids is not None and R1 is not None:
            kwargs["inputs_embeds"] = self.embed_tokens(input_ids, R1=R1)
            input_ids = None
        return super().forward(input_ids=input_ids, R1=R1, **kwargs)


@auto_docstring
class LlamaForCausalLM(_BaseLlamaForCausalLM):
    """LlamaForCausalLM that holds ``R1`` and passes it through the model."""

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.R1 = None
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(self, input_ids=None, labels=None, **kwargs):
        R1_weight = self.R1.weight if self.R1 is not None else None

        outputs = self.model(
            input_ids=input_ids,
            R1=R1_weight,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits_to_keep = kwargs.pop("logits_to_keep", 0)
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :], R1=R1_weight)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
