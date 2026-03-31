"""Quantization-aware model wrappers for rotation training.

Provides QuantLinear, QuantRMSNorm, QuantEmbedding, QuantLlama/Qwen3 decoder
layers, WeightQuantizer, RotateModule, and ScalingModule.  Used exclusively
during the training phase of ``prepare_rotated_model()``.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami
"""

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .hadamard_utils import get_hadK, matmul_hadU_cuda
from .modeling_llama import (
    LlamaConfig,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    eager_attention_forward as llama_eager_attention_forward,
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
)
from .modeling_qwen3 import (
    Qwen3Config,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
    eager_attention_forward as qwen3_eager_attention_forward,
    apply_rotary_pos_emb as qwen3_apply_rotary_pos_emb,
)


def _get_attention_fn(impl_name, eager_fallback):
    """Resolve the attention implementation callable for the current transformers version.

    Args:
        impl_name: Name of the attention backend (e.g. ``"eager"``, ``"sdpa"``,
            ``"flash_attention_2"``) as stored on ``config._attn_implementation``.
        eager_fallback: Fallback forward function used when ``impl_name`` is
            ``"eager"`` or when the requested implementation is unavailable.

    Returns:
        A callable with the same contract as ``eager_fallback`` for the active
        transformers major version (4.x dict lookup vs 5.x ``get_interface``).
    """
    if impl_name == "eager":
        return eager_fallback
    if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
        # transformers >= 5.x
        fn = ALL_ATTENTION_FUNCTIONS.get_interface(impl_name, eager_fallback)
        return fn if fn is not None else eager_fallback
    # transformers 4.x
    return ALL_ATTENTION_FUNCTIONS.get(impl_name, eager_fallback)


# ============================================================
# STE Quantizers
# ============================================================


class STEQuantize(torch.autograd.Function):
    """Straight-through estimator (STE) for symmetric fixed-point quantization.

    Rounds activations or weights to a symmetric integer grid
    ``[-(maxq+1), maxq]`` with step ``scale``; gradients pass through
    unmodified (STE).
    """

    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    """Straight-through estimator for asymmetric fixed-point quantization.

    Maps values to integer codes in ``[0, maxq]`` via ``round(x/scale)+zero``,
    then maps back to the dequantized range; gradients pass through (STE).
    """

    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


# ============================================================
# WeightQuantizer (RTN)
# ============================================================


class WeightQuantizer(nn.Module):
    """RTN-style weight quantizer with learnable-compatible STE forward.

    Holds per-tensor or per-group ``scale`` / ``zero`` buffers, populated by
    :meth:`find_params`, and applies :class:`STEQuantize` or
    :class:`AsymSTEQuantize` in :meth:`quantize` when enabled and ready.
    """

    def __init__(self, shape: int = 1):
        super().__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(self, bits, perchannel=True, sym=True, weight_groupsize=-1):
        """Set quantization mode and bit width.

        Args:
            bits: Target bit width (e.g. 4, 8). ``16`` disables integer
                quantization in :meth:`quantize`.
            perchannel: If True, compute one scale/zero per output channel
                (row). If False, use a single scale/zero for the entire
                tensor (per-tensor). Ignored when ``weight_groupsize > 0``.
            sym: If True, symmetric range ``[-(maxq+1), maxq]``; else
                asymmetric ``[0, maxq]``.
            weight_groupsize: If positive, ``find_params`` groups the last
                dimension into chunks of this size (overrides *perchannel*).
        """
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.weight_groupsize = weight_groupsize
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        """Compute ``scale`` and ``zero`` from tensor statistics (no-op if ``bits==16``).

        Args:
            x: Weight tensor whose min/max statistics define quantization
                parameters.

        The granularity is determined by ``weight_groupsize`` and
        ``perchannel`` (set via :meth:`configure`):

        * ``weight_groupsize > 0`` – group-wise (overrides *perchannel*).
        * ``perchannel=True``  – one scale/zero per output channel (row).
        * ``perchannel=False`` – one scale/zero for the whole tensor.
        """
        if self.bits == 16:
            return
        shape = x.shape
        if self.weight_groupsize > 0:
            x = x.reshape(-1, x.shape[-1] // self.weight_groupsize, self.weight_groupsize)
        elif self.perchannel:
            x = x.reshape(-1, 1, x.shape[-1])
        else:
            x = x.flatten().reshape(1, 1, -1)

        if self.sym:
            q_max, q_min = self.maxq, -(self.maxq + 1)
        else:
            q_max, q_min = self.maxq, 0

        x_max = x.amax(dim=-1, keepdim=True)
        x_min = x.amin(dim=-1, keepdim=True)
        self.scale = ((x_max - x_min) / (q_max - q_min)).clamp(min=1e-5)

        if self.sym:
            self.zero = torch.zeros_like(self.scale)
        else:
            self.zero = torch.round(q_min - x_min / self.scale).clamp(q_min, q_max)

        if self.weight_groupsize > 0:
            pass
        elif self.perchannel:
            s = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(s)
            self.zero = self.zero.reshape(s)
        else:
            self.scale = self.scale.reshape([1] * len(shape))
            self.zero = self.zero.reshape([1] * len(shape))

    def quantize(self, x):
        """Apply STE quantization when configured and buffers are initialized.

        Args:
            x: Tensor to quantize (typically linear weights).

        Returns:
            Quantized tensor in the same floating dtype as ``x``, or ``x``
            unchanged when quantization is off (``bits >= 16``) or not ready.
        """
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.weight_groupsize > 0:
                orig_shape = x.shape
                x = x.reshape(-1, x.shape[-1] // self.weight_groupsize, self.weight_groupsize)
                if self.sym:
                    x = STEQuantize.apply(x, self.scale, self.maxq)
                else:
                    x = AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq)
                return x.reshape(orig_shape).to(x_dtype)
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        """Whether integer quantization is configured (non-zero ``maxq``).

        Returns:
            ``True`` if ``self.maxq`` is strictly positive.
        """
        return self.maxq > 0

    def ready(self):
        """Whether ``scale`` has been set by :meth:`find_params`.

        Returns:
            ``True`` if every element of ``self.scale`` is non-zero.
        """
        return torch.all(self.scale != 0)


# ============================================================
# RotateModule / ScalingModule
# ============================================================


class RotateModule(nn.Module):
    """Learnable orthogonal-style rotation applied as a fixed matrix multiply.

    Args:
        R_init: Initial rotation matrix; stored as trainable ``weight`` in
            ``float32`` regardless of activation dtype.
    """

    def __init__(self, R_init):
        super().__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32))

    def forward(self, x, transpose=False):
        """Apply ``R @ x`` or ``x @ R`` depending on operand layout.

        Args:
            x: Input tensor.
            transpose: If False, ``self.weight @ x``; if True, ``x @ self.weight``.

        Returns:
            Rotated tensor, dtype promoted as usual for matmul.
        """
        return x @ self.weight if transpose else self.weight @ x


class ScalingModule(nn.Module):
    """Per-channel scaling with optional reciprocal (inverse) application.

    Args:
        S_init: Initial scaling factors (may include negative values depending
            on ``scaling_mode``); stored as trainable ``weight`` in ``float32``.
    """

    def __init__(self, S_init):
        super().__init__()
        self.weight = nn.Parameter(S_init.to(torch.float32))

    def forward(self, x, inverse=False):
        """Scale ``x`` by ``weight`` or divide by it.

        Args:
            x: Input tensor, broadcast-compatible with ``self.weight``.
            inverse: If True, returns ``x / self.weight``; else ``x * self.weight``.

        Returns:
            Scaled tensor.
        """
        return x / self.weight if inverse else x * self.weight


# ============================================================
# QuantRMSNorm / QuantEmbedding
# ============================================================


class QuantRMSNorm(nn.Module):
    """Frozen RMSNorm that optionally absorbs an extra scaling factor ``S``.

    Args:
        norm: Source ``RMSNorm`` (or compatible) module; ``weight`` is frozen;
            ``eps`` is taken from ``variance_epsilon``.
    """

    def __init__(self, norm):
        super().__init__()
        self.eps = norm.variance_epsilon
        self.weight = norm.weight.requires_grad_(False)

    def forward(self, hidden_states, S=None):
        """RMS normalize and scale by ``weight``, optionally divided by ``S``.

        Args:
            hidden_states: Last dimension matches the norm ``weight`` length.
            S: Optional tensor broadcastable to ``weight``; when set,
                effective scale is ``weight / S``.

        Returns:
            Normalized activations in the input dtype of ``hidden_states``.
        """
        i_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        weight = self.weight
        if S is not None:
            weight = weight / S
        return (weight * hidden_states).to(i_dtype)


class QuantEmbedding(nn.Embedding):
    """Frozen token embedding with optional learned rotation or fixed Hadamard mix.

    Args:
        embedding: Source ``nn.Embedding``; weights are copied into a non-trainable
            buffer (no gradients through embedding lookup).
    """

    def __init__(self, embedding: nn.Embedding):
        super().__init__(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            padding_idx=embedding.padding_idx,
            max_norm=embedding.max_norm,
            norm_type=embedding.norm_type,
            scale_grad_by_freq=embedding.scale_grad_by_freq,
            sparse=embedding.sparse,
            _weight=embedding.weight,
            _freeze=True,
            device=embedding.weight.device,
            dtype=embedding.weight.dtype,
        )
        del self.weight
        self.register_buffer("weight", embedding.weight.data)

    def forward(self, input: torch.Tensor, R1=None):
        """Lookup embeddings, then apply ``R1`` rotation or a fixed Hadamard transform.

        Args:
            input: Long tensor of token indices, same as ``nn.Embedding.forward``.
            R1: Optional rotation matrix applied on the right to embedding vectors
                in float64 for numerical stability. If ``None``, applies a
                normalised Hadamard transform via ``matmul_hadU_cuda``.

        Returns:
            Tensor of shape ``(*input.shape, embedding_dim)`` after the optional
            rotation / Hadamard step, cast back to the embedding dtype.
        """
        out = nn.functional.embedding(
            input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if R1 is not None:
            ori_dtype = out.dtype
            out = (out.to(torch.float64) @ R1.to(dtype=torch.float64, device=out.device)).to(
                ori_dtype
            )
        else:
            dim = out.shape[-1]
            ori_dtype = out.dtype
            had_K, K = get_hadK(dim)
            out = matmul_hadU_cuda(out.to(torch.float64), had_K, K).to(ori_dtype)
        return out


# ============================================================
# QuantLinear
# ============================================================


class QuantLinear(nn.Linear):
    """Linear layer with frozen weights and rotation / scaling hooks for QAT.

    Applies optional per-role scaling (``S_*``), left/right rotations (``R1``,
    ``R2``), fixed Hadamard substitutes when rotations are omitted, optional
    online Hadamard on ``down``, and attaches an optional ``quantizer`` for RTN.

    Args:
        module: Source ``nn.Linear``; ``weight`` and ``bias`` are frozen.
        name: Role tag (``"q"``, ``"k"``, ``"v"``, ``"o"``, ``"up"``, ``"gate"``,
            ``"down"``, ``"head"``) controlling which transforms run.
        attn_instance: Attention module providing head layout metadata
            (``num_heads``, ``num_key_value_heads``, ``head_dim``, etc.) when
            ``name`` refers to attention projections.
    """

    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, module: nn.Linear, name=None, attn_instance=None):
        nn.Module.__init__(self)
        self.weight = module.weight.requires_grad_(False)
        self.bias = module.bias.requires_grad_(False) if module.bias is not None else None
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.name = name
        self.num_key_value_heads = getattr(attn_instance, "num_key_value_heads", None)
        self.num_key_value_groups = getattr(attn_instance, "num_key_value_groups", None)
        self.num_heads = getattr(attn_instance, "num_heads", None)
        self.head_dim = getattr(attn_instance, "head_dim", None)

    def forward(
        self, x, R1=None, R2=None, S_up_down=None, S_qk=None, S_ov=None, S_attn=None, S_mlp=None
    ):
        """Compute ``linear(x)`` after optional scaling, rotation, and quantization.

        Args:
            x: Input activations for ``F.linear``.
            R1: Optional first rotation; interpretation depends on ``self.name``
                (multiply on input or output side of the weight matrix).
            R2: Optional second rotation for head-block layout (``"v"`` / ``"o"``).
            S_up_down: MLP up/down scaling factors when ``name`` is ``"up"`` or
                ``"down"``.
            S_qk: Per-head scaling for query/key when ``name`` is ``"q"`` or ``"k"``.
            S_ov: Scaling for value/output when ``name`` is ``"v"`` or ``"o"``.
            S_attn: Broadcast scale applied to ``q``/``k``/``v`` weights after
                rotations.
            S_mlp: Broadcast scale for ``"up"`` / ``"gate"`` weights.

        Returns:
            Output of ``F.linear`` with the same batch/sequence layout as ``x``.
        """
        ori_dtype = self.weight.dtype
        dtype = torch.float64
        weight = self.weight.to(dtype)
        bias = self.bias.to(dtype) if self.bias is not None else None

        # --- Scaling ---
        if S_up_down is not None:
            if self.name == "up":
                weight = weight / S_up_down.to(dtype).view(-1, 1)
                if bias is not None:
                    bias = bias / S_up_down.to(dtype).view(-1)
            elif self.name == "down":
                weight = weight * S_up_down.to(dtype).view(1, -1)

        if S_qk is not None:
            half = self.head_dim // 2
            S_half = S_qk.view(self.num_key_value_heads, half)
            if self.name == "k":
                S = torch.cat([S_half, S_half], dim=-1).reshape(-1)
                weight = weight * S.to(dtype).view(-1, 1)
                if bias is not None:
                    bias = bias * S.to(dtype).view(-1)
            elif self.name == "q":
                S = torch.cat([S_half, S_half], dim=-1)
                S = (
                    S[:, None, :]
                    .expand(self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
                    .reshape(-1)
                )
                weight = weight / S.to(dtype).view(-1, 1)
                if bias is not None:
                    bias = bias / S.to(dtype).view(-1)

        if S_ov is not None:
            if self.name == "v":
                weight = weight / S_ov.to(dtype).view(-1, 1)
                if bias is not None:
                    bias = bias / S_ov.to(dtype).view(-1)
            elif self.name == "o":
                S = S_ov
                if S.numel() < self.in_features:
                    S = S.reshape(self.num_key_value_heads, -1)
                    n_head, d = S.shape
                    S = S[:, None, :].expand(n_head, self.num_key_value_groups, d).reshape(-1)
                weight = weight * S.to(dtype).view(1, -1)

        # --- Rotation R1 ---
        if R1 is not None:
            if self.name in ("q", "k", "v", "up", "gate", "head"):
                weight = weight @ R1.to(dtype)
            elif self.name in ("down", "o"):
                weight = R1.T.to(dtype) @ weight
                if bias is not None:
                    bias = R1.T.to(dtype) @ bias
        else:
            if self.name in ("q", "k", "v", "up", "gate", "head"):
                had_K, K = get_hadK(self.in_features)
                weight = matmul_hadU_cuda(weight, had_K, K)
            elif self.name in ("down", "o"):
                had_K, K = get_hadK(self.out_features)
                weight = matmul_hadU_cuda(weight.t(), had_K, K).t()
                if bias is not None:
                    bias = matmul_hadU_cuda(bias.unsqueeze(0), had_K, K).squeeze(0)

        # --- Rotation R2 ---
        if R2 is not None:
            had_dim = R2.shape[0]
            if self.name == "v":
                W_ = weight.t()
                s = W_.shape
                weight = (W_.reshape(-1, s[-1] // had_dim, had_dim) @ R2.to(dtype)).reshape(s).t()
                if bias is not None:
                    bias = (bias.reshape(-1, had_dim) @ R2.to(dtype)).reshape(-1)
            elif self.name == "o":
                s = weight.shape
                weight = (weight.reshape(-1, s[-1] // had_dim, had_dim) @ R2.to(dtype)).reshape(s)
        else:
            if self.name in ("v", "o"):
                had_dim = self.head_dim
                had_K, K = get_hadK(had_dim)
                if self.name == "v":
                    W_ = weight.t()
                    s = W_.shape
                    weight = (
                        matmul_hadU_cuda(W_.reshape(-1, s[-1] // had_dim, had_dim), had_K, K)
                        .reshape(s)
                        .t()
                    )
                    if bias is not None:
                        bias = matmul_hadU_cuda(bias.reshape(-1, had_dim), had_K, K).reshape(-1)
                else:
                    s = weight.shape
                    weight = matmul_hadU_cuda(
                        weight.reshape(-1, s[-1] // had_dim, had_dim), had_K, K
                    ).reshape(s)

        # --- Post-rotation scaling ---
        if S_attn is not None and self.name in ("q", "k", "v"):
            weight = weight * S_attn.to(dtype).view(1, -1)
        if S_mlp is not None and self.name in ("up", "gate"):
            weight = weight * S_mlp.to(dtype).view(1, -1)

        # --- Online Hadamard for down_proj ---
        if self.name == "down":
            had_K, K = get_hadK(self.in_features)
            weight = matmul_hadU_cuda(weight, had_K, K)

        weight = weight.to(ori_dtype)
        if bias is not None:
            bias = bias.to(ori_dtype)
        else:
            bias = self.bias

        if hasattr(self, "quantizer"):
            self.quantizer.find_params(weight.data)
            weight = self.quantizer.quantize(weight).to(weight.dtype)

        return nn.functional.linear(x, weight, bias)


# ============================================================
# Quant Decoder Layers (Llama & Qwen3)
# ============================================================


class QuantLlamaDecoderLayer(GradientCheckpointingLayer):
    """Llama decoder block with quantized projections and RMSNorm wrappers.

    Composes :class:`_QuantLlamaAttention`, :class:`_QuantMLP`, and
    :class:`QuantRMSNorm`; ``R_S_modules`` holds optional learned ``RotateModule`` /
    ``ScalingModule`` entries merged into ``R_S_factors`` at forward time.

    All layer-level metadata is derived from ``config`` and ``layer_idx`` so
    that the wrapper is independent of attributes that may appear or disappear
    across transformers versions.

    Args:
        config: ``LlamaConfig`` (or compatible) with ``hidden_size`` and related
            fields.
        layer_idx: Position of this layer in the decoder stack.
        layer: Source ``LlamaDecoderLayer`` whose submodules are wrapped.
    """

    def __init__(self, config, layer_idx, layer):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _QuantLlamaAttention(config, layer_idx, layer.self_attn)
        self.mlp = _QuantMLP(config, layer.mlp)
        self.input_layernorm = QuantRMSNorm(layer.input_layernorm)
        self.post_attention_layernorm = QuantRMSNorm(layer.post_attention_layernorm)
        self.R_S_modules = nn.ModuleDict(dict())

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        R1=None,
        **kwargs,
    ):
        """Pre-norm residual block: self-attn, then MLP (:class:`_QuantMLP`).

        Args:
            hidden_states: ``(batch, seq, hidden)`` tensor.
            attention_mask: Passed through to the attention implementation.
            position_ids: Optional positions; forwarded when supported by attention.
            past_key_value: Optional KV cache object.
            use_cache: Whether to populate ``past_key_value``.
            cache_position: Cache index tensor for incremental decoding.
            position_embeddings: ``(cos, sin)`` tuple for rotary embeddings.
            R1: Optional global rotation matrix merged into ``R_S_factors`` as
                ``R1`` for child modules.
            **kwargs: Extra arguments forwarded to the attention backend (e.g.
                flash-attention flags).

        Returns:
            Hidden states after the decoder block, shape matching ``hidden_states``.
        """
        R_S_factors = dict(R1=R1)
        R_S_factors.update({k: v.weight for k, v in self.R_S_modules.items() if v is not None})
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, S=R_S_factors.get("S_attn"))
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            R_S_factors=R_S_factors,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, S=R_S_factors.get("S_mlp"))
        hidden_states = self.mlp(hidden_states, R_S_factors=R_S_factors)
        hidden_states = residual + hidden_states
        return hidden_states


class _QuantLlamaAttention(nn.Module):
    """Llama self-attention using :class:`QuantLinear` projections and RoPE.

    All layer-level metadata is derived from ``config`` and ``layer_idx`` so
    that the wrapper is independent of instance attributes on the source
    ``LlamaAttention`` that may change across transformers versions.

    Args:
        config: Model config providing head counts, dropout, and
            ``_attn_implementation``.
        layer_idx: Position of the parent decoder layer in the stack.
        attention: Source ``LlamaAttention`` module (for projection weights).
    """

    def __init__(self, config, layer_idx, attention):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.q_proj = QuantLinear(attention.q_proj, name="q", attn_instance=self)
        self.k_proj = QuantLinear(attention.k_proj, name="k", attn_instance=self)
        self.v_proj = QuantLinear(attention.v_proj, name="v", attn_instance=self)
        self.o_proj = QuantLinear(attention.o_proj, name="o", attn_instance=self)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        R_S_factors=None,
        **kwargs,
    ):
        """Project to Q/K/V, apply RoPE, run attention, project output.

        Args:
            hidden_states: ``(batch, seq, hidden)`` input.
            position_embeddings: ``(cos, sin)`` for rotary position embedding.
            attention_mask: Mask passed to the attention function.
            past_key_value: Optional KV cache updated when provided.
            cache_position: Positions for cache writes during decoding.
            R_S_factors: Dict of tensors unpacked into :class:`QuantLinear.forward`
                (e.g. ``R1``, ``S_qk``, ``S_ov``, ...).
            **kwargs: Additional kwargs for the attention implementation.

        Returns:
            A tuple ``(attn_output, attn_weights)`` as returned by the attention
            backend (weights may be ``None`` depending on implementation).
        """
        if R_S_factors is None:
            R_S_factors = {}
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states, **R_S_factors).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states, **R_S_factors).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states, **R_S_factors).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        attn_fn = _get_attention_fn(
            self.config._attn_implementation, llama_eager_attention_forward
        )
        attn_output, attn_weights = attn_fn(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, **R_S_factors)
        return attn_output, attn_weights


class QuantQwen3DecoderLayer(GradientCheckpointingLayer):
    """Qwen3 decoder block with quantized projections and RMSNorm wrappers.

    Same layout as :class:`QuantLlamaDecoderLayer` but uses
    :class:`_QuantQwen3Attention`.  All layer-level metadata is derived from
    ``config`` and ``layer_idx`` so that the wrapper is independent of
    attributes that may appear or disappear across transformers versions.

    Args:
        config: ``Qwen3Config`` (or compatible) with ``hidden_size`` and related
            fields.
        layer_idx: Position of this layer in the decoder stack.
        layer: Source ``Qwen3DecoderLayer`` whose submodules are wrapped.
    """

    def __init__(self, config, layer_idx, layer):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _QuantQwen3Attention(config, layer_idx, layer.self_attn)
        self.mlp = _QuantMLP(config, layer.mlp)
        self.input_layernorm = QuantRMSNorm(layer.input_layernorm)
        self.post_attention_layernorm = QuantRMSNorm(layer.post_attention_layernorm)
        self.R_S_modules = nn.ModuleDict(dict())
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        R1=None,
        **kwargs,
    ):
        """Pre-norm residual block: self-attn, then MLP (:class:`_QuantMLP`).

        Args:
            hidden_states: ``(batch, seq, hidden)`` tensor.
            attention_mask: Passed through to the attention implementation.
            position_ids: Optional positions; forwarded when supported by attention.
            past_key_value: Optional KV cache object.
            use_cache: Whether to populate ``past_key_value``.
            cache_position: Cache index tensor for incremental decoding.
            position_embeddings: ``(cos, sin)`` tuple for rotary embeddings.
            R1: Optional global rotation matrix merged into ``R_S_factors`` as
                ``R1`` for child modules.
            **kwargs: Extra arguments forwarded to the attention backend (e.g.
                flash-attention flags).

        Returns:
            Hidden states after the decoder block, shape matching ``hidden_states``.
        """
        R_S_factors = dict(R1=R1)
        R_S_factors.update({k: v.weight for k, v in self.R_S_modules.items() if v is not None})
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, S=R_S_factors.get("S_attn"))
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            R_S_factors=R_S_factors,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, S=R_S_factors.get("S_mlp"))
        hidden_states = self.mlp(hidden_states, R_S_factors=R_S_factors)
        hidden_states = residual + hidden_states
        return hidden_states


class _QuantQwen3Attention(nn.Module):
    """Qwen3 self-attention with Q/K norms, :class:`QuantLinear`, and RoPE.

    All layer-level metadata is derived from ``config`` and ``layer_idx`` so
    that the wrapper is independent of instance attributes on the source
    ``Qwen3Attention`` that may change across transformers versions.

    Args:
        config: Model config providing head counts, ``layer_types`` (for sliding
            window), dropout, and ``_attn_implementation``.
        layer_idx: Position of the parent decoder layer in the stack.
        attention: Source ``Qwen3Attention`` module (for norms and projection
            weights).
    """

    def __init__(self, config, layer_idx, attention):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.q_proj = QuantLinear(attention.q_proj, name="q", attn_instance=self)
        self.k_proj = QuantLinear(attention.k_proj, name="k", attn_instance=self)
        self.v_proj = QuantLinear(attention.v_proj, name="v", attn_instance=self)
        self.o_proj = QuantLinear(attention.o_proj, name="o", attn_instance=self)
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.sliding_window = (
            config.sliding_window if self.layer_type == "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        R_S_factors=None,
        **kwargs,
    ):
        """Project to Q/K/V, apply norms, RoPE, attention, and output projection.

        Args:
            hidden_states: ``(batch, seq, hidden)`` input.
            position_embeddings: ``(cos, sin)`` for rotary position embedding.
            attention_mask: Mask passed to the attention function.
            past_key_value: Optional KV cache updated when provided.
            cache_position: Positions for cache writes during decoding.
            R_S_factors: Dict of tensors unpacked into :class:`QuantLinear.forward`.
            **kwargs: Additional kwargs for the attention implementation (e.g.
                ``sliding_window`` is applied inside the attention call from
                config-derived ``self.sliding_window``).

        Returns:
            A tuple ``(attn_output, attn_weights)`` from the attention backend.
        """
        if R_S_factors is None:
            R_S_factors = {}
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(
            self.q_proj(hidden_states, **R_S_factors).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states, **R_S_factors).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states, **R_S_factors).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = qwen3_apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        attn_fn = _get_attention_fn(
            self.config._attn_implementation, qwen3_eager_attention_forward
        )
        attn_output, attn_weights = attn_fn(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, **R_S_factors)
        return attn_output, attn_weights


class _QuantMLP(nn.Module):
    """SwiGLU MLP built from three :class:`QuantLinear` layers.

    Args:
        config: Config with ``hidden_act`` naming the activation in ``ACT2FN``.
        mlp: Source MLP providing ``gate_proj``, ``up_proj``, and ``down_proj``.
    """

    def __init__(self, config, mlp):
        super().__init__()
        self.gate_proj = QuantLinear(mlp.gate_proj, name="gate")
        self.down_proj = QuantLinear(mlp.down_proj, name="down")
        self.up_proj = QuantLinear(mlp.up_proj, name="up")
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, R_S_factors=None):
        """Compute ``down( act(gate(x)) * up(x) )`` with shared rotation/scaling kwargs.

        Args:
            x: Hidden states before the MLP block.
            R_S_factors: Optional dict passed to each :class:`QuantLinear` (e.g.
                ``R1``, ``S_up_down``, ``S_mlp``).

        Returns:
            MLP output tensor, same leading shape as ``x``.
        """
        if R_S_factors is None:
            R_S_factors = {}
        return self.down_proj(
            self.act_fn(self.gate_proj(x, **R_S_factors)) * self.up_proj(x, **R_S_factors),
            **R_S_factors,
        )
