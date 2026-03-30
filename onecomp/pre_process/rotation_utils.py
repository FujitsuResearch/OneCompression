"""Weight rotation utilities for preprocessing.

Provides functions to fuse layer norms and apply rotation/scaling matrices
to model weights in-place.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yusei Kawakami
"""

import gc

import torch
import torch.nn as nn

from .hadamard_utils import get_hadK, matmul_hadU_cuda

# ============================================================
# Helper
# ============================================================


def cleanup_memory():
    """Run Python GC and clear the CUDA cache to free allocator memory.

    Useful between rotation passes on large models to lower peak VRAM usage.
    """
    gc.collect()
    torch.cuda.empty_cache()


def find_linear_layers(module, layers=None, name=""):
    """Recursively find all layers whose type is in *layers*.

    Args:
        module: Root module to search.
        layers: List of target layer classes.  Defaults to ``[nn.Linear]``,
            which also matches ``QuantLinear`` (a subclass of ``nn.Linear``).
        name: Name prefix (used for recursion).

    Returns:
        dict[str, nn.Module]: Dotted names of matching submodules mapped to their
            module instances.
    """
    if layers is None:
        layers = [nn.Linear]
    if type(module) in layers:
        return {name: module}
    res = {}
    for child_name, child in module.named_children():
        full = f"{name}.{child_name}" if name else child_name
        res.update(find_linear_layers(child, layers=layers, name=full))
    return res


# ============================================================
# Weight-tying & LayerNorm fusion
# ============================================================


def untie_word_embeddings(model: nn.Module) -> None:
    """Break the weight sharing between ``embed_tokens`` and ``lm_head``.

    Must be called before ``fuse_layer_norms`` / ``rotate_model`` so that
    independent transforms on each do not overwrite each other.

    Args:
        model: Causal LM with ``config``, ``model.embed_tokens``, and optional
            ``lm_head`` when ``tie_word_embeddings`` is enabled.
    """
    if getattr(model.config, "tie_word_embeddings", False) and hasattr(model, "lm_head"):
        if model.lm_head.weight is model.model.embed_tokens.weight:
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())
        model.config.tie_word_embeddings = False


def fuse_ln_linear(layernorm: nn.Module, linear: nn.Module):
    """Fuse RMSNorm weight into the adjacent Linear layer.

    Args:
        layernorm: Normalization module whose ``weight`` scales the linear input
            row-wise; must match ``linear`` feature width.
        linear: ``nn.Linear`` updated in-place on ``weight``.
    """
    dtype = linear.weight.dtype
    device = linear.weight.device
    W = linear.weight.data.to(dtype=torch.float64)
    linear.weight.data = (W * layernorm.weight.to(dtype=torch.float64, device=device)).to(dtype)


def fuse_layer_norms(model: nn.Module):
    """Fuse all RMSNorm weights into adjacent Linear layers.

    Args:
        model: Decoder-only LM with ``model.layers`` (each with ``input_layernorm``,
            ``post_attention_layernorm``, attention projections, and MLP), and
            optionally ``model.norm`` plus ``lm_head``.
    """
    for layer in model.model.layers:
        fuse_ln_linear(layer.input_layernorm, layer.self_attn.q_proj)
        fuse_ln_linear(layer.input_layernorm, layer.self_attn.k_proj)
        fuse_ln_linear(layer.input_layernorm, layer.self_attn.v_proj)
        layer.input_layernorm.weight.data = torch.ones_like(layer.input_layernorm.weight.data)

        fuse_ln_linear(layer.post_attention_layernorm, layer.mlp.up_proj)
        fuse_ln_linear(layer.post_attention_layernorm, layer.mlp.gate_proj)
        layer.post_attention_layernorm.weight.data = torch.ones_like(
            layer.post_attention_layernorm.weight.data
        )

    if hasattr(model, "lm_head"):
        fuse_ln_linear(model.model.norm, model.lm_head)
        model.model.norm.weight.data = torch.ones_like(model.model.norm.weight.data)


# ============================================================
# Online Hadamard hook for down_proj
# ============================================================


def make_online_hadamard_hook(had_K, K, fp32_had: bool = False):
    """Build a forward pre-hook that Hadamard-transforms the module input tensor.

    Keeps inference consistent with offline Hadamard rotation applied to
    ``down_proj`` weights when ``R1`` is absent.

    Args:
        had_K: Hadamard operator (or factorization) from ``get_hadK``.
        K: Block size passed to ``matmul_hadU_cuda``.
        fp32_had: If True, compute the Hadamard product in float32, then cast
            back to the input dtype.

    Returns:
        Callable: ``hook(module, inp)`` suitable for
            ``register_forward_pre_hook``; returns ``(x_transformed,)``.
    """

    def hook(module, inp):
        x = inp[0]
        x_dtype = x.dtype
        if fp32_had:
            x = matmul_hadU_cuda(x.float(), had_K, K).to(x_dtype)
        else:
            x = matmul_hadU_cuda(x, had_K, K)
        return (x,)

    return hook


# ============================================================
# Per-layer rotation helpers
# ============================================================


def _to_f64(tensor, dev):
    """Cast *tensor* to float64 on *dev* for stable rotation arithmetic.

    Args:
        tensor: Source tensor.
        dev: Target device string or ``torch.device``.

    Returns:
        torch.Tensor: Same shape as *tensor*, ``dtype=torch.float64`` on *dev*.
    """
    return tensor.to(dtype=torch.float64, device=dev)


def rotate_embeddings(model, R1, dev):
    """Apply the global hidden-state rotation (or Hadamard) to embedding weights.

    Mutates ``model.model.embed_tokens.weight`` in-place.

    Args:
        model: Causal LM exposing ``model.embed_tokens``.
        R1: Global rotation ``(hidden, hidden)``, or ``None`` to apply a
            Hadamard transform sized to the embedding dimension.
        dev: Device for intermediate float64 tensors.
    """
    embed = model.model.embed_tokens
    dtype, device = embed.weight.dtype, embed.weight.device
    W = embed.weight.data.to(dtype=torch.float64, device=dev)
    if R1 is not None:
        W = W @ _to_f64(R1, dev)
    else:
        had_K, K = get_hadK(W.shape[-1])
        W = matmul_hadU_cuda(W, had_K, K)
    embed.weight.data = W.to(device=device, dtype=dtype)


def rotate_head(model, R1, dev):
    """Apply the same global rotation as embeddings to the output projection.

    No-op if ``lm_head`` is missing. Updates ``lm_head.weight`` in-place.

    Args:
        model: Causal LM with optional ``lm_head``.
        R1: Same semantics as ``rotate_embeddings``; ``None`` selects Hadamard.
        dev: Device for intermediate float64 tensors.
    """
    if not hasattr(model, "lm_head"):
        return
    layer = model.lm_head
    dtype, device = layer.weight.dtype, layer.weight.device
    W = layer.weight.data.to(dtype=torch.float64, device=dev)
    if R1 is not None:
        W = W @ _to_f64(R1, dev)
    else:
        had_K, K = get_hadK(layer.in_features)
        W = matmul_hadU_cuda(W, had_K, K)
    layer.weight.data = W.to(device=device, dtype=dtype)


def rotate_q_proj(self_attn, num_kv_heads, R1, S_attn, S_qk, dev):
    """Update query projection weights for fused RMSNorm, Q/K scaling, and ``R1``.

    Args:
        self_attn: Attention block with ``q_proj``, ``head_dim``, and GQA fields.
        num_kv_heads: KV head count (used to broadcast ``S_qk`` across groups).
        R1: Global rotation on input features, or ``None`` for Hadamard.
        S_attn: Per-channel scaling after rotation, or ``None``.
        S_qk: Half-head RoPE scaling for Q (and K), or ``None``.
        dev: Device for intermediate float64 tensors.
    """
    layer = self_attn.q_proj
    dtype, device = layer.weight.dtype, layer.weight.device
    W = layer.weight.data.to(dtype=torch.float64, device=dev)
    bias = layer.bias.data.to(dtype=torch.float64, device=dev) if layer.bias is not None else None

    if S_qk is not None:
        half = self_attn.head_dim // 2
        S_half = S_qk.view(num_kv_heads, half)
        S = torch.cat([S_half, S_half], dim=-1)
        S = (
            S[:, None, :]
            .expand(num_kv_heads, self_attn.num_key_value_groups, self_attn.head_dim)
            .reshape(-1)
        )
        W = W / S.view(-1, 1).to(dtype=torch.float64, device=dev)
        if bias is not None:
            bias = bias / S.view(-1).to(dtype=torch.float64, device=dev)

    if R1 is not None:
        W = W @ _to_f64(R1, dev)
    else:
        had_K, K = get_hadK(layer.in_features)
        W = matmul_hadU_cuda(W, had_K, K)

    if S_attn is not None:
        W = W * _to_f64(S_attn, dev)

    layer.weight.data = W.to(device=device, dtype=dtype)
    if bias is not None:
        layer.bias.data = bias.to(device=device, dtype=dtype)


def rotate_k_proj(self_attn, num_kv_heads, R1, S_attn, S_qk, dev):
    """Update key projection weights for fused RMSNorm, Q/K scaling, and ``R1``.

    Args:
        self_attn: Attention block with ``k_proj`` and GQA metadata.
        num_kv_heads: KV head count for reshaping ``S_qk``.
        R1: Global rotation on input features, or ``None`` for Hadamard.
        S_attn: Per-channel scaling after rotation, or ``None``.
        S_qk: Half-head RoPE scaling for K, or ``None``.
        dev: Device for intermediate float64 tensors.
    """
    layer = self_attn.k_proj
    dtype, device = layer.weight.dtype, layer.weight.device
    W = layer.weight.data.to(dtype=torch.float64, device=dev)
    bias = layer.bias.data.to(dtype=torch.float64, device=dev) if layer.bias is not None else None

    if S_qk is not None:
        half = self_attn.head_dim // 2
        S_half = S_qk.view(num_kv_heads, half)
        S = torch.cat([S_half, S_half], dim=-1).reshape(-1)
        W = W * S.view(-1, 1).to(dtype=torch.float64, device=dev)
        if bias is not None:
            bias = bias * S.view(-1).to(dtype=torch.float64, device=dev)

    if R1 is not None:
        W = W @ _to_f64(R1, dev)
    else:
        had_K, K = get_hadK(layer.in_features)
        W = matmul_hadU_cuda(W, had_K, K)

    if S_attn is not None:
        W = W * _to_f64(S_attn, dev)

    layer.weight.data = W.to(device=device, dtype=dtype)
    if bias is not None:
        layer.bias.data = bias.to(device=device, dtype=dtype)


def rotate_v_proj(self_attn, num_kv_heads, R1, R2, S_attn, S_ov, dev):
    """Update value projection for input rotation, head-dim ``R2``, and scalings.

    Args:
        self_attn: Attention block with ``v_proj`` and ``head_dim``.
        num_kv_heads: KV head count; included so ``rotate_model`` can call all
            ``rotate_*`` helpers with the same positional arguments.
        R1: Global rotation on input features, or ``None`` for Hadamard.
        R2: Per-head hidden rotation ``(head_dim, head_dim)``, or ``None`` for
            Hadamard on the head axis.
        S_attn: Per-channel scaling after rotation, or ``None``.
        S_ov: Value/output path scaling absorbed before ``R2``, or ``None``.
        dev: Device for intermediate float64 tensors.
    """
    layer = self_attn.v_proj
    dtype, device = layer.weight.dtype, layer.weight.device
    W = layer.weight.data.to(dtype=torch.float64, device=dev)
    bias = layer.bias.data.to(dtype=torch.float64, device=dev) if layer.bias is not None else None

    if S_ov is not None:
        W = W / S_ov.view(-1, 1).to(dtype=torch.float64, device=dev)
        if bias is not None:
            bias = bias / S_ov.view(-1).to(dtype=torch.float64, device=dev)

    if R1 is not None:
        W = W @ _to_f64(R1, dev)
    else:
        had_K, K = get_hadK(layer.in_features)
        W = matmul_hadU_cuda(W, had_K, K)

    if R2 is not None:
        had_dim = R2.shape[0]
        W_ = W.t()
        shape = W_.shape
        temp = W_.reshape(-1, shape[-1] // had_dim, had_dim) @ _to_f64(R2, dev)
        W = temp.reshape(shape).t()
        if bias is not None:
            bias = (bias.reshape(-1, had_dim) @ _to_f64(R2, dev)).reshape(-1)
    else:
        had_dim = self_attn.head_dim
        had_K, K = get_hadK(had_dim)
        W_ = W.t()
        shape = W_.shape
        temp = matmul_hadU_cuda(W_.reshape(-1, shape[-1] // had_dim, had_dim), had_K, K)
        W = temp.reshape(shape).t()
        if bias is not None:
            bias = matmul_hadU_cuda(bias.reshape(-1, had_dim), had_K, K).reshape(-1)

    if S_attn is not None:
        W = W * _to_f64(S_attn, dev)

    layer.weight.data = W.to(device=device, dtype=dtype)
    if bias is not None:
        layer.bias.data = bias.to(device=device, dtype=dtype)


def rotate_o_proj(self_attn, num_kv_heads, num_heads, R1, R2, S_ov, dev):
    """Update output projection for ``S_ov``, global rotation on outputs, and ``R2``.

    Args:
        self_attn: Attention block with ``o_proj`` and GQA layout.
        num_kv_heads: KV head count for expanding ``S_ov`` when its length is
            below ``o_proj`` input width.
        num_heads: Total attention heads; kept for call-site uniformity with
            ``rotate_model`` (not read in this function).
        R1: Left-multiplied on output features, or ``None`` for Hadamard on
            ``out_features``.
        R2: Right-applied head-dim rotation on the flattened head axis, or
            ``None`` for Hadamard on ``head_dim``.
        S_ov: Scales aligned with V/O features, or ``None``.
        dev: Device for intermediate float64 tensors.
    """
    layer = self_attn.o_proj
    dtype, device = layer.weight.dtype, layer.weight.device
    W = layer.weight.data.to(dtype=torch.float64, device=dev)
    bias = layer.bias.data.to(dtype=torch.float64, device=dev) if layer.bias is not None else None

    if S_ov is not None:
        in_features = W.shape[1]
        S = S_ov
        if S.numel() < in_features:
            S = S.reshape(num_kv_heads, -1)
            n_head, d = S.shape
            S = S[:, None, :].expand(n_head, self_attn.num_key_value_groups, d).reshape(-1)
        W = W * S.view(1, -1).to(dtype=torch.float64, device=dev)

    if R1 is not None:
        W = _to_f64(R1, dev).T @ W
        if bias is not None:
            bias = _to_f64(R1, dev).T @ bias
    else:
        had_K, K = get_hadK(layer.out_features)
        W = matmul_hadU_cuda(W.t(), had_K, K).t()
        if bias is not None:
            bias = matmul_hadU_cuda(bias.unsqueeze(0), had_K, K).squeeze(0)

    if R2 is not None:
        had_dim = R2.shape[0]
        shape = W.shape
        temp = W.reshape(-1, shape[-1] // had_dim, had_dim) @ _to_f64(R2, dev)
        W = temp.reshape(shape)
    else:
        had_dim = self_attn.head_dim
        had_K, K = get_hadK(had_dim)
        shape = W.shape
        temp = matmul_hadU_cuda(W.reshape(-1, shape[-1] // had_dim, had_dim), had_K, K)
        W = temp.reshape(shape)

    layer.weight.data = W.to(device=device, dtype=dtype)
    if bias is not None:
        layer.bias.data = bias.to(device=device, dtype=dtype)


def rotate_input_layernorm(layer, S_attn, dev):
    """Scale pre-attention RMSNorm weights by ``1 / S_attn`` after QKV fusion.

    Args:
        layer: Decoder layer with ``input_layernorm``.
        S_attn: Per-hidden scaling from the attention path, or ``None`` to skip.
        dev: Device for intermediate float64 tensors.
    """
    if S_attn is None:
        return
    ln = layer.input_layernorm
    dtype, device = ln.weight.dtype, ln.weight.device
    W = ln.weight.data.to(dtype=torch.float64, device=dev)
    ln.weight.data = (W / _to_f64(S_attn, dev)).to(device=device, dtype=dtype)


def rotate_post_attention_layernorm(layer, S_mlp, dev):
    """Scale post-attention RMSNorm weights by ``1 / S_mlp`` for the MLP branch.

    Args:
        layer: Decoder layer with ``post_attention_layernorm``.
        S_mlp: Per-hidden scaling shared by MLP projections, or ``None``.
        dev: Device for intermediate float64 tensors.
    """
    if S_mlp is None:
        return
    ln = layer.post_attention_layernorm
    dtype, device = ln.weight.dtype, ln.weight.device
    W = ln.weight.data.to(dtype=torch.float64, device=dev)
    ln.weight.data = (W / _to_f64(S_mlp, dev)).to(device=device, dtype=dtype)


def rotate_up_proj(layer, R1, S_mlp, S_up_down, dev):
    """Update ``up_proj`` for ``S_up_down``, global ``R1``, and MLP scaling.

    Args:
        layer: ``nn.Linear`` for the MLP up projection.
        R1: Global rotation on input features, or ``None`` for Hadamard.
        S_mlp: Per-channel scaling after rotation, or ``None``.
        S_up_down: Shared scaling between up and down paths, or ``None``.
        dev: Device for intermediate float64 tensors.
    """
    dtype, device = layer.weight.dtype, layer.weight.device
    W = layer.weight.data.to(dtype=torch.float64, device=dev)
    bias = layer.bias.data.to(dtype=torch.float64, device=dev) if layer.bias is not None else None

    if S_up_down is not None:
        W = W / _to_f64(S_up_down, dev).view(-1, 1)
        if bias is not None:
            bias = bias / _to_f64(S_up_down, dev).view(-1)

    if R1 is not None:
        W = W @ _to_f64(R1, dev)
    else:
        had_K, K = get_hadK(layer.in_features)
        W = matmul_hadU_cuda(W, had_K, K)

    if S_mlp is not None:
        W = W * _to_f64(S_mlp, dev)

    layer.weight.data = W.to(device=device, dtype=dtype)
    if bias is not None:
        layer.bias.data = bias.to(device=device, dtype=dtype)


def rotate_gate_proj(layer, R1, S_mlp, dev):
    """Update ``gate_proj`` for global ``R1`` and optional MLP scaling.

    Args:
        layer: ``nn.Linear`` for the gated activation branch (e.g. SwiGLU gate).
        R1: Global rotation on input features, or ``None`` for Hadamard.
        S_mlp: Per-channel scaling after rotation, or ``None``.
        dev: Device for intermediate float64 tensors.
    """
    dtype, device = layer.weight.dtype, layer.weight.device
    W = layer.weight.data.to(dtype=torch.float64, device=dev)

    if R1 is not None:
        W = W @ _to_f64(R1, dev)
    else:
        had_K, K = get_hadK(layer.in_features)
        W = matmul_hadU_cuda(W, had_K, K)

    if S_mlp is not None:
        W = W * _to_f64(S_mlp, dev)

    layer.weight.data = W.to(device=device, dtype=dtype)


def rotate_down_proj(layer, R1, S_up_down, dev):
    """Update ``down_proj`` for ``S_up_down``, output-side ``R1``, and Hadamard.

    Applies an additional Hadamard on input features to pair with online hooks.

    Args:
        layer: ``nn.Linear`` for the MLP down projection.
        R1: Left-multiplied on output features, or ``None`` for Hadamard on
            ``out_features``.
        S_up_down: Scales applied along the intermediate width, or ``None``.
        dev: Device for intermediate float64 tensors.
    """
    dtype, device = layer.weight.dtype, layer.weight.device
    W = layer.weight.data.to(dtype=torch.float64, device=dev)
    bias = layer.bias.data.to(dtype=torch.float64, device=dev) if layer.bias is not None else None

    if S_up_down is not None:
        W = W * _to_f64(S_up_down, dev).view(1, -1)

    if R1 is not None:
        W = _to_f64(R1, dev).T @ W
        if bias is not None:
            bias = _to_f64(R1, dev).T @ bias
    else:
        had_K, K = get_hadK(layer.out_features)
        W = matmul_hadU_cuda(W.t(), had_K, K).t()
        if bias is not None:
            bias = matmul_hadU_cuda(bias.unsqueeze(0), had_K, K).squeeze(0)

    had_K, K = get_hadK(layer.in_features)
    W = matmul_hadU_cuda(W, had_K, K)

    layer.weight.data = W.to(device=device, dtype=dtype)
    if bias is not None:
        layer.bias.data = bias.to(device=device, dtype=dtype)


# ============================================================
# High-level: apply rotation to an entire model (eval path)
# ============================================================


@torch.no_grad()
def rotate_model(
    model,
    R1,
    R2_per_layer,
    S_attn_per_layer=None,
    S_qk_per_layer=None,
    S_ov_per_layer=None,
    S_mlp_per_layer=None,
    S_up_down_per_layer=None,
    dev="cpu",
):
    """Apply learned rotation / scaling matrices to model weights in-place.

    Args:
        model: HuggingFace CausalLM model.
        R1: Global rotation matrix ``(hidden, hidden)`` or ``None``.
        R2_per_layer: ``dict[int, Tensor]`` — per-layer head-dim rotation, or ``None``.
        S_*_per_layer: ``dict[int, Tensor]`` — per-layer scaling vectors, or ``None``.
        dev: Device for intermediate computation.
    """
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    R1_data = R1.to(device=dev, dtype=torch.float64) if R1 is not None else None

    rotate_embeddings(model, R1_data, dev)
    rotate_head(model, R1_data, dev)
    cleanup_memory()

    for i, layer in enumerate(model.model.layers):
        R2 = (
            R2_per_layer[i].to(device=dev, dtype=torch.float64)
            if R2_per_layer is not None
            else None
        )
        S_attn = (
            S_attn_per_layer[i].to(device=dev, dtype=torch.float64)
            if S_attn_per_layer is not None
            else None
        )
        S_qk = (
            S_qk_per_layer[i].to(device=dev, dtype=torch.float64)
            if S_qk_per_layer is not None
            else None
        )
        S_ov = (
            S_ov_per_layer[i].to(device=dev, dtype=torch.float64)
            if S_ov_per_layer is not None
            else None
        )
        S_mlp = (
            S_mlp_per_layer[i].to(device=dev, dtype=torch.float64)
            if S_mlp_per_layer is not None
            else None
        )
        S_up_down = (
            S_up_down_per_layer[i].to(device=dev, dtype=torch.float64)
            if S_up_down_per_layer is not None
            else None
        )

        rotate_q_proj(layer.self_attn, num_kv_heads, R1_data, S_attn, S_qk, dev)
        rotate_k_proj(layer.self_attn, num_kv_heads, R1_data, S_attn, S_qk, dev)
        rotate_v_proj(layer.self_attn, num_kv_heads, R1_data, R2, S_attn, S_ov, dev)
        rotate_o_proj(layer.self_attn, num_kv_heads, num_heads, R1_data, R2, S_ov, dev)

        rotate_input_layernorm(layer, S_attn, dev)
        rotate_post_attention_layernorm(layer, S_mlp, dev)

        rotate_up_proj(layer.mlp.up_proj, R1_data, S_mlp, S_up_down, dev)
        rotate_gate_proj(layer.mlp.gate_proj, R1_data, S_mlp, dev)
        rotate_down_proj(layer.mlp.down_proj, R1_data, S_up_down, dev)

    cleanup_memory()


def register_online_hadamard_hooks(model, fp32_had: bool = False, layers_cls=None):
    """Register Hadamard ``forward_pre_hook`` on all ``down_proj`` layers.

    Args:
        model: Target model.
        fp32_had: Use FP32 for Hadamard transform.
        layers_cls: Layer classes to search for (passed to ``find_linear_layers``).

    Returns:
        list[torch.utils.hooks.RemovableHandle]: One handle per registered hook;
            keep references if you need to call ``handle.remove()`` later.
    """
    layers = find_linear_layers(model, layers=layers_cls)
    hooks = []
    for name, module in layers.items():
        if "down_proj" in name:
            had_K, K = get_hadK(module.in_features)
            hook = make_online_hadamard_hook(had_K, K, fp32_had=fp32_had)
            handle = module.register_forward_pre_hook(hook)
            hooks.append(handle)
    return hooks
