"""GPTQ INT4 FusedMoE method for gpt-oss (WNA16 + swigluoai + expert bias).

Copyright 2025-2026 Fujitsu Ltd.

vLLM's stock ``MoeWNA16Method`` runs int4 grouped-weight experts through the
Triton ``fused_experts`` kernel, but it (a) hard-asserts SiLU activation and
(b) never registers or applies per-expert bias.  gpt-oss experts use the
``swigluoai`` activation and carry gate/up/down bias, so this subclass extends
``MoeWNA16Method`` to:

  * register ``w13_bias``/``w2_bias`` parameters and load them with a dedicated
    weight loader that maps gate/up into the fused ``w13`` block layout, and
  * forward the layer's actual activation to ``fused_experts`` (which already
    supports ``SWIGLUOAI`` and ``w1_bias``/``w2_bias``).

This is selected by ``MixedGPTQConfig`` for gpt-oss-style expert layers whose
``hidden_size`` (2880) is not Marlin-MoE compatible.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.model_executor.utils import set_weight_attrs


def _bias_weight_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str,
    shard_id: str,
    expert_id: int,
    return_success: bool = False,
):
    """Load a per-expert projection bias into the fused w13/w2 bias params.

    ``FusedMoE.weight_loader`` has no code path for bias tensors, so gpt-oss
    expert bias must be placed manually.  Gate (``w1``) fills the first half of
    ``w13_bias`` and up (``w3``) the second half; down (``w2``) fills
    ``w2_bias`` directly.
    """
    data = param.data
    loaded_weight = loaded_weight.to(device=data.device, dtype=data.dtype)
    if "w13_bias" in weight_name:
        shard_size = data.shape[1] // 2
        if shard_id == "w1":
            data[expert_id, :shard_size].copy_(loaded_weight)
        else:  # "w3"
            data[expert_id, shard_size:].copy_(loaded_weight)
    else:  # "w2_bias"
        data[expert_id].copy_(loaded_weight)
    return True if return_success else None


class GptOssWNA16MoEMethod(MoeWNA16Method):
    """MoE WNA16 method with per-expert bias and non-SiLU activation support."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        super().create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

        # gpt-oss experts carry gate/up/down bias.  MoeWNA16Method does not
        # register these, so add them with a dedicated loader.
        w13_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_bias", w13_bias)
        set_weight_attrs(w13_bias, extra_weight_attrs)
        w13_bias.weight_loader = _bias_weight_loader

        w2_bias = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_bias", w2_bias)
        set_weight_attrs(w2_bias, extra_weight_attrs)
        w2_bias.weight_loader = _bias_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # gate/up load as blocks [gate; up] but swigluoai_and_mul reads them
        # interleaved (gate=x[::2], up=x[1::2]); reorder w13 output channels.
        inter = layer.intermediate_size_per_partition
        perm = torch.empty(2 * inter, dtype=torch.long)
        perm[0::2] = torch.arange(0, inter)
        perm[1::2] = torch.arange(inter, 2 * inter)
        for name in ("w13_qweight", "w13_scales", "w13_bias"):
            p = getattr(layer, name, None)
            if p is None or p.numel() == 0:
                continue
            reordered = p.data.index_select(1, perm.to(p.device))
            p.data.copy_(reordered)
        # Asymmetric quant only: w13_qzeros packs bit8_pack_factor output
        # channels into each uint8 along dim 1 (channel c -> byte c // pack,
        # nibble c % pack), so a plain index_select would mix channels across
        # bytes.  Unpack to per-channel nibbles, apply the same interleave, and
        # repack to keep zero points aligned with the reordered weights/scales.
        # (Symmetric gptq leaves w13_qzeros as an empty placeholder -> skipped.)
        qzeros = getattr(layer, "w13_qzeros", None)
        if qzeros is not None and qzeros.numel() != 0:
            self._reorder_packed_qzeros(layer, qzeros.data, perm.to(qzeros.device))

    @staticmethod
    def _reorder_packed_qzeros(
        layer: torch.nn.Module, z: torch.Tensor, perm: torch.Tensor
    ) -> None:
        pack = layer.quant_config.bit8_pack_factor
        bits = 8 // pack
        mask = (1 << bits) - 1
        num_experts, num_bytes, num_groups = z.shape
        shifts = (torch.arange(pack, device=z.device) * bits).view(1, 1, pack, 1)
        # 1. Unpack: (E, bytes, G) -> (E, bytes, pack, G) -> (E, 2*inter, G).
        #    Split each byte into its nibbles and lay them out in channel order
        #    (channel index = byte * pack + nibble) so one channel == one row.
        unpacked = (z.unsqueeze(2) >> shifts) & mask
        unpacked = unpacked.reshape(num_experts, num_bytes * pack, num_groups)
        # 2. Now the length-(2*inter) per-channel perm applies directly.
        unpacked = unpacked.index_select(1, perm)
        # 3. Repack: pack `pack` channels back into each uint8.
        unpacked = unpacked.reshape(num_experts, num_bytes, pack, num_groups)
        repacked = torch.zeros_like(z)
        for i in range(pack):
            repacked |= (unpacked[:, :, i, :] & mask).to(z.dtype) << (i * bits)
        z.copy_(repacked)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        config = super().get_fused_moe_quant_config(layer)
        if config is not None:
            config._w1.bias = layer.w13_bias
            config._w2.bias = layer.w2_bias
        return config

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe import fused_experts

        return fused_experts(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not self.moe.disable_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )


def wrap_moe_method(quant_method, layer):
    """Substitute this method for swigluoai (gpt-oss) WNA16 MoE layers only.

    Registered as a ``MixedGPTQConfig`` MoE adapter so the generic plugin needs
    no gpt-oss-specific import.  Only gpt-oss-style layers -- ``swigluoai``
    activation with per-expert bias -- need this subclass; SiLU MoE layers
    (e.g. Qwen3.6-A3B) keep the stock ``MoeWNA16Method`` unchanged, since their
    block-layout ``w13`` weights must NOT go through the swigluoai interleave in
    ``GptOssWNA16MoEMethod.process_weights_after_loading``.  Non-WNA16 methods
    and non-swigluoai layers are returned unchanged.
    """
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    # NOTE: swigluoai activation is the sole discriminator for gpt-oss here.
    # If a non-gpt-oss MoE model that also uses swigluoai (but not gpt-oss's
    # per-expert bias / w13 block layout) appears, this needs a stricter check.
    if (
        type(quant_method) is MoeWNA16Method
        and getattr(layer, "activation", None) == MoEActivation.SWIGLUOAI
    ):
        return GptOssWNA16MoEMethod(quant_method.quant_config, quant_method.moe)
    return quant_method
