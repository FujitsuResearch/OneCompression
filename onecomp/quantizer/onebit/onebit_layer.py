"""
OneBit Layer implementation

Inference implementation for OneBit quantized Linear layers.
W ≈ a ⊙ sign(W) ⊙ b^T

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa
"""

import traceback
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# ========================================
# Bit packing / unpacking (same as DBF)
# ========================================


def my_pack(x: torch.Tensor) -> torch.Tensor:
    """Convert ±1 to {0,1} and pack into uint8 at 8:1 ratio. Tail is padded with +1."""
    flat = (x.flatten() >= 0).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = F.pad(flat, (0, pad), value=1)
    out = torch.zeros((flat.numel() // 8,), device=flat.device, dtype=torch.uint8)
    for i in range(8):
        out += flat[i::8] << (7 - i)
    return out


def my_unpack(x: torch.Tensor) -> torch.Tensor:
    """Expand uint8 to int8 {-1,+1} at 8x expansion (slice to required size downstream)."""
    out = torch.zeros((x.shape[0], 8), device=x.device, dtype=torch.int8)
    for i in range(8):
        out[:, i] = (x >> (7 - i)) & 1
    return out.flatten() * 2 - 1


# ========================================
# OneBitLinear layer (with bit packing support)
# ========================================


class OneBitLinear(nn.Module):
    """
    OneBit quantized Linear layer (with bit packing support).

    Computation: out = (a ⊙ sign(W) ⊙ b^T) @ x

    Where:
    - a: Row-wise scaling (out_features,)
    - b: Column-wise scaling (in_features,)
    - sign: Sign matrix {-1, +1} (out_features, in_features)

    Memory efficiency:
    - Sign matrix is bit-packed at 8:1 ratio
    - preunpack=True: unpack at initialization and keep in memory (fast, more memory)
    - preunpack=False: unpack on every forward pass (slow, less memory)
    """

    def __init__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        sign: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        preunpack: bool = True,
    ):
        super().__init__()

        # Scaling vectors
        self.register_buffer("a", a)
        self.register_buffer("b", b)

        # Dimension information
        self.out_features = a.shape[0]
        self.in_features = b.shape[0]

        # Bit packing of the sign matrix
        if sign.dtype == torch.uint8:
            # Already packed
            self.register_buffer("sign_packed", sign)
            self._sign_numel = self.out_features * self.in_features
        else:
            # Pack ±1 matrix
            self._sign_numel = sign.numel()
            sign_packed = my_pack(sign.flatten())
            self.register_buffer("sign_packed", sign_packed)

        # preunpack option: unpack and keep at initialization
        if preunpack:
            sign_unpacked = (
                my_unpack(self.sign_packed)[: self._sign_numel]
                .reshape(self.out_features, self.in_features)
                .to(torch.int8)
            )
            self.register_buffer("sign_matrix", sign_unpacked)
        else:
            self.register_buffer("sign_matrix", None)

        # Bias
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: out = (a ⊙ sign(W) ⊙ b^T) @ x

        Efficient computation:
        1. x = x * b (column-wise scaling)
        2. out = (sign * a[:, None]) @ x (sign matrix with row-wise scaling)
        """
        # Get sign matrix (branch based on preunpack setting)
        if self.sign_matrix is None:
            # Unpack every time
            sign = my_unpack(self.sign_packed)[: self._sign_numel].reshape(
                self.out_features, self.in_features
            )
        else:
            # Already pre-unpacked
            sign = self.sign_matrix

        # Apply b (column-wise scaling) to input
        x_scaled = x * self.b.to(x.dtype)

        # Weight combining sign matrix with a (row-wise scaling)
        # sign: {-1, +1} → float
        weight_matrix = sign.to(x.dtype) * self.a.to(x.dtype).unsqueeze(1)

        # Matrix multiplication
        out = torch.matmul(x_scaled, weight_matrix.t())

        # Add bias
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)

        return out

    def extra_repr(self) -> str:
        preunpack_status = "preunpacked" if self.sign_matrix is not None else "packed"
        return f"in_features={self.in_features}, out_features={self.out_features},\
              bias={self.bias is not None}, mode={preunpack_status}"


def replace_linear_with_onebit_layer(
    module: nn.Module, preunpack: bool = True
) -> Optional[OneBitLinear]:
    """
    Convert a Linear layer to OneBitLinear.

    Args:
        module: Target Linear layer (OneBit quantized)
        preunpack: Whether to unpack and keep the sign matrix at initialization

    Returns:
        OneBitLinear layer, or None if conversion is not possible
    """
    # Check if OneBit quantized
    if not hasattr(module, "is_quantized") or not module.is_quantized:
        return None

    # Check for OneBit metadata existence
    if not (
        hasattr(module.weight, "a")
        and hasattr(module.weight, "b")
        and hasattr(module.weight, "sign")
    ):
        print("[OneBit] Module missing metadata (a, b, sign)")
        return None

    try:
        # Get OneBit metadata
        a = module.weight.a.clone()
        b = module.weight.b.clone()
        sign = module.weight.sign.clone()

        # Get bias
        bias = module.bias.clone() if hasattr(module, "bias") and module.bias is not None else None

        # Create OneBitLinear layer (with bit packing support)
        onebit_layer = OneBitLinear(a, b, sign, bias, preunpack=preunpack)

        # Move to the original device
        onebit_layer = onebit_layer.to(module.weight.device)

        print(f"[OneBit] Created OneBitLinear layer: {module.out_features}x{module.in_features}")

        return onebit_layer

    except Exception as e:
        print(f"[OneBit] Error creating OneBitLinear: {e}")

        traceback.print_exc()
        return None


def extract_onebit_weights_for_save(model: torch.nn.Module) -> dict:
    """Extract OneBit weights from a model and create a dictionary for saving.

    Since the model has already been converted to OneBitLinear in run_layerwise_ptq.py,
    parameters are extracted directly from there.

    Args:
        model: Model containing OneBitLinear layers

    Returns:
        dict: Dictionary of OneBit weights (a, b, sign_packed)
    """
    onebit_weights = {}

    # For Vision LLM, get the language model part
    if hasattr(model, "is_vision_llm") and model.is_vision_llm:
        if hasattr(model.model, "language_model"):
            layers = model.model.language_model.layers
        elif hasattr(model.model, "text_model"):
            layers = model.model.text_model.layers
        else:
            layers = model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        print("[OneBit] Unsupported model structure")
        return onebit_weights

    # Recursively search all layers and submodules
    def find_onebit_modules(module, layer_idx, module_path=""):
        found_count = 0
        for name, submodule in module.named_children():
            current_path = f"{module_path}.{name}" if module_path else name

            # Detect OneBitLinear
            if isinstance(submodule, OneBitLinear):
                try:
                    # Extract parameters directly (move to CPU for memory efficiency)
                    prefix = f"model.layers.{layer_idx}.{current_path}"

                    # Scaling vectors
                    onebit_weights[f"{prefix}.a"] = submodule.a.cpu().clone()
                    onebit_weights[f"{prefix}.b"] = submodule.b.cpu().clone()

                    # Sign matrix (packed)
                    onebit_weights[f"{prefix}.sign_packed"] = submodule.sign_packed.cpu().clone()

                    # Metadata (for reconstruction)
                    onebit_weights[f"{prefix}.out_features"] = torch.tensor(submodule.out_features)
                    onebit_weights[f"{prefix}.in_features"] = torch.tensor(submodule.in_features)

                    # Bias (if present)
                    if submodule.bias is not None:
                        onebit_weights[f"{prefix}.bias"] = submodule.bias.cpu().clone()

                    found_count += 1

                except Exception as e:
                    print(f"[OneBit] Error extracting weights for {current_path}: {e}")

                    traceback.print_exc()

            # Legacy OneBit quantization format (kept for compatibility)
            elif hasattr(submodule, "is_quantized") and submodule.is_quantized:
                if (
                    hasattr(submodule, "weight")
                    and hasattr(submodule.weight, "a")
                    and hasattr(submodule.weight, "b")
                    and hasattr(submodule.weight, "sign")
                ):
                    try:
                        prefix = f"model.layers.{layer_idx}.{current_path}"

                        # Extract metadata
                        onebit_weights[f"{prefix}.a"] = submodule.weight.a.cpu().clone()
                        onebit_weights[f"{prefix}.b"] = submodule.weight.b.cpu().clone()

                        # Pack the sign matrix
                        sign = submodule.weight.sign.cpu()
                        if sign.dtype != torch.uint8:
                            sign_packed = my_pack(sign.flatten())
                        else:
                            sign_packed = sign
                        onebit_weights[f"{prefix}.sign_packed"] = sign_packed

                        # Metadata
                        onebit_weights[f"{prefix}.out_features"] = torch.tensor(
                            submodule.out_features
                        )
                        onebit_weights[f"{prefix}.in_features"] = torch.tensor(
                            submodule.in_features
                        )

                        # Bias
                        if hasattr(submodule, "bias") and submodule.bias is not None:
                            onebit_weights[f"{prefix}.bias"] = submodule.bias.cpu().clone()

                        found_count += 1

                    except Exception as e:
                        print(f"[OneBit] Error extracting weights for {current_path}: {e}")

            # Recursively search child modules
            found_count += find_onebit_modules(submodule, layer_idx, current_path)

        return found_count

    total_found = 0
    for i, layer in enumerate(layers):
        found_count = find_onebit_modules(layer, i)
        total_found += found_count

        # Free GPU memory per layer
        if i % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[OneBit] Total OneBit modules found: {total_found}")
    return onebit_weights
