"""Copyright 2025-2026 Fujitsu Ltd."""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def unpack_sign_bits(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpack an int8 tensor into a ±1 tensor.

    Args:
        packed: Packed int8 tensor.
        original_shape: Shape of the original tensor.

    Returns:
        unpacked: Tensor unpacked ±1 values.
    """
    device = packed.device
    dtype = torch.float16

    # Unpack bits in the same order as my_unpack in the quantization script:
    # extract bits as (x >> 7, >> 6, >> 5, ..., >> 0)
    int8_tensor = packed

    # Bit-shift amounts: 7, 6, 5, ..., 0
    shifts = torch.arange(7, -1, -1, device=device).view(1, 8)
    expanded_int8 = int8_tensor.unsqueeze(-1)

    # Extract each bit and convert {0, 1} → {-1, +1}
    unpacked_bits = ((expanded_int8 >> shifts) & 1).to(dtype)
    unpacked_bits = unpacked_bits.view(int8_tensor.shape[0], -1)

    # {0,1} → {-1,+1}
    fp16_tensor = (unpacked_bits.to(torch.int8)) * 2 - 1

    # Reshape to original shape, truncating any excess elements
    total_elements = 1
    for dim in original_shape:
        total_elements *= dim

    return fp16_tensor.flatten()[:total_elements].reshape(original_shape)


class BitLinearPacked(nn.Module):
    """Linear transformation using a packed binary weight matrix.

    Stores a ±1 binary matrix in an 8:1 packed form and unpacks it
    at forward time to compute the matrix multiplication.

    Args:
        packed_binary: Packed uint8 tensor.
        shape: Original shape of the packed tensor.
        preunpack: Whether to unpack and cache at initialization for speed.
    """

    def __init__(
        self,
        packed_binary: Optional[torch.Tensor] = None,
        shape: Optional[Tuple[int, int]] = None,
        preunpack: bool = True,
    ):
        super().__init__()

        # Generating from a packed tensor
        self.shape = tuple(shape)
        self._numel = shape[0] * shape[1]
        self.register_buffer("bp", packed_binary)

        # Optimization: pre-unpack and cache the binary matrix
        if preunpack:
            unpacked = unpack_sign_bits(self.bp, self.shape).to(torch.int8)
            self.bit_mat = unpacked
        else:
            self.bit_mat = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Retrieve or unpack the binary matrix
        if self.bit_mat is None:
            bit_mat = unpack_sign_bits(self.bp, self.shape)
        else:
            bit_mat = self.bit_mat
        # Compute matrix multiplication
        weight_matrix = (bit_mat.to(x.dtype)).t()
        return x.matmul(weight_matrix)


class DBFLinear_NAIVE(nn.Module):
    """
    Naive implementation of a DBF (Differentiable Binary Factorization) linear layer.
    Approximates a linear transformation using two binary matrices, three element-wise
    scaling vectors, and an optional bias. All weights and biases are handled in float16.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        mid_features (int): Number of intermediate features.
                            Computed as the minimum of
                            ``w_bit * (in_features * out_features) / (in_features + out_features)``
                            and ``min(in_features, out_features)``.
        w_bit (int): Bit-width used to determine the intermediate feature count.
        training (bool): Flag indicating training or evaluation mode. Currently unused.

        # Buffers: registered via register_buffer, saved as part of the model state
        # but excluded from gradient computation by default.

        scaling0 (torch.Tensor): Scaling factor applied to the input ``x``. Shape: ``(in_features,)``, dtype: float16.
        bp1 (torch.Tensor): Packed uint8 representation of the first binary matrix ``binary1``.
                            Shape: ``((mid_features * in_features + 7) // 8,)``.
                            Unpacked shape: ``(mid_features, in_features)``.
        scaling2 (torch.Tensor): Scaling factor applied after the first binary transform.
                                 Shape: ``(mid_features,)``, dtype: float16.
        bp3 (torch.Tensor): Packed uint8 representation of the second binary matrix ``binary3``.
                            Shape: ``((out_features * mid_features + 7) // 8,)``.
                            Unpacked shape: ``(out_features, mid_features)``.
        scaling4 (torch.Tensor): Scaling factor applied to the final output.
                                 Shape: ``(out_features,)``, dtype: float16.
        bias (torch.Tensor, optional): Bias added to the output before returning.
                                       Shape: ``(out_features,)``, dtype: float16.
                                       Present only when the ``bias`` argument is True.

        # Members initialized at runtime by post_init()

        binary1 (BitLinearPacked): First binary linear transform module, built from ``bp1``.
        binary3 (BitLinearPacked): Second binary linear transform module, built from ``bp3``.
    """

    def __init__(self, w_bit, in_features, out_features, bias, dev, training=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Compute the number of intermediate features.
        # Use the w_bit-based formula and clip by min(in_features, out_features)
        # to control factorization cost while avoiding unnecessary dimension growth.
        mid_features1 = int(w_bit * (in_features * out_features) / (in_features + out_features))
        mid_features = min(min(in_features, out_features), mid_features1)
        self.mid_features = mid_features
        self.w_bit = w_bit
        self.training = training

        # Assert that in_features is a multiple of 8
        assert in_features % (8) == 0

        # ========== Register model parameters via register_buffer ==========
        # These tensors are part of the model state but are not tracked for gradients by default.

        # Input scaling factor
        self.register_buffer(
            "scaling0",
            torch.zeros(
                (in_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        # Packed uint8 representation of the first binary matrix
        # Number of bytes = ceil(mid_features * in_features / 8)
        self.register_buffer(
            "bp1",
            torch.zeros(
                ((mid_features * in_features + (8) - 1) // (8)),
                dtype=torch.uint8,
                device=dev,
            ),
        )
        # Intermediate output scaling factor
        self.register_buffer(
            "scaling2",
            torch.zeros(
                (mid_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        # Packed uint8 representation of the second binary matrix
        # Number of bytes = ceil(out_features * mid_features / 8)
        self.register_buffer(
            "bp3",
            torch.zeros(
                ((out_features * mid_features + (8) - 1) // (8)),
                dtype=torch.uint8,
                device=dev,
            ),
        )
        # Final output scaling factor
        self.register_buffer(
            "scaling4",
            torch.zeros(
                (out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        # Bias term
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear,
        w_bit,
        init_only=False,
    ) -> "DBFLinear_NAIVE":
        """
        Class method to create a DBFLinear_NAIVE instance from a torch.nn.Linear layer.

        Args:
            linear (nn.Linear): Source nn.Linear layer.
            w_bit (int): Bit-width used to determine the feature.
            init_only (bool): If True, only allocate the skeleton without converting weights.
                              Useful for preparing the module to receive a loaded state dict.

        Returns:
            DBFLinear_NAIVE: The constructed DBFLinear_NAIVE instance.
        """
        dbf_linear = cls(
            w_bit,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return dbf_linear

        return dbf_linear

    def post_init(self):
        self.binary1 = BitLinearPacked(self.bp1, (self.mid_features, self.in_features))
        self.binary3 = BitLinearPacked(self.bp3, (self.out_features, self.mid_features))

    def forward(self, x):
        """
        Apply the DBF linear transformation to input x.
        The transformation proceeds as follows:
        1. Input scaling  (``scaling0``)
        2. First binary linear transform  (``binary1``)
        3. Intermediate scaling  (``scaling2``)
        4. Second binary linear transform  (``binary3``)
        5. Final output scaling  (``scaling4``)
        6. Bias addition (if present)

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch_size, in_features)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch_size, out_features)``.
        """
        assert hasattr(self, "binary1"), (
            "module.post_init() must be called before module.forward(). "
            "Use naive_post_init() on the whole model."
        )
        assert hasattr(self, "binary3"), (
            "module.post_init() must be called before module.forward(). "
            "Use naive_post_init() on the whole model."
        )

        x = x * self.scaling0.to(x.dtype)
        x = self.binary1(x)
        x = x * self.scaling2.to(x.dtype)
        x = self.binary3(x)
        x = x * self.scaling4.to(x.dtype)
        if self.bias is not None:
            x = x + self.bias.to(x.dtype)

        return x


def naive_post_init(model):
    for _, submodule in model.named_modules():
        if isinstance(submodule, DBFLinear_NAIVE):
            submodule.post_init()

    return model
