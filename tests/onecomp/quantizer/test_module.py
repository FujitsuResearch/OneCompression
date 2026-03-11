"""Common utilities and base specs for quantization tests.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura
"""

import random

import pytest
import torch
from torch import nn


class QuantizeTestHelper:
    """Shared utilities for quantization tests."""

    def seed_everything(self, seed: int = 42) -> None:
        """Seed Python and Torch RNGs for repeatable test runs."""
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def set_deterministic(self) -> None:
        """Force deterministic Torch behavior where supported."""
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def make_linear(
        self,
        in_features,
        out_features,
        device="cpu",
        dtype=torch.float32,
    ):
        """Create a linear layer for test use."""
        return torch.nn.Linear(
            in_features, out_features, bias=False, device=device, dtype=dtype
        )

    def make_input(
        self, batch=2, seq=3, hidden=8, device="cpu", dtype=torch.float32
    ):
        """Create random input tensors for quantization tests."""
        return torch.randn(batch, seq, hidden, device=device, dtype=dtype)


class TestModel(nn.Module):
    """Small attention+MLP stack used for error-bound checks."""
    def __init__(self, hidden_size=2048):
        super().__init__()
        self.model = nn.ModuleDict({
            'layers': nn.ModuleList([
                nn.ModuleDict({
                    'self_attn': nn.ModuleDict({
                        'q_proj': nn.Linear(hidden_size, hidden_size // 8, bias=False),
                        'k_proj': nn.Linear(hidden_size, hidden_size // 8, bias=False),
                        'v_proj': nn.Linear(hidden_size, hidden_size // 8, bias=False),
                        'o_proj': nn.Linear(hidden_size // 8, hidden_size, bias=False),
                    }),
                    'mlp': nn.ModuleDict({
                        'gate_proj': nn.Linear(hidden_size, hidden_size * 11 // 4, bias=False),
                        'up_proj': nn.Linear(hidden_size, hidden_size * 11 // 4, bias=False),
                        'down_proj': nn.Linear(hidden_size * 11 // 4, hidden_size, bias=False),
                    })
                })
                for _ in range(2)
            ])
        })

    def forward(self, x):
        """Forward pass is not needed for quantization tests."""
        pass


class BaseQuantizeSpec:
    """Base class that groups common quantization tests.

    Subclasses must implement or define the following:
        - quantizer_cls: quantizer class
        - result_cls: quantization result class
        - default_parameter_for_test: default quantization parameters
        - boundary_parameters: boundary-value parameter cases
        - abnormal_parameters: abnormal-value parameter cases
        - check_quantize_layer: validator for quantize_layer output
        - check_equal_results: equality check for result objects
        - check_quantize_error: error bound validation
        - apply_quantized_weights: apply quantized weights to a module
    """

    __test__ = False

    # Subclasses specify the quantizer and result classes.
    quantizer_cls = None
    result_cls = None
    # Subclasses specify default quantization parameters.
    default_parameter_for_test = {}
    # Subclasses specify boundary and abnormal parameter cases.
    boundary_parameters = []
    abnormal_parameters = []

    @pytest.fixture
    def helper(self):
        """Provide a shared helper instance to tests."""
        return QuantizeTestHelper()

    def make_quantizer(self, **params):
        """Instantiate the quantizer with the supplied parameters."""
        return self.quantizer_cls(**params)

    def check_quantize_layer(
        self,
        result,
        layer: torch.nn.Module,
    ):
        """Validate types, shapes, and device of quantize_layer outputs."""
        raise NotImplementedError

    def check_equal_results(self, r1, r2):
        """Validate equality of quantization result objects."""
        raise NotImplementedError

    def check_quantize_error(self, error: float, max_error: float):
        """Validate that quantization error is within tolerance."""
        raise NotImplementedError

    def apply_quantization(self, quantizer, layer, inp):
        """Quantize a layer using a Hessian derived from the input."""
        hessian = quantizer.calculate_hessian(layer, inp)
        return quantizer.quantize_layer(layer, hessian=hessian)

    def apply_quantized_weights(self, module, result, device):
        """Apply quantized weights to a module."""
        raise NotImplementedError

    def check_forward_error(
        self,
        error_original_vs_dequantized: float,
        error_dequantized_vs_applied: float,
        max_error_dequantized_vs_applied: float
    ):
        """Validate forward errors."""
        raise NotImplementedError

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_quantize_layer_returns(self, device, helper):
        """Validate types, shapes, and device of quantize_layer outputs."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        helper.set_deterministic()
        helper.seed_everything(123)

        layer = helper.make_linear(
            8, 8, device=device, dtype=torch.float32
        )
        inp = helper.make_input(device=device, dtype=torch.float32)

        q = self.make_quantizer(**self.default_parameter_for_test)
        hessian = q.calculate_hessian(layer, inp)

        result = q.quantize_layer(layer, inp, hessian=hessian)

        self.check_quantize_layer(
            result,
            layer,
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_quantize_layer_reproducibility(self, device, helper):
        """Validate reproducibility of quantize_layer.

        args:
            device (str): Device to run the test on ("cpu" or "cuda").
        """
        helper.set_deterministic()

        layer1 = helper.make_linear(
            8, 8, device=device, dtype=torch.float32
        )
        layer2 = helper.make_linear(
            8, 8, device=device, dtype=torch.float32
        )
        layer2.weight.data.copy_(layer1.weight.data)

        inp = helper.make_input(device=device, dtype=torch.float32)

        q = self.make_quantizer(**self.default_parameter_for_test)
        h1 = q.calculate_hessian(layer1, inp)
        h2 = q.calculate_hessian(layer2, inp)

        helper.seed_everything(123)
        r1 = q.quantize_layer(layer1, inp, hessian=h1)
        helper.seed_everything(123)
        r2 = q.quantize_layer(layer2, inp, hessian=h2)

        self.check_equal_results(r1, r2)

    def test_parameters_boundary(self, params, helper):
        """Validate behavior at boundary parameter values.

        args:
            params (dict): Boundary parameter set.
        """
        layer = helper.make_linear(4, 4, device="cpu", dtype=torch.float32)
        inp = helper.make_input(
            batch=1, seq=1, hidden=4, device="cpu", dtype=torch.float32
        )

        q = self.make_quantizer(**params)
        hessian = q.calculate_hessian(layer, inp)

        result = q.quantize_layer(layer, inp, hessian=hessian)

        self.check_quantize_layer(
            result,
            layer,
        )

    def test_parameters_abnormal_values_raise(self, params):
        """Validate that abnormal parameter values raise exceptions.

        Args:
            params (dict): Abnormal parameter set.
        """
        q = self.make_quantizer(**params)
        # any model is sufficient
        # since we expect parameter validation to fail before quantization logic runs
        model = nn.Sequential(nn.Linear(4, 4, bias=False))

        with pytest.raises(Exception):
            q.setup(model)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cpu_gpu_output_match(self, helper):
        """Validate that CPU and GPU quantization results match."""
        cpu_layer = helper.make_linear(8, 8, device="cpu", dtype=torch.float32)
        gpu_layer = helper.make_linear(
            8, 8, device="cuda", dtype=torch.float32
        )
        gpu_layer.weight.data.copy_(cpu_layer.weight.data.to("cuda"))

        cpu_inp = helper.make_input(device="cpu", dtype=torch.float32)
        gpu_inp = cpu_inp.to("cuda")

        q = self.make_quantizer(**self.default_parameter_for_test)
        cpu_hessian = q.calculate_hessian(cpu_layer, cpu_inp)
        gpu_hessian = q.calculate_hessian(gpu_layer, gpu_inp)

        cpu_out = q.quantize_layer(
            cpu_layer, cpu_inp, hessian=cpu_hessian
        ).dequantized_weight
        gpu_out = q.quantize_layer(
            gpu_layer, gpu_inp, hessian=gpu_hessian
        ).dequantized_weight.cpu()

        assert torch.allclose(cpu_out, gpu_out, rtol=1, atol=1)

    def test_quantize_error(self, helper):
        """Validate that quantization error is within tolerance."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        quantizer = self.make_quantizer()

        hidden_size = 2048
        model = TestModel(hidden_size).to(device)
        inp = helper.make_input(
            batch=1,
            seq=4,
            hidden=hidden_size,
            device=device,
            dtype=torch.float32,
        )

        with torch.no_grad():
            y_original = inp
            for layer in model.model.layers:
                q = layer.self_attn.q_proj(y_original)
                k = layer.self_attn.k_proj(y_original)
                v = layer.self_attn.v_proj(y_original)
                head_dim = k.shape[-1]
                qk = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
                attn_weights = torch.nn.functional.softmax(qk, dim=-1)
                v = torch.matmul(attn_weights, v)
                attn_out = layer.self_attn.o_proj(v)

                gate = torch.nn.functional.silu(
                    layer.mlp.gate_proj(y_original)
                )
                up = layer.mlp.up_proj(y_original)
                mlp_out = layer.mlp.down_proj(gate * up)

                y_original = y_original + attn_out + mlp_out

        for layer in model.model.layers:
            for _, module_dict in [
                ("self_attn", layer.self_attn),
                ("mlp", layer.mlp),
            ]:
                for _, module in module_dict.items():
                    if isinstance(module, nn.Linear):
                        # make input for hessian calculation
                        module_inp = helper.make_input(
                            batch=1,
                            seq=4,
                            hidden=module.in_features,
                            device=device,
                            dtype=torch.float32,
                        )
                        H = quantizer.calculate_hessian(module, module_inp)
                        dbf_result = quantizer.quantize_layer(
                            module, module_inp, H
                        )
                        self.apply_quantized_weights(
                            module, dbf_result, device
                        )

        model = model.to(device)

        with torch.no_grad():
            y_replaced = inp
            for layer in model.model.layers:
                q = layer.self_attn.q_proj(y_original)
                k = layer.self_attn.k_proj(y_original)
                v = layer.self_attn.v_proj(y_original)
                head_dim = k.shape[-1]
                qk = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
                attn_weights = torch.nn.functional.softmax(qk, dim=-1)
                v = torch.matmul(attn_weights, v)
                attn_out = layer.self_attn.o_proj(v)

                gate = torch.nn.functional.silu(
                    layer.mlp.gate_proj(y_replaced)
                )
                up = layer.mlp.up_proj(y_replaced)
                mlp_out = layer.mlp.down_proj(gate * up)

                y_replaced = y_replaced + attn_out + mlp_out

        error = torch.norm(y_original - y_replaced) / torch.norm(y_original)
        max_error = torch.abs(y_original - y_replaced).max().item()

        self.check_quantize_error(error, max_error)

    def test_forward_error(self, helper):
        """Validate forward error."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        helper.set_deterministic()
        helper.seed_everything(123)

        # Prepare a linear layer and input
        layer = helper.make_linear(
            8, 8, device=device, dtype=torch.float32
        )
        inp = helper.make_input(device=device, dtype=torch.float32)

        # original output
        with torch.no_grad():
            y_original = layer(inp)

        # Quantize the layer
        q = self.make_quantizer(**self.default_parameter_for_test)
        hessian = q.calculate_hessian(layer, inp)
        result = q.quantize_layer(layer, inp, hessian=hessian)

        dequantized_layer = helper.make_linear(
            8, 8, device=device, dtype=torch.float32
        )
        dequantized_layer.weight.data.copy_(result.dequantized_weight.to(device))
        with torch.no_grad():
            y_dequantized = dequantized_layer(inp)

        # Apply quantized weights to the original layer for inference.
        dbl = q.create_inference_layer(
            result=result,
            linear_module=layer,
            use_gemlite=False,
        )

        # Run the forward pass with quantized weights
        with torch.no_grad():
            y_applied = dbl(inp.to(torch.float16)).to(torch.float32)

        # original_vs_dequantized
        error_original_vs_dequantized = (
            torch.norm(y_original - y_dequantized) / torch.norm(y_original)
            ).item()
        # dequantized_vs_applied
        error_dequantized_vs_applied = (
            torch.norm(y_dequantized - y_applied)/ (torch.norm(y_dequantized))
            ).item()
        # dequantized_vs_applied (max)
        max_error_dequantized_vs_applied = (
            torch.abs(y_dequantized - y_applied).max().item()
            )

        self.check_forward_error(
            error_original_vs_dequantized,
            error_dequantized_vs_applied,
            max_error_dequantized_vs_applied
        )
