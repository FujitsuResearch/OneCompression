"""Copyright 2025-2026 Fujitsu Ltd."""

import pytest
import torch
import torch.nn as nn
import re
import logging

logger = logging.getLogger(__name__)
from vllm_plugins.dbf.modules.naive import (
    unpack_sign_bits,
    naive_post_init,
    BitLinearPacked,
    DBFLinear_NAIVE,
)


class TestUnpackSignBits:
    """
    Tests for the `unpack_sign_bits` function.
    - test_basic_unpacking: verifies expansion from one int8 value into 8 elements
    - test_multiple_elemets_and_shape: verifies expansion from two int8 values into a 2x8 tensor
    - test_empty_input: verifies behavior for an empty input tensor
    - test_device_preservation: verifies device preservation
    """

    def test_basic_unpacking(self):
        # uint8 representing 0b10101010 (0xA0 = 1600)
        # It should unpack to 1, -1, 1, -1, 1, -1, 1, -1
        packed = torch.tensor([0b10101010], dtype=torch.uint8)  # 170
        original_shape = torch.Size([1, 8])
        unpacked = unpack_sign_bits(packed, original_shape)
        expected = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]], dtype=torch.int8)

        assert torch.equal(unpacked, expected)
        assert unpacked.shape == original_shape
        assert unpacked.dtype == torch.int8

    def test_multiple_elements_and_shapes(self):
        # 0b11001100 (C) -> 1, 1, -1, -1, 1, 1, -1, -1
        # 0b00110011 (33) -> -1, -1, 1, 1, -1, -1, 1, 1
        packed = torch.tensor([0b11001100, 0b00110011], dtype=torch.uint8)
        original_shape = torch.Size([2, 8])
        unpacked = unpack_sign_bits(packed, original_shape)

        expected = torch.tensor(
            [
                [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            ],
            dtype=torch.int8,
        )

        assert torch.equal(unpacked, expected)
        assert unpacked.shape == original_shape

    # def test_empty_input(self):
    #    packed = torch.tensor([], dtype=torch.uint8)
    #    original_shape = torch.Size([0, 0])
    #    unpacked = unpack_sign_bits(packed, original_shape)
    #    assert unpacked.shape == original_shape
    #    assert unpacked.numel() == 0

    def test_device_preservation(self):
        if torch.cuda.is_available():
            packed_cpu = torch.tensor([0b10101010], dtype=torch.uint8, device="cpu")
            unpacked_cpu = unpack_sign_bits(packed_cpu, torch.Size([1, 8]))
            assert unpacked_cpu.device.type == "cpu"

            packed_cuda = torch.tensor([0b10101010], dtype=torch.uint8, device="cuda")
            unpacked_cuda = unpack_sign_bits(packed_cuda, torch.Size([1, 8]))
            assert unpacked_cuda.device.type == "cuda"
        else:
            logger.info("CUDA not available, skipping device test.")


class TestBitLinearPacked:
    """
    Tests for the `BitLinearPacked` class.
    - test_init_preunpack_true: initialization when the binary matrix is unpacked and cached up front
    - test_init_preunpack_false: initialization when unpacking happens on each forward pass
    - test_forward_preunpack_true: forward path with pre-unpacked cached weights
    - test_forward_preunpack_false: forward path with unpacking on each forward pass
    """

    def test_init_preunpack_true(self):
        packed_binary = torch.tensor([0b10101010], dtype=torch.uint8)
        shape = (1, 8)
        layer = BitLinearPacked(packed_binary=packed_binary, shape=shape, preunpack=True)

        assert hasattr(layer, "bit_mat")
        assert layer.bit_mat is not None
        expected_bit_mat = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]], dtype=torch.int8
        )
        assert torch.equal(layer.bit_mat, expected_bit_mat)
        assert layer.bit_mat.dtype == torch.int8
        assert tuple(layer.shape) == shape

    def test_init_preunpack_false(self):
        acked_binary = torch.tensor([0b10101010], dtype=torch.uint8)
        shape = (1, 8)
        layer = BitLinearPacked(packed_binary=acked_binary, shape=shape, preunpack=False)

        assert not hasattr(layer, "bit_mat") or layer.bit_mat is None
        assert torch.equal(layer.bp, acked_binary)
        assert tuple(layer.shape) == shape

    def test_forward_preunpack_true(self):
        packed_binary = torch.tensor([0b10101010, 0b01010101], dtype=torch.uint8)  # 2x8=16 bits
        shape = (2, 8)
        layer = BitLinearPacked(packed_binary=packed_binary, shape=shape, preunpack=True)

        input_tensor = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                # batch_size=2, in_features=8
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )

        # bit_mat:
        # [1., -1., 1., -1., 1., -1., 1., -1.]
        # [-1., 1., -1., 1., -1., 1., -1., 1.]
        #
        # weight_matrix (bit_mat.t()):
        # [[ 1.,  -1.],
        #  [-1.,   1.],
        #  [ 1.,  -1.],
        #  [-1.,   1.],
        #  [ 1.,  -1.],
        #  [-1.,   1.],
        #  [ 1.,  -1.],
        #  [-1.,   1.]]

        # input_tensor.matmul(weight_matrix)
        # row 0: (1-2+3-4+5-6+7-8), (-1+2-3+4-5+6-7+8) -> -4, 4
        # row 1: (8-7+6-5+4-3+2-1), (-8+7-6+5-4+3-2+1) ->  4,-4

        output = layer.forward(input_tensor)
        expected_output = torch.tensor([[-4.0, 4.0], [4.0, -4.0]], dtype=torch.float32)

        assert torch.allclose(output, expected_output)
        # (batch_size, out_features)
        assert output.shape == (input_tensor.shape[0], shape[0])

    def test_forward_preunpack_false(self):
        packed_binary = torch.tensor([0b10101010, 0b01010101], dtype=torch.uint8)  # 2x8=16 bits
        shape = (2, 8)
        layer = BitLinearPacked(packed_binary=packed_binary, shape=shape, preunpack=False)

        input_tensor = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                # batch_size=2, in_features=8
                [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )

        # bit_mat:
        # [1., -1., 1., -1., 1., -1., 1., -1.]
        # [-1., 1., -1., 1., -1., 1., -1., 1.]
        #
        # weight_matrix (bit_mat.t()):
        # [[ 1.,  -1.],
        #  [-1.,   1.],
        #  [ 1.,  -1.],
        #  [-1.,   1.],
        #  [ 1.,  -1.],
        #  [-1.,   1.],
        #  [ 1.,  -1.],
        #  [-1.,   1.]]

        # input_tensor.matmul(weight_matrix)
        # row 0: (1-2+3-4+5-6+7-8), (-1+2-3+4-5+6-7+8) -> -4, 4
        # row 1: (8-7+6-5+4-3+2-1), (-8+7-6+5-4+3-2+1) ->  4,-4

        output = layer.forward(input_tensor)
        expected_output = torch.tensor([[-4.0, 4.0], [4.0, -4.0]], dtype=torch.float32)

        assert torch.allclose(output, expected_output)
        # (batch_size, out_features)
        assert output.shape == (input_tensor.shape[0], shape[0])


class TestDBFLinearNAIVE:
    """
    Tests for the `DBFLinear_NAIVE` class.
    - test_init: initialization test
    - test_from_linear: test for the `from_linear` function
    - test_post_init: test for the `post_init` function
    - test_forward: test for the `forward` function
    - test_forward_pre_init_failure: verifies `AssertionError` behavior
    """

    def setup_method(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Device formatting differs between CUDA and CPU; compare string forms.
        # CUDA: torch.device('cuda', 0) -> 'cuda:0'
        # CPU: torch.device('cpu') -> 'cpu'
        self.device_str = str(self.device)

    def test_init(self):
        in_features = 16
        out_features = 8
        w_bit = 1
        bias = True

        linear = DBFLinear_NAIVE(w_bit, in_features, out_features, bias, self.device)

        assert linear.in_features == in_features
        assert linear.out_features == out_features
        assert linear.w_bit == w_bit
        assert linear.bias is not None

        mid_features1 = int(w_bit * (in_features * out_features) / (in_features + out_features))
        mid_features_expected = min(min(in_features, out_features), mid_features1)
        assert linear.mid_features == mid_features_expected

        assert linear.scaling0.shape == (in_features,)
        assert linear.bp1.shape[0] == (linear.mid_features * in_features + 7) // 8
        assert linear.scaling2.shape == (linear.mid_features,)
        assert linear.bp3.shape[0] == (out_features * linear.mid_features + 7) // 8
        assert linear.scaling4.shape == (out_features,)
        assert linear.bias.shape == (out_features,)

        for buffer_name in ["scaling0", "bp1", "scaling2", "bp3", "scaling4"]:
            buffer = getattr(linear, buffer_name)
            # assert buffer.device == self.device
            assert str(buffer.device) == self.device_str
            if "scaling" in buffer_name:
                assert buffer.dtype == torch.float16
            elif "bp" in buffer_name:
                assert buffer.dtype == torch.uint8
        # assert linear.bias.device == self.device
        assert str(linear.bias.device) == self.device_str
        assert linear.bias.dtype == torch.float16

    def test_from_linear(self):
        in_features = 16
        out_features = 8
        w_bit = 1

        # Dummy nn.Linear module
        linear_module = nn.Linear(in_features, out_features, bias=True).to(self.device)

        dbf_linear_init_only = DBFLinear_NAIVE.from_linear(linear_module, w_bit, init_only=True)
        assert isinstance(dbf_linear_init_only, DBFLinear_NAIVE)
        assert dbf_linear_init_only.in_features == in_features
        assert dbf_linear_init_only.out_features == out_features
        assert dbf_linear_init_only.bias is not None

        dbf_linear = DBFLinear_NAIVE.from_linear(linear_module, w_bit, init_only=False)
        assert isinstance(dbf_linear, DBFLinear_NAIVE)
        # from_linear does not currently copy parameter values, so no further state checks are needed

    def test_post_init(self):
        in_features = 16
        out_features = 8
        w_bit = 1

        linear = DBFLinear_NAIVE(w_bit, in_features, out_features, True, self.device)

        assert not hasattr(linear, "binary1")
        assert not hasattr(linear, "binary3")

        linear.post_init()

        assert hasattr(linear, "binary1")
        assert isinstance(linear.binary1, BitLinearPacked)
        assert linear.binary1.bp.shape == linear.bp1.shape
        assert tuple(linear.binary1.shape) == (linear.mid_features, in_features)

        assert hasattr(linear, "binary3")
        assert isinstance(linear.binary3, BitLinearPacked)
        assert linear.binary3.bp.shape == linear.bp3.shape
        assert tuple(linear.binary3.shape) == (out_features, linear.mid_features)

    def test_forward(self):
        in_features = 16
        out_features = 8
        # mid_features will be min(16, 8, int(1 * 16*8/(16+8))) = min(16, 8, 5) = 5
        w_bit = 1
        batch_size = 4

        dbf_linear = DBFLinear_NAIVE(w_bit, in_features, out_features, True, self.device)
        dbf_linear.post_init()

        # Initialize buffers with dummy data
        dbf_linear.scaling0.data.fill_(0.5)
        dbf_linear.scaling2.data.fill_(1.0)
        dbf_linear.scaling4.data.fill_(2.0)
        dbf_linear.bias.data.fill_(0.1)

        # Set meaningful values for bp1 and bp3 (simple case)
        # binary1 (mid_features, in_features) -> (5, 16)
        # binary3 (out_features, mid_features) -> (8, 5)

        # Set all bits to 1 (they unpack to 1)
        dbf_linear.bp1.data.fill_(0b11111111)
        dbf_linear.bp3.data.fill_(0b11111111)

        # Rebuild BitLinearPacked modules to reflect the updated packed values
        dbf_linear.binary1 = BitLinearPacked(
            dbf_linear.bp1, (dbf_linear.mid_features, in_features)
        )
        dbf_linear.binary3 = BitLinearPacked(
            dbf_linear.bp3, (out_features, dbf_linear.mid_features)
        )

        input_tensor = torch.randn(batch_size, in_features, dtype=torch.float32).to(self.device)

        output = dbf_linear.forward(input_tensor)

        # Expected output computed manually
        step1 = input_tensor * dbf_linear.scaling0.to(torch.float32)  # (4, 16) * (16)

        # Weight matrix from binary1: every element in (5, 16) is -1
        weight1 = (
            torch.full(
                (dbf_linear.mid_features, in_features),
                -1.0,
                # (16, 5)
                dtype=torch.float32,
            )
            .to(self.device)
            .t()
        )
        step2 = step1.matmul(weight1)  # (4, 16) @ (16, 5) = (4, 5)

        step3 = step2 * dbf_linear.scaling2.to(torch.float32)  # (4, 5) * (5)

        # Weight matrix from binary3: every element in (8, 5) is -1
        weight3 = (
            torch.full((out_features, dbf_linear.mid_features), -1.0, dtype=torch.float32)
            .to(self.device)
            .t()
        )  # (5, 8)
        step4 = step3.matmul(weight3)  # (4, 5) @ (5, 8) = (4, 8)

        step5 = step4 * dbf_linear.scaling4.to(torch.float32)  # (4, 8) * (8)
        expected_output = step5 + dbf_linear.bias.to(torch.float32)  # (4, 8) + (8)

        assert torch.allclose(output, expected_output, atol=1e-3, rtol=1e-3)
        assert output.shape == (batch_size, out_features)
        assert output.dtype == input_tensor.dtype  # Verify that the input dtype is preserved

    def test_forward_pre_init_failure(self):
        in_features = 16
        out_features = 8
        w_bit = 1
        dbf_linear = DBFLinear_NAIVE(w_bit, in_features, out_features, False, self.device)
        input_tensor = torch.randn(4, in_features, dtype=torch.float32).to(self.device)

        with pytest.raises(
            AssertionError,
            match=re.escape(
                "module.post_init() must be called before module.forward(). Use naive_post_init() on the whole model."
            ),
        ):
            dbf_linear.forward(input_tensor)


class DummyModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.linear1 = nn.Linear(32, 16)
        self.dbf_linear1 = DBFLinear_NAIVE(1, 32, 16, True, device)
        self.linear2 = nn.Linear(16, 8)
        self.dbf_linear2 = DBFLinear_NAIVE(1, 16, 8, False, device)
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        pass  # Forward is not used in these tests


class TestNaivePostInit:
    """
    Tests for the `naive_post_init` function.
    """

    def setup_method(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_naive_post_init(self):
        model = DummyModel(self.device)

        # Before calling post_init, the binary1 and binary3 should not exist
        assert not hasattr(model.dbf_linear1, "binary1")
        assert not hasattr(model.dbf_linear1, "binary3")
        assert not hasattr(model.dbf_linear2, "binary1")
        assert not hasattr(model.dbf_linear2, "binary3")

        naive_post_init(model)

        # After calling post_init, the binary1 and binary3 should exist and initialized
        assert hasattr(model.dbf_linear1, "binary1")
        assert isinstance(model.dbf_linear1.binary1, BitLinearPacked)
        assert hasattr(model.dbf_linear1, "binary3")
        assert isinstance(model.dbf_linear1.binary3, BitLinearPacked)

        assert hasattr(model.dbf_linear2, "binary1")
        assert isinstance(model.dbf_linear2.binary1, BitLinearPacked)
        assert hasattr(model.dbf_linear2, "binary3")
        assert isinstance(model.dbf_linear2.binary3, BitLinearPacked)

        # Ensure other modules remain untouched
        assert not hasattr(model.linear1, "binary1")
        assert not hasattr(model.conv, "binary1")
