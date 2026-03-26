"""Copyright 2025-2026 Fujitsu Ltd."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from gemlite.core import GemLiteLinearTriton

    _HAS_GEMLITE = True
except (ImportError, AttributeError):
    _HAS_GEMLITE = False

pytestmark = pytest.mark.skipif(
    not _HAS_GEMLITE or not torch.cuda.is_available(),
    reason="GemLite unavailable or CUDA not available",
)

if _HAS_GEMLITE:
    from vllm_plugins.dbf.modules.gemlite_linear import (
        GROUP_SIZE,
        pad_to_multiple,
        pad_cols_to_multiple,
        unpack_sign_bits,
        get_gemlite_linear,
        DBFLinear_GEMLITE,
        gemlitelinear_post_init,
    )

# Global device configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def device():
    return DEVICE


# --- Helper functions tests ---


def test_pad_to_multiple():
    tensor = torch.randn(3, 7, 13)
    multiple = 4
    padded_tensor = pad_to_multiple(tensor, multiple)

    assert padded_tensor.shape == (4, 8, 16)
    assert torch.equal(padded_tensor[0:3, 0:7, 0:13], tensor)
    assert torch.all(padded_tensor[3, :, :] == 0)
    assert torch.all(padded_tensor[:, 7, :] == 0)
    assert torch.all(padded_tensor[:, :, 13] == 0)


def test_pad_cols_to_multiple():
    t_no_pad = torch.randn(5, GROUP_SIZE * 2)
    t_padded_no_change = pad_cols_to_multiple(t_no_pad, GROUP_SIZE, 1)
    assert t_padded_no_change.shape == t_no_pad.shape
    assert torch.equal(t_padded_no_change, t_no_pad)

    t_needs_pad = torch.randn(5, GROUP_SIZE * 2 + 10)
    t_padded_changed = pad_cols_to_multiple(t_needs_pad, GROUP_SIZE, 1)
    expected_cols = ((GROUP_SIZE * 2 + 10 + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    assert t_padded_changed.shape[1] == expected_cols
    assert torch.equal(t_padded_changed[:, : t_needs_pad.shape[1]], t_needs_pad)
    assert torch.all(t_padded_changed[:, t_needs_pad.shape[1] :] == 1)

    t_ndim_not_2 = torch.randn(2, 3, 4)
    t_unchanged = pad_cols_to_multiple(t_ndim_not_2, GROUP_SIZE, 1)
    assert torch.equal(t_unchanged, t_ndim_not_2)


def test_unpack_sign_bits(device):
    # original_shape: (4, 16) -> 64 bits = 8 bytes
    original_shape = (4, 16)
    # create a tensor with +1 and -1 values
    original_data = (
        torch.randint(0, 2, original_shape, dtype=torch.int8, device=device) * 2 - 1
    )  # +1 or -1

    # Pack it manually for testing unpack_sign_bits
    # Here we simulate packing by converting +/-1 to 0/1, then into uint8
    # This is inverse of `fp16_tensor = (unpacked_bits.to(torch.int8)) * 2 - 1`
    # and `unpacked_bits = ((expanded_int8 >> shifts) & 1).to(dtype)`
    # Since original_data is +/-1, we map it to 0/1
    packed_bits_01 = (original_data + 1) // 2  # Convert -1 to 0, 1 to 1

    # Simulate packing into uint8. This is a bit tricky as the original packing
    # is `(x >> 7, >> 6, ..., >> 0)`. So we need to reverse this.
    # The `unpack_sign_bits` function extracts bits from MSB to LSB.
    # Let's create a uint8 tensor bit by bit.
    # We need to reshape packed_bits_01 to (num_bytes, 8) and then combine bits.
    flattened_bits = packed_bits_01.flatten()
    num_elements = flattened_bits.numel()
    num_bytes_needed = (num_elements + 7) // 8

    # Pad with zeros if not a multiple of 8
    padded_bits = F.pad(
        flattened_bits, (0, num_bytes_needed * 8 - num_elements), mode="constant", value=0
    )

    # Reshape for packing into bytes, assuming 8 bits per byte
    # For `unpack_sign_bits`, it unpacks with shifts 7,6,...,0.
    # So `padded_bits[i*8+0]` corresponds to bit 7, `padded_bits[i*8+1]` to bit 6, etc.
    packed_byte_data = torch.zeros(num_bytes_needed, dtype=torch.uint8, device=device)
    for i in range(8):
        packed_byte_data |= (padded_bits.view(-1, 8)[:, i] << (7 - i)).to(torch.uint8)

    # Now call the function to test
    unpacked_data = unpack_sign_bits(packed_byte_data, original_shape)

    assert unpacked_data.shape == original_shape
    assert torch.equal(unpacked_data.to(torch.int8), original_data.to(torch.int8))


# --- get_gemlite_linear tests ---


def test_get_gemlite_linear_padded_for_unaligned_in_features(device):
    # in_features not a multiple of GROUP_SIZE — now handled via padding
    out_raw, in_raw = 64, GROUP_SIZE + 1
    weights = torch.randn(out_raw, in_raw, device=device)
    gemlite_linear = get_gemlite_linear(weights)
    assert isinstance(gemlite_linear, GemLiteLinearTriton)
    expected_in = ((in_raw + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    expected_out = ((out_raw + GROUP_SIZE - 1) // GROUP_SIZE) * GROUP_SIZE
    assert gemlite_linear.in_features == expected_in
    assert gemlite_linear.out_features == expected_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_get_gemlite_linear_valid_output(device):
    out_features = GROUP_SIZE  # Use GROUP_SIZE multiple to avoid padding
    in_features = GROUP_SIZE * 2  # Multiple of GROUP_SIZE
    weights = torch.randn(out_features, in_features, device=device)

    gemlite_linear = get_gemlite_linear(weights)

    assert isinstance(gemlite_linear, GemLiteLinearTriton)

    # Dimensions are already GROUP_SIZE-aligned, so no padding expected
    assert gemlite_linear.in_features == in_features
    assert gemlite_linear.out_features == out_features


# --- DBFLinear_GEMLITE tests ---


def test_dbflinear_gemlite_init(device):
    in_features = 256  # multiple of 8 and ideally GROUP_SIZE for later ops
    out_features = 128
    w_bit = 1
    has_bias = True

    model = DBFLinear_GEMLITE(w_bit, in_features, out_features, has_bias, device)

    assert model.in_features == in_features
    assert model.out_features == out_features
    assert model.w_bit == w_bit
    assert model.training is False
    assert model.mid_features > 0  # check if calculated correctly
    assert hasattr(model, "scaling0")
    assert hasattr(model, "bp1")
    assert hasattr(model, "scaling2")
    assert hasattr(model, "bp3")
    assert hasattr(model, "scaling4")
    assert hasattr(model, "bias")
    assert model.bias is not None
    assert model.scaling0.device == device
    assert model.bp1.device == device
    assert model.bias.device == device

    model_no_bias = DBFLinear_GEMLITE(w_bit, in_features, out_features, False, device)
    assert model_no_bias.bias is None


def test_dbflinear_gemlite_from_linear(device):
    in_features = 256
    out_features = 128
    w_bit = 1
    linear_layer = nn.Linear(in_features, out_features, bias=True).to(device)

    dbf_linear_init_only = DBFLinear_GEMLITE.from_linear(linear_layer, w_bit, init_only=True)
    assert dbf_linear_init_only.in_features == in_features
    assert dbf_linear_init_only.out_features == out_features
    assert dbf_linear_init_only.bias is not None

    dbf_linear_full = DBFLinear_GEMLITE.from_linear(linear_layer, w_bit, init_only=False)
    assert dbf_linear_full.in_features == in_features
    assert dbf_linear_full.out_features == out_features


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for GemLiteLinearTriton")
def test_dbflinear_gemlite_post_init(device):
    in_features = GROUP_SIZE * 8
    out_features = GROUP_SIZE * 4
    w_bit = 1
    model = DBFLinear_GEMLITE(w_bit, in_features, out_features, True, device)

    # Simulate some initial data in bp1 and bp3 for unpacking
    model.bp1.data = torch.randint(0, 256, model.bp1.shape, dtype=torch.uint8, device=device)
    model.bp3.data = torch.randint(0, 256, model.bp3.shape, dtype=torch.uint8, device=device)

    model.post_init()

    assert hasattr(model, "binary1")
    assert isinstance(model.binary1, GemLiteLinearTriton)
    assert hasattr(model, "binary3")
    assert isinstance(model.binary3, GemLiteLinearTriton)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for GemLiteLinearTriton")
def test_dbflinear_gemlite_forward(device):
    in_features = GROUP_SIZE * 8
    out_features = GROUP_SIZE * 4
    w_bit = 1
    model = DBFLinear_GEMLITE(w_bit, in_features, out_features, True, device)

    # Initialize some dummy data for buffers
    model.scaling0.data = torch.rand(in_features, dtype=torch.float16, device=device) * 2 - 1
    model.scaling2.data = (
        torch.rand(model.mid_features, dtype=torch.float16, device=device) * 2 - 1
    )
    model.scaling4.data = torch.rand(out_features, dtype=torch.float16, device=device) * 2 - 1
    model.bias.data = torch.rand(out_features, dtype=torch.float16, device=device) * 2 - 1

    # Populate bp1 and bp3 with some data that can be unpacked to +/- 1
    # For w_bit=1, it means binary weights -1 or +1

    model.bp1.data = torch.randint(0, 256, model.bp1.shape, dtype=torch.uint8, device=device)
    model.bp3.data = torch.randint(0, 256, model.bp3.shape, dtype=torch.uint8, device=device)

    model.post_init()

    batch_size = 4
    x = torch.randn(batch_size, in_features, dtype=torch.float32, device=device)

    output = model(x)

    assert output.shape == (batch_size, out_features)
    assert output.dtype == torch.float32  # Input dtype should be preserved


def test_gemlitelinear_post_init_model_traversal(device):
    class MyModel(nn.Module):
        def __init__(self, in_f, out_f, w_b):
            super().__init__()
            self.linear1 = nn.Linear(in_f, out_f, bias=True).to(device)
            self.dbf_linear1 = DBFLinear_GEMLITE(w_b, in_f, out_f, True, device)
            self.linear2 = nn.Linear(out_f, in_f, bias=False).to(device)
            self.dbf_linear2 = DBFLinear_GEMLITE(w_b, in_f, out_f, False, device)

    in_f = GROUP_SIZE * 8
    out_f = GROUP_SIZE * 4
    w_b = 1
    model = MyModel(in_f, out_f, w_b)

    # Manually populate bp1 and bp3 for DBFLinear_GEMLITE instances
    # to ensure post_init can run without errors
    for name, m in model.named_modules():
        if isinstance(m, DBFLinear_GEMLITE):
            m.scaling0.data = torch.rand(m.in_features, dtype=torch.float16, device=device)
            m.scaling2.data = torch.rand(m.mid_features, dtype=torch.float16, device=device)
            m.scaling4.data = torch.rand(m.out_features, dtype=torch.float16, device=device)
            if m.bias is not None:
                m.bias.data = torch.rand(m.out_features, dtype=torch.float16, device=device)
            m.bp1.data = torch.randint(0, 256, m.bp1.shape, dtype=torch.uint8, device=device)
            m.bp3.data = torch.randint(0, 256, m.bp3.shape, dtype=torch.uint8, device=device)

    model_inited = gemlitelinear_post_init(model)

    assert isinstance(model_inited, MyModel)
    assert isinstance(model_inited.dbf_linear1.binary1, GemLiteLinearTriton)
    assert isinstance(model_inited.dbf_linear1.binary3, GemLiteLinearTriton)
    assert isinstance(model_inited.dbf_linear2.binary1, GemLiteLinearTriton)
    assert isinstance(model_inited.dbf_linear2.binary3, GemLiteLinearTriton)

    # Ensure other linear layers are untouched
    assert isinstance(model_inited.linear1, nn.Linear)
    assert isinstance(model_inited.linear2, nn.Linear)
