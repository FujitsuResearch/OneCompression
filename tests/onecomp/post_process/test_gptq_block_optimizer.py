"""Regression tests for blockwise GPTQ zero restoration.

These tests intentionally stay narrow: they only pin the `_get_float_zeros`
logic added for GPTQ v1 qzero=0 handling in the blockwise PTQ path.
"""

import pytest
import torch
import torch.nn as nn

from onecomp.post_process._blockwise.gptq_block_optimizer import (
    _get_float_zeros,
    optimize_gptq_block,
)
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear, pack_zeros


def _make_layer(*, pack_weights: bool, wbits: int = 4, device: str = "cpu") -> GPTQLinear:
    in_features = 32
    out_features = 32
    groupsize = 32

    qweight = torch.zeros((out_features, in_features), dtype=torch.int32, device=device)
    scales = torch.ones((1, out_features), dtype=torch.float32, device=device)
    zeros = torch.ones((1, out_features), dtype=torch.float32, device=device)
    zeros[:, :4] = 0.0

    return GPTQLinear(
        in_features=in_features,
        out_features=out_features,
        wbits=wbits,
        groupsize=groupsize,
        actorder=False,
        quantized_weight=qweight,
        scale=scales,
        zero=zeros,
        bias=None,
        device=device,
        pack_weights=pack_weights,
        use_gemlite=False,
    )


class _TinyBlock(nn.Module):
    def __init__(self, proj: GPTQLinear):
        super().__init__()
        self.proj = proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_get_float_zeros_restores_packed_v1_qzero_zero():
    """Packed GPTQ v1 qzero=0 must wrap back to 0 in blockwise PTQ.

    This is the bug-fixing path: packed v1 qzeros unpack to ``2^wbits - 1``
    for raw qzero=0, so the restoration must apply modular wrap after ``+1``.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = _make_layer(pack_weights=True, device=device)

    zeros = _get_float_zeros(layer)

    assert torch.all(zeros[0, :4] == 0), (
        f"expected bug columns to restore to 0, "
        f"got {zeros[0, :8].tolist()}"
    )
    assert torch.all(zeros[0, 4:] == 1), (
        f"expected control columns to stay at 1, "
        f"got sample {zeros[0, 4:8].tolist()}"
    )


def test_get_float_zeros_keeps_unpacked_v1_behavior():
    """Unpacked GPTQ v1 remains correct after the modular-wrap change.

    In the unpacked path qzeros are stored as signed INT32 values, so raw
    qzero=0 is held as ``-1`` and already returns to 0 under plain ``+1``.
    This test is a non-regression guard, not a reproduction of the packed bug.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer = _make_layer(pack_weights=False, device=device)

    zeros = _get_float_zeros(layer)

    assert torch.all(zeros[0, :4] == 0), f"expected bug columns to stay at 0, got {zeros[0, :8].tolist()}"
    assert torch.all(zeros[0, 4:] == 1), (
        f"expected control columns to stay at 1, got sample {zeros[0, 4:8].tolist()}"
    )


@pytest.mark.parametrize("pack_weights", [True, False])
def test_get_float_zeros_leaves_v2_values_unshifted(pack_weights):
    """The modular +1 restoration must not run on gptq_v2 tensors."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wbits = 4

    layer = _make_layer(pack_weights=pack_weights, wbits=wbits, device=device)
    zeros_raw = torch.full((1, layer.out_features), 3, dtype=torch.int32, device=device)
    zeros_raw[:, :4] = 0

    layer.checkpoint_format = "gptq_v2"
    if pack_weights:
        layer.qzeros.copy_(pack_zeros(zeros_raw, wbits).to(layer.qzeros.device))
    else:
        layer.qzeros.copy_(zeros_raw.to(layer.qzeros.device))

    zeros = _get_float_zeros(layer)

    assert torch.equal(zeros.to(torch.int32), zeros_raw), (
        f"pack_weights={pack_weights}: expected v2 zeros to remain unshifted, "
        f"got {zeros[0, :8].tolist()}"
    )


@pytest.mark.parametrize("pack_weights", [True, False])
def test_optimize_gptq_block_smoke_preserves_qzero_zero(pack_weights):
    """Smoke test the blockwise PTQ entry point on a tiny GPTQLinear wrapper."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proj = _make_layer(pack_weights=pack_weights, device=device)
    block = _TinyBlock(proj).to(device)

    x = torch.zeros((proj.in_features,), dtype=torch.float16, device=device)
    x[0] = 1.0
    target = block(x.unsqueeze(0)).detach()

    initial_error, final_error = optimize_gptq_block(
        block,
        inps=[x],
        target_outputs=[target],
        layer_kwargs={},
        lr=1e-3,
        epochs=1,
        dev=torch.device(device),
    )

    out = block(x.unsqueeze(0)).float().squeeze(0)
    expected_bug = torch.zeros(4, dtype=torch.float32, device=out.device)
    expected_ctrl = -torch.ones(4, dtype=torch.float32, device=out.device)

    assert initial_error == pytest.approx(0.0, abs=1e-8)
    assert final_error == pytest.approx(0.0, abs=1e-8)
    assert torch.allclose(out[:4], expected_bug), (
        f"pack_weights={pack_weights}: bug columns drifted after optimize_gptq_block, "
        f"got {out[:8].tolist()}"
    )
    assert torch.allclose(out[4:8], expected_ctrl), (
        f"pack_weights={pack_weights}: control columns drifted after optimize_gptq_block, "
        f"got {out[:8].tolist()}"
    )
    assert not hasattr(proj, "_opt_scales")
    assert not hasattr(proj, "_opt_zeros")