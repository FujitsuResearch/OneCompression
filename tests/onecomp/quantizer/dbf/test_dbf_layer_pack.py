"""Regression tests for DBF binary-factor layer packing.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

These tests pin the packing contract implemented by
``onecomp.quantizer.dbf.dbf_layer``. DBF stores two +/-1 binary factors
(``dbf_B`` as ``bp1`` and ``dbf_A`` as ``bp3``) and may receive those factors
as either unpacked float matrices or already-packed uint8 buffers.
"""

import torch

from onecomp.quantizer.dbf.dbf_layer import (
    DoubleBinaryLinear,
    pack_binary_factor,
    unpack_binary_factor,
)


def _make_pm1(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (rows, cols), generator=g) * 2 - 1).to(torch.float16)


def _make_factors(out_dim=6, mid_dim=5, in_dim=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    dbf_Da = torch.randn(out_dim, generator=g).to(torch.float16)
    dbf_A = _make_pm1(out_dim, mid_dim, seed=seed + 1)
    dbf_mid = torch.randn(mid_dim, generator=g).to(torch.float16)
    dbf_B = _make_pm1(mid_dim, in_dim, seed=seed + 2)
    dbf_Db = torch.randn(in_dim, generator=g).to(torch.float16)
    return dbf_Da, dbf_A, dbf_mid, dbf_B, dbf_Db


def _reference_weight(dbf_Da, dbf_A, dbf_mid, dbf_B, dbf_Db):
    return (
        dbf_Da.unsqueeze(1) * (dbf_A @ (dbf_mid.unsqueeze(1) * dbf_B)) * dbf_Db.unsqueeze(0)
    ).to(torch.float16)


def test_pack_binary_factor_padding_roundtrip_bit_exact():
    """Packed +/-1 matrices round-trip exactly, including padded shapes."""
    for rows, cols in [(8, 8), (5, 7), (3, 1), (1, 9), (13, 11)]:
        mat = _make_pm1(rows, cols, seed=rows * 100 + cols)
        packed = pack_binary_factor(mat)

        assert packed.dtype == torch.uint8
        assert packed.ndim == 1
        assert packed.numel() == (mat.numel() + 7) // 8

        restored = unpack_binary_factor(packed, (rows, cols))
        assert restored.shape == (rows, cols)
        assert torch.equal(restored.to(torch.float16), mat)


def test_double_binary_linear_packs_unpacked_factors_to_bp_buffers():
    """Unpacked dbf_A/dbf_B inputs are packed into bp3/bp1 buffers."""
    dbf_Da, dbf_A, dbf_mid, dbf_B, dbf_Db = _make_factors()

    layer = DoubleBinaryLinear(
        dbf_Da,
        dbf_A,
        dbf_mid,
        dbf_B,
        dbf_Db,
        use_gemlite=False,
    )

    assert layer._bp1_shape == tuple(dbf_B.shape)
    assert layer._bp3_shape == tuple(dbf_A.shape)
    assert torch.equal(layer.bp1, pack_binary_factor(dbf_B))
    assert torch.equal(layer.bp3, pack_binary_factor(dbf_A))

    x = torch.randn(2, dbf_B.shape[1], dtype=torch.float16)
    expected = torch.nn.functional.linear(
        x, _reference_weight(dbf_Da, dbf_A, dbf_mid, dbf_B, dbf_Db)
    )
    assert torch.allclose(layer(x), expected, atol=5e-2, rtol=5e-2)


def test_double_binary_linear_keeps_already_packed_factors_without_repack():
    """Packed dbf_A/dbf_B inputs are registered directly as bp3/bp1."""
    dbf_Da, dbf_A, dbf_mid, dbf_B, dbf_Db = _make_factors()
    packed_A = pack_binary_factor(dbf_A)
    packed_B = pack_binary_factor(dbf_B)

    layer = DoubleBinaryLinear(
        dbf_Da,
        packed_A,
        dbf_mid,
        packed_B,
        dbf_Db,
        use_gemlite=False,
        dbf_A_is_packed=True,
        dbf_B_is_packed=True,
        dbf_A_original_shape=tuple(dbf_A.shape),
        dbf_B_original_shape=tuple(dbf_B.shape),
    )

    assert layer._bp1_shape == tuple(dbf_B.shape)
    assert layer._bp3_shape == tuple(dbf_A.shape)
    assert torch.equal(layer.bp1, packed_B)
    assert torch.equal(layer.bp3, packed_A)

    x = torch.randn(2, dbf_B.shape[1], dtype=torch.float16)
    expected = torch.nn.functional.linear(
        x, _reference_weight(dbf_Da, dbf_A, dbf_mid, dbf_B, dbf_Db)
    )
    assert torch.allclose(layer(x), expected, atol=5e-2, rtol=5e-2)


def test_from_saved_state_restores_packed_buffers_and_forward():
    """Saved bp1/bp3 buffers restore their inferred shapes and forward path."""
    out_dim, mid_dim, in_dim = 6, 5, 7
    dbf_Da, dbf_A, dbf_mid, dbf_B, dbf_Db = _make_factors(out_dim, mid_dim, in_dim)
    layer = DoubleBinaryLinear(
        dbf_Da,
        dbf_A,
        dbf_mid,
        dbf_B,
        dbf_Db,
        use_gemlite=False,
    )
    restored = DoubleBinaryLinear.from_saved_state(
        layer.state_dict(),
        in_features=in_dim,
        out_features=out_dim,
    )

    assert restored._bp1_shape == (mid_dim, in_dim)
    assert restored._bp3_shape == (out_dim, mid_dim)
    assert torch.equal(restored.bp1, layer.bp1)
    assert torch.equal(restored.bp3, layer.bp3)

    x = torch.randn(3, in_dim, dtype=torch.float16)
    assert torch.equal(restored(x), layer(x))
