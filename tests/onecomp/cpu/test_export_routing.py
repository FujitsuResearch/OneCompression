"""Tests for CPU/GGUF export routing, dense reconstruction, and rotation de-folding.

These tests require no model download or llama.cpp build.

Copyright 2025-2026 Fujitsu Ltd.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

_MDBF_IN_FEATURES = 16
_MDBF_OUT_FEATURES = 8
_MDBF_RANK = 6


def _write_quant_config(tmp_path, quant_method, **extra):
    cfg = {"model_type": "llama", "quantization_config": {"quant_method": quant_method, **extra}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


class _MDBFDenseStub(torch.nn.Module):
    """Dense model exposing one MDBF target layer."""

    def __init__(self, *, with_bias: bool) -> None:
        """Create the target dense linear."""
        super().__init__()
        self.lin = torch.nn.Linear(_MDBF_IN_FEATURES, _MDBF_OUT_FEATURES, bias=with_bias)


def _build_mdbf_state(
    path_count: int, amplitude_rank: int, with_bias: bool
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Build a deterministic MDBF reference layer and flat checkpoint state.

    Args:
        path_count: Number of MDBF paths.
        amplitude_rank: Multi-scale amplitude rank.
        with_bias: Whether the layer carries bias.

    Returns:
        Reference MDBF layer and its state dict under the ``lin`` prefix.
    """
    from onecomp.quantizer.mdbf.mdbf_layer import MultipathMDBFLinear
    from tests.onecomp.fixtures.mdbf_checkpoint import make_mdbf_params

    bias_generator = torch.Generator().manual_seed(999)
    bias = torch.randn(_MDBF_OUT_FEATURES, generator=bias_generator) if with_bias else None
    reference = MultipathMDBFLinear(
        [
            make_mdbf_params(
                _MDBF_OUT_FEATURES,
                _MDBF_IN_FEATURES,
                _MDBF_RANK,
                amplitude_rank,
                seed,
            )
            for seed in range(path_count)
        ],
        bias=bias,
        use_gemlite=False,
    ).eval()
    state = {f"lin.{key}": tensor for key, tensor in reference.state_dict().items()}
    return reference, state


@pytest.mark.parametrize(
    "method,extra,expected_path,is_family",
    [
        ("gptq", {}, "direct", True),
        ("mixed_gptq", {}, "mixed", True),
        ("jointq", {}, "direct", True),
        ("rtn", {}, "direct", True),
        ("dbf", {}, "fallback", False),
        ("autobit", {}, "fallback", False),
        ("onebit", {}, "unsupported", False),
        # MDBF has no lossless layout; rotated MDBF uses the same fallback.
        ("mdbf", {}, "fallback", False),
        ("mdbf", {"rotated": True}, "fallback", False),
        ("gptq", {"rotated": True}, "fallback", True),
        ("mixed_gptq", {"rotated": True}, "fallback", True),
        # act-order uniform GPTQ must go to mixed (direct packing isn't block-aligned)
        ("gptq", {"desc_act": True}, "mixed", True),
        ("gptq", {"actorder": True}, "mixed", True),
        # low-bit uniform GPTQ must go to mixed (no lossless GGUF block type)
        ("gptq", {"bits": 2}, "mixed", True),
        ("gptq", {"bits": 3}, "mixed", True),
        ("jointq", {"bits": 2}, "mixed", True),
        # rotation takes precedence over act-order routing
        ("gptq", {"desc_act": True, "rotated": True}, "fallback", True),
    ],
)
def test_plan_export_routing(tmp_path, method, extra, expected_path, is_family):
    from onecomp.cpu.export.auto import plan_export
    from onecomp.cpu.export.checkpoint import read_quant_meta

    d = _write_quant_config(tmp_path, method, **extra)
    meta = read_quant_meta(d)
    assert meta.quant_method == method
    assert meta.is_gptq_family is is_family
    assert meta.rotated is bool(extra.get("rotated", False))
    assert meta.actorder is bool(extra.get("desc_act", extra.get("actorder", False)))

    plan = plan_export(d)
    assert plan["path"] == expected_path


def test_plan_export_mixed_for_per_layer_low_bits(tmp_path):
    from onecomp.cpu.export.auto import plan_export

    qbits = [{"self_attn.q_proj": {"bits": 3, "method": "gptq", "params": {"group_size": 128}}}]
    d = _write_quant_config(tmp_path, "gptq", bits=4, quantization_bits=qbits)
    assert plan_export(d)["path"] == "mixed"


def test_needs_mixed_export_helpers():
    from onecomp.cpu.export.checkpoint import configured_bit_widths, needs_mixed_export

    assert not needs_mixed_export({"bits": 4, "sym": True})
    assert needs_mixed_export({"bits": 2, "sym": True})
    assert needs_mixed_export({"bits": 8, "sym": False})
    assert needs_mixed_export({"bits": 4, "desc_act": True})
    assert configured_bit_widths(
        {"bits": 4, "quantization_bits": [{}, {"mlp.down_proj": {"bits": 2}}]}
    ) == {4, 2}


@pytest.mark.parametrize("mode", ["auto", "direct", "mixed", "fallback"])
def test_export_to_gguf_rejects_unsupported(tmp_path: Path, mode: str) -> None:
    """An explicit ``mode`` names a path, not a capability: it must not bypass the guard."""
    from onecomp.cpu.export.auto import export_to_gguf

    d = _write_quant_config(tmp_path, "onebit")
    with pytest.raises(ValueError, match="not supported"):
        export_to_gguf(d, str(tmp_path / "out.gguf"), mode=mode)


@pytest.mark.parametrize(
    "method,extra",
    [
        ("dbf", {}),
        ("mdbf", {}),
        ("autobit", {}),
        ("gptq", {"rotated": True}),
    ],
)
@pytest.mark.parametrize("mode", ["direct", "mixed"])
def test_export_to_gguf_rejects_incompatible_forced_mode(
    tmp_path: Path, method: str, extra: dict[str, bool], mode: str
) -> None:
    """A forced packed path must reject checkpoints without that capability."""
    from onecomp.cpu.export.auto import export_to_gguf

    d = _write_quant_config(tmp_path, method, **extra)
    with pytest.raises(ValueError, match="needs the AutoGPTQ block layout"):
        export_to_gguf(d, str(tmp_path / "out.gguf"), mode=mode)


@pytest.mark.parametrize("mode", ["auto", "fallback"])
def test_export_to_gguf_mdbf_dispatches_fallback(tmp_path: Path, mode: str) -> None:
    """The public exporter dispatches supported MDBF modes to fallback."""
    from onecomp.cpu.export.auto import export_to_gguf

    quantized_dir = _write_quant_config(tmp_path, "mdbf")
    out_gguf = str(tmp_path / "out.gguf")
    with patch("onecomp.cpu.export.fallback.export_via_dequantize") as export_mock:
        result = export_to_gguf(quantized_dir, out_gguf, mode=mode)

    export_mock.assert_called_once_with(quantized_dir, out_gguf, qtype="Q4_K_M", work_dir=None)
    assert result["path"] == "fallback"


def test_dequantize_to_hf_rejects_unsupported(tmp_path: Path) -> None:
    """The low-level entry point guards too; it is public and reached via other paths."""
    from onecomp.cpu.export.dequantize import dequantize_to_hf

    d = _write_quant_config(tmp_path, "onebit")
    with pytest.raises(ValueError, match="no dense reconstruction"):
        dequantize_to_hf(d, str(tmp_path / "dense"))


def test_reject_unfilled_weights_flags_random_init_tensors() -> None:
    """An unknown layout leaves dense weights unsourced; that must raise, not warn."""
    from onecomp.cpu.export.dequantize import _reject_unfilled_weights

    missing = [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.bias",
        "model.rotary_emb.inv_freq",  # a buffer, legitimately absent
    ]
    with pytest.raises(RuntimeError, match="random init"):
        _reject_unfilled_weights(missing, set(), "/ckpt", "future_method")


def test_reject_unfilled_weights_ignores_buffers_and_retied_lm_head() -> None:
    """Buffers and weights restored by tying are not unfilled parameters."""
    from onecomp.cpu.export.dequantize import _reject_unfilled_weights

    _reject_unfilled_weights(["model.rotary_emb.inv_freq"], set(), "/ckpt", "gptq")
    _reject_unfilled_weights(["lm_head.weight"], {"lm_head.weight"}, "/ckpt", "gptq")


def test_dequantize_to_hf_rejects_unknown_layout_end_to_end(tmp_path: Path) -> None:
    """An unrecognized tensor layout must fail before saving a random-init model.

    This end-to-end call proves ``_reject_unfilled_weights`` remains wired into
    ``dequantize_to_hf`` for layouts no supported reconstructor recognizes.
    """
    from safetensors.torch import save_file
    from transformers import LlamaConfig

    from onecomp.cpu.export.dequantize import dequantize_to_hf

    config = LlamaConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        num_key_value_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        vocab_size=32,
        tie_word_embeddings=False,
    )
    # Keep the output outside the checkpoint so the shard glob cannot see it.
    ckpt = tmp_path / "ckpt"
    out = tmp_path / "dense"
    ckpt.mkdir()

    cfg_dict = config.to_dict()
    cfg_dict["quantization_config"] = {"quant_method": "future_method", "bits": 2}
    (ckpt / "config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")

    # This unknown factorization has no ``.weight``, and none of the supported
    # reconstructors recognizes its keys.
    save_file(
        {
            "model.layers.0.self_attn.q_proj.factor_a": torch.zeros(16, 4),
            "model.layers.0.self_attn.q_proj.factor_b": torch.zeros(4, 16),
        },
        str(ckpt / "model.safetensors"),
    )

    with pytest.raises(RuntimeError, match="random init"):
        dequantize_to_hf(str(ckpt), str(out))
    assert not (out / "model.safetensors").exists(), "must not write a broken model"


def test_dbf_dequantize_matches_forward():
    """Identity-forward reconstruction reproduces DoubleBinaryLinear outputs."""
    from onecomp.cpu.export.dequantize import _dequantize_dbf_layers
    from onecomp.quantizer.dbf.dbf_layer import DoubleBinaryLinear, pack_binary

    torch.manual_seed(0)
    in_f, mid, out_f = 16, 24, 8

    def _rand_binary(rows, cols):
        return (torch.randint(0, 2, (rows, cols)) * 2 - 1).to(torch.float16)

    dbf_B = _rand_binary(mid, in_f)
    dbf_A = _rand_binary(out_f, mid)
    state = {
        "lin.scaling0": torch.randn(in_f).to(torch.float16),
        "lin.scaling2": torch.randn(mid).to(torch.float16),
        "lin.scaling4": torch.randn(out_f).to(torch.float16),
        "lin.bp1": pack_binary(dbf_B),
        "lin.bp3": pack_binary(dbf_A),
        "lin.bias": torch.randn(out_f).to(torch.float16),
    }

    # Reference layer.
    lsd = {k.split(".")[-1]: v for k, v in state.items()}
    ref = DoubleBinaryLinear.from_saved_state({k: v for k, v in lsd.items()}, in_f, out_f).eval()

    # A tiny dense model stub exposing the layer's in/out features.
    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(in_f, out_f, bias=True)

    model = _Stub()
    dense, consumed = _dequantize_dbf_layers(model, state, torch.float32)

    assert "lin.weight" in dense and "lin.bias" in dense
    w = dense["lin.weight"]
    b = dense["lin.bias"]

    x = torch.randn(5, in_f, dtype=torch.float16)
    with torch.no_grad():
        expected = ref(x).float()
        got = (x.float() @ w.t()) + b.float()
    assert torch.allclose(expected, got, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("path_count", [1, 2])
@pytest.mark.parametrize("amplitude_rank", [1, 2])
@pytest.mark.parametrize("with_bias", [False, True])
def test_mdbf_dequantize_matches_forward(
    tmp_path: Path, path_count: int, amplitude_rank: int, with_bias: bool
) -> None:
    """Dense MDBF reconstruction matches its factorized fp32 forward."""
    from onecomp.cpu.export.dequantize import _dequantize_mdbf_layers

    reference, state = _build_mdbf_state(path_count, amplitude_rank, with_bias)
    save_directory = _write_quant_config(tmp_path, "mdbf", P=path_count)
    dense, consumed = _dequantize_mdbf_layers(
        _MDBFDenseStub(with_bias=with_bias), state, torch.float32, save_directory
    )

    assert consumed == set(state)
    assert set(dense) == ({"lin.weight", "lin.bias"} if with_bias else {"lin.weight"})

    generator = torch.Generator().manual_seed(100)
    inputs = torch.randn(5, _MDBF_IN_FEATURES, generator=generator)
    with torch.no_grad():
        expected = reference(inputs.float())
        actual = inputs.float() @ dense["lin.weight"].t()
        if with_bias:
            actual += dense["lin.bias"].float()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)


def test_mdbf_dequantize_rejects_missing_path(tmp_path: Path) -> None:
    """A path missing from the checkpoint must not silently reduce P."""
    from onecomp.cpu.export.dequantize import _dequantize_mdbf_layers

    _, state = _build_mdbf_state(path_count=2, amplitude_rank=1, with_bias=False)
    damaged = {key: tensor for key, tensor in state.items() if not key.startswith("lin.paths.1.")}
    save_directory = _write_quant_config(tmp_path, "mdbf", P=2)

    with pytest.raises(ValueError, match="Incomplete MDBF checkpoint"):
        _dequantize_mdbf_layers(
            _MDBFDenseStub(with_bias=False), damaged, torch.float32, save_directory
        )


def test_mdbf_dequantize_rejects_rank_mismatch(tmp_path: Path) -> None:
    """A silent factor-rank mismatch is rejected using packed byte counts."""
    from onecomp.cpu.export.dequantize import _dequantize_mdbf_layers

    _, state = _build_mdbf_state(path_count=1, amplitude_rank=2, with_bias=False)
    state["lin.paths.0.Q_U_amp"] = state["lin.paths.0.Q_U_amp"][:-1]
    state["lin.paths.0.Q_V_amp"] = state["lin.paths.0.Q_V_amp"][:-1]
    save_directory = _write_quant_config(tmp_path, "mdbf", P=1)

    with pytest.raises(ValueError, match="A_sign_packed"):
        _dequantize_mdbf_layers(
            _MDBFDenseStub(with_bias=False), state, torch.float32, save_directory
        )


@pytest.mark.parametrize("tensor_name", ["A_amp", "B_amp", "Q_V_amp"])
def test_mdbf_dequantize_rejects_amp_shape_mismatch(tmp_path: Path, tensor_name: str) -> None:
    """Singleton amplitude axes must not broadcast into a wrong weight."""
    from onecomp.cpu.export.dequantize import _dequantize_mdbf_layers

    _, state = _build_mdbf_state(path_count=1, amplitude_rank=2, with_bias=False)
    key = f"lin.paths.0.{tensor_name}"
    state[key] = state[key][:1]
    save_directory = _write_quant_config(tmp_path, "mdbf", P=1)

    with pytest.raises(ValueError, match=tensor_name):
        _dequantize_mdbf_layers(
            _MDBFDenseStub(with_bias=False), state, torch.float32, save_directory
        )


@pytest.mark.parametrize("factor_dimension", ["rank", "amplitude_rank"])
def test_mdbf_dequantize_rejects_empty_factor_dimension(
    tmp_path: Path, factor_dimension: str
) -> None:
    """Empty factor dimensions must not reconstruct an all-zero weight."""
    from onecomp.cpu.export.dequantize import _dequantize_mdbf_layers

    _, state = _build_mdbf_state(path_count=1, amplitude_rank=2, with_bias=False)
    prefix = "lin.paths.0."
    if factor_dimension == "rank":
        for tensor_name in ("Q_U_amp", "Q_V_amp"):
            state[prefix + tensor_name] = state[prefix + tensor_name][:0]
    else:
        for tensor_name in ("A_amp", "B_amp", "Q_U_amp", "Q_V_amp"):
            state[prefix + tensor_name] = state[prefix + tensor_name][:, :0]
    save_directory = _write_quant_config(tmp_path, "mdbf", P=1)

    with pytest.raises(ValueError, match="positive rank and amplitude dimensions"):
        _dequantize_mdbf_layers(
            _MDBFDenseStub(with_bias=False), state, torch.float32, save_directory
        )


@pytest.mark.parametrize("tensor_name", ["_A_sign_shape", "_B_sign_shape"])
def test_mdbf_dequantize_rejects_sign_shape_mismatch(tmp_path: Path, tensor_name: str) -> None:
    """Persisted sign shapes must agree with reconstruction dimensions."""
    from onecomp.cpu.export.dequantize import _dequantize_mdbf_layers

    _, state = _build_mdbf_state(path_count=1, amplitude_rank=1, with_bias=False)
    key = f"lin.paths.0.{tensor_name}"
    state[key] = state[key] + 1
    save_directory = _write_quant_config(tmp_path, "mdbf", P=1)

    with pytest.raises(ValueError, match=tensor_name):
        _dequantize_mdbf_layers(
            _MDBFDenseStub(with_bias=False), state, torch.float32, save_directory
        )


@pytest.mark.parametrize("tensor_name", ["_A_sign_shape", "_B_sign_shape"])
def test_mdbf_dequantize_rejects_sign_shape_buffer_shape(tmp_path: Path, tensor_name: str) -> None:
    """Malformed sign-shape buffers report shape without expanding values."""
    from onecomp.cpu.export.dequantize import _dequantize_mdbf_layers

    _, state = _build_mdbf_state(path_count=1, amplitude_rank=1, with_bias=False)
    key = f"lin.paths.0.{tensor_name}"
    state[key] = state[key].unsqueeze(0)
    save_directory = _write_quant_config(tmp_path, "mdbf", P=1)

    with pytest.raises(ValueError, match=r"expected \(2,\), got \(1, 2\)"):
        _dequantize_mdbf_layers(
            _MDBFDenseStub(with_bias=False), state, torch.float32, save_directory
        )


def test_mdbf_layer_absent_from_dense_model_raises(tmp_path: Path) -> None:
    """A checkpoint MDBF layer without a dense target is a mapping error."""
    from onecomp.cpu.export.dequantize import _dequantize_mdbf_layers

    _, state = _build_mdbf_state(path_count=1, amplitude_rank=1, with_bias=False)
    save_directory = _write_quant_config(tmp_path, "mdbf", P=1)

    with pytest.raises(RuntimeError, match="no matching nn.Linear"):
        _dequantize_mdbf_layers(torch.nn.Module(), state, torch.float32, save_directory)


@pytest.mark.parametrize(
    "rotated,torch_dtype,rtol,atol",
    [
        pytest.param(False, torch.float32, 1e-4, 1e-3, id="plain-fp32"),
        pytest.param(True, torch.float32, 1e-4, 1e-3, id="rotated-fp32"),
        pytest.param(False, torch.float16, 0.0, 5e-3, id="plain-fp16"),
    ],
)
def test_mdbf_dequantize_to_hf_matches_loader_logits(
    tmp_path: Path,
    rotated: bool,
    torch_dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    """Dense export matches MDBF loader logits for plain and rotated models.

    FP16 uses absolute tolerance because prototype relative error reached 15.7%
    near zero while the worst absolute error was one ULP (9.766e-4).
    """
    from transformers import AutoModelForCausalLM

    from onecomp.cpu.export.dequantize import dequantize_to_hf
    from onecomp.quantized_model_loader import QuantizedModelLoader
    from tests.onecomp.fixtures.mdbf_checkpoint import (
        build_mdbf_model,
        write_mdbf_save_dir,
    )

    reference, config, quantized_names = build_mdbf_model(with_bias=False)
    checkpoint_dir = tmp_path / "checkpoint"
    dense_dir = tmp_path / "dense"
    write_mdbf_save_dir(
        checkpoint_dir,
        config,
        reference.state_dict(),
        quantized_names,
        rotated=rotated,
    )

    with patch(
        "onecomp.quantized_model_loader.AutoTokenizer.from_pretrained",
        return_value=object(),
    ):
        quantized_model, _ = QuantizedModelLoader.load_quantized_model(
            str(checkpoint_dir),
            device_map="",
            local_files_only=True,
        )
    quantized_model.to(dtype=torch_dtype).eval()

    dequantize_to_hf(str(checkpoint_dir), str(dense_dir), torch_dtype=torch_dtype)
    dense_model = AutoModelForCausalLM.from_pretrained(
        dense_dir,
        torch_dtype=torch_dtype,
        local_files_only=True,
    ).eval()

    generator = torch.Generator().manual_seed(102)
    input_ids = torch.randint(0, config.vocab_size, (2, 8), generator=generator)
    with torch.no_grad():
        expected = quantized_model(input_ids).logits.float()
        actual = dense_model(input_ids).logits.float()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_hadamard_defold_roundtrip():
    """De-fold inverts the online down_proj Hadamard applied during rotation."""
    from onecomp.cpu.export.rotation import defold_down_proj_hadamard
    from onecomp.pre_process.hadamard_utils import get_hadK, matmul_hadU_cuda

    torch.manual_seed(1)
    out_f, in_f = 7, 64  # power-of-2 in_features

    w_orig = torch.randn(out_f, in_f, dtype=torch.float32)
    # Simulate rotate_down_proj's online Hadamard fold on the input dim.
    w_stored = matmul_hadU_cuda(w_orig, *get_hadK(in_f))
    # De-folding must recover the original (pre-Hadamard) weight.
    w_recovered = defold_down_proj_hadamard(w_stored)
    assert torch.allclose(w_orig, w_recovered, atol=1e-4)


def test_hadamard_defold_roundtrip_block():
    """Same round-trip for a non-power-of-2 dim using a Hadamard block kernel."""
    from onecomp.cpu.export.rotation import defold_down_proj_hadamard
    from onecomp.pre_process.hadamard_utils import get_hadK, matmul_hadU_cuda

    torch.manual_seed(2)
    out_f, in_f = 5, 12 * 8  # 96 = 12 * 2^3 -> uses the had12 block

    w_orig = torch.randn(out_f, in_f, dtype=torch.float32)
    w_stored = matmul_hadU_cuda(w_orig, *get_hadK(in_f))
    w_recovered = defold_down_proj_hadamard(w_stored)
    assert torch.allclose(w_orig, w_recovered, atol=1e-4)
