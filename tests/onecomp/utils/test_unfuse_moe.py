"""

Copyright 2025-2026 Fujitsu Ltd.

Roundtrip tests for unfuse_moe_experts / fuse_moe_experts (no full model).

"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "unfuse_moe",
    _REPO_ROOT / "onecomp" / "utils" / "unfuse_moe.py",
)
_unfuse_mod = importlib.util.module_from_spec(_spec)
sys.modules["unfuse_moe"] = _unfuse_mod
assert _spec.loader is not None
_spec.loader.exec_module(_unfuse_mod)

_GenericFusedExperts = _unfuse_mod._GenericFusedExperts
_fuse_gpt_oss_one = _unfuse_mod._fuse_gpt_oss_one
_fuse_one = _unfuse_mod._fuse_one
_gpt_oss_experts_type = _unfuse_mod._gpt_oss_experts_type
_unfuse_gpt_oss_one = _unfuse_mod._unfuse_gpt_oss_one
_unfuse_one = _unfuse_mod._unfuse_one
fuse_moe_experts = _unfuse_mod.fuse_moe_experts
unfuse_moe_experts = _unfuse_mod.unfuse_moe_experts


def _silence_logger():
    return logging.getLogger("test_unfuse_moe")


class _TinyMoEBlock(nn.Module):
    def __init__(self, fused: nn.Module):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = fused


class _TinyModel(nn.Module):
    def __init__(self, fused: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([_TinyMoEBlock(fused)])


def test_generic_fused_experts_unfuse_fuse_roundtrip():
    torch.manual_seed(0)
    num_experts, inter, hidden = 2, 4, 8
    gate_up = torch.randn(num_experts, 2 * inter, hidden, dtype=torch.float32)
    down = torch.randn(num_experts, hidden, inter, dtype=torch.float32)
    act_fn = torch.nn.functional.silu
    fused = _GenericFusedExperts(gate_up.clone(), down.clone(), act_fn)

    unfused = _unfuse_one(fused)
    ref_gate_up = gate_up
    ref_down = down

    refuzed = _fuse_one(unfused, override=None)
    assert torch.allclose(refuzed.gate_up_proj.data, ref_gate_up, atol=1e-6, rtol=1e-5)
    assert torch.allclose(refuzed.down_proj.data, ref_down, atol=1e-6, rtol=1e-5)

    model = _TinyModel(_GenericFusedExperts(gate_up.clone(), down.clone(), act_fn))
    logger = _silence_logger()
    assert unfuse_moe_experts(model, logger)
    assert fuse_moe_experts(model, logger, fused_weight_overrides=None)
    out = model.layers[0].mlp.experts
    assert torch.allclose(out.gate_up_proj.data, ref_gate_up, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out.down_proj.data, ref_down, atol=1e-6, rtol=1e-5)


def test_gpt_oss_experts_roundtrip_if_available():
    gpt_oss_cls = _gpt_oss_experts_type()
    if gpt_oss_cls is None:
        return

    torch.manual_seed(1)
    num_experts, hidden, inter = 2, 8, 4
    config = SimpleNamespace(
        intermediate_size=inter,
        num_local_experts=num_experts,
        hidden_size=hidden,
    )
    fused = gpt_oss_cls(config)
    fused.gate_up_proj = nn.Parameter(
        torch.randn(num_experts, hidden, 2 * inter, dtype=torch.float32)
    )
    fused.gate_up_proj_bias = nn.Parameter(
        torch.randn(num_experts, 2 * inter, dtype=torch.float32)
    )
    fused.down_proj = nn.Parameter(
        torch.randn(num_experts, inter, hidden, dtype=torch.float32)
    )
    fused.down_proj_bias = nn.Parameter(
        torch.randn(num_experts, hidden, dtype=torch.float32)
    )
    fused.alpha = 1.702
    fused.limit = 7.0

    ref = {
        "gate_up_proj": fused.gate_up_proj.data.clone(),
        "gate_up_proj_bias": fused.gate_up_proj_bias.data.clone(),
        "down_proj": fused.down_proj.data.clone(),
        "down_proj_bias": fused.down_proj_bias.data.clone(),
        "alpha": fused.alpha,
        "limit": fused.limit,
    }

    unfused = _unfuse_gpt_oss_one(fused)
    refuzed = _fuse_gpt_oss_one(unfused, override=None)
    for key in ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"):
        assert torch.allclose(getattr(refuzed, key).data, ref[key], atol=1e-5, rtol=1e-4)
    assert refuzed.alpha == ref["alpha"]
    assert refuzed.limit == ref["limit"]

_dequantized_weight_bias = _unfuse_mod._dequantized_weight_bias
_gpt_oss_combine_gate_up = _unfuse_mod._gpt_oss_combine_gate_up
_UnfusedExperts = _unfuse_mod._UnfusedExperts
_ExpertMLP = _unfuse_mod._ExpertMLP
_GPTQ_STACK = None


def _get_gptq_classes():
    global _GPTQ_STACK
    if _GPTQ_STACK is not None:
        return _GPTQ_STACK

    from types import ModuleType

    for pkg in (
        "onecomp",
        "onecomp.quantizer",
        "onecomp.utils",
        "onecomp.quantizer.gptq",
    ):
        if pkg not in sys.modules:
            sys.modules[pkg] = ModuleType(pkg)

    gemlite_stub = ModuleType("onecomp.quantizer.gemlite")
    gemlite_stub.create_gemlite_linear = lambda *args, **kwargs: None
    gemlite_stub.is_gemlite_available = lambda: False
    sys.modules["onecomp.quantizer.gemlite"] = gemlite_stub

    def _load(fullname, relpath):
        path = _REPO_ROOT / relpath
        spec = importlib.util.spec_from_file_location(fullname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    _load("onecomp.quantizer._quantizer", "onecomp/quantizer/_quantizer.py")
    _load("onecomp.utils.quant_config", "onecomp/utils/quant_config.py")
    gptq_mod = _load("onecomp.quantizer.gptq._gptq", "onecomp/quantizer/gptq/_gptq.py")
    layer_mod = _load(
        "onecomp.quantizer.gptq.gptq_layer",
        "onecomp/quantizer/gptq/gptq_layer.py",
    )
    _GPTQ_STACK = (gptq_mod.GPTQ, layer_mod.GPTQLinear)
    return _GPTQ_STACK


def _linear_to_gptq(linear: nn.Linear, inp: torch.Tensor) -> nn.Module:
    GPTQ, GPTQLinear = _get_gptq_classes()
    quantizer = GPTQ(wbits=4, bitpack_on_quantize=True)
    hessian = quantizer.calculate_hessian(linear, inp)
    result = quantizer.quantize_layer(linear, inp, hessian=hessian)
    return GPTQLinear.from_quantization_result(
        result,
        device="cpu",
        pack_weights=True,
        use_gemlite=False,
    )


def _dense_gpt_oss_gate_up(gate_w, up_w):
    hidden, inter = gate_w.shape[1], gate_w.shape[0]
    gate_up = torch.empty(hidden, 2 * inter, dtype=gate_w.dtype)
    gate_up[:, ::2] = gate_w.T
    gate_up[:, 1::2] = up_w.T
    return gate_up


def _expected_gpt_oss_fuse_from_unfused(unfused):
    num_experts = len(unfused)
    expert0 = unfused[0]
    hidden = expert0.gate_proj.in_features
    inter = expert0.gate_proj.out_features
    gate_w0, gate_b0 = _dequantized_weight_bias(expert0.gate_proj)
    _, down_b0 = _dequantized_weight_bias(expert0.down_proj)
    bias_dtype = (
        gate_b0.dtype
        if gate_b0 is not None
        else (down_b0.dtype if down_b0 is not None else gate_w0.dtype)
    )
    gate_up_3d = torch.empty(num_experts, hidden, 2 * inter, dtype=gate_w0.dtype)
    gate_up_bias_2d = torch.zeros(num_experts, 2 * inter, dtype=bias_dtype)
    down_3d = torch.empty(num_experts, inter, hidden, dtype=gate_w0.dtype)
    down_bias_2d = torch.zeros(num_experts, hidden, dtype=bias_dtype)
    for i in range(num_experts):
        expert = unfused[i]
        gate_w, gate_b = _dequantized_weight_bias(expert.gate_proj)
        up_w, up_b = _dequantized_weight_bias(expert.up_proj)
        down_w, down_b = _dequantized_weight_bias(expert.down_proj)
        gate_up_3d[i, :, ::2] = gate_w.T
        gate_up_3d[i, :, 1::2] = up_w.T
        if gate_b is not None:
            gate_up_bias_2d[i, ::2] = gate_b
        if up_b is not None:
            gate_up_bias_2d[i, 1::2] = up_b
        down_3d[i] = down_w.T
        if down_b is not None:
            down_bias_2d[i] = down_b
    return gate_up_3d, gate_up_bias_2d, down_3d, down_bias_2d


def test_gptq_linear_expert_fuse_uses_dequantized_weights():
    if _gpt_oss_experts_type() is None:
        return

    torch.manual_seed(42)
    num_experts, hidden, inter = 2, 16, 8
    inp = torch.randn(8, hidden)
    combine_fn = _gpt_oss_combine_gate_up(1.702, 7.0)

    expert_modules = []
    dense_gate_up_3d = torch.empty(num_experts, hidden, 2 * inter)
    for ei in range(num_experts):
        gate_lin = nn.Linear(hidden, inter, bias=True)
        up_lin = nn.Linear(hidden, inter, bias=True)
        down_lin = nn.Linear(inter, hidden, bias=True)
        dense_gate_up_3d[ei] = _dense_gpt_oss_gate_up(
            gate_lin.weight.detach(), up_lin.weight.detach()
        )
        gate_gptq = _linear_to_gptq(gate_lin, inp)
        up_gptq = _linear_to_gptq(up_lin, inp)
        down_gptq = _linear_to_gptq(down_lin, torch.randn(8, inter))
        expert_modules.append(
            _ExpertMLP(gate_gptq, up_gptq, down_gptq, act_fn=None)
        )

    unfused = _UnfusedExperts(
        num_experts,
        expert_modules,
        act_fn=None,
        combine_fn=combine_fn,
    )
    fused = _fuse_gpt_oss_one(unfused, override=None)
    exp_gu, exp_gu_b, exp_down, exp_down_b = _expected_gpt_oss_fuse_from_unfused(unfused)

    assert torch.allclose(fused.gate_up_proj.data, exp_gu, atol=1e-4, rtol=1e-3)
    assert torch.allclose(fused.gate_up_proj_bias.data, exp_gu_b, atol=1e-4, rtol=1e-3)
    assert torch.allclose(fused.down_proj.data, exp_down, atol=1e-4, rtol=1e-3)
    assert torch.allclose(fused.down_proj_bias.data, exp_down_b, atol=1e-4, rtol=1e-3)

    assert not torch.allclose(
        fused.gate_up_proj.data, dense_gate_up_3d, atol=1e-3, rtol=0
    )

def test_fused_moe_state_dict_has_per_layer_weights():
    gpt_oss_cls = _gpt_oss_experts_type()
    if gpt_oss_cls is None:
        return

    torch.manual_seed(7)
    num_experts, hidden, inter = 2, 8, 4
    config = SimpleNamespace(
        intermediate_size=inter,
        num_local_experts=num_experts,
        hidden_size=hidden,
    )

    class _TwoLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [_TinyMoEBlock(gpt_oss_cls(config)), _TinyMoEBlock(gpt_oss_cls(config))]
            )

    model = _TwoLayerModel()
    logger = _silence_logger()
    assert unfuse_moe_experts(model, logger)
    for layer in model.layers:
        for ei in range(len(layer.mlp.experts)):
            expert = layer.mlp.experts[ei]
            nn.init.uniform_(expert.gate_proj.weight)
            nn.init.uniform_(expert.up_proj.weight)
            nn.init.uniform_(expert.down_proj.weight)
    assert fuse_moe_experts(model, logger)

    state_dict = model.state_dict()
    assert not any(k.endswith("$") for k in state_dict)
    assert "layers.0.mlp.experts.gate_up_proj" in state_dict
    assert "layers.1.mlp.experts.gate_up_proj" in state_dict
    ptr0 = model.layers[0].mlp.experts.gate_up_proj.data.untyped_storage().data_ptr()
    ptr1 = model.layers[1].mlp.experts.gate_up_proj.data.untyped_storage().data_ptr()
    assert ptr0 != ptr1

def test_ensure_unique_fused_moe_parameters():
    gpt_oss_cls = _gpt_oss_experts_type()
    if gpt_oss_cls is None:
        return

    config = SimpleNamespace(intermediate_size=4, num_local_experts=2, hidden_size=8)
    a = gpt_oss_cls(config)
    b = gpt_oss_cls(config)
    # simulate safetensors-style shared storage
    b.gate_up_proj = a.gate_up_proj
    b.down_proj = a.down_proj

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_TinyMoEBlock(a), _TinyMoEBlock(b)])

    model = M()
    _unfuse_mod._ensure_unique_fused_moe_parameters(model)
    ptr0 = model.layers[0].mlp.experts.gate_up_proj.untyped_storage().data_ptr()
    ptr1 = model.layers[1].mlp.experts.gate_up_proj.untyped_storage().data_ptr()
    assert ptr0 != ptr1
    _unfuse_mod.validate_fused_moe_state_dict(model)

def test_fused_state_dict_has_per_layer_moe_keys():
    gpt_oss_cls = _gpt_oss_experts_type()
    if gpt_oss_cls is None:
        return

    torch.manual_seed(11)
    config = SimpleNamespace(intermediate_size=4, num_local_experts=2, hidden_size=8)

    class _TwoLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [_TinyMoEBlock(gpt_oss_cls(config)), _TinyMoEBlock(gpt_oss_cls(config))]
            )

    model = _TwoLayerModel()
    logger = _silence_logger()
    assert unfuse_moe_experts(model, logger)
    for layer in model.layers:
        for ei in range(len(layer.mlp.experts)):
            expert = layer.mlp.experts[ei]
            nn.init.uniform_(expert.gate_proj.weight)
            nn.init.uniform_(expert.up_proj.weight)
            nn.init.uniform_(expert.down_proj.weight)
    assert fuse_moe_experts(model, logger)

    sd = model.state_dict()
    assert not any(k.endswith("$") for k in sd)
    assert "layers.0.mlp.experts.gate_up_proj" in sd
    assert "layers.1.mlp.experts.gate_up_proj" in sd
    _unfuse_mod.validate_fused_moe_state_dict(model)

