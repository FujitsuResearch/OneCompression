"""
Lightweight coverage for post-processes on unpacked GPTQLinear inputs.

These tests avoid downloading a Hugging Face model.  They build a minimal
HF-like causal-LM shape locally and feed BlockWisePTQ.run() / GlobalPTQ.run()
unpacked GPTQLinear layers produced through the same quantizer-specific
inference-layer builders used by Runner-created quantized models.

Covered unpacked cases (the bit widths that GPTQLinear packing cannot represent,
so ``pack_weights=False`` is mandatory):
- GPTQ 5-bit: GPTQLinear packing helpers support only {2, 3, 4, 8}.
- RTN 5-bit: RTN also uses GPTQLinear storage and the same packing limits.
- JointQ 1-bit: JointQ 1-bit has no GPTQLinear packing layout.

Usage:
    pytest tests/onecomp/post_process/test_post_process_unpacked.py -v

Copyright 2025-2026 Fujitsu Ltd.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import onecomp.calibration as calibration_module
from onecomp import CalibrationConfig
from onecomp.post_process.blockwise_ptq import BlockWisePTQ
from onecomp.post_process.global_ptq import GlobalPTQ
from onecomp.quantizer.gptq.gptq_layer import GPTQLinear


HIDDEN_SIZE = 8
GROUP_SIZE = 4
SEQ_LEN = 3


class _TinyBlock(nn.Module):
    """Small transformer-block stand-in containing one projection."""

    def __init__(self, proj: nn.Module):
        super().__init__()
        self.proj = proj

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        **kwargs,
    ):
        _ = attention_mask, kwargs
        return self.proj(hidden_states)


class _TinyBackbone(nn.Module):
    """Backbone shape expected by blockwise helpers: model.layers + embed_tokens."""

    def __init__(self, proj: nn.Module, *, hidden_size: int, vocab_size: int = 16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_TinyBlock(proj)])


class _TinyCausalLM(nn.Module):
    """Minimal HF-like causal-LM wrapper for post-process .run()."""

    def __init__(self, proj: nn.Module, *, hidden_size: int, quant_config: dict | None = None):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, use_cache=True)
        if quant_config is not None:
            self.config.quantization_config = quant_config
        self.model = _TinyBackbone(proj, hidden_size=hidden_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden_states = self.model.embed_tokens(input_ids)
        hidden_states = hidden_states.squeeze(0)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs,
            )
        return SimpleNamespace(logits=hidden_states.unsqueeze(0))


class _TinyModelConfig:
    """ModelConfig stand-in returning the local teacher model and dummy tokenizer."""

    def __init__(self, teacher_model: nn.Module):
        self.teacher_model = teacher_model

    def load_tokenizer(self):
        return object()

    def load_model(self, device_map=None):  # pylint: disable=unused-argument
        return self.teacher_model.cpu()


def _quant_config(bits: int) -> dict:
    return {
        "quant_method": "gptq",
        "bits": bits,
        "groupsize": GROUP_SIZE,
        "group_size": GROUP_SIZE,
        "checkpoint_format": "gptq",
        "modules_in_block_to_quantize": [["proj"]],
    }


def _fake_prepare_calibration_dataset(
    tokenizer,  # pylint: disable=unused-argument
    device,  # pylint: disable=unused-argument
    calibration_config,  # pylint: disable=unused-argument
    model,  # pylint: disable=unused-argument
    logger,  # pylint: disable=unused-argument
):
    return {
        "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "attention_mask": torch.ones((1, SEQ_LEN), dtype=torch.long),
    }


def _base_linear() -> nn.Linear:
    linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
    with torch.no_grad():
        values = torch.linspace(-0.5, 0.5, HIDDEN_SIZE * HIDDEN_SIZE)
        linear.weight.copy_(values.reshape(HIDDEN_SIZE, HIDDEN_SIZE))
        linear.bias.zero_()
    return linear


def _make_gptq_5bit_layer() -> nn.Module:
    from onecomp.quantizer.gptq._gptq import GPTQ, GPTQResult

    bits = 5
    num_groups = HIDDEN_SIZE // GROUP_SIZE
    qweight = (
        torch.arange(HIDDEN_SIZE * HIDDEN_SIZE, dtype=torch.int32).reshape(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
        )
        % (1 << bits)
    )
    scales = torch.full((num_groups, HIDDEN_SIZE), 0.02, dtype=torch.float16)
    qzeros = torch.zeros((num_groups, HIDDEN_SIZE), dtype=torch.int32)
    result = GPTQResult(
        wbits=bits,
        groupsize=GROUP_SIZE,
        actorder=False,
        sym=False,
        qweight=qweight,
        scales=scales,
        qzeros=qzeros,
        perm=None,
    )
    return GPTQ(wbits=bits, groupsize=GROUP_SIZE, sym=False).create_inference_layer(
        result=result,
        linear_module=_base_linear(),
        pack_weights=False,
        use_gemlite=False,
    )


def _make_rtn_5bit_layer() -> nn.Module:
    from onecomp.quantizer.rtn._rtn import RTN

    quantizer = RTN(wbits=5, groupsize=GROUP_SIZE, sym=False)
    linear = _base_linear()
    result = quantizer.quantize_layer(linear)
    return quantizer.create_inference_layer(
        result=result,
        linear_module=linear,
        pack_weights=False,
        use_gemlite=False,
    )


def _make_jointq_1bit_layer() -> nn.Module:
    from onecomp.quantizer.jointq._jointq import JointQ, JointQResult

    bits = 1
    num_groups = HIDDEN_SIZE // GROUP_SIZE
    assignment = (
        torch.arange(HIDDEN_SIZE * num_groups * GROUP_SIZE, dtype=torch.int8).reshape(
            HIDDEN_SIZE,
            num_groups,
            GROUP_SIZE,
        )
        % 2
    )
    result = JointQResult(
        bits=bits,
        symmetric=False,
        group_size=GROUP_SIZE,
        scale=torch.full((HIDDEN_SIZE, num_groups), 0.1, dtype=torch.float16),
        zero_point=torch.zeros((HIDDEN_SIZE, num_groups), dtype=torch.float16),
        assignment=assignment,
        perm=None,
    )
    return JointQ(bits=bits, group_size=GROUP_SIZE, symmetric=False).create_inference_layer(
        result=result,
        linear_module=_base_linear(),
        pack_weights=False,
        use_gemlite=False,
    )


def _make_teacher_model() -> _TinyCausalLM:
    teacher_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
    with torch.no_grad():
        teacher_proj.weight.copy_(torch.eye(HIDDEN_SIZE))
        teacher_proj.bias.zero_()
    return _TinyCausalLM(
        teacher_proj,
        hidden_size=HIDDEN_SIZE,
        quant_config=None,
    )


_UNPACKED_CASES = [
    pytest.param("gptq_5bit", _make_gptq_5bit_layer, 5, id="gptq-5bit"),
    pytest.param("rtn_5bit", _make_rtn_5bit_layer, 5, id="rtn-5bit"),
    pytest.param("jointq_1bit", _make_jointq_1bit_layer, 1, id="jointq-1bit"),
]


@pytest.mark.parametrize(("case_name", "make_layer", "bits"), _UNPACKED_CASES)
def test_blockwise_ptq_run_accepts_unpacked_gptqlinear_inputs(
    monkeypatch,
    case_name,
    make_layer,
    bits,
):
    """BlockWisePTQ.run() accepts unpacked GPTQLinear buffers from GPTQ/RTN/JointQ."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    monkeypatch.setattr(
        calibration_module,
        "prepare_calibration_dataset",
        _fake_prepare_calibration_dataset,
    )

    quantized_layer = make_layer()
    assert isinstance(quantized_layer, GPTQLinear)
    assert quantized_layer.wbits == bits
    assert quantized_layer._weight_is_packed is False

    model = _TinyCausalLM(
        quantized_layer,
        hidden_size=HIDDEN_SIZE,
        quant_config=_quant_config(bits),
    )
    model_config = _TinyModelConfig(_make_teacher_model())

    blockwise_ptq = BlockWisePTQ(
        epochs=1,
        cbq_enable=False,
        gptq_lr=1e-4,
        calibration_config=CalibrationConfig(
            num_calibration_samples=1,
            max_length=SEQ_LEN,
        ),
    )
    blockwise_ptq.run(model, model_config)

    gptq_layers = [module for module in model.modules() if isinstance(module, GPTQLinear)]
    assert len(gptq_layers) == 1, f"{case_name}: expected one GPTQLinear layer"
    assert gptq_layers[0]._weight_is_packed is False
    assert not hasattr(gptq_layers[0], "_opt_scales")
    assert not hasattr(gptq_layers[0], "_opt_zeros")

    assert not model.training
    assert {str(param.device) for param in model.parameters()} == {"cpu"}

    history = model.config.quantization_config["onecomp_post_processes"]
    assert history[-1]["class"] == "BlockWisePTQ"


@pytest.mark.parametrize(("case_name", "make_layer", "bits"), _UNPACKED_CASES)
def test_global_ptq_run_accepts_unpacked_gptqlinear_inputs(
    monkeypatch,
    case_name,
    make_layer,
    bits,
):
    """GlobalPTQ.run() accepts unpacked GPTQLinear buffers from GPTQ/RTN/JointQ."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    monkeypatch.setattr(
        calibration_module,
        "prepare_calibration_dataset",
        _fake_prepare_calibration_dataset,
    )

    quantized_layer = make_layer()
    assert isinstance(quantized_layer, GPTQLinear)
    assert quantized_layer.wbits == bits
    assert quantized_layer._weight_is_packed is False

    model = _TinyCausalLM(
        quantized_layer,
        hidden_size=HIDDEN_SIZE,
        quant_config=_quant_config(bits),
    )
    model_config = _TinyModelConfig(_make_teacher_model())

    global_ptq = GlobalPTQ(
        epochs=1,
        gptq_lr=1e-4,
        calibration_config=CalibrationConfig(
            num_calibration_samples=1,
            max_length=SEQ_LEN,
        ),
        eval_interval=1,
        use_gradient_checkpointing=False,
    )
    global_ptq.run(model, model_config)

    gptq_layers = [module for module in model.modules() if isinstance(module, GPTQLinear)]
    assert len(gptq_layers) == 1, f"{case_name}: expected one GPTQLinear layer"
    assert gptq_layers[0]._weight_is_packed is False
    assert not hasattr(gptq_layers[0], "_opt_scales")
    assert not hasattr(gptq_layers[0], "_opt_zeros")

    assert not model.training
    assert {str(param.device) for param in model.parameters()} == {"cpu"}

    history = model.config.quantization_config["onecomp_post_processes"]
    assert history[-1]["class"] == "GlobalPTQ"
