"""CPU-only integration test for MoE expert recovery + RTN fallback in

``run_quantize_with_qep_arch``.

A tiny hand-built model with a hard-routed top-1 MoE layer replaces a real
HF model + dataset download (see ``tests/onecomp/test_qep_gptq_regression.py``
for the GPU/network-gated equivalent using TinyLlama). Routing is pinned via
crafted embedding/router weights instead of a trained router.

Copyright 2025-2026 Fujitsu Ltd.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_layers import GradientCheckpointingLayer

from onecomp.qep._qep_config import QEPConfig
from onecomp.qep._quantize_with_qep_arch import run_quantize_with_qep_arch
from onecomp.quantizer.gptq import GPTQ

HIDDEN = 4
INTERMEDIATE = 6
NUM_EXPERTS = 3
SEQ_LEN = 3


class _ToyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _ToyMoE(nn.Module):
    """Hard top-1 routed MoE: only experts actually selected are invoked,

    so an unselected expert's Linear layers never fire their forward hook
    (mirrors real MoE token-dropping behaviour used by the production code
    under test).
    """

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(HIDDEN, NUM_EXPERTS, bias=False)
        self.experts = nn.ModuleList([_ToyExpert() for _ in range(NUM_EXPERTS)])

    def forward(self, hidden_states):
        bsz, seq, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        expert_idx = self.router(flat).argmax(dim=-1)
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            mask = expert_idx == i
            if mask.any():
                out[mask] = expert(flat[mask])
        return out.reshape(bsz, seq, hidden)


class _ToyBlock(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.mlp = _ToyMoE()

    def forward(self, hidden_states, **kwargs):
        return hidden_states + self.mlp(hidden_states)


class _ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ToyBlock()])


class _ToyModel(nn.Module):
    """Minimal CausalLM-shaped model: embed -> single MoE block.

    Structured as ``model.model.layers`` (Llama-style) so
    ``onecomp.utils.blockwise._get_blocks`` finds the block list. Router
    weights are pinned so that token 0 always selects expert 0, token 1
    always selects expert 1, and expert 2 is never selected by any token.
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(2, HIDDEN)
        # Basis-vector embeddings so the router's argmax routing decision
        # is exactly controlled by the crafted router weights below.
        self.embed.weight.data = torch.eye(2, HIDDEN)
        self.model = _ToyDecoder()

        block = self.model.layers[0]
        router_weight = torch.zeros(NUM_EXPERTS, HIDDEN)
        router_weight[0, 0] = 10.0
        router_weight[1, 1] = 10.0
        block.mlp.router.weight.data = router_weight

    def forward(self, input_ids, **kwargs):
        return self.model.layers[0](self.embed(input_ids), **kwargs)


class _FakeModelConfig:
    def __init__(self, model):
        self._model = model

    def load_model(self, device_map=None):
        return self._model

    def load_tokenizer(self):
        return None


def _fake_prepare_calibration_dataset(**_kwargs):
    # sample 0: all token 0 (selects expert 0). sample 1: all token 1
    # (selects expert 1). Expert 2 is never selected by either sample.
    input_ids = torch.tensor([[0] * SEQ_LEN, [1] * SEQ_LEN], dtype=torch.long)
    return {"input_ids": input_ids}


def _names_for_expert(model, idx):
    return [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and f"experts.{idx}." in name
    ]


def _is_rtn_fallback(quantizer, name):
    """A real QEP/GPTQ pass always sets ``quantization_time``; the RTN

    fallback assigns a ``GPTQResult`` directly and bypasses it entirely, so
    its absence identifies which path a given expert went through.
    """
    return quantizer.results[name].quantization_time is None


def _run_quantized(monkeypatch, caplog):
    """Run ``run_quantize_with_qep_arch`` on the toy MoE model above.

    Calibration sample 0 (the single sample used for activation grouping)
    only activates expert 0, so expert 0 is captured directly. Sample 1
    only activates expert 1, so expert 1 is invisible to that single-sample
    grouping but still active over the full calibration set. Expert 2 is
    never activated by any sample.

    Called directly from each test body (rather than as a fixture) so that
    the quantization run and its log capture happen in the same pytest
    phase as the assertions -- ``caplog.records`` does not carry over from
    a fixture's setup phase into the test's call phase.
    """
    torch.manual_seed(0)
    model = _ToyModel()

    monkeypatch.setattr(
        "onecomp.qep._quantize_with_qep_arch.prepare_calibration_dataset",
        _fake_prepare_calibration_dataset,
    )

    quantizer = GPTQ(wbits=4, groupsize=2, sym=True, include_layer_keywords=["experts"])
    qep_config = QEPConfig(device="cpu", percdamp=0.01, perccorr=0.5)

    caplog.set_level(logging.INFO, logger="onecomp.qep._quantize_with_qep_arch")
    run_quantize_with_qep_arch(
        model_config=_FakeModelConfig(model),
        quantizer=quantizer,
        qep_config=qep_config,
        calibration_config=None,
        report_progress=False,
    )
    return model, quantizer


def test_no_expert_layer_is_silently_skipped(monkeypatch, caplog):
    """Every expert projection ends up quantized, regardless of whether it

    was captured directly, recovered with a real Hessian, or recovered
    with no Hessian at all.
    """
    model, quantizer = _run_quantized(monkeypatch, caplog)
    all_names = {
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and ".experts." in name
    }
    assert len(all_names) == NUM_EXPERTS * 3
    assert set(quantizer.results.keys()) == all_names


def test_expert_captured_directly_is_quantized_normally(monkeypatch, caplog):
    """Expert 0 is selected by the single calibration sample used for

    activation grouping, so it needs no recovery and is quantized through
    the ordinary QEP/GPTQ path.
    """
    model, quantizer = _run_quantized(monkeypatch, caplog)
    names = _names_for_expert(model, 0)
    assert all(not _is_rtn_fallback(quantizer, n) for n in names)


def test_expert_missed_by_grouping_recovers_a_real_hessian(monkeypatch, caplog):
    """Expert 1 is only selected by the second calibration sample, so the

    single-sample activation grouping misses it. The full calibration set
    still routes tokens to it, so its recovered Hessian is real and it is
    quantized through the ordinary QEP/GPTQ path rather than falling back
    to RTN.
    """
    model, quantizer = _run_quantized(monkeypatch, caplog)
    names = _names_for_expert(model, 1)
    assert all(not _is_rtn_fallback(quantizer, n) for n in names)


def test_expert_never_selected_falls_back_to_rtn(monkeypatch, caplog):
    """Expert 2 is never selected by any calibration token, so even after

    recovery its Hessian is ``None``. It falls back to RTN quantization
    instead of being skipped, and RTN never applies activation ordering.
    """
    model, quantizer = _run_quantized(monkeypatch, caplog)
    names = _names_for_expert(model, 2)
    assert all(_is_rtn_fallback(quantizer, n) for n in names)
    assert all(quantizer.results[n].actorder is False for n in names)


def test_rtn_fallback_weight_is_actually_applied_to_the_module(monkeypatch, caplog):
    """The RTN-fallback ``GPTQResult`` must actually flow into the module's

    live weight, not just sit unused in ``quantizer.results``. A
    ``compute_dequantized_weight()`` shape/broadcast bug (e.g. wrong
    scale/zero orientation for per-channel groupsize) would only surface as
    a caught-and-logged error here, leaving the original weight in place
    without failing the run -- so this checks both the weight and the log.
    """
    torch.manual_seed(0)
    model = _ToyModel()
    expert2_down_proj = model.model.layers[0].mlp.experts[2].down_proj
    original_weight = expert2_down_proj.weight.data.clone()

    monkeypatch.setattr(
        "onecomp.qep._quantize_with_qep_arch.prepare_calibration_dataset",
        _fake_prepare_calibration_dataset,
    )
    quantizer = GPTQ(wbits=4, groupsize=2, sym=True, include_layer_keywords=["experts"])
    qep_config = QEPConfig(device="cpu", percdamp=0.01, perccorr=0.5)

    caplog.set_level(logging.INFO, logger="onecomp.qep._quantize_with_qep_arch")
    run_quantize_with_qep_arch(
        model_config=_FakeModelConfig(model),
        quantizer=quantizer,
        qep_config=qep_config,
        calibration_config=None,
        report_progress=False,
    )

    assert not any("Failed to compute dequantized weight" in r.message for r in caplog.records)
    # quantizer.module_to_name is cleared by execute_post_processing() at the
    # end of the run, so look the name up via the model tree instead.
    name = next(n for n in _names_for_expert(model, 2) if n.endswith("down_proj"))
    expected = quantizer.results[name].compute_dequantized_weight().to(original_weight.dtype)
    assert torch.equal(expert2_down_proj.weight.data, expected)
    assert not torch.equal(expert2_down_proj.weight.data, original_weight)


def test_recovery_is_logged_once_with_correct_count(monkeypatch, caplog):
    """Both missed experts (1 and 2, three projections each) are reported

    together in a single recovery log line.
    """
    _run_quantized(monkeypatch, caplog)
    recovered_logs = [r for r in caplog.records if "Recovered" in r.message]
    assert len(recovered_logs) == 1
    assert "6 expert module(s)" in recovered_logs[0].message


def test_rtn_fallback_is_logged_per_module(monkeypatch, caplog):
    """One RTN-fallback warning is emitted per expert-2 projection."""
    model, _quantizer = _run_quantized(monkeypatch, caplog)
    names = _names_for_expert(model, 2)
    rtn_fallback_logs = [r for r in caplog.records if "falling back to RTN" in r.message]
    assert len(rtn_fallback_logs) == len(names)


class _NotGPTQMarker:
    """Unrelated to the real GPTQ class.

    Used to force ``isinstance(quantizer, GPTQ)`` to False in
    ``run_quantize_with_qep_arch`` for the test below, without needing a
    second, fully-featured ``Quantizer`` implementation wired through the
    whole toy-model harness.
    """


def test_non_gptq_quantizer_skips_expert_with_no_tokens(monkeypatch, caplog):
    """Non-GPTQ quantizers keep the pre-RTN-fallback behaviour: an expert

    that never receives calibration tokens is skipped outright (absent from
    ``quantizer.results``, discarded from the remaining targets), not
    RTN-quantized. Only ``GPTQ`` gets the RTN fallback.
    """
    torch.manual_seed(0)
    model = _ToyModel()

    monkeypatch.setattr(
        "onecomp.qep._quantize_with_qep_arch.prepare_calibration_dataset",
        _fake_prepare_calibration_dataset,
    )
    monkeypatch.setattr("onecomp.qep._quantize_with_qep_arch.GPTQ", _NotGPTQMarker)

    quantizer = GPTQ(wbits=4, groupsize=2, sym=True, include_layer_keywords=["experts"])
    qep_config = QEPConfig(device="cpu", percdamp=0.01, perccorr=0.5)

    caplog.set_level(logging.INFO, logger="onecomp.qep._quantize_with_qep_arch")
    run_quantize_with_qep_arch(
        model_config=_FakeModelConfig(model),
        quantizer=quantizer,
        qep_config=qep_config,
        calibration_config=None,
        report_progress=False,
    )

    names = _names_for_expert(model, 2)
    assert all(n not in quantizer.results for n in names)
    skip_logs = [r for r in caplog.records if "skipping" in r.message]
    assert len(skip_logs) == len(names)
    assert not any("falling back to RTN" in r.message for r in caplog.records)
