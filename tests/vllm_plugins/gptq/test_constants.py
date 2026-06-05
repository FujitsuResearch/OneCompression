"""Tests for shared GPTQ Marlin dispatch constants."""

from vllm_plugins.gptq.constants import GPTQ_MARLIN_SUPPORTED_BITS, should_use_gptq_marlin


def test_gptq_marlin_supported_bits():
    assert GPTQ_MARLIN_SUPPORTED_BITS == frozenset({4, 8})


def test_should_use_gptq_marlin_true_for_4bit_symmetric():
    assert should_use_gptq_marlin(bits=4, sym=True, desc_act=False)


def test_should_use_gptq_marlin_false_for_desc_act():
    assert not should_use_gptq_marlin(bits=4, sym=True, desc_act=True)


def test_should_use_gptq_marlin_false_for_asymmetric():
    assert not should_use_gptq_marlin(bits=4, sym=False, desc_act=False)


def test_should_use_gptq_marlin_false_for_unsupported_bits():
    assert not should_use_gptq_marlin(bits=2, sym=True, desc_act=False)


def test_should_use_gptq_marlin_false_when_sym_unknown():
    assert not should_use_gptq_marlin(bits=4, sym=None, desc_act=False)
    assert not should_use_gptq_marlin(bits=4, desc_act=False)  # default sym=False
