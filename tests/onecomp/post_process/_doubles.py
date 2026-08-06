"""Shared model/model_config doubles for post_process tests.

Test doubles used by both ``test_base_run`` and ``test_runtime``: a model
stand-in that records device-mode normalisation calls, and the two
``model_config`` stand-ins (plain and rotated) the consistency check inspects.

Copyright 2025-2026 Fujitsu Ltd.
"""

from onecomp.rotated_model_config import RotatedModelConfig
from tests.onecomp.fixtures.quant_config import FakeConfig


class FakeModel:
    """Model stand-in that records device-mode normalisation calls.

    Tracks the device, plus the number of ``cpu()`` and ``eval()`` calls so a
    test can assert how ``run`` restored the model. ``cpu_called`` is exposed
    for tests that only care whether the model was moved to CPU at all.
    """

    def __init__(self, quantization_config):
        self.config = FakeConfig(quantization_config)
        self.device = "cpu"
        self.cpu_calls = 0
        self.eval_calls = 0

    @property
    def cpu_called(self) -> bool:
        return self.cpu_calls > 0

    def cpu(self):
        self.device = "cpu"
        self.cpu_calls += 1
        return self

    def cuda(self):
        self.device = "cuda"
        return self

    def eval(self):
        self.eval_calls += 1
        return self


class PlainModelConfig:
    """A non-``RotatedModelConfig`` model_config stand-in."""

    def __init__(self, fp32_had=False):
        self.fp32_had = fp32_had


def make_rotated_model_config(fp32_had=False) -> RotatedModelConfig:
    """Build a ``RotatedModelConfig`` instance without touching the filesystem.

    ``RotatedModelConfig.__init__`` requires a real on-disk model path, so we
    bypass it: the consistency check only inspects ``isinstance`` and the
    ``fp32_had`` attribute.
    """
    model_config = object.__new__(RotatedModelConfig)
    model_config.fp32_had = fp32_had
    return model_config
