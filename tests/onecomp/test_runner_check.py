"""Unit tests for ``Runner.check()`` parameter validation.

These tests exercise only the configuration validation path
(``Runner.check()``) and therefore do not require a GPU or model
loading.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest

from onecomp import CalibrationConfig, ModelConfig, Runner
from onecomp.quantizer.autobit import AutoBitQuantizer
from onecomp.quantizer.gptq import GPTQ
from onecomp.quantizer.jointq import JointQ


def _model_config() -> ModelConfig:
    return ModelConfig(
        model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        device="cuda:0",
    )


class TestRunnerCheckQEPSupport:
    """``qep=True`` must raise a clear error when the quantizer does
    not support the generic QEP pipeline.
    """

    def test_jointq_with_qep_true_raises_clear_error(self):
        """JointQ + qep=True should raise a clear ValueError, not an
        obscure error from deep inside the QEP runtime.
        """
        runner = Runner(
            model_config=_model_config(),
            quantizer=JointQ(bits=4, group_size=128),
            calibration_config=CalibrationConfig(max_length=128, num_calibration_samples=8),
            qep=True,
        )

        with pytest.raises(ValueError, match=r"JointQ.*does not support QEP"):
            runner.check()

    def test_jointq_with_qep_false_passes_check(self):
        """JointQ + qep=False is the supported configuration."""
        runner = Runner(
            model_config=_model_config(),
            quantizer=JointQ(bits=4, group_size=128),
            calibration_config=CalibrationConfig(max_length=128, num_calibration_samples=8),
            qep=False,
        )
        runner.check()

    def test_gptq_with_qep_true_passes_check(self):
        """GPTQ supports QEP, so check() must pass."""
        runner = Runner(
            model_config=_model_config(),
            quantizer=GPTQ(wbits=4, groupsize=128),
            calibration_config=CalibrationConfig(max_length=128, num_calibration_samples=8),
            qep=True,
        )
        runner.check()

    def test_autobit_with_jointq_candidate_and_qep_true_raises(self):
        """AutoBit with a JointQ candidate must also raise on qep=True.

        ``AutoBitQuantizer.flag_qep_supported`` is True only when *all*
        candidate quantizers support QEP, so a JointQ candidate must
        propagate the unsupported state.
        """
        autobit = AutoBitQuantizer(
            quantizers=[GPTQ(wbits=4), JointQ(bits=2)],
            target_bit=3.0,
        )
        runner = Runner(
            model_config=_model_config(),
            quantizer=autobit,
            calibration_config=CalibrationConfig(max_length=128, num_calibration_samples=8),
            qep=True,
        )

        with pytest.raises(
            ValueError,
            match=r"AutoBitQuantizer.*does not support QEP",
        ):
            runner.check()

    def test_autobit_with_only_gptq_candidates_and_qep_true_passes(self):
        """AutoBit with only QEP-compatible candidates must pass."""
        autobit = AutoBitQuantizer(
            quantizers=[GPTQ(wbits=4), GPTQ(wbits=2)],
            target_bit=3.0,
        )
        runner = Runner(
            model_config=_model_config(),
            quantizer=autobit,
            calibration_config=CalibrationConfig(max_length=128, num_calibration_samples=8),
            qep=True,
        )
        runner.check()


class TestRunnerCheckMPS:
    """MPS-specific validation in ``Runner.check()``."""

    @staticmethod
    def _mps_model_config() -> ModelConfig:
        return ModelConfig(
            model_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            device="mps",
        )

    def test_autobit_dbf_only_target_on_mps_raises(self):
        """Ultra-low target_bit triggers DBF-only path; must fail on MPS."""
        autobit = AutoBitQuantizer(
            quantizers=[GPTQ(wbits=4), GPTQ(wbits=2)],
            target_bit=1.5,
            auto_dbf=True,
        )
        runner = Runner(
            model_config=self._mps_model_config(),
            quantizer=autobit,
            calibration_config=CalibrationConfig(max_length=128, num_calibration_samples=8),
        )

        with pytest.raises(ValueError, match=r"DBF fallback is not supported on MPS"):
            runner.check()

    def test_autobit_low_target_on_mps_with_auto_dbf_disabled_passes(self):
        """auto_dbf=False skips DBF fallback even for low target_bit."""
        autobit = AutoBitQuantizer(
            quantizers=[GPTQ(wbits=4), GPTQ(wbits=2)],
            target_bit=1.5,
            auto_dbf=False,
        )
        runner = Runner(
            model_config=self._mps_model_config(),
            quantizer=autobit,
            calibration_config=CalibrationConfig(max_length=128, num_calibration_samples=8),
        )
        runner.check()
