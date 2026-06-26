"""Shared helpers for DBF bitpack Runner smoke tests.

Copyright 2025-2026 Fujitsu Ltd.
"""

import gc
import math

import torch

from onecomp import CalibrationConfig, DBF, LPCDConfig, ModelConfig, QEPConfig, Runner, setup_logger

MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def calibration_config(*, batch_size: int | None = None) -> CalibrationConfig:
    """Small calibration config for DBF runner smoke tests."""
    return CalibrationConfig(
        num_calibration_samples=8,
        max_length=128,
        strategy="drop_rand",
        seed=0,
        batch_size=batch_size,
        num_layers_per_group=1,
    )


def make_dbf(*, num_layers: int, calc_quant_error: bool = False) -> DBF:
    """Create a lightweight DBF quantizer that still uses the packed path."""
    return DBF(
        target_bits=1.0,
        iters=1,
        balance_iters=1,
        num_layers=num_layers,
        calc_quant_error=calc_quant_error,
        bitpack_on_quantize=True,
    )


def run_runner(
    quantizer: DBF,
    calib_config: CalibrationConfig,
    *,
    qep: bool = False,
    qep_config: QEPConfig | None = None,
    lpcd: bool = False,
    lpcd_config: LPCDConfig | None = None,
) -> dict:
    """Run a DBF Runner smoke configuration and return ``quantizer.results``."""
    setup_logger()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calib_config,
        qep=qep,
        qep_config=qep_config,
        lpcd=lpcd,
        lpcd_config=lpcd_config,
    )
    runner.run()

    results = quantizer.results

    del runner, quantizer, model_config
    gc.collect()
    torch.cuda.empty_cache()

    return results


def assert_packed_dbf_results(results: dict):
    """Verify every result is a packed DBF result and can dequantize."""
    assert results
    for name, result in results.items():
        assert result.is_dbf_quantized is True, f"{name}: DBF quantization did not run"
        assert result.dbf_A_is_packed is True, f"{name}: dbf_A is not packed"
        assert result.dbf_B_is_packed is True, f"{name}: dbf_B is not packed"
        assert result.dbf_A.dtype == torch.uint8, f"{name}: dbf_A is not uint8"
        assert result.dbf_B.dtype == torch.uint8, f"{name}: dbf_B is not uint8"

        weight = result.compute_dequantized_weight()
        assert weight.dtype == torch.float16, f"{name}: dequantized weight dtype mismatch"
        assert weight.device == torch.device("cpu"), f"{name}: dequantized weight is not on CPU"
        assert torch.isfinite(weight).all(), f"{name}: dequantized weight has NaN/Inf"
        assert weight.abs().sum().item() > 0, f"{name}: dequantized weight is all zeros"


def assert_error_metrics(result):
    """Verify calc_quant_error metrics were populated with finite values."""
    metric_names = (
        "output_squared_error",
        "mean_output_squared_error",
        "weight_squared_error",
        "mean_weight_squared_error",
        "relative_output_squared_error",
        "relative_weight_squared_error",
    )
    for metric_name in metric_names:
        value = getattr(result, metric_name)
        assert value is not None, f"{metric_name} was not populated"
        assert math.isfinite(value), f"{metric_name} is not finite: {value!r}"
