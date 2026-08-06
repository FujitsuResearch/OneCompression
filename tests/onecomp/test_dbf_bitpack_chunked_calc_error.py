"""Chunked calibration smoke test for DBF quantize-time bitpacking.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest
import torch

from tests.onecomp.quantizer.dbf.dbf_bitpack_runner_helpers import (
    assert_error_metrics,
    assert_packed_dbf_results,
    calibration_config,
    make_dbf,
    run_runner,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_chunked_calc_quant_error_accepts_packed_dbf_results():
    """Chunked calibration error recording can dequantize packed DBFResult."""
    quantizer = make_dbf(num_layers=1, calc_quant_error=True)
    results = run_runner(
        quantizer,
        calibration_config(batch_size=4),
        qep=False,
    )

    assert_packed_dbf_results(results)
    for result in results.values():
        assert_error_metrics(result)
