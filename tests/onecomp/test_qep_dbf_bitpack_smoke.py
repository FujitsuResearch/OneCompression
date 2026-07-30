"""QEP smoke test for DBF quantize-time bitpacking.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest
import torch

from onecomp import QEPConfig
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


def test_qep_runner_accepts_packed_dbf_results():
    """Generic QEP can update model weights from packed DBFResult."""
    quantizer = make_dbf(num_layers=1, calc_quant_error=True)
    results = run_runner(
        quantizer,
        calibration_config(),
        qep=True,
        qep_config=QEPConfig(general=True),
    )

    assert_packed_dbf_results(results)
    for result in results.values():
        assert_error_metrics(result)
