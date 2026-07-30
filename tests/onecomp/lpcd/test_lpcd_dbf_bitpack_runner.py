"""LPCD smoke test for DBF quantize-time bitpacking.

Copyright 2025-2026 Fujitsu Ltd.
"""

import pytest
import torch

from onecomp import LPCDConfig, QEPConfig
from tests.onecomp.quantizer.dbf.dbf_bitpack_runner_helpers import (
    assert_packed_dbf_results,
    calibration_config,
    make_dbf,
    run_runner,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_lpcd_runner_accepts_packed_dbf_results():
    """LPCD can refine/project layers when DBF stores packed results."""
    quantizer = make_dbf(num_layers=7)
    results = run_runner(
        quantizer,
        calibration_config(),
        qep=True,
        qep_config=QEPConfig(general=False),
        lpcd=True,
        lpcd_config=LPCDConfig(),
    )

    assert_packed_dbf_results(results)
