from unittest.mock import patch

import pytest
import torch

import onecomp.quantizer._quantizer as quantizer_module
from onecomp.quantizer._quantizer import Quantizer


class FakeQuantizer(Quantizer):
    def quantize_layer(self, *args, **kwargs):
        raise NotImplementedError


@pytest.fixture
def fake_quantizer():
    return FakeQuantizer()


def test_adjust_weight_succeeds_on_first_cholesky_attempt(fake_quantizer):
    module = torch.nn.Linear(2, 2, bias=False)
    original_solve = quantizer_module._safe_cholesky_and_solve

    with patch.object(
        quantizer_module,
        "_safe_cholesky_and_solve",
        wraps=original_solve,
    ) as solve:
        Quantizer.adjust_weight(
            fake_quantizer,
            module,
            quant_input_activation=None,
            original_input_activation=None,
            original_hessian=torch.eye(2),
            original_delta_hatX=torch.zeros(2, 2),
        )

    solve.assert_called_once()


def test_adjust_weight_retries_cholesky_with_increased_damping(fake_quantizer):
    module = torch.nn.Linear(2, 2, bias=False)

    # Set up a Hessian that is not positive definite, which will cause the first Cholesky attempt to fail.
    hessian = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, -0.015],
        ]
    )

    original_solve = quantizer_module._safe_cholesky_and_solve

    with patch.object(
        quantizer_module,
        "_safe_cholesky_and_solve",
        wraps=original_solve,
    ) as solve:
        Quantizer.adjust_weight(
            fake_quantizer,
            module,
            quant_input_activation=None,
            original_input_activation=None,
            original_hessian=hessian,
            original_delta_hatX=torch.zeros(2, 2),
        )

    assert solve.call_count > 1
