"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Command:

```bash
pytest -v -s --log-cli-level=INFO test_quantizer.py
```


"""

from unittest import TestCase

import torch

from onecomp.quantizer.jointq.core import Quantizer


def make_sample_data():
    """Make sample data for testing."""

    device = torch.device("cpu")

    # Define the size of the dataset
    row_size = 10
    col_size = 6

    # Generate random data
    torch.manual_seed(0)
    matrix_X = torch.rand(row_size, col_size).to(torch.float64).to(device)

    # Generate weight matrix
    matrix_W = torch.tensor(
        [
            [1.2, 0.85, -0.9, -0.58, 0.38, 0.18],
            [10.1, -11.8, 12.0, 3.5, -2.5, -1.5],
        ],
        dtype=torch.float64,
    ).to(device)

    return matrix_X, matrix_W


class TestQuantizer(TestCase):
    """Test the Quantizer class."""

    # TODO: add tests
    pass
