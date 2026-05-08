"""Regression test helper for quantize.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Helper module to ensure that results do not change before and after
refactoring of jointq.quantize. Calls a common execution function from both
the result saving script and the test code.

Functions
---------
run_quantize(data_path, device_id=0)
    Load weight matrix W and input matrix X from a .pth file, and
    run jointq.quantize with default parameters (bits=4, symmetric=False,
    group_size=128, batch_size=2048).
    Returns a dictionary containing scales, zero_point, integers_z (CPU Tensors)
    and mse (float).

save_result(result_dict, output_path)
    Save the return value of run_quantize to a .pth file via torch.save.

load_result(path)
    Load a dictionary from a .pth file saved by save_result via torch.load.

Usage (saving results)
----------------------
::

    python -m tests.jointq.regression_quantize_helper [-f DATA_PATH] [-d DEVICE_ID] [-o OUTPUT_PATH]

By default, uses data under tests/jointq/data/ and saves expected results to the same directory.
"""

from pathlib import Path

import torch

from onecomp.quantizer.jointq.core import quantize

DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_DATA_PATH = DATA_DIR / "model_layers_0_self_attn_k_proj.pth"

DEFAULT_EXPECTED_PATH = DATA_DIR / "quantize_regression_expected.pth"

# Expected MSE for regression testing.
# Update this value when the quantization algorithm is intentionally changed
# or when the baseline data is regenerated.
# Generated with: bits=4, symmetric=False, group_size=128, batch_size=2048
EXPECTED_MSE = 9.127208860704012e-06


def run_quantize(data_path, device_id=0):
    """Load data and run quantize with default parameters.

    Parameters
    ----------
    data_path : str or Path
        Path to the .pth file containing weight and activation data.
    device_id : int
        GPU device ID.

    Returns
    -------
    dict
        Dictionary with keys: scales, zero_point, integers_z (Tensors on CPU),
        and mse (float).
    """

    weight_and_activation = torch.load(data_path, weights_only=True)

    device = torch.device(device_id)

    matrix_W = weight_and_activation["W"].to("cpu").to(torch.float64)
    matrix_X = weight_and_activation["X"].to("cpu").to(torch.float64)

    if matrix_X.ndim == 3:
        matrix_X = matrix_X.reshape(-1, matrix_X.shape[-1])
    elif matrix_X.ndim != 2:
        raise ValueError(f"Unsupported matrix_X shape: {matrix_X.shape}")

    solution = quantize(
        matrix_W=matrix_W,
        matrix_X=matrix_X,
        bits=4,
        symmetric=False,
        group_size=128,
        batch_size=2048,
        device=device,
        log_level=2,
    )

    # Compute MSE: mean(||WX^T - hat_W X^T||^2)
    matrix_W_hat = solution.get_dequantized_weight_matrix()
    matrix_W_gpu = matrix_W.to(device)
    matrix_X_gpu = matrix_X.to(device)
    mse = float(torch.mean((matrix_W_gpu @ matrix_X_gpu.T - matrix_W_hat @ matrix_X_gpu.T) ** 2))

    return {
        "scales": solution.scales.cpu(),
        "zero_point": solution.zero_point.cpu(),
        "integers_z": solution.integers_z.cpu(),
        "mse": mse,
    }


def save_result(result_dict, output_path):
    """Save result dict to a .pth file via torch.save."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result_dict, output_path)
    print(f"Result saved to {output_path}")


def load_result(path):
    """Load result dict from a .pth file via torch.load."""

    return torch.load(path, weights_only=True)


# ---------------------------------------------------------------------------
# main: Script to save results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run quantize and save result.")
    parser.add_argument(
        "-f",
        "--file-path",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to the .pth data file",
    )
    parser.add_argument(
        "-d",
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(DEFAULT_EXPECTED_PATH),
        help="Output .pth path",
    )
    args = parser.parse_args()

    result = run_quantize(args.file_path, args.device_id)
    save_result(result, args.output)
