"""

Example: Compare DBF quantization with and without bitpack_on_quantize.

This script checks that, for each DBF configuration, the *dequantized* weights
are identical between two ``DBF.quantize_layer`` paths:

  - Unpacked result : ``bitpack_on_quantize=False`` (keep ``dbf_A``/``dbf_B`` as
    unpacked +/-1 float16 matrices)
  - Packed result   : ``bitpack_on_quantize=True`` (pack ``dbf_A``/``dbf_B`` into
    uint8 immediately after the module is quantized)

DBF's binary factorization is stochastic, so both modes are seeded identically
before each ``quantize_layer`` call and quantize a fresh copy of the same layer
with the same calibration input and Hessian.  This guarantees the two modes
produce the *same* DBF factors; the only difference left is the storage layout.

Because bit-packing only changes the storage layout of the already-quantized
+/-1 binary factors (pack -> unpack is a lossless, bit-exact round-trip), the
two modes MUST produce bit-identical dequantized weights.  The expected result
is an exact match (``torch.equal``), NOT merely a close one -- any non-zero
difference indicates a mismatch in the pack/unpack path (for example a wrong
original-shape or padding handling).

Note: DBF always factorizes weights into +/-1 (1-bit) matrices, so bit-packing
applies to every configuration regardless of ``target_bits``.

Copyright 2025-2026 Fujitsu Ltd.

"""

import copy
import os

import torch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from onecomp import setup_logger
from onecomp.quantizer.dbf import DBF

# Target bit-widths to sweep.
TARGET_BITS = [1.0, 1.5, 2.0]

# Quantization configurations to sweep.  Each entry is merged on top of the
# default DBF parameters; target_bits is varied separately for every config.
CONFIGS = [
    {"iters": 1, "balance_iters": 1, "use_balancing": True},
    {"iters": 2, "balance_iters": 1, "use_balancing": True},
    {"iters": 2, "balance_iters": 2, "use_balancing": True},
    {"iters": 2, "balance_iters": 1, "use_balancing": False},
]

# Layer / calibration sizes (kept small so the demo runs fast on CPU).
IN_FEATURES = 512
OUT_FEATURES = 256
BATCH = 4
SEQ = 16
SEED = 123


def _seed(seed):
    """Seed torch (and CUDA) so DBF's stochastic optimization is reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_layer_and_input(device, dtype=torch.float32):
    """Create a reproducible linear layer and calibration input."""
    torch.manual_seed(SEED)
    layer = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, device=device, dtype=dtype)
    inp = torch.randn(BATCH, SEQ, IN_FEATURES, device=device, dtype=dtype)
    return layer, inp


def compare_modes(device, target_bits, **params):
    """Quantize two layer copies and compare dequantized weights across modes.

    Returns a dict describing the comparison outcome.
    """
    layer, inp = make_layer_and_input(device)

    q_unpacked = DBF(target_bits=target_bits, bitpack_on_quantize=False, **params)
    q_packed = DBF(target_bits=target_bits, bitpack_on_quantize=True, **params)

    q_unpacked.validate_params()
    q_packed.validate_params()

    hessian = q_unpacked.calculate_hessian(layer, inp)

    _seed(SEED)
    result_bitpack = q_packed.quantize_layer(copy.deepcopy(layer), inp, hessian=hessian.clone())
    _seed(SEED)
    result_normal = q_unpacked.quantize_layer(copy.deepcopy(layer), inp, hessian=hessian.clone())

    # Sanity check: the two results really do take different storage paths.
    assert result_normal.dbf_A_is_packed is False
    assert result_normal.dbf_B_is_packed is False
    assert result_bitpack.dbf_A_is_packed is True
    assert result_bitpack.dbf_B_is_packed is True

    w_normal = result_normal.compute_dequantized_weight()
    w_bitpack = result_bitpack.compute_dequantized_weight()

    exact = torch.equal(w_normal, w_bitpack)
    diff = (w_normal.float() - w_bitpack.float()).abs()
    num_mismatch = int((diff > 0).sum().item())

    return {
        "exact": exact,
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "num_mismatch": num_mismatch,
        "num_elements": diff.numel(),
    }


def main():
    setup_logger()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Deterministic algorithms so repeated runs agree exactly.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Device: {device}")
    print(f"Layer : Linear(in={IN_FEATURES}, out={OUT_FEATURES})\n")

    header = (
        f"{'iters':>5} {'bal_it':>6} {'balance':>7} {'tbits':>5} "
        f"{'exact':>6} {'max_diff':>10} {'mean_diff':>10} {'mismatch':>14}"
    )
    print(header)
    print("-" * len(header))

    all_exact = True
    for config in CONFIGS:
        for target_bits in TARGET_BITS:
            res = compare_modes(device, target_bits=target_bits, **config)
            all_exact = all_exact and res["exact"]
            mismatch_str = f"{res['num_mismatch']}/{res['num_elements']}"
            print(
                f"{config['iters']:>5} {config['balance_iters']:>6} "
                f"{str(config['use_balancing']):>7} {target_bits:>5} "
                f"{('OK' if res['exact'] else 'DIFF'):>6} "
                f"{res['max_abs_diff']:>10.3e} {res['mean_abs_diff']:>10.3e} "
                f"{mismatch_str:>14}"
            )

    print("-" * len(header))
    if all_exact:
        print(
            "\n[PASS] All configurations match bit-exactly: per-module bitpacking "
            "does not change the dequantized weights."
        )
    else:
        print(
            "\n[FAIL] Some configurations differ. Bit-packing is expected to be a "
            "lossless storage transform, so any difference indicates a bug in the "
            "pack/unpack path (check the original-shape and padding handling)."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
