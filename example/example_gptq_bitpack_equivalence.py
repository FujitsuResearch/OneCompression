"""

Example: Compare GPTQ quantization with and without bitpack_on_quantize.

This script checks that, for each supported weight bit-width, the *dequantized*
weights are identical between two ``GPTQ.quantize_layer`` paths:

  - Unpacked result : ``bitpack_on_quantize=False``
  - Packed result   : ``bitpack_on_quantize=True``

Both modes quantize a fresh copy of the same layer using the same calibration
input and cloned Hessian.  The comparison therefore exercises the actual
``bitpack_on_quantize`` flag path, rather than a local hand-written packing
helper.

Because bit-packing only changes the *storage layout* of the already-quantized
integer weights (pack -> unpack is a lossless, bit-exact round-trip), the two
modes MUST produce bit-identical dequantized weights.  The expected result is an
exact match (``torch.equal``), NOT merely a close one -- any non-zero difference
indicates a mismatch in the packed path (for example qzeros v1 ``-1/+1`` offset
handling or shape normalization).

Note: bit-packing supports wbits in {2, 3, 4, 8} only.  Other bit-widths cannot
be packed, so they are skipped here (in those cases both modes share the same
unpacked code path anyway).

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import copy

import torch

from onecomp import setup_logger
from onecomp.quantizer.gptq import GPTQ

# Bit-widths that the GPTQ bit-packer supports.
SUPPORTED_WBITS = [2, 3, 4, 8]

# Quantization configurations to sweep.  Each entry is merged on top of the
# default GPTQ parameters; wbits is varied separately for every config.
CONFIGS = [
    {"groupsize": -1, "actorder": False, "sym": True},
    {"groupsize": -1, "actorder": False, "sym": False},
    {"groupsize": 128, "actorder": False, "sym": False},
    {"groupsize": 128, "actorder": True, "sym": False},
]

# Layer / calibration sizes (kept small so the demo runs fast on CPU).
IN_FEATURES = 512
OUT_FEATURES = 256
BATCH = 4
SEQ = 16
SEED = 123


def make_layer_and_input(device, dtype=torch.float32):
    """Create a reproducible linear layer and calibration input."""
    torch.manual_seed(SEED)
    layer = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, device=device, dtype=dtype)
    inp = torch.randn(BATCH, SEQ, IN_FEATURES, device=device, dtype=dtype)
    return layer, inp


def compare_modes(device, wbits, **params):
    """Quantize two layer copies and compare dequantized weights across modes.

    Returns a dict describing the comparison outcome.
    """
    layer, inp = make_layer_and_input(device)

    q_unpacked = GPTQ(wbits=wbits, bitpack_on_quantize=False, **params)
    q_packed = GPTQ(wbits=wbits, bitpack_on_quantize=True, **params)

    q_unpacked.validate_params()
    q_packed.validate_params()
    hessian = q_unpacked.calculate_hessian(layer, inp)
    result_normal = q_unpacked.quantize_layer(copy.deepcopy(layer), inp, hessian=hessian.clone())
    result_bitpack = q_packed.quantize_layer(copy.deepcopy(layer), inp, hessian=hessian.clone())

    # Sanity check: the two results really do take different storage paths.
    assert result_normal.qweight_is_packed is False
    assert result_bitpack.qweight_is_packed is True

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
        f"{'groupsize':>9} {'actorder':>8} {'sym':>5} {'wbits':>5} "
        f"{'exact':>6} {'max_diff':>10} {'mean_diff':>10} {'mismatch':>14}"
    )
    print(header)
    print("-" * len(header))

    all_exact = True
    for config in CONFIGS:
        for wbits in SUPPORTED_WBITS:
            res = compare_modes(device, wbits=wbits, **config)
            all_exact = all_exact and res["exact"]
            mismatch_str = f"{res['num_mismatch']}/{res['num_elements']}"
            print(
                f"{config['groupsize']:>9} {str(config['actorder']):>8} "
                f"{str(config['sym']):>5} {wbits:>5} "
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
            "pack/unpack path (check the qzeros v1 -1/+1 offset and shape handling)."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
