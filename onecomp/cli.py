"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import argparse

from .__version__ import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="onecomp",
        description="OneComp: One-liner LLM quantization (AutoBit + QEP)",
    )
    parser.add_argument(
        "model_id",
        help="Hugging Face model ID or local path",
    )
    parser.add_argument(
        "--wbits",
        type=float,
        default=None,
        help="target bitwidth (default: auto-estimated from VRAM)",
    )
    parser.add_argument(
        "--total-vram-gb",
        type=float,
        default=None,
        help="VRAM budget in GB for bitwidth estimation (default: auto-detect)",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="GPTQ group size (default: 128, -1 to disable)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device to place the model on (default: auto-detect CUDA -> MPS -> CPU)",
    )
    parser.add_argument(
        "--no-qep",
        action="store_true",
        help="disable QEP (enabled by default)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="skip perplexity and accuracy evaluation",
    )
    parser.add_argument(
        "--eval-original",
        action="store_true",
        help="also evaluate the original (unquantized) model",
    )
    parser.add_argument(
        "--save-dir",
        default="auto",
        help='save directory (default: auto-generated, "none" to skip)',
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    save_dir = None if args.save_dir.lower() == "none" else args.save_dir

    # Lazy import to keep --help fast
    from .runner import Runner  # pylint: disable=import-outside-toplevel

    Runner.auto_run(
        model_id=args.model_id,
        wbits=args.wbits,
        total_vram_gb=args.total_vram_gb,
        groupsize=args.groupsize,
        device=args.device,
        qep=not args.no_qep,
        evaluate=not args.no_eval,
        eval_original_model=args.eval_original,
        save_dir=save_dir,
    )
