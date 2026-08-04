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
        default="cuda:0",
        help="device to place the model on (default: cuda:0)",
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
        "--format",
        choices=("onecomp", "gguf"),
        default="onecomp",
        help="save format: onecomp (safetensors, default) or gguf "
        "(additionally export a GGUF F16 file into the save directory)",
    )
    parser.add_argument(
        "--push-to-hub",
        metavar="REPO_ID",
        default=None,
        help="push the save directory to the Hugging Face Hub "
        "as a private repository (e.g. user/model-name)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args()

    save_dir = None if args.save_dir.lower() == "none" else args.save_dir

    if (save_dir is None or args.save_dir.lower() == "auto") and (
        args.format == "gguf" or args.push_to_hub
    ):
        parser.error("--format gguf and --push-to-hub require an explicit --save-dir")

    # Lazy import to keep --help fast
    from .runner import Runner  # pylint: disable=import-outside-toplevel

    runner = Runner.auto_run(
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

    if save_dir is None or args.save_dir.lower() == "auto":
        return

    if args.format == "gguf":
        import os  # pylint: disable=import-outside-toplevel
        import shutil  # pylint: disable=import-outside-toplevel
        import tempfile  # pylint: disable=import-outside-toplevel

        from .export import (  # pylint: disable=import-outside-toplevel
            GGUFExportConfig,
            export_gguf,
        )

        save_name = os.path.basename(os.path.normpath(save_dir))
        out_path = os.path.join(save_dir, f"{save_name}-f16.gguf")
        # The save directory contains packed quantized tensors, which the
        # GGUF F16 export cannot consume.  Save dequantized FP16 weights
        # to a temporary directory and convert from there.
        parent_dir = os.path.dirname(os.path.abspath(save_dir))
        tmp_dir = tempfile.mkdtemp(prefix="onecomp-gguf-", dir=parent_dir)
        try:
            runner.save_dequantized_model(tmp_dir)
            export_gguf(tmp_dir, GGUFExportConfig(out_path=out_path, name=save_name))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    if args.push_to_hub:
        from .export import (  # pylint: disable=import-outside-toplevel
            generate_model_card,
            push_to_hub,
        )

        recipe = {
            "method": "AutoBit + QEP" if not args.no_qep else "AutoBit",
            "wbits": args.wbits if args.wbits is not None else "auto",
            "groupsize": args.groupsize,
        }
        card = generate_model_card(args.model_id, recipe=recipe, results={})
        push_to_hub(save_dir, args.push_to_hub, model_card=card)
