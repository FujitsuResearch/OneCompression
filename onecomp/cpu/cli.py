"""Command line interface for OneComp GGUF export, CPU inference and evaluation.

Examples:
    # Direct, lossless GPTQ -> GGUF (preserves QEP-corrected codes):
    onecomp-gguf export --quantized-dir ./model-gptq-4bit --out ./model.gguf

    # Per-module mixed precision (4/8-bit lossless + 2/3-bit K-quant):
    onecomp-gguf export --quantized-dir ./model-mixed --out ./model.gguf --mode mixed

    # Fallback (dequantize then llama-quantize to Q4_K_M):
    onecomp-gguf export --quantized-dir ./model --out ./model.gguf \
        --mode dequantize --qtype Q4_K_M

    # CPU inference:
    onecomp-gguf run --gguf ./model.gguf --prompt "Fujitsu is"

    # Inspect per-tensor quant types / size:
    onecomp-gguf inspect --gguf ./model.gguf

    # CPU perplexity / throughput:
    onecomp-gguf ppl --gguf ./model.gguf --text-file ./wiki.txt
    onecomp-gguf bench --gguf ./model.gguf --gen-tokens 64

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import argparse
import json
import sys
from logging import getLogger

logger = getLogger(__name__)


def _add_export_parser(sub):
    p = sub.add_parser("export", help="Export an OneComp quantized model to GGUF.")
    p.add_argument("--quantized-dir", required=True, help="OneComp quantized model directory.")
    p.add_argument("--out", required=True, help="Output .gguf path.")
    p.add_argument(
        "--mode",
        choices=["auto", "direct", "mixed", "dequantize", "fallback"],
        default="auto",
        help=(
            "direct: lossless GPTQ->GGUF block packing (uniform 4/8-bit). "
            "mixed: per-module mixed precision (4/8-bit lossless + 2/3-bit K-quant). "
            "dequantize: fp16->llama-quantize fallback."
        ),
    )
    p.add_argument("--qtype", default=None, help="llama-quantize type for dequantize mode.")
    p.add_argument("--original-model", default=None, help="Original FP model dir for skeleton.")
    p.add_argument("--work-dir", default=None, help="Scratch directory.")
    return p


def _add_run_parser(sub):
    p = sub.add_parser("run", help="Run CPU inference on a GGUF model.")
    p.add_argument("--gguf", required=True, help="Path to a .gguf model.")
    p.add_argument("--prompt", required=True, help="Prompt text.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--n-threads", type=int, default=None)
    p.add_argument("--stream", action="store_true", help="Stream tokens as they are generated.")
    return p


def _add_inspect_parser(sub):
    p = sub.add_parser("inspect", help="Show per-tensor quant types / size of a GGUF.")
    p.add_argument("--gguf", required=True, help="Path to a .gguf model.")
    p.add_argument("--json", action="store_true", help="Emit the aggregate summary as JSON.")
    p.add_argument("--max-blocks", type=int, default=4, help="Per-block rows to print.")
    return p


def _add_ppl_parser(sub):
    p = sub.add_parser("ppl", help="Compute CPU perplexity of a GGUF model on text.")
    p.add_argument("--gguf", required=True, help="Path to a .gguf model.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Raw text to score.")
    g.add_argument("--text-file", help="File whose contents are scored.")
    p.add_argument("--n-ctx", type=int, default=512)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--n-threads", type=int, default=None)
    return p


def _add_bench_parser(sub):
    p = sub.add_parser("bench", help="Benchmark CPU prefill / decode throughput.")
    p.add_argument("--gguf", required=True, help="Path to a .gguf model.")
    p.add_argument("--prompt", default="Fujitsu is a Japanese multinational company that")
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--n-threads", type=int, default=None)
    p.add_argument("--runs", type=int, default=1)
    return p


def _add_serve_parser(sub):
    p = sub.add_parser(
        "serve",
        help="Serve a GGUF or packed OneComp checkpoint over an OpenAI-compatible API.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="A .gguf file, a dir containing one, or an OneComp quantized checkpoint "
        "(auto-exported to GGUF on first use).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--n-ctx", type=int, default=4096)
    p.add_argument("--n-threads", type=int, default=None)
    p.add_argument(
        "--mode",
        choices=["auto", "direct", "mixed", "fallback"],
        default="auto",
        help="Export mode when the input is a packed checkpoint.",
    )
    p.add_argument("--original-model", default=None, help="FP model dir for skeleton metadata.")
    p.add_argument("--chat-format", default=None, help="Override llama.cpp chat format.")
    return p


def _cmd_export(args) -> int:
    # auto / direct / mixed / fallback all flow through the single entry point;
    # "dequantize" is kept as an alias of "fallback" for backwards compatibility.
    if args.mode in ("auto", "direct", "mixed", "fallback", "dequantize"):
        from onecomp.cpu.export.auto import export_to_gguf

        mode = "fallback" if args.mode == "dequantize" else args.mode
        summary = export_to_gguf(
            quantized_dir=args.quantized_dir,
            out_gguf=args.out,
            mode=mode,
            qtype=args.qtype or "Q4_K_M",
            original_model=args.original_model,
            work_dir=args.work_dir,
        )
        print(json.dumps(summary, indent=2))
    else:  # pragma: no cover - argparse restricts choices
        from onecomp.cpu.export.fallback import export_via_dequantize

        out = export_via_dequantize(
            quantized_dir=args.quantized_dir,
            out_gguf=args.out,
            qtype=args.qtype,
            work_dir=args.work_dir,
        )
        print(f"Wrote {out}")
    return 0


def _cmd_run(args) -> int:
    from onecomp.cpu.inference import LlamaCppModel

    model = LlamaCppModel(args.gguf, n_ctx=args.n_ctx, n_threads=args.n_threads)
    if args.stream:
        for piece in model.stream(
            args.prompt, max_tokens=args.max_tokens, temperature=args.temperature
        ):
            print(piece, end="", flush=True)
        print()
    else:
        print(
            model.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        )
    return 0


def _cmd_inspect(args) -> int:
    from onecomp.cpu.eval.inspect_gguf import format_report, inspect_gguf

    report = inspect_gguf(args.gguf)
    if args.json:
        print(json.dumps(report.summary(), indent=2))
    else:
        print(format_report(report, max_block_rows=args.max_blocks))
    return 0


def _cmd_ppl(args) -> int:
    from onecomp.cpu.eval.perplexity import perplexity

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text
    result = perplexity(
        args.gguf, text, n_ctx=args.n_ctx, stride=args.stride, n_threads=args.n_threads
    )
    print(
        json.dumps(
            {
                "perplexity": round(result.perplexity, 4),
                "nll": round(result.nll, 4),
                "n_tokens": result.n_tokens,
                "n_windows": result.n_windows,
            },
            indent=2,
        )
    )
    return 0


def _cmd_bench(args) -> int:
    from onecomp.cpu.eval.benchmark import benchmark

    results = benchmark(
        args.gguf,
        prompt=args.prompt,
        gen_tokens=args.gen_tokens,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        runs=args.runs,
    )
    print(json.dumps([r.summary() for r in results], indent=2))
    return 0


def _cmd_serve(args) -> int:
    from onecomp.cpu.serve import serve

    serve(
        args.model,
        host=args.host,
        port=args.port,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        mode=args.mode,
        original_model=args.original_model,
        chat_format=args.chat_format,
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="onecomp-gguf", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_export_parser(sub)
    _add_run_parser(sub)
    _add_inspect_parser(sub)
    _add_ppl_parser(sub)
    _add_bench_parser(sub)
    _add_serve_parser(sub)
    args = parser.parse_args(argv)

    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    handlers = {
        "export": _cmd_export,
        "run": _cmd_run,
        "inspect": _cmd_inspect,
        "ppl": _cmd_ppl,
        "bench": _cmd_bench,
        "serve": _cmd_serve,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
