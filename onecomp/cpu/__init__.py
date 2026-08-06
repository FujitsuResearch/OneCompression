"""OneComp CPU inference and GGUF export (llama.cpp).

Layout:
    export/   GGUF export (blocks, checkpoint reader, dequantize, skeleton,
              direct lossless path, dequantize fallback path)
    eval/     CPU-side evaluation (inspect, perplexity, parity, benchmark)
    inference LlamaCppModel CPU text generation
    cli       ``onecomp-gguf`` command line interface

Public API:
    export_to_gguf        -- single entry: routes any supported checkpoint to GGUF
    convert_gptq_to_gguf  -- direct, lossless GPTQ -> GGUF (preserves QEP codes)
    export_via_dequantize -- fallback: dequantize -> convert -> llama-quantize
    dequantize_to_hf      -- reconstruct a dense fp16 HF model (GPTQ/DBF/rotated)
    LlamaCppModel         -- CPU text generation on a GGUF model
    inspect_gguf          -- per-tensor quant types / size / effective bit-width
    perplexity            -- CPU perplexity of a GGUF model on text
    benchmark             -- CPU prefill / decode throughput
    teacher_forced_parity -- HF (PyTorch) vs GGUF (llama.cpp) output agreement
    serve                 -- one-command OpenAI-compatible CPU server
    resolve_to_gguf       -- resolve a path / packed checkpoint to a ready GGUF
"""

from onecomp.cpu.eval.benchmark import benchmark
from onecomp.cpu.eval.inspect_gguf import inspect_gguf
from onecomp.cpu.eval.parity import teacher_forced_parity
from onecomp.cpu.eval.perplexity import perplexity
from onecomp.cpu.export.auto import export_to_gguf, plan_export
from onecomp.cpu.export.dequantize import dequantize_to_hf
from onecomp.cpu.export.direct import convert_gptq_to_gguf
from onecomp.cpu.export.fallback import export_via_dequantize
from onecomp.cpu.inference import LlamaCppModel
from onecomp.cpu.serve import resolve_to_gguf, serve

__all__ = [
    "export_to_gguf",
    "plan_export",
    "convert_gptq_to_gguf",
    "export_via_dequantize",
    "dequantize_to_hf",
    "LlamaCppModel",
    "inspect_gguf",
    "perplexity",
    "benchmark",
    "teacher_forced_parity",
    "serve",
    "resolve_to_gguf",
]
