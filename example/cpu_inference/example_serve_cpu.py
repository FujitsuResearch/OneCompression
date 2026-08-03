"""Serve a packed OneComp checkpoint (or GGUF) on CPU with one command.

The barrier-free path: drop in a *packed quantized* model directory and anyone
can run it on CPU behind an OpenAI-compatible API. If given a packed OneComp
checkpoint, the GGUF is auto-exported (losslessly, cached) on first launch.

Run:
    # Packed OneComp GPTQ checkpoint (auto-exports to GGUF once):
    python example/cpu_inference/example_serve_cpu.py ./TinyLlama-1.1B-gptq-4bit
    # ...or an existing GGUF:
    python example/cpu_inference/example_serve_cpu.py ./model.gguf

Then, from another shell:
    curl http://localhost:8080/v1/chat/completions \
      -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'

Equivalent CLI: onecomp-gguf serve --model <path> --port 8080

Requires: pip install 'onecomp[llamacpp]'  (no FastAPI/uvicorn needed)

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import sys

from onecomp.cpu import serve
from onecomp.log import setup_logger


def main():
    setup_logger()
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./TinyLlama-1.1B-gptq-4bit"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
    # host=0.0.0.0 to expose on the network; chat_format is auto-detected from
    # the GGUF architecture when no chat template is embedded.
    serve(model_path, host="0.0.0.0", port=port, n_ctx=2048)


if __name__ == "__main__":
    main()
