"""One-command CPU serving for OneComp models — OpenAI-compatible, no extra deps.

The goal is zero friction: point ``serve`` at *either* a ``.gguf`` file or an
OneComp **packed** quantized checkpoint directory and it will

  1. resolve the input to a GGUF (auto-exporting a packed GPTQ/mixed checkpoint
     to a cached ``.gguf`` the first time, with no re-quantization), and
  2. start an OpenAI-compatible HTTP server (``/v1/models``,
     ``/v1/completions``, ``/v1/chat/completions``, with SSE streaming).

The server uses only the Python standard library plus ``llama-cpp-python`` (no
FastAPI/uvicorn), so any environment that can run inference can also serve.

Example::

    onecomp-gguf serve --model ./model-gptq-4bit --port 8080
    curl http://localhost:8080/v1/chat/completions -d \
      '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from glob import glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging import getLogger
from typing import Optional

logger = getLogger(__name__)


def _is_onecomp_checkpoint(path: str) -> bool:
    """True if ``path`` is an OneComp quantized checkpoint dir (has quant config)."""
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        with open(cfg, encoding="utf-8") as f:
            return json.load(f).get("quantization_config") is not None
    except (OSError, ValueError):
        return False


def _checkpoint_mtime(checkpoint_dir: str) -> float:
    """Latest modification time among config.json and safetensors shards."""
    latest = 0.0
    cfg = os.path.join(checkpoint_dir, "config.json")
    if os.path.isfile(cfg):
        latest = max(latest, os.path.getmtime(cfg))
    for shard in glob(os.path.join(checkpoint_dir, "*.safetensors")):
        latest = max(latest, os.path.getmtime(shard))
    return latest


def resolve_to_gguf(
    model_path: str,
    mode: str = "auto",
    original_model: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> str:
    """Resolve a path to a GGUF file, auto-exporting a packed checkpoint if needed.

    Args:
        model_path: a ``.gguf`` file, or a directory that is either an OneComp
            quantized checkpoint or already contains a ``.gguf``.
        mode: export mode for packed checkpoints: ``auto`` (routes via
            :func:`onecomp.cpu.export.auto.plan_export`), ``direct``, ``mixed``,
            or ``fallback``.
        original_model: optional original FP model dir for skeleton metadata
            (recommended for multimodal/exotic architectures).
        cache_path: where to write the exported GGUF (defaults next to the
            checkpoint as ``onecomp-cpu.gguf``).

    Returns:
        Path to a ready-to-load ``.gguf`` file.
    """
    if model_path.endswith(".gguf"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        return model_path

    if os.path.isdir(model_path):
        existing = sorted(glob(os.path.join(model_path, "*.gguf")))
        if len(existing) > 1:
            raise ValueError(
                f"Multiple .gguf files found in {model_path!r}: {existing[:4]}. "
                "Pass an explicit .gguf path or keep only one file in the directory."
            )
        if len(existing) == 1:
            logger.info("Using existing GGUF in %s: %s", model_path, existing[0])
            return existing[0]

        if _is_onecomp_checkpoint(model_path):
            cache_path = cache_path or os.path.join(model_path, "onecomp-cpu.gguf")
            ckpt_mtime = _checkpoint_mtime(model_path)
            if os.path.isfile(cache_path):
                if os.path.getmtime(cache_path) >= ckpt_mtime:
                    logger.info("Reusing cached export %s", cache_path)
                    return cache_path
                logger.info(
                    "Checkpoint newer than cached GGUF (%s); re-exporting to %s",
                    model_path,
                    cache_path,
                )
            # Route every supported family (gptq/mixed/jointq/rtn/dbf/autobit and
            # rotated variants) through the single export entry point.
            from onecomp.cpu.export.auto import export_to_gguf

            summary = export_to_gguf(
                model_path, cache_path, mode=mode, original_model=original_model
            )
            logger.info(
                "Auto-exported packed checkpoint -> %s (%s)", cache_path, summary.get("path")
            )
            return cache_path

    raise ValueError(
        f"Cannot serve {model_path!r}: expected a .gguf file, a directory with a "
        ".gguf, or an OneComp quantized checkpoint (config.json with "
        "quantization_config). Plain (unquantized) HF models must be quantized first."
    )


class _OpenAIHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible request handler (set ``server.engine``)."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # pragma: no cover - quieten default logging
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # -- helpers ---------------------------------------------------------
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg, status=400):
        self._send_json({"error": {"message": msg, "type": "invalid_request_error"}}, status)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def _begin_stream(self):
        # No Content-Length for SSE: signal end-of-body by closing the connection.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def _sse_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # -- routes ----------------------------------------------------------
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send_json(
                {"object": "list", "data": [{"id": self.server.model_id, "object": "model"}]}
            )
        elif self.path.rstrip("/") in ("/health", "/healthz"):
            self._send_json({"status": "ok"})
        else:
            self._send_error("not found", 404)

    def do_POST(self):
        try:
            payload = self._read_body()
        except ValueError:
            return self._send_error("invalid JSON body")

        route = self.path.rstrip("/")
        if route == "/v1/chat/completions":
            if not self.server.chat_supported:
                return self._send_error(
                    "Could not infer chat format for this model's architecture. "
                    "Specify it explicitly with --chat-template when serving. "
                    f"Known architectures: {', '.join(sorted(_ARCH_CHAT_FORMAT))}.",
                    400,
                )
            self._handle(payload, chat=True)
        elif route == "/v1/completions":
            self._handle(payload, chat=False)
        else:
            self._send_error("not found", 404)

    def _handle(self, payload, chat: bool):
        engine = self.server.engine
        stream = bool(payload.get("stream", False))
        created = int(time.time())
        cid = f"chatcmpl-{uuid.uuid4().hex}" if chat else f"cmpl-{uuid.uuid4().hex}"
        obj_type = "chat.completion" if chat else "text_completion"

        # llama.cpp is single-context: serialize requests.
        with self.server.lock:
            try:
                if stream:
                    self._begin_stream()
                    for piece in engine.stream(payload, chat=chat):
                        delta = (
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": piece},
                                        "finish_reason": None,
                                    }
                                ]
                            }
                            if chat
                            else {"choices": [{"index": 0, "text": piece, "finish_reason": None}]}
                        )
                        delta.update(
                            {
                                "id": cid,
                                "object": obj_type + ".chunk",
                                "created": created,
                                "model": self.server.model_id,
                            }
                        )
                        self._sse(delta)
                    self._sse_done()
                else:
                    text = engine.generate(payload, chat=chat)
                    self._send_json(
                        _completion_response(text, chat, cid, created, self.server.model_id)
                    )
            except BrokenPipeError:  # pragma: no cover - client disconnected
                logger.info("client disconnected mid-stream")
            except Exception as exc:  # pragma: no cover - surfaced to client
                logger.exception("generation failed")
                if not stream:
                    self._send_error(f"generation error: {exc}", 500)


def _completion_response(text, chat, cid, created, model_id):
    choice = (
        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        if chat
        else {"index": 0, "text": text, "finish_reason": "stop"}
    )
    return {
        "id": cid,
        "object": "chat.completion" if chat else "text_completion",
        "created": created,
        "model": model_id,
        "choices": [choice],
    }


# Architecture -> llama-cpp-python built-in chat format, used only when the GGUF
# carries no embedded chat template (so /v1/chat/completions still works).
_ARCH_CHAT_FORMAT = {
    "gemma": "gemma",
    "gemma2": "gemma",
    "gemma3": "gemma",
    "gemma3n": "gemma",
    "gemma4": "gemma",
    "llama": "llama-3",
    "llama4": "llama-3",
    "qwen2": "qwen",
    "qwen3": "chatml",
    "phi3": "chatml",
    "mistral": "mistral-instruct",
}


def _infer_chat_format(architecture: str) -> Optional[str]:
    arch = (architecture or "").lower()
    if arch in _ARCH_CHAT_FORMAT:
        return _ARCH_CHAT_FORMAT[arch]
    for key, fmt in _ARCH_CHAT_FORMAT.items():
        if arch.startswith(key):
            return fmt
    return None


def _ensure_chat_format(model, override: Optional[str]) -> Optional[str]:
    """Pick a chat format for ``/v1/chat/completions`` when none is embedded.

    Returns the effective chat format (or None if the GGUF's own template is
    used). Mutating ``_llm.chat_format`` is honoured by ``create_chat_completion``.
    """
    llm = model._llm
    if override:
        llm.chat_format = override
        return override
    metadata = getattr(llm, "metadata", None) or {}
    if "tokenizer.chat_template" in metadata:
        return None  # embedded template wins
    fmt = _infer_chat_format(metadata.get("general.architecture", ""))
    if fmt:
        llm.chat_format = fmt
        logger.info("No embedded chat template; using chat_format=%r", fmt)
    return fmt


# Special end-of-turn markers some chat templates emit as literal text when the
# tokenizer renders the special token; stripped from chat output for clean text.
_TERMINATORS = ("<end_of_turn>", "<eos>", "<|eot_id|>", "<|im_end|>", "</s>", "<|end|>")


def _strip_terminators(text: str) -> str:
    out = text
    changed = True
    while changed:
        changed = False
        for term in _TERMINATORS:
            if out.endswith(term):
                out = out[: -len(term)]
                changed = True
    return out.rstrip()


class _Engine:
    """Adapts an :class:`onecomp.cpu.inference.LlamaCppModel` to OpenAI payloads."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _gen_kwargs(payload):
        return {
            "max_tokens": int(payload.get("max_tokens", 256)),
            "temperature": float(payload.get("temperature", 0.0)),
            "top_p": float(payload.get("top_p", 1.0)),
            "stop": payload.get("stop") or [],
        }

    def generate(self, payload, chat: bool) -> str:
        kw = self._gen_kwargs(payload)
        if chat:
            return _strip_terminators(self.model.chat(payload.get("messages", []), **kw))
        return self.model.generate(payload.get("prompt", ""), **kw)

    def stream(self, payload, chat: bool):
        kw = self._gen_kwargs(payload)
        if not chat:
            yield from self.model.stream(payload.get("prompt", ""), **kw)
            return

        out = self.model._llm.create_chat_completion(
            messages=payload.get("messages", []), stream=True, **kw
        )

        def pieces():
            for chunk in out:
                piece = chunk["choices"][0].get("delta", {}).get("content")
                if piece:
                    yield piece

        # Terminator-aware streaming: hold back a small tail (a terminator could
        # be split across chunks) and stop cleanly at the first end-of-turn marker.
        max_tail = max(len(t) for t in _TERMINATORS) - 1
        buf = ""
        emitted = 0
        for piece in pieces():
            buf += piece
            cut = min((buf.find(t) for t in _TERMINATORS if t in buf), default=-1)
            if cut != -1:
                if cut > emitted:
                    yield buf[emitted:cut]
                return
            safe_upto = max(len(buf) - max_tail, emitted)
            if safe_upto > emitted:
                yield buf[emitted:safe_upto]
                emitted = safe_upto
        rest = _strip_terminators(buf[emitted:])
        if rest:
            yield rest


def serve(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    n_ctx: int = 4096,
    n_threads: Optional[int] = None,
    mode: str = "auto",
    original_model: Optional[str] = None,
    chat_format: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Resolve ``model_path`` to a GGUF and serve it over an OpenAI-compatible API.

    Blocks until interrupted. Packed OneComp checkpoints are auto-exported to a
    cached GGUF on first use.
    """
    from onecomp.cpu.inference import LlamaCppModel

    gguf_path = resolve_to_gguf(model_path, mode=mode, original_model=original_model)
    logger.info("Loading GGUF %s", gguf_path)
    model = LlamaCppModel(gguf_path, n_ctx=n_ctx, n_threads=n_threads, verbose=verbose)
    effective_fmt = _ensure_chat_format(model, chat_format)
    if effective_fmt:
        logger.info("chat completions use chat_format=%r", effective_fmt)

    httpd = ThreadingHTTPServer((host, port), _OpenAIHandler)
    httpd.engine = _Engine(model)
    httpd.model_id = os.path.basename(os.path.normpath(model_path))
    httpd.lock = threading.Lock()

    # Determine whether the model supports chat completions
    metadata = getattr(model._llm, "metadata", None) or {}
    chat_supported = "tokenizer.chat_template" in metadata or effective_fmt is not None
    httpd.chat_supported = chat_supported

    logger.info("OneComp CPU server ready at http://%s:%d (model=%s)", host, port, httpd.model_id)
    print(f"OneComp CPU server: http://{host}:{port}/v1  (model={httpd.model_id})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("shutting down")
    finally:
        httpd.server_close()
