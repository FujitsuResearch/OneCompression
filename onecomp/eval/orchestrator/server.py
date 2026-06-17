"""vLLM OpenAI-compatible HTTP server lifecycle manager.

Drop-in context manager that:

- Auto-detects quantization / dtype from the model directory
  (via onecomp.eval.utils.model_utils).
- Allocates a free port if port == 0.
- Spawns python -m vllm.entrypoints.openai.api_server as a subprocess,
  with stdout/stderr forwarded to <log_dir>/vllm_server.log.
- Polls /health until the server is ready.
- Sends SIGTERM (then SIGKILL after a grace period) on __exit__.

Used by both run_pipeline and downstream tests; not invoked
directly by evaluator subprocesses.

Copyright 2025-2026 Fujitsu Ltd.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from logging import getLogger
from pathlib import Path
from types import TracebackType

from ..schema import InferenceConfig
from ..utils.model_utils import (
    detect_model_config,
    print_model_summary,
    resolve_dtype,
    resolve_quantization,
)
from ..utils.ports import find_free_port, wait_for_http

logger = getLogger(__name__)


class VllmServerManager:
    """Context manager that owns a vLLM HTTP server lifecycle."""

    def __init__(
        self,
        cfg: InferenceConfig,
        model_path: str | Path,
        log_dir: str | Path,
    ) -> None:
        self.cfg = cfg
        self.model_path = str(model_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._proc: subprocess.Popen | None = None
        self._log_file = None
        self._host = cfg.host
        self._port = cfg.port or find_free_port(cfg.host)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self._host}:{self._port}/health"

    @property
    def api_key(self) -> str:
        return self.cfg.api_key

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------
    def __enter__(self) -> "VllmServerManager":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Spawn the vLLM HTTP server and wait until /health is 200."""
        if self.is_alive():
            logger.warning("vLLM server already running on port %d", self._port)
            return

        argv = self._build_argv()
        log_path = self.log_dir / "vllm_server.log"
        logger.info("Starting vLLM server: %s", " ".join(argv))
        logger.info("vLLM server log: %s", log_path)

        # Open in line-buffered mode so the log file is useful while the
        # server is still booting.
        self._log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        env = dict(os.environ)
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")

        self._proc = subprocess.Popen(
            argv,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if os.name == "posix" else None,
        )

        try:
            wait_for_http(
                self.health_url,
                timeout_sec=float(self.cfg.startup_timeout_sec),
                interval_sec=2.0,
                is_alive=self.is_alive,
            )
        except (TimeoutError, RuntimeError):
            logger.error("vLLM server failed to start; see log: %s", log_path)
            self.stop()
            raise

        logger.info("vLLM server ready at %s", self.base_url)

    def stop(self) -> None:
        """Shut the server down gracefully, then force-kill if needed."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            logger.info("vLLM server already exited with code %d", self._proc.returncode)
            self._close_log()
            self._proc = None
            return

        pgid = self._safe_pgid()
        logger.info("Stopping vLLM server (pid=%d, pgid=%s)", self._proc.pid, pgid)
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                self._proc.terminate()
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("SIGTERM timed out; sending SIGKILL")
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    self._proc.kill()
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.error("vLLM server failed to exit after SIGKILL")
        except ProcessLookupError:
            pass

        logger.info("vLLM server stopped (exit=%s)", self._proc.returncode)
        self._close_log()
        self._proc = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _safe_pgid(self) -> int | None:
        if os.name != "posix" or self._proc is None:
            return None
        try:
            return os.getpgid(self._proc.pid)
        except ProcessLookupError:
            return None

    def _close_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _build_argv(self) -> list[str]:
        """Construct the vLLM api_server command line."""
        info = detect_model_config(Path(self.model_path))
        print_model_summary(info)

        quantization = self.cfg.quantization
        if quantization is None and info.get("quant_method"):
            quantization = resolve_quantization(
                info["quant_method"],
                info.get("bits"),
                desc_act=info.get("desc_act", False),
                sym=info.get("sym", False),
            )
        dtype = resolve_dtype(
            quantization,
            self.cfg.dtype,
            torch_dtype=info.get("torch_dtype") or info.get("dtype"),
        )

        argv: list[str] = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--host",
            self._host,
            "--port",
            str(self._port),
            "--tensor-parallel-size",
            str(self.cfg.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(self.cfg.gpu_memory_utilization),
            "--max-model-len",
            str(self.cfg.max_model_len),
            "--dtype",
            dtype,
        ]
        if self.cfg.trust_remote_code:
            argv.append("--trust-remote-code")
        if self.cfg.enforce_eager:
            argv.append("--enforce-eager")
        if quantization:
            argv.extend(["--quantization", quantization])
        argv.extend(self.cfg.extra_args)
        return argv
