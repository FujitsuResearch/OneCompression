"""Copyright 2025-2026 Fujitsu Ltd."""

import gc
import logging
import os
import signal
import subprocess
import time

import httpx
import torch
from app.constants import InferenceStatus
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.job import Job
from app.schemas.job import ChatMessage
from app.services.job_store import update_job

logger = logging.getLogger(__name__)

# HPC clusters often route even localhost through an HTTP proxy (Squid).
for _var in ("no_proxy", "NO_PROXY"):
    _cur = os.environ.get(_var, "")
    if "localhost" not in _cur:
        os.environ[_var] = f"{_cur},localhost,127.0.0.1" if _cur else "localhost,127.0.0.1"

_loaded_models: dict[str, dict] = {}


# ── Mock implementations ──────────────────────────────────────


def deploy_mock(job_id: str) -> None:
    update_job(job_id, inference_status=InferenceStatus.DEPLOYING)
    time.sleep(1)
    update_job(job_id, inference_status=InferenceStatus.READY)


# ── OneComp implementations (CPU / MPS) ──────────────────────


def deploy_onecomp(job_id: str, model_name: str, model_dir: str) -> None:
    """Load quantized model directly from local path."""
    update_job(job_id, inference_status=InferenceStatus.DEPLOYING)

    try:
        from onecomp import load_quantized_model

        model, tokenizer = load_quantized_model(model_dir)

        if settings.device == "mps" and torch.backends.mps.is_available():
            model = model.to("mps")

        _loaded_models[job_id] = {"model": model, "tokenizer": tokenizer}
        update_job(job_id, inference_status=InferenceStatus.READY)
    except Exception as exc:
        update_job(job_id, inference_status=InferenceStatus.FAILED, error_message=str(exc))
        raise


def chat_onecomp(
    messages: list[ChatMessage],
    job_id: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> ChatMessage:
    """Run inference using a model loaded via OneComp."""
    ctx = _loaded_models.get(job_id)
    if not ctx:
        raise RuntimeError("Model not loaded. Deploy first.")

    model = ctx["model"]
    tokenizer = ctx["tokenizer"]
    device = next(model.parameters()).device

    prompt_parts = []
    for m in messages:
        if m.role == "user":
            prompt_parts.append(f"User: {m.content}")
        else:
            prompt_parts.append(f"Assistant: {m.content}")
    prompt_parts.append("Assistant:")
    prompt = "\n".join(prompt_parts)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )

    return ChatMessage(role="assistant", content=generated.strip())


# ── vLLM implementations (CUDA) ──────────────────────────────

_VLLM_HEALTH_TIMEOUT = 300  # seconds to wait for vLLM to become healthy


def _stop_existing_deployment(port: int) -> None:
    """Kill any vLLM left over from a previous deploy."""
    db = SessionLocal()
    try:
        jobs = db.query(Job).filter(Job.inference_pid.isnot(None)).all()
        for job in jobs:
            try:
                os.killpg(os.getpgid(job.inference_pid), signal.SIGKILL)
                logger.info("Killed previous vLLM pid=%d (job %s)", job.inference_pid, job.id)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            job.inference_pid = None
            job.inference_port = None
            job.inference_url = None
            job.inference_status = InferenceStatus.NONE
        db.commit()
    finally:
        db.close()

    subprocess.run(
        ["pkill", "-9", "-f", "vllm.entrypoints.openai.api_server"],
        stderr=subprocess.DEVNULL,
    )

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            httpx.get(f"http://localhost:{port}/health", timeout=1)
            time.sleep(1)
        except (httpx.ConnectError, httpx.ReadError):
            break


def _pick_gpu_memory_utilization() -> float:
    """Choose a safe gpu-memory-utilization based on actual free VRAM."""
    if not torch.cuda.is_available():
        return 0.45
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / (1 << 30), total / (1 << 30)
    util = round(min(max((free_gb * 0.9) / total_gb, 0.1), 0.9), 2)
    logger.info(
        "GPU memory: %.1f/%.1f GiB free → gpu-memory-utilization=%.2f", free_gb, total_gb, util
    )
    return util


def deploy_vllm(job_id: str, model_name: str, model_dir: str) -> None:
    """Start a vLLM OpenAI-compatible server for the quantized model."""
    update_job(job_id, inference_status=InferenceStatus.DEPLOYING)

    port = settings.vllm_port
    host = settings.worker_host

    _stop_existing_deployment(port)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gpu_util = _pick_gpu_memory_utilization()

    try:
        log_path = f"/tmp/vllm-{job_id}.log"
        log_file = open(log_path, "w")

        env = {k: v for k, v in os.environ.items()}
        env.pop("VLLM_API_KEY", None)
        env["VLLM_NO_USAGE_STATS"] = "1"
        env.setdefault("NCCL_SOCKET_IFNAME", "lo")
        env.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")

        cmd = [
            settings.vllm_python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_dir,
            "--served-model-name",
            model_name,
            "--port",
            str(port),
            "--host",
            "0.0.0.0",
            "--trust-remote-code",
            "--gpu-memory-utilization",
            str(gpu_util),
            "--no-enable-log-requests",
            "--enforce-eager",
        ]
        logger.info("Starting vLLM: %s", " ".join(cmd))

        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        health_url = f"http://localhost:{port}/health"
        deadline = time.monotonic() + _VLLM_HEALTH_TIMEOUT

        while time.monotonic() < deadline:
            time.sleep(3)
            if proc.poll() is not None:
                log_file.close()
                with open(log_path) as f:
                    lines = f.readlines()
                error_lines = [
                    l.rstrip()
                    for l in lines
                    if "ERROR" in l or "ValueError" in l or "RuntimeError" in l
                ]
                tail = "".join(lines[-40:])
                raise RuntimeError(
                    f"vLLM exited with code {proc.returncode}:\n"
                    + "\n".join(error_lines[-20:])
                    + "\n---\n"
                    + tail
                )
            try:
                resp = httpx.get(health_url, timeout=5)
                if resp.status_code == 200:
                    break
            except httpx.ConnectError:
                continue

        else:
            proc.terminate()
            raise RuntimeError(f"vLLM failed to become healthy within {_VLLM_HEALTH_TIMEOUT}s")

        inference_url = f"http://{host}:{port}"
        update_job(
            job_id,
            inference_status=InferenceStatus.READY,
            inference_port=port,
            inference_pid=proc.pid,
            inference_url=inference_url,
        )
        logger.info("vLLM ready at %s (pid=%d)", inference_url, proc.pid)

    except Exception as exc:
        update_job(job_id, inference_status=InferenceStatus.FAILED, error_message=str(exc))
        raise


def chat_vllm(
    messages: list[ChatMessage],
    inference_url: str,
    model_name: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> ChatMessage:
    """Proxy chat request to a running vLLM server."""
    payload = {
        "model": model_name,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = httpx.post(
        f"{inference_url}/v1/chat/completions",
        json=payload,
        timeout=settings.chat_timeout,
    )
    if resp.status_code != 200:
        logger.error("vLLM responded %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()

    choice = resp.json()["choices"][0]["message"]
    return ChatMessage(role=choice["role"], content=choice["content"])


# ── Lifecycle ─────────────────────────────────────────────────


def stop_inference(job_id: str) -> None:
    _loaded_models.pop(job_id, None)

    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            return

        if job.inference_pid:
            try:
                os.kill(job.inference_pid, signal.SIGTERM)
                logger.info("Sent SIGTERM to vLLM pid=%d", job.inference_pid)
            except (ProcessLookupError, PermissionError):
                pass

        job.inference_status = InferenceStatus.NONE
        job.inference_port = None
        job.inference_pid = None
        job.inference_url = None
        db.commit()
    finally:
        db.close()
