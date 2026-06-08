"""Copyright 2025-2026 Fujitsu Ltd."""

import gc
import json
import logging
import math
import os
import threading
import time

from app.constants import InferenceStatus, JobStatus
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.job import Job
from app.services.job_store import update_job
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


class _QuantProgressMonitor:
    """Background thread that maps onecomp quantizer state to job.progress.

    Quantization progress is derived from the ratio
    ``len(quantizer.results) / len(quantizer.module_to_name)``. The mapping
    keeps the existing milestone semantics intact:

    - ``[0, START_PCT)``  : pre-quantization setup (model load, calibration)
    - ``[START_PCT, END_PCT)`` : per-layer quantization progress (live)
    - ``[END_PCT, 100]``  : post-processing (saving, GemLite repack, etc.)
    """

    START_PCT = 20
    END_PCT = 95

    def __init__(self, job_id: str, quantizer, poll_interval: float = 1.5):
        self._job_id = job_id
        self._quantizer = quantizer
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._last_pct: int | None = None

    def start(self) -> "_QuantProgressMonitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def __enter__(self) -> "_QuantProgressMonitor":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                total = len(getattr(self._quantizer, "module_to_name", {}) or {})
                done = len(getattr(self._quantizer, "results", {}) or {})
                if total > 0:
                    span = self.END_PCT - self.START_PCT
                    pct = self.START_PCT + int(span * min(done / total, 1.0))
                    if self._last_pct is None or pct > self._last_pct:
                        update_job(self._job_id, progress=pct)
                        self._last_pct = pct
            except Exception:  # noqa: BLE001
                logger.debug("progress monitor tick failed", exc_info=True)
            self._stop.wait(self._poll_interval)


def _run_mock_quantization(job_id: str, model_name: str, quant_method: str, params: dict):
    """Simulate quantization with sleep for development without GPU."""
    steps = 10
    for i in range(steps):
        time.sleep(3)
        update_job(job_id, progress=int((i + 1) / steps * 100))

    save_dir = os.path.join(settings.quantized_dir, job_id)
    os.makedirs(save_dir, exist_ok=True)
    dummy = json.dumps({"mock": True, "model": model_name, "method": quant_method}).encode()
    with open(os.path.join(save_dir, "model_quantized.safetensors"), "wb") as f:
        f.write(dummy)
    return save_dir


# Maps the user-facing quant_method to (onecomp quantizer name, force_qep).
# QEP is normally controlled by the `use_qep` checkbox; `force_qep` is only
# set for methods whose semantics require QEP regardless of the user toggle
# (currently auto_run, to stay faithful to onecomp.Runner.auto_run).
QUANTIZER_MAP = {
    "gptq": ("GPTQ", False),
    "autobit": ("AutoBit", False),
    "jointq": ("JointQ", False),
    "auto_run": ("AutoBit", True),
}


def _build_quantizer(quantizer_name: str, wbits: float, groupsize: int):
    """Construct the OneComp quantizer for a non-auto_run method."""
    from onecomp import GPTQ

    if quantizer_name == "GPTQ":
        # GPTQ requires integer wbits
        return GPTQ(wbits=int(wbits), groupsize=groupsize)
    if quantizer_name == "AutoBit":
        # ④ AutoBit interprets `wbits` as the target average bit budget
        # (can be fractional, e.g. 3.5). Candidate bit-widths are fixed to
        # the set vLLM supports for fused GPTQ groups.
        from onecomp import AutoBitQuantizer

        return AutoBitQuantizer(
            assignment_strategy="activation_aware",
            target_bit=float(wbits),
            quantizers=[GPTQ(wbits=b, groupsize=groupsize) for b in (2, 3, 4, 8)],
            enable_fused_groups=True,
        )
    if quantizer_name == "JointQ":
        from onecomp import JointQ

        return JointQ(bits=int(wbits), group_size=groupsize)
    raise ValueError(
        f"Unknown quantizer name: {quantizer_name!r}. "
        f"Expected one of: 'GPTQ', 'AutoBit', 'JointQ'."
    )


def _run_real_quantization(job_id: str, model_name: str, quant_method: str, params: dict):
    """Run actual quantization using OneComp."""
    from onecomp import CalibrationConfig, ModelConfig, Runner, setup_logger

    setup_logger()

    update_job(job_id, progress=5)

    quant_device = "cuda" if settings.device == "cuda" else "cpu"
    model_config = ModelConfig(model_id=model_name, device=quant_device)

    try:
        quantizer_name, force_qep = QUANTIZER_MAP[quant_method]
    except KeyError:
        raise ValueError(
            f"Unsupported quant_method: {quant_method!r}. "
            f"Expected one of: {sorted(QUANTIZER_MAP)}."
        ) from None
    use_qep = force_qep or params.get("use_qep", True)

    if use_qep and quant_device != "cuda":
        logger.warning(
            "QEP requires CUDA — disabling QEP and falling back to standard quantization (device=%s)",
            quant_device,
        )
        use_qep = False

    wbits = params.get("bits", 4)
    groupsize = params.get("group_size", 128)

    # auto_run: always estimate wbits from VRAM (Runner.auto_run default) and
    # persist the resolved params so the UI can show the actual bit width used.
    if quant_method == "auto_run":
        from onecomp.utils import estimate_wbits_from_vram

        groupsize = 128
        est = estimate_wbits_from_vram(
            model_name,
            total_vram_gb=params.get("total_vram_gb"),
            group_size=groupsize,
            logger=logger,
        )
        # Truncate (floor) to 2 decimals so the realized bpw never exceeds the
        # VRAM budget. Matches onecomp.Runner.auto_run.
        wbits = math.floor(est.target_bitwidth * 100) / 100
        if wbits <= 0:
            raise RuntimeError(f"VRAM-based wbits estimation returned non-positive value: {wbits}")
        resolved_params = {
            **params,
            "bits": wbits,
            "group_size": groupsize,
            "auto_bits": True,
        }
        update_job(job_id, quant_params=resolved_params)
        params = resolved_params
        logger.info(
            "auto_run: estimated target wbits=%.2f from VRAM=%.2f GB",
            wbits,
            est.total_vram_gb,
        )
        quantizer = _build_quantizer("AutoBit", float(wbits), groupsize)
    else:
        quantizer = _build_quantizer(quantizer_name, wbits, groupsize)

    update_job(job_id, progress=10)

    calibration_config = CalibrationConfig(
        calibration_dataset=params.get("dataset", "wikitext2"),
        max_length=512,
        num_calibration_samples=params.get("num_samples", 128),
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        qep=use_qep,
        calibration_config=calibration_config,
    )

    update_job(job_id, progress=_QuantProgressMonitor.START_PCT)
    with _QuantProgressMonitor(job_id, quantizer):
        runner.run()
    update_job(job_id, progress=_QuantProgressMonitor.END_PCT)

    save_dir = os.path.join(settings.quantized_dir, job_id)
    os.makedirs(save_dir, exist_ok=True)
    runner.save_quantized_model(save_dir)
    update_job(job_id, progress=100)
    return save_dir


@celery_app.task(bind=True, name="run_quantization")
def run_quantization(self, job_id: str):
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            return
        model_name = job.model_name
        quant_method = job.quant_method
        params = job.quant_params or {}
    finally:
        db.close()

    update_job(job_id, status=JobStatus.RUNNING, progress=0)

    try:
        if settings.mock_quantization:
            result_path = _run_mock_quantization(job_id, model_name, quant_method, params)
        else:
            result_path = _run_real_quantization(job_id, model_name, quant_method, params)

        update_job(job_id, status=JobStatus.COMPLETED, progress=100, result_path=result_path)
    except Exception as exc:
        update_job(job_id, status=JobStatus.FAILED, error_message=str(exc))
        raise self.retry(exc=exc, max_retries=0)
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Released CUDA cache after quantization")
        except Exception:
            pass


@celery_app.task(bind=True, name="deploy_model")
def deploy_model(self, job_id: str):
    from app.services.inference import deploy_mock, deploy_onecomp, deploy_vllm

    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            return
        model_name = job.model_name
        result_path = job.result_path
    finally:
        db.close()

    try:
        if settings.mock_quantization:
            deploy_mock(job_id)
        elif settings.device == "cuda":
            deploy_vllm(job_id, model_name, result_path)
        else:
            deploy_onecomp(job_id, model_name, result_path)
    except Exception as exc:
        update_job(job_id, inference_status=InferenceStatus.FAILED, error_message=str(exc))
        raise self.retry(exc=exc, max_retries=0)


@celery_app.task(name="chat_with_model")
def chat_with_model(
    job_id: str, messages_data: list[dict], max_tokens: int = 256, temperature: float = 0.7
) -> dict:
    from app.schemas.job import ChatMessage
    from app.services.inference import chat_onecomp, chat_vllm

    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            raise RuntimeError(f"Job {job_id} not found")
        model_name = job.model_name
        inference_url = job.inference_url
    finally:
        db.close()

    messages = [ChatMessage(**m) for m in messages_data]

    if settings.device == "cuda" and inference_url:
        result = chat_vllm(
            messages=messages,
            inference_url=inference_url,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        result = chat_onecomp(
            messages=messages, job_id=job_id, max_tokens=max_tokens, temperature=temperature
        )

    return {"role": result.role, "content": result.content}
