"""Copyright 2025-2026 Fujitsu Ltd."""

from app.constants import ChatTaskStatus, InferenceStatus, JobStatus
from app.core.config import settings
from app.core.database import get_db
from app.models.job import Job
from app.schemas.job import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatTaskResult,
    EstimateWbitsRequest,
    EstimateWbitsResponse,
    JobCreate,
    JobListResponse,
    JobResponse,
)
from app.services.huggingface import check_model_exists
from app.services.inference import chat_vllm, stop_inference
from app.worker.celery_app import celery_app
from app.worker.tasks import chat_with_model, deploy_model, run_quantization
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_to_response(job: Job) -> JobResponse:
    return JobResponse(
        **{c.name: getattr(job, c.name) for c in job.__table__.columns},
    )


@router.post("", status_code=201)
def create_job(body: JobCreate, db: Session = Depends(get_db)) -> JobResponse:
    # ① Reject unknown / unreachable HF model ids before queuing the job
    try:
        check_model_exists(body.model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ④ Fractional wbits is allowed only for autobit / auto_run
    bits = body.quant_params.bits
    if body.quant_method not in ("autobit", "auto_run") and bits != int(bits):
        raise HTTPException(
            status_code=400,
            detail=(
                f"quant_method={body.quant_method!r} requires an integer 'bits' value, "
                f"got {bits}. Fractional bits are only supported by autobit / auto_run."
            ),
        )

    # auto_bits is only meaningful for auto_run
    if body.quant_params.auto_bits and body.quant_method != "auto_run":
        raise HTTPException(
            status_code=400,
            detail="auto_bits=True requires quant_method='auto_run'.",
        )

    if body.quant_method == "jointq" and body.quant_params.use_qep:
        raise HTTPException(
            status_code=400,
            detail=(
                "JointQ does not support QEP. Set use_qep=false or choose "
                "a different quant_method."
            ),
        )

    job = Job(
        model_name=body.model_name,
        quant_method=body.quant_method,
        quant_params=body.quant_params.model_dump(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    run_quantization.delay(job.id)

    return _job_to_response(job)


@router.post("/estimate-wbits")
def estimate_wbits(body: EstimateWbitsRequest) -> EstimateWbitsResponse:
    """③ Estimate the effective bit width that fits the given VRAM budget."""
    try:
        check_model_exists(body.model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        from onecomp.utils import estimate_wbits_from_vram
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"OneComp library is not available: {exc}",
        )

    try:
        result = estimate_wbits_from_vram(
            body.model_name,
            vram_ratio=body.vram_ratio,
            total_vram_gb=body.total_vram_gb,
            group_size=body.group_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"VRAM estimation failed: {exc}")

    return EstimateWbitsResponse(
        target_bitwidth=round(result.target_bitwidth, 4),
        total_vram_gb=result.total_vram_gb,
        budget_gb=result.budget_gb,
        non_quant_weight_gb=result.non_quant_weight_gb,
        available_for_quant_gb=result.available_for_quant_gb,
        total_params=result.total_params,
        quantizable_params=result.quantizable_params,
        meta_bits_per_param=result.meta_bits_per_param,
    )


@router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_response(job)


@router.post("/{job_id}/deploy")
def deploy(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    if job.inference_status in (InferenceStatus.DEPLOYING, InferenceStatus.READY):
        raise HTTPException(status_code=400, detail=f"Already {job.inference_status}")

    deploy_model.delay(job.id)

    job.inference_status = InferenceStatus.DEPLOYING
    db.commit()
    db.refresh(job)
    return _job_to_response(job)


@router.post("/{job_id}/stop")
def stop(job_id: str, db: Session = Depends(get_db)) -> JobResponse:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    stop_inference(job_id)

    db.refresh(job)
    return _job_to_response(job)


@router.post("/{job_id}/chat")
def chat(job_id: str, body: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job is not completed yet")
    if job.inference_status != InferenceStatus.READY:
        raise HTTPException(status_code=400, detail="Model is not deployed. Deploy first.")

    if settings.device == "cuda" and job.inference_url:
        msg = chat_vllm(
            messages=body.messages,
            inference_url=job.inference_url,
            model_name=job.model_name,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )
        return ChatResponse(message=msg)

    task = chat_with_model.delay(
        job.id,
        [m.model_dump() for m in body.messages],
        body.max_tokens,
        body.temperature,
    )
    return ChatResponse(task_id=task.id)


@router.get("/chat-result/{task_id}")
def get_chat_result(task_id: str) -> ChatTaskResult:
    result = celery_app.AsyncResult(task_id)
    if result.ready():
        if result.successful():
            return ChatTaskResult(
                status=ChatTaskStatus.COMPLETED,
                message=ChatMessage(**result.result),
            )
        return ChatTaskResult(status=ChatTaskStatus.FAILED, error=str(result.result))
    return ChatTaskResult(status=ChatTaskStatus.PENDING)


@router.get("")
def list_jobs(limit: int = 20, offset: int = 0, db: Session = Depends(get_db)) -> JobListResponse:
    total = db.query(Job).count()
    jobs = db.query(Job).order_by(Job.created_at.desc()).offset(offset).limit(limit).all()
    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=total,
    )
