from datetime import datetime

from app.constants import ChatTaskStatus, InferenceStatus, JobStatus
from pydantic import BaseModel, Field


class QuantParams(BaseModel):
    bits: float = Field(
        4,
        ge=2.0,
        le=8.0,
        description=(
            "Quantization bit width. Integer for fixed-bit methods (gptq, jointq); "
            "may be fractional for autobit / auto_run (target average bpw)."
        ),
    )
    group_size: int = Field(128, description="Group size for quantization")
    use_qep: bool = Field(True, description="Enable Quantization Error Propagation")
    dataset: str = Field(
        "wikitext2",
        description="Calibration dataset name (e.g. wikitext2, c4) or local path",
    )
    num_samples: int = Field(
        128,
        ge=1,
        le=1024,
        description="Number of calibration samples (maps to CalibrationConfig.num_calibration_samples)",
    )
    auto_bits: bool = Field(
        False,
        description=(
            "When true (auto_run only), ignore 'bits' and estimate the target "
            "bit width automatically from available VRAM."
        ),
    )
    total_vram_gb: float | None = Field(
        None,
        ge=0.5,
        description=(
            "VRAM budget in GB used by auto_run for automatic wbits estimation. "
            "When omitted, the installed GPU VRAM is detected automatically."
        ),
    )


class JobCreate(BaseModel):
    model_name: str = Field(..., min_length=1, examples=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
    quant_method: str = Field(..., pattern="^(gptq|autobit|jointq|auto_run)$")
    quant_params: QuantParams = Field(default_factory=QuantParams)


class EstimateWbitsRequest(BaseModel):
    model_name: str = Field(..., min_length=1, examples=["TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
    total_vram_gb: float = Field(..., gt=0.0, le=2048.0, description="Available VRAM in GB")
    group_size: int = Field(128, description="Quantization group size")
    vram_ratio: float = Field(0.8, gt=0.0, le=1.0, description="Fraction of VRAM to use")


class EstimateWbitsResponse(BaseModel):
    target_bitwidth: float = Field(..., description="Recommended average bits per parameter")
    total_vram_gb: float
    budget_gb: float
    non_quant_weight_gb: float
    available_for_quant_gb: float
    total_params: int
    quantizable_params: int
    meta_bits_per_param: float


class JobResponse(BaseModel):
    id: str
    status: JobStatus
    progress: int
    model_name: str
    quant_method: str
    quant_params: dict
    result_path: str | None
    error_message: str | None
    inference_status: InferenceStatus
    inference_url: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = Field(64, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    message: ChatMessage | None = None
    task_id: str | None = None


class ChatTaskResult(BaseModel):
    status: ChatTaskStatus
    message: ChatMessage | None = None
    error: str | None = None
