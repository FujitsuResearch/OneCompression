import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.constants import InferenceStatus, JobStatus


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    status: Mapped[str] = mapped_column(
        String(20), default=JobStatus.PENDING, index=True
    )
    progress: Mapped[int] = mapped_column(Integer, default=0)

    model_name: Mapped[str] = mapped_column(String(256))
    quant_method: Mapped[str] = mapped_column(String(50))
    quant_params: Mapped[dict] = mapped_column(JSON, default=dict)

    result_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    inference_status: Mapped[str] = mapped_column(String(20), default=InferenceStatus.NONE)
    inference_port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inference_pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    inference_url: Mapped[str | None] = mapped_column(String(512), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
