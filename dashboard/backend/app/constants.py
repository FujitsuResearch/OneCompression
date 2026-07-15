"""Copyright 2025-2026 Fujitsu Ltd."""

from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InferenceStatus(StrEnum):
    NONE = "none"
    DEPLOYING = "deploying"
    READY = "ready"
    FAILED = "failed"


class ChatTaskStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
