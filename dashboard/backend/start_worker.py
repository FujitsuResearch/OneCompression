#!/usr/bin/env python
"""Celery worker entrypoint – applies cuda patch (no-op when CUDA is available)."""

import cpu_patch  # noqa: F401
from app.worker.celery_app import celery_app

if __name__ == "__main__":
    celery_app.worker_main(
        [
            "worker",
            "--loglevel=info",
            "--pool=solo",
            "--without-heartbeat",
        ]
    )
