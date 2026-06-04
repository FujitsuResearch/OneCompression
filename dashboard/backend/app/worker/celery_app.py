from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "onecomp_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_concurrency=1,
    worker_prefetch_multiplier=1,
)
