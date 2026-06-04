from app.core.database import SessionLocal
from app.models.job import Job


def update_job(job_id: str, **kwargs) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job:
            for key, value in kwargs.items():
                setattr(job, key, value)
            db.commit()
    finally:
        db.close()
