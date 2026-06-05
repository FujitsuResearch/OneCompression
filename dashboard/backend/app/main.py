from contextlib import asynccontextmanager

from app.api.jobs import router as jobs_router
from app.core.database import Base, engine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text


def _sync_schema():
    """Create missing tables and add missing columns to existing tables."""
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    with engine.begin() as conn:
        for table_name, table in Base.metadata.tables.items():
            if not inspector.has_table(table_name):
                continue
            existing = {c["name"] for c in inspector.get_columns(table_name)}
            for col in table.columns:
                if col.name in existing:
                    continue
                col_type = col.type.compile(dialect=engine.dialect)
                nullable = "NULL" if col.nullable else "NOT NULL"
                conn.execute(
                    text(
                        f'ALTER TABLE "{table_name}" ADD COLUMN "{col.name}" {col_type} {nullable}'
                    )
                )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _sync_schema()
    yield


app = FastAPI(
    title="OneComp Quantization Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
