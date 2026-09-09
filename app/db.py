"""
Database layer. SQLite locally, Postgres in production.

Set DATABASE_URL in the environment to switch:
    sqlite:///./local.db                      (default, local dev)
    postgresql+psycopg://user:pass@host/db    (production, e.g. Neon)
"""
import os
import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, String, Integer, Float, DateTime, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./local.db")

# check_same_thread is a SQLite-only quirk: background tasks run on a
# different thread than the request that created them.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Run(Base):
    """One end-to-end pipeline execution.

    status: queued -> running -> succeeded | failed
    This table is the whole reason long-running jobs work over HTTP.
    """
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(32), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100

    # Request parameters, stored as JSON text so you can replay a run later.
    params_json: Mapped[str] = mapped_column(Text, default="{}")

    # Results, stored as JSON text. Swap to JSONB when you move to Postgres.
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class StageTrace(Base):
    """One row per pipeline stage. This is your instrumentation.

    Filling this in from day one is what lets you say
    'forecasting takes 4s, the solver takes 90s' in an interview.
    """
    __tablename__ = "stage_traces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(String(32), index=True)
    step_index: Mapped[int] = mapped_column(Integer)
    stage_name: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


def init_db() -> None:
    """Create tables if they do not exist. Fine at this scale; use Alembic later."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
