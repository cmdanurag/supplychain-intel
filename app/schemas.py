"""
Pydantic schemas. Every boundary in this system is typed.

This matters more than it looks: once you add a real forecaster, a solver,
and an LLM agent, untyped dict handoffs are where the bugs live.
"""
from typing import Literal, Any
from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """What the client sends to start a pipeline run."""
    # Defaults must exist in the deployed model's catalog or a bare POST fails.
    # GET /api/catalog is the source of truth for what is servable.
    store_ids: list[str] = Field(default_factory=lambda: ["CA_1"], min_length=1)
    item_ids: list[str] = Field(default_factory=lambda: ["FOODS_1_011"], min_length=1)
    horizon_days: int = Field(default=28, ge=1, le=90)

    # Optimiser knobs. Ignored in Phase 0, used from Stage 3B onward.
    service_level: float = Field(default=0.95, ge=0.5, le=0.999)
    holding_cost_per_unit_day: float = Field(default=0.02, ge=0)
    stockout_penalty_per_unit: float = Field(default=3.0, ge=0)


class RunAccepted(BaseModel):
    """Returned immediately. The whole point of the async pattern."""
    run_id: str
    status: str
    poll_url: str


class StageTraceOut(BaseModel):
    step_index: int
    stage_name: str
    latency_ms: int
    detail: str | None = None


class ForecastPoint(BaseModel):
    date: str
    point: float
    lower: float
    upper: float


class RunStatus(BaseModel):
    """What polling returns. Includes partial results while still running."""
    run_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str
    progress: int
    duration_ms: int | None = None
    cost_usd: float = 0.0
    error: str | None = None
    traces: list[StageTraceOut] = Field(default_factory=list)
    result: dict[str, Any] | None = None
