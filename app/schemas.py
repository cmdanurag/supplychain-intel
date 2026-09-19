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

    # --- Optimiser knobs. Live from Stage 3B.
    #
    # The two Phase 0 placeholders were flat per-unit dollar figures
    # (`holding_cost_per_unit_day`, `stockout_penalty_per_unit`). They are gone,
    # because the real cost model does not work that way: holding and stockout
    # cost are derived per item from its actual M5 sell price, so a single
    # dollar figure across a $1.68 food item and a $22.98 hobby item would have
    # to be either ignored or applied wrongly. What the caller can still set is
    # the *rates* applied to those measured prices, which is exactly what the
    # model takes. See optimisation/network.py.
    service_level: float = Field(default=0.95, ge=0.5, le=0.999)
    annual_holding_rate: float = Field(default=0.25, ge=0.0, le=2.0)
    stockout_multiplier: float = Field(default=2.0, ge=0.0, le=20.0)


class ScenarioRequest(BaseModel):
    """A what-if question, as parameters rather than prose.

    Stage 3C's first half. Bounds are duplicated from optimisation/scenario.py
    on purpose: this layer rejects a bad request before a job is ever created,
    and the domain layer re-checks because it is also reachable from the CLI.
    A field left unset means "unchanged from the base network".
    """
    item_ids: list[str] = Field(default_factory=list)
    horizon_days: int = Field(default=28, ge=7, le=90)

    # Either name a preset, or set fields directly. A preset plus overrides is
    # allowed, and the overrides win.
    preset: str | None = None

    supplier_lead_days: int | None = Field(default=None, ge=1, le=28)
    store_lead_days: int | None = Field(default=None, ge=0, le=14)
    demand_pct: float | None = Field(default=None, ge=0.5, le=2.0)
    dc_capacity_pct: float | None = Field(default=None, ge=0.3, le=3.0)
    truck_capacity_pct: float | None = Field(default=None, ge=0.3, le=3.0)
    shelf_capacity_pct: float | None = Field(default=None, ge=0.3, le=3.0)
    order_cost_pct: float | None = Field(default=None, ge=0.0, le=5.0)
    delivery_cost_pct: float | None = Field(default=None, ge=0.0, le=5.0)
    stockout_multiplier: float | None = Field(default=None, ge=0.5, le=20.0)
    annual_holding_rate: float | None = Field(default=None, ge=0.0, le=2.0)
    service_level: float | None = Field(default=None, ge=0.5, le=0.999)


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
