"""
FastAPI service.

THE CORE IDEA: POST /api/runs does NOT wait for the pipeline. It writes a
row, schedules a background job, and returns a run_id immediately. The client
polls GET /api/runs/{id} until status is succeeded or failed.

This is why the app survives a 3-minute solver run on a host that kills
requests after 60 seconds.
"""
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from forecasting import predict

from .db import init_db, get_session, Run, StageTrace, new_id
from .schemas import RunRequest, RunAccepted, RunStatus, StageTraceOut
from .worker import run_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Supply Chain Intelligence API",
    version="0.3.0",
    description=(
        "Demand forecasting (LightGBM) and multi-echelon inventory optimisation "
        "(OR-Tools CP-SAT), benchmarked on held-out M5 sales. Runs are async: "
        "POST /api/runs returns a run_id, then poll GET /api/runs/{run_id}. "
        "The explain stage (3C) is still a placeholder."
    ),
    lifespan=lifespan,
)

# Frontend and backend will live on different domains once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    """The bare URL is what ends up on a CV or in a message, so send visitors to
    the interactive docs rather than FastAPI's default 404."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Hosts ping this. Keep it dependency-free and fast."""
    return {"status": "ok", "stages": ["forecast", "optimise", "simulate"]}


@app.get("/api/catalog")
def catalog():
    """What the deployed model can actually answer for.

    The frontend builds its pickers from this, so the UI cannot offer a store or
    item the model has never seen - which it did when both were hardcoded.
    """
    try:
        return predict.available()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/runs", response_model=RunAccepted, status_code=202)
def create_run(
    req: RunRequest,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
):
    run = Run(id=new_id(), params_json=req.model_dump_json(), status="queued")
    session.add(run)
    session.commit()

    # Phase 0: FastAPI BackgroundTasks. Swap for RQ + Redis when you need
    # retries, concurrency limits, or to survive a process restart.
    background.add_task(run_pipeline, run.id)

    return RunAccepted(
        run_id=run.id,
        status=run.status,
        poll_url=f"/api/runs/{run.id}",
    )


@app.get("/api/runs/{run_id}", response_model=RunStatus)
def get_run(run_id: str, session: Session = Depends(get_session)):
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    traces = session.scalars(
        select(StageTrace)
        .where(StageTrace.run_id == run_id)
        .order_by(StageTrace.step_index)
    ).all()

    return RunStatus(
        run_id=run.id,
        status=run.status,
        stage=run.stage,
        progress=run.progress,
        duration_ms=run.duration_ms,
        cost_usd=run.cost_usd,
        error=run.error,
        traces=[
            StageTraceOut(
                step_index=t.step_index,
                stage_name=t.stage_name,
                latency_ms=t.latency_ms,
                detail=t.detail,
            )
            for t in traces
        ],
        result=json.loads(run.result_json) if run.result_json else None,
    )


@app.get("/api/runs")
def list_runs(limit: int = 20, session: Session = Depends(get_session)):
    runs = session.scalars(
        select(Run).order_by(Run.created_at.desc()).limit(min(limit, 100))
    ).all()
    return [
        {
            "run_id": r.id,
            "status": r.status,
            "stage": r.stage,
            "created_at": r.created_at.isoformat(),
            "duration_ms": r.duration_ms,
        }
        for r in runs
    ]
