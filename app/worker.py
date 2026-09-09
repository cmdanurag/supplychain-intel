"""
The worker. Stages are replaced with real implementations one at a time:

    Stage 3A -> DONE: fake_forecast() replaced by LightGBM
    Stage 3B -> replace fake_optimise() with OR-Tools CP-SAT
    Stage 3B -> replace fake_simulate() with SimPy
    Stage 3C -> replace fake_explain() with an LLM tool-calling agent

Nothing else in the codebase changed when the first one landed, which was the
whole point of building the plumbing first.
"""
import json
import random
import time

from forecasting import predict

from .db import SessionLocal, Run, StageTrace, new_id, utcnow


def real_forecast(params: dict) -> dict:
    """Stage 3A: LightGBM point forecast plus an 80% interval from two quantile
    models. Artifacts are built offline by `python -m forecasting.fit`.
    """
    return predict.forecast(
        store_ids=params["store_ids"],
        item_ids=params["item_ids"],
        horizon_days=params["horizon_days"],
    )


# --------------------------------------------------------------------------
# FAKE STAGES - replace these one at a time
# --------------------------------------------------------------------------


def fake_optimise(params: dict, forecast: dict) -> dict:
    """Replace in Stage 3B with an OR-Tools CP-SAT model."""
    time.sleep(5)
    return {
        "solver_status": "PLACEHOLDER",
        "objective_value": round(random.uniform(8000, 12000), 2),
        "solve_time_ms": 5000,
        "orders": [
            {"node": s["store_id"], "item": s["item_id"],
             "reorder_point": random.randint(20, 80),
             "order_qty": random.randint(50, 200)}
            for s in forecast["series"]
        ],
    }


def fake_simulate(params: dict, solution: dict) -> dict:
    """Replace in Stage 3B with a SimPy simulation + benchmark policies."""
    time.sleep(3)
    optimised = solution["objective_value"]
    return {
        "policies": [
            {"name": "optimised (ours)", "total_cost": optimised,
             "service_level": params["service_level"]},
            {"name": "fixed reorder point (EOQ)",
             "total_cost": round(optimised * 1.16, 2), "service_level": 0.94},
            {"name": "run to failure",
             "total_cost": round(optimised * 1.48, 2), "service_level": 0.81},
        ],
        "improvement_vs_baseline_pct": 16.0,
    }


def fake_explain(params: dict, sim: dict) -> dict:
    """Replace in Stage 3C with an LLM tool-calling agent."""
    time.sleep(2)
    return {
        "summary": "PLACEHOLDER explanation. Stage 3C replaces this with a real agent.",
        "tokens": 0,
        "cost_usd": 0.0,
    }


# --------------------------------------------------------------------------
# PIPELINE RUNNER - this part does NOT change as you build
# --------------------------------------------------------------------------

STAGES = [
    ("forecast", real_forecast),
    ("optimise", fake_optimise),
    ("simulate", fake_simulate),
    ("explain", fake_explain),
]


def _trace(session, run_id: str, idx: int, name: str, ms: int, detail: str = ""):
    session.add(StageTrace(id=new_id(), run_id=run_id, step_index=idx,
                           stage_name=name, latency_ms=ms, detail=detail))
    session.commit()


def run_pipeline(run_id: str) -> None:
    """Entrypoint handed to BackgroundTasks (later: an RQ job)."""
    session = SessionLocal()
    t_start = time.monotonic()
    try:
        run = session.get(Run, run_id)
        if run is None:
            return

        params = json.loads(run.params_json)
        run.status = "running"
        session.commit()

        results: dict = {}
        payload = None

        for idx, (name, fn) in enumerate(STAGES):
            run.stage = name
            run.progress = int(idx / len(STAGES) * 100)
            run.updated_at = utcnow()
            session.commit()

            t0 = time.monotonic()
            if idx == 0:
                payload = fn(params)
            else:
                payload = fn(params, payload)
            ms = int((time.monotonic() - t0) * 1000)

            results[name] = payload
            _trace(session, run_id, idx, name, ms)

            # Save partial results so the UI can render progressively.
            run.result_json = json.dumps(results)
            session.commit()

        run.status = "succeeded"
        run.stage = "done"
        run.progress = 100
        run.duration_ms = int((time.monotonic() - t_start) * 1000)
        run.cost_usd = float(results.get("explain", {}).get("cost_usd", 0.0))
        session.commit()

    except Exception as exc:  # noqa: BLE001 - top-level job guard
        session.rollback()
        run = session.get(Run, run_id)
        if run is not None:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"[:2000]
            run.duration_ms = int((time.monotonic() - t_start) * 1000)
            session.commit()
    finally:
        session.close()
