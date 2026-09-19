"""
The worker. Stages are replaced with real implementations one at a time:

    Stage 3A -> DONE: fake_forecast() replaced by LightGBM
    Stage 3B -> DONE: fake_optimise() replaced by OR-Tools CP-SAT
    Stage 3B -> DONE: fake_simulate() replaced by the policy benchmark
    Stage 3C -> replace fake_explain() with an LLM tool-calling agent

Nothing else in the codebase changed when any of them landed, which was the
whole point of building the plumbing first. Note what the two Stage 3B stages
each report on: `optimise` plans the window the deployed forecaster actually
serves, which has no ground truth because it is M5's held-back future, while
`simulate` measures policy cost on a held-out window that does. See
optimisation/service.py for why that split is deliberate.

Stage 3B is also the point where the async architecture stops being a
precaution: a run now spends 1-3 minutes inside CP-SAT, well past the 30-60s
request timeouts on typical hosting.
"""
import json
import time

from forecasting import predict
from optimisation import service as optimiser

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


def real_optimise(params: dict, forecast: dict) -> dict:
    """Stage 3B: a CP-SAT multi-echelon replenishment plan for the forecast.

    Costs here are what the plan projects under its own forecast, not measured
    outcomes - that is what the simulate stage is for.
    """
    return optimiser.plan_for_forecast(forecast, params)


def real_simulate(params: dict, solution: dict) -> dict:
    """Stage 3B: the measured policy comparison, on held-out actual demand.

    Every policy - ours and three baselines - is re-run here rather than having
    its number read from a file, so the table in the UI describes the items this
    run actually asked about.
    """
    return optimiser.benchmark_for_items(params, solution)


# --------------------------------------------------------------------------
# FAKE STAGES - replace these one at a time
# --------------------------------------------------------------------------


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
    ("optimise", real_optimise),
    ("simulate", real_simulate),
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


def run_scenario(run_id: str) -> None:
    """Stage 3C job: solve the base network and a changed one, then diff them.

    Deliberately a second entry point rather than a fifth pipeline stage. A
    scenario answers a different question from a run - "what changes if the
    world does" rather than "what should we do now" - and it reuses the same
    row, the same status vocabulary and the same polling endpoint, so the client
    needs no new machinery to wait for it.
    """
    session = SessionLocal()
    t_start = time.monotonic()
    marks: dict[str, float] = {}
    try:
        run = session.get(Run, run_id)
        if run is None:
            return

        params = json.loads(run.params_json)
        run.status = "running"
        session.commit()

        def on_stage(name: str, index: int) -> None:
            """Called as each leg starts, so a poll can say which one is running."""
            marks[name] = time.monotonic()
            run.stage = f"solving {name}"
            run.progress = 10 + index * 45
            run.updated_at = utcnow()
            session.commit()

        t0 = time.monotonic()
        report = optimiser.scenario_for_request(params, on_stage=on_stage)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # One trace per leg, split by the solve time each leg reported, so the
        # timings panel says where the wait went rather than showing one blob.
        base_ms = int(report["base"]["solve_seconds"] * 1000)
        scen_ms = int(report["scenario_result"]["solve_seconds"] * 1000)
        _trace(session, run_id, 0, "scenario:base", base_ms,
               report["scenario"]["label"])
        _trace(session, run_id, 1, "scenario:changed", scen_ms,
               json.dumps(report["scenario"]["changes"]))
        _trace(session, run_id, 2, "scenario:diff",
               max(0, elapsed_ms - base_ms - scen_ms), report["headline"])

        run.result_json = json.dumps({"scenario": report})
        run.status = "succeeded"
        run.stage = "done"
        run.progress = 100
        run.duration_ms = int((time.monotonic() - t_start) * 1000)
        session.commit()

    except Exception as exc:  # noqa: BLE001 - top-level job guard, as run_pipeline
        session.rollback()
        run = session.get(Run, run_id)
        if run is not None:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"[:2000]
            run.duration_ms = int((time.monotonic() - t_start) * 1000)
            session.commit()
    finally:
        session.close()
