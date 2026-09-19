"""
Serving-time entry points for Stage 3B. What app/worker.py actually calls.

Kept out of worker.py so the pipeline runner stays a pipeline runner - the same
reason forecasting/predict.py exists separately from the forecasting code that
trains the model.

**The two stages answer questions about two different windows, and the responses
say so.** This is not tidy, but the alternative is worse:

  - `optimise` plans the window the deployed model actually forecasts
    (2016-05-23 onward). That window is M5's held-back competition future, so no
    actuals for it exist anywhere in this repo. A real deployment is in exactly
    this position every day - you plan against a forecast and find out later -
    so this stage returns a plan, a solver status and what the plan is projected
    to cost, and makes no claim about realised cost.

  - `simulate` answers "does this policy actually beat the alternatives", which
    requires ground truth. So it back-tests on the most recent window where both
    forecasts and actuals exist (2016-04-25 to 2016-05-22; see
    optimisation/evalset.py). Every number it returns is measured against demand
    the forecaster never saw.

Both payloads carry their own `window` and `evaluation` fields. Reporting a
projected cost as if it were a measured one would be the single most misleading
thing this service could do.
"""
import os

import numpy as np

from . import benchmark, evalset, scenario as scenarios
from .cpsat import State, integerise_demand, solve
from .network import build_network, select_items

# A request for all 401 items would build a solve far larger than anything that
# finishes inside a job a user is waiting on. Cap it, rank by revenue, and say
# in the response that it was capped.
MAX_OPTIMISE_ITEMS = int(os.getenv("MAX_OPTIMISE_ITEMS", "40"))
SOLVER_TIME_LIMIT_S = float(os.getenv("SOLVER_TIME_LIMIT_S", "15"))
PLANNING_HORIZON_DAYS = int(os.getenv("PLANNING_HORIZON_DAYS", "14"))


def _jsonable(value):
    """numpy scalars are not JSON-serialisable and sqlite stores JSON text."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _network_for(item_ids: list[str], params: dict):
    """Build the network for the requested items, capped and revenue-ranked."""
    _, history, meta = evalset.load()
    available = set(history["item_id"])
    wanted = [i for i in item_ids if i in available]
    if not wanted:
        raise ValueError(
            f"None of the requested items can be costed. The cost model covers "
            f"{len(available)} items of store {meta['store']}.")
    chosen = select_items(history[history["item_id"].isin(wanted)],
                          MAX_OPTIMISE_ITEMS)
    network = build_network(
        chosen, meta["store"],
        stockout_multiplier=_stockout_multiplier(params),
        holding_rate=_holding_rate(params),
    )
    dropped = [i for i in wanted if i not in set(network.item_ids)]
    return network, meta, dropped


def _stockout_multiplier(params: dict) -> float:
    from .network import STOCKOUT_MULTIPLIER
    value = params.get("stockout_multiplier")
    return STOCKOUT_MULTIPLIER if value is None else float(value)


def _holding_rate(params: dict) -> float:
    from .network import ANNUAL_HOLDING_RATE
    value = params.get("annual_holding_rate")
    return ANNUAL_HOLDING_RATE if value is None else float(value)


def plan_for_forecast(forecast: dict, params: dict) -> dict:
    """Stage 3B `optimise`: a CP-SAT replenishment plan for the served forecast.

    Returns the plan and what it is *projected* to cost under that forecast. No
    realised-cost claim is made here - see the module docstring.
    """
    series = forecast.get("series") or []
    if not series:
        raise ValueError("no forecast series to optimise")

    item_ids = [s["item_id"] for s in series]
    network, meta, dropped = _network_for(item_ids, params)
    keep = set(network.item_ids)

    by_item = {s["item_id"]: s for s in series if s["item_id"] in keep}
    horizon = min(
        PLANNING_HORIZON_DAYS,
        min(len(s["points"]) for s in by_item.values()),
    )
    demand = {i: integerise_demand(
        np.array([p["point"] for p in by_item[i]["points"]][:horizon]))
        for i in network.item_ids}

    plan = solve(
        network, demand, State.initial(network, horizon),
        service_level=float(params.get("service_level", 0.95)),
        time_limit_s=SOLVER_TIME_LIMIT_S,
    )

    dates = [p["date"] for p in next(iter(by_item.values()))["points"]][:horizon]
    orders = [
        {
            "item_id": i,
            "case_pack": network.by_id(i).case_pack,
            "order_days": [
                {"date": dates[t], "units": int(plan.orders[i][t])}
                for t in range(horizon) if plan.orders[i][t] > 0
            ],
            "shipment_days": [
                {"date": dates[t], "units": int(plan.shipments[i][t])}
                for t in range(horizon) if plan.shipments[i][t] > 0
            ],
        }
        for i in network.item_ids
    ]

    return _jsonable({
        "window": [dates[0], dates[-1]] if dates else None,
        "evaluation": (
            "plan only - this window is M5's held-back future, so no actuals "
            "exist to score it against. Measured policy performance is in the "
            "simulate stage, which back-tests on a window that has ground truth."
        ),
        "solver": "ortools-cpsat",
        "solver_status": plan.status,
        "objective_usd": round(plan.objective, 2),
        "optimality_gap_pct": plan.gap_pct,
        "solve_time_ms": plan.solve_time_ms,
        "planning_horizon_days": horizon,
        "relaxations": plan.relaxations,
        "projected_costs_usd": plan.planned_costs,
        "items_optimised": len(network.item_ids),
        "items_dropped_for_size": dropped,
        "network": network.summary(),
        "orders": orders,
        "total_order_units": int(sum(int(v.sum()) for v in plan.orders.values())),
        "total_shipment_units": int(sum(int(v.sum()) for v in plan.shipments.values())),
        "delivery_days_planned": int(sum(
            1 for t in range(horizon)
            if any(plan.shipments[i][t] > 0 for i in network.item_ids))),
    })


def benchmark_for_items(params: dict, optimise_result: dict | None = None) -> dict:
    """Stage 3B `simulate`: the measured policy comparison, on held-out actuals.

    Runs the same harness as `python -m optimisation.benchmark`, restricted to
    the items this run asked about, so the cost reduction shown in the UI is the
    one this instance actually produced rather than a figure read from a file.
    """
    item_ids = list(params.get("item_ids") or [])
    report = benchmark.run(
        n_items=MAX_OPTIMISE_ITEMS,
        horizon=int(params.get("horizon_days", 28)),
        service_level=float(params.get("service_level", 0.95)),
        planning_horizon=PLANNING_HORIZON_DAYS,
        time_limit_s=SOLVER_TIME_LIMIT_S,
        stockout_multiplier=_stockout_multiplier(params),
        holding_rate=_holding_rate(params),
        item_ids=item_ids or None,
        quiet=True,
    )

    # The frontend's table wants one row per policy, cheapest last, plus the
    # single percentage the whole project is judged on.
    order = ["reorder_point_eoq", "safety_stock_base_stock", "forecast_base_stock",
             "cpsat_rolling_horizon", "perfect_information"]
    labels = {
        "reorder_point_eoq": "fixed reorder point (EOQ)",
        "safety_stock_base_stock": "safety stock (z-sigma)",
        "forecast_base_stock": "forecast base-stock",
        "cpsat_rolling_horizon": "CP-SAT rolling horizon (ours)",
        "perfect_information": "perfect information (reference)",
    }
    policies = [
        {
            "name": labels[name],
            "key": name,
            "total_cost": report["policies"][name]["total_cost"],
            "service_level": report["policies"][name]["fill_rate"],
            "holding": report["policies"][name]["holding"],
            "stockout": report["policies"][name]["stockout"],
            "fixed_costs": round(report["policies"][name]["ordering"]
                                 + report["policies"][name]["delivery"], 2),
            "delivery_days": report["policies"][name]["delivery_days"],
            "po_lines": report["policies"][name]["po_lines"],
        }
        for name in order if name in report["policies"]
    ]

    return _jsonable({
        "window": report["instance"]["window"],
        "evaluation": report["instance"]["evaluation"],
        "instance": report["instance"],
        "protocol": report["protocol"],
        "policies": policies,
        "improvement_vs_baseline_pct": report["headline"][
            "cost_reduction_vs_reorder_point_pct"],
        "headline": report["headline"],
        "ablation": report["ablation"],
        "cost_mechanism": report["cost_mechanism"],
        "solver": report["solver"],
        "feasibility_repairs": report["feasibility_repairs"],
    })


# --------------------------------------------------------------------------
# Stage 3C (first half): what-if scenarios
# --------------------------------------------------------------------------

# A scenario is two full policy runs, so it is roughly twice the work of one
# simulate stage. Cap it below the optimise cap rather than at it.
MAX_SCENARIO_ITEMS = int(os.getenv("MAX_SCENARIO_ITEMS",
                                   str(max(4, MAX_OPTIMISE_ITEMS // 2))))


def scenario_presets() -> dict:
    """What the UI offers as one-click questions."""
    return {
        "presets": [
            {"key": key, "label": label, "changes": changes}
            for key, (label, changes) in scenarios.PRESETS.items()
        ],
        "bounds": {k: list(v) for k, v in scenarios.BOUNDS.items()},
        "max_items": MAX_SCENARIO_ITEMS,
    }


def _scenario_from(params: dict) -> scenarios.Scenario:
    """Request parameters -> a validated Scenario.

    A preset supplies the starting point; explicit fields override it. Unknown
    or out-of-range values raise here rather than being clamped, because a
    silently altered question gets a confidently wrong answer.
    """
    overrides = {
        field: params[field]
        for field in scenarios.BOUNDS
        if params.get(field) is not None
    }
    if params.get("preset"):
        base = scenarios.preset(params["preset"])
        if not overrides:
            return base
        label = base.label + " (adjusted)" if overrides else base.label
        return scenarios.replace(base, label=label, **overrides)
    if not overrides:
        raise ValueError(
            "a scenario needs either a preset or at least one changed field; "
            f"presets are {sorted(scenarios.PRESETS)}")
    return scenarios.Scenario(label="custom scenario", **overrides)


def scenario_for_request(params: dict, on_stage=None) -> dict:
    """Stage 3C `scenario`: re-solve under a changed network and diff.

    Both legs run in this process against the same instance, so the comparison
    is like-for-like. Two separate requests will differ by a few percent even
    with identical inputs, because CP-SAT stops at a wall-clock limit - which is
    also why the base leg is re-run here instead of cached.
    """
    spec = _scenario_from(params)
    item_ids = list(params.get("item_ids") or [])
    report = scenarios.compare(
        spec,
        item_ids=item_ids or None,
        n_items=MAX_SCENARIO_ITEMS,
        horizon=int(params.get("horizon_days", 28)),
        planning_horizon=PLANNING_HORIZON_DAYS,
        time_limit_s=SOLVER_TIME_LIMIT_S,
        on_stage=on_stage,
    )
    return _jsonable(report)


def validate_scenario(params: dict) -> dict:
    """Parse a scenario request without running it. Raises ValueError if unsound.

    The API calls this inside the request so a malformed what-if returns 422
    immediately rather than a job id that fails two minutes later.
    """
    spec = _scenario_from(params)
    return {"label": spec.label, "changes": spec.changes()}
