"""
Stage 3B's headline number.

Runs every policy over the same evaluation window, against the same actual
demand, on the same network, and reports what each one cost.

**The comparison protocol, which is the part that matters.** "Our optimiser is
15% cheaper" means nothing on its own, because any inventory policy can be made
cheaper by holding less stock and serving fewer customers. So:

1. The CP-SAT policy runs first, at its service-level target, and whatever fill
   rate it actually *achieves* becomes the bar.
2. Every baseline is then tuned to clear that same bar as cheaply as it can -
   an exhaustive grid over both its safety factor and its review period, keeping
   its best configuration. Baselines get their best shot on purpose. A
   number won against a deliberately hobbled baseline is worse than no number,
   because it will not survive the first interview question about it.
3. Only then are costs compared, at equal achieved service level.

The perfect-information run gives the comparison a scale: the same controller,
fed the actual demand instead of the forecast, which separates what forecast
error costs from what the constraints make unavoidable. That is what turns a
percentage into a statement about where the remaining headroom actually is.

Run:
    python -m optimisation.benchmark                     # default 60 items
    python -m optimisation.benchmark --items 30 --sensitivity
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from simulation.simulate import simulate

from . import evalset
from .network import build_network, select_items
from .policies import (
    CpSatRollingHorizon, ForecastBaseStock, PerfectInformation, ReorderPointEOQ,
    SafetyStockBaseStock, interval_sigma,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BENCHMARK_PATH = MODELS_DIR / "optimiser_benchmark.json"

# Review periods each baseline is allowed to choose from. 1 is daily review; 7
# is the weekly cadence a real grocery DC tends to run.
REVIEW_GRID = (1, 2, 3, 4, 7, 14)

# Safety factors searched for each baseline. z = 0 is no safety stock; 3.0 is
# already a 99.9% one-sided normal target, so the grid spans everything a real
# policy would use. Step 0.125 keeps it to 25 points - a few hundred cheap
# simulations per policy, against a search that has to be exhaustive because
# aggregate fill rate is not monotone in z. See tune_to_service.
Z_GRID = np.round(np.arange(0.0, 3.0001, 0.125), 4)


def _series(evalframe, item_ids: list[str], column: str) -> dict[str, np.ndarray]:
    sub = evalframe[evalframe["item_id"].isin(item_ids)].sort_values(["item_id", "t"])
    return {i: g[column].to_numpy() for i, g in sub.groupby("item_id", observed=True)}


def tune_to_service(make_policy, network, actual, target_fill: float) -> tuple:
    """Cheapest configuration of a baseline that still clears `target_fill`.

    An exhaustive grid over safety factor and review period, keeping the cheapest
    combination that meets the bar.

    **This was a bisection on z and that was wrong.** Bisection needs fill rate
    to rise monotonically with the safety factor, which is true for a single item
    in isolation and false here: the DC, the shelf and the truck are shared, so
    raising one item's target makes it claim more of a fixed truck load and the
    simulator's proportional rescaling takes that capacity from other items,
    whose fill then drops. Aggregate fill is therefore not monotone in z, and the
    bisection would land in whichever branch it happened to bracket - converging
    on an expensive high-z configuration while a cheap one at z near zero met the
    same bar.

    The damage was not subtle. Re-running the benchmark six times, our own cost
    stayed inside +-2.3% and our fill rate inside 2.6 points, while the reported
    cost reduction swung between 24.3% and 50.0% - entirely because the baseline
    it was measured against jumped between ~$7,200 and ~$10,400 depending on
    which branch the bisection fell into. A grid is a few hundred simulations,
    all of them cheap, and it cannot miss the cheap configuration.
    """
    best = None
    for review in REVIEW_GRID:
        for z in Z_GRID:
            result = simulate(network, make_policy(float(z), review), actual)
            if result.fill_rate < target_fill:
                continue
            if best is None or result.total_cost < best[0].total_cost:
                best = (result, {"z": round(float(z), 4), "review_days": review})
    if best is None:
        # Report the least-bad configuration rather than silently omitting the
        # policy - a baseline that cannot reach the target is itself a finding.
        fallback = simulate(network, make_policy(float(Z_GRID[-1]), 1), actual)
        return fallback, {"z": float(Z_GRID[-1]), "review_days": 1,
                          "met_target": False}
    return best[0], best[1] | {"met_target": True}


def run(n_items: int = 60, horizon: int = 28, service_level: float = 0.95,
        resolve_every: int = 7, planning_horizon: int = 14,
        time_limit_s: float = 20.0, stockout_multiplier: float | None = None,
        holding_rate: float | None = None,
        item_ids: list[str] | None = None, quiet: bool = False) -> dict:
    """Benchmark every policy on one instance.

    `item_ids` names the items explicitly - what the API passes when a user picks
    them. Left out, the instance is the `n_items` largest by revenue.
    """
    evalframe, history, meta = evalset.load(horizon_days=horizon)
    if item_ids:
        available = set(history["item_id"])
        wanted = [i for i in item_ids if i in available]
        if not wanted:
            raise ValueError(
                "None of the requested items are in the evaluation window; "
                f"it covers {len(available)} items of store {meta['store']}.")
        # Still ranked and capped: a request for 400 items would build a solve
        # far too large to finish inside a web request's patience.
        chosen = select_items(history[history["item_id"].isin(wanted)], n_items)
    else:
        chosen = select_items(history, n_items)

    # Revenue share is always quoted against the whole modelled store, not
    # against whatever subset was requested - otherwise a one-item instance
    # would report covering 100% of the store's revenue.
    all_revenue = (history["mean_daily"] * history["sell_price"]).sum()
    revenue_share = float(
        (chosen["mean_daily"] * chosen["sell_price"]).sum() / all_revenue)
    kwargs = {}
    if stockout_multiplier is not None:
        kwargs["stockout_multiplier"] = stockout_multiplier
    if holding_rate is not None:
        kwargs["holding_rate"] = holding_rate
    network = build_network(chosen, meta["store"], **kwargs)
    ids = network.item_ids

    actual = _series(evalframe, ids, "demand")
    point = _series(evalframe, ids, "point")
    lower = _series(evalframe, ids, "lower")
    upper = _series(evalframe, ids, "upper")
    sigma = {i: interval_sigma(lower[i], upper[i]) for i in ids}

    def say(*a):
        if not quiet:
            print(*a)

    say(f"\nInstance: {len(ids)} items, {horizon} days, store {meta['store']}, "
        f"window {meta['window_start']}..{meta['window_end']}")
    say(f"  {revenue_share:.1%} of the store's modelled revenue; "
        f"{sum(int(v.sum()) for v in actual.values()):,} actual units")
    say(f"  DC capacity {network.dc_capacity} units, truck "
        f"{network.truck_capacity} units/day")

    # --- 1. Ours, first: its achieved fill rate sets the bar. -------------
    say("\nSolving CP-SAT rolling horizon "
        f"(plan {planning_horizon}d / commit {resolve_every}d, "
        f"{time_limit_s:.0f}s per solve)...")
    t0 = time.time()
    ours = simulate(network, CpSatRollingHorizon(
        point, service_level=service_level, resolve_every=resolve_every,
        planning_horizon=planning_horizon, time_limit_s=time_limit_s,
    ), actual, keep_daily=True)
    say(f"  {time.time() - t0:.1f}s wall, fill rate {ours.fill_rate:.2%}, "
        f"cost ${ours.total_cost:,.2f}")
    say(f"  solver: {ours.solver}")

    target = ours.fill_rate

    # --- 2. Baselines, each tuned to match that fill rate. ----------------
    say(f"\nTuning baselines to match {target:.2%} fill rate...")
    baselines = {}

    result, config = tune_to_service(
        lambda k, r: ReorderPointEOQ(k=k, review_days=r), network, actual, target)
    baselines["reorder_point_eoq"] = (result, config)
    say(f"  reorder_point_eoq       ${result.total_cost:>9,.2f}  "
        f"fill {result.fill_rate:.2%}  {config}")

    result, config = tune_to_service(
        lambda z, r: SafetyStockBaseStock(z=z, review_days=r), network, actual, target)
    baselines["safety_stock_base_stock"] = (result, config)
    say(f"  safety_stock_base_stock ${result.total_cost:>9,.2f}  "
        f"fill {result.fill_rate:.2%}  {config}")

    result, config = tune_to_service(
        lambda z, r: ForecastBaseStock(point, sigma, z=z, review_days=r),
        network, actual, target)
    baselines["forecast_base_stock"] = (result, config)
    say(f"  forecast_base_stock     ${result.total_cost:>9,.2f}  "
        f"fill {result.fill_rate:.2%}  {config}")

    # --- 3. The value-of-information reference. --------------------------
    say("\nSolving perfect information (same controller, actual demand)...")
    bound = simulate(network, PerfectInformation(
        actual, service_level=service_level, resolve_every=resolve_every,
        planning_horizon=planning_horizon, time_limit_s=time_limit_s,
    ), actual)
    say(f"  cost ${bound.total_cost:,.2f}, fill {bound.fill_rate:.2%}, "
        f"status {bound.solver.get('statuses')}")

    # --- 4. Assemble. -----------------------------------------------------
    primary = baselines["reorder_point_eoq"][0]
    best_baseline_name, best_baseline = min(
        ((k, v[0]) for k, v in baselines.items()), key=lambda kv: kv[1].total_cost)

    def reduction(base) -> float:
        return (base.total_cost - ours.total_cost) / base.total_cost * 100

    policies = {"cpsat_rolling_horizon": ours.headline()
                | {"config": {"service_level": service_level,
                              "resolve_every": resolve_every,
                              "planning_horizon": planning_horizon}}}
    for name, (result, config) in baselines.items():
        policies[name] = result.headline() | {"config": config}
    policies["perfect_information"] = bound.headline() | {
        "config": {"note": "same CP-SAT controller, fed actual demand"}}

    report = {
        "instance": {
            "store": meta["store"],
            "items": len(ids),
            "revenue_share_of_modelled_store": round(revenue_share, 4),
            "horizon_days": horizon,
            "window": [meta["window_start"], meta["window_end"]],
            "actual_units": int(sum(int(v.sum()) for v in actual.values())),
            "forecast_units": round(float(sum(v.sum() for v in point.values())), 1),
            "evaluation": (
                "policies decided on LightGBM forecasts from a model trained "
                "strictly before the window; costs scored on actual M5 sales"),
        },
        "network": network.summary(),
        "protocol": {
            "matched_on": "achieved aggregate fill rate",
            "target_fill_rate": round(target, 4),
            "baseline_tuning": (
                "exhaustive grid over safety factor (0 to 3.0, step 0.125) and "
                "review period; each baseline keeps its cheapest configuration "
                "that still meets the fill rate our policy achieved. The grid is "
                "exhaustive because aggregate fill rate is not monotone in the "
                "safety factor under shared truck and DC capacity"),
            "review_grid": list(REVIEW_GRID),
            "z_grid": [float(z) for z in Z_GRID],
            # --- How well the match actually held.
            #
            # A baseline can only be tuned *down* to zero safety stock. If it
            # still over-serves at z=0 - which happens on small instances, where
            # a single case pack is many days of cover - then the cheapest
            # configuration meeting our fill rate overshoots it, pays for service
            # we did not buy, and flatters us. The overshoot is therefore
            # reported rather than left for a reader to spot in the fill column,
            # and the comparison is flagged when it exceeds 1.5 points.
            "baseline_fill_overshoot_pp": {
                name: round((result.fill_rate - target) * 100, 2)
                for name, (result, _) in baselines.items()
            },
            "matched_within_tolerance": bool(
                (primary.fill_rate - target) * 100 <= 1.5),
            "tolerance_pp": 1.5,
            "match_caveat": (
                "Baselines cannot be tuned below zero safety stock, so on small "
                "instances they over-serve at their cheapest feasible setting "
                "and the cost comparison is not strictly at equal service. "
                "Check baseline_fill_overshoot_pp before quoting the headline; "
                "it is ~2.5pp on the 60-item instance and much larger on "
                "few-item ones."
            ),
        },
        "policies": policies,
        "headline": {
            "cost_reduction_vs_reorder_point_pct": round(reduction(primary), 2),
            "cost_reduction_vs_best_baseline_pct": round(reduction(best_baseline), 2),
            "best_baseline": best_baseline_name,
            "cost_reduction_vs_forecast_base_stock_pct": round(
                reduction(baselines["forecast_base_stock"][0]), 2),
            "gap_to_perfect_information_pct": round(
                (ours.total_cost - bound.total_cost) / ours.total_cost * 100, 2),
            "at_fill_rate": round(target, 4),
        },
        "perfect_information_caveat": (
            # This does happen, and on small instances it happens often enough
            # that leaving it unexplained would look like a broken benchmark.
            "Not a lower bound, and deliberately not presented as one. It is the "
            "same receding-horizon controller, which plans 14 days, commits 7, "
            "and is therefore myopic about everything past the planning window - "
            "so perfect demand knowledge inside that window does not guarantee a "
            "cheaper outcome over the full horizon, and a forecast error that "
            "happens to favour the committed decisions can beat it. Its job is "
            "to price the value of information, not to bound the problem."
            + ("" if ours.total_cost >= bound.total_cost else
               f" On this instance that is exactly what happened: our policy came "
               f"in {(bound.total_cost - ours.total_cost) / bound.total_cost * 100:.1f}% "
               f"below the perfect-information run.")
        ),
        "ablation": {
            "note": (
                "forecast_base_stock uses the identical decision rule to "
                "safety_stock_base_stock and differs only in taking its "
                "lead-time demand from the LightGBM forecast, so the step "
                "between them prices the forecast and the step from it to "
                "CP-SAT prices the optimiser. The two percentages have "
                "different denominators, named in their keys, because the "
                "chain of policies is not monotone in cost - stating a single "
                "shared denominator would imply a decomposition that does not "
                "hold."),
            "forecast_step_pct_of_safety_stock": round(
                (baselines["safety_stock_base_stock"][0].total_cost
                 - baselines["forecast_base_stock"][0].total_cost)
                / baselines["safety_stock_base_stock"][0].total_cost * 100, 2),
            "optimiser_step_pct_of_forecast_base_stock": round(
                reduction(baselines["forecast_base_stock"][0]), 2),
            "dollars": {
                "safety_stock_base_stock": round(
                    baselines["safety_stock_base_stock"][0].total_cost, 2),
                "forecast_base_stock": round(
                    baselines["forecast_base_stock"][0].total_cost, 2),
                "cpsat_rolling_horizon": round(ours.total_cost, 2),
                "forecast_step_usd": round(
                    baselines["safety_stock_base_stock"][0].total_cost
                    - baselines["forecast_base_stock"][0].total_cost, 2),
                "optimiser_step_usd": round(
                    baselines["forecast_base_stock"][0].total_cost
                    - ours.total_cost, 2),
            },
        },
        "cost_mechanism": {
            "note": (
                "where the saving actually comes from. Holding cost is a "
                "rounding error at grocery margins; the optimiser wins by "
                "consolidating the fixed costs a per-item rule cannot see."),
            "po_lines": {name: result.po_lines
                         for name, (result, _) in baselines.items()}
                        | {"cpsat_rolling_horizon": ours.po_lines},
            "delivery_days": {name: result.delivery_days
                              for name, (result, _) in baselines.items()}
                             | {"cpsat_rolling_horizon": ours.delivery_days},
        },
        "solver": ours.solver,
        "feasibility_repairs": {name: (result.clips if hasattr(result, "clips") else {})
                                for name, (result, _) in baselines.items()}
                               | {"cpsat_rolling_horizon": ours.clips},
        "per_item_fill_quantiles": {
            q: round(float(np.quantile(list(ours.per_item_fill.values()), float(q))), 4)
            for q in ("0.05", "0.25", "0.5", "0.75")
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3B policy benchmark")
    parser.add_argument("--items", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--service-level", type=float, default=0.95)
    parser.add_argument("--resolve-every", type=int, default=7)
    parser.add_argument("--planning-horizon", type=int, default=14)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--sensitivity", action="store_true",
                        help="re-run at other stockout multipliers")
    parser.add_argument("--repeat", type=int, default=1,
                        help="re-run the whole benchmark N times and report the "
                             "spread of the headline number. CP-SAT searches on "
                             "several threads under a wall-clock limit and does "
                             "not prove optimality at this size, so each run "
                             "commits a slightly different plan. Quote the "
                             "spread, not a single run.")
    parser.add_argument("--out", type=Path, default=BENCHMARK_PATH)
    args = parser.parse_args()

    report = run(n_items=args.items, horizon=args.horizon,
                 service_level=args.service_level, resolve_every=args.resolve_every,
                 planning_horizon=args.planning_horizon, time_limit_s=args.time_limit)

    print("\n" + "=" * 78)
    print(f"{'Policy':<26}{'Total cost':>12}{'Fill':>8}{'Holding':>10}"
          f"{'Stockout':>10}{'Deliv days':>12}")
    print("-" * 78)
    order = ["reorder_point_eoq", "safety_stock_base_stock", "forecast_base_stock",
             "cpsat_rolling_horizon", "perfect_information"]
    for name in order:
        p = report["policies"][name]
        print(f"{name:<26}{p['total_cost']:>12,.2f}{p['fill_rate']:>8.1%}"
              f"{p['holding']:>10,.2f}{p['stockout']:>10,.2f}{p['delivery_days']:>12}")
    print("=" * 78)
    h = report["headline"]
    print(f"\nCost reduction vs fixed reorder point: "
          f"{h['cost_reduction_vs_reorder_point_pct']:.1f}%  "
          f"(at {h['at_fill_rate']:.1%} fill rate)")
    proto = report["protocol"]
    overshoot = proto["baseline_fill_overshoot_pp"]["reorder_point_eoq"]
    if not proto["matched_within_tolerance"]:
        print(f"  WARNING: that baseline over-serves by {overshoot:.1f}pp. It "
              f"cannot be tuned below zero safety stock, so this is not strictly "
              f"an equal-service comparison and the figure flatters us. Use a "
              f"larger instance before quoting it.")
    else:
        print(f"  (baseline over-serves by {overshoot:.1f}pp, within tolerance)")
    print(f"Cost reduction vs best baseline ({h['best_baseline']}): "
          f"{h['cost_reduction_vs_best_baseline_pct']:.1f}%")
    ab = report["ablation"]
    print(f"Ablation - swapping history for the forecast in the same rule saves "
          f"{ab['forecast_step_pct_of_safety_stock']:.1f}% "
          f"(${ab['dollars']['forecast_step_usd']:,.2f}); replacing that rule "
          f"with the optimiser saves a further "
          f"{ab['optimiser_step_pct_of_forecast_base_stock']:.1f}% "
          f"(${ab['dollars']['optimiser_step_usd']:,.2f})")
    mech = report["cost_mechanism"]
    print(f"Mechanism - PO lines {mech['po_lines']}, "
          f"delivery days {mech['delivery_days']}")
    print(f"Cost of forecast error (gap to perfect information): "
          f"{h['gap_to_perfect_information_pct']:.1f}%")

    if args.repeat > 1:
        print(f"\nRepeating the benchmark {args.repeat}x to measure run-to-run "
              f"spread (CP-SAT is not deterministic here)...")
        values, fills, costs = [], [], []
        for k in range(args.repeat):
            rep = run(n_items=args.items, horizon=args.horizon,
                      service_level=args.service_level,
                      resolve_every=args.resolve_every,
                      planning_horizon=args.planning_horizon,
                      time_limit_s=args.time_limit, quiet=True)
            value = rep["headline"]["cost_reduction_vs_reorder_point_pct"]
            values.append(value)
            fills.append(rep["headline"]["at_fill_rate"])
            costs.append(rep["policies"]["cpsat_rolling_horizon"]["total_cost"])
            print(f"  run {k + 1}/{args.repeat}: {value:>6.1f}% reduction  "
                  f"at fill {rep['headline']['at_fill_rate']:.1%}  "
                  f"our cost ${costs[-1]:,.2f}")
        report["repeats"] = {
            "runs": args.repeat,
            "reduction_vs_reorder_point_pct": {
                "mean": round(float(np.mean(values)), 2),
                "min": round(float(np.min(values)), 2),
                "max": round(float(np.max(values)), 2),
                "std": round(float(np.std(values, ddof=1)), 2) if len(values) > 1 else 0.0,
                "values": values,
            },
            "our_cost_usd": {
                "mean": round(float(np.mean(costs)), 2),
                "min": round(float(np.min(costs)), 2),
                "max": round(float(np.max(costs)), 2),
            },
            "achieved_fill_rate": {
                "mean": round(float(np.mean(fills)), 4),
                "min": round(float(np.min(fills)), 4),
                "max": round(float(np.max(fills)), 4),
            },
            "why": (
                "CP-SAT searches on 8 threads under a wall-clock limit and does "
                "not prove optimality at this size, so each run commits a "
                "slightly different plan and our own cost moves by a couple of "
                "percent. The baselines are deterministic given the service bar, "
                "so any spread much wider than that is a measurement artifact "
                "rather than solver noise - which is exactly how the bisection "
                "bug in DECISIONS.md entry 23 was found. Report the mean and "
                "range, and investigate a wide one."
            ),
        }
        print(f"\n  mean {np.mean(values):.1f}%, range "
              f"{np.min(values):.1f}-{np.max(values):.1f}%, "
              f"sd {np.std(values, ddof=1):.1f} over {args.repeat} runs")

    if args.sensitivity:
        print("\nSensitivity to the stockout multiplier (the softest assumption):")
        sensitivity = {}
        for multiplier in (1.0, 2.0, 4.0):
            alt = run(n_items=args.items, horizon=args.horizon,
                      service_level=args.service_level,
                      resolve_every=args.resolve_every,
                      planning_horizon=args.planning_horizon,
                      time_limit_s=args.time_limit,
                      stockout_multiplier=multiplier, quiet=True)
            sensitivity[str(multiplier)] = alt["headline"]
            print(f"  x{multiplier:<4} reduction vs reorder point "
                  f"{alt['headline']['cost_reduction_vs_reorder_point_pct']:>6.1f}%  "
                  f"at fill {alt['headline']['at_fill_rate']:.1%}")
        report["sensitivity_stockout_multiplier"] = sensitivity

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWritten to {args.out.relative_to(args.out.parent.parent)}")


if __name__ == "__main__":
    main()
