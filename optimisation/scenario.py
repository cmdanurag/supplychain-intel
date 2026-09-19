"""
Stage 3C, first half: what-if scenarios.

**The question this answers.** Stage 3B says what a good replenishment policy
costs under the network as it is. The question an operations manager actually
asks is the next one: *what happens if it changes?* If the supplier's lead time
doubles, if we lose a third of the truck, if demand jumps 20% before a festival
- what does that do to cost and to service, and where does it hurt?

**How it works, and what it deliberately is not.** A scenario is a small,
validated set of parameter changes. It rebuilds the network with those changes,
re-runs the same CP-SAT rolling-horizon policy against the same actual demand,
and diffs the two outcomes. Nothing about the optimiser, the simulator or the
cost model is special-cased for scenarios: the whole point is that the machinery
already built is what answers the question, so the answer is as trustworthy as
the Stage 3B benchmark is.

It does **not** re-tune the classical baselines. Tuning is 150 simulations per
baseline, which is right for a headline number measured once and wrong for an
interactive question. A scenario compares our policy against itself, which is
the honest framing anyway: "here is what this change costs us", not "here is a
new percentage to quote".

**Why demand scaling does not scale capacity.** Capacities are sized from the
measured mean demand of the base instance, so a demand surge runs against the
same warehouse, truck and shelf. That squeeze is the interesting part; scaling
the capacity with the demand would answer a question nobody asked.

Run:
    python -m optimisation.scenario --preset supplier_lead_doubles --items 20
    python -m optimisation.scenario --list
"""
import argparse
import json
import time
from dataclasses import dataclass, asdict, replace

import numpy as np

from simulation.simulate import simulate

from . import evalset
from .network import (
    ANNUAL_HOLDING_RATE, DC_CAPACITY_DAYS, DELIVERY_FIXED_COST,
    ORDER_COST_PER_LINE, SHELF_CAPACITY_DAYS, STOCKOUT_MULTIPLIER,
    STORE_LEAD_DAYS, SUPPLIER_LEAD_DAYS, TRUCK_CAPACITY_DAYS,
    build_network, select_items,
)
from .policies import CpSatRollingHorizon

# Bounds on every field. A scenario outside these is refused rather than
# clamped: silently solving a different question than the one asked is the
# failure mode this whole module exists to avoid. The ranges are wide enough
# for any realistic disruption and narrow enough that the solve still finishes.
BOUNDS = {
    "supplier_lead_days": (1, 28),
    "store_lead_days": (0, 14),
    "demand_pct": (0.5, 2.0),
    "dc_capacity_pct": (0.3, 3.0),
    "truck_capacity_pct": (0.3, 3.0),
    "shelf_capacity_pct": (0.3, 3.0),
    "order_cost_pct": (0.0, 5.0),
    "delivery_cost_pct": (0.0, 5.0),
    "stockout_multiplier": (0.5, 20.0),
    "annual_holding_rate": (0.0, 2.0),
    "service_level": (0.5, 0.999),
}


@dataclass(frozen=True)
class Scenario:
    """One what-if, as a validated set of changes to the base network.

    Multipliers (`*_pct`) are relative to the base instance; lead times and the
    service level are absolute, because "lead time becomes 14 days" is how the
    question is actually asked. A field left at its default changes nothing.
    """
    label: str = "custom"
    supplier_lead_days: int = SUPPLIER_LEAD_DAYS
    store_lead_days: int = STORE_LEAD_DAYS
    demand_pct: float = 1.0
    dc_capacity_pct: float = 1.0
    truck_capacity_pct: float = 1.0
    shelf_capacity_pct: float = 1.0
    order_cost_pct: float = 1.0
    delivery_cost_pct: float = 1.0
    stockout_multiplier: float = STOCKOUT_MULTIPLIER
    annual_holding_rate: float = ANNUAL_HOLDING_RATE
    service_level: float = 0.95

    def __post_init__(self):
        for field_name, (low, high) in BOUNDS.items():
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be a number, got {value!r}")
            if not low <= value <= high:
                raise ValueError(
                    f"{field_name}={value} is outside the allowed range "
                    f"[{low}, {high}]")

    def changes(self, base: "Scenario | None" = None) -> dict[str, str]:
        """Only the fields this scenario actually moves, phrased for a human."""
        base = base or Scenario()
        out = {}
        for field_name in BOUNDS:
            mine, theirs = getattr(self, field_name), getattr(base, field_name)
            if mine == theirs:
                continue
            if field_name.endswith("_pct"):
                out[field_name] = f"{theirs:.0%} -> {mine:.0%} of base"
            elif field_name == "service_level":
                out[field_name] = f"{theirs:.0%} -> {mine:.0%}"
            else:
                out[field_name] = f"{theirs:g} -> {mine:g}"
        return out

    def network_kwargs(self) -> dict:
        return {
            "stockout_multiplier": self.stockout_multiplier,
            "holding_rate": self.annual_holding_rate,
            "supplier_lead_days": int(self.supplier_lead_days),
            "store_lead_days": int(self.store_lead_days),
            "dc_capacity_days": DC_CAPACITY_DAYS * self.dc_capacity_pct,
            "truck_capacity_days": TRUCK_CAPACITY_DAYS * self.truck_capacity_pct,
            "shelf_capacity_days": SHELF_CAPACITY_DAYS * self.shelf_capacity_pct,
            "order_cost_per_line": ORDER_COST_PER_LINE * self.order_cost_pct,
            "delivery_fixed_cost": DELIVERY_FIXED_COST * self.delivery_cost_pct,
        }


# Presets are the questions worth asking out loud, not a menu of every knob.
# Each one is a disruption an operations team has actually lived through.
PRESETS: dict[str, tuple[str, dict]] = {
    "supplier_lead_doubles": (
        "Supplier lead time doubles, 7 days to 14",
        {"supplier_lead_days": 14}),
    "truck_capacity_down_30": (
        "Lose 30% of daily truck capacity",
        {"truck_capacity_pct": 0.7}),
    "demand_surge_20": (
        "Demand runs 20% above plan",
        {"demand_pct": 1.2}),
    "dc_space_down_40": (
        "DC storage cut by 40%",
        {"dc_capacity_pct": 0.6}),
    "freight_costs_double": (
        "Delivery cost doubles, $60 to $120 a day",
        {"delivery_cost_pct": 2.0}),
    "premium_service": (
        "Raise the service target from 95% to 98%",
        {"service_level": 0.98}),
    "perfect_storm": (
        "Lead time up, truck down, demand up, all at once",
        {"supplier_lead_days": 14, "truck_capacity_pct": 0.7, "demand_pct": 1.2}),
}


def preset(name: str, base: Scenario | None = None) -> Scenario:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    label, changes = PRESETS[name]
    return replace(base or Scenario(), label=label, **changes)


def _series(evalframe, item_ids: list[str], column: str) -> dict[str, np.ndarray]:
    sub = evalframe[evalframe["item_id"].isin(item_ids)].sort_values(["item_id", "t"])
    return {i: g[column].to_numpy() for i, g in sub.groupby("item_id", observed=True)}


def _scale(series: dict[str, np.ndarray], factor: float) -> dict[str, np.ndarray]:
    """Scale demand, keeping units integral.

    Rounding each day independently is right here: demand is a count, and a
    cumulative scheme would smear a surge across days that did not see it.
    """
    if factor == 1.0:
        return series
    return {i: np.rint(v * factor).astype(int) for i, v in series.items()}


def _outcome(result, network, elapsed: float) -> dict:
    # solver_stats() flattens every re-solve's notes into one list; the same
    # constraint relaxed in three consecutive solves is one fact, not three.
    notes = list(result.solver.get("relaxations", []))
    relaxations = sorted(set(notes))
    # The bottom of the ladder is a different kind of answer from a rung of it.
    # "No feasible plan found" means that re-solve committed nothing at all, so
    # the week it covered ran on whatever stock happened to be on the shelf -
    # a physical impossibility, not a costed trade-off, and the one result a
    # reader must not mistake for an expensive-but-working plan.
    gave_up = sum(1 for n in notes if "no feasible plan found" in n)
    return {
        "total_cost": round(result.total_cost, 2),
        "costs": {k: round(v, 2) for k, v in result.costs.items()},
        "fill_rate": round(result.fill_rate, 4),
        "units_short": int(result.units_short),
        "units_demanded": int(result.units_demanded),
        "po_lines": int(result.po_lines),
        "delivery_days": int(result.delivery_days),
        "dc_utilisation_peak": round(result.dc_utilisation_peak, 3),
        "truck_utilisation_mean": round(result.truck_utilisation_mean, 3),
        "mean_store_inventory": round(result.mean_store_inventory, 1),
        "relaxations": relaxations,
        "solves_with_no_feasible_plan": gave_up,
        "network": {
            "supplier_lead_days": network.supplier_lead_days,
            "store_lead_days": network.store_lead_days,
            "dc_capacity_units": network.dc_capacity,
            "truck_capacity_units_per_day": network.truck_capacity,
            "order_cost_per_line": network.order_cost_per_line,
            "delivery_fixed_cost": network.delivery_fixed_cost,
        },
        "solve_seconds": round(elapsed, 1),
    }


def _delta(base: dict, alt: dict) -> dict:
    def pct(a, b):
        return None if a == 0 else round((b - a) / abs(a) * 100, 2)
    return {
        "cost_usd": round(alt["total_cost"] - base["total_cost"], 2),
        "cost_pct": pct(base["total_cost"], alt["total_cost"]),
        "fill_rate_pp": round((alt["fill_rate"] - base["fill_rate"]) * 100, 2),
        "units_short": alt["units_short"] - base["units_short"],
        "po_lines": alt["po_lines"] - base["po_lines"],
        "delivery_days": alt["delivery_days"] - base["delivery_days"],
        "by_cost_line": {
            k: round(alt["costs"].get(k, 0.0) - v, 2)
            for k, v in base["costs"].items()
        },
    }


def _headline(scenario: Scenario, base: dict, alt: dict, delta: dict) -> str:
    """One sentence a manager can read without the table underneath."""
    cost = delta["cost_pct"]
    direction = "more" if (cost or 0) >= 0 else "less"
    fill = delta["fill_rate_pp"]
    service = (f"service holds at {alt['fill_rate']:.1%}" if abs(fill) < 0.5
               else f"service moves {fill:+.1f} points to {alt['fill_rate']:.1%}")
    tail = ""
    if alt["relaxations"] and not base["relaxations"]:
        tail = (" The plan could only be built by relaxing a service "
                "constraint, so this scenario is at the edge of feasibility.")
    extra = alt["solves_with_no_feasible_plan"] - base["solves_with_no_feasible_plan"]
    if extra > 0:
        tail += (f" {extra} weekly re-solve(s) found no feasible plan at all and "
                 f"committed nothing, so this is beyond what the network can "
                 f"absorb, not merely expensive.")
    return (f"{scenario.label}: costs {abs(cost or 0):.1f}% {direction} "
            f"(${base['total_cost']:,.0f} -> ${alt['total_cost']:,.0f}) and "
            f"{service}.{tail}")


def compare(
    scenario: Scenario,
    item_ids: list[str] | None = None,
    n_items: int = 20,
    horizon: int = 28,
    resolve_every: int = 7,
    planning_horizon: int = 14,
    time_limit_s: float = 10.0,
    base: Scenario | None = None,
    on_stage=None,
) -> dict:
    """Run the same policy under `base` and under `scenario`, and diff them.

    Both runs face the same actual demand series (scaled, if the scenario says
    so) and the same forecasts, so every difference in the result is caused by
    the scenario and not by re-drawing anything.
    """
    base = base or Scenario(label="base network")
    evalframe, history, meta = evalset.load(horizon_days=horizon)

    if item_ids:
        available = set(history["item_id"])
        wanted = [i for i in item_ids if i in available]
        if not wanted:
            raise ValueError(
                "None of the requested items are in the evaluation window; "
                f"it covers {len(available)} items of store {meta['store']}.")
        chosen = select_items(history[history["item_id"].isin(wanted)], n_items)
    else:
        chosen = select_items(history, n_items)

    results = {}
    for index, (key, spec) in enumerate((("base", base), ("scenario", scenario))):
        # Each leg is a minute of solving on a small host, so the caller gets a
        # say-what-you-are-doing hook rather than a silent wait.
        if on_stage is not None:
            on_stage(key, index)
        network = build_network(chosen, meta["store"], **spec.network_kwargs())
        ids = network.item_ids
        actual = _scale(_series(evalframe, ids, "demand"), spec.demand_pct)
        point = _scale(_series(evalframe, ids, "point"), spec.demand_pct)
        # The planning window has to outlast the inbound lead time, or a supplier
        # order placed on day 0 arrives after the last day the solver can see and
        # therefore never earns anything: measured, a 14-day lead time against a
        # 14-day plan ordered *nothing at all* for the whole window. Stretching
        # the horizon to lead + one commit period is the fix, and it is the same
        # rule a planner follows by hand.
        horizon_used = max(planning_horizon, int(spec.supplier_lead_days) + resolve_every)
        t0 = time.time()
        run = simulate(network, CpSatRollingHorizon(
            point, service_level=spec.service_level, resolve_every=resolve_every,
            planning_horizon=horizon_used, time_limit_s=time_limit_s,
        ), actual)
        results[key] = _outcome(run, network, time.time() - t0)
        results[key]["planning_horizon_days"] = horizon_used

    delta = _delta(results["base"], results["scenario"])
    return {
        "scenario": {"label": scenario.label, "changes": scenario.changes(base),
                     "spec": asdict(scenario)},
        "instance": {
            "store": meta["store"],
            "items": len(chosen),
            "horizon_days": horizon,
            "window": [meta["window_start"], meta["window_end"]],
            "evaluation": (
                "both runs are scored on the same actual M5 sales; the scenario "
                "changes the network, not the demand realisation, unless "
                "demand_pct says otherwise"),
        },
        "base": results["base"],
        "scenario_result": results["scenario"],
        "delta": delta,
        "headline": _headline(scenario, results["base"], results["scenario"], delta),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Re-solve under a what-if scenario.")
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--list", action="store_true", help="show the presets")
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--supplier-lead-days", type=int)
    parser.add_argument("--demand-pct", type=float)
    parser.add_argument("--truck-capacity-pct", type=float)
    parser.add_argument("--dc-capacity-pct", type=float)
    parser.add_argument("--service-level", type=float)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()

    if args.list:
        for name, (label, changes) in PRESETS.items():
            print(f"  {name:<24} {label}")
        return

    if args.preset:
        scenario = preset(args.preset)
    else:
        overrides = {k: v for k, v in {
            "supplier_lead_days": args.supplier_lead_days,
            "demand_pct": args.demand_pct,
            "truck_capacity_pct": args.truck_capacity_pct,
            "dc_capacity_pct": args.dc_capacity_pct,
            "service_level": args.service_level,
        }.items() if v is not None}
        if not overrides:
            parser.error("give --preset, or at least one override; --list shows presets")
        scenario = Scenario(label="custom scenario", **overrides)

    report = compare(scenario, n_items=args.items, horizon=args.horizon,
                     time_limit_s=args.time_limit)

    print(f"\n{report['headline']}\n")
    print(f"  changes: {report['scenario']['changes']}")
    base, alt, delta = report["base"], report["scenario_result"], report["delta"]
    print(f"\n  {'':<22}{'base':>12}{'scenario':>12}{'delta':>12}")
    rows = [
        ("total cost", f"${base['total_cost']:,.0f}", f"${alt['total_cost']:,.0f}",
         f"{delta['cost_pct']:+.1f}%"),
        ("fill rate", f"{base['fill_rate']:.1%}", f"{alt['fill_rate']:.1%}",
         f"{delta['fill_rate_pp']:+.1f}pp"),
        ("units short", f"{base['units_short']:,}", f"{alt['units_short']:,}",
         f"{delta['units_short']:+,}"),
        ("PO lines", base["po_lines"], alt["po_lines"], f"{delta['po_lines']:+}"),
        ("delivery days", base["delivery_days"], alt["delivery_days"],
         f"{delta['delivery_days']:+}"),
        ("truck utilisation", f"{base['truck_utilisation_mean']:.0%}",
         f"{alt['truck_utilisation_mean']:.0%}", ""),
    ]
    for name, b, s, d in rows:
        print(f"  {name:<22}{b:>12}{s:>12}{d:>12}")
    if alt["relaxations"]:
        print(f"\n  relaxed to stay feasible: {'; '.join(alt['relaxations'])}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n  written to {args.out}")


if __name__ == "__main__":
    _cli()
