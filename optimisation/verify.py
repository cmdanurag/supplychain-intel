"""
Invariant checks for Stage 3B. Run after touching the solver or the simulator.

    python -m optimisation.verify

There are two independent models of the same physical system in this stage: the
CP-SAT formulation in cpsat.py, which *plans*, and the simulator in
simulation/simulate.py, which *scores*. They were written separately and they
have to agree, because the benchmark's whole claim rests on the cost a plan is
charged being the cost that plan really incurs.

They did not agree at first. The solver capped shelf space on end-of-day
inventory while the simulator capped it on inventory position, so the solver
planned shipments the simulator then clipped, and the perfect-information policy
came back at 74% fill against demand it knew exactly (DECISIONS.md entry 15).
Nothing raised an error - the policy was just quietly worse. That class of bug is
invisible to a unit test of either component on its own, and it is what this
file exists to catch.

Deliberately dependency-free and runnable, rather than a pytest suite, matching
how the rest of the repo is driven. Exits non-zero on failure so CI can use it.
"""
import sys

import numpy as np

from simulation.simulate import simulate

from . import evalset
from .cpsat import State, integerise_demand, solve
from .network import build_network, select_items
from .policies import Policy, ReorderPointEOQ

TOLERANCE_USD = 0.02


class _Replay(Policy):
    """Executes a fixed plan open-loop, so a plan can be scored by the simulator."""
    name = "replay"

    def __init__(self, plan):
        self.plan = plan

    def decide(self, state):
        t = state.t
        if t >= self.plan.horizon:
            return {}, {}
        return ({i: int(v[t]) for i, v in self.plan.orders.items()},
                {i: int(v[t]) for i, v in self.plan.shipments.items()})


def _instance(n_items: int = 8, horizon: int = 14):
    evalframe, history, meta = evalset.load()
    chosen = select_items(history, n_items)
    network = build_network(chosen, meta["store"])
    ids = network.item_ids
    sub = evalframe[evalframe["item_id"].isin(ids)].sort_values(["item_id", "t"])
    point = {i: g["point"].to_numpy()[:horizon]
             for i, g in sub.groupby("item_id", observed=True)}
    actual = {i: g["demand"].to_numpy()[:horizon]
              for i, g in sub.groupby("item_id", observed=True)}
    return network, point, actual, horizon


def check_cost_models_agree(network, point, horizon) -> list[str]:
    """The simulator must charge a plan exactly what the solver said it costs.

    Run against the same demand the plan was built on and from the same opening
    inventory, the two must match to the cent, term by term. A gap means the
    models disagree about the physics, and every cost comparison built on them
    is suspect.
    """
    failures = []
    demand = {i: integerise_demand(point[i]) for i in network.item_ids}
    state = State.initial(network, horizon)
    plan = solve(network, demand, state, time_limit_s=25)

    if plan.status not in ("OPTIMAL", "FEASIBLE"):
        return [f"solver returned {plan.status} on the verification instance"]

    result = simulate(network, _Replay(plan), demand,
                      initial_inv_dc=state.inv_dc,
                      initial_inv_store=state.inv_store)

    for term, planned in plan.planned_costs.items():
        if term == "total":
            continue
        realised = result.costs.get(term, 0.0)
        if abs(planned - realised) > TOLERANCE_USD:
            failures.append(
                f"cost term '{term}' disagrees: solver ${planned:,.2f} vs "
                f"simulator ${realised:,.2f}")

    total_gap = abs(plan.planned_costs["total"] - result.total_cost)
    if total_gap > TOLERANCE_USD:
        failures.append(
            f"total cost disagrees by ${total_gap:,.2f} "
            f"(solver ${plan.planned_costs['total']:,.2f}, "
            f"simulator ${result.total_cost:,.2f})")

    # A plan the solver considers feasible must need no repair on replay. Any
    # clip means the simulator enforces a constraint the solver does not.
    if any(result.clips.values()):
        failures.append(
            f"simulator had to repair a solver-feasible plan: {result.clips} - "
            f"the two disagree about a constraint, not just a cost")
    return failures


def check_simulator_conserves_units(network, actual, horizon) -> list[str]:
    """Nothing may be created or destroyed: sold + short must equal demand, and
    stock must be accounted for across both echelons."""
    failures = []
    result = simulate(network, ReorderPointEOQ(k=1.64, review_days=3), actual,
                      keep_daily=True)
    demanded = sum(int(v[:horizon].sum()) for v in actual.values())
    if result.units_sold + result.units_short != demanded:
        failures.append(
            f"unit conservation broken: sold {result.units_sold} + short "
            f"{result.units_short} != demanded {demanded}")
    if result.units_demanded != demanded:
        failures.append(
            f"reported demand {result.units_demanded} != actual {demanded}")
    if not 0.0 <= result.fill_rate <= 1.0:
        failures.append(f"fill rate out of range: {result.fill_rate}")
    return failures


def check_shelf_and_capacity_respected(network, actual, horizon) -> list[str]:
    """After repair, no policy may exceed shelf, DC or truck capacity."""
    failures = []
    result = simulate(network, ReorderPointEOQ(k=2.0, review_days=1), actual,
                      keep_daily=True)
    for day in result.daily:
        if day["dc_inventory"] > network.dc_capacity:
            failures.append(
                f"day {day['t']}: DC inventory {day['dc_inventory']} exceeds "
                f"capacity {network.dc_capacity}")
        if day["shipped"] > network.truck_capacity:
            failures.append(
                f"day {day['t']}: shipped {day['shipped']} exceeds truck "
                f"capacity {network.truck_capacity}")
    shelf_total = sum(i.shelf_capacity for i in network.items)
    for day in result.daily:
        if day["store_inventory"] > shelf_total:
            failures.append(
                f"day {day['t']}: store inventory {day['store_inventory']} "
                f"exceeds total shelf {shelf_total}")
    return failures


def check_demand_integerisation() -> list[str]:
    """Rounding a fractional demand path must preserve its total to within a unit
    - the property that stops sub-1/day items being rounded out of existence."""
    failures = []
    rng = np.random.default_rng(42)
    for scale in (0.1, 0.4, 0.9, 3.0, 20.0):
        path = rng.gamma(2.0, scale / 2.0, size=28)
        rounded = integerise_demand(path)
        if abs(rounded.sum() - path.sum()) > 1.0:
            failures.append(
                f"integerise_demand lost {path.sum() - rounded.sum():.2f} units "
                f"at scale {scale} (total {path.sum():.2f} -> {rounded.sum()})")
        if (rounded < 0).any():
            failures.append(f"integerise_demand produced negative demand at {scale}")
    return failures


def main() -> int:
    print("Building verification instance...")
    network, point, actual, horizon = _instance()
    print(f"  {len(network.items)} items, {horizon} days, store {network.store_id}")

    checks = [
        ("demand integerisation preserves totals", lambda: check_demand_integerisation()),
        ("simulator conserves units", lambda: check_simulator_conserves_units(
            network, actual, horizon)),
        ("capacities respected after repair", lambda: check_shelf_and_capacity_respected(
            network, actual, horizon)),
        ("solver and simulator agree on cost", lambda: check_cost_models_agree(
            network, point, horizon)),
    ]

    total = 0
    for name, check in checks:
        failures = check()
        total += len(failures)
        print(f"  [{'FAIL' if failures else ' ok '}] {name}")
        for failure in failures:
            print(f"         - {failure}")

    print(f"\n{'FAILED' if total else 'PASSED'}: {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
