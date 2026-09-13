"""
The multi-echelon replenishment model, in OR-Tools CP-SAT.

One solve answers: over the next T days, how many units of each item to order
from the supplier into the DC, and how many to ship DC -> store, so that total
cost is minimised and the service-level target is met.

    decision variables   order[i,t]  units bought supplier -> DC  (case multiples)
                         ship[i,t]   units moved DC -> store
                         idc[i,t]    end-of-day DC inventory
                         ist[i,t]    end-of-day store inventory
                         sold[i,t]   units actually sold

    objective            holding (both echelons) + fixed cost per PO line
                         + fixed cost per delivery day + per-unit freight
                         + stockout penalty + end-of-horizon shortfall

Three things in here are worth reading closely, because they are where a model
like this usually goes quietly wrong.

**Integer costs.** CP-SAT optimises an integer objective. Grocery holding cost is
fractions of a cent per unit-day, so rounding the objective to cents would round
the entire holding term to zero and the solver would happily hoard stock. Costs
are therefore scaled by 10,000 - hundredths of a cent - which keeps every term
distinguishable inside int64.

**Integerised demand.** Forecasts are fractional and many M5 items sell under a
unit a day, so rounding each day independently would erase a 0.4/day item's
demand entirely. Demand is instead rounded cumulatively and differenced, which
preserves the horizon total to within one unit and places the fractional demand
on individual days.

**End-of-horizon effects.** A finite horizon tempts the optimiser to run stock
to zero on the last day and to stop ordering once the supplier lead time can no
longer deliver inside the window - neither of which is a real decision, just an
artifact of where the model stops. Ending below a lead-time-sized buffer is
therefore penalised at the item's own stockout cost. It is a soft penalty rather
than a constraint on purpose: as a hard constraint, a short final re-solve with
an empty DC would be infeasible through no fault of the policy.
"""
import os
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from .network import Network

# Costs are carried as integers in hundredths of a cent. See module docstring.
SCALE = 10_000

# Search threads. 8 suits a laptop; a free-tier host gets a fraction of one CPU,
# where 8 threads only contend with each other and the web process for it.
DEFAULT_WORKERS = int(os.getenv("SOLVER_WORKERS", "8"))


def integerise_demand(forecast: np.ndarray) -> np.ndarray:
    """Round a fractional demand path to integers, preserving its total.

    Rounds the cumulative sum and differences it, so a 0.4/day item gets demand
    on roughly every third day instead of never.
    """
    cumulative = np.rint(np.cumsum(np.asarray(forecast, dtype=float))).astype(int)
    return np.diff(np.concatenate(([0], cumulative))).clip(min=0)


@dataclass
class State:
    """Inventory and in-transit position at the moment a solve starts.

    `inbound_dc` / `inbound_store` are arrivals that were committed by earlier
    decisions and land on day t of *this* horizon. Carrying them explicitly is
    what makes rolling-horizon re-solving correct - without them each re-solve
    would forget the pipeline it had already paid for.
    """
    inv_dc: dict[str, int]
    inv_store: dict[str, int]
    inbound_dc: dict[str, np.ndarray]
    inbound_store: dict[str, np.ndarray]

    @classmethod
    def empty(cls, network: Network, horizon: int) -> "State":
        return cls(
            inv_dc={i.item_id: 0 for i in network.items},
            inv_store={i.item_id: 0 for i in network.items},
            inbound_dc={i.item_id: np.zeros(horizon, dtype=int) for i in network.items},
            inbound_store={i.item_id: np.zeros(horizon, dtype=int) for i in network.items},
        )

    @classmethod
    def initial(cls, network: Network, horizon: int, days_of_cover: float = 7.0) -> "State":
        """A neutral warm start: every policy begins with the same stock.

        Starting empty would score the first lead time of every policy as one
        long stockout and drown the differences between them in a shared
        artifact. Cover is sized in days of mean demand, capped by shelf space.
        """
        state = cls.empty(network, horizon)
        for item in network.items:
            state.inv_store[item.item_id] = int(min(
                item.shelf_capacity,
                np.ceil(days_of_cover * item.mean_daily),
            ))
            state.inv_dc[item.item_id] = int(np.ceil(days_of_cover * item.mean_daily))
        return state


@dataclass
class Plan:
    """What one solve decided, plus how the solve itself went."""
    orders: dict[str, np.ndarray]
    shipments: dict[str, np.ndarray]
    status: str
    objective: float
    best_bound: float
    solve_time_ms: int
    horizon: int
    relaxations: list[str] = field(default_factory=list)
    planned_costs: dict[str, float] = field(default_factory=dict)

    @property
    def gap_pct(self) -> float | None:
        """How far from proven optimal. None when no bound is available."""
        if self.objective in (0.0, None) or self.best_bound is None:
            return None
        if self.objective == 0:
            return 0.0
        return round(abs(self.objective - self.best_bound) / abs(self.objective) * 100, 3)


# How much cover a single purchase order is ever allowed to buy. Bounding this
# by the horizon total instead leaves the solver free to consider orders it would
# never place, which costs search time and nothing else.
MAX_ORDER_COVER_DAYS = 21


def _upper_bounds(network: Network, demand: dict[str, np.ndarray], horizon: int
                  ) -> tuple[dict[str, int], dict[str, int]]:
    """Per-item caps on a single order and a single shipment.

    Tight bounds are most of CP-SAT's performance on a model this shape, so both
    are derived from what an item could physically absorb rather than from the
    horizon total. A shipment in particular can never usefully exceed the shelf
    plus what sells while it is in transit - the shelf cap would reject the rest
    on arrival - and that bound is many times tighter than the horizon's demand.
    """
    order_ub, ship_ub = {}, {}
    l_st = network.store_lead_days
    for item in network.items:
        i = item.item_id
        total = int(demand[i].sum())
        daily = total / max(horizon, 1)

        cover = MAX_ORDER_COVER_DAYS * daily + 3 * item.sigma_daily * np.sqrt(
            MAX_ORDER_COVER_DAYS)
        order_ub[i] = int(min(network.dc_capacity,
                              max(item.case_pack, np.ceil(cover))))

        transit_demand = int(np.ceil(daily * l_st + 3 * item.sigma_daily * np.sqrt(l_st + 1)))
        ship_ub[i] = int(min(network.truck_capacity,
                             max(1, item.shelf_capacity + transit_demand)))
    return order_ub, ship_ub


def solve(
    network: Network,
    demand: dict[str, np.ndarray],
    state: State | None = None,
    service_level: float = 0.95,
    per_item_service_floor: float | None = 0.80,
    time_limit_s: float = 30.0,
    workers: int = DEFAULT_WORKERS,
    terminal_cover_days: int | None = None,
    hint_period: int | None = 7,
) -> Plan:
    """Solve one replenishment horizon.

    `demand` is integer units per item per day - forecasts when planning, actuals
    when computing the perfect-foresight bound. `service_level` is a hard
    aggregate fill-rate floor; `per_item_service_floor` stops the solver from
    hitting that aggregate by abandoning awkward items wholesale, and is dropped
    (and reported) if it makes the instance infeasible.
    """
    items = network.items
    horizon = len(next(iter(demand.values())))
    if any(len(v) != horizon for v in demand.values()):
        raise ValueError("demand paths have inconsistent lengths")
    state = state or State.initial(network, horizon)
    l_sup, l_st = network.supplier_lead_days, network.store_lead_days
    terminal_cover_days = l_st if terminal_cover_days is None else terminal_cover_days

    order_ub, ship_ub = _upper_bounds(network, demand, horizon)
    relaxations: list[str] = []

    def build(tier: int) -> tuple[cp_model.CpModel, dict]:
        m = cp_model.CpModel()
        v: dict = {"order": {}, "ship": {}, "idc": {}, "ist": {}, "sold": {},
                   "z_order": {}, "cases": {}, "term_short": {}}
        deliver = [m.NewBoolVar(f"deliver_{t}") for t in range(horizon)]

        for item in items:
            i = item.item_id
            d = demand[i]
            for t in range(horizon):
                max_cases = max(1, order_ub[i] // item.case_pack)
                v["cases"][i, t] = m.NewIntVar(0, max_cases, f"cases_{i}_{t}")
                v["order"][i, t] = m.NewIntVar(0, max_cases * item.case_pack, f"ord_{i}_{t}")
                v["z_order"][i, t] = m.NewBoolVar(f"zord_{i}_{t}")
                # Suppliers ship whole cases, and a PO line is either raised or
                # not - that boolean is what the fixed ordering cost attaches to.
                m.Add(v["order"][i, t] == item.case_pack * v["cases"][i, t])
                m.Add(v["cases"][i, t] >= 1).OnlyEnforceIf(v["z_order"][i, t])
                m.Add(v["cases"][i, t] == 0).OnlyEnforceIf(v["z_order"][i, t].Not())

                v["ship"][i, t] = m.NewIntVar(0, ship_ub[i], f"ship_{i}_{t}")
                v["idc"][i, t] = m.NewIntVar(0, network.dc_capacity, f"idc_{i}_{t}")
                # On-hand shelf stock capped in the variable's own domain. This
                # is the weaker half of the shelf rule - the constraint further
                # down also caps stock already in transit towards the shelf.
                v["ist"][i, t] = m.NewIntVar(0, item.shelf_capacity, f"ist_{i}_{t}")
                v["sold"][i, t] = m.NewIntVar(0, int(d[t]), f"sold_{i}_{t}")

                # --- DC balance. Orders placed at t land at t + l_sup, so what
                # arrives today was decided l_sup days ago; before that, it comes
                # from the pipeline this solve inherited.
                arriving_dc = state.inbound_dc[i][t]
                prev_dc = state.inv_dc[i] if t == 0 else v["idc"][i, t - 1]
                inflow = [prev_dc, arriving_dc]
                if t - l_sup >= 0:
                    inflow.append(v["order"][i, t - l_sup])
                m.Add(v["idc"][i, t] == sum(inflow) - v["ship"][i, t])

                # --- Store balance. Non-negativity of ist is what enforces
                # "you cannot sell stock you do not have"; no separate
                # availability constraint is needed.
                arriving_st = state.inbound_store[i][t]
                prev_st = state.inv_store[i] if t == 0 else v["ist"][i, t - 1]
                inflow_st = [prev_st, arriving_st]
                if t - l_st >= 0:
                    inflow_st.append(v["ship"][i, t - l_st])
                m.Add(v["ist"][i, t] == sum(inflow_st) - v["sold"][i, t])

                # --- Shelf space binds on the inventory *position* - what is on
                # the shelf plus what is already rolling towards it - not just on
                # what is on hand tonight.
                #
                # This is the subtler of the two readings, and it has to match
                # the simulator exactly. The simulator decides shipments before
                # it knows the day's sales, so it can only hold back stock on
                # position; if the solver were allowed to plan against
                # end-of-day on-hand instead, it would legitimately ship
                # quantities that the simulator would then clip on arrival, and
                # its own plans would be quietly degraded on replay. Measured
                # before this constraint existed: the perfect-information solve
                # came back with 74% fill against actual demand it knew in full.
                in_transit = [v["ship"][i, s]
                              for s in range(max(0, t - l_st + 1), t + 1)]
                inherited = int(state.inbound_store[i][t + 1:].sum())
                m.Add(v["ist"][i, t] + sum(in_transit) + inherited
                      <= item.shelf_capacity)

            # --- End-of-horizon buffer, penalised not mandated.
            target = int(np.ceil(terminal_cover_days * item.mean_daily))
            v["term_short"][i] = m.NewIntVar(0, max(target, 1), f"term_{i}")
            m.Add(v["term_short"][i] >= target - v["ist"][i, horizon - 1])

        # --- Shared capacity. These are the constraints a per-item reorder
        # rule structurally cannot respect, and the reason for a joint solve.
        # The aggregate truck constraint also forces every shipment to zero on a
        # non-delivery day, so no separate per-item big-M is needed - and leaving
        # one in would only hand the solver a weaker linear relaxation.
        for t in range(horizon):
            m.Add(sum(v["idc"][i.item_id, t] for i in items) <= network.dc_capacity)
            m.Add(sum(v["ship"][i.item_id, t] for i in items)
                  <= network.truck_capacity * deliver[t])

        # --- Service level. Hard at tiers 0 and 1, absent at tier 2; see the
        # relaxation ladder below for why it has to be droppable at all.
        total_demand = int(sum(int(demand[i.item_id].sum()) for i in items))
        if tier <= 1 and total_demand > 0:
            m.Add(sum(v["sold"][i.item_id, t] for i in items for t in range(horizon))
                  >= int(np.ceil(service_level * total_demand)))

            # --- Valid inequality: a floor on delivery days. Everything sold
            # from the shelf beyond what is already there or inbound has to
            # arrive on some truck, and a truck carries at most truck_capacity.
            # Redundant by construction, but it gives the linear relaxation a
            # real bound on the fixed delivery cost, which is otherwise the term
            # the search flounders on. Implied by the service constraint, so it
            # is only sound while that constraint is present.
            if network.truck_capacity > 0:
                on_hand_or_inbound = sum(
                    state.inv_store[i.item_id] for i in items) + sum(
                    int(state.inbound_store[i.item_id].sum()) for i in items)
                must_arrive = int(np.ceil(service_level * total_demand)) - on_hand_or_inbound
                if must_arrive > 0:
                    m.Add(sum(deliver)
                          >= int(np.ceil(must_arrive / network.truck_capacity)))

        # --- Per-item floor, so the aggregate is not met by writing items off.
        if tier == 0 and per_item_service_floor:
            for item in items:
                item_demand = int(demand[item.item_id].sum())
                if item_demand > 0:
                    m.Add(sum(v["sold"][item.item_id, t] for t in range(horizon))
                          >= int(np.floor(per_item_service_floor * item_demand)))

        terms = []
        for item in items:
            i = item.item_id
            h_dc = round(item.holding_cost_dc * SCALE)
            h_st = round(item.holding_cost_store * SCALE)
            pen = round(item.stockout_cost * SCALE)
            freight = round(network.transport_cost_per_unit * SCALE)
            setup = round(network.order_cost_per_line * SCALE)
            for t in range(horizon):
                terms.append(h_dc * v["idc"][i, t])
                terms.append(h_st * v["ist"][i, t])
                terms.append(setup * v["z_order"][i, t])
                terms.append(freight * v["ship"][i, t])
                # Unmet demand is demand minus sales; no separate variable.
                terms.append(pen * (int(demand[i][t]) - v["sold"][i, t]))
            terms.append(pen * v["term_short"][i])
        terms += [round(network.delivery_fixed_cost * SCALE) * deliver[t]
                  for t in range(horizon)]
        m.Minimize(sum(terms))

        # --- Search hint: deliver and reorder on a weekly cadence.
        #
        # This is a capacitated joint-replenishment problem, which is NP-hard and
        # whose difficulty lives almost entirely in the binaries - *which* days a
        # truck runs and *which* items are on it. Fix those and what remains is
        # an easy flow problem, so handing the solver a sane cadence to start
        # from beats making it discover that delivering daily is expensive.
        #
        # A hint is only advice: CP-SAT may ignore or improve on it and the
        # objective is untouched, so this biases search effort, not the answer.
        # The weekly period is a starting point, not an imposed cadence, and
        # solutions that beat it are accepted - which is what happens.
        #
        # Honest accounting of what this is worth: at the 14-day planning horizon
        # the policy actually uses, solves reach proven optimality in seconds. At
        # 28 days it made no reliable difference to the incumbent, and the two
        # runs that suggested otherwise were measured at different time limits,
        # so no improvement is claimed there.
        if hint_period:
            for t in range(horizon):
                on = 1 if t % hint_period == 0 else 0
                m.AddHint(deliver[t], on)
                for item in items:
                    m.AddHint(v["z_order"][item.item_id, t], on)

        v["deliver"] = deliver
        return m, v

    def run(m: cp_model.CpModel) -> tuple[cp_model.CpSolver, int]:
        solver = cp_model.CpSolver()
        # A time limit is not optional: this runs inside a web worker, and an
        # unbounded solve on a bad input would hang the job forever.
        solver.parameters.max_time_in_seconds = float(time_limit_s)
        solver.parameters.num_workers = int(workers)
        return solver, solver.Solve(m)

    # --- The relaxation ladder.
    #
    # A hard service-level floor is not always satisfiable, and that is a
    # property of the situation rather than a bug. Re-solving mid-window, the
    # first two days of the horizon can only be served from stock already on the
    # shelf, because a shipment takes `store_lead_days` to arrive - so if the
    # shelf is low, no decision available to the solver can hit 95% over the
    # window, and the model is correctly infeasible. Returning an empty plan
    # there is the worst possible answer: measured before this ladder existed,
    # three of four re-solves came back infeasible, the policy ordered nothing
    # for 21 days, and it finished at 34% fill against baselines near 88%.
    #
    # So the constraints come off in a defined order, worst-to-keep last, and
    # every relaxation is recorded on the Plan and surfaced through the API.
    # Stage 3C needs exactly this contract when an agent proposes an impossible
    # scenario: explain what had to give, do not crash and do not silently
    # pretend the original question was answered.
    # Each rung names what it gives up *relative to the rung before*, and the
    # note is recorded on entering the rung rather than on failing it. That
    # ordering matters: recorded on failure instead, a rung that succeeded would
    # report nothing, and a plan built with the service constraint dropped would
    # come back looking like a plan that had met it.
    ladder: list[tuple[int, str | None]] = [(0, None)]
    if per_item_service_floor:
        ladder.append((1, f"per-item service floor of "
                          f"{per_item_service_floor:.0%} dropped"))
    ladder.append((2, f"aggregate service level of {service_level:.0%} dropped; "
                      f"stockout penalties alone now drive availability"))

    solver = None
    status = cp_model.UNKNOWN
    for tier, gave_up in ladder:
        if gave_up is not None:
            # This rung is only reached because the previous one failed with the
            # constraint in place.
            relaxations.append(
                f"{gave_up}: instance was {solver.StatusName(status)} with it")
        model, v = build(tier)
        solver, status = run(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Plan(
            orders={i.item_id: np.zeros(horizon, dtype=int) for i in items},
            shipments={i.item_id: np.zeros(horizon, dtype=int) for i in items},
            status=solver.StatusName(status),
            objective=float("inf"),
            best_bound=float("nan"),
            solve_time_ms=int(solver.WallTime() * 1000),
            horizon=horizon,
            relaxations=relaxations + ["no feasible plan found"],
        )

    orders = {i.item_id: np.array([solver.Value(v["order"][i.item_id, t])
                                   for t in range(horizon)], dtype=int) for i in items}
    shipments = {i.item_id: np.array([solver.Value(v["ship"][i.item_id, t])
                                      for t in range(horizon)], dtype=int) for i in items}

    # The plan's own cost breakdown, under the demand it was planned against.
    # Kept separate from the simulator's numbers: comparing the two is how you
    # see what the forecast error cost.
    planned = {"holding": 0.0, "ordering": 0.0, "delivery": 0.0,
               "freight": 0.0, "stockout": 0.0, "terminal": 0.0}
    for item in items:
        i = item.item_id
        for t in range(horizon):
            planned["holding"] += (
                item.holding_cost_dc * solver.Value(v["idc"][i, t])
                + item.holding_cost_store * solver.Value(v["ist"][i, t]))
            planned["ordering"] += network.order_cost_per_line * solver.Value(v["z_order"][i, t])
            planned["freight"] += network.transport_cost_per_unit * solver.Value(v["ship"][i, t])
            planned["stockout"] += item.stockout_cost * (
                int(demand[i][t]) - solver.Value(v["sold"][i, t]))
        planned["terminal"] += item.stockout_cost * solver.Value(v["term_short"][i])
    planned["delivery"] = network.delivery_fixed_cost * sum(
        solver.Value(d) for d in v["deliver"])
    planned = {k: round(val, 2) for k, val in planned.items()}
    planned["total"] = round(sum(planned.values()), 2)

    return Plan(
        orders=orders,
        shipments=shipments,
        status=solver.StatusName(status),
        objective=solver.ObjectiveValue() / SCALE,
        best_bound=solver.BestObjectiveBound() / SCALE,
        solve_time_ms=int(solver.WallTime() * 1000),
        horizon=horizon,
        relaxations=relaxations,
        planned_costs=planned,
    )
