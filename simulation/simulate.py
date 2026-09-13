"""
The simulator every policy is scored in.

One day at a time: receive what arrived, let the policy decide, move stock,
observe *actual* demand, charge costs. The same loop runs the CP-SAT policy and
the classical baselines, and the cost accounting is deliberately identical to
the CP-SAT objective in optimisation/cpsat.py - if the two disagreed about what
a dollar of holding cost is, the benchmark would compare nothing.

**Why not SimPy**, which the roadmap originally named: SimPy earns its keep when
events arrive asynchronously and entities queue for resources. Inventory review
here is periodic and daily, every quantity is known at the start of the day, and
there is no contention to resolve - so a SimPy version would wrap this same
arithmetic in a process-and-event API, add a dependency to the deployed image,
and make the cost accounting harder to audit line by line. The loop below is the
whole model, and it is 60 lines.

**Feasibility repair.** A classical policy does not know about shared truck or
DC capacity, so it will sometimes ask for more than the network can carry. The
simulator clips the request rather than rejecting it, and counts every clip.
Those counts matter: they are the mechanism behind the CP-SAT policy's advantage,
so they belong in the output rather than hidden inside it.
"""
from dataclasses import dataclass, field

import numpy as np

from optimisation.network import Item, Network


@dataclass
class SimState:
    """Live inventory position, handed to the policy each day."""
    t: int
    inv_dc: dict[str, int]
    inv_store: dict[str, int]
    inbound_dc: dict[str, np.ndarray]     # arrivals by absolute day index
    inbound_store: dict[str, np.ndarray]
    horizon: int

    def dc_on_hand_total(self) -> int:
        return sum(self.inv_dc.values())

    def in_transit_to_store(self, item_id: str) -> int:
        """Units already dispatched to the store and not yet on the shelf."""
        return int(self.inbound_store[item_id][self.t:].sum())

    def in_transit_to_dc(self, item_id: str) -> int:
        return int(self.inbound_dc[item_id][self.t:].sum())

    def inventory_position_store(self, item_id: str) -> int:
        """On hand plus on order - what a reorder-point rule compares against s."""
        return self.inv_store[item_id] + self.in_transit_to_store(item_id)

    def inventory_position_dc(self, item_id: str) -> int:
        return self.inv_dc[item_id] + self.in_transit_to_dc(item_id)


@dataclass
class SimResult:
    policy: str
    total_cost: float
    costs: dict[str, float]
    fill_rate: float
    units_sold: int
    units_short: int
    units_demanded: int
    per_item_fill: dict[str, float]
    delivery_days: int
    po_lines: int
    mean_dc_inventory: float
    mean_store_inventory: float
    dc_utilisation_peak: float
    truck_utilisation_mean: float
    clips: dict[str, int] = field(default_factory=dict)
    daily: list[dict] = field(default_factory=list)
    solver: dict = field(default_factory=dict)

    def headline(self) -> dict:
        return {
            "policy": self.policy,
            "total_cost": round(self.total_cost, 2),
            "fill_rate": round(self.fill_rate, 4),
            "holding": round(self.costs["holding"], 2),
            "ordering": round(self.costs["ordering"], 2),
            "delivery": round(self.costs["delivery"], 2),
            "freight": round(self.costs["freight"], 2),
            "stockout": round(self.costs["stockout"], 2),
            "terminal": round(self.costs["terminal"], 2),
            "delivery_days": self.delivery_days,
            "po_lines": self.po_lines,
            "mean_store_inventory": round(self.mean_store_inventory, 1),
            "units_short": self.units_short,
        }


def _largest_remainder_scale(requested: dict[str, int], cap: int) -> dict[str, int]:
    """Scale a set of integer requests down to fit `cap`, keeping the total exact.

    Proportional scaling then rounding loses or invents units; allocating the
    rounding residue to the largest fractional parts does not.
    """
    total = sum(requested.values())
    if total <= cap or total == 0:
        return dict(requested)
    exact = {k: v * cap / total for k, v in requested.items()}
    floors = {k: int(np.floor(v)) for k, v in exact.items()}
    residue = cap - sum(floors.values())
    order = sorted(exact, key=lambda k: exact[k] - floors[k], reverse=True)
    for k in order[:residue]:
        floors[k] += 1
    return floors


def simulate(
    network: Network,
    policy,
    actual_demand: dict[str, np.ndarray],
    initial_inv_dc: dict[str, int] | None = None,
    initial_inv_store: dict[str, int] | None = None,
    keep_daily: bool = False,
) -> SimResult:
    """Run `policy` against `actual_demand` and return its realised cost.

    `policy` needs `.name`, `.reset(network, horizon)` and
    `.decide(state) -> (orders, shipments)`, both keyed by item id in units.
    """
    items: list[Item] = network.items
    horizon = len(next(iter(actual_demand.values())))
    l_sup, l_st = network.supplier_lead_days, network.store_lead_days

    # Pipelines are sized past the horizon so a late order can be placed and
    # charged for without indexing off the end; arrivals beyond the horizon
    # simply never land, which is the correct economics.
    span = horizon + l_sup + l_st + 1
    inbound_dc = {i.item_id: np.zeros(span, dtype=int) for i in items}
    inbound_store = {i.item_id: np.zeros(span, dtype=int) for i in items}

    inv_dc = dict(initial_inv_dc) if initial_inv_dc else {
        i.item_id: int(np.ceil(7 * i.mean_daily)) for i in items}
    inv_store = dict(initial_inv_store) if initial_inv_store else {
        i.item_id: int(min(i.shelf_capacity, np.ceil(7 * i.mean_daily))) for i in items}

    costs = {"holding": 0.0, "ordering": 0.0, "delivery": 0.0,
             "freight": 0.0, "stockout": 0.0, "terminal": 0.0}
    clips = {"truck_capacity": 0, "dc_capacity": 0, "shelf_space": 0, "dc_stock": 0}
    sold_total = {i.item_id: 0 for i in items}
    short_total = {i.item_id: 0 for i in items}
    dc_levels, store_levels, truck_loads = [], [], []
    delivery_days = 0
    po_lines = 0
    daily: list[dict] = []

    policy.reset(network, horizon)

    for t in range(horizon):
        # --- 1. Receive. Arrivals are already netted against capacity at the
        # time they were dispatched, so nothing can be turned away here.
        #
        # The slot is cleared as it is consumed. Leaving it filled would make
        # today's arrival count once in on-hand and again in in-transit, which
        # inflates every inventory position by one day's inbound and quietly
        # suppresses reordering - a slow leak that shows up as unexplained
        # stockouts rather than as an error.
        for item in items:
            i = item.item_id
            inv_dc[i] += int(inbound_dc[i][t])
            inv_store[i] += int(inbound_store[i][t])
            inbound_dc[i][t] = 0
            inbound_store[i][t] = 0

        state = SimState(
            t=t, inv_dc=inv_dc, inv_store=inv_store,
            inbound_dc=inbound_dc, inbound_store=inbound_store, horizon=horizon,
        )

        # --- 2. Decide.
        orders, shipments = policy.decide(state)
        orders = {i.item_id: max(0, int(orders.get(i.item_id, 0))) for i in items}
        shipments = {i.item_id: max(0, int(shipments.get(i.item_id, 0))) for i in items}

        # --- 3. Repair shipments: DC stock, then shelf space, then the truck.
        for item in items:
            i = item.item_id
            if shipments[i] > inv_dc[i]:
                clips["dc_stock"] += 1
                shipments[i] = inv_dc[i]
            shelf_room = item.shelf_capacity - state.inventory_position_store(i)
            if shipments[i] > max(0, shelf_room):
                clips["shelf_space"] += 1
                shipments[i] = max(0, shelf_room)
        if sum(shipments.values()) > network.truck_capacity:
            clips["truck_capacity"] += 1
            shipments = _largest_remainder_scale(shipments, network.truck_capacity)

        # --- 4. Repair orders against DC space, counting stock already inbound.
        committed = sum(inv_dc.values()) + sum(
            int(inbound_dc[i.item_id][t + 1:].sum()) for i in items)
        space = network.dc_capacity - committed
        if sum(orders.values()) > max(0, space):
            clips["dc_capacity"] += 1
            orders = _largest_remainder_scale(orders, max(0, space))
        # Case-pack integrality is the supplier's rule, so re-impose it after any
        # clipping - a repaired order is still a real order.
        for item in items:
            i = item.item_id
            if orders[i] > 0:
                orders[i] = (orders[i] // item.case_pack) * item.case_pack

        # --- 5. Dispatch.
        shipped_today = sum(shipments.values())
        for item in items:
            i = item.item_id
            if shipments[i] > 0:
                inv_dc[i] -= shipments[i]
                if t + l_st < span:
                    inbound_store[i][t + l_st] += shipments[i]
            if orders[i] > 0:
                po_lines += 1
                costs["ordering"] += network.order_cost_per_line
                if t + l_sup < span:
                    inbound_dc[i][t + l_sup] += orders[i]
        if shipped_today > 0:
            delivery_days += 1
            costs["delivery"] += network.delivery_fixed_cost
            costs["freight"] += network.transport_cost_per_unit * shipped_today
        truck_loads.append(shipped_today)

        # --- 6. Demand arrives. This is the only place actuals enter, and no
        # policy has seen them.
        day_sold = day_short = 0
        for item in items:
            i = item.item_id
            d = int(actual_demand[i][t])
            sell = min(d, inv_store[i])
            inv_store[i] -= sell
            short = d - sell
            sold_total[i] += sell
            short_total[i] += short
            costs["stockout"] += item.stockout_cost * short
            day_sold += sell
            day_short += short

        # --- 7. Charge holding on what is left at close of day.
        for item in items:
            i = item.item_id
            costs["holding"] += (item.holding_cost_dc * inv_dc[i]
                                 + item.holding_cost_store * inv_store[i])
        dc_levels.append(sum(inv_dc.values()))
        store_levels.append(sum(inv_store.values()))

        if keep_daily:
            daily.append({
                "t": t,
                "sold": day_sold,
                "short": day_short,
                "shipped": shipped_today,
                "dc_inventory": sum(inv_dc.values()),
                "store_inventory": sum(inv_store.values()),
            })

    # --- Terminal shortfall, priced exactly as cpsat.py prices it, so a policy
    # cannot look cheap by ending the horizon on an empty shelf.
    for item in items:
        target = int(np.ceil(network.store_lead_days * item.mean_daily))
        costs["terminal"] += item.stockout_cost * max(0, target - inv_store[item.item_id])

    demanded = {i.item_id: int(actual_demand[i.item_id].sum()) for i in items}
    total_demand = sum(demanded.values())
    total_sold = sum(sold_total.values())

    return SimResult(
        policy=policy.name,
        total_cost=sum(costs.values()),
        costs=costs,
        fill_rate=(total_sold / total_demand) if total_demand else 1.0,
        units_sold=total_sold,
        units_short=sum(short_total.values()),
        units_demanded=total_demand,
        per_item_fill={
            i: (sold_total[i] / demanded[i]) if demanded[i] else 1.0 for i in demanded},
        delivery_days=delivery_days,
        po_lines=po_lines,
        mean_dc_inventory=float(np.mean(dc_levels)),
        mean_store_inventory=float(np.mean(store_levels)),
        dc_utilisation_peak=(max(dc_levels) / network.dc_capacity) if network.dc_capacity else 0.0,
        truck_utilisation_mean=(
            float(np.mean([load / network.truck_capacity for load in truck_loads]))
            if network.truck_capacity else 0.0),
        clips=clips,
        daily=daily,
        solver=getattr(policy, "solver_stats", lambda: {})(),
    )
