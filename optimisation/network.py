"""
The network and its cost model.

M5 gives store-level sales and real weekly sell prices, and nothing else - no
distribution centre, no lead times, no capacities. Those have to be supplied,
and the honest way to do it is to derive everything derivable from the real
prices and state the rest as named assumptions in one place that the README and
the solver both read from. Every number below is either measured from M5 or
carries a comment explaining where it comes from.

**Network.** supplier -> DC -> store, with a 7-day inbound lead time and a
2-day DC-to-store lead time. The deployed forecaster covers one store, so a demo
instance is a two-echelon, multi-item system; nothing in the model or the solver
assumes a single store, and extra stores drop in once 3A is retrained wider.

**Where the optimisation problem actually lives.** Per-unit holding cost for
grocery is tiny next to a stockout - a few tenths of a cent per unit-day against
a dollar or more of lost margin. Taken alone that would make "hold everything"
optimal and the whole exercise pointless. What makes the problem real is the
things a per-item reorder-point rule cannot see:

  - a fixed cost per purchase-order line, so replenishing an item has a setup
    price and you want to batch;
  - a fixed cost per store delivery day shared by every item, so the question
    is not how much of item X to ship but *which items ride the same truck*;
  - DC storage, shelf space and truck capacity, all shared and all finite.

Those couplings are why this is a constraint program and not 60 independent
EOQ calculations, and they are where the cost reduction in the benchmark comes
from.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- Economics. Retail-grocery conventions; stated, not measured. -----------

# Gross margin on grocery runs roughly 25-30%; 0.30 puts unit cost at 70% of the
# observed M5 sell price, which is what holding cost is charged against.
GROSS_MARGIN = 0.30

# Annual inventory carrying rate: cost of capital, storage, insurance, shrink.
# Textbook range is 20-30% of unit cost per year. Charged per unit per day.
ANNUAL_HOLDING_RATE = 0.25

# A stockout costs the lost gross margin plus goodwill. The multiplier is the
# assumption most worth arguing with, which is why `benchmark.py --sensitivity`
# exists to re-run the comparison at other values. Note that the sweep currently
# committed to models/ predates the tuning fix in DECISIONS.md entry 23, so its
# numbers are not trustworthy and it needs re-running alongside --repeat before
# any sensitivity claim is made.
STOCKOUT_MULTIPLIER = 2.0

# Fixed cost of raising one purchase-order line on the supplier: buyer time,
# receiving, put-away. Independent of quantity, which is what creates batching.
ORDER_COST_PER_LINE = 12.0

# Fixed cost of running a DC-to-store delivery on a given day, shared across
# every item on that truck. This is the joint-replenishment coupling.
DELIVERY_FIXED_COST = 60.0

# Variable handling/freight per unit moved DC -> store.
TRANSPORT_COST_PER_UNIT = 0.04

# --- Lead times (days) -----------------------------------------------------
SUPPLIER_LEAD_DAYS = 7
STORE_LEAD_DAYS = 2

# --- Capacities, expressed in days of aggregate mean demand so they scale with
#     whatever item subset is being solved. Tuned to bind sometimes: the
#     benchmark prints utilisation so you can check they are not decoration.
DC_CAPACITY_DAYS = 10.0
TRUCK_CAPACITY_DAYS = 2.5
SHELF_CAPACITY_DAYS = 7.0
SHELF_CAPACITY_FLOOR = 6  # units; a slow seller still gets a facing

# --- Case packs. Suppliers ship whole cases, so order quantities are multiples.
#     Keyed by category rather than randomised, so the parameter is reproducible
#     and explainable.
CASE_PACK_BY_CATEGORY = {"FOODS": 12, "HOUSEHOLD": 6, "HOBBIES": 4}
DEFAULT_CASE_PACK = 6


@dataclass
class Item:
    """One item's economics and physical constraints, all per-unit."""
    item_id: str
    price: float
    unit_cost: float
    holding_cost_dc: float      # $/unit/day at the DC
    holding_cost_store: float   # $/unit/day at the store
    stockout_cost: float        # $/unit short
    case_pack: int
    shelf_capacity: int
    mean_daily: float
    sigma_daily: float


@dataclass
class Network:
    """Everything the solver and the simulator need, resolved to numbers."""
    store_id: str
    items: list[Item]
    supplier_lead_days: int = SUPPLIER_LEAD_DAYS
    store_lead_days: int = STORE_LEAD_DAYS
    order_cost_per_line: float = ORDER_COST_PER_LINE
    delivery_fixed_cost: float = DELIVERY_FIXED_COST
    transport_cost_per_unit: float = TRANSPORT_COST_PER_UNIT
    dc_capacity: int = 0
    truck_capacity: int = 0
    assumptions: dict = field(default_factory=dict)

    @property
    def item_ids(self) -> list[str]:
        return [i.item_id for i in self.items]

    def by_id(self, item_id: str) -> Item:
        return next(i for i in self.items if i.item_id == item_id)

    def summary(self) -> dict:
        return {
            "store_id": self.store_id,
            "n_items": len(self.items),
            "echelons": ["supplier", "dc", "store"],
            "supplier_lead_days": self.supplier_lead_days,
            "store_lead_days": self.store_lead_days,
            "dc_capacity_units": self.dc_capacity,
            "truck_capacity_units_per_day": self.truck_capacity,
            "order_cost_per_line": self.order_cost_per_line,
            "delivery_fixed_cost": self.delivery_fixed_cost,
            "transport_cost_per_unit": self.transport_cost_per_unit,
            "mean_holding_cost_store_per_unit_day": round(
                float(np.mean([i.holding_cost_store for i in self.items])), 5),
            "mean_stockout_cost_per_unit": round(
                float(np.mean([i.stockout_cost for i in self.items])), 4),
            "mean_case_pack": round(float(np.mean([i.case_pack for i in self.items])), 2),
            "assumptions": self.assumptions,
        }


def _case_pack(item_id: str) -> int:
    for category, pack in CASE_PACK_BY_CATEGORY.items():
        if item_id.startswith(category):
            return pack
    return DEFAULT_CASE_PACK


def build_network(
    history: pd.DataFrame,
    store_id: str,
    stockout_multiplier: float = STOCKOUT_MULTIPLIER,
    holding_rate: float = ANNUAL_HOLDING_RATE,
    supplier_lead_days: int = SUPPLIER_LEAD_DAYS,
    store_lead_days: int = STORE_LEAD_DAYS,
    dc_capacity_days: float = DC_CAPACITY_DAYS,
    truck_capacity_days: float = TRUCK_CAPACITY_DAYS,
    shelf_capacity_days: float = SHELF_CAPACITY_DAYS,
    order_cost_per_line: float = ORDER_COST_PER_LINE,
    delivery_fixed_cost: float = DELIVERY_FIXED_COST,
    transport_cost_per_unit: float = TRANSPORT_COST_PER_UNIT,
) -> Network:
    """Resolve per-item economics from real prices and recent demand statistics.

    `history` needs one row per item with `sell_price`, `mean_daily`,
    `sigma_daily` - what evalset.build() writes. Prices and demand levels are
    measured; the rates applied to them are the assumptions named above.

    Every structural parameter is an argument rather than a constant read at use
    site, because Stage 3C's what-if questions ("lead time doubles", "we lose a
    third of the truck") are exactly these numbers changing. The module-level
    constants remain the defaults, so an unparameterised call is unchanged.
    """
    required = {"item_id", "sell_price", "mean_daily", "sigma_daily"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")
    if history["sell_price"].isna().any():
        raise ValueError("history has items with no sell price; cannot cost them")

    items = []
    for row in history.itertuples():
        unit_cost = float(row.sell_price) * (1.0 - GROSS_MARGIN)
        holding_store = unit_cost * holding_rate / 365.0
        items.append(Item(
            item_id=str(row.item_id),
            price=float(row.sell_price),
            unit_cost=unit_cost,
            holding_cost_store=holding_store,
            # Holding at the DC is cheaper per unit than retail shelf space:
            # racked pallets in a warehouse, not prime square footage.
            holding_cost_dc=holding_store * 0.6,
            stockout_cost=float(row.sell_price) * GROSS_MARGIN * stockout_multiplier,
            case_pack=_case_pack(str(row.item_id)),
            shelf_capacity=max(
                SHELF_CAPACITY_FLOOR,
                int(np.ceil(shelf_capacity_days * float(row.mean_daily))),
            ),
            mean_daily=float(row.mean_daily),
            sigma_daily=float(row.sigma_daily),
        ))

    total_mean = sum(i.mean_daily for i in items)
    return Network(
        store_id=store_id,
        items=items,
        supplier_lead_days=int(supplier_lead_days),
        store_lead_days=int(store_lead_days),
        order_cost_per_line=float(order_cost_per_line),
        delivery_fixed_cost=float(delivery_fixed_cost),
        transport_cost_per_unit=float(transport_cost_per_unit),
        # Capacities are sized from *measured* mean demand, so a scenario that
        # raises demand does not silently grow the warehouse and the truck along
        # with it - the squeeze is the whole point of asking the question.
        dc_capacity=int(np.ceil(dc_capacity_days * total_mean)),
        truck_capacity=int(np.ceil(truck_capacity_days * total_mean)),
        assumptions={
            "gross_margin": GROSS_MARGIN,
            "annual_holding_rate": holding_rate,
            "stockout_multiplier": stockout_multiplier,
            "dc_capacity_days": dc_capacity_days,
            "truck_capacity_days": truck_capacity_days,
            "shelf_capacity_days": shelf_capacity_days,
            "case_pack_by_category": CASE_PACK_BY_CATEGORY,
            "prices": "measured from M5 sell_prices.csv over the evaluation window",
            "demand_stats": "measured over the 56 days before the window opens",
        },
    )


def select_items(history: pd.DataFrame, n_items: int) -> pd.DataFrame:
    """The `n_items` largest items by revenue over the history window.

    Revenue rather than units: the optimiser's job is to reduce cost, and cost
    concentrates where the money moves. Taking items at random would fill the
    instance with near-zero sellers whose optimal policy is "order a case when
    you run out", which is not a problem worth a constraint solver.
    """
    ranked = history.assign(revenue=history["mean_daily"] * history["sell_price"])
    ranked = ranked.sort_values("revenue", ascending=False)
    chosen = ranked.head(n_items).copy().reset_index(drop=True)
    chosen.attrs["revenue_share"] = float(
        chosen["revenue"].sum() / ranked["revenue"].sum())
    return chosen
