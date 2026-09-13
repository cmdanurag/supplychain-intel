"""
The policies being compared.

Every one of them implements the same tiny protocol - `reset`, `decide` - so
simulation/simulate.py cannot treat any of them preferentially, and so the
headline comparison is between decision rules rather than between two different
pieces of accounting.

    ReorderPointEOQ     textbook (s, Q). No forecast. The policy the resume
                        number is quoted against.
    SafetyStockBaseStock  periodic order-up-to with z * sigma * sqrt(L). No
                        forecast. The better classical policy.
    ForecastBaseStock   same rule, but the lead-time demand estimate comes from
                        the LightGBM forecast and sigma from its prediction
                        interval. **This is the ablation that matters**: it
                        isolates the value of the forecast from the value of the
                        optimiser, so a cost win over it cannot be explained by
                        "you just had a better forecast".
    CpSatRollingHorizon our policy - re-solve the constraint program as actual
                        demand arrives, implement only the near-term decisions.
    PerfectForesight    one CP-SAT solve on the actual demand path. Not a
                        policy anyone could run; it is the lower bound that says
                        how much of the remaining gap is forecast error rather
                        than bad optimisation.

Each classical policy exposes a single scalar safety knob (`k` / `z`). The
benchmark bisects on it so every policy is compared at the *same achieved
service level* - without that, "cheaper" just means "held less stock and served
fewer customers".
"""
import numpy as np

from .cpsat import Plan, State, integerise_demand, solve
from .network import Network

# For a symmetric 80% interval, upper - lower spans 2 * z(0.9) * sigma.
Z80_WIDTH = 2 * 1.2815515655446004


class Policy:
    """Base class: names the policy and holds the network after reset.

    **Review periods, and why every classical policy here has one.** A delivery
    costs a fixed $60 whatever rides on it. A policy reviewed daily therefore
    pays 28 delivery charges over a 28-day window while the CP-SAT policy
    consolidates onto a handful of truck days - and almost the entire cost
    difference would be that cadence rather than any decision quality. Comparing
    against a daily-review baseline would be a strawman, and the resulting
    percentage would evaporate the moment anyone asked about it. So every
    classical policy takes a `review_days`, and the benchmark searches it and
    keeps each baseline's *best* setting.
    """
    name = "policy"
    review_days = 1

    def reset(self, network: Network, horizon: int) -> None:
        self.network = network
        self.horizon = horizon

    def _is_review_day(self, t: int) -> bool:
        return t % max(1, self.review_days) == 0

    def decide(self, state) -> tuple[dict[str, int], dict[str, int]]:
        raise NotImplementedError

    def solver_stats(self) -> dict:
        return {}

    @staticmethod
    def _to_cases(units: float, case_pack: int) -> int:
        """Round an order up to whole cases. Suppliers do not split cases."""
        if units <= 0:
            return 0
        return int(np.ceil(units / case_pack) * case_pack)


class ReorderPointEOQ(Policy):
    """Continuous-review (s, Q) at both echelons, parameters from history only.

    This is the policy the headline cost reduction is quoted against, because it
    is what a firm without a forecasting or optimisation function actually runs.

    Two caps are applied to the textbook quantities, and both are findings
    rather than fudges. Unconstrained EOQ here comes out at 50-plus days of
    demand, because grocery holding cost per unit-day is minute next to a $12
    purchase-order line - so Q is capped at the shelf and at a DC cover
    allowance. Leaving EOQ uncapped would not make the baseline stronger or
    more honest, only permanently clipped by the simulator's shelf repair.
    """
    name = "reorder_point_eoq"

    def __init__(self, k: float = 1.64, dc_cover_days: float = 14.0,
                 review_days: int = 1):
        self.k = k
        self.dc_cover_days = dc_cover_days
        self.review_days = review_days

    def reset(self, network: Network, horizon: int) -> None:
        super().reset(network, horizon)
        self.q_store, self.q_dc, self.s_store, self.s_dc = {}, {}, {}, {}
        l_st, l_sup = network.store_lead_days, network.supplier_lead_days
        review = max(1, self.review_days)
        for item in network.items:
            i = item.item_id
            mu, sigma = item.mean_daily, item.sigma_daily
            eoq = (np.sqrt(2 * network.order_cost_per_line * mu / item.holding_cost_store)
                   if mu > 0 and item.holding_cost_store > 0 else item.case_pack)
            self.q_store[i] = max(item.case_pack,
                                  min(self._to_cases(eoq, item.case_pack),
                                      item.shelf_capacity))
            self.q_dc[i] = max(item.case_pack,
                               min(self._to_cases(eoq, item.case_pack),
                                   self._to_cases(self.dc_cover_days * mu, item.case_pack)))
            # Reordering only on review days means the position must also cover
            # the wait until the next one, not just the lead time.
            cover_st, cover_dc = l_st + review - 1, l_sup + review - 1
            self.s_store[i] = mu * cover_st + self.k * sigma * np.sqrt(max(cover_st, 1))
            self.s_dc[i] = mu * cover_dc + self.k * sigma * np.sqrt(max(cover_dc, 1))

    def decide(self, state):
        orders, shipments = {}, {}
        if not self._is_review_day(state.t):
            return orders, shipments
        for item in self.network.items:
            i = item.item_id
            shipments[i] = (self.q_store[i]
                            if state.inventory_position_store(i) <= self.s_store[i] else 0)
            # The DC's own demand is the store's replenishment stream, so its
            # position is measured on DC stock plus what is already inbound.
            orders[i] = (self.q_dc[i]
                         if state.inventory_position_dc(i) <= self.s_dc[i] else 0)
        return orders, shipments


class SafetyStockBaseStock(Policy):
    """Periodic-review order-up-to, with safety stock z * sigma * sqrt(L+1).

    The stronger classical policy: reviewing daily and topping up to a target
    tracks demand better than a fixed Q, and this is the rule most textbooks
    would actually recommend. Still no forecast - mu and sigma are the trailing
    56-day statistics.
    """
    name = "safety_stock_base_stock"

    def __init__(self, z: float = 1.64, review_days: int = 1):
        self.z = z
        self.review_days = review_days

    def reset(self, network: Network, horizon: int) -> None:
        super().reset(network, horizon)
        self.s_store, self.s_dc = {}, {}
        l_st, l_sup = network.store_lead_days, network.supplier_lead_days
        review = max(1, self.review_days)
        for item in network.items:
            i = item.item_id
            mu, sigma = item.mean_daily, item.sigma_daily
            # Order-up-to has to span lead time plus review interval: the
            # classic (R, S) protection period.
            cover_st = l_st + review
            self.s_store[i] = mu * cover_st + self.z * sigma * np.sqrt(cover_st)
            # The DC must cover its own lead time *and* keep the store's target
            # topped up - the echelon-stock view of a two-level system.
            cover_dc = l_sup + review
            self.s_dc[i] = (mu * cover_dc + self.z * sigma * np.sqrt(cover_dc)
                            + self.s_store[i])

    def decide(self, state):
        orders, shipments = {}, {}
        if not self._is_review_day(state.t):
            return orders, shipments
        for item in self.network.items:
            i = item.item_id
            gap_store = self.s_store[i] - state.inventory_position_store(i)
            shipments[i] = int(np.ceil(gap_store)) if gap_store > 0 else 0
            gap_dc = self.s_dc[i] - (state.inventory_position_dc(i)
                                     + state.inventory_position_store(i))
            orders[i] = self._to_cases(gap_dc, item.case_pack) if gap_dc > 0 else 0
        return orders, shipments


class ForecastBaseStock(Policy):
    """Base-stock, but the lead-time demand estimate is the LightGBM forecast.

    The ablation. Identical decision rule to SafetyStockBaseStock; the only
    change is where mu and sigma come from - forward forecast and its prediction
    interval instead of trailing history. Whatever this policy beats the
    classical ones by is the forecast's contribution; whatever CP-SAT beats
    *this* by is the optimiser's.
    """
    name = "forecast_base_stock"

    def __init__(self, forecast: dict[str, np.ndarray],
                 sigma: dict[str, np.ndarray], z: float = 1.64,
                 review_days: int = 1):
        self.forecast = forecast
        self.sigma = sigma
        self.z = z
        self.review_days = review_days

    def _window(self, series: np.ndarray, start: int, days: int) -> np.ndarray:
        """Forecast over [start, start+days), padded at the end with its own
        tail mean so the last days of the horizon are not treated as zero demand.
        """
        end = start + days
        if end <= len(series):
            return series[start:end]
        tail = series[start:] if start < len(series) else series[-1:]
        pad = np.full(end - max(start, 0) - len(tail),
                      float(tail.mean()) if len(tail) else 0.0)
        return np.concatenate([tail, pad])

    def decide(self, state):
        l_st = self.network.store_lead_days
        l_sup = self.network.supplier_lead_days
        t = state.t
        orders, shipments = {}, {}
        if not self._is_review_day(t):
            return orders, shipments
        review = max(1, self.review_days)
        for item in self.network.items:
            i = item.item_id
            f = self.forecast[i]
            sig = self.sigma[i]

            cover_st = l_st + review
            mu_st = float(self._window(f, t, cover_st).sum())
            # Independent daily errors, so interval sigmas add in quadrature.
            sd_st = float(np.sqrt((self._window(sig, t, cover_st) ** 2).sum()))
            target_st = mu_st + self.z * sd_st
            gap_store = target_st - state.inventory_position_store(i)
            shipments[i] = int(np.ceil(gap_store)) if gap_store > 0 else 0

            cover_dc = l_sup + review
            mu_dc = float(self._window(f, t, cover_dc).sum())
            sd_dc = float(np.sqrt((self._window(sig, t, cover_dc) ** 2).sum()))
            target_dc = mu_dc + self.z * sd_dc + target_st
            gap_dc = target_dc - (state.inventory_position_dc(i)
                                  + state.inventory_position_store(i))
            orders[i] = self._to_cases(gap_dc, item.case_pack) if gap_dc > 0 else 0
        return orders, shipments


class CpSatRollingHorizon(Policy):
    """Ours: re-solve the constraint program as actual demand arrives.

    Model-predictive control. Every `resolve_every` days the solver is handed the
    true current inventory and pipeline position and the forecast for the rest of
    the window, and only the next `resolve_every` days of its plan are
    implemented before it is re-solved. An open-loop plan solved once on day zero
    would be strictly worse and also dishonest - it would never face its own
    forecast error, which is exactly what a real deployment cannot avoid.
    """
    name = "cpsat_rolling_horizon"

    def __init__(self, forecast: dict[str, np.ndarray], service_level: float = 0.95,
                 resolve_every: int = 7, planning_horizon: int = 14,
                 time_limit_s: float = 30.0,
                 per_item_service_floor: float | None = 0.80):
        self.forecast = forecast
        self.service_level = service_level
        self.resolve_every = resolve_every
        # Plan 14 days, commit 7. Solving the full remaining 28 days each time
        # is both far harder - this is a capacitated joint-replenishment problem,
        # and the measured optimality gap goes from single digits at 14 days to
        # 40%+ at 28 - and largely wasted, since everything past the next commit
        # window gets re-decided anyway. Planning past the commit horizon still
        # matters: it is what stops the solver emptying the shelf on day 7.
        self.planning_horizon = planning_horizon
        self.time_limit_s = time_limit_s
        self.per_item_service_floor = per_item_service_floor

    def reset(self, network: Network, horizon: int) -> None:
        super().reset(network, horizon)
        self.plan: Plan | None = None
        self.plan_origin = 0
        self.solves: list[Plan] = []

    def _resolve(self, state) -> None:
        t = state.t
        remaining = min(self.horizon - t, self.planning_horizon)
        demand = {i.item_id: integerise_demand(self.forecast[i.item_id][t:t + remaining])
                  for i in self.network.items}

        # Pipeline commitments, re-indexed relative to this solve's day zero.
        # Dropping them would make the solver re-buy stock already in transit.
        sub = State(
            inv_dc={i.item_id: int(state.inv_dc[i.item_id]) for i in self.network.items},
            inv_store={i.item_id: int(state.inv_store[i.item_id]) for i in self.network.items},
            inbound_dc={i.item_id: np.asarray(
                state.inbound_dc[i.item_id][t:t + remaining], dtype=int)
                for i in self.network.items},
            inbound_store={i.item_id: np.asarray(
                state.inbound_store[i.item_id][t:t + remaining], dtype=int)
                for i in self.network.items},
        )
        self.plan = solve(
            self.network, demand, sub,
            service_level=self.service_level,
            per_item_service_floor=self.per_item_service_floor,
            time_limit_s=self.time_limit_s,
        )
        self.plan_origin = t
        self.solves.append(self.plan)

    def decide(self, state):
        if self.plan is None or (state.t - self.plan_origin) >= self.resolve_every:
            self._resolve(state)

        offset = state.t - self.plan_origin
        plan = self.plan
        if plan is None or offset >= plan.horizon:
            return {}, {}
        orders = {i: int(v[offset]) for i, v in plan.orders.items()}
        shipments = {i: int(v[offset]) for i, v in plan.shipments.items()}
        return orders, shipments

    def solver_stats(self) -> dict:
        if not self.solves:
            return {}
        return {
            "solves": len(self.solves),
            "statuses": [p.status for p in self.solves],
            "total_solve_time_ms": sum(p.solve_time_ms for p in self.solves),
            "max_solve_time_ms": max(p.solve_time_ms for p in self.solves),
            "mean_gap_pct": round(float(np.mean(
                [p.gap_pct for p in self.solves if p.gap_pct is not None] or [0.0])), 3),
            "relaxations": [r for p in self.solves for r in p.relaxations],
        }


class PerfectInformation(CpSatRollingHorizon):
    """The same controller, fed actual demand instead of a forecast.

    Not runnable in reality - it knows the future. Its purpose is to scale the
    result: it says how much of the remaining cost is forecast error and how
    much the constraints make unavoidable. Without it a percentage has no
    reference - 8% is excellent if perfect information is only 10% better and
    mediocre if it is 40% better.

    Deliberately the *same* rolling-horizon controller rather than one
    all-knowing 28-day solve. A single full-horizon solve would be a tighter
    bound in principle, but this is a capacitated joint-replenishment problem
    and at 28 days the solver leaves a 40%+ optimality gap - so the "bound" would
    be a badly-solved instance, and any comparison against it would measure
    solver difficulty rather than the value of information. Holding the
    controller fixed and changing only the demand it is given is the comparison
    that isolates one variable.
    """
    name = "perfect_information"

    def __init__(self, actual: dict[str, np.ndarray], service_level: float = 0.95,
                 resolve_every: int = 7, planning_horizon: int = 14,
                 time_limit_s: float = 30.0,
                 per_item_service_floor: float | None = 0.80):
        super().__init__(
            forecast={i: np.asarray(v, dtype=float) for i, v in actual.items()},
            service_level=service_level, resolve_every=resolve_every,
            planning_horizon=planning_horizon, time_limit_s=time_limit_s,
            per_item_service_floor=per_item_service_floor,
        )


def interval_sigma(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Back out a daily sigma from the model's 80% prediction interval.

    The quantile models were measured at 80.3% coverage against 80% nominal on
    the backtest, so treating the interval as a calibrated normal spread is a
    fair reading of it rather than an assumption pulled from nowhere.
    """
    return np.clip(np.asarray(upper) - np.asarray(lower), 0, None) / Z80_WIDTH
