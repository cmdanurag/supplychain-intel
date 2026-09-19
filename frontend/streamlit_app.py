"""
Streamlit frontend. Demonstrates the client side of the polling pattern.

Run locally (with the API already running on :8000):
    export API_URL=http://localhost:8000
    streamlit run frontend/streamlit_app.py
"""
import os
import time

import requests
import streamlit as st

def _api_url() -> str:
    """Streamlit Community Cloud exposes secrets through st.secrets and does not
    put them in the environment, while local runs use an env var. Checking only
    one of the two silently falls back to localhost on the deployed app.
    """
    from_env = os.getenv("API_URL")
    if from_env:
        return from_env
    try:
        return st.secrets["API_URL"]
    except Exception:
        return "http://localhost:8000"


API_URL = _api_url()

st.set_page_config(page_title="Supply Chain Intelligence", layout="wide")
st.title("Supply Chain Intelligence Platform")
st.caption(
    "LightGBM forecasts, a CP-SAT multi-echelon optimiser benchmarked on "
    "held-out demand, and what-if scenarios that re-price a disruption. The "
    "natural-language layer over those scenarios is still to come."
)


@st.cache_data(ttl=300)
def load_catalog() -> dict:
    response = requests.get(f"{API_URL}/api/catalog", timeout=10)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def load_presets() -> dict:
    """What-if questions the API offers. Empty dict if the API is older."""
    try:
        response = requests.get(f"{API_URL}/api/scenarios/presets", timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {}


def poll_run(run_id: str, label: str) -> dict | None:
    """Shared polling loop: both a pipeline run and a scenario report here."""
    bar = st.progress(0, text="queued")
    state = None
    for _ in range(300):
        try:
            state = requests.get(f"{API_URL}/api/runs/{run_id}", timeout=20).json()
        except requests.RequestException:
            time.sleep(2)
            continue
        bar.progress(state["progress"] / 100,
                     text=f"{label}: {state['stage']} ({state['progress']}%)")
        if state["status"] in ("succeeded", "failed"):
            break
        time.sleep(2)
    return state


def wait_for_catalog(attempts: int = 9) -> dict:
    """Render's free tier sleeps an idle API, and waking it takes 30-60s. A
    single 10s request made the first visitor after a quiet spell see an error
    for a service that was only starting up, so retry for up to ~90s.
    """
    for attempt in range(attempts):
        try:
            return load_catalog()
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            with st.spinner("Waking the API - the free host sleeps when idle, "
                            "this can take up to a minute..."):
                time.sleep(10)


try:
    catalog = wait_for_catalog()
except requests.RequestException as exc:
    st.error(f"Could not load the model catalog from {API_URL}: {exc}")
    st.stop()

with st.sidebar:
    st.header("Run parameters")
    stores = st.multiselect("Stores", catalog["stores"], default=catalog["stores"][:1])
    # Default to the store's biggest sellers by revenue. A single-item default made
    # the first click a benchmark too small to be an equal-service comparison.
    top_sellers = ["FOODS_3_080", "FOODS_3_455", "FOODS_2_244", "FOODS_3_546",
                   "FOODS_3_534", "HOBBIES_1_158", "FOODS_3_136", "FOODS_2_183"]
    default_items = [i for i in top_sellers if i in catalog["items"]] or catalog["items"][:1]
    items = st.multiselect("Items", catalog["items"], default=default_items)
    horizon = st.slider(
        "Forecast horizon (days)", 7, catalog["max_horizon_days"], catalog["max_horizon_days"]
    )
    service_level = st.slider("Target service level", 0.80, 0.99, 0.95)
    with st.expander("Cost assumptions"):
        st.caption(
            "Holding and stockout costs are derived per item from real M5 sell "
            "prices; these are the rates applied to them."
        )
        holding_rate = st.slider("Annual holding rate", 0.05, 0.60, 0.25, 0.05)
        stockout_multiplier = st.slider(
            "Stockout penalty (x lost gross margin)", 0.5, 6.0, 2.0, 0.5)
    go = st.button("Run pipeline", type="primary", use_container_width=True)

    # --- Stage 3C: what-if. Kept in the same sidebar because it asks about the
    # same item selection; the answer lands in its own section below.
    st.divider()
    st.header("What-if scenario")
    st.caption(
        "Re-solves the same policy under a changed network and reports what "
        "moves. Two solves, so it takes about twice a pipeline run."
    )
    presets = load_presets()
    preset_labels = {p["key"]: p["label"] for p in presets.get("presets", [])}
    preset_labels["custom"] = "Custom: set the knobs myself"
    preset_key = st.selectbox(
        "Question", list(preset_labels), format_func=lambda k: preset_labels[k])
    overrides: dict = {}
    if preset_key == "custom":
        overrides["supplier_lead_days"] = st.slider("Supplier lead time (days)", 1, 28, 7)
        overrides["demand_pct"] = st.slider("Demand vs plan", 0.5, 2.0, 1.0, 0.05)
        overrides["truck_capacity_pct"] = st.slider("Truck capacity", 0.3, 3.0, 1.0, 0.05)
        overrides["dc_capacity_pct"] = st.slider("DC capacity", 0.3, 3.0, 1.0, 0.05)
        overrides["delivery_cost_pct"] = st.slider("Delivery cost", 0.0, 5.0, 1.0, 0.25)
    ask = st.button("Run what-if", use_container_width=True)
    st.caption(
        f"Trained through {catalog['trained_through']}. Forecast window "
        f"{catalog['forecast_window'][0]} to {catalog['forecast_window'][1]}."
    )

if ask:
    payload: dict = {"item_ids": items, "horizon_days": horizon}
    if preset_key == "custom":
        # Send only what the user actually moved; an unchanged knob must not
        # look like a deliberate "set it to the default" instruction.
        payload.update({k: v for k, v in overrides.items()})
    else:
        payload["preset"] = preset_key

    st.subheader("What-if")
    try:
        resp = requests.post(f"{API_URL}/api/scenarios", json=payload, timeout=20)
        if resp.status_code == 422:
            st.error(resp.json().get("detail", "That scenario was rejected."))
            st.stop()
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        st.stop()

    state = poll_run(resp.json()["run_id"], "scenario")
    if state is None or state["status"] == "failed":
        st.error(f"Scenario failed: {(state or {}).get('error')}")
        st.stop()

    report = (state.get("result") or {}).get("scenario", {})
    if report:
        base, alt, delta = (report["base"], report["scenario_result"],
                            report["delta"])
        st.success(report["headline"])
        st.caption(
            f"Changed: {report['scenario']['changes']} · both legs solved in "
            f"this run against the same actual sales, "
            f"{report['instance']['items']} items, "
            f"{report['instance']['window'][0]} to {report['instance']['window'][1]}."
        )
        cols = st.columns(4)
        cols[0].metric("Total cost", f"${alt['total_cost']:,.0f}",
                       f"{delta['cost_pct']:+.1f}%", delta_color="inverse")
        cols[1].metric("Fill rate", f"{alt['fill_rate']:.1%}",
                       f"{delta['fill_rate_pp']:+.1f} pp")
        cols[2].metric("Delivery days", alt["delivery_days"],
                       f"{delta['delivery_days']:+}", delta_color="off")
        cols[3].metric("PO lines", alt["po_lines"], f"{delta['po_lines']:+}",
                       delta_color="off")
        st.dataframe(
            [{"measure": name, "base": b, "scenario": s}
             for name, b, s in [
                 ("total cost", f"${base['total_cost']:,.2f}",
                  f"${alt['total_cost']:,.2f}"),
                 ("fill rate", f"{base['fill_rate']:.1%}", f"{alt['fill_rate']:.1%}"),
                 ("units short", f"{base['units_short']:,}", f"{alt['units_short']:,}"),
                 ("stockout cost", f"${base['costs']['stockout']:,.2f}",
                  f"${alt['costs']['stockout']:,.2f}"),
                 ("ordering cost", f"${base['costs']['ordering']:,.2f}",
                  f"${alt['costs']['ordering']:,.2f}"),
                 ("delivery cost", f"${base['costs']['delivery']:,.2f}",
                  f"${alt['costs']['delivery']:,.2f}"),
                 ("holding cost", f"${base['costs']['holding']:,.2f}",
                  f"${alt['costs']['holding']:,.2f}"),
                 ("truck utilisation", f"{base['truck_utilisation_mean']:.0%}",
                  f"{alt['truck_utilisation_mean']:.0%}"),
                 ("planning horizon (days)", base["planning_horizon_days"],
                  alt["planning_horizon_days"]),
             ]],
            use_container_width=True, hide_index=True,
        )
        if alt["relaxations"]:
            st.warning(
                "To stay feasible the solver had to give up: "
                + "; ".join(alt["relaxations"])
                + ". A scenario that relaxes a service constraint is at the "
                  "edge of what the network can physically do."
            )
        st.caption(
            "Baselines are not re-tuned here - this compares the optimiser "
            "against itself under two networks, which is the honest way to "
            "price a disruption. Re-running gives slightly different numbers "
            "because CP-SAT stops at a time limit."
        )
        with st.expander("Raw scenario response"):
            st.json(report)

if go:
    if not stores or not items:
        st.error("Pick at least one store and one item.")
        st.stop()

    try:
        resp = requests.post(
            f"{API_URL}/api/runs",
            json={
                "store_ids": stores,
                "item_ids": items,
                "horizon_days": horizon,
                "service_level": service_level,
                "annual_holding_rate": holding_rate,
                "stockout_multiplier": stockout_multiplier,
            },
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
        st.stop()

    run_id = resp.json()["run_id"]
    st.info(f"Run **{run_id}** accepted. Polling for results.")

    status_box = st.empty()
    state = poll_run(run_id, "pipeline")

    if state is None:
        st.error("No response from the API.")
        st.stop()

    if state["status"] == "failed":
        st.error(f"Run failed: {state['error']}")
        st.stop()

    status_box.success(f"Completed in {state['duration_ms'] / 1000:.1f}s")

    # ---- Instrumentation. Do not skip this; it is a differentiator. ----
    st.subheader("Stage timings")
    st.dataframe(
        [
            {"stage": t["stage_name"], "latency (ms)": t["latency_ms"]}
            for t in state["traces"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    result = state.get("result") or {}

    if "simulate" in result:
        sim = result["simulate"]
        st.subheader("Policy benchmark")
        st.caption(
            f"Back-tested on {sim['window'][0]} to {sim['window'][1]} - a window "
            f"the forecaster was trained strictly before, so these are costs "
            f"against demand it never saw. Every baseline is tuned to at least "
            f"the fill rate our policy achieved "
            f"({sim['protocol']['target_fill_rate']:.1%}), so the comparison is "
            f"at equal service, not equal stock."
        )
        left, mid, right = st.columns(3)
        left.metric(
            "Cost reduction vs fixed reorder point",
            f"{sim['improvement_vs_baseline_pct']:.1f}%",
        )
        mid.metric("At fill rate", f"{sim['headline']['at_fill_rate']:.1%}")
        right.metric(
            "Cost of forecast error",
            f"{sim['headline']['gap_to_perfect_information_pct']:.1f}%",
            help=sim.get("perfect_information_caveat", ""),
        )
        st.dataframe(sim["policies"], use_container_width=True, hide_index=True)

        proto = sim.get("protocol", {})
        if proto and not proto.get("matched_within_tolerance", True):
            st.warning(
                f"The compared baseline over-serves by "
                f"{proto['baseline_fill_overshoot_pp']['reorder_point_eoq']:.1f} "
                f"percentage points. A baseline cannot be tuned below zero safety "
                f"stock, so on a small item set it buys more service than ours at "
                f"its cheapest feasible setting - which flatters the percentage "
                f"above. Pick more items for a comparison worth quoting."
            )

        st.caption(
            "Where the saving comes from: holding cost is a rounding error at "
            "grocery margins. The optimiser wins by consolidating the fixed "
            "costs a per-item rule cannot see."
        )
        mech = sim.get("cost_mechanism", {})
        if mech:
            st.dataframe(
                [{"policy": k, "PO lines": v,
                  "delivery days": mech["delivery_days"].get(k)}
                 for k, v in mech["po_lines"].items()],
                use_container_width=True, hide_index=True,
            )

        ablation = sim.get("ablation", {})
        if ablation:
            st.caption(
                f"Ablation - the same rule fed the forecast instead of trailing "
                f"history saves "
                f"{ablation['forecast_step_pct_of_safety_stock']:.1f}%; replacing "
                f"the rule with the optimiser saves a further "
                f"{ablation['optimiser_step_pct_of_forecast_base_stock']:.1f}%. "
                f"That split is what separates the forecast's contribution from "
                f"the optimiser's."
            )

    if "optimise" in result:
        opt = result["optimise"]
        st.subheader("Replenishment plan")
        st.caption(
            f"CP-SAT plan for {opt['window'][0]} to {opt['window'][1]}, the window "
            f"the deployed model forecasts. No actuals exist for it - it is M5's "
            f"held-back future - so these are projected costs, not measured ones."
        )
        cols = st.columns(4)
        cols[0].metric("Solver status", opt["solver_status"])
        cols[1].metric(
            "Optimality gap",
            "-" if opt["optimality_gap_pct"] is None
            else f"{opt['optimality_gap_pct']:.1f}%",
        )
        cols[2].metric("Solve time", f"{opt['solve_time_ms'] / 1000:.1f}s")
        cols[3].metric("Planned delivery days", opt["delivery_days_planned"])
        st.write("Projected cost breakdown (USD)")
        st.dataframe([opt["projected_costs_usd"]], use_container_width=True,
                     hide_index=True)
        if opt.get("relaxations"):
            st.warning(
                "Constraints relaxed to keep the instance feasible: "
                + "; ".join(opt["relaxations"])
            )
        if opt.get("items_dropped_for_size"):
            st.info(
                f"{len(opt['items_dropped_for_size'])} requested item(s) were "
                f"dropped to keep the solve inside its time budget: "
                + ", ".join(opt["items_dropped_for_size"][:8])
            )
        with st.expander("Per-item order and shipment schedule"):
            st.json(opt["orders"])

    if "forecast" in result and result["forecast"]["series"]:
        forecast = result["forecast"]
        st.subheader("Forecast")
        interval = forecast.get("interval", {})
        st.caption(
            f"Model: {forecast['model']} | backtested WRMSSE {forecast.get('wrmsse')} "
            f"| {interval.get('coverage', 0):.1%} interval coverage"
        )
        if forecast.get("unavailable"):
            st.warning(
                "Not forecastable (model has no history for them): "
                + ", ".join(u["item_id"] for u in forecast["unavailable"])
            )
        first = forecast["series"][0]
        st.caption(f"{first['store_id']} / {first['item_id']}")
        st.line_chart(
            {"point": [p["point"] for p in first["points"]],
             "lower": [p["lower"] for p in first["points"]],
             "upper": [p["upper"] for p in first["points"]]}
        )

    with st.expander("Raw response"):
        st.json(state)
elif not ask:
    st.info(
        "Set parameters in the sidebar, then press **Run pipeline** for the "
        "forecast-optimise-benchmark pipeline, or **Run what-if** to price a "
        "disruption against the same items."
    )
