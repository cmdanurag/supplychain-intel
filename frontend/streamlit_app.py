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
st.caption("Stages 3A-3B live: LightGBM forecasts, CP-SAT multi-echelon optimiser, benchmarked on held-out demand. Explain is still a placeholder.")


@st.cache_data(ttl=300)
def load_catalog() -> dict:
    response = requests.get(f"{API_URL}/api/catalog", timeout=10)
    response.raise_for_status()
    return response.json()


try:
    catalog = load_catalog()
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
    st.caption(
        f"Trained through {catalog['trained_through']}. Forecast window "
        f"{catalog['forecast_window'][0]} to {catalog['forecast_window'][1]}."
    )

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

    bar = st.progress(0, text="queued")
    status_box = st.empty()
    state = None

    # The polling loop. This is the client half of the async pattern.
    for _ in range(300):
        try:
            state = requests.get(f"{API_URL}/api/runs/{run_id}", timeout=10).json()
        except requests.RequestException:
            time.sleep(2)
            continue

        bar.progress(state["progress"] / 100, text=f"{state['stage']} ({state['progress']}%)")
        if state["status"] in ("succeeded", "failed"):
            break
        time.sleep(2)

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
else:
    st.info("Set parameters in the sidebar and press **Run pipeline**.")
