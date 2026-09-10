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
st.caption("Stage 3A - real LightGBM forecasts; optimise/simulate/explain still placeholders.")


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
    items = st.multiselect("Items", catalog["items"], default=catalog["items"][:1])
    horizon = st.slider(
        "Forecast horizon (days)", 7, catalog["max_horizon_days"], catalog["max_horizon_days"]
    )
    service_level = st.slider("Target service level", 0.80, 0.99, 0.95)
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
    for _ in range(180):
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
        st.subheader("Policy benchmark")
        st.caption("The headline number of the whole project comes from this table.")
        st.dataframe(result["simulate"]["policies"], use_container_width=True, hide_index=True)
        st.metric(
            "Cost reduction vs fixed reorder point",
            f"{result['simulate']['improvement_vs_baseline_pct']:.1f}%",
        )

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
