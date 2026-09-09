"""
Session 5 - baselines. Every model built after this is measured against
these numbers. Skipping this step means you can never tell whether a fancy
model (LightGBM, later) is actually better than doing almost nothing.

Backtesting is rolling-origin (walk-forward): each score comes from
forecasting a 28-day window using only data that existed before that window
started. A random train/test split would leak future dates into "training"
and make every number meaningless - this is the #1 mistake the plan warns
about, so it's built in here from day one rather than left for later.

Run:
    python -m forecasting.baselines
"""
import numpy as np
import pandas as pd

from forecasting.data import load_store_sales, attach_dates

STORE = "CA_1"
N_ITEMS = 400  # approximate; stratified proportionally across departments
HORIZON = 28   # matches M5's actual forecast horizon
SEASON = 7     # weekly seasonality
N_ORIGINS = 6  # rolling-origin backtest windows per item


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season: int = SEASON) -> np.ndarray:
    """Forecast day t as the actual value from `season` days earlier."""
    tail = history[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(tail, reps)[:horizon]


def moving_average_forecast(history: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    """Forecast every day in the horizon as the mean of the last `window` days."""
    avg = history[-window:].mean()
    return np.full(horizon, avg)


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mase(actual: np.ndarray, forecast: np.ndarray, history: np.ndarray, season: int = SEASON) -> float:
    """Mean Absolute Scaled Error: MAE scaled by the in-sample seasonal-naive
    error. Stays meaningful even when most days are zero units sold, unlike
    plain RMSE, which is dominated by the rare non-zero days on sparse items.
    """
    mae = np.mean(np.abs(actual - forecast))
    naive_errors = np.abs(history[season:] - history[:-season])
    scale = naive_errors.mean()
    if scale == 0:
        return np.nan  # item never varies week to week - can't scale against it
    return mae / scale


def rmsse(actual: np.ndarray, forecast: np.ndarray, history: np.ndarray) -> float:
    """Root Mean Squared Scaled Error - the per-series component of M5's official
    WRMSSE. Scaled by the in-sample one-step naive error, which is what makes
    errors comparable across items selling 0.2 versus 20 units a day.
    """
    scale = np.mean(np.diff(history) ** 2)
    if scale == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual - forecast) ** 2) / scale))


def backtest_item(series: np.ndarray, method, horizon: int, n_origins: int) -> dict:
    """Walk backward through n_origins non-overlapping windows, each time
    forecasting `horizon` days ahead using only the data that would have
    been available at that point in time.
    """
    rmses, mases, rmsses = [], [], []
    total_needed = n_origins * horizon
    if len(series) < total_needed + SEASON:
        return {"rmse": np.nan, "mase": np.nan, "rmsse": np.nan, "n_origins": 0}

    for i in range(n_origins):
        test_end = len(series) - i * horizon
        test_start = test_end - horizon
        history = series[:test_start]
        actual = series[test_start:test_end]

        forecast = method(history, horizon)
        rmses.append(rmse(actual, forecast))
        mases.append(mase(actual, forecast, history))
        rmsses.append(rmsse(actual, forecast, history))

    return {
        "rmse": float(np.nanmean(rmses)),
        "mase": float(np.nanmean(mases)),
        "rmsse": float(np.nanmean(rmsses)),
        "n_origins": n_origins,
    }


METHODS = {
    "seasonal_naive": lambda h, n: seasonal_naive_forecast(h, n, SEASON),
    "moving_average_28d": lambda h, n: moving_average_forecast(h, n, 28),
}


def score_all_items(long_df: pd.DataFrame, horizon: int = HORIZON,
                    n_origins: int = N_ORIGINS) -> pd.DataFrame:
    """Per-item, per-method scores. Returned rather than aggregated so callers
    can break results down - by volume, department, anything - instead of only
    seeing a single average that hides where a method wins and loses.
    """
    rows = []
    ordered = long_df.sort_values(["item_id", "date"])
    for item_id, group in ordered.groupby("item_id", observed=True):
        series = group["sales"].to_numpy()
        for name, method in METHODS.items():
            scores = backtest_item(series, method, horizon, n_origins)
            if scores["n_origins"] > 0:
                rows.append({"item_id": item_id, "method": name, **scores})
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Loading store {STORE}, ~{N_ITEMS} items stratified by department...")
    long_df = attach_dates(load_store_sales(STORE, N_ITEMS))

    per_dept = long_df.groupby("dept_id")["item_id"].nunique()
    zero_pct = (long_df["sales"] == 0).mean() * 100
    print(f"{per_dept.sum()} items across {len(per_dept)} departments:")
    print(per_dept.to_string())
    print(f"Zero-sales days in this sample: {zero_pct:.1f}%")

    scored = score_all_items(long_df)
    n_items = long_df["item_id"].nunique()
    skipped = n_items - scored.groupby("method").size().min()

    print(f"\nBacktested {N_ORIGINS} rolling-origin windows of {HORIZON} days each per item")
    print(f"({skipped} items had too little history and were skipped).\n")

    summary = scored.groupby("method")[["rmse", "mase"]].mean()
    counts = scored.groupby("method").size()

    print(f"{'Method':<22}{'RMSE':>10}{'MASE':>10}{'Items scored':>15}")
    print("-" * 57)
    for name in METHODS:
        row = summary.loc[name]
        print(f"{name:<22}{row['rmse']:>10.3f}{row['mase']:>10.3f}{counts[name]:>15}")

    print("\nA MASE below 1.0 means a method beats plain seasonal-naive; above 1.0 means")
    print("it's worse. LightGBM (Stage 3A's real model) must beat these numbers on both")
    print("metrics, on the same rolling-origin windows, or it isn't worth deploying.")


if __name__ == "__main__":
    main()
