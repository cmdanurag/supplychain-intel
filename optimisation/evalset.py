"""
The evaluation window for Stage 3B.

**Why this file has to exist.** The deployed forecaster is trained through
2016-05-22 and predicts the 28 days after it - but `sales_train_evaluation.csv`
also ends on 2016-05-22, because that window is M5's held-back competition
future. So for the window the API actually serves, no ground truth exists
anywhere in this repo. Benchmarking inventory policies there would mean scoring
them against the same forecast that produced them, which measures nothing.

The fix is to benchmark one window earlier. Stage 3A's most recent
rolling-origin window (2016-04-25 to 2016-05-22) has both halves: a model
trained strictly before it produces genuine out-of-sample forecasts, and the
actual sales are in the file. Policies are decided on the forecasts and scored
on the actuals, which is the only arrangement that makes a cost comparison mean
anything.

The window, the features, the parameters and the early-stopping split are all
taken from forecasting.train, not re-declared, so the forecasts here are the
same quality as the ones its WRMSSE describes.

Also written: mean and sigma of daily demand over the 56 days *before* the
window opens. The classical baselines are not allowed to see the forecast, so
this is the information a firm running a reorder-point policy would have had.

Build with:
    python -m optimisation.evalset
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting.baselines import HORIZON, N_ITEMS, N_ORIGINS, STORE
from forecasting.data import attach_dates, load_store_sales
from forecasting.features import build_features, feature_columns
from forecasting.train import BASE_PARAMS, POINT_PARAMS, QUANTILES, VALIDATION_DAYS
from forecasting.train import rolling_origin_windows

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
EVAL_PATH = MODELS_DIR / "eval_window.parquet"
HISTORY_PATH = MODELS_DIR / "demand_history.parquet"
EVAL_META_PATH = MODELS_DIR / "eval_window.json"

# Eight full weeks, so a weekly demand cycle is represented evenly and the sigma
# estimate does not depend on which weekday the window happens to end on.
HISTORY_DAYS = 56


def _fit(fit: pd.DataFrame, valid: pd.DataFrame, features: list[str], params: dict):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        fit[features], fit["sales"],
        eval_set=[(valid[features], valid["sales"])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def build() -> None:
    print(f"Building features for store {STORE}, ~{N_ITEMS} items...")
    df = build_features(STORE, N_ITEMS, HORIZON)
    features = feature_columns(df)

    # windows[0] is the most recent - the latest 28 days for which actuals exist.
    start, end = (pd.Timestamp(x) for x in rolling_origin_windows(
        df["date"], HORIZON, N_ORIGINS)[0])
    print(f"Evaluation window: {start.date()} to {end.date()}")

    train = df[df["date"] < start]
    test = df[(df["date"] >= start) & (df["date"] <= end)]
    cutoff = start - pd.Timedelta(days=VALIDATION_DAYS)
    fit, valid = train[train["date"] < cutoff], train[train["date"] >= cutoff]
    print(f"  train {len(fit):,} rows, early-stopping holdout {len(valid):,}, "
          f"test {len(test):,}")

    out = test[["store_id", "item_id", "date", "sales"]].copy()
    out = out.rename(columns={"sales": "demand"})

    configs = {"point": POINT_PARAMS}
    for alpha, name in zip(QUANTILES, ("lower", "upper")):
        configs[name] = dict(BASE_PARAMS, objective="quantile", alpha=alpha)
    for name, params in configs.items():
        model = _fit(fit, valid, features, params)
        out[name] = np.clip(model.predict(test[features]), 0, None)
        print(f"  fitted {name}: {model.best_iteration_} trees")

    # Same ordering guarantee predict.forecast() gives the API, so the simulator
    # can index by integer day offset without re-sorting.
    out["item_id"] = out["item_id"].astype(str)
    out = out.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)
    out["t"] = out.groupby(["store_id", "item_id"], observed=True).cumcount()

    # --- history stats, strictly before the window opens ---
    raw = attach_dates(load_store_sales(STORE, N_ITEMS))
    hist = raw[(raw["date"] < start) & (raw["date"] >= start - pd.Timedelta(days=HISTORY_DAYS))]
    grouped = hist.groupby("item_id", observed=True)["sales"]
    history = pd.DataFrame({
        "mean_daily": grouped.mean(),
        "sigma_daily": grouped.std(ddof=1),
        "nonzero_share": grouped.apply(lambda s: float((s > 0).mean())),
        "history_days": grouped.size(),
    }).reset_index()
    history["item_id"] = history["item_id"].astype(str)
    history["store_id"] = STORE
    history = history[history["item_id"].isin(out["item_id"].unique())]

    # Mean sell price over the window. The cost model derives per-item holding
    # and stockout economics from real M5 prices rather than inventing them.
    prices = test.groupby("item_id", observed=True)["sell_price"].mean().reset_index()
    prices["item_id"] = prices["item_id"].astype(str)
    history = history.merge(prices, on="item_id", how="left")

    MODELS_DIR.mkdir(exist_ok=True)
    out.to_parquet(EVAL_PATH, index=False)
    history.to_parquet(HISTORY_PATH, index=False)
    EVAL_META_PATH.write_text(json.dumps({
        "store": STORE,
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "horizon_days": HORIZON,
        "items": int(out["item_id"].nunique()),
        "history_days": HISTORY_DAYS,
        "history_start": str((start - pd.Timedelta(days=HISTORY_DAYS)).date()),
        "total_actual_units": int(out["demand"].sum()),
        "forecast_bias_units_per_day": round(
            float(out["point"].mean() - out["demand"].mean()), 4),
        "interval_coverage": round(float(
            ((out["demand"] >= out["lower"]) & (out["demand"] <= out["upper"])).mean()), 4),
    }, indent=2))

    print(f"\neval_window.parquet   {out.shape}  {out['item_id'].nunique()} items")
    print(f"demand_history.parquet {history.shape}")
    print(f"total actual units: {out['demand'].sum():,}  "
          f"forecast total: {out['point'].sum():,.0f}")
    print(f"interval coverage on this window: "
          f"{((out['demand'] >= out['lower']) & (out['demand'] <= out['upper'])).mean():.1%}")


def load(item_ids: list[str] | None = None, horizon_days: int = HORIZON
         ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Forecasts+actuals, history stats, and window metadata."""
    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"No evaluation window in {MODELS_DIR}. Run `python -m optimisation.evalset`."
        )
    evalset = pd.read_parquet(EVAL_PATH)
    history = pd.read_parquet(HISTORY_PATH)
    meta = json.loads(EVAL_META_PATH.read_text())

    if item_ids is not None:
        evalset = evalset[evalset["item_id"].isin(item_ids)]
        history = history[history["item_id"].isin(item_ids)]
        if evalset.empty:
            raise ValueError("None of the requested items are in the evaluation window.")
    evalset = evalset[evalset["t"] < horizon_days].reset_index(drop=True)
    return evalset, history.reset_index(drop=True), meta


if __name__ == "__main__":
    build()
