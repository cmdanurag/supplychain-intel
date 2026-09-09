"""
Stage 3A - the real model. LightGBM, pooled across every item in the slice.

Scored on exactly the same six rolling-origin windows as forecasting/baselines.py,
with the same per-item MASE denominators, because a model that beats a baseline
measured differently has not beaten anything. Settings are imported from
baselines rather than re-declared, so the two cannot silently drift apart.

One model is trained per window on data strictly preceding it, so the six scores
come from six genuinely out-of-sample forecasts. The last 28 days before each
cutoff are held out for early stopping - still leak-free, since they precede the
test window.

Pooling every item into one model is the point: day-of-week and SNAP effects are
weak per series but strong across the panel, which is exactly where the
per-series baselines run out of signal.

Run:
    python -m forecasting.train
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting.baselines import (
    HORIZON, N_ITEMS, N_ORIGINS, SEASON, STORE, score_all_items,
)
from forecasting.data import attach_dates, load_store_sales
from forecasting.features import build_features, feature_columns

VALIDATION_DAYS = 28
QUANTILES = (0.1, 0.9)  # an 80% interval; Stage 3B needs this to set safety stock

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_PATH = MODELS_DIR / "backtest_metrics.json"

BASE_PARAMS = dict(
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,  # leaf-wise growth overfits without this
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    n_estimators=1000,
    random_state=42,
    verbose=-1,
)

# Tweedie for the point forecast: non-negative and zero-inflated, and ~65% of
# item-days sell nothing. The quantile models answer a different question - not
# "what is the expected demand" but "what level is demand unlikely to exceed" -
# which is the one safety stock actually depends on.
POINT_PARAMS = dict(BASE_PARAMS, objective="tweedie", tweedie_variance_power=1.1)


def rolling_origin_windows(dates: pd.Series, horizon: int, n_origins: int) -> list:
    """Test windows, most recent first, matching baselines.backtest_item."""
    unique = np.sort(dates.unique())
    windows = []
    for i in range(n_origins):
        end = len(unique) - i * horizon
        windows.append((unique[end - horizon], unique[end - 1]))
    return windows


def mase_scales(raw_pivot: pd.DataFrame, window_start) -> pd.Series:
    """In-sample seasonal-naive error per item, using only pre-window history.

    Identical denominator to baselines.mase, so the MASE columns compare directly.
    """
    hist = raw_pivot.loc[raw_pivot.index < window_start].to_numpy()
    scales = np.abs(hist[SEASON:] - hist[:-SEASON]).mean(axis=0)
    return pd.Series(scales, index=raw_pivot.columns).replace(0.0, np.nan)


def rmsse_scales(raw_pivot: pd.DataFrame, window_start) -> pd.Series:
    """One-step naive squared error per item - the RMSSE denominator, matching
    baselines.rmsse."""
    hist = raw_pivot.loc[raw_pivot.index < window_start].to_numpy()
    scales = (np.diff(hist, axis=0) ** 2).mean(axis=0)
    return pd.Series(scales, index=raw_pivot.columns).replace(0.0, np.nan)


def dollar_weights(df: pd.DataFrame, before, horizon: int = HORIZON) -> pd.Series:
    """Share of revenue per item over the 28 days preceding `before`.

    M5's WRMSSE weights each series by its dollar sales, because a 10% error on an
    item turning over $500 a week matters more than the same error on one selling
    a unit a fortnight. Weights are taken from before the earliest evaluation
    window, so they are out-of-sample for every window scored.
    """
    window = df[(df["date"] < before) & (df["date"] >= before - pd.Timedelta(days=horizon))]
    revenue = (window["sales"] * window["sell_price"]).groupby(
        window["item_id"], observed=True
    ).sum()
    return revenue / revenue.sum()


def main() -> None:
    print(f"Building features for store {STORE}, ~{N_ITEMS} items...")
    df = build_features(STORE, N_ITEMS, HORIZON)
    features = feature_columns(df)
    print(f"{len(df):,} rows x {len(features)} features after dropping unusable rows.")

    raw = attach_dates(load_store_sales(STORE, N_ITEMS))
    raw_pivot = raw.pivot(index="date", columns="item_id", values="sales").sort_index()

    windows = rolling_origin_windows(df["date"], HORIZON, N_ORIGINS)
    per_item_rmse: dict[str, list[float]] = {}
    per_item_mase: dict[str, list[float]] = {}
    per_item_rmsse: dict[str, list[float]] = {}
    per_item_coverage: dict[str, list[float]] = {}
    per_item_width: dict[str, list[float]] = {}
    tree_counts: dict[str, list[int]] = {}
    importances = []

    for i, (start, end) in enumerate(windows, 1):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        train = df[df["date"] < start]
        test = df[(df["date"] >= start) & (df["date"] <= end)]

        cutoff = start - pd.Timedelta(days=VALIDATION_DAYS)
        fit = train[train["date"] < cutoff]
        valid = train[train["date"] >= cutoff]

        def train_model(params: dict):
            model = lgb.LGBMRegressor(**params)
            model.fit(
                fit[features], fit["sales"],
                eval_set=[(valid[features], valid["sales"])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            return model

        def record_trees(name: str, fitted) -> None:
            tree_counts.setdefault(name, []).append(
                int(fitted.best_iteration_ or fitted.n_estimators)
            )

        model = train_model(POINT_PARAMS)
        record_trees("point", model)

        scored = test[["item_id", "sales"]].copy()
        scored["prediction"] = np.clip(model.predict(test[features]), 0, None)
        for alpha, name in zip(QUANTILES, ("lower", "upper")):
            quantile_model = train_model(dict(BASE_PARAMS, objective="quantile", alpha=alpha))
            record_trees(name, quantile_model)
            scored[name] = np.clip(quantile_model.predict(test[features]), 0, None)

        scales = mase_scales(raw_pivot, start)
        squared_scales = rmsse_scales(raw_pivot, start)
        for item, group in scored.groupby("item_id", observed=True):
            error = np.abs(group["sales"] - group["prediction"])
            mse = float(np.mean((group["sales"] - group["prediction"]) ** 2))
            per_item_rmse.setdefault(item, []).append(float(np.sqrt(mse)))
            per_item_mase.setdefault(item, []).append(
                float(error.mean()) / scales.get(item, np.nan)
            )
            per_item_rmsse.setdefault(item, []).append(
                float(np.sqrt(mse / squared_scales.get(item, np.nan)))
            )
            inside = (group["sales"] >= group["lower"]) & (group["sales"] <= group["upper"])
            per_item_coverage.setdefault(item, []).append(float(inside.mean()))
            per_item_width.setdefault(item, []).append(
                float((group["upper"] - group["lower"]).mean())
            )

        importances.append(pd.Series(model.feature_importances_, index=features))
        print(
            f"  window {i}/{N_ORIGINS}  {start.date()} to {end.date()}  "
            f"trees={model.best_iteration_}  train_rows={len(fit):,}"
        )

    lgb_scores = pd.DataFrame({
        "item_id": list(per_item_rmse),
        "method": "lightgbm",
        "rmse": [float(np.mean(v)) for v in per_item_rmse.values()],
        "mase": [float(np.nanmean(v)) for v in per_item_mase.values()],
        "rmsse": [float(np.nanmean(v)) for v in per_item_rmsse.values()],
        "coverage": [float(np.mean(v)) for v in per_item_coverage.values()],
        "width": [float(np.mean(v)) for v in per_item_width.values()],
    })

    print("\nScoring baselines on the same windows...")
    columns = ["item_id", "method", "rmse", "mase", "rmsse"]
    scored = pd.concat([score_all_items(raw)[columns], lgb_scores[columns]])

    # Terciles come from mean daily sales, because a per-item average weights a
    # near-zero seller the same as a fast mover - which is exactly what hides
    # whether the model earns its keep where the money is.
    volume = raw.groupby("item_id")["sales"].mean()
    scored["tercile"] = scored["item_id"].map(pd.qcut(volume, 3, labels=["low", "mid", "high"]))

    weights = dollar_weights(df, windows[-1][0])
    scored["weight"] = scored["item_id"].map(weights)

    def weighted_rmsse(group: pd.DataFrame) -> float:
        usable = group.dropna(subset=["rmsse", "weight"])
        return float((usable["rmsse"] * usable["weight"]).sum() / usable["weight"].sum())

    print(f"\n{'Method':<22}{'RMSE':>10}{'MASE':>10}{'WRMSSE':>10}{'Items':>8}")
    print("-" * 60)
    averages = scored.groupby("method")[["rmse", "mase"]].mean()
    for name, row in averages.iterrows():
        subset = scored[scored["method"] == name]
        print(
            f"{name:<22}{row['rmse']:>10.3f}{row['mase']:>10.3f}"
            f"{weighted_rmsse(subset):>10.3f}{len(subset):>8}"
        )
    print("\nWRMSSE is revenue-weighted (item level only; the official metric also")
    print("averages across 12 aggregation levels). RMSE and MASE weight every item")
    print("equally, so they are dominated by the near-zero sellers.")

    print("\nBy item volume tercile (mean daily units in parentheses):")
    sizes = volume.groupby(pd.qcut(volume, 3, labels=["low", "mid", "high"]),
                           observed=True).mean()
    for metric in ("rmse", "mase"):
        table = scored.pivot_table(index="tercile", columns="method",
                                   values=metric, observed=True)
        table.index = [f"{t} ({sizes[t]:.2f})" for t in table.index]
        print(f"\n{metric.upper()}:")
        print(table.round(3).to_string())

    nominal = (QUANTILES[1] - QUANTILES[0]) * 100
    lgb_scores["tercile"] = lgb_scores["item_id"].map(
        pd.qcut(volume, 3, labels=["low", "mid", "high"])
    )
    print(f"\nPrediction intervals (quantile {QUANTILES[0]}-{QUANTILES[1]}, "
          f"{nominal:.0f}% nominal coverage):")
    print(f"  overall {lgb_scores['coverage'].mean():.1%} coverage, "
          f"mean width {lgb_scores['width'].mean():.2f} units")
    intervals = lgb_scores.groupby("tercile", observed=True)[["coverage", "width"]].mean()
    print(intervals.round(3).to_string())
    print("  Coverage far above nominal means the intervals are too wide to be")
    print("  useful; far below means safety stock built on them would run short.")

    top = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    print("\nTop features by average split gain:")
    print(top.head(12).to_string())

    # Written out so the deployed artifact can quote its own measured accuracy
    # instead of a number copy-pasted into a second file and left to drift.
    MODELS_DIR.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "store": STORE,
        "items": int(scored["item_id"].nunique()),
        "windows": N_ORIGINS,
        "horizon_days": HORIZON,
        "methods": {
            name: {
                "rmse": round(float(averages.loc[name, "rmse"]), 4),
                "mase": round(float(averages.loc[name, "mase"]), 4),
                "wrmsse": round(weighted_rmsse(scored[scored["method"] == name]), 4),
            }
            for name in averages.index
        },
        "interval": {
            "quantiles": list(QUANTILES),
            "coverage": round(float(lgb_scores["coverage"].mean()), 4),
            "mean_width": round(float(lgb_scores["width"].mean()), 4),
        },
        # Median of what early stopping actually chose per window. The deployed
        # model is fitted without a holdout, so it needs a tree count from
        # somewhere - and the honest source is the backtest that validated it.
        "trees": {name: int(np.median(counts)) for name, counts in tree_counts.items()},
    }, indent=2))
    print(f"\nMetrics written to {METRICS_PATH.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    main()
