"""
Produce the artifacts the service loads. Separate from train.py on purpose:
train.py answers "is this model any good", this answers "build the thing that
gets deployed". You do not want to regenerate a deployed artifact every time you
run an experiment.

Three models are fitted on the full history - a Tweedie point forecast and two
quantile models for the interval - and saved in LightGBM's native text format
rather than pickled. Pickles of sklearn wrappers break when library versions
move; the text format does not, which matters given how much of this project has
already been spent on version drift.

Alongside them goes a precomputed inference snapshot: every feature, for every
item, for each of the 28 days after the sales history ends. The service then
never touches the raw 325MB of CSVs, never recomputes a rolling window, and
therefore cannot disagree with training about what a feature means. This works
because M5 is a static dataset with exactly one forecast window available; a
system receiving live data would recompute features from a feature store on each
request instead.

Run:
    python -m forecasting.fit
"""
import json

import lightgbm as lgb
import pandas as pd

from forecasting.data import append_future_days, load_store_sales
from forecasting.features import CATEGORICALS, build_feature_frame, feature_columns
from forecasting.train import (
    BASE_PARAMS, METRICS_PATH, MODELS_DIR, N_ITEMS, POINT_PARAMS, QUANTILES, STORE,
)

HORIZON = 28
SNAPSHOT_PATH = MODELS_DIR / "inference_snapshot.parquet"
METADATA_PATH = MODELS_DIR / "metadata.json"


def fit_models(train: pd.DataFrame, features: list[str],
               trees: dict[str, int]) -> dict[str, lgb.LGBMRegressor]:
    """Fit on everything, at the tree counts early stopping chose during the
    backtest.

    Holding data back for early stopping here would make the shipped model worse
    than the one that was measured. But running to the full n_estimators instead
    would ship something materially more overfit - the backtest windows stopped
    between 46 and 543 trees, nowhere near 1000. Reusing the median of what was
    actually chosen is the only option that keeps the artifact and the reported
    numbers describing the same model.
    """
    configs = {"point": POINT_PARAMS}
    for alpha, name in zip(QUANTILES, ("lower", "upper")):
        configs[name] = dict(BASE_PARAMS, objective="quantile", alpha=alpha)

    models = {}
    for name, params in configs.items():
        n_trees = trees.get(name, params["n_estimators"])
        models[name] = lgb.LGBMRegressor(**dict(params, n_estimators=n_trees))
        models[name].fit(train[features], train["sales"])
        print(f"  fitted {name}: {n_trees} trees")
    return models


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Building training frame for {STORE}, ~{N_ITEMS} items...")
    train = build_feature_frame(load_store_sales(STORE, N_ITEMS), HORIZON)
    features = feature_columns(train)
    print(f"{len(train):,} rows x {len(features)} features.")

    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    if "trees" not in metrics:
        raise SystemExit(
            "models/backtest_metrics.json has no tree counts - run "
            "`python -m forecasting.train` first so the deployed model inherits "
            "the tree count its backtest actually validated."
        )

    print("Fitting on full history...")
    models = fit_models(train, features, metrics["trees"])

    # Rebuilt separately, with future days appended, so that the horizon blocks
    # land on the forecast window instead of on history.
    print("Building inference snapshot...")
    history = load_store_sales(STORE, N_ITEMS)
    full = build_feature_frame(append_future_days(history, HORIZON), HORIZON)
    snapshot = full[full["sales"].isna()].copy()

    # Category codes are what LightGBM actually sees, so the snapshot must use
    # the training frame's exact category lists. Rebuilding them from whatever
    # happens to appear in the snapshot would silently shift every code.
    for col in CATEGORICALS:
        snapshot[col] = pd.Categorical(snapshot[col], categories=train[col].cat.categories)

    keep = list(dict.fromkeys(["item_id", "store_id", "date", "h"] + features))
    snapshot = snapshot[keep].sort_values(["item_id", "date"]).reset_index(drop=True)

    for name, model in models.items():
        model.booster_.save_model(str(MODELS_DIR / f"{name}.txt"))
    snapshot.to_parquet(SNAPSHOT_PATH, index=False)

    METADATA_PATH.write_text(json.dumps({
        "store": STORE,
        "horizon_days": HORIZON,
        "features": features,
        "categoricals": {col: list(train[col].cat.categories) for col in CATEGORICALS},
        "trained_through": str(train["date"].max().date()),
        "forecast_dates": [str(d.date()) for d in sorted(snapshot["date"].unique())],
        "items": sorted(snapshot["item_id"].astype(str).unique().tolist()),
        "backtest": metrics.get("methods", {}).get("lightgbm", {}),
        "interval": metrics.get("interval", {}),
    }, indent=2))

    print(f"\nSaved to {MODELS_DIR.name}/:")
    print(f"  point.txt, lower.txt, upper.txt")
    print(f"  inference_snapshot.parquet  ({len(snapshot):,} rows, "
          f"{SNAPSHOT_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  metadata.json")
    print(f"\nForecast window: {snapshot['date'].min().date()} to {snapshot['date'].max().date()}")


if __name__ == "__main__":
    main()
