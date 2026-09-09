"""
Serving-time forecasting.

Artifacts load once per process and are reused across requests - a worker that
reloaded 3 models and a parquet file on every job would spend most of its time
on I/O. Loading is lazy, so an image built without artifacts still starts and
fails per-run with a clear message rather than refusing to boot.

No feature engineering happens here. Everything was precomputed by
forecasting/fit.py using the same code path as training, so serving cannot drift
from training by construction. See fit.py for why that trade is worth making on
a static dataset.
"""
import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_NAMES = ("point", "lower", "upper")


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[dict, pd.DataFrame, dict]:
    if not (MODELS_DIR / "metadata.json").exists():
        raise FileNotFoundError(
            f"No model artifacts in {MODELS_DIR}. Run `python -m forecasting.fit`."
        )
    metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
    snapshot = pd.read_parquet(MODELS_DIR / "inference_snapshot.parquet")
    models = {
        name: lgb.Booster(model_file=str(MODELS_DIR / f"{name}.txt"))
        for name in MODEL_NAMES
    }
    return metadata, snapshot, models


def available() -> dict:
    """What the deployed model can actually answer for."""
    metadata, snapshot, _ = load_artifacts()
    return {
        "stores": sorted(snapshot["store_id"].unique().tolist()),
        "items": metadata["items"],
        "max_horizon_days": metadata["horizon_days"],
        "forecast_window": [metadata["forecast_dates"][0], metadata["forecast_dates"][-1]],
        "trained_through": metadata["trained_through"],
    }


def forecast(store_ids: list[str], item_ids: list[str], horizon_days: int) -> dict:
    """Point forecast plus an 80% interval per store-item-day.

    Items the model has never seen are reported in `unavailable` rather than
    silently omitted or guessed at - a forecast for an item with no history
    would be indistinguishable from a real one in the response.
    """
    metadata, snapshot, models = load_artifacts()
    features = metadata["features"]
    horizon = min(horizon_days, metadata["horizon_days"])

    rows = snapshot[
        snapshot["store_id"].isin(store_ids)
        & snapshot["item_id"].isin(item_ids)
        & (snapshot["h"] <= horizon)
    ]

    requested = {(store, item) for store in store_ids for item in item_ids}
    found = set(map(tuple, rows[["store_id", "item_id"]].drop_duplicates().to_numpy()))
    unavailable = [{"store_id": s, "item_id": i} for s, i in sorted(requested - found)]

    if rows.empty:
        raise ValueError(
            f"No forecastable store-item pairs. The deployed model covers store "
            f"{metadata['store']} and {len(metadata['items'])} items; "
            f"examples: {', '.join(metadata['items'][:3])}."
        )

    predictions = {
        name: np.clip(model.predict(rows[features]), 0, None)
        for name, model in models.items()
    }

    # The three models are trained independently, so nothing couples them and
    # the interval can cross the point forecast - measured at 6 rows in 11,228
    # (0.05%), all point > upper, by margins under 0.01 units. The interval is
    # widened to contain the point rather than the point being clipped into the
    # interval: the point forecast is what WRMSSE was measured on, and serving a
    # different number than the one that was evaluated is not a trade worth
    # making to tidy up a rounding-scale inconsistency.
    point = predictions["point"]
    output = rows[["store_id", "item_id", "date"]].copy()
    output["point"] = point
    output["lower"] = np.minimum(predictions["lower"], point)
    output["upper"] = np.maximum(predictions["upper"], point)

    series = []
    for (store, item), group in output.groupby(["store_id", "item_id"], observed=True):
        series.append({
            "store_id": str(store),
            "item_id": str(item),
            "points": [
                {
                    "date": row.date.date().isoformat(),
                    "point": round(float(row.point), 2),
                    "lower": round(float(row.lower), 2),
                    "upper": round(float(row.upper), 2),
                }
                for row in group.sort_values("date").itertuples()
            ],
        })

    return {
        "series": series,
        "model": "lightgbm",
        "wrmsse": metadata.get("backtest", {}).get("wrmsse"),
        "interval": metadata.get("interval", {}),
        "horizon_days": horizon,
        "trained_through": metadata["trained_through"],
        "unavailable": unavailable,
    }
