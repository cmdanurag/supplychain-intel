"""
Feature engineering for Stage 3A - origin-anchored direct multi-step.

THE RULE: a feature for target day t must be computable from what was known at
the *forecast origin* - the last day before the 28-day window opens - not merely
from "some time before t".

The first version of this file honoured that rule but paid too much for it. It
built lags sliding relative to each target day, so `roll_mean_7` at row t covered
sales from t-34 to t-28. On the first day of a window that window ended 27 days
before the origin, meaning the model could not see the 27 most recent days while
`moving_average_28d` used all 28 right up to the origin. Leak-free, but starved.
LightGBM duly lost to the moving average in all three volume terciles by a
near-uniform margin - the signature of a handicap applied evenly to every row.

The fix here: history features are computed once, as of the origin, and the same
values are attached to all 28 target days in that window. What differs per target
day is `h` (1-28, how far ahead we are projecting) plus the calendar and price
facts that are genuinely known in advance for that day. The model now sees
everything the baseline sees, and knows how far out it is guessing.

Blocks are counted backward from the last date, so training origins land on the
same 28-day grid as the backtest windows and training matches scoring exactly.

`year` was also dropped: trees cannot extrapolate, and the test window is always
the most recent period, so year splits fit history that does not generalise
forward.
"""
import numpy as np
import pandas as pd

from forecasting.data import load_store_sales, attach_calendar, attach_prices

HORIZON = 28
ORIGIN_LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 28, 56]

CATEGORICALS = ["item_id", "dept_id", "cat_id", "event_name_1", "event_type_1"]
KNOWN_AT_ORIGIN = ["wday", "month", "snap", "is_event", "h", "sell_price", "price_rel_28"]

# item_id stays in the model despite eating roughly a third of all split gain.
# Removing it was tested: the high-volume tercile improved (MASE 0.902 -> 0.891)
# but low and mid both degraded, because on sparse series a trailing mean is a
# high-variance estimate of level and item identity supplies a stable pooled
# prior. On fast movers the rolling means already pin the level, so identity only
# adds overfitting risk. Net effect across all items is positive, so it stays.


def build_origin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sales history summarised as of each candidate origin date.

    Windows are inclusive of the origin itself, which is the whole point: the
    origin's own sales are known when the forecast is made.
    """
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    grouped = df.groupby("item_id", observed=True)["sales"]

    out = df[["item_id", "date"]].copy()
    for lag in ORIGIN_LAGS:
        out[f"o_lag_{lag}"] = grouped.shift(lag - 1)
    for window in ROLL_WINDOWS:
        out[f"o_roll_mean_{window}"] = grouped.transform(lambda s: s.rolling(window).mean())
        out[f"o_roll_std_{window}"] = grouped.transform(lambda s: s.rolling(window).std())

    return out


def add_horizon_index(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Assign each day its position in a 28-day block and that block's origin.

    Blocks are counted backward from the final date so they coincide exactly with
    the rolling-origin backtest windows.
    """
    dates = np.sort(df["date"].unique())
    position = pd.Series(np.arange(len(dates)), index=dates)

    from_end = (len(dates) - 1) - df["date"].map(position)
    df["h"] = horizon - (from_end % horizon)
    df["origin_date"] = df["date"] - pd.to_timedelta(df["h"], unit="D")
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    trailing = df.groupby("item_id", observed=True)["sell_price"].transform(
        lambda s: s.shift(1).rolling(28).mean()
    )
    df["price_rel_28"] = df["sell_price"] / trailing
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["event_name_1"] = df["event_name_1"].fillna("none")
    df["event_type_1"] = df["event_type_1"].fillna("none")
    df["is_event"] = (df["event_name_1"] != "none").astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    history = [c for c in df.columns if c.startswith("o_")]
    return history + KNOWN_AT_ORIGIN + CATEGORICALS


def build_feature_frame(long_df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """One row per item-day, carrying that day's origin-anchored history
    alongside what is known in advance about the day.

    Training and inference both go through here. That is deliberate: features
    computed one way for training and another for serving is the classic reason a
    model that backtested well produces nonsense in production.

    Rows are dropped where history is incomplete (an item's first ~56 days) or
    the origin predates the data. Rows with a NaN target survive - that is how
    future days to be forecast are represented.
    """
    df = attach_prices(attach_calendar(long_df))
    df = add_calendar_features(df)
    df = add_price_features(df)

    origin = build_origin_features(df)
    df = add_horizon_index(df, horizon)
    df = df.merge(
        origin,
        left_on=["item_id", "origin_date"],
        right_on=["item_id", "date"],
        how="left",
        suffixes=("", "_origin"),
    ).drop(columns=["date_origin"])

    for col in CATEGORICALS:
        df[col] = df[col].astype("category")

    features = feature_columns(df)
    return df.dropna(subset=features).reset_index(drop=True)


def build_features(store: str, n_items: int, horizon: int = HORIZON) -> pd.DataFrame:
    return build_feature_frame(load_store_sales(store, n_items), horizon)
