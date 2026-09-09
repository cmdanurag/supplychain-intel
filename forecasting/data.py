"""
Shared M5 loading code. Used by explore.py, baselines.py, and (later) train.py
so the same store/item slice and date-joining logic isn't duplicated.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_store_sales(store_id: str, n_items: int, seed: int = 42) -> pd.DataFrame:
    """Load a department-stratified sample of one store's items, reshaped from
    wide (one column per day) to long (one row per item-day).

    Sampling is proportional to each department's size, so the slice mirrors the
    store's real composition. Taking items in file order instead lands you in
    FOODS_1 only, which is both unrepresentative and far sparser (98.8% zero-sales
    days) than the dataset average, and would tune feature choices against a
    pathological case. `n_items` is approximate - per-department rounding moves it
    by a few either way. The seed is fixed so README numbers stay reproducible.
    """
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    store = sales[sales["store_id"] == store_id]
    sampled = store.groupby("dept_id").sample(
        frac=n_items / len(store), random_state=seed
    )

    day_cols = [c for c in sampled.columns if c.startswith("d_")]
    long_df = sampled.melt(
        id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    return long_df


def attach_dates(long_df: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", usecols=["d", "date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    return long_df.merge(calendar, on="d", how="left")


def append_future_days(long_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Append `horizon` target rows per item for the days after sales history ends.

    M5's calendar and price files deliberately run 28 days past the last day of
    sales - that gap is the competition's forecast window - so every feature the
    model needs on those days exists except the target itself, which is what we
    are predicting. Sales are left NaN to mark them.
    """
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", usecols=["d"])
    known = set(long_df["d"].unique())
    future_days = [d for d in calendar["d"] if d not in known][:horizon]

    keys = long_df.drop_duplicates("item_id")[
        ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    ]
    future = keys.merge(pd.DataFrame({"d": future_days}), how="cross")
    future["sales"] = np.nan
    return pd.concat([long_df, future], ignore_index=True)


CALENDAR_COLS = [
    "d", "date", "wm_yr_wk", "wday", "month", "year",
    "event_name_1", "event_type_1",
    "snap_CA", "snap_TX", "snap_WI",
]


def attach_calendar(long_df: pd.DataFrame) -> pd.DataFrame:
    """Join dates, weekday, month, events and SNAP flags.

    SNAP (food assistance) payout days differ by state and drive visible demand
    spikes on eligible items, so each row keeps only its own state's flag.
    """
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", usecols=CALENDAR_COLS)
    calendar["date"] = pd.to_datetime(calendar["date"])
    df = long_df.merge(calendar, on="d", how="left")

    df["snap"] = 0
    for state in ("CA", "TX", "WI"):
        mask = df["state_id"] == state
        df.loc[mask, "snap"] = df.loc[mask, f"snap_{state}"]

    return df.drop(columns=["snap_CA", "snap_TX", "snap_WI"])


def attach_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Join weekly sell prices on (store, item, week).

    Using the current week's price as a feature is not leakage: M5 ships prices
    covering the forecast period too, because retailers set them in advance.
    Rows are NaN before an item starts selling in a store - it did not exist to
    forecast yet, so build_features drops them.
    """
    prices = pd.read_csv(
        DATA_DIR / "sell_prices.csv",
        dtype={"store_id": "category", "item_id": "category", "sell_price": "float32"},
    )
    prices = prices[prices["store_id"].isin(df["store_id"].unique())]
    prices["store_id"] = prices["store_id"].astype(str)
    prices["item_id"] = prices["item_id"].astype(str)

    return df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
