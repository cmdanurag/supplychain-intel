"""
Session 4 - just look at the data. No modelling yet.

Loads one store's daily sales history from M5, joins in real calendar dates,
and plots a couple of series so you can see the zero-inflation problem with
your own eyes before picking a metric or a model.

Run:
    python -m forecasting.explore
"""
import matplotlib.pyplot as plt

from forecasting.data import DATA_DIR, load_store_sales, attach_dates

OUT_DIR = DATA_DIR / "exploration_plots"

STORE = "CA_1"
N_ITEMS = 400  # approximate; stratified proportionally across departments


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading store {STORE}, first {N_ITEMS} items...")
    long_df = attach_dates(load_store_sales(STORE, N_ITEMS))
    print(f"{len(long_df):,} item-day rows loaded.")

    # ---- The zero-inflation problem ----
    zero_pct = (long_df["sales"] == 0).mean() * 100
    print(f"\n{zero_pct:.1f}% of ALL item-days in this sample have ZERO units sold.")
    print("This is why plain RMSE misleads here, and why the plan flags MASE/WRMSSE.")

    per_item_zero_pct = (
        long_df.groupby("item_id")["sales"]
        .apply(lambda s: (s == 0).mean() * 100)
        .sort_values(ascending=False)
    )
    print("\nSparsest sellers (highest % zero-sales days):")
    print(per_item_zero_pct.head(5).to_string())
    print("\nSteadiest sellers (lowest % zero-sales days):")
    print(per_item_zero_pct.tail(5).to_string())

    # ---- Plot one steady seller against one intermittent one ----
    steadiest_item = per_item_zero_pct.tail(1).index[0]
    sparsest_item = per_item_zero_pct.head(1).index[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, item in zip(axes, [steadiest_item, sparsest_item]):
        series = long_df[long_df["item_id"] == item].sort_values("date")
        ax.plot(series["date"], series["sales"])
        ax.set_title(f"{item} — {per_item_zero_pct[item]:.1f}% zero-sales days")
        ax.set_ylabel("units sold")
    plt.tight_layout()

    out_path = OUT_DIR / "sample_series.png"
    plt.savefig(out_path)
    print(f"\nSaved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
