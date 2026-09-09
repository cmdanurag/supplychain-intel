# Engineering log

Every problem hit, what actually caused it, and how it was fixed. Kept in the
order it happened. Open items are marked as such rather than quietly dropped.

---

## Phase 0 — getting the skeleton running

### 1. The virtualenv kept deactivating itself

**Symptom.** Activating `venv\Scripts\Activate.ps1` appeared to work, then the
environment silently deactivated a moment later, and again on every `cd`.

**Cause.** A custom `prompt` function in the PowerShell profile auto-activated a
venv per directory, but only ever looked for `.venv`. This project's folder is
named `venv`, so the lookup missed, and the `elif` branch — "we have
`$env:VIRTUAL_ENV` set but no venv here" — ran `deactivate` on every prompt
redraw.

**Fix.** Profile now checks both `.venv` and `venv`, and only deactivates when
neither exists and a `deactivate` function is actually defined.

**Worth knowing.** Environment problems that look like tooling bugs are often
local shell configuration. The symptom (venv "randomly" dropping) pointed
nowhere near the cause (a prompt-render hook).

---

### 2. Four cascading dependency failures on Python 3.14

All four share one root cause: **Python 3.14 is newer than the pins in the
original `requirements.txt`**, and a package only ships prebuilt wheels for
Python versions that existed when it was released. No wheel means pip falls back
to compiling from source, which needs a toolchain this machine doesn't have.

| # | Package | Symptom | Fix |
|---|---|---|---|
| 2a | `psycopg-binary==3.2.3` | `Could not find a version that satisfies the requirement` | → `3.2.13` |
| 2b | `pydantic==2.10.4` | `pydantic-core` tried to build from Rust source; `error: linker link.exe not found` | → `2.13.5` |
| 2c | `sqlalchemy==2.0.36` | `TypeError: descriptor '__getitem__' requires a 'typing.Union' object` when declaring `Mapped[str \| None]` | → `2.0.52` |
| 2d | `altair` 5.5.0 | `TypeError: _TypedDictMeta.__new__() got an unexpected keyword argument 'closed'` | → pinned `6.2.2` |

Two of these had a second-order twist worth recording:

- **2a's real damage was invisible.** pip resolves *all* packages before
  installing *any*, so one unsatisfiable pin meant nothing at all got installed —
  which surfaced as the misleading `uvicorn: The term 'uvicorn' is not
  recognized`. The missing command was a symptom of a failed resolve, not a PATH
  problem.
- **2d needed three separate changes.** `altair` was capped by
  `streamlit==1.41.1` at `<6`, so upgrading Streamlit came first — but
  `streamlit==1.62.0` requires `starlette>=0.46`, which collided with
  `fastapi==0.115.6`'s `starlette<0.42`, so FastAPI went to `0.141.1` too. Even
  then `altair` stayed at 5.5.0, because **pip's default upgrade strategy only
  touches packages named in the requirements file**, not transitive
  dependencies that still satisfy their constraint. It needed an explicit pin.

**Worth knowing.** Being on a bleeding-edge interpreter is a real cost, paid in
exactly this kind of debugging. The alternative — installing Python 3.12
alongside — was considered and rejected once the pins were working, but remains
the better call if ML libraries fight back at scale.

---

## Stage 3A — forecasting

### 3. The first data sample was pathologically unrepresentative

**Symptom.** Exploration reported **98.8% zero-sales days**, far above M5's
roughly 68% average.

**Cause.** Items were taken in file order (`.iloc[:300]`). M5 is sorted by item
ID, so the first 300 items of a store are all `FOODS_1` — a small department of
low-volume specialty goods.

**Fix.** `load_store_sales` now samples stratified by department, proportional to
each department's real size, with a fixed seed for reproducibility. Result: 401
items across all 7 departments, **65.0% zeros**, which lines up with the dataset
average.

**Worth knowing.** The bug produced *plausible* output — sparse retail data is
supposed to look sparse. It was only caught by comparing against a known
published property of the dataset. Convenience slicing is not sampling.

---

### 4. A prediction that turned out wrong: seasonal-naive did not overtake

Before re-running on the stratified sample, the expectation stated was that
seasonal-naive would likely overtake moving-average, since real weekly
seasonality exists in `FOODS_3` fast movers that the `FOODS_1`-only slice lacked.

**It didn't.** Moving-average still won on both metrics (RMSE 1.286 vs 1.656,
MASE 1.381 vs 1.568).

**Why.** Seasonal-naive copies a single noisy week and repeats it four times
across a 28-day horizon, carrying that one week's random spikes into every
prediction. Moving-average shrinks toward the mean, and both RMSE and MAE reward
predicting the conditional mean on low-count data. Per-item day-of-week signal is
weak relative to count noise; it is far stronger at store and department
aggregate level.

**Worth knowing.** This is the argument for a *pooled* model — learn calendar
effects across the whole panel at once, rather than from seven noisy
observations per series.

---

### 5. LightGBM lost to a moving average — feature design, not tuning

**Symptom.** First LightGBM run scored RMSE 1.300 / MASE 1.417 against
moving-average's 1.286 / 1.381. Feature importances showed `item_id` dominating
at roughly 4x the next feature.

**Cause — the main one.** Lags were built *sliding relative to each target day*
rather than *anchored at the forecast origin*. `roll_mean_7` at row `t` averaged
sales from `t-34` to `t-28`. For the first day of a 28-day test window, that
window ends 27 days before the origin — so the model could not see the 27 most
recent days of history, while `moving_average_28d` used all 28 days right up to
the origin. Leak-free, but structurally starved of the information that best
predicts early-horizon days.

**Contributing causes.**
- `year` as a numeric feature: trees cannot extrapolate, and the test window is
  always the most recent period, so year splits fit history that does not
  generalise forward.
- `item_id` with 401 categories absorbed most of the split budget re-learning
  per-item level information that `roll_mean_*` already encodes.
- Averaging scores per item weights a near-zero seller identically to a fast
  mover. Most items in M5 are sparse, and on those a smoothed mean is close to
  optimal, so ML gains get averaged away. The official WRMSSE weights by dollar
  sales precisely because that is where accuracy has business value.

**Diagnostic first, and one hypothesis died.** Before rewriting anything, results
were broken down by item volume tercile, to test the idea that LightGBM might
already be winning on fast movers with the equal-weighted average hiding it.

| Tercile (mean daily units) | LightGBM RMSE | Moving average RMSE |
|---|---|---|
| low (0.18) | 0.581 | **0.575** |
| mid (0.57) | 1.048 | **1.042** |
| high (2.64) | 2.270 | **2.239** |

**It lost in all three.** The metric was not hiding a win, so that hypothesis is
dead. What the breakdown did show is that the gap is near-uniform, roughly
0.6-1.4% everywhere — the signature of a handicap applied evenly to every row,
rather than a model failing on some particular segment. That is consistent with
an information deficit and inconsistent with, say, sparse items poisoning the
average.

**Fix applied.** `features.py` rewritten to origin-anchored direct multi-step:
history features are computed once as of the origin and shared by all 28 target
days, while `h` (days ahead) plus the target day's calendar and price facts vary
per row. Blocks are counted backward from the final date so training origins land
on the same grid as the backtest windows. `year` was dropped at the same time.

**Result: the gap closed and reversed on RMSE.**

| | LightGBM before | LightGBM after | Moving average |
|---|---|---|---|
| RMSE (overall) | 1.300 | **1.259** | 1.286 |
| MASE (overall) | 1.417 | 1.392 | **1.381** |
| RMSE high tercile | 2.270 | **2.173** | 2.239 |
| MASE high tercile | 0.939 | **0.902** | 0.925 |

`wday` rose to the second-most-important feature and `h` ranked seventh — both
unreachable under the old design, which is corroboration that the diagnosis was
right rather than a lucky fix.

---

### 5b. Remaining loss is confined to sparse items

**Symptom.** Post-fix, LightGBM wins RMSE overall and wins *both* metrics on the
mid and high terciles, but still loses overall MASE (1.392 vs 1.381). The entire
deficit sits in the low-volume tercile: 2.021 vs 1.951.

**Cause.** Two compounding effects. Sparse series have small MASE denominators,
so any deviation is amplified; and on items averaging 0.18 units/day a shrunk
constant is close to optimal, leaving little for a model to learn. Equal-weighting
items then lets that one tercile outweigh gains in the other two.

Note this does *not* revive the hypothesis killed in §5. Before the fix the model
lost in every tercile, so no weighting could have rescued it. After the fix it
wins where the volume is. The fix changed the fact, not the standard.

**Attempted fix — did not work, reverted.** Croston-style intermittency features
were added at origin: `o_days_since_sale`, `o_nonzero_rate_28/56`, and
`o_demand_size_56` (mean size of a sale given one happened). Classical
intermittent-demand practice decomposes demand into frequency and size rather
than modelling a single mean, which lands between the two and matches neither.

Results got marginally *worse*: overall RMSE 1.259 → 1.261, MASE 1.392 → 1.400,
and the low tercile it was aimed at went 2.021 → 2.046.

The instructive part is that `o_days_since_sale` ranked **second by split gain**
and `o_demand_size_56` tenth. The model leaned on them hard and predicted no
better — high in-sample gain with zero holdout gain is the signature of a feature
that fits without generalising. The rolling means already carried the
information; the extra columns only supplied more ways to overfit. Reverted on
parsimony: simpler model, better score.

**Standing conclusion.** Near-zero series at a 28-day horizon appear to carry
very little learnable structure, and a shrunk constant is close to the best
answer available. This is a property of the data, not a defect to engineer
around, and it belongs in the README as a stated limitation.

---

### 7. `item_id` as a model feature

**Symptom.** Across every run, `item_id` consumed roughly a third of all split
gain (4441, then 3229, then 3170) — far above any behavioural feature.

**Cause.** 401 categories is where LightGBM's categorical splitting starts to
overfit, and the origin-anchored rolling means already encode what the model was
using `item_id` for: each item's typical level.

**Change tested.** `item_id` excluded from the feature set, still cast to
category for grouping and merging. Because the reverted configuration had a known
score (RMSE 1.259 / MASE 1.392), the revert and this change could be measured in
one run without confounding the attribution.

**Result: split, and reverted.** The high tercile improved to its best recorded
MASE (0.902 → 0.891), but low and mid both degraded, taking overall MASE from
1.392 to 1.422.

The mechanism is coherent rather than noise. On sparse series a trailing mean is
a high-variance estimate of level, so item identity supplies a stable pooled
prior and genuinely helps. On fast movers the rolling means already pin the level
precisely, so identity contributes nothing but overfitting risk. Net effect
across all items is positive, so `item_id` stays — with the caveat that a model
intended to generalise to *unseen* items would have to drop it, and would lose
accuracy on sparse items in exchange.

One useful side effect: with `item_id` removed, the importance ranking became
legible — `sell_price` first, then the rolling statistics, `month`, `h`, `wday`.
Price being the dominant driver is what retail intuition predicts, and it had
been buried under item identity the whole time.

---

### 8. A run that silently executed stale code

**Symptom.** After restoring `item_id` and adding the WRMSSE column, a re-run
produced output byte-identical to the previous run — no WRMSSE column, no
`item_id` in the importances.

**Cause.** Cached bytecode. The source on disk was correct, which was confirmed
by grepping the files rather than trusting the output.

**Fix.** Cleared `forecasting/__pycache__` and re-ran.

**Worth knowing.** The tell was that results were *identical*, not merely similar.
Genuine changes to a deterministic pipeline essentially never reproduce to three
decimal places. Verifying the file on disk took one command and settled it; the
alternative was debugging a model that was never running.

---

### 9. Adding a weighted metric without moving the goalposts

WRMSSE was introduced *after* LightGBM had already lost equal-weighted MASE
twice, which is exactly when a weighted metric looks like special pleading.

Three things keep it honest. It was specified in the project plan from the start,
before any results existed. RMSE and MASE remain in the table, unchanged and
unhidden, including the one metric LightGBM still loses. And the tercile
breakdown is published alongside, so the reader can see precisely where the model
wins and where it does not, rather than taking a single aggregate on trust.

Weights are computed from the 28 days preceding the *earliest* evaluation window,
so they are out-of-sample for every window scored.

---

### 10. Prediction intervals, and a second prediction that was wrong

Two extra LightGBM models per window at the 10th and 90th percentiles
(`objective="quantile"`). Coverage came out at **80.3% against an 80% nominal
target**, without tuning.

The stated expectation beforehand was that intervals on the sparse tercile would
be "near-useless" — degenerate bounds of roughly 0 to 0 or 0 to 1. That was
wrong. Those intervals are the *best* calibrated of the three at 81.7%.

What is true is a different thing, visible only after dividing width by mean
daily demand:

| Tercile | Coverage | Width | Width ÷ mean demand |
|---|---|---|---|
| low | 81.7% | 1.14 | 6.3x |
| mid | 80.7% | 2.08 | 3.7x |
| high | 78.3% | 5.05 | 1.9x |

Relative uncertainty falls as volume rises. That is not a modelling defect, it is
the reason slow-moving items tie up disproportionate safety-stock capital in real
inventory systems, and it is a direct input to Stage 3B's cost model.

**Open item.** The high tercile is *under* nominal at 78.3%. On the highest-revenue
items that is the wrong direction to err, because stockout cost exceeds holding
cost. Widening that segment to 0.05/0.95 is the obvious candidate, untested so far.

---

### 11. The deployed model nearly shipped 4x more trees than the one measured

**Symptom.** `fit.py` printed `fitted point: 1000 trees`. During backtesting,
early stopping had chosen 149, 278, 183, 46, 211 and 543 trees across the six
windows — median 197.

**Cause.** The final models are deliberately fitted without an early-stopping
holdout, since withholding data would make the shipped artifact worse than the
one that was validated. The code comment asserted that tree counts "were already
established by the backtest" — but nothing actually passed them through, so
training simply ran to `n_estimators=1000`.

The result would have been an artifact roughly five times deeper than anything
that had ever been evaluated, quoting backtest numbers in its own metadata that
described a materially different model. Nothing would have failed; the API would
have returned confident, slightly worse forecasts indefinitely.

**Fix.** `train.py` records `best_iteration_` per model per window and writes the
median to `backtest_metrics.json`. `fit.py` reads those counts and refuses to run
if they are absent, rather than falling back to a default. A missing artifact is
recoverable; a plausible-looking wrong one is not.

**Worth knowing.** The comment describing the intended behaviour was written
before the behaviour was, and then read as if it were true. The only reason it
surfaced was printing the tree count at fit time — an assertion in the output
that could visibly disagree with expectation.

---

### 12. Two bugs found by testing the serving path

A 26-point verification pass over the deployed forecast path found two real
defects that the implementation review had missed.

**The request schema defaulted to an item the model cannot serve.**
`RunRequest.item_ids` defaulted to `FOODS_3_090`, left over from the placeholder
era. That item is not among the 401 the model was fitted on, so every bare POST —
including Swagger's "Try it out" button, the first thing anyone clicks — failed
immediately. Fixed to a valid item, with `GET /api/catalog` documented in the
schema as the source of truth. The deeper point is that a hardcoded identifier
drifted out of sync with the artifact the moment the artifact changed; the
catalog endpoint exists precisely so the frontend does not repeat that mistake.

**Quantile crossing is real, at 0.05%.** The three models are trained
independently, so nothing enforces `lower <= point <= upper`. Spot-checking one
item's 28 rows found none. Checking all 11,228 rows found **6 crossings**, every
one of them `point > upper`, by margins under 0.01 units.

Fixed by widening the interval to contain the point, not by clipping the point
into the interval. The point forecast is the number WRMSSE was measured on, and
serving something different from what was evaluated is not a trade worth making
to resolve a rounding-scale inconsistency.

**Worth knowing.** A 28-row sample was too small to observe a 0.05% event and
returned a clean bill of health. The rate only became visible by checking the
whole snapshot, which cost one command. When the expected frequency of a defect
is unknown, sample size is the difference between "verified" and "not yet seen".

**Worth knowing.** The instinct on losing to a trivial baseline is to tune
hyperparameters. The actual cause was upstream, in what the features could see —
tuning would have burned hours and closed none of the gap. Worth noting too that
the diagnostic which disproved the metric hypothesis cost about ten lines, and
would have been just as valuable had it come out the other way.

---

### 6. Removed a hardcoding trap before it bit

Baseline numbers were briefly hardcoded into `train.py`'s comparison table for
display. Any change to the sample would have silently made that table lie.
`baselines.score_all_items()` now returns per-item scores, and `train.py` calls
it, so both scripts compute from the same source.
