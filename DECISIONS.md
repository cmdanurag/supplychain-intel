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

---

## Stage 3B — multi-echelon inventory optimisation

### 13. The window the model serves has no ground truth

**Symptom.** The plan for 3B was to score inventory policies on the 28 days the
deployed forecaster predicts (2016-05-23 to 2016-06-19). Building the actuals
artifact failed outright: `Usecols do not match columns, columns expected but
not found: ['d_1942', ... 'd_1969']`.

**Cause.** `sales_train_evaluation.csv` ends at `d_1941` = 2016-05-22, which is
exactly `metadata.trained_through`. The served forecast window *is* M5's
held-back competition future. Nothing in this repo contains its answers, and
that is by design — the file is the training half of the competition.

**Why this mattered more than a missing file.** The tempting workaround is to
score policies against the forecast itself. That measures nothing: the policies
are *built* from that forecast, so the optimiser would be marked against its own
assumptions and would look flawless regardless of quality.

**Fix.** Benchmark one window earlier. Stage 3A's most recent rolling-origin
window (2016-04-25 to 2016-05-22) has both halves — a model trained strictly
before it yields genuine out-of-sample forecasts, and the actual sales are in
the file. `optimisation/evalset.py` rebuilds that fold, reusing train.py's
window logic, features, parameters and early-stopping split rather than
re-declaring them, and writes forecasts and actuals side by side.

The API keeps both windows and labels each: `optimise` plans the served window
and reports *projected* cost, `simulate` back-tests the earlier window and
reports *measured* cost. Every payload carries its own `window` and `evaluation`
field, because reporting a projected number as a measured one is the most
misleading thing this service could do.

**Worth knowing.** "Which of my numbers has ground truth behind it" is worth
asking before building the thing that produces them, not after. The failure here
was cheap because it surfaced as a missing column; had the dataset happened to
carry 28 more days of *something*, it would have surfaced as a suspiciously
excellent benchmark.

---

### 14. OR-Tools, and Python 3.14 again

The plan pinned `ortools==9.11.4210`. `pip index versions ortools` on this
interpreter lists exactly one release: **9.15.6755**, the only one shipping a
`cp314` wheel. Same root cause as entry 2 — a bleeding-edge interpreter — and
the same fix, checked *before* writing code against the library rather than
after. It pulls `protobuf` from 5.29.6 to 6.33.6; `pip check` reports no broken
requirements, and CP-SAT solves a two-variable test model, so the bump is clean.

`simpy`, also named in the plan, was dropped rather than pinned. See entry 20.

---

### 15. The solver and the simulator disagreed about shelf space

**Symptom.** The perfect-information run — the same CP-SAT controller fed the
*actual* demand it would face — came back at **73.9% fill rate**. Knowing the
future exactly, it still left a quarter of demand unserved.

**Cause.** Two different readings of one constraint. CP-SAT capped shelf space on
end-of-day inventory, after that day's sales. The simulator, which has to decide
shipments before it knows the day's sales, capped it on inventory *position* —
on hand plus already in transit. So the solver legitimately planned shipments
that the simulator then clipped on arrival, and the plans were silently degraded
on replay. Nothing errored; the policy just underperformed.

**Fix.** The solver now constrains the same quantity the simulator does:
`ist[i,t] + in-transit + inherited inbound <= shelf_capacity`. It is the more
conservative of the two readings, and it is the correct one, because no policy
can ship against sales it has not observed yet.

**Worth knowing.** When a model and its simulator are written separately, the
cost accounting gets checked and the *constraint semantics* do not. This one was
only visible because a policy with perfect information was in the benchmark —
without that row there is no result obviously absurd enough to investigate. A
reference case whose answer you already know is worth its runtime.

---

### 16. Infeasible re-solves returned an empty plan

**Symptom.** The first full benchmark had the CP-SAT policy at **34.2% fill rate
and $3,832 cost**, against baselines at 88% and $2,068 — dramatically worse than
the rules it was supposed to beat. Solver statuses: `FEASIBLE, INFEASIBLE,
INFEASIBLE, INFEASIBLE`.

**Cause.** Two compounding faults. The hard 95% aggregate service constraint is
genuinely unsatisfiable on a mid-window re-solve: the first two days of any
horizon can only be served from stock already on the shelf, because a shipment
takes `store_lead_days` to arrive, so a low shelf makes the target arithmetically
out of reach. That part is the model correctly reporting an impossible request.
The fault was the response — `solve()` returned a plan of all zeros, so the
policy ordered nothing for 21 of 28 days.

**Fix.** A relaxation ladder. Constraints come off in a defined order — per-item
service floor, then the aggregate service level, leaving stockout penalties to
drive availability — and every relaxation is recorded on the `Plan`, surfaced
through the API and shown in the UI. With it, all four solves return OPTIMAL and
the same instance lands at 84.5% fill and $1,549.

Stage 3C needs exactly this contract when an agent proposes an impossible
scenario: say what had to give, do not crash and do not quietly answer a
different question.

**Worth knowing.** "Infeasible" is a legitimate answer to a badly-posed
question, and a solver saying it is not a bug. Returning zeros *as though it were
a plan* is the bug. The failure mode was a silent 50-point drop in a metric, not
an exception.

---

### 17. Today's delivery counted twice

Inventory position — on hand plus on order — is what every reorder rule compares
against its reorder point. The simulator added each day's arrival to on-hand
stock but left it sitting in the inbound pipeline array, so `in_transit` counted
it again. Every position was inflated by one day's inbound, which suppresses
reordering across every policy at once.

Fixed by clearing the pipeline slot as it is consumed. Worth recording because
of how it presents: not as an error, but as slightly worse service than expected
across the board — the kind of thing that gets attributed to the demand data.

---

### 18. A daily-review baseline would have been a strawman

**Symptom.** Early runs showed the optimiser winning by a wide margin, with the
gap concentrated almost entirely in fixed delivery cost.

**Cause.** A delivery costs $60 whatever rides on it. The classical policies
reviewed stock every day, so they paid up to 28 delivery charges over the window
while CP-SAT consolidated onto 7 truck days. Nearly the whole "win" was that
cadence difference — and no real firm reviewing daily would dispatch daily,
precisely because deliveries cost money.

**Fix.** Every classical policy now takes a `review_days`, with the protection
period widened to lead time plus review interval as the textbook (R, S) rule
requires. The benchmark grid-searches it over {1, 2, 3, 4, 7, 14} and keeps each
baseline's **cheapest** configuration that still meets the fill rate our policy
achieved, bisecting the safety factor within each. Baselines get their best shot
deliberately.

On the 60-item instance the baselines choose review periods of 2-3 days, and the
optimiser's margin lands at 24.6% rather than the inflated figure.

**Worth knowing.** A number won against a hobbled baseline is worse than no
number, because it will not survive the first question about it. The tuning
harness here exists to attack our own result, and it cost more code than the
optimiser's objective function.

---

### 19. 28 days is intractable; 14 is not

Solve quality against horizon and instance size, 20s limit, measured:

| Items | Horizon | Status | Optimality gap |
|---|---|---|---|
| 10 | 14 | OPTIMAL | 0% |
| 10 | 28 | FEASIBLE | 16% |
| 30 | 14 | FEASIBLE | 7% |
| 30 | 28 | FEASIBLE | 42% |
| 60 | 14 | FEASIBLE | 29% |
| 60 | 28 | FEASIBLE | 55% |

This is a capacitated joint-replenishment problem — NP-hard, with the difficulty
living in the binaries deciding which days a truck runs and which items are on
it. The fix is standard model-predictive control: **plan 14 days, commit 7**.
Everything past the commit window is re-decided anyway, so solving it to
optimality is largely wasted effort; planning past the commit window still
matters, because it is what stops the solver emptying the shelf on day 7.

Two other changes bought real headroom. Upper bounds on order and shipment
quantities are now derived from what an item can physically absorb — a shipment
can never usefully exceed the shelf plus what sells in transit — rather than from
horizon totals. And a valid inequality gives the linear relaxation a floor on
delivery days: everything sold beyond current stock must arrive on some truck,
and a truck holds at most `truck_capacity`. It is implied by the service
constraint, so it is only added while that constraint is present.

**Honest caveat, carried into the README.** At 60 items the committed solves
still leave a mean 23% optimality gap. The reported saving is therefore a
*lower* bound on what this formulation can achieve — a better solve can only
improve it — but our policy is not optimal, and saying otherwise would be wrong.

---

### 20. Why there is no SimPy, and where the optimisation problem actually lives

The plan named SimPy for the simulator. It is not used. SimPy earns its keep when
events arrive asynchronously and entities queue for contended resources.
Inventory review here is periodic and daily, every quantity is known at the start
of the day, and there is no contention to resolve — so a SimPy version would wrap
the same arithmetic in a process-and-event API, add a dependency to the deployed
image, and make the cost accounting harder to audit. `simulation/simulate.py` is
about sixty lines and is the whole model.

The more interesting discovery is what makes this a constraint program at all.
Grocery holding cost is minute — a $1.68 item at a 25% annual rate carries for
about **0.08 cents per unit per day**, against a stockout costing over a dollar.
On those numbers alone the optimal policy is "hold everything", and there is no
problem to solve. What creates one is the structure a per-item reorder rule
cannot see:

- a fixed cost per purchase-order line, so replenishing has a setup price;
- a fixed cost per delivery day, shared by every item on the truck, so the
  question is not how much of item X to ship but *which items ride together*;
- DC storage, shelf space and truck capacity, all shared and all finite.

The measured cost breakdown confirms it. Holding cost is $50 of a $5,435 total;
the optimiser's entire advantage is consolidation — **109 purchase-order lines
against the baseline's 247, and 9 delivery days against 13** — at a comparable
stockout bill. That is the joint-replenishment story, and it is worth being able
to state plainly, because "we saved 25% on inventory cost" invites the assumption
that it came from carrying less stock. It did not.

---

### 21. Perfect information is not a lower bound

The benchmark includes the same rolling-horizon controller fed actual demand
instead of forecasts, to price the value of information. On the 60-item instance
it lands 10.4% below our policy, as expected. On a 3-item instance our policy
came in **9.4% below it**.

That is not a bug and the report says so explicitly. The controller is myopic by
construction — it plans 14 days and commits 7 — so perfect knowledge inside the
planning window does not guarantee a cheaper outcome over the full horizon, and a
forecast error that happens to favour the committed decisions can beat it. A
single all-knowing 28-day solve would be a tighter bound in principle, but at 28
days the solver leaves a 40%+ gap (entry 19), so that "bound" would measure
solver difficulty rather than the value of information.

It is therefore labelled a reference, not a bound, and the caveat travels with
the number in the JSON rather than living only here.

---

### 22. What the service-level target does and does not promise

The optimiser is given a 95% aggregate fill-rate target and meets it *in plan*,
on every committed solve. Realised fill on the 60-item instance is **80.2%**.

The gap is not a constraint violation — it is forecast error. The LightGBM point
forecast under-predicts this window by about 5% in total units (14,214 forecast
against 14,945 actual), so a plan that serves 95% of forecast demand serves less
than that of real demand, and only the first 7 days of each 14-day plan are
committed before re-solving. The comparison is unaffected, because every baseline
is tuned to the fill rate our policy *achieved* rather than the one it targeted.

Two honest consequences, both in the README's limitations: the service target is
a planning target, not a delivered guarantee; and planning against an upper
quantile of the predictive distribution instead of the point forecast is the
obvious next lever, since the 80% interval is already calibrated (76.0% measured
coverage on this window, 80.3% on the 3A backtest).

---

### 23. The headline number moved 24% → 50% on an identical re-run

**Symptom.** The sensitivity sweep re-runs the whole benchmark at several
stockout multipliers, and `x2.0` *is* the default configuration — the same
instance the main run had just reported. The main run said **24.4% cost
reduction**. The sweep said **49.9%**. Same items, same window, same costs,
same code.

**First diagnosis, which was wrong.** CP-SAT searches on eight threads under a
wall-clock limit and does not prove optimality at 60 items (entry 19), so the
obvious culprit was solver nondeterminism, amplified by a protocol that tunes the
baselines to whatever fill rate our plan happened to achieve. That story is
plausible, it is partly true, and it accounts for almost none of the variance. It
was written into this file before it was checked.

**What the measurement actually showed.** Running the benchmark six times:

| Run | Our cost | Our fill | Reported reduction |
|---|---|---|---|
| 1 | \$5,457.85 | 80.5% | 24.3% |
| 2 | \$5,281.95 | 80.3% | 26.7% |
| 3 | \$5,448.35 | 81.7% | **47.7%** |
| 4 | \$5,211.73 | 82.3% | **50.0%** |
| 5 | \$5,227.05 | 81.3% | **49.9%** |
| 6 | \$5,341.56 | 79.7% | 25.9% |

Our own cost is stable to ±2.3% and our fill rate to 2.6 points. The solver was
never the problem. The number moving was the **baseline**, jumping between about
\$7,200 and \$10,400 — and the baselines contain no randomness at all.

**Actual cause.** `tune_to_service` bisected on the safety factor `z`, which
requires aggregate fill rate to be monotone in `z`. For one item in isolation it
is. Here it is not: the DC, the shelf and the truck are shared, so raising one
item's target makes it claim more of a fixed truck load, and the simulator's
proportional rescaling takes that capacity away from other items, whose fill
drops. Aggregate fill can therefore *fall* as `z` rises. The bisection then
converged on whichever branch it happened to bracket — sometimes an expensive
high-`z` configuration, while a configuration at `z` near zero met the same bar
for \$3,000 less. Which branch it found depended on the exact bar, which moved
with our achieved fill. Hence the bimodal result.

**Fix.** An exhaustive grid over both parameters — `z` from 0 to 3.0 in steps of
0.125, crossed with the review grid. A few hundred simulations per policy, all of
them cheap, and it cannot miss the cheap configuration. The tuner no longer
assumes a shape the system does not have.

**Worth knowing — two things, and the second is the real one.**

A near-miss on the number: a single run had already been written into the README
as "24.6%", and it was caught only because a sensitivity sweep happened to re-run
the default configuration and disagree with itself. An evaluation harness that
re-runs one configuration and compares is worth building deliberately, not by
accident, around any number destined for a CV.

The more uncomfortable lesson is about the first diagnosis. "Nondeterministic
multi-threaded solver" was the available explanation, it fit the symptom, and it
was wrong — and because it was wrong in a *sophisticated* direction it was
convincing enough to write down. What disproved it cost one table: our own cost
next to the reported reduction, run by run. The moment those two columns sat side
by side, a stable numerator against a bimodal ratio pointed straight at the
denominator. Reaching for the explanation before the disaggregation is how
plausible causes get mistaken for real ones, and the bisection would have
survived indefinitely underneath a caveat about solver noise.
