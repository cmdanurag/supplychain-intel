# Supply Chain Intelligence Platform

Demand forecasting → inventory optimisation → scenario analysis, deployed as an async service.

> **Status: Stages 3A and 3B complete, not yet deployed.** Forecasting is a real
> LightGBM model with calibrated quantile intervals, serving in ~20ms from
> precomputed features. Optimisation is a real OR-Tools CP-SAT multi-echelon
> model, re-solved on a rolling horizon and benchmarked against three classical
> inventory policies on held-out demand — **25.6% cheaper than a tuned fixed
> reorder-point policy at matched service level** (range 23.2–28.6% over 6 runs).
> Only `explain` (Stage 3C) is still a placeholder. Nothing is on a public URL
> yet — see [Roadmap](#roadmap).

---

## Why the architecture looks like this

A forecast takes seconds. A constraint solve takes minutes. Render, Railway, and most
hosts kill an HTTP request after 30–60 seconds, so a synchronous `POST /run` that waits
for the pipeline **cannot work in production**.

So:

```
POST /api/runs        -> writes a row, schedules a background job,
                         returns {run_id} in <200ms with HTTP 202
GET  /api/runs/{id}   -> client polls this until status is
                         succeeded or failed; partial results
                         are returned while still running
```

This is the pattern every serious ML product uses. It is the most transferable thing in
this repo.

---

## Run it locally

```bash
git clone <your-repo-url>
cd supply-chain-intel

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# requirements.txt is the API + serving deps (what the Docker image installs).
# requirements-dev.txt adds training, plotting and the frontend.
pip install -r requirements-dev.txt

cp .env.example .env

# Build the model artifacts the service loads. Both are offline steps; the API
# never touches the raw CSVs.
python -m forecasting.train        # rolling-origin backtest -> metrics + tree counts
python -m forecasting.fit          # Stage 3A: deployed models + inference snapshot
python -m optimisation.evalset     # Stage 3B: held-out forecasts + actuals + demand stats
python -m optimisation.benchmark   # Stage 3B: the policy comparison (~3 min)

# Invariant checks for Stage 3B. The solver and the simulator are two separate
# models of the same system; this asserts they agree to the cent on a plan's
# cost, and that no policy escapes a capacity constraint. Run after touching
# either. Exits non-zero on failure.
python -m optimisation.verify

# Terminal 1 — API
uvicorn app.main:app --reload

# Terminal 2 — frontend
streamlit run frontend/streamlit_app.py
```

API docs at http://localhost:8000/docs — FastAPI generates them from the Pydantic
schemas, and they are a legitimate thing to screenshot for your README.

### Try it without the frontend

```bash
RUN_ID=$(curl -s -X POST http://localhost:8000/api/runs \
  -H "Content-Type: application/json" \
  -d '{"store_ids":["CA_1"],"item_ids":["FOODS_1_011"],"horizon_days":28}' \
  | python -c "import sys,json;print(json.load(sys.stdin)['run_id'])")

# Poll a few times and watch the stage change
curl -s http://localhost:8000/api/runs/$RUN_ID | python -m json.tool
```

---

## Deploy (do this on day one, before any modelling)

**Backend — Render**

1. Push this repo to GitHub.
2. Render → New → Web Service → connect the repo.
3. Runtime: **Docker**. Render detects the `Dockerfile` automatically.
4. Environment variables: `DATABASE_URL`, `ALLOWED_ORIGINS`.
5. Health check path: `/health`.

**Database — Neon**

1. Create a free Postgres project at neon.tech.
2. Copy the connection string, convert the prefix to `postgresql+psycopg://`.
3. Paste it into Render as `DATABASE_URL`.

**Frontend — Streamlit Community Cloud**

1. share.streamlit.io → deploy from the same repo.
2. Main file: `frontend/streamlit_app.py`.
3. Secrets: `API_URL = "https://your-service.onrender.com"`.

Once both URLs load, Phase 0 is done. Everything after this is filling in logic.

---

## Layout

```
app/
  main.py      FastAPI routes — POST /api/runs, GET /api/runs/{id}
  schemas.py   Pydantic models; every boundary is typed
  db.py        SQLAlchemy models (runs, stage_traces) + session
  worker.py    pipeline runner + the four pipeline stages
forecasting/   Stage 3A — data, features, baselines, training, serving
optimisation/  Stage 3B — network/cost model, CP-SAT solver, policies,
               evaluation window, benchmark harness, serving entry points
simulation/
  simulate.py  the daily-bucket simulator every policy is scored in
frontend/
  streamlit_app.py   polling client + results UI
Dockerfile
requirements.txt     PINNED
```

---

## Roadmap

Each stage replaces one function in `app/worker.py`. Nothing else changes.

| Stage | Replace | With | Difficulty |
|---|---|---|---|
| **3A** ✅ | `fake_forecast()` | LightGBM on M5 data, rolling-origin backtested | 6 |
| **3B** ✅ | `fake_optimise()` | OR-Tools CP-SAT multi-echelon inventory model | 8 |
| **3B** ✅ | `fake_simulate()` | Daily-bucket playout + 3 classical benchmark policies | 8 |
| **3C** | `fake_explain()` | LLM tool-calling agent for what-if scenarios | 9 |

### Stage 3A checklist
- [x] Load M5 data, build the item → dept → store → state hierarchy
- [x] Baselines: seasonal naive, moving average — **recorded below**
- [x] Features: origin-anchored history, calendar, SNAP days, prices, promos
- [x] LightGBM; compared against baselines honestly (untuned so far)
- [x] Rolling-origin backtesting (never a random split)
- [x] Prediction intervals via quantile regression (80.3% coverage vs 80% nominal)
- [x] Swap `fake_forecast()` for the real thing; redeploy

### Stage 3B checklist
- [x] Cost model: holding, ordering, stockout, transport — derived from real M5
      sell prices where derivable, assumptions named in `optimisation/network.py`
- [x] CP-SAT model, built small first (10 items, 14 days) before scaling to 60
- [x] Constraints: DC/shelf/truck capacity, two lead times, case-pack MOQ,
      aggregate **and** per-item service level
- [x] Rolling-horizon re-optimisation — plan 14 days, commit 7
- [x] Simulator + three benchmark policies + a perfect-information reference
      (plain daily-bucket, not SimPy — see DECISIONS.md entry 20)
- [x] Solver time limit, plus a relaxation ladder so an infeasible instance
      explains itself instead of returning an empty plan
- [x] Cross-model verification: `optimisation/verify.py` asserts the solver and
      the simulator charge a plan the same cost, term by term
- [ ] Replace the moving service bar with a fixed cost-service frontier — the
      current protocol amplifies solver noise (see Results, and DECISIONS.md 23)
- [ ] Move from BackgroundTasks to RQ + Redis — a 60-item run takes ~3 min, so
      this is now the next real task, not a hypothetical one

### Stage 3C checklist
- [ ] Expose `forecast`, `optimise`, `simulate` as agent tools
- [ ] Natural language → Pydantic constraint diff (never free text)
- [ ] Infeasibility handling: explain, don't crash
- [ ] Log tool, args, latency, tokens, cost per step
- [ ] Hard spend cap and per-session request quota

---

## Results

### Stage 3B — inventory optimisation

**A CP-SAT rolling-horizon policy costs 25.6% less than a tuned fixed
reorder-point policy at the same achieved service level** — mean over 6 runs,
range 23.2–28.6%, sd 2.0.

The range is quoted rather than a single figure because CP-SAT does not prove
optimality at this size and commits a slightly different plan each run. Any
single run is worth ±3 points; see [DECISIONS.md](DECISIONS.md) entry 23 for how
that was found, and for the tuning bug it uncovered.

One representative run (the one stored in `models/optimiser_benchmark.json`, at
28.7%):

| Policy | Total cost | Fill rate | Holding | Stockout | PO lines | Delivery days |
|---|---|---|---|---|---|---|
| Fixed reorder point (EOQ) | $7,178.21 | 83.0% | $57.03 | $2,573.08 | 201 | 25 |
| Safety stock (z·σ·√L) | $8,118.94 | 82.2% | $49.41 | $2,803.76 | 317 | 10 |
| Forecast base-stock | $7,870.81 | 86.0% | $52.59 | $2,185.62 | 339 | 13 |
| **CP-SAT rolling horizon (ours)** | **$5,120.55** | 82.0% | $52.34 | $2,648.38 | **114** | **10** |
| *Perfect information (reference)* | *$4,457.48* | *83.8%* | *$41.97* | *$1,963.39* | — | — |

Scope: store `CA_1`, the 60 largest items by revenue (49.9% of the modelled
store's revenue), 28 days, 6,001 actual units. Plan 14 days, commit 7, re-solve.

**How the comparison is made, which matters more than the number.** Any
inventory policy can be made cheaper by holding less and serving fewer
customers, so cost alone is meaningless. The protocol:

1. The CP-SAT policy runs at its 95% service target, and whatever fill rate it
   *achieves* becomes the bar.
2. Each baseline is then tuned to clear that same bar as cheaply as it can — an
   exhaustive grid over safety factor (0 to 3.0, step 0.125) crossed with review
   period {1,2,3,4,7,14}, keeping its cheapest qualifying configuration.
   Baselines get their best shot deliberately.
3. Only then are costs compared.

A baseline cannot be tuned below zero safety stock, so on a small instance it
over-serves at its cheapest feasible setting and the comparison stops being
equal-service. The overshoot is reported with every run and flagged past 1.5
points; here the comparator over-serves by 1.1 points.

Policies are decided on LightGBM forecasts from a model trained strictly before
the window and scored on **actual M5 sales it never saw** (2016-04-25 to
2016-05-22). The served forecast window has no ground truth anywhere in this
repo — it is M5's held-back future — which is why the benchmark runs one window
earlier; see [DECISIONS.md](DECISIONS.md) entry 13.

**Where the saving comes from, and where it does not.** Not from holding less
stock. Holding cost is $52 of a $5,121 total, because grocery carrying cost is
fractions of a cent per unit-day — a $1.68 item at a 25% annual rate carries for
about 0.08 cents a day. The whole advantage is consolidating the fixed costs a
per-item reorder rule structurally cannot see: a fixed cost per purchase-order
line, and a fixed cost per delivery day shared by every item on the truck.
**114 PO lines against 201, and 10 delivery days against 25**, at a comparable
stockout bill. That is a joint-replenishment result, not a safety-stock one.

**Ablation — forecast or optimiser?** `forecast_base_stock` is the identical
decision rule to `safety_stock_base_stock`, differing only in taking its
lead-time demand from the LightGBM forecast instead of trailing history. That
step is worth **3.1%** ($248). Replacing the rule with the optimiser is worth a
further **34.9%** ($2,750). So nearly all of the gain is the optimisation, not
the forecast — worth knowing before claiming either, and consistent with a cost
structure dominated by fixed charges that a forecast cannot help with.

Feeding the same controller actual demand instead of forecasts costs 12.9% less
than our policy, which prices what the forecast error is worth. It is a
reference, not a lower bound — see [DECISIONS.md](DECISIONS.md) entry 21.

Reproduce with `python -m optimisation.benchmark --items 60 --repeat 6`; full
output, including per-item fill quantiles and every constraint relaxation, in
`models/optimiser_benchmark.json`.

### Stage 3A — forecasting

Scope: store `CA_1`, 401 items sampled stratified by department (proportional to
each department's real size; seed 42, reproducible). 65.0% zero-sales days, against
M5's ~68% average — a representative slice. Backtested on 6 rolling-origin 28-day
windows per item, walk-forward, never a random split. Three metrics are reported because they answer different questions. RMSE and MASE
weight every item equally, so they are dominated by near-zero sellers. WRMSSE is
revenue-weighted — M5's own metric — because a 10% error on an item turning over
$500 a week matters more than the same error on one selling a unit a fortnight.
Ours is item-level only; the official version also averages across 12 aggregation
levels.

| Forecast model | RMSE | MASE | WRMSSE | vs seasonal naive |
|---|---|---|---|---|
| Seasonal naive | 1.656 | 1.568 | 1.204 | baseline |
| Moving average (28d) | 1.286 | **1.381** | 0.933 | −22.5% |
| **LightGBM** | **1.259** | 1.392 | **0.896** | **−25.6%** |

LightGBM beats the stronger baseline by 4.0% WRMSSE and seasonal-naive by 25.6%.
It loses equal-weighted MASE by 0.8%, and that is not hidden here: the entire
deficit is in the sparse tercile.

| Volume tercile (mean units/day) | LightGBM MASE | Moving average MASE |
|---|---|---|
| low (0.18) | 2.021 | **1.951** |
| mid (0.57) | **1.254** | 1.266 |
| high (2.64) | **0.902** | 0.925 |

On items averaging 0.18 units/day a shrunk constant is close to optimal and there
is little structure to learn; the model earns its keep on the mid and high
terciles, which is where the revenue is. Note also that moving-average beats
seasonal-naive throughout: seasonal-naive repeats one noisy week four times and
carries its spikes forward, while averaging shrinks toward the conditional mean,
which is what these metrics reward on low-count data. Per-item day-of-week signal
is weak relative to count noise — it is far clearer across the pooled panel,
which is precisely what the LightGBM model exploits.

### Prediction intervals

Separate LightGBM quantile models at the 10th and 90th percentiles. Stage 3B
needs these: safety stock is set from the level demand is unlikely to exceed, not
from expected demand.

| Volume tercile | Coverage | Mean width | Width ÷ mean daily demand |
|---|---|---|---|
| low (0.18/day) | 81.7% | 1.14 | 6.3× |
| mid (0.57/day) | 80.7% | 2.08 | 3.7× |
| high (2.64/day) | 78.3% | 5.05 | 1.9× |
| **overall** | **80.3%** | 2.76 | |

Coverage lands within 0.3 points of the 80% nominal target. The relative width
column is the operationally interesting one: uncertainty shrinks as a *proportion*
of demand as volume rises, which is why slow movers tie up disproportionate
safety-stock capital. The high tercile sits slightly under nominal at 78.3%,
which is the wrong direction for the highest-revenue items, since stockout cost
exceeds holding cost — a candidate fix is widening to 0.05/0.95 for that segment.

Full record of what broke and what was tried, including two changes that made
things worse and were reverted, is in [DECISIONS.md](DECISIONS.md).

---

## Engineering log

[DECISIONS.md](DECISIONS.md) records every problem hit, its actual root cause,
and the fix — including the four cascading Python 3.14 dependency failures, an
unrepresentative data sample that produced plausible-looking output, and a
feature-design flaw that made LightGBM lose to a moving average.

---

## Assumptions and limitations

**Stage 3B**

- **The optimiser is not solving to optimality at scale.** At 60 items the
  committed solves leave a mean 25% optimality gap inside their 30s budget; this
  is a capacitated joint-replenishment problem and it is NP-hard. The reported
  saving is what the policy *actually achieved* in simulation, so a better solve
  can only improve it — but our plans are not optimal, 25.6% is not the
  formulation's ceiling, and this is why the headline is quoted as a range.
- **95% is a planning target, and often not even a binding constraint.**
  Realised fill is ~81%, for two separate reasons both worth stating. The point
  forecast under-predicts this window by ~5% in total units, so serving 95% of
  forecast demand serves less of real demand. And mid-window the target is
  frequently *unsatisfiable* — the first two days of any horizon can only be
  served from stock already on the shelf — so the relaxation ladder drops it and
  availability is then driven by stockout penalties alone. The representative run
  records 6 such relaxations across 4 solves. That is economically sensible
  behaviour and every instance of it is reported, but it means the service level
  is not a guarantee. Making it a soft constraint with a steep penalty, rather
  than a hard one that switches off, is the better design and is not yet done.
- **The cost structure drives the result.** With a $12 purchase-order line and a
  $60 delivery, fixed costs dominate and the optimiser's edge is consolidation.
  Lower those and the margin narrows. The delivery cost is itself an allocation —
  a real truck serves far more than 60 items — so it stands in for a share of one.
- **Aggregate fill rate hides per-item misery, and here it does.** A per-item
  80% floor is imposed to stop the solver meeting the aggregate by abandoning
  awkward items, but it is dropped whenever it makes an instance infeasible. The
  result is a wide spread: median item fill 84.9%, but the 5th percentile is
  **52.7%**. An aggregate number would not show that, so the quantiles are in
  the benchmark JSON.
- **The comparison protocol still amplifies noise.** The service bar moves with
  whatever fill rate our policy happens to achieve, so solver variation shifts
  our cost and the baselines' target together. Comparing cost-service
  *frontiers* over a fixed grid of service levels would remove the coupling
  entirely, and is the first thing to fix next.
- **Two echelons, one store.** The network code takes arbitrarily many stores;
  the deployed forecaster covers one, so the demo instance is supplier → DC →
  single store with items competing for shared DC, shelf and truck capacity.
- Distribution-centre layer is synthetic; M5 provides store-level data only.
  Lead times, capacities, case packs and the fixed-cost structure are stated
  assumptions in `optimisation/network.py`, not measurements. Prices, and the
  demand statistics the policies are parameterised from, are real.
- Cost parameters are estimated, not from a real firm.

**Stage 3A**

- Baseline numbers above cover one store (`CA_1`) and 401 of its ~3,049 items.
  Department composition is representative, but a single store is not: `CA_1` has
  its own SNAP calendar and price levels, so cross-store variation is untested.
  Scale to multiple stores before quoting these numbers as dataset-wide.
- (add yours)
