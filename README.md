# Supply Chain Intelligence Platform

Demand forecasting → inventory optimisation → scenario analysis, deployed as an async service.

> **Status: Stage 3A complete, not yet deployed.** The plumbing is real and the
> forecasting stage is a real LightGBM model with quantile intervals, serving in
> ~20ms from precomputed features. Optimise, simulate and explain are still
> placeholders that sleep and return synthetic numbers. Nothing is on a public URL
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
  -d '{"store_ids":["CA_1"],"item_ids":["FOODS_3_090"],"horizon_days":28}' \
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
  worker.py    pipeline runner + the four placeholder stages
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
| **3A** | `fake_forecast()` | LightGBM on M5 data, rolling-origin backtested | 6 |
| **3B** | `fake_optimise()` | OR-Tools CP-SAT multi-echelon inventory model | 8 |
| **3B** | `fake_simulate()` | SimPy playout + EOQ / reorder-point benchmarks | 8 |
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
- [ ] Cost model: holding, ordering, stockout, transport — document assumptions
- [ ] CP-SAT model, small first (1 DC, 3 stores, 5 items, 14 days)
- [ ] Constraints: capacity, lead times, MOQ, service level, truck capacity
- [ ] Rolling-horizon re-optimisation every 7 simulated days
- [ ] SimPy simulator + three benchmark policies
- [ ] Solver time limit so a bad input cannot hang the worker
- [ ] Move from BackgroundTasks to RQ + Redis if runs exceed ~2 min

### Stage 3C checklist
- [ ] Expose `forecast`, `optimise`, `simulate` as agent tools
- [ ] Natural language → Pydantic constraint diff (never free text)
- [ ] Infeasibility handling: explain, don't crash
- [ ] Log tool, args, latency, tokens, cost per step
- [ ] Hard spend cap and per-session request quota

---

## Results

> Fill this in as you go. **This table is the most important part of the README.**

| Policy | Total cost | Service level |
|---|---|---|
| Optimised (ours) | — | — |
| Fixed reorder point (EOQ) | — | — |
| Run to failure | — | — |

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

> Fill this in honestly. Stating where your model is weak reads as seniority.

- Distribution-centre layer is synthetic; M5 provides store-level data only.
- Cost parameters are estimated, not from a real firm.
- Baseline numbers above cover one store (`CA_1`) and 401 of its ~3,049 items.
  Department composition is representative, but a single store is not: `CA_1` has
  its own SNAP calendar and price levels, so cross-store variation is untested.
  Scale to multiple stores before quoting these numbers as dataset-wide.
- (add yours)
