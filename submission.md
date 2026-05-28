# European Power Fair Value: Forecasting Day-Ahead and Translating to Prompt Curve Views

**Name:** Eyerusalem Hawoltu Afework
**Email:** eh3115@nyu.edu

---

## 1. Public Data Ingestion & Data Quality

### Market & Source Selection

I chose **Germany (DE)** — the largest and most liquid European power market, with the broadest public data availability. All data is sourced from the **ENTSO-E Transparency Platform** (https://transparency.entsoe.eu), a mandatory reporting platform for European TSOs.

| Dataset | ENTSO-E Document Type | Endpoint Parameters | Resolution |
|---|---|---|---|
| Day-Ahead Prices | A44 | `in_Domain=10Y1001A1001A63L`, `out_Domain=10Y1001A1001A63L` | Hourly |
| Actual Total Load | A65 | `processType=A16`, `outBiddingZone_Domain=10Y1001A1001A63L` | 15-min → resampled 1h |
| Day-Ahead Load Forecast | A65 | `processType=A01`, `outBiddingZone_Domain=10Y1001A1001A63L` | 15-min → resampled 1h |
| Wind+Solar Onshore Gen Forecast | A69 | `processType=A01`, `psrType=B18+B16`, `outBiddingZone_Domain=10Y1001A1001A63L` | 15-min → resampled 1h |

**Full API documentation**: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

Data was manually exported from ENTSO-E Transparency and loaded from local CSV files (`data/raw/energyprice{YYYY}.csv`, `data/raw/load{YYYY}.csv`). The pipeline also supports direct ENTSO-E API ingestion via `entsoe-py` when `ENTSOE_API_KEY` is set. Coverage: **2024-01-01 → 2026-05-27** (21,070 hourly rows).

**Three fundamental drivers (two distinct physical categories):**
1. `load_mw` / `load_forecast_mw` — ENTSO-E A65 actual load and TSO day-ahead load forecast; gate-closure safe (published before noon D for all D+1 hours)
2. `wind_solar_da_mw` — ENTSO-E A69 combined wind onshore + solar day-ahead generation forecast for DE-LU bidding zone; gate-closure safe (TSO publishes before the DA auction closes)

### Timezone & DST Handling

ENTSO-E exports timestamps in local CET/CEST. The pipeline parses them with explicit UTC conversion (`tz_convert("UTC")`) immediately on load. All internal storage is UTC. Local time is reconstructed only for display (`tz_convert("Europe/Berlin")`). DST transitions (23h/25h days) are handled correctly: UTC storage is monotonic and gap-free, and all resampling runs in UTC hours.

### Data QA Checks

Three layers of automated checks in `src/data/qa.py`:

**1. Structural checks**
- Missingness: 0 NaNs across all 3 columns (21,070 rows)
- Duplicate index/row detection: 0 duplicates
- Temporal coverage: 21,070/21,070 expected hours, 0 gaps

**2. Hard-limit checks (physical bounds)**
| Column | Min observed | Max observed | Violations |
|---|---|---|---|
| `da_price` | −500.00 EUR/MWh | 936.28 EUR/MWh | 0 |
| `load_mw` | 32,813 MW | 79,008 MW | 0 |

**3. LLM-proposed checks** — see §4

Results written to `outputs/qa_reports/service_report.json`.

---

## 2. Forecasting & Model Validation

### Target: Option A — Hourly Day-Ahead Prices

I forecast next-day hourly DA prices, then aggregate to delivery-period averages (day, week, month). This directly mirrors the DA auction structure and allows natural aggregation to any delivery period.

### Features

All features respect the gate-closure constraint: the model only uses information available at noon on day D when predicting all 24 hours of day D+1. Price features use ≥24h lags; load actuals are shifted 24h.

| Feature Group | Features |
|---|---|
| Calendar | `hour`, `dow`, `month`, `doy`, `is_weekend`, `is_summer` |
| Price lags | `lag_24h`, `lag_48h`, `lag_168h`, `roll7d_mean`, `roll7d_std`, `roll30d_mean` |
| Load fundamentals | `load_mw` (actual load), `load_forecast_mw` (D+1 TSO forecast, gate-closure safe), `load_roll7d_mean` (24h-shifted rolling mean) |
| Renewable fundamental | `wind_solar_da_mw` (ENTSO-E D+1 wind+solar forecast, gate-closure safe) |
| Derived | `ren_pen` = wind_solar / load, `residual_load_mw` = load − wind_solar, `wind_solar_roll7d_mean` |

All features respect the gate-closure constraint: the model only uses information available at noon D when predicting D+1 prices.

### Models

**Baseline 1 — Seasonal Naive 168h** (`src/models/baseline.py`)
Predict next hour = same hour exactly 7 days ago. No fitting required; exploits the strong weekly seasonality in power prices.

**Baseline 2 — Ridge Linear Regression** (`src/models/baseline.py`)
L2-penalised OLS on calendar + price lag + load features. Captures linear relationships; robust to collinear features.

**Improved Model — LightGBM** (`src/models/forecasting.py`)
Gradient-boosted trees. Key hyperparameters: 800 estimators, learning_rate=0.03, num_leaves=63, subsample=0.8. Hour-of-day included as an ordinal feature, allowing the model to learn non-linear intraday price shapes.

### Validation: Walk-Forward Expanding Window CV

```
Timeline:  2024-01-01 ──────────────────── 2026-02-26 │ 2026-02-26 ── 2026-05-27
           <──── Walk-forward CV (expanding) ─────────> │ <──── Hold-out test ────>
```

At each fold: train on all data strictly before the window (minimum 180 days), predict 24h ahead, advance 7 days and re-train. This mirrors real gate-closure constraints with zero leakage.

### Performance Metrics

**Walk-forward CV results (14,421 samples):**

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) | Tail MAE P90 (EUR/MWh) |
|---|---|---|---|
| **LightGBM (CV)** | **18.11** | **30.46** | **40.86** |

**Hold-out test set (2026-02-26 → 2026-05-27, 2,161 hours):**

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) | Tail MAE P90 (EUR/MWh) |
|---|---|---|---|
| Seasonal Naive 168h | 44.84 | 67.00 | 70.65 |
| Ridge Linear | 33.04 | 47.50 | 58.47 |
| **LightGBM** | **23.08** | **35.02** | **45.01** |

LightGBM beats Seasonal Naive by **−48.5% MAE** and Ridge by **−30.1% MAE** on the hold-out set. Adding the wind+solar day-ahead generation forecast as a third fundamental driver was the single largest driver of accuracy improvement (MAE dropped from 32.5 → 23.1 EUR/MWh versus the load-only model). The tail metric (P90 gate) is the most trading-relevant: fat-tail price events drive P&L.

---

## 3. Prompt Curve Translation

### Converting Forecast → Tradable View

The pipeline implements a three-step translation in `src/trading/curve_translation.py`:

**Step 1 — Delivery-period aggregation**
Hourly forecasts are resampled to base/peak/off-peak averages for day, week, and month delivery:
- Base = all 24 hours
- Peak = hours 8–19 CET (HE08–HE19)
- Off-peak = hours 0–7, 20–23

**Step 2 — Distribution bands**
P10/P90 quantile range across forecast hours gives an uncertainty band per delivery day. Wide bands = high intraday spread = shape risk.

**Step 3 — Signal Z-score**
```
signal_z = (forecast_base_avg − 30d_rolling_mean) / 30d_rolling_std
```
- `signal_z > +0.5` → **LONG** (forecast above recent norm)
- `signal_z < −0.5` → **SHORT** (forecast below recent norm)
- Otherwise → **FLAT**

**Live example (2026-05-27):** forecast base = 69.45 EUR/MWh vs. 30d benchmark = 95.37 EUR/MWh → Z = −0.88 → **SHORT**

### What the Desk Does With It

| Horizon | Instrument | Trigger |
|---|---|---|
| D+1 | EPEX Spot block order | Signal at gate-closure (noon D) |
| Week+1 | EEX Week baseload | Forecast weekly base vs. ICE midprice + 2 EUR/MWh threshold |
| Month+1 | ICE EEX front month | Forecast monthly base outside bid-offer + 2 EUR/MWh |

A wide peak/base spread in the forecast suggests shape positioning (buy peak, sell base).

### What Would Invalidate the Signal

1. **Fundamental surprise** — unplanned large outage (nuclear unit, major interconnector) not yet reflected in TSO Transparency data
2. **Model degradation** — live MAE exceeds 2× backtested average → suspend signal, flag regime change
3. **Fat-tail alert** — P90−P10 > 40 EUR/MWh on a single forecast day → halve position size
4. **Illiquidity** — prompt-month bid-ask > 1 EUR/MWh → signal not executable at model price
5. **Gas spike** — TTF Day-Ahead > 3× 30d rolling average → German merit order shifts in ways not captured by load-only fundamentals

---

## 4. AI-Accelerated Workflow

Two **programmatic** LLM features are implemented in `src/ai/llm_component.py`, both using the **Groq API** (free tier, no billing required).

### Feature A: LLM-Driven Data QA Rules

**Model**: `llama-3.1-8b-instant` (Groq free tier, ~1s latency)

**Problem it solves**: Writing per-column validation rules for a new data schema is tedious and error-prone. An LLM can propose physically-motivated rules from schema + summary statistics automatically.

**How it works**:
1. The pipeline extracts column names/dtypes, descriptive statistics (`df.describe()`), and 5 sample rows
2. These are passed to the LLM with a structured prompt requesting a JSON array of validation rules
3. Each rule's `condition` expression is evaluated against the DataFrame and pass/fail counts are reported
4. Results are merged into the QA report JSON

**Sample output from actual run (`outputs/logs/llm_qa_rules_*.json`):**
```json
[
  {"field": "da_price",        "rule": "Non-negative DA price",           "condition": "df['da_price'] >= 0",        "severity": "error"},
  {"field": "load_mw",         "rule": "Non-negative load",               "condition": "df['load_mw'] >= 0",         "severity": "error"},
  {"field": "load_forecast_mw","rule": "Non-negative load forecast",      "condition": "df['load_forecast_mw'] >= 0","severity": "error"}
]
```

**Productivity gain**: Writing 8–15 column-specific rules manually takes ~30 minutes; the LLM produces them in ~1 second.

### Feature B: Automated Daily Market Commentary

**Model**: `llama-3.3-70b-versatile` (Groq free tier)

**Problem it solves**: Writing a daily morning note from model outputs requires copying numbers, calculating spreads, and drafting prose — ~15–20 minutes per day.

**How it works**:
1. `build_commentary_metrics()` assembles a dict of computed values: base/peak/off-peak forecast, WoW change, load, rolling price history
2. Only these computed values — no invented numbers — are passed to the LLM via a structured prompt
3. The LLM generates a 3–4 sentence note and closes with a directional view on the prompt month
4. Full prompt + response + token usage are logged to `outputs/logs/llm_daily_commentary_*.json`

**Sample output from actual run:**
> "The day-ahead market is expected to be characterised by a base average price of 70.35 EUR/MWh, with a notable week-over-week decrease of 57.61 EUR/MWh (−45.0%). The peak and off-peak averages reflect a wide intraday spread, with the 7-day and 30-day rolling averages at 98.01 EUR/MWh suggesting that current forecast levels are meaningfully below recent norms."

**Failure modes handled**:
- Missing API key → `ValueError` caught, returns bracketed fallback string; pipeline continues
- JSON parse failure (QA rules) → logged as `qa_rules_parse_error`, returns empty list
- Any API exception → caught, logged with full traceback, pipeline continues without AI step

**Audit trail**: Every LLM call writes a JSON log to `outputs/logs/llm_<tag>_<epoch>.json` containing full prompt, raw response, model name, latency, and token usage.

### Security: No Secrets in Code

```bash
export GROQ_API_KEY=your_key_here   # free at console.groq.com
# export ENTSOE_API_KEY=...         # optional — not needed when local CSVs present
```

`.env.example` documents all required variables. `.env` is gitignored.

---

## Reproducibility

See `README.md` for full setup and run instructions. The pipeline is orchestrated by a single CLI command:

```bash
python pipeline.py --no-llm   # fast run without LLM calls
python pipeline.py            # full run with QA rules + commentary
```

All random seeds are fixed (`random_state=42`). Data is cached to Parquet after first load so the pipeline runs offline. The backend (FastAPI) and frontend (React + Vite) are independently runnable.
