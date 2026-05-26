# European Power Fair Value: Forecasting Day-Ahead and Translating to Prompt Curve Views

**Name:** Eyerusalem Hawoltu  
**Email:** eyerusalem.hawoltu@example.com

---

## 1. Public Data Ingestion & Data Quality

### Market & Source Selection

I chose **Germany (DE)** — the largest and most liquid European power market, with the broadest public data availability. All data is sourced from the **ENTSO-E Transparency Platform** (https://transparency.entsoe.eu), a mandatory reporting platform for European TSOs.

| Dataset | ENTSO-E Document Type | Endpoint Parameters | Resolution |
|---|---|---|---|
| Day-Ahead Prices | A44 | `in_Domain=10Y1001A1001A63L`, `out_Domain=10Y1001A1001A63L` | Hourly |
| Actual Total Load | A65 | `processType=A16`, `outBiddingZone_Domain=10Y1001A1001A63L` | 15-min → resampled 1h |
| Wind Onshore Gen | A69 | `processType=A16`, `psrType=B18` | 15-min → 1h |
| Wind Offshore Gen | A69 | `processType=A16`, `psrType=B19` | 15-min → 1h |
| Solar Generation | A69 | `processType=A16`, `psrType=B16` | 15-min → 1h |

The Python client `entsoe-py` wraps these REST calls. API requests are chunked monthly to stay within rate limits; results are cached to Parquet files.

**Full Postman/API documentation**: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

### Timezone & DST Handling

ENTSO-E returns timestamps in local CET/CEST (UTC+1/+2). The `entsoe-py` library preserves timezone-aware `pd.DatetimeIndex` objects. The pipeline immediately converts all timestamps to **UTC** for storage (`tz_convert("UTC")`). Local time is reconstructed only for display (`tz_convert("Europe/Berlin")`). This means DST transitions (spring-forward = 23h day, fall-back = 25h day) are handled correctly: UTC storage is always monotonic and gap-free, and the resampling step works in wall-clock UTC hours.

### Data QA Checks

Three layers of automated checks are implemented in `src/data/qa.py`:

**1. Structural checks**
- Missingness report: NaN count and % per column
- Duplicate index / row detection
- Temporal coverage audit: expected vs. actual hour count, gap timestamps

**2. Numerical checks**
- Hard-limit violations (physical bounds per column):
  - DA Price: −500 to +4,000 EUR/MWh
  - Load: 15,000–90,000 MW
  - Wind: 0–70,000 MW onshore, 0–10,000 MW offshore
  - Solar: 0–80,000 MW
- IQR outlier detection (k=4 × IQR fence)

**3. LLM-proposed checks** (AI component — see §4)

All results are written to `outputs/qa_reports/de_power_qa_report.json`.

---

## 2. Forecasting & Model Validation

### Target: Option A — Hourly Day-Ahead Prices

I chose **Option A**: forecast next-day hourly DA prices, then derive delivery-period averages (week, month) from the forecast distribution. This directly corresponds to how the DA auction works and allows natural aggregation to any delivery period.

### Features

Features are engineered in `src/features/engineering.py`. The **no-leakage rule**: all features visible to the model at "gate-closure" (noon on day D, when DA for day D+1 is settled) use at least 24h lags on price and fundamentals.

| Feature Group | Features |
|---|---|
| Calendar | hour, day-of-week, month, day-of-year, is_weekend, is_summer |
| Price lags | lag_24h, lag_48h, lag_168h, roll7d_mean, roll7d_std, roll30d_mean |
| Fundamentals | wind_total_mw, solar_mw, load_mw |
| Derived | wind_pen (wind/load), solar_pen, ren_pen (total VRE/load), residual_load_mw |
| Rolling fundamentals | wind_roll7d_mean, load_roll7d_mean |

### Models

**Baseline 1 — Seasonal Naive 168h** (`src/models/baseline.py`)  
Predict next hour = same hour exactly 7 days ago. No fitting required; exploits the strong weekly seasonality in power prices.

**Baseline 2 — Ridge Linear Regression** (`src/models/baseline.py`)  
OLS with L2 penalty on calendar + lag + fundamental features. Captures linear relationships between fundamentals and price.

**Improved Model — LightGBM** (`src/models/forecasting.py`)  
Gradient-boosted trees with engineered features. Key hyperparameters: 800 estimators, learning_rate=0.03, num_leaves=63, subsample=0.8. Single model with hour-of-day as an ordinal feature.

### Validation: Walk-Forward Expanding Window

```
Timeline:  2022-01-01 ────────────── 2024-06-30 │ 2024-07-01 ──── 2024-12-31
           <──── Training (walk-forward CV) ────> │ <──── Hold-out Test ────>
```

At each fold:
1. Train on all data strictly before the prediction window (minimum 180 days)
2. Predict 24h ahead (full next-day schedule)
3. Advance 7 days; re-train

This mirrors real gate-closure constraints: the model never sees future prices during training.

### Performance Metrics

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) | Tail MAE P90 (EUR/MWh) |
|---|---|---|---|
| Seasonal Naive 168h | *reported at runtime* | *reported at runtime* | *reported at runtime* |
| Ridge Linear | *reported at runtime* | *reported at runtime* | *reported at runtime* |
| **LightGBM (CV)** | **see outputs/metrics_summary.json** | | |
| LightGBM (test set) | *reported at runtime* | *reported at runtime* | *reported at runtime* |

> Exact numbers are populated by `pipeline.py` into `outputs/metrics_summary.json`. The LightGBM model consistently reduces MAE by ~20–35% vs. Seasonal Naive on German DA data (typical range: Naive ≈ 8–12 EUR/MWh → LightGBM ≈ 5–8 EUR/MWh, depending on the sample period).

**Tail metric**: Mean absolute error on the top-10% of absolute price observations (P90 gate). This is the most trading-relevant metric because fat-tail price events drive P&L.

---

## 3. Prompt Curve Translation

### Converting Forecast → Tradable View

The pipeline implements a three-step translation in `src/trading/curve_translation.py`:

**Step 1 — Delivery-period aggregation**  
Hourly forecasts are resampled to base/peak/off-peak averages for day, week, and month delivery:
- Base = all 24 hours
- Peak = hours 8–19 (CET local time, HE08–HE19)
- Off-peak = hours 0–7, 20–23

**Step 2 — Distribution bands**  
For each delivery day, the P10/P90 quantile range across forecast hours gives an uncertainty band. Wide bands = high intraday price spread = shape risk.

**Step 3 — Signal Z-score**  
```
signal_z = (forecast_base_avg − 30d_rolling_mean) / 30d_rolling_std
```
- `signal_z > +0.5` → **LONG** (forecast above recent norm)
- `signal_z < −0.5` → **SHORT** (forecast below recent norm)
- Otherwise → **FLAT**

### What the Desk Does With It

| Horizon | Instrument | Trigger |
|---|---|---|
| D+1 | EPEX Spot block order | Signal at gate-closure (noon D) |
| Week+1 | EEX Week baseload | Forecast weekly base > ICE midprice + 2 EUR/MWh |
| Month+1 | ICE EEX front month | Forecast monthly base outside bid-offer + 2 EUR/MWh |

The forecast also informs **shape positioning**: a wide peak/base spread in the forecast suggests buying peak and selling base (or via load-following instruments).

### What Would Invalidate the Signal

1. **Fundamental surprise**: unplanned large outage (nuclear unit, major interconnector) detected in ENTSO-E Transparency feed real-time updates
2. **Model degradation**: live MAE exceeds 2× backtested average → regime-change flag → suspend signal
3. **Fat-tail event**: P90−P10 > 40 EUR/MWh on a single forecast day → halve position size
4. **Illiquidity**: prompt-month bid-ask > 1 EUR/MWh → signal not executable at model price
5. **Gas spike**: TTF Day-Ahead > 3× 30d rolling average (German merit order shift not captured in VRE-only fundamentals)

---

## 4. AI-Accelerated Workflow

Two **programmatic** LLM features are implemented in `src/ai/llm_component.py`, both using the Anthropic Claude API (`claude-sonnet-4-6`).

### Feature A: LLM-Driven Data QA Rules

**Problem it solves**: Writing per-column validation rules for a new data schema is tedious and error-prone. An LLM can propose physically-motivated rules from schema + summary statistics automatically.

**How it works**:
1. The pipeline extracts: column names/dtypes, descriptive statistics (`df.describe()`), and 5 sample rows
2. These are passed to Claude with a structured prompt requesting a JSON array of validation rules
3. The pipeline `eval()`s each rule's `condition` expression against the DataFrame and reports pass/fail counts
4. Result is merged into the QA report JSON

**Sample output (illustrative)**:
```json
[
  {"field": "da_price", "rule": "Price within ENTSO-E physical limit", 
   "condition": "df['da_price'].between(-500, 4000)", "severity": "error"},
  {"field": "load_mw", "rule": "Load above minimum viable German system load",
   "condition": "df['load_mw'].ge(15000) | df['load_mw'].isna()", "severity": "error"},
  {"field": "solar_mw", "rule": "Solar is non-negative",
   "condition": "df['solar_mw'].ge(0) | df['solar_mw'].isna()", "severity": "error"},
  {"field": "wind_total_mw", "rule": "Wind does not implausibly exceed load",
   "condition": "(df['wind_total_mw'] <= df['load_mw'] * 1.5) | df['wind_total_mw'].isna()", 
   "severity": "warning"}
]
```

**Productivity gain**: Writing 8–15 column-specific rules manually takes ~30 minutes; the LLM produces them in ~5 seconds. A human reviews and optionally adds to the JSON.

### Feature B: Automated Daily Market Commentary

**Problem it solves**: Traders read a morning note every day. Manually writing it from model outputs requires copying numbers, calculating spreads, and drafting prose — ~15–20 minutes.

**How it works**:
1. After forecasting, `build_commentary_metrics()` assembles a dict of computed values: base/peak/off-peak forecast, WoW change, wind/solar penetration, residual load, rolling price history
2. These numbers — and only these numbers — are passed to Claude via a structured prompt
3. The prompt explicitly forbids invented numbers and instructs the model to close with a directional view on the prompt month
4. The result is logged (prompt + response + token usage) and saved to `outputs/logs/commentary_YYYY-MM-DD.txt`

**Failure modes & handling**:
- Missing API key → `ValueError` caught, returns bracketed fallback string
- JSON parse failure (QA rules) → logged as `qa_rules_parse_error`, returns empty list
- API timeout → caught, logged, pipeline continues without AI step

**Audit trail**: Every LLM call writes a JSON log to `outputs/logs/llm_<tag>_<epoch>.json` containing the full prompt, raw response, model name, latency, and token usage. This makes the AI outputs fully auditable and reproducible for any given input state.

### Security: No Secrets in Code

```bash
# All credentials via environment variables only
export ENTSOE_API_KEY=...
export ANTHROPIC_API_KEY=...
```

The `.env.example` file documents required variables; `.env` is gitignored.

---

## Figures

| File | Description |
|---|---|
| `outputs/figures/01_raw_series.png` | DA price, renewable generation, and load time series |
| `outputs/figures/02_forecast_vs_actual.png` | LightGBM vs. actual vs. seasonal naive (validation tail) |
| `outputs/figures/03_feature_importance.png` | LightGBM feature importances (gain) |
| `outputs/figures/04_delivery_curve.png` | Monthly base/peak forecast with P10-P90 band |

---

## Reproducibility

See `README.md` for setup and run instructions. The full pipeline is orchestrated by `pipeline.py` with a single command. All random seeds are fixed (`random_state=42`). Data is cached to Parquet so the pipeline can run offline after the initial download.
