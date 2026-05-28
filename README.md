# CobbleStoneEnergy — DE Power Forecasting Pipeline

**German Day-Ahead Power Price Forecasting → Prompt Curve Translation**

End-to-end prototype built for the CobbleStoneEnergy case study. Forecasts hourly DA prices for Germany using ENTSO-E public data, derives delivery-period trading signals, and integrates two programmatic LLM features as productivity levers.

| | |
|---|---|
| **Market** | Germany DE-LU (bidding zone `10Y1001A1001A63L`) |
| **Data** | ENTSO-E local CSVs — 2024-01-01 → 2026-05-27 (21,070 hourly rows) |
| **Models** | Seasonal Naive 168h · Ridge Linear · **LightGBM** |
| **LightGBM test MAE** | **23.08 EUR/MWh** (−48.5% vs Seasonal Naive) |
| **AI** | Groq free tier — Llama 3.3 70B (agent + commentary) · Llama 3.1 8B (QA rules) |

---

## Project Structure

```
.
├── pipeline.py                  # CLI orchestrator (standalone, no server needed)
├── requirements.txt
├── .env.example                 # Copy to .env — only GROQ_API_KEY needed
├── submission.md                # Case study write-up (2 pages)
├── notebooks/
│   └── 01_analysis.ipynb        # EDA and results review
├── src/
│   ├── data/
│   │   ├── ingestion.py         # Local CSV loader + ENTSO-E API fallback
│   │   └── qa.py                # QA checks + LLM rule executor
│   ├── features/
│   │   └── engineering.py       # No-leakage feature engineering
│   ├── models/
│   │   ├── baseline.py          # Seasonal Naive 168h + Ridge Linear
│   │   └── forecasting.py       # LightGBM + walk-forward expanding CV
│   ├── trading/
│   │   └── curve_translation.py # DA → delivery-period aggregation + Z-score signal
│   └── ai/
│       └── llm_component.py     # Groq LLM: QA rule proposer + daily commentary
├── backend/
│   ├── main.py                  # FastAPI app + CORS + lifespan startup
│   ├── routers/                 # REST endpoints: /forecast /qa /chat
│   └── services/
│       ├── pipeline_service.py  # Wraps src/ pipeline; serves API results
│       └── ai_agent.py          # Groq agent with 6 tool-calling tools
├── frontend/
│   └── src/
│       ├── App.tsx              # Layout: sidebar + chat + charts
│       ├── components/
│       │   ├── ChatPanel.tsx    # AI chat interface + quick-action buttons
│       │   ├── ForecastChart.tsx # Recharts hourly + delivery curve charts
│       │   ├── MetricsPanel.tsx  # Model performance table
│       │   ├── QAPanel.tsx       # Data quality report
│       │   └── SignalBadge.tsx   # LONG/SHORT/FLAT badge
│       └── api/client.ts        # Typed API client
├── data/
│   ├── raw/                     # ENTSO-E CSV files (tracked) + Parquet cache (gitignored)
│   └── processed/               # Reserved for derived outputs
└── outputs/
    ├── figures/                 # Generated PNG charts
    ├── logs/                    # LLM call logs (JSON, one per call)
    └── qa_reports/              # QA JSON report (service_report.json)
```

---

## Setup

### 1. Python environment

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment variables

```bash
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)
```

| Variable | Required? | Notes |
|---|---|---|
| `GROQ_API_KEY` | Recommended (free) | Powers AI agent + QA rules + commentary. Without it the chat still works but LLM features return a fallback message. |
| `ENTSOE_API_KEY` | Not required | Local CSV files in `data/raw/` are already present. Only needed to re-download from the API. |

### 3. Data

All CSV files are already in `data/raw/`. No download needed.

| File | Source | Content |
|---|---|---|
| `energyprice{2024–2026}.csv` | ENTSO-E A44 | Hourly DA prices (EUR/MWh), Sequence 1 |
| `load{2024–2026}.csv` | ENTSO-E A65 | Actual load + DA load forecast (15-min → 1h) |
| `wind_solar{2024–2026}.csv` | ENTSO-E A69 | Wind+solar onshore DA generation forecast (15-min → 1h) |

---

## Running the Web App

### Terminal 1 — FastAPI backend

```bash
python3 -m uvicorn backend.main:app --port 8000
```

The backend initialises the full pipeline on startup. **Each run takes ~7 minutes** (walk-forward CV re-runs on every start; data loading reuses the Parquet cache and is fast). API docs at http://localhost:8000/docs.

### Terminal 2 — React frontend

```bash
cd frontend
npm install      # first time only
npm run dev
```

Open http://localhost:5173. The UI polls `/api/health` every 2 seconds and unlocks once the pipeline is ready.

### What you can ask the AI agent

```
"What's the forecast for tomorrow?"
"Show me the monthly delivery curve"
"How accurate is the LightGBM model?"
"What's the current trading signal?"
"Run data quality checks"
```

The Groq agent (Llama 3.3 70B) interprets the question, calls the appropriate pipeline tool, and returns a natural-language answer. Charts update automatically when forecast or delivery data is returned.

---

## Running the CLI Pipeline

```bash
# Full pipeline run
python3 pipeline.py

# Skip LLM calls (no GROQ_API_KEY needed)
python3 pipeline.py --no-llm

# Force re-download from ENTSO-E API (requires ENTSOE_API_KEY)
python3 pipeline.py --force-refresh

# Custom date range
python3 pipeline.py --start 2024-01-01 --end 2026-05-25 --test-start 2025-07-01
```

---

## Model Performance

Walk-forward expanding CV, 7-day retrain frequency, 180-day minimum training window.

**Hold-out test set: 2026-02-26 → 2026-05-27 (2,161 hours)**

| Model | MAE (EUR/MWh) | RMSE (EUR/MWh) | Tail MAE P90 |
|---|---|---|---|
| Seasonal Naive 168h | 44.84 | 67.00 | 70.65 |
| Ridge Linear | 33.04 | 47.50 | 58.47 |
| **LightGBM** | **23.08** | **35.02** | **45.01** |
| LightGBM (CV MAE) | 18.11 | 30.46 | 40.86 |

LightGBM beats the Seasonal Naive baseline by **−48.5% MAE**. Adding the ENTSO-E wind+solar DA generation forecast as a third fundamental driver reduced MAE by a further ~30% compared to the load-only model.

---

## Features Used

| Group | Features |
|---|---|
| Calendar | `hour`, `dow`, `month`, `doy`, `is_weekend`, `is_summer` |
| Price lags | `lag_24h`, `lag_48h`, `lag_168h`, `roll7d_mean`, `roll7d_std`, `roll30d_mean` |
| Load | `load_mw` (actual load, gate-safe via training on historical data), `load_forecast_mw` (TSO D+1 forecast), `load_roll7d_mean` (24h-shifted rolling mean) |
| Wind+solar | `wind_solar_da_mw` (ENTSO-E D+1 forecast), `wind_solar_roll7d_mean` |
| Derived | `ren_pen` (VRE / load), `residual_load_mw` (load − wind_solar) |

All features are gate-closure safe: only information available at noon D is used to predict all hours of D+1.

---

## Data Sources

Base URL: `https://web-api.tp.entsoe.eu/api`  
Full API guide: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

| Dataset | documentType | Parameters |
|---|---|---|
| DA Prices | A44 | `in_Domain=10Y1001A1001A63L`, `out_Domain=10Y1001A1001A63L` |
| Actual Load + DA Forecast | A65 | `processType=A16/A01`, `outBiddingZone_Domain=10Y1001A1001A63L` |
| Wind+Solar Onshore DA Forecast | A69 | `processType=A01`, `psrType=B18+B16`, `outBiddingZone_Domain=10Y1001A1001A63L` |

Timestamps are stored in UTC throughout. CET/CEST local time is used only for display and peak-hour classification (hours 8–19 CET = peak).

---

## AI Components

Both are implemented in `src/ai/llm_component.py` and called programmatically from the pipeline.

**1. LLM-driven QA rules** (`llama-3.1-8b-instant`)  
Given the dataset schema + descriptive statistics + 5 sample rows, the LLM proposes pandas validation rules as a JSON array. The pipeline executes each rule and reports pass/fail counts in the QA report.

**2. Automated market commentary** (`llama-3.3-70b-versatile`)  
After forecasting, a structured prompt containing only computed metrics (base/peak/off-peak price, WoW change, load, renewable penetration, rolling history) is sent to the LLM. It generates a 3–4 sentence morning note. No numbers are invented.

Every LLM call is logged to `outputs/logs/llm_<tag>_<epoch>.json` with full prompt, response, model, latency, and token usage. All failures are caught and logged; the pipeline continues without the AI step if the API is unavailable.

---

## Tests

```bash
python3 -m pytest tests/test_pipeline.py -v
```

All 20 tests pass. Tests cover ingestion, feature engineering, model validation, QA checks, and curve translation.
