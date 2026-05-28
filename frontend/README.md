# Frontend — React + TypeScript + Vite

Real-time dashboard for the CobbleStoneEnergy DE power forecasting pipeline.

## Stack

- **React 18** + TypeScript
- **Vite** dev server (HMR)
- **Recharts** — hourly forecast and delivery curve charts
- **Lucide React** — icons

## Running

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # production build → dist/
```

The frontend polls `http://localhost:8000/api/health` every 2 seconds until the backend pipeline finishes initialising (~7 min on first run, faster on subsequent runs using the Parquet cache).

## What it shows

| Panel | Content |
|---|---|
| **Header** | Live/loading status, trading signal (LONG/SHORT/FLAT), Z-score |
| **Performance tab** | CV MAE/RMSE/Tail-P90, model comparison table, test period dates |
| **Data QA tab** | Missingness, duplicates, temporal gaps, hard-limit violations, LLM-proposed rules |
| **Chat panel** | Groq AI agent (Llama 3.3 70B) — natural language queries + chart rendering |
| **Chart area** | Hourly forecast vs actual, monthly delivery curve with P10–P90 bands |

## API dependency

Requires the FastAPI backend running on port 8000. All data is fetched via typed calls in `src/api/client.ts`.
