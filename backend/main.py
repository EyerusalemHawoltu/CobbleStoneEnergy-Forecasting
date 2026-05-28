"""
FastAPI backend — Cobblestone Energy Power Forecasting Pipeline.

Start with: uvicorn backend.main:app --reload
Or:         python backend/main.py
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from backend.routers import forecast, qa, chat as chat_router
from backend.services.pipeline_service import pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run the heavy pipeline init (~7 min) in a background thread so the server
    # starts accepting requests immediately (Railway health check, frontend polling).
    logger.info("Starting up — spawning pipeline initialisation in background thread…")

    def _bg_init():
        try:
            pipeline.initialise()
        except Exception as exc:
            logger.error(f"Pipeline init failed: {exc}. App running in degraded state.")

    threading.Thread(target=_bg_init, daemon=True).start()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Cobblestone Energy — DE Power Forecasting API",
    description=(
        "Day-ahead price forecasting for the German power market. "
        "Data: ENTSO-E Transparency. AI: Groq (free Llama 3 models)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router, prefix="/api/forecast", tags=["Forecast"])
app.include_router(qa.router, prefix="/api/qa", tags=["Data Quality"])
app.include_router(chat_router.router, prefix="/api/chat", tags=["AI Chat"])


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline.ready,
        "demo_mode": pipeline.demo_mode,
    }


# Serve React build (production)
frontend_build = ROOT / "frontend" / "dist"
if frontend_build.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
