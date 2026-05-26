from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.pipeline_service import pipeline

router = APIRouter()


@router.get("/daily/{date}")
def daily_forecast(date: str):
    """Hourly forecast for a specific date (YYYY-MM-DD)."""
    if not pipeline.ready:
        raise HTTPException(503, "Pipeline not ready yet")
    result = pipeline.get_daily_forecast(date)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.get("/delivery")
def delivery_summary(period: str = "monthly"):
    """Aggregated delivery-period view. period: daily | weekly | monthly"""
    if not pipeline.ready:
        raise HTTPException(503, "Pipeline not ready yet")
    if period not in ("daily", "weekly", "monthly"):
        raise HTTPException(400, "period must be daily, weekly, or monthly")
    return pipeline.get_delivery_summary(period)


@router.get("/metrics")
def model_metrics():
    """Model performance metrics (MAE, RMSE, Tail-MAE)."""
    if not pipeline.ready:
        raise HTTPException(503, "Pipeline not ready yet")
    return pipeline.get_model_metrics()


@router.get("/signal")
def trading_signal(date: Optional[str] = None):
    """Current trading signal (LONG / SHORT / FLAT)."""
    if not pipeline.ready:
        raise HTTPException(503, "Pipeline not ready yet")
    return pipeline.get_trading_signal(date)


@router.get("/commentary")
def market_commentary(date: Optional[str] = None):
    """AI-generated market commentary for a date."""
    if not pipeline.ready:
        raise HTTPException(503, "Pipeline not ready yet")
    return pipeline.generate_commentary(date)
