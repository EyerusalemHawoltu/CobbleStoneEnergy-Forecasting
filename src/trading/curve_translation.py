"""
DA → Prompt Curve Translation.

Converts hourly DA price forecasts into delivery-period views used
by the trading desk for prompt month/quarter positioning.

Delivery periods:
  - Day        : 24-hour average (base), peak (h8-h20), off-peak
  - Cal-Week   : Mon–Sun averages
  - Cal-Month  : calendar month averages
  - Cal-Quarter: Q1-Q4 averages

Signal logic (simplified risk-premium framework):
  fair_value = model_forecast_mean
  signal_z   = (fair_value - rolling_benchmark) / rolling_std
  Action      = {long if signal_z > +0.5, short if signal_z < -0.5, flat otherwise}

What invalidates the signal:
  - Sudden fundamental surprise (unplanned nuclear outage, extreme weather)
  - Liquidity drought in the prompt contract
  - Model MAE > 2× historical average (regime shift)
  - Forecast spread (p90-p10) > 2× historical spread (fat-tail event)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


PEAK_HOURS = list(range(8, 20))  # HE08–HE19, CET local time


def _local_hour(ts: pd.DatetimeIndex) -> pd.Index:
    if ts.tz is not None:
        return pd.Index(ts.tz_convert("Europe/Berlin").hour)
    return pd.Index(ts.hour)


def aggregate_to_delivery(
    forecast: pd.Series,
    period: str = "D",
) -> pd.DataFrame:
    """
    Aggregate hourly forecasts to delivery-period statistics.

    period: 'D' = daily, 'W' = weekly (Mon), 'ME' = month-end, 'QE' = quarter-end
    """
    local_hour = _local_hour(forecast.index)
    peak_mask = pd.Series(np.isin(local_hour.to_numpy(), PEAK_HOURS), index=forecast.index)

    df = forecast.to_frame("y_pred")
    df["is_peak"] = peak_mask.values

    base = df["y_pred"].resample(period).mean().rename("base_avg")
    peak = df.loc[df["is_peak"], "y_pred"].resample(period).mean().rename("peak_avg")
    offpeak = df.loc[~df["is_peak"], "y_pred"].resample(period).mean().rename("offpeak_avg")
    std = df["y_pred"].resample(period).std().rename("std")
    p10 = df["y_pred"].resample(period).quantile(0.10).rename("p10")
    p90 = df["y_pred"].resample(period).quantile(0.90).rename("p90")
    n = df["y_pred"].resample(period).count().rename("n_hours")

    out = pd.concat([base, peak, offpeak, std, p10, p90, n], axis=1)
    out["peak_base_spread"] = out["peak_avg"] - out["base_avg"]
    return out


def compute_confidence_bands(
    forecast: pd.Series,
    quantiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> pd.DataFrame:
    """Resample to daily, compute quantile bands across hours."""
    return (
        forecast.resample("D")
        .quantile(list(quantiles))
        .unstack()
        .rename(columns={q: f"q{int(q*100):02d}" for q in quantiles})
    )


def compute_risk_premium(
    forecast: pd.Series,
    actuals: pd.Series,
    rolling_window: int = 30,
) -> pd.DataFrame:
    """
    Estimate an implicit risk premium:
      risk_premium = forward_implied_price - realised_spot_average

    Here we proxy forward price with the DA forecast aggregated to delivery month.
    In production, this would be crossed against OTC/ICE forward quotes.
    """
    daily_actual = actuals.resample("D").mean()
    daily_forecast = forecast.resample("D").mean()

    rolling_benchmark = daily_actual.rolling(rolling_window, min_periods=7).mean()
    rolling_std = daily_actual.rolling(rolling_window, min_periods=7).std().replace(0, np.nan)

    df = pd.DataFrame({
        "forecast_price": daily_forecast,
        "benchmark_price": rolling_benchmark,
        "rolling_std": rolling_std,
    })
    df["signal_z"] = (df["forecast_price"] - df["benchmark_price"]) / df["rolling_std"]
    df["action"] = np.where(
        df["signal_z"] > 0.5, "LONG",
        np.where(df["signal_z"] < -0.5, "SHORT", "FLAT"),
    )
    return df


def generate_trading_view(
    forecast: pd.Series,
    actuals: pd.Series | None = None,
    periods: tuple[str, ...] = ("D", "W", "ME"),
) -> dict:
    """
    Master function: produce the full DA→curve trading view.

    Returns a dict with aggregated delivery-period DataFrames and a signal table.
    """
    view: dict = {}
    for period in periods:
        label = {"D": "daily", "W": "weekly", "ME": "monthly", "QE": "quarterly"}.get(period, period)
        view[label] = aggregate_to_delivery(forecast, period)

    view["confidence_bands"] = compute_confidence_bands(forecast)

    if actuals is not None and len(actuals) > 0:
        view["risk_premium_signal"] = compute_risk_premium(forecast, actuals)

    logger.info(
        f"Trading view generated: {', '.join(view.keys())} | "
        f"Forecast horizon: {forecast.index.min().date()} → {forecast.index.max().date()}"
    )
    return view


TRADING_DESK_RATIONALE = """
DA → Prompt Curve Translation — Desk Rationale
================================================

WHAT THE DESK DOES WITH THIS:
  • Base signal:   Compare forecast delivery-month base average to ICE EEX
                   front-month bid/offer. If forecast > midprice + 2 EUR/MWh,
                   express as long prompt-month baseload (small notional).
  • Shape signal:  Use peak/base spread to size a peak-vs-base calendar spread.
                   Wide spread → sell peak, buy base; narrow → reverse.
  • Risk sizing:   Scale position by 1/forecast_std. High uncertainty = half-size.
  • Rolling update: Forecast refreshes at gate-closure (noon D); desk reviews
                   signal vs. prior day and adjusts resting orders.

WHAT WOULD INVALIDATE THE SIGNAL:
  1. Unplanned large outage (nuclear / interconnector) in ENTSO-E transparency feed.
  2. Model MAE in live production exceeds 2× backtested average (regime flag).
  3. Forecast P90-P10 band > 40 EUR/MWh on any single day (extreme uncertainty).
  4. Market illiquidity: bid-ask > 1 EUR/MWh on the prompt month contract.
  5. Macro regime shift (e.g., gas price spike > 3× rolling average).

PROMPT CURVE EXPOSURE:
  Horizon    Instrument              Typical size
  ─────────  ──────────────────────  ────────────
  D+1        EPEX Spot block order   1–5 MW
  Week+1     EEX Week baseload       5–10 MW
  Month+1    ICE EEX front month     10–20 MW
  Quarter+1  ICE EEX front quarter   5–10 MW

  All positions are delta-1 (no optionality in this prototype).
"""
