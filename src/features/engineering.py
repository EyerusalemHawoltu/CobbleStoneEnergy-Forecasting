"""
Feature engineering for German DA power price forecasting.

Design principle: only use information that would be known at gate-closure
(noon on day D) when forecasting all hours of day D+1. Lagged price features
use at least 24h lags; fundamentals use day-ahead published forecasts or
same-day actuals shifted by 24h to prevent leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "da_price"

FEATURE_COLS = [
    "hour",
    "dow",
    "month",
    "doy",
    "is_weekend",
    "is_summer",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_roll7d_mean",
    "price_roll7d_std",
    "price_roll30d_mean",
    "wind_total_mw",
    "solar_mw",
    "load_mw",
    "wind_pen",
    "solar_pen",
    "ren_pen",
    "residual_load_mw",
    "wind_roll7d_mean",
    "load_roll7d_mean",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    if idx.tz is not None:
        local_idx = idx.tz_convert("Europe/Berlin")
    else:
        local_idx = idx
    df = df.copy()
    df["hour"] = local_idx.hour
    df["dow"] = local_idx.dayofweek          # 0=Mon
    df["month"] = local_idx.month
    df["doy"] = local_idx.dayofyear
    df["is_weekend"] = (local_idx.dayofweek >= 5).astype(int)
    df["is_summer"] = local_idx.month.isin([5, 6, 7, 8, 9]).astype(int)
    return df


def add_price_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_lag_24h"] = df[TARGET_COL].shift(24)
    df["price_lag_48h"] = df[TARGET_COL].shift(48)
    df["price_lag_168h"] = df[TARGET_COL].shift(168)
    df["price_roll7d_mean"] = (
        df[TARGET_COL].shift(24).rolling(168).mean()
    )
    df["price_roll7d_std"] = (
        df[TARGET_COL].shift(24).rolling(168).std()
    )
    df["price_roll30d_mean"] = (
        df[TARGET_COL].shift(24).rolling(720).mean()
    )
    return df


def add_renewable_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_wind = "wind_onshore_mw" in df.columns or "wind_offshore_mw" in df.columns
    has_solar = "solar_mw" in df.columns
    load = df.get("load_mw", pd.Series(np.nan, index=df.index))

    if has_wind:
        if "wind_onshore_mw" in df.columns and "wind_offshore_mw" in df.columns:
            df["wind_total_mw"] = df["wind_onshore_mw"].fillna(0) + df["wind_offshore_mw"].fillna(0)
        else:
            df["wind_total_mw"] = df["wind_onshore_mw"].fillna(0)

        solar = df["solar_mw"].fillna(0) if has_solar else pd.Series(0, index=df.index)
        df["wind_pen"] = (df["wind_total_mw"] / load.replace(0, np.nan)).clip(0, 1)
        df["solar_pen"] = (solar / load.replace(0, np.nan)).clip(0, 1)
        df["ren_pen"] = ((df["wind_total_mw"] + solar) / load.replace(0, np.nan)).clip(0, 1)
        df["residual_load_mw"] = load - df["wind_total_mw"] - solar
        df["wind_roll7d_mean"] = df["wind_total_mw"].shift(24).rolling(168).mean()

    # Lagged load feature (shift 24h to avoid leakage)
    if "load_mw" in df.columns:
        df["load_roll7d_mean"] = df["load_mw"].shift(24).rolling(168).mean()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature transformations and return the enriched DataFrame."""
    df = add_calendar_features(df)
    df = add_price_lags(df)
    df = add_renewable_features(df)
    return df


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) dropping rows with NaN in any feature or target."""
    cols = feature_cols or FEATURE_COLS
    available = [c for c in cols if c in df.columns]
    subset = df[available + [TARGET_COL]].dropna()
    X = subset[available]
    y = subset[TARGET_COL]
    return X, y
