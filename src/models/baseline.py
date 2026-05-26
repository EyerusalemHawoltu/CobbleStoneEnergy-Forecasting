"""
Baseline forecasting models for German DA power prices.

Baselines:
  1. SeasonalNaive168   — last-week-same-hour (168h lag)
  2. LinearBaseline      — OLS on calendar + lag features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "da_price"
CALENDAR_FEATURES = ["hour", "dow", "month", "is_weekend", "is_summer", "doy"]
LINEAR_FEATURES = CALENDAR_FEATURES + [
    "price_lag_24h",
    "price_lag_168h",
    "price_roll7d_mean",
    "wind_total_mw",
    "solar_mw",
    "load_mw",
]


class SeasonalNaive168:
    """Predict next hour's DA price = same hour exactly 168 hours (1 week) ago."""

    name = "seasonal_naive_168h"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive168":
        # No fitting needed; history is baked into the lag feature
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "price_lag_168h" not in X.columns:
            raise ValueError("price_lag_168h feature required for SeasonalNaive168")
        return X["price_lag_168h"].values


class LinearBaseline:
    """Ridge regression on calendar + fundamental + lag features."""

    name = "ridge_linear"

    def __init__(self, alpha: float = 10.0) -> None:
        self._model = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearBaseline":
        cols = [c for c in LINEAR_FEATURES if c in X.columns]
        self._cols = cols
        self._model.fit(X[cols], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X[self._cols])
