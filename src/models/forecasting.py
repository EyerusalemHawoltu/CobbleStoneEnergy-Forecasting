"""
LightGBM forecasting model with walk-forward (expanding-window) validation.

Architecture:
  - Single model, hour-of-day included as an ordinal feature (not 24 separate models)
  - Expanding training window; re-trained every RETRAIN_FREQ days
  - Horizon: 24 h ahead (full next-day DA schedule)

Leakage guard: the model is only trained on data whose INDEX is at least
  24 hours before the first prediction timestamp in each fold, matching
  the real gate-closure constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.engineering import FEATURE_COLS, TARGET_COL

RETRAIN_FREQ_DAYS = 7        # re-fit weekly during walk-forward
MIN_TRAIN_DAYS = 180         # at least 6 months before first prediction
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


@dataclass
class ForecastResult:
    predictions: pd.Series                   # index: utc_timestamp
    actuals: pd.Series
    model_params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    errors = y_true - y_pred
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    # Tail metric: mean absolute error for the top-10% price hours
    threshold = np.percentile(np.abs(y_true), 90)
    tail_mask = np.abs(y_true) >= threshold
    tail_mae = float(np.mean(np.abs(errors[tail_mask]))) if tail_mask.any() else np.nan
    mbe = float(np.mean(errors))
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "tail_mae_p90": round(tail_mae, 2),
        "mbe": round(mbe, 2),
        "n_samples": int(len(y_true)),
    }


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: Optional[list[str]] = None,
) -> lgb.LGBMRegressor:
    cols = feature_cols or [c for c in FEATURE_COLS if c in X_train.columns]
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train[cols], y_train)
    return model


def walk_forward_validate(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    min_train_days: int = MIN_TRAIN_DAYS,
    retrain_freq_days: int = RETRAIN_FREQ_DAYS,
) -> ForecastResult:
    """
    Expanding-window walk-forward validation.

    At each step we:
      1. Train on everything strictly before the current window.
      2. Predict 24 h (one DA day) ahead.
      3. Advance by retrain_freq_days.

    Returns aligned predictions and actuals for the entire validation window.
    """
    cols = feature_cols or [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[cols + [TARGET_COL]].dropna()

    first_pred_date = df_clean.index.min() + pd.Timedelta(days=min_train_days)
    pred_dates = pd.date_range(
        start=first_pred_date.normalize(),
        end=df_clean.index.max().normalize(),
        freq=f"{retrain_freq_days}D",
    )

    all_preds: list[pd.Series] = []
    all_actuals: list[pd.Series] = []

    for pred_start in pred_dates:
        pred_end = pred_start + pd.Timedelta(days=retrain_freq_days)
        cutoff = pred_start - pd.Timedelta(hours=24)  # gate-closure

        train_mask = df_clean.index < cutoff
        test_mask = (df_clean.index >= pred_start) & (df_clean.index < pred_end)

        X_train = df_clean.loc[train_mask, cols]
        y_train = df_clean.loc[train_mask, TARGET_COL]
        X_test = df_clean.loc[test_mask, cols]
        y_test = df_clean.loc[test_mask, TARGET_COL]

        if len(X_train) < min_train_days * 24 or len(X_test) == 0:
            continue

        model = train_lgbm(X_train, y_train, cols)
        preds = pd.Series(model.predict(X_test), index=X_test.index, name="y_pred")

        all_preds.append(preds)
        all_actuals.append(y_test.rename("y_actual"))

    if not all_preds:
        raise RuntimeError("No walk-forward folds produced — check data coverage.")

    predictions = pd.concat(all_preds).sort_index()
    actuals = pd.concat(all_actuals).sort_index()

    aligned_pred, aligned_act = predictions.align(actuals, join="inner")
    met = _metrics(aligned_act.values, aligned_pred.values)
    logger.info(f"Walk-forward CV — MAE: {met['mae']:.2f}, RMSE: {met['rmse']:.2f}, Tail MAE(P90): {met['tail_mae_p90']:.2f}")

    return ForecastResult(
        predictions=predictions,
        actuals=actuals,
        model_params=LGBM_PARAMS,
        metrics=met,
    )


def train_final_model(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    cutoff: Optional[pd.Timestamp] = None,
) -> tuple[lgb.LGBMRegressor, list[str]]:
    """Train on all data up to `cutoff` (or all data if None)."""
    cols = feature_cols or [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[cols + [TARGET_COL]].dropna()
    if cutoff is not None:
        df_clean = df_clean.loc[df_clean.index < cutoff]
    model = train_lgbm(df_clean[cols], df_clean[TARGET_COL], cols)
    logger.info(f"Final model trained on {len(df_clean):,} samples up to {cutoff or df_clean.index.max()}")
    return model, cols


def generate_forecast(
    model: lgb.LGBMRegressor,
    df_features: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Produce point forecasts for a given feature matrix."""
    X = df_features[feature_cols].dropna()
    preds = model.predict(X)
    return pd.Series(preds, index=X.index, name="y_pred")


def evaluate_baseline(
    baseline_model,
    df: pd.DataFrame,
    test_mask: pd.Series,
    feature_cols: list[str],
) -> dict:
    cols = [c for c in feature_cols if c in df.columns]
    df_clean = df[cols + [TARGET_COL]].dropna()
    X_test = df_clean.loc[test_mask & df_clean.index.isin(df_clean.index), cols]
    y_test = df_clean.loc[X_test.index, TARGET_COL]
    preds = baseline_model.predict(X_test)
    return _metrics(y_test.values, preds)
