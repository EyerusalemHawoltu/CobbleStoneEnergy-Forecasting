"""
Integration tests using synthetic data — no API keys required.
Run: python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineering import build_features, get_feature_matrix, FEATURE_COLS
from src.models.baseline import SeasonalNaive168, LinearBaseline
from src.models.forecasting import train_lgbm, walk_forward_validate, generate_forecast
from src.trading.curve_translation import aggregate_to_delivery, generate_trading_view
from src.data.qa import (
    check_missingness, check_duplicates, check_temporal_coverage,
    check_hard_limits, run_qa,
)


def make_synthetic_df(n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic hourly DE power market dataset."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days * 24, freq="h", tz="UTC")
    hour = idx.hour
    dow = idx.dayofweek

    # Realistic price: base + hourly shape + weekday/weekend + noise
    price = (
        60
        + 15 * np.sin(2 * np.pi * hour / 24)
        + 5 * (dow < 5).astype(float)
        + rng.normal(0, 8, len(idx))
    )
    load = (
        55_000
        - 10_000 * np.cos(2 * np.pi * hour / 24)
        + rng.normal(0, 2_000, len(idx))
    )
    wind = np.abs(rng.normal(15_000, 6_000, len(idx)))
    solar = np.where(
        (hour >= 6) & (hour <= 20),
        np.abs(rng.normal(10_000, 5_000, len(idx))),
        0.0,
    )

    return pd.DataFrame({
        "da_price": price,
        "load_mw": load,
        "wind_onshore_mw": wind * 0.75,
        "wind_offshore_mw": wind * 0.25,
        "solar_mw": solar,
    }, index=idx)


@pytest.fixture(scope="module")
def df_raw():
    return make_synthetic_df(n_days=400)


@pytest.fixture(scope="module")
def df_feat(df_raw):
    return build_features(df_raw)


# ── QA Tests ──────────────────────────────────────────────────────────────────

class TestQA:
    def test_missingness_clean(self, df_raw):
        result = check_missingness(df_raw)
        assert all(v["n_missing"] == 0 for v in result.values())

    def test_missingness_detects_nans(self, df_raw):
        df_dirty = df_raw.copy()
        df_dirty.iloc[10:15, 0] = np.nan
        result = check_missingness(df_dirty)
        assert result["da_price"]["n_missing"] == 5

    def test_duplicates_clean(self, df_raw):
        result = check_duplicates(df_raw)
        assert result["duplicate_index"] == 0
        assert result["duplicate_rows"] == 0

    def test_temporal_coverage(self, df_raw):
        result = check_temporal_coverage(df_raw)
        assert result["n_gaps"] == 0
        assert result["n_hours_actual"] == result["n_hours_expected"]

    def test_hard_limits(self, df_raw):
        result = check_hard_limits(df_raw)
        assert "da_price" in result
        # Synthetic prices should be within limits
        assert result["da_price"]["n_violations"] == 0

    def test_run_qa_returns_report(self, df_raw, tmp_path):
        import src.data.qa as qa_mod
        qa_mod.REPORT_DIR = tmp_path
        report = run_qa(df_raw, tag="test")
        assert "missingness" in report
        assert "duplicates" in report
        assert "temporal_coverage" in report


# ── Feature Engineering Tests ─────────────────────────────────────────────────

class TestFeatures:
    def test_feature_columns_present(self, df_feat):
        expected = ["hour", "dow", "month", "price_lag_24h", "price_lag_168h", "wind_total_mw"]
        for col in expected:
            assert col in df_feat.columns, f"Missing feature: {col}"

    def test_no_leakage_lag(self, df_feat):
        # price_lag_24h at row i == da_price at row i-24
        df = df_feat.dropna(subset=["price_lag_24h", "da_price"])
        for i in range(24, min(50, len(df))):
            lag_val = df["price_lag_24h"].iloc[i]
            orig_val = df["da_price"].iloc[i - 24]
            assert abs(lag_val - orig_val) < 1e-6, "Leakage: lag_24h does not match da_price 24h earlier"

    def test_wind_total(self, df_feat):
        assert (df_feat["wind_total_mw"] >= 0).all()

    def test_renewable_penetration_range(self, df_feat):
        pen = df_feat["ren_pen"].dropna()
        assert (pen >= 0).all()
        assert (pen <= 5).all()  # allows over-generation, generous bound

    def test_get_feature_matrix_no_nan(self, df_feat):
        X, y = get_feature_matrix(df_feat)
        assert X.isna().sum().sum() == 0
        assert y.isna().sum() == 0


# ── Model Tests ───────────────────────────────────────────────────────────────

class TestModels:
    def test_seasonal_naive_predict(self, df_feat):
        X, y = get_feature_matrix(df_feat)
        X_train, y_train = X.iloc[:-168], y.iloc[:-168]
        X_test = X.iloc[-168:]
        model = SeasonalNaive168().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert not np.isnan(preds).any()

    def test_linear_baseline(self, df_feat):
        X, y = get_feature_matrix(df_feat)
        X_train, y_train = X.iloc[:-168], y.iloc[:-168]
        X_test = X.iloc[-168:]
        model = LinearBaseline().fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)

    def test_lgbm_train_predict(self, df_feat):
        X, y = get_feature_matrix(df_feat)
        cols = list(X.columns)
        model = train_lgbm(X.iloc[:2000], y.iloc[:2000], cols)
        preds = model.predict(X.iloc[2000:2100])
        assert len(preds) == 100
        assert not np.isnan(preds).any()

    def test_walk_forward_returns_metrics(self, df_feat):
        result = walk_forward_validate(df_feat, min_train_days=180, retrain_freq_days=7)
        assert "mae" in result.metrics
        assert result.metrics["mae"] > 0
        assert len(result.predictions) > 0

    def test_lgbm_beats_naive_mae(self, df_feat):
        result = walk_forward_validate(df_feat, min_train_days=180, retrain_freq_days=7)
        # Align predictions and actuals
        pred, act = result.predictions.align(result.actuals, join="inner")
        lgbm_mae = float((pred - act).abs().mean())

        # Naive on same period
        X, y = get_feature_matrix(df_feat)
        test_mask = X.index.isin(act.index)
        X_test = X.loc[test_mask]
        naive = SeasonalNaive168()
        naive._fitted = True
        naive_preds = naive.predict(X_test)
        naive_mae = float(np.abs(naive_preds - act.values).mean())

        assert lgbm_mae < naive_mae, (
            f"LightGBM MAE ({lgbm_mae:.2f}) should be < Naive MAE ({naive_mae:.2f})"
        )


# ── Curve Translation Tests ───────────────────────────────────────────────────

class TestCurveTranslation:
    def get_forecast(self, df_feat):
        X, y = get_feature_matrix(df_feat)
        cols = list(X.columns)
        model = train_lgbm(X.iloc[:3000], y.iloc[:3000], cols)
        return generate_forecast(model, X.iloc[3000:], cols)

    def test_aggregate_daily(self, df_feat):
        fc = self.get_forecast(df_feat)
        daily = aggregate_to_delivery(fc, period="D")
        assert "base_avg" in daily.columns
        assert "peak_avg" in daily.columns
        assert "peak_base_spread" in daily.columns
        assert len(daily) > 0

    def test_aggregate_monthly(self, df_feat):
        fc = self.get_forecast(df_feat)
        monthly = aggregate_to_delivery(fc, period="ME")
        assert len(monthly) >= 1

    def test_generate_trading_view(self, df_feat):
        X, y = get_feature_matrix(df_feat)
        cols = list(X.columns)
        model = train_lgbm(X.iloc[:3000], y.iloc[:3000], cols)
        fc = generate_forecast(model, X.iloc[3000:], cols)
        view = generate_trading_view(fc, actuals=y.iloc[3000:])
        assert "daily" in view
        assert "monthly" in view
        assert "confidence_bands" in view
        assert "risk_premium_signal" in view

    def test_trading_signal_values(self, df_feat):
        X, y = get_feature_matrix(df_feat)
        cols = list(X.columns)
        model = train_lgbm(X.iloc[:3000], y.iloc[:3000], cols)
        fc = generate_forecast(model, X.iloc[3000:], cols)
        view = generate_trading_view(fc, actuals=y.iloc[3000:])
        actions = view["risk_premium_signal"]["action"]
        assert set(actions.dropna()).issubset({"LONG", "SHORT", "FLAT"})
