"""
Pipeline Service — initialises the forecasting pipeline once and serves
results to the FastAPI routers.

Demo mode (no ENTSO-E key): uses synthetic data that mirrors real DE market patterns.
Production mode: downloads real ENTSO-E data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from src.features.engineering import build_features, get_feature_matrix, FEATURE_COLS, TARGET_COL
from src.models.baseline import SeasonalNaive168, LinearBaseline
from src.models.forecasting import (
    walk_forward_validate,
    train_final_model,
    generate_forecast,
)
from src.trading.curve_translation import generate_trading_view, aggregate_to_delivery
from src.data.qa import run_qa, summarise_qa


def _make_synthetic_data(n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    """Realistic synthetic DE power market dataset (no API key needed)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days * 24, freq="h", tz="UTC")
    hour = np.array(idx.hour, dtype=float)
    dow = np.array(idx.dayofweek, dtype=float)
    month = np.array(idx.month, dtype=float)

    # Realistic hourly price shape
    hourly_shape = 12 * np.sin(np.pi * (hour - 6) / 12)
    seasonal = 10 * np.cos(2 * np.pi * (month - 1) / 12)
    weekday_premium = 5 * (dow < 5).astype(float)
    price = 65 + hourly_shape + seasonal + weekday_premium + rng.normal(0, 9, len(idx))
    price = np.clip(price, -50, 400)

    load = 55_000 - 10_000 * np.cos(2 * np.pi * hour / 24) + rng.normal(0, 2_500, len(idx))
    wind_base = np.abs(rng.normal(14_000, 7_000, len(idx)))
    solar = np.where(
        (hour >= 6) & (hour <= 19),
        np.abs(rng.normal(12_000, 6_000, len(idx))),
        0.0,
    )

    return pd.DataFrame({
        "da_price": price,
        "load_mw": np.clip(load, 20_000, 85_000),
        "wind_onshore_mw": wind_base * 0.75,
        "wind_offshore_mw": wind_base * 0.25,
        "solar_mw": solar,
    }, index=idx)


class PipelineService:
    """Singleton-style service that holds pipeline state."""

    def __init__(self) -> None:
        self.ready = False
        self.demo_mode = True
        self.df_raw: pd.DataFrame | None = None
        self.df_feat: pd.DataFrame | None = None
        self.predictions: pd.Series | None = None
        self.actuals: pd.Series | None = None
        self.model = None
        self.feature_cols: list[str] = []
        self.metrics: dict = {}
        self.qa_report: dict = {}
        self.trading_view: dict = {}
        self.cv_result = None

    # ── Initialisation ─────────────────────────────────────────────────────────

    def initialise(self, force_demo: bool = False) -> None:
        logger.info("PipelineService: initialising…")
        from src.data.ingestion import (
            fetch_all_data,
            detect_local_data_end,
            _local_price_files_exist,
            _local_load_files_exist,
        )

        start = pd.Timestamp("2024-01-01", tz="UTC")
        # Auto-detect latest available date from local CSVs
        detected_end = detect_local_data_end()
        end = detected_end if detected_end is not None else pd.Timestamp("2026-05-27", tz="UTC")
        logger.info(f"Data range: {start.date()} → {end.date()}")

        has_api_key = bool(os.environ.get("ENTSOE_API_KEY"))
        has_local = _local_price_files_exist(start, end) and _local_load_files_exist(start, end)
        self.demo_mode = force_demo or (not has_api_key and not has_local)

        if self.demo_mode:
            logger.info("Demo mode: using synthetic data (no API key and no local CSV files found)")
            self.df_raw = _make_synthetic_data(n_days=420)
        else:
            logger.info("Loading data (local CSVs or ENTSO-E API)…")
            try:
                self.df_raw = fetch_all_data(start=start, end=end, force_refresh=True)
            except Exception as exc:
                logger.warning(f"Data load failed ({exc}) — falling back to demo mode")
                self.demo_mode = True
                self.df_raw = _make_synthetic_data(n_days=420)

        # QA
        from src.ai.llm_component import propose_qa_rules
        llm_rules = propose_qa_rules(self.df_raw) if os.environ.get("GROQ_API_KEY") else []
        self.qa_report = run_qa(self.df_raw, llm_rules=llm_rules or None, tag="service")

        # Features
        df_filled = self.df_raw.ffill(limit=4)
        self.df_feat = build_features(df_filled)

        # Split: last 90 days = test
        last_date = self.df_feat.index.max()
        test_start = last_date - pd.Timedelta(days=90)
        df_train = self.df_feat.loc[self.df_feat.index < test_start]
        df_test = self.df_feat.loc[self.df_feat.index >= test_start]

        # Walk-forward CV on training set
        logger.info("Running walk-forward CV…")
        self.cv_result = walk_forward_validate(df_train, min_train_days=150, retrain_freq_days=7)

        # Baselines on test
        X_all, y_all = get_feature_matrix(self.df_feat)
        test_mask = X_all.index >= test_start
        X_train_bl, y_train_bl = X_all.loc[~test_mask], y_all.loc[~test_mask]
        X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]

        naive = SeasonalNaive168().fit(X_train_bl, y_train_bl)
        naive_preds = pd.Series(naive.predict(X_test), index=X_test.index)

        linear = LinearBaseline().fit(X_train_bl, y_train_bl)
        linear_preds = pd.Series(linear.predict(X_test), index=X_test.index)

        # Final LightGBM model
        self.model, self.feature_cols = train_final_model(self.df_feat, cutoff=test_start)
        self.predictions = generate_forecast(self.model, X_test, self.feature_cols)
        self.actuals = y_test

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        def _met(yt, yp, name):
            mae = float(mean_absolute_error(yt, yp))
            rmse = float(mean_squared_error(yt, yp) ** 0.5)
            tail = float(np.mean(np.abs(yt - yp)[np.abs(yt) >= np.percentile(np.abs(yt), 90)]))
            return {"name": name, "mae": round(mae, 2), "rmse": round(rmse, 2), "tail_mae_p90": round(tail, 2)}

        self.metrics = {
            "cv": self.cv_result.metrics,
            "test": {
                "lgbm": _met(y_test.values, self.predictions.values, "LightGBM"),
                "naive": _met(y_test.values, naive_preds.values, "Seasonal Naive 168h"),
                "linear": _met(y_test.values, linear_preds.values, "Ridge Linear"),
            },
            "test_period": {
                "start": str(test_start.date()),
                "end": str(last_date.date()),
                "n_hours": int(len(y_test)),
            },
            "demo_mode": self.demo_mode,
        }

        # Curve view
        self.trading_view = generate_trading_view(
            self.predictions, actuals=self.actuals
        )

        self.ready = True
        logger.success(
            f"PipelineService ready — "
            f"LightGBM test MAE={self.metrics['test']['lgbm']['mae']:.2f} EUR/MWh"
        )

    # ── Query methods ──────────────────────────────────────────────────────────

    def _build_future_features(self, target: pd.Timestamp) -> pd.DataFrame:
        """
        Construct hourly feature rows for a date beyond the loaded data window.
        Uses published price lags and DA load forecasts from df_raw / df_feat.
        """
        from src.features.engineering import add_calendar_features

        hours = pd.date_range(target, periods=24, freq="h", tz="UTC")
        df_future = pd.DataFrame(index=hours)
        df_future.index.name = "utc_timestamp"

        # Calendar features
        df_future = add_calendar_features(df_future)

        # Price lags from historical df_feat
        hist_price = self.df_feat["da_price"] if self.df_feat is not None else pd.Series(dtype=float)
        for h in hours:
            for lag_h, col in [(24, "price_lag_24h"), (48, "price_lag_48h"), (168, "price_lag_168h")]:
                ts_lag = h - pd.Timedelta(hours=lag_h)
                if ts_lag in hist_price.index:
                    df_future.loc[h, col] = hist_price.loc[ts_lag]
                elif not hist_price.empty:
                    # Use nearest available past value as fallback
                    past = hist_price[hist_price.index <= ts_lag]
                    if not past.empty:
                        df_future.loc[h, col] = past.iloc[-1]

        # Rolling price features from recent history (last 30 days)
        recent = hist_price.tail(24 * 30) if len(hist_price) >= 24 else hist_price
        df_future["price_roll7d_mean"] = recent.tail(24 * 7).mean()
        df_future["price_roll7d_std"] = recent.tail(24 * 7).std()
        df_future["price_roll30d_mean"] = recent.mean()

        # Load: use DA load forecast from df_raw if available, else rolling mean
        if self.df_raw is not None and "load_forecast_mw" in self.df_raw.columns:
            for h in hours:
                if h in self.df_raw.index and not np.isnan(self.df_raw.loc[h, "load_forecast_mw"]):
                    df_future.loc[h, "load_mw"] = self.df_raw.loc[h, "load_forecast_mw"]
                    df_future.loc[h, "load_forecast_mw"] = self.df_raw.loc[h, "load_forecast_mw"]
        # Fill any missing load values (including when target date is beyond df_raw)
        if self.df_raw is not None and "load_mw" in self.df_raw.columns:
            load_fallback = float(self.df_raw["load_mw"].tail(24 * 7).mean())
        elif self.df_feat is not None and "load_mw" in self.df_feat.columns:
            load_fallback = float(self.df_feat["load_mw"].tail(24 * 7).mean())
        else:
            load_fallback = 55000.0
        if "load_mw" not in df_future.columns:
            df_future["load_mw"] = load_fallback
        else:
            df_future["load_mw"] = df_future["load_mw"].fillna(load_fallback)
        if "load_forecast_mw" not in df_future.columns:
            df_future["load_forecast_mw"] = load_fallback
        else:
            df_future["load_forecast_mw"] = df_future["load_forecast_mw"].fillna(load_fallback)

        # Lagged load rolling mean
        if self.df_feat is not None and "load_mw" in self.df_feat.columns:
            df_future["load_roll7d_mean"] = float(self.df_feat["load_mw"].tail(24 * 7).mean())

        # Wind+solar DA forecast: use published values from df_raw if available
        if self.df_raw is not None and "wind_solar_da_mw" in self.df_raw.columns:
            ws_fallback = float(self.df_raw["wind_solar_da_mw"].tail(24 * 7).mean())
            for h in hours:
                if h in self.df_raw.index:
                    val = self.df_raw.loc[h, "wind_solar_da_mw"]
                    if not (isinstance(val, float) and np.isnan(val)):
                        df_future.loc[h, "wind_solar_da_mw"] = val
            if "wind_solar_da_mw" not in df_future.columns:
                df_future["wind_solar_da_mw"] = ws_fallback
            else:
                df_future["wind_solar_da_mw"] = df_future["wind_solar_da_mw"].fillna(ws_fallback)
        elif self.df_feat is not None and "wind_solar_da_mw" in self.df_feat.columns:
            df_future["wind_solar_da_mw"] = float(self.df_feat["wind_solar_da_mw"].tail(24 * 7).mean())

        # Derived renewable features for future rows
        if "wind_solar_da_mw" in df_future.columns and "load_mw" in df_future.columns:
            ws = df_future["wind_solar_da_mw"].fillna(0)
            load = df_future["load_mw"].replace(0, np.nan)
            df_future["ren_pen"] = (ws / load).clip(0, 1)
            df_future["residual_load_mw"] = df_future["load_mw"] - ws
        if self.df_feat is not None and "wind_solar_da_mw" in self.df_feat.columns:
            df_future["wind_solar_roll7d_mean"] = float(
                self.df_feat["wind_solar_da_mw"].tail(24 * 7).mean()
            )

        return df_future

    def get_daily_forecast(self, date_str: str) -> dict:
        """Hourly forecast for a calendar day. Works for any date: historical, recent, or future."""
        self._check_ready()
        try:
            target = pd.Timestamp(date_str, tz="UTC")
        except Exception:
            return {"error": f"Invalid date: {date_str}"}

        is_future_fc = False
        mask = self.predictions.index.normalize() == target.normalize()
        if not mask.any():
            # Try features already in df_feat first
            feat_mask = self.df_feat.index.normalize() == target.normalize() if self.df_feat is not None else pd.Series(False)
            X_day = self.df_feat.loc[feat_mask, self.feature_cols].dropna() if feat_mask.any() else pd.DataFrame()

            if len(X_day) == 0:
                # Build features for a truly future date
                X_day_raw = self._build_future_features(target)
                available_cols = [c for c in self.feature_cols if c in X_day_raw.columns]
                X_day = X_day_raw[available_cols].dropna()
                is_future_fc = True

            if len(X_day) == 0:
                return {"error": f"Cannot construct features for {date_str} — insufficient history"}

            preds = generate_forecast(self.model, X_day, [c for c in self.feature_cols if c in X_day.columns])
            day_fc = preds
        else:
            day_fc = self.predictions.loc[mask]

        actual_mask = self.actuals.index.normalize() == target.normalize()
        day_actual = self.actuals.loc[actual_mask] if actual_mask.any() else None

        local_idx = day_fc.index.tz_convert("Europe/Berlin")
        peak_mask = np.isin(np.array(local_idx.hour), list(range(8, 20)))

        records = [
            {
                "hour": int(h),
                "utc_timestamp": str(ts),
                "forecast": round(float(p), 2),
                "actual": round(float(day_actual.loc[day_fc.index[i]]), 2)
                if day_actual is not None and day_fc.index[i] in day_actual.index
                else None,
                "is_peak": bool(peak_mask[i]),
            }
            for i, (ts, p, h) in enumerate(
                zip(day_fc.index, day_fc.values, local_idx.hour)
            )
        ]
        base = float(day_fc.mean())
        peak = float(day_fc.values[peak_mask].mean()) if peak_mask.any() else None
        offpeak = float(day_fc.values[~peak_mask].mean()) if (~peak_mask).any() else None

        return {
            "date": date_str,
            "is_projected": is_future_fc,
            "summary": {
                "base_avg": round(base, 2),
                "peak_avg": round(peak, 2) if peak is not None else None,
                "offpeak_avg": round(offpeak, 2) if offpeak is not None else None,
                "p10": round(float(day_fc.quantile(0.10)), 2),
                "p90": round(float(day_fc.quantile(0.90)), 2),
            },
            "hourly": records,
        }

    def get_delivery_summary(self, period: str = "monthly") -> dict:
        """Delivery-period base/peak/offpeak averages."""
        self._check_ready()
        freq_map = {"daily": "D", "weekly": "W", "monthly": "ME"}
        freq = freq_map.get(period, "ME")
        df = aggregate_to_delivery(self.predictions, period=freq)
        records = []
        for ts, row in df.iterrows():
            records.append({
                "period": str(ts.date()),
                "base_avg": round(float(row["base_avg"]), 2) if not np.isnan(row["base_avg"]) else None,
                "peak_avg": round(float(row["peak_avg"]), 2) if not np.isnan(row.get("peak_avg", float("nan"))) else None,
                "offpeak_avg": round(float(row["offpeak_avg"]), 2) if not np.isnan(row.get("offpeak_avg", float("nan"))) else None,
                "p10": round(float(row["p10"]), 2) if not np.isnan(row.get("p10", float("nan"))) else None,
                "p90": round(float(row["p90"]), 2) if not np.isnan(row.get("p90", float("nan"))) else None,
            })
        return {"period_type": period, "data": records}

    def get_model_metrics(self) -> dict:
        self._check_ready()
        return self.metrics

    def get_qa_summary(self) -> dict:
        self._check_ready()
        return {
            "summary_text": summarise_qa(self.qa_report),
            "missingness": self.qa_report.get("missingness", {}),
            "duplicates": self.qa_report.get("duplicates", {}),
            "temporal_coverage": self.qa_report.get("temporal_coverage", {}),
            "hard_limits": self.qa_report.get("hard_limits", {}),
            "llm_rules": self.qa_report.get("llm_rules", {}).get("llm_rules", []),
        }

    def get_trading_signal(self, date_str: str | None = None) -> dict:
        self._check_ready()
        signal_df = self.trading_view.get("risk_premium_signal")
        if signal_df is None or len(signal_df) == 0:
            return {"error": "No signal data available"}
        if date_str:
            try:
                target = pd.Timestamp(date_str).normalize()
                row = signal_df.loc[signal_df.index.normalize() == target]
                if len(row) == 0:
                    row = signal_df.tail(1)
            except Exception:
                row = signal_df.tail(1)
        else:
            row = signal_df.tail(5)

        records = []
        for ts, r in row.iterrows():
            records.append({
                "date": str(ts.date()),
                "forecast_price": round(float(r["forecast_price"]), 2) if not np.isnan(r["forecast_price"]) else None,
                "benchmark_price": round(float(r["benchmark_price"]), 2) if not np.isnan(r["benchmark_price"]) else None,
                "signal_z": round(float(r["signal_z"]), 3) if not np.isnan(r["signal_z"]) else None,
                "action": r["action"],
            })
        return {"signals": records}

    def generate_commentary(self, date_str: str | None = None) -> dict:
        self._check_ready()
        from src.ai.llm_component import generate_daily_commentary, build_commentary_metrics
        if date_str:
            try:
                target = pd.Timestamp(date_str, tz="UTC")
            except Exception:
                target = self.predictions.index.normalize().unique()[-1]
        else:
            target = self.predictions.index.normalize().unique()[-1]

        metrics = build_commentary_metrics(
            self.predictions, self.actuals, self.df_feat, target
        )
        commentary = generate_daily_commentary(metrics)
        return {"date": str(target.date()), "commentary": commentary, "metrics": metrics}

    def _check_ready(self):
        if not self.ready:
            raise RuntimeError("Pipeline not initialised. Call initialise() first.")


# Module-level singleton
pipeline = PipelineService()
