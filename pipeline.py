#!/usr/bin/env python3
"""
CobbleStoneEnergy Forecasting Pipeline
German (DE) Day-Ahead Power Price Forecasting → Prompt Curve Translation

Usage:
    python pipeline.py --start 2022-01-01 --end 2024-12-31 --test-start 2024-07-01
    python pipeline.py --skip-download        # use cached data only
    python pipeline.py --no-llm              # skip AI component (no API key needed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from loguru import logger

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

load_dotenv()

from src.data.ingestion import fetch_all_data
from src.data.qa import run_qa, summarise_qa
from src.features.engineering import build_features, get_feature_matrix, FEATURE_COLS, TARGET_COL
from src.models.baseline import SeasonalNaive168, LinearBaseline
from src.models.forecasting import (
    walk_forward_validate,
    train_final_model,
    generate_forecast,
)
from src.trading.curve_translation import (
    generate_trading_view,
    TRADING_DESK_RATIONALE,
)
from src.ai.llm_component import (
    propose_qa_rules,
    generate_daily_commentary,
    build_commentary_metrics,
)

FIGURES_DIR = ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = ROOT / "outputs" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger.add(LOGS_DIR / "pipeline.log", rotation="10 MB", level="INFO")


# ── Plotting helpers ──────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")


def plot_price_series(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    local_idx = df.index.tz_convert("Europe/Berlin")

    axes[0].plot(local_idx, df["da_price"], linewidth=0.4, alpha=0.8, color="#1f77b4")
    axes[0].set_ylabel("DA Price (EUR/MWh)")
    axes[0].set_title("German Day-Ahead Electricity Price", fontweight="bold")

    axes[1].stackplot(
        local_idx,
        df.get("wind_onshore_mw", 0),
        df.get("wind_offshore_mw", 0),
        df.get("solar_mw", 0),
        labels=["Wind Onshore", "Wind Offshore", "Solar"],
        alpha=0.75,
        colors=["#2ca02c", "#98df8a", "#ffbb78"],
    )
    axes[1].set_ylabel("Generation (MW)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_title("Renewable Generation", fontweight="bold")

    axes[2].plot(local_idx, df.get("load_mw", np.nan), linewidth=0.4, color="#d62728")
    axes[2].set_ylabel("Load (MW)")
    axes[2].set_title("Total Load", fontweight="bold")

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved → {out}")


def plot_forecast_vs_actual(
    result,
    baseline_preds: pd.Series,
    out: Path,
    window_days: int = 14,
) -> None:
    idx = result.actuals.index[-window_days * 24 :]
    local_idx = idx.tz_convert("Europe/Berlin")
    actual = result.actuals.loc[idx]
    lgbm_pred = result.predictions.reindex(idx)
    naive_pred = baseline_preds.reindex(idx)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(local_idx, actual.values, label="Actual DA Price", linewidth=1.5, color="#1f77b4")
    ax.plot(local_idx, lgbm_pred.values, label="LightGBM Forecast", linewidth=1.0, linestyle="--", color="#ff7f0e")
    ax.plot(local_idx, naive_pred.values, label="Seasonal Naive (168h)", linewidth=1.0, linestyle=":", color="#2ca02c")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"Forecast vs Actual — last {window_days} days of validation window", fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved → {out}")


def plot_delivery_curve(view: dict, out: Path) -> None:
    monthly = view.get("monthly")
    if monthly is None or len(monthly) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(monthly.index, monthly["base_avg"], color="#1f77b4", alpha=0.7, label="Base avg")
    ax.bar(monthly.index, monthly["peak_avg"] - monthly["base_avg"],
           bottom=monthly["base_avg"], color="#ff7f0e", alpha=0.7, label="Peak premium")
    ax.fill_between(monthly.index, monthly["p10"], monthly["p90"],
                    alpha=0.2, color="#1f77b4", label="P10–P90 band")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Monthly Delivery Curve View (Forecast)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved → {out}")


def plot_feature_importance(model, feature_cols: list[str], out: Path) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importances.index, importances.values, color="#1f77b4", alpha=0.8)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("LightGBM — Top 15 Feature Importances", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved → {out}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("CobbleStoneEnergy Forecasting Pipeline — START")
    logger.info(f"Market: Germany (DE)  |  Period: {args.start} → {args.end}")
    logger.info("=" * 60)

    # ── 1. Data ingestion ─────────────────────────────────────────────────────
    logger.info("STEP 1: Data Ingestion")
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    test_start = pd.Timestamp(args.test_start, tz="UTC")

    df_raw = fetch_all_data(
        start=start_ts,
        end=end_ts,
        force_refresh=args.force_refresh,
    )
    logger.info(f"Raw dataset: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

    # ── 2. Data QA ────────────────────────────────────────────────────────────
    logger.info("STEP 2: Data Quality Assurance")

    llm_rules: list[dict] = []
    if not args.no_llm:
        logger.info("  → Requesting LLM-proposed QA rules...")
        llm_rules = propose_qa_rules(df_raw)
        logger.info(f"  → {len(llm_rules)} rules proposed by Claude")

    qa_report = run_qa(df_raw, llm_rules=llm_rules if llm_rules else None, tag="de_power")
    logger.info("\nQA Summary:\n" + summarise_qa(qa_report))

    # ── 3. Feature engineering ────────────────────────────────────────────────
    logger.info("STEP 3: Feature Engineering")

    # Forward-fill short gaps (≤4 h) before feature computation
    df_filled = df_raw.copy()
    for col in df_raw.columns:
        df_filled[col] = df_raw[col].ffill(limit=4)

    df_feat = build_features(df_filled)
    logger.info(f"Feature matrix: {df_feat.shape[1]} columns after engineering")

    # Visualise raw data
    plot_price_series(df_feat.dropna(subset=["da_price"]), FIGURES_DIR / "01_raw_series.png")

    # ── 4. Baseline models ────────────────────────────────────────────────────
    logger.info("STEP 4: Baseline Models")

    X_all, y_all = get_feature_matrix(df_feat)
    test_mask = X_all.index >= test_start
    train_mask = ~test_mask

    X_train_bl = X_all.loc[train_mask]
    y_train_bl = y_all.loc[train_mask]
    X_test_bl  = X_all.loc[test_mask]
    y_test_bl  = y_all.loc[test_mask]

    naive = SeasonalNaive168().fit(X_train_bl, y_train_bl)
    naive_preds_test = pd.Series(naive.predict(X_test_bl), index=X_test_bl.index)

    linear = LinearBaseline().fit(X_train_bl, y_train_bl)
    linear_preds_test = pd.Series(linear.predict(X_test_bl), index=X_test_bl.index)

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    def quick_metrics(y_true, y_pred, name):
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        p90_err = np.mean(np.abs(y_true - y_pred)[np.abs(y_true) >= np.percentile(np.abs(y_true), 90)])
        logger.info(f"  {name:<25} MAE={mae:.2f}  RMSE={rmse:.2f}  Tail-MAE={p90_err:.2f}")
        return {"name": name, "mae": round(mae,2), "rmse": round(rmse,2), "tail_mae": round(p90_err,2)}

    metrics_table = []
    metrics_table.append(quick_metrics(y_test_bl.values, naive_preds_test.values, "Seasonal Naive 168h"))
    metrics_table.append(quick_metrics(y_test_bl.values, linear_preds_test.values, "Ridge Linear"))

    # ── 5. LightGBM walk-forward CV ───────────────────────────────────────────
    logger.info("STEP 5: LightGBM Walk-Forward Validation")

    df_train_only = df_feat.loc[df_feat.index < test_start]
    lgbm_result = walk_forward_validate(df_train_only)

    metrics_table.append({
        "name": "LightGBM (walk-forward CV)",
        **lgbm_result.metrics,
    })
    logger.info(f"  LightGBM CV — MAE={lgbm_result.metrics['mae']:.2f}  "
                f"RMSE={lgbm_result.metrics['rmse']:.2f}  "
                f"Tail-MAE={lgbm_result.metrics['tail_mae_p90']:.2f}")

    # Print metrics table
    logger.info("\nModel Comparison:")
    logger.info(f"  {'Model':<30} {'MAE':>8} {'RMSE':>8} {'Tail-MAE':>10}")
    logger.info("  " + "-" * 60)
    for m in metrics_table:
        logger.info(f"  {m['name']:<30} {m.get('mae',0):>8.2f} {m.get('rmse',0):>8.2f} {m.get('tail_mae_p90', m.get('tail_mae',0)):>10.2f}")

    # Align naive preds for the validation window (to compare visually)
    X_cv_val, _ = get_feature_matrix(df_train_only)
    naive_preds_cv = pd.Series(naive.predict(X_cv_val), index=X_cv_val.index)
    plot_forecast_vs_actual(lgbm_result, naive_preds_cv, FIGURES_DIR / "02_forecast_vs_actual.png")

    # ── 6. Final model + test-set predictions ─────────────────────────────────
    logger.info("STEP 6: Final Model & Test Predictions")

    final_model, feature_cols = train_final_model(df_feat, cutoff=test_start)

    X_test, y_test = get_feature_matrix(df_feat.loc[df_feat.index >= test_start])
    test_preds = generate_forecast(final_model, X_test, feature_cols)

    test_met = quick_metrics(y_test.values, test_preds.values, "LightGBM (test set)")
    metrics_table.append(test_met)
    logger.info(f"  Test-set performance: MAE={test_met['mae']:.2f} RMSE={test_met['rmse']:.2f}")

    plot_feature_importance(final_model, feature_cols, FIGURES_DIR / "03_feature_importance.png")

    # ── 7. Curve translation ──────────────────────────────────────────────────
    logger.info("STEP 7: DA → Prompt Curve Translation")

    trading_view = generate_trading_view(
        forecast=test_preds,
        actuals=y_test,
        periods=("D", "W", "ME"),
    )
    plot_delivery_curve(trading_view, FIGURES_DIR / "04_delivery_curve.png")

    # Print monthly view
    monthly = trading_view.get("monthly", pd.DataFrame())
    if not monthly.empty:
        logger.info("\nMonthly Delivery View (first 6 months):")
        logger.info("\n" + monthly.head(6).round(2).to_string())

    print("\n" + TRADING_DESK_RATIONALE)

    # ── 8. AI Commentary ─────────────────────────────────────────────────────
    if not args.no_llm:
        logger.info("STEP 8: AI-Generated Daily Commentary")
        target_date = test_start + pd.Timedelta(days=1)
        if target_date in test_preds.index.normalize():
            metrics_dict = build_commentary_metrics(
                forecast=test_preds,
                actuals=y_test,
                df_features=df_feat,
                target_date=target_date,
            )
            commentary = generate_daily_commentary(metrics_dict)
            logger.info(f"\nAI Daily Commentary ({target_date.date()}):\n{commentary}")
            commentary_path = LOGS_DIR / f"commentary_{target_date.date()}.txt"
            commentary_path.write_text(commentary)
        else:
            logger.warning("Target date not in test forecast range; skipping commentary.")

    # ── 9. submission.csv ─────────────────────────────────────────────────────
    logger.info("STEP 9: Writing submission.csv")
    submission = pd.DataFrame({
        "id": test_preds.index.strftime("%Y-%m-%d %H:%M:%S"),
        "y_pred": test_preds.values.round(4),
    })
    sub_path = ROOT / "submission.csv"
    submission.to_csv(sub_path, index=False)
    logger.success(f"submission.csv written → {sub_path} ({len(submission):,} rows)")

    # ── 10. Metrics summary JSON ──────────────────────────────────────────────
    summary = {
        "market": "DE (Germany)",
        "train_period": f"{args.start} → {args.test_start}",
        "test_period": f"{args.test_start} → {args.end}",
        "n_train_hours": int((X_all.loc[train_mask].shape[0])),
        "n_test_hours": int(len(test_preds)),
        "model_metrics": metrics_table,
    }
    summary_path = ROOT / "outputs" / "metrics_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.success(f"Metrics summary → {summary_path}")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CobbleStoneEnergy Forecasting Pipeline")
    p.add_argument("--start", default="2024-01-01", help="Dataset start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2026-05-25", help="Dataset end date (YYYY-MM-DD)")
    p.add_argument("--test-start", default="2025-07-01", help="Test split start date")
    p.add_argument("--force-refresh", action="store_true", help="Re-download data ignoring cache")
    p.add_argument("--skip-download", action="store_true", help="Use cached data only (no API call)")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM/AI components (no API key needed)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
