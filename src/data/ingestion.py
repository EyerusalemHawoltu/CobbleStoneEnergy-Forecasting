"""
ENTSO-E Transparency Platform data ingestion for German (DE) power market.

Source endpoints documented at:
  https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
  Postman collection: https://documenter.getpostman.com/view/7009892/2s9Xy5KpTi

Bidding zone: Germany — 10Y1001A1001A63L
All timestamps are stored in UTC and converted to CET/CEST only for display.

Local CSV mode: when no ENTSOE_API_KEY is set and the manually downloaded CSV files
exist in data/raw/, the pipeline loads directly from disk (no API key required).

Expected files in data/raw/:
  energyprice{YYYY}.csv  — ENTSO-E DA price export (Sequence 1 used; Sequence 2 skipped)
  load{YYYY}.csv         — ENTSO-E actual total load export (15-min, resampled to hourly)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

COUNTRY_CODE = "DE"
BIDDING_ZONE = "10Y1001A1001A63L"

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Local CSV helpers ─────────────────────────────────────────────────────────

def _local_price_files_exist(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    years = range(start.year, end.year + 1)
    return all((CACHE_DIR / f"energyprice{y}.csv").exists() for y in years)


def _local_load_files_exist(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    years = range(start.year, end.year + 1)
    return all((CACHE_DIR / f"load{y}.csv").exists() for y in years)


def _local_wind_solar_files_exist(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    years = range(start.year, end.year + 1)
    return all((CACHE_DIR / f"wind_solar{y}.csv").exists() for y in years)


def detect_local_data_end() -> Optional[pd.Timestamp]:
    """
    Return the latest timestamp available across local price CSV files.
    This determines how far ahead the pipeline can load published DA prices.
    """
    import glob
    price_files = sorted(glob.glob(str(CACHE_DIR / "energyprice*.csv")), reverse=True)
    for pf in price_files:
        try:
            df = pd.read_csv(pf, usecols=["MTU (UTC)", "Sequence"])
            df = df[df["Sequence"] == "Sequence 1"]
            last_ts = pd.to_datetime(
                df["MTU (UTC)"].str.split(" - ").str[0].iloc[-1],
                format="%d/%m/%Y %H:%M:%S",
                utc=True,
            )
            return last_ts.floor("h")
        except Exception:
            continue
    return None


def _load_local_price_csvs(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Read manually downloaded ENTSO-E DA price CSVs.
    Uses Sequence 1 only (hourly DA price; identical across all 4 quarter-hour slots).
    Timestamps are parsed as UTC.
    """
    years = range(start.year, end.year + 1)
    dfs = []
    for year in years:
        p = CACHE_DIR / f"energyprice{year}.csv"
        if not p.exists():
            logger.warning(f"Local price file missing: {p}")
            continue
        df = pd.read_csv(p)
        df = df[df["Sequence"] == "Sequence 1"].copy()
        df["timestamp"] = pd.to_datetime(
            df["MTU (UTC)"].str.split(" - ").str[0],
            format="%d/%m/%Y %H:%M:%S",
            utc=True,
        )
        df["da_price"] = pd.to_numeric(df["Day-ahead Price (EUR/MWh)"], errors="coerce")
        dfs.append(df[["timestamp", "da_price"]])
        logger.info(f"Loaded price file: {p.name} ({len(df):,} Seq-1 rows)")

    if not dfs:
        return pd.Series(dtype=float, name="da_price")

    combined = pd.concat(dfs).set_index("timestamp")
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    # Sequence 1 repeats the same hourly value 4 times — take first per hour
    result = combined["da_price"].resample("h").first()
    result = result.loc[start:end]
    result.name = "da_price"
    return result


def _load_local_load_csvs(
    start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """
    Read manually downloaded ENTSO-E load CSVs.
    Returns a DataFrame with columns:
      load_mw          — actual total load (hourly mean of 15-min data)
      load_forecast_mw — day-ahead load forecast (hourly mean of 15-min data)
                         Useful for constructing features for future dates.
    Data is at 15-min resolution; resampled to hourly.
    """
    years = range(start.year, end.year + 1)
    dfs = []
    for year in years:
        p = CACHE_DIR / f"load{year}.csv"
        if not p.exists():
            logger.warning(f"Local load file missing: {p}")
            continue
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(
            df["MTU (UTC)"].str.split(" - ").str[0],
            format="%d/%m/%Y %H:%M",
            utc=True,
        )
        df["load_mw"] = pd.to_numeric(df["Actual Total Load (MW)"], errors="coerce")
        df["load_forecast_mw"] = pd.to_numeric(
            df["Day-ahead Total Load Forecast (MW)"], errors="coerce"
        )
        dfs.append(df[["timestamp", "load_mw", "load_forecast_mw"]])
        logger.info(f"Loaded load file: {p.name} ({len(df):,} rows)")

    if not dfs:
        return pd.DataFrame(columns=["load_mw", "load_forecast_mw"])

    combined = pd.concat(dfs).set_index("timestamp")
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined["load_mw"] = combined["load_mw"].interpolate(limit=4)
    combined["load_forecast_mw"] = combined["load_forecast_mw"].interpolate(limit=4)
    result = combined.resample("h").mean()
    result = result.loc[start:end]
    return result


def _load_local_wind_solar_csvs(
    start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """
    Read manually downloaded ENTSO-E wind+solar onshore generation forecasts.
    Files: wind_solar{YYYY}.csv — 15-min CET/CEST timestamps.

    Returns DataFrame with columns:
      wind_solar_da_mw     — day-ahead generation forecast (gate-closure safe)
      wind_solar_actual_mw — actual measured generation (for lagged features)
    """
    years = range(start.year, end.year + 1)
    dfs = []
    for year in years:
        p = CACHE_DIR / f"wind_solar{year}.csv"
        if not p.exists():
            logger.warning(f"Local wind/solar file missing: {p}")
            continue
        df = pd.read_csv(p)
        # Timestamps are in CET/CEST local time — convert to UTC
        raw_ts = df["MTU (CET/CEST)"].str.split(" - ").str[0]
        try:
            df["timestamp"] = (
                pd.to_datetime(raw_ts, format="%d/%m/%Y %H:%M:%S")
                .dt.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward")
                .dt.tz_convert("UTC")
            )
        except Exception:
            df["timestamp"] = (
                pd.to_datetime(raw_ts, format="%d/%m/%Y %H:%M:%S", errors="coerce")
                .dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT")
                .dt.tz_convert("UTC")
            )
        df["wind_solar_da_mw"] = pd.to_numeric(
            df["Day-ahead (MW)"].replace("-", float("nan")).replace(" ", float("nan")),
            errors="coerce",
        )
        df["wind_solar_actual_mw"] = pd.to_numeric(
            df["Actual (MW)"].replace("-", float("nan")).replace(" ", float("nan")),
            errors="coerce",
        )
        dfs.append(df[["timestamp", "wind_solar_da_mw", "wind_solar_actual_mw"]])
        logger.info(f"Loaded wind/solar file: {p.name} ({len(df):,} rows)")

    if not dfs:
        return pd.DataFrame(columns=["wind_solar_da_mw", "wind_solar_actual_mw"])

    combined = pd.concat(dfs).set_index("timestamp")
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()
    combined["wind_solar_da_mw"] = combined["wind_solar_da_mw"].interpolate(limit=4)
    combined["wind_solar_actual_mw"] = combined["wind_solar_actual_mw"].interpolate(limit=4)
    result = combined.resample("h").mean()
    result = result.loc[start:end]
    return result


def load_local_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Build the merged hourly dataset from manually downloaded CSVs.

    Returns a DataFrame indexed by utc_timestamp with columns:
      da_price             — hourly DA price (EUR/MWh)
      load_mw              — actual load; filled from load_forecast_mw for future hours
      load_forecast_mw     — TSO day-ahead load forecast
      wind_solar_da_mw     — ENTSO-E wind+solar onshore DA generation forecast (if available)
      wind_solar_actual_mw — actual wind+solar onshore generation (if available)
    """
    logger.info("Loading data from local CSV files (no ENTSO-E API key required)")
    prices = _load_local_price_csvs(start, end)
    load_df = _load_local_load_csvs(start, end)

    df = pd.DataFrame({
        "da_price": prices,
        "load_mw": load_df.get("load_mw"),
        "load_forecast_mw": load_df.get("load_forecast_mw"),
    })
    df.index.name = "utc_timestamp"
    df = df.loc[start:end]

    # Merge wind+solar if files exist
    if _local_wind_solar_files_exist(start, end):
        ws_df = _load_local_wind_solar_csvs(start, end)
        df["wind_solar_da_mw"] = ws_df.get("wind_solar_da_mw")
        df["wind_solar_actual_mw"] = ws_df.get("wind_solar_actual_mw")
        n_ws_null = df["wind_solar_da_mw"].isna().sum()
        if n_ws_null:
            logger.warning(f"wind_solar_da_mw has {n_ws_null} NaN rows — forward-filling")
            df["wind_solar_da_mw"] = df["wind_solar_da_mw"].ffill(limit=4)
            df["wind_solar_actual_mw"] = df["wind_solar_actual_mw"].ffill(limit=4)

    # For future rows where actual load is missing/zero, use the DA load forecast
    if "load_forecast_mw" in df.columns:
        missing_actual = df["load_mw"].isna() | (df["load_mw"] == 0)
        df.loc[missing_actual, "load_mw"] = df.loc[missing_actual, "load_forecast_mw"]

    n_price_null = df["da_price"].isna().sum()
    n_load_null = df["load_mw"].isna().sum()
    if n_price_null:
        logger.warning(f"DA price has {n_price_null} NaN rows (future dates not yet published)")
        df["da_price"] = df["da_price"].ffill(limit=4)
    if n_load_null:
        logger.warning(f"Load has {n_load_null} NaN rows — forward-filling ≤4h gaps")
        df["load_mw"] = df["load_mw"].ffill(limit=4)

    ws_col = "wind_solar_da_mw" if "wind_solar_da_mw" in df.columns else None
    logger.info(
        f"Local dataset: {len(df):,} hourly rows  "
        f"({df.index.min().date()} → {df.index.max().date()})  "
        f"da_price NaN={df['da_price'].isna().sum()}  "
        f"load NaN={df['load_mw'].isna().sum()}  "
        f"wind_solar={'yes' if ws_col else 'absent'}"
    )
    return df


# ── ENTSO-E API helpers (used only when API key is present) ───────────────────

def _get_client(api_key: Optional[str] = None):
    from entsoe import EntsoePandasClient

    key = api_key or os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise ValueError(
            "ENTSO-E API key not found. Set ENTSOE_API_KEY environment variable."
        )
    return EntsoePandasClient(api_key=key)


def _cache_path(name: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    tag = f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    return CACHE_DIR / f"{name}_{tag}.parquet"


def _fetch_with_retry(fn, *args, retries: int = 3, wait: int = 5, **kwargs):
    from entsoe.exceptions import NoMatchingDataError

    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except NoMatchingDataError:
            logger.warning(f"No data available for {fn.__name__} on this interval.")
            return None
        except Exception as exc:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(wait * (attempt + 1))
            else:
                raise


def fetch_da_prices(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    logger.info(f"Fetching DA prices {start.date()} → {end.date()}")
    result = _fetch_with_retry(client.query_day_ahead_prices, COUNTRY_CODE, start=start, end=end)
    if result is None:
        return pd.Series(dtype=float, name="da_price")
    result.name = "da_price"
    return result.tz_convert("UTC")


def fetch_load(client, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    logger.info(f"Fetching load {start.date()} → {end.date()}")
    result = _fetch_with_retry(client.query_load, COUNTRY_CODE, start=start, end=end)
    if result is None:
        return pd.Series(dtype=float, name="load_mw")
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    result.name = "load_mw"
    return result.tz_convert("UTC")


def _fetch_generation_by_type(client, start, end, psr_type, col_name) -> pd.Series:
    logger.info(f"Fetching {col_name} generation {start.date()} → {end.date()}")
    result = _fetch_with_retry(client.query_generation, COUNTRY_CODE, start=start, end=end, psr_type=psr_type)
    if result is None:
        return pd.Series(dtype=float, name=col_name)
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    result.name = col_name
    return result.tz_convert("UTC")


def fetch_wind_onshore(client, start, end) -> pd.Series:
    return _fetch_generation_by_type(client, start, end, "B18", "wind_onshore_mw")


def fetch_wind_offshore(client, start, end) -> pd.Series:
    return _fetch_generation_by_type(client, start, end, "B19", "wind_offshore_mw")


def fetch_solar(client, start, end) -> pd.Series:
    return _fetch_generation_by_type(client, start, end, "B16", "solar_mw")


def _chunk_dates(start: pd.Timestamp, end: pd.Timestamp, freq: str = "MS"):
    periods = pd.date_range(start, end, freq=freq, tz="UTC")
    chunks = []
    for i, s in enumerate(periods):
        e = periods[i + 1] if i + 1 < len(periods) else end
        chunks.append((s, e))
    return chunks


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_all_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Return a merged hourly DataFrame indexed by utc_timestamp.

    Priority order:
      1. Parquet cache (if exists and not force_refresh)
      2. Local CSV files in data/raw/ (if present and no API key configured)
      3. ENTSO-E API (requires ENTSOE_API_KEY env var or api_key arg)

    Columns from local CSVs: da_price, load_mw
    Columns from API:        da_price, wind_onshore_mw, wind_offshore_mw, solar_mw, load_mw
    """
    cache = _cache_path("de_all", start, end)
    if cache.exists() and not force_refresh:
        logger.info(f"Loading merged dataset from cache: {cache}")
        return pd.read_parquet(cache)

    has_api_key = bool(api_key or os.environ.get("ENTSOE_API_KEY"))
    has_local = _local_price_files_exist(start, end) and _local_load_files_exist(start, end)

    if has_local and not has_api_key:
        df = load_local_data(start, end)
        logger.info(f"Saving local dataset to cache → {cache}")
        df.to_parquet(cache)
        return df

    if not has_api_key:
        raise ValueError(
            "No ENTSOE_API_KEY set and local CSV files not found in data/raw/. "
            "Either set the API key or place energyprice{YYYY}.csv and load{YYYY}.csv "
            "files in data/raw/ for each year in the requested range."
        )

    client = _get_client(api_key)
    chunks = _chunk_dates(start, end)

    parts: dict[str, list[pd.Series]] = {
        "da_price": [],
        "wind_onshore_mw": [],
        "wind_offshore_mw": [],
        "solar_mw": [],
        "load_mw": [],
    }

    for chunk_start, chunk_end in chunks:
        parts["da_price"].append(fetch_da_prices(client, chunk_start, chunk_end))
        parts["wind_onshore_mw"].append(fetch_wind_onshore(client, chunk_start, chunk_end))
        parts["wind_offshore_mw"].append(fetch_wind_offshore(client, chunk_start, chunk_end))
        parts["solar_mw"].append(fetch_solar(client, chunk_start, chunk_end))
        parts["load_mw"].append(fetch_load(client, chunk_start, chunk_end))
        time.sleep(1)

    series = {}
    for col, chunks_list in parts.items():
        combined = pd.concat([s for s in chunks_list if not s.empty])
        combined = combined[~combined.index.duplicated(keep="first")]
        series[col] = combined.resample("h").mean()

    df = pd.DataFrame(series)
    df.index.name = "utc_timestamp"
    df = df.loc[start:end]

    logger.info(f"Saving merged dataset → {cache}")
    df.to_parquet(cache)
    return df
