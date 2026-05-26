import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# LOAD FILES
# =========================

folder = "./"  # change if needed

files = [
    "2024.csv",
    "2025.csv",
    "2026.csv"
]

dfs = []

for file in files:
    df = pd.read_csv(Path(folder) / file)
    dfs.append(df)

# =========================
# MERGE DATA
# =========================

df = pd.concat(dfs, ignore_index=True)

# =========================
# CLEAN COLUMN NAMES
# =========================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

print(df.columns)

# =========================
# RENAME IMPORTANT COLUMNS
# =========================

df = df.rename(columns={
    "mtu_utc": "timestamp_range",
    "actual_total_load_mw": "actual_load",
    "day-ahead_total_load_forecast_mw": "forecast_load",
})

# =========================
# EXTRACT START TIMESTAMP
# =========================

# Example:
# 01/01/2026 00:00 - 01/01/2026 00:15

df["timestamp"] = (
    df["timestamp_range"]
    .str.split(" - ")
    .str[0]
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%d/%m/%Y %H:%M"
)

# =========================
# REMOVE DUPLICATES
# =========================

df = df.drop_duplicates(subset=["timestamp"])

# =========================
# SORT TIME
# =========================

df = df.sort_values("timestamp")

# =========================
# HANDLE MISSING VALUES
# =========================

df["actual_load"] = pd.to_numeric(
    df["actual_load"],
    errors="coerce"
)

df["forecast_load"] = pd.to_numeric(
    df["forecast_load"],
    errors="coerce"
)

# fill missing values
df["actual_load"] = df["actual_load"].interpolate()
df["forecast_load"] = df["forecast_load"].interpolate()

# =========================
# FEATURE ENGINEERING
# =========================

df["forecast_error"] = (
    df["actual_load"] - df["forecast_load"]
)

df["absolute_error"] = (
    df["forecast_error"].abs()
)

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = df["timestamp"].dt.day_name()

df["month"] = df["timestamp"].dt.month

df["weekend"] = (
    df["timestamp"].dt.weekday >= 5
).astype(int)

# =========================
# ROLLING FEATURES
# =========================

df["rolling_load_24h"] = (
    df["actual_load"]
    .rolling(96)  # 96 quarter-hours = 24h
    .mean()
)

df["rolling_error_24h"] = (
    df["absolute_error"]
    .rolling(96)
    .mean()
)

# =========================
# SAVE CLEAN DATA
# =========================

df.to_csv("clean_power_data.csv", index=False)

print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDONE CLEANING")
