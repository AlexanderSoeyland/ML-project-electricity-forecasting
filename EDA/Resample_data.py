import pandas as pd


# ============================================================
# LOAD RAW DATA
# ============================================================

df = pd.read_csv(
    "NO1_entsoe_energy_dataset_2018_2025.csv",
    parse_dates=["timestamp"]
)


# ============================================================
# KEEP RELEVANT RAW VARIABLES ONLY
# ============================================================

columns = [
    "timestamp",
    "day_ahead_price",
    "load_forecast",
    "actual_load",
    "wind_forecast",
    "hydro_run_of_river",
    "hydro_reservoir"
]

df = df[columns].copy()


# ============================================================
# USE 2019-2025
# ============================================================

df = df[
    (df["timestamp"] >= "2019-01-01") &
    (df["timestamp"] < "2026-01-01")
].copy()


# ============================================================
# CONVERT TO HOURLY RESOLUTION
# ============================================================

df = (
    df.set_index("timestamp")
      .resample("1h")
      .mean()
      .reset_index()
)


# ============================================================
# CREATE CALENDAR FEATURES
# ============================================================

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["year"] = df["timestamp"].dt.year

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# CREATE PRICE LAGS
# ============================================================

df["price_lag_24h"] = (
    df["day_ahead_price"].shift(24)
)

df["price_lag_168h"] = (
    df["day_ahead_price"].shift(168)
)


# ============================================================
# CREATE ACTUAL LOAD LAGS
# ============================================================

df["actual_load_lag_24h"] = (
    df["actual_load"].shift(24)
)

df["actual_load_lag_168h"] = (
    df["actual_load"].shift(168)
)


# ============================================================
# CREATE HYDRO LAGS
# ============================================================

df["hydro_run_of_river_lag_24h"] = (
    df["hydro_run_of_river"].shift(24)
)

df["hydro_run_of_river_lag_168h"] = (
    df["hydro_run_of_river"].shift(168)
)

df["hydro_reservoir_lag_24h"] = (
    df["hydro_reservoir"].shift(24)
)

df["hydro_reservoir_lag_168h"] = (
    df["hydro_reservoir"].shift(168)
)


# ============================================================
# CHECK RESULT
# ============================================================

print("\nDataset shape:")
print(df.shape)

print("\nTime intervals:")
print(
    df["timestamp"]
    .diff()
    .value_counts()
    .head()
)

print("\nMissing values:")
print(
    df.isna().sum()
)


# ============================================================
# SAVE PROCESSED DATASET
# ============================================================

df.to_csv(
    "NO1_entsoe_hourly_2019_2025_resampled.csv",
    index=False
)

print("\nProcessed hourly dataset saved.")