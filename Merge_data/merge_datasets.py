import pandas as pd


energy = pd.read_csv(
    "Datasets_ENTSOE/NO1_entsoe_hourly_2019_2025_resampled.csv",
    parse_dates=["timestamp"]
)

weather = pd.read_csv(
    "Datasets_Frost/NO1_temperature_2019_2025.csv",
    parse_dates=["timestamp"]
)


# Ensure UTC timestamps
energy["timestamp"] = pd.to_datetime(
    energy["timestamp"],
    utc=True
)

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"],
    utc=True
)


# Merge using timestamp
df = pd.merge(
    energy,
    weather,
    on="timestamp",
    how="left",
    validate="one_to_one"
)


print("Energy shape:", energy.shape)
print("Weather shape:", weather.shape)
print("Merged shape:", df.shape)


print("\nMissing temperature after merge:")
print(
    df["mean_temperature_NO1"]
    .isna()
    .sum()
)


# Data-quality flag
df["temperature_low_coverage"] = (
    df["temperature_station_count"] < 4
).astype(int)


# Save
df.to_csv(
    "NO1_energy_weather_2019_2025.csv",
    index=False
)

print("\nMerged dataset saved.")