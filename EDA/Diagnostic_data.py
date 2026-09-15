# ============================================================
# INVESTIGATE MISSING DATA
# ============================================================
import pandas as pd



def load_dataset(filename="NO1_entsoe_energy_dataset_2018_2025.csv"):
    """
    Load a previously saved ENTSO-E dataset.

    Automatically converts the timestamp column
    back into pandas datetime format.

    Returns
    -------
    pandas.DataFrame
    """

    df = pd.read_csv(
        filename,
        parse_dates=["timestamp"]
    )

    print(f"Dataset loaded successfully from: {filename}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df



data = load_dataset("NO1_entsoe_energy_dataset_2018_2025.csv")

important_columns = [
    "day_ahead_price",
    "load_forecast",
    "actual_load",
    "wind_forecast",
    "hydro_run_of_river",
    "hydro_reservoir"
]

print("\n========================================")
print("DATASET SIZE")
print("========================================")

print("Total rows:", len(data))
print("Start:", data["timestamp"].min())
print("End:", data["timestamp"].max())


# ------------------------------------------------------------
# Missing values by year
# ------------------------------------------------------------

print("\n========================================")
print("MISSING VALUES BY YEAR")
print("========================================")

for col in important_columns:

    if col not in data.columns:
        continue

    print(f"\n{col}")

    yearly = data.groupby("year")[col].agg(
        total="size",
        available="count"
    )

    yearly["missing"] = (
        yearly["total"]
        - yearly["available"]
    )

    yearly["missing_percent"] = (
        yearly["missing"]
        / yearly["total"]
        * 100
    )

    print(yearly)


# ------------------------------------------------------------
# Available observations by year
# ------------------------------------------------------------

print("\n========================================")
print("AVAILABLE OBSERVATIONS BY YEAR")
print("========================================")

for col in important_columns:

    if col in data.columns:

        print(
            f"\n{col}:"
        )

        print(
            data.groupby("year")[col].count()
        )


# ------------------------------------------------------------
# Inspect timestamp spacing
# ------------------------------------------------------------

print("\n========================================")
print("TIMESTAMP INTERVALS")
print("========================================")

time_differences = (
    data["timestamp"]
    .sort_values()
    .diff()
    .value_counts()
)

print(time_differences.head(10))


# ------------------------------------------------------------
# Missing price observations by year
# ------------------------------------------------------------

if "day_ahead_price" in data.columns:

    print("\n========================================")
    print("MISSING PRICE TIMESTAMPS")
    print("========================================")

    missing_prices = data[
        data["day_ahead_price"].isna()
    ]

    print(
        missing_prices[
            ["timestamp", "year"]
        ].head(30)
    )


# ------------------------------------------------------------
# Missing wind observations by year
# ------------------------------------------------------------

if "wind_forecast" in data.columns:

    print("\n========================================")
    print("MISSING WIND TIMESTAMPS")
    print("========================================")

    missing_wind = data[
        data["wind_forecast"].isna()
    ]

    print(
        missing_wind[
            ["timestamp", "year"]
        ].head(30)
    )