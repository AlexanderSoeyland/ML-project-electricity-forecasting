import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constructing the final model_df from the cleaned datasets 
# This is the final dataset that will be used for 
# feature analysis, training, validation and testing the model

def clean_df(df): 

    df = df.copy()
    original_n = len(df)

    # fixing timestamp from object to datetime 
    df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Fixing timestamp to correct timezone (Norway) to derive correct calendar features (hour, day of week, month, year, is_weekend)
    # Keeping the actual timestamp for each row in UTC

    local_time = (
    df["timestamp"]
    .dt.tz_convert("Europe/Oslo")
)

    df["hour"] = local_time.dt.hour
    df["day_of_week"] = local_time.dt.dayofweek
    df["month"] = local_time.dt.month
    df["year"] = local_time.dt.year

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # frequency result is perfect, one datapoint per hour 

    #Cleaning the data:
    #Interpolation of rows with missing values: 

    

    # Load: 
    # very few missing values in actual load and load_forecast so we interpolate

    df["load_forecast"] = (
    df["load_forecast"]
    .interpolate(method="linear", limit=4)
)

    df["actual_load"] = (
    df["actual_load"]
    .interpolate(method="linear", limit=4)
)

    # Hydro: 
    # Short gaps of hydro values, we interpolate these

    df["hydro_run_of_river"] = interpolate_short_gaps(
    df["hydro_run_of_river"],
    max_gap=6
)

    df["hydro_reservoir"] = interpolate_short_gaps(
    df["hydro_reservoir"],
    max_gap=6
)

    # Wind: 
    # Wind has 955 missing values, with some large gaps (742 hours), not interpolating this, 
    # but with shorter gaps as well (48 hours, 23 hours), these we will interpolate

    df["wind_forecast"] = interpolate_short_gaps(
    df["wind_forecast"],
    max_gap=6
)

    


    # Temperature: 

    # Some large gaps at certain weather stations like Kongsvinger: 1216 hours and Hamar: 1189 hours
    # But no missing values at mean_temperature_NO1, as this is the value we will be using in the ML models
    # No interpolation or removal will be done here 


    # Removing lag columns and updating the lags: 

    lag_columns = [
    col for col in df.columns
    if "_lag_" in col
]

    df = df.drop(
    columns=lag_columns
)

    # Updating the lags after cleaning: 

        # Price
    df["price_lag_24h"] = (
        df["day_ahead_price"].shift(24)
    )

    df["price_lag_48h"] = (
        df["day_ahead_price"].shift(48)
    )

    df["price_lag_168h"] = (
        df["day_ahead_price"].shift(168)
    )


    # Actual load
    df["actual_load_lag_24h"] = (
        df["actual_load"].shift(24)
    )

    df["actual_load_lag_168h"] = (
        df["actual_load"].shift(168)
    )


    # Hydro
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


    # Temperature
    df["temperature_lag_24h"] = (
        df["mean_temperature_NO1"].shift(24)
    )

    df["temperature_lag_48h"] = (
        df["mean_temperature_NO1"].shift(48)
    )

    df["temperature_lag_168h"] = (
        df["mean_temperature_NO1"].shift(168)
    )

    
    # Final check of clean dataset and saving: 

    required_columns = [
    "day_ahead_price",
    "load_forecast",
    "wind_forecast",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "actual_load_lag_24h",
    "actual_load_lag_168h",
    "hydro_run_of_river_lag_24h",
    "hydro_run_of_river_lag_168h",
    "hydro_reservoir_lag_24h",
    "hydro_reservoir_lag_168h",
    "mean_temperature_NO1",
    "temperature_lag_24h",
    "temperature_lag_48h",
    "temperature_lag_168h",
    "hour",
    "day_of_week",
    "month",
    "is_weekend"
]

    print("\nMissing values BEFORE dropping rows:")
    print(
        df[required_columns]
        .isna()
        .sum()
    )
    
# Dropping remaining missing values in the required columns for the model_df, 
# as these are critical for the ML models to work properly, critical this is done after updating lags 
    model_df = df.dropna(
    subset=required_columns
).copy()

    # Final inspection: 

    print("Original observations:", original_n)
    print("Final observations:", len(model_df))

    print(
        "Data retained:",
        100 * len(model_df) / original_n,
        "%"
    )

    print("\nRemaining missing values:")
    print(
    model_df[required_columns]
    .isna()
    .sum()
)

    # Saving the cleaned dataset to a CSV file for future use
    model_df.to_csv('NO1_ML_dataset_2019_2025.csv', index=False)






def interpolate_short_gaps(series, max_gap=6):

    missing = series.isna()
    groups = (missing != missing.shift()).cumsum()

    interpolated = series.interpolate(
        method="linear"
    )

    result = series.copy()

    for _, indices in (
        series[missing]
        .groupby(groups[missing])
        .groups.items()
    ):

        if len(indices) <= max_gap:
            result.loc[indices] = interpolated.loc[indices]

    return result


def check_cleaned_data(df, model_df):

    # Make sure timestamps are datetime
    df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

    model_df["timestamp"] = pd.to_datetime(
    model_df["timestamp"],
    utc=True
)

   
    # ---------------------------------------------------------
    # DATA RETENTION DIAGNOSTIC
    # ---------------------------------------------------------

    # Use UTC year purely for data-quality diagnostics
    df["data_year"] = df["timestamp"].dt.year
    model_df["data_year"] = model_df["timestamp"].dt.year


    # Find removed observations using timestamps
    removed_df = df[
        ~df["timestamp"].isin(model_df["timestamp"])
    ].copy()


    print("Original observations:", len(df))
    print("Retained observations:", len(model_df))
    print("Removed observations:", len(removed_df))

    print("\nCheck:")
    print(
        len(df),
        "=",
        len(model_df),
        "+",
        len(removed_df)
    )


    # Original observations by year
    original_by_year = (
        df.groupby("data_year")
        .size()
    )


    # Retained observations by year
    retained_by_year = (
        model_df.groupby("data_year")
        .size()
    )


    # Removed observations by year
    removed_by_year = (
        removed_df.groupby(
            removed_df["timestamp"].dt.year
        )
        .size()
    )


    # Retention percentage
    retention_by_year = (
        retained_by_year
        / original_by_year
        * 100
    )


    print("\nOriginal observations by year:")
    print(original_by_year)

    print("\nRemoved observations by year:")
    print(removed_by_year)

    print("\nRetained observations by year:")
    print(retained_by_year)

    print("\nPercentage retained by year:")
    print(retention_by_year)


