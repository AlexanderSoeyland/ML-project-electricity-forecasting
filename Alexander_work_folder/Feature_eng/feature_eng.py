import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Common functions for all models: loading data, temporal split, evaluation metrics and plots

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "EDA_final_data"
    / "Final_dataset"
    / "NO1_ML_dataset_2019_2025.csv"
)



def load_data(path=DATA_PATH):

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df






def historical_price_average(df): 

    # Rather than relying on a single previous price, 
    # combine prices from the same hour over several previous days.
    # Finding average from the existing features: 24 h, 48h, 168 h lags and adding to the dataframe as a new feature


    df["historical_price_average"] = df[["price_lag_24h", "price_lag_48h", "price_lag_168h"]].mean(axis=1)
    df["historical_price_average"] = df["historical_price_average"].round(2)
    return df


def historical_price_volatility(df): 

    
    # Finding volatility from the existing features: 24 h, 48h, 168 h lags and adding to the dataframe as a new feature
    df["historical_price_volatility"] = df[["price_lag_24h", "price_lag_48h", "price_lag_168h"]].std(axis=1)
    df["historical_price_volatility"] = df["historical_price_volatility"].round(2)
    return df



def historical_hydrpower_generation_share(df): 

    # H_share_lag_168 = (H_run_of_river_168_lag + H_reservoir_168_lag)/ Load_forecast_168_lag)
    # need to round to 2 decimals
    df["hydropower_generation_share_168h_lag"] = (df["hydro_run_of_river_lag_168h"] + df["hydro_reservoir_lag_168h"]) / df["actual_load_lag_168h"]
    df["hydropower_generation_share_168h_lag"] = df["hydropower_generation_share_168h_lag"].round(2)

    return df


def residual_electricity_demand(df): 

    # Residual electricity demand = actual load - (wind + solar + hydro)
    # L_{residual,t} = L_{forecast,t} - W_{forecast,t} - H_{RoR,t-168} - H_{reservoir,t-168}
    df["residual_electricity_demand"] = df["load_forecast"] - (df["wind_forecast"] + df["hydro_run_of_river_lag_168h"] + df["hydro_reservoir_lag_168h"])
    df["residual_electricity_demand"] = df["residual_electricity_demand"].round(2)
    return df



def heating_degree_indicator(df): 

    # Heating degree indicator = max(0, 15 - T_{avg,t})
    # The feature distinguishes between temperatures that are likely
    # to generate heating demand and warmer temperatures where the heating effect is smaller.
    # Here, 15°C is an initial threshold that should be tuned using training data

    df["heating_degree_indicator"] = np.maximum(0, 15 - df["mean_temperature_NO1"])
    df["heating_degree_indicator"] = df["heating_degree_indicator"].round(2)
    return df

def add_features(df): 

    df = historical_price_average(df)
    df = historical_price_volatility(df)
    df = historical_hydrpower_generation_share(df)
    df = residual_electricity_demand(df)
    df = heating_degree_indicator(df)

    return df


def save_feature_engineered_data(df, path):
    df.to_csv(path, index=False)


