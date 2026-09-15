import requests
import pandas as pd
import xml.etree.ElementTree as ET


# ============================================================
# USER SETTINGS
# ============================================================

TOKEN = "25144c7e-835b-43a3-aef8-b653adff3248"

# NO1 bidding zone
AREA = "10YNO-1--------2"

START_YEAR = 2018
END_YEAR = 2025

BASE_URL = "https://web-api.tp.entsoe.eu/api"

OUTPUT_FILE = "NO1_entsoe_energy_dataset_2018_2025.csv"


# ============================================================
# API REQUEST
# ============================================================

def entsoe_request(params):
    """
    Send request to ENTSO-E API.
    """

    params = params.copy()
    params["securityToken"] = TOKEN

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=60
    )

    if response.status_code != 200:
        print("\nHTTP error:", response.status_code)
        print(response.text[:1000])
        raise RuntimeError("ENTSO-E request failed")

    # ENTSO-E sometimes returns an acknowledgement XML instead
    # of data when no matching data is available
    if "Acknowledgement_MarketDocument" in response.text:
        print("\nENTSO-E returned no matching data:")
        print(response.text[:1000])
        raise RuntimeError("No matching ENTSO-E data")

    return response.text


# ============================================================
# XML PARSER
# ============================================================

def parse_entsoe_xml(xml_text, value_name):
    """
    Parse ENTSO-E XML into a dataframe.

    Handles:
    - quantity
    - price.amount
    - PT60M
    - PT30M
    - PT15M
    """

    root = ET.fromstring(xml_text)

    # Remove XML namespaces
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    records = []

    for timeseries in root.findall(".//TimeSeries"):

        for period in timeseries.findall("./Period"):

            start_element = period.find("./timeInterval/start")

            if start_element is None:
                continue

            start_time = pd.Timestamp(start_element.text)

            resolution_element = period.find("./resolution")

            if resolution_element is None:
                continue

            resolution = resolution_element.text

            if resolution == "PT60M":
                delta = pd.Timedelta(hours=1)

            elif resolution == "PT30M":
                delta = pd.Timedelta(minutes=30)

            elif resolution == "PT15M":
                delta = pd.Timedelta(minutes=15)

            else:
                print("Unknown resolution:", resolution)
                continue

            for point in period.findall("./Point"):

                position_element = point.find("position")

                if position_element is None:
                    continue

                position = int(position_element.text)

                quantity = point.find("quantity")
                price = point.find("price.amount")

                if quantity is not None:
                    value = float(quantity.text)

                elif price is not None:
                    value = float(price.text)

                else:
                    continue

                timestamp = start_time + (position - 1) * delta

                records.append({
                    "timestamp": timestamp,
                    value_name: value
                })

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # If multiple ENTSO-E time series exist for the same timestamp,
    # aggregate them by sum.
    # This is useful especially for generation data.
    df = (
        df.groupby("timestamp", as_index=False)[value_name]
        .sum()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# CREATE MONTHLY INTERVALS
# ============================================================

def month_intervals(start_year, end_year):
    """
    Generate monthly API request intervals.
    """

    start = pd.Timestamp(
        f"{start_year}-01-01",
        tz="UTC"
    )

    end = pd.Timestamp(
        f"{end_year + 1}-01-01",
        tz="UTC"
    )

    dates = pd.date_range(
        start=start,
        end=end,
        freq="MS"
    )

    intervals = []

    for i in range(len(dates) - 1):

        period_start = dates[i].strftime("%Y%m%d%H%M")
        period_end = dates[i + 1].strftime("%Y%m%d%H%M")

        intervals.append(
            (period_start, period_end)
        )

    return intervals


# ============================================================
# DOWNLOAD A DATASET MONTH-BY-MONTH
# ============================================================

def download_monthly(download_function, *args):
    """
    Download all months from START_YEAR to END_YEAR.
    """

    frames = []

    intervals = month_intervals(
        START_YEAR,
        END_YEAR
    )

    for start, end in intervals:

        print(f"Downloading {start[:6]}...")

        try:

            df = download_function(
                start,
                end,
                *args
            )

            if not df.empty:
                frames.append(df)

        except Exception as e:

            print(
                f"Failed for {start[:6]}:",
                e
            )

    if not frames:
        return pd.DataFrame()

    df = pd.concat(
        frames,
        ignore_index=True
    )

    df = (
        df.groupby("timestamp", as_index=False)
        .first()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 1. DAY-AHEAD PRICES
# ============================================================

def get_day_ahead_prices(start, end):

    params = {
        "documentType": "A44",
        "in_Domain": AREA,
        "out_Domain": AREA,
        "periodStart": start,
        "periodEnd": end
    }

    xml = entsoe_request(params)

    return parse_entsoe_xml(
        xml,
        "day_ahead_price"
    )


# ============================================================
# 2. DAY-AHEAD TOTAL LOAD FORECAST
# ============================================================

def get_load_forecast(start, end):

    params = {
        "documentType": "A65",
        "processType": "A01",
        "outBiddingZone_Domain": AREA,
        "periodStart": start,
        "periodEnd": end
    }

    xml = entsoe_request(params)

    return parse_entsoe_xml(
        xml,
        "load_forecast"
    )


# ============================================================
# 3. ACTUAL TOTAL LOAD
# ============================================================

def get_actual_load(start, end):

    params = {
        "documentType": "A65",
        "processType": "A16",
        "outBiddingZone_Domain": AREA,
        "periodStart": start,
        "periodEnd": end
    }

    xml = entsoe_request(params)

    return parse_entsoe_xml(
        xml,
        "actual_load"
    )


# ============================================================
# 4. DAY-AHEAD WIND FORECAST
# ============================================================

def get_wind_forecast(start, end):

    params = {
        "documentType": "A69",
        "processType": "A01",
        "in_Domain": AREA,
        "periodStart": start,
        "periodEnd": end
    }

    xml = entsoe_request(params)

    return parse_entsoe_xml(
        xml,
        "wind_forecast"
    )


# ============================================================
# 5. ACTUAL GENERATION BY PRODUCTION TYPE
# ============================================================

def get_actual_generation(
    start,
    end,
    psr_type,
    column_name
):

    params = {
        "documentType": "A75",
        "processType": "A16",
        "in_Domain": AREA,
        "psrType": psr_type,
        "periodStart": start,
        "periodEnd": end
    }

    xml = entsoe_request(params)

    return parse_entsoe_xml(
        xml,
        column_name
    )


# ENTSO-E PSR type codes
HYDRO_TYPES = {
    "B10": "hydro_pumped_storage",
    "B11": "hydro_run_of_river",
    "B12": "hydro_reservoir"
}


# ============================================================
# DOWNLOAD DATASETS
# ============================================================

print("\n========================================")
print("Downloading day-ahead prices")
print("========================================")

prices = download_monthly(
    get_day_ahead_prices
)


print("\n========================================")
print("Downloading day-ahead load forecast")
print("========================================")

load_forecast = download_monthly(
    get_load_forecast
)


print("\n========================================")
print("Downloading actual load")
print("========================================")

actual_load = download_monthly(
    get_actual_load
)


print("\n========================================")
print("Downloading wind forecast")
print("========================================")

wind_forecast = download_monthly(
    get_wind_forecast
)


hydro_dataframes = []

for psr_type, column_name in HYDRO_TYPES.items():

    print("\n========================================")
    print(f"Downloading {column_name}")
    print("========================================")

    hydro_df = download_monthly(
        get_actual_generation,
        psr_type,
        column_name
    )

    if not hydro_df.empty:
        hydro_dataframes.append(
            hydro_df
        )


# ============================================================
# SHOW WHICH DATASETS WERE SUCCESSFULLY DOWNLOADED
# ============================================================

print("\n========================================")
print("DOWNLOAD SUMMARY")
print("========================================")

datasets = {
    "prices": prices,
    "load_forecast": load_forecast,
    "actual_load": actual_load,
    "wind_forecast": wind_forecast
}

for name, df in datasets.items():

    if df.empty:
        print(f"{name}: NO DATA")

    else:
        print(
            f"{name}: {len(df)} rows "
            f"from {df['timestamp'].min()} "
            f"to {df['timestamp'].max()}"
        )


# ============================================================
# MERGE DATASETS
# ============================================================

all_dfs = [
    prices,
    load_forecast,
    actual_load,
    wind_forecast
] + hydro_dataframes


# Remove empty dataframes
all_dfs = [
    df for df in all_dfs
    if not df.empty
]


if not all_dfs:
    raise RuntimeError(
        "No datasets were downloaded successfully."
    )


data = all_dfs[0].copy()


for df in all_dfs[1:]:

    data = pd.merge(
        data,
        df,
        on="timestamp",
        how="outer"
    )


data = (
    data.sort_values("timestamp")
    .reset_index(drop=True)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# --------------------------
# Price lag features
# --------------------------

if "day_ahead_price" in data.columns:

    data["price_lag_24h"] = (
        data["day_ahead_price"]
        .shift(24)
    )

    data["price_lag_168h"] = (
        data["day_ahead_price"]
        .shift(168)
    )


# --------------------------
# Actual load lag features
# --------------------------

if "actual_load" in data.columns:

    data["actual_load_lag_24h"] = (
        data["actual_load"]
        .shift(24)
    )

    data["actual_load_lag_168h"] = (
        data["actual_load"]
        .shift(168)
    )

else:
    print(
        "\nWARNING: actual_load was not available."
    )


# --------------------------
# Hydro lag features
# --------------------------

for hydro_col in HYDRO_TYPES.values():

    if hydro_col in data.columns:

        data[f"{hydro_col}_lag_24h"] = (
            data[hydro_col]
            .shift(24)
        )

        data[f"{hydro_col}_lag_168h"] = (
            data[hydro_col]
            .shift(168)
        )


# --------------------------
# Calendar features
# --------------------------

data["hour"] = (
    data["timestamp"].dt.hour
)

data["day_of_week"] = (
    data["timestamp"].dt.dayofweek
)

data["month"] = (
    data["timestamp"].dt.month
)

data["year"] = (
    data["timestamp"].dt.year
)


# Weekend feature
data["is_weekend"] = (
    data["day_of_week"] >= 5
).astype(int)


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

print("\n========================================")
print("FINAL DATASET")
print("========================================")

print(
    "Rows:",
    len(data)
)

print(
    "Columns:",
    len(data.columns)
)

print("\nColumns:")
print(
    data.columns.tolist()
)

print("\nFirst rows:")
print(
    data.head()
)

print("\nMissing values:")
print(
    data.isna().sum()
)


# ============================================================
# SAVE / LOAD DATASET
# ============================================================

def save_dataset(df, filename="NO1_entsoe_energy_dataset_2018_2025.csv"):
    """
    Save the ENTSO-E dataframe to a CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to save.

    filename : str
        Name/path of the CSV file.
    """

    df.to_csv(
        filename,
        index=False
    )

    print(f"Dataset saved successfully to: {filename}")
    print(f"Rows saved: {len(df)}")
    print(f"Columns saved: {len(df.columns)}")


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

print(
    f"\nSaved dataset as:\n{OUTPUT_FILE}"
)