import requests
import pandas as pd
import time


# ============================================================
# USER SETTINGS
# ============================================================

CLIENT_ID = "1dca95d9-38b2-4c52-acd0-5b14e41b6842"

START_YEAR = 2019
END_YEAR = 2025

OUTPUT_FILE = "NO1_temperature_2019_2025.csv"

OBSERVATION_URL = (
    "https://frost.met.no/observations/v0.jsonld"
)


# ============================================================
# WEATHER STATIONS
# ============================================================



STATIONS = {
    "SN18700": "oslo_blindern",
    "SN12290": "hamar_ii",
    "SN12680": "lillehammer_saetherengen",
    "SN5590": "kongsvinger",
    "SN23420": "fagernes",
}


# ============================================================
# FROST API REQUEST
# ============================================================

def frost_request(url, params):
    """
    Send authenticated request to MET Norway Frost API.
    """

    response = requests.get(
        url,
        params=params,
        auth=(CLIENT_ID, ""),
        timeout=60
    )

    if response.status_code != 200:

        print("\nFrost API request failed")
        print("Status:", response.status_code)

        try:
            print(response.json())
        except Exception:
            print(response.text[:1000])

        raise RuntimeError(
            f"Frost API error: {response.status_code}"
        )

    return response.json()


# ============================================================
# CREATE MONTHLY INTERVALS
# ============================================================

def month_intervals(start_year, end_year):
    """
    Generate monthly intervals between start_year
    and end_year.
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

        start_date = dates[i]

        # Frost interval uses inclusive-looking ISO date ranges.
        # Use the last second before the next month.
        end_date = dates[i + 1] - pd.Timedelta(seconds=1)

        intervals.append(
            (
                start_date.isoformat(),
                end_date.isoformat()
            )
        )

    return intervals


# ============================================================
# DOWNLOAD ONE MONTH
# ============================================================

def get_temperature_month(
    station_id,
    start_date,
    end_date
):
    """
    Download air temperature observations for one
    station over one month.
    """

    params = {

    "sources": f"{station_id}:0",

    "referencetime":
        f"{start_date}/{end_date}",

    "elements": "air_temperature",

    # We specifically want hourly observations
    "timeresolutions": "PT1H",

    # Select standard/default observation series
    "timeoffsets": "default",
    "levels": "default",

    # Explicitly request the official/default time series
    "timeseriesids": "0",

    # Accept normal quality-controlled observations
    "qualities": "0,1,2,3,4"
}

    result = frost_request(
        OBSERVATION_URL,
        params
    )

    records = []

    for item in result.get("data", []):

        timestamp = pd.to_datetime(
            item["referenceTime"],
            utc=True
        )

        observations = item.get(
            "observations",
            []
        )

        for observation in observations:

            if (
                observation.get("elementId")
                != "air_temperature"
            ):
                continue

            value = observation.get("value")

            if value is None:
                continue

            records.append({
    "timestamp": timestamp,
    "temperature": float(value),
    "time_resolution": observation.get("timeResolution"),
    "time_series_id": observation.get("timeSeriesId"),
    "quality_code": observation.get("qualityCode")
})

    return pd.DataFrame(records)


# ============================================================
# DOWNLOAD ONE STATION MONTH-BY-MONTH
# ============================================================

def download_station(
    station_id,
    station_name
):
    """
    Download temperature for one station month-by-month.
    """

    intervals = month_intervals(
        START_YEAR,
        END_YEAR
    )

    frames = []

    print(
        f"\nDownloading {station_name} "
        f"({station_id})"
    )

    for start, end in intervals:

        month_name = start[:7]

        print(
            f"Downloading {month_name}..."
        )

        try:

            df = get_temperature_month(
                station_id,
                start,
                end
            )

            if not df.empty:

                frames.append(df)

                print(
                    f"  Retrieved {len(df)} observations"
                )

            else:

                print(
                    "  No observations returned"
                )

        except Exception as e:

            print(
                f"  FAILED: {e}"
            )

        # Small delay between API requests
        time.sleep(0.2)

    if not frames:

        print(
            f"No data retrieved for {station_name}"
        )

        return pd.DataFrame()

    # Combine all months
    df = pd.concat(
        frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Multiple temperature observations can sometimes occur
    # at the same timestamp.
    #
    # For the initial dataset we aggregate duplicates using
    # the mean.
    # --------------------------------------------------------

    df = (
            df.sort_values("timestamp")
            .reset_index(drop=True)
        )
    
    # Check for duplicate timestamps
    duplicate_count = df["timestamp"].duplicated().sum()

    if duplicate_count > 0:

        print(
        f"WARNING: {duplicate_count} duplicate "
        f"timestamps still returned."
    )

        print("\nTime resolutions:")
        print(
        df["time_resolution"]
        .value_counts(dropna=False)
    )

        print("\nTime series IDs:")
        print(
        df["time_series_id"]
        .value_counts(dropna=False)
    )

        raise RuntimeError(
        "Multiple temperature observations still exist "
        "for the same timestamp. Do not average them."
    )

    # Keep only columns required for modelling
    df = df[["timestamp","temperature"]].copy()

    # Rename station column
    df = df.rename(
        columns={
            "temperature":
            f"temperature_{station_name}"
        }
    )

   

    print(
        f"\nFinished {station_name}: "
        f"{len(df)} unique timestamps"
    )

    return df


# ============================================================
# DOWNLOAD ALL STATIONS
# ============================================================

def download_all_stations():

    station_data = []

    for station_id, station_name in STATIONS.items():

        print("\n========================================")
        print(
            f"{station_name} ({station_id})"
        )
        print("========================================")

        df = download_station(
            station_id,
            station_name
        )

        if not df.empty:

            station_data.append(
                df
            )

    if not station_data:

        raise RuntimeError(
            "No temperature data were downloaded."
        )

    # Start with first station
    combined = station_data[0]

    # Merge additional stations
    for df in station_data[1:]:

        combined = pd.merge(
            combined,
            df,
            on="timestamp",
            how="outer"
        )

    combined = (
        combined
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return combined


# ============================================================
# CREATE REPRESENTATIVE NO1 TEMPERATURE
# ============================================================

def create_mean_temperature(df):

    temperature_columns = [
        col
        for col in df.columns
        if col.startswith("temperature_")
    ]

    # Number of stations available each hour
    df["temperature_station_count"] = (
        df[temperature_columns]
        .notna()
        .sum(axis=1)
    )

    df["mean_temperature_NO1"] = (
        df[temperature_columns]
        .mean(
            axis=1,
            skipna=True
        )
    )

    return df


def create_complete_hourly_index(df):
    """
    Ensure every hour between START_YEAR and END_YEAR
    exists in the dataframe.

    Missing observations become NaN.
    """

    full_index = pd.date_range(
        start=f"{START_YEAR}-01-01 00:00:00",
        end=f"{END_YEAR}-12-31 23:00:00",
        freq="1h",
        tz="UTC"
    )

    df = (
        df
        .set_index("timestamp")
        .reindex(full_index)
    )

    df.index.name = "timestamp"

    return df.reset_index()

# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

def print_summary(df):

    print("\n========================================")
    print("TEMPERATURE DATA SUMMARY")
    print("========================================")

    print("\nShape:")
    print(df.shape)

    print("\nTime period:")
    print(
        df["timestamp"].min(),
        "to",
        df["timestamp"].max()
    )

    print("\nMissing values:")
    print(
        df.isna().sum()
    )

    print("\nTemperature statistics:")

    temperature_columns = [
        col
        for col in df.columns
        if col.startswith("temperature_")
    ]

    print(
        df[temperature_columns].describe()
    )

    print("\nTime intervals:")

    print(
        df["timestamp"]
        .diff()
        .value_counts()
        .head()
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(
    df,
    filename=OUTPUT_FILE
):

    df.to_csv(
        filename,
        index=False
    )

    print(
        f"\nDataset successfully saved to:\n"
        f"{filename}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("MET NORWAY FROST DATA RETRIEVAL")
    print("========================================")

   # Download all stations
    temperature_data = download_all_stations()

     # Create complete hourly timeline
    temperature_data = create_complete_hourly_index(
        temperature_data
    )
# Calculate representative NO1 temperature
    temperature_data = create_mean_temperature(
    temperature_data)

    # Diagnostics
    print_summary(
        temperature_data
    )

    # Save
    save_dataset(
        temperature_data
    )

    print("\nFinished.")