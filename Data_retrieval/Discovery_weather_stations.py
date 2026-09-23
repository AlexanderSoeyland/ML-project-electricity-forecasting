import requests
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

CLIENT_ID = "1dca95d9-38b2-4c52-acd0-5b14e41b6842"

SOURCE_URL = "https://frost.met.no/sources/v0.jsonld"

TIMESERIES_URL = (
    "https://frost.met.no/observations/"
    "availableTimeSeries/v0.jsonld"
)

START_DATE = pd.Timestamp("2019-01-01", tz="UTC")
END_DATE = pd.Timestamp("2025-12-31 23:00", tz="UTC")


# Areas we want represented in NO1
SEARCH_LOCATIONS = [
    "Hamar",
    "Lillehammer",
    "Kongsvinger",
    "Fagernes"
]


# ============================================================
# BASIC FROST REQUEST
# ============================================================

def frost_get(url, params):

    response = requests.get(
        url,
        params=params,
        auth=(CLIENT_ID, ""),
        timeout=60
    )

    if response.status_code != 200:

        print("\nRequest failed:")
        print(response.status_code)
        print(response.text[:1000])

        return None

    return response.json()


# ============================================================
# FIND STATIONS BY NAME
# ============================================================

def find_stations(place):

    params = {
        "types": "SensorSystem",
        "name": f"*{place}*"
    }

    result = frost_get(
        SOURCE_URL,
        params
    )

    if result is None:
        return []

    stations = []

    for station in result.get("data", []):

        stations.append({
            "id": station.get("id"),
            "name": station.get("name"),
            "municipality": station.get(
                "municipality"
            ),
            "validFrom": station.get(
                "validFrom"
            ),
            "validTo": station.get(
                "validTo"
            )
        })

    return stations


# ============================================================
# CHECK HOURLY TEMPERATURE COVERAGE
# ============================================================

def check_hourly_temperature(station_id):

    params = {
        "sources": f"{station_id}:0",
        "elements": "air_temperature"
    }

    result = frost_get(
        TIMESERIES_URL,
        params
    )

    if result is None:
        return False, []

    valid_series = []

    for series in result.get("data", []):

        # We specifically require hourly temperature
        if series.get("elementId") != "air_temperature":
            continue

        if series.get("timeResolution") != "PT1H":
            continue

        if series.get("timeSeriesId") != 0:
            continue

        valid_from = pd.to_datetime(
            series.get("validFrom"),
            utc=True
        )

        valid_to_raw = series.get("validTo")

        # No validTo usually means series is still active
        if valid_to_raw is None:
            valid_to = pd.Timestamp.max.tz_localize("UTC")
        else:
            valid_to = pd.to_datetime(
                valid_to_raw,
                utc=True
            )

        valid_series.append({
            "valid_from": valid_from,
            "valid_to": valid_to,
            "source_id": series.get("sourceId"),
            "resolution": series.get(
                "timeResolution"
            ),
            "series_id": series.get(
                "timeSeriesId"
            )
        })

    # --------------------------------------------------------
    # Important:
    #
    # Frost can split one continuous official series into
    # several metadata periods. Therefore don't require one
    # metadata row to cover the entire period.
    #
    # Instead check whether the combined periods span
    # 2019-2025.
    # --------------------------------------------------------

    if not valid_series:
        return False, []

    earliest = min(
        x["valid_from"]
        for x in valid_series
    )

    latest = max(
        x["valid_to"]
        for x in valid_series
    )

    covers_period = (
        earliest <= START_DATE
        and
        latest >= END_DATE
    )

    return covers_period, valid_series


# ============================================================
# SEARCH ALL TARGET AREAS
# ============================================================

def main():
    for place in SEARCH_LOCATIONS:

        print("\n")
        print("=" * 60)
        print(f"SEARCHING: {place}")
        print("=" * 60)

        stations = find_stations(place)

        if not stations:

            print("No stations found.")
            continue

        for station in stations:

            station_id = station["id"]
            station_name = station["name"]

            print(
                f"\nChecking {station_name} "
                f"({station_id})..."
            )

            valid, series = (
                check_hourly_temperature(
                    station_id
                )
            )

            if valid:

                print(
                    "  ✓ SUITABLE"
                )

                print(
                    "  PT1H air_temperature "
                    "available for 2019–2025"
                )

                for s in series:

                    print(
                        "   ",
                        s["valid_from"],
                        "→",
                        s["valid_to"]
                    )

            else:

                print(
                    "  ✗ Does not provide full "
                    "2019–2025 PT1H coverage"
                )