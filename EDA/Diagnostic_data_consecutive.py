import pandas as pd

df = pd.read_csv(
    "NO1_entsoe_hourly_2019_2025_resampled.csv")

def missing_gap_summary(df, columns):

    for col in columns:

        missing = df[col].isna()

        # Identify consecutive groups
        groups = (missing != missing.shift()).cumsum()

        gaps = (
            missing[missing]
            .groupby(groups[missing])
            .size()
        )

        print(f"\n{col}")

        print("Total missing:", missing.sum())

        if len(gaps) == 0:
            print("No missing gaps")

        else:
            print("Number of gaps:", len(gaps))
            print("Largest gap:", gaps.max(), "hours")
            print("Median gap:", gaps.median(), "hours")

            print(
                "Five largest gaps:",
                gaps.nlargest(5).tolist()
            )


columns_to_check = [
    "day_ahead_price",
    "load_forecast",
    "actual_load",
    "wind_forecast",
    "hydro_run_of_river",
    "hydro_reservoir"
]

missing_gap_summary(
    df,
    columns_to_check
)