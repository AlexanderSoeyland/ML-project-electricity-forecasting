import numpy as np
import pandas as pd

# Naive baseline models for day-ahead price forecasting in NO1
# The baselines need no training, they only copy earlier prices:
#   - daily naive:  price at the same hour the day before (lag 24h)
#   - weekly naive: price at the same hour one week before (lag 168h)
#   - hybrid naive (Lago et al., 2021): weekly naive for Monday, Saturday and Sunday,
#     daily naive for Tuesday to Friday


def naive_predictions(df):

    predictions = pd.DataFrame(index=df.index)

    predictions["daily_naive"] = df["price_lag_24h"]
    predictions["weekly_naive"] = df["price_lag_168h"]

    # Monday = 0, Saturday = 5, Sunday = 6
    use_weekly = df["day_of_week"].isin([0, 5, 6])

    predictions["hybrid_naive"] = np.where(
        use_weekly,
        df["price_lag_168h"],
        df["price_lag_24h"]
    )

    return predictions
