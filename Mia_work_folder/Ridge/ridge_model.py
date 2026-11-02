import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import TARGET, evaluate

# Ridge regression: linear regression with an L2 penalty (alpha) that shrinks the coefficients,
# which makes the model more stable when features are correlated (e.g. the different price lags)

# Same-hour actual_load, hydro_run_of_river and hydro_reservoir are not known at forecast time, so they are left out
# NOTE: the 24h lags of actual values are only partly known at forecast time (leakage), to be revisited
NUMERIC_FEATURES = [
    "load_forecast",
    "wind_forecast",
    "mean_temperature_NO1",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "actual_load_lag_24h",
    "actual_load_lag_168h",
    "hydro_run_of_river_lag_24h",
    "hydro_run_of_river_lag_168h",
    "hydro_reservoir_lag_24h",
    "hydro_reservoir_lag_168h",
    "temperature_lag_24h",
    "temperature_lag_48h",
    "temperature_lag_168h",
]

# Calendar features are one-hot encoded, otherwise the linear model would treat hour 23 as "more" than hour 1
CATEGORICAL_FEATURES = [
    "hour",
    "day_of_week",
    "month",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_ridge(alpha=1.0):

    # Scaling is fitted inside the pipeline, so it only ever learns from the training data

    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("calendar", OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("ridge", Ridge(alpha=alpha)),
    ])


def tune_alpha(train, val, alphas):

    # Fit one model per alpha on the training set and score it on the validation set

    rows = []

    for alpha in alphas:

        model = build_ridge(alpha).fit(train[FEATURES], train[TARGET])

        rows.append({
            "alpha": alpha,
            "train_MAE": evaluate(train[TARGET], model.predict(train[FEATURES]))["MAE"],
            "val_MAE": evaluate(val[TARGET], model.predict(val[FEATURES]))["MAE"],
        })

    return pd.DataFrame(rows)


def coefficients(model):

    names = model.named_steps["preprocessing"].get_feature_names_out()
    values = model.named_steps["ridge"].coef_

    return pd.Series(values, index=names).sort_values(key=abs, ascending=False)
