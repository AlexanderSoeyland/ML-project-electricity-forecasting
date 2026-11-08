import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import ParameterGrid

from model_utils import FEATURES, TARGET, evaluate

# Gradient boosting: an ensemble of small decision trees, built one after another,
# where each new tree tries to correct the errors of the trees before it.
# HistGradientBoostingRegressor is scikit-learn's histogram-based version (same idea as LightGBM).
# Trees do not need scaled features or one-hot encoding, so hour, day_of_week and month are used as numbers.

RANDOM_STATE = 42


def build_gradient_boosting(**params):

    # early_stopping is turned off: it would hold out a random (not temporal) part of the training data
    return HistGradientBoostingRegressor(
        early_stopping=False,
        random_state=RANDOM_STATE,
        **params
    )


def tune_gradient_boosting(train, val, param_grid):

    # Fit one model per hyperparameter combination on the training set and score it on the validation set

    rows = []

    for params in ParameterGrid(param_grid):

        fitted = build_gradient_boosting(**params).fit(train[FEATURES], train[TARGET])

        rows.append({
            **params,
            "train_MAE": evaluate(train[TARGET], fitted.predict(train[FEATURES]))["MAE"],
            "val_MAE": evaluate(val[TARGET], fitted.predict(val[FEATURES]))["MAE"],
        })

    return pd.DataFrame(rows).sort_values("val_MAE").reset_index(drop=True)


def feature_importance(fitted, data, n_repeats=5):

    # Permutation importance: how much the MAE increases when one feature is randomly shuffled.
    # Correlated features (e.g. the price lags) share importance, so each of them can look less important than the group

    result = permutation_importance(
        fitted,
        data[FEATURES],
        data[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    return pd.DataFrame({
        "MAE_increase": result.importances_mean,
        "std": result.importances_std,
    }, index=FEATURES).sort_values("MAE_increase", ascending=False)
