import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, evaluate

# Regularized linear regression, the penalty strength is set by alpha:
#   - Ridge (L2 penalty, alpha * sum of squared coefficients): shrinks all coefficients towards zero,
#     which makes the model more stable when features are correlated (e.g. the different price lags)
#   - Lasso (L1 penalty, alpha * sum of absolute coefficients): can set coefficients exactly to zero,
#     so it also selects features

REGRESSORS = {
    "ridge": lambda alpha: Ridge(alpha=alpha),
    "lasso": lambda alpha: Lasso(alpha=alpha, max_iter=10000),
}


def build_linear(model="ridge", alpha=1.0):

    # Scaling is fitted inside the pipeline, so it only ever learns from the training data
    # Calendar features are one-hot encoded, otherwise the linear model would treat hour 23 as "more" than hour 1

    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("calendar", OneHotEncoder(drop="first"), CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", REGRESSORS[model](alpha)),
    ])


def tune_alpha(train, val, alphas, model="ridge"):

    # Fit one model per alpha on the training set and score it on the validation set

    rows = []

    for alpha in alphas:

        fitted = build_linear(model, alpha).fit(train[FEATURES], train[TARGET])

        rows.append({
            "alpha": alpha,
            "train_MAE": evaluate(train[TARGET], fitted.predict(train[FEATURES]))["MAE"],
            "val_MAE": evaluate(val[TARGET], fitted.predict(val[FEATURES]))["MAE"],
            "nonzero_coefs": (fitted.named_steps["regressor"].coef_ != 0).sum(),
        })

    return pd.DataFrame(rows)


def coefficients(fitted):

    names = fitted.named_steps["preprocessing"].get_feature_names_out()
    values = fitted.named_steps["regressor"].coef_

    return pd.Series(values, index=names).sort_values(key=abs, ascending=False)
