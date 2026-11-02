import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Common functions for all models: loading data, temporal split, evaluation metrics and plots

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "Alexander_work_folder"
    / "EDA_final_data"
    / "Final_dataset"
    / "NO1_ML_dataset_2019_2025.csv"
)

TARGET = "day_ahead_price"


def load_data(path=DATA_PATH):

    df = pd.read_csv(path)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def split_data(df):

    # Temporal split based on the local (Norwegian) calendar year in the "year" column
    # Train: 2019-2023, validation: 2024, test: 2025
    # The last row (2026-01-01 00:00 local time) falls outside all three sets

    train = df[df["year"] <= 2023].copy()
    val = df[df["year"] == 2024].copy()
    test = df[df["year"] == 2025].copy()

    return train, val, test


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate(y_true, y_pred, y_reference=None):

    # MAE:  mean absolute error, average size of the error in EUR/MWh
    # RMSE: root mean squared error, punishes large errors (price spikes) more than MAE
    # R2:   share of the variance in the price explained by the model (1 is perfect)
    # rMAE: MAE relative to a reference forecast (daily naive), below 1 means better than the reference

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_true - y_pred

    scores = {
        "MAE": np.mean(np.abs(errors)),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "R2": 1 - np.sum(errors ** 2) / np.sum((y_true - y_true.mean()) ** 2),
    }

    if y_reference is not None:
        reference_mae = np.mean(np.abs(y_true - np.asarray(y_reference)))
        scores["rMAE"] = scores["MAE"] / reference_mae

    return scores


def results_table(y_true, predictions, reference="daily_naive"):

    # predictions: DataFrame with one column per model, same index as y_true

    rows = []

    for model in predictions.columns:

        scores = evaluate(
            y_true,
            predictions[model],
            predictions[reference]
        )

        rows.append({"model": model, **scores})

    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------

def plot_time_series(data, predictions, start, end, title=""):

    # Actual price and predictions for a chosen period (start/end as "YYYY-MM-DD", UTC)

    mask = (
        (data["timestamp"] >= pd.Timestamp(start, tz="UTC"))
        & (data["timestamp"] < pd.Timestamp(end, tz="UTC"))
    )

    plt.figure(figsize=(15, 5))
    plt.plot(data.loc[mask, "timestamp"], data.loc[mask, TARGET], color="black", linewidth=2, label="Actual price")

    for model in predictions.columns:
        plt.plot(data.loc[mask, "timestamp"], predictions.loc[mask, model], linewidth=1.2, label=model)

    plt.title(title)
    plt.xlabel("Time (UTC)")
    plt.ylabel("Day-ahead price (EUR/MWh)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_parity(y_true, predictions):

    # Predicted versus observed price, a perfect model lies on the diagonal

    models = predictions.columns
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    low = min(y_true.min(), predictions.min().min())
    high = max(y_true.max(), predictions.max().max())

    for ax, model in zip(axes, models):
        ax.scatter(y_true, predictions[model], s=3, alpha=0.3)
        ax.plot([low, high], [low, high], color="red", linestyle="--", label="Perfect prediction")
        ax.set_title(model)
        ax.set_xlabel("Actual price (EUR/MWh)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Predicted price (EUR/MWh)")
    axes[0].legend()
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, predictions):

    # Left: distribution of residuals (actual - predicted)
    # Right: residuals versus predicted value, to see if errors grow with the price level

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for model in predictions.columns:
        residuals = y_true - predictions[model]
        axes[0].hist(residuals, bins=200, range=(-100, 100), histtype="step", linewidth=1.5, label=model)
        axes[1].scatter(predictions[model], residuals, s=3, alpha=0.3, label=model)

    axes[0].set_title("Residual distribution (clipped to ±100)")
    axes[0].set_xlabel("Residual (EUR/MWh)")
    axes[0].set_ylabel("Number of hours")
    axes[0].legend()

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Residuals versus predicted price")
    axes[1].set_xlabel("Predicted price (EUR/MWh)")
    axes[1].set_ylabel("Residual (EUR/MWh)")
    axes[1].legend(markerscale=5)

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_error_by_group(data, predictions):

    # MAE per hour of the day and per month, to see when the models struggle

    abs_errors = predictions.rsub(data[TARGET], axis=0).abs()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    abs_errors.groupby(data["hour"]).mean().plot(ax=axes[0], marker="o")
    axes[0].set_title("MAE per hour of the day (local time)")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("MAE (EUR/MWh)")

    abs_errors.groupby(data["month"]).mean().plot(ax=axes[1], marker="o")
    axes[1].set_title("MAE per month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("MAE (EUR/MWh)")

    for ax in axes:
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
