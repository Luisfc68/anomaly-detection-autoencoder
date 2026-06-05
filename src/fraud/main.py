import numpy as np
from numpy.random import standard_cauchy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fraud.config import FIGURES_DIR, set_seed
from fraud.data import ensure_dataset, load_raw
from fraud.eda import (
    plot_amount,
    plot_amount_by_class,
    plot_class_balance,
    plot_time_of_day,
    summarize,
)


def main():
    # Initial setup
    set_seed()

    ensure_dataset()
    df = load_raw()

    summarize(df)
    df.info()

    plot_class_balance(df)
    plot_amount(df)
    plot_amount_by_class(df)
    plot_time_of_day(df)
    print(f"\nFigures saved to {FIGURES_DIR}")

    ###########################################################
    ### Preprocess
    ###########################################################

    # As we can see in the figure, time data is important. In Kaggle:
    # Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.

    # Convert seconds to hours within a 24-hour cycle.
    df["hour"] = (df["Time"] / 3600) % 24

    # Cyclical encoding: represent hour as a point on a circle.
    # This ensures hour 23 and hour 0 are close together in feature space.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Drop the raw Time and intermediate hour columns
    df = df.drop(columns=["Time", "hour"])

    # Separate labels from data
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_normal = X[y == 0]
    X_fraud = X[y == 1]

    X_normal_train, X_normal_test = train_test_split(X_normal, test_size=0.1)

    print(f"\nTrain (normal) : {len(X_normal_train):,}")
    print(f"Val   (normal) : {len(X_normal_test):,}")
    print(f"Test  (fraud)  : {len(X_fraud):,}")

    # Normalize features
    scaler = StandardScaler()
    X_normal_train = scaler.fit_transform(X_train_raw)  # fit + transform
    X_normal_test = scaler.transform(X_val_raw)  # transform only
    X_fraud = scaler.transform(X_fraud)  # transform only
    X_normal = scaler.transform(X_normal)


if __name__ == "__main__":
    main()
