import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fraud.config import DEVICE, FIGURES_DIR, SEED, set_seed
from fraud.data import build_torch_data_loader, ensure_dataset, load_raw
from fraud.eda import (
    plot_amount,
    plot_amount_by_class,
    plot_class_balance,
    plot_time_of_day,
    save_precision_recall_curve,
    summarize,
)
from fraud.fraud_autoencoder import FraudAutoencoder
from fraud.metrics import get_f1_maximizing_threshold, get_precision_threshold, get_recall_threshold


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

    # 'Time' is the seconds elapsed since the first transaction (span ~48 h).
    # Encode the time-of-day cyclically so that hour 23 and hour 0 are close in
    # feature space.
    df["hour"] = (df["Time"] / 3600) % 24
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df = df.drop(columns=["Time", "hour"])

    # 'Amount' is heavily right-skewed (skew ~16.98 -> ~0.16 after log1p).
    # Standardization is applied later, fitted on train only.
    df["Amount"] = np.log1p(df["Amount"])

    # V1..V28 are already PCA components no need to re-scale
    feature_cols = [c for c in df.columns if c != "Class"]

    ###########################################################
    ### Split — stratified three-way (train / val / test)
    ###########################################################
    # Stratify by Class so validation and test keep the real fraud proportion
    df_train, df_temp = train_test_split(
        df, test_size=0.30, stratify=df["Class"], random_state=SEED
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, stratify=df_temp["Class"], random_state=SEED
    )

    df_train = df_train[df_train["Class"] == 0].copy()

    # Standardize Amount. Scaler fitted on train only, then applied to val/test.
    scaler = StandardScaler()
    df_train["Amount"] = scaler.fit_transform(df_train[["Amount"]])
    df_val = df_val.copy()
    df_test = df_test.copy()
    df_val["Amount"] = scaler.transform(df_val[["Amount"]])
    df_test["Amount"] = scaler.transform(df_test[["Amount"]])

    X_train = df_train[feature_cols].to_numpy()
    X_val_normal = df_val.loc[df_val["Class"] == 0, feature_cols].to_numpy()

    X_val = df_val[feature_cols].to_numpy()
    y_val = df_val["Class"].to_numpy()

    X_test = df_test[feature_cols].to_numpy()
    y_test = df_test["Class"].to_numpy()

    INPUT_DIM = X_train.shape[1]
    print(f"\nTrain (legit only) : {len(X_train):,}")
    print(f"Val   (total/fraud): {len(df_val):,} / {int(df_val['Class'].sum()):,}")
    print(f"Test  (total/fraud): {len(df_test):,} / {int(df_test['Class'].sum()):,}")
    print(f"Input dimension    : {INPUT_DIM} features")

    ##########################################################
    ### Training
    ##########################################################

    train_loader = build_torch_data_loader(X_train, batch_size=32, shuffle=True)
    val_loader = build_torch_data_loader(X_val_normal, batch_size=32, shuffle=False)

    model = FraudAutoencoder(input_dim=INPUT_DIM, latent_dim=16, lr=1e-3).to(DEVICE)
    model.fit(train_loader, val_loader, epochs=50)

    errors_val = model.reconstruction_errors(X_val)

    # Define 3 thresholds:
    threshold_f1 = get_f1_maximizing_threshold(y_val, errors_val)
    threshold_rec = get_recall_threshold(y_val, errors_val, min_recall=0.75)
    threshold_prec = get_precision_threshold(y_val, errors_val, min_precision=0.75)

    thresholds_to_test = {
        "F1-Maximizing": threshold_f1,
        "High Recall (Catch more fraud)": threshold_rec,
        "High Precision (Fewer false alarms)": threshold_prec,
    }

    ##########################################################
    ### Evaluation
    ##########################################################

    errors_test = model.reconstruction_errors(X_test)

    save_precision_recall_curve(
        y_true=y_test,
        y_scores=errors_test,
        output_file=FIGURES_DIR / "precision_recall_curve.png",
    )

    # PR-AUC is the primary metric under extreme imbalance; ROC-AUC for reference.
    pr_auc = average_precision_score(y_test, errors_test)
    roc_auc = roc_auc_score(y_test, errors_test)
    print(f"PR-AUC  score: {pr_auc:.4f}")
    print(f"ROC-AUC score: {roc_auc:.4f}")

    print("=" * 40)

    # Evaluate each threshold strategy
    for name, thresh in thresholds_to_test.items():
        print(f"\n--- Strategy: {name} ---")
        print(f"Threshold Value: {thresh:.6f}")

        y_pred = (errors_test > thresh).astype(int)
        print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))


if __name__ == "__main__":
    main()
