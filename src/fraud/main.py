import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from fraud.config import (
    DEVICE,
    FIGURES_DIR,
    METRICS_DIR,
    SPLIT_STRATEGIES,
    set_seed,
)
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
from fraud.metrics import (
    bootstrap_pr_roc_ci,
    get_f1_maximizing_threshold,
    get_precision_threshold,
    get_recall_threshold,
)
from fraud.models.gaussian import GaussianDensityDetector
from fraud.models.isolation_forest import IsolationForestDetector
from fraud.models.one_class_svm import OneClassSVMDetector
from fraud.splits import get_splitter

_CACHE_KEYS = {
    "Autoencoder": "autoencoder",
    "Isolation Forest": "isolation_forest",
    "One-Class SVM": "one_class_svm",
    "Gaussian Density": "gaussian",
}


def _cache_path(split_name: str):
    return METRICS_DIR / f"scores_cache_{split_name}.npz"


def compute_or_load_scores(split_name, X_train, X_val_normal, X_val, X_test, input_dim):
    cache = _cache_path(split_name)
    if cache.exists():
        print(f"\nLoading cached scores from {cache}")
        print("(delete this file to retrain from scratch)")
        data = np.load(cache)
        errors_val = data["errors_val"]
        scores_by_model = {name: data[slug] for name, slug in _CACHE_KEYS.items()}
        return errors_val, scores_by_model

    # Autoencoder: trained on legit train, early-stopped on legit val
    train_loader = build_torch_data_loader(X_train, batch_size=32, shuffle=True)
    val_loader = build_torch_data_loader(X_val_normal, batch_size=32, shuffle=False)

    model = FraudAutoencoder(input_dim=input_dim, latent_dim=16, lr=1e-3).to(DEVICE)
    model.fit(train_loader, val_loader, epochs=50)

    errors_val = model.reconstruction_errors(X_val)
    scores_by_model = {"Autoencoder": model.reconstruction_errors(X_test)}

    # Baselines: fit on legit train only, score test
    baselines = [
        IsolationForestDetector(),
        OneClassSVMDetector(),
        GaussianDensityDetector(),
    ]
    for detector in baselines:
        print(f"\nFitting baseline: {detector.name} ...")
        detector.fit(X_train)
        scores_by_model[detector.name] = detector.anomaly_score(X_test)
        if getattr(detector, "fit_time_seconds", None) is not None:
            print(f"  full-sample fit time: {detector.fit_time_seconds:.1f} s")

    np.savez(
        cache,
        errors_val=errors_val,
        **{slug: scores_by_model[name] for name, slug in _CACHE_KEYS.items()},
    )
    print(f"\nSaved scores cache to {cache}")
    return errors_val, scores_by_model


def _operation_thresholds(y_val, errors_val):
    specs = {
        "F1-Maximizing": lambda: get_f1_maximizing_threshold(y_val, errors_val),
        "High Recall (Catch more fraud)": lambda: get_recall_threshold(
            y_val, errors_val, min_recall=0.75
        ),
        "High Precision (Fewer false alarms)": lambda: get_precision_threshold(
            y_val, errors_val, min_precision=0.75
        ),
    }
    thresholds = {}
    for name, fn in specs.items():
        try:
            thresholds[name] = fn()
        except ValueError as e:
            print(f"  [skipped operation point '{name}': {e}]")
    return thresholds


def run_split_experiment(df, splitter, feature_cols):
    print("\n" + "#" * 72)
    print(f"# SPLIT STRATEGY: {splitter.name}")
    print("#" * 72)

    df_train, df_val, df_test = splitter.split(df)

    # Autoencoder trains on legitimate transactions only.
    df_train = df_train[df_train["Class"] == 0].copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    # Standardize Amount. Scaler fitted on this split's train only (Rule 2).
    scaler = StandardScaler()
    df_train["Amount"] = scaler.fit_transform(df_train[["Amount"]])
    df_val["Amount"] = scaler.transform(df_val[["Amount"]])
    df_test["Amount"] = scaler.transform(df_test[["Amount"]])

    X_train = df_train[feature_cols].to_numpy()
    X_val_normal = df_val.loc[df_val["Class"] == 0, feature_cols].to_numpy()
    X_val = df_val[feature_cols].to_numpy()
    y_val = df_val["Class"].to_numpy()
    X_test = df_test[feature_cols].to_numpy()
    y_test = df_test["Class"].to_numpy()

    input_dim = X_train.shape[1]
    n_val_fraud, n_test_fraud = int(y_val.sum()), int(y_test.sum())
    print(f"\nTrain (legit only) : {len(X_train):,}")
    print(
        f"Val   (total/fraud): {len(X_val):,} / {n_val_fraud} "
        f"({n_val_fraud / len(X_val):.4%} fraud)"
    )
    print(
        f"Test  (total/fraud): {len(X_test):,} / {n_test_fraud} "
        f"({n_test_fraud / len(X_test):.4%} fraud)"
    )
    print(f"Input dimension    : {input_dim} features")

    errors_val, scores_by_model = compute_or_load_scores(
        splitter.name, X_train, X_val_normal, X_val, X_test, input_dim
    )
    errors_test = scores_by_model["Autoencoder"]

    save_precision_recall_curve(
        y_true=y_test,
        y_scores=errors_test,
        output_file=FIGURES_DIR / f"precision_recall_curve_{splitter.name}.png",
    )

    # Evaluation using CI
    # Bootstrap the test set (resample with replacement) to put a 95% CI around
    # each point estimate, without retraining any model. PR-AUC is primary; its
    # no-skill baseline is the prevalence, not 0.5
    prevalence = y_test.mean()
    results = {
        name: bootstrap_pr_roc_ci(y_test, scores)
        for name, scores in scores_by_model.items()
    }

    print("\n" + "=" * 72)
    print(f"Model comparison on test [{splitter.name}] - point [95% bootstrap CI]")
    print(f"No-skill PR-AUC baseline = prevalence = {prevalence:.4f}")
    print("=" * 72)
    print(f"{'Model':<18}{'PR-AUC (primary)':>26}{'ROC-AUC':>26}")
    print("-" * 72)
    for name, ci in sorted(
        results.items(), key=lambda kv: kv[1]["pr_auc"][0], reverse=True
    ):
        pr, roc = ci["pr_auc"], ci["roc_auc"]
        pr_str = f"{pr[0]:.4f} [{pr[1]:.4f}, {pr[2]:.4f}]"
        roc_str = f"{roc[0]:.4f} [{roc[1]:.4f}, {roc[2]:.4f}]"
        print(f"{name:<18}{pr_str:>26}{roc_str:>26}")

    print("\n" + "=" * 40)
    thresholds_to_test = _operation_thresholds(y_val, errors_val)
    for name, thresh in thresholds_to_test.items():
        print(f"\n--- Strategy: {name} ---")
        print(f"Threshold Value: {thresh:.6f}")
        y_pred = (errors_test > thresh).astype(int)
        print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))


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
    # feature space. 'Time' itself is kept so the temporal splitter can order by it,
    # but it is excluded from the feature columns below
    df["hour"] = (df["Time"] / 3600) % 24
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df = df.drop(columns=["hour"])

    # 'Amount' is heavily right-skewed (skew ~16.98 -> ~0.16 after log1p).
    # Standardization is stateful and is applied per split, fitted on train only.
    df["Amount"] = np.log1p(df["Amount"])

    # V1..V28 are already PCA components, no need to re-scale. 'Time' and 'Class'
    # are not model inputs.
    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]

    ###########################################################
    ### Run and compare each split strategy
    ###########################################################
    for strategy in SPLIT_STRATEGIES:
        splitter = get_splitter(strategy)
        run_split_experiment(df, splitter, feature_cols)


if __name__ == "__main__":
    main()
