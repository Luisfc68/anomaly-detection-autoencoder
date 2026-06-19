import time

import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from fraud.config import (
    COST_FN,
    COST_FP,
    COST_RATIOS,
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
    plot_pr_curve,
    plot_roc_curve,
    plot_time_of_day,
    summarize,
    summarize_duplicates,
)
from fraud.fraud_autoencoder import FraudAutoencoder
from fraud.metrics import (
    bootstrap_pr_roc_ci,
    bootstrap_precision_recall_ci,
    confusion_cost,
    cost_sensitivity,
    get_f1_maximizing_threshold,
    get_precision_threshold,
    get_recall_threshold,
    min_cost_threshold,
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

        val_scores_by_model = {name: data[f"{slug}_val"] for name, slug in _CACHE_KEYS.items()}
        test_scores_by_model = {name: data[f"{slug}_test"] for name, slug in _CACHE_KEYS.items()}
        # Load the new cached fit times
        fit_times = {name: float(data[f"{slug}_time"]) for name, slug in _CACHE_KEYS.items()}

        return val_scores_by_model, test_scores_by_model, fit_times

    fit_times = {}

    # Autoencoder
    train_loader = build_torch_data_loader(X_train, batch_size=32, shuffle=True)
    val_loader = build_torch_data_loader(X_val_normal, batch_size=32, shuffle=False)
    model = FraudAutoencoder(input_dim=input_dim, latent_dim=16, lr=1e-3).to(DEVICE)

    start_time = time.time()
    model.fit(train_loader, val_loader, epochs=100)
    fit_times["Autoencoder"] = time.time() - start_time

    val_scores_by_model = {"Autoencoder": model.reconstruction_errors(X_val)}
    test_scores_by_model = {"Autoencoder": model.reconstruction_errors(X_test)}

    # Baselines
    baselines = [
        IsolationForestDetector(),
        OneClassSVMDetector(),
        GaussianDensityDetector(),
    ]
    for detector in baselines:
        print(f"\nFitting baseline: {detector.name} ...")

        start_time = time.time()
        detector.fit(X_train)
        measured_time = time.time() - start_time

        # If the detector natively tracked its time, use that, else fallback to our measured time
        final_time = getattr(detector, "fit_time_seconds", measured_time)
        fit_times[detector.name] = final_time

        val_scores_by_model[detector.name] = detector.anomaly_score(X_val)
        test_scores_by_model[detector.name] = detector.anomaly_score(X_test)

        print(f"  full-sample fit time: {final_time:.1f} s")

    # Prepare dictionary for saving
    save_dict = {}
    for name, slug in _CACHE_KEYS.items():
        save_dict[f"{slug}_val"] = val_scores_by_model[name]
        save_dict[f"{slug}_test"] = test_scores_by_model[name]
        save_dict[f"{slug}_time"] = fit_times[name]

    np.savez(cache, **save_dict)
    print(f"\nSaved scores cache to {cache}")

    return val_scores_by_model, test_scores_by_model, fit_times


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

    val_scores_by_model, test_scores_by_model, fit_times = compute_or_load_scores(
        splitter.name, X_train, X_val_normal, X_val, X_test, input_dim
    )

    model_to_plot = "Autoencoder"
    model_scores = test_scores_by_model[model_to_plot]

    plot_roc_curve(
        y_true=y_test, y_scores=model_scores, model_name=model_to_plot, split_name=splitter.name
    )

    # Generate the single PR plot
    plot_pr_curve(
        y_true=y_test, y_scores=model_scores, model_name=model_to_plot, split_name=splitter.name
    )

    # Evaluation using CI
    prevalence = y_test.mean()
    results = {
        # Note: We pass test_scores_by_model here to evaluate the test set
        name: bootstrap_pr_roc_ci(y_test, scores)
        for name, scores in test_scores_by_model.items()
    }

    print("\n" + "=" * 86)
    print(f"Model comparison on test [{splitter.name}] - point [95% bootstrap CI]")
    print(f"No-skill PR-AUC baseline = prevalence = {prevalence:.4f}")
    print("=" * 86)
    print(f"{'Model':<18}{'Fit Time':>12}{'PR-AUC (primary)':>28}{'ROC-AUC':>28}")
    print("-" * 86)

    for name, ci in sorted(results.items(), key=lambda kv: kv[1]["pr_auc"][0], reverse=True):
        pr, roc = ci["pr_auc"], ci["roc_auc"]
        pr_str = f"{pr[0]:.4f} [{pr[1]:.4f}, {pr[2]:.4f}]"
        roc_str = f"{roc[0]:.4f} [{roc[1]:.4f}, {roc[2]:.4f}]"

        # Pull the specific time and format it
        time_str = f"{fit_times[name]:.1f}s"

        print(f"{name:<18}{time_str:>12}{pr_str:>28}{roc_str:>28}")

    for model_name, test_scores in test_scores_by_model.items():
        print(f"\n\n{'=' * 60}")
        print(f"Evaluating Threshold Strategies for: {model_name}")
        print(f"{'=' * 60}")

        val_scores = val_scores_by_model[model_name]
        thresholds_to_test = _operation_thresholds(y_val, val_scores)

        for strat_name, thresh in thresholds_to_test.items():
            print(f"\n--- Strategy: {strat_name} ---")
            print(f"Threshold Value: {thresh:.6f}")

            # Generate predictions using THIS model's threshold
            y_pred = (test_scores > thresh).astype(int)
            print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

            # Bootstrapped CI for Precision & Recall
            pr_metrics = bootstrap_precision_recall_ci(y_test, test_scores, threshold=thresh)

            p_pt, p_lo, p_hi = pr_metrics["precision"]
            r_pt, r_lo, r_hi = pr_metrics["recall"]

            print("  [95% Bootstrap CI for Fraud Class]")
            print(f"  -> Precision: {p_pt:.4f} [{p_lo:.4f}, {p_hi:.4f}]")
            print(f"  -> Recall:    {r_pt:.4f} [{r_lo:.4f}, {r_hi:.4f}]")

    # Minimum expected cost
    print("\n\n" + "=" * 72)
    print(
        f"Minimum-cost operating point [{splitter.name}]  "
        f"(C_FN={COST_FN:g}, C_FP={COST_FP:g})"
    )
    print("Threshold chosen on validation; FP/FN and cost reported on test.")
    print("=" * 72)
    print(f"{'Model':<18}{'Test FP':>9}{'Test FN':>9}{'Test cost':>12}")
    print("-" * 72)

    cost_rows = {}
    for model_name in test_scores_by_model:
        t_star = min_cost_threshold(
            y_val, val_scores_by_model[model_name], COST_FN, COST_FP
        )
        y_pred = (test_scores_by_model[model_name] >= t_star).astype(int)
        cost_rows[model_name] = confusion_cost(y_test, y_pred, COST_FN, COST_FP)

    for model_name, c in sorted(cost_rows.items(), key=lambda kv: kv[1]["cost"]):
        print(f"{model_name:<18}{c['fp']:>9}{c['fn']:>9}{c['cost']:>12.1f}")

    print("-" * 72)
    print(
        f"{'(flag nothing)':<18}{0:>9}{n_test_fraud:>9}{COST_FN * n_test_fraud:>12.1f}"
    )

    # Cost-ratio sensitivity (how the cost-optimal operating point shifts as a missed
    # fraud (FN) becomes more expensive vs a false alarm (FP))
    print("\n\n" + "=" * 72)
    print(f"Cost-ratio sensitivity [{splitter.name}]  (C_FP=1; columns are C_FN:C_FP)")
    print("Test cost per model; '*' = model's cost-optimal action is to flag nothing")
    print("=" * 72)
    header = f"{'Model':<18}" + "".join(f"{f'{r}:1':>11}" for r in COST_RATIOS)
    print(header)
    print("-" * len(header))

    for model_name in test_scores_by_model:
        sens = cost_sensitivity(
            y_val,
            val_scores_by_model[model_name],
            y_test,
            test_scores_by_model[model_name],
            COST_RATIOS,
        )
        cells = "".join(
            f"{sens[r]['cost']:>10.0f}{'*' if sens[r]['flags_nothing'] else ' '}"
            for r in COST_RATIOS
        )
        print(f"{model_name:<18}{cells}")

    print("-" * len(header))
    baseline = "".join(f"{r * n_test_fraud:>10.0f} " for r in COST_RATIOS)
    print(f"{'(flag nothing)':<18}{baseline}")


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

    # Inspect duplicates before dropping them
    summarize_duplicates(df)
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {n_before - len(df):,} rows -> {len(df):,} remain")

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
