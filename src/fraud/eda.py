import matplotlib
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")  # save figures, no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew

from fraud.config import FIGURES_DIR


def summarize(df: pd.DataFrame) -> None:
    n = len(df)
    n_fraud = int(df["Class"].sum())
    rate = n_fraud / n
    print(f"Rows: {n:,} | Columns: {df.shape[1]}")
    print(f"Missing values (total): {int(df.isna().sum().sum())}")
    print(f"Duplicated rows: {int(df.duplicated().sum())}")
    print(f"Fraud: {n_fraud:,} ({rate:.4%}) | Legit: {n - n_fraud:,}")
    amt = df["Amount"].to_numpy()
    print("\nAmount summary (all transactions):")
    print(df["Amount"].describe().to_string())
    print("zeros:\t", int((amt == 0).sum()))
    print("skewness raw:   ", skew(amt))
    print("skewness log1p: ", skew(np.log1p(amt)))
    print("\nAmount median by class:")
    print(df.groupby("Class")["Amount"].median().to_string())
    span_h = (df["Time"].max() - df["Time"].min()) / 3600.0
    print(f"\nTime span: {df['Time'].min():.0f}..{df['Time'].max():.0f} s (~{span_h:.1f} h)")


def summarize_duplicates(df: pd.DataFrame) -> dict:
    redundant = df.duplicated()  # extra copies that drop_duplicates() will remove
    n_redundant = int(redundant.sum())
    n_redundant_fraud = int(df.loc[redundant, "Class"].sum())
    n_redundant_legit = n_redundant - n_redundant_fraud
    n_rows_involved = int(df.duplicated(keep=False).sum())

    print("\nDuplicate analysis (before dropping):")
    print(f"  Rows involved in any duplication : {n_rows_involved:,}")
    print(f"  Redundant copies (to be removed) : {n_redundant:,}")
    print(f"    of which fraud                 : {n_redundant_fraud:,}")
    print(f"    of which legit                 : {n_redundant_legit:,}")

    return {
        "redundant": n_redundant,
        "redundant_fraud": n_redundant_fraud,
        "redundant_legit": n_redundant_legit,
        "rows_involved": n_rows_involved,
    }


def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Legit (0)", "Fraud (1)"], counts.to_numpy(), color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Class balance — fraud = {counts[1] / counts.sum():.3%}")
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)


def plot_amount(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["Amount"], bins=100, color="#4C72B0")
    axes[0].set_yscale("log")
    axes[0].set_title("Amount (raw)")
    axes[0].set_xlabel("Amount")
    axes[1].hist(np.log1p(df["Amount"]), bins=100, color="#4C72B0")
    axes[1].set_title("log1p(Amount)")
    axes[1].set_xlabel("log1p(Amount)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "amount_distribution.png", dpi=150)
    plt.close(fig)


def plot_amount_by_class(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [np.log1p(df.loc[df["Class"] == c, "Amount"].to_numpy()) for c in (0, 1)]
    ax.boxplot(data, showfliers=False)
    ax.set_xticks([1, 2], ["Legit", "Fraud"])
    ax.set_ylabel("log1p(Amount)")
    ax.set_title("Transaction amount by class")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "amount_by_class.png", dpi=150)
    plt.close(fig)


def plot_time_of_day(df: pd.DataFrame) -> None:
    hour = (df["Time"].to_numpy() % (24 * 3600)) / 3600.0
    is_fraud = df["Class"].to_numpy() == 1
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(hour[~is_fraud], bins=48, density=True, alpha=0.6, label="Legit")
    ax.hist(hour[is_fraud], bins=48, density=True, alpha=0.6, label="Fraud")
    ax.set_xlabel("Within-day phase (Time mod 24h, in hours)")
    ax.set_ylabel("Density")
    ax.set_title("Transactions by within-day phase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "time_of_day.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true, y_scores, model_name, split_name):
    """
    Plots the ROC curve for a single model.
    """
    plt.figure(figsize=(9, 6))

    # --- Calculate ROC metrics ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # --- Plot the curves ---
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})", lw=2, color="#1f77b4")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="No Skill")

    # --- Formatting ---
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # --- Save and close ---
    plt.tight_layout()

    # Format the filename cleanly (e.g., "One-Class SVM" -> "one_class_svm")
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    output_path = FIGURES_DIR / f"roc_curve_{safe_name}_{split_name}.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {model_name} ROC curve to: {output_path}")


def plot_pr_curve(y_true, y_scores, model_name, split_name):
    """
    Plots the Precision-Recall curve for a single model.
    """
    plt.figure(figsize=(9, 6))

    # --- Calculate PR metrics ---
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    prevalence = y_true.mean()

    # --- Plot the curves ---
    plt.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.4f})", lw=2, color="#1f77b4")

    # The no-skill line for PR is a horizontal line at the prevalence rate
    plt.plot(
        [0, 1],
        [prevalence, prevalence],
        color="gray",
        linestyle="--",
        label=f"No Skill (Prevalence = {prevalence:.4f})",
    )

    # --- Formatting ---
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)

    # --- Save and close ---
    plt.tight_layout()

    # Format the filename cleanly (e.g., "One-Class SVM" -> "one_class_svm")
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    output_path = FIGURES_DIR / f"pr_curve_{safe_name}_{split_name}.png"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {model_name} PR curve to: {output_path}")
