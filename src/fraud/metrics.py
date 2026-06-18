import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud.config import SEED


def get_f1_maximizing_threshold(y_true, y_scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)

    best_idx = f1.argmax()
    return thresholds[best_idx]


def get_recall_threshold(y_true, y_scores, min_recall=0.95) -> float:
    _, recall, thresholds = precision_recall_curve(y_true, y_scores)

    valid = np.where(recall[:-1] >= min_recall)[0]

    if len(valid) == 0:
        raise ValueError(f"No threshold achieves recall >= {min_recall:.3f}")

    # highest threshold satisfying recall constraint
    idx = valid[-1]

    return float(thresholds[idx])


def get_precision_threshold(y_true, y_scores, min_precision=0.95) -> float:
    precision, _, thresholds = precision_recall_curve(y_true, y_scores)

    valid = np.where(precision[:-1] >= min_precision)[0]

    if len(valid) == 0:
        raise ValueError(f"No threshold achieves precision >= {min_precision:.3f}")

    # lowest threshold satisfying precision constraint
    idx = valid[0]

    return float(thresholds[idx])


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """
    Resamples the test set with replacement n_boot times and recomputes
    metric_fn(y_true, y_scores) on each resample. This quantifies the
    uncertainty of a single test evaluation without retraining any model,
    this is essential here because the One-Class SVM costs about an hour to fit,
    so multi-seed retraining is infeasible

    Returns (point, lo, hi) where point is the metric on the full test set
    and [lo, hi] is the (1 - alpha) percentile interval. Resamples that happen
    to contain a single class (no fraud) are skipped, since AP/ROC are undefined
    there
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(metric_fn(y_true, y_scores))

    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # need both classes for AP / ROC-AUC
            continue
        stats.append(metric_fn(yt, y_scores[idx]))

    stats = np.asarray(stats)
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return point, lo, hi


def bootstrap_pr_roc_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, tuple[float, float, float]]:
    # PR-AUC is the primary metric under extreme imbalance
    return {
        "pr_auc": bootstrap_metric_ci(
            y_true, y_scores, average_precision_score, n_boot, alpha, seed
        ),
        "roc_auc": bootstrap_metric_ci(y_true, y_scores, roc_auc_score, n_boot, alpha, seed),
    }


def bootstrap_precision_recall_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict[str, tuple[float, float, float]]:
    """
    Computes bootstrapped CI for threshold-dependent metrics (Precision and Recall).
    Reuses the existing `bootstrap_metric_ci` function by converting continuous
    scores to binary predictions based on the provided threshold.
    """
    # Convert continuous scores to hard binary predictions
    y_pred = (y_scores >= threshold).astype(int)

    # Wrappers to handle edge cases where a bootstrap sample predicts 0 frauds (TP+FP=0)
    def safe_precision(yt, yp):
        return precision_score(yt, yp, zero_division=0)

    def safe_recall(yt, yp):
        return recall_score(yt, yp, zero_division=0)

    return {
        "precision": bootstrap_metric_ci(y_true, y_pred, safe_precision, n_boot, alpha, seed),
        "recall": bootstrap_metric_ci(y_true, y_pred, safe_recall, n_boot, alpha, seed),
    }
