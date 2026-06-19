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


def confusion_cost(y_true, y_pred, c_fn: float, c_fp: float) -> dict:
    """
    Total cost of a hard decision under an asymmetric cost matrix
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "cost": c_fn * fn + c_fp * fp}


def min_cost_threshold(y_true, y_scores, c_fn: float, c_fp: float) -> float:
    """
    Threshold that minimizes the expected cost ``c_fn * FN + c_fp * FP``
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_pos = int(y_true.sum())

    order = np.argsort(y_scores)[::-1]
    y_sorted = y_true[order]
    s_sorted = y_scores[order]

    tp = np.cumsum(y_sorted)                       # TP when flagging the top-k
    fp = np.arange(1, len(y_sorted) + 1) - tp      # FP when flagging the top-k
    fn = n_pos - tp
    cost_flag = c_fn * fn + c_fp * fp

    best_idx = int(np.argmin(cost_flag))
    # Flagging nothing misses every fraud: cost = c_fn * n_pos.
    if c_fn * n_pos <= cost_flag[best_idx]:
        return float("inf")
    return float(s_sorted[best_idx])


def cost_sensitivity(y_val, scores_val, y_test, scores_test, ratios, c_fp: float = 1.0):
    """
    Sensitivity of the cost-optimal operating point to the C_FN / C_FP ratio.

    For each ratio ``r`` (with ``c_fn = r * c_fp``), the cost-minimizing threshold is
    chosen on validation and the realized confusion counts and cost are reported on
    test.

    Returns ``{ratio: {"threshold", "fp", "fn", "cost", "flags_nothing"}}``.
    """
    out = {}
    for r in ratios:
        c_fn = r * c_fp
        t = min_cost_threshold(y_val, scores_val, c_fn, c_fp)
        y_pred = (scores_test >= t).astype(int)
        stats = confusion_cost(y_test, y_pred, c_fn, c_fp)
        stats["threshold"] = t
        stats["flags_nothing"] = not np.isfinite(t)
        out[r] = stats
    return out
