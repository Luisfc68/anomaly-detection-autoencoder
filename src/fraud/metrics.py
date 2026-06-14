import numpy as np
from sklearn.metrics import precision_recall_curve


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
