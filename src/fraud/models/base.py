from abc import ABC, abstractmethod

import numpy as np


class AnomalyDetector(ABC):
    """
    Common interface for the unsupervised baselines

    Every detector is trained on LEGITIMATE transactions only and exposes an
    anomaly_score where higher means more anomalous. This matches the
    autoencoder's reconstruction error, so the same evaluation code
    (average_precision_score(y, score) / roc_auc_score(y, score)) works
    unchanged across all four methods, without flipping signs per model.
    Note that scikit-learn's score_samples/decision_function use the
    opposite convention (higher = more normal); concrete subclasses negate them.
    """

    name: str = "AnomalyDetector"

    @abstractmethod
    def fit(self, X_legit: np.ndarray) -> "AnomalyDetector":
        """Fit the detector on legitimate transactions only. Returns self"""

    @abstractmethod
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Per-sample anomaly score, higher = more anomalous"""
