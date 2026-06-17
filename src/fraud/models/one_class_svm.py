import time

import numpy as np
from sklearn.svm import OneClassSVM

from fraud.models.base import AnomalyDetector


class OneClassSVMDetector(AnomalyDetector):
    name = "One-Class SVM"

    def __init__(self, kernel: str = "rbf", gamma="scale", nu: float = 0.5):
        # nu upper-bounds the fraction of training points treated as outliers and
        # lower-bounds the fraction of support vectors. We keep scikit-learn's
        # defaults to avoid tuning on the evaluation data; only the continuous
        # score is consumed downstream.
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.fit_time_seconds: float | None = None

    def fit(self, X_legit: np.ndarray) -> "OneClassSVMDetector":
        start = time.perf_counter()
        self.model.fit(X_legit)
        self.fit_time_seconds = time.perf_counter() - start
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal -> negate so higher = anomalous
        return -self.model.decision_function(X)
