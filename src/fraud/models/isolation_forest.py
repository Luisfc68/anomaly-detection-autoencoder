import numpy as np
from sklearn.ensemble import IsolationForest

from fraud.config import SEED
from fraud.models.base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    name = "Isolation Forest"

    def __init__(self, n_estimators: int = 100):
        # contamination="auto"`` is used because we never
        # expose the detector to the real fraud rate during
        # training; we only consume the continuous score, not its
        # internal binary decision, so the contamination value
        # does not affect ranking
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=SEED,
            n_jobs=-1,
        )

    def fit(self, X_legit: np.ndarray) -> "IsolationForestDetector":
        self.model.fit(X_legit)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # score_samples: higher = more normal -> negate so higher = more anomalous
        return -self.model.score_samples(X)
