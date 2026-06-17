import numpy as np
from scipy.stats import multivariate_normal

from fraud.models.base import AnomalyDetector


class GaussianDensityDetector(AnomalyDetector):
    name = "Gaussian Density"

    def __init__(self, reg_covar: float = 1e-6):
        self.reg_covar = reg_covar
        self._rv: multivariate_normal | None = None

    def fit(self, X_legit: np.ndarray) -> "GaussianDensityDetector":
        mean = X_legit.mean(axis=0)
        cov = np.cov(X_legit, rowvar=False)
        cov += self.reg_covar * np.eye(cov.shape[0])
        self._rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # negative log-likelihood: lower density -> higher anomaly score
        return -self._rv.logpdf(X)
