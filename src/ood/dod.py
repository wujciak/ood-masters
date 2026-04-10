from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from .base import BaseDetector

Metric = Literal["mahalanobis", "minkowski"]


class DODDetector(BaseDetector):
    """Distance-based OOD Detector (DOD).

    Fits centroids on ID features, then scores samples by their distance to the nearest centroid.
    The decision threshold is set as a percentile of ID training distances during fit.

    Supported metrics:
        - "minkowski" with p=1 (Manhattan), p=2 (Euclidean), p=inf (Chebyshev)
        - "mahalanobis"
    """

    def __init__(
        self,
        n_clusters: int = 10,
        metric: Metric = "minkowski",
        p: float = 2,
        threshold_percentile: float = 95.0,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.metric = metric
        self.p = p
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state

        self._centroids: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._threshold: float | None = None

    def _distances(self, features: np.ndarray) -> np.ndarray:
        if self.metric == "mahalanobis":
            dists = cdist(
                features, self._centroids, metric="mahalanobis", VI=self._cov_inv
            )
        else:
            p = np.inf if self.p == np.inf else self.p
            dists = cdist(features, self._centroids, metric="minkowski", p=p)
        return dists.min(axis=1)

    def fit(self, id_features: np.ndarray) -> "DODDetector":
        km = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto"
        )
        km.fit(id_features)
        self._centroids = km.cluster_centers_

        if self.metric == "mahalanobis":
            cov = np.cov(id_features, rowvar=False)
            self._cov_inv = np.linalg.pinv(cov)

        id_scores = self._distances(id_features)
        self._threshold = float(np.percentile(id_scores, self.threshold_percentile))
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._centroids is None:
            raise RuntimeError("Call fit() before score().")
        return self._distances(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._threshold is None:
            raise RuntimeError("Call fit() before predict().")
        return (self.score(features) > self._threshold).astype(int)
