import numpy as np
from sklearn.cluster import KMeans

from .base_detector import BaseDetector


class KMeansDetector(BaseDetector):
    """OOD scoring through distance to nearest K-Means centroid."""

    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kmeans: KMeans | None = None

    def fit(self, id_features_2d: np.ndarray) -> "KMeansDetector":
        self._kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto"
        )
        self._kmeans.fit(id_features_2d)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._kmeans is None:
            raise RuntimeError("Call fit() before score().")
        dists = np.linalg.norm(
            features[:, None] - self._kmeans.cluster_centers_[None], axis=-1
        )
        return dists.min(axis=1)
