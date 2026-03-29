import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .base_detector import BaseDetector


class DbscanDetector(BaseDetector):
    """OOD scoring through distance to nearest core point. Falls back to all points if none found."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self._nn: NearestNeighbors | None = None

    def fit(self, id_features: np.ndarray) -> "DbscanDetector":
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(id_features)
        core = db.core_sample_indices_
        ref = id_features[core] if len(core) > 0 else id_features
        self._nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(ref)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("Call fit() before score().")
        distances, _ = self._nn.kneighbors(features)
        return distances[:, 0]
