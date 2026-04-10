import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from .base import BaseReductor


class PCAReductor(BaseReductor):
    """Principal Component Analysis."""

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self._pca: PCA | None = None

    def fit(self, features: np.ndarray) -> "PCAReductor":
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(normalize(features, norm="l2"))
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Call fit() before transform().")
        return self._pca.transform(normalize(features, norm="l2"))
