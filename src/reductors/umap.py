import numpy as np
from sklearn.preprocessing import normalize
from umap import UMAP

from .base import BaseReductor


class UmapReductor(BaseReductor):
    """UMAP dimensionality reduction."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self._umap: UMAP | None = None

    def fit(self, features: np.ndarray) -> "UmapReductor":
        self._umap = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )
        self._umap.fit(normalize(features, norm="l2"))
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._umap is None:
            raise RuntimeError("Call fit() before transform().")
        return self._umap.transform(normalize(features, norm="l2"))
