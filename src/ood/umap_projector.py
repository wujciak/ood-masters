import numpy as np
from sklearn.preprocessing import normalize
from umap import UMAP


class UmapProjector:
    """L2-normalises features then projects with UMAP."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        self._kwargs = dict(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )
        self._umap: UMAP | None = None

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return normalize(x, norm="l2")

    def fit(self, features: np.ndarray) -> "UmapProjector":
        self._umap = UMAP(**self._kwargs)
        self._umap.fit(self._norm(features))
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._umap is None:
            raise RuntimeError("Call fit() before transform().")
        return self._umap.transform(self._norm(features))
