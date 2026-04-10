import numpy as np
from sklearn.preprocessing import normalize

from .base import BaseReductor


class RandomSubspaceReductor(BaseReductor):
    """Random Subspace method."""

    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self._indices: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "RandomSubspaceReductor":
        rng = np.random.default_rng(self.random_state)
        self._indices = rng.choice(
            features.shape[1], size=self.n_components, replace=False
        )
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._indices is None:
            raise RuntimeError("Call fit() before transform().")
        return normalize(features[:, self._indices], norm="l2")
