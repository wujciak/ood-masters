import numpy as np
from sklearn.preprocessing import normalize

from src.reductors.base import BaseReductor


class L2NormReductor(BaseReductor):
    def fit(self, features: np.ndarray) -> "L2NormReductor":
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return normalize(features, norm="l2")
