from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """OOD detector interface."""

    @abstractmethod
    def fit(self, id_features: np.ndarray) -> "BaseDetector":
        """Train on ID features and set the decision threshold."""
        ...

    @abstractmethod
    def score(self, features: np.ndarray) -> np.ndarray:
        """Continuous OOD score. Used for metrics."""
        ...

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Binary prediction (1=OOD, 0=ID) using threshold set during fit."""
        ...
