from abc import ABC, abstractmethod

import numpy as np


class BaseReductor(ABC):
    """Interface for dimensionality reduction methods."""

    @abstractmethod
    def fit(self, features: np.ndarray) -> "BaseReductor":
        """Fit on ID training features."""
        ...

    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project features to reduced space."""
        ...
