from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """OOD detector interface. Higher score = more likely OOD."""

    @abstractmethod
    def fit(self, id_features: np.ndarray) -> "BaseDetector": ...

    @abstractmethod
    def score(self, features: np.ndarray) -> np.ndarray: ...
