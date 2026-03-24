import torch
import torch.nn as nn
import timm
from abc import ABC, abstractmethod


class BaseExtractor(nn.Module, ABC):
    """Base class for pretrained feature extractors.

    Subclasses implement forward() to define how raw backbone output is pooled into a flat feature vector.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.embed_dim: int = self.backbone.num_features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a (B, embed_dim) feature tensor."""
        ...
