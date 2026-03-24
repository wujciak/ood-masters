import torch
from .base_extractor import BaseExtractor


class CnnExtractor(BaseExtractor):
    """CNN feature extractor. Global average pooling over spatial dims of backbone output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)  # (B, C, H, W)
        return features.mean(dim=[2, 3])  # (B, C)
