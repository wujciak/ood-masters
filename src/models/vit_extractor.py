import torch
from typing import Literal
from .base_extractor import BaseExtractor


class VitExtractor(BaseExtractor):
    """ViT feature extractor.

    pool="cls"  — returns the CLS token (index 0), standard for classification/OOD.
    pool="mean" — mean over patch tokens only (excludes CLS).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        pool: Literal["cls", "mean"] = "cls",
    ):
        super().__init__(model_name, pretrained)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)  # (B, N, D), N = 1 + num_patches
        if self.pool == "cls":
            return features[:, 0, :]            # cls
        return features[:, 1:, :].mean(dim=1)   # mean
