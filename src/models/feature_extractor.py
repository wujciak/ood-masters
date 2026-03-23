import torch
import torch.nn as nn
import timm
from typing import Literal


class FeatureExtractor(nn.Module):
    """Extracts feature embeddings from a pretrained model."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        pool: Literal["cls", "mean"] = "cls",
    ):
        super().__init__()
        self.pool = pool
        self.is_vit = "vit" in model_name.lower()

        # num_classes=0 removes the classification head
        self.layers = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.layers.eval()
        self.embed_dim = self.layers.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.layers.forward_features(x)
        if self.is_vit:
            return (
                features[:, 0, :]
                if self.pool == "cls"
                else features[:, 1:, :].mean(dim=1)
            )
        return features.mean(dim=[-2, -1])

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)
