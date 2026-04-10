from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.base import BaseExtractor


def extract_features(
    extractor: BaseExtractor, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    extractor.eval()
    extractor.to(device)
    features, labels = [], []
    with torch.no_grad():
        for images, labels_batch in tqdm(loader, leave=False):
            features.append(extractor(images.to(device)).cpu().numpy())
            labels.append(labels_batch.squeeze().numpy())
    return np.concatenate(features), np.concatenate(labels)


def extract_all(
    extractor: BaseExtractor, loaders: dict[str, DataLoader], device: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    results = {}
    for name, loader in loaders.items():
        print(f"  {name}...")
        results[name] = extract_features(extractor, loader, device)
    return results


def save_embeddings(
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{
            f"{k}_{t}": v
            for k, (f, l) in embeddings.items()
            for t, v in (("features", f), ("labels", l))
        },
    )


def load_embeddings(path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = np.load(Path(path))
    keys = {k.rsplit("_", 1)[0] for k in data.files}
    return {k: (data[f"{k}_features"], data[f"{k}_labels"]) for k in keys}
