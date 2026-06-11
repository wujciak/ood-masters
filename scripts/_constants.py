"""Shared constants and helpers used across all scripts."""

from pathlib import Path

import yaml

SPACES = ["raw", "raw_l2", "pca", "random_subspace", "umap"]
DETECTORS = ["mahalanobis", "minkowski_l1", "minkowski_l2", "minkowski_inf"]
SCENARIOS = ["far_ood", "near_ood"]
ARCHS = ["vit", "cnn"]
METRICS = ["auroc", "aupr", "bal_acc"]

SPACE_LABELS = {
    "raw": "Raw",
    "raw_l2": "Raw L2",
    "pca": "PCA",
    "random_subspace": "Rand. Subspace",
    "umap": "UMAP",
}
DETECTOR_LABELS = {
    "mahalanobis": "Mahalanobis",
    "minkowski_l1": "Manhattan",
    "minkowski_l2": "Euclidean",
    "minkowski_inf": "Chebyshev",
}
SCENARIO_LABELS = {"far_ood": "Far-OOD", "near_ood": "Near-OOD"}
METRIC_LABELS = {"auroc": "AUROC", "aupr": "AUPR", "bal_acc": "Bal. Acc."}
ARCH_COLORS = {"vit": "#4878CF", "cnn": "#D65F5F"}


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def latest_npz() -> Path:
    files = sorted(Path("data/results").glob("eval_*.npz"))
    if not files:
        raise FileNotFoundError("No eval_*.npz files found in data/results/")
    return files[-1]
