import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all
from src.ood.base_detector import BaseDetector
from src.ood.umap_projector import UmapProjector


def _run_detectors(
    detectors: dict[str, BaseDetector],
    feature_space: dict[str, np.ndarray],
    space_label: str,
) -> list[dict]:
    scenarios = {
        "near_ood": feature_space["near_ood"],
        "far_ood": feature_space["far_ood"],
    }
    rows = []
    for det_name, detector in detectors.items():
        print(f"  {det_name} [{space_label}]...")
        detector.fit(feature_space["id_train"])
        id_scores = detector.score(feature_space["id_test"])
        for scenario, ood_feats in scenarios.items():
            rows.append(
                {
                    "space": space_label,
                    "detector": det_name,
                    "scenario": scenario,
                    **compute_all(id_scores, detector.score(ood_feats)),
                }
            )
    return rows


def run_all(
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
    projector: UmapProjector,
    detectors: dict[str, BaseDetector],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run detectors on both raw embeddings and UMAP-projected features."""
    print("Fitting UMAP...")
    projector.fit(embeddings["id_train"][0])
    projections = {
        name: projector.transform(feats) for name, (feats, _) in embeddings.items()
    }

    raw_features = {name: feats for name, (feats, _) in embeddings.items()}

    rows = _run_detectors(detectors, raw_features, "raw")
    rows += _run_detectors(detectors, projections, "umap")

    return pd.DataFrame(rows), projections
