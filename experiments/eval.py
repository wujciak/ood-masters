"""Main experiment: compare all feature spaces and distance metrics via k-fold CV."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.kfold_runner import aggregate_folds, run_kfold
from src.ood.dod import DODDetector
from src.reductors.pca import PCAReductor
from src.reductors.random_subspace import RandomSubspaceReductor
from src.reductors.umap import UmapReductor
from src.training.feature_pipeline import load_embeddings

CACHE_DIR = Path("data/embeddings")
RESULTS_DIR = Path("data/results")
ARCHITECTURES = ["vit", "cnn"]


def build_reductors(cfg: dict, seed: int) -> dict:
    rc = cfg["reductors"]
    return {
        "raw": None,
        "pca": PCAReductor(n_components=rc["pca"]["n_components"]),
        "random_subspace": RandomSubspaceReductor(
            n_components=rc["random_subspace"]["n_components"], random_state=seed
        ),
        "umap": UmapReductor(
            n_neighbors=rc["umap"]["n_neighbors"],
            min_dist=rc["umap"]["min_dist"],
            n_components=rc["umap"]["n_components"],
            random_state=seed,
        ),
    }


def build_detectors(cfg: dict, seed: int) -> dict:
    dc = cfg["dod"]
    return {
        m["name"]: DODDetector(
            n_clusters=dc["n_clusters"],
            metric=m["metric"],
            p=m["p"],
            threshold_percentile=dc["threshold_percentile"],
            random_state=seed,
        )
        for m in dc["metrics"]
    }


def save_results(fold_df: pd.DataFrame, agg_df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_df.to_csv(RESULTS_DIR / f"eval_{timestamp}.csv", index=False)

    group_cols = ["architecture", "space", "detector", "scenario"]
    npz_data = {}
    for keys, group in fold_df.groupby(group_cols):
        prefix = "__".join(keys)
        for metric in ("auroc", "aupr", "bal_acc"):
            npz_data[f"{prefix}__{metric}"] = group[metric].to_numpy()
    np.savez(RESULTS_DIR / f"eval_{timestamp}.npz", **npz_data)
    print(f"Results saved to data/results/eval_{timestamp}.*")


def main() -> None:
    cfg = load_config()
    seed = cfg["training"]["seed"]

    all_fold: list[pd.DataFrame] = []
    all_agg: list[pd.DataFrame] = []

    for arch in ARCHITECTURES:
        cache_path = CACHE_DIR / f"{arch}_embeddings.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"No cache for {arch}. Run extract.py first.")

        print(f"\n[{arch.upper()}]")
        embeddings = load_embeddings(cache_path)
        reductors = build_reductors(cfg, seed)
        detectors = build_detectors(cfg, seed)

        fold_df = run_kfold(
            embeddings,
            reductors,
            detectors,
            n_splits=cfg["kfold"]["n_splits"],
            random_state=seed,
        )
        fold_df.insert(0, "architecture", arch)
        all_fold.append(fold_df)

        agg = aggregate_folds(fold_df)
        agg.insert(0, "architecture", arch)
        all_agg.append(agg)

    final_fold = pd.concat(all_fold, ignore_index=True)
    final_agg = pd.concat(all_agg, ignore_index=True)

    print("\n" + "=" * 70)
    print(final_agg.to_string(index=False))
    print("=" * 70)

    save_results(final_fold, final_agg)


if __name__ == "__main__":
    main()
