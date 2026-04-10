from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.datasets.medmnist_loader import get_far_ood_loader, get_loaders
from src.evaluation.kfold_runner import aggregate_folds, run_kfold
from src.models.cnn import CnnExtractor
from src.models.vit import VitExtractor
from src.ood.dod import DODDetector
from src.reductors.pca import PCAReductor
from src.reductors.random_subspace import RandomSubspaceReductor
from src.reductors.umap import UmapReductor
from src.training.feature_pipeline import extract_all, load_embeddings, save_embeddings
from src.visualization.tsne_plot import plot_tsne
from src.visualization.umap_plot import plot_umap


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_or_extract(
    arch_name: str,
    extractor,
    split_loaders,
    far_ood_loader,
    device: str,
    cache_dir: Path,
) -> dict:
    cache_path = cache_dir / f"{arch_name}_embeddings.npz"
    if cache_path.exists():
        print(f"[{arch_name}] Loading cached embeddings from {cache_path}")
        return load_embeddings(cache_path)

    print(f"[{arch_name}] Extracting features...")
    loaders = {
        "id_train": split_loaders.id_train,
        "id_val": split_loaders.id_val,
        "id_test": split_loaders.id_test,
        "near_ood": split_loaders.near_ood,
        "far_ood": far_ood_loader,
    }
    embeddings = extract_all(extractor, loaders, device)
    save_embeddings(embeddings, cache_path)
    print(f"[{arch_name}] Embeddings saved to {cache_path}")
    return embeddings


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


def main() -> None:
    cfg = load_config()
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)

    device = cfg["training"]["device"]
    cache_dir = Path("data/embeddings")
    plots_dir = Path("data/plots")

    data_cfg = cfg["data"]
    split_loaders = get_loaders(
        name=data_cfg["primary_dataset"],
        id_classes=data_cfg["in_dist_classes"],
        near_ood_classes=data_cfg["near_ood_classes"],
        batch_size=cfg["training"]["batch_size"],
        root=data_cfg["root"],
        image_size=data_cfg["image_size"],
        num_workers=data_cfg["num_workers"],
    )
    far_ood_loader = get_far_ood_loader(
        dataset=data_cfg["far_ood"],
        batch_size=cfg["training"]["batch_size"],
        root=data_cfg["root"],
        image_size=data_cfg["image_size"],
        num_workers=data_cfg["num_workers"],
    )

    architectures = {
        "vit": VitExtractor(cfg["models"]["vit"]),
        "cnn": CnnExtractor(cfg["models"]["cnn"]),
    }

    all_results: list[pd.DataFrame] = []

    for arch_name, extractor in architectures.items():
        print(f"\n{'='*50}")
        print(f" Architecture: {arch_name.upper()}")
        print(f"{'='*50}")

        embeddings = get_or_extract(
            arch_name, extractor, split_loaders, far_ood_loader, device, cache_dir
        )

        projectors = build_reductors(cfg, seed)
        detectors = build_detectors(cfg, seed)

        print("Running k-fold evaluation...")
        fold_results = run_kfold(
            embeddings,
            projectors,
            detectors,
            n_splits=cfg["kfold"]["n_splits"],
            random_state=seed,
        )
        results = aggregate_folds(fold_results)
        results.insert(0, "architecture", arch_name)
        all_results.append(results)

        # UMAP visualization on full id_train set not only fold
        umap_proj = projectors["umap"]
        umap_proj.fit(embeddings["id_train"][0])
        umap_projections = {
            name: umap_proj.transform(feats)
            for name, (feats, _) in embeddings.items()
            if name != "id_val"
        }
        plot_umap(
            umap_projections,
            title=f"UMAP - {arch_name.upper()}",
            save_path=plots_dir / f"umap_{arch_name}.png",
        )
        plot_tsne(
            embeddings,
            splits=["id_train", "id_test", "near_ood", "far_ood"],
            title=f"t-SNE - {arch_name.upper()}",
            save_path=plots_dir / f"tsne_{arch_name}.png",
        )

    final = pd.concat(all_results, ignore_index=True)
    print("\n" + "=" * 70)
    print(final.to_string(index=False))
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"data/results_{timestamp}.csv")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
