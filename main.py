from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.datasets.medmnist_loader import get_far_ood_loader, get_loaders
from src.evaluation.runner import run_all
from src.models.cnn_extractor import CnnExtractor
from src.models.vit_extractor import VitExtractor
from src.ood.dbscan_detector import DbscanDetector
from src.ood.kmeans_detector import KMeansDetector
from src.ood.umap_projector import UmapProjector
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


def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg["training"]["seed"])

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

        projector = UmapProjector(
            n_neighbors=cfg["umap"]["n_neighbors"],
            min_dist=cfg["umap"]["min_dist"],
            random_state=cfg["training"]["seed"],
        )
        detectors = {
            "kmeans": KMeansDetector(
                n_clusters=cfg["ood"]["kmeans_clusters"],
                random_state=cfg["training"]["seed"],
            ),
            "dbscan": DbscanDetector(
                eps=cfg["ood"]["dbscan_eps"],
                min_samples=cfg["ood"]["dbscan_min_samples"],
            ),
        }

        results, projections = run_all(embeddings, projector, detectors)
        results.insert(0, "architecture", arch_name)
        all_results.append(results)

        vis_splits = ["id_train", "id_test", "near_ood", "far_ood"]
        plot_umap(
            {k: v for k, v in projections.items() if k in vis_splits},
            title=f"UMAP - {arch_name.upper()}",
            save_path=plots_dir / f"umap_{arch_name}.png",
        )
        plot_tsne(
            embeddings,
            splits=vis_splits,
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
