"""Extract and cache embeddings for all splits and architectures."""

from pathlib import Path

import torch
from src.config import load_config
from src.datasets.medmnist_loader import get_far_ood_loader, get_loaders
from src.models.cnn import CnnExtractor
from src.models.vit import VitExtractor
from src.training.feature_pipeline import extract_all, load_embeddings, save_embeddings
from src.visualization.tsne_plot import plot_tsne


def main() -> None:
    cfg = load_config()
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)

    device = cfg["training"]["device"]
    cache_dir = Path("data/embeddings")
    plots_dir = Path("data/plots/embeddings")

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

    for arch_name, extractor in architectures.items():
        cache_path = cache_dir / f"{arch_name}_embeddings.npz"
        if cache_path.exists():
            print(f"[{arch_name}] Cache exists, skipping.")
            embeddings = load_embeddings(cache_path)
        else:
            print(f"[{arch_name}] Extracting features...")
            loaders = {
                "id_train": split_loaders.id_train,
                "id_test": split_loaders.id_test,
                "near_ood": split_loaders.near_ood,
                "far_ood": far_ood_loader,
            }
            embeddings = extract_all(extractor, loaders, device)
            save_embeddings(embeddings, cache_path)
            print(f"[{arch_name}] Saved to {cache_path}")

        plot_tsne(
            embeddings,
            splits=["id_train", "id_test", "near_ood", "far_ood"],
            title=f"t-SNE {arch_name.upper()}",
            save_path=plots_dir / f"tsne_{arch_name}.png",
        )


if __name__ == "__main__":
    main()
