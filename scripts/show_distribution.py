"""Sample counts per experiment split."""

from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from src.datasets.medmnist_loader import filter_by_classes, load_split

ID_CLASSES = [0, 2, 3, 4, 5]
NEAR_OOD_CLASSES = [6, 7, 8]


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    root = cfg["data"]["root"]

    path_train = load_split("pathmnist", "train", root, image_size=28)
    path_test = load_split("pathmnist", "test", root, image_size=28)
    pneumonia = load_split("pneumoniamnist", "test", root, image_size=28)

    id_total = len(filter_by_classes(path_train, ID_CLASSES))
    n_splits = 5
    counts = {
        "ID (val fold)": id_total // n_splits,
        "Near-OOD": len(filter_by_classes(path_test, NEAR_OOD_CLASSES)),
        "Far-OOD": len(pneumonia),
    }
    colors = ["#4878CF", "#D65F5F", "#FF7F0E"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.keys(), counts.values(), color=colors, alpha=0.85)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("Samples")
    ax.set_ylim(0, max(counts.values()) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = Path("data/plots/distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
