from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.datasets.medmnist_loader import filter_by_classes, load_split

SPLITS = [
    ("PathMNIST (ID)", "pathmnist", "test", [0, 2, 3, 4, 5]),
    ("PathMNIST (Near-OOD)", "pathmnist", "test", [6, 7, 8]),
    ("PneumoniaMNIST (Far-OOD)", "pneumoniamnist", "test", None),
]
N_COLS = 3


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_images(name: str, split: str, classes: list[int] | None, n: int, root: str):
    dataset = load_split(name, split, root, image_size=64)
    if classes is not None:
        dataset = filter_by_classes(dataset, classes)
    indices = np.random.default_rng(0).choice(
        len(dataset), size=min(n, len(dataset)), replace=False
    )
    imgs = []
    for i in indices:
        img, _ = dataset[int(i)]
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        imgs.append(np.clip(img, 0, 1))
    return imgs


def main() -> None:
    cfg = load_config()
    root = cfg["data"]["root"]

    fig, axes = plt.subplots(
        len(SPLITS), N_COLS, figsize=(N_COLS * 2, len(SPLITS) * 2.2)
    )

    for row, (label, name, split, classes) in enumerate(SPLITS):
        imgs = get_images(name, split, classes, N_COLS, root)
        for col in range(N_COLS):
            ax = axes[row][col]
            ax.imshow(imgs[col])
            ax.axis("off")
        axes[row][1].set_title(label, fontsize=9, pad=4)

    plt.tight_layout()
    out = Path("data/plots/data/samples.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
