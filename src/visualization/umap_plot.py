from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_COLORS = {
    "id_train": "#4878CF",
    "id_val": "#6ACC65",
    "id_test": "#2ca02c",
    "near_ood": "#D65F5F",
    "far_ood": "#B47CC7",
}


def _color_for(name: str) -> str:
    return _COLORS.get(name, "#888888")


def plot_umap(
    projections: dict[str, np.ndarray],
    title: str = "",
    save_path: str | Path | None = None,
    alpha: float = 0.4,
    point_size: int = 8,
    max_points: int = 5_000,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(0)

    for name, coords in projections.items():
        if len(coords) > max_points:
            coords = coords[rng.choice(len(coords), max_points, replace=False)]
        color = _color_for(name)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=color,
            alpha=alpha,
            s=point_size,
            rasterized=True,
        )

    ax.legend(
        handles=[mpatches.Patch(color=_color_for(n), label=n) for n in projections],
        loc="best",
        fontsize=9,
    )
    ax.set(title=title, xlabel="UMAP-1", ylabel="UMAP-2")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")

    plt.show()
    plt.close(fig)
