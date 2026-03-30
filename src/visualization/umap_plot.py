from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Keys that share the same visual style get the same entry in _STYLES.
# id_train and id_test are merged under one "ID" legend entry.
_STYLES: dict[str, dict] = {
    "id_train": {"color": "#4878CF", "marker": "o", "label": "ID"},
    "id_test": {"color": "#4878CF", "marker": "o", "label": "ID"},
    "near_ood": {"color": "#D65F5F", "marker": "o", "label": "Near-OOD"},
    "far_ood": {"color": "#FF7F0E", "marker": "x", "label": "Far-OOD"},
}
_DEFAULT_STYLE = {"color": "#888888", "marker": "o", "label": None}


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
    legend_seen: set[str] = set()

    for name, coords in projections.items():
        if len(coords) > max_points:
            coords = coords[rng.choice(len(coords), max_points, replace=False)]

        style = _STYLES.get(name, _DEFAULT_STYLE)
        label = style["label"] if style["label"] not in legend_seen else None
        if style["label"]:
            legend_seen.add(style["label"])

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=style["color"],
            marker=style["marker"],
            alpha=alpha,
            s=point_size if style["marker"] == "o" else point_size * 3,
            linewidths=0.8,
            label=label,
            rasterized=True,
        )

    ax.legend(loc="best", fontsize=9)
    ax.set(title=title, xlabel="UMAP-1", ylabel="UMAP-2")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")

    plt.show()
    plt.close(fig)
