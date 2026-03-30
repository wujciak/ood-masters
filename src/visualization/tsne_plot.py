from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

_SPLIT_LABEL = {
    "id_train": "ID",
    "id_test": "ID",
    "near_ood": "Near-OOD",
    "far_ood": "Far-OOD",
}

_STYLES = {
    "ID": {"color": "#4878CF", "marker": "o"},
    "Near-OOD": {"color": "#D65F5F", "marker": "o"},
    "Far-OOD": {"color": "#FF7F0E", "marker": "x"},
}


def plot_tsne(
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
    splits: list[str],
    title: str = "",
    save_path: str | Path | None = None,
    alpha: float = 0.4,
    point_size: int = 8,
    max_points: int = 5_000,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    rng = np.random.default_rng(random_state)

    merged: dict[str, np.ndarray] = {}
    for name in splits:
        label = _SPLIT_LABEL.get(name, name)
        chunk = embeddings[name][0]
        merged[label] = (
            np.concatenate([merged[label], chunk]) if label in merged else chunk
        )

    subsets = {}
    for label, feats in merged.items():
        if len(feats) > max_points:
            feats = feats[rng.choice(len(feats), max_points, replace=False)]
        subsets[label] = feats

    all_feats = normalize(np.concatenate(list(subsets.values())), norm="l2")
    print(f"Fitting t-SNE on {len(all_feats)} points...")
    coords = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1
    ).fit_transform(all_feats)

    projections, offset = {}, 0
    for label, feats in subsets.items():
        projections[label] = coords[offset : offset + len(feats)]
        offset += len(feats)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, pts in projections.items():
        style = _STYLES.get(label, {"color": "#888888", "marker": "o"})
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=style["color"],
            marker=style["marker"],
            alpha=alpha,
            s=point_size if style["marker"] == "o" else point_size * 3,
            linewidths=0.8,
            label=label,
            rasterized=True,
        )

    ax.legend(loc="best", fontsize=9)
    ax.set(title=title, xlabel="t-SNE-1", ylabel="t-SNE-2")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved {save_path}")

    plt.show()
    plt.close(fig)
