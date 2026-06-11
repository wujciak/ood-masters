"""Generate statistical figure visualisations: Nemenyi heatmaps and CD diagrams."""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy.stats import rankdata

from _constants import (
    ARCHS,
    DETECTORS,
    DETECTOR_LABELS,
    METRICS,
    METRIC_LABELS,
    SCENARIOS,
    SCENARIO_LABELS,
    SPACES,
    SPACE_LABELS,
    latest_npz,
)

ALPHA = 0.05
OUT_DIR = Path("data/plots/stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_scores(npz_path: Path) -> dict:
    data = np.load(npz_path)
    return {
        (arch, space, det, scen, metric): float(data[key].mean())
        for arch, space, det, scen, metric in product(
            ARCHS, SPACES, DETECTORS, SCENARIOS, METRICS
        )
        if (key := f"{arch}__{space}__{det}__{scen}__{metric}") in data
    }


def get_groups(scores: dict, metric: str, scen: str, mode: str) -> dict:
    if mode == "spaces":
        return {
            sp: [
                scores.get((arch, sp, det, scen, metric), np.nan)
                for arch, det in product(ARCHS, DETECTORS)
            ]
            for sp in SPACES
        }
    else:  # detectors
        return {
            det: [
                scores.get((arch, sp, det, scen, metric), np.nan)
                for arch, sp in product(ARCHS, SPACES)
            ]
            for det in DETECTORS
        }


def nemenyi_heatmap(
    groups: dict,
    group_keys: list[str],
    labels: dict[str, str],
    title: str,
    path: Path,
) -> None:
    matrix = np.array([groups[g] for g in group_keys])
    pvals = sp.posthoc_nemenyi_friedman(matrix.T)
    pvals.index = pvals.columns = [labels[g] for g in group_keys]

    n = len(group_keys)
    fig, ax = plt.subplots(figsize=(n * 1.5 + 0.8, n * 1.2 + 0.6))

    mask = np.eye(n, dtype=bool)
    sns.heatmap(
        pvals,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        mask=mask,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "p-value"},
    )
    for i in range(n):
        for j in range(n):
            if i != j and pvals.values[i, j] < ALPHA:
                ax.text(
                    j + 0.5, i + 0.82, "*", ha="center", va="center",
                    color="black", fontsize=14, fontweight="bold",
                )

    ax.set_title(title, pad=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def cd_diagram(
    groups: dict,
    group_keys: list[str],
    labels: dict[str, str],
    title: str,
    path: Path,
) -> None:
    matrix = np.array([groups[g] for g in group_keys])
    pvals = sp.posthoc_nemenyi_friedman(matrix.T)
    pvals.index = pvals.columns = [labels[g] for g in group_keys]

    ranked = np.apply_along_axis(lambda x: rankdata(-x), 0, matrix)
    avg_ranks = pd.Series(
        {labels[g]: ranked[i].mean() for i, g in enumerate(group_keys)}
    )

    fig, ax = plt.subplots(figsize=(8, 2.5))
    try:
        sp.critical_difference_diagram(avg_ranks, pvals, ax=ax)
    except TypeError:
        sp.critical_difference_diagram(avg_ranks, pvals)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")
        return

    ax.set_title(title, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def main() -> None:
    npz_path = latest_npz()
    print(f"Loading {npz_path}")
    scores = load_scores(npz_path)

    for metric in METRICS:
        for scen in SCENARIOS:
            slabel = SCENARIO_LABELS[scen]
            mlabel = METRIC_LABELS[metric]
            print(f"\n{slabel} / {mlabel}")

            for mode, keys, lbls in [
                ("spaces", SPACES, SPACE_LABELS),
                ("detectors", DETECTORS, DETECTOR_LABELS),
            ]:
                groups = get_groups(scores, metric, scen, mode)
                stem = f"{mode}_{scen}_{metric}"
                group_label = "Feature Spaces" if mode == "spaces" else "Distance Metrics"

                nemenyi_heatmap(
                    groups, keys, lbls,
                    f"Nemenyi p-values — {group_label}\n{slabel}, {mlabel}",
                    OUT_DIR / f"nemenyi_{stem}.png",
                )
                cd_diagram(
                    groups, keys, lbls,
                    f"CD Diagram — {group_label} ({slabel}, {mlabel})",
                    OUT_DIR / f"cd_{stem}.png",
                )


if __name__ == "__main__":
    main()
