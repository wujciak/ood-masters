"""Experiment: effect of number of K-Means clusters on OOD detection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.evaluation.metrics import auroc, aupr
from src.ood.dod import DODDetector
from src.training.feature_pipeline import load_embeddings

K_VALUES = [2, 5, 10, 20, 30]
N_SPLITS = 5
SEED = 1410
THRESHOLD_PERCENTILE = 95.0
ARCHITECTURES = ["vit", "cnn"]
SCENARIOS = ["near_ood", "far_ood"]
CACHE_DIR = Path("data/embeddings")
RESULTS_DIR = Path("data/results")


def run_sweep(embeddings: dict) -> pd.DataFrame:
    id_feats, _ = embeddings["id_train"]
    ood_splits = {name: embeddings[name][0] for name in SCENARIOS}
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []

    for k in K_VALUES:
        print(f"  k={k}...", flush=True)
        fold_metrics: dict[str, dict[str, list]] = {
            s: {"auroc": [], "aupr": []} for s in SCENARIOS
        }

        for train_idx, val_idx in kf.split(id_feats):
            detector = DODDetector(
                n_clusters=k,
                metric="mahalanobis",
                threshold_percentile=THRESHOLD_PERCENTILE,
                random_state=SEED,
            )
            detector.fit(id_feats[train_idx])
            id_scores = detector.score(id_feats[val_idx])

            for scenario, ood_feats in ood_splits.items():
                ood_scores = detector.score(ood_feats)
                fold_metrics[scenario]["auroc"].append(auroc(id_scores, ood_scores))
                fold_metrics[scenario]["aupr"].append(aupr(id_scores, ood_scores))

        for scenario in SCENARIOS:
            m = fold_metrics[scenario]
            rows.append(
                {
                    "k": k,
                    "scenario": scenario,
                    "auroc_mean": np.mean(m["auroc"]),
                    "auroc_std": np.std(m["auroc"], ddof=1),
                    "aupr_mean": np.mean(m["aupr"]),
                    "aupr_std": np.std(m["aupr"], ddof=1),
                }
            )

    return pd.DataFrame(rows)


def plot_results(results: dict, save_path: Path) -> None:
    colors = {"vit": "#4878CF", "cnn": "#D65F5F"}
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(10, 4), sharey=False)

    for col, scenario in enumerate(SCENARIOS):
        ax = axes[col]
        for arch, df in results.items():
            sub = df[df["scenario"] == scenario]
            ax.plot(
                sub["k"],
                sub["auroc_mean"],
                marker="o",
                color=colors[arch],
                label=arch.upper(),
            )
            ax.fill_between(
                sub["k"],
                sub["auroc_mean"] - sub["auroc_std"],
                sub["auroc_mean"] + sub["auroc_std"],
                alpha=0.2,
                color=colors[arch],
            )
        ax.set_title(scenario.replace("_", "-").upper(), fontsize=10)
        ax.set_xlabel("K (clusters)")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    results = {}
    for arch in ARCHITECTURES:
        cache_path = CACHE_DIR / f"{arch}_embeddings.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"No cache for {arch}. Run extract.py first.")
        print(f"[{arch.upper()}]")
        results[arch] = run_sweep(load_embeddings(cache_path))

    plot_results(results, Path("data/plots/results/k_clusters.png"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = [df.assign(architecture=arch) for arch, df in results.items()]
    pd.concat(frames, ignore_index=True).to_csv(
        RESULTS_DIR / "k_clusters.csv", index=False
    )
    print(f"Results saved to {RESULTS_DIR / 'k_clusters.csv'}")


if __name__ == "__main__":
    main()
