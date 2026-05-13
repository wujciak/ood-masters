from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.evaluation.metrics import auroc
from src.ood.dod import DODDetector
from src.reductors.pca import PCAReductor
from src.reductors.random_subspace import RandomSubspaceReductor
from src.training.feature_pipeline import load_embeddings

N_COMPONENTS_VALUES = [8, 16, 32, 64, 128, 256, 512]
N_SPLITS = 5
SEED = 1410
N_CLUSTERS = 10
THRESHOLD_PERCENTILE = 95.0
ARCHITECTURES = ["vit", "cnn"]
SCENARIOS = ["near_ood", "far_ood"]
REDUCTORS = {"pca": PCAReductor, "random_subspace": RandomSubspaceReductor}
CACHE_DIR = Path("data/embeddings")
RESULTS_DIR = Path("data/results")


def run_sweep(embeddings: dict, reductor_cls) -> pd.DataFrame:
    id_feats, _ = embeddings["id_train"]
    ood_splits = {name: embeddings[name][0] for name in SCENARIOS}
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []

    for n_components in N_COMPONENTS_VALUES:
        print(f"  n_components={n_components}...", flush=True)
        fold_metrics: dict[str, list] = {s: [] for s in SCENARIOS}

        for train_idx, val_idx in kf.split(id_feats):
            kwargs = {"n_components": n_components}
            if reductor_cls is RandomSubspaceReductor:
                kwargs["random_state"] = SEED
            reductor = reductor_cls(**kwargs)
            reductor.fit(id_feats[train_idx])

            train_proj = reductor.transform(id_feats[train_idx])
            val_proj = reductor.transform(id_feats[val_idx])

            detector = DODDetector(
                n_clusters=N_CLUSTERS,
                metric="mahalanobis",
                threshold_percentile=THRESHOLD_PERCENTILE,
                random_state=SEED,
            )
            detector.fit(train_proj)
            id_scores = detector.score(val_proj)

            for scenario, ood_feats in ood_splits.items():
                ood_scores = detector.score(reductor.transform(ood_feats))
                fold_metrics[scenario].append(auroc(id_scores, ood_scores))

        for scenario in SCENARIOS:
            vals = fold_metrics[scenario]
            rows.append(
                {
                    "n_components": n_components,
                    "scenario": scenario,
                    "auroc_mean": np.mean(vals),
                    "auroc_std": np.std(vals, ddof=1),
                }
            )

    return pd.DataFrame(rows)


def plot_results(results: dict, save_path: Path) -> None:
    colors = {"pca": "#4878CF", "random_subspace": "#D65F5F"}
    labels = {"pca": "PCA", "random_subspace": "Random Subspace"}

    fig, axes = plt.subplots(
        len(SCENARIOS), len(ARCHITECTURES), figsize=(10, 6), sharex=True
    )

    for col, arch in enumerate(ARCHITECTURES):
        for row, scenario in enumerate(SCENARIOS):
            ax = axes[row][col]
            for name, df in results[arch].items():
                sub = df[df["scenario"] == scenario]
                ax.plot(
                    sub["n_components"],
                    sub["auroc_mean"],
                    marker="o",
                    color=colors[name],
                    label=labels[name],
                )
                ax.fill_between(
                    sub["n_components"],
                    sub["auroc_mean"] - sub["auroc_std"],
                    sub["auroc_mean"] + sub["auroc_std"],
                    alpha=0.2,
                    color=colors[name],
                )

            ax.set_title(
                f"{arch.upper()}, {scenario.replace('_', '-').upper()}", fontsize=10
            )
            ax.set_ylim(0, 1)
            ax.set_xscale("log", base=2)
            ax.set_xticks(N_COMPONENTS_VALUES)
            ax.set_xticklabels(N_COMPONENTS_VALUES)
            ax.grid(True, alpha=0.3)
            if row == len(SCENARIOS) - 1:
                ax.set_xlabel("n_components", fontsize=9)
            if col == 0:
                ax.set_ylabel("AUROC", fontsize=9)
            if row == 0 and col == 0:
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
        embeddings = load_embeddings(cache_path)
        results[arch] = {}

        for name, cls in REDUCTORS.items():
            print(f"  {name}...")
            results[arch][name] = run_sweep(embeddings, cls)

    plot_results(results, Path("data/plots/results/n_components.png"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = [
        df.assign(architecture=arch, reductor=name)
        for arch, arch_res in results.items()
        for name, df in arch_res.items()
    ]
    pd.concat(frames, ignore_index=True).to_csv(
        RESULTS_DIR / "n_components.csv", index=False
    )
    print(f"Results saved to {RESULTS_DIR / 'n_components.csv'}")


if __name__ == "__main__":
    main()
