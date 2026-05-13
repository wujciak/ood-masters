"""Experiment: effect of threshold percentile q on balanced accuracy."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.evaluation.metrics import balanced_acc
from src.ood.dod import DODDetector
from src.training.feature_pipeline import load_embeddings

Q_VALUES = [80, 85, 90, 95, 97, 99]
N_SPLITS = 5
SEED = 1410
N_CLUSTERS = 10
ARCHITECTURES = ["vit", "cnn"]
SCENARIOS = ["near_ood", "far_ood"]
CACHE_DIR = Path("data/embeddings")
RESULTS_DIR = Path("data/results")


def run_sweep(embeddings: dict) -> pd.DataFrame:
    id_feats, _ = embeddings["id_train"]
    ood_splits = {name: embeddings[name][0] for name in SCENARIOS}
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []

    for q in Q_VALUES:
        print(f"  q={q}...", flush=True)
        fold_bal: dict[str, list] = {s: [] for s in SCENARIOS}

        for train_idx, val_idx in kf.split(id_feats):
            detector = DODDetector(
                n_clusters=N_CLUSTERS,
                metric="mahalanobis",
                threshold_percentile=q,
                random_state=SEED,
            )
            detector.fit(id_feats[train_idx])
            id_preds = detector.predict(id_feats[val_idx])

            for scenario, ood_feats in ood_splits.items():
                ood_preds = detector.predict(ood_feats)
                fold_bal[scenario].append(balanced_acc(id_preds, ood_preds))

        for scenario in SCENARIOS:
            rows.append(
                {
                    "q": q,
                    "scenario": scenario,
                    "bal_acc_mean": np.mean(fold_bal[scenario]),
                    "bal_acc_std": np.std(fold_bal[scenario], ddof=1),
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
                sub["q"],
                sub["bal_acc_mean"],
                marker="o",
                color=colors[arch],
                label=arch.upper(),
            )
            ax.fill_between(
                sub["q"],
                sub["bal_acc_mean"] - sub["bal_acc_std"],
                sub["bal_acc_mean"] + sub["bal_acc_std"],
                alpha=0.2,
                color=colors[arch],
            )
        ax.axvline(x=95, color="gray", linestyle="--", linewidth=1, label="q=95")
        ax.set_title(scenario.replace("_", "-").upper(), fontsize=10)
        ax.set_xlabel("Percentile q")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xticks(Q_VALUES)
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

    plot_results(results, Path("data/plots/results/threshold.png"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = [df.assign(architecture=arch) for arch, df in results.items()]
    pd.concat(frames, ignore_index=True).to_csv(
        RESULTS_DIR / "threshold.csv", index=False
    )
    print(f"Results saved to {RESULTS_DIR / 'threshold.csv'}")


if __name__ == "__main__":
    main()
