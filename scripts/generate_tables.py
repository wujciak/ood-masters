from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES_DIR = Path("data/results/tables")
PLOTS_DIR = Path("data/plots")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SPACE_LABELS = {
    "raw": "Raw",
    "raw_l2": "Raw L2",
    "pca": "PCA",
    "random_subspace": "Rand. Subspace",
    "umap": "UMAP",
}
DETECTOR_LABELS = {
    "mahalanobis": "Mahalanobis",
    "minkowski_l1": "Manhattan",
    "minkowski_l2": "Euclidean",
    "minkowski_inf": "Chebyshev",
}
SPACE_ORDER = ["raw", "raw_l2", "pca", "random_subspace", "umap"]
DETECTOR_ORDER = ["mahalanobis", "minkowski_l1", "minkowski_l2", "minkowski_inf"]
METRICS = ["auroc", "aupr", "bal_acc"]
METRIC_LABELS = {"auroc": "AUROC", "aupr": "AUPR", "bal_acc": "Bal. Acc."}


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def make_table(df: pd.DataFrame, scenario: str, metric: str) -> str:
    sub = df[df["scenario"] == scenario].copy()
    col = metric
    col_std = f"{metric}_std"

    lines = []
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(f"Space & Distance & ViT & CNN \\\\")
    lines.append(r"\midrule")

    for space in SPACE_ORDER:
        space_rows = sub[sub["space"] == space]
        first = True
        for det in DETECTOR_ORDER:
            v = space_rows[
                (space_rows["architecture"] == "vit") & (space_rows["detector"] == det)
            ]
            c = space_rows[
                (space_rows["architecture"] == "cnn") & (space_rows["detector"] == det)
            ]
            if v.empty or c.empty:
                continue
            v, c = v.iloc[0], c.iloc[0]

            vit_val = fmt(v[col], v[col_std])
            cnn_val = fmt(c[col], c[col_std])
            if v[col] > c[col]:
                vit_val = bold(vit_val)
            elif c[col] > v[col]:
                cnn_val = bold(cnn_val)

            space_label = SPACE_LABELS[space] if first else ""
            first = False
            lines.append(
                f"{space_label} & {DETECTOR_LABELS[det]} & {vit_val} & {cnn_val} \\\\"
            )

        lines.append(r"\midrule" if space != SPACE_ORDER[-1] else "")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def make_bar_chart(
    df: pd.DataFrame, scenario: str, metric: str, save_path: Path
) -> None:
    sub = df[(df["scenario"] == scenario) & (df["detector"] == "mahalanobis")].copy()
    x = np.arange(len(SPACE_ORDER))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, arch in enumerate(["vit", "cnn"]):
        arch_sub = sub[sub["architecture"] == arch].set_index("space")
        means = [
            arch_sub.loc[s, metric] if s in arch_sub.index else 0 for s in SPACE_ORDER
        ]
        stds = [
            arch_sub.loc[s, f"{metric}_std"] if s in arch_sub.index else 0
            for s in SPACE_ORDER
        ]
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=stds,
            label=arch.upper(),
            capsize=4,
            color=["#4878CF", "#D65F5F"][i],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SPACE_LABELS[s] for s in SPACE_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(
        f"Mahalanobis-DOD, {scenario.replace('_', '-').upper()}, {METRIC_LABELS[metric]}"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    csv_files = sorted(Path("data/results/csv").glob("eval_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No results found in data/results/csv/. Run experiments/eval.py first."
        )
    df = pd.read_csv(csv_files[-1])

    for scenario in ["far_ood", "near_ood"]:
        for metric in METRICS:
            table = make_table(df, scenario, metric)
            path = TABLES_DIR / f"table_{scenario}_{metric}.tex"
            path.write_text(table)
            print(f"Saved {path}")
            make_bar_chart(
                df, scenario, metric, PLOTS_DIR / f"bar_{scenario}_{metric}.png"
            )


if __name__ == "__main__":
    main()
