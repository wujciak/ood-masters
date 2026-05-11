"""Generate LaTeX tables and bar charts from k-fold results CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_CSV = sorted(Path("data").glob("results_*.csv"))[-1]
OUT_DIR = Path("data/plots/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPACE_LABELS = {
    "raw": "Raw",
    "pca": "PCA",
    "random_subspace": "Rand. Subspace",
    "umap": "UMAP",
}
METRIC_LABELS = {
    "mahalanobis": "Mahalanobis",
    "minkowski_l1": r"$\ell_1$",
    "minkowski_l2": r"$\ell_2$",
    "minkowski_inf": r"$\ell_\infty$",
}
SPACE_ORDER = ["raw", "pca", "random_subspace", "umap"]
METRIC_ORDER = ["mahalanobis", "minkowski_l1", "minkowski_l2", "minkowski_inf"]


def fmt(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\pm$ {std:.3f}"


def make_table(df: pd.DataFrame, scenario: str) -> str:
    sub = df[df["scenario"] == scenario].copy()

    lines = []
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"& & \multicolumn{2}{c}{ViT} & \multicolumn{2}{c}{CNN} \\")
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    lines.append(r"Space & Distance & AUROC & FPR95 & AUROC & FPR95 \\")
    lines.append(r"\midrule")

    # find column-wise best AUROC for bolding
    best_vit = sub[sub["architecture"] == "vit"]["auroc"].max()
    best_cnn = sub[sub["architecture"] == "cnn"]["auroc"].max()

    for space in SPACE_ORDER:
        space_rows = sub[sub["space"] == space]
        first = True
        for metric in METRIC_ORDER:
            row_vit = space_rows[
                (space_rows["architecture"] == "vit")
                & (space_rows["detector"] == metric)
            ]
            row_cnn = space_rows[
                (space_rows["architecture"] == "cnn")
                & (space_rows["detector"] == metric)
            ]
            if row_vit.empty or row_cnn.empty:
                continue

            v = row_vit.iloc[0]
            c = row_cnn.iloc[0]

            vit_auroc = fmt(v["auroc"], v["auroc_std"])
            vit_fpr = fmt(v["fpr95"], v["fpr95_std"])
            cnn_auroc = fmt(c["auroc"], c["auroc_std"])
            cnn_fpr = fmt(c["fpr95"], c["fpr95_std"])

            if v["auroc"] >= best_vit - 1e-4:
                vit_auroc = r"\textbf{" + vit_auroc + "}"
            if c["auroc"] >= best_cnn - 1e-4:
                cnn_auroc = r"\textbf{" + cnn_auroc + "}"

            space_label = SPACE_LABELS[space] if first else ""
            first = False
            metric_label = METRIC_LABELS[metric]
            lines.append(
                f"{space_label} & {metric_label} & {vit_auroc} & {vit_fpr} & {cnn_auroc} & {cnn_fpr} \\\\"
            )

        lines.append(r"\midrule" if space != SPACE_ORDER[-1] else "")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def make_bar_chart(df: pd.DataFrame, scenario: str, save_path: Path) -> None:
    sub = df[(df["scenario"] == scenario) & (df["detector"] == "mahalanobis")].copy()

    spaces = SPACE_ORDER
    x = np.arange(len(spaces))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, arch in enumerate(["vit", "cnn"]):
        arch_sub = sub[sub["architecture"] == arch].set_index("space")
        means = [arch_sub.loc[s, "auroc"] if s in arch_sub.index else 0 for s in spaces]
        stds = [
            arch_sub.loc[s, "auroc_std"] if s in arch_sub.index else 0 for s in spaces
        ]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=arch.upper(),
            capsize=4,
            color=["#4878CF", "#D65F5F"][i],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SPACE_LABELS[s] for s in spaces])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title(f"Mahalanobis-DOD, {scenario.replace('_', '-').upper()}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


def main() -> None:
    df = pd.read_csv(RESULTS_CSV)

    for scenario in ["far_ood", "near_ood"]:
        label = scenario.replace("_", "-")
        table = make_table(df, scenario)
        path = OUT_DIR / f"table_{scenario}.tex"
        path.write_text(table)
        print(f"Saved {path}")
        print(f"\n--- {label} ---")
        print(table)

        make_bar_chart(df, scenario, OUT_DIR / f"bar_{scenario}.png")


if __name__ == "__main__":
    main()
