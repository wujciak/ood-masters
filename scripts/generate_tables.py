"""Generate LaTeX result tables and bar charts from the latest eval CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _constants import (
    ARCH_COLORS,
    DETECTORS,
    DETECTOR_LABELS,
    METRICS,
    METRIC_LABELS,
    SPACES,
    SPACE_LABELS,
)

TABLES_DIR = Path("data/results/tables")
PLOTS_DIR = Path("data/plots")
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def fmt(mean: float, std: float, decimals: int = 3) -> str:
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def _build_rows(
    sub: pd.DataFrame,
    col: str,
    col_std: str,
    fmt_fn,
    lower_is_better: bool = False,
) -> list[str]:
    lines = []
    for space in SPACES:
        space_rows = sub[sub["space"] == space]
        first = True
        for det in DETECTORS:
            v = space_rows[
                (space_rows["architecture"] == "vit") & (space_rows["detector"] == det)
            ]
            c = space_rows[
                (space_rows["architecture"] == "cnn") & (space_rows["detector"] == det)
            ]
            if v.empty or c.empty:
                continue
            v, c = v.iloc[0], c.iloc[0]
            vit_val = fmt_fn(v[col], v[col_std])
            cnn_val = fmt_fn(c[col], c[col_std])
            vit_better = v[col] < c[col] if lower_is_better else v[col] > c[col]
            cnn_better = c[col] < v[col] if lower_is_better else c[col] > v[col]
            if vit_better:
                vit_val = bold(vit_val)
            elif cnn_better:
                cnn_val = bold(cnn_val)
            space_label = SPACE_LABELS[space] if first else ""
            first = False
            lines.append(
                f"{space_label} & {DETECTOR_LABELS[det]} & {vit_val} & {cnn_val} \\\\"
            )
        lines.append(r"\midrule" if space != SPACES[-1] else "")
    return lines


def make_table(df: pd.DataFrame, scenario: str, metric: str) -> str:
    sub = df[df["scenario"] == scenario]
    rows = _build_rows(sub, metric, f"{metric}_std", fmt)
    return "\n".join(
        [
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"Space & Distance & ViT & CNN \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def make_timing_table(df: pd.DataFrame) -> str:
    sub = df[df["scenario"] == "far_ood"]
    rows = _build_rows(
        sub,
        "score_time_us",
        "score_time_us_std",
        lambda m, s: fmt(m, s, decimals=1),
        lower_is_better=True,
    )
    return "\n".join(
        [
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"Space & Distance & ViT ($\mu$s) & CNN ($\mu$s) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def make_bar_chart(
    df: pd.DataFrame, scenario: str, metric: str, save_path: Path
) -> None:
    sub = df[(df["scenario"] == scenario) & (df["detector"] == "mahalanobis")]
    x = np.arange(len(SPACES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, arch in enumerate(["vit", "cnn"]):
        arch_sub = sub[sub["architecture"] == arch].set_index("space")
        means = [arch_sub.loc[s, metric] if s in arch_sub.index else 0 for s in SPACES]
        stds = [
            arch_sub.loc[s, f"{metric}_std"] if s in arch_sub.index else 0
            for s in SPACES
        ]
        ax.bar(
            x + (i - 0.5) * width,
            means,
            width,
            yerr=stds,
            label=arch.upper(),
            capsize=4,
            color=ARCH_COLORS[arch],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SPACE_LABELS[s] for s in SPACES])
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

    timing = make_timing_table(df)
    path = TABLES_DIR / "table_timing.tex"
    path.write_text(timing)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
