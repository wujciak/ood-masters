"""Run Wilcoxon (ViT vs CNN) and Friedman + Nemenyi post-hoc tests on eval results."""

from itertools import product

import numpy as np
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon

from _constants import (
    ARCHS,
    DETECTORS,
    DETECTOR_LABELS,
    METRICS,
    SCENARIOS,
    SCENARIO_LABELS,
    SPACES,
    SPACE_LABELS,
    latest_npz,
)

ALPHA = 0.05


def load_all_scores(data: dict) -> dict:
    return {
        (arch, space, det, scen, metric): float(data[key].mean())
        for arch, space, det, scen, metric in product(
            ARCHS, SPACES, DETECTORS, SCENARIOS, METRICS
        )
        if (key := f"{arch}__{space}__{det}__{scen}__{metric}") in data
    }


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


def wilcoxon_vit_vs_cnn(scores: dict, metric: str) -> None:
    section("1. ViT vs CNN - Wilcoxon signed-rank test")
    print(
        f"  Datasets per scenario: space x detector = {len(SPACES)}x{len(DETECTORS)} = {len(SPACES)*len(DETECTORS)}"
    )

    for scen in SCENARIOS:
        vit_vals = np.array(
            [
                scores.get(("vit", sp, det, scen, metric), np.nan)
                for sp, det in product(SPACES, DETECTORS)
            ]
        )
        cnn_vals = np.array(
            [
                scores.get(("cnn", sp, det, scen, metric), np.nan)
                for sp, det in product(SPACES, DETECTORS)
            ]
        )
        mask = ~(np.isnan(vit_vals) | np.isnan(cnn_vals))
        vit_vals, cnn_vals = vit_vals[mask], cnn_vals[mask]

        stat, p = wilcoxon(vit_vals, cnn_vals, alternative="two-sided")
        winner = "ViT" if vit_vals.mean() > cnn_vals.mean() else "CNN"
        sig = "*significant*" if p < ALPHA else "not significant"
        print(f"\n  {SCENARIO_LABELS[scen]}:")
        print(
            f"    ViT mean={vit_vals.mean():.4f}  CNN mean={cnn_vals.mean():.4f}  -> {winner} higher"
        )
        print(f"    Wilcoxon W={stat:.1f}  p={p:.4f}  {sig}")


def _run_friedman(
    groups: dict,
    group_keys: list[str],
    labels: dict[str, str],
) -> None:
    matrix = np.array([groups[g] for g in group_keys])
    if np.isnan(matrix).any():
        return

    stat, p = friedmanchisquare(*matrix)
    for g in group_keys:
        print(f"    {labels[g]:20s} mean={np.mean(groups[g]):.4f}")
    sig = "*significant*" if p < ALPHA else "not significant"
    print(f"    Friedman chi2={stat:.3f}  p={p:.4f}  {sig}")

    if p < ALPHA:
        nemenyi = sp.posthoc_nemenyi_friedman(matrix.T)
        nemenyi.index = nemenyi.columns = [labels[g] for g in group_keys]
        print("    Nemenyi p-values:")
        print("    " + nemenyi.round(3).to_string().replace("\n", "\n    "))


def friedman_spaces(scores: dict, metric: str) -> None:
    section("2. Feature spaces - Friedman + Nemenyi post-hoc")
    print(
        f"  Datasets per scenario: arch x detector = {len(ARCHS)}x{len(DETECTORS)} = {len(ARCHS)*len(DETECTORS)}"
    )

    for scen in SCENARIOS:
        subsection(SCENARIO_LABELS[scen])
        groups = {
            sp: [
                scores.get((arch, sp, det, scen, metric), np.nan)
                for arch, det in product(ARCHS, DETECTORS)
            ]
            for sp in SPACES
        }
        _run_friedman(groups, SPACES, SPACE_LABELS)


def friedman_detectors(scores: dict, metric: str) -> None:
    section("3. Distance metrics - Friedman + Nemenyi post-hoc")
    print(
        f"  Datasets per scenario: arch x space = {len(ARCHS)}x{len(SPACES)} = {len(ARCHS)*len(SPACES)}"
    )

    for scen in SCENARIOS:
        subsection(SCENARIO_LABELS[scen])
        groups = {
            det: [
                scores.get((arch, sp, det, scen, metric), np.nan)
                for arch, sp in product(ARCHS, SPACES)
            ]
            for det in DETECTORS
        }
        _run_friedman(groups, DETECTORS, DETECTOR_LABELS)


def main() -> None:
    npz_path = latest_npz()
    print(f"Loading: {npz_path}  |  alpha={ALPHA}")
    scores = load_all_scores(np.load(npz_path))

    for metric in METRICS:
        print(f"\n{'#' * 60}")
        print(f"#  METRIC: {metric.upper()}")
        print(f"{'#' * 60}")
        wilcoxon_vit_vs_cnn(scores, metric)
        friedman_spaces(scores, metric)
        friedman_detectors(scores, metric)


if __name__ == "__main__":
    main()
