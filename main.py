"""Entry point. Usage: python -m main <experiment>

Available experiments:
  extract       extract and cache embeddings
  eval          main evaluation (all spaces x metrics)
  n_components  sweep n_components for PCA and Random Subspace
  k_clusters    sweep number of K-Means clusters
  threshold     sweep threshold percentile q
"""

import argparse
import importlib

EXPERIMENTS = {
    "extract": "extract",
    "eval": "experiments.eval",
    "n_components": "experiments.n_components",
    "k_clusters": "experiments.k_clusters",
    "threshold": "experiments.threshold",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=EXPERIMENTS.keys())
    args = parser.parse_args()

    module = importlib.import_module(EXPERIMENTS[args.experiment])
    module.main()


if __name__ == "__main__":
    main()
