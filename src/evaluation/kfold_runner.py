import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.evaluation.metrics import compute_all
from src.ood.base import BaseDetector


def run_kfold(
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
    projectors: dict[str, object],
    detectors: dict[str, BaseDetector],
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    id_feats, id_labels = embeddings["id_train"]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    ood_splits = {name: embeddings[name][0] for name in ("near_ood", "far_ood")}

    rows = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(id_feats)):
        train_feats = id_feats[train_idx]
        val_feats = id_feats[val_idx]

        for proj_name, projector in projectors.items():
            if projector is None:
                train_proj = train_feats
                val_proj = val_feats
                ood_proj = ood_splits
            else:
                projector.fit(train_feats)
                train_proj = projector.transform(train_feats)
                val_proj = projector.transform(val_feats)
                ood_proj = {
                    name: projector.transform(f) for name, f in ood_splits.items()
                }

            for det_name, detector in detectors.items():
                detector.fit(train_proj)
                id_scores = detector.score(val_proj)
                id_preds = detector.predict(val_proj)

                for scenario, ood_feats in ood_proj.items():
                    ood_scores = detector.score(ood_feats)
                    ood_preds = detector.predict(ood_feats)
                    rows.append(
                        {
                            "fold": fold,
                            "space": proj_name,
                            "detector": det_name,
                            "scenario": scenario,
                            **compute_all(id_scores, ood_scores, id_preds, ood_preds),
                        }
                    )

    return pd.DataFrame(rows)


def aggregate_folds(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["space", "detector", "scenario"]
    metric_cols = ["auroc", "fpr95", "aupr", "bal_acc"]
    mean = df.groupby(group_cols)[metric_cols].mean()
    std = df.groupby(group_cols)[metric_cols].std()
    std.columns = [f"{c}_std" for c in std.columns]
    return pd.concat([mean, std], axis=1).reset_index()
