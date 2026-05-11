import numpy as np
from sklearn.metrics import roc_auc_score


def _stack(
    id_scores: np.ndarray, ood_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))]),
        np.concatenate([id_scores, ood_scores]),
    )


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    return float(roc_auc_score(*_stack(id_scores, ood_scores)))


def fpr_at_tpr(
    id_scores: np.ndarray, ood_scores: np.ndarray, tpr: float = 0.95
) -> float:
    threshold = np.percentile(ood_scores, 100.0 * (1.0 - tpr))
    return float((id_scores >= threshold).mean())


def compute_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": auroc(id_scores, ood_scores),
        "fpr95": fpr_at_tpr(id_scores, ood_scores),
    }
