import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


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
    return float((id_scores >= np.percentile(ood_scores, 100.0 * (1.0 - tpr))).mean())


def aupr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    return float(average_precision_score(*_stack(id_scores, ood_scores)))


def balanced_acc(id_preds: np.ndarray, ood_preds: np.ndarray) -> float:
    """Balanced accuracy for binary OOD predictions (0=ID, 1=OOD)."""
    y_true = np.concatenate([np.zeros(len(id_preds)), np.ones(len(ood_preds))])
    y_pred = np.concatenate([id_preds, ood_preds])
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_all(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    id_preds: np.ndarray,
    ood_preds: np.ndarray,
) -> dict[str, float]:
    return {
        "auroc": auroc(id_scores, ood_scores),
        "fpr95": fpr_at_tpr(id_scores, ood_scores),
        "aupr": aupr(id_scores, ood_scores),
        "bal_acc": balanced_acc(id_preds, ood_preds),
    }
