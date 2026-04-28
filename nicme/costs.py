"""Cost-matrix convention, decisions, and metrics for NICME.

NICME uses one convention everywhere:

    C[true_label][predicted_label]

is the cost incurred when an example with ``true_label`` is predicted as
``predicted_label``.  Rows are ground truth, columns are predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _cost_matrix_size(cost_matrix: Any) -> int | None:
    if cost_matrix is None:
        return None
    arr = np.asarray(cost_matrix, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return int(arr.shape[0])
    return None


def _infer_num_classes(y_true: np.ndarray, y_pred: np.ndarray | None = None, cost_matrix: Any = None) -> int:
    candidates: list[int] = []
    cm_size = _cost_matrix_size(cost_matrix)
    if cm_size is not None:
        candidates.append(cm_size)
    labels = [np.asarray(y_true, dtype=int)]
    if y_pred is not None:
        labels.append(np.asarray(y_pred, dtype=int))
    for values in labels:
        if values.size:
            candidates.append(int(np.max(values)) + 1)
    return max(candidates) if candidates else 0


@dataclass(frozen=True)
class CostMatrix:
    """Validated cost matrix using rows=true labels and columns=predicted labels."""

    values: np.ndarray
    class_names: tuple[str, ...] | None = None

    @classmethod
    def from_config(
        cls,
        cost_matrix: Any,
        num_classes: int | None = None,
        class_names: list[str] | tuple[str, ...] | None = None,
        require_zero_diagonal: bool = False,
    ) -> CostMatrix:
        if cost_matrix is None:
            if num_classes is None:
                raise ValueError("num_classes is required when cost_matrix is None")
            arr = np.ones((num_classes, num_classes), dtype=float)
            np.fill_diagonal(arr, 0.0)
        else:
            arr = np.asarray(cost_matrix, dtype=float)

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"cost_matrix must be square, got shape {arr.shape}")

        if num_classes is not None and arr.shape != (num_classes, num_classes):
            raise ValueError(f"cost_matrix must have shape {(num_classes, num_classes)}, got {arr.shape}")

        if np.any(arr < 0):
            raise ValueError("cost_matrix entries must be non-negative")

        if require_zero_diagonal and not np.allclose(np.diag(arr), 0.0):
            raise ValueError("cost_matrix diagonal must be zero")

        names = tuple(class_names) if class_names is not None else None
        if names is not None and len(names) != arr.shape[0]:
            raise ValueError("class_names length must match cost_matrix size")

        return cls(values=arr, class_names=names)

    @property
    def num_classes(self) -> int:
        return int(self.values.shape[0])

    @property
    def max_cost(self) -> float:
        return float(np.max(self.values)) if self.values.size else 0.0

    @property
    def max_off_diagonal_cost(self) -> float:
        arr = self.values.copy()
        np.fill_diagonal(arr, 0.0)
        return float(np.max(arr)) if arr.size else 0.0

    def normalized(self) -> np.ndarray:
        denom = self.max_cost
        return self.values / denom if denom > 0 else self.values.copy()


def resolve_class_id(
    target_class: str | int | None,
    class_names: list[str] | tuple[str, ...] | None,
    default: int = 0,
) -> int:
    """Resolve a target recall class by id or class name."""
    if target_class is None:
        return int(default)
    if isinstance(target_class, int):
        return int(target_class)
    text = str(target_class).strip()
    if text == "":
        return int(default)
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    if class_names is None:
        raise ValueError(f"target class {target_class!r} requires class_names")
    if text not in class_names:
        raise ValueError(f"target class {target_class!r} not found in class_names={class_names}")
    return list(class_names).index(text)


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temp = max(float(temperature), 1e-12)
    scaled = np.asarray(logits, dtype=float) / temp
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cost_min_predictions(probabilities: np.ndarray, cost_matrix: Any) -> np.ndarray:
    """Return minimum expected cost decisions for calibrated probabilities.

    For each candidate prediction k, expected cost is
    sum_j C[j][k] * P(Y=j | x).
    """
    probs = np.asarray(probabilities, dtype=float)
    cm = CostMatrix.from_config(cost_matrix, num_classes=probs.shape[1]).values
    expected_cost = probs @ cm
    return np.argmin(expected_cost, axis=1)


def average_test_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: Any) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = CostMatrix.from_config(cost_matrix, num_classes=_infer_num_classes(y_true, y_pred, cost_matrix)).values
    return float(np.mean(cm[y_true, y_pred])) if y_true.size else 0.0


def normalized_atc(y_true: np.ndarray, y_pred: np.ndarray, cost_matrix: Any) -> float:
    spec = CostMatrix.from_config(cost_matrix, num_classes=_infer_num_classes(y_true, y_pred, cost_matrix))
    denom = spec.max_off_diagonal_cost
    atc = average_test_cost(y_true, y_pred, spec.values)
    return atc / denom if denom > 0 else atc


def cost_reduction_ratio(baseline_atc: float | None, proposed_atc: float) -> float | None:
    if baseline_atc is None or baseline_atc <= 0:
        return None
    return float((baseline_atc - proposed_atc) / baseline_atc)


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int), strict=False):
        cm[truth, pred] += 1
    return cm


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    probs = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = predictions == y_true
    ece = 0.0
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = np.mean(correct[mask])
        bin_conf = np.mean(confidences[mask])
        ece += (np.sum(mask) / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def nll_np(probabilities: np.ndarray, y_true: np.ndarray, epsilon: float = 1e-12) -> float:
    probs = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    picked = probs[np.arange(labels.size), labels]
    return float(-np.mean(np.log(np.clip(picked, epsilon, 1.0)))) if labels.size else 0.0


def brier_score_np(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    labels = np.asarray(y_true, dtype=int)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(labels.size), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))) if labels.size else 0.0


def selection_score(
    normalized_cost: float,
    target_recall: float,
    accuracy_value: float,
    target_recall_floor: float = 0.95,
    accuracy_floor: float = 0.80,
    recall_weight: float = 4.0,
    accuracy_weight: float = 1.0,
) -> float:
    recall_gap = max(0.0, float(target_recall_floor) - float(target_recall))
    accuracy_gap = max(0.0, float(accuracy_floor) - float(accuracy_value))
    return float(normalized_cost + recall_weight * recall_gap**2 + accuracy_weight * accuracy_gap**2)


def class_prevalence(y_true: np.ndarray, num_classes: int) -> dict[int, float]:
    labels = np.asarray(y_true, dtype=int)
    total = max(int(labels.size), 1)
    return {idx: float(np.sum(labels == idx) / total) for idx in range(num_classes)}


def _binary_curve_metrics(probabilities: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if probabilities.shape[1] == 2 and len(np.unique(y_true)) == 2:
            return {
                "auroc": float(roc_auc_score(y_true, probabilities[:, 1])),
                "auprc": float(average_precision_score(y_true, probabilities[:, 1])),
            }
    except Exception:
        pass
    return {"auroc": 0.0, "auprc": 0.0}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    cost_matrix: Any,
    target_class_id: int = 0,
    target_recall_floor: float = 0.95,
    accuracy_floor: float = 0.80,
    selection_accuracy_metric: str = "accuracy",
    baseline_atc: float | None = None,
) -> dict[str, Any]:
    """Compute NICME paper-facing metrics for one inference mode."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    num_classes = _infer_num_classes(y_true, y_pred, cost_matrix)
    spec = CostMatrix.from_config(cost_matrix, num_classes=num_classes)
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    total = max(int(np.sum(cm)), 1)
    accuracy = float(np.trace(cm) / total)

    recalls = {}
    precisions = {}
    f1s = {}
    fnrs = {}
    fprs = {}
    for idx in range(num_classes):
        tp = cm[idx, idx]
        fn = int(np.sum(cm[idx, :]) - tp)
        fp = int(np.sum(cm[:, idx]) - tp)
        tn = int(total - tp - fn - fp)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recalls[idx] = float(recall)
        precisions[idx] = float(precision)
        f1s[idx] = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        fnrs[idx] = float(fn / (tp + fn)) if (tp + fn) else 0.0
        fprs[idx] = float(fp / (fp + tn)) if (fp + tn) else 0.0

    balanced_acc = float(np.mean(list(recalls.values()))) if recalls else 0.0
    macro_f1 = float(np.mean(list(f1s.values()))) if f1s else 0.0
    atc = average_test_cost(y_true, y_pred, spec.values)
    natc = atc / spec.max_off_diagonal_cost if spec.max_off_diagonal_cost > 0 else atc
    target_recall = recalls.get(int(target_class_id), 0.0)
    accuracy_metric = str(selection_accuracy_metric or "accuracy").strip().lower()
    if accuracy_metric in {"balanced_accuracy", "balanced_acc", "bacc"}:
        selected_accuracy = balanced_acc
    elif accuracy_metric == "accuracy":
        selected_accuracy = accuracy
    else:
        raise ValueError("selection_accuracy_metric must be 'accuracy' or 'balanced_accuracy'")
    score = selection_score(natc, target_recall, selected_accuracy, target_recall_floor, accuracy_floor)

    if probabilities is None:
        probabilities = np.eye(num_classes, dtype=float)[y_pred]

    metrics: dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "atc": atc,
        "normalized_atc": float(natc),
        "crr": cost_reduction_ratio(baseline_atc, atc),
        "selection_score": score,
        "selection_accuracy_metric": accuracy_metric,
        "selection_accuracy_value": float(selected_accuracy),
        "target_class_id": int(target_class_id),
        "target_recall": float(target_recall),
        "target_precision": precisions.get(int(target_class_id), 0.0),
        "target_fnr": fnrs.get(int(target_class_id), 0.0),
        "target_fpr": fprs.get(int(target_class_id), 0.0),
        "nll": nll_np(probabilities, y_true),
        "brier": brier_score_np(probabilities, y_true),
        "ece": expected_calibration_error(probabilities, y_true),
        "class_prevalence": class_prevalence(y_true, num_classes),
        "confusion_matrix": cm.tolist(),
        "per_class_recall": recalls,
        "per_class_precision": precisions,
        "per_class_f1": f1s,
    }
    metrics.update(_binary_curve_metrics(probabilities, y_true))
    for key, value in list(metrics.items()):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            metrics[key] = 0.0
    return metrics
