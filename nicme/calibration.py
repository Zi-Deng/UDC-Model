"""Temperature scaling and calibrated cost-sensitive decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nicme.costs import classification_metrics, cost_min_predictions, nll_np, softmax_np


@dataclass(frozen=True)
class TemperatureScalingResult:
    temperature: float
    nll_before: float
    nll_after: float


@dataclass(frozen=True)
class ThresholdSelectionResult:
    threshold: float
    selection_score: float
    target_recall: float
    accuracy: float
    normalized_atc: float


def fit_temperature_np(logits: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None) -> TemperatureScalingResult:
    """Fit scalar temperature by validation/calibration NLL grid search.

    This pure NumPy fallback is deterministic and dependency-light.  Training
    code may use a torch optimizer later, but this is sufficient for reproducible
    calibration and unit tests.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if grid is None:
        grid = np.concatenate(
            [
                np.linspace(0.5, 3.0, 51),
                np.linspace(3.25, 10.0, 28),
            ]
        )
    nll_before = nll_np(softmax_np(logits, 1.0), labels)
    best_t = 1.0
    best_nll = nll_before
    for temp in grid:
        nll = nll_np(softmax_np(logits, float(temp)), labels)
        if nll < best_nll:
            best_nll = nll
            best_t = float(temp)
    return TemperatureScalingResult(temperature=best_t, nll_before=nll_before, nll_after=float(best_nll))


def calibrated_cost_min_predictions(
    logits: np.ndarray,
    cost_matrix,
    temperature: float = 1.0,
) -> np.ndarray:
    probabilities = softmax_np(logits, temperature)
    return cost_min_predictions(probabilities, cost_matrix)


def binary_threshold_predictions(
    probabilities: np.ndarray,
    threshold: float,
    target_class_id: int,
) -> np.ndarray:
    """Predict the target class when its probability exceeds ``threshold``.

    This is a binary-only threshold-moving decision rule. It is intentionally
    separate from the multiclass minimum-expected-cost rule so experiments can
    distinguish Bayes cost-min decisions from accuracy-constrained calibration.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("binary_threshold_predictions requires probabilities with shape (n_samples, 2)")
    target = int(target_class_id)
    if target not in {0, 1}:
        raise ValueError("target_class_id must be 0 or 1 for binary thresholding")
    other = 1 - target
    return np.where(probs[:, target] >= float(threshold), target, other)


def fit_binary_threshold_np(
    probabilities: np.ndarray,
    labels: np.ndarray,
    cost_matrix,
    target_class_id: int,
    target_recall_floor: float,
    accuracy_floor: float,
    selection_accuracy_metric: str = "accuracy",
) -> ThresholdSelectionResult:
    """Choose a binary target-class threshold on calibration data.

    The objective is the same lower-is-better NICME selection score used for
    checkpointing. Ties prefer higher target recall, then higher selected
    accuracy, then lower normalized ATC.
    """
    probs = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(labels, dtype=int)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("fit_binary_threshold_np requires probabilities with shape (n_samples, 2)")
    if y_true.shape[0] != probs.shape[0]:
        raise ValueError("labels and probabilities must contain the same number of samples")

    target = int(target_class_id)
    target_probs = probs[:, target]
    unique = np.unique(target_probs)
    candidates = [0.0, 1.0]
    candidates.extend(float(value) for value in unique)
    if unique.size > 1:
        candidates.extend(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:], strict=False))
    candidates = sorted(set(min(max(threshold, 0.0), 1.0) for threshold in candidates))

    best_result: ThresholdSelectionResult | None = None
    best_key: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        y_pred = binary_threshold_predictions(probs, threshold, target)
        report = classification_metrics(
            y_true,
            y_pred,
            probs,
            cost_matrix,
            target_class_id=target,
            target_recall_floor=target_recall_floor,
            accuracy_floor=accuracy_floor,
            selection_accuracy_metric=selection_accuracy_metric,
        )
        selected_accuracy = float(report["selection_accuracy_value"])
        result = ThresholdSelectionResult(
            threshold=float(threshold),
            selection_score=float(report["selection_score"]),
            target_recall=float(report["target_recall"]),
            accuracy=selected_accuracy,
            normalized_atc=float(report["normalized_atc"]),
        )
        key = (
            result.selection_score,
            -result.target_recall,
            -result.accuracy,
            result.normalized_atc,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_result = result

    if best_result is None:
        raise ValueError("No threshold candidates were generated")
    return best_result
