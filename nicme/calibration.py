"""Temperature scaling and calibrated cost-sensitive decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nicme.costs import cost_min_predictions, nll_np, softmax_np


@dataclass(frozen=True)
class TemperatureScalingResult:
    temperature: float
    nll_before: float
    nll_after: float


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
