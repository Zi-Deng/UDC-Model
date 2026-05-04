import numpy as np
import pytest

from nicme.calibration import (
    binary_threshold_predictions,
    calibrated_cost_min_predictions,
    fit_binary_threshold_np,
    fit_temperature_np,
)


def test_temperature_grid_search_does_not_worsen_nll():
    logits = np.array(
        [
            [4.0, -1.0],
            [2.0, 0.0],
            [-1.0, 3.0],
            [0.0, 2.0],
        ]
    )
    labels = np.array([0, 1, 1, 0])

    result = fit_temperature_np(logits, labels)

    assert result.temperature > 0
    assert result.nll_after <= result.nll_before + 1e-12


def test_calibrated_cost_min_matches_cost_decision_rule():
    logits = np.array([[2.0, 0.0]])
    cost_matrix = [[0.0, 1.0], [10.0, 0.0]]

    assert calibrated_cost_min_predictions(logits, cost_matrix, temperature=10.0).tolist() == [1]


def test_binary_threshold_selection_can_prefer_accuracy_constrained_threshold():
    probabilities = np.array(
        [
            [0.92, 0.08],
            [0.61, 0.39],
            [0.72, 0.28],
            [0.20, 0.80],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    cost_matrix = [[0.0, 10.0], [1.0, 0.0]]

    result = fit_binary_threshold_np(
        probabilities,
        labels,
        cost_matrix,
        target_class_id=0,
        target_recall_floor=1.0,
        accuracy_floor=0.75,
    )
    predictions = binary_threshold_predictions(probabilities, result.threshold, target_class_id=0)

    assert predictions.tolist() == [0, 0, 0, 1]
    assert result.target_recall == 1.0
    assert result.accuracy == 0.75


def test_binary_threshold_helpers_reject_multiclass_probabilities():
    probabilities = np.array([[0.2, 0.3, 0.5]])

    with pytest.raises(ValueError):
        binary_threshold_predictions(probabilities, threshold=0.5, target_class_id=2)

    with pytest.raises(ValueError):
        fit_binary_threshold_np(
            probabilities,
            np.array([2]),
            [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
            target_class_id=2,
            target_recall_floor=1.0,
            accuracy_floor=0.0,
        )
