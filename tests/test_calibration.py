import numpy as np

from nicme.calibration import calibrated_cost_min_predictions, fit_temperature_np


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
