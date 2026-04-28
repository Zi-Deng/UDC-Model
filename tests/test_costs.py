import math

import numpy as np

from nicme.costs import (
    CostMatrix,
    average_test_cost,
    classification_metrics,
    cost_min_predictions,
    normalized_atc,
    resolve_class_id,
    selection_score,
)


def test_cost_matrix_convention_true_by_predicted():
    cost_matrix = [[0.0, 10.0], [1.0, 0.0]]
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])

    assert average_test_cost(y_true, y_pred, cost_matrix) == 5.5
    assert normalized_atc(y_true, y_pred, cost_matrix) == 0.55


def test_cost_min_binary_uses_expected_cost_not_argmax():
    cost_matrix = [[0.0, 1.0], [10.0, 0.0]]
    probabilities = np.array([[0.85, 0.15], [0.95, 0.05]])

    assert cost_min_predictions(probabilities, cost_matrix).tolist() == [1, 0]


def test_cost_min_multiclass_hand_computed():
    cost_matrix = [
        [0.0, 2.0, 5.0],
        [3.0, 0.0, 1.0],
        [9.0, 2.0, 0.0],
    ]
    probabilities = np.array([[0.20, 0.30, 0.50]])
    expected_costs = probabilities @ np.array(cost_matrix)

    assert expected_costs.argmin() == 2
    assert cost_min_predictions(probabilities, cost_matrix).tolist() == [2]


def test_metrics_keep_declared_class_count_when_split_misses_class():
    cost_matrix = [[0.0, 10.0], [1.0, 0.0]]
    metrics = classification_metrics(
        np.array([0, 0]),
        np.array([0, 1]),
        None,
        cost_matrix,
        target_class_id=0,
    )

    assert metrics["confusion_matrix"] == [[1, 1], [0, 0]]
    assert metrics["class_prevalence"] == {0: 1.0, 1: 0.0}


def test_selection_can_use_balanced_accuracy_gap():
    cost_matrix = [[0.0, 1.0], [10.0, 0.0]]
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])

    accuracy_metrics = classification_metrics(
        y_true,
        y_pred,
        None,
        cost_matrix,
        target_class_id=1,
        target_recall_floor=0.0,
        accuracy_floor=0.70,
        selection_accuracy_metric="accuracy",
    )
    balanced_metrics = classification_metrics(
        y_true,
        y_pred,
        None,
        cost_matrix,
        target_class_id=1,
        target_recall_floor=0.0,
        accuracy_floor=0.70,
        selection_accuracy_metric="balanced_accuracy",
    )

    assert accuracy_metrics["selection_accuracy_value"] == 0.75
    assert balanced_metrics["selection_accuracy_value"] == 0.5
    assert balanced_metrics["selection_score"] > accuracy_metrics["selection_score"]


def test_cost_matrix_validation_and_target_resolution():
    spec = CostMatrix.from_config([[0, 3], [2, 0]], num_classes=2, class_names=["a", "b"])

    assert spec.class_names == ("a", "b")
    assert resolve_class_id("b", ["a", "b"]) == 1
    assert math.isclose(selection_score(0.1, 0.9, 0.75, 0.95, 0.8), 0.1125)
