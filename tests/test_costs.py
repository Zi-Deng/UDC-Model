import math

import numpy as np

from nicme.costs import (
    CostMatrix,
    average_test_cost,
    classification_metrics,
    cost_min_predictions,
    cost_weighted_confusion_matrix,
    critical_pair_errors,
    normalized_atc,
    quadratic_weighted_kappa_np,
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


def test_multiclass_target_recall_aggregate_uses_worst_cared_class():
    cost_matrix = np.ones((3, 3)).tolist()
    for idx in range(3):
        cost_matrix[idx][idx] = 0.0

    metrics = classification_metrics(
        np.array([0, 1, 1, 2, 2]),
        np.array([0, 1, 0, 2, 1]),
        None,
        cost_matrix,
        target_class_id=1,
        target_class_ids=[1, 2],
        target_recall_floor=0.0,
    )

    assert metrics["target_recall_by_class"] == {1: 0.5, 2: 0.5}
    assert metrics["target_recall"] == 0.5
    assert metrics["target_recall_macro"] == 0.5


def test_cost_weighted_confusion_matrix_multiplies_counts_by_true_pred_cost():
    cost_matrix = [[0.0, 2.0], [5.0, 0.0]]
    weighted = cost_weighted_confusion_matrix(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 0]),
        cost_matrix,
    )

    assert weighted.tolist() == [[0.0, 2.0], [10.0, 0.0]]


def test_quadratic_weighted_kappa_and_critical_pair_summary():
    y_true = np.array([0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 2, 1, 1, 2, 2])
    cost_matrix = [[0.0, 1.0, 4.0], [1.0, 0.0, 10.0], [1.0, 8.0, 0.0]]

    assert quadratic_weighted_kappa_np(y_true, y_true, num_classes=3) == 1.0

    report = critical_pair_errors(
        y_true,
        y_pred,
        [(1, 2, 10.0), (2, 1, 8.0)],
        cost_matrix,
        class_names=["a", "b", "c"],
    )

    assert report["b->c"]["errors"] == 1
    assert report["b->c"]["atc_contribution"] == 10.0 / 6.0
    assert report["c->b"]["errors"] == 1
