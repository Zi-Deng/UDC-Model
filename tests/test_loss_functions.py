import pytest

torch = pytest.importorskip("torch")

from utils.loss_functions import LossFunctions  # noqa: E402


def test_nicme_logit_adjustment_uses_true_by_predicted_cost_convention():
    high_false_negative = LossFunctions(cost_matrix=[[0.0, 10.0], [1.0, 0.0]])
    low_false_negative = LossFunctions(cost_matrix=[[0.0, 1.0], [10.0, 0.0]])
    logits = torch.tensor([[0.0, 1.0]], device=high_false_negative.device)
    targets = torch.tensor([0], device=high_false_negative.device)

    high_loss = high_false_negative.CELogitAdjustmentV2(logits, targets)
    low_loss = low_false_negative.CELogitAdjustmentV2(logits, targets)

    assert high_loss > low_loss


def test_nicme_logit_cost_scale_controls_logit_adjustment_strength():
    full_scale = LossFunctions(cost_matrix=[[0.0, 10.0], [1.0, 0.0]], nicme_logit_cost_scale=1.0)
    reduced_scale = LossFunctions(cost_matrix=[[0.0, 10.0], [1.0, 0.0]], nicme_logit_cost_scale=0.1)
    logits = torch.tensor([[0.0, 1.0]], device=full_scale.device)
    targets = torch.tensor([0], device=full_scale.device)

    assert full_scale.CELogitAdjustmentV2(logits, targets) > reduced_scale.CELogitAdjustmentV2(logits, targets)


def test_nicme_hybrid_cost_scale_controls_logit_adjustment_strength():
    full_scale = LossFunctions(
        cost_matrix=[[0.0, 10.0], [1.0, 0.0]],
        cs_lambda=0.0,
        nicme_logit_cost_scale=1.0,
    )
    reduced_scale = LossFunctions(
        cost_matrix=[[0.0, 10.0], [1.0, 0.0]],
        cs_lambda=0.0,
        nicme_logit_cost_scale=0.1,
    )
    logits = torch.tensor([[0.0, 1.0]], device=full_scale.device)
    targets = torch.tensor([0], device=full_scale.device)

    assert full_scale.CELogitAdjustmentRegularized(logits, targets) > reduced_scale.CELogitAdjustmentRegularized(
        logits,
        targets,
    )


def test_nicme_default_cost_one_matches_ce():
    loss_functions = LossFunctions(cost_matrix=[[0.0, 1.0], [1.0, 0.0]], nicme_logit_cost_scale=1.0)
    logits = torch.tensor([[3.0, 1.0], [0.5, 2.0]], device=loss_functions.device)
    targets = torch.tensor([0, 1], device=loss_functions.device)

    assert torch.allclose(
        loss_functions.CELogitAdjustmentV3Pairwise(logits, targets),
        loss_functions.cross_entropy(logits, targets),
    )


def test_nicme_scale_zero_and_lambda_zero_matches_ce():
    loss_functions = LossFunctions(
        cost_matrix=[[0.0, 10.0], [8.0, 0.0]],
        nicme_logit_cost_scale=0.0,
        cs_lambda=0.0,
    )
    logits = torch.tensor([[3.0, 1.0], [0.5, 2.0]], device=loss_functions.device)
    targets = torch.tensor([0, 1], device=loss_functions.device)

    assert torch.allclose(
        loss_functions.CELogitAdjustmentV3Regularized(logits, targets),
        loss_functions.cross_entropy(logits, targets),
    )


def test_nicme_applies_to_correctly_classified_samples():
    high_cost = LossFunctions(cost_matrix=[[0.0, 10.0], [1.0, 0.0]], nicme_logit_cost_scale=1.0)
    logits = torch.tensor([[3.0, 1.0]], device=high_cost.device)
    targets = torch.tensor([0], device=high_cost.device)

    assert torch.argmax(logits, dim=1).item() == 0
    assert high_cost.CELogitAdjustmentV3Pairwise(logits, targets) > high_cost.cross_entropy(logits, targets)


def test_nicme_larger_cost_increases_non_target_gradient_pressure():
    high_cost = LossFunctions(cost_matrix=[[0.0, 10.0], [1.0, 0.0]], nicme_logit_cost_scale=1.0)
    low_cost = LossFunctions(cost_matrix=[[0.0, 1.0], [10.0, 0.0]], nicme_logit_cost_scale=1.0)
    targets = torch.tensor([0], device=high_cost.device)

    high_logits = torch.tensor([[3.0, 1.0]], device=high_cost.device, requires_grad=True)
    low_logits = torch.tensor([[3.0, 1.0]], device=low_cost.device, requires_grad=True)
    high_loss = high_cost.CELogitAdjustmentV3Pairwise(high_logits, targets)
    low_loss = low_cost.CELogitAdjustmentV3Pairwise(low_logits, targets)
    high_loss.backward()
    low_loss.backward()

    assert high_loss > low_loss
    assert high_logits.grad[0, 1] > low_logits.grad[0, 1]


def test_nicme_hybrid_adds_cs_regularizer_on_original_logits():
    loss_functions = LossFunctions(
        cost_matrix=[[0.0, 10.0], [8.0, 0.0]],
        nicme_logit_cost_scale=0.5,
        cs_lambda=0.25,
        cs_warmup_epochs=0,
    )
    logits = torch.tensor([[3.0, 1.0], [0.5, 2.0]], device=loss_functions.device)
    targets = torch.tensor([0, 1], device=loss_functions.device)

    pairwise = loss_functions.CELogitAdjustmentV3Pairwise(logits, targets)
    expected = pairwise + 0.25 * loss_functions._normalized_cs_penalty(logits, targets)

    assert torch.allclose(loss_functions.CELogitAdjustmentV3Regularized(logits, targets), expected)


def test_loss_dispatcher_uses_versionless_nicme_names():
    loss_functions = LossFunctions(cost_matrix=[[0.0, 10.0], [8.0, 0.0]])

    assert loss_functions.loss_function("nicme_logit_adjustment") == loss_functions.CELogitAdjustmentV3Pairwise
    assert loss_functions.loss_function("nicme_hybrid") == loss_functions.CELogitAdjustmentV3Regularized


def test_loss_dispatcher_accepts_deprecated_nicme_v3_aliases():
    loss_functions = LossFunctions(cost_matrix=[[0.0, 10.0], [8.0, 0.0]])

    assert loss_functions.loss_function("nicme_v3_logit_adjustment") == loss_functions.CELogitAdjustmentV3Pairwise
    assert loss_functions.loss_function("nicme_v3_hybrid") == loss_functions.CELogitAdjustmentV3Regularized


def test_loss_dispatcher_accepts_legacy_nicme_names_for_old_formulation():
    loss_functions = LossFunctions(cost_matrix=[[0.0, 10.0], [8.0, 0.0]])

    assert loss_functions.loss_function("legacy_nicme_logit_adjustment") == loss_functions.CELogitAdjustmentV2
    assert loss_functions.loss_function("legacy_nicme_hybrid") == loss_functions.CELogitAdjustmentRegularized


def test_sota_loss_dispatcher_accepts_baseline_names():
    loss_functions = LossFunctions(cost_matrix=[[0.0, 10.0], [8.0, 0.0]])

    assert loss_functions.loss_function("cost_weighted_ce") == loss_functions.CostWeightedCE
    assert loss_functions.loss_function("balanced_softmax") == loss_functions.BalancedSoftmaxCE
    assert loss_functions.loss_function("class_balanced_focal") == loss_functions.ClassBalancedFocal
    assert loss_functions.loss_function("ldam_drw") == loss_functions.LDAMDRW
    assert loss_functions.loss_function("ap_csada") == loss_functions.APCSADA
    assert loss_functions.loss_function("sosr_cnn") == loss_functions.SOSRCNN


def test_menon_and_balanced_softmax_reduce_to_ce_on_uniform_counts():
    loss_functions = LossFunctions(
        class_priors=[0.5, 0.5],
        class_counts=[10, 10],
        logit_adjustment_tau=1.0,
    )
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device=loss_functions.device)
    targets = torch.tensor([0, 1], device=loss_functions.device)
    ce = loss_functions.cross_entropy(logits, targets)

    assert torch.allclose(loss_functions.menon_logit_adjusted(logits, targets), ce)
    assert torch.allclose(loss_functions.BalancedSoftmaxCE(logits, targets), ce)


def test_cost_weighted_ce_uses_true_class_row_costs():
    high_row0 = LossFunctions(cost_matrix=[[0.0, 10.0], [1.0, 0.0]])
    high_row1 = LossFunctions(cost_matrix=[[0.0, 1.0], [10.0, 0.0]])
    logits = torch.tensor([[0.0, 1.0]], device=high_row0.device)
    targets = torch.tensor([0], device=high_row0.device)

    assert high_row0.CostWeightedCE(logits, targets) > high_row1.CostWeightedCE(logits, targets)


def test_ap_csada_loss_increases_when_probability_mass_moves_to_costly_wrong_class():
    loss_functions = LossFunctions(cost_matrix=[[0.0, 10.0], [1.0, 0.0]], ap_alpha=5.0)
    targets = torch.tensor([0], device=loss_functions.device)
    costly_logits = torch.tensor([[0.0, 2.0]], device=loss_functions.device)
    safe_logits = torch.tensor([[2.0, 0.0]], device=loss_functions.device)

    assert loss_functions.APCSADA(costly_logits, targets) > loss_functions.APCSADA(safe_logits, targets)


def test_sosr_trains_against_cost_vectors_and_predicts_by_argmin():
    loss_functions = LossFunctions(cost_matrix=[[0.0, 10.0], [1.0, 0.0]])
    targets = torch.tensor([0], device=loss_functions.device)
    good_estimated_costs = torch.tensor([[0.0, 10.0]], device=loss_functions.device)
    bad_estimated_costs = torch.tensor([[10.0, 0.0]], device=loss_functions.device)

    assert loss_functions.SOSRCNN(good_estimated_costs, targets) < loss_functions.SOSRCNN(bad_estimated_costs, targets)
    assert torch.argmin(good_estimated_costs, dim=1).item() == 0
