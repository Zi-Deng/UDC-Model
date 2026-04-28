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
