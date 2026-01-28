"""Loss functions for cost-sensitive image classification.

Provides a dispatch-based :class:`LossFunctions` class used by
:class:`train.CustomTrainer` to select a loss at runtime via a string name.

Dispatch table (``loss_function()`` method)::

    "cross_entropy"                   → cross_entropy()
    "seesaw"                          → seesaw_loss()
    "cost_matrix_cross_entropy"       → CELossLTV1()
    "logit_adjustment" / "test"       → CELogitAdjustmentV2()
    "logit_adjustment_regularized"    → CELogitAdjustmentRegularized()

Additional methods (CELossLT_LossMult, CELogitAdjustment, CELogitAdjustmentV3)
are kept as reference implementations but are **not** in the dispatch table.
"""

from typing import Literal

import torch

from utils.utils import get_device


class LossFunctions:
    """Configurable loss function collection with optional cost-matrix support.

    Args:
        epsilon: Small constant to avoid numerical instability (default 1e-9).
        cost_matrix: Optional 2-D list/array of shape ``(num_labels, num_labels)``
            defining per-class misclassification costs.  ``None`` means the
            cost-matrix-aware losses will fall back to uniform (ones) weighting.
    """

    def __init__(self, epsilon=1e-9, cost_matrix=None, cs_lambda=0.0, cs_warmup_epochs=0):
        self.epsilon = epsilon
        self.device = get_device()
        print(f"Using device: {self.device}")

        if cost_matrix is not None:
            self.cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32, device=self.device)
        else:
            self.cost_matrix = None

        self.cs_lambda = cs_lambda
        self.cs_warmup_epochs = cs_warmup_epochs
        self.current_epoch = 0.0

    def calculate_dynamic_alpha(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Dynamically calculates alpha based on the statistical properties of the logits.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).

        Returns:
            torch.Tensor: The dynamically calculated alpha value.
        """
        # Calculate the absolute mean and standard deviation of the logits
        mean_logit = torch.mean(torch.abs(logits))
        std_logit = torch.std(logits)

        # Calculate alpha using the mathematically defined formula
        alpha = mean_logit / (std_logit + self.epsilon)
        return alpha

    def cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss from scratch.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (batch_size).

        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=-1)

        # Select the predicted probabilities corresponding to the target class
        batch_size = logits.shape[0]
        target_probs = probs[range(batch_size), targets]

        # Take the log of the probabilities
        log_probs = -torch.log(target_probs + 1e-9)  # Add small value to avoid log(0)

        # Compute the mean loss
        loss = log_probs.mean()
        return loss

    # NOTE: Not in dispatch — kept as reference implementation.
    def CELossLT_LossMult(self, output_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cost-matrix CE with dynamic alpha scaling and loss multiplication.

        Weights logits by ``alpha * cost_values`` before softmax, then
        multiplies the final loss by the mean cost value.

        Args:
            output_logits: Model logits of shape ``(batch_size, num_classes)``.
            targets: Ground-truth labels of shape ``(batch_size,)``.

        Returns:
            Scalar loss tensor.
        """
        alpha = self.calculate_dynamic_alpha(output_logits)
        batch_size = output_logits.shape[0]
        cost_matrix = self.cost_matrix

        _, predicted_classes = torch.max(output_logits, dim=1)
        cost_values = cost_matrix[predicted_classes, targets].view(-1, 1)

        weighted_logits = output_logits * (alpha * cost_values)
        weighted_probs = torch.softmax(weighted_logits, dim=-1)
        target_probs = weighted_probs[range(batch_size), targets]

        log_probs = -torch.log(target_probs + 1e-9)
        loss = log_probs.mean() * cost_values.mean()
        return loss

    # NOTE: Not in dispatch — kept as reference implementation.
    def CELogitAdjustment(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mode: Literal["standard", "positive", "negative"] = "negative",
    ) -> torch.Tensor:
        """CE with per-sample logit adjustment on the *target* class.

        For misclassified examples, adjusts the target-class logit by
        ``+/- cost_value * |max_logit - target_logit|`` before computing CE.

        Args:
            logits: Model logits ``(batch_size, num_classes)``.
            targets: Ground-truth labels ``(batch_size,)``.
            mode: ``"positive"`` boosts target logit, ``"negative"`` penalises
                it, ``"standard"`` applies no correction.

        Returns:
            Scalar loss tensor.
        """
        modified_logits = logits.clone()
        batch_size, num_classes = logits.shape

        pred_classes = torch.argmax(modified_logits, dim=1)
        max_logits = modified_logits[range(batch_size), pred_classes]
        target_logits = modified_logits[range(batch_size), targets]
        misclassified = pred_classes != targets
        diff = torch.abs(max_logits - target_logits)

        if self.cost_matrix is not None:
            cost_values = self.cost_matrix[targets, pred_classes]
        else:
            cost_values = torch.ones_like(targets, dtype=torch.float32)

        if mode == "positive":
            corrected = target_logits + cost_values * diff
        elif mode == "negative":
            corrected = target_logits - cost_values * diff
        else:
            corrected = target_logits

        modified_logits[range(batch_size), targets] = torch.where(misclassified, corrected, target_logits)

        probs = torch.softmax(modified_logits, dim=-1)
        target_probs = probs[range(batch_size), targets]
        log_probs = -torch.log(target_probs + 1e-9)
        return log_probs.mean()

    def CELogitAdjustmentV2(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """CE with logit adjustment on the *predicted* (max) class.

        For misclassified examples, *increases* the max-class logit by
        ``cost_value * |max_logit - target_logit|``, making the model more
        confident in the wrong prediction and thereby amplifying the loss
        gradient for costly misclassifications.

        This is the dispatched method for ``"logit_adjustment"`` / ``"test"``.

        Args:
            logits: Model logits ``(batch_size, num_classes)``.
            targets: Ground-truth labels ``(batch_size,)``.

        Returns:
            Scalar loss tensor.
        """
        modified_logits = logits.clone()
        batch_size, num_classes = logits.shape

        pred_classes = torch.argmax(modified_logits, dim=1)
        max_logits = modified_logits[range(batch_size), pred_classes]
        target_logits = modified_logits[range(batch_size), targets]
        misclassified = pred_classes != targets
        diff = torch.abs(max_logits - target_logits)

        if self.cost_matrix is not None:
            cost_values = self.cost_matrix[targets, pred_classes]
        else:
            cost_values = torch.ones_like(targets, dtype=torch.float32)

        corrected_max_logits = max_logits + cost_values * diff

        modified_logits[range(batch_size), pred_classes] = torch.where(misclassified, corrected_max_logits, max_logits)

        probs = torch.softmax(modified_logits, dim=-1)
        target_probs = probs[range(batch_size), targets]
        log_probs = -torch.log(target_probs + 1e-9)
        return log_probs.mean()

    def CELogitAdjustmentRegularized(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """CE with logit adjustment on the predicted class + CS regularization.

        Combines two cost-sensitive mechanisms:

        1. **Logit adjustment** (same as V2): for misclassified samples,
           increases the max-class logit by ``cost * |max_logit - target_logit|``,
           amplifying the CE loss gradient for costly misclassifications.
        2. **Cost-sensitive regularization**: adds
           ``lambda_eff * mean(M_norm[target, :] · softmax(logits))`` using a
           normalized cost matrix and optional warmup schedule.

        The regularization uses the **original** (non-adjusted) logits so that
        the two gradient paths are independent.  The cost matrix is normalized
        to [0, 1] for the regularization term to prevent unbounded penalties.

        This is the dispatched method for ``"logit_adjustment_regularized"``.

        Args:
            logits: Model logits ``(batch_size, num_classes)``.
            targets: Ground-truth labels ``(batch_size,)``.

        Returns:
            Scalar loss tensor.
        """
        # --- Part 1: Logit-adjusted CE (identical to CELogitAdjustmentV2) ---
        modified_logits = logits.clone()
        batch_size, num_classes = logits.shape

        pred_classes = torch.argmax(modified_logits, dim=1)
        max_logits = modified_logits[range(batch_size), pred_classes]
        target_logits = modified_logits[range(batch_size), targets]
        misclassified = pred_classes != targets
        diff = torch.abs(max_logits - target_logits)

        if self.cost_matrix is not None:
            cost_values = self.cost_matrix[targets, pred_classes]
        else:
            cost_values = torch.ones_like(targets, dtype=torch.float32)

        corrected_max_logits = max_logits + cost_values * diff

        modified_logits[range(batch_size), pred_classes] = torch.where(misclassified, corrected_max_logits, max_logits)

        probs_adj = torch.softmax(modified_logits, dim=-1)
        target_probs = probs_adj[range(batch_size), targets]
        ce_loss = (-torch.log(target_probs + self.epsilon)).mean()

        # --- Part 2: CS regularization with normalization + warmup ---
        if self.cs_lambda > 0 and self.cost_matrix is not None:
            # Warmup: ramp from 0 to cs_lambda over cs_warmup_epochs
            if self.cs_warmup_epochs > 0 and self.current_epoch < self.cs_warmup_epochs:
                effective_lambda = self.cs_lambda * (self.current_epoch / self.cs_warmup_epochs)
            else:
                effective_lambda = self.cs_lambda

            # Normalize cost matrix to [0, 1] for bounded regularization
            m_max = self.cost_matrix.max()
            m_norm = self.cost_matrix / (m_max + self.epsilon) if m_max > 0 else self.cost_matrix

            # CS penalty on ORIGINAL logits (independent gradient path)
            probs_orig = torch.softmax(logits, dim=-1)
            cs_penalty = (m_norm[targets, :] * probs_orig).sum(dim=-1).mean()

            return ce_loss + effective_lambda * cs_penalty

        return ce_loss

    # NOTE: Not in dispatch — kept as reference implementation.
    def CELogitAdjustmentV3(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """CE with logit adjustment on *all non-max* classes.

        For misclassified examples, subtracts ``cost_value * diff`` from every
        logit except the max, effectively widening the gap between the incorrect
        prediction and all other classes.

        Args:
            logits: Model logits ``(batch_size, num_classes)``.
            targets: Ground-truth labels ``(batch_size,)``.

        Returns:
            Scalar loss tensor.
        """
        # Clone logits to avoid in-place ops
        modified_logits = logits.clone()

        batch_size, num_classes = logits.shape

        # Get predicted class (argmax)
        pred_classes = torch.argmax(modified_logits, dim=1)

        # Gather predicted logits and target logits
        max_logits = modified_logits[range(batch_size), pred_classes]
        target_logits = modified_logits[range(batch_size), targets]

        # Identify misclassified examples
        misclassified = pred_classes != targets

        # Compute difference where misclassified (between max logit and target logit)
        diff = torch.abs(max_logits - target_logits)

        # Get cost values from cost matrix based on true label (row) and predicted label (column)
        if self.cost_matrix is not None:
            cost_values = self.cost_matrix[targets, pred_classes]  # Shape: (batch_size,)
        else:
            # Fallback to fixed values if cost matrix is not available
            cost_values = torch.ones_like(targets, dtype=torch.float32)

        # Create adjustment values for misclassified examples
        adjustment = cost_values * diff  # Shape: (batch_size,)

        # Apply adjustment to all logits except the maximum logit for misclassified examples
        # First, subtract adjustment from ALL logits for misclassified examples
        misclassified_mask = misclassified.unsqueeze(1)  # Shape: (batch_size, 1)
        adjustment_expanded = adjustment.unsqueeze(1)  # Shape: (batch_size, 1)

        modified_logits = torch.where(misclassified_mask, modified_logits - adjustment_expanded, modified_logits)

        # Then add back the adjustment to the maximum logit for misclassified examples
        # to restore it to its original value (effectively leaving only non-max logits adjusted)
        modified_logits[range(batch_size), pred_classes] = torch.where(
            misclassified,
            modified_logits[range(batch_size), pred_classes] + adjustment,
            modified_logits[range(batch_size), pred_classes],
        )

        # Compute softmax and cross-entropy
        probs = torch.softmax(modified_logits, dim=-1)
        target_probs = probs[range(batch_size), targets]
        log_probs = -torch.log(target_probs + 1e-9)
        return log_probs.mean()

    def CELossLTV1(self, output_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cost-matrix CE with dynamic alpha scaling (dispatched as ``"cost_matrix_cross_entropy"``).

        Weights logits by ``alpha * cost_values`` before softmax, where
        ``alpha`` is derived from logit statistics.

        Args:
            output_logits: Model logits ``(batch_size, num_classes)``.
            targets: Ground-truth labels ``(batch_size,)``.

        Returns:
            Scalar loss tensor.
        """
        alpha = self.calculate_dynamic_alpha(output_logits)
        batch_size = output_logits.shape[0]

        _, predicted_classes = torch.max(output_logits, dim=1)
        cost_values = self.cost_matrix[predicted_classes, targets].view(-1, 1)

        weighted_logits = output_logits * (alpha * cost_values)
        weighted_probs = torch.softmax(weighted_logits, dim=-1)
        target_probs = weighted_probs[range(batch_size), targets]

        log_probs = -torch.log(target_probs + 1e-9)
        return log_probs.mean()

    def seesaw_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 0.8,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Implements the Seesaw Loss function.

        Args:
            logits (torch.Tensor): Logits predicted by the model (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels (batch_size).
            alpha (float): Scaling factor for the positive sample term.
            beta (float): Scaling factor for the negative sample term.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

        Returns:
            torch.Tensor: Computed scalar loss value.
        """
        # Error checking for debugging
        if logits is None:
            raise ValueError("seesaw_loss: logits is None")
        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"seesaw_loss: logits must be a torch.Tensor, got {type(logits)}")
        if logits.dim() < 2:
            raise ValueError(f"seesaw_loss: logits must have at least 2 dimensions, got shape {logits.shape}")

        batch_size = logits.size(0)

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Create one-hot encoded target tensor
        target_one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

        # Positive term
        pos_probs = probs * target_one_hot
        pos_loss = -alpha * torch.log(pos_probs + 1e-9) * target_one_hot

        # Negative term
        neg_probs = probs * (1 - target_one_hot)
        neg_factor = torch.pow(1 - neg_probs, beta)
        neg_loss = -neg_factor * torch.log(1 - probs + 1e-9) * (1 - target_one_hot)

        # Total loss
        loss = pos_loss + neg_loss
        if reduction == "mean":
            return loss.sum() / batch_size
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    def loss_function(self, loss_name):
        """Return the loss callable for the given *loss_name* string.

        Supported names::

            "cross_entropy"                   → cross_entropy()
            "seesaw"                          → seesaw_loss()
            "cost_matrix_cross_entropy"       → CELossLTV1()
            "logit_adjustment" / "test"       → CELogitAdjustmentV2()
            "logit_adjustment_regularized"    → CELogitAdjustmentRegularized()

        Raises:
            ValueError: If *loss_name* is not recognised.
        """
        if loss_name == "cross_entropy":
            return self.cross_entropy
        elif loss_name == "seesaw":
            return self.seesaw_loss
        elif loss_name == "cost_matrix_cross_entropy":
            return self.CELossLTV1
        elif loss_name in ("logit_adjustment", "test"):
            return self.CELogitAdjustmentV2
        elif loss_name == "logit_adjustment_regularized":
            return self.CELogitAdjustmentRegularized
        else:
            raise ValueError(f"Invalid loss function: {loss_name}")
