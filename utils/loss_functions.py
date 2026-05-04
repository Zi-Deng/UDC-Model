"""Loss functions for cost-sensitive image classification.

Provides a dispatch-based :class:`LossFunctions` class used by
:class:`train.CustomTrainer` to select a loss at runtime via a string name.

Dispatch table (``loss_function()`` method)::

    "cross_entropy"                   → cross_entropy()
    "seesaw"                          → seesaw_loss()
    "cost_matrix_cross_entropy"       → CELossLTV1()
    "logit_adjustment" / "test"       → CELogitAdjustmentV2()
    "logit_adjustment_regularized"    → CELogitAdjustmentRegularized()
    "nicme_v3_logit_adjustment"       → CELogitAdjustmentV3Pairwise()
    "nicme_v3_hybrid"                 → CELogitAdjustmentV3Regularized()
    "cost_weighted_ce"                → CostWeightedCE()
    "balanced_softmax"                → BalancedSoftmaxCE()
    "class_balanced_focal"            → ClassBalancedFocal()
    "ldam_drw"                        → LDAMDRW()
    "ap_csada"                        → APCSADA()
    "sosr_cnn"                        → SOSRCNN()

Additional older methods are kept as reference implementations.
"""

from typing import Literal

import torch
import torch.nn.functional as F

from utils.utils import get_device


class LossFunctions:
    """Configurable loss function collection with optional cost-matrix support.

    Args:
        epsilon: Small constant to avoid numerical instability (default 1e-9).
        cost_matrix: Optional 2-D list/array of shape ``(num_labels, num_labels)``
            defining per-class misclassification costs.  ``None`` means the
            cost-matrix-aware losses will fall back to uniform (ones) weighting.
    """

    def __init__(
        self,
        epsilon=1e-9,
        cost_matrix=None,
        cs_lambda=0.0,
        cs_warmup_epochs=0,
        class_priors=None,
        class_counts=None,
        logit_adjustment_tau=1.0,
        nicme_logit_cost_scale=1.0,
        cb_beta=0.9999,
        focal_gamma=2.0,
        ldam_max_m=0.5,
        ldam_scale=30.0,
        ldam_drw_start_epoch=16,
        ap_alpha=5.0,
    ):
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
        if class_priors is not None:
            priors = torch.tensor(class_priors, dtype=torch.float32, device=self.device)
            self.class_priors = priors / priors.sum().clamp_min(self.epsilon)
        else:
            self.class_priors = None
        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32, device=self.device)
            self.class_counts = counts.clamp_min(1.0)
        else:
            self.class_counts = None
        self.logit_adjustment_tau = logit_adjustment_tau
        self.nicme_logit_cost_scale = nicme_logit_cost_scale
        self.cb_beta = cb_beta
        self.focal_gamma = focal_gamma
        self.ldam_max_m = ldam_max_m
        self.ldam_scale = ldam_scale
        self.ldam_drw_start_epoch = ldam_drw_start_epoch
        self.ap_alpha = ap_alpha

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
        return F.cross_entropy(logits, targets)

    def menon_logit_adjusted(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Class-prior logit adjustment from Menon et al. for long-tail baselines.

        This is not a cost-matrix loss. It is included as an imbalance-coupled
        baseline to distinguish class-prior correction from NICME's explicit
        user-defined cost matrix.
        """
        if self.class_priors is None:
            priors = torch.ones(logits.shape[-1], dtype=logits.dtype, device=logits.device) / logits.shape[-1]
        else:
            priors = self.class_priors.to(device=logits.device, dtype=logits.dtype)
        adjusted_logits = logits + self.logit_adjustment_tau * torch.log(priors.clamp_min(self.epsilon))
        return F.cross_entropy(adjusted_logits, targets)

    def _class_counts(self, num_classes: int, device, dtype) -> torch.Tensor:
        if self.class_counts is not None:
            counts = self.class_counts.to(device=device, dtype=dtype)
            if counts.numel() == num_classes:
                return counts.clamp_min(1.0)
        if self.class_priors is not None:
            priors = self.class_priors.to(device=device, dtype=dtype)
            if priors.numel() == num_classes:
                return (priors / priors.min().clamp_min(self.epsilon)).clamp_min(1.0)
        return torch.ones(num_classes, dtype=dtype, device=device)

    def _effective_number_weights(self, num_classes: int, device, dtype) -> torch.Tensor:
        counts = self._class_counts(num_classes, device, dtype)
        beta = float(self.cb_beta)
        if beta <= 0.0:
            weights = torch.ones_like(counts)
        elif beta >= 1.0:
            weights = 1.0 / counts.clamp_min(1.0)
        else:
            beta_t = torch.tensor(beta, dtype=dtype, device=device)
            weights = (1.0 - beta_t) / (1.0 - torch.pow(beta_t, counts).clamp_min(self.epsilon))
        return weights * (num_classes / weights.sum().clamp_min(self.epsilon))

    def _cost_row_weights(self, num_classes: int, device, dtype) -> torch.Tensor:
        if self.cost_matrix is None:
            return torch.ones(num_classes, dtype=dtype, device=device)
        costs = self.cost_matrix.to(device=device, dtype=dtype)
        if costs.shape != (num_classes, num_classes):
            raise ValueError(f"cost_matrix shape {tuple(costs.shape)} does not match logits classes={num_classes}")
        off_diag_mask = ~torch.eye(num_classes, dtype=torch.bool, device=device)
        row_means = (costs * off_diag_mask).sum(dim=1) / max(num_classes - 1, 1)
        return row_means / row_means.mean().clamp_min(self.epsilon)

    def CostWeightedCE(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Classic cost-weighted empirical risk using true-class row mean costs."""
        weights = self._cost_row_weights(logits.shape[-1], logits.device, logits.dtype)
        ce = F.cross_entropy(logits, targets, reduction="none")
        return (ce * weights[targets]).mean()

    def BalancedSoftmaxCE(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Balanced Softmax baseline from Ren et al.; uniform counts reduce to CE."""
        counts = self._class_counts(logits.shape[-1], logits.device, logits.dtype)
        adjusted_logits = logits + torch.log(counts.clamp_min(self.epsilon))
        return F.cross_entropy(adjusted_logits, targets)

    def ClassBalancedFocal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Effective-number class-balanced focal loss from Cui et al."""
        weights = self._effective_number_weights(logits.shape[-1], logits.device, logits.dtype)
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce).clamp(min=self.epsilon, max=1.0)
        focal = torch.pow(1.0 - pt, float(self.focal_gamma))
        return (weights[targets] * focal * ce).mean()

    def LDAMDRW(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """LDAM with deferred reweighting using effective-number weights after DRW starts."""
        counts = self._class_counts(logits.shape[-1], logits.device, logits.dtype)
        margins = 1.0 / torch.sqrt(torch.sqrt(counts))
        margins = margins * (float(self.ldam_max_m) / margins.max().clamp_min(self.epsilon))
        adjusted_logits = logits.clone()
        adjusted_logits[torch.arange(logits.shape[0], device=logits.device), targets] -= margins[targets]
        weights = None
        if float(self.current_epoch) >= float(self.ldam_drw_start_epoch):
            weights = self._effective_number_weights(logits.shape[-1], logits.device, logits.dtype)
        return F.cross_entropy(float(self.ldam_scale) * adjusted_logits, targets, weight=weights)

    def APCSADA(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Adjusted-penalty CSADA-style baseline: CE plus normalized expected cost."""
        ce_loss = F.cross_entropy(logits, targets)
        if self.cost_matrix is None or float(self.ap_alpha) == 0.0:
            return ce_loss
        return ce_loss + float(self.ap_alpha) * self._normalized_cs_penalty(logits, targets)

    def SOSRCNN(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ConvNeXt-compatible CSDNN-family smooth cost-vector regression loss.

        The model outputs are interpreted as estimated costs. Evaluation uses
        argmin over these estimates rather than argmax over class logits.
        """
        if self.cost_matrix is None:
            target_costs = F.one_hot(targets, num_classes=logits.shape[-1]).to(dtype=logits.dtype, device=logits.device)
            target_costs = 1.0 - target_costs
        else:
            costs = self.cost_matrix.to(device=logits.device, dtype=logits.dtype)
            target_costs = costs[targets, :]
        return F.smooth_l1_loss(logits, target_costs)

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
        cost_values = cost_matrix[targets, predicted_classes].view(-1, 1)

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
            cost_values = self.cost_matrix[targets, pred_classes] * self.nicme_logit_cost_scale
        else:
            cost_values = torch.ones_like(targets, dtype=torch.float32)

        if mode == "positive":
            corrected = target_logits + cost_values * diff
        elif mode == "negative":
            corrected = target_logits - cost_values * diff
        else:
            corrected = target_logits

        modified_logits[range(batch_size), targets] = torch.where(misclassified, corrected, target_logits)

        return F.cross_entropy(modified_logits, targets)

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
            cost_values = self.cost_matrix[targets, pred_classes] * self.nicme_logit_cost_scale
        else:
            cost_values = torch.ones_like(targets, dtype=torch.float32)

        corrected_max_logits = max_logits + cost_values * diff

        modified_logits[range(batch_size), pred_classes] = torch.where(misclassified, corrected_max_logits, max_logits)

        return F.cross_entropy(modified_logits, targets)

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
            cost_values = self.cost_matrix[targets, pred_classes] * self.nicme_logit_cost_scale
        else:
            cost_values = torch.ones_like(targets, dtype=torch.float32)

        corrected_max_logits = max_logits + cost_values * diff

        modified_logits[range(batch_size), pred_classes] = torch.where(misclassified, corrected_max_logits, max_logits)

        ce_loss = F.cross_entropy(modified_logits, targets)

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

    def CSRegularizedCE(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Galdran-style CE + normalized expected-cost regularization only."""
        ce_loss = F.cross_entropy(logits, targets)
        if self.cs_lambda > 0 and self.cost_matrix is not None:
            if self.cs_warmup_epochs > 0 and self.current_epoch < self.cs_warmup_epochs:
                effective_lambda = self.cs_lambda * (self.current_epoch / self.cs_warmup_epochs)
            else:
                effective_lambda = self.cs_lambda
            m_max = self.cost_matrix.max()
            m_norm = self.cost_matrix / (m_max + self.epsilon) if m_max > 0 else self.cost_matrix
            probs_orig = torch.softmax(logits, dim=-1)
            cs_penalty = (m_norm[targets, :] * probs_orig).sum(dim=-1).mean()
            return ce_loss + effective_lambda * cs_penalty
        return ce_loss

    def _normalized_cs_penalty(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        m_max = self.cost_matrix.max()
        m_norm = self.cost_matrix / (m_max + self.epsilon) if m_max > 0 else self.cost_matrix
        probs_orig = torch.softmax(logits, dim=-1)
        return (m_norm[targets, :] * probs_orig).sum(dim=-1).mean()

    def _effective_cs_lambda(self) -> float:
        if self.cs_warmup_epochs > 0 and self.current_epoch < self.cs_warmup_epochs:
            return float(self.cs_lambda) * (float(self.current_epoch) / float(self.cs_warmup_epochs))
        return float(self.cs_lambda)

    def _v3_pairwise_adjusted_logits(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Apply unconditional positive pairwise cost margins to non-target logits."""
        if self.cost_matrix is None or self.nicme_logit_cost_scale == 0:
            return logits
        costs = self.cost_matrix.to(device=logits.device, dtype=logits.dtype)
        pair_costs = costs[targets, :]
        margin_offsets = torch.log(torch.clamp(pair_costs, min=1.0))
        target_mask = F.one_hot(targets, num_classes=logits.shape[-1]).to(dtype=torch.bool, device=logits.device)
        margin_offsets = margin_offsets.masked_fill(target_mask, 0.0)
        return logits + float(self.nicme_logit_cost_scale) * margin_offsets

    def CELogitAdjustmentV3Pairwise(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Unconditional pairwise cost-margin CE.

        For every example, non-target class ``k`` receives a positive additive
        offset of ``alpha * log(max(M[y, k], 1))`` before CE. High-cost wrong
        classes therefore need a larger raw-logit margin to become harmless.
        """
        return F.cross_entropy(self._v3_pairwise_adjusted_logits(logits, targets), targets)

    def CELogitAdjustmentV3Regularized(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """V3 pairwise cost-margin CE plus normalized expected-cost regularization."""
        ce_loss = self.CELogitAdjustmentV3Pairwise(logits, targets)
        if self.cs_lambda > 0 and self.cost_matrix is not None:
            return ce_loss + self._effective_cs_lambda() * self._normalized_cs_penalty(logits, targets)
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
        cost_values = self.cost_matrix[targets, predicted_classes].view(-1, 1)

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
            "nicme_v3_logit_adjustment"       → CELogitAdjustmentV3Pairwise()
            "nicme_v3_hybrid"                 → CELogitAdjustmentV3Regularized()
            "cost_weighted_ce"                → CostWeightedCE()
            "balanced_softmax"                → BalancedSoftmaxCE()
            "class_balanced_focal"            → ClassBalancedFocal()
            "ldam_drw"                        → LDAMDRW()
            "ap_csada"                        → APCSADA()
            "sosr_cnn"                        → SOSRCNN()

        Raises:
            ValueError: If *loss_name* is not recognised.
        """
        if loss_name in ("cross_entropy", "ce", "ce_calibrated_cost_min"):
            return self.cross_entropy
        elif loss_name in ("menon_logit_adjusted", "prior_logit_adjusted"):
            return self.menon_logit_adjusted
        elif loss_name in ("cs_regularized_ce", "nuls_cs"):
            return self.CSRegularizedCE
        elif loss_name == "seesaw":
            return self.seesaw_loss
        elif loss_name == "cost_matrix_cross_entropy":
            return self.CELossLTV1
        elif loss_name in ("logit_adjustment", "test", "nicme_logit_adjustment"):
            return self.CELogitAdjustmentV2
        elif loss_name in ("logit_adjustment_regularized", "nicme_hybrid"):
            return self.CELogitAdjustmentRegularized
        elif loss_name in ("nicme_v3_logit_adjustment", "logit_adjustment_v3_pairwise"):
            return self.CELogitAdjustmentV3Pairwise
        elif loss_name in ("nicme_v3_hybrid", "logit_adjustment_v3_regularized"):
            return self.CELogitAdjustmentV3Regularized
        elif loss_name in ("cost_weighted_ce", "row_mean_cost_weighted_ce"):
            return self.CostWeightedCE
        elif loss_name == "balanced_softmax":
            return self.BalancedSoftmaxCE
        elif loss_name in ("class_balanced_focal", "cb_focal"):
            return self.ClassBalancedFocal
        elif loss_name == "ldam_drw":
            return self.LDAMDRW
        elif loss_name in ("ap_csada", "adjusted_penalty_csada"):
            return self.APCSADA
        elif loss_name in ("sosr_cnn", "csdnn_sosr"):
            return self.SOSRCNN
        elif loss_name == "csada":
            return self.cross_entropy
        else:
            raise ValueError(f"Invalid loss function: {loss_name}")
