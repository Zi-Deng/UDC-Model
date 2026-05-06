"""Main training entry point for cost-sensitive image classification.

Usage:
    micromamba activate ml
    python scripts/train.py --config config/modelConfig.json

Loads a JSON config, instantiates a ResNet or ConvNeXt model with pretrained
weights, trains with a configurable loss function (including cost-matrix-aware
variants), and saves comprehensive evaluation metrics and confusion matrix
visualizations to results/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import torch
import torch.nn.functional as F
import wandb
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging

from nicme.modeling import build_model
from utils.image_processor import CustomImageProcessor
from utils.loss_functions import LossFunctions
from utils.utils import (
    collate_fn,
    compute_argmax_decision_report,
    config_flag_enabled,
    get_device,
    make_compute_metrics,
    parse_HF_args,
    perform_comprehensive_evaluation,
    prediction_mode_for_args,
    preprocess_hf_dataset,
    preprocess_kg_dataset,
    preprocess_local_csv_dataset,
    preprocess_local_folder_dataset,
    preprocess_manifest_dataset,
)


def _limit_dataset(dataset, max_samples, seed):
    if dataset is None or max_samples is None:
        return dataset
    limit = int(max_samples)
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    return dataset.shuffle(seed=int(seed)).select(range(limit))


def _resolve_checkpoint_file(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"initial_checkpoint_path does not exist: {path}")
    candidates = [path / "model.safetensors", path / "pytorch_model.bin"]
    checkpoint_dirs = sorted(path.glob("checkpoint-*"), key=lambda item: item.stat().st_mtime, reverse=True)
    for checkpoint_dir in checkpoint_dirs:
        candidates.extend([checkpoint_dir / "model.safetensors", checkpoint_dir / "pytorch_model.bin"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found under {path}")


def load_initial_checkpoint(model: torch.nn.Module, path_value: str | None) -> Path | None:
    checkpoint_file = _resolve_checkpoint_file(path_value)
    if checkpoint_file is None:
        return None
    if checkpoint_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_file), device="cpu")
    else:
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded initial checkpoint: {checkpoint_file}")
    if missing_keys:
        print(f"Missing keys while loading initial checkpoint: {missing_keys[:20]}")
    if unexpected_keys:
        print(f"Unexpected keys while loading initial checkpoint: {unexpected_keys[:20]}")
    return checkpoint_file


class CustomTrainer(Trainer):
    """HuggingFace Trainer subclass that injects a custom loss function.

    Instead of using the model's built-in loss, this trainer dispatches to
    a LossFunctions instance that supports cost-matrix-aware loss variants.

    Args:
        loss_fxn: String name of the loss function (e.g. "cross_entropy",
            "logit_adjustment"). Passed to LossFunctions.loss_function().
        cost_matrix: Optional 2D list defining per-class misclassification costs.
            Dimensions must match num_labels x num_labels.
        cs_lambda: Cost-sensitive regularization weight. Only used by
            "logit_adjustment_regularized" loss. Default 0.0 (disabled).
        cs_warmup_epochs: Number of epochs to linearly ramp cs_lambda from
            0 to its full value. Default 0 (no warmup).
    """

    def __init__(
        self,
        loss_fxn,
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
        cost_regularization_alpha=5.0,
        csada_lambda=2.0,
        csada_attack_steps=5,
        csada_epsilon=2.0 / 255.0,
        csada_step_size=1.0 / 255.0,
        csada_cost_temperature=3.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lossClass = LossFunctions(
            cost_matrix=cost_matrix,
            cs_lambda=cs_lambda,
            cs_warmup_epochs=cs_warmup_epochs,
            class_priors=class_priors,
            class_counts=class_counts,
            logit_adjustment_tau=logit_adjustment_tau,
            nicme_logit_cost_scale=nicme_logit_cost_scale,
            cb_beta=cb_beta,
            focal_gamma=focal_gamma,
            ldam_max_m=ldam_max_m,
            ldam_scale=ldam_scale,
            ldam_drw_start_epoch=ldam_drw_start_epoch,
            cost_regularization_alpha=cost_regularization_alpha,
        )
        self.loss_fxn = loss_fxn
        self.csada_lambda = float(csada_lambda)
        self.csada_attack_steps = int(csada_attack_steps)
        self.csada_epsilon = float(csada_epsilon)
        self.csada_step_size = float(csada_step_size)
        self.csada_cost_temperature = float(csada_cost_temperature)

    def _normalized_image_bounds(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor([0.485, 0.456, 0.406], device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
        lower = (0.0 - mean) / std
        upper = (1.0 - mean) / std
        return lower, upper

    def _critical_targets(self, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.lossClass.cost_matrix is None:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            return labels, torch.ones_like(labels, dtype=torch.float32), mask
        costs = self.lossClass.cost_matrix.to(device=labels.device)
        row_costs = costs[labels, :]
        selected_costs, target_labels = torch.max(row_costs, dim=1)
        mask = selected_costs > 1.0
        return target_labels, selected_costs, mask

    def _targeted_csada_examples(self, model, pixel_values: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_labels, selected_costs, mask = self._critical_targets(labels)
        if not torch.any(mask) or self.csada_attack_steps <= 0:
            return pixel_values.detach(), selected_costs, mask

        lower, upper = self._normalized_image_bounds(pixel_values)
        std = torch.tensor([0.229, 0.224, 0.225], device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
        epsilon = self.csada_epsilon / std
        step_size = self.csada_step_size / std
        delta = torch.zeros_like(pixel_values, requires_grad=True)

        for _ in range(self.csada_attack_steps):
            adv = torch.max(torch.min(pixel_values + delta, upper), lower)
            adv_logits = model(adv).logits.float().contiguous()
            targeted_loss = F.cross_entropy(adv_logits[mask], target_labels[mask])
            grad = torch.autograd.grad(targeted_loss, delta, only_inputs=True)[0]
            with torch.no_grad():
                delta -= step_size * grad.sign()
                delta.clamp_(min=-epsilon, max=epsilon)
                delta.copy_(torch.max(torch.min(pixel_values + delta, upper), lower) - pixel_values)
            delta.requires_grad_(True)
        adv = torch.max(torch.min(pixel_values + delta.detach(), upper), lower)
        return adv, selected_costs, mask

    def _compute_csada_loss(self, model, pixel_values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        clean_logits = model(pixel_values).logits.float().contiguous()
        clean_loss = F.cross_entropy(clean_logits, labels)
        adv_values, selected_costs, mask = self._targeted_csada_examples(model, pixel_values.detach(), labels)
        if not torch.any(mask):
            return clean_loss
        adv_logits = model(adv_values).logits.float().contiguous()
        adv_loss = F.cross_entropy(adv_logits[mask], labels[mask], reduction="none")
        weights = torch.pow(selected_costs[mask].clamp_min(1.0), 1.0 / max(self.csada_cost_temperature, 1e-6))
        weights = weights / weights.mean().clamp_min(1e-9)
        return clean_loss + self.csada_lambda * (weights * adv_loss).mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Forward pass through the model and compute loss via the configured loss function."""
        pixel_values = inputs["pixel_values"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device)
        self.lossClass.current_epoch = getattr(self.state, "epoch", None) or 0.0

        if self.loss_fxn == "csada":
            loss = self._compute_csada_loss(model, pixel_values, labels)
            return (loss, model(pixel_values)) if return_outputs else loss

        outputs = model(pixel_values)
        logits = outputs.logits

        if logits is None:
            raise ValueError("Model outputs.logits is None. Check model forward method.")
        if not isinstance(logits, torch.Tensor):
            raise ValueError(f"Model outputs.logits is not a tensor. Got type: {type(logits)}")
        logits = logits.float().contiguous()

        loss_fxn = self.lossClass.loss_function(self.loss_fxn)
        loss = loss_fxn(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction_step to properly handle dict-based image inputs during evaluation."""
        pixel_values = inputs["pixel_values"].to(self.args.device)
        labels = inputs["labels"].to(self.args.device) if "labels" in inputs else None
        self.lossClass.current_epoch = getattr(self.state, "epoch", None) or 0.0

        with torch.no_grad():
            outputs = model(pixel_values)

        logits = outputs.logits
        if isinstance(logits, torch.Tensor):
            logits = logits.float().contiguous()
        loss = None
        if labels is not None:
            loss_fxn = self.lossClass.loss_function(self.loss_fxn)
            loss = loss_fxn(logits, labels)
        return (loss, logits, labels)


def main(script_args):
    """Run the full training and evaluation pipeline.

    Args:
        script_args: A ScriptTrainingArguments instance containing all
            training configuration (model, dataset, hyperparameters, etc.).

    Returns:
        str: Path to the results directory containing metrics and visualizations.
    """
    set_seed(getattr(script_args, "seed", 42))

    if config_flag_enabled(getattr(script_args, "disable_cudnn", False)):
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled for this run")

    if config_flag_enabled(script_args.wandb):
        wandb.login()  # Uses WANDB_API_KEY environment variable or interactive prompt
        wandb.init(
            project="convnext",
            name=script_args.output_dir,
        )
        wandb.config.update(
            {
                "model_checkpoint": script_args.output_dir,
                "batch_size": script_args.batch_size,
                "learning_rate": script_args.learning_rate,
                "num_train_epochs": script_args.num_train_epochs,
            }
        )

    # Load dataset
    image_processor = CustomImageProcessor.from_pretrained(
        script_args.model,
        image_size=getattr(script_args, "image_size", None),
    )
    class_names = None
    calibration_ds = None

    if script_args.dataset_host == "huggingface":
        train_ds, val_ds, test_ds = preprocess_hf_dataset(script_args.dataset, script_args.model)
    elif script_args.dataset_host == "kaggle":
        train_ds, val_ds, test_ds = preprocess_kg_dataset(
            script_args.dataset,
            script_args.local_dataset_name,
            script_args.model,
        )
    elif script_args.dataset_host == "local_folder":
        if script_args.local_folder_path is None:
            raise ValueError("local_folder_path must be specified when dataset_host is 'local_folder'")

        if script_args.local_dataset_format == "csv":
            train_ds, val_ds, test_ds, class_names = preprocess_local_csv_dataset(
                script_args.local_folder_path,
                script_args.model,
            )
        elif script_args.local_dataset_format == "manifest":
            train_ds, val_ds, calibration_ds, test_ds, class_names = preprocess_manifest_dataset(
                script_args.local_folder_path,
                script_args.model,
                script_args=script_args,
            )
        elif script_args.local_dataset_format == "folder":
            train_ds, val_ds, test_ds, class_names = preprocess_local_folder_dataset(
                script_args.local_folder_path,
                script_args.model,
            )
        else:
            raise ValueError(
                f"Unknown local_dataset_format: {script_args.local_dataset_format}. Must be 'folder', 'csv', or 'manifest'"
            )
    else:
        raise ValueError(f"Unknown dataset_host: {script_args.dataset_host}")

    seed = getattr(script_args, "seed", 42)
    train_ds = _limit_dataset(train_ds, getattr(script_args, "max_train_samples", None), seed)
    val_ds = _limit_dataset(val_ds, getattr(script_args, "max_eval_samples", None), seed + 1)
    test_ds = _limit_dataset(test_ds, getattr(script_args, "max_test_samples", None), seed + 2)
    calibration_ds = _limit_dataset(calibration_ds, getattr(script_args, "max_test_samples", None), seed + 3)

    # Device setup
    device = get_device()
    print(device)

    # Model configuration and creation
    model = build_model(script_args, class_names=class_names)
    print(
        f"Using model backend={getattr(script_args, 'model_backend', 'custom')} "
        f"type={script_args.model_type} with {script_args.num_labels} labels"
    )

    model.to(device)
    load_initial_checkpoint(model, getattr(script_args, "initial_checkpoint_path", None))

    # Load legacy custom pretrained weights (excluding classifier head)
    if getattr(script_args, "model_backend", "custom").lower() == "custom" and script_args.weights:
        print(script_args.weights)
        pretrained_weights_path = os.path.abspath(script_args.weights)
        pretrained_weights = torch.load(pretrained_weights_path, map_location="cpu")
        filtered_weights = {k: v for k, v in pretrained_weights.items() if "classifier" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(filtered_weights, strict=False)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        print("Pretrained weights loaded successfully!")

    # Freeze early ResNet stages if configured
    if (
        getattr(script_args, "model_backend", "custom").lower() == "custom"
        and script_args.model_type.lower() == "resnet"
        and script_args.num_frozen_stages > 0
    ):
        if script_args.num_frozen_stages >= 1:
            for param in model.resnet.embedder.parameters():
                param.requires_grad = False
        if script_args.num_frozen_stages >= 2:
            for param in model.resnet.encoder.stages[0].parameters():
                param.requires_grad = False
        if script_args.num_frozen_stages >= 3:
            for param in model.resnet.encoder.stages[1].parameters():
                param.requires_grad = False
        print(f"Froze {script_args.num_frozen_stages} early ResNet stage(s)")

    # Training arguments
    report_to = "wandb" if config_flag_enabled(script_args.wandb) else "none"

    metric_for_best = getattr(script_args, "model_selection_metric", None)
    if metric_for_best is None:
        metric_for_best = "selection_score" if script_args.cost_matrix is not None else "accuracy"
    greater_is_better = metric_for_best not in {"selection_score", "normalized_atc", "atc", "loss"}

    checkpoint_root = Path(getattr(script_args, "checkpoint_root", None) or "checkpoints")
    training_args = TrainingArguments(
        output_dir=str(checkpoint_root / script_args.output_dir),
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=script_args.save_total_limit,
        save_only_model=bool(getattr(script_args, "save_only_model", False)),
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        per_device_eval_batch_size=script_args.batch_size,
        num_train_epochs=script_args.num_train_epochs,
        warmup_ratio=script_args.warmup_ratio,
        lr_scheduler_type=script_args.lr_scheduler_type,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
        push_to_hub=config_flag_enabled(script_args.push_to_hub),
        report_to=report_to,
    )

    callbacks = []
    if script_args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=script_args.early_stopping_patience))
        print(f"Early stopping enabled with patience={script_args.early_stopping_patience}")

    try:
        labels_for_priors = train_ds["label"]
        counts = torch.bincount(torch.tensor(labels_for_priors, dtype=torch.long), minlength=script_args.num_labels)
        class_priors = (counts.float() / counts.sum().clamp_min(1)).tolist()
        class_counts = counts.tolist()
    except Exception:
        class_priors = None
        class_counts = None

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=image_processor,
        compute_metrics=make_compute_metrics(script_args, class_names),
        data_collator=collate_fn,
        loss_fxn=script_args.loss_function,
        cost_matrix=script_args.cost_matrix,
        cs_lambda=getattr(script_args, "cs_lambda", 0.0),
        cs_warmup_epochs=getattr(script_args, "cs_warmup_epochs", 0),
        class_priors=class_priors,
        class_counts=class_counts,
        logit_adjustment_tau=getattr(script_args, "logit_adjustment_tau", 1.0),
        nicme_logit_cost_scale=getattr(script_args, "nicme_logit_cost_scale", 1.0),
        cb_beta=getattr(script_args, "cb_beta", 0.9999),
        focal_gamma=getattr(script_args, "focal_gamma", 2.0),
        ldam_max_m=getattr(script_args, "ldam_max_m", 0.5),
        ldam_scale=getattr(script_args, "ldam_scale", 30.0),
        ldam_drw_start_epoch=getattr(script_args, "ldam_drw_start_epoch", 16),
        cost_regularization_alpha=getattr(script_args, "cost_regularization_alpha", 5.0),
        csada_lambda=getattr(script_args, "csada_lambda", 2.0),
        csada_attack_steps=getattr(script_args, "csada_attack_steps", 5),
        csada_epsilon=getattr(script_args, "csada_epsilon", 2.0 / 255.0),
        csada_step_size=getattr(script_args, "csada_step_size", 1.0 / 255.0),
        csada_cost_temperature=getattr(script_args, "csada_cost_temperature", 3.0),
        callbacks=callbacks,
    )

    train_results = trainer.train()
    trainer.log_metrics("train", train_results.metrics)
    validation_metrics = trainer.evaluate(eval_dataset=val_ds)
    validation_predictions = trainer.predict(val_ds)
    validation_report = {
        "metrics": validation_metrics,
        prediction_mode_for_args(script_args): compute_argmax_decision_report(
            validation_predictions.predictions,
            validation_predictions.label_ids,
            script_args,
            class_names=class_names,
        ),
    }

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=image_processor,
        compute_metrics=make_compute_metrics(script_args, class_names),
        data_collator=collate_fn,
        loss_fxn=script_args.loss_function,
        cost_matrix=script_args.cost_matrix,
        cs_lambda=getattr(script_args, "cs_lambda", 0.0),
        cs_warmup_epochs=getattr(script_args, "cs_warmup_epochs", 0),
        class_priors=class_priors,
        class_counts=class_counts,
        logit_adjustment_tau=getattr(script_args, "logit_adjustment_tau", 1.0),
        nicme_logit_cost_scale=getattr(script_args, "nicme_logit_cost_scale", 1.0),
        cb_beta=getattr(script_args, "cb_beta", 0.9999),
        focal_gamma=getattr(script_args, "focal_gamma", 2.0),
        ldam_max_m=getattr(script_args, "ldam_max_m", 0.5),
        ldam_scale=getattr(script_args, "ldam_scale", 30.0),
        ldam_drw_start_epoch=getattr(script_args, "ldam_drw_start_epoch", 16),
        cost_regularization_alpha=getattr(script_args, "cost_regularization_alpha", 5.0),
        csada_lambda=getattr(script_args, "csada_lambda", 2.0),
        csada_attack_steps=getattr(script_args, "csada_attack_steps", 5),
        csada_epsilon=getattr(script_args, "csada_epsilon", 2.0 / 255.0),
        csada_step_size=getattr(script_args, "csada_step_size", 1.0 / 255.0),
        csada_cost_temperature=getattr(script_args, "csada_cost_temperature", 3.0),
    )

    results_dir = perform_comprehensive_evaluation(
        trainer=trainer,
        test_ds=test_ds,
        script_args=script_args,
        dataset_name=script_args.dataset,
        class_names=class_names,
        calibration_ds=calibration_ds,
        validation_report=validation_report,
    )

    return results_dir


if __name__ == "__main__":
    set_seed(42)
    logger = logging.get_logger(__name__)
    args = parse_HF_args()
    main(args)
