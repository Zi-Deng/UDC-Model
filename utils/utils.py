"""Core utilities for training, evaluation, dataset preprocessing, and metrics.

Provides argument parsing (ScriptTrainingArguments), dataset loading for
HuggingFace / Kaggle / local-folder / local-CSV sources, image preprocessing
with data augmentation, comprehensive model evaluation with confusion-matrix
visualizations, and helper functions used throughout the training pipeline.
"""

import argparse
import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from evaluate import load
from PIL import Image
from scipy.special import softmax as scipy_softmax
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import HfArgumentParser

from nicme.calibration import binary_threshold_predictions, fit_binary_threshold_np, fit_temperature_np
from nicme.costs import (
    classification_metrics,
    cost_matrix_sha256,
    cost_min_predictions,
    resolve_class_id,
    softmax_np,
)
from nicme.dataset_profiles import get_profile
from utils.image_processor import CustomImageProcessor

# Try to import visualization libraries (they might not be installed)
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import seaborn as sns

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib and seaborn not available. Visualizations will be skipped.")


def get_device() -> torch.device:
    """Detect the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_fn(images):
    """Stack a list of image dicts into a batched dict for the Trainer."""
    pixel_values = torch.stack([image["pixel_values"] for image in images])
    labels = torch.tensor([image["label"] for image in images])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    """Compute accuracy and macro-F1 for HuggingFace Trainer evaluation callbacks."""
    metric_accuracy = load("accuracy")
    metric_f1 = load("f1")
    outputs, labels = eval_pred
    predictions = np.argmax(outputs, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]

    return {"accuracy": accuracy, "f1": f1}


def compute_metrics_test_no_confusion(eval_pred):
    """Compute accuracy and macro-F1 without confusion matrix.

    Used during test-set evaluation to avoid TensorBoard logging warnings.
    The confusion matrix is computed separately in perform_comprehensive_evaluation().
    """
    metric_accuracy = load("accuracy")
    metric_f1 = load("f1")
    outputs, labels = eval_pred
    predictions = np.argmax(outputs, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]

    return {"accuracy": accuracy, "f1": f1}


def prediction_mode_for_args(script_args) -> str:
    """Return the primary test-time decision rule for a training config."""
    explicit = getattr(script_args, "prediction_mode", None)
    if explicit:
        return str(explicit)
    loss_name = str(getattr(script_args, "loss_function", "") or "").strip().lower()
    if loss_name in {"sosr_cnn", "csdnn_sosr"}:
        return "sosr_argmin"
    return "argmax"


def predictions_and_probabilities(outputs, script_args):
    """Convert raw model outputs into predictions and probability-like scores."""
    outputs = np.asarray(outputs)
    mode = prediction_mode_for_args(script_args)
    if mode == "sosr_argmin":
        predictions = np.argmin(outputs, axis=-1)
        # SOSR outputs are estimated costs, so lower is better.
        probabilities = scipy_softmax(-outputs, axis=-1)
    elif mode == "argmax":
        predictions = np.argmax(outputs, axis=-1)
        probabilities = scipy_softmax(outputs, axis=-1)
    else:
        raise ValueError(f"Unsupported prediction_mode={mode!r}. Expected 'argmax' or 'sosr_argmin'.")
    return predictions, probabilities, mode


def _json_load_if_needed(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            return json.loads(stripped)
    return value


def _get_class_names(script_args) -> list[str] | None:
    names = getattr(script_args, "class_names", None)
    names = _json_load_if_needed(names)
    return list(names) if names is not None else None


def _get_target_class_id(script_args, class_names: list[str] | None, default: int = 0) -> int:
    explicit_id = getattr(script_args, "target_recall_class_id", None)
    if explicit_id is not None:
        return int(explicit_id)
    return resolve_class_id(getattr(script_args, "target_recall_class", None), class_names, default=default)


def _as_list(value) -> list:
    value = _json_load_if_needed(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _get_target_class_ids(script_args, class_names: list[str] | None, default: int = 0) -> list[int]:
    values = _as_list(getattr(script_args, "target_recall_classes", None))
    if not values:
        return [_get_target_class_id(script_args, class_names, default=default)]
    return [resolve_class_id(value, class_names, default=default) for value in values]


def _get_critical_pair_ids(script_args, class_names: list[str] | None) -> list[tuple[int, int, float]]:
    pairs = _json_load_if_needed(getattr(script_args, "critical_pairs", None))
    if not pairs:
        return []
    resolved = []
    for pair in pairs:
        if isinstance(pair, dict):
            true_value = pair.get("true_id", pair.get("true"))
            pred_value = pair.get("pred_id", pair.get("pred"))
            cost = float(pair.get("cost", 0.0))
        else:
            true_value, pred_value, *rest = pair
            cost = float(rest[0]) if rest else 0.0
        resolved.append(
            (
                resolve_class_id(true_value, class_names),
                resolve_class_id(pred_value, class_names),
                cost,
            )
        )
    return resolved


def make_compute_metrics(script_args, class_names: list[str] | None = None):
    """Create a Trainer metrics callback with NICME cost-sensitive metrics."""
    resolved_class_names = class_names or _get_class_names(script_args)
    target_class_id = _get_target_class_id(script_args, resolved_class_names)
    target_class_ids = _get_target_class_ids(script_args, resolved_class_names)
    critical_pairs = _get_critical_pair_ids(script_args, resolved_class_names)
    cost_matrix = getattr(script_args, "cost_matrix", None)
    target_recall_floor = float(getattr(script_args, "target_recall_floor", 0.95))
    accuracy_floor = float(getattr(script_args, "accuracy_floor", 0.80))
    selection_accuracy_metric = getattr(script_args, "selection_accuracy_metric", "accuracy")
    baseline_atc = getattr(script_args, "baseline_atc", None)

    def _compute(eval_pred):
        outputs, labels = eval_pred
        predictions, probs, _mode = predictions_and_probabilities(outputs, script_args)
        metric_accuracy = load("accuracy")
        metric_f1 = load("f1")
        base = {
            "accuracy": metric_accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": metric_f1.compute(predictions=predictions, references=labels, average="macro")["f1"],
        }
        if cost_matrix is None:
            return base
        nicme_metrics = classification_metrics(
            labels,
            predictions,
            probs,
            cost_matrix,
            target_class_id=target_class_id,
            target_recall_floor=target_recall_floor,
            accuracy_floor=accuracy_floor,
            selection_accuracy_metric=selection_accuracy_metric,
            baseline_atc=baseline_atc,
            target_class_ids=target_class_ids,
            class_names=resolved_class_names,
            critical_pairs=critical_pairs,
            include_quadratic_weighted_kappa=getattr(script_args, "metric_profile", "") == "diabetic_retinopathy",
        )
        base.update(
            {
                "balanced_accuracy": nicme_metrics["balanced_accuracy"],
                "macro_f1": nicme_metrics["macro_f1"],
                "atc": nicme_metrics["atc"],
                "normalized_atc": nicme_metrics["normalized_atc"],
                "target_recall": nicme_metrics["target_recall"],
                "target_recall_macro": nicme_metrics["target_recall_macro"],
                "target_fnr": nicme_metrics["target_fnr"],
                "selection_score": nicme_metrics["selection_score"],
                "selection_accuracy_value": nicme_metrics["selection_accuracy_value"],
                "nll": nicme_metrics["nll"],
                "brier": nicme_metrics["brier"],
                "ece": nicme_metrics["ece"],
                "auroc": nicme_metrics["auroc"],
                "auprc": nicme_metrics["auprc"],
            }
        )
        return base

    return _compute


def compute_argmax_decision_report(logits, labels, script_args, class_names: list[str] | None = None):
    """Compute the full primary-decision cost-sensitive report outside Trainer metrics."""
    resolved_class_names = class_names or _get_class_names(script_args)
    cost_matrix = getattr(script_args, "cost_matrix", None)
    if cost_matrix is None:
        return {}
    predictions, probs, mode = predictions_and_probabilities(logits, script_args)
    report = classification_metrics(
        labels,
        predictions,
        probs,
        cost_matrix,
        target_class_id=_get_target_class_id(script_args, resolved_class_names),
        target_recall_floor=float(getattr(script_args, "target_recall_floor", 0.95)),
        accuracy_floor=float(getattr(script_args, "accuracy_floor", 0.80)),
        selection_accuracy_metric=getattr(script_args, "selection_accuracy_metric", "accuracy"),
        target_class_ids=_get_target_class_ids(script_args, resolved_class_names),
        class_names=resolved_class_names,
        critical_pairs=_get_critical_pair_ids(script_args, resolved_class_names),
        include_quadratic_weighted_kappa=getattr(script_args, "metric_profile", "") == "diabetic_retinopathy",
    )
    report["prediction_mode"] = mode
    return report


def compute_cost_min_decision_report(probabilities, labels, script_args, class_names: list[str] | None = None):
    """Compute an uncalibrated Bayes cost-min report from probability estimates."""
    resolved_class_names = class_names or _get_class_names(script_args)
    cost_matrix = getattr(script_args, "cost_matrix", None)
    if cost_matrix is None:
        return {}
    predictions = cost_min_predictions(probabilities, cost_matrix)
    report = classification_metrics(
        labels,
        predictions,
        probabilities,
        cost_matrix,
        target_class_id=_get_target_class_id(script_args, resolved_class_names),
        target_recall_floor=float(getattr(script_args, "target_recall_floor", 0.95)),
        accuracy_floor=float(getattr(script_args, "accuracy_floor", 0.80)),
        selection_accuracy_metric=getattr(script_args, "selection_accuracy_metric", "accuracy"),
        target_class_ids=_get_target_class_ids(script_args, resolved_class_names),
        class_names=resolved_class_names,
        critical_pairs=_get_critical_pair_ids(script_args, resolved_class_names),
        include_quadratic_weighted_kappa=getattr(script_args, "metric_profile", "") == "diabetic_retinopathy",
    )
    report["prediction_mode"] = "cost_min"
    return report


def split_to_train_val_test(dataset):
    """Split a HuggingFace dataset into 80/10/10 train/validation/test splits."""
    split1 = dataset.train_test_split(test_size=0.2)
    train_ds = split1["train"]
    split2 = split1["test"].train_test_split(test_size=0.5)
    val_ds = split2["train"]
    test_ds = split2["test"]
    return train_ds, val_ds, test_ds


def _normalize_bool_string(value):
    """Normalize legacy string booleans and canonical JSON booleans.

    The original configs used "True"/"False" strings for wandb and push_to_hub.
    NICME configs may use real JSON booleans; the training code still accepts
    both representations for backward compatibility.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return "True"
        if lowered in {"false", "0", "no", "n"}:
            return "False"
    return value


def config_flag_enabled(value) -> bool:
    """Return True for either canonical booleans or legacy truthy strings."""
    return _normalize_bool_string(value) == "True"


def validate_training_config(config):
    """Validate high-impact training config invariants before running.

    Args:
        config: A ScriptTrainingArguments instance or dict-like config.

    Raises:
        ValueError: If fields that commonly produce confusing runtime failures
            are invalid.
    """
    if isinstance(config, dict):
        num_labels = config.get("num_labels")
        cost_matrix = config.get("cost_matrix")
        model_type = str(config.get("model_type", "resnet")).lower()
        model_backend = str(config.get("model_backend", "custom")).lower()
        selection_accuracy_metric = str(config.get("selection_accuracy_metric", "accuracy")).lower()
    else:
        num_labels = config.num_labels
        cost_matrix = config.cost_matrix
        model_type = str(config.model_type).lower()
        model_backend = str(getattr(config, "model_backend", "custom")).lower()
        selection_accuracy_metric = str(getattr(config, "selection_accuracy_metric", "accuracy")).lower()

    if model_backend == "custom" and model_type not in {"resnet", "convnext"}:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported types are 'resnet' and 'convnext'.")
    if model_backend not in {"custom", "hf_auto", "hf_backbone", "dinov3_feature", "timm"}:
        raise ValueError(
            f"Unsupported model_backend: {model_backend}. "
            "Supported backends are 'custom', 'hf_auto', 'hf_backbone', 'dinov3_feature', and 'timm'."
        )
    if selection_accuracy_metric not in {"accuracy", "balanced_accuracy", "balanced_acc", "bacc"}:
        raise ValueError("selection_accuracy_metric must be 'accuracy' or 'balanced_accuracy'.")

    if cost_matrix is not None:
        if isinstance(cost_matrix, str):
            cost_matrix = json.loads(cost_matrix)
        if not isinstance(cost_matrix, list) or len(cost_matrix) != num_labels:
            raise ValueError(f"cost_matrix must have {num_labels} rows to match num_labels={num_labels}.")
        for idx, row in enumerate(cost_matrix):
            if not isinstance(row, list) or len(row) != num_labels:
                raise ValueError(
                    f"cost_matrix row {idx} must have {num_labels} columns to match num_labels={num_labels}."
                )


def _apply_dataset_profile_defaults(json_args: dict) -> dict:
    profile_name = json_args.get("dataset_profile")
    if not profile_name:
        return json_args
    profile = get_profile(str(profile_name))
    json_args.setdefault("class_names", list(profile.class_names))
    json_args.setdefault("num_labels", profile.num_classes)
    json_args.setdefault("cost_matrix_source", profile.cost_matrix_source)
    if json_args.get("cost_matrix_source", "dataset") == "dataset":
        if "cost_matrix" in json_args and json_args["cost_matrix"] is not None:
            configured_hash = cost_matrix_sha256(json_args["cost_matrix"])
            if configured_hash != profile.cost_matrix_sha256:
                raise ValueError(
                    f"Configured cost_matrix for {profile.name} does not match the fixed dataset matrix. "
                    "Set cost_matrix_source to a non-dataset value only for explicit ablations."
                )
        json_args["cost_matrix"] = [list(row) for row in profile.cost_matrix]
        json_args["cost_matrix_sha256"] = profile.cost_matrix_sha256
    json_args.setdefault("target_recall_class", profile.target_recall_class)
    json_args.setdefault("target_recall_classes", list(profile.cared_classes))
    json_args.setdefault("secondary_target_recall_classes", list(profile.secondary_target_recall_classes))
    json_args.setdefault("metric_profile", profile.metric_profile)
    json_args.setdefault(
        "critical_pairs",
        [{"true": true_name, "pred": pred_name, "cost": cost} for true_name, pred_name, cost in profile.critical_pairs],
    )
    json_args.setdefault("model_selection_metric", "target_recall")
    return json_args


def parse_HF_args():
    """Parse a ``--config`` JSON file into a :class:`ScriptTrainingArguments` instance.

    The JSON keys map directly to :class:`ScriptTrainingArguments` fields.
    ``cost_matrix`` is serialised to/from a JSON string for HfArgumentParser
    compatibility and converted back to a Python list on return.

    Returns:
        A populated :class:`ScriptTrainingArguments` instance.
    """
    parser = argparse.ArgumentParser(description="Run Hugging Face model with JSON config")
    parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
    args = parser.parse_args()

    with open(args.config) as f:
        json_args = json.load(f)

    json_args = _apply_dataset_profile_defaults(json_args)

    # Allow profile-style configs with a nested selection block.
    if "selection" in json_args and isinstance(json_args["selection"], dict):
        selection = json_args["selection"]
        json_args.setdefault("target_recall_floor", selection.get("target_recall_floor"))
        json_args.setdefault("accuracy_floor", selection.get("accuracy_floor"))
        json_args.setdefault("selection_accuracy_metric", selection.get("accuracy_metric"))
    if "split_dir" in json_args:
        json_args.setdefault("dataset_host", "local_folder")
        json_args.setdefault("local_dataset_format", "manifest")
        json_args.setdefault("local_folder_path", json_args["split_dir"])
    if "num_labels" not in json_args:
        if isinstance(json_args.get("class_names"), list):
            json_args["num_labels"] = len(json_args["class_names"])
        elif isinstance(json_args.get("cost_matrix"), list):
            json_args["num_labels"] = len(json_args["cost_matrix"])

    for bool_key in (
        "wandb",
        "push_to_hub",
        "disable_cudnn",
        "skip_visualizations",
        "train_random_resize_crop",
        "train_horizontal_flip",
        "train_color_jitter",
        "train_random_quarter_turns",
    ):
        if bool_key in json_args:
            json_args[bool_key] = _normalize_bool_string(json_args[bool_key])

    validate_training_config(json_args)

    from dataclasses import fields as dataclass_fields

    field_names = {f.name for f in dataclass_fields(ScriptTrainingArguments)}
    json_args = {k: v for k, v in json_args.items() if k in field_names}

    # HfArgumentParser requires cost_matrix as a JSON string, not a list
    for json_field in ("cost_matrix", "class_names", "target_recall_classes", "secondary_target_recall_classes", "critical_pairs"):
        if json_field in json_args and json_args[json_field] is not None:
            if isinstance(json_args[json_field], (list, dict)):
                json_args[json_field] = json.dumps(json_args[json_field])

    if "peft_enabled" in json_args:
        json_args["peft_enabled"] = bool(json_args["peft_enabled"])

    if "cost_matrix" in json_args and json_args["cost_matrix"] is not None:
        if isinstance(json_args["cost_matrix"], list):
            json_args["cost_matrix"] = json.dumps(json_args["cost_matrix"])

    hf_parser = HfArgumentParser(ScriptTrainingArguments)
    parsed_args = hf_parser.parse_dict(json_args)[0]

    # Convert cost_matrix back to a Python list
    if parsed_args.cost_matrix is not None:
        parsed_args.cost_matrix = json.loads(parsed_args.cost_matrix)
    if parsed_args.class_names is not None:
        parsed_args.class_names = json.loads(parsed_args.class_names)
    for json_field in ("target_recall_classes", "secondary_target_recall_classes", "critical_pairs"):
        value = getattr(parsed_args, json_field)
        if value is not None and str(value).strip().startswith(("[", "{")):
            setattr(parsed_args, json_field, json.loads(value))

    validate_training_config(parsed_args)
    return parsed_args


@dataclass
class ScriptTrainingArguments:
    """All configuration fields parsed from a training JSON config file.

    ``wandb`` and ``push_to_hub`` accept canonical JSON booleans in NICME
    configs and legacy ``"True"``/``"False"`` strings in older configs.
    """

    dataset: str = field(default=None, metadata={"help": "Name of dataset from HG hub"})
    model: str = field(default=None, metadata={"help": "Name of model from HG hub"})
    model_backend: str = field(
        default="custom", metadata={"help": "Model backend: custom, hf_auto, hf_backbone, or dinov3_feature"}
    )
    model_type: str = field(default="resnet", metadata={"help": "Type of model to use: 'resnet' or 'convnext'"})
    weights: str = field(default=None, metadata={"help": "Weight of model from HG hub"})
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate for training"})
    num_train_epochs: int = field(default=5, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=16, metadata={"help": "Batch size of training epochs"})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps"})
    image_size: int | None = field(default=None, metadata={"help": "Override square image size for preprocessing"})
    num_labels: int = field(default=5, metadata={"help": "Number of training labels"})
    wandb: str = field(default="False", metadata={"help": "Wandb upload ('True' or 'False' as strings)"})
    push_to_hub: str = field(default="False", metadata={"help": "Push to hub ('True' or 'False' as strings)"})
    output_dir: str = field(default="convnext", metadata={"help": "Output directory for model checkpoints"})
    dataset_host: str = field(default="huggingface", metadata={"help": "Dataset host"})
    local_dataset_name: str = field(default=None, metadata={"help": "Name of local dataset"})
    local_folder_path: str = field(
        default=None, metadata={"help": "Path to local folder containing class subfolders with images"}
    )
    loss_function: str = field(default="cross_entropy", metadata={"help": "Loss function to use"})
    cost_matrix: str | None = field(
        default=None, metadata={"help": "Cost matrix for cost-sensitive loss functions (JSON string)"}
    )
    sweep_mode: bool = field(default=False, metadata={"help": "Whether we are running in cost matrix sweep mode"})
    sweep_cost_value: float = field(
        default=None, metadata={"help": "Current cost value being swept (only used in sweep mode)"}
    )
    sweep_matrix_row: int = field(
        default=None, metadata={"help": "Row index of cost matrix cell being swept (only used in sweep mode)"}
    )
    sweep_matrix_col: int = field(
        default=None, metadata={"help": "Column index of cost matrix cell being swept (only used in sweep mode)"}
    )
    local_dataset_format: str = field(
        default="folder",
        metadata={
            "help": "Format of local dataset: 'folder' for class subfolders or 'csv' for CSV files with image paths and labels"
        },
    )
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW optimizer"})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio for learning rate scheduler"})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "LR scheduler type: 'linear', 'cosine', etc."})
    num_frozen_stages: int = field(
        default=0,
        metadata={"help": "Number of early ResNet stages to freeze (0-3). 0=none, 1=embedder, 2=+stage0, 3=+stage1"},
    )
    early_stopping_patience: int = field(
        default=0, metadata={"help": "Early stopping patience (epochs without improvement). 0 = disabled."}
    )
    cs_lambda: float = field(default=0.0, metadata={"help": "Cost-sensitive regularization weight (0 = disabled)"})
    cs_warmup_epochs: int = field(
        default=0, metadata={"help": "Epochs to warmup CS lambda from 0 to full value (0 = no warmup)"}
    )
    class_names: str | None = field(default=None, metadata={"help": "JSON list of class names in label-id order"})
    target_recall_class: str | None = field(default=None, metadata={"help": "Class name/id optimized for recall"})
    target_recall_class_id: int | None = field(default=None, metadata={"help": "Target recall class id override"})
    target_recall_floor: float = field(default=0.95, metadata={"help": "Minimum desired target-class recall"})
    accuracy_floor: float = field(default=0.80, metadata={"help": "Minimum desired accuracy/balanced accuracy"})
    selection_accuracy_metric: str = field(
        default="accuracy",
        metadata={"help": "Metric used in selection gap: 'accuracy' or 'balanced_accuracy'"},
    )
    baseline_atc: float | None = field(default=None, metadata={"help": "Baseline ATC for CRR reporting"})
    dataset_profile: str | None = field(default=None, metadata={"help": "NICME dataset profile name"})
    cost_matrix_source: str = field(default="dataset", metadata={"help": "Source for cost matrix: dataset or ablation"})
    cost_matrix_sha256: str | None = field(default=None, metadata={"help": "SHA256 hash of the active cost matrix"})
    target_recall_classes: str | None = field(default=None, metadata={"help": "JSON/CSV list of target recall classes"})
    secondary_target_recall_classes: str | None = field(
        default=None, metadata={"help": "JSON/CSV list of secondary target recall classes"}
    )
    metric_profile: str = field(default="", metadata={"help": "Dataset-specific metric profile"})
    critical_pairs: str | None = field(default=None, metadata={"help": "JSON list of critical true->pred pairs"})
    model_selection_metric: str | None = field(
        default=None, metadata={"help": "Trainer best-checkpoint metric, e.g. selection_score or target_recall"}
    )
    max_train_samples: int | None = field(default=None, metadata={"help": "Optional train subset limit for smoke runs"})
    max_eval_samples: int | None = field(default=None, metadata={"help": "Optional validation subset limit for smoke runs"})
    max_test_samples: int | None = field(default=None, metadata={"help": "Optional test/calibration subset limit for smoke runs"})
    logit_adjustment_tau: float = field(default=1.0, metadata={"help": "Tau for Menon prior-logit adjustment"})
    nicme_logit_cost_scale: float = field(
        default=1.0,
        metadata={"help": "Scale applied to raw cost values inside NICME logit adjustment losses"},
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})
    calibration_enabled: bool = field(default=False, metadata={"help": "Enable calibration split evaluation"})
    decision_modes: str = field(default="argmax,calibrated_cost_min", metadata={"help": "Comma-separated modes"})
    prediction_mode: str | None = field(default=None, metadata={"help": "Primary prediction rule: argmax or sosr_argmin"})
    peft_enabled: bool = field(default=False, metadata={"help": "Enable PEFT/LoRA"})
    peft_r: int = field(default=8, metadata={"help": "LoRA rank"})
    peft_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    peft_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    peft_target_modules: str | None = field(default=None, metadata={"help": "Comma-separated LoRA target modules"})
    peft_modules_to_save: str | None = field(default=None, metadata={"help": "Comma-separated trainable modules to save"})
    timm_pretrained: bool = field(default=True, metadata={"help": "Load pretrained weights for timm backends"})
    disable_cudnn: bool = field(default=False, metadata={"help": "Disable cuDNN for native Conv2d stability"})
    skip_visualizations: bool = field(default=False, metadata={"help": "Skip optional confusion-matrix figures"})
    save_total_limit: int = field(default=1, metadata={"help": "Maximum checkpoints to retain per run"})
    train_random_resize_crop: bool = field(default=True, metadata={"help": "Use RandomResizedCrop for training"})
    train_horizontal_flip: bool = field(default=True, metadata={"help": "Use random horizontal flips for training"})
    train_color_jitter: bool = field(default=False, metadata={"help": "Use light ColorJitter augmentation for training"})
    train_random_quarter_turns: bool = field(
        default=False,
        metadata={"help": "Randomly rotate training images by 0/90/180/270 degrees"},
    )
    train_random_resized_crop_scale_min: float = field(
        default=0.08,
        metadata={"help": "Minimum scale for RandomResizedCrop"},
    )
    train_random_resized_crop_scale_max: float = field(
        default=1.0,
        metadata={"help": "Maximum scale for RandomResizedCrop"},
    )
    initial_checkpoint_path: str | None = field(
        default=None,
        metadata={"help": "Optional checkpoint file or directory to initialize model weights from"},
    )
    cb_beta: float = field(default=0.9999, metadata={"help": "Class-balanced effective-number beta"})
    focal_gamma: float = field(default=2.0, metadata={"help": "Focal loss gamma"})
    ldam_max_m: float = field(default=0.5, metadata={"help": "LDAM maximum margin"})
    ldam_scale: float = field(default=30.0, metadata={"help": "LDAM logit scale"})
    ldam_drw_start_epoch: int = field(default=16, metadata={"help": "Epoch at which LDAM deferred reweighting starts"})
    ap_alpha: float = field(default=5.0, metadata={"help": "Adjusted-penalty expected-cost weight"})
    csada_lambda: float = field(default=2.0, metadata={"help": "CSADA adversarial CE weight"})
    csada_attack_steps: int = field(default=5, metadata={"help": "CSADA targeted PGD steps"})
    csada_epsilon: float = field(default=2.0 / 255.0, metadata={"help": "CSADA epsilon in input pixel units"})
    csada_step_size: float = field(default=1.0 / 255.0, metadata={"help": "CSADA step size in input pixel units"})
    csada_cost_temperature: float = field(default=3.0, metadata={"help": "CSADA critical-pair cost weighting temperature"})


def preprocess_hf_dataset(dataset_name, model_name):
    """Download and preprocess a HuggingFace Hub dataset with train/val/test splits.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. ``"beans"``).
        model_name: Model name for the image processor configuration.

    Returns:
        Tuple of (train_ds, val_ds, test_ds) with transforms applied.
    """
    dataset = load_dataset(dataset_name)
    image_processor = CustomImageProcessor.from_pretrained(model_name, use_fast=True)
    image_preprocessor = ImagePreprocessor(dataset, image_processor)

    train_ds, val_ds, test_ds = split_to_train_val_test(dataset["train"])
    train_ds.set_transform(image_preprocessor.preprocess_train)
    val_ds.set_transform(image_preprocessor.preprocess_val)
    test_ds.set_transform(image_preprocessor.preprocess_test)

    return train_ds, val_ds, test_ds


def preprocess_kg_dataset(cloud_dataset_name, local_dataset_name, model_name):
    """Download and preprocess a Kaggle dataset with train/val/test splits.

    Args:
        cloud_dataset_name: Kaggle dataset identifier (e.g. ``"user/dataset"``).
        local_dataset_name: Subdirectory name inside the downloaded dataset.
        model_name: Model name for the image processor configuration.

    Returns:
        Tuple of (train_ds, val_ds, test_ds) with transforms applied.
    """
    path = kagglehub.dataset_download(cloud_dataset_name)
    local_path = f"{path}/{local_dataset_name}"
    dataset = load_dataset("imagefolder", data_dir=local_path)
    image_processor = CustomImageProcessor.from_pretrained(model_name, use_fast=True)
    image_preprocessor = ImagePreprocessor(dataset, image_processor)

    train_ds, val_ds, test_ds = split_to_train_val_test(dataset["train"])
    train_ds.set_transform(image_preprocessor.preprocess_train)
    val_ds.set_transform(image_preprocessor.preprocess_val)
    test_ds.set_transform(image_preprocessor.preprocess_test)

    return train_ds, val_ds, test_ds


class ImagePreprocessor:
    """Applies per-split image transforms to HuggingFace dataset batches.

    Handles both image-path-based (local) and pre-loaded-image (HuggingFace)
    datasets transparently. Training uses data augmentation; validation and
    test use deterministic transforms.

    Args:
        dataset: The source dataset (unused; kept for API compatibility).
        image_processor: A CustomImageProcessor instance that provides the
            torchvision transform pipelines.
    """

    def __init__(
        self,
        dataset,
        image_processor,
        random_resize_crop: bool = True,
        horizontal_flip: bool = True,
        color_jitter: bool = False,
        random_quarter_turns: bool = False,
        random_resized_crop_scale: tuple[float, float] = (0.08, 1.0),
    ):
        self.image_processor = image_processor
        self.train_transforms = image_processor.get_transform_for_training(
            random_resize_crop=random_resize_crop,
            horizontal_flip=horizontal_flip,
            color_jitter=color_jitter,
            random_quarter_turns=random_quarter_turns,
            random_resized_crop_scale=random_resized_crop_scale,
        )
        self.val_transforms = image_processor.get_transform_for_validation()
        self.test_transforms = image_processor.get_transform_for_test()

    def _preprocess_batch(self, image_batch, transform):
        """Load images from a batch and apply *transform* to each one.

        Supports batches keyed by ``"image_path"`` (local datasets) or
        ``"image"`` (HuggingFace datasets).  Missing / unreadable images are
        replaced with a black 224x224 placeholder so the batch size stays
        consistent.
        """
        images = []
        if "image_path" in image_batch:
            for img_path in image_batch["image_path"]:
                try:
                    images.append(Image.open(img_path).convert("RGB"))
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    images.append(Image.new("RGB", (224, 224), color="black"))
        elif "image" in image_batch:
            for image in image_batch["image"]:
                if isinstance(image, Image.Image):
                    images.append(image.convert("RGB"))
                else:
                    images.append(Image.new("RGB", (224, 224), color="black"))
        else:
            raise ValueError("Image batch must contain either 'image_path' or 'image' column")

        image_batch["pixel_values"] = [transform(img) for img in images]
        return image_batch

    def preprocess_train(self, image_batch):
        """Apply training transforms (with data augmentation) to *image_batch*."""
        return self._preprocess_batch(image_batch, self.train_transforms)

    def preprocess_val(self, image_batch):
        """Apply validation transforms (deterministic) to *image_batch*."""
        return self._preprocess_batch(image_batch, self.val_transforms)

    def preprocess_test(self, image_batch):
        """Apply test transforms (deterministic) to *image_batch*."""
        return self._preprocess_batch(image_batch, self.test_transforms)


def _create_dataset_dict(image_paths, labels):
    """Create a HuggingFace-compatible dict storing image paths and labels."""
    return {"image_path": image_paths, "label": labels}


def _make_image_preprocessor(model_name, script_args=None):
    """Build an ImagePreprocessor from a model name.

    Uses a lightweight placeholder dataset object since ImagePreprocessor
    only needs the image_processor, not the dataset itself.
    """
    image_size = getattr(script_args, "image_size", None) if script_args is not None else None
    image_processor = CustomImageProcessor.from_pretrained(model_name, use_fast=True, image_size=image_size)
    random_resize_crop = True
    horizontal_flip = True
    color_jitter = False
    random_quarter_turns = False
    crop_scale = (0.08, 1.0)
    if script_args is not None:
        random_resize_crop = config_flag_enabled(getattr(script_args, "train_random_resize_crop", True))
        horizontal_flip = config_flag_enabled(getattr(script_args, "train_horizontal_flip", True))
        color_jitter = config_flag_enabled(getattr(script_args, "train_color_jitter", False))
        random_quarter_turns = config_flag_enabled(getattr(script_args, "train_random_quarter_turns", False))
        scale_min = float(getattr(script_args, "train_random_resized_crop_scale_min", 0.08))
        scale_max = float(getattr(script_args, "train_random_resized_crop_scale_max", 1.0))
        crop_scale = (max(0.0, min(scale_min, scale_max)), min(1.0, max(scale_min, scale_max)))
    return ImagePreprocessor(
        None,
        image_processor,
        random_resize_crop=random_resize_crop,
        horizontal_flip=horizontal_flip,
        color_jitter=color_jitter,
        random_quarter_turns=random_quarter_turns,
        random_resized_crop_scale=crop_scale,
    )


def save_metrics_to_file(metrics, class_metrics, class_counts, output_dir, filename_suffix="", validation_report=None):
    """
    Save all metrics to JSON and text files for easy analysis.

    Args:
        metrics: Dictionary containing evaluation metrics
        class_metrics: Dictionary containing per-class metrics
        class_counts: Dictionary containing sample counts per class
        output_dir: Directory to save the metrics files
        filename_suffix: Optional suffix for the filename
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"metrics_{timestamp}{filename_suffix}"

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj

    # Prepare comprehensive metrics dictionary
    comprehensive_metrics = {
        "timestamp": timestamp,
        "overall_metrics": {
            "accuracy": float(metrics.get("eval_accuracy", 0.0)),
            "f1_score": float(metrics.get("eval_f1", 0.0)),
            "loss": float(metrics.get("eval_loss", 0.0)),
            "balanced_accuracy": float(metrics.get("eval_balanced_accuracy", 0.0)),
            "macro_f1": float(metrics.get("eval_macro_f1", metrics.get("eval_f1", 0.0))),
            "kappa": float(metrics.get("eval_kappa", 0.0)),
            "quadratic_weighted_kappa": float(metrics.get("eval_quadratic_weighted_kappa", 0.0)),
            "auc": float(metrics.get("eval_auc", 0.0)),
            "auroc": float(metrics.get("eval_auroc", metrics.get("eval_auc", 0.0))),
            "auprc": float(metrics.get("eval_auprc", 0.0)),
            "recall_class0": float(metrics.get("eval_recall_class0", 0.0)),
            "expected_cost": float(metrics.get("eval_expected_cost", 0.0)),
            "atc": float(metrics.get("eval_atc", metrics.get("eval_expected_cost", 0.0))),
            "normalized_atc": float(metrics.get("eval_normalized_atc", 0.0)),
            "selection_score": float(metrics.get("eval_selection_score", 0.0)),
            "selection_accuracy_value": float(metrics.get("eval_selection_accuracy_value", 0.0)),
            "selection_accuracy_metric": metrics.get(
                "eval_decision_reports", {}
            )
            .get(metrics.get("eval_primary_decision_mode", "argmax"), {})
            .get("selection_accuracy_metric", "accuracy"),
            "primary_decision_mode": metrics.get("eval_primary_decision_mode", "argmax"),
            "target_recall": float(metrics.get("eval_target_recall", 0.0)),
            "target_recall_macro": float(metrics.get("eval_target_recall_macro", metrics.get("eval_target_recall", 0.0))),
            "target_fnr": float(metrics.get("eval_target_fnr", 0.0)),
            "nll": float(metrics.get("eval_nll", 0.0)),
            "brier": float(metrics.get("eval_brier", 0.0)),
            "ece": float(metrics.get("eval_ece", 0.0)),
            "prevalence_class0": float(metrics.get("eval_prevalence_class0", 0.0)),
        },
        "class_counts": convert_numpy_types(class_counts),
        "per_class_metrics": convert_numpy_types(class_metrics),
        "confusion_matrix": convert_numpy_types(metrics.get("eval_confusion_matrix", {}).get("confusion_matrix", [])),
        "decision_reports": convert_numpy_types(metrics.get("eval_decision_reports", {})),
        "validation_report": convert_numpy_types(validation_report or {}),
        "skipped_decision_modes": convert_numpy_types(metrics.get("eval_skipped_decision_modes", {})),
        "cost_matrix_sha256": metrics.get("eval_cost_matrix_sha256"),
    }

    # Save as JSON
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, "w") as f:
        json.dump(comprehensive_metrics, f, indent=2)

    # Save as readable text file
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Training Metrics Report - {timestamp}\n")
        f.write("=" * 50 + "\n\n")

        f.write("OVERALL METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Accuracy: {comprehensive_metrics['overall_metrics']['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {comprehensive_metrics['overall_metrics']['balanced_accuracy']:.4f}\n")
        f.write(f"F1 Score: {comprehensive_metrics['overall_metrics']['f1_score']:.4f}\n")
        f.write(f"Kappa: {comprehensive_metrics['overall_metrics']['kappa']:.4f}\n")
        f.write(f"Quadratic Weighted Kappa: {comprehensive_metrics['overall_metrics']['quadratic_weighted_kappa']:.4f}\n")
        f.write(f"AUC: {comprehensive_metrics['overall_metrics']['auc']:.4f}\n")
        f.write(f"Loss: {comprehensive_metrics['overall_metrics']['loss']:.4f}\n")
        f.write(f"ATC: {comprehensive_metrics['overall_metrics']['atc']:.4f}\n")
        f.write(f"Normalized ATC: {comprehensive_metrics['overall_metrics']['normalized_atc']:.4f}\n")
        f.write(f"Selection Score: {comprehensive_metrics['overall_metrics']['selection_score']:.4f}\n")
        f.write(f"Target Recall: {comprehensive_metrics['overall_metrics']['target_recall']:.4f}\n")
        f.write(f"Target FNR: {comprehensive_metrics['overall_metrics']['target_fnr']:.4f}\n")
        f.write(f"Recall Class 0: {comprehensive_metrics['overall_metrics']['recall_class0']:.4f}\n")
        f.write(f"Expected Cost: {comprehensive_metrics['overall_metrics']['expected_cost']:.4f}\n")
        f.write(f"Prevalence Class 0: {comprehensive_metrics['overall_metrics']['prevalence_class0']:.4f}\n\n")

        f.write("CLASS SAMPLE COUNTS:\n")
        f.write("-" * 20 + "\n")
        for class_id, count in sorted(comprehensive_metrics["class_counts"].items()):
            f.write(f"Class {class_id}: {count} samples\n")
        f.write("\n")

        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 20 + "\n")
        class_metrics_data = comprehensive_metrics["per_class_metrics"]
        for class_id in sorted(class_metrics_data["accuracy"].keys()):
            f.write(f"Class {class_id}:\n")
            f.write(f"  True Positive Rate (TPR): {class_metrics_data['accuracy'][class_id]:.4f}\n")
            f.write(f"  False Positive Rate (FPR): {class_metrics_data['false_positive_rate'][class_id]:.4f}\n")
            f.write(f"  False Negative Rate (FNR): {class_metrics_data['false_negative_rate'][class_id]:.4f}\n")
            f.write("\n")

        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 20 + "\n")
        confusion_matrix = comprehensive_metrics["confusion_matrix"]
        if confusion_matrix:
            for i, row in enumerate(confusion_matrix):
                f.write(f"Row {i}: {row}\n")

    print(f"Metrics saved to: {json_path} and {txt_path}")
    return json_path, txt_path


def create_confusion_matrix_visualization(confusion_matrix, class_names=None, output_dir="results", filename_suffix=""):
    """
    Create a professional confusion matrix visualization with three views:
    1. Raw counts
    2. Recall (normalized by true class)
    3. Precision (normalized by predicted class)

    Args:
        confusion_matrix: 2D array/list containing the confusion matrix
        class_names: List of class names (optional)
        output_dir: Directory to save the visualization
        filename_suffix: Optional suffix for the filename
    """
    if not VISUALIZATION_AVAILABLE:
        print("Skipping visualization: matplotlib/seaborn not available")
        return None

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert to numpy array if needed
    cm = np.array(confusion_matrix)

    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]

    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up the matplotlib figure with professional styling for three subplots
    plt.style.use("default")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # Plot 1: Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
        cbar_kws={"label": "Count"},
    )
    ax1.set_title("Confusion Matrix - Raw Counts", fontsize=16, fontweight="bold", pad=20)
    ax1.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax1.set_ylabel("True Label", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="both", which="major", labelsize=12)

    # Plot 2: Normalized by true class (recall/sensitivity)
    cm_recall = np.divide(
        cm.astype("float"),
        cm.sum(axis=1)[:, np.newaxis],
        out=np.zeros_like(cm, dtype=float),
        where=cm.sum(axis=1)[:, np.newaxis] != 0,
    )

    sns.heatmap(
        cm_recall,
        annot=True,
        fmt=".3f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
        cbar_kws={"label": "Recall (TPR)"},
    )
    ax2.set_title("Confusion Matrix - Recall\n(Normalized by True Class)", fontsize=16, fontweight="bold", pad=20)
    ax2.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax2.set_ylabel("True Label", fontsize=14, fontweight="bold")
    ax2.tick_params(axis="both", which="major", labelsize=12)

    # Plot 3: Normalized by predicted class (precision)
    cm_precision = np.divide(
        cm.astype("float"),
        cm.sum(axis=0)[np.newaxis, :],
        out=np.zeros_like(cm, dtype=float),
        where=cm.sum(axis=0)[np.newaxis, :] != 0,
    )

    sns.heatmap(
        cm_precision,
        annot=True,
        fmt=".3f",
        cmap="Oranges",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax3,
        cbar_kws={"label": "Precision (PPV)"},
    )
    ax3.set_title(
        "Confusion Matrix - Precision\n(Normalized by Predicted Class)", fontsize=16, fontweight="bold", pad=20
    )
    ax3.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax3.set_ylabel("True Label", fontsize=14, fontweight="bold")
    ax3.tick_params(axis="both", which="major", labelsize=12)

    # Adjust layout and save
    plt.tight_layout(pad=3.0)

    # Save the figure
    output_path = os.path.join(output_dir, f"confusion_matrix_{timestamp}{filename_suffix}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")

    print(
        f"Confusion matrix visualizations (3-panel) saved to: {output_path} and {output_path.replace('.png', '.pdf')}"
    )

    # Also create a separate detailed visualization
    create_detailed_confusion_matrix(cm, class_names, output_dir, filename_suffix, timestamp)

    plt.close()
    return output_path


def create_detailed_confusion_matrix(cm, class_names, output_dir, filename_suffix, timestamp):
    """
    Create a detailed confusion matrix with additional statistics.
    """
    if not VISUALIZATION_AVAILABLE:
        return None

    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate additional statistics
    total_samples = np.sum(cm)
    accuracy_per_class = np.divide(
        np.diag(cm),
        np.sum(cm, axis=1),
        out=np.zeros(len(class_names), dtype=float),
        where=np.sum(cm, axis=1) != 0,
    )
    precision_per_class = np.divide(
        np.diag(cm),
        np.sum(cm, axis=0),
        out=np.zeros(len(class_names), dtype=float),
        where=np.sum(cm, axis=0) != 0,
    )

    # Create the heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=270, labelpad=20, fontsize=12, fontweight="bold")

    # Set up the plot
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=14, fontweight="bold")
    ax.set_title("Detailed Confusion Matrix with Statistics", fontsize=16, fontweight="bold", pad=20)

    # Annotate each cell with count and percentage
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percentage = count / total_samples * 100
            text_color = "white" if count > thresh else "black"
            ax.text(
                j, i, f"{count}\n({percentage:.1f}%)", ha="center", va="center", color=text_color, fontweight="bold"
            )

    # Add statistics text
    stats_text = "Per-Class Statistics:\n"
    for i, class_name in enumerate(class_names):
        recall = accuracy_per_class[i] if not np.isnan(accuracy_per_class[i]) else 0
        precision = precision_per_class[i] if not np.isnan(precision_per_class[i]) else 0
        stats_text += f"{class_name}: Recall={recall:.3f}, Precision={precision:.3f}\n"

    # Add text box with statistics
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment="center", bbox=props)

    plt.tight_layout()

    # Save the detailed figure
    detailed_path = os.path.join(output_dir, f"confusion_matrix_detailed_{timestamp}{filename_suffix}.png")
    plt.savefig(detailed_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Detailed confusion matrix saved to: {detailed_path}")
    return detailed_path


def perform_comprehensive_evaluation(
    trainer,
    test_ds,
    script_args,
    dataset_name=None,
    class_names=None,
    calibration_ds=None,
    validation_report=None,
):
    """
    Perform comprehensive evaluation including confusion matrix, per-class metrics, and visualizations.

    Args:
        trainer: The trained model trainer
        test_ds: Test dataset
        script_args: Script arguments containing configuration
        dataset_name: Name of the dataset (optional)
        class_names: List of class names (optional, for local datasets)

    Returns:
        str: Path to the results directory
    """
    print("\n" + "=" * 60)
    print("PERFORMING COMPREHENSIVE EVALUATION")
    print("=" * 60)

    # Create results directory with timestamp
    now = datetime.datetime.now()
    date_str = now.strftime("%m-%d")  # Format: MM-DD
    time_str = now.strftime("%H-%M")  # Format: HH-MM

    # Build directory name based on sweep mode
    if script_args.sweep_mode and script_args.sweep_cost_value is not None:
        # Sweep mode: {output_dir}__modifier{cost_value}_true{row}_predict{col}_{date}_{time}
        cost_str = str(script_args.sweep_cost_value)  # Keep decimal point as specified
        dir_name = f"{script_args.output_dir}__modifier{cost_str}_true{script_args.sweep_matrix_row}_predict{script_args.sweep_matrix_col}_{date_str}_{time_str}"
    else:
        # Normal mode: {output_dir}_{date}_{time}
        dir_name = f"{script_args.output_dir}_{date_str}_{time_str}"

    model_family = getattr(script_args, "model_type", "model").lower()
    results_dir = f"results/{model_family}_test/{dir_name}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {results_dir}")

    # Save run configuration
    print("Saving run configuration...")
    save_run_configuration(script_args, results_dir, dataset_name)

    # Perform evaluation
    print("Running model evaluation...")
    metrics = trainer.evaluate()

    # Get confusion matrix
    print("Computing detailed metrics...")
    metric_confusion = load("confusion_matrix")

    # Get predictions
    predictions = trainer.predict(test_ds)
    y_true = predictions.label_ids
    y_pred, y_proba, primary_decision_mode = predictions_and_probabilities(predictions.predictions, script_args)

    # Compute confusion matrix
    confusion_result = metric_confusion.compute(predictions=y_pred, references=y_true)
    confusion_matrix = confusion_result["confusion_matrix"]

    # Add confusion matrix to metrics
    metrics["eval_confusion_matrix"] = confusion_result

    # Calculate per-class metrics
    num_classes = len(confusion_matrix)
    total_samples = np.sum(confusion_matrix)

    # Count samples per class
    class_counts = {}
    total_samples_per_class = {}
    for i in range(num_classes):
        class_total = sum(confusion_matrix[i])
        class_counts[i] = class_total
        total_samples_per_class[i] = class_total

    # Initialize per-class metrics dictionary
    class_metrics = {
        "accuracy": {},
        "false_positive_rate": {},
        "false_negative_rate": {},
        "precision": {},
        "recall": {},
        "f1_score": {},
        "support": {},
    }

    # Calculate metrics for each class
    for i in range(num_classes):
        true_positives = confusion_matrix[i][i]
        false_negatives = sum(confusion_matrix[i]) - true_positives
        false_positives = sum(row[i] for row in confusion_matrix) - true_positives
        true_negatives = total_samples - (true_positives + false_negatives + false_positives)

        # Recall (True Positive Rate) - same as accuracy per class
        class_metrics["recall"][i] = (
            true_positives / total_samples_per_class[i] if total_samples_per_class[i] > 0 else 0.0
        )
        class_metrics["accuracy"][i] = class_metrics["recall"][i]  # For backward compatibility

        # Precision
        class_metrics["precision"][i] = (
            true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        )

        # F1 Score
        precision = class_metrics["precision"][i]
        recall = class_metrics["recall"][i]
        class_metrics["f1_score"][i] = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        # False Positive Rate (FPR)
        class_metrics["false_positive_rate"][i] = (
            false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
        )

        # False Negative Rate (FNR)
        class_metrics["false_negative_rate"][i] = (
            false_negatives / total_samples_per_class[i] if total_samples_per_class[i] > 0 else 0.0
        )

        # Support (number of true samples for this class)
        class_metrics["support"][i] = int(total_samples_per_class[i])

    # -- New overall metrics --
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa_val = cohen_kappa_score(y_true, y_pred)
    qwk_val = cohen_kappa_score(y_true, y_pred, weights="quadratic") if num_classes > 1 else 0.0

    # AUC from probability-like scores for the primary decision rule.
    try:
        if num_classes == 2:
            auc_val = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc_val = roc_auc_score(y_true, y_proba, multi_class="ovo", average="weighted")
    except ValueError:
        auc_val = 0.0

    recall_class0 = class_metrics["recall"].get(0, 0.0)
    prevalence_class0 = float(np.mean(y_true == 0))

    # Cost-sensitive reports using NICME's row=true, column=pred convention.
    cost_matrix_cfg = None
    if hasattr(script_args, "cost_matrix") and script_args.cost_matrix is not None:
        cost_matrix_cfg = script_args.cost_matrix
    decision_reports = {}
    if cost_matrix_cfg is not None:
        configured_class_names = class_names or _get_class_names(script_args)
        target_class_id = _get_target_class_id(script_args, configured_class_names)
        target_class_ids = _get_target_class_ids(script_args, configured_class_names)
        critical_pairs = _get_critical_pair_ids(script_args, configured_class_names)
        include_qwk = getattr(script_args, "metric_profile", "") == "diabetic_retinopathy"
        primary_report = classification_metrics(
            y_true,
            y_pred,
            y_proba,
            cost_matrix_cfg,
            target_class_id=target_class_id,
            target_recall_floor=float(getattr(script_args, "target_recall_floor", 0.95)),
            accuracy_floor=float(getattr(script_args, "accuracy_floor", 0.80)),
            selection_accuracy_metric=getattr(script_args, "selection_accuracy_metric", "accuracy"),
            target_class_ids=target_class_ids,
            class_names=configured_class_names,
            critical_pairs=critical_pairs,
            include_quadratic_weighted_kappa=include_qwk,
        )
        primary_report["prediction_mode"] = primary_decision_mode
        decision_reports[primary_decision_mode] = primary_report
        if primary_decision_mode == "argmax":
            decision_reports["argmax"] = primary_report
        expected_cost = primary_report["atc"]

        decision_modes = {
            mode.strip() for mode in str(getattr(script_args, "decision_modes", "argmax")).split(",") if mode.strip()
        }
        if "cost_min" in decision_modes or "ce_cost_min_inference" in decision_modes:
            decision_reports["cost_min"] = compute_cost_min_decision_report(
                y_proba,
                y_true,
                script_args,
                class_names=configured_class_names,
            )
        if calibration_ds is not None and (
            "calibrated_cost_min" in decision_modes or "calibrated_threshold" in decision_modes
        ):
            calibration_predictions = trainer.predict(calibration_ds)
            calibration_result = fit_temperature_np(calibration_predictions.predictions, calibration_predictions.label_ids)
            calibration_probs = softmax_np(calibration_predictions.predictions, calibration_result.temperature)
            calibrated_probs = softmax_np(predictions.predictions, calibration_result.temperature)
            if "calibrated_cost_min" in decision_modes:
                calibrated_cost_pred = cost_min_predictions(calibrated_probs, cost_matrix_cfg)
                calibrated_report = classification_metrics(
                    y_true,
                    calibrated_cost_pred,
                    calibrated_probs,
                    cost_matrix_cfg,
                    target_class_id=target_class_id,
                    target_recall_floor=float(getattr(script_args, "target_recall_floor", 0.95)),
                    accuracy_floor=float(getattr(script_args, "accuracy_floor", 0.80)),
                    selection_accuracy_metric=getattr(script_args, "selection_accuracy_metric", "accuracy"),
                )
                calibrated_report["temperature"] = calibration_result.temperature
                calibrated_report["calibration_nll_before"] = calibration_result.nll_before
                calibrated_report["calibration_nll_after"] = calibration_result.nll_after
                decision_reports["calibrated_cost_min"] = calibrated_report
            if "calibrated_threshold" in decision_modes and num_classes != 2:
                print("Skipping calibrated_threshold: binary thresholding is only valid for two classes.")
                metrics.setdefault("eval_skipped_decision_modes", {})["calibrated_threshold"] = (
                    "binary_only_for_multiclass"
                )
            elif "calibrated_threshold" in decision_modes:
                threshold_result = fit_binary_threshold_np(
                    calibration_probs,
                    calibration_predictions.label_ids,
                    cost_matrix_cfg,
                    target_class_id=target_class_id,
                    target_recall_floor=float(getattr(script_args, "target_recall_floor", 0.95)),
                    accuracy_floor=float(getattr(script_args, "accuracy_floor", 0.80)),
                    selection_accuracy_metric=getattr(script_args, "selection_accuracy_metric", "accuracy"),
                    target_class_ids=target_class_ids,
                    class_names=configured_class_names,
                    critical_pairs=critical_pairs,
                    include_quadratic_weighted_kappa=include_qwk,
                )
                threshold_pred = binary_threshold_predictions(
                    calibrated_probs,
                    threshold_result.threshold,
                    target_class_id=target_class_id,
                )
                threshold_report = classification_metrics(
                    y_true,
                    threshold_pred,
                    calibrated_probs,
                    cost_matrix_cfg,
                    target_class_id=target_class_id,
                    target_recall_floor=float(getattr(script_args, "target_recall_floor", 0.95)),
                    accuracy_floor=float(getattr(script_args, "accuracy_floor", 0.80)),
                    selection_accuracy_metric=getattr(script_args, "selection_accuracy_metric", "accuracy"),
                    target_class_ids=target_class_ids,
                    class_names=configured_class_names,
                    critical_pairs=critical_pairs,
                    include_quadratic_weighted_kappa=include_qwk,
                )
                threshold_report["temperature"] = calibration_result.temperature
                threshold_report["threshold"] = threshold_result.threshold
                threshold_report["calibration_selection_score"] = threshold_result.selection_score
                threshold_report["calibration_target_recall"] = threshold_result.target_recall
                threshold_report["calibration_accuracy"] = threshold_result.accuracy
                threshold_report["calibration_normalized_atc"] = threshold_result.normalized_atc
                threshold_report["calibration_nll_before"] = calibration_result.nll_before
                threshold_report["calibration_nll_after"] = calibration_result.nll_after
                decision_reports["calibrated_threshold"] = threshold_report
    else:
        expected_cost = 0.0

    # Add new metrics to the metrics dict (with eval_ prefix for HF Trainer convention)
    metrics["eval_balanced_accuracy"] = balanced_acc
    metrics["eval_macro_f1"] = float(np.mean(list(class_metrics["f1_score"].values()))) if class_metrics["f1_score"] else 0.0
    metrics["eval_kappa"] = kappa_val
    metrics["eval_quadratic_weighted_kappa"] = qwk_val
    metrics["eval_auc"] = auc_val
    if cost_matrix_cfg is not None:
        primary_report = decision_reports[primary_decision_mode]
        metrics["eval_primary_decision_mode"] = primary_decision_mode
        metrics["eval_atc"] = primary_report["atc"]
        metrics["eval_normalized_atc"] = primary_report["normalized_atc"]
        metrics["eval_selection_score"] = primary_report["selection_score"]
        metrics["eval_selection_accuracy_value"] = primary_report["selection_accuracy_value"]
        metrics["eval_target_recall"] = primary_report["target_recall"]
        metrics["eval_target_recall_macro"] = primary_report["target_recall_macro"]
        metrics["eval_cost_matrix_sha256"] = primary_report["cost_matrix_sha256"]
        metrics["eval_target_fnr"] = primary_report["target_fnr"]
        metrics["eval_nll"] = primary_report["nll"]
        metrics["eval_brier"] = primary_report["brier"]
        metrics["eval_ece"] = primary_report["ece"]
        metrics["eval_auroc"] = primary_report["auroc"]
        metrics["eval_auprc"] = primary_report["auprc"]
        metrics["eval_decision_reports"] = decision_reports
    metrics["eval_recall_class0"] = recall_class0
    metrics["eval_expected_cost"] = expected_cost
    metrics["eval_prevalence_class0"] = prevalence_class0

    # Save all metrics to files
    print("Saving metrics to files...")
    save_metrics_to_file(
        metrics,
        class_metrics,
        class_counts,
        results_dir,
        f"_{script_args.loss_function}",
        validation_report=validation_report,
    )

    # Use provided class names or try to get from dataset, or fall back to generic names
    if class_names is not None:
        # Use provided class names (from local folder dataset)
        final_class_names = class_names
        print(f"Using provided class names: {final_class_names}")
    else:
        # Try to get class names from dataset if available
        try:
            if dataset_name:
                dataset = load_dataset(dataset_name)
                if hasattr(dataset["train"].features["label"], "names"):
                    final_class_names = dataset["train"].features["label"].names
                else:
                    final_class_names = [f"Class {i}" for i in range(num_classes)]
            else:
                final_class_names = [f"Class {i}" for i in range(num_classes)]
        except Exception:
            final_class_names = [f"Class {i}" for i in range(num_classes)]
        print(f"Using class names: {final_class_names}")

    if config_flag_enabled(getattr(script_args, "skip_visualizations", False)):
        print("Skipping confusion matrix visualizations by configuration.")
    else:
        # Create professional confusion matrix visualizations
        print("Creating confusion matrix visualizations...")
        create_confusion_matrix_visualization(
            confusion_matrix,
            class_names=final_class_names,
            output_dir=results_dir,
            filename_suffix=f"_{script_args.loss_function}",
        )

    # Print summary to console (minimal output)
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETED - RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Output Directory: {results_dir}")
    print(f"Overall Accuracy: {metrics.get('eval_accuracy', 0.0):.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Overall F1 Score: {metrics.get('eval_f1', 0.0):.4f}")
    print(f"Kappa: {kappa_val:.4f}")
    print(f"AUC: {auc_val:.4f}")
    print(f"Expected Cost: {expected_cost:.4f}")
    print(f"Loss Function Used: {script_args.loss_function}")
    print(f"Total Test Samples: {total_samples}")
    if class_names:
        print(f"Classes: {', '.join(final_class_names)}")
    if config_flag_enabled(getattr(script_args, "skip_visualizations", False)):
        print(f"\nDetailed metrics saved to: {results_dir}")
    else:
        print(f"\nDetailed metrics and visualizations saved to: {results_dir}")
    print(f"{'=' * 60}")

    return results_dir


def preprocess_local_folder_dataset(folder_path, model_name, test_size=0.1, val_size=0.1, random_state=42):
    """
    Load images from a local folder structure where subfolders represent classes.

    Args:
        folder_path (str): Path to the root folder containing class subfolders
        model_name (str): Model name for the image processor
        test_size (float): Proportion of data to use for testing (default: 0.1)
        val_size (float): Proportion of remaining data to use for validation (default: 0.1)
        random_state (int): Random seed for reproducible splits

    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names) - preprocessed datasets and class names
    """
    print(f"Loading dataset from: {folder_path}")

    # Check if path exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Dataset path does not exist: {folder_path}")

    # Get all class folders (subdirectories)
    class_folders = [
        d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and not d.startswith(".")
    ]
    class_folders.sort()  # Ensure consistent ordering

    if not class_folders:
        raise ValueError(f"No class folders found in {folder_path}")

    print(f"Found {len(class_folders)} classes: {class_folders}")

    # Create class to index mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    # Load all images and labels
    image_paths = []
    labels = []

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for class_name in class_folders:
        class_folder = os.path.join(folder_path, class_name)
        class_idx = class_to_idx[class_name]

        # Get all image files in this class folder
        class_images = []
        for file in os.listdir(class_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(class_folder, file)
                class_images.append(image_path)

        print(f"Class '{class_name}' (index {class_idx}): {len(class_images)} images")

        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    print(f"Total images loaded: {len(image_paths)}")

    # Split the data: first split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Then split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Build HuggingFace Dataset objects
    train_dataset = Dataset.from_dict(_create_dataset_dict(X_train, y_train))
    val_dataset = Dataset.from_dict(_create_dataset_dict(X_val, y_val))
    test_dataset = Dataset.from_dict(_create_dataset_dict(X_test, y_test))

    # Apply image preprocessing transforms
    image_preprocessor = _make_image_preprocessor(model_name)
    train_dataset.set_transform(image_preprocessor.preprocess_train)
    val_dataset.set_transform(image_preprocessor.preprocess_val)
    test_dataset.set_transform(image_preprocessor.preprocess_test)

    class_names = [idx_to_class[i] for i in range(len(class_folders))]
    print(f"Dataset created — classes: {idx_to_class}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, class_names


def preprocess_local_csv_dataset(folder_path, model_name, test_size=0.1, val_size=0.1, random_state=42):
    """
    Load images from a local folder with CSV files containing image paths and labels.

    Expected structure:
    - folder_path/train_dataset.csv (contains image_path and binary_label columns)
    - folder_path/test_dataset.csv (contains image_path and binary_label columns)
    - Images referenced by relative paths in the CSV files (e.g., "jpeg/image1.jpg")

    CSV columns expected:
    - image_path: Relative path to image file from the base folder_path
    - binary_label: Class label for the image

    Args:
        folder_path (str): Path to the root folder containing CSV files and images
        model_name (str): Model name for the image processor
        test_size (float): Proportion of data to use for testing (default: 0.1)
        val_size (float): Proportion of remaining data to use for validation (default: 0.1)
        random_state (int): Random seed for reproducible splits

    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names) - preprocessed datasets and class names
    """
    print(f"Loading CSV-based dataset from: {folder_path}")

    # Check if path exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Dataset path does not exist: {folder_path}")

    # Check for required files and folders
    train_csv_path = os.path.join(folder_path, "train_dataset.csv")
    test_csv_path = os.path.join(folder_path, "test_dataset.csv")

    if not os.path.exists(train_csv_path):
        raise ValueError(f"Train CSV file not found: {train_csv_path}")
    if not os.path.exists(test_csv_path):
        raise ValueError(f"Test CSV file not found: {test_csv_path}")

    # Load CSV files
    try:
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV files: {e}") from e

    # Validate CSV structure
    required_columns = ["image_path", "binary_label"]

    for col in required_columns:
        if col not in train_df.columns:
            raise ValueError(f"Train CSV missing required column: {col}")
        if col not in test_df.columns:
            raise ValueError(f"Test CSV missing required column: {col}")

    # Get unique class names and create mappings
    all_labels = pd.concat([train_df["binary_label"], test_df["binary_label"]]).unique()
    class_names = sorted(all_labels)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}

    print(f"Found {len(class_names)} classes: {class_names}")

    # Function to load and validate images from CSV
    def load_images_from_csv(df, base_folder, class_to_idx):
        image_paths = []
        labels = []
        skipped_images = 0

        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        for _, row in df.iterrows():
            # The image_path already includes the subfolder (e.g., "jpeg/filename.jpg")
            relative_path = row["image_path"]
            label_name = row["binary_label"]

            # Construct full image path by joining base folder with the relative path
            image_path = os.path.join(base_folder, relative_path)

            # Check if image file exists and has valid extension
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                skipped_images += 1
                continue

            # Extract filename from path for extension check
            filename = os.path.basename(relative_path)
            if not any(filename.lower().endswith(ext) for ext in image_extensions):
                print(f"Warning: Unsupported image format: {filename}")
                skipped_images += 1
                continue

            # Check if label is valid
            if label_name not in class_to_idx:
                print(f"Warning: Unknown label '{label_name}' for image {relative_path}")
                skipped_images += 1
                continue

            image_paths.append(image_path)
            labels.append(class_to_idx[label_name])

        if skipped_images > 0:
            print(f"Skipped {skipped_images} images due to missing files or invalid labels")

        return image_paths, labels

    # Load images and labels from CSV files (note: no longer using images_folder, using folder_path directly)
    train_image_paths, train_labels = load_images_from_csv(train_df, folder_path, class_to_idx)
    test_image_paths, test_labels = load_images_from_csv(test_df, folder_path, class_to_idx)

    print(f"Loaded {len(train_image_paths)} training images")
    print(f"Loaded {len(test_image_paths)} test images")

    # Split training data into train and validation sets
    if len(train_image_paths) == 0:
        raise ValueError("No valid training images found")

    X_train, X_val, y_train, y_val = train_test_split(
        train_image_paths, train_labels, test_size=val_size, random_state=random_state, stratify=train_labels
    )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(test_image_paths)}")

    # Build HuggingFace Dataset objects
    train_dataset = Dataset.from_dict(_create_dataset_dict(X_train, y_train))
    val_dataset = Dataset.from_dict(_create_dataset_dict(X_val, y_val))
    test_dataset = Dataset.from_dict(_create_dataset_dict(test_image_paths, test_labels))

    # Apply image preprocessing transforms
    image_preprocessor = _make_image_preprocessor(model_name)
    train_dataset.set_transform(image_preprocessor.preprocess_train)
    val_dataset.set_transform(image_preprocessor.preprocess_val)
    test_dataset.set_transform(image_preprocessor.preprocess_test)

    print(f"CSV dataset created — classes: {idx_to_class}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, class_names


def preprocess_manifest_dataset(folder_path, model_name, script_args=None):
    """Load prepared split CSVs produced by ``nicme-prepare-data``.

    Expected files are ``train.csv``, ``validation.csv``, optional
    ``calibration.csv``, and ``test.csv``. Each CSV must contain ``image_path``
    and ``label`` columns; ``label_name`` is used for class names when present.
    """
    base = Path(folder_path)
    required = {"train": base / "train.csv", "validation": base / "validation.csv", "test": base / "test.csv"}
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise ValueError(f"Prepared manifest split is missing required files: {missing}")

    frames = {name: pd.read_csv(path) for name, path in required.items()}
    calibration_path = base / "calibration.csv"
    if calibration_path.exists():
        frames["calibration"] = pd.read_csv(calibration_path)

    for name, frame in frames.items():
        for column in ("image_path", "label"):
            if column not in frame.columns:
                raise ValueError(f"{name}.csv missing required column {column!r}")

    all_labels = pd.concat([frame["label"] for frame in frames.values()]).astype(int)
    num_classes = int(all_labels.max()) + 1
    class_names = [f"class_{idx}" for idx in range(num_classes)]
    if all("label_name" in frame.columns for frame in frames.values()):
        names = (
            pd.concat([frame[["label", "label_name"]] for frame in frames.values()])
            .drop_duplicates()
            .sort_values("label")
        )
        if len(names) == num_classes:
            class_names = names["label_name"].tolist()

    def to_dataset(frame):
        dataset = Dataset.from_dict(
            {
                "image_path": frame["image_path"].astype(str).tolist(),
                "label": frame["label"].astype(int).tolist(),
            }
        )
        return dataset

    image_preprocessor = _make_image_preprocessor(model_name, script_args=script_args)
    train_dataset = to_dataset(frames["train"])
    val_dataset = to_dataset(frames["validation"])
    test_dataset = to_dataset(frames["test"])
    train_dataset.set_transform(image_preprocessor.preprocess_train)
    val_dataset.set_transform(image_preprocessor.preprocess_val)
    test_dataset.set_transform(image_preprocessor.preprocess_test)

    calibration_dataset = None
    if "calibration" in frames:
        calibration_dataset = to_dataset(frames["calibration"])
        calibration_dataset.set_transform(image_preprocessor.preprocess_val)

    print(
        "Prepared manifest loaded — "
        f"Train={len(train_dataset)}, Val={len(val_dataset)}, "
        f"Calibration={len(calibration_dataset) if calibration_dataset is not None else 0}, "
        f"Test={len(test_dataset)}"
    )
    return train_dataset, val_dataset, calibration_dataset, test_dataset, class_names


def save_run_configuration(script_args, output_dir, dataset_name=None):
    """
    Save the input configuration for the current run to a JSON file.

    Args:
        script_args: Script arguments containing all configuration parameters
        output_dir: Directory to save the configuration file
        dataset_name: Optional dataset name for better identification
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create timestamp for the configuration
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine dataset path/location based on dataset_host
    dataset_location = None
    if script_args.dataset_host == "huggingface":
        dataset_location = script_args.dataset
    elif script_args.dataset_host == "kaggle":
        dataset_location = f"kaggle:{script_args.dataset}"
        if script_args.local_dataset_name:
            dataset_location += f"/{script_args.local_dataset_name}"
    elif script_args.dataset_host == "local_folder":
        dataset_location = script_args.local_folder_path
    else:
        dataset_location = script_args.dataset if script_args.dataset else "unknown"

    # Prepare configuration dictionary
    config = {
        "run_info": {"timestamp": timestamp, "output_directory": output_dir},
        "model_configuration": {
            "model": script_args.model,
            "model_backend": getattr(script_args, "model_backend", "custom"),
            "model_type": getattr(script_args, "model_type", None),
            "weights": script_args.weights,
            "num_labels": script_args.num_labels,
            "peft_enabled": getattr(script_args, "peft_enabled", False),
        },
        "dataset_configuration": {
            "dataset_profile": getattr(script_args, "dataset_profile", None),
            "dataset_host": script_args.dataset_host,
            "dataset": script_args.dataset,
            "dataset_location": dataset_location,
            "local_dataset_name": script_args.local_dataset_name,
            "local_folder_path": script_args.local_folder_path,
            "local_dataset_format": script_args.local_dataset_format,
        },
        "training_parameters": {
            "batch_size": script_args.batch_size,
            "learning_rate": script_args.learning_rate,
            "num_train_epochs": script_args.num_train_epochs,
            "loss_function": script_args.loss_function,
            "cs_lambda": getattr(script_args, "cs_lambda", 0.0),
            "cs_warmup_epochs": getattr(script_args, "cs_warmup_epochs", 0),
            "nicme_logit_cost_scale": getattr(script_args, "nicme_logit_cost_scale", 1.0),
            "prediction_mode": getattr(script_args, "prediction_mode", None),
            "initial_checkpoint_path": getattr(script_args, "initial_checkpoint_path", None),
            "cb_beta": getattr(script_args, "cb_beta", 0.9999),
            "focal_gamma": getattr(script_args, "focal_gamma", 2.0),
            "ldam_max_m": getattr(script_args, "ldam_max_m", 0.5),
            "ldam_scale": getattr(script_args, "ldam_scale", 30.0),
            "ldam_drw_start_epoch": getattr(script_args, "ldam_drw_start_epoch", 16),
            "ap_alpha": getattr(script_args, "ap_alpha", 5.0),
            "csada_lambda": getattr(script_args, "csada_lambda", 2.0),
            "csada_attack_steps": getattr(script_args, "csada_attack_steps", 5),
            "csada_epsilon": getattr(script_args, "csada_epsilon", 2.0 / 255.0),
            "csada_step_size": getattr(script_args, "csada_step_size", 1.0 / 255.0),
            "csada_cost_temperature": getattr(script_args, "csada_cost_temperature", 3.0),
            "seed": getattr(script_args, "seed", None),
            "augmentations": {
                "train_random_resize_crop": getattr(script_args, "train_random_resize_crop", True),
                "train_random_resized_crop_scale_min": getattr(
                    script_args, "train_random_resized_crop_scale_min", 0.08
                ),
                "train_random_resized_crop_scale_max": getattr(
                    script_args, "train_random_resized_crop_scale_max", 1.0
                ),
                "train_horizontal_flip": getattr(script_args, "train_horizontal_flip", True),
                "train_color_jitter": getattr(script_args, "train_color_jitter", False),
                "train_random_quarter_turns": getattr(script_args, "train_random_quarter_turns", False),
            },
        },
        "cost_matrix_configuration": {
            "cost_matrix": script_args.cost_matrix,
            "cost_matrix_source": getattr(script_args, "cost_matrix_source", None),
            "cost_matrix_sha256": getattr(script_args, "cost_matrix_sha256", None)
            or (cost_matrix_sha256(script_args.cost_matrix) if script_args.cost_matrix is not None else None),
            "cost_matrix_convention": "C[true_label][predicted_label]",
            "target_recall_class": getattr(script_args, "target_recall_class", None),
            "target_recall_classes": getattr(script_args, "target_recall_classes", None),
            "secondary_target_recall_classes": getattr(script_args, "secondary_target_recall_classes", None),
            "target_recall_floor": getattr(script_args, "target_recall_floor", None),
            "accuracy_floor": getattr(script_args, "accuracy_floor", None),
            "selection_accuracy_metric": getattr(script_args, "selection_accuracy_metric", "accuracy"),
            "metric_profile": getattr(script_args, "metric_profile", None),
            "critical_pairs": getattr(script_args, "critical_pairs", None),
            "sweep_mode": script_args.sweep_mode,
            "sweep_cost_value": script_args.sweep_cost_value,
            "sweep_matrix_row": script_args.sweep_matrix_row,
            "sweep_matrix_col": script_args.sweep_matrix_col,
        },
        "other_settings": {
            "wandb_enabled": config_flag_enabled(script_args.wandb),
            "push_to_hub": config_flag_enabled(script_args.push_to_hub),
        },
    }

    # Save configuration to JSON file
    config_path = os.path.join(output_dir, f"run_configuration_{timestamp}.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Run configuration saved to: {config_path}")
    return config_path
