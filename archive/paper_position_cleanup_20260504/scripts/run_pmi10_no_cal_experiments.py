"""Focused PMI-10 no-calibration experiment runner.

This runner is intentionally separate from the older MC stop runner. It plans
and optionally executes a single-seed PMI-10 track with argmax-only decisions,
full fine-tuning, fixed dataset costs, and no calibration split.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from nicme.dataset_profiles import get_profile

DATASET = "pmi_pills_10_no_cal"
VARIANT = "natural"
CONVNEXT_BASE_MODEL = "convnext_base.fb_in22k_ft_in1k"
DEFAULT_OUTPUT_ROOT = Path("results/pmi10_no_cal_convnext_base_20260501")
V3_OUTPUT_ROOT = Path("results/pmi10_v3_balanced_convnext_base_20260502")
LEGACY_OUTPUT_ROOT = Path("results/pmi10_no_cal_20260501")
DEFAULT_SPLIT_DIR = Path("data/prepared/pmi_pills_10_no_cal/splits/natural")
BALANCED_SPLIT_DIR = Path("data/prepared/pmi_pills_10_no_cal/splits/balanced")
DEFAULT_HPO_PROFILE = "default"
CONVNEXT_BASE_FAST_PROFILE = "convnext_base_fast"
CONVNEXT_BASE_V3_BALANCED_PROFILE = "convnext_base_v3_balanced"
BAD_TRIAL_SCORE = -1e9

MODEL_IDS = (
    "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_small.fb_in22k_ft_in1k",
    "convnext_base.fb_in22k_ft_in1k",
    "convnext_tiny.dinov3_lvd1689m",
    "convnext_small.dinov3_lvd1689m",
    "vit_small_patch16_dinov3.lvd1689m",
    "vit_base_patch16_dinov3.lvd1689m",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.mae",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
    "efficientnet_b3.ra2_in1k",
    "efficientnet_b5.sw_in12k_ft_in1k",
    "convnextv2_tiny.fcmae_ft_in1k",
    "hiera_base_224.mae_in1k_ft_in1k",
)

SMOKE_MODEL_IDS = (
    "convnext_tiny.fb_in22k_ft_in1k",
    "vit_small_patch16_dinov3.lvd1689m",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
)

DEFAULT_LOSSES = ("ce", "nicme_hybrid")

LOSS_PRESETS = {
    "ce": {
        "loss_function": "cross_entropy",
        "cs_lambda": 0.0,
        "cs_warmup_epochs": 0,
        "nicme_logit_cost_scale": 1.0,
    },
    "nicme_hybrid": {
        "loss_function": "nicme_hybrid",
        "cs_lambda": 0.10,
        "cs_warmup_epochs": 1,
        "nicme_logit_cost_scale": 0.20,
    },
    "nicme_v3_logit_adjustment": {
        "loss_function": "nicme_v3_logit_adjustment",
        "cs_lambda": 0.0,
        "cs_warmup_epochs": 0,
        "nicme_logit_cost_scale": 0.20,
    },
    "nicme_v3_hybrid": {
        "loss_function": "nicme_v3_hybrid",
        "cs_lambda": 0.10,
        "cs_warmup_epochs": 1,
        "nicme_logit_cost_scale": 0.20,
    },
}

HPO_PROFILES: dict[str, dict[str, Any]] = {
    DEFAULT_HPO_PROFILE: {
        "description": "Original broad HPO profile with model-dependent image-size choices.",
        "fixed": {},
        "models": None,
        "epochs": 30,
    },
    CONVNEXT_BASE_FAST_PROFILE: {
        "description": "Narrow ConvNeXt-base-only PMI-10 recovery HPO profile.",
        "models": [CONVNEXT_BASE_MODEL],
        "epochs": 24,
        "fixed": {
            "image_size": 224,
            "early_stopping_patience": 4,
            "batch_size": 16,
            "gradient_accumulation_steps": 4,
            "lr_scheduler_type": "cosine",
            "train_random_resize_crop": True,
            "train_horizontal_flip": True,
        },
        "search": {
            "learning_rate": {"type": "float", "low": 1.5e-5, "high": 1.0e-4, "log": True},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.05},
            "warmup_ratio": {"type": "float", "low": 0.02, "high": 0.10},
            "train_color_jitter": {"type": "categorical", "choices": [False, True]},
        },
        "loss_search": {
            "nicme_hybrid": {
                "nicme_logit_cost_scale": {"type": "float", "low": 0.08, "high": 0.50, "log": True},
                "cs_lambda": {"type": "float", "low": 0.02, "high": 0.25},
                "cs_warmup_epochs": {"type": "categorical", "choices": [1, 2, 3]},
            }
        },
    },
    CONVNEXT_BASE_V3_BALANCED_PROFILE: {
        "description": "Balanced PMI-10 ConvNeXt-base NICME V3 HPO profile.",
        "models": [CONVNEXT_BASE_MODEL],
        "epochs": 32,
        "fixed": {
            "image_size": 224,
            "early_stopping_patience": 5,
            "batch_size": 16,
            "gradient_accumulation_steps": 4,
            "lr_scheduler_type": "cosine",
            "train_random_resize_crop": True,
            "train_random_resized_crop_scale_min": 0.8,
            "train_random_resized_crop_scale_max": 1.0,
            "train_horizontal_flip": False,
            "train_random_quarter_turns": True,
        },
        "search": {
            "learning_rate": {"type": "float", "low": 1.5e-5, "high": 9.0e-5, "log": True},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.05},
            "warmup_ratio": {"type": "float", "low": 0.02, "high": 0.12},
            "train_color_jitter": {"type": "categorical", "choices": [False, True]},
        },
        "loss_search": {
            "nicme_v3_hybrid": {
                "nicme_logit_cost_scale": {"type": "float", "low": 0.05, "high": 0.70, "log": True},
                "cs_lambda": {"type": "float", "low": 0.0, "high": 0.25},
                "cs_warmup_epochs": {"type": "categorical", "choices": [0, 1, 2, 3, 5]},
            }
        },
    },
}

LEDGER_FIELDS = [
    "run_id",
    "phase",
    "backbone",
    "loss",
    "seed",
    "status",
    "config_hash",
    "config_path",
    "metric_path",
    "target_recall_min",
    "target_recall_macro",
    "normalized_atc",
    "atc",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "critical_pair_error_count",
    "cost_matrix_sha256",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "stdout_log",
    "stderr_log",
    "returncode",
    "failure_reason",
]


def parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def timm_data_config(model_id: str) -> dict[str, Any]:
    try:
        from timm.data import resolve_model_data_config
    except ImportError:
        return {}
    try:
        return dict(resolve_model_data_config(model_id))
    except Exception:
        return {}


def image_size_choices(model_id: str) -> list[int]:
    lowered = model_id.lower()
    if "swin" in lowered or "vit_" in lowered or "hiera" in lowered:
        return [224]
    if "dinov3" in lowered:
        return [224, 256]
    return [224, 256, 320]


def hpo_profile_name(args: argparse.Namespace) -> str:
    name = str(getattr(args, "hpo_profile", DEFAULT_HPO_PROFILE) or DEFAULT_HPO_PROFILE)
    if name not in HPO_PROFILES:
        raise ValueError(f"Unknown HPO profile {name!r}. Available: {sorted(HPO_PROFILES)}")
    return name


def validate_static_default_hpo_space(models: list[str], profile_name: str) -> None:
    if profile_name != DEFAULT_HPO_PROFILE:
        return
    image_spaces = {tuple(image_size_choices(model)) for model in models}
    if len(image_spaces) > 1:
        raise ValueError(
            "Default multi-model HPO would create a dynamic Optuna categorical space for image_size. "
            "Use --hpo-profile convnext_base_fast or pass models with the same image-size choices."
        )


def hpo_models(args: argparse.Namespace) -> list[str]:
    profile_name = hpo_profile_name(args)
    profile_models = HPO_PROFILES[profile_name].get("models")
    if profile_models:
        models = list(profile_models)
    elif args.models:
        models = parse_csv(args.models, MODEL_IDS)
    else:
        selected_path = Path(args.output_root) / "screen" / "selected_backbones.json"
        if selected_path.exists():
            models = json.loads(selected_path.read_text())["selected_backbones"]
            if not models:
                raise RuntimeError(f"Selected backbone list is empty: {selected_path}")
        else:
            models = list(MODEL_IDS[:3])
    unknown = [model for model in models if model not in MODEL_IDS]
    if unknown:
        raise ValueError(f"Unknown PMI-10 backbone(s) {unknown!r}. Available: {list(MODEL_IDS)}")
    validate_static_default_hpo_space(models, profile_name)
    return models


def suggest_from_spec(trial: Any, name: str, spec: dict[str, Any]) -> Any:
    if spec["type"] == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=bool(spec.get("log", False)))
    if spec["type"] == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"Unsupported HPO search spec for {name!r}: {spec!r}")


def hpo_search_space(loss: str, profile_name: str) -> dict[str, Any]:
    if profile_name in {CONVNEXT_BASE_FAST_PROFILE, CONVNEXT_BASE_V3_BALANCED_PROFILE}:
        profile = HPO_PROFILES[profile_name]
        search = dict(profile["search"])
        search.update(profile.get("loss_search", {}).get(loss, {}))
        return {"fixed": dict(profile["fixed"]), "search": search}
    return {
        "fixed": {},
        "search": {
            "learning_rate": {"type": "float", "low": 1e-6, "high": 5e-4, "log": True},
            "weight_decay": {"type": "float", "low": 0.0, "high": 0.20},
            "warmup_ratio": {"type": "float", "low": 0.0, "high": 0.20},
            "lr_scheduler_type": {"type": "categorical", "choices": ["cosine", "linear"]},
            "batch_size": {"type": "categorical", "choices": [4, 8, 16]},
            "gradient_accumulation_steps": {"type": "categorical", "choices": [2, 4, 8]},
            "train_random_resize_crop": {"type": "categorical", "choices": [True, False]},
            "train_horizontal_flip": {"type": "categorical", "choices": [True, False]},
            "train_color_jitter": {"type": "categorical", "choices": [False, True]},
            **(
                {
                    "nicme_logit_cost_scale": {"type": "float", "low": 0.05, "high": 1.0, "log": True},
                    "cs_lambda": {"type": "float", "low": 0.0, "high": 0.50},
                    "cs_warmup_epochs": {"type": "categorical", "choices": [0, 1, 2, 3]},
                }
                if loss == "nicme_hybrid"
                else {}
            ),
        },
    }


def hpo_epochs(args: argparse.Namespace) -> int:
    if args.epochs is not None:
        return int(args.epochs)
    return int(HPO_PROFILES[hpo_profile_name(args)].get("epochs", 30))


def variant_for_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "variant", VARIANT) or VARIANT)


def selection_for_variant(profile: Any, variant: str) -> Any:
    return profile.balanced_selection if variant == "balanced" else profile.natural_selection


def role_for_loss(loss: str) -> str:
    if loss == "nicme_v3_hybrid":
        return "nicme_v3_best"
    if loss == "nicme_hybrid":
        return "nicme_best"
    return "ce_tuned"


def infer_loss_key(cfg: dict[str, Any]) -> str:
    if cfg.get("pmi10_loss"):
        return str(cfg["pmi10_loss"])
    loss_function = str(cfg.get("loss_function", "cross_entropy"))
    if loss_function in {"cross_entropy", "ce"}:
        return "ce"
    if loss_function == "nicme_hybrid":
        return "nicme_hybrid"
    if loss_function == "nicme_v3_hybrid":
        return "nicme_v3_hybrid"
    if loss_function == "nicme_v3_logit_adjustment":
        return "nicme_v3_logit_adjustment"
    return slugify(loss_function)


def infer_backbone(cfg: dict[str, Any]) -> str:
    if cfg.get("pmi10_backbone"):
        return str(cfg["pmi10_backbone"])
    model = str(cfg.get("model", ""))
    if model.startswith("timm/"):
        return model.split("/", 1)[1]
    return model


def base_config(
    phase: str,
    backbone: str,
    loss: str,
    epochs: int,
    seed: int,
    split_dir: Path,
    output_root: Path,
    overrides: dict[str, Any] | None = None,
    variant: str = VARIANT,
) -> dict[str, Any]:
    profile = get_profile(DATASET)
    selection = selection_for_variant(profile, variant)
    data_config = timm_data_config(backbone)
    model_slug = slugify(backbone)
    cfg: dict[str, Any] = {
        "dataset_profile": profile.name,
        "variant": variant,
        "dataset": f"{profile.name}_{variant}",
        "dataset_host": "local_folder",
        "local_dataset_format": "manifest",
        "local_folder_path": str(split_dir),
        "class_names": list(profile.class_names),
        "num_labels": profile.num_classes,
        "target_recall_class": profile.target_recall_class,
        "target_recall_classes": list(profile.cared_classes),
        "secondary_target_recall_classes": list(profile.secondary_target_recall_classes),
        "target_recall_floor": selection.target_recall_floor,
        "accuracy_floor": selection.accuracy_floor,
        "selection_accuracy_metric": selection.accuracy_metric,
        "cost_matrix": [list(row) for row in profile.cost_matrix],
        "cost_matrix_source": profile.cost_matrix_source,
        "cost_matrix_sha256": profile.cost_matrix_sha256,
        "metric_profile": profile.metric_profile,
        "critical_pairs": [
            {"true": true_name, "pred": pred_name, "cost": cost}
            for true_name, pred_name, cost in profile.critical_pairs
        ],
        "model": f"timm/{backbone}",
        "model_backend": "timm",
        "model_type": f"pmi10_{model_slug}",
        "peft_enabled": False,
        "timm_pretrained": True,
        "num_train_epochs": epochs,
        "early_stopping_patience": 5,
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "wandb": False,
        "push_to_hub": False,
        "calibration_enabled": False,
        "decision_modes": "argmax",
        "model_selection_metric": "target_recall",
        "save_total_limit": 1,
        "skip_visualizations": phase not in {"final", "train_plus_val_final"},
        "seed": seed,
        "output_dir": f"{phase}_{model_slug}_{loss}_s{seed}",
        "pmi10_phase": phase,
        "pmi10_backbone": backbone,
        "pmi10_loss": loss,
        "pmi10_output_root": str(output_root),
        "timm_data_config": data_config,
    }
    cfg.update(LOSS_PRESETS[loss])
    if overrides:
        cfg.update(overrides)
    return cfg


def final_config_paths(args: argparse.Namespace) -> list[Path]:
    if getattr(args, "configs", None):
        return [Path(path) for path in parse_csv(args.configs, ())]
    output_root = Path(args.output_root)
    if getattr(args, "phase", "") == "tpt-stress":
        return [
            path
            for path in (
                output_root / "hpo-v3" / "best_config.json",
                output_root / "hpo-v2-matched" / "best_config.json",
            )
            if path.exists()
        ]
    if (output_root / "hpo-v3" / "best_config.json").exists():
        return [
            path
            for path in (
                output_root / "hpo-v3" / "best_config.json",
                output_root / "hpo-ce-matched" / "best_config.json",
                output_root / "hpo-ce" / "best_config.json",
                output_root / "hpo-v2-matched" / "best_config.json",
            )
            if path.exists()
        ]
    return [
        path
        for path in (
            output_root / "hpo-nicme" / "best_config.json",
            output_root / "hpo-ce-matched" / "best_config.json",
            output_root / "hpo-ce" / "best_config.json",
        )
        if path.exists()
    ]


def build_final_runs_from_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs = []
    profile = get_profile(DATASET)
    variant = variant_for_args(args)
    selection = selection_for_variant(profile, variant)
    for config_path in final_config_paths(args):
        cfg = json.loads(config_path.read_text())
        backbone = infer_backbone(cfg)
        loss = infer_loss_key(cfg)
        role = str(
            cfg.get("pmi10_run_role")
            or ("nicme_best" if loss == "nicme_hybrid" else slugify(config_path.parent.name or "ce"))
        )
        if args.epochs is not None:
            cfg["num_train_epochs"] = int(args.epochs)
        if args.phase == "tpt-stress":
            cfg["num_train_epochs"] = int(args.epochs or 64)
            cfg["early_stopping_patience"] = 0
            role = f"tpt_{role}"
        cfg.update(
            {
                "dataset_profile": DATASET,
                "variant": variant,
                "dataset": f"{DATASET}_{variant}",
                "local_folder_path": str(Path(args.split_dir)),
                "class_names": list(profile.class_names),
                "num_labels": profile.num_classes,
                "target_recall_class": profile.target_recall_class,
                "target_recall_classes": list(profile.cared_classes),
                "secondary_target_recall_classes": list(profile.secondary_target_recall_classes),
                "target_recall_floor": selection.target_recall_floor,
                "accuracy_floor": selection.accuracy_floor,
                "selection_accuracy_metric": selection.accuracy_metric,
                "cost_matrix": [list(row) for row in profile.cost_matrix],
                "calibration_enabled": False,
                "decision_modes": "argmax",
                "cost_matrix_source": "dataset",
                "cost_matrix_sha256": profile.cost_matrix_sha256,
                "metric_profile": profile.metric_profile,
                "critical_pairs": [
                    {"true": true_name, "pred": pred_name, "cost": cost}
                    for true_name, pred_name, cost in profile.critical_pairs
                ],
                "peft_enabled": False,
                "seed": int(args.seed),
                "pmi10_phase": args.phase,
                "pmi10_backbone": backbone,
                "pmi10_loss": loss,
                "pmi10_run_role": role,
                "pmi10_output_root": str(Path(args.output_root)),
                "output_dir": f"{args.phase}_{slugify(backbone)}_{role}_{loss}_s{int(args.seed)}_locked",
            }
        )
        runs.append(cfg)
    return runs


def build_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    variant = variant_for_args(args)
    if args.phase == "smoke":
        models = parse_csv(args.models, SMOKE_MODEL_IDS)
        losses = parse_csv(args.losses, DEFAULT_LOSSES)
        epochs = int(args.epochs or 2)
    elif args.phase == "screen":
        models = parse_csv(args.models, MODEL_IDS)
        losses = parse_csv(args.losses, DEFAULT_LOSSES)
        epochs = int(args.epochs or 20)
    elif args.phase in {"final", "train_plus_val_final", "tpt-stress"}:
        final_runs = build_final_runs_from_configs(args)
        if final_runs:
            return final_runs
        if not args.models:
            raise FileNotFoundError(
                "Final PMI-10 phases require --configs or completed HPO best_config.json files "
                "under hpo-nicme/ and hpo-ce/. Pass --models only for a manual final run."
            )
        models = parse_csv(args.models, ())
        losses = parse_csv(args.losses, tuple(LOSS_PRESETS))
        epochs = int(args.epochs or 30)
    else:
        return []

    runs = []
    for backbone in models:
        if backbone not in MODEL_IDS:
            raise ValueError(f"Unknown PMI-10 backbone {backbone!r}. Available: {list(MODEL_IDS)}")
        for loss in losses:
            if loss not in LOSS_PRESETS:
                raise ValueError(f"Unknown PMI-10 loss {loss!r}. Available: {list(LOSS_PRESETS)}")
            cfg = base_config(
                args.phase,
                backbone,
                loss,
                epochs,
                int(args.seed),
                Path(args.split_dir),
                Path(args.output_root),
                variant=variant,
            )
            runs.append(cfg)
    return runs


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def find_metric_path(cfg: dict[str, Any]) -> str:
    model_type = cfg["model_type"].lower()
    output_dir = cfg["output_dir"]
    loss_name = cfg["loss_function"]
    matches = sorted(
        Path("results").glob(f"{model_type}_test/{output_dir}_*/metrics_*_{loss_name}.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(matches[0]) if matches else ""


def load_report(metric_path: str) -> dict[str, Any]:
    if not metric_path:
        return {}
    with open(metric_path) as f:
        payload = json.load(f)
    validation_report = payload.get("validation_report", {}).get("argmax") or {}
    test_report = payload.get("decision_reports", {}).get("argmax") or {}
    return validation_report or test_report


def critical_pair_error_count(report: dict[str, Any]) -> int:
    return int(sum(pair.get("errors", 0) for pair in report.get("critical_pair_errors", {}).values()))


def report_objective_key(report: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
    return (
        float(report.get("target_recall_min", report.get("target_recall", 0.0))),
        -float(report.get("normalized_atc", 1.0)),
        float(report.get("target_recall_macro", 0.0)),
        float(report.get("balanced_accuracy", 0.0)),
        float(report.get("accuracy", 0.0)),
        -critical_pair_error_count(report),
    )


def report_objective_scalar(report: dict[str, Any]) -> float:
    target_recall, neg_natc, macro_recall, balanced_acc, accuracy, neg_critical_errors = report_objective_key(report)
    return (
        1000.0 * target_recall
        + 100.0 * neg_natc
        + 10.0 * macro_recall
        + balanced_acc
        + 0.1 * accuracy
        + 0.001 * neg_critical_errors
    )


def row_from_config(cfg: dict[str, Any], phase: str, run_id: int, status: str = "planned") -> dict[str, Any]:
    cfg_hash = config_hash(cfg)
    return {
        "run_id": run_id,
        "phase": phase,
        "backbone": cfg["pmi10_backbone"],
        "loss": cfg["pmi10_loss"],
        "seed": cfg["seed"],
        "status": status,
        "config_hash": cfg_hash,
        "config_path": "",
        "metric_path": "",
        "target_recall_min": "",
        "target_recall_macro": "",
        "normalized_atc": "",
        "atc": "",
        "balanced_accuracy": "",
        "accuracy": "",
        "macro_f1": "",
        "critical_pair_error_count": "",
        "cost_matrix_sha256": cfg["cost_matrix_sha256"],
        "start_time": "",
        "end_time": "",
        "elapsed_seconds": "",
        "stdout_log": "",
        "stderr_log": "",
        "returncode": "",
        "failure_reason": "",
    }


def update_row_with_report(row: dict[str, Any], report: dict[str, Any], metric_path: str) -> None:
    row["metric_path"] = metric_path
    row["target_recall_min"] = report.get("target_recall_min", report.get("target_recall", ""))
    row["target_recall_macro"] = report.get("target_recall_macro", "")
    row["normalized_atc"] = report.get("normalized_atc", "")
    row["atc"] = report.get("atc", "")
    row["balanced_accuracy"] = report.get("balanced_accuracy", "")
    row["accuracy"] = report.get("accuracy", "")
    row["macro_f1"] = report.get("macro_f1", "")
    row["critical_pair_error_count"] = critical_pair_error_count(report)
    row["cost_matrix_sha256"] = report.get("cost_matrix_sha256", row["cost_matrix_sha256"])


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEDGER_FIELDS})


def execute_run(cfg: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    cfg_hash = row["config_hash"]
    config_path = output_root / args.phase / "configs" / f"{int(row['run_id']):04d}_{cfg_hash}.json"
    stdout_log = output_root / args.phase / "logs" / f"{int(row['run_id']):04d}_{cfg_hash}.stdout.log"
    stderr_log = output_root / args.phase / "logs" / f"{int(row['run_id']):04d}_{cfg_hash}.stderr.log"
    write_json(config_path, cfg)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    row["config_path"] = str(config_path)
    row["stdout_log"] = str(stdout_log)
    row["stderr_log"] = str(stderr_log)
    row["start_time"] = dt.datetime.now().isoformat(timespec="seconds")
    start = time.time()
    command = [sys.executable, "scripts/train.py", "--config", str(config_path)]
    try:
        with open(stdout_log, "w") as out, open(stderr_log, "w") as err:
            completed = subprocess.run(
                command,
                stdout=out,
                stderr=err,
                text=True,
                timeout=int(args.per_run_timeout_minutes) * 60,
                check=False,
            )
        row["returncode"] = completed.returncode
        row["status"] = "completed" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            row["failure_reason"] = f"returncode_{completed.returncode}"
    except subprocess.TimeoutExpired:
        row["returncode"] = 124
        row["status"] = "timeout"
        row["failure_reason"] = "timeout"
    except Exception as exc:
        row["returncode"] = 1
        row["status"] = "failed"
        row["failure_reason"] = f"exception_{type(exc).__name__}: {exc}"
    row["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
    row["elapsed_seconds"] = f"{time.time() - start:.1f}"
    if row["status"] == "completed":
        metric_path = find_metric_path(cfg)
        update_row_with_report(row, load_report(metric_path), metric_path)
    return row


def failed_trial_row(
    args: argparse.Namespace,
    loss: str,
    run_id: int,
    backbone: str,
    reason: str,
    start_time: str | None = None,
) -> dict[str, Any]:
    cfg = base_config(
        args.phase,
        backbone,
        loss,
        hpo_epochs(args),
        int(args.seed),
        Path(args.split_dir),
        Path(args.output_root),
        overrides=HPO_PROFILES[hpo_profile_name(args)].get("fixed", {}),
        variant=variant_for_args(args),
    )
    row = row_from_config(cfg, args.phase, run_id, status="failed")
    row["start_time"] = start_time or dt.datetime.now().isoformat(timespec="seconds")
    row["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
    row["elapsed_seconds"] = "0.0"
    row["returncode"] = 1
    row["failure_reason"] = reason
    return row


def write_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs = build_runs(args)
    rows = []
    output_root = Path(args.output_root)
    for idx, cfg in enumerate(runs, start=1):
        row = row_from_config(cfg, args.phase, idx)
        cfg_hash = row["config_hash"]
        config_path = output_root / args.phase / "configs" / f"{idx:04d}_{cfg_hash}.json"
        row["config_path"] = str(config_path)
        write_json(config_path, cfg)
        rows.append(row)
    write_json(output_root / args.phase / "planned_runs.json", runs)
    write_ledger(output_root / args.phase / "run_ledger.csv", rows)
    return rows


def select_top_backbones(args: argparse.Namespace) -> list[str]:
    ledger_path = Path(args.output_root) / "screen" / "run_ledger.csv"
    if not ledger_path.exists():
        raise FileNotFoundError(f"Screen ledger not found: {ledger_path}")
    rows = list(csv.DictReader(ledger_path.open()))
    completed = [row for row in rows if row.get("status") == "completed" and row.get("loss") == "nicme_hybrid"]
    completed = [row for row in completed if row.get("metric_path")]
    if not completed:
        raise RuntimeError(
            "No completed NICME hybrid screen runs with metric paths were found. "
            "Run the screen phase with --execute before selecting top backbones."
        )
    ranked = sorted(
        completed,
        key=lambda row: report_objective_key(load_report(row["metric_path"])),
        reverse=True,
    )
    selected = []
    for row in ranked:
        if row["backbone"] not in selected:
            selected.append(row["backbone"])
        if len(selected) >= int(args.top_k):
            break
    if not selected:
        raise RuntimeError("Unable to select PMI-10 backbones from the completed screen ledger.")
    payload = {"top_k": int(args.top_k), "selected_backbones": selected, "source_ledger": str(ledger_path)}
    write_json(Path(args.output_root) / "screen" / "selected_backbones.json", payload)
    return selected


def build_hpo_trial_config(
    trial: Any,
    args: argparse.Namespace,
    loss: str,
    models: list[str],
    profile_name: str,
) -> dict[str, Any]:
    if len(models) == 1:
        backbone = models[0]
    else:
        backbone = trial.suggest_categorical("backbone", models)
    search_space = hpo_search_space(loss, profile_name)
    overrides = dict(search_space["fixed"])
    for name, spec in search_space["search"].items():
        overrides[name] = suggest_from_spec(trial, name, spec)
    if profile_name == DEFAULT_HPO_PROFILE:
        overrides["image_size"] = trial.suggest_categorical("image_size", image_size_choices(backbone))
    cfg = base_config(
        args.phase,
        backbone,
        loss,
        hpo_epochs(args),
        int(args.seed),
        Path(args.split_dir),
        Path(args.output_root),
        overrides=overrides,
        variant=variant_for_args(args),
    )
    cfg["output_dir"] = f"{cfg['output_dir']}_trial{trial.number:04d}"
    cfg["pmi10_hpo_profile"] = profile_name
    cfg["pmi10_run_role"] = role_for_loss(loss)
    return cfg


def run_hpo_trial(
    trial: Any,
    args: argparse.Namespace,
    loss: str,
    models: list[str],
    profile_name: str,
    trial_rows: list[dict[str, Any]],
    hpo_dir: Path,
) -> float:
    row: dict[str, Any] | None = None
    started = dt.datetime.now().isoformat(timespec="seconds")
    try:
        cfg = build_hpo_trial_config(trial, args, loss, models, profile_name)
        row = row_from_config(cfg, args.phase, trial.number + 1)
        row = execute_run(cfg, row, args)
        if row["status"] == "completed" and not row.get("metric_path"):
            row["status"] = "failed"
            row["failure_reason"] = "missing_metric_path"
        trial_rows.append(row)
        write_ledger(hpo_dir / "hpo_trials.csv", trial_rows)
        if row["status"] != "completed":
            return BAD_TRIAL_SCORE
        return report_objective_scalar(load_report(row["metric_path"]))
    except Exception as exc:
        backbone = models[0] if models else CONVNEXT_BASE_MODEL
        failed = row or failed_trial_row(
            args,
            loss,
            trial.number + 1,
            backbone,
            f"exception_{type(exc).__name__}: {exc}",
            start_time=started,
        )
        failed["status"] = "failed"
        failed["returncode"] = failed.get("returncode") or 1
        failed["failure_reason"] = f"exception_{type(exc).__name__}: {exc}"
        failed["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
        trial_rows.append(failed)
        write_ledger(hpo_dir / "hpo_trials.csv", trial_rows)
        return BAD_TRIAL_SCORE


def best_completed_hpo_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("metric_path")]
    if not completed:
        raise RuntimeError("HPO completed with zero successful trials; no best config can be selected.")
    reports = [(row, load_report(row["metric_path"])) for row in completed]
    best_recall = max(
        float(report.get("target_recall_min", report.get("target_recall", 0.0))) for _, report in reports
    )
    near_best = [
        (row, report)
        for row, report in reports
        if float(report.get("target_recall_min", report.get("target_recall", 0.0))) >= best_recall - 0.005
    ]
    return sorted(
        near_best,
        key=lambda item: (
            -float(item[1].get("normalized_atc", 1.0)),
            float(item[1].get("balanced_accuracy", 0.0)),
            float(item[1].get("target_recall_macro", 0.0)),
            float(item[1].get("accuracy", 0.0)),
            -critical_pair_error_count(item[1]),
        ),
        reverse=True,
    )[0][0]


def write_hpo_plan(args: argparse.Namespace, loss: str) -> None:
    profile_name = hpo_profile_name(args)
    models = hpo_models(args)
    output_root = Path(args.output_root)
    hpo_dir = output_root / args.phase
    fixed_overrides = hpo_search_space(loss, profile_name)["fixed"]
    example_config = base_config(
        args.phase,
        models[0],
        loss,
        hpo_epochs(args),
        int(args.seed),
        Path(args.split_dir),
        output_root,
        overrides=fixed_overrides,
        variant=variant_for_args(args),
    )
    example_config["pmi10_hpo_profile"] = profile_name
    example_config["pmi10_run_role"] = role_for_loss(loss)
    write_json(
        hpo_dir / "hpo_plan.json",
        {
            "phase": args.phase,
            "loss": loss,
            "trials": int(args.trials),
            "hpo_profile": profile_name,
            "selected_models": models,
            "epochs": hpo_epochs(args),
            "search_space": hpo_search_space(loss, profile_name),
            "example_config_path": str(hpo_dir / "example_config.json"),
        },
    )
    write_json(hpo_dir / "example_config.json", example_config)


def run_hpo(args: argparse.Namespace, loss: str) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required for PMI-10 HPO phases") from exc

    output_root = Path(args.output_root)
    hpo_dir = output_root / args.phase
    profile_name = hpo_profile_name(args)
    models = hpo_models(args)
    trial_rows: list[dict[str, Any]] = []

    def objective(trial):
        return run_hpo_trial(trial, args, loss, models, profile_name, trial_rows, hpo_dir)

    study = optuna.create_study(direction="maximize", study_name=f"pmi10_{args.phase}_{profile_name}")
    study.optimize(objective, n_trials=int(args.trials), catch=(Exception,))
    best_row = best_completed_hpo_row(trial_rows)
    best_config = json.loads(Path(best_row["config_path"]).read_text())
    best_config["pmi10_run_role"] = role_for_loss(loss)
    best_config["pmi10_hpo_profile"] = profile_name
    best_config["pmi10_output_root"] = str(output_root)
    write_json(hpo_dir / "best_config.json", best_config)
    write_json(
        hpo_dir / "best_trial.json",
        {
            "phase": args.phase,
            "loss": loss,
            "best_run_id": int(best_row["run_id"]),
            "best_value": report_objective_scalar(load_report(best_row["metric_path"])),
            "best_metric_path": best_row["metric_path"],
            "best_config_path": str(hpo_dir / "best_config.json"),
            "selected_models": models,
            "hpo_profile": profile_name,
            "attempted_trials": len(trial_rows),
            "completed_trials": sum(1 for row in trial_rows if row.get("status") == "completed"),
        },
    )


def create_ce_matched_config(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    source_paths = [Path(path) for path in parse_csv(args.configs, ())] if args.configs else []
    source_path = source_paths[0] if source_paths else output_root / "hpo-v3" / "best_config.json"
    if not source_path.exists() and not source_paths:
        source_path = output_root / "hpo-nicme" / "best_config.json"
    if not source_path.exists():
        raise FileNotFoundError(f"NICME best config not found for CE matched config: {source_path}")
    cfg = json.loads(source_path.read_text())
    backbone = infer_backbone(cfg)
    cfg.update(LOSS_PRESETS["ce"])
    cfg.update(
        {
            "loss_function": "cross_entropy",
            "pmi10_phase": "hpo-ce-matched",
            "pmi10_loss": "ce",
            "pmi10_run_role": "ce_matched",
            "pmi10_backbone": backbone,
            "pmi10_output_root": str(output_root),
            "seed": int(args.seed),
            "output_dir": f"hpo-ce-matched_{slugify(backbone)}_ce_s{int(args.seed)}_matched",
        }
    )
    if args.epochs is not None:
        cfg["num_train_epochs"] = int(args.epochs)
    target_path = output_root / "hpo-ce-matched" / "best_config.json"
    write_json(target_path, cfg)
    write_json(
        output_root / "hpo-ce-matched" / "matched_config_manifest.json",
        {
            "source_nicme_config": str(source_path),
            "matched_ce_config": str(target_path),
            "copied_non_loss_hyperparameters": True,
            "loss_fields_reset": LOSS_PRESETS["ce"],
        },
    )
    return target_path


def create_v2_matched_config(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    source_paths = [Path(path) for path in parse_csv(args.configs, ())] if args.configs else []
    source_path = source_paths[0] if source_paths else output_root / "hpo-v3" / "best_config.json"
    if not source_path.exists():
        raise FileNotFoundError(f"NICME V3 best config not found for V2 matched config: {source_path}")
    cfg = json.loads(source_path.read_text())
    backbone = infer_backbone(cfg)
    cfg["loss_function"] = "nicme_hybrid"
    cfg["pmi10_phase"] = "hpo-v2-matched"
    cfg["pmi10_loss"] = "nicme_hybrid"
    cfg["pmi10_run_role"] = "nicme_v2_matched"
    cfg["pmi10_backbone"] = backbone
    cfg["pmi10_output_root"] = str(output_root)
    cfg["seed"] = int(args.seed)
    cfg["output_dir"] = f"hpo-v2-matched_{slugify(backbone)}_nicme_hybrid_s{int(args.seed)}_matched"
    if args.epochs is not None:
        cfg["num_train_epochs"] = int(args.epochs)
    target_path = output_root / "hpo-v2-matched" / "best_config.json"
    write_json(target_path, cfg)
    write_json(
        output_root / "hpo-v2-matched" / "matched_config_manifest.json",
        {
            "source_v3_config": str(source_path),
            "matched_v2_config": str(target_path),
            "copied_non_loss_hyperparameters": True,
            "loss_fields_reset": {"loss_function": "nicme_hybrid", "pmi10_loss": "nicme_hybrid"},
        },
    )
    return target_path


def split_counts(split_dir: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "validation", "test"):
        path = split_dir / f"{split}.csv"
        if not path.exists():
            counts[split] = -1
            continue
        with path.open() as f:
            counts[split] = max(sum(1 for _ in f) - 1, 0)
    return counts


def active_training_processes() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-af", "scripts/train.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        return [f"pgrep_failed_returncode_{completed.returncode}"]
    current_pid = os.getpid()
    return [
        line
        for line in completed.stdout.splitlines()
        if line.strip()
        and not line.lstrip().startswith(f"{current_pid} ")
        and "pgrep -af" not in line
    ]


def gpu_visible() -> bool:
    completed = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return completed.returncode == 0


def preflight_report(args: argparse.Namespace) -> dict[str, Any]:
    profile_name = hpo_profile_name(args)
    profile = get_profile(DATASET)
    models = hpo_models(args)
    variant = variant_for_args(args)
    split_dir = Path(args.split_dir)
    counts = split_counts(split_dir)
    nicme_loss = "nicme_v3_hybrid" if profile_name == CONVNEXT_BASE_V3_BALANCED_PROFILE else "nicme_hybrid"
    nicme_cfg = base_config(
        "hpo-v3" if nicme_loss == "nicme_v3_hybrid" else "hpo-nicme",
        models[0],
        nicme_loss,
        hpo_epochs(args),
        int(args.seed),
        split_dir,
        Path(args.output_root),
        overrides=hpo_search_space(nicme_loss, profile_name)["fixed"],
        variant=variant,
    )
    ce_cfg = base_config(
        "hpo-ce",
        models[0],
        "ce",
        hpo_epochs(args),
        int(args.seed),
        split_dir,
        Path(args.output_root),
        overrides=hpo_search_space("ce", profile_name)["fixed"],
        variant=variant,
    )
    active = active_training_processes()
    expected_counts = (
        {"train": 970, "validation": 310, "test": 320}
        if variant == "balanced"
        else {"train": 3578, "validation": 1181, "test": 1186}
    )
    invariant_errors = []
    for cfg in (nicme_cfg, ce_cfg):
        if cfg["calibration_enabled"] is not False:
            invariant_errors.append("calibration_enabled_not_false")
        if cfg["decision_modes"] != "argmax":
            invariant_errors.append("decision_modes_not_argmax")
        if cfg["peft_enabled"] is not False:
            invariant_errors.append("peft_enabled_not_false")
        if infer_backbone(cfg) != CONVNEXT_BASE_MODEL and profile_name in {
            CONVNEXT_BASE_FAST_PROFILE,
            CONVNEXT_BASE_V3_BALANCED_PROFILE,
        }:
            invariant_errors.append("model_not_convnext_base")
        if cfg.get("image_size") != 224 and profile_name in {
            CONVNEXT_BASE_FAST_PROFILE,
            CONVNEXT_BASE_V3_BALANCED_PROFILE,
        }:
            invariant_errors.append("image_size_not_224")
        if profile_name == CONVNEXT_BASE_V3_BALANCED_PROFILE:
            if cfg.get("train_horizontal_flip") is not False:
                invariant_errors.append("v3_horizontal_flip_not_false")
            if cfg.get("train_random_quarter_turns") is not True:
                invariant_errors.append("v3_quarter_turns_not_true")
            if cfg.get("train_random_resized_crop_scale_min") != 0.8:
                invariant_errors.append("v3_crop_scale_min_not_0_8")
            if cfg.get("train_random_resized_crop_scale_max") != 1.0:
                invariant_errors.append("v3_crop_scale_max_not_1_0")
        if cfg["cost_matrix_sha256"] != profile.cost_matrix_sha256:
            invariant_errors.append("cost_hash_changed")
    checks = {
        "gpu_visible": gpu_visible(),
        "no_active_training_processes": not active,
        "split_counts_match": counts == expected_counts,
        "config_invariants_ok": not invariant_errors,
    }
    return {
        "hpo_profile": profile_name,
        "models": models,
        "split_dir": str(split_dir),
        "split_counts": counts,
        "expected_split_counts": expected_counts,
        "active_training_processes": active,
        "cost_matrix_sha256": profile.cost_matrix_sha256,
        "checks": checks,
        "invariant_errors": invariant_errors,
        "ok": all(checks.values()),
    }


def run_preflight(args: argparse.Namespace) -> None:
    report = preflight_report(args)
    report_path = Path(args.output_root) / "preflight" / "preflight_report.json"
    write_json(report_path, report)
    if not report["ok"]:
        raise RuntimeError(f"PMI-10 ConvNeXt-base preflight failed; see {report_path}")
    print(f"Preflight passed: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused PMI-10 no-calibration experiments")
    parser.add_argument(
        "--phase",
        choices=[
            "preflight",
            "smoke",
            "screen",
            "select",
            "hpo-nicme",
            "hpo-v3",
            "hpo-ce-matched",
            "hpo-v2-matched",
            "hpo-ce",
            "final",
            "train_plus_val_final",
            "tpt-stress",
        ],
        required=True,
    )
    parser.add_argument("--models", default=None, help="Comma-separated timm model ids. Defaults depend on phase.")
    parser.add_argument("--losses", default=None, help="Comma-separated losses: ce,nicme_hybrid")
    parser.add_argument("--configs", default=None, help="Comma-separated config JSON paths for final locked phases.")
    parser.add_argument("--hpo-profile", default=DEFAULT_HPO_PROFILE, choices=sorted(HPO_PROFILES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--variant", default=VARIANT, choices=["natural", "balanced"])
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--per-run-timeout-minutes", type=int, default=180)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.phase == "preflight":
        run_preflight(args)
        return
    if args.phase == "select":
        selected = select_top_backbones(args)
        print("Selected backbones:", ", ".join(selected))
        return
    if args.phase == "hpo-nicme":
        if args.execute:
            run_hpo(args, "nicme_hybrid")
        else:
            write_hpo_plan(args, "nicme_hybrid")
        return
    if args.phase == "hpo-v3":
        if args.execute:
            run_hpo(args, "nicme_v3_hybrid")
        else:
            write_hpo_plan(args, "nicme_v3_hybrid")
        return
    if args.phase == "hpo-ce-matched":
        matched_path = create_ce_matched_config(args)
        print(f"Wrote matched CE config: {matched_path}")
        return
    if args.phase == "hpo-v2-matched":
        matched_path = create_v2_matched_config(args)
        print(f"Wrote matched V2 config: {matched_path}")
        return
    if args.phase == "hpo-ce":
        if args.execute:
            run_hpo(args, "ce")
        else:
            write_hpo_plan(args, "ce")
        return

    rows = write_plan(args)
    if args.execute:
        executed_rows = []
        for idx, cfg in enumerate(build_runs(args), start=1):
            row = row_from_config(cfg, args.phase, idx)
            executed_rows.append(execute_run(cfg, row, args))
            write_ledger(Path(args.output_root) / args.phase / "run_ledger.csv", executed_rows)
    print(f"Wrote {len(rows)} planned runs under {Path(args.output_root) / args.phase}")


if __name__ == "__main__":
    main()
