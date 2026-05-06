"""Matched SOTA baseline runner for balanced PMI-10.

The suite keeps the balanced PMI-10 split, ConvNeXt-base ImageNet-22k
backbone, seed, cost matrix, preprocessing, and non-loss training defaults
fixed so the main comparison variable is the cost-sensitive method.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from nicme.dataset_profiles import get_profile

DATASET = "pmi_pills_10_no_cal"
VARIANT = "balanced"
MODEL_ID = "convnext_base.fb_in22k_ft_in1k"
RESULT_ROOT = Path("results/pmi10_sota_baselines_balanced_20260502")
SPLIT_DIR = Path("data/prepared/pmi_pills_10_no_cal/splits/balanced")
NICME_PHASE3_ROOT = Path("results/pmi10_v3_balanced_convnext_base_20260502")
PRETTY_RESULT_ROOTS = {
    "lr1e5": Path("results/pmi10_sota_pretty_balanced_lr1e5_20260503"),
    "lr5e5": Path("results/pmi10_sota_pretty_balanced_lr5e5_20260503"),
    "lr1e4": Path("results/pmi10_sota_pretty_balanced_lr1e4_20260503"),
}
COMBINED_PRETTY_ROOT = Path("results/pmi10_sota_pretty_balanced_dual_lr_20260503")
TRIPLE_PRETTY_ROOT = Path("results/pmi10_sota_pretty_balanced_triple_lr_20260503")
NICME_PRETTY_GRID_ROOT = Path("results/pmi10_nicme_pretty_alpha_lambda_lr5e5_20260503")
LEARNING_RATE_PROFILES = {"lr1e5": 1e-5, "lr5e5": 5e-5, "lr1e4": 1e-4}
LEARNING_RATE_LABELS = {
    "lr1e5": "LR 1e-5 (very conservative LR)",
    "lr5e5": "LR 5e-5",
    "lr1e4": "LR 1e-4 (aggressive LR)",
}
LEARNING_RATE_SUMMARY_LINES = {
    "lr1e5": "- `lr1e5`: very conservative clean fine-tuning LR, `1e-5`.",
    "lr5e5": "- `lr5e5`: conservative clean fine-tuning LR, `5e-5`.",
    "lr1e4": "- `lr1e4`: aggressive LR sensitivity chain, `1e-4`.",
}

MATCHED_DEFAULTS: dict[str, Any] = {
    "num_train_epochs": 32,
    "early_stopping_patience": 5,
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "lr_scheduler_type": "cosine",
    "learning_rate": 7.381097541389142e-05,
    "weight_decay": 0.0039047956499555798,
    "warmup_ratio": 0.1103352023905837,
    "image_size": 224,
    "train_random_resize_crop": True,
    "train_random_resized_crop_scale_min": 0.8,
    "train_random_resized_crop_scale_max": 1.0,
    "train_horizontal_flip": False,
    "train_random_quarter_turns": True,
    "train_color_jitter": False,
}
PRETTY_DEFAULTS: dict[str, Any] = {
    **MATCHED_DEFAULTS,
    "learning_rate": 5e-5,
    "weight_decay": 0.005,
    "warmup_ratio": 0.10,
}

DIRECT_BASELINES = (
    "cost_weighted_ce",
    "menon_logit_adjusted",
    "balanced_softmax",
    "class_balanced_focal",
    "ldam_drw",
)
ADAPTATION_BASELINES = ("cost_sensitive_regularized_ce", "sosr_cnn", "csada")
ALL_BASELINES = DIRECT_BASELINES + ADAPTATION_BASELINES
PRETTY_ANCHOR = "ce_anchor_pretty"
PRETTY_DIRECT_METHODS = (
    "nicme_hybrid_pretty",
) + DIRECT_BASELINES
PRETTY_METHODS = (PRETTY_ANCHOR,) + PRETTY_DIRECT_METHODS + ADAPTATION_BASELINES
PRETTY_GRID_ALPHA_VALUES = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
PRETTY_GRID_LAMBDA_VALUES = (0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3)
PRETTY_GRID_SMOKE_POINTS = ((0.04, 0.02), (0.08, 0.03), (0.1, 0.07), (0.4, 0.2), (0.6, 0.07))

LEDGER_FIELDS = [
    "run_id",
    "phase",
    "method",
    "role",
    "comparison_profile",
    "learning_rate_profile",
    "learning_rate",
    "parent_learning_rate",
    "alpha",
    "cs_lambda",
    "status",
    "config_hash",
    "config_path",
    "metric_path",
    "checkpoint_path",
    "prediction_mode",
    "target_recall_min",
    "target_recall_macro",
    "normalized_atc",
    "atc",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "critical_pair_error_count",
    "validation_target_recall_min",
    "validation_target_recall_macro",
    "validation_normalized_atc",
    "validation_atc",
    "validation_balanced_accuracy",
    "validation_accuracy",
    "validation_macro_f1",
    "validation_critical_pair_error_count",
    "cost_matrix_sha256",
    "start_time",
    "end_time",
    "elapsed_seconds",
    "stdout_log",
    "stderr_log",
    "returncode",
    "failure_reason",
]

RETRYABLE_LOG_PATTERNS = (
    "module.py\", line 2797, in named_children",
    "ValueError: not enough values to unpack (expected 2, got 1)",
    "TypeError: cannot unpack non-iterable Dropout object",
)


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cost_matrix_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config["cost_matrix"], separators=(",", ":")).encode("utf-8")).hexdigest()


def matched_defaults(comparison_profile: str, learning_rate_profile: str) -> dict[str, Any]:
    if comparison_profile == "pretty_balanced":
        defaults = dict(PRETTY_DEFAULTS)
        defaults["learning_rate"] = LEARNING_RATE_PROFILES[learning_rate_profile]
        return defaults
    return dict(MATCHED_DEFAULTS)


def anchor_method(comparison_profile: str) -> str:
    return PRETTY_ANCHOR if comparison_profile == "pretty_balanced" else "ce_anchor"


def direct_methods(comparison_profile: str) -> tuple[str, ...]:
    return PRETTY_DIRECT_METHODS if comparison_profile == "pretty_balanced" else DIRECT_BASELINES


def all_methods(comparison_profile: str) -> tuple[str, ...]:
    if comparison_profile == "pretty_balanced":
        return PRETTY_METHODS
    return ("ce_anchor",) + ALL_BASELINES


def lr_root(learning_rate_profile: str) -> Path:
    return PRETTY_RESULT_ROOTS[learning_rate_profile]


def profile_config() -> tuple[Any, Any]:
    profile = get_profile(DATASET)
    return profile, profile.balanced_selection


def base_config(
    phase: str,
    method: str,
    split_dir: Path = SPLIT_DIR,
    output_root: Path = RESULT_ROOT,
    seed: int = 42,
    epochs: int | None = None,
    overrides: dict[str, Any] | None = None,
    comparison_profile: str = "matched",
    learning_rate_profile: str = "lr5e5",
) -> dict[str, Any]:
    profile, selection = profile_config()
    method_slug = slugify(method)
    role = method
    decision_modes = "argmax"
    prediction_mode = "argmax"
    defaults = matched_defaults(comparison_profile, learning_rate_profile)
    cfg: dict[str, Any] = {
        "dataset_profile": profile.name,
        "variant": VARIANT,
        "dataset": f"{profile.name}_{VARIANT}",
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
        "model": f"timm/{MODEL_ID}",
        "model_backend": "timm",
        "model_type": "pmi10_sota_convnext_base",
        "peft_enabled": False,
        "timm_pretrained": True,
        "wandb": False,
        "push_to_hub": False,
        "calibration_enabled": False,
        "decision_modes": decision_modes,
        "prediction_mode": prediction_mode,
        "model_selection_metric": "target_recall",
        "save_total_limit": 1,
        "skip_visualizations": True,
        "seed": seed,
        "output_dir": f"{phase}_{method_slug}_s{seed}",
        "sota_phase": phase,
        "sota_method": method,
        "sota_role": role,
        "sota_comparison_profile": comparison_profile,
        "learning_rate_profile": learning_rate_profile,
        "parent_learning_rate": defaults["learning_rate"],
        "sota_output_root": str(output_root),
        **defaults,
    }
    if comparison_profile == "pretty_balanced":
        cfg["output_dir"] = f"{phase}_{method_slug}_{learning_rate_profile}_s{seed}"
    if epochs is not None:
        cfg["num_train_epochs"] = int(epochs)

    if method in {"ce_anchor", "ce_cost_min_inference", "ce_anchor_pretty", "ce_cost_min_inference_pretty"}:
        cfg["loss_function"] = "cross_entropy"
        cfg["decision_modes"] = "argmax,cost_min"
        cfg["sota_role"] = method
    elif method == "nicme_hybrid_pretty":
        cfg["loss_function"] = "nicme_hybrid"
        cfg["nicme_logit_cost_scale"] = 0.4
        cfg["cs_lambda"] = 0.25
        cfg["cs_warmup_epochs"] = 2
    elif method == "menon_logit_adjusted":
        cfg["loss_function"] = "menon_logit_adjusted"
        cfg["logit_adjustment_tau"] = 1.0
    elif method == "class_balanced_focal":
        cfg["loss_function"] = "class_balanced_focal"
        cfg["cb_beta"] = 0.9999
        cfg["focal_gamma"] = 2.0
    elif method == "ldam_drw":
        cfg["loss_function"] = "ldam_drw"
        cfg["ldam_max_m"] = 0.5
        cfg["ldam_scale"] = 30.0
        cfg["ldam_drw_start_epoch"] = 16
        cfg["cb_beta"] = 0.9999
    elif method == "cost_sensitive_regularized_ce":
        cfg["loss_function"] = "cost_sensitive_regularized_ce"
        cfg["cost_regularization_alpha"] = 5.0
        cfg["num_train_epochs"] = int(epochs if epochs is not None else 10)
        cfg["early_stopping_patience"] = 3
        cfg["learning_rate"] = 1e-5
        cfg["parent_learning_rate"] = defaults["learning_rate"]
    elif method == "sosr_cnn":
        cfg["loss_function"] = "sosr_cnn"
        cfg["prediction_mode"] = "sosr_argmin"
        cfg["num_train_epochs"] = int(epochs if epochs is not None else 10)
        cfg["early_stopping_patience"] = 3
        cfg["learning_rate"] = 1e-5
        cfg["parent_learning_rate"] = defaults["learning_rate"]
    elif method == "csada":
        cfg["loss_function"] = "csada"
        cfg["num_train_epochs"] = int(epochs if epochs is not None else 10)
        cfg["early_stopping_patience"] = 3
        cfg["learning_rate"] = 1e-5
        cfg["parent_learning_rate"] = defaults["learning_rate"]
        cfg["csada_attack_steps"] = 5
        cfg["csada_epsilon"] = 2.0 / 255.0
        cfg["csada_step_size"] = 1.0 / 255.0
        cfg["csada_lambda"] = 2.0
        cfg["csada_cost_temperature"] = 3.0
    else:
        cfg["loss_function"] = method

    if overrides:
        cfg.update(overrides)
    return cfg


def checkpoint_parent(cfg: dict[str, Any]) -> Path:
    return Path(str(cfg.get("checkpoint_root", "checkpoints"))) / str(cfg["output_dir"])


def find_metric_path(cfg: dict[str, Any]) -> str:
    model_type = str(cfg["model_type"]).lower()
    output_dir = str(cfg["output_dir"])
    loss_name = str(cfg["loss_function"])
    matches = sorted(
        Path("results").glob(f"{model_type}_test/{output_dir}_*/metrics_*_{loss_name}.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(matches[0]) if matches else ""


def preferred_report(payload: dict[str, Any], method: str) -> tuple[str, dict[str, Any]]:
    reports = payload.get("decision_reports", {})
    if method in {"ce_cost_min_inference", "ce_cost_min_inference_pretty"}:
        return "cost_min", reports.get("cost_min", {})
    if method == "sosr_cnn":
        return "sosr_argmin", reports.get("sosr_argmin", {})
    if reports.get("argmax"):
        return "argmax", reports["argmax"]
    if reports:
        key = next(iter(reports))
        return key, reports[key]
    return "", {}


def load_metric_report(metric_path: str, method: str) -> tuple[str, dict[str, Any]]:
    if not metric_path:
        return "", {}
    with open(metric_path) as f:
        payload = json.load(f)
    return preferred_report(payload, method)


def critical_pair_error_count(report: dict[str, Any]) -> int:
    return int(sum(item.get("errors", 0) for item in report.get("critical_pair_errors", {}).values()))


def row_from_config(cfg: dict[str, Any], phase: str, run_id: int, status: str = "planned") -> dict[str, Any]:
    method = str(cfg["sota_method"])
    return {
        "run_id": run_id,
        "phase": phase,
        "method": method,
        "role": cfg.get("sota_role", method),
        "comparison_profile": cfg.get("sota_comparison_profile", "matched"),
        "learning_rate_profile": cfg.get("learning_rate_profile", ""),
        "learning_rate": cfg.get("learning_rate", ""),
        "parent_learning_rate": cfg.get("parent_learning_rate", ""),
        "alpha": cfg.get("nicme_logit_cost_scale", ""),
        "cs_lambda": cfg.get("cs_lambda", ""),
        "status": status,
        "config_hash": config_hash(cfg),
        "config_path": "",
        "metric_path": "",
        "checkpoint_path": str(checkpoint_parent(cfg)),
        "prediction_mode": cfg.get("prediction_mode", "argmax"),
        "target_recall_min": "",
        "target_recall_macro": "",
        "normalized_atc": "",
        "atc": "",
        "balanced_accuracy": "",
        "accuracy": "",
        "macro_f1": "",
        "critical_pair_error_count": "",
        "validation_target_recall_min": "",
        "validation_target_recall_macro": "",
        "validation_normalized_atc": "",
        "validation_atc": "",
        "validation_balanced_accuracy": "",
        "validation_accuracy": "",
        "validation_macro_f1": "",
        "validation_critical_pair_error_count": "",
        "cost_matrix_sha256": cfg["cost_matrix_sha256"],
        "start_time": "",
        "end_time": "",
        "elapsed_seconds": "",
        "stdout_log": "",
        "stderr_log": "",
        "returncode": "",
        "failure_reason": "",
    }


def update_row_with_report(row: dict[str, Any], report: dict[str, Any], metric_path: str, mode: str) -> None:
    row["metric_path"] = metric_path
    row["prediction_mode"] = mode or row.get("prediction_mode", "")
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


def pretty_numeric_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def has_one_nonzero_digit(value: float) -> bool:
    text = f"{value:g}".replace(".", "").replace("-", "").lstrip("0")
    return sum(char != "0" for char in text) == 1


def update_row_with_validation_report(row: dict[str, Any], payload: dict[str, Any]) -> None:
    validation_report = payload.get("validation_report", {})
    metrics = validation_report.get("metrics", {})
    report = validation_report.get("argmax", {})
    row["validation_target_recall_min"] = metrics.get(
        "eval_target_recall",
        report.get("target_recall_min", report.get("target_recall", "")),
    )
    row["validation_target_recall_macro"] = metrics.get(
        "eval_target_recall_macro",
        report.get("target_recall_macro", ""),
    )
    row["validation_normalized_atc"] = metrics.get(
        "eval_normalized_atc",
        report.get("normalized_atc", ""),
    )
    row["validation_atc"] = metrics.get("eval_atc", report.get("atc", ""))
    row["validation_balanced_accuracy"] = metrics.get(
        "eval_balanced_accuracy",
        report.get("balanced_accuracy", ""),
    )
    row["validation_accuracy"] = metrics.get("eval_accuracy", report.get("accuracy", ""))
    row["validation_macro_f1"] = metrics.get("eval_macro_f1", report.get("macro_f1", ""))
    row["validation_critical_pair_error_count"] = critical_pair_error_count(report) if report else ""


def update_grid_row_from_metric(row: dict[str, Any]) -> None:
    metric_path = row.get("metric_path", "")
    if not metric_path:
        return
    with open(metric_path) as f:
        payload = json.load(f)
    update_row_with_validation_report(row, payload)


def read_grid_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_grid_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fields = fields or LEDGER_FIELDS
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_grid_config(
    phase: str,
    alpha: float,
    cs_lambda: float,
    split_dir: Path,
    output_root: Path,
    seed: int,
    epochs: int | None = None,
) -> dict[str, Any]:
    cfg = base_config(
        phase,
        "nicme_hybrid_pretty",
        split_dir,
        output_root,
        seed,
        epochs=epochs,
        comparison_profile="pretty_balanced",
        learning_rate_profile="lr5e5",
    )
    cfg["nicme_logit_cost_scale"] = float(alpha)
    cfg["cs_lambda"] = float(cs_lambda)
    cfg["cs_warmup_epochs"] = 2
    cfg["sota_method"] = "nicme_hybrid_pretty"
    cfg["sota_role"] = "nicme_pretty_alpha_lambda_grid"
    cfg["sota_comparison_profile"] = "nicme_pretty_alpha_lambda_grid"
    cfg["alpha_lambda_grid"] = True
    cfg["grid_alpha"] = float(alpha)
    cfg["grid_cs_lambda"] = float(cs_lambda)
    cfg["output_dir"] = (
        f"{phase}_nicme_hybrid_pretty_a{pretty_numeric_label(alpha)}_"
        f"l{pretty_numeric_label(cs_lambda)}_lr5e5_s{seed}"
    )
    return cfg


def build_grid_configs(args: argparse.Namespace, phase: str, smoke: bool = False) -> list[dict[str, Any]]:
    points = PRETTY_GRID_SMOKE_POINTS if smoke else [
        (alpha, cs_lambda) for alpha in PRETTY_GRID_ALPHA_VALUES for cs_lambda in PRETTY_GRID_LAMBDA_VALUES
    ]
    epochs = 1 if smoke else None
    configs = [
        build_grid_config(
            phase,
            alpha,
            cs_lambda,
            Path(args.split_dir),
            Path(args.output_root),
            args.seed,
            epochs=epochs,
        )
        for alpha, cs_lambda in points
    ]
    if smoke:
        for cfg in configs:
            cfg["save_total_limit"] = 1
    return configs


def assert_grid_configs(configs: list[dict[str, Any]]) -> None:
    if len(configs) != len({cfg["output_dir"] for cfg in configs}):
        raise ValueError("Grid configs must have unique output directories")
    for cfg in configs:
        alpha = float(cfg["nicme_logit_cost_scale"])
        cs_lambda = float(cfg["cs_lambda"])
        if not has_one_nonzero_digit(alpha) or not has_one_nonzero_digit(cs_lambda):
            raise ValueError(f"Grid value is not pretty: alpha={alpha}, cs_lambda={cs_lambda}")
        if cfg["loss_function"] != "nicme_hybrid":
            raise ValueError("Grid must use nicme_hybrid")
        if float(cfg["learning_rate"]) != 5e-5:
            raise ValueError("Grid must keep learning_rate=5e-5")
        if int(cfg["cs_warmup_epochs"]) != 2:
            raise ValueError("Grid must keep cs_warmup_epochs=2")
        if cfg["calibration_enabled"] or cfg["peft_enabled"] or cfg["decision_modes"] != "argmax":
            raise ValueError("Grid invariants failed: no calibration, no LoRA, argmax only")


def validation_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        -float(row.get("validation_target_recall_min") or 0.0),
        float(row.get("validation_normalized_atc") or 1.0),
        -float(row.get("validation_balanced_accuracy") or 0.0),
        -float(row.get("validation_target_recall_macro") or 0.0),
        int(float(row.get("validation_critical_pair_error_count") or row.get("critical_pair_error_count") or 999)),
    )


def _stderr_contains_retryable_failure(stderr_log: Path) -> bool:
    if not stderr_log.exists() or stderr_log.stat().st_size == 0:
        return False
    text = stderr_log.read_text(errors="replace")
    return any(pattern in text for pattern in RETRYABLE_LOG_PATTERNS)


def _grid_runtime_config(cfg: dict[str, Any], attempt: int) -> dict[str, Any]:
    runtime_cfg = dict(cfg)
    if attempt > 1:
        runtime_cfg["output_dir"] = f"{cfg['output_dir']}_retry{attempt}"
    return runtime_cfg


def _grid_attempt_path(base_path: Path, attempt: int) -> Path:
    if attempt == 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}.attempt{attempt}{base_path.suffix}")


def select_grid_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        row
        for row in rows
        if row.get("status") == "completed" and row.get("validation_target_recall_min") not in {"", None}
    ]
    if not completed:
        raise RuntimeError("No completed grid rows with validation metrics")
    best_target = max(float(row.get("validation_target_recall_min") or 0.0) for row in completed)
    eligible = [
        row
        for row in completed
        if float(row.get("validation_target_recall_min") or 0.0) >= best_target - 0.005
    ]
    return sorted(
        eligible,
        key=lambda row: (
            float(row.get("validation_normalized_atc") or 1.0),
            -float(row.get("validation_balanced_accuracy") or 0.0),
            -float(row.get("validation_target_recall_macro") or 0.0),
            int(float(row.get("validation_critical_pair_error_count") or row.get("critical_pair_error_count") or 999)),
        ),
    )[0]


def write_grid_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    configs = build_grid_configs(args, "grid-dry-run")
    assert_grid_configs(configs)
    rows = []
    for idx, cfg in enumerate(configs, start=1):
        row = row_from_config(cfg, "grid-dry-run", idx)
        cfg_hash = row["config_hash"]
        config_path = output_root / "grid-dry-run" / "configs" / f"{idx:04d}_{cfg_hash}.json"
        write_json(config_path, cfg)
        row["config_path"] = str(config_path)
        rows.append(row)
    write_grid_csv(output_root / "grid_plan.csv", rows)
    write_grid_csv(output_root / "grid-dry-run" / "grid_plan.csv", rows)
    return rows


def execute_grid_phase(args: argparse.Namespace, phase: str, smoke: bool = False) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    configs = build_grid_configs(args, phase, smoke=smoke)
    assert_grid_configs(configs)
    existing_rows = {
        int(row["run_id"]): row
        for row in read_grid_csv(output_root / phase / "grid_run_ledger.csv")
        if str(row.get("run_id", "")).isdigit()
    }
    rows: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs, start=1):
        existing = existing_rows.get(idx)
        if existing and existing.get("status") == "completed":
            if existing.get("metric_path") and not existing.get("validation_target_recall_min"):
                update_grid_row_from_metric(existing)
            row = existing
        else:
            row = execute_config(cfg, phase, idx, args)
            update_grid_row_from_metric(row)
        rows.append(row)
        write_grid_csv(output_root / phase / "grid_run_ledger.csv", rows)
        if not smoke:
            write_grid_csv(output_root / "grid_run_ledger.csv", rows)
        if row["status"] != "completed":
            raise RuntimeError(f"Grid run failed for alpha={row.get('alpha')} lambda={row.get('cs_lambda')}")
    return rows


def write_grid_analysis(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    rows = read_ledger(output_root / "grid_run_ledger.csv")
    for row in rows:
        if row.get("status") == "completed" and row.get("metric_path") and not row.get("validation_target_recall_min"):
            update_grid_row_from_metric(row)
    completed = [row for row in rows if row.get("status") == "completed"]
    ranked = sorted(completed, key=validation_sort_key)
    selected = select_grid_row(ranked)

    selected_config = {}
    if selected.get("config_path"):
        with open(selected["config_path"]) as f:
            selected_config = json.load(f)
    write_json(output_root / "selected_config.json", selected_config)
    write_grid_csv(output_root / "validation_ranked_table.csv", ranked)
    heatmap_fields = [
        "alpha",
        "cs_lambda",
        "validation_target_recall_min",
        "validation_target_recall_macro",
        "validation_normalized_atc",
        "validation_balanced_accuracy",
        "target_recall_min",
        "normalized_atc",
        "balanced_accuracy",
        "critical_pair_error_count",
        "status",
        "metric_path",
    ]
    write_grid_csv(output_root / "alpha_lambda_heatmap_data.csv", ranked, heatmap_fields)

    lines = [
        "# NICME Pretty Alpha/Lambda Grid Summary",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "- Selection uses validation metrics only.",
        "- Test metrics below are reported after validation-based selection.",
        "",
        "## Selected Config",
        "",
        f"- alpha: `{selected.get('alpha')}`",
        f"- cs_lambda: `{selected.get('cs_lambda')}`",
        f"- learning_rate: `{selected.get('learning_rate')}`",
        f"- config: `{selected.get('config_path')}`",
        f"- metrics: `{selected.get('metric_path')}`",
        "",
        "## Validation Metrics",
        "",
        f"- target-min recall: `{selected.get('validation_target_recall_min')}`",
        f"- target-macro recall: `{selected.get('validation_target_recall_macro')}`",
        f"- normalized ATC: `{selected.get('validation_normalized_atc')}`",
        f"- balanced accuracy: `{selected.get('validation_balanced_accuracy')}`",
        f"- critical-pair errors: `{selected.get('validation_critical_pair_error_count')}`",
        "",
        "## Test Metrics",
        "",
        f"- target-min recall: `{selected.get('target_recall_min')}`",
        f"- target-macro recall: `{selected.get('target_recall_macro')}`",
        f"- normalized ATC: `{selected.get('normalized_atc')}`",
        f"- balanced accuracy: `{selected.get('balanced_accuracy')}`",
        f"- critical-pair errors: `{selected.get('critical_pair_error_count')}`",
        "",
        "## Top 15 By Validation",
        "",
        "| Rank | Alpha | Lambda | Val Target-Min | Val Norm. ATC | Val Bal. Acc. | Test Target-Min | Test Norm. ATC | Test Bal. Acc. |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(ranked[:15], start=1):
        lines.append(
            "| "
            f"{idx} | {row.get('alpha')} | {row.get('cs_lambda')} | "
            f"{float(row.get('validation_target_recall_min') or 0.0):.4f} | "
            f"{float(row.get('validation_normalized_atc') or 0.0):.6f} | "
            f"{float(row.get('validation_balanced_accuracy') or 0.0):.4f} | "
            f"{float(row.get('target_recall_min') or 0.0):.4f} | "
            f"{float(row.get('normalized_atc') or 0.0):.6f} | "
            f"{float(row.get('balanced_accuracy') or 0.0):.4f} |"
        )
    summary = "\n".join(lines) + "\n"
    (output_root / "selected_result_summary.md").write_text(summary)
    (output_root / "alpha_lambda_summary.md").write_text(summary)
    print(f"Wrote grid ranking: {output_root / 'validation_ranked_table.csv'}")
    print(f"Wrote selected config: {output_root / 'selected_config.json'}")
    print(f"Wrote summary: {output_root / 'selected_result_summary.md'}")
    return output_root / "validation_ranked_table.csv"


def execute_config(cfg: dict[str, Any], phase: str, run_id: int, args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    row = row_from_config(cfg, phase, run_id, status="running")
    cfg_hash = row["config_hash"]
    row["start_time"] = dt.datetime.now().isoformat(timespec="seconds")
    start = time.time()
    base_config_path = output_root / phase / "configs" / f"{run_id:04d}_{cfg_hash}.json"
    base_stdout_log = output_root / phase / "logs" / f"{run_id:04d}_{cfg_hash}.stdout.log"
    base_stderr_log = output_root / phase / "logs" / f"{run_id:04d}_{cfg_hash}.stderr.log"
    base_stdout_log.parent.mkdir(parents=True, exist_ok=True)

    for attempt in (1, 2):
        runtime_cfg = _grid_runtime_config(cfg, attempt)
        config_path = _grid_attempt_path(base_config_path, attempt)
        stdout_log = _grid_attempt_path(base_stdout_log, attempt)
        stderr_log = _grid_attempt_path(base_stderr_log, attempt)
        write_json(config_path, runtime_cfg)
        row["config_path"] = str(config_path)
        row["stdout_log"] = str(stdout_log)
        row["stderr_log"] = str(stderr_log)
        row["checkpoint_path"] = str(checkpoint_parent(runtime_cfg))
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
            row["failure_reason"] = "" if completed.returncode == 0 else f"returncode_{completed.returncode}"
        except subprocess.TimeoutExpired:
            row["returncode"] = 124
            row["status"] = "timeout"
            row["failure_reason"] = "timeout"
        except Exception as exc:
            row["returncode"] = 1
            row["status"] = "failed"
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"

        if row["status"] == "completed":
            if attempt > 1:
                row["failure_reason"] = f"succeeded_after_retry_{attempt}"
            break
        if attempt == 1 and _stderr_contains_retryable_failure(stderr_log):
            row["failure_reason"] = f"{row['failure_reason']}; retrying_transient_module_traversal_failure"
            continue
        break

    row["end_time"] = dt.datetime.now().isoformat(timespec="seconds")
    row["elapsed_seconds"] = f"{time.time() - start:.1f}"
    if row["status"] == "completed":
        metric_path = find_metric_path(runtime_cfg)
        mode, report = load_metric_report(metric_path, str(runtime_cfg["sota_method"]))
        update_row_with_report(row, report, metric_path, mode)
    return row


def assert_split_ready(split_dir: Path) -> None:
    expected = {"train": 970, "validation": 310, "test": 320}
    for split, count in expected.items():
        path = split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing balanced PMI-10 split file: {path}")
        with open(path) as f:
            row_count = sum(1 for _ in f) - 1
        if row_count != count:
            raise ValueError(f"{path} has {row_count} rows; expected {count}")
    if (split_dir / "calibration.csv").exists():
        raise ValueError(f"No-calibration split must not contain calibration.csv: {split_dir}")


def run_preflight(args: argparse.Namespace) -> None:
    split_dir = Path(args.split_dir)
    assert_split_ready(split_dir)
    subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
    cfg = base_config(
        "preflight",
        "cost_weighted_ce",
        split_dir,
        Path(args.output_root),
        args.seed,
        comparison_profile=args.comparison_profile,
        learning_rate_profile=args.learning_rate_profile,
    )
    profile = get_profile(DATASET)
    if cfg["cost_matrix_sha256"] != profile.cost_matrix_sha256 or cost_matrix_hash(cfg) != profile.cost_matrix_sha256:
        raise ValueError("PMI-10 cost matrix hash mismatch")
    if cfg["model"] != f"timm/{MODEL_ID}" or cfg["variant"] != VARIANT or cfg["calibration_enabled"]:
        raise ValueError("SOTA baseline config invariants failed")
    print("SOTA PMI-10 preflight passed")


def build_plan_configs(args: argparse.Namespace, phase: str, anchor_path: str | None = None) -> list[dict[str, Any]]:
    split_dir = Path(args.split_dir)
    output_root = Path(args.output_root)
    methods = all_methods(args.comparison_profile)
    if phase == "smoke":
        epochs = 1
    elif phase == "run":
        epochs = None
    else:
        epochs = None
    configs = []
    for method in methods:
        cfg = base_config(
            phase,
            method,
            split_dir,
            output_root,
            args.seed,
            epochs=epochs,
            comparison_profile=args.comparison_profile,
            learning_rate_profile=args.learning_rate_profile,
        )
        if method in ADAPTATION_BASELINES and anchor_path:
            cfg["initial_checkpoint_path"] = anchor_path
        if phase == "smoke":
            cfg["output_dir"] = f"smoke_{slugify(method)}_s{args.seed}"
            cfg["save_total_limit"] = 1
        configs.append(cfg)
    return configs


def write_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    rows = []
    configs = build_plan_configs(args, "dry-run")
    for idx, cfg in enumerate(configs, start=1):
        row = row_from_config(cfg, "dry-run", idx)
        cfg_hash = row["config_hash"]
        config_path = output_root / "dry-run" / "configs" / f"{idx:04d}_{cfg_hash}.json"
        write_json(config_path, cfg)
        row["config_path"] = str(config_path)
        rows.append(row)
    write_ledger(output_root / "dry-run" / "run_ledger.csv", rows)
    return rows


def execute_smoke(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []
    anchor = anchor_method(args.comparison_profile)
    anchor_cfg = base_config(
        "smoke",
        anchor,
        Path(args.split_dir),
        output_root,
        args.seed,
        epochs=1,
        comparison_profile=args.comparison_profile,
        learning_rate_profile=args.learning_rate_profile,
    )
    anchor_cfg["output_dir"] = f"smoke_{slugify(anchor)}_{args.learning_rate_profile}_s{args.seed}"
    anchor_row = execute_config(anchor_cfg, "smoke", 1, args)
    rows.append(anchor_row)
    write_ledger(output_root / "smoke" / "run_ledger.csv", rows)
    if anchor_row["status"] != "completed":
        raise RuntimeError("Smoke CE anchor failed; refusing to launch baseline smoke runs")
    anchor_path = anchor_row["checkpoint_path"]
    run_id = 2
    for method in all_methods(args.comparison_profile):
        if method == anchor:
            continue
        cfg = base_config(
            "smoke",
            method,
            Path(args.split_dir),
            output_root,
            args.seed,
            epochs=1,
            comparison_profile=args.comparison_profile,
            learning_rate_profile=args.learning_rate_profile,
        )
        cfg["output_dir"] = f"smoke_{slugify(method)}_{args.learning_rate_profile}_s{args.seed}"
        if method in ADAPTATION_BASELINES:
            cfg["initial_checkpoint_path"] = anchor_path
        row = execute_config(cfg, "smoke", run_id, args)
        rows.append(row)
        write_ledger(output_root / "smoke" / "run_ledger.csv", rows)
        if row["status"] != "completed":
            raise RuntimeError(f"Smoke baseline failed for {method}: {row.get('failure_reason')}")
        run_id += 1
    return rows


def execute_run_chain(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []
    run_id = 1
    anchor = anchor_method(args.comparison_profile)
    anchor_cfg = base_config(
        "run",
        anchor,
        Path(args.split_dir),
        output_root,
        args.seed,
        comparison_profile=args.comparison_profile,
        learning_rate_profile=args.learning_rate_profile,
    )
    anchor_row = execute_config(anchor_cfg, "run", run_id, args)
    rows.append(anchor_row)
    write_ledger(output_root / "run" / "run_ledger.csv", rows)
    if anchor_row["status"] != "completed":
        raise RuntimeError("Shared CE anchor failed; stopping SOTA baseline chain")
    anchor_path = anchor_row["checkpoint_path"]

    for method in direct_methods(args.comparison_profile):
        run_id += 1
        cfg = base_config(
            "run",
            method,
            Path(args.split_dir),
            output_root,
            args.seed,
            comparison_profile=args.comparison_profile,
            learning_rate_profile=args.learning_rate_profile,
        )
        row = execute_config(cfg, "run", run_id, args)
        rows.append(row)
        write_ledger(output_root / "run" / "run_ledger.csv", rows)
        if row["status"] != "completed":
            raise RuntimeError(f"Direct baseline failed for {method}: {row.get('failure_reason')}")

    for method in ADAPTATION_BASELINES:
        run_id += 1
        cfg = base_config(
            "run",
            method,
            Path(args.split_dir),
            output_root,
            args.seed,
            comparison_profile=args.comparison_profile,
            learning_rate_profile=args.learning_rate_profile,
        )
        cfg["initial_checkpoint_path"] = anchor_path
        row = execute_config(cfg, "run", run_id, args)
        rows.append(row)
        write_ledger(output_root / "run" / "run_ledger.csv", rows)
        if row["status"] != "completed":
            raise RuntimeError(f"Adaptation baseline failed for {method}: {row.get('failure_reason')}")
    return rows


def config_metadata(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            cfg = json.load(f)
    except Exception:
        return {}
    return {
        "comparison_profile": cfg.get("sota_comparison_profile", ""),
        "lr_profile": cfg.get("learning_rate_profile", ""),
        "learning_rate": cfg.get("learning_rate", ""),
        "parent_learning_rate": cfg.get("parent_learning_rate", ""),
    }


def metric_row(
    method: str,
    role: str,
    metric_path: str,
    decision_mode: str,
    report: dict[str, Any],
    source: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "method": method,
        "role": role,
        "source": source,
        "comparison_profile": metadata.get("comparison_profile", ""),
        "lr_profile": metadata.get("lr_profile", ""),
        "learning_rate": metadata.get("learning_rate", ""),
        "parent_learning_rate": metadata.get("parent_learning_rate", ""),
        "metric_path": metric_path,
        "decision_mode": decision_mode,
        "target_recall_min": report.get("target_recall_min", report.get("target_recall", "")),
        "target_recall_macro": report.get("target_recall_macro", ""),
        "normalized_atc": report.get("normalized_atc", ""),
        "atc": report.get("atc", ""),
        "balanced_accuracy": report.get("balanced_accuracy", ""),
        "accuracy": report.get("accuracy", ""),
        "macro_f1": report.get("macro_f1", ""),
        "critical_pair_error_count": critical_pair_error_count(report),
        "cost_matrix_sha256": report.get("cost_matrix_sha256", ""),
    }


def read_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def collect_previous_nicme_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_ledger(NICME_PHASE3_ROOT / "final" / "run_ledger.csv"):
        if row.get("status") != "completed" or not row.get("metric_path"):
            continue
        method = row.get("role") or row.get("loss") or "nicme_reference"
        mode, report = load_metric_report(row["metric_path"], method)
        rows.append(metric_row(method, method, row["metric_path"], mode or "argmax", report, "nicme_phase3_final"))
    return rows


def sort_key(item: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        -float(item.get("target_recall_min") or 0.0),
        float(item.get("normalized_atc") or 1.0),
        -float(item.get("balanced_accuracy") or 0.0),
        int(float(item.get("critical_pair_error_count") or 999)),
    )


def write_hyperparameter_summary(path: Path, output_root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# PMI-10 SOTA Pretty Hyperparameters" if args.comparison_profile == "pretty_balanced" else "# PMI-10 SOTA Hyperparameters",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Result root: `{output_root}`",
        f"- Comparison profile: `{args.comparison_profile}`",
        f"- Learning-rate profile: `{args.learning_rate_profile}`",
        f"- Chain learning rate: `{LEARNING_RATE_PROFILES.get(args.learning_rate_profile, MATCHED_DEFAULTS['learning_rate'])}`",
        "- Shared model: `convnext_base.fb_in22k_ft_in1k`",
        "- Shared split: balanced PMI-10 no-calibration",
        "",
        "| Method | LR | Parent LR | Notes |",
        "|---|---:|---:|---|",
    ]
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get("method")), str(row.get("learning_rate")), str(row.get("parent_learning_rate")))
        if key in seen:
            continue
        seen.add(key)
        method = row.get("method", "")
        notes = ""
        if method == "nicme_hybrid_pretty":
            notes = "`alpha=0.4`, `cs_lambda=0.25`, `cs_warmup_epochs=2`"
        elif method in ADAPTATION_BASELINES:
            notes = "CE-anchor adaptation; native adaptation LR `1e-5`"
        elif method == "ce_cost_min_inference_pretty":
            notes = "Inference-only cost-min report from CE anchor"
        lines.append(
            f"| {method} | `{row.get('learning_rate', '')}` | `{row.get('parent_learning_rate', '')}` | {notes} |"
        )
    path.write_text("\n".join(lines) + "\n")


def read_analysis_table(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def combined_analyze(args: argparse.Namespace) -> Path:
    combined_root = Path(args.combined_output_root)
    sources = [
        ("lr1e5", Path(args.lr1e5_output_root)),
        ("lr5e5", Path(args.lr5e5_output_root)),
        ("lr1e4", Path(args.lr1e4_output_root)),
    ]
    rows: list[dict[str, Any]] = []
    included_profiles: list[str] = []
    for lr_profile, root in sources:
        profile_rows = read_analysis_table(root / "analysis" / "comparison_table.csv")
        if not profile_rows:
            continue
        included_profiles.append(lr_profile)
        for row in profile_rows:
            row["lr_profile"] = row.get("lr_profile") or lr_profile
            row["learning_rate"] = row.get("learning_rate") or str(LEARNING_RATE_PROFILES[lr_profile])
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No pretty SOTA analysis tables were found for combined analysis")
    rows.sort(key=sort_key)

    combined_root.mkdir(parents=True, exist_ok=True)
    table_path = combined_root / "comparison_table.csv"
    summary_path = combined_root / "comparison_summary.md"
    fields = [
        "method",
        "role",
        "source",
        "comparison_profile",
        "lr_profile",
        "learning_rate",
        "parent_learning_rate",
        "decision_mode",
        "target_recall_min",
        "target_recall_macro",
        "normalized_atc",
        "atc",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "critical_pair_error_count",
        "cost_matrix_sha256",
        "metric_path",
    ]
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    title = (
        "# PMI-10 Balanced SOTA Pretty Triple-LR Comparison"
        if "lr1e5" in included_profiles
        else "# PMI-10 Balanced SOTA Pretty Dual-LR Comparison"
    )
    lines = [
        title,
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        *[LEARNING_RATE_SUMMARY_LINES[profile] for profile in included_profiles],
        "",
        "## Overall Ranking",
        "",
        "| Rank | Method | LR Profile | LR | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{idx} | {row.get('method')} | {row.get('lr_profile')} | {row.get('learning_rate')} | "
            f"{row.get('decision_mode')} | {float(row.get('target_recall_min') or 0.0):.4f} | "
            f"{float(row.get('normalized_atc') or 0.0):.6f} | "
            f"{float(row.get('balanced_accuracy') or 0.0):.4f} | "
            f"{row.get('critical_pair_error_count', '')} |"
        )
    for lr_profile in included_profiles:
        subset = [row for row in rows if row.get("lr_profile") == lr_profile]
        subset.sort(key=sort_key)
        label = LEARNING_RATE_LABELS[lr_profile]
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                "| Rank | Method | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for idx, row in enumerate(subset, start=1):
            lines.append(
                "| "
                f"{idx} | {row.get('method')} | {row.get('decision_mode')} | "
                f"{float(row.get('target_recall_min') or 0.0):.4f} | "
                f"{float(row.get('normalized_atc') or 0.0):.6f} | "
                f"{float(row.get('balanced_accuracy') or 0.0):.4f} | "
                f"{row.get('critical_pair_error_count', '')} |"
            )
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote combined comparison table: {table_path}")
    print(f"Wrote combined comparison summary: {summary_path}")
    return table_path


def analyze(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    rows: list[dict[str, Any]] = []
    for row in read_ledger(output_root / "run" / "run_ledger.csv"):
        if row.get("status") != "completed" or not row.get("metric_path"):
            continue
        method = row["method"]
        metadata = config_metadata(row.get("config_path", ""))
        mode, report = load_metric_report(row["metric_path"], method)
        rows.append(metric_row(method, row.get("role", method), row["metric_path"], mode, report, "sota_baseline_run", metadata))
        if method in {"ce_anchor", "ce_anchor_pretty"}:
            cost_mode, cost_report = load_metric_report(row["metric_path"], "ce_cost_min_inference")
            if cost_report:
                cost_method = "ce_cost_min_inference_pretty" if args.comparison_profile == "pretty_balanced" else "ce_cost_min_inference"
                rows.append(
                    metric_row(
                        cost_method,
                        cost_method,
                        row["metric_path"],
                        cost_mode,
                        cost_report,
                        "sota_baseline_run",
                        metadata,
                    )
                )
    if args.comparison_profile != "pretty_balanced":
        rows.extend(collect_previous_nicme_rows())
    rows.sort(key=sort_key)
    table_path = output_root / "analysis" / "comparison_table.csv"
    summary_path = output_root / "analysis" / "comparison_summary.md"
    hyper_path = output_root / "analysis" / "method_hyperparameters.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "role",
        "source",
        "comparison_profile",
        "lr_profile",
        "learning_rate",
        "parent_learning_rate",
        "decision_mode",
        "target_recall_min",
        "target_recall_macro",
        "normalized_atc",
        "atc",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "critical_pair_error_count",
        "cost_matrix_sha256",
        "metric_path",
    ]
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    lines = [
        "# PMI-10 Balanced SOTA Baseline Comparison",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "| Rank | Method | LR Profile | LR | Decision | Target-Min Recall | Norm. ATC | Balanced Acc. | Critical Errors |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{idx} | {row.get('method')} | {row.get('lr_profile', '')} | "
            f"{row.get('learning_rate', '')} | {row.get('decision_mode')} | "
            f"{float(row.get('target_recall_min') or 0.0):.4f} | "
            f"{float(row.get('normalized_atc') or 0.0):.6f} | "
            f"{float(row.get('balanced_accuracy') or 0.0):.4f} | "
            f"{row.get('critical_pair_error_count', '')} |"
        )
    summary_path.write_text("\n".join(lines) + "\n")
    write_hyperparameter_summary(hyper_path, output_root, rows, args)
    print(f"Wrote comparison table: {table_path}")
    print(f"Wrote comparison summary: {summary_path}")
    print(f"Wrote hyperparameter summary: {hyper_path}")
    return table_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=[
            "preflight",
            "dry-run",
            "smoke",
            "run",
            "analyze",
            "combined-analyze",
            "grid-dry-run",
            "grid-smoke",
            "grid-run",
            "grid-analyze",
        ],
        required=True,
    )
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    parser.add_argument("--output-root", default=str(RESULT_ROOT))
    parser.add_argument("--comparison-profile", choices=["matched", "pretty_balanced"], default="matched")
    parser.add_argument("--learning-rate-profile", choices=sorted(LEARNING_RATE_PROFILES), default="lr5e5")
    parser.add_argument("--combined-output-root", default=str(TRIPLE_PRETTY_ROOT))
    parser.add_argument("--lr1e5-output-root", default=str(PRETTY_RESULT_ROOTS["lr1e5"]))
    parser.add_argument("--lr5e5-output-root", default=str(PRETTY_RESULT_ROOTS["lr5e5"]))
    parser.add_argument("--lr1e4-output-root", default=str(PRETTY_RESULT_ROOTS["lr1e4"]))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-run-timeout-minutes", type=int, default=720)
    args = parser.parse_args(argv)

    if args.comparison_profile == "pretty_balanced" and args.output_root == str(RESULT_ROOT):
        args.output_root = str(PRETTY_RESULT_ROOTS[args.learning_rate_profile])
    if args.phase.startswith("grid-") and args.output_root == str(RESULT_ROOT):
        args.output_root = str(NICME_PRETTY_GRID_ROOT)

    if args.phase == "preflight":
        run_preflight(args)
        return
    if args.phase == "grid-dry-run":
        rows = write_grid_plan(args)
        print(f"Wrote {len(rows)} grid configs under {Path(args.output_root)}")
        return
    if args.phase == "grid-smoke":
        rows = execute_grid_phase(args, "grid-smoke", smoke=True)
        print(f"Completed {len(rows)} grid smoke runs")
        return
    if args.phase == "grid-run":
        rows = execute_grid_phase(args, "grid-run", smoke=False)
        print(f"Completed {len(rows)} grid runs")
        return
    if args.phase == "grid-analyze":
        write_grid_analysis(args)
        return
    if args.phase == "dry-run":
        rows = write_plan(args)
        print(f"Wrote {len(rows)} dry-run configs under {Path(args.output_root) / 'dry-run'}")
        return
    if args.phase == "smoke":
        rows = execute_smoke(args)
        print(f"Completed {len(rows)} smoke runs")
        return
    if args.phase == "run":
        rows = execute_run_chain(args)
        print(f"Completed {len(rows)} long baseline runs")
        return
    if args.phase == "analyze":
        analyze(args)
        return
    if args.phase == "combined-analyze":
        combined_analyze(args)
        return


if __name__ == "__main__":
    main()
