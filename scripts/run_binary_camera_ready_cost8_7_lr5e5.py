"""Camera-ready binary LR 5e-5 multi-seed SOTA comparison runner.

This runner applies the PMI-20 camera-ready ConvNeXt-base protocol to the
balanced Spider and BreaKHis binary datasets, using the evidence-derived
integer cost matrices selected on 2026-05-05.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nicme.dataset_profiles import get_profile
from scripts import run_pmi10_camera_ready_lr5e5 as camera
from scripts import run_pmi10_sota_baselines as base

DEFAULT_OUTPUT_ROOT = Path("results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505")
DEFAULT_CHECKPOINT_ROOT = Path("checkpoints/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505")
DEFAULT_SEEDS = (42, 43, 44)
LEARNING_RATE_PROFILE = "lr5e5"
COMPARISON_PROFILE = "binary_camera_ready_cost8_7_lr5e5_multiseed"
MODEL_TYPE = "binary_sota_convnext_base"
VARIANT = "balanced"
TOL = 1e-12

SPIDER_COST_HASH = "8e2443d43cadb3596d502d689ff24e73b9f5663f409c5ddd43e48cc70442b753"
BREAKHIS_COST_HASH = "2a349b00571563d87c41c05572ebc2ce0b2446142bc29da08f302ae4229af11b"

CE_METHOD = "ce_anchor_binary"
CE_COST_MIN_METHOD = "ce_cost_min_inference_binary"
NICME_METHOD = "nicme_a0p5_l0p1_binary"
NICME_ALPHA = 0.5
NICME_LAMBDA = 0.1

TRAIN_METHODS = (
    CE_METHOD,
    NICME_METHOD,
    "menon_logit_adjusted",
    "cost_weighted_ce",
    "cost_sensitive_regularized_ce",
    "csada",
)
REPORT_METHODS = (
    CE_METHOD,
    CE_COST_MIN_METHOD,
    NICME_METHOD,
    "menon_logit_adjusted",
    "cost_weighted_ce",
    "cost_sensitive_regularized_ce",
    "csada",
)
ADAPTATION_METHODS = ("cost_sensitive_regularized_ce", "csada")

METRICS = (
    "target_recall_min",
    "target_recall_macro",
    "normalized_atc",
    "atc",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "critical_pair_error_count",
)
HIGHER_IS_BETTER = {"target_recall_min", "target_recall_macro", "accuracy", "balanced_accuracy", "macro_f1"}
LOWER_IS_BETTER = {"normalized_atc", "atc", "critical_pair_error_count"}
COST_SENSITIVE_ENDPOINTS = (
    "target_recall_min",
    "target_recall_macro",
    "normalized_atc",
    "atc",
    "critical_pair_error_count",
)


@dataclass(frozen=True)
class BinaryDatasetSpec:
    key: str
    profile_name: str
    split_dir: Path
    expected_counts: dict[str, int]
    required_cost_hash: str
    display_name: str


DATASETS: dict[str, BinaryDatasetSpec] = {
    "spider": BinaryDatasetSpec(
        key="spider",
        profile_name="spider",
        split_dir=Path("data/prepared/spider/splits/balanced"),
        expected_counts={"train": 2098, "validation": 300, "test": 300},
        required_cost_hash=SPIDER_COST_HASH,
        display_name="Spider",
    ),
    "breakhis": BinaryDatasetSpec(
        key="breakhis",
        profile_name="breakhis",
        split_dir=Path("data/prepared/breakhis/splits/balanced"),
        expected_counts={"train": 3448, "validation": 410, "test": 564},
        required_cost_hash=BREAKHIS_COST_HASH,
        display_name="BreaKHis",
    ),
}


def parse_seeds(value: str) -> tuple[int, ...]:
    return camera.parse_seeds(value)


def parse_datasets(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        datasets = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        datasets = tuple(value)
    if not datasets:
        raise ValueError("At least one dataset is required")
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown binary datasets: {unknown}; valid: {sorted(DATASETS)}")
    if len(set(datasets)) != len(datasets):
        raise ValueError(f"Datasets must be unique: {datasets}")
    return datasets


def dataset_root(output_root: Path, dataset_key: str) -> Path:
    return output_root / "datasets" / dataset_key


def seed_root(output_root: Path, dataset_key: str, seed: int) -> Path:
    return dataset_root(output_root, dataset_key) / "seeds" / f"seed_{seed}"


def display_name(method: str) -> str:
    names = {
        CE_METHOD: "CE",
        CE_COST_MIN_METHOD: "CE + cost-min inference",
        NICME_METHOD: "NICME (alpha=0.5, lambda=0.1)",
        "menon_logit_adjusted": "Menon logit adjustment",
        "cost_weighted_ce": "Cost-weighted CE",
        "cost_sensitive_regularized_ce": "cost-sensitive regularized CE",
        "csada": "CSADA",
    }
    return names.get(method, method)


def split_counts(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with open(path) as f:
            counts[split] = sum(1 for _ in f) - 1
    return counts


def class_counts(split_dir: Path, split: str, class_names: tuple[str, ...]) -> dict[str, int]:
    path = split_dir / f"{split}.csv"
    counts = {name: 0 for name in class_names}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label_name", "")
            if not label and row.get("label", "").isdigit():
                label = class_names[int(row["label"])]
            if label not in counts:
                raise ValueError(f"{path} has unexpected label {label!r}")
            counts[label] += 1
    return counts


def assert_split_ready(spec: BinaryDatasetSpec) -> None:
    profile = get_profile(spec.profile_name)
    counts = split_counts(spec.split_dir)
    if counts != spec.expected_counts:
        raise ValueError(f"{spec.split_dir} counts {counts}; expected {spec.expected_counts}")
    for split, total in spec.expected_counts.items():
        counts_by_class = class_counts(spec.split_dir, split, profile.class_names)
        expected_per_class = total // len(profile.class_names)
        expected = {name: expected_per_class for name in profile.class_names}
        if counts_by_class != expected:
            raise ValueError(f"{spec.split_dir / f'{split}.csv'} class counts {counts_by_class}; expected {expected}")


def base_config(
    phase: str,
    dataset_key: str,
    method: str,
    output_root: Path,
    seed: int,
    epochs: int | None = None,
    anchor_path: str | None = None,
) -> dict[str, Any]:
    spec = DATASETS[dataset_key]
    profile = get_profile(spec.profile_name)
    selection = profile.balanced_selection
    defaults = dict(base.PRETTY_DEFAULTS)
    defaults["learning_rate"] = base.LEARNING_RATE_PROFILES[LEARNING_RATE_PROFILE]
    cfg: dict[str, Any] = {
        "dataset_profile": profile.name,
        "variant": VARIANT,
        "dataset": f"{profile.name}_{VARIANT}",
        "dataset_host": "local_folder",
        "local_dataset_format": "manifest",
        "local_folder_path": str(spec.split_dir),
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
        "model": f"timm/{base.MODEL_ID}",
        "model_backend": "timm",
        "model_type": MODEL_TYPE,
        "peft_enabled": False,
        "timm_pretrained": True,
        "wandb": False,
        "push_to_hub": False,
        "calibration_enabled": False,
        "decision_modes": "argmax,cost_min",
        "prediction_mode": "argmax",
        "model_selection_metric": "target_recall",
        "checkpoint_root": str(DEFAULT_CHECKPOINT_ROOT),
        "save_total_limit": 1,
        "save_only_model": True,
        "skip_visualizations": True,
        "seed": seed,
        "sota_phase": phase,
        "sota_method": method,
        "sota_role": method,
        "sota_comparison_profile": COMPARISON_PROFILE,
        "learning_rate_profile": LEARNING_RATE_PROFILE,
        "parent_learning_rate": defaults["learning_rate"],
        "sota_output_root": str(output_root),
        **defaults,
    }
    if epochs is not None:
        cfg["num_train_epochs"] = int(epochs)

    if method == CE_METHOD:
        cfg["loss_function"] = "cross_entropy"
    elif method == NICME_METHOD:
        cfg["loss_function"] = "nicme_hybrid"
        cfg["nicme_logit_cost_scale"] = NICME_ALPHA
        cfg["cs_lambda"] = NICME_LAMBDA
        cfg["cs_warmup_epochs"] = 2
        cfg["grid_alpha"] = NICME_ALPHA
        cfg["grid_cs_lambda"] = NICME_LAMBDA
        cfg["paper_binary_primary"] = True
    elif method == "menon_logit_adjusted":
        cfg["loss_function"] = "menon_logit_adjusted"
        cfg["logit_adjustment_tau"] = 1.0
    elif method == "cost_weighted_ce":
        cfg["loss_function"] = "cost_weighted_ce"
    elif method == "cost_sensitive_regularized_ce":
        cfg["loss_function"] = "cost_sensitive_regularized_ce"
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
        raise ValueError(f"Unknown binary method: {method}")

    if method in ADAPTATION_METHODS and anchor_path:
        cfg["initial_checkpoint_path"] = anchor_path
    cfg["output_dir"] = f"binary_{dataset_key}_{phase}_{base.slugify(method)}_lr5e5_s{seed}"
    return cfg


def build_training_config(
    phase: str,
    dataset_key: str,
    method: str,
    output_root: Path,
    seed: int,
    smoke: bool = False,
    anchor_path: str | None = None,
) -> dict[str, Any]:
    return base_config(
        phase,
        dataset_key,
        method,
        output_root,
        seed,
        epochs=1 if smoke else None,
        anchor_path=anchor_path,
    )


def build_seed_configs(phase: str, dataset_key: str, output_root: Path, seed: int, smoke: bool = False) -> list[dict[str, Any]]:
    return [build_training_config(phase, dataset_key, method, output_root, seed, smoke=smoke) for method in TRAIN_METHODS]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    camera.write_csv(path, rows, fields)


def write_dry_run(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    all_rows: list[dict[str, Any]] = []
    for dataset_key in parse_datasets(args.datasets):
        for seed in args.seeds:
            root = seed_root(output_root, dataset_key, seed)
            rows: list[dict[str, Any]] = []
            configs = build_seed_configs("dry-run", dataset_key, root, seed)
            for idx, cfg in enumerate(configs, start=1):
                row = base.row_from_config(cfg, "dry-run", idx)
                cfg_hash = row["config_hash"]
                config_path = root / "dry-run" / "configs" / f"{idx:04d}_{cfg_hash}.json"
                base.write_json(config_path, cfg)
                row["config_path"] = str(config_path)
                rows.append(row)
                all_rows.append({"dataset_key": dataset_key, "seed": seed, **row})
            base.write_ledger(root / "dry-run" / "run_ledger.csv", rows)
    write_csv(output_root / "dry_run_plan.csv", all_rows, ["dataset_key", "seed", *base.LEDGER_FIELDS])
    return all_rows


def execute_dataset_seed_phase(
    args: argparse.Namespace,
    dataset_key: str,
    seed: int,
    phase: str,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    root = seed_root(Path(args.output_root), dataset_key, seed)
    ledger_path = root / phase / "run_ledger.csv"
    existing = camera.completed_existing_rows(ledger_path)
    previous_failures = camera.noncompleted_existing_rows(ledger_path)
    rows: list[dict[str, Any]] = []
    anchor_path = ""
    seed_args = argparse.Namespace(**vars(args))
    seed_args.output_root = str(root)

    for idx, method in enumerate(TRAIN_METHODS, start=1):
        cfg = build_training_config(
            phase,
            dataset_key,
            method,
            root,
            seed,
            smoke=smoke,
            anchor_path=anchor_path if method in ADAPTATION_METHODS else None,
        )
        if method in existing:
            row = existing[method]
        else:
            if method in previous_failures:
                camera.preserve_failed_attempt(previous_failures[method], ledger_path, "resume_preserved_previous_noncompleted")
            row = camera.execute_config_with_camera_retry(cfg, phase, idx, seed_args, ledger_path)
        rows.append(row)
        base.write_ledger(ledger_path, rows)
        if row.get("status") != "completed":
            raise RuntimeError(f"{phase} failed for dataset={dataset_key}, seed={seed}, method={method}: {row.get('failure_reason')}")
        if method == CE_METHOD:
            anchor_path = row["checkpoint_path"]
    return rows


def execute_phase(args: argparse.Namespace, phase: str, smoke: bool = False) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for dataset_key in parse_datasets(args.datasets):
        for seed in args.seeds:
            rows = execute_dataset_seed_phase(args, dataset_key, seed, phase, smoke=smoke)
            all_rows.extend({"dataset_key": dataset_key, "seed": seed, **row} for row in rows)
    return all_rows


def config_metadata(config_path: str) -> dict[str, Any]:
    metadata = camera.config_metadata(config_path)
    if not config_path:
        return metadata
    path = Path(config_path)
    if not path.exists():
        return metadata
    with open(path) as f:
        cfg = json.load(f)
    metadata.update(
        {
            "dataset_profile": cfg.get("dataset_profile", ""),
            "dataset": cfg.get("dataset", ""),
            "num_labels": cfg.get("num_labels", ""),
            "cost_matrix_sha256": cfg.get("cost_matrix_sha256", ""),
            "loss_function": cfg.get("loss_function", ""),
            "alpha": cfg.get("nicme_logit_cost_scale", ""),
            "cs_lambda": cfg.get("cs_lambda", ""),
        }
    )
    return metadata


def metric_row_from_ledger(dataset_key: str, seed: int, row: dict[str, Any], method: str | None = None) -> dict[str, Any]:
    method = method or row["method"]
    metadata = config_metadata(row.get("config_path", ""))
    if method == CE_COST_MIN_METHOD:
        mode, report = base.load_metric_report(row["metric_path"], "ce_cost_min_inference")
    else:
        mode, report = base.load_metric_report(row["metric_path"], method)
    metric_row = base.metric_row(
        method,
        method,
        row["metric_path"],
        mode,
        report,
        f"binary_camera_ready_{dataset_key}_seed_{seed}",
        metadata,
    )
    metric_row.update(
        {
            "dataset_key": dataset_key,
            "dataset_display": DATASETS[dataset_key].display_name,
            "seed": seed,
            "config_path": row.get("config_path", ""),
            "checkpoint_path": row.get("checkpoint_path", ""),
            "alpha": metadata.get("alpha", ""),
            "cs_lambda": metadata.get("cs_lambda", ""),
            "loss_function": metadata.get("loss_function", ""),
            "dataset_profile": metadata.get("dataset_profile", ""),
            "num_labels": metadata.get("num_labels", ""),
        }
    )
    return metric_row


def load_dataset_seed_metric_rows(
    output_root: Path,
    dataset_key: str,
    seed: int,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    ledger_path = seed_root(output_root, dataset_key, seed) / "run" / "run_ledger.csv"
    rows = base.read_ledger(ledger_path)
    if not rows:
        if allow_incomplete:
            return []
        raise RuntimeError(f"Missing run ledger for dataset={dataset_key}, seed={seed}: {ledger_path}")
    by_method = {row["method"]: row for row in rows}
    missing = [method for method in TRAIN_METHODS if by_method.get(method, {}).get("status") != "completed"]
    if missing and not allow_incomplete:
        raise RuntimeError(f"Dataset {dataset_key} seed {seed} has incomplete methods: {missing}")

    metric_rows: list[dict[str, Any]] = []
    for method in TRAIN_METHODS:
        row = by_method.get(method)
        if not row or row.get("status") != "completed":
            continue
        if not row.get("metric_path"):
            if allow_incomplete:
                continue
            raise RuntimeError(f"Dataset {dataset_key} seed {seed} method {method} is missing metric_path")
        metric_rows.append(metric_row_from_ledger(dataset_key, seed, row, method))
        if method == CE_METHOD:
            metric_rows.append(metric_row_from_ledger(dataset_key, seed, row, CE_COST_MIN_METHOD))
    return metric_rows


def to_float(value: Any) -> float:
    return camera.to_float(value)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def composite_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -float(row.get("target_recall_min_mean") or 0.0),
        float(row.get("normalized_atc_mean") or 1.0),
        -float(row.get("balanced_accuracy_mean") or 0.0),
        -float(row.get("macro_f1_mean") or 0.0),
    )


def aggregate_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault((row["dataset_key"], row["method"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (dataset_key, method), rows in grouped.items():
        out: dict[str, Any] = {
            "dataset_key": dataset_key,
            "dataset_display": DATASETS[dataset_key].display_name,
            "method": method,
            "display_name": display_name(method),
            "decision_mode": rows[0].get("decision_mode", ""),
            "n_seeds": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda item: int(item["seed"]))),
            "learning_rate": rows[0].get("learning_rate", ""),
            "parent_learning_rate": rows[0].get("parent_learning_rate", ""),
            "alpha": rows[0].get("alpha", ""),
            "cs_lambda": rows[0].get("cs_lambda", ""),
            "loss_function": rows[0].get("loss_function", ""),
        }
        for metric in METRICS:
            values = [to_float(row.get(metric)) for row in rows]
            values = [value for value in values if not math.isnan(value)]
            out[f"{metric}_mean"] = mean(values) if values else ""
            out[f"{metric}_std"] = sample_std(values) if values else ""
            if metric == "critical_pair_error_count":
                out[f"{metric}_total"] = int(sum(values)) if values else ""
        aggregate_rows.append(out)

    ranked_rows: list[dict[str, Any]] = []
    for dataset_key in parse_datasets(tuple(DATASETS)):
        dataset_rows = [row for row in aggregate_rows if row["dataset_key"] == dataset_key]
        dataset_rows.sort(key=composite_sort_key)
        for idx, row in enumerate(dataset_rows, start=1):
            row["composite_rank"] = idx
        ranked_rows.extend(dataset_rows)
    return ranked_rows


def endpoint_winners(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    winners: list[dict[str, Any]] = []
    for dataset_key in sorted({row["dataset_key"] for row in aggregate_rows}):
        dataset_rows = sorted([row for row in aggregate_rows if row["dataset_key"] == dataset_key], key=composite_sort_key)
        for metric in COST_SENSITIVE_ENDPOINTS:
            field = f"{metric}_mean"
            values = [float(row[field]) for row in dataset_rows if row.get(field) not in {"", None}]
            if not values:
                continue
            best = max(values) if metric in HIGHER_IS_BETTER else min(values)
            for row in dataset_rows:
                if row.get(field) not in {"", None} and abs(float(row[field]) - best) <= TOL:
                    winners.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset_display": DATASETS[dataset_key].display_name,
                            "endpoint": metric,
                            "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                            "method": row["method"],
                            "display_name": row["display_name"],
                            "value": best,
                            "is_nicme": row["method"] == NICME_METHOD,
                        }
                    )
        if dataset_rows:
            composite = dataset_rows[0]
            winners.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_display": DATASETS[dataset_key].display_name,
                    "endpoint": "composite_recall_first_cost_sensitive_rank",
                    "direction": "rank",
                    "method": composite["method"],
                    "display_name": composite["display_name"],
                    "value": 1,
                    "is_nicme": composite["method"] == NICME_METHOD,
                }
            )
    return winners


def format_mean_std(row: dict[str, Any], metric: str, precision: int = 4) -> str:
    return camera.format_mean_std(row, metric, precision)


def format_tex_mean_std(row: dict[str, Any], metric: str, precision: int = 4) -> str:
    return camera.format_tex_mean_std(row, metric, precision)


def table_lines_for_dataset(dataset_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rank | Method | Target Recall | Norm. ATC | ATC | Accuracy | Balanced Acc. | Macro F1 | Critical Errors |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_rows:
        lines.append(
            "| "
            f"{row['composite_rank']} | {row['display_name']} | "
            f"{format_mean_std(row, 'target_recall_min')} | "
            f"{format_mean_std(row, 'normalized_atc', 6)} | "
            f"{format_mean_std(row, 'atc', 6)} | "
            f"{format_mean_std(row, 'accuracy')} | "
            f"{format_mean_std(row, 'balanced_accuracy')} | "
            f"{format_mean_std(row, 'macro_f1')} | "
            f"{format_mean_std(row, 'critical_pair_error_count')} total {row.get('critical_pair_error_count_total', '')} |"
        )
    return lines


def write_sota_tables(output_root: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    sections = [
        "# Binary Camera-Ready SOTA Tables",
        "",
        "Metrics are mean +/- sample standard deviation over seeds 42, 43, and 44 on fixed balanced splits.",
        "",
    ]
    for dataset_key in parse_datasets(tuple(DATASETS)):
        dataset_rows = sorted([row for row in aggregate_rows if row["dataset_key"] == dataset_key], key=lambda row: int(row["composite_rank"]))
        title = DATASETS[dataset_key].display_name
        lines = [f"# {title} Binary SOTA Table", "", *table_lines_for_dataset(dataset_rows), ""]
        (output_root / "analysis" / f"{dataset_key}_sota_table.md").write_text("\n".join(lines))
        sections.extend([f"## {title}", "", *table_lines_for_dataset(dataset_rows), ""])
    (output_root / "analysis" / "binary_sota_table.md").write_text("\n".join(sections))

    tex_lines = [
        "\\begin{tabular}{rlllllllll}",
        "\\toprule",
        "Dataset & Rank & Method & Target recall & Norm. ATC & ATC & Acc. & Bal. Acc. & Macro F1 & Critical \\\\",
        "\\midrule",
    ]
    for row in sorted(aggregate_rows, key=lambda item: (item["dataset_key"], int(item["composite_rank"]))):
        tex_lines.append(
            f"{row['dataset_display']} & {row['composite_rank']} & {row['display_name']} & "
            f"{format_tex_mean_std(row, 'target_recall_min')} & "
            f"{format_tex_mean_std(row, 'normalized_atc', 6)} & "
            f"{format_tex_mean_std(row, 'atc', 6)} & "
            f"{format_tex_mean_std(row, 'accuracy')} & "
            f"{format_tex_mean_std(row, 'balanced_accuracy')} & "
            f"{format_tex_mean_std(row, 'macro_f1')} & "
            f"{format_tex_mean_std(row, 'critical_pair_error_count')} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_root / "analysis" / "binary_sota_table.tex").write_text("\n".join(tex_lines) + "\n")


def write_winner_report(output_root: Path, winners: list[dict[str, Any]]) -> None:
    lines = [
        "# Binary Cost-Sensitive Endpoint Winners",
        "",
        "| Dataset | Endpoint | Direction | Winner | Value | NICME? |",
        "|---|---|---|---|---:|---|",
    ]
    for row in winners:
        value = row["value"]
        value_text = f"{float(value):.6f}" if isinstance(value, float) else str(value)
        lines.append(
            f"| {row['dataset_display']} | {row['endpoint']} | {row['direction']} | {row['display_name']} | "
            f"{value_text} | {'yes' if row['is_nicme'] else 'no'} |"
        )
    (output_root / "analysis" / "cost_sensitive_winners.md").write_text("\n".join(lines) + "\n")


def write_claim_audit(output_root: Path, winners: list[dict[str, Any]]) -> None:
    lines = [
        "# Binary Camera-Ready Claim Audit",
        "",
        "Primary NICME config: alpha `0.5`, lambda `0.1`, LR `5e-5`.",
        "",
        "## Supported Claims By Dataset",
        "",
    ]
    for dataset_key in parse_datasets(tuple(DATASETS)):
        dataset_winners = [row for row in winners if row["dataset_key"] == dataset_key]
        nicme_wins = [row for row in dataset_winners if row["is_nicme"]]
        lines.append(f"### {DATASETS[dataset_key].display_name}")
        if nicme_wins:
            for row in nicme_wins:
                lines.append(f"- NICME won or tied `{row['endpoint']}`.")
        else:
            lines.append("- NICME did not win a predeclared cost-sensitive endpoint in this aggregate.")
        non_nicme = [row for row in dataset_winners if not row["is_nicme"]]
        if non_nicme:
            lines.append("- Non-NICME endpoint wins are preserved in `cost_sensitive_winners.md`.")
        lines.append("")
    lines.extend(
        [
            "## Guardrails",
            "",
            "- Do not average Spider and BreaKHis into one headline rank.",
            "- Do not claim external global SOTA.",
            "- Error bars are sample standard deviations over training seeds on one fixed split per dataset.",
        ]
    )
    (output_root / "analysis" / "claim_audit.md").write_text("\n".join(lines) + "\n")


def write_paired_deltas(output_root: Path, metric_rows: list[dict[str, Any]], baseline_method: str, file_name: str) -> None:
    by_dataset_seed_method = {(row["dataset_key"], int(row["seed"]), row["method"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for dataset_key in sorted({row["dataset_key"] for row in metric_rows}):
        for seed in sorted({int(row["seed"]) for row in metric_rows if row["dataset_key"] == dataset_key}):
            nicme = by_dataset_seed_method.get((dataset_key, seed, NICME_METHOD))
            baseline_row = by_dataset_seed_method.get((dataset_key, seed, baseline_method))
            if not nicme or not baseline_row:
                continue
            for metric in METRICS:
                n_val = to_float(nicme.get(metric))
                b_val = to_float(baseline_row.get(metric))
                if math.isnan(n_val) or math.isnan(b_val):
                    continue
                improvement = n_val - b_val if metric in HIGHER_IS_BETTER else b_val - n_val
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "seed": seed,
                        "metric": metric,
                        "nicme": n_val,
                        "baseline_method": baseline_method,
                        "baseline": b_val,
                        "improvement_positive_favors_nicme": improvement,
                        "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                    }
                )
    write_csv(
        output_root / "analysis" / file_name,
        rows,
        ["dataset_key", "seed", "metric", "nicme", "baseline_method", "baseline", "improvement_positive_favors_nicme", "direction"],
    )


def write_method_hyperparameters(output_root: Path, metric_rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    lines = [
        "# Binary Camera-Ready Method Hyperparameters",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "- Shared LR profile: `lr5e5`.",
        "- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.",
        "- Shared splits: balanced Spider and balanced BreaKHis.",
        "- Primary matrices: Spider `[[0,8],[1,0]]`; BreaKHis `[[0,1],[7,0]]`.",
        "",
        "| Method | LR | Parent LR | Alpha | Lambda | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    order = {method: idx for idx, method in enumerate(REPORT_METHODS)}
    for row in sorted(metric_rows, key=lambda item: order.get(item["method"], 999)):
        method = row["method"]
        if method in seen:
            continue
        seen.add(method)
        notes = ""
        if method == NICME_METHOD:
            notes = "Primary NICME setting from PMI-20 camera-ready protocol"
        elif method == CE_COST_MIN_METHOD:
            notes = "Inference-only report from CE anchor"
        elif method in ADAPTATION_METHODS:
            notes = "CE-anchor adaptation; native adaptation LR `1e-5`"
        lines.append(
            f"| {display_name(method)} | `{row.get('learning_rate', '')}` | `{row.get('parent_learning_rate', '')}` | "
            f"`{row.get('alpha', '')}` | `{row.get('cs_lambda', '')}` | {notes} |"
        )
    (output_root / "analysis" / "method_hyperparameters.md").write_text("\n".join(lines) + "\n")


def write_readme(output_root: Path, args: argparse.Namespace, aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Binary Camera-Ready Cost 8/7 LR 5e-5 Multi-Seed Results",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "This folder contains the fixed-hyperparameter binary SOTA/baseline comparison for Spider and BreaKHis.",
        "",
        "## Protocol",
        "",
        f"- Seeds: `{','.join(str(seed) for seed in args.seeds)}`",
        "- Splits: balanced Spider and balanced BreaKHis.",
        "- Backbone: `timm/convnext_base.fb_in22k_ft_in1k`",
        "- LR profile: `lr5e5`",
        "- NICME: alpha `0.5`, lambda `0.1`.",
        "- Spider cost matrix: `[[0,8],[1,0]]`.",
        "- BreaKHis cost matrix: `[[0,1],[7,0]]`.",
        "",
        "## Main Outputs",
        "",
        "- `analysis/per_seed_metrics.csv`",
        "- `analysis/aggregate_metrics.csv`",
        "- `analysis/binary_sota_table.md`",
        "- `analysis/spider_sota_table.md`",
        "- `analysis/breakhis_sota_table.md`",
        "- `analysis/cost_sensitive_winners.md`",
        "- `analysis/claim_audit.md`",
        "",
        "Error bars are sample standard deviations over training seeds on one fixed split per dataset.",
    ]
    for dataset_key in parse_datasets(tuple(DATASETS)):
        rows = [row for row in aggregate_rows if row["dataset_key"] == dataset_key and int(row["composite_rank"]) == 1]
        if rows:
            lines.append(f"- {DATASETS[dataset_key].display_name} rank 1: `{rows[0]['display_name']}`")
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    metric_rows: list[dict[str, Any]] = []
    for dataset_key in parse_datasets(args.datasets):
        for seed in args.seeds:
            metric_rows.extend(load_dataset_seed_metric_rows(output_root, dataset_key, seed, allow_incomplete=args.allow_incomplete_analysis))
    if not metric_rows:
        raise RuntimeError("No metric rows were available for analysis")
    for row in metric_rows:
        row["display_name"] = display_name(row["method"])

    per_seed_fields = [
        "dataset_key",
        "dataset_display",
        "seed",
        "method",
        "display_name",
        "source",
        "decision_mode",
        *METRICS,
        "learning_rate",
        "parent_learning_rate",
        "alpha",
        "cs_lambda",
        "metric_path",
        "config_path",
    ]
    write_csv(output_root / "analysis" / "per_seed_metrics.csv", metric_rows, per_seed_fields)

    aggregate_rows = aggregate_metric_rows(metric_rows)
    aggregate_fields = [
        "dataset_key",
        "dataset_display",
        "composite_rank",
        "method",
        "display_name",
        "decision_mode",
        "n_seeds",
        "seeds",
        "learning_rate",
        "parent_learning_rate",
        "alpha",
        "cs_lambda",
        "loss_function",
    ]
    for metric in METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
        if metric == "critical_pair_error_count":
            aggregate_fields.append(f"{metric}_total")
    write_csv(output_root / "analysis" / "aggregate_metrics.csv", aggregate_rows, aggregate_fields)

    winners = endpoint_winners(aggregate_rows)
    write_sota_tables(output_root, aggregate_rows)
    write_winner_report(output_root, winners)
    write_claim_audit(output_root, winners)
    write_paired_deltas(output_root, metric_rows, "csada", "paired_deltas_vs_csada.csv")
    write_paired_deltas(output_root, metric_rows, "menon_logit_adjusted", "paired_deltas_vs_menon.csv")
    write_paired_deltas(output_root, metric_rows, CE_METHOD, "paired_deltas_vs_ce.csv")
    write_method_hyperparameters(output_root, metric_rows)
    write_readme(output_root, args, aggregate_rows)
    print(f"Wrote binary camera-ready analysis under {output_root / 'analysis'}")


def assert_binary_config(dataset_key: str, cfg: dict[str, Any]) -> None:
    spec = DATASETS[dataset_key]
    profile = get_profile(spec.profile_name)
    if cfg["dataset_profile"] != spec.profile_name or cfg["num_labels"] != 2:
        raise ValueError("Binary runner must use binary dataset profiles")
    if cfg["cost_matrix_sha256"] != spec.required_cost_hash or base.cost_matrix_hash(cfg) != spec.required_cost_hash:
        raise ValueError(f"{dataset_key} cost matrix hash mismatch")
    observed_pairs = {(item["true"], item["pred"], float(item["cost"])) for item in cfg["critical_pairs"]}
    expected_pairs = set(profile.critical_pairs)
    if observed_pairs != expected_pairs:
        raise ValueError(f"{dataset_key} critical-pair set mismatch")
    if cfg["model"] != f"timm/{base.MODEL_ID}" or cfg["variant"] != VARIANT or cfg["calibration_enabled"]:
        raise ValueError("Binary camera-ready config invariants failed")
    if cfg["model_type"] != MODEL_TYPE or cfg["model_backend"] != "timm":
        raise ValueError("Binary runner must use ConvNeXt-base timm backend")
    if cfg["decision_modes"] != "argmax,cost_min":
        raise ValueError("Binary runner must use argmax,cost_min decision modes")


def run_preflight(args: argparse.Namespace) -> None:
    if not args.skip_nvidia_smi:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
    for dataset_key in parse_datasets(args.datasets):
        spec = DATASETS[dataset_key]
        assert_split_ready(spec)
        profile = get_profile(spec.profile_name)
        if profile.cost_matrix_sha256 != spec.required_cost_hash:
            raise ValueError(f"Unexpected {dataset_key} profile cost matrix hash")
        for method in TRAIN_METHODS:
            cfg = build_training_config("preflight", dataset_key, method, Path(args.output_root), args.seeds[0])
            assert_binary_config(dataset_key, cfg)
        nicme = build_training_config("preflight", dataset_key, NICME_METHOD, Path(args.output_root), args.seeds[0])
        if float(nicme["nicme_logit_cost_scale"]) != NICME_ALPHA or float(nicme["cs_lambda"]) != NICME_LAMBDA:
            raise ValueError("NICME paper hyperparameters changed")
    print("Binary camera-ready preflight passed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preflight", "dry-run", "smoke", "run", "analyze"], required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--datasets", type=parse_datasets, default=tuple(DATASETS))
    parser.add_argument("--per-run-timeout-minutes", type=int, default=720)
    parser.add_argument("--allow-incomplete-analysis", action="store_true")
    parser.add_argument("--skip-nvidia-smi", action="store_true")
    args = parser.parse_args(argv)

    if args.phase == "preflight":
        run_preflight(args)
    elif args.phase == "dry-run":
        rows = write_dry_run(args)
        print(f"Wrote {len(rows)} dry-run training configs under {Path(args.output_root)}")
    elif args.phase == "smoke":
        rows = execute_phase(args, "smoke", smoke=True)
        print(f"Completed {len(rows)} smoke training runs")
    elif args.phase == "run":
        rows = execute_phase(args, "run", smoke=False)
        print(f"Completed {len(rows)} long training runs")
    elif args.phase == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
