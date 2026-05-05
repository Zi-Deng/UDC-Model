"""Camera-ready PMI-20 LR 5e-5 multi-seed comparison runner.

This runner mirrors the PMI-10 camera-ready protocol on the full 20-class
balanced PMI pill dataset. It fixes the split, ConvNeXt-base recipe, three
paired seeds, and a reduced SOTA/baseline method set.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

from nicme.dataset_profiles import get_profile
from scripts import run_pmi10_camera_ready_lr5e5 as camera
from scripts import run_pmi10_sota_baselines as base

DATASET = "pmi_pills"
VARIANT = "balanced"
DEFAULT_SPLIT_DIR = Path("data/prepared/pmi_pills/splits/balanced")
DEFAULT_OUTPUT_ROOT = Path("results/pmi20_camera_ready_lr5e5_multiseed_20260504")
DEFAULT_SEEDS = (42, 43, 44)
LEARNING_RATE_PROFILE = "lr5e5"
COMPARISON_PROFILE = "pmi20_camera_ready_lr5e5_multiseed"
MODEL_TYPE = "pmi20_sota_convnext_base"
REQUIRED_COST_HASH = "af92dbd367c299056f5f41dab4f3b4330a6171b856e8822bac0357fdfe4289d2"
TOL = 1e-12

CE_METHOD = "ce_anchor_pmi20"
CE_COST_MIN_METHOD = "ce_cost_min_inference_pmi20"
NICME_METHOD = "nicme_a0p5_l0p1_pmi20"
NICME_ALPHA = 0.5
NICME_LAMBDA = 0.1

TRAIN_METHODS = (
    CE_METHOD,
    NICME_METHOD,
    "menon_logit_adjusted",
    "cost_weighted_ce",
    "ap_csada",
    "csada",
)
REPORT_METHODS = (
    CE_METHOD,
    CE_COST_MIN_METHOD,
    NICME_METHOD,
    "menon_logit_adjusted",
    "cost_weighted_ce",
    "ap_csada",
    "csada",
)
ADAPTATION_METHODS = ("ap_csada", "csada")
METRICS = camera.METRICS
HIGHER_IS_BETTER = camera.HIGHER_IS_BETTER
LOWER_IS_BETTER = camera.LOWER_IS_BETTER
COST_SENSITIVE_ENDPOINTS = camera.COST_SENSITIVE_ENDPOINTS


def parse_seeds(value: str) -> tuple[int, ...]:
    return camera.parse_seeds(value)


def seed_root(output_root: Path, seed: int) -> Path:
    return output_root / "seeds" / f"seed_{seed}"


def display_name(method: str) -> str:
    names = {
        CE_METHOD: "CE",
        CE_COST_MIN_METHOD: "CE + cost-min inference",
        NICME_METHOD: "NICME (alpha=0.5, lambda=0.1)",
        "menon_logit_adjusted": "Menon logit adjustment",
        "cost_weighted_ce": "Cost-weighted CE",
        "ap_csada": "AP-CSADA",
        "csada": "CSADA",
    }
    return names.get(method, method)


def split_counts(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        path = split_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing balanced PMI-20 split file: {path}")
        with open(path) as f:
            counts[split] = sum(1 for _ in f) - 1
    return counts


def assert_split_ready(split_dir: Path) -> None:
    expected = {"train": 1940, "validation": 320, "test": 640}
    counts = split_counts(split_dir)
    for split, expected_count in expected.items():
        if counts[split] != expected_count:
            raise ValueError(f"{split_dir / f'{split}.csv'} has {counts[split]} rows; expected {expected_count}")


def base_config(
    phase: str,
    method: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    epochs: int | None = None,
    anchor_path: str | None = None,
) -> dict[str, Any]:
    profile = get_profile(DATASET)
    selection = profile.balanced_selection
    defaults = dict(base.PRETTY_DEFAULTS)
    defaults["learning_rate"] = base.LEARNING_RATE_PROFILES[LEARNING_RATE_PROFILE]
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
        "save_total_limit": 1,
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
        cfg["added_fixed_candidate"] = True
        cfg["pilot_selected_hyperparameters"] = True
    elif method == "menon_logit_adjusted":
        cfg["loss_function"] = "menon_logit_adjusted"
        cfg["logit_adjustment_tau"] = 1.0
    elif method == "cost_weighted_ce":
        cfg["loss_function"] = "cost_weighted_ce"
    elif method == "ap_csada":
        cfg["loss_function"] = "ap_csada"
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
        raise ValueError(f"Unknown PMI-20 method: {method}")

    if method in ADAPTATION_METHODS and anchor_path:
        cfg["initial_checkpoint_path"] = anchor_path
    cfg["output_dir"] = f"pmi20_{phase}_{base.slugify(method)}_lr5e5_s{seed}"
    return cfg


def build_training_config(
    phase: str,
    method: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
    anchor_path: str | None = None,
) -> dict[str, Any]:
    return base_config(
        phase,
        method,
        split_dir,
        output_root,
        seed,
        epochs=1 if smoke else None,
        anchor_path=anchor_path,
    )


def build_seed_configs(
    phase: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
    anchor_path: str | None = None,
) -> list[dict[str, Any]]:
    return [
        build_training_config(
            phase,
            method,
            split_dir,
            output_root,
            seed,
            smoke=smoke,
            anchor_path=anchor_path if method in ADAPTATION_METHODS else None,
        )
        for method in TRAIN_METHODS
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    camera.write_csv(path, rows, fields)


def write_dry_run(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    split_dir = Path(args.split_dir)
    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        root = seed_root(output_root, seed)
        rows: list[dict[str, Any]] = []
        configs = build_seed_configs("dry-run", split_dir, root, seed)
        for idx, cfg in enumerate(configs, start=1):
            row = base.row_from_config(cfg, "dry-run", idx)
            cfg_hash = row["config_hash"]
            config_path = root / "dry-run" / "configs" / f"{idx:04d}_{cfg_hash}.json"
            base.write_json(config_path, cfg)
            row["config_path"] = str(config_path)
            rows.append(row)
            all_rows.append({"seed": seed, **row})
        base.write_ledger(root / "dry-run" / "run_ledger.csv", rows)
    write_csv(output_root / "dry_run_plan.csv", all_rows, ["seed", *base.LEDGER_FIELDS])
    return all_rows


def execute_seed_phase(args: argparse.Namespace, seed: int, phase: str, smoke: bool = False) -> list[dict[str, Any]]:
    root = seed_root(Path(args.output_root), seed)
    split_dir = Path(args.split_dir)
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
            method,
            split_dir,
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
            raise RuntimeError(f"{phase} failed for seed={seed}, method={method}: {row.get('failure_reason')}")
        if method == CE_METHOD:
            anchor_path = row["checkpoint_path"]
    return rows


def execute_phase(args: argparse.Namespace, phase: str, smoke: bool = False) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows = execute_seed_phase(args, seed, phase, smoke=smoke)
        all_rows.extend({"seed": seed, **row} for row in rows)
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
            "num_labels": cfg.get("num_labels", ""),
            "cost_matrix_sha256": cfg.get("cost_matrix_sha256", ""),
            "pilot_hpo_run": cfg.get("pilot_hpo_run", ""),
        }
    )
    return metadata


def metric_row_from_ledger(seed: int, row: dict[str, Any], method: str | None = None) -> dict[str, Any]:
    method = method or row["method"]
    metadata = config_metadata(row.get("config_path", ""))
    if method == CE_COST_MIN_METHOD:
        mode, report = base.load_metric_report(row["metric_path"], "ce_cost_min_inference_pretty")
    else:
        mode, report = base.load_metric_report(row["metric_path"], method)
    metric_row = base.metric_row(
        method,
        method,
        row["metric_path"],
        mode,
        report,
        f"pmi20_camera_ready_seed_{seed}",
        metadata,
    )
    metric_row.update(
        {
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


def load_seed_metric_rows(output_root: Path, seed: int, allow_incomplete: bool = False) -> list[dict[str, Any]]:
    ledger_path = seed_root(output_root, seed) / "run" / "run_ledger.csv"
    rows = base.read_ledger(ledger_path)
    if not rows:
        if allow_incomplete:
            return []
        raise RuntimeError(f"Missing run ledger for seed {seed}: {ledger_path}")
    by_method = {row["method"]: row for row in rows}
    missing = [method for method in TRAIN_METHODS if by_method.get(method, {}).get("status") != "completed"]
    if missing and not allow_incomplete:
        raise RuntimeError(f"Seed {seed} has incomplete PMI-20 methods: {missing}")

    metric_rows: list[dict[str, Any]] = []
    for method in TRAIN_METHODS:
        row = by_method.get(method)
        if not row or row.get("status") != "completed":
            continue
        if not row.get("metric_path"):
            if allow_incomplete:
                continue
            raise RuntimeError(f"Seed {seed} method {method} is missing metric_path")
        metric_rows.append(metric_row_from_ledger(seed, row, method))
        if method == CE_METHOD:
            metric_rows.append(metric_row_from_ledger(seed, row, CE_COST_MIN_METHOD))
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
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault(row["method"], []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        out: dict[str, Any] = {
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

    aggregate_rows.sort(key=composite_sort_key)
    for idx, row in enumerate(aggregate_rows, start=1):
        row["composite_rank"] = idx
    return aggregate_rows


def endpoint_winners(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate_rows = sorted(aggregate_rows, key=composite_sort_key)
    winners: list[dict[str, Any]] = []
    for metric in COST_SENSITIVE_ENDPOINTS:
        field = f"{metric}_mean"
        values = [float(row[field]) for row in aggregate_rows if row.get(field) not in {"", None}]
        if not values:
            continue
        best = max(values) if metric in HIGHER_IS_BETTER else min(values)
        for row in aggregate_rows:
            if row.get(field) not in {"", None} and abs(float(row[field]) - best) <= TOL:
                winners.append(
                    {
                        "endpoint": metric,
                        "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                        "method": row["method"],
                        "display_name": row["display_name"],
                        "value": best,
                        "is_nicme": row["method"] == NICME_METHOD,
                    }
                )
    if aggregate_rows:
        composite = aggregate_rows[0]
        winners.append(
            {
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


def write_sota_tables(output_root: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    md_lines = [
        "# PMI-20 Camera-Ready Multi-Seed SOTA Table",
        "",
        "Metrics are mean +/- sample standard deviation over seeds 42, 43, and 44 on one fixed balanced 20-class PMI split.",
        "",
        "| Rank | Method | Target-Min Recall | Target-Macro Recall | Norm. ATC | ATC | Balanced Acc. | Macro F1 | Critical Errors |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        md_lines.append(
            "| "
            f"{row['composite_rank']} | {row['display_name']} | "
            f"{format_mean_std(row, 'target_recall_min')} | "
            f"{format_mean_std(row, 'target_recall_macro')} | "
            f"{format_mean_std(row, 'normalized_atc', 6)} | "
            f"{format_mean_std(row, 'atc', 6)} | "
            f"{format_mean_std(row, 'balanced_accuracy')} | "
            f"{format_mean_std(row, 'macro_f1')} | "
            f"{format_mean_std(row, 'critical_pair_error_count')} |"
        )
    (output_root / "analysis" / "pmi20_sota_table.md").write_text("\n".join(md_lines) + "\n")

    tex_lines = [
        "\\begin{tabular}{rllllllll}",
        "\\toprule",
        "Rank & Method & Target-min & Target-macro & Norm. ATC & ATC & Bal. Acc. & Macro F1 & Critical \\\\",
        "\\midrule",
    ]
    for row in aggregate_rows:
        tex_lines.append(
            f"{row['composite_rank']} & {row['display_name']} & "
            f"{format_tex_mean_std(row, 'target_recall_min')} & "
            f"{format_tex_mean_std(row, 'target_recall_macro')} & "
            f"{format_tex_mean_std(row, 'normalized_atc', 6)} & "
            f"{format_tex_mean_std(row, 'atc', 6)} & "
            f"{format_tex_mean_std(row, 'balanced_accuracy')} & "
            f"{format_tex_mean_std(row, 'macro_f1')} & "
            f"{format_tex_mean_std(row, 'critical_pair_error_count')} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_root / "analysis" / "pmi20_sota_table.tex").write_text("\n".join(tex_lines) + "\n")


def write_winner_report(output_root: Path, winners: list[dict[str, Any]]) -> None:
    lines = [
        "# PMI-20 Cost-Sensitive Endpoint Winners",
        "",
        "Endpoints were predeclared before the PMI-20 camera-ready multi-seed run.",
        "",
        "| Endpoint | Direction | Winner | Value | NICME? |",
        "|---|---|---|---:|---|",
    ]
    for row in winners:
        value = row["value"]
        value_text = f"{float(value):.6f}" if isinstance(value, float) else str(value)
        lines.append(
            f"| {row['endpoint']} | {row['direction']} | {row['display_name']} | {value_text} | "
            f"{'yes' if row['is_nicme'] else 'no'} |"
        )
    (output_root / "analysis" / "cost_sensitive_winners.md").write_text("\n".join(lines) + "\n")


def write_claim_audit(output_root: Path, winners: list[dict[str, Any]]) -> None:
    by_endpoint: dict[str, list[dict[str, Any]]] = {}
    for row in winners:
        by_endpoint.setdefault(row["endpoint"], []).append(row)
    claim_map = {
        "composite_recall_first_cost_sensitive_rank": "best recall-first cost-sensitive tradeoff",
        "normalized_atc": "best normalized expected-cost performance",
        "atc": "best expected-cost performance",
        "critical_pair_error_count": "best critical-confusion control",
        "target_recall_min": "best cared-class minimum recall",
        "target_recall_macro": "best cared-class macro recall",
    }
    supported = []
    unsupported = []
    for endpoint, claim in claim_map.items():
        endpoint_rows = by_endpoint.get(endpoint, [])
        if any(row["is_nicme"] for row in endpoint_rows):
            verb = "tied for" if len(endpoint_rows) > 1 else "achieved"
            supported.append(f"- NICME {verb} {claim}.")
        else:
            winners_text = ", ".join(row["display_name"] for row in endpoint_rows) or "none"
            unsupported.append(f"- NICME did not win {claim}; winner: {winners_text}.")
    lines = [
        "# PMI-20 Camera-Ready Claim Audit",
        "",
        "Primary NICME config: alpha `0.5`, lambda `0.1`, LR `5e-5`, selected from the completed PMI-20 six-candidate robustness rerun.",
        "",
        "## Supported Claims",
        "",
        *(supported or ["- No predeclared cost-sensitive endpoint was won by NICME in this aggregate."]),
        "",
        "## Claims Not Supported By This Aggregate",
        "",
        *unsupported,
        "",
        "## Guardrail",
        "",
        "This is a repository SOTA/baseline comparison for selected PMI-20 methods, not an external global SOTA claim.",
    ]
    (output_root / "analysis" / "claim_audit.md").write_text("\n".join(lines) + "\n")


def write_paired_deltas(output_root: Path, metric_rows: list[dict[str, Any]], baseline_method: str, file_name: str) -> None:
    by_seed_method = {(int(row["seed"]), row["method"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in metric_rows}):
        nicme = by_seed_method.get((seed, NICME_METHOD))
        baseline_row = by_seed_method.get((seed, baseline_method))
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
        ["seed", "metric", "nicme", "baseline_method", "baseline", "improvement_positive_favors_nicme", "direction"],
    )


def write_method_hyperparameters(output_root: Path, metric_rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    lines = [
        "# PMI-20 Camera-Ready Method Hyperparameters",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "- Shared LR profile: `lr5e5`.",
        "- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.",
        "- Shared split: balanced full 20-class PMI.",
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
            notes = "Main paper NICME setting selected from the completed PMI-20 six-candidate robustness rerun"
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
    top = aggregate_rows[0] if aggregate_rows else {}
    lines = [
        "# PMI-20 Camera-Ready LR 5e-5 Multi-Seed Results",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "This folder contains the fixed-hyperparameter full 20-class PMI comparison for the selected SOTA/baseline methods.",
        "",
        "## Protocol",
        "",
        f"- Seeds: `{','.join(str(seed) for seed in args.seeds)}`",
        "- Split: `data/prepared/pmi_pills/splits/balanced`",
        "- Backbone: `timm/convnext_base.fb_in22k_ft_in1k`",
        "- LR profile: `lr5e5`",
        "- Primary NICME: alpha `0.5`, lambda `0.1`, selected from the completed PMI-20 six-candidate robustness rerun.",
        "",
        "## Main Outputs",
        "",
        "- `analysis/per_seed_metrics.csv`",
        "- `analysis/aggregate_metrics.csv`",
        "- `analysis/pmi20_sota_table.md`",
        "- `analysis/pmi20_sota_table.tex`",
        "- `analysis/cost_sensitive_winners.md`",
        "- `analysis/paired_deltas_vs_csada.csv`",
        "- `analysis/paired_deltas_vs_menon.csv`",
        "- `analysis/method_hyperparameters.md`",
        "- `analysis/claim_audit.md`",
        "",
        "## Current Composite Leader",
        "",
        f"- Rank 1: `{top.get('display_name', '')}`",
        "",
        "Error bars are sample standard deviations over training seeds on one fixed split.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    metric_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        metric_rows.extend(load_seed_metric_rows(output_root, seed, allow_incomplete=args.allow_incomplete_analysis))
    if not metric_rows:
        raise RuntimeError("No metric rows were available for analysis")
    for row in metric_rows:
        row["display_name"] = display_name(row["method"])

    per_seed_fields = [
        "seed",
        "method",
        "display_name",
        "source",
        "decision_mode",
        "target_recall_min",
        "target_recall_macro",
        "normalized_atc",
        "atc",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "critical_pair_error_count",
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
    write_method_hyperparameters(output_root, metric_rows)
    write_readme(output_root, args, aggregate_rows)
    print(f"Wrote PMI-20 camera-ready analysis under {output_root / 'analysis'}")


def assert_pmi20_config(cfg: dict[str, Any]) -> None:
    if cfg["dataset_profile"] != DATASET or cfg["num_labels"] != 20:
        raise ValueError("PMI-20 runner must use full pmi_pills profile")
    if cfg["cost_matrix_sha256"] != REQUIRED_COST_HASH or base.cost_matrix_hash(cfg) != REQUIRED_COST_HASH:
        raise ValueError("PMI-20 cost matrix hash mismatch")
    expected_pairs = {
        ("50111-0434", "00591-0461", 10.0),
        ("53489-0156", "68382-0227", 10.0),
        ("53746-0544", "00378-0208", 10.0),
        ("68382-0227", "53489-0156", 8.0),
    }
    observed_pairs = {(item["true"], item["pred"], float(item["cost"])) for item in cfg["critical_pairs"]}
    if observed_pairs != expected_pairs:
        raise ValueError("PMI-20 critical-pair set mismatch")
    if cfg["model"] != f"timm/{base.MODEL_ID}" or cfg["variant"] != VARIANT or cfg["calibration_enabled"]:
        raise ValueError("PMI-20 config invariants failed")
    if cfg["model_type"] != MODEL_TYPE or cfg["model_backend"] != "timm":
        raise ValueError("PMI-20 runner must use ConvNeXt-base timm backend")


def run_preflight(args: argparse.Namespace) -> None:
    split_dir = Path(args.split_dir)
    assert_split_ready(split_dir)
    if not args.skip_nvidia_smi:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
    profile = get_profile(DATASET)
    if profile.cost_matrix_sha256 != REQUIRED_COST_HASH:
        raise ValueError("Unexpected PMI-20 profile cost matrix hash")
    for method in TRAIN_METHODS:
        cfg = build_training_config("preflight", method, split_dir, Path(args.output_root), args.seeds[0])
        assert_pmi20_config(cfg)
    nicme = build_training_config("preflight", NICME_METHOD, split_dir, Path(args.output_root), args.seeds[0])
    if float(nicme["nicme_logit_cost_scale"]) != NICME_ALPHA or float(nicme["cs_lambda"]) != NICME_LAMBDA:
        raise ValueError("NICME paper hyperparameters changed")
    print("Camera-ready PMI-20 preflight passed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preflight", "dry-run", "smoke", "run", "analyze"], required=True)
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
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
