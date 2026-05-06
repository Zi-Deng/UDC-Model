"""Camera-ready PMI-10 LR 5e-5 multi-seed comparison runner.

This runner fixes the paper-final comparison protocol:

* one fixed balanced PMI-10 no-calibration split,
* one fixed ConvNeXt-base backbone/training recipe,
* three paired seeds,
* the pilot-best NICME alpha/lambda setting,
* the existing LR 5e-5 SOTA baseline hyperparameters.

The output is intended to feed the NeurIPS table with mean +/- sample
standard deviation over seeds. It deliberately does not perform HPO.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from nicme.dataset_profiles import get_profile
from scripts import run_pmi10_sota_baselines as base

DEFAULT_OUTPUT_ROOT = Path("results/pmi10_camera_ready_lr5e5_multiseed_20260504")
DEFAULT_SEEDS = (42, 43, 44)
LEARNING_RATE_PROFILE = "lr5e5"
COMPARISON_PROFILE = "pretty_balanced"
NICME_PRIMARY_METHOD = "nicme_hybrid_pretty_a0p5_l0p07"
NICME_PRIMARY_ALPHA = 0.5
NICME_PRIMARY_LAMBDA = 0.07
TOL = 1e-12

TRAIN_METHODS = (
    base.PRETTY_ANCHOR,
    NICME_PRIMARY_METHOD,
    *base.DIRECT_BASELINES,
    *base.ADAPTATION_BASELINES,
)
REPORT_METHODS = (
    "ce_anchor_pretty",
    "ce_cost_min_inference_pretty",
    NICME_PRIMARY_METHOD,
    *base.DIRECT_BASELINES,
    *base.ADAPTATION_BASELINES,
)
METRICS = (
    "target_recall_min",
    "target_recall_macro",
    "normalized_atc",
    "atc",
    "balanced_accuracy",
    "macro_f1",
    "critical_pair_error_count",
)
HIGHER_IS_BETTER = {"target_recall_min", "target_recall_macro", "balanced_accuracy", "macro_f1"}
LOWER_IS_BETTER = {"normalized_atc", "atc", "critical_pair_error_count"}
COST_SENSITIVE_ENDPOINTS = (
    "target_recall_min",
    "target_recall_macro",
    "normalized_atc",
    "atc",
    "critical_pair_error_count",
)
NATIVE_CRASH_RETURNCODES = {-11}


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique: {seeds}")
    return seeds


def seed_root(output_root: Path, seed: int) -> Path:
    return output_root / "seeds" / f"seed_{seed}"


def display_name(method: str) -> str:
    names = {
        "ce_anchor_pretty": "CE",
        "ce_cost_min_inference_pretty": "CE + cost-min inference",
        NICME_PRIMARY_METHOD: "NICME (alpha=0.5, lambda=0.07)",
        "cost_weighted_ce": "Cost-weighted CE",
        "menon_logit_adjusted": "Menon logit adjustment",
        "balanced_softmax": "Balanced softmax",
        "class_balanced_focal": "Class-balanced focal",
        "ldam_drw": "LDAM-DRW",
        "cost_sensitive_regularized_ce": "cost-sensitive regularized CE",
        "csada": "CSADA",
        "sosr_cnn": "SOSR-CNN",
    }
    return names.get(method, method)


def build_training_config(
    phase: str,
    method: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
    anchor_path: str | None = None,
) -> dict[str, Any]:
    base_method = "nicme_hybrid_pretty" if method == NICME_PRIMARY_METHOD else method
    cfg = base.base_config(
        phase,
        base_method,
        split_dir,
        output_root,
        seed,
        epochs=1 if smoke else None,
        comparison_profile=COMPARISON_PROFILE,
        learning_rate_profile=LEARNING_RATE_PROFILE,
    )
    cfg["sota_comparison_profile"] = "camera_ready_lr5e5_multiseed"
    cfg["camera_ready"] = True
    cfg["camera_ready_seed"] = seed
    if method == NICME_PRIMARY_METHOD:
        cfg["sota_method"] = NICME_PRIMARY_METHOD
        cfg["sota_role"] = NICME_PRIMARY_METHOD
        cfg["nicme_logit_cost_scale"] = NICME_PRIMARY_ALPHA
        cfg["cs_lambda"] = NICME_PRIMARY_LAMBDA
        cfg["grid_alpha"] = NICME_PRIMARY_ALPHA
        cfg["grid_cs_lambda"] = NICME_PRIMARY_LAMBDA
        cfg["pilot_selected_hyperparameters"] = True
    if method in base.ADAPTATION_BASELINES and anchor_path:
        cfg["initial_checkpoint_path"] = anchor_path
    method_slug = base.slugify(method)
    cfg["output_dir"] = f"camera_ready_{phase}_{method_slug}_lr5e5_s{seed}"
    return cfg


def build_seed_configs(
    phase: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
    anchor_path: str | None = None,
) -> list[dict[str, Any]]:
    return [
        build_training_config(phase, method, split_dir, output_root, seed, smoke=smoke, anchor_path=anchor_path)
        for method in TRAIN_METHODS
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


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


def completed_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = base.read_ledger(path)
    return {row["method"]: row for row in rows if row.get("status") == "completed"}


def noncompleted_existing_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = base.read_ledger(path)
    return {
        row["method"]: row
        for row in rows
        if row.get("method") and row.get("status") and row.get("status") != "completed"
    }


def returncode_int(row: dict[str, Any]) -> int | None:
    value = row.get("returncode", "")
    if value in {"", None}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_native_crash(row: dict[str, Any]) -> bool:
    return returncode_int(row) in NATIVE_CRASH_RETURNCODES


def preserve_failed_attempt(row: dict[str, Any], ledger_path: Path, reason: str) -> Path:
    failed_dir = ledger_path.parent / "failed_attempts"
    failed_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(row.get("run_id") or "unknown").zfill(4)
    method = base.slugify(str(row.get("method") or "unknown"))
    config_hash = str(row.get("config_hash") or "nohash")
    prefix = f"{run_id}_{method}_{config_hash}_{stamp}"
    copied_files: dict[str, str] = {}
    for field in ("config_path", "stdout_log", "stderr_log"):
        source_text = str(row.get(field) or "")
        if not source_text:
            continue
        source = Path(source_text)
        if not source.exists():
            continue
        suffix = "".join(source.suffixes) or ".artifact"
        destination = failed_dir / f"{prefix}.{field}{suffix}"
        shutil.copy2(source, destination)
        copied_files[field] = str(destination)

    record = {
        "preserved_time": dt.datetime.now().isoformat(timespec="seconds"),
        "preserve_reason": reason,
        "original_ledger_path": str(ledger_path),
        "row": row,
        "copied_files": copied_files,
    }
    manifest_path = failed_dir / f"{prefix}.json"
    base.write_json(manifest_path, record)

    csv_path = failed_dir / "failed_attempts.csv"
    fields = [*base.LEDGER_FIELDS, "preserved_time", "preserve_reason", "original_ledger_path", "manifest_path"]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                **{field: row.get(field, "") for field in base.LEDGER_FIELDS},
                "preserved_time": record["preserved_time"],
                "preserve_reason": reason,
                "original_ledger_path": str(ledger_path),
                "manifest_path": str(manifest_path),
            }
        )
    return manifest_path


def execute_config_with_camera_retry(
    cfg: dict[str, Any],
    phase: str,
    run_id: int,
    args: argparse.Namespace,
    ledger_path: Path,
) -> dict[str, Any]:
    row = base.execute_config(cfg, phase, run_id, args)
    if phase != "run" or row.get("status") == "completed" or not is_native_crash(row):
        return row

    preserve_failed_attempt(row, ledger_path, "native_returncode_-11_retry")
    retry_cfg = dict(cfg)
    retry_cfg["output_dir"] = f"{cfg['output_dir']}_native_retry2"
    retry_cfg["camera_ready_retry_attempt"] = 2
    retry_cfg["camera_ready_retry_reason"] = "native_returncode_-11"
    retry_row = base.execute_config(retry_cfg, phase, run_id, args)
    if retry_row.get("status") == "completed":
        retry_row["failure_reason"] = "succeeded_after_native_returncode_-11_retry"
    elif is_native_crash(retry_row):
        preserve_failed_attempt(retry_row, ledger_path, "native_returncode_-11_retry_failed")
    return retry_row


def execute_seed_phase(args: argparse.Namespace, seed: int, phase: str, smoke: bool = False) -> list[dict[str, Any]]:
    root = seed_root(Path(args.output_root), seed)
    split_dir = Path(args.split_dir)
    ledger_path = root / phase / "run_ledger.csv"
    existing = completed_existing_rows(ledger_path)
    previous_failures = noncompleted_existing_rows(ledger_path)
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
            anchor_path=anchor_path if method in base.ADAPTATION_BASELINES else None,
        )
        if method in existing:
            row = existing[method]
        else:
            if method in previous_failures:
                preserve_failed_attempt(previous_failures[method], ledger_path, "resume_preserved_previous_noncompleted")
            row = execute_config_with_camera_retry(cfg, phase, idx, seed_args, ledger_path)
        rows.append(row)
        base.write_ledger(ledger_path, rows)
        if row.get("status") != "completed":
            raise RuntimeError(f"{phase} failed for seed={seed}, method={method}: {row.get('failure_reason')}")
        if method == base.PRETTY_ANCHOR:
            anchor_path = row["checkpoint_path"]
    return rows


def execute_phase(args: argparse.Namespace, phase: str, smoke: bool = False) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows = execute_seed_phase(args, seed, phase, smoke=smoke)
        all_rows.extend({"seed": seed, **row} for row in rows)
    return all_rows


def config_metadata(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        cfg = json.load(f)
    return {
        "comparison_profile": cfg.get("sota_comparison_profile", ""),
        "lr_profile": cfg.get("learning_rate_profile", ""),
        "learning_rate": cfg.get("learning_rate", ""),
        "parent_learning_rate": cfg.get("parent_learning_rate", ""),
        "alpha": cfg.get("nicme_logit_cost_scale", ""),
        "cs_lambda": cfg.get("cs_lambda", ""),
        "loss_function": cfg.get("loss_function", ""),
        "num_train_epochs": cfg.get("num_train_epochs", ""),
        "early_stopping_patience": cfg.get("early_stopping_patience", ""),
        "batch_size": cfg.get("batch_size", ""),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", ""),
        "weight_decay": cfg.get("weight_decay", ""),
        "warmup_ratio": cfg.get("warmup_ratio", ""),
        "image_size": cfg.get("image_size", ""),
        "model": cfg.get("model", ""),
    }


def metric_row_from_ledger(seed: int, row: dict[str, Any], method: str | None = None) -> dict[str, Any]:
    method = method or row["method"]
    metadata = config_metadata(row.get("config_path", ""))
    mode, report = base.load_metric_report(row["metric_path"], method)
    metric_row = base.metric_row(
        method,
        method,
        row["metric_path"],
        mode,
        report,
        f"camera_ready_seed_{seed}",
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
        raise RuntimeError(f"Seed {seed} has incomplete methods: {missing}")

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
        if method == base.PRETTY_ANCHOR:
            metric_rows.append(metric_row_from_ledger(seed, row, "ce_cost_min_inference_pretty"))
    return metric_rows


def to_float(value: Any) -> float:
    if value in {"", None}:
        return math.nan
    return float(value)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


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


def composite_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -float(row.get("target_recall_min_mean") or 0.0),
        float(row.get("normalized_atc_mean") or 1.0),
        -float(row.get("balanced_accuracy_mean") or 0.0),
        -float(row.get("macro_f1_mean") or 0.0),
    )


def endpoint_winners(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate_rows = sorted(aggregate_rows, key=composite_sort_key)
    winners: list[dict[str, Any]] = []
    for metric in COST_SENSITIVE_ENDPOINTS:
        field = f"{metric}_mean"
        values = [float(row[field]) for row in aggregate_rows if row.get(field) not in {"", None}]
        if not values:
            continue
        best = max(values) if metric in HIGHER_IS_BETTER else min(values)
        metric_winners = [
            row
            for row in aggregate_rows
            if row.get(field) not in {"", None} and abs(float(row[field]) - best) <= TOL
        ]
        for row in metric_winners:
            winners.append(
                {
                    "endpoint": metric,
                    "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                    "method": row["method"],
                    "display_name": row["display_name"],
                    "value": best,
                    "is_nicme_primary": row["method"] == NICME_PRIMARY_METHOD,
                }
            )
    composite_winner = aggregate_rows[0]
    winners.append(
        {
            "endpoint": "composite_recall_first_cost_sensitive_rank",
            "direction": "rank",
            "method": composite_winner["method"],
            "display_name": composite_winner["display_name"],
            "value": 1,
            "is_nicme_primary": composite_winner["method"] == NICME_PRIMARY_METHOD,
        }
    )
    return winners


def format_mean_std(row: dict[str, Any], metric: str, precision: int = 4) -> str:
    mean_value = row.get(f"{metric}_mean", "")
    std_value = row.get(f"{metric}_std", "")
    if mean_value in {"", None}:
        return ""
    if metric == "critical_pair_error_count":
        total = row.get(f"{metric}_total", "")
        return f"{float(mean_value):.{precision}f} +/- {float(std_value):.{precision}f} (total {total})"
    return f"{float(mean_value):.{precision}f} +/- {float(std_value):.{precision}f}"


def format_tex_mean_std(row: dict[str, Any], metric: str, precision: int = 4) -> str:
    if row.get(f"{metric}_mean", "") in {"", None}:
        return ""
    if metric == "critical_pair_error_count":
        mean_value = row.get(f"{metric}_mean", "")
        std_value = row.get(f"{metric}_std", "")
        total = row.get(f"{metric}_total", "")
        return f"${float(mean_value):.{precision}f} \\pm {float(std_value):.{precision}f}$ ({total})"
    mean_value = row.get(f"{metric}_mean", "")
    std_value = row.get(f"{metric}_std", "")
    return f"${float(mean_value):.{precision}f} \\pm {float(std_value):.{precision}f}$"


def write_neurips_tables(output_root: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    md_lines = [
        "# PMI-10 Camera-Ready Multi-Seed Table",
        "",
        "Metrics are mean +/- sample standard deviation over seeds 42, 43, and 44 on one fixed split.",
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
    (output_root / "analysis" / "neurips_table.md").write_text("\n".join(md_lines) + "\n")

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
    (output_root / "analysis" / "neurips_table.tex").write_text("\n".join(tex_lines) + "\n")


def write_winner_report(output_root: Path, winners: list[dict[str, Any]]) -> None:
    lines = [
        "# Cost-Sensitive Endpoint Winners",
        "",
        "Endpoints were predeclared before the camera-ready multi-seed run. Non-NICME winners are reported explicitly.",
        "",
        "| Endpoint | Direction | Winner | Value | NICME primary? |",
        "|---|---|---|---:|---|",
    ]
    for row in winners:
        value = row["value"]
        value_text = f"{float(value):.6f}" if isinstance(value, float) else str(value)
        lines.append(
            f"| {row['endpoint']} | {row['direction']} | {row['display_name']} | {value_text} | "
            f"{'yes' if row['is_nicme_primary'] else 'no'} |"
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
        if any(row["is_nicme_primary"] for row in endpoint_rows):
            verb = "tied for" if len(endpoint_rows) > 1 else "achieved"
            supported.append(f"- NICME {verb} {claim}.")
        else:
            winners_text = ", ".join(row["display_name"] for row in endpoint_rows) or "none"
            unsupported.append(f"- NICME did not win {claim}; winner: {winners_text}.")

    lines = [
        "# Camera-Ready Claim Audit",
        "",
        "Primary NICME config: alpha `0.5`, lambda `0.07`, LR `5e-5`, fixed from pilot HPO before this multi-seed rerun.",
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
        "Do not claim universal superiority unless NICME wins every endpoint in the aggregate table.",
    ]
    (output_root / "analysis" / "claim_audit.md").write_text("\n".join(lines) + "\n")


def write_paired_deltas(output_root: Path, metric_rows: list[dict[str, Any]]) -> None:
    by_seed_method = {(int(row["seed"]), row["method"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in metric_rows}):
        nicme = by_seed_method.get((seed, NICME_PRIMARY_METHOD))
        csada = by_seed_method.get((seed, "csada"))
        if not nicme or not csada:
            continue
        for metric in METRICS:
            n_val = to_float(nicme.get(metric))
            c_val = to_float(csada.get(metric))
            if math.isnan(n_val) or math.isnan(c_val):
                continue
            improvement = n_val - c_val if metric in HIGHER_IS_BETTER else c_val - n_val
            rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "nicme": n_val,
                    "csada": c_val,
                    "improvement_positive_favors_nicme": improvement,
                    "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                }
            )
    fields = ["seed", "metric", "nicme", "csada", "improvement_positive_favors_nicme", "direction"]
    write_csv(output_root / "analysis" / "paired_deltas_vs_csada.csv", rows, fields)


def write_method_hyperparameters(output_root: Path, metric_rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    lines = [
        "# PMI-10 Camera-Ready Method Hyperparameters",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "- Shared LR profile: `lr5e5`.",
        "- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.",
        "- Shared split: balanced PMI-10 no-calibration.",
        "",
        "| Method | LR | Parent LR | Alpha | Lambda | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(metric_rows, key=lambda item: REPORT_METHODS.index(item["method"]) if item["method"] in REPORT_METHODS else 999):
        method = row["method"]
        if method in seen:
            continue
        seen.add(method)
        notes = ""
        if method == NICME_PRIMARY_METHOD:
            notes = "Pilot-best fixed config; no post-rerun HPO"
        elif method == "ce_cost_min_inference_pretty":
            notes = "Inference-only report from CE anchor"
        elif method in base.ADAPTATION_BASELINES:
            notes = "CE-anchor adaptation; native adaptation LR `1e-5`"
        lines.append(
            f"| {display_name(method)} | `{row.get('learning_rate', '')}` | `{row.get('parent_learning_rate', '')}` | "
            f"`{row.get('alpha', '')}` | `{row.get('cs_lambda', '')}` | {notes} |"
        )
    (output_root / "analysis" / "method_hyperparameters.md").write_text("\n".join(lines) + "\n")


def write_readme(output_root: Path, args: argparse.Namespace, aggregate_rows: list[dict[str, Any]]) -> None:
    top = aggregate_rows[0] if aggregate_rows else {}
    lines = [
        "# PMI-10 Camera-Ready LR 5e-5 Multi-Seed Results",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "This folder contains the paper-final fixed-hyperparameter PMI-10 comparison for the NICME NeurIPS table.",
        "",
        "## Protocol",
        "",
        f"- Seeds: `{','.join(str(seed) for seed in args.seeds)}`",
        "- Split: `data/prepared/pmi_pills_10_no_cal/splits/balanced`",
        "- Backbone: `timm/convnext_base.fb_in22k_ft_in1k`",
        "- LR profile: `lr5e5`",
        "- Primary NICME: alpha `0.5`, lambda `0.07`, fixed from pilot HPO before this rerun.",
        "",
        "## Main Outputs",
        "",
        "- `analysis/aggregate_metrics.csv`",
        "- `analysis/neurips_table.md`",
        "- `analysis/neurips_table.tex`",
        "- `analysis/cost_sensitive_winners.md`",
        "- `analysis/paired_deltas_vs_csada.csv`",
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

    per_seed_fields = [
        "seed",
        "method",
        "role",
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
    write_neurips_tables(output_root, aggregate_rows)
    write_winner_report(output_root, winners)
    write_claim_audit(output_root, winners)
    write_paired_deltas(output_root, metric_rows)
    write_method_hyperparameters(output_root, metric_rows)
    write_readme(output_root, args, aggregate_rows)
    print(f"Wrote camera-ready analysis under {output_root / 'analysis'}")


def run_preflight(args: argparse.Namespace) -> None:
    split_dir = Path(args.split_dir)
    base.assert_split_ready(split_dir)
    if not args.skip_nvidia_smi:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
    profile = get_profile(base.DATASET)
    cfg = build_training_config("preflight", NICME_PRIMARY_METHOD, split_dir, Path(args.output_root), args.seeds[0])
    if cfg["cost_matrix_sha256"] != profile.cost_matrix_sha256 or base.cost_matrix_hash(cfg) != profile.cost_matrix_sha256:
        raise ValueError("PMI-10 cost matrix hash mismatch")
    if cfg["model"] != f"timm/{base.MODEL_ID}" or cfg["variant"] != base.VARIANT or cfg["calibration_enabled"]:
        raise ValueError("Camera-ready PMI-10 config invariants failed")
    print("Camera-ready PMI-10 preflight passed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preflight", "dry-run", "smoke", "run", "analyze"], required=True)
    parser.add_argument("--split-dir", default=str(base.SPLIT_DIR))
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
