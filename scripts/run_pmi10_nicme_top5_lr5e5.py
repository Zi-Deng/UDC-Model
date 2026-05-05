"""PMI-10 NICME top-5 alpha/lambda multi-seed runner.

This runner repeats the five best observed LR 5e-5 NICME HPO settings
over three paired training seeds. It is a robustness/sensitivity follow-up
to the pilot HPO, not another hyperparameter search.
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

DEFAULT_OUTPUT_ROOT = Path("results/pmi10_nicme_top5_alpha_lambda_lr5e5_multiseed_20260504")
DEFAULT_SEEDS = (42, 43, 44)
COMPARISON_PROFILE = "pretty_balanced"
LEARNING_RATE_PROFILE = "lr5e5"
REQUIRED_COST_HASH = "1160cbf20c4fac24dc7bb84cbbb229a16545259d585ae1ef1c066e0007082e08"
TOL = 1e-12

TOP5_POINTS = (
    {"hpo_run": 95, "alpha": 0.50, "cs_lambda": 0.07, "pilot_rank": 1},
    {"hpo_run": 69, "alpha": 0.20, "cs_lambda": 0.09, "pilot_rank": 2},
    {"hpo_run": 20, "alpha": 0.06, "cs_lambda": 0.03, "pilot_rank": 3},
    {"hpo_run": 50, "alpha": 0.09, "cs_lambda": 0.07, "pilot_rank": 4},
    {"hpo_run": 53, "alpha": 0.09, "cs_lambda": 0.20, "pilot_rank": 5},
)

METRICS = camera.METRICS
HIGHER_IS_BETTER = camera.HIGHER_IS_BETTER
LOWER_IS_BETTER = camera.LOWER_IS_BETTER
COST_SENSITIVE_ENDPOINTS = camera.COST_SENSITIVE_ENDPOINTS


def value_label(value: float) -> str:
    return base.pretty_numeric_label(value)


def method_slug(point: dict[str, Any]) -> str:
    return (
        f"nicme_run{point['hpo_run']}_"
        f"a{value_label(float(point['alpha']))}_l{value_label(float(point['cs_lambda']))}"
    )


TOP5_METHODS = tuple(method_slug(point) for point in TOP5_POINTS)
POINT_BY_METHOD = {method_slug(point): point for point in TOP5_POINTS}


def parse_seeds(value: str) -> tuple[int, ...]:
    return camera.parse_seeds(value)


def seed_root(output_root: Path, seed: int) -> Path:
    return output_root / "seeds" / f"seed_{seed}"


def display_name(method: str) -> str:
    point = POINT_BY_METHOD.get(method)
    if not point:
        return method
    return (
        f"NICME run {point['hpo_run']} "
        f"(alpha={point['alpha']:g}, lambda={point['cs_lambda']:g})"
    )


def build_training_config(
    phase: str,
    method: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    if method not in POINT_BY_METHOD:
        raise ValueError(f"Unknown top-5 NICME method: {method}")
    point = POINT_BY_METHOD[method]
    cfg = base.base_config(
        phase,
        "nicme_hybrid_pretty",
        split_dir,
        output_root,
        seed,
        epochs=1 if smoke else None,
        comparison_profile=COMPARISON_PROFILE,
        learning_rate_profile=LEARNING_RATE_PROFILE,
    )
    cfg["sota_method"] = method
    cfg["loss_function"] = "nicme_hybrid"
    cfg["sota_role"] = "nicme_top5_alpha_lambda_multiseed"
    cfg["sota_comparison_profile"] = "nicme_top5_alpha_lambda_lr5e5_multiseed"
    cfg["nicme_top5_multiseed"] = True
    cfg["pilot_hpo_run"] = int(point["hpo_run"])
    cfg["pilot_hpo_rank"] = int(point["pilot_rank"])
    cfg["pilot_selected_hyperparameters"] = True
    cfg["nicme_logit_cost_scale"] = float(point["alpha"])
    cfg["cs_lambda"] = float(point["cs_lambda"])
    cfg["grid_alpha"] = float(point["alpha"])
    cfg["grid_cs_lambda"] = float(point["cs_lambda"])
    cfg["output_dir"] = f"top5_{phase}_{method}_lr5e5_s{seed}"
    return cfg


def build_seed_configs(
    phase: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    return [
        build_training_config(phase, method, split_dir, output_root, seed, smoke=smoke)
        for method in TOP5_METHODS
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
    seed_args = argparse.Namespace(**vars(args))
    seed_args.output_root = str(root)

    for idx, method in enumerate(TOP5_METHODS, start=1):
        cfg = build_training_config(phase, method, split_dir, root, seed, smoke=smoke)
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
            "hpo_run": cfg.get("pilot_hpo_run", ""),
            "pilot_rank": cfg.get("pilot_hpo_rank", ""),
        }
    )
    return metadata


def metric_row_from_ledger(seed: int, row: dict[str, Any], method: str | None = None) -> dict[str, Any]:
    method = method or row["method"]
    metadata = config_metadata(row.get("config_path", ""))
    mode, report = base.load_metric_report(row["metric_path"], method)
    metric_row = base.metric_row(
        method,
        row.get("role") or method,
        row["metric_path"],
        mode,
        report,
        f"nicme_top5_seed_{seed}",
        metadata,
    )
    metric_row.update(
        {
            "seed": seed,
            "config_path": row.get("config_path", ""),
            "checkpoint_path": row.get("checkpoint_path", ""),
            "hpo_run": metadata.get("hpo_run", ""),
            "pilot_rank": metadata.get("pilot_rank", ""),
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
    missing = [method for method in TOP5_METHODS if by_method.get(method, {}).get("status") != "completed"]
    if missing and not allow_incomplete:
        raise RuntimeError(f"Seed {seed} has incomplete top-5 methods: {missing}")

    metric_rows: list[dict[str, Any]] = []
    for method in TOP5_METHODS:
        row = by_method.get(method)
        if not row or row.get("status") != "completed":
            continue
        if not row.get("metric_path"):
            if allow_incomplete:
                continue
            raise RuntimeError(f"Seed {seed} method {method} is missing metric_path")
        metric_rows.append(metric_row_from_ledger(seed, row, method))
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


def per_seed_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -float(row.get("target_recall_min") or 0.0),
        float(row.get("normalized_atc") or 1.0),
        -float(row.get("balanced_accuracy") or 0.0),
        -float(row.get("macro_f1") or 0.0),
    )


def aggregate_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault(row["method"], []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        point = POINT_BY_METHOD.get(method, {})
        out: dict[str, Any] = {
            "method": method,
            "display_name": display_name(method),
            "decision_mode": rows[0].get("decision_mode", ""),
            "n_seeds": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda item: int(item["seed"]))),
            "hpo_run": point.get("hpo_run", rows[0].get("hpo_run", "")),
            "pilot_rank": point.get("pilot_rank", rows[0].get("pilot_rank", "")),
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
            }
        )
    return winners


def format_mean_std(row: dict[str, Any], metric: str, precision: int = 4) -> str:
    return camera.format_mean_std(row, metric, precision)


def format_tex_mean_std(row: dict[str, Any], metric: str, precision: int = 4) -> str:
    return camera.format_tex_mean_std(row, metric, precision)


def write_top5_tables(output_root: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    md_lines = [
        "# PMI-10 NICME Top-5 Alpha/Lambda Multi-Seed Table",
        "",
        "Metrics are mean +/- sample standard deviation over seeds 42, 43, and 44 on one fixed split.",
        "These settings were selected from the earlier pilot HPO; this table is a robustness confirmation, not a new HPO.",
        "",
        "| Rank | HPO Run | alpha | lambda | Target-Min Recall | Target-Macro Recall | Norm. ATC | ATC | Balanced Acc. | Macro F1 | Critical Errors |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        md_lines.append(
            "| "
            f"{row['composite_rank']} | {row.get('hpo_run', '')} | "
            f"{float(row.get('alpha') or 0.0):g} | {float(row.get('cs_lambda') or 0.0):g} | "
            f"{format_mean_std(row, 'target_recall_min')} | "
            f"{format_mean_std(row, 'target_recall_macro')} | "
            f"{format_mean_std(row, 'normalized_atc', 6)} | "
            f"{format_mean_std(row, 'atc', 6)} | "
            f"{format_mean_std(row, 'balanced_accuracy')} | "
            f"{format_mean_std(row, 'macro_f1')} | "
            f"{format_mean_std(row, 'critical_pair_error_count')} |"
        )
    (output_root / "analysis" / "nicme_top5_table.md").write_text("\n".join(md_lines) + "\n")

    tex_lines = [
        "\\begin{tabular}{r r r r l l l l l l l}",
        "\\toprule",
        "Rank & HPO Run & $\\alpha$ & $\\lambda$ & Target-min & Target-macro & Norm. ATC & ATC & Bal. Acc. & Macro F1 & Critical \\\\",
        "\\midrule",
    ]
    for row in aggregate_rows:
        tex_lines.append(
            f"{row['composite_rank']} & {row.get('hpo_run', '')} & "
            f"{float(row.get('alpha') or 0.0):g} & {float(row.get('cs_lambda') or 0.0):g} & "
            f"{format_tex_mean_std(row, 'target_recall_min')} & "
            f"{format_tex_mean_std(row, 'target_recall_macro')} & "
            f"{format_tex_mean_std(row, 'normalized_atc', 6)} & "
            f"{format_tex_mean_std(row, 'atc', 6)} & "
            f"{format_tex_mean_std(row, 'balanced_accuracy')} & "
            f"{format_tex_mean_std(row, 'macro_f1')} & "
            f"{format_tex_mean_std(row, 'critical_pair_error_count')} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (output_root / "analysis" / "nicme_top5_table.tex").write_text("\n".join(tex_lines) + "\n")


def write_winner_report(output_root: Path, winners: list[dict[str, Any]]) -> None:
    lines = [
        "# NICME Top-5 Cost-Sensitive Winners",
        "",
        "Winners are computed only among the five fixed alpha/lambda candidates.",
        "",
        "| Endpoint | Direction | Winner | Value |",
        "|---|---|---|---:|",
    ]
    for row in winners:
        value = row["value"]
        value_text = f"{float(value):.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {row['endpoint']} | {row['direction']} | {row['display_name']} | {value_text} |")
    (output_root / "analysis" / "cost_sensitive_winners.md").write_text("\n".join(lines) + "\n")


def per_seed_ranks(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in metric_rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    ranked_rows: list[dict[str, Any]] = []
    for seed, rows in sorted(by_seed.items()):
        ranked = sorted(rows, key=per_seed_sort_key)
        for rank, row in enumerate(ranked, start=1):
            ranked_rows.append(
                {
                    "seed": seed,
                    "per_seed_rank": rank,
                    "method": row["method"],
                    "display_name": display_name(row["method"]),
                    "hpo_run": row.get("hpo_run", ""),
                    "alpha": row.get("alpha", ""),
                    "cs_lambda": row.get("cs_lambda", ""),
                    "target_recall_min": row.get("target_recall_min", ""),
                    "normalized_atc": row.get("normalized_atc", ""),
                    "balanced_accuracy": row.get("balanced_accuracy", ""),
                    "macro_f1": row.get("macro_f1", ""),
                    "critical_pair_error_count": row.get("critical_pair_error_count", ""),
                }
            )
    return ranked_rows


def write_rank_stability(output_root: Path, metric_rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    ranked = per_seed_ranks(metric_rows)
    write_csv(
        output_root / "analysis" / "per_seed_ranks.csv",
        ranked,
        [
            "seed",
            "per_seed_rank",
            "method",
            "display_name",
            "hpo_run",
            "alpha",
            "cs_lambda",
            "target_recall_min",
            "normalized_atc",
            "balanced_accuracy",
            "macro_f1",
            "critical_pair_error_count",
        ],
    )
    win_counts = {row["method"]: 0 for row in aggregate_rows}
    for row in ranked:
        if int(row["per_seed_rank"]) == 1:
            win_counts[row["method"]] = win_counts.get(row["method"], 0) + 1

    lines = [
        "# NICME Top-5 Rank Stability",
        "",
        "Ranks use the predeclared recall-first cost-sensitive order within each seed.",
        "",
        "| Aggregate Rank | HPO Run | alpha | lambda | Seed Rank-1 Wins | Seeds |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate_rows:
        seed_ranks = [
            f"{rank_row['seed']}:#{rank_row['per_seed_rank']}"
            for rank_row in ranked
            if rank_row["method"] == row["method"]
        ]
        lines.append(
            "| "
            f"{row['composite_rank']} | {row.get('hpo_run', '')} | "
            f"{float(row.get('alpha') or 0.0):g} | {float(row.get('cs_lambda') or 0.0):g} | "
            f"{win_counts.get(row['method'], 0)} | {', '.join(seed_ranks)} |"
        )
    (output_root / "analysis" / "rank_stability.md").write_text("\n".join(lines) + "\n")


def write_paired_deltas(output_root: Path, metric_rows: list[dict[str, Any]]) -> None:
    baseline_method = method_slug(TOP5_POINTS[0])
    by_seed_method = {(int(row["seed"]), row["method"]): row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for seed in sorted({int(row["seed"]) for row in metric_rows}):
        baseline_row = by_seed_method.get((seed, baseline_method))
        if not baseline_row:
            continue
        for method in TOP5_METHODS:
            if method == baseline_method:
                continue
            row = by_seed_method.get((seed, method))
            if not row:
                continue
            for metric in METRICS:
                b_val = to_float(baseline_row.get(metric))
                value = to_float(row.get(metric))
                if math.isnan(b_val) or math.isnan(value):
                    continue
                delta_vs_run95 = value - b_val if metric in HIGHER_IS_BETTER else b_val - value
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "display_name": display_name(method),
                        "hpo_run": row.get("hpo_run", ""),
                        "metric": metric,
                        "run95": b_val,
                        "candidate": value,
                        "delta_positive_favors_candidate": delta_vs_run95,
                        "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                    }
                )
    write_csv(
        output_root / "analysis" / "paired_deltas_vs_run95.csv",
        rows,
        [
            "seed",
            "method",
            "display_name",
            "hpo_run",
            "metric",
            "run95",
            "candidate",
            "delta_positive_favors_candidate",
            "direction",
        ],
    )


def write_claim_audit(output_root: Path, winners: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    leader = aggregate_rows[0] if aggregate_rows else {}
    lines = [
        "# NICME Top-5 Claim Audit",
        "",
        "This analysis compares only five NICME alpha/lambda candidates selected from pilot HPO.",
        "It should not be presented as evidence that NICME beats external baselines; use the separate camera-ready baseline table for that.",
        "",
        "## Aggregate Leader",
        "",
        f"- Composite leader: `{leader.get('display_name', '')}`",
        "",
        "## Endpoint Winners",
        "",
        "| Endpoint | Winner | Value |",
        "|---|---|---:|",
    ]
    for row in winners:
        value = row["value"]
        value_text = f"{float(value):.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {row['endpoint']} | {row['display_name']} | {value_text} |")
    lines.extend(
        [
            "",
            "## Guardrail",
            "",
            "- Describe this as a robustness confirmation of pilot-selected candidates, not as fresh HPO.",
            "- Report all five settings even if run 95 is not the aggregate winner.",
        ]
    )
    (output_root / "analysis" / "claim_audit.md").write_text("\n".join(lines) + "\n")


def write_method_hyperparameters(output_root: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# NICME Top-5 Hyperparameters",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "- Shared LR profile: `lr5e5`.",
        "- Shared backbone: `timm/convnext_base.fb_in22k_ft_in1k`.",
        "- Shared split: balanced PMI-10 no-calibration.",
        "- Loss: `nicme_hybrid`.",
        "",
        "| HPO Run | Pilot Rank | Method | LR | alpha | lambda |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in sorted(aggregate_rows, key=lambda item: int(item.get("pilot_rank") or 999)):
        lines.append(
            "| "
            f"{row.get('hpo_run', '')} | {row.get('pilot_rank', '')} | {row['display_name']} | "
            f"`{row.get('learning_rate', '')}` | `{row.get('alpha', '')}` | `{row.get('cs_lambda', '')}` |"
        )
    (output_root / "analysis" / "method_hyperparameters.md").write_text("\n".join(lines) + "\n")


def write_readme(output_root: Path, args: argparse.Namespace, aggregate_rows: list[dict[str, Any]]) -> None:
    top = aggregate_rows[0] if aggregate_rows else {}
    lines = [
        "# PMI-10 NICME Top-5 Alpha/Lambda Multi-Seed Results",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "This folder contains a fixed-protocol robustness rerun for the top five observed NICME alpha/lambda settings from the LR 5e-5 pilot HPO.",
        "",
        "## Protocol",
        "",
        f"- Seeds: `{','.join(str(seed) for seed in args.seeds)}`",
        "- Split: `data/prepared/pmi_pills_10_no_cal/splits/balanced`",
        "- Backbone: `timm/convnext_base.fb_in22k_ft_in1k`",
        "- LR profile: `lr5e5`",
        "- Candidate HPO runs: `95, 69, 20, 50, 53`",
        "",
        "## Main Outputs",
        "",
        "- `analysis/per_seed_metrics.csv`",
        "- `analysis/aggregate_metrics.csv`",
        "- `analysis/nicme_top5_table.md`",
        "- `analysis/nicme_top5_table.tex`",
        "- `analysis/rank_stability.md`",
        "- `analysis/paired_deltas_vs_run95.csv`",
        "- `analysis/claim_audit.md`",
        "- `analysis/method_hyperparameters.md`",
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
        "display_name",
        "hpo_run",
        "pilot_rank",
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
    for row in metric_rows:
        row["display_name"] = display_name(row["method"])
    write_csv(output_root / "analysis" / "per_seed_metrics.csv", metric_rows, per_seed_fields)

    aggregate_rows = aggregate_metric_rows(metric_rows)
    aggregate_fields = [
        "composite_rank",
        "method",
        "display_name",
        "decision_mode",
        "n_seeds",
        "seeds",
        "hpo_run",
        "pilot_rank",
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
    write_top5_tables(output_root, aggregate_rows)
    write_winner_report(output_root, winners)
    write_rank_stability(output_root, metric_rows, aggregate_rows)
    write_paired_deltas(output_root, metric_rows)
    write_claim_audit(output_root, winners, aggregate_rows)
    write_method_hyperparameters(output_root, aggregate_rows)
    write_readme(output_root, args, aggregate_rows)
    print(f"Wrote NICME top-5 analysis under {output_root / 'analysis'}")


def assert_top5_config(cfg: dict[str, Any]) -> None:
    if cfg["loss_function"] != "nicme_hybrid":
        raise ValueError("Top-5 runner must use nicme_hybrid")
    if float(cfg["learning_rate"]) != 5e-5 or float(cfg["parent_learning_rate"]) != 5e-5:
        raise ValueError("Top-5 runner must keep LR 5e-5")
    if int(cfg["num_train_epochs"]) != 32 or int(cfg["early_stopping_patience"]) != 5:
        raise ValueError("Top-5 runner must keep camera-ready epoch/patience defaults")
    if int(cfg["batch_size"]) != 16 or int(cfg["gradient_accumulation_steps"]) != 4:
        raise ValueError("Top-5 runner must keep camera-ready batch defaults")
    if float(cfg["weight_decay"]) != 0.005 or float(cfg["warmup_ratio"]) != 0.10:
        raise ValueError("Top-5 runner must keep camera-ready regularization defaults")
    if cfg["lr_scheduler_type"] != "cosine":
        raise ValueError("Top-5 runner must keep cosine LR")
    if cfg["calibration_enabled"] or cfg["peft_enabled"] or cfg["decision_modes"] != "argmax":
        raise ValueError("Top-5 invariants failed: no calibration, no LoRA, argmax only")


def run_preflight(args: argparse.Namespace) -> None:
    split_dir = Path(args.split_dir)
    base.assert_split_ready(split_dir)
    if not args.skip_nvidia_smi:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
    profile = get_profile(base.DATASET)
    if profile.cost_matrix_sha256 != REQUIRED_COST_HASH:
        raise ValueError("Unexpected PMI-10 profile cost matrix hash")
    seen: set[tuple[int, float, float]] = set()
    for method in TOP5_METHODS:
        cfg = build_training_config("preflight", method, split_dir, Path(args.output_root), args.seeds[0])
        point = POINT_BY_METHOD[method]
        seen.add((int(point["hpo_run"]), float(cfg["nicme_logit_cost_scale"]), float(cfg["cs_lambda"])))
        if cfg["cost_matrix_sha256"] != REQUIRED_COST_HASH or base.cost_matrix_hash(cfg) != REQUIRED_COST_HASH:
            raise ValueError("PMI-10 cost matrix hash mismatch")
        if cfg["model"] != f"timm/{base.MODEL_ID}" or cfg["variant"] != base.VARIANT:
            raise ValueError("Top-5 PMI-10 config invariants failed")
        assert_top5_config(cfg)
    expected = {(95, 0.50, 0.07), (69, 0.20, 0.09), (20, 0.06, 0.03), (50, 0.09, 0.07), (53, 0.09, 0.20)}
    if seen != expected:
        raise ValueError(f"Top-5 point mismatch: {seen}")
    print("NICME top-5 PMI-10 preflight passed")


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
