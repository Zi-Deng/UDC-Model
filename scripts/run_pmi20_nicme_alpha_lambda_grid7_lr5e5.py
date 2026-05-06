"""PMI-20 NICME 7x7 alpha/lambda sensitivity runner.

This runner executes a fresh single-seed, predeclared alpha/lambda grid on the
balanced full 20-class PMI dataset. Hyperparameters are selected by validation
metrics only; held-out test metrics are reported as an audit.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from scripts import run_pmi10_camera_ready_lr5e5 as camera
from scripts import run_pmi10_sota_baselines as base
from scripts import run_pmi20_camera_ready_lr5e5 as pmi20

DEFAULT_OUTPUT_ROOT = Path("results/pmi20_nicme_alpha_lambda_grid7_lr5e5_seed42_20260506")
DEFAULT_SEEDS = (42,)
COMPARISON_PROFILE = "pmi20_nicme_alpha_lambda_grid7_lr5e5_seed42"
TOL = 1e-12

ALPHA_VALUES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
LAMBDA_VALUES = (0.0, 0.02, 0.05, 0.07, 0.1, 0.2, 0.3)
CURRENT_MAIN_POINT = (0.5, 0.1)
SMOKE_POINTS = (
    (0.0, 0.0),
    (0.0, 0.1),
    (0.5, 0.0),
    (0.5, 0.1),
    (0.7, 0.3),
    (0.05, 0.02),
    (0.3, 0.07),
    (0.1, 0.2),
    (0.7, 0.0),
)

METRICS = camera.METRICS


def parse_seeds(value: str) -> tuple[int, ...]:
    return camera.parse_seeds(value)


def value_label(value: float) -> str:
    return base.pretty_numeric_label(value)


def method_slug(alpha: float, cs_lambda: float) -> str:
    return f"nicme_grid7_a{value_label(alpha)}_l{value_label(cs_lambda)}_pmi20"


GRID_POINTS = tuple(
    {
        "candidate_order": idx + 1,
        "alpha": alpha,
        "cs_lambda": cs_lambda,
        "method": method_slug(alpha, cs_lambda),
    }
    for idx, (alpha, cs_lambda) in enumerate((alpha, cs_lambda) for alpha in ALPHA_VALUES for cs_lambda in LAMBDA_VALUES)
)
ALPHA_LAMBDA_METHODS = tuple(point["method"] for point in GRID_POINTS)
POINT_BY_METHOD = {point["method"]: point for point in GRID_POINTS}
METHOD_BY_POINT = {(float(point["alpha"]), float(point["cs_lambda"])): point["method"] for point in GRID_POINTS}
SMOKE_METHODS = tuple(METHOD_BY_POINT[point] for point in SMOKE_POINTS)


def seed_root(output_root: Path, seed: int) -> Path:
    return output_root / "seeds" / f"seed_{seed}"


def display_name(method: str) -> str:
    point = POINT_BY_METHOD.get(method)
    if not point:
        return method
    return f"NICME grid7 (alpha={point['alpha']:g}, lambda={point['cs_lambda']:g})"


def build_training_config(
    phase: str,
    method: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    if method not in POINT_BY_METHOD:
        raise ValueError(f"Unknown PMI-20 grid7 candidate: {method}")
    point = POINT_BY_METHOD[method]
    cfg = pmi20.base_config(
        phase,
        pmi20.NICME_METHOD,
        split_dir,
        output_root,
        seed,
        epochs=1 if smoke else None,
    )
    cfg["loss_function"] = "nicme_hybrid"
    cfg["sota_method"] = method
    cfg["sota_role"] = "pmi20_nicme_alpha_lambda_grid7_seed42"
    cfg["sota_comparison_profile"] = COMPARISON_PROFILE
    cfg["nicme_alpha_lambda_grid7_seed42"] = True
    cfg["candidate_order"] = int(point["candidate_order"])
    cfg["candidate_source_label"] = "predeclared 7x7 PMI-20 grid"
    cfg["nicme_logit_cost_scale"] = float(point["alpha"])
    cfg["cs_lambda"] = float(point["cs_lambda"])
    cfg["grid_alpha"] = float(point["alpha"])
    cfg["grid_cs_lambda"] = float(point["cs_lambda"])
    cfg["pilot_selected_hyperparameters"] = False
    cfg["added_fixed_candidate"] = False
    cfg.pop("pilot_hpo_run", None)
    cfg["cs_warmup_epochs"] = 2
    cfg["save_total_limit"] = 1
    cfg["output_dir"] = (
        f"pmi20_alpha_lambda_grid7_{phase}_a{value_label(float(point['alpha']))}_"
        f"l{value_label(float(point['cs_lambda']))}_lr5e5_s{seed}"
    )
    return cfg


def phase_methods(smoke: bool = False) -> tuple[str, ...]:
    return SMOKE_METHODS if smoke else ALPHA_LAMBDA_METHODS


def build_seed_configs(
    phase: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    return [build_training_config(phase, method, split_dir, output_root, seed, smoke=smoke) for method in phase_methods(smoke)]


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

    for idx, method in enumerate(phase_methods(smoke), start=1):
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
    metadata = pmi20.config_metadata(config_path)
    if not config_path:
        return metadata
    path = Path(config_path)
    if not path.exists():
        return metadata
    with open(path) as f:
        cfg = json.load(f)
    metadata.update(
        {
            "candidate_order": cfg.get("candidate_order", ""),
            "source_label": cfg.get("candidate_source_label", ""),
            "alpha": cfg.get("nicme_logit_cost_scale", ""),
            "cs_lambda": cfg.get("cs_lambda", ""),
            "loss_function": cfg.get("loss_function", ""),
            "dataset_profile": cfg.get("dataset_profile", ""),
            "num_labels": cfg.get("num_labels", ""),
        }
    )
    return metadata


def load_payload(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def report_metric_values(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_recall_min": report.get("target_recall_min", report.get("target_recall", "")),
        "target_recall_macro": report.get("target_recall_macro", ""),
        "normalized_atc": report.get("normalized_atc", ""),
        "atc": report.get("atc", ""),
        "balanced_accuracy": report.get("balanced_accuracy", ""),
        "accuracy": report.get("accuracy", ""),
        "macro_f1": report.get("macro_f1", ""),
        "critical_pair_error_count": base.critical_pair_error_count(report) if report else "",
    }


def validation_values(payload: dict[str, Any]) -> dict[str, Any]:
    validation_report = payload.get("validation_report", {})
    report = validation_report.get("argmax", {})
    values = report_metric_values(report)
    metrics = validation_report.get("metrics", {})
    fallbacks = {
        "target_recall_min": "eval_target_recall",
        "target_recall_macro": "eval_target_recall_macro",
        "normalized_atc": "eval_normalized_atc",
        "atc": "eval_atc",
        "balanced_accuracy": "eval_balanced_accuracy",
        "accuracy": "eval_accuracy",
        "macro_f1": "eval_macro_f1",
    }
    for key, metric_key in fallbacks.items():
        if values.get(key) in {"", None} and metric_key in metrics:
            values[key] = metrics[metric_key]
    return values


def test_values(payload: dict[str, Any], method: str) -> dict[str, Any]:
    _mode, report = base.preferred_report(payload, method)
    return report_metric_values(report)


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def critical_as_int(value: Any) -> int:
    numeric = to_float(value, 999999.0)
    return int(numeric) if not math.isnan(numeric) else 999999


def validation_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        -to_float(row.get("validation_target_recall_min"), 0.0),
        to_float(row.get("validation_normalized_atc"), 1.0),
        -to_float(row.get("validation_balanced_accuracy"), 0.0),
        -to_float(row.get("validation_macro_f1"), 0.0),
        critical_as_int(row.get("validation_critical_pair_error_count")),
    )


def test_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    return (
        -to_float(row.get("test_target_recall_min"), 0.0),
        to_float(row.get("test_normalized_atc"), 1.0),
        -to_float(row.get("test_balanced_accuracy"), 0.0),
        -to_float(row.get("test_macro_f1"), 0.0),
        critical_as_int(row.get("test_critical_pair_error_count")),
    )


def metric_row_from_ledger(seed: int, row: dict[str, Any], method: str | None = None) -> dict[str, Any]:
    method = method or row["method"]
    metadata = config_metadata(row.get("config_path", ""))
    payload = load_payload(row["metric_path"])
    point = POINT_BY_METHOD[method]
    out: dict[str, Any] = {
        "seed": seed,
        "method": method,
        "display_name": display_name(method),
        "candidate_order": point["candidate_order"],
        "source_label": metadata.get("source_label", "predeclared 7x7 PMI-20 grid"),
        "alpha": metadata.get("alpha", point["alpha"]),
        "cs_lambda": metadata.get("cs_lambda", point["cs_lambda"]),
        "loss_function": metadata.get("loss_function", ""),
        "learning_rate": metadata.get("learning_rate", ""),
        "parent_learning_rate": metadata.get("parent_learning_rate", ""),
        "metric_path": row.get("metric_path", ""),
        "config_path": row.get("config_path", ""),
        "checkpoint_path": row.get("checkpoint_path", ""),
        "is_current_main": str((float(point["alpha"]), float(point["cs_lambda"])) == CURRENT_MAIN_POINT).lower(),
    }
    for prefix, values in (("validation", validation_values(payload)), ("test", test_values(payload, method))):
        for metric in METRICS:
            out[f"{prefix}_{metric}"] = values.get(metric, "")
    return out


def load_seed_metric_rows(output_root: Path, seed: int, allow_incomplete: bool = False) -> list[dict[str, Any]]:
    ledger_path = seed_root(output_root, seed) / "run" / "run_ledger.csv"
    rows = base.read_ledger(ledger_path)
    if not rows:
        if allow_incomplete:
            return []
        raise RuntimeError(f"Missing run ledger for seed {seed}: {ledger_path}")
    by_method = {row["method"]: row for row in rows}
    missing = [method for method in ALPHA_LAMBDA_METHODS if by_method.get(method, {}).get("status") != "completed"]
    if missing and not allow_incomplete:
        raise RuntimeError(f"Seed {seed} has incomplete PMI-20 grid7 methods: {missing}")
    metric_rows: list[dict[str, Any]] = []
    for method in ALPHA_LAMBDA_METHODS:
        row = by_method.get(method)
        if not row or row.get("status") != "completed":
            continue
        if not row.get("metric_path"):
            if allow_incomplete:
                continue
            raise RuntimeError(f"Seed {seed} method {method} is missing metric_path")
        metric_rows.append(metric_row_from_ledger(seed, row, method))
    return metric_rows


def add_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validation_ranked = sorted(rows, key=validation_sort_key)
    test_ranked = sorted(rows, key=test_sort_key)
    for idx, row in enumerate(validation_ranked, start=1):
        row["validation_rank"] = idx
        row["is_validation_selected"] = str(idx == 1).lower()
    for idx, row in enumerate(test_ranked, start=1):
        row["test_rank"] = idx
    return validation_ranked


def analysis_fields() -> list[str]:
    fields = [
        "seed",
        "validation_rank",
        "test_rank",
        "method",
        "display_name",
        "candidate_order",
        "source_label",
        "alpha",
        "cs_lambda",
        "loss_function",
        "learning_rate",
        "parent_learning_rate",
        "is_current_main",
        "is_validation_selected",
    ]
    for prefix in ("validation", "test"):
        fields.extend(f"{prefix}_{metric}" for metric in METRICS)
    fields.extend(["metric_path", "config_path", "checkpoint_path"])
    return fields


def format_value(value: Any, precision: int = 4) -> str:
    numeric = to_float(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.{precision}f}"


def write_grid_table(output_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PMI-20 NICME 7x7 Alpha/Lambda Validation-Ranked Grid",
        "",
        "Single-seed sensitivity surface. Hyperparameters are ranked by validation metrics only; test metrics are reported as an audit.",
        "",
        "| Val. Rank | alpha | lambda | Val Target-Min | Val nATC | Val Bal. Acc. | Test Target-Min | Test nATC | Test Bal. Acc. | Flags |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        flags = []
        if row.get("is_current_main") == "true":
            flags.append("current main")
        if row.get("is_validation_selected") == "true":
            flags.append("validation selected")
        lines.append(
            "| "
            f"{row['validation_rank']} | {format_value(row['alpha'], 2)} | {format_value(row['cs_lambda'], 2)} | "
            f"{format_value(row['validation_target_recall_min'])} | "
            f"{format_value(row['validation_normalized_atc'], 6)} | "
            f"{format_value(row['validation_balanced_accuracy'])} | "
            f"{format_value(row['test_target_recall_min'])} | "
            f"{format_value(row['test_normalized_atc'], 6)} | "
            f"{format_value(row['test_balanced_accuracy'])} | {', '.join(flags)} |"
        )
    (output_root / "analysis" / "pmi20_alpha_lambda_grid7_table.md").write_text("\n".join(lines) + "\n")

    tex = [
        "\\begin{tabular}{r r r r r r r r r l}",
        "\\toprule",
        "Val. Rank & $\\alpha$ & $\\lambda$ & Val Target-min & Val nATC & Val Bal. Acc. & Test Target-min & Test nATC & Test Bal. Acc. & Flags \\\\",
        "\\midrule",
    ]
    for row in rows:
        flags = []
        if row.get("is_current_main") == "true":
            flags.append("current")
        if row.get("is_validation_selected") == "true":
            flags.append("selected")
        tex.append(
            f"{row['validation_rank']} & {format_value(row['alpha'], 2)} & {format_value(row['cs_lambda'], 2)} & "
            f"{format_value(row['validation_target_recall_min'])} & "
            f"{format_value(row['validation_normalized_atc'], 6)} & "
            f"{format_value(row['validation_balanced_accuracy'])} & "
            f"{format_value(row['test_target_recall_min'])} & "
            f"{format_value(row['test_normalized_atc'], 6)} & "
            f"{format_value(row['test_balanced_accuracy'])} & {', '.join(flags)} \\\\"
        )
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    (output_root / "analysis" / "pmi20_alpha_lambda_grid7_table.tex").write_text("\n".join(tex) + "\n")


def write_selected_report(output_root: Path, rows: list[dict[str, Any]]) -> None:
    selected = rows[0]
    current = next((row for row in rows if row.get("is_current_main") == "true"), None)
    lines = [
        "# PMI-20 NICME 7x7 Selection By Validation",
        "",
        "Selection uses validation metrics only, ordered by target-min recall, normalized ATC, balanced accuracy, and macro-F1.",
        "",
        "## Validation-Selected Point",
        "",
        f"- alpha: `{format_value(selected['alpha'], 2)}`",
        f"- lambda: `{format_value(selected['cs_lambda'], 2)}`",
        f"- validation target-min recall: `{format_value(selected['validation_target_recall_min'])}`",
        f"- validation normalized ATC: `{format_value(selected['validation_normalized_atc'], 6)}`",
        f"- test target-min recall: `{format_value(selected['test_target_recall_min'])}`",
        f"- test normalized ATC: `{format_value(selected['test_normalized_atc'], 6)}`",
        "",
        "## Current Main Paper Point",
        "",
    ]
    if current:
        lines.extend(
            [
                f"- alpha: `{format_value(current['alpha'], 2)}`",
                f"- lambda: `{format_value(current['cs_lambda'], 2)}`",
                f"- validation rank: `{current['validation_rank']}`",
                f"- validation target-min recall: `{format_value(current['validation_target_recall_min'])}`",
                f"- validation normalized ATC: `{format_value(current['validation_normalized_atc'], 6)}`",
                f"- test target-min recall: `{format_value(current['test_target_recall_min'])}`",
                f"- test normalized ATC: `{format_value(current['test_normalized_atc'], 6)}`",
            ]
        )
    (output_root / "analysis" / "selected_by_validation.md").write_text("\n".join(lines) + "\n")


def write_claim_audit(output_root: Path, rows: list[dict[str, Any]]) -> None:
    selected = rows[0]
    lines = [
        "# PMI-20 NICME 7x7 Claim Audit",
        "",
        "- This is a single-seed predeclared sensitivity/HPO surface, not multi-seed robustness evidence.",
        "- Hyperparameter selection is validation-ranked; test metrics are only a post-selection audit.",
        "- The 3-seed PMI-20 SOTA table remains the primary comparison against baselines.",
        "- Do not expand this grid after launch without labeling the expansion as a separate follow-up.",
        "",
        "## Supported Use",
        "",
        f"- Validation-selected point: alpha `{format_value(selected['alpha'], 2)}`, lambda `{format_value(selected['cs_lambda'], 2)}`.",
        "- Use the heatmap to show broad sensitivity structure and whether the current main point lies in a stable low-cost/high-recall region.",
    ]
    (output_root / "analysis" / "claim_audit.md").write_text("\n".join(lines) + "\n")


def write_readme(output_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    selected = rows[0]
    (output_root / "README.md").write_text(
        "# PMI-20 NICME 7x7 Alpha/Lambda Sensitivity Sweep\n\n"
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}\n\n"
        "This result root contains a single-seed predeclared PMI-20 NICME alpha/lambda grid.\n\n"
        "## Protocol\n\n"
        f"- Seed: `{','.join(str(seed) for seed in args.seeds)}`\n"
        "- Grid: 7 alpha values x 7 lambda values = 49 jobs.\n"
        "- Selection: validation composite only.\n"
        "- Test metrics: post-selection audit only.\n\n"
        "## Validation-Selected Point\n\n"
        f"- alpha `{format_value(selected['alpha'], 2)}`, lambda `{format_value(selected['cs_lambda'], 2)}`\n\n"
        "## Outputs\n\n"
        "- `analysis/per_point_metrics.csv`\n"
        "- `analysis/validation_grid.csv`\n"
        "- `analysis/test_grid.csv`\n"
        "- `analysis/selected_by_validation.md`\n"
        "- `analysis/pmi20_alpha_lambda_grid7_table.md`\n"
        "- `analysis/pmi20_alpha_lambda_grid7_table.tex`\n"
        "- `analysis/claim_audit.md`\n"
    )


def analyze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    metric_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        metric_rows.extend(load_seed_metric_rows(output_root, seed, allow_incomplete=args.allow_incomplete_analysis))
    if not metric_rows:
        raise RuntimeError("No PMI-20 grid7 metric rows were available for analysis")
    ranked = add_ranks(metric_rows)
    fields = analysis_fields()
    write_csv(output_root / "analysis" / "per_point_metrics.csv", ranked, fields)
    write_csv(output_root / "analysis" / "validation_grid.csv", ranked, fields)
    write_csv(output_root / "analysis" / "test_grid.csv", sorted(ranked, key=lambda row: int(row["test_rank"])), fields)
    write_grid_table(output_root, ranked)
    write_selected_report(output_root, ranked)
    write_claim_audit(output_root, ranked)
    write_readme(output_root, args, ranked)
    print(f"Wrote PMI-20 grid7 analysis under {output_root / 'analysis'}")


def assert_grid_config(cfg: dict[str, Any]) -> None:
    pmi20.assert_pmi20_config(cfg)
    if cfg["loss_function"] != "nicme_hybrid":
        raise ValueError("Grid7 configs must use nicme_hybrid")
    if float(cfg["learning_rate"]) != 5e-5 or float(cfg["parent_learning_rate"]) != 5e-5:
        raise ValueError("Grid7 configs must use LR=5e-5")
    if int(cfg["num_train_epochs"]) != 32 or int(cfg["early_stopping_patience"]) != 5:
        raise ValueError("Grid7 training defaults changed")
    if int(cfg["save_total_limit"]) != 1:
        raise ValueError("Grid7 must retain save_total_limit=1")
    if int(cfg["cs_warmup_epochs"]) != 2:
        raise ValueError("Grid7 must use cs_warmup_epochs=2")


def run_preflight(args: argparse.Namespace) -> None:
    if tuple(args.seeds) != DEFAULT_SEEDS:
        raise ValueError("PMI-20 grid7 sensitivity is fixed to seed 42")
    split_dir = Path(args.split_dir)
    pmi20.assert_split_ready(split_dir)
    if not args.skip_nvidia_smi:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
    if set(METHOD_BY_POINT) != {(alpha, lam) for alpha in ALPHA_VALUES for lam in LAMBDA_VALUES}:
        raise ValueError("Grid7 point set changed")
    if CURRENT_MAIN_POINT not in METHOD_BY_POINT:
        raise ValueError("Current main point (0.5, 0.1) must be included")
    for method in ALPHA_LAMBDA_METHODS:
        assert_grid_config(build_training_config("preflight", method, split_dir, Path(args.output_root), args.seeds[0]))
    print("PMI-20 NICME grid7 preflight passed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preflight", "dry-run", "smoke", "run", "analyze"], required=True)
    parser.add_argument("--split-dir", default=str(pmi20.DEFAULT_SPLIT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--per-run-timeout-minutes", type=int, default=960)
    parser.add_argument("--allow-incomplete-analysis", action="store_true")
    parser.add_argument("--skip-nvidia-smi", action="store_true")
    args = parser.parse_args(argv)

    if args.phase == "preflight":
        run_preflight(args)
    elif args.phase == "dry-run":
        rows = write_dry_run(args)
        print(f"Wrote {len(rows)} PMI-20 grid7 dry-run configs under {Path(args.output_root)}")
    elif args.phase == "smoke":
        rows = execute_phase(args, "smoke", smoke=True)
        print(f"Completed {len(rows)} PMI-20 grid7 smoke runs")
    elif args.phase == "run":
        rows = execute_phase(args, "run", smoke=False)
        print(f"Completed {len(rows)} PMI-20 grid7 long runs")
    elif args.phase == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
