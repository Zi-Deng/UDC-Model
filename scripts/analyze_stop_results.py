"""Aggregate stop-gated NICME experiment results into paper-facing tables.

The training script writes metrics under model-specific result roots, while the
stop runner writes a compact run log under the stop phase root.  This script
joins those two sources so Stop 3/4 reads can be regenerated reproducibly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

METRIC_KEYS = [
    "selection_score",
    "normalized_atc",
    "atc",
    "target_recall",
    "target_fnr",
    "target_precision",
    "target_fpr",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "auroc",
    "auprc",
    "nll",
    "brier",
    "ece",
    "selected_accuracy",
]


FULL_FIELDNAMES = [
    "metrics_file",
    "run_dir",
    "dataset",
    "model",
    "method",
    "seed",
    "mode",
    "target_class",
    "target_recall_floor",
    "accuracy_floor",
    "selection_accuracy_metric",
    "cost_ratio",
    "cost_matrix",
    "test_class_prevalence",
    "selection_score",
    "normalized_atc",
    "atc",
    "crr",
    "target_recall",
    "target_fnr",
    "target_precision",
    "target_fpr",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "auroc",
    "auprc",
    "nll",
    "brier",
    "ece",
    "temperature",
    "threshold",
    "calibration_selection_score",
    "calibration_target_recall",
    "calibration_accuracy",
    "calibration_normalized_atc",
    "confusion_matrix",
    "selected_accuracy",
    "recall_floor_met",
    "accuracy_floor_met",
    "all_floors_met",
]


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def find_metrics_file(output_dir: str) -> Path | None:
    matches = sorted(Path("results").glob(f"*/*{output_dir}_*/metrics_*.json"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def read_config(run: dict[str, Any]) -> dict[str, Any]:
    config_path = Path(run["config"])
    return load_json(config_path)


def decision_rows(run_log_path: Path) -> list[dict[str, Any]]:
    runs = load_json(run_log_path)
    rows: list[dict[str, Any]] = []
    for run in runs:
        if run.get("returncode") != 0:
            continue
        cfg = read_config(run)
        metrics_file = find_metrics_file(run["output_dir"])
        if metrics_file is None:
            continue
        metrics = load_json(metrics_file)
        reports = metrics.get("decision_reports") or {"argmax": metrics.get("overall_metrics", {})}
        for mode, report in reports.items():
            selected_accuracy = report.get("selection_accuracy_value", report.get("accuracy"))
            target_recall = report.get("target_recall")
            recall_floor = float(cfg.get("target_recall_floor", 0.0))
            accuracy_floor = float(cfg.get("accuracy_floor", 0.0))
            recall_floor_met = target_recall is not None and float(target_recall) >= recall_floor
            accuracy_floor_met = selected_accuracy is not None and float(selected_accuracy) >= accuracy_floor
            row = {
                "metrics_file": str(metrics_file),
                "run_dir": str(metrics_file.parent),
                "dataset": cfg.get("dataset", run.get("dataset")),
                "model": run.get("model_type", cfg.get("model_type")),
                "method": run.get("method", cfg.get("loss_function")),
                "seed": run.get("seed", cfg.get("seed")),
                "mode": mode,
                "target_class": cfg.get("target_recall_class"),
                "target_recall_floor": recall_floor,
                "accuracy_floor": accuracy_floor,
                "selection_accuracy_metric": cfg.get("selection_accuracy_metric", report.get("selection_accuracy_metric")),
                "cost_ratio": cfg.get("cost_ratio"),
                "cost_matrix": compact_json(cfg.get("cost_matrix")),
                "test_class_prevalence": compact_json(report.get("class_prevalence", {})),
                "selection_score": report.get("selection_score"),
                "normalized_atc": report.get("normalized_atc"),
                "atc": report.get("atc"),
                "crr": report.get("crr"),
                "target_recall": target_recall,
                "target_fnr": report.get("target_fnr"),
                "target_precision": report.get("target_precision"),
                "target_fpr": report.get("target_fpr"),
                "accuracy": report.get("accuracy"),
                "balanced_accuracy": report.get("balanced_accuracy"),
                "macro_f1": report.get("macro_f1"),
                "auroc": report.get("auroc"),
                "auprc": report.get("auprc"),
                "nll": report.get("nll"),
                "brier": report.get("brier"),
                "ece": report.get("ece"),
                "temperature": report.get("temperature"),
                "threshold": report.get("threshold"),
                "calibration_selection_score": report.get("calibration_selection_score"),
                "calibration_target_recall": report.get("calibration_target_recall"),
                "calibration_accuracy": report.get("calibration_accuracy"),
                "calibration_normalized_atc": report.get("calibration_normalized_atc"),
                "confusion_matrix": compact_json(report.get("confusion_matrix")),
                "selected_accuracy": selected_accuracy,
                "recall_floor_met": recall_floor_met,
                "accuracy_floor_met": accuracy_floor_met,
                "all_floors_met": recall_floor_met and accuracy_floor_met,
            }
            rows.append(row)
    return rows


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def sd(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0 if len(vals) == 1 else None


def ci95(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0 if vals else None
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


def add_matrices(matrices: list[str]) -> str:
    total: list[list[int]] | None = None
    for matrix_text in matrices:
        if not matrix_text:
            continue
        matrix = json.loads(matrix_text)
        if total is None:
            total = [[0 for _ in row] for row in matrix]
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                total[i][j] += int(value)
    return compact_json(total) if total is not None else ""


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["model"], row["method"], row["mode"], row.get("cost_ratio"), row.get("cost_matrix"))
        groups[key].append(row)

    aggregates = []
    for (dataset, model, method, mode, cost_ratio, cost_matrix), group in groups.items():
        first = group[0]
        agg: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "method": method,
            "mode": mode,
            "cost_ratio": cost_ratio,
            "n": len(group),
            "target_class": first.get("target_class"),
            "target_recall_floor": first.get("target_recall_floor"),
            "accuracy_floor": first.get("accuracy_floor"),
            "selection_accuracy_metric": first.get("selection_accuracy_metric"),
            "cost_matrix": cost_matrix,
            "test_class_prevalence": first.get("test_class_prevalence"),
            "floor_pass_rate": mean([1.0 if row["all_floors_met"] else 0.0 for row in group]),
            "all_seeds_floor_met": all(row["all_floors_met"] for row in group),
            "any_seed_floor_met": any(row["all_floors_met"] for row in group),
            "confusion_matrix_sum": add_matrices([row.get("confusion_matrix", "") for row in group]),
        }
        for metric in METRIC_KEYS:
            values = [row.get(metric) for row in group]
            agg[f"{metric}_mean"] = mean(values)
            agg[f"{metric}_sd"] = sd(values)
            agg[f"{metric}_ci95"] = ci95(values)
        aggregates.append(agg)

    aggregates.sort(key=lambda row: (row["dataset"], row["selection_score_mean"] if row["selection_score_mean"] is not None else 999))
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def fmt_pm(row: dict[str, Any], key: str) -> str:
    return f"{fmt(row.get(key + '_mean'))} +/- {fmt(row.get(key + '_sd'))}"


def table_rows(rows: list[dict[str, Any]], limit: int = 10) -> str:
    show_ratio = any(row.get("cost_ratio") not in (None, "") for row in rows[:limit])
    if show_ratio:
        header = "| rank | model | method | mode | cost ratio | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |\n"
        sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    else:
        header = "| rank | model | method | mode | floors | selection | nATC | recall | selected acc | acc | bal acc | macro-F1 |\n"
        sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    lines = [header, sep]
    for idx, row in enumerate(rows[:limit], start=1):
        floors = "all" if row.get("all_seeds_floor_met") else f"{100*float(row.get('floor_pass_rate') or 0):.0f}%"
        cells = [
            str(idx),
            str(row["model"]),
            str(row["method"]),
            str(row["mode"]),
        ]
        if show_ratio:
            cells.append(fmt(row.get("cost_ratio")))
        cells.extend(
            [
                floors,
                fmt_pm(row, "selection_score"),
                fmt_pm(row, "normalized_atc"),
                fmt_pm(row, "target_recall"),
                fmt_pm(row, "selected_accuracy"),
                fmt_pm(row, "accuracy"),
                fmt_pm(row, "balanced_accuracy"),
                fmt(row.get("macro_f1_mean")),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |\n")
    return "".join(lines)


def write_markdown(path: Path, phase: str, rows: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        by_dataset[row["dataset"]].append(row)

    lines = [
        f"# {phase} Results And Scientific Read\n\n",
        "Generated from local run logs and metrics files.\n\n",
        "## Executive Summary\n\n",
        f"- Decision rows: `{len(rows)}`.\n",
        f"- Aggregate rows: `{len(aggregates)}`.\n",
        "- Lower `selection_score` is better; floor satisfaction uses the configured `selection_accuracy_metric` for each dataset.\n",
        "- Interpret natural/imbalanced results as deployment realism and decoupling stress tests, not replacements for balanced-dataset evidence.\n\n",
    ]
    for dataset, ds_rows in sorted(by_dataset.items()):
        ranked = sorted(ds_rows, key=lambda row: row.get("selection_score_mean") if row.get("selection_score_mean") is not None else 999)
        nicme = [row for row in ranked if str(row["method"]).startswith("nicme")]
        floor = [row for row in ranked if row.get("all_seeds_floor_met")]
        mean_floor = [
            row
            for row in ranked
            if (row.get("target_recall_mean") or 0) >= float(row.get("target_recall_floor") or 0)
            and (row.get("selected_accuracy_mean") or 0) >= float(row.get("accuracy_floor") or 0)
        ]
        lines.extend(
            [
                f"## {dataset}\n\n",
                f"Target: `{ranked[0].get('target_class')}`. Floors: recall `{ranked[0].get('target_recall_floor')}`, "
                f"{ranked[0].get('selection_accuracy_metric')} `{ranked[0].get('accuracy_floor')}`.\n\n",
                "### Top Overall Rows\n\n",
                table_rows(ranked, 10),
                "\n### Top NICME Rows\n\n",
                table_rows(nicme, 10) if nicme else "No NICME rows found.\n",
                "\n### Strict All-Seed Floor Rows\n\n",
                table_rows(floor, 10) if floor else "No aggregate row met both floors in all seeds.\n",
                "\n### Mean-Floor-Compliant Rows\n\n",
                table_rows(mean_floor, 10) if mean_floor else "No aggregate row met both floors on mean metrics.\n",
                "\n",
            ]
        )
        if ranked:
            best = ranked[0]
            lines.append(
                f"Best mean selection row: `{best['model']} + {best['method']} + {best['mode']}` "
                f"with selection `{fmt(best.get('selection_score_mean'))}`, nATC `{fmt(best.get('normalized_atc_mean'))}`, "
                f"target recall `{fmt(best.get('target_recall_mean'))}`, and selected accuracy `{fmt(best.get('selected_accuracy_mean'))}`.\n\n"
            )
        if nicme:
            best_nicme = nicme[0]
            lines.append(
                f"Best NICME row: `{best_nicme['model']} + {best_nicme['method']} + {best_nicme['mode']}` "
                f"with selection `{fmt(best_nicme.get('selection_score_mean'))}`, nATC `{fmt(best_nicme.get('normalized_atc_mean'))}`, "
                f"target recall `{fmt(best_nicme.get('target_recall_mean'))}`, and selected accuracy `{fmt(best_nicme.get('selected_accuracy_mean'))}`.\n\n"
            )

    lines.extend(
        [
            "## Artifacts\n\n",
            f"- Full decision rows: `{path.parent / (phase + '_full_decision_rows.csv')}`\n",
            f"- Aggregate summary: `{path.parent / (phase + '_aggregate_summary.csv')}`\n",
            f"- Ranked summary: `{path.parent / (phase + '_ranked_summary.csv')}`\n",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate stop-gated NICME results")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rows = decision_rows(Path(args.run_log))
    aggregates = aggregate_rows(rows)

    write_csv(output_root / f"{args.phase}_full_decision_rows.csv", rows, FULL_FIELDNAMES)
    write_csv(output_root / f"{args.phase}_aggregate_summary.csv", aggregates)
    write_csv(output_root / f"{args.phase}_ranked_summary.csv", aggregates)
    (output_root / f"{args.phase}_full_decision_rows.json").write_text(json.dumps(rows, indent=2))
    (output_root / f"{args.phase}_aggregate_summary.json").write_text(json.dumps(aggregates, indent=2))
    write_markdown(output_root / f"{args.phase}_scientific_read.md", args.phase, rows, aggregates)
    print(f"Wrote {len(rows)} decision rows and {len(aggregates)} aggregate rows to {output_root}")


if __name__ == "__main__":
    main()
