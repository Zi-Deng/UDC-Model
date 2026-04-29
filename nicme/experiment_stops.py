"""Stop-gated experiment reporting for NICME binary paper runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

STOP_PROTOCOL_PLAN = """# NICME Stop-Gated Experimental Plan With Plan Versioning

Run NICME experiments as a staged, NeurIPS-quality evaluation of cared-class recall and ATC/normalized ATC under explicit user-defined cost matrices, while separating cost sensitivity from class imbalance.

At every stop, Codex must:
- Summarize results from the just-completed stage.
- Archive the plan that governed that stage into both `docs/` and `memory/`.
- Construct a new revised plan for the next stage using the observed results.
- Ask the user to inspect results and explicitly continue before any next-stage work.

Stop rule:
- Do not begin Stop `N+1` execution until Stop `N` has a saved summary/next-plan artifact and the user has approved continuing.
"""

STOP_NEXT_PLANS = {
    0: """## Next Plan: Stop 1 Smoke Runs

- Use only audited, approved data splits.
- Run balanced spider and a balanced BreaKHis subset.
- Models: ConvNeXt, ViT, DINOv3-ViT.
- Methods: `ce`, `nicme_hybrid`.
- Seeds: 1; epochs: 1.
- Confirm no crashes, no NaNs, valid metric export, and no calibration/test leakage.
- Record runtime per run and projected Tier 1 runtime.
""",
    1: """## Next Plan: Stop 2 Prototype Runs

- Use the non-gated timm DINOv3 ViT-S equivalent for the interim Stop 2 prototype runs while Hugging Face approval for the official Meta DINOv3 repositories is pending.
- Preserve the official `facebook/dinov3-*` presets and rerun/append official DINOv3 results after gated access is approved.
- Datasets: balanced spider and balanced BreaKHis.
- Models: ConvNeXt and timm DINOv3-ViT LoRA.
- Methods: `ce`, `ce_calibrated_cost_min`, `menon_logit_adjusted`, `nicme_hybrid`.
- Seeds: 1.
- Epochs: 5 for spider, 8 for BreaKHis, early stopping enabled.
- Target runtime: under 1 hour total on RTX 5090.
- Continue only if at least one cost-sensitive path improves ATC or cared-class recall without unacceptable accuracy collapse.
""",
    2: """## Next Plan: Stop 3 Main Binary Paper Runs

- Proceed with Stop 3 only after the user reviews the Stop 2 prototype metrics and explicitly approves continuing.
- Treat Stop 2 as operationally successful but scientifically mixed: several cost-sensitive inference paths improved cared-class recall/ATC, but often by accepting large accuracy drops on small prototype splits.
- Use non-gated timm DINOv3 LoRA as the main DINOv3 path while official Meta `facebook/dinov3-*` checkpoints remain pending Hugging Face approval.
- Keep official Meta DINOv3 rows out of the main table until they can be run under the same stop-gated protocol.
- Datasets: balanced spider, spider target-minority, spider target-majority, balanced BreaKHis, natural BreaKHis.
- Primary model: `timm_dinov3_vit_lora`.
- Runtime/control models: ConvNeXt and ViT only where needed to anchor interpretation against Stop 2 behavior.
- Optional PEFT backbone comparison: `timm_dinov3_convnext_lora`, reported explicitly as ConvNeXt MLP-LoRA rather than attention LoRA.
- Methods: all six implemented methods, with special attention to `ce`, `ce_calibrated_cost_min`, `cs_regularized_ce`, `nicme_logit_adjustment`, and `nicme_hybrid`.
- Seeds: 3 for all methods first; extend to 5 seeds only for `ce`, `ce_calibrated_cost_min`, and `nicme_hybrid` if projected runtime and disk usage remain within the one-day target.
- Epoch cap: 20; early stopping patience 3; `save_total_limit=1`.
- Before launching the full grid, confirm checkpoint cleanup leaves at least 20 GB free, because Stop 2 exposed checkpoint growth as the main operational bottleneck.
- Continue paper-level claims only if balanced-dataset results show ATC or cared-class recall gains without violating the configured accuracy or balanced-accuracy floors.
""",
    3: """## Next Plan: Stop 4 Ablations And Robustness

- Backbone ablation on balanced spider and balanced BreaKHis: ConvNeXt, ViT, DINOv3-ConvNeXt, DINOv3-ViT.
- Methods for backbone ablation: `ce_calibrated_cost_min`, `nicme_hybrid`; seeds: 3.
- Cost-ratio sensitivity: 1:1, 2:1, 5:1, 10:1, 20:1 on balanced spider and balanced BreaKHis.
- Calibration ablation: argmax vs calibrated-cost-min for all trained models.
- Component ablation: CE, CS regularizer only, NICME logit only, NICME hybrid.
""",
    4: """## Next Plan: Final Paper Experiment Summary

- Freeze or extend experiments based on Stop 4 results.
- Produce the final paper-facing tables, unresolved-risk memo, and conservative claim wording.
- Do not add new experiments unless the Stop 4 checkpoint identifies a concrete gap.
""",
}

USER_CHECKPOINT = "Are the results satisfactory enough to continue, revise, or pause?"
SPLIT_NAMES = ("train", "validation", "calibration", "test")
TARGET_SPLIT_RATIOS = {"train": 0.70, "validation": 0.10, "calibration": 0.10, "test": 0.10}


def _named_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(value)
    return path.name, path


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _json_dumpable_counts(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.sort_index().items()}


def summarize_split_dir(split_dir: Path) -> dict[str, Any]:
    """Summarize a prepared split directory containing train/validation/calibration/test CSV files."""
    split_dir = Path(split_dir)
    summary: dict[str, Any] = {
        "path": str(split_dir),
        "exists": split_dir.exists(),
        "splits": {},
        "patient_overlap": {},
        "total_rows": 0,
        "ratio_check": {},
    }
    if not split_dir.exists():
        return summary

    patient_sets: dict[str, set[str]] = {}
    for split_name in SPLIT_NAMES:
        csv_path = split_dir / f"{split_name}.csv"
        if not csv_path.exists():
            summary["splits"][split_name] = {"exists": False}
            continue

        frame = pd.read_csv(csv_path)
        row_count = int(len(frame))
        summary["total_rows"] += row_count
        split_summary: dict[str, Any] = {"exists": True, "rows": row_count}

        if "label" in frame.columns:
            counts = frame["label"].value_counts()
            split_summary["label_counts"] = _json_dumpable_counts(counts)
            split_summary["label_prevalence"] = {
                str(key): float(value / max(row_count, 1)) for key, value in counts.sort_index().items()
            }
        if "label_name" in frame.columns:
            split_summary["label_name_counts"] = _json_dumpable_counts(frame["label_name"].value_counts())
        if "patient_id" in frame.columns:
            patients = set(frame["patient_id"].dropna().astype(str))
            patient_sets[split_name] = patients
            split_summary["unique_patients"] = len(patients)
        if "magnification" in frame.columns:
            split_summary["magnification_counts"] = _json_dumpable_counts(frame["magnification"].value_counts())
        if "tumor_type" in frame.columns:
            split_summary["tumor_type_counts"] = _json_dumpable_counts(frame["tumor_type"].value_counts())
        if "image_path" in frame.columns:
            paths = [Path(str(path)) for path in frame["image_path"]]
            split_summary["missing_images"] = int(sum(not path.exists() for path in paths))

        summary["splits"][split_name] = split_summary

    total = max(int(summary["total_rows"]), 1)
    for split_name, split_summary in summary["splits"].items():
        if not split_summary.get("exists"):
            continue
        observed = split_summary["rows"] / total
        target = TARGET_SPLIT_RATIOS[split_name]
        summary["ratio_check"][split_name] = {
            "observed": observed,
            "target": target,
            "absolute_error": abs(observed - target),
            "within_5pp": abs(observed - target) <= 0.05,
        }

    for left, right in combinations(sorted(patient_sets), 2):
        overlap = len(patient_sets[left] & patient_sets[right])
        summary["patient_overlap"][f"{left}__{right}"] = overlap

    return summary


def _find_metrics_files(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(results_dir.rglob("metrics_*.json"))


def summarize_results_dir(results_dir: Path) -> dict[str, Any]:
    """Summarize NICME metrics JSON files under a results directory."""
    metrics_files = _find_metrics_files(results_dir)
    rows = []
    for metrics_file in metrics_files:
        with open(metrics_file) as f:
            payload = json.load(f)
        overall = payload.get("overall_metrics", {})
        reports = payload.get("decision_reports", {})
        base_row = {
            "file": str(metrics_file),
            "run": metrics_file.parent.name,
            "accuracy": overall.get("accuracy"),
            "balanced_accuracy": overall.get("balanced_accuracy"),
            "macro_f1": overall.get("macro_f1"),
            "crr": overall.get("crr"),
            "atc": overall.get("atc"),
            "normalized_atc": overall.get("normalized_atc"),
            "selection_score": overall.get("selection_score"),
            "target_recall": overall.get("target_recall"),
            "target_fnr": overall.get("target_fnr"),
            "target_precision": overall.get("target_precision"),
            "target_fpr": overall.get("target_fpr"),
            "nll": overall.get("nll"),
            "brier": overall.get("brier"),
            "ece": overall.get("ece"),
            "auroc": overall.get("auroc"),
            "auprc": overall.get("auprc"),
            "temperature": overall.get("temperature"),
            "confusion_matrix": payload.get("confusion_matrix"),
            "class_prevalence": overall.get("class_prevalence"),
        }
        if not reports:
            rows.append(base_row)
            continue
        for mode, report in reports.items():
            row = dict(base_row)
            row["decision_mode"] = mode
            for key in [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "crr",
                "atc",
                "normalized_atc",
                "selection_score",
                "target_recall",
                "target_fnr",
                "target_precision",
                "target_fpr",
                "nll",
                "brier",
                "ece",
                "auroc",
                "auprc",
                "temperature",
                "confusion_matrix",
                "class_prevalence",
            ]:
                if key in report:
                    row[key] = report.get(key)
            rows.append(row)
    return {"path": str(results_dir), "exists": results_dir.exists(), "metrics_files": len(metrics_files), "rows": rows}


def summarize_run_log(run_log_path: Path) -> dict[str, Any]:
    """Summarize a Stop runner run_log.json file, including failures and runtime."""
    run_log_path = Path(run_log_path)
    summary: dict[str, Any] = {
        "path": str(run_log_path),
        "exists": run_log_path.exists(),
        "rows": [],
        "total_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "total_elapsed_seconds": 0.0,
        "successful_elapsed_seconds": 0.0,
    }
    if not run_log_path.exists():
        return summary

    with open(run_log_path) as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected run log to contain a JSON list: {run_log_path}")

    rows = []
    for item in payload:
        returncode = int(item.get("returncode", 1))
        elapsed = float(item.get("elapsed_seconds", 0.0) or 0.0)
        status = "ok" if returncode == 0 else "failed"
        row = {
            "index": item.get("index"),
            "name": item.get("name"),
            "dataset": item.get("dataset"),
            "model_family": item.get("model_family"),
            "method": item.get("method"),
            "returncode": returncode,
            "status": status,
            "elapsed_seconds": elapsed,
            "config": item.get("config"),
            "stdout_log": item.get("stdout_log"),
            "stderr_log": item.get("stderr_log"),
        }
        rows.append(row)
        summary["total_runs"] += 1
        summary["total_elapsed_seconds"] += elapsed
        if returncode == 0:
            summary["successful_runs"] += 1
            summary["successful_elapsed_seconds"] += elapsed
        else:
            summary["failed_runs"] += 1

    summary["rows"] = rows
    successful = max(int(summary["successful_runs"]), 1)
    summary["mean_successful_elapsed_seconds"] = summary["successful_elapsed_seconds"] / successful
    return summary


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join([header, sep, *body])


def render_split_summary(name: str, summary: dict[str, Any]) -> str:
    lines = [f"### Data Audit: {name}", "", f"- Path: `{summary['path']}`", f"- Exists: `{summary['exists']}`"]
    if not summary["exists"]:
        return "\n".join(lines)

    rows = []
    for split_name in SPLIT_NAMES:
        split = summary["splits"].get(split_name, {})
        if not split.get("exists"):
            rows.append([split_name, "missing", "", "", "", "", "", "", "", "", ""])
            continue
        ratio = summary["ratio_check"].get(split_name, {})
        rows.append(
            [
                split_name,
                split.get("rows", 0),
                _pct(ratio.get("observed", 0.0)),
                _pct(ratio.get("target", TARGET_SPLIT_RATIOS[split_name])),
                split.get("label_counts", {}),
                split.get("label_prevalence", {}),
                split.get("label_name_counts", {}),
                split.get("unique_patients", ""),
                split.get("missing_images", ""),
                split.get("magnification_counts", {}),
                split.get("tumor_type_counts", {}),
            ]
        )
    lines.extend(
        [
            "",
            _markdown_table(
                [
                    "split",
                    "rows",
                    "observed ratio",
                    "target ratio",
                    "label counts",
                    "label prevalence",
                    "label-name counts",
                    "unique patients",
                    "missing images",
                    "magnification counts",
                    "tumor-type counts",
                ],
                rows,
            ),
            "",
            f"- Patient overlap: `{summary.get('patient_overlap', {})}`",
        ]
    )
    return "\n".join(lines)


def render_results_summary(name: str, summary: dict[str, Any]) -> str:
    lines = [f"### Metrics Summary: {name}", "", f"- Path: `{summary['path']}`", f"- Exists: `{summary['exists']}`"]
    if not summary["exists"] or not summary["rows"]:
        lines.append("- Metrics files: `0`")
        return "\n".join(lines)

    rows = []
    for row in summary["rows"]:
        rows.append(
            [
                row.get("run"),
                row.get("decision_mode", "overall"),
                row.get("normalized_atc"),
                row.get("target_recall"),
                row.get("target_fnr"),
                row.get("target_precision"),
                row.get("target_fpr"),
                row.get("accuracy"),
                row.get("balanced_accuracy"),
                row.get("macro_f1"),
                row.get("auroc"),
                row.get("auprc"),
                row.get("nll"),
                row.get("brier"),
                row.get("ece"),
                row.get("temperature"),
                row.get("confusion_matrix"),
            ]
        )
    lines.extend(
        [
            f"- Metrics files: `{summary['metrics_files']}`",
            "",
            _markdown_table(
                [
                    "run",
                    "mode",
                    "normalized ATC",
                    "target recall",
                    "target FNR",
                    "target precision",
                    "target FPR",
                    "accuracy",
                    "balanced accuracy",
                    "macro-F1",
                    "AUROC",
                    "AUPRC",
                    "NLL",
                    "Brier",
                    "ECE",
                    "T",
                    "confusion matrix",
                ],
                rows,
            ),
        ]
    )
    return "\n".join(lines)


def render_run_log_summary(name: str, summary: dict[str, Any]) -> str:
    lines = [f"### Run Log: {name}", "", f"- Path: `{summary['path']}`", f"- Exists: `{summary['exists']}`"]
    if not summary["exists"]:
        return "\n".join(lines)

    lines.extend(
        [
            f"- Total runs: `{summary['total_runs']}`",
            f"- Successful runs: `{summary['successful_runs']}`",
            f"- Failed runs: `{summary['failed_runs']}`",
            f"- Total elapsed seconds: `{summary['total_elapsed_seconds']:.2f}`",
            f"- Mean successful-run elapsed seconds: `{summary['mean_successful_elapsed_seconds']:.2f}`",
            "",
        ]
    )
    rows = []
    for row in summary["rows"]:
        rows.append(
            [
                row.get("index"),
                row.get("dataset"),
                row.get("model_family"),
                row.get("method"),
                row.get("status"),
                row.get("returncode"),
                f"{row.get('elapsed_seconds', 0.0):.2f}",
                row.get("stderr_log", ""),
            ]
        )
    lines.append(
        _markdown_table(
            ["#", "dataset", "model", "method", "status", "rc", "seconds", "stderr log"],
            rows,
        )
    )
    return "\n".join(lines)


def build_stop_report(
    stop: int,
    archived_plan: str,
    split_summaries: dict[str, dict[str, Any]],
    result_summaries: dict[str, dict[str, Any]],
    run_log_summaries: dict[str, dict[str, Any]],
    notes: str,
) -> str:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    observed_sections = []
    if split_summaries:
        observed_sections.extend(render_split_summary(name, summary) for name, summary in split_summaries.items())
    if result_summaries:
        observed_sections.extend(render_results_summary(name, summary) for name, summary in result_summaries.items())
    if run_log_summaries:
        observed_sections.extend(render_run_log_summary(name, summary) for name, summary in run_log_summaries.items())
    if not observed_sections:
        observed_sections.append("_No observed split or result artifacts were supplied when this report was generated._")

    changed = notes.strip() if notes.strip() else "No manual changes or result-driven plan revisions recorded yet."
    next_plan = STOP_NEXT_PLANS.get(stop, "## Next Plan\n\nNo next stop is defined. Freeze or revise manually.")
    return f"""# STOP {stop} Results And Next Plan

Generated: {timestamp}

## Stop Status

This artifact archives the plan used for Stop {stop}, summarizes observed artifacts supplied to the stop reporter, and defines the next plan. Do not begin the next stop until the user checkpoint at the end is approved.

## Observed Results Summary

{chr(10).join(observed_sections)}

## What Changed And Why

{changed}

{next_plan}

## User Checkpoint

{USER_CHECKPOINT}

## Archived Previous Plan

{archived_plan.strip()}
"""


def write_stop_report(
    stop: int,
    archived_plan: str,
    split_summaries: dict[str, dict[str, Any]],
    result_summaries: dict[str, dict[str, Any]],
    run_log_summaries: dict[str, dict[str, Any]] | None,
    notes: str,
    docs_dir: Path = Path("docs/experiment_plans"),
    memory_dir: Path = Path("memory"),
) -> tuple[Path, Path]:
    report = build_stop_report(stop, archived_plan, split_summaries, result_summaries, run_log_summaries or {}, notes)
    docs_dir.mkdir(parents=True, exist_ok=True)
    memory_dir.mkdir(parents=True, exist_ok=True)
    docs_path = docs_dir / f"STOP_{stop}_results_and_next_plan.md"
    memory_path = memory_dir / f"experiment_plan_stop_{stop}.md"
    docs_path.write_text(report)
    memory_path.write_text(report)
    return docs_path, memory_path


def _read_archived_plan(path: str | None) -> str:
    if path is None:
        return STOP_PROTOCOL_PLAN
    plan_path = Path(path)
    if not plan_path.exists():
        raise FileNotFoundError(f"Archived plan path does not exist: {plan_path}")
    return plan_path.read_text()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate NICME stop-gated experiment reports")
    parser.add_argument("--stop", type=int, required=True, choices=range(0, 5), help="Completed stop number")
    parser.add_argument("--archived-plan", default=None, help="Plan file to archive verbatim in the stop report")
    parser.add_argument(
        "--split-dir",
        action="append",
        default=[],
        help="Prepared split directory to audit. Use name=path or just path. Repeatable.",
    )
    parser.add_argument(
        "--results-dir",
        action="append",
        default=[],
        help="Results directory containing metrics_*.json files. Use name=path or just path. Repeatable.",
    )
    parser.add_argument(
        "--run-log",
        action="append",
        default=[],
        help="Stop runner run_log.json file. Use name=path or just path. Repeatable.",
    )
    parser.add_argument("--notes", default="", help="Manual notes for 'What Changed And Why'")
    args = parser.parse_args(argv)

    archived_plan = _read_archived_plan(args.archived_plan)
    split_summaries = {name: summarize_split_dir(path) for name, path in map(_named_path, args.split_dir)}
    result_summaries = {name: summarize_results_dir(path) for name, path in map(_named_path, args.results_dir)}
    run_log_summaries = {name: summarize_run_log(path) for name, path in map(_named_path, args.run_log)}
    docs_path, memory_path = write_stop_report(
        args.stop,
        archived_plan,
        split_summaries,
        result_summaries,
        run_log_summaries,
        args.notes,
    )
    print(f"Wrote stop report: {docs_path}")
    print(f"Wrote memory report: {memory_path}")


__all__ = [
    "STOP_NEXT_PLANS",
    "STOP_PROTOCOL_PLAN",
    "USER_CHECKPOINT",
    "build_stop_report",
    "summarize_run_log",
    "summarize_results_dir",
    "summarize_split_dir",
    "write_stop_report",
]
