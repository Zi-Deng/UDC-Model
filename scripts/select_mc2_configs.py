"""Select MC3 multiclass runs from a completed MC2 ledger.

Selection is recall-first: candidates are ranked by cared-class recall, and
normalized ATC breaks ties inside a configurable recall window. The script also
keeps CE and the best non-NICME baseline so MC3 comparisons remain fair even
when a NICME variant is the top recall candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NICME_METHODS = {"nicme_logit_adjustment", "nicme_hybrid"}
BASELINE_METHODS = {"ce", "menon_logit_adjusted", "cs_regularized_ce"}


@dataclass(frozen=True)
class Candidate:
    dataset: str
    variant: str
    model_family: str
    method: str
    seed: int
    config_hash: str
    config_path: str
    metric_path: str
    target_recall: float
    normalized_atc: float
    balanced_accuracy: float
    macro_f1: float
    status: str


def _read_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _metric_path_for(config: dict[str, Any]) -> Path | None:
    model = config["model_type"]
    loss = config["loss_function"]
    output_dir = config["output_dir"]
    result_root = Path("results") / f"{model}_test"
    candidates = sorted(result_root.glob(f"{output_dir}_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        metric_files = sorted(candidate.glob(f"metrics_*_{loss}.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if metric_files:
            return metric_files[0]
    return None


def _overall_metrics(metric_path: Path | None) -> dict[str, Any]:
    if metric_path is None:
        return {}
    payload = _read_json(metric_path)
    return payload.get("overall_metrics", {})


def load_candidates(ledger_path: Path) -> list[Candidate]:
    rows: list[Candidate] = []
    with open(ledger_path, newline="") as f:
        for row in csv.DictReader(f):
            config = _read_json(row["config_path"])
            metric_path = _metric_path_for(config)
            metrics = _overall_metrics(metric_path)
            rows.append(
                Candidate(
                    dataset=row["dataset"],
                    variant=row["variant"],
                    model_family=row["model_family"],
                    method=row["method"],
                    seed=int(row["seed"]),
                    config_hash=row["config_hash"],
                    config_path=row["config_path"],
                    metric_path=str(metric_path or ""),
                    target_recall=float(metrics.get("target_recall", 0.0)),
                    normalized_atc=float(metrics.get("normalized_atc", float("inf"))),
                    balanced_accuracy=float(metrics.get("balanced_accuracy", 0.0)),
                    macro_f1=float(metrics.get("macro_f1", metrics.get("f1_score", 0.0))),
                    status=row["status"],
                )
            )
    return rows


def _best(candidates: list[Candidate], recall_window: float = 0.005) -> Candidate | None:
    completed = [candidate for candidate in candidates if candidate.status == "completed" and candidate.metric_path]
    if not completed:
        return None
    best_recall = max(candidate.target_recall for candidate in completed)
    tied = [candidate for candidate in completed if best_recall - candidate.target_recall <= recall_window]
    return sorted(tied, key=lambda candidate: (candidate.normalized_atc, -candidate.balanced_accuracy))[0]


def select_for_dataset(candidates: list[Candidate], recall_window: float) -> dict[str, Any]:
    completed = [candidate for candidate in candidates if candidate.status == "completed" and candidate.metric_path]
    selected: dict[tuple[str, str], Candidate] = {}

    top_overall = _best(completed, recall_window)
    if top_overall:
        selected[(top_overall.model_family, top_overall.method)] = top_overall

    top_nicme = _best([candidate for candidate in completed if candidate.method in NICME_METHODS], recall_window)
    if top_nicme:
        selected[(top_nicme.model_family, top_nicme.method)] = top_nicme

    top_non_nicme = _best(
        [candidate for candidate in completed if candidate.method in BASELINE_METHODS and candidate.method != "ce"],
        recall_window,
    )
    if top_non_nicme:
        selected[(top_non_nicme.model_family, top_non_nicme.method)] = top_non_nicme

    top_ce = _best([candidate for candidate in completed if candidate.method == "ce"], recall_window)
    if top_ce:
        selected[(top_ce.model_family, top_ce.method)] = top_ce

    ordered = sorted(
        selected.values(),
        key=lambda candidate: (
            candidate.method != "ce",
            candidate.method not in BASELINE_METHODS,
            -candidate.target_recall,
            candidate.normalized_atc,
        ),
    )
    return {
        "selected": [candidate.__dict__ for candidate in ordered],
        "completed_count": len(completed),
        "failed_count": len([candidate for candidate in candidates if candidate.status != "completed"]),
    }


def write_outputs(candidates: list[Candidate], output_dir: Path, recall_window: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_dataset.setdefault(candidate.dataset, []).append(candidate)

    selection = {
        dataset: select_for_dataset(dataset_candidates, recall_window)
        for dataset, dataset_candidates in sorted(by_dataset.items())
    }
    with open(output_dir / "mc2_selection.json", "w") as f:
        json.dump(selection, f, indent=2)

    with open(output_dir / "mc3_launch_commands.sh", "w") as f:
        f.write(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cd /mnt/storage/github/NICME\n"
            'export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."\n'
        )
        for dataset, dataset_selection in selection.items():
            for candidate in dataset_selection["selected"]:
                f.write(
                    "micromamba run -n ml python scripts/run_multiclass_experiments.py "
                    f"--stop mc3 --datasets {dataset} --variants {candidate['variant']} "
                    f"--models {candidate['model_family']} --methods {candidate['method']} "
                    "--output-dir results/multiclass_mc3_selected_20260430 "
                    "--execute --per-run-timeout-minutes 120\n"
                )

    with open(output_dir / "mc2_selection.md", "w") as f:
        f.write("# MC2 Selection\n\n")
        f.write(f"- Recall tie window: {recall_window:.4f}\n\n")
        for dataset, dataset_selection in selection.items():
            f.write(f"## {dataset}\n\n")
            f.write(
                f"- Completed candidates: {dataset_selection['completed_count']}\n"
                f"- Failed candidates: {dataset_selection['failed_count']}\n\n"
            )
            f.write("| Model | Method | Target Recall | Normalized ATC | Balanced Accuracy | Metrics |\n")
            f.write("|---|---|---:|---:|---:|---|\n")
            for candidate in dataset_selection["selected"]:
                f.write(
                    f"| {candidate['model_family']} | {candidate['method']} | "
                    f"{candidate['target_recall']:.4f} | {candidate['normalized_atc']:.4f} | "
                    f"{candidate['balanced_accuracy']:.4f} | `{candidate['metric_path']}` |\n"
                )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select MC3 configs from MC2 results")
    parser.add_argument("--ledger", default="results/multiclass_mc2_20260430/mc2/run_ledger.csv")
    parser.add_argument("--output-dir", default="results/multiclass_mc2_20260430/selection")
    parser.add_argument("--recall-window", type=float, default=0.005)
    args = parser.parse_args()
    candidates = load_candidates(Path(args.ledger))
    write_outputs(candidates, Path(args.output_dir), args.recall_window)


if __name__ == "__main__":
    main()
