import json
from pathlib import Path

import pytest

pandas = pytest.importorskip("pandas")

from nicme.experiment_stops import (  # noqa: E402
    USER_CHECKPOINT,
    summarize_results_dir,
    summarize_run_log,
    summarize_split_dir,
    write_stop_report,
)


def _write_split(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pandas.DataFrame(rows).to_csv(path, index=False)


def test_summarize_split_dir_reports_ratios_overlap_and_missing_images(tmp_path):
    split_dir = tmp_path / "splits"
    image = tmp_path / "image.png"
    image.write_bytes(b"fake")
    for split_name, patient_start in [
        ("train", 0),
        ("validation", 70),
        ("calibration", 80),
        ("test", 90),
    ]:
        size = 70 if split_name == "train" else 10
        rows = [
            {
                "image_path": str(image if idx == 0 else tmp_path / f"missing_{split_name}_{idx}.png"),
                "label": idx % 2,
                "label_name": "black_widow" if idx % 2 == 0 else "false_widow",
                "patient_id": f"p{patient_start + idx}",
            }
            for idx in range(size)
        ]
        _write_split(split_dir / f"{split_name}.csv", rows)

    summary = summarize_split_dir(split_dir)

    assert summary["total_rows"] == 100
    assert summary["ratio_check"]["train"]["within_5pp"]
    assert summary["splits"]["train"]["label_counts"] == {"0": 35, "1": 35}
    assert summary["splits"]["train"]["missing_images"] == 69
    assert all(overlap == 0 for overlap in summary["patient_overlap"].values())


def test_summarize_results_dir_reads_decision_reports(tmp_path):
    result_dir = tmp_path / "results" / "run"
    result_dir.mkdir(parents=True)
    payload = {
        "overall_metrics": {
            "accuracy": 0.9,
            "balanced_accuracy": 0.88,
            "macro_f1": 0.87,
            "normalized_atc": 0.12,
            "target_recall": 0.95,
            "ece": 0.04,
        },
        "decision_reports": {
            "argmax": {"normalized_atc": 0.12, "target_recall": 0.95},
            "calibrated_cost_min": {
                "normalized_atc": 0.08,
                "target_recall": 0.98,
                "temperature": 1.4,
            },
        },
    }
    (result_dir / "metrics_20260428_nicme.json").write_text(json.dumps(payload))

    summary = summarize_results_dir(tmp_path / "results")

    assert summary["metrics_files"] == 1
    assert {row["decision_mode"] for row in summary["rows"]} == {"argmax", "calibrated_cost_min"}
    assert any(row["temperature"] == 1.4 for row in summary["rows"])


def test_summarize_run_log_reports_failures_and_runtime(tmp_path):
    run_log = tmp_path / "run_log.json"
    run_log.write_text(
        json.dumps(
            [
                {
                    "index": 1,
                    "dataset": "spider_balanced_smoke",
                    "model_family": "convnext",
                    "method": "ce",
                    "name": "ok_run",
                    "returncode": 0,
                    "elapsed_seconds": 12.5,
                    "stderr_log": "ok.stderr.log",
                },
                {
                    "index": 2,
                    "dataset": "spider_balanced_smoke",
                    "model_family": "dinov3_vit",
                    "method": "ce",
                    "name": "failed_run",
                    "returncode": 1,
                    "elapsed_seconds": 3.0,
                    "stderr_log": "failed.stderr.log",
                },
            ]
        )
    )

    summary = summarize_run_log(run_log)

    assert summary["total_runs"] == 2
    assert summary["successful_runs"] == 1
    assert summary["failed_runs"] == 1
    assert summary["mean_successful_elapsed_seconds"] == 12.5
    assert summary["rows"][1]["status"] == "failed"


def test_write_stop_report_creates_docs_and_memory_artifacts(tmp_path):
    docs_dir = tmp_path / "docs" / "experiment_plans"
    memory_dir = tmp_path / "memory"

    docs_path, memory_path = write_stop_report(
        0,
        "# Prior Plan\n\nArchive me.",
        {},
        {},
        {},
        "Data audit found no blocking changes.",
        docs_dir=docs_dir,
        memory_dir=memory_dir,
    )

    assert docs_path.exists()
    assert memory_path.exists()
    text = docs_path.read_text()
    assert "Archived Previous Plan" in text
    assert "Data audit found no blocking changes." in text
    assert USER_CHECKPOINT in text
