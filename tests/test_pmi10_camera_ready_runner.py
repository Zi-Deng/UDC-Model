import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts import run_pmi10_camera_ready_lr5e5 as camera
from scripts import run_pmi10_sota_baselines as baseline


def args(tmp_path):
    return argparse.Namespace(
        phase="dry-run",
        split_dir=str(tmp_path / "splits" / "balanced"),
        output_root=str(tmp_path / "camera_ready"),
        seeds=(42, 43, 44),
        per_run_timeout_minutes=1,
        allow_incomplete_analysis=False,
        skip_nvidia_smi=True,
    )


def test_dry_run_builds_three_seed_unique_training_configs(tmp_path):
    rows = camera.write_dry_run(args(tmp_path))

    assert len(rows) == 3 * len(camera.TRAIN_METHODS)
    assert {int(row["seed"]) for row in rows} == {42, 43, 44}
    assert len({row["config_hash"] for row in rows}) == len(rows)
    assert len({row["checkpoint_path"] for row in rows}) == len(rows)


def test_primary_nicme_config_is_pilot_best(tmp_path):
    cfg = camera.build_training_config(
        "run",
        camera.NICME_PRIMARY_METHOD,
        tmp_path / "splits" / "balanced",
        tmp_path / "results",
        42,
    )

    assert cfg["loss_function"] == "nicme_v3_hybrid"
    assert cfg["sota_method"] == camera.NICME_PRIMARY_METHOD
    assert cfg["learning_rate"] == 5e-5
    assert cfg["parent_learning_rate"] == 5e-5
    assert cfg["nicme_logit_cost_scale"] == 0.5
    assert cfg["cs_lambda"] == 0.07
    assert cfg["cs_warmup_epochs"] == 2
    assert cfg["pilot_selected_hyperparameters"] is True


def test_baseline_hyperparameters_match_existing_lr5e5_profile(tmp_path):
    split_dir = tmp_path / "splits" / "balanced"
    root = tmp_path / "results"
    for method in (
        "ce_anchor_pretty",
        "cost_weighted_ce",
        "menon_logit_adjusted",
        "balanced_softmax",
        "class_balanced_focal",
        "ldam_drw",
        "ap_csada",
        "sosr_cnn",
        "csada",
        "nicme_v2_hybrid_pretty",
    ):
        expected = baseline.base_config(
            "run",
            method,
            split_dir,
            root,
            42,
            comparison_profile="pretty_balanced",
            learning_rate_profile="lr5e5",
        )
        actual = camera.build_training_config("run", method, split_dir, root, 42)

        ignored = {
            "output_dir",
            "sota_output_root",
            "sota_comparison_profile",
            "camera_ready",
            "camera_ready_seed",
        }
        assert {k: v for k, v in actual.items() if k not in ignored} == {
            k: v for k, v in expected.items() if k not in ignored
        }


def test_aggregation_uses_sample_standard_deviation():
    rows = [
        {"seed": 42, "method": camera.NICME_PRIMARY_METHOD, "decision_mode": "argmax", "target_recall_min": 0.9, "target_recall_macro": 0.9, "normalized_atc": 0.03, "atc": 0.3, "balanced_accuracy": 0.8, "macro_f1": 0.8, "critical_pair_error_count": 3},
        {"seed": 43, "method": camera.NICME_PRIMARY_METHOD, "decision_mode": "argmax", "target_recall_min": 0.8, "target_recall_macro": 0.8, "normalized_atc": 0.02, "atc": 0.2, "balanced_accuracy": 0.7, "macro_f1": 0.7, "critical_pair_error_count": 2},
        {"seed": 44, "method": camera.NICME_PRIMARY_METHOD, "decision_mode": "argmax", "target_recall_min": 1.0, "target_recall_macro": 1.0, "normalized_atc": 0.01, "atc": 0.1, "balanced_accuracy": 0.9, "macro_f1": 0.9, "critical_pair_error_count": 1},
        {"seed": 42, "method": "csada", "decision_mode": "argmax", "target_recall_min": 0.1, "target_recall_macro": 0.1, "normalized_atc": 0.10, "atc": 1.0, "balanced_accuracy": 0.1, "macro_f1": 0.1, "critical_pair_error_count": 9},
        {"seed": 43, "method": "csada", "decision_mode": "argmax", "target_recall_min": 0.2, "target_recall_macro": 0.2, "normalized_atc": 0.20, "atc": 2.0, "balanced_accuracy": 0.2, "macro_f1": 0.2, "critical_pair_error_count": 8},
        {"seed": 44, "method": "csada", "decision_mode": "argmax", "target_recall_min": 0.3, "target_recall_macro": 0.3, "normalized_atc": 0.30, "atc": 3.0, "balanced_accuracy": 0.3, "macro_f1": 0.3, "critical_pair_error_count": 7},
    ]

    aggregate = {row["method"]: row for row in camera.aggregate_metric_rows(rows)}
    nicme = aggregate[camera.NICME_PRIMARY_METHOD]

    assert nicme["target_recall_min_mean"] == pytest.approx(0.9)
    assert nicme["target_recall_min_std"] == pytest.approx(0.1)
    assert nicme["critical_pair_error_count_total"] == 6


def test_endpoint_winners_report_non_nicme_winners():
    aggregate = [
        {
            "method": camera.NICME_PRIMARY_METHOD,
            "display_name": "NICME",
            "target_recall_min_mean": 0.9,
            "target_recall_macro_mean": 0.9,
            "normalized_atc_mean": 0.02,
            "atc_mean": 0.2,
            "balanced_accuracy_mean": 0.9,
            "macro_f1_mean": 0.9,
            "critical_pair_error_count_mean": 2.0,
        },
        {
            "method": "csada",
            "display_name": "CSADA",
            "target_recall_min_mean": 1.0,
            "target_recall_macro_mean": 1.0,
            "normalized_atc_mean": 0.03,
            "atc_mean": 0.3,
            "balanced_accuracy_mean": 0.8,
            "macro_f1_mean": 0.8,
            "critical_pair_error_count_mean": 1.0,
        },
    ]

    winners = camera.endpoint_winners(aggregate)
    by_endpoint = {(row["endpoint"], row["method"]) for row in winners}

    assert ("normalized_atc", camera.NICME_PRIMARY_METHOD) in by_endpoint
    assert ("critical_pair_error_count", "csada") in by_endpoint


def metric_payload(value):
    report = {
        "target_recall_min": value,
        "target_recall_macro": value,
        "normalized_atc": 1.0 - value,
        "atc": 10.0 * (1.0 - value),
        "balanced_accuracy": value,
        "accuracy": value,
        "macro_f1": value,
        "critical_pair_errors": {},
        "cost_matrix_sha256": "hash",
    }
    return {"decision_reports": {"argmax": report, "cost_min": report, "sosr_argmin": report}}


def test_analysis_refuses_incomplete_rows(tmp_path):
    root = tmp_path / "camera_ready"
    seed_root = camera.seed_root(root, 42)
    seed_root.mkdir(parents=True)
    baseline.write_ledger(seed_root / "run" / "run_ledger.csv", [])
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    run_args.seeds = (42,)

    with pytest.raises(RuntimeError, match="Missing run ledger|incomplete"):
        camera.analyze(run_args)


def test_analysis_writes_winner_report_from_fake_ledgers(tmp_path):
    root = tmp_path / "camera_ready"
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    run_args.seeds = (42, 43, 44)

    for seed in run_args.seeds:
        rows = []
        for idx, method in enumerate(camera.TRAIN_METHODS, start=1):
            metric_path = root / "metrics" / f"{seed}_{method}.json"
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            value = 0.9 if method == camera.NICME_PRIMARY_METHOD else 0.5
            metric_path.write_text(json.dumps(metric_payload(value)))
            cfg = camera.build_training_config("run", method, tmp_path / "split", camera.seed_root(root, seed), seed)
            config_path = root / "configs" / f"{seed}_{method}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(cfg))
            row = baseline.row_from_config(cfg, "run", idx, status="completed")
            row["config_path"] = str(config_path)
            row["metric_path"] = str(metric_path)
            rows.append(row)
        baseline.write_ledger(camera.seed_root(root, seed) / "run" / "run_ledger.csv", rows)

    camera.analyze(run_args)

    assert (root / "analysis" / "aggregate_metrics.csv").exists()
    winners = (root / "analysis" / "cost_sensitive_winners.md").read_text()
    assert "NICME" in winners
    assert "critical_pair_error_count" in winners


def test_preserve_failed_attempt_copies_audit_artifacts(tmp_path):
    ledger_path = tmp_path / "seed_43" / "run" / "run_ledger.csv"
    config_path = tmp_path / "config.json"
    stdout_path = tmp_path / "failed.stdout.log"
    stderr_path = tmp_path / "failed.stderr.log"
    config_path.write_text('{"ok": true}')
    stdout_path.write_text("")
    stderr_path.write_text("")
    row = {
        **baseline.row_from_config(
            camera.build_training_config("run", "class_balanced_focal", tmp_path / "split", tmp_path / "results", 43),
            "run",
            7,
            status="failed",
        ),
        "config_path": str(config_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "returncode": -11,
        "failure_reason": "returncode_-11",
    }

    manifest = camera.preserve_failed_attempt(row, ledger_path, "unit_test")

    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["preserve_reason"] == "unit_test"
    assert Path(payload["copied_files"]["config_path"]).exists()
    csv_path = ledger_path.parent / "failed_attempts" / "failed_attempts.csv"
    rows = list(csv.DictReader(open(csv_path)))
    assert rows[-1]["returncode"] == "-11"
    assert rows[-1]["preserve_reason"] == "unit_test"


def test_execute_config_retries_native_crash_once(monkeypatch, tmp_path):
    cfg = camera.build_training_config("run", "class_balanced_focal", tmp_path / "split", tmp_path / "results", 43)
    run_args = args(tmp_path)
    ledger_path = tmp_path / "run" / "run_ledger.csv"
    calls = []

    def fake_execute_config(config, phase, run_id, execute_args):
        calls.append(config)
        row = baseline.row_from_config(config, phase, run_id, status="failed")
        row["returncode"] = -11
        row["failure_reason"] = "returncode_-11"
        row["config_path"] = str(tmp_path / f"config_{len(calls)}.json")
        row["stdout_log"] = str(tmp_path / f"stdout_{len(calls)}.log")
        row["stderr_log"] = str(tmp_path / f"stderr_{len(calls)}.log")
        Path(row["config_path"]).write_text(json.dumps(config))
        Path(row["stdout_log"]).write_text("")
        Path(row["stderr_log"]).write_text("")
        if len(calls) == 2:
            row["status"] = "completed"
            row["returncode"] = 0
            row["failure_reason"] = ""
            row["metric_path"] = str(tmp_path / "metrics.json")
        return row

    monkeypatch.setattr(baseline, "execute_config", fake_execute_config)

    row = camera.execute_config_with_camera_retry(cfg, "run", 7, run_args, ledger_path)

    assert row["status"] == "completed"
    assert row["failure_reason"] == "succeeded_after_native_returncode_-11_retry"
    assert len(calls) == 2
    assert calls[1]["output_dir"].endswith("_native_retry2")
    assert (ledger_path.parent / "failed_attempts" / "failed_attempts.csv").exists()
