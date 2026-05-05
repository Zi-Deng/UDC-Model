import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts import run_pmi10_nicme_top5_lr5e5 as top5
from scripts import run_pmi10_sota_baselines as baseline


def args(tmp_path):
    return argparse.Namespace(
        phase="dry-run",
        split_dir=str(tmp_path / "splits" / "balanced"),
        output_root=str(tmp_path / "top5"),
        seeds=(42, 43, 44),
        per_run_timeout_minutes=1,
        allow_incomplete_analysis=False,
        skip_nvidia_smi=True,
    )


def test_dry_run_builds_three_seed_top5_unique_configs(tmp_path):
    rows = top5.write_dry_run(args(tmp_path))

    assert len(rows) == 3 * 5
    assert {int(row["seed"]) for row in rows} == {42, 43, 44}
    assert len({row["config_hash"] for row in rows}) == len(rows)
    assert len({row["checkpoint_path"] for row in rows}) == len(rows)


def test_top5_configs_match_selected_hpo_runs(tmp_path):
    split_dir = tmp_path / "splits" / "balanced"
    root = tmp_path / "results"
    observed = []

    for method in top5.TOP5_METHODS:
        cfg = top5.build_training_config("run", method, split_dir, root, 42)
        observed.append(
            (
                cfg["pilot_hpo_run"],
                cfg["nicme_logit_cost_scale"],
                cfg["cs_lambda"],
            )
        )

    assert observed == [
        (95, 0.50, 0.07),
        (69, 0.20, 0.09),
        (20, 0.06, 0.03),
        (50, 0.09, 0.07),
        (53, 0.09, 0.20),
    ]


def test_top5_shared_defaults_match_camera_ready_protocol(tmp_path):
    cfg = top5.build_training_config(
        "run",
        top5.TOP5_METHODS[0],
        tmp_path / "splits" / "balanced",
        tmp_path / "results",
        42,
    )

    assert cfg["loss_function"] == "nicme_hybrid"
    assert cfg["learning_rate"] == 5e-5
    assert cfg["parent_learning_rate"] == 5e-5
    assert cfg["num_train_epochs"] == 32
    assert cfg["early_stopping_patience"] == 5
    assert cfg["batch_size"] == 16
    assert cfg["gradient_accumulation_steps"] == 4
    assert cfg["lr_scheduler_type"] == "cosine"
    assert cfg["weight_decay"] == 0.005
    assert cfg["warmup_ratio"] == 0.10
    assert cfg["cs_warmup_epochs"] == 2
    assert cfg["decision_modes"] == "argmax"
    assert cfg["calibration_enabled"] is False


def test_aggregation_uses_sample_standard_deviation():
    method = top5.TOP5_METHODS[0]
    rows = [
        {
            "seed": 42,
            "method": method,
            "decision_mode": "argmax",
            "target_recall_min": 0.9,
            "target_recall_macro": 0.9,
            "normalized_atc": 0.03,
            "atc": 0.3,
            "balanced_accuracy": 0.8,
            "macro_f1": 0.8,
            "critical_pair_error_count": 3,
            "learning_rate": 5e-5,
            "parent_learning_rate": 5e-5,
            "alpha": 0.5,
            "cs_lambda": 0.07,
            "loss_function": "nicme_hybrid",
        },
        {
            "seed": 43,
            "method": method,
            "decision_mode": "argmax",
            "target_recall_min": 0.8,
            "target_recall_macro": 0.8,
            "normalized_atc": 0.02,
            "atc": 0.2,
            "balanced_accuracy": 0.7,
            "macro_f1": 0.7,
            "critical_pair_error_count": 2,
            "learning_rate": 5e-5,
            "parent_learning_rate": 5e-5,
            "alpha": 0.5,
            "cs_lambda": 0.07,
            "loss_function": "nicme_hybrid",
        },
        {
            "seed": 44,
            "method": method,
            "decision_mode": "argmax",
            "target_recall_min": 1.0,
            "target_recall_macro": 1.0,
            "normalized_atc": 0.01,
            "atc": 0.1,
            "balanced_accuracy": 0.9,
            "macro_f1": 0.9,
            "critical_pair_error_count": 1,
            "learning_rate": 5e-5,
            "parent_learning_rate": 5e-5,
            "alpha": 0.5,
            "cs_lambda": 0.07,
            "loss_function": "nicme_hybrid",
        },
    ]

    aggregate = top5.aggregate_metric_rows(rows)[0]

    assert aggregate["target_recall_min_mean"] == pytest.approx(0.9)
    assert aggregate["target_recall_min_std"] == pytest.approx(0.1)
    assert aggregate["critical_pair_error_count_total"] == 6


def metric_payload(target_recall, normalized_atc, balanced_accuracy, macro_f1, critical_errors):
    report = {
        "target_recall_min": target_recall,
        "target_recall_macro": target_recall,
        "normalized_atc": normalized_atc,
        "atc": normalized_atc * 10.0,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "critical_pair_errors": {"x->y": {"errors": critical_errors}},
        "cost_matrix_sha256": top5.REQUIRED_COST_HASH,
    }
    return {"decision_reports": {"argmax": report}}


def write_fake_completed_ledgers(root: Path, run_args: argparse.Namespace):
    scores = {
        top5.TOP5_METHODS[0]: (0.8, 0.05, 0.8, 0.8, 3),
        top5.TOP5_METHODS[1]: (0.9, 0.01, 0.9, 0.9, 1),
        top5.TOP5_METHODS[2]: (0.7, 0.06, 0.7, 0.7, 4),
        top5.TOP5_METHODS[3]: (0.6, 0.07, 0.6, 0.6, 5),
        top5.TOP5_METHODS[4]: (0.5, 0.08, 0.5, 0.5, 6),
    }
    for seed in run_args.seeds:
        rows = []
        for idx, method in enumerate(top5.TOP5_METHODS, start=1):
            metric_path = root / "metrics" / f"{seed}_{method}.json"
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_path.write_text(json.dumps(metric_payload(*scores[method])))
            cfg = top5.build_training_config("run", method, Path(run_args.split_dir), top5.seed_root(root, seed), seed)
            config_path = root / "configs" / f"{seed}_{method}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(cfg))
            row = baseline.row_from_config(cfg, "run", idx, status="completed")
            row["config_path"] = str(config_path)
            row["metric_path"] = str(metric_path)
            rows.append(row)
        baseline.write_ledger(top5.seed_root(root, seed) / "run" / "run_ledger.csv", rows)


def test_analysis_refuses_incomplete_rows(tmp_path):
    root = tmp_path / "top5"
    seed_root = top5.seed_root(root, 42)
    seed_root.mkdir(parents=True)
    baseline.write_ledger(seed_root / "run" / "run_ledger.csv", [])
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    run_args.seeds = (42,)

    with pytest.raises(RuntimeError, match="Missing run ledger|incomplete"):
        top5.analyze(run_args)


def test_analysis_preserves_all_settings_and_reports_non_run95_winner(tmp_path):
    root = tmp_path / "top5"
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    write_fake_completed_ledgers(root, run_args)

    top5.analyze(run_args)

    table = (root / "analysis" / "nicme_top5_table.md").read_text()
    winners = (root / "analysis" / "cost_sensitive_winners.md").read_text()
    aggregate_rows = list(csv.DictReader(open(root / "analysis" / "aggregate_metrics.csv")))

    assert len(aggregate_rows) == 5
    for hpo_run in ("95", "69", "20", "50", "53"):
        assert f"| {hpo_run} |" in table
    assert "run 69" in winners
    assert aggregate_rows[0]["hpo_run"] == "69"
