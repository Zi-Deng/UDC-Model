import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts import run_pmi10_sota_baselines as baseline
from scripts import run_pmi20_nicme_alpha_lambda6_lr5e5 as alpha6


def args(tmp_path):
    return argparse.Namespace(
        phase="dry-run",
        split_dir=str(tmp_path / "splits" / "balanced"),
        output_root=str(tmp_path / "pmi20_alpha_lambda6"),
        seeds=(42, 43, 44),
        per_run_timeout_minutes=1,
        allow_incomplete_analysis=False,
        skip_nvidia_smi=True,
    )


def write_split_files(split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, count in {"train": 1940, "validation": 320, "test": 640}.items():
        path = split_dir / f"{split}.csv"
        with path.open("w") as f:
            f.write("image_path,label,label_name\n")
            for idx in range(count):
                f.write(f"/tmp/{split}_{idx}.jpg,{idx % 20},class_{idx % 20}\n")


def test_dry_run_builds_three_seed_six_candidate_unique_configs(tmp_path):
    rows = alpha6.write_dry_run(args(tmp_path))

    assert len(rows) == 3 * 6
    assert {int(row["seed"]) for row in rows} == {42, 43, 44}
    assert len({row["config_hash"] for row in rows}) == len(rows)
    assert len({row["checkpoint_path"] for row in rows}) == len(rows)


def test_alpha_lambda_settings_match_plan(tmp_path):
    split_dir = tmp_path / "splits" / "balanced"
    root = tmp_path / "results"
    observed = []

    for method in alpha6.ALPHA_LAMBDA_METHODS:
        cfg = alpha6.build_training_config("run", method, split_dir, root, 42)
        observed.append(
            (
                method,
                cfg.get("pilot_hpo_run", ""),
                cfg["candidate_source_label"],
                cfg["nicme_logit_cost_scale"],
                cfg["cs_lambda"],
            )
        )

    assert observed == [
        ("nicme_run53_a0p09_l0p2_pmi20", 53, "PMI-10 run 53", 0.09, 0.20),
        ("nicme_run20_a0p06_l0p03_pmi20", 20, "PMI-10 run 20", 0.06, 0.03),
        ("nicme_run50_a0p09_l0p07_pmi20", 50, "PMI-10 run 50", 0.09, 0.07),
        ("nicme_run95_a0p5_l0p07_pmi20", 95, "PMI-10 run 95", 0.50, 0.07),
        ("nicme_run69_a0p2_l0p09_pmi20", 69, "PMI-10 run 69", 0.20, 0.09),
        ("nicme_extra_a0p5_l0p1_pmi20", "", "added PMI-20 candidate", 0.50, 0.10),
    ]


def test_pmi20_profile_split_counts_and_cost_hash(tmp_path):
    run_args = args(tmp_path)
    split_dir = Path(run_args.split_dir)
    write_split_files(split_dir)

    cfg = alpha6.build_training_config("run", alpha6.RUN50_METHOD, split_dir, tmp_path / "results", 42)
    alpha6.run_preflight(run_args)

    assert alpha6.pmi20.split_counts(split_dir) == {"train": 1940, "validation": 320, "test": 640}
    assert cfg["dataset_profile"] == "pmi_pills"
    assert cfg["num_labels"] == 20
    assert cfg["cost_matrix_sha256"] == alpha6.pmi20.REQUIRED_COST_HASH
    assert baseline.cost_matrix_hash(cfg) == alpha6.pmi20.REQUIRED_COST_HASH


def test_shared_training_defaults_match_pmi20_camera_ready_protocol(tmp_path):
    split_dir = tmp_path / "splits" / "balanced"
    root = tmp_path / "results"

    for method in alpha6.ALPHA_LAMBDA_METHODS:
        cfg = alpha6.build_training_config("run", method, split_dir, root, 42)
        assert cfg["loss_function"] == "nicme_hybrid"
        assert cfg["model"] == "timm/convnext_base.fb_in22k_ft_in1k"
        assert cfg["image_size"] == 224
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
        assert cfg["decision_modes"] == "argmax,cost_min"
        assert cfg["calibration_enabled"] is False


def test_aggregation_uses_sample_standard_deviation():
    method = alpha6.RUN50_METHOD
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
            "alpha": 0.09,
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
            "alpha": 0.09,
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
            "alpha": 0.09,
            "cs_lambda": 0.07,
            "loss_function": "nicme_hybrid",
        },
    ]

    aggregate = alpha6.aggregate_metric_rows(rows)[0]

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
        "cost_matrix_sha256": alpha6.pmi20.REQUIRED_COST_HASH,
    }
    cost_min = dict(report)
    cost_min["prediction_mode"] = "cost_min"
    return {"decision_reports": {"argmax": report, "cost_min": cost_min}}


def write_fake_completed_ledgers(root: Path, run_args: argparse.Namespace):
    scores = {
        "nicme_run53_a0p09_l0p2_pmi20": (0.6, 0.06, 0.6, 0.6, 5),
        "nicme_run20_a0p06_l0p03_pmi20": (0.7, 0.05, 0.7, 0.7, 4),
        "nicme_run50_a0p09_l0p07_pmi20": (0.8, 0.04, 0.8, 0.8, 3),
        "nicme_run95_a0p5_l0p07_pmi20": (0.5, 0.07, 0.5, 0.5, 6),
        "nicme_run69_a0p2_l0p09_pmi20": (0.9, 0.01, 0.9, 0.9, 1),
        "nicme_extra_a0p5_l0p1_pmi20": (0.85, 0.02, 0.85, 0.85, 2),
    }
    for seed in run_args.seeds:
        rows = []
        for idx, method in enumerate(alpha6.ALPHA_LAMBDA_METHODS, start=1):
            metric_path = root / "metrics" / f"{seed}_{method}.json"
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_path.write_text(json.dumps(metric_payload(*scores[method])))
            cfg = alpha6.build_training_config("run", method, Path(run_args.split_dir), alpha6.seed_root(root, seed), seed)
            config_path = root / "configs" / f"{seed}_{method}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(cfg))
            row = baseline.row_from_config(cfg, "run", idx, status="completed")
            row["config_path"] = str(config_path)
            row["metric_path"] = str(metric_path)
            rows.append(row)
        baseline.write_ledger(alpha6.seed_root(root, seed) / "run" / "run_ledger.csv", rows)


def test_analysis_refuses_incomplete_rows(tmp_path):
    root = tmp_path / "pmi20_alpha_lambda6"
    seed_root = alpha6.seed_root(root, 42)
    seed_root.mkdir(parents=True)
    baseline.write_ledger(seed_root / "run" / "run_ledger.csv", [])
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    run_args.seeds = (42,)

    with pytest.raises(RuntimeError, match="Missing run ledger|incomplete"):
        alpha6.analyze(run_args)


def test_analysis_preserves_all_six_settings_and_reports_non_run50_winner(tmp_path):
    root = tmp_path / "pmi20_alpha_lambda6"
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    write_fake_completed_ledgers(root, run_args)

    alpha6.analyze(run_args)

    table = (root / "analysis" / "pmi20_nicme_alpha_lambda6_table.md").read_text()
    winners = (root / "analysis" / "cost_sensitive_winners.md").read_text()
    aggregate_rows = list(csv.DictReader(open(root / "analysis" / "aggregate_metrics.csv")))

    assert len(aggregate_rows) == 6
    for source in ("PMI-10 run 53", "PMI-10 run 20", "PMI-10 run 50", "PMI-10 run 95", "PMI-10 run 69", "added PMI-20 candidate"):
        assert source in table
    assert "PMI-10 run 69" in winners
    assert aggregate_rows[0]["method"] == "nicme_run69_a0p2_l0p09_pmi20"
