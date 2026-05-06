import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts import run_pmi20_camera_ready_lr5e5 as pmi20
from scripts import run_pmi10_sota_baselines as baseline


def args(tmp_path):
    return argparse.Namespace(
        phase="dry-run",
        split_dir=str(tmp_path / "splits" / "balanced"),
        output_root=str(tmp_path / "pmi20"),
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


def test_dry_run_builds_three_seed_unique_training_configs(tmp_path):
    rows = pmi20.write_dry_run(args(tmp_path))

    assert len(rows) == 3 * 6
    assert {int(row["seed"]) for row in rows} == {42, 43, 44}
    assert len({row["config_hash"] for row in rows}) == len(rows)
    assert len({row["checkpoint_path"] for row in rows}) == len(rows)


def test_pmi20_profile_split_counts_and_cost_hash(tmp_path):
    run_args = args(tmp_path)
    split_dir = Path(run_args.split_dir)
    write_split_files(split_dir)

    counts = pmi20.split_counts(split_dir)
    cfg = pmi20.build_training_config("run", pmi20.NICME_METHOD, split_dir, tmp_path / "results", 42)
    pmi20.run_preflight(run_args)

    assert counts == {"train": 1940, "validation": 320, "test": 640}
    assert cfg["dataset_profile"] == "pmi_pills"
    assert cfg["num_labels"] == 20
    assert cfg["cost_matrix_sha256"] == pmi20.REQUIRED_COST_HASH
    assert baseline.cost_matrix_hash(cfg) == pmi20.REQUIRED_COST_HASH


def test_primary_nicme_config_is_paper_alpha_lambda(tmp_path):
    cfg = pmi20.build_training_config(
        "run",
        pmi20.NICME_METHOD,
        tmp_path / "splits" / "balanced",
        tmp_path / "results",
        42,
    )

    assert cfg["loss_function"] == "nicme_hybrid"
    assert cfg["learning_rate"] == 5e-5
    assert cfg["parent_learning_rate"] == 5e-5
    assert cfg["nicme_logit_cost_scale"] == 0.5
    assert cfg["cs_lambda"] == 0.1
    assert cfg["cs_warmup_epochs"] == 2
    assert cfg["added_fixed_candidate"] is True


def test_selected_method_set_matches_plan():
    assert pmi20.TRAIN_METHODS == (
        pmi20.CE_METHOD,
        pmi20.NICME_METHOD,
        "menon_logit_adjusted",
        "cost_weighted_ce",
        "cost_sensitive_regularized_ce",
        "csada",
    )
    assert pmi20.REPORT_METHODS == (
        pmi20.CE_METHOD,
        pmi20.CE_COST_MIN_METHOD,
        pmi20.NICME_METHOD,
        "menon_logit_adjusted",
        "cost_weighted_ce",
        "cost_sensitive_regularized_ce",
        "csada",
    )
    assert pmi20.display_name("cost_sensitive_regularized_ce") == "cost-sensitive regularized CE"


def test_shared_training_defaults_match_camera_ready_protocol(tmp_path):
    split_dir = tmp_path / "splits" / "balanced"
    root = tmp_path / "results"
    for method in pmi20.TRAIN_METHODS:
        cfg = pmi20.build_training_config("run", method, split_dir, root, 42)
        assert cfg["model"] == "timm/convnext_base.fb_in22k_ft_in1k"
        assert cfg["image_size"] == 224
        assert cfg["batch_size"] == 16
        assert cfg["gradient_accumulation_steps"] == 4
        assert cfg["lr_scheduler_type"] == "cosine"
        if method in pmi20.ADAPTATION_METHODS:
            assert cfg["learning_rate"] == 1e-5
            assert cfg["parent_learning_rate"] == 5e-5
            assert cfg["num_train_epochs"] == 10
            assert cfg["early_stopping_patience"] == 3
        else:
            assert cfg["learning_rate"] == 5e-5
            assert cfg["parent_learning_rate"] == 5e-5
            assert cfg["num_train_epochs"] == 32
            assert cfg["early_stopping_patience"] == 5
            assert cfg["weight_decay"] == 0.005
            assert cfg["warmup_ratio"] == 0.10


def test_aggregation_uses_sample_standard_deviation():
    rows = [
        {"seed": 42, "method": pmi20.NICME_METHOD, "decision_mode": "argmax", "target_recall_min": 0.9, "target_recall_macro": 0.9, "normalized_atc": 0.03, "atc": 0.3, "balanced_accuracy": 0.8, "macro_f1": 0.8, "critical_pair_error_count": 3},
        {"seed": 43, "method": pmi20.NICME_METHOD, "decision_mode": "argmax", "target_recall_min": 0.8, "target_recall_macro": 0.8, "normalized_atc": 0.02, "atc": 0.2, "balanced_accuracy": 0.7, "macro_f1": 0.7, "critical_pair_error_count": 2},
        {"seed": 44, "method": pmi20.NICME_METHOD, "decision_mode": "argmax", "target_recall_min": 1.0, "target_recall_macro": 1.0, "normalized_atc": 0.01, "atc": 0.1, "balanced_accuracy": 0.9, "macro_f1": 0.9, "critical_pair_error_count": 1},
    ]

    aggregate = pmi20.aggregate_metric_rows(rows)[0]

    assert aggregate["target_recall_min_mean"] == pytest.approx(0.9)
    assert aggregate["target_recall_min_std"] == pytest.approx(0.1)
    assert aggregate["critical_pair_error_count_total"] == 6


def metric_payload(value):
    argmax = {
        "target_recall_min": value,
        "target_recall_macro": value,
        "normalized_atc": 1.0 - value,
        "atc": 10.0 * (1.0 - value),
        "balanced_accuracy": value,
        "accuracy": value,
        "macro_f1": value,
        "critical_pair_errors": {"x->y": {"errors": int(round(10 * (1.0 - value)))}},
        "cost_matrix_sha256": pmi20.REQUIRED_COST_HASH,
    }
    cost_min = dict(argmax)
    cost_min["target_recall_min"] = max(0.0, value - 0.1)
    cost_min["prediction_mode"] = "cost_min"
    return {"decision_reports": {"argmax": argmax, "cost_min": cost_min}}


def write_fake_completed_ledgers(root: Path, run_args: argparse.Namespace):
    for seed in run_args.seeds:
        rows = []
        for idx, method in enumerate(pmi20.TRAIN_METHODS, start=1):
            metric_path = root / "metrics" / f"{seed}_{method}.json"
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            value = 0.9 if method == pmi20.NICME_METHOD else 0.5
            metric_path.write_text(json.dumps(metric_payload(value)))
            cfg = pmi20.build_training_config("run", method, Path(run_args.split_dir), pmi20.seed_root(root, seed), seed)
            config_path = root / "configs" / f"{seed}_{method}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(cfg))
            row = baseline.row_from_config(cfg, "run", idx, status="completed")
            row["config_path"] = str(config_path)
            row["metric_path"] = str(metric_path)
            rows.append(row)
        baseline.write_ledger(pmi20.seed_root(root, seed) / "run" / "run_ledger.csv", rows)


def test_analysis_refuses_incomplete_rows(tmp_path):
    root = tmp_path / "pmi20"
    seed_root = pmi20.seed_root(root, 42)
    seed_root.mkdir(parents=True)
    baseline.write_ledger(seed_root / "run" / "run_ledger.csv", [])
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    run_args.seeds = (42,)

    with pytest.raises(RuntimeError, match="Missing run ledger|incomplete"):
        pmi20.analyze(run_args)


def test_analysis_includes_derived_cost_min_and_all_report_rows(tmp_path):
    root = tmp_path / "pmi20"
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    write_fake_completed_ledgers(root, run_args)

    pmi20.analyze(run_args)

    aggregate_rows = list(csv.DictReader(open(root / "analysis" / "aggregate_metrics.csv")))
    methods = {row["method"] for row in aggregate_rows}
    table = (root / "analysis" / "pmi20_sota_table.md").read_text()

    assert len(aggregate_rows) == 7
    assert pmi20.CE_COST_MIN_METHOD in methods
    assert "CE + cost-min inference" in table
    assert "NICME" in table
