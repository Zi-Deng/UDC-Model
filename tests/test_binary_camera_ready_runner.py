import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts import run_binary_camera_ready_cost8_7_lr5e5 as binary
from scripts import run_pmi10_sota_baselines as baseline


def write_split_files(split_dir: Path, class_names: tuple[str, str], counts: dict[str, int]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, count in counts.items():
        path = split_dir / f"{split}.csv"
        with path.open("w") as f:
            f.write("image_path,label,label_name\n")
            per_class = count // 2
            for label, name in enumerate(class_names):
                for idx in range(per_class):
                    f.write(f"/tmp/{split}_{name}_{idx}.jpg,{label},{name}\n")


def patch_dataset_specs(tmp_path, monkeypatch):
    spider_dir = tmp_path / "spider" / "balanced"
    breakhis_dir = tmp_path / "breakhis" / "balanced"
    write_split_files(spider_dir, ("black_widow", "false_widow"), {"train": 2098, "validation": 300, "test": 300})
    write_split_files(breakhis_dir, ("benign", "malignant"), {"train": 3448, "validation": 410, "test": 564})
    specs = {
        "spider": binary.BinaryDatasetSpec(
            key="spider",
            profile_name="spider",
            split_dir=spider_dir,
            expected_counts={"train": 2098, "validation": 300, "test": 300},
            required_cost_hash=binary.SPIDER_COST_HASH,
            display_name="Spider",
        ),
        "breakhis": binary.BinaryDatasetSpec(
            key="breakhis",
            profile_name="breakhis",
            split_dir=breakhis_dir,
            expected_counts={"train": 3448, "validation": 410, "test": 564},
            required_cost_hash=binary.BREAKHIS_COST_HASH,
            display_name="BreaKHis",
        ),
    }
    monkeypatch.setattr(binary, "DATASETS", specs)
    return specs


def args(tmp_path):
    return argparse.Namespace(
        phase="dry-run",
        output_root=str(tmp_path / "binary_results"),
        seeds=(42, 43, 44),
        datasets=("spider", "breakhis"),
        per_run_timeout_minutes=1,
        allow_incomplete_analysis=False,
        skip_nvidia_smi=True,
    )


def test_dry_run_builds_binary_dataset_seed_configs(tmp_path, monkeypatch):
    patch_dataset_specs(tmp_path, monkeypatch)
    rows = binary.write_dry_run(args(tmp_path))

    assert len(rows) == 2 * 3 * 6
    assert {row["dataset_key"] for row in rows} == {"spider", "breakhis"}
    assert {int(row["seed"]) for row in rows} == {42, 43, 44}
    assert len({row["config_hash"] for row in rows}) == len(rows)
    assert len({row["checkpoint_path"] for row in rows}) == len(rows)


def test_preflight_split_counts_and_cost_hashes(tmp_path, monkeypatch):
    patch_dataset_specs(tmp_path, monkeypatch)
    run_args = args(tmp_path)

    binary.run_preflight(run_args)

    assert binary.split_counts(binary.DATASETS["spider"].split_dir) == {"train": 2098, "validation": 300, "test": 300}
    assert binary.class_counts(binary.DATASETS["breakhis"].split_dir, "test", ("benign", "malignant")) == {
        "benign": 282,
        "malignant": 282,
    }
    spider_cfg = binary.build_training_config("run", "spider", binary.NICME_METHOD, tmp_path / "results", 42)
    breakhis_cfg = binary.build_training_config("run", "breakhis", binary.NICME_METHOD, tmp_path / "results", 42)
    assert spider_cfg["cost_matrix"] == [[0.0, 8.0], [1.0, 0.0]]
    assert breakhis_cfg["cost_matrix"] == [[0.0, 1.0], [7.0, 0.0]]
    assert baseline.cost_matrix_hash(spider_cfg) == binary.SPIDER_COST_HASH
    assert baseline.cost_matrix_hash(breakhis_cfg) == binary.BREAKHIS_COST_HASH


def test_primary_nicme_and_adaptation_configs_match_protocol(tmp_path, monkeypatch):
    patch_dataset_specs(tmp_path, monkeypatch)
    nicme = binary.build_training_config("run", "spider", binary.NICME_METHOD, tmp_path / "results", 42)
    csada = binary.build_training_config(
        "run",
        "breakhis",
        "csada",
        tmp_path / "results",
        42,
        anchor_path="checkpoints/ce_anchor",
    )
    cost_sensitive_regularized_ce = binary.build_training_config(
        "run",
        "breakhis",
        "cost_sensitive_regularized_ce",
        tmp_path / "results",
        42,
        anchor_path="checkpoints/ce_anchor",
    )

    assert nicme["loss_function"] == "nicme_hybrid"
    assert nicme["learning_rate"] == 5e-5
    assert nicme["parent_learning_rate"] == 5e-5
    assert nicme["nicme_logit_cost_scale"] == 0.5
    assert nicme["cs_lambda"] == 0.1
    assert nicme["cs_warmup_epochs"] == 2
    assert nicme["num_train_epochs"] == 32
    assert nicme["early_stopping_patience"] == 5
    assert nicme["batch_size"] == 16
    assert nicme["gradient_accumulation_steps"] == 4
    assert nicme["lr_scheduler_type"] == "cosine"
    assert nicme["weight_decay"] == 0.005
    assert nicme["warmup_ratio"] == 0.10
    assert nicme["calibration_enabled"] is False
    assert nicme["decision_modes"] == "argmax,cost_min"
    assert nicme["checkpoint_root"] == str(binary.DEFAULT_CHECKPOINT_ROOT)
    assert nicme["save_only_model"] is True
    assert baseline.checkpoint_parent(nicme) == binary.DEFAULT_CHECKPOINT_ROOT / nicme["output_dir"]

    for cfg in (csada, cost_sensitive_regularized_ce):
        assert cfg["learning_rate"] == 1e-5
        assert cfg["parent_learning_rate"] == 5e-5
        assert cfg["num_train_epochs"] == 10
        assert cfg["early_stopping_patience"] == 3
        assert cfg["initial_checkpoint_path"] == "checkpoints/ce_anchor"
    assert csada["csada_attack_steps"] == 5
    assert csada["csada_epsilon"] == 2.0 / 255.0
    assert csada["csada_step_size"] == 1.0 / 255.0
    assert csada["csada_lambda"] == 2.0
    assert csada["csada_cost_temperature"] == 3.0


def test_selected_method_set_matches_plan():
    assert binary.TRAIN_METHODS == (
        binary.CE_METHOD,
        binary.NICME_METHOD,
        "menon_logit_adjusted",
        "cost_weighted_ce",
        "cost_sensitive_regularized_ce",
        "csada",
    )
    assert binary.REPORT_METHODS == (
        binary.CE_METHOD,
        binary.CE_COST_MIN_METHOD,
        binary.NICME_METHOD,
        "menon_logit_adjusted",
        "cost_weighted_ce",
        "cost_sensitive_regularized_ce",
        "csada",
    )
    assert binary.display_name("cost_sensitive_regularized_ce") == "cost-sensitive regularized CE"


def metric_payload(value, cost_hash):
    argmax = {
        "target_recall_min": value,
        "target_recall_macro": value,
        "normalized_atc": 1.0 - value,
        "atc": 10.0 * (1.0 - value),
        "accuracy": value,
        "balanced_accuracy": value,
        "macro_f1": value,
        "critical_pair_errors": {"critical": {"errors": int(round(10 * (1.0 - value)))}},
        "cost_matrix_sha256": cost_hash,
    }
    cost_min = dict(argmax)
    cost_min["target_recall_min"] = max(0.0, value - 0.1)
    cost_min["prediction_mode"] = "cost_min"
    return {"decision_reports": {"argmax": argmax, "cost_min": cost_min}}


def write_fake_completed_ledgers(root: Path, run_args: argparse.Namespace):
    for dataset_key in run_args.datasets:
        cost_hash = binary.DATASETS[dataset_key].required_cost_hash
        for seed in run_args.seeds:
            rows = []
            for idx, method in enumerate(binary.TRAIN_METHODS, start=1):
                metric_path = root / "metrics" / f"{dataset_key}_{seed}_{method}.json"
                metric_path.parent.mkdir(parents=True, exist_ok=True)
                value = 0.9 if method == binary.NICME_METHOD else 0.5
                metric_path.write_text(json.dumps(metric_payload(value, cost_hash)))
                cfg = binary.build_training_config("run", dataset_key, method, root, seed)
                config_path = root / "configs" / f"{dataset_key}_{seed}_{method}.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(json.dumps(cfg))
                row = baseline.row_from_config(cfg, "run", idx, status="completed")
                row["config_path"] = str(config_path)
                row["metric_path"] = str(metric_path)
                rows.append(row)
            baseline.write_ledger(binary.seed_root(root, dataset_key, seed) / "run" / "run_ledger.csv", rows)


def test_analysis_refuses_incomplete_rows(tmp_path, monkeypatch):
    patch_dataset_specs(tmp_path, monkeypatch)
    root = tmp_path / "binary_results"
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    seed_root = binary.seed_root(root, "spider", 42)
    seed_root.mkdir(parents=True)
    baseline.write_ledger(seed_root / "run" / "run_ledger.csv", [])

    with pytest.raises(RuntimeError, match="Missing run ledger|incomplete"):
        binary.analyze(run_args)


def test_analysis_includes_derived_cost_min_and_all_report_rows(tmp_path, monkeypatch):
    patch_dataset_specs(tmp_path, monkeypatch)
    root = tmp_path / "binary_results"
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    write_fake_completed_ledgers(root, run_args)

    binary.analyze(run_args)

    per_seed_rows = list(csv.DictReader(open(root / "analysis" / "per_seed_metrics.csv")))
    aggregate_rows = list(csv.DictReader(open(root / "analysis" / "aggregate_metrics.csv")))
    methods = {row["method"] for row in aggregate_rows}
    table = (root / "analysis" / "binary_sota_table.md").read_text()
    spider_table = (root / "analysis" / "spider_sota_table.md").read_text()
    breakhis_table = (root / "analysis" / "breakhis_sota_table.md").read_text()

    assert len(per_seed_rows) == 2 * 3 * 7
    assert len(aggregate_rows) == 2 * 7
    assert binary.CE_COST_MIN_METHOD in methods
    assert "CE + cost-min inference" in table
    assert "NICME" in table
    assert "Spider" in spider_table
    assert "BreaKHis" in breakhis_table
    assert "nicme_a0p5_l0p1_binary" in {row["method"] for row in aggregate_rows}
    assert any(row["critical_pair_error_count_total"] for row in aggregate_rows)


def test_aggregation_uses_sample_standard_deviation():
    rows = [
        {"dataset_key": "spider", "method": binary.NICME_METHOD, "decision_mode": "argmax", "target_recall_min": 0.9, "target_recall_macro": 0.9, "normalized_atc": 0.03, "atc": 0.3, "accuracy": 0.8, "balanced_accuracy": 0.8, "macro_f1": 0.8, "critical_pair_error_count": 3, "seed": 42},
        {"dataset_key": "spider", "method": binary.NICME_METHOD, "decision_mode": "argmax", "target_recall_min": 0.8, "target_recall_macro": 0.8, "normalized_atc": 0.02, "atc": 0.2, "accuracy": 0.7, "balanced_accuracy": 0.7, "macro_f1": 0.7, "critical_pair_error_count": 2, "seed": 43},
        {"dataset_key": "spider", "method": binary.NICME_METHOD, "decision_mode": "argmax", "target_recall_min": 1.0, "target_recall_macro": 1.0, "normalized_atc": 0.01, "atc": 0.1, "accuracy": 0.9, "balanced_accuracy": 0.9, "macro_f1": 0.9, "critical_pair_error_count": 1, "seed": 44},
    ]

    aggregate = binary.aggregate_metric_rows(rows)[0]

    assert aggregate["target_recall_min_mean"] == pytest.approx(0.9)
    assert aggregate["target_recall_min_std"] == pytest.approx(0.1)
    assert aggregate["critical_pair_error_count_total"] == 6
