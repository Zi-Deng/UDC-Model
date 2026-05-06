import argparse
import json
from pathlib import Path

import pytest

from scripts import run_pmi20_component_ablation_lr5e5 as ablation
from scripts import run_pmi10_sota_baselines as baseline


def args(tmp_path):
    return argparse.Namespace(
        phase="dry-run",
        split_dir=str(tmp_path / "splits" / "balanced"),
        output_root=str(tmp_path / "ablation"),
        seeds=(42, 43, 44),
        per_run_timeout_minutes=1,
        allow_incomplete_analysis=False,
        skip_nvidia_smi=True,
        ce_per_seed=str(tmp_path / "ce.csv"),
        full_nicme_per_seed=str(tmp_path / "full.csv"),
    )


def write_split_files(split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, count in {"train": 1940, "validation": 320, "test": 640}.items():
        with (split_dir / f"{split}.csv").open("w") as f:
            f.write("image_path,label,label_name\n")
            for idx in range(count):
                f.write(f"/tmp/{split}_{idx}.jpg,{idx % 20},class_{idx % 20}\n")


def test_dry_run_creates_two_methods_over_three_seeds(tmp_path):
    run_args = args(tmp_path)
    rows = ablation.write_dry_run(run_args)

    assert len(rows) == 2 * 3
    assert {row["method"] for row in rows} == set(ablation.TRAIN_METHODS)
    assert {int(row["seed"]) for row in rows} == {42, 43, 44}
    assert len({row["config_hash"] for row in rows}) == len(rows)


def test_regularizer_and_margin_configs_match_protocol(tmp_path):
    split_dir = tmp_path / "splits" / "balanced"
    root = tmp_path / "results"
    regularizer = ablation.build_training_config("run", ablation.REGULARIZER_METHOD, split_dir, root, 42)
    margin = ablation.build_training_config("run", ablation.MARGIN_METHOD, split_dir, root, 42)

    assert regularizer["loss_function"] == "cs_regularized_ce"
    assert regularizer["nicme_logit_cost_scale"] == 0.0
    assert regularizer["cs_lambda"] == 0.1
    assert regularizer["cs_warmup_epochs"] == 2
    assert margin["loss_function"] == "nicme_logit_adjustment"
    assert margin["nicme_logit_cost_scale"] == 0.5
    assert margin["cs_lambda"] == 0.0
    assert margin["cs_warmup_epochs"] == 0

    for cfg in (regularizer, margin):
        assert cfg["learning_rate"] == 5e-5
        assert cfg["parent_learning_rate"] == 5e-5
        assert cfg["num_train_epochs"] == 32
        assert cfg["early_stopping_patience"] == 5
        assert cfg["decision_modes"] == "argmax,cost_min"
        assert cfg["cost_matrix_sha256"] == ablation.pmi20.REQUIRED_COST_HASH


def metric_payload(value):
    return {
        "decision_reports": {
            "argmax": {
                "target_recall_min": value,
                "target_recall_macro": value,
                "normalized_atc": 1.0 - value,
                "atc": 10.0 * (1.0 - value),
                "balanced_accuracy": value,
                "accuracy": value,
                "macro_f1": value,
                "critical_pair_errors": {"x->y": {"errors": int(round(10 * (1.0 - value)))}},
            }
        }
    }


def write_reuse_csv(path: Path, method: str, renamed_value: float):
    fields = [
        "seed",
        "method",
        "decision_mode",
        "target_recall_min",
        "target_recall_macro",
        "normalized_atc",
        "atc",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "critical_pair_error_count",
        "metric_path",
        "config_path",
        "checkpoint_path",
        "loss_function",
    ]
    rows = []
    for seed in (42, 43, 44):
        rows.append(
            {
                "seed": seed,
                "method": method,
                "decision_mode": "argmax",
                "target_recall_min": renamed_value,
                "target_recall_macro": renamed_value,
                "normalized_atc": 1.0 - renamed_value,
                "atc": 10.0 * (1.0 - renamed_value),
                "balanced_accuracy": renamed_value,
                "accuracy": renamed_value,
                "macro_f1": renamed_value,
                "critical_pair_error_count": 1,
                "metric_path": f"/tmp/{method}_{seed}.json",
                "config_path": "",
                "checkpoint_path": "",
                "loss_function": "cross_entropy",
            }
        )
    ablation.write_csv(path, rows, fields)


def test_analysis_reuses_ce_and_full_nicme_rows(tmp_path):
    run_args = args(tmp_path)
    root = Path(run_args.output_root)
    write_reuse_csv(Path(run_args.ce_per_seed), ablation.pmi20.CE_METHOD, 0.7)
    write_reuse_csv(Path(run_args.full_nicme_per_seed), "nicme_extra_a0p5_l0p1_pmi20", 0.9)

    for seed in run_args.seeds:
        rows = []
        for idx, method in enumerate(ablation.TRAIN_METHODS, start=1):
            metric_path = root / "metrics" / f"{seed}_{method}.json"
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_path.write_text(json.dumps(metric_payload(0.8)))
            cfg = ablation.build_training_config("run", method, Path(run_args.split_dir), root, seed)
            config_path = root / "configs" / f"{seed}_{method}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(cfg))
            row = baseline.row_from_config(cfg, "run", idx, status="completed")
            row["metric_path"] = str(metric_path)
            row["config_path"] = str(config_path)
            rows.append(row)
        baseline.write_ledger(ablation.seed_root(root, seed) / "run" / "run_ledger.csv", rows)

    ablation.analyze(run_args)
    aggregate = (root / "analysis" / "aggregate_metrics.csv").read_text()
    assert "ce" in aggregate
    assert "nicme_regularizer_only_l0p1" in aggregate
    assert "nicme_margin_only_a0p5" in aggregate
    assert "nicme_full_a0p5_l0p1" in aggregate


def test_preflight_checks_split_and_configs(tmp_path):
    run_args = args(tmp_path)
    write_split_files(Path(run_args.split_dir))

    ablation.run_preflight(run_args)
