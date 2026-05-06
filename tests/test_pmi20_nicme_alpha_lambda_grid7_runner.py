import argparse
import csv
import json
from pathlib import Path

import pytest

from scripts import run_pmi10_sota_baselines as baseline
from scripts import run_pmi20_nicme_alpha_lambda_grid7_lr5e5 as grid7


def args(tmp_path):
    return argparse.Namespace(
        phase="dry-run",
        split_dir=str(tmp_path / "splits" / "balanced"),
        output_root=str(tmp_path / "pmi20_grid7"),
        seeds=(42,),
        per_run_timeout_minutes=1,
        allow_incomplete_analysis=False,
        skip_nvidia_smi=True,
    )


def write_split_files(split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, count in {"train": 1940, "validation": 320, "test": 640}.items():
        with (split_dir / f"{split}.csv").open("w") as f:
            f.write("image_path,label,label_name\n")
            for idx in range(count):
                f.write(f"/tmp/{split}_{idx}.jpg,{idx % 20},class_{idx % 20}\n")


def report(target_recall, normalized_atc, balanced_accuracy, macro_f1, critical_errors):
    return {
        "target_recall_min": target_recall,
        "target_recall_macro": target_recall,
        "normalized_atc": normalized_atc,
        "atc": normalized_atc * 10.0,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "critical_pair_errors": {"x->y": {"errors": critical_errors}},
        "cost_matrix_sha256": grid7.pmi20.REQUIRED_COST_HASH,
    }


def metric_payload(validation_report, test_report):
    return {
        "validation_report": {
            "argmax": validation_report,
            "metrics": {
                "eval_target_recall": validation_report["target_recall_min"],
                "eval_target_recall_macro": validation_report["target_recall_macro"],
                "eval_normalized_atc": validation_report["normalized_atc"],
                "eval_atc": validation_report["atc"],
                "eval_balanced_accuracy": validation_report["balanced_accuracy"],
                "eval_accuracy": validation_report["accuracy"],
                "eval_macro_f1": validation_report["macro_f1"],
            },
        },
        "decision_reports": {"argmax": test_report},
    }


def test_dry_run_creates_single_seed_7x7_unique_configs(tmp_path):
    rows = grid7.write_dry_run(args(tmp_path))

    assert len(rows) == 49
    assert {int(row["seed"]) for row in rows} == {42}
    assert len({row["config_hash"] for row in rows}) == 49
    assert len({row["checkpoint_path"] for row in rows}) == 49


def test_grid_values_are_predeclared_and_include_current_main_point(tmp_path):
    split_dir = tmp_path / "splits" / "balanced"
    root = tmp_path / "results"
    observed = {
        (
            grid7.build_training_config("run", method, split_dir, root, 42)["nicme_logit_cost_scale"],
            grid7.build_training_config("run", method, split_dir, root, 42)["cs_lambda"],
        )
        for method in grid7.ALPHA_LAMBDA_METHODS
    }

    assert observed == {(alpha, lam) for alpha in grid7.ALPHA_VALUES for lam in grid7.LAMBDA_VALUES}
    assert (0.5, 0.1) in observed
    assert len(grid7.SMOKE_METHODS) == 9


def test_configs_match_pmi20_protocol(tmp_path):
    split_dir = tmp_path / "splits" / "balanced"
    root = tmp_path / "results"

    for method in grid7.ALPHA_LAMBDA_METHODS:
        cfg = grid7.build_training_config("run", method, split_dir, root, 42)
        assert cfg["loss_function"] == "nicme_hybrid"
        assert cfg["model"] == "timm/convnext_base.fb_in22k_ft_in1k"
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
        assert cfg["save_total_limit"] == 1
        assert cfg["decision_modes"] == "argmax,cost_min"
        assert cfg["cost_matrix_sha256"] == grid7.pmi20.REQUIRED_COST_HASH


def test_preflight_checks_split_counts_and_seed(tmp_path):
    run_args = args(tmp_path)
    write_split_files(Path(run_args.split_dir))

    grid7.run_preflight(run_args)

    bad_args = args(tmp_path)
    bad_args.seeds = (43,)
    with pytest.raises(ValueError, match="seed 42"):
        grid7.run_preflight(bad_args)


def write_fake_completed_ledgers(root: Path, run_args: argparse.Namespace):
    validation_best = grid7.METHOD_BY_POINT[(0.3, 0.07)]
    test_best = grid7.METHOD_BY_POINT[(0.7, 0.3)]
    rows = []
    for idx, method in enumerate(grid7.ALPHA_LAMBDA_METHODS, start=1):
        val = report(0.7, 0.04, 0.7, 0.7, 4)
        test = report(0.7, 0.04, 0.7, 0.7, 4)
        if method == validation_best:
            val = report(0.99, 0.001, 0.99, 0.99, 0)
            test = report(0.5, 0.10, 0.5, 0.5, 8)
        if method == test_best:
            val = report(0.8, 0.02, 0.8, 0.8, 3)
            test = report(1.0, 0.001, 1.0, 1.0, 0)
        metric_path = root / "metrics" / f"{method}.json"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(json.dumps(metric_payload(val, test)))
        cfg = grid7.build_training_config("run", method, Path(run_args.split_dir), grid7.seed_root(root, 42), 42)
        config_path = root / "configs" / f"{method}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(cfg))
        row = baseline.row_from_config(cfg, "run", idx, status="completed")
        row["config_path"] = str(config_path)
        row["metric_path"] = str(metric_path)
        rows.append(row)
    baseline.write_ledger(grid7.seed_root(root, 42) / "run" / "run_ledger.csv", rows)


def test_analysis_ranks_by_validation_not_test(tmp_path):
    root = tmp_path / "pmi20_grid7"
    run_args = args(tmp_path)
    run_args.output_root = str(root)
    write_fake_completed_ledgers(root, run_args)

    grid7.analyze(run_args)

    validation_rows = list(csv.DictReader(open(root / "analysis" / "validation_grid.csv")))
    test_rows = list(csv.DictReader(open(root / "analysis" / "test_grid.csv")))
    selected = validation_rows[0]

    assert len(validation_rows) == 49
    assert selected["method"] == grid7.METHOD_BY_POINT[(0.3, 0.07)]
    assert selected["is_validation_selected"] == "true"
    assert test_rows[0]["method"] == grid7.METHOD_BY_POINT[(0.7, 0.3)]
    assert "single-seed" in (root / "analysis" / "claim_audit.md").read_text()


def test_analysis_refuses_incomplete_rows(tmp_path):
    root = tmp_path / "pmi20_grid7"
    seed_root = grid7.seed_root(root, 42)
    seed_root.mkdir(parents=True)
    baseline.write_ledger(seed_root / "run" / "run_ledger.csv", [])
    run_args = args(tmp_path)
    run_args.output_root = str(root)

    with pytest.raises(RuntimeError, match="Missing run ledger|incomplete"):
        grid7.analyze(run_args)
