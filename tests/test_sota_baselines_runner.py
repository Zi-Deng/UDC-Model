import argparse
from pathlib import Path

import pytest

from nicme.dataset_profiles import PMI_PILLS_10_NO_CAL_PROFILE
from scripts import run_pmi10_sota_baselines as runner

torch = pytest.importorskip("torch")


def args(tmp_path):
    return argparse.Namespace(
        phase="dry-run",
        split_dir=str(tmp_path / "splits" / "balanced"),
        output_root=str(tmp_path / "results"),
        comparison_profile="matched",
        learning_rate_profile="lr5e5",
        combined_output_root=str(tmp_path / "combined"),
        lr1e5_output_root=str(tmp_path / "results_lr1e5"),
        lr5e5_output_root=str(tmp_path / "results_lr5e5"),
        lr1e4_output_root=str(tmp_path / "results_lr1e4"),
        seed=42,
        per_run_timeout_minutes=1,
    )


def test_sota_base_configs_are_matched_balanced_no_calibration(tmp_path):
    cfg = runner.base_config(
        "run",
        "cost_weighted_ce",
        tmp_path / "splits" / "balanced",
        tmp_path / "results",
        42,
    )

    assert cfg["model"] == f"timm/{runner.MODEL_ID}"
    assert cfg["variant"] == "balanced"
    assert cfg["dataset_profile"] == "pmi_pills_10_no_cal"
    assert cfg["seed"] == 42
    assert cfg["calibration_enabled"] is False
    assert cfg["decision_modes"] == "argmax"
    assert cfg["peft_enabled"] is False
    assert cfg["cost_matrix_sha256"] == PMI_PILLS_10_NO_CAL_PROFILE.cost_matrix_sha256
    assert cfg["image_size"] == 224
    assert cfg["learning_rate"] == runner.MATCHED_DEFAULTS["learning_rate"]
    assert cfg["weight_decay"] == runner.MATCHED_DEFAULTS["weight_decay"]
    assert cfg["warmup_ratio"] == runner.MATCHED_DEFAULTS["warmup_ratio"]


def test_sota_runner_builds_all_baselines_with_unique_output_dirs(tmp_path):
    configs = runner.build_plan_configs(args(tmp_path), "dry-run", anchor_path="checkpoints/ce_anchor")

    assert [cfg["sota_method"] for cfg in configs] == ["ce_anchor", *runner.ALL_BASELINES]
    assert len({cfg["output_dir"] for cfg in configs}) == len(configs)
    assert {cfg["model"] for cfg in configs} == {f"timm/{runner.MODEL_ID}"}
    assert {cfg["variant"] for cfg in configs} == {"balanced"}
    assert {cfg["calibration_enabled"] for cfg in configs} == {False}
    assert {cfg["peft_enabled"] for cfg in configs} == {False}
    assert all(cfg["initial_checkpoint_path"] == "checkpoints/ce_anchor" for cfg in configs if cfg["sota_method"] in runner.ADAPTATION_BASELINES)


def pretty_args(tmp_path, lr_profile="lr5e5"):
    return argparse.Namespace(
        phase="dry-run",
        split_dir=str(tmp_path / "splits" / "balanced"),
        output_root=str(tmp_path / f"results_{lr_profile}"),
        combined_output_root=str(tmp_path / "combined"),
        lr1e5_output_root=str(tmp_path / "results_lr1e5"),
        lr5e5_output_root=str(tmp_path / "results_lr5e5"),
        lr1e4_output_root=str(tmp_path / "results_lr1e4"),
        comparison_profile="pretty_balanced",
        learning_rate_profile=lr_profile,
        seed=42,
        per_run_timeout_minutes=1,
    )


def test_pretty_lr_profiles_differ_only_in_lr_profile_fields(tmp_path):
    cfg_1e5 = runner.base_config(
        "run",
        "cost_weighted_ce",
        tmp_path / "splits" / "balanced",
        tmp_path / "results_lr1e5",
        42,
        comparison_profile="pretty_balanced",
        learning_rate_profile="lr1e5",
    )
    cfg_5e5 = runner.base_config(
        "run",
        "cost_weighted_ce",
        tmp_path / "splits" / "balanced",
        tmp_path / "results_lr5e5",
        42,
        comparison_profile="pretty_balanced",
        learning_rate_profile="lr5e5",
    )
    cfg_1e4 = runner.base_config(
        "run",
        "cost_weighted_ce",
        tmp_path / "splits" / "balanced",
        tmp_path / "results_lr1e4",
        42,
        comparison_profile="pretty_balanced",
        learning_rate_profile="lr1e4",
    )

    assert cfg_1e5["learning_rate"] == 1e-5
    assert cfg_5e5["learning_rate"] == 5e-5
    assert cfg_1e4["learning_rate"] == 1e-4
    assert cfg_1e5["weight_decay"] == cfg_5e5["weight_decay"] == cfg_1e4["weight_decay"] == 0.005
    assert cfg_1e5["warmup_ratio"] == cfg_5e5["warmup_ratio"] == cfg_1e4["warmup_ratio"] == 0.10

    ignored = {"learning_rate", "parent_learning_rate", "learning_rate_profile", "sota_output_root", "output_dir"}
    assert {k: v for k, v in cfg_1e5.items() if k not in ignored} == {
        k: v for k, v in cfg_5e5.items() if k not in ignored
    }
    assert {k: v for k, v in cfg_5e5.items() if k not in ignored} == {
        k: v for k, v in cfg_1e4.items() if k not in ignored
    }


def test_pretty_builds_nicme_and_baselines_with_unique_output_dirs(tmp_path):
    configs = runner.build_plan_configs(pretty_args(tmp_path, "lr5e5"), "dry-run", anchor_path="checkpoints/ce_anchor")

    assert [cfg["sota_method"] for cfg in configs] == list(runner.PRETTY_METHODS)
    assert len({cfg["output_dir"] for cfg in configs}) == len(configs)
    assert all("lr5e5" in cfg["output_dir"] for cfg in configs)
    assert {cfg["model"] for cfg in configs} == {f"timm/{runner.MODEL_ID}"}
    assert {cfg["calibration_enabled"] for cfg in configs} == {False}
    assert {cfg["peft_enabled"] for cfg in configs} == {False}


def test_pretty_nicme_configs_use_locked_alpha_lambda(tmp_path):
    cfg = runner.base_config(
        "run",
        "nicme_hybrid_pretty",
        tmp_path / "splits" / "balanced",
        tmp_path / "results",
        comparison_profile="pretty_balanced",
        learning_rate_profile="lr1e4",
    )

    assert cfg["loss_function"] == "nicme_hybrid"
    assert cfg["learning_rate"] == 1e-4
    assert cfg["nicme_logit_cost_scale"] == 0.4
    assert cfg["cs_lambda"] == 0.25
    assert cfg["cs_warmup_epochs"] == 2


def test_pretty_lr1e5_uses_locked_nicme_values(tmp_path):
    cfg = runner.base_config(
        "run",
        "nicme_hybrid_pretty",
        tmp_path / "splits" / "balanced",
        tmp_path / "results",
        comparison_profile="pretty_balanced",
        learning_rate_profile="lr1e5",
    )

    assert cfg["learning_rate"] == 1e-5
    assert cfg["parent_learning_rate"] == 1e-5
    assert cfg["nicme_logit_cost_scale"] == 0.4
    assert cfg["cs_lambda"] == 0.25
    assert cfg["cs_warmup_epochs"] == 2


def test_pretty_adaptation_methods_use_native_lr_and_record_parent_lr(tmp_path):
    for method in runner.ADAPTATION_BASELINES:
        cfg = runner.base_config(
            "run",
            method,
            tmp_path / "splits" / "balanced",
            tmp_path / "results",
            comparison_profile="pretty_balanced",
            learning_rate_profile="lr1e4",
        )

        assert cfg["learning_rate"] == 1e-5
        assert cfg["parent_learning_rate"] == 1e-4


def test_pretty_lr1e5_adaptation_records_matching_parent_lr(tmp_path):
    for method in runner.ADAPTATION_BASELINES:
        cfg = runner.base_config(
            "run",
            method,
            tmp_path / "splits" / "balanced",
            tmp_path / "results",
            comparison_profile="pretty_balanced",
            learning_rate_profile="lr1e5",
        )

        assert cfg["learning_rate"] == 1e-5
        assert cfg["parent_learning_rate"] == 1e-5


def test_pretty_alpha_lambda_grid_has_expected_values_and_invariants(tmp_path):
    configs = runner.build_grid_configs(pretty_args(tmp_path, "lr5e5"), "grid-dry-run")

    assert len(configs) == 108
    assert {cfg["nicme_logit_cost_scale"] for cfg in configs} == set(runner.PRETTY_GRID_ALPHA_VALUES)
    assert {cfg["cs_lambda"] for cfg in configs} == set(runner.PRETTY_GRID_LAMBDA_VALUES)
    assert all(runner.has_one_nonzero_digit(cfg["nicme_logit_cost_scale"]) for cfg in configs)
    assert all(runner.has_one_nonzero_digit(cfg["cs_lambda"]) for cfg in configs)
    assert {cfg["learning_rate"] for cfg in configs} == {5e-5}
    assert {cfg["loss_function"] for cfg in configs} == {"nicme_hybrid"}
    assert {cfg["cs_warmup_epochs"] for cfg in configs} == {2}
    assert {cfg["calibration_enabled"] for cfg in configs} == {False}
    assert {cfg["peft_enabled"] for cfg in configs} == {False}
    assert {cfg["decision_modes"] for cfg in configs} == {"argmax"}

    ignored = {"output_dir", "nicme_logit_cost_scale", "cs_lambda", "grid_alpha", "grid_cs_lambda"}
    reference = {key: value for key, value in configs[0].items() if key not in ignored}
    for cfg in configs[1:]:
        assert {key: value for key, value in cfg.items() if key not in ignored} == reference


def test_pretty_alpha_lambda_grid_smoke_points_are_representative(tmp_path):
    configs = runner.build_grid_configs(pretty_args(tmp_path, "lr5e5"), "grid-smoke", smoke=True)

    assert len(configs) == 5
    assert [(cfg["nicme_logit_cost_scale"], cfg["cs_lambda"]) for cfg in configs] == list(
        runner.PRETTY_GRID_SMOKE_POINTS
    )
    assert {cfg["num_train_epochs"] for cfg in configs} == {1}


def test_grid_validation_metric_extraction_uses_validation_report_metrics():
    row = {}
    payload = {
        "validation_report": {
            "metrics": {
                "eval_target_recall": 0.95,
                "eval_target_recall_macro": 0.96,
                "eval_normalized_atc": 0.01,
                "eval_atc": 0.1,
                "eval_balanced_accuracy": 0.97,
                "eval_accuracy": 0.98,
                "eval_macro_f1": 0.99,
            },
            "argmax": {
                "target_recall": 0.1,
                "normalized_atc": 0.9,
                "critical_pair_errors": {"a->b": {"errors": 2}},
            },
        }
    }

    runner.update_row_with_validation_report(row, payload)

    assert row["validation_target_recall_min"] == 0.95
    assert row["validation_target_recall_macro"] == 0.96
    assert row["validation_normalized_atc"] == 0.01
    assert row["validation_balanced_accuracy"] == 0.97
    assert row["validation_critical_pair_error_count"] == 2


def test_grid_selection_is_validation_based_not_test_based():
    rows = [
        {
            "status": "completed",
            "alpha": 0.4,
            "cs_lambda": 0.2,
            "validation_target_recall_min": 0.9,
            "validation_normalized_atc": 0.001,
            "validation_balanced_accuracy": 1.0,
            "target_recall_min": 1.0,
            "normalized_atc": 0.0,
        },
        {
            "status": "completed",
            "alpha": 0.08,
            "cs_lambda": 0.03,
            "validation_target_recall_min": 1.0,
            "validation_normalized_atc": 0.02,
            "validation_balanced_accuracy": 0.99,
            "validation_target_recall_macro": 0.99,
            "target_recall_min": 0.8,
            "normalized_atc": 0.5,
        },
        {
            "status": "completed",
            "alpha": 0.06,
            "cs_lambda": 0.03,
            "validation_target_recall_min": 1.0,
            "validation_normalized_atc": 0.01,
            "validation_balanced_accuracy": 0.98,
            "validation_target_recall_macro": 0.99,
            "target_recall_min": 0.7,
            "normalized_atc": 0.7,
        },
    ]

    selected = runner.select_grid_row(rows)

    assert selected["alpha"] == 0.06
    assert selected["cs_lambda"] == 0.03


def test_grid_module_traversal_typeerror_is_retryable(tmp_path):
    stderr = tmp_path / "run.stderr.log"
    stderr.write_text(
        '  File "/env/lib/python3.12/site-packages/torch/nn/modules/module.py", '
        "line 2797, in named_children\n"
        "TypeError: cannot unpack non-iterable Dropout object\n"
    )

    assert runner._stderr_contains_retryable_failure(stderr)


def test_grid_phase_resumes_completed_rows(tmp_path, monkeypatch):
    pa = pretty_args(tmp_path, "lr5e5")
    configs = runner.build_grid_configs(pa, "grid-run")
    first = runner.row_from_config(configs[0], "grid-run", 1, status="completed")
    first["metric_path"] = "existing_metrics.json"
    runner.write_grid_csv(Path(pa.output_root) / "grid-run" / "grid_run_ledger.csv", [first])

    calls = []

    def fake_execute_config(cfg, phase, run_id, args):
        calls.append(run_id)
        row = runner.row_from_config(cfg, phase, run_id, status="completed")
        row["metric_path"] = f"metrics_{run_id}.json"
        return row

    monkeypatch.setattr(runner, "execute_config", fake_execute_config)
    monkeypatch.setattr(runner, "update_grid_row_from_metric", lambda row: None)

    rows = runner.execute_grid_phase(pa, "grid-run")

    assert rows[0]["metric_path"] == "existing_metrics.json"
    assert calls[0] == 2
    assert calls[-1] == 108


def test_pretty_analysis_does_not_import_previous_nicme_rows(tmp_path, monkeypatch):
    out = tmp_path / "results"
    metric = tmp_path / "metrics.json"
    metric.write_text(
        """{
          "decision_reports": {
            "argmax": {
              "target_recall_min": 0.5,
              "target_recall_macro": 0.5,
              "normalized_atc": 0.1,
              "atc": 1.0,
              "balanced_accuracy": 0.7,
              "accuracy": 0.7,
              "macro_f1": 0.7,
              "critical_pair_errors": {},
              "cost_matrix_sha256": "hash"
            },
            "cost_min": {
              "target_recall_min": 0.6,
              "target_recall_macro": 0.6,
              "normalized_atc": 0.09,
              "atc": 0.9,
              "balanced_accuracy": 0.65,
              "accuracy": 0.65,
              "macro_f1": 0.65,
              "critical_pair_errors": {},
              "cost_matrix_sha256": "hash"
            }
          }
        }"""
    )
    cfg = runner.base_config(
        "run",
        "ce_anchor_pretty",
        tmp_path / "splits" / "balanced",
        out,
        comparison_profile="pretty_balanced",
        learning_rate_profile="lr5e5",
    )
    config_path = out / "run" / "configs" / "0001.json"
    runner.write_json(config_path, cfg)
    row = runner.row_from_config(cfg, "run", 1, status="completed")
    row["config_path"] = str(config_path)
    row["metric_path"] = str(metric)
    runner.write_ledger(out / "run" / "run_ledger.csv", [row])
    monkeypatch.setattr(runner, "collect_previous_nicme_rows", lambda: [{"method": "old_nicme"}])

    pa = pretty_args(tmp_path, "lr5e5")
    pa.output_root = str(out)
    runner.analyze(pa)

    table = (out / "analysis" / "comparison_table.csv").read_text()
    assert "old_nicme" not in table
    assert "ce_anchor_pretty" in table
    assert "ce_cost_min_inference_pretty" in table


def test_combined_analysis_includes_all_three_lr_profiles(tmp_path):
    fields = [
        "method",
        "role",
        "source",
        "comparison_profile",
        "lr_profile",
        "learning_rate",
        "parent_learning_rate",
        "decision_mode",
        "target_recall_min",
        "target_recall_macro",
        "normalized_atc",
        "atc",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "critical_pair_error_count",
        "cost_matrix_sha256",
        "metric_path",
    ]
    for lr_profile, lr in [("lr1e5", "1e-05"), ("lr5e5", "5e-05"), ("lr1e4", "0.0001")]:
        root = tmp_path / f"results_{lr_profile}" / "analysis"
        root.mkdir(parents=True)
        with open(root / "comparison_table.csv", "w", newline="") as f:
            writer = runner.csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "method": f"nicme_{lr_profile}",
                    "role": f"nicme_{lr_profile}",
                    "source": "test",
                    "comparison_profile": "pretty_balanced",
                    "lr_profile": lr_profile,
                    "learning_rate": lr,
                    "decision_mode": "argmax",
                    "target_recall_min": "0.9",
                    "normalized_atc": "0.01",
                    "balanced_accuracy": "0.95",
                    "critical_pair_error_count": "1",
                }
            )
    pa = pretty_args(tmp_path, "lr5e5")
    pa.lr1e5_output_root = str(tmp_path / "results_lr1e5")
    pa.lr5e5_output_root = str(tmp_path / "results_lr5e5")
    pa.lr1e4_output_root = str(tmp_path / "results_lr1e4")

    runner.combined_analyze(pa)

    table = (tmp_path / "combined" / "comparison_table.csv").read_text()
    summary = (tmp_path / "combined" / "comparison_summary.md").read_text()
    assert "lr1e5" in table
    assert "lr5e5" in table
    assert "lr1e4" in table
    assert "nicme_lr1e5" in table
    assert "nicme_lr5e5" in table
    assert "nicme_lr1e4" in table
    assert "Triple-LR" in summary
    assert "very conservative" in summary


def test_sota_special_prediction_and_inference_configs(tmp_path):
    sosr = runner.base_config("run", "sosr_cnn", tmp_path / "splits" / "balanced", tmp_path / "results")
    anchor = runner.base_config("run", "ce_anchor", tmp_path / "splits" / "balanced", tmp_path / "results")

    assert sosr["loss_function"] == "sosr_cnn"
    assert sosr["prediction_mode"] == "sosr_argmin"
    assert anchor["loss_function"] == "cross_entropy"
    assert anchor["decision_modes"] == "argmax,cost_min"


def test_checkpoint_parent_points_to_trainer_output_dir(tmp_path):
    cfg = runner.base_config("run", "ce_anchor", tmp_path / "splits" / "balanced", tmp_path / "results")

    assert runner.checkpoint_parent(cfg) == Path("checkpoints") / cfg["output_dir"]


def test_checkpoint_parent_honors_checkpoint_root(tmp_path):
    cfg = runner.base_config("run", "ce_anchor", tmp_path / "splits" / "balanced", tmp_path / "results")
    cfg["checkpoint_root"] = str(tmp_path / "scratch_checkpoints")

    assert runner.checkpoint_parent(cfg) == tmp_path / "scratch_checkpoints" / cfg["output_dir"]


def test_initial_checkpoint_loader_loads_weights(tmp_path):
    from scripts.train import load_initial_checkpoint

    model = torch.nn.Linear(2, 2)
    expected = torch.nn.Linear(2, 2)
    with torch.no_grad():
        expected.weight.fill_(3.0)
        expected.bias.fill_(1.5)
    checkpoint = tmp_path / "pytorch_model.bin"
    torch.save(expected.state_dict(), checkpoint)

    loaded = load_initial_checkpoint(model, str(checkpoint))

    assert loaded == checkpoint
    assert torch.allclose(model.weight, expected.weight)
    assert torch.allclose(model.bias, expected.bias)


def test_csada_targets_only_critical_directed_pairs_and_preserves_bounds():
    from scripts.train import CustomTrainer
    from utils.loss_functions import LossFunctions

    trainer = object.__new__(CustomTrainer)
    trainer.lossClass = LossFunctions(cost_matrix=[[0.0, 10.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    trainer.csada_attack_steps = 1
    trainer.csada_epsilon = 2.0 / 255.0
    trainer.csada_step_size = 1.0 / 255.0
    trainer.csada_cost_temperature = 3.0

    labels = torch.tensor([0, 1], device=trainer.lossClass.device)
    target_labels, selected_costs, mask = trainer._critical_targets(labels)

    assert target_labels.tolist() == [1, 0]
    assert selected_costs.tolist() == [10.0, 1.0]
    assert mask.tolist() == [True, False]

    class ToyModel(torch.nn.Module):
        def forward(self, pixel_values):
            score = pixel_values.flatten(1).mean(dim=1)
            logits = torch.stack([score, -score, score * 0.0], dim=1)
            return type("Output", (), {"logits": logits})

    pixel_values = torch.zeros((2, 3, 4, 4), device=trainer.lossClass.device)
    adv, _costs, _mask = trainer._targeted_csada_examples(ToyModel().to(trainer.lossClass.device), pixel_values, labels)
    lower, upper = trainer._normalized_image_bounds(pixel_values)

    assert adv.shape == pixel_values.shape
    assert torch.all(adv >= lower)
    assert torch.all(adv <= upper)
