import argparse
import csv
import json
from pathlib import Path

import pytest

from nicme.dataset_profiles import PMI_PILLS_10_NO_CAL_PROFILE
from scripts import run_pmi10_no_cal_experiments as runner


def test_pmi10_smoke_configs_are_argmax_no_calibration_full_finetune(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})
    args = argparse.Namespace(
        phase="smoke",
        models=None,
        losses=None,
        configs=None,
        hpo_profile=runner.DEFAULT_HPO_PROFILE,
        seed=42,
        epochs=None,
        trials=1,
        split_dir=str(tmp_path / "splits" / "natural"),
        output_root=str(tmp_path / "results"),
    )

    runs = runner.build_runs(args)

    assert len(runs) == 6
    assert {cfg["decision_modes"] for cfg in runs} == {"argmax"}
    assert {cfg["calibration_enabled"] for cfg in runs} == {False}
    assert {cfg["peft_enabled"] for cfg in runs} == {False}
    assert {cfg["dataset_profile"] for cfg in runs} == {"pmi_pills_10_no_cal"}
    assert {cfg["cost_matrix_source"] for cfg in runs} == {"dataset"}
    assert {cfg["seed"] for cfg in runs} == {42}
    assert {cfg["num_labels"] for cfg in runs} == {PMI_PILLS_10_NO_CAL_PROFILE.num_classes}
    assert {cfg["loss_function"] for cfg in runs} == {"cross_entropy", "nicme_hybrid"}
    assert len({cfg["cost_matrix_sha256"] for cfg in runs}) == 1


def test_pmi10_runner_objective_prefers_recall_then_lower_cost():
    high_recall = {
        "target_recall": 0.80,
        "target_recall_macro": 0.75,
        "normalized_atc": 0.20,
        "balanced_accuracy": 0.70,
        "accuracy": 0.80,
        "critical_pair_errors": {"a->b": {"errors": 3}},
    }
    lower_cost_lower_recall = {
        "target_recall": 0.79,
        "target_recall_macro": 0.78,
        "normalized_atc": 0.10,
        "balanced_accuracy": 0.80,
        "accuracy": 0.85,
        "critical_pair_errors": {"a->b": {"errors": 0}},
    }

    assert runner.report_objective_key(high_recall) > runner.report_objective_key(lower_cost_lower_recall)


def test_select_requires_completed_screen_metrics(tmp_path):
    screen_dir = tmp_path / "results" / "screen"
    screen_dir.mkdir(parents=True)
    with open(screen_dir / "run_ledger.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=runner.LEDGER_FIELDS)
        writer.writeheader()
        writer.writerow({"status": "planned", "loss": "nicme_hybrid"})

    args = argparse.Namespace(output_root=str(tmp_path / "results"), top_k=3)

    with pytest.raises(RuntimeError, match="No completed NICME hybrid screen runs"):
        runner.select_top_backbones(args)


def test_final_phase_uses_hpo_best_configs(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})
    output_root = tmp_path / "results"
    nicme_config = runner.base_config(
        "hpo-nicme",
        "convnext_tiny.fb_in22k_ft_in1k",
        "nicme_hybrid",
        30,
        42,
        tmp_path / "splits" / "natural",
        output_root,
        overrides={"learning_rate": 1e-5},
    )
    ce_config = runner.base_config(
        "hpo-ce",
        "convnext_tiny.fb_in22k_ft_in1k",
        "ce",
        30,
        42,
        tmp_path / "splits" / "natural",
        output_root,
        overrides={"learning_rate": 2e-5},
    )
    for phase, cfg in (("hpo-nicme", nicme_config), ("hpo-ce", ce_config)):
        path = output_root / phase / "best_config.json"
        path.parent.mkdir(parents=True)
        runner.write_json(path, cfg)

    args = argparse.Namespace(
        phase="final",
        models=None,
        losses=None,
        configs=None,
        hpo_profile=runner.DEFAULT_HPO_PROFILE,
        seed=42,
        epochs=None,
        trials=1,
        split_dir=str(tmp_path / "splits" / "natural"),
        output_root=str(output_root),
    )

    runs = runner.build_runs(args)

    assert [cfg["pmi10_loss"] for cfg in runs] == ["nicme_hybrid", "ce"]
    assert {cfg["pmi10_phase"] for cfg in runs} == {"final"}
    assert {cfg["calibration_enabled"] for cfg in runs} == {False}
    assert {cfg["decision_modes"] for cfg in runs} == {"argmax"}
    assert {cfg["peft_enabled"] for cfg in runs} == {False}
    assert {cfg["cost_matrix_sha256"] for cfg in runs} == {PMI_PILLS_10_NO_CAL_PROFILE.cost_matrix_sha256}
    assert runs[0]["learning_rate"] == 1e-5
    assert runs[1]["learning_rate"] == 2e-5
    assert all(cfg["output_dir"].startswith("final_") for cfg in runs)


def test_final_phase_requires_best_configs_or_manual_models(tmp_path):
    args = argparse.Namespace(
        phase="final",
        models=None,
        losses=None,
        configs=None,
        hpo_profile=runner.DEFAULT_HPO_PROFILE,
        seed=42,
        epochs=None,
        trials=1,
        split_dir=str(tmp_path / "splits" / "natural"),
        output_root=str(tmp_path / "results"),
    )

    with pytest.raises(FileNotFoundError, match="Final PMI-10 phases require"):
        runner.build_runs(args)


class FakeTrial:
    number = 0

    def __init__(self):
        self.float_calls = {}
        self.categorical_calls = {}

    def suggest_float(self, name, low, high, log=False):
        self.float_calls[name] = {"low": low, "high": high, "log": log}
        return low

    def suggest_categorical(self, name, choices):
        self.categorical_calls[name] = list(choices)
        return list(choices)[0]


def hpo_args(tmp_path, phase="hpo-nicme", profile=runner.CONVNEXT_BASE_FAST_PROFILE):
    return argparse.Namespace(
        phase=phase,
        models=None,
        losses=None,
        configs=None,
        hpo_profile=profile,
        seed=42,
        epochs=None,
        trials=1,
        variant="natural",
        split_dir=str(tmp_path / "splits" / "natural"),
        output_root=str(tmp_path / "results"),
        per_run_timeout_minutes=1,
        execute=False,
    )


def test_convnext_base_fast_hpo_profile_is_fixed_and_narrow(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})
    args = hpo_args(tmp_path)
    trial = FakeTrial()

    cfg = runner.build_hpo_trial_config(
        trial,
        args,
        "nicme_hybrid",
        runner.hpo_models(args),
        runner.CONVNEXT_BASE_FAST_PROFILE,
    )

    assert cfg["pmi10_backbone"] == runner.CONVNEXT_BASE_MODEL
    assert cfg["image_size"] == 224
    assert cfg["batch_size"] == 16
    assert cfg["gradient_accumulation_steps"] == 4
    assert cfg["lr_scheduler_type"] == "cosine"
    assert cfg["early_stopping_patience"] == 4
    assert cfg["num_train_epochs"] == 24
    assert "image_size" not in trial.categorical_calls
    assert trial.float_calls["learning_rate"] == {"low": 1.5e-5, "high": 1.0e-4, "log": True}
    assert trial.float_calls["weight_decay"] == {"low": 0.0, "high": 0.05, "log": False}
    assert trial.float_calls["warmup_ratio"] == {"low": 0.02, "high": 0.10, "log": False}
    assert trial.float_calls["nicme_logit_cost_scale"] == {"low": 0.08, "high": 0.50, "log": True}
    assert trial.float_calls["cs_lambda"] == {"low": 0.02, "high": 0.25, "log": False}
    assert trial.categorical_calls["cs_warmup_epochs"] == [1, 2, 3]


def test_convnext_base_v3_balanced_hpo_profile_is_fixed_and_narrow(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})
    args = hpo_args(tmp_path, phase="hpo-v3", profile=runner.CONVNEXT_BASE_V3_BALANCED_PROFILE)
    args.variant = "balanced"
    args.split_dir = str(tmp_path / "splits" / "balanced")
    trial = FakeTrial()

    cfg = runner.build_hpo_trial_config(
        trial,
        args,
        "nicme_v3_hybrid",
        runner.hpo_models(args),
        runner.CONVNEXT_BASE_V3_BALANCED_PROFILE,
    )

    assert cfg["pmi10_backbone"] == runner.CONVNEXT_BASE_MODEL
    assert cfg["variant"] == "balanced"
    assert cfg["dataset"] == "pmi_pills_10_no_cal_balanced"
    assert cfg["loss_function"] == "nicme_v3_hybrid"
    assert cfg["image_size"] == 224
    assert cfg["batch_size"] == 16
    assert cfg["gradient_accumulation_steps"] == 4
    assert cfg["lr_scheduler_type"] == "cosine"
    assert cfg["early_stopping_patience"] == 5
    assert cfg["num_train_epochs"] == 32
    assert cfg["train_random_resize_crop"] is True
    assert cfg["train_random_resized_crop_scale_min"] == 0.8
    assert cfg["train_random_resized_crop_scale_max"] == 1.0
    assert cfg["train_horizontal_flip"] is False
    assert cfg["train_random_quarter_turns"] is True
    assert "image_size" not in trial.categorical_calls
    assert trial.float_calls["learning_rate"] == {"low": 1.5e-5, "high": 9.0e-5, "log": True}
    assert trial.float_calls["weight_decay"] == {"low": 0.0, "high": 0.05, "log": False}
    assert trial.float_calls["warmup_ratio"] == {"low": 0.02, "high": 0.12, "log": False}
    assert trial.float_calls["nicme_logit_cost_scale"] == {"low": 0.05, "high": 0.70, "log": True}
    assert trial.float_calls["cs_lambda"] == {"low": 0.0, "high": 0.25, "log": False}
    assert trial.categorical_calls["cs_warmup_epochs"] == [0, 1, 2, 3, 5]
    assert trial.categorical_calls["train_color_jitter"] == [False, True]


def test_default_multi_model_hpo_rejects_dynamic_image_size_space():
    with pytest.raises(ValueError, match="dynamic Optuna categorical space"):
        runner.validate_static_default_hpo_space(
            [
                "convnext_base.fb_in22k_ft_in1k",
                "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
            ],
            runner.DEFAULT_HPO_PROFILE,
        )


def test_ce_matched_config_copies_non_loss_hyperparameters(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})
    output_root = tmp_path / "results"
    nicme_config = runner.base_config(
        "hpo-nicme",
        runner.CONVNEXT_BASE_MODEL,
        "nicme_hybrid",
        24,
        42,
        tmp_path / "splits" / "natural",
        output_root,
        overrides={
            "learning_rate": 3e-5,
            "weight_decay": 0.02,
            "warmup_ratio": 0.03,
            "train_color_jitter": True,
            "nicme_logit_cost_scale": 0.12,
            "cs_lambda": 0.08,
        },
    )
    nicme_path = output_root / "hpo-nicme" / "best_config.json"
    runner.write_json(nicme_path, nicme_config)
    args = hpo_args(tmp_path, phase="hpo-ce-matched")

    matched_path = runner.create_ce_matched_config(args)
    matched = json.loads(Path(matched_path).read_text())

    assert matched["pmi10_run_role"] == "ce_matched"
    assert matched["loss_function"] == "cross_entropy"
    assert matched["cs_lambda"] == 0.0
    assert matched["cs_warmup_epochs"] == 0
    assert matched["nicme_logit_cost_scale"] == 1.0
    assert matched["learning_rate"] == 3e-5
    assert matched["weight_decay"] == 0.02
    assert matched["warmup_ratio"] == 0.03
    assert matched["train_color_jitter"] is True


def test_v2_matched_config_copies_v3_non_loss_hyperparameters(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})
    output_root = tmp_path / "results"
    v3_config = runner.base_config(
        "hpo-v3",
        runner.CONVNEXT_BASE_MODEL,
        "nicme_v3_hybrid",
        32,
        42,
        tmp_path / "splits" / "balanced",
        output_root,
        overrides={
            "learning_rate": 3e-5,
            "weight_decay": 0.02,
            "warmup_ratio": 0.03,
            "train_color_jitter": True,
            "nicme_logit_cost_scale": 0.12,
            "cs_lambda": 0.08,
            "cs_warmup_epochs": 3,
        },
        variant="balanced",
    )
    runner.write_json(output_root / "hpo-v3" / "best_config.json", v3_config)
    args = hpo_args(tmp_path, phase="hpo-v2-matched", profile=runner.CONVNEXT_BASE_V3_BALANCED_PROFILE)
    args.variant = "balanced"
    args.split_dir = str(tmp_path / "splits" / "balanced")

    matched_path = runner.create_v2_matched_config(args)
    matched = json.loads(Path(matched_path).read_text())

    assert matched["pmi10_run_role"] == "nicme_v2_matched"
    assert matched["loss_function"] == "nicme_hybrid"
    assert matched["pmi10_loss"] == "nicme_hybrid"
    assert matched["learning_rate"] == 3e-5
    assert matched["weight_decay"] == 0.02
    assert matched["warmup_ratio"] == 0.03
    assert matched["train_color_jitter"] is True
    assert matched["nicme_logit_cost_scale"] == 0.12
    assert matched["cs_lambda"] == 0.08
    assert matched["cs_warmup_epochs"] == 3


def test_final_phase_uses_three_unique_roles(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})
    output_root = tmp_path / "results"
    configs = [
        ("hpo-nicme", "nicme_hybrid", "nicme_best"),
        ("hpo-ce-matched", "ce", "ce_matched"),
        ("hpo-ce", "ce", "ce_tuned"),
    ]
    for phase, loss, role in configs:
        cfg = runner.base_config(
            phase,
            runner.CONVNEXT_BASE_MODEL,
            loss,
            24,
            42,
            tmp_path / "splits" / "natural",
            output_root,
        )
        cfg["pmi10_run_role"] = role
        runner.write_json(output_root / phase / "best_config.json", cfg)
    args = hpo_args(tmp_path, phase="final")

    runs = runner.build_runs(args)

    assert [cfg["pmi10_run_role"] for cfg in runs] == ["nicme_best", "ce_matched", "ce_tuned"]
    assert len({cfg["output_dir"] for cfg in runs}) == 3
    assert all(cfg["output_dir"].startswith("final_") for cfg in runs)


def test_v3_final_phase_uses_four_unique_roles(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})
    output_root = tmp_path / "results"
    configs = [
        ("hpo-v3", "nicme_v3_hybrid", "nicme_v3_best"),
        ("hpo-ce-matched", "ce", "ce_matched"),
        ("hpo-ce", "ce", "ce_tuned"),
        ("hpo-v2-matched", "nicme_hybrid", "nicme_v2_matched"),
    ]
    for phase, loss, role in configs:
        cfg = runner.base_config(
            phase,
            runner.CONVNEXT_BASE_MODEL,
            loss,
            32,
            42,
            tmp_path / "splits" / "balanced",
            output_root,
            variant="balanced",
        )
        cfg["pmi10_run_role"] = role
        runner.write_json(output_root / phase / "best_config.json", cfg)
    args = hpo_args(tmp_path, phase="final", profile=runner.CONVNEXT_BASE_V3_BALANCED_PROFILE)
    args.variant = "balanced"
    args.split_dir = str(tmp_path / "splits" / "balanced")

    runs = runner.build_runs(args)

    assert [cfg["pmi10_run_role"] for cfg in runs] == [
        "nicme_v3_best",
        "ce_matched",
        "ce_tuned",
        "nicme_v2_matched",
    ]
    assert len({cfg["output_dir"] for cfg in runs}) == 4
    assert all(cfg["variant"] == "balanced" for cfg in runs)
    assert all(cfg["output_dir"].startswith("final_") for cfg in runs)


def test_tpt_stress_uses_only_v3_and_v2_with_early_stopping_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})
    output_root = tmp_path / "results"
    for phase, loss, role in (
        ("hpo-v3", "nicme_v3_hybrid", "nicme_v3_best"),
        ("hpo-v2-matched", "nicme_hybrid", "nicme_v2_matched"),
    ):
        cfg = runner.base_config(
            phase,
            runner.CONVNEXT_BASE_MODEL,
            loss,
            32,
            42,
            tmp_path / "splits" / "balanced",
            output_root,
            variant="balanced",
        )
        cfg["pmi10_run_role"] = role
        runner.write_json(output_root / phase / "best_config.json", cfg)
    args = hpo_args(tmp_path, phase="tpt-stress", profile=runner.CONVNEXT_BASE_V3_BALANCED_PROFILE)
    args.variant = "balanced"
    args.split_dir = str(tmp_path / "splits" / "balanced")
    args.epochs = 64

    runs = runner.build_runs(args)

    assert [cfg["pmi10_run_role"] for cfg in runs] == ["tpt_nicme_v3_best", "tpt_nicme_v2_matched"]
    assert {cfg["early_stopping_patience"] for cfg in runs} == {0}
    assert {cfg["num_train_epochs"] for cfg in runs} == {64}
    assert all(cfg["output_dir"].startswith("tpt-stress_") for cfg in runs)


def test_hpo_trial_exception_is_recorded_without_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "timm_data_config", lambda model_id: {})

    def explode(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(runner, "execute_run", explode)
    args = hpo_args(tmp_path)
    trial_rows = []

    score = runner.run_hpo_trial(
        FakeTrial(),
        args,
        "nicme_hybrid",
        runner.hpo_models(args),
        runner.CONVNEXT_BASE_FAST_PROFILE,
        trial_rows,
        tmp_path / "results" / "hpo-nicme",
    )

    assert score == runner.BAD_TRIAL_SCORE
    assert len(trial_rows) == 1
    assert trial_rows[0]["status"] == "failed"
    assert "simulated training failure" in trial_rows[0]["failure_reason"]
    assert (tmp_path / "results" / "hpo-nicme" / "hpo_trials.csv").exists()
