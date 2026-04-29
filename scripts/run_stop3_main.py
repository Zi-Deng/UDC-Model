"""Run the adjusted Stop 3 NICME main-paper experiment queue.

The generic binary runner is useful for simple grids, but Stop 3 needs a
decision-complete queue that encodes the Stop 2A-2C tuning lessons:

* full audited Stop 0 data splits;
* dataset-specific floors and selection metrics;
* tuned NICME logit-cost scales/lambdas;
* calibrated-threshold reporting in addition to argmax and cost-min decisions;
* automatic checkpoint cleanup after each run.

The script is intentionally sequential.  On a single workstation GPU this makes
failure modes, disk growth, and runtime accounting much easier to audit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    class_names: list[str]
    target_recall_class: str
    cost_matrix: list[list[float]]
    target_recall_floor: float
    accuracy_floor: float
    selection_accuracy_metric: str
    epochs: int = 20


DATASETS: dict[str, DatasetSpec] = {
    "spider_balanced": DatasetSpec(
        name="spider_balanced",
        path="data/prepared/spider/splits/balanced",
        class_names=["black_widow", "false_widow"],
        target_recall_class="black_widow",
        cost_matrix=[[0.0, 10.0], [1.0, 0.0]],
        target_recall_floor=0.95,
        accuracy_floor=0.80,
        selection_accuracy_metric="accuracy",
    ),
    "spider_target_minority": DatasetSpec(
        name="spider_target_minority",
        path="data/prepared/spider/splits/target_minority",
        class_names=["black_widow", "false_widow"],
        target_recall_class="black_widow",
        cost_matrix=[[0.0, 10.0], [1.0, 0.0]],
        target_recall_floor=0.95,
        accuracy_floor=0.75,
        selection_accuracy_metric="balanced_accuracy",
    ),
    "spider_target_majority": DatasetSpec(
        name="spider_target_majority",
        path="data/prepared/spider/splits/target_majority",
        class_names=["black_widow", "false_widow"],
        target_recall_class="black_widow",
        cost_matrix=[[0.0, 10.0], [1.0, 0.0]],
        target_recall_floor=0.95,
        accuracy_floor=0.75,
        selection_accuracy_metric="balanced_accuracy",
    ),
    "breakhis_balanced": DatasetSpec(
        name="breakhis_balanced",
        path="data/prepared/breakhis/splits/balanced",
        class_names=["benign", "malignant"],
        target_recall_class="malignant",
        cost_matrix=[[0.0, 1.0], [10.0, 0.0]],
        target_recall_floor=0.97,
        accuracy_floor=0.85,
        selection_accuracy_metric="accuracy",
    ),
    "breakhis_natural": DatasetSpec(
        name="breakhis_natural",
        path="data/prepared/breakhis/splits/natural",
        class_names=["benign", "malignant"],
        target_recall_class="malignant",
        cost_matrix=[[0.0, 1.0], [10.0, 0.0]],
        target_recall_floor=0.97,
        accuracy_floor=0.80,
        selection_accuracy_metric="balanced_accuracy",
    ),
}


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "convnext": {
        "model": "facebook/convnext-tiny-224",
        "model_backend": "hf_auto",
        "model_type": "convnext",
        "peft_enabled": False,
        "learning_rate": 1e-4,
    },
    "vit": {
        "model": "google/vit-base-patch16-224-in21k",
        "model_backend": "hf_auto",
        "model_type": "vit",
        "peft_enabled": False,
        "learning_rate": 1e-4,
    },
    "timm_dinov3_vit_lora": {
        "model": "timm/vit_small_patch16_dinov3.lvd1689m",
        "model_backend": "timm",
        "model_type": "timm_dinov3_vit_lora",
        "timm_pretrained": True,
        "peft_enabled": True,
        "peft_r": 8,
        "peft_alpha": 16,
        "peft_dropout": 0.1,
        "peft_target_modules": "qkv",
        "peft_modules_to_save": "model.head",
        "learning_rate": 1e-4,
    },
    "timm_dinov3_convnext_lora": {
        "model": "timm/convnext_tiny.dinov3_lvd1689m",
        "model_backend": "timm",
        "model_type": "timm_dinov3_convnext_lora",
        "timm_pretrained": True,
        "peft_enabled": True,
        "peft_r": 8,
        "peft_alpha": 16,
        "peft_dropout": 0.1,
        # ConvNeXt has no q/v attention projections.  This is the Stop 1
        # audited MLP-adapter comparison, not attention LoRA.
        "peft_target_modules": "mlp.fc1,mlp.fc2",
        "peft_modules_to_save": "model.head",
        "learning_rate": 1e-4,
    },
}


METHODS = [
    "ce",
    "ce_calibrated_cost_min",
    "menon_logit_adjusted",
    "cs_regularized_ce",
    "nicme_logit_adjustment",
    "nicme_hybrid",
]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def tuned_method_settings(dataset_name: str, method: str) -> dict[str, Any]:
    """Return Stop 2-informed method-specific config values."""
    settings: dict[str, Any] = {
        "loss_function": method,
        "cs_lambda": 0.0,
        "cs_warmup_epochs": 0,
        "nicme_logit_cost_scale": 1.0,
        "logit_adjustment_tau": 1.0,
    }
    is_spider = dataset_name.startswith("spider")

    if method == "cs_regularized_ce":
        settings["cs_lambda"] = 0.10 if is_spider else 0.25
        settings["cs_warmup_epochs"] = 1
    elif method == "nicme_logit_adjustment":
        # Stop 2C found Spider needed a much softer but still meaningful logit
        # cost scale; BreaKHis tolerated smaller hybrid scales.
        settings["nicme_logit_cost_scale"] = 0.50 if is_spider else 0.30
    elif method == "nicme_hybrid":
        settings["nicme_logit_cost_scale"] = 0.50 if is_spider else 0.30
        settings["cs_lambda"] = 0.10 if is_spider else 0.25
        settings["cs_warmup_epochs"] = 1
    return settings


def make_config(dataset: DatasetSpec, model_family: str, method: str, seed: int, phase: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "dataset_host": "local_folder",
        "local_dataset_format": "manifest",
        "local_folder_path": dataset.path,
        "dataset": dataset.name,
        "class_names": dataset.class_names,
        "target_recall_class": dataset.target_recall_class,
        "target_recall_floor": dataset.target_recall_floor,
        "accuracy_floor": dataset.accuracy_floor,
        "selection_accuracy_metric": dataset.selection_accuracy_metric,
        "cost_matrix": dataset.cost_matrix,
        "num_labels": len(dataset.class_names),
        "num_train_epochs": dataset.epochs,
        "early_stopping_patience": 3,
        "batch_size": 16,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "wandb": False,
        "push_to_hub": False,
        "calibration_enabled": True,
        "decision_modes": "argmax,calibrated_cost_min,calibrated_threshold",
        "save_total_limit": 1,
        "seed": seed,
    }
    cfg.update(MODEL_PRESETS[model_family])
    cfg.update(tuned_method_settings(dataset.name, method))
    cfg["output_dir"] = f"{phase}_{dataset.name}_{model_family}_{method}_seed{seed}"
    cfg["stop3_phase"] = phase
    cfg["stop3_note"] = "Adjusted after Stop 2A-2C tuning; checkpoints deleted after metrics export."
    return cfg


def build_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_keys = parse_csv(args.datasets)
    model_keys = parse_csv(args.models)
    methods = parse_csv(args.methods)
    seeds = [int(value) for value in parse_csv(args.seeds)]

    for key in dataset_keys:
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset '{key}'. Valid: {sorted(DATASETS)}")
    for key in model_keys:
        if key not in MODEL_PRESETS:
            raise ValueError(f"Unknown model '{key}'. Valid: {sorted(MODEL_PRESETS)}")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Valid: {METHODS}")

    runs = []
    for dataset_key in dataset_keys:
        dataset = DATASETS[dataset_key]
        for model_key in model_keys:
            for seed in seeds:
                for method in methods:
                    runs.append(make_config(dataset, model_key, method, seed, args.phase))
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    return runs


def write_manifest(runs: list[dict[str, Any]], output_root: Path, args: argparse.Namespace) -> list[Path]:
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths = []
    for idx, cfg in enumerate(runs, start=1):
        path = config_dir / f"{idx:03d}_{cfg['output_dir']}.json"
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        config_paths.append(path)

    manifest = {
        "phase": args.phase,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": parse_csv(args.datasets),
        "models": parse_csv(args.models),
        "methods": parse_csv(args.methods),
        "seeds": parse_csv(args.seeds),
        "run_count": len(runs),
        "cleanup_checkpoints": args.cleanup_checkpoints,
        "time_budget_hours": args.time_budget_hours,
        "runs": [
            {
                "index": idx,
                "config": str(path),
                "output_dir": cfg["output_dir"],
                "dataset": cfg["dataset"],
                "model_type": cfg["model_type"],
                "method": cfg["loss_function"],
                "seed": cfg["seed"],
                "scale": cfg.get("nicme_logit_cost_scale"),
                "cs_lambda": cfg.get("cs_lambda"),
            }
            for idx, (path, cfg) in enumerate(zip(config_paths, runs, strict=True), start=1)
        ],
    }
    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return config_paths


def run_queue(config_paths: list[Path], output_root: Path, cleanup_checkpoints: bool, time_budget_hours: float | None) -> None:
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = output_root / "run_log.json"
    run_log: list[dict[str, Any]] = []
    start = time.time()

    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    for idx, cfg_path in enumerate(config_paths, start=1):
        if time_budget_hours is not None and (time.time() - start) / 3600.0 >= time_budget_hours:
            break
        with open(cfg_path) as f:
            cfg = json.load(f)

        stdout_log = logs_dir / f"{idx:03d}_{cfg['output_dir']}.stdout.log"
        stderr_log = logs_dir / f"{idx:03d}_{cfg['output_dir']}.stderr.log"
        cmd = [sys.executable, "scripts/train.py", "--config", str(cfg_path)]
        elapsed_start = time.time()
        with open(stdout_log, "w") as stdout_f, open(stderr_log, "w") as stderr_f:
            proc = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f, env=env, check=False)
        elapsed = time.time() - elapsed_start

        checkpoint_dir = Path("checkpoints") / cfg["output_dir"]
        checkpoint_removed = False
        if cleanup_checkpoints and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            checkpoint_removed = True

        row = {
            "index": idx,
            "config": str(cfg_path),
            "output_dir": cfg["output_dir"],
            "dataset": cfg["dataset"],
            "model_type": cfg["model_type"],
            "method": cfg["loss_function"],
            "seed": cfg["seed"],
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout_log": str(stdout_log),
            "stderr_log": str(stderr_log),
            "checkpoint_removed": checkpoint_removed,
        }
        run_log.append(row)
        with open(run_log_path, "w") as f:
            json.dump(run_log, f, indent=2)
        if proc.returncode != 0:
            raise RuntimeError(f"Stop 3 run failed: {cfg['output_dir']} (see {stderr_log})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adjusted Stop 3 NICME experiments")
    parser.add_argument("--phase", default="stop3a_balanced_primary")
    parser.add_argument("--datasets", default="spider_balanced,breakhis_balanced")
    parser.add_argument("--models", default="vit,timm_dinov3_vit_lora")
    parser.add_argument("--methods", default="ce,ce_calibrated_cost_min,menon_logit_adjusted,cs_regularized_ce,nicme_logit_adjustment,nicme_hybrid")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-root", default="results/stop3a_balanced_primary")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--cleanup-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = build_runs(args)
    config_paths = write_manifest(runs, output_root, args)
    print(f"Wrote {len(config_paths)} Stop 3 configs to {output_root}")
    if args.execute:
        run_queue(config_paths, output_root, args.cleanup_checkpoints, args.time_budget_hours)


if __name__ == "__main__":
    main()
