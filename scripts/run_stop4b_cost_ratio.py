"""Run Stop 4B cost-ratio sensitivity experiments.

This runner reuses the Stop 3 training configuration surface but varies the
explicit user-defined false-negative cost ratio on balanced datasets.  It is
kept separate from ``run_stop3_main.py`` so the currently running Stop 3B/4A
queue is not disturbed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from run_stop3_main import DATASETS, METHODS, MODEL_PRESETS, DatasetSpec, make_config, parse_csv


def ratio_label(ratio: float) -> str:
    text = f"{ratio:g}".replace(".", "p")
    return f"costr{text}"


def cost_matrix_for_ratio(dataset: DatasetSpec, ratio: float) -> list[list[float]]:
    if len(dataset.class_names) != 2:
        raise ValueError("Stop 4B currently supports binary datasets only")
    target = dataset.class_names.index(dataset.target_recall_class)
    other = 1 - target
    matrix = [[0.0, 0.0], [0.0, 0.0]]
    matrix[target][other] = float(ratio)
    matrix[other][target] = 1.0
    return matrix


def build_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    datasets = parse_csv(args.datasets)
    models = parse_csv(args.models)
    methods = parse_csv(args.methods)
    seeds = [int(seed) for seed in parse_csv(args.seeds)]
    ratios = [float(ratio) for ratio in parse_csv(args.ratios)]

    for dataset in datasets:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. Valid: {sorted(DATASETS)}")
    for model in models:
        if model not in MODEL_PRESETS:
            raise ValueError(f"Unknown model '{model}'. Valid: {sorted(MODEL_PRESETS)}")
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method '{method}'. Valid: {METHODS}")

    runs: list[dict[str, Any]] = []
    for dataset_key in datasets:
        base_dataset = DATASETS[dataset_key]
        for ratio in ratios:
            dataset = replace(base_dataset, cost_matrix=cost_matrix_for_ratio(base_dataset, ratio))
            for model in models:
                for seed in seeds:
                    for method in methods:
                        cfg = make_config(dataset, model, method, seed, args.phase)
                        label = ratio_label(ratio)
                        cfg["cost_ratio"] = ratio
                        cfg["stop4b_note"] = "Cost-ratio sensitivity: target-class false-negative cost is ratio and reverse error is 1."
                        cfg["output_dir"] = f"{args.phase}_{dataset.name}_{label}_{model}_{method}_seed{seed}"
                        runs.append(cfg)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    return runs


def write_manifest(runs: list[dict[str, Any]], output_root: Path, args: argparse.Namespace) -> list[Path]:
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths: list[Path] = []
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
        "ratios": parse_csv(args.ratios),
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
                "cost_ratio": cfg["cost_ratio"],
                "cost_matrix": cfg["cost_matrix"],
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

    for idx, cfg_path in enumerate(config_paths, start=1):
        if time_budget_hours is not None and (time.time() - start) / 3600.0 >= time_budget_hours:
            break
        cfg = json.load(open(cfg_path))
        stdout_log = logs_dir / f"{idx:03d}_{cfg['output_dir']}.stdout.log"
        stderr_log = logs_dir / f"{idx:03d}_{cfg['output_dir']}.stderr.log"
        cmd = [sys.executable, "scripts/train.py", "--config", str(cfg_path)]
        elapsed_start = time.time()
        with open(stdout_log, "w") as stdout_f, open(stderr_log, "w") as stderr_f:
            proc = subprocess.run(cmd, stdout=stdout_f, stderr=stderr_f, check=False)
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
            "cost_ratio": cfg["cost_ratio"],
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
            raise RuntimeError(f"Stop 4B run failed: {cfg['output_dir']} (see {stderr_log})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stop 4B cost-ratio sensitivity experiments")
    parser.add_argument("--phase", default="stop4b_cost_ratio_sensitivity")
    parser.add_argument("--datasets", default="spider_balanced,breakhis_balanced")
    parser.add_argument("--models", default="vit")
    parser.add_argument("--methods", default="ce_calibrated_cost_min,nicme_logit_adjustment,nicme_hybrid")
    parser.add_argument("--ratios", default="1,2,5,10,20")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--output-root", default="results/stop4b_cost_ratio_sensitivity")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--cleanup-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = build_runs(args)
    config_paths = write_manifest(runs, output_root, args)
    print(f"Wrote {len(config_paths)} Stop 4B configs to {output_root}")
    if args.execute:
        run_queue(config_paths, output_root, args.cleanup_checkpoints, args.time_budget_hours)


if __name__ == "__main__":
    main()
