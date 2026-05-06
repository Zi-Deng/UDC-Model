"""PMI-20 NICME component ablation runner.

This runner trains only the missing PMI-20 ablation components for the paper:
regularizer-only and margin-only. CE and full NICME are reused from completed
camera-ready result roots during analysis.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

from scripts import run_pmi10_camera_ready_lr5e5 as camera
from scripts import run_pmi10_sota_baselines as base
from scripts import run_pmi20_camera_ready_lr5e5 as pmi20
from scripts import run_pmi20_nicme_alpha_lambda6_lr5e5 as alpha6

DEFAULT_OUTPUT_ROOT = Path("results/pmi20_component_ablation_lr5e5_multiseed_20260506")
DEFAULT_SEEDS = (42, 43, 44)
COMPARISON_PROFILE = "pmi20_component_ablation_lr5e5_multiseed"

CE_ABLATION_METHOD = "ce"
REGULARIZER_METHOD = "nicme_regularizer_only_l0p1"
MARGIN_METHOD = "nicme_margin_only_a0p5"
FULL_NICME_METHOD = "nicme_full_a0p5_l0p1"
TRAIN_METHODS = (REGULARIZER_METHOD, MARGIN_METHOD)
REPORT_METHODS = (CE_ABLATION_METHOD, REGULARIZER_METHOD, MARGIN_METHOD, FULL_NICME_METHOD)

REGULARIZER_LAMBDA = 0.1
MARGIN_ALPHA = 0.5

DEFAULT_CE_PER_SEED = Path("results/pmi20_camera_ready_lr5e5_multiseed_20260504/analysis/per_seed_metrics.csv")
DEFAULT_FULL_NICME_PER_SEED = Path(
    "results/pmi20_nicme_alpha_lambda6_lr5e5_multiseed_20260504/analysis/per_seed_metrics.csv"
)

METRICS = pmi20.METRICS
HIGHER_IS_BETTER = pmi20.HIGHER_IS_BETTER
LOWER_IS_BETTER = pmi20.LOWER_IS_BETTER
TOL = pmi20.TOL


def parse_seeds(value: str) -> tuple[int, ...]:
    return camera.parse_seeds(value)


def seed_root(output_root: Path, seed: int) -> Path:
    return output_root / "seeds" / f"seed_{seed}"


def display_name(method: str) -> str:
    names = {
        CE_ABLATION_METHOD: "CE",
        REGULARIZER_METHOD: "Regularizer-only",
        MARGIN_METHOD: "Margin-only",
        FULL_NICME_METHOD: "Full NICME",
    }
    return names.get(method, method)


def build_training_config(
    phase: str,
    method: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
) -> dict[str, Any]:
    if method not in TRAIN_METHODS:
        raise ValueError(f"Unknown PMI-20 ablation method: {method}")
    cfg = pmi20.base_config(
        phase,
        pmi20.NICME_METHOD,
        split_dir,
        output_root,
        seed,
        epochs=1 if smoke else None,
    )
    cfg["sota_method"] = method
    cfg["sota_role"] = "pmi20_component_ablation"
    cfg["sota_comparison_profile"] = COMPARISON_PROFILE
    cfg["component_ablation"] = True
    cfg["pilot_selected_hyperparameters"] = False
    cfg.pop("added_fixed_candidate", None)
    cfg.pop("pilot_hpo_run", None)
    cfg.pop("grid_alpha", None)
    cfg.pop("grid_cs_lambda", None)

    if method == REGULARIZER_METHOD:
        cfg["loss_function"] = "cs_regularized_ce"
        cfg["nicme_logit_cost_scale"] = 0.0
        cfg["cs_lambda"] = REGULARIZER_LAMBDA
        cfg["cs_warmup_epochs"] = 2
    elif method == MARGIN_METHOD:
        cfg["loss_function"] = "nicme_logit_adjustment"
        cfg["nicme_logit_cost_scale"] = MARGIN_ALPHA
        cfg["cs_lambda"] = 0.0
        cfg["cs_warmup_epochs"] = 0
    cfg["output_dir"] = f"pmi20_component_ablation_{phase}_{method}_lr5e5_s{seed}"
    return cfg


def build_seed_configs(
    phase: str,
    split_dir: Path,
    output_root: Path,
    seed: int,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    return [build_training_config(phase, method, split_dir, output_root, seed, smoke=smoke) for method in TRAIN_METHODS]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    camera.write_csv(path, rows, fields)


def write_dry_run(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    split_dir = Path(args.split_dir)
    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        root = seed_root(output_root, seed)
        rows: list[dict[str, Any]] = []
        configs = build_seed_configs("dry-run", split_dir, root, seed)
        for idx, cfg in enumerate(configs, start=1):
            row = base.row_from_config(cfg, "dry-run", idx)
            cfg_hash = row["config_hash"]
            config_path = root / "dry-run" / "configs" / f"{idx:04d}_{cfg_hash}.json"
            base.write_json(config_path, cfg)
            row["config_path"] = str(config_path)
            rows.append(row)
            all_rows.append({"seed": seed, **row})
        base.write_ledger(root / "dry-run" / "run_ledger.csv", rows)
    write_csv(output_root / "dry_run_plan.csv", all_rows, ["seed", *base.LEDGER_FIELDS])
    return all_rows


def execute_seed_phase(args: argparse.Namespace, seed: int, phase: str, smoke: bool = False) -> list[dict[str, Any]]:
    root = seed_root(Path(args.output_root), seed)
    ledger_path = root / phase / "run_ledger.csv"
    existing = camera.completed_existing_rows(ledger_path)
    previous_failures = camera.noncompleted_existing_rows(ledger_path)
    rows: list[dict[str, Any]] = []
    seed_args = argparse.Namespace(**vars(args))
    seed_args.output_root = str(root)

    for idx, method in enumerate(TRAIN_METHODS, start=1):
        cfg = build_training_config(phase, method, Path(args.split_dir), root, seed, smoke=smoke)
        if method in existing:
            row = existing[method]
        else:
            if method in previous_failures:
                camera.preserve_failed_attempt(
                    previous_failures[method],
                    ledger_path,
                    "resume_preserved_previous_noncompleted",
                )
            row = camera.execute_config_with_camera_retry(cfg, phase, idx, seed_args, ledger_path)
        rows.append(row)
        base.write_ledger(ledger_path, rows)
        if row.get("status") != "completed":
            raise RuntimeError(f"{phase} failed for seed={seed}, method={method}: {row.get('failure_reason')}")
    return rows


def execute_phase(args: argparse.Namespace, phase: str, smoke: bool = False) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        rows = execute_seed_phase(args, seed, phase, smoke=smoke)
        all_rows.extend({"seed": seed, **row} for row in rows)
    return all_rows


def config_metadata(config_path: str) -> dict[str, Any]:
    return pmi20.config_metadata(config_path)


def metric_row_from_ledger(seed: int, row: dict[str, Any], method: str | None = None) -> dict[str, Any]:
    method = method or row["method"]
    metadata = config_metadata(row.get("config_path", ""))
    mode, report = base.load_metric_report(row["metric_path"], method)
    metric_row = base.metric_row(
        method,
        method,
        row["metric_path"],
        mode,
        report,
        f"pmi20_component_ablation_seed_{seed}",
        metadata,
    )
    metric_row.update(
        {
            "seed": seed,
            "config_path": row.get("config_path", ""),
            "checkpoint_path": row.get("checkpoint_path", ""),
            "alpha": metadata.get("alpha", ""),
            "cs_lambda": metadata.get("cs_lambda", ""),
            "loss_function": metadata.get("loss_function", ""),
            "component": method,
        }
    )
    return metric_row


def load_trained_metric_rows(output_root: Path, seed: int, allow_incomplete: bool = False) -> list[dict[str, Any]]:
    ledger_path = seed_root(output_root, seed) / "run" / "run_ledger.csv"
    rows = base.read_ledger(ledger_path)
    if not rows:
        if allow_incomplete:
            return []
        raise RuntimeError(f"Missing ablation run ledger for seed {seed}: {ledger_path}")
    by_method = {row["method"]: row for row in rows}
    missing = [method for method in TRAIN_METHODS if by_method.get(method, {}).get("status") != "completed"]
    if missing and not allow_incomplete:
        raise RuntimeError(f"Seed {seed} has incomplete PMI-20 ablation methods: {missing}")
    metric_rows: list[dict[str, Any]] = []
    for method in TRAIN_METHODS:
        row = by_method.get(method)
        if not row or row.get("status") != "completed":
            continue
        metric_rows.append(metric_row_from_ledger(seed, row, method))
    return metric_rows


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def reuse_existing_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    ce_rows = read_rows(Path(args.ce_per_seed))
    full_rows = read_rows(Path(args.full_nicme_per_seed))
    reused: list[dict[str, Any]] = []
    seed_set = {int(seed) for seed in args.seeds}
    for row in ce_rows:
        if row.get("method") == pmi20.CE_METHOD and int(row["seed"]) in seed_set:
            new = dict(row)
            new["method"] = CE_ABLATION_METHOD
            new["display_name"] = display_name(CE_ABLATION_METHOD)
            new["component"] = CE_ABLATION_METHOD
            new["alpha"] = "0.0"
            new["cs_lambda"] = "0.0"
            reused.append(new)
    for row in full_rows:
        if row.get("method") == "nicme_extra_a0p5_l0p1_pmi20" and int(row["seed"]) in seed_set:
            new = dict(row)
            new["method"] = FULL_NICME_METHOD
            new["display_name"] = display_name(FULL_NICME_METHOD)
            new["component"] = FULL_NICME_METHOD
            new["alpha"] = "0.5"
            new["cs_lambda"] = "0.1"
            reused.append(new)
    expected = {(seed, method) for seed in seed_set for method in (CE_ABLATION_METHOD, FULL_NICME_METHOD)}
    observed = {(int(row["seed"]), row["method"]) for row in reused}
    missing = sorted(expected - observed)
    if missing and not args.allow_incomplete_analysis:
        raise RuntimeError(f"Missing reused CE/full-NICME rows: {missing}")
    return reused


def to_float(value: Any) -> float:
    return camera.to_float(value)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def composite_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        -float(row.get("target_recall_min_mean") or 0.0),
        float(row.get("normalized_atc_mean") or 1.0),
        -float(row.get("balanced_accuracy_mean") or 0.0),
        -float(row.get("macro_f1_mean") or 0.0),
    )


def aggregate_metric_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault(row["method"], []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        out: dict[str, Any] = {
            "method": method,
            "display_name": display_name(method),
            "decision_mode": rows[0].get("decision_mode", "argmax"),
            "n_seeds": len(rows),
            "seeds": ",".join(str(row["seed"]) for row in sorted(rows, key=lambda item: int(item["seed"]))),
            "alpha": rows[0].get("alpha", ""),
            "cs_lambda": rows[0].get("cs_lambda", ""),
            "loss_function": rows[0].get("loss_function", ""),
        }
        for metric in METRICS:
            values = [to_float(row.get(metric)) for row in rows]
            values = [value for value in values if not math.isnan(value)]
            out[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            out[f"{metric}_std"] = sample_std(values) if values else ""
            if metric == "critical_pair_error_count":
                out[f"{metric}_total"] = int(sum(values)) if values else ""
        aggregate_rows.append(out)
    order = {method: idx for idx, method in enumerate(REPORT_METHODS, start=1)}
    aggregate_rows.sort(key=lambda row: order.get(row["method"], 999))
    for idx, row in enumerate(sorted(aggregate_rows, key=composite_sort_key), start=1):
        row["composite_rank"] = idx
    return aggregate_rows


def format_mean_std(row: dict[str, Any], metric: str, precision: int = 4) -> str:
    return camera.format_mean_std(row, metric, precision)


def format_tex_mean_std(row: dict[str, Any], metric: str, precision: int = 4) -> str:
    return camera.format_tex_mean_std(row, metric, precision)


def write_ablation_table(output_root: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    rows = sorted(aggregate_rows, key=lambda row: REPORT_METHODS.index(row["method"]))
    lines = [
        "# PMI-20 NICME Component Ablation",
        "",
        "Metrics are mean +/- sample standard deviation over seeds 42, 43, and 44.",
        "",
        "| Component | alpha | lambda | Target-Min Recall | Norm. ATC | Balanced Acc. | Macro F1 | Critical Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['display_name']} | {float(row.get('alpha') or 0):g} | {float(row.get('cs_lambda') or 0):g} | "
            f"{format_mean_std(row, 'target_recall_min')} | "
            f"{format_mean_std(row, 'normalized_atc', 6)} | "
            f"{format_mean_std(row, 'balanced_accuracy')} | "
            f"{format_mean_std(row, 'macro_f1')} | "
            f"{format_mean_std(row, 'critical_pair_error_count')} |"
        )
    (output_root / "analysis" / "pmi20_component_ablation_table.md").write_text("\n".join(lines) + "\n")

    tex = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Component & $\\alpha$ & $\\lambda$ & Target-min & Norm. ATC & Bal. Acc. & Macro F1 & Critical \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex.append(
            f"{row['display_name']} & {float(row.get('alpha') or 0):g} & {float(row.get('cs_lambda') or 0):g} & "
            f"{format_tex_mean_std(row, 'target_recall_min')} & "
            f"{format_tex_mean_std(row, 'normalized_atc', 6)} & "
            f"{format_tex_mean_std(row, 'balanced_accuracy')} & "
            f"{format_tex_mean_std(row, 'macro_f1')} & "
            f"{format_tex_mean_std(row, 'critical_pair_error_count')} \\\\"
        )
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    (output_root / "analysis" / "pmi20_component_ablation_table.tex").write_text("\n".join(tex) + "\n")


def analyze(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    metric_rows: list[dict[str, Any]] = []
    metric_rows.extend(reuse_existing_rows(args))
    for seed in args.seeds:
        metric_rows.extend(load_trained_metric_rows(output_root, seed, args.allow_incomplete_analysis))
    if not metric_rows:
        raise RuntimeError("No PMI-20 ablation metric rows available")
    for row in metric_rows:
        row["display_name"] = display_name(row["method"])

    per_seed_fields = [
        "seed",
        "method",
        "display_name",
        "component",
        "decision_mode",
        "alpha",
        "cs_lambda",
        "loss_function",
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
    ]
    write_csv(output_root / "analysis" / "per_seed_metrics.csv", metric_rows, per_seed_fields)
    aggregate_rows = aggregate_metric_rows(metric_rows)
    aggregate_fields = [
        "composite_rank",
        "method",
        "display_name",
        "decision_mode",
        "n_seeds",
        "seeds",
        "alpha",
        "cs_lambda",
        "loss_function",
    ]
    for metric in METRICS:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_std"])
        if metric == "critical_pair_error_count":
            aggregate_fields.append(f"{metric}_total")
    write_csv(output_root / "analysis" / "aggregate_metrics.csv", aggregate_rows, aggregate_fields)
    write_ablation_table(output_root, aggregate_rows)
    (output_root / "README.md").write_text(
        "# PMI-20 NICME Component Ablation\n\n"
        "This result root trains regularizer-only and margin-only NICME components over seeds 42, 43, and 44. "
        "CE and full NICME rows are reused from the completed paper-facing PMI-20 result roots.\n"
    )
    print(f"Wrote PMI-20 component ablation analysis under {output_root / 'analysis'}")


def assert_ablation_config(cfg: dict[str, Any]) -> None:
    pmi20.assert_pmi20_config(cfg)
    if cfg["loss_function"] == "cs_regularized_ce":
        if float(cfg["nicme_logit_cost_scale"]) != 0.0 or float(cfg["cs_lambda"]) != REGULARIZER_LAMBDA:
            raise ValueError("Regularizer-only config changed")
    elif cfg["loss_function"] == "nicme_logit_adjustment":
        if float(cfg["nicme_logit_cost_scale"]) != MARGIN_ALPHA or float(cfg["cs_lambda"]) != 0.0:
            raise ValueError("Margin-only config changed")
    else:
        raise ValueError(f"Unexpected ablation loss: {cfg['loss_function']}")


def run_preflight(args: argparse.Namespace) -> None:
    split_dir = Path(args.split_dir)
    pmi20.assert_split_ready(split_dir)
    if not args.skip_nvidia_smi:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True)
    for method in TRAIN_METHODS:
        assert_ablation_config(build_training_config("preflight", method, split_dir, Path(args.output_root), args.seeds[0]))
    print("PMI-20 component ablation preflight passed")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preflight", "dry-run", "smoke", "run", "analyze"], required=True)
    parser.add_argument("--split-dir", default=str(pmi20.DEFAULT_SPLIT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--per-run-timeout-minutes", type=int, default=720)
    parser.add_argument("--allow-incomplete-analysis", action="store_true")
    parser.add_argument("--skip-nvidia-smi", action="store_true")
    parser.add_argument("--ce-per-seed", default=str(DEFAULT_CE_PER_SEED))
    parser.add_argument("--full-nicme-per-seed", default=str(DEFAULT_FULL_NICME_PER_SEED))
    args = parser.parse_args(argv)

    if args.phase == "preflight":
        run_preflight(args)
    elif args.phase == "dry-run":
        rows = write_dry_run(args)
        print(f"Wrote {len(rows)} dry-run ablation configs under {Path(args.output_root)}")
    elif args.phase == "smoke":
        rows = execute_phase(args, "smoke", smoke=True)
        print(f"Completed {len(rows)} smoke ablation runs")
    elif args.phase == "run":
        rows = execute_phase(args, "run", smoke=False)
        print(f"Completed {len(rows)} long ablation runs")
    elif args.phase == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
