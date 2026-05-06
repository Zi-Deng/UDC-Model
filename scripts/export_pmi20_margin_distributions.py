"""Export PMI-20 critical-pair raw logit margins from saved checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

from nicme.modeling import build_model
from scripts import run_pmi20_camera_ready_lr5e5 as pmi20
from scripts import run_pmi20_nicme_alpha_lambda6_lr5e5 as alpha6
from scripts.train import load_initial_checkpoint
from utils.utils import collate_fn, get_device, preprocess_manifest_dataset

DEFAULT_OUTPUT = Path("paper_figures/data/pmi20_critical_pair_margins.csv")
DEFAULT_PMI20_PER_SEED = Path("results/pmi20_camera_ready_lr5e5_multiseed_20260504/analysis/per_seed_metrics.csv")
DEFAULT_ALPHA_PER_SEED = Path("results/pmi20_nicme_alpha_lambda6_lr5e5_multiseed_20260504/analysis/per_seed_metrics.csv")

SOURCE_METHODS = {
    "ce_anchor_pmi20": ("CE", "CE"),
    "cost_sensitive_regularized_ce": ("cost_sensitive_regularized_ce", "cost-sensitive regularized CE"),
    "csada": ("CSADA", "CSADA"),
    "nicme_extra_a0p5_l0p1_pmi20": ("NICME", "NICME (alpha=0.5, lambda=0.1)"),
}


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def selected_metric_rows(pmi20_per_seed: Path, alpha_per_seed: Path, seeds: tuple[int, ...]) -> list[dict[str, str]]:
    seed_set = {int(seed) for seed in seeds}
    rows = []
    for row in read_csv(pmi20_per_seed):
        if row.get("method") in {"ce_anchor_pmi20", "cost_sensitive_regularized_ce", "csada"} and int(row["seed"]) in seed_set:
            rows.append(row)
    for row in read_csv(alpha_per_seed):
        if row.get("method") == "nicme_extra_a0p5_l0p1_pmi20" and int(row["seed"]) in seed_set:
            rows.append(row)
    expected = {(seed, method) for seed in seed_set for method in SOURCE_METHODS}
    observed = {(int(row["seed"]), row["method"]) for row in rows}
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Missing metric rows for margin export: {missing}")
    return rows


def resolve_pair_ids(config: dict[str, Any], class_names: list[str]) -> list[tuple[int, int, float, str, str]]:
    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    pairs = []
    for pair in config.get("critical_pairs", []):
        true_value = pair.get("true_id", pair.get("true"))
        pred_value = pair.get("pred_id", pair.get("pred"))
        true_id = int(true_value) if str(true_value).isdigit() else label_to_id[str(true_value)]
        pred_id = int(pred_value) if str(pred_value).isdigit() else label_to_id[str(pred_value)]
        pairs.append((true_id, pred_id, float(pair.get("cost", 0.0)), class_names[true_id], class_names[pred_id]))
    return pairs


def compute_pair_margins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pairs: list[tuple[int, int, float, str, str]],
    start_index: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictions = torch.argmax(logits, dim=1)
    for local_idx, label in enumerate(labels.tolist()):
        for true_id, pred_id, cost, true_name, pred_name in pairs:
            if int(label) != int(true_id):
                continue
            margin = float(logits[local_idx, true_id].item() - logits[local_idx, pred_id].item())
            rows.append(
                {
                    "sample_index": start_index + local_idx,
                    "true_id": true_id,
                    "pred_id": pred_id,
                    "true_label": true_name,
                    "costly_pred_label": pred_name,
                    "critical_cost": cost,
                    "predicted_id": int(predictions[local_idx].item()),
                    "margin": margin,
                    "is_positive_margin": margin > 0.0,
                }
            )
    return rows


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def resolve_checkpoint_path(row: dict[str, str], config: dict[str, Any]) -> str:
    """Resolve a checkpoint path for older per-seed CSVs that omit it.

    Several live analysis CSVs intentionally keep only metric/config
    provenance. The original training config still contains the output
    directory, which is enough to recover the checkpoint parent.
    """
    if row.get("checkpoint_path"):
        return row["checkpoint_path"]
    output_dir = config.get("output_dir")
    if not output_dir:
        raise KeyError("checkpoint_path")
    checkpoint_root = Path(str(config.get("checkpoint_root", "checkpoints")))
    return str(checkpoint_root / str(output_dir))


def export_for_row(row: dict[str, str], batch_size: int, max_batches: int | None = None) -> list[dict[str, Any]]:
    config = load_config(row["config_path"])
    checkpoint_path = resolve_checkpoint_path(row, config)
    config["timm_pretrained"] = False
    script_args = SimpleNamespace(**config)
    _train_ds, _val_ds, _cal_ds, test_ds, class_names = preprocess_manifest_dataset(
        config["local_folder_path"],
        config["model"],
        script_args=script_args,
    )
    model = build_model(script_args, class_names=class_names)
    load_initial_checkpoint(model, checkpoint_path)
    device = get_device()
    model.to(device)
    model.eval()
    pairs = resolve_pair_ids(config, class_names)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    method_key, display_name = SOURCE_METHODS[row["method"]]
    out_rows: list[dict[str, Any]] = []
    sample_offset = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            labels = batch["labels"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits.detach().cpu()
            labels_cpu = labels.detach().cpu()
            margins = compute_pair_margins(logits, labels_cpu, pairs, start_index=sample_offset)
            for margin_row in margins:
                margin_row.update(
                    {
                        "seed": row["seed"],
                        "method": method_key,
                        "display_name": display_name,
                        "source_method": row["method"],
                        "checkpoint_path": checkpoint_path,
                        "metric_path": row.get("metric_path", ""),
                    }
                )
            out_rows.extend(margins)
            sample_offset += int(labels_cpu.numel())
    return out_rows


def export_margins(args: argparse.Namespace) -> list[dict[str, Any]]:
    metric_rows = selected_metric_rows(Path(args.pmi20_per_seed), Path(args.alpha_per_seed), args.seeds)
    all_rows: list[dict[str, Any]] = []
    for row in sorted(metric_rows, key=lambda item: (int(item["seed"]), item["method"])):
        all_rows.extend(export_for_row(row, batch_size=args.batch_size, max_batches=args.max_batches))
    fields = [
        "seed",
        "method",
        "display_name",
        "source_method",
        "sample_index",
        "true_id",
        "pred_id",
        "true_label",
        "costly_pred_label",
        "critical_cost",
        "predicted_id",
        "margin",
        "is_positive_margin",
        "checkpoint_path",
        "metric_path",
    ]
    write_csv(Path(args.output), all_rows, fields)
    return all_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--pmi20-per-seed", default=str(DEFAULT_PMI20_PER_SEED))
    parser.add_argument("--alpha-per-seed", default=str(DEFAULT_ALPHA_PER_SEED))
    parser.add_argument("--seeds", type=parse_seeds, default=pmi20.DEFAULT_SEEDS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args(argv)
    rows = export_margins(args)
    if not rows:
        raise RuntimeError("No critical-pair margin rows were exported")
    finite = [float(row["margin"]) for row in rows if not math.isnan(float(row["margin"]))]
    print(f"Wrote {len(rows)} margin rows to {args.output}; mean_margin={sum(finite) / len(finite):.4f}")


if __name__ == "__main__":
    main()
