import argparse
import csv
from pathlib import Path

import pytest

from scripts import build_paper_figures as builder


AGG_FIELDS = [
    "composite_rank",
    "method",
    "display_name",
    "decision_mode",
    "n_seeds",
    "seeds",
    "alpha",
    "cs_lambda",
    "loss_function",
    "target_recall_min_mean",
    "target_recall_min_std",
    "target_recall_macro_mean",
    "target_recall_macro_std",
    "normalized_atc_mean",
    "normalized_atc_std",
    "atc_mean",
    "atc_std",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "critical_pair_error_count_mean",
    "critical_pair_error_count_std",
    "critical_pair_error_count_total",
]


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def row(rank, method, display, recall, natc, alpha="", lam="", dataset_key="", dataset_display=""):
    item = {
        "composite_rank": rank,
        "method": method,
        "display_name": display,
        "decision_mode": "argmax",
        "n_seeds": 3,
        "seeds": "42,43,44",
        "alpha": alpha,
        "cs_lambda": lam,
        "loss_function": method,
        "target_recall_min_mean": recall,
        "target_recall_min_std": 0.01,
        "target_recall_macro_mean": recall,
        "target_recall_macro_std": 0.01,
        "normalized_atc_mean": natc,
        "normalized_atc_std": 0.001,
        "atc_mean": natc * 10,
        "atc_std": 0.01,
        "accuracy_mean": recall,
        "accuracy_std": 0.01,
        "balanced_accuracy_mean": recall,
        "balanced_accuracy_std": 0.01,
        "macro_f1_mean": recall,
        "macro_f1_std": 0.01,
        "critical_pair_error_count_mean": 1,
        "critical_pair_error_count_std": 0.5,
        "critical_pair_error_count_total": 3,
    }
    if dataset_key:
        item["dataset_key"] = dataset_key
        item["dataset_display"] = dataset_display
    return item


def make_inputs(tmp_path):
    pmi20 = tmp_path / "pmi20.csv"
    alpha = tmp_path / "alpha.csv"
    binary = tmp_path / "binary.csv"
    ablation = tmp_path / "ablation.csv"
    margins = tmp_path / "margins.csv"
    write_csv(
        pmi20,
        [
            row(1, "nicme_full", "NICME (alpha=0.5, lambda=0.1)", 0.92, 0.004, 0.5, 0.1),
            row(2, "cost_sensitive_regularized_ce", "cost-sensitive regularized CE", 0.91, 0.006),
            row(3, "csada", "CSADA", 0.88, 0.02),
        ],
        AGG_FIELDS,
    )
    write_csv(
        alpha,
        [
            row(1, "a", "NICME a", 0.92, 0.005, 0.5, 0.07),
            row(2, "b", "NICME b", 0.91, 0.004, 0.5, 0.1),
        ],
        AGG_FIELDS,
    )
    binary_fields = ["dataset_key", "dataset_display", *AGG_FIELDS]
    write_csv(
        binary,
        [
            row(1, "nicme", "NICME (alpha=0.5, lambda=0.1)", 0.9, 0.01, dataset_key="spider", dataset_display="Spider"),
            row(2, "ce", "CE", 0.8, 0.02, dataset_key="breakhis", dataset_display="BreaKHis"),
        ],
        binary_fields,
    )
    write_csv(
        ablation,
        [
            row(1, "ce", "CE", 0.8, 0.02, 0, 0),
            row(2, "nicme_regularizer_only_l0p1", "Regularizer-only", 0.85, 0.01, 0, 0.1),
            row(3, "nicme_margin_only_a0p5", "Margin-only", 0.86, 0.011, 0.5, 0),
            row(4, "nicme_full_a0p5_l0p1", "Full NICME", 0.9, 0.005, 0.5, 0.1),
        ],
        AGG_FIELDS,
    )
    write_csv(
        margins,
        [
            {"seed": 42, "method": "CE", "display_name": "CE", "margin": -0.1},
            {"seed": 42, "method": "NICME", "display_name": "NICME (alpha=0.5, lambda=0.1)", "margin": 1.0},
        ],
        ["seed", "method", "display_name", "margin"],
    )
    return pmi20, alpha, binary, ablation, margins


def build_args(tmp_path):
    pmi20, alpha, binary, ablation, margins = make_inputs(tmp_path)
    return argparse.Namespace(
        phase="all",
        output_root=str(tmp_path / "paper_figures"),
        pmi20_sota=str(pmi20),
        pmi20_alpha=str(alpha),
        binary_results=str(binary),
        ablation_results=str(ablation),
        margin_csv=str(margins),
        pmi20_camera_root=str(tmp_path / "empty_camera"),
        pmi20_alpha_root=str(tmp_path / "empty_alpha"),
        allow_missing_ablation=False,
        allow_missing_margins=False,
    )


def test_builder_refuses_missing_required_inputs(tmp_path):
    args = build_args(tmp_path)
    Path(args.ablation_results).unlink()
    with pytest.raises(FileNotFoundError):
        builder.ensure_inputs(args)


def test_builder_writes_tables_figures_and_summary(tmp_path):
    args = build_args(tmp_path)

    builder.ensure_inputs(args)
    builder.build_tables(args)
    builder.build_figures(args)
    builder.build_summary(args)

    output = Path(args.output_root)
    expected = [
        output / "tables" / "baselines_decision_modes.tex",
        output / "tables" / "pmi20_cost_sensitive_comparison.md",
        output / "tables" / "pmi20_component_ablation.csv",
        output / "tables" / "binary_cost_sensitive_results.tex",
        output / "tables" / "compute_implementation_cost.csv",
        output / "figures" / "pmi20_recall_cost_tradeoff.pdf",
        output / "figures" / "pmi20_alpha_lambda_sensitivity.png",
        output / "figures" / "pmi20_critical_pair_margin_ecdf.pdf",
        output / "README.md",
    ]
    for path in expected:
        assert path.exists()
        assert path.stat().st_size > 0
    combined_text = "\n".join(path.read_text(errors="ignore") for path in (output / "tables").glob("*.md"))
    assert "AP-CSADA" not in combined_text
    assert "cost-sensitive regularized CE" in combined_text
