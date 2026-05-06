"""Build paper-facing NICME tables and figures.

The builder consolidates completed experiment artifacts into a compact
`paper_figures/` package intended for manuscript use. It does not train models.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("paper_figures")
DEFAULT_PMI20_SOTA = Path("results/pmi20_nicme_sota_lr5e5_multiseed_20260504/analysis/aggregate_metrics.csv")
DEFAULT_PMI20_ALPHA = Path("results/pmi20_nicme_alpha_lambda6_lr5e5_multiseed_20260504/analysis/aggregate_metrics.csv")
DEFAULT_BINARY = Path("results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505/analysis/aggregate_metrics.csv")
DEFAULT_ABLATION = Path("results/pmi20_component_ablation_lr5e5_multiseed_20260506/analysis/aggregate_metrics.csv")
DEFAULT_MARGINS = Path("paper_figures/data/pmi20_critical_pair_margins.csv")

PMI20_CAMERA_ROOT = Path("results/pmi20_camera_ready_lr5e5_multiseed_20260504")
PMI20_ALPHA_ROOT = Path("results/pmi20_nicme_alpha_lambda6_lr5e5_multiseed_20260504")
BINARY_ROOT = Path("results/binary_camera_ready_cost8_7_lr5e5_multiseed_20260505")

METRIC_COLUMNS = (
    "target_recall_min",
    "target_recall_macro",
    "normalized_atc",
    "atc",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "critical_pair_error_count",
)

METHOD_SHORT = {
    "NICME (alpha=0.5, lambda=0.1)": "NICME",
    "cost-sensitive regularized CE": "CS-reg CE",
    "Menon logit adjustment": "Menon",
    "CE": "CE",
    "Cost-weighted CE": "Cost-wt CE",
    "CSADA": "CSADA",
    "CE + cost-min inference": "CE cost-min",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_pm(row: dict[str, Any], metric: str, precision: int = 4, tex: bool = False) -> str:
    mean = fnum(row.get(f"{metric}_mean"))
    std = fnum(row.get(f"{metric}_std"), 0.0)
    if math.isnan(mean):
        return ""
    sep = r" $\pm$ " if tex else " +/- "
    return f"{mean:.{precision}f}{sep}{std:.{precision}f}"


def critical_text(row: dict[str, Any], tex: bool = False) -> str:
    text = mean_pm(row, "critical_pair_error_count", 2, tex=tex)
    total = row.get("critical_pair_error_count_total", "")
    return f"{text} ({total})" if total not in {"", None} else text


def latex_escape(text: Any) -> str:
    s = str(text)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def markdown_table(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def tex_table(path: Path, headers: list[str], rows: list[list[Any]], alignment: str | None = None) -> None:
    alignment = alignment or ("l" * len(headers))
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(str(item) for item in headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(item) for item in row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def ensure_inputs(args: argparse.Namespace) -> None:
    required = [Path(args.pmi20_sota), Path(args.pmi20_alpha), Path(args.binary_results)]
    if not args.allow_missing_ablation:
        required.append(Path(args.ablation_results))
    if not args.allow_missing_margins:
        required.append(Path(args.margin_csv))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paper-figure inputs: " + ", ".join(missing))


def display_rows_for_main_table(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: int(float(row.get("composite_rank") or 999)))


def build_baselines_table(output_root: Path) -> None:
    rows = [
        {
            "method": "CE",
            "cost_training": "no",
            "cost_inference": "no",
            "class_frequency": "no",
            "adversarial_loop": "no",
            "special_output": "no",
            "decision_mode": "argmax",
        },
        {
            "method": "CE + cost-min inference",
            "cost_training": "no",
            "cost_inference": "yes",
            "class_frequency": "no",
            "adversarial_loop": "no",
            "special_output": "no",
            "decision_mode": "cost-min",
        },
        {
            "method": "Menon logit adjustment",
            "cost_training": "no",
            "cost_inference": "no",
            "class_frequency": "yes",
            "adversarial_loop": "no",
            "special_output": "no",
            "decision_mode": "argmax",
        },
        {
            "method": "Cost-weighted CE",
            "cost_training": "row mean",
            "cost_inference": "no",
            "class_frequency": "no",
            "adversarial_loop": "no",
            "special_output": "no",
            "decision_mode": "argmax",
        },
        {
            "method": "cost-sensitive regularized CE",
            "cost_training": "expected cost",
            "cost_inference": "no",
            "class_frequency": "no",
            "adversarial_loop": "no",
            "special_output": "no",
            "decision_mode": "argmax",
        },
        {
            "method": "CSADA",
            "cost_training": "target selection",
            "cost_inference": "no",
            "class_frequency": "no",
            "adversarial_loop": "yes",
            "special_output": "no",
            "decision_mode": "argmax",
        },
        {
            "method": "NICME",
            "cost_training": "pairwise margin + expected cost",
            "cost_inference": "no",
            "class_frequency": "no",
            "adversarial_loop": "no",
            "special_output": "no",
            "decision_mode": "argmax",
        },
    ]
    fields = list(rows[0])
    write_csv(output_root / "tables" / "baselines_decision_modes.csv", rows, fields)
    table_rows = [[row[field] for field in fields] for row in rows]
    headers = ["Method", "Costs in training", "Costs at inference", "Frequency", "Adversarial", "Special output", "Decision"]
    markdown_table(output_root / "tables" / "baselines_decision_modes.md", headers, table_rows)
    tex_table(output_root / "tables" / "baselines_decision_modes.tex", headers, table_rows, "lllllll")


def build_pmi20_table(output_root: Path, source: Path) -> None:
    rows = display_rows_for_main_table(read_csv(source))
    fields = [
        "rank",
        "method",
        "target_min_recall",
        "target_macro_recall",
        "normalized_atc",
        "atc",
        "balanced_accuracy",
        "macro_f1",
        "critical_errors",
    ]
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "rank": row.get("composite_rank", ""),
                "method": row.get("display_name", row.get("method", "")),
                "target_min_recall": mean_pm(row, "target_recall_min"),
                "target_macro_recall": mean_pm(row, "target_recall_macro"),
                "normalized_atc": mean_pm(row, "normalized_atc", 6),
                "atc": mean_pm(row, "atc", 6),
                "balanced_accuracy": mean_pm(row, "balanced_accuracy"),
                "macro_f1": mean_pm(row, "macro_f1"),
                "critical_errors": critical_text(row),
            }
        )
    write_csv(output_root / "tables" / "pmi20_cost_sensitive_comparison.csv", out_rows, fields)
    headers = ["Rank", "Method", "Target-min", "Target-macro", "nATC", "ATC", "Bal. Acc.", "Macro-F1", "Critical"]
    table_rows = [[row[field] for field in fields] for row in out_rows]
    markdown_table(output_root / "tables" / "pmi20_cost_sensitive_comparison.md", headers, table_rows)
    tex_table(output_root / "tables" / "pmi20_cost_sensitive_comparison.tex", headers, table_rows, "rllllllll")


def build_ablation_table(output_root: Path, source: Path) -> None:
    rows = read_csv(source)
    order = {"ce": 0, "nicme_regularizer_only_l0p1": 1, "nicme_margin_only_a0p5": 2, "nicme_full_a0p5_l0p1": 3}
    rows = sorted(rows, key=lambda row: order.get(row.get("method", ""), 999))
    fields = ["component", "alpha", "lambda", "target_min_recall", "normalized_atc", "balanced_accuracy", "macro_f1", "critical_errors"]
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "component": row.get("display_name", row.get("method", "")),
                "alpha": f"{fnum(row.get('alpha'), 0):g}",
                "lambda": f"{fnum(row.get('cs_lambda'), 0):g}",
                "target_min_recall": mean_pm(row, "target_recall_min"),
                "normalized_atc": mean_pm(row, "normalized_atc", 6),
                "balanced_accuracy": mean_pm(row, "balanced_accuracy"),
                "macro_f1": mean_pm(row, "macro_f1"),
                "critical_errors": critical_text(row),
            }
        )
    write_csv(output_root / "tables" / "pmi20_component_ablation.csv", out_rows, fields)
    headers = ["Component", "$\\alpha$", "$\\lambda$", "Target-min", "nATC", "Bal. Acc.", "Macro-F1", "Critical"]
    table_rows = [[row[field] for field in fields] for row in out_rows]
    markdown_table(output_root / "tables" / "pmi20_component_ablation.md", headers, table_rows)
    tex_table(output_root / "tables" / "pmi20_component_ablation.tex", headers, table_rows, "lrrlllll")


def build_binary_table(output_root: Path, source: Path) -> None:
    rows = sorted(read_csv(source), key=lambda row: (row.get("dataset_key", ""), int(float(row.get("composite_rank") or 999))))
    fields = ["dataset", "rank", "method", "target_recall", "normalized_atc", "atc", "balanced_accuracy", "macro_f1", "critical_errors"]
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "dataset": row.get("dataset_display", row.get("dataset_key", "")),
                "rank": row.get("composite_rank", ""),
                "method": row.get("display_name", row.get("method", "")),
                "target_recall": mean_pm(row, "target_recall_min"),
                "normalized_atc": mean_pm(row, "normalized_atc", 6),
                "atc": mean_pm(row, "atc", 6),
                "balanced_accuracy": mean_pm(row, "balanced_accuracy"),
                "macro_f1": mean_pm(row, "macro_f1"),
                "critical_errors": critical_text(row),
            }
        )
    write_csv(output_root / "tables" / "binary_cost_sensitive_results.csv", out_rows, fields)
    headers = ["Dataset", "Rank", "Method", "Target recall", "nATC", "ATC", "Bal. Acc.", "Macro-F1", "Critical"]
    table_rows = [[row[field] for field in fields] for row in out_rows]
    markdown_table(output_root / "tables" / "binary_cost_sensitive_results.md", headers, table_rows)
    tex_table(output_root / "tables" / "binary_cost_sensitive_results.tex", headers, table_rows, "lrlllllll")


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (1024 * 1024)


def read_ledgers(root: Path, glob_pattern: str = "**/run/run_ledger.csv") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in root.glob(glob_pattern):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_ledger_path"] = str(path)
                rows.append(row)
    return rows


def build_compute_table(output_root: Path, pmi20_camera_root: Path = PMI20_CAMERA_ROOT, pmi20_alpha_root: Path = PMI20_ALPHA_ROOT) -> None:
    camera_rows = read_ledgers(pmi20_camera_root)
    alpha_rows = read_ledgers(pmi20_alpha_root)
    method_sources = {
        "CE": [row for row in camera_rows if row.get("method") == "ce_anchor_pmi20"],
        "cost-sensitive regularized CE": [row for row in camera_rows if row.get("method") == "cost_sensitive_regularized_ce"],
        "CSADA": [row for row in camera_rows if row.get("method") == "csada"],
        "NICME": [row for row in alpha_rows if row.get("method") == "nicme_extra_a0p5_l0p1_pmi20"],
    }
    ce_mean = 1.0
    out_rows = []
    for method, rows in method_sources.items():
        elapsed = [fnum(row.get("elapsed_seconds")) for row in rows if not math.isnan(fnum(row.get("elapsed_seconds")))]
        mean_seconds = sum(elapsed) / len(elapsed) if elapsed else 0.0
        if method == "CE" and mean_seconds > 0:
            ce_mean = mean_seconds
        ckpt_mb = sum(directory_size_mb(Path(row.get("checkpoint_path", ""))) for row in rows) / len(rows) if rows else 0.0
        out_rows.append(
            {
                "method": method,
                "training_cost_use": {
                    "CE": "none",
                    "cost-sensitive regularized CE": "expected-cost regularizer",
                    "CSADA": "cost-targeted adversarial augmentation",
                    "NICME": "pairwise margin + expected-cost regularizer",
                }[method],
                "adversarial_inner_loop": "yes" if method == "CSADA" else "no",
                "mean_wall_time_min": f"{mean_seconds / 60:.2f}",
                "relative_to_ce": f"{(mean_seconds / ce_mean) if ce_mean else 0:.2f}",
                "mean_checkpoint_mb": f"{ckpt_mb:.1f}",
                "n_runs": len(rows),
            }
        )
    fields = ["method", "training_cost_use", "adversarial_inner_loop", "mean_wall_time_min", "relative_to_ce", "mean_checkpoint_mb", "n_runs"]
    write_csv(output_root / "tables" / "compute_implementation_cost.csv", out_rows, fields)
    headers = ["Method", "Cost use", "Adv. loop", "Wall-time min", "Rel. CE", "Checkpoint MB", "Runs"]
    table_rows = [[row[field] for field in fields] for row in out_rows]
    markdown_table(output_root / "tables" / "compute_implementation_cost.md", headers, table_rows)
    tex_table(output_root / "tables" / "compute_implementation_cost.tex", headers, table_rows, "llllrrr")


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def save_figure(fig, output_root: Path, name: str) -> None:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")


def method_label(row: dict[str, str]) -> str:
    return METHOD_SHORT.get(row.get("display_name", ""), row.get("display_name", row.get("method", "")))


def build_recall_cost_tradeoff(output_root: Path, source: Path) -> None:
    plt = setup_matplotlib()
    rows = display_rows_for_main_table(read_csv(source))
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True)
    for idx, row in enumerate(rows):
        x = fnum(row.get("normalized_atc_mean"))
        y1 = fnum(row.get("target_recall_min_mean"))
        y2 = fnum(row.get("target_recall_macro_mean"))
        crit = fnum(row.get("critical_pair_error_count_total"), 0.0)
        size = 45 + 20 * crit
        label = method_label(row)
        for ax, y, title in ((axes[0], y1, "Target-min recall"), (axes[1], y2, "Target-macro recall")):
            ax.scatter(x, y, s=size, color=colors(idx % 10), alpha=0.82, edgecolor="black", linewidth=0.35)
            ax.annotate(label, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
            ax.set_title(title)
            ax.set_xlabel("Normalized ATC (lower is better)")
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    axes[0].set_ylabel("Recall (higher is better)")
    save_figure(fig, output_root, "pmi20_recall_cost_tradeoff")
    plt.close(fig)


def build_alpha_lambda_plot(output_root: Path, source: Path) -> None:
    plt = setup_matplotlib()
    rows = read_csv(source)
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    for row in rows:
        x = fnum(row.get("normalized_atc_mean"))
        y = fnum(row.get("target_recall_min_mean"))
        crit = fnum(row.get("critical_pair_error_count_total"), 0.0)
        ax.scatter(x, y, s=55 + 22 * crit, color="#0072B2", alpha=0.78, edgecolor="black", linewidth=0.35)
        label = f"({fnum(row.get('alpha'), 0):g}, {fnum(row.get('cs_lambda'), 0):g})"
        ax.annotate(label, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Normalized ATC (lower is better)")
    ax.set_ylabel("Target-min recall (higher is better)")
    ax.set_title("PMI-20 NICME sensitivity")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    save_figure(fig, output_root, "pmi20_alpha_lambda_sensitivity")
    plt.close(fig)


def build_margin_ecdf(output_root: Path, source: Path) -> None:
    plt = setup_matplotlib()
    rows = read_csv(source)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row.get("display_name", row.get("method", "")), []).append(fnum(row.get("margin")))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    colors = plt.get_cmap("tab10")
    for idx, (name, values) in enumerate(sorted(grouped.items())):
        vals = sorted(value for value in values if not math.isnan(value))
        if not vals:
            continue
        ys = [(i + 1) / len(vals) for i in range(len(vals))]
        ax.step(vals, ys, where="post", label=METHOD_SHORT.get(name, name), color=colors(idx % 10), linewidth=1.4)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"Critical-pair raw margin $f_y(x)-f_k(x)$")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("PMI-20 critical-pair margins")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.7)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, output_root, "pmi20_critical_pair_margin_ecdf")
    plt.close(fig)


def build_result_inventory(output_root: Path, args: argparse.Namespace) -> None:
    rows = [
        {"artifact": "PMI-20 SOTA aggregate", "path": args.pmi20_sota},
        {"artifact": "PMI-20 alpha/lambda aggregate", "path": args.pmi20_alpha},
        {"artifact": "Binary aggregate", "path": args.binary_results},
        {"artifact": "PMI-20 component ablation aggregate", "path": args.ablation_results},
        {"artifact": "PMI-20 critical-pair margins", "path": args.margin_csv},
    ]
    for row in rows:
        row["exists"] = str(Path(row["path"]).exists()).lower()
    write_csv(output_root / "data" / "result_inventory.csv", rows, ["artifact", "path", "exists"])


def build_tables(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    build_baselines_table(output_root)
    build_pmi20_table(output_root, Path(args.pmi20_sota))
    build_ablation_table(output_root, Path(args.ablation_results))
    build_binary_table(output_root, Path(args.binary_results))
    build_compute_table(output_root, Path(args.pmi20_camera_root), Path(args.pmi20_alpha_root))
    build_result_inventory(output_root, args)


def build_figures(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    build_recall_cost_tradeoff(output_root, Path(args.pmi20_sota))
    build_alpha_lambda_plot(output_root, Path(args.pmi20_alpha))
    build_margin_ecdf(output_root, Path(args.margin_csv))


def build_summary(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    lines = [
        "# NICME Paper Figures And Tables",
        "",
        "This folder contains camera-ready table and figure assets generated from completed NICME result ledgers.",
        "",
        "## Captions",
        "",
        "**Table: Baselines and decision modes.** Methods differ in where cost information enters the pipeline: training loss, inference decision rule, class-frequency correction, adversarial inner loop, or output parameterization.",
        "",
        "**Table: Main PMI-20 cost-sensitive comparison.** Mean +/- sample standard deviation over three seeds on the balanced PMI-20 split; rankings follow target-min recall, normalized ATC, balanced accuracy, and macro-F1.",
        "",
        "**Table: NICME component ablation.** CE, expected-cost regularization, pairwise cost margins, and full NICME are compared under the same ConvNeXt-base LR 5e-5 protocol.",
        "",
        "**Table: Binary cost-sensitive results.** Spider and BreaKHis comparisons use evidence-derived integer matrices, reporting both recall-first cost-sensitive metrics and clean balanced accuracy.",
        "",
        "**Table: Computational and implementation cost.** Wall-time and storage estimates are taken from completed run ledgers and checkpoint directories; CSADA is the only method with an adversarial inner loop.",
        "",
        "**Figure: PMI-20 recall-cost tradeoff.** Each point is a method; lower normalized ATC and higher target recall are preferred, and point size reflects total critical-pair errors.",
        "",
        "**Figure: PMI-20 alpha-lambda sensitivity.** Six fixed NICME candidates show how expected cost and cared-class recall change with alpha/lambda.",
        "",
        "**Figure: Critical-pair logit margin distributions.** ECDFs of raw margins `f_y(x)-f_k(x)` on critical pairs; right-shifted curves indicate larger clean margins against high-cost confusions.",
        "",
        "## Provenance",
        "",
        "- See `data/result_inventory.csv` for source paths.",
        "- Metrics are repository SOTA/baseline comparisons, not external global SOTA claims.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["preflight", "tables", "figures", "summary", "all"], required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pmi20-sota", default=str(DEFAULT_PMI20_SOTA))
    parser.add_argument("--pmi20-alpha", default=str(DEFAULT_PMI20_ALPHA))
    parser.add_argument("--binary-results", default=str(DEFAULT_BINARY))
    parser.add_argument("--ablation-results", default=str(DEFAULT_ABLATION))
    parser.add_argument("--margin-csv", default=str(DEFAULT_MARGINS))
    parser.add_argument("--pmi20-camera-root", default=str(PMI20_CAMERA_ROOT))
    parser.add_argument("--pmi20-alpha-root", default=str(PMI20_ALPHA_ROOT))
    parser.add_argument("--allow-missing-ablation", action="store_true")
    parser.add_argument("--allow-missing-margins", action="store_true")
    args = parser.parse_args(argv)

    if args.phase == "preflight":
        ensure_inputs(args)
        print("Paper figure input preflight passed")
    elif args.phase == "tables":
        ensure_inputs(args)
        build_tables(args)
    elif args.phase == "figures":
        ensure_inputs(args)
        build_figures(args)
    elif args.phase == "summary":
        build_summary(args)
    elif args.phase == "all":
        ensure_inputs(args)
        build_tables(args)
        build_figures(args)
        build_summary(args)
        print(f"Wrote paper figures under {Path(args.output_root)}")


if __name__ == "__main__":
    main()
