"""3-way comparison of cost-sensitive sweep results.

Reads sweep_results.csv from three pipelines (Parent, Playground, Hybrid)
and generates comparison tables, overlaid line plots, a stability analysis,
and a comprehensive markdown report.

Pipelines compared:
  1. Parent:      CELogitAdjustmentV2 (logit adjustment only)
  2. Playground:  CostSensitiveRegularizedLoss (CE + lambda * CS)
  3. Hybrid:      CELogitAdjustmentRegularized (logit adj + CS reg + M-norm + warmup)

Usage:
    micromamba activate ml
    python scripts/compare_sweeps.py
    python scripts/compare_sweeps.py --output-dir results/my_comparison
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from datetime import datetime

import duckdb
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPELINE_KEYS = ["parent", "playground", "hybrid"]
PIPELINE_LABELS = {
    "parent": "Parent (LogitAdj)",
    "playground": "Playground (CE+CS)",
    "hybrid": "Hybrid (LogitAdj+CS)",
}
PIPELINE_COLORS = {
    "parent": "#636EFA",
    "playground": "#EF553B",
    "hybrid": "#00CC96",
}

# Columns guaranteed across all 3 CSVs
SHARED_COLUMNS = [
    "cost_value",
    "accuracy",
    "f1_score",
    "class_0_precision",
    "class_0_recall",
    "class_0_f1_score",
    "class_0_false_positive_rate",
    "class_0_false_negative_rate",
    "class_1_precision",
    "class_1_recall",
    "class_1_f1_score",
    "class_1_false_positive_rate",
    "class_1_false_negative_rate",
    "cm_0_0",
    "cm_0_1",
    "cm_1_0",
    "cm_1_1",
]

# Columns only in playground/hybrid (28-col schema)
EXTENDED_COLUMNS = [
    "balanced_accuracy",
    "kappa",
    "auc",
    "expected_cost",
]

# Metrics used for main comparison visualisations
COMPARISON_METRICS = [
    "class_0_recall",
    "accuracy",
    "f1_score",
    "class_1_recall",
    "class_0_precision",
    "class_1_precision",
]

# Cost values at which to show confusion matrices
CM_COST_VALUES = [1, 10, 50, 100]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-way sweep comparison: Parent vs Playground vs Hybrid")
    p.add_argument(
        "--parent",
        default="results/sweep_cost_0_1/sweep_results.csv",
        help="Path to parent sweep CSV",
    )
    p.add_argument(
        "--playground",
        default="playground/cost_sensitive_loss_classification/results/sweep_cost_ratio/sweep_results.csv",
        help="Path to playground sweep CSV",
    )
    p.add_argument(
        "--hybrid",
        default="results/sweep_cost_0_1_reg/sweep_results.csv",
        help="Path to hybrid sweep CSV",
    )
    p.add_argument(
        "--output-dir",
        default="results/sweep_comparison",
        help="Output directory for comparison artifacts",
    )
    p.add_argument(
        "--cost-values",
        default=None,
        help="Comma-separated subset of cost values to compare (default: all overlapping)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading & alignment
# ---------------------------------------------------------------------------


def _read_csv_safe(path: str, label: str) -> pl.DataFrame | None:
    """Read a CSV, returning None if the file does not exist."""
    if not os.path.exists(path):
        print(f"  WARNING: {label} CSV not found at {path} — skipping")
        return None
    df = pl.read_csv(path)
    print(f"  {label}: {df.shape[0]} rows, {df.shape[1]} cols — {path}")
    return df


def _ensure_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Add any missing columns as null Float64."""
    for col in columns:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    return df


def load_and_align_data(
    parent_path: str,
    playground_path: str,
    hybrid_path: str,
    cost_values: list[float] | None = None,
) -> dict[str, pl.DataFrame | None]:
    """Load all three CSVs and align to overlapping cost values.

    Returns a dict with keys 'parent', 'playground', 'hybrid' (None if missing),
    plus 'overlap_values' (sorted list of shared cost values).
    """
    print("Loading sweep CSVs...")
    raw = {
        "parent": _read_csv_safe(parent_path, "Parent"),
        "playground": _read_csv_safe(playground_path, "Playground"),
        "hybrid": _read_csv_safe(hybrid_path, "Hybrid"),
    }

    # Find overlapping cost values
    available = {k: v for k, v in raw.items() if v is not None}
    if not available:
        raise FileNotFoundError("No sweep CSVs found. Cannot generate comparison.")

    cost_sets = [set(df["cost_value"].cast(pl.Float64).to_list()) for df in available.values()]
    overlap = sorted(cost_sets[0].intersection(*cost_sets[1:]))

    if cost_values is not None:
        overlap = sorted(set(cost_values) & set(overlap))

    print(f"  Overlapping cost values ({len(overlap)}): {overlap}")

    # All columns we want in the aligned DataFrames
    all_columns = SHARED_COLUMNS + EXTENDED_COLUMNS

    result: dict[str, pl.DataFrame | None] = {"overlap_values": overlap}

    for key in PIPELINE_KEYS:
        df = raw[key]
        if df is None:
            result[key] = None
            continue

        # Filter to overlapping cost values
        df = df.filter(pl.col("cost_value").cast(pl.Float64).is_in(overlap))
        df = df.sort("cost_value")

        # Ensure all columns exist
        df = _ensure_columns(df, all_columns)

        # Derive balanced_accuracy if missing (exact: mean of per-class recalls)
        if df["balanced_accuracy"].is_null().all() and "class_0_recall" in df.columns:
            df = df.with_columns(
                ((pl.col("class_0_recall") + pl.col("class_1_recall")) / 2.0).alias("balanced_accuracy")
            )

        result[key] = df

    return result


# ---------------------------------------------------------------------------
# Wide comparison & stability
# ---------------------------------------------------------------------------


def build_wide_comparison(data: dict, metrics: list[str]) -> pl.DataFrame:
    """Build a single wide DataFrame: cost_value + metric_parent/playground/hybrid + spread."""
    # Start with cost_value from any available pipeline
    for key in PIPELINE_KEYS:
        if data[key] is not None:
            base = data[key].select("cost_value").sort("cost_value")
            break

    wide = base.clone()

    for metric in metrics:
        for key in PIPELINE_KEYS:
            col_name = f"{metric}_{key}"
            df = data[key]
            if df is not None and metric in df.columns:
                rename_df = df.select(
                    [
                        pl.col("cost_value"),
                        pl.col(metric).alias(col_name),
                    ]
                )
                wide = wide.join(rename_df, on="cost_value", how="left")
            else:
                wide = wide.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

        # Compute spread
        suffix_cols = [f"{metric}_{k}" for k in PIPELINE_KEYS]
        existing = [c for c in suffix_cols if c in wide.columns]
        if len(existing) >= 2:
            wide = wide.with_columns(
                (pl.max_horizontal(existing) - pl.min_horizontal(existing)).alias(f"{metric}_spread")
            )

    return wide


def compute_stability(data: dict, metrics: list[str]) -> pl.DataFrame:
    """Compute per-pipeline metric spread (max - min across cost values)."""
    rows = []
    for metric in metrics:
        row = {"metric": metric}
        for key in PIPELINE_KEYS:
            df = data[key]
            if df is not None and metric in df.columns:
                col = df[metric].drop_nulls()
                if len(col) > 0:
                    row[f"{key}_spread"] = float(col.max() - col.min())
                else:
                    row[f"{key}_spread"] = None
            else:
                row[f"{key}_spread"] = None
        rows.append(row)

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotly visualizations
# ---------------------------------------------------------------------------


def _save_fig(fig: go.Figure, graphs_dir: str, name: str) -> None:
    fig.write_html(os.path.join(graphs_dir, f"{name}.html"))
    try:
        fig.write_image(os.path.join(graphs_dir, f"{name}.png"), width=1200, height=600, scale=2)
    except (ValueError, ImportError, OSError):
        print(f"    PNG export skipped for {name} (kaleido not available)")


def plot_metric_comparison(
    data: dict,
    metric: str,
    title: str,
    y_label: str,
    graphs_dir: str,
    filename: str,
) -> None:
    """Overlaid line chart of one metric across 3 pipelines."""
    fig = go.Figure()
    for key in PIPELINE_KEYS:
        df = data[key]
        if df is None or metric not in df.columns:
            continue
        col = df[metric]
        if col.is_null().all():
            continue
        fig.add_trace(
            go.Scatter(
                x=df["cost_value"].to_list(),
                y=col.to_list(),
                mode="lines+markers",
                name=PIPELINE_LABELS[key],
                line={"color": PIPELINE_COLORS[key]},
                marker={"size": 6},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Cost Value (M[0][1])",
        yaxis_title=y_label,
        template="plotly_white",
        legend={"yanchor": "bottom", "y": 0.01, "xanchor": "right", "x": 0.99},
    )
    _save_fig(fig, graphs_dir, filename)
    print(f"    {filename}")


def plot_dashboard(data: dict, graphs_dir: str) -> None:
    """2x3 subplot combining key metrics."""
    subplot_specs = [
        ("class_0_recall", "Black Widow Recall"),
        ("accuracy", "Accuracy"),
        ("class_1_recall", "False Widow Recall"),
        ("f1_score", "F1 Score"),
        ("auc", "AUC"),
        ("expected_cost", "Expected Cost"),
    ]

    fig = make_subplots(rows=2, cols=3, subplot_titles=[s[1] for s in subplot_specs])

    for idx, (metric, _title) in enumerate(subplot_specs):
        row = idx // 3 + 1
        col = idx % 3 + 1
        has_data = False

        for key in PIPELINE_KEYS:
            df = data[key]
            if df is None or metric not in df.columns:
                continue
            series = df[metric]
            if series.is_null().all():
                continue
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=df["cost_value"].to_list(),
                    y=series.to_list(),
                    mode="lines+markers",
                    name=PIPELINE_LABELS[key],
                    line={"color": PIPELINE_COLORS[key]},
                    marker={"size": 4},
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )

        if not has_data:
            fig.add_annotation(
                text="N/A",
                xref=f"x{idx + 1}" if idx > 0 else "x",
                yref=f"y{idx + 1}" if idx > 0 else "y",
                x=0.5,
                y=0.5,
                xanchor="center",
                showarrow=False,
                font={"size": 18, "color": "gray"},
                row=row,
                col=col,
            )

    fig.update_layout(
        height=800,
        width=1500,
        title_text="3-Way Sweep Comparison Dashboard",
        template="plotly_white",
        showlegend=True,
    )
    _save_fig(fig, graphs_dir, "dashboard")
    print("    dashboard")


def plot_stability_bar(stability_df: pl.DataFrame, graphs_dir: str) -> None:
    """Grouped bar chart: metric spread per pipeline."""
    fig = go.Figure()
    metrics = stability_df["metric"].to_list()

    for key in PIPELINE_KEYS:
        col_name = f"{key}_spread"
        if col_name not in stability_df.columns:
            continue
        values = stability_df[col_name].to_list()
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=values,
                name=PIPELINE_LABELS[key],
                marker_color=PIPELINE_COLORS[key],
            )
        )

    fig.update_layout(
        title="Metric Stability Across Cost Values (Lower = More Stable)",
        xaxis_title="Metric",
        yaxis_title="Spread (max − min)",
        barmode="group",
        template="plotly_white",
    )
    _save_fig(fig, graphs_dir, "stability_bar")
    print("    stability_bar")


# ---------------------------------------------------------------------------
# DuckDB export
# ---------------------------------------------------------------------------


def export_to_duckdb(
    wide_df: pl.DataFrame,
    stability_df: pl.DataFrame,
    data: dict,
    output_dir: str,
) -> None:
    db_path = os.path.join(output_dir, "sweep_comparison.duckdb")
    con = duckdb.connect(db_path)

    tables = [("comparison_wide", wide_df), ("stability", stability_df)]
    for key in PIPELINE_KEYS:
        if data[key] is not None:
            tables.append((f"{key}_raw", data[key]))

    for name, table_data in tables:
        con.register("_tmp", table_data)
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _tmp")
        con.unregister("_tmp")

    con.close()
    print(f"  DuckDB: {db_path} ({len(tables)} tables)")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt(val, fmt=".4f") -> str:
    if val is None:
        return "N/A"
    return f"{val:{fmt}}"


def _fmt_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _get_val(df: pl.DataFrame | None, cost: float, col: str):
    """Extract a single metric value at a given cost_value."""
    if df is None or col not in df.columns:
        return None
    row = df.filter(pl.col("cost_value").cast(pl.Float64) == float(cost))
    if row.is_empty():
        return None
    val = row[col][0]
    return val


def _cm_block(df: pl.DataFrame | None, cost: float, label: str) -> str:
    """Format a 2x2 confusion matrix at a given cost_value."""
    cm00 = _get_val(df, cost, "cm_0_0")
    cm01 = _get_val(df, cost, "cm_0_1")
    cm10 = _get_val(df, cost, "cm_1_0")
    cm11 = _get_val(df, cost, "cm_1_1")
    if cm00 is None:
        return f"**{label}**: N/A\n"
    cm00, cm01, cm10, cm11 = int(cm00), int(cm01), int(cm10), int(cm11)
    return (
        f"**{label}**:\n"
        f"```\n"
        f"            Pred BW  Pred FW\n"
        f"True BW      {cm00:>5}    {cm01:>5}\n"
        f"True FW      {cm10:>5}    {cm11:>5}\n"
        f"```\n"
    )


def _find_best(df: pl.DataFrame | None, metric: str, maximize: bool = True):
    """Find the cost_value that optimises a metric. Returns (best_val, cost_at_best)."""
    if df is None or metric not in df.columns:
        return None, None
    col = df.select(["cost_value", metric]).drop_nulls()
    if col.is_empty():
        return None, None
    if maximize:
        idx = col[metric].arg_max()
    else:
        idx = col[metric].arg_min()
    best_val = col[metric][idx]
    best_cost = col["cost_value"][idx]
    return best_val, best_cost


def generate_markdown_report(data: dict, wide_df: pl.DataFrame, stability_df: pl.DataFrame, output_dir: str) -> None:
    overlap = data["overlap_values"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines.append("# Cost-Sensitive Sweep Comparison: 3-Way Analysis\n")
    lines.append(f"**Generated**: {now}\n")

    # --- Section 1: Overview ---
    lines.append("## 1. Overview\n")
    lines.append(
        "Three cost-sensitive sweep pipelines compared on the binary spider classification task "
        "(Black Widow = class 0, False Widow = class 1), all sweeping the M[0][1] cost matrix cell.\n"
    )
    lines.append(f"**Overlapping cost values** ({len(overlap)}): {', '.join(str(int(v)) for v in overlap)}\n")

    for key in PIPELINE_KEYS:
        df = data[key]
        if df is None:
            lines.append(f"- **{PIPELINE_LABELS[key]}**: Not available\n")
        else:
            n = df.shape[0]
            lines.append(f"- **{PIPELINE_LABELS[key]}**: {n} cost values\n")

    # --- Section 2: Setup Differences ---
    lines.append("\n## 2. Setup Differences\n")
    lines.append("| Property | Parent | Playground | Hybrid |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| Loss function | CELogitAdjustmentV2 | CostSensitiveRegularizedLoss | CELogitAdjustmentRegularized |"
    )
    lines.append("| Framework | HuggingFace Trainer | Custom PyTorch loop | HuggingFace Trainer |")
    lines.append("| CS lambda | N/A | 10.0 | 10.0 |")
    lines.append("| CS warmup | N/A | N/A | 5 epochs |")
    lines.append("| M normalization | No | Implicit (softmax) | Explicit (M/max(M)) |")
    lines.append("| Test set size | 300 (150/150) | 449 (225/224) | 300 (150/150) |")
    lines.append("| Batch size | 32 | 16 | 32 |")
    lines.append("| Weight decay | 0.01 | 0.0 | 0.01 |")
    lines.append("| Frozen stages | 3 | 0 | 3 |")
    lines.append("| Optimizer | AdamW | Adam | AdamW |")
    lines.append("| Max epochs | 30 | 30 | 30 |")
    lines.append("| Early stopping | 5 (eval_loss) | 5 (AUC) | 5 (eval_loss) |")
    lines.append("")

    # --- Section 3: Side-by-Side Results ---
    lines.append("## 3. Side-by-Side Results\n")
    lines.append("Abbreviations: **P** = Parent, **PG** = Playground, **H** = Hybrid\n")

    for metric, metric_label in [
        ("class_0_recall", "Class 0 Recall (Black Widow)"),
        ("accuracy", "Accuracy"),
        ("f1_score", "F1 Score"),
        ("class_1_recall", "Class 1 Recall (False Widow)"),
    ]:
        lines.append(
            f"### 3.{['class_0_recall', 'accuracy', 'f1_score', 'class_1_recall'].index(metric) + 1} {metric_label}\n"
        )
        lines.append("| Cost | P | PG | H | Spread |")
        lines.append("|------|---|---|---|--------|")
        for cost in overlap:
            p_val = _get_val(data["parent"], cost, metric)
            pg_val = _get_val(data["playground"], cost, metric)
            h_val = _get_val(data["hybrid"], cost, metric)
            vals = [v for v in [p_val, pg_val, h_val] if v is not None]
            spread = max(vals) - min(vals) if len(vals) >= 2 else None
            lines.append(f"| {int(cost)} | {_fmt(p_val)} | {_fmt(pg_val)} | {_fmt(h_val)} | {_fmt(spread)} |")
        lines.append("")

    # --- Section 4: Key Findings ---
    lines.append("## 4. Key Findings\n")

    # 4.1 Baseline (cost=1)
    lines.append("### 4.1 Baseline Comparison (cost=1)\n")
    lines.append("| Metric | Parent | Playground | Hybrid |")
    lines.append("|---|---|---|---|")
    for metric in ["accuracy", "class_0_recall", "class_1_recall", "f1_score"]:
        p = _get_val(data["parent"], 1, metric)
        pg = _get_val(data["playground"], 1, metric)
        h = _get_val(data["hybrid"], 1, metric)
        lines.append(f"| {metric} | {_fmt(p)} | {_fmt(pg)} | {_fmt(h)} |")
    lines.append("")

    # 4.2 Best class_0_recall per pipeline
    lines.append("### 4.2 Best Class 0 Recall per Pipeline\n")
    lines.append("| Pipeline | Best C0 Recall | At Cost | Accuracy at that Cost |")
    lines.append("|---|---|---|---|")
    for key in PIPELINE_KEYS:
        best_r, best_c = _find_best(data[key], "class_0_recall")
        acc_at_best = _get_val(data[key], best_c, "accuracy") if best_c is not None else None
        lines.append(f"| {PIPELINE_LABELS[key]} | {_fmt(best_r)} | {_fmt(best_c, '.0f')} | {_fmt(acc_at_best)} |")
    lines.append("")

    # 4.3 Collapse detection
    lines.append("### 4.3 Collapse Detection\n")
    lines.append("A pipeline is considered collapsed if accuracy falls below 55% at any cost value.\n")
    for key in PIPELINE_KEYS:
        df = data[key]
        if df is None:
            continue
        min_acc = df["accuracy"].min()
        if min_acc is not None and min_acc < 0.55:
            collapse_rows = df.filter(pl.col("accuracy") < 0.55)
            costs = collapse_rows["cost_value"].to_list()
            lines.append(
                f"- **{PIPELINE_LABELS[key]}**: COLLAPSED at cost(s) "
                f"{', '.join(str(int(c)) for c in costs)} (min accuracy: {_fmt_pct(min_acc)})"
            )
        else:
            lines.append(f"- **{PIPELINE_LABELS[key]}**: No collapse (min accuracy: {_fmt_pct(min_acc)})")
    lines.append("")

    # --- Section 5: Stability Analysis ---
    lines.append("## 5. Stability Analysis\n")
    lines.append("Spread = max(metric) - min(metric) across all cost values. Lower is more stable.\n")
    lines.append("| Metric | Parent | Playground | Hybrid | Most Stable |")
    lines.append("|---|---|---|---|---|")
    for row in stability_df.iter_rows(named=True):
        metric = row["metric"]
        spreads = {}
        for key in PIPELINE_KEYS:
            val = row.get(f"{key}_spread")
            spreads[key] = val
        most_stable = min(
            (k for k in PIPELINE_KEYS if spreads.get(k) is not None),
            key=lambda k: spreads[k],
            default="N/A",
        )
        most_stable_label = PIPELINE_LABELS.get(most_stable, "N/A")
        lines.append(
            f"| {metric} | {_fmt(spreads.get('parent'))} "
            f"| {_fmt(spreads.get('playground'))} "
            f"| {_fmt(spreads.get('hybrid'))} "
            f"| {most_stable_label} |"
        )
    lines.append("")

    # --- Section 6: Confusion Matrices ---
    lines.append("## 6. Confusion Matrix at Selected Cost Values\n")
    lines.append("Note: Parent/Hybrid test set = 300 samples (150/150). Playground = 449 (225/224).\n")
    for cost in CM_COST_VALUES:
        if cost not in overlap:
            continue
        lines.append(f"### cost = {int(cost)}\n")
        for key in PIPELINE_KEYS:
            lines.append(_cm_block(data[key], cost, PIPELINE_LABELS[key]))
        lines.append("")

    # --- Section 7: Best Operating Points ---
    lines.append("## 7. Best Operating Points\n")
    lines.append("| Pipeline | Best C0 Recall | At Cost | Best Accuracy | At Cost | Best F1 | At Cost |")
    lines.append("|---|---|---|---|---|---|---|")
    for key in PIPELINE_KEYS:
        r_val, r_cost = _find_best(data[key], "class_0_recall")
        a_val, a_cost = _find_best(data[key], "accuracy")
        f_val, f_cost = _find_best(data[key], "f1_score")
        lines.append(
            f"| {PIPELINE_LABELS[key]} "
            f"| {_fmt(r_val)} | {_fmt(r_cost, '.0f')} "
            f"| {_fmt(a_val)} | {_fmt(a_cost, '.0f')} "
            f"| {_fmt(f_val)} | {_fmt(f_cost, '.0f')} |"
        )
    lines.append("")

    # --- Section 8: Conclusions ---
    lines.append("## 8. Conclusions\n")

    # Programmatic summary
    best_stable = None
    for key in PIPELINE_KEYS:
        df = data[key]
        if df is None:
            continue
        min_acc = df["accuracy"].min()
        if min_acc is not None and (best_stable is None or min_acc > best_stable[1]):
            best_stable = (key, min_acc)

    if best_stable:
        lines.append(
            f"1. **Most stable pipeline**: {PIPELINE_LABELS[best_stable[0]]} "
            f"(minimum accuracy {_fmt_pct(best_stable[1])} across all cost values)."
        )

    # Best recall
    best_recall_pipeline = None
    best_recall_val = 0
    for key in PIPELINE_KEYS:
        val, _ = _find_best(data[key], "class_0_recall")
        if val is not None and val > best_recall_val:
            best_recall_val = val
            best_recall_pipeline = key
    if best_recall_pipeline:
        lines.append(
            f"2. **Highest Class 0 Recall**: {PIPELINE_LABELS[best_recall_pipeline]} "
            f"achieves {_fmt_pct(best_recall_val)}."
        )

    # Collapse info
    collapse_count = 0
    for key in PIPELINE_KEYS:
        df = data[key]
        if df is not None and df["accuracy"].min() is not None and df["accuracy"].min() < 0.55:
            collapse_count += 1
    lines.append(f"3. **Collapse**: {collapse_count} of 3 pipelines collapse at high cost values.")
    lines.append("")

    # --- Data Sources ---
    lines.append("## 9. Data Sources\n")
    lines.append("| Dataset | Path |")
    lines.append("|---|---|")
    lines.append("| Parent sweep CSV | `results/sweep_cost_0_1/sweep_results.csv` |")
    lines.append(
        "| Playground sweep CSV | `playground/cost_sensitive_loss_classification/results/sweep_cost_ratio/sweep_results.csv` |"
    )
    lines.append("| Hybrid sweep CSV | `results/sweep_cost_0_1_reg/sweep_results.csv` |")
    lines.append(f"| Comparison output | `{output_dir}/` |")
    lines.append("")

    # Write file
    md_path = os.path.join(output_dir, "comparison_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Markdown report: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    graphs_dir = os.path.join(output_dir, "graphs")
    os.makedirs(graphs_dir, exist_ok=True)

    # Parse optional cost values
    cost_values = None
    if args.cost_values is not None:
        cost_values = sorted({float(v.strip()) for v in args.cost_values.split(",")})

    # Load data
    data = load_and_align_data(args.parent, args.playground, args.hybrid, cost_values)
    overlap = data["overlap_values"]

    print(f"\n{'=' * 60}")
    print("SWEEP COMPARISON — 3-Way Analysis")
    print(f"{'=' * 60}")
    for key in PIPELINE_KEYS:
        df = data[key]
        status = f"{df.shape[0]} rows" if df is not None else "MISSING"
        print(f"  {PIPELINE_LABELS[key]:30s} {status}")
    print(f"  Overlapping cost values: {len(overlap)}")
    print(f"  Output directory: {output_dir}")
    print(f"{'=' * 60}\n")

    # Build wide comparison
    wide_df = build_wide_comparison(data, COMPARISON_METRICS)
    csv_path = os.path.join(output_dir, "comparison_wide.csv")
    wide_df.write_csv(csv_path)
    print(f"  Wide CSV: {csv_path}")

    # Stability
    stability_df = compute_stability(data, COMPARISON_METRICS)

    # DuckDB
    export_to_duckdb(wide_df, stability_df, data, output_dir)

    # Graphs
    print("\nGenerating visualizations...")
    for metric, title, y_label, filename in [
        ("class_0_recall", "Black Widow Recall Comparison", "Class 0 Recall", "class_0_recall_comparison"),
        ("accuracy", "Accuracy Comparison", "Accuracy", "accuracy_comparison"),
        ("f1_score", "F1 Score Comparison", "F1 Score", "f1_comparison"),
        ("class_1_recall", "False Widow Recall Comparison", "Class 1 Recall", "class_1_recall_comparison"),
    ]:
        plot_metric_comparison(data, metric, title, y_label, graphs_dir, filename)
    plot_dashboard(data, graphs_dir)
    plot_stability_bar(stability_df, graphs_dir)

    # Markdown report
    print("\nGenerating markdown report...")
    generate_markdown_report(data, wide_df, stability_df, output_dir)

    print(f"\n{'=' * 60}")
    print("COMPARISON COMPLETE")
    print(f"{'=' * 60}")
    print("  comparison_wide.csv")
    print("  comparison_summary.md")
    print("  sweep_comparison.duckdb")
    print(f"  graphs/ ({len(os.listdir(graphs_dir))} files)")
    print(f"  All outputs in: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
